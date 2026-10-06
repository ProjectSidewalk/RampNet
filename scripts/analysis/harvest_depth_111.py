"""Archive Google's native depth payload for the benchmark GSV panoramas (#111).

Project Sidewalk's pano store now carries Google depth for its own cities (paterson,
gainesville, sao_paulo here), and the auto-labeler archived bend and laurens_gsv. The
1,000 ``manual_gold`` panoramas are in neither place. This script fetches the depth
payload for a benchmark split's panoramas from Google's photometa metadata endpoint (the
same one streetlevel's ``find_panorama_by_id(..., download_depth=True)`` uses), stores it
verbatim, and commits a per-pano sha256 manifest so the archive can be verified, compared
against a later re-fetch, or rebuilt while the endpoint still serves it.

Layout:

* ``benchmark/<split>/depth/<pano_id>.json.gz`` -- gitignored archive, one file per
  served pano, in the auto-labeler's shape ``{"pano_id", "depth_b64", "fetched_at"}``
  (plus a ``meta`` dict: heading/pitch/roll and capture month from the same response),
  so the readers that take a labeler depth dir (``recall_by_depth_112.load_payload``)
  read it unchanged. ``gone.txt`` / ``no_depth.txt`` beside them are skip caches.
* ``benchmark/<split>/depth_manifest.json`` -- committed. Status per pano
  (saved | gone | no_depth | error | not_fetched), the sha256 of the base64 payload
  STRING (not of the gzip file, so the hash does not depend on ``fetched_at`` or gzip
  mtime and a re-fetch compares byte for byte), and the ground-plane summary.
* ``benchmark/<split>/depth_labeler_compare.json`` -- committed, written by
  ``compare-labeler``: per-pano payload equality against the labeler's archive, with the
  labeler's own ``camera_height_m`` quoted so the decoder here can be tested against it
  offline.

Usage (``pip install streetlevel==0.12.10`` is needed for ``harvest`` only; nothing else
imports it)::

    python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --limit 20
    python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --resume
    python scripts/analysis/harvest_depth_111.py verify --split manual_gold
    python scripts/analysis/harvest_depth_111.py compare-labeler --split bend \\
        --labeler-root D:/Git/sidewalk-auto-labeler
    python scripts/analysis/harvest_depth_111.py summarize
    python scripts/analysis/harvest_depth_111.py --check      # offline; CI-safe

Pacing follows sidewalk-panorama-tools ``downloaders/gsv.py``: one request at a time,
sleep uniform(interval, 2*interval) between requests, interval starting at 1.0 s with a
0.25 s floor and a 30 s ceiling, x0.8 after 200 consecutive clean requests, doubled on
429/5xx. A 403, a redirect to ``/sorry/`` or
``consent.google.com``, or a body that is not the expected JSON prefix stops the run at
once with zero retries; so do 25 consecutive failures, and a ``no_depth`` share above 5%
after 100 panos (the labeler's poisoning alarm: a high rate means the response is being
misread, not that the panos lack depth). A stopped run still writes its manifest, with an
``aborted_at`` record.

The decoder is a numpy-free stdlib port of the auto-labeler's ``depth.parse`` and
``depth.ground_plane`` (sidewalk-auto-labeler ``depth.py``; the same wire layout is
decoded by sidewalk-panorama-tools ``_decode_depth_planes``): an 8-byte header
``<BHHHB`` (header size, n_planes, width, height, offset as a uint8), then width*height
uint8 plane indices, then n_planes ``<ffff`` (nx, ny, nz, d). The ground plane is the
labeler's rule -- among planes within 18 deg of horizontal with at least 90% of their
pixels below the horizon, the one with the most pixels -- so camera heights match the
labeler's ``index.csv`` exactly. Pano ids can start with ``-``: they are always joined
into paths, never passed as bare CLI tokens.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_NAME = "depth_manifest.json"
COMPARE_NAME = "depth_labeler_compare.json"
ENDPOINT = "photometa/v1 (GeoPhotoService) via streetlevel 0.12.10"

SAVED, GONE, NO_DEPTH, ERROR, NOT_FETCHED = "saved", "gone", "no_depth", "error", "not_fetched"
STATUSES = (SAVED, GONE, NO_DEPTH, ERROR, NOT_FETCHED)

# Pacing (sidewalk-panorama-tools DepthPacer numbers).
PACE_START_S, PACE_FLOOR_S, PACE_CEIL_S = 1.0, 0.25, 30.0
PACE_RECOVER_AFTER = 200         # clean requests at the current interval before speeding up
PACE_RECOVER_FACTOR = 0.8        # ...by this factor (pano-tools DEPTH_PACE_RECOVER_*)
MAX_CONSECUTIVE_FAILURES = 25
TRANSIENT_TRIES = 3              # per pano, for 429/5xx/network errors only
NO_DEPTH_ALARM_AFTER = 100
NO_DEPTH_ALARM_RATE = 0.05

# Decoder constants (sidewalk-auto-labeler depth.py).
HEADER_BYTES = 8
SKY = 0
GROUND_MAX_TILT_DEG = 18.0
GROUND_MIN_BELOW_HORIZON = 0.9
DEGENERATE_MAX_PLANES = 2

# Which benchmark splits are GSV. Everything else is Mapillary / Panoramax, which serve
# no Google depth. HOLDER says who already holds Google depth for a split before this
# script; it is a statement about the world on 2026-10-05, quoted by `summarize`.
HARVEST_SPLITS = ("manual_gold", "bend", "laurens_gsv")
HOLDER = {
    "manual_gold": "none before #111",
    "bend": "labeler runs/bend/depth (unpublished)",
    "laurens_gsv": "labeler runs/laurens_gsv/depth (unpublished, single copy)",
    "paterson": "labeler archive + Project Sidewalk pano store",
    "gainesville": "labeler archive + Project Sidewalk pano store",
    "sao_paulo": "labeler archive + Project Sidewalk pano store",
}
# The 12 scored bundles plus the two unscored GSV-free splits the plan names.
COVERAGE_SPLITS = ("manual_gold", "bend", "laurens_gsv", "paterson", "gainesville",
                   "sao_paulo", "annapolis", "budapest_district5", "clovis", "morgantown",
                   "richmond", "laurens_mapillary", "richmond_neighbourhood", "bayonne")

# manual_gold carries no city key; the three source cities are far apart, so a coarse
# coordinate box is unambiguous (lat_min, lat_max, lon_min, lon_max).
CITY_BOXES = {
    "nyc": (40.4, 41.0, -74.3, -73.6),
    "portland": (45.3, 45.8, -123.0, -122.4),
    "bend": (43.9, 44.2, -121.5, -121.1),
}


class Blocked(RuntimeError):
    """Google refused us (403 / sorry / consent / unparseable body): stop, never retry."""


# ---------------------------------------------------------------------------- decoder

def parse_payload(b64_string):
    """Decode a base64 depth payload into (width, height, planes, indices).

    ``planes`` is a list of (nx, ny, nz, d) tuples indexed by plane id (id 0 is the sky
    sentinel); ``indices`` is width*height bytes of plane ids in raw (image) column order.

    Example:
        >>> raw = struct.pack("<BHHHB", 8, 1, 2, 1, 8) + bytes([0, 0]) + struct.pack("<ffff", 0, 0, -1, 2.5)
        >>> parse_payload(base64.urlsafe_b64encode(raw).decode())[:2]
        (2, 1)
    """
    b64_string = b64_string + "=" * ((4 - len(b64_string) % 4) % 4)
    raw = base64.urlsafe_b64decode(b64_string)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"depth payload truncated: {len(raw)} bytes")
    if raw[0] != HEADER_BYTES:
        raise ValueError(f"unexpected depth header size {raw[0]}, expected {HEADER_BYTES}")
    n_planes, width, height = struct.unpack_from("<HHH", raw, 1)
    offset = raw[7]          # a uint8, not a uint16 (the upstream streetlevel bug)
    body = width * height
    expected = offset + body + 16 * n_planes
    if offset < HEADER_BYTES or len(raw) < expected:
        raise ValueError(f"depth payload truncated: {len(raw)} bytes, need {expected}")
    indices = raw[offset:offset + body]
    base = offset + body
    planes = [struct.unpack_from("<ffff", raw, base + 16 * i) for i in range(n_planes)]
    return width, height, planes, indices


def ground_summary(b64_string):
    """Per-pano summary fields for the manifest, from one payload.

    Returns ``{n_planes, degenerate, camera_height_m, ground_tilt_deg, exactly_level}``;
    the ground fields are None when no plane qualifies. ``exactly_level`` marks Google's
    stand-in ground (normal exactly (0, 0, +-1), almost always at 2.500 m), which is a
    default, not a measurement.
    """
    width, height, planes, indices = parse_payload(b64_string)
    counts = Counter(indices)
    below = Counter(indices[(height // 2) * width:])
    best = None
    for idx, count in counts.items():
        if idx == SKY or idx >= len(planes):
            continue
        nx, ny, nz, d = planes[idx]
        tilt = math.degrees(math.acos(min(1.0, abs(nz))))
        if tilt > GROUND_MAX_TILT_DEG or below[idx] / count < GROUND_MIN_BELOW_HORIZON:
            continue
        # Strict '>' over Counter order (first occurrence in the index array) keeps the
        # first of any tie, which is what the labeler's stable sort by -count does.
        if best is None or count > best[0]:
            best = (count, idx, planes[idx], tilt)
    out = {"n_planes": len(planes), "degenerate": len(planes) <= DEGENERATE_MAX_PLANES,
           "camera_height_m": None, "ground_tilt_deg": None, "exactly_level": None}
    if best is not None:
        _, _, (nx, ny, nz, d), tilt = best
        out.update(camera_height_m=round(d, 4), ground_tilt_deg=round(tilt, 4),
                   exactly_level=(nx == 0.0 and ny == 0.0))
    return out


def payload_sha256(b64_string):
    return hashlib.sha256(b64_string.encode("ascii")).hexdigest()


# --------------------------------------------------------------------- response walk

def classify_response(response):
    """(status, depth_b64 or None, meta) from a raw by-id photometa response.

    Status mapping (the labeler's): response code ``resp[1][0][0][0]`` in (1, 3) is a
    served pano, anything else is gone; a served pano without a payload at
    ``resp[1][0][5][0][5][1][2]`` is no_depth. An unrecognized shape is an error, so it
    is never cached.

    Example:
        >>> classify_response([None, [[[2]]]])[0]
        'gone'
    """
    try:
        code = response[1][0][0][0]
    except (IndexError, KeyError, TypeError):
        return ERROR, None, {"reason": "unrecognized response shape"}
    if code not in (1, 3):
        return GONE, None, {"code": code}
    msg = response[1][0]
    meta = {}
    try:
        hpr = msg[5][0][1][2]
        meta["heading_deg"] = round(float(hpr[0]), 4)
        meta["pitch_deg"] = round(90.0 - float(hpr[1]), 4)     # streetlevel's convention
        meta["roll_deg"] = round(float(hpr[2]), 4)
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    try:
        date = msg[6][7]
        meta["capture_ym"] = f"{int(date[0]):04d}-{int(date[1]):02d}"
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    try:
        blob = msg[5][0][5][1][2]
    except (IndexError, KeyError, TypeError):
        blob = None
    if not blob or not isinstance(blob, str):
        return NO_DEPTH, None, meta
    return SAVED, blob, meta


# ----------------------------------------------------------------------- pano lists

def split_dir(split, repo=REPO):
    return os.path.join(repo, "benchmark", split)


def depth_dir(split, repo=REPO):
    return os.path.join(split_dir(split, repo), "depth")


def archive_path(split, pid, repo=REPO):
    """Path join, never a glob or a bare CLI token: ids can start with '-'."""
    return os.path.join(depth_dir(split, repo), pid + ".json.gz")


def read_records(split, repo=REPO):
    out = []
    with open(os.path.join(split_dir(split, repo), "records.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line)["pano"])
    return out


def split_pano_ids(split, repo=REPO):
    """Pano ids in records.jsonl order, asserted equal to imagery_manifest.json's set."""
    ids = list(dict.fromkeys(p["panorama_id"] for p in read_records(split, repo)))
    im_path = os.path.join(split_dir(split, repo), "imagery_manifest.json")
    if os.path.exists(im_path):
        with open(im_path, encoding="utf-8") as fh:
            im = set(json.load(fh)["panos"])
        if im != set(ids):
            raise SystemExit(f"{split}: records.jsonl and imagery_manifest.json disagree "
                             f"({len(set(ids) - im)} only in records, {len(im - set(ids))} "
                             f"only in the manifest)")
    return ids


def _load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def _write_ids(path, ids):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(i + "\n" for i in sorted(ids)))


def read_archive_file(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_archive_file(path, pid, blob, meta, fetched_at):
    part = path + ".part"
    with gzip.open(part, "wt", encoding="utf-8") as fh:
        json.dump({"pano_id": pid, "depth_b64": blob, "fetched_at": fetched_at,
                   "meta": meta}, fh)
    os.replace(part, path)


# ------------------------------------------------------------------------- manifest

def digest_of(panos):
    """One sha256 (16 hex) over sorted ``pid|sha256`` (``pid|<status>`` when unsaved)."""
    spec = ";".join(f"{pid}|{e.get('sha256') or e['status']}" for pid, e in sorted(panos.items()))
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def counts_of(panos):
    c = Counter(e["status"] for e in panos.values())
    return {f"n_{s}": c.get(s, 0) for s in STATUSES}


def entry_from_archive(path):
    """A manifest entry rebuilt from one archive file (all derived fields recomputed)."""
    rec = read_archive_file(path)
    blob = rec["depth_b64"]
    entry = {"status": SAVED, "sha256": payload_sha256(blob), "bytes": len(blob),
             "fetched_at": rec.get("fetched_at")}
    entry.update(ground_summary(blob))
    entry.update({k: v for k, v in (rec.get("meta") or {}).items()
                  if k in ("heading_deg", "pitch_deg", "roll_deg", "capture_ym")})
    return entry


def build_manifest(split, ids, errors=None, aborted_at=None, repo=REPO, prior=None):
    """Manifest for every id from the local archive + skip caches + this run's errors.

    A pano with no archive file, no cache entry and no error this run keeps its prior
    manifest entry's status if that was gone/no_depth (the caches are local, the manifest
    is committed), else it is ``not_fetched``.
    """
    errors = errors or {}
    ddir = depth_dir(split, repo)
    gone = _load_ids(os.path.join(ddir, "gone.txt"))
    no_depth = _load_ids(os.path.join(ddir, "no_depth.txt"))
    prior_panos = (prior or {}).get("panos", {})
    panos = {}
    for pid in ids:
        path = archive_path(split, pid, repo)
        if os.path.exists(path):
            panos[pid] = entry_from_archive(path)
        elif pid in gone:
            panos[pid] = {"status": GONE}
        elif pid in no_depth:
            panos[pid] = {"status": NO_DEPTH}
        elif pid in errors:
            panos[pid] = {"status": ERROR, "error": errors[pid]}
        elif prior_panos.get(pid, {}).get("status") in (GONE, NO_DEPTH):
            panos[pid] = {"status": prior_panos[pid]["status"]}
        else:
            panos[pid] = {"status": NOT_FETCHED}
    fetched = sorted(e["fetched_at"] for e in panos.values() if e.get("fetched_at"))
    man = {"split": split, "endpoint": ENDPOINT, "n_requested": len(ids),
           "fetched_at_utc": {"first": fetched[0] if fetched else None,
                              "last": fetched[-1] if fetched else None},
           "sha256_of": "the base64 depth_b64 string, ASCII-encoded (not the .json.gz)",
           "panos": panos}
    man.update(counts_of(panos))
    man["digest"] = digest_of(panos)
    if aborted_at:
        man["aborted_at"] = aborted_at
    elif prior and prior.get("aborted_at") and man["n_not_fetched"]:
        man["aborted_at"] = prior["aborted_at"]
    return man


def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def dump_json(path, obj):
    """LF, indent=1, sorted keys, floats at 4 dp -- byte-stable across builds."""
    text = json.dumps(_round_floats(obj), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def load_manifest(split, repo=REPO):
    path = os.path.join(split_dir(split, repo), MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------- fetch

class Pacer:
    """Serial pacing: sleep uniform(interval, 2*interval) before each request."""

    def __init__(self, start=PACE_START_S, floor=PACE_FLOOR_S, ceil=PACE_CEIL_S, sleep=time.sleep):
        self.interval, self.floor, self.ceil, self._sleep = start, floor, ceil, sleep
        self.slept = 0.0
        self.clean = 0

    def wait(self):
        s = random.uniform(self.interval, 2 * self.interval)
        self.slept += s
        self._sleep(s)

    def ok(self):
        self.clean += 1
        if self.clean >= PACE_RECOVER_AFTER:
            self.clean = 0
            self.interval = max(self.floor, self.interval * PACE_RECOVER_FACTOR)

    def push_back(self):
        self.clean = 0
        self.interval = min(self.ceil, max(self.interval * 2, PACE_START_S))


def is_block(status_code, url, text):
    """Does this HTTP response mean Google is refusing us? Then stop, zero retries."""
    u = (url or "").lower()
    # The body is only inspected when it is not the expected JSON: a real payload can
    # carry arbitrary place names.
    t = "" if (text or "").startswith(")]}'") else (text or "")[:2000].lower()
    return (status_code == 403 or "/sorry/" in u or "consent.google.com" in u
            or "/sorry/" in t or "consent.google.com" in t
            or "unusual traffic" in t)


def fetch_one(session, pid, pacer, log):
    """One pano: (status, blob, meta). Raises Blocked. Retries only 429/5xx/network."""
    from streetlevel.streetview import api   # lazy: nothing but `harvest` needs streetlevel
    import requests

    url = api.build_find_panorama_by_id_request_url(pid, True, "en")
    headers = {"Accept": "*/*", "Host": "www.google.com", "Referer": "https://www.google.com/",
               "Alt-Used": "www.google.com",
               # streetlevel 0.12.10's own UA, kept as-is (pano-tools does the same).
               "User-Agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:151.0) "
                             "Gecko/20100101 Firefox/151.0"}
    last = None
    for attempt in range(TRANSIENT_TRIES):
        pacer.wait()
        log["requests"] += 1
        try:
            r = session.get(url, headers=headers, timeout=30)
        except requests.RequestException as e:
            last = f"network: {type(e).__name__}"
            pacer.push_back()
            continue
        if is_block(r.status_code, r.url, r.text):
            raise Blocked(f"HTTP {r.status_code} at {r.url[:120]}")
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            log["push_backs"] += 1
            pacer.push_back()
            continue
        if r.status_code != 200:
            return ERROR, None, {"reason": f"HTTP {r.status_code}"}
        text = r.text
        if not text.startswith(")]}'"):
            raise Blocked(f"unexpected body (no )]}}' prefix): {text[:80]!r}")
        try:
            response = json.loads(text[4:])
        except ValueError:
            raise Blocked(f"unparseable JSON body: {text[:80]!r}")
        pacer.ok()
        return classify_response(response)
    return ERROR, None, {"reason": last}


def harvest(split, limit=None, resume=True, repo=REPO, session=None):
    """Fetch every pano of a split not already resolved. Returns (manifest, run log)."""
    import requests

    ids = split_pano_ids(split, repo)
    ddir = depth_dir(split, repo)
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(ddir):                       # orphans of a hard kill
        if f.endswith(".part"):
            os.remove(os.path.join(ddir, f))
    prior = load_manifest(split, repo)
    gone_path, nd_path = os.path.join(ddir, "gone.txt"), os.path.join(ddir, "no_depth.txt")
    gone, no_depth = _load_ids(gone_path), _load_ids(nd_path)
    if not resume and (gone or no_depth or any(f.endswith(".json.gz") for f in os.listdir(ddir))):
        raise SystemExit(f"{ddir} already holds results; pass --resume to continue it")
    todo = [p for p in ids if p not in gone and p not in no_depth
            and not os.path.exists(archive_path(split, p, repo))]
    print(f"{split}: {len(ids)} panos, {len(ids) - len(todo)} already resolved, {len(todo)} to fetch")
    if limit is not None:
        todo = todo[:limit]
        print(f"  --limit: fetching {len(todo)}")

    session = session or requests.Session()
    pacer = Pacer()
    log = {"requests": 0, "push_backs": 0, "attempted": 0, "outcomes": Counter()}
    errors, aborted, consecutive = {}, None, 0
    t0 = time.time()
    try:
        for i, pid in enumerate(todo):
            try:
                status, blob, meta = fetch_one(session, pid, pacer, log)
            except Blocked as e:
                aborted = {"pano_id": pid, "index": i, "reason": f"blocked: {e}"}
                print(f"  STOP: {aborted['reason']} at pano {i} ({pid}); zero retries")
                break
            log["attempted"] += 1
            log["outcomes"][status] += 1
            if status == SAVED:
                parse_payload(blob)                  # reject a corrupt payload now
                write_archive_file(archive_path(split, pid, repo), pid, blob, meta,
                                   datetime.now(timezone.utc).isoformat(timespec="seconds"))
                consecutive = 0
            elif status == GONE:
                gone.add(pid)
                _write_ids(gone_path, gone)
                consecutive = 0
            elif status == NO_DEPTH:
                no_depth.add(pid)
                _write_ids(nd_path, no_depth)
                consecutive = 0
            else:
                errors[pid] = meta.get("reason", "error")
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    aborted = {"pano_id": pid, "index": i,
                               "reason": f"{consecutive} consecutive failures (last: {errors[pid]})"}
                    print(f"  STOP: {aborted['reason']}")
                    break
            n_nd = log["outcomes"][NO_DEPTH]
            if (log["attempted"] >= NO_DEPTH_ALARM_AFTER
                    and n_nd / log["attempted"] > NO_DEPTH_ALARM_RATE):
                aborted = {"pano_id": pid, "index": i,
                           "reason": f"no_depth alarm: {n_nd} of {log['attempted']} "
                                     f"(> {NO_DEPTH_ALARM_RATE:.0%}) -- response likely misread"}
                print(f"  STOP: {aborted['reason']}")
                break
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)}  {dict(log['outcomes'])}  "
                      f"interval {pacer.interval:.2f}s  {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        aborted = {"pano_id": None, "index": None, "reason": "interrupted (Ctrl-C)"}
    finally:
        log["elapsed_s"] = round(time.time() - t0, 1)
        log["slept_s"] = round(pacer.slept, 1)
        man = build_manifest(split, ids, errors, aborted, repo, prior)
        dump_json(os.path.join(split_dir(split, repo), MANIFEST_NAME), man)
    log["outcomes"] = dict(log["outcomes"])
    print(f"  done: {log}")
    print(f"  manifest: saved {man['n_saved']}, gone {man['n_gone']}, no_depth "
          f"{man['n_no_depth']}, error {man['n_error']}, not_fetched {man['n_not_fetched']}, "
          f"digest {man['digest']}")
    return man, log, aborted


# --------------------------------------------------------------------------- verify

def verify(split, repo=REPO):
    """(problems, note). Re-hashes local files against the committed manifest.

    With no archive directory (a clean clone), only the manifest's self-consistency is
    checked and the note says so; that is not a failure.
    """
    man = load_manifest(split, repo)
    if man is None:
        return [f"{split}: no {MANIFEST_NAME}"], ""
    problems = []
    ids = split_pano_ids(split, repo)
    if set(man["panos"]) != set(ids):
        problems.append(f"{split}: manifest pano set != records/imagery_manifest set")
    if digest_of(man["panos"]) != man.get("digest"):
        problems.append(f"{split}: digest mismatch (recomputed {digest_of(man['panos'])})")
    for k, v in counts_of(man["panos"]).items():
        if man.get(k) != v:
            problems.append(f"{split}: {k} says {man.get(k)}, entries say {v}")
    if man.get("n_requested") != len(ids):
        problems.append(f"{split}: n_requested {man.get('n_requested')} != {len(ids)}")
    ddir = depth_dir(split, repo)
    has_archive = os.path.isdir(ddir) and any(f.endswith(".json.gz") for f in os.listdir(ddir))
    if not has_archive:
        return problems, "archive absent, manifest self-consistent" if not problems else ""
    for pid, e in sorted(man["panos"].items()):
        path = archive_path(split, pid, repo)
        if e["status"] != SAVED:
            if os.path.exists(path):
                problems.append(f"{split}/{pid}: archived but manifest says {e['status']}")
            continue
        if not os.path.exists(path):
            problems.append(f"{split}/{pid}: missing from the archive")
            continue
        try:
            blob = read_archive_file(path)["depth_b64"]
        except Exception as ex:  # noqa: BLE001 -- reported, never fatal
            problems.append(f"{split}/{pid}: unreadable ({ex})")
            continue
        if payload_sha256(blob) != e["sha256"]:
            problems.append(f"{split}/{pid}: sha256 drift")
    return problems, "archive verified"


# -------------------------------------------------------------------------- compare

def compare_labeler(split, labeler_root, repo=REPO):
    """Payload equality of our archive against the labeler's runs/<split>/depth."""
    import csv
    ldir = os.path.join(labeler_root, "runs", split, "depth")
    if not os.path.isdir(ldir):
        raise SystemExit(f"no labeler archive at {ldir}; compare-labeler skipped")
    index = {}
    ipath = os.path.join(ldir, "index.csv")
    if os.path.exists(ipath):
        with open(ipath, newline="", encoding="utf-8") as fh:
            index = {r["panorama_id"]: r for r in csv.DictReader(fh)}
    man = load_manifest(split, repo)
    if man is None:
        raise SystemExit(f"{split}: no manifest; harvest first")
    panos, tally = {}, Counter()
    for pid, e in sorted(man["panos"].items()):
        lpath = os.path.join(ldir, pid + ".json.gz")
        rec = {"ours": e["status"]}
        if os.path.exists(lpath):
            lrec = read_archive_file(lpath)
            rec["labeler_sha256"] = payload_sha256(lrec["depth_b64"])
            rec["labeler_fetched_at"] = lrec.get("fetched_at")
            row = index.get(pid)
            if row and row.get("camera_height_m"):
                rec["labeler_camera_height_m"] = float(row["camera_height_m"])
            if e["status"] == SAVED:
                rec["identical"] = rec["labeler_sha256"] == e["sha256"]
                tally["identical" if rec["identical"] else "differs"] += 1
            else:
                tally[f"labeler_only (ours {e['status']})"] += 1
        else:
            tally["ours_only" if e["status"] == SAVED else f"neither ({e['status']})"] += 1
        panos[pid] = rec
    out = {"split": split, "labeler_dir": f"runs/{split}/depth",
           "labeler_commit_note": "labeler archive is local and unpublished; see docs/benchmark_depth_111.md",
           "tally": dict(sorted(tally.items())), "panos": panos}
    dump_json(os.path.join(split_dir(split, repo), COMPARE_NAME), out)
    print(f"{split}: {dict(sorted(tally.items()))}")
    return out


# ------------------------------------------------------------------------ summarize

def city_of(pano):
    lat, lon = pano.get("pano_coord") or (pano.get("lat"), pano.get("lng"))
    for name, (a, b, c, d) in CITY_BOXES.items():
        if lat is not None and a <= lat <= b and c <= lon <= d:
            return name
    return "other"


def _pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def summarize(repo=REPO, out=print):
    out("## Coverage (all splits)")
    out("| split | panos | source | Google depth held by | this archive: saved / gone / no_depth / error / not_fetched |")
    out("|---|---:|---|---|---|")
    for split in COVERAGE_SPLITS:
        if not os.path.exists(os.path.join(split_dir(split, repo), "records.jsonl")):
            continue
        recs = read_records(split, repo)
        sources = sorted({str(p.get("source")) for p in recs})
        src = ", ".join("(none; GSV)" if s == "None" else s for s in sources)
        gsv = all(s in ("None", "launch") for s in sources)
        holder = HOLDER.get(split, "no Google depth (not GSV imagery)" if not gsv else "?")
        man = load_manifest(split, repo)
        cell = (f"{man['n_saved']} / {man['n_gone']} / {man['n_no_depth']} / {man['n_error']} / "
                f"{man['n_not_fetched']}") if man else "-"
        out(f"| {split} | {len(recs)} | {src} | {holder} | {cell} |")

    for split in HARVEST_SPLITS:
        man = load_manifest(split, repo)
        if man is None:
            continue
        recs = {p["panorama_id"]: p for p in read_records(split, repo)}
        out(f"\n## {split}: availability")
        if split == "manual_gold":
            out("| source city (coordinate box) | panos | saved | gone | no_depth | other |")
            out("|---|---:|---:|---:|---:|---:|")
            by = {}
            for pid, e in man["panos"].items():
                by.setdefault(city_of(recs[pid]), Counter())[e["status"]] += 1
            for city in sorted(by):
                c = by[city]
                out(f"| {city} | {sum(c.values())} | {c[SAVED]} | {c[GONE]} | {c[NO_DEPTH]} | "
                    f"{c[ERROR] + c[NOT_FETCHED]} |")
        years = Counter()
        for pid, e in man["panos"].items():
            if e["status"] == SAVED and e.get("capture_ym"):
                years[e["capture_ym"][:4]] += 1
        if years:
            out("capture year of saved panos (from the response): "
                + ", ".join(f"{y} {n}" for y, n in sorted(years.items())))
        saved = [e for e in man["panos"].values() if e["status"] == SAVED]
        if not saved:
            continue
        degenerate = sum(e["degenerate"] for e in saved)
        no_ground = sum(e["camera_height_m"] is None for e in saved)
        level = sum(bool(e.get("exactly_level")) for e in saved)
        level_25 = sum(bool(e.get("exactly_level")) and abs(e["camera_height_m"] - 2.5) < 1e-4
                       for e in saved)
        measured = sorted(e["camera_height_m"] for e in saved
                          if e["camera_height_m"] is not None and not e["degenerate"]
                          and not e.get("exactly_level"))
        tilts = sorted(e["ground_tilt_deg"] for e in saved
                       if e["camera_height_m"] is not None and not e["degenerate"]
                       and not e.get("exactly_level"))
        out(f"saved {len(saved)}: degenerate (<= {DEGENERATE_MAX_PLANES} planes) {degenerate}; "
            f"no ground plane {no_ground}; exactly-level stand-in ground {level} "
            f"({level / len(saved):.1%} of saved; {level_25} of them at 2.500 m)")
        if measured:
            out(f"camera height, measured ground (n={len(measured)}): min {measured[0]:.3f}  "
                f"p10 {_pct(measured, .1):.3f}  median {_pct(measured, .5):.3f}  "
                f"p90 {_pct(measured, .9):.3f}  max {measured[-1]:.3f} m")
            out(f"ground tilt, measured ground: median {_pct(tilts, .5):.2f} deg, "
                f"p90 {_pct(tilts, .9):.2f} deg")
        cmp_path = os.path.join(split_dir(split, repo), COMPARE_NAME)
        if os.path.exists(cmp_path):
            with open(cmp_path, encoding="utf-8") as fh:
                out(f"vs the labeler archive: {json.load(fh)['tally']}")


# ---------------------------------------------------------------------------- check

def check_heights_against_labeler(split, repo=REPO, tol=1e-3):
    """Problems where an identical payload's camera height differs from the labeler's."""
    path = os.path.join(split_dir(split, repo), COMPARE_NAME)
    man = load_manifest(split, repo)
    if not os.path.exists(path) or man is None:
        return []
    with open(path, encoding="utf-8") as fh:
        cmp = json.load(fh)
    problems = []
    for pid, rec in cmp["panos"].items():
        if not rec.get("identical") or "labeler_camera_height_m" not in rec:
            continue
        ours = man["panos"][pid].get("camera_height_m")
        if ours is None or abs(ours - rec["labeler_camera_height_m"]) > tol:
            problems.append(f"{split}/{pid}: camera height {ours} vs labeler "
                            f"{rec['labeler_camera_height_m']}")
    return problems


def run_check(repo=REPO):
    failed = False
    for split in COVERAGE_SPLITS:
        if load_manifest(split, repo) is None:
            continue
        problems, note = verify(split, repo)
        problems += check_heights_against_labeler(split, repo)
        if problems:
            failed = True
            print(f"{split}: FAIL")
            for p in problems[:20]:
                print(f"  {p}")
        else:
            print(f"{split}: ok ({note})")
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="offline: verify every committed manifest (and the archive if present)")
    sub = ap.add_subparsers(dest="cmd")
    h = sub.add_parser("harvest")
    h.add_argument("--split", required=True, choices=HARVEST_SPLITS)
    h.add_argument("--limit", type=int)
    h.add_argument("--resume", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--split", required=True)
    c = sub.add_parser("compare-labeler")
    c.add_argument("--split", required=True, choices=("bend", "laurens_gsv"))
    c.add_argument("--labeler-root", required=True)
    sub.add_parser("summarize")
    args = ap.parse_args(argv)

    if args.check:
        return run_check()
    if args.cmd == "harvest":
        if args.limit is not None and args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        _, _, aborted = harvest(args.split, args.limit, args.resume)
        return 2 if aborted else 0
    if args.cmd == "verify":
        problems, note = verify(args.split)
        for p in problems:
            print(p)
        print(f"{args.split}: {'FAIL' if problems else 'ok'} ({note})")
        return 1 if problems else 0
    if args.cmd == "compare-labeler":
        compare_labeler(args.split, args.labeler_root)
        return 0
    if args.cmd == "summarize":
        summarize()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
