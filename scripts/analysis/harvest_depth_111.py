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
  (saved | gone | no_depth | error | not_fetched; gone carries the response ``code`` when
  recorded), two hashes per saved payload -- ``sha256`` of the base64 payload STRING
  (independent of ``fetched_at`` and gzip metadata, so a re-fetch compares byte for byte)
  and ``file_sha256`` of the .json.gz as stored (what #112's reader pins) -- and the
  ground-plane summary. ``harvest`` never overwrites a resolved pano in this record: a
  rebuild that would change one is written to ``depth_manifest.refetch-<stamp>.json``.
* ``benchmark/<split>/depth_labeler_compare.json`` -- committed, written by
  ``compare-labeler``: per-pano payload equality against the labeler's archive, with the
  labeler's own ``camera_height_m`` quoted so the decoder here can be tested against it
  offline.

Usage (``pip install streetlevel==0.12.10`` is needed for ``harvest`` only; nothing else
imports it)::

    python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --limit 20
    python scripts/analysis/harvest_depth_111.py harvest --split manual_gold --resume
    python scripts/analysis/harvest_depth_111.py verify --split manual_gold
    python scripts/analysis/harvest_depth_111.py verify --split manual_gold --archive-dir <copy>
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
PLAUSIBLE_HEIGHT_M = (0.8, 3.5)   # labeler depth.PLAUSIBLE_HEIGHT_M; reported, not filtered

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


def depth_dir(split, repo=REPO, archive_dir=None):
    """The archive directory: ``benchmark/<split>/depth`` unless ``archive_dir`` is given
    (a re-fetch kept apart from the record, or a downloaded copy to check)."""
    return archive_dir or os.path.join(split_dir(split, repo), "depth")


def archive_path(split, pid, repo=REPO, archive_dir=None):
    """Path join, never a glob or a bare CLI token: ids can start with '-'."""
    return os.path.join(depth_dir(split, repo, archive_dir), pid + ".json.gz")


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


def _load_codes(path):
    """{pano_id: response code or None} from a skip cache.

    One id per line, optionally followed by a tab and the response code. The codes were
    added after the 2026-10-06 harvest, whose gone.txt files hold bare ids.
    """
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.strip().split("\t")
            if parts[0]:
                code = parts[1] if len(parts) > 1 else ""
                out[parts[0]] = int(code) if code.lstrip("-").isdigit() else None
    return out


def _load_ids(path):
    return set(_load_codes(path))


def _write_ids(path, ids):
    """Write a skip cache from a set of ids or a {id: code} dict (code may be None)."""
    codes = ids if isinstance(ids, dict) else dict.fromkeys(ids)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(pid + ("" if codes[pid] is None else f"\t{codes[pid]}") + "\n"
                         for pid in sorted(codes)))


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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_from_archive(path):
    """A manifest entry rebuilt from one archive file (all derived fields recomputed).

    ``sha256`` is of the base64 payload string (independent of fetch time: the key for
    comparing re-fetches); ``file_sha256`` / ``file_bytes`` are of the .json.gz as stored,
    which is what ``recall_by_depth_112.load_payload`` pins and what ``sha256sum`` or an
    LFS hash of an uploaded copy gives.
    """
    rec = read_archive_file(path)
    blob = rec["depth_b64"]
    entry = {"status": SAVED, "sha256": payload_sha256(blob), "payload_b64_chars": len(blob),
             "file_sha256": file_sha256(path), "file_bytes": os.path.getsize(path),
             "fetched_at": rec.get("fetched_at")}
    entry.update(ground_summary(blob))
    entry.update({k: v for k, v in (rec.get("meta") or {}).items()
                  if k in ("heading_deg", "pitch_deg", "roll_deg", "capture_ym")})
    return entry


def build_manifest(split, ids, errors=None, aborted_at=None, repo=REPO, prior=None,
                   archive_dir=None):
    """Manifest for every id from the archive + skip caches + this run's errors.

    A pano with no archive file, no cache entry and no error this run keeps its prior
    manifest entry if that was gone / no_depth / error (the caches are local, the manifest
    is committed), else it is ``not_fetched``. A gone entry carries the response ``code``
    when it was recorded (harvests after 2026-10-06).
    """
    errors = errors or {}
    ddir = depth_dir(split, repo, archive_dir)
    gone = _load_codes(os.path.join(ddir, "gone.txt"))
    no_depth = _load_ids(os.path.join(ddir, "no_depth.txt"))
    prior_panos = (prior or {}).get("panos", {})
    panos = {}
    for pid in ids:
        path = archive_path(split, pid, repo, archive_dir)
        if os.path.exists(path):
            panos[pid] = entry_from_archive(path)
        elif pid in gone:
            panos[pid] = {"status": GONE}
            if gone[pid] is not None:
                panos[pid]["code"] = gone[pid]
        elif pid in no_depth:
            panos[pid] = {"status": NO_DEPTH}
        elif pid in errors:
            panos[pid] = {"status": ERROR, "error": errors[pid]}
        elif prior_panos.get(pid, {}).get("status") in (GONE, NO_DEPTH, ERROR):
            panos[pid] = dict(prior_panos[pid])
        else:
            panos[pid] = {"status": NOT_FETCHED}
    fetched = sorted(e["fetched_at"] for e in panos.values() if e.get("fetched_at"))
    man = {"split": split, "endpoint": ENDPOINT, "n_requested": len(ids),
           "fetched_at_utc": {"first": fetched[0] if fetched else None,
                              "last": fetched[-1] if fetched else None},
           "sha256_of": "the base64 depth_b64 string, ASCII-encoded (not the .json.gz)",
           "file_sha256_of": "the .json.gz file as stored (what recall_by_depth_112 pins)",
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


def load_manifest(split, repo=REPO, path=None):
    path = path or os.path.join(split_dir(split, repo), MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def conflicts_with_record(prior, man):
    """Pano ids whose resolved entry in ``prior`` (saved / gone / no_depth) ``man`` would
    change in ANY field, or lose.

    The whole entry, not just status and payload hash: an identical payload re-fetched into
    a clean clone still has a new ``fetched_at`` and so a new ``file_sha256`` -- the hash
    #112-style analyses pin -- and a gone entry may gain a ``code``. A harvest never writes
    any such change into the committed record; it goes to a separate manifest instead. A
    resumed run over the same archive rebuilds identical entries and so conflicts with
    nothing."""
    out = []
    for pid, e in (prior or {}).get("panos", {}).items():
        if e["status"] not in (SAVED, GONE, NO_DEPTH):
            continue
        if _round_floats(man["panos"].get(pid, {})) != e:
            out.append(pid)
    return sorted(out)


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


STREETLEVEL_HEADERS = {
    "Accept": "*/*", "Host": "www.google.com", "Referer": "https://www.google.com/",
    "Alt-Used": "www.google.com",
    # streetlevel 0.12.10's own UA, kept as-is (pano-tools does the same).
    "User-Agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0",
}


def streetlevel_url(pid):
    """The by-id photometa URL with depth requested, built by streetlevel 0.12.10."""
    from streetlevel.streetview import api   # lazy: nothing but `harvest` needs streetlevel
    return api.build_find_panorama_by_id_request_url(pid, True, "en")


def fetch_one(session, pid, pacer, log, url_builder=streetlevel_url):
    """One pano: (status, blob, meta). Raises Blocked. Retries only 429/5xx/network.

    ``session`` needs only ``get(url, headers=, timeout=)`` returning an object with
    ``status_code``, ``url`` and ``text``; ``url_builder`` maps a pano id to the URL. Both
    are injectable so the stop paths are tested offline.
    """
    url = url_builder(pid)
    last = None
    for _ in range(TRANSIENT_TRIES):
        pacer.wait()
        log["requests"] += 1
        try:
            r = session.get(url, headers=STREETLEVEL_HEADERS, timeout=30)
        except Exception as e:  # noqa: BLE001 -- requests' errors, or anything the transport raises
            last = f"network: {type(e).__name__}"
            log["push_backs"] += 1
            pacer.push_back()
            continue
        if is_block(r.status_code, r.url, r.text):
            raise Blocked(f"HTTP {r.status_code} at {str(r.url)[:120]}")
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


def harvest(split, limit=None, resume=True, repo=REPO, session=None, archive_dir=None,
            manifest_out=None, url_builder=streetlevel_url, pacer=None):
    """Fetch every pano of a split not already resolved. Returns (manifest, log, aborted).

    The manifest goes to ``manifest_out`` if given. Otherwise it goes to the committed
    ``benchmark/<split>/depth_manifest.json`` only when that would not change or drop any
    pano the committed record resolved (``conflicts_with_record``): a partial rebuild in a
    clean clone, or a re-fetch that Google has since revised, is written beside it as
    ``depth_manifest.refetch-<UTC stamp>.json`` and the record is left untouched. Compare the
    two with ``verify --archive-dir`` (re-hashes a re-fetched archive against the record).
    """
    ids = split_pano_ids(split, repo)
    ddir = depth_dir(split, repo, archive_dir)
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(ddir):                       # orphans of a hard kill
        if f.endswith(".part"):
            os.remove(os.path.join(ddir, f))
    prior = load_manifest(split, repo)
    gone_path, nd_path = os.path.join(ddir, "gone.txt"), os.path.join(ddir, "no_depth.txt")
    gone, no_depth = _load_codes(gone_path), _load_ids(nd_path)
    if not resume and (gone or no_depth or any(f.endswith(".json.gz") for f in os.listdir(ddir))):
        raise SystemExit(f"{ddir} already holds results; pass --resume to continue it")
    todo = [p for p in ids if p not in gone and p not in no_depth
            and not os.path.exists(archive_path(split, p, repo, archive_dir))]
    print(f"{split}: {len(ids)} panos, {len(ids) - len(todo)} already resolved, {len(todo)} to fetch")
    if limit is not None:
        todo = todo[:limit]
        print(f"  --limit: fetching {len(todo)}")

    if session is None:
        import requests
        session = requests.Session()
    pacer = pacer or Pacer()
    log = {"requests": 0, "push_backs": 0, "attempted": 0, "outcomes": Counter()}
    errors, aborted, consecutive = {}, None, 0
    t0 = time.time()
    try:
        for i, pid in enumerate(todo):
            try:
                status, blob, meta = fetch_one(session, pid, pacer, log, url_builder)
            except Blocked as e:
                aborted = {"pano_id": pid, "index": i, "reason": f"blocked: {e}"}
                print(f"  STOP: {aborted['reason']} at pano {i} ({pid}); zero retries")
                break
            if status == SAVED:
                try:
                    parse_payload(blob)              # reject a corrupt payload now
                except (ValueError, struct.error) as e:
                    status, meta = ERROR, {"reason": f"corrupt payload: {e}"}
            log["attempted"] += 1
            log["outcomes"][status] += 1
            if status == SAVED:
                write_archive_file(archive_path(split, pid, repo, archive_dir), pid, blob, meta,
                                   datetime.now(timezone.utc).isoformat(timespec="seconds"))
                consecutive = 0
            elif status == GONE:
                gone[pid] = meta.get("code")
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
        man = build_manifest(split, ids, errors, aborted, repo, prior, archive_dir)
        out_path = manifest_out
        if out_path is None:
            out_path = os.path.join(split_dir(split, repo), MANIFEST_NAME)
            conflicts = conflicts_with_record(prior, man) if prior is not None else []
            if conflicts:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                out_path = os.path.join(split_dir(split, repo), f"depth_manifest.refetch-{stamp}.json")
                rec = prior["panos"]
                payload = [c for c in conflicts
                           if man["panos"].get(c, {}).get("status") != rec[c]["status"]
                           or man["panos"].get(c, {}).get("sha256") != rec[c].get("sha256")
                           or any(man["panos"].get(c, {}).get(k) != rec[c].get(k)
                                  for k in PAYLOAD_FIELDS)]
                print(f"  {len(conflicts)} pano(s) would change their committed entry (status, "
                      f"payload or file/fetch fields): {len(payload)} status/payload/payload-field change(s), "
                      f"{len(conflicts) - len(payload)} file/fetch-field-only change(s) "
                      f"(e.g. {conflicts[:3]}); the committed {MANIFEST_NAME} is left untouched "
                      f"and this run's manifest goes to {os.path.basename(out_path)}")
        dump_json(out_path, man)
        log["manifest_path"] = out_path
    log["outcomes"] = dict(log["outcomes"])
    print(f"  done: {log}")
    print(f"  manifest {out_path}: saved {man['n_saved']}, gone {man['n_gone']}, no_depth "
          f"{man['n_no_depth']}, error {man['n_error']}, not_fetched {man['n_not_fetched']}, "
          f"digest {man['digest']}")
    return man, log, aborted


# --------------------------------------------------------------------------- verify

# Fields decoded from the payload itself: with an identical payload sha256 these can only
# differ if the decoder changed or the manifest was edited -- always a failure.
PAYLOAD_FIELDS = ("payload_b64_chars", "n_planes", "degenerate", "camera_height_m",
                  "ground_tilt_deg", "exactly_level")
# Fields of this particular copy and fetch (the .json.gz bytes, when it was fetched, and the
# response metadata stored beside the payload). A re-fetched or re-written copy of an
# unchanged payload differs here and nowhere else.
PROVENANCE_FIELDS = ("file_sha256", "file_bytes", "fetched_at", "heading_deg", "pitch_deg",
                     "roll_deg", "capture_ym")


def verify(split, repo=REPO, archive_dir=None, payload_only=False):
    """(problems, note). Checks an archive against the committed manifest.

    Always: the manifest's pano set, counts, n_requested and digest. With an archive present
    (``benchmark/<split>/depth`` or ``archive_dir``): every saved pano's file is re-read and
    its whole manifest entry rebuilt by ``entry_from_archive``, and any archived file the
    manifest does not list as saved is flagged. Three labels, in this order per pano:

    * ``payload sha256 drift`` -- the payload itself differs (Google revised it);
    * ``payload fields differ`` -- same payload, different decoded numbers (a decoder change
      or a hand edit);
    * ``file differs (payload identical)`` -- same payload, but this copy's file hash,
      fetch time or stored response metadata differ (a re-fetch or a re-written file).
      ``payload_only`` skips this one, which is the mode for checking a re-fetched or
      downloaded copy for real payload changes.

    With no archive (a clean clone) only the self-consistency is checked; that is not a
    failure.
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
    ddir = depth_dir(split, repo, archive_dir)
    archived = (sorted(f[:-len(".json.gz")] for f in os.listdir(ddir) if f.endswith(".json.gz"))
                if os.path.isdir(ddir) else [])
    if not archived:
        if archive_dir:
            problems.append(f"{split}: no .json.gz files in {archive_dir}")
            return problems, ""
        return problems, "archive absent, manifest self-consistent" if not problems else ""
    saved = {pid for pid, e in man["panos"].items() if e["status"] == SAVED}
    for pid in archived:
        if pid not in saved:
            status = man["panos"].get(pid, {}).get("status", "not in the manifest")
            problems.append(f"{split}/{pid}: archived but manifest says {status}")
    for pid in sorted(saved):
        e = man["panos"][pid]
        path = archive_path(split, pid, repo, archive_dir)
        if not os.path.exists(path):
            problems.append(f"{split}/{pid}: missing from the archive")
            continue
        try:
            rebuilt = _round_floats(entry_from_archive(path))
        except Exception as ex:  # noqa: BLE001 -- reported, never fatal
            problems.append(f"{split}/{pid}: unreadable ({ex})")
            continue
        if rebuilt["sha256"] != e.get("sha256"):
            problems.append(f"{split}/{pid}: payload sha256 drift")
            continue
        bad = [k for k in PAYLOAD_FIELDS if rebuilt.get(k) != e.get(k)]
        if bad:
            problems.append(f"{split}/{pid}: payload fields differ: {bad}")
            continue
        prov = [k for k in PROVENANCE_FIELDS if rebuilt.get(k) != e.get(k)]
        if prov and not payload_only:
            problems.append(f"{split}/{pid}: file differs (payload identical): {prov}")
    note = "archive verified" if not problems else f"{len(problems)} problem(s)"
    return problems, note + (f" ({archive_dir})" if archive_dir else "")


# -------------------------------------------------------------------------- compare

def index_agreement(b64_a, b64_b):
    """Share of pixels whose plane id is the same in two payloads (None if sizes differ).

    Plane ids are positions in each payload's own plane list, so a revision that renumbers
    planes lowers this even where the geometry is unchanged: a floor, not an exact measure.
    """
    wa, ha, _, ia = parse_payload(b64_a)
    wb, hb, _, ib = parse_payload(b64_b)
    if (wa, ha) != (wb, hb):
        return None
    return round(sum(x == y for x, y in zip(ia, ib)) / len(ia), 4)


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
            lg = ground_summary(lrec["depth_b64"])
            rec["labeler_n_planes"] = lg["n_planes"]
            rec["labeler_exactly_level"] = lg["exactly_level"]
            if e["status"] == SAVED:
                rec["identical"] = rec["labeler_sha256"] == e["sha256"]
                tally["identical" if rec["identical"] else "differs"] += 1
                if not rec["identical"]:
                    rec["index_agreement"] = index_agreement(
                        read_archive_file(archive_path(split, pid, repo))["depth_b64"],
                        lrec["depth_b64"])
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


def check_decoder_on_labeler_run(split, labeler_root):
    """Offline: decode EVERY payload in the labeler's runs/<split>/depth (the whole run, not
    just the benchmark panos) and compare with its index.csv: plane count, degenerate flag,
    camera height (1e-3 m) and ground tilt (1e-3 deg; the index keeps 3 dp). No network.
    Returns (n_compared, mismatches)."""
    import csv
    ldir = os.path.join(labeler_root, "runs", split, "depth")
    ipath = os.path.join(ldir, "index.csv")
    if not os.path.exists(ipath):
        raise SystemExit(f"no labeler index at {ipath}")
    n, bad = 0, []
    with open(ipath, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g = ground_summary(read_archive_file(os.path.join(ldir, row["filename"]))["depth_b64"])
            n += 1
            lab_h = row["camera_height_m"]
            ok = (int(row["n_planes"]) == g["n_planes"]
                  and (row["degenerate"] == "1") == g["degenerate"]
                  and (lab_h == "") == (g["camera_height_m"] is None))
            if ok and lab_h:
                ok = (abs(float(lab_h) - g["camera_height_m"]) <= 1e-3
                      and abs(float(row["ground_tilt_deg"]) - g["ground_tilt_deg"]) <= 1e-3)
            if not ok:
                bad.append(row["panorama_id"])
    print(f"{split}: decoder vs labeler index.csv over the whole run: {n} payloads, "
          f"{len(bad)} mismatch(es){'' if not bad else ', e.g. ' + str(bad[:3])}")
    return n, bad


# ------------------------------------------------------------------------ summarize

def city_of(pano):
    lat, lon = pano.get("pano_coord") or (pano.get("lat"), pano.get("lng"))
    for name, (a, b, c, d) in CITY_BOXES.items():
        if lat is not None and lon is not None and a <= lat <= b and c <= lon <= d:
            return name
    return "other"


# The labeler's depth.classify_height statuses, in its precedence order.
H_DEGENERATE, H_NO_GROUND, H_SYNTHETIC, H_IMPLAUSIBLE, H_MEASURED = (
    "degenerate", "no_ground", "synthetic_ground", "implausible", "measured")


def classify_height(e):
    """A saved entry's camera-height status, with the labeler's precedence: degenerate,
    then no ground, then the exactly-level stand-in, then outside the plausible window.

    Example:
        >>> classify_height({"degenerate": False, "camera_height_m": 2.5, "exactly_level": True})
        'synthetic_ground'
    """
    if e["degenerate"]:
        return H_DEGENERATE
    if e["camera_height_m"] is None:
        return H_NO_GROUND
    if e.get("exactly_level"):
        return H_SYNTHETIC
    lo, hi = PLAUSIBLE_HEIGHT_M
    if not lo <= e["camera_height_m"] <= hi:
        return H_IMPLAUSIBLE
    return H_MEASURED


def capture_era(ym):
    """Coarse capture era for the stand-in share: up to 2012, 2013-2018, 2019 on."""
    if not ym:
        return "unknown"
    y = int(ym[:4])
    return "<=2012" if y <= 2012 else ("2013-2018" if y <= 2018 else ">=2019")


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
        az = sorted(abs(((e["heading_deg"] - recs[pid]["pano_azimuth"]) + 180) % 360 - 180)
                    for pid, e in man["panos"].items()
                    if e["status"] == SAVED and e.get("heading_deg") is not None
                    and recs[pid].get("pano_azimuth") is not None)
        if az:
            out(f"response heading vs records.jsonl pano_azimuth, {len(az)} saved panos: "
                f"max |difference| {az[-1]:.3f} deg (same panoramas)")
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
        status = Counter(classify_height(e) for e in saved)
        level = sum(bool(e.get("exactly_level")) for e in saved)
        out(f"saved {len(saved)}, by camera-height status (labeler precedence: degenerate > "
            f"no ground > stand-in > implausible): measured {status[H_MEASURED]}, "
            f"stand-in {status[H_SYNTHETIC]}, degenerate {status[H_DEGENERATE]}, "
            f"implausible (outside {PLAUSIBLE_HEIGHT_M[0]}-{PLAUSIBLE_HEIGHT_M[1]} m) "
            f"{status[H_IMPLAUSIBLE]}, no ground {status[H_NO_GROUND]}")
        out(f"exactly-level stand-in ground on {level} of {len(saved)} saved "
            f"({level / len(saved):.1%}), counting the degenerate ones too")
        eras = {}
        for e in saved:
            eras.setdefault(capture_era(e.get("capture_ym")), []).append(bool(e.get("exactly_level")))
        out("stand-in share by capture era: " + ", ".join(
            f"{k} {sum(v)}/{len(v)} ({sum(v) / len(v):.0%})"
            for k, v in sorted(eras.items(), key=lambda kv: ("<", "2", ">", "u").index(kv[0][0]))))
        measured = sorted(e["camera_height_m"] for e in saved if classify_height(e) == H_MEASURED)
        tilts = sorted(e["ground_tilt_deg"] for e in saved if classify_height(e) == H_MEASURED)
        if measured:
            out(f"camera height, measured (n={len(measured)}): min {measured[0]:.3f}  "
                f"p10 {_pct(measured, .1):.3f}  median {_pct(measured, .5):.3f}  "
                f"p90 {_pct(measured, .9):.3f}  max {measured[-1]:.3f} m")
            out(f"ground tilt, measured: median {_pct(tilts, .5):.2f} deg, "
                f"p90 {_pct(tilts, .9):.2f} deg")
        cmp_path = os.path.join(split_dir(split, repo), COMPARE_NAME)
        if os.path.exists(cmp_path):
            with open(cmp_path, encoding="utf-8") as fh:
                cmp = json.load(fh)
            out(f"vs the labeler archive: {cmp['tally']}; labeler fetch dates "
                f"{dict(sorted(Counter((r.get('labeler_fetched_at') or '')[:10] for r in cmp['panos'].values()).items()))}")
            diff = {pid: r for pid, r in cmp["panos"].items() if r.get("identical") is False}
            if diff:
                ag = sorted(r["index_agreement"] for r in diff.values()
                            if r.get("index_agreement") is not None)
                dh, flips, same_np = [], 0, 0
                for pid, r in diff.items():
                    ours = man["panos"][pid]
                    same_np += ours["n_planes"] == r["labeler_n_planes"]
                    flips += bool(ours.get("exactly_level")) != bool(r.get("labeler_exactly_level"))
                    if ours.get("camera_height_m") is not None and "labeler_camera_height_m" in r:
                        dh.append(abs(ours["camera_height_m"] - r["labeler_camera_height_m"]))
                dh.sort()
                agree = (f"min {ag[0]:.3f} median {_pct(ag, .5):.3f} max {ag[-1]:.3f}"
                         if ag else "not comparable (grid sizes differ)")
                height = (f"camera height unchanged (<1 mm) {sum(d < 1e-3 for d in dh)} of "
                          f"{len(dh)}, max change {dh[-1]:.3f} m" if dh
                          else "no comparable camera heights")
                out(f"revised payloads {len(diff)}: same plane count {same_np}; plane-index "
                    f"agreement {agree}; {height}; stand-in status flipped {flips}")


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
    h.add_argument("--archive-dir", help="write payloads here instead of benchmark/<split>/depth "
                                         "(a re-fetch kept apart from the record)")
    h.add_argument("--manifest-out", help="write the manifest here (default: the committed "
                                          "record, unless that would change a resolved pano)")
    v = sub.add_parser("verify")
    v.add_argument("--split", required=True)
    v.add_argument("--archive-dir", help="check this archive copy (a re-fetch, a download) "
                                         "against the committed manifest")
    v.add_argument("--payload-only", action="store_true",
                   help="ignore file/fetch-time/metadata differences; report only payload "
                        "drift and decoded-field differences")
    c = sub.add_parser("compare-labeler")
    c.add_argument("--split", required=True, choices=("bend", "laurens_gsv"))
    c.add_argument("--labeler-root", required=True)
    c.add_argument("--all-run-panos", action="store_true",
                   help="instead: decode every payload of the labeler's whole run and compare "
                        "with its index.csv (decoder check; writes nothing)")
    sub.add_parser("summarize")
    args = ap.parse_args(argv)

    if args.check:
        return run_check()
    if args.cmd == "harvest":
        if args.limit is not None and args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        _, _, aborted = harvest(args.split, args.limit, args.resume,
                                archive_dir=args.archive_dir, manifest_out=args.manifest_out)
        return 2 if aborted else 0
    if args.cmd == "verify":
        problems, note = verify(args.split, archive_dir=args.archive_dir,
                                payload_only=args.payload_only)
        for p in problems:
            print(p)
        print(f"{args.split}: {'FAIL' if problems else 'ok'} ({note})")
        return 1 if problems else 0
    if args.cmd == "compare-labeler":
        if args.all_run_panos:
            _, bad = check_decoder_on_labeler_run(args.split, args.labeler_root)
            return 1 if bad else 0
        compare_labeler(args.split, args.labeler_root)
        return 0
    if args.cmd == "summarize":
        summarize()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
