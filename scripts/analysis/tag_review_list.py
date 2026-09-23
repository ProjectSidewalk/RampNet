"""Build the stratified, seeded curb-ramp tag review list (RampNet 2.0 plan item 3, #86).

The list is what Jon, and then a second rater, review on production under the draft rubric
in ``docs/tag_rubric_draft.md``; the protocol is ``docs/tag_review_protocol.md``. Per-tag
agreement between the two passes is the human ceiling every tag model is measured against.

Input is the Project Sidewalk API cache written by PR #175's
``scripts/analysis/ps_supervision_audit.py fetch`` (``analysis_out/ps_audit/raw/``,
gitignored): per deployment ``<city>__rawLabels__CurbRamp.csv``,
``<city>__validations__CurbRamp.csv``, ``<city>__labelTags.json`` and the
``fetch_manifest.json`` that records each deployment's host and each file's sha256.
Production is a live database, so a fresh fetch gives a different pool: **the committed
``benchmark/tag_review/review_list.csv`` is the artifact of record**, and this script
reproduces it byte for byte only from a cache whose hashes match
``review_list.meta.json``.

    # the list (defaults = the committed one)
    python scripts/analysis/tag_review_list.py build --cache analysis_out/ps_audit/raw

    # the kappa-precision table the default size rests on
    python scripts/analysis/tag_review_list.py power

Strata (full definitions in the rubric doc, section "The review list"):

- **tag state**, precedence top down: ``affirmed_empty`` (no tags now, and a tag-review
  pass looked at it: ``ExpertValidate`` or the ASSETS'24 ``ExternalTagValidationASSETS2024``
  pass), ``tagged_trusted`` (tagged, placed by a trusted account), ``tagged`` (tagged by
  anyone else), ``untagged`` (no tags, never tag-reviewed; *not* a negative);
- **distance band** from the label's depression angle below the horizon,
  ``(pano_y / pano_height - 0.5) * 180 - camera_pitch`` degrees (the image-to-world pitch
  relation in SidewalkWebpage's ``PannellumViewer.js``: world = image + cameraPitch), turned
  into a flat-ground distance with a fixed 2.5 m camera height, the same constant as
  ``crop_window_eval.py`` and ``size_analysis.py``: near < 8 m, mid 8-15 m, far >= 15 m;
- **city**, with equal allocation across eligible cities (capped by what each city has), so
  Chicago's 67k crop-era labels do not outvote Laurens's 143.

Within a tagged stratum a label is drawn with weight 1 / (frequency of its rarest tag), so
the rare tags reach enough positives for a per-tag kappa; within a (state, city) cell the
draw rotates through the distance bands. At most one label per (city, pano) and none within
``--min-sep-m`` of an already-drawn label in the same city, so two labels of one physical
ramp (plan §2.5) are not two items. Item order is a seeded shuffle, so a rater never sees a
block of one stratum.
"""
import argparse
import csv
import datetime as dt
import glob
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rampnet import tag_review as tr  # noqa: E402

ANALYSIS_OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
DEFAULT_CACHE = os.path.join(ANALYSIS_OUT, "ps_audit", "raw")
DEFAULT_OUT = os.path.join(REPO, "benchmark", "tag_review", "review_list.csv")

#: The SidewalkAI account (model output, not supervision). Same id as PR #175's audit.
SIDEWALK_AI_USER = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"
#: Owner accounts = the default trusted placers. Ids are shared across deployments.
OWNERS = {
    "549187e0-82c9-4014-a48d-31f18083d575": "jonfroehlich",
    "18b26a38-24ab-402d-a64e-158fc0bb8a8a": "mikey",
}
#: Validation sources that looked at the tag set (every other source is a position vote).
TAG_REVIEW_SOURCES = ("ExpertValidate", "ExternalTagValidationASSETS2024")
#: The validation-study deployment holds the 94 disputed "missing tactile warning" ramps
#: whose adjudication is deferred (Jon, #86, 2026-09-22); none of its labels enter the list.
DEFAULT_EXCLUDE_CITIES = ("validation-study",)
#: Production stores a browser crop for every label placed on or after this date.
CROP_DATE = "2023-10-12"

STATES = ("affirmed_empty", "tagged_trusted", "tagged", "untagged")
DEFAULT_SHARES = "affirmed_empty=0.20,tagged_trusted=0.15,tagged=0.35,untagged=0.30"
BANDS = ("near", "mid", "far")
CAMERA_HEIGHT_M = 2.5
BAND_EDGES_M = (8.0, 15.0)

LIST_COLUMNS = (
    "item_id", "city", "label_id", "label_uid", "pano_id", "tag_state", "distance_band",
    "depression_deg", "est_distance_m", "camera_pitch", "label_heading_deg", "label_pitch_deg", "placed_at", "image_capture_date",
    "tags_at_list", "severity_at_list", "applicable_tags", "tag_reviewed_by", "placed_by_rater",
    "has_server_crop", "editor_url", "labelmap_url", "gsv_url",
)

RAW_COLS = ["label_id", "user_id", "pano_id", "pano_source", "severity", "tags", "time_created",
            "correct", "pano_y", "pano_height", "camera_pitch", "latitude", "longitude",
            "image_capture_date", "pano_x", "pano_width", "camera_heading"]


# ----------------------------------------------------------------------------- geometry

def depression_deg(pano_y, pano_height, camera_pitch):
    """Degrees below the horizon of a pano pixel row, corrected for camera pitch."""
    pitch = np.nan_to_num(np.asarray(camera_pitch, dtype=float), nan=0.0)
    return (np.asarray(pano_y, dtype=float) / np.asarray(pano_height, dtype=float) - 0.5) * 180.0 - pitch


def flat_ground_distance_m(dep_deg, camera_height_m=CAMERA_HEIGHT_M):
    """Flat-ground distance for a depression angle; ``inf`` at or above the horizon."""
    dep = np.asarray(dep_deg, dtype=float)
    with np.errstate(divide="ignore"):
        d = camera_height_m / np.tan(np.radians(dep))
    return np.where(dep > 0, d, np.inf)


def distance_band(dist_m, edges=BAND_EDGES_M):
    d = np.asarray(dist_m, dtype=float)
    return np.where(d < edges[0], "near", np.where(d < edges[1], "mid", "far"))


def label_view(pano_x, pano_width, camera_heading, dep_deg):
    """(heading, pitch) in degrees that centre a viewer on the label.

    Column 0 of a GSV equirectangular pano faces ``camera_heading - 180``; pitch is minus
    the pitch-corrected depression. Checked against the labeller's own POV on seattle
    label 9 (POV heading 299.3 with the label right of centre; this gives 302.3)."""
    heading = (np.asarray(pano_x, dtype=float) / np.asarray(pano_width, dtype=float) * 360.0
               + np.nan_to_num(np.asarray(camera_heading, dtype=float)) - 180.0) % 360.0
    return heading, -np.asarray(dep_deg, dtype=float)


def gsv_url(pano_id, heading, pitch, fov=60):
    """A Google Maps viewer link pinned to this pano id, centred on the label. Shows the
    imagery only: no Project Sidewalk tags, so it is the blind rater's view (protocol)."""
    return (f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}"
            f"&heading={heading:.1f}&pitch={pitch:.1f}&fov={fov}")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------- load

def load_manifest(cache):
    path = os.path.join(cache, "fetch_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run PR #175's `ps_supervision_audit.py fetch` first")
    return tr.read_json(path)


def city_vocab(cache, city):
    """CurbRamp tags this deployment offers today (its ``/v3/api/labelTags`` view)."""
    with open(os.path.join(cache, f"{city}__labelTags.json"), encoding="utf-8") as fh:
        tags = json.load(fh)["label_tags"]
    return sorted({t["tag"] for t in tags if t["label_type"] == "CurbRamp"})


def load_pool(cache, cities):
    frames = []
    for city in cities:
        d = pd.read_csv(os.path.join(cache, f"{city}__rawLabels__CurbRamp.csv"), usecols=RAW_COLS,
                        dtype={"correct": "object", "pano_id": "object", "image_capture_date": "object"})
        d["city"] = city
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_tag_reviews(cache, cities):
    """(city, label_id) -> sorted tag-review sources, from the validations cache."""
    out = {}
    for city in cities:
        path = os.path.join(cache, f"{city}__validations__CurbRamp.csv")
        if not os.path.exists(path):
            continue
        v = pd.read_csv(path, usecols=["label_id", "source"])
        v = v[v.source.isin(TAG_REVIEW_SOURCES)]
        for lid, grp in v.groupby("label_id"):
            out[(city, int(lid))] = sorted(set(grp.source))
    return out


def read_trusted(path):
    """User ids from a ``role|username|user_id`` file (PR #175's ``--trusted-users`` format)."""
    ids = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0] in ("Owner", "Administrator", "Researcher"):
                ids[parts[2]] = parts[1]
    return ids


def parse_shares(s):
    shares = {}
    for part in s.split(","):
        k, v = part.split("=")
        if k not in STATES:
            raise SystemExit(f"unknown tag state {k!r}; states are {STATES}")
        shares[k] = float(v)
    total = sum(shares.values())
    return {k: shares.get(k, 0.0) / total for k in STATES}


# ----------------------------------------------------------------------------- candidates

def build_candidates(cache, *, exclude_cities, require_tags, crop_date, sources, trusted,
                     min_city_pool, raters):
    manifest = load_manifest(cache)
    hosts = {c: h["url"] for c, h in manifest.get("hosts", {}).items()}
    cities_all = sorted(os.path.basename(p).split("__")[0]
                        for p in glob.glob(os.path.join(cache, "*__rawLabels__CurbRamp.csv")))
    dropped = {}
    cities = []
    vocab = {}
    for c in cities_all:
        if c in exclude_cities:
            dropped[c] = "excluded by --exclude-cities"
            continue
        v = city_vocab(cache, c)
        missing = sorted(set(require_tags) - set(v))
        if missing:
            dropped[c] = "vocabulary lacks " + ", ".join(missing)
            continue
        if c not in hosts:
            dropped[c] = "no host in fetch_manifest.json"
            continue
        vocab[c] = [t for t in v if t not in tr.RETIRED_TAGS]
        cities.append(c)

    d = load_pool(cache, cities)
    n0 = len(d)
    funnel = [("CurbRamp labels in the vocabulary-eligible cities, SidewalkAI included", n0)]
    d = d[d.user_id != SIDEWALK_AI_USER]
    funnel.append(("minus SidewalkAI", len(d)))
    d["t"] = pd.to_datetime(d.time_created, utc=True, format="ISO8601")
    d = d[d.t >= pd.Timestamp(crop_date, tz="UTC")]
    funnel.append((f"placed on/after {crop_date} (has a production crop)", len(d)))
    d = d[d.correct.fillna("").str.lower() != "false"]
    funnel.append(("minus crowd-validated incorrect (correct == false)", len(d)))
    d = d[d.pano_source.isin(sources)]
    funnel.append((f"pano source in {sorted(sources)}", len(d)))
    d = d[(d.pano_height > 0) & d.pano_y.notna() & d.latitude.notna()]
    funnel.append(("with pano geometry and a location", len(d)))

    d = d.copy()
    d["tag_list"] = d.tags.map(tr.parse_tag_list)
    d["n_tags"] = d.tag_list.map(len)
    reviews = load_tag_reviews(cache, cities)
    d["tag_reviewed_by"] = [";".join(reviews.get((c, int(l)), [])) for c, l in zip(d.city, d.label_id)]
    trusted_ids = set(trusted)
    d["state"] = np.select(
        [(d.n_tags == 0) & (d.tag_reviewed_by != ""),
         (d.n_tags > 0) & d.user_id.isin(trusted_ids),
         d.n_tags > 0],
        ["affirmed_empty", "tagged_trusted", "tagged"], default="untagged")
    d["dep"] = depression_deg(d.pano_y, d.pano_height, d.camera_pitch)
    d["dist"] = flat_ground_distance_m(d.dep)
    d["band"] = distance_band(d.dist)
    d["lab_heading"], d["lab_pitch"] = label_view(d.pano_x, d.pano_width, d.camera_heading, d.dep)
    d["placed_by_rater"] = d.user_id.map(lambda u: raters.get(u, ""))

    pool = d.groupby("city").size().reindex(cities, fill_value=0)
    small = sorted(pool[pool < min_city_pool].index)
    for c in small:
        dropped[c] = f"fewer than {min_city_pool} eligible labels ({int(pool[c])})"
    d = d[~d.city.isin(small)]
    funnel.append((f"in cities with >= {min_city_pool} eligible labels", len(d)))
    d = d.sort_values(["city", "label_id"]).reset_index(drop=True)
    return d, dict(hosts=hosts, vocab=vocab, dropped=dropped, funnel=funnel)


# ----------------------------------------------------------------------------- allocation

def largest_remainder(total, shares):
    raw = {k: total * s for k, s in shares.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
        out[k] += 1
    return out


def water_fill(total, avail, rng):
    """Split ``total`` as equally as possible over keys, never exceeding ``avail[k]``."""
    alloc = {k: 0 for k in avail}
    active = sorted(k for k in avail if avail[k] > 0)
    remaining = total
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for k in list(rng.permutation(active))[:remaining]:
                alloc[k] += 1
            remaining = 0
            break
        for k in active:
            g = min(share, avail[k] - alloc[k])
            alloc[k] += g
            remaining -= g
        active = [k for k in active if alloc[k] < avail[k]]
    return alloc


def rare_tag_weights(frame):
    """1 / global frequency of each label's rarest tag; 1 for untagged labels."""
    freq = {}
    for tags in frame.tag_list:
        for t in tags:
            freq[t] = freq.get(t, 0) + 1
    return np.array([1.0 / min(freq[t] for t in tags) if tags else 1.0 for tags in frame.tag_list])


def draw(cands, n_total, shares, seed, min_sep_m, rare_power=1.5):
    """The stratified draw. Returns (selected index list, per-cell shortfall report)."""
    rng = np.random.default_rng(seed)
    targets = largest_remainder(n_total, shares)
    taken, taken_panos, taken_pts = [], set(), {}
    shortfall = []

    def ok(i):
        r = cands.loc[i]
        if (r.city, r.pano_id) in taken_panos:
            return False
        for lat, lon in taken_pts.get(r.city, []):
            if haversine_m(r.latitude, r.longitude, lat, lon) < min_sep_m:
                return False
        return True

    def take(i):
        r = cands.loc[i]
        taken.append(i)
        taken_panos.add((r.city, r.pano_id))
        taken_pts.setdefault(r.city, []).append((r.latitude, r.longitude))

    # Rarest state first, so its few candidates are not blocked by a commoner state's picks.
    city_order = {c: k for k, c in enumerate(sorted(cands.city.unique()))}
    for state in STATES:
        sub = cands[cands.state == state]
        if sub.empty:
            if targets[state]:
                shortfall.append((state, "*", targets[state]))
            continue
        w = rare_tag_weights(sub) ** rare_power if state.startswith("tagged") else np.ones(len(sub))
        # Efraimidis-Spirakis keys: ascending -log(u)/w is a weighted sample w/o replacement.
        keys = -np.log(rng.random(len(sub))) / w
        sub = sub.assign(_key=keys).sort_values(["_key", "city", "label_id"])
        queues = {c: {b: list(g.index[g.band == b]) for b in BANDS} for c, g in sub.groupby("city")}
        # Each city's band rotation starts at a different band, so the remainder of a small
        # per-city quota does not always land on "near".
        turn = {c: city_order[c] % len(BANDS) for c in queues}
        remaining = targets[state]
        # Rounds: a city whose candidates are blocked by the de-duplication hands its unmet
        # quota back, and the next round re-splits it over the cities that still have some.
        while remaining > 0:
            avail = {c: sum(len(q) for q in qs.values()) for c, qs in queues.items()}
            if not any(avail.values()):
                break
            alloc = water_fill(remaining, avail, rng)
            got_round = 0
            for city in sorted(alloc):
                got = 0
                while got < alloc[city] and any(queues[city].values()):
                    b = BANDS[turn[city]]
                    turn[city] = (turn[city] + 1) % len(BANDS)
                    while queues[city][b]:
                        i = queues[city][b].pop(0)
                        if ok(i):
                            take(i)
                            got += 1
                            break
                got_round += got
            remaining -= got_round
        if remaining > 0:
            shortfall.append((state, "*", remaining))
    return taken, targets, shortfall


# ----------------------------------------------------------------------------- output

def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def list_rows(sel, hosts, vocab, seed):
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(sel))
    sel = sel.iloc[order].reset_index(drop=True)
    rows = []
    for k, r in sel.iterrows():
        host = hosts[r.city]
        lid = int(r.label_id)
        sev = "" if pd.isna(r.severity) else str(int(r.severity))
        rows.append({
            "item_id": f"tr{k + 1:04d}",
            "city": r.city,
            "label_id": lid,
            "label_uid": f"{r.city}:{lid}",
            "pano_id": r.pano_id,
            "tag_state": r.state,
            "distance_band": r.band,
            "depression_deg": _fmt(float(r.dep), 2),
            "est_distance_m": _fmt(float(r.dist), 1),
            "camera_pitch": _fmt(float(r.camera_pitch), 3) if pd.notna(r.camera_pitch) else "",
            "label_heading_deg": _fmt(float(r.lab_heading), 1),
            "label_pitch_deg": _fmt(float(r.lab_pitch), 1),
            "placed_at": r.t.strftime("%Y-%m-%d"),
            "image_capture_date": "" if pd.isna(r.image_capture_date) else str(r.image_capture_date),
            "tags_at_list": tr.join_tags(r.tag_list),
            "severity_at_list": sev,
            "applicable_tags": tr.join_tags(vocab[r.city]),
            "tag_reviewed_by": r.tag_reviewed_by,
            "placed_by_rater": r.placed_by_rater,
            "has_server_crop": "true",
            "editor_url": f"{host}/gallery?labelType=CurbRamp&labelId={lid}",
            "labelmap_url": f"{host}/labelMap?labelId={lid}",
            "gsv_url": gsv_url(r.pano_id, float(r.lab_heading), float(r.lab_pitch)),
        })
    return rows


def write_csv(path, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(LIST_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return tr.sha256_bytes(data)


def read_list(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def composition(rows):
    """Tables the doc and the PR quote: state x band, per city, per-tag list-time positives."""
    df = pd.DataFrame(rows)
    sxb = pd.crosstab(df.tag_state, df.distance_band).reindex(index=list(STATES), columns=list(BANDS),
                                                            fill_value=0)
    per_city = df.groupby("city").size().sort_values(ascending=False)
    tag_pos = {}
    for tags in df.tags_at_list:
        for t in tr.parse_tag_list(tags):
            tag_pos[t] = tag_pos.get(t, 0) + 1
    return {
        "n": len(df),
        "state_by_band": {s: {b: int(sxb.loc[s, b]) for b in BANDS} for s in STATES},
        "per_city": {c: int(n) for c, n in per_city.items()},
        "tag_positives_at_list": dict(sorted(tag_pos.items())),
        "placed_by_rater": {k or "(neither)": int(v) for k, v in df.placed_by_rater.value_counts().sort_index().items()},
        "tag_reviewed_before": int((df.tag_reviewed_by != "").sum()),
    }


def cmd_build(args):
    raters = dict(OWNERS)
    trusted = dict(OWNERS)
    if args.trusted_users:
        trusted.update(read_trusted(args.trusted_users))
    cands, info = build_candidates(
        args.cache, exclude_cities=set(args.exclude_cities), require_tags=args.require_tags,
        crop_date=args.crop_date, sources=set(args.sources), trusted=trusted,
        min_city_pool=args.min_city_pool, raters=raters)
    shares = parse_shares(args.shares)
    idx, targets, shortfall = draw(cands, args.n, shares, args.seed, args.min_sep_m, args.rare_power)
    rows = list_rows(cands.loc[idx], info["hosts"], info["vocab"], args.seed)
    digest = write_csv(args.out, rows)
    comp = composition(rows)
    manifest = load_manifest(args.cache)
    used = sorted({r["city"] for r in rows})
    inputs = {}
    for c in sorted(info["vocab"]):
        for ep in ("rawLabels__CurbRamp", "validations__CurbRamp", "labelTags"):
            m = manifest["files"].get(f"{c}__{ep}", {})
            inputs[f"{c}__{ep}"] = {"sha256": m.get("sha256"), "fetched_at": m.get("fetched_at")}
    meta = {
        "generated_by": "scripts/analysis/tag_review_list.py build",
        "generated_on": dt.date.today().isoformat(),
        "list": {"path": "benchmark/tag_review/review_list.csv", "sha256": digest, "rows": len(rows)},
        "params": {"n": args.n, "seed": args.seed, "shares": shares, "crop_date": args.crop_date,
                   "sources": sorted(args.sources), "exclude_cities": sorted(args.exclude_cities),
                   "require_tags": sorted(args.require_tags), "min_city_pool": args.min_city_pool,
                   "min_sep_m": args.min_sep_m, "rare_power": args.rare_power, "camera_height_m": CAMERA_HEIGHT_M,
                   "band_edges_m": list(BAND_EDGES_M),
                   "trusted_users": "owners only" if not args.trusted_users else "owners + --trusted-users file"},
        "funnel": [{"step": s, "labels": int(n)} for s, n in info["funnel"]],
        "pool_by_state": {s: int((cands.state == s).sum()) for s in STATES},
        "pool_by_city": {c: int(n) for c, n in cands.groupby("city").size().items()},
        "cities_dropped": dict(sorted(info["dropped"].items())),
        "targets": targets,
        "shortfall": [{"state": s, "city": c, "missing": int(m)} for s, c, m in shortfall],
        "composition": comp,
        "cities_in_list": used,
        "inputs": inputs,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    tr.write_json(meta_path, meta)
    print(f"wrote {args.out} ({len(rows)} items, sha256 {digest})")
    print(f"wrote {meta_path}")
    print(json.dumps({"targets": targets, "shortfall": meta["shortfall"], **comp}, indent=1))


def cmd_power(args):
    print(f"95% CI half-width of Cohen's kappa (simulated, {args.sims} draws), true kappa {args.kappa:.2f}")
    print("| positives (n x prevalence) | " + " | ".join(f"n={n}" for n in args.ns) + " |")
    print("|---|" + "---|" * len(args.ns))
    for pos in args.positives:
        cells = []
        for n in args.ns:
            p = pos / n
            cells.append(_fmt(tr.kappa_ci_halfwidth(n, p, args.kappa, sims=args.sims, seed=args.seed), 2)
                         if 0 < p < 0.5 else "")
        print(f"| {pos} | " + " | ".join(cells) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="draw the list")
    b.add_argument("--cache", default=DEFAULT_CACHE, help="PR #175 audit cache (analysis_out/ps_audit/raw)")
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--n", type=int, default=500, help="list size (see `power` for the rationale)")
    b.add_argument("--seed", type=int, default=86)
    b.add_argument("--shares", default=DEFAULT_SHARES, help="tag-state shares, state=frac,...")
    b.add_argument("--crop-date", default=CROP_DATE)
    b.add_argument("--sources", nargs="+", default=["gsv"], help="pano sources to keep")
    b.add_argument("--exclude-cities", nargs="*", default=list(DEFAULT_EXCLUDE_CITIES))
    b.add_argument("--require-tags", nargs="*", default=list(tr.CORE_TAGS),
                   help="a city must offer all of these tags to be eligible")
    b.add_argument("--rare-power", type=float, default=1.5,
                   help="tagged-state draw weight = (1 / freq of the label's rarest tag) ** this")
    b.add_argument("--min-city-pool", type=int, default=100)
    b.add_argument("--min-sep-m", type=float, default=10.0,
                   help="minimum distance between two listed labels in one city")
    b.add_argument("--trusted-users", default=None,
                   help="role|username|user_id file to widen the trusted placers (not committed)")
    b.set_defaults(func=cmd_build)
    p = sub.add_parser("power", help="kappa CI half-width vs list size and positives")
    p.add_argument("--ns", type=int, nargs="+", default=[300, 500, 800])
    p.add_argument("--positives", type=int, nargs="+", default=[10, 20, 30, 50, 80])
    p.add_argument("--kappa", type=float, default=0.6)
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_power)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
