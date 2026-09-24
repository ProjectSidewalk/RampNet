"""Recall by distance on GSV's own depth instead of flat-ground geometry (#112).

``docs/detection_recall_analysis.md`` conditions its headline ("reliable to ~18 m, blind past
25 m") on a distance axis built from flat-ground geometry at an assumed camera height,
``d = CAM_H / tan(depression)`` with ``CAM_H = 2.5``, cross-checked against Depth Anything 3.
GSV serves a metric depth payload with every panorama -- a list of planes plus a per-pixel
plane index -- and the dominant ground plane's distance *is* the camera height. This script
re-derives the distance axis from that payload for every GSV benchmark split that has one
archived (bend, paterson, gainesville, sao_paulo; laurens_gsv was never harvested) and
re-issues the doc's tables on both axes, side by side.

The payloads and the parser come from the sidewalk-auto-labeler repo (``depth.py`` at its
root, stdlib only; ``runs/<city>/depth/<pano_id>.json.gz`` + ``index.csv`` written by its
``scripts/harvest_depth.py``). The parser is public; the payloads are an unpublished input
(see the doc's caveats). Which is why the committed JSON carries every per-point row: all of
the tables re-derive from the rows on CPU with no payload in sight (``--check``).

The distance rule, stated once:

  * **flat** -- ``h / tan(depression)`` at the fixed 2.5 m (and 2.6 m, the labeler's old
    constant, for the issue's check). Horizontal range. Infinite at or above the horizon.
  * **depth** -- the horizontal range along the exact ray through the point to the payload
    plane under its pixel: plane lookup is per pixel of the 512x256 index, the intersection
    is continuous. If that plane is not ground-like (tilt > ``GROUND_MAX_TILT_DEG`` -- a
    wall, a car) or the pixel is sky, the point falls back to **level ground at the
    measured camera height** (``flat_range`` at the pano's ground-plane distance; the
    ground plane's tilt is not used there), and the row says so (``depth_source``).
    Euclidean ray distance is kept beside it and is what apparent size uses (a subtended
    angle depends on the ray, not the horizontal).
  * A panorama whose ground is a stand-in (Google's exactly-level 2.500 m plane,
    ``depth.SYNTHETIC_GROUND``), a fallback reconstruction (``DEGENERATE``), or has no
    usable floor is **excluded from the depth axis**, never silently backfilled: its rows
    carry ``camera_height_status`` and no depth range. The doc reports the count per split.

The image <-> payload mapping, stated once because it is the easiest thing here to get
wrong (#112, PR #184 review B1): **benchmark image column c is raw payload index column c**,
and the ray through image x has azimuth ``phi = (1 - x) * 2pi + pi/2`` -- the labeler's
``depth._direction`` evaluated at raw column ``x * width``. The labeler's own lookup
(``depth._plane_at`` / ``ray_depth_at`` / ``ground_range_at``) instead maps a stored column
c to raw column ``width - 1 - c``: correct for streetlevel's rastered depth map, which is
mirrored, and **mirrored in azimuth relative to the RampNet benchmark JPEGs**. This script
therefore uses the labeler only to parse payloads and classify the ground plane, and does
the lookup and the ray itself (``raw_column``, ``image_ray``). The mapping is evidenced, not
assumed: ``depth_image_alignment_112.py`` checks it against the benchmark JPEGs (sky mask
vs image, ground plane under the GT points, plane boundaries vs image edges, and seam
continuity of the raw-space ray) and commits the result.

Ground truth and hits are exactly the doc's: ``build_ground_truth`` over each split's
``verdicts.json``, hit = RampNet's committed detection (the deployed 0.55 operating point,
``records.jsonl``) claims the GT point under the greedy confidence-ordered matcher
``depth_extract_da3.py`` used; the pooled richmond+bend count reproduces the doc's 637 / 0.765.

    # derive (needs the labeler checkout with its depth archive; ~10 s on CPU)
    python scripts/analysis/recall_by_depth_112.py --labeler-root D:/Git/sidewalk-auto-labeler

    # re-derive every table from the committed rows, no payloads, and fail on drift
    python scripts/analysis/recall_by_depth_112.py --check

    # the markdown the doc's tables are pasted from
    python scripts/analysis/recall_by_depth_112.py --check --markdown
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import statistics as st
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, _xy, build_ground_truth, radius_sq_for)

OUT_JSON = os.path.join(REPO, "analysis_out", "recall_by_depth_112.json")
OUT_MD = os.path.join(REPO, "analysis_out", "recall_by_depth_112.md")

CAM_H = 2.5            # the doc's constant (depth_extract_da3.py, size_analysis.py)
CAM_H_LABELER = 2.6    # the labeler's old geo.DEFAULT_CAMERA_HEIGHT_M, the issue's check
RAMP_W = 1.2           # the doc's nominal ramp width
PX_PER_RAD = 4096.0 / (2 * math.pi)
R = math.sqrt(radius_sq_for())
DEPTH_SPLITS = ("bend", "paterson", "gainesville", "sao_paulo")   # GSV, harvested
FLAT_ONLY_SPLITS = ("richmond",)   # the doc's other city: Mapillary, no depth exists
M_BUCKETS = [(0, 8), (8, 12), (12, 18), (18, 25), (25, 40), (40, 1e9)]
PX_BUCKETS = [(0, 12), (12, 20), (20, 32), (32, 50), (50, 80), (80, 1e9)]
RESOLUTION_FACTORS = (1.5, 2.0, 3.0)
PUBLISHED_THRESHOLDS_M = (18.0, 25.0)   # "reliable to ~18 m, blind past 25 m"
THRESHOLD_WINDOW = 0.2   # deflate a threshold t by the median ratio of points with flat in t*(1 +/- this)
ND = 4                 # decimals in every committed float (never the x / y coordinates)
GROUND_MAX_TILT_DEG = 18.0   # the labeler's depth.GROUND_MAX_TILT_DEG; derive() asserts they agree
SKY = 0                      # plane index 0 = no plane (the labeler's depth.SKY)
NEW_RIG = (("paterson", "2025"), ("gainesville", "2026"))   # Google's 2025-26 rig, by capture year
MIN_YEAR_N = 20          # a (split, capture year) row is tabulated from this many GT points

# The depth frame runs short of the height the imagery's own geometry implies, by a
# per-city factor the labeler measured by bearing-only triangulation (its
# docs/camera-height-study.md on branch camera-height-40, "self-consistent scale"; stated
# there as approximate, and whether it is a scale or an additive offset is an open question
# in that study). Applied here as a multiplicative scale on every depth range, as a second
# depth column, never silently folded into the first.
DEPTH_FRAME_SCALE = {"bend": 1.06, "paterson": 1.08, "gainesville": 1.095, "sao_paulo": 1.16}


# ---------------------------------------------------------------------------
# pure geometry (unit-tested)

def flat_range(y_norm, cam_h):
    """Horizontal range of a ground point at normalized y, level ground, camera at cam_h.

    ``None`` at or above the horizon (``y_norm <= 0.5``), where the flat-ground model has
    no answer; the doc's scripts return inf there and drop the point.
    """
    dep = (y_norm - 0.5) * math.pi
    if dep <= 1e-4:
        return None
    return cam_h / math.tan(dep)


def apparent_px(ray_m, ramp_w=RAMP_W):
    """Pixels a ramp of width ramp_w subtends at ray distance ray_m in the 4096-px input."""
    return ramp_w / ray_m * PX_PER_RAD


def bucket_of(value, buckets):
    for lo, hi in buckets:
        if lo <= value < hi:
            return (lo, hi)
    return None


def bucket_label(lo, hi, unit):
    return f"{lo:g}-{hi:g} {unit}" if hi < 1e8 else f"{lo:g} {unit}+"


def recall_table(points, axis_key, buckets, unit):
    """[{bucket, n, hit, recall}] over points that have the axis; plus an ``all`` row."""
    rows = []
    have = [p for p in points if p.get(axis_key) is not None]
    for lo, hi in buckets:
        b = [p for p in have if lo <= p[axis_key] < hi]
        if not b:
            continue
        hit = sum(1 for p in b if p["hit"])
        rows.append({"bucket": bucket_label(lo, hi, unit), "lo": lo, "hi": hi, "n": len(b),
                     "hit": hit, "recall": round(hit / len(b), ND)})
    hit = sum(1 for p in have if p["hit"])
    rows.append({"bucket": "all", "lo": 0, "hi": 1e9, "n": len(have), "hit": hit,
                 "recall": round(hit / len(have), ND) if have else None})
    return rows


def resolution_forecast(points, px_key, factors=RESOLUTION_FACTORS):
    """The doc's §4 forecast: each point takes the recall of the bucket at k x its size."""
    have = [p for p in points if p.get(px_key) is not None]
    if not have:
        return []
    by = {}
    for lo, hi in PX_BUCKETS:
        b = [p for p in have if lo <= p[px_key] < hi]
        if b:
            by[(lo, hi)] = sum(1 for p in b if p["hit"]) / len(b)
    top = max(by.values())

    def recall_at(px):
        for (lo, hi), v in by.items():
            if lo <= px < hi:
                return v
        return top

    base = sum(1 for p in have if p["hit"]) / len(have)
    out = []
    for k in factors:
        pred = sum(recall_at(p[px_key] * k) for p in have) / len(have)
        out.append({"factor": k, "base": round(base, ND), "forecast": round(pred, ND),
                    "gain": round(pred - base, ND)})
    return out


def precision_table(dets, axis_key, buckets, unit):
    rows = []
    have = [d for d in dets if d.get(axis_key) is not None and d["kind"] != "IGN"]
    for lo, hi in buckets:
        b = [d for d in have if lo <= d[axis_key] < hi]
        if not b:
            continue
        tp = sum(1 for d in b if d["kind"] == "TP")
        rows.append({"bucket": bucket_label(lo, hi, unit), "lo": lo, "hi": hi, "n": len(b),
                     "tp": tp, "precision": round(tp / len(b), ND)})
    tp = sum(1 for d in have if d["kind"] == "TP")
    rows.append({"bucket": "all", "lo": 0, "hi": 1e9, "n": len(have), "tp": tp,
                 "precision": round(tp / len(have), ND) if have else None})
    return rows


def deflation(points, flat_key="flat_2p5", depth_key="depth_range"):
    """How much longer the flat axis is than the depth axis, two ways.

    ``ratio_of_medians`` is what the issue tabulated; ``median_ratio`` is the per-point
    median of flat/depth, the stretch at the *median point*. Neither is the factor to
    deflate a far-field threshold by: the ratio grows with distance, so 18 m and 25 m are
    deflated by ``window_threshold`` instead, from the points near each threshold.
    """
    pairs = [(p[flat_key], p[depth_key]) for p in points
             if p.get(flat_key) is not None and p.get(depth_key)]
    if len(pairs) < 3:
        return None
    ratios = sorted(f / d for f, d in pairs)
    return {"n": len(pairs),
            "median_flat": round(st.median(f for f, _ in pairs), ND),
            "median_depth": round(st.median(d for _, d in pairs), ND),
            "ratio_of_medians": round(st.median(f for f, _ in pairs)
                                      / st.median(d for _, d in pairs), ND),
            "median_ratio": round(st.median(ratios), ND),
            "p10_ratio": round(ratios[int(0.1 * (len(ratios) - 1))], ND),
            "p90_ratio": round(ratios[int(0.9 * (len(ratios) - 1))], ND)}


def deflated_thresholds(median_ratio, thresholds=PUBLISHED_THRESHOLDS_M):
    return [round(t / median_ratio, 1) for t in thresholds]


def window_threshold(points, t, flat_key="flat_2p5", depth_key="depth_range", rel=THRESHOLD_WINDOW):
    """Where a flat-axis threshold t lands on the depth axis, from the points near it.

    Median of flat/depth over the points whose flat distance lies in ``t * (1 -/+ rel)``,
    and ``t`` divided by it. ``None`` for the threshold when fewer than 10 points fall in the
    window. Example: bend's 18 m uses its GT points at flat 14.4-21.6 m.
    """
    lo, hi = t * (1 - rel), t * (1 + rel)
    ratios = sorted(p[flat_key] / p[depth_key] for p in points
                    if p.get(flat_key) is not None and p.get(depth_key) and lo <= p[flat_key] <= hi)
    if len(ratios) < 10:
        return {"t": t, "window_m": [round(lo, 1), round(hi, 1)], "n": len(ratios),
                "median_ratio": None, "deflated_m": None}
    r = st.median(ratios)
    return {"t": t, "window_m": [round(lo, 1), round(hi, 1)], "n": len(ratios),
            "median_ratio": round(r, ND), "deflated_m": round(t / r, 1)}


# ---------------------------------------------------------------------------
# the image <-> payload mapping (B1 of the PR #184 review; see the module docstring)

def raw_column(x_norm, width):
    """Raw payload index column under benchmark-image x: the identity, image column c is raw c.

    NOT the labeler's ``depth._raw_column`` (``width - 1 - c``), which is for streetlevel's
    mirrored raster. ``x`` wraps at the seam. Example: ``raw_column(0.75, 512) == 384``.
    """
    return min(width - 1, max(0, int((x_norm % 1.0) * width)))


def image_ray(x_norm, y_norm):
    """Unit ray through an exact benchmark-image coordinate, in the payload's plane frame.

    The labeler's raw-column ray ``depth._direction`` (``phi = (width - col - 0.5)/width *
    2pi + pi/2``, +z down) made continuous at raw column ``col + 0.5 = x * width``, so
    ``phi = (1 - x) * 2pi + pi/2``; ``theta = (1 - y) * pi``. At a pixel centre it equals
    ``depth._direction(payload, row, raw_column(x))`` exactly.
    """
    theta = (1.0 - y_norm) * math.pi
    phi = (1.0 - x_norm) * 2.0 * math.pi + math.pi / 2.0
    s = math.sin(theta)
    return s * math.cos(phi), s * math.sin(phi), math.cos(theta)


def plane_under(payload, x_norm, y_norm):
    """The payload plane under a benchmark-image coordinate, or None for sky / no plane."""
    w, h = payload.width, payload.height
    row = min(h - 1, max(0, int(y_norm * h)))
    idx = payload.indices[row * w + raw_column(x_norm, w)]
    if idx == SKY or idx >= len(payload.planes):
        return None
    return payload.planes[idx]


def intersect(plane, direction):
    """Distance along a unit ray to a plane (n . p = d), or None if parallel."""
    vx, vy, vz = direction
    denom = vx * plane.nx + vy * plane.ny + vz * plane.nz
    if denom == 0:
        return None
    return abs(plane.d / denom)


# ---------------------------------------------------------------------------
# the labeler's parser and archive

def load_depthlib(labeler_root):
    path = os.path.join(labeler_root, "depth.py")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: --labeler-root must be a sidewalk-auto-labeler "
                         f"checkout (its depth.py parses the payloads)")
    spec = importlib.util.spec_from_file_location("labeler_depth", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def labeler_commit(labeler_root):
    """The labeler checkout's HEAD commit -- only that, so the committed JSON does not depend on
    the machine (no path, no branch, no remote-tracking state)."""
    try:
        return subprocess.run(["git", "-C", labeler_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_index(depth_dir):
    path = os.path.join(depth_dir, "index.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["panorama_id"]: r for r in csv.DictReader(fh)}


def load_payload(depthlib, depth_dir, pano_id):
    """(payload, sha256 of the archived file) or (None, None) if not archived.

    Explicit path join, never a glob or a bare CLI token: pano ids can start with '-'.
    """
    path = os.path.join(depth_dir, pano_id + ".json.gz")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as fh:
        raw = fh.read()
    blob = json.loads(gzip.decompress(raw).decode("utf-8"))["depth_b64"]
    return depthlib.parse(blob), hashlib.sha256(raw).hexdigest()


def pano_geometry(depthlib, payload):
    """Status + the dominant ground plane (None unless MEASURED)."""
    ground = None if payload.degenerate else depthlib.ground_plane(payload)
    status = depthlib.classify_height(ground and ground.camera_height_m,
                                      ground and ground.tilt_deg,
                                      degenerate=payload.degenerate,
                                      exactly_level=ground and ground.exactly_level)
    return status, (ground if status == depthlib.MEASURED else None)


def depth_ranges(payload, camera_height_m, x, y):
    """(horizontal range, ray distance, source) at a benchmark-image point, per the docstring rule.

    Plane lookup by ``plane_under`` (image column = raw column), exact-ray intersection
    along ``image_ray``. If the plane under the pixel is not ground-like, or the pixel is
    sky, the range is ``flat_range`` over level ground at ``camera_height_m`` (the pano's
    measured ground-plane distance; that plane's tilt is not applied) and the source says
    ``fallback_wall`` / ``fallback_sky`` (``_none`` if the point is at or above the horizon).
    """
    plane = plane_under(payload, x, y)
    elev = (0.5 - y) * math.pi
    if plane is not None:
        tilt = math.degrees(math.acos(min(1.0, abs(plane.nz))))
        if tilt <= GROUND_MAX_TILT_DEG:
            ray = intersect(plane, image_ray(x, y))
            if ray is not None:
                return ray * math.cos(elev), ray, "pixel_plane"
        source = "fallback_wall"
    else:
        source = "fallback_sky"
    rng = flat_range(y, camera_height_m)
    if rng is None:
        return None, None, source + "_none"
    return rng, math.hypot(rng, camera_height_m), source


# ---------------------------------------------------------------------------
# benchmark side

def load_bundle(city):
    d = os.path.join(REPO, "benchmark", city)
    records = {}
    with open(os.path.join(d, "records.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    with open(os.path.join(d, "verdicts.json"), encoding="utf-8") as fh:
        verdicts = json.load(fh)["panos"]
    return records, verdicts


def _d2(p, q):
    return math.hypot((p[0] - q[0]) * PANO_SCALE_X, (p[1] - q[1]) * PANO_SCALE_Y)


def match(preds, gt):
    """Greedy, confidence-ordered, nearest unclaimed GT within R (depth_extract_da3.py).

    Returns (hit GT indices, per-prediction kind TP/FP/IGN).
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    claimed, hit, kinds = [False] * len(gt.gt_points), set(), [None] * len(preds)
    for i in order:
        p = _xy(preds[i])
        bk, bd = -1, R
        for k, g in enumerate(gt.gt_points):
            if claimed[k]:
                continue
            d = _d2(p, g)
            if d < bd:
                bd, bk = d, k
        if bk >= 0:
            claimed[bk] = True
            hit.add(bk)
            kinds[i] = "TP"
        else:
            kinds[i] = "IGN" if any(_d2(p, q) < R for q in gt.ignore_points) else "FP"
    return hit, kinds


def derive(labeler_root, splits):
    depthlib = load_depthlib(labeler_root)
    if depthlib.GROUND_MAX_TILT_DEG != GROUND_MAX_TILT_DEG or depthlib.SKY != SKY:
        raise SystemExit("the labeler's ground-tilt / sky constants changed; update this script's copies")
    points, dets, panos, index_sha = [], [], [], {}
    for city in list(splits) + list(FLAT_ONLY_SPLITS):
        records, verdicts = load_bundle(city)
        depth_dir = os.path.join(labeler_root, "runs", city, "depth")
        index = read_index(depth_dir) if city in splits else {}
        index_sha[city] = sha256_of(os.path.join(depth_dir, "index.csv")) if index else None
        for pid in sorted(verdicts):
            entry = verdicts[pid]
            rec = records[pid]
            gt = build_ground_truth(rec["detections"], entry["dets"], entry["missed"],
                                    entry["no_missed"])
            preds = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                     for d in rec["detections"]]
            hit, kinds = match(preds, gt)
            status, ground, sha, payload = "no_archive", None, None, None
            if city in splits:
                payload, sha = load_payload(depthlib, depth_dir, pid)
                if payload is None:
                    status = "not_archived"
                else:
                    status, ground = pano_geometry(depthlib, payload)
            pano_row = {"city": city, "pano": pid, "camera_height_status": status,
                        "camera_height_m": round(ground.camera_height_m, ND) if ground else None,
                        "ground_tilt_deg": round(ground.tilt_deg, ND) if ground else None,
                        "n_planes": payload.n_planes if payload else None,
                        "sha256": sha,
                        "sha256_matches_index": (index[pid]["sha256"] == sha) if (sha and pid in index) else None,
                        "capture_date": rec["pano"].get("capture_date"),
                        "fn_confirmed": gt.fn_confirmed, "n_gt": len(gt.gt_points)}
            panos.append(pano_row)

            def geom(x, y):
                row = {"flat_2p5": _r(flat_range(y, CAM_H)), "flat_2p6": _r(flat_range(y, CAM_H_LABELER)),
                       "flat_h": None, "depth_range": None, "depth_ray": None, "depth_source": None,
                       "apparent_px_flat": None, "apparent_px_depth": None,
                       "depth_range_scaled": None, "depth_ray_scaled": None,
                       "apparent_px_depth_scaled": None}
                if row["flat_2p5"] is not None:
                    ray_flat = math.hypot(row["flat_2p5"], CAM_H)
                    row["apparent_px_flat"] = _r(apparent_px(ray_flat))
                if ground is not None:
                    row["flat_h"] = _r(flat_range(y, ground.camera_height_m))
                    rng, ray, src = depth_ranges(payload, ground.camera_height_m, x, y)
                    row["depth_range"], row["depth_ray"], row["depth_source"] = _r(rng), _r(ray), src
                    if ray:
                        row["apparent_px_depth"] = _r(apparent_px(ray))
                        k = DEPTH_FRAME_SCALE[city]
                        row["depth_range_scaled"], row["depth_ray_scaled"] = _r(rng * k), _r(ray * k)
                        row["apparent_px_depth_scaled"] = _r(apparent_px(ray * k))
                return row

            if gt.fn_confirmed and gt.gt_points:
                for k, g in enumerate(gt.gt_points):
                    # x / y unrounded: detections sit on exact binary fractions (e.g. 138/256),
                    # and rounding them can move a point across a payload row (review N2)
                    points.append({"city": city, "pano": pid, "x": g[0], "y": g[1],
                                   "hit": k in hit, "camera_height_status": status, **geom(g[0], g[1])})
            for i, (x, y, conf) in enumerate(preds):
                dets.append({"city": city, "pano": pid, "x": x, "y": y,
                             "confidence": round(conf, 6), "kind": kinds[i],
                             "camera_height_status": status, **geom(x, y)})
    return {"labeler_commit": labeler_commit(labeler_root),
            "index_sha256": index_sha,
            "constants": {"depth_frame_scale": DEPTH_FRAME_SCALE, "cam_h": CAM_H, "cam_h_labeler": CAM_H_LABELER, "ramp_w": RAMP_W,
                          "radius_px": round(R, ND), "operating_point": 0.55,
                          "depth_splits": list(splits), "flat_only_splits": list(FLAT_ONLY_SPLITS)},
            "panos": panos, "points": points, "detections": dets}


def _r(v):
    return None if v is None else round(v, ND)


# ---------------------------------------------------------------------------
# tables from rows (no payloads)

def tables(data):
    pts, dets, panos = data["points"], data["detections"], data["panos"]
    splits = data["constants"]["depth_splits"]
    measured = [p for p in pts if p["camera_height_status"] == "measured"]
    t = {"inventory": [], "issue_check": [], "deflation": {}, "deflation_scaled": {}, "thresholds": {},
         "recall_distance": {}, "recall_size": {}, "forecast": {}, "precision_distance": {},
         "excluded": {}, "depth_source": {}, "published_reproduction": {}, "by_capture_year": []}
    year = {(p["city"], p["pano"]): (p.get("capture_date") or "")[:4] for p in panos}

    for city in splits + list(FLAT_ONLY_SPLITS):
        cp = [p for p in panos if p["city"] == city]
        counts = {}
        for p in cp:
            counts[p["camera_height_status"]] = counts.get(p["camera_height_status"], 0) + 1
        heights = sorted(p["camera_height_m"] for p in cp if p["camera_height_m"] is not None)
        t["inventory"].append({
            "city": city, "panos": len(cp), "status": dict(sorted(counts.items())),
            "sha256_verified": sum(1 for p in cp if p["sha256_matches_index"]),
            "sha256_mismatch": sum(1 for p in cp if p["sha256_matches_index"] is False),
            "index_sha256": data.get("index_sha256", {}).get(city),
            "camera_height_median_m": round(st.median(heights), ND) if heights else None,
            "camera_height_min_m": heights[0] if heights else None,
            "camera_height_max_m": heights[-1] if heights else None})

    # the issue's check: operational detections, flat @ 2.6 vs depth, measured panos only
    for city in splits:
        cd = [d for d in dets if d["city"] == city and d["camera_height_status"] == "measured"
              and d["kind"] != "IGN"]
        ratio = deflation(cd, "flat_2p6", "depth_range")
        if ratio is None:
            continue
        pairs = [(d["flat_h"], d["depth_range"]) for d in cd
                 if d.get("flat_h") is not None and d.get("depth_range")]
        t["issue_check"].append({
            "city": city, "n_detections": ratio["n"],
            "n_panos": len({d["pano"] for d in cd}),
            "median_flat_2p6": ratio["median_flat"], "median_depth_range": ratio["median_depth"],
            "median_depth_ray": round(st.median(d["depth_ray"] for d in cd if d.get("depth_ray")), ND),
            "ratio_of_medians": ratio["ratio_of_medians"], "median_ratio": ratio["median_ratio"],
            "ratio_after_height_only": round(st.median(f / d for f, d in pairs), ND) if pairs else None})

    # GT points: deflation, thresholds, tables, per split + pooled GSV
    groups = {c: [p for p in measured if p["city"] == c] for c in splits}
    groups["gsv_pooled"] = list(measured)
    for name, g in groups.items():
        dfl = deflation(g)
        t["deflation"][name] = dfl
        if dfl:
            t["thresholds"][name] = {
                "published_m": list(PUBLISHED_THRESHOLDS_M),
                "median_point_m": deflated_thresholds(dfl["median_ratio"]),
                "window": [window_threshold(g, th) for th in PUBLISHED_THRESHOLDS_M],
                "window_scaled": [window_threshold(g, th, depth_key="depth_range_scaled")
                                  for th in PUBLISHED_THRESHOLDS_M]}
        t["recall_distance"][name] = {"flat_2p5": recall_table(g, "flat_2p5", M_BUCKETS, "m"),
                                      "depth": recall_table(g, "depth_range", M_BUCKETS, "m"),
                                      "depth_scaled": recall_table(g, "depth_range_scaled", M_BUCKETS, "m")}
        t["recall_size"][name] = {"flat_2p5": recall_table(g, "apparent_px_flat", PX_BUCKETS, "px"),
                                  "depth": recall_table(g, "apparent_px_depth", PX_BUCKETS, "px"),
                                  "depth_scaled": recall_table(g, "apparent_px_depth_scaled", PX_BUCKETS, "px")}
        t["forecast"][name] = {"flat_2p5": resolution_forecast(g, "apparent_px_flat"),
                               "depth": resolution_forecast(g, "apparent_px_depth"),
                               "depth_scaled": resolution_forecast(g, "apparent_px_depth_scaled")}
        t["deflation_scaled"][name] = deflation(g, "flat_2p5", "depth_range_scaled")
        if t["deflation_scaled"][name]:
            t["thresholds"][name]["median_point_scaled_m"] = deflated_thresholds(
                t["deflation_scaled"][name]["median_ratio"])
        src = {}
        for p in g:
            src[p["depth_source"]] = src.get(p["depth_source"], 0) + 1
        t["depth_source"][name] = dict(sorted(src.items(), key=lambda kv: str(kv[0])))
        cd = [d for d in dets if d["camera_height_status"] == "measured"
              and (name == "gsv_pooled" or d["city"] == name)]
        t["precision_distance"][name] = {"flat_2p5": precision_table(cd, "flat_2p5", M_BUCKETS, "m"),
                                         "depth": precision_table(cd, "depth_range", M_BUCKETS, "m"),
                                         "depth_scaled": precision_table(cd, "depth_range_scaled", M_BUCKETS, "m")}
        # what the exclusion rule left out, on the flat axis, so a reader can see it is not
        # a biased subset
        ex = [p for p in pts if p["camera_height_status"] != "measured"
              and (name == "gsv_pooled" or p["city"] == name)]
        if name == "gsv_pooled":
            ex = [p for p in ex if p["city"] in splits]
        t["excluded"][name] = {"n": len(ex), "hit": sum(1 for p in ex if p["hit"]),
                               "recall_flat_2p5": round(sum(1 for p in ex if p["hit"]) / len(ex), ND) if ex else None,
                               "included_recall_flat_2p5": recall_table(g, "flat_2p5", M_BUCKETS, "m")[-1]["recall"]}

    # by capture year: the rig, reported apart from the split (review S1). A split mixes
    # vintages; Google's 2025-26 rig is paterson 2025 + gainesville 2026.
    def rig_row(label, g):
        keys = {(q["city"], q["pano"]) for q in g}
        heights = sorted(p["camera_height_m"] for p in panos if (p["city"], p["pano"]) in keys)
        dfl = deflation(g)
        d26 = deflation(g, "flat_2p6", "depth_range")
        return {"group": label, "n": len(g), "n_panos": len(heights),
                "camera_height_median_m": round(st.median(heights), ND) if heights else None,
                "median_ratio": dfl and dfl["median_ratio"],
                "p10_ratio": dfl and dfl["p10_ratio"], "p90_ratio": dfl and dfl["p90_ratio"],
                "median_ratio_flat_2p6": d26 and d26["median_ratio"],
                "window": [window_threshold(g, th) for th in PUBLISHED_THRESHOLDS_M]}

    for city in splits:
        cm = [p for p in measured if p["city"] == city]
        for y in sorted({year[(p["city"], p["pano"])] for p in cm}):
            g = [p for p in cm if year[(p["city"], p["pano"])] == y]
            if len(g) >= MIN_YEAR_N:
                t["by_capture_year"].append(rig_row(f"{city} {y}", g))
    t["by_capture_year"].append(rig_row(
        "2025-26 rig (" + " + ".join(f"{c} {y}" for c, y in NEW_RIG) + ")",
        [p for p in measured if (p["city"], year[(p["city"], p["pano"])]) in NEW_RIG]))
    t["by_capture_year"].append(rig_row(
        "older US vintages (bend, paterson, gainesville; the rest)",
        [p for p in measured if p["city"] != "sao_paulo"
         and (p["city"], year[(p["city"], p["pano"])]) not in NEW_RIG]))

    # the doc's population, richmond + bend, all panos, flat axis: must reproduce 637 / 0.765
    doc = [p for p in pts if p["city"] in ("richmond", "bend")]
    t["published_reproduction"] = {
        "n": len(doc), "hit": sum(1 for p in doc if p["hit"]),
        "recall": round(sum(1 for p in doc if p["hit"]) / len(doc), ND),
        "per_city": {c: {"n": sum(1 for p in doc if p["city"] == c),
                         "hit": sum(1 for p in doc if p["city"] == c and p["hit"])}
                     for c in ("richmond", "bend")},
        "flat_2p5_richmond_bend": recall_table(doc, "flat_2p5", M_BUCKETS, "m"),
        "flat_2p5_richmond": recall_table([p for p in doc if p["city"] == "richmond"], "flat_2p5", M_BUCKETS, "m"),
        "size_flat_richmond_bend": recall_table(doc, "apparent_px_flat", PX_BUCKETS, "px")}
    return t


# ---------------------------------------------------------------------------
# markdown

def _fmt(v):
    return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _side_by_side(columns, key="recall"):
    """columns: [(label, rows)] -> one table, buckets in first-seen order."""
    order, seen = [], set()
    for _, rows in columns:
        for r in rows:
            if r["bucket"] not in seen:
                seen.add(r["bucket"])
                order.append(r["bucket"])
    lines = ["| bucket | " + " | ".join(f"n ({lab}) | {key} ({lab})" for lab, _ in columns) + " |",
             "|---|" + "---:|---:|" * len(columns)]
    maps = [{r["bucket"]: r for r in rows} for _, rows in columns]
    for b in order:
        cells = []
        for m in maps:
            r = m.get(b)
            cells += [_fmt(r and r["n"]), _fmt(r and r[key])]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


DOC_TABLE_NOTE = ("Thresholds: 18 m and 25 m on the flat axis divided by the median flat/depth ratio of "
                  f"the points whose flat distance is within +/-{THRESHOLD_WINDOW:.0%} of the threshold "
                  "(window n in brackets); the median-point ratio is the whole population's.\n")


def _n(v):
    return f"{v:,}"


def _thr(w):
    return "–" if w["deflated_m"] is None else f"{w['deflated_m']} m ({w['n']})"


def _band(b):
    return b if b == "all" else b.replace("-", "–")


def doc_tables(data, t):
    """The tables of docs/detection_recall_analysis.md §0, verbatim.

    The doc is pasted from these (``--doc-tables``), and tests/test_recall_by_depth_112.py
    checks that every one appears in the doc unchanged.
    """
    out = {}
    splits = data["constants"]["depth_splits"]
    L = ["| split | panos | measured | stand-in ground | degenerate / implausible | camera height median (min–max), measured |",
         "|---|---:|---:|---:|---:|---|"]
    for r in t["inventory"]:
        if r["city"] not in splits:
            continue
        s_ = r["status"]
        L.append(f"| {r['city']} | {r['panos']} | {s_.get('measured', 0)} | {s_.get('synthetic_ground', 0)} | "
                 f"{s_.get('degenerate', 0)} / {s_.get('implausible', 0)} | "
                 f"{r['camera_height_median_m']:.2f} m ({r['camera_height_min_m']:.2f}–{r['camera_height_max_m']:.2f}) |")
    out["inventory"] = "\n".join(L)

    label = {"bend": "**bend** (this document's GSV city)", "gsv_pooled": "GSV pooled"}
    L = ["| population | n | median flat / median depth | median-point ratio (p10–p90) | 18 m becomes (window n) | 25 m becomes (window n) | depth × scale: 18 m / 25 m become |",
         "|---|---:|---:|---|---|---|---|"]
    for name, d in t["deflation"].items():
        if not d:
            continue
        th = t["thresholds"][name]
        ws = th["window_scaled"]
        L.append(f"| {label.get(name, name)} | {_n(d['n'])} | {d['median_flat']:.2f} / {d['median_depth']:.2f} = "
                 f"{d['ratio_of_medians']:.2f} | {d['median_ratio']:.3f} ({d['p10_ratio']:.2f}–{d['p90_ratio']:.2f}) | "
                 f"{_thr(th['window'][0])} | {_thr(th['window'][1])} | "
                 f"{ws[0]['deflated_m']} m / {ws[1]['deflated_m']} m |")
    out["deflation"] = "\n".join(L)

    L = ["| capture vintage | GT points (panos) | camera height, median | flat 2.5 m / depth, median point (p10–p90) | flat 2.6 m / depth | 18 m becomes (window n) | 25 m becomes (window n) |",
         "|---|---:|---:|---|---:|---|---|"]
    for r in t["by_capture_year"]:
        L.append(f"| {r['group']} | {r['n']} ({r['n_panos']}) | {r['camera_height_median_m']:.2f} m | "
                 f"{r['median_ratio']:.3f} ({r['p10_ratio']:.2f}–{r['p90_ratio']:.2f}) | {r['median_ratio_flat_2p6']:.2f} | "
                 f"{_thr(r['window'][0])} | {_thr(r['window'][1])} |")
    out["by_capture_year"] = "\n".join(L)

    heads = {"flat_2p5": ("flat 2.5 m", "recall (flat)"), "depth": ("depth", "recall (depth)"),
             "depth_scaled": ("depth × scale", "recall")}

    def side(name, keys, which, first):
        tabs = [t[which][name][k] for k in keys]
        h = [first]
        for k in keys:
            h += [f"n ({heads[k][0]})", heads[k][1]]
        L = ["| " + " | ".join(h) + " |", "|---|" + "---:|---:|" * len(keys)]
        order = []
        for tb in tabs:
            order += [r["bucket"] for r in tb if r["bucket"] not in order and r["bucket"] != "all"]
        order.append("all")
        maps = [{r["bucket"]: r for r in tb} for tb in tabs]
        for b in order:
            cells = []
            for m in maps:
                r = m.get(b)
                cells += ([_n(r["n"]), f"{r['recall']:.3f}"] if r else ["–", "–"])
            L.append(f"| {_band(b)} | " + " | ".join(cells) + " |")
        return "\n".join(L)

    out["bend_distance"] = side("bend", ("flat_2p5", "depth", "depth_scaled"), "recall_distance", "distance")
    out["pooled_distance"] = side("gsv_pooled", ("flat_2p5", "depth"), "recall_distance", "distance")
    out["bend_size"] = side("bend", ("flat_2p5", "depth"), "recall_size", "apparent size")
    out["pooled_size"] = side("gsv_pooled", ("flat_2p5", "depth"), "recall_size", "apparent size")

    L = ["| factor | bend, flat | bend, depth | bend, depth × scale | pooled, flat | pooled, depth |",
         "|---|---|---|---|---|---|"]
    fc = t["forecast"]
    for i, f in enumerate(fc["bend"]["flat_2p5"]):
        cells = [fc["bend"]["flat_2p5"][i], fc["bend"]["depth"][i], fc["bend"]["depth_scaled"][i],
                 fc["gsv_pooled"]["flat_2p5"][i], fc["gsv_pooled"]["depth"][i]]
        L.append(f"| {f['factor']:g}× | " + " | ".join(f"{c['gain']:+.3f}" for c in cells) + " |")
    out["forecast"] = "\n".join(L)

    L = ["| distance (depth) | detections (TP + FP) | precision |", "|---|---:|---:|"]
    for r in t["precision_distance"]["gsv_pooled"]["depth"]:
        L.append(f"| {_band(r['bucket'])} | {_n(r['n'])} | {r['precision']:.3f} |")
    out["pooled_precision"] = "\n".join(L)
    return out


def markdown(data, t):
    L = []
    L.append("# Recall by distance on GSV depth vs flat-ground geometry (#112)\n")
    lc = data["labeler_commit"]
    L.append(f"Generated by `scripts/analysis/recall_by_depth_112.py`; labeler parser at "
             f"`{lc}`. Image column = raw payload column (see the script docstring). "
             f"Depth axis = horizontal range along the exact ray to the "
             f"plane under the point's pixel; flat axis = {CAM_H} m / tan(depression). "
             f"Measured-ground panoramas only on the depth axis.\n")
    L.append("## Payload inventory\n")
    L.append("| split | panos | status counts | sha256 verified | index.csv sha256 | camera height median / min / max (m) |")
    L.append("|---|---:|---|---:|---|---|")
    for r in t["inventory"]:
        L.append(f"| {r['city']} | {r['panos']} | {r['status']} | {r['sha256_verified']} | {r['index_sha256'] or ''} | "
                 f"{_fmt(r['camera_height_median_m'])} / {_fmt(r['camera_height_min_m'])} / {_fmt(r['camera_height_max_m'])} |")
    L.append("\n## The issue's check: operational detections, flat @ 2.6 m vs depth\n")
    L.append("| split | detections | panos | median flat @2.6 | median depth range | median ray | ratio of medians | median per-det ratio | ratio after height only |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in t["issue_check"]:
        L.append(f"| {r['city']} | {r['n_detections']} | {r['n_panos']} | {_fmt(r['median_flat_2p6'])} | "
                 f"{_fmt(r['median_depth_range'])} | {_fmt(r['median_depth_ray'])} | {_fmt(r['ratio_of_medians'])} | "
                 f"{_fmt(r['median_ratio'])} | {_fmt(r['ratio_after_height_only'])} |")
    dt = doc_tables(data, t)
    L.append("\n## Deflation of the flat axis at the GT points, and the published thresholds\n")
    L.append(DOC_TABLE_NOTE)
    L.append(dt["deflation"])
    L.append("\n## By capture year (the rig), not by split\n")
    L.append(dt["by_capture_year"])
    pr = t["published_reproduction"]
    L.append(f"\n## The doc's population reproduces: richmond + bend n = {pr['n']}, hit {pr['hit']}, "
             f"recall {pr['recall']} (per city {pr['per_city']})\n")
    cols = (("flat 2.5 m", "flat_2p5"), ("depth", "depth"), ("depth x scale", "depth_scaled"))
    for name in t["recall_distance"]:
        L.append(f"\n## {name}: recall by distance\n")
        L.append(_side_by_side([(lab, t["recall_distance"][name][k]) for lab, k in cols]))
        L.append(f"\n## {name}: recall by apparent size\n")
        L.append(_side_by_side([(lab, t["recall_size"][name][k]) for lab, k in cols]))
        L.append(f"\n## {name}: resolution forecast\n")
        L.append("| factor | " + " | ".join(f"{lab}: base -> forecast (gain)" for lab, _ in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for i in range(len(t["forecast"][name]["flat_2p5"])):
            cells = []
            for _, k in cols:
                f = t["forecast"][name][k][i]
                cells.append(f"{f['base']} -> {f['forecast']} ({f['gain']:+.3f})")
            L.append(f"| {t['forecast'][name]['flat_2p5'][i]['factor']}x | " + " | ".join(cells) + " |")
        L.append(f"\n## {name}: precision by distance (measured panos, TP+FP)\n")
        L.append(_side_by_side([(lab, t["precision_distance"][name][k]) for lab, k in cols], "precision"))
        ex = t["excluded"][name]
        L.append(f"\nExcluded from the depth axis (non-measured ground): {ex['n']} GT points, recall on the flat "
                 f"axis {_fmt(ex['recall_flat_2p5'])} vs {_fmt(ex['included_recall_flat_2p5'])} for the included. "
                 f"Depth source of the included points: {t['depth_source'][name]}.\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------

def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler-root", default=os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler"),
                    help="sidewalk-auto-labeler checkout holding depth.py and runs/<city>/depth")
    ap.add_argument("--splits", nargs="+", default=list(DEPTH_SPLITS))
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--md", default=OUT_MD)
    ap.add_argument("--check", action="store_true",
                    help="re-derive the tables from the committed rows (no payloads) and fail on drift")
    ap.add_argument("--markdown", action="store_true", help="print the markdown tables")
    ap.add_argument("--doc-tables", action="store_true",
                    help="with --check: print the §0 tables of docs/detection_recall_analysis.md")
    a = ap.parse_args(argv)

    if a.check:
        with open(a.out, encoding="utf-8") as fh:
            data = json.load(fh)
        fresh = tables(data)
        if fresh != data["tables"]:
            raise SystemExit(f"{a.out}: tables do not re-derive from the committed rows")
        print(f"{a.out}: {len(data['points'])} GT points, {len(data['detections'])} detections; "
              f"tables re-derive from the rows")
        if a.markdown:
            print(markdown(data, fresh))
        if a.doc_tables:
            for name, tab in doc_tables(data, fresh).items():
                print(f"<!-- {name} -->\n{tab}\n")
        return 0

    data = derive(a.labeler_root, a.splits)
    data["tables"] = tables(data)
    write_json(a.out, data)
    md = markdown(data, data["tables"])
    with open(a.md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    print(f"wrote {a.out} ({len(data['points'])} GT points, {len(data['detections'])} detections) and {a.md}")
    if a.markdown:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
