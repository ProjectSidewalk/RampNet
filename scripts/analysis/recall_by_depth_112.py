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
    plane under its pixel (``depth.ground_range_at``): plane lookup is per pixel of the
    512x256 index, the intersection is continuous. If that plane is not ground-like (tilt
    > ``GROUND_MAX_TILT_DEG`` -- a wall, a car) or the pixel is sky, the point falls back
    to the dominant ground plane intersected along the same ray (a ramp is on the ground),
    and the row says so (``depth_source``). Euclidean ray distance is kept beside it and is
    what apparent size uses (a subtended angle depends on the ray, not the horizontal).
  * A panorama whose ground is a stand-in (Google's exactly-level 2.500 m plane,
    ``depth.SYNTHETIC_GROUND``), a fallback reconstruction (``DEGENERATE``), or has no
    usable floor is **excluded from the depth axis**, never silently backfilled: its rows
    carry ``camera_height_status`` and no depth range. The doc reports the count per split.

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
ND = 4                 # decimals in every committed float

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
    median of flat/depth, which is the factor to deflate a threshold by (it is not
    dominated by the far tail). Both are reported because they differ.
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
    """The labeler checkout's HEAD and branch, so the doc can name the parser revision used."""
    def git(*args):
        try:
            return subprocess.run(["git", "-C", labeler_root, *args],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "origin_main": git("rev-parse", "origin/main")}


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


def depth_ranges(depthlib, payload, ground, x, y):
    """(horizontal range, ray distance, source) at a normalized point, per the docstring rule.

    The plane lookup and the exact-ray intersection are the library's (``_plane_at`` is
    what ``ray_depth_at`` / ``ground_range_at`` are built on); this only adds the "is that
    plane ground-like" test that decides the fallback.
    """
    plane, _, _ = depthlib._plane_at(payload, x, y)
    theta = (0.5 - y) * math.pi
    if plane is not None:
        tilt = math.degrees(math.acos(min(1.0, abs(plane.nz))))
        if tilt <= depthlib.GROUND_MAX_TILT_DEG:
            ray = depthlib.ray_depth_at(payload, x, y)
            if ray is not None:
                return ray * math.cos(theta), ray, "pixel_plane"
        source = "fallback_wall"
    else:
        source = "fallback_sky"
    # not a ground surface under the pixel: the measured camera height over level ground
    rng = flat_range(y, ground.camera_height_m)
    if rng is None:
        return None, None, source + "_none"
    return rng, math.hypot(rng, ground.camera_height_m), source


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
                    rng, ray, src = depth_ranges(depthlib, payload, ground, x, y)
                    row["depth_range"], row["depth_ray"], row["depth_source"] = _r(rng), _r(ray), src
                    if ray:
                        row["apparent_px_depth"] = _r(apparent_px(ray))
                        k = DEPTH_FRAME_SCALE[city]
                        row["depth_range_scaled"], row["depth_ray_scaled"] = _r(rng * k), _r(ray * k)
                        row["apparent_px_depth_scaled"] = _r(apparent_px(ray * k))
                return row

            if gt.fn_confirmed and gt.gt_points:
                for k, g in enumerate(gt.gt_points):
                    points.append({"city": city, "pano": pid, "x": round(g[0], 6), "y": round(g[1], 6),
                                   "hit": k in hit, "camera_height_status": status, **geom(g[0], g[1])})
            for i, (x, y, conf) in enumerate(preds):
                dets.append({"city": city, "pano": pid, "x": round(x, 6), "y": round(y, 6),
                             "confidence": round(conf, 6), "kind": kinds[i],
                             "camera_height_status": status, **geom(x, y)})
    return {"labeler_root": os.path.abspath(labeler_root).replace("\\", "/"),
            "labeler_commit": labeler_commit(labeler_root),
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
         "excluded": {}, "depth_source": {}, "published_reproduction": {}}

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
            t["thresholds"][name] = {"published_m": list(PUBLISHED_THRESHOLDS_M),
                                     "deflated_m": deflated_thresholds(dfl["median_ratio"])}
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
            t["thresholds"][name]["deflated_scaled_m"] = deflated_thresholds(t["deflation_scaled"][name]["median_ratio"])
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


def markdown(data, t):
    L = []
    L.append("# Recall by distance on GSV depth vs flat-ground geometry (#112)\n")
    lc = data["labeler_commit"]
    L.append(f"Generated by `scripts/analysis/recall_by_depth_112.py`; labeler parser at "
             f"`{lc['head']}` (branch `{lc['branch']}`; origin/main was `{lc['origin_main']}`). Depth axis = horizontal range along the exact ray to the "
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
    L.append("\n## Deflation of the flat axis at the GT points, and the published thresholds\n")
    L.append("| population | n | median flat @2.5 | median depth | ratio of medians | median ratio (p10-p90) | 18 m / 25 m become | with the depth-frame scale: ratio, thresholds |")
    L.append("|---|---:|---:|---:|---:|---|---|---|")
    for name, d in t["deflation"].items():
        if not d:
            continue
        th = t["thresholds"][name]
        ds = t["deflation_scaled"].get(name)
        sc = th.get("deflated_scaled_m", ["", ""])
        L.append(f"| {name} | {d['n']} | {_fmt(d['median_flat'])} | {_fmt(d['median_depth'])} | "
                 f"{_fmt(d['ratio_of_medians'])} | {_fmt(d['median_ratio'])} ({_fmt(d['p10_ratio'])}-{_fmt(d['p90_ratio'])}) | "
                 f"{th['deflated_m'][0]} m / {th['deflated_m'][1]} m | "
                 f"{_fmt(ds and ds['median_ratio'])}, {sc[0]} m / {sc[1]} m |")
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
