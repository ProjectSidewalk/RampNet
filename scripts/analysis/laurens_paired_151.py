"""The Laurens rig effect as a paired measurement: the corners both rigs saw (#151).

Laurens is the one benchmark city with two imagery arms over one footprint: ``laurens_gsv``
(Google, September 2024, 16384 px) and ``laurens_mapillary`` (GoPro Max, November 2025,
5760 px). The 2026-09-03 comparison on #151 read the rig effect off the two whole arms, which
are different panorama sets with different ground truth. This script restricts both arms to the
corners both rigs saw and measures the effect there, per pano pair, with intervals.

What it builds, all from committed inputs (the bundles, ``benchmark/model_detections/``,
``analysis_out/op_cache/``, ``analysis_out/input_res_sweep_25/cache/r2048/`` and the camera
heights in ``analysis_out/recall_by_depth_112.json``):

1. **Pairs.** Every ``laurens_gsv`` bundle pano is paired with the nearest ``laurens_mapillary``
   bundle pano within ``PAIR_RADIUS_M`` (20 m), one-to-one, greedy by ascending distance. Both
   bundles were de-clustered at 30 m, so no two panos of one arm share a corner and a pair is a
   corner (the minimum within-arm spacing is reported, not assumed).
2. **Physical ramps.** Each arm's GT points (``build_ground_truth`` over its verdict review) are
   placed on the ground with the labeler's flat-ground convention (``eval_sites.py`` /
   ``geo.detection_ground_point`` with ``apply_pose=False``: bearing = heading + (x - 0.5) * 360,
   range = h / tan(depression), dropped above ``MIN_DEPRESSION_RAD`` or beyond ``MAX_RANGE_M``),
   and within each pair the two arms' points are matched one-to-one by ascending distance within
   ``MATCH_RADIUS_M`` (the labeler's 5 m). Camera height: the GSV pano's measured depth-payload
   height where it has one, else the labeler's 2.6 m; Mapillary 2.6 m (the labeler's GoPro Max
   rig estimate did not pass its own gate). A rotation null and a height sensitivity are reported
   beside the count.
3. **Paired scores.** Every committed leg on both arms, scored with the benchmark scorer
   (``rampnet.detection_eval.score_pano`` / ``aggregate``) at the scoreboard's operating points,
   restricted to the paired panos; paired deltas (GSV minus Mapillary) with a pano-pair bootstrap
   (resample pairs, ``SEED``, ``DRAWS``). RampNet at 0.55 from the committed records. RampNet at
   0.30 exists only for Mapillary (``analysis_out/op_cache/laurens_mapillary.json``); there is no
   ``laurens_gsv`` op_cache, so that row has no GSV side and no delta. The #25 sweep's r2048
   re-extraction covers both arms at both thresholds and is reported as its own leg, because on
   ``laurens_gsv`` it does not reproduce the deployed records (it is the same model on the
   benchmark JPEGs, not the labeler's production run).
4. **The per-ramp 2x2** for RampNet on the matched ramps: hit on both arms, GSV only, Mapillary
   only, neither.
5. **The near-miss delta**: median normalized y of missed GT minus median y of detected GT
   (RampNet at 0.55, the verdict GT, unsure marks excluded), per arm, whole arm and paired subset.
6. **A curb-reveal probe** on the GSV arm (derive-time only; needs the depth payloads). Around
   each GT point on a measured-ground pano, the payload's ground-like planes in a window, and the
   vertical step between adjacent ground-like planes at their shared boundary; the same at three
   azimuth-shifted windows on the same image row as a null. Image column = raw payload column and
   the plane convention ``n . p + d = 0`` (+z down) are ``recall_by_depth_112.py``'s. The rows are
   committed so the tables re-derive; the payloads are an unpublished input.

Every table re-derives from the committed rows (``--check``), and every row except the curb probe
re-derives from committed inputs (``--check`` does both). Floats are rounded, the JSON is LF, and
``rows_sha256`` is the content hash of the row sections.

    # derive (the curb probe needs the labeler checkout with runs/laurens_gsv/depth)
    python scripts/analysis/laurens_paired_151.py --labeler-root D:/Git/sidewalk-auto-labeler

    # re-derive rows (minus the probe) from committed inputs and every table from the rows
    python scripts/analysis/laurens_paired_151.py --check

    # the markdown the doc's tables are pasted from
    python scripts/analysis/laurens_paired_151.py --check --markdown
"""
import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet import roster  # noqa: E402
from rampnet.detection_eval import aggregate, prediction_confidence, radius_sq_for, score_pano  # noqa: E402

OUT_JSON = os.path.join(REPO, "analysis_out", "laurens_paired_151.json")
OUT_MD = os.path.join(REPO, "analysis_out", "laurens_paired_151.md")
GSV, MLY = "laurens_gsv", "laurens_mapillary"
ARMS = (GSV, MLY)

PAIR_RADIUS_M = 20.0          # the plan's "corners both rigs saw"
MATCH_RADIUS_M = 5.0          # eval_sites.py --match-radius-m default
MAX_RANGE_M = 25.0            # geo.DEFAULT_MAX_RANGE_M
MIN_DEPRESSION_RAD = 0.02     # geo.MIN_DEPRESSION_RAD
METERS_PER_DEG_LAT = 111320.0  # geo.METERS_PER_DEG_LAT
CAM_H_DEFAULT = 2.6           # geo.DEFAULT_CAMERA_HEIGHT_M (unmeasured GSV, and Mapillary)
# sensitivity: the labeler's GoPro Max scale-identity height (runs/laurens/camera_heights.json,
# h_scale 2.949; its gate failed, so the labeler applies 2.6) and every camera at 2.6 m
MLY_CAM_H_SENS = 2.95
NULL_ROTATIONS_DEG = (90.0, 180.0, 270.0)
RADIUS_SWEEP_M = (2.0, 3.0, 4.0, 5.0)
TIGHT_RADIUS_M = 3.0
SEED = 151
DRAWS = 10000
NEAR_MISS_DRAWS = 2000   # medians are resampled in Python, so fewer draws
ND = 4
DEPLOYED = 0.55
RECOMMENDED = 0.30
YOLO_PANO = ("y11x_pano_h200", "y11l_pano", "y26_pano")
HEADLINE_RAMPNET_LEGS = ("rampnet@0.55", "rampnet_r2048@0.55", "rampnet_r2048@0.30")
R2048_DIR = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "cache", "r2048")
OP_CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
DEPTH_ROWS = os.path.join(REPO, "analysis_out", "recall_by_depth_112.json")

# curb probe window, in payload cells (512 x 256 grid: 0.70 degrees per cell both ways)
PROBE_HALF_COLS = 6
PROBE_HALF_ROWS = 3
PROBE_NULL_SHIFTS = (0.25, 0.5, 0.75)   # azimuth shifts of the null window, same image row
CURB_STEP_M = (0.05, 0.30)              # a step in this range is "curb-sized"
GROUND_MAX_TILT_DEG = 18.0              # recall_by_depth_112 / the labeler's depth.py
PROBE_RANGE_BANDS = ((0, 8), (8, 12), (12, 18), (18, 1e9))


def _r(v, nd=ND):
    return None if v is None else round(float(v), nd)


# ---------------------------------------------------------------------------
# geometry (pure, unit-tested)

def local_en(lat, lng, lat0, lng0):
    """East/north metres of (lat, lng) from an origin, the labeler's equirectangular form."""
    return ((lng - lng0) * METERS_PER_DEG_LAT * math.cos(math.radians(lat0)),
            (lat - lat0) * METERS_PER_DEG_LAT)


def ground_point(e0, n0, heading_deg, x, y, cam_h, max_range_m=MAX_RANGE_M):
    """(east, north, range) of an equirect point on flat ground, or None if unplaceable.

    ``geo.detection_ground_point`` with ``apply_pose=False``: the centre column is the camera
    heading, y = 0.5 the horizon. Example: x = 0.5, y = 0.75 at h = 2 m is 2 m along the heading.
    """
    dep = (y - 0.5) * math.pi
    if dep <= MIN_DEPRESSION_RAD:
        return None
    d = cam_h / math.tan(dep)
    if d > max_range_m:
        return None
    b = math.radians(heading_deg) + (x - 0.5) * 2 * math.pi
    return e0 + d * math.sin(b), n0 + d * math.cos(b), d


def greedy_pairs(items_a, items_b, radius, dist):
    """One-to-one greedy matching by ascending distance within ``radius``.

    Returns [(i, j, d)] sorted by (i). Ties break on (d, i, j), so the result is deterministic.
    """
    cand = []
    for i, a in enumerate(items_a):
        for j, b in enumerate(items_b):
            d = dist(a, b)
            if d <= radius:
                cand.append((d, i, j))
    cand.sort()
    used_a, used_b, out = set(), set(), []
    for d, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, d))
    return sorted(out)


def prf(tp, fp, tp_r, n_gt):
    """P, R, F1 from summed counts, exactly as ``aggregate`` computes them."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp_r / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def prf_arrays(c):
    """Vectorised ``prf`` over the last axis of a (..., 4) array of (tp, fp, tp_r, n_gt)."""
    tp, fp, tpr, ngt = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r = np.where(ngt > 0, tpr / np.maximum(ngt, 1), 0.0)
        f = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    return p, r, f


def bootstrap_weights(n, draws=DRAWS, seed=SEED):
    """(draws, n) multiplicities of a pair bootstrap: each row resamples n pairs with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    w = np.zeros((draws, n), dtype=np.int64)
    np.add.at(w, (np.repeat(np.arange(draws), n), idx.ravel()), 1)
    return w


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p on the discordant counts b, c (binomial, p = 1/2).

    Example: ``mcnemar_exact(23, 16)`` is 0.3368; ``mcnemar_exact(5, 5)`` is 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [_r(lo), _r(hi)]


# ---------------------------------------------------------------------------
# inputs

def load_arm(split):
    """(records {pid: record}, gts {pid: GroundTruth}) via the scoreboard's own loader."""
    import scoreboard as sb
    return sb.load_split(split)


def camera_heights_gsv():
    """{pano: measured depth-payload camera height} for laurens_gsv, from the committed rows."""
    with open(DEPTH_ROWS, encoding="utf-8") as fh:
        rows = json.load(fh)
    return {p["pano"]: p["camera_height_m"] for p in rows["panos"]
            if p["city"] == GSV and p["camera_height_status"] == "measured"}


def legs():
    """[(name, kind, {arm: {pid: preds}}, op, note)] for every scored leg.

    kind: rampnet / rampnet_r2048 / challenger. A leg missing an arm carries None there.
    """
    import scoreboard as sb
    from export_model_cache import load_detections
    out = []
    recs = {a: load_arm(a)[0] for a in ARMS}
    rn = {a: {pid: r["detections"] for pid, r in recs[a].items()} for a in ARMS}
    out.append(("rampnet@0.55", "rampnet", rn, DEPLOYED, "committed records.jsonl (deployed run)"))
    opc = {}
    for a in ARMS:
        path = os.path.join(OP_CACHE_DIR, f"{a}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                opc[a] = {p["pano"]: [tuple(t) for t in p["preds"]] for p in json.load(fh)["panos"]}
        else:
            opc[a] = None
    out.append(("rampnet@0.30", "rampnet", opc, RECOMMENDED,
                "analysis_out/op_cache (laurens_gsv has none)"))
    r2 = {}
    for a in ARMS:
        with open(os.path.join(R2048_DIR, f"{a}.json"), encoding="utf-8") as fh:
            r2[a] = {p["pano"]: [tuple(t) for t in p["preds"]] for p in json.load(fh)["panos"]}
    for op in (DEPLOYED, RECOMMENDED):
        out.append((f"rampnet_r2048@{op:.2f}", "rampnet_r2048", r2, op,
                    "#25 sweep r2048 re-extraction of the benchmark JPEGs"))
    for leg in roster.ROSTER:
        if leg.provider == "rampnet":
            continue
        name = roster.published_name(leg)
        preds = {a: load_detections(leg.label, a, publish_as=name) for a in ARMS}
        if all(v is None for v in preds.values()):
            continue
        op = sb.OPERATING_POINT[sb.class_of(leg)]
        out.append((name, "challenger", preds, op, sb.class_of(leg)))
    return out


def at_op(preds, op):
    """Predictions kept at an operating point: ``compare.rescore``'s rule (no score = kept)."""
    return [p for p in preds if prediction_confidence(p) is None or prediction_confidence(p) >= op]


def input_hashes():
    """sha256 of every input, on LF-normalized bytes (review M5: a CRLF checkout under
    core.autocrlf must not change a hash when no content changed). From
    recall_by_depth_112.json only the rows this script reads are hashed (laurens_gsv's
    measured-ground ``panos`` rows), so adding another split there does not move it. An
    op_cache is hashed for every arm that has one (review N8)."""
    paths = []
    for a in ARMS:
        for f in ("records.jsonl", "verdicts.json"):
            paths.append(os.path.join("benchmark", a, f))  # noqa: PERF401
        paths.append(os.path.join("analysis_out", "input_res_sweep_25", "cache", "r2048", f"{a}.json"))
        if os.path.exists(os.path.join(OP_CACHE_DIR, f"{a}.json")):
            paths.append(os.path.join("analysis_out", "op_cache", f"{a}.json"))
    for name in sorted(os.listdir(os.path.join(REPO, "benchmark", "model_detections"))):
        if name.endswith(f"__{GSV}.json") or name.endswith(f"__{MLY}.json"):
            paths.append(os.path.join("benchmark", "model_detections", name))
    out = {}
    for rel in paths:
        with open(os.path.join(REPO, rel), "rb") as fh:
            out[rel.replace(os.sep, "/")] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    with open(DEPTH_ROWS, encoding="utf-8") as fh:
        rows = [p for p in json.load(fh)["panos"] if p["city"] == GSV]
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out["analysis_out/recall_by_depth_112.json#panos[city=laurens_gsv]"] = hashlib.sha256(blob).hexdigest()
    return out


# ---------------------------------------------------------------------------
# derive rows (no payloads)

def derive_rows():
    radius_sq = radius_sq_for()
    from complementarity import matched_gt
    recs, gts = {}, {}
    for a in ARMS:
        recs[a], gts[a] = load_arm(a)
    pos = {a: {pid: (r["pano"]["lat"], r["pano"]["lng"], r["pano"]["camera_heading"])
               for pid, r in recs[a].items()} for a in ARMS}
    lat0 = st.mean(p[0] for a in ARMS for p in pos[a].values())
    lng0 = st.mean(p[1] for a in ARMS for p in pos[a].values())
    en = {a: {pid: local_en(la, ln, lat0, lng0) for pid, (la, ln, _) in pos[a].items()} for a in ARMS}

    # 1. pairs
    g_ids, m_ids = sorted(en[GSV]), sorted(en[MLY])
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
    pr = greedy_pairs([en[GSV][p] for p in g_ids], [en[MLY][p] for p in m_ids], PAIR_RADIUS_M, dist)
    pairs = [{"gsv": g_ids[i], "mly": m_ids[j], "dist_m": _r(d, 2)} for i, j, d in pr]

    def nn(a, pid):
        others = [q for q in en[a] if q != pid]
        return min(dist(en[a][pid], en[a][q]) for q in others)

    def nn_other(a, b, pid):
        return min(dist(en[a][pid], en[b][q]) for q in en[b])

    panos = [{"arm": a, "pano": pid, "e": _r(en[a][pid][0], 2), "n": _r(en[a][pid][1], 2),
              "heading": _r(pos[a][pid][2], 3), "nn_same_arm_m": _r(nn(a, pid), 2),
              "nn_other_arm_m": _r(nn_other(a, MLY if a == GSV else GSV, pid), 2),
              "capture_date": recs[a][pid]["pano"].get("capture_date"),
              "fn_confirmed": gts[a][pid].fn_confirmed, "n_gt": len(gts[a][pid].gt_points)}
             for a in ARMS for pid in sorted(en[a])]

    # 2. GT points in world space, with RampNet hits
    n_true = {}
    for a in ARMS:
        with open(os.path.join(REPO, "benchmark", a, "verdicts.json"), encoding="utf-8") as fh:
            vp = json.load(fh)["panos"]
        # build_ground_truth lists verdict-true detections first, then non-unsure missed marks
        n_true[a] = {pid: sum(1 for v in e["dets"] if v is True or v == "true") for pid, e in vp.items()}
    heights = camera_heights_gsv()
    all_legs = legs()
    # per-GT hits for every RampNet leg and the three YOLO pano arms (review B1: the per-ramp
    # 2x2 on ramps both reviews contain is the GT-completeness-robust comparison)
    rn_legs = [(name, preds, op) for name, kind, preds, op, _ in all_legs
               if kind != "challenger" or name in YOLO_PANO]
    gt_rows = []
    for a in ARMS:
        for pid in sorted(gts[a]):
            g = gts[a][pid]
            hits = {}
            for name, preds, op in rn_legs:
                if preds.get(a) is None:
                    continue
                hits[name] = matched_gt(at_op(preds[a].get(pid, []), op), g.gt_points, radius_sq)
            e0, n0 = en[a][pid]
            hd = pos[a][pid][2]
            h_used = heights.get(pid, CAM_H_DEFAULT) if a == GSV else CAM_H_DEFAULT
            for k, (x, y) in enumerate(g.gt_points):
                row = {"arm": a, "pano": pid, "k": k, "x": x, "y": y,
                       "source": "det" if k < n_true[a][pid] else "missed",
                       "fn_confirmed": g.fn_confirmed, "cam_h": _r(h_used),
                       "hit": {name: (k in hs) for name, hs in hits.items()}}
                for tag, h in (("", h_used), ("_h26", CAM_H_DEFAULT),
                               ("_hsens", h_used if a == GSV else MLY_CAM_H_SENS)):
                    gp = ground_point(e0, n0, hd, x, y, h)
                    row["e" + tag], row["n" + tag], row["range" + tag] = (
                        (None, None, None) if gp is None else (_r(gp[0], 3), _r(gp[1], 3), _r(gp[2], 3)))
                gt_rows.append(row)

    # 3. per-pano scores for every leg, every pano of each arm (the paired subset is a filter)
    scores = []
    for name, kind, preds, op, note in all_legs:
        for a in ARMS:
            if preds.get(a) is None:
                continue
            for pid in sorted(gts[a]):
                s = score_pano(at_op(preds[a].get(pid, []), op), gts[a][pid], radius_sq=radius_sq)
                scores.append({"leg": name, "arm": a, "pano": pid, "tp": s.tp, "fp": s.fp,
                               "ignored": s.ignored, "n_gt": s.n_gt, "fn_confirmed": s.fn_confirmed})
    leg_meta = [{"leg": name, "kind": kind, "op": op, "note": note,
                 "arms": [a for a in ARMS if preds.get(a) is not None]}
                for name, kind, preds, op, note in all_legs]
    return {"origin": {"lat": _r(lat0, 8), "lng": _r(lng0, 8)}, "pairs": pairs, "panos": panos,
            "gt": gt_rows, "scores": scores, "legs": leg_meta}


# ---------------------------------------------------------------------------
# the curb probe (derive time only: needs the payloads)

def _ground_like(pl):
    return math.degrees(math.acos(min(1.0, abs(pl.nz)))) <= GROUND_MAX_TILT_DEG


def plane_z(pl, x, y):
    """Height coordinate (+z down) of plane ``n . p + d = 0`` at horizontal (x, y)."""
    return -(pl.d + pl.nx * x + pl.ny * y) / pl.nz


def window_steps(payload, x, y, image_ray, intersect):
    """Distinct ground-like planes in the window around (x, y), and the vertical step at every
    boundary between two ground-like planes inside it (one per adjacent-cell pair)."""
    w, h = payload.width, payload.height
    c0, r0 = int((x % 1.0) * w), min(h - 1, int(y * h))
    idx = payload.indices
    planes, steps = set(), []
    n = len(payload.planes)
    for r in range(max(0, r0 - PROBE_HALF_ROWS), min(h, r0 + PROBE_HALF_ROWS + 1)):
        for dc in range(-PROBE_HALF_COLS, PROBE_HALF_COLS + 1):
            c = (c0 + dc) % w
            i = idx[r * w + c]
            if 0 < i < n and _ground_like(payload.planes[i]):
                planes.add(i)
            # right neighbour (edge at column c+1) and lower neighbour (edge at row r+1)
            for rb, cb, xe, ye in ((r, (c + 1) % w, ((c + 1) % w) / w, (r + 0.5) / h),
                                   (r + 1, c, (c + 0.5) / w, (r + 1) / h)):
                if rb >= h or rb > r0 + PROBE_HALF_ROWS:
                    continue
                if rb == r and dc == PROBE_HALF_COLS:   # the right neighbour is outside the window
                    continue
                j = idx[rb * w + cb]
                if i == j or not (0 < i < n and 0 < j < n):
                    continue
                A, B = payload.planes[i], payload.planes[j]
                if not (_ground_like(A) and _ground_like(B)) or abs(B.nz) < 1e-6:
                    continue
                v = image_ray(xe, ye)
                t = intersect(A, v)
                if t is None:
                    continue
                px, py, pz = v[0] * t, v[1] * t, v[2] * t
                steps.append(abs(plane_z(B, px, py) - pz))
    return len(planes), steps


def derive_probe(labeler_root, gt_rows):
    import recall_by_depth_112 as rbd
    depthlib = rbd.load_depthlib(labeler_root)
    with open(DEPTH_ROWS, encoding="utf-8") as fh:
        drows = json.load(fh)
    measured = {p["pano"] for p in drows["panos"]
                if p["city"] == GSV and p["camera_height_status"] == "measured"}
    rng = {(p["pano"], p["x"], p["y"]): p["depth_range"] for p in drows["points"] if p["city"] == GSV}
    depth_dir = os.path.join(labeler_root, "runs", GSV, "depth")
    rows, payloads = [], {}
    for g in gt_rows:
        if g["arm"] != GSV or g["pano"] not in measured:
            continue
        if g["pano"] not in payloads:
            payloads[g["pano"]], _ = rbd.load_payload(depthlib, depth_dir, g["pano"])
        payload = payloads[g["pano"]]
        for shift in (0.0,) + PROBE_NULL_SHIFTS:
            n_pl, steps = window_steps(payload, g["x"] + shift, g["y"], rbd.image_ray, rbd.intersect)
            rows.append({"pano": g["pano"], "k": g["k"], "shift": shift,
                         "depth_range": rng.get((g["pano"], g["x"], g["y"])),
                         "n_ground_planes": n_pl, "n_boundaries": len(steps),
                         "max_step_m": _r(max(steps)) if steps else None,
                         "median_step_m": _r(st.median(steps)) if steps else None})
    return {"labeler_commit": rbd.labeler_commit(labeler_root), "rows": rows}


# ---------------------------------------------------------------------------
# tables (from rows only)

def _pair_counts(data, leg, arm, pids):
    """(n_pairs, 4) array of (tp, fp, tp_recall, n_gt_recall) for one leg and arm."""
    by = {(s["leg"], s["arm"], s["pano"]): s for s in data["scores"]}
    out = np.zeros((len(pids), 4), dtype=np.int64)
    for i, pid in enumerate(pids):
        s = by[(leg, arm, pid)]
        out[i] = (s["tp"], s["fp"], s["tp"] if s["fn_confirmed"] else 0,
                  s["n_gt"] if s["fn_confirmed"] else 0)
    return out


def op_label(lm):
    """The operating point as the scoreboard prints it (review N2): '0.05 floor' for the
    open-vocabulary detectors, 'no score' for chat VLMs and pointers, a number otherwise."""
    if lm["kind"] != "challenger":
        return f"{lm['op']:.2f}"
    import scoreboard as sb
    return sb.OPERATING_POINT_NOTE.get(lm["note"], f"{lm['op']:.2f}")


def _score(c):
    p, r, f = prf(*[int(v) for v in c.sum(0)])
    tp, fp, tpr, ngt = (int(v) for v in c.sum(0))
    return {"P": _r(p), "R": _r(r), "F1": _r(f), "tp": tp, "fp": fp, "fn": ngt - tpr, "n_gt": ngt}


def tables(data):
    pairs = data["pairs"]
    gp, mp = [p["gsv"] for p in pairs], [p["mly"] for p in pairs]
    W = bootstrap_weights(len(pairs))
    t = {"bootstrap": {"unit": "pano pair", "seed": SEED, "draws": DRAWS, "n_pairs": len(pairs),
                       "interval": "percentile 2.5-97.5"}}

    # pairing
    pan = data["panos"]
    dists = sorted(p["dist_m"] for p in pairs)
    t["pairing"] = {
        "panos": {a: sum(1 for p in pan if p["arm"] == a) for a in ARMS},
        "within_radius_of_other_arm": {a: sum(1 for p in pan if p["arm"] == a
                                              and p["nn_other_arm_m"] <= PAIR_RADIUS_M) for a in ARMS},
        "pairs": len(pairs), "radius_m": PAIR_RADIUS_M,
        "dist_median_m": _r(st.median(dists), 2), "dist_max_m": _r(max(dists), 2),
        "min_same_arm_spacing_m": {a: _r(min(p["nn_same_arm_m"] for p in pan if p["arm"] == a), 2)
                                   for a in ARMS},
        "capture": {a: sorted({p["capture_date"] for p in pan if p["arm"] == a
                               and p["pano"] in (gp if a == GSV else mp)}) for a in ARMS}}

    # physical ramps
    gt = data["gt"]
    by_pano = {}
    for g in gt:
        by_pano.setdefault((g["arm"], g["pano"]), []).append(g)

    def match_pair(pr, tag="", rot=0.0, radius=MATCH_RADIUS_M):
        G = by_pano.get((GSV, pr["gsv"]), [])
        M = by_pano.get((MLY, pr["mly"]), [])
        gpts = [g for g in G if g["e" + tag] is not None]
        mpts = [m for m in M if m["e" + tag] is not None]
        if rot:
            pa = next(p for p in pan if p["arm"] == GSV and p["pano"] == pr["gsv"])

            def rotate(g):
                de, dn = g["e" + tag] - pa["e"], g["n" + tag] - pa["n"]
                c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                return (pa["e"] + c * de + s * dn, pa["n"] - s * de + c * dn)
            ga = [rotate(g) for g in gpts]
        else:
            ga = [(g["e" + tag], g["n" + tag]) for g in gpts]
        ma = [(m["e" + tag], m["n" + tag]) for m in mpts]
        mt = greedy_pairs(ga, ma, radius, lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]))
        return G, M, gpts, mpts, [(gpts[i], mpts[j], d) for i, j, d in mt]

    ramps, unplace, only = [], {GSV: 0, MLY: 0}, {GSV: 0, MLY: 0}
    for pr in pairs:
        G, M, gpts, mpts, mt = match_pair(pr)
        unplace[GSV] += len(G) - len(gpts)
        unplace[MLY] += len(M) - len(mpts)
        only[GSV] += len(gpts) - len(mt)
        only[MLY] += len(mpts) - len(mt)
        ramps += [{"gsv": pr["gsv"], "mly": pr["mly"], "g": g, "m": m, "d": d} for g, m, d in mt]
    n_gt = {GSV: sum(len(by_pano.get((GSV, p), [])) for p in gp),
            MLY: sum(len(by_pano.get((MLY, p), [])) for p in mp)}
    sens = {}
    for label, tag in (("all cameras 2.6 m", "_h26"), (f"Mapillary {MLY_CAM_H_SENS} m", "_hsens")):
        sens[label] = sum(len(match_pair(pr, tag)[4]) for pr in pairs)
    null = {f"{int(r)} deg": sum(len(match_pair(pr, rot=r)[4]) for pr in pairs)
            for r in NULL_ROTATIONS_DEG}
    sweep = []
    for rad in RADIUS_SWEEP_M:
        m_ = sum(len(match_pair(pr, radius=rad)[4]) for pr in pairs)
        nl = [sum(len(match_pair(pr, rot=r, radius=rad)[4]) for pr in pairs) for r in NULL_ROTATIONS_DEG]
        sweep.append({"radius_m": rad, "matched": m_, "null_mean": _r(st.mean(nl), 1),
                      "excess_over_null": _r(m_ - st.mean(nl), 1),
                      "share_excess": _r((m_ - st.mean(nl)) / m_) if m_ else None})
    t["ramps"] = {"gt_points": n_gt, "unplaceable": unplace, "radius_sweep": sweep,
                  "placeable": {a: n_gt[a] - unplace[a] for a in ARMS},
                  "matched": len(ramps), "only_one_arm": only,
                  "match_dist_median_m": _r(st.median(r["d"] for r in ramps), 2) if ramps else None,
                  "match_dist_p90_m": _r(float(np.percentile([r["d"] for r in ramps], 90)), 2) if ramps else None,
                  "sensitivity_matched": sens, "rotation_null_matched": null,
                  "match_radius_m": MATCH_RADIUS_M, "max_range_m": MAX_RANGE_M}

    # 2x2 per physical ramp, for every RampNet leg with both arms, at the primary radius and
    # at the tighter sensitivity radius (fewer chance matches, fewer ramps)
    t["two_by_two"] = {}
    ramps_at = {MATCH_RADIUS_M: ramps,
                TIGHT_RADIUS_M: [{"g": g, "m": m, "d": d} for pr in pairs
                                 for g, m, d in match_pair(pr, radius=TIGHT_RADIUS_M)[4]]}
    for lm, rad in [(lm, rad) for rad in (MATCH_RADIUS_M, TIGHT_RADIUS_M) for lm in data["legs"]]:
        if (lm["kind"] == "challenger" and lm["leg"] not in YOLO_PANO) or len(lm["arms"]) < 2:
            continue
        cells = {"both": 0, "gsv_only": 0, "mly_only": 0, "neither": 0}
        for r in ramps_at[rad]:
            if not (r["g"]["fn_confirmed"] and r["m"]["fn_confirmed"]):
                continue
            hg, hm = r["g"]["hit"][lm["leg"]], r["m"]["hit"][lm["leg"]]
            cells["both" if hg and hm else "gsv_only" if hg else "mly_only" if hm else "neither"] += 1
        n = sum(cells.values())
        cells.update({"n": n, "recall_gsv": _r((cells["both"] + cells["gsv_only"]) / n) if n else None,
                      "recall_mly": _r((cells["both"] + cells["mly_only"]) / n) if n else None,
                      "net": cells["gsv_only"] - cells["mly_only"], "radius_m": rad,
                      "leg": lm["leg"],
                      "mcnemar_p": _r(mcnemar_exact(cells["gsv_only"], cells["mly_only"]))})
        t["two_by_two"][lm["leg"] if rad == MATCH_RADIUS_M else f"{lm['leg']} ({rad:g} m)"] = cells

    # paired scores + bootstrap deltas
    t["scores"], draws_f1 = [], {}
    for lm in data["legs"]:
        leg = lm["leg"]
        row = {"leg": leg, "kind": lm["kind"], "op": lm["op"], "op_label": op_label(lm)}
        cnt, whole = {}, {}
        for a, pids in ((GSV, gp), (MLY, mp)):
            if a not in lm["arms"]:
                row[a] = None
                continue
            cnt[a] = _pair_counts(data, leg, a, pids)
            row[a] = _score(cnt[a])
            pset = set(pids)
            row[a]["preds"] = sum(s["tp"] + s["fp"] + s["ignored"] for s in data["scores"]
                                  if s["leg"] == leg and s["arm"] == a and s["pano"] in pset)
            allp = sorted({s["pano"] for s in data["scores"] if s["arm"] == a and s["leg"] == leg})
            wc = _pair_counts(data, leg, a, allp)
            row[a + "_whole_arm"] = _score(wc)
            whole[a] = prf(*wc.sum(0).tolist())
        if len(cnt) == 2:
            bg, bm = W @ cnt[GSV], W @ cnt[MLY]
            pg, rg, fg = prf_arrays(bg.astype(float))
            pm, rm, fm = prf_arrays(bm.astype(float))
            # deltas from unrounded counts, never from the rounded cells above
            g3, m3 = prf(*cnt[GSV].sum(0).tolist()), prf(*cnt[MLY].sum(0).tolist())
            row["delta"] = {
                "P": _r(g3[0] - m3[0]), "P_ci": ci(pg - pm),
                "R": _r(g3[1] - m3[1]), "R_ci": ci(rg - rm),
                "F1": _r(g3[2] - m3[2]), "F1_ci": ci(fg - fm)}
            row["delta_whole_arm_F1"] = _r(whole[GSV][2] - whole[MLY][2])
            draws_f1[leg] = fg - fm
        else:
            row["delta"] = None
        t["scores"].append(row)

    # headline: each RampNet leg's paired dF1 against each YOLO pano arm's, same bootstrap
    # draws. The deployed GSV run is not the same input path the YOLO arms saw; rampnet_r2048
    # is (the committed JPEGs, review B1), so it is reported beside it at both thresholds.
    head = []
    for rleg in HEADLINE_RAMPNET_LEGS:
        if rleg not in draws_f1:
            continue
        base = draws_f1[rleg]
        rd = next(r for r in t["scores"] if r["leg"] == rleg)["delta"]["F1"]
        for y in YOLO_PANO:
            if y not in draws_f1:
                continue
            yd = next(r for r in t["scores"] if r["leg"] == y)["delta"]["F1"]
            diff = base - draws_f1[y]
            c = ci(diff)
            head.append({"rampnet_leg": rleg, "yolo": y, "rampnet_dF1": rd, "yolo_dF1": yd,
                         "difference": _r(rd - yd), "difference_ci": c,
                         "clears_zero": bool(c[0] > 0 or c[1] < 0),
                         "share_draws_rampnet_larger": _r(float(np.mean(diff > 0))),
                         "ratio_of_point_estimates": _r(rd / yd) if yd > 0 else None})
    t["headline"] = head

    # near-miss delta: median y missed - median y detected, RampNet at 0.55
    def ydelta(rows, by="verdict", leg="rampnet@0.55"):
        """median y(missed) - median y(detected). ``verdict``: the issue's definition, detected =
        verdict-true detections, missed = non-unsure missed marks (every pano). ``scorer``: GT
        points the benchmark matcher hits vs does not, on recall-eligible panos."""
        if by == "verdict":
            hit = [g["y"] for g in rows if g["source"] == "det"]
            miss = [g["y"] for g in rows if g["source"] == "missed"]
        else:
            hit = [g["y"] for g in rows if g["fn_confirmed"] and g["hit"].get(leg)]
            miss = [g["y"] for g in rows if g["fn_confirmed"] and not g["hit"].get(leg)]
        if not hit or not miss:
            return None, len(hit), len(miss)
        return st.median(miss) - st.median(hit), len(hit), len(miss)

    nm = {}
    for a, pids in ((GSV, gp), (MLY, mp)):
        whole = [g for g in gt if g["arm"] == a]
        sub = [g for g in gt if g["arm"] == a and g["pano"] in set(pids)]
        nm[a] = {}
        for by in ("verdict", "scorer"):
            dw, hw, mw = ydelta(whole, by)
            ds, hs_, ms = ydelta(sub, by)
            # pair bootstrap on the paired subset's delta (medians, so resampled explicitly)
            rng = np.random.default_rng(SEED)
            per = [by_pano.get((a, pid), []) for pid in pids]
            bs = []
            for _ in range(NEAR_MISS_DRAWS):
                pick = rng.integers(0, len(pids), len(pids))
                d, _, _ = ydelta([g for i in pick for g in per[i]], by)
                if d is not None:
                    bs.append(d)
            nm[a][by] = {"whole_arm": {"delta": _r(dw), "n_detected": hw, "n_missed": mw},
                         "paired": {"delta": _r(ds), "n_detected": hs_, "n_missed": ms,
                                    "delta_ci": ci(bs), "ci_draws": NEAR_MISS_DRAWS}}
    t["near_miss"] = nm

    # curb probe
    probe = data.get("curb_probe")
    if probe:
        hitmap = {(g["pano"], g["k"]): g["hit"]["rampnet@0.55"] for g in gt if g["arm"] == GSV}

        def summ(rows):
            steps = [r["max_step_m"] for r in rows if r["max_step_m"] is not None]
            n = len(rows)
            return {"windows": n,
                    "one_ground_plane_or_none": sum(1 for r in rows if r["n_ground_planes"] <= 1),
                    "two_plus_ground_planes": sum(1 for r in rows if r["n_ground_planes"] >= 2),
                    "with_boundary": len(steps),
                    "curb_sized_step": sum(1 for s in steps if CURB_STEP_M[0] <= s <= CURB_STEP_M[1]),
                    "share_curb_sized": _r(sum(1 for s in steps if CURB_STEP_M[0] <= s <= CURB_STEP_M[1]) / n) if n else None,
                    "max_step_median_m": _r(st.median(steps)) if steps else None,
                    "max_step_p25_m": _r(float(np.percentile(steps, 25))) if steps else None,
                    "max_step_p75_m": _r(float(np.percentile(steps, 75))) if steps else None,
                    "depth_range_median_m": _r(st.median(r["depth_range"] for r in rows
                                                         if r["depth_range"] is not None))
                    if any(r["depth_range"] is not None for r in rows) else None}

        rows = probe["rows"]
        g0 = [r for r in rows if r["shift"] == 0.0]
        nul = [r for r in rows if r["shift"] != 0.0]
        near = [r for r in g0 if r["depth_range"] is not None and r["depth_range"] < 8.0]
        t["curb_probe"] = {
            "gt_detected": summ([r for r in g0 if hitmap[(r["pano"], r["k"])]]),
            "gt_missed": summ([r for r in g0 if not hitmap[(r["pano"], r["k"])]]),
            "gt_all": summ(g0),
            "null_same_row": summ(nul),
            "gt_within_8m_depth": summ(near),
            "by_range": [{"band": f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g} m+",
                          "gt": summ([r for r in g0 if r["depth_range"] is not None
                                      and lo <= r["depth_range"] < hi]),
                          "null": summ([r for r in nul if r["depth_range"] is not None
                                        and lo <= r["depth_range"] < hi])}
                         for lo, hi in PROBE_RANGE_BANDS],
            "window_cells": [2 * PROBE_HALF_ROWS + 1, 2 * PROBE_HALF_COLS + 1],
            "curb_step_m": list(CURB_STEP_M)}
    return t


# ---------------------------------------------------------------------------
# markdown

def _f(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def _sd(v, nd=3):
    return "–" if v is None else f"{v:+.{nd}f}"


def _ci(c, nd=3):
    return "" if not c else f" [{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def md_tables(t):
    out = {}
    pa = t["pairing"]
    out["pairing"] = "\n".join([
        "| | laurens_gsv | laurens_mapillary |", "|---|---:|---:|",
        f"| bundle panos | {pa['panos'][GSV]} | {pa['panos'][MLY]} |",
        f"| with a pano of the other arm within {pa['radius_m']:g} m | {pa['within_radius_of_other_arm'][GSV]} | {pa['within_radius_of_other_arm'][MLY]} |",
        f"| in a one-to-one pair | {pa['pairs']} | {pa['pairs']} |",
        f"| minimum spacing between two panos of the same arm | {pa['min_same_arm_spacing_m'][GSV]:.1f} m | {pa['min_same_arm_spacing_m'][MLY]:.1f} m |",
        f"| capture months of the paired panos | {', '.join(pa['capture'][GSV])} | {', '.join(pa['capture'][MLY])} |",
        f"| pair distance, median / max | {pa['dist_median_m']:.1f} / {pa['dist_max_m']:.1f} m | |"])
    r = t["ramps"]
    out["ramps"] = "\n".join([
        "| | laurens_gsv | laurens_mapillary |", "|---|---:|---:|",
        f"| GT ramps on the paired panos | {r['gt_points'][GSV]} | {r['gt_points'][MLY]} |",
        f"| not placeable (above the horizon or beyond {r['max_range_m']:g} m) | {r['unplaceable'][GSV]} | {r['unplaceable'][MLY]} |",
        f"| placed | {r['placeable'][GSV]} | {r['placeable'][MLY]} |",
        f"| **matched across arms within {r['match_radius_m']:g} m (physical ramps both GTs have)** | **{r['matched']}** | **{r['matched']}** |",
        f"| placed but in this arm's GT only | {r['only_one_arm'][GSV]} | {r['only_one_arm'][MLY]} |"])
    out["ramps_checks"] = "\n".join(
        ["| check | matched ramps |", "|---|---:|",
         f"| as placed (GSV measured height else 2.6 m; Mapillary 2.6 m) | {r['matched']} |"]
        + [f"| sensitivity: {k} | {v} |" for k, v in r["sensitivity_matched"].items()]
        + [f"| null: GSV points rotated {k} about their camera | {v} |" for k, v in r["rotation_null_matched"].items()]
        + [f"| match distance, median / p90 | {r['match_dist_median_m']:.2f} / {r['match_dist_p90_m']:.2f} m |"])
    out["ramps_radius"] = "\n".join(
        ["| match radius | matched | null (mean of 3 rotations) | excess over null | share of matches in excess |",
         "|---:|---:|---:|---:|---:|"]
        + [f"| {w['radius_m']:g} m | {w['matched']} | {w['null_mean']:.1f} | {w['excess_over_null']:.1f} | "
           f"{_f(w['share_excess'], 2)} |" for w in r["radius_sweep"]])
    L = ["| leg | op | predictions GSV / Mly | GSV P / R / F1 | Mapillary P / R / F1 | ΔP [95% CI] | ΔR [95% CI] | **ΔF1 [95% CI]** | whole-arm ΔF1 |",
         "|---|---:|---:|---|---|---|---|---|---:|"]
    for s in t["scores"]:
        g, m, d = s[GSV], s[MLY], s["delta"]
        gs = "–" if g is None else f"{g['P']:.3f} / {g['R']:.3f} / {g['F1']:.3f}"
        ms = "–" if m is None else f"{m['P']:.3f} / {m['R']:.3f} / {m['F1']:.3f}"
        npred = f"{'–' if g is None else g['preds']} / {'–' if m is None else m['preds']}"
        if d:
            L.append(f"| {s['leg']} | {s['op_label']} | {npred} | {gs} | {ms} | {d['P']:+.3f}{_ci(d['P_ci'])} | "
                     f"{d['R']:+.3f}{_ci(d['R_ci'])} | **{d['F1']:+.3f}**{_ci(d['F1_ci'])} | {s['delta_whole_arm_F1']:+.3f} |")
        else:
            L.append(f"| {s['leg']} | {s['op_label']} | {npred} | {gs} | {ms} | – | – | – | – |")
    out["scores"] = "\n".join(L)
    L = ["| RampNet leg | YOLO arm (0.25) | RampNet ΔF1 | YOLO ΔF1 | RampNet minus YOLO [95% CI] | draws with RampNet larger | ratio of point estimates |",
         "|---|---|---:|---:|---|---:|---:|"]
    for h in t["headline"]:
        L.append(f"| {h['rampnet_leg']} | {h['yolo']} | {h['rampnet_dF1']:+.3f} | {h['yolo_dF1']:+.3f} | "
                 f"{h['difference']:+.3f}{_ci(h['difference_ci'])} | {h['share_draws_rampnet_larger']:.3f} | "
                 f"{_f(h['ratio_of_point_estimates'], 2)} |")
    out["headline"] = "\n".join(L)
    L = ["| match radius | leg | matched ramps | hit on both | GSV only | Mapillary only | neither | net (GSV − Mly) | exact McNemar p | recall GSV | recall Mapillary |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(t["two_by_two"].values(), key=lambda c: (-c["radius_m"], c["leg"])):
        L.append(f"| {c['radius_m']:g} m | {c['leg']} | {c['n']} | {c['both']} | {c['gsv_only']} | {c['mly_only']} | "
                 f"{c['neither']} | {c['net']:+d} | {c['mcnemar_p']:.2f} | "
                 f"{_f(c['recall_gsv'])} | {_f(c['recall_mly'])} |")
    out["two_by_two"] = "\n".join(L)
    L = ["| definition | arm | whole arm: delta (detected / missed) | paired panos: delta [95% CI] (detected / missed) |",
         "|---|---|---|---|"]
    for by, lab in (("verdict", "verdict (the issue's)"), ("scorer", "scorer hit at 0.55")):
        for a in ARMS:
            w, p = t["near_miss"][a][by]["whole_arm"], t["near_miss"][a][by]["paired"]
            L.append(f"| {lab} | {a} | {_sd(w['delta'], 4)} ({w['n_detected']} / {w['n_missed']}) | "
                     f"{_sd(p['delta'], 4)}{_ci(p['delta_ci'], 4)} ({p['n_detected']} / {p['n_missed']}) |")
    out["near_miss"] = "\n".join(L)
    cp = t.get("curb_probe")
    if cp:
        L = ["| windows | n | median depth range | ≤ 1 ground-like plane | ≥ 2 ground-like planes | with a ground/ground boundary | curb-sized step (5–30 cm) | largest step per window: p25 / median / p75 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
        for key, lab in (("gt_detected", "GT ramps RampNet detected (0.55)"),
                         ("gt_missed", "GT ramps RampNet missed"),
                         ("gt_all", "all GT ramps"), ("gt_within_8m_depth", "GT ramps within 8 m (depth axis)"),
                         ("null_same_row", "null: same image row, azimuth +90°/180°/270°")):
            c = cp[key]
            L.append(f"| {lab} | {c['windows']} | {_f(c['depth_range_median_m'], 1)} m | {c['one_ground_plane_or_none']} | {c['two_plus_ground_planes']} | "
                     f"{c['with_boundary']} | {c['curb_sized_step']} ({_f(c['share_curb_sized'])}) | "
                     f"{_f(c['max_step_p25_m'], 3)} / {_f(c['max_step_median_m'], 3)} / {_f(c['max_step_p75_m'], 3)} m |")
        out["curb_probe"] = "\n".join(L)
        L = ["| depth range | GT windows | GT: curb-sized step | GT: largest step, median | null windows | null: curb-sized step | null: largest step, median |",
             "|---|---:|---:|---:|---:|---:|---:|"]
        for b in cp["by_range"]:
            g, n_ = b["gt"], b["null"]
            L.append(f"| {b['band']} | {g['windows']} | {g['curb_sized_step']} ({_f(g['share_curb_sized'])}) | "
                     f"{_f(g['max_step_median_m'])} m | {n_['windows']} | {n_['curb_sized_step']} ({_f(n_['share_curb_sized'])}) | "
                     f"{_f(n_['max_step_median_m'])} m |")
        out["curb_probe_by_range"] = "\n".join(L)
    return out


def markdown(data, t):
    L = ["# Laurens, paired: the corners both rigs saw (#151)\n",
         f"Generated by `scripts/analysis/laurens_paired_151.py`. rows_sha256 `{data['rows_sha256']}`. "
         f"Bootstrap: {t['bootstrap']['draws']} draws over {t['bootstrap']['n_pairs']} pano pairs, "
         f"seed {t['bootstrap']['seed']}, percentile intervals.\n"]
    for name, tab in md_tables(t).items():
        L.append(f"## {name}\n\n{tab}\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------

ROW_KEYS = ("origin", "pairs", "panos", "gt", "scores", "legs", "curb_probe")


def rows_sha256(data):
    blob = json.dumps({k: data.get(k) for k in ROW_KEYS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def build(labeler_root=None, probe=None):
    data = derive_rows()
    if probe is None and labeler_root:
        probe = derive_probe(labeler_root, data["gt"])
    data["curb_probe"] = probe
    data["inputs_sha256"] = input_hashes()
    data["constants"] = {"pair_radius_m": PAIR_RADIUS_M, "match_radius_m": MATCH_RADIUS_M,
                         "max_range_m": MAX_RANGE_M, "cam_h_default": CAM_H_DEFAULT,
                         "mly_cam_h_sensitivity": MLY_CAM_H_SENS, "seed": SEED, "draws": DRAWS,
                         "probe_window_cells": [2 * PROBE_HALF_ROWS + 1, 2 * PROBE_HALF_COLS + 1],
                         "probe_null_shifts": list(PROBE_NULL_SHIFTS), "curb_step_m": list(CURB_STEP_M)}
    data["rows_sha256"] = rows_sha256(data)
    data["tables"] = tables(data)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler-root", default=os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler"))
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--md", default=OUT_MD)
    ap.add_argument("--check", action="store_true",
                    help="re-derive every row but the curb probe from committed inputs, every table "
                         "from the rows, and the content hash; fail on any drift")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        with open(a.out, encoding="utf-8") as fh:
            committed = json.load(fh)
        if rows_sha256(committed) != committed["rows_sha256"]:
            raise SystemExit("rows_sha256 does not match the committed rows")
        fresh = build(probe=committed["curb_probe"])
        for k in ROW_KEYS + ("inputs_sha256", "constants", "rows_sha256"):
            if json.dumps(fresh.get(k), sort_keys=True) != json.dumps(committed.get(k), sort_keys=True):
                raise SystemExit(f"{k} does not re-derive from the committed inputs")
        if json.loads(json.dumps(tables(committed))) != committed["tables"]:
            raise SystemExit("tables do not re-derive from the committed rows")
        if json.loads(json.dumps(fresh["tables"])) != committed["tables"]:
            raise SystemExit("tables do not re-derive from the committed inputs")
        print(f"{a.out}: rows and tables re-derive; rows_sha256 {committed['rows_sha256']}")
        if a.markdown:
            print(markdown(committed, committed["tables"]))
        return 0
    data = build(labeler_root=a.labeler_root)
    write_json(a.out, data)
    with open(a.md, "w", encoding="utf-8", newline="") as fh:
        fh.write(markdown(data, data["tables"]))
    print(f"wrote {a.out} and {a.md}")
    if a.markdown:
        print(markdown(data, data["tables"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
