"""Stage 0 of the Argon-teacher plan: background leak (gate B) and the Argon-only box filter
(gates P, Y, S), on the four inspected open-rig splits, from outputs already on disk.

The plan (``argon-teacher-2026-11-15``, revised after review) asks two questions before any
money is spent on fresh Argon calls:

1. **Background leak.** If training labels put whole-frame background everywhere no model
   fired, how many sure ramps end up labelled background? Three rules:

   * **H** -- ignore discs of radius 2R around every RampNet peak >= 0.05, every Argon box
     and every YOLO box >= 0.05; everything else is background.
   * **H2** -- H plus discs at every fused site and every stored peak projected in from
     neighbouring panoramas of the labeler run, plus the far band (less than 6 deg below
     the horizon) ignored.
   * **E** -- labelled panoramas are ``sparse``; whole-frame background only from panoramas
     all three models call empty, matched to labelled panoramas by sequence and rig. Leak =
     share of such panoramas that hold a sure ramp.

   **Gate B:** H2 <= 5%, else E <= 5%, else stop.

2. **Filter.** Argon-only boxes (no RampNet peak >= 0.05 within R), after the rig mask and
   after dropping boxes within 2R of a peak >= 0.30, are described by six features. Features
   1-5 alone and in AND-pairs, with and without the snap (feature 6), are the 80 candidate
   filters. Each held-out split gets the candidate chosen on the other three (pooled
   precision >= 0.80, then most blind ramps kept) and is scored at R. **Gates P, Y, S** read
   the four held-out results pooled.

Subcommands::

    python scripts/analysis/argon_teacher_s0.py extract-neighbours --labeler-root ... --sites-dir ...
    python scripts/analysis/argon_teacher_s0.py extract-heatmaps --panos-root D:/Git/RampNet/benchmark
    python scripts/analysis/argon_teacher_s0.py run        # leak + features + filters -> results.json
    python scripts/analysis/argon_teacher_s0.py leak       # print the leak tables from results.json
    python scripts/analysis/argon_teacher_s0.py features   # print the per-box feature table
    python scripts/analysis/argon_teacher_s0.py filters    # print the LOSO choice and gates P/Y/S
    python scripts/analysis/argon_teacher_s0.py figures    # docs/figures/argon_teacher/*.png
    python scripts/analysis/argon_teacher_s0.py --check    # rebuild results.json, byte-compare

The two ``extract-*`` steps write compact, committed derived inputs
(``analysis_out/argon_teacher_s0/neighbours.json`` and ``heatmap_features.json``). The first
needs the labeler runs (not in this repo), the second the benchmark panoramas and a GPU.
Everything after them reads committed files only, CPU, offline. ``--check`` covers ``run``.

Keys: every pano is addressed as ``(split, pano_id)``; nothing is joined across splits on a
bare id.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cascade_cost_35 as cc  # noqa: E402  (puts the repo and model_comparison on sys.path)
from rampnet.detection_eval import radius_sq_for, score_pano  # noqa: E402
from rampnet.geometry import dist_sq  # noqa: E402
from complementarity import load_floor_peaks  # noqa: E402

SPLITS = ("laurens_mapillary", "juneau", "bayonne", "budapest_district5")
OUT_DIR = os.path.join(cc.OUT, "argon_teacher_s0")
RESULTS = "results.json"
NEIGHBOURS = "neighbours.json"
HEATMAPS = "heatmap_features.json"
FIG_DIR = os.path.join(cc.REPO, "docs", "figures", "argon_teacher")

ARGON_ZS = "gemini-4-argon"                    # the leg the plan's "65 of 196" counts
ARGON_CONF = "gemini-4-argon-confidence-on"    # feature 3's score source
YOLO_ARMS = ("y11l_pano", "y11x_pano_h200", "y26_pano")

R = cc.RADIUS                       # 0.022 of pano width
R_SQ = radius_sq_for(R)
R2_SQ = radius_sq_for(2 * R)
SX, SY = cc.PANO_SCALE_X, cc.PANO_SCALE_Y
PEAK_FLOOR = 0.05                   # RampNet peak floor for blind / H / Argon-only
YOLO_FLOOR = 0.05                   # YOLO box floor for H, E and feature 4
KEPT_PEAK = 0.30                    # drop Argon-only boxes within 2R of a peak >= this
FAR_BAND_DEG = 6.0                  # H2: ignore everything less than this below the horizon
FAR_BAND_Y = 0.5 + FAR_BAND_DEG / 180.0
NADIR_MASK_DEG = 49.0               # the labeler's camera-rig mask (detectors.NADIR_MASK_DEG)
NADIR_MASK_Y = 0.5 + NADIR_MASK_DEG / 180.0
LEAK_BAR = 0.05                     # gate B
SELECT_PRECISION = 0.80             # LOSO eligibility
HEATMAP_LEVELS = (0.01, 0.02, 0.03)   # feature 1
SUPPORT_M = (3.0, 5.0)                # feature 2
SNAP_FLOOR = 0.01                     # feature 6
ELEV_Q = (0.05, 0.95)                 # feature 5
CONF_Q = {"median": 0.50, "q75": 0.75}  # feature 3

# Gate bars (plan section 3). Fixed; never tuned on these data.
P_GO, P_SPLIT_MIN, P_SPLIT_N, P_KILL = 0.80, 0.60, 10, 0.65
Y_GO, Y_KILL = 30, 20
S_GO_N, S_GO_FRAC = 10, 0.25


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def near(x, y, pts, radius_sq):
    """Is any of ``pts`` (``(x, y, ...)``) within ``radius_sq`` of ``(x, y)``, wrapped?"""
    return any(dist_sq(x, y, p[0], p[1], SX, SY, wrap_x=True) < radius_sq for p in pts)


def nearest(x, y, pts, radius_sq):
    """The point of ``pts`` closest to ``(x, y)`` within ``radius_sq`` (wrapped), or None."""
    best, bd = None, radius_sq
    for p in pts:
        d = dist_sq(x, y, p[0], p[1], SX, SY, wrap_x=True)
        if d < bd:
            best, bd = p, d
    return best


def quantile(vals, q):
    """Linear-interpolated quantile (numpy's default), without numpy."""
    v = sorted(vals)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, cc.REPO).replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_records(split):
    """``{pano_id: pano block}`` from the benchmark bundle's ``records.jsonl``."""
    out = {}
    with open(os.path.join(cc.REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["pano"]["panorama_id"]] = r["pano"]
    return out


def load_split(split):
    """Everything the CPU stages need for one split, keyed by pano id within the split."""
    gts = cc.load_gts(split)
    peaks = load_floor_peaks(split)
    zs = cc.load_challenger(split, ARGON_ZS)[0]
    conf = cc.load_challenger(split, ARGON_CONF)[0]
    yolo = {arm: cc.load_challenger(split, arm)[0] for arm in YOLO_ARMS}
    return {"gts": gts, "peaks": peaks, "zs": zs, "conf": conf, "yolo": yolo,
            "records": load_records(split)}


def yolo_union(d, pid, floor=YOLO_FLOOR, arms=YOLO_ARMS):
    return [c for arm in arms for c in d["yolo"][arm].get(pid, [])
            if c[2] is not None and c[2] >= floor]


def peaks_ge(d, pid, t):
    return [p for p in d["peaks"].get(pid, []) if p[2] >= t]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# extract-neighbours (needs the labeler runs; writes a committed derived file)
# --------------------------------------------------------------------------- #
#: Where each split's labeler run lives, relative to ``--labeler-root``. ``juneau`` is read
#: from a labeler worktree because the main checkout's ``runs/juneau/`` holds only the
#: submission records. ``budapest_district5`` has no usable run: its ``results.jsonl`` holds
#: 300 panos at the old 0.55 floor and only 2 of the 125 benchmark panos.
RUN_FILES = {"laurens_mapillary": "runs/laurens/results.jsonl",
             "bayonne": "runs/bayonne/results.jsonl",
             "juneau": "../sal-juneau/runs/juneau/results.jsonl",
             "budapest_district5": None}
NEIGHBOUR_RADIUS_M = 50.0     # two 25 m raycasts: nothing farther can land within range


def extract_neighbours(args):
    """For each benchmark pano with a labeler run: every stored peak of every other pano
    within 50 m, raycast to flat ground (2.6 m, the fusion default) and projected into this
    pano, and every fused site projected likewise; plus, per zero-shot Argon box, the
    distance from its ground point to the nearest other-pano peak or multi-pano-supported
    site (feature 2)."""
    sys.path.insert(0, os.path.abspath(args.labeler_root))
    import geo  # the labeler's stdlib-only geodesy (read only)

    out = {"note": "derived from the labeler runs; see docs/argon_teacher.md (Replication)",
           "camera_height_m": geo.DEFAULT_CAMERA_HEIGHT_M, "max_range_m": geo.DEFAULT_MAX_RANGE_M,
           "neighbour_radius_m": NEIGHBOUR_RADIUS_M, "splits": {}}
    for split in SPLITS:
        runf = RUN_FILES[split]
        if runf is None:
            out["splits"][split] = {"available": False,
                                    "reason": "no labeler run with stored peaks: runs/"
                                              "budapest_district5/results.jsonl holds 300 panos "
                                              "at the 0.55 floor, 2 of them benchmark panos"}
            print(f"{split}: unavailable", flush=True)
            continue
        run_path = os.path.normpath(os.path.join(args.labeler_root, runf))
        sites_path = os.path.join(args.sites_dir, f"{split}.sites.jsonl")
        records = load_records(split)
        zs = cc.load_challenger(split, ARGON_ZS)[0]
        panos = {}
        with open(run_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = r["pano"]
                if p.get("lat") is None or p.get("camera_heading") is None:
                    continue
                panos[p["panorama_id"]] = (p, [(d["x_normalized"], d["y_normalized"],
                                                d["confidence"]) for d in r["detections"]])
        sites = []
        with open(sites_path, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                sites.append((s["lat"], s["lng"], {m["pano_id"] for m in s["members"]}))
        missing = [pid for pid in records if pid not in panos]
        by_cell = {}
        for pid, (p, _) in panos.items():
            by_cell.setdefault((round(p["lat"], 3), round(p["lng"], 3)), []).append(pid)
        site_cell = {}
        for i, (la, ln, _) in enumerate(sites):
            site_cell.setdefault((round(la, 3), round(ln, 3)), []).append(i)

        def cells(lat, lng):
            a, b = round(lat, 3), round(lng, 3)
            return [(round(a + i * 0.001, 3), round(b + j * 0.001, 3))
                    for i in (-1, 0, 1) for j in (-1, 0, 1)]

        per = {}
        for pid in sorted(records):
            if pid not in panos:
                continue
            tp, _ = panos[pid]
            tpose = geo.pano_pose(tp)
            proj_peaks, ground_pts = [], []
            for c in cells(tp["lat"], tp["lng"]):
                for nid in by_cell.get(c, []):
                    if nid == pid:
                        continue
                    npano, dets = panos[nid]
                    if geo.haversine_m(tp["lat"], tp["lng"], npano["lat"], npano["lng"]) \
                            > NEIGHBOUR_RADIUS_M:
                        continue
                    npose = geo.pano_pose(npano)
                    errs = geo.error_model_for(npose.source)
                    for (x, y, conf) in dets:
                        g = geo.detection_ground_point(npose, x, y, errors=errs,
                                                       apply_pose=False)
                        if g is None:
                            continue
                        ground_pts.append((g.lat, g.lng))
                        q = geo.ground_point_to_pano(tpose, g.lat, g.lng)
                        if q is not None:
                            proj_peaks.append([round(q.x_norm, 6), round(q.y_norm, 6),
                                               round(conf, 6), round(q.range_m, 3)])
            proj_sites, support_sites = [], []
            for c in cells(tp["lat"], tp["lng"]):
                for i in site_cell.get(c, []):
                    la, ln, members = sites[i]
                    q = geo.ground_point_to_pano(tpose, la, ln)
                    if q is not None:
                        proj_sites.append([round(q.x_norm, 6), round(q.y_norm, 6),
                                           len(members), pid in members, round(q.range_m, 3)])
                    if members - {pid}:
                        support_sites.append((la, ln))
            support = []
            for (x, y, _c) in zs.get(pid, []):
                g = geo.detection_ground_point(tpose, x, y,
                                               errors=geo.error_model_for(tpose.source),
                                               apply_pose=False)
                if g is None:
                    support.append(None)
                    continue
                ds = [geo.haversine_m(g.lat, g.lng, a, b) for a, b in ground_pts + support_sites]
                support.append(round(min(ds), 3) if ds else 1e9)
            per[pid] = {"peaks": sorted(proj_peaks), "sites": sorted(proj_sites),
                        "argon_support_m": support}
        out["splits"][split] = {
            "available": True, "run": runf, "run_sha256": sha256_of(run_path),
            "sites_file": os.path.basename(sites_path), "sites_sha256": sha256_of(sites_path),
            "n_run_panos": len(panos), "n_sites": len(sites),
            "benchmark_panos_missing_from_run": sorted(missing), "panos": per}
        print(f"{split}: {len(per)} panos, {sum(len(v['peaks']) for v in per.values())} "
              f"projected peaks, {sum(len(v['sites']) for v in per.values())} projected sites, "
              f"{len(missing)} missing", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    cc.write_json(os.path.join(OUT_DIR, NEIGHBOURS), out)
    print(f"-> {os.path.join(OUT_DIR, NEIGHBOURS)}")


# --------------------------------------------------------------------------- #
# extract-heatmaps (GPU; writes a committed derived file)
# --------------------------------------------------------------------------- #
def disc_max(h, x, y, radius_px):
    """Max of heatmap ``h`` within ``radius_px`` (heatmap pixels, isotropic on a 2:1 map)
    of normalized ``(x, y)``, x wrapped round the seam."""
    import numpy as np
    H, W = h.shape
    cx, cy = x * W, y * H
    r = int(np.ceil(radius_px))
    rows = np.arange(max(0, int(cy) - r), min(H, int(cy) + r + 2))
    cols = np.arange(int(cx) - r, int(cx) + r + 2)
    dy = rows[:, None] + 0.0 - cy
    dx = cols[None, :] + 0.0 - cx
    mask = dx * dx + dy * dy < radius_px * radius_px
    sub = h[np.ix_(rows, cols % W)]
    return float(sub[mask].max()) if mask.any() else 0.0


def extract_heatmaps(args):
    """One RampNet forward pass per benchmark pano of the four splits (the op_cache's exact
    preprocessing and decode). Keeps only what features 1 and 6 need: per zero-shot Argon
    box the raw heatmap max within R and 2R, and the heatmap modes >= 0.01 (op_cache decode,
    min_distance 10, exclude_border=False) lying within 2R of some zero-shot Argon box."""
    import torch
    import threshold_sweep as ts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    out = {"model": "projectsidewalk/rampnet-model", "device": device.type,
           "preprocess": "threshold_sweep.PRE (2048x4096 bilinear, ImageNet norm, no TTA)",
           "mode_decode": {"threshold_abs": SNAP_FLOOR, "min_distance": 10,
                           "exclude_border": False},
           "splits": {}}
    for split in SPLITS:
        manifest = read_json(os.path.join(cc.REPO, "benchmark", split, "imagery_manifest.json"))
        zs = cc.load_challenger(split, ARGON_ZS)[0]
        op = load_floor_peaks(split)
        per, n_op_match, n_op = {}, 0, 0
        for pid in sorted(cc.load_gts(split)):
            path = os.path.join(args.panos_root, split, "panos", f"{pid}.jpg")
            want = manifest["panos"][pid]["sha256"]
            if sha256_of(path) != want:
                raise SystemExit(f"{path}: sha256 does not match imagery_manifest.json")
            h = ts.heatmap_for(model, device, path, False)
            import numpy as np
            h = np.clip(h, 0, 1)
            H, W = h.shape
            boxes = zs.get(pid, [])
            rpx = R * W
            bmax = [[round(disc_max(h, b[0], b[1], rpx), 6),
                     round(disc_max(h, b[0], b[1], 2 * rpx), 6)] for b in boxes]
            modes = ts.peaks_to_dets(h, SNAP_FLOOR, 10)
            keep = [[round(m[0], 6), round(m[1], 6), round(m[2], 6)] for m in modes
                    if near(m[0], m[1], boxes, R2_SQ)]
            # sanity: modes >= 0.05 against the committed op_cache peaks (same decode, but
            # the op_cache predates the seam fix, so seam peaks may differ)
            hi = [m for m in modes if m[2] >= PEAK_FLOOR]
            n_op += len(op.get(pid, []))
            n_op_match += sum(1 for p in op.get(pid, []) if near(p[0], p[1], hi, 1.0))
            per[pid] = {"heatmap_shape": [H, W], "box_max_R_2R": bmax, "modes_near_boxes": keep}
            del h
        out["splits"][split] = {"panos": per, "op_cache_peaks": n_op,
                                "op_cache_peaks_reproduced_within_1px": n_op_match}
        print(f"{split}: {len(per)} panos; op_cache peaks reproduced {n_op_match}/{n_op}",
              flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    cc.write_json(os.path.join(OUT_DIR, HEATMAPS), out)
    print(f"-> {os.path.join(OUT_DIR, HEATMAPS)}")


# --------------------------------------------------------------------------- #
# counts shared by every stage
# --------------------------------------------------------------------------- #
def base_counts(d):
    """Recall-confirmed pano counts, GT, blind (no peak >= 0.05 within R), strict-silent
    (none within 2R), and how many of each the zero-shot Argon leg reaches (any box within
    R). The plan's 196 / 65 come from here."""
    c = {"n_panos": len(d["gts"]), "n_confirmed": 0, "n_gt": 0, "n_gt_all_panos": 0,
         "n_blind": 0, "blind_reached": 0, "n_strict_silent": 0, "ss_reached": 0}
    for pid, gt in d["gts"].items():
        c["n_gt_all_panos"] += len(gt.gt_points)
        if not gt.fn_confirmed:
            continue
        c["n_confirmed"] += 1
        pk = peaks_ge(d, pid, PEAK_FLOOR)
        boxes = d["zs"].get(pid, [])
        for g in gt.gt_points:
            c["n_gt"] += 1
            if not near(g[0], g[1], pk, R_SQ):
                c["n_blind"] += 1
                c["blind_reached"] += near(g[0], g[1], boxes, R_SQ)
            if not near(g[0], g[1], pk, R2_SQ):
                c["n_strict_silent"] += 1
                c["ss_reached"] += near(g[0], g[1], boxes, R_SQ)
    return c


# --------------------------------------------------------------------------- #
# leak (gate B)
# --------------------------------------------------------------------------- #
def disc_centres_H(d, pid, argon_legs=(ARGON_ZS,), yolo_arms=YOLO_ARMS):
    pts = list(peaks_ge(d, pid, PEAK_FLOOR))
    for leg in argon_legs:
        pts += list((d["zs"] if leg == ARGON_ZS else d["conf"]).get(pid, []))
    pts += yolo_union(d, pid, arms=yolo_arms)
    return pts


def leak_H(d, centres_fn=disc_centres_H, extra=None, far_band=False):
    """Sure ramps (GT on recall-confirmed panos) outside every 2R ignore disc. ``extra``
    (pid -> list of points) adds discs; ``far_band`` ignores y < FAR_BAND_Y."""
    n = leaked = 0
    for pid, gt in d["gts"].items():
        if not gt.fn_confirmed:
            continue
        pts = centres_fn(d, pid) + (extra.get(pid, []) if extra else [])
        for g in gt.gt_points:
            n += 1
            if far_band and g[1] < FAR_BAND_Y:
                continue
            if not near(g[0], g[1], pts, R2_SQ):
                leaked += 1
    return {"n_gt": n, "leaked": leaked, "leak": leaked / n if n else None}


def is_empty(d, pid):
    """All three models call the pano empty: no RampNet peak >= 0.05, no zero-shot Argon
    box, no YOLO box >= 0.05 on any of the three arms (camera rig included)."""
    return not (peaks_ge(d, pid, PEAK_FLOOR) or d["zs"].get(pid) or yolo_union(d, pid))


def leak_E(d):
    """Rule E on one split: empties (recall-confirmed) whose (sequence, rig) also holds a
    labelled (non-empty) pano of the split; leak = share of those holding a sure ramp."""
    recs = d["records"]
    key = {pid: (recs[pid].get("sequence_id"), recs[pid].get("camera_model")) for pid in d["gts"]}
    labelled_keys = {key[pid] for pid in d["gts"] if not is_empty(d, pid)}
    out = {"n_confirmed": 0, "n_empty": 0, "n_empty_matched": 0, "matched_with_ramp": 0,
           "unmatched_with_ramp": 0, "ramps_in_matched": 0, "matched_panos_with_ramp": []}
    for pid, gt in d["gts"].items():
        if not gt.fn_confirmed:
            continue
        out["n_confirmed"] += 1
        if not is_empty(d, pid):
            continue
        out["n_empty"] += 1
        has = bool(gt.gt_points)
        if key[pid] in labelled_keys:
            out["n_empty_matched"] += 1
            out["matched_with_ramp"] += has
            out["ramps_in_matched"] += len(gt.gt_points)
            if has:
                out["matched_panos_with_ramp"].append(pid)
        else:
            out["unmatched_with_ramp"] += has
    out["leak"] = (out["matched_with_ramp"] / out["n_empty_matched"]
                   if out["n_empty_matched"] else None)
    return out


def neighbour_points(nb, split):
    """pid -> projected neighbour peaks and sites (H2's extra discs), or None if the split
    has no labeler run."""
    s = nb["splits"][split]
    if not s["available"]:
        return None
    return {pid: [tuple(p[:2]) for p in v["peaks"]] + [tuple(p[:2]) for p in v["sites"]]
            for pid, v in s["panos"].items()}


def _pool(rows, keys=("n_gt", "leaked")):
    tot = {k: sum(r[k] for r in rows) for k in keys}
    tot["leak"] = tot["leaked"] / tot["n_gt"] if tot["n_gt"] else None
    return tot


def leak(data, nb):
    res = {"per_split": {}, "pooled": {}}
    for split in SPLITS:
        d = data[split]
        extra = neighbour_points(nb, split)
        res["per_split"][split] = {
            "H": leak_H(d),
            "H_both_argon_legs": leak_H(d, lambda dd, p: disc_centres_H(dd, p, (ARGON_ZS, ARGON_CONF))),
            **{f"H_yolo_{arm}_only": leak_H(d, lambda dd, p, a=arm: disc_centres_H(dd, p, yolo_arms=(a,)))
               for arm in YOLO_ARMS},
            "H_plus_far_band": leak_H(d, far_band=True),
            "H_plus_neighbours": None if extra is None else leak_H(d, extra=extra),
            "H2": leak_H(d, extra=extra or {}, far_band=True),
            "H2_neighbours_available": extra is not None,
            "E": leak_E(d)}
    ps = res["per_split"]
    for k in ("H", "H_both_argon_legs", "H_plus_far_band", "H2") + tuple(
            f"H_yolo_{a}_only" for a in YOLO_ARMS):
        res["pooled"][k] = _pool([ps[s][k] for s in SPLITS])
    have = [s for s in SPLITS if ps[s]["H2_neighbours_available"]]
    res["pooled"]["H2_splits_with_neighbours"] = {**_pool([ps[s]["H2"] for s in have]),
                                                 "splits": have}
    e = [ps[s]["E"] for s in SPLITS]
    em = sum(x["n_empty_matched"] for x in e)
    res["pooled"]["E"] = {"n_empty": sum(x["n_empty"] for x in e), "n_empty_matched": em,
                          "matched_with_ramp": sum(x["matched_with_ramp"] for x in e),
                          "leak": sum(x["matched_with_ramp"] for x in e) / em if em else None}
    res["gate_B"] = gate_B(res)
    return res


def gate_B(res):
    """H2 <= 5%, else E <= 5%, else stop. H2 is read over all four splits with any split
    that lacks neighbour data at its H + far-band value (an upper bound there), and over the
    splits that have neighbour data. If the two reads disagree, H2 is unreadable and the
    gate falls through to E, as for an H2 failure."""
    p = res["pooled"]
    h2_all, h2_have = p["H2"]["leak"], p["H2_splits_with_neighbours"]["leak"]
    if h2_all <= LEAK_BAR:
        h2 = "pass"
    elif h2_have > LEAK_BAR:
        h2 = "fail"
    else:
        h2 = "unreadable"
    e = p["E"]["leak"]
    e_state = None if e is None else ("pass" if e <= LEAK_BAR else "fail")
    if h2 == "pass":
        verdict, rule = "go", "H2"
    elif e_state == "pass":
        verdict, rule = "go", "E"
    else:
        verdict, rule = "stop", None
    return {"bar": LEAK_BAR, "H2_four_splits": h2_all, "H2_splits_with_neighbours": h2_have,
            "H2": h2, "E": e, "E_state": e_state, "verdict": verdict, "rule_passed": rule}


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def label_free_thresholds(data):
    """Feature 3 and 5 thresholds. Both are label-free statistics of Argon's own output (no
    GT), so they are taken over all four splits rather than per fold."""
    ys = [c[1] for s in SPLITS for cs in data[s]["zs"].values() for c in cs]
    sc = [c[2] for s in SPLITS for cs in data[s]["conf"].values() for c in cs if c[2] is not None]
    return {"elev_y": [quantile(ys, ELEV_Q[0]), quantile(ys, ELEV_Q[1])],
            "n_zero_shot_boxes": len(ys), "n_conf_scores": len(sc),
            "conf": {k: quantile(sc, q) for k, q in CONF_Q.items()}}


def features(data, nb, hm, thr):
    """One row per Argon-only zero-shot box on a recall-confirmed pano (no peak >= 0.05
    within R), with the rig-mask and >= 0.30-within-2R drops recorded, and features 1-6."""
    rows, drops = [], {}
    for split in SPLITS:
        d = data[split]
        nbs = nb["splits"][split]
        hms = hm["splits"][split]["panos"]
        dr = {"argon_boxes_confirmed": 0, "argon_only": 0, "rig_mask": 0, "near_peak_030": 0,
              "kept": 0}
        for pid, gt in d["gts"].items():
            if not gt.fn_confirmed:
                continue
            pk05, pk30 = peaks_ge(d, pid, PEAK_FLOOR), peaks_ge(d, pid, KEPT_PEAK)
            yolo = yolo_union(d, pid)
            conf = [c for c in d["conf"].get(pid, []) if c[2] is not None]
            hmp = hms[pid]
            for i, b in enumerate(d["zs"].get(pid, [])):
                dr["argon_boxes_confirmed"] += 1
                if near(b[0], b[1], pk05, R_SQ):
                    continue
                dr["argon_only"] += 1
                if b[1] > NADIR_MASK_Y:
                    dr["rig_mask"] += 1
                    continue
                if near(b[0], b[1], pk30, R2_SQ):
                    dr["near_peak_030"] += 1
                    continue
                dr["kept"] += 1
                cm = nearest(b[0], b[1], conf, R_SQ)
                snap = nearest(b[0], b[1], hmp["modes_near_boxes"], R2_SQ)
                sup = (nbs["panos"][pid]["argon_support_m"][i]
                       if nbs["available"] and pid in nbs["panos"] else None)
                rows.append({
                    "split": split, "pano": pid, "box": i, "x": b[0], "y": b[1],
                    "f1_heatmap_max_R": hmp["box_max_R_2R"][i][0],
                    "f2_support_m": sup,
                    "f2_available": bool(nbs["available"]),
                    "f3_conf_score": None if cm is None else cm[2],
                    "f4_yolo_2R": near(b[0], b[1], yolo, R2_SQ),
                    "f5_elev_in_range": thr["elev_y"][0] <= b[1] <= thr["elev_y"][1],
                    "f6_snap": None if snap is None else [snap[0], snap[1], snap[2]]})
        drops[split] = dr
    return rows, drops


#: Feature tests: name -> (feature number, predicate on a row).
def feature_tests(thr):
    t = {}
    for lv in HEATMAP_LEVELS:
        t[f"f1>={lv:g}"] = (1, lambda r, lv=lv: r["f1_heatmap_max_R"] >= lv)
    for m in SUPPORT_M:
        t[f"f2<={m:g}m"] = (2, lambda r, m=m: r["f2_support_m"] is not None and r["f2_support_m"] <= m)
    for k, v in thr["conf"].items():
        t[f"f3>={k}"] = (3, lambda r, v=v: r["f3_conf_score"] is not None and r["f3_conf_score"] >= v)
    t["f4_yolo"] = (4, lambda r: r["f4_yolo_2R"])
    t["f5_elev"] = (5, lambda r: r["f5_elev_in_range"])
    return t


def candidates(thr):
    """The 80 candidates: 9 single tests and 31 AND-pairs of tests from different features,
    each with and without the snap. ``(name, tests, snap)``."""
    tests = feature_tests(thr)
    names = list(tests)
    combos = [(n,) for n in names] + [
        (a, b) for a, b in itertools.combinations(names, 2) if tests[a][0] != tests[b][0]]
    out = []
    for combo in combos:
        for snap in (False, True):
            name = " & ".join(combo) + (" +snap" if snap else "")
            out.append((name, combo, snap))
    return out, tests


def score_candidate(rows, data, combo, snap, tests, split):
    """Kept boxes of one split under one candidate, scored at R on the recall-confirmed
    panos: TP / FP / ignored, blind ramps kept, strict-silent ramps kept."""
    d = data[split]
    kept = {}
    for r in rows:
        if r["split"] != split or not all(tests[n][1](r) for n in combo):
            continue
        xy = (r["f6_snap"][0], r["f6_snap"][1]) if snap and r["f6_snap"] else (r["x"], r["y"])
        kept.setdefault(r["pano"], []).append(xy)
    o = {"boxes": 0, "tp": 0, "fp": 0, "ignored": 0, "blind_kept": 0, "ss_kept": 0}
    for pid, gt in d["gts"].items():
        if not gt.fn_confirmed:
            continue
        boxes = kept.get(pid, [])
        if not boxes:
            continue
        s = score_pano(boxes, gt, R_SQ)
        o["boxes"] += len(boxes)
        o["tp"] += s.tp
        o["fp"] += s.fp
        o["ignored"] += s.ignored
        pk = peaks_ge(d, pid, PEAK_FLOOR)
        for g in gt.gt_points:
            if near(g[0], g[1], boxes, R_SQ):
                if not near(g[0], g[1], pk, R_SQ):
                    o["blind_kept"] += 1
                if not near(g[0], g[1], pk, R2_SQ):
                    o["ss_kept"] += 1
    return o


def _prec(o):
    return o["tp"] / (o["tp"] + o["fp"]) if o["tp"] + o["fp"] else None


def _add(rows):
    keys = ("boxes", "tp", "fp", "ignored", "blind_kept", "ss_kept")
    o = {k: sum(r[k] for r in rows) for k in keys}
    o["precision"] = _prec(o)
    return o


def filters(data, rows, thr, counts):
    cands, tests = candidates(thr)
    table = {}
    for name, combo, snap in cands:
        table[name] = {s: score_candidate(rows, data, combo, snap, tests, s) for s in SPLITS}
        for s in SPLITS:
            table[name][s]["precision"] = _prec(table[name][s])
    order = {name: (len(combo), snap) for name, combo, snap in cands}
    loso = {}
    for held in SPLITS:
        train = [s for s in SPLITS if s != held]
        scored = []
        for name, _, _ in cands:
            tr = _add([table[name][s] for s in train])
            if tr["precision"] is not None and tr["precision"] >= SELECT_PRECISION:
                scored.append((name, tr))
        # most blind ramps kept on the training splits; then precision; then simpler;
        # then no snap; then name
        scored.sort(key=lambda t: (-t[1]["blind_kept"], -t[1]["precision"],
                                   order[t[0]][0], order[t[0]][1], t[0]))
        if scored:
            name, tr = scored[0]
            loso[held] = {"selected": name, "n_eligible": len(scored), "train": tr,
                          "held_out": dict(table[name][held])}
        else:
            loso[held] = {"selected": None, "n_eligible": 0, "train": None,
                          "held_out": {"boxes": 0, "tp": 0, "fp": 0, "ignored": 0,
                                       "blind_kept": 0, "ss_kept": 0, "precision": None}}
    pooled = _add([loso[s]["held_out"] for s in SPLITS])
    return {"n_candidates": len(cands), "candidates": table, "loso": loso,
            "held_out_pooled": pooled, "gates": gates(loso, pooled, counts)}


def gates(loso, pooled, counts):
    tp, fp, ig = pooled["tp"], pooled["fp"], pooled["ignored"]
    p = pooled["precision"]
    p_opt = (tp + ig) / (tp + fp + ig) if tp + fp + ig else None
    low = [s for s in SPLITS if loso[s]["held_out"]["boxes"] >= P_SPLIT_N
           and (loso[s]["held_out"]["precision"] or 0.0) < P_SPLIT_MIN]
    if p is not None and p >= P_GO and not low:
        gp = "go"
    elif p_opt is None or p_opt < P_KILL:
        gp = "kill"
    else:
        gp = "in-between"
    y = pooled["blind_kept"]
    gy = "go" if y >= Y_GO else "kill" if y < Y_KILL else "in-between"
    ss_reached = sum(counts[s]["ss_reached"] for s in SPLITS)
    ss = pooled["ss_kept"]
    frac = ss / ss_reached if ss_reached else None
    gs = "go" if ss >= S_GO_N and frac is not None and frac >= S_GO_FRAC else "kill"
    return {
        "P": {"precision": p, "precision_ignored_counted_true": p_opt,
              "splits_10plus_boxes_under_0.60": low, "verdict": gp,
              "bars": {"go": P_GO, "split_min": P_SPLIT_MIN, "split_n": P_SPLIT_N, "kill": P_KILL}},
        "Y": {"blind_kept": y, "blind_reached_by_argon": sum(counts[s]["blind_reached"] for s in SPLITS),
              "verdict": gy, "bars": {"go": Y_GO, "kill": Y_KILL}},
        "S": {"ss_kept": ss, "ss_reached_by_argon": ss_reached, "fraction": frac,
              "n_strict_silent": sum(counts[s]["n_strict_silent"] for s in SPLITS),
              "verdict": gs, "bars": {"go_n": S_GO_N, "go_fraction": S_GO_FRAC}}}


def filters_without_f2(data, rows, thr, counts):
    """Sensitivity: the same LOSO with every feature-2 candidate removed, because feature 2
    is unavailable on budapest_district5 (every box there fails it)."""
    cands, tests = candidates(thr)
    keep = {name for name, combo, _ in cands if not any(tests[n][0] == 2 for n in combo)}
    full = filters(data, rows, thr, counts)
    table = {k: v for k, v in full["candidates"].items() if k in keep}
    order = {name: (len(combo), snap) for name, combo, snap in cands}
    loso = {}
    for held in SPLITS:
        train = [s for s in SPLITS if s != held]
        scored = [(n, _add([table[n][s] for s in train])) for n in table]
        scored = [(n, t) for n, t in scored if t["precision"] is not None
                  and t["precision"] >= SELECT_PRECISION]
        scored.sort(key=lambda t: (-t[1]["blind_kept"], -t[1]["precision"],
                                   order[t[0]][0], order[t[0]][1], t[0]))
        if scored:
            loso[held] = {"selected": scored[0][0], "n_eligible": len(scored),
                          "train": scored[0][1], "held_out": dict(table[scored[0][0]][held])}
        else:
            loso[held] = {"selected": None, "n_eligible": 0, "train": None,
                          "held_out": {"boxes": 0, "tp": 0, "fp": 0, "ignored": 0,
                                       "blind_kept": 0, "ss_kept": 0, "precision": None}}
    pooled = _add([loso[s]["held_out"] for s in SPLITS])
    return {"n_candidates": len(keep), "loso": loso, "held_out_pooled": pooled,
            "gates": gates(loso, pooled, counts)}


# --------------------------------------------------------------------------- #
# run / check
# --------------------------------------------------------------------------- #
def build():
    nb = read_json(os.path.join(OUT_DIR, NEIGHBOURS))
    hm = read_json(os.path.join(OUT_DIR, HEATMAPS))
    data = {s: load_split(s) for s in SPLITS}
    counts = {s: base_counts(data[s]) for s in SPLITS}
    pooled_counts = {k: sum(counts[s][k] for s in SPLITS) for k in counts[SPLITS[0]]}
    thr = label_free_thresholds(data)
    rows, drops = features(data, nb, hm, thr)
    return {
        "plan": "argon-teacher-2026-11-15, Stage 0",
        "constants": {"R": R, "peak_floor": PEAK_FLOOR, "yolo_floor": YOLO_FLOOR,
                      "yolo_arms": list(YOLO_ARMS), "argon_leg": ARGON_ZS,
                      "argon_score_leg": ARGON_CONF, "kept_peak": KEPT_PEAK,
                      "far_band_deg": FAR_BAND_DEG, "nadir_mask_deg": NADIR_MASK_DEG,
                      "leak_bar": LEAK_BAR, "select_precision": SELECT_PRECISION},
        "inputs": {"neighbours": {"file": rel(os.path.join(OUT_DIR, NEIGHBOURS)),
                                  "sha256": sha256_of(os.path.join(OUT_DIR, NEIGHBOURS))},
                   "heatmaps": {"file": rel(os.path.join(OUT_DIR, HEATMAPS)),
                                "sha256": sha256_of(os.path.join(OUT_DIR, HEATMAPS))}},
        "counts": {"per_split": counts, "pooled": pooled_counts},
        "leak": leak(data, nb),
        "thresholds": thr,
        "feature_drops": drops,
        "features": rows,
        "filters": filters(data, rows, thr, counts),
        "filters_without_f2": filters_without_f2(data, rows, thr, counts),
        "heatmap_sanity": {s: {k: hm["splits"][s][k] for k in
                               ("op_cache_peaks", "op_cache_peaks_reproduced_within_1px")}
                           for s in SPLITS}}


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def _p(v, fmt=".3f"):
    return "–" if v is None else format(v, fmt)


def print_leak(res):
    L = res["leak"]
    print("| split | GT (confirmed panos) | H | H, both Argon legs | H + far band | "
          "H + neighbours | H2 | E: matched empties (with a ramp) | E leak |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        r = L["per_split"][s]
        hn = r["H_plus_neighbours"]
        e = r["E"]
        print(f"| {s} | {r['H']['n_gt']} | {r['H']['leaked']} ({_pct(r['H']['leak'])}) | "
              f"{r['H_both_argon_legs']['leaked']} ({_pct(r['H_both_argon_legs']['leak'])}) | "
              f"{r['H_plus_far_band']['leaked']} ({_pct(r['H_plus_far_band']['leak'])}) | "
              + ("unavailable" if hn is None else f"{hn['leaked']} ({_pct(hn['leak'])})")
              + f" | {r['H2']['leaked']} ({_pct(r['H2']['leak'])})"
              + ("" if r["H2_neighbours_available"] else " *")
              + f" | {e['n_empty_matched']} ({e['matched_with_ramp']}) | {_pct(e['leak'])} |")
    p = L["pooled"]
    print(f"| **pooled** | {p['H']['n_gt']} | {p['H']['leaked']} ({_pct(p['H']['leak'])}) | "
          f"{p['H_both_argon_legs']['leaked']} ({_pct(p['H_both_argon_legs']['leak'])}) | "
          f"{p['H_plus_far_band']['leaked']} ({_pct(p['H_plus_far_band']['leak'])}) | – | "
          f"{p['H2']['leaked']} ({_pct(p['H2']['leak'])}) | {p['E']['n_empty_matched']} "
          f"({p['E']['matched_with_ramp']}) | {_pct(p['E']['leak'])} |")
    h3 = p["H2_splits_with_neighbours"]
    print(f"\nH2 over the splits with neighbour data ({', '.join(h3['splits'])}): "
          f"{h3['leaked']} / {h3['n_gt']} ({_pct(h3['leak'])})")
    print("H with one YOLO arm: " + ", ".join(
        f"{a} {p[f'H_yolo_{a}_only']['leaked']} ({_pct(p[f'H_yolo_{a}_only']['leak'])})"
        for a in YOLO_ARMS))
    print(f"\nGate B: {json.dumps(L['gate_B'])}")


def print_features(res):
    print(f"thresholds: {json.dumps(res['thresholds'])}")
    print(f"drops: {json.dumps(res['feature_drops'])}")
    print("| split | pano | box | x | y | f1 | f2 (m) | f3 | f4 | f5 | snap |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in res["features"]:
        print(f"| {r['split']} | {r['pano']} | {r['box']} | {r['x']:.4f} | {r['y']:.4f} | "
              f"{r['f1_heatmap_max_R']:.3f} | {_p(r['f2_support_m'], '.1f')} | "
              f"{_p(r['f3_conf_score'], '.2f')} | {r['f4_yolo_2R']} | {r['f5_elev_in_range']} | "
              f"{'–' if not r['f6_snap'] else _p(r['f6_snap'][2])} |")


def print_filters(res):
    for key in ("filters", "filters_without_f2"):
        F = res[key]
        print(f"\n**{key}** ({F['n_candidates']} candidates)\n")
        print("| held-out split | selected (training: TP / FP, blind kept) | eligible | "
              "held-out boxes | TP / FP / ignored | precision | blind kept | strict-silent kept |")
        print("|---|---|---:|---:|---|---:|---:|---:|")
        for s in SPLITS:
            lo = F["loso"][s]
            tr, ho = lo["train"], lo["held_out"]
            print(f"| {s} | {lo['selected']}"
                  + ("" if tr is None else f" ({tr['tp']} / {tr['fp']}, {tr['blind_kept']})")
                  + f" | {lo['n_eligible']} | {ho['boxes']} | {ho['tp']} / {ho['fp']} / "
                    f"{ho['ignored']} | {_p(ho['precision'])} | {ho['blind_kept']} | "
                    f"{ho['ss_kept']} |")
        po = F["held_out_pooled"]
        print(f"| **pooled** | | | {po['boxes']} | {po['tp']} / {po['fp']} / {po['ignored']} | "
              f"{_p(po['precision'])} | {po['blind_kept']} | {po['ss_kept']} |")
        print(f"\ngates: {json.dumps(F['gates'])}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    L = res["leak"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rules = [("H", "H"), ("H + far band", "H_plus_far_band"), ("H2", "H2")]
    xs = list(range(len(SPLITS) + 1))
    w = 0.2
    for i, (lab, k) in enumerate(rules):
        vals = [100 * (L["per_split"][s][k]["leak"] or 0) for s in SPLITS] + \
               [100 * (L["pooled"][k]["leak"] or 0)]
        ax.bar([x + (i - 1.5) * w for x in xs], vals, w, label=lab)
    ev = [100 * (L["per_split"][s]["E"]["leak"] or 0) for s in SPLITS] + \
         [100 * (L["pooled"]["E"]["leak"] or 0)]
    ax.bar([x + 1.5 * w for x in xs], ev, w, label="E (share of matched empties)")
    ax.axhline(100 * LEAK_BAR, color="k", ls="--", lw=1)
    ax.text(len(SPLITS) + 0.45, 100 * LEAK_BAR, " gate B 5%", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(SPLITS) + ["pooled"], fontsize=8)
    ax.set_ylabel("leak (%)")
    ax.set_title("Background leak by rule (budapest_district5: no neighbour data)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "leak.png"), dpi=150)
    plt.close(fig)

    F = res["filters"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, per in F["candidates"].items():
        o = _add([per[s] for s in SPLITS])
        if o["precision"] is None:
            continue
        ax.scatter(o["blind_kept"], o["precision"], s=10, alpha=0.5,
                   color="tab:orange" if name.endswith("+snap") else "tab:blue")
    po = F["held_out_pooled"]
    if po["precision"] is not None:
        ax.scatter(po["blind_kept"], po["precision"], s=80, marker="*", color="k",
                   label="LOSO held-out, pooled")
    ax.axhline(P_GO, color="k", ls="--", lw=1)
    ax.axvline(Y_GO, color="k", ls=":", lw=1)
    ax.set_xlabel("blind ramps kept (all four splits, in sample)")
    ax.set_ylabel("precision at R")
    ax.set_title("80 candidate filters (blue: no snap, orange: snap)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "filters.png"), dpi=150)
    plt.close(fig)
    print(f"-> {FIG_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=("run", "leak", "features", "filters", "figures",
                             "extract-neighbours", "extract-heatmaps"))
    ap.add_argument("--check", action="store_true",
                    help="rebuild results.json from committed inputs and byte-compare")
    ap.add_argument("--labeler-root", default=os.path.join(os.path.dirname(cc.REPO),
                                                           "sidewalk-auto-labeler"))
    ap.add_argument("--sites-dir", help="extract-neighbours: dir with <split>.sites.jsonl")
    ap.add_argument("--panos-root", help="extract-heatmaps: dir holding <split>/panos/*.jpg")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = os.path.join(OUT_DIR, RESULTS)
    if args.check:
        with open(path, encoding="utf-8", newline="") as f:
            same = f.read() == cc.dumps(build())
        print(f"{'same' if same else 'DIFF'}  {rel(path)}")
        sys.exit(0 if same else 1)
    if args.cmd == "extract-neighbours":
        if not args.sites_dir:
            raise SystemExit("--sites-dir is required")
        return extract_neighbours(args)
    if args.cmd == "extract-heatmaps":
        if not args.panos_root:
            raise SystemExit("--panos-root is required")
        return extract_heatmaps(args)
    if args.cmd == "run":
        res = build()
        cc.write_json(path, res)
        res = json.loads(cc.dumps(res))
        print_leak(res)
        print_filters(res)
        print(f"-> {path}")
        return
    res = read_json(path)
    {"leak": print_leak, "features": print_features, "filters": print_filters,
     "figures": figures}[args.cmd](res)


if __name__ == "__main__":
    main()
