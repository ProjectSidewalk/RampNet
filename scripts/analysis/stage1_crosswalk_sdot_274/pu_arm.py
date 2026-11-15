#!/usr/bin/env python3
"""Crosswalk supervision fix (`crosswalk_s7`): day-1 diagnostics D1, D2 and the D4 leak check.

Plan: `crosswalk-supervision-fix-2026-11-15.md`, results in `docs/crosswalk_supervision_s7.md`. Fix H turns the
1,705 auto-placed panos of `root_v6_full` into `dense_ignore` panos: background everywhere except ignore boxes over
every Vistas crosswalk component (dilated by the 30 heatmap px training-disc radius), ignore points at every
crosswalk_s4 peak >= 0.05 (any of three seeds) and at every projected OSM crossing node. A crosswalk that all three
miss becomes background. D4 measures how often that happens on human-marked crosswalks on panos s4 never trained on.

Subcommands:

``diagnose`` (CPU, committed predictions only)
    D1: detections (true and false together) per model, at recall-matched thresholds and at a fixed 0.25, on the
    fresh Juneau draw (three seeds) and, seed 1 only, on the confirmation draw's 200 Mapillary and 80 GSV panos and
    on Columbia. D2: on a threshold grid, per model, Juneau recall on Jon's 110 rated marks and guard 6's false
    alarms on the 119 confirmed-empty panos (rig-masked and not). Both exploratory.

``leak-check --population a`` (GPU for the Vistas pass, one Overpass query per city, then CPU)
    H's full rule on the 200 Mapillary panos of the confirmation `cities` draw; the marks are Jon's `crosswalk`
    verdicts on the cards of every rated model (s1, s3 on the confirmation page; s4 on Step B), merged at 0.01.

``leak-check --population b --jon-approved-b`` (same steps)
    The rule on Jon's missed-crosswalk marks (sure, within 25 m: 110 Juneau, 11 Laurens). **Not approved as of
    2026-11-15.** It refuses to run, and reads no mark, without the flag.

``figures``  D2's trade-off figure from the committed grid.

``--check``  Recomputes D1, D2 and leak (a)'s scoring from committed files and compares with the committed outputs.

Usage::

    S=scripts/analysis/stage1_crosswalk_sdot_274; K=D:/Git/sal-scratch-keep
    python $S/pu_arm.py diagnose
    python $S/pu_arm.py leak-check --population a --keep $K       # Vistas on the RTX 3070: about 15 minutes
    python $S/pu_arm.py figures
    python $S/pu_arm.py --check
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, os.path.join(REPO, "scripts", "analysis"), os.path.join(REPO, "scripts", "model_comparison"),
           os.path.join(REPO, "scripts", "analysis", "crosswalk_phase0_274"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recall_check as rc  # noqa: E402
from equirect_tiling import _camera_basis, default_views  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "crosswalk_supervision_s7")
HARDNEG = os.path.join(REPO, "analysis_out", "stage1_crosswalk_detector_274", "hardneg")
RECALL = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "recall")

# ---- H's rule, as the plan fixes it -------------------------------------------------------------------------
SHAPE = (512, 1024)          # train.py heatmap_output_shape
SIGMA = 10.0                 # train.py's target sigma; discs have radius ceil(3 * sigma) = 30 heatmap px
DISC_HM = int(math.ceil(3 * SIGMA))
BOX_DILATE_HM = DISC_HM      # the plan: Vistas boxes "dilated by 30 px" -- read as heatmap px (the disc radius)
S4_FLOOR = 0.05              # every stored s4 peak (score_274.SCORE_FLOOR)
OSM_RANGE_M = 20.0           # osm_arm.PLACE_RANGE_M, the placer's projection range
CAMERA_M = 2.6               # osm_arm.NON_GSV_CAMERA_M
MIN_AREA_PX = 16             # vistas_crosswalk_probe.MIN_AREA_PX, as neg-check
CLASS_IDS = (8, 23)          # Vistas "Crosswalk - Plain", "Lane Marking - Crosswalk"
MERGE_R = 0.01               # score_274.PREFILL_R: cards this close on one pano are one card
BLIND_R = 0.05               # recall_check.RADII[0]: "s4-blind" = no s4 peak >= S4_FLOOR within this of the mark
GATE_MAX_LEAK = 0.05

# ---- D1 / D2 -------------------------------------------------------------------------------------------------
# family -> [(label, matched threshold)] for seeds 1, 2, 3 (docs/crosswalk_followups_285.md, A4 result)
MODELS = {
    "s4": [("crosswalk_s4_klone", 0.23), ("crosswalk_s4_seed2", 0.30), ("crosswalk_s4_seed3", 0.13)],
    "full": [("crosswalk_s6full", 0.17), ("crosswalk_s6full_seed2", 0.27), ("crosswalk_s6full_seed3", 0.24)],
    "half": [("crosswalk_s6half", 0.23), ("crosswalk_s6half_seed2", 0.23), ("crosswalk_s6half_seed3", 0.25)],
    "steps": [("crosswalk_s4_steps1397", 0.24)],
}
FIXED_T = 0.25
D2_GRID = [round(0.10 + 0.01 * i, 2) for i in range(31)]     # 0.10 .. 0.40; starts above 0.05 on purpose (see doc)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_preds(path):
    return {r["pano_id"]: r["preds"] for r in read_jsonl(path)}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols or list(rows[0]), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def wilson(k, n, z=1.96):
    """Wilson 95% interval on k of n, rounded to 3 places.

    >>> wilson(0, 10)
    [0.0, 0.278]
    """
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - h) / d), 3), round(min(1.0, (c + h) / d), 3)]


def wrap_dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    return math.hypot(min(dx, 1.0 - dx), y1 - y2)


# =============================================================================================================
# D1 and D2
# =============================================================================================================

def n_dets(preds, t, ids=None):
    return sum(1 for pid, ps in preds.items() if ids is None or pid in ids for _, _, c in ps if c >= t)


def d1():
    """D1 as the plan reports it: every detection at or above the threshold, no rig mask, no verdicts."""
    je = os.path.join(HARDNEG, "s5_eval", "juneau_eval", "preds")
    je6 = os.path.join(HARDNEG, "s6_eval", "juneau_eval", "preds")
    path = {"crosswalk_s4_klone": os.path.join(je, "crosswalk_s4"), "crosswalk_s4_seed2": os.path.join(je, "crosswalk_s4_seed2"),
            "crosswalk_s4_seed3": os.path.join(je, "crosswalk_s4_seed3")}
    for fam in ("full", "half", "steps"):
        for lab, _ in MODELS[fam]:
            path[lab] = os.path.join(je6, lab)
    res = {"fresh_juneau_120": {}, "seed1_only": {}}
    for fam, specs in MODELS.items():
        rows = []
        for lab, t in specs:
            p = read_preds(os.path.join(path[lab], "juneau_eval.jsonl"))
            if len(p) != 120:
                raise SystemExit(f"{lab}: {len(p)} fresh Juneau panos, expected 120")
            rows.append({"label": lab, "threshold": t, "matched": n_dets(p, t), "fixed_0.25": n_dets(p, FIXED_T)})
        res["fresh_juneau_120"][fam] = rows
    cities = [l.strip()[:-4] for l in open(os.path.join(HARDNEG, "confirm_s3", "cities_draw.txt"), encoding="utf-8") if l.strip()]
    mly = {i for i in cities if i.isdigit()}
    gsv = set(cities) - mly
    conf = {"s4": os.path.join(HARDNEG, "confirm_s3", "preds", "crosswalk_s4", "cities.jsonl")}
    col = {"s4": os.path.join(HARDNEG, "s4_eval", "columbia", "preds", "crosswalk_s4", "columbia.jsonl")}
    for fam in ("full", "half", "steps"):
        lab = MODELS[fam][0][0]
        conf[fam] = os.path.join(HARDNEG, "s6_eval", "confirm", "preds", lab, "cities.jsonl")
        col[fam] = os.path.join(HARDNEG, "s6_eval", "columbia", "preds", lab, "columbia.jsonl")
    for fam in ("s4", "full", "half", "steps"):
        t = MODELS[fam][0][1]
        cp, cl = read_preds(conf[fam]), read_preds(col[fam])
        if set(cp) != set(cities):
            raise SystemExit(f"{fam}: confirmation predictions do not cover the 280 cities panos")
        res["seed1_only"][fam] = {"label": MODELS[fam][0][0], "threshold": t,
                                  "confirmation_mapillary_200": n_dets(cp, t, mly), "confirmation_gsv_80": n_dets(cp, t, gsv),
                                  "columbia_gsv": n_dets(cl, t), "columbia_panos": len(cl)}
    return res


def d2():
    """Per model and grid threshold: Juneau recall on the 110 marks (radius 0.05) and guard 6's false alarms."""
    verdicts = json.load(open(os.path.join(RECALL, "ratings", "recall285_verdicts_jon_2026-11-11.json"), encoding="utf-8"))["panos"]
    rows = []
    for fam, specs in MODELS.items():
        for seed, (lab, matched) in enumerate(specs, 1):
            for t in sorted(set(D2_GRID) | {matched}):
                marks, empty = rc._pano_stats(verdicts, ["laurens-ia", "juneau-ak"], lab, t, "juneau-ak")
                n, k = sum(v[0] for v in marks.values()), sum(v[1] for v in marks.values())
                rows.append({"family": fam, "seed": seed, "label": lab, "threshold": t, "matched": int(t == matched),
                             "in_grid": int(t in D2_GRID), "marks": n, "found": k, "recall": round(k / n, 4),
                             "fa_rig_masked": sum(v[2] for v in empty.values()),
                             "fa_unmasked": sum(v[1] for v in empty.values()), "empty_panos": len(empty)})
    if {r["marks"] for r in rows} != {110} or {r["empty_panos"] for r in rows} != {119}:
        raise SystemExit("D2: the populations are not 110 marks and 119 empty panos")
    summ = {}
    for fam, specs in MODELS.items():
        fr = [r for r in rows if r["family"] == fam]
        at = lambda key: [r for r in fr if r["matched"]] if key == "matched" else [r for r in fr if r["threshold"] == key]
        def mean(rs, col):
            return round(sum(r[col] for r in rs) / len(rs), 3)
        summ[fam] = {"seeds": len(specs),
                     "matched": {"found": [r["found"] for r in at("matched")], "fa_rig_masked": [r["fa_rig_masked"] for r in at("matched")],
                                 "mean_recall": mean(at("matched"), "recall"), "mean_fa_rig_masked": mean(at("matched"), "fa_rig_masked")},
                     "fixed_0.25": {"found": [r["found"] for r in at(0.25)], "fa_rig_masked": [r["fa_rig_masked"] for r in at(0.25)],
                                    "mean_recall": mean(at(0.25), "recall"), "mean_fa_rig_masked": mean(at(0.25), "fa_rig_masked")}}
        # the seed-mean curve, and the highest grid threshold whose mean guard-6 count is within s4's matched mean + 4
        curve = []
        for t in D2_GRID:
            rs = at(t)
            curve.append({"threshold": t, "mean_recall": mean(rs, "recall"), "mean_fa_rig_masked": mean(rs, "fa_rig_masked")})
        summ[fam]["mean_curve"] = curve
    lim = summ["s4"]["matched"]["mean_fa_rig_masked"] + 4
    for fam in MODELS:
        ok = [c for c in summ[fam]["mean_curve"] if c["mean_fa_rig_masked"] <= lim]
        best = max(ok, key=lambda c: (c["mean_recall"], -c["threshold"])) if ok else None
        summ[fam]["best_recall_within_guard6_mean"] = {"limit": round(lim, 3), **(best or {"threshold": None})}
    return rows, summ


def cmd_diagnose(a):
    out = os.path.join(a.out or OUT, "diagnose")
    res1 = d1()
    write_json(os.path.join(out, "d1.json"), res1)
    rows, summ = d2()
    write_csv(os.path.join(out, "d2_grid.csv"), rows)
    write_json(os.path.join(out, "d2.json"), {"grid": D2_GRID, "radius": rc.RADII[0], "rig_mask_y": round(rc.RIG_MASK_Y, 6),
                                              "families": summ})
    log(f"diagnose -> {out}")
    print(json.dumps(res1, indent=1))
    print(json.dumps({f: {k: v for k, v in s.items() if k != "mean_curve"} for f, s in summ.items()}, indent=1))


# =============================================================================================================
# H's rule: Vistas boxes, s4 points, OSM points -> the dense_ignore weight map (train.py's own code)
# =============================================================================================================

def persp_to_equirect(u, v, view):
    """Vectorised `equirect_tiling.perspective_point_to_equirect`: arrays of view (u, v) -> pano (x, y)."""
    import numpy as np
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a = (np.asarray(u, dtype=float) * 2.0 - 1.0) * th
    b = (1.0 - np.asarray(v, dtype=float) * 2.0) * tv
    dx, dy, dz = (f[i] + a * r[i] + b * up[i] for i in range(3))
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    lon = np.arctan2(dx / n, dz / n)
    lat = np.arcsin(np.clip(dy / n, -1.0, 1.0))
    return (lon / (2 * math.pi) + 0.5) % 1.0, np.clip(0.5 - lat / math.pi, 0.0, 1.0)


def span_x(xs):
    """(x_left, width) of the shortest arc covering pano x values (seam-aware; width < 1).

    >>> [round(v, 3) for v in span_x([0.98, 0.99, 0.01, 0.02])]
    [0.98, 0.04]
    """
    s = sorted(float(x) % 1.0 for x in xs)
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)] + [(s[0] + 1.0 - s[-1], len(s) - 1)]
    g, i = max(gaps)
    left = s[(i + 1) % len(s)]
    return left, 1.0 - g


def view_boxes(seg, view):
    """Each Vistas crosswalk component of one view (connectivity 2, area >= MIN_AREA_PX, as neg-check) as a pano
    box {xl, xw, yt, yb} (undilated; x from xl over width xw, may cross the seam) plus area and centroid."""
    import numpy as np
    from skimage.measure import label as cc_label, regionprops
    sel = np.isin(seg, list(CLASS_IDS))
    out = []
    if not sel.any():
        return out
    h, w = seg.shape
    for region in regionprops(cc_label(sel, connectivity=2)):
        if region.area < MIN_AREA_PX:
            continue
        rr, cc = region.coords[:, 0], region.coords[:, 1]
        xs, ys = persp_to_equirect((cc + 0.5) / w, (rr + 0.5) / h, view)
        xl, xw = span_x(xs)
        cy, cx = region.centroid
        x0, y0 = persp_to_equirect([(cx + 0.5) / w], [(cy + 0.5) / h], view)
        out.append({"view_yaw": view.yaw_deg, "area_px": int(region.area), "x": round(float(x0[0]), 6),
                    "y": round(float(y0[0]), 6), "xl": round(xl, 6), "xw": round(xw, 6),
                    "yt": round(float(ys.min()), 6), "yb": round(float(ys.max()), 6)})
    return out


def dilate_box(c, d_hm=BOX_DILATE_HM):
    """A component box grown by d_hm heatmap px on every side, in train.py's ignore_boxes format
    [x_left, x_right, y_top, y_bottom] (x_left > x_right wraps the seam).

    >>> dilate_box({"xl": 0.99, "xw": 0.02, "yt": 0.6, "yb": 0.62}, 0)
    [0.99, 0.01, 0.6, 0.62]
    """
    dx, dy = d_hm / SHAPE[1], d_hm / SHAPE[0]
    yt, yb = max(0.0, c["yt"] - dy), min(1.0, c["yb"] + dy)
    if c["xw"] + 2 * dx >= 1.0:
        return [0.0, 1.0, round(yt, 6), round(yb, 6)]
    xl, xr = (c["xl"] - dx) % 1.0, (c["xl"] + c["xw"] + dx) % 1.0
    return [round(xl, 6), round(xr, 6), round(yt, 6), round(yb, 6)]


_TRAIN = None


def train_fns():
    """train.py's own `generate_dense_ignore_weight` (lifted from the AST, as check_root_negatives.py does)."""
    global _TRAIN
    if _TRAIN is None:
        import numpy as np
        import torch
        from check_lr_schedule_135 import load_from_train_py
        _TRAIN = load_from_train_py("generate_dense_ignore_weight", "_nearest_dist_sq", math=math, np=np, torch=torch)
    return _TRAIN


def background_at(points, ignore_points, ignore_boxes):
    """For each (x, y): True when train.py's dense_ignore weight there is 1 (the pano has no positives here: a
    hold-out pano under H's rule). The probe pixel is check_root_negatives.py's (int, clamped)."""
    w = train_fns().generate_dense_ignore_weight([], heatmap_shape=SHAPE, sigma=SIGMA,
                                                 ignore_points=list(ignore_points), ignore_boxes=list(ignore_boxes))
    w = w.numpy()[0]
    out = []
    for x, y in points:
        r, c = min(int(y * SHAPE[0]), SHAPE[0] - 1), min(int(x * SHAPE[1]), SHAPE[1] - 1)
        out.append(bool(w[r, c] == 1))
    return out


# =============================================================================================================
# Populations
# =============================================================================================================

def merge_marks(card_rows, r=MERGE_R):
    """One card per paint from several rating exports (earliest first), as score_274.merge_candidates groups
    them: a card within r (seam-wrapped, pano widths) of an earlier card on the same pano joins it. Returns the
    merged cards, each with every member's verdict, and `verdict` = the verdict all members share, else 'conflict'.

    >>> rows = [{"pano_id": "p", "x": 0.5, "y": 0.6, "verdict": "crosswalk", "src": "a"},
    ...         {"pano_id": "p", "x": 0.505, "y": 0.6, "verdict": "crosswalk", "src": "b"},
    ...         {"pano_id": "p", "x": 0.7, "y": 0.6, "verdict": "not_crosswalk", "src": "b"}]
    >>> [(m["x"], m["verdict"], m["sources"]) for m in merge_marks(rows)]
    [(0.5, 'crosswalk', 'a;b'), (0.7, 'not_crosswalk', 'b')]
    """
    out = []
    for c in card_rows:
        near = [u for u in out if u["pano_id"] == c["pano_id"] and wrap_dist(u["x"], u["y"], c["x"], c["y"]) <= r]
        if near:
            u = min(near, key=lambda u: wrap_dist(u["x"], u["y"], c["x"], c["y"]))
            u["verdicts"].append(c["verdict"])
            if c["src"] not in u["sources"].split(";"):
                u["sources"] += ";" + c["src"]
        else:
            out.append({"pano_id": c["pano_id"], "x": c["x"], "y": c["y"], "verdicts": [c["verdict"]], "sources": c["src"]})
    for u in out:
        u["verdict"] = u["verdicts"][0] if len(set(u["verdicts"])) == 1 else "conflict"
    return out


RATING_FILES_A = [("confirm", os.path.join(HARDNEG, "confirm_s3", "page", "ratings_jon_2026-11-08_confirm.csv")),
                  ("s4_stepB", os.path.join(HARDNEG, "s4_stepB", "ratings_jon_2026-11-09_s4_stepB.csv")),
                  ("s5_stepB", os.path.join(HARDNEG, "s5_stepB", "ratings_jon_2026-11-10_s5_stepB_fixed.csv"))]


def population_a(keep):
    """The 200 Mapillary panos of the confirmation cities draw, their records and image paths, and the marks."""
    ids = [l.strip()[:-4] for l in open(os.path.join(OUT, "leak_a", "panos.txt"), encoding="utf-8") if l.strip()]
    rows = []
    for src, path in RATING_FILES_A:
        for r in read_csv(path):
            if r["pano_id"] in ids and r.get("verdict"):
                rows.append({"pano_id": r["pano_id"], "x": float(r["x"]), "y": float(r["y"]),
                             "verdict": r["verdict"], "src": src, "card_id": r["card_id"]})
    merged = merge_marks(rows)
    recs = {}
    if keep:
        for f in ("juneau_draw_records.jsonl", "laurens_mly_draw_records.jsonl"):
            city = "juneau-ak" if f.startswith("juneau") else "laurens-ia"
            for r in read_jsonl(os.path.join(keep, "crosswalk274-confirm", f)):
                p = r["pano"]
                recs[str(p["panorama_id"])] = {**p, "city": city}
    s4 = [os.path.join(HARDNEG, "confirm_s3", "preds", "crosswalk_s4", "cities.jsonl"),
          os.path.join(OUT, "preds", "crosswalk_s4_seed2", "cities_mly.jsonl"),
          os.path.join(OUT, "preds", "crosswalk_s4_seed3", "cities_mly.jsonl")]
    return {"name": "a", "ids": ids, "card_rows": rows, "merged": merged,
            "marks": [m for m in merged if m["verdict"] == "crosswalk"], "records": recs,
            "image": lambda pid: os.path.join(keep, "crosswalk274-confirm", "prep_4096", pid + ".jpg"), "s4_preds": s4,
            "inputs": [p for _, p in RATING_FILES_A] + s4}


def population_b(keep):
    """Jon's missed-crosswalk marks: sure, within 25 m, both cities (110 Juneau, 11 Laurens). Pending Jon's OK."""
    vpath = os.path.join(RECALL, "ratings", "recall285_verdicts_jon_2026-11-11.json")
    v = json.load(open(vpath, encoding="utf-8"))["panos"]
    recs, city_of = {}, {}
    for city in ("juneau-ak", "laurens-ia"):
        for r in read_jsonl(os.path.join(keep, "recall285", f"{city}_draw_records.jsonl")):
            p = r["pano"]
            recs[str(p["panorama_id"])] = {**p, "city": city}
        for r in read_csv(os.path.join(RECALL, city, "intersections.csv")):
            if r["in_draw"] == "1":
                city_of[r["pano_id"]] = city
    marks = []
    for pid in sorted(v):
        if pid not in city_of:
            continue
        for m in v[pid].get("missed", []):
            if not m.get("unsure") and m["y"] >= rc.Y_25M:
                marks.append({"pano_id": pid, "x": m["x"], "y": m["y"], "verdict": "crosswalk", "sources": "recall285",
                              "city": city_of[pid]})
    ids = sorted({m["pano_id"] for m in marks})
    s4 = [os.path.join(RECALL, "preds", lab, f"{c}.jsonl")
          for lab in ("crosswalk_s4_klone", "crosswalk_s4_seed2", "crosswalk_s4_seed3") for c in ("juneau-ak", "laurens-ia")]
    return {"name": "b", "ids": ids, "card_rows": [], "merged": marks, "marks": marks, "records": recs,
            "image": lambda pid: os.path.join(keep, "recall285", "prep_4096", pid + ".jpg"), "s4_preds": s4,
            "inputs": [vpath] + s4}


# =============================================================================================================
# Inputs that need the keep dir, the GPU or the network
# =============================================================================================================

def vistas_pass(pop, path):
    """Whole-pano six-view Vistas pass (neg-check's views) on every pano of the population; resumable JSONL."""
    import vistas_crosswalk_probe as probe
    import crowd_arm
    from equirect_tiling import equirect_to_perspective
    done = {r["pano_id"] for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [p for p in pop["ids"] if p not in done]
    if not todo:
        return
    seg = probe.Segmenter()
    log(f"vistas: {len(todo)} panos to go ({seg.backend}, revision {seg.revision})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for k, pid in enumerate(todo, 1):
            img = crowd_arm.load_pano(pop["image"](pid))
            comps = []
            for view in default_views():
                s, _ = seg.seg_prob(equirect_to_perspective(img, view))
                comps += view_boxes(s, view)
            fh.write(json.dumps({"pano_id": pid, "size": list(img.size), "backend": seg.backend,
                                 "revision": seg.revision, "components": comps}) + "\n")
            fh.flush()
            if k % 20 == 0:
                log(f"  vistas {k}/{len(todo)}")


def osm_fetch(records, path, endpoint=None):
    """Every highway=crossing node in the records' bounding box (padded 0.001 deg), plus the highway ways through
    them (for grade separation). One Overpass query, cached."""
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    import osm_city
    if endpoint:
        osm_city.OVERPASS = endpoint
    la = [p["lat"] for p in records]
    ln = [p["lng"] for p in records]
    s, w, n, e = min(la) - 0.001, min(ln) - 0.001, max(la) + 0.001, max(ln) + 0.001
    q = (f'[out:json][timeout:120];node["highway"="crossing"]({s:.6f},{w:.6f},{n:.6f},{e:.6f})->.c;'
         '.c out body;way(bn.c)["highway"];out body;')
    log(f"overpass: crossings in {s:.4f},{w:.4f},{n:.4f},{e:.4f}")
    data = osm_city.overpass(q)
    data["query"] = q
    data["endpoint"] = osm_city.OVERPASS
    write_json(path, data)
    return data


def osm_points(pop, osm_by_city):
    """Every crossing node projected into every pano (B1, 2.6 m camera), with range, class and grade."""
    import osm_city
    import pilot
    rows = []
    for city, data in osm_by_city.items():
        sep = set()
        for e in data["elements"]:
            if e["type"] == "way":
                t = e.get("tags") or {}
                if t.get("tunnel", "no") != "no" or t.get("bridge", "no") != "no" or t.get("layer", "0") not in ("0", ""):
                    sep.update(e["nodes"])
        nodes = [(e["lat"], e["lon"], e["id"], osm_city.osm_class(e.get("tags") or {})[0])
                 for e in data["elements"] if e["type"] == "node" and (e.get("tags") or {}).get("highway") == "crossing"]
        for pid in pop["ids"]:
            p = pop["records"][pid]
            if p["city"] != city:
                continue
            for lat, lng, nid, cls in nodes:
                if abs(lat - p["lat"]) > 0.0005:
                    continue
                x, y, d, _ = pilot.project(p["lat"], p["lng"], p["camera_heading"], lat, lng, camera_h=CAMERA_M)
                if d <= 25.0:
                    rows.append({"pano_id": pid, "city": city, "node_id": nid, "osm_class": cls,
                                 "grade_separated": int(nid in sep), "range_m": round(d, 2),
                                 "x": round(x, 6), "y": round(y, 6)})
    return rows


# =============================================================================================================
# Scoring (CPU, committed files only)
# =============================================================================================================

RULES = {   # name -> (osm range m, osm classes or None for all, box dilation in heatmap px); "primary" gates
    "primary": (OSM_RANGE_M, None, BOX_DILATE_HM),
    "osm_marked_only": (OSM_RANGE_M, {"marked"}, BOX_DILATE_HM),
    "osm_25m": (25.0, None, BOX_DILATE_HM),
    "box_dilate_30_image_px": (OSM_RANGE_M, None, 30 * SHAPE[1] / 4096),
    "no_box_dilation": (OSM_RANGE_M, None, 0),
}


def score(marks, ids, s4_paths, vistas_rows, osm_rows):
    """Per mark: background or not under each rule, which source covers it, and whether it is s4-blind."""
    s4 = [{} for _ in s4_paths]
    for i, paths in enumerate(s4_paths):            # one path per seed, or a list of that seed's files
        for p in ([paths] if isinstance(paths, str) else paths):
            for pid, ps in read_preds(p).items():
                s4[i][pid] = [(x, y) for x, y, c in ps if c >= S4_FLOOR]
    for i, d in enumerate(s4):
        miss = [pid for pid in ids if pid not in d]
        if miss:
            raise SystemExit(f"s4 seed {i + 1}: no predictions for {len(miss)} panos, e.g. {miss[:3]}")
    vis = {r["pano_id"]: r["components"] for r in vistas_rows}
    if set(ids) - set(vis):
        raise SystemExit(f"vistas: {len(set(ids) - set(vis))} panos missing")
    osm = defaultdict(list)
    for r in osm_rows:
        osm[r["pano_id"]].append(r)
    by_pano = defaultdict(list)
    for i, m in enumerate(marks):
        by_pano[m["pano_id"]].append(i)
    out = [dict(m) for m in marks]
    for pid, idx in by_pano.items():
        pts = [(marks[i]["x"], marks[i]["y"]) for i in idx]
        s4pts = [q for d in s4 for q in d[pid]]
        for name, (rng, classes, dil) in RULES.items():
            op = [(float(r["x"]), float(r["y"])) for r in osm[pid]
                  if float(r["range_m"]) <= rng and (classes is None or r["osm_class"] in classes)]
            boxes = [dilate_box(c, dil) for c in vis[pid]]
            bg = background_at(pts, s4pts + op, boxes)
            for i, b in zip(idx, bg):
                out[i][f"bg_{name}"] = int(b)
            if name == "primary":
                cover = {"s4": background_at(pts, s4pts, []), "osm": background_at(pts, op, []),
                         "vistas": background_at(pts, [], boxes)}
                for k, i in enumerate(idx):
                    for src, bgs in cover.items():
                        out[i][f"covered_{src}"] = int(not bgs[k])
        for i in idx:
            mx, my = marks[i]["x"], marks[i]["y"]
            near = [any(wrap_dist(mx, my, x, y) <= BLIND_R for x, y in d[pid]) for d in s4]
            out[i]["s4_peak_within_0.05_seed1"] = int(near[0])
            out[i]["s4_blind_0.05"] = int(not any(near))
    return out


def summarise(pop_name, scored, extra):
    n = len(scored)
    res = {"population": pop_name, "marks": n, "panos_with_marks": len({m["pano_id"] for m in scored}),
           "gate_max_leak": GATE_MAX_LEAK, "rules": {}}
    for name in RULES:
        k = sum(m[f"bg_{name}"] for m in scored)
        res["rules"][name] = {"leaked": k, "leak": round(k / n, 4) if n else None, "wilson95": wilson(k, n),
                              "osm_range_m": RULES[name][0],
                              "osm_classes": sorted(RULES[name][1]) if RULES[name][1] else "all",
                              "box_dilate_heatmap_px": RULES[name][2]}
    blind = sum(m["s4_blind_0.05"] for m in scored)
    res["s4_blind_at_0.05"] = {"k": blind, "share": round(blind / n, 4) if n else None, "wilson95": wilson(blind, n),
                               "rule": "no peak >= 0.05 from any of the three s4 seeds within 0.05 (recall_check radius)"}
    s1 = sum(1 - m["s4_peak_within_0.05_seed1"] for m in scored)
    res["s4_seed1_blind_at_0.05"] = {"k": s1, "share": round(s1 / n, 4) if n else None}
    res["covered_by"] = {src: sum(m[f"covered_{src}"] for m in scored) for src in ("s4", "osm", "vistas")}
    res["covered_by_vistas_only"] = sum(m["covered_vistas"] and not m["covered_s4"] and not m["covered_osm"] for m in scored)
    res["covered_by_osm_only"] = sum(m["covered_osm"] and not m["covered_s4"] and not m["covered_vistas"] for m in scored)
    res["leaked_among_s4_blind"] = sum(m["bg_primary"] for m in scored if m["s4_blind_0.05"])
    p = res["rules"]["primary"]
    res["gate_half"] = {"verdict": "holds" if p["leak"] is not None and p["leak"] <= GATE_MAX_LEAK else "fails",
                        "rule": f"leak <= {GATE_MAX_LEAK} on the primary rule"}
    res.update(extra)
    return res


def cmd_leak_check(a):
    if a.population == "b" and not a.jon_approved_b:
        raise SystemExit("population (b) uses Jon's missed-crosswalk marks; that use is Jon's decision and is not "
                         "approved. Pass --jon-approved-b only after he approves it.")
    out = os.path.join(OUT, f"leak_{a.population}")
    pop = population_a(a.keep) if a.population == "a" else population_b(a.keep)
    if not pop["records"]:
        raise SystemExit("--keep is needed (records and images)")
    missing = [p for p in pop["ids"] if p not in pop["records"]]
    if missing:
        raise SystemExit(f"{len(missing)} panos have no record, e.g. {missing[:3]}")
    if a.population == "b":
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "panos.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(p + ".jpg\n" for p in pop["ids"]))
    vpath = os.path.join(out, "vistas.jsonl")
    vistas_pass(pop, vpath)
    osm_by_city = {}
    for city in sorted({pop["records"][p]["city"] for p in pop["ids"]}):
        recs = [pop["records"][p] for p in pop["ids"] if pop["records"][p]["city"] == city]
        osm_by_city[city] = osm_fetch(recs, os.path.join(out, f"osm_{city}.json"), a.overpass)
    orows = osm_points(pop, osm_by_city)
    write_csv(os.path.join(out, "osm_points.csv"), orows or [{}],
              ["pano_id", "city", "node_id", "osm_class", "grade_separated", "range_m", "x", "y"])
    marks_cols = ["pano_id", "x", "y", "verdict", "sources"]
    write_csv(os.path.join(out, "marks.csv"), pop["marks"], marks_cols + (["city"] if a.population == "b" else []))
    if a.population == "a":
        write_csv(os.path.join(out, "merged_cards.csv"),
                  [{**m, "verdicts": ";".join(m["verdicts"])} for m in pop["merged"]],
                  ["pano_id", "x", "y", "verdict", "verdicts", "sources"])
    finish(a.population, out, pop)


def finish(pop_name, out, pop=None, dest=None):
    """Score from the files written above and write marks_scored.csv and summary.json to `dest` (default `out`).
    --check passes a scratch `dest`, so a check writes nothing in the repo."""
    dest = dest or out
    ids = [l.strip()[:-4] for l in open(os.path.join(out, "panos.txt"), encoding="utf-8") if l.strip()]
    marks = [{**m, "x": float(m["x"]), "y": float(m["y"])} for m in read_csv(os.path.join(out, "marks.csv"))]
    if pop_name == "a":
        s4_paths = population_a(None)["s4_preds"]
    else:                                     # one entry per seed: that seed's two city files
        s4_paths = [[os.path.join(RECALL, "preds", lab, f"{c}.jsonl") for c in ("juneau-ak", "laurens-ia")]
                    for lab in ("crosswalk_s4_klone", "crosswalk_s4_seed2", "crosswalk_s4_seed3")]
    vistas_rows = read_jsonl(os.path.join(out, "vistas.jsonl"))
    orows = [r for r in read_csv(os.path.join(out, "osm_points.csv")) if r.get("pano_id")]
    scored = score(marks, ids, s4_paths, vistas_rows, orows)
    write_csv(os.path.join(dest, "marks_scored.csv"), scored)
    flat = [q for p in s4_paths for q in ([p] if isinstance(p, str) else p)]
    extra = {"panos": len(ids), "osm_points_within_25m": len(orows),
             "vistas_components": sum(len(r["components"]) for r in vistas_rows),
             "vistas_panos_with_a_component": sum(1 for r in vistas_rows if r["components"]),
             "inputs_sha256": {os.path.relpath(p, REPO).replace(os.sep, "/"): sha256(p)[:12]
                               for p in flat + [os.path.join(out, n) for n in ("panos.txt", "marks.csv", "vistas.jsonl",
                                                                               "osm_points.csv")]}}
    if pop_name == "a":
        mc = read_csv(os.path.join(out, "merged_cards.csv"))
        extra["merged_cards"] = dict(Counter(r["verdict"] for r in mc))
        extra["rating_files"] = {os.path.relpath(p, REPO).replace(os.sep, "/"): sha256(p)[:12] for _, p in RATING_FILES_A}
        jids = juneau_ids()
        extra["marks_by_city"] = dict(Counter("juneau-ak" if m["pano_id"] in jids else "laurens-ia" for m in marks))
        extra["panos_by_city"] = dict(Counter("juneau-ak" if p in jids else "laurens-ia" for p in ids))
    res = summarise(pop_name, scored, extra)
    write_json(os.path.join(dest, "summary.json"), res)
    print(json.dumps(res, indent=1))
    return res


def juneau_ids():
    return {l.strip()[:-4] if l.strip().endswith(".jpg") else l.strip()
            for l in open(os.path.join(HARDNEG, "confirm_s3", "juneau", "draw.txt"), encoding="utf-8") if l.strip()}


# =============================================================================================================
# figures and --check
# =============================================================================================================

def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(os.path.join(OUT, "diagnose", "d2_grid.csv"))
    # the dataviz skill's reference categorical slots 1-4 in fixed order, plus a line style so colour is not alone
    colors = {"s4": "#2a78d6", "full": "#eb6834", "half": "#1baf7a", "steps": "#eda100"}
    styles = {"s4": "-", "full": "--", "half": "-.", "steps": ":"}
    names = {"s4": "crosswalk_s4", "full": "s6 full dose", "half": "s6 half dose", "steps": "step control (s4 data, 1,397 steps)"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for fam in MODELS:
        for seed in sorted({r["seed"] for r in rows if r["family"] == fam}):
            rs = sorted([r for r in rows if r["family"] == fam and int(r["seed"]) == int(seed) and r["in_grid"] == "1"],
                        key=lambda r: float(r["threshold"]))
            ax.plot([int(r["fa_rig_masked"]) for r in rs], [float(r["recall"]) for r in rs], color=colors[fam], ls=styles[fam],
                    lw=2, alpha=0.9, label=names[fam] if seed == "1" else None)
            m = [r for r in rows if r["family"] == fam and r["seed"] == seed and r["matched"] == "1"][0]
            ax.plot(int(m["fa_rig_masked"]), float(m["recall"]), "o", color=colors[fam], ms=5)
    lim = 6.67 + 4
    ax.axvline(lim, color="#333", ls="--", lw=0.8)
    ax.text(lim * 1.05, 0.88, "guard 6 limit on the\nthree-seed mean (10.7)", fontsize=8, color="#333", va="top")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, 400)
    ax.set_xlabel("False alarms on the 119 confirmed-empty panos, rig-masked (guard 6), one seed")
    ax.set_ylabel("Juneau recall, 110 marks (radius 0.05)")
    ax.set_title("D2: recall against guard 6 as the threshold moves (0.10 to 0.40)", fontsize=10)
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.text(0.01, 0.005, "One line per seed; dots are each seed's recall-matched threshold. Exploratory.", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.join(REPO, "docs", "figures", "crosswalk_supervision_s7"), exist_ok=True)
    p = os.path.join(REPO, "docs", "figures", "crosswalk_supervision_s7", "d2_tradeoff.png")
    fig.savefig(p)
    log(f"figure -> {p}")


def cmd_check(_a):
    """Recompute D1, D2 and leak (a)'s scoring from committed files into a scratch dir; compare with the
    committed outputs. Writes nothing in the repo."""
    import shutil
    import tempfile
    bad = []
    tmp = tempfile.mkdtemp(prefix="pu_arm_check_")
    try:
        committed = os.path.join(OUT, "diagnose")
        if json.load(open(os.path.join(committed, "d1.json"), encoding="utf-8")) != json.loads(json.dumps(d1())):
            bad.append("diagnose/d1.json")
        rows, summ = d2()
        write_csv(os.path.join(tmp, "d2_grid.csv"), rows)
        write_json(os.path.join(tmp, "d2.json"), {"grid": D2_GRID, "radius": rc.RADII[0],
                                                  "rig_mask_y": round(rc.RIG_MASK_Y, 6), "families": summ})
        out = os.path.join(OUT, "leak_a")
        finish("a", out, dest=tmp)
        for got, ref in (("d2_grid.csv", "diagnose/d2_grid.csv"), ("d2.json", "diagnose/d2.json"),
                         ("summary.json", "leak_a/summary.json"), ("marks_scored.csv", "leak_a/marks_scored.csv")):
            if open(os.path.join(tmp, got), encoding="utf-8").read() != open(os.path.join(OUT, ref), encoding="utf-8").read():
                bad.append(ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(f"--check: differs from the committed file: {bad}")
    log("--check: D1, D2 and leak (a) reproduce the committed outputs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="recompute from committed files and compare; writes nothing")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("diagnose")
    d.add_argument("--out", default=None)
    lk = sub.add_parser("leak-check")
    lk.add_argument("--population", choices=("a", "b"), required=True)
    lk.add_argument("--keep", required=True, help="local scratch root (records, 4096 px panos)")
    lk.add_argument("--jon-approved-b", action="store_true",
                    help="required for population (b): Jon has approved using his missed-crosswalk marks")
    lk.add_argument("--overpass", default=None, help="Overpass interpreter URL (default: osm_city.OVERPASS)")
    sub.add_parser("figures")
    a = ap.parse_args()
    if a.check:
        return cmd_check(a)
    if not a.cmd:
        ap.error("a subcommand or --check is required")
    {"diagnose": cmd_diagnose, "leak-check": cmd_leak_check, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
