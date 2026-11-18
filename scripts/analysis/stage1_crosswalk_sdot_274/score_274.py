"""Score the first crosswalk detector exactly as pre-registered (RampNet#274).

The protocol is fixed in `docs/stage1_crosswalk_sdot_274.md`, "First crosswalk detector: training set and
pre-registered evaluation", written before any checkpoint existed. The choices that text left open are fixed in
the same doc's "Scoring implementation" subsection, committed before the first score was read. Constants below
carry those choices; changing one is an amendment and goes in that doc.

  infer  (GPU) one forward pass per pano of the Stage 2 root's val and test splits, peaks down to a floor
         -> <preds>/<label>/<split>.jsonl  (resumable; one line per pano)
  score  (CPU) (a) pixel P/R with the val-F1-max threshold applied to test, overall and per city / supervision
         mode / capture year, for every label; (b) per-OSM-node F1 on Seattle's test panos against the map-only
         baseline -> report.md, scores.json, threshold_sweep.csv, pixel_breakdown.csv, nodes.csv
  figures (CPU) PR / F1-vs-threshold curves and the per-city bars, from scores.json + the preds
  gallery / cards (CPU) local rating pages (test items / any detection list); bprime-audit the hard-negative
         plan's Step 0 census of one model's detections on drawn (b') panos

  python score_274.py infer --root <Stage 2 root> --checkpoint final_step_1000.pth --label crosswalk_s1 \
      --preds <dir> [--sha256 <expected>]
  python score_274.py infer --root <root> --checkpoint released_rampnet_state_dict.pth --label released ...
  python score_274.py score --root <root> --preds <dir> --labels crosswalk_s1,released \
      --baseline-nodes <crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274/baseline_nodes.csv \
      --census <Phase 0 census out> --meta-dir analysis_out/stage1_crosswalk_crowd_274 --out <out>
  python score_274.py figures --preds <dir> --out <out>

The peaks are the frozen benchmark extractor's (`operating_point_curve.py extract`, via threshold_sweep): resize
to 2048x4096, ImageNet normalisation, no TTA, fp32, `peak_local_max(min_distance=10, exclude_border=False)` down
to SCORE_FLOOR, confidence = the clipped heatmap at the peak.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, os.path.join(REPO, "scripts", "analysis"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rampnet.geometry import dist_sq  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

SPLITS = ("val", "test")
SCORE_FLOOR = 0.05            # the frozen extractor's floor; the threshold grid starts here
MIN_DISTANCE = 10
NEG_DISC = 30 / 1024           # a negative point's training disc (train.py add_negative_weight), pano widths
RADIUS = 0.05                 # Phase 0's crosswalk radius, RampNet matcher units (pano widths, y counts half)
SCALE_X, SCALE_Y = 1024, 512
RADIUS_SQ = (RADIUS * SCALE_X) ** 2
GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]   # 0.05 .. 0.95
COMPLETE = ("dense_ignore", "verified_empty")
YEAR_BINS = (("<=2018", None, 2018), ("2019-2022", 2019, 2022), ("2023+", 2023, None))
# (b) per-node geometry: the pilot's B1, inverted
CAMERA_H = 2.5
MAX_RANGE_M = 25.0
NODE_NEAR_PANO_M = 25.0
NODE_MATCH_M = 10.0
SEATTLE = "seattle-wa"
BOOT_N, BOOT_SEED = 2000, 274


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def load_split(root, split):
    """pano_id -> the pano's Stage 2 JSON, for one split, sorted by id."""
    d = os.path.join(root, split)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                j = json.load(f)
            out[j["pano_id"]] = j
    return out


def year_bin(capture_date):
    """'2021-08' -> '2019-2022'; missing or unparseable -> 'unknown'.

    >>> [year_bin(d) for d in ("2011-07", "2018-12", "2019-01", "2022-10", "2023-02", None)]
    ['<=2018', '<=2018', '2019-2022', '2019-2022', '2023+', 'unknown']
    """
    try:
        y = int(str(capture_date)[:4])
    except (TypeError, ValueError):
        return "unknown"
    for name, lo, hi in YEAR_BINS:
        if (lo is None or y >= lo) and (hi is None or y <= hi):
            return name
    return "unknown"


def in_box(x, y, box):
    """Inside an ignore box [x_left, x_right, y_top, y_bottom]; x_left > x_right wraps the seam.

    >>> in_box(0.5, 0.6, [0.4, 0.6, 0.5, 0.7]), in_box(0.99, 0.6, [0.9, 0.1, 0.5, 0.7]), in_box(0.5, 0.6, [0.9, 0.1, 0.5, 0.7])
    (True, True, False)
    """
    xl, xr, yt, yb = box
    if not (yt <= y <= yb):
        return False
    return xl <= x <= xr if xl <= xr else (x >= xl or x <= xr)


def classify_pano(preds, ann):
    """Each prediction of one pano -> (confidence, kind), kind in tp / fp / ignored / unscored.

    Greedy one-to-one in confidence order at RADIUS with seam wrap (rampnet.metrics.greedy_match, as
    detection_eval.score_pano does). GT takes priority. An unmatched prediction on a complete pano
    (dense_ignore, verified_empty) is `ignored` when it lies within RADIUS of an ignore point or inside an
    ignore box, else `fp`; on a `sparse` pano it is `unscored` (its background is unverified). Because the
    matching is greedy in descending confidence, the classification of the predictions at or above any
    threshold is exactly this list's prefix, so a threshold sweep never re-matches.

    >>> ann = {"curb_ramp_points_normalized": [[0.5, 0.6]], "supervision": {"mode": "dense_ignore",
    ...        "ignore_points": [[0.2, 0.6]], "ignore_boxes": [[0.7, 0.8, 0.5, 0.7]]}}
    >>> classify_pano([[0.51, 0.6, 0.9], [0.5, 0.6, 0.8], [0.21, 0.6, 0.7], [0.75, 0.6, 0.6], [0.9, 0.9, 0.5]], ann)
    [(0.9, 'tp'), (0.8, 'fp'), (0.7, 'ignored'), (0.6, 'ignored'), (0.5, 'fp')]
    >>> classify_pano([[0.9, 0.9, 0.5]], {"curb_ramp_points_normalized": [[0.5, 0.6]],
    ...                                   "supervision": {"mode": "sparse"}})
    [(0.5, 'unscored')]
    >>> classify_pano([[0.999, 0.6, 0.5]], {"curb_ramp_points_normalized": [[0.002, 0.6]],
    ...                                     "supervision": {"mode": "sparse"}})
    [(0.5, 'tp')]
    """
    sup = ann.get("supervision") or {}
    mode = sup.get("mode")
    gt = [tuple(p) for p in ann.get("curb_ramp_points_normalized", [])]
    ign = [tuple(p) for p in sup.get("ignore_points", [])]
    boxes = sup.get("ignore_boxes", []) if mode == "dense_ignore" else []
    order = sorted(preds, key=lambda p: -p[2])
    assign = greedy_match([(p[0], p[1]) for p in order], gt, RADIUS_SQ, SCALE_X, SCALE_Y, wrap_x=True)
    out = []
    for p, (k, _) in zip(order, assign):
        if k >= 0:
            kind = "tp"
        elif mode not in COMPLETE:
            kind = "unscored"
        elif any(dist_sq(p[0], p[1], ix, iy, SCALE_X, SCALE_Y, True) < RADIUS_SQ for ix, iy in ign) or \
                any(in_box(p[0], p[1], b) for b in boxes):
            kind = "ignored"
        else:
            kind = "fp"
        out.append((p[2], kind))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def counts_at(rows, t):
    """rows: [(ann, classified)] -> tp_all, tp_complete, fp, ignored, unscored, n_gt at threshold t."""
    c = {"tp": 0, "tp_complete": 0, "fp": 0, "ignored": 0, "unscored": 0, "n_gt": 0}
    for ann, cl in rows:
        c["n_gt"] += len(ann.get("curb_ramp_points_normalized", []))
        complete = (ann.get("supervision") or {}).get("mode") in COMPLETE
        for conf, kind in cl:
            if conf < t:
                break                                  # classified is in descending confidence
            c[kind] += 1
            if kind == "tp" and complete:
                c["tp_complete"] += 1
    return c


def metrics_from(c):
    n_prec = c["tp_complete"] + c["fp"]
    prec = c["tp_complete"] / n_prec if n_prec else float("nan")
    rec = c["tp"] / c["n_gt"] if c["n_gt"] else float("nan")
    ff = f1(prec, rec) if not (math.isnan(prec) or math.isnan(rec)) else float("nan")
    return {**c, "precision": prec, "recall": rec, "f1": ff,
            "precision_ci": wilson(c["tp_complete"], n_prec), "recall_ci": wilson(c["tp"], c["n_gt"])}


def pick_threshold(rows):
    """The grid threshold maximising F1 on these rows (val); ties go to the lowest threshold."""
    best = None
    sweep = []
    for t in GRID:
        m = metrics_from(counts_at(rows, t))
        sweep.append((t, m))
        if not math.isnan(m["f1"]) and (best is None or m["f1"] > best[1]["f1"]):
            best = (t, m)
    return best[0], sweep


def groups_of(ann):
    sup = ann.get("supervision") or {}
    return {"overall": "all", "city": sup.get("city", "?"), "mode": sup.get("mode", "?"),
            "year": year_bin(ann.get("capture_date"))}


# --------------------------------------------------------------------------- #
# infer (GPU)
# --------------------------------------------------------------------------- #
def cmd_infer(a):
    import torch
    import threshold_sweep as ts
    from rampnet.loading import load_checkpoint
    from rampnet.model import KeypointModel

    digest = sha256(a.checkpoint)
    if a.sha256 and digest != a.sha256:
        raise SystemExit(f"{a.checkpoint}: sha256 {digest} != expected {a.sha256}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointModel()
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = model.eval().to(device)
    out_dir = os.path.join(a.preds, a.label)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"label": a.label, "checkpoint": os.path.abspath(a.checkpoint), "sha256": digest,
            "score_floor": SCORE_FLOOR, "min_distance": MIN_DISTANCE, "tta": False, "fp16": False,
            "extractor": "threshold_sweep.heatmap_for + peaks_to_dets", "root": os.path.abspath(a.root),
            "device": device.type, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    mp = os.path.join(out_dir, "meta.json")
    if os.path.exists(mp):
        old = json.load(open(mp, encoding="utf-8"))
        if old.get("sha256") != digest:
            raise SystemExit(f"{mp} records sha256 {old.get('sha256')}, this checkpoint is {digest}: "
                             f"refusing to mix two models' predictions under one label")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(f"{a.label}: sha256 {digest[:12]} on {meta['gpu'] or device.type}")
    if a.ids_file:      # (b'): a flat dir of bprime-prep JPEGs, one "split"
        ids = sorted(l.strip().split("/")[-1][:-4] for l in open(a.ids_file, encoding="utf-8") if l.strip())
        jobs = [(a.split_name, {pid: os.path.join(a.pano_dir, f"{pid}.jpg") for pid in ids})]
    else:
        jobs = [(split, {pid: os.path.join(a.root, split, f"{pid}.jpg") for pid in load_split(a.root, split)})
                for split in a.splits.split(",")]
    for split, anns in jobs:
        path = os.path.join(out_dir, f"{split}.jsonl")
        done = {r["pano_id"] for r in read_jsonl(path)}
        todo = [pid for pid in anns if pid not in done]
        log(f"{split}: {len(anns)} panos, {len(done)} done, {len(todo)} to go")
        t0 = time.time()
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for i, pid in enumerate(todo, 1):
                h = ts.heatmap_for(model, device, anns[pid], False)
                preds = [[round(x, 6), round(y, 6), round(c, 6)]
                         for x, y, c in ts.peaks_to_dets(h, SCORE_FLOOR, MIN_DISTANCE)]
                f.write(json.dumps({"pano_id": pid, "preds": preds}) + "\n")
                f.flush()
                if i % 100 == 0:
                    log(f"  {split} {i}/{len(todo)} ({(time.time() - t0) / i:.2f} s/pano)")
        got = read_jsonl(path)
        if {r["pano_id"] for r in got} != set(anns) or len(got) != len(anns):
            raise SystemExit(f"{path}: {len(got)} lines for {len(anns)} panos -- incomplete or duplicated")
        log(f"{split}: complete, {len(got)} panos -> {path}")


# --------------------------------------------------------------------------- #
# (b) per OSM node
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def ground_point(pano_lat, pano_lng, heading_deg, x, y, camera_h=CAMERA_H, max_range=MAX_RANGE_M):
    """Inverse of pilot.project (B1): a pano pixel -> (lat, lng, range_m) on flat ground, or None when the
    pixel is at or above the horizon or lands beyond max_range.

    >>> import pilot
    >>> x, y, d, _ = pilot.project(47.6, -122.3, 75.0, 47.6001, -122.29985)
    >>> lat, lng, r = ground_point(47.6, -122.3, 75.0, x, y)
    >>> haversine_m(lat, lng, 47.6001, -122.29985) < 1e-6, abs(r - d) < 1e-6
    (True, True)
    >>> ground_point(47.6, -122.3, 75.0, 0.3, 0.5) is None, ground_point(47.6, -122.3, 75.0, 0.3, 0.51) is None
    (True, True)
    """
    from pilot import GEOD
    from rampnet.gsv import heading_to_azimuth
    dip = math.radians((y - 0.5) * 180.0)
    if dip <= 0:
        return None
    rng = camera_h / math.tan(dip)
    if rng > max_range:
        return None
    az = heading_to_azimuth(heading_deg) + (x - 0.5) * 360.0
    lng, lat, _ = GEOD.fwd(pano_lng, pano_lat, az, rng)
    return lat, lng, rng


def pano_positions(meta_dir, pano_ids):
    """pano_id -> (lat, lng, heading_deg, source). Source order: streetlevel (crowd_meta.jsonl, the frame the
    SDOT arm projected in), else Project Sidewalk's record (ps_meta.jsonl), else the negatives' PS record."""
    srcs = [("streetlevel", os.path.join(meta_dir, "crowd_meta.jsonl")),
            ("ps", os.path.join(meta_dir, "ps_meta.jsonl")),
            ("ps_negatives", os.path.join(meta_dir, "negatives", "neg_meta.jsonl"))]
    tables = [(name, {m["pano_id"]: m for m in read_jsonl(p) if m.get("found")}) for name, p in srcs]
    out = {}
    for pid in pano_ids:
        for name, t in tables:
            if pid in t:
                m = t[pid]
                out[pid] = (float(m["lat"]), float(m["lon"]), float(m["heading_deg"]), name)
                break
    return out


def census_nodes(census):
    """node_id -> (lat, lng) for every Seattle OSM highway=crossing node in the Phase 0 census tiles."""
    import glob
    out = {}
    for fp in sorted(glob.glob(os.path.join(census, "osm", SEATTLE, "Crosswalk", "*.json"))):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            t = e.get("tags") or {}
            if e.get("type") == "node" and t.get("highway") == "crossing":
                out[str(e["id"])] = (e["lat"], e["lon"])
    return out


def prf(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if t and not p)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "n": len(truth), "precision": p, "recall": r,
            "f1": f1(p, r) if not (math.isnan(p) or math.isnan(r)) else float("nan")}


def boot_f1_diff(a, b, truth, n=BOOT_N, seed=BOOT_SEED):
    """Exploratory: 95% percentile interval of F1(a) - F1(b) over a node bootstrap (nodes are spatially
    correlated, so this is narrower than the truth)."""
    rng = random.Random(seed)
    idx = list(range(len(truth)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        fa = prf([a[i] for i in s], [truth[i] for i in s])["f1"]
        fb = prf([b[i] for i in s], [truth[i] for i in s])["f1"]
        if not (math.isnan(fa) or math.isnan(fb)):
            diffs.append(fa - fb)
    diffs.sort()
    return (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]) if diffs else (None, None)


def score_nodes(test_anns, preds_by_pano, threshold, positions, nodes_rows, node_xy):
    """(b): one row per node within NODE_NEAR_PANO_M of a Seattle test pano."""
    from pilot import Grid
    sea = [pid for pid, ann in test_anns.items() if (ann.get("supervision") or {}).get("city") == SEATTLE]
    missing = [pid for pid in sea if pid not in positions]
    if missing:
        raise SystemExit(f"{len(missing)} Seattle test panos have no position/heading (e.g. {missing[:3]})")
    gpano = Grid([(positions[pid][0], positions[pid][1], pid) for pid in sea])
    hits = []
    n_det = n_placed = 0
    for pid in sea:
        lat, lng, hd, _ = positions[pid]
        for x, y, c in preds_by_pano.get(pid, []):
            if c < threshold:
                continue
            n_det += 1
            g = ground_point(lat, lng, hd, x, y)
            if g:
                n_placed += 1
                hits.append((g[0], g[1], pid))
    ghit = Grid(hits)
    rows = []
    for r in nodes_rows:
        nid = r["node_id"]
        if nid not in node_xy:
            continue
        la, ln = node_xy[nid]
        if not gpano.within(la, ln, NODE_NEAR_PANO_M):
            continue
        rows.append({"node_id": nid, "osm_class": r["osm_class"], "truth": int(r["truth"]),
                     "osm_marked_signal": int(r["osm_marked_signal"]),
                     "detector": int(bool(ghit.within(la, ln, NODE_MATCH_M)))})
    return rows, {"seattle_test_panos": len(sea), "detections": n_det, "placed_within_range": n_placed}


def node_exploratory(test_anns, test_preds, positions, node_rows, node_xy, nrows):
    """POST HOC, not pre-registered: node F1 across thresholds, and how the detector and OSM disagree at the
    val threshold. Reported to explain (b), never to re-decide it."""
    sweep = []
    for t in GRID[::5]:
        rr, _ = score_nodes(test_anns, test_preds, t, positions, node_rows, node_xy)
        truth = [r["truth"] for r in rr]
        d = prf([r["detector"] for r in rr], truth)
        sweep.append({"threshold": t, "detector_f1": d["f1"], "detector_p": d["precision"],
                      "detector_r": d["recall"],
                      "or_f1": prf([int(r["detector"] or r["osm_marked_signal"]) for r in rr], truth)["f1"],
                      "and_f1": prf([int(r["detector"] and r["osm_marked_signal"]) for r in rr], truth)["f1"]})
    cells = {}
    for r in nrows:
        k = f"det{r['detector']}_osm{r['osm_marked_signal']}_truth{r['truth']}"
        cells[k] = cells.get(k, 0) + 1
    truth = [r["truth"] for r in nrows]
    return {"threshold_sweep": sweep, "agreement": dict(sorted(cells.items(), reverse=True)),
            "always_yes": prf([1] * len(truth), truth)}


def node_summary(rows):
    truth = [r["truth"] for r in rows]
    det = [r["detector"] for r in rows]
    osm = [r["osm_marked_signal"] for r in rows]
    either = [int(d or o) for d, o in zip(det, osm)]
    un = [r for r in rows if r["osm_class"] == "unknown"]
    s = {"n_nodes": len(rows), "n_true": sum(truth),
         "detector": prf(det, truth), "osm_marked_signal": prf(osm, truth), "detector_or_osm": prf(either, truth),
         "untagged": {"n_nodes": len(un), "n_true": sum(r["truth"] for r in un),
                      "detector": prf([r["detector"] for r in un], [r["truth"] for r in un]),
                      "osm_marked_signal": prf([r["osm_marked_signal"] for r in un], [r["truth"] for r in un])},
         "f1_diff_boot95_exploratory": boot_f1_diff(det, osm, truth)}
    d, o = s["detector"]["f1"], s["osm_marked_signal"]["f1"]
    s["passes"] = (not math.isnan(d)) and (not math.isnan(o)) and d >= o
    return s


# --------------------------------------------------------------------------- #
# score (CPU)
# --------------------------------------------------------------------------- #
def load_preds(preds_dir, label, split, anns):
    rows = read_jsonl(os.path.join(preds_dir, label, f"{split}.jsonl"))
    by = {r["pano_id"]: r["preds"] for r in rows}
    if set(by) != set(anns) or len(rows) != len(anns):
        raise SystemExit(f"{label}/{split}: predictions cover {len(by)} panos ({len(rows)} lines), the split "
                         f"has {len(anns)} -- run infer to completion first")
    return by


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def cmd_score(a):
    import pilot  # noqa: F401  (Grid, GEOD: imported lazily below; fail fast if pyproj is missing)
    labels = a.labels.split(",")
    fixed = {k: float(v) for k, v in (x.split("=", 1) for x in a.fixed_threshold)}
    if set(fixed) - set(labels):
        raise SystemExit(f"--fixed-threshold names labels not scored: {sorted(set(fixed) - set(labels))}")
    anns = {s: load_split(a.root, s) for s in SPLITS}
    os.makedirs(a.out, exist_ok=True)
    with open(a.baseline_nodes, encoding="utf-8") as f:
        node_rows = list(csv.DictReader(f))
    node_xy = census_nodes(a.census)
    positions = pano_positions(a.meta_dir, list(anns["test"]))
    res = {"protocol": {"radius": RADIUS, "score_floor": SCORE_FLOOR, "min_distance": MIN_DISTANCE, "tta": False,
                        "grid": [GRID[0], GRID[-1], 0.01], "complete_modes": list(COMPLETE),
                        "camera_h": CAMERA_H, "max_range_m": MAX_RANGE_M, "node_near_pano_m": NODE_NEAR_PANO_M,
                        "node_match_m": NODE_MATCH_M},
           "inputs": {"baseline_nodes_sha256": sha256(a.baseline_nodes), "root": os.path.abspath(a.root),
                      "panos": {s: len(anns[s]) for s in SPLITS}},
           "models": {}}
    sweep_rows, breakdown_rows, node_out = [], [], []
    for lab in labels:
        meta = json.load(open(os.path.join(a.preds, lab, "meta.json"), encoding="utf-8"))
        cls = {}
        for s in SPLITS:
            pr = load_preds(a.preds, lab, s, anns[s])
            cls[s] = {pid: (anns[s][pid], classify_pano(pr[pid], anns[s][pid])) for pid in anns[s]}
        t_star, sweep = pick_threshold(list(cls["val"].values()))
        rule = "val_f1"
        if lab in fixed:      # a pre-registered fixed threshold (e.g. s3's recall-matched 0.27) instead of the pick
            t_star, rule = fixed[lab], "fixed"
        for s in SPLITS:
            for t in GRID:
                m = metrics_from(counts_at(list(cls[s].values()), t))
                sweep_rows.append({"model": lab, "split": s, "threshold": t, "precision": m["precision"],
                                   "recall": m["recall"], "f1": m["f1"], "tp": m["tp"],
                                   "tp_complete": m["tp_complete"], "fp": m["fp"], "ignored": m["ignored"],
                                   "unscored": m["unscored"], "n_gt": m["n_gt"]})
        mres = {"sha256": meta["sha256"], "checkpoint": meta["checkpoint"], "threshold": t_star,
                "threshold_rule": rule, "split": {}}
        for s in SPLITS:
            rows = list(cls[s].values())
            mres["split"][s] = {"overall": metrics_from(counts_at(rows, t_star)), "by": {}}
            for key in ("city", "mode", "year"):
                grp = {}
                for ann, cl in rows:
                    grp.setdefault(groups_of(ann)[key], []).append((ann, cl))
                mres["split"][s]["by"][key] = {}
                for g in sorted(grp):
                    m = metrics_from(counts_at(grp[g], t_star))
                    m["panos"] = len(grp[g])
                    mres["split"][s]["by"][key][g] = m
                    breakdown_rows.append({"model": lab, "split": s, "by": key, "group": g, "panos": len(grp[g]),
                                           "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                                           "tp": m["tp"], "tp_complete": m["tp_complete"], "fp": m["fp"],
                                           "ignored": m["ignored"], "n_gt": m["n_gt"]})
        # (b)
        test_preds = load_preds(a.preds, lab, "test", anns["test"])
        nrows, ninfo = score_nodes(anns["test"], test_preds, t_star, positions, node_rows, node_xy)
        mres["nodes"] = {**node_summary(nrows), **ninfo}
        if meta.get("sha256") != "n/a":
            mres["nodes_exploratory"] = node_exploratory(anns["test"], test_preds, positions, node_rows, node_xy,
                                                         nrows)
        for r in nrows:
            node_out.append({"model": lab, **r})
        res["models"][lab] = mres
        log(f"{lab}: val-F1 threshold {t_star}; test P {fmt(mres['split']['test']['overall']['precision'])} "
            f"R {fmt(mres['split']['test']['overall']['recall'])}; node F1 {fmt(mres['nodes']['detector']['f1'])} "
            f"vs OSM {fmt(mres['nodes']['osm_marked_signal']['f1'])}")
    src_counts = {}
    for pid in anns["test"]:
        if (anns["test"][pid].get("supervision") or {}).get("city") == SEATTLE:
            src_counts[positions[pid][3]] = src_counts.get(positions[pid][3], 0) + 1
    res["inputs"]["seattle_test_heading_sources"] = src_counts
    res["inputs"]["census_crossing_nodes"] = len(node_xy)
    res["scored_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(a.out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for name, rows in (("threshold_sweep.csv", sweep_rows), ("pixel_breakdown.csv", breakdown_rows),
                       ("nodes.csv", node_out)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    write_report(res, a.out, labels)


def m_sha(res, lab):
    return res["models"][lab]["sha256"]


def write_report(res, out, labels):
    L = []
    w = L.append
    w("# First crosswalk detector: pre-registered scores (RampNet#274)\n")
    w(f"Scored {res['scored_utc']} by `score_274.py score`. Protocol: `docs/stage1_crosswalk_sdot_274.md`.\n")
    w("## Models\n")
    w("| label | sha256 | val-F1 threshold |\n|---|---|---|")
    for lab in labels:
        m = res["models"][lab]
        w(f"| {lab} | `{m['sha256'][:16]}…` | {m['threshold']:.2f} |")
    w("\n## (a) Pixels, test split, at each model's own val-F1 threshold\n")
    w("Precision over complete panos (dense_ignore + verified_empty); recall over every test positive. "
      "95% Wilson intervals.\n")
    w("| model | precision | recall | F1 | TP (complete) | FP | ignored | positives |\n|---|---|---|---|---|---|---|---|")
    for lab in labels:
        o = res["models"][lab]["split"]["test"]["overall"]
        w(f"| {lab} | {fmt(o['precision'])} [{fmt(o['precision_ci'][0])}, {fmt(o['precision_ci'][1])}] | "
          f"{fmt(o['recall'])} [{fmt(o['recall_ci'][0])}, {fmt(o['recall_ci'][1])}] | {fmt(o['f1'])} | "
          f"{o['tp']} ({o['tp_complete']}) | {o['fp']} | {o['ignored']} | {o['n_gt']} |")
    for key, title in (("city", "per city"), ("mode", "per supervision mode"), ("year", "per capture year")):
        w(f"\n### Test, {title}\n")
        w("| model | group | panos | precision | recall | F1 | FP | positives |\n|---|---|---|---|---|---|---|---|")
        for lab in labels:
            for g, m in res["models"][lab]["split"]["test"]["by"][key].items():
                w(f"| {lab} | {g} | {m['panos']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                  f"{m['fp']} | {m['n_gt']} |")
    w("\nPrecision is n/a where a group has no complete pano (every non-Seattle city is `sparse`), or where a "
      "complete group drew no detection at all (`verified_empty` with 0 FP).\n")
    w("\n### Val, overall (where the threshold was chosen)\n")
    w("| model | threshold | precision | recall | F1 |\n|---|---|---|---|---|")
    for lab in labels:
        m = res["models"][lab]
        o = m["split"]["val"]["overall"]
        w(f"| {lab} | {m['threshold']:.2f} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])} |")
    w("\n## (b) Per OSM node, Seattle test panos, against the map-only baseline\n")
    w(f"Nodes of `baseline_nodes.csv` within {NODE_NEAR_PANO_M:g} m of a Seattle test pano. A node is predicted "
      f"marked when a detection at the model's val-F1 threshold, raycast to flat ground at {CAMERA_H} m and kept "
      f"within {MAX_RANGE_M:g} m range, lands within {NODE_MATCH_M:g} m. Truth: a current SDOT point within "
      f"10 m.\n")
    w("| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 | detector OR OSM F1 | "
      "passes (detector F1 ≥ OSM) |\n|---|---|---|---|---|---|")
    for lab in labels:
        n = res["models"][lab]["nodes"]
        d, o, e = n["detector"], n["osm_marked_signal"], n["detector_or_osm"]
        w(f"| {lab} | {n['n_nodes']} ({n['n_true']}) | {fmt(d['precision'])} / {fmt(d['recall'])} / {fmt(d['f1'])} | "
          f"{fmt(o['precision'])} / {fmt(o['recall'])} / {fmt(o['f1'])} | {fmt(e['f1'])} | "
          f"{'n/a (ceiling)' if m_sha(res, lab) == 'n/a' else '**yes**' if n['passes'] else 'no'} |")
    w("\nOn nodes OSM leaves untagged (`osm_class` unknown):\n")
    w("| model | nodes (true) | detector P / R / F1 | `osm_marked_signal` P / R / F1 |\n|---|---|---|---|")
    for lab in labels:
        u = res["models"][lab]["nodes"]["untagged"]
        d, o = u["detector"], u["osm_marked_signal"]
        w(f"| {lab} | {u['n_nodes']} ({u['n_true']}) | {fmt(d['precision'])} / {fmt(d['recall'])} / {fmt(d['f1'])} | "
          f"{fmt(o['precision'])} / {fmt(o['recall'])} / {fmt(o['f1'])} |")
    w("\nExploratory (not part of the pass rule): a node-bootstrap 95% interval for detector F1 minus OSM F1 "
      "(nodes are spatially correlated, so it is too narrow):\n")
    for lab in labels:
        n = res["models"][lab]["nodes"]
        lo, hi = n["f1_diff_boot95_exploratory"]
        w(f"- {lab}: [{fmt(lo)}, {fmt(hi)}]; {n['detections']} detections on {n['seattle_test_panos']} panos, "
          f"{n['placed_within_range']} placed within range")
    w(f"\nHeading source for the Seattle test panos: {res['inputs']['seattle_test_heading_sources']}.\n")
    w("\n## Exploratory, post hoc (not pre-registered: explains (b), never re-decides it)\n")
    for lab in labels:
        ex = res["models"][lab].get("nodes_exploratory")
        if not ex:
            continue
        w(f"### {lab}: node F1 across thresholds (val pick {res['models'][lab]['threshold']:.2f})\n")
        w("| threshold | detector P / R / F1 | detector OR OSM F1 | detector AND OSM F1 |\n|---|---|---|---|")
        for r in ex["threshold_sweep"]:
            w(f"| {r['threshold']:.2f} | {fmt(r['detector_p'])} / {fmt(r['detector_r'])} / {fmt(r['detector_f1'])} | "
              f"{fmt(r['or_f1'])} | {fmt(r['and_f1'])} |")
        w(f"\nOSM alone: F1 {fmt(res['models'][lab]['nodes']['osm_marked_signal']['f1'])}. Saying yes at every "
          f"node: F1 {fmt(ex['always_yes']['f1'])} (its precision is the share of nodes that are marked).\n")
        w(f"### {lab}: detector vs OSM vs truth at the val threshold (node counts)\n")
        w("| detector | OSM | truth | nodes |\n|---|---|---|---|")
        for k, v in ex["agreement"].items():
            d, o, t = (part[-1] for part in k.split("_"))
            w(f"| {'yes' if d == '1' else 'no'} | {'yes' if o == '1' else 'no'} | "
              f"{'marked' if t == '1' else 'not'} | {v} |")
        w("")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log(f"report -> {os.path.join(out, 'report.md')}")


# --------------------------------------------------------------------------- #
# figures (CPU)
# --------------------------------------------------------------------------- #
def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(a.out, "scores.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(a.out, "threshold_sweep.csv"), encoding="utf-8")))
    labels = list(res["models"])
    colors = ["#3b6ea8", "#c0504d", "#4f9a5b", "#8064a2"]
    os.makedirs(a.fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, lab in enumerate(labels):
        for s, ls in (("val", "--"), ("test", "-")):
            rr = [r for r in rows if r["model"] == lab and r["split"] == s]
            ts_ = [float(r["threshold"]) for r in rr]
            p = [float(r["precision"]) for r in rr]
            rc = [float(r["recall"]) for r in rr]
            ff = [float(r["f1"]) for r in rr]
            axes[0].plot(rc, p, ls, color=colors[i], label=f"{lab} {s}")
            axes[1].plot(ts_, ff, ls, color=colors[i], label=f"{lab} {s}")
        t = res["models"][lab]["threshold"]
        o = res["models"][lab]["split"]["test"]["overall"]
        axes[0].plot([o["recall"]], [o["precision"]], "o", color=colors[i])
        axes[1].axvline(t, color=colors[i], lw=0.8, alpha=0.6)
    axes[0].set_xlabel("recall (all positives)")
    axes[0].set_ylabel("precision (complete panos)")
    axes[0].set_title("Pixel PR, threshold swept (dot = test at val-F1 threshold)")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs threshold (vertical = val-F1 pick)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(a.fig_dir, "pr_f1.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    groups = sorted({g for lab in labels for g in res["models"][lab]["split"]["test"]["by"]["city"]})
    wbar = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        by = res["models"][lab]["split"]["test"]["by"]["city"]
        ax.bar([k + i * wbar for k in range(len(groups))], [by[g]["recall"] if g in by else 0 for g in groups],
               wbar, color=colors[i], label=lab)
    ax.set_xticks([k + wbar * (len(labels) - 1) / 2 for k in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("test recall at val-F1 threshold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(a.fig_dir, "recall_by_city.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    log(f"figures -> {p1}, {p2}")


def card_items(anns, preds, threshold):
    """Test-split cards at a threshold: every FP (complete panos), every missed positive, and every TP, each with
    the pano's other marks. Returns a list of dicts sorted by kind then confidence."""
    items = []
    for pid, ann in anns.items():
        sup = ann.get("supervision") or {}
        gt = [tuple(q) for q in ann.get("curb_ramp_points_normalized", [])]
        kept = sorted([q for q in preds.get(pid, []) if q[2] >= threshold], key=lambda q: -q[2])
        cl = classify_pano(kept, ann)
        assign = greedy_match([(q[0], q[1]) for q in kept], gt, RADIUS_SQ, SCALE_X, SCALE_Y, wrap_x=True)
        hit = {k for k, _ in assign if k >= 0}
        used = {(kept[j][0], kept[j][1]) for j, (k, _) in enumerate(assign) if k >= 0}   # spent on another label
        base = {"pano_id": pid, "city": sup.get("city"), "mode": sup.get("mode"),
                "year": year_bin(ann.get("capture_date")), "gt": gt, "dets": kept,
                "ignore": [tuple(q) for q in sup.get("ignore_points", [])]}
        for q, (conf, kind) in zip(kept, cl):
            if kind in ("tp", "fp"):
                items.append({**base, "kind": kind, "x": q[0], "y": q[1], "conf": conf})
        for k, g in enumerate(gt):
            if k not in hit:
                # the best peak in the radius that no other label took (any confidence down to the floor)
                inr = [q for q in preds.get(pid, [])
                       if dist_sq(q[0], q[1], g[0], g[1], SCALE_X, SCALE_Y, True) < RADIUS_SQ]
                near = max([q[2] for q in inr if (q[0], q[1]) not in used] or [0.0])
                # sort_conf (claimed peaks included) is the first gallery's order; the seeded sample draws from it,
                # so the rated sample of misses stays the same cards
                items.append({**base, "kind": "fn", "x": g[0], "y": g[1], "conf": near,
                              "sort_conf": max([q[2] for q in inr] or [0.0])})
    order = {"fp": 0, "fn": 1, "tp": 2}
    return sorted(items, key=lambda it: (order[it["kind"]], -it.get("sort_conf", it["conf"])))


def crop_card(img, it, half_w=0.08, out_w=480, marks=True):
    """A crop centred on the item (seam-wrapped), marks drawn: green GT, red detections above threshold, grey
    ignore points, the item ringed in yellow. marks=False returns the bare crop. Returns JPEG bytes."""
    import io
    from PIL import ImageDraw
    W, H = img.size
    cw, ch = int(2 * half_w * W), int(half_w * W)
    cx, cy = int(it["x"] * W), int(it["y"] * H)
    x0, y0 = cx - cw // 2, max(0, min(H - ch, cy - ch // 2))
    import numpy as np
    from PIL import Image
    arr = np.asarray(img)
    cols = np.arange(x0, x0 + cw) % W             # seam wrap
    crop = Image.fromarray(arr[y0:y0 + ch][:, cols])
    s = out_w / cw
    crop = crop.resize((out_w, int(ch * s)))
    if not marks:
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    d = ImageDraw.Draw(crop)

    def local(px, py):
        lx = (px * W - x0) % W
        return lx * s, (py * H - y0) * s

    r_match = RADIUS * W * s      # the matcher's radius is RADIUS pano widths in x and in y (y counts half of H)
    sx, sy = it["x"], it["y"]
    for px, py in it["ignore"]:
        lx, ly = local(px, py)
        d.ellipse((lx - 5, ly - 5, lx + 5, ly + 5), outline=(160, 160, 160), width=2)
    for px, py in it["gt"]:
        lx, ly = local(px, py)
        big = it["kind"] == "fn" and (px, py) == (sx, sy)          # the card's subject when it is a miss
        r, w = (12, 5) if big else (7, 2)
        d.ellipse((lx - r, ly - r, lx + r, ly + r), outline=(40, 220, 60), width=w)
    for px, py, c in it["dets"]:
        lx, ly = local(px, py)
        if it["kind"] != "fn" and (px, py) == (sx, sy):
            continue                                                # the subject is drawn last, large
        d.line((lx - 5, ly, lx + 5, ly), fill=(255, 90, 90), width=1)
        d.line((lx, ly - 5, lx, ly + 5), fill=(255, 90, 90), width=1)
    lx, ly = local(sx, sy)
    if it["kind"] != "fn":                                          # the card's subject detection: outlined, ringed
        for col, w in (((255, 255, 255), 7), ((230, 0, 0), 3)):
            d.line((lx - 11, ly, lx + 11, ly), fill=col, width=w)
            d.line((lx, ly - 11, lx, ly + 11), fill=col, width=w)
        d.ellipse((lx - 16, ly - 16, lx + 16, ly + 16), outline=(255, 255, 255), width=2)
    d.ellipse((lx - r_match, ly - r_match, lx + r_match, ly + r_match), outline=(255, 220, 0), width=1)
    if it.get("neg_disc"):        # the disc a confirmed negative would train as background (train.py: ceil(3*sigma)
        r_neg = NEG_DISC * W * s  # heatmap px of 1024), drawn dashed in cyan
        for k in range(0, 360, 12):
            d.arc((lx - r_neg, ly - r_neg, lx + r_neg, ly + r_neg), k, k + 7, fill=(0, 230, 255), width=3)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=82)
    return buf.getvalue()


VERDICTS = {   # per card kind: (code, button text); keys 1-4 pick them in order
    "fp": (("crosswalk", "Correct: it is a crosswalk"), ("not_crosswalk", "Wrong: not a crosswalk"),
           ("unsure", "Unsure")),
    "fn": (("real_miss", "Real crosswalk, model missed it"), ("no_crosswalk", "No crosswalk here: label wrong"),
           ("unsure", "Unsure"), ("found_same", "Model did find it: a cross is on this crosswalk")),
    "tp": (("crosswalk", "Correct: it is a crosswalk"), ("not_crosswalk", "Wrong: not a crosswalk"),
           ("unsure", "Unsure")),
}
VERDICTS["det"] = VERDICTS["fp"]   # a detection with no label context (`cards`): the same question as an FP

DEFINITIONS = {    # what "crosswalk" means on a page (`cards --definition`); paint is every page before s4
    "paint": "<strong>Crosswalk</strong> means painted pedestrian crossing markings: zebra bars or two parallel lines "
             "across a\nroadway.",
    "paint_brick": "<strong>Crosswalk</strong> means a marked pedestrian crossing across a roadway: painted markings "
                   "(zebra bars or two parallel lines, on any road surface, cobblestone included) <em>or</em> a brick, "
                   "paver, stamped or coloured band laid across the road as a crossing, with or without white edge "
                   "lines (Jon, 2026-11-08). A cobblestone or brick <em>road</em>, plaza or sidewalk with no crossing "
                   "band is not a crosswalk.",
}

SECTION_HEAD = {   # section title and the one question its cards ask
    "fp": ("False positives", "The model fired where no label is. <strong>Is the outlined cross on a crosswalk?</strong> "
           "Ignore labels elsewhere on the crop."),
    "fn": ("Misses", "A label no detection was matched to. <strong>Is there a crosswalk at the large green ring?</strong> "
           "If yes, it is a real miss even when the model found the intersection's other crosswalks; each label is "
           "one crosswalk. If a small cross already sits on that same crosswalk (another label on it took the match: a "
           "duplicate label), press 4. \"Best unclaimed peak\" is the strongest model response inside the yellow circle that "
           "another label did not already take (0.00 = nothing there)."),
    "tp": ("True positives", "The model matched a label. <strong>Is the outlined cross on a crosswalk?</strong>"),
    "det": ("Detections", "The model fired here. <strong>Is the outlined cross on a crosswalk?</strong> "
            "Ignore other marks on the crop."),
}


# Inline SVG glyphs for the legend, drawn like crop_card draws them (aria-hidden: the text beside each says it).
_G = '<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" style="vertical-align:middle;background:#777">'
LEGEND_GLYPHS = {
    "subject_det": _G + '<g stroke="#fff" stroke-width="7"><line x1="6" y1="17" x2="28" y2="17"/>'
                   '<line x1="17" y1="6" x2="17" y2="28"/></g><g stroke="#e60000" stroke-width="3">'
                   '<line x1="6" y1="17" x2="28" y2="17"/><line x1="17" y1="6" x2="17" y2="28"/></g>'
                   '<circle cx="17" cy="17" r="15" fill="none" stroke="#fff" stroke-width="2"/></svg>',
    "other_det": _G + '<g stroke="#ff5a5a" stroke-width="1.5"><line x1="12" y1="17" x2="22" y2="17"/>'
                 '<line x1="17" y1="12" x2="17" y2="22"/></g></svg>',
    "subject_gt": _G + '<circle cx="17" cy="17" r="12" fill="none" stroke="#28dc3c" stroke-width="5"/></svg>',
    "other_gt": _G + '<circle cx="17" cy="17" r="7" fill="none" stroke="#28dc3c" stroke-width="2"/></svg>',
    "ignore": _G + '<circle cx="17" cy="17" r="5" fill="none" stroke="#a0a0a0" stroke-width="2"/></svg>',
    "radius": _G + '<ellipse cx="17" cy="17" rx="15" ry="15" fill="none" stroke="#ffdc00" stroke-width="1.5"/></svg>',
}


def card_id(it):
    """Stable id of one card: kind, pano and the subject point to 3 decimals (as the caption prints it).

    >>> card_id({"kind": "fp", "pano_id": "abc", "x": 0.41049, "y": 0.5512})
    'fp:abc:0.410:0.551'
    """
    return f"{it['kind']}:{it['pano_id']}:{it['x']:.3f}:{it['y']:.3f}"


PREFILL_R = 0.01     # a verdict carries to a card on the same pano within this (pano widths): the same paint, peak moved
VERDICT_FAMILY = {"fp": "det", "tp": "det", "det": "det", "fn": "fn"}   # kinds whose verdicts mean the same thing


def split_card_id(cid):
    """(kind, pano_id, x, y) from a card id.

    >>> split_card_id('fp:abc:0.410:0.551')
    ('fp', 'abc', 0.41, 0.551)
    """
    kind, rest = cid.split(":", 1)
    pid, x, y = rest.rsplit(":", 2)
    return kind, pid, float(x), float(y)


def wrap_dist(x1, y1, x2, y2):
    """Plain distance in pano widths with the seam wrapped (x and y in the same normalised units).

    >>> round(wrap_dist(0.995, 0.5, 0.004, 0.5), 3)
    0.009
    """
    dx = abs(x1 - x2)
    return math.hypot(min(dx, 1.0 - dx), y1 - y2)


def load_prefill(path):
    """Rated rows of an exported ratings CSV (needs card_id and verdict; note optional). Unrated rows are skipped."""
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("verdict"):
                continue
            kind, pid, x, y = split_card_id(r["card_id"])
            rows.append({"card_id": r["card_id"], "family": VERDICT_FAMILY.get(kind, kind), "pano_id": pid,
                         "x": x, "y": y, "verdict": r["verdict"], "note": r.get("note", "")})
    return rows


def prefill_for(items, rows, r=PREFILL_R):
    """{card_id: {verdict, note}} for the items that already have a verdict in `rows`.

    An exact card id wins. Otherwise the nearest verdict of the same family (fp/tp/det share their verdicts, fn
    has its own) on the same pano within `r` carries over, with its source named in the note: a retrained model
    puts its peak a little off the rated one on the same paint (#274: 20 of 21 such pairs got the same verdict
    when rated twice).

    >>> rows = [{"card_id": "fp:p:0.500:0.550", "family": "det", "pano_id": "p", "x": 0.5, "y": 0.55,
    ...          "verdict": "not_crosswalk", "note": "bus lane"}]
    >>> prefill_for([{"kind": "det", "pano_id": "p", "x": 0.506, "y": 0.552}], rows)
    {'det:p:0.506:0.552': {'verdict': 'not_crosswalk', 'note': 'carried from fp:p:0.500:0.550: bus lane'}}
    >>> prefill_for([{"kind": "fn", "pano_id": "p", "x": 0.5, "y": 0.55}], rows)
    {}
    """
    exact = {row["card_id"]: row for row in rows}
    by = {}
    for row in rows:
        by.setdefault((row["pano_id"], row["family"]), []).append(row)
    out = {}
    for it in items:
        cid = card_id(it)
        if cid in exact:
            out[cid] = {"verdict": exact[cid]["verdict"], "note": exact[cid]["note"]}
            continue
        near = [(wrap_dist(it["x"], it["y"], row["x"], row["y"]), row)
                for row in by.get((it["pano_id"], VERDICT_FAMILY.get(it["kind"], it["kind"])), [])]
        near = [(d, row) for d, row in near if d <= r]
        if near:
            _, row = min(near, key=lambda t: t[0])
            note = f"carried from {row['card_id']}" + (f": {row['note']}" if row["note"] else "")
            out[cid] = {"verdict": row["verdict"], "note": note}
    return out


def card_figure(it, jpeg, cap, views=None):
    """One rating card: the crop, its caption, verdict buttons for the card's kind and a note box. `views`
    (optional {"big": jpeg, "big_clean": jpeg}) feeds the page's large view: click the image to open it,
    M toggles the marks."""
    import base64
    import html
    subject = "this miss (large green ring)" if it["kind"] == "fn" else "this detection (large outlined cross)"
    btns = "".join(f'<button type="button" data-v="{v}" aria-pressed="false">{k + 1}. {html.escape(txt)}'
                   f'</button>' for k, (v, txt) in enumerate(VERDICTS[it["kind"]]))
    meta = json.dumps({"card_id": card_id(it), "kind": it["kind"], "pano_id": it["pano_id"], "x": round(it["x"], 4),
                       "y": round(it["y"], 4), "conf": round(it["conf"], 3), "city": it.get("city"),
                       "mode": it.get("mode"), "year": it.get("year"), "stratum": it.get("stratum", "")})
    b64 = base64.b64encode(jpeg).decode("ascii")
    extra = "".join(f' data-{k.replace("_", "-")}="data:image/jpeg;base64,{base64.b64encode(v).decode("ascii")}"'
                    for k, v in (views or {}).items())
    return (f'<figure class="card {it["kind"]}" tabindex="0" data-meta="{html.escape(meta)}"{extra}>'
            f'<img src="data:image/jpeg;base64,{b64}" alt="{html.escape(cap)}">'
            f'<figcaption>{html.escape(cap)}</figcaption>'
            f'<div class="rate" role="group" aria-label="Verdict on {subject}">{btns}</div>'
            f'<label>Note <input type="text" class="note" placeholder="e.g. bus lane, stop bar"></label>'
            f'</figure>')


def rating_page(label, heading, intro_html, nav_html, cards_html, prefill):
    """The self-contained rating page around `cards_html`: progress, hide-rated, keyboard rating, CSV export.
    Verdicts live in the browser's localStorage under a key per `label`; `prefill` seeds unrated cards."""
    import html
    store_key = f"cw274-rate-{label}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(label)} rating</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#222}}
.bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc;z-index:1}}
figure{{display:inline-block;margin:6px;width:480px;max-width:100%;vertical-align:top;padding-bottom:6px}}
img{{width:100%;display:block}}figcaption{{font-size:12px}}
.fp,.det{{border-top:4px solid #d33}}.fn{{border-top:4px solid #e90}}.tp{{border-top:4px solid #2a2}}
figure:focus{{outline:3px solid #36f;outline-offset:2px}}figure.done{{opacity:.55}}
[hidden]{{display:none!important}}
.rate button{{margin:4px 4px 0 0;font:inherit;padding:3px 8px;border:1px solid #888;background:#f4f4f4;border-radius:4px}}
.rate button[aria-pressed=true]{{background:#225;color:#fff;border-color:#225}}
label{{display:block;margin-top:4px;font-size:12px}}.note{{width:380px;max-width:90%;font:inherit}}
h2{{margin-top:28px;padding-top:56px;margin-bottom:4px}}.q{{max-width:980px;margin-top:0}}.legend{{list-style:none;padding:0;columns:2 420px;margin:8px 0}}
.legend li{{margin:3px 0;break-inside:avoid}}.sw{{display:inline-block;width:34px;height:6px;vertical-align:middle}}
figure[data-big] img{{cursor:zoom-in}}
#lb{{position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:10;display:flex;flex-direction:column;align-items:center;
justify-content:center;padding:12px;color:#fff}}
#lb img{{max-width:100%;max-height:78vh;display:block}}#lb .lbbar{{margin-top:8px;text-align:center}}
#lb button{{font:inherit;margin:2px 4px;padding:4px 10px;border-radius:4px;border:1px solid #bbb;background:#333;color:#fff}}
#lb button[aria-pressed=true]{{background:#fff;color:#111}}</style></head><body>
<h1>{html.escape(heading)}</h1>
{intro_html}
<div class="bar">{nav_html}<span id="prog"></span> &nbsp;
<label style="display:inline"><input type="checkbox" id="hide"> hide rated</label> &nbsp;
<button type="button" id="exp">Export CSV</button></div>
{''.join(cards_html)}
<div id="lb" role="dialog" aria-modal="true" aria-label="Large view" hidden>
<img id="lbimg" alt="">
<div class="lbbar"><span id="lbcap"></span><br>
<span id="lbrate"></span>
<button type="button" id="lbmarks" aria-pressed="true">M: marks on</button>
<button type="button" id="lbclose">Esc: close</button></div></div>
<script>
const KEY = {json.dumps(store_key)}, PREFILL = {json.dumps(prefill)};
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
for (const [k, v] of Object.entries(PREFILL)) if (!state[k]) state[k] = {{...v, rated_at: "prefill"}};
const cards = [...document.querySelectorAll("figure.card")];
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} }};
const meta = c => JSON.parse(c.dataset.meta);
function paint(c) {{
  const s = state[meta(c).card_id] || {{}};
  c.querySelectorAll(".rate button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === s.verdict)));
  c.querySelector(".note").value = s.note || "";
  c.classList.toggle("done", !!s.verdict);
  c.hidden = document.getElementById("hide").checked && !!s.verdict;
}}
function progress() {{
  const by = {{}};
  for (const c of cards) {{ const m = meta(c); by[m.kind] = by[m.kind] || [0, 0]; by[m.kind][1]++;
    if ((state[m.card_id] || {{}}).verdict) by[m.kind][0]++; }}
  const name = {{fp: "FPs", fn: "misses", tp: "TPs", det: "detections"}};
  document.getElementById("prog").textContent = "Rated: " +
    Object.entries(by).map(([k, [d, n]]) => `${{name[k]}} ${{d}}/${{n}}`).join(" · ");
}}
function setVerdict(c, v) {{
  const id = meta(c).card_id, s = state[id] || {{}};
  state[id] = {{...s, verdict: s.verdict === v ? "" : v, rated_at: new Date().toISOString()}};
  save(); paint(c); progress();
}}
for (const c of cards) {{
  c.querySelectorAll(".rate button").forEach(b => b.addEventListener("click", () => setVerdict(c, b.dataset.v)));
  c.querySelector(".note").addEventListener("input", e => {{
    const id = meta(c).card_id; state[id] = {{...(state[id] || {{}}), note: e.target.value}}; save(); }});
  c.addEventListener("keydown", e => {{
    if (e.target.tagName === "INPUT") return;
    const btns = c.querySelectorAll(".rate button"), k = Number(e.key);
    if (k >= 1 && k <= btns.length) {{
      setVerdict(c, btns[k - 1].dataset.v);
      const nxt = cards.slice(cards.indexOf(c) + 1).find(x => !x.hidden); if (nxt) nxt.focus();
    }} else if (e.key === "n") {{
      const nxt = cards.find(x => !(state[meta(x).card_id] || {{}}).verdict); if (nxt) nxt.focus();
    }}
  }});
  paint(c);
}}
document.getElementById("hide").addEventListener("change", () => cards.forEach(paint));
// Large view (cards built with --big, the default): click an image; 1-3 rate and move on; M toggles marks; arrows; Esc closes.
const lb = document.getElementById("lb"), lbimg = document.getElementById("lbimg");
let lbCard = null, lbMarks = true;
function lbShow(c) {{
  lbCard = c;
  lbimg.src = lbMarks ? c.dataset.big : c.dataset.bigClean;
  lbimg.alt = c.querySelector("figcaption").textContent;
  document.getElementById("lbcap").textContent = lbimg.alt;
  const s = state[meta(c).card_id] || {{}};
  document.getElementById("lbrate").innerHTML = [...c.querySelectorAll(".rate button")].map(b =>
    `<button type="button" data-v="${{b.dataset.v}}" aria-pressed="${{b.dataset.v === s.verdict}}">${{b.textContent}}</button>`).join("");
  document.querySelectorAll("#lbrate button").forEach(b => b.addEventListener("click", () => lbRate(b.dataset.v)));
  const mb = document.getElementById("lbmarks");
  mb.setAttribute("aria-pressed", String(lbMarks)); mb.textContent = lbMarks ? "M: marks on" : "M: marks off";
  lb.hidden = false; document.getElementById("lbclose").focus();
}}
function lbClose() {{ lb.hidden = true; if (lbCard) lbCard.focus(); lbCard = null; }}
function lbStep(d) {{
  const vis = cards.filter(x => !x.hidden && x.dataset.big), i = vis.indexOf(lbCard);
  if (i + d >= 0 && i + d < vis.length) lbShow(vis[i + d]); else lbClose();
}}
function lbRate(v) {{ setVerdict(lbCard, v); lbStep(1); }}
for (const c of cards) if (c.dataset.big) c.querySelector("img").addEventListener("click", () => lbShow(c));
document.getElementById("lbclose").addEventListener("click", lbClose);
document.getElementById("lbmarks").addEventListener("click", () => {{ lbMarks = !lbMarks; lbShow(lbCard); }});
lb.addEventListener("click", e => {{ if (e.target === lb) lbClose(); }});
document.addEventListener("keydown", e => {{
  if (lb.hidden) return;
  const btns = lbCard.querySelectorAll(".rate button"), k = Number(e.key);
  if (e.key === "Escape") lbClose();
  else if (e.key === "m" || e.key === "M") {{ lbMarks = !lbMarks; lbShow(lbCard); }}
  else if (e.key === "ArrowRight") lbStep(1);
  else if (e.key === "ArrowLeft") lbStep(-1);
  else if (k >= 1 && k <= btns.length) lbRate(btns[k - 1].dataset.v);
  else return;
  e.preventDefault(); e.stopPropagation();
}}, true);
document.getElementById("exp").addEventListener("click", () => {{
  const cols = ["card_id", "kind", "pano_id", "x", "y", "conf", "city", "mode", "year", "stratum", "verdict", "note",
                "rated_at"];
  const q = v => `"${{String(v ?? "").replace(/"/g, '""')}}"`;
  const rows = cards.map(c => {{ const m = meta(c), s = state[m.card_id] || {{}};
    return cols.map(k => q(k in m ? m[k] : s[k])).join(","); }});
  const blob = new Blob([[cols.join(","), ...rows].join("\\n") + "\\n"], {{type: "text/csv"}});
  const u = URL.createObjectURL(blob), el = document.createElement("a");
  el.href = u; el.download = "ratings_{label}.csv"; el.click(); URL.revokeObjectURL(u);
}});
progress();
</script></body></html>"""


def cmd_gallery(a):
    """Local rating page (untracked: it embeds imagery) of a model's test items, one card per item.

    A card's subject is ONE detection (FP/TP: the large outlined cross) or ONE labelled point (miss: the large
    green ring); other marks on the crop are context. Every FP is shown, plus seeded samples of misses and TPs
    (the same seeded misses as the first, view-only gallery). Verdicts save in the browser and export as CSV;
    --prefill seeds verdicts from such a CSV (`prefill_for`: exact card id, else the same paint within 0.01).
    """
    from PIL import Image
    res = json.load(open(os.path.join(a.scores, "scores.json"), encoding="utf-8"))
    t = res["models"][a.label]["threshold"] if a.threshold is None else a.threshold
    anns = load_split(a.root, "test")
    items = card_items(anns, load_preds(a.preds, a.label, "test", anns), t)
    rows = load_prefill(a.prefill)
    prefill = prefill_for(items, rows)
    exact = {row["card_id"] for row in rows}   # the sample keeps exact-id semantics; carried verdicts only prefill
    rng = random.Random(BOOT_SEED)
    pick = []
    for kind, n in (("fp", a.n), ("fn", a.n), ("tp", a.n_tp)):
        group = [it for it in items if it["kind"] == kind]
        sample = rng.sample(group, min(n, len(group)))   # drawn even for fp, so the miss sample stays the same
        # After a relabel, keep every already-rated card that is still a card of this kind, and top up from the
        # seeded sample, so only new items need a verdict. The labels changed by removal and new ignore regions,
        # so still-present rated misses are close to a uniform draw from the new misses (not exactly: see doc).
        rated = [it for it in group if card_id(it) in exact]
        sample = (rated + [it for it in sample if card_id(it) not in exact])[:max(n, len(rated))]
        pick += sorted(group if kind == "fp" else sample, key=lambda it: -it["conf"])
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    cards = []
    last_kind = None
    for it in pick:
        if it["pano_id"] not in cache:
            cache.clear()
            cache[it["pano_id"]] = Image.open(os.path.join(a.root, "test", f"{it['pano_id']}.jpg")).convert("RGB")
        head = (f"MISS · best unclaimed peak {it['conf']:.2f}" if it["kind"] == "fn"
                else f"{it['kind'].upper()} conf {it['conf']:.2f}")
        cap = (f"{head} · {it['city']} · {it['mode']} · {it['year']} · "
               f"{it['pano_id']} ({it['x']:.3f}, {it['y']:.3f})")
        if it["kind"] != last_kind:                                 # one section per card kind, with its question
            last_kind = it["kind"]
            title, question = SECTION_HEAD[last_kind]
            n_kind = sum(1 for x in pick if x["kind"] == last_kind)
            cards.append(f'<h2 id="sec-{last_kind}">{title} ({n_kind} cards)</h2><p class="q">{question}</p>')
        cards.append(card_figure(it, crop_card(cache[it["pano_id"]], it), cap))
    counts = {k: sum(1 for it in items if it["kind"] == k) for k in ("fp", "fn", "tp")}
    shown = {k: sum(1 for it in pick if it["kind"] == k) for k in ("fp", "fn", "tp")}
    intro = f"""<p>Each card asks about <strong>one</strong> item, drawn large. Everything else on the crop is context and,
if it is a false positive or a miss, has its own card.</p>
<ul class="legend">
<li><span class="sw" style="background:#d33"></span> <strong>False positive card</strong>: the model fired where
no label is. Is the large cross on a crosswalk?</li>
<li><span class="sw" style="background:#e90"></span> <strong>Miss card</strong>: a label the model did not
match. Is there a crosswalk at the large green ring?</li>
<li><span class="sw" style="background:#2a2"></span> <strong>True positive card</strong>: the model matched a
label. Is the large cross on a crosswalk?</li>
<li>{LEGEND_GLYPHS['subject_det']} the detection this card asks about</li>
<li>{LEGEND_GLYPHS['subject_gt']} the label this card asks about (miss cards)</li>
<li>{LEGEND_GLYPHS['other_det']} another detection at or above the threshold (context)</li>
<li>{LEGEND_GLYPHS['other_gt']} another label (context)</li>
<li>{LEGEND_GLYPHS['ignore']} ignore point: detections here are not scored</li>
<li>{LEGEND_GLYPHS['radius']} match radius: a detection and a label inside it can pair up</li>
</ul>
<p>Keys: click or Tab to a card, then <kbd>1</kbd>-<kbd>4</kbd> to rate (jumps to the next card),
<kbd>n</kbd> = next unrated. Verdicts save in this browser; press <em>Export CSV</em> when done.
Test split has {counts['fp']} FPs (all shown), {counts['fn']} misses ({shown['fn']} sampled),
{counts['tp']} TPs ({shown['tp']} sampled).</p>"""
    nav = ('<nav aria-label="Sections"><a href="#sec-fp">False positives</a> · '
           '<a href="#sec-fn">Misses</a> · <a href="#sec-tp">True positives</a></nav>')
    page = rating_page(a.label, f"{a.label}: rate test items at threshold {t:.2f}", intro, nav, cards, prefill)
    path = os.path.join(a.out, f"rate_{a.label}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    log(f"rating page -> {path} ({len(pick)} cards, {len(prefill)} prefilled)")


def read_candidates(path):
    """Candidate detections for `cards`: CSV with pano_id, x, y, conf and optional city, mode, year, stratum."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{**r, "x": float(r["x"]), "y": float(r["y"]), "conf": float(r["conf"])} for r in rows]


def cmd_cards(a):
    """Rating page (untracked: it embeds imagery) for an arbitrary list of detections, one card per detection.

    Used for the (b') deployment audit and for hard-negative candidates (#274). Context marks: the same model's
    other detections on the pano at or above --threshold (--preds-file), and, with --root/--split, the pano's
    labels and ignore points. --blind drops confidence, city, mode and year from the captions and shuffles the
    cards with a fixed seed, so a rater cannot tell which model or stratum a card came from.
    """
    from PIL import Image
    cands = read_candidates(a.candidates)
    if a.only:
        with open(a.only, encoding="utf-8", newline="") as f:
            keep = {r["card_id"] for r in csv.DictReader(f)}
        cands = [c for c in cands if card_id({**c, "kind": "det"}) in keep]
    others = {}
    if a.preds_file and not os.path.exists(a.preds_file):
        raise SystemExit(f"no {a.preds_file}")
    if a.preds_file:
        for r in read_jsonl(a.preds_file):
            others[r["pano_id"]] = [tuple(q) for q in r["preds"] if q[2] >= a.threshold]
    anns = load_split(a.root, a.split) if a.root else {}
    items = []
    for c in cands:
        ann = anns.get(c["pano_id"], {})
        sup = ann.get("supervision") or {}
        items.append({**c, "kind": "det", "neg_disc": a.neg_disc,
                      "gt": [tuple(q) for q in ann.get("curb_ramp_points_normalized", [])],
                      "ignore": [tuple(q) for q in sup.get("ignore_points", [])],
                      "dets": others.get(c["pano_id"], [(c["x"], c["y"], c["conf"])])})
    if a.blind:
        random.Random(BOOT_SEED).shuffle(items)
    else:
        items.sort(key=lambda it: -it["conf"])
    prefill = prefill_for(items, load_prefill(a.prefill))

    def image_path(pid):
        if a.pano_dir:                      # flat, or a pano store's <id[:2]>/<id>.jpg
            flat = os.path.join(a.pano_dir, f"{pid}.jpg")
            return flat if os.path.exists(flat) else os.path.join(a.pano_dir, pid[:2], f"{pid}.jpg")
        return os.path.join(a.root, a.split, f"{pid}.jpg")

    os.makedirs(a.out, exist_ok=True)
    title, question = SECTION_HEAD["det"]
    if a.question:
        question = a.question
    cards = [f'<h2 id="sec-det">{title} ({len(items)} cards)</h2><p class="q">{question}</p>']
    cache, crop_index = {}, []
    if a.crops_out:
        os.makedirs(a.crops_out, exist_ok=True)
    for it in items:
        if it["pano_id"] not in cache:
            cache.clear()
            cache[it["pano_id"]] = Image.open(image_path(it["pano_id"])).convert("RGB")
        cap = (f"{it['pano_id']} ({it['x']:.3f}, {it['y']:.3f})" if a.blind else
               f"conf {it['conf']:.2f} · {it.get('city') or '?'} · {it.get('mode') or '?'} · {it.get('year') or '?'} · "
               f"{it['pano_id']} ({it['x']:.3f}, {it['y']:.3f})")
        jpeg = crop_card(cache[it["pano_id"]], it, half_w=0.08 / a.zoom)
        views = None
        if a.big:      # the large view: a wider field at about native resolution, with and without marks
            views = {"big": crop_card(cache[it["pano_id"]], it, half_w=BIG_HALF_W, out_w=BIG_W),
                     "big_clean": crop_card(cache[it["pano_id"]], it, half_w=BIG_HALF_W, out_w=BIG_W, marks=False)}
        cards.append(card_figure(it, jpeg, cap, views))
        if a.crops_out:                       # the same crop, as a file, for a blind pre-screen (no conf in the name)
            crop_index.append({"crop": f"{len(crop_index):05d}.jpg", "card_id": card_id(it)})
            with open(os.path.join(a.crops_out, crop_index[-1]["crop"]), "wb") as f:
                f.write(jpeg)
    if a.crops_out:
        with open(os.path.join(a.crops_out, "index.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["crop", "card_id"])
            w.writeheader()
            w.writerows(crop_index)
    intro = f"""<p>Each card asks about <strong>one</strong> detection: the large outlined cross. Other marks are
context.</p>
<ul class="legend">
<li>{LEGEND_GLYPHS['subject_det']} the detection this card asks about</li>
<li>{LEGEND_GLYPHS['other_det']} another detection on this pano (context; it may have its own card)</li>
<li>{LEGEND_GLYPHS['other_gt']} a training label, where the pano has one (context)</li>
<li>{LEGEND_GLYPHS['radius']} the scorer's match radius</li>
{NEG_LEGEND if a.neg_disc else ""}
</ul>
<p>{"<strong>Click an image</strong> for a large view (wider, sharper). There: <kbd>1</kbd>-<kbd>3</kbd> rate and move to the next card, <kbd>M</kbd> hides or shows all marks, arrows step, <kbd>Esc</kbd> closes. " if a.big else ""}{DEFINITIONS[a.definition]} Bike-lane, bus-lane, parking and rail markings are <em>not</em> crosswalks. Keys: <kbd>1</kbd>-<kbd>3</kbd>
rate (jumps to the next card), <kbd>n</kbd> = next unrated. Verdicts save in this browser; press <em>Export CSV</em>
when done.</p>"""
    if a.no_page:
        log(f"{len(crop_index)} crops -> {a.crops_out} (no page)")
        return
    page = rating_page(a.label, a.heading or f"{a.label}: rate detections", intro, "", cards, prefill)
    path = os.path.join(a.out, f"rate_{a.label}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    log(f"rating page -> {path} ({len(items)} cards, {len(prefill)} prefilled)")


MINE_DEDUP = 0.02     # two mined peaks on one pano closer than this (pano widths) are one candidate


def mine_candidates(ann, preds, threshold):
    """Hard-negative candidates on one TRAIN pano (#274): peaks >= threshold where training gives no background
    signal, more than RADIUS (matcher units) from every positive. `sparse` panos: any such peak (`sparse_bg`, or
    `sparse_ignore_pt` within RADIUS of an ignore point). `dense_ignore` panos: only peaks inside an ignore point's
    radius or an ignore box (`dense_ignored`); their other background already trains as "no". `verified_empty`
    panos train everywhere and give none. Peaks closer than MINE_DEDUP keep the most confident.

    >>> ann = {"curb_ramp_points_normalized": [[0.5, 0.6]], "supervision": {"mode": "dense_ignore",
    ...        "ignore_points": [[0.2, 0.6]], "ignore_boxes": [[0.7, 0.8, 0.5, 0.7]]}}
    >>> [(c["x"], c["reason"]) for c in mine_candidates(ann, [[0.51, 0.6, 0.9], [0.21, 0.6, 0.7], [0.75, 0.6, 0.6],
    ...                                                      [0.76, 0.6, 0.3], [0.9, 0.9, 0.5], [0.3, 0.6, 0.1]], 0.21)]
    [(0.21, 'dense_ignored'), (0.75, 'dense_ignored')]
    >>> [c["reason"] for c in mine_candidates({"curb_ramp_points_normalized": [[0.5, 0.6]], "supervision": {
    ...     "mode": "sparse", "ignore_points": [[0.2, 0.6]]}}, [[0.2, 0.6, 0.5], [0.9, 0.6, 0.4]], 0.21)]
    ['sparse_ignore_pt', 'sparse_bg']
    """
    sup = ann.get("supervision") or {}
    mode = sup.get("mode")
    if mode not in ("sparse", "dense_ignore"):
        return []
    gt = [tuple(p) for p in ann.get("curb_ramp_points_normalized", [])]
    ign = [tuple(p) for p in sup.get("ignore_points", [])]
    boxes = sup.get("ignore_boxes", []) if mode == "dense_ignore" else []

    def near(x, y, pts):
        return any(dist_sq(x, y, px, py, SCALE_X, SCALE_Y, True) < RADIUS_SQ for px, py in pts)

    out = []
    for x, y, c in sorted((q for q in preds if q[2] >= threshold), key=lambda q: -q[2]):
        if near(x, y, gt):
            continue
        on_ign = near(x, y, ign)
        if mode == "sparse":
            reason = "sparse_ignore_pt" if on_ign else "sparse_bg"
        elif on_ign or any(in_box(x, y, b) for b in boxes):
            reason = "dense_ignored"
        else:
            continue
        if any(wrap_dist(x, y, o["x"], o["y"]) < MINE_DEDUP for o in out):
            continue
        out.append({"x": x, "y": y, "conf": c, "reason": reason})
    return out


def cmd_mine(a):
    """Source A of the hard-negative plan (#274): `mine_candidates` over every train pano -> candidates.csv
    (pano_id, x, y, conf, city, mode, year, stratum = reason) and mine_meta.json with counts."""
    anns = load_split(a.root, "train")
    preds = {r["pano_id"]: r["preds"] for r in read_jsonl(a.preds_file)}
    missing = set(anns) - set(preds)
    if missing:
        raise SystemExit(f"{len(missing)} train panos have no prediction line in {a.preds_file}")
    rows = []
    for pid, ann in anns.items():
        sup = ann.get("supervision") or {}
        for c in mine_candidates(ann, preds[pid], a.threshold):
            rows.append({"pano_id": pid, "x": c["x"], "y": c["y"], "conf": c["conf"], "city": sup.get("city"),
                         "mode": sup.get("mode"), "year": year_bin(ann.get("capture_date")), "stratum": c["reason"]})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "candidates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        k = f"{r['city']}/{r['stratum']}"
        by[k] = by.get(k, 0) + 1
    meta = {"label": a.label, "threshold": a.threshold, "train_panos": len(anns), "candidates": len(rows),
            "panos_with_candidates": len({r["pano_id"] for r in rows}), "by_city_reason": dict(sorted(by.items())),
            "preds_sha256": sha256(a.preds_file), "dedup": MINE_DEDUP, "radius": RADIUS}
    with open(os.path.join(a.out, "mine_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(json.dumps(meta, indent=1))


AI_VERDICTS = ("crosswalk", "not_crosswalk", "unsure")
BIG_HALF_W, BIG_W = 0.12, 960   # the large view: 0.24 of the pano width (983 px of a 4096 pano) shown at 960 px
NEG_LEGEND = ('<li><svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" style="vertical-align:middle;'
              'background:#777"><circle cx="17" cy="17" r="13" fill="none" stroke="#00e6ff" stroke-width="3" '
              'stroke-dasharray="4 3"/></svg> <strong>hard-negative disc</strong>: if you answer "not a crosswalk", '
              'everything inside it trains as "no crosswalk". So answer <em>not a crosswalk</em> only when there is '
              'no crosswalk paint anywhere inside the dashed circle; if some crosswalk paint is inside it, answer '
              '<em>crosswalk</em>.</li>')


def cmd_sheets(a):
    """Tile --crops-out crops into numbered contact sheets (cols x rows, cell number drawn in each corner) so a
    blind rater reads a dozen crops per image. sheets.csv maps (sheet, cell) -> crop -> card_id."""
    from PIL import Image, ImageDraw
    with open(os.path.join(a.crops, "index.csv"), encoding="utf-8", newline="") as f:
        idx = list(csv.DictReader(f))
    os.makedirs(a.out, exist_ok=True)
    per = a.cols * a.rows
    rows = []
    for k in range(0, len(idx), per):
        chunk = idx[k:k + per]
        tiles = [Image.open(os.path.join(a.crops, r["crop"])).convert("RGB") for r in chunk]
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (a.cols * tw, a.rows * th), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        name = f"sheet_{k // per + 1:04d}.jpg"
        for i, (r, t) in enumerate(zip(chunk, tiles)):
            x, y = (i % a.cols) * tw, (i // a.cols) * th
            sheet.paste(t, (x, y))
            d.rectangle((x, y, x + 34, y + 24), fill=(255, 255, 255), outline=(0, 0, 0))
            d.text((x + 6, y + 5), str(i + 1), fill=(0, 0, 0))
            d.rectangle((x, y, x + tw - 1, y + th - 1), outline=(0, 0, 0), width=2)
            rows.append({"sheet": name, "cell": i + 1, "crop": r["crop"], "card_id": r["card_id"]})
        sheet.save(os.path.join(a.out, name), quality=88)
    with open(os.path.join(a.out, "sheets.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "cell", "crop", "card_id"])
        w.writeheader()
        w.writerows(rows)
    log(f"{len(rows)} crops -> {len(rows) // per + (1 if len(rows) % per else 0)} sheets in {a.out}")


def cmd_prescreen_merge(a):
    """Join the AI's per-cell verdicts to card ids (refusing a missing, duplicate or unknown cell), write
    prescreen.csv, and pick the cards Jon confirms: every AI not/unsure call (or a seeded --max-flagged of them)
    plus a seeded --n-crosswalk of its crosswalk calls (to measure what the AI misses) -> jon_selection.csv."""
    with open(os.path.join(a.sheets, "sheets.csv"), encoding="utf-8", newline="") as f:
        cells = {(r["sheet"], int(r["cell"])): r for r in csv.DictReader(f)}
    got = {}
    for path in a.verdicts:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = (r["sheet"].strip(), int(r["cell"]))
                v = r["verdict"].strip()
                if key not in cells or v not in AI_VERDICTS or key in got:
                    raise SystemExit(f"{path}: bad or duplicate row {r}")
                got[key] = (v, r.get("reason", "").strip())
    missing = set(cells) - set(got)
    if missing:
        raise SystemExit(f"{len(missing)} cells have no AI verdict, e.g. {sorted(missing)[:3]}")
    out = [{"card_id": cells[k]["card_id"], "sheet": k[0], "cell": k[1], "ai_verdict": got[k][0],
            "ai_reason": got[k][1]} for k in sorted(cells)]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prescreen.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    rng = random.Random(BOOT_SEED)
    flag = set(a.flag_verdicts.split(","))
    flagged = [r for r in out if r["ai_verdict"] in flag]
    if a.candidates:              # every flagged card in --full-cities, a seeded --n-other from the rest
        with open(a.candidates, encoding="utf-8", newline="") as f:
            city = {card_id({**c, "kind": "det", "x": float(c["x"]), "y": float(c["y"])}): c["city"]
                    for c in csv.DictReader(f)}
        full = set(a.full_cities.split(",")) if a.full_cities else set()
        keep = [r for r in flagged if city[r["card_id"]] in full]
        rest = [r for r in flagged if city[r["card_id"]] not in full]
        flagged = keep + rng.sample(rest, min(a.n_other, len(rest)))
    elif a.max_flagged and len(flagged) > a.max_flagged:
        flagged = rng.sample(flagged, a.max_flagged)
    yes = [r for r in out if r["ai_verdict"] == "crosswalk"]
    yes = rng.sample(yes, min(a.n_crosswalk, len(yes)))
    sel = [{"card_id": r["card_id"], "ai_verdict": r["ai_verdict"]} for r in flagged + yes]
    with open(os.path.join(a.out, "jon_selection.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["card_id", "ai_verdict"])
        w.writeheader()
        w.writerows(sel)
    counts = {v: sum(1 for r in out if r["ai_verdict"] == v) for v in AI_VERDICTS}
    meta = {"cards": len(out), "ai": counts, "selected_flagged": len(flagged), "selected_crosswalk": len(yes),
            "seed": BOOT_SEED, "flag_verdicts": a.flag_verdicts, "full_cities": a.full_cities,
            "n_other": a.n_other}
    with open(os.path.join(a.out, "prescreen_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(json.dumps(meta))


def cmd_select(a):
    """A seeded draw (BOOT_SEED) of --n pre-screened candidates matching the filters and not in any --exclude
    ratings file -> a selection CSV for `cards --only`. Records the pool size in <out>.json."""
    with open(a.prescreen, encoding="utf-8", newline="") as f:
        ai = {r["card_id"]: r["ai_verdict"] for r in csv.DictReader(f)}
    done = set()
    for path in a.exclude:
        with open(path, encoding="utf-8", newline="") as f:
            done |= {r["card_id"] for r in csv.DictReader(f)}
    want_ai = set(a.ai.split(","))
    strata = set(a.strata.split(",")) if a.strata else None
    cities = set(a.cities.split(",")) if a.cities else None
    pool = []
    with open(a.candidates, encoding="utf-8", newline="") as f:
        for c in csv.DictReader(f):
            cid = card_id({**c, "kind": "det", "x": float(c["x"]), "y": float(c["y"])})
            if (ai[cid] in want_ai and cid not in done and (strata is None or c["stratum"] in strata)
                    and (cities is None or c["city"] in cities)):
                pool.append(cid)
    pick = random.Random(BOOT_SEED).sample(sorted(pool), min(a.n, len(pool)))
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "ai_verdict"])
        w.writerows([cid, ai[cid]] for cid in pick)
    meta = {"pool": len(pool), "picked": len(pick), "ai": a.ai, "strata": a.strata, "cities": a.cities,
            "excluded_rated": len(done), "seed": BOOT_SEED}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(json.dumps(meta))


AUDIT_PANOS = 120     # (b') panos drawn for the deployment audit (#274 hard negatives, Step 0)


def cmd_bprime_audit(a):
    """Step 0 of the hard-negative plan (#274): a census of one model's detections on a seeded draw of (b') panos.

    Draws --n-panos of the (b') panos uniformly (seed BOOT_SEED) and lists EVERY detection at or above
    --threshold on them (stratum `random`), so look-alikes per 100 panos and precision are census figures on
    that draw. Adds every detection on the panos of the model's node false alarms at nodes SDOT calls unmarked
    whose OSM class is unmarked or unknown (stratum `node_fa`; a pano in both is `random+node_fa`); those only
    explain (b')'s false alarms and never enter the per-pano rates. Writes candidates.csv (for `cards`) and
    panos.csv (every drawn pano, with its detection count and the (b') weights of the nodes it was scored for).
    """
    drawn = set(draw_panos(a.ids_file, a.n_panos, a.draw_seed, a.exclude_panos))
    fa = set()
    if a.nodes:
        with open(a.nodes, encoding="utf-8", newline="") as f:
            nodes = [r for r in csv.DictReader(f) if r["model"] == a.label]
        fa = {r["pano_id"] for r in nodes
              if r["truth"] == "0" and r["detector"] == "1" and r["osm_class"] in ("unmarked", "unknown")}
    weights, years = {}, {}
    if a.sample:
        with open(a.sample, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                weights.setdefault(r["pano_id"], []).append(r["weight"])
                years[r["pano_id"]] = year_bin(r["capture_date"])
    preds = {r["pano_id"]: r["preds"] for r in read_jsonl(a.preds_file)}
    missing = (drawn | fa) - set(preds)
    if missing:
        raise SystemExit(f"{len(missing)} audit panos have no prediction line in {a.preds_file}")
    os.makedirs(a.out, exist_ok=True)
    cand, pano_rows = [], []
    for pid in sorted(drawn | fa):
        stratum = "+".join(s for s, inn in (("random", pid in drawn), ("node_fa", pid in fa)) if inn)
        dets = [q for q in preds[pid] if q[2] >= a.threshold]
        pano_rows.append({"pano_id": pid, "stratum": stratum, "year": years.get(pid), "n_dets": len(dets),
                          "node_weights": ";".join(weights.get(pid, []))})
        for x, y, c in dets:
            cand.append({"pano_id": pid, "x": x, "y": y, "conf": c, "city": a.city, "mode": a.mode,
                         "year": years.get(pid), "stratum": stratum})
    fields = {"candidates.csv": ["pano_id", "x", "y", "conf", "city", "mode", "year", "stratum"],
              "panos.csv": ["pano_id", "stratum", "year", "n_dets", "node_weights"]}
    for name, rows in (("candidates.csv", cand), ("panos.csv", pano_rows)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields[name])
            w.writeheader()
            w.writerows(rows)
    meta = {"label": a.label, "threshold": a.threshold, "seed": a.draw_seed, "n_panos": a.n_panos,
            "excluded_from": a.exclude_panos, "city": a.city,
            "random_panos": len(drawn), "node_fa_panos": len(fa), "node_fa_only_panos": len(fa - drawn),
            "detections": len(cand), "detections_random": sum(1 for c in cand if "random" in c["stratum"])}
    with open(os.path.join(a.out, "audit_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(json.dumps(meta))


def audit_rates(panos, verdicts):
    """Census rates on a set of drawn panos. `panos` = [pano_id], `verdicts` = {pano_id: [verdict, ...]} (one per
    detection; a drawn pano with no detection has none). Unsure counts as an error in `precision` and is left
    out of `precision_excl_unsure`.

    >>> r = audit_rates(["a", "b", "c", "d"], {"a": ["crosswalk", "not_crosswalk"], "b": ["unsure"]})
    >>> r["detections"], r["precision"], r["precision_excl_unsure"], r["lookalikes_per_100_panos"]
    (3, 0.3333333333333333, 0.5, 25.0)
    """
    v = [x for p in panos for x in verdicts.get(p, [])]
    n, ok = len(v), v.count("crosswalk")
    bad, uns = v.count("not_crosswalk"), v.count("unsure")
    return {"panos": len(panos), "detections": n, "crosswalk": ok, "not_crosswalk": bad, "unsure": uns,
            "precision": ok / n if n else float("nan"),
            "precision_excl_unsure": ok / (ok + bad) if ok + bad else float("nan"),
            "lookalikes_per_100_panos": 100 * bad / len(panos),
            "errors_per_100_panos": 100 * (bad + uns) / len(panos)}


def cmd_audit_summary(a):
    """Rates from a rated `bprime-audit` draw: precision and look-alikes per 100 panos on the `random` stratum,
    with 95% intervals from a bootstrap over panos (detections on one pano are not independent), plus verdict
    counts on the `node_fa` stratum. Writes audit_summary.json and per_pano.csv to --out."""
    with open(a.panos, encoding="utf-8", newline="") as f:
        prow = list(csv.DictReader(f))
    verdicts, n_rated = {}, 0
    with open(a.ratings, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if not r["verdict"]:
                raise SystemExit(f"unrated card {r['card_id']}: rate every card before summarising")
            verdicts.setdefault(r["pano_id"], []).append(r["verdict"])
            n_rated += 1
    drawn = [r["pano_id"] for r in prow if "random" in r["stratum"]]
    fa = [r["pano_id"] for r in prow if "node_fa" in r["stratum"]]
    want = sum(int(r["n_dets"]) for r in prow)
    if n_rated != want:
        raise SystemExit(f"{n_rated} rated cards but panos.csv lists {want} detections")
    point = audit_rates(drawn, verdicts)
    rng = random.Random(BOOT_SEED)
    keys = ("precision", "precision_excl_unsure", "lookalikes_per_100_panos", "errors_per_100_panos")
    boots = {k: [] for k in keys}
    for _ in range(BOOT_N):
        r = audit_rates([rng.choice(drawn) for _ in drawn], verdicts)
        for k in keys:
            boots[k].append(r[k])
    ci = {}
    for k in keys:
        b = sorted(x for x in boots[k] if x == x)
        ci[k] = [b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]]
    fa_v = [x for p in fa for x in verdicts.get(p, [])]
    out = {"label": a.label, "random": point, "random_ci95_pano_bootstrap": ci, "boot_n": BOOT_N,
           "boot_seed": BOOT_SEED,
           "node_fa": {"panos": len(fa), "detections": len(fa_v),
                       **{k: fa_v.count(k) for k in ("crosswalk", "not_crosswalk", "unsure")}}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out, "per_pano.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pano_id", "stratum", "year", "detections", "crosswalk", "not_crosswalk", "unsure"])
        for r in prow:
            v = verdicts.get(r["pano_id"], [])
            w.writerow([r["pano_id"], r["stratum"], r["year"], len(v), v.count("crosswalk"),
                        v.count("not_crosswalk"), v.count("unsure")])
    log(json.dumps(out, indent=1))


def read_ids(path):
    """Pano ids from a file of `<id>.jpg` lines (paths allowed), as (b')'s pano_files.txt lists them."""
    return sorted(l.strip().split("/")[-1][:-4] for l in open(path, encoding="utf-8") if l.strip())


def draw_panos(ids_file, n, seed=BOOT_SEED, exclude=()):
    """The seeded audit draw: `n` ids sampled from `ids_file` minus every pano listed in the `exclude` panos.csv
    files. With the defaults it is Step 0's draw exactly (seed 274, nothing excluded)."""
    ids = read_ids(ids_file)
    if exclude:
        gone = set()
        for p in exclude:
            with open(p, encoding="utf-8", newline="") as f:
                gone |= {r["pano_id"] for r in csv.DictReader(f)}
        ids = [i for i in ids if i not in gone]
    if n > len(ids):
        raise SystemExit(f"cannot draw {n} of {len(ids)} panos")
    return random.Random(seed).sample(ids, n)


def cmd_draw(a):
    """Write the seeded draw (one `<id>.jpg` per line) before any imagery or prediction exists for it, so the panos
    to fetch and infer are fixed first; `bprime-audit` with the same arguments redraws the identical set."""
    drawn = sorted(draw_panos(a.ids_file, a.n_panos, a.draw_seed, a.exclude_panos))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{p}.jpg\n" for p in drawn)
    log(f"draw: {len(drawn)} panos (seed {a.draw_seed}) -> {a.out}")


def recall_matched_threshold(sweep, base, new, split="val", base_threshold=None):
    """The highest grid threshold at which `new`'s recall on `split` is at least `base`'s at base's own val-F1
    pick, or at `base_threshold` when given (s5 matches s4 at its adopted, itself recall-matched, thresholds).
    `sweep` = rows of threshold_sweep.csv; base's pick is the threshold maximising its val F1 (lowest on a tie,
    as pick_threshold does).

    >>> rows = [{"model": m, "split": "val", "threshold": t, "f1": f, "recall": r} for m, t, f, r in [
    ...     ("a", "0.2", "0.80", "0.81"), ("a", "0.3", "0.79", "0.78"),
    ...     ("b", "0.2", "0.80", "0.84"), ("b", "0.3", "0.81", "0.80")]]
    >>> recall_matched_threshold(rows, "a", "b")
    0.2
    >>> recall_matched_threshold(rows, "a", "b", base_threshold=0.3)
    0.3
    """
    def mine(m):
        return sorted(((float(r["threshold"]), float(r["f1"]), float(r["recall"])) for r in sweep
                       if r["model"] == m and r["split"] == split), key=lambda q: q[0])
    if base_threshold is None:
        t_b, _, r_b = max(mine(base), key=lambda q: (q[1], -q[0]))
    else:
        at = [q for q in mine(base) if abs(q[0] - base_threshold) < 1e-9]
        if not at:
            raise SystemExit(f"{base_threshold} is not on {base}'s {split} sweep grid")
        t_b, _, r_b = at[0]
    ok = [t for t, _, r in mine(new) if r >= r_b]
    if not ok:
        raise SystemExit(f"{new} never reaches {base}'s val recall {r_b}")
    return max(ok)


def cmd_match_threshold(a):
    with open(a.sweep, encoding="utf-8", newline="") as f:
        t = recall_matched_threshold(list(csv.DictReader(f)), a.base, a.new, base_threshold=a.base_threshold)
    print(t)


def merge_candidates(lists, r=PREFILL_R):
    """One card per paint across several models' candidate lists: a candidate within `r` (pano widths, seam
    wrapped) of an earlier card on the same pano is that card, so the rater sees it once. Returns the union rows
    (the first list's copy of each card), each with a `models` column naming every list it came from.

    >>> a = [{"pano_id": "p", "x": 0.500, "y": 0.55, "conf": 0.3}]
    >>> b = [{"pano_id": "p", "x": 0.506, "y": 0.55, "conf": 0.4}, {"pano_id": "p", "x": 0.7, "y": 0.55, "conf": 0.5}]
    >>> [(u["x"], u["models"]) for u in merge_candidates([("s1", a), ("s3", b)])]
    [(0.5, 's1;s3'), (0.7, 's3')]
    """
    out = []
    for lab, rows in lists:
        for c in rows:
            near = [u for u in out
                    if u["pano_id"] == c["pano_id"] and wrap_dist(u["x"], u["y"], c["x"], c["y"]) <= r]
            if near:
                u = min(near, key=lambda u: wrap_dist(u["x"], u["y"], c["x"], c["y"]))
                if lab not in u["models"].split(";"):
                    u["models"] += ";" + lab
            else:
                out.append({**c, "models": lab})
    return out


def cmd_merge_cards(a):
    """Union of several models' `bprime-audit` candidates (and several draws) for ONE blind page: `--candidates`
    takes label=path pairs. Writes candidates.csv (with `models`) and context.jsonl (every union card on its pano,
    for `cards --preds-file ... --threshold 0`, so the context marks name no model)."""
    lists = []
    for spec in a.candidates:
        lab, path = spec.split("=", 1)
        lists.append((lab, read_candidates(path)))
    union = merge_candidates(lists)
    os.makedirs(a.out, exist_ok=True)
    cols = ["pano_id", "x", "y", "conf", "city", "mode", "year", "stratum", "models"]
    with open(os.path.join(a.out, "candidates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(union)
    by = {}
    for u in union:
        by.setdefault(u["pano_id"], []).append([u["x"], u["y"], 1.0])
    with open(os.path.join(a.out, "context.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(by):
            f.write(json.dumps({"pano_id": pid, "preds": by[pid]}) + "\n")
    log(f"merge-cards: {sum(len(l) for _, l in lists)} candidates -> {len(union)} cards -> {a.out}")


def cmd_carry(a):
    """One model's rated cards from a merged page's export: each of its candidates takes the verdict of its own
    card (exact id, or the nearest card on the pano within PREFILL_R, as `merge-cards` grouped them). Refuses if
    any candidate has no rated card. The output feeds `audit-summary` and `audit-compare` unchanged."""
    items = [{**c, "kind": "det"} for c in read_candidates(a.candidates)]
    got = prefill_for(items, load_prefill(a.ratings))
    miss = [card_id(it) for it in items if card_id(it) not in got]
    if miss:
        raise SystemExit(f"{len(miss)} candidates have no rated card, e.g. {miss[:3]}")
    cols = ["card_id", "kind", "pano_id", "x", "y", "conf", "city", "mode", "year", "stratum", "verdict", "note"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({**it, "card_id": card_id(it), **got[card_id(it)]})
    log(f"carry: {len(items)} cards -> {a.out}")


RETENTION_MIN = 0.95   # crosswalk_s3 guard 2: share of the Step 0 crosswalks the new model still detects


def pct_interval(boots, level=0.95):
    """Percentile interval of sorted bootstrap values, at the indices every #274 interval has used at 95%.

    >>> pct_interval(list(range(2000)))
    [50, 1949]
    >>> pct_interval(list(range(2000)), 0.975)
    [25, 1974]
    """
    k = int(round((1 - level) / 2 * len(boots)))
    return [boots[k], boots[len(boots) - k - 1]]


def ci_key(level):
    """Output key suffix for an interval level: 0.95 -> 'ci95' (as written before s4), 0.975 -> 'ci975'.

    >>> ci_key(0.95), ci_key(0.975)
    ('ci95', 'ci975')
    """
    return "ci" + f"{level:g}"[2:]


def paired_lookalike_diff(panos, base, new, rng, n=BOOT_N, level=0.95):
    """Paired change in look-alikes per 100 panos, new minus base, on one set of drawn panos.

    `base` and `new` map pano_id -> look-alike count (a pano with none may be absent). The interval resamples
    panos with replacement and keeps each pano's two counts together, so pano-to-pano variation cancels.
    Returns (point, [lo, hi]) with a `level` percentile interval (95% unless the pre-registration says otherwise;
    s4's two primaries use 97.5%, Bonferroni).

    >>> import random as _r
    >>> pt, ci = paired_lookalike_diff(["a", "b", "c", "d"], {"a": 2, "b": 1}, {"a": 1}, _r.Random(0), n=200)
    >>> pt, ci[1] <= 0
    (-50.0, True)
    """
    d = [new.get(p, 0) - base.get(p, 0) for p in panos]
    point = 100 * sum(d) / len(d)
    boots = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return point, pct_interval(boots, level)


def cmd_retention(a):
    """How many of a base model's confirmed crosswalks each model still detects (crosswalk_s6 guards 4 and 5).

    A card is confirmed when its verdict in --ratings is `crosswalk`; a model keeps it when it has a detection at
    its threshold within the scorer's radius (0.05), the rule `audit-compare` uses. With several models the
    seed spread is the largest difference in the kept count between the first model and any other.
    """
    with open(a.ratings, encoding="utf-8", newline="") as f:
        real = [r for r in csv.DictReader(f) if r["verdict"] == "crosswalk"]
    res = {"ratings": a.ratings, "confirmed": len(real), "models": {}}
    for spec in a.models:
        label, rest = spec.split("=", 1)
        path, thr = rest.rsplit("@", 1)
        preds = {r["pano_id"]: [q for q in r["preds"] if q[2] >= float(thr)] for r in read_jsonl(path)}
        kept = sum(1 for r in real if any(dist_sq(float(r["x"]), float(r["y"]), q[0], q[1], SCALE_X, SCALE_Y, True)
                                          < RADIUS_SQ for q in preds.get(r["pano_id"], [])))
        res["models"][label] = {"threshold": float(thr), "kept": kept, "retention": round(kept / len(real), 4)}
    ks = [m["kept"] for m in res["models"].values()]
    res["seed_spread"] = max([abs(ks[0] - k) for k in ks[1:]] or [0])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "retention.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def cmd_audit_compare(a):
    """The crosswalk_s3 primary and retention guard (docs/stage1_crosswalk_sdot_274.md, pre-registered 2026-11-06).

    --base-* is the rated Step 0 draw of the working model, --new-* the same draw for the retrained one (the
    same seeded panos, so `bprime-audit` reproduces them). Primary: the paired difference in look-alikes per 100
    panos on the `random` stratum, with a pano bootstrap (BOOT_N, BOOT_SEED); it is met when the interval's upper
    end is below 0. Guard 2: of the base model's random-stratum detections rated crosswalk, the share with a new
    detection at or above --new-threshold within RADIUS (matcher units) on the same pano, at least
    RETENTION_MIN. `--ci` sets the interval level (s4's primaries: 0.975) for both paired intervals. Writes
    audit_compare.json to --out."""
    def load(panos_csv, ratings_csv):
        with open(panos_csv, encoding="utf-8", newline="") as f:
            drawn = sorted(r["pano_id"] for r in csv.DictReader(f) if "random" in r["stratum"])
        look, uns, cards = {}, {}, []
        with open(ratings_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if not r["verdict"]:
                    raise SystemExit(f"unrated card {r['card_id']} in {ratings_csv}")
                if r["pano_id"] not in drawn:
                    continue
                if r["verdict"] == "not_crosswalk":
                    look[r["pano_id"]] = look.get(r["pano_id"], 0) + 1
                elif r["verdict"] == "unsure":
                    uns[r["pano_id"]] = uns.get(r["pano_id"], 0) + 1
                cards.append(r)
        return drawn, look, uns, cards
    drawn, look_a, uns_a, cards_a = load(a.base_panos, a.base_ratings)
    drawn_b, look_b, uns_b, _ = load(a.new_panos, a.new_ratings)
    if drawn != drawn_b:
        raise SystemExit("the two draws differ: run bprime-audit with the same --ids-file and --n-panos")
    point, ci = paired_lookalike_diff(drawn, look_a, look_b, random.Random(BOOT_SEED), level=a.ci)
    err_a = {p: look_a.get(p, 0) + uns_a.get(p, 0) for p in drawn}
    err_b = {p: look_b.get(p, 0) + uns_b.get(p, 0) for p in drawn}
    e_point, e_ci = paired_lookalike_diff(drawn, err_a, err_b, random.Random(BOOT_SEED), level=a.ci)
    preds = {r["pano_id"]: [q for q in r["preds"] if q[2] >= a.new_threshold] for r in read_jsonl(a.new_preds)}
    real = [r for r in cards_a if r["verdict"] == "crosswalk"]
    kept = [r for r in real if any(dist_sq(float(r["x"]), float(r["y"]), q[0], q[1], SCALE_X, SCALE_Y, True)
                                   < RADIUS_SQ for q in preds.get(r["pano_id"], []))]
    retention = len(kept) / len(real) if real else float("nan")
    out = {"base": a.base_label, "new": a.new_label, "new_threshold": a.new_threshold, "panos": len(drawn),
           "lookalikes_per_100": {"base": 100 * sum(look_a.values()) / len(drawn),
                                  "new": 100 * sum(look_b.values()) / len(drawn)},
           "paired_diff_lookalikes_per_100": point, f"paired_diff_{ci_key(a.ci)}": ci,
           "primary_met": ci[1] < 0,
           "paired_diff_errors_per_100": e_point, f"paired_diff_errors_{ci_key(a.ci)}": e_ci,
           "retention": {"base_crosswalks": len(real), "still_detected": len(kept), "share": retention,
                         "min": RETENTION_MIN, "met": retention >= RETENTION_MIN},
           "boot_n": BOOT_N, "boot_seed": BOOT_SEED}
    if a.ci != 0.95:
        out["ci_level"] = a.ci
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "audit_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(json.dumps(out, indent=1))


def label_hits(labels, preds, threshold):
    """Per label: the best confidence of any detection within RADIUS (matcher units, seam wrapped) on its pano,
    and whether it reaches `threshold`. `labels` are dicts with pano_id, x, y; `preds` maps pano_id -> [[x, y, c]].

    >>> hs = label_hits([{"pano_id": "p", "x": 0.50, "y": 0.55}, {"pano_id": "q", "x": 0.2, "y": 0.6}],
    ...                 {"p": [[0.52, 0.55, 0.30], [0.9, 0.5, 0.9]]}, 0.21)
    >>> [(h["best_conf_within_radius"], h["hit"]) for h in hs]
    [(0.3, 1), (0.0, 0)]
    """
    out = []
    for r in labels:
        x, y = float(r["x"]), float(r["y"])
        near = [q[2] for q in preds.get(r["pano_id"], [])
                if dist_sq(x, y, q[0], q[1], SCALE_X, SCALE_Y, True) < RADIUS_SQ]
        best = max(near, default=0.0)
        out.append({**r, "best_conf_within_radius": round(best, 4), "hit": int(best >= threshold)})
    return out


def paired_recall_diff(hits_base, hits_new, rng, n=BOOT_N, level=0.95):
    """Label recall of two models on the same labels, and new minus base, with a pano bootstrap that keeps every
    label of a resampled pano and both models' hits on it together. Recall in a resample is hits over labels
    (a ratio, so a pano with more labels weighs more). `hits_*` are label_hits rows in the same order.
    Returns {"base": (point, ci), "new": (point, ci), "diff": (point, ci)}.

    >>> base = [{"pano_id": p, "hit": 0} for p in "abcd"]
    >>> new = [{"pano_id": p, "hit": 1} for p in "abc"] + [{"pano_id": "d", "hit": 0}]
    >>> r = paired_recall_diff(base, new, random.Random(0), n=200)
    >>> r["diff"][0], r["base"][0]
    (0.75, 0.0)
    """
    per = {}
    for hb, hn in zip(hits_base, hits_new):
        assert hb["pano_id"] == hn["pano_id"]
        k, b, w = per.get(hb["pano_id"], (0, 0, 0))
        per[hb["pano_id"]] = (k + 1, b + hb["hit"], w + hn["hit"])
    rows = [per[p] for p in sorted(per)]

    def rec(rs):
        k = sum(r[0] for r in rs)
        return sum(r[1] for r in rs) / k, sum(r[2] for r in rs) / k
    pb, pn = rec(rows)
    bb, bn, bd = [], [], []
    for _ in range(n):
        b, w = rec([rng.choice(rows) for _ in rows])
        bb.append(b)
        bn.append(w)
        bd.append(w - b)
    return {"base": (pb, pct_interval(sorted(bb), level)), "new": (pn, pct_interval(sorted(bn), level)),
            "diff": (pn - pb, pct_interval(sorted(bd), level))}


def cmd_label_recall(a):
    """(#274 crosswalk_s4, P1) Label recall on a fixed set of human-confirmed labels (Columbia SC brick: Jon's 210
    Step A "crosswalk" verdicts) for a base and a new model at their thresholds: a label is hit when the model has
    a detection at its threshold within RADIUS. The statistic is the paired difference new - base, bootstrapped
    over panos (BOOT_N, BOOT_SEED) at `--ci` (s4: 0.975); met when the interval's lower end is above 0. `--also`
    adds models reported alongside, each paired against base. Every label's pano must have a prediction line in
    every preds file. Writes label_hits.csv and label_recall.json to --out."""
    with open(a.labels, encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    uids = [r["label_uid"] for r in labels]
    if len(set(uids)) != len(uids):
        raise SystemExit("label_uid is not unique in --labels")

    def parse(spec):
        lab, rest = spec.split("=", 1)
        path, thr = rest.rsplit("@", 1)
        preds = {r["pano_id"]: r["preds"] for r in read_jsonl(path)}
        missing = sorted({r["pano_id"] for r in labels} - set(preds))
        if missing:
            raise SystemExit(f"{lab}: {len(missing)} label panos have no prediction line, e.g. {missing[:3]}")
        return lab, path, float(thr), preds
    models = [parse(s) for s in [a.base, a.new] + list(a.also or [])]
    hits = {lab: label_hits(labels, preds, thr) for lab, _, thr, preds in models}
    base, new = models[0][0], models[1][0]
    out = {"labels": len(labels), "panos": len({r["pano_id"] for r in labels}), "base": base, "new": new,
           "ci_level": a.ci, "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "radius": RADIUS,
           "models": {lab: {"preds": path.replace(os.sep, "/"), "sha256": sha256(path), "threshold": thr,
                            "hits": sum(h["hit"] for h in hits[lab])} for lab, path, thr, _ in models},
           "paired": {}}
    for lab, _, _, _ in models[1:]:
        r = paired_recall_diff(hits[base], hits[lab], random.Random(BOOT_SEED), level=a.ci)
        out["models"][base]["recall"], out["models"][base][ci_key(a.ci)] = r["base"]
        out["models"][lab]["recall"], out["models"][lab][ci_key(a.ci)] = r["new"]
        out["paired"][f"{lab} - {base}"] = {"diff": r["diff"][0], ci_key(a.ci): r["diff"][1]}
    out["primary_met"] = out["paired"][f"{new} - {base}"][ci_key(a.ci)][0] > 0
    os.makedirs(a.out, exist_ok=True)
    cols = ["model", "threshold"] + list(labels[0].keys()) + ["best_conf_within_radius", "hit"]
    with open(os.path.join(a.out, "label_hits.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for lab, _, thr, _ in models:
            w.writerows({"model": lab, "threshold": thr, **h} for h in hits[lab])
    with open(os.path.join(a.out, "label_recall.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--root", required=True)
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--label", required=True)
    i.add_argument("--preds", required=True)
    i.add_argument("--splits", default=",".join(SPLITS))
    i.add_argument("--sha256", default=None, help="refuse unless the checkpoint hashes to this")
    i.add_argument("--ids-file", default=None, help="(b') pano_files.txt; infer these from --pano-dir instead")
    i.add_argument("--pano-dir", default=None, help="(b') flat dir of bprime-prep JPEGs")
    i.add_argument("--split-name", default="bprime")
    s = sub.add_parser("score")
    s.add_argument("--root", required=True)
    s.add_argument("--preds", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--baseline-nodes", required=True)
    s.add_argument("--census", required=True)
    s.add_argument("--meta-dir", default=os.path.join(REPO, "analysis_out", "stage1_crosswalk_crowd_274"))
    s.add_argument("--fixed-threshold", nargs="*", default=[],
                   help="label=t: score that label at t instead of its val-F1 pick (recorded as threshold_rule)")
    s.add_argument("--out", required=True)
    g = sub.add_parser("figures")
    g.add_argument("--out", required=True)
    g.add_argument("--fig-dir", default=os.path.join(REPO, "docs", "figures", "stage1_crosswalk_detector_274"))
    o = sub.add_parser("oracle", help="write the training labels themselves as a prediction set (the ceiling)")
    o.add_argument("--root", required=True)
    o.add_argument("--preds", required=True)
    o.add_argument("--label", default="label_ceiling")
    q = sub.add_parser("gallery", help="local HTML of a model's test FPs / misses / TPs (untracked)")
    q.add_argument("--root", required=True)
    q.add_argument("--preds", required=True)
    q.add_argument("--scores", required=True, help="the score --out dir (for the threshold)")
    q.add_argument("--label", required=True)
    q.add_argument("--threshold", type=float, default=None)
    q.add_argument("--n", type=int, default=60, help="misses sampled (every FP is shown)")
    q.add_argument("--n-tp", type=int, default=30, help="TPs sampled")
    q.add_argument("--prefill", default=None, help="CSV of earlier verdicts: card_id, verdict, note")
    q.add_argument("--out", required=True)
    c = sub.add_parser("cards", help="local HTML rating page for any list of detections (untracked)")
    c.add_argument("--candidates", required=True, help="CSV: pano_id, x, y, conf [, city, mode, year, stratum]")
    c.add_argument("--label", required=True, help="names the page, its browser storage and the export")
    c.add_argument("--heading", default=None)
    c.add_argument("--pano-dir", default=None, help="dir of <pano_id>.jpg, flat or sharded as a pano store (else --root/--split)")
    c.add_argument("--root", default=None, help="Stage 2 root, for labels / ignore points as context")
    c.add_argument("--split", default="train")
    c.add_argument("--preds-file", default=None, help="the model's <split>.jsonl, for its other detections")
    c.add_argument("--threshold", type=float, default=0.21, help="context detections at or above this")
    c.add_argument("--blind", action="store_true", help="captions without conf/city/mode/year; seeded shuffle")
    c.add_argument("--prefill", default=None, help="CSV of earlier verdicts: card_id, verdict, note")
    c.add_argument("--out", required=True)
    c.add_argument("--crops-out", default=None, help="also write each card's crop as <n>.jpg + index.csv here")
    c.add_argument("--no-page", action="store_true", help="with --crops-out: write the crops only")
    c.add_argument("--neg-disc", action="store_true", help="draw the background disc a confirmed negative trains")
    c.add_argument("--zoom", type=float, default=1.0, help="crop magnification (1 = the standard 0.16-width crop)")
    c.add_argument("--definition", choices=sorted(DEFINITIONS), default="paint",
                   help="what counts as a crosswalk on the page (paint_brick from s4 on)")
    c.add_argument("--question", default=None, help="replace the section's question (e.g. for crowd-label cards)")
    c.add_argument("--big", action=argparse.BooleanOptionalAction, default=True,
                   help="click-to-enlarge view (wider, ~native res) with a marks toggle; on by default (Jon, 2026-11-08), "
                        "--no-big rebuilds the pages made before that")
    e = sub.add_parser("select", help="seeded selection of pre-screened candidates not yet rated, for another page")
    e.add_argument("--prescreen", required=True, help="prescreen.csv")
    e.add_argument("--candidates", required=True, help="mine candidates.csv")
    e.add_argument("--exclude", nargs="*", default=[], help="ratings CSVs whose card_ids are already rated")
    e.add_argument("--ai", default="not_crosswalk", help="comma list of AI verdicts to draw from")
    e.add_argument("--strata", default=None, help="comma list of mining reasons (candidates.csv stratum)")
    e.add_argument("--cities", default=None, help="comma list of cities")
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--out", required=True, help="selection CSV (card_id, ai_verdict)")
    h = sub.add_parser("sheets", help="numbered contact sheets of --crops-out crops, for the blind AI pre-screen")
    h.add_argument("--crops", required=True, help="a --crops-out dir (index.csv + <n>.jpg)")
    h.add_argument("--cols", type=int, default=3)
    h.add_argument("--rows", type=int, default=4)
    h.add_argument("--out", required=True)
    g = sub.add_parser("prescreen-merge", help="AI sheet verdicts -> per-card prescreen.csv + the page selection")
    g.add_argument("--sheets", required=True, help="the sheets dir (sheets.csv)")
    g.add_argument("--verdicts", required=True, nargs="+", help="AI CSVs: sheet, cell, verdict, reason")
    g.add_argument("--n-crosswalk", type=int, default=50, help="AI 'crosswalk' calls sampled for Jon (miss rate)")
    g.add_argument("--max-flagged", type=int, default=0, help="cap on AI not/unsure cards for Jon (0 = all)")
    g.add_argument("--flag-verdicts", default="not_crosswalk,unsure", help="AI verdicts that put a card on the page")
    g.add_argument("--candidates", default=None, help="mine candidates.csv (cities), for --full-cities/--n-other")
    g.add_argument("--full-cities", default=None, help="comma list: every flagged card from these cities")
    g.add_argument("--n-other", type=int, default=0, help="seeded sample of flagged cards from the other cities")
    g.add_argument("--out", required=True)
    c.add_argument("--only", default=None, help="CSV with a card_id column: build cards for these candidates only")
    m = sub.add_parser("mine", help="(#274 hard negatives, Source A) candidate peaks on train panos")
    m.add_argument("--root", required=True)
    m.add_argument("--preds-file", required=True, help="the model's train.jsonl")
    m.add_argument("--label", default="crosswalk_s1")
    m.add_argument("--threshold", type=float, default=0.21)
    m.add_argument("--out", required=True)
    u = sub.add_parser("bprime-audit", help="(#274 hard negatives, Step 0) census of detections on drawn (b') panos")
    u.add_argument("--ids-file", required=True, help="bprime/pano_files.txt")
    u.add_argument("--sample", default=None, help="bprime/sample.csv (node weights, capture dates); omit for a city")
    u.add_argument("--nodes", default=None, help="the (b') nodes.csv with the model's per-node calls; omit for no "
                                                 "node_fa stratum")
    u.add_argument("--draw-seed", type=int, default=BOOT_SEED)
    u.add_argument("--exclude-panos", nargs="*", default=[], help="panos.csv of earlier draws: never redrawn")
    u.add_argument("--city", default=SEATTLE)
    u.add_argument("--mode", default="bprime")
    u.add_argument("--preds-file", required=True, help="the model's bprime.jsonl")
    u.add_argument("--label", required=True)
    u.add_argument("--threshold", type=float, required=True)
    u.add_argument("--n-panos", type=int, default=AUDIT_PANOS)
    u.add_argument("--out", required=True)
    d = sub.add_parser("draw", help="the seeded audit draw as an ids file, fixed before fetching or inference")
    d.add_argument("--ids-file", required=True)
    d.add_argument("--n-panos", type=int, default=AUDIT_PANOS)
    d.add_argument("--draw-seed", type=int, default=BOOT_SEED)
    d.add_argument("--exclude-panos", nargs="*", default=[])
    d.add_argument("--out", required=True)
    t = sub.add_parser("match-threshold", help="highest threshold where NEW's val recall >= BASE's at its pick")
    t.add_argument("--sweep", required=True, help="threshold_sweep.csv from `score`")
    t.add_argument("--base", default="crosswalk_s1")
    t.add_argument("--new", default="crosswalk_s3")
    t.add_argument("--base-threshold", type=float, help="match BASE's val recall at this threshold, not its pick")
    mc = sub.add_parser("merge-cards", help="one blind page's cards from several models' candidates")
    mc.add_argument("--candidates", nargs="+", required=True, help="label=candidates.csv, one per model and draw")
    mc.add_argument("--out", required=True)
    cr = sub.add_parser("carry", help="one model's rated cards from a merged page's export")
    cr.add_argument("--ratings", required=True)
    cr.add_argument("--candidates", required=True)
    cr.add_argument("--out", required=True)
    v = sub.add_parser("audit-summary", help="rates + pano-bootstrap intervals from a rated bprime-audit draw")
    v.add_argument("--panos", required=True, help="the draw's panos.csv")
    v.add_argument("--ratings", required=True, help="the exported ratings CSV for the draw's page")
    v.add_argument("--label", default="crosswalk_s1")
    v.add_argument("--out", required=True)
    k = sub.add_parser("audit-compare", help="(#274 crosswalk_s3) paired look-alike change + retention guard")
    for side in ("base", "new"):
        k.add_argument(f"--{side}-panos", required=True, help=f"{side} model's bprime-audit panos.csv")
        k.add_argument(f"--{side}-ratings", required=True, help=f"{side} model's rated cards (exported CSV)")
    k.add_argument("--new-preds", required=True, help="the new model's bprime.jsonl")
    k.add_argument("--new-threshold", type=float, required=True, help="the new model's val-F1 threshold")
    k.add_argument("--base-label", default="crosswalk_s1")
    k.add_argument("--new-label", default="crosswalk_s3")
    k.add_argument("--ci", type=float, default=0.95, help="interval level (s4's primaries: 0.975)")
    k.add_argument("--out", required=True)
    lr = sub.add_parser("label-recall", help="(#274 crosswalk_s4 P1) paired label recall on confirmed labels")
    lr.add_argument("--labels", required=True, help="CSV with label_uid, pano_id, x, y (e.g. columbia_gt.csv)")
    lr.add_argument("--base", required=True, help="label=preds.jsonl@threshold")
    lr.add_argument("--new", required=True, help="label=preds.jsonl@threshold")
    lr.add_argument("--also", nargs="*", help="more label=preds.jsonl@threshold, reported against base")
    lr.add_argument("--ci", type=float, default=0.975)
    lr.add_argument("--out", required=True)
    b = sub.add_parser("bprime-sample", help="(b') stratified Seattle node sample + the store panos to fetch")
    b.add_argument("--root", required=True)
    b.add_argument("--baseline-nodes", required=True)
    b.add_argument("--census", required=True)
    b.add_argument("--pano-data", required=True, help="pipe-separated export of sidewalk_seattle.pano_data")
    b.add_argument("--store-listing", required=True)
    b.add_argument("--meta-dir", default=os.path.join(REPO, "analysis_out", "stage1_crosswalk_crowd_274"))
    b.add_argument("--per-cell", type=int, default=BP_PER_CELL)
    b.add_argument("--out", required=True)
    rt = sub.add_parser("retention", help="confirmed crosswalks of a base model that each model still detects")
    rt.add_argument("--ratings", required=True, help="the base model's rated cards on the draw (ratings.csv)")
    rt.add_argument("--models", nargs="+", required=True, metavar="LABEL=PREDS.jsonl@THRESHOLD")
    rt.add_argument("--out", required=True)
    b = sub.add_parser("bprime-prep", help="(b') store JPEGs -> 4096x2048 as the training root was written")
    b.add_argument("--ids-file", required=True)
    b.add_argument("--store", nargs="+", required=True, help="store roots laid out <id[:2]>/<id>.jpg, or flat dirs")
    b.add_argument("--out-dir", required=True)
    b = sub.add_parser("bprime-score", help="(b') weighted node scores from the bprime predictions")
    b.add_argument("--bprime", required=True, help="dir holding sample.csv and preds/<label>/bprime.jsonl")
    b.add_argument("--scores", required=True, help="the (a)/(b) score dir: thresholds come from its scores.json")
    b.add_argument("--pano-data", required=True)
    b.add_argument("--labels", default="crosswalk_s1,released")
    b.add_argument("--out", default=None, help="where scores/nodes/report go (default: --bprime)")
    a = ap.parse_args()
    {"infer": cmd_infer, "score": cmd_score, "figures": cmd_figures, "oracle": cmd_oracle,
     "gallery": cmd_gallery, "bprime-sample": cmd_bprime_sample, "bprime-prep": cmd_bprime_prep, "retention": cmd_retention,
     "bprime-score": cmd_bprime_score, "cards": cmd_cards, "bprime-audit": cmd_bprime_audit,
     "audit-summary": cmd_audit_summary, "mine": cmd_mine, "sheets": cmd_sheets,
     "prescreen-merge": cmd_prescreen_merge, "select": cmd_select, "audit-compare": cmd_audit_compare,
     "draw": cmd_draw, "match-threshold": cmd_match_threshold, "merge-cards": cmd_merge_cards,
     "carry": cmd_carry, "label-recall": cmd_label_recall}[a.cmd](a)


BP_PER_CELL = 150             # (b') nodes per (osm_class x truth) cell
BP_EXCLUDE_M = 60.0           # split_dataset.py's grouping distance, from every train/val pano
BP_PANO_M = 25.0              # a node is scored from one store pano within this range


def read_pano_data(path):
    """pano_id -> (lat, lng, heading_deg, capture_date) from a pipe-separated export of a city's
    `pano_data` (pano_id|lat|lng|camera_heading|capture_date|has_backup|source|width)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) >= 5 and p[1] and p[3]:
                out[p[0]] = (float(p[1]), float(p[2]), float(p[3]), p[4] or None)
    return out


def cmd_bprime_sample(a):
    """(b') node sample, drawn before any (b') inference: every baseline node with census coordinates, more than
    BP_EXCLUDE_M from any train/val pano and with a store pano within BP_PANO_M, stratified by
    (osm_class x SDOT truth), BP_PER_CELL per cell (all of a smaller cell), seed 274. Each node gets the newest
    store pano within BP_PANO_M (ties: nearest). Weights restore the eligible population's cell sizes."""
    from pilot import Grid
    rng = random.Random(BOOT_SEED)
    pd = read_pano_data(a.pano_data)
    store = {l.split()[0][:-4] for l in open(a.store_listing, encoding="utf-8") if l.strip()}
    store_pos = [(v[0], v[1], pid) for pid, v in pd.items() if pid in store]
    gstore = Grid(store_pos)
    trainval = []
    for split in ("train", "val"):
        ids = [pid for pid, ann in load_split(a.root, split).items()
               if (ann.get("supervision") or {}).get("city") == SEATTLE]
        pos = pano_positions(a.meta_dir, ids)
        for pid in ids:
            if pid in pos:
                trainval.append((pos[pid][0], pos[pid][1], pid))
            elif pid in pd:
                trainval.append((pd[pid][0], pd[pid][1], pid))
            else:
                raise SystemExit(f"train/val pano {pid} has no position")
    gtv = Grid(trainval)
    node_xy = census_nodes(a.census)
    with open(a.baseline_nodes, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    drop = {"no_census_coords": 0, "near_train_val": 0, "no_store_pano": 0}
    eligible = {}
    for r in rows:
        nid = r["node_id"]
        if nid not in node_xy:
            drop["no_census_coords"] += 1
            continue
        la, ln = node_xy[nid]
        if gtv.within(la, ln, BP_EXCLUDE_M):
            drop["near_train_val"] += 1
            continue
        near = gstore.within(la, ln, BP_PANO_M)
        if not near:
            drop["no_store_pano"] += 1
            continue
        d, (_, _, pid) = max(near, key=lambda t: (pd[t[1][2]][3] or "", -t[0]))
        eligible.setdefault((r["osm_class"], int(r["truth"])), []).append(
            {"node_id": nid, "lat": la, "lng": ln, "osm_class": r["osm_class"], "truth": int(r["truth"]),
             "osm_marked_signal": int(r["osm_marked_signal"]), "pano_id": pid, "pano_dist_m": round(d, 2),
             "capture_date": pd[pid][3]})
    sample = []
    cells = {}
    for cell in sorted(eligible):
        pool = sorted(eligible[cell], key=lambda x: x["node_id"])
        pick = rng.sample(pool, min(a.per_cell, len(pool)))
        w = len(pool) / len(pick)
        cells[f"{cell[0]}|{cell[1]}"] = {"eligible": len(pool), "sampled": len(pick), "weight": round(w, 4)}
        for x in sorted(pick, key=lambda x: x["node_id"]):
            sample.append({**x, "weight": round(w, 6)})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sample.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sample)
    panos = sorted({x["pano_id"] for x in sample})
    with open(os.path.join(a.out, "pano_files.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"./{p[:2]}/{p}.jpg\n" for p in panos))
    meta = {"seed": BOOT_SEED, "per_cell": a.per_cell, "exclude_m": BP_EXCLUDE_M, "pano_m": BP_PANO_M,
            "cells": cells, "dropped": drop, "nodes": len(sample), "panos": len(panos),
            "train_val_seattle_panos": len(trainval), "store_panos": len(store),
            "store_panos_with_position": len(store_pos),
            "inputs_sha256": {"baseline_nodes": sha256(a.baseline_nodes), "store_listing": sha256(a.store_listing),
                              "pano_data": sha256(a.pano_data)},
            "drawn_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    log(f"(b') {len(sample)} nodes on {len(panos)} panos; cells {cells}; dropped {drop}")


def cmd_bprime_prep(a):
    """(b') store JPEGs -> 4096x2048 JPEGs written exactly as export_stage2.write_image wrote the training root,
    so (b') panos reach the model through the same decode/resize path as the test split."""
    import export_stage2
    ids = sorted(l.strip().split("/")[-1][:-4] for l in open(a.ids_file, encoding="utf-8") if l.strip())
    os.makedirs(a.out_dir, exist_ok=True)
    missing = []
    for k, pid in enumerate(ids, 1):
        dst = os.path.join(a.out_dir, f"{pid}.jpg")
        if os.path.exists(dst):
            continue
        src = next((p for s in a.store for p in (os.path.join(s, pid[:2], f"{pid}.jpg"), os.path.join(s, f"{pid}.jpg"))
                    if os.path.exists(p)), None)     # the PS store's <id[:2]>/ layout, or a flat dir (export_benchmark)
        if src is None:
            missing.append(pid)
            continue
        export_stage2.write_image(src, dst + ".tmp.jpg")
        os.replace(dst + ".tmp.jpg", dst)
        if k % 100 == 0:
            log(f"  {k}/{len(ids)}")
    log(f"(b') prep: {len(ids) - len(missing)} of {len(ids)} panos in {a.out_dir}; missing {len(missing)}")
    if missing:
        raise SystemExit(f"missing from every store dir: {missing[:5]}")


def weighted_metrics(rows, key):
    """Population-weighted P/R/F1/MCC of a 0/1 column against `truth`, with each row's `weight`."""
    tp = sum(r["weight"] for r in rows if r[key] and r["truth"])
    fp = sum(r["weight"] for r in rows if r[key] and not r["truth"])
    fn = sum(r["weight"] for r in rows if not r[key] and r["truth"])
    tn = sum(r["weight"] for r in rows if not r[key] and not r["truth"])
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": p, "recall": rc, "f1": f1(p, rc) if not (math.isnan(p) or math.isnan(rc)) else float("nan"),
            "mcc": (tp * tn - fp * fn) / den if den else float("nan"), "tp_w": tp, "fp_w": fp, "fn_w": fn, "tn_w": tn}


def strat_boot(rows, keys, n=BOOT_N, seed=BOOT_SEED):
    """Within-cell bootstrap of weighted F1 for each key; returns {key: sorted list of F1}."""
    rng = random.Random(seed)
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    out = {k: [] for k in keys}
    for _ in range(n):
        s = [rng.choice(c) for c in cells.values() for _ in c]
        for k in keys:
            out[k].append(weighted_metrics(s, k)["f1"])
    return out


def cmd_bprime_score(a):
    """(b'): each sampled node scored from its own pano's detections at the model's val-F1 threshold (from the
    pre-registered scores.json), B1 inverted at CAMERA_H, kept within MAX_RANGE_M, hit within NODE_MATCH_M."""
    res = json.load(open(os.path.join(a.scores, "scores.json"), encoding="utf-8"))
    out_dir = a.out or a.bprime
    os.makedirs(out_dir, exist_ok=True)
    pd = read_pano_data(a.pano_data)
    sample = list(csv.DictReader(open(os.path.join(a.bprime, "sample.csv"), encoding="utf-8")))
    out = {"models": {}, "n_nodes": len(sample), "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    node_rows = []
    for lab in a.labels.split(","):
        t = res["models"][lab]["threshold"]
        preds = {r["pano_id"]: r["preds"] for r in read_jsonl(os.path.join(a.bprime, "preds", lab, "bprime.jsonl"))}
        rows = []
        for s in sample:
            pid = s["pano_id"]
            if pid not in preds:
                raise SystemExit(f"{lab}: no (b') prediction for pano {pid}")
            lat, lng, hd, _ = pd[pid]
            nla, nln = float(s["lat"]), float(s["lng"])
            hit = 0
            for x, y, c in preds[pid]:
                if c >= t:
                    g = ground_point(lat, lng, hd, x, y)
                    if g and haversine_m(g[0], g[1], nla, nln) <= NODE_MATCH_M:
                        hit = 1
                        break
            r = {"node_id": s["node_id"], "cell": f"{s['osm_class']}|{s['truth']}", "osm_class": s["osm_class"],
                 "truth": int(s["truth"]), "weight": float(s["weight"]), "osm": int(s["osm_marked_signal"]),
                 "detector": hit, "year": year_bin(s["capture_date"]), "pano_id": pid}
            r["either"] = int(r["detector"] or r["osm"])
            r["always_yes"] = 1
            rows.append(r)
            node_rows.append({"model": lab, **{k: v for k, v in r.items()}})
        keys = ("detector", "osm", "either", "always_yes")
        m = {k: weighted_metrics(rows, k) for k in keys}
        boot = strat_boot(rows, ("detector", "osm", "either"))
        diff = sorted(d - o for d, o in zip(boot["detector"], boot["osm"]))
        diff_or = sorted(e - o for e, o in zip(boot["either"], boot["osm"]))
        per_cell = {}
        for r in rows:
            c = per_cell.setdefault(r["cell"], {"n": 0, "detector_yes": 0, "osm_yes": 0})
            c["n"] += 1
            c["detector_yes"] += r["detector"]
            c["osm_yes"] += r["osm"]
        by_year = {}
        for r in rows:
            if r["truth"]:
                y = by_year.setdefault(r["year"], {"n_true": 0, "detector_hit": 0})
                y["n_true"] += 1
                y["detector_hit"] += r["detector"]
        out["models"][lab] = {"threshold": t, "weighted": m,
                              "f1_diff_detector_minus_osm_95": [diff[int(.025 * len(diff))], diff[int(.975 * len(diff)) - 1]],
                              "f1_diff_or_minus_osm_95": [diff_or[int(.025 * len(diff_or))], diff_or[int(.975 * len(diff_or)) - 1]],
                              # primary (adopted before inference): OR beats OSM with the interval above 0
                              "passes": diff_or[int(.025 * len(diff_or))] > 0 and m["either"]["f1"] > m["osm"]["f1"],
                              "secondary_detector_ge_osm": m["detector"]["f1"] >= m["osm"]["f1"],
                              "per_cell": per_cell, "true_nodes_by_pano_year": by_year}
        log(f"(b') {lab}: weighted F1 detector {fmt(m['detector']['f1'])} vs OSM {fmt(m['osm']['f1'])}; "
            f"OR {fmt(m['either']['f1'])}")
    with open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "nodes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(node_rows)
    L = ["# (b') Seattle node sample: detector vs the map-only baseline (RampNet#274)\n",
         f"Scored {out['scored_utc']} by `score_274.py bprime-score`; protocol in docs/stage1_crosswalk_sdot_274.md "
         f"(b'). {len(sample)} nodes, weighted to the eligible population.\n",
         "| model | threshold | detector P / R / F1 / MCC | OSM P / R / F1 / MCC | OR F1 | always-yes F1 | "
         "F1 det−OSM 95% | F1 OR−OSM 95% | primary: OR beats OSM | secondary: det ≥ OSM |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, mm in out["models"].items():
        d, o, e, y = (mm["weighted"][k] for k in ("detector", "osm", "either", "always_yes"))
        L.append(f"| {lab} | {mm['threshold']:.2f} | {fmt(d['precision'])} / {fmt(d['recall'])} / {fmt(d['f1'])} / "
                 f"{fmt(d['mcc'])} | {fmt(o['precision'])} / {fmt(o['recall'])} / {fmt(o['f1'])} / {fmt(o['mcc'])} | "
                 f"{fmt(e['f1'])} | {fmt(y['f1'])} | [{fmt(mm['f1_diff_detector_minus_osm_95'][0])}, "
                 f"{fmt(mm['f1_diff_detector_minus_osm_95'][1])}] | [{fmt(mm['f1_diff_or_minus_osm_95'][0])}, "
                 f"{fmt(mm['f1_diff_or_minus_osm_95'][1])}] | {'**yes**' if mm['passes'] else 'no'} | "
                 f"{'yes' if mm['secondary_detector_ge_osm'] else 'no'} |")
    L += ["\n## Per cell (unweighted: the share of sampled nodes each predictor calls marked)\n",
          "| model | OSM class | truth | nodes | detector yes | OSM yes |", "|---|---|---|---|---|---|"]
    for lab, mm in out["models"].items():
        for cell, c in sorted(mm["per_cell"].items()):
            cls, tr = cell.split("|")
            L.append(f"| {lab} | {cls} | {'marked' if tr == '1' else 'not'} | {c['n']} | "
                     f"{c['detector_yes'] / c['n']:.3f} | {c['osm_yes'] / c['n']:.3f} |")
    L += ["\n## Detector recall on marked nodes, by the scored pano's capture year (unweighted)\n",
          "| model | year | marked nodes | hit |", "|---|---|---|---|"]
    for lab, mm in out["models"].items():
        for yb, c in sorted(mm["true_nodes_by_pano_year"].items()):
            L.append(f"| {lab} | {yb} | {c['n_true']} | {c['detector_hit'] / c['n_true']:.3f} |")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log(f"(b') report -> {os.path.join(out_dir, 'report.md')}")


def cmd_oracle(a):
    """Every positive of every pano as a confidence-1.0 detection: what the scoring geometry allows."""
    out_dir = os.path.join(a.preds, a.label)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"label": a.label, "checkpoint": "training labels (oracle)", "sha256": "n/a"}, f, indent=1)
    for split in SPLITS:
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for pid, ann in load_split(a.root, split).items():
                f.write(json.dumps({"pano_id": pid, "preds": [[x, y, 1.0] for x, y in
                                                              ann.get("curb_ramp_points_normalized", [])]}) + "\n")
    log(f"oracle -> {out_dir}")


if __name__ == "__main__":
    main()
