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
        for s in SPLITS:
            for t in GRID:
                m = metrics_from(counts_at(list(cls[s].values()), t))
                sweep_rows.append({"model": lab, "split": s, "threshold": t, "precision": m["precision"],
                                   "recall": m["recall"], "f1": m["f1"], "tp": m["tp"],
                                   "tp_complete": m["tp_complete"], "fp": m["fp"], "ignored": m["ignored"],
                                   "unscored": m["unscored"], "n_gt": m["n_gt"]})
        mres = {"sha256": meta["sha256"], "checkpoint": meta["checkpoint"], "threshold": t_star, "split": {}}
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


def crop_card(img, it, half_w=0.08, out_w=480):
    """A crop centred on the item (seam-wrapped), marks drawn: green GT, red detections above threshold, grey
    ignore points, the item ringed in yellow. Returns JPEG bytes."""
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


def card_figure(it, jpeg, cap):
    """One rating card: the crop, its caption, verdict buttons for the card's kind and a note box."""
    import base64
    import html
    subject = "this miss (large green ring)" if it["kind"] == "fn" else "this detection (large outlined cross)"
    btns = "".join(f'<button type="button" data-v="{v}" aria-pressed="false">{k + 1}. {html.escape(txt)}'
                   f'</button>' for k, (v, txt) in enumerate(VERDICTS[it["kind"]]))
    meta = json.dumps({"card_id": card_id(it), "kind": it["kind"], "pano_id": it["pano_id"], "x": round(it["x"], 4),
                       "y": round(it["y"], 4), "conf": round(it["conf"], 3), "city": it.get("city"),
                       "mode": it.get("mode"), "year": it.get("year"), "stratum": it.get("stratum", "")})
    b64 = base64.b64encode(jpeg).decode("ascii")
    return (f'<figure class="card {it["kind"]}" tabindex="0" data-meta="{html.escape(meta)}">'
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
.legend li{{margin:3px 0;break-inside:avoid}}.sw{{display:inline-block;width:34px;height:6px;vertical-align:middle}}</style></head><body>
<h1>{html.escape(heading)}</h1>
{intro_html}
<div class="bar">{nav_html}<span id="prog"></span> &nbsp;
<label style="display:inline"><input type="checkbox" id="hide"> hide rated</label> &nbsp;
<button type="button" id="exp">Export CSV</button></div>
{''.join(cards_html)}
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
        items.append({**c, "kind": "det", "gt": [tuple(q) for q in ann.get("curb_ramp_points_normalized", [])],
                      "ignore": [tuple(q) for q in sup.get("ignore_points", [])],
                      "dets": others.get(c["pano_id"], [(c["x"], c["y"], c["conf"])])})
    if a.blind:
        random.Random(BOOT_SEED).shuffle(items)
    else:
        items.sort(key=lambda it: -it["conf"])
    prefill = prefill_for(items, load_prefill(a.prefill))

    def image_path(pid):
        return os.path.join(a.pano_dir, f"{pid}.jpg") if a.pano_dir else os.path.join(a.root, a.split, f"{pid}.jpg")

    os.makedirs(a.out, exist_ok=True)
    title, question = SECTION_HEAD["det"]
    cards = [f'<h2 id="sec-det">{title} ({len(items)} cards)</h2><p class="q">{question}</p>']
    cache = {}
    for it in items:
        if it["pano_id"] not in cache:
            cache.clear()
            cache[it["pano_id"]] = Image.open(image_path(it["pano_id"])).convert("RGB")
        cap = (f"{it['pano_id']} ({it['x']:.3f}, {it['y']:.3f})" if a.blind else
               f"conf {it['conf']:.2f} · {it.get('city') or '?'} · {it.get('mode') or '?'} · {it.get('year') or '?'} · "
               f"{it['pano_id']} ({it['x']:.3f}, {it['y']:.3f})")
        cards.append(card_figure(it, crop_card(cache[it["pano_id"]], it), cap))
    intro = f"""<p>Each card asks about <strong>one</strong> detection: the large outlined cross. Other marks are
context.</p>
<ul class="legend">
<li>{LEGEND_GLYPHS['subject_det']} the detection this card asks about</li>
<li>{LEGEND_GLYPHS['other_det']} another detection on this pano (context; it may have its own card)</li>
<li>{LEGEND_GLYPHS['other_gt']} a training label, where the pano has one (context)</li>
<li>{LEGEND_GLYPHS['radius']} the scorer's match radius</li>
</ul>
<p><strong>Crosswalk</strong> means painted pedestrian crossing markings: zebra bars or two parallel lines across a
roadway. Bike-lane, bus-lane, parking and rail markings are <em>not</em> crosswalks. Keys: <kbd>1</kbd>-<kbd>3</kbd>
rate (jumps to the next card), <kbd>n</kbd> = next unrated. Verdicts save in this browser; press <em>Export CSV</em>
when done.</p>"""
    page = rating_page(a.label, a.heading or f"{a.label}: rate detections", intro, "", cards, prefill)
    path = os.path.join(a.out, f"rate_{a.label}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    log(f"rating page -> {path} ({len(items)} cards, {len(prefill)} prefilled)")


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
    ids = sorted(l.strip().split("/")[-1][:-4] for l in open(a.ids_file, encoding="utf-8") if l.strip())
    drawn = set(random.Random(BOOT_SEED).sample(ids, a.n_panos))
    with open(a.nodes, encoding="utf-8", newline="") as f:
        nodes = [r for r in csv.DictReader(f) if r["model"] == a.label]
    fa = {r["pano_id"] for r in nodes
          if r["truth"] == "0" and r["detector"] == "1" and r["osm_class"] in ("unmarked", "unknown")}
    weights, years = {}, {}
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
            cand.append({"pano_id": pid, "x": x, "y": y, "conf": c, "city": SEATTLE, "mode": "bprime",
                         "year": years.get(pid), "stratum": stratum})
    for name, rows in (("candidates.csv", cand), ("panos.csv", pano_rows)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    meta = {"label": a.label, "threshold": a.threshold, "seed": BOOT_SEED, "n_panos": a.n_panos,
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
    c.add_argument("--pano-dir", default=None, help="flat dir of <pano_id>.jpg (else --root/--split)")
    c.add_argument("--root", default=None, help="Stage 2 root, for labels / ignore points as context")
    c.add_argument("--split", default="train")
    c.add_argument("--preds-file", default=None, help="the model's <split>.jsonl, for its other detections")
    c.add_argument("--threshold", type=float, default=0.21, help="context detections at or above this")
    c.add_argument("--blind", action="store_true", help="captions without conf/city/mode/year; seeded shuffle")
    c.add_argument("--prefill", default=None, help="CSV of earlier verdicts: card_id, verdict, note")
    c.add_argument("--out", required=True)
    u = sub.add_parser("bprime-audit", help="(#274 hard negatives, Step 0) census of detections on drawn (b') panos")
    u.add_argument("--ids-file", required=True, help="bprime/pano_files.txt")
    u.add_argument("--sample", required=True, help="bprime/sample.csv (node weights, capture dates)")
    u.add_argument("--nodes", required=True, help="the (b') nodes.csv with the model's per-node calls")
    u.add_argument("--preds-file", required=True, help="the model's bprime.jsonl")
    u.add_argument("--label", required=True)
    u.add_argument("--threshold", type=float, required=True)
    u.add_argument("--n-panos", type=int, default=AUDIT_PANOS)
    u.add_argument("--out", required=True)
    v = sub.add_parser("audit-summary", help="rates + pano-bootstrap intervals from a rated bprime-audit draw")
    v.add_argument("--panos", required=True, help="the draw's panos.csv")
    v.add_argument("--ratings", required=True, help="the exported ratings CSV for the draw's page")
    v.add_argument("--label", default="crosswalk_s1")
    v.add_argument("--out", required=True)
    b = sub.add_parser("bprime-sample", help="(b') stratified Seattle node sample + the store panos to fetch")
    b.add_argument("--root", required=True)
    b.add_argument("--baseline-nodes", required=True)
    b.add_argument("--census", required=True)
    b.add_argument("--pano-data", required=True, help="pipe-separated export of sidewalk_seattle.pano_data")
    b.add_argument("--store-listing", required=True)
    b.add_argument("--meta-dir", default=os.path.join(REPO, "analysis_out", "stage1_crosswalk_crowd_274"))
    b.add_argument("--per-cell", type=int, default=BP_PER_CELL)
    b.add_argument("--out", required=True)
    b = sub.add_parser("bprime-prep", help="(b') store JPEGs -> 4096x2048 as the training root was written")
    b.add_argument("--ids-file", required=True)
    b.add_argument("--store", nargs="+", required=True, help="store roots laid out <id[:2]>/<id>.jpg")
    b.add_argument("--out-dir", required=True)
    b = sub.add_parser("bprime-score", help="(b') weighted node scores from the bprime predictions")
    b.add_argument("--bprime", required=True, help="dir holding sample.csv and preds/<label>/bprime.jsonl")
    b.add_argument("--scores", required=True, help="the (a)/(b) score dir: thresholds come from its scores.json")
    b.add_argument("--pano-data", required=True)
    b.add_argument("--labels", default="crosswalk_s1,released")
    a = ap.parse_args()
    {"infer": cmd_infer, "score": cmd_score, "figures": cmd_figures, "oracle": cmd_oracle,
     "gallery": cmd_gallery, "bprime-sample": cmd_bprime_sample, "bprime-prep": cmd_bprime_prep,
     "bprime-score": cmd_bprime_score, "cards": cmd_cards, "bprime-audit": cmd_bprime_audit,
     "audit-summary": cmd_audit_summary}[a.cmd](a)


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
        src = next((p for p in (os.path.join(s, pid[:2], f"{pid}.jpg") for s in a.store) if os.path.exists(p)), None)
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
    with open(os.path.join(a.bprime, "scores.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.bprime, "nodes.csv"), "w", encoding="utf-8", newline="") as f:
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
    with open(os.path.join(a.bprime, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log(f"(b') report -> {os.path.join(a.bprime, 'report.md')}")


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
