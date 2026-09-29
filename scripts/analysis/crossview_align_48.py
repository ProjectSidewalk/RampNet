"""Cross-view alignment of a GT curb-ramp point: a harness for scoring "arms" (#48).

``multiview_evidence_48.py`` carries a world GT point into other captures by raycasting
it from its source view onto flat ground at 2.6 m and projecting it back with the
labeler's ``geo.ground_point_to_pano``. GT placement error (p50 1.9 m / p90 4.4 m), camera
height and pose error all move that projected point, so at 12-18 m it can land beside the
ramp. This harness scores techniques ("arms") that try to place it better, all on one
frozen known-answer pair list, with the same metrics.

Known-answer set (``pairs.csv``, frozen by PAIRS_SHA256): the source point is a
verdict-true operational detection (a GT point that is itself a detection peak); the other
view is a non-source capture within 18 m whose own >= 0.55 detection claims the ramp by the
world test (raycast within 5 m, one-to-one in confidence order, as in
``multiview_evidence_48.capture_table``). That detection's pixel is the reference. Pairs
that could be ambiguous (another GT ramp within 6 m, or another >= 0.55 detection in the
other view landing within 8 m) are dropped.

**An arm** is one function ``fn(pair, ctx) -> {"x": .., "y": .., ...} | None`` registered
with ``@register(name, needs=..., description=..., config=...)`` in any module under
``scripts/analysis/crossview_arms/`` (see ``crossview_arms/_registry.py``). It returns the
predicted equirect (x_norm, y_norm) of the ramp in the OTHER pano, or None to fall back to
the projection. Extra keys are kept as diagnostics. ``ctx`` (``Context``) gives lazy access
to the rectilinear views, the labeler's code and the raw pano records. The ``projection``
arm is built in (the pair row's proj_x / proj_y).

Subcommands, in order:

    # 1. pair list (desktop CPU; needs the labeler checkout and runs, like
    #    multiview_evidence_48 run). FROZEN: refuses to overwrite pairs.csv without --force.
    python scripts/analysis/crossview_align_48.py pairs --labeler-root LABELER \\
        --runs-root LABELER/runs --results-root RUNS_ARCHIVE
    # 2. rectilinear views (makelab2 CPU, where the native-res panos are)
    python scripts/analysis/crossview_align_48.py cut-views \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out VIEWS
    # 3. one arm -> predictions/<arm>.jsonl (+ .meta.json with wall-clock and host)
    python scripts/analysis/crossview_align_48.py predict --arm lg --views VIEWS
    python scripts/analysis/crossview_align_48.py predict --arm proj_height_auto \\
        --labeler-root LABELER --runs-root LABELER/runs --results-root RUNS_ARCHIVE
    # 4. score every arm with predictions (CPU, committed inputs only) -> results.json
    python scripts/analysis/crossview_align_48.py score
    # reference-noise estimate from manual_gold (CPU, committed inputs only)
    python scripts/analysis/crossview_align_48.py noise
    # list registered arms
    python scripts/analysis/crossview_align_48.py arms

Outputs go to ``analysis_out/crossview_align_48/``. ``score`` and ``noise`` read only
committed files, so their JSON re-derives from a clean clone.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

OUT_ROOT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
OUT = os.path.join(OUT_ROOT, "crossview_align_48")
ELIGIBLE_CSV = os.path.join(OUT, "eligible_pairs.csv")
PAIRS_CSV = os.path.join(OUT, "pairs.csv")
#: sha256 of the frozen pairs.csv. Every arm is scored on exactly these pairs; `pairs`
#: refuses to overwrite the file, and predict / score refuse any other bytes.
PAIRS_SHA256 = "a85a11bceb57e7d4db4bc5914c35cb17b5fdaf9bc8c9189957574a13260db388"
REF_WIDTH_PX = 4096       # pixel errors are reported on a 4096 x 2048 equirect
RESULTS_JSON = os.path.join(OUT, "results.json")
NOISE_JSON = os.path.join(OUT, "reference_noise.json")

CITIES = ("richmond", "paterson", "gainesville", "bend", "sao_paulo")
IMAGERY = {"richmond": "mapillary", "paterson": "gsv", "gainesville": "gsv",
           "bend": "gsv", "sao_paulo": "gsv"}

# Known-answer set (fixed before any matching was run)
OPERATIONAL = 0.55
R_OTHER_M = 18.0          # other camera within this of the source point (B.1's R)
WORLD_HIT_M = 5.0         # the world test: reference raycast within this of the GT point
AMBIG_RAMP_M = 6.0        # drop a ramp with another pool ramp this close
AMBIG_DET_M = 8.0         # drop a pair with a second >= 0.55 detection landing this close
PAIRS_PER_CITY = 60
MAX_PAIRS_PER_RAMP = 2
SEED = 48

# Views
VIEW_W, VIEW_H = 1024, 768
HFOV_DEG = 75.0
# Keypoints must sit this far below the pano-frame horizon (in both views) ...
# v1 (pre-specified): 0.5 deg. It admitted far-field points just under the horizon, whose
# homography is close to a pure rotation and does not transfer to a ramp 5-18 m away (see
# docs/crossview_align_48.md). v2 (post hoc, after looking at failures): 5 deg, i.e. flat
# ground within ~30 m of a 2.6 m camera. Both are reported.
HORIZON_MARGIN_DEG = 5.0
RIG_LIMIT_DEG = -70.0      # ... and above this (the capture vehicle / rig)

RANGE_BINS = ((0.0, 6.0), (6.0, 12.0), (12.0, 18.0))
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# Pure geometry (tested)
# --------------------------------------------------------------------------- #


def focal_px(w=VIEW_W, hfov_deg=HFOV_DEG):
    return (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def pano_dir(x_norm, y_norm):
    """Unit vector(s) in the pano frame for equirect (x, y): x=0.5 is the centre column,
    y=0.5 the horizon, azimuth increasing to the right. Axes: (right, down, forward)."""
    az = (np.asarray(x_norm, dtype=float) - 0.5) * 2.0 * np.pi
    el = (0.5 - np.asarray(y_norm, dtype=float)) * np.pi
    return np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)], axis=-1)


def dir_to_pano(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    az = np.arctan2(v[..., 0], v[..., 2])
    el = np.arcsin(np.clip(-v[..., 1], -1.0, 1.0))
    return np.mod(0.5 + az / (2.0 * np.pi), 1.0), 0.5 - el / np.pi


def view_rotation(cx_norm, cy_norm):
    """Camera-to-pano rotation for a view looking at (cx, cy): yaw about the vertical,
    then pitch. Camera axes (right, down, forward), like OpenCV."""
    az = (cx_norm - 0.5) * 2.0 * math.pi
    el = (0.5 - cy_norm) * math.pi
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return ry @ rx


def view_to_pano(u, v, cx_norm, cy_norm, w=VIEW_W, h=VIEW_H, hfov_deg=HFOV_DEG):
    """View pixel(s) -> equirect (x_norm, y_norm)."""
    f = focal_px(w, hfov_deg)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cam = np.stack([(u - w / 2.0) / f, (v - h / 2.0) / f, np.ones_like(u)], axis=-1)
    return dir_to_pano(cam @ view_rotation(cx_norm, cy_norm).T)


def pano_to_view(x_norm, y_norm, cx_norm, cy_norm, w=VIEW_W, h=VIEW_H, hfov_deg=HFOV_DEG):
    """Equirect (x, y) -> view pixel (u, v, in_front)."""
    f = focal_px(w, hfov_deg)
    cam = pano_dir(x_norm, y_norm) @ view_rotation(cx_norm, cy_norm)
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = w / 2.0 + f * cam[..., 0] / z
        v = h / 2.0 + f * cam[..., 1] / z
    return u, v, z > 1e-6


def elevation_deg(y_norm):
    return (0.5 - np.asarray(y_norm, dtype=float)) * 180.0


def angular_error_deg(x1, y1, x2, y2):
    """Great-circle angle between two equirect points, degrees."""
    d = np.sum(pano_dir(x1, y1) * pano_dir(x2, y2), axis=-1)
    return np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))


def within_benchmark_radius(x1, y1, x2, y2):
    """The benchmark's per-pano match test (0.022 normalized, seam-wrapped)."""
    from rampnet.detection_eval import (PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y,
                                        radius_sq_for)
    from rampnet.geometry import dist_sq
    return dist_sq(x1, y1, x2, y2, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True) < \
        radius_sq_for(PANO_RADIUS_NORMALIZED)


def map_point(H, u, v):
    """Apply a 3x3 homography to one point; None if it is degenerate there."""
    if H is None:
        return None
    p = np.asarray(H, dtype=float) @ np.array([u, v, 1.0])
    if not np.all(np.isfinite(p)) or abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def ground_mask(u, v, cx_norm, cy_norm, margin=HORIZON_MARGIN_DEG):
    """Keypoints at least ``margin`` degrees below the pano-frame horizon and above the rig."""
    _, y = view_to_pano(u, v, cx_norm, cy_norm)
    el = elevation_deg(y)
    return (el < -margin) & (el > RIG_LIMIT_DEG)


def range_bin(d, bins=RANGE_BINS):
    for lo, hi in bins:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}m"
    return f">={bins[-1][1]:g}m"


# --------------------------------------------------------------------------- #
# Committed-artifact I/O
# --------------------------------------------------------------------------- #


def rnd(v, nd=4):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(rnd(payload), indent=1, sort_keys=True) + "\n")
    return path


PAIR_COLUMNS = ["pair_id", "city", "imagery", "ramp_uid", "src_pano", "src_x", "src_y",
                "src_range_m", "src_date", "oth_pano", "oth_date", "same_date",
                "oth_range_m", "baseline_m", "proj_x", "proj_y", "ref_x", "ref_y",
                "ref_conf", "ref_world_gap_m"]


def write_rows(path, rows, columns=PAIR_COLUMNS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([(round(r[c], 6) if isinstance(r[c], float) else r[c]) for c in columns])
    return path


FLOAT_COLS = {"src_x", "src_y", "src_range_m", "oth_range_m", "baseline_m", "proj_x",
              "proj_y", "ref_x", "ref_y", "ref_conf", "ref_world_gap_m"}


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in FLOAT_COLS:
            r[c] = float(r[c])
        r["same_date"] = int(r["same_date"])
    return rows


def sample_pairs(eligible, per_city=PAIRS_PER_CITY, per_ramp=MAX_PAIRS_PER_RAMP, seed=SEED):
    """Deterministic sample: per city, ramps in seeded random order, up to ``per_ramp``
    of each ramp's other views (seeded), until ``per_city`` pairs."""
    by_city = defaultdict(lambda: defaultdict(list))
    for r in eligible:
        by_city[r["city"]][r["ramp_uid"]].append(r)
    out = []
    for city in CITIES:
        ramps = by_city.get(city, {})
        rng = random.Random(f"{seed}:{city}")
        uids = sorted(ramps, key=lambda u: int(u.split(":")[1]))
        rng.shuffle(uids)
        taken = []
        for u in uids:
            views = sorted(ramps[u], key=lambda r: r["oth_pano"])
            rng.shuffle(views)
            for r in views[:per_ramp]:
                if len(taken) < per_city:
                    taken.append(r)
            if len(taken) >= per_city:
                break
        out.extend(taken)
    for i, r in enumerate(out):
        r["pair_id"] = f"p{i:03d}"
    return out


# --------------------------------------------------------------------------- #
# 1. pairs (labeler geometry)
# --------------------------------------------------------------------------- #


def cmd_pairs(args):
    import multiview_evidence_48 as mv
    if os.path.exists(PAIRS_CSV) and not args.force:
        raise SystemExit(f"{PAIRS_CSV} is frozen (PAIRS_SHA256); pass --force to rebuild it, "
                         "and expect every committed prediction to be invalidated")
    L = mv.import_labeler(args.labeler_root)
    runs_root = args.runs_root or os.path.join(args.labeler_root, "runs")
    geo = L.geo
    rows, stats = [], {}
    for city in CITIES:
        print(f"== {city}", flush=True)
        verdicts, bundle_ops, run_panos, info = mv.load_city(L, city, runs_root,
                                                             results_root=args.results_root)
        params = mv.fuse_params(L)
        gtw = mv.world_gt(L, city, verdicts, bundle_ops, run_panos, params)
        frame = gtw.frame
        by_id = {p.pano_id: p for p in run_panos}
        pose = {p.pano_id: L.fs.pano_pose(p, params.apply_pose) for p in run_panos}
        cams = {pid: frame.to_enu(p.lat, p.lng) for pid, p in by_id.items()}
        dets, _, _ = L.fs.project(run_panos, params)
        ops = defaultdict(list)
        for d in dets:
            if d.conf >= OPERATIONAL:
                ops[d.pano_id].append(d)
        pool = gtw.pool
        st = defaultdict(int)
        for i, r in enumerate(pool):
            # source: the first GT reference that is a verdict-true operational detection
            src = None
            for pid, gi in r["gt_refs"]:
                gx, gy = gtw.judged_gt[pid].gt_points[gi]
                entry = verdicts[pid]
                op = [(x, y) for _, x, y, c in by_id[pid].detections if c >= OPERATIONAL]
                if any(v is True and abs(x - gx) < 1e-12 and abs(y - gy) < 1e-12
                       for v, (x, y) in zip(entry["dets"], op)):
                    src = (pid, gx, gy)
                    break
            if src is None:
                st["ramp_no_det_source"] += 1
                continue
            if any(j != i and math.hypot(o["e"] - r["e"], o["n"] - r["n"]) < AMBIG_RAMP_M
                   for j, o in enumerate(pool)):
                st["ramp_ambiguous_neighbour"] += 1
                continue
            spid, sx, sy = src
            errors = geo.error_model_for(by_id[spid].source)
            g = geo.detection_ground_point(pose[spid], sx, sy,
                                           camera_height=params.camera_height_m,
                                           max_range_m=params.max_range_m, errors=errors,
                                           apply_pose=params.rotates)
            if g is None:
                st["ramp_source_unplaceable"] += 1
                continue
            se, sn = frame.to_enu(g.lat, g.lng)
            st["ramps_considered"] += 1
            for pid, p in by_id.items():
                if pid in r["source_panos"]:
                    continue
                ce, cn = cams[pid]
                rng_m = math.hypot(ce - se, cn - sn)
                if rng_m > R_OTHER_M:
                    continue
                st["other_views_within_R"] += 1
                # one-to-one claims in confidence order against every pool ramp in reach
                claim = {}
                for d in sorted(ops.get(pid, ()), key=lambda d: -d.conf):
                    best = None
                    for j, o in enumerate(pool):
                        if j in claim:
                            continue
                        dd = math.hypot(d.e - o["e"], d.n - o["n"])
                        if dd < WORLD_HIT_M and (best is None or (dd, j) < best):
                            best = (dd, j)
                    if best is not None:
                        claim[best[1]] = d
                ref = claim.get(i)
                if ref is None:
                    continue
                st["claimed"] += 1
                near = [d for d in ops.get(pid, ())
                        if d is not ref and math.hypot(d.e - r["e"], d.n - r["n"]) < AMBIG_DET_M]
                if near:
                    st["claimed_but_ambiguous"] += 1
                    continue
                pr = geo.ground_point_to_pano(pose[pid], g.lat, g.lng,
                                              camera_height=params.camera_height_m,
                                              max_range_m=R_OTHER_M + 1.0)
                if pr is None:
                    st["projection_none"] += 1
                    continue
                sd, od = by_id[spid].capture_date or "", p.capture_date or ""
                rows.append({
                    "pair_id": "", "city": city, "imagery": IMAGERY[city], "ramp_uid": r["uid"],
                    "src_pano": spid, "src_x": sx, "src_y": sy, "src_range_m": g.range_m,
                    "src_date": sd, "oth_pano": pid, "oth_date": od,
                    "same_date": int(bool(sd) and sd[:7] == od[:7]),
                    "oth_range_m": rng_m,
                    "baseline_m": math.hypot(ce - cams[spid][0], cn - cams[spid][1]),
                    "proj_x": pr.x_norm, "proj_y": pr.y_norm, "ref_x": ref.x, "ref_y": ref.y,
                    "ref_conf": ref.conf, "ref_world_gap_m": math.hypot(ref.e - se, ref.n - sn)})
                st["eligible_pairs"] += 1
        stats[city] = dict(st)
        print(f"   {dict(st)}", flush=True)
    rows.sort(key=lambda r: (CITIES.index(r["city"]), int(r["ramp_uid"].split(":")[1]),
                             r["oth_pano"]))
    write_rows(ELIGIBLE_CSV, rows)
    sampled = sample_pairs([dict(r) for r in rows])
    write_rows(PAIRS_CSV, sampled)
    write_json(os.path.join(OUT, "pairs_meta.json"), {
        "labeler": L.prov, "selection": selection_constants(), "per_city": stats,
        "eligible_pairs": len(rows), "sampled_pairs": len(sampled)})
    print(f"eligible {len(rows)}, sampled {len(sampled)} -> {PAIRS_CSV}")


def selection_constants():
    return {"operational": OPERATIONAL, "r_other_m": R_OTHER_M, "world_hit_m": WORLD_HIT_M,
            "ambig_ramp_m": AMBIG_RAMP_M, "ambig_det_m": AMBIG_DET_M,
            "pairs_per_city": PAIRS_PER_CITY, "max_pairs_per_ramp": MAX_PAIRS_PER_RAMP,
            "seed": SEED, "camera_height_m": 2.6}


# --------------------------------------------------------------------------- #
# 2. cut-views (makelab2)
# --------------------------------------------------------------------------- #


def render_view(equi, cx_norm, cy_norm, w=VIEW_W, h=VIEW_H, hfov_deg=HFOV_DEG):
    """Rectilinear view of an equirect image (already resized to the view's angular
    resolution), bilinear, wrapped at the seam."""
    import cv2
    H_, W_ = equi.shape[:2]
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = view_to_pano(uu, vv, cx_norm, cy_norm, w, h, hfov_deg)
    mx = (x * W_ - 0.5).astype(np.float32)
    my = np.clip(y * H_ - 0.5, 0, H_ - 1).astype(np.float32)
    return cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _cut_pano(job):
    import cv2
    path, items, out_dir = job
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return [("missing", path)]
    tw = int(round(360.0 / HFOV_DEG * VIEW_W))
    img = cv2.resize(img, (tw, tw // 2), interpolation=cv2.INTER_AREA)
    for name, cx, cy in items:
        cv2.imwrite(os.path.join(out_dir, name), render_view(img, cx, cy),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    return [("ok", name) for name, _, _ in items]


def cmd_cut_views(args):
    from multiprocessing import Pool
    pairs = read_rows(args.pairs)
    os.makedirs(args.out, exist_ok=True)
    jobs = defaultdict(list)
    for r in pairs:
        jobs[(r["city"], r["src_pano"])].append((f"{r['pair_id']}_src.jpg", r["src_x"], r["src_y"]))
        jobs[(r["city"], r["oth_pano"])].append((f"{r['pair_id']}_oth.jpg", r["proj_x"], r["proj_y"]))
    work = [(os.path.join(args.archive_root, c, "panos", f"{p}.jpg"), items, args.out)
            for (c, p), items in sorted(jobs.items())]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = [x for chunk in pool.imap_unordered(_cut_pano, work) for x in chunk]
    miss = [p for s, p in res if s == "missing"]
    print(f"{sum(1 for s, _ in res if s == 'ok')} views from {len(work)} panos in "
          f"{time.time() - t0:.1f} s; {len(miss)} panos missing")
    for m in miss[:20]:
        print("  missing", m)


# --------------------------------------------------------------------------- #
# 3. arms: registry, context, predict
# --------------------------------------------------------------------------- #


def pairs_sha256(path=None):
    import hashlib
    with open(path or PAIRS_CSV, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_frozen_pairs(path=None):
    """The frozen pair list, refused unless its bytes hash to PAIRS_SHA256, so every arm is
    scored on identical pairs."""
    path = path or PAIRS_CSV
    got = pairs_sha256(path)
    if got != PAIRS_SHA256:
        raise SystemExit(f"{path}: sha256 {got[:12]}... is not the frozen pair list "
                         f"({PAIRS_SHA256[:12]}...). Arms are only comparable on identical pairs.")
    return read_rows(path)


def load_arms():
    """Import every module in scripts/analysis/crossview_arms/ so its @register calls run;
    returns the registry {name: Arm}."""
    import importlib
    import pkgutil
    import crossview_arms
    from crossview_arms._registry import ARMS as registry
    for m in pkgutil.iter_modules(crossview_arms.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"crossview_arms.{m.name}")
    return registry


class Context:
    """What an arm may use besides its pair. Everything is loaded lazily and cached, so an
    arm that needs only the committed pair row never touches imagery or the labeler.

    * ``view(pair, "src"|"oth")``: the rectilinear view as a BGR uint8 array (needs --views);
      ``view_centre(pair, which)``: the equirect (x, y) it is centred on. Convert with
      ``view_to_pano`` / ``pano_to_view``.
    * ``labeler()``: the labeler's geo / fuse_sites / eval_sites (needs --labeler-root).
    * ``pano(city, pano_id)``: that pano's raw results.jsonl ``pano`` block (dict), and
      ``slim(city, pano_id)``: the labeler's SlimPano for it (needs --runs-root /
      --results-root as for ``pairs``). Only the panos in the pair list are kept.
    * ``cache``: a dict arms may use to keep models between pairs.
    * ``args``: the parsed CLI args (for arm-specific inputs, e.g. ``--extra``).

    **Answer hiding is enforced here, not left to convention.** ``pairs`` is stored with the
    answer columns (``ANSWER_KEYS``: ``ref_*``) removed, and ``slim()`` hands out SlimPanos
    with ``detections`` emptied: the reference IS one of the other view's detections, and the
    ambiguity filter guarantees it is the only >= 0.55 one near the projection, so "nearest
    detection" would be close to an oracle. ``crossview_arms.geometry.at_height`` does the
    same. Scoring and post-scoring diagnostics read the answers from ``read_frozen_pairs()``
    directly, never through a Context.
    """

    def __init__(self, args, pairs):
        self.args = args
        self.pairs = [strip_answers(p) for p in pairs]
        self.cache = {}
        self._panos = {}
        self._slim = {}

    def view_centre(self, pair, which):
        return ((pair["src_x"], pair["src_y"]) if which == "src"
                else (pair["proj_x"], pair["proj_y"]))

    def view(self, pair, which, flags=None):
        import cv2
        if not getattr(self.args, "views", None):
            raise SystemExit("this arm needs --views (the directory cut-views wrote)")
        path = os.path.join(self.args.views, f"{pair['pair_id']}_{which}.jpg")
        img = cv2.imread(path, cv2.IMREAD_COLOR if flags is None else flags)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def labeler(self):
        if "L" not in self.cache:
            if not getattr(self.args, "labeler_root", None):
                raise SystemExit("this arm needs --labeler-root")
            import multiview_evidence_48 as mv
            self.cache["L"] = mv.import_labeler(self.args.labeler_root)
        return self.cache["L"]

    def _results_path(self, city):
        import multiview_evidence_48 as mv
        runs_root = self.args.runs_root or os.path.join(self.args.labeler_root, "runs")
        return os.path.join(mv.results_dir(city, runs_root, self.args.results_root),
                            "results.jsonl")

    def pano(self, city, pano_id):
        if city not in self._panos:
            want = {p for r in self.pairs if r["city"] == city
                    for p in (r["src_pano"], r["oth_pano"])}
            got = {}
            with open(self._results_path(city), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    pid = rec["pano"]["panorama_id"]
                    if pid in want:
                        got[pid] = rec["pano"]
            self._panos[city] = got
        return self._panos[city][pano_id]

    def slim(self, city, pano_id, **load_kw):
        key = (city, tuple(sorted(load_kw.items())))
        if key not in self._slim:
            L = self.labeler()
            from pathlib import Path
            panos = L.fs.load_results(Path(self._results_path(city)), **load_kw)
            want = {p for r in self.pairs if r["city"] == city
                    for p in (r["src_pano"], r["oth_pano"])}
            self._slim[key] = {p.pano_id: without_detections(p) for p in panos
                               if p.pano_id in want}
        return self._slim[key][pano_id]


def strip_answers(pair):
    """A copy of one pairs.csv row without the answer columns (``ANSWER_KEYS``)."""
    from crossview_arms._registry import ANSWER_KEYS
    return {k: v for k, v in pair.items() if k not in ANSWER_KEYS}


def without_detections(slim_pano):
    """A copy of a labeler SlimPano with its RampNet detections removed. Arms get pose and
    height from a SlimPano; its detections include the reference, so they are withheld."""
    import dataclasses
    return dataclasses.replace(slim_pano, detections=[])


def prediction_paths(name):
    d = os.path.join(OUT, "predictions")
    return os.path.join(d, f"{name}.jsonl"), os.path.join(d, f"{name}.meta.json")


def run_arm(arm, pairs, ctx):
    """Apply ``arm`` to every pair, hiding the answer columns. Returns (rows, n_missing):
    one {"pair_id", "x", "y", ...} per pair, x / y None for a fallback."""
    random.seed(SEED)
    np.random.seed(SEED)
    rows, errors = [], 0
    for p in pairs:
        visible = strip_answers(p)
        try:
            out = arm.fn(visible, ctx)
        except FileNotFoundError as e:
            out, errors = {"x": None, "y": None, "error": f"missing input: {e}"}, errors + 1
        out = dict(out or {})
        out.setdefault("x", None)
        out.setdefault("y", None)
        if out["x"] is None or out["y"] is None:
            out["x"] = out["y"] = None
        rows.append({"pair_id": p["pair_id"], **out})
    return rows, errors


def cmd_predict(args):
    """Run one registered arm over the frozen pairs; write predictions/<arm>.jsonl (one row
    per pair: x, y, or null for fallback, plus the arm's diagnostics) and <arm>.meta.json
    (wall-clock, host, device, versions, the pair list's hash)."""
    import platform
    registry = load_arms()
    if args.arm not in registry:
        raise SystemExit(f"unknown arm {args.arm!r}; registered: {', '.join(sorted(registry))}")
    arm = registry[args.arm]
    pairs = read_frozen_pairs()
    ctx = Context(args, pairs)
    t0 = time.time()
    rows, errors = run_arm(arm, pairs, ctx)
    elapsed = time.time() - t0
    pred_path, meta_path = prediction_paths(args.arm)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(rnd(r, 6), sort_keys=True) + "\n")
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    meta = {"arm": args.arm, "description": arm.description, "needs": list(arm.needs),
            "pairs_sha256": PAIRS_SHA256, "pairs": len(pairs), "elapsed_s": elapsed,
            "host": platform.node(), "gpu_visible": gpu, "missing_inputs": errors,
            "fallback": sum(1 for r in rows if r["x"] is None),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": arm.config, "versions": _versions()}
    if "L" in ctx.cache:
        meta["labeler"] = ctx.cache["L"].prov
    write_json(meta_path, meta)
    print(f"{args.arm}: {len(rows)} pairs in {elapsed:.1f} s, fallback {meta['fallback']}, "
          f"missing inputs {errors} -> {pred_path}")


def _versions():
    out = {"numpy": np.__version__, "python": sys.version.split()[0]}
    for mod in ("cv2", "torch", "kornia"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return out


# --------------------------------------------------------------------------- #
# 4. score
# --------------------------------------------------------------------------- #


def equirect_px_error(x1, y1, x2, y2, width=REF_WIDTH_PX):
    """Seam-wrapped pixel distance on a width x width/2 equirect (4096 x 2048 by default,
    the benchmark's frame). Angular error is the primary metric; this is for readers who
    think in pixels."""
    dx = (np.asarray(x1) - np.asarray(x2) + 0.5) % 1.0 - 0.5
    dy = np.asarray(y1) - np.asarray(y2)
    return np.hypot(dx * width, dy * width / 2.0)


def read_predictions(name):
    pred_path, meta_path = prediction_paths(name)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("pairs_sha256") != PAIRS_SHA256:
        raise SystemExit(f"{pred_path} was predicted on a different pair list")
    out = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("x") is not None and r.get("y") is not None:
                    x, y = float(r["x"]), float(r["y"])
                    # arm_errors wraps x but not y: a y outside [0, 1] would be scored as
                    # a point past the pole instead of being refused (review of #210)
                    if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= y <= 1.0):
                        raise SystemExit(f"{pred_path}: {r['pair_id']} has x={x}, y={y}; "
                                         "y must be finite and in [0, 1]")
                out[r["pair_id"]] = r
    return out


def arm_errors(pairs, preds):
    """[(angle_deg, px_err, within_0022, fell_back), ...] aligned with ``pairs``; a missing
    or null prediction falls back to the projection. ``preds`` None = the projection."""
    rows = []
    for p in pairs:
        r = None if preds is None else preds.get(p["pair_id"])
        if r is None or r.get("x") is None or r.get("y") is None:
            x, y, fb = p["proj_x"], p["proj_y"], preds is not None
        else:
            x, y, fb = float(r["x"]) % 1.0, float(r["y"]), False
        rows.append((float(angular_error_deg(x, y, p["ref_x"], p["ref_y"])),
                     float(equirect_px_error(x, y, p["ref_x"], p["ref_y"])),
                     bool(within_benchmark_radius(x, y, p["ref_x"], p["ref_y"])), fb))
    return rows


def cluster_bootstrap(groups, stat, n_boot=None, seed=SEED):
    """Percentile CI of ``stat`` over ramps resampled with replacement.
    ``groups`` is a list of per-ramp lists of items; ``stat`` takes a flat list."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    if k == 0:
        return None
    vals = []
    for _ in range(N_BOOT if n_boot is None else n_boot):
        pick = rng.integers(0, k, k)
        flat = [x for i in pick for x in groups[i]]
        vals.append(stat(flat))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(pairs, errs, idx, proj):
    """One arm's summary on the pairs ``idx``. ``errs`` / ``proj``: arm_errors rows for the
    arm and for the projection."""
    by_ramp = defaultdict(list)
    for i in idx:
        by_ramp[pairs[i]["ramp_uid"]].append(i)
    groups = [v for _, v in sorted(by_ramp.items())]
    e = errs

    def med(ii):
        return float(np.median([e[i][0] for i in ii])) if ii else float("nan")

    def med_px(ii):
        return float(np.median([e[i][1] for i in ii])) if ii else float("nan")

    def p90(ii):
        return float(np.percentile([e[i][0] for i in ii], 90)) if ii else float("nan")

    def within2(ii):
        return float(np.mean([e[i][0] <= 2.0 for i in ii])) if ii else float("nan")

    def gain(ii):
        return float(np.median([proj[i][0] - e[i][0] for i in ii])) if ii else float("nan")

    def pmed(ii):
        return float(np.median([proj[i][0] for i in ii])) if ii else float("nan")

    row = {"n_pairs": len(idx), "n_ramps": len(groups),
           "median_deg": med(idx), "median_ci": cluster_bootstrap(groups, med),
           "median_px": med_px(idx), "median_px_ci": cluster_bootstrap(groups, med_px),
           "p90_deg": p90(idx), "p90_ci": cluster_bootstrap(groups, p90),
           "within_2deg": within2(idx), "within_2deg_ci": cluster_bootstrap(groups, within2),
           "within_0022": float(np.mean([e[i][2] for i in idx])) if idx else None,
           "fallback_rate": float(np.mean([e[i][3] for i in idx])) if idx else None}
    if e is not proj:
        used = [i for i in idx if not e[i][3]]
        ug = [v for v in ([i for i in g if not e[i][3]] for g in groups) if v]
        row["median_gain_deg"] = gain(idx)
        row["median_gain_ci"] = cluster_bootstrap(groups, gain)
        row["aligned_only"] = {
            "n_pairs": len(used),
            "median_deg": med(used) if used else None,
            "median_ci": cluster_bootstrap(ug, med) if ug else None,
            "median_px": med_px(used) if used else None,
            "projection_median_deg": pmed(used) if used else None,
            "projection_median_ci": cluster_bootstrap(ug, pmed) if ug else None,
            "median_gain_deg": gain(used) if used else None,
            "median_gain_ci": cluster_bootstrap(ug, gain) if ug else None,
            "within_2deg": within2(used) if used else None,
            "projection_within_2deg": float(np.mean([proj[i][0] <= 2.0 for i in used]))
            if used else None,
            "closer_than_projection": float(np.mean([e[i][0] < proj[i][0] - 1e-9 for i in used]))
            if used else None}
    return row


def strata_of(pairs):
    strata = {"all": list(range(len(pairs)))}
    for key, fn in (("imagery", lambda p: p["imagery"]), ("city", lambda p: p["city"]),
                    ("range", lambda p: range_bin(p["oth_range_m"])),
                    ("date", lambda p: "same_month" if p["same_date"] else "different_month")):
        for i, p in enumerate(pairs):
            strata.setdefault(f"{key}={fn(p)}", []).append(i)
    return dict(sorted(strata.items()))


def score(pairs, preds_by_arm):
    """{"arms": {arm: {stratum: summary}}, "per_pair": [...]} for the projection plus every
    arm in ``preds_by_arm`` ({arm: {pair_id: prediction}})."""
    proj = arm_errors(pairs, None)
    errs = {"projection": proj}
    for name, preds in sorted(preds_by_arm.items()):
        errs[name] = arm_errors(pairs, preds)
    strata = strata_of(pairs)
    out = {"arms": {name: {k: summarize(pairs, e, idx, proj) for k, idx in strata.items()}
                    for name, e in errs.items()}}
    out["per_pair"] = [{"pair_id": p["pair_id"],
                        **{a: round(errs[a][i][0], 3) for a in errs},
                        **{f"{a}_fb": int(errs[a][i][3]) for a in errs if a != "projection"}}
                       for i, p in enumerate(pairs)]
    return out


def available_predictions():
    d = os.path.join(OUT, "predictions")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-len(".jsonl")] for f in os.listdir(d) if f.endswith(".jsonl"))


def cmd_score(args):
    """Score arms into results.json. With no --arms it takes EVERY predictions/*.jsonl, so an
    exploratory file left there joins the results and the multiplicity screen
    (crossview_combined_48.py) silently; the arms new since the committed results.json are
    named here, and the rederive test fails until results.json and the docs catch up."""
    pairs = read_frozen_pairs()
    names = args.arms.split(",") if args.arms else available_predictions()
    if not args.arms and os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            before = set(json.load(f)["config"]["arms"])
        new = sorted(set(names) - before)
        if new:
            print(f"NOTE: {len(new)} arm(s) not in the committed results.json join the scored "
                  f"set, which grows the multiplicity to {len(names)} arms: {', '.join(new)}")
    res = score(pairs, {n: read_predictions(n) for n in names})
    res["config"] = {"pairs_sha256": PAIRS_SHA256, "n_boot": N_BOOT, "seed": SEED,
                     "ci": "2.5-97.5 percentile, ramps resampled",
                     "error": "great-circle angle to the reference detection, degrees; px on a "
                              f"{REF_WIDTH_PX}x{REF_WIDTH_PX // 2} equirect",
                     "arms": ["projection"] + sorted(names)}
    write_json(args.out, res)
    print(f"{'arm':22s} {'median deg [CI]':>26s} {'px':>6s} {'fallback':>8s} "
          f"{'aligned n':>9s} {'aligned vs proj':>16s}")
    for name, s in res["arms"].items():
        a = s["all"]
        ci = a["median_ci"]
        line = (f"{name:22s} {a['median_deg']:8.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                f"{'':>6s} {a['median_px']:6.1f} {a['fallback_rate']:8.2f}")
        if "aligned_only" in a and a["aligned_only"]["n_pairs"]:
            o = a["aligned_only"]
            line += f" {o['n_pairs']:9d} {o['median_deg']:6.2f} vs {o['projection_median_deg']:.2f}"
        print(line)
    print(f"-> {args.out}")


# --------------------------------------------------------------------------- #
# reference noise (manual_gold)
# --------------------------------------------------------------------------- #


def cmd_noise(args):
    """How far a RampNet detection peak sits from the human box centre it matches on
    manual_gold (GSV, independently labelled). Bounds the reference's own noise."""
    from rampnet.detection_eval import (PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y,
                                        radius_sq_for)
    from rampnet.geometry import dist_sq
    r2 = radius_sq_for(PANO_RADIUS_NORMALIZED)
    errs, dx, dy = [], [], []
    with open(os.path.join(REPO, "benchmark", "manual_gold", "records.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for rec in recs:
        pid = rec["pano"]["panorama_id"]
        path = os.path.join(REPO, "manual_labels", f"{pid}.txt")
        if not os.path.exists(path):
            continue
        gt = []
        with open(path, encoding="utf-8") as g:
            for line in g:
                parts = line.split()
                if len(parts) >= 5:
                    gt.append((float(parts[1]), float(parts[2])))
        dets = sorted(((d["x_normalized"], d["y_normalized"], d["confidence"])
                       for d in rec["detections"] if d["confidence"] >= OPERATIONAL),
                      key=lambda d: -d[2])
        used = set()
        for x, y, _ in dets:
            best = None
            for j, (gx, gy) in enumerate(gt):
                if j in used:
                    continue
                d2 = dist_sq(x, y, gx, gy, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
                if d2 < r2 and (best is None or d2 < best[0]):
                    best = (d2, j)
            if best is None:
                continue
            used.add(best[1])
            gx, gy = gt[best[1]]
            errs.append(float(angular_error_deg(x, y, gx, gy)))
            ddx = (x - gx + 0.5) % 1.0 - 0.5
            dx.append(ddx * 360.0)
            dy.append((y - gy) * 180.0)
    e = np.array(errs)
    res = {"source": "benchmark/manual_gold detections >= 0.55 vs manual_labels box centres, "
                     "greedy one-to-one within the 0.022 radius",
           "n_matched": len(e), "median_deg": float(np.median(e)),
           "p90_deg": float(np.percentile(e, 90)), "mean_deg": float(e.mean()),
           "median_abs_azimuth_deg": float(np.median(np.abs(dx))),
           "median_abs_elevation_deg": float(np.median(np.abs(dy))),
           "mean_elevation_offset_deg": float(np.mean(dy)),
           "note": "a rough scale for peak noise, not a bound: a box centre is not the ramp point the peak "
                   "is trained on, and the 0.022 radius truncates the tail"}
    write_json(args.out, res)
    print(json.dumps(rnd(res), indent=1))


# --------------------------------------------------------------------------- #


def cmd_arms(args):
    for name, a in sorted(load_arms().items()):
        print(f"{name:22s} needs={','.join(a.needs) or '-':18s} {a.description}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def inputs(p):
        p.add_argument("--labeler-root", help="labeler checkout (read-only)")
        p.add_argument("--runs-root", help="default: <labeler-root>/runs")
        p.add_argument("--results-root", help="archived results.jsonl copies (per city)")

    p = sub.add_parser("pairs")
    inputs(p)
    p.add_argument("--force", action="store_true", help="rebuild the frozen pair list")
    p.set_defaults(fn=cmd_pairs)
    p = sub.add_parser("cut-views")
    p.add_argument("--pairs", default=PAIRS_CSV)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_cut_views)
    p = sub.add_parser("predict", help="run one registered arm over the frozen pairs")
    p.add_argument("--arm", required=True)
    p.add_argument("--views", help="directory cut-views wrote")
    inputs(p)
    p.add_argument("--extra", action="append", default=[],
                   help="KEY=VALUE for arm-specific inputs (read from ctx.args.extra)")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("score")
    p.add_argument("--arms", help="comma-separated; default: every predictions/*.jsonl")
    p.add_argument("--out", default=RESULTS_JSON)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("noise")
    p.add_argument("--out", default=NOISE_JSON)
    p.set_defaults(fn=cmd_noise)
    p = sub.add_parser("arms")
    p.set_defaults(fn=cmd_arms)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
