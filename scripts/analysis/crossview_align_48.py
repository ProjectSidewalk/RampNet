"""Cross-view image alignment of a GT curb-ramp point (#48 pilot).

``multiview_evidence_48.py`` carries a world GT point into other captures by raycasting
it from its source view onto flat ground at 2.6 m and projecting it back with the
labeler's ``geo.ground_point_to_pano``. GT placement error (p50 1.9 m / p90 4.4 m), camera
height and pose error all move that projected point, so at 12-18 m it can land beside the
ramp. This pilot asks whether aligning the two images places the point better.

Known-answer set: the source point is a verdict-true operational detection (a GT point
that is itself a detection peak); the other view is a non-source capture within 18 m whose
own >= 0.55 detection claims the ramp by the world test (raycast within 5 m, one-to-one in
confidence order, as in ``multiview_evidence_48.capture_table``). That detection's pixel is
the reference. Pairs that could be ambiguous (another GT ramp within 6 m, or another
>= 0.55 detection in the other view landing within 8 m) are dropped.

Arms, all scored against the reference by angular error:

* ``projection`` -- today's flat-ground projection;
* ``lg`` -- ALIKED + LightGlue (kornia) on rectilinear views, matches restricted to below
  the horizon, RANSAC homography, source point mapped through it; falls back to the
  projection under ``--min-inliers`` inliers (the primary arm, chosen before scoring);
* ``lg_local`` -- the same matches, homography fitted only to those near the source point;
* ``sift`` -- OpenCV SIFT + ratio test + the same RANSAC (cheap baseline);
* ``ncc`` -- scale-corrected normalized cross-correlation template search (cheaper still).

Subcommands, in order:

    # 1. pair list (desktop CPU; needs the labeler checkout and runs, like multiview_evidence_48 run)
    python scripts/analysis/crossview_align_48.py pairs --labeler-root LABELER \\
        --runs-root LABELER/runs --results-root RUNS_ARCHIVE
    # 2. rectilinear views (makelab2 CPU, where the native-res panos are)
    python scripts/analysis/crossview_align_48.py cut-views \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out VIEWS
    # 3. matching (GPU for LightGlue; CPU works, slowly)
    python scripts/analysis/crossview_align_48.py match --views VIEWS
    # 4. scoring (CPU, committed inputs only)
    python scripts/analysis/crossview_align_48.py score
    # reference-noise estimate from manual_gold (CPU, committed inputs only)
    python scripts/analysis/crossview_align_48.py noise

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
MATCHES_JSONL = os.path.join(OUT, "matches.jsonl")
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
HORIZON_MARGIN_DEG = 0.5   # keypoints must sit this far below the pano-frame horizon ...
RIG_LIMIT_DEG = -70.0      # ... and above this (the capture vehicle / rig)

# Matching and fallback (fixed before scoring; the sweep is reported as sensitivity)
RANSAC_PX = 4.0
MIN_INLIERS = 15
INLIER_SWEEP = (8, 15, 30, 60)
LOCAL_RADIUS_PX = 160.0
LOCAL_MIN = 12
NCC_TEMPLATE_PX = 64
NCC_MIN = 0.5
MAX_KEYPOINTS = 2048

RANGE_BINS = ((0.0, 6.0), (6.0, 12.0), (12.0, 18.0))
ARMS = ("projection", "lg", "lg_local", "sift", "ncc")
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


def ground_mask(u, v, cx_norm, cy_norm):
    """Keypoints below the pano-frame horizon and above the rig."""
    _, y = view_to_pano(u, v, cx_norm, cy_norm)
    el = elevation_deg(y)
    return (el < -HORIZON_MARGIN_DEG) & (el > RIG_LIMIT_DEG)


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
# 3. match
# --------------------------------------------------------------------------- #


def fit_homography(p_src, p_oth, seed=SEED):
    import cv2
    if len(p_src) < 4:
        return None, 0, None
    cv2.setRNGSeed(seed)
    H, m = cv2.findHomography(np.float32(p_src), np.float32(p_oth), cv2.RANSAC, RANSAC_PX,
                              maxIters=5000, confidence=0.999)
    if H is None or m is None:
        return None, 0, None
    return H, int(m.sum()), m.ravel().astype(bool)


def align(p_src, p_oth, src_centre, oth_view, seed=SEED):
    """Global and local homographies for ground-filtered matches; each maps the source
    view centre (the GT point) into the other view and on to the pano."""
    out = {"n_matches": int(len(p_src))}
    H, n_in, _ = fit_homography(p_src, p_oth, seed)
    m = map_point(H, *src_centre)
    out["global"] = {"inliers": n_in, "uv": m}
    if len(p_src):
        d = np.hypot(p_src[:, 0] - src_centre[0], p_src[:, 1] - src_centre[1])
        sel = d < LOCAL_RADIUS_PX
    else:
        sel = np.zeros(0, bool)
    if sel.sum() >= LOCAL_MIN:
        Hl, nl, _ = fit_homography(p_src[sel], p_oth[sel], seed)
        out["local"] = {"inliers": nl, "n_near": int(sel.sum()), "uv": map_point(Hl, *src_centre)}
    else:
        out["local"] = {"inliers": 0, "n_near": int(sel.sum()), "uv": None}
    for k in ("global", "local"):
        uv = out[k]["uv"]
        if uv is not None:
            x, y = view_to_pano(uv[0], uv[1], *oth_view)
            out[k]["xy"] = (float(x), float(y))
        else:
            out[k]["xy"] = None
    return out


class LightGlueArm:
    def __init__(self, device):
        import kornia.feature as KF
        import torch
        self.torch, self.KF, self.device = torch, KF, device
        self.extractor = KF.ALIKED.from_pretrained("aliked-n16", device=device).eval()
        self.matcher = KF.LightGlueMatcher("aliked").to(device).eval()

    def features(self, gray):
        t = self.torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0
        t = t.repeat(1, 3, 1, 1)
        with self.torch.inference_mode():
            f = self.extractor(t)[0]
        return f.keypoints, f.descriptors

    def match(self, g1, g2):
        KF, torch = self.KF, self.torch
        k1, d1 = self.features(g1)
        k2, d2 = self.features(g2)
        k1, d1, k2, d2 = k1[:MAX_KEYPOINTS], d1[:MAX_KEYPOINTS], k2[:MAX_KEYPOINTS], d2[:MAX_KEYPOINTS]
        lafs1 = KF.laf_from_center_scale_ori(k1[None], torch.ones(1, len(k1), 1, 1, device=self.device))
        lafs2 = KF.laf_from_center_scale_ori(k2[None], torch.ones(1, len(k2), 1, 1, device=self.device))
        with torch.inference_mode():
            _, idx = self.matcher(d1, d2, lafs1, lafs2, hw1=g1.shape[:2], hw2=g2.shape[:2])
        idx = idx.cpu().numpy()
        return k1.cpu().numpy()[idx[:, 0]], k2.cpu().numpy()[idx[:, 1]]


def sift_match(g1, g2):
    import cv2
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    a = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    b = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return a, b


def ncc_search(g1, g2, scale):
    """Template of the source view's centre, rescaled by the range ratio, searched over
    the whole other view. Returns ((u, v), peak score)."""
    import cv2
    t = NCC_TEMPLATE_PX
    c1 = (g1.shape[1] // 2, g1.shape[0] // 2)
    tpl = g1[c1[1] - t // 2:c1[1] + t // 2, c1[0] - t // 2:c1[0] + t // 2]
    s = float(np.clip(scale, 0.25, 4.0))
    size = max(8, int(round(t * s)))
    tpl = cv2.resize(tpl, (size, size), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if tpl.shape[0] >= g2.shape[0] or tpl.shape[1] >= g2.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(g2, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return (loc[0] + size / 2.0, loc[1] + size / 2.0), float(mx)


def filter_ground(a, b, src_view, oth_view):
    if len(a) == 0:
        return a, b
    keep = ground_mask(a[:, 0], a[:, 1], *src_view) & ground_mask(b[:, 0], b[:, 1], *oth_view)
    return a[keep], b[keep]


def cmd_match(args):
    import cv2
    import torch
    torch.manual_seed(SEED)
    pairs = read_rows(args.pairs)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    lg = LightGlueArm(device)
    centre = (VIEW_W / 2.0, VIEW_H / 2.0)
    out, t0, t_lg = [], time.time(), 0.0
    for r in pairs:
        f1 = os.path.join(args.views, f"{r['pair_id']}_src.jpg")
        f2 = os.path.join(args.views, f"{r['pair_id']}_oth.jpg")
        g1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            out.append({"pair_id": r["pair_id"], "status": "views_missing"})
            continue
        sv = (r["src_x"], r["src_y"])
        ov = (r["proj_x"], r["proj_y"])
        rec = {"pair_id": r["pair_id"], "status": "ok"}
        t = time.time()
        a, b = lg.match(g1, g2)
        t_lg += time.time() - t
        rec["lg_raw_matches"] = int(len(a))
        a, b = filter_ground(a, b, sv, ov)
        rec["lg"] = align(a, b, centre, ov)
        a, b = sift_match(g1, g2)
        rec["sift_raw_matches"] = int(len(a))
        a, b = filter_ground(a, b, sv, ov)
        rec["sift"] = align(a, b, centre, ov)
        uv, score = ncc_search(g1, g2, r["src_range_m"] / max(r["oth_range_m"], 0.5))
        rec["ncc"] = {"score": score, "uv": uv,
                      "xy": None if uv is None else tuple(float(q) for q in view_to_pano(uv[0], uv[1], *ov))}
        out.append(rec)
    elapsed = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    with open(MATCHES_JSONL, "w", encoding="utf-8", newline="") as f:
        for rec in out:
            f.write(json.dumps(rnd(rec, 6), sort_keys=True) + "\n")
    meta = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "elapsed_s": elapsed, "lightglue_s": t_lg, "pairs": len(pairs),
            "versions": {"torch": torch.__version__, "cv2": cv2.__version__,
                         "kornia": __import__("kornia").__version__, "numpy": np.__version__},
            "matching": {"extractor": "ALIKED aliked-n16", "matcher": "LightGlue (kornia)",
                         "max_keypoints": MAX_KEYPOINTS, "ransac_px": RANSAC_PX,
                         "local_radius_px": LOCAL_RADIUS_PX, "local_min": LOCAL_MIN,
                         "ncc_template_px": NCC_TEMPLATE_PX, "sift_ratio": 0.8,
                         "view": [VIEW_W, VIEW_H, HFOV_DEG],
                         "ground": [HORIZON_MARGIN_DEG, RIG_LIMIT_DEG]}}
    write_json(os.path.join(OUT, "match_meta.json"), meta)
    print(f"matched {len(pairs)} pairs in {elapsed:.1f} s (LightGlue {t_lg:.1f} s on {device})")


# --------------------------------------------------------------------------- #
# 4. score
# --------------------------------------------------------------------------- #


def estimate(arm, pair, rec, min_inliers=MIN_INLIERS):
    """(x, y, fell_back) for one arm on one pair."""
    proj = (pair["proj_x"], pair["proj_y"], False)
    if arm == "projection":
        return proj
    if rec is None or rec.get("status") != "ok":
        return pair["proj_x"], pair["proj_y"], True
    if arm in ("lg", "lg_local", "sift"):
        key = "lg" if arm.startswith("lg") else "sift"
        part = rec[key]["local" if arm == "lg_local" else "global"]
        ok = part["xy"] is not None and part["inliers"] >= min_inliers and \
            part["uv"] is not None and 0 <= part["uv"][0] < VIEW_W and 0 <= part["uv"][1] < VIEW_H
    elif arm == "ncc":
        part = rec["ncc"]
        ok = part["xy"] is not None and part["score"] >= NCC_MIN
    else:
        raise ValueError(arm)
    if not ok:
        return pair["proj_x"], pair["proj_y"], True
    return part["xy"][0], part["xy"][1], False


def pair_errors(pairs, recs, min_inliers=MIN_INLIERS):
    """{arm: [(err_deg, within_radius, fell_back), ...]} aligned with ``pairs``."""
    out = {}
    for arm in ARMS:
        rows = []
        for p in pairs:
            x, y, fb = estimate(arm, p, recs.get(p["pair_id"]), min_inliers)
            e = float(angular_error_deg(x, y, p["ref_x"], p["ref_y"]))
            rows.append((e, bool(within_benchmark_radius(x, y, p["ref_x"], p["ref_y"])), fb))
        out[arm] = rows
    return out


def cluster_bootstrap(groups, stat, n_boot=N_BOOT, seed=SEED):
    """Percentile CI of ``stat`` over ramps resampled with replacement.
    ``groups`` is a list of per-ramp lists of items; ``stat`` takes a flat list."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    if k == 0:
        return None
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        flat = [x for i in pick for x in groups[i]]
        vals.append(stat(flat))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(pairs, errs, idx):
    """Per-arm summary on the pairs ``idx`` (indices into ``pairs``)."""
    by_ramp = defaultdict(list)
    for i in idx:
        by_ramp[pairs[i]["ramp_uid"]].append(i)
    groups = [v for _, v in sorted(by_ramp.items())]
    res = {"n_pairs": len(idx), "n_ramps": len(groups)}
    proj = errs["projection"]
    for arm in ARMS:
        e = errs[arm]

        def med(ii, e=e):
            return float(np.median([e[i][0] for i in ii])) if ii else float("nan")

        def p90(ii, e=e):
            return float(np.percentile([e[i][0] for i in ii], 90)) if ii else float("nan")

        def within(ii, e=e):
            return float(np.mean([e[i][1] for i in ii])) if ii else float("nan")

        def within2(ii, e=e):
            return float(np.mean([e[i][0] <= 2.0 for i in ii])) if ii else float("nan")

        def gain(ii, e=e):
            return float(np.median([proj[i][0] - e[i][0] for i in ii])) if ii else float("nan")

        row = {"median_deg": med(idx), "median_ci": cluster_bootstrap(groups, med),
               "p90_deg": p90(idx), "p90_ci": cluster_bootstrap(groups, p90),
               "within_0022": within(idx), "within_0022_ci": cluster_bootstrap(groups, within),
               "within_2deg": within2(idx), "within_2deg_ci": cluster_bootstrap(groups, within2),
               "fallback_rate": float(np.mean([e[i][2] for i in idx])) if idx else None}
        if arm != "projection":
            used = [i for i in idx if not e[i][2]]
            ug = [v for v in ([i for i in g if not e[i][2]] for g in groups) if v]
            row["median_gain_vs_projection_deg"] = gain(idx)
            row["median_gain_ci"] = cluster_bootstrap(groups, gain)
            row["better_than_projection"] = float(np.mean(
                [e[i][0] < proj[i][0] - 1e-9 for i in used])) if used else None
            row["aligned_only"] = {
                "n_pairs": len(used),
                "median_deg": med(used), "median_ci": cluster_bootstrap(ug, med) if ug else None,
                "projection_median_deg": float(np.median([proj[i][0] for i in used])) if used else None,
                "projection_median_ci": cluster_bootstrap(
                    ug, lambda ii: float(np.median([proj[i][0] for i in ii]))) if ug else None,
                "within_2deg": within2(used), "projection_within_2deg": float(np.mean(
                    [proj[i][0] <= 2.0 for i in used])) if used else None}
        res[arm] = row
    return res


def score(pairs, recs):
    errs = pair_errors(pairs, recs)
    allidx = list(range(len(pairs)))
    strata = {"all": allidx}
    for key, fn in (("imagery", lambda p: p["imagery"]), ("city", lambda p: p["city"]),
                    ("range", lambda p: range_bin(p["oth_range_m"])),
                    ("date", lambda p: "same_month" if p["same_date"] else "different_month")):
        for i, p in enumerate(pairs):
            strata.setdefault(f"{key}={fn(p)}", []).append(i)
    out = {"strata": {k: summarize(pairs, errs, v) for k, v in sorted(strata.items())}}
    sweep = {}
    for m in INLIER_SWEEP:
        e2 = pair_errors(pairs, recs, min_inliers=m)
        sweep[str(m)] = {arm: {"median_deg": float(np.median([x[0] for x in e2[arm]])),
                               "fallback_rate": float(np.mean([x[2] for x in e2[arm]])),
                               "within_2deg": float(np.mean([x[0] <= 2.0 for x in e2[arm]]))}
                         for arm in ("lg", "lg_local", "sift")}
    out["inlier_sweep"] = sweep
    out["per_pair"] = [{"pair_id": p["pair_id"], **{a: round(errs[a][i][0], 3) for a in ARMS},
                        **{f"{a}_fb": int(errs[a][i][2]) for a in ARMS if a != "projection"}}
                       for i, p in enumerate(pairs)]
    return out


def read_matches(path=MATCHES_JSONL):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pair_id"]] = r
    return recs


def cmd_score(args):
    pairs = read_rows(args.pairs)
    recs = read_matches(args.matches)
    res = score(pairs, recs)
    res["config"] = {"min_inliers": MIN_INLIERS, "ncc_min": NCC_MIN, "n_boot": N_BOOT,
                     "seed": SEED, "ci": "2.5-97.5 percentile, ramps resampled",
                     "error": "great-circle angle to the reference detection, degrees"}
    write_json(args.out, res)
    a = res["strata"]["all"]
    for arm in ARMS:
        print(f"{arm:10s} median {a[arm]['median_deg']:.2f} {a[arm]['median_ci']}  "
              f"fallback {a[arm]['fallback_rate']}")


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
           "note": "an upper bound on peak noise: a box centre is not the ramp point the peak "
                   "is trained on, and the 0.022 radius truncates the tail"}
    write_json(args.out, res)
    print(json.dumps(rnd(res), indent=1))


# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--labeler-root", required=True)
    p.add_argument("--runs-root")
    p.add_argument("--results-root")
    p.set_defaults(fn=cmd_pairs)
    p = sub.add_parser("cut-views")
    p.add_argument("--pairs", default=PAIRS_CSV)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_cut_views)
    p = sub.add_parser("match")
    p.add_argument("--pairs", default=PAIRS_CSV)
    p.add_argument("--views", required=True)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(fn=cmd_match)
    p = sub.add_parser("score")
    p.add_argument("--pairs", default=PAIRS_CSV)
    p.add_argument("--matches", default=MATCHES_JSONL)
    p.add_argument("--out", default=RESULTS_JSON)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("noise")
    p.add_argument("--out", default=NOISE_JSON)
    p.set_defaults(fn=cmd_noise)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
