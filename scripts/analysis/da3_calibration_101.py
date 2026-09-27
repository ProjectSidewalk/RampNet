"""Calibrate Depth Anything 3 against GSV's own depth, then carry it to the Mapillary splits (#101).

GSV serves a metric depth payload with every panorama; Mapillary serves none. #112 put the four
harvested GSV splits on Google's depth axis (``recall_by_depth_112.py``); the six Mapillary splits
are still on flat ground at an assumed 2.5 m camera height. This script measures a monocular
metric depth model (DA3METRIC-LARGE) against Google's depth where both exist, and then uses the
measured DA3 on the splits where only DA3 can exist.

Four stages, each a subcommand:

  * ``extract`` (GPU, klone): for every verdict-reviewed pano of the eleven bundles, render the
    six perspective views ``depth_extract_da3.py`` uses (90 deg FOV, pitch -30 deg, 1024 px, from
    the pano downscaled to 4096 px wide), run DA3 once per view with the known intrinsics, and
    record (a) the raw DA3 value at every GT point and every committed 0.55 RampNet detection,
    sampled in the view where the point is most central (7x7 median at DA3's output resolution),
    and (b) a ground-plane fit per pano from the DA3 depth of the road band (every depth pixel
    whose ray is 20-45 deg below the pano horizon; 20-60 deg as a sensitivity band). Resumable:
    one JSONL line per pano under ``--out-dir``; a requeued job skips what it already wrote.
  * ``derive`` (CPU): joins the raw DA3 values to the bundles' ground truth (the same matcher
    as ``recall_by_depth_112.py``) and to the committed #112 rows (Google's depth at the same
    points; no payload needed) and writes the committed row files and every table.
  * ``--check`` (CPU): re-derives the rows from the committed raw files + bundles + #112 JSON,
    and the tables from the rows, and fails on any byte of drift. No GPU, no payload, no panos.
  * ``--markdown``: prints the tables the doc is pasted from.

Conventions, stated once:

  * **Raw DA3 value.** ``prediction.depth`` with the synthetic views' exact intrinsics passed
    (``focal = 512 px`` for the 1024-px, 90-deg views). The ``x focal / 300`` formula in DA3's
    README is *not* applied (``scripts/analysis/README.md``, "Depth Anything 3 setup"). The
    calibration against Google measures DA3's scale directly, so nothing downstream depends on
    that choice being right; the doc reports what the scale turned out to be.
  * **z-depth vs ray depth.** DA3 is read as planar (z) depth along the view's optical axis:
    ray distance = z * |f + a r + b u| for the view basis (f, r, u) and the pixel's tangent-plane
    offsets (a, b). ``extract`` fits the ground plane under both readings and the doc reports
    which one gives a flat road (``DEPTH_CONVENTION`` is the one used).
  * **Frame.** The pano frame of ``equirect_tiling.py``: +y up, +z at pano x = 0.5. Horizontal
    range = ray * cos(latitude of the point), the same definition as #112's ``depth_range``.
  * **Ground plane.** RANSAC (fixed seed per pano) for a plane with normal within 20 deg of
    vertical, inliers within 0.10 m, then least-squares refit on the inliers. Height = the
    plane's distance from the camera (the same quantity as Google's ground-plane distance).
    The fit passes when inlier share >= 0.5 and >= 200 band pixels were fitted.

    # 1. GPU (klone), see scripts/analysis/da3_calibration_101.slurm
    python scripts/analysis/da3_calibration_101.py extract --panos-root /path/to/rampnet_benchmark \\
        --out-dir analysis_out/da3_calibration_101/raw

    # 2. CPU: rows + tables (reads the labeler's Laurens tables read-only for the cross-read)
    python scripts/analysis/da3_calibration_101.py derive --labeler-root D:/Git/sidewalk-auto-labeler

    # 3. CPU, from a clean clone: everything re-derives, byte for byte
    python scripts/analysis/da3_calibration_101.py --check
    python scripts/analysis/da3_calibration_101.py --check --markdown
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from rampnet.detection_eval import build_ground_truth  # noqa: E402
from equirect_tiling import (  # noqa: E402
    _camera_basis, _dir_from_equirect, default_views, equirect_point_to_perspective)
import recall_by_depth_112 as rbd  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "da3_calibration_101")
RAW_DIR = os.path.join(OUT_DIR, "raw")
ROWS_PANOS = os.path.join(OUT_DIR, "rows_panos.jsonl")
ROWS_POINTS = os.path.join(OUT_DIR, "rows_points.jsonl")
TABLES_JSON = os.path.join(OUT_DIR, "tables.json")
TABLES_MD = os.path.join(OUT_DIR, "tables.md")
SUMS = os.path.join(OUT_DIR, "SHA256SUMS")
RBD_JSON = rbd.OUT_JSON

GSV_DEPTH_SPLITS = ("bend", "paterson", "gainesville", "sao_paulo")   # Google depth in #112
GSV_OTHER_SPLITS = ("laurens_gsv",)                                  # GSV, never harvested
MAPILLARY_SPLITS = ("richmond", "annapolis", "morgantown", "clovis", "laurens_mapillary",
                    "budapest_district5")
ALL_SPLITS = GSV_DEPTH_SPLITS + GSV_OTHER_SPLITS + MAPILLARY_SPLITS

MODEL_ID = "depth-anything/DA3METRIC-LARGE"
PANO_MAX_EDGE = 4096          # depth_extract_da3.py's load_pano_image(path, 4096)
PATCH_HALF = 3                # 7x7 median, depth_extract_da3.sample_depth
GROUND_STRIDE = 4             # every 4th DA3 output pixel in each direction feeds the fit
BANDS = {"b20_45": (20.0, 45.0), "b20_60": (20.0, 60.0)}
PRIMARY_BAND = "b20_45"       # 45 deg keeps clear of the capture vehicle, visible from ~49 deg
                              # on the Laurens GoPro Max (labeler runs/laurens/rig_labels.json)
RANSAC_ITERS = 256
RANSAC_THRESH_M = 0.10
MAX_TILT_DEG = 20.0
FIT_MIN_INLIER_SHARE = 0.5
FIT_MIN_POINTS = 200
DEPTH_CONVENTION = "z"        # see the module docstring; "ray" is kept as the alternative
ND = 4
N_BOOT = 1000
BOOT_SEED = 101
RATIO_BUCKETS = [(0, 8), (8, 12), (12, 18), (18, 25), (25, 1e9)]   # Google range, metres
WITHIN = 0.10                 # "agrees" = within +/-10% of Google's range
NEW_RIG = rbd.NEW_RIG


# ---------------------------------------------------------------------------
# pure geometry (unit-tested; numpy only where a whole depth map is involved)

def view_dir_unnormalized(view, u, v):
    """f + a r + b up for normalized view coords (u, v); its f-component is exactly 1."""
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a, b = (2.0 * u - 1.0) * th, (1.0 - 2.0 * v) * tv
    return tuple(f[i] + a * r[i] + b * up[i] for i in range(3))


def ray_from_value(value, view, u, v, convention=DEPTH_CONVENTION):
    """Euclidean ray distance for a raw DA3 value at view coords (u, v)."""
    if value is None:
        return None
    if convention == "ray":
        return value
    d = view_dir_unnormalized(view, u, v)
    return value * math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


def horizontal_range(ray, y_norm):
    """Horizontal range of a point at ray distance ``ray`` and pano row ``y_norm``."""
    if ray is None:
        return None
    return ray * math.cos((0.5 - y_norm) * math.pi)


def best_view(x, y, views):
    """(centrality, view index, (u, v)) of the view where the point is most central, or None."""
    best = None
    for vi, v in enumerate(views):
        uv = equirect_point_to_perspective(x, y, v)
        if uv is None:
            continue
        c = max(abs(uv[0] - 0.5), abs(uv[1] - 0.5))
        if best is None or c < best[0]:
            best = (c, vi, uv)
    return best


def plane_range(normal, height, x, y):
    """Horizontal range to the plane n.p = -height along the ray through pano (x, y).

    ``normal`` is the unit normal pointing up (n_y > 0) in the pano frame, so level ground at
    height h is ((0, 1, 0), h). ``None`` when the ray does not meet the plane below the camera.
    Example: level ground, 2 m, 45 deg below the horizon -> 2.0.
    """
    d = _dir_from_equirect(x, y)
    denom = normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2]
    if denom >= -1e-6:
        return None
    t = -height / denom
    return t * math.cos((0.5 - y) * math.pi)


def fit_ground_plane(P, seed, thresh=RANSAC_THRESH_M, iters=RANSAC_ITERS, max_tilt=MAX_TILT_DEG):
    """RANSAC + least-squares plane through road-band points P (N x 3, pano frame, +y up).

    Returns {"h", "n", "tilt_deg", "inlier_share", "resid_med", "n_points"} with the plane
    written n.p = -h, n pointing up; ``None`` fields when no plane was found.
    """
    import numpy as np
    P = np.asarray(P, dtype=np.float64)
    out = {"h": None, "n": None, "tilt_deg": None, "inlier_share": None, "resid_med": None,
           "n_points": int(len(P))}
    if len(P) < 3:
        return out
    rng = np.random.default_rng(seed)
    cos_max = math.cos(math.radians(max_tilt))
    best_n, best_h, best_count = None, None, -1
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[1] < 0:
            n = -n
        if n[1] < cos_max:
            continue
        h = -float(n @ a)
        if h <= 0:
            continue
        count = int(np.count_nonzero(np.abs(P @ n + h) < thresh))
        if count > best_count:
            best_n, best_h, best_count = n, h, count
    if best_n is None:
        return out
    n, h = best_n, best_h
    for _ in range(2):   # least-squares refit on the inliers, twice
        inl = P[np.abs(P @ n + h) < thresh]
        if len(inl) < 3:
            break
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n2 = vt[-1]
        if n2[1] < 0:
            n2 = -n2
        h2 = -float(n2 @ c)
        if n2[1] < cos_max or h2 <= 0:
            break
        n, h = n2, h2
    resid = np.abs(P @ n + h)
    inl = resid < thresh
    out.update({"h": float(h), "n": [float(v) for v in n],
                "tilt_deg": math.degrees(math.acos(min(1.0, float(n[1])))),
                "inlier_share": float(inl.mean()),
                "resid_med": float(np.median(resid[inl])) if inl.any() else None})
    return out


def band_points(depth, view, lo_deg, hi_deg, convention, stride=GROUND_STRIDE, az_half=None):
    """3-D points of one view's depth map whose rays are lo..hi deg below the pano horizon.

    ``az_half``: keep only rays within this many degrees of azimuth of the view's yaw, so the
    six overlapping views partition the ring instead of double-counting it.
    """
    import numpy as np
    H, W = depth.shape
    rows = np.arange(stride // 2, H, stride)
    cols = np.arange(stride // 2, W, stride)
    u = (cols + 0.5) / W
    v = (rows + 0.5) / H
    f, r, up = (np.array(t) for t in _camera_basis(view.yaw_deg, view.pitch_deg))
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a = ((2 * u - 1) * th)[None, :, None]
    b = ((1 - 2 * v) * tv)[:, None, None]
    D = f[None, None, :] + a * r[None, None, :] + b * up[None, None, :]      # (h, w, 3)
    Dn = D / np.linalg.norm(D, axis=2, keepdims=True)
    z = depth[np.ix_(rows, cols)].astype(np.float64)
    P = D * z[..., None] if convention == "z" else Dn * z[..., None]
    dep = np.degrees(-np.arcsin(np.clip(Dn[..., 1], -1, 1)))
    keep = (dep >= lo_deg) & (dep <= hi_deg) & np.isfinite(z) & (z > 0)
    if az_half is not None:
        az = np.degrees(np.arctan2(Dn[..., 0], Dn[..., 2]))
        daz = (az - view.yaw_deg + 180.0) % 360.0 - 180.0
        keep &= np.abs(daz) <= az_half
    return P[keep]


def rig_key(make, model):
    """Normalized (make, model) rig class, as the labeler's mapillary_height.rig_key does.

    Case-folded, the make dropped from the model, firmware-like tokens (digits and dots)
    dropped. Example: ("GoPro", "GoPro Fusion FS1.04.01.80.00") -> "gopro/fusion".
    """
    make = (make or "").strip().lower()
    model = (model or "").strip().lower()
    if make in ("", "none") and model in ("", "none"):
        return "unknown"
    toks = [t for t in model.split() if t != make]
    toks = [t for t in toks if not (any(ch.isdigit() for ch in t) and "." in t)]
    return f"{make or 'unknown'}/{' '.join(toks) or 'unknown'}"


def _r(v, nd=ND):
    return None if v is None else round(float(v), nd)


def pano_seed(pano_id):
    return int(hashlib.sha256(pano_id.encode("utf-8")).hexdigest()[:8], 16)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# bundles

def load_manifest(split):
    with open(os.path.join(REPO, "benchmark", split, "imagery_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)["panos"]


def pano_points(rec, entry):
    """[(kind, index, x, y)] for one pano: every GT point, then every committed detection."""
    gt = build_ground_truth(rec["detections"], entry["dets"], entry["missed"], entry["no_missed"])
    pts = [("gt", k, g[0], g[1]) for k, g in enumerate(gt.gt_points)]
    pts += [("det", i, d["x_normalized"], d["y_normalized"]) for i, d in enumerate(rec["detections"])]
    return gt, pts


# ---------------------------------------------------------------------------
# extract (GPU)

def _load_pano(path):
    """depth_extract_da3.py's load_pano_image(path, 4096), inlined to avoid importing detectors."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    if max(img.size) > PANO_MAX_EDGE:
        s = PANO_MAX_EDGE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    return img


def _sample(dmap, u, v, half=PATCH_HALF):
    import numpy as np
    H, W = dmap.shape
    r, c = int(np.clip(v * H, 0, H - 1)), int(np.clip(u * W, 0, W - 1))
    patch = dmap[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1]
    return float(np.median(patch)) if patch.size else float(dmap[r, c])


def extract(args):
    import numpy as np
    import torch
    from equirect_tiling import equirect_to_perspective
    da3_src = args.da3_src or os.environ.get("DA3_SRC")
    if not da3_src:
        raise SystemExit("--da3-src / DA3_SRC must point at Depth-Anything-3/src (scripts/analysis/README.md)")
    sys.path.insert(0, os.path.join(da3_src, "..", "stubs"))
    sys.path.insert(0, da3_src)
    import logging
    logging.disable(logging.INFO)
    from depth_anything_3.api import DepthAnything3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.time()
    model = DepthAnything3.from_pretrained(MODEL_ID).to(dev).eval()
    t_load = time.time() - t_load
    views = default_views()
    W0 = views[0].width
    F0 = (W0 / 2) / math.tan(math.radians(views[0].fov_h_deg / 2))
    K = np.array([[[F0, 0, W0 / 2], [0, F0, W0 / 2], [0, 0, 1]]], dtype=np.float32)
    az_half = 180.0 / len(views)
    os.makedirs(args.out_dir, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"DA3 {MODEL_ID} on {gpu}; focal {F0:.0f} px; {len(views)} views; load {t_load:.1f} s", flush=True)

    run_t0 = time.time()
    per_split = {}
    for split in args.splits:
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        done = set()
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["pano"])
                    except ValueError:
                        pass   # a line cut by preemption; that pano is redone
        records, verdicts = rbd.load_bundle(split)
        manifest = load_manifest(split)
        todo = [p for p in sorted(verdicts) if p not in done]
        if args.limit:
            todo = todo[:args.limit]
        t0, n = time.time(), 0
        probed = False
        with open(out_path, "a", encoding="utf-8", newline="") as fh:
            for pid in todo:
                tp = time.time()
                path = os.path.join(args.panos_root, split, "panos", f"{pid}.jpg")
                row = {"split": split, "pano": pid}
                if not os.path.exists(path):
                    row["status"] = "missing"
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    continue
                sha = sha256_of(path)
                if sha != manifest[pid]["sha256"]:
                    row.update(status="sha256_mismatch", sha256=sha)
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    continue
                gt, pts = pano_points(records[pid], verdicts[pid])
                pano = _load_pano(path)
                t_dec = time.time() - tp
                depths = []
                t1 = time.time()
                for vw in views:
                    vimg = equirect_to_perspective(pano, vw)
                    with torch.no_grad():
                        pr = model.inference([vimg], intrinsics=K)
                    depths.append(np.asarray(pr.depth)[0].astype(np.float32))
                    if not probed:
                        # does passing intrinsics change the output? (README claim, checked)
                        with torch.no_grad():
                            pr0 = model.inference([vimg])
                        d0 = np.asarray(pr0.depth)[0]
                        row["probe_intrinsics_ratio"] = _r(float(np.median(depths[-1] / d0)))
                        probed = True
                t_da3 = time.time() - t1
                pano.close()
                prow = []
                for kind, k, x, y in pts:
                    b = best_view(x, y, views)
                    if b is None:
                        prow.append({"kind": kind, "i": k, "x": x, "y": y, "view": None})
                        continue
                    _, vi, (u, v) = b
                    prow.append({"kind": kind, "i": k, "x": x, "y": y, "view": vi,
                                 "u": _r(u, 6), "v": _r(v, 6), "value": _r(_sample(depths[vi], u, v))})
                ground = {}
                for conv in ("z", "ray"):
                    for band, (lo, hi) in BANDS.items():
                        P = np.concatenate([band_points(d, vw, lo, hi, conv, az_half=az_half)
                                            for d, vw in zip(depths, views)])
                        hsin = float(np.median(-P[:, 1])) if len(P) else None
                        fit = fit_ground_plane(P, pano_seed(pid))
                        ground[f"{conv}_{band}"] = {
                            "h": _r(fit["h"]), "n": [_r(c, 6) for c in fit["n"]] if fit["n"] else None,
                            "tilt_deg": _r(fit["tilt_deg"]), "inlier_share": _r(fit["inlier_share"]),
                            "resid_med": _r(fit["resid_med"]), "n_points": fit["n_points"],
                            "h_sin_median": _r(hsin)}
                row.update(status="ok", sha256=sha, fn_confirmed=gt.fn_confirmed, points=prow,
                           ground=ground, depth_hw=list(depths[0].shape),
                           t_decode_s=round(t_dec, 3), t_da3_s=round(t_da3, 3),
                           t_total_s=round(time.time() - tp, 3))
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                n += 1
                if n % 10 == 0:
                    print(f"  {split}: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/pano)", flush=True)
        per_split[split] = {"panos": n, "elapsed_s": round(time.time() - t0, 3)}
        print(f"{split}: {n} panos in {time.time() - t0:.1f} s", flush=True)

    total = sum(v["panos"] for v in per_split.values())
    wall = time.time() - run_t0
    usage = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "bundle": ",".join(s for s in args.splits if per_split.get(s, {}).get("panos")),
             "label": "da3-calibration-101:extract", "panos_scored": total,
             "elapsed_s": round(wall, 3), "s_per_pano": round(wall / total, 4) if total else None,
             "model_load_s": round(t_load, 3), "per_split": per_split,
             "what": "da3_calibration_101.py extract: 6 DA3METRIC-LARGE views per pano, point sampling + ground fits",
             "run_id": f"da3-calibration-101:{os.environ.get('SLURM_JOB_ID', 'local')}:"
                       f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             "provider": "depth-anything", "model_id": MODEL_ID, "paid": False,
             "hardware": {"host": os.uname().nodename if hasattr(os, "uname") else "unknown",
                          "gpus": [gpu], "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
             "status": "ok", "est_cost_usd": 0.0, "pricing": None,
             "script": "scripts/analysis/da3_calibration_101.py", "issue": 101}
    with open(os.path.join(args.out_dir, "usage_rows.jsonl"), "a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(usage) + "\n")
    print(f"done: {total} panos, {wall:.1f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# derive (CPU): raw + bundles + #112 rows -> committed rows

def read_raw(raw_dir, split):
    """{pano: row} from one split's raw JSONL; the last line per pano wins (a requeue rewrite)."""
    out = {}
    path = os.path.join(raw_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return out   # every pano of the split then reads status "not_extracted"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["pano"]] = r
    return out


def read_labeler_laurens(labeler_root):
    """The labeler's Laurens camera-height tables, read only, with the commit and file hashes."""
    base = os.path.join(labeler_root, "runs", "laurens")
    groups_csv = os.path.join(base, "camera_height", "groups.csv")
    heights_json = os.path.join(base, "camera_heights.json")
    try:
        commit = subprocess.run(["git", "-C", labeler_root, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    with open(groups_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(heights_json, encoding="utf-8") as fh:
        hj = json.load(fh)

    def num(v):
        return None if v in ("", None) else float(v)

    keep = []
    for r in rows:
        if r["grouping"] not in ("rig", "sequence"):
            continue
        keep.append({"grouping": r["grouping"], "group": r["group"], "panos": num(r["panos"]),
                     "h_bearing": num(r["h_bearing"]), "h_bearing_lo": num(r["h_bearing_lo"]),
                     "h_bearing_hi": num(r["h_bearing_hi"]), "slope": num(r["slope"]),
                     "h_scale": num(r["h_scale"]), "h_scale_lo": num(r["h_scale_lo"]),
                     "h_scale_hi": num(r["h_scale_hi"])})
    g = hj["groups"].get("gopro/max", {})
    return {"labeler_commit": commit,
            "files": {"runs/laurens/camera_height/groups.csv": sha256_of(groups_csv),
                      "runs/laurens/camera_heights.json": sha256_of(heights_json)},
            "rig_verdict": {"group": "gopro/max", "applied": g.get("applied"),
                            "height_m": g.get("height_m"), "reason": g.get("reason"),
                            "gate_passes": hj.get("gate", {}).get("passes")},
            "groups": keep}


def derive_rows(raw_dir=RAW_DIR, splits=ALL_SPLITS):
    """(pano rows, point rows) from the raw DA3 files, the bundles and the #112 JSON."""
    with open(RBD_JSON, encoding="utf-8") as fh:
        rbd_data = json.load(fh)
    g_pts = {(p["city"], p["pano"], p["x"], p["y"]): p for p in rbd_data["points"]}
    g_dets = {(p["city"], p["pano"], p["x"], p["y"]): p for p in rbd_data["detections"]}
    g_panos = {(p["city"], p["pano"]): p for p in rbd_data["panos"]}
    views = default_views()
    panos, points = [], []
    for split in splits:
        raw = read_raw(raw_dir, split)
        records, verdicts = rbd.load_bundle(split)
        for pid in sorted(verdicts):
            rec, entry = records[pid], verdicts[pid]
            r = raw.get(pid, {"status": "not_extracted"})
            meta = rec["pano"]
            gp = g_panos.get((split, pid)) if split in GSV_DEPTH_SPLITS else None
            prow = {"split": split, "pano": pid, "status": r.get("status"),
                    "source": meta.get("source"), "capture_date": meta.get("capture_date"),
                    "rig": rig_key(meta.get("camera_make"), meta.get("camera_model"))
                    if meta.get("source") == "mapillary" else "google",
                    "camera_make": meta.get("camera_make"), "camera_model": meta.get("camera_model"),
                    "sequence_id": meta.get("sequence_id"), "width": meta.get("width"),
                    "google_height_status": gp["camera_height_status"] if gp else None,
                    "google_height_m": gp["camera_height_m"] if gp else None,
                    "google_tilt_deg": gp["ground_tilt_deg"] if gp else None,
                    "probe_intrinsics_ratio": r.get("probe_intrinsics_ratio")}
            for key, fit in sorted((r.get("ground") or {}).items()):
                for f in ("h", "n", "tilt_deg", "inlier_share", "resid_med", "n_points", "h_sin_median"):
                    prow[f"{key}_{f}"] = fit.get(f)
            panos.append(prow)

            gt, pts = pano_points(rec, entry)
            preds = [(d["x_normalized"], d["y_normalized"], d["confidence"]) for d in rec["detections"]]
            hit, kinds = rbd.match(preds, gt)
            rawpts = {(q["kind"], q["i"]): q for q in r.get("points", [])}
            for kind, k, x, y in pts:
                q = rawpts.get((kind, k), {})
                if q and (q["x"] != x or q["y"] != y):
                    raise SystemExit(f"{split}/{pid}: raw point {kind}{k} moved; re-extract")
                value = q.get("value")
                ray = range_ = None
                if value is not None:
                    vw = views[q["view"]]
                    ray = ray_from_value(value, vw, q["u"], q["v"])
                    range_ = horizontal_range(ray, y)
                g = None
                if split in GSV_DEPTH_SPLITS:
                    g = (g_pts if kind == "gt" else g_dets).get((split, pid, x, y))
                row = {"split": split, "pano": pid, "kind": kind, "i": k, "x": x, "y": y,
                       "fn_confirmed": gt.fn_confirmed,
                       "hit": (k in hit) if kind == "gt" else None,
                       "det_kind": kinds[k] if kind == "det" else None,
                       "confidence": _r(preds[k][2], 6) if kind == "det" else None,
                       "flat_2p5": _r(rbd.flat_range(y, rbd.CAM_H)),
                       "da3_value": value, "da3_view": q.get("view"),
                       "da3_ray": _r(ray), "da3_range": _r(range_),
                       "google_range": g["depth_range"] if g else None,
                       "google_ray": g["depth_ray"] if g else None,
                       "google_source": g["depth_source"] if g else None,
                       "google_status": g["camera_height_status"] if g else None}
                points.append(row)
    return panos, points


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# statistics (pure Python, so the tables are identical on every numpy build)

def _q(sorted_vals, p):
    """Quantile by the nearest-rank-below rule #112 uses for p10 / p90."""
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def ols(xs, ys):
    """(slope, intercept) of y on x, or (None, None) with < 3 points."""
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def cluster_bootstrap(items, cluster_key, stat, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI of ``stat(items)`` resampling whole clusters (panos) with replacement."""
    clusters = {}
    for it in items:
        clusters.setdefault(cluster_key(it), []).append(it)
    keys = sorted(clusters)
    if len(keys) < 5:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = []
        for _ in keys:
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    vals.sort()
    return [_r(vals[int(0.025 * (len(vals) - 1))]), _r(vals[int(0.975 * (len(vals) - 1))])]


def _median_ratio(pairs):
    return st.median(a / b for a, b in pairs) if pairs else None


def _slope(pairs):
    return ols([b for _, b in pairs], [a for a, _ in pairs])[0]


def _loglog(pairs):
    return ols([math.log(b) for _, b in pairs], [math.log(a) for a, _ in pairs])[0]


# ---------------------------------------------------------------------------
# tables

def fit_ok(p, band=PRIMARY_BAND, conv=DEPTH_CONVENTION):
    k = f"{conv}_{band}"
    return (p.get("status") == "ok" and p.get(f"{k}_h") is not None
            and (p.get(f"{k}_inlier_share") or 0) >= FIT_MIN_INLIER_SHARE
            and (p.get(f"{k}_n_points") or 0) >= FIT_MIN_POINTS)


def _year(p):
    return (p.get("capture_date") or "")[:4]


def rig_group(split, year):
    """The #112 capture-vintage groups for the GSV splits."""
    if (split, year) in NEW_RIG:
        return "2025-26 rig"
    if split == "sao_paulo":
        return "sao_paulo"
    return "older US vintages"


def calib_locations(points, panos_by):
    """Unique (split, pano, x, y) GSV locations with DA3 and Google pixel-plane range, measured ground.

    A true-positive detection and its GT point share coordinates; they are one location.
    """
    seen, out = set(), []
    for p in points:
        if p["split"] not in GSV_DEPTH_SPLITS:
            continue
        key = (p["split"], p["pano"], p["x"], p["y"])
        if key in seen:
            continue
        if (p["google_status"] == "measured" and p["google_source"] == "pixel_plane"
                and p["google_range"] and p["da3_range"]):
            seen.add(key)
            pn = panos_by[(p["split"], p["pano"])]
            out.append({"split": p["split"], "pano": p["pano"], "x": p["x"], "y": p["y"],
                        "da3": p["da3_range"], "google": p["google_range"],
                        "flat_2p5": p["flat_2p5"], "group": rig_group(p["split"], _year(pn))})
    return out


def point_calib_row(label, locs):
    pairs = [(q["da3"], q["google"]) for q in locs]
    if len(pairs) < 3:
        return {"group": label, "n": len(pairs)}
    ratios = sorted(a / b for a, b in pairs)
    slope, icpt = ols([b for _, b in pairs], [a for a, _ in pairs])
    ck = lambda q: (q["split"], q["pano"])  # noqa: E731
    return {"group": label, "n": len(pairs), "n_panos": len({ck(q) for q in locs}),
            "median_ratio": _r(st.median(ratios)),
            "median_ratio_ci": cluster_bootstrap(locs, ck, lambda s: _median_ratio([(q["da3"], q["google"]) for q in s])),
            "p10_ratio": _r(_q(ratios, 0.1)), "p90_ratio": _r(_q(ratios, 0.9)),
            "ols_slope": _r(slope), "ols_intercept_m": _r(icpt),
            "ols_slope_ci": cluster_bootstrap(locs, ck, lambda s: _slope([(q["da3"], q["google"]) for q in s])),
            "loglog_exponent": _r(_loglog(pairs)),
            "loglog_exponent_ci": cluster_bootstrap(locs, ck, lambda s: _loglog([(q["da3"], q["google"]) for q in s])),
            "median_abs_log_ratio": _r(st.median(abs(math.log(r)) for r in ratios)),
            "median_google_m": _r(st.median(b for _, b in pairs))}


def height_pairs(panos, splits):
    out = []
    k = f"{DEPTH_CONVENTION}_{PRIMARY_BAND}"
    for p in panos:
        if p["split"] in splits and fit_ok(p) and p["google_height_status"] == "measured":
            out.append({"split": p["split"], "pano": p["pano"], "da3": p[f"{k}_h"],
                        "google": p["google_height_m"], "group": rig_group(p["split"], _year(p)),
                        "da3_sin": p[f"{k}_h_sin_median"]})
    return out


def height_calib_row(label, hp):
    if len(hp) < 3:
        return {"group": label, "n_panos": len(hp)}
    ratios = sorted(q["da3"] / q["google"] for q in hp)
    slope, icpt = ols([q["google"] for q in hp], [q["da3"] for q in hp])
    ck = lambda q: q["pano"]  # noqa: E731
    return {"group": label, "n_panos": len(hp),
            "median_da3_h_m": _r(st.median(q["da3"] for q in hp)),
            "median_google_h_m": _r(st.median(q["google"] for q in hp)),
            "median_ratio": _r(st.median(ratios)),
            "median_ratio_ci": cluster_bootstrap(hp, ck, lambda s: _median_ratio([(q["da3"], q["google"]) for q in s])),
            "p10_ratio": _r(_q(ratios, 0.1)), "p90_ratio": _r(_q(ratios, 0.9)),
            "pearson_r": _r(pearson([q["google"] for q in hp], [q["da3"] for q in hp])),
            "ols_slope": _r(slope), "ols_intercept_m": _r(icpt),
            "median_abs_log_ratio": _r(st.median(abs(math.log(r)) for r in ratios))}


def axis_values(p, pano, k_pt, k_h):
    """The candidate distance axes at one point, given the two calibration constants."""
    key = f"{DEPTH_CONVENTION}_{PRIMARY_BAND}"
    out = {"flat_2p5": p["flat_2p5"],
           "da3_point": (p["da3_range"] / k_pt) if (p["da3_range"] and k_pt) else None,
           "da3_plane": None, "flat_da3_height": None}
    if fit_ok(pano) and k_h:
        h = pano[f"{key}_h"] / k_h
        out["da3_plane"] = plane_range(pano[f"{key}_n"], h, p["x"], p["y"])
        out["flat_da3_height"] = rbd.flat_range(p["y"], h)
    return out


AXES = ("flat_2p5", "da3_point", "da3_plane", "flat_da3_height")
DA3_AXES = ("da3_point", "da3_plane", "flat_da3_height")


def agreement(pairs):
    """How well an axis reproduces Google's range: median ratio, median |ln ratio|, share within 10%."""
    if not pairs:
        return {"n": 0}
    ratios = sorted(a / b for a, b in pairs)
    return {"n": len(pairs), "median_ratio": _r(st.median(ratios)),
            "p10_ratio": _r(_q(ratios, 0.1)), "p90_ratio": _r(_q(ratios, 0.9)),
            "median_abs_log_ratio": _r(st.median(abs(math.log(r)) for r in ratios)),
            "share_within_10pct": _r(sum(1 for r in ratios if abs(r - 1) <= WITHIN) / len(ratios))}


def tables(panos, points, labeler):
    panos_by = {(p["split"], p["pano"]): p for p in panos}
    first_at = {}
    for p in points:   # the first point row at a location (the GT row when a TP shares it)
        first_at.setdefault((p["split"], p["pano"], p["x"], p["y"]), p)
    t = {}
    key = f"{DEPTH_CONVENTION}_{PRIMARY_BAND}"

    # -- inventory
    inv = []
    for s in ALL_SPLITS:
        sp = [p for p in panos if p["split"] == s]
        spts = [q for q in points if q["split"] == s]
        inv.append({"split": s, "panos": len(sp),
                    "extracted": sum(1 for p in sp if p["status"] == "ok"),
                    "fit_ok": sum(1 for p in sp if fit_ok(p)),
                    "gt_points": sum(1 for q in spts if q["kind"] == "gt"),
                    "gt_points_fn_confirmed": sum(1 for q in spts if q["kind"] == "gt" and q["fn_confirmed"]),
                    "detections": sum(1 for q in spts if q["kind"] == "det"),
                    "points_with_da3": sum(1 for q in spts if q["da3_range"] is not None)})
    t["inventory"] = inv
    probes = sorted(p["probe_intrinsics_ratio"] for p in panos if p.get("probe_intrinsics_ratio") is not None)
    t["intrinsics_probe"] = {"n": len(probes), "min": probes[0] if probes else None,
                             "max": probes[-1] if probes else None}

    # -- z vs ray: which reading gives a flatter road (GSV + Mapillary, primary band)
    conv = {}
    for c in ("z", "ray"):
        k = f"{c}_{PRIMARY_BAND}"
        ok = [p for p in panos if p["status"] == "ok" and p.get(f"{k}_h") is not None]
        conv[c] = {"n_panos": len(ok),
                   "median_inlier_share": _r(st.median(p[f"{k}_inlier_share"] for p in ok)) if ok else None,
                   "median_resid_m": _r(st.median(p[f"{k}_resid_med"] for p in ok if p[f"{k}_resid_med"] is not None)) if ok else None,
                   "median_tilt_deg": _r(st.median(p[f"{k}_tilt_deg"] for p in ok)) if ok else None,
                   "fit_ok": sum(1 for p in panos if fit_ok(p, conv=c))}
    t["convention"] = conv

    # -- 1. DA3 vs Google at the same points
    locs = calib_locations(points, panos_by)
    rows = [point_calib_row(s, [q for q in locs if q["split"] == s]) for s in GSV_DEPTH_SPLITS]
    for g in ("2025-26 rig", "older US vintages", "sao_paulo"):
        rows.append(point_calib_row(g, [q for q in locs if q["group"] == g]))
    rows.append(point_calib_row("gsv_pooled", locs))
    t["point_calibration"] = rows
    by_bucket = []
    for lo, hi in RATIO_BUCKETS:
        b = [q for q in locs if lo <= q["google"] < hi]
        if b:
            rr = sorted(q["da3"] / q["google"] for q in b)
            fr = sorted(q["flat_2p5"] / q["google"] for q in b if q["flat_2p5"])
            by_bucket.append({"bucket": rbd.bucket_label(lo, hi, "m"), "n": len(b),
                              "median_da3_over_google": _r(st.median(rr)),
                              "p10": _r(_q(rr, 0.1)), "p90": _r(_q(rr, 0.9)),
                              "median_flat_over_google": _r(st.median(fr)) if fr else None})
    t["ratio_by_google_range"] = by_bucket

    # -- 2. DA3 camera height vs Google camera height, per pano
    hp = height_pairs(panos, GSV_DEPTH_SPLITS)
    hrows = [height_calib_row(s, [q for q in hp if q["split"] == s]) for s in GSV_DEPTH_SPLITS]
    for g in ("2025-26 rig", "older US vintages", "sao_paulo"):
        hrows.append(height_calib_row(g, [q for q in hp if q["group"] == g]))
    hrows.append(height_calib_row("gsv_pooled", hp))
    t["height_calibration"] = hrows

    # -- 3. constants: pooled, and leave-one-split-out
    def consts(splits):
        lp = [(q["da3"], q["google"]) for q in locs if q["split"] in splits]
        hq = [(q["da3"], q["google"]) for q in hp if q["split"] in splits]
        return _median_ratio(lp), _median_ratio(hq)

    k_pt, k_h = consts(GSV_DEPTH_SPLITS)
    t["constants"] = {"k_point": _r(k_pt), "k_height": _r(k_h),
                      "loso": {s: dict(zip(("k_point", "k_height"),
                                           (_r(v) for v in consts([o for o in GSV_DEPTH_SPLITS if o != s]))))
                               for s in GSV_DEPTH_SPLITS},
                      "depth_convention": DEPTH_CONVENTION, "band": PRIMARY_BAND,
                      "fit_min_inlier_share": FIT_MIN_INLIER_SHARE, "fit_min_points": FIT_MIN_POINTS}

    # -- 4. validation on GSV: each axis against Google, leave-one-split-out constants
    val = {a: [] for a in AXES}
    val_by_split = {}
    for s in GSV_DEPTH_SPLITS:
        c = t["constants"]["loso"][s]
        per = {a: [] for a in AXES}
        for q in locs:
            if q["split"] != s:
                continue
            p = first_at[(q["split"], q["pano"], q["x"], q["y"])]
            ax = axis_values(p, panos_by[(q["split"], q["pano"])], c["k_point"], c["k_height"])
            for a in AXES:
                if ax[a]:
                    per[a].append((ax[a], q["google"]))
        val_by_split[s] = {a: agreement(per[a]) for a in AXES}
        for a in AXES:
            val[a].extend(per[a])
    # the common subset: locations where every axis has a value, so the axes are compared on
    # the same points
    common = []
    for q in locs:
        p = first_at[(q["split"], q["pano"], q["x"], q["y"])]
        c = t["constants"]["loso"][q["split"]]
        ax = axis_values(p, panos_by[(q["split"], q["pano"])], c["k_point"], c["k_height"])
        if all(ax[a] for a in AXES):
            common.append((ax, q["google"]))
    t["validation"] = {"by_split": val_by_split,
                       "pooled": {a: agreement(val[a]) for a in AXES},
                       "pooled_common": {a: agreement([(ax[a], g) for ax, g in common]) for a in AXES}}
    headline = min(DA3_AXES, key=lambda a: t["validation"]["pooled_common"][a].get("median_abs_log_ratio", math.inf))
    t["headline_axis"] = headline

    # -- 5. camera height per split and per rig (calibrated by k_height)
    def hstats(label, ps):
        hs = sorted(p[f"{key}_h"] / k_h for p in ps)
        if not hs:
            return {"group": label, "n_panos": 0}
        tilts = [p[f"{key}_tilt_deg"] for p in ps]
        return {"group": label, "n_panos": len(hs), "median_h_m": _r(st.median(hs)),
                "p25_h_m": _r(_q(hs, 0.25)), "p75_h_m": _r(_q(hs, 0.75)),
                "min_h_m": _r(hs[0]), "max_h_m": _r(hs[-1]),
                "median_tilt_deg": _r(st.median(tilts)),
                "median_raw_da3_h_m": _r(st.median(p[f"{key}_h"] for p in ps))}

    hs_split, hs_rig = [], []
    for s in ALL_SPLITS:
        ps = [p for p in panos if p["split"] == s and fit_ok(p)]
        row = hstats(s, ps)
        row["panos"] = sum(1 for p in panos if p["split"] == s)
        gh = sorted(p["google_height_m"] for p in ps if p["google_height_status"] == "measured")
        row["median_google_h_m"] = _r(st.median(gh)) if gh else None
        hs_split.append(row)
        if s in MAPILLARY_SPLITS:
            for rg in sorted({p["rig"] for p in ps}):
                rr = hstats(f"{s} | {rg}", [p for p in ps if p["rig"] == rg])
                hs_rig.append(rr)
    for rg in sorted({p["rig"] for p in panos if p["split"] in MAPILLARY_SPLITS}):
        hs_rig.append(hstats(f"all Mapillary | {rg}",
                             [p for p in panos if p["split"] in MAPILLARY_SPLITS and p["rig"] == rg and fit_ok(p)]))
    t["camera_height_by_split"] = hs_split
    t["camera_height_by_rig"] = hs_rig

    # -- 6. recall by distance on Mapillary (and laurens_gsv), flat 2.5 m beside the DA3 axes
    rec, thr = {}, {}
    groups = {s: s for s in MAPILLARY_SPLITS + GSV_OTHER_SPLITS}
    groups["mapillary_pooled"] = MAPILLARY_SPLITS
    for name, members in groups.items():
        members = (members,) if isinstance(members, str) else members
        gp = []
        for p in points:
            if p["split"] in members and p["kind"] == "gt" and p["fn_confirmed"]:
                ax = axis_values(p, panos_by[(p["split"], p["pano"])], k_pt, k_h)
                gp.append({"hit": p["hit"], **{a: ax[a] for a in AXES}})
        rec[name] = {a: rbd.recall_table(gp, a, rbd.M_BUCKETS, "m") for a in AXES}
        rec[name]["n_gt"] = len(gp)
        thr[name] = {a: [rbd.window_threshold(gp, th, flat_key="flat_2p5", depth_key=a)
                         for th in rbd.PUBLISHED_THRESHOLDS_M] for a in DA3_AXES}
    t["recall_distance"] = rec
    t["thresholds"] = thr

    # -- 7. Laurens cross-read: DA3 heights by sequence vs the labeler's instruments
    lm = [p for p in panos if p["split"] == "laurens_mapillary" and fit_ok(p)]
    lg = [p for p in panos if p["split"] == "laurens_gsv" and fit_ok(p)]
    lab = {g["group"]: g for g in labeler["groups"] if g["grouping"] == "sequence"}
    seq_rows = []
    for sq in sorted({p["sequence_id"] for p in lm}):
        ps = [p for p in lm if p["sequence_id"] == sq]
        lr = lab.get(sq, {})
        seq_rows.append({"sequence_id": sq, "n_panos": len(ps),
                         "da3_h_m": _r(st.median(p[f"{key}_h"] / k_h for p in ps)),
                         "labeler_h_scale": lr.get("h_scale"), "labeler_h_bearing": lr.get("h_bearing")})
    rig = next((g for g in labeler["groups"] if g["grouping"] == "rig" and g["group"] == "gopro/max"), {})
    paired = [r for r in seq_rows if r["labeler_h_scale"] is not None]
    t["laurens_cross_read"] = {
        "labeler_commit": labeler["labeler_commit"], "labeler_files": labeler["files"],
        "labeler_rig_verdict": labeler["rig_verdict"],
        "rig": {"group": "gopro/max", "da3_n_panos": len(lm),
                "da3_median_h_m": _r(st.median(p[f"{key}_h"] / k_h for p in lm)) if lm else None,
                "labeler_h_scale": rig.get("h_scale"), "labeler_h_scale_ci": [rig.get("h_scale_lo"), rig.get("h_scale_hi")],
                "labeler_h_bearing": rig.get("h_bearing"),
                "labeler_h_bearing_ci": [rig.get("h_bearing_lo"), rig.get("h_bearing_hi")]},
        "laurens_gsv_da3_median_h_m": _r(st.median(p[f"{key}_h"] / k_h for p in lg)) if lg else None,
        "laurens_gsv_n_panos": len(lg),
        "sequences": seq_rows,
        "sequence_pearson_r_vs_h_scale": _r(pearson([r["labeler_h_scale"] for r in paired],
                                                     [r["da3_h_m"] for r in paired])) if len(paired) >= 3 else None,
        "n_sequences_paired": len(paired)}

    # -- 8. the band sensitivity: 20-60 vs 20-45 heights on the same panos
    alt = f"{DEPTH_CONVENTION}_b20_60"
    sens = []
    for s in ALL_SPLITS:
        ps = [p for p in panos if p["split"] == s and fit_ok(p) and fit_ok(p, band="b20_60")]
        if ps:
            sens.append({"split": s, "n_panos": len(ps),
                         "median_ratio_60_over_45": _r(st.median(p[f"{alt}_h"] / p[f"{key}_h"] for p in ps)),
                         "median_inlier_share_45": _r(st.median(p[f"{key}_inlier_share"] for p in ps)),
                         "median_inlier_share_60": _r(st.median(p[f"{alt}_inlier_share"] for p in ps))})
    t["band_sensitivity"] = sens
    return t


# ---------------------------------------------------------------------------
# markdown

def _f(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(ci, nd=3):
    return "–" if not ci else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def markdown(t):
    L = ["# DA3 calibration against GSV depth, carried to Mapillary (#101)\n",
         "Generated by `scripts/analysis/da3_calibration_101.py --check --markdown` from the committed rows.\n"]
    c = t["constants"]
    L.append(f"Depth convention `{c['depth_convention']}`, ground band `{c['band']}`, "
             f"k_point = {c['k_point']}, k_height = {c['k_height']}, headline axis `{t['headline_axis']}`.\n")
    L.append("## Inventory\n")
    L.append("| split | panos | extracted | ground fit ok | GT points (fn-confirmed) | detections | points with DA3 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in t["inventory"]:
        L.append(f"| {r['split']} | {r['panos']} | {r['extracted']} | {r['fit_ok']} | {r['gt_points']} "
                 f"({r['gt_points_fn_confirmed']}) | {r['detections']} | {r['points_with_da3']} |")
    L.append("\n## z-depth vs ray-depth reading (primary band, all panos)\n")
    L.append("| reading | panos fitted | fit ok | median inlier share | median residual (m) | median tilt (deg) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, r in t["convention"].items():
        L.append(f"| {k} | {r['n_panos']} | {r['fit_ok']} | {_f(r['median_inlier_share'])} | "
                 f"{_f(r['median_resid_m'])} | {_f(r['median_tilt_deg'], 2)} |")
    L.append(f"\nIntrinsics probe (DA3 output with / without the known intrinsics, first pano of each "
             f"split-run): n = {t['intrinsics_probe']['n']}, range {t['intrinsics_probe']['min']}–{t['intrinsics_probe']['max']}.\n")
    L.append("## DA3 range vs Google range at the same points (GSV, measured ground, pixel-plane)\n")
    L.append("| group | locations (panos) | DA3/Google median [95% CI] | p10–p90 | OLS slope [CI] | intercept (m) | log-log exponent [CI] | median abs ln ratio |")
    L.append("|---|---:|---|---|---|---:|---|---:|")
    for r in t["point_calibration"]:
        if "median_ratio" not in r:
            continue
        L.append(f"| {r['group']} | {r['n']} ({r['n_panos']}) | {_f(r['median_ratio'])} {_ci(r['median_ratio_ci'])} | "
                 f"{_f(r['p10_ratio'], 2)}–{_f(r['p90_ratio'], 2)} | {_f(r['ols_slope'])} {_ci(r['ols_slope_ci'])} | "
                 f"{_f(r['ols_intercept_m'], 2)} | {_f(r['loglog_exponent'])} {_ci(r['loglog_exponent_ci'])} | "
                 f"{_f(r['median_abs_log_ratio'])} |")
    L.append("\n## DA3/Google by Google range (pooled GSV)\n")
    L.append("| Google range | n | DA3/Google median (p10–p90) | flat 2.5 m / Google median |")
    L.append("|---|---:|---|---:|")
    for r in t["ratio_by_google_range"]:
        L.append(f"| {r['bucket']} | {r['n']} | {_f(r['median_da3_over_google'])} ({_f(r['p10'], 2)}–{_f(r['p90'], 2)}) | "
                 f"{_f(r['median_flat_over_google'])} |")
    L.append("\n## DA3 camera height vs Google camera height, per pano\n")
    L.append("| group | panos | DA3 h median | Google h median | DA3/Google median [CI] | p10–p90 | Pearson r | OLS slope | median abs ln ratio |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---:|")
    for r in t["height_calibration"]:
        if "median_ratio" not in r:
            continue
        L.append(f"| {r['group']} | {r['n_panos']} | {_f(r['median_da3_h_m'], 2)} | {_f(r['median_google_h_m'], 2)} | "
                 f"{_f(r['median_ratio'])} {_ci(r['median_ratio_ci'])} | {_f(r['p10_ratio'], 2)}–{_f(r['p90_ratio'], 2)} | "
                 f"{_f(r['pearson_r'])} | {_f(r['ols_slope'])} | {_f(r['median_abs_log_ratio'])} |")
    L.append("\n## Leave-one-split-out validation: each axis against Google's range\n")
    L.append("Constants fitted on the other three GSV splits. `pooled_common` = the locations where every axis has a value.\n")
    L.append("| population | axis | n | median axis/Google (p10–p90) | median abs ln ratio | share within 10% |")
    L.append("|---|---|---:|---|---:|---:|")
    for pop in ("pooled_common", "pooled"):
        for a in AXES:
            r = t["validation"][pop][a]
            if r.get("n"):
                L.append(f"| {pop} | {a} | {r['n']} | {_f(r['median_ratio'])} ({_f(r['p10_ratio'], 2)}–{_f(r['p90_ratio'], 2)}) | "
                         f"{_f(r['median_abs_log_ratio'])} | {_f(r['share_within_10pct'])} |")
    for s, rows in t["validation"]["by_split"].items():
        for a in AXES:
            r = rows[a]
            if r.get("n"):
                L.append(f"| {s} | {a} | {r['n']} | {_f(r['median_ratio'])} ({_f(r['p10_ratio'], 2)}–{_f(r['p90_ratio'], 2)}) | "
                         f"{_f(r['median_abs_log_ratio'])} | {_f(r['share_within_10pct'])} |")
    L.append("\n## Camera height by split (DA3, calibrated by k_height)\n")
    L.append("| split | panos | fit ok | median h (p25–p75) | min–max | median tilt (deg) | Google h median |")
    L.append("|---|---:|---:|---|---|---:|---:|")
    for r in t["camera_height_by_split"]:
        if not r["n_panos"]:
            continue
        L.append(f"| {r['group']} | {r['panos']} | {r['n_panos']} | {_f(r['median_h_m'], 2)} ({_f(r['p25_h_m'], 2)}–{_f(r['p75_h_m'], 2)}) | "
                 f"{_f(r['min_h_m'], 2)}–{_f(r['max_h_m'], 2)} | {_f(r['median_tilt_deg'], 1)} | {_f(r['median_google_h_m'], 2)} |")
    L.append("\n## Camera height by Mapillary rig (make/model)\n")
    L.append("| split / rig | panos | median h (p25–p75) | min–max | median tilt (deg) |")
    L.append("|---|---:|---|---|---:|")
    for r in t["camera_height_by_rig"]:
        if not r["n_panos"]:
            continue
        L.append(f"| {r['group']} | {r['n_panos']} | {_f(r['median_h_m'], 2)} ({_f(r['p25_h_m'], 2)}–{_f(r['p75_h_m'], 2)}) | "
                 f"{_f(r['min_h_m'], 2)}–{_f(r['max_h_m'], 2)} | {_f(r['median_tilt_deg'], 1)} |")
    for name, tb in t["recall_distance"].items():
        L.append(f"\n## Recall by distance: {name} ({tb['n_gt']} fn-confirmed GT points)\n")
        L.append(rbd._side_by_side([(a, tb[a]) for a in AXES]))
        L.append("")
        th = t["thresholds"][name]
        L.append("| axis | 18 m on flat becomes (window n) | 25 m on flat becomes (window n) |")
        L.append("|---|---|---|")
        for a in DA3_AXES:
            w = th[a]
            L.append(f"| {a} | {rbd._thr(w[0])} | {rbd._thr(w[1])} |")
    lc = t["laurens_cross_read"]
    L.append(f"\n## Laurens cross-read (labeler commit `{lc['labeler_commit']}`)\n")
    r = lc["rig"]
    L.append(f"GoPro Max rig: DA3 median {r['da3_median_h_m']} m over {r['da3_n_panos']} panos; labeler "
             f"h_scale {r['labeler_h_scale']} {r['labeler_h_scale_ci']}, h_bearing {r['labeler_h_bearing']} "
             f"{r['labeler_h_bearing_ci']}; labeler verdict {lc['labeler_rig_verdict']}. laurens_gsv (Google rig, "
             f"same footprint): DA3 median {lc['laurens_gsv_da3_median_h_m']} m over {lc['laurens_gsv_n_panos']} panos. "
             f"Sequence-level Pearson r (DA3 vs h_scale) = {lc['sequence_pearson_r_vs_h_scale']} over "
             f"{lc['n_sequences_paired']} sequences.\n")
    L.append("| sequence | panos | DA3 h (m) | labeler h_scale | labeler h_bearing |")
    L.append("|---|---:|---:|---:|---:|")
    for s in lc["sequences"]:
        L.append(f"| {s['sequence_id']} | {s['n_panos']} | {_f(s['da3_h_m'], 2)} | {_f(s['labeler_h_scale'], 2)} | "
                 f"{_f(s['labeler_h_bearing'], 2)} |")
    L.append("\n## Band sensitivity: fitted height, 20–60 deg band / 20–45 deg band\n")
    L.append("| split | panos | median ratio | inlier share 20–45 | inlier share 20–60 |")
    L.append("|---|---:|---:|---:|---:|")
    for r in t["band_sensitivity"]:
        L.append(f"| {r['split']} | {r['n_panos']} | {_f(r['median_ratio_60_over_45'])} | "
                 f"{_f(r['median_inlier_share_45'])} | {_f(r['median_inlier_share_60'])} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main

def committed_files():
    raws = [os.path.join(RAW_DIR, f"{s}.jsonl") for s in ALL_SPLITS]
    return raws + [ROWS_PANOS, ROWS_POINTS, TABLES_JSON, TABLES_MD]


def write_sums():
    lines = []
    for p in committed_files() + [RBD_JSON]:
        lines.append(f"{sha256_of(p)}  {os.path.relpath(p, REPO).replace(os.sep, '/')}")
    with open(SUMS, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def check(markdown_out=False):
    """Re-derive rows from raw + bundles + #112, tables from rows; fail on drift."""
    with open(TABLES_JSON, encoding="utf-8") as fh:
        stored = json.load(fh)
    panos, points = derive_rows()
    if read_jsonl(ROWS_PANOS) != panos or read_jsonl(ROWS_POINTS) != points:
        raise SystemExit("rows do not re-derive from the committed raw files, bundles and #112 JSON")
    fresh = tables(panos, points, stored["labeler_laurens"])
    fresh = json.loads(json.dumps(fresh))
    if fresh != stored["tables"]:
        raise SystemExit(f"{TABLES_JSON}: tables do not re-derive from the committed rows")
    with open(TABLES_MD, encoding="utf-8", newline="") as fh:
        if fh.read() != markdown(fresh):
            raise SystemExit(f"{TABLES_MD} is not the markdown of the committed tables")
    with open(SUMS, encoding="utf-8") as fh:
        for line in fh:
            digest, rel = line.split()
            if sha256_of(os.path.join(REPO, rel)) != digest:
                raise SystemExit(f"SHA256SUMS: {rel} changed")
    print(f"ok: {len(panos)} panos, {len(points)} points; rows, tables, markdown and SHA256SUMS re-derive")
    if markdown_out:
        print(markdown(fresh))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("extract", "derive"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--splits", nargs="+", default=list(ALL_SPLITS))
    ap.add_argument("--panos-root", help="directory holding <split>/panos/<pano>.jpg (extract)")
    ap.add_argument("--out-dir", default=RAW_DIR, help="raw JSONL directory (extract)")
    ap.add_argument("--da3-src", help="Depth-Anything-3/src (default: $DA3_SRC)")
    ap.add_argument("--limit", type=int, default=0, help="extract: at most this many new panos per split")
    ap.add_argument("--labeler-root", default=os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler"))
    a = ap.parse_args(argv)
    if a.check:
        return check(a.markdown)
    if a.stage == "extract":
        if not a.panos_root:
            ap.error("extract needs --panos-root")
        return extract(a)
    if a.stage == "derive":
        panos, points = derive_rows()
        labeler = read_labeler_laurens(a.labeler_root)
        t = json.loads(json.dumps(tables(panos, points, labeler)))
        write_jsonl(ROWS_PANOS, panos)
        write_jsonl(ROWS_POINTS, points)
        write_json(TABLES_JSON, {"labeler_laurens": labeler, "tables": t,
                                 "inputs": {"recall_by_depth_112.json": sha256_of(RBD_JSON)}})
        with open(TABLES_MD, "w", encoding="utf-8", newline="") as fh:
            fh.write(markdown(t))
        write_sums()
        print(f"wrote {len(panos)} pano rows, {len(points)} point rows, tables -> {OUT_DIR}")
        if a.markdown:
            print(markdown(t))
        return 0
    ap.error("give a stage (extract / derive) or --check")


if __name__ == "__main__":
    sys.exit(main())
