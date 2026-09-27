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
# The first full extraction (klone job 40774944, commit 4ab47a2): the single dominant plane of the
# band, pass rule inlier share >= 0.5. Kept so the post-hoc ground-fit change can be attributed
# to its two parts from committed rows (review of #203, M1 / B2).
RAW_RUN1_DIR = os.path.join(OUT_DIR, "raw_run1")
RUN1_FIT_MIN_INLIER_SHARE = 0.5
SAME_PLANE_REL = 0.03         # two fitted heights within 3% are "the same plane"
ROOF_BELOW_M = 1.3            # an uncalibrated fitted height below this is read as a vehicle roof
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
# Pinned inputs of the GPU step (review of #203, M5). Both runs used exactly these: the weights
# snapshot is the one in the run's HF cache (refs/main), the code the commit of its DA3 clone.
# The runs predate the pin in code; extract now loads this revision and refuses other DA3 code.
MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
DA3_CODE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PANO_MAX_EDGE = 4096          # depth_extract_da3.py's load_pano_image(path, 4096)
PATCH_HALF = 3                # 7x7 median, depth_extract_da3.sample_depth
GROUND_STRIDE = 4             # every 4th DA3 output pixel in each direction feeds the fit
BANDS = {"b20_45": (20.0, 45.0), "b20_60": (20.0, 60.0)}
PRIMARY_BAND = "b20_45"       # 45 deg keeps clear of the capture vehicle, visible from ~49 deg
                              # on the Laurens GoPro Max (labeler runs/laurens/rig_labels.json)
RANSAC_ITERS = 256
RANSAC_THRESH_M = 0.10
MAX_TILT_DEG = 20.0
FIT_MIN_INLIER_SHARE = 0.25   # share of ALL band points on the selected (lowest) plane
FIT_MIN_POINTS = 200
MAX_PLANES = 3                # sequential RANSAC: up to this many near-horizontal planes...
MIN_PLANE_SHARE = 0.15        # ...each holding at least this share of the band points
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


def fit_lowest_plane(P, seed, max_planes=MAX_PLANES, min_share=MIN_PLANE_SHARE):
    """The ground: the LOWEST well-supported near-horizontal plane in the road band.

    Sequential RANSAC (``fit_ground_plane`` on what the previous planes left) finds up to
    ``max_planes`` planes that each hold >= ``min_share`` of the band points, and returns the
    one farthest below the camera. The first extraction took the single dominant plane, and on
    car-roof consumer rigs (morgantown's GoPro Max most of all) that plane was the vehicle's
    own roof ~0.75 m under the camera, not the road. The returned dict is
    ``fit_ground_plane``'s, with ``inlier_share`` re-expressed as a share of ALL band points,
    plus ``dominant_h`` (the first plane found) and ``planes`` ([h, share] of each plane).
    """
    import numpy as np
    P = np.asarray(P, dtype=np.float64)
    total = len(P)
    remaining, planes = P, []
    for k in range(max_planes):
        fit = fit_ground_plane(remaining, seed + k)
        if fit["h"] is None:
            break
        n = np.array(fit["n"])
        inl = np.abs(remaining @ n + fit["h"]) < RANSAC_THRESH_M
        share = float(inl.sum()) / total if total else 0.0
        if share < min_share:
            break
        fit["inlier_share"] = share
        planes.append(fit)
        remaining = remaining[~inl]
        if len(remaining) < 3:
            break
    if not planes:
        out = fit_ground_plane(P[:0], seed)
        out.update(n_points=int(total), dominant_h=None, planes=[])
        return out
    best = max(planes, key=lambda f: f["h"])
    out = dict(best)
    out.update(n_points=int(total), dominant_h=planes[0]["h"],
               planes=[[round(f["h"], ND), round(f["inlier_share"], ND)] for f in planes])
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


def normalized_sha256(path):
    """sha256 of a text file with CRLF folded to LF, so a core.autocrlf=true checkout hashes the
    same as the LF bytes this script writes (review B1). The files are also pinned LF in
    .gitattributes; this makes the check true even where that pin has not been applied."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


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

    try:
        da3_commit = subprocess.run(["git", "-C", os.path.join(da3_src, ".."), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        da3_commit = None
    if da3_commit != DA3_CODE_COMMIT and not args.allow_other_da3:
        raise SystemExit(f"DA3 code at {da3_commit}, pinned {DA3_CODE_COMMIT} (pass --allow-other-da3 to run anyway)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.time()
    model = DepthAnything3.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(dev).eval()
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
                        fit = fit_lowest_plane(P, pano_seed(pid))
                        ground[f"{conv}_{band}"] = {
                            "dominant_h": _r(fit["dominant_h"]), "planes": fit["planes"],
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
             "label": args.label, "panos_scored": total,
             "elapsed_s": round(wall, 3), "s_per_pano": round(wall / total, 4) if total else None,
             "model_load_s": round(t_load, 3), "per_split": per_split,
             "what": "da3_calibration_101.py extract: 6 DA3METRIC-LARGE views per pano, point sampling + ground fits",
             "run_id": f"da3-calibration-101:{os.environ.get('SLURM_JOB_ID', 'local')}:"
                       f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             "provider": "depth-anything", "model_id": MODEL_ID, "paid": False,
             "model_revision": MODEL_REVISION, "da3_code_commit": da3_commit,
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


# The labeler input, pinned (review of #203, M2). Only camera_heights.json is read: it is tracked
# in sidewalk-auto-labeler (force-added under the ignored runs/), so `git show` at this commit
# reproduces it. The first version also read runs/laurens/camera_height/groups.csv, which is an
# untracked run output (.gitignore: runs/**) and so in no labeler commit; it is no longer read.
# 2653a49 is the re-issue after sidewalk-auto-labeler#89 (1d8127e), which found no validated
# instrument-B estimator.
LABELER_COMMIT = "2653a49c420465bd792d165ece5681ad6c2ace4a"
LABELER_FILES = {"runs/laurens/camera_heights.json":
                 "74900b62edf5bd9470e17c78c72c6a9c343455ce738086103b1d4252eb34c54d"}


def read_labeler_laurens(labeler_root, commit=LABELER_COMMIT):
    """The labeler's Laurens rig table at a pinned commit, read with `git show` (never the working tree).

    Refuses if the file's sha256 is not the pinned one, so a re-run of ``derive`` cannot silently
    pick up a newer labeler state. Returns the values the tables use plus the commit and hash.
    """
    path = "runs/laurens/camera_heights.json"
    try:
        raw = subprocess.run(["git", "-C", labeler_root, "show", f"{commit}:{path}"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"cannot read {path} at labeler commit {commit} from {labeler_root}: {e}")
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if digest != LABELER_FILES[path]:
        raise SystemExit(f"labeler {path} at {commit} has sha256 {digest}, pinned {LABELER_FILES[path]}")
    hj = json.loads(raw.decode("utf-8"))
    g = hj["groups"]["gopro/max"]
    ib = g.get("instrument_b") or {}
    ev = hj.get("estimator_validation") or {}
    return {"labeler_commit": commit, "files": {path: digest},
            "rig": {"group": "gopro/max", "n_panos": g.get("n_panos"), "applied": g.get("applied"),
                    "height_m": g.get("height_m"), "reason": g.get("reason"),
                    "h_bearing": g.get("h_bearing"), "h_bearing_ci": g.get("h_bearing_ci"),
                    "h_scale": g.get("h_scale"), "h_scale_ci": g.get("h_scale_ci"), "slope": g.get("slope"),
                    "b_validated": ib.get("validated"), "b_h_line": ib.get("h_line"), "b_h_local": ib.get("h_local")},
            "gate_passes": hj.get("gate", {}).get("passes"),
            "estimator_validation": {"validated": ev.get("validated"), "rule": ev.get("rule"),
                                     "line_max_abs_mean_err_m": (ev.get("line") or {}).get("max_abs_mean_err_m"),
                                     "local_max_abs_mean_err_m": (ev.get("local") or {}).get("max_abs_mean_err_m")}}


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
        raw1 = read_raw(RAW_RUN1_DIR, split)
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
                for f in ("h", "n", "tilt_deg", "inlier_share", "resid_med", "n_points", "h_sin_median",
                          "dominant_h", "planes"):
                    prow[f"{key}_{f}"] = fit.get(f)
            g1 = ((raw1.get(pid) or {}).get("ground") or {}).get(f"{DEPTH_CONVENTION}_{PRIMARY_BAND}") or {}
            prow["run1_h"] = g1.get("h")
            prow["run1_inlier_share"] = g1.get("inlier_share")
            prow["run1_n_points"] = g1.get("n_points")
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
                       "google_range_scaled": g["depth_range_scaled"] if g else None,
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


def spearman(xs, ys):
    """Spearman rank correlation (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


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
RECALL_GROUPS = MAPILLARY_SPLITS + ("mapillary_pooled",) + GSV_OTHER_SPLITS
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
    # the same, per capture-vintage group: on Google's 2025-26 rig the ratio falls with range
    # (review of #203, M3), so "a scale" holds pooled, not on every rig
    byg = []
    for g in ("2025-26 rig", "older US vintages", "sao_paulo"):
        for lo, hi in RATIO_BUCKETS:
            b = [q for q in locs if q["group"] == g and lo <= q["google"] < hi]
            if b:
                byg.append({"group": g, "bucket": rbd.bucket_label(lo, hi, "m"), "n": len(b),
                            "median_da3_over_google": _r(st.median(q["da3"] / q["google"] for q in b))})
    t["ratio_by_google_range_by_group"] = byg

    # -- 2. DA3 camera height vs Google camera height, per pano
    hp = height_pairs(panos, GSV_DEPTH_SPLITS)
    hrows = [height_calib_row(s, [q for q in hp if q["split"] == s]) for s in GSV_DEPTH_SPLITS]
    for g in ("2025-26 rig", "older US vintages", "sao_paulo"):
        hrows.append(height_calib_row(g, [q for q in hp if q["group"] == g]))
    hrows.append(height_calib_row("gsv_pooled", hp))
    t["height_calibration"] = hrows

    # -- 2b. three DA3 height readings against Google's, per GSV split: the lowest plane (used),
    # the dominant plane (the first run's reading), and the height implied by the GT points
    # themselves (median of DA3 ray x sin(depression) over the pano's GT points more than 3 deg
    # below the horizon), which never looks at the nadir
    pt_h = {}
    for p in points:
        if p["kind"] == "gt" and p["da3_ray"] and (p["y"] - 0.5) * math.pi > math.radians(3):
            pt_h.setdefault((p["split"], p["pano"]), []).append(p["da3_ray"] * math.sin((p["y"] - 0.5) * math.pi))
    est = []
    for s in GSV_DEPTH_SPLITS + ("gsv_pooled",):
        ps = [p for p in panos if (s == "gsv_pooled" or p["split"] == s) and p["split"] in GSV_DEPTH_SPLITS
              and p["google_height_status"] == "measured" and p["status"] == "ok"]
        row = {"split": s}
        for name, get in (("lowest_plane", lambda p: p[f"{key}_h"] if fit_ok(p) else None),
                          ("dominant_plane", lambda p: p.get(f"{key}_dominant_h") if fit_ok(p) else None),
                          ("gt_points", lambda p: st.median(pt_h[(p["split"], p["pano"])])
                           if pt_h.get((p["split"], p["pano"])) else None)):
            pr = [(get(p), p["google_height_m"]) for p in ps]
            pr = [(a, b) for a, b in pr if a]
            row[name] = {"n_panos": len(pr),
                         "median_ratio": _r(_median_ratio(pr)) if pr else None,
                         "median_abs_log_ratio": _r(st.median(abs(math.log(a / b)) for a, b in pr)) if pr else None,
                         "pearson_r": _r(pearson([b for _, b in pr], [a for a, _ in pr])) if len(pr) >= 3 else None}
        est.append(row)
    t["height_estimators"] = est

    # -- 2c. DA3 against Google's range times the labeler's per-city depth-frame scale
    # (recall_by_depth_112.DEPTH_FRAME_SCALE, from the labeler's bearing-only triangulation)
    fs = []
    for s in GSV_DEPTH_SPLITS:
        pr = []
        for q in locs:
            if q["split"] != s:
                continue
            p = first_at[(q["split"], q["pano"], q["x"], q["y"])]
            if p.get("google_range_scaled"):
                pr.append((p["da3_range"], p["google_range_scaled"]))
        fs.append({"split": s, "n": len(pr), "labeler_scale": rbd.DEPTH_FRAME_SCALE[s],
                   "da3_over_google": next(r["median_ratio"] for r in t["point_calibration"] if r["group"] == s),
                   "da3_over_google_scaled": _r(_median_ratio(pr)) if pr else None})
    t["vs_labeler_frame_scale"] = fs

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

    # -- 4b. the downstream method on GSV (review of #203, M4): recall by distance and the
    # threshold mapping on Google's axis vs the DA3 axis (leave-one-split-out k_point) vs flat,
    # fn-confirmed GT points on measured-ground panos
    gv = []
    for p in points:
        if (p["split"] in GSV_DEPTH_SPLITS and p["kind"] == "gt" and p["fn_confirmed"]
                and p["google_status"] == "measured"):
            kp = t["constants"]["loso"][p["split"]]["k_point"]
            gv.append({"split": p["split"], "hit": p["hit"], "flat_2p5": p["flat_2p5"],
                       "google": p["google_range"],
                       "da3_loso": (p["da3_range"] / kp) if p["da3_range"] else None,
                       "group": rig_group(p["split"], _year(panos_by[(p["split"], p["pano"])]))})
    mv = {"n_gt": len(gv),
          "recall": {a: rbd.recall_table(gv, a, rbd.M_BUCKETS, "m") for a in ("google", "da3_loso", "flat_2p5")},
          "thresholds": []}
    for name, sel in [("gsv_pooled", gv)] + [(sp, [q for q in gv if q["split"] == sp]) for sp in GSV_DEPTH_SPLITS] \
            + [("2025-26 rig", [q for q in gv if q["group"] == "2025-26 rig"])]:
        row = {"group": name, "n": len(sel)}
        for a in ("google", "da3_loso"):
            row[a] = [rbd.window_threshold(sel, th, flat_key="flat_2p5", depth_key=a)["deflated_m"]
                      for th in rbd.PUBLISHED_THRESHOLDS_M]
        mv["thresholds"].append(row)
    t["gsv_method_validation"] = mv

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

    # -- 7. Laurens cross-read: DA3's rig height vs the labeler's (pinned, read-only)
    lm = [p for p in panos if p["split"] == "laurens_mapillary" and fit_ok(p)]
    lg = [p for p in panos if p["split"] == "laurens_gsv" and fit_ok(p)]
    t["laurens_cross_read"] = {
        "labeler_commit": labeler["labeler_commit"], "labeler_files": labeler["files"],
        "labeler_rig": labeler["rig"], "labeler_gate_passes": labeler["gate_passes"],
        "labeler_estimator_validation": labeler["estimator_validation"],
        "da3_n_panos": len(lm),
        "da3_median_h_m": _r(st.median(p[f"{key}_h"] / k_h for p in lm)) if lm else None,
        "laurens_gsv_da3_median_h_m": _r(st.median(p[f"{key}_h"] / k_h for p in lg)) if lg else None,
        "laurens_gsv_n_panos": len(lg)}

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

    # -- 8b. the post-hoc ground-fit change, split into its two parts (review of #203, M1):
    # (a) the plane: the lowest of up to three supported planes instead of the dominant one;
    # (b) the pass rule: inlier share >= 0.25 instead of >= 0.5. Heights here are uncalibrated.
    def run1_ok(p, thr):
        return (p["run1_h"] is not None and (p["run1_inlier_share"] or 0) >= thr
                and (p["run1_n_points"] or 0) >= FIT_MIN_POINTS)

    def same(a, b):
        return a is not None and b is not None and abs(a / b - 1) < SAME_PLANE_REL

    gfc = []
    for name, members in [(s, (s,)) for s in ALL_SPLITS] + [("gsv_google_depth", GSV_DEPTH_SPLITS),
                                                            ("mapillary_all", MAPILLARY_SPLITS)]:
        ps = [p for p in panos if p["split"] in members]
        now = [p for p in ps if fit_ok(p)]
        lowest_is_dominant = [p for p in now if p[f"{key}_h"] == p[f"{key}_dominant_h"]]
        gfc.append({
            "group": name, "panos": len(ps),
            "run1_pass_0p5": sum(1 for p in ps if run1_ok(p, RUN1_FIT_MIN_INLIER_SHARE)),
            "run1_pass_0p25": sum(1 for p in ps if run1_ok(p, FIT_MIN_INLIER_SHARE)),
            "run1_pass_0p25_roof": sum(1 for p in ps if run1_ok(p, FIT_MIN_INLIER_SHARE)
                                       and p["run1_h"] < ROOF_BELOW_M),
            "now_pass": len(now),
            "now_roof": sum(1 for p in now if p[f"{key}_h"] < ROOF_BELOW_M),
            "now_lowest_is_dominant": len(lowest_is_dominant),
            "now_switched_plane": len(now) - len(lowest_is_dominant),
            "now_same_as_run1_passing": sum(1 for p in lowest_is_dominant
                                            if run1_ok(p, RUN1_FIT_MIN_INLIER_SHARE) and same(p[f"{key}_h"], p["run1_h"])),
            "now_threshold_only": sum(1 for p in lowest_is_dominant
                                      if not run1_ok(p, RUN1_FIT_MIN_INLIER_SHARE) and same(p[f"{key}_h"], p["run1_h"])),
            "now_other": None})
        r = gfc[-1]
        r["now_other"] = r["now_lowest_is_dominant"] - r["now_same_as_run1_passing"] - r["now_threshold_only"]
    t["ground_fit_change"] = gfc

    # -- 9. the published DA3 figures of detection_recall_analysis.md ("agree to within
    # 6.5-8.5%, Spearman 0.95 Bend / 0.81 Richmond", "4 Richmond ramps above the horizon"),
    # re-derived under depth_analysis.py's own filters: points with flat < 150 m, ratio over
    # DA3 > 0.5 m, "unusable" = at/above the horizon or flat >= 150 m. That script compared the
    # raw DA3 value (planar z-depth) with the flat horizontal range; the like-for-like column
    # uses this script's horizontal range under the same filter.
    rep_rows = []
    for s_ in ("bend", "richmond"):
        allg = [p for p in points if p["split"] == s_ and p["kind"] == "gt" and p["fn_confirmed"]]
        fin = [p for p in allg if p["flat_2p5"] is not None and p["flat_2p5"] < 150 and p["da3_value"]]
        row = {"split": s_, "n": len(fin),
               "n_unusable_flat": sum(1 for p in allg if p["flat_2p5"] is None or p["flat_2p5"] >= 150),
               "n_at_or_above_horizon": sum(1 for p in allg if p["flat_2p5"] is None)}
        for k in ("da3_value", "da3_range"):
            row[k] = {"median_flat_over_da3": _r(st.median(p["flat_2p5"] / p[k] for p in fin if p[k] > 0.5)),
                      "spearman": _r(spearman([p["flat_2p5"] for p in fin], [p[k] for p in fin]))}
        rep_rows.append(row)
    t["published_reproduction"] = rep_rows

    # -- 10. DA3 plane tilt vs Google ground-plane tilt, GSV measured panos (review N5): the noise
    # floor under the consumer-rig tilts of the camera-height table
    tp = [(p[f"{key}_tilt_deg"], p["google_tilt_deg"]) for p in panos
          if p["split"] in GSV_DEPTH_SPLITS and fit_ok(p) and p["google_height_status"] == "measured"
          and p["google_tilt_deg"] is not None]
    t["tilt_vs_google"] = {"n_panos": len(tp),
                           "da3_median_deg": _r(st.median(a for a, _ in tp)) if tp else None,
                           "google_median_deg": _r(st.median(b for _, b in tp)) if tp else None,
                           "pearson_r": _r(pearson([b for _, b in tp], [a for a, _ in tp])) if len(tp) >= 3 else None,
                           "median_abs_diff_deg": _r(st.median(abs(a - b) for a, b in tp)) if tp else None}
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
    for k in ("z", "ray"):   # fixed order: tables.json is written with sorted keys
        r = t["convention"][k]
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
    L.append("\n## DA3/Google by Google range, per capture-vintage group\n")
    L.append("| group | Google range | n | DA3/Google median |")
    L.append("|---|---|---:|---:|")
    for r in t["ratio_by_google_range_by_group"]:
        L.append(f"| {r['group']} | {r['bucket']} | {r['n']} | {_f(r['median_da3_over_google'])} |")
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
    L.append("\n## Three DA3 height readings against Google's camera height (GSV, measured ground)\n")
    L.append("| split | reading | panos | DA3/Google median | median abs ln ratio | Pearson r |")
    L.append("|---|---|---:|---:|---:|---:|")
    for r in t["height_estimators"]:
        for name in ("lowest_plane", "dominant_plane", "gt_points"):
            e = r[name]
            L.append(f"| {r['split']} | {name} | {e['n_panos']} | {_f(e['median_ratio'])} | "
                     f"{_f(e['median_abs_log_ratio'])} | {_f(e['pearson_r'])} |")
    L.append("\n## DA3 against Google x the labeler's depth-frame scale (per city)\n")
    L.append("| split | locations | labeler scale | DA3/Google | DA3/(Google x scale) |")
    L.append("|---|---:|---:|---:|---:|")
    for r in t["vs_labeler_frame_scale"]:
        L.append(f"| {r['split']} | {r['n']} | {r['labeler_scale']} | {_f(r['da3_over_google'])} | "
                 f"{_f(r['da3_over_google_scaled'])} |")
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
    for s in GSV_DEPTH_SPLITS:
        rows = t["validation"]["by_split"][s]
        for a in AXES:
            r = rows[a]
            if r.get("n"):
                L.append(f"| {s} | {a} | {r['n']} | {_f(r['median_ratio'])} ({_f(r['p10_ratio'], 2)}–{_f(r['p90_ratio'], 2)}) | "
                         f"{_f(r['median_abs_log_ratio'])} | {_f(r['share_within_10pct'])} |")
    mv = t["gsv_method_validation"]
    L.append(f"\n## The downstream method on GSV: recall by distance on Google's axis vs the DA3 axis "
             f"(leave-one-split-out) vs flat ({mv['n_gt']} fn-confirmed GT points, measured ground)\n")
    L.append(rbd._side_by_side([(a, mv["recall"][a]) for a in ("google", "da3_loso", "flat_2p5")]))
    L.append("\n| group | n | Google: 18 m / 25 m become | DA3 (LOSO): 18 m / 25 m become |")
    L.append("|---|---:|---|---|")
    for r in mv["thresholds"]:
        L.append(f"| {r['group']} | {r['n']} | {r['google'][0]} / {r['google'][1]} m | {r['da3_loso'][0]} / {r['da3_loso'][1]} m |")
    tv = t["tilt_vs_google"]
    L.append(f"\nDA3 plane tilt vs Google ground-plane tilt, {tv['n_panos']} GSV panos: median {tv['da3_median_deg']} deg vs "
             f"{tv['google_median_deg']} deg, Pearson r {tv['pearson_r']}, median |difference| {tv['median_abs_diff_deg']} deg.\n")
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
    for name in RECALL_GROUPS:
        tb = t["recall_distance"][name]
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
    r = lc["labeler_rig"]
    ev = lc["labeler_estimator_validation"]
    L.append(f"\n## Laurens cross-read (labeler commit `{lc['labeler_commit']}`, camera_heights.json only)\n")
    L.append(f"GoPro Max rig: DA3 median {lc['da3_median_h_m']} m over {lc['da3_n_panos']} panos "
             f"(laurens_gsv, Google rig, same footprint: {lc['laurens_gsv_da3_median_h_m']} m over "
             f"{lc['laurens_gsv_n_panos']} panos). Labeler: bearing fixed point {r['h_bearing']} {r['h_bearing_ci']}, "
             f"scale identity at 2.6 m {r['h_scale']} {r['h_scale_ci']}, instrument B validated: {r['b_validated']} "
             f"(estimator validation: line max |mean error| {ev['line_max_abs_mean_err_m']} m, local "
             f"{ev['local_max_abs_mean_err_m']} m, validated {ev['validated']}); applied {r['applied']}, height used "
             f"{r['height_m']} m; reason: {r['reason']}\n")
    L.append("\n## Band sensitivity: fitted height, 20–60 deg band / 20–45 deg band\n")
    L.append("| split | panos | median ratio | inlier share 20–45 | inlier share 20–60 |")
    L.append("|---|---:|---:|---:|---:|")
    for r in t["band_sensitivity"]:
        L.append(f"| {r['split']} | {r['n_panos']} | {_f(r['median_ratio_60_over_45'])} | "
                 f"{_f(r['median_inlier_share_45'])} | {_f(r['median_inlier_share_60'])} |")
    L.append("\n## The ground-fit change, split into its two parts (uncalibrated heights)\n")
    L.append("Run 1 = the first extraction's single dominant plane. `now` = lowest supported plane, pass at share >= 0.25. "
             "Of today's passing fits: `same as run 1` = lowest plane is the dominant one, run 1 passed at 0.5, same height "
             "within 3%; `threshold only` = the same, but run 1 failed the 0.5 rule; `switched` = lowest plane is not the "
             "dominant one; `other` = lowest is dominant but its height moved > 3% from run 1 (run-to-run RANSAC/DA3 noise).\n")
    L.append("| group | panos | run 1 pass @0.5 | run 1 plane pass @0.25 (of which < 1.3 m) | now pass (< 1.3 m) | same as run 1 | threshold only | switched plane | other |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in t["ground_fit_change"]:
        L.append(f"| {r['group']} | {r['panos']} | {r['run1_pass_0p5']} | {r['run1_pass_0p25']} ({r['run1_pass_0p25_roof']}) | "
                 f"{r['now_pass']} ({r['now_roof']}) | {r['now_same_as_run1_passing']} | {r['now_threshold_only']} | "
                 f"{r['now_switched_plane']} | {r['now_other']} |")
    L.append("\n## The published DA3 agreement figures, re-derived under depth_analysis.py's filters\n")
    L.append("| split | n (flat < 150 m) | flat/DA3 raw value, median | Spearman | flat/DA3 horizontal range, median | Spearman | unusable on flat (at/above horizon + flat >= 150 m) |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r in t["published_reproduction"]:
        L.append(f"| {r['split']} | {r['n']} | {_f(r['da3_value']['median_flat_over_da3'])} | "
                 f"{_f(r['da3_value']['spearman'])} | {_f(r['da3_range']['median_flat_over_da3'])} | "
                 f"{_f(r['da3_range']['spearman'])} | {r['n_unusable_flat']} ({r['n_at_or_above_horizon']} + "
                 f"{r['n_unusable_flat'] - r['n_at_or_above_horizon']}) |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main

def committed_files():
    raws = [os.path.join(d, f"{s}.jsonl") for d in (RAW_DIR, RAW_RUN1_DIR) for s in ALL_SPLITS]
    return raws + [ROWS_PANOS, ROWS_POINTS, TABLES_JSON, TABLES_MD]


def write_sums():
    lines = []
    for p in committed_files() + [RBD_JSON]:
        lines.append(f"{normalized_sha256(p)}  {os.path.relpath(p, REPO).replace(os.sep, '/')}")
    with open(SUMS, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def check(markdown_out=False):
    """Re-derive rows from raw + bundles + #112, tables from rows; fail on drift."""
    with open(TABLES_JSON, encoding="utf-8") as fh:
        stored = json.load(fh)
    panos, points = derive_rows()
    if read_jsonl(ROWS_PANOS) != panos or read_jsonl(ROWS_POINTS) != points:
        raise SystemExit("rows do not re-derive from the committed raw files, bundles and #112 JSON")
    if stored["labeler_laurens"]["files"] != LABELER_FILES or stored["labeler_laurens"]["labeler_commit"] != LABELER_COMMIT:
        raise SystemExit("tables.json's labeler block is not the pinned labeler commit / file hash")
    fresh = tables(panos, points, stored["labeler_laurens"])
    fresh = json.loads(json.dumps(fresh, sort_keys=True))
    if fresh != stored["tables"]:
        raise SystemExit(f"{TABLES_JSON}: tables do not re-derive from the committed rows")
    with open(TABLES_MD, encoding="utf-8", newline="") as fh:
        if fh.read().replace("\r\n", "\n") != markdown(fresh):
            raise SystemExit(f"{TABLES_MD} is not the markdown of the committed tables")
    with open(SUMS, encoding="utf-8") as fh:
        for line in fh:
            digest, rel = line.split()
            if normalized_sha256(os.path.join(REPO, rel)) != digest:
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
    ap.add_argument("--label", default="da3-calibration-101:extract",
                    help="extract: the usage row's label (the committed rows were relabelled by hand; see the doc §9)")
    ap.add_argument("--allow-other-da3", action="store_true",
                    help="extract: run with DA3 code other than DA3_CODE_COMMIT")
    ap.add_argument("--labeler-root", default=os.environ.get("LABELER_ROOT"),
                    help="derive: a sidewalk-auto-labeler clone (or $LABELER_ROOT); read with git show at the pinned commit")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.markdown)
    if a.stage == "extract":
        if not a.panos_root:
            ap.error("extract needs --panos-root")
        return extract(a)
    if a.stage == "derive":
        panos, points = derive_rows()
        if not a.labeler_root:
            ap.error("derive needs --labeler-root (or $LABELER_ROOT): a sidewalk-auto-labeler clone")
        labeler = read_labeler_laurens(a.labeler_root)
        # sorted keys, exactly as tables.json stores them, so markdown() sees one dict order
        t = json.loads(json.dumps(tables(panos, points, labeler), sort_keys=True))
        write_jsonl(ROWS_PANOS, panos)
        write_jsonl(ROWS_POINTS, points)
        write_json(TABLES_JSON, {"labeler_laurens": labeler, "tables": t,
                                 "inputs": {"recall_by_depth_112.json": normalized_sha256(RBD_JSON)}})
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
