"""Metric depth for every benchmark GT ramp, via Depth-Anything-V2 on rectilinear views.

The flat-ground distance proxy (d = camera_height / tan(depression)) demonstrably
breaks on Mapillary: unleveled consumer rigs / hilly terrain put some GT ramps at or
ABOVE the horizon, where it returns infinity. Monocular depth fixes that, but depth
models fail on raw equirectangular input -- so reuse the perspective reprojection:
for each GT point pick the view where it sits most centrally (least lens distortion),
render it, run metric depth, and sample at the point.

Writes gt_depth.json: one record per GT ramp with both the geometric and the depth
distance, so the two can be cross-validated (Bend/GSV has a level ~2.5 m mast, where
the geometric estimate SHOULD be reliable -- that's the sanity check).

Usage: depth_extract.py [n_panos_per_city]
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import numpy as np
import torch
from rampnet.detection_eval import (
    build_ground_truth, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
from compare import load_bundle, DetectionCache, cache_key
from detectors import build_detector, load_pano_image
from equirect_tiling import default_views, equirect_to_perspective, equirect_point_to_perspective

CAM_H, RAMP_W = 2.5, 1.2
PX_PER_RAD = 4096.0 / (2 * math.pi)
R = math.sqrt(radius_sq_for())
PRO = "gemini-3.1-pro-preview"
MODEL_ID = 'depth-anything/DA3METRIC-LARGE'
OUT_FILE = os.path.join(OUT, "gt_depth_da3.json")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


class Args:
    gemini_model = PRO; qwen_model = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_coord_space = "auto"; tiling = "perspective"


def geom_dist(y):
    dep = (y - 0.5) * math.pi
    return CAM_H / math.tan(dep) if dep > 1e-4 else float("inf")


def dist2(p, q):
    return math.hypot((p[0] - q[0]) * PANO_SCALE_X, (p[1] - q[1]) * PANO_SCALE_Y)


def matched_gt(preds, gt_points):
    confs = [_confidence(p) for p in preds]
    order = (sorted(range(len(preds)), key=lambda i: confs[i] if confs[i] is not None else -1e9,
                    reverse=True) if any(c is not None for c in confs) else range(len(preds)))
    claimed, hit = [False] * len(gt_points), set()
    for i in order:
        p = _xy(preds[i]); bk, bd = -1, R
        for k, g in enumerate(gt_points):
            if claimed[k]:
                continue
            d = dist2(p, g)
            if d < bd:
                bd, bk = d, k
        if bk >= 0:
            claimed[bk] = True; hit.add(bk)
    return hit


def best_view(x, y, views):
    """View where the point is most central (least distortion); None if unseen."""
    best = None
    for vi, v in enumerate(views):
        uv = equirect_point_to_perspective(x, y, v)
        if uv is None:
            continue
        c = max(abs(uv[0] - 0.5), abs(uv[1] - 0.5))
        if best is None or c < best[0]:
            best = (c, vi, uv)
    return best


def sample_depth(dmap, u, v, half=3):
    H, W = dmap.shape
    r, c = int(np.clip(v * H, 0, H - 1)), int(np.clip(u * W, 0, W - 1))
    patch = dmap[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1]
    return float(np.median(patch)) if patch.size else float(dmap[r, c])


def main():
    sys.path.insert(0, os.path.join(DA3_SRC or "", "..", "stubs"))
    sys.path.insert(0, DA3_SRC)
    import logging; logging.disable(logging.INFO)
    from depth_anything_3.api import DepthAnything3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dmodel = DepthAnything3.from_pretrained(MODEL_ID).to(dev).eval()
    views = default_views()
    _W = views[0].width
    _F = (_W / 2) / math.tan(math.radians(views[0].fov_h_deg / 2))
    K = np.array([[[_F, 0, _W / 2], [0, _F, _W / 2], [0, 0, 1]]], dtype=np.float32)
    print(f"DA3 metric; known focal {_F:.0f}px", flush=True)
    print(f"depth model on {dev}; {len(views)} candidate views", flush=True)

    rows = []
    for city in ("richmond", "bend"):
        records, verdicts, panos_dir = load_bundle(os.path.join(REPO, "benchmark", city))
        label, gem = build_detector("gemini", PRO, records, Args())
        sig, cache = gem.signature(), DetectionCache(os.path.join(REPO, ".model_cache"))
        items = list(verdicts.items())[:LIMIT] if LIMIT else list(verdicts.items())
        for n, (pid, entry) in enumerate(items, 1):
            gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                    entry["missed"], entry["no_missed"])
            if not gt.fn_confirmed or not gt.gt_points:
                continue
            rp = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                  for d in records[pid]["detections"]]
            pro = cache.get(cache_key(label, sig, city, pid)) or []
            mr, mp = matched_gt(rp, gt.gt_points), matched_gt(pro, gt.gt_points)

            need = {}
            for i, g in enumerate(gt.gt_points):
                b = best_view(g[0], g[1], views)
                if b:
                    need.setdefault(b[1], []).append((i, b[2]))
            path = os.path.join(panos_dir, f"{pid}.jpg")
            if not os.path.exists(path):
                continue
            pano = load_pano_image(path, 4096)
            depths = {}
            for vi, pts in need.items():
                vimg = equirect_to_perspective(pano, views[vi])
                with torch.no_grad():
                    pr = dmodel.inference([vimg], intrinsics=K)
                dm = np.asarray(pr.depth)[0]
                for i, uv in pts:
                    depths[i] = sample_depth(dm, uv[0], uv[1])
            pano.close()
            for i, g in enumerate(gt.gt_points):
                rows.append({"city": city, "pid": pid, "x": g[0], "y": g[1],
                             "hit": i in mr, "pro": i in mp,
                             "geom": geom_dist(g[1]), "depth": depths.get(i)})
            if n % 20 == 0:
                print(f"  {city}: {n}/{len(items)}", flush=True)
        print(f"{city} done", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f)
    got = sum(1 for r in rows if r["depth"] is not None)
    print(f"\nwrote {len(rows)} GT ramps ({got} with depth) -> {OUT}")


if __name__ == "__main__":
    main()
