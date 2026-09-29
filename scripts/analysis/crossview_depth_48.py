"""Monocular metric depth at the source GT click, for the #48 cross-view depth arms.

The cross-view harness (``crossview_align_48.py``) places a source-view GT ramp point in another
pano by raycasting the click to flat ground at an assumed camera height, then projecting that
world point into the other view. The depth arms (``crossview_arms/depth_mono.py``) replace only
the first half: the source click's horizontal range comes from a monocular metric depth model
instead of flat ground. This script is the GPU half. It runs a model on the source panos and
writes, per unique source click, everything the arms need; the arms then run on CPU against the
labeler's geometry, like every other geometry arm.

Nothing here reads the answer columns: the frozen pair list is read through the harness (sha256
checked) and the ``ref_*`` columns are dropped before anything else happens.

Per source pano (144 panos, 174 unique clicks):

  * The pano is downscaled to 4096 px wide and rendered into the six perspective views #101 used
    (``equirect_tiling.default_views()``: 90 deg FOV, pitch -30 deg, 1024 px, yaw every 60 deg;
    exact focal 512 px, passed to every model that accepts intrinsics).
  * Each model's output is read as planar (z) depth, and resized to ``GRID`` x ``GRID`` (504, DA3's
    own output size) so that every model is sampled on the same angular grid.
  * **(a) point**: the 7x7 median at the click, in the view where the click is most central
    (``da3_calibration_101.best_view`` / ``_sample``), converted to horizontal range.
  * **(b) local plane**: the 3-D points of that view within ``LOCAL_RADIUS_M`` (horizontal) of
    the click's own 3-D point, and at least ``LOCAL_MIN_DEP_DEG`` below the horizon, are fitted
    with #101's RANSAC plane (``fit_ground_plane``: normal within 20 deg of vertical, 0.10 m
    inliers, least-squares refit). The click ray is intersected with that plane. The fit passes
    at >= ``LOCAL_MIN_POINTS`` points with >= ``LOCAL_MIN_SHARE`` of them on the plane.
  * **(c) camera height**: #101's ground fit over the whole ring (road band 20-45 deg below the
    horizon, the lowest plane holding >= 15% of the band, passing at >= 25% of >= 200 points).
    The arms divide the labeler's known camera height by this to rescale (a) and (b).

All of these constants were fixed before any arm was scored.

Usage (makelab2, see docs/crossview_align_48/depth.md for the env)::

    python scripts/analysis/crossview_depth_48.py extract --model da3 \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs \\
        --src-root /homes/gws/jonf/crossview48/depth/src

writes ``analysis_out/crossview_align_48/depth/<model>.jsonl`` (one row per unique source click)
and appends one ``paid: false`` usage row to ``.../depth/usage_rows.jsonl``.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import crossview_align_48 as H  # noqa: E402

OUT_DIR = os.path.join(H.OUT, "depth")
GRID = 504                    # every model's depth map is resized to this (DA3's native output)
PANO_MAX_EDGE = 4096
LOCAL_RADIUS_M = 2.5          # (b): neighbourhood of the click's 3-D point, horizontal metres
LOCAL_MIN_DEP_DEG = 2.0       # (b): rays at least this far below the pano horizon
LOCAL_STRIDE = 2              # (b): every 2nd pixel of the 504 grid
LOCAL_MIN_POINTS = 50
LOCAL_MIN_SHARE = 0.30
BAND = (20.0, 45.0)           # (c): #101's primary road band
BAND_MIN_SHARE = 0.25         # (c): #101's pass rule
BAND_MIN_POINTS = 200
ND = 5

#: pinned model code and weights (the extract refuses other code unless --allow-other-code)
MODELS = {
    "da3": {"label": "Depth Anything 3 (DA3METRIC-LARGE)",
            "hf": "depth-anything/DA3METRIC-LARGE",
            "hf_revision": "4010e39f3634a45bc60553321fb49fb760bd594e",
            "code": "Depth-Anything-3", "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"},
    "depthpro": {"label": "Depth Pro (apple/DepthPro)", "hf": "apple/DepthPro",
                 "hf_file": "depth_pro.pt", "code": "ml-depth-pro"},
    "unidepth": {"label": "UniDepth v2 ViT-L", "hf": "lpiccinelli/unidepth-v2-vitl14",
                 "code": "UniDepth"},
    "metric3d": {"label": "Metric3D v2 ViT-L", "hf": "JUGGHM/Metric3D",
                 "hf_file": "metric_depth_vit_large_800k.pth", "code": "Metric3D"},
}


def _r(v, nd=ND):
    return None if v is None else round(float(v), nd)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_clicks(pairs):
    """{(city, src_pano): [(src_x, src_y, [pair_id, ...]), ...]} with the answer columns gone."""
    from crossview_arms._registry import ANSWER_KEYS
    out = {}
    for p in pairs:
        p = {k: v for k, v in p.items() if k not in ANSWER_KEYS}
        clicks = out.setdefault((p["city"], p["src_pano"]), {})
        clicks.setdefault((p["src_x"], p["src_y"]), []).append(p["pair_id"])
    return {k: [(x, y, ids) for (x, y), ids in sorted(v.items())] for k, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# geometry on a depth map (numpy)
# --------------------------------------------------------------------------- #


def view_points(depth, view, stride=1):
    """(P, dep_deg) for a z-depth map of ``view``: 3-D points in the pano frame (+y up, +z at
    pano x = 0.5) and each ray's depression below the horizon, both (h, w, ...)."""
    import numpy as np
    from equirect_tiling import _camera_basis
    Hh, Ww = depth.shape
    rows = np.arange(stride // 2, Hh, stride)
    cols = np.arange(stride // 2, Ww, stride)
    u = (cols + 0.5) / Ww
    v = (rows + 0.5) / Hh
    f, r, up = (np.array(t) for t in _camera_basis(view.yaw_deg, view.pitch_deg))
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a = ((2 * u - 1) * th)[None, :, None]
    b = ((1 - 2 * v) * tv)[:, None, None]
    D = f[None, None, :] + a * r[None, None, :] + b * up[None, None, :]
    z = depth[np.ix_(rows, cols)].astype(np.float64)
    P = D * z[..., None]
    Dn = D / np.linalg.norm(D, axis=2, keepdims=True)
    dep = np.degrees(-np.arcsin(np.clip(Dn[..., 1], -1, 1)))
    return P, dep, z


def local_plane(depth, view, click_xyz, seed):
    """(b): RANSAC plane through the view's points near the click's own 3-D point."""
    import numpy as np
    import da3_calibration_101 as C
    P, dep, z = view_points(depth, view, LOCAL_STRIDE)
    d = np.hypot(P[..., 0] - click_xyz[0], P[..., 2] - click_xyz[2])
    keep = (d <= LOCAL_RADIUS_M) & (dep >= LOCAL_MIN_DEP_DEG) & np.isfinite(z) & (z > 0)
    pts = P[keep]
    fit = C.fit_ground_plane(pts, seed)
    ok = (fit["h"] is not None and len(pts) >= LOCAL_MIN_POINTS
          and fit["inlier_share"] >= LOCAL_MIN_SHARE)
    return fit, bool(ok)


# --------------------------------------------------------------------------- #
# models: each returns a z-depth map (H, W) float32 in metres for one 1024x1024 view
# --------------------------------------------------------------------------- #


def _git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_model(name, src_root, dev):
    """(infer(pil_view, focal_px) -> np.ndarray, provenance dict)."""
    import numpy as np
    import torch
    spec = MODELS[name]
    code = os.path.join(src_root, spec["code"])
    prov = {"code": spec["code"], "code_commit": _git_commit(code), "hf": spec["hf"]}

    if name == "da3":
        sys.path.insert(0, os.path.join(code, "stubs"))
        sys.path.insert(0, os.path.join(code, "src"))
        import logging
        logging.disable(logging.INFO)
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(spec["hf"], revision=spec["hf_revision"]).to(dev).eval()
        prov["hf_revision"] = spec["hf_revision"]

        def infer(img, focal):
            W = img.width
            K = np.array([[[focal, 0, W / 2], [0, focal, img.height / 2], [0, 0, 1]]], dtype=np.float32)
            with torch.no_grad():
                pr = model.inference([img], intrinsics=K)
            return np.asarray(pr.depth)[0].astype(np.float32)
        return infer, prov

    if name == "depthpro":
        from huggingface_hub import hf_hub_download, HfApi
        ckpt = hf_hub_download(spec["hf"], spec["hf_file"])
        prov["hf_revision"] = HfApi().model_info(spec["hf"]).sha
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        import dataclasses
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ckpt)
        model, transform = depth_pro.create_model_and_transforms(config=cfg, device=torch.device(dev),
                                                                  precision=torch.half)
        model.eval()

        probes = prov.setdefault("probe_estimated_focal_px", [])

        def infer(img, focal):
            x = transform(np.asarray(img))
            with torch.no_grad():
                pred = model.infer(x, f_px=torch.tensor(float(focal), device=dev))
                if len(probes) < 12:     # what focal would Depth Pro have estimated itself?
                    probes.append(round(float(model.infer(x)["focallength_px"]), 1))
            return pred["depth"].float().cpu().numpy().astype(np.float32)
        return infer, prov

    if name == "unidepth":
        sys.path.insert(0, code)
        from unidepth.models import UniDepthV2
        model = UniDepthV2.from_pretrained(spec["hf"]).to(dev).eval()
        try:
            from huggingface_hub import HfApi
            prov["hf_revision"] = HfApi().model_info(spec["hf"]).sha
        except Exception as e:  # provenance only
            prov["hf_revision_error"] = str(e)
        try:
            from unidepth.utils.camera import Pinhole
        except ImportError:
            Pinhole = None

        def infer(img, focal):
            rgb = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
            K = torch.tensor([[focal, 0, img.width / 2], [0, focal, img.height / 2], [0, 0, 1]],
                             dtype=torch.float32)
            cam = Pinhole(K=K[None]) if Pinhole is not None else K
            with torch.no_grad():
                pred = model.infer(rgb.to(dev), cam.to(dev) if hasattr(cam, "to") else cam)
            return pred["depth"].squeeze().float().cpu().numpy().astype(np.float32)
        return infer, prov

    if name == "metric3d":
        from huggingface_hub import HfApi
        # mmcv stub (re-exports mmengine.Config), see docs/crossview_align_48/depth.md
        sys.path.insert(0, os.path.join(src_root, "stubs"))
        model = torch.hub.load(code, "metric3d_vit_large", pretrain=True, source="local").to(dev).eval()
        try:
            prov["hf_revision"] = HfApi().model_info(spec["hf"]).sha
        except Exception as e:
            prov["hf_revision_error"] = str(e)
        import cv2
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        in_h, in_w = 616, 1064            # the ViT models' input size (Metric3D hubconf example)

        def infer(img, focal):
            rgb = np.asarray(img)
            h, w = rgb.shape[:2]
            s = min(in_h / h, in_w / w)
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
            fs = focal * s
            ph, pw = in_h - rgb.shape[0], in_w - rgb.shape[1]
            pad = [ph // 2, ph - ph // 2, pw // 2, pw - pw // 2]
            rgb = cv2.copyMakeBorder(rgb, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT,
                                     value=[123.675, 116.28, 103.53])
            t = (torch.from_numpy(rgb.transpose(2, 0, 1)).float() - mean) / std
            with torch.no_grad():
                d, _, _ = model.inference({"input": t[None].to(dev)})
            d = d.squeeze()
            d = d[pad[0]:d.shape[0] - pad[1], pad[2]:d.shape[1] - pad[3]]
            d = torch.nn.functional.interpolate(d[None, None], (h, w), mode="bilinear").squeeze()
            return (d * (fs / 1000.0)).float().cpu().numpy().astype(np.float32)
        return infer, prov

    raise SystemExit(f"unknown model {name!r}")


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def extract(args):
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    import da3_calibration_101 as C
    from equirect_tiling import default_views, equirect_to_perspective
    Image.MAX_IMAGE_PIXELS = None

    pairs = H.read_frozen_pairs()
    jobs = source_clicks(pairs)
    if args.limit:
        jobs = dict(list(jobs.items())[:args.limit])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    t_load = time.time()
    infer, prov = load_model(args.model, args.src_root, dev)
    t_load = time.time() - t_load
    spec = MODELS[args.model]
    if "commit" in spec and prov["code_commit"] != spec["commit"] and not args.allow_other_code:
        raise SystemExit(f"{spec['code']} at {prov['code_commit']}, pinned {spec['commit']}")
    views = default_views()
    focal = (views[0].width / 2) / math.tan(math.radians(views[0].fov_h_deg / 2))
    az_half = 180.0 / len(views)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}.jsonl")
    print(f"{args.model} on {gpu}; load {t_load:.1f} s; {len(jobs)} source panos", flush=True)

    rows, t0, t_model = [], time.time(), 0.0
    for (city, pid), clicks in jobs.items():
        path = os.path.join(args.archive_root, city, "panos", f"{pid}.jpg")
        base = {"city": city, "src_pano": pid, "model": args.model}
        if not os.path.exists(path):
            rows += [{**base, "src_x": x, "src_y": y, "pair_ids": ids, "status": "missing_pano"}
                     for x, y, ids in clicks]
            continue
        sha = sha256_of(path)
        img = Image.open(path).convert("RGB")
        if max(img.size) > PANO_MAX_EDGE:
            s = PANO_MAX_EDGE / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
        depths, native = [], None
        tm = time.time()
        for vw in views:
            d = infer(equirect_to_perspective(img, vw), focal)
            native = list(d.shape)
            depths.append(cv2.resize(d, (GRID, GRID), interpolation=cv2.INTER_LINEAR))
        if dev == "cuda":
            torch.cuda.synchronize()
        t_model += time.time() - tm
        img.close()
        # (c) the ring's ground plane -> this model's camera height
        P = np.concatenate([C.band_points(d, vw, BAND[0], BAND[1], "z", az_half=az_half)
                            for d, vw in zip(depths, views)])
        g = C.fit_lowest_plane(P, C.pano_seed(pid))
        g_ok = (g["h"] is not None and g["n_points"] >= BAND_MIN_POINTS
                and g["inlier_share"] >= BAND_MIN_SHARE)
        ground = {"h": _r(g["h"]), "n": [_r(c, 6) for c in g["n"]] if g["n"] else None,
                  "tilt_deg": _r(g["tilt_deg"]), "inlier_share": _r(g["inlier_share"]),
                  "n_points": g["n_points"], "planes": g["planes"], "ok": bool(g_ok)}
        for x, y, ids in clicks:
            row = {**base, "src_x": x, "src_y": y, "pair_ids": ids, "sha256": sha,
                   "native_hw": native, "ground": ground}
            b = C.best_view(x, y, views)
            if b is None:
                rows.append({**row, "status": "no_view"})
                continue
            _, vi, (u, v) = b
            val = C._sample(depths[vi], u, v)
            ray = C.ray_from_value(val, views[vi], u, v, "z")
            rng = C.horizontal_range(ray, y)
            # the click's own 3-D point, pano frame
            dx, dy, dz = C._dir_from_equirect(x, y)
            click_xyz = (ray * dx, ray * dy, ray * dz)
            fit, ok = local_plane(depths[vi], views[vi], click_xyz, C.pano_seed(pid) + 7)
            prange = C.plane_range(fit["n"], fit["h"], x, y) if fit["h"] is not None else None
            row.update(status="ok", view=vi, u=_r(u, 6), v=_r(v, 6), z_click=_r(val),
                       ray_m=_r(ray), range_point_m=_r(rng),
                       local={"h": _r(fit["h"]), "n": [_r(c, 6) for c in fit["n"]] if fit["n"] else None,
                              "tilt_deg": _r(fit["tilt_deg"]), "inlier_share": _r(fit["inlier_share"]),
                              "n_points": fit["n_points"], "ok": ok},
                       range_plane_m=_r(prange) if ok else None)
            rows.append(row)
        print(f"  {city} {pid}: {len(clicks)} click(s), {(time.time() - t0) / max(1, len(rows)):.2f} s/click",
              flush=True)
    wall = time.time() - t0
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for r in sorted(rows, key=lambda r: (r["city"], r["src_pano"], r["src_x"], r["src_y"])):
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    meta = {"model": args.model, "label": spec["label"], "prov": prov, "gpu": gpu, "host": os.uname().nodename
            if hasattr(os, "uname") else "unknown", "model_load_s": round(t_load, 2),
            "elapsed_s": round(wall, 2), "model_s": round(t_model, 2), "panos": len(jobs),
            "clicks": len(rows), "pairs_sha256": H.PAIRS_SHA256,
            "constants": {"GRID": GRID, "PANO_MAX_EDGE": PANO_MAX_EDGE, "LOCAL_RADIUS_M": LOCAL_RADIUS_M,
                          "LOCAL_MIN_DEP_DEG": LOCAL_MIN_DEP_DEG, "LOCAL_STRIDE": LOCAL_STRIDE,
                          "LOCAL_MIN_POINTS": LOCAL_MIN_POINTS, "LOCAL_MIN_SHARE": LOCAL_MIN_SHARE,
                          "BAND": BAND, "BAND_MIN_SHARE": BAND_MIN_SHARE,
                          "BAND_MIN_POINTS": BAND_MIN_POINTS, "views": "default_views() 6 x 90deg, pitch -30"},
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    H.write_json(os.path.join(args.out_dir, f"{args.model}.meta.json"), meta)
    usage = {"ts": meta["ts"], "bundle": "crossview_align_48/pairs.csv (source panos)",
             "label": f"crossview-depth-48:extract-{args.model}", "panos_scored": len(jobs),
             "elapsed_s": round(wall, 3), "s_per_pano": round(wall / max(1, len(jobs)), 4),
             "model_load_s": round(t_load, 3), "gpu_hours": round((wall + t_load) / 3600.0, 4),
             "what": f"crossview_depth_48.py extract --model {args.model}: 6 views per source pano, "
                     "depth at the click, local plane, ring ground fit",
             "run_id": f"crossview-depth-48:{args.model}:{meta['host']}:{meta['ts']}",
             "provider": args.model, "model_id": spec["hf"], "paid": False,
             "model_revision": prov.get("hf_revision"), "code_commit": prov.get("code_commit"),
             "hardware": {"host": meta["host"], "gpus": [gpu]}, "status": "ok",
             "est_cost_usd": 0.0, "pricing": None, "script": "scripts/analysis/crossview_depth_48.py",
             "issue": 48}
    with open(os.path.join(args.out_dir, "usage_rows.jsonl"), "a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(usage) + "\n")
    print(f"done: {len(rows)} clicks from {len(jobs)} panos in {wall:.1f} s (model {t_model:.1f} s) -> {out_path}",
          flush=True)


# --------------------------------------------------------------------------- #
# summarize (CPU, committed inputs only)
# --------------------------------------------------------------------------- #

SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")
BASE_ARMS = ("proj_height_auto",)


def _range_check(pairs, preds, key):
    """Per unique GSV source click with Google range: the arm's range vs Google's."""
    import numpy as np
    seen, rat = set(), []
    for p in pairs:
        r = preds.get(p["pair_id"]) or {}
        g, d = r.get("range_google_m"), r.get(key)
        k = (p["src_pano"], p["src_x"], p["src_y"])
        if p["imagery"] != "gsv" or k in seen or not g or not d or g <= 0 or d <= 0:
            continue
        seen.add(k)
        rat.append(d / g)
    if not rat:
        return None
    a = np.array(rat)
    return {"n_clicks": len(a), "median_ratio": float(np.median(a)),
            "p10_p90": [float(np.percentile(a, 10)), float(np.percentile(a, 90))],
            "median_abs_ln": float(np.median(np.abs(np.log(a)))),
            "within_10pct": float(np.mean(np.abs(a - 1) <= 0.10))}


def summarize_cmd(args):
    """Scores the depth arms against the projection AND against proj_height_auto (paired, same
    ramp bootstrap as the harness), a composite 'depth where it applies, else auto', and the
    range check against Google's depth at the click. Writes depth/summary.json."""
    import numpy as np
    names = args.arms.split(",") if args.arms else sorted(
        f[:-6] for f in os.listdir(os.path.join(H.OUT, "predictions"))
        if f.startswith("mono_") and f.endswith(".jsonl"))
    pairs = H.read_frozen_pairs()
    proj = H.arm_errors(pairs, None)
    auto_preds = H.read_predictions("proj_height_auto")
    auto = H.arm_errors(pairs, auto_preds)
    strata = {k: v for k, v in H.strata_of(pairs).items()
              if k == "all" or k.startswith("imagery=") or k.startswith("range=")}
    out = {"arms": {}, "config": {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT,
                                  "seed": H.SEED, "baselines": ["projection", "proj_height_auto"]}}
    # flat 'auto' range vs Google, from any depth arm's diagnostics (same for all)
    any_preds = H.read_predictions(names[0]) if names else {}
    out["range_check_flat_auto"] = _range_check(pairs, any_preds, "range_flat_auto_m")
    for n in names:
        preds = H.read_predictions(n)
        e = H.arm_errors(pairs, preds)
        comp = [(c[0], c[1], c[2], False) for c in (a if x[3] else x for x, a in zip(e, auto))]
        row = {}
        for k, idx in strata.items():
            row[k] = {"vs_projection": H.summarize(pairs, e, idx, proj),
                      "vs_auto": H.summarize(pairs, e, idx, auto),
                      "composite_else_auto_vs_auto": H.summarize(pairs, comp, idx, auto)}
        row["range_check"] = _range_check(pairs, preds, "range_used_m")
        out["arms"][n] = row
    out["auto_vs_projection"] = {k: H.summarize(pairs, auto, idx, proj) for k, idx in strata.items()}
    H.write_json(args.out, out)

    def ci(c):
        return f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "-"
    print("| arm | fallback | all: median ° [CI] | applied n: arm vs auto °, paired gain vs auto [CI] "
          "| vs projection, applied: gain [CI] | composite (else auto) median °, gain vs auto [CI] "
          "| range/Google median (within 10%) |")
    print("|---|---|---|---|---|---|---|")
    a0 = out["auto_vs_projection"]["all"]
    print(f"| proj_height_auto | 0 | {a0['median_deg']:.2f} {ci(a0['median_ci'])} | - | - | - | "
          + (lambda r: f"{r['median_ratio']:.3f} ({r['within_10pct']:.2f})" if r else "-")(
              out["range_check_flat_auto"]) + " |")
    for n, row in out["arms"].items():
        va, vp, cm = row["all"]["vs_auto"], row["all"]["vs_projection"], row["all"]["composite_else_auto_vs_auto"]
        ao, po = va["aligned_only"], vp["aligned_only"]
        rc = row["range_check"]
        print(f"| {n} | {va['fallback_rate']:.2f} | {vp['median_deg']:.2f} {ci(vp['median_ci'])} | "
              f"{ao['n_pairs']}: {ao['median_deg'] or float('nan'):.2f} vs {ao['projection_median_deg'] or float('nan'):.2f}, "
              f"{ao['median_gain_deg'] or float('nan'):.2f} {ci(ao['median_gain_ci'])} | "
              f"{po['median_gain_deg'] or float('nan'):.2f} {ci(po['median_gain_ci'])} | "
              f"{cm['median_deg']:.2f}, {cm['median_gain_deg']:.2f} {ci(cm['median_gain_ci'])} | "
              + (f"{rc['median_ratio']:.3f} ({rc['within_10pct']:.2f})" if rc else "-") + " |")
    print(f"-> {args.out}")


def flatcheck(args):
    """Instrument check for the arms' placement path: feed them a synthetic depth row whose
    range is the flat-ground range at the 'auto' height, and require that the point and hcal
    arms reproduce the committed proj_height_auto predictions. Needs the labeler inputs."""
    import crossview_arms.depth_mono as M
    registry = H.load_arms()
    pairs = H.read_frozen_pairs()
    ctx = H.Context(args, pairs)
    rows = {}
    for p in pairs:
        h = M._h_auto(ctx, p)
        dep = (p["src_y"] - 0.5) * math.pi
        rows[p["pair_id"]] = {"status": "ok", "range_point_m": h / math.tan(dep), "range_plane_m": None,
                              "ground": {"h": h, "ok": True}, "local": {"ok": False}}
    ctx.cache[("mono_rows", "da3")] = rows
    auto = H.read_predictions("proj_height_auto")
    from crossview_arms._registry import ANSWER_KEYS
    worst, fb = 0.0, 0
    for p in pairs:
        visible = {k: v for k, v in p.items() if k not in ANSWER_KEYS}
        a = auto[p["pair_id"]]
        for name in ("mono_da3_point", "mono_da3_hcal"):
            o = registry[name].fn(visible, ctx)
            if o["x"] is None:
                fb += 1
                continue
            worst = max(worst, float(H.angular_error_deg(o["x"], o["y"], a["x"], a["y"])))
    print(f"flatcheck: max {worst:.6f} deg from proj_height_auto over {2 * len(pairs)} placements, "
          f"{fb} fallbacks")
    if worst > 0.01 or fb:
        raise SystemExit("flatcheck FAILED")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--archive-root", required=True, help="<root>/<city>/panos/<pano>.jpg")
    p.add_argument("--src-root", required=True, help="directory holding the model code clones")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--limit", type=int, default=0, help="first N source panos (smoke test)")
    p.add_argument("--allow-other-code", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--arms", help="comma-separated; default: every predictions/mono_*.jsonl")
    s.add_argument("--out", default=SUMMARY_JSON)
    fc = sub.add_parser("flatcheck")
    fc.add_argument("--labeler-root", required=True)
    fc.add_argument("--runs-root")
    fc.add_argument("--results-root")
    args = ap.parse_args(argv)
    if args.cmd == "flatcheck":
        args.views, args.extra = None, []
        flatcheck(args)
    elif args.cmd == "extract":
        extract(args)
    elif args.cmd == "summarize":
        summarize_cmd(args)


if __name__ == "__main__":
    main()
