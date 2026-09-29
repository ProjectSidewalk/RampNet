"""Feed-forward 3D arms (#48): MASt3R / DUSt3R, VGGT and MapAnything.

Each arm reconstructs the source view and the pair's other view -- alone ("pair") or with
the corner's other captures ("corner", up to ``CORNER_VIEWS`` views, nearest the ramp
first) -- and transfers the click by **lift and project**: the source click (the source
view's centre) is lifted to a 3D point with the model's own geometry, then projected into
the other view with the model's own camera for it, and mapped back to the equirect.

* ``mast3r_pair`` / ``dust3r_pair`` -- the pairwise pointmaps (both in the source camera's
  frame). The other camera is solved by PnP (RANSAC) from its pointmap and its known
  pinhole intrinsics; the click's 3D point is then projected with it.
* ``vggt_pair`` / ``vggt_corner`` -- VGGT-1B's cameras (extrinsics + intrinsics) and depth.
  The click is unprojected from the source depth map with the source camera and projected
  with the other camera.
* ``mapa_k_pair`` / ``mapa_k_corner`` -- MapAnything, given the known intrinsics; its
  metric pointmap and cameras.
* ``mapa_posed_corner`` -- MapAnything given the known intrinsics AND the pose priors
  (position, heading, 'auto' height, flat) for every view, metric. The lifted point is
  projected with MapAnything's output camera for the other view.
* ``mapa_posed_depthonly`` -- the same run, but the lifted point is projected with the
  other view's *prior* camera: only MapAnything's depth at the click is used (the ground
  geometry half of the question).

Views are the harness's 1024 x 768, 75 deg views, resized to the model's input size
without cropping (the pixel scale is applied per axis), so the click and the intrinsics map
exactly. The models' 3D is only used through their own cameras, so no scale is needed
except in the posed runs, which are metric by construction.
"""
import math
import os
import time

import numpy as np

import crossview_align_48 as H
from crossview_arms import _mv3d as M
from crossview_arms._registry import register

CORNER_VIEWS = 12
CLICK = (H.VIEW_W / 2.0, H.VIEW_H / 2.0)

MODEL_IDS = {
    "vggt": "facebook/VGGT-1B",
    "mapanything": "facebook/map-anything",
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
}
SIZES = {"vggt": (518, 392), "mapanything": (518, 392), "mast3r": (512, 384),
         "dust3r": (512, 384)}


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def _device(ctx):
    import torch
    return "cuda" if torch.cuda.is_available() and not getattr(ctx.args, "cpu", False) else "cpu"


def _model(ctx, name):
    key = ("ff3d_model", name)
    if key in ctx.cache:
        return ctx.cache[key]
    for k in [k for k in ctx.cache if isinstance(k, tuple) and k[0] == "ff3d_model"]:
        del ctx.cache[k]                            # one big model on the GPU at a time
    import torch
    torch.cuda.empty_cache()
    dev = _device(ctx)
    if name == "vggt":
        from vggt.models.vggt import VGGT
        m = VGGT.from_pretrained(MODEL_IDS[name])
    elif name == "mapanything":
        from mapanything.models import MapAnything
        m = MapAnything.from_pretrained(MODEL_IDS[name])
    elif name == "mast3r":
        from mast3r.model import AsymmetricMASt3R
        m = AsymmetricMASt3R.from_pretrained(MODEL_IDS[name])
    elif name == "dust3r":
        from dust3r.model import AsymmetricCroCo3DStereo
        m = AsymmetricCroCo3DStereo.from_pretrained(MODEL_IDS[name])
    else:
        raise ValueError(name)
    m = m.to(dev).eval()
    ctx.cache[key] = m
    return m


def _rgb(ctx, view, size):
    """RGB float array in [0, 1] at ``size`` (W, H), no crop."""
    import cv2
    img = M.read_view(ctx, view)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _scale(size):
    return size[0] / H.VIEW_W, size[1] / H.VIEW_H


def _K_at(size):
    sx, sy = _scale(size)
    K = M.intrinsics().copy()
    K[0] *= sx
    K[1] *= sy
    return K


def _bilinear(arr, u, v):
    """arr[H, W, C] sampled at pixel-centre coordinates (u, v) (pixel i centred at i+0.5)."""
    h, w = arr.shape[:2]
    x, y = u - 0.5, v - 0.5
    x0, y0 = int(np.clip(math.floor(x), 0, w - 2)), int(np.clip(math.floor(y), 0, h - 2))
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    a = arr[y0, x0] * (1 - fx) + arr[y0, x0 + 1] * fx
    b = arr[y0 + 1, x0] * (1 - fx) + arr[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def _pixels(size):
    w, h = size
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return uu, vv


def _project(T_wc, K, X):
    """Pixel of world point X for a camera with cam-to-world T_wc (4x4) and K."""
    Rm, t = T_wc[:3, :3], T_wc[:3, 3]
    p = Rm.T @ (X - t)
    if p[2] <= 1e-6:
        return None
    q = K @ (p / p[2])
    return float(q[0]), float(q[1])


def _to_pano(view, uv_model, size):
    sx, sy = _scale(size)
    u, v = uv_model[0] / sx, uv_model[1] / sy
    x, y = M.view_pixel_to_pano(view, u, v)
    return {"x": x, "y": y, "u": u, "v": v}


def _fit_K(rays, size):
    """Pinhole intrinsics from per-pixel camera-frame ray directions (H, W, 3), by least
    squares on u = fx * rx / rz + cx (and v likewise)."""
    uu, vv = _pixels(size)
    r = rays.reshape(-1, 3)
    ok = r[:, 2] > 1e-3
    a, b = r[ok, 0] / r[ok, 2], r[ok, 1] / r[ok, 2]
    fx, cx = np.linalg.lstsq(np.column_stack([a, np.ones_like(a)]), uu.ravel()[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack([b, np.ones_like(b)]), vv.ravel()[ok], rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _views_for(ctx, pair, corner_mode):
    corner = M.corner_for(ctx, pair)
    return M.select_views(corner, pair, CORNER_VIEWS if corner_mode else 2)


def _rel_diag(T_src, T_oth, views):
    from crossview_arms.sfm import rel_pose_agreement
    return rel_pose_agreement(T_src[:3, :3], T_src[:3, 3], T_oth[:3, :3], T_oth[:3, 3],
                              views[0], views[1])


def _finish(out, X, T_oth, K_oth, oth_view, size):
    uv = _project(T_oth, K_oth, X)
    if uv is None:
        out["reason"] = "behind_other"
        return out
    out.update(_to_pano(oth_view, uv, size))
    return out


# --------------------------------------------------------------------------- #
# MASt3R / DUSt3R (pairwise)
# --------------------------------------------------------------------------- #


def _croco_pair(ctx, pair, name):
    import cv2
    import torch
    from dust3r.inference import inference
    size = SIZES[name]
    views = _views_for(ctx, pair, False)
    ims = []
    for i, v in enumerate(views):
        t = torch.from_numpy(_rgb(ctx, v, size)).permute(2, 0, 1)[None] * 2.0 - 1.0
        ims.append(dict(img=t, true_shape=np.int32([[size[1], size[0]]]), idx=i, instance=str(i)))
    model = _model(ctx, name)
    t0 = time.time()
    with torch.no_grad():
        res = inference([(ims[0], ims[1])], model, _device(ctx), batch_size=1, verbose=False)
    p1 = res["pred1"]["pts3d"][0].float().cpu().numpy()
    p2 = res["pred2"]["pts3d_in_other_view"][0].float().cpu().numpy()
    c1 = res["pred1"]["conf"][0].float().cpu().numpy()
    c2 = res["pred2"]["conf"][0].float().cpu().numpy()
    out = {"x": None, "y": None, "infer_s": round(time.time() - t0, 3)}
    sx, sy = _scale(size)
    X = _bilinear(p1, CLICK[0] * sx, CLICK[1] * sy)
    out["click_conf"] = float(_bilinear(c1[..., None], CLICK[0] * sx, CLICK[1] * sy)[0])
    out["click_depth"] = float(X[2])
    # the other camera, by PnP from its pointmap (in the source camera's frame)
    uu, vv = _pixels(size)
    keep = c2 > np.percentile(c2, 50)
    obj = p2[keep].astype(np.float64)
    img = np.column_stack([uu[keep], vv[keep]]).astype(np.float64)
    K2 = _K_at(size)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K2, None, iterationsCount=200,
                                             reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 50:
        out["reason"] = "pnp_failed"
        return out
    Rw2c, _ = cv2.Rodrigues(rvec)
    T_oth = np.eye(4)
    T_oth[:3, :3] = Rw2c.T
    T_oth[:3, 3] = (-Rw2c.T @ tvec).ravel()
    out["pnp_inliers"] = int(len(inl))
    out.update(_rel_diag(np.eye(4), T_oth, views))
    return _finish(out, X, T_oth, K2, views[1], size)


@register("mast3r_pair", needs=("views",),
          config={"model": MODEL_IDS["mast3r"], "input": SIZES["mast3r"],
                  "other_camera": "PnP-RANSAC on its pointmap, known K, top-50% confidence"},
          description="MASt3R pairwise pointmap; click lifted in the source frame, other "
                      "camera by PnP")
def mast3r_pair(pair, ctx):
    return _croco_pair(ctx, pair, "mast3r")


@register("dust3r_pair", needs=("views",),
          config={"model": MODEL_IDS["dust3r"], "input": SIZES["dust3r"],
                  "other_camera": "PnP-RANSAC on its pointmap, known K, top-50% confidence"},
          description="DUSt3R pairwise pointmap; click lifted in the source frame, other "
                      "camera by PnP")
def dust3r_pair(pair, ctx):
    return _croco_pair(ctx, pair, "dust3r")


# --------------------------------------------------------------------------- #
# VGGT
# --------------------------------------------------------------------------- #


def _vggt(ctx, pair, corner_mode):
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    size = SIZES["vggt"]
    views = _views_for(ctx, pair, corner_mode)
    imgs = torch.stack([torch.from_numpy(_rgb(ctx, v, size)).permute(2, 0, 1) for v in views])
    model = _model(ctx, "vggt")
    dev = _device(ctx)
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=dev == "cuda"):
        pred = model(imgs.to(dev)[None])
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (size[1], size[0]))
    ext = ext[0].float().cpu().numpy()        # (S, 3, 4) world-to-camera, OpenCV
    intr = intr[0].float().cpu().numpy()
    depth = pred["depth"][0, 0, ..., 0].float().cpu().numpy()
    dconf = pred["depth_conf"][0, 0].float().cpu().numpy()
    out = {"x": None, "y": None, "n_views": len(views), "infer_s": round(time.time() - t0, 3)}

    def T(i):
        Tm = np.eye(4)
        Rm, t = ext[i][:, :3], ext[i][:, 3]
        Tm[:3, :3], Tm[:3, 3] = Rm.T, -Rm.T @ t
        return Tm

    sx, sy = _scale(size)
    u, v = CLICK[0] * sx, CLICK[1] * sy
    d = float(_bilinear(depth[..., None], u, v)[0])
    out["click_depth"] = d
    out["click_conf"] = float(_bilinear(dconf[..., None], u, v)[0])
    if not d > 0:
        out["reason"] = "no_depth"
        return out
    p_cam = d * np.linalg.solve(intr[0], np.array([u, v, 1.0]))
    T0 = T(0)
    X = T0[:3, :3] @ p_cam + T0[:3, 3]
    out.update(_rel_diag(T0, T(1), views))
    return _finish(out, X, T(1), intr[1], views[1], size)


VGGT_CONFIG = {"model": MODEL_IDS["vggt"], "input": SIZES["vggt"], "dtype": "bf16",
               "lift": "depth head + predicted camera (not the point head)"}


@register("vggt_pair", needs=("views",), config={**VGGT_CONFIG, "views": 2},
          description="VGGT on the source and other view; depth lift, predicted cameras")
def vggt_pair(pair, ctx):
    return _vggt(ctx, pair, False)


@register("vggt_corner", needs=("views",), config={**VGGT_CONFIG, "views": CORNER_VIEWS},
          description=f"VGGT on up to {CORNER_VIEWS} captures of the corner")
def vggt_corner(pair, ctx):
    return _vggt(ctx, pair, True)


# --------------------------------------------------------------------------- #
# MapAnything
# --------------------------------------------------------------------------- #


def _mapa_run(ctx, pair, corner_mode, posed):
    key = ("mapa", pair["pair_id"], corner_mode, posed)
    if key in ctx.cache:
        return ctx.cache[key]
    import torch
    from mapanything.utils.image import preprocess_inputs
    size = SIZES["mapanything"]
    views = _views_for(ctx, pair, corner_mode)
    K = torch.from_numpy(_K_at(size)).float()
    inputs = []
    for v in views:
        d = {"img": torch.from_numpy((_rgb(ctx, v, size) * 255.0)).float(), "intrinsics": K}
        if posed:
            Rm, C = M.cam_pose_world(v)
            Tm = np.eye(4)
            Tm[:3, :3], Tm[:3, 3] = Rm, C
            d["camera_poses"] = torch.from_numpy(Tm).float()
            d["is_metric_scale"] = torch.tensor([True])
        inputs.append(d)
    model = _model(ctx, "mapanything")
    t0 = time.time()
    proc = preprocess_inputs(inputs)
    with torch.no_grad():
        preds = model.infer(proc, memory_efficient_inference=False, use_amp=True,
                            amp_dtype="bf16", apply_mask=False, mask_edges=False,
                            apply_confidence_mask=False)
    got = []
    for p in preds:
        got.append({k: p[k][0].float().cpu().numpy() for k in
                    ("pts3d", "camera_poses", "intrinsics", "conf")})
    shape = got[0]["pts3d"].shape[:2]
    res = (views, got, round(time.time() - t0, 3), (shape[1], shape[0]))
    ctx.cache[key] = res
    return res


def _mapa(ctx, pair, corner_mode, posed, prior_projection=False):
    views, got, dt, shape = _mapa_run(ctx, pair, corner_mode, posed)
    size = SIZES["mapanything"]
    out = {"x": None, "y": None, "n_views": len(views), "infer_s": dt}
    if tuple(shape) != tuple(size):
        out["reason"] = f"unexpected_output_size_{shape[0]}x{shape[1]}"
        return out
    sx, sy = _scale(size)
    u, v = CLICK[0] * sx, CLICK[1] * sy
    X = _bilinear(got[0]["pts3d"], u, v)
    out["click_conf"] = float(_bilinear(got[0]["conf"][..., None], u, v)[0])
    T0, T1 = got[0]["camera_poses"], got[1]["camera_poses"]
    out["click_depth"] = float((T0[:3, :3].T @ (X - T0[:3, 3]))[2])
    out.update(_rel_diag(T0, T1, views))
    if posed:
        for i, name in ((0, "src"), (1, "oth")):
            Rm, C = M.cam_pose_world(views[i])
            out[f"{name}_pose_shift_m"] = float(np.linalg.norm(got[i]["camera_poses"][:3, 3] - C))
    if prior_projection:
        Rm, C = M.cam_pose_world(views[1])
        T_prior = np.eye(4)
        T_prior[:3, :3], T_prior[:3, 3] = Rm, C
        return _finish(out, X, T_prior, _K_at(size), views[1], size)
    return _finish(out, X, T1, got[1]["intrinsics"], views[1], size)


MAPA_CONFIG = {"model": MODEL_IDS["mapanything"], "input": SIZES["mapanything"],
               "amp": "bf16", "masking": "off"}


@register("mapa_k_pair", needs=("views",),
          config={**MAPA_CONFIG, "views": 2, "given": "intrinsics"},
          description="MapAnything on the pair, given intrinsics; its cameras and pointmap")
def mapa_k_pair(pair, ctx):
    return _mapa(ctx, pair, False, False)


@register("mapa_k_corner", needs=("views",),
          config={**MAPA_CONFIG, "views": CORNER_VIEWS, "given": "intrinsics"},
          description=f"MapAnything on up to {CORNER_VIEWS} captures, given intrinsics")
def mapa_k_corner(pair, ctx):
    return _mapa(ctx, pair, True, False)


@register("mapa_posed_corner", needs=("views",),
          config={**MAPA_CONFIG, "views": CORNER_VIEWS,
                  "given": "intrinsics + pose priors (metric ENU, flat, 'auto' height)",
                  "projection": "MapAnything's output camera for the other view"},
          description="MapAnything given intrinsics and pose priors; its output cameras")
def mapa_posed_corner(pair, ctx):
    return _mapa(ctx, pair, True, True)


@register("mapa_posed_depthonly", needs=("views",),
          config={**MAPA_CONFIG, "views": CORNER_VIEWS,
                  "given": "intrinsics + pose priors (metric ENU, flat, 'auto' height)",
                  "projection": "the other view's PRIOR camera (only MapAnything's 3D point "
                                "at the click is used)"},
          description="MapAnything posed run, click's 3D point projected with the prior camera")
def mapa_posed_depthonly(pair, ctx):
    return _mapa(ctx, pair, True, True, prior_projection=True)
