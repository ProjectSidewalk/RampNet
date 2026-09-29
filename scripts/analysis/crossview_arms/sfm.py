"""Per-corner structure from motion (#48): COLMAP incremental mapping over every capture
of the corner, with and without GPS position priors.

For each ramp ("corner"), every capture within 25 m is rendered toward the corner
(``_mv3d.py``), ALIKED + LightGlue matches every pair of views, COLMAP (pycolmap 4.2)
verifies the matches and runs incremental mapping with the known pinhole intrinsics held
fixed. If the source view and the pair's other view are both registered in one model, the
source click is lifted to 3D and projected into the other view:

* **lift:** the reconstructed points observed in the source view within ``SUPPORT_PX`` of
  the click (the click is the view centre); a RANSAC plane through them, intersected with
  the click's ray. With fewer than ``MIN_SUPPORT`` points there is no geometry at the click
  and the arm falls back.
* **project:** into the other view's reconstructed camera, then back to the equirect.

Arms (all fall back to the projection when the source or the other view is not registered,
or there is no support at the click; the reason is kept in the row):

* ``sfm_colmap`` -- no priors. The transfer is scale-free: it uses only the reconstruction's
  own relative geometry.
* ``sfm_colmap_prior`` -- COLMAP's position-prior mapping (``use_prior_position``): each
  image's pose prior is its camera centre in the corner's ENU frame (GSV / Mapillary
  position, the labeler's 'auto' height), with a 3 m horizontal / 1 m vertical sigma. The
  model comes out metric, in the ENU frame. Same lift and project.
* ``sfm_prior_poseonly`` -- the same prior reconstruction, but only its two camera poses
  are used: the click is raycast onto flat ground (z = 0) from the refined source camera
  and projected into the refined other camera. Against ``sfm_colmap_prior`` this isolates
  pose from ground geometry.

Heading has no prior: COLMAP position priors constrain the camera centre only.
"""
import os
import shutil
import tempfile
import time

import numpy as np

import crossview_align_48 as H
from crossview_arms import _mv3d as M
from crossview_arms._registry import register

SUPPORT_PX = 120.0
MIN_SUPPORT = 3
PLANE_TOL_FRAC = 0.02       # plane inlier distance, as a fraction of the median point depth
MIN_TWO_VIEW_INLIERS = 15
PRIOR_SIGMA_H_M = 3.0
PRIOR_SIGMA_V_M = 1.0
MAX_VIEWS = None            # every capture within 25 m

SFM_CONFIG = {"features": "ALIKED aliked-n16 + LightGlue (kornia 0.8.3), all view pairs",
              "mapper": "pycolmap incremental_mapping, PINHOLE intrinsics fixed",
              "support_px": SUPPORT_PX, "min_support": MIN_SUPPORT,
              "plane_tol_frac_of_depth": PLANE_TOL_FRAC,
              "min_two_view_inliers": MIN_TWO_VIEW_INLIERS, "max_views": MAX_VIEWS,
              "views": "harness src / oth views plus every capture within 25 m, 1024x768 75 deg"}


# --------------------------------------------------------------------------- #
# features and matches
# --------------------------------------------------------------------------- #


def _lightglue(ctx):
    if "lightglue" not in ctx.cache:
        import torch
        from crossview_arms.matching import LightGlue
        dev = "cuda" if torch.cuda.is_available() and not getattr(ctx.args, "cpu", False) else "cpu"
        ctx.cache["lightglue"] = LightGlue(dev)
    return ctx.cache["lightglue"]


def _features(ctx, view):
    """(keypoints Nx2 in COLMAP pixel convention, i.e. +0.5, descriptors tensor)."""
    key = ("lg_feat", view["view"])
    if key not in ctx.cache:
        import cv2
        img = M.read_view(ctx, view)
        k, d = _lightglue(ctx).features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ctx.cache[key] = (k, d)
    return ctx.cache[key]


def _match(ctx, va, vb):
    lg = _lightglue(ctx)
    KF, torch = lg.KF, lg.torch
    ka, da = _features(ctx, va)
    kb, db = _features(ctx, vb)
    if len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2), dtype=np.uint32)
    la = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=lg.device))
    lb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=lg.device))
    with torch.inference_mode():
        _, idx = lg.matcher(da, db, la, lb, hw1=(H.VIEW_H, H.VIEW_W), hw2=(H.VIEW_H, H.VIEW_W))
    return idx.cpu().numpy().astype(np.uint32)


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #


def reconstruct(ctx, corner, pair, priors):
    """Run COLMAP on the corner's views for ``pair``. Returns (recon or None, names, diag):
    ``names[i]`` is the image name of the i-th selected view (0 = source, 1 = other)."""
    import pycolmap
    views = M.select_views(corner, pair, MAX_VIEWS)
    tmp = tempfile.mkdtemp(prefix="mv3d_sfm_", dir=M.extra(ctx, "tmp"))
    diag = {"n_views": len(views)}
    try:
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir)
        names = []
        for i, v in enumerate(views):
            name = f"{i:02d}.jpg"
            shutil.copyfile(M.view_path(ctx, v), os.path.join(img_dir, name))
            names.append(name)
        db_path = os.path.join(tmp, "db.db")
        K = M.intrinsics()
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        pycolmap.Database.open(db_path).close()          # creates the schema
        pycolmap.import_images(db_path, img_dir, pycolmap.CameraMode.SINGLE, options=ro)
        db = pycolmap.Database.open(db_path)
        ids = {im.name: im.image_id for im in db.read_all_images()}
        kps = []
        for name, v in zip(names, views):
            k, _ = _features(ctx, v)
            kp = k.cpu().numpy().astype(np.float32) + 0.5
            kps.append(kp)
            db.write_keypoints(ids[name], kp)
        pairs_txt = os.path.join(tmp, "pairs.txt")
        n_raw = 0
        with open(pairs_txt, "w") as f:
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    m = _match(ctx, views[i], views[j])
                    if len(m) < MIN_TWO_VIEW_INLIERS:
                        continue
                    db.write_matches(ids[names[i]], ids[names[j]], m)
                    f.write(f"{names[i]} {names[j]}\n")
                    n_raw += 1
        diag["matched_pairs"] = n_raw
        if priors:
            for name, v in zip(names, views):
                im = db.read_image(ids[name])
                cov = np.diag([PRIOR_SIGMA_H_M ** 2, PRIOR_SIGMA_H_M ** 2, PRIOR_SIGMA_V_M ** 2])
                pp = pycolmap.PosePrior(
                    position=np.array([v["e"], v["n"], v["h"]], dtype=float),
                    position_covariance=cov,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                    corr_data_id=im.data_id)
                db.write_pose_prior(pp)
        db.close()
        pycolmap.verify_matches(db_path, pairs_txt)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        opts.min_model_size = 2
        opts.min_num_matches = MIN_TWO_VIEW_INLIERS
        opts.multiple_models = True
        opts.random_seed = H.SEED
        opts.num_threads = 8
        opts.mapper.init_min_num_inliers = 30
        opts.mapper.abs_pose_min_num_inliers = 15
        opts.mapper.abs_pose_refine_focal_length = False
        opts.mapper.abs_pose_refine_extra_params = False
        opts.mapper.init_min_tri_angle = 4.0
        opts.mapper.random_seed = H.SEED
        if priors:
            opts.use_prior_position = True
            opts.use_robust_loss_on_prior_position = True
        out_dir = os.path.join(tmp, "sparse")
        os.makedirs(out_dir)
        pycolmap.set_random_seed(H.SEED)
        recs = pycolmap.incremental_mapping(db_path, img_dir, out_dir, options=opts)
        diag["n_models"] = len(recs)
        best = None
        for r in recs.values():
            reg = {r.image(i).name for i in r.reg_image_ids()}
            if names[0] in reg and names[1] in reg:
                best = r
                break
        if best is None:
            sizes = sorted((r.num_reg_images() for r in recs.values()), reverse=True)
            diag["largest_model"] = sizes[0] if sizes else 0
            diag["src_registered"] = any(names[0] in {r.image(i).name for i in r.reg_image_ids()}
                                         for r in recs.values())
            diag["oth_registered"] = any(names[1] in {r.image(i).name for i in r.reg_image_ids()}
                                         for r in recs.values())
            return None, names, diag
        diag["n_registered"] = best.num_reg_images()
        diag["n_points3d"] = best.num_points3D()
        diag["mean_reproj_px"] = float(best.compute_mean_reprojection_error())
        return best, names, diag
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _image(rec, name):
    for i in rec.reg_image_ids():
        im = rec.image(i)
        if im.name == name:
            return im
    return None


def _cam(im):
    """(R_cw, C) from a COLMAP image: camera-to-world rotation and centre."""
    T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    Rm = T.rotation.matrix()
    t = np.asarray(T.translation)
    return Rm.T, -Rm.T @ t


def lift_click(rec, src_im, u=H.VIEW_W / 2.0, v=H.VIEW_H / 2.0):
    """3D point where the click's ray meets a RANSAC plane through the reconstructed points
    the source view observes within SUPPORT_PX of the click. (X, diag) or (None, diag)."""
    R_cw, C = _cam(src_im)
    K = M.intrinsics()
    pts = []
    for p2 in src_im.points2D:
        if p2.has_point3D() and np.hypot(p2.xy[0] - u, p2.xy[1] - v) < SUPPORT_PX:
            pts.append(np.asarray(rec.point3D(p2.point3D_id).xyz))
    diag = {"n_support": len(pts)}
    if len(pts) < MIN_SUPPORT:
        return None, diag
    P = np.array(pts)
    ray = R_cw @ np.linalg.solve(K, np.array([u, v, 1.0]))
    ray /= np.linalg.norm(ray)
    depth = np.median((P - C) @ ray)
    X = _plane_hit(P, C, ray, PLANE_TOL_FRAC * abs(depth))
    if X is None:
        X = C + depth * ray
        diag["lift"] = "median_depth"
    else:
        diag["lift"] = "plane"
    if (X - C) @ ray <= 0:
        return None, diag
    diag["click_depth"] = float((X - C) @ ray)
    return X, diag


def _plane_hit(P, C, ray, tol, iters=200, seed=H.SEED):
    """Intersect the ray with a RANSAC plane through P (None if degenerate)."""
    if len(P) < 3:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best_n:
            best_n, best = inl.sum(), inl
    if best is None or best_n < 3:
        return None
    Q = P[best]
    cen = Q.mean(0)
    n = np.linalg.svd(Q - cen)[2][-1]
    den = ray @ n
    if abs(den) < 1e-6:
        return None
    t = ((cen - C) @ n) / den
    return None if t <= 0 else C + t * ray


def _to_pano(rec_oth_im, X, oth_view):
    R_cw, C = _cam(rec_oth_im)
    uv = M.project_cam(R_cw, C, M.intrinsics(), X)
    if uv is None:
        return None
    x, y = M.view_pixel_to_pano(oth_view, *uv)
    return {"x": x, "y": y, "u": uv[0], "v": uv[1]}


def _recon_for(ctx, pair, priors):
    key = ("sfm", pair["pair_id"], priors)
    if key not in ctx.cache:
        corner = M.corner_for(ctx, pair)
        t0 = time.time()
        rec, names, diag = reconstruct(ctx, corner, pair, priors)
        diag["sfm_s"] = round(time.time() - t0, 2)
        views = M.select_views(corner, pair, MAX_VIEWS)
        ctx.cache[key] = (rec, names, diag, views)
    return ctx.cache[key]


def _transfer(pair, ctx, priors):
    rec, names, diag, views = _recon_for(ctx, pair, priors)
    out = {"x": None, "y": None, **diag}
    if rec is None:
        out["reason"] = "not_co_registered"
        return out
    s, o = _image(rec, names[0]), _image(rec, names[1])
    X, d2 = lift_click(rec, s)
    out.update(d2)
    if X is None:
        out["reason"] = "no_support"
        return out
    r = _to_pano(o, X, views[1])
    if r is None:
        out["reason"] = "behind_other"
        return out
    out.update(r)
    return out


@register("sfm_colmap", needs=("views",), config={**SFM_CONFIG, "priors": "none"},
          description="per-corner COLMAP SfM (no priors); click lifted by a local plane, "
                      "projected into the other view")
def sfm_colmap(pair, ctx):
    return _transfer(pair, ctx, priors=False)


@register("sfm_colmap_prior", needs=("views",),
          config={**SFM_CONFIG, "priors": f"position, sigma {PRIOR_SIGMA_H_M} m horizontal / "
                                           f"{PRIOR_SIGMA_V_M} m vertical, robust"},
          description="per-corner COLMAP SfM with GPS position priors; local-plane lift")
def sfm_colmap_prior(pair, ctx):
    return _transfer(pair, ctx, priors=True)


@register("sfm_prior_poseonly", needs=("views",),
          config={**SFM_CONFIG, "priors": "as sfm_colmap_prior",
                  "ground": "flat z = 0 in the ENU prior frame (cameras at their 'auto' height)"},
          description="COLMAP prior poses only: flat-ground raycast between the refined cameras")
def sfm_prior_poseonly(pair, ctx):
    rec, names, diag, views = _recon_for(ctx, pair, True)
    out = {"x": None, "y": None, **diag}
    if rec is None:
        out["reason"] = "not_co_registered"
        return out
    R_s, C_s = _cam(_image(rec, names[0]))
    ray = R_s @ np.linalg.solve(M.intrinsics(), np.array([H.VIEW_W / 2.0, H.VIEW_H / 2.0, 1.0]))
    if ray[2] >= -1e-9 or C_s[2] <= 0:
        out["reason"] = "no_ground_hit"
        return out
    X = C_s + (C_s[2] / -ray[2]) * ray
    out["refined_src_height"] = float(C_s[2])
    r = _to_pano(_image(rec, names[1]), X, views[1])
    if r is None:
        out["reason"] = "behind_other"
        return out
    out.update(r)
    return out
