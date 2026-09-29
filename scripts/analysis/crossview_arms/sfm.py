"""Per-corner structure from motion (#48): COLMAP incremental mapping over every capture
of the corner, with and without GPS position priors.

For each ramp ("corner"), every capture within 25 m is rendered toward the corner
(``_mv3d.py``), ALIKED + LightGlue matches every pair of views, COLMAP (pycolmap 4.2)
verifies the matches and runs incremental mapping with the known pinhole intrinsics held
fixed. If the source view and the pair's other view are both registered in one model, the
source click is lifted to 3D and projected into the other view:

* **lift:** the click's ray meets a ground plane perpendicular to gravity. Gravity in the
  model's frame comes from the source camera (its view was cut from a level pano, so its
  down axis is known); the ground's level is the mode of the reconstructed points the
  source view observes below the horizon within ``SUPPORT_PX`` of the click (the click is
  the view centre). Fewer than ``MIN_SUPPORT`` such points: no geometry at the click, fall
  back. (The pilot's first lift, a free RANSAC plane through the same neighbourhood, locked
  onto background structure and is recorded in the doc, not kept.)
* **project:** into the other view's reconstructed camera, then back to the equirect.

Arms (all fall back to the projection when the source or the other view is not registered,
or there is no support at the click; the reason is kept in the row):

* ``sfm_colmap`` -- no priors. The transfer is scale-free: it uses only the reconstruction's
  own relative geometry.
* ``sfm_colmap_prior`` -- COLMAP's position-prior mapping (``use_prior_position``): each
  image's pose prior is its camera centre in the corner's ENU frame (GSV / Mapillary
  position, the labeler's 'auto' height), with a 3 m horizontal / 1 m vertical sigma. The
  model comes out metric, in the ENU frame. Same lift and project.
* ``sfm_poseonly`` -- the no-prior reconstruction's POSE only: the source camera keeps
  its prior pose, the other camera gets the model's relative rotation and baseline
  direction (baseline length from the priors), and the click goes through today's
  flat-ground transfer at the 'auto' height (``_mv3d.poseonly_transfer``). Against
  ``sfm_colmap`` this separates pose from ground geometry.

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
GROUND_TOL_FRAC = 0.05      # ground-level mode tolerance, a fraction of the level
GROUND_MIN_DEPRESSION_DEG = 2.0   # support points must be this far below the horizon
MIN_TWO_VIEW_INLIERS = 15
PRIOR_SIGMA_H_M = 3.0
PRIOR_SIGMA_V_M = 1.0
MAX_VIEWS = None            # every capture within 25 m

SFM_CONFIG = {"features": "ALIKED aliked-n16 + LightGlue (kornia 0.8.3), all view pairs",
              "mapper": "pycolmap incremental_mapping, PINHOLE intrinsics fixed",
              "support_px": SUPPORT_PX, "min_support": MIN_SUPPORT,
              "lift": "gravity-perpendicular ground at the mode of the support points' level",
              "ground_tol_frac": GROUND_TOL_FRAC,
              "ground_min_depression_deg": GROUND_MIN_DEPRESSION_DEG,
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


def lift_click(rec, src_im, src_view, u=H.VIEW_W / 2.0, v=H.VIEW_H / 2.0):
    """The click's 3D point: its ray from the source camera meets a ground plane
    perpendicular to gravity, at the height the reconstructed ground points give. Gravity
    in the reconstruction's frame comes from the source camera itself -- the view was
    rendered from a level pano, so its down axis in the pano frame is known. The ground
    height is the densest level (mode, tolerance ``GROUND_TOL_FRAC`` of the level) among
    the points the source view observes below the horizon within ``SUPPORT_PX`` of the
    click. (X, diag) or (None, diag)."""
    R_cw, C = _cam(src_im)
    g = R_cw @ H.view_rotation(src_view["cx"], src_view["cy"]).T @ np.array([0.0, 1.0, 0.0])
    pts = []
    for p2 in src_im.points2D:
        if not p2.has_point3D() or np.hypot(p2.xy[0] - u, p2.xy[1] - v) >= SUPPORT_PX:
            continue
        _, y = H.view_to_pano(p2.xy[0], p2.xy[1], src_view["cx"], src_view["cy"])
        if H.elevation_deg(y) < -GROUND_MIN_DEPRESSION_DEG:
            pts.append(np.asarray(rec.point3D(p2.point3D_id).xyz))
    diag = {"n_support": len(pts)}
    if len(pts) < MIN_SUPPORT:
        return None, diag
    h = (np.array(pts) - C) @ g                   # how far below the camera, along gravity
    h = h[h > 0]
    if len(h) < MIN_SUPPORT:
        return None, diag
    counts = [(np.sum(np.abs(h - hi) < GROUND_TOL_FRAC * hi), hi) for hi in h]
    n_best, h0 = max(counts)
    inl = h[np.abs(h - h0) < GROUND_TOL_FRAC * h0]
    diag["n_ground"] = int(len(inl))
    if len(inl) < MIN_SUPPORT:
        return None, diag
    hg = float(np.median(inl))
    ray = R_cw @ np.linalg.solve(M.intrinsics(), np.array([u, v, 1.0]))
    ray /= np.linalg.norm(ray)
    if ray @ g <= 1e-6:
        return None, diag
    X = C + (hg / (ray @ g)) * ray
    diag["click_depth"] = float(hg / (ray @ g))
    diag["ground_below_cam"] = hg
    return X, diag


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


def rel_pose_agreement(Rs, Cs, Ro, Co, vs, vo):
    """Diagnostics: how far a reconstruction's src->oth relative pose is from the pose
    prior's -- relative rotation angle and baseline-direction angle, degrees."""
    Rs_p, Cs_p = M.cam_pose_world(vs)
    Ro_p, Co_p = M.cam_pose_world(vo)
    rel_est, rel_pri = Ro.T @ Rs, Ro_p.T @ Rs_p
    c = (np.trace(rel_est.T @ rel_pri) - 1.0) / 2.0
    rot = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    b_est, b_pri = Rs.T @ (Co - Cs), Rs_p.T @ (Co_p - Cs_p)
    cb = b_est @ b_pri / (np.linalg.norm(b_est) * np.linalg.norm(b_pri) + 1e-12)
    return {"rel_rot_vs_prior_deg": rot,
            "baseline_dir_vs_prior_deg": float(np.degrees(np.arccos(np.clip(cb, -1, 1))))}


def _transfer(pair, ctx, priors):
    rec, names, diag, views = _recon_for(ctx, pair, priors)
    out = {"x": None, "y": None, **diag}
    if rec is None:
        out["reason"] = "not_co_registered"
        return out
    s, o = _image(rec, names[0]), _image(rec, names[1])
    out.update(rel_pose_agreement(*_cam(s), *_cam(o), views[0], views[1]))
    X, d2 = lift_click(rec, s, views[0])
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


@register("sfm_poseonly", needs=("views",),
          config={**SFM_CONFIG, "priors": "none (as sfm_colmap)",
                  "transfer": "prior source camera + the reconstruction's src->oth relative "
                              "rotation and baseline direction (baseline length from the "
                              "priors); flat ground at the 'auto' height"},
          description="SfM relative pose only: today's flat-ground transfer with the other "
                      "camera re-posed by the reconstruction")
def sfm_poseonly(pair, ctx):
    rec, names, diag, views = _recon_for(ctx, pair, False)
    out = {"x": None, "y": None, **diag}
    if rec is None:
        out["reason"] = "not_co_registered"
        return out
    s, o = _image(rec, names[0]), _image(rec, names[1])
    out.update(rel_pose_agreement(*_cam(s), *_cam(o), views[0], views[1]))
    r = M.poseonly_transfer(*_cam(s), *_cam(o), views[0], views[1])
    if r is None:
        out["reason"] = "no_ground_hit"
        return out
    out.update(r)
    return out
