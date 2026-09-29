"""Per-corner 3D reconstruction from flat Mapillary images plus the 360 pano views (#214,
part of #48), and the click transfer the ``flat_*`` harness arms read.

One corner per Richmond ramp in the frozen harness pairs. For a corner (its image list is
``analysis_out/flat_mapillary_3d/manifest.json``, built by ``flat_mapillary_48.py select``):

1. **Images.** The harness's own pano views (``<pair>_src.jpg`` centred on the GT click,
   ``<pair>_oth.jpg`` centred on today's projection; 1024 x 768, 75 deg), one view of every
   other run pano within 25 m aimed at the corner, and the flat Mapillary thumbnails.
2. **Cameras.** Pano views: exact pinhole. Flat images: COLMAP RADIAL from Mapillary's own
   SfM-refined ``camera_parameters`` (OpenSfM focal normalised by the longer side, k1, k2).
   Intrinsics are held fixed in bundle adjustment.
3. **Features.** ALIKED (top ``MAX_KP`` by score, never a raster slice) + LightGlue
   (kornia 0.8.3), every image pair; flat images are resized to ``FLAT_MAX_SIDE`` first.
4. **SfM.** pycolmap incremental mapping with position priors: each image's Mapillary SfM
   position (``computed_geometry``) in the corner's ENU frame, 3 m horizontal sigma,
   robust. Heights have no measurement: z prior 2.6 m (pano) / 1.5 m (flat), 2 m sigma.
   The model comes out metric and approximately in the ENU frame.
5. **Lifts** of the source click (the centre of the ``_src`` view) to 3D, each a separate
   arm:
   * ``sparse`` -- as ``sfm.py`` on ``analysis/crossview-sfm-48``: the click ray meets a
     plane perpendicular to gravity (gravity from the source view, which was cut level from
     a gravity-rectified pano) at the modal level of the reconstructed points seen near the
     click;
   * ``gs`` -- the expected depth a 3D Gaussian Splatting model (gsplat) renders at the
     click in the source camera;
   * ``mvs`` -- COLMAP patch-match (geometric) depth at the click, when CUDA COLMAP is
     available.
6. **Transfer.** The 3D point is projected into each pair's ``_oth`` camera (same model)
   and mapped to that pano's equirect. Nothing reads the reference: the manifest carries
   the pairs without their answer columns.

Outputs, per corner, under ``--out/<ramp_uid>/``: the sparse model, the GS model
(``splat.ply``), MVS depth maps, renders, and ``result.json`` (stats, registered cameras,
lifts, per-pair transfers). ``result.json`` is what gets committed
(``analysis_out/flat_mapillary_3d/corners/``) and what the arms read.

    python scripts/analysis/flat3d/reconstruct.py --corner richmond:180 \\
        --flat-dir FLAT --views VIEWS --archive-root ARCHIVE --out OUT [--no-flat] [--no-gs]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, ANALYSIS)

import crossview_align_48 as H  # noqa: E402
import flat_mapillary_48 as F  # noqa: E402

FLAT_MAX_SIDE = 1600
MAX_KP = 4096
MIN_MATCHES = 15
PRIOR_SIGMA_H_M = 3.0
PRIOR_SIGMA_V_M = 2.0
# sparse lift (the constants of sfm.py on analysis/crossview-sfm-48)
SUPPORT_PX = 120.0
MIN_SUPPORT = 3
GROUND_TOL_FRAC = 0.05
GROUND_MIN_DEPRESSION_DEG = 2.0
# gaussian splatting
GS_ITERS = 7000
GS_MAX_SIDE = 1024
GS_SH_DEGREE = 0
DEPTH_WIN = 2            # median over a (2w+1)^2 window at the click
SEED = 48

CONFIG = {"flat_max_side": FLAT_MAX_SIDE, "max_kp": MAX_KP, "min_matches": MIN_MATCHES,
          "prior_sigma_h_m": PRIOR_SIGMA_H_M, "prior_sigma_v_m": PRIOR_SIGMA_V_M,
          "support_px": SUPPORT_PX, "min_support": MIN_SUPPORT,
          "ground_tol_frac": GROUND_TOL_FRAC,
          "ground_min_depression_deg": GROUND_MIN_DEPRESSION_DEG, "gs_iters": GS_ITERS,
          "gs_max_side": GS_MAX_SIDE, "gs_sh_degree": GS_SH_DEGREE, "depth_win": DEPTH_WIN,
          "features": "ALIKED aliked-n16 top-k + LightGlue (kornia 0.8.3), all image pairs",
          "mapper": "pycolmap incremental_mapping, position priors, intrinsics fixed"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #


def prepare_images(corner, args, img_dir):
    """Write every image of the corner into img_dir; returns the kept entries with their
    pixel size and intrinsics [f, cx, cy, k1, k2]."""
    import cv2
    os.makedirs(img_dir, exist_ok=True)
    kept, equi_cache = [], {}
    fview = H.focal_px()
    for im in corner["images"]:
        dst = os.path.join(img_dir, im["name"])
        if im["kind"] in ("pano_src", "pano_oth"):
            src = os.path.join(args.views, im["name"])
            if not os.path.exists(src):
                log("missing harness view", im["name"])
                continue
            shutil.copyfile(src, dst)
            w, h = H.VIEW_W, H.VIEW_H
            params = [fview, w / 2.0, h / 2.0, 0.0, 0.0]
        elif im["kind"] in ("pano_extra", "mly_pano"):
            if im["kind"] == "mly_pano" and not args.mly_panos:
                continue
            p = (os.path.join(args.archive_root, "richmond", "panos", f"{im['id']}.jpg")
                 if im["kind"] == "pano_extra" else os.path.join(args.mly_pano_dir,
                                                                 f"{im['id']}.jpg"))
            if im["id"] not in equi_cache:
                e = cv2.imread(p, cv2.IMREAD_COLOR)
                if e is None:
                    log("missing pano", p)
                    continue
                tw = int(round(360.0 / H.HFOV_DEG * H.VIEW_W))
                if e.shape[1] != 2 * e.shape[0]:
                    log("not a 2:1 equirect, skipped", p)
                    continue
                equi_cache[im["id"]] = cv2.resize(e, (tw, tw // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, H.render_view(equi_cache[im["id"]], im["cx"], im["cy"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            w, h = H.VIEW_W, H.VIEW_H
            params = [fview, w / 2.0, h / 2.0, 0.0, 0.0]
        else:
            if args.no_flat:
                continue
            src = os.path.join(args.flat_dir, f"{im['id']}.jpg")
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                log("missing flat image", im["id"])
                continue
            h0, w0 = img.shape[:2]
            s = min(1.0, FLAT_MAX_SIDE / max(w0, h0))
            if s < 1.0:
                img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = img.shape[:2]
            cp = im["camera_parameters"]
            f_norm = float(cp[0])
            k1 = float(cp[1]) if len(cp) > 1 else 0.0
            k2 = float(cp[2]) if len(cp) > 2 else 0.0
            params = [f_norm * max(w, h), w / 2.0, h / 2.0, k1, k2]
        kept.append(dict(im, w=w, h=h, params=params))
    return kept


# --------------------------------------------------------------------------- #
# features and matching
# --------------------------------------------------------------------------- #


class Matcher:
    def __init__(self, device):
        import kornia.feature as KF
        import torch
        torch.manual_seed(SEED)
        self.torch, self.KF, self.device = torch, KF, device
        self.extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=MAX_KP,
                                                   device=device).eval()
        self.matcher = KF.LightGlueMatcher("aliked").to(device).eval()

    def features(self, path):
        import cv2
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        t = self.torch.from_numpy(g).float()[None, None].to(self.device) / 255.0
        with self.torch.inference_mode():
            f = self.extractor(t.repeat(1, 3, 1, 1))[0]
        return f.keypoints, f.descriptors, g.shape[:2]

    def match(self, fa, fb):
        KF, torch = self.KF, self.torch
        ka, da, ha = fa
        kb, db, hb = fb
        if len(ka) < 8 or len(kb) < 8:
            return np.zeros((0, 2), dtype=np.uint32)
        la = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=self.device))
        lb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=self.device))
        with torch.inference_mode():
            _, idx = self.matcher(da, db, la, lb, hw1=ha, hw2=hb)
        return idx.cpu().numpy().astype(np.uint32)


# --------------------------------------------------------------------------- #
# SfM
# --------------------------------------------------------------------------- #


def run_sfm(kept, img_dir, work, matcher):
    import pycolmap
    db_path = os.path.join(work, "db.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "RADIAL"
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, img_dir, pycolmap.CameraMode.PER_IMAGE, options=ro,
                           image_names=[k["name"] for k in kept])
    db = pycolmap.Database.open(db_path)
    ids = {im.name: im for im in db.read_all_images()}
    t0 = time.time()
    feats = {}
    for k in kept:
        im = ids[k["name"]]
        cam = db.read_camera(im.camera_id)
        cam.params = np.array(k["params"], dtype=float)
        cam.has_prior_focal_length = True
        db.update_camera(cam)
        f = matcher.features(os.path.join(img_dir, k["name"]))
        feats[k["name"]] = f
        db.write_keypoints(im.image_id, f[0].cpu().numpy().astype(np.float32) + 0.5)
        cov = np.diag([PRIOR_SIGMA_H_M ** 2, PRIOR_SIGMA_H_M ** 2, PRIOR_SIGMA_V_M ** 2])
        db.write_pose_prior(pycolmap.PosePrior(
            position=np.array([k["e"], k["n"], k["z_prior"]], dtype=float),
            position_covariance=cov,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
            corr_data_id=im.data_id))
    t_feat = time.time() - t0
    pairs_txt = os.path.join(work, "pairs.txt")
    n_pairs = n_matched = 0
    names = [k["name"] for k in kept]
    with open(pairs_txt, "w") as fh:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n_pairs += 1
                m = matcher.match(feats[names[i]], feats[names[j]])
                if len(m) < MIN_MATCHES:
                    continue
                db.write_matches(ids[names[i]].image_id, ids[names[j]].image_id, m)
                fh.write(f"{names[i]} {names[j]}\n")
                n_matched += 1
    db.close()
    t_match = time.time() - t0 - t_feat
    pycolmap.verify_matches(db_path, pairs_txt)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.min_model_size = 2
    opts.min_num_matches = MIN_MATCHES
    opts.multiple_models = True
    opts.random_seed = SEED
    opts.num_threads = 16
    opts.mapper.init_min_num_inliers = 30
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.random_seed = SEED
    opts.use_prior_position = True
    opts.use_robust_loss_on_prior_position = True
    out_dir = os.path.join(work, "sparse")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    pycolmap.set_random_seed(SEED)
    recs = pycolmap.incremental_mapping(db_path, img_dir, out_dir, options=opts)
    t_map = time.time() - t0 - t_feat - t_match
    diag = {"n_images": len(kept), "pairs_tried": n_pairs, "pairs_matched": n_matched,
            "n_models": len(recs), "t_features_s": round(t_feat, 1),
            "t_match_s": round(t_match, 1), "t_map_s": round(t_map, 1),
            "model_sizes": sorted((r.num_reg_images() for r in recs.values()), reverse=True)}
    return recs, diag


def cam_of(im):
    T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    Rm = T.rotation.matrix()
    t = np.asarray(T.translation)
    return Rm.T, -Rm.T @ t        # (R_cw, C)


def K_of(cam):
    return np.asarray(cam.calibration_matrix())


# --------------------------------------------------------------------------- #
# lifts
# --------------------------------------------------------------------------- #


def src_gravity(R_cw, src):
    """World-frame down direction from the level source view."""
    return R_cw @ H.view_rotation(src["cx"], src["cy"]).T @ np.array([0.0, 1.0, 0.0])


def click_ray(R_cw, K):
    r = R_cw @ np.linalg.solve(K, np.array([H.VIEW_W / 2.0, H.VIEW_H / 2.0, 1.0]))
    return r / np.linalg.norm(r)


def lift_sparse(rec, im, src):
    """sfm.py's lift (analysis/crossview-sfm-48), unchanged in its constants."""
    R_cw, C = cam_of(im)
    g = src_gravity(R_cw, src)
    u0, v0 = H.VIEW_W / 2.0, H.VIEW_H / 2.0
    pts = []
    for p2 in im.points2D:
        if not p2.has_point3D() or np.hypot(p2.xy[0] - u0, p2.xy[1] - v0) >= SUPPORT_PX:
            continue
        _, y = H.view_to_pano(p2.xy[0], p2.xy[1], src["cx"], src["cy"])
        if H.elevation_deg(y) < -GROUND_MIN_DEPRESSION_DEG:
            pts.append(np.asarray(rec.point3D(p2.point3D_id).xyz))
    diag = {"n_support": len(pts)}
    if len(pts) < MIN_SUPPORT:
        return None, diag
    hgt = (np.array(pts) - C) @ g
    hgt = hgt[hgt > 0]
    if len(hgt) < MIN_SUPPORT:
        return None, diag
    counts = [(np.sum(np.abs(hgt - hi) < GROUND_TOL_FRAC * hi), hi) for hi in hgt]
    _, h0 = max(counts)
    inl = hgt[np.abs(hgt - h0) < GROUND_TOL_FRAC * h0]
    diag["n_ground"] = int(len(inl))
    if len(inl) < MIN_SUPPORT:
        return None, diag
    hg = float(np.median(inl))
    ray = click_ray(R_cw, K_of(rec.camera(im.camera_id)))
    if ray @ g <= 1e-6:
        return None, diag
    X = C + (hg / (ray @ g)) * ray
    diag.update(click_depth=float(hg / (ray @ g)), ground_below_cam=hg)
    return X, diag


def lift_from_depth(R_cw, C, K, depth_z, diag):
    """X from a z-depth at the click pixel."""
    if depth_z is None or not np.isfinite(depth_z) or depth_z <= 0:
        return None, diag
    r = np.linalg.solve(K, np.array([H.VIEW_W / 2.0, H.VIEW_H / 2.0, 1.0]))
    X = C + R_cw @ (r * depth_z)
    diag["click_depth"] = float(depth_z * np.linalg.norm(r))
    return X, diag


def window_median(D, u, v, w=DEPTH_WIN):
    a = D[max(0, v - w):v + w + 1, max(0, u - w):u + w + 1]
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if len(a) else None


# --------------------------------------------------------------------------- #
# transfer
# --------------------------------------------------------------------------- #


def transfer(rec, X, oth_im, oth):
    R_cw, C = cam_of(oth_im)
    p = R_cw.T @ (np.asarray(X) - C)
    if p[2] <= 1e-6:
        return {"reason": "behind_other"}
    q = K_of(rec.camera(oth_im.camera_id)) @ (p / p[2])
    x, y = H.view_to_pano(q[0], q[1], oth["cx"], oth["cy"])
    return {"x": float(x), "y": float(y), "u": float(q[0]), "v": float(q[1]),
            "range_oth_m": float(np.linalg.norm(np.asarray(X) - C))}


# --------------------------------------------------------------------------- #
# dense: undistortion, MVS, gaussian splatting
# --------------------------------------------------------------------------- #


def undistort(rec_dir, img_dir, dense):
    import pycolmap
    shutil.rmtree(dense, ignore_errors=True)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = GS_MAX_SIDE * 2
    pycolmap.undistort_images(dense, rec_dir, img_dir, output_type="COLMAP",
                              num_patch_match_src_images=20, undistort_options=opts)


def read_colmap_array(path):
    with open(path, "rb") as fid:
        w, h, c = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        n = 0
        while True:
            if fid.read(1) == b"&":
                n += 1
                if n == 3:
                    break
        a = np.fromfile(fid, np.float32)
    a = a.reshape((w, h, c), order="F")
    return np.transpose(a, (1, 0, 2)).squeeze()


def run_mvs(dense, colmap_bin, max_size, ref_name):
    """Photometric patch-match depth for the source view only (its 20 best source images,
    COLMAP's __auto__ choice). Every view with geometric consistency took over 25 min per
    corner in the pilot; the click needs one depth map."""
    t0 = time.time()
    cfg = os.path.join(dense, "stereo", "patch-match.cfg")
    with open(cfg, "w") as f:
        f.write(f"{ref_name}\n__auto__, 20\n")
    cmd = colmap_bin + ["patch_match_stereo", "--workspace_path", dense,
                        "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency",
                        "false", "--PatchMatchStereo.max_image_size", str(max_size)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("MVS failed:", r.stderr[-500:])
        return None
    return round(time.time() - t0, 1)


def train_gs(dense, out_ply, render_dir, want_names, iters=GS_ITERS):
    """Minimal gsplat trainer (L1 + 0.2 D-SSIM, gsplat DefaultStrategy densification, SH 0).
    Returns (renders {name: (rgb, depth, K, R_cw, C)}, diag)."""
    import cv2
    import pycolmap
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from kornia.losses import ssim_loss
    torch.manual_seed(SEED)
    dev = "cuda"
    rec = pycolmap.Reconstruction(os.path.join(dense, "sparse"))
    views = []
    for iid in sorted(rec.reg_image_ids()):
        im = rec.image(iid)
        cam = rec.camera(im.camera_id)
        img = cv2.imread(os.path.join(dense, "images", im.name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        s = min(1.0, GS_MAX_SIDE / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        K = K_of(cam).copy()
        K[:2] *= np.array([[img.shape[1] / cam.width], [img.shape[0] / cam.height]])
        R_cw, C = cam_of(im)
        vm = np.eye(4)
        vm[:3, :3] = R_cw.T
        vm[:3, 3] = -R_cw.T @ C
        views.append({"name": im.name, "img": torch.from_numpy(img[..., ::-1].copy()).float()
                      .div(255).to(dev), "K": torch.tensor(K, dtype=torch.float32, device=dev),
                      "vm": torch.tensor(vm, dtype=torch.float32, device=dev),
                      "R_cw": R_cw, "C": C, "Knp": K, "wh": (img.shape[1], img.shape[0])})
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()]) / 255.0
    n = len(xyz)
    # scale init: mean distance to 3 nearest neighbours
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=4)
    scale = np.log(np.clip(d[:, 1:].mean(1), 1e-3, None))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(np.repeat(scale[:, None], 3, 1),
                                                  dtype=torch.float32, device=dev)),
        "quats": torch.nn.Parameter(torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)),
                                                 dtype=torch.float32, device=dev)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(0.1 / 0.9), device=dev)),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5)
                                   / 0.28209479177387814)[:, None, :]),
    })
    extent = float(np.linalg.norm(np.array([v["C"] for v in views]) -
                                  np.mean([v["C"] for v in views], 0), axis=1).max()) * 1.1
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strategy = DefaultStrategy(refine_stop_iter=int(iters * 0.7), verbose=False)
    strategy.check_sanity(params, opt)
    state = strategy.initialize_state(scene_scale=extent)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / iters))
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    loss_hist = []
    for step in range(iters):
        v = views[rng.integers(len(views))]
        w, h = v["wh"]
        colors, alphas, info = rasterization(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["sh0"],
            v["vm"][None], v["K"][None], w, h, sh_degree=0, packed=False,
            absgrad=False)
        strategy.step_pre_backward(params, opt, state, step, info)
        pred = colors[0]
        gt = v["img"]
        l1 = (pred - gt).abs().mean()
        ss = ssim_loss(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], 11)
        loss = 0.8 * l1 + 0.2 * ss
        loss.backward()
        strategy.step_post_backward(params, opt, state, step, info, packed=False)
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if step % 1000 == 0 or step == iters - 1:
            loss_hist.append([step, round(float(loss), 4), int(len(params["means"]))])
    t_train = time.time() - t0
    renders = {}
    with torch.no_grad():
        for v in views:
            if v["name"] not in want_names:
                continue
            w, h = v["wh"]
            out, _, _ = rasterization(
                params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["sh0"],
                v["vm"][None], v["K"][None], w, h, sh_degree=0, render_mode="RGB+ED")
            o = out[0].cpu().numpy()
            renders[v["name"]] = (o[..., :3], o[..., 3], v["Knp"], v["R_cw"], v["C"])
            os.makedirs(render_dir, exist_ok=True)
            both = np.concatenate([v["img"].cpu().numpy(), np.clip(o[..., :3], 0, 1)], 1)
            cv2.imwrite(os.path.join(render_dir, f"gs_{v['name'][:-4]}.jpg"),
                        (both[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
        # PSNR on every training view (no held-out views: this is a fit, not a benchmark)
        psnr = []
        for v in views:
            w, h = v["wh"]
            out, _, _ = rasterization(
                params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["sh0"],
                v["vm"][None], v["K"][None], w, h, sh_degree=0)
            mse = float(((out[0].clamp(0, 1) - v["img"]) ** 2).mean())
            psnr.append(-10 * math.log10(max(mse, 1e-10)))
    export_ply(params, out_ply)
    diag = {"n_gaussians": int(len(params["means"])), "t_train_s": round(t_train, 1),
            "iters": iters, "n_train_views": len(views), "loss": loss_hist,
            "train_psnr_median": round(float(np.median(psnr)), 2),
            "ply_bytes": os.path.getsize(out_ply)}
    return renders, diag


def export_ply(params, path):
    """Standard 3DGS .ply (SH degree 0), readable by common splat viewers."""
    from plyfile import PlyData, PlyElement
    m = params["means"].detach().cpu().numpy()
    dc = params["sh0"].detach().cpu().numpy()[:, 0, :]
    op = params["opacities"].detach().cpu().numpy()[:, None]
    sc = params["scales"].detach().cpu().numpy()
    q = params["quats"].detach().cpu().numpy()
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.concatenate([m, np.zeros_like(m), dc, op, sc, q], 1).astype(np.float32)
    el = np.empty(len(arr), dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        el[n] = arr[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def export_points_ply(rec, path, max_points=200000):
    from plyfile import PlyData, PlyElement
    pts = list(rec.points3D.values())
    rng = np.random.default_rng(SEED)
    if len(pts) > max_points:
        pts = [pts[i] for i in rng.choice(len(pts), max_points, replace=False)]
    el = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    xyz = np.array([p.xyz for p in pts])
    col = np.array([p.color for p in pts])
    for i, n in enumerate("xyz"):
        el[n] = xyz[:, i]
    for i, n in enumerate(("red", "green", "blue")):
        el[n] = col[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def variant_name(args):
    """flat (flat + run panos), noflat (run panos only), mlypano (flat + run panos + the
    un-thinned Mapillary panos)."""
    return "mlypano" if args.mly_panos else ("noflat" if args.no_flat else "flat")


def variant_suffix(args):
    v = variant_name(args)
    return "" if v == "flat" else f"_{v}"


def run_corner(corner, args, matcher):
    uid = corner["ramp_uid"]
    tag = uid.replace(":", "_") + variant_suffix(args)
    work = os.path.join(args.out, tag)
    os.makedirs(work, exist_ok=True)
    img_dir = os.path.join(work, "images")
    shutil.rmtree(img_dir, ignore_errors=True)
    t0 = time.time()
    kept = prepare_images(corner, args, img_dir)
    by_name = {k["name"]: k for k in kept}
    src = next(k for k in kept if k["kind"] == "pano_src")
    result = {"ramp_uid": uid, "variant": variant_name(args),
              "config": CONFIG, "n_images_by_kind": {}, "pairs": {}}
    for k in kept:
        result["n_images_by_kind"][k["kind"]] = result["n_images_by_kind"].get(k["kind"], 0) + 1
    recs, diag = run_sfm(kept, img_dir, work, matcher)
    result["sfm"] = diag
    rec, rec_idx = None, None
    for i, r in recs.items():
        if src["name"] in {r.image(j).name for j in r.reg_image_ids()}:
            rec, rec_idx = r, i
            break
    oths = [k for k in kept if k["kind"] == "pano_oth"]
    if rec is None:
        result["status"] = "src_not_registered"
        for o in oths:
            result["pairs"][o["pair_id"]] = {"reason": "src_not_registered"}
        log(uid, "source view not registered; models", diag["model_sizes"])
        return finish(result, work, t0)
    reg = {rec.image(j).name: rec.image(j) for j in rec.reg_image_ids()}
    rec_dir = os.path.join(work, "sparse", str(rec_idx))
    rec.write(rec_dir)
    reg_kind = {}
    for nm in reg:
        kd = by_name[nm]["kind"]
        reg_kind[kd] = reg_kind.get(kd, 0) + 1
    # prior residual: registered camera centre vs its ENU prior (horizontal)
    res_h = [float(np.hypot(*(cam_of(im)[1][:2] - np.array([by_name[nm]["e"], by_name[nm]["n"]]))))
             for nm, im in reg.items()]
    result["model"] = {
        "n_registered": len(reg), "registered_by_kind": reg_kind,
        "n_points3d": rec.num_points3D(),
        "mean_reproj_px": round(float(rec.compute_mean_reprojection_error()), 3),
        "prior_residual_h_m_median": round(float(np.median(res_h)), 2),
        "prior_residual_h_m_p90": round(float(np.percentile(res_h, 90)), 2)}
    result["cameras"] = []
    for nm, im in sorted(reg.items()):
        R_cw, C = cam_of(im)
        k = by_name[nm]
        cam = rec.camera(im.camera_id)
        result["cameras"].append({
            "name": nm, "kind": k["kind"], "id": k["id"], "pair_id": k.get("pair_id", ""),
            "date": k["date"], "sequence": k["sequence"], "R_cw": R_cw.round(6).tolist(),
            "C": C.round(4).tolist(), "prior_enu": [k["e"], k["n"], k["z_prior"]],
            "width": int(cam.width), "height": int(cam.height),
            "params": [round(float(p), 4) for p in cam.params], "model": "RADIAL"})
    s_im = reg[src["name"]]
    R_s, C_s = cam_of(s_im)
    K_s = K_of(rec.camera(s_im.camera_id))
    lifts = {}
    X, d = lift_sparse(rec, s_im, src)
    lifts["sparse"] = (X, d)
    dense = os.path.join(work, "dense")
    want_views = {src["name"]} | {o["name"] for o in oths}
    if not args.no_gs or args.colmap:
        undistort(rec_dir, img_dir, dense)
    if args.colmap:
        t_mvs = run_mvs(dense, args.colmap.split(), args.mvs_max_size, src["name"])
        dm = os.path.join(dense, "stereo", "depth_maps", src["name"] + ".photometric.bin")
        dd = {"t_mvs_s": t_mvs}
        if t_mvs is not None and os.path.exists(dm):
            D = read_colmap_array(dm)
            sc = D.shape[1] / H.VIEW_W
            z = window_median(D, int(H.VIEW_W / 2 * sc), int(H.VIEW_H / 2 * sc))
            dd["depth_valid_frac_src"] = round(float(np.mean(D > 0)), 3)
            lifts["mvs"] = lift_from_depth(R_s, C_s, K_s, z, dd)
        else:
            lifts["mvs"] = (None, dd)
    if not args.no_gs:
        ply = os.path.join(work, "splat.ply")
        renders, gd = train_gs(dense, ply, os.path.join(work, "renders"), want_views)
        result["gs"] = gd
        if src["name"] in renders:
            _, D, Kr, _, _ = renders[src["name"]]
            sc = D.shape[1] / H.VIEW_W
            z = window_median(D, int(H.VIEW_W / 2 * sc), int(H.VIEW_H / 2 * sc))
            lifts["gs"] = lift_from_depth(R_s, C_s, K_s, z, {})
        else:
            lifts["gs"] = (None, {"reason": "src_not_rendered"})
    export_points_ply(rec, os.path.join(work, "points.ply"))
    result["lifts"] = {m: {"X": None if X is None else [round(float(v), 4) for v in X], **d}
                       for m, (X, d) in lifts.items()}
    for o in oths:
        pr = {}
        if o["name"] not in reg:
            pr = {m: {"reason": "oth_not_registered"} for m in lifts}
        else:
            for m, (X, d) in lifts.items():
                pr[m] = transfer(rec, X, reg[o["name"]], o) if X is not None else \
                    {"reason": d.get("reason", "no_lift")}
        result["pairs"][o["pair_id"]] = pr
    result["status"] = "ok"
    log(uid, json.dumps(result["model"]), {m: (v["X"] is not None) for m, v in result["lifts"].items()})
    return finish(result, work, t0)


def finish(result, work, t0):
    result["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(work, "result.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(H.rnd(result, 6), indent=1, sort_keys=True) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corner", nargs="*", default=[], help="ramp uids (default: all)")
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--views", required=True, help="the harness's cut views")
    ap.add_argument("--archive-root", required=True, help="labeler runs with full-res panos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-flat", action="store_true", help="control: pano views only")
    ap.add_argument("--mly-panos", action="store_true",
                    help="variant: also the un-thinned Mapillary panos (kind mly_pano)")
    ap.add_argument("--mly-pano-dir", default="", help="full-res Mapillary panos")
    ap.add_argument("--no-gs", action="store_true")
    ap.add_argument("--colmap", default="", help="command for CUDA COLMAP (enables MVS)")
    ap.add_argument("--mvs-max-size", type=int, default=1600)
    args = ap.parse_args(argv)
    m = F.load_manifest()
    corners = [c for c in m["corners"] if not args.corner or c["ramp_uid"] in args.corner]
    import torch
    matcher = Matcher("cuda" if torch.cuda.is_available() else "cpu")
    for c in corners:
        try:
            run_corner(c, args, matcher)
        except Exception as e:          # one corner failing must not stop the batch
            import traceback
            traceback.print_exc()
            work = os.path.join(args.out, c["ramp_uid"].replace(":", "_") +
                                variant_suffix(args))
            os.makedirs(work, exist_ok=True)
            finish({"ramp_uid": c["ramp_uid"], "status": f"error: {type(e).__name__}: {e}",
                    "variant": variant_name(args), "pairs": {}},
                   work, time.time())


if __name__ == "__main__":
    main()
