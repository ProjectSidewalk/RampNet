"""InstantSplat per corner (#48): the heavy step behind the ``splat_*`` arms (``splat.py``).

Runs in InstantSplat's own environment (it needs its CUDA rasterizer), NOT the repo's:

    INSTANTSPLAT=/path/to/InstantSplat          # NVlabs/InstantSplat checkout, built
    $SPLAT_PY scripts/analysis/crossview_arms/_splat_run.py --instantsplat $INSTANTSPLAT \\
        --views VIEWS --corner-views CORNER_VIEWS --out SPLAT_OUT [--ramps richmond:180,...]

For each corner (one per ramp in the frozen pair list) it

1. copies the corner's views into ``<work>/<ramp>/images`` in a fixed order -- the source
   view first, then every pair's other view, then the nearest other captures, up to
   ``MAX_VIEWS`` -- so an image's index is known;
2. runs InstantSplat's ``init_geo.py`` (MASt3R on every view pair, global alignment, shared
   focal, co-visibility downsampling) and ``train.py`` (3D Gaussians and camera poses
   optimised jointly, ``ITERS`` iterations), both unmodified, with the settings of its
   ``scripts/run_infer.sh``;
3. renders expected depth and accumulated opacity at the source click from the source
   camera -- once for the Gaussians saved after the first iteration ("init": essentially the
   MASt3R global-alignment point cloud) and once for the trained ones ("final") -- with
   InstantSplat's own ``render()``, colouring each Gaussian by its camera-space depth;
4. writes ``<out>/<ramp_uid>.json``: the view order, the intrinsics, every camera's
   world-to-camera pose for both stages, and the click's 3D point (source ray at the
   rendered depth) for both stages. A per-view photometric fit (PSNR on the training views)
   is recorded too: a splat that only memorises its inputs shows high PSNR and unchanged
   geometry.

No answer is read: the inputs are the corner manifest and the views.
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
sys.path.insert(0, os.path.dirname(HERE))

MAX_VIEWS = 12
ITERS = 1000


def corner_views(corner):
    """Source, every pair's other view, then the nearest captures, up to MAX_VIEWS."""
    vs = corner["views"]
    src = [v for v in vs if v["role"] == "src"]
    oth = [v for v in vs if v["role"] == "oth"]
    rest = sorted((v for v in vs if v["role"] == "extra"), key=lambda v: (v["dist_m"], v["pano"]))
    return (src + oth + rest)[:MAX_VIEWS]


def run(cmd, cwd, log):
    # InstantSplat's MASt3R checkpoint is a full pickle (argparse.Namespace inside), which
    # torch >= 2.6 refuses under its new weights_only default; it is the published file
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def render_click(root, model_path, iteration, n_views):
    """Expected depth / opacity at the click in view 0, per-view PSNR, and the poses."""
    sys.path.insert(0, root)
    import torch
    from gaussian_renderer import render, GaussianModel
    from utils.pose_utils import get_tensor_from_camera, get_camera_from_tensor
    from utils.graphics_utils import getProjectionMatrix
    from scene.colmap_loader import read_intrinsics_text, read_extrinsics_text
    sparse = os.path.join(model_path.replace("/model", "/scene"), f"sparse_{n_views}", "0")
    cams = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    exts = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    names = [exts[k].name for k in sorted(exts)]
    cam = cams[exts[sorted(exts)[0]].camera_id]
    W, H = cam.width, cam.height
    click_small = (W / 2.0, H / 2.0)       # the view centre, in the camera's own pixel frame
    fx, fy, cx, cy = (cam.params[0], cam.params[0], cam.params[1], cam.params[2]) \
        if cam.model == "SIMPLE_PINHOLE" else tuple(cam.params[:4])
    g = GaussianModel(3)
    g.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    poses = np.load(os.path.join(model_path, "pose", f"ours_{iteration}", "pose_optimized.npy"))

    class Pipe:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    class View:
        pass

    v = View()
    v.image_width, v.image_height = W, H
    v.FoVx, v.FoVy = 2 * math.atan(W / (2 * fx)), 2 * math.atan(H / (2 * fy))
    v.projection_matrix = getProjectionMatrix(0.01, 100.0, v.FoVx, v.FoVy).transpose(0, 1).cuda()
    bg = torch.zeros(3, device="cuda")
    xyz = g.get_xyz.detach()
    homo = torch.cat([xyz, torch.ones(len(xyz), 1, device="cuda")], 1)
    out = {"W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "names": names,
           "w2c": poses.tolist(), "n_gaussians": int(len(xyz))}
    with torch.no_grad():
        pose0 = get_tensor_from_camera(torch.from_numpy(poses[0]).float().cuda())
        z = (get_camera_from_tensor(pose0) @ homo.T).T[:, 2:3]
        dimg = render(v, g, Pipe, bg, override_color=z.repeat(1, 3), camera_pose=pose0)["render"][0]
        aimg = render(v, g, Pipe, bg, override_color=torch.ones_like(z).repeat(1, 3),
                      camera_pose=pose0)["render"][0]
        u, w = click_small[0] - 0.5, click_small[1] - 0.5      # pixel-centre coordinates
        x0, y0 = int(math.floor(u)), int(math.floor(w))
        fx_, fy_ = u - x0, w - y0

        def samp(img):
            a = img[y0, x0] * (1 - fx_) + img[y0, x0 + 1] * fx_
            b = img[y0 + 1, x0] * (1 - fx_) + img[y0 + 1, x0 + 1] * fx_
            return float(a * (1 - fy_) + b * fy_)

        acc, dsum = samp(aimg), samp(dimg)
        out["click_alpha"] = acc
        out["click_depth"] = dsum / acc if acc > 1e-3 else None
        if out["click_depth"]:
            ray = np.array([(click_small[0] - cx) / fx, (click_small[1] - cy) / fy, 1.0])
            p_cam = out["click_depth"] * ray
            w2c = poses[0]
            out["X"] = (np.linalg.inv(w2c) @ np.append(p_cam, 1.0))[:3].tolist()
        # photometric fit on the training views: overfitting shows up as high PSNR here
        from PIL import Image
        psnr = []
        img_dir = os.path.join(model_path.replace("/model", "/scene"), "images")
        for i, nm in enumerate(names):
            pose = get_tensor_from_camera(torch.from_numpy(poses[i]).float().cuda())
            img = render(v, g, Pipe, bg, camera_pose=pose)["render"].clamp(0, 1)
            p = os.path.join(img_dir, nm)
            if not os.path.exists(p):
                continue
            gt = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((W, H)),
                                             dtype=np.float32) / 255.0).permute(2, 0, 1).cuda()
            mse = float(((img - gt) ** 2).mean())
            psnr.append(10 * math.log10(1.0 / max(mse, 1e-10)))
        out["psnr_train_views"] = psnr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instantsplat", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--views", required=True)
    ap.add_argument("--corner-views", required=True)
    ap.add_argument("--work", required=True, help="scratch dir (local disk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ramps", default="", help="comma-separated ramp_uids (default: all)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--keep", action="store_true", help="keep the scene / model dirs")
    args = ap.parse_args()
    from crossview_arms import _mv3d as M
    m = M.load_manifest(args.manifest)
    want = set(filter(None, args.ramps.split(",")))
    os.makedirs(args.out, exist_ok=True)
    py = sys.executable
    for c in m["corners"]:
        if want and c["ramp_uid"] not in want:
            continue
        uid = c["ramp_uid"]
        dest = os.path.join(args.out, uid.replace(":", "_") + ".json")
        if os.path.exists(dest):
            continue
        t0 = time.time()
        views = corner_views(c)
        base = os.path.join(args.work, uid.replace(":", "_"))
        scene, model = os.path.join(base, "scene"), os.path.join(base, "model")
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(os.path.join(scene, "images"))
        order = []
        for i, v in enumerate(views):
            d = args.corner_views if v["role"] == "extra" else args.views
            name = f"{i:02d}.jpg"
            shutil.copyfile(os.path.join(d, v["view"]), os.path.join(scene, "images", name))
            order.append({"name": name, "pano": v["pano"], "role": v["role"], "view": v["view"],
                          "pair_id": v.get("pair_id"), "cx": v["cx"], "cy": v["cy"]})
        n = len(views)
        res = {"ramp_uid": uid, "views": order, "n_views": n, "iters": args.iters}
        rc = run([py, "-W", "ignore", "init_geo.py", "-s", scene, "-m", model, "--n_views", str(n),
                  "--focal_avg", "--co_vis_dsp", "--conf_aware_ranking", "--infer_video"],
                 args.instantsplat, os.path.join(base, "init_geo.log"))
        t1 = time.time()
        if rc == 0:
            rc = run([py, "train.py", "-s", scene, "-m", model, "-r", "1", "--n_views", str(n),
                      "--iterations", str(args.iters), "--pp_optimizer", "--optim_pose",
                      "--save_iterations", "1"],
                     args.instantsplat, os.path.join(base, "train.log"))
        t2 = time.time()
        res.update({"init_s": round(t1 - t0, 2), "train_s": round(t2 - t1, 2), "returncode": rc})
        if rc == 0:
            for stage, it in (("init", 1), ("final", args.iters)):
                try:
                    res[stage] = render_click(args.instantsplat, model, it, n)
                except Exception as e:           # recorded, not fatal: the arm falls back
                    res[stage] = {"error": repr(e)}
        res["total_s"] = round(time.time() - t0, 2)
        with open(dest, "w", encoding="utf-8", newline="") as f:
            json.dump(res, f, indent=1)
        print(f"{uid}: {n} views, rc {rc}, init {res['init_s']} s, train {res['train_s']} s", flush=True)
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
