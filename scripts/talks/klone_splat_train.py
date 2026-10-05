"""Minimal 3D Gaussian splatting of one corner with gsplat, from the COLMAP text model that
``uchicago_2026_dense_scene.py`` wrote (posed faces + MapAnything points), then render the
camera path from ``uchicago_2026_mesh.py path``.

    python klone_splat_train.py train  --data DATA/richmond_99/colmap --out OUT/richmond_99 [--iters 15000]
    python klone_splat_train.py render --out OUT/richmond_99 --path DATA/richmond_99/path.json

Dependencies: torch, gsplat (>= 1.4), numpy, pillow, tqdm. Written against gsplat 1.4's
``rasterization`` and ``DefaultStrategy``; the loss is L1 + 0.2 (1 - SSIM) as in the paper,
spherical harmonics degree 0 (one colour per Gaussian), no camera optimisation (poses are
MapAnything's refinement of the GPS + compass priors). This is a plain re-implementation of
the standard recipe, enough for a talk video, not a benchmarked trainer.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------
# COLMAP text model
# ---------------------------------------------------------------------------------------------

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_colmap(root):
    sp = os.path.join(root, "sparse", "0")
    cams = {}
    for line in open(os.path.join(sp, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        assert p[1] == "PINHOLE", p[1]
        fx, fy, cx, cy = map(float, p[4:8])
        cams[int(p[0])] = dict(w=int(p[2]), h=int(p[3]), K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    images = []
    lines = [l for l in open(os.path.join(sp, "images.txt"), encoding="utf-8") if not l.startswith("#")]
    for l in lines:
        p = l.split()
        if len(p) < 10:
            continue
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])))
        Rwc = quat_to_rot(q)
        Twc = np.eye(4)
        Twc[:3, :3], Twc[:3, 3] = Rwc, t
        images.append(dict(name=p[9], cam=cams[int(p[8])], Twc=Twc))
    pts, cols = [], []
    for l in open(os.path.join(sp, "points3D.txt"), encoding="utf-8"):
        if l.startswith("#"):
            continue
        p = l.split()
        pts.append([float(p[1]), float(p[2]), float(p[3])])
        cols.append([int(p[4]), int(p[5]), int(p[6])])
    return images, np.array(pts, np.float32), np.array(cols, np.float32) / 255.0


def load_images(root, images, device):
    from PIL import Image
    out = []
    for im in images:
        img = np.asarray(Image.open(os.path.join(root, "images", im["name"])).convert("RGB"), np.float32) / 255.0
        out.append(torch.from_numpy(img).to(device))
    return out


# ---------------------------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------------------------

def _gauss(win=11, sigma=1.5, device="cuda"):
    x = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(a, b, win=None):
    """a, b: [1, 3, H, W] in [0, 1]."""
    if win is None:
        win = _gauss(device=a.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    s_aa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


# ---------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------

def knn_scale(pts, k=4):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=k)
    return np.clip(d[:, 1:].mean(1), 1e-3, None)


def cmd_train(args):
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    dev = "cuda"
    images, pts, cols = read_colmap(args.data)
    gts = load_images(args.data, images, dev)
    print(f"{len(images)} images, {len(pts):,} init points", flush=True)
    centers = np.array([np.linalg.inv(im["Twc"])[:3, 3] for im in images])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(pts).to(dev)),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(knn_scale(pts).astype(np.float32)).to(dev))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(cols).to(dev) - 0.5) / 0.28209479177387814)[:, None, :]),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    opts = {k: torch.optim.Adam([{"params": [v], "lr": lrs[k], "name": k}], eps=1e-15) for k, v in params.items()}
    strategy = DefaultStrategy(verbose=False, refine_start_iter=500, refine_stop_iter=int(args.iters * 0.6),
                               reset_every=3000, refine_every=100, prune_opa=0.005, grow_grad2d=0.0002,
                               grow_scale3d=0.01)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    win = _gauss(device=dev)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(args.iters):
        i = int(rng.integers(len(images)))
        im, gt = images[i], gts[i]
        H, W = gt.shape[:2]
        viewmat = torch.from_numpy(im["Twc"]).float().to(dev)[None]
        K = torch.from_numpy(im["cam"]["K"]).float().to(dev)[None]
        render, alpha, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"], viewmat, K, W, H,
            sh_degree=0, packed=False, absgrad=False)
        strategy.step_pre_backward(params, opts, state, step, info)
        pred = render[0].clamp(0, 1)
        l1 = (pred - gt).abs().mean()
        s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win)
        loss = 0.8 * l1 + 0.2 * (1 - s)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 500 == 0 or step == args.iters - 1:
            print(f"step {step} loss {loss.item():.4f} l1 {l1.item():.4f} ssim {s.item():.3f} "
                  f"gaussians {len(params['means']):,} {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.detach().cpu() for k, v in params.items()}, os.path.join(args.out, "gaussians.pt"))
    with open(os.path.join(args.out, "train.json"), "w") as f:
        json.dump(dict(iters=args.iters, n_images=len(images), n_init=N, n_final=len(params["means"]),
                       scene_scale=scene_scale, seconds=time.time() - t0,
                       gpu=torch.cuda.get_device_name(0)), f, indent=1)
    print("saved", os.path.join(args.out, "gaussians.pt"), flush=True)


# ---------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------

def cmd_render(args):
    from gsplat import rasterization
    from PIL import Image
    dev = "cuda"
    p = torch.load(os.path.join(args.out, "gaussians.pt"))
    p = {k: v.to(dev) for k, v in p.items()}
    with open(args.path, encoding="utf-8") as f:
        path = json.load(f)
    W, H = path["width"], path["height"]
    K = torch.tensor(path["K"], dtype=torch.float32, device=dev)[None]
    fdir = os.path.join(args.out, "frames")
    os.makedirs(fdir, exist_ok=True)
    bg = torch.tensor([[0.07, 0.08, 0.11]], device=dev)
    for i, fr in enumerate(path["frames"]):
        if i % args.step:
            continue
        Twc = torch.from_numpy(np.linalg.inv(np.array(fr["T"]))).float().to(dev)[None]
        with torch.no_grad():
            render, alpha, _ = rasterization(p["means"], p["quats"], torch.exp(p["scales"]),
                                             torch.sigmoid(p["opacities"]), p["sh0"], Twc, K, W, H,
                                             sh_degree=0, backgrounds=bg)
        img = (render[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(fdir, f"frame_{i:04d}.png"))
        if i % 60 == 0:
            print("frame", i, flush=True)
    print("frames ->", fdir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=15000)
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--step", type=int, default=1)
    args = ap.parse_args(argv)
    {"train": cmd_train, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
