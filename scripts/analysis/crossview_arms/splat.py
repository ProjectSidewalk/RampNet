"""Sparse-view Gaussian splatting arms (#48): InstantSplat on each corner.

The heavy step is ``_splat_run.py`` (InstantSplat's own environment): per corner it runs
InstantSplat's MASt3R initialisation and joint Gaussian + pose training, renders the depth
at the source click from the source camera, and writes ``<ramp>.json``. These arms read those
files (``--extra splat_dir=DIR``) and do the lift-and-project:

* ``splat_instantsplat`` -- the trained splat: the click's 3D point (source ray at the
  rendered expected depth) projected into the other view's optimised camera.
* ``splat_instantsplat_init`` -- the same run read after its first iteration, i.e.
  essentially MASt3R's multi-view global alignment point cloud and poses. Against
  ``splat_instantsplat`` this isolates what the splat training adds; against
  ``mast3r_pair`` it isolates multi-view global alignment vs one pair.

An arm falls back when the corner has no output, InstantSplat failed, the click's rendered
opacity is ~0, or the point is behind the other camera. No answer is read.
"""
import json
import os

import numpy as np

import crossview_align_48 as H
from crossview_arms import _mv3d as M
from crossview_arms._registry import register

CONFIG = {"method": "InstantSplat (NVlabs, b951567): init_geo.py --focal_avg --co_vis_dsp "
                    "--conf_aware_ranking --infer_video; train.py -r 1 --pp_optimizer "
                    "--optim_pose, 1000 iterations",
          "views": "source, every pair's other view, then the nearest captures; up to 12",
          "lift": "expected depth rendered at the source click (Gaussians coloured by "
                  "camera-space depth, normalised by accumulated opacity)",
          "intrinsics": "InstantSplat's own shared focal (not the known one)"}


def _corner(ctx, pair):
    d = M.extra(ctx, "splat_dir")
    if not d:
        raise SystemExit("this arm needs --extra splat_dir=DIR (from _splat_run.py)")
    path = os.path.join(d, pair["ramp_uid"].replace(":", "_") + ".json")
    key = ("splat", path)
    if key not in ctx.cache:
        ctx.cache[key] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
    return ctx.cache[key]


def _transfer(pair, ctx, stage):
    res = _corner(ctx, pair)
    out = {"x": None, "y": None}
    if res is None:
        out["reason"] = "no_splat_output"
        return out
    if res.get("returncode"):
        out["reason"] = "instantsplat_failed"
        return out
    st = res.get(stage) or {}
    if "error" in st or st.get("click_depth") is None:
        out["reason"] = "no_depth_at_click"
        out["error"] = st.get("error")
        return out
    names = [v["name"] for v in res["views"]]
    j = next((i for i, v in enumerate(res["views"]) if v.get("pair_id") == pair["pair_id"]
              and v["role"] == "oth"), None)
    if j is None or names[j] not in st["names"]:
        out["reason"] = "other_view_not_in_run"
        return out
    k = st["names"].index(names[j])
    w2c = np.asarray(st["w2c"][k])
    X = np.asarray(st["X"])
    p = w2c[:3, :3] @ X + w2c[:3, 3]
    out.update({"click_depth": st["click_depth"], "click_alpha": st["click_alpha"],
                "n_views": res["n_views"],
                "psnr_train_median": float(np.median(st["psnr_train_views"]))
                if st.get("psnr_train_views") else None})
    # relative pose vs the prior (source is view 0)
    c2w = [np.linalg.inv(np.asarray(st["w2c"][i])) for i in (0, k)]
    views = M.corner_for(ctx, pair)["views"]
    vs = views[0]
    vo = next(v for v in views if v["role"] == "oth" and v["pair_id"] == pair["pair_id"])
    from crossview_arms.sfm import rel_pose_agreement
    out.update(rel_pose_agreement(c2w[0][:3, :3], c2w[0][:3, 3], c2w[1][:3, :3], c2w[1][:3, 3],
                                  vs, vo))
    if p[2] <= 1e-6:
        out["reason"] = "behind_other"
        return out
    s = H.VIEW_W / st["W"]
    u = (st["fx"] * p[0] / p[2] + st["cx"]) * s
    v = (st["fy"] * p[1] / p[2] + st["cy"]) * s
    x, y = M.view_pixel_to_pano(vo, u, v)
    out.update({"x": x, "y": y, "u": u, "v": v})
    return out


@register("splat_instantsplat", needs=(), config={**CONFIG, "stage": "trained (1000 it)"},
          description="InstantSplat per corner: rendered depth at the click, optimised cameras")
def splat_instantsplat(pair, ctx):
    return _transfer(pair, ctx, "final")


@register("splat_instantsplat_init", needs=(), config={**CONFIG, "stage": "after iteration 1"},
          description="InstantSplat's MASt3R multi-view initialisation, before splat training")
def splat_instantsplat_init(pair, ctx):
    return _transfer(pair, ctx, "init")
