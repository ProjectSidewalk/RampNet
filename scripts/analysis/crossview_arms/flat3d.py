"""Arms that read the per-corner 3D reconstructions from flat Mapillary imagery (#214).

The heavy work (SfM over flat Mapillary images + the pano views, Gaussian splatting, MVS)
is done by ``scripts/analysis/flat3d/reconstruct.py``, one corner per Richmond ramp in the
harness pairs. Each corner writes a ``result.json`` holding, per pair, the source click
lifted to 3D and projected into the other view; those files are committed under
``analysis_out/flat_mapillary_3d/corners/`` and these arms only read them. So ``predict``
for these arms runs from a clean clone.

Only Richmond (Mapillary) pairs are eligible: the GSV cities have no flat imagery. Every
GSV pair, and every Richmond pair whose corner did not reconstruct / co-register / lift,
falls back to the projection (the reason is kept in the row).

    python scripts/analysis/crossview_align_48.py predict --arm flat_sfm
"""
import json
import os

import crossview_align_48 as H
from crossview_arms._registry import register

CORNERS = os.path.join(H.OUT_ROOT, "flat_mapillary_3d", "corners")


def _corner(ctx, uid, variant):
    key = ("flat3d", uid, variant)
    if key not in ctx.cache:
        name = uid.replace(":", "_") + ("" if variant == "flat" else f"_{variant}") + ".json"
        path = os.path.join(CORNERS, name)
        ctx.cache[key] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
    return ctx.cache[key]


def _arm(pair, ctx, lift, variant="flat"):
    if pair["city"] != "richmond":
        return {"reason": "not_mapillary"}
    c = _corner(ctx, pair["ramp_uid"], variant)
    if c is None:
        return {"reason": "corner_not_run"}
    if c.get("status") != "ok":
        return {"reason": c.get("status", "no_model")}
    r = (c.get("pairs", {}).get(pair["pair_id"]) or {}).get(lift)
    if r is None:
        return {"reason": f"no_{lift}"}
    out = dict(r)
    out.setdefault("x", None)
    out.setdefault("y", None)
    m = c.get("model", {})
    out["n_registered"] = m.get("n_registered")
    return out


_CFG = {"reconstruction": "scripts/analysis/flat3d/reconstruct.py (config in each corner's "
                          "result.json)", "corners": "analysis_out/flat_mapillary_3d/corners/"}


@register("flat_sfm", config={**_CFG, "lift": "sparse, gravity-level ground at the click"},
          description="SfM over flat Mapillary images + pano views; click lifted onto the "
                      "sparse ground, projected into the other view (Richmond only)")
def flat_sfm(pair, ctx):
    return _arm(pair, ctx, "sparse")


@register("flat_gs", config={**_CFG, "lift": "3DGS expected depth at the click"},
          description="the same model, click depth rendered from a Gaussian splat "
                      "(Richmond only)")
def flat_gs(pair, ctx):
    return _arm(pair, ctx, "gs")


@register("flat_mvs", config={**_CFG, "lift": "COLMAP patch-match geometric depth at the click"},
          description="the same model, click depth from MVS (Richmond only)")
def flat_mvs(pair, ctx):
    return _arm(pair, ctx, "mvs")


@register("noflat_sfm", config={**_CFG, "lift": "sparse", "control": "pano views only"},
          description="control: the same pipeline with the flat images left out")
def noflat_sfm(pair, ctx):
    return _arm(pair, ctx, "sparse", "noflat")


@register("noflat_gs", config={**_CFG, "lift": "3DGS depth", "control": "pano views only"},
          description="control: GS depth with the flat images left out")
def noflat_gs(pair, ctx):
    return _arm(pair, ctx, "gs", "noflat")


@register("noflat_mvs", config={**_CFG, "lift": "MVS depth", "control": "pano views only"},
          description="control: MVS depth with the flat images left out")
def noflat_mvs(pair, ctx):
    return _arm(pair, ctx, "mvs", "noflat")


@register("mlypano_sfm", config={**_CFG, "lift": "sparse", "images": "flat + run panos + up "
                                 "to 30 un-thinned Mapillary panos within 25 m"},
          description="flat_sfm plus the un-thinned Mapillary panos near the corner")
def mlypano_sfm(pair, ctx):
    return _arm(pair, ctx, "sparse", "mlypano")


@register("mlypano_gs", config={**_CFG, "lift": "3DGS depth", "images": "as mlypano_sfm"},
          description="GS depth on the flat + run + un-thinned Mapillary pano model")
def mlypano_gs(pair, ctx):
    return _arm(pair, ctx, "gs", "mlypano")


_GSMED = {**_CFG, "lift": "3DGS median depth along the click ray (post hoc, "
                          "scripts/analysis/flat3d/gs_median_depth.py)"}


@register("flat_gsmed", config=_GSMED,
          description="flat model, splat MEDIAN depth at the click (post hoc)")
def flat_gsmed(pair, ctx):
    return _arm(pair, ctx, "gs_median")


@register("noflat_gsmed", config={**_GSMED, "control": "pano views only"},
          description="control: splat median depth with the flat images left out (post hoc)")
def noflat_gsmed(pair, ctx):
    return _arm(pair, ctx, "gs_median", "noflat")


@register("mlypano_gsmed", config={**_GSMED, "images": "as mlypano_sfm"},
          description="flat + run + Mapillary pano model, splat median depth (post hoc)")
def mlypano_gsmed(pair, ctx):
    return _arm(pair, ctx, "gs_median", "mlypano")


@register("mlypano_mvs", config={**_CFG, "lift": "MVS depth", "images": "as mlypano_sfm"},
          description="MVS depth on the flat + run + un-thinned Mapillary pano model")
def mlypano_mvs(pair, ctx):
    return _arm(pair, ctx, "mvs", "mlypano")
