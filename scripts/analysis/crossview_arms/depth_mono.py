"""Monocular metric depth arms: the source click's range from a depth model, not flat ground.

Today's projection raycasts the source GT click to flat ground at an assumed camera height, and
``proj_height_auto`` does the same at the labeler's per-rig height. These arms keep the source
bearing and replace only the RANGE, with a monocular metric depth model's reading at the click.
The world point is then placed in the other view exactly as ``proj_height_auto`` places it (flat
ground at the other pano's 'auto' height, the labeler's inverse). So an arm here differs from
``proj_height_auto`` in the source range and nothing else, and the paired comparison against it
isolates what depth adds.

The depth values come from ``scripts/analysis/crossview_depth_48.py extract`` (GPU, makelab2),
committed per model as ``analysis_out/crossview_align_48/depth/<model>.jsonl``: one row per
unique source click, read by ``pair_id``. Neither file carries an answer column.

Variants, for each model (``da3``, ``depthpro``, ``unidepth``, ``metric3d``):

* ``mono_<m>_point`` (a): the model's horizontal range at the click (7x7 median).
* ``mono_<m>_plane`` (b): a robust plane fitted to the model's 3-D points within 2.5 m of the
  click, intersected with the click ray. Falls back when the plane fit fails.
* ``mono_<m>_hcal`` (c): (a) rescaled by known / model camera height, where the model's camera
  height is its own ring ground fit (#101's method) and the known height is the labeler's
  'auto' height for the source pano (GSV per-rig 2.0 / 2.5 m; Mapillary 2.6 m). Falls back when
  the ring fit fails.
* ``mono_<m>_plane_hcal``: (b) with (c)'s rescale.
* ``mono_da3_point_k101``: (a) for DA3 divided by #101's pooled DA3/Google ratio k_point =
  1.1059 (``docs/da3_calibration_101.md`` §1), i.e. DA3 on Google's depth frame.

Every arm falls back (returns no point; the harness scores the 2.6 m projection) when its range
is missing or outside 0.5-60 m, like ``proj_gsv_depth``. Diagnostics per row: the range used,
flat-ground range at the auto height, and, on GSV, Google's depth-map range at the click
(``depth.ground_range_at``), which is the independent range check.
"""
import json
import math
import os

import crossview_align_48 as H
from crossview_arms._registry import register
from crossview_arms import geometry as G

DEPTH_DIR = os.path.join(H.OUT, "depth")
MODELS = ("da3", "depthpro", "unidepth", "metric3d")
K_POINT_101 = 1.1059          # docs/da3_calibration_101.md §1: pooled GSV DA3/Google, points
RANGE_OK = (0.5, 60.0)


def _rows(ctx, model):
    key = ("mono_rows", model)
    if key not in ctx.cache:
        path = os.path.join(DEPTH_DIR, f"{model}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        by_pair = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    for pid in r["pair_ids"]:
                        by_pair[pid] = r
        ctx.cache[key] = by_pair
    return ctx.cache[key]


def _google_range(ctx, pair):
    if pair["imagery"] != "gsv":
        return None
    payload = G._payload(ctx, pair["city"], pair["src_pano"])
    if payload is None:
        return None
    import depth as depthlib
    return depthlib.ground_range_at(payload, pair["src_x"], pair["src_y"])


def place_at_range(ctx, pair, d, diag):
    """Raycast the source click to horizontal range ``d`` and place it in the other view at the
    labeler's 'auto' height. The shared second half of every arm here."""
    L = ctx.labeler()
    slims, height, _ = G.at_height(ctx, pair["city"], L.fs.HEIGHT_AUTO)
    s, o = slims.get(pair["src_pano"]), slims.get(pair["oth_pano"])
    if s is None or o is None:
        return {"x": None, "y": None, "missing_pano": True, **diag}
    ps, po = L.fs.pano_pose(s, "off"), L.fs.pano_pose(o, "off")
    h_auto = L.geo.camera_height_for(ps, camera_height=height)[0]
    dep = (pair["src_y"] - 0.5) * math.pi
    diag = {**diag, "h_auto_src": h_auto,
            "range_flat_auto_m": h_auto / math.tan(dep) if dep > 0 else None,
            "range_google_m": _google_range(ctx, pair)}
    if d is None or not (RANGE_OK[0] < d < RANGE_OK[1]) or dep <= 0:
        return {"x": None, "y": None, "range_rejected": d, **diag}
    g = G.forward(L, ps, pair["src_x"], pair["src_y"], d * math.tan(dep))
    if g is None:
        return {"x": None, "y": None, "source_above_horizon": True, **diag}
    xy = G.inverse(L, po, g.lat, g.lng, height)
    if xy is None:
        return {"x": None, "y": None, "no_inverse": True, **diag}
    return {"x": xy[0], "y": xy[1], "range_used_m": d, **diag}


def _h_auto(ctx, pair):
    L = ctx.labeler()
    slims, height, _ = G.at_height(ctx, pair["city"], L.fs.HEIGHT_AUTO)
    s = slims.get(pair["src_pano"])
    return None if s is None else L.geo.camera_height_for(L.fs.pano_pose(s, "off"),
                                                          camera_height=height)[0]


def _arm(model, variant):
    def fn(pair, ctx):
        r = _rows(ctx, model).get(pair["pair_id"])
        if r is None or r.get("status") != "ok":
            return {"x": None, "y": None, "depth_status": None if r is None else r.get("status")}
        diag = {"range_point_m": r["range_point_m"], "range_plane_m": r["range_plane_m"],
                "h_model": r["ground"]["h"], "ground_ok": r["ground"]["ok"],
                "local_ok": r["local"]["ok"]}
        base = r["range_plane_m"] if variant in ("plane", "plane_hcal") else r["range_point_m"]
        d = base
        if variant == "point_k101":
            d = None if base is None else base / K_POINT_101
        elif variant in ("hcal", "plane_hcal"):
            h_known = _h_auto(ctx, pair)
            if not r["ground"]["ok"] or not r["ground"]["h"] or h_known is None or base is None:
                d = None
            else:
                scale = h_known / r["ground"]["h"]
                diag["hcal_scale"] = scale
                d = base * scale
        return place_at_range(ctx, pair, d, diag)
    return fn


_DESC = {"point": "range = the model's metric depth at the click (7x7 median)",
         "plane": "range = click ray meets a robust plane fitted within 2.5 m of the click",
         "hcal": "point range x (labeler auto camera height / the model's ring ground-fit height)",
         "plane_hcal": "plane range x (labeler auto camera height / the model's ring ground-fit height)",
         "point_k101": "DA3 point range / 1.1059 (#101's DA3-to-Google scale)"}

for _m in MODELS:
    for _v in ("point", "plane", "hcal", "plane_hcal"):
        register(f"mono_{_m}_{_v}", needs=("labeler", "panos"),
                 description=f"{_m}: {_DESC[_v]}; other view flat at its 'auto' height",
                 config={"model": _m, "variant": _v, "depth_rows": f"depth/{_m}.jsonl",
                         "other_view": "flat ground at the labeler's 'auto' height",
                         "range_ok_m": list(RANGE_OK)})(_arm(_m, _v))
register("mono_da3_point_k101", needs=("labeler", "panos"),
         description=f"da3: {_DESC['point_k101']}; other view flat at its 'auto' height",
         config={"model": "da3", "variant": "point_k101", "k_point": K_POINT_101,
                 "depth_rows": "depth/da3.jsonl",
                 "other_view": "flat ground at the labeler's 'auto' height",
                 "range_ok_m": list(RANGE_OK)})(_arm("da3", "point_k101"))
