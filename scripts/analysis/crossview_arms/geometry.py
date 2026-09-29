"""Geometry-prior arms: no image matching, a better camera model instead.

Every arm re-does both halves of today's projection with the labeler's own code: raycast
the source GT pixel to a world point, then place that point in the other pano. Only the
camera model changes. They need ``--labeler-root`` (a checkout with ``depth.py`` and
``fuse_sites.load_at_height``, i.e. sidewalk-auto-labeler main since #101), plus the same
``--runs-root`` / ``--results-root`` as ``pairs``.

An arm that does not apply to a pair's imagery returns ``{"x": None, "not_applicable":
True}``: it falls back to the projection and is counted as a fallback, so read these arms
by the ``imagery=`` strata.

What the Richmond run already uses (checked 2026-09-28): Mapillary's SfM position
(``computed_geometry``) and SfM heading (``computed_compass_angle``); only pitch / roll are
left out (``apply_pose`` off). So "SfM-refined pose" here means adding the SfM rotation's
pitch / roll (``proj_mly_gravity``, ``proj_mly_road``); ``proj_mly_rawgps`` is the contrast
that swaps the SfM position for the raw GPS fix. Everything is read from the stored
``source_metadata``; no Mapillary API call and no token is needed.
"""
import dataclasses
import gzip
import json
import math
import os

import numpy as np

import crossview_align_48 as H
from crossview_arms._registry import register

FLAT_M = 2.6                # the constant every committed fusion_eval report raycasts at


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _runs_root(ctx):
    return ctx.args.runs_root or os.path.join(ctx.args.labeler_root, "runs")


def at_height(ctx, city, camera_height):
    """({pano_id: SlimPano}, height, auto) for the pair list's panos, loaded through the
    labeler's one height resolver (fuse_sites.load_at_height), so 'per-pano' / 'auto' mean
    what they mean in the labeler. The depth index is the labeler checkout's
    runs/<city>/depth. The load lives in ``Context.at_height``, which withholds detections
    (the reference is one of them) and keeps the loader itself away from arms."""
    return ctx.at_height(city, camera_height)


def forward(L, pose, x, y, height, apply_pose=False):
    """The labeler's raycast, uncapped in range. geo.GroundEstimate or None (horizon)."""
    return L.geo.detection_ground_point(pose, x, y, camera_height=height, max_range_m=math.inf,
                                        errors=L.geo.error_model_for(pose.source),
                                        apply_pose=apply_pose)


def inverse(L, pose, lat, lng, height, apply_pose=False, guess=None, tol_m=1e-4):
    """Where a ground point lands in a pano: the labeler's exact inverse on the flat path,
    and a Gauss-Newton solve of ``forward`` on the posed path (the labeler has no posed
    inverse). Returns (x_norm, y_norm) or None."""
    flat = L.geo.ground_point_to_pano(pose, lat, lng, camera_height=height,
                                      max_range_m=math.inf)
    if not (apply_pose and pose.has_pitch_roll):
        return None if flat is None else (flat.x_norm, flat.y_norm)
    frame = L.geo.LocalFrame(pose.lat, pose.lng)
    te, tn = frame.to_enu(lat, lng)
    x, y = guess or ((flat.x_norm, flat.y_norm) if flat else (0.5, 0.6))

    def resid(x, y):
        g = forward(L, pose, x % 1.0, y, height, apply_pose=True)
        if g is None:
            return None
        e, n = frame.to_enu(g.lat, g.lng)
        return np.array([e - te, n - tn])

    for _ in range(50):
        r = resid(x, y)
        if r is None:
            y = min(0.999, y + 0.01)
            continue
        if np.hypot(*r) < tol_m:
            return x % 1.0, y
        eps = 1e-6
        rx, ry = resid(x + eps, y), resid(x, y + eps)
        if rx is None or ry is None:
            return None
        J = np.column_stack([(rx - r) / eps, (ry - r) / eps])
        try:
            dx, dy = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None
        step = max(1.0, abs(dy) / 0.02, abs(dx) / 0.02)     # damp steps over ~3.6 deg
        x, y = x + dx / step, min(0.999, max(0.5001, y + dy / step))
    return None


def _na():
    return {"x": None, "y": None, "not_applicable": True}


def _flat_pair(ctx, pair, slims, height, mode="off"):
    """Raycast the source GT pixel with each pano's own pose/height, place it in the other."""
    L = ctx.labeler()
    s, o = slims.get(pair["src_pano"]), slims.get(pair["oth_pano"])
    if s is None or o is None:
        return {"x": None, "y": None, "missing_pano": True}
    ps, po = L.fs.pano_pose(s, mode), L.fs.pano_pose(o, mode)
    rot_s, rot_o = mode != "off" and ps.has_pitch_roll, mode != "off" and po.has_pitch_roll
    g = forward(L, ps, pair["src_x"], pair["src_y"], height, apply_pose=rot_s)
    if g is None:
        return {"x": None, "y": None, "source_above_horizon": True}
    xy = inverse(L, po, g.lat, g.lng, height, apply_pose=rot_o,
                 guess=(pair["proj_x"], pair["proj_y"]))
    if xy is None:
        return {"x": None, "y": None, "no_inverse": True}
    hs = L.geo.camera_height_for(ps, camera_height=height)[0]
    ho = L.geo.camera_height_for(po, camera_height=height)[0]
    return {"x": xy[0], "y": xy[1], "h_src": hs, "h_oth": ho, "range_src_m": g.range_m,
            "posed_src": bool(rot_s), "posed_oth": bool(rot_o)}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


@register("proj_flat_check", needs=("labeler", "panos"),
          config={"camera_height_m": FLAT_M, "pose": "off"},
          description="instrument check: today's projection recomputed through this module")
def proj_flat_check(pair, ctx):
    slims, _, _ = at_height(ctx, pair["city"], FLAT_M)
    return _flat_pair(ctx, pair, slims, FLAT_M)


@register("proj_height_perpano", needs=("labeler", "panos"),
          config={"camera_height": "per-pano (GSV depth index via depth.believe_height; "
                                    "else 2.6)", "pose": "off"},
          description="flat ground at each pano's measured camera height (GSV); 2.6 m otherwise")
def proj_height_perpano(pair, ctx):
    L = ctx.labeler()
    slims, height, _ = at_height(ctx, pair["city"], L.geo.PER_PANO)
    return _flat_pair(ctx, pair, slims, height)


@register("proj_height_auto", needs=("labeler", "panos"),
          config={"camera_height": "auto (labeler default since #79: GSV per capture-year rig "
                                    "2.0 / 2.5 m; Mapillary 2.6)", "pose": "off"},
          description="flat ground at the labeler's 'auto' per-rig height")
def proj_height_auto(pair, ctx):
    L = ctx.labeler()
    slims, height, _ = at_height(ctx, pair["city"], L.fs.HEIGHT_AUTO)
    return _flat_pair(ctx, pair, slims, height)


def _mly(pair, ctx, mode):
    if pair["imagery"] != "mapillary":
        return _na()
    slims, _, _ = at_height(ctx, pair["city"], FLAT_M)
    return _flat_pair(ctx, pair, slims, FLAT_M, mode=mode)


@register("proj_mly_gravity", needs=("labeler", "panos"),
          config={"camera_height_m": FLAT_M, "pose": "gravity (SfM computed_rotation pitch/roll)"},
          description="Mapillary: add the SfM rotation's pitch/roll (gravity frame)")
def proj_mly_gravity(pair, ctx):
    return _mly(pair, ctx, "gravity")


@register("proj_mly_road", needs=("labeler", "panos"),
          config={"camera_height_m": FLAT_M,
                  "pose": "road (SfM pitch/roll minus the sequence's SfM road grade)"},
          description="Mapillary: SfM pitch/roll relative to the road grade (labeler 'road')")
def proj_mly_road(pair, ctx):
    return _mly(pair, ctx, "road")


@register("proj_mly_rawgps", needs=("labeler", "panos"),
          config={"camera_height_m": FLAT_M, "pose": "off",
                  "position": "raw GPS source_metadata.geometry instead of computed_geometry"},
          description="Mapillary contrast: raw GPS position instead of the SfM one")
def proj_mly_rawgps(pair, ctx):
    if pair["imagery"] != "mapillary":
        return _na()
    L = ctx.labeler()
    slims, _, _ = at_height(ctx, pair["city"], FLAT_M)
    raw = {}
    for pid in (pair["src_pano"], pair["oth_pano"]):
        geom = (ctx.pano(pair["city"], pid).get("source_metadata") or {}).get("geometry")
        if not geom:
            return {"x": None, "y": None, "no_raw_gps": True}
        lng, lat = geom["coordinates"][:2]
        raw[pid] = dataclasses.replace(slims[pid], lat=lat, lng=lng)
    return _flat_pair(ctx, pair, raw, FLAT_M)


def _payload(ctx, city, pano_id):
    """The labeler's harvested GSV depth payload for a pano (runs/<city>/depth/<id>.json.gz),
    parsed with depth.parse, or None."""
    key = ("depth", city, pano_id)
    if key not in ctx.cache:
        path = os.path.join(_runs_root(ctx), city, "depth", f"{pano_id}.json.gz")
        if not os.path.exists(path):
            ctx.cache[key] = None
        else:
            import depth as depthlib
            with gzip.open(path, "rt", encoding="utf-8") as f:
                ctx.cache[key] = depthlib.parse(json.load(f)["depth_b64"])
    return ctx.cache[key]


@register("proj_gsv_depth", needs=("labeler", "panos"),
          config={"source": "range from Google's depth planes at the GT pixel "
                            "(depth.ground_range_at, exact ray)",
                  "other": "flat ground at the other pano's measured height (per-pano)"},
          description="GSV: source range from the depth map; other view at its measured height")
def proj_gsv_depth(pair, ctx):
    if pair["imagery"] != "gsv":
        return _na()
    L = ctx.labeler()
    import depth as depthlib
    payload = _payload(ctx, pair["city"], pair["src_pano"])
    if payload is None:
        return {"x": None, "y": None, "no_depth": True}
    d = depthlib.ground_range_at(payload, pair["src_x"], pair["src_y"])
    if d is None or not (0.5 < d < 60.0):
        return {"x": None, "y": None, "depth_range": d}
    slims, height, _ = at_height(ctx, pair["city"], L.geo.PER_PANO)
    s, o = slims.get(pair["src_pano"]), slims.get(pair["oth_pano"])
    if s is None or o is None:
        return {"x": None, "y": None, "missing_pano": True}
    ps, po = L.fs.pano_pose(s, "off"), L.fs.pano_pose(o, "off")
    # the height that makes the flat raycast land at the depth map's range
    dep = (pair["src_y"] - 0.5) * math.pi
    g = forward(L, ps, pair["src_x"], pair["src_y"], d * math.tan(dep))
    if g is None:
        return {"x": None, "y": None, "source_above_horizon": True}
    xy = inverse(L, po, g.lat, g.lng, height)
    if xy is None:
        return {"x": None, "y": None, "no_inverse": True}
    return {"x": xy[0], "y": xy[1], "range_src_depth_m": d, "range_src_flat_m": pair["src_range_m"],
            "h_oth": L.geo.camera_height_for(po, camera_height=height)[0]}
