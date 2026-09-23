"""SAM2 extent vs the whole-apron gold: path 1 of #83 (RampNet 2.0 plan, item 7).

Question: given a curb-ramp *point* (the thing RampNet emits), can a point-prompted
SAM2 recover the ramp's *extent* well enough to measure it? Scored against the only
whole-apron extent gold in the ecosystem, ``benchmark/<city>/boxes.json`` (#116,
BOX_RULE v2; Richmond is complete, Annapolis / Sao Paulo / Paterson are partial
random-pano samples). ``manual_labels/`` boxes are NOT usable gold: they are
tactile-pad marks, not aprons (#114, ``docs/crop_window_eval.md``).

What SAM2 sees is the design question the 2026-09-01 comment on #83 raised: an
equirect crop is still equirect (horizontal stretch grows as 1/cos(latitude), worst
exactly on the near-field ramps), while SAM2 is trained on perspective imagery. So
every item is segmented twice from the same prompt, over the same field of view:

- ``gnomonic`` -- a rectilinear view centered on the prompt point (the in-repo
  gnomonic math of ``scripts/model_comparison/equirect_tiling.py``); the mask is
  mapped back to equirect coordinates for scoring, because the gold lives there.
- ``equirect`` -- the plain seam-wrapped equirect crop ``box_gallery.py`` cuts for
  annotation (``crop_rect`` / ``cut_crop``, reused verbatim).

and from two prompt sources:

- ``boxcenter`` -- the gold box's center (an oracle prompt: "GT-center").
- ``point`` -- the item's recorded point: on ``det:<i>`` items that is the RampNet
  detection the reviewer judged true (the end-to-end, production-realistic prompt);
  on ``missed:<i>`` items it is the reviewer's click.

An *arm* is ``<prompt>_<projection>``. Per view the image embedding is computed once
and four prompt *variants* are decoded from it: ``pt_multi`` (one positive point,
multimask, keep the highest predicted IoU -- the pre-stated headline, since a lone
point is exactly the ambiguous case SAM's multimask output exists for),
``pt_single`` (one point, single mask), and ``ptbox_multi`` / ``ptbox_single`` (the
point plus a *box prior* built only from production-available geometry -- the prompt's
depression angle and a nominal apron size on flat ground, see :func:`prior_box`; it
never sees the gold).

Every mask is reduced to its tight bounding box in pano-normalized equirect
coordinates (seam-aware) and scored against the gold box: IoU, the share of the gold
box the SAM2 box covers, and the share of the mask's pixels that fall inside the gold
box.

Subcommands::

    # GPU (makelab2): segment every boxed item, one CSV row per item x arm x fov x variant
    python scripts/analysis/sam2_extent_83.py run --city richmond \
        --arm boxcenter_gnomonic,boxcenter_equirect,point_gnomonic,point_equirect \
        --fov 90,76,60 --checkpoint /path/sam2.1_hiera_large.pt \
        --out analysis_out/sam2_extent_83

    # CPU: tables, paired projection delta with a pano-clustered bootstrap
    python scripts/analysis/sam2_extent_83.py summarize --city richmond --out analysis_out/sam2_extent_83

    # CPU + panos: worst / median contact sheets
    python scripts/analysis/sam2_extent_83.py gallery --city richmond --out analysis_out/sam2_extent_83 \
        --assets docs/assets

The geometry helpers are pure numpy and unit-tested on synthetic data
(``tests/test_sam2_extent_83.py``); SAM2 and torch are imported only inside ``run``.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from equirect_tiling import View, _camera_basis  # noqa: E402
from crop_window_eval import STRATA, depression_deg, load_bundle_boxes, stratum_label  # noqa: E402

TWO_PI = 2.0 * math.pi

PROJECTIONS = ("gnomonic", "equirect")
PROMPTS = ("boxcenter", "point")
ARMS = tuple(f"{p}_{q}" for p in PROMPTS for q in PROJECTIONS)
VARIANTS = ("pt_multi", "pt_single", "ptbox_multi", "ptbox_single")
HEADLINE_VARIANT = "pt_multi"          # stated before any number was seen
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BOOTSTRAP_SEED = 83
BOOTSTRAP_REPS = 10000
ROUND = 5

CSV_FIELDS = [
    "city", "pano_id", "key", "kind", "arm", "prompt", "projection", "fov", "variant",
    "pano_w", "pano_h", "view_px",
    "prompt_x", "prompt_y",
    "gold_cx", "gold_cy", "gold_w", "gold_h",
    "sam_cx", "sam_cy", "sam_w", "sam_h",
    "iou", "gold_frac_covered", "mask_frac_in_gold", "pred_iou",
    "mask_px", "mask_touches_edge", "depression_deg", "band",
]


# ---------------------------------------------------------------------------
# Geometry (pure numpy; the unit-tested core)

def view_for_point(x, y, fov_deg, size):
    """Square gnomonic view centered on pano-normalized ``(x, y)``.

    Yaw/pitch follow ``equirect_tiling``'s convention (lon = (x-0.5)*2pi,
    lat = (0.5-y)*pi), so the prompt point lands exactly at the view center.
    """
    return View((x - 0.5) * 360.0, (0.5 - y) * 180.0, float(fov_deg), float(fov_deg),
                int(size), int(size))


def view_uv_to_equirect(u, v, view):
    """Vectorized ``perspective_point_to_equirect``: view-normalized ``(u, v)``
    arrays (origin top-left, [0, 1]) to pano-normalized ``(X, Y)`` arrays."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a = (u * 2.0 - 1.0) * th
    b = (1.0 - v * 2.0) * tv
    dx = f[0] + a * r[0] + b * up[0]
    dy = f[1] + a * r[1] + b * up[1]
    dz = f[2] + a * r[2] + b * up[2]
    inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx * inv, dy * inv, dz * inv
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))
    return (lon / TWO_PI + 0.5) % 1.0, np.clip(0.5 - lat / math.pi, 0.0, 1.0)


def equirect_to_view_uv(X, Y, view):
    """Inverse of :func:`view_uv_to_equirect`. Returns ``(u, v, in_front)``; ``u, v``
    are unbounded (a point outside the FOV lands outside [0, 1]) and ``in_front`` is
    False for directions behind the camera, where ``u, v`` are meaningless."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    lon = (X - 0.5) * TWO_PI
    lat = (0.5 - Y) * math.pi
    cl = np.cos(lat)
    d = (cl * np.sin(lon), np.sin(lat), cl * np.cos(lon))
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    zc = d[0] * f[0] + d[1] * f[1] + d[2] * f[2]
    in_front = zc > 1e-9
    zs = np.where(in_front, zc, 1.0)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    u = ((d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / zs / th + 1.0) / 2.0
    v = (1.0 - (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / zs / tv) / 2.0
    return u, v, in_front


def render_gnomonic(src, view, chunk_rows=512):
    """Bilinear gnomonic render of an equirect ``src`` (H x W x 3 uint8 array).

    Samples at pixel centers; x wraps around the seam, y clamps at the poles. The
    in-repo ``equirect_to_perspective`` is nearest-neighbour ("sufficient for feeding
    a detector"); a segmenter's edges are the measurement here, so this one
    interpolates. Rendered in row chunks to bound memory at 3-4k px views.
    """
    sh, sw = src.shape[:2]
    W, H = view.width, view.height
    out = np.empty((H, W, src.shape[2]), dtype=np.uint8)
    u = (np.arange(W) + 0.5) / W
    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        v = (np.arange(r0, r1) + 0.5) / H
        uu, vv = np.meshgrid(u, v)
        X, Y = view_uv_to_equirect(uu, vv, view)
        sx = X * sw - 0.5
        sy = np.clip(Y * sh - 0.5, 0.0, sh - 1.0)
        x0 = np.floor(sx)
        fx = (sx - x0)[..., None]
        x0 = x0.astype(np.int64) % sw
        x1 = (x0 + 1) % sw
        y0 = np.floor(sy).astype(np.int64)
        fy = (sy - y0)[..., None]
        y1 = np.minimum(y0 + 1, sh - 1)
        top = src[y0, x0] * (1.0 - fx) + src[y0, x1] * fx
        bot = src[y1, x0] * (1.0 - fx) + src[y1, x1] * fx
        out[r0:r1] = np.clip(top * (1.0 - fy) + bot * fy + 0.5, 0, 255).astype(np.uint8)
    return out


def unwrap_x(xs, ref_x):
    """Pano-normalized x values re-expressed as signed offsets from ``ref_x`` in
    [-0.5, 0.5): the nearest representation across the seam."""
    return (np.asarray(xs, dtype=np.float64) - ref_x + 0.5) % 1.0 - 0.5


def seam_bbox(xs, ys, ref_x):
    """Tight ``(cx, cy, w, h)`` (pano-normalized) of a point set that may straddle
    the seam. x is unwrapped around ``ref_x`` (the view/crop center), so a mask that
    crosses x = 0 gets a narrow box, not one spanning the whole pano; ``cx`` is
    folded back into [0, 1)."""
    dx = unwrap_x(xs, ref_x)
    x0, x1 = float(dx.min()), float(dx.max())
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    return ((ref_x + (x0 + x1) / 2.0) % 1.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def mask_boundary(mask):
    """Boolean array: mask pixels with at least one 4-neighbour outside the mask
    (or on the image border). The bbox of a mask is decided by these alone."""
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1, constant_values=False)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~interior


def gnomonic_mask_to_equirect_bbox(mask, view):
    """Tight pano-normalized bbox of a gnomonic-view mask, or None if empty.

    Maps the four *corners* of every boundary pixel (not its center) back to the
    equirect, so the box covers the pixels' full footprint -- the same convention
    as :func:`crop_mask_to_equirect_bbox`, which uses pixel edges."""
    b = mask_boundary(mask)
    rows, cols = np.nonzero(b)
    if rows.size == 0:
        return None
    H, W = b.shape
    us = np.concatenate([cols, cols + 1, cols, cols + 1]) / W
    vs = np.concatenate([rows, rows, rows + 1, rows + 1]) / H
    X, Y = view_uv_to_equirect(us, vs, view)
    ref_x = (view.yaw_deg / 360.0 + 0.5) % 1.0
    return seam_bbox(X, Y, ref_x)


def crop_mask_to_equirect_bbox(mask, left, top, pano_w, pano_h):
    """Tight pano-normalized bbox of a mask on a seam-wrapped equirect crop whose
    top-left is native pixel ``(left, top)`` (``left`` may wrap: columns past the
    right edge are taken modulo ``pano_w``, as ``box_gallery.cut_crop`` builds it).
    Uses pixel edges, so a 1-pixel mask has a 1-pixel box. None if empty."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    c0, c1 = cols.min(), cols.max() + 1
    r0, r1 = rows.min(), rows.max() + 1
    w = (c1 - c0) / pano_w
    cx = ((left + (c0 + c1) / 2.0) / pano_w) % 1.0
    return (cx, (top + (r0 + r1) / 2.0) / pano_h, w, (r1 - r0) / pano_h)


def mask_pixels_equirect(mask, to_equirect, max_points=200000):
    """Pano-normalized centers of (a strided subsample of) the mask's pixels.
    ``to_equirect(us_px, vs_px)`` maps pixel-center coordinates (in pixels)."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size > max_points:
        step = int(math.ceil(rows.size / max_points))
        rows, cols = rows[::step], cols[::step]
    return to_equirect(cols + 0.5, rows + 0.5)


def box_x_interval(box, ref_x):
    """(x0, x1) of a pano-normalized box as signed offsets from ``ref_x``."""
    c = float(unwrap_x(box[0], ref_x))
    return c - box[2] / 2.0, c + box[2] / 2.0


def seam_iou(a, b):
    """IoU of two pano-normalized ``(cx, cy, w, h)`` boxes; x wraps at the seam
    (``b`` is compared in its nearest representation to ``a``). Pano-normalized
    units scale x and y uniformly, so this equals the IoU in native pixels."""
    if a is None or b is None:
        return 0.0
    inter = seam_intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def seam_intersection(a, b):
    ax0, ax1 = box_x_interval(a, a[0])
    bx0, bx1 = box_x_interval(b, a[0])
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    return iw * ih


def points_in_box(xs, ys, box):
    """Boolean array: which pano-normalized points fall inside ``box`` (seam-aware)."""
    x0, x1 = box_x_interval(box, box[0])
    dx = unwrap_x(xs, box[0])
    ys = np.asarray(ys, dtype=np.float64)
    return (dx >= x0) & (dx <= x1) & (ys >= box[1] - box[3] / 2) & (ys <= box[1] + box[3] / 2)


def band_of(cy):
    """Distance band from the gold box's center row: depression below the horizon,
    labelled with crop_window_eval's strata (flat ground, 2.5 m camera -- read as
    bands, not distances)."""
    return stratum_label(depression_deg(cy))


BAND_ORDER = [label for _, _, label in STRATA]


def prior_box(x, y, cam_height=2.5, apron_m=1.5, scale=2.0, min_dep_deg=1.0):
    """A pano-normalized box prior around a prompt, from geometry alone.

    Flat ground, camera at ``cam_height``: the prompt's depression gives a ground
    distance ``d``; a ``scale * apron_m`` square on the ground centered there
    subtends the returned box (horizontal angle at slant range, vertical from the
    near and far ground edges). Production has exactly these inputs (a point, and
    the pano geometry), so this prior never touches the gold. ``scale = 2`` was meant
    as a loose bound; measured on Richmond it lands near the gold's own size (median
    sqrt-area ratio 0.88, the 1.5 m nominal being ~1.25x small and the 2.5 m camera
    ~1.5x high there, ``crop_window_eval.md`` Finding 4), so it is effectively a size
    *estimate* -- which is why ``summarize`` scores the prior alone as a control.
    """
    dep = max(math.radians(min_dep_deg), math.radians(depression_deg(y)))
    d = cam_height / math.tan(dep)
    half = scale * apron_m / 2.0
    slant = math.hypot(d, cam_height)
    lat = (0.5 - y) * math.pi
    ang_w = 2.0 * math.atan(half / slant)
    w = min(0.5, ang_w / max(math.cos(lat), 1e-3) / TWO_PI)
    near = max(d - half, 0.05)
    far = d + half
    y_near = 0.5 + math.degrees(math.atan(cam_height / near)) / 180.0
    y_far = 0.5 + math.degrees(math.atan(cam_height / far)) / 180.0
    return (x % 1.0, (y_near + y_far) / 2.0, w, y_near - y_far)


def box_to_view_rect(box, view, n=9):
    """A pano-normalized box's bbox in view *pixels* (x0, y0, x1, y1), sampled along
    its edges; clipped to the view. Used to hand the prior to SAM2."""
    ts = np.linspace(0.0, 1.0, n)
    x0, x1 = box[0] - box[2] / 2, box[0] + box[2] / 2
    y0, y1 = box[1] - box[3] / 2, box[1] + box[3] / 2
    xs = np.concatenate([x0 + (x1 - x0) * ts, x0 + (x1 - x0) * ts, np.full(n, x0), np.full(n, x1)])
    ys = np.concatenate([np.full(n, y0), np.full(n, y1), y0 + (y1 - y0) * ts, y0 + (y1 - y0) * ts])
    u, v, ok = equirect_to_view_uv(xs % 1.0, ys, view)
    u, v = u[ok], v[ok]
    W, H = view.width, view.height
    return (float(np.clip(u.min() * W, 0, W)), float(np.clip(v.min() * H, 0, H)),
            float(np.clip(u.max() * W, 0, W)), float(np.clip(v.max() * H, 0, H)))


def box_to_crop_rect(box, left, top, pano_w, pano_h, side):
    """A pano-normalized box in equirect-crop pixels (x0, y0, x1, y1), clipped."""
    cx_px = float(unwrap_x(box[0], ((left + side / 2.0) / pano_w) % 1.0)) * pano_w + side / 2.0
    x0 = cx_px - box[2] * pano_w / 2
    x1 = cx_px + box[2] * pano_w / 2
    y0 = box[1] * pano_h - top - box[3] * pano_h / 2
    y1 = box[1] * pano_h - top + box[3] * pano_h / 2
    return (float(np.clip(x0, 0, side)), float(np.clip(y0, 0, side)),
            float(np.clip(x1, 0, side)), float(np.clip(y1, 0, side)))


def crop_side(width, height, fov_deg):
    """Same as box_gallery.crop_side (kept local so this module imports without PIL)."""
    return min(int(round(width * fov_deg / 360.0)), height, width)


def crop_rect(x, y, width, height, side):
    """Same as box_gallery.crop_rect: x wraps, y clamps by shifting."""
    left = int(round(x * width - side / 2)) % width
    top = int(min(max(round(y * height - side / 2), 0), height - side))
    return left, top


def cut_crop_array(src, left, top, side):
    """Seam-wrapped square crop of an H x W x C array (box_gallery.cut_crop in numpy)."""
    w = src.shape[1]
    cols = (np.arange(left, left + side)) % w
    return src[top:top + side][:, cols]


# ---------------------------------------------------------------------------
# Items

def load_items(city_dir):
    """Boxed items with gold and both prompt sources, plus the gold meta.

    Reuses ``crop_window_eval.load_bundle_boxes`` so the population, the
    "can't determine extent" exclusions and the completeness check are identical
    to the crop-window scorer's. ``det:<i>`` points are cross-checked against
    ``records.jsonl`` -- the end-to-end arm's claim is that the prompt *is* the
    detection RampNet emitted, so a drifted point must fail loudly.
    """
    gold, prompts, meta = load_bundle_boxes(city_dir)
    records = {}
    with open(os.path.join(city_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    items = []
    for (pid, idx), p in sorted(prompts.items()):
        box = gold[pid][idx]
        key = p["key"]
        pt = p["point"]
        if key.startswith("det:"):
            det = records[pid]["detections"][int(key.split(":")[1])]
            if (abs(det["x_normalized"] - pt["x"]) > 1e-4
                    or abs(det["y_normalized"] - pt["y"]) > 1e-4):
                raise ValueError(f"{pid} {key}: boxes.json point {pt} is not the "
                                 f"records.jsonl detection {det}")
        items.append({"pano_id": pid, "key": key, "kind": key.split(":")[0],
                      "gold": tuple(float(v) for v in box),
                      "point": (float(pt["x"]), float(pt["y"])),
                      "pano_w": records[pid]["pano"]["width"],
                      "pano_h": records[pid]["pano"]["height"]})
    return items, meta


def prompt_xy(item, prompt):
    if prompt == "boxcenter":
        return item["gold"][0] % 1.0, item["gold"][1]
    if prompt == "point":
        return item["point"]
    raise ValueError(prompt)


# ---------------------------------------------------------------------------
# Scoring one mask

def score_mask(mask, geom, item, score):
    """Row fields for one mask. ``geom`` is ("gnomonic", view) or
    ("equirect", left, top, side)."""
    pano_w, pano_h = item["pano_w"], item["pano_h"]
    gold = item["gold"]
    m = np.asarray(mask, dtype=bool)
    if geom[0] == "gnomonic":
        view = geom[1]
        sam = gnomonic_mask_to_equirect_bbox(m, view)
        H, W = m.shape
        to_eq = lambda cu, rv: view_uv_to_equirect(cu / W, rv / H, view)  # noqa: E731
    else:
        _, left, top, side = geom
        sam = crop_mask_to_equirect_bbox(m, left, top, pano_w, pano_h)
        to_eq = lambda cu, rv: (((left + cu) / pano_w) % 1.0, (top + rv) / pano_h)  # noqa: E731
    n_px = int(m.sum())
    if sam is None:
        return {"sam": None, "iou": 0.0, "gold_frac_covered": 0.0,
                "mask_frac_in_gold": None, "pred_iou": score, "mask_px": 0,
                "mask_touches_edge": False}
    xs, ys = mask_pixels_equirect(m, to_eq)
    in_gold = float(points_in_box(xs, ys, gold).mean())
    touches = bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())
    gold_area = gold[2] * gold[3]
    return {"sam": sam, "iou": seam_iou(gold, sam),
            "gold_frac_covered": seam_intersection(gold, sam) / gold_area if gold_area else 0.0,
            "mask_frac_in_gold": in_gold, "pred_iou": score, "mask_px": n_px,
            "mask_touches_edge": touches}


def _r(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, ROUND)
    return v


def make_row(city, item, arm, fov, variant, view_px, pxy, scored):
    prompt, projection = arm.split("_", 1)
    sam = scored["sam"] or (None, None, None, None)
    gold = item["gold"]
    row = {
        "city": city, "pano_id": item["pano_id"], "key": item["key"], "kind": item["kind"],
        "arm": arm, "prompt": prompt, "projection": projection, "fov": int(fov),
        "variant": variant, "pano_w": item["pano_w"], "pano_h": item["pano_h"],
        "view_px": view_px, "prompt_x": pxy[0], "prompt_y": pxy[1],
        "gold_cx": gold[0], "gold_cy": gold[1], "gold_w": gold[2], "gold_h": gold[3],
        "sam_cx": sam[0], "sam_cy": sam[1], "sam_w": sam[2], "sam_h": sam[3],
        "iou": scored["iou"], "gold_frac_covered": scored["gold_frac_covered"],
        "mask_frac_in_gold": scored["mask_frac_in_gold"], "pred_iou": scored["pred_iou"],
        "mask_px": scored["mask_px"], "mask_touches_edge": scored["mask_touches_edge"],
        "depression_deg": depression_deg(gold[1]), "band": band_of(gold[1]),
    }
    return {k: _r(float(v) if isinstance(v, (np.floating,)) else v) for k, v in row.items()}


def write_csv(path, rows):
    """LF-pinned (``newline=""``), rounded floats, stable column order."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# run (GPU)

def _variants_for(predictor, pxy_px, prior_rect):
    """{variant: (mask, predicted_iou)} from one embedded image."""
    pc = np.array([pxy_px], dtype=np.float32)
    pl = np.array([1], dtype=np.int32)
    box = np.array(prior_rect, dtype=np.float32)
    out = {}
    for name, use_box, multi in (("pt_multi", False, True), ("pt_single", False, False),
                                 ("ptbox_multi", True, True), ("ptbox_single", True, False)):
        masks, scores, _ = predictor.predict(point_coords=pc, point_labels=pl,
                                             box=box if use_box else None,
                                             multimask_output=multi)
        i = int(np.argmax(scores))
        out[name] = (masks[i] > 0, float(scores[i]))
    return out


def cmd_run(args):
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    Image.MAX_IMAGE_PIXELS = None

    t_start = time.time()
    city_dir = os.path.join(REPO, "benchmark", args.city)
    panos_dir = os.path.join(args.panos_root or REPO, "benchmark", args.city, "panos")
    items, meta = load_items(city_dir)
    if args.limit:
        keep = sorted({it["pano_id"] for it in items})[:args.limit]
        items = [it for it in items if it["pano_id"] in keep]
    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    fovs = [int(f) for f in args.fov.split(",") if f.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS}")

    with open(os.path.join(city_dir, "imagery_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    pano_ids = sorted({it["pano_id"] for it in items})
    bad = []
    for pid in pano_ids:
        path = os.path.join(panos_dir, f"{pid}.jpg")
        want = manifest["panos"][pid]["sha256"]
        if not os.path.exists(path) or sha256_file(path) != want:
            bad.append(pid)
    if bad:
        raise SystemExit(f"{len(bad)} pano(s) missing or not matching imagery_manifest.json: "
                         f"{bad[:5]}")

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CONFIG, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    load_s = time.time() - t0

    rows = []
    n_views = 0
    t_render = t_embed = t_decode = 0.0
    by_pano = {}
    for it in items:
        by_pano.setdefault(it["pano_id"], []).append(it)
    t_detect0 = time.time()
    for pi, pid in enumerate(pano_ids):
        src = np.asarray(Image.open(os.path.join(panos_dir, f"{pid}.jpg")).convert("RGB"))
        ph, pw = src.shape[:2]
        for it in by_pano[pid]:
            if (pw, ph) != (it["pano_w"], it["pano_h"]):
                raise SystemExit(f"{pid}: image {pw}x{ph} != records {it['pano_w']}x{it['pano_h']}")
            for arm in arms:
                prompt, projection = arm.split("_", 1)
                pxy = prompt_xy(it, prompt)
                prior = prior_box(pxy[0], pxy[1], cam_height=args.prior_cam_height,
                                  apron_m=args.prior_apron_m, scale=args.prior_scale)
                for fov in fovs:
                    side = crop_side(pw, ph, fov)
                    tr = time.time()
                    if projection == "gnomonic":
                        view = view_for_point(pxy[0], pxy[1], fov, side)
                        img = render_gnomonic(src, view)
                        pxy_px = (side / 2.0, side / 2.0)
                        prior_rect = box_to_view_rect(prior, view)
                        geom = ("gnomonic", view)
                    else:
                        left, top = crop_rect(pxy[0], pxy[1], pw, ph, side)
                        img = cut_crop_array(src, left, top, side)
                        pxy_px = (float(unwrap_x(pxy[0], ((left + side / 2) / pw) % 1.0)) * pw
                                  + side / 2.0, pxy[1] * ph - top)
                        prior_rect = box_to_crop_rect(prior, left, top, pw, ph, side)
                        geom = ("equirect", left, top, side)
                    te = time.time()
                    t_render += te - tr
                    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                                                                enabled=device == "cuda"):
                        predictor.set_image(np.ascontiguousarray(img))
                        td = time.time()
                        t_embed += td - te
                        outs = _variants_for(predictor, pxy_px, prior_rect)
                    t_decode += time.time() - td
                    n_views += 1
                    for variant in VARIANTS:
                        mask, score = outs[variant]
                        scored = score_mask(mask, geom, it, score)
                        rows.append(make_row(args.city, it, arm, fov, variant, side, pxy, scored))
        print(f"[{pi + 1}/{len(pano_ids)}] {pid}: {len(by_pano[pid])} items, "
              f"{n_views} views so far, {time.time() - t_start:.0f}s", flush=True)
    detect_s = time.time() - t_detect0
    elapsed = time.time() - t_start

    os.makedirs(args.out, exist_ok=True)
    rows.sort(key=lambda r: (r["pano_id"], r["key"], r["arm"], r["fov"], r["variant"]))
    csv_path = os.path.join(args.out, f"{args.city}_rows.csv")
    write_csv(csv_path, rows)

    sam2_version = _sam2_provenance()
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device == "cuda" else [])
    run_meta = {
        "city": args.city, "arms": arms, "fovs": fovs, "variants": list(VARIANTS),
        "headline_variant": HEADLINE_VARIANT, "n_items": len(items), "n_panos": len(pano_ids),
        "n_views": n_views, "n_rows": len(rows),
        "gold": {k: meta[k] for k in ("n_boxed", "n_cant", "n_other", "n_adjudicated",
                                      "completeness_warning", "crop_fov_deg",
                                      "crop_px_by_pano_dims")},
        "box_rule_version": (meta.get("box_rule") or {}).get("version"),
        "boxes_json_sha256": sha256_file(os.path.join(city_dir, "boxes.json")),
        "records_jsonl_sha256": sha256_file(os.path.join(city_dir, "records.jsonl")),
        "imagery_manifest_digest": manifest.get("digest"),
        "panos_verified_against_manifest": len(pano_ids),
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "sam2_config": SAM2_CONFIG, "sam2": sam2_version,
        "torch": torch.__version__, "python": platform.python_version(),
        "autocast": "bfloat16" if device == "cuda" else None,
        "prior": {"cam_height_m": args.prior_cam_height, "apron_m": args.prior_apron_m,
                  "scale": args.prior_scale},
        "gnomonic_render": "bilinear, view side = box_gallery crop side at the same fov",
        "host": socket.getfqdn(), "gpus": gpus,
        "elapsed_s": round(elapsed, 3), "load_s": round(load_s, 3),
        "detect_s": round(detect_s, 3), "render_s": round(t_render, 3),
        "embed_s": round(t_embed, 3), "decode_s": round(t_decode, 3),
        "rows_csv_sha256": sha256_file(csv_path),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(os.path.join(args.out, f"{args.city}_run.json"), run_meta)

    if args.usage_log.lower() != "none":
        from rampnet.ledger import append_rows
        append_rows(args.usage_log, [{
            "ts": run_meta["finished_at"], "bundle": args.city,
            "label": "sam2.1-hiera-large-extent-83", "provider": "sam2",
            "model_id": "sam2.1_hiera_large", "paid": False, "model_versions": None,
            "panos_scored": len(pano_ids), "panos_called": len(pano_ids),
            "items": len(items), "views": n_views,
            "elapsed_s": run_meta["elapsed_s"], "load_s": run_meta["load_s"],
            "detect_s": run_meta["detect_s"],
            "s_per_pano": round(detect_s / max(1, len(pano_ids)), 4),
            "hardware": {"host": run_meta["host"], "gpus": gpus},
            "signature": {"provider": "sam2", "model_id": "sam2.1_hiera_large",
                          "checkpoint_sha256": run_meta["checkpoint_sha256"],
                          "config": SAM2_CONFIG, "arms": arms, "fovs": fovs,
                          "variants": list(VARIANTS), "dtype": run_meta["autocast"],
                          "script": "scripts/analysis/sam2_extent_83.py", "issue": 83},
            "serving_path": None, "stop_reasons": None, "est_cost_usd": None,
            "pricing": None,
        }])
    print(f"wrote {len(rows)} rows ({n_views} views) to {csv_path} in {elapsed:.1f}s")


def _sam2_provenance():
    """pip version and, for a source install, the git commit it was built from."""
    out = {}
    try:
        from importlib.metadata import version
        out["dist_version"] = version("SAM-2")
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        out["dist_version"] = None
    try:
        import subprocess
        import sam2
        src = os.path.dirname(os.path.dirname(os.path.abspath(sam2.__file__)))
        p = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        out["git_commit"] = p.stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    return out


# ---------------------------------------------------------------------------
# summarize (CPU)

def read_rows(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("iou", "gold_frac_covered", "pred_iou", "depression_deg",
                      "gold_cx", "gold_cy", "gold_w", "gold_h", "prompt_x", "prompt_y"):
                r[k] = float(r[k])
            for k in ("sam_cx", "sam_cy", "sam_w", "sam_h", "mask_frac_in_gold"):
                r[k] = float(r[k]) if r[k] != "" else None
            for k in ("fov", "pano_w", "pano_h", "view_px", "mask_px", "mask_touches_edge"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def _q(vals, q):
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q)) if vals else None


def dist_stats(rows):
    """IoU distribution + the extent diagnostics for one cell."""
    iou = [r["iou"] for r in rows]
    n = len(iou)
    if not n:
        return {"n": 0}
    size = []
    for r in rows:
        if r["sam_w"]:
            size.append(math.sqrt((r["sam_w"] * r["sam_h"]) / (r["gold_w"] * r["gold_h"])))
    mig = [r["mask_frac_in_gold"] for r in rows if r["mask_frac_in_gold"] is not None]
    # Width is the measurement #86 wants first, so its error gets its own column:
    # the box-width ratio SAM2/gold (an empty mask counts as ratio 0, i.e. a miss).
    wr = [(r["sam_w"] or 0.0) / r["gold_w"] for r in rows]
    return {
        "n": n,
        "iou_median": round(_q(iou, 0.5), 4), "iou_mean": round(float(np.mean(iou)), 4),
        "iou_ge_050": round(sum(v >= 0.5 for v in iou) / n, 4),
        "iou_ge_075": round(sum(v >= 0.75 for v in iou) / n, 4),
        "iou_p25": round(_q(iou, 0.25), 4), "iou_p75": round(_q(iou, 0.75), 4),
        "gold_frac_covered_median": round(_q([r["gold_frac_covered"] for r in rows], 0.5), 4),
        "mask_frac_in_gold_median": round(_q(mig, 0.5), 4) if mig else None,
        "size_ratio_median": round(_q(size, 0.5), 4) if size else None,
        "size_ratio_p10": round(_q(size, 0.1), 4) if size else None,
        "size_ratio_p90": round(_q(size, 0.9), 4) if size else None,
        "width_ratio_median": round(_q(wr, 0.5), 4),
        "width_within_20pct": round(sum(0.8 <= v <= 1.25 for v in wr) / n, 4),
        "empty_masks": sum(1 for r in rows if not r["sam_w"]),
        "mask_touches_edge": sum(r["mask_touches_edge"] for r in rows),
        "pred_iou_median": (round(_q(pi, 0.5), 4) if (pi := [r["pred_iou"] for r in rows
                                                              if r["pred_iou"] == r["pred_iou"]])
                            else None),
    }


def paired_delta(rows, a, b, seed=BOOTSTRAP_SEED, reps=BOOTSTRAP_REPS):
    """Mean IoU(a) - IoU(b) over items present in both, with a pano-clustered
    percentile bootstrap CI (items in one pano share imagery and rig, so resampling
    items independently would understate the interval).

    ``a``/``b`` are filters: dicts of column -> value.
    """
    def pick(flt):
        return {(r["pano_id"], r["key"]): r["iou"] for r in rows
                if all(r[k] == v for k, v in flt.items())}
    A, B = pick(a), pick(b)
    keys = sorted(set(A) & set(B))
    if not keys:
        return {"n": 0}
    d = np.array([A[k] - B[k] for k in keys])
    panos = sorted({k[0] for k in keys})
    idx = {p: [i for i, k in enumerate(keys) if k[0] == p] for p in panos}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    sums = np.array([d[idx[p]].sum() for p in panos])
    cnts = np.array([len(idx[p]) for p in panos])
    for i in range(reps):
        s = rng.integers(0, len(panos), len(panos))
        means[i] = sums[s].sum() / cnts[s].sum()
    return {"n": len(keys), "n_panos": len(panos),
            "mean_delta": round(float(d.mean()), 4),
            "median_delta": round(float(np.median(d)), 4),
            "ci95": [round(float(np.quantile(means, 0.025)), 4),
                     round(float(np.quantile(means, 0.975)), 4)],
            "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
            "bootstrap": {"reps": reps, "seed": seed, "unit": "pano"}}


def prior_only_rows(rows, prior_kw=None):
    """The no-SAM2 control for the ``ptbox_*`` variants: score the geometry prior box
    itself against the gold, one row per (item, prompt source). The prior depends on
    the prompt only, not on the projection or FOV, so it is emitted once per prompt
    as arm ``<prompt>_prior``, fov 0, variant ``prior_only``. Without this row a
    ``ptbox`` IoU cannot be read: part of it is the prior's own overlap."""
    prior_kw = prior_kw or {}
    out, seen = [], set()
    for r in rows:
        prompt = r["arm"].split("_", 1)[0]
        key = (r["pano_id"], r["key"], prompt)
        if key in seen:
            continue
        seen.add(key)
        pb = prior_box(r["prompt_x"], r["prompt_y"], **prior_kw)
        gold = (r["gold_cx"], r["gold_cy"], r["gold_w"], r["gold_h"])
        q = dict(r)
        q.update({"arm": f"{prompt}_prior", "projection": "prior", "fov": 0,
                  "variant": "prior_only", "sam_cx": pb[0], "sam_cy": pb[1],
                  "sam_w": pb[2], "sam_h": pb[3], "iou": seam_iou(gold, pb),
                  "gold_frac_covered": seam_intersection(gold, pb) / (gold[2] * gold[3]),
                  "mask_frac_in_gold": None, "pred_iou": float("nan"), "mask_px": 0,
                  "mask_touches_edge": 0})
        out.append(q)
    return out


def summarize_rows(rows, prior_kw=None):
    prior = prior_only_rows(rows, prior_kw)
    rows = rows + prior
    arms = sorted({r["arm"] for r in rows if not r["arm"].endswith("_prior")})
    fovs = sorted({r["fov"] for r in rows if r["fov"]})
    out = {"cells": {}, "bands": {}, "deltas": {}, "subsets": {}}
    for prompt in PROMPTS:
        sel = [r for r in prior if r["arm"] == f"{prompt}_prior"]
        if sel:
            key = f"{prompt}_prior|0|prior_only"
            out["cells"][key] = dist_stats(sel)
            out["bands"][key] = {b: dist_stats([r for r in sel if r["band"] == b])
                                 for b in BAND_ORDER}
            out["subsets"][key] = {k: dist_stats([r for r in sel if r["kind"] == k])
                                   for k in ("det", "missed")}
            for arm in (a for a in arms if a.startswith(prompt + "_")):
                for fov in fovs:
                    for variant in ("ptbox_multi", "ptbox_single"):
                        res = paired_delta(rows, {"arm": arm, "fov": fov, "variant": variant},
                                           {"arm": f"{prompt}_prior", "fov": 0,
                                            "variant": "prior_only"})
                        if res.get("n"):
                            out["deltas"][f"sam-prior|{arm}|{fov}|{variant}"] = res
    for arm in arms:
        for fov in fovs:
            for variant in VARIANTS:
                sel = [r for r in rows if r["arm"] == arm and r["fov"] == fov
                       and r["variant"] == variant]
                if sel:
                    out["cells"][f"{arm}|{fov}|{variant}"] = dist_stats(sel)
                    out["bands"][f"{arm}|{fov}|{variant}"] = {
                        b: dist_stats([r for r in sel if r["band"] == b]) for b in BAND_ORDER}
                    out["subsets"][f"{arm}|{fov}|{variant}"] = {
                        k: dist_stats([r for r in sel if r["kind"] == k])
                        for k in ("det", "missed")}
    for prompt in PROMPTS:
        # Cross-FOV pairs too: at equal FOV a gnomonic view shows the prompted ramp
        # smaller than the equirect crop does (center magnification (side/2)/tan(fov/2)
        # vs side/fov px per radian: 0.79x at 90 deg), so gnomonic@76 vs equirect@90 is
        # the matched-magnification comparison. Keyed "gnomonic@<g>-equirect@<e>".
        for fg in fovs:
            for fe in fovs:
                if fg == fe:
                    continue
                for variant in VARIANTS:
                    res = paired_delta(rows,
                                       {"arm": f"{prompt}_gnomonic", "fov": fg, "variant": variant},
                                       {"arm": f"{prompt}_equirect", "fov": fe, "variant": variant})
                    if res.get("n"):
                        out["deltas"][f"gnomonic@{fg}-equirect@{fe}|{prompt}|{variant}"] = res
        for fov in fovs:
            for variant in VARIANTS:
                g = {"arm": f"{prompt}_gnomonic", "fov": fov, "variant": variant}
                e = {"arm": f"{prompt}_equirect", "fov": fov, "variant": variant}
                res = paired_delta(rows, g, e)
                if res.get("n"):
                    out["deltas"][f"gnomonic-equirect|{prompt}|{fov}|{variant}"] = res
                    per_band = {}
                    for b in BAND_ORDER:
                        rb = [r for r in rows if r["band"] == b]
                        pb = paired_delta(rb, g, e, reps=2000)
                        if pb.get("n"):
                            per_band[b] = pb
                    out["deltas"][f"gnomonic-equirect|{prompt}|{fov}|{variant}|by_band"] = per_band
        for proj in PROJECTIONS:
            for variant in VARIANTS:
                if len(fovs) > 1:
                    a = {"arm": f"{prompt}_{proj}", "fov": fovs[-1], "variant": variant}
                    b = {"arm": f"{prompt}_{proj}", "fov": fovs[0], "variant": variant}
                    res = paired_delta(rows, a, b)
                    if res.get("n"):
                        out["deltas"][f"fov{fovs[-1]}-fov{fovs[0]}|{prompt}_{proj}|{variant}"] = res
    # point vs boxcenter on det items: the cost of prompting from the detection
    for proj in PROJECTIONS:
        for fov in fovs:
            for variant in VARIANTS:
                dr = [r for r in rows if r["kind"] == "det"]
                res = paired_delta(dr, {"arm": f"point_{proj}", "fov": fov, "variant": variant},
                                   {"arm": f"boxcenter_{proj}", "fov": fov, "variant": variant})
                if res.get("n"):
                    out["deltas"][f"detpoint-boxcenter|det|{proj}|{fov}|{variant}"] = res
    return out


def cmd_summarize(args):
    cities = [c.strip() for c in args.city.split(",") if c.strip()]
    rows = []
    for city in cities:
        rows.extend(read_rows(os.path.join(args.out, f"{city}_rows.csv")))
    priors = set()
    for c in cities:
        rj = os.path.join(args.out, f"{c}_run.json")
        if os.path.exists(rj):
            with open(rj, encoding="utf-8") as f:
                p = json.load(f)["prior"]
            priors.add((p["cam_height_m"], p["apron_m"], p["scale"]))
    if len(priors) > 1:
        raise SystemExit(f"cities were run with different box priors: {priors}")
    prior_kw = {}
    if priors:
        h, a, k = priors.pop()
        prior_kw = {"cam_height": h, "apron_m": a, "scale": k}
    summary = {"cities": cities, "headline_variant": HEADLINE_VARIANT, "prior": prior_kw,
               "rows_csv_sha256": {c: sha256_file(os.path.join(args.out, f"{c}_rows.csv"))
                                   for c in cities}}
    summary.update(summarize_rows(rows, prior_kw))
    name = args.name or "+".join(cities)
    write_json(os.path.join(args.out, f"{name}_summary.json"), summary)
    print(format_tables(summary))


def format_tables(summary):
    lines = ["arm | fov | variant | n | IoU med | mean | >=0.5 | >=0.75 | cover med | "
             "in-gold med | size p10/p50/p90 | width ±20% | empty | edge"]
    for k, s in summary["cells"].items():
        arm, fov, var = k.split("|")
        lines.append(f"{arm} | {fov} | {var} | {s['n']} | {s['iou_median']} | {s['iou_mean']} | "
                     f"{s['iou_ge_050']} | {s['iou_ge_075']} | {s['gold_frac_covered_median']} | "
                     f"{s['mask_frac_in_gold_median']} | {s['size_ratio_p10']}/"
                     f"{s['size_ratio_median']}/{s['size_ratio_p90']} | "
                     f"{s['width_within_20pct']} | {s['empty_masks']} | "
                     f"{s['mask_touches_edge']}")
    lines.append("")
    for k, d in summary["deltas"].items():
        if k.endswith("by_band"):
            continue
        lines.append(f"{k}: n={d['n']} mean={d['mean_delta']} CI={d['ci95']} "
                     f"median={d['median_delta']} a>b {d['a_better']} / b>a {d['b_better']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gallery (CPU + panos)

def cmd_gallery(args):
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    rows = read_rows(os.path.join(args.out, f"{args.city}_rows.csv"))
    head = [r for r in rows if r["arm"] == args.gallery_arm and r["fov"] == args.gallery_fov
            and r["variant"] == args.gallery_variant]
    other_arm = args.gallery_arm.replace("gnomonic", "equirect")
    other = {(r["pano_id"], r["key"]): r for r in rows
             if r["arm"] == other_arm and r["fov"] == args.gallery_fov
             and r["variant"] == args.gallery_variant}
    head.sort(key=lambda r: (r["iou"], r["pano_id"], r["key"]))
    n = args.gallery_n
    worst = head[:n]
    mid = len(head) // 2
    median = head[max(0, mid - n // 2): max(0, mid - n // 2) + n]
    panos_dir = os.path.join(args.panos_root or REPO, "benchmark", args.city, "panos")
    os.makedirs(args.assets, exist_ok=True)
    cache = {}
    written = []
    for label, sel in (("worst", worst), ("median", median)):
        tiles = []
        for r in sel:
            pid = r["pano_id"]
            if pid not in cache:
                cache.clear()
                cache[pid] = Image.open(os.path.join(panos_dir, f"{pid}.jpg")).convert("RGB")
            tiles.append(_gallery_tile(cache[pid], r, other.get((pid, r["key"])),
                                       args.tile, ImageDraw))
        sheet = _sheet(tiles, cols=4, tile=args.tile)
        path = os.path.join(args.assets, f"sam2_extent_83_{args.city}_{args.gallery_variant}_{label}.jpg")
        sheet.save(path, quality=85)
        written.append(path)
    print("wrote", *written)


def _gallery_tile(img, r, other, tile, ImageDraw):
    """Equirect crop around the gold box: gold green, SAM2 (gallery arm) magenta,
    the paired equirect-arm box orange (dashed-looking thin), prompt white dot."""
    from PIL import Image
    W, H = img.size
    gold = (r["gold_cx"], r["gold_cy"], r["gold_w"], r["gold_h"])
    boxes = [gold]
    if r["sam_w"]:
        boxes.append((r["sam_cx"], r["sam_cy"], r["sam_w"], r["sam_h"]))
    ext = []
    for b in boxes:
        x0, x1 = box_x_interval(b, gold[0])
        ext.append((x0, x1, b[1] - b[3] / 2, b[1] + b[3] / 2))
    x0 = min(e[0] for e in ext) * W
    x1 = max(e[1] for e in ext) * W
    y0 = min(e[2] for e in ext) * H
    y1 = max(e[3] for e in ext) * H
    side = max(x1 - x0, y1 - y0) * 1.6 + 48
    side = int(min(max(side, 96), H))
    cxp = gold[0] * W + (x0 + x1) / 2
    cyp = (y0 + y1) / 2
    left = int(round(cxp - side / 2)) % W
    top = int(min(max(round(cyp - side / 2), 0), H - side))
    arr = cut_crop_array(np.asarray(img), left, top, side)
    crop = Image.fromarray(arr).resize((tile, tile), Image.BILINEAR)
    d = ImageDraw.Draw(crop)
    s = tile / side
    ref = ((left + side / 2) / W) % 1.0

    def rect(b, color, width):
        bx0, bx1 = box_x_interval(b, ref)
        px0 = (bx0 * W + side / 2) * s
        px1 = (bx1 * W + side / 2) * s
        py0 = ((b[1] - b[3] / 2) * H - top) * s
        py1 = ((b[1] + b[3] / 2) * H - top) * s
        d.rectangle([px0, py0, px1, py1], outline=color, width=width)

    if other is not None and other["sam_w"]:
        rect((other["sam_cx"], other["sam_cy"], other["sam_w"], other["sam_h"]), (255, 150, 0), 1)
    rect(gold, (0, 230, 0), 2)
    if r["sam_w"]:
        rect((r["sam_cx"], r["sam_cy"], r["sam_w"], r["sam_h"]), (255, 0, 255), 2)
    px = (float(unwrap_x(r["prompt_x"], ref)) * W + side / 2) * s
    py = (r["prompt_y"] * H - top) * s
    d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255), outline=(0, 0, 0))
    eq = f" eq {other['iou']:.2f}" if other is not None else ""
    txt = f"IoU {r['iou']:.2f}{eq} | {r['band']} | {r['key']}"
    d.rectangle([0, tile - 16, tile, tile], fill=(0, 0, 0))
    d.text((3, tile - 14), txt, fill=(255, 255, 255))
    return crop


def _sheet(tiles, cols, tile):
    from PIL import Image
    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile, rows * tile), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tile, (i // cols) * tile))
    return sheet


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--city", required=True,
                       help="Bundle under benchmark/ with a boxes.json (comma list for summarize).")
        p.add_argument("--out", default=os.path.join("analysis_out", "sam2_extent_83"),
                       help="Directory for <city>_rows.csv / _run.json / _summary.json.")
        p.add_argument("--panos-root", default=None,
                       help="Checkout holding benchmark/<city>/panos (default: this repo).")

    r = sub.add_parser("run", help="GPU: segment every boxed item.")
    common(r)
    r.add_argument("--arm", default=",".join(ARMS), help=f"Comma list from {ARMS}.")
    r.add_argument("--fov", default="90,76,60", help="Comma list of square FOVs in degrees.")
    r.add_argument("--checkpoint", required=True, help="Path to sam2.1_hiera_large.pt.")
    r.add_argument("--prior-cam-height", type=float, default=2.5)
    r.add_argument("--prior-apron-m", type=float, default=1.5)
    r.add_argument("--prior-scale", type=float, default=2.0)
    r.add_argument("--limit", type=int, default=0, help="First N panos only (smoke test).")
    r.add_argument("--usage-log", required=True,
                   help="JSONL ledger to append the paid:false time row to ('none' to skip). "
                        "Point it somewhere that outlives the run, then commit the row into "
                        "analysis_out/usage_log.jsonl.")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="CPU: tables + paired deltas.")
    common(s)
    s.add_argument("--name", default=None, help="Summary basename (default: the city list).")
    s.set_defaults(func=cmd_summarize)

    g = sub.add_parser("gallery", help="CPU + panos: worst / median contact sheets.")
    common(g)
    g.add_argument("--assets", default=os.path.join("docs", "assets"))
    g.add_argument("--gallery-arm", default="boxcenter_gnomonic")
    g.add_argument("--gallery-fov", type=int, default=90)
    g.add_argument("--gallery-variant", default=HEADLINE_VARIANT, choices=VARIANTS)
    g.add_argument("--gallery-n", type=int, default=8)
    g.add_argument("--tile", type=int, default=256)
    g.set_defaults(func=cmd_gallery)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
