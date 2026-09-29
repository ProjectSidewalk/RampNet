"""Semantic and structural arms (#48): align the scene's structure, not its texture.

Curb lines, the sidewalk/road boundary and painted markings (crosswalk stripes, stop and
lane lines) should look the same across capture dates and lighting, where the texture the
matching arms rely on does not. Every arm here does the same three things:

1. **Structure in each view.** A per-pixel structure map of the source view (centred on
   the GT point) and the other view (centred on today's projection):

   * ``sem_*`` arms: Mask2Former (Swin-L) trained on Mapillary Vistas v1.2, 65 classes, run
     at the views' native 1024x768. **The "Curb Cut" class (id 9) is suppressed before the
     argmax**, so its pixels go to their next-best class. A segmenter that finds curb cuts
     in the other view is a ramp detector, and the reference here is the other view's ramp
     detection, so reading that class would be circular. Two edge channels are kept: the
     *curb edge* (raised curb / sidewalk / pedestrian-area pixels touching road-like
     pixels) and the *marking edge* (the outline of painted markings).
   * ``lsd_*`` arms: OpenCV's LSD line segments on the grey view, rasterized. No model.

2. **Onto the ground.** Edge pixels at least 5 deg below the pano horizon (the ``lg``
   band) are raycast onto flat ground with each pano's own heading and position -- the
   labeler's flat path, re-implemented analytically so a whole view projects in one numpy
   call (checked against ``proj_x`` / ``proj_y``, see ``sem_geom_check``). Only points
   within ``KEEP_M`` of the GT point's world position W are kept. That puts both views'
   structure in one metric bird's-eye frame, centred on W.

3. **Align, then re-project W.** Either a translation that best overlays the other
   view's structure on the source's (``*_chamfer``: truncated chamfer, soft-argmin under a
   Gaussian prior on the pose error, so a single straight curb only moves the point across
   the curb, never along it), or a snap to the nearest curb edge (``sem_snap``,
   ``sem_curb_shift``). The corrected W is placed in the other pano with the flat inverse.

The camera height is 2.6 m (today's projection) unless the arm name ends ``_auto`` (the
labeler's per-rig 'auto' heights, as in ``proj_height_auto``). An arm returns None, i.e.
falls back to today's 2.6 m projection, when it has too little structure to align.

**What no arm here reads:** the answer columns (the harness removes them), and any ramp
detector's output in either view -- RampNet, or Vistas' own "Curb Cut" class.

Inputs: ``--views`` (as for ``lg``), the labeler (``--labeler-root`` etc., as for the
geometry arms: poses and heights), and ``--extra seg_dir=DIR`` for the ``sem_*`` arms:
the label maps ``segment`` writes. A missing label map is computed on the fly (slow on
CPU), so ``predict`` alone reproduces them; ``segment`` is the batch GPU path::

    python scripts/analysis/crossview_arms/semantic.py segment --views VIEWS --out SEG_DIR
    python scripts/analysis/crossview_align_48.py predict --arm sem_chamfer --views VIEWS \\
        --extra seg_dir=SEG_DIR --labeler-root LABELER --runs-root LABELER/runs \\
        --results-root RUNS_ARCHIVE
    python scripts/analysis/crossview_arms/semantic.py compare --out PATH   # doc tables

Aerial anchors (a sidewalk / curb map from overhead imagery) are out of scope here; they
belong to sidewalk-auto-labeler#104.
"""
import hashlib
import json
import math
import os
import sys

if __name__ == "__main__":                                   # direct run: find the harness
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import crossview_align_48 as H
from crossview_arms._registry import register

# --------------------------------------------------------------------------- #
# constants (fixed before any arm was scored)
# --------------------------------------------------------------------------- #

VISTAS_CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
VISTAS_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
SEG_INPUT_HW = (H.VIEW_H, H.VIEW_W)        # native view size, not the processor's 384x384
CURB_CUT = 9                               # suppressed: a curb-cut segmenter is a ramp detector
CURB, SIDEWALK, PED_AREA = 2, 15, 11
RAISED = (CURB, SIDEWALK, PED_AREA)
ROADLIKE = (13, 7, 10, 14, 8, 23, 24, 36, 41, 43)   # road, bike lane, parking, service lane,
#                                           crosswalk-plain, crosswalk + general markings,
#                                           catch basin, manhole, pothole
MARKING = (23, 24)                          # Lane Marking - Crosswalk / - General
EXPECTED_LABELS = {2: "Curb", 9: "Curb Cut", 11: "Pedestrian Area", 13: "Road", 15: "Sidewalk",
                   23: "Lane Marking - Crosswalk", 24: "Lane Marking - General"}

FLAT_M = 2.6
GROUND_MARGIN_DEG = 5.0                     # the lg band; also drops the unreliable far field
MAX_RANGE_M = 30.0
KEEP_M = 10.0                               # structure within this of W takes part
CELL_M = 0.05                               # bird's-eye raster
TRUNC_M = 0.5                               # chamfer truncation
SEARCH_M = 3.0                              # translation search half-width
COARSE_M = 0.1
PRIOR_SIGMA_M = 1.5                         # prior on the relative pose error at W
TEMP_M = 0.02                               # soft-argmin temperature (mean-distance units)
MIN_CELLS = 40                              # structure cells needed in each view
MAX_SHRINK = 0.8                            # data must shrink the posterior sd below this x prior
SNAP_M = 2.5                                # snap radius around W
LSD_MIN_LEN_PX = 20.0
LSD_SEM_PX = 4                              # lsd_sem: segment pixels within this of a structure edge

SEM_CONFIG = {"checkpoint": VISTAS_CHECKPOINT, "revision": VISTAS_REVISION,
              "input_hw": list(SEG_INPUT_HW), "suppressed_class": "Curb Cut (9)",
              "curb_edge": "Curb/Sidewalk/Pedestrian Area touching road-like classes",
              "marking_edge": "outline of Lane Marking - Crosswalk / - General",
              "ground_margin_deg": GROUND_MARGIN_DEG, "max_range_m": MAX_RANGE_M,
              "keep_m": KEEP_M}
CHAMFER_CONFIG = {"cell_m": CELL_M, "trunc_m": TRUNC_M, "search_m": SEARCH_M,
                  "prior_sigma_m": PRIOR_SIGMA_M, "temp_m": TEMP_M, "min_cells": MIN_CELLS,
                  "max_shrink": MAX_SHRINK}


# --------------------------------------------------------------------------- #
# segmentation (Mask2Former / Vistas, curb cut suppressed)
# --------------------------------------------------------------------------- #


class Segmenter:
    """Mask2Former Vistas at the views' native size. ``__call__`` returns the uint8 label map
    with Curb Cut suppressed and the number of pixels Curb Cut would have won."""

    def __init__(self, device=None):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(VISTAS_CHECKPOINT, revision=VISTAS_REVISION)
        self.proc.size = {"height": SEG_INPUT_HW[0], "width": SEG_INPUT_HW[1]}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            VISTAS_CHECKPOINT, revision=VISTAS_REVISION).to(self.device).eval()
        id2label = self.model.config.id2label
        for cid, name in EXPECTED_LABELS.items():
            got = id2label.get(cid, id2label.get(str(cid)))
            if got != name:
                raise RuntimeError(f"class {cid} is {got!r}, expected {name!r}")

    def __call__(self, bgr_images):
        torch = self.torch
        rgb = [im[:, :, ::-1].copy() for im in bgr_images]
        inp = self.proc(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(pixel_values=inp["pixel_values"].to(self.device))
            cls = out.class_queries_logits.softmax(-1)[..., :-1]          # B Q C
            msk = out.masks_queries_logits.sigmoid()                        # B Q h w
            sc = torch.einsum("bqc,bqhw->bchw", cls, msk)
            sc = torch.nn.functional.interpolate(sc, size=bgr_images[0].shape[:2],
                                                 mode="bilinear", align_corners=False)
            full = sc.argmax(1)
            sc[:, CURB_CUT] = -1.0
            sup = sc.argmax(1)
        res = []
        for i in range(len(bgr_images)):
            res.append((sup[i].to(torch.uint8).cpu().numpy(),
                        int((full[i] == CURB_CUT).sum().item())))
        return res


def _seg_dir(ctx):
    extra = dict(e.split("=", 1) for e in getattr(ctx.args, "extra", []) or [])
    d = extra.get("seg_dir")
    if not d:
        raise SystemExit("sem_* arms need --extra seg_dir=DIR (see `semantic.py segment`)")
    return d


def label_map(pair, which, ctx):
    """The curb-cut-suppressed Vistas label map of a view, from seg_dir, computed and
    written there if missing."""
    import cv2
    d = _seg_dir(ctx)
    path = os.path.join(d, f"{pair['pair_id']}_{which}.png")
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        if "segmenter" not in ctx.cache:
            ctx.cache["segmenter"] = Segmenter("cpu" if getattr(ctx.args, "cpu", False) else None)
        lab, _ = ctx.cache["segmenter"]([ctx.view(pair, which)])[0]
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(path, lab)
    return lab


def structure_edges(lab):
    """(curb_edge, marking_edge) boolean maps from a label map."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    raised = np.isin(lab, RAISED).astype(np.uint8)
    road = np.isin(lab, ROADLIKE).astype(np.uint8)
    curb = raised.astype(bool) & cv2.dilate(road, k).astype(bool)
    mark = np.isin(lab, MARKING).astype(np.uint8)
    edge = mark.astype(bool) & ~cv2.erode(mark, k).astype(bool)
    return curb, edge


def lsd_edges(gray):
    """Rasterized LSD segments (length >= LSD_MIN_LEN_PX), 1 px wide."""
    import cv2
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    out = np.zeros(gray.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if math.hypot(x2 - x1, y2 - y1) >= LSD_MIN_LEN_PX:
                cv2.line(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1, 1)
    return out.astype(bool)


# --------------------------------------------------------------------------- #
# analytic flat-ground geometry (the labeler's flat path, vectorized)
# --------------------------------------------------------------------------- #


class Cam:
    """A pano on flat ground: its ENU position in a frame at the source camera, heading
    (radians) and camera height. Mirrors geo.detection_ground_point / ground_point_to_pano
    on the flat path (apply_pose=False)."""

    def __init__(self, e, n, heading_rad, h):
        self.e, self.n, self.hd, self.h = e, n, heading_rad, h

    def to_ground(self, x, y, min_dep_rad=0.02):
        phi = (np.asarray(x, float) - 0.5) * 2.0 * np.pi
        dep = (np.asarray(y, float) - 0.5) * np.pi
        ok = dep > min_dep_rad
        d = np.where(ok, self.h / np.tan(np.where(ok, dep, 1.0)), np.nan)
        b = self.hd + phi
        return self.e + d * np.sin(b), self.n + d * np.cos(b), d, ok

    def to_pano(self, e, n):
        de, dn = e - self.e, n - self.n
        d = math.hypot(de, dn)
        phi = math.atan2(de, dn) - self.hd
        return (0.5 + phi / (2.0 * math.pi)) % 1.0, 0.5 + math.atan2(self.h, d) / math.pi


def cams(pair, ctx, height_mode):
    """(src Cam, oth Cam) or None, from the labeler's SlimPanos at 2.6 m or 'auto'."""
    from crossview_arms import geometry as G
    L = ctx.labeler()
    mode = FLAT_M if height_mode == "flat" else L.fs.HEIGHT_AUTO
    slims, height, _ = G.at_height(ctx, pair["city"], mode)
    s, o = slims.get(pair["src_pano"]), slims.get(pair["oth_pano"])
    if s is None or o is None:
        return None
    ps, po = L.fs.pano_pose(s, "off"), L.fs.pano_pose(o, "off")
    frame = L.geo.LocalFrame(ps.lat, ps.lng)
    eo, no = frame.to_enu(po.lat, po.lng)
    hs = L.geo.camera_height_for(ps, camera_height=height)[0]
    ho = L.geo.camera_height_for(po, camera_height=height)[0]
    return (Cam(0.0, 0.0, math.radians(ps.heading_deg), hs),
            Cam(eo, no, math.radians(po.heading_deg), ho))


def ground_points(mask, cam, centre, W):
    """ENU points of the True pixels of a view mask that are on the ground band, within
    MAX_RANGE_M of their camera and KEEP_M of W."""
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 2))
    x, y = H.view_to_pano(u + 0.5, v + 0.5, *centre)
    el = H.elevation_deg(y)
    keep = (el < -GROUND_MARGIN_DEG) & (el > H.RIG_LIMIT_DEG)
    e, n, d, ok = cam.to_ground(x[keep], y[keep])
    ok = ok & (d < MAX_RANGE_M)
    P = np.stack([e[ok], n[ok]], 1)
    P = P[np.hypot(P[:, 0] - W[0], P[:, 1] - W[1]) < KEEP_M]
    return P


def _cells(P):
    """Unique CELL_M cells of a point set (dedups the dense near field)."""
    if len(P) == 0:
        return P
    return np.unique(np.round(P / CELL_M).astype(np.int64), axis=0) * CELL_M


# --------------------------------------------------------------------------- #
# chamfer alignment
# --------------------------------------------------------------------------- #


def _dt(P, W):
    """Truncated distance transform (metres) of point set P on a raster centred on W."""
    import cv2
    half = KEEP_M + SEARCH_M + TRUNC_M
    n = int(math.ceil(2 * half / CELL_M)) + 1
    img = np.ones((n, n), np.uint8)
    ij = np.round((P - (np.asarray(W) - half)) / CELL_M).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    img[ij[ok, 1], ij[ok, 0]] = 0
    dt = cv2.distanceTransform(img, cv2.DIST_L2, 5) * CELL_M
    return np.minimum(dt, TRUNC_M), np.asarray(W) - half, n


def _cost(channels, t):
    """Mean truncated distance of the other view's points shifted by t (per channel, pooled)."""
    tot, cnt = 0.0, 0
    for dt, org, n, Q in channels:
        ij = np.round((Q + t - org) / CELL_M).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        vals = np.full(len(Q), TRUNC_M)
        vals[ok] = dt[ij[ok, 1], ij[ok, 0]]
        tot += vals.sum()
        cnt += len(Q)
    return tot / max(cnt, 1)


def chamfer_align(src_sets, oth_sets, W):
    """Translation t (ENU metres) such that other + t overlays source, as the posterior mean
    of exp(-(J(t) - Jmin) / TEMP_M) x N(0, PRIOR_SIGMA_M^2) over a grid (J = mean truncated
    chamfer). Returns (t or None, diagnostics). None when either view has < MIN_CELLS
    structure cells or the data do not shrink the posterior sd below MAX_SHRINK x prior
    along its best-constrained direction."""
    ns = sum(len(s) for s in src_sets)
    no = sum(len(o) for o in oth_sets)
    diag = {"src_cells": int(ns), "oth_cells": int(no)}
    if ns < MIN_CELLS or no < MIN_CELLS:
        diag["reason"] = "too_little_structure"
        return None, diag
    channels = []
    for S, Q in zip(src_sets, oth_sets):
        if len(S) and len(Q):
            dt, org, n = _dt(S, W)
            channels.append((dt, org, n, Q))
    if not channels:
        diag["reason"] = "no_shared_channel"
        return None, diag
    g = np.arange(-SEARCH_M, SEARCH_M + 1e-9, COARSE_M)
    T = np.array([(a, b) for b in g for a in g])
    J = np.array([_cost(channels, t) for t in T])
    lp = -(J - J.min()) / TEMP_M - (T ** 2).sum(1) / (2 * PRIOR_SIGMA_M ** 2)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mu = (w[:, None] * T).sum(0)
    C = ((T - mu).T * w) @ (T - mu)
    sd = np.sqrt(np.clip(np.linalg.eigvalsh(C), 0, None))
    j0 = _cost(channels, np.zeros(2))
    diag.update({"t_e": float(mu[0]), "t_n": float(mu[1]), "t_m": float(np.hypot(*mu)),
                 "post_sd_min": float(sd[0]), "post_sd_max": float(sd[1]),
                 "cost0": float(j0), "cost_min": float(J.min()),
                 "cost_at_t": float(_cost(channels, mu))})
    if sd[0] > MAX_SHRINK * PRIOR_SIGMA_M:
        diag["reason"] = "uninformative"
        return None, diag
    return mu, diag


# --------------------------------------------------------------------------- #
# per-pair plumbing
# --------------------------------------------------------------------------- #


def _setup(pair, ctx, height_mode):
    c = cams(pair, ctx, height_mode)
    if c is None:
        return None
    cs, co = c
    e, n, _, ok = cs.to_ground(pair["src_x"], pair["src_y"])
    if not bool(ok):
        return None
    return cs, co, (float(e), float(n))


def _place(co, W, t):
    x, y = co.to_pano(W[0] - t[0], W[1] - t[1])
    return float(x), float(y)


def _sem_sets(pair, ctx, cs, co, W, channels=("curb", "marking")):
    out = []
    for which, cam in (("src", cs), ("oth", co)):
        curb, mark = structure_edges(label_map(pair, which, ctx))
        m = {"curb": curb, "marking": mark}
        out.append([_cells(ground_points(m[c], cam, ctx.view_centre(pair, which), W))
                    for c in channels])
    return out


def _lsd_sets(pair, ctx, cs, co, W, sem_filter=False):
    import cv2
    out = []
    for which, cam in (("src", cs), ("oth", co)):
        e = lsd_edges(ctx.view(pair, which, cv2.IMREAD_GRAYSCALE))
        if sem_filter:
            curb, mark = structure_edges(label_map(pair, which, ctx))
            k = np.ones((2 * LSD_SEM_PX + 1, 2 * LSD_SEM_PX + 1), np.uint8)
            near = cv2.dilate((curb | mark).astype(np.uint8), k).astype(bool)
            e = e & near
        out.append([_cells(ground_points(e, cam, ctx.view_centre(pair, which), W))])
    return out


def _chamfer_arm(pair, ctx, height_mode, sets_fn):
    s = _setup(pair, ctx, height_mode)
    if s is None:
        return {"x": None, "y": None, "no_geometry": True}
    cs, co, W = s
    src_sets, oth_sets = sets_fn(pair, ctx, cs, co, W)
    t, diag = chamfer_align(src_sets, oth_sets, W)
    if t is None:
        return {"x": None, "y": None, **diag}
    x, y = _place(co, W, t)
    return {"x": x, "y": y, **diag}


def _nearest(P, W, r):
    if len(P) == 0:
        return None
    d = np.hypot(P[:, 0] - W[0], P[:, 1] - W[1])
    i = int(np.argmin(d))
    return P[i] if d[i] <= r else None


def _snap_arm(pair, ctx, shift):
    s = _setup(pair, ctx, "flat")
    if s is None:
        return {"x": None, "y": None, "no_geometry": True}
    cs, co, W = s
    (src_curb,), (oth_curb,) = _sem_sets(pair, ctx, cs, co, W, channels=("curb",))
    c_o = _nearest(oth_curb, W, SNAP_M)
    if c_o is None:
        return {"x": None, "y": None, "reason": "no_oth_curb_near"}
    if shift:
        c_s = _nearest(src_curb, W, SNAP_M)
        if c_s is None:
            return {"x": None, "y": None, "reason": "no_src_curb_near"}
        t = c_s - c_o                     # other + t overlays source
    else:
        t = np.asarray(W) - c_o           # put W on the other view's curb
    x, y = _place(co, W, t)
    return {"x": x, "y": y, "t_e": float(t[0]), "t_n": float(t[1]), "t_m": float(np.hypot(*t))}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

NEEDS = ("views", "labeler", "panos")


@register("sem_geom_check", needs=("labeler", "panos"), config={"camera_height_m": FLAT_M},
          description="instrument check: this module's analytic flat geometry, zero shift")
def sem_geom_check(pair, ctx):
    s = _setup(pair, ctx, "flat")
    if s is None:
        return {"x": None, "y": None, "no_geometry": True}
    cs, co, W = s
    x, y = _place(co, W, (0.0, 0.0))
    return {"x": x, "y": y}


@register("sem_chamfer", needs=NEEDS,
          config={**SEM_CONFIG, **CHAMFER_CONFIG, "channels": ["curb", "marking"],
                  "camera_height_m": FLAT_M},
          description="Vistas curb + marking edges (curb cut suppressed), bird's-eye chamfer shift")
def sem_chamfer(pair, ctx):
    return _chamfer_arm(pair, ctx, "flat", _sem_sets)


@register("sem_chamfer_auto", needs=NEEDS,
          config={**SEM_CONFIG, **CHAMFER_CONFIG, "channels": ["curb", "marking"],
                  "camera_height": "auto (as proj_height_auto)"},
          description="sem_chamfer on the labeler's 'auto' camera heights")
def sem_chamfer_auto(pair, ctx):
    return _chamfer_arm(pair, ctx, "auto", _sem_sets)


@register("sem_chamfer_curb", needs=NEEDS,
          config={**SEM_CONFIG, **CHAMFER_CONFIG, "channels": ["curb"], "camera_height_m": FLAT_M},
          description="sem_chamfer on the curb-edge channel only (no markings)")
def sem_chamfer_curb(pair, ctx):
    return _chamfer_arm(pair, ctx, "flat",
                        lambda *a: _sem_sets(*a, channels=("curb",)))


@register("lsd_chamfer", needs=("views", "labeler", "panos"),
          config={**CHAMFER_CONFIG, "lines": "cv2 LSD, length >= 20 px",
                  "ground_margin_deg": GROUND_MARGIN_DEG, "keep_m": KEEP_M,
                  "camera_height_m": FLAT_M},
          description="LSD ground line segments, bird's-eye chamfer shift (no segmentation)")
def lsd_chamfer(pair, ctx):
    return _chamfer_arm(pair, ctx, "flat", _lsd_sets)


@register("lsd_sem_chamfer", needs=NEEDS,
          config={**SEM_CONFIG, **CHAMFER_CONFIG, "lines": "cv2 LSD, length >= 20 px, "
                  f"kept within {LSD_SEM_PX} px of a Vistas curb/marking edge",
                  "camera_height_m": FLAT_M},
          description="LSD segments on Vistas curb/marking edges only, bird's-eye chamfer shift")
def lsd_sem_chamfer(pair, ctx):
    return _chamfer_arm(pair, ctx, "flat", lambda *a: _lsd_sets(*a, sem_filter=True))


@register("sem_snap", needs=NEEDS,
          config={**SEM_CONFIG, "snap_m": SNAP_M, "camera_height_m": FLAT_M},
          description="move W onto the other view's nearest curb edge (within 2.5 m)")
def sem_snap(pair, ctx):
    return _snap_arm(pair, ctx, shift=False)


@register("sem_curb_shift", needs=NEEDS,
          config={**SEM_CONFIG, "snap_m": SNAP_M, "camera_height_m": FLAT_M},
          description="shift W by (source's nearest curb point - other's nearest curb point)")
def sem_curb_shift(pair, ctx):
    return _snap_arm(pair, ctx, shift=True)


# --------------------------------------------------------------------------- #
# CLI: batch segmentation, and the doc's comparison tables
# --------------------------------------------------------------------------- #


def cmd_segment(args):
    """Segment every view in --views into --out/<name>.png (uint8 labels, curb cut
    suppressed) and write --out/manifest.json (sha256 per map, curb-cut pixel counts,
    wall-clock, device)."""
    import time
    import cv2
    import torch
    names = sorted(f for f in os.listdir(args.views) if f.endswith(".jpg"))
    os.makedirs(args.out, exist_ok=True)
    seg = Segmenter(args.device)
    t0 = time.time()
    man = {}
    for i in range(0, len(names), args.batch):
        chunk = names[i:i + args.batch]
        imgs = [cv2.imread(os.path.join(args.views, n)) for n in chunk]
        for n, (lab, ncc) in zip(chunk, seg(imgs)):
            p = os.path.join(args.out, n[:-4] + ".png")
            cv2.imwrite(p, lab)
            with open(p, "rb") as f:
                man[n[:-4]] = {"sha256": hashlib.sha256(f.read()).hexdigest(),
                               "curb_cut_px_suppressed": ncc}
        print(f"{i + len(chunk)}/{len(names)} {time.time() - t0:.0f} s", flush=True)
    meta = {"checkpoint": VISTAS_CHECKPOINT, "revision": VISTAS_REVISION,
            "input_hw": list(SEG_INPUT_HW), "device": seg.device,
            "gpu": torch.cuda.get_device_name(0) if seg.device == "cuda" else None,
            "elapsed_s": time.time() - t0, "n_views": len(names),
            "versions": {"torch": torch.__version__,
                         "transformers": __import__("transformers").__version__},
            "maps": man}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")


def paired(pairs, a, b, idx=None):
    """Median of (b - a) per pair (a's gain over b; positive = a closer) and its ramp
    bootstrap CI, over ``idx`` (default all). ``a`` / ``b``: arm_errors rows."""
    from collections import defaultdict
    idx = list(range(len(pairs))) if idx is None else idx
    by = defaultdict(list)
    for i in idx:
        by[pairs[i]["ramp_uid"]].append(i)
    groups = [v for _, v in sorted(by.items())]

    def st(ii):
        return float(np.median([b[i][0] - a[i][0] for i in ii]))

    def better(ii):
        return float(np.mean([a[i][0] < b[i][0] - 1e-9 for i in ii]))

    return {"n": len(idx), "median_gain_deg": st(idx) if idx else None,
            "ci": H.cluster_bootstrap(groups, st) if idx else None,
            "share_closer": better(idx) if idx else None,
            "median_deg": float(np.median([a[i][0] for i in idx])) if idx else None,
            "base_median_deg": float(np.median([b[i][0] for i in idx])) if idx else None,
            "within_2deg": float(np.mean([a[i][0] <= 2 for i in idx])) if idx else None,
            "base_within_2deg": float(np.mean([b[i][0] <= 2 for i in idx])) if idx else None}


def cmd_compare(args):
    """Paired comparisons of each arm against projection and proj_height_auto, over all
    pairs and over the pairs the arm did not fall back on, by imagery -> JSON."""
    pairs = H.read_frozen_pairs()
    base = {"projection": H.arm_errors(pairs, None),
            "proj_height_auto": H.arm_errors(pairs, H.read_predictions("proj_height_auto"))}
    out = {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT, "seed": H.SEED, "arms": {}}
    for arm in args.arms.split(","):
        preds = H.read_predictions(arm)
        e = H.arm_errors(pairs, preds)
        res = {}
        for stratum, keep in (("all", lambda p: True), ("gsv", lambda p: p["imagery"] == "gsv"),
                              ("mapillary", lambda p: p["imagery"] == "mapillary")):
            idx = [i for i, p in enumerate(pairs) if keep(p)]
            al = [i for i in idx if not e[i][3]]
            res[stratum] = {"fallback_rate": float(np.mean([e[i][3] for i in idx])),
                            **{f"vs_{bn}": paired(pairs, e, be, idx) for bn, be in base.items()},
                            **{f"aligned_vs_{bn}": paired(pairs, e, be, al)
                               for bn, be in base.items()}}
        moves = [r.get("t_m") for r in preds.values() if r.get("x") is not None and "t_m" in r]
        res["shift_m"] = {"median": float(np.median(moves)) if moves else None,
                          "p90": float(np.percentile(moves, 90)) if moves else None}
        reasons = {}
        for r in preds.values():
            if r.get("x") is None:
                k = r.get("reason") or next((k for k in ("no_geometry", "error") if k in r), "other")
                reasons[k] = reasons.get(k, 0) + 1
        res["fallback_reasons"] = reasons
        out["arms"][arm] = res
    H.write_json(args.out, out)
    for arm, r in out["arms"].items():
        a = r["all"]
        print(f"{arm:18s} fb {a['fallback_rate']:.2f}  "
              f"all vs proj {a['vs_projection']['median_gain_deg']:+.2f} {a['vs_projection']['ci']}  "
              f"aligned n={a['aligned_vs_projection']['n']} vs proj "
              f"{a['aligned_vs_projection']['median_gain_deg']} vs auto "
              f"{a['aligned_vs_proj_height_auto']['median_gain_deg']}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="semantic arms (#48): segmentation and tables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("segment")
    p.add_argument("--views", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=4)
    p.set_defaults(fn=cmd_segment)
    p = sub.add_parser("compare")
    p.add_argument("--arms", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
