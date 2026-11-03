"""Sidewalk width from one street-level photo, scored on the Seoul set (#217, arm 1).

Pipeline: Mask2Former Vistas label map -> per image row, the walkable span that contains
the walking line -> back-project both span ends to the ground plane at the known camera
height -> width across the path, yaw-corrected from the fitted edge lines.

Ground truth is the Seoul Sidewalk Accessibility Image Dataset (Zenodo 22699523; fetch
with ``seoul_fetch_217.py``): 514 iPhone photos taken 1.0 m above the ground from the
centre of the walking path, with laser-measured effective (unobstructed) width.

Three stages::

    # 1. GPU: label maps (makelab2 A40, ~10 min). Not committed; the committed run's maps
    #    are at makelab2:/homes/gws/jonf/sw217_seg and their sha256s in seg_meta.json.
    #    Env: NOT environment.yml -- see docs/sidewalk_width_217.md (torch 2.8.0+cu128,
    #    transformers 4.57.6, Python 3.9).
    python scripts/analysis/sidewalk_width_217.py segment \
        --images /homes/gws/jonf/seoul_sidewalk/images --out SEGDIR
    python scripts/analysis/sidewalk_width_217.py verify-seg --seg SEGDIR

    # 2. CPU: every configuration's width for every image -> one committed CSV
    python scripts/analysis/sidewalk_width_217.py measure --seg SEGDIR \
        --out analysis_out/sidewalk_width_217/widths.csv.gz

    # 3. CPU, from the committed CSV alone: tune on half A, report on half B
    python scripts/analysis/sidewalk_width_217.py score \
        --widths analysis_out/sidewalk_width_217/widths.csv.gz \
        --out analysis_out/sidewalk_width_217/results.json \
        --sensitivity-out analysis_out/sidewalk_width_217/sensitivity.json

Geometry (``backproject``): pinhole camera, principal point at the image centre, focal
length from the 35 mm equivalent (``F35_MM``; diagonal convention, 43.27 mm), camera
``h`` metres above a flat ground plane, pitched down by ``pitch`` radians, no roll.
Image u right, v down. World: X right, Z forward (horizontal), camera foot at origin.

Example -- a level camera 1 m up, f = 1000 px, a pixel 200 px below the centre and
100 px right of it lands 5 m ahead and 0.5 m right::

    >>> backproject(np.array([600.]), np.array([700.]), 1000., 500., 500., 1.0, 0.0)
    (array([0.5]), array([5.]))
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_CSV = os.path.join(REPO, "data", "seoul_sidewalk_217", "summary_attributes.csv")

# --------------------------------------------------------------------------- #
# constants (fixed before anything was scored)
# --------------------------------------------------------------------------- #

VISTAS_CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
VISTAS_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"   # as crossview_arms/semantic.py
WORK_W = 1440               # label maps are written at this width (aspect kept)
SEG_INPUT_HW = (768, 1024)  # Mask2Former input (h, w) for 4:3 landscape; transposed for portrait
CAMERA_H_M = 1.0            # the Seoul capture protocol
#: The Zenodo JPEGs carry NO EXIF (checked on all 514, 2026-09-30), so the focal length
#: cannot be read per image. The paper names two phones, an iPhone 17 (26 mm equivalent)
#: and an iPhone 16 Pro (24 mm equivalent), without saying which photo came from which.
#: 25 mm is the midpoint; either phone is then off by 4%. With the horizon row taken from
#: the image (the VP), width barely depends on f: a ground point's lateral offset is
#: X = h (u - cx) / (cos p (v - v_h)), so f enters only through cos p and through which
#: rows fall in the depth band (Z scales with f). On synthetic sidewalks f x0.9-1.1 changes
#: width by < 0.1% (tests/test_sidewalk_width_217.py::test_focal_error_barely_moves_width).
F35_MM = 25.0
DIAG_35MM = math.hypot(36.0, 24.0)
#: Every Vistas v1.2 class id this script relies on, checked against the checkpoint's
#: id2label at load time (verified against config.json at VISTAS_REVISION, 2026-09-30), so a
#: checkpoint whose ids differ fails loudly instead of silently re-meaning a class group.
EXPECTED_LABELS = {
    0: "Bird", 1: "Ground Animal", 2: "Curb", 5: "Barrier", 7: "Bike Lane", 9: "Curb Cut",
    11: "Pedestrian Area", 15: "Sidewalk", 19: "Person", 20: "Bicyclist",
    21: "Motorcyclist", 22: "Other Rider", 23: "Lane Marking - Crosswalk",
    24: "Lane Marking - General", 29: "Terrain", 30: "Vegetation", 32: "Banner",
    33: "Bench", 34: "Bike Rack", 35: "Billboard", 36: "Catch Basin", 37: "CCTV Camera",
    38: "Fire Hydrant", 39: "Junction Box", 40: "Mailbox", 41: "Manhole",
    42: "Phone Booth", 43: "Pothole", 44: "Street Light", 45: "Pole",
    46: "Traffic Sign Frame", 47: "Utility Pole", 48: "Traffic Light",
    49: "Traffic Sign (Back)", 50: "Traffic Sign (Front)", 51: "Trash Can", 52: "Bicycle",
    57: "Motorcycle", 62: "Wheeled Slow"}

#: Vistas v1.2 class groups. Every class not named here is a BOUNDARY: a span stops at it.
WALK_BASE = (15, 11, 9, 41, 36, 43)       # sidewalk, pedestrian area, curb cut, manhole,
#                                           catch basin, pothole (surface features on it)
WALK_SETS = {"base": WALK_BASE, "bike": WALK_BASE + (7,)}          # + Bike Lane
#: Not fixed: the GT excludes only permanent obstacles, so people and bicycles are passable
TRANSIENT = (0, 1, 19, 20, 21, 22, 52, 57, 62)
FURNITURE = (5, 32, 33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51)
#: fixed obstacles: street furniture (+ barrier/bollard), optionally vegetation + terrain
#: (tree pits, planters); where they are not obstacles they are boundaries.
#: These only change the TOTAL span. The clear span's passable set is TRANSIENT + markings,
#: so in clear mode every obstacle class (vegetation and terrain included) always ends the
#: span; for clear width the ``obst`` key only chooses which total spans the vanishing
#: point is read from (``image_vp``), i.e. it is a VP-source knob, not an obstacle rule.
OBSTACLE_SETS = {"furn": FURNITURE, "furn_veg": FURNITURE + (29, 30)}
#: Lane Marking - Crosswalk / - General. Seoul's yellow tactile paving strip, which runs
#: down many sidewalks, can be labelled a lane marking; as a boundary it cuts the span in
#: two. "surface" makes markings passable (still trimmed off the span ends).
MARK_SETS = {"boundary": (), "surface": (23, 24)}

BORDER_PX = 3               # a span end this close to the frame is truncated, row dropped
Z_MAX_M = 15.0              # rows farther than this are never used
PITCH_SENS_DEG = (-2.0, 2.0)
VP_MAX_PITCH_DEG = 15.0
YAW_MAX_DEG = 30.0

#: the tuning grid (stage 3 picks one cell per measure on half A)
BAND_ZMIN = (1.5, 2.5, 4.0)  # band starts at the first valid row at or beyond this depth
BAND_LEN = (1.0, 3.0)        # ... and is this long (m)
STATS = ("median", "p10")    # near-band width, or the 10th percentile ("min over the band")
HORIZONS = ("level", "vp", "vp_prior")

SPLIT_SEED = 217
GROUP_CELL_DEG = 0.001       # ~100 m lat/lon cells: neighbouring photos share a cell and a half
N_BOOT = 10000
THRESH_M = (1.2, 1.5)


# --------------------------------------------------------------------------- #
# geometry (pure, tested in tests/test_sidewalk_width_217.py)
# --------------------------------------------------------------------------- #


def focal_px(f35_mm, w_px, h_px):
    """Focal length in pixels from the 35 mm-equivalent focal length (diagonal convention).

    >>> round(focal_px(26, 5712, 4284), 1)
    4290.6
    """
    return f35_mm * math.hypot(w_px, h_px) / DIAG_35MM


def backproject(u, v, f, cx, cy, h, pitch):
    """Ground-plane (X, Z) of pixels (u, v); NaN where the ray does not hit the ground.

    ``pitch`` > 0 tilts the camera down. Exact for a pinhole with no roll."""
    dx = (np.asarray(u, float) - cx) / f
    dy = (np.asarray(v, float) - cy) / f
    c, s = math.cos(pitch), math.sin(pitch)
    down = dy * c + s                  # world-down component of the ray (camera z = 1)
    fwd = c - dy * s
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-9, h / down, np.nan)
    return t * dx, t * fwd


def horizon_row(f, cy, pitch):
    """Image row of the horizon for a camera pitched down by ``pitch``."""
    return cy - f * math.tan(pitch)


def row_spans(labels, seed_u, walk, passable):
    """For each row, the span of pixels connected to the walking line.

    A span is the maximal run of ``walk`` or ``passable`` pixels containing the seed pixel
    (the seed is moved to the nearest ``walk`` pixel within 5% of the width when it is not
    walkable), trimmed so both ends are ``walk`` pixels. Returns (left, right) int arrays,
    -1 where the row has no span. ``left``/``right`` are the outermost walkable columns."""
    H, W = labels.shape
    is_walk = np.isin(labels, walk)
    ok = is_walk | np.isin(labels, passable)
    left = np.full(H, -1)
    right = np.full(H, -1)
    reach = max(1, int(0.05 * W))
    s0 = int(round(seed_u))
    for r in range(H):
        wrow = is_walk[r]
        seed = s0
        if not wrow[seed]:
            lo, hi = max(0, seed - reach), min(W, seed + reach + 1)
            cand = np.flatnonzero(wrow[lo:hi])
            if cand.size == 0:
                continue
            seed = lo + cand[np.argmin(np.abs(cand + lo - s0))]
        orow = ok[r]
        # extend over ok pixels
        blk_l = np.flatnonzero(~orow[:seed])
        a = blk_l[-1] + 1 if blk_l.size else 0
        blk_r = np.flatnonzero(~orow[seed:])
        b = seed + blk_r[0] - 1 if blk_r.size else W - 1
        wl = np.flatnonzero(wrow[a:b + 1])          # trim to walkable ends
        left[r], right[r] = a + wl[0], a + wl[-1]
    return left, right


def robust_line(x, y, iters=3, k=2.5):
    """Fit y = a + b x with MAD-trimmed least squares. Returns (a, b, n_inliers)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan, 0
    for _ in range(iters):
        b, a = np.polyfit(x[m], y[m], 1)
        res = np.abs(y - (a + b * x))
        mad = np.median(res[m]) * 1.4826 + 1e-9
        new = np.isfinite(res) & (res <= k * mad + 1e-6)
        if new.sum() < 5 or (new == m).all():
            break
        m = new
    b, a = np.polyfit(x[m], y[m], 1)
    return a, b, int(m.sum())


def vp_pitch(left, right, rows, f, cy):
    """Pitch implied by the vanishing point of the two sidewalk edges, or None.

    Parallel ground lines meet on the horizon; with no roll the horizon row fixes the
    pitch. Edges are fitted as u = a + b v on the given rows."""
    aL, bL, nL = robust_line(rows, left[rows])
    aR, bR, nR = robust_line(rows, right[rows])
    if min(nL, nR) < 20 or not (bL < -0.05 and bR > 0.05):
        return None
    v_vp = (aR - aL) / (bL - bR)
    pitch = math.atan2(cy - v_vp, f)
    if abs(math.degrees(pitch)) > VP_MAX_PITCH_DEG:
        return None
    return pitch


def ground_widths(left, right, f, cx, cy, h, pitch):
    """Per-row (Z, width across the path) after yaw correction, NaN for unusable rows.

    Both edges are back-projected; the path direction is the mean slope dX/dZ of the two
    edge lines fitted in ground coordinates, and width is the separation of the two end
    points measured along the path's normal. Returns (Z, W, yaw_rad)."""
    H = left.shape[0]
    rows = np.arange(H, dtype=float)
    valid = (left >= 0) & (right > left)
    XL, ZL = backproject(np.where(valid, left - 0.5, np.nan), rows, f, cx, cy, h, pitch)
    XR, ZR = backproject(np.where(valid, right + 0.5, np.nan), rows, f, cx, cy, h, pitch)
    Zm = 0.5 * (ZL + ZR)
    use = valid & np.isfinite(Zm) & (Zm > 0) & (Zm <= Z_MAX_M)
    yaw = 0.0
    if use.sum() >= 10:
        _, bL, nL = robust_line(ZL[use], XL[use])
        _, bR, nR = robust_line(ZR[use], XR[use])
        slopes = [b for b, n in ((bL, nL), (bR, nR)) if n >= 10 and np.isfinite(b)]
        if slopes:
            yaw = math.atan(float(np.mean(slopes)))
            lim = math.radians(YAW_MAX_DEG)
            yaw = max(-lim, min(lim, yaw))
    nx, nz = math.cos(yaw), -math.sin(yaw)
    Wd = np.abs((XR - XL) * nx + (ZR - ZL) * nz)
    Wd = np.where(use, Wd, np.nan)
    return np.where(use, Zm, np.nan), Wd, yaw


def band_stat(Z, Wd, zmin, length, stat):
    """Width over the band [z0, z0 + length], z0 = the first valid depth >= zmin."""
    m = np.isfinite(Z) & np.isfinite(Wd) & (Z >= zmin)
    if not m.any():
        return np.nan
    z0 = Z[m].min()
    b = m & (Z <= z0 + length)
    if b.sum() < 3:
        return np.nan
    w = Wd[b]
    return float(np.median(w) if stat == "median" else np.percentile(w, 10))


def exclude_border(left, right, width):
    """Drop rows whose span touches the frame: the true edge is outside the image."""
    bad = (left < BORDER_PX) | (right > width - 1 - BORDER_PX)
    return np.where(bad, -1, left), np.where(bad, -1, right)


# --------------------------------------------------------------------------- #
# stage 1: segment
# --------------------------------------------------------------------------- #


def _exif(img):
    ex = img.getexif()
    sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}

    def g(tag):
        v = sub.get(tag, ex.get(tag))
        return float(v) if v is not None else None
    return {"make": ex.get(0x010F), "model": ex.get(0x0110), "orientation": ex.get(0x0112),
            "focal_mm": g(0x920A), "f35_mm": g(0xA405), "lens": sub.get(0xA434)}


def segment(args):
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(VISTAS_CHECKPOINT, revision=VISTAS_REVISION)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        VISTAS_CHECKPOINT, revision=VISTAS_REVISION).to(dev).eval()
    for cid, name in EXPECTED_LABELS.items():
        got = model.config.id2label.get(cid, model.config.id2label.get(str(cid)))
        if got != name:
            raise RuntimeError(f"class {cid} is {got!r}, expected {name!r}")
    os.makedirs(args.out, exist_ok=True)
    gt = _read_gt()
    # the CSV names IMG_8956.JPG and IMG_4282.HEIC; the archives hold IMG_8956.jpg and
    # IMG_4282.jpg. Match on the case-folded stem.
    stem = lambda n: os.path.splitext(n)[0].lower()  # noqa: E731
    by_stem = {stem(n): n for n in gt}
    assert len(by_stem) == len(gt), "two GT rows share a stem"
    paths = {}
    for dirpath, _, names in os.walk(args.images):
        for n in names:
            if stem(n) in by_stem and not n.startswith("._"):
                paths[by_stem[stem(n)]] = os.path.join(dirpath, n)
    missing = sorted(set(gt) - set(paths))
    if missing:
        sys.exit(f"{len(missing)} GT images not found under {args.images}: {missing[:5]}")
    names = sorted(paths)[: args.limit] if args.limit else sorted(paths)
    meta = {}
    t_loop = time.time()
    for i, n in enumerate(names):
        raw = Image.open(paths[n])
        ex = _exif(raw)
        img = ImageOps.exif_transpose(raw).convert("RGB")
        W0, H0 = img.size
        ww = WORK_W
        wh = int(round(H0 * ww / W0))
        small = img.resize((ww, wh), Image.BILINEAR)
        ih, iw = SEG_INPUT_HW if W0 >= H0 else SEG_INPUT_HW[::-1]
        proc.size = {"height": ih, "width": iw}
        inp = proc(images=small, return_tensors="pt")
        with torch.inference_mode():
            out = model(pixel_values=inp["pixel_values"].to(dev))
            cls = out.class_queries_logits.softmax(-1)[..., :-1]
            msk = out.masks_queries_logits.sigmoid()
            sc = torch.einsum("bqc,bqhw->bchw", cls, msk)
            sc = torch.nn.functional.interpolate(sc, size=(wh, ww), mode="bilinear",
                                                 align_corners=False)
            lab = sc.argmax(1)[0].to(torch.uint8).cpu().numpy()
        png = os.path.join(args.out, os.path.splitext(n)[0] + ".png")
        Image.fromarray(lab).save(png, optimize=True)
        with open(png, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        meta[n] = dict(ex, orig_w=W0, orig_h=H0, work_w=ww, work_h=wh, seg_input=[ih, iw],
                       label_png_sha256=sha)
        if i % 50 == 0:
            print(f"{i}/{len(names)} {n} {W0}x{H0} f35={ex['f35_mm']}", flush=True)
    t_end = time.time()
    run = {"checkpoint": VISTAS_CHECKPOINT, "revision": VISTAS_REVISION, "device": dev,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "work_w": WORK_W, "n": len(names), "elapsed_s": round(t_end - t0, 1),
           "loop_s": round(t_end - t_loop, 1),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8", newline="") as f:
        json.dump({"run": run, "images": meta}, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(run))


# --------------------------------------------------------------------------- #
# stage 2: measure
# --------------------------------------------------------------------------- #


def _read_gt(path=GT_CSV):
    with open(path, encoding="utf-8-sig") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def image_spans(lab):
    """{(walk, obst, mark, measure): (left, right)} border-trimmed spans for one label map."""
    Ww = lab.shape[1]
    cx = (Ww - 1) / 2.0
    out = {}
    for wname, walk in WALK_SETS.items():
        for oname, obst in OBSTACLE_SETS.items():
            for mname, mark in MARK_SETS.items():
                for meas, passable in (("total", obst + TRANSIENT + mark),
                                       ("clear", TRANSIENT + mark)):
                    l, r = row_spans(lab, cx, walk, passable)
                    out[(wname, oname, mname, meas)] = exclude_border(l, r, Ww)
    return out


def image_vp(spans, f, shape, h=CAMERA_H_M):
    """{(walk, obst, mark): VP pitch or None}, from the total spans (the physical edges)."""
    Hh = shape[0]
    cy = (Hh - 1) / 2.0
    # rows within Z_MAX_M of a level camera: nearer the horizon the edges are noise
    below = np.arange(int(math.ceil(cy + f * h / Z_MAX_M)), Hh)
    out = {}
    for (w, o, m, meas), (l, r) in spans.items():
        if meas != "total":
            continue
        rows = below[l[below] >= 0]
        out[(w, o, m)] = vp_pitch(l, r, rows, f, cy) if rows.size else None
    return out


def image_rows(spans, vps, prior, f, shape, h=CAMERA_H_M):
    """Wide rows: one per (walk, obst, mark, horizon, dpitch, measure), a column per band cell.

    Horizons: ``level`` (pitch 0), ``vp`` (the edges' vanishing point; no estimate when it
    is not found), ``vp_prior`` (the VP, else ``prior[(walk, obst, mark)]``, the median VP
    pitch over half A)."""
    Hh, Ww = shape
    cx, cy = (Ww - 1) / 2.0, (Hh - 1) / 2.0
    for (w, o, m, meas), (l, r) in spans.items():
        p_vp = vps[(w, o, m)]
        for hname in HORIZONS:
            if hname == "level":
                base = 0.0
            elif hname == "vp":
                base = p_vp
            else:
                base = p_vp if p_vp is not None else prior[(w, o, m)]
            for dp in (0.0,) + PITCH_SENS_DEG:
                row = {"walk": w, "obst": o, "mark": m, "horizon": hname,
                       "vp_found": int(p_vp is not None), "dpitch": dp, "measure": meas}
                if base is None:
                    row.update(pitch_deg="", yaw_deg="")
                    row.update({c: "" for c in BAND_COLS})
                    yield row
                    continue
                pitch = base + math.radians(dp)
                Z, Wd, yaw = ground_widths(l, r, f, cx, cy, h, pitch)
                row.update(pitch_deg=f"{math.degrees(pitch):.3f}",
                           yaw_deg=f"{math.degrees(yaw):.2f}")
                for zmin, ln, st in BAND_CELLS:
                    wv = band_stat(Z, Wd, zmin, ln, st)
                    row[band_col(zmin, ln, st)] = "" if not np.isfinite(wv) else f"{wv:.3f}"
                yield row


def band_col(zmin, ln, st):
    return f"w_z{zmin:g}_l{ln:g}_{st}"


BAND_CELLS = [(z, ln, st) for z in BAND_ZMIN for ln in BAND_LEN for st in STATS]
BAND_COLS = [band_col(*c) for c in BAND_CELLS]
FIELDS = (["filename", "walk", "obst", "mark", "horizon", "vp_found", "dpitch", "measure",
           "pitch_deg", "yaw_deg"] + BAND_COLS)


SEG_META = os.path.join(REPO, "analysis_out", "sidewalk_width_217", "seg_meta.json")


def verify_seg(args):
    """Check a directory of label maps against the committed sha256 values (seg_meta.json).
    A re-run of ``segment`` on other hardware may differ (GPU nondeterminism); this says how
    many maps did, so a drifted widths CSV is explained rather than silently different."""
    with open(args.ref, encoding="utf-8") as f:
        want = json.load(f)["images"]
    missing, bad = [], []
    for n, m in sorted(want.items()):
        p = os.path.join(args.seg, os.path.splitext(n)[0] + ".png")
        if not os.path.exists(p):
            missing.append(n)
            continue
        with open(p, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != m["label_png_sha256"]:
                bad.append(n)
    print(f"{len(want)} label maps in {args.ref}: missing {len(missing)}, "
          f"sha256 mismatch {len(bad)} {bad[:5]}")
    if missing or bad:
        sys.exit(1)
    print("label maps identical to the committed run")


def measure(args):
    """Two passes: VP pitches for every image (the prior is their median over half A),
    then every configuration's widths."""
    from PIL import Image
    t0 = time.time()
    with open(os.path.join(args.seg, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)["images"]
    gt = _read_gt()
    half, _ = split_groups(gt)
    names = [n for n in sorted(gt) if n in meta]
    if len(names) != len(gt) and not args.allow_partial:
        sys.exit(f"label maps for {len(names)} of {len(gt)} GT images")
    cache = {}
    for n in names:
        m = meta[n]
        lab = np.array(Image.open(os.path.join(args.seg, os.path.splitext(n)[0] + ".png")))
        f35 = m.get("f35_mm") or args.f35     # EXIF when present (it is not, on Seoul)
        f = focal_px(f35, m["work_w"], m["work_h"])
        spans = image_spans(lab)
        cache[n] = (spans, image_vp(spans, f, lab.shape), f, lab.shape)
    keys = list(next(iter(cache.values()))[1])
    prior = {}
    for k in keys:
        vals = [cache[n][1][k] for n in names if half[n] == "A" and cache[n][1][k] is not None]
        prior[k] = float(np.median(vals)) if vals else 0.0
    out_rows = []
    for n in names:
        spans, vps, f, shape = cache[n]
        for row in image_rows(spans, vps, prior, f, shape):
            row["filename"] = n
            out_rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    buf = io.StringIO(newline="")
    wr = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(out_rows)
    # mtime=0 and no filename: the gzip bytes depend only on the content
    with open(args.out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                    filename="") as g:
        g.write(buf.getvalue().encode("utf-8"))
    side = os.path.splitext(os.path.splitext(args.out)[0])[0] + "_prior.json"
    with open(side, "w", encoding="utf-8", newline="") as fh:
        json.dump({"f35_mm": args.f35, "n_images": len(names),
                   "vp_prior_pitch_deg_from_half_A":
                   {"|".join(k): round(math.degrees(v), 3) for k, v in prior.items()},
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"{len(out_rows)} rows for {len(names)} images in {time.time() - t0:.0f} s "
          f"-> {args.out}")


# --------------------------------------------------------------------------- #
# stage 3: score
# --------------------------------------------------------------------------- #


def split_groups(gt):
    """Half A / half B by ~100 m cell, so a sidewalk run photographed several times does
    not sit in both halves. Cells are stratified by whether they hold any GT < 1.2 m photo
    (there are only 22 such photos), and each stratum is split in half by a seeded shuffle.
    Returns ({filename: 'A'|'B'}, {filename: cell})."""
    groups = {n: f"{math.floor(float(r['latitude']) / GROUP_CELL_DEG)}_"
                 f"{math.floor(float(r['longitude']) / GROUP_CELL_DEG)}" for n, r in gt.items()}
    narrow = {g for n, g in groups.items() if float(gt[n]["width"]) < THRESH_M[0]}
    rng = np.random.default_rng(SPLIT_SEED)
    half = {}
    for stratum in (sorted(narrow), sorted(set(groups.values()) - narrow)):
        perm = rng.permutation(len(stratum))
        for j, i in enumerate(perm):
            half[stratum[i]] = "A" if j < (len(stratum) + 1) // 2 else "B"
    return {n: half[g] for n, g in groups.items()}, groups


def cls3(w):
    return np.where(w < THRESH_M[0], 0, np.where(w < THRESH_M[1], 1, 2))


def metrics(est, gt):
    """Scalar metrics on paired arrays; NaN estimates are failures (counted, not flagged)."""
    ok = np.isfinite(est)
    e, g = est[ok], gt[ok]
    out = {"n": int(len(gt)), "n_estimated": int(ok.sum()),
           "coverage": float(ok.mean()) if len(gt) else np.nan}
    if ok.sum():
        err = e - g
        out.update(mae=float(np.abs(err).mean()), bias_mean=float(err.mean()),
                   bias_median=float(np.median(err)),
                   rel_mae=float((np.abs(err) / g).mean()),
                   within_0_3=float((np.abs(err) <= 0.3).mean()),
                   within_0_5=float((np.abs(err) <= 0.5).mean()),
                   # the paper's VLMs are judged by interval width: the error quantiles
                   # give the empirical 90% band of this estimator for comparison
                   err_q05=float(np.percentile(err, 5)),
                   err_q95=float(np.percentile(err, 95)),
                   abs_err_q90=float(np.percentile(np.abs(err), 90)))
    narrow_gt = gt < THRESH_M[0]
    flagged = ok & (np.nan_to_num(est, nan=np.inf) < THRESH_M[0])
    tp = int((flagged & narrow_gt).sum())
    out.update(n_narrow_gt=int(narrow_gt.sum()), n_flagged=int(flagged.sum()), tp=tp,
               recall_lt_1_2=tp / narrow_gt.sum() if narrow_gt.sum() else np.nan,
               precision_lt_1_2=tp / flagged.sum() if flagged.sum() else np.nan)
    if ok.sum():
        # acc3 is over estimated photos only; acc3_all counts "no estimate" as wrong, the
        # same denominator as recall and precision
        out["acc3"] = float((cls3(e) == cls3(g)).mean())
    if len(gt):
        out["acc3_all"] = float((ok & (cls3(np.nan_to_num(est, nan=99)) == cls3(gt))).mean())
    return out


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI for k of n, by bisection on the binomial tail (no scipy).

    >>> [round(x, 3) for x in clopper_pearson(9, 9)]
    [0.664, 1.0]
    """
    if n == 0:
        return [float("nan"), float("nan")]

    def tail_ge(p):   # P(X >= k)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):   # P(X <= k)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(fn, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (fn(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(tail_ge, alpha / 2, True)
    upper = 1.0 if k == n else solve(tail_le, alpha / 2, False)
    return [lower, upper]


def by_gt_bin(est, gt):
    """MAE and bias by GT width bin: where the error comes from."""
    out = {}
    for lo, hi in ((0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 99.0)):
        m = (gt >= lo) & (gt < hi)
        ok = m & np.isfinite(est)
        err = est[ok] - gt[ok]
        rel = err / gt[ok]
        out[f"{lo:g}-{hi:g}m"] = {"n": int(m.sum()), "n_estimated": int(ok.sum()),
                                  "mae": float(np.abs(err).mean()) if ok.any() else None,
                                  "bias_mean": float(err.mean()) if ok.any() else None,
                                  "rel_mae": float(np.abs(rel).mean()) if ok.any() else None,
                                  "rel_bias_mean": float(rel.mean()) if ok.any() else None,
                                  "rel_bias_median": float(np.median(rel))
                                  if ok.any() else None}
    return out


def confusion(est, gt):
    ok = np.isfinite(est)
    cm = np.zeros((3, 4), int)                 # rows GT class, cols est class + "none"
    ge = cls3(gt)
    ee = np.where(ok, cls3(np.nan_to_num(est, nan=99)), 3)
    for a, b in zip(ge, ee):
        cm[a, b] += 1
    return cm.tolist()


def bootstrap(est, gt, grp, keys, n_boot=N_BOOT, seed=SPLIT_SEED):
    """95% percentile CIs, resampling ~100 m groups (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(grp)
    idx = {g: np.flatnonzero(grp == g) for g in ug}
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        m = metrics(est[pick], gt[pick])
        for k in keys:
            draws[k].append(m.get(k, np.nan))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items()}


CFG_KEYS = ("walk", "obst", "mark", "horizon", "zmin", "band", "stat")


def load_table(path):
    """{(measure, dpitch, walk, obst, mark, horizon, zmin, band, stat): {filename: width}}
    plus {(walk, obst, mark): {filename: vp_found}} from the wide widths CSV."""
    table, vp = {}, {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            vp.setdefault((r["walk"], r["obst"], r["mark"]), {})[r["filename"]] = \
                int(r["vp_found"])
            head = (r["measure"], float(r["dpitch"]), r["walk"], r["obst"], r["mark"],
                    r["horizon"])
            for (zmin, ln, st), col in zip(BAND_CELLS, BAND_COLS):
                table.setdefault(head + (zmin, ln, st), {})[r["filename"]] = (
                    float(r[col]) if r[col] else np.nan)
    return table, vp


def _clean(o):
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if not math.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def series(n):
    """Filename series (IMG_4xxx / IMG_6xxx / IMG_8xxx-9xxx): a proxy for capture session."""
    return n[4] if n[4] in "46" else "8-9"


def tune(table, meas, names, A, G, min_coverage):
    """Every dpitch-0 configuration of ``meas`` that estimates at least ``min_coverage`` of
    half A, as (mae_A, key, metrics_A), best first (ties broken by the key's text)."""
    scored = []
    for k, d in table.items():
        if k[0] != meas or k[1] != 0.0:
            continue
        e = np.array([d.get(n, np.nan) for n in names])
        mA = metrics(e[A], G[A])
        if "mae" in mA and mA["coverage"] >= min_coverage:
            scored.append((mA["mae"], k, mA))
    scored.sort(key=lambda t: (t[0], str(t[1])))
    return scored


def score(args):
    gt = _read_gt()
    half, groups = split_groups(gt)
    table, vp = load_table(args.widths)
    names = sorted(gt)
    G = np.array([float(gt[n]["width"]) for n in names])
    H = np.array([half[n] for n in names])
    GR = np.array([groups[n] for n in names])
    SE = np.array([series(n) for n in names])
    A, B = H == "A", H == "B"

    def arr(k):
        d = table[k]
        return np.array([d.get(n, np.nan) for n in names])

    boot_keys = ["mae", "bias_mean", "bias_median", "rel_mae", "recall_lt_1_2",
                 "precision_lt_1_2", "acc3", "acc3_all", "coverage", "abs_err_q90"]
    res = {"split": {"seed": SPLIT_SEED, "group_cell_deg": GROUP_CELL_DEG,
                     "n_A": int(A.sum()), "n_B": int(B.sum()),
                     "groups_A": len(set(GR[A])), "groups_B": len(set(GR[B])),
                     "narrow_A": int((G[A] < 1.2).sum()), "narrow_B": int((G[B] < 1.2).sum())},
           "gt": {"n": len(names), "mean": float(G.mean()), "min": float(G.min()),
                  "max": float(G.max()), "n_lt_1_2": int((G < 1.2).sum()),
                  "n_lt_1_5": int((G < 1.5).sum())},
           "tuning_rule": "lowest MAE on half A among configurations that estimate at least "
                          f"{args.min_coverage:.0%} of half A",
           "measures": {}}
    for meas in ("clear", "total"):
        scored = tune(table, meas, names, A, G, args.min_coverage)
        best_mae, best, mA = scored[0]
        cfg = dict(zip(CFG_KEYS, best[2:]))
        e = arr(best)
        mB = metrics(e[B], G[B])
        ciB = bootstrap(e[B], G[B], GR[B], boot_keys, n_boot=args.n_boot)
        sens = {}
        for dp in PITCH_SENS_DEG:
            es = arr((meas, dp) + best[2:])
            ok = np.isfinite(es[B]) & np.isfinite(e[B])
            mm = metrics(es[B], G[B])
            sens[f"{dp:+.0f}deg"] = {
                "median_rel_change": float(np.median(es[B][ok] / e[B][ok] - 1)),
                "mae_B": mm.get("mae"), "bias_mean_B": mm.get("bias_mean")}
        # a single scale fitted on half A (median GT/estimate), applied to B -- reported
        # beside the raw number, because it absorbs the unknown focal length and height
        okA = np.isfinite(e[A])
        ratio = float(np.median(G[A][okA] / e[A][okA]))
        mBcal = metrics(e[B] * ratio, G[B])
        ciBcal = bootstrap(e[B] * ratio, G[B], GR[B], boot_keys, n_boot=args.n_boot)
        # the matching level-horizon configuration: what the VP buys
        lvl = (meas, 0.0) + best[2:5] + ("level",) + best[6:]
        mBlvl = metrics(arr(lvl)[B], G[B])
        by_series = {s_: metrics(e[B & (SE == s_)], G[B & (SE == s_)])
                     for s_ in sorted(set(SE))}
        vpd = vp[best[2:5]]
        res["measures"][meas] = {
            "config": cfg, "tuning_mae_A": best_mae, "metrics_A": mA,
            "n_configs_considered": len(scored),
            "top5_A": [{"mae_A": t[0], "coverage_A": t[2]["coverage"],
                        "config": dict(zip(CFG_KEYS, t[1][2:]))} for t in scored[:5]],
            "metrics_B": mB, "ci95_B": ciB, "confusion_B": confusion(e[B], G[B]),
            "exact_ci95_B": {
                "recall_lt_1_2": clopper_pearson(mB["tp"], mB["n_narrow_gt"]),
                "precision_lt_1_2": clopper_pearson(mB["tp"], mB["n_flagged"])},
            "by_gt_bin_B": by_gt_bin(e[B], G[B]),
            "pitch_sensitivity_B": sens,
            "level_horizon_same_config_B": mBlvl,
            "calibrated_B": {"scale_from_A": ratio, "metrics": mBcal, "ci95": ciBcal,
                             "confusion": confusion(e[B] * ratio, G[B])},
            "by_series_B": by_series,
            "vp_found_rate": {"A": float(np.mean([vpd[n] for n in names if half[n] == "A"])),
                              "B": float(np.mean([vpd[n] for n in names if half[n] == "B"]))},
            "per_image": {n: (None if not np.isfinite(x) else round(float(x), 3))
                          for n, x in zip(names, e)}}
    res["half"] = {n: half[n] for n in names}
    res["confusion_layout"] = ("rows: GT <1.2, 1.2-1.5, >=1.5; cols: estimate <1.2, "
                               "1.2-1.5, >=1.5, no estimate")
    with open(args.widths, "rb") as fh:
        res["widths_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    res["n_boot"] = args.n_boot
    res = _clean(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    for meas, r in res["measures"].items():
        print(meas, r["config"])
        print("  A mae", r["tuning_mae_A"], " B:", {k: r["metrics_B"].get(k) for k in boot_keys})
        print("  B CI:", r["ci95_B"])
        print("  B calibrated x%.3f:" % r["calibrated_B"]["scale_from_A"],
              {k: r["calibrated_B"]["metrics"].get(k) for k in boot_keys})
        print("  level horizon B mae", r["level_horizon_same_config_B"].get("mae"),
              "vp found", r["vp_found_rate"])
    if args.sensitivity_out:
        sens = sensitivity(table, args.widths, gt, half, groups, names,
                           {m: res["measures"][m]["config"] for m in res["measures"]},
                           args.min_coverage)
        with open(args.sensitivity_out, "w", encoding="utf-8", newline="") as f:
            json.dump(_clean(sens), f, indent=1, sort_keys=True)
            f.write("\n")
        print("sensitivity ->", args.sensitivity_out)


# --------------------------------------------------------------------------- #
# sensitivity reads (#225 review S1, S4, S5, S6; N2): disclosed beside the headline,
# never used to choose anything. All from the committed widths CSV + GT table.
# --------------------------------------------------------------------------- #

#: Development contact before scoring: per-image output was printed for the half-B photos
#: IMG_4293-IMG_4335 while debugging (docs/sidewalk_width_217.md, caveats).
CONTACT_B_RANGE = (4293, 4335)
#: The VP pitch cap before that contact; it was raised to VP_MAX_PITCH_DEG by hand.
PRE_CONTACT_VP_CAP_DEG = 10.0
LEAK_RADII_M = (10.0, 20.0, 30.0)
N_BOOT_SENS = 2000


def img_number(n):
    """IMG_4293.HEIC -> 4293."""
    return int(os.path.splitext(n)[0].split("_")[1])


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (broadcasts)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def load_vp_pitch(path):
    """{(walk, obst, mark): {filename: VP pitch in degrees, NaN when no VP}} (pitch > 0 is
    down), read from the ``vp`` horizon rows at dpitch 0."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["horizon"] == "vp" and float(r["dpitch"]) == 0.0 and r["measure"] == "total":
                out.setdefault((r["walk"], r["obst"], r["mark"]), {})[r["filename"]] = (
                    float(r["pitch_deg"]) if r["pitch_deg"] else np.nan)
    return out


def _rank_avg(a):
    """Ranks with ties given their average rank (as scipy.stats.rankdata). Bootstrap
    resamples always repeat rows, so tie handling must not depend on the sort algorithm:
    an argsort-of-argsort rank did, and the CI's last digit moved between numpy builds.

    >>> _rank_avg([3.0, 1.0, 3.0, 2.0]).tolist()
    [2.5, 0.0, 2.5, 1.0]
    """
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    s = a[order]
    new = np.r_[True, s[1:] != s[:-1]]
    first = np.flatnonzero(new)
    counts = np.diff(np.r_[first, len(s)])
    r = np.empty(len(a))
    r[order] = (first + (counts - 1) / 2.0)[np.cumsum(new) - 1]
    return r


def _spearman(x, y):
    return float(np.corrcoef(_rank_avg(x), _rank_avg(y))[0, 1])


def _brief(m):
    keys = ("n", "n_estimated", "coverage", "mae", "bias_mean", "rel_mae", "acc3",
            "acc3_all", "tp", "n_narrow_gt", "n_flagged", "recall_lt_1_2", "precision_lt_1_2")
    return {k: m.get(k) for k in keys}


def sensitivity(table, widths_path, gt, half, groups, names, chosen, min_coverage):
    """Counterfactual and leak reads for the doc's sensitivity table. Nothing here feeds
    back into the tuned configuration or the headline."""
    G = np.array([float(gt[n]["width"]) for n in names])
    A = np.array([half[n] == "A" for n in names])
    B = ~A
    GR = np.array([groups[n] for n in names])
    num = np.array([img_number(n) for n in names])
    lat = np.array([float(gt[n]["latitude"]) for n in names])
    lon = np.array([float(gt[n]["longitude"]) for n in names])
    vpp = load_vp_pitch(widths_path)

    def arr(k, tab=table):
        return np.array([tab[k].get(n, np.nan) for n in names])

    def key(meas, cfg, **over):
        c = dict(cfg, **over)
        return (meas, 0.0) + tuple(c[k] for k in CFG_KEYS)

    contact = B & (num >= CONTACT_B_RANGE[0]) & (num <= CONTACT_B_RANGE[1])
    contact_cells = sorted(set(GR[contact]))
    run = (num >= CONTACT_B_RANGE[0]) & (num <= CONTACT_B_RANGE[1])
    out = {"contact_B": {"range": list(CONTACT_B_RANGE), "n_photos": int(contact.sum()),
                         "cells_B": contact_cells,
                         "run_cells_by_half": {g: sorted({str(h) for h in
                                                          np.where(A, "A", "B")[run & (GR == g)]})
                                               for g in sorted(set(GR[run]))},
                         "run_photos_in_A": [n for n, r_, a in zip(names, run, A) if r_ and a]},
           "measures": {}}

    # S4: how close is each half-B photo to the nearest half-A photo?
    d = haversine_m(lat[B][:, None], lon[B][:, None], lat[A][None, :], lon[A][None, :])
    nn = d.min(1)
    out["leak_B_to_A"] = {"n_B": int(B.sum()),
                          "nearest_A_m_median": float(np.median(nn)),
                          **{f"within_{r:g}m": int((nn <= r).sum()) for r in LEAK_RADII_M}}
    in10 = np.zeros(len(names), bool)
    in10[np.flatnonzero(B)[nn <= LEAK_RADII_M[0]]] = True

    for meas, cfg in chosen.items():
        k0 = key(meas, cfg)
        e = arr(k0)
        vk = (cfg["walk"], cfg["obst"], cfg["mark"])
        pv = np.array([vpp[vk].get(n, np.nan) for n in names])
        prior = float(np.nanmedian(pv[A]))
        r = {"published_B": _brief(metrics(e[B], G[B]))}

        # S6: the same cell with the vp_prior horizon (VP, else half-A median pitch)
        r["vp_prior_same_cell_B"] = _brief(metrics(arr(key(meas, cfg, horizon="vp_prior"))[B],
                                                   G[B]))

        # S5a: the pre-contact 10 deg VP cap, re-tuned on A by the same rule. A photo whose
        # VP implies more than 10 deg gets no estimate under the vp horizon. Under vp_prior it
        # would fall back to the prior pitch, but those widths are not in the CSV, so they
        # are dropped here too: this is conservative for coverage, and the prior itself (a
        # median over half A) moves by at most the few photos above the cap.
        capped = {}
        for k, dct in table.items():
            if k[0] != meas or k[1] != 0.0:
                continue
            if k[5] in ("vp", "vp_prior"):
                pk = vpp[k[2:5]]
                dct = {n: (np.nan if abs(pk.get(n, np.nan)) > PRE_CONTACT_VP_CAP_DEG else w)
                       for n, w in dct.items()}
            capped[k] = dct
        sc = tune(capped, meas, names, A, G, min_coverage)
        kc = sc[0][1]
        over = np.abs(pv) > PRE_CONTACT_VP_CAP_DEG
        r["cap10_retuned"] = {
            "config": dict(zip(CFG_KEYS, kc[2:])), "mae_A": sc[0][0],
            "metrics_B": _brief(metrics(arr(kc, capped)[B], G[B])),
            "n_vp_over_cap_A": int((over & A).sum()), "n_vp_over_cap_B": int((over & B).sum()),
            "n_vp_over_cap_B_contacted": int((over & contact).sum())}

        # S5b: markings as a boundary, same cell otherwise, and the best such cell overall
        kb = key(meas, cfg, mark="boundary")
        eb = arr(kb)
        best_b = [t for t in tune(table, meas, names, A, G, 0.0) if t[1][4] == "boundary"]
        best_b_ok = [t for t in best_b if t[2]["coverage"] >= min_coverage]
        r["markings_boundary"] = {
            "same_cell": {"coverage_A": metrics(eb[A], G[A])["coverage"],
                          "mae_A": metrics(eb[A], G[A]).get("mae"),
                          "metrics_B": _brief(metrics(eb[B], G[B]))},
            "max_coverage_A_any_boundary_cell": max(t[2]["coverage"] for t in best_b),
            "n_boundary_cells_passing_coverage": len(best_b_ok),
            "best_passing_boundary_cell": None if not best_b_ok else {
                "config": dict(zip(CFG_KEYS, best_b_ok[0][1][2:])), "mae_A": best_b_ok[0][0],
                "coverage_A": best_b_ok[0][2]["coverage"],
                "metrics_B": _brief(metrics(arr(best_b_ok[0][1])[B], G[B]))}}

        # S5c: half B without the cells that held the contacted photos; and those photos
        keep = B & ~np.isin(GR, contact_cells)
        r["drop_contacted_cells_B"] = _brief(metrics(e[keep], G[keep]))
        r["contacted_photos_B"] = _brief(metrics(e[contact], G[contact]))
        r["contacted_photos_B"]["gt_min"] = float(G[contact].min())
        # S4: half B without the photos that have a half-A photo within 10 m
        keep10 = B & ~in10
        r["drop_B_within_10m_of_A"] = _brief(metrics(e[keep10], G[keep10]))

        # S1: is per-image error explained by pitch? |relative error| against how far the
        # photo's VP pitch is from the half-A median (a proxy for how wrong a fixed-pitch
        # model would be, and for VP-fit trouble). Estimated half-B photos with a VP.
        ok = B & np.isfinite(e) & np.isfinite(pv)
        x = np.abs(pv[ok] - prior)
        y = np.abs(e[ok] - G[ok]) / G[ok]
        rho = _spearman(x, y)
        rng = np.random.default_rng(SPLIT_SEED)
        grp = GR[ok]
        ug = np.unique(grp)
        idx = {g: np.flatnonzero(grp == g) for g in ug}
        draws = []
        for _ in range(N_BOOT_SENS):
            pick = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug), replace=True)])
            draws.append(_spearman(x[pick], y[pick]))
        q = np.quantile(x, [1 / 3, 2 / 3])
        terc = np.digitize(x, q)
        r["pitch_vs_error_B"] = {
            "n": int(ok.sum()), "prior_pitch_deg": prior, "spearman_rho": rho,
            "spearman_rho_ci95": [float(np.nanpercentile(draws, 2.5)),
                                  float(np.nanpercentile(draws, 97.5))],
            "tercile_edges_deg": [float(v) for v in q],
            "median_abs_rel_err_by_tercile": [float(np.median(y[terc == t])) for t in range(3)],
            "mean_abs_rel_err_by_tercile": [float(np.mean(y[terc == t])) for t in range(3)]}

        # N2: the VP pitch distribution by half (negative = camera tilted up)
        def dist(mask):
            v = pv[mask & np.isfinite(pv)]
            nm = np.array(names)[mask & np.isfinite(pv)]
            return {"n": int(v.size), "median_deg": float(np.median(v)),
                    "n_up_beyond_8deg": int((v < -8).sum()),
                    "n_abs_over_10deg": int((np.abs(v) > 10).sum()),
                    "max_up_deg": float(-v.min()), "max_up_image": str(nm[np.argmin(v)])}
        r["vp_pitch_by_half"] = {"A": dist(A), "B": dist(B)}
        out["measures"][meas] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("segment")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("verify-seg")
    p.add_argument("--seg", required=True)
    p.add_argument("--ref", default=SEG_META)
    p = sub.add_parser("measure")
    p.add_argument("--seg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--f35", type=float, default=F35_MM,
                   help="35 mm-equivalent focal length when EXIF has none")
    p.add_argument("--allow-partial", action="store_true",
                   help="debugging only: measure whatever label maps exist")
    p = sub.add_parser("score")
    p.add_argument("--widths", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--min-coverage", type=float, default=0.9)
    p.add_argument("--sensitivity-out", default=None,
                   help="also write the disclosed sensitivity reads (10-deg VP cap re-tune, "
                        "contacted cells dropped, split leak, pitch vs error) to this JSON")
    args = ap.parse_args()
    {"segment": segment, "verify-seg": verify_seg, "measure": measure,
     "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
