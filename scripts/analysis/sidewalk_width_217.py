"""Sidewalk width from one street-level photo, scored on the Seoul set (#217, arm 1).

Pipeline: Mask2Former Vistas label map -> per image row, the walkable span that contains
the walking line -> back-project both span ends to the ground plane at the known camera
height -> width across the path, yaw-corrected from the fitted edge lines.

Ground truth is the Seoul Sidewalk Accessibility Image Dataset (Zenodo 22699523; fetch
with ``seoul_fetch_217.py``): 514 iPhone photos taken 1.0 m above the ground from the
centre of the walking path, with laser-measured effective (unobstructed) width.

Three stages::

    # 1. GPU: label maps (makelab2 A40, ~10 min). Not committed; sha256 manifest is.
    python scripts/analysis/sidewalk_width_217.py segment \
        --images /homes/gws/jonf/seoul_sidewalk/images --out SEGDIR

    # 2. CPU: every configuration's width for every image -> one committed CSV
    python scripts/analysis/sidewalk_width_217.py measure --seg SEGDIR \
        --out analysis_out/sidewalk_width_217/widths.csv.gz

    # 3. CPU, from the committed CSV alone: tune on half A, report on half B
    python scripts/analysis/sidewalk_width_217.py score \
        --widths analysis_out/sidewalk_width_217/widths.csv.gz \
        --out analysis_out/sidewalk_width_217/results.json

Geometry (``backproject``): pinhole camera, principal point at the image centre, focal
length from EXIF ``FocalLengthIn35mmFilm`` (diagonal convention, 43.27 mm), camera
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
DIAG_35MM = math.hypot(36.0, 24.0)
EXPECTED_LABELS = {2: "Curb", 7: "Bike Lane", 9: "Curb Cut", 11: "Pedestrian Area",
                   15: "Sidewalk", 19: "Person", 30: "Vegetation", 45: "Pole"}

#: Vistas v1.2 class groups. Every class not named here is a BOUNDARY: a span stops at it.
WALK_BASE = (15, 11, 9, 41, 36, 43)       # sidewalk, pedestrian area, curb cut, manhole,
#                                           catch basin, pothole (surface features on it)
WALK_SETS = {"base": WALK_BASE, "bike": WALK_BASE + (7,)}          # + Bike Lane
#: Not fixed: the GT excludes only permanent obstacles, so people and bicycles are passable
TRANSIENT = (0, 1, 19, 20, 21, 22, 52, 57, 62)
FURNITURE = (5, 32, 33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51)
#: fixed obstacles: street furniture (+ barrier/bollard), optionally vegetation + terrain
#: (tree pits, planters); where they are not obstacles they are boundaries
OBSTACLE_SETS = {"furn": FURNITURE, "furn_veg": FURNITURE + (29, 30)}

BORDER_PX = 3               # a span end this close to the frame is truncated, row dropped
Z_MAX_M = 15.0              # rows farther than this are never used
PITCH_SENS_DEG = (-2.0, 2.0)
VP_MAX_PITCH_DEG = 10.0
YAW_MAX_DEG = 30.0

#: the tuning grid (stage 3 picks one cell per measure on half A)
BAND_ZMIN = (1.5, 2.5, 4.0)  # band starts at the first valid row at or beyond this depth
BAND_LEN = (1.0, 3.0)        # ... and is this long (m)
STATS = ("median", "p10")    # near-band width, or the 10th percentile ("min over the band")
HORIZONS = ("level", "vp")

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
    paths = {}
    for dirpath, _, names in os.walk(args.images):
        for n in names:
            if n in gt and not n.startswith("._"):
                paths[n] = os.path.join(dirpath, n)
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


def measure_image(lab, f, h=CAMERA_H_M):
    """Every configuration's widths for one label map. Yields dict rows (no image id)."""
    Hh, Ww = lab.shape
    cx, cy = (Ww - 1) / 2.0, (Hh - 1) / 2.0
    for wname, walk in WALK_SETS.items():
        for oname, obst in OBSTACLE_SETS.items():
            spans = {"total": row_spans(lab, cx, walk, obst + TRANSIENT),
                     "clear": row_spans(lab, cx, walk, TRANSIENT)}
            spans = {k: exclude_border(l, r, Ww) for k, (l, r) in spans.items()}
            # horizon: level, or the edges' vanishing point (total spans: the physical edges)
            lt, rt = spans["total"]
            # rows within Z_MAX_M of a level camera: nearer the horizon the edges are noise
            below = np.arange(int(math.ceil(cy + f * h / Z_MAX_M)), Hh)
            rows = below[(lt[below] >= 0)]
            p_vp = vp_pitch(lt, rt, rows, f, cy) if rows.size else None
            for hname in HORIZONS:
                base = 0.0 if hname == "level" else p_vp
                if base is None:
                    base, fell_back = 0.0, True
                else:
                    fell_back = False
                for dp in (0.0,) + PITCH_SENS_DEG:
                    pitch = base + math.radians(dp)
                    for mname, (l, r) in spans.items():
                        Z, Wd, yaw = ground_widths(l, r, f, cx, cy, h, pitch)
                        for zmin in BAND_ZMIN:
                            for ln in BAND_LEN:
                                for st in STATS:
                                    yield {"walk": wname, "obst": oname, "horizon": hname,
                                           "vp_fallback": int(fell_back),
                                           "pitch_deg": round(math.degrees(pitch), 3),
                                           "dpitch": dp, "measure": mname, "zmin": zmin,
                                           "band": ln, "stat": st,
                                           "yaw_deg": round(math.degrees(yaw), 2),
                                           "width": band_stat(Z, Wd, zmin, ln, st)}


FIELDS = ["filename", "walk", "obst", "horizon", "vp_fallback", "pitch_deg", "dpitch",
          "measure", "zmin", "band", "stat", "yaw_deg", "width"]


def measure(args):
    from PIL import Image
    t0 = time.time()
    with open(os.path.join(args.seg, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)["images"]
    gt = _read_gt()
    out_rows = []
    for n in sorted(gt):
        m = meta.get(n)
        if m is None:
            continue
        lab = np.array(Image.open(os.path.join(args.seg, os.path.splitext(n)[0] + ".png")))
        if m.get("f35_mm"):
            f = focal_px(m["f35_mm"], m["work_w"], m["work_h"])
        else:
            sys.exit(f"{n}: no FocalLengthIn35mmFilm in EXIF")
        for row in measure_image(lab, f):
            row["filename"] = n
            w = row["width"]
            row["width"] = "" if not np.isfinite(w) else f"{w:.3f}"
            out_rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    buf = io.StringIO(newline="")
    wr = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(out_rows)
    # mtime=0: the gzip bytes depend only on the content
    with open(args.out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                    filename="") as g:
        g.write(buf.getvalue().encode("utf-8"))
    print(f"{len(out_rows)} rows for {len({r['filename'] for r in out_rows})} images "
          f"in {time.time() - t0:.0f} s -> {args.out}")


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
                   within_0_5=float((np.abs(err) <= 0.5).mean()))
    narrow_gt = gt < THRESH_M[0]
    flagged = ok & (np.nan_to_num(est, nan=np.inf) < THRESH_M[0])
    tp = int((flagged & narrow_gt).sum())
    out.update(n_narrow_gt=int(narrow_gt.sum()), n_flagged=int(flagged.sum()), tp=tp,
               recall_lt_1_2=tp / narrow_gt.sum() if narrow_gt.sum() else np.nan,
               precision_lt_1_2=tp / flagged.sum() if flagged.sum() else np.nan)
    if ok.sum():
        out["acc3"] = float((cls3(e) == cls3(g)).mean())
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


def _load_widths(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(args):
    gt = _read_gt()
    half, groups = split_groups(gt)
    rows = _load_widths(args.widths)
    cfg_keys = ("walk", "obst", "horizon", "zmin", "band", "stat")
    table = {}
    for r in rows:
        k = (r["measure"], float(r["dpitch"])) + tuple(r[c] for c in cfg_keys)
        table.setdefault(k, {})[r["filename"]] = (float(r["width"]) if r["width"] else np.nan)
    names = sorted(gt)
    G = np.array([float(gt[n]["width"]) for n in names])
    H = np.array([half[n] for n in names])
    GR = np.array([groups[n] for n in names])
    A, B = H == "A", H == "B"

    def arr(k):
        d = table[k]
        return np.array([d.get(n, np.nan) for n in names])

    boot_keys = ["mae", "bias_mean", "bias_median", "recall_lt_1_2", "precision_lt_1_2",
                 "acc3", "coverage"]
    res = {"split": {"seed": SPLIT_SEED, "group_cell_deg": GROUP_CELL_DEG,
                     "n_A": int(A.sum()), "n_B": int(B.sum()),
                     "groups_A": int(len(set(GR[A]))), "groups_B": int(len(set(GR[B]))),
                     "narrow_A": int((G[A] < 1.2).sum()), "narrow_B": int((G[B] < 1.2).sum())},
           "gt": {"n": len(names), "mean": float(G.mean()), "min": float(G.min()),
                  "max": float(G.max()), "n_lt_1_2": int((G < 1.2).sum()),
                  "n_lt_1_5": int((G < 1.5).sum())},
           "measures": {}}
    for meas in ("clear", "total"):
        cands = [k for k in table if k[0] == meas and k[1] == 0.0]
        # tuning rule: lowest MAE on half A among configs covering >= 90% of half A
        scored = []
        for k in cands:
            e = arr(k)
            mA = metrics(e[A], G[A])
            if mA["coverage"] >= 0.9:
                scored.append((mA["mae"], k, mA))
        scored.sort(key=lambda t: t[0])
        best_mae, best, mA = scored[0]
        cfg = dict(zip(cfg_keys, best[2:]))
        e = arr(best)
        mB = metrics(e[B], G[B])
        ciB = bootstrap(e[B], G[B], GR[B], boot_keys, n_boot=args.n_boot)
        sens = {}
        for dp in PITCH_SENS_DEG:
            es = arr((meas, dp) + best[2:])
            ok = np.isfinite(es[B]) & np.isfinite(e[B])
            sens[f"{dp:+.0f}deg"] = {
                "median_rel_change": float(np.median(es[B][ok] / e[B][ok] - 1)),
                "mae_B": metrics(es[B], G[B]).get("mae"),
                "bias_mean_B": metrics(es[B], G[B]).get("bias_mean")}
        # bias-corrected (calibration fit on A only: median ratio), reported beside raw
        okA = np.isfinite(e[A])
        ratio = float(np.median(G[A][okA] / e[A][okA]))
        mBcal = metrics(e[B] * ratio, G[B])
        ciBcal = bootstrap(e[B] * ratio, G[B], GR[B], boot_keys, n_boot=args.n_boot)
        # the same config on the whole set, for context only (it was tuned on A)
        res["measures"][meas] = {
            "config": cfg, "tuning_mae_A": best_mae, "metrics_A": mA,
            "n_configs_considered": len(scored),
            "top5_A": [{"mae_A": s[0], "config": dict(zip(cfg_keys, s[1][2:]))}
                       for s in scored[:5]],
            "metrics_B": mB, "ci95_B": ciB, "confusion_B": confusion(e[B], G[B]),
            "pitch_sensitivity_B": sens,
            "calibrated_B": {"scale_from_A": ratio, "metrics": mBcal, "ci95": ciBcal,
                             "confusion": confusion(e[B] * ratio, G[B])},
            "per_image": {n: (None if not np.isfinite(x) else round(float(x), 3))
                          for n, x in zip(names, e)}}
        # the VP variant's fallback rate, for the chosen walk/obst
        fb = [int(r["vp_fallback"]) for r in rows
              if r["measure"] == meas and r["horizon"] == "vp" and float(r["dpitch"]) == 0
              and r["walk"] == cfg["walk"] and r["obst"] == cfg["obst"]
              and r["zmin"] == cfg["zmin"] and r["band"] == cfg["band"]
              and r["stat"] == cfg["stat"]]
        res["measures"][meas]["vp_fallback_rate"] = float(np.mean(fb)) if fb else None
    res["confusion_layout"] = ("rows: GT <1.2, 1.2-1.5, >=1.5; cols: estimate <1.2, "
                               "1.2-1.5, >=1.5, no estimate")
    res["widths_sha256"] = hashlib.sha256(open(args.widths, "rb").read()).hexdigest()
    res["n_boot"] = args.n_boot

    def _clean(o):
        if isinstance(o, float):
            return None if not math.isfinite(o) else round(o, 4)
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return _clean(float(o))
        return o
    res = _clean(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    for meas, r in res["measures"].items():
        print(meas, r["config"], "B:", {k: r["metrics_B"].get(k) for k in boot_keys},
              "CI:", r["ci95_B"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("segment")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("measure")
    p.add_argument("--seg", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--widths", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    {"segment": segment, "measure": measure, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
