"""Frozen-model nuisance probe for issue #82 (Step 1: no training).

Question: **which pixel-statistics axis, if any, does the released RampNet checkpoint
react to?** The model loses 0.112 F1 going from GSV to GoPro Max over the same Laurens
corners (docs/laurens_paired_151.md), almost all of it recall. If a single pixel-level
nuisance (resolution, blur, compression, exposure, colour) carries a share of that gap,
degrading GSV panos along that axis toward the GoPro value should cost recall, and
repairing GoPro panos toward GSV should recover some. An axis the frozen model does not
react to is not worth an augmentation arm.

Subcommands::

    # CPU (makelab2; needs benchmark panos). Per-pano rig statistics at the model's
    # 2048x4096 input for every split, plus the calibration curves that place the
    # sharpness levels -> analysis_out/aug_transfer_82/stats.json
    python scripts/analysis/aug_probe_82.py stats --panos-root /homes/gws/jonf/RampNet --workers 6

    # CPU, no panos. The arm table derived from stats.json (printed; also in report)
    python scripts/analysis/aug_probe_82.py arms

    # GPU (makelab2 A40). One cache per arm x split under
    # analysis_out/aug_transfer_82/probe/<arm>/<split>.json (preds only; GT is rebuilt
    # from the committed bundle at report time). Skip-if-exists per arm x split.
    python scripts/analysis/aug_probe_82.py extract --panos-root /homes/gws/jonf/RampNet

    # CPU, no panos. The instrument check: the untransformed arm must reproduce the
    # committed r2048 caches of #25 (same checkpoint, same panos), exit 1 otherwise.
    python scripts/analysis/aug_probe_82.py check

    # CPU, no panos. Per split x arm metrics at 0.30 / 0.55 and the paired pano
    # bootstrap against the untransformed run -> probe_results.json + probe_results.md
    python scripts/analysis/aug_probe_82.py report [--check]

**Where each transform is applied.** The native pano is decoded, resized to 2048x4096 by
``torchvision.transforms.Resize`` bilinear (the first step of the scorer's own
``threshold_sweep.PRE``), transformed by ``rampnet.augment`` as a PIL image, and then
passed through ``PRE`` unchanged (its resize is a no-op at that size; PIL returns a copy
when the target size equals the current one). The untransformed arm is exactly that path
with no transform in the middle, and ``check`` proves it equals the committed instrument.
The brief asked for the transform *before* the resize; it is applied right after it
instead, because native widths run from 5,760 (GoPro Max) to 16,384 (GSV), so a native-
resolution level (a blur sigma, a JPEG quality) would mean a different thing on every
split, and because training panos are stored at 2048x4096, so training augmentation can
only happen at that size. Levels here are therefore in the same units the training flags
use. See docs/aug_transfer_82.md, "Deviations".
"""
import argparse
import hashlib
import json
import math
import os
import socket
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from queue import Queue

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import augment as A  # noqa: E402
from rampnet import ledger  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402
from operating_point_curve import _score_at, bundle_ground_truths, pr_curve_and_ap  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "aug_transfer_82")
STATS = os.path.join(OUT_DIR, "stats.json")
PROBE_ROOT = os.path.join(OUT_DIR, "probe")
RESULTS = os.path.join(OUT_DIR, "probe_results.json")
RESULTS_MD = os.path.join(OUT_DIR, "probe_results.md")
REF_CACHE = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "cache", "r2048")
USAGE_LOG = os.path.join(REPO, "analysis_out", "usage_log.jsonl")

BASE_SIZE = (2048, 4096)
SCORE_FLOOR = 0.05
MIN_DISTANCE = 10
THRESHOLDS = (0.30, 0.55)
N_REPS = 2000
SEED = 82
ND = 4
COORD_ND = 6
CHECK_TOL = 2e-4      # same cross-machine tolerance as input_res_sweep_25.CHECK_TOL

#: Rig groups. Degrade the GSV splits toward GoPro; repair the GoPro splits toward GSV.
GSV_SPLITS = ("laurens_gsv", "bend", "gainesville", "paterson")
GOPRO_SPLITS = ("laurens_mapillary", "clovis", "richmond")
# Extraction order = priority order: the paired Laurens footprint first, so a run cut short
# still answers the main question.
PROBE_SPLITS = ("laurens_gsv", "laurens_mapillary", "clovis", "bend", "richmond", "gainesville",
                "paterson")
assert set(PROBE_SPLITS) == set(GSV_SPLITS + GOPRO_SPLITS)
#: Every bundle the stats are measured on (manual_gold is subsampled: 1,000 panos).
STATS_SPLITS = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville",
                "laurens_gsv", "laurens_mapillary", "manual_gold", "morgantown",
                "paterson", "richmond", "sao_paulo")
STATS_MAX_PANOS = 125
#: The paired footprint that places every level: GSV side and GoPro side.
REF_GSV, REF_GOPRO = "laurens_gsv", "laurens_mapillary"
CALIB_N = 24          # panos per side used for the sharpness calibration curves
CONTROL = "none"

# ground band used for the sharpness statistics: just below the horizon, above the
# capture vehicle -- where curb ramps are (rows 1024..1791 of 2048).
BAND = (1024, 1792)

#: Calibration grids (level -> statistic) for the axes whose GoPro level is not a
#: direct ratio of a measured statistic.
CALIB_GRID = {
    "downscale": (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3, 0.25, 0.2),
    "blur": (0.0, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    "unsharp": (0.0, 50.0, 100.0, 150.0, 200.0, 300.0),
}


# --------------------------------------------------------------------------- #
# statistics (pure numpy on a PIL image at model input size)
# --------------------------------------------------------------------------- #
def jpeg_quality_estimate(qtable):
    """IJG quality that would produce this luminance quantization table (64 values in
    natural or zigzag order: the ratio of sums is order-free). None if no table."""
    if not qtable:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
                   dtype=np.float64)
    q = np.asarray(list(qtable), dtype=np.float64)
    if q.size != 64:
        return None
    s = float(q.sum() / std.sum() * 100.0)
    if s <= 0:
        return 100.0
    return float(min(100.0, (200.0 - s) / 2.0 if s <= 100.0 else 5000.0 / s))


def _gray(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def sharpness(gray_band):
    """(Laplacian variance, high-frequency power fraction) of a 2-D grey band.

    The fraction is spectral power at radial frequency >= 0.25 cycles/px over power at
    >= 0.02 cycles/px (DC and the slow illumination gradient excluded): a resolution
    proxy that does not depend on scene contrast the way the Laplacian variance does."""
    g = gray_band
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    lapvar = float(lap.var())
    f = np.fft.rfft2(g - g.mean())
    p = (f.real ** 2 + f.imag ** 2)
    fy = np.fft.fftfreq(g.shape[0])[:, None]
    fx = np.fft.rfftfreq(g.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    lo = p[r >= 0.02].sum()
    hf = float(p[r >= 0.25].sum() / lo) if lo > 0 else 0.0
    return lapvar, hf


def noise_sigma(gray_band):
    """Robust noise estimate (Immerkaer 1996: a Laplacian-difference kernel that
    cancels locally planar image structure; median-free form) in 8-bit units."""
    g = gray_band
    k = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
         - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
         + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    h, w = g.shape
    return float(math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.abs(k).sum())


def image_stats(img):
    """Per-pano statistics at model input size. ``img`` is RGB PIL, 2048x4096."""
    a = np.asarray(img, dtype=np.float64)
    gray = _gray(img)
    band = gray[BAND[0]:BAND[1]]
    lapvar, hf = sharpness(band)
    mean_rgb = a.reshape(-1, 3).mean(axis=0)
    std_rgb = a.reshape(-1, 3).std(axis=0)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    return {
        "lum_mean": float(gray.mean()), "lum_std": float(gray.std()),
        "band_lum_mean": float(band.mean()), "band_lum_std": float(band.std()),
        "sat_mean": float(hsv[..., 1].mean()),
        "r_mean": float(mean_rgb[0]), "g_mean": float(mean_rgb[1]),
        "b_mean": float(mean_rgb[2]),
        "r_std": float(std_rgb[0]), "g_std": float(std_rgb[1]), "b_std": float(std_rgb[2]),
        "log2_r_over_b": float(math.log2(max(mean_rgb[0], 1e-6) / max(mean_rgb[2], 1e-6))),
        "lap_var": lapvar, "hf_frac": hf, "noise_sigma": noise_sigma(band),
    }


def sharpness_only(img):
    band = _gray(img)[BAND[0]:BAND[1]]
    lapvar, hf = sharpness(band)
    return {"lap_var": lapvar, "hf_frac": hf}


def to_model_input(native):
    """Native PIL -> RGB PIL at 2048x4096 via the scorer's own Resize (bilinear)."""
    from torchvision import transforms
    return transforms.Resize(BASE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR)(
        native)


# --------------------------------------------------------------------------- #
# stats subcommand
# --------------------------------------------------------------------------- #
def _stats_one(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    q = None
    try:
        q = jpeg_quality_estimate(im.quantization.get(0)) if getattr(
            im, "quantization", None) else None
    except Exception:  # noqa: BLE001
        q = None
    native = im.size
    img = to_model_input(im.convert("RGB"))
    st = image_stats(img)
    st.update({"native_w": native[0], "native_h": native[1], "jpeg_q_native": q})
    return st


def _calib_one(job):
    """(axis, split, pid, path) -> {level: sharpness} over CALIB_GRID[axis]."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    axis, _split, _pid, path = job
    img = to_model_input(Image.open(path).convert("RGB"))
    return {str(lv): sharpness_only(A.apply_op(img, axis, lv)) for lv in CALIB_GRID[axis]}


def _stats_pids(split, gts):
    pids = sorted(gts)
    if len(pids) > STATS_MAX_PANOS:   # manual_gold: a fixed, seeded subsample
        rng = np.random.default_rng(SEED)
        pids = sorted(rng.choice(pids, STATS_MAX_PANOS, replace=False).tolist())
    return pids


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) or rows[0][k] is None]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] is not None], dtype=np.float64)
        if v.size == 0:
            out[k] = None
            continue
        out[k] = {"median": rnd(np.median(v), 6), "mean": rnd(v.mean(), 6),
                  "p25": rnd(np.percentile(v, 25), 6), "p75": rnd(np.percentile(v, 75), 6)}
    return out


def cmd_stats(args):
    from multiprocessing import Pool
    t0 = time.perf_counter()
    started = _now()
    per_split, jobs = {}, []
    for city in args.cities:
        gts, _ = bundle_ground_truths(city)
        pdir = os.path.join(args.panos_root, "benchmark", city, "panos")
        for pid in _stats_pids(city, gts):
            jobs.append((city, pid, os.path.join(pdir, f"{pid}.jpg")))
    print(f"stats: {len(jobs)} panos over {len(args.cities)} splits, {args.workers} workers",
          flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(_stats_one, [j[2] for j in jobs], chunksize=4)
    for (city, pid, _), st in zip(jobs, res):
        per_split.setdefault(city, {})[pid] = st
    calib_jobs = []
    for axis, split in (("downscale", REF_GSV), ("blur", REF_GSV), ("unsharp", REF_GOPRO)):
        gts, _ = bundle_ground_truths(split)
        pids = sorted(gts)[:CALIB_N]
        pdir = os.path.join(args.panos_root, "benchmark", split, "panos")
        calib_jobs += [(axis, split, pid, os.path.join(pdir, f"{pid}.jpg")) for pid in pids]
    with Pool(args.workers) as pool:
        cres = pool.map(_calib_one, calib_jobs, chunksize=2)
    calib = {}
    for (axis, split, pid, _), r in zip(calib_jobs, cres):
        calib.setdefault(axis, {"split": split, "panos": {}})["panos"][pid] = r
    out = {"protocol": {"input_size": list(BASE_SIZE), "band_rows": list(BAND),
                        "max_panos_per_split": STATS_MAX_PANOS, "subsample_seed": SEED,
                        "calib_n": CALIB_N, "calib_grid": {k: list(v) for k, v in
                                                           CALIB_GRID.items()},
                        "ref_gsv": REF_GSV, "ref_gopro": REF_GOPRO},
           "summary": {c: summarize(list(per_split[c].values())) for c in per_split},
           "per_pano": {c: {p: {k: rnd(v, 6) for k, v in d.items()}
                            for p, d in sorted(per_split[c].items())} for c in per_split},
           "calibration": {ax: {"split": d["split"], "panos": {
               p: {lv: {k: rnd(v, 6) for k, v in s.items()} for lv, s in r.items()}
               for p, r in sorted(d["panos"].items())}} for ax, d in calib.items()}}
    write_json(args.out, out)
    wall = time.perf_counter() - t0
    print(f"stats -> {args.out}  ({wall:.0f} s)", flush=True)
    if args.usage_log.lower() != "none":
        ledger.append_rows(args.usage_log, [{
            "ts": started, "bundle": ",".join(args.cities), "label": "aug-probe-82:stats",
            "provider": "rampnet", "model_id": None, "paid": False,
            "hardware": {"host": socket.getfqdn(), "gpus": [], "cpu_workers": args.workers},
            "status": "ok", "est_cost_usd": 0.0, "pricing": None, "panos_scored": len(jobs),
            "elapsed_s": round(wall, 3), "s_per_pano": round(wall / max(1, len(jobs)), 4),
            "what": "aug_probe_82.py stats: pixel statistics + sharpness calibration (CPU)",
            "run_id": f"aug-probe-82:stats:{started}", "script": "scripts/analysis/aug_probe_82.py",
            "issue": 82}])


# --------------------------------------------------------------------------- #
# arm table, derived from stats.json
# --------------------------------------------------------------------------- #
def _med(stats, split, key):
    return stats["summary"][split][key]["median"]


def _calib_curve(stats, axis, key="hf_frac"):
    """[(level, median over calibration panos of the statistic)] in grid order."""
    d = stats["calibration"][axis]["panos"]
    levels = [float(x) for x in stats["protocol"]["calib_grid"][axis]]
    return [(lv, float(np.median([d[p][repr_level(lv)][key] for p in d]))) for lv in levels]


def repr_level(lv):
    return str(float(lv)) if not isinstance(lv, str) else lv


def invert_curve(curve, target):
    """Level at which a monotone (level, stat) curve reaches ``target``, by linear
    interpolation; clamps to the grid ends. Returns (level, clamped?)."""
    lv = np.array([c[0] for c in curve])
    st = np.array([c[1] for c in curve])
    order = np.argsort(st)
    st_s, lv_s = st[order], lv[order]
    if target <= st_s[0]:
        return float(lv_s[0]), True
    if target >= st_s[-1]:
        return float(lv_s[-1]), True
    return float(np.interp(target, st_s, lv_s)), False


def _geo3(mid, neutral=1.0):
    """Three multiplicative levels: halfway (geometric), the measured value, and the
    same step again beyond it."""
    r = mid / neutral
    return (neutral * r ** 0.5, mid, neutral * r ** 1.5)


def _lin3(mid, neutral=0.0):
    d = mid - neutral
    return (neutral + d / 2.0, mid, neutral + 1.5 * d)


def derive_levels(stats):
    """{axis: {"levels": (lo, mid, hi), "measured": ..., "note": ...}} for the GSV
    degradation axes, every mid at the measured REF_GOPRO value relative to REF_GSV.

    Ratio axes (brightness, contrast, saturation) use the ratio of the paired splits'
    medians; ``wb`` the difference in log2(R/B); ``gamma`` the exponent that maps the
    GSV median luminance onto the GoPro one; ``noise`` the quadrature difference of the
    noise estimates; ``jpeg`` the estimated native quality of the GoPro files;
    ``downscale`` / ``blur`` invert the calibration curves of the high-frequency power
    fraction on REF_GSV panos at REF_GOPRO's median. Where the measured difference is
    negligible (or points the other way for a degradation-only axis) the level is a
    fixed fallback and the note says so."""
    g, m = REF_GSV, REF_GOPRO
    out = {}
    # Sharpness. The paired GoPro split is NOT softer than GSV at 2048x4096 (its 5,760 px
    # native is downsampled too), so laurens_mapillary cannot place these levels. The
    # target is the softest GoPro split by median Laplacian variance (clovis, the 2018
    # GoPro Fusion split #82 was filed about), and the three levels sit at the
    # log-midpoint, at the target, and as far again beyond it, each inverted on the
    # REF_GSV calibration curve of the same statistic (monotone, unlike hf_frac).
    lv_g = _med(stats, g, "lap_var")
    soft = min(GOPRO_SPLITS, key=lambda c: _med(stats, c, "lap_var"))
    lv_t = _med(stats, soft, "lap_var")
    targets = (math.sqrt(lv_g * lv_t), lv_t, lv_t * lv_t / lv_g)
    for axis in ("downscale", "blur"):
        curve = [(lv, math.log(v)) for lv, v in _calib_curve(stats, axis, "lap_var")]
        inv = [invert_curve(curve, math.log(t)) for t in targets]
        out[axis] = {"levels": tuple(rnd(x[0], 3) for x in inv), "measured": rnd(inv[1][0], 4),
                     "note": (f"lap_var {g} {lv_g:.0f} -> {soft} {lv_t:.0f} (softest GoPro "
                              f"split; {m} is {_med(stats, m, 'lap_var'):.0f}, not softer); "
                              f"targets {', '.join(f'{t:.0f}' for t in targets)}; calibrated "
                              f"on {CALIB_N} {g} panos"
                              + ("; a level CLAMPED to the grid end" if any(x[1] for x in inv)
                                 else ""))}
    for axis, key in (("brightness", "lum_mean"), ("contrast", "lum_std"),
                      ("saturation", "sat_mean")):
        r = _med(stats, m, key) / _med(stats, g, key)
        if abs(math.log(r)) < 0.02:
            lv, note = (0.9, 0.8, 0.7), f"ratio {r:.3f} is negligible; fixed fallback levels"
        else:
            lv, note = _geo3(r), f"median {key} ratio GoPro/GSV = {r:.3f}"
        out[axis] = {"levels": tuple(rnd(x, 3) for x in lv), "measured": rnd(r, 4),
                     "note": note}
    lg, lm = _med(stats, g, "lum_mean") / 255.0, _med(stats, m, "lum_mean") / 255.0
    gam = math.log(lm) / math.log(lg)
    out["gamma"] = {"levels": tuple(rnd(x, 3) for x in _geo3(gam)), "measured": rnd(gam, 4),
                    "note": "exponent mapping median GSV luminance onto the GoPro one"}
    d = _med(stats, m, "log2_r_over_b") - _med(stats, g, "log2_r_over_b")
    lv = _lin3(d) if abs(d) >= 0.02 else (0.1, 0.2, 0.3)
    out["wb"] = {"levels": tuple(rnd(x, 3) for x in lv), "measured": rnd(d, 4),
                 "note": f"log2(R/B) GoPro - GSV = {d:+.3f}"
                 + ("" if abs(d) >= 0.02 else "; negligible, fixed fallback")}
    ng, nm = _med(stats, g, "noise_sigma"), _med(stats, m, "noise_sigma")
    add = math.sqrt(max(nm ** 2 - ng ** 2, 0.0))
    lv = _lin3(add) if add >= 0.5 else (2.0, 4.0, 8.0)
    out["noise"] = {"levels": tuple(rnd(x, 3) for x in lv), "measured": rnd(add, 4),
                    "note": f"noise sigma GSV {ng:.2f}, GoPro {nm:.2f}; added sigma = "
                            f"sqrt(diff of squares)" + ("" if add >= 0.5 else
                                                        "; negligible, fixed fallback")}
    qm = _med(stats, m, "jpeg_q_native")
    mid = qm if qm is not None and qm < 90 else 75.0
    out["jpeg"] = {"levels": (rnd(min(95.0, mid + 15), 1), rnd(mid, 1), rnd(max(10.0, mid - 25), 1)),
                   "measured": rnd(qm, 2) if qm is not None else None,
                   "note": f"estimated native JPEG quality of {m} files = {qm}"
                   + ("" if qm is not None and qm < 90 else
                      "; >= 90 (near-lossless at native size), fixed fallback 75")}
    return out


def derive_repairs(stats):
    """GoPro -> GSV repair arms: colour-statistics match to REF_GSV's median channel
    mean / std (alpha 0.5, 1.0), unsharp mask at the level whose hf_frac on REF_GOPRO
    panos reaches REF_GSV's median (and half of it), CLAHE at two clip limits."""
    g = REF_GSV
    ref_mean = [_med(stats, g, k) for k in ("r_mean", "g_mean", "b_mean")]
    ref_std = [_med(stats, g, k) for k in ("r_std", "g_std", "b_std")]
    # Unsharp: fixed levels. The calibration (on REF_GOPRO) cannot place them -- that
    # split is already at or above GSV sharpness -- and the soft GoPro splits are the
    # ones a sharpening repair is for, so two conventional strengths are probed.
    return {"colour_match": {"levels": (0.5, 1.0), "ref_mean": [rnd(x, 3) for x in ref_mean],
                             "ref_std": [rnd(x, 3) for x in ref_std]},
            "unsharp": {"levels": (50.0, 100.0),
                        "note": "fixed (radius 2 px): the paired GoPro split is not softer "
                                "than GSV, so no measured level"},
            "clahe": {"levels": (0.01, 0.02)}}


LEVEL_NAMES = ("half", "gopro", "beyond")


def build_arms(stats):
    """{arm: {"splits": tuple, "ops": [(op, level), ...], "axis", "level_name"}}."""
    arms = {CONTROL: {"splits": PROBE_SPLITS, "ops": [], "axis": None, "level_name": None}}
    lv = derive_levels(stats)
    for axis, d in lv.items():
        for name, level in zip(LEVEL_NAMES, d["levels"]):
            arms[f"{axis}@{name}"] = {"splits": GSV_SPLITS, "ops": [(axis, float(level))],
                                      "axis": axis, "level_name": name}
    # Every degradation at its GoPro-measured level at once, in the training order.
    combo = [(ax, float(lv[ax]["levels"][1])) for ax in A.TRAIN_ORDER if ax in lv]
    arms["all@gopro"] = {"splits": GSV_SPLITS, "ops": combo, "axis": "all",
                         "level_name": "gopro"}
    rp = derive_repairs(stats)
    for a in rp["colour_match"]["levels"]:
        arms[f"colour_match@{a:g}"] = {"splits": GOPRO_SPLITS,
                                       "ops": [("colour_match", float(a))],
                                       "axis": "colour_match", "level_name": f"{a:g}"}
    for p in rp["unsharp"]["levels"]:
        arms[f"unsharp@{p:g}"] = {"splits": GOPRO_SPLITS, "ops": [("unsharp", float(p))],
                                  "axis": "unsharp", "level_name": f"{p:g}"}
    for c in rp["clahe"]["levels"]:
        arms[f"clahe@{c:g}"] = {"splits": GOPRO_SPLITS, "ops": [("clahe", float(c))],
                                "axis": "clahe", "level_name": f"{c:g}"}
    arms["repair_all"] = {"splits": GOPRO_SPLITS,
                          "ops": [("colour_match", 1.0),
                                  ("unsharp", float(rp["unsharp"]["levels"][1]))],
                          "axis": "repair_all", "level_name": "gsv"}
    return arms, lv, rp


def apply_arm(img, ops, rp, pid):
    """Apply an arm's ops to a 2048x4096 RGB PIL image. ``noise`` draws from a generator
    keyed on the pano id, so a re-run draws the same noise."""
    for op, level in ops:
        if op == "colour_match":
            img = A.colour_match(img, rp["colour_match"]["ref_mean"],
                                 rp["colour_match"]["ref_std"], alpha=level)
        elif op == "noise":
            img = A.noise(img, level, A.sample_rng(SEED, 0, zlib.crc32(pid.encode())))
        else:
            img = A.apply_op(img, op, level)
    return img


# --------------------------------------------------------------------------- #
# extract (GPU)
# --------------------------------------------------------------------------- #
def cache_path(root, arm, city):
    return os.path.join(root, arm.replace("@", "_at_"), f"{city}.json")


def write_cache(path, city, arm, preds_by_pid, meta):
    payload = {"city": city, "arm": arm, "meta": meta,
               "panos": [{"pano": pid, "preds": [[round(x, COORD_ND), round(y, COORD_ND),
                                                  round(s, COORD_ND)] for (x, y, s) in pr]}
                         for pid, pr in preds_by_pid]}
    write_json(path, payload)


def read_cache(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {p["pano"]: [tuple(t) for t in p["preds"]] for p in payload["panos"]}, payload


def cmd_extract(args):
    import torch
    from PIL import Image
    import threshold_sweep as ts
    Image.MAX_IMAGE_PIXELS = None
    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)
    arms, _, rp = build_arms(stats)
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in want if a not in arms]
        if bad:
            raise SystemExit(f"unknown arms {bad}; known: {', '.join(arms)}")
        arms = {a: arms[a] for a in want}
    if args.limit and os.path.abspath(args.cache_root) == os.path.abspath(PROBE_ROOT):
        raise SystemExit("--limit writes truncated caches: pass a scratch --cache-root")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda" else [])
    started = _now()
    t_start = time.perf_counter()
    gpu_s, wait_s, cpu_s, n_fwd = 0.0, 0.0, 0.0, 0
    attempted = []
    status = "failed"
    print(f"device={device} gpus={gpus} arms={len(arms)} torch={torch.__version__}",
          flush=True)
    try:
        for city in args.cities:
            todo = [a for a, d in arms.items() if city in d["splits"]
                    and (args.force or not os.path.exists(cache_path(args.cache_root, a, city)))]
            if not todo:
                print(f"{city}: nothing to do", flush=True)
                continue
            attempted.append(city)
            gts, _ = bundle_ground_truths(city)
            pids = sorted(gts)[:args.limit] if args.limit else sorted(gts)
            pdir = os.path.join(args.panos_root, "benchmark", city, "panos")
            q = Queue(maxsize=2)

            def work(pids=pids, pdir=pdir, todo=todo):
                pool = ThreadPoolExecutor(args.prep_threads)
                for pid in pids:
                    t0 = time.perf_counter()
                    try:
                        base = to_model_input(Image.open(
                            os.path.join(pdir, f"{pid}.jpg")).convert("RGB"))

                        def one(a, base=base, pid=pid):
                            return ts.PRE(apply_arm(base, arms[a]["ops"], rp, pid))
                        tensors = dict(zip(todo, pool.map(one, todo)))
                        q.put((pid, tensors, time.perf_counter() - t0, None))
                    except Exception as e:  # noqa: BLE001
                        q.put((pid, None, 0.0, e))
                q.put(None)
                pool.shutdown()
            threading.Thread(target=work, daemon=True).start()
            res = {a: [] for a in todo}
            i = 0
            while True:
                tw = time.perf_counter()
                item = q.get()
                wait_s += time.perf_counter() - tw
                if item is None:
                    break
                pid, tensors, c, err = item
                if err is not None:
                    raise SystemExit(f"{city}/{pid}: {err}")
                cpu_s += c
                for a in todo:
                    t0 = time.perf_counter()
                    t = tensors[a].unsqueeze(0).to(device)
                    with torch.no_grad():
                        h = model(t).squeeze().float().cpu().numpy()
                    res[a].append((pid, ts.peaks_to_dets(h, SCORE_FLOOR, MIN_DISTANCE)))
                    gpu_s += time.perf_counter() - t0
                    n_fwd += 1
                    del t, h
                del tensors
                i += 1
                if i % 25 == 0:
                    print(f"  {city}: {i}/{len(pids)}  wall {time.perf_counter() - t_start:.0f}s",
                          flush=True)
            for a in todo:
                meta = {"arm": a, "ops": arms[a]["ops"], "score_floor": SCORE_FLOOR,
                        "min_distance": MIN_DISTANCE, "radius_normalized": 0.022,
                        "fp16": False, "tta": False, "n_panos": len(res[a]),
                        "model": "projectsidewalk/rampnet-model", "device": device.type,
                        "gpus": gpus, "torch": torch.__version__,
                        "where": "native -> Resize 2048x4096 bilinear -> transform -> PRE"}
                write_cache(cache_path(args.cache_root, a, city), city, a, res[a], meta)
            print(f"{city} done: {len(todo)} arms x {len(pids)} panos; gpu {gpu_s:.0f}s "
                  f"wait {wait_s:.0f}s wall {time.perf_counter() - t_start:.0f}s", flush=True)
        status = "ok"
    finally:
        wall = time.perf_counter() - t_start
        if args.usage_log.lower() != "none" and (n_fwd or status != "ok"):
            ledger.append_rows(args.usage_log, [{
                "ts": started, "bundle": ",".join(attempted or args.cities),
                "label": "aug-probe-82:extract", "provider": "rampnet",
                "model_id": "projectsidewalk/rampnet-model", "paid": False,
                "hardware": {"host": socket.getfqdn(), "gpus": gpus}, "status": status,
                "est_cost_usd": 0.0, "pricing": None, "panos_scored": n_fwd,
                "elapsed_s": round(wall, 3), "gpu_side_s": round(gpu_s, 3),
                "cpu_wait_s": round(wait_s, 3), "cpu_prep_s": round(cpu_s, 3),
                "s_per_pano": round(gpu_s / n_fwd, 4) if n_fwd else None,
                "what": ("aug_probe_82.py extract: panos_scored counts forward passes "
                         "(arm x pano); elapsed_s is the run's wall-clock on the A40"),
                "run_id": f"aug-probe-82:extract:{started}",
                "script": "scripts/analysis/aug_probe_82.py", "issue": 82,
                **({"note": args.note} if args.note else {})}])
            print(f"usage_log row ({status}) -> {args.usage_log}", flush=True)


# --------------------------------------------------------------------------- #
# check (CPU): the untransformed arm reproduces #25's committed r2048 caches
# --------------------------------------------------------------------------- #
def compare_preds(mine, ref, tol=CHECK_TOL):
    """Pano-by-pano peak comparison. Returns (n_mismatched, max |score diff|, details)."""
    bad, worst, details = 0, 0.0, []
    if set(mine) != set(ref):
        return len(ref), float("inf"), ["pano sets differ"]
    for pid in sorted(ref):
        a = {(round(x, 6), round(y, 6)): s for x, y, s in mine[pid]}
        b = {(round(x, 6), round(y, 6)): s for x, y, s in ref[pid]}
        d = max((abs(a[k] - b[k]) for k in a if k in b), default=0.0)
        worst = max(worst, d)
        if set(a) != set(b) or d > tol:
            bad += 1
            if len(details) < 10:
                details.append(f"{pid}: +{len(set(a) - set(b))} / -{len(set(b) - set(a))} "
                               f"peaks, max |score diff| {d:.2e}")
    return bad, worst, details


def cmd_check(args):
    rows, ok = [], True
    for city in PROBE_SPLITS:
        p = cache_path(args.cache_root, CONTROL, city)
        if not os.path.exists(p):
            print(f"{city}: no control cache -> FAIL")
            ok = False
            continue
        mine, _ = read_cache(p)
        with open(os.path.join(REF_CACHE, f"{city}.json"), encoding="utf-8") as f:
            ref = {q["pano"]: [tuple(t) for t in q["preds"]] for q in json.load(f)["panos"]}
        bad, worst, det = compare_preds(mine, ref)
        rows.append({"city": city, "n_panos": len(ref), "panos_mismatched": bad,
                     "max_abs_score_diff": rnd(worst, 8), "details": det})
        ok &= bad == 0
        print(f"{city}: {len(ref)} panos, mismatched {bad}, max |score diff| {worst:.2e}"
              + ("" if bad == 0 else "  <-- " + "; ".join(det[:3])))
    write_json(args.out, {"reference": os.path.relpath(REF_CACHE, REPO).replace(os.sep, "/"),
                          "score_tolerance": CHECK_TOL, "pass": ok, "cities": rows})
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# report (CPU)
# --------------------------------------------------------------------------- #
def _panos(preds, gts):
    return [{"pano": pid, "preds": preds[pid], "gt": gts[pid]} for pid in sorted(gts)]


def _scored(city, panos, rsq):
    records = {p["pano"]: {"detections": [list(t) for t in p["preds"]]} for p in panos}
    return bp.score_model(REPO, city, "rampnet", records, {p["pano"]: p["gt"] for p in panos},
                          rsq)


def _metrics(panos, rsq):
    out = {}
    for thr in THRESHOLDS:
        s = _score_at(panos, thr, rsq)
        out[f"{thr:.2f}"] = {"P": rnd(s.precision), "R": rnd(s.recall), "F1": rnd(s.f1),
                             "tp": s.tp, "fp": s.fp, "fn": s.fn}
    out["AP"] = rnd(pr_curve_and_ap(panos, rsq).ap)
    return out


def _contrast(s_arm, s_ref, sizes, thr):
    rng = np.random.default_rng(SEED)
    r = bp.observed_and_se(s_arm, sizes, thr, rng, N_REPS, paired=s_ref)
    return {k: {"observed": rnd(v["observed"]), "ci_lo": rnd(v["ci_lo"]),
                "ci_hi": rnd(v["ci_hi"])} for k, v in r.items() if k != "max_f1"}


def build_report(stats, cache_root=PROBE_ROOT):
    rsq = radius_sq_for()
    arms, lv, rp = build_arms(stats)
    gts = {c: bundle_ground_truths(c)[0] for c in PROBE_SPLITS}
    data = {}
    for a, d in arms.items():
        for c in d["splits"]:
            p = cache_path(cache_root, a, c)
            if os.path.exists(p):
                data[(a, c)] = _panos(read_cache(p)[0], gts[c])
    scored = {k: _scored(k[1], v, rsq) for k, v in data.items()}
    rep = {"protocol": {"thresholds": list(THRESHOLDS), "score_floor": SCORE_FLOOR,
                        "min_distance": MIN_DISTANCE, "n_reps": N_REPS, "seed": SEED,
                        "bootstrap": "pano-level paired cluster bootstrap "
                                     "(benchmark_power_135.observed_and_se)",
                        "where": "native -> Resize 2048x4096 bilinear -> transform -> PRE"},
           "levels": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                          for kk, vv in v.items()} for k, v in lv.items()},
           "repairs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                           for kk, vv in v.items()} for k, v in rp.items()},
           "arms": {a: {"splits": list(d["splits"]), "ops": [list(o) for o in d["ops"]]}
                    for a, d in arms.items()},
           "per_split": {}, "pooled": {}}
    for c in PROBE_SPLITS:
        if (CONTROL, c) not in data:
            continue
        ent = {"metrics": {}, "vs_none": {}}
        for a in arms:
            if (a, c) not in data:
                continue
            ent["metrics"][a] = _metrics(data[(a, c)], rsq)
            if a != CONTROL:
                n = len(scored[(a, c)].pids)
                ent["vs_none"][a] = {f"{t:.2f}": _contrast(scored[(a, c)],
                                                           scored[(CONTROL, c)], [n], t)
                                     for t in THRESHOLDS}
        rep["per_split"][c] = ent
    for name, members in (("GSV pool (degrade)", GSV_SPLITS),
                          ("GoPro pool (repair)", GOPRO_SPLITS)):
        members = [c for c in members if (CONTROL, c) in data]
        ent = {"members": members, "vs_none": {}}
        for a, d in arms.items():
            if a == CONTROL or not all((a, c) in data for c in members) or not members:
                continue
            if not set(members) <= set(d["splits"]):
                continue
            sa = bp.stack([scored[(a, c)] for c in members])
            sc = bp.stack([scored[(CONTROL, c)] for c in members])
            sizes = [len(scored[(a, c)].pids) for c in members]
            ent["vs_none"][a] = {f"{t:.2f}": _contrast(sa, sc, sizes, t) for t in THRESHOLDS}
        rep["pooled"][name] = ent
    # The Laurens gap, and what share of it each degradation reproduces on laurens_gsv.
    if (CONTROL, REF_GSV) in data and (CONTROL, REF_GOPRO) in data:
        gap = {}
        for t in THRESHOLDS:
            k = f"{t:.2f}"
            rg = rep["per_split"][REF_GSV]["metrics"][CONTROL][k]["R"]
            rm = rep["per_split"][REF_GOPRO]["metrics"][CONTROL][k]["R"]
            shares = {}
            for a, cs in rep["per_split"][REF_GSV]["vs_none"].items():
                dr = cs[k]["recall"]["observed"]
                shares[a] = rnd(dr / (rm - rg)) if rm != rg else None
            gap[k] = {"recall_gsv": rg, "recall_gopro": rm, "gap": rnd(rm - rg),
                      "share_reproduced": shares}
        rep["laurens_gap"] = gap
    return rep


def markdown(rep, stats):
    L = ["# Issue #82 Step 1: frozen-model nuisance probe", "",
         "Generated by `scripts/analysis/aug_probe_82.py report`; do not edit by hand.", "",
         "## Rig statistics (median per split, at 2048x4096)", "",
         "| split | native w | JPEG q (native) | lum mean | lum std | sat mean | log2 R/B | "
         "hf_frac | lap var | noise sigma |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, s in stats["summary"].items():
        def g(k, nd=3):
            v = s.get(k)
            return "-" if v is None else f"{v['median']:.{nd}f}"
        L.append(f"| {c} | {g('native_w', 0)} | {g('jpeg_q_native', 0)} | {g('lum_mean', 1)} | "
                 f"{g('lum_std', 1)} | {g('sat_mean', 1)} | {g('log2_r_over_b')} | "
                 f"{g('hf_frac', 4)} | {g('lap_var', 0)} | {g('noise_sigma', 2)} |")
    L += ["", "## Levels (GSV degradation; mid level = measured GoPro value)", "",
          "| axis | levels (half, gopro, beyond) | measured | note |", "|---|---|---:|---|"]
    for ax, d in rep["levels"].items():
        L.append(f"| {ax} | {', '.join(str(x) for x in d['levels'])} | {d['measured']} | "
                 f"{d['note']} |")
    for thr in ("0.30", "0.55"):
        L += ["", f"## Recall / precision / F1 change vs the untransformed run, at {thr}", "",
              "Paired pano bootstrap, 95% interval in brackets.", "",
              "| split | arm | dR | dP | dF1 |", "|---|---|---|---|---|"]
        for c, ent in rep["per_split"].items():
            for a, cs in ent["vs_none"].items():
                d = cs[thr]
                L.append(f"| {c} | {a} | {_fmt(d['recall'])} | {_fmt(d['precision'])} | "
                         f"{_fmt(d['f1'])} |")
        for name, ent in rep["pooled"].items():
            for a, cs in ent["vs_none"].items():
                d = cs[thr]
                L.append(f"| **{name}** | {a} | {_fmt(d['recall'])} | "
                         f"{_fmt(d['precision'])} | {_fmt(d['f1'])} |")
    if "laurens_gap" in rep:
        L += ["", "## Share of the Laurens recall gap reproduced on laurens_gsv", ""]
        for thr, g in rep["laurens_gap"].items():
            L.append(f"- at {thr}: recall GSV {g['recall_gsv']}, GoPro {g['recall_gopro']}, "
                     f"gap {g['gap']}")
        L += ["", "| arm | share @0.30 | share @0.55 |", "|---|---:|---:|"]
        for a in rep["laurens_gap"]["0.30"]["share_reproduced"]:
            L.append(f"| {a} | {rep['laurens_gap']['0.30']['share_reproduced'][a]} | "
                     f"{rep['laurens_gap']['0.55']['share_reproduced'][a]} |")
    return "\n".join(L) + "\n"


def _fmt(d):
    if d["ci_lo"] is None:
        return f"{d['observed']:+.4f}"
    return f"{d['observed']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"


def cmd_report(args):
    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)
    rep = build_report(stats, args.cache_root)
    md = markdown(rep, stats)
    if args.check:
        with open(RESULTS, encoding="utf-8") as f:
            old = json.load(f)
        same = json.loads(json.dumps(rep)) == old
        print("results.json up to date" if same else "results.json STALE")
        return 0 if same else 1
    write_json(RESULTS, rep)
    with open(RESULTS_MD, "w", encoding="utf-8", newline="") as f:
        f.write(md)
    print(md)
    return 0


def cmd_arms(args):
    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)
    arms, lv, rp = build_arms(stats)
    for a, d in arms.items():
        print(f"{a:24s} {','.join(d['splits']):60s} {d['ops']}")
    print(json.dumps(lv, indent=1))
    print(json.dumps(rp, indent=1))
    n = sum(len(d["splits"]) for d in arms.values())
    print(f"{len(arms)} arms, {n} arm x split caches")


# --------------------------------------------------------------------------- #
def rnd(v, nd=ND):
    if v is None:
        return None
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--panos-root", default=REPO)
    s.add_argument("--cities", type=lambda x: [c for c in x.split(",") if c],
                   default=list(STATS_SPLITS))
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--out", default=STATS)
    s.add_argument("--usage-log", default=USAGE_LOG)
    a = sub.add_parser("arms")
    a.add_argument("--stats", default=STATS)
    e = sub.add_parser("extract")
    e.add_argument("--panos-root", default=REPO)
    e.add_argument("--stats", default=STATS)
    e.add_argument("--cities", type=lambda x: [c for c in x.split(",") if c],
                   default=list(PROBE_SPLITS))
    e.add_argument("--arms", default=None, help="comma list; default every arm")
    e.add_argument("--cache-root", default=PROBE_ROOT)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--prep-threads", type=int, default=4)
    e.add_argument("--usage-log", default=USAGE_LOG)
    e.add_argument("--note", default=None)
    c = sub.add_parser("check")
    c.add_argument("--cache-root", default=PROBE_ROOT)
    c.add_argument("--out", default=os.path.join(OUT_DIR, "instrument_check.json"))
    r = sub.add_parser("report")
    r.add_argument("--stats", default=STATS)
    r.add_argument("--cache-root", default=PROBE_ROOT)
    r.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    fn = {"stats": cmd_stats, "arms": cmd_arms, "extract": cmd_extract, "check": cmd_check,
          "report": cmd_report}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
