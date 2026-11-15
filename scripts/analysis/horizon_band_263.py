"""An upscaled second pass over the horizon band, for the shipped RampNet weights (#263).

Stage 0b and 0c of ``detector-phase1-horizon-2x-2026-11-15.md`` (the plan, section 4). The
idea: keep the production 1x pass everywhere, and add a second pass at 1.5x or 2x over a
horizontal band around the horizon, where the far-field ramps sit. Band peaks are merged with
the 1x peaks by #197's seam-wrapped score-ordered ``dedupe`` at the scorer's match radius.

Two band cores (normalized image rows) are compared throughout:

    wide    0.45-0.60
    narrow  0.47-0.56

Subcommands:

    simulate   Stage 0b, CPU, EXPLORATORY (binds nothing). The band is simulated by
               keeping only the in-core peaks of a committed full-pano upscaled cache
               (#196: u4096 for 2x, r3072 for 1.5x) on the seven selection splits S7.
               Paired split-stratified bootstrap intervals, the matched-recall threshold
               baseline, both bounds for far-field detections, and a leave-one-split-out
               choice between the two cores.
    figures    Redraw the Stage 0a and 0b figures from the committed json.
    extract    Stage 0c, GPU (the A40 on makelab2): a REAL strip forward. Rows around the
               core are cut from the 2048x4096 production image upsampled 1.5x or 2x, padded
               with a circular wrap, and run with the head at input/4. Also runs the 1x pass
               and, on S7, the full-pano pass at the same scale and head for the context
               rule. Writes peak caches and GPU-side time and peak memory per pass.
    calibrate  Stage 0c, CPU, after the selection extract: the context rule on S7 and the
               band threshold on sao_paulo, frozen in calibration.json BEFORE any held-out
               split is read.
    report     Stage 0c, CPU, after the held-out extract: the binding gate on manual_gold,
               the open-rig claim on juneau + bayonne, and the cost rule.
    run0c      extract (selection) -> calibrate -> extract (held out) -> report, stopping
               before the held-out read if the context rule fails.

**Held-out rule.** manual_gold, juneau and bayonne are read ONCE, at Stage 0c, after
calibration.json is frozen. ``extract`` refuses a held-out split without
``--heldout-read-once``, without a (non-smoke) calibration.json, with ``--limit`` or
``--force``, mixed with selection splits, or when its cache already exists. ``simulate``
never reads them (no committed upscaled cache exists for them). laurens_mapillary is reported
only, never pooled and never in a band rule.

The Stage 0c command, on makelab2 (about 4 GPU-h on the A40)::

    python scripts/analysis/horizon_band_263.py run0c --panos-root /path/to/RampNet

Stage 0b, and its reproduction::

    python scripts/analysis/horizon_band_263.py simulate           # -> simulate.json + .md
    python scripts/analysis/horizon_band_263.py simulate --check   # regenerate, compare bytes
    python scripts/analysis/horizon_band_263.py figures
"""
import argparse
import gc
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from queue import Queue

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)
from rampnet.geometry import dist_sq  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402
import input_res_sweep_25 as irs  # noqa: E402
import two_scale_197 as tsc  # noqa: E402
from recall_by_depth_112 import CAM_H, flat_range  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "horizon_band_263")
SIM_JSON = os.path.join(OUT_DIR, "simulate.json")
MERGED_JSON = os.path.join(OUT_DIR, "merged_pairs.json")
CACHE_0C = os.path.join(OUT_DIR, "cache_0c")
CALIBRATION = os.path.join(OUT_DIR, "calibration.json")
GATE_JSON = os.path.join(OUT_DIR, "gate_0c.json")
FIG_DIR = os.path.join(REPO, "docs", "figures", "horizon_band_263")
USAGE_LOG = irs.USAGE_LOG

# --------------------------------------------------------------------------- #
# splits (plan section 3)
# --------------------------------------------------------------------------- #
S7 = ("richmond", "annapolis", "clovis", "morgantown", "bend", "paterson", "gainesville")
MAPILLARY4 = ("richmond", "annapolis", "clovis", "morgantown")
GSV3 = ("bend", "paterson", "gainesville")
CALIB_SPLIT = "sao_paulo"
SELECTION = S7 + (CALIB_SPLIT,)
GOLD = "manual_gold"
OPEN_RIG = ("juneau", "bayonne")
HELD_OUT = (GOLD,) + OPEN_RIG
REPORT_ONLY = ("laurens_mapillary",)
SIM_POOLS = {
    "S7": S7,
    "Mapillary4 (context)": MAPILLARY4,
    "GSV3 (context)": GSV3,
    "S7 minus paterson (context)": tuple(c for c in S7 if c != "paterson"),
}

# --------------------------------------------------------------------------- #
# the band (plan section 4)
# --------------------------------------------------------------------------- #
CORES = {"wide": (0.45, 0.60), "narrow": (0.47, 0.56)}
CORE_ORDER = ("wide", "narrow")          # LOSO ties go to the first (wide is tested first)
SIM_ARMS = {"2x": "u4096", "1.5x": "r3072"}   # 0b: committed #196 full-pano caches
SCALES = {"1.5x": 1.5, "2x": 2.0}
SIM_CONFIGS = ("2x-wide", "2x-narrow", "1.5x-wide", "1.5x-narrow")
CONFIGS_0C = ("1.5x-wide", "1.5x-narrow", "2x-wide")
PRIMARY_0C = "1.5x-wide"                 # "Tested first" -- the binding config
CONTEXT = 0.05                           # rows of context either side of the core
WRAP_1X = 128                            # circular wrap each side, 1x px (192 px at 1.5x)
STRIDE = 32                              # ConvNeXt total stride: strips sit on its grid
HEAD_DIV = 4                             # head at input/4
BASE_HW = (2048, 4096)
BASE_MIN_DISTANCE = 10                   # peak_local_max at the 512x1024 heatmap
SCORE_FLOOR = 0.05
T_OP = 0.30
FAR_IGNORE_M = 25.0                      # the far bound: unmatched detections beyond 25 m
MATCH_GRID = tsc.MATCH_GRID
N_REPS = 2000
SEED = 263
ND = 4
RANGE_BINS = ((0.0, 8.0), (8.0, 18.0), (18.0, 25.0), (25.0, math.inf))

# 0c rules, numbers exactly as the plan states them
CONTEXT_MIN_REPRODUCED = 0.95
CONTEXT_MAX_MEDIAN_SHIFT = 0.05
CALIB_SHIFT_TRIGGER = 0.03
CALIB_PRECISION_TRIGGER = 0.05
OPEN_RIG_MIN_GAIN = 0.02
COST_LIMIT = 1.75
TWO_X_MIN_GAIN_RATIO = 1.25


def config_parts(cfg):
    """'1.5x-wide' -> ('1.5x', 'wide')."""
    scale, core = cfg.split("-")
    if scale not in SCALES or core not in CORES:
        raise ValueError(cfg)
    return scale, core


# --------------------------------------------------------------------------- #
# pure band helpers (unit-tested)
# --------------------------------------------------------------------------- #
def in_core(y, core):
    """Inclusive both ends, as the 0b scratch run and the plan's table used."""
    return core[0] <= y <= core[1]


def band_preds(preds, core, t=T_OP):
    return [p for p in preds if p[2] >= t and in_core(p[1], core)]


def band_merge(base, band, core, rsq, t_base=T_OP, t_band=T_OP):
    """1x peaks >= t_base everywhere, plus in-core band peaks >= t_band, deduped across the
    two passes by #197's seam-wrapped score-ordered suppression at the match radius."""
    return tsc.dedupe(tsc.thresholded(base, t_base), band_preds(band, core, t_band), rsq)[0]


def is_far_for_ignore(y, far_m=FAR_IGNORE_M):
    """Beyond ``far_m`` on flat ground, or at/above the horizon (no ground range)."""
    r = flat_range(y, CAM_H)
    return r is None or r >= far_m


def drop_far_unmatched(preds, gt, rsq, far_m=FAR_IGNORE_M):
    """The 'ignore' bound: remove every prediction that matches no GT ramp and sits beyond
    ``far_m``. Greedy matching runs in descending score (``score_pano``'s order); a removed
    prediction claimed nothing, so every other assignment, and every TP, is unchanged."""
    order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    ordered = [preds[i] for i in order]
    assign = greedy_match([(p[0], p[1]) for p in ordered], gt.gt_points, rsq,
                          PANO_SCALE_X, PANO_SCALE_Y, True)
    return [p for p, (gi, _) in zip(ordered, assign)
            if gi >= 0 or not is_far_for_ignore(p[1], far_m)]


def apply_bound(panos, bound, rsq):
    if bound == "fp":
        return panos
    if bound == "ignore":
        return [{**p, "preds": drop_far_unmatched(p["preds"], p["gt"], rsq)} for p in panos]
    raise ValueError(bound)


def loso_core_choice(f1_by_split_core):
    """Leave-one-split-out choice between the cores.

    ``f1_by_split_core`` = {split: {core: counts dict}}. For each held-out split, the core
    with the higher pooled F1 on the OTHER splits is chosen; ties go to CORE_ORDER[0].
    Returns ({held_out: core}, {held_out: {core: F1 on the others}})."""
    chosen, table = {}, {}
    splits = list(f1_by_split_core)
    for held in splits:
        f = {}
        for core in CORE_ORDER:
            f[core] = tsc.prf(tsc.add_counts(
                f1_by_split_core[c][core] for c in splits if c != held))[2]
        best = CORE_ORDER[0]
        for core in CORE_ORDER[1:]:
            if f[core] > f[best] + 1e-12:
                best = core
        chosen[held], table[held] = best, f
    return chosen, table


def range_bin(y):
    r = flat_range(y, CAM_H)
    if r is None:
        return "at/above horizon"
    for lo, hi in RANGE_BINS:
        if lo <= r < hi:
            return bin_label(lo, hi)
    return bin_label(*RANGE_BINS[-1])


def bin_label(lo, hi):
    return f">= {lo:g} m" if math.isinf(hi) else f"{lo:g}-{hi:g} m"


RANGE_LABELS = tuple(bin_label(lo, hi) for lo, hi in RANGE_BINS) + ("at/above horizon",)


# --------------------------------------------------------------------------- #
# pure strip geometry (unit-tested)
# --------------------------------------------------------------------------- #
def row_step(scale, stride=STRIDE):
    """Smallest 1x row count k with k*scale an integer multiple of ``stride``: a strip whose
    first row and height are multiples of k sits exactly on the backbone's stride grid of the
    full upscaled image, so its interior features equal the full pass's (context aside)."""
    for k in range(1, 4097):
        v = k * scale
        if abs(v - round(v)) < 1e-9 and int(round(v)) % stride == 0:
            return k
    raise ValueError(scale)


def strip_geometry(core, scale, context=CONTEXT, hw=BASE_HW, wrap_1x=WRAP_1X,
                   stride=STRIDE, head_div=HEAD_DIV):
    """The strip for one band core at one scale.

    Rows [r0, r1) of the 1x production image: the smallest stride-aligned window that
    contains [core_lo - context, core_hi + context]. So the context is AT LEAST ``context``
    on both sides (the plan's 0.40-0.65 for the wide core becomes 0.375-0.65625 at 1.5x).
    Columns: the full width plus ``wrap_1x`` px of circular wrap each side.

    >>> g = strip_geometry((0.45, 0.60), 1.5)
    >>> (g["r0"], g["r1"], g["in_h"], g["in_w"], g["hm_h"], g["hm_w"])
    (768, 1344, 864, 6528, 216, 1632)
    """
    H, W = hw
    step = row_step(scale, stride)
    lo_px = (core[0] - context) * H
    hi_px = (core[1] + context) * H
    r0 = max(0, int(math.floor(lo_px / step + 1e-9)) * step)
    r1 = min(H, int(math.ceil(hi_px / step - 1e-9)) * step)
    sw = wrap_1x * scale
    assert abs(sw - round(sw)) < 1e-9 and int(round(sw)) % stride == 0, "wrap off the grid"
    in_h = int(round(scale * (r1 - r0)))
    in_w = int(round(scale * (W + 2 * wrap_1x)))
    full_h, full_w = int(round(scale * H)), int(round(scale * W))
    assert in_h % stride == 0 and in_w % stride == 0 and full_w % stride == 0
    return {"core": list(core), "scale": scale, "context": context, "r0": r0, "r1": r1,
            "rows_norm": [round(r0 / H, 6), round(r1 / H, 6)],
            "in_h": in_h, "in_w": in_w, "R0": int(round(scale * r0)),
            "wrap_in": int(round(sw)),
            "hm_h": in_h // head_div, "hm_w": in_w // head_div,
            "full_in": [full_h, full_w],
            "full_hm": [full_h // head_div, full_w // head_div],
            "row_off_hm": int(round(scale * r0)) // head_div,
            "wrap_hm": int(round(sw)) // head_div,
            "min_distance": int(round(BASE_MIN_DISTANCE * scale)),
            "megapixels": round(in_h * in_w / 1e6, 3)}


def strip_peaks_to_pano(peaks_rc, geom):
    """Strip-heatmap peaks [(row, col, score)] -> pano-normalized core peaks [(x, y, s)].

    Same convention as ``threshold_sweep.peaks_to_dets`` on the full upscaled heatmap
    (x = col / W, y = row / H): a strip peak lands on exactly the coordinates the full pass
    would give it. Peaks in the wrap margins (they belong to the unwrapped copy) and peaks
    outside the core rows are dropped."""
    fh, fw = geom["full_hm"]
    out = []
    for r, c, s in peaks_rc:
        col = c - geom["wrap_hm"]
        if not 0 <= col < fw:
            continue
        y = (r + geom["row_off_hm"]) / fh
        if not in_core(y, geom["core"]):
            continue
        out.append((col / fw, y, s))
    return out


def peaks_rc(h, floor, md):
    """``peak_local_max`` exactly as ``threshold_sweep.peaks_to_dets`` calls it
    (clip to [0, 1], ``exclude_border=False``), returning (row, col, score)."""
    from skimage.feature import peak_local_max
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=md, threshold_abs=floor,
                        exclude_border=False)
    return [(int(r), int(c), float(h[r][c])) for r, c in pk]


def make_strip(full, geom):
    """(C, H, W) full upscaled tensor/array -> the (C, in_h, in_w) strip: rows [R0, R0+in_h)
    with ``wrap_in`` columns of circular wrap each side. Works on numpy or torch."""
    R0, h, w = geom["R0"], geom["in_h"], geom["wrap_in"]
    rows = full[:, R0:R0 + h, :]
    if hasattr(rows, "new_empty"):           # torch
        import torch
        return torch.cat([rows[..., -w:], rows, rows[..., :w]], dim=-1).contiguous()
    return np.concatenate([rows[..., -w:], rows, rows[..., :w]], axis=-1)


# --------------------------------------------------------------------------- #
# pure 0c rules (unit-tested)
# --------------------------------------------------------------------------- #
def match_peaks(ref, cand, rsq):
    """Greedy 1:1: each ``ref`` peak in descending score claims the nearest unclaimed
    ``cand`` peak within the match radius (seam-wrapped). Returns [(ref, cand or None)]."""
    ref = sorted(ref, key=lambda p: -p[2])
    assign = greedy_match([(p[0], p[1]) for p in ref], [(q[0], q[1]) for q in cand], rsq,
                          PANO_SCALE_X, PANO_SCALE_Y, True)
    return [(p, cand[k] if k >= 0 else None) for p, (k, _) in zip(ref, assign)]


def context_stats(pairs_by_pano, rsq, t_ref=T_OP):
    """The context rule over panos: ``pairs_by_pano`` = [(full core peaks, strip core peaks)].

    Reference = full-pass in-core peaks >= t_ref. Reproduced (the literal reading) = a strip
    core peak at any score >= the floor within the match radius; the stricter reading also
    needs that strip peak >= t_ref. Score shift = strip - full on the reproduced pairs."""
    n_ref = n_rep = n_strict = 0
    shifts, offs = [], []
    for full, strip in pairs_by_pano:
        ref = [p for p in full if p[2] >= t_ref]
        for p, q in match_peaks(ref, strip, rsq):
            n_ref += 1
            if q is None:
                continue
            n_rep += 1
            n_strict += q[2] >= t_ref
            shifts.append(q[2] - p[2])
            offs.append(math.sqrt(dist_sq(p[0], p[1], q[0], q[1], PANO_SCALE_X, PANO_SCALE_Y,
                                          True)))
    frac = n_rep / n_ref if n_ref else None
    med = float(np.median(shifts)) if shifts else None
    return {"n_ref": n_ref, "n_reproduced": n_rep,
            "reproduced_fraction": frac,
            "reproduced_fraction_strict": n_strict / n_ref if n_ref else None,
            "median_score_shift": med,
            "median_offset_px": float(np.median(offs)) if offs else None,
            "passes": (frac is not None and frac >= CONTEXT_MIN_REPRODUCED
                       and med is not None and abs(med) < CONTEXT_MAX_MEDIAN_SHIFT),
            "passes_strict": (n_ref > 0 and n_strict / n_ref >= CONTEXT_MIN_REPRODUCED
                              and med is not None and abs(med) < CONTEXT_MAX_MEDIAN_SHIFT)}


def paired_shifts(base_core, strip_core, rsq, t=T_OP):
    """Score shift strip - 1x on location-paired peaks, symmetric in the two passes: all
    cross pairs within the radius are paired closest-first, 1:1, and a pair counts when
    either peak reaches ``t`` (pairs of two sub-threshold peaks never act)."""
    cands = []
    for i, p in enumerate(base_core):
        for j, q in enumerate(strip_core):
            d = dist_sq(p[0], p[1], q[0], q[1], PANO_SCALE_X, PANO_SCALE_Y, True)
            if d < rsq:
                cands.append((d, i, j))
    cands.sort()
    used_i, used_j, out = set(), set(), []
    for d, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        if max(base_core[i][2], strip_core[j][2]) >= t:
            out.append(strip_core[j][2] - base_core[i][2])
    return out


def band_threshold(strip_precision_by_t, p_1x):
    """Lowest grid threshold whose in-core strip precision reaches the 1x in-core precision
    at 0.30 ("set to equal 1x precision"); None if no grid threshold reaches it."""
    for t in MATCH_GRID:
        p = strip_precision_by_t.get(t)
        if p is not None and p >= p_1x - 1e-12:
            return t
    return None


def calibration_rule(median_shift, p_strip_030, p_1x_030):
    """True when a band threshold must be set (plan: median paired shift above 0.03, or
    precision at 0.30 differing by more than 0.05)."""
    return (abs(median_shift) > CALIB_SHIFT_TRIGGER
            or abs(p_strip_030 - p_1x_030) > CALIB_PRECISION_TRIGGER)


def gate_bound(d_base, d_matched):
    """One bound of the manual_gold gate. ``d_base`` = merged - 1x@0.30 contrast;
    ``d_matched`` = merged - 1x at the matched-recall threshold (None if no grid threshold
    reaches the merged recall, which counts as not beaten, as in #197)."""
    recall_ok = d_base["recall"]["ci_lo"] > 0
    beats_matched = d_matched is not None and d_matched["f1"]["ci_lo"] > 0
    f1_ok = d_base["f1"]["ci_hi"] >= 0          # "not wholly below zero"
    return {"recall_ci_above_zero": recall_ok, "beats_matched_threshold": beats_matched,
            "f1_vs_base_not_wholly_below_zero": f1_ok,
            "passes": recall_ok and beats_matched and f1_ok}


def gate_verdict(bounds):
    """PASS only if the gate holds under BOTH bounds (far unmatched as FP and as ignore)."""
    return "PASS" if all(b["passes"] for b in bounds.values()) else "FAIL"


def open_rig_verdict(d_recall):
    ok = d_recall["observed"] >= OPEN_RIG_MIN_GAIN and d_recall["ci_lo"] > 0
    return "OFFERED" if ok else "NOT OFFERED"


# --------------------------------------------------------------------------- #
# shared scoring
# --------------------------------------------------------------------------- #
def rnd(v, nd=ND):
    return irs.rnd(v, nd)


def scored(city, panos, rsq):
    return irs.scored_from_panos(city, panos, rsq)


def contrast(sa, sb, sizes):
    """Paired pano bootstrap of sa - sb (lists of per-split Scored), stratified by split,
    on final lists (threshold 0.0). Unrounded."""
    rng = np.random.default_rng(SEED)
    r = bp.observed_and_se(bp.stack(sa), sizes, 0.0, rng, N_REPS, paired=bp.stack(sb))
    return {k: {"observed": v["observed"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
            for k, v in r.items() if k != "max_f1"}


def round_contrast(c):
    return None if c is None else {k: {kk: rnd(vv) for kk, vv in v.items()}
                                   for k, v in c.items()}


def metrics_of(c):
    p, r, f = tsc.prf(c)
    return {"P": rnd(p), "R": rnd(r), "F1": rnd(f), "tp": c["tp"], "fp": c["fp"],
            "tp_rec": c["tp_rec"], "n_gt": c["n_gt"]}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_group(splits, variant_panos, rsq, cfg_names, base_name="base"):
    """Shared by simulate and report: for each config in ``cfg_names`` and each bound, the
    merged-vs-base and merged-vs-matched-threshold contrasts over ``splits``.

    ``variant_panos[(name, split)]`` = final pano lists, for "base", each config, and
    f"base@{t:.2f}" on MATCH_GRID."""
    sizes = [len(variant_panos[(base_name, c)]) for c in splits]
    cache = {}

    def sc(name, bound):
        key = (name, bound)
        if key not in cache:
            cache[key] = [scored(c, apply_bound(variant_panos[(name, c)], bound, rsq), rsq)
                          for c in splits]
        return cache[key]

    def cnt(name, bound):
        return tsc.add_counts(tsc.counts(apply_bound(variant_panos[(name, c)], bound, rsq),
                                         rsq) for c in splits)

    by_t = {t: cnt(f"base@{t:.2f}", "fp") for t in MATCH_GRID}
    out = {"n_panos": sum(sizes), "members": list(splits), "configs": {}}
    out["base"] = {b: metrics_of(cnt(base_name, b)) for b in ("fp", "ignore")}
    for cfg in cfg_names:
        ent = {}
        fc = cnt(cfg, "fp")
        t_m = tsc._matched_threshold(by_t, tsc.prf(fc)[1])   # recall: bound-independent
        for b in ("fp", "ignore"):
            mc, bc = cnt(cfg, b), cnt(base_name, b)
            d_base = contrast(sc(cfg, b), sc(base_name, b), sizes)
            d_m, matched = None, None
            if t_m is not None:
                tc = cnt(f"base@{t_m:.2f}", b)
                d_m = contrast(sc(cfg, b), sc(f"base@{t_m:.2f}", b), sizes)
                matched = {"threshold": t_m, **metrics_of(tc),
                           "extra_fp_vs_base": tc["fp"] - bc["fp"],
                           "merged_minus_matched": round_contrast(d_m)}
            ent[b] = {"metrics": metrics_of(mc),
                      "ramps_gained": mc["tp_rec"] - bc["tp_rec"],
                      "fps_added": mc["fp"] - bc["fp"],
                      "vs_base": round_contrast(d_base),
                      "matched_recall_threshold": matched,
                      "label_197": tsc.verdict(d_base, d_m),
                      "_raw": {"vs_base": d_base, "vs_matched": d_m}}
        out["configs"][cfg] = ent
    return out


def strip_raw(obj):
    if isinstance(obj, dict):
        return {k: strip_raw(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_raw(v) for v in obj]
    return obj


def _d(c):
    return f"{c['observed']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]"


def check_or_write(paths_texts, check, label):
    if check:
        bad = []
        for path, text in paths_texts:
            if not os.path.exists(path):
                bad.append(f"{path}: missing")
                continue
            with open(path, "rb") as f:
                got = f.read()
            want = text.encode("utf-8")
            if got != want:
                bad.append(f"{path}: committed sha256 {hashlib.sha256(got).hexdigest()[:12]} "
                           f"!= regenerated {hashlib.sha256(want).hexdigest()[:12]}")
        for b in bad:
            print(b)
        print("CHECK:", "FAIL" if bad else f"OK ({label} reproduce byte for byte)")
        return 1 if bad else 0
    for path, text in paths_texts:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"-> {path}")
    return 0


# --------------------------------------------------------------------------- #
# 0b: simulate (CPU)
# --------------------------------------------------------------------------- #
def load_sim_rows(cache_root, cities):
    """{city: [(pano, gt, r2048 preds, {scale: preds})]} from the committed #196 caches."""
    out = {}
    for c in cities:
        r, _ = irs.read_arm_cache(irs.cache_path(cache_root, "r2048", c))
        ups = {}
        for scale, arm in SIM_ARMS.items():
            u, _ = irs.read_arm_cache(irs.cache_path(cache_root, arm, c))
            ups[scale] = {p["pano"]: p for p in u}
            assert set(ups[scale]) == {p["pano"] for p in r}, (c, arm, "pano sets differ")
        rows = []
        for p in sorted(r, key=lambda q: q["pano"]):
            for scale in SIM_ARMS:
                q = ups[scale][p["pano"]]
                assert irs._gt_to_json(p["gt"]) == irs._gt_to_json(q["gt"]), (c, p["pano"])
            rows.append((p["pano"], p["gt"], [tuple(t) for t in p["preds"]],
                         {s: [tuple(t) for t in ups[s][p["pano"]]["preds"]] for s in SIM_ARMS}))
        out[c] = rows
    return out


def sim_variants(rows, rsq, loso=None):
    """{variant: pano list} for one split. ``loso`` = {scale: core} adds f"{scale}-loso"."""
    v = {}

    def put(name, f):
        v[name] = [{"pano": pid, "gt": gt, "preds": f(r, u)} for pid, gt, r, u in rows]

    put("base", lambda r, u: tsc.thresholded(r, T_OP))
    for t in MATCH_GRID:
        put(f"base@{t:.2f}", lambda r, u, t=t: tsc.thresholded(r, t))
    for cfg in SIM_CONFIGS:
        scale, core = config_parts(cfg)
        put(cfg, lambda r, u, s=scale, k=core: band_merge(r, u[s], CORES[k], rsq))
    for scale in SIM_ARMS:
        put(f"full{scale}", lambda r, u, s=scale: tsc.thresholded(u[s], T_OP))
        if loso:
            put(f"{scale}-loso", lambda r, u, s=scale: band_merge(r, u[s], CORES[loso[s]], rsq))
    return v


def approx_hit(g, preds, rsq):
    """The 0b scratch run's approximate test: any peak within the radius (no 1:1)."""
    return any(dist_sq(g[0], g[1], p[0], p[1], PANO_SCALE_X, PANO_SCALE_Y, True) < rsq
               for p in preds)


def build_sim(cache_root=irs.CACHE_ROOT):
    rsq = radius_sq_for()
    rows = load_sim_rows(cache_root, S7 + REPORT_ONLY)

    # leave-one-split-out core choice per scale (FP bound, F1 at 0.30), on S7 only
    pre = {c: sim_variants(rows[c], rsq) for c in S7}
    loso, loso_tab = {}, {}
    for scale in SIM_ARMS:
        cnts = {c: {core: tsc.counts(pre[c][f"{scale}-{core}"], rsq) for core in CORES}
                for c in S7}
        loso[scale], loso_tab[scale] = loso_core_choice(cnts)
    var = {}
    for c in S7:
        per = sim_variants(rows[c], rsq, loso={s: loso[s][c] for s in SIM_ARMS})
        for name, ps in per.items():
            var[(name, c)] = ps
    for c in REPORT_ONLY:
        for name, ps in sim_variants(rows[c], rsq).items():
            var[(name, c)] = ps

    cfgs = SIM_CONFIGS + tuple(f"{s}-loso" for s in SIM_ARMS)
    pools = {name: compare_group(m, var, rsq, cfgs) for name, m in SIM_POOLS.items()}
    report_only = {c: compare_group((c,), var, rsq, SIM_CONFIGS) for c in REPORT_ONLY}

    # threshold frontier on S7 (for the figure): lowering the 1x threshold from 0.30
    base_c = tsc.add_counts(tsc.counts(var[("base", c)], rsq) for c in S7)
    frontier = []
    for t in MATCH_GRID:
        if t > T_OP + 1e-9:
            continue
        tc = tsc.add_counts(tsc.counts(var[(f"base@{t:.2f}", c)], rsq) for c in S7)
        frontier.append([t, tc["tp_rec"] - base_c["tp_rec"], tc["fp"] - base_c["fp"]])

    # misses by flat-ground range (scorer definition: base hit_gt == -inf at 0.30)
    ranges = {}
    for scale in SIM_ARMS:
        tab = {lb: {"gt": 0, "misses": 0, "in_wide": 0, "in_narrow": 0,
                    "recovered_full_pass": 0, "recovered_wide_band": 0,
                    "recovered_narrow_band": 0} for lb in RANGE_LABELS}
        for c in S7:
            pts, _ = irs.gt_points_in_scored_order(var[("base", c)])
            hb = scored(c, var[("base", c)], rsq).hit_gt
            hf = scored(c, var[(f"full{scale}", c)], rsq).hit_gt
            hw = scored(c, var[(f"{scale}-wide", c)], rsq).hit_gt
            hn = scored(c, var[(f"{scale}-narrow", c)], rsq).hit_gt
            for k, (_, gy) in enumerate(pts):
                e = tab[range_bin(gy)]
                e["gt"] += 1
                if hb[k] > -np.inf:
                    continue
                e["misses"] += 1
                e["in_wide"] += in_core(gy, CORES["wide"])
                e["in_narrow"] += in_core(gy, CORES["narrow"])
                e["recovered_full_pass"] += bool(hf[k] > -np.inf)
                e["recovered_wide_band"] += bool(hw[k] > -np.inf)
                e["recovered_narrow_band"] += bool(hn[k] > -np.inf)
        ranges[scale] = tab

    # the scratch run's approximate figures, reproduced (plan section 4 quotes these)
    approx = {}
    for scale in SIM_ARMS:
        by_range = {lb: {"misses": 0, "in_wide": 0, "recovered": 0} for lb in RANGE_LABELS}
        by_split = {}
        for c in S7 + REPORT_ONLY:
            m = rec = n_far = n = 0
            for pid, gt, r, u in rows[c]:
                if not gt.fn_confirmed:
                    continue
                r30, u30 = tsc.thresholded(r, T_OP), tsc.thresholded(u[scale], T_OP)
                for g in gt.gt_points:
                    n += 1
                    rr = flat_range(g[1], CAM_H)
                    n_far += rr is not None and rr >= FAR_IGNORE_M
                    if approx_hit(g, r30, rsq):
                        continue
                    m += 1
                    hit_u = approx_hit(g, u30, rsq)
                    rec += hit_u
                    if c in S7:
                        e = by_range[range_bin(g[1])]
                        e["misses"] += 1
                        e["in_wide"] += in_core(g[1], CORES["wide"])
                        e["recovered"] += hit_u
            by_split[c] = {"gt": n, "gt_ge_25m": n_far, "misses": m, "recovered": rec}
        approx[scale] = {"by_range": by_range, "by_split": by_split}

    per_split = {}
    for c in S7 + REPORT_ONLY:
        bc = tsc.counts(var[("base", c)], rsq)
        ent = {"base": metrics_of(bc), "configs": {}}
        for cfg in SIM_CONFIGS:
            mc = tsc.counts(var[(cfg, c)], rsq)
            ent["configs"][cfg] = {"ramps_gained": mc["tp_rec"] - bc["tp_rec"],
                                   "fps_added": mc["fp"] - bc["fp"], "F1": metrics_of(mc)["F1"]}
        if c in S7:
            ent["loso_core"] = {s: loso[s][c] for s in SIM_ARMS}
        per_split[c] = ent

    inputs = {}
    for c in S7 + REPORT_ONLY:
        for arm in ("r2048",) + tuple(SIM_ARMS.values()):
            p = irs.cache_path(cache_root, arm, c)
            inputs[os.path.relpath(p, REPO).replace(os.sep, "/")] = sha256_file(p)

    geoms = {cfg: strip_geometry(CORES[config_parts(cfg)[1]], SCALES[config_parts(cfg)[0]])
             for cfg in SIM_CONFIGS}
    rep = {
        "protocol": {
            "plan": "detector-phase1-horizon-2x-2026-11-15.md section 4, Stage 0b (exploratory)",
            "pooled_statistic": "micro (pooled counts over the splits), as the plan's 0b table",
            "splits": list(S7), "report_only": list(REPORT_ONLY),
            "cores": {k: list(v) for k, v in CORES.items()}, "sim_arms": SIM_ARMS,
            "t_op": T_OP, "far_ignore_m": FAR_IGNORE_M, "cam_h_m": CAM_H,
            "n_reps": N_REPS, "seed": SEED, "match_grid": [MATCH_GRID[0], MATCH_GRID[-1], 0.01],
            "loso_criterion": "pooled F1 at 0.30 (far unmatched as FP) on the other six splits;"
                              " ties to wide",
            "inputs": inputs,
            "strip_geometry_for_0c": geoms},
        "loso": {s: {"chosen": loso[s], "f1_on_others": {h: {k: rnd(v, 6) for k, v in t.items()}
                                                         for h, t in loso_tab[s].items()}}
                 for s in SIM_ARMS},
        "pools": pools, "report_only": report_only, "per_split": per_split,
        "misses_by_range": ranges, "scratch_reproduction": approx,
        "threshold_frontier_S7": frontier,
    }
    return json.loads(json.dumps(strip_raw(rep)))


def markdown_sim(rep):
    L = []
    cfgs = list(next(iter(rep["pools"].values()))["configs"])
    for b, title in (("fp", "far unmatched detections scored as FPs (the standard scorer)"),
                     ("ignore", "far (>= 25 m) unmatched detections ignored (upper bound)")):
        L.append(f"**Band vs 1x at 0.30, {title}. Paired 95% CI, split-stratified pano "
                 "bootstrap, 2,000 draws**\n")
        L.append("| pool | config | P | R | F1 | ramps gained | FPs added | ΔR | ΔF1 | matched "
                 "1x threshold | its FPs added | band − matched ΔF1 | #197 label |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pool, g in rep["pools"].items():
            m = g["base"][b]
            L.append(f"| {pool} | 1x@0.30 | {m['P']:.4f} | {m['R']:.4f} | {m['F1']:.4f} | — | — "
                     "| — | — | — | — | — | |")
            for cfg in cfgs:
                e = g["configs"][cfg][b]
                mm, mt = e["metrics"], e["matched_recall_threshold"]
                mcols = ("unreachable | — | —" if mt is None else
                         f"{mt['threshold']:.2f} | {mt['extra_fp_vs_base']:+d} | "
                         f"{_d(mt['merged_minus_matched']['f1'])}")
                L.append(f"| {pool} | {cfg} | {mm['P']:.4f} | {mm['R']:.4f} | {mm['F1']:.4f} | "
                         f"{e['ramps_gained']:+d} | {e['fps_added']:+d} | "
                         f"{_d(e['vs_base']['recall'])} | {_d(e['vs_base']['f1'])} | {mcols} | "
                         f"{e['label_197']} |")
        L.append("")
    L.append("**Leave-one-split-out choice between the cores (F1 at 0.30 on the other six "
             "splits)**\n")
    L.append("| held-out split | 2x: F1 wide / narrow | 2x choice | 1.5x: F1 wide / narrow | "
             "1.5x choice |")
    L.append("|---|---|---|---|---|")
    for h in S7:
        a, b = rep["loso"]["2x"], rep["loso"]["1.5x"]
        L.append(f"| {h} | {a['f1_on_others'][h]['wide']:.4f} / "
                 f"{a['f1_on_others'][h]['narrow']:.4f} | {a['chosen'][h]} | "
                 f"{b['f1_on_others'][h]['wide']:.4f} / {b['f1_on_others'][h]['narrow']:.4f} | "
                 f"{b['chosen'][h]} |")
    L.append("")
    for scale in SIM_ARMS:
        L.append(f"**S7 misses at 0.30 by flat-ground range, {scale} (scorer definition: "
                 "1:1 greedy match)**\n")
        L.append("| range | GT | misses | GT row in wide core | in narrow core | recovered by the "
                 "full pass alone | by the wide band | by the narrow band |")
        L.append("|---|---|---|---|---|---|---|---|")
        for lb, e in rep["misses_by_range"][scale].items():
            L.append(f"| {lb} | {e['gt']} | {e['misses']} | {e['in_wide']} | {e['in_narrow']} | "
                     f"{e['recovered_full_pass']} | {e['recovered_wide_band']} | "
                     f"{e['recovered_narrow_band']} |")
        L.append("")
    L.append("**Per split at 0.30, far unmatched as FP (ramps gained / FPs added)**\n")
    L.append("| split | 1x F1 | " + " | ".join(SIM_CONFIGS) + " | LOSO core 2x / 1.5x |")
    L.append("|---|---|" + "---|" * len(SIM_CONFIGS) + "---|")
    for c, e in rep["per_split"].items():
        lo = e.get("loso_core")
        L.append(f"| {c}{' (report only)' if c in REPORT_ONLY else ''} | {e['base']['F1']:.4f} | "
                 + " | ".join(f"{e['configs'][k]['ramps_gained']:+d} / "
                              f"{e['configs'][k]['fps_added']:+d}" for k in SIM_CONFIGS)
                 + f" | {lo['2x'] + ' / ' + lo['1.5x'] if lo else '—'} |")
    L.append("")
    L.append("**The 0b scratch run's approximate figures, reproduced (miss = no 1x peak >= 0.30 "
             "within the radius; recovered = an upscaled peak >= 0.30 within it, anywhere)**\n")
    L.append("| range (S7) | 2x: misses / in wide core / recovered | 1.5x: misses / in wide core "
             "/ recovered |")
    L.append("|---|---|---|")
    for lb in RANGE_LABELS:
        a = rep["scratch_reproduction"]["2x"]["by_range"][lb]
        b = rep["scratch_reproduction"]["1.5x"]["by_range"][lb]
        L.append(f"| {lb} | {a['misses']} / {a['in_wide']} / {a['recovered']} | {b['misses']} / "
                 f"{b['in_wide']} / {b['recovered']} |")
    L.append("")
    L.append("| split | GT | GT >= 25 m | 2x: recovered / misses | 1.5x: recovered / misses |")
    L.append("|---|---|---|---|---|")
    for c in S7 + REPORT_ONLY:
        a = rep["scratch_reproduction"]["2x"]["by_split"][c]
        b = rep["scratch_reproduction"]["1.5x"]["by_split"][c]
        L.append(f"| {c} | {a['gt']} | {a['gt_ge_25m']} | {a['recovered']} / {a['misses']} | "
                 f"{b['recovered']} / {b['misses']} |")
    L.append("")
    L.append("**Strip geometry Stage 0c will run (stride-aligned; context >= 0.05)**\n")
    L.append("| config | 1x rows | normalized rows | strip input | heatmap | min_distance | MP |")
    L.append("|---|---|---|---|---|---|---|")
    for cfg, g in rep["protocol"]["strip_geometry_for_0c"].items():
        L.append(f"| {cfg} | {g['r0']}-{g['r1']} | {g['rows_norm'][0]:.4f}-{g['rows_norm'][1]:.4f}"
                 f" | {g['in_h']}x{g['in_w']} | {g['hm_h']}x{g['hm_w']} | {g['min_distance']} | "
                 f"{g['megapixels']:.2f} |")
    L.append("")
    return "\n".join(L)


def cmd_simulate(args):
    rep = build_sim(args.cache_root)
    js = json.dumps(rep, indent=1) + "\n"
    md = markdown_sim(rep) + "\n"
    if not args.check:
        print(md)
    return check_or_write(((args.out, js), (os.path.splitext(args.out)[0] + ".md", md)),
                          args.check, "simulate.json and simulate.md")


# --------------------------------------------------------------------------- #
# figures (0a + 0b)
# --------------------------------------------------------------------------- #
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "svg.hashsalt": "263",
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(args.fig_dir, exist_ok=True)
    written = []

    with open(args.merged, encoding="utf-8") as f:
        mp = json.load(f)
    pat = [p["chebyshev"] for p in mp["pairs"] if p["city"] == "paterson"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = np.arange(0, 24, 1.0)
    other = [p["chebyshev"] for p in mp["pairs"] if p["city"] != "paterson"]
    ax.hist([other, pat],
            bins=bins, stacked=True, color=[BLUE, ORANGE], edgecolor="white", linewidth=1,
            label=["outside paterson", "paterson"])
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.45)
    for x, lab, dy in ((8, "8 px\none cell\n(arm C line)", 1.40),
                       (12, "12 px\n2σ at σ=6\n(arm A line)", 1.40),
                       (20, "20 px\n2σ at σ=10", 1.40)):
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
        ax.text(x + 0.25, top * dy, lab, va="top", ha="left", fontsize=8, color=INK)
    t = mp["pool_8"]["triggers"]["chebyshev"]
    ax.set_xlabel("separation of a merged pair, heatmap px (Chebyshev, 512x1024 grid)")
    ax.set_ylabel("pairs")
    ax.set_title(f"Merged pairs, 8 splits: {t['under_8px']} of {t['n_pairs']} under 8 px; "
                 f"{t['ge_12px_outside_paterson']} at 12 px or more outside paterson",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p = os.path.join(args.fig_dir, "merged_separation.png")
    fig.savefig(p, dpi=150, metadata={"Software": None})
    plt.close(fig)
    written.append(p)

    with open(args.sim, encoding="utf-8") as f:
        sim = json.load(f)
    fr = sim["threshold_frontier_S7"]
    g = sim["pools"]["S7"]["configs"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([r[2] for r in fr], [r[1] for r in fr], color=MUTED, linewidth=2,
            label="1x alone, threshold lowered from 0.30")
    for t_lab in (0.25, 0.20, 0.15, 0.10):
        r = next(r for r in fr if abs(r[0] - t_lab) < 1e-9)
        ax.annotate(f"{t_lab:.2f}", (r[2], r[1]), textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color=MUTED)
    marks = {"2x-wide": (ORANGE, "o", (8, -3)), "2x-narrow": (ORANGE, "s", (8, -8)),
             "1.5x-wide": (BLUE, "o", (8, 4)), "1.5x-narrow": (BLUE, "s", (-56, -15))}
    for cfg, (col, mk, off) in marks.items():
        e = g[cfg]["fp"]
        ax.plot([e["fps_added"]], [e["ramps_gained"]], marker=mk, markersize=9, color=col,
                markeredgecolor="white", markeredgewidth=2, linestyle="none", label=cfg)
        ax.annotate(cfg, (e["fps_added"], e["ramps_gained"]), textcoords="offset points",
                    xytext=off, fontsize=8, color=INK)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("FPs added vs 1x at 0.30 (S7)")
    ax.set_ylabel("ramps gained vs 1x at 0.30 (S7)")
    ax.set_title("Stage 0b (simulated band, exploratory): each band config buys its ramps\n"
                 "for fewer FPs than lowering the 1x threshold to the same recall",
                 fontsize=10, loc="left", color=INK)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(args.fig_dir, "band_vs_threshold.png")
    fig.savefig(p, dpi=150, metadata={"Software": None})
    plt.close(fig)
    written.append(p)
    for p in written:
        print(f"-> {p}")
    return 0


# --------------------------------------------------------------------------- #
# 0c: extract (GPU)
# --------------------------------------------------------------------------- #
def resolve_splits(s):
    if s == "selection":
        return list(SELECTION)
    if s == "heldout":
        return list(HELD_OUT)
    return [c.strip() for c in s.split(",") if c.strip()]


def load_calibration(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def guard_extract(splits, heldout_flag, cache_root, calibration_path, limit, force,
                  despite_failed_context=False):
    """Refuse anything that would break the held-out rule. Returns the calibration sha256
    for a held-out run (recorded in each held-out cache), else None."""
    unknown = [s for s in splits if s not in SELECTION + HELD_OUT]
    if unknown:
        raise SystemExit(f"extract: {unknown} are not Stage 0c splits ({SELECTION + HELD_OUT})")
    ho = [s for s in splits if s in HELD_OUT]
    if not ho:
        if heldout_flag:
            raise SystemExit("--heldout-read-once given but no held-out split named")
        if limit and os.path.abspath(cache_root) == os.path.abspath(CACHE_0C):
            raise SystemExit("--limit writes truncated caches: pass a scratch --cache-root")
        return None
    if len(ho) != len(splits):
        raise SystemExit("extract: held-out and selection splits must run separately")
    if not heldout_flag:
        raise SystemExit(f"{ho} are held out (read once, at Stage 0c): pass --heldout-read-once")
    if limit or force:
        raise SystemExit("a held-out read takes every pano once: --limit and --force refused")
    cal = load_calibration(calibration_path)
    if cal is None:
        raise SystemExit(f"no {calibration_path}: run calibrate (and commit it) first")
    if cal.get("smoke"):
        raise SystemExit(f"{calibration_path} is a smoke calibration: refused for a held-out read")
    if not cal["context"][PRIMARY_0C]["passes"] and not despite_failed_context:
        raise SystemExit(f"the context rule FAILED for {PRIMARY_0C}: the held-out read is not "
                         "spent on it. Jon decides; --despite-failed-context overrides")
    done = [s for s in ho if os.path.exists(cache_file(cache_root, s))]
    if done:
        raise SystemExit(f"{done}: held-out cache already exists -- these splits were read once")
    return sha256_file(calibration_path)


def cache_file(cache_root, split):
    return os.path.join(cache_root, f"{split}.json")


def passes_for(split, configs):
    """The passes Stage 0c runs on one split: 1x always; the strips; the full-pano pass at
    each strip scale only on S7 (the context rule reads S7 only)."""
    names = ["r2048"]
    scales = sorted({config_parts(c)[0] for c in configs}, key=lambda s: SCALES[s])
    if split in S7:
        names += [f"full@{s}" for s in scales]
    names += [f"strip@{c}" for c in configs]
    return names


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _prefetch(paths, scales):
    """Decode, resize to the 2048x4096 production image (threshold_sweep.PRE's bilinear) and
    upsample (bicubic, as #196's u4096) in a worker thread, one pano ahead of the GPU."""
    from PIL import Image
    from torchvision import transforms
    Image.MAX_IMAGE_PIXELS = None
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    q = Queue(maxsize=1)

    def work():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                native = img.size
                img2048 = transforms.Resize(BASE_HW, interpolation=irs._interp("bilinear"))(img)
                del img
                ts_ = {"r2048": norm(img2048)}
                for s in scales:
                    hw = (int(round(BASE_HW[0] * SCALES[s])), int(round(BASE_HW[1] * SCALES[s])))
                    ts_[s] = norm(transforms.Resize(hw, interpolation=irs._interp("bicubic"))(
                        img2048))
                q.put((p, native, ts_, None))
            except Exception as e:  # noqa: BLE001
                q.put((p, None, None, e))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def usage_rows(stats, wall_s, host, gpus, cities, started, status, held_out):
    common = {"provider": "rampnet", "model_id": "projectsidewalk/rampnet-model",
              "paid": False, "hardware": {"host": host, "gpus": gpus}, "status": status,
              "est_cost_usd": 0.0, "pricing": None, "concurrent_with": [], "gpu_share": 1.0,
              "run_wall_s": round(wall_s, 3), "script": "scripts/analysis/horizon_band_263.py",
              "issue": 263}
    rows = []
    for name, st in stats.items():
        if not st["n"]:
            continue
        rows.append({"ts": started, "bundle": ",".join(cities),
                     "label": f"horizon-band-263:{name}", "panos_scored": st["n"],
                     "elapsed_s": round(st["elapsed_s"], 3),
                     "s_per_pano": round(st["elapsed_s"] / st["n"], 4),
                     "what": f"horizon_band_263.py extract, pass {name}: GPU-side seconds "
                             f"(copy + forward + peak extraction); held_out={held_out}",
                     "run_id": f"horizon-band-263:{name}:{started}", **common})
    if not rows and status != "ok":
        rows.append({"ts": started, "bundle": ",".join(cities),
                     "label": "horizon-band-263:no-pano", "panos_scored": 0,
                     "elapsed_s": round(wall_s, 3), "s_per_pano": None,
                     "what": "horizon_band_263.py extract died before scoring any pano",
                     "run_id": f"horizon-band-263:no-pano:{started}", **common})
    return rows


def cmd_extract(args):
    import torch
    from rampnet import ledger
    from operating_point_curve import bundle_ground_truths
    import threshold_sweep

    splits = resolve_splits(args.splits)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        config_parts(c)
    cal_sha = guard_extract(splits, args.heldout_read_once, args.cache_root, args.calibration,
                            args.limit, args.force, args.despite_failed_context)
    if args.usage_log.lower() == "none" and not args.allow_unrecorded_spend:
        raise SystemExit("--usage-log none drops the GPU-time record; pass "
                         "--allow-unrecorded-spend if that is really intended")
    held_out = cal_sha is not None
    geoms = {c: strip_geometry(CORES[config_parts(c)[1]], SCALES[config_parts(c)[0]])
             for c in configs}

    # preflight: every pano present before the GPU is touched
    plan = {}
    for city in splits:
        if os.path.exists(cache_file(args.cache_root, city)) and not args.force:
            print(f"{city}: cached -> skip", flush=True)
            continue
        gts, _ = bundle_ground_truths(city)
        pids = list(gts)[:args.limit] if args.limit else list(gts)
        pdir = os.path.join(args.panos_root, "benchmark", city, "panos")
        missing = [p for p in pids if not os.path.exists(os.path.join(pdir, f"{p}.jpg"))]
        if missing:
            raise SystemExit(f"{city}: {len(missing)} panos missing under {pdir} "
                             f"(first: {missing[0]})")
        plan[city] = (gts, pids, pdir)
    if not plan:
        print("extract: nothing to do", flush=True)
        return 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = threshold_sweep.load_model().to(device).eval()
    upsample = model.head[2]
    assert isinstance(upsample, torch.nn.Upsample), "head layout changed"
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda" else [])
    host = socket.getfqdn()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.perf_counter()
    print(f"device={device} gpus={gpus} configs={configs} held_out={held_out} "
          f"torch={torch.__version__}", flush=True)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run(t_cpu, hm, md):
        """GPU-side: copy + forward + peaks. Returns (peaks_rc, seconds, peak GiB)."""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.perf_counter()
        upsample.size = tuple(hm)
        with torch.no_grad():
            h = model(t_cpu.unsqueeze(0).to(device)).squeeze().float().cpu().numpy()
        pk = peaks_rc(h, SCORE_FLOOR, md)
        sync()
        dt = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
        return pk, h.shape, dt, mem

    all_stats, attempted, status = {}, [], "failed"
    try:
        for city, (gts, pids, pdir) in plan.items():
            attempted.append(city)
            names = passes_for(city, configs)
            stats = {n: {"elapsed_s": 0.0, "n": 0, "peak_mem_gib": 0.0} for n in names}
            scales = sorted({config_parts(c)[0] for c in configs}, key=lambda s: SCALES[s])
            out = []
            stream = _prefetch([os.path.join(pdir, f"{p}.jpg") for p in pids], scales)
            for i, pid in enumerate(pids, 1):
                path, native, tens, err = next(stream)
                if err is not None:
                    raise SystemExit(f"{path}: {err}")
                passes = {}
                pk, shp, dt, mem = run(tens["r2048"], (512, 1024), BASE_MIN_DISTANCE)
                passes["r2048"] = [(c / shp[1], r / shp[0], s) for r, c, s in pk]
                rec = [("r2048", dt, mem)]
                for s in scales:
                    full = tens[s]
                    if f"full@{s}" in names:
                        hm = (full.shape[1] // HEAD_DIV, full.shape[2] // HEAD_DIV)
                        md = int(round(BASE_MIN_DISTANCE * SCALES[s]))
                        pk, shp, dt, mem = run(full, hm, md)
                        passes[f"full@{s}"] = [(c / shp[1], r / shp[0], sc) for r, c, sc in pk]
                        rec.append((f"full@{s}", dt, mem))
                    for cfg in configs:
                        if config_parts(cfg)[0] != s:
                            continue
                        g = geoms[cfg]
                        strip = make_strip(full, g)
                        assert tuple(strip.shape[1:]) == (g["in_h"], g["in_w"]), strip.shape
                        pk, shp, dt, mem = run(strip, (g["hm_h"], g["hm_w"]), g["min_distance"])
                        passes[f"strip@{cfg}"] = strip_peaks_to_pano(pk, g)
                        rec.append((f"strip@{cfg}", dt, mem))
                for n, dt, mem in rec:
                    stats[n]["elapsed_s"] += dt
                    stats[n]["n"] += 1
                    if mem is not None:
                        stats[n]["peak_mem_gib"] = max(stats[n]["peak_mem_gib"], mem)
                out.append({"pano": pid, "gt": irs._gt_to_json(gts[pid]), "native": list(native),
                            "passes": {n: [[round(x, 6), round(y, 6), round(sc, 6)]
                                           for x, y, sc in passes[n]] for n in names}})
                del tens
                gc.collect()
                if i % 25 == 0:
                    print(f"  {city}: {i}/{len(pids)}", flush=True)
            meta = {"split": city, "held_out": held_out, "calibration_sha256": cal_sha,
                    "limit": args.limit, "configs": configs, "passes": names,
                    "geometry": {c: geoms[c] for c in configs},
                    "timing": {n: {"elapsed_s": round(v["elapsed_s"], 3), "n": v["n"],
                                   "s_per_pano": round(v["elapsed_s"] / v["n"], 4) if v["n"] else None,
                                   "peak_mem_gib": round(v["peak_mem_gib"], 2)}
                               for n, v in stats.items()},
                    "score_floor": SCORE_FLOOR, "base_min_distance": BASE_MIN_DISTANCE,
                    "head_div": HEAD_DIV, "upsample": "PRE bilinear to 2048x4096, then bicubic",
                    "model": "projectsidewalk/rampnet-model", "device": device.type,
                    "gpus": gpus, "host": host, "torch": torch.__version__, "fp16": False,
                    "git_head": git_head(), "started": started}
            os.makedirs(args.cache_root, exist_ok=True)
            with open(cache_file(args.cache_root, city), "w", encoding="utf-8", newline="") as f:
                json.dump({"city": city, "meta": meta, "panos": out}, f)
                f.write("\n")
            for n, v in stats.items():
                a = all_stats.setdefault(n, {"elapsed_s": 0.0, "n": 0})
                a["elapsed_s"] += v["elapsed_s"]
                a["n"] += v["n"]
            print(f"{city} done: " + ", ".join(
                f"{n} {v['elapsed_s'] / max(1, v['n']):.3f} s/pano" for n, v in stats.items())
                + f"; wall {time.perf_counter() - t_start:.0f}s", flush=True)
        status = "ok"
    finally:
        rows = usage_rows(all_stats, time.perf_counter() - t_start, host, gpus,
                          attempted or splits, started, status, held_out)
        if args.usage_log.lower() != "none" and rows:
            ledger.append_rows(args.usage_log, rows)
            print(f"usage_log: +{len(rows)} rows ({status}) -> {args.usage_log}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# 0c: calibrate (CPU)
# --------------------------------------------------------------------------- #
def read_cache_0c(cache_root, split):
    path = cache_file(cache_root, split)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    panos = [{"pano": p["pano"], "gt": irs._gt_from_json(p["gt"]),
              "passes": {k: [tuple(t) for t in v] for k, v in p["passes"].items()}}
             for p in payload["panos"]]
    return panos, payload["meta"], sha256_file(path)


def build_calibration(cache_root, smoke=False):
    rsq = radius_sq_for()
    inputs, context, calib = {}, {}, {}
    present = [s for s in SELECTION if os.path.exists(cache_file(cache_root, s))]
    if not smoke and set(present) != set(SELECTION):
        raise SystemExit(f"calibrate: missing selection caches {sorted(set(SELECTION) - set(present))}")
    data = {}
    for s in present:
        panos, meta, sha = read_cache_0c(cache_root, s)
        if meta.get("limit") and not smoke:
            raise SystemExit(f"{s}: a --limit cache; calibrate needs complete caches")
        data[s], inputs[s] = (panos, meta), sha
    configs = data[present[0]][1]["configs"]
    for cfg in configs:
        scale, core = config_parts(cfg)
        pairs = [([p for p in pd["passes"][f"full@{scale}"] if in_core(p[1], CORES[core])],
                  pd["passes"][f"strip@{cfg}"])
                 for s in S7 if s in data for pd in data[s][0]]
        st = context_stats(pairs, rsq)
        context[cfg] = {k: (rnd(v, 6) if isinstance(v, float) else v) for k, v in st.items()}
        if CALIB_SPLIT not in data:
            continue
        panos = data[CALIB_SPLIT][0]
        shifts = []
        base_core = [{"pano": p["pano"], "gt": p["gt"],
                      "preds": band_preds(p["passes"]["r2048"], CORES[core], T_OP)} for p in panos]
        for p in panos:
            shifts += paired_shifts([q for q in p["passes"]["r2048"] if in_core(q[1], CORES[core])],
                                    p["passes"][f"strip@{cfg}"], rsq)
        c1 = tsc.counts(base_core, rsq)
        p1 = c1["tp"] / (c1["tp"] + c1["fp"]) if c1["tp"] + c1["fp"] else 0.0
        prec = {}
        for t in MATCH_GRID:
            cs = tsc.counts([{"pano": p["pano"], "gt": p["gt"],
                              "preds": band_preds(p["passes"][f"strip@{cfg}"], CORES[core], t)}
                             for p in panos], rsq)
            prec[t] = cs["tp"] / (cs["tp"] + cs["fp"]) if cs["tp"] + cs["fp"] else None
        med = float(np.median(shifts)) if shifts else 0.0
        trig = calibration_rule(med, prec[T_OP] or 0.0, p1)
        t_band = band_threshold(prec, p1) if trig else T_OP
        calib[cfg] = {"n_paired": len(shifts), "median_paired_shift": rnd(med, 6),
                      "precision_1x_in_core_at_030": rnd(p1, 6),
                      "precision_strip_in_core_at_030": rnd(prec[T_OP], 6),
                      "n_1x_in_core_at_030": c1["tp"] + c1["fp"],
                      "threshold_set": trig, "band_threshold": t_band,
                      "strip_precision_by_threshold": {f"{t:.2f}": rnd(v, 6)
                                                       for t, v in prec.items()}}
    return {"protocol": {
        "plan": "detector-phase1-horizon-2x-2026-11-15.md section 4, Stage 0c",
        "context_rule": "S7: >= 95% of full-pass in-core peaks >= 0.30 reproduced by a strip core "
                        "peak within the match radius; |median score shift| < 0.05",
        "calibration_rule": "sao_paulo: band threshold set to equal 1x in-core precision at 0.30 "
                            "if |median paired shift| > 0.03 or |precision diff at 0.30| > 0.05",
        "primary": PRIMARY_0C, "inputs": inputs, "git_head": git_head()},
        "smoke": smoke, "context": context, "calibration": calib}


def cmd_calibrate(args):
    rep = build_calibration(args.cache_root, args.smoke)
    js = json.dumps(rep, indent=1) + "\n"
    if not args.check:
        print(js)
    return check_or_write(((args.out, js),), args.check, "calibration.json")


# --------------------------------------------------------------------------- #
# 0c: report (CPU)
# --------------------------------------------------------------------------- #
def build_gate(cache_root, calibration_path):
    rsq = radius_sq_for()
    cal = load_calibration(calibration_path)
    if cal is None or cal.get("smoke"):
        raise SystemExit("report needs the frozen, non-smoke calibration.json")
    cal_sha = sha256_file(calibration_path)
    data, inputs = {}, {}
    for s in HELD_OUT:
        panos, meta, sha = read_cache_0c(cache_root, s)
        if meta.get("calibration_sha256") != cal_sha:
            raise SystemExit(f"{s}: read under calibration {meta.get('calibration_sha256')}, "
                             f"not the current {cal_sha}")
        data[s], inputs[s] = (panos, meta), sha
    configs = [c for c in CONFIGS_0C if c in cal["calibration"]]

    def variants(split):
        panos = data[split][0]
        v = {}
        for name, f in [("base", lambda p: tsc.thresholded(p["passes"]["r2048"], T_OP))] + [
                (f"base@{t:.2f}", lambda p, t=t: tsc.thresholded(p["passes"]["r2048"], t))
                for t in MATCH_GRID] + [
                (cfg, lambda p, cfg=cfg: band_merge(
                    p["passes"]["r2048"], p["passes"][f"strip@{cfg}"],
                    CORES[config_parts(cfg)[1]], rsq,
                    t_band=cal["calibration"][cfg]["band_threshold"] or 1.01))
                for cfg in configs]:
            v[name] = [{"pano": p["pano"], "gt": p["gt"], "preds": f(p)} for p in panos]
        return v

    var = {}
    for s in HELD_OUT:
        for n, ps in variants(s).items():
            var[(n, s)] = ps
    gold = compare_group((GOLD,), var, rsq, configs)
    rig = compare_group(OPEN_RIG, var, rsq, configs)
    gate = {}
    for cfg in configs:
        bounds = {b: gate_bound(gold["configs"][cfg][b]["_raw"]["vs_base"],
                                gold["configs"][cfg][b]["_raw"]["vs_matched"])
                  for b in ("fp", "ignore")}
        uncal = cal["calibration"][cfg]["band_threshold"] is None
        gate[cfg] = {"bounds": bounds,
                     "verdict": "FAIL" if uncal else gate_verdict(bounds),
                     "band_threshold": cal["calibration"][cfg]["band_threshold"],
                     "context_passes": cal["context"][cfg]["passes"],
                     "open_rig": open_rig_verdict(rig["configs"][cfg]["fp"]["_raw"]["vs_base"]
                                                  ["recall"]),
                     "ramps_gained_manual_gold": gold["configs"][cfg]["fp"]["ramps_gained"]}

    # cost: GPU-side s/pano pooled over every 0c cache present
    tot = {}
    for s in SELECTION + HELD_OUT:
        if not os.path.exists(cache_file(cache_root, s)):
            continue
        with open(cache_file(cache_root, s), encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        for n, t in meta["timing"].items():
            a = tot.setdefault(n, {"elapsed_s": 0.0, "n": 0, "peak_mem_gib": 0.0})
            a["elapsed_s"] += t["elapsed_s"]
            a["n"] += t["n"]
            a["peak_mem_gib"] = max(a["peak_mem_gib"], t["peak_mem_gib"])
    base_s = tot["r2048"]["elapsed_s"] / tot["r2048"]["n"]
    cost = {"base_s_per_pano": rnd(base_s), "limit": COST_LIMIT, "configs": {}}
    for cfg in configs:
        t = tot[f"strip@{cfg}"]
        sp = t["elapsed_s"] / t["n"]
        mult = (base_s + sp) / base_s
        cost["configs"][cfg] = {"strip_s_per_pano": rnd(sp), "multiple": rnd(mult, 3),
                                "within_limit": mult <= COST_LIMIT,
                                "peak_mem_gib": round(t["peak_mem_gib"], 2)}
    two = None
    if "2x-wide" in gate and PRIMARY_0C in gate:
        g15 = gate[PRIMARY_0C]["ramps_gained_manual_gold"]
        g2 = gate["2x-wide"]["ramps_gained_manual_gold"]
        two = {"ramps_gained_2x": g2, "ramps_gained_1.5x": g15,
               "ratio": rnd(g2 / g15, 3) if g15 > 0 else None,
               "gains_enough": g15 > 0 and g2 >= TWO_X_MIN_GAIN_RATIO * g15,
               "within_limit": cost["configs"]["2x-wide"]["within_limit"]}
    primary = gate.get(PRIMARY_0C)
    decision = ("band kept for the shipped weights" if primary and primary["verdict"] == "PASS"
                else "band dropped for the shipped weights; kept only as a read for arm B")
    rep = {"protocol": {"plan": "detector-phase1-horizon-2x-2026-11-15.md section 4, Stage 0c",
                        "primary": PRIMARY_0C, "calibration_sha256": cal_sha, "inputs": inputs,
                        "n_reps": N_REPS, "seed": SEED, "far_ignore_m": FAR_IGNORE_M},
           "gate": gate, "decision": decision, "two_x_rule": two, "cost": cost,
           "manual_gold": gold, "open_rig_pool": rig}
    return json.loads(json.dumps(strip_raw(rep)))


def markdown_gate(rep):
    L = [f"**Decision ({rep['protocol']['primary']} is binding): {rep['decision']}**\n"]
    L.append("| config | context | band threshold | gate, FP bound | gate, ignore bound | verdict "
             "| open rig | cost multiple | within 1.75x |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg, g in rep["gate"].items():
        c = rep["cost"]["configs"][cfg]
        L.append(f"| {cfg} | {'pass' if g['context_passes'] else 'FAIL'} | {g['band_threshold']} | "
                 f"{'pass' if g['bounds']['fp']['passes'] else 'fail'} | "
                 f"{'pass' if g['bounds']['ignore']['passes'] else 'fail'} | {g['verdict']} | "
                 f"{g['open_rig']} | {c['multiple']:.3f} | {c['within_limit']} |")
    L.append("")
    for name, grp in (("manual_gold", rep["manual_gold"]), ("juneau + bayonne", rep["open_rig_pool"])):
        for b in ("fp", "ignore"):
            L.append(f"**{name}, bound {b}**\n")
            L.append("| config | F1 | ramps gained | FPs added | ΔR | ΔF1 | matched thr | "
                     "band − matched ΔF1 |")
            L.append("|---|---|---|---|---|---|---|---|")
            m = grp["base"][b]
            L.append(f"| 1x@0.30 | {m['F1']:.4f} | — | — | — | — | — | — |")
            for cfg, e in grp["configs"].items():
                e = e[b]
                mt = e["matched_recall_threshold"]
                L.append(f"| {cfg} | {e['metrics']['F1']:.4f} | {e['ramps_gained']:+d} | "
                         f"{e['fps_added']:+d} | {_d(e['vs_base']['recall'])} | "
                         f"{_d(e['vs_base']['f1'])} | "
                         f"{'unreachable' if mt is None else format(mt['threshold'], '.2f')} | "
                         f"{'—' if mt is None else _d(mt['merged_minus_matched']['f1'])} |")
            L.append("")
    if rep["two_x_rule"]:
        t = rep["two_x_rule"]
        L.append(f"2x rule: gains {t['ramps_gained_2x']} vs 1.5x {t['ramps_gained_1.5x']} "
                 f"(ratio {t['ratio']}, needs >= 1.25: {t['gains_enough']}); within the limit: "
                 f"{t['within_limit']}.\n")
    return "\n".join(L)


def cmd_report(args):
    rep = build_gate(args.cache_root, args.calibration)
    js = json.dumps(rep, indent=1) + "\n"
    md = markdown_gate(rep) + "\n"
    if not args.check:
        print(md)
    return check_or_write(((args.out, js), (os.path.splitext(args.out)[0] + ".md", md)),
                          args.check, "gate_0c.json and gate_0c.md")


def cmd_run0c(args):
    base = ["--cache-root", args.cache_root, "--panos-root", args.panos_root,
            "--usage-log", args.usage_log, "--device", args.device]
    rc = main(["extract", "--splits", "selection"] + base)
    if rc:
        return rc
    rc = main(["calibrate", "--cache-root", args.cache_root, "--out", args.calibration])
    if rc:
        return rc
    cal = load_calibration(args.calibration)
    if not cal["context"][PRIMARY_0C]["passes"]:
        print(f"STOP: the context rule failed for {PRIMARY_0C}; the held-out splits were NOT "
              "read. Commit calibration.json and hand the decision to Jon.", flush=True)
        return 4
    print("calibration.json frozen; commit it now if you can (the held-out caches record its "
          "sha256). Reading the held-out splits once.", flush=True)
    rc = main(["extract", "--splits", "heldout", "--heldout-read-once",
               "--calibration", args.calibration] + base)
    if rc:
        return rc
    return main(["report", "--cache-root", args.cache_root, "--calibration", args.calibration])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="Stage 0b (CPU, exploratory)")
    s.add_argument("--cache-root", default=irs.CACHE_ROOT)
    s.add_argument("--out", default=SIM_JSON)
    s.add_argument("--check", action="store_true")

    f = sub.add_parser("figures", help="redraw the 0a and 0b figures")
    f.add_argument("--sim", default=SIM_JSON)
    f.add_argument("--merged", default=MERGED_JSON)
    f.add_argument("--fig-dir", default=FIG_DIR)

    def common_0c(p, panos=True):
        p.add_argument("--cache-root", default=CACHE_0C)
        if panos:
            p.add_argument("--panos-root", default=REPO,
                           help="a RampNet checkout holding benchmark/<split>/panos/")
            p.add_argument("--usage-log", default=USAGE_LOG)

    e = sub.add_parser("extract", help="Stage 0c GPU passes")
    common_0c(e)
    e.add_argument("--splits", default="selection",
                   help="'selection' (S7 + sao_paulo), 'heldout', or a comma list")
    e.add_argument("--configs", default=",".join(CONFIGS_0C))
    e.add_argument("--calibration", default=CALIBRATION)
    e.add_argument("--heldout-read-once", action="store_true")
    e.add_argument("--despite-failed-context", action="store_true")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--force", action="store_true")
    e.add_argument("--allow-unrecorded-spend", action="store_true")
    e.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help="'cpu' for a smoke test on a machine whose GPU is shared")

    c = sub.add_parser("calibrate", help="Stage 0c context rule + band threshold (CPU)")
    common_0c(c, panos=False)
    c.add_argument("--out", default=CALIBRATION)
    c.add_argument("--smoke", action="store_true",
                   help="accept partial/--limit caches; the output can never gate a held-out read")
    c.add_argument("--check", action="store_true")

    r = sub.add_parser("report", help="Stage 0c gate (CPU)")
    common_0c(r, panos=False)
    r.add_argument("--calibration", default=CALIBRATION)
    r.add_argument("--out", default=GATE_JSON)
    r.add_argument("--check", action="store_true")

    z = sub.add_parser("run0c", help="the whole of Stage 0c, in order")
    common_0c(z)
    z.add_argument("--calibration", default=CALIBRATION)
    z.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))

    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    return {"simulate": cmd_simulate, "figures": cmd_figures, "extract": cmd_extract,
            "calibrate": cmd_calibrate, "report": cmd_report, "run0c": cmd_run0c}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
