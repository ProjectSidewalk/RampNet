"""What does a gated cascade cost? (#35, fills the blank #126 left)

#126 (``cascade_gate.py``) showed the cascade is *live*: on richmond at RampNet's
recommended 0.30 operating point, 19 of the 38 ramps the Vistas parity arm recovers and
RampNet misses have a RampNet floor peak in radius scoring 0.05-0.30 -- something a
challenger-gated threshold could promote. It stated a ceiling (+6.1 recall points,
0.829 -> 0.890) and left the false-positive cost blank. This script builds the gate and
scores it, so the cost is a measurement.

The rule, per pano and per setting ``(T_lo, r_gate, c_min)``::

    kept     = RampNet floor peaks with score >= T_hi           (the operating point)
    cands    = challenger boxes with score >= c_min (or no score at all)
    promoted = floor peaks with T_lo <= score < T_hi that lie within r_gate of a cand
    preds    = kept + promoted     (each keeps its OWN RampNet score)

scored with the benchmark's own scorer (``score_pano`` at radius 0.022, wrapped at the
seam, highest score first) and ``aggregate`` (precision over every pano, recall over the
``fn_confirmed`` panos). Because every kept peak outscores every promoted one and the
matcher is greedy in score order, promotion can never change what a kept peak matched:
``promoted_tp`` / ``promoted_fp`` (the net change against the baseline) are exactly the
promoted peaks' own outcomes.

Controls, same scorer, same panos, same run:

* ``baseline`` -- kept only. On richmond at 0.30 it must reproduce 257 / 28 / 53
  (``analysis_out/op/corrected_at_0.3.csv``); the tests pin it.
* ``threshold_only[T_lo]`` -- every floor peak >= T_lo, no gate. This is what the gate
  must beat: if lowering the threshold everywhere buys the same F1, the challenger is
  doing nothing.
* ``naive_union`` -- kept + every challenger box. Reported under two conventions that
  differ in how a ramp hit by both models is counted. ``complementarity.py``'s (the
  published 0.549) scores each model's list on its own and adds the two FP bills with
  no dedup, and takes TP as the oracle union of the ramps either model matched.
  ``aggregate``'s scores ONE merged list through ``score_pano``, so when a RampNet peak
  and a challenger box land on the same ramp the second hit is an FP, and greedy
  matching on the merged list can hand a box to a neighbouring ramp. On richmond x the
  parity arm that is 692 FP against 470 (+222: 229 hits that were TPs when each list is
  scored alone become 222 FPs and 7 ignored) and 302 TP against 295 (+7 by
  reassignment). Both conventions use the same panos there: all 124 richmond panos are
  ``fn_confirmed``. Each carries ``fp_per_recovered_ramp`` (extra FP over extra TP
  against ``baseline``).

**Null (deterministic).** For shift k = 1..n-1 over the sorted pano list, pano i gets
the challenger boxes of pano (i + k) mod n -- the same boxes, density and clustering, on
the wrong pano -- and the cascade is re-scored at every setting. ``attributable_dR`` is
the real recall gain minus the mean shifted gain. This is ``complementarity``'s and
``null_recall.py``'s construction. ``--null random`` draws a random subset of those
cyclic shifts (seeded) instead of evenly spaced ones; it is still a cyclic-shift null,
not a random-position one (no mode here scatters boxes uniformly, which would destroy
the challenger's density and clustering).

**Calibration (``calibration``).** The same shifted challengers are also run through
the WHOLE rule as if each were real: a wrong-pano challenger at shift k takes the other
shifts plus the true alignment as its null, gets its own attributable ΔR on every
setting, and gets a verdict from ``verdict_of``. The fraction reading VIABLE is the
rule's false-VIABLE rate for that pair, max-over-grid selection included. With every
shift (``--null-shifts all``) this is exact: the shifts of a shifted challenger are the
original's shifts. With a subset it is the same construction over the subset.

**Bootstrap.** ``bootstrap_vs_baseline`` is the pano-resampled interval for the chosen
settings, *conditional on the in-sample selection* (the setting is picked once, on all
panos, and held fixed), on raw ΔR and ΔF1. ``bootstrap_selection_aware`` repeats the
selection inside every resample: the whole rule (null mean, controls, viability,
best-by-(F1, attributable ΔR)) is re-applied to the resampled panos; when no setting is
viable in a resample the cascade is not deployed and its Δ is 0. It also reports how
often each verdict comes out.

**Volatile fields.** Wall-clock and host are not in the per-pair JSON, so a re-run is
byte-identical to the committed file (``--check`` proves it). They go to
``runs.json`` in the same directory, keyed by file name.

**Why the op_cache and not the bundle.** The bundle records are the shipped point --
on richmond every one scores >= 0.5519 -- so they contain nothing below the threshold
to promote. ``analysis_out/op_cache/<split>.json`` holds ``peak_local_max`` output down
to the 0.05 floor.

**Caveat that travels with every number.** The committed op_caches were written at
``c7098be`` (2026-07-28), before the seam fix ``f4c71c8``; they can lack peaks in a
~3.5 degree strip beside the seam (``docs/seam.md``). The cascade can only promote peaks
the cache lists, so a seam-side gain is under-stated, never inflated. Regenerating the
op_caches needs a GPU and the native-resolution panoramas and is out of scope here.

Inputs, all committed: ``benchmark/<split>/`` (GT), ``analysis_out/op_cache/<split>.json``
(RampNet floor peaks), ``benchmark/model_detections/<leg>__<split>.json`` (challenger).
CPU only; no network, no ``.model_cache``.

Usage::

    python scripts/analysis/cascade_cost_35.py --split richmond \\
        --challenger mask2former-vistas-curb-cut-1024x1024
    python scripts/analysis/cascade_cost_35.py --split richmond \\
        --challenger mask2former-vistas-curb-cut-1024x1024 --t-hi 0.5519 \\
        --t-lo 0.05 0.10 0.15 0.20 0.25 0.30 0.40
    python scripts/analysis/cascade_cost_35.py --all-published --null-shifts 20
    python scripts/analysis/cascade_cost_35.py --summary
    python scripts/analysis/cascade_cost_35.py --check      # regenerate + byte-compare
"""
import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet import roster                                             # noqa: E402
from rampnet.detection_eval import (                                   # noqa: E402
    build_ground_truth, score_pano, aggregate, radius_sq_for,
    PANO_SCALE_X, PANO_SCALE_Y)
from rampnet.geometry import dist_sq                                   # noqa: E402
from compare import load_bundle, load_manual_ground_truths             # noqa: E402
from complementarity import load_floor_peaks, matched_gt               # noqa: E402
from export_model_cache import load_detections                         # noqa: E402

DEFAULT_OUT = os.path.join(OUT, "cascade_cost_35")
T_HI = 0.30
T_LO = (0.05, 0.10, 0.15, 0.20, 0.25)
R_GATE = (0.011, 0.022, 0.044)
RADIUS = 0.022
PRIMARY = ("richmond", "mask2former-vistas-curb-cut-1024x1024")
#: The pre-stated bar (plan section 1, #35): attributable recall gain after the null.
MIN_ATTRIBUTABLE_DR = 0.020
#: One setting read on EVERY pair, chosen AFTER the primary richmond x vistas-1024 run
#: (it is that pair's best viable row). Not part of the pre-stated rule: it exists so the
#: in-sample choice on richmond can be looked at on data it was not chosen on.
#: ``c_min`` is a rank ("q50" = the challenger's own median box score on that split),
#: so it transfers across challengers whose scores live on different scales.
FIXED_SETTING = (0.05, 0.011, "q50")
#: Fine threshold-only sweep: the strongest no-challenger control, not just the one at T_lo.
THR_SWEEP = tuple(round(0.05 + 0.01 * i, 2) for i in range(91))
BOOTSTRAP = 2000
OP_CACHE_NOTE = ("op_cache written at c7098be (2026-07-28), before the seam fix f4c71c8: "
                 "pre-f4c71c8 seam strip, peaks beside the seam can be missing, so any "
                 "seam-side gain is under-stated")
CEILING_ARTIFACT = os.path.join(REPO, "analysis_out", "cascade_gate_op030.json")


# --------------------------------------------------------------------------- #
# pure rule
# --------------------------------------------------------------------------- #
def _score(p):
    return p[2] if len(p) > 2 else None


def threshold_preds(peaks, t):
    """Every floor peak scoring >= ``t``."""
    return [p for p in peaks if p[2] >= t]


def filter_cands(cands, c_min):
    """Challenger boxes at or above ``c_min``; a box with no score always passes."""
    return [c for c in cands if _score(c) is None or _score(c) >= c_min]


def cascade_preds(peaks, cands, t_hi, t_lo, r_gate_sq):
    """``(preds, n_promoted)``: kept peaks plus sub-threshold peaks gated by ``cands``.

    ``cands`` are already filtered by ``c_min``. Distances are wrapped at the seam in
    the benchmark's anisotropic pano space, exactly as ``score_pano`` measures them.
    """
    kept, promoted = [], []
    for p in peaks:
        s = p[2]
        if s >= t_hi:
            kept.append(p)
        elif s >= t_lo and any(
                dist_sq(p[0], p[1], c[0], c[1], PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
                < r_gate_sq for c in cands):
            promoted.append(p)
    return kept + promoted, len(promoted)


def union_preds(kept, cands):
    """Naive union: every kept peak and every challenger box, as one prediction list."""
    return list(kept) + [tuple(c) for c in cands]


def score_split(pred_fn, gts, radius_sq):
    """``{tp, fp, fn, n_gt, P, R, F1}`` over every pano in ``gts`` via ``aggregate``."""
    rep = aggregate([score_pano(pred_fn(pid), gt, radius_sq) for pid, gt in gts.items()])
    return _report(rep)


def _report(rep):
    return {"tp": rep.tp, "fp": rep.fp, "fn": rep.fn, "n_gt": rep.n_gt_recall,
            "P": rep.precision, "R": rep.recall, "F1": rep.f1}


def shifted(cands_by_pano, pids, k):
    """Pano ``pids[i]`` gets pano ``pids[(i + k) % n]``'s boxes (the cyclic null)."""
    n = len(pids)
    return {pids[i]: cands_by_pano.get(pids[(i + k) % n], []) for i in range(n)}


def c_min_grid(cands_by_pano):
    """``[0.0, q25, q50, q75]`` of the challenger's box scores, or ``[0.0]`` if score-less.

    Linear-interpolated quartiles, rounded to 6 dp (the rounded value is the one used,
    so a re-run reproduces the committed rows exactly).
    """
    scores = sorted(_score(c) for cs in cands_by_pano.values() for c in cs
                    if _score(c) is not None)
    if not scores:
        return [0.0]

    def q(f):
        pos = f * (len(scores) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(scores) - 1)
        return round(scores[lo] + (scores[hi] - scores[lo]) * (pos - lo), 6)
    return [0.0, q(0.25), q(0.5), q(0.75)]


def verdict_of(baseline, threshold_only, grid, min_dr=MIN_ATTRIBUTABLE_DR):
    """The pre-stated decision rule (plan section 1), as code.

    VIABLE      some row has attributable_dR >= min_dr, F1 >= baseline F1, and F1 above
                threshold-only at the same T_lo.
    NOT VIABLE  no row reaches attributable_dR >= min_dr, or every row that does is
                matched (F1 <=) by threshold-only at its T_lo.
    PARTIAL     otherwise: recall rises by >= min_dr and beats threshold-only, but F1
                falls below baseline -- an operating-point dial, not a free lunch.
    """
    scored = [r for r in grid if r.get("attributable_dR") is not None]
    if not scored:
        return None                     # no null was run (--null none): no verdict
    raising = [r for r in scored if r["attributable_dR"] >= min_dr]
    if not raising:
        return "NOT VIABLE"
    beats = [r for r in raising
             if r["F1"] > threshold_only[_tkey(r["t_lo"])]["F1"]]
    if any(r["F1"] >= baseline["F1"] for r in beats):
        return "VIABLE"
    if not beats:
        return "NOT VIABLE"
    return "PARTIAL"


def _tkey(t):
    return f"{t:g}"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_gts(split):
    """``{pano_id: GroundTruth}``, sorted by pano id, for either bundle kind."""
    bdir = os.path.join(REPO, "benchmark", split)
    records, verdicts, _ = load_bundle(bdir)
    if verdicts is None:
        gts = load_manual_ground_truths(bdir)
    else:
        gts = {pid: build_ground_truth(records[pid]["detections"], e["dets"],
                                       e["missed"], e["no_missed"])
               for pid, e in verdicts.items()}
    return dict(sorted(gts.items()))


def op_cache_meta(split):
    with open(os.path.join(REPO, "analysis_out", "op_cache", f"{split}.json"),
              encoding="utf-8") as f:
        return json.load(f).get("meta", {})


def load_challenger(split, published):
    """``(cands_by_pano, header)`` from the published file, or ``(None, None)``."""
    leg = roster.BY_PUBLISHED[published]
    path = os.path.join(REPO, "benchmark", "model_detections",
                        roster.published_filename(leg, split))
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    dets = load_detections(leg.label, split, publish_as=published)
    header = {k: v for k, v in payload.items() if k != "detections"}
    return {pid: [tuple(c) for c in cs] for pid, cs in dets.items()}, header


def ceiling_for(split, t_hi, published=PRIMARY[1]):
    """#126's promotable-ramp ceiling (``cascade_gate_op030.json``): richmond, T_hi 0.30,
    and only for the challenger that artifact was built from (the Vistas parity arm)."""
    if (split != "richmond" or published != PRIMARY[1] or abs(t_hi - 0.30) > 1e-9
            or not os.path.exists(CEILING_ARTIFACT)):
        return None
    with open(CEILING_ARTIFACT, encoding="utf-8") as f:
        art = json.load(f)
    c_only = [s for s in art["sites"] if s["cell"] == "challenger_only"]
    promotable = sum(1 for s in c_only if s["peak_in_radius"]
                     and s["nearest_peak_score"] is not None
                     and s["nearest_peak_score"] < art["rampnet_op_threshold"])
    handoff = sum(1 for s in c_only if s["peak_in_radius"]) - promotable
    return {"promotable": promotable, "handoff_ge_t_hi": handoff,
            "_promotable_sites": [(s["pano"], s["x"], s["y"]) for s in c_only
                                  if s["peak_in_radius"]
                                  and s["nearest_peak_score"] is not None
                                  and s["nearest_peak_score"] < art["rampnet_op_threshold"]],
            "no_peak_in_radius": len(c_only) - promotable - handoff,
            "challenger_only": len(c_only), "n_gt": art["n_sites"],
            "promotable_dR": round(promotable / art["n_sites"], 6),
            "challenger": art["challenger"],
            "vistas_input_size": art.get("vistas_input_size"),
            "source": "analysis_out/cascade_gate_op030.json"}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
class _Scorer:
    """Memoised per-pano scoring: many (setting, shift) pairs give the same pred set."""

    def __init__(self, gts, peaks, radius_sq):
        self.gts, self.peaks, self.radius_sq = gts, peaks, radius_sq
        self.memo = {}

    def pano(self, pid, idx):
        key = (pid, idx)
        s = self.memo.get(key)
        if s is None:
            pk = self.peaks.get(pid, [])
            s = score_pano([pk[i] for i in idx], self.gts[pid], self.radius_sq)
            self.memo[key] = s
        return s


def _gate_indices(peaks, cands, t_hi, t_lo, r_gate_sq):
    idx = []
    for i, p in enumerate(peaks):
        if p[2] >= t_hi:
            idx.append(i)
        elif p[2] >= t_lo and any(
                dist_sq(p[0], p[1], c[0], c[1], PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
                < r_gate_sq for c in cands):
            idx.append(i)
    return tuple(idx)


def _pano_counts(score):
    rc = score.fn_confirmed
    return (score.tp, score.fp, score.tp if rc else 0, score.n_gt if rc else 0)


def _f1(tp, fp, tpr, ngt):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tpr / ngt if ngt else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), r


def bootstrap_delta(scorer, idx_a, idx_b, n=BOOTSTRAP, seed=0):
    """Pano-resampled 95% interval for (F1_b - F1_a, R_b - R_a), paired on panos.

    Resamples panos with replacement (a pano's ramps and false positives move
    together) and recomputes ``aggregate``'s P/R/F1 from summed counts. Deterministic.
    """
    pids = list(scorer.gts)
    a = [_pano_counts(scorer.pano(pid, idx_a[pid])) for pid in pids]
    b = [_pano_counts(scorer.pano(pid, idx_b[pid])) for pid in pids]
    rng = random.Random(seed)
    dF, dR = [], []
    m = len(pids)
    for _ in range(n):
        sa, sb = [0, 0, 0, 0], [0, 0, 0, 0]
        for _j in range(m):
            i = rng.randrange(m)
            for t in range(4):
                sa[t] += a[i][t]
                sb[t] += b[i][t]
        fa, ra = _f1(*sa)
        fb, rb = _f1(*sb)
        dF.append(fb - fa)
        dR.append(rb - ra)
    dF.sort()
    dR.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return {"dF1_ci95": [dF[lo], dF[hi]], "dR_ci95": [dR[lo], dR[hi]],
            "resamples": n, "seed": seed, "unit": "pano"}


def _setting_indices(scorer, cands_by_pano, t_hi, t_lo, r_gate, c_min):
    rsq = radius_sq_for(r_gate)
    return {pid: _gate_indices(scorer.peaks.get(pid, []),
                               filter_cands(cands_by_pano.get(pid, []), c_min),
                               t_hi, t_lo, rsq) for pid in scorer.gts}


def _fast_report(scores):
    """``_report(aggregate(scores))`` without the AP/PR-curve work (same formulas, same
    floats: P = tp/(tp+fp), R = recall tp/recall n_gt, F1 = 2PR/(P+R))."""
    tp = sum(x.tp for x in scores)
    fp = sum(x.fp for x in scores)
    tpr = sum(x.tp for x in scores if x.fn_confirmed)
    ngt = sum(x.n_gt for x in scores if x.fn_confirmed)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tpr / ngt if ngt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": ngt - tpr, "n_gt": ngt, "P": p, "R": r, "F1": f}


def _eval_cascade(scorer, filtered, t_hi, t_lo, r_gate_sq, n_kept_by_pano):
    """``(report, n_promoted, per-pano scores)`` for one setting; ``filtered`` is the
    challenger map already filtered by ``c_min`` (real or shifted)."""
    scores, n_prom = [], 0
    for pid in scorer.gts:
        idx = _gate_indices(scorer.peaks.get(pid, []), filtered.get(pid, []), t_hi, t_lo,
                            r_gate_sq)
        n_prom += len(idx) - n_kept_by_pano[pid]
        scores.append(scorer.pano(pid, idx))
    return _fast_report(scores), n_prom, scores


def _counts(scores):
    """``[m, 4]`` int64: ``(tp, fp, recall tp, recall n_gt)`` per pano (``aggregate``'s)."""
    return np.array([_pano_counts(s) for s in scores], dtype=np.int64).reshape(-1, 4)


def _prf(c):
    """P, R, F1 arrays from summed ``(tp, fp, recall tp, recall n_gt)`` along the last axis."""
    tp, fp, tpr, ngt = (c[..., i].astype(np.float64) for i in range(4))
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tpr, ngt, out=np.zeros_like(tpr), where=ngt > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return p, r, f


def select_under_weights(W, base_c, thr_c, grid_c, null_tpr_sum, n_null, tlo_index,
                         min_dr=MIN_ATTRIBUTABLE_DR):
    """Apply the whole pre-stated rule to pano-weighted counts, one row of ``W`` at a time.

    ``W`` is ``[B, m]`` pano multiplicities (a bootstrap resample, or all ones for the
    in-sample run); ``base_c`` ``[m, 4]``, ``thr_c`` ``[T, m, 4]`` and ``grid_c``
    ``[S, m, 4]`` are per-pano counts; ``null_tpr_sum`` ``[S, m]`` is the recall-TP
    summed over the ``n_null`` shifted challengers, so the null mean recall of a
    weighted set is ``(W @ null_tpr_sum.T) / (n_null * n_gt)``. Integer arithmetic up to
    the final divisions, so the result is identical on every platform.

    Returns a dict of ``[B]`` arrays: ``viable`` (any viable row), ``sel`` (the
    best-by-(F1, attributable dR) viable row, first on ties, as ``max`` picks it),
    ``dF1`` / ``dR`` / ``attr_dR`` of that row against the weighted baseline (0 where
    nothing is viable: the cascade is not deployed), and ``verdict`` codes (0 VIABLE,
    1 PARTIAL, 2 NOT VIABLE, resolved exactly as ``verdict_of``).
    """
    B = W @ base_c                                    # [B, 4]
    G = np.einsum("bm,smk->bsk", W, grid_c)           # [B, S, 4]
    T = np.einsum("bm,tmk->btk", W, thr_c)            # [B, T, 4]
    NT = W @ null_tpr_sum.T                           # [B, S]
    _, rb, fb = _prf(B)
    _, rg, fg = _prf(G)
    _, _, ft = _prf(T)
    ngt = B[:, 3].astype(np.float64)
    rnull = np.divide(NT.astype(np.float64), (n_null * ngt)[:, None],
                      out=np.zeros(NT.shape), where=ngt[:, None] > 0)
    att = (rg - rb[:, None]) - (rnull - rb[:, None])
    ft_row = ft[:, tlo_index]                         # threshold-only F1 at each row's T_lo
    raising = att >= min_dr
    beats = raising & (fg > ft_row)
    viable_rows = beats & (fg >= fb[:, None])
    anyv = viable_rows.any(axis=1)
    fmax = np.where(viable_rows, fg, -np.inf).max(axis=1)
    tie = viable_rows & (fg == fmax[:, None])
    sel = np.argmax(np.where(tie, att, -np.inf), axis=1)
    rows = np.arange(W.shape[0])
    verdict = np.where(anyv, 0, np.where(~raising.any(axis=1) | ~beats.any(axis=1), 2, 1))
    return {"viable": anyv, "sel": np.where(anyv, sel, -1),
            "dF1": np.where(anyv, fg[rows, sel] - fb, 0.0),
            "dR": np.where(anyv, rg[rows, sel] - rb, 0.0),
            "attr_dR": np.where(anyv, att[rows, sel], 0.0),
            "verdict": verdict}


def _ci95(v):
    s = sorted(float(x) for x in v)
    n = len(s)
    return [s[int(0.025 * n)], s[int(0.975 * n) - 1]]


def bootstrap_selection_aware(base_c, thr_c, grid_c, null_tpr_sum, n_null, tlo_index,
                              chosen, verdict, n=BOOTSTRAP, seed=0):
    """Pano bootstrap with the selection repeated inside every resample (#35 review item 4).

    ``chosen`` is the grid index the in-sample run picked (or ``None``) and ``verdict``
    its verdict. Before resampling, the vectorised rule is run once with every pano
    weighted 1 and must reproduce both -- so the numpy rule cannot drift from
    ``verdict_of`` and the ``max`` selection unnoticed. Deterministic:
    ``numpy.random.default_rng(seed)``.
    """
    m = base_c.shape[0]
    ins = select_under_weights(np.ones((1, m), dtype=np.int64), base_c, thr_c, grid_c,
                               null_tpr_sum, n_null, tlo_index)
    names = ("VIABLE", "PARTIAL", "NOT VIABLE")
    assert int(ins["sel"][0]) == (-1 if chosen is None else chosen), (ins["sel"], chosen)
    assert names[int(ins["verdict"][0])] == verdict, (ins["verdict"], verdict)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    W = np.stack([np.bincount(r, minlength=m) for r in idx]).astype(np.int64)
    out = select_under_weights(W, base_c, thr_c, grid_c, null_tpr_sum, n_null, tlo_index)
    return {"resamples": n, "seed": seed, "unit": "pano", "rng": "numpy.default_rng",
            "viable_frac": float(out["viable"].mean()),
            "verdict_counts": {names[i]: int((out["verdict"] == i).sum()) for i in range(3)},
            "same_setting_frac": (float((out["sel"] == chosen).mean())
                                  if chosen is not None else None),
            "dF1_ci95": _ci95(out["dF1"]), "dR_ci95": _ci95(out["dR"]),
            "attr_dR_ci95": _ci95(out["attr_dR"]),
            "given_viable": ({"dF1_ci95": _ci95(out["dF1"][out["viable"]]),
                              "dR_ci95": _ci95(out["dR"][out["viable"]]),
                              "attr_dR_ci95": _ci95(out["attr_dR"][out["viable"]])}
                             if out["viable"].any() else None),
            "note": ("the rule re-applied in every resample (null mean, controls, "
                     "viability, best by (F1, attributable dR)); Delta = 0 in a resample "
                     "with no viable setting, where the cascade is not deployed")}


def calibrate(base, thr, grid, null_dR, null_F1, ks, n_panos):
    """The verdict rule applied to wrong-pano challengers (#35 review item 5).

    ``null_dR[s][i]`` / ``null_F1[s][i]`` are grid row ``s`` scored with the challenger
    shifted by ``ks[i]``. Wrong challenger ``i`` takes the true alignment plus every other
    evaluated shift as its null (exactly the shifts of a shifted challenger when ``ks`` is
    every k), gets attributable dR per row, and a verdict from ``verdict_of``.
    """
    if not ks:
        return None
    verdicts, max_att = [], []
    for i in range(len(ks)):
        rows = []
        for s, r in enumerate(grid):
            pool = [r["dR"]] + [null_dR[s][j] for j in range(len(ks)) if j != i]
            rows.append({"t_lo": r["t_lo"], "F1": null_F1[s][i],
                         "attributable_dR": null_dR[s][i] - sum(pool) / len(pool)})
        verdicts.append(verdict_of(base, thr, rows))
        max_att.append(max(x["attributable_dR"] for x in rows))
    real_max = max(r["attributable_dR"] for r in grid)
    srt = sorted(max_att)
    return {"challengers": len(ks),
            "exact": len(ks) == n_panos - 1,
            "verdicts": {v: verdicts.count(v) for v in ("VIABLE", "PARTIAL", "NOT VIABLE")},
            "false_viable_rate": verdicts.count("VIABLE") / len(ks),
            "max_attr_dR_median": srt[len(srt) // 2], "max_attr_dR_max": srt[-1],
            "real_max_attr_dR": real_max,
            "wrong_at_or_above_real": sum(1 for v in max_att if v >= real_max),
            "construction": ("each shifted challenger run through the whole rule, its null "
                             "= the true alignment plus the other evaluated shifts")}


#: Coordinate tolerance for matching a gained GT ramp to a #126 promotable site (the
#: ceiling artifact stores the same GT coordinates, possibly rounded).
SITE_TOL = 1e-4


def run_pair(split, published, t_hi=T_HI, t_lo=T_LO, r_gate=R_GATE, c_min="auto",
             null="shift", null_shifts="all", seed=0, radius=RADIUS, gts=None,
             peaks=None, bootstrap=BOOTSTRAP):
    """Score the cascade grid, its controls and its null for one (split, leg).

    Returns ``(payload, run)``: ``payload`` is fully determined by the committed inputs
    and the arguments (it is what gets committed and byte-compared); ``run`` holds the
    volatile wall-clock and host.
    """
    t0 = time.time()
    gts = load_gts(split) if gts is None else gts
    peaks = load_floor_peaks(split) if peaks is None else peaks
    cands_by_pano, header = load_challenger(split, published)
    if cands_by_pano is None:
        return None, None
    pids = list(gts)
    radius_sq = radius_sq_for(radius)
    scorer = _Scorer(gts, peaks, radius_sq)
    kept_idx = {pid: tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t_hi)
                for pid in pids}
    n_kept = {pid: len(v) for pid, v in kept_idx.items()}

    base_scores = [scorer.pano(pid, kept_idx[pid]) for pid in pids]
    base = _report(aggregate(base_scores))
    thr, thr_scores = {}, []
    for t in t_lo:
        sc = [scorer.pano(pid, tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t))
              for pid in pids]
        thr[_tkey(t)] = _report(aggregate(sc))
        thr_scores.append(sc)

    # naive union, both conventions
    u_scores = [score_pano(union_preds([peaks.get(pid, [])[i] for i in kept_idx[pid]],
                                       cands_by_pano.get(pid, [])), gts[pid], radius_sq)
                for pid in pids]
    union_agg = _report(aggregate(u_scores))
    u_tp = u_n = r_fp = c_fp = 0
    for pid in pids:
        gt = gts[pid]
        if not gt.fn_confirmed:
            continue
        rp = [peaks.get(pid, [])[i] for i in kept_idx[pid]]
        cp = cands_by_pano.get(pid, [])
        u_tp += len(matched_gt(rp, gt.gt_points, radius_sq)
                    | matched_gt(cp, gt.gt_points, radius_sq))
        u_n += len(gt.gt_points)
        r_fp += score_pano(rp, gt, radius_sq).fp
        c_fp += score_pano(cp, gt, radius_sq).fp
    up = u_tp / (u_tp + r_fp + c_fp) if u_tp + r_fp + c_fp else 0.0
    ur = u_tp / u_n if u_n else 0.0
    union_comp = {"tp": u_tp, "fp": r_fp + c_fp, "fn": u_n - u_tp, "n_gt": u_n,
                  "P": up, "R": ur, "F1": 2 * up * ur / (up + ur) if up + ur else 0.0,
                  "rampnet_fp": r_fp, "challenger_fp": c_fp}
    for u in (union_agg, union_comp):
        d_tp = u["tp"] - base["tp"]
        u["fp_per_recovered_ramp"] = (u["fp"] - base["fp"]) / d_tp if d_tp > 0 else None

    cmins = c_min_grid(cands_by_pano) if c_min == "auto" else [float(v) for v in c_min]
    settings = [(tl, rg, cm) for tl in t_lo for rg in r_gate for cm in cmins]

    # null shift list
    n = len(pids)
    if null == "none" or n < 2:
        ks = []
    elif null == "random":
        rng = random.Random(seed)
        ks = sorted(rng.sample(range(1, n), min(n - 1, 20 if null_shifts == "all"
                                                 else int(null_shifts))))
    elif null_shifts == "all" or int(null_shifts) >= n - 1:
        ks = list(range(1, n))
    else:
        m = int(null_shifts)
        ks = sorted({max(1, min(n - 1, round(1 + j * (n - 2) / max(1, m - 1))))
                     for j in range(m)})
    null_sets = [shifted(cands_by_pano, pids, k) for k in ks]
    # c_min filtering done once per c_min; a shifted map reuses the filtered lists
    filt = {cm: {pid: filter_cands(cs, cm) for pid, cs in cands_by_pano.items()}
            for cm in cmins}
    filt_null = {cm: [shifted(filt[cm], pids, k) for k in ks] for cm in cmins}

    grid, grid_c, null_tpr_sum, null_dR, null_F1 = [], [], [], [], []
    for tl, rg, cm in settings:
        rsq = radius_sq_for(rg)
        rep, n_prom, sc = _eval_cascade(scorer, filt[cm], t_hi, tl, rsq, n_kept)
        grid_c.append(_counts(sc))
        row = {"t_lo": tl, "r_gate": rg, "c_min": cm, **rep, "n_promoted": n_prom,
               "promoted_tp": rep["tp"] - base["tp"], "promoted_fp": rep["fp"] - base["fp"],
               "promoted_ignored": n_prom - (rep["tp"] - base["tp"]) - (rep["fp"] - base["fp"]),
               "dR": rep["R"] - base["R"], "dP": rep["P"] - base["P"],
               "dF1": rep["F1"] - base["F1"],
               "threshold_only_F1": thr[_tkey(tl)]["F1"]}
        tpr_sum = np.zeros(len(pids), dtype=np.int64)
        if null_sets:
            dRs, dFPs, F1s = [], [], []
            for cs in filt_null[cm]:
                nrep, _, nsc = _eval_cascade(scorer, cs, t_hi, tl, rsq, n_kept)
                dRs.append(nrep["R"] - base["R"])
                dFPs.append(nrep["fp"] - base["fp"])
                F1s.append(nrep["F1"])
                tpr_sum += _counts(nsc)[:, 2]
            row["null_dR_mean"] = sum(dRs) / len(dRs)
            row["null_dR_max"] = max(dRs)
            row["null_dFP_mean"] = sum(dFPs) / len(dFPs)
            null_dR.append(dRs)
            null_F1.append(F1s)
        else:
            row["null_dR_mean"] = row["null_dR_max"] = row["null_dFP_mean"] = None
        null_tpr_sum.append(tpr_sum)
        row["attributable_dR"] = (row["dR"] - row["null_dR_mean"]
                                  if row["null_dR_mean"] is not None else None)
        att_ramps = (row["attributable_dR"] * base["n_gt"]
                     if row["attributable_dR"] is not None else None)
        row["fp_per_attributable_ramp"] = (row["promoted_fp"] / att_ramps
                                           if att_ramps and att_ramps > 0 else None)
        grid.append(row)

    # Selected rows are COPIES, so annotating them never touches the grid rows (#35
    # review item 9: which grid rows carried the annotation used to depend on the null).
    best_f1_i = max(range(len(grid)), key=lambda i: (grid[i]["F1"], grid[i]["R"],
                                                     -grid[i]["fp"]))
    best_f1 = dict(grid[best_f1_i])
    viable_i = [i for i, r in enumerate(grid) if r["attributable_dR"] is not None
                and r["attributable_dR"] >= MIN_ATTRIBUTABLE_DR
                and r["F1"] >= base["F1"] and r["F1"] > r["threshold_only_F1"]]
    viable = [dict(grid[i]) for i in viable_i]
    best_viable_i = (max(viable_i, key=lambda i: (grid[i]["F1"], grid[i]["attributable_dR"]))
                     if viable_i else None)
    best_viable = dict(grid[best_viable_i]) if best_viable_i is not None else None

    # the strongest no-challenger control: the best single threshold anywhere
    sweep = {}
    for t in THR_SWEEP:
        sweep[t] = _report(aggregate([scorer.pano(
            pid, tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t))
            for pid in pids]))
    t_best = max(sweep, key=lambda t: (sweep[t]["F1"], -t))
    thr_best = {"t": t_best, **sweep[t_best], "sweep": [0.05, 0.95, 0.01]}

    # the fixed post-hoc setting, read on every pair
    ft, fr, fc = FIXED_SETTING
    fc_val = (cmins[2] if len(cmins) == 4 else 0.0) if fc == "q50" else float(fc)
    fixed = next((r for r in grid if r["t_lo"] == ft and r["r_gate"] == fr
                  and r["c_min"] == fc_val), None)
    fixed = dict(fixed, c_min_rule=fc) if fixed else None

    def matched_recall(r):
        """Best threshold-only point reaching at least this row's recall: what the same
        recall costs WITHOUT the challenger."""
        ok = [t for t in sweep if sweep[t]["R"] >= r["R"]]
        if not ok:
            return None
        t = max(ok, key=lambda t: (sweep[t]["F1"], t))
        return {"t": t, **sweep[t]}

    for r in [best_f1, best_viable, fixed] + viable:
        if r is not None:
            r["threshold_only_at_matched_recall"] = matched_recall(r)

    boot = {}
    for lab, r in (("best_by_f1", best_f1), ("best_viable", best_viable), ("fixed", fixed)):
        if r is not None and bootstrap:
            boot[lab] = bootstrap_delta(
                scorer, kept_idx,
                _setting_indices(scorer, cands_by_pano, t_hi, r["t_lo"], r["r_gate"],
                                 r["c_min"]), n=bootstrap, seed=seed)
            boot[lab]["conditioning"] = ("conditional on the in-sample selection: the "
                                         "setting is held fixed across resamples")
            boot[lab]["dR_kind"] = "raw (not attributable)"
    verdict = verdict_of(base, thr, grid) if null_sets else None
    sel_boot = None
    if null_sets and bootstrap:
        tl_pos = {t: i for i, t in enumerate(t_lo)}
        sel_boot = bootstrap_selection_aware(
            _counts(base_scores), np.stack([_counts(s) for s in thr_scores]),
            np.stack(grid_c), np.stack(null_tpr_sum), len(null_sets),
            np.array([tl_pos[r["t_lo"]] for r in grid]), best_viable_i, verdict,
            n=bootstrap, seed=seed)
    at_p = [r for r in grid if r["P"] >= base["P"] and r["R"] > base["R"]]
    best_r = dict(max(at_p, key=lambda r: (r["R"], r["F1"]))) if at_p else None
    calib = (calibrate(base, thr, grid, null_dR, null_F1, ks, n) if null_sets else None)

    ceiling = ceiling_for(split, t_hi, published)
    if ceiling is not None:
        sites = ceiling.pop("_promotable_sites")
        for lab, r in (("best_viable", best_viable), ("best_by_f1", best_f1)):
            if r is None:
                continue
            idx = _setting_indices(scorer, cands_by_pano, t_hi, r["t_lo"], r["r_gate"],
                                   r["c_min"])
            gained = []
            for pid in pids:
                gt = gts[pid]
                if not gt.fn_confirmed:
                    continue
                pk = peaks.get(pid, [])
                before = matched_gt([pk[i] for i in kept_idx[pid]], gt.gt_points, radius_sq)
                after = matched_gt([pk[i] for i in idx[pid]], gt.gt_points, radius_sq)
                gained += [(pid, gt.gt_points[g][0], gt.gt_points[g][1])
                           for g in after - before]
            in_prom = sum(1 for g in gained if any(
                g[0] == s[0] and abs(g[1] - s[1]) <= SITE_TOL and abs(g[2] - s[2]) <= SITE_TOL
                for s in sites))
            assert in_prom <= len(gained)
            ceiling[f"{lab}_gained_ramps"] = len(gained)
            ceiling[f"{lab}_gained_in_promotable"] = in_prom

    missing_ch = [pid for pid in pids if pid not in cands_by_pano]
    missing_op = [pid for pid in pids if pid not in peaks]
    payload = {
        "split": split, "challenger": published, "challenger_signature": header,
        "t_hi": t_hi, "radius": radius, "op_cache_meta": op_cache_meta(split),
        "op_cache_commit_note": OP_CACHE_NOTE, "n_panos": n,
        "n_gt_recall": base["n_gt"],
        "args": {"t_lo": list(t_lo), "r_gate": list(r_gate), "c_min": c_min,
                 "null": null, "null_shifts": null_shifts, "seed": seed,
                 "bootstrap": bootstrap},
        "panos_missing_challenger": missing_ch, "panos_missing_op_cache": missing_op,
        "null": {"mode": null, "shifts": ks if null == "random" else len(ks),
                 "seed": seed if null == "random" else None},
        "baseline": base, "threshold_only": thr,
        "naive_union": {"aggregate": union_agg, "complementarity": union_comp},
        "c_min_grid": cmins, "grid": grid, "best_by_f1": best_f1,
        "best_recall_at_precision_ge_baseline": best_r,
        "viable_rows": viable, "best_viable": best_viable,
        "threshold_only_best": thr_best, "fixed_setting": fixed,
        "fixed_setting_rule": {"t_lo": ft, "r_gate": fr, "c_min": fc,
                               "chosen": "post hoc, from the primary pair's best viable row"},
        "bootstrap_vs_baseline": boot,
        "bootstrap_selection_aware": sel_boot,
        "verdict": verdict, "min_attributable_dR": MIN_ATTRIBUTABLE_DR,
        "calibration": calib,
        "ceiling": ceiling,
    }
    run = {"elapsed_s": round(time.time() - t0, 1), "host": platform.node(),
           "python": platform.python_version()}
    return payload, run


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _round(o, nd=6):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def out_name(split, published, t_hi):
    stem = f"{split}__{roster.slug(published)}"
    if abs(t_hi - T_HI) > 1e-9:
        stem += f"__thi{t_hi:g}"
    return stem + ".json"


def dumps(payload):
    """The committed serialisation: floats rounded to 6 dp, sorted keys, LF, trailing LF."""
    return json.dumps(_round(payload), indent=1, sort_keys=True) + "\n"


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(dumps(payload))


RUNS = "runs.json"


def record_run(out_dir, name, run):
    """Wall-clock and host for ``name`` in ``<out_dir>/runs.json`` -- kept out of the
    per-pair file so that file is byte-identical on every re-run."""
    path = os.path.join(out_dir, RUNS)
    runs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)
    runs[name] = run
    write_json(path, runs)


def pair_files(out_dir):
    """Every per-pair JSON in ``out_dir`` (not ``summary.json`` / ``runs.json``)."""
    return sorted(n for n in os.listdir(out_dir)
                  if n.endswith(".json") and n not in ("summary.json", RUNS))


def check(out_dir, names=None):
    """Regenerate each committed per-pair file from its recorded ``args`` and compare
    bytes. Returns the list of names that differ (empty = every file reproduced)."""
    bad, cache = [], {}
    for name in names or pair_files(out_dir):
        with open(os.path.join(out_dir, name), encoding="utf-8", newline="") as f:
            text = f.read()
        old = json.loads(text)
        a = old["args"]
        split = old["split"]
        if split not in cache:
            cache = {split: (load_gts(split), load_floor_peaks(split))}
        gts, peaks = cache[split]
        t0 = time.time()
        p, _ = run_pair(split, old["challenger"], t_hi=old["t_hi"], t_lo=tuple(a["t_lo"]),
                        r_gate=tuple(a["r_gate"]), c_min=a["c_min"], null=a["null"],
                        null_shifts=a["null_shifts"], seed=a["seed"],
                        bootstrap=a["bootstrap"], radius=old["radius"], gts=gts, peaks=peaks)
        same = p is not None and dumps(p) == text
        print(f"{'same' if same else 'DIFF'}  {name}  ({time.time() - t0:.0f}s)", flush=True)
        if not same:
            bad.append(name)
    return bad


def splits_with_op_cache():
    d = os.path.join(REPO, "analysis_out", "op_cache")
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json"))


def all_benchmark_splits():
    d = os.path.join(REPO, "benchmark")
    return sorted(n for n in os.listdir(d)
                  if os.path.exists(os.path.join(d, n, "records.jsonl")))


def summary(out_dir, out_path=None, quiet=False):
    """``summary.json`` over every per-pair file in ``out_dir`` (written to ``out_path``,
    default ``<out_dir>/summary.json``), plus markdown to stdout unless ``quiet``.

    A pair run with ``--null none`` has no attributable dR and no verdict; it is listed
    under ``gaps`` rather than counted.
    """
    rows, gaps = [], []
    have_op = set(splits_with_op_cache())
    for split in all_benchmark_splits():
        if split not in have_op:
            gaps.append({"split": split, "reason": "no analysis_out/op_cache/<split>.json"})
            continue
        for leg in roster.PUBLISHED:
            name = roster.published_name(leg)
            path = os.path.join(out_dir, out_name(split, name, T_HI))
            if not os.path.exists(os.path.join(REPO, "benchmark", "model_detections",
                                               roster.published_filename(leg, split))):
                continue
            if not os.path.exists(path):
                gaps.append({"split": split, "challenger": name,
                             "reason": "published detections exist but this pair was not run"})
                continue
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
            if p["verdict"] is None:
                gaps.append({"split": split, "challenger": name,
                             "reason": "run without a null (--null none): no verdict"})
                continue
            b, bf, bv, fx = (p["baseline"], p["best_by_f1"], p["best_viable"],
                             p["fixed_setting"])
            tb = p["threshold_only_best"]
            row = {
                "split": split, "challenger": name, "n_gt": p["n_gt_recall"],
                "null_shifts": p["null"]["shifts"], "verdict": p["verdict"],
                "base_F1": b["F1"], "base_R": b["R"],
                "thr_best_t": tb["t"], "thr_best_F1": tb["F1"],
                "union_F1": p["naive_union"]["aggregate"]["F1"],
                "best_F1": bf["F1"], "best_F1_setting": [bf["t_lo"], bf["r_gate"], bf["c_min"]],
                "best_F1_dR": bf["dR"], "best_F1_attr_dR": bf["attributable_dR"],
                "max_attr_dR": max(r["attributable_dR"] for r in p["grid"]
                                   if r["attributable_dR"] is not None),
                "n_viable_rows": len(p["viable_rows"]),
                "calib_false_viable_rate": (p["calibration"] or {}).get("false_viable_rate"),
                "calib_challengers": (p["calibration"] or {}).get("challengers"),
                "selection_aware": None if p["bootstrap_selection_aware"] is None else {
                    k: p["bootstrap_selection_aware"][k]
                    for k in ("viable_frac", "dF1_ci95", "dR_ci95", "attr_dR_ci95")},
            }
            for lab, r in (("viable", bv), ("fixed", fx)):
                row[lab] = None if r is None else {
                    "setting": [r["t_lo"], r["r_gate"], r["c_min"]], "F1": r["F1"],
                    "dF1": r["dF1"], "dR": r["dR"], "attr_dR": r["attributable_dR"],
                    "promoted_fp": r["promoted_fp"],
                    "fp_per_attr_ramp": r["fp_per_attributable_ramp"],
                    "thr_at_matched_R_F1": (r["threshold_only_at_matched_recall"] or {}).get("F1"),
                    "boot": p["bootstrap_vs_baseline"].get(
                        "best_viable" if lab == "viable" else "fixed")}
            rows.append(row)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    rates = [r["calib_false_viable_rate"] for r in rows
             if r["calib_false_viable_rate"] is not None]
    calib = {
        "pairs": len(rates),
        "expected_false_viable": sum(rates),
        "pairs_with_any_false_viable": sum(1 for x in rates if x > 0),
        "max_false_viable_rate": max(rates) if rates else None,
        "note": ("sum over pairs of the per-pair false-VIABLE rate of wrong-pano "
                 "challengers: the number of VIABLE verdicts expected if no challenger "
                 "were aligned with the panos. Per-pair rates are not family-wise "
                 "calibrated; this sum is the family-wise read."),
    }
    out = {"rows": rows, "gaps": gaps, "verdict_counts": counts, "calibration": calib,
           "fixed_setting_rule": {"t_lo": FIXED_SETTING[0], "r_gate": FIXED_SETTING[1],
                                  "c_min": FIXED_SETTING[2]}}
    write_json(out_path or os.path.join(out_dir, "summary.json"), out)
    if quiet:
        return out

    def f(v, fmt):
        return "-" if v is None else format(v, fmt)
    print("| split | challenger | shifts | verdict | base F1 | best thr-only F1 (t) "
          "| best viable (T_lo, r, c_min) | F1 | attr dR | FP / attr ramp | union F1 |")
    print("|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        v = r["viable"] or {}
        st = v.get("setting")
        sts = "-" if not st else f"{st[0]:g}, {st[1]:g}, {st[2]:.3g}"
        print(f"| {r['split']} | {r['challenger']} | {r['null_shifts']} | {r['verdict']} | "
              f"{r['base_F1']:.4f} | {r['thr_best_F1']:.4f} ({r['thr_best_t']:g}) | {sts} | "
              f"{f(v.get('F1'), '.4f')} | {f(v.get('attr_dR'), '+.4f')} | "
              f"{f(v.get('fp_per_attr_ramp'), '.2f')} | {r['union_F1']:.4f} |")
    print()
    print(f"Fixed setting {FIXED_SETTING} on every pair (post hoc):" + chr(10))
    print("| split | challenger | base F1 | fixed F1 | dF1 [95% CI] | dR | attr dR "
          "| promoted FP | FP / attr ramp | thr-only F1 at matched R |")
    print("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        x = r["fixed"]
        if x is None:
            continue
        ci = (x["boot"] or {}).get("dF1_ci95")
        cis = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"| {r['split']} | {r['challenger']} | {r['base_F1']:.4f} | {x['F1']:.4f} | "
              f"{x['dF1']:+.4f}{cis} | {x['dR']:+.4f} | {f(x['attr_dR'], '+.4f')} | "
              f"{x['promoted_fp']} | {f(x['fp_per_attr_ramp'], '.2f')} | "
              f"{f(x['thr_at_matched_R_F1'], '.4f')} |")
    print(chr(10) + f"verdicts: {counts}")
    print(f"calibration: {calib['pairs']} pairs, expected false VIABLE "
          f"{calib['expected_false_viable']:.2f}, pairs with any "
          f"{calib['pairs_with_any_false_viable']}, max rate {calib['max_false_viable_rate']}")
    print("\ngaps:")
    for g in gaps:
        print(" ", g)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=PRIMARY[0])
    ap.add_argument("--challenger", default=PRIMARY[1],
                    help="Published leg name (the model_detections filename stem).")
    ap.add_argument("--t-hi", type=float, default=T_HI)
    ap.add_argument("--t-lo", type=float, nargs="+", default=list(T_LO))
    ap.add_argument("--r-gate", type=float, nargs="+", default=list(R_GATE))
    ap.add_argument("--c-min", nargs="+", default=["auto"],
                    help="'auto' (0 and the challenger's score quartiles) or explicit values.")
    ap.add_argument("--null", choices=["shift", "random", "none"], default="shift",
                    help="'shift': evenly spaced cyclic shifts (default); 'random': a "
                         "seeded random subset of cyclic shifts (still a shift null, not "
                         "random box positions); 'none': no null, no verdict.")
    ap.add_argument("--null-shifts", default="all",
                    help="'all' (every k = 1..n-1) or N evenly spaced shifts.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random shift subset (--null random) and both bootstraps.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--all-published", action="store_true",
                    help="Every (split, leg) with published detections and an op_cache.")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="Regenerate every per-pair file in --out from its recorded args "
                         "and byte-compare; exit 1 on any difference. Writes nothing.")
    args = ap.parse_args()

    if args.summary:
        summary(args.out)
        return
    if args.check:
        bad = check(args.out)
        print(f"{len(bad)} differ" + ("" if not bad else ": " + ", ".join(bad)))
        sys.exit(1 if bad else 0)
    c_min = "auto" if args.c_min == ["auto"] else args.c_min
    kw = dict(t_hi=args.t_hi, t_lo=tuple(args.t_lo), r_gate=tuple(args.r_gate),
              c_min=c_min, null=args.null, null_shifts=args.null_shifts, seed=args.seed)

    if args.all_published:
        have_op = set(splits_with_op_cache())
        splits = args.splits or all_benchmark_splits()
        for split in splits:
            if split not in have_op:
                print(f"skip {split}: no op_cache")
                continue
            gts, peaks = load_gts(split), load_floor_peaks(split)
            for leg in roster.PUBLISHED:
                name = roster.published_name(leg)
                path = os.path.join(args.out, out_name(split, name, args.t_hi))
                if args.skip_existing and os.path.exists(path):
                    continue
                p, run = run_pair(split, name, gts=gts, peaks=peaks, **kw)
                if p is None:
                    continue
                write_json(path, p)
                record_run(args.out, os.path.basename(path), run)
                print(f"{split:20s} {name:40s} {p['verdict']}  "
                      f"best F1 {p['best_by_f1']['F1']:.4f} (base {p['baseline']['F1']:.4f})"
                      f"  {run['elapsed_s']:.0f}s", flush=True)
        return

    if args.challenger not in roster.BY_PUBLISHED:
        sys.exit(f"unknown published leg {args.challenger!r}; choose from:\n  "
                 + "\n  ".join(sorted(roster.BY_PUBLISHED)))
    p, run = run_pair(args.split, args.challenger, **kw)
    if p is None:
        sys.exit(f"no published detections for {args.challenger} on {args.split}")
    path = os.path.join(args.out, out_name(args.split, args.challenger, args.t_hi))
    write_json(path, p)
    record_run(args.out, os.path.basename(path), run)
    b = p["baseline"]
    print(f"{args.split} x {args.challenger} @T_hi {args.t_hi:g}: baseline "
          f"{b['tp']}/{b['fp']}/{b['fn']} P {b['P']:.4f} R {b['R']:.4f} F1 {b['F1']:.4f}")
    for k, r in p["threshold_only"].items():
        print(f"  threshold-only >= {k}: {r['tp']}/{r['fp']}/{r['fn']} "
              f"P {r['P']:.4f} R {r['R']:.4f} F1 {r['F1']:.4f}")
    for conv, u in p["naive_union"].items():
        print(f"  naive union ({conv}): {u['tp']}/{u['fp']}/{u['fn']} "
              f"P {u['P']:.4f} R {u['R']:.4f} F1 {u['F1']:.4f}")
    for lab, r in (("best by F1", p["best_by_f1"]),
                   ("best R at P>=base", p["best_recall_at_precision_ge_baseline"])):
        if r:
            print(f"  {lab}: T_lo {r['t_lo']:g} r {r['r_gate']:g} c_min {r['c_min']:.3g} "
                  f"{r['tp']}/{r['fp']}/{r['fn']} P {r['P']:.4f} R {r['R']:.4f} "
                  f"F1 {r['F1']:.4f} dR {r['dR']:+.4f} attr {r['attributable_dR']} "
                  f"thr-only F1 {r['threshold_only_F1']:.4f}")
    sa = p["bootstrap_selection_aware"]
    if sa:
        print(f"  selection-aware bootstrap: viable in {sa['viable_frac']:.3f} of resamples, "
              f"dF1 {sa['dF1_ci95']}, dR {sa['dR_ci95']}, attr dR {sa['attr_dR_ci95']}")
    c = p["calibration"]
    if c:
        print(f"  calibration: {c['challengers']} wrong-pano challengers, verdicts "
              f"{c['verdicts']}, max attr dR {c['max_attr_dR_max']:.4f} "
              f"(real {c['real_max_attr_dR']:.4f})")
    print(f"  verdict: {p['verdict']}   ({run['elapsed_s']:.0f}s on {run['host']})  -> {path}")


if __name__ == "__main__":
    main()
