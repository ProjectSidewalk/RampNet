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
* ``naive_union`` -- kept + every challenger box. Reported under two conventions:
  ``aggregate``'s (one prediction list through ``score_pano``, all panos, a second hit
  on a ramp is an FP) and ``complementarity.py``'s (recall-eligible panos only, oracle
  union TP, the two models' FP bills added with no dedup), which is the published 0.549.

**Null (deterministic).** For shift k = 1..n-1 over the sorted pano list, pano i gets
the challenger boxes of pano (i + k) mod n -- the same boxes, density and clustering, on
the wrong pano -- and the cascade is re-scored at every setting. ``attributable_dR`` is
the real recall gain minus the mean shifted gain. This is ``complementarity``'s and
``null_recall.py``'s construction; a random-position null is available
(``--null random``) but is not the default, because a uniform draw destroys the
challenger's real density and clustering.

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
"""
import argparse
import json
import os
import platform
import random
import sys
import time

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
    raising = [r for r in grid if r["attributable_dR"] >= min_dr]
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


def _eval_cascade(scorer, cands_by_pano, t_hi, t_lo, r_gate_sq, c_min, n_kept_by_pano):
    scores, n_prom = [], 0
    for pid in scorer.gts:
        peaks = scorer.peaks.get(pid, [])
        cands = filter_cands(cands_by_pano.get(pid, []), c_min)
        idx = _gate_indices(peaks, cands, t_hi, t_lo, r_gate_sq)
        n_prom += len(idx) - n_kept_by_pano[pid]
        scores.append(scorer.pano(pid, idx))
    return _report(aggregate(scores)), n_prom


def run_pair(split, published, t_hi=T_HI, t_lo=T_LO, r_gate=R_GATE, c_min="auto",
             null="shift", null_shifts="all", seed=0, radius=RADIUS, gts=None,
             peaks=None, bootstrap=BOOTSTRAP):
    """Score the cascade grid, its controls and its null for one (split, leg)."""
    t0 = time.time()
    gts = load_gts(split) if gts is None else gts
    peaks = load_floor_peaks(split) if peaks is None else peaks
    cands_by_pano, header = load_challenger(split, published)
    if cands_by_pano is None:
        return None
    pids = list(gts)
    radius_sq = radius_sq_for(radius)
    scorer = _Scorer(gts, peaks, radius_sq)
    kept_idx = {pid: tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t_hi)
                for pid in pids}
    n_kept = {pid: len(v) for pid, v in kept_idx.items()}

    base = _report(aggregate([scorer.pano(pid, kept_idx[pid]) for pid in pids]))
    thr = {}
    for t in t_lo:
        thr[_tkey(t)] = _report(aggregate([scorer.pano(
            pid, tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t))
            for pid in pids]))

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

    grid = []
    for tl, rg, cm in settings:
        rsq = radius_sq_for(rg)
        rep, n_prom = _eval_cascade(scorer, cands_by_pano, t_hi, tl, rsq, cm, n_kept)
        row = {"t_lo": tl, "r_gate": rg, "c_min": cm, **rep, "n_promoted": n_prom,
               "promoted_tp": rep["tp"] - base["tp"], "promoted_fp": rep["fp"] - base["fp"],
               "promoted_ignored": n_prom - (rep["tp"] - base["tp"]) - (rep["fp"] - base["fp"]),
               "dR": rep["R"] - base["R"], "dP": rep["P"] - base["P"],
               "dF1": rep["F1"] - base["F1"],
               "threshold_only_F1": thr[_tkey(tl)]["F1"]}
        if null_sets:
            dRs, dFPs = [], []
            for cs in null_sets:
                nrep, _ = _eval_cascade(scorer, cs, t_hi, tl, rsq, cm, n_kept)
                dRs.append(nrep["R"] - base["R"])
                dFPs.append(nrep["fp"] - base["fp"])
            row["null_dR_mean"] = sum(dRs) / len(dRs)
            row["null_dR_max"] = max(dRs)
            row["null_dFP_mean"] = sum(dFPs) / len(dFPs)
        else:
            row["null_dR_mean"] = row["null_dR_max"] = row["null_dFP_mean"] = None
        row["attributable_dR"] = (row["dR"] - row["null_dR_mean"]
                                  if row["null_dR_mean"] is not None else None)
        att_ramps = (row["attributable_dR"] * base["n_gt"]
                     if row["attributable_dR"] is not None else None)
        row["fp_per_attributable_ramp"] = (row["promoted_fp"] / att_ramps
                                           if att_ramps and att_ramps > 0 else None)
        grid.append(row)

    best_f1 = max(grid, key=lambda r: (r["F1"], r["R"], -r["fp"]))
    viable = [r for r in grid if r["attributable_dR"] is not None
              and r["attributable_dR"] >= MIN_ATTRIBUTABLE_DR
              and r["F1"] >= base["F1"] and r["F1"] > r["threshold_only_F1"]]
    best_viable = max(viable, key=lambda r: (r["F1"], r["attributable_dR"])) if viable else None

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

    for r in [best_f1, best_viable, fixed] + list(viable):
        if r is not None:
            r["threshold_only_at_matched_recall"] = matched_recall(r)

    boot = {}
    for lab, r in (("best_by_f1", best_f1), ("best_viable", best_viable), ("fixed", fixed)):
        if r is not None and bootstrap:
            boot[lab] = bootstrap_delta(
                scorer, kept_idx,
                _setting_indices(scorer, cands_by_pano, t_hi, r["t_lo"], r["r_gate"],
                                 r["c_min"]), n=bootstrap, seed=seed)
    at_p = [r for r in grid if r["P"] >= base["P"] and r["R"] > base["R"]]
    best_r = max(at_p, key=lambda r: (r["R"], r["F1"])) if at_p else None
    verdict = (verdict_of(base, thr, grid) if null_sets else None)

    ceiling = ceiling_for(split, t_hi, published)
    if ceiling is not None:
        sites = set(tuple(x) for x in ceiling.pop("_promotable_sites"))
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
            ceiling[f"{lab}_gained_ramps"] = len(gained)
            ceiling[f"{lab}_gained_in_promotable"] = sum(1 for g in gained if g in sites)

    missing_ch = [pid for pid in pids if pid not in cands_by_pano]
    missing_op = [pid for pid in pids if pid not in peaks]
    payload = {
        "split": split, "challenger": published, "challenger_signature": header,
        "t_hi": t_hi, "radius": radius, "op_cache_meta": op_cache_meta(split),
        "op_cache_commit_note": OP_CACHE_NOTE, "n_panos": n,
        "n_gt_recall": base["n_gt"],
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
        "verdict": verdict, "min_attributable_dR": MIN_ATTRIBUTABLE_DR,
        "ceiling": ceiling,
        "elapsed_s": time.time() - t0, "host": platform.node(),
    }
    return payload


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


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(_round(payload), indent=1, sort_keys=True) + "\n")


def splits_with_op_cache():
    d = os.path.join(REPO, "analysis_out", "op_cache")
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json"))


def all_benchmark_splits():
    d = os.path.join(REPO, "benchmark")
    return sorted(n for n in os.listdir(d)
                  if os.path.exists(os.path.join(d, n, "records.jsonl")))


def summary(out_dir):
    """``summary.json`` over every per-pair file in ``out_dir``, plus markdown to stdout."""
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
                "max_attr_dR": max(r["attributable_dR"] for r in p["grid"]),
                "n_viable_rows": len(p["viable_rows"]),
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
    out = {"rows": rows, "gaps": gaps, "verdict_counts": counts,
           "fixed_setting_rule": {"t_lo": FIXED_SETTING[0], "r_gate": FIXED_SETTING[1],
                                  "c_min": FIXED_SETTING[2]}}
    write_json(os.path.join(out_dir, "summary.json"), out)

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
    ap.add_argument("--null", choices=["shift", "random", "none"], default="shift")
    ap.add_argument("--null-shifts", default="all",
                    help="'all' (every k = 1..n-1) or N evenly spaced shifts.")
    ap.add_argument("--seed", type=int, default=0, help="Only for --null random.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--all-published", action="store_true",
                    help="Every (split, leg) with published detections and an op_cache.")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summary(args.out)
        return
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
                p = run_pair(split, name, gts=gts, peaks=peaks, **kw)
                if p is None:
                    continue
                write_json(path, p)
                print(f"{split:20s} {name:40s} {p['verdict']}  "
                      f"best F1 {p['best_by_f1']['F1']:.4f} (base {p['baseline']['F1']:.4f})"
                      f"  {p['elapsed_s']:.0f}s")
        return

    if args.challenger not in roster.BY_PUBLISHED:
        sys.exit(f"unknown published leg {args.challenger!r}; choose from:\n  "
                 + "\n  ".join(sorted(roster.BY_PUBLISHED)))
    p = run_pair(args.split, args.challenger, **kw)
    if p is None:
        sys.exit(f"no published detections for {args.challenger} on {args.split}")
    path = os.path.join(args.out, out_name(args.split, args.challenger, args.t_hi))
    write_json(path, p)
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
    print(f"  verdict: {p['verdict']}   ({p['elapsed_s']:.0f}s on {p['host']})  -> {path}")


if __name__ == "__main__":
    main()
