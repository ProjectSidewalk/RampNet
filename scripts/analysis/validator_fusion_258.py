"""Step 3 of #258: fuse RampNet's score with the validator's, fitted on sao_paulo only.

The posted design (#258 comment 6316627866, step 3): fuse the two scores rather than
hard-gating, fit on sao_paulo only (the dev split, per the #51 convention), and report
recall at matched precision against the threshold-only control of
``cascade_cost_35.py``. Every other split is held out: nothing below is fitted on it.

**The fusion, fixed before any held-out split was read.** A two-feature logistic
regression, ``P(tp) = sigmoid(b0 + b1 * logit(s) + b2 * logit(v))``, with ``s`` RampNet's
peak score and ``v`` the validator's P(correct), fitted by unpenalized maximum likelihood
(Newton/IRLS) on sao_paulo's TP/FP candidates (``ignored`` dropped, redundant FPs kept).
One sentence of justification: it is the smallest rule that weighs the two scores
against each other with fitted weights, three parameters fitted on 462 rows, and it
needs no search (no regularization path, no feature choice, no model family choice).
Both scores are clipped to [1e-6, 1 - 1e-6] before the logit.

**Scoring.** Every rule keeps a subset of each pano's committed op_cache floor peaks
(``analysis_out/op_cache/<split>.json``, score >= 0.05), and the kept set is re-scored
with the benchmark's own scorer, ``score_pano`` at the cache's radius with the seam
wrap, each peak keeping its OWN RampNet score as its match order (as in
``cascade_cost_35.py``). Precision is over every pano and recall over the
``fn_confirmed`` panos, ``aggregate``'s convention, through ``cascade_cost_35``'s
``_pano_counts``/``_prf``. Re-scoring matters: a fused rule can drop a high-scored peak
and keep a lower one near the same ramp, which the per-candidate labels of step 1 (valid
only for score-order prefixes) would mis-count.

**Matched precision, the cascade_cost_35 definition.** ``cascade_cost_35.py`` reads a
cascade against its baseline as ``best_recall_at_precision_ge_baseline``: among the
settings whose precision is at least the baseline's (RampNet at the 0.30 operating
point), the one with the highest recall (ties by F1). This applies that definition
exactly, to both arms, on the same split:

* ``threshold_only``: RampNet alone, every threshold in ``cascade_cost_35.THR_SWEEP``
  (0.05 to 0.95 by 0.01); its best recall at P >= P(RampNet@0.30).
* ``fused``: the fused probability, thresholds 0.005 to 0.995 by 0.005; its best
  recall at the same precision bar.

The difference is ``fused - threshold_only`` recall at matched precision. Reading both
curves at a precision bar on the scored split uses that split's labels for the threshold
of both arms alike, so it compares the two rankings at one operating precision; the
fused weights themselves come from sao_paulo.

**Fully held out (``deployed``).** The fused threshold is also fixed on sao_paulo: the
threshold that reaches the best recall at P >= P(RampNet@0.30) there. Applied unchanged
to every other split, it is compared with RampNet@0.30 on P, R and F1. This is the
number a deployment would get without touching the target split.

**Intervals.** 95% percentile intervals from a pano bootstrap (B = 2000, seed 258),
paired: one draw of panos scores every arm. The matched-precision read is selection
aware: in every draw the precision bar is recomputed and both arms' best thresholds are
re-picked. Pooled rows resample (split, pano) clusters and use one threshold across the
pool. Pooled rows are the inferential ones; per-split rows are descriptive.

**Appendix: the production cut as a filter, for reference.** The validator's own
verdict (``pred == "correct"``, P(correct) > 0.5) applied to RampNet's peaks at a 0.05
or 0.30 floor, against RampNet@0.30, with paired bootstrap intervals on the F1
difference. Nothing is fitted. This replaces the step-2 read-out's "untuned two-pass"
table, now exactly re-scored.

Caveats that travel with every number: TP labels on sub-threshold peaks are a lower
bound on verdict splits (step 1), which penalizes any rule that admits sub-threshold
peaks; the op_caches predate the seam fix (``cascade_cost_35.OP_CACHE_NOTE``).

    python scripts/analysis/validator_fusion_258.py            # write JSON, print tables
    python scripts/analysis/validator_fusion_258.py --check    # re-derive, compare bytes

CPU only, no network, no imagery. Writes ``analysis_out/verifier_258/validator_fusion.json``.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import radius_sq_for, score_pano  # noqa: E402
from rampnet.eval import PINNED_SPLITS  # noqa: E402

from cascade_cost_35 import OP_CACHE_NOTE, THR_SWEEP, _pano_counts, _prf  # noqa: E402
from operating_point_curve import read_cache  # noqa: E402
from validator_auc_258 import (GOPRO_MAX_POOL, OPEN_POOL,  # noqa: E402
                               REFERENCE_SPLITS, load_split)

CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
JOB_LOG = os.path.join(REPO, "docs", "data", "validator_258_42152480.log")
DEFAULT_OUT = os.path.join(OUT, "verifier_258", "validator_fusion.json")
DEV_SPLIT = "sao_paulo"
OPERATING_POINT = 0.30
FUSED_GRID = tuple(round(0.005 * i, 3) for i in range(1, 200))
PRODUCTION_FLOORS = (0.05, 0.30)
EPS = 1e-6
B = 2000
SEED = 258
ROUND = 4


# --------------------------------------------------------------------------- #
# Pure helpers -- unit-tested in tests/test_validator_fusion_258.py
# --------------------------------------------------------------------------- #
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def features(s, v):
    """``[1, logit(s), logit(v)]`` design rows."""
    s, v = np.atleast_1d(s), np.atleast_1d(v)
    return np.column_stack([np.ones(len(s)), logit(s), logit(v)])


def fit_logistic(X, y, iters=100, tol=1e-10):
    """Unpenalized logistic regression by Newton/IRLS. Deterministic; raises if it
    does not converge (separable data has no finite maximum)."""
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < tol:
            return w
    raise RuntimeError("logistic fit did not converge")


def fused_prob(w, s, v):
    return 1.0 / (1.0 + np.exp(-(features(s, v) @ w)))


def best_at_precision(c, p_min):
    """``cascade_cost_35``'s ``best_recall_at_precision_ge_baseline`` over a threshold
    axis: ``c`` is ``[..., T, 4]`` summed counts; returns the index along T of the
    highest recall with precision >= ``p_min`` (ties by F1, then the first), or -1."""
    p, r, f = _prf(c)
    ok = p >= p_min - 1e-12
    key = np.where(ok, r * 4.0 + f, -1.0)   # recall first, F1 breaks ties (F1 <= 1 < 4)
    idx = np.argmax(key, axis=-1)
    return np.where(np.take_along_axis(ok, idx[..., None], -1)[..., 0], idx, -1)


def bootstrap_weights(n, b=B, seed=SEED):
    """``[b, n]`` multinomial pano counts: row k is how often each pano is drawn."""
    rng = np.random.default_rng(seed)
    w = np.zeros((b, n), dtype=np.int64)
    for k in range(b):
        np.add.at(w[k], rng.integers(0, n, n), 1)
    return w


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return [round(float(np.percentile(v, 2.5)), ROUND), round(float(np.percentile(v, 97.5)), ROUND)]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_panos(split):
    """Per pano: op_cache peaks ``[(x, y, s), ...]`` with the validator's P(correct)
    and argmax aligned, plus GT. Every op_cache pano is kept (a pano with no peaks still
    contributes its GT to recall)."""
    rows, _ = load_split(split)
    by_xy = {(r["pano"], r["x"], r["y"]): r for r in rows}
    panos, meta = read_cache(os.path.join(CACHE_DIR, f"{split}.json"))
    out = []
    for pd in panos:
        peaks = list(pd["preds"])
        cand = [by_xy[(pd["pano"], x, y)] for x, y, _ in peaks]
        out.append({"pano": pd["pano"], "gt": pd["gt"], "peaks": peaks,
                    "s": np.array([p[2] for p in peaks], dtype=float),
                    "v": np.array([c["p_correct"] for c in cand], dtype=float),
                    "pred": [c["pred"] for c in cand],
                    "label": [c["label"] for c in cand]})
    assert sum(len(p["peaks"]) for p in out) == len(rows), split
    return out, radius_sq_for(meta["radius_normalized"])


class Counts:
    """Memoised ``score_pano`` counts per (pano, kept-index tuple)."""

    def __init__(self, panos, radius_sq):
        self.panos, self.radius_sq, self.memo = panos, radius_sq, {}

    def pano(self, k, keep):
        idx = tuple(np.flatnonzero(keep).tolist())
        key = (k, idx)
        c = self.memo.get(key)
        if c is None:
            p = self.panos[k]
            c = _pano_counts(score_pano([p["peaks"][i] for i in idx], p["gt"], self.radius_sq))
            self.memo[key] = c
        return c

    def matrix(self, keep_fn, grid):
        """``[T, m, 4]`` counts for ``keep_fn(pano, t)`` over threshold grid ``grid``."""
        return np.array([[self.pano(k, keep_fn(p, t)) for k, p in enumerate(self.panos)]
                         for t in grid], dtype=np.int64).reshape(len(grid), len(self.panos), 4)


def split_tables(panos, radius_sq, w):
    """Count matrices for one split: threshold-only sweep, fused sweep, RampNet@0.30,
    the deployed fused threshold is applied later, and the production-cut cells."""
    cnt = Counts(panos, radius_sq)
    for p in panos:
        p["f"] = fused_prob(w, p["s"], p["v"]) if len(p["s"]) else np.zeros(0)
    thr = cnt.matrix(lambda p, t: p["s"] >= t, THR_SWEEP)
    fus = cnt.matrix(lambda p, t: p["f"] >= t, FUSED_GRID)
    base = cnt.matrix(lambda p, t: p["s"] >= t, (OPERATING_POINT,))[0]
    prod = {f"rampnet@{t:.2f}+validator": cnt.matrix(
        lambda p, t: (p["s"] >= t) & np.array([x == "correct" for x in p["pred"]], bool),
        (t,))[0] for t in PRODUCTION_FLOORS}
    return {"thr": thr, "fus": fus, "base": base, "prod": prod}


def check_against_labels(panos, base):
    """RampNet@0.30 re-scored must equal the step-1 labels summed (a prefix keeps them)."""
    tp = sum(sum(1 for s, lab in zip(p["s"], p["label"]) if s >= OPERATING_POINT and lab == "tp")
             for p in panos)
    fp = sum(sum(1 for s, lab in zip(p["s"], p["label"]) if s >= OPERATING_POINT and lab == "fp")
             for p in panos)
    assert (tp, fp) == (int(base[:, 0].sum()), int(base[:, 1].sum())), (tp, fp)


def report(c):
    p, r, f = _prf(c)
    return {"tp": int(c[0]), "fp": int(c[1]), "recall_tp": int(c[2]), "recall_n_gt": int(c[3]),
            "precision": round(float(p), ROUND), "recall": round(float(r), ROUND),
            "f1": round(float(f), ROUND)}


def matched_block(t, W, tau_dev):
    """Matched-precision read and the deployed-threshold read for one split or pool."""
    thr, fus, base = t["thr"], t["fus"], t["base"]
    # point estimate
    b_sum = base.sum(0)
    p_star = float(_prf(b_sum)[0])
    i_thr = int(best_at_precision(thr.sum(1), p_star))
    i_fus = int(best_at_precision(fus.sum(1), p_star))
    out = {"precision_bar": round(p_star, ROUND), "rampnet@0.30": report(b_sum)}
    out["threshold_only"] = (dict(report(thr[i_thr].sum(0)), threshold=THR_SWEEP[i_thr])
                             if i_thr >= 0 else None)
    out["fused"] = (dict(report(fus[i_fus].sum(0)), threshold=FUSED_GRID[i_fus])
                    if i_fus >= 0 else None)
    r_thr = float(_prf(thr[i_thr].sum(0))[1]) if i_thr >= 0 else np.nan
    r_fus = float(_prf(fus[i_fus].sum(0))[1]) if i_fus >= 0 else np.nan
    out["delta_recall"] = round(r_fus - r_thr, ROUND) if np.isfinite(r_fus - r_thr) else None
    # selection-aware bootstrap: bar and both thresholds re-picked per draw
    bs = W @ base                                   # [B, 4]
    pb = _prf(bs)[0]
    ts = np.einsum("bm,tmk->btk", W, thr)            # [B, T, 4]
    fs = np.einsum("bm,tmk->btk", W, fus)
    rows = np.arange(len(W))
    it = best_at_precision(ts, pb[:, None])
    jf = best_at_precision(fs, pb[:, None])
    rt = np.where(it >= 0, _prf(ts[rows, np.maximum(it, 0)])[1], np.nan)
    rf = np.where(jf >= 0, _prf(fs[rows, np.maximum(jf, 0)])[1], np.nan)
    out["delta_recall_ci"] = ci(rf - rt)
    out["threshold_only_recall_ci"] = ci(rt)
    out["fused_recall_ci"] = ci(rf)
    # deployed: the sao_paulo threshold applied unchanged
    j = FUSED_GRID.index(tau_dev)
    dep = fus[j]
    d_sum = dep.sum(0)
    pd_, rd, fd = (float(x) for x in _prf(d_sum))
    pb_, rb, fb = (float(x) for x in _prf(b_sum))
    ds = W @ dep
    P1, R1, F1 = _prf(ds)
    P0, R0, F0 = _prf(bs)
    out["deployed"] = dict(report(d_sum), threshold=tau_dev,
                           delta_precision=round(pd_ - pb_, ROUND),
                           delta_recall=round(rd - rb, ROUND), delta_f1=round(fd - fb, ROUND),
                           delta_precision_ci=ci(P1 - P0), delta_recall_ci=ci(R1 - R0),
                           delta_f1_ci=ci(F1 - F0))
    # appendix: the production cut
    app = {}
    for name, c in t["prod"].items():
        s = c.sum(0)
        f1 = float(_prf(s)[2])
        app[name] = dict(report(s), delta_f1=round(f1 - fb, ROUND),
                         delta_f1_ci=ci(_prf(W @ c)[2] - F0))
    out["production_cut"] = app
    return out


def validator_seconds(path=JOB_LOG):
    """``{split: seconds}`` of validator wall-clock from the committed job log."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(\w+): wrote in (\d+)s", line.strip())
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def add_cost(blk):
    """Validator wall-clock per ramp the deployed rule adds over RampNet@0.30 (net TP
    gain over every pano). The validator scored every candidate >= 0.05, so this is the
    whole second-pass cost; RampNet's own pass is common to both arms."""
    gain = blk["deployed"]["tp"] - blk["rampnet@0.30"]["tp"]
    blk["deployed"]["net_tp_gain"] = gain
    sec = blk.get("validator_seconds")
    blk["deployed"]["validator_seconds_per_added_tp"] = (
        round(sec / gain, 1) if sec is not None and gain > 0 else None)


def stack(tables):
    """Concatenate per-split tables along the pano axis (pooled rows)."""
    return {"thr": np.concatenate([t["thr"] for t in tables], 1),
            "fus": np.concatenate([t["fus"] for t in tables], 1),
            "base": np.concatenate([t["base"] for t in tables], 0),
            "prod": {k: np.concatenate([t["prod"][k] for t in tables], 0)
                     for k in tables[0]["prod"]}}


def build(splits):
    assert DEV_SPLIT in splits, "the fusion is fitted on sao_paulo"
    data = {s: load_panos(s) for s in splits}
    dev, _ = data[DEV_SPLIT]
    s_dev = np.concatenate([p["s"] for p in dev])
    v_dev = np.concatenate([p["v"] for p in dev])
    lab = np.concatenate([np.array(p["label"]) for p in dev])
    keep = lab != "ignored"
    w = fit_logistic(features(s_dev[keep], v_dev[keep]), lab[keep] == "tp")

    tables = {}
    for s, (panos, rsq) in data.items():
        tables[s] = split_tables(panos, rsq, w)
        check_against_labels(panos, tables[s]["base"])

    # the deployed threshold, fixed on sao_paulo
    t_dev = tables[DEV_SPLIT]
    p_star = float(_prf(t_dev["base"].sum(0))[0])
    j = int(best_at_precision(t_dev["fus"].sum(1), p_star))
    assert j >= 0, "no fused threshold reaches RampNet@0.30's precision on sao_paulo"
    tau = FUSED_GRID[j]

    secs = validator_seconds()
    result = {"script": "scripts/analysis/validator_fusion_258.py",
              "dev_split": DEV_SPLIT, "operating_point": OPERATING_POINT,
              "bootstrap": {"B": B, "seed": SEED, "unit": "pano",
                            "matched_precision": "selection aware: bar and thresholds re-picked per draw"},
              "fusion": {"model": "logistic: sigmoid(b0 + b1*logit(rampnet) + b2*logit(validator))",
                         "fit": "unpenalized ML (IRLS) on sao_paulo tp/fp candidates, redundant kept",
                         "n_fit": int(keep.sum()), "n_fit_tp": int((lab[keep] == "tp").sum()),
                         "coef": {"intercept": round(float(w[0]), ROUND),
                                  "logit_rampnet": round(float(w[1]), ROUND),
                                  "logit_validator": round(float(w[2]), ROUND)},
                         "deployed_threshold": tau,
                         "deployed_threshold_rule": "best recall at P >= P(RampNet@0.30) on sao_paulo"},
              "grids": {"threshold_only": [THR_SWEEP[0], THR_SWEEP[-1], 0.01],
                        "fused": [FUSED_GRID[0], FUSED_GRID[-1], 0.005]},
              "op_cache_note": OP_CACHE_NOTE,
              "validator_seconds_source": "docs/data/validator_258_42152480.log",
              "splits": {}, "pooled": {}}
    for s, t in tables.items():
        W = bootstrap_weights(t["base"].shape[0])
        blk = matched_block(t, W, tau)
        blk["held_out"] = s != DEV_SPLIT
        blk["validator_seconds"] = secs.get(s)
        add_cost(blk)
        result["splits"][s] = blk
    pools = {"open": OPEN_POOL, "gopro_max": GOPRO_MAX_POOL, "reference": REFERENCE_SPLITS,
             "held_out_all": tuple(s for s in splits if s != DEV_SPLIT)}
    for name, members in pools.items():
        members = [m for m in members if m in tables and m != DEV_SPLIT]
        if not members:
            continue
        t = stack([tables[m] for m in members])
        W = bootstrap_weights(t["base"].shape[0])
        blk = matched_block(t, W, tau)
        blk["members"] = members
        blk["validator_seconds"] = (sum(secs[m] for m in members)
                                    if all(m in secs for m in members) else None)
        add_cost(blk)
        result["pooled"][name] = blk
    return result


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def _f(v, ci_=None, fmt="{:+.3f}"):
    if v is None:
        return "n/a"
    s = fmt.format(v)
    return f"{s} [{fmt.format(ci_[0])}, {fmt.format(ci_[1])}]" if ci_ else s


def print_tables(result):
    print("\n### Recall at matched precision (bar = RampNet@0.30's precision)\n")
    print("| split | P bar | threshold-only R (t) | fused R (t) | fused - threshold-only |")
    print("|---|--:|--:|--:|--:|")
    rows = list(result["splits"].items()) + [(f"**{k} pooled**", v)
                                             for k, v in result["pooled"].items()]
    for s, b in rows:
        to, fu = b["threshold_only"], b["fused"]
        tos = f"{to['recall']:.3f} ({to['threshold']:.2f})" if to else "n/a"
        fus = f"{fu['recall']:.3f} ({fu['threshold']:.3f})" if fu else "n/a"
        tag = " (dev)" if b.get("held_out") is False else ""
        print(f"| {s}{tag} | {b['precision_bar']:.3f} | {tos} | {fus} | "
              f"{_f(b['delta_recall'], b['delta_recall_ci'])} |")
    print(f"\n### Deployed: fused threshold {result['fusion']['deployed_threshold']} "
          "fixed on sao_paulo, vs RampNet@0.30\n")
    print("| split | RampNet@0.30 P / R / F1 | fused P / R / F1 | dP | dR | dF1 |")
    print("|---|--:|--:|--:|--:|--:|")
    for s, b in rows:
        a, d = b["rampnet@0.30"], b["deployed"]
        print(f"| {s} | {a['precision']:.3f} / {a['recall']:.3f} / {a['f1']:.3f} | "
              f"{d['precision']:.3f} / {d['recall']:.3f} / {d['f1']:.3f} | "
              f"{_f(d['delta_precision'], d['delta_precision_ci'])} | "
              f"{_f(d['delta_recall'], d['delta_recall_ci'])} | "
              f"{_f(d['delta_f1'], d['delta_f1_ci'])} |")
    print("\n### Appendix: the production cut as a filter (F1, and dF1 vs RampNet@0.30)\n")
    keys = list(next(iter(result["splits"].values()))["production_cut"])
    print("| split | RampNet@0.30 | " + " | ".join(keys) + " |")
    print("|---|--:|" + "--:|" * len(keys))
    for s, b in rows:
        cells = [f"{b['production_cut'][k]['f1']:.3f} "
                 f"({_f(b['production_cut'][k]['delta_f1'], b['production_cut'][k]['delta_f1_ci'])})"
                 for k in keys]
        print(f"| {s} | {b['rampnet@0.30']['f1']:.3f} | " + " | ".join(cells) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="*", default=list(PINNED_SPLITS))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="re-derive and compare bytes")
    args = ap.parse_args(argv)
    result = build(args.splits)
    text = dumps(result)
    if args.check:
        try:
            with open(args.out, encoding="utf-8", newline="") as f:
                same = f.read() == text
        except OSError:
            same = False
        print("current" if same else f"stale or missing: {args.out}")
        return 0 if same else 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(json.dumps(result["fusion"], indent=1))
    print_tables(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
