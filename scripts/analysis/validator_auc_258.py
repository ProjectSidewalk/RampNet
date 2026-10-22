"""Does the production CurbRamp validator separate RampNet's true ramps from its false positives? (#258)

Step 2's read-out (docs/verifier_258.md). Joins each split's candidate rows
(``analysis_out/verifier_258/candidates/<split>.json``) with the validator's scores
(``analysis_out/verifier_258/validator_scores/<split>.json``, from
``validator_score_258.py``) and asks one question first:

**Primary: AUC.** Over the candidates labelled ``tp`` or ``fp`` (``ignored`` rows dropped),
the AUC of the validator's P(correct) and the AUC of RampNet's own peak score, and their
difference, per split. A verifier earns its place only if it ranks TPs above FPs better
than the detector already does. 95% intervals are a pano bootstrap (panos resampled with
replacement, ``B`` draws, fixed seed), paired: the same draw scores both.

Each AUC is reported on ``all`` panos or the ``unbiased`` subset (``group != "top"``,
every pano on manual_gold), in four strata:

* ``prereg_band`` -- RampNet score in [0.05, 0.55), the candidate set of the posted
  design (#258 comment 6316627866, step 1). **The primary stratum**: the design's stop
  rule is read on it.
* ``below_op`` -- RampNet score in [0.05, 0.30), the candidates a lowered threshold adds
  below the 0.30 operating point (which moved from 0.55 after the plan's wording). Added
  on 2026-10-22 after richmond, the first split scored, and before any other split's
  scores were read; richmond's difference there was +0.084 [-0.039, 0.222] (``b8287cc``).
* ``with_redundant`` -- every peak >= 0.05 (the step-1 candidate set, which widened the
  design's band). Secondary: the candidate set was selected by RampNet's own score, so
  over the full range RampNet is asked to separate its confident TPs from its floor FPs,
  a separation it wins by construction.
* ``without_redundant`` -- the same, without the ``redundant`` FPs (second hits on a ramp
  another peak already claimed, where a validator that says "ramp" is right about the
  image).

Pools: ``open`` is the four non-GSV splits (laurens_mapillary, bayonne,
budapest_district5, juneau); laurens_gsv is reported beside laurens_mapillary as the
same-town control. ``open_with_laurens_gsv`` is the five-split pool the first read-out
used (``open_rig_misses_258.OPEN_SPLITS``), kept so its numbers stay re-derivable;
``gopro_max`` is #265's pool; ``reference`` the seven US splits of #46. Pooled rows are
the inferential ones; per-split rows are descriptive (14 splits x 4 strata x 2 subsets).

The two-pass and fusion read-outs moved to ``validator_fusion_258.py``, which re-scores
every kept set exactly with ``score_pano``.

    python scripts/analysis/validator_auc_258.py            # write JSON, print tables
    python scripts/analysis/validator_auc_258.py --check    # re-derive, compare bytes

CPU only. Writes ``analysis_out/verifier_258/validator_auc.json``.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.eval import PINNED_SPLITS  # noqa: E402

from open_rig_misses_258 import OPEN_SPLITS, REFERENCE_SPLITS  # noqa: E402

#: The open pool for the rig-transfer question: the four non-GSV splits. laurens_gsv is
#: GSV, so it is reported beside laurens_mapillary as the same-town control, not pooled
#: (PR #269 review, finding 5).
OPEN_POOL = ("laurens_mapillary", "bayonne", "budapest_district5", "juneau")
#: The pool the first step-2 read-out used (open_rig_misses_258.OPEN_SPLITS, laurens_gsv
#: included). Kept under its own key so the earlier numbers stay re-derivable.
OPEN_POOL_WITH_LAURENS_GSV = tuple(OPEN_SPLITS)
#: #265's GoPro Max pool (open_rig_misses_258.GROUPS["gopro_max"]).
GOPRO_MAX_POOL = ("laurens_mapillary", "bayonne")
from operating_point_curve import read_cache  # noqa: E402
from verifier_candidates_258 import pano_groups  # noqa: E402

CAND_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "candidates")
SCORE_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "validator_scores")
#: The resolution control (PR #269 review, finding 4): richmond's panos downsampled to
#: 4096x2048 with manual_gold's transform, scored by the same scorer.
RES_CONTROL_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "validator_scores_4096")
CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
DEFAULT_OUT = os.path.join(OUT, "verifier_258", "validator_auc.json")
B = 2000
SEED = 258
OPERATING_POINT = 0.30
#: The candidate band of the posted design (#258 comment 6316627866, step 1):
#: RampNet peaks with score in [0.05, 0.55), 0.55 being the shipped threshold.
PREREG_BAND = (0.05, 0.55)
ROUND = 4


# --------------------------------------------------------------------------- #
# Pure helpers -- unit-tested in tests/test_validator_auc_258.py
# --------------------------------------------------------------------------- #
def average_ranks(v):
    """1-based ranks of ``v``, ties given their average rank."""
    _, inverse, counts = np.unique(np.asarray(v, dtype=float), return_inverse=True,
                                   return_counts=True)
    last = np.cumsum(counts)               # 1-based rank of each tie group's last member
    return (last - (counts - 1) / 2.0)[inverse]


def auc(score, positive):
    """Mann-Whitney AUC: P(a random positive outscores a random negative), ties half."""
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prf(tp, fp, n_gt):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / n_gt if n_gt else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0
    return p, r, f


def cluster_bootstrap(clusters, stat, b=B, seed=SEED):
    """Percentile 95% interval of ``stat(idx)`` over cluster-resampled row indices.

    ``clusters`` is a list of row-index arrays, one per pano. Returns ``(lo, hi)`` over
    the draws where ``stat`` is finite, or ``(nan, nan)`` if none are.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[k] for k in pick]) if len(pick) else np.array([], int)
        v = stat(idx)
        if np.all(np.isfinite(v)):
            vals.append(v)
    if not vals:
        return None
    vals = np.array(vals)
    return np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_split(split, score_dir=SCORE_DIR):
    """Joined rows: candidate fields plus ``p_correct`` and ``pred``; and per-pano GT counts."""
    with open(os.path.join(CAND_DIR, f"{split}.json"), encoding="utf-8") as f:
        cand = json.load(f)
    with open(os.path.join(score_dir, f"{split}.json"), encoding="utf-8") as f:
        scored = json.load(f)
    assert scored["n_candidates"] == cand["n_candidates"], split
    rows = []
    for i, (c, s) in enumerate(zip(cand["candidates"], scored["rows"])):
        assert s["index"] == i and s["pano"] == c["pano"] and s["x"] == c["x"] and s["y"] == c["y"], (split, i)
        rows.append({**c, "p_correct": s["p_correct"], "pred": s["pred"]})
    panos, _ = read_cache(os.path.join(CACHE_DIR, f"{split}.json"))
    n_gt = {p["pano"]: len(p["gt"].gt_points) for p in panos}
    assert sum(n_gt.values()) == cand["n_gt"], split
    return rows, n_gt


def subset(split, rows, n_gt, unbiased):
    """All panos, or the ``group != "top"`` panos (groups from verdicts.json, so a top
    pano with no candidates is still excluded from the GT count)."""
    if not unbiased:
        return rows, n_gt
    groups = pano_groups(split)
    keep = {p for p in n_gt if groups.get(p) != "top"}
    return [r for r in rows if r["pano"] in keep], {p: n for p, n in n_gt.items() if p in keep}


def below_op(rows):
    """The candidates a lowered threshold adds: RampNet score in [floor, 0.30)."""
    return [r for r in rows if r["score"] < OPERATING_POINT]


def prereg_band(rows):
    """The posted design's candidate set: RampNet score in [0.05, 0.55)."""
    lo, hi = PREREG_BAND
    return [r for r in rows if lo <= r["score"] < hi]


def strata(rows):
    """Every AUC stratum, in the order the doc reads them."""
    return {"prereg_band": auc_block(prereg_band(rows), False),
            "below_op": auc_block(below_op(rows), False),
            "with_redundant": auc_block(rows, False),
            "without_redundant": auc_block(rows, True)}


def auc_block(rows, drop_redundant):
    """AUCs of RampNet score and validator P(correct), their difference, and intervals."""
    rows = [r for r in rows if r["label"] != "ignored"
            and not (drop_redundant and r["redundant"])]
    y = np.array([r["label"] == "tp" for r in rows])
    rs = np.array([r["score"] for r in rows])
    vs = np.array([r["p_correct"] for r in rows])
    by_pano = {}
    for i, r in enumerate(rows):
        by_pano.setdefault(r["pano"], []).append(i)
    clusters = [np.array(v) for v in by_pano.values()]

    def stat(idx):
        a_r, a_v = auc(rs[idx], y[idx]), auc(vs[idx], y[idx])
        return np.array([a_r, a_v, a_v - a_r])

    point = stat(np.arange(len(rows)))
    ci = cluster_bootstrap(clusters, stat) if len(rows) else None
    out = {"n_tp": int(y.sum()), "n_fp": int((~y).sum()), "n_panos": len(clusters)}
    for k, name in enumerate(("rampnet", "validator", "delta")):
        out[name] = round(float(point[k]), ROUND) if np.isfinite(point[k]) else None
        out[name + "_ci"] = ([round(float(ci[0][k]), ROUND), round(float(ci[1][k]), ROUND)]
                             if ci is not None else None)
    return out


def resolution_delta(native, low, drop_redundant=False):
    """Paired AUC of the validator on the 4096 copy minus on native, same candidates."""
    keep = [i for i, r in enumerate(native) if r["label"] != "ignored"
            and not (drop_redundant and r["redundant"])]
    y = np.array([native[i]["label"] == "tp" for i in keep])
    vn = np.array([native[i]["p_correct"] for i in keep])
    vl = np.array([low[i]["p_correct"] for i in keep])
    by_pano = {}
    for j, i in enumerate(keep):
        by_pano.setdefault(native[i]["pano"], []).append(j)
    clusters = [np.array(v) for v in by_pano.values()]

    def stat(idx):
        a_n, a_l = auc(vn[idx], y[idx]), auc(vl[idx], y[idx])
        return np.array([a_n, a_l, a_l - a_n])

    point = stat(np.arange(len(keep)))
    ci = cluster_bootstrap(clusters, stat)
    out = {"n_tp": int(y.sum()), "n_fp": int((~y).sum()), "n_panos": len(clusters)}
    for k, name in enumerate(("validator_native", "validator_4096", "delta_4096_minus_native")):
        out[name] = round(float(point[k]), ROUND)
        out[name + "_ci"] = [round(float(ci[0][k]), ROUND), round(float(ci[1][k]), ROUND)]
    return out


def resolution_control(split="richmond"):
    """Every stratum on the 4096 copy, and the paired native-vs-4096 validator difference."""
    path = os.path.join(RES_CONTROL_DIR, f"{split}.json")
    if not os.path.exists(path):
        return None
    native, _ = load_split(split)
    low, _ = load_split(split, RES_CONTROL_DIR)
    lo, hi = PREREG_BAND
    sel = {"prereg_band": lambda r: lo <= r["score"] < hi,
           "below_op": lambda r: r["score"] < OPERATING_POINT,
           "with_redundant": lambda r: True}
    out = {"split": split, "scores": os.path.relpath(path, REPO).replace(os.sep, "/"),
           "strata_4096": strata(low), "paired": {}}
    for name, f in sel.items():
        idx = [i for i, r in enumerate(native) if f(r)]
        out["paired"][name] = resolution_delta([native[i] for i in idx], [low[i] for i in idx])
    return out


def build(splits):
    data = {s: load_split(s) for s in splits}
    result = {"script": "scripts/analysis/validator_auc_258.py", "bootstrap": {"B": B, "seed": SEED},
              "open_splits": [s for s in OPEN_POOL if s in data],
              "open_splits_with_laurens_gsv": [s for s in OPEN_POOL_WITH_LAURENS_GSV if s in data],
              "gopro_max_splits": [s for s in GOPRO_MAX_POOL if s in data],
              "reference_splits": [s for s in REFERENCE_SPLITS if s in data],
              "prereg_band": list(PREREG_BAND), "operating_point": OPERATING_POINT,
              "splits": {}, "pooled": {}}
    for s, (rows, n_gt) in data.items():
        block = {}
        for sub in ("all", "unbiased"):
            r, g = subset(s, rows, n_gt, sub == "unbiased")
            block[sub] = {"auc": strata(r), "n_gt": sum(g.values())}
        result["splits"][s] = block
    # Pooled AUCs, panos keyed by split so the bootstrap resamples (split, pano) clusters.
    pools = {"open": result["open_splits"],
             "open_with_laurens_gsv": result["open_splits_with_laurens_gsv"],
             "gopro_max": result["gopro_max_splits"],
             "reference": result["reference_splits"]}
    for name, members in pools.items():
        if not members:
            continue
        block = {}
        for sub in ("all", "unbiased"):
            pooled = []
            for s in members:
                r, _ = subset(s, *data[s], sub == "unbiased")
                pooled += [{**x, "pano": f"{s}:{x['pano']}"} for x in r]
            block[sub] = strata(pooled)
        result["pooled"][name] = block
    control = resolution_control()
    if control is not None:
        result["resolution_control"] = control
    return result


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def fmt_ci(block, key):
    v, ci = block[key], block[key + "_ci"]
    if v is None:
        return "n/a"
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v:.3f}"


def print_tables(result, sub="all", red="with_redundant"):
    print(f"\n### AUC, {sub} panos, {red.replace('_', ' ')}\n")
    print("| split | TP | FP | RampNet score | validator P(correct) | validator - RampNet |")
    print("|---|--:|--:|--:|--:|--:|")
    for s, block in result["splits"].items():
        a = block[sub]["auc"][red]
        print(f"| {s} | {a['n_tp']} | {a['n_fp']} | {fmt_ci(a, 'rampnet')} | "
              f"{fmt_ci(a, 'validator')} | {fmt_ci(a, 'delta')} |")
    for name, block in result["pooled"].items():
        a = block[sub][red]
        print(f"| **{name} pooled** | {a['n_tp']} | {a['n_fp']} | {fmt_ci(a, 'rampnet')} | "
              f"{fmt_ci(a, 'validator')} | {fmt_ci(a, 'delta')} |")


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
    for sub in ("all", "unbiased"):
        for red in ("prereg_band", "below_op", "with_redundant", "without_redundant"):
            print_tables(result, sub, red)
    return 0


if __name__ == "__main__":
    sys.exit(main())
