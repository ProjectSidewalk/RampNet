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
every pano on manual_gold), each three ways: with or without ``redundant`` FPs (second
hits on a ramp another peak already claimed, where a validator that says "ramp" is right
about the image), and ``below_op``, only the candidates under RampNet's 0.30 operating
point -- the ones a lowered threshold adds, where a second pass would have to earn its
recall. ``below_op`` was added on 2026-10-22 after richmond, the first split scored, and
before any other split's scores were read.

**Secondary: an untuned two-pass rule.** RampNet alone at 0.30 (the operating point,
docs/operating_point_correction) against RampNet at a lowered threshold followed by the
validator's own verdict (``pred == "correct"``, i.e. P(correct) > 0.5, the label
production returns). No threshold is fitted on these rows. Precision, recall and F1 are
at the candidate level, FN = GT ramps not claimed by a kept TP; every caveat on the
candidate labels applies (sub-threshold TPs are a lower bound on verdict splits).

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
from operating_point_curve import read_cache  # noqa: E402
from verifier_candidates_258 import pano_groups  # noqa: E402

CAND_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "candidates")
SCORE_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "validator_scores")
CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
DEFAULT_OUT = os.path.join(OUT, "verifier_258", "validator_auc.json")
B = 2000
SEED = 258
OPERATING_POINT = 0.30
TWO_PASS_FLOORS = (0.05, 0.30)
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
def load_split(split):
    """Joined rows: candidate fields plus ``p_correct`` and ``pred``; and per-pano GT counts."""
    with open(os.path.join(CAND_DIR, f"{split}.json"), encoding="utf-8") as f:
        cand = json.load(f)
    with open(os.path.join(SCORE_DIR, f"{split}.json"), encoding="utf-8") as f:
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


def two_pass_block(rows, n_gt):
    """RampNet at the operating point vs RampNet >= floor then the validator's verdict."""
    g = sum(n_gt.values())
    scored = [r for r in rows if r["label"] != "ignored"]

    def cell(keep):
        tp = sum(1 for r in scored if keep(r) and r["label"] == "tp")
        fp = sum(1 for r in scored if keep(r) and r["label"] == "fp")
        p, rc, f = prf(tp, fp, g)
        return {"tp": tp, "fp": fp, "fn": g - tp, "precision": round(p, ROUND),
                "recall": round(rc, ROUND), "f1": round(f, ROUND)}

    out = {"n_gt": g, f"rampnet@{OPERATING_POINT:.2f}":
           cell(lambda r: r["score"] >= OPERATING_POINT)}
    for t in TWO_PASS_FLOORS:
        out[f"rampnet@{t:.2f}+validator"] = cell(
            lambda r, t=t: r["score"] >= t and r["pred"] == "correct")
    return out


def build(splits):
    data = {s: load_split(s) for s in splits}
    result = {"script": "scripts/analysis/validator_auc_258.py", "bootstrap": {"B": B, "seed": SEED},
              "open_splits": [s for s in OPEN_SPLITS if s in data],
              "reference_splits": [s for s in REFERENCE_SPLITS if s in data],
              "splits": {}, "pooled": {}}
    for s, (rows, n_gt) in data.items():
        block = {}
        for sub in ("all", "unbiased"):
            r, g = subset(s, rows, n_gt, sub == "unbiased")
            block[sub] = {"auc": {"with_redundant": auc_block(r, False),
                                  "without_redundant": auc_block(r, True),
                                  "below_op": auc_block(below_op(r), False)},
                          "two_pass": two_pass_block(r, g)}
        result["splits"][s] = block
    # Pooled AUCs, panos keyed by split so the bootstrap resamples (split, pano) clusters.
    pools = {"open": result["open_splits"], "reference": result["reference_splits"]}
    for name, members in pools.items():
        if not members:
            continue
        block = {}
        for sub in ("all", "unbiased"):
            pooled = []
            for s in members:
                r, _ = subset(s, *data[s], sub == "unbiased")
                pooled += [{**x, "pano": f"{s}:{x['pano']}"} for x in r]
            block[sub] = {"with_redundant": auc_block(pooled, False),
                          "without_redundant": auc_block(pooled, True),
                          "below_op": auc_block(below_op(pooled), False)}
        result["pooled"][name] = block
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


def print_two_pass(result, sub="all"):
    keys = None
    print(f"\n### Two-pass rule, {sub} panos (F1; precision / recall)\n")
    for s, block in result["splits"].items():
        tp = block[sub]["two_pass"]
        if keys is None:
            keys = [k for k in tp if k != "n_gt"]
            print("| split | GT | " + " | ".join(keys) + " |")
            print("|---|--:|" + "--:|" * len(keys))
        cells = [f"{tp[k]['f1']:.3f} ({tp[k]['precision']:.2f} / {tp[k]['recall']:.2f})"
                 for k in keys]
        print(f"| {s} | {tp['n_gt']} | " + " | ".join(cells) + " |")


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
        for red in ("with_redundant", "without_redundant", "below_op"):
            print_tables(result, sub, red)
    for sub in ("all", "unbiased"):
        print_two_pass(result, sub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
