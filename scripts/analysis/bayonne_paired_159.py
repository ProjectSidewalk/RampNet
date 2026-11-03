"""Paired RampNet-vs-Gemini tests on Bayonne (#159, PR #239 review S3).

Two claims in the Bayonne write-up compare RampNet with a chat VLM on the same 301
ground-truth ramps, so they need paired tests, not overlapping marginal intervals:

1. **Recall.** gemini-3.6-flash finds 126 of the 301 GT ramps at its operating point and
   RampNet 113 at its deployed 0.55. Is that difference distinguishable from noise? An
   exact two-sided McNemar test over per-GT-ramp hits answers it: only the discordant
   ramps (found by one model and not the other) carry information.
2. **F1.** RampNet's F1 lead over gemini-3.1-pro-preview is 0.086. A pano-level paired
   bootstrap (resample panos with replacement, recompute both F1s on the same draw)
   gives its interval.

Scoring is the scoreboard's: the same bundle ground truth, the same committed detections,
and the same matcher (``rampnet.detection_eval.score_pano``'s ordering and
``rampnet.metrics.greedy_match``), so the per-model totals reproduce the
``results:bayonne`` block in ``docs/model_comparison.md``. Every pano of the split is in
the recall pool (all 125 are recall-confirmed), so per-GT hits are well defined.

CPU only, committed inputs only::

    python scripts/analysis/bayonne_paired_159.py           # check against the committed JSON
    python scripts/analysis/bayonne_paired_159.py --write   # analysis_out/bayonne_159/paired_tests.json
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, _xy, prediction_confidence, radius_sq_for, score_pano)
from rampnet.metrics import greedy_match  # noqa: E402

import scoreboard as SB  # noqa: E402
from export_model_cache import load_detections  # noqa: E402

SPLIT = "bayonne"
OUT = os.path.join(REPO, "analysis_out", "bayonne_159", "paired_tests.json")
RAMPNET = "rampnet"
#: (published name, role). Chat VLMs emit no confidence, so no operating point applies.
CHALLENGERS = ("gemini-3.6-flash", "gemini-3.1-pro-preview")
N_BOOT = 10_000
SEED = 0


def hits_and_counts(preds, gt, radius_sq):
    """(set of matched GT indices, tp, fp, n_gt) for one pano, matched as score_pano does."""
    confs = [prediction_confidence(p) for p in preds]
    if any(c is not None for c in confs):
        order = sorted(range(len(preds)),
                       key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                       reverse=True)
        preds = [preds[i] for i in order]
    assign = greedy_match([_xy(p) for p in preds], gt.gt_points, radius_sq,
                          PANO_SCALE_X, PANO_SCALE_Y, True)
    hits = {k for k, _ in assign if k >= 0}
    s = score_pano(preds, gt, radius_sq=radius_sq)
    assert s.tp == len(hits)
    return hits, s.tp, s.fp, s.n_gt


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial(b + c, 0.5) on the discordant pairs."""
    n, k = b + c, min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def build():
    radius_sq = radius_sq_for()
    records, gts = SB.load_split(SPLIT)
    assert all(g.fn_confirmed for g in gts.values()), "every pano must be in the recall pool"
    preds = {RAMPNET: {pid: records[pid]["detections"] for pid in gts}}
    for name in CHALLENGERS:
        d = load_detections(name, SPLIT)
        assert d is not None, name
        preds[name] = d
    panos = sorted(gts)
    per = {m: {pid: hits_and_counts(preds[m].get(pid, []), gts[pid], radius_sq)
               for pid in panos} for m in preds}
    n_gt = sum(len(gts[p].gt_points) for p in panos)

    totals = {}
    for m, rows in per.items():
        tp = sum(r[1] for r in rows.values())
        fp = sum(r[2] for r in rows.values())
        fn = n_gt - tp
        totals[m] = {"tp": tp, "fp": fp, "fn": fn,
                     "recall": round(tp / n_gt, 4),
                     "precision": round(tp / (tp + fp), 4), "f1": round(f1(tp, fp, fn), 4)}

    recall = {}
    for m in CHALLENGERS:
        only_m = only_r = both = 0
        for pid in panos:
            a, b = per[m][pid][0], per[RAMPNET][pid][0]
            only_m += len(a - b)
            only_r += len(b - a)
            both += len(a & b)
        recall[m] = {"challenger_only": only_m, "rampnet_only": only_r, "both": both,
                     "union": only_m + only_r + both,
                     "union_recall": round((only_m + only_r + both) / n_gt, 4),
                     "mcnemar_exact_p": round(mcnemar_exact(only_m, only_r), 4)}

    rng = random.Random(SEED)
    boot = {}
    for m in CHALLENGERS:
        diffs = []
        for _ in range(N_BOOT):
            draw = [panos[rng.randrange(len(panos))] for _ in panos]
            agg = {}
            for mm in (RAMPNET, m):
                tp = sum(per[mm][p][1] for p in draw)
                fp = sum(per[mm][p][2] for p in draw)
                fn = sum(per[mm][p][3] for p in draw) - tp
                agg[mm] = f1(tp, fp, fn)
            diffs.append(agg[RAMPNET] - agg[m])
        diffs.sort()
        boot[m] = {"rampnet_minus_challenger_f1": round(totals[RAMPNET]["f1"] - totals[m]["f1"], 4),
                   "ci95": [round(diffs[int(0.025 * N_BOOT)], 4),
                            round(diffs[int(0.975 * N_BOOT) - 1], 4)],
                   "share_draws_rampnet_ahead": round(sum(d > 0 for d in diffs) / N_BOOT, 4)}

    return {"split": SPLIT, "n_panos": len(panos), "n_gt": n_gt,
            "radius_normalized": 0.022,
            "rampnet_operating_point": "committed records (deployed 0.55)",
            "totals": totals, "paired_recall_mcnemar": recall,
            "pano_bootstrap_f1": {"n_draws": N_BOOT, "seed": SEED, "legs": boot}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    res = build()
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if args.write:
        with open(OUT, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(OUT, REPO)}")
        print(text)
        return 0
    if not os.path.exists(OUT):
        print(f"MISSING {os.path.relpath(OUT, REPO)} (run with --write)")
        return 1
    with open(OUT, encoding="utf-8") as f:
        same = f.read() == text
    print(f"{'OK' if same else 'DIFFERS'}: {os.path.relpath(OUT, REPO)}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
