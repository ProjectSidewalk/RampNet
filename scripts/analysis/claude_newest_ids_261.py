"""Paired intervals and the radius check for the newest-ids Claude legs (PR #261).

The newest-ids campaign (``docs/claude_newest_ids.md``) ran ``claude-opus-5-5`` and
``claude-fable-5-1`` (effort low, first-party API) on all 13 splits. This script derives the
two analyses that write-up quotes beyond the plain P / R / F1 table, from committed files only:
the bundles (``benchmark/<split>/records.jsonl`` + ``verdicts.json``, or ``manual_labels/`` for
``manual_gold``) and the published detections in ``benchmark/model_detections/``. No
``.model_cache``, no panoramas, no API key, no GPU.

1. **Paired F1 intervals against RampNet.** For every (split, leg), RampNet minus the leg in
   F1, with a pano-level paired bootstrap: resample the split's panos with replacement and
   recompute both F1s on the same draw. RampNet is read from the committed records at the
   deployed 0.55 (confidence >= 0.55), the operating point of the scoreboard's per-split cells. The
   draw construction and seed are ``bayonne_paired_159.py``'s (``random.Random(SEED)``,
   ``N_BOOT`` draws, percentile 2.5 / 97.5), with one generator per split, so both legs on a
   split are measured on the same draws and the result does not depend on the order of the
   legs. F1 per draw is ``aggregate``'s: precision over every pano, recall over the panos
   whose recall is confirmed (all of them, on these 13 splits; asserted).
2. **The radius check.** Both legs re-scored at the standard radius 0.022 and at 1.5x and 2x
   it (0.033, 0.044), on every split, with ``rampnet.detection_eval.score_pano``. The share of
   the Opus-minus-Fable gap that closes at 0.044 is computed from unrounded F1s.

It also records, per leg and split, how many points the published file emits and how many
were scored (emitted points on panos outside the bundle are not scored; points that land
near an ignore point are neither TP nor FP).

Every leg's F1 at 0.022 is asserted equal to ``analysis_out/scoreboard.json``'s cell, so the
two artifacts cannot drift apart silently.

    python scripts/analysis/claude_newest_ids_261.py            # --check (default)
    python scripts/analysis/claude_newest_ids_261.py --write    # analysis_out/claude_newest_ids_261/results.json
    python scripts/analysis/claude_newest_ids_261.py --markdown # the tables the doc quotes
"""
import argparse
import hashlib
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import prediction_confidence, radius_sq_for, score_pano  # noqa: E402

import scoreboard as SB  # noqa: E402
from export_model_cache import load_detections  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "claude_newest_ids_261", "results.json")
SCOREBOARD = os.path.join(REPO, "analysis_out", "scoreboard.json")

#: The 13 splits the campaign ran, fixed here rather than read from the scoreboard's split
#: list: a split added later (juneau, PR #260) has no leg from this campaign.
SPLITS = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville", "laurens_gsv",
          "laurens_mapillary", "morgantown", "paterson", "richmond", "sao_paulo", "bayonne",
          "manual_gold")
#: Published names (``rampnet.roster.published_name``) of the two legs.
OPUS = "claude-opus-5-5-effort-low-anthropic"
FABLE = "claude-fable-5-1-effort-low-anthropic"
LEGS = (OPUS, FABLE)
RAMPNET = "rampnet"
#: The scoreboard's RampNet operating point. Most bundles already stop there; manual_gold's is
#: exported at a 0.05 floor, so the filter matters there (``rampnet.eval.rescore``'s ``>=``).
RAMPNET_OP = SB.DEPLOYED_THRESHOLD
RADII = (0.022, 0.033, 0.044)
N_BOOT = 10_000
SEED = 0


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def prf(tp, fp, tp_rc, n_gt_rc):
    """Precision, recall, F1 the way ``aggregate`` computes them."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp_rc / n_gt_rc if n_gt_rc else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def per_pano(preds, gts, radius):
    """{pano: (tp, fp, n_gt, fn_confirmed)} at one normalized radius."""
    rsq = radius_sq_for(radius)
    out = {}
    for pid, gt in gts.items():
        s = score_pano(preds.get(pid, []), gt, radius_sq=rsq)
        out[pid] = (s.tp, s.fp, s.n_gt, s.fn_confirmed)
    return out


def totals(rows, panos):
    tp = sum(rows[p][0] for p in panos)
    fp = sum(rows[p][1] for p in panos)
    tp_rc = sum(rows[p][0] for p in panos if rows[p][3])
    n_rc = sum(rows[p][2] for p in panos if rows[p][3])
    return tp, fp, tp_rc, n_rc


def cell(rows, panos):
    tp, fp, tp_rc, n_rc = totals(rows, panos)
    p, r, f = prf(tp, fp, tp_rc, n_rc)
    return {"tp": tp, "fp": fp, "fn": n_rc - tp_rc, "precision": _r(p, 6), "recall": _r(r, 6),
            "f1": _r(f, 6)}


def paired_bootstrap(rows_a, rows_b, panos, rng):
    """Sorted F1(a) - F1(b) over ``N_BOOT`` pano resamples drawn from ``rng``."""
    diffs = []
    n = len(panos)
    for _ in range(N_BOOT):
        draw = [panos[rng.randrange(n)] for _ in panos]
        diffs.append(prf(*totals(rows_a, draw))[2] - prf(*totals(rows_b, draw))[2])
    diffs.sort()
    return diffs


def load_board():
    with open(SCOREBOARD, encoding="utf-8") as fh:
        return json.load(fh)["per_split"]


def split_result(split, board, bootstrap=True):
    """Everything this script records for one split. ``bootstrap=False`` skips the paired
    intervals (the slow part) and leaves ``paired_f1_vs_rampnet`` out; the tests use it to
    re-derive the radius table on every split cheaply."""
    records, gts = SB.load_split(split)
    assert all(g.fn_confirmed for g in gts.values()), f"{split}: unconfirmed recall panos"
    panos = sorted(gts)
    preds = {RAMPNET: {pid: [p for p in records[pid]["detections"]
                             if prediction_confidence(p) >= RAMPNET_OP] for pid in gts}}
    emitted = {}
    for leg in LEGS:
        d = load_detections(leg, split, publish_as=leg)
        assert d is not None, (leg, split)
        assert set(gts) <= set(d), f"{leg}/{split}: missing panos"
        preds[leg] = d
        emitted[leg] = sum(len(v) for v in d.values())
    rows = {m: {rad: per_pano(preds[m], gts, rad) for rad in RADII} for m in preds}
    cells = {m: {str(rad): cell(rows[m][rad], panos) for rad in RADII} for m in preds}
    for m in preds:
        sb = board[m][split]
        assert abs(cells[m]["0.022"]["f1"] - sb["f1"]) < 1e-5, (m, split, sb["f1"])
        assert (cells[m]["0.022"]["tp"], cells[m]["0.022"]["fp"]) == (sb["tp"], sb["fp"]), (m, split)
    f = {m: {rad: cells[m][str(rad)]["f1"] for rad in RADII} for m in LEGS}
    gap = {str(rad): _r(f[OPUS][rad] - f[FABLE][rad], 6) for rad in RADII}
    closed = (f[OPUS][0.022] - f[FABLE][0.022]) - (f[OPUS][0.044] - f[FABLE][0.044])
    res = {
        "n_panos": len(panos), "n_gt": sum(len(gts[p].gt_points) for p in panos),
        "scores": cells,
        "points": {leg: {"emitted": emitted[leg],
                         "scored": cells[leg]["0.022"]["tp"] + cells[leg]["0.022"]["fp"]}
                   for leg in LEGS},
        "opus_minus_fable_gap": gap,
        "gap_share_closed_at_0.044": _r(closed / (f[OPUS][0.022] - f[FABLE][0.022])),
    }
    if bootstrap:
        rng = random.Random(SEED)
        paired = {}
        for leg in LEGS:
            diffs = paired_bootstrap(rows[RAMPNET][0.022], rows[leg][0.022], panos, rng)
            point = cells[RAMPNET]["0.022"]["f1"] - cells[leg]["0.022"]["f1"]
            lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]
            paired[leg] = {"rampnet_minus_leg_f1": _r(point),
                           "ci95": [_r(lo), _r(hi)],
                           "ci_excludes_zero": lo > 0 or hi < 0,
                           "share_draws_rampnet_ahead": _r(sum(d > 0 for d in diffs) / N_BOOT)}
        res["paired_f1_vs_rampnet"] = paired
    return res


def build():
    board = load_board()
    out = {"splits": list(SPLITS), "legs": list(LEGS), "radii": list(RADII),
           "rampnet_operating_point": RAMPNET_OP,
           "bootstrap": {"unit": "pano", "n_draws": N_BOOT, "seed": SEED,
                         "rng": "random.Random(SEED), one generator per split",
                         "interval": "percentile 2.5 / 97.5"},
           "per_split": {split: split_result(split, board) for split in SPLITS}}
    shares = [v["gap_share_closed_at_0.044"] for v in out["per_split"].values()]
    out["gap_share_closed_at_0.044"] = {"min": min(shares), "median": _r(st.median(shares)),
                                        "max": max(shares)}
    blob = json.dumps(out["per_split"], sort_keys=True).encode()
    out["per_split_sha256"] = hashlib.sha256(blob).hexdigest()
    return out


def markdown(res):
    lines = ["| split | leg | RampNet F1 | leg F1 | RampNet - leg [95% CI] | draws RampNet ahead |",
             "|---|---|---:|---:|---|---:|"]
    for s, v in res["per_split"].items():
        for leg in LEGS:
            p = v["paired_f1_vs_rampnet"][leg]
            lines.append(f"| {s} | {leg.split('-effort')[0]} | {v['scores'][RAMPNET]['0.022']['f1']:.3f} "
                         f"| {v['scores'][leg]['0.022']['f1']:.3f} | {p['rampnet_minus_leg_f1']:+.3f} "
                         f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['share_draws_rampnet_ahead']:.3f} |")
    lines += ["", "| Split | Gap at r=0.022 | r=0.033 | r=0.044 | share closed at 0.044 "
              "| Opus 5.5 F1 at 0.044 | Fable 5.1 F1 at 0.044 |", "|---|---|---|---|---:|---|---|"]
    for s, v in res["per_split"].items():
        g = v["opus_minus_fable_gap"]
        lines.append(f"| {s} | {g['0.022']:+.3f} | {g['0.033']:+.3f} | {g['0.044']:+.3f} "
                     f"| {v['gap_share_closed_at_0.044']:.0%} "
                     f"| {v['scores'][OPUS]['0.044']['f1']:.3f} | {v['scores'][FABLE]['0.044']['f1']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the committed JSON")
    ap.add_argument("--check", action="store_true", help="compare against it (the default)")
    ap.add_argument("--markdown", action="store_true", help="print the doc's tables")
    args = ap.parse_args()
    res = build()
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if args.markdown:
        print(markdown(res))
    if args.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(OUT, REPO)}")
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
