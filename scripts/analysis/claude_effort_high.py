"""Opus 5.5 at effort high against effort low, paired, on every split (docs/claude_effort_high.md).

The effort=high campaign (2026-11-17, paid from the ``claude20`` org's expiring credit) re-ran
``claude-opus-5-5`` with the rig of the newest-ids effort=low leg (``claude_newest_ids_261.py``)
changed in exactly two settings: ``--claude-effort high`` and ``--claude-max-tokens 20000``
(room for thinking; 4096 is the as-run default), on all thirteen city splits. (The first 150
``manual_gold`` panos also ran, ``compare.py --limit 150``; a partial split cannot be published,
so that slice stays in the producing ``.model_cache`` and is not analysed here.)

From committed files only -- the bundles, the published detections in
``benchmark/model_detections/`` and ``analysis_out/usage_log.jsonl`` -- it derives:

1. **Per split, high minus low in F1**, with a pano-level paired bootstrap (resample the split's
   panos with replacement, recompute both F1s on the same draw).
2. **Pooled**, over ``rampnet.eval.POOLED_SPLITS`` (the nine US cities) and over the four other
   city splits: micro-pooled tp/fp/fn, with a stratified bootstrap (each split's panos resampled
   within that split, every draw summed across splits).
3. **High against RampNet** per split, the same paired construction as ``claude_newest_ids_261``
   (RampNet at the deployed 0.55), so the high leg's standing is stated against the reference and
   not only against its own low sibling.
4. **Operating point and cost**: boxes emitted per pano by each leg, and cost / thinking per pano
   from the ledger rows of each leg.

Bootstrap: ``random.Random(SEED)``, ``N_BOOT`` draws, percentile 2.5 / 97.5, one generator per
split for the per-split intervals and one per pool for the pooled ones.

Every low-leg city cell at r=0.022 is asserted equal to ``analysis_out/scoreboard.json``, so this
script and the scoreboard cannot drift apart.

    python scripts/analysis/claude_effort_high.py            # --check (default)
    python scripts/analysis/claude_effort_high.py --write    # analysis_out/claude_effort_high/results.json
    python scripts/analysis/claude_effort_high.py --markdown # the tables the doc quotes
    python scripts/analysis/claude_effort_high.py --figures  # docs/figures/claude_effort_high_forest.png
"""
import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import prediction_confidence  # noqa: E402
from rampnet.eval import POOLED_SPLITS  # noqa: E402
from rampnet.ledger import read_rows  # noqa: E402

import scoreboard as SB  # noqa: E402
from claude_newest_ids_261 import cell, per_pano, prf, totals  # noqa: E402
from export_model_cache import load_detections  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "claude_effort_high", "results.json")
SCOREBOARD = os.path.join(REPO, "analysis_out", "scoreboard.json")
LEDGER = os.path.join(REPO, "analysis_out", "usage_log.jsonl")

#: Published names (``rampnet.roster.published_name``).
LOW = "claude-opus-5-5-effort-low-anthropic"
HIGH = "claude-opus-5-5-effort-high-anthropic-maxtokens-20000"
LEGS = (LOW, HIGH)
RAMPNET = "rampnet"
RAMPNET_OP = SB.DEPLOYED_THRESHOLD
RADIUS = 0.022
#: The four city splits outside the US pool.
OTHER_SPLITS = ("budapest_district5", "sao_paulo", "bayonne", "laurens_gsv")
CITY_SPLITS = tuple(POOLED_SPLITS) + OTHER_SPLITS
SPLITS = CITY_SPLITS
N_BOOT = 10_000
SEED = 0


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def interval(diffs, point):
    """Point estimate, percentile interval and share of draws above zero for sorted diffs."""
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return {"diff": _r(point), "ci95": [_r(lo), _r(hi)], "ci_excludes_zero": lo > 0 or hi < 0,
            "share_draws_above_zero": _r(sum(d > 0 for d in diffs) / len(diffs))}


def f1_of(rows, panos):
    return prf(*totals(rows, panos))[2]


def paired(rows_a, rows_b, panos, rng):
    """F1(a) - F1(b) on the same pano resamples: point estimate and interval."""
    n = len(panos)
    diffs = []
    for _ in range(N_BOOT):
        draw = [panos[rng.randrange(n)] for _ in panos]
        diffs.append(f1_of(rows_a, draw) - f1_of(rows_b, draw))
    diffs.sort()
    return interval(diffs, f1_of(rows_a, panos) - f1_of(rows_b, panos))


def load_split(split):
    """(rows per model at r=0.022, scored panos, emitted boxes per leg) for one split."""
    records, gts = SB.load_split(split)
    assert all(g.fn_confirmed for g in gts.values()), f"{split}: unconfirmed recall panos"
    preds = {RAMPNET: {pid: [p for p in records[pid]["detections"]
                             if prediction_confidence(p) >= RAMPNET_OP] for pid in gts}}
    for leg in LEGS:
        d = load_detections(leg, split, publish_as=leg)
        assert d is not None, (leg, split)
        preds[leg] = d
    for leg in LEGS:
        assert set(gts) <= set(preds[leg]), f"{split}: {leg} is missing panos"
    panos = sorted(gts)
    rows = {m: per_pano(preds[m], gts, RADIUS) for m in preds}
    emitted = {leg: sum(len(preds[leg].get(pid, [])) for pid in panos) for leg in LEGS}
    return rows, panos, emitted


def ledger_cost():
    """Per (leg, split): cost, panos called, calls and thinking tokens from the ledger rows."""
    out = {}
    for r in read_rows(LEDGER) or []:
        sig = r.get("signature") or {}
        if r.get("model_id") != "claude-opus-5-5" or not r.get("panos_called"):
            continue
        if sig.get("effort") == "low" and "max_tokens" not in sig:
            leg = LOW
        elif sig.get("effort") == "high" and sig.get("max_tokens") == 20000:
            leg = HIGH
        else:
            continue
        a = out.setdefault((leg, r["bundle"]), {"usd": 0.0, "panos": 0, "calls": 0, "thinking": 0})
        a["usd"] += r.get("est_cost_usd") or 0
        a["panos"] += r["panos_called"]
        a["calls"] += r.get("calls") or 0
        a["thinking"] += r.get("thoughts_tokens") or 0
    return out


def pooled(per, splits, seed):
    """Micro-pooled F1 for each leg, high minus low with a stratified bootstrap."""
    keyed = {m: {} for m in (LOW, HIGH, RAMPNET)}
    strata = []
    for s in splits:
        rows, panos = per[s]
        for m in keyed:
            keyed[m].update({(s, p): rows[m][p] for p in panos})
        strata.append([(s, p) for p in panos])
    allp = [k for st in strata for k in st]
    rng = random.Random(seed)
    diffs = []
    for _ in range(N_BOOT):
        draw = [st[rng.randrange(len(st))] for st in strata for _ in st]
        diffs.append(f1_of(keyed[HIGH], draw) - f1_of(keyed[LOW], draw))
    diffs.sort()
    # The scoreboard's headline pooled F1 is the MACRO mean of per-split F1s; the interval
    # is on the micro (count-pooled) F1. Both are recorded so either can be quoted.
    macro = {m: _r(sum(f1_of(per[s][0][m], per[s][1]) for s in splits) / len(splits), 6)
             for m in keyed}
    return {"splits": list(splits), "n_panos": len(allp),
            "scores": {m: cell(keyed[m], allp) for m in keyed},
            "macro_f1": macro, "macro_high_minus_low": _r(macro[HIGH] - macro[LOW]),
            "high_minus_low": interval(diffs, f1_of(keyed[HIGH], allp) - f1_of(keyed[LOW], allp))}


def build():
    with open(SCOREBOARD, encoding="utf-8") as fh:
        board = json.load(fh)["per_split"]
    cost = ledger_cost()
    out = {"splits": list(SPLITS), "legs": list(LEGS), "radius": RADIUS,
           "rampnet_operating_point": RAMPNET_OP,
           "bootstrap": {"unit": "pano", "n_draws": N_BOOT, "seed": SEED,
                         "rng": "random.Random(SEED), one per split and one per pool; pools "
                                "resample within each split",
                         "interval": "percentile 2.5 / 97.5"},
           "per_split": {}}
    per = {}
    for split in SPLITS:
        rows, panos, emitted = load_split(split)
        per[split] = (rows, panos)
        cells = {m: cell(rows[m], panos) for m in rows}
        sb = board[LOW][split]
        assert abs(cells[LOW]["f1"] - sb["f1"]) < 1e-5, (split, sb["f1"])
        assert (cells[LOW]["tp"], cells[LOW]["fp"]) == (sb["tp"], sb["fp"]), split
        rng = random.Random(SEED)
        res = {"n_panos": len(panos), "scores": cells,
               "boxes_per_pano": {leg: _r(emitted[leg] / len(panos), 3) for leg in LEGS},
               "high_minus_low": paired(rows[HIGH], rows[LOW], panos, rng),
               "rampnet_minus_high": paired(rows[RAMPNET], rows[HIGH], panos, rng)}
        c = {}
        for leg in LEGS:
            a = cost.get((leg, split))
            if a and a["panos"]:
                c[leg] = {"usd": _r(a["usd"], 2), "panos_called": a["panos"],
                          "usd_per_pano": _r(a["usd"] / a["panos"]),
                          "thinking_per_call": _r(a["thinking"] / a["calls"], 1) if a["calls"] else None}
        res["cost"] = c
        out["per_split"][split] = res
    out["pooled_us"] = pooled(per, POOLED_SPLITS, SEED)
    out["pooled_other"] = pooled(per, OTHER_SPLITS, SEED + 1)
    out["pooled_all_cities"] = pooled(per, CITY_SPLITS, SEED + 2)
    hi = [v["cost"].get(HIGH, {}) for v in out["per_split"].values()]
    out["high_total_usd"] = _r(sum(h.get("usd", 0) for h in hi), 2)
    blob = json.dumps(out["per_split"], sort_keys=True).encode()
    out["per_split_sha256"] = hashlib.sha256(blob).hexdigest()
    return out


def markdown(res):
    lines = ["| Split | Low P / R / **F1** | High P / R / **F1** | High - low [95% CI] "
             "| Boxes/pano low -> high | RampNet - high [95% CI] |",
             "|---|---|---|---|---|---|"]

    def prf3(c):
        return f"{c['precision']:.3f} / {c['recall']:.3f} / **{c['f1']:.3f}**"

    for s in res["splits"]:
        v = res["per_split"][s]
        hl, rh, b = v["high_minus_low"], v["rampnet_minus_high"], v["boxes_per_pano"]
        lines.append(f"| {s} | {prf3(v['scores'][LOW])} | {prf3(v['scores'][HIGH])} "
                     f"| {hl['diff']:+.3f} [{hl['ci95'][0]:+.3f}, {hl['ci95'][1]:+.3f}] "
                     f"| {b[LOW]:.2f} -> {b[HIGH]:.2f} "
                     f"| {rh['diff']:+.3f} [{rh['ci95'][0]:+.3f}, {rh['ci95'][1]:+.3f}] |")
    lines += ["", "| Pool | Panos | Low P / R / **F1** (micro) | High P / R / **F1** (micro) "
              "| High - low, micro [95% CI] | Macro F1 low -> high |",
              "|---|---:|---|---|---|---|"]
    for key, label in (("pooled_us", "9 US cities"), ("pooled_other", "4 other cities"),
                       ("pooled_all_cities", "all 13 cities")):
        p = res[key]
        hl = p["high_minus_low"]
        lines.append(f"| {label} | {p['n_panos']} | {prf3(p['scores'][LOW])} "
                     f"| {prf3(p['scores'][HIGH])} "
                     f"| {hl['diff']:+.3f} [{hl['ci95'][0]:+.3f}, {hl['ci95'][1]:+.3f}] "
                     f"| {p['macro_f1'][LOW]:.3f} -> {p['macro_f1'][HIGH]:.3f} "
                     f"({p['macro_high_minus_low']:+.3f}) |")
    lines += ["", "| Split | Low $/pano | High $/pano | Low thinking/call | High thinking/call |",
              "|---|---:|---:|---:|---:|"]
    for s in res["splits"]:
        v = res["per_split"][s]
        lo, hi = v["cost"].get(LOW, {}), v["cost"].get(HIGH, {})

        def fmt(d, k, f):
            return f.format(d[k]) if d.get(k) is not None else "-"
        lines.append(f"| {s} | {fmt(lo, 'usd_per_pano', '{:.4f}')} | {fmt(hi, 'usd_per_pano', '{:.4f}')} "
                     f"| {fmt(lo, 'thinking_per_call', '{:.0f}')} | {fmt(hi, 'thinking_per_call', '{:.0f}')} |")
    return "\n".join(lines)


FIGURE = os.path.join(REPO, "docs", "figures", "claude_effort_high_forest.png")


def figures():
    """The doc's takeaway figure, drawn from the committed results.json (no recompute):
    high minus low F1 per split and per pool, with the paired 95% intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUT, encoding="utf-8") as fh:
        res = json.load(fh)
    # res["splits"] keeps the run order (US pool, then the other four); the per_split
    # dict was written with sorted keys, so it must not set the order.
    rows = [(s, res["per_split"][s]["high_minus_low"], False) for s in res["splits"]]
    rows += [(label, res[key]["high_minus_low"], True)
             for key, label in (("pooled_us", "Pooled: 9 US cities"),
                                ("pooled_other", "Pooled: 4 other cities"),
                                ("pooled_all_cities", "Pooled: all 13 cities"))]
    ink, muted, grid = "#1f2328", "#59636e", "#d0d7de"
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    fig.patch.set_facecolor("white")
    n = len(rows)
    for i, (label, iv, is_pool) in enumerate(rows):
        y = n - 1 - i + (0 if is_pool else 1)          # a blank row between splits and pools
        lo, hi = iv["ci95"]
        ax.plot([lo, hi], [y, y], color=ink, lw=2, solid_capstyle="round")
        ax.plot(iv["diff"], y, marker="D" if is_pool else "o", ms=9 if is_pool else 7,
                color=ink, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.text(0.125, y, f"{iv['diff']:+.3f}", va="center", ha="right", fontsize=8.5, color=muted)
    ax.axvline(0, color=muted, lw=1, ls="--", zorder=0)
    ticks = [n - 1 - i + (0 if p else 1) for i, (_, _, p) in enumerate(rows)]
    ax.set_yticks(ticks, [r[0] for r in rows], fontsize=9, color=ink)
    ax.set_xlim(-0.11, 0.13)
    ax.set_xlabel("F1 at effort high minus F1 at effort low (paired pano bootstrap, 95% CI)",
                  fontsize=9, color=ink)
    ax.set_title("Claude Opus 5.5: effort=high vs effort=low, same rig\n"
                 "left of the dashed line = high effort scored worse", fontsize=10, color=ink,
                 loc="left")
    ax.grid(axis="x", color=grid, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=muted, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURE, facecolor="white")
    print(f"wrote {os.path.relpath(FIGURE, REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the committed JSON")
    ap.add_argument("--check", action="store_true", help="compare against it (the default)")
    ap.add_argument("--markdown", action="store_true", help="print the doc's tables")
    ap.add_argument("--figures", action="store_true",
                    help="redraw docs/figures/claude_effort_high_forest.png from results.json")
    args = ap.parse_args()
    if args.figures:
        figures()
        return 0
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
