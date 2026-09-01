"""The canonical operating-point figure for issue #54: PR response per split.

This is the artifact #54 asks for — precision/recall as a function of the peak
threshold, one line per split, with the deployed 0.55 point and the recommended
0.30 point marked — in two panels:

- **Left, the PR curve.** Precision against recall, traced by sweeping the
  threshold. This is the standard form for "what does the trade actually look
  like", and it makes the shape of the trade visible in a way a table cannot: the
  curves are nearly flat over the operating region, so moving the threshold buys
  recall at a shallow precision cost until it doesn't.
- **Right, F1 against threshold.** The same sweep read the other way, which is
  what shows *how little* F1 discriminates here — it varies by under 0.01 across
  0.25–0.55 pooled, so the choice of operating point cannot rest on F1 alone.

Both panels colour by split, consistently, so a line is the same entity in both.
The five US/VA splits are solid; ``budapest_district5`` and ``manual_gold`` are
dashed because they are held out of the pooled recommendation for different
reasons (single-rater GT; in-distribution control). Line style carries the
held-out status so colour is free to carry identity alone.

Colours are the validated categorical slots in fixed order. Three of them sit below
3:1 on a light surface, so the palette's relief rule applies: it is discharged by
the **table view** (``analysis_out/op/low_floor_sweep.csv`` holds every plotted
number, and ``docs/operating_point.md`` tabulates the operating points) plus a
shared legend, rather than by per-line labels. Direct labels were tried first and
removed — in the PR panel the splits converge into a cluster around
(0.80, 0.88) where seven labels cannot be placed without overlapping, which is a
worse accessibility outcome than the legend.

Reads the committed caches only — CPU, no GPU, no imagery:

    python scripts/analysis/plot_operating_point.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import radius_sq_for  # noqa: E402

from low_floor_sweep import (  # noqa: E402
    ALL_SPLITS, CACHE_DIR, DEPLOYED_THRESHOLD, US_SPLITS, best_f1_row, load_split,
    pool_of, row_at, sweep_rows, threshold_grid)

RECOMMENDED = 0.30

# Validated categorical slots, fixed order (never cycled). All 8 slots were taken at
# paterson; gainesville (the 9th split) forces the documented fold — see
# docs/adding_a_benchmark_city.md: the two held-out reference splits drop to neutral
# ink (identity carried by dash pattern + legend, matching their held-out status),
# vacating slots 6/7; gainesville takes slot 6. The remaining 7-hue set re-validates
# in this order (validate_palette.js: ALL PASS; the new green↔red adjacency is ΔE 7.2
# protan — inside the 6–8 band that is legal only with secondary encoding, provided
# here by the legend, the operating-point dots, and the CSV table view, the same
# relief that already covers the three sub-3:1-contrast slots). sao_paulo (the 10th
# split, held out like budapest/manual_gold) joins the neutral-ink group with its
# own dash — no categorical slot consumed.
SERIES = {
    "richmond": "#2a78d6",
    "bend": "#eb6834",
    "clovis": "#1baf7a",
    "morgantown": "#eda100",
    "annapolis": "#e87ba4",
    "gainesville": "#008300",
    "paterson": "#e34948",
    "laurens_mapillary": "#4a3aa7",    # slot 7, vacated by the manual_gold fold
    "laurens_gsv": "#52514e",          # neutral ink — held out, not a categorical slot
    "budapest_district5": "#52514e",   # neutral ink — held out, not a categorical slot
    "sao_paulo": "#52514e",            # neutral ink — held out, not a categorical slot
    "manual_gold": "#52514e",          # neutral ink — held out, not a categorical slot
}
POOLED_COLOR = "#0b0b0b"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# The neutral-ink held-out splits share a hue, so each carries its own dash.
HELD_DASH = {"budapest_district5": (0, (5, 2)), "manual_gold": (0, (1, 1.6)),
             "sao_paulo": (0, (4, 1.4, 1, 1.4)), "laurens_gsv": (0, (2, 1.2))}
LABEL = {"budapest_district5": "budapest*", "manual_gold": "manual_gold†",
         "sao_paulo": "sao_paulo‡", "laurens_gsv": "laurens_gsv§"}


def collect(cities=ALL_SPLITS, cache_dir=CACHE_DIR):
    radius_sq = radius_sq_for()
    grid = threshold_grid(0.05, 0.90)
    loaded = {c: load_split(c, cache_dir)[0] for c in cities}
    out = {c: sweep_rows(p, grid, radius_sq) for c, p in loaded.items()}
    poolable = pool_of(cities)
    if len(poolable) > 1:
        out["POOLED"] = sweep_rows([pd for c in poolable for pd in loaded[c]],
                                   grid, radius_sq)
    # Counts for the legend/footer text, so adding a split can't strand a stale "5
    # US"/"1,625 panos" in the figure.
    meta = {"n_panos": sum(len(p) for p in loaded.values()),
            "n_pooled": len(poolable)}
    return out, meta


def build(curves, path, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.4))
    fig.patch.set_facecolor("white")

    order = [c for c in SERIES if c in curves]

    # ---- Left: the PR curve ------------------------------------------------ #
    for city in order:
        rows = curves[city]
        solid = city in US_SPLITS
        ax1.plot([r["recall"] for r in rows], [r["precision"] for r in rows],
                 color=SERIES[city], lw=2 if solid else 1.6,
                 ls="-" if solid else HELD_DASH[city], zorder=3, solid_capstyle="round")
    if "POOLED" in curves:
        rows = curves["POOLED"]
        ax1.plot([r["recall"] for r in rows], [r["precision"] for r in rows],
                 color=POOLED_COLOR, lw=3, zorder=4, solid_capstyle="round")

    # Operating points: deployed (hollow) vs recommended (filled), on every curve.
    for city in order + (["POOLED"] if "POOLED" in curves else []):
        rows = curves[city]
        colour = SERIES.get(city, POOLED_COLOR)
        dep, rec = row_at(rows, DEPLOYED_THRESHOLD), row_at(rows, RECOMMENDED)
        ax1.plot([dep["recall"]], [dep["precision"]], "o", ms=7.5, mfc="white",
                 mec=colour, mew=2, zorder=6)
        ax1.plot([rec["recall"]], [rec["precision"]], "o", ms=7.5, color=colour,
                 mec="white", mew=1.4, zorder=6)

    ax1.set_xlabel("recall", fontsize=9.5, color=INK_MUTED)
    ax1.set_ylabel("precision", fontsize=9.5, color=INK_MUTED)
    ax1.set_title("Precision–recall response to the peak threshold",
                  fontsize=11.5, color=INK, loc="left", pad=10)
    ax1.set_xlim(0.42, 1.0)
    ax1.set_ylim(0.55, 1.005)
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=7.5, mfc="white", mec=INK_MUTED, mew=2,
               label=f"deployed {DEPLOYED_THRESHOLD:.2f}"),
        Line2D([], [], marker="o", ls="", ms=7.5, color=INK_MUTED, mec="white",
               label=f"recommended {RECOMMENDED:.2f}"),
        Line2D([], [], color=POOLED_COLOR, lw=3,
               label=f"pooled ({meta['n_pooled']} US splits)"),
        Line2D([], [], color=INK_MUTED, lw=1.6, ls=(0, (5, 2)),
               label="held out of the pooled recommendation"),
    ], fontsize=8.4, frameon=False, loc="lower left")

    # ---- Right: F1 against threshold --------------------------------------- #
    ax2.axvspan(0.25, 0.40, color="#f0efec", zorder=0)
    for city in order:
        rows = curves[city]
        solid = city in US_SPLITS
        ax2.plot([r["threshold"] for r in rows], [r["f1"] for r in rows],
                 color=SERIES[city], lw=2 if solid else 1.6,
                 ls="-" if solid else HELD_DASH[city], zorder=3)
        best = best_f1_row(rows)
        ax2.plot([best["threshold"]], [best["f1"]], "o", ms=6, color=SERIES[city],
                 mec="white", mew=1.2, zorder=5)
    if "POOLED" in curves:
        rows = curves["POOLED"]
        ax2.plot([r["threshold"] for r in rows], [r["f1"] for r in rows],
                 color=POOLED_COLOR, lw=3, zorder=4)
        best = best_f1_row(rows)
        ax2.plot([best["threshold"]], [best["f1"]], "o", ms=7, color=POOLED_COLOR,
                 mec="white", mew=1.4, zorder=5)

    for x, lab in ((DEPLOYED_THRESHOLD, "deployed"), (RECOMMENDED, "recommended")):
        ax2.axvline(x, color=INK_MUTED, lw=1, ls=":", zorder=2)
        ax2.annotate(f"{lab} {x:.2f}", (x, 0.935), ha="center", va="bottom",
                     fontsize=8.2, color=INK_MUTED, zorder=6)
    ax2.annotate("pooled F1 varies by <0.01 across the shaded band —\n"
                 "F1 alone cannot pick the operating point",
                 (0.075, 0.345), ha="left", fontsize=8.3, color=INK_MUTED, zorder=6)
    ax2.annotate("• = each split's F1 optimum", (0.075, 0.315), ha="left",
                 fontsize=8.3, color=INK_MUTED, zorder=6)

    ax2.set_xlabel("peak threshold", fontsize=9.5, color=INK_MUTED)
    ax2.set_ylabel("F1", fontsize=9.5, color=INK_MUTED)
    ax2.set_title("F1 is flat over the operating region", fontsize=11.5,
                  color=INK, loc="left", pad=10)
    ax2.set_xlim(0.05, 0.90)
    ax2.set_ylim(0.30, 0.95)

    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.6, colors=INK_MUTED)

    # One shared split legend outside both panels: colour means the same entity in
    # each, and the series are too similar in the PR panel for per-line labels to
    # land without collisions. The palette's relief rule is satisfied by the table
    # view — analysis_out/op/low_floor_sweep.csv holds every plotted number.
    handles = [Line2D([], [], color=SERIES[c], lw=2.4,
                      ls="-" if c in US_SPLITS else HELD_DASH[c],
                      label=LABEL.get(c, c)) for c in order]
    if "POOLED" in curves:
        handles.append(Line2D([], [], color=POOLED_COLOR, lw=3,
                              label=f"POOLED ({meta['n_pooled']} US)"))
    fig.legend(handles=handles, fontsize=8.8, frameon=False,
               loc="center left", bbox_to_anchor=(0.878, 0.55),
               title="split", title_fontsize=9)

    fig.suptitle("Lowering the peak threshold 0.55 → 0.30 buys ~7 pooled recall points "
                 "at a shallow precision cost — every split gains",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.012,
             f"RampNet peaks extracted at a 0.05 floor over {meta['n_panos']:,} benchmark panos "
             "(min_distance 10, no TTA); threshold swept post-hoc. Precision below 0.55 is a "
             "LOWER BOUND on the city splits — their GT was\nassembled from detections at or "
             "above the deployed floor, so a real ramp nobody marked scores as a false "
             "positive (see docs/operating_point.md; the #55 correction is applied "
             "separately).\n*budapest GT is single-rater, low confidence.   †manual_gold is "
             "in-distribution GSV with independent, un-anchored GT.   ‡sao_paulo is non-US "
             "(held out of the pooled recommendation; GT is high confidence).",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.050, right=0.872, top=0.885, bottom=0.165, wspace=0.185)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    curves, meta = collect()
    path = build(curves, os.path.join(REPO, "docs", "figures", "operating_point_pr.png"),
                 meta)
    print(f"wrote {path}")
    for city, rows in curves.items():
        dep, rec, best = (row_at(rows, DEPLOYED_THRESHOLD), row_at(rows, RECOMMENDED),
                          best_f1_row(rows))
        print(f"  {city:<20} 0.55: P {dep['precision']:.3f} R {dep['recall']:.3f}"
              f"   0.30: P {rec['precision']:.3f} R {rec['recall']:.3f}"
              f"   F1-max @ {best['threshold']:.2f}")


if __name__ == "__main__":
    main()
