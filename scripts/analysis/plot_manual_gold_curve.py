"""Figure for the `manual_gold` half of the Stage 2 epoch curve (issue #84).

The companion to `plot_epoch_curve.py`, which draws the auto-label half. Two panels:

- **Left — is there a peak at all?** `manual_gold` F1 across the 8 epochs, at the #54
  protocol point (0.30) and calibration-free (max-F1). The pre-registered **0.01 tie bar**
  is drawn as a band below the maximum, because the whole answer is which epochs fall
  outside it: exactly one does. Without the band this panel reads as "epoch 6 wins", which
  is the wrong conclusion.
- **Right — the two signals disagree about what happens after epoch 5.** Auto-label
  validation loss and human F1, each expressed as *percent worse than its own optimum*.
  Indexing to a common base is what makes them comparable on one axis; plotting a loss and
  an F1 against two y-scales would be a dual-axis chart, which lies about relative
  magnitude. The point is the difference in swing: auto-val moves 13.5 points, human F1
  moves 1.2.

Colour carries the *signal*, consistently across both panels: blue = the human measurement
(`manual_gold`), orange = the auto-label measurement. Within the left panel the two human
readings share the blue hue and are separated by line style, because they are two readings
of one thing rather than two identities. Validated pair — `node scripts/validate_palette.js
"#2a78d6,#eb6834" --mode light` passes every check, worst adjacent CVD ΔE 24.7.

Reads committed data only — the derived `manual_gold` summary and the committed event
files. No GPU, no cluster access:

    python scripts/analysis/plot_manual_gold_curve.py
"""
import csv
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_epoch_curve import (  # noqa: E402
    STEPS_PER_EPOCH, VAL_TAG, read_curve)

# Identity is the SIGNAL, not the run: blue = measured against human gold labels, orange =
# measured against Stage 1's automatic labels. Same meaning in both panels.
C_HUMAN = "#2a78d6"
C_AUTO = "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

SUMMARY = Path(REPO) / "docs" / "data" / "run_a_84_manual_gold" / "summary.csv"
EVENTS_DIR = Path(REPO) / "stage_two" / "run_a_84_events"

# Pre-registered in #84: counting noise alone is ~0.008 s.e. on recall over 3,919 gold
# instances, so differences below this are not differences.
TIE_BAR = 0.01

# The epoch the released Stage 2 checkpoint was taken at.
RELEASED_EPOCH = 1


def read_summary(path=SUMMARY):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run stage2_manual_gold_curve.py first")
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["epoch"])] = {k: (float(v) if v else None)
                                       for k, v in row.items()
                                       if k not in ("epoch", "checkpoint_fingerprint")}
    return rows


def build(summary, val_curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    epochs = sorted(summary)
    f1_protocol = [summary[e]["f1_at_protocol"] for e in epochs]
    f1_max = [summary[e]["max_f1"] for e in epochs]

    best_protocol_epoch = max(epochs, key=lambda e: summary[e]["f1_at_protocol"])
    best_protocol = summary[best_protocol_epoch]["f1_at_protocol"]
    best_max_epoch = max(epochs, key=lambda e: summary[e]["max_f1"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.9),
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.21})
    fig.patch.set_facecolor("white")

    # ---- Left: the curve, with the tie bar that decides how to read it -------- #
    # Drawn first and behind: the band is the interpretive frame, not an annotation.
    ax1.axhspan(best_protocol - TIE_BAR, best_protocol, color=C_HUMAN, alpha=0.10, zorder=1)
    ax1.axhline(best_protocol, color=C_HUMAN, lw=1.0, ls=":", zorder=2)

    ax1.plot(epochs, f1_max, "--o", color=C_HUMAN, lw=1.8, ms=7, mfc="white",
             markeredgewidth=1.8, zorder=4)
    ax1.plot(epochs, f1_protocol, "-o", color=C_HUMAN, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=5)

    # The one epoch outside the band is the entire finding, so it is marked, not left
    # to the reader to notice.
    ax1.plot([RELEASED_EPOCH], [summary[RELEASED_EPOCH]["f1_at_protocol"]], "o", ms=17,
             mfc="none", color=C_AUTO, markeredgewidth=2.2, zorder=6)
    ax1.annotate(
        "epoch 1 — the released checkpoint —\nis the ONLY epoch below the tie bar",
        xy=(RELEASED_EPOCH, summary[RELEASED_EPOCH]["f1_at_protocol"]),
        xytext=(1.75, 0.9035), fontsize=8.8, color=INK, va="top", ha="left",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9, shrinkA=3, shrinkB=11))
    # Anchored to the SOLID series' maximum, and said so -- the dashed max-F1 series rides
    # above the band, which would otherwise read as "better than the best".
    ax1.text(8.35, best_protocol - TIE_BAR + 0.0006,
             "shaded: 0.01 tie bar below the F1@0.30 max\n"
             "(the same one epoch falls outside on max-F1)",
             fontsize=8.2, color=INK_MUTED, va="bottom", ha="right", zorder=6)

    ax1.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax1.set_ylabel("`manual_gold` F1  (1,000 panos, 3,919 ramps)", fontsize=9, color=INK_MUTED)
    ax1.set_title("No resolvable peak — epochs 2–8 are all tied", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax1.set_xlim(0.55, 8.45)
    ax1.set_ylim(0.900, 0.9225)
    ax1.legend(handles=[
        Line2D([], [], marker="o", color=C_HUMAN, lw=2.2, ms=8, markeredgecolor="white",
               label=f"F1 @ conf 0.30  (peak: epoch {best_protocol_epoch})"),
        Line2D([], [], marker="o", color=C_HUMAN, lw=1.8, ls="--", ms=7, mfc="white",
               label=f"max-F1, calibration-free  (peak: epoch {best_max_epoch})"),
    ], fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2)

    # ---- Right: each signal against its own optimum -------------------------- #
    val_best = min(val_curve.values())
    auto_excess = [(val_curve[e] / val_best - 1.0) * 100.0 for e in epochs]
    human_deficit = [(best_protocol - summary[e]["f1_at_protocol"]) / best_protocol * 100.0
                     for e in epochs]

    ax2.plot(epochs, auto_excess, "-o", color=C_AUTO, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=4)
    ax2.plot(epochs, human_deficit, "-o", color=C_HUMAN, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=5)

    # The divergence is specifically after epoch 5, where auto-val turns up and human F1
    # does not follow. Shade that span rather than describing it.
    ax2.axvspan(4.6, 8.45, color=GRID, alpha=0.5, zorder=1)
    ax2.annotate(
        "after epoch 5 auto-val climbs 4.6 points\nwhile human F1 moves 0.6 — inside the tie bar",
        xy=(6.9, 3.4), xytext=(4.3, 10.2), fontsize=8.5, color=INK, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9, shrinkA=6, shrinkB=3))

    ax2.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax2.set_ylabel("percent worse than that signal's own optimum", fontsize=9, color=INK_MUTED)
    ax2.set_title("The two signals disagree about epochs 6–8", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax2.set_xlim(0.55, 8.45)
    ax2.set_ylim(-0.6, 15.0)
    ax2.legend(handles=[
        Line2D([], [], marker="o", color=C_AUTO, lw=2.2, ms=8, markeredgecolor="white",
               label="auto-label val loss"),
        Line2D([], [], marker="o", color=C_HUMAN, lw=2.2, ms=8, markeredgecolor="white",
               label="`manual_gold` F1 @ 0.30"),
    ], fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.5, colors=INK_MUTED)

    fig.suptitle("The selection signal reports an overfit that human-labelled quality cannot see",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.012,
             "Run A (#84): 8 checkpoints of the released Stage 2 recipe, scored on manual_gold — 1,000 panoramas, 3,919 ground-truth ramps, labelled with no model in the loop.\n"
             "Single-pass, match radius 0.022. The 0.01 tie bar is pre-registered: counting noise alone is ~0.008 s.e. on recall at this sample size.   "
             "Re-derives from committed data via scripts/analysis/plot_manual_gold_curve.py.\n"
             "Single-pass is the #54 protocol headline, but NOT what the paper and the July-2026 erratum used (both flip-TTA) — so these epochs are comparable to each other and deliberately not to the paper.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.070, right=0.995, top=0.870, bottom=0.245)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    summary = read_summary()
    val_curve = read_curve(EVENTS_DIR, VAL_TAG, STEPS_PER_EPOCH)
    path = build(summary, val_curve,
                 os.path.join(REPO, "docs", "figures", "stage2_manual_gold_curve_84.png"))
    print(f"wrote {path}")
    best_p = max(summary, key=lambda e: summary[e]["f1_at_protocol"])
    for epoch in sorted(summary):
        row = summary[epoch]
        gap = summary[best_p]["f1_at_protocol"] - row["f1_at_protocol"]
        flag = "  <- OUTSIDE tie bar" if gap > TIE_BAR else ""
        print(f"  epoch {epoch}  F1@0.30 {row['f1_at_protocol']:.4f}  "
              f"max-F1 {row['max_f1']:.4f}  gap {gap:.4f}{flag}")


if __name__ == "__main__":
    main()
