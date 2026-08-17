"""Figure for the Stage 2 epoch curve (issue #84).

Two panels, because the replication question and the selection question want different
y-axes and a single one cannot carry both honestly:

- **Left — does Run A reproduce the paper run?** Both auto-label validation-loss curves on
  the same absolute axis, epochs 1–8, with each run's minimum ringed. Two runs = two
  identities, so a categorical pair; Run A takes the blue slot because it is the new
  result and the paper run is the reference it is measured against.
- **Right — how much does the epoch choice actually move the number?** Run A's distance
  from its *own* minimum, per epoch, in percent. One series, so no legend — the title
  names it. This is the decision-relevant view: it shows the epoch 3–6 basin is shallow
  while epoch 1 (the epoch the released checkpoint was taken at) sits far outside it.
  The measured noise floor is drawn on the same axis, where it is invisible by two orders
  of magnitude — which is the point being made, not a rendering failure.

Colors are the validated reference palette's categorical slots 1 and 2, the same pair
`plot_storage_floor.py` uses (`node scripts/validate_palette.js "#2a78d6,#eb6834"
--mode light` → all checks pass, worst adjacent CVD ΔE 24.7). Epoch 1 is called out with
a direct label rather than its own color, so blue means "Run A" in both panels.

Reads the committed event files only — CPU, no GPU, no cluster access:

    python scripts/analysis/plot_epoch_curve.py
"""
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_epoch_curve import (  # noqa: E402
    PAPER_RUN_VAL_LOSS, STEPS_PER_EPOCH, VAL_TAG, read_scalars)

# Validated categorical slots (light mode): blue = Run A (the new result), orange = the
# paper run (the reference). Identity, not rank -- the same hue means the same run in both
# panels.
C_RUN_A = "#2a78d6"
C_PAPER = "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

EVENTS_DIR = os.path.join(REPO, "stage_two", "run_a_84_events")

# Epoch 5 was computed twice, by two job incarnations on two nodes, because a requeue
# landed mid-epoch (see docs/stage2_epoch_curve_84.md). The spread between them bounds
# resume-path nondeterminism plus evaluation together.
EPOCH5_REPEATS = (0.00045980, 0.00045976)


def collect(events_dir=EVENTS_DIR):
    scalars = read_scalars(Path(events_dir), VAL_TAG)
    if not scalars:
        raise SystemExit(f"no '{VAL_TAG}' scalars in {events_dir}")
    curve = {round(step / STEPS_PER_EPOCH): value for step, value in sorted(scalars.items())}
    return curve


def build(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    epochs = sorted(curve)
    run_a = [curve[e] for e in epochs]
    paper = [PAPER_RUN_VAL_LOSS.get(e) for e in epochs]

    best_epoch = min(curve, key=curve.get)
    best = curve[best_epoch]
    paper_best_epoch = min(PAPER_RUN_VAL_LOSS, key=PAPER_RUN_VAL_LOSS.get)
    excess = [(curve[e] / best - 1.0) * 100.0 for e in epochs]
    noise_pct = abs(EPOCH5_REPEATS[0] / EPOCH5_REPEATS[1] - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.9),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.20})
    fig.patch.set_facecolor("white")

    # ---- Left: both curves on the absolute axis ---------------------------- #
    # Scaled to 1e-4 so the axis reads in plain numbers instead of eight decimals.
    ax1.plot(epochs, [v * 1e4 for v in paper], "-o", color=C_PAPER, lw=2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    ax1.plot(epochs, [v * 1e4 for v in run_a], "-o", color=C_RUN_A, lw=2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=4)

    # Ring each run's minimum. Both land on epoch 5 -- that is the headline, so it is
    # marked on the marks themselves rather than left to the reader to find.
    for xs, ys, colour in ((paper_best_epoch, PAPER_RUN_VAL_LOSS[paper_best_epoch], C_PAPER),
                           (best_epoch, best, C_RUN_A)):
        ax1.plot([xs], [ys * 1e4], "o", ms=17, mfc="none", color=colour,
                 markeredgewidth=2.0, zorder=5)
    ax1.annotate(f"minimum, both runs — epoch {best_epoch}",
                 xy=(best_epoch, best * 1e4), xytext=(best_epoch + 0.85, 4.545),
                 fontsize=8.8, color=INK, va="bottom", ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                 shrinkA=3, shrinkB=11))
    ax1.annotate("epoch the released\ncheckpoint was taken at",
                 xy=(1, run_a[0] * 1e4), xytext=(2.15, 5.215),
                 fontsize=8.8, color=INK_MUTED, va="top", ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                 shrinkA=3, shrinkB=9))

    ax1.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax1.set_ylabel("auto-label validation loss  ($\\times 10^{-4}$)", fontsize=9,
                   color=INK_MUTED)
    ax1.set_title("The curve replicates, minimum included", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax1.set_xlim(0.55, 8.45)
    ax1.set_ylim(4.50, 5.32)
    ax1.legend(handles=[
        Line2D([], [], marker="o", color=C_RUN_A, lw=2, ms=8, markeredgecolor="white",
               label="Run A  (2026-08, this run)"),
        Line2D([], [], marker="o", color=C_PAPER, lw=2, ms=8, markeredgecolor="white",
               label="paper run  (2025-06, 3 s.f.)"),
    ], fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.155),
        ncol=2)

    # ---- Right: distance from Run A's own minimum -------------------------- #
    bars = ax2.bar(epochs, excess, width=0.62, color=C_RUN_A, zorder=3)
    for epoch, bar, value in zip(epochs, bars, excess):
        if epoch == best_epoch:
            ax2.text(epoch, 0.22, "min", ha="center", va="bottom", fontsize=8.6,
                     color=INK, fontweight="bold", zorder=4)
        else:
            ax2.text(epoch, value + 0.22, f"+{value:.1f}%", ha="center", va="bottom",
                     fontsize=8.6, color=INK if value > 3 else INK_MUTED, zorder=4)

    # The basin. Drawn as a span rather than described, because "epochs 3-6 are within
    # 1.9% of each other" is a statement about a region.
    basin = [e for e in epochs if excess[e - 1] <= 2.0]
    # The tightest real gap inside the basin -- excluding the minimum itself, whose excess
    # is 0 by construction. This is the hardest thing the curve is asked to resolve.
    smallest_gap = min(excess[e - 1] for e in basin if e != best_epoch)
    ax2.axvspan(min(basin) - 0.45, max(basin) + 0.45, color=GRID, alpha=0.45, zorder=1)
    ax2.text((min(basin) + max(basin)) / 2, max(excess) * 1.14,
             f"epochs {min(basin)}–{max(basin)} all within "
             f"{max(excess[e - 1] for e in basin):.1f}%",
             ha="center", va="top", fontsize=8.8, color=INK_MUTED, zorder=4)

    # The measured floor, drawn on the same axis, where it is ~1/1500 of the panel height
    # and therefore indistinguishable from the baseline. That is the finding, so it is
    # annotated rather than exaggerated onto a broken axis. Deliberately INK and not a
    # categorical hue: blue means Run A and orange means the paper run, in both panels.
    ax2.axhline(noise_pct, color=INK, lw=1.5, zorder=5)
    ax2.annotate(
        f"measurement floor {noise_pct:.3f}%\n"
        f"(epoch {best_epoch} computed twice — two nodes,\n"
        "two resume paths). Drawn to scale: it sits on\n"
        f"the baseline, {smallest_gap / noise_pct:.0f}× below the basin's smallest gap.",
        xy=(4.5, noise_pct), xytext=(4.5, max(excess) * 0.42),
        fontsize=8.4, color=INK_MUTED, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8, shrinkA=3, shrinkB=0))

    ax2.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax2.set_ylabel("excess over Run A's own minimum  (%)", fontsize=9, color=INK_MUTED)
    ax2.set_title("How much the epoch choice is worth", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax2.set_xlim(0.45, 8.55)
    ax2.set_ylim(0, max(excess) * 1.20)

    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.5, colors=INK_MUTED)

    fig.suptitle("Stage 2 validation loss bottoms at epoch 5 — and the released checkpoint "
                 f"is {excess[0]:.1f}% above that minimum",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.012,
             "Run A (#84): the released Stage 2 recipe for 8 epochs instead of 1 — world size 16, constant lr 1e-5, seed 42, "
             "150,063 train / 42,875 val panoramas, 9,378 steps/epoch. Re-derive with\n"
             "scripts/analysis/plot_epoch_curve.py from the committed events in stage_two/run_a_84_events/.   "
             "Both runs use seed 42 on the same data, so these are near-identical draws rather than independent samples: "
             "the agreement bounds what the\nunrecoverable June-2025 code could have changed (≤1.7% at any epoch), it is not eight independent confirmations.   "
             "This is the auto-label half only — manual_gold F1 across the curve, the question #84 exists to answer, is not scored yet.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.062, right=0.995, top=0.870, bottom=0.245)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    curve = collect()
    path = build(curve, os.path.join(REPO, "docs", "figures", "stage2_epoch_curve_84.png"))
    print(f"wrote {path}")
    best_epoch = min(curve, key=curve.get)
    for epoch in sorted(curve):
        excess = (curve[epoch] / curve[best_epoch] - 1.0) * 100.0
        mark = "  <- min" if epoch == best_epoch else ""
        print(f"  epoch {epoch}  {curve[epoch]:.8f}  +{excess:.2f}%{mark}")


if __name__ == "__main__":
    main()
