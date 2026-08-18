"""Figure for the Stage 2 epoch curve (issue #84).

Two panels, because the replication question and the selection question want different
y-axes and a single one cannot carry both honestly:

- **Left — does Run A reproduce the paper run?** Both auto-label validation-loss curves on
  the same absolute axis, over the epochs both runs cover, with each run's minimum ringed.
  Two runs = two identities, so a categorical pair; Run A takes the blue slot because it is
  the new result and the paper run is the reference it is measured against.
- **Right — how much does the epoch choice actually move the number?** Run A's distance
  from its *own* minimum, per epoch, in percent. One series, so no legend — the title
  names it. This is the decision-relevant view: it shows the epoch 3–6 basin is shallow
  while epoch 1 (the epoch the released checkpoint was taken at) sits far outside it.
  The measured noise floor is drawn on the same axis, where it is invisible by two orders
  of magnitude — which is the point being made, not a rendering failure.

Every number is read from committed event files: Run A's own, and the paper run's rescued
events under `docs/data/rampnet1_stage2_run/`. Nothing here is transcribed by hand, so a
regenerated figure is checkable against the data rather than against a memory.

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
    STEPS_PER_EPOCH, VAL_TAG, read_curve, read_paper_curve, repeat_measurements,
    spread_pct)

# Validated categorical slots (light mode): blue = Run A (the new result), orange = the
# paper run (the reference). Identity, not rank -- the same hue means the same run in both
# panels.
C_RUN_A = "#2a78d6"
C_PAPER = "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

EVENTS_DIR = os.path.join(REPO, "stage_two", "run_a_84_events")

# The epoch the released Stage 2 checkpoint was taken at: best_model.pth is byte-identical
# to that run's epoch_1_step_9378.pth.
RELEASED_EPOCH = 1


def collect(events_dir=EVENTS_DIR):
    curve = read_curve(Path(events_dir), VAL_TAG, STEPS_PER_EPOCH)
    if not curve:
        raise SystemExit(f"no '{VAL_TAG}' scalars in {events_dir}")
    return curve


def build(curve, path, paper=None, repeats=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if paper is None:
        paper = read_paper_curve()
    if repeats is None:
        repeats = repeat_measurements()

    epochs = sorted(curve)
    best_epoch = min(curve, key=curve.get)
    best = curve[best_epoch]
    # Keyed by epoch, not by position: a curve that does not start at epoch 1, or that has
    # a hole in it, must not silently read a neighbour's value.
    excess = {epoch: (curve[epoch] / best - 1.0) * 100.0 for epoch in epochs}

    # Only the epochs both runs cover. The paper run reached 11 epochs and Run A stopped at
    # 8, so the left panel compares the overlap rather than plotting a ragged pair.
    shared = [epoch for epoch in epochs if epoch in paper]
    paper_best_epoch = min(paper, key=paper.get) if paper else None
    deltas = [abs(curve[e] / paper[e] - 1.0) * 100.0 for e in shared]

    # The measurement floor, from an epoch a requeue caused to be computed twice. Derived,
    # not transcribed -- if a future run has no requeue there is simply no floor to draw.
    floor_epoch = max(repeats, key=lambda e: spread_pct(repeats[e])) if repeats else None
    noise_pct = spread_pct(repeats[floor_epoch]) if floor_epoch else None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.9),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.20})
    fig.patch.set_facecolor("white")

    # ---- Left: both curves on the absolute axis ---------------------------- #
    # Scaled to 1e-4 so the axis reads in plain numbers instead of eight decimals.
    if shared:
        ax1.plot(shared, [paper[e] * 1e4 for e in shared], "-o", color=C_PAPER, lw=2, ms=8,
                 markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    ax1.plot(epochs, [curve[e] * 1e4 for e in epochs], "-o", color=C_RUN_A, lw=2, ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=4)

    # Ring each run's minimum. Both land on epoch 5 -- that is the headline, so it is
    # marked on the marks themselves rather than left to the reader to find.
    rings = [(best_epoch, best, C_RUN_A)]
    if paper_best_epoch in shared:
        rings.append((paper_best_epoch, paper[paper_best_epoch], C_PAPER))
    for xs, ys, colour in rings:
        ax1.plot([xs], [ys * 1e4], "o", ms=17, mfc="none", color=colour,
                 markeredgewidth=2.0, zorder=5)
    both = paper_best_epoch == best_epoch
    ax1.annotate(f"minimum, both runs — epoch {best_epoch}" if both
                 else f"Run A minimum — epoch {best_epoch}",
                 xy=(best_epoch, best * 1e4), xytext=(best_epoch + 0.85, 4.545),
                 fontsize=8.8, color=INK, va="bottom", ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                 shrinkA=3, shrinkB=11))
    if RELEASED_EPOCH in curve:
        ax1.annotate("epoch the released\ncheckpoint was taken at",
                     xy=(RELEASED_EPOCH, curve[RELEASED_EPOCH] * 1e4), xytext=(2.15, 5.215),
                     fontsize=8.8, color=INK_MUTED, va="top", ha="left",
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                     shrinkA=3, shrinkB=9))

    ax1.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax1.set_ylabel("auto-label validation loss  ($\\times 10^{-4}$)", fontsize=9,
                   color=INK_MUTED)
    ax1.set_title("The curve replicates, minimum included", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax1.set_xlim(min(epochs) - 0.45, max(epochs) + 0.45)
    # The tuned window for this run, widened only if a future curve would fall outside it
    # -- silently clipping a point would misrepresent the comparison.
    plotted = [curve[e] for e in epochs] + [paper[e] for e in shared]
    ax1.set_ylim(min(4.50, min(plotted) * 1e4 - 0.05),
                 max(5.32, max(plotted) * 1e4 + 0.05))
    ax1.legend(handles=[
        Line2D([], [], marker="o", color=C_RUN_A, lw=2, ms=8, markeredgecolor="white",
               label="Run A  (2026-08, this run)"),
        Line2D([], [], marker="o", color=C_PAPER, lw=2, ms=8, markeredgecolor="white",
               label="paper run  (2025-06, rescued events)"),
    ], fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.155),
        ncol=2)

    # ---- Right: distance from Run A's own minimum -------------------------- #
    bars = ax2.bar(epochs, [excess[e] for e in epochs], width=0.62, color=C_RUN_A, zorder=3)
    for epoch, bar in zip(epochs, bars):
        value = excess[epoch]
        if epoch == best_epoch:
            ax2.text(epoch, 0.22, "min", ha="center", va="bottom", fontsize=8.6,
                     color=INK, fontweight="bold", zorder=4)
        else:
            ax2.text(epoch, value + 0.22, f"+{value:.1f}%", ha="center", va="bottom",
                     fontsize=8.6, color=INK if value > 3 else INK_MUTED, zorder=4)

    # The basin. Drawn as a span rather than described, because "epochs 3-6 are within
    # 1.9% of each other" is a statement about a region.
    basin = [e for e in epochs if excess[e] <= 2.0]
    peak = max(excess.values())
    ax2.axvspan(min(basin) - 0.45, max(basin) + 0.45, color=GRID, alpha=0.45, zorder=1)
    ax2.text((min(basin) + max(basin)) / 2, peak * 1.14,
             f"epochs {min(basin)}–{max(basin)} all within "
             f"{max(excess[e] for e in basin):.1f}%",
             ha="center", va="top", fontsize=8.8, color=INK_MUTED, zorder=4)

    # The measured floor, drawn on the same axis, where it is ~1/1500 of the panel height
    # and therefore indistinguishable from the baseline. That is the finding, so it is
    # annotated rather than exaggerated onto a broken axis. Deliberately INK and not a
    # categorical hue: blue means Run A and orange means the paper run, in both panels.
    if noise_pct:
        # The tightest real gap inside the basin -- excluding the minimum itself, whose
        # excess is 0 by construction. This is the hardest thing the curve is asked to
        # resolve.
        inner = [excess[e] for e in basin if e != best_epoch]
        smallest_gap = min(inner) if inner else None
        ax2.axhline(noise_pct, color=INK, lw=1.5, zorder=5)
        ratio = (f"two resume paths). Drawn to scale: it sits on\nthe baseline, "
                 f"{smallest_gap / noise_pct:.0f}× below the basin's smallest gap."
                 if smallest_gap else "two resume paths). Drawn to scale.")
        ax2.annotate(
            f"measurement floor {noise_pct:.3f}%\n"
            f"(epoch {floor_epoch} computed twice — two nodes,\n"
            f"{ratio}",
            xy=(4.5, noise_pct), xytext=(4.5, peak * 0.42),
            fontsize=8.4, color=INK_MUTED, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8, shrinkA=3, shrinkB=0))

    ax2.set_xlabel("training epoch", fontsize=9, color=INK_MUTED)
    ax2.set_ylabel("excess over Run A's own minimum  (%)", fontsize=9, color=INK_MUTED)
    ax2.set_title("How much the epoch choice is worth", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax2.set_xlim(min(epochs) - 0.55, max(epochs) + 0.55)
    ax2.set_ylim(0, peak * 1.20)

    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.5, colors=INK_MUTED)

    headline = f"Stage 2 validation loss bottoms at epoch {best_epoch}"
    if RELEASED_EPOCH in excess:
        headline += (f" — and the released checkpoint is "
                     f"{excess[RELEASED_EPOCH]:.1f}% above that minimum")
    fig.suptitle(headline, fontsize=13.5, color=INK, x=0.008, ha="left", y=0.982)
    # Three lines of roughly equal length, deliberately: a fourth grows upward into the
    # legend, and a line past ~230 characters runs off the right edge at this font size.
    agreement = (f"their agreement bounds what the unrecoverable June-2025 code could have "
                 f"changed (≤{max(deltas):.2f}% at any epoch), not "
                 f"{len(deltas)} confirmations.   " if deltas else "")
    fig.text(0.008, 0.012,
             "Run A (#84): the released Stage 2 recipe for 8 epochs instead of 1 — world size 16, constant lr 1e-5, seed 42, "
             "150,063 train / 42,875 val panoramas, 9,378 steps/epoch. Re-derive with\n"
             "scripts/analysis/plot_epoch_curve.py from the committed events in stage_two/run_a_84_events/ and docs/data/rampnet1_stage2_run/.   "
             "Both runs use seed 42 on the same data, so these are near-identical draws,\n"
             "not independent samples: " + agreement +
             "Auto-label half only — manual_gold F1 across the curve is not scored yet.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.062, right=0.995, top=0.870, bottom=0.245)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    curve = collect()
    paper = read_paper_curve()
    repeats = repeat_measurements()
    path = build(curve, os.path.join(REPO, "docs", "figures", "stage2_epoch_curve_84.png"),
                 paper=paper, repeats=repeats)
    print(f"wrote {path}")
    best_epoch = min(curve, key=curve.get)
    for epoch in sorted(curve):
        excess = (curve[epoch] / curve[best_epoch] - 1.0) * 100.0
        mark = "  <- min" if epoch == best_epoch else ""
        delta = f"  vs paper {(curve[epoch] / paper[epoch] - 1.0) * 100.0:+.3f}%" if epoch in paper else ""
        print(f"  epoch {epoch}  {curve[epoch]:.8f}  +{excess:.2f}%{delta}{mark}")
    for epoch, values in repeats.items():
        print(f"  epoch {epoch} computed {len(values)}x -- spread {spread_pct(values):.4f}%")


if __name__ == "__main__":
    main()
