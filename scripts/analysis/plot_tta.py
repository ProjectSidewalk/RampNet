"""The flip-TTA figure for issue #78: what TTA buys at the operating points.

Two panels over the pooled five US/VA splits, one line per **arm** (single-pass
deployment path vs the horizontal-flip-TTA composition the paper's evaluation
used), colour carrying arm identity in both panels:

- **Left, the PR curve per arm.** The TTA curve sits above-right of the
  single-pass curve; the deployed 0.55 (hollow) and recommended 0.30 (filled)
  operating points are marked on both, so the four corners of the #78 decision
  are visible as four dots.
- **Right, recall and precision against threshold.** Solid = recall, dashed =
  precision, colour = arm. This is the overlap picture: TTA's recall curve is
  roughly the single-pass curve shifted left, which is why its marginal value
  shrinks once the threshold drops.

Only two series, so the two-slot prefix of the validated categorical palette is
used in fixed order (single-pass first — it is the incumbent). Per-split numbers
are printed to stdout and live in ``analysis_out/op/tta_compare.csv`` (the table
view); the figure keeps the pooled decision.

Reads the committed caches only — CPU, no GPU, no imagery:

    python scripts/analysis/plot_tta.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import radius_sq_for  # noqa: E402

from low_floor_sweep import (  # noqa: E402
    CACHE_DIR, DEPLOYED_THRESHOLD, TTA_CACHE_DIR, US_SPLITS, _load_tta_arm,
    load_split, row_at, sweep_rows, threshold_grid)

CANDIDATE = 0.30

# Validated categorical slots 1-2, fixed order: the incumbent single-pass arm
# first, the challenger TTA arm second (same slots as plot_operating_point.py).
ARM_COLOR = {"single-pass": "#2a78d6", "flip-TTA": "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def collect(cities=US_SPLITS, cache_dir=CACHE_DIR, tta_cache_dir=TTA_CACHE_DIR):
    """{split: {"single-pass": rows, "flip-TTA": rows}} + the pooled entry."""
    radius_sq = radius_sq_for()
    grid = threshold_grid(0.05, 0.90)
    singles, ttas, out = {}, {}, {}
    for city in cities:
        single, _ = load_split(city, cache_dir)
        tta, why = _load_tta_arm(city, single, tta_cache_dir, 0.05)
        if tta is None:
            raise SystemExit(f"[{city}] TTA arm unavailable: {why}")
        singles[city], ttas[city] = single, tta
        out[city] = {"single-pass": sweep_rows(single, grid, radius_sq),
                     "flip-TTA": sweep_rows(tta, grid, radius_sq)}
    out["POOLED"] = {
        "single-pass": sweep_rows([p for c in cities for p in singles[c]],
                                  grid, radius_sq),
        "flip-TTA": sweep_rows([p for c in cities for p in ttas[c]],
                               grid, radius_sq)}
    return out


def build(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pooled = curves["POOLED"]
    n_panos = pooled["single-pass"][0]["n_panos"]
    s_cand = row_at(pooled["single-pass"], CANDIDATE)
    t_cand = row_at(pooled["flip-TTA"], CANDIDATE)
    d_r, d_p = t_cand["recall"] - s_cand["recall"], t_cand["precision"] - s_cand["precision"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.4))
    fig.patch.set_facecolor("white")

    # ---- Left: PR curve per arm -------------------------------------------- #
    for arm, rows in pooled.items():
        ax1.plot([r["recall"] for r in rows], [r["precision"] for r in rows],
                 color=ARM_COLOR[arm], lw=2.4, zorder=3, solid_capstyle="round")
        dep, cand = row_at(rows, DEPLOYED_THRESHOLD), row_at(rows, CANDIDATE)
        ax1.plot([dep["recall"]], [dep["precision"]], "o", ms=7.5, mfc="white",
                 mec=ARM_COLOR[arm], mew=2, zorder=6)
        ax1.plot([cand["recall"]], [cand["precision"]], "o", ms=7.5,
                 color=ARM_COLOR[arm], mec="white", mew=1.4, zorder=6)
    ax1.set_xlabel("recall", fontsize=9.5, color=INK_MUTED)
    ax1.set_ylabel("precision", fontsize=9.5, color=INK_MUTED)
    ax1.set_title("PR response per arm, pooled US splits",
                  fontsize=11.5, color=INK, loc="left", pad=10)
    ax1.set_xlim(0.55, 1.0)
    ax1.set_ylim(0.55, 1.005)
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=7.5, mfc="white", mec=INK_MUTED, mew=2,
               label=f"deployed {DEPLOYED_THRESHOLD:.2f}"),
        Line2D([], [], marker="o", ls="", ms=7.5, color=INK_MUTED, mec="white",
               label=f"recommended {CANDIDATE:.2f}"),
    ], fontsize=8.4, frameon=False, loc="lower left")

    # ---- Right: recall (solid) and precision (dashed) vs threshold --------- #
    for arm, rows in pooled.items():
        thr = [r["threshold"] for r in rows]
        ax2.plot(thr, [r["recall"] for r in rows], color=ARM_COLOR[arm], lw=2.4,
                 zorder=3)
        ax2.plot(thr, [r["precision"] for r in rows], color=ARM_COLOR[arm], lw=2,
                 ls=(0, (5, 2)), zorder=3)
    for x, lab in ((DEPLOYED_THRESHOLD, "deployed"), (CANDIDATE, "recommended")):
        ax2.axvline(x, color=INK_MUTED, lw=1, ls=":", zorder=2)
        ax2.annotate(f"{lab} {x:.2f}", (x, 1.007), ha="center", va="bottom",
                     fontsize=8.2, color=INK_MUTED, zorder=6)
    # Direct labels for the linestyle channel, in ink (text never wears series colour).
    rows = pooled["flip-TTA"]
    lo = row_at(rows, 0.08)
    ax2.annotate("recall", (0.08, lo["recall"] + 0.012), fontsize=8.6,
                 color=INK_MUTED, zorder=6)
    ax2.annotate("precision", (0.08, lo["precision"] - 0.03), fontsize=8.6,
                 color=INK_MUTED, zorder=6)
    ax2.set_xlabel("peak threshold", fontsize=9.5, color=INK_MUTED)
    ax2.set_ylabel("recall / precision", fontsize=9.5, color=INK_MUTED)
    ax2.set_title("TTA's recall curve is the single-pass curve shifted left",
                  fontsize=11.5, color=INK, loc="left", pad=10)
    # The panel ends at 0.80: pooled recall still >= 0.49 there, while the 0.85-0.90
    # tail plunges to 0.23 and would either clip through the frame or compress the
    # operating region nobody decides in.
    ax2.set_xlim(0.05, 0.80)
    ax2.set_ylim(0.45, 1.04)

    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.6, colors=INK_MUTED)

    handles = [Line2D([], [], color=ARM_COLOR[a], lw=2.4, label=a)
               for a in ("single-pass", "flip-TTA")]
    handles.append(Line2D([], [], color=INK_MUTED, lw=2, ls=(0, (5, 2)),
                          label="precision (dashed)"))
    fig.legend(handles=handles, fontsize=8.8, frameon=False,
               loc="center left", bbox_to_anchor=(0.878, 0.55),
               title="arm", title_fontsize=9)

    fig.suptitle(f"After the 0.55 → 0.30 threshold drop, flip-TTA adds "
                 f"{d_r * 100:+.1f} recall points for {d_p * 100:+.1f} precision "
                 f"points — at 2× GPU per pano",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.012,
             f"Both arms extracted at a 0.05 floor over the {n_panos} pooled US "
             "benchmark panos (min_distance 10); threshold swept post-hoc on "
             "identical GT. TTA = max(original, mirrored) heatmaps, exactly\n"
             "stage_two/evaluate.py's composition. Sub-0.55 precision is a lower "
             "bound on these splits (GT reviewed at the deployed floor — see "
             "docs/operating_point.md); the bias is arm-symmetric.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.050, right=0.872, top=0.885, bottom=0.165,
                        wspace=0.185)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    curves = collect()
    path = build(curves, os.path.join(REPO, "docs", "figures", "tta_operating_point.png"))
    print(f"wrote {path}")
    for group, arms in curves.items():
        for arm, rows in arms.items():
            dep, cand = row_at(rows, DEPLOYED_THRESHOLD), row_at(rows, CANDIDATE)
            print(f"  {group:<14} {arm:<12} 0.55: P {dep['precision']:.3f} "
                  f"R {dep['recall']:.3f}   0.30: P {cand['precision']:.3f} "
                  f"R {cand['recall']:.3f}")


if __name__ == "__main__":
    main()
