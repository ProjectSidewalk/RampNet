"""Figure for the storage-floor / recall-ceiling question (issue #54; labeler#28, #27).

Two panels, because they answer two different questions and a single axis cannot
carry both honestly:

- **Left — what a 0.1 storage floor costs.** Per split, the count of ground-truth
  ramps whose *best* candidate falls in ``[0.05, 0.10)`` (discarded by the floor,
  and unrecoverable afterwards) beside those in ``[0.10, 0.20)`` (kept, but only
  because the floor is where it is). Two categories with a decision attached, so a
  categorical pair; the discarded band takes the warmer hue because it is the one
  under scrutiny.
- **Right — the recall ceiling.** Per split, recall at the deployed threshold, at
  the 0.1 storage floor, and at the 0.05 extraction floor. Thresholds are an
  *ordered* quantity, so this uses one hue stepped light→dark rather than three
  unrelated colors. The 0.1 marker is the hard ceiling on any multi-view consensus
  policy (labeler#27 stage 4): a candidate never stored cannot be promoted.

Colors come from the validated reference palette (categorical slots 2 and 1 for the
left panel; the blue ordinal ramp steps 250/450/650 for the right, which is the
lightest start that still clears 2:1 on a light surface). Both sets were checked with
the palette validator rather than eyeballed.

Reads the committed caches only — CPU, no GPU, no imagery:

    python scripts/analysis/plot_storage_floor.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import radius_sq_for  # noqa: E402

from low_floor_sweep import (  # noqa: E402
    ALL_SPLITS, CACHE_DIR, DEPLOYED_THRESHOLD, STORAGE_FLOOR, floor_report,
    load_split, pool_of)

# Validated categorical slots (light mode): orange = the band under scrutiny.
C_LOST = "#eb6834"
C_KEPT = "#2a78d6"
# Validated ordinal blue ramp, light mode: steps 250 / 450 / 650.
RAMP = ("#86b6ef", "#2a78d6", "#104281")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

LABEL = {"budapest_district5": "budapest*", "manual_gold": "manual_gold†"}


def collect(cities, cache_dir=CACHE_DIR):
    radius_sq = radius_sq_for()
    loaded = {c: load_split(c, cache_dir)[0] for c in cities}
    rows = []
    for city, panos in loaded.items():
        rows.append((LABEL.get(city, city), floor_report(panos, radius_sq)))
    poolable = pool_of(cities)
    if len(poolable) > 1:
        pooled = [pd for c in poolable for pd in loaded[c]]
        rows.append((f"POOLED ({len(poolable)} US)", floor_report(pooled, radius_sq)))
    # Pano count for the footer text, so adding a split can't strand a stale total.
    return rows, sum(len(p) for p in loaded.values())


def build(rows, path, n_panos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = [n for n, _ in rows]
    y = list(range(len(names)))[::-1]        # top-down reading order

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 0.52 * len(names) + 3.1), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.08})
    fig.patch.set_facecolor("white")

    # ---- Left: where the marginal ramps sit -------------------------------- #
    # Bars are a SHARE of each split's ground truth, not raw counts: manual_gold has
    # 3,919 GT ramps against ~300 for a city split, so counts would make it tower
    # while its rate is in fact the lowest here. Counts ride along as labels.
    h = 0.34
    gap = 0.02                                # 2px-equivalent surface gap
    pct = lambda n, d: 100.0 * n / d if d else 0.0   # noqa: E731
    for yi, (_n, rep) in zip(y, rows):
        lost, kept, n_gt = (rep["bands"]["[0.05,0.10)"], rep["bands"]["[0.10,0.20)"],
                            rep["n_gt"])
        ax1.barh(yi + (h + gap) / 2, pct(lost, n_gt), height=h, color=C_LOST, zorder=3)
        ax1.barh(yi - (h + gap) / 2, pct(kept, n_gt), height=h, color=C_KEPT, zorder=3)
        ax1.text(pct(lost, n_gt) + 0.09, yi + (h + gap) / 2,
                 f"{pct(lost, n_gt):.2f}%  ({lost})", va="center", ha="left",
                 fontsize=8.3, color=INK, zorder=4)
        ax1.text(pct(kept, n_gt) + 0.09, yi - (h + gap) / 2,
                 f"{pct(kept, n_gt):.2f}%  ({kept})", va="center", ha="left",
                 fontsize=8.3, color=INK_MUTED, zorder=4)

    ax1.set_xlabel("share of the split's ground-truth ramps  (count in brackets)",
                   fontsize=9, color=INK_MUTED)
    ax1.set_title("What the 0.1 storage floor discards", fontsize=11,
                  color=INK, loc="left", pad=10)
    ax1.set_xlim(0, max(pct(max(r["bands"]["[0.05,0.10)"], r["bands"]["[0.10,0.20)"]),
                            r["n_gt"]) for _, r in rows) * 1.75)
    ax1.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=C_LOST, label="[0.05, 0.10)  discarded"),
        plt.Rectangle((0, 0), 1, 1, color=C_KEPT, label="[0.10, 0.20)  kept"),
    ], fontsize=8.5, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), ncol=2)

    # ---- Right: the recall ceiling ----------------------------------------- #
    series = [("0.05 extraction floor", "0.05", RAMP[0]),
              (f"{STORAGE_FLOOR:.2f} storage floor  — the ceiling", "0.10", RAMP[1]),
              (f"{DEPLOYED_THRESHOLD:.2f} deployed", "0.55", RAMP[2])]
    for yi, (_n, rep) in zip(y, rows):
        lo, ceil = rep["recall_at"]["0.55"], rep["recall_at"]["0.10"]
        hi = rep["recall_at"]["0.05"]
        ax2.plot([lo, hi], [yi, yi], color=GRID, lw=2, zorder=1,
                 solid_capstyle="round")
        for _lab, key, colour in series:
            ax2.plot([rep["recall_at"][key]], [yi], "o", ms=9, color=colour,
                     markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        # The actionable gap is deployed -> the 0.10 ceiling: what consensus could
        # still recover from what production actually stores.
        ax2.text(hi + 0.014, yi, f"+{ceil - lo:.3f} recoverable", va="center",
                 ha="left", fontsize=8.3, color=INK_MUTED, zorder=4)

    ax2.set_xlabel("recall (share of ground-truth ramps with a candidate at or above the floor)",
                   fontsize=9, color=INK_MUTED)
    ax2.set_title("Recall ceiling: headroom above the deployed threshold",
                  fontsize=11, color=INK, loc="left", pad=10)
    ax2.set_xlim(0.44, 1.10)
    ax2.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=c,
                               markeredgecolor="white", label=lab)
                        for lab, _k, c in series],
               fontsize=8.5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=3)

    pooled_y = next((y[i] for i, (n, _) in enumerate(rows)
                     if n.startswith("POOLED")), None)
    for ax in (ax1, ax2):
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9.5, color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.5, colors=INK_MUTED)
        ax.set_ylim(min(y) - 0.8, max(y) + 0.8)
        # POOLED is an aggregate, not another split — rule it off so it is not read
        # as a peer row.
        if pooled_y is not None:
            ax.axhline(pooled_y + 0.62, color=GRID, lw=1.0, zorder=1)
    for lab in ax1.get_yticklabels():
        if lab.get_text().startswith("POOLED"):
            lab.set_fontweight("bold")

    pooled_rep = next((rep for name, rep in rows if name.startswith("POOLED")), None)
    lost_pct = (100 * pooled_rep["bands"]["[0.05,0.10)"] / pooled_rep["n_gt"]
                if pooled_rep else None)
    lost_txt = f"~{lost_pct:.1f}%" if lost_pct is not None else "a share"
    fig.suptitle(f"A 0.1 detection storage floor discards {lost_txt} of findable curb "
                 "ramps, permanently",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.012,
             f"RampNet peaks extracted at a 0.05 floor over {n_panos:,} benchmark panos "
             "(min_distance 10, no TTA). A candidate below the storage floor is never "
             "written, so no downstream\nmulti-view consensus can recover it — the 0.10 "
             "marker is a hard ceiling on labeler#27 stage 4.   "
             "*budapest GT is single-rater, low confidence.   †manual_gold is "
             "in-distribution GSV with independent GT.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.105, right=0.995, top=0.885, bottom=0.20, wspace=0.08)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    rows, n_panos = collect(ALL_SPLITS)
    path = build(rows, os.path.join(REPO, "docs", "figures", "storage_floor_ceiling.png"),
                 n_panos)
    print(f"wrote {path}")
    for name, rep in rows:
        print(f"  {name:<16} lost@0.10 {rep['bands']['[0.05,0.10)']:>3}  "
              f"ceiling {rep['recall_at']['0.10']:.3f}  deployed {rep['recall_at']['0.55']:.3f}")


if __name__ == "__main__":
    main()
