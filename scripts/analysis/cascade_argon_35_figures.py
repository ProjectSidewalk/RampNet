"""The two figures for docs/cascade_argon_35.md, from the committed argon_pairs.json.

Both regenerate from ``analysis_out/cascade_cost_35/argon/argon_pairs.json`` alone: no
cache, no network, no model. ``cascade_argon_35.py`` regenerates that JSON (and its
``--check`` byte-compares it).

1. ``pooled_f1.png`` -- the narrow claim. Pooled F1 over the four open-rig splits for
   every arm on its own (no in-sample max over the three YOLO arms), one panel per read
   (``best_by_f1``, in sample; ``fixed``, selected on none of these data). The 0.30
   baseline and the per-split best single threshold are drawn as labelled lines. The
   five arms sit within 0.004 (``best_by_f1``) and 0.009 (``fixed``) of each other.
2. ``blind_gt.png`` -- why the teacher question is open. Each arm's unique points (boxes
   with no RampNet op_cache peak >= 0.05 within the group radius), pooled over the four
   open-rig splits: x = their false positives, y = how many of the "RampNet-blind" GT
   ramps the arm puts a box on. Left panel at the scoring radius R (primary), right at 2R
   (grouped and scored at 2R; sensitivity). The cascade cannot use any of these boxes.

**Colour.** One hue per family (orange for Argon, blue for the YOLO arms: validated
categorical slots 2 and 1, the same assignment as ``prompt_276_figures.py``), plus the
neutral inks. Within a family the arm is carried by the row label or a direct label, and
filled vs hollow marks all boxes vs the YOLO headline confidence 0.25. Every plotted
number is also in the doc's tables and in ``cascade_argon_35.py --tables``.

Usage::

    python scripts/analysis/cascade_argon_35_figures.py           # writes docs/figures/cascade_argon_35/
    python scripts/analysis/cascade_argon_35_figures.py --out-dir /tmp/f35
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIRS = os.path.join(REPO, "analysis_out", "cascade_cost_35", "argon", "argon_pairs.json")
FIG_DIR = os.path.join(REPO, "docs", "figures", "cascade_argon_35")

ARGON_C, YOLO_C = "#eb6834", "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

ARMS = [("gemini-4-argon", "Argon zero-shot"),
        ("gemini-4-argon-confidence-on", "Argon + confidence"),
        ("y11l_pano", "y11l_pano"), ("y11x_pano_h200", "y11x_pano_h200"),
        ("y26_pano", "y26_pano")]
POOL = "open_rig"
CONF = "conf_ge_0.25"


def _color(leg):
    return ARGON_C if leg.startswith("gemini") else YOLO_C


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)


def fig_pooled_f1(res, path, plt):
    p = res["pools"][POOL]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(ARMS)))[::-1]
    titles = {"best_by_f1": "best_by_f1 (in-sample best grid row)",
              "fixed": "fixed (study's setting, not selected here)"}
    for ax, read in zip(axes, ("best_by_f1", "fixed")):
        _style(ax)
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        for x, name in ((p["baseline"]["F1"], "baseline @0.30"),
                        (p["threshold_only_best_per_split"]["F1"], "best threshold\n(per split)")):
            ax.axvline(x, color=INK_MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
            ax.text(x, len(ARMS) - 0.35, name, fontsize=7.4, color=INK_SECONDARY,
                    ha="center", va="bottom")
        for y, (leg, _) in zip(ys, ARMS):
            f1 = p["arms"][leg][read]["F1"]
            ax.plot([p["baseline"]["F1"], f1], [y, y], color=_color(leg), lw=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot(f1, y, "o", ms=8, color=_color(leg), mec=SURFACE, mew=2, zorder=3)
            ax.text(f1 + 0.0015, y + 0.22, f"{f1:.4f}", fontsize=7.6, color=INK_SECONDARY,
                    va="bottom", ha="left")
        ax.set_xlim(0.69, 0.75)
        ax.set_ylim(-0.6, len(ARMS) + 0.2)
        ax.set_title(titles[read], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("pooled F1, four open-rig splits (count sums)", fontsize=8.5,
                      color=INK_SECONDARY)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([n for _, n in ARMS], fontsize=8.5, color=INK)
    fig.suptitle("As a cascade gate, Argon ties each YOLO arm on open-rig imagery "
                 "(pooled rows carry no interval)", fontsize=10.5, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


#: Direct-label offsets (points) where two marks would collide.
NUDGE = {("R", "y11l_pano", CONF): (8, 6), ("R", "y26_pano", CONF): (9, -9),
         ("R", "y11x_pano_h200", CONF): (6, -13), ("R", "y11x_pano_h200", "all"): (-8, -12),
         ("R", "y11l_pano", "all"): (6, 4),
         ("2R", "y11x_pano_h200", CONF): (6, -10), ("2R", "y11l_pano", CONF): (6, 4),
         ("2R", "y11x_pano_h200", "all"): (6, -10), ("2R", "y11l_pano", "all"): (-8, 6)}


def fig_blind_gt(res, path, plt):
    pu = res["pools_unique_points"][POOL]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    titles = {"R": "grouped and scored at R = 0.022 (primary)",
              "2R": "grouped and scored at 2R (sensitivity, generous to all)"}
    for ax, cfg in zip(axes, ("R", "2R")):
        _style(ax)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        u = pu[cfg]
        n = u["n_blind_gt"]
        for leg, name in ARMS:
            for var, o in u["arms"][leg].items():
                filled = var == "all"
                c = _color(leg)
                ax.plot(o["fp"], o["blind_gt_reached"], "o", ms=8.5,
                        color=c if filled else SURFACE, mec=c, mew=1.8, zorder=3)
                label = name if leg.startswith("gemini") else (
                    f"{name} (all)" if filled else f"{name} @0.25")
                dx, dy = NUDGE.get((cfg, leg, var), (6, 3))
                ax.annotate(label, (o["fp"], o["blind_gt_reached"]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=7.2, color=INK_SECONDARY,
                            ha="right" if dx < 0 else "left")
        pts = [o for arms in u["arms"].values() for o in arms.values()]
        ax.set_xlim(-0.03 * max(o["fp"] for o in pts), 1.3 * max(o["fp"] for o in pts))
        ax.set_ylim(0, 1.2 * max(o["blind_gt_reached"] for o in pts))
        ax.set_title(f"{titles[cfg]}\n{n} RampNet-blind GT ramps", fontsize=9.2, color=INK,
                     loc="left")
        ax.set_xlabel("unique-point false positives (no RampNet peak nearby)", fontsize=8.5,
                      color=INK_SECONDARY)
    axes[0].set_ylabel("RampNet-blind GT ramps the arm puts a box on", fontsize=8.5,
                       color=INK_SECONDARY)
    fig.suptitle("Ramps the cascade cannot reach: Argon puts boxes on more of them than any "
                 "YOLO arm, at a cost in false positives", fontsize=10.5, color=INK, x=0.01,
                 ha="left")
    fig.text(0.01, 0.005, "Pooled over laurens_mapillary, juneau, bayonne, budapest_district5; "
             "recall-confirmed panos; RampNet floor 0.05. Filled = all boxes, hollow = YOLO "
             "at its headline conf 0.25.\nFPs are upper bounds (GT is anchored on RampNet "
             "detections plus reviewer marks). Counts, not an experiment.",
             fontsize=7.2, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--out-dir", default=FIG_DIR)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(args.pairs, encoding="utf-8") as f:
        res = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, fn in (("pooled_f1.png", fig_pooled_f1), ("blind_gt.png", fig_blind_gt)):
        path = os.path.join(args.out_dir, name)
        fn(res, path, plt)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
