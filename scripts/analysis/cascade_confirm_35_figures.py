"""The Stage 0 figure for docs/cascade_confirm_35.md, from the committed stage0.json alone.

``heldout.png``, two panels:

* left: held-out F1 of N, T and C on each GoPro Max split (leave-one-split-out) and pooled;
* right: the two pooled gate quantities with their selection-aware 95% intervals, T minus N
  and C minus max(N, T), against 0 and the +0.03 bar Rule 2 sets for C.

Colours are the validated categorical slots 1-3 (blue N, orange T, aqua C) in fixed order;
every series is also named in the legend, and every plotted number is in the doc's tables.

    python scripts/analysis/cascade_confirm_35_figures.py          # writes docs/figures/cascade_confirm_35/
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT = os.path.join(REPO, "analysis_out", "cascade_confirm_35", "stage0.json")
FIG_DIR = os.path.join(REPO, "docs", "figures", "cascade_confirm_35")
COL = {"N": "#2a78d6", "T": "#eb6834", "C": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=FIG_DIR)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(RESULT, encoding="utf-8") as f:
        p = json.load(f)
    rows = [(f["held_out"], f["N"]["F1"], f["T"]["F1"], f["C"]["F1"]) for f in p["folds"]]
    po = p["pooled"]
    rows.append(("pooled", po["N"]["F1"], po["T"]["F1"], po["C"]["F1"]))

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK2)
    ys = list(range(len(rows)))[::-1]
    for (name, *vals), y in zip(rows, ys):
        a.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2, zorder=1)
        for k, v, dy in zip("NTC", vals, (0.12, 0, -0.12)):
            a.scatter(v, y + dy, s=64, color=COL[k], edgecolor="white", linewidth=1.5,
                      zorder=3, label=k if y == ys[0] else None)
    a.set_yticks(ys, [r[0] for r in rows], color=INK)
    a.set_xlabel("held-out F1 (rig mask applied)", color=INK2)
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    from matplotlib.lines import Line2D
    names = {"N": "N (0.30)", "T": "T (t* from other splits)", "C": "C (cascade, other splits)"}
    a.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, color=COL[k], label=names[k])
                      for k in "NTC"], frameon=False, loc="upper center",
             bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=9)

    items = [("T - N", po["T_minus_N"], po["T_minus_N_ci95"], COL["T"]),
             ("C - max(N, T)", po["C_minus_maxNT"], po["C_minus_maxNT_ci95"], COL["C"])]
    for i, (lab, v, (lo, hi), c) in enumerate(items):
        y = 1 - i
        b.plot([lo, hi], [y, y], color=c, lw=2)
        b.scatter(v, y, s=64, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        b.text(v, y + 0.15, f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]", ha="center", color=INK,
               fontsize=9)
    b.axvline(0, color=INK2, lw=1)
    b.axvline(0.03, color=INK2, lw=1, ls="--")
    b.text(0.0315, -0.55, "+0.03 bar for C", color=INK2, fontsize=8, ha="left")
    b.set_yticks([1, 0], [it[0] for it in items], color=INK)
    b.set_ylim(-0.7, 1.6)
    b.set_xlabel("pooled held-out delta F1, 95% interval", color=INK2,
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "heldout.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
