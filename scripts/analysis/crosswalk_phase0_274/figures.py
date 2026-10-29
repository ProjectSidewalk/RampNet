"""Redraw the two Phase 0 figures from the committed tables (no GPU, no network).

  python scripts/analysis/crosswalk_phase0_274/figures.py
      [--tables analysis_out/crosswalk_phase0_274] [--out docs/figures/crosswalk_phase0_274]

Figure 1: signed along-crossing offset of crowd Crosswalk labels from their OSM crossing node, pooled over
the 17 dense cities (item 2). Under an end-of-crossing convention this would be bimodal at +/- half the
carriageway (3-6 m); it is unimodal at zero.
Figure 2: distance from each crowd label to the nearest merged Vistas component on its pano, against the
0.05 reading radius and the chance floor (item 3).
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, FILL = "#1f2328", "#59636e", "#d0d7de", "#3b6fb6"


def style(ax, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def fig_offsets(tables, out):
    along = []
    for f in glob.glob(os.path.join(tables, "offsets", "*.csv")):
        along += [float(r["along_m"]) for r in csv.DictReader(open(f)) if r["axis"] != "none"]
    along = [v for v in along if abs(v) <= 10]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(along, bins=[-10 + 0.25 * i for i in range(81)], color=FILL, edgecolor="white", linewidth=0.4)
    for x in (-3, 3):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
    ax.text(3.15, ax.get_ylim()[1] * 0.92, "±3 m: an end-of-crossing\nmode would sit out here", color=MUTED,
            fontsize=8.5, va="top")
    style(ax, "label − OSM node, along the crossing (m; sign arbitrary)", "labels")
    ax.set_title(f"Crowd Crosswalk labels sit on the OSM crossing node (n = {len(along):,}, 17 cities)",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "offset_along_crossing.png"))
    plt.close(fig)


def fig_probe(tables, out):
    rows = list(csv.DictReader(open(os.path.join(tables, "vistas_probe", "labels_scored.csv"))))
    d = [min(0.125, float(r["nearest_m_dist"])) if r["nearest_m_dist"] else 0.125 for r in rows]  # last bin = beyond 0.12 or none
    extra = json.load(open(os.path.join(tables, "vistas_probe", "summary_extra.json")))
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(d, bins=[0.01 * i for i in range(13)], color=FILL, edgecolor="white", linewidth=0.4)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.axvline(0.05, color=MUTED, linewidth=1, linestyle="--")
    rec = extra["pinned_recall@0.05"]
    ax.text(0.052, ax.get_ylim()[1] * 0.92,
            f"radius 0.05: recall {rec:.2f}\nchance floor {extra['chance_floor_recall@0.05']['mean']:.2f}"
            f" (p95 {extra['chance_floor_recall@0.05']['p95']:.2f})", color=MUTED, fontsize=8.5, va="top")
    style(ax, "crowd label → nearest merged Vistas component, pano widths (matcher metric)", "labels")
    ax.set_title(f"Vistas crosswalk classes vs {len(rows)} crowd Crosswalk labels on {extra['n_panos']} benchmark panos",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vistas_probe_nearest.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="analysis_out/crosswalk_phase0_274")
    ap.add_argument("--out", default="docs/figures/crosswalk_phase0_274")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_offsets(a.tables, a.out)
    fig_probe(a.tables, a.out)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
