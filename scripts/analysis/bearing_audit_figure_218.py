"""Figure for docs/bearing_audit_218.md, from bearing_audit_218.py's committed outputs.

Left: heading-shift scan. Every detection's bearing turned by a constant; flat
canvas_level @ 0.30 above its swap null, and the panos @ 0.55 above their rotation null
(the pano floor is held at its unshifted value). Right: where detections sit relative to
the ramp's projected bearing, real minus chance per in-view pair, 5-deg bins.

    python scripts/analysis/bearing_audit_figure_218.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "analysis_out", "perspective_photos_218", "bearing_audit")
DST = os.path.join(REPO, "docs", "figures", "bearing_audit_218", "bearing_audit.png")
FLAT, PANO = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"


def main():
    with open(os.path.join(SRC, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    flat = s["arms"]["canvas_level"]["0.30"]
    pano = s["pano_control"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax in (a, b):
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    xs = [r["shift"] for r in flat["shift_scan"]]
    a.plot(xs, [r["above"] for r in flat["shift_scan"]], color=FLAT, lw=2, marker="o", ms=3)
    a.plot([r["shift"] for r in pano["shift_scan"]],
           [r["rate"] - pano["rotation_null"] for r in pano["shift_scan"]],
           color=PANO, lw=2, marker="o", ms=3)
    a.text(30, flat["shift_scan"][-4]["above"] + 0.03, "flat, canvas @ 0.30", color=INK,
           fontsize=8)
    a.text(18, 0.36, "360 panos @ 0.55", color=INK, fontsize=8)
    a.set_xlabel("heading shift applied to every detection (deg)", color=INK, fontsize=9)
    a.set_ylabel("hit rate above chance", color=INK, fontsize=9)
    a.set_title("Heading-shift scan: both peak at 0", color=INK, fontsize=10, loc="left")

    rows = list(csv.DictReader(open(os.path.join(SRC, "offsets.csv"), encoding="utf-8")))
    n = {"flat canvas_level @ 0.30": flat["offsets"]["all"]["n_pairs"],
         "pano @ 0.55": pano["offsets"]["n_pairs"]}
    for src, col, lab in (("flat canvas_level @ 0.30", FLAT, "flat, canvas @ 0.30"),
                          ("pano @ 0.55", PANO, "360 panos @ 0.55")):
        rr = [r for r in rows if r["source"] == src]
        mid = [(float(r["bin_lo"]) + float(r["bin_hi"])) / 2 for r in rr]
        ex = [float(r["excess"]) / n[src] for r in rr]
        b.step(mid, ex, where="mid", color=col, lw=2, label=lab)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xlabel("detection bearing minus projected ramp bearing (deg; + = right)",
                 color=INK, fontsize=9)
    b.set_ylabel("detections per pair, real minus chance", color=INK, fontsize=9)
    b.set_title("Signed offsets: excess centred at 0", color=INK, fontsize=10, loc="left")
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    fig.savefig(DST)
    print("wrote", DST)


if __name__ == "__main__":
    main()
