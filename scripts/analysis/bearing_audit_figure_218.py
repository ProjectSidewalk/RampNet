"""Figure for docs/bearing_audit_218.md, from bearing_audit_218.py's committed outputs.

Left: heading-shift scan. Every detection's bearing turned by a constant; flat
canvas_level @ 0.30 above its swap null, and the panos @ 0.55 above their rotation null
(the pano floor is held at its unshifted value). Right: where detections sit relative to
the ramp's projected bearing, real minus chance per in-view pair, 5-deg bins.

    python scripts/analysis/bearing_audit_figure_218.py
    # the unnamed-camera heading sheet (needs the #218 thumbnails, IMG = fetch --out)
    python scripts/analysis/bearing_audit_figure_218.py --heading-sheet IMG
"""
import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "analysis_out", "perspective_photos_218", "bearing_audit")
DST = os.path.join(REPO, "docs", "figures", "bearing_audit_218", "bearing_audit.png")
FLAT, PANO = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"


# five unnamed-camera frames: four drawn with a seeded sample (random.seed(218)) from those
# whose SfM heading is 15-26 deg clockwise of the device-GPS direction of travel, plus the
# first figure miss of #227 (590984823847247, 34 deg)
SHEET_IDS = ["534028659033255", "1271468953967182", "1472599527030957", "599747162768949",
             "590984823847247"]
SHEET = os.path.join(REPO, "docs", "figures", "bearing_audit_218", "unnamed_heading_check.jpg")


def heading_sheet(img_dir):
    """Each frame with two vertical lines: the SfM heading (blue) and the device-GPS
    direction of travel (orange), each projected through the frame's camera and SfM pose."""
    import numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from rampnet import perspective as P
    import bearing_audit_218 as B
    import perspective_photos_218 as PP
    meta = B.all_meta()
    rows = {r["image_id"]: r for r in PP.read_csv(PP.IMAGES_CSV)}
    tiles = []
    for i in SHEET_IDS:
        im = Image.open(os.path.join(img_dir, i + ".jpg")).convert("RGB")
        w, h = im.size
        cam, R = PP.camera_of(rows[i], w, h), PP.pose_of(rows[i])
        head = float(rows[i]["computed_compass_angle"])
        dr = ImageDraw.Draw(im)
        for b, col in ((head, FLAT), (head + meta[i]["d_travel_raw"], PANO)):
            v = R @ np.array([math.sin(math.radians(b)), math.cos(math.radians(b)), 0.0])
            u, vv = P.project_cam(cam, v)
            dr.line([(u, 0), (u, h)], fill=col, width=6)
            dr.ellipse([u - 12, vv - 12, u + 12, vv + 12], outline=col, width=5)
        im.thumbnail((1024, 1024))
        tiles.append(im)
    sheet = Image.new("RGB", (max(t.size[0] for t in tiles), sum(t.size[1] for t in tiles)),
                      "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.size[1]
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    sheet.save(SHEET, quality=80)
    print("wrote", SHEET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heading-sheet", metavar="IMG")
    a = ap.parse_args()
    if a.heading_sheet:
        heading_sheet(a.heading_sheet)
        return
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
    a.set_title("Heading-shift scan: no shift closes the gap", color=INK, fontsize=10, loc="left")

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
    b.set_title("Signed offsets: excess peaks at 0", color=INK, fontsize=10, loc="left")
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    fig.savefig(DST)
    print("wrote", DST)


if __name__ == "__main__":
    main()
