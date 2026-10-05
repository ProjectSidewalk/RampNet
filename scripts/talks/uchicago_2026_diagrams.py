"""System diagrams of RampNet for the UChicago lecture, written as SVG (PowerPoint imports SVG as
editable shapes) and rasterized to PNG with headless Edge when it is available.

    python scripts/talks/uchicago_2026_diagrams.py            # all four
    python scripts/talks/uchicago_2026_diagrams.py --only pipeline

Diagrams
--------
pipeline     the two stages and deployment, left to right: what goes in, what comes out
dataflow     where the labels come from and where they go: the loop through cities, the model,
             Project Sidewalk and its validators, and back into training (RampNet 2.0's engine)
deployment   the runtime: imagery sources, the auto-labeler's steps, the Project Sidewalk server
model        the Stage 2 network: panorama in, heatmap out, peaks to points

Numbers on the diagrams are the ones in docs/rampnet1_findings.md (dataset size, gold-set
agreement, pooled F1) and the labeler's docs (deployment counts). Every diagram is 1920 x 1080
on the same palette as the figures (scripts/analysis/scoreboard_figures.py).
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "diagrams")
W, H = 1920, 1080

BLUE = "#2a78d6"
BLUE_DEEP = "#184f95"
BLUE_LIGHT = "#cde2fb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
PANEL = "#f3f2ee"
GOLD = "#b8962e"
GOLD_LIGHT = "#fbf1cf"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


class SVG:
    def __init__(self):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                      f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
                      '<defs>'
                      '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
                      'markerHeight="9" orient="auto-start-reverse">'
                      f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK2}"/></marker>'
                      '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
                      'markerHeight="9" orient="auto-start-reverse">'
                      f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>'
                      '</defs>',
                      f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    def rect(self, x, y, w, h, fill=PANEL, stroke=GRID, rx=14, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=22, fill=INK, weight="normal", anchor="start", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
                          f'font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')

    def lines(self, x, y, items, size=20, fill=INK2, lh=None, anchor="start", weight="normal"):
        lh = lh or int(size * 1.4)
        for i, s in enumerate(items):
            self.text(x, y + i * lh, s, size=size, fill=fill, anchor=anchor, weight=weight)

    def arrow(self, x1, y1, x2, y2, color=INK2, sw=3, marker="arrow", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')

    def path(self, d, color=INK2, sw=3, marker="arrow", dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{m}{dd}/>')

    def circle(self, cx, cy, r, fill=BLUE, stroke=SURFACE, sw=2):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{sw}"/>')

    def pill(self, x, y, s, fill=BLUE, size=17):
        w = int(len(s) * size * 0.58 + 26)
        self.rect(x, y, w, size + 16, fill=fill, stroke="none", rx=(size + 16) // 2)
        self.text(x + 13, y + size + 3, s, size=size, fill=SURFACE, weight="bold")
        return w

    def title(self, title, subtitle=None):
        self.text(40, 70, title, size=40, weight="bold")
        if subtitle:
            self.text(40, 108, subtitle, size=22, fill=INK2)

    def footer(self, s):
        self.text(40, H - 28, s, size=16, fill=MUTED)

    def save(self, name):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name + ".svg")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.parts + ["</svg>"]) + "\n")
        print("wrote", os.path.relpath(path, REPO))
        return path


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(svg, x, y, w, h, head, body, fill=PANEL, head_color=INK, head_size=26, body_size=19,
        stroke=GRID, pad=22):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke)
    svg.text(x + pad, y + pad + head_size - 6, head, size=head_size, fill=head_color, weight="bold")
    svg.lines(x + pad, y + pad + head_size + body_size + 14, body, size=body_size, fill=INK2)


# ---------------------------------------------------------------------------------------------
# 1. Pipeline
# ---------------------------------------------------------------------------------------------

def diagram_pipeline():
    s = SVG()
    s.title("RampNet: two stages, then deployment",
            "Labels come from cities, not from annotators. One model reads the whole panorama. "
            "A labeler turns detections into a city inventory.")

    # Three bands.
    bx, bw, by, bh = 40, 580, 150, 760
    gap = 50
    for i, (head, tone) in enumerate([("Stage 1  ·  make the dataset", BLUE),
                                      ("Stage 2  ·  train the detector", BLUE),
                                      ("Deployment  ·  label a city", GOLD)]):
        x = bx + i * (bw + gap)
        s.rect(x, by, bw, bh, fill=SURFACE, stroke=GRID, rx=18)
        s.pill(x + 22, by + 20, head, fill=tone, size=19)

    # Stage 1 contents.
    x = bx + 22
    y = by + 90
    box(s, x, y, bw - 44, 118, "City curb-ramp inventories",
        ["Open GIS points from NYC, Portland, Bend", "276,071 records, with install dates"],
        fill=GOLD_LIGHT, stroke="#e6d79a")
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Project to the panorama",
        ["GPS point + panorama pose → a bearing", "GSV panoramas within 35 m of a point"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Refine with a crop model",
        ["Small keypoint net on a ±15° strip,", "trained on Project Sidewalk crops"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "214,376 labelled panoramas",
        ["849,895 curb-ramp points, no human in the loop", "P 0.92 / R 0.93 vs a 1,000-pano gold set"],
        fill=BLUE_LIGHT, stroke="#9ec5f4", head_color=BLUE_DEEP)

    # Stage 2 contents.
    x = bx + bw + gap + 22
    y = by + 90
    box(s, x, y, bw - 44, 118, "Input: the whole 360° panorama",
        ["2048 × 4096 equirectangular image", "Nothing is cropped or tiled"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "ConvNeXt V2 backbone + heatmap head",
        ["One channel at 512 × 1024", "Gaussian targets, σ = 10 px; one epoch"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Peaks are detections",
        ["Local maxima above a threshold", "Points, not boxes, with a confidence"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Released model",
        ["F1 0.91 in distribution; 0.79 pooled over", "eight US cities, ahead of every zero-shot model"],
        fill=BLUE_LIGHT, stroke="#9ec5f4", head_color=BLUE_DEEP)

    # Deployment contents.
    x = bx + 2 * (bw + gap) + 22
    y = by + 90
    box(s, x, y, bw - 44, 118, "Any street imagery",
        ["Google Street View, Mapillary 360°, Panoramax", "Enumerate a city polygon, thin to a grid"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Detect, then place in the world",
        ["RampNet on every panorama", "Bearing + range → a GPS point per detection"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Fuse views into sites",
        ["One ramp is seen from several panoramas", "Cluster within 7.5 m → one site per ramp"])
    s.arrow(x + (bw - 44) // 2, y + 118, x + (bw - 44) // 2, y + 160)
    y += 160
    box(s, x, y, bw - 44, 118, "Project Sidewalk",
        ["Submitted as AI labels; volunteers validate", "Live in Vancouver, Richmond, Laurens"],
        fill=GOLD_LIGHT, stroke="#e6d79a", head_color="#7a6210")

    # Band-to-band arrows.
    s.arrow(bx + bw - 10, by + 380, bx + bw + gap + 10, by + 380, color=BLUE, sw=4, marker="arrow-blue")
    s.text(bx + bw + gap // 2, by + 360, "train", size=18, fill=BLUE, anchor="middle")
    s.arrow(bx + 2 * bw + gap - 10, by + 380, bx + 2 * (bw + gap) + 10, by + 380, color=BLUE, sw=4,
            marker="arrow-blue")
    s.text(bx + 2 * bw + gap + gap // 2, by + 360, "run", size=18, fill=BLUE, anchor="middle")
    s.footer("Numbers: docs/rampnet1_findings.md (dataset, gold-set agreement, pooled F1); "
             "sidewalk-auto-labeler docs/server-agree-check.md (deployments).")
    return s.save("pipeline")


# ---------------------------------------------------------------------------------------------
# 2. Data flow loop
# ---------------------------------------------------------------------------------------------

def diagram_dataflow():
    s = SVG()
    s.title("Where the labels come from, and where they go",
            "A loop: cities supply the first labels, the model labels new cities, people check "
            "the model, and their checks become the next training set.")

    nodes = {
        "cities": (160, 300, 400, 150, "City open data", ["Curb-ramp GPS inventories", "NYC, Portland, Bend; 65 publishers found"], GOLD_LIGHT, "#e6d79a", "#7a6210"),
        "dataset": (160, 640, 400, 150, "Auto-labelled dataset", ["850k points on 214k GSV panoramas", "no human in the loop"], BLUE_LIGHT, "#9ec5f4", BLUE_DEEP),
        "model": (760, 470, 400, 150, "RampNet", ["Keypoint heatmap on whole panoramas", "GSV, Mapillary and Panoramax"], BLUE, BLUE, SURFACE),
        "ps": (1360, 300, 400, 150, "Project Sidewalk", ["AI labels on a city's map", "64,814 in Vancouver alone"], GOLD_LIGHT, "#e6d79a", "#7a6210"),
        "people": (1360, 640, 400, 150, "Volunteers and experts", ["Validate, tag and rate the AI's labels", "90–97% of ramp clusters agreed"], PANEL, GRID, INK),
    }
    for key, (x, y, w, h, head, body, fill, stroke, hc) in nodes.items():
        s.rect(x, y, w, h, fill=fill, stroke=stroke, rx=18)
        s.text(x + 24, y + 48, head, size=27, fill=hc, weight="bold")
        s.lines(x + 24, y + 86, body, size=19, fill=SURFACE if fill == BLUE else INK2)

    def edge(d, label, lx, ly, color=INK2, dash=None, marker="arrow"):
        s.path(d, color=color, sw=3.5, dash=dash, marker=marker)
        s.text(lx, ly, label, size=19, fill=color, anchor="middle", weight="bold")

    edge("M 360 450 L 360 640", "project to pixels (Stage 1)", 360, 555)
    edge("M 560 715 C 660 715, 660 545, 760 545", "train (Stage 2)", 640, 760)
    edge("M 1160 545 C 1260 545, 1260 375, 1360 375", "label new cities", 1250, 330)
    edge("M 1560 450 L 1560 640", "shown in the validation queue", 1560, 555)
    edge("M 1560 790 L 1560 850 L 960 850 L 960 620", "RampNet 2.0: validated labels become the next training set", 1260, 890,
         color=BLUE, dash="10 8", marker="arrow-blue")
    edge("M 1560 300 C 1560 180, 560 180, 560 300", "an inventory for cities that never had one", 1060, 165,
         color="#7a6210", dash="10 8")

    s.footer("Counts: docs/rampnet1_findings.md; docs/curb_ramp_data_sourcing.md (publishers); "
             "sidewalk-auto-labeler docs/server-agree-check.md (Vancouver, agreement).")
    return s.save("dataflow")


# ---------------------------------------------------------------------------------------------
# 3. Deployment runtime
# ---------------------------------------------------------------------------------------------

def diagram_deployment():
    s = SVG()
    s.title("Labeling a city: the runtime",
            "sidewalk-auto-labeler drives RampNet over a city's imagery and hands the result to "
            "that city's Project Sidewalk server.")

    # Sources column.
    x0, y0 = 40, 170
    s.text(x0, y0, "Imagery", size=22, fill=INK2, weight="bold")
    for i, (name, note) in enumerate([("Google Street View", "car rig, 2008–2026 history"),
                                      ("Mapillary 360°", "GoPro, iSTAR, Trimble rigs"),
                                      ("Panoramax", "French open imagery")]):
        y = y0 + 30 + i * 150
        s.rect(x0, y, 330, 120, fill=PANEL, stroke=GRID)
        s.text(x0 + 20, y + 48, name, size=23, weight="bold")
        s.text(x0 + 20, y + 84, note, size=18, fill=INK2)
        s.arrow(x0 + 330, y + 60, 470, 440)

    # Labeler block with steps.
    lx, ly, lw, lh = 470, 170, 900, 640
    s.rect(lx, ly, lw, lh, fill=SURFACE, stroke=GRID, rx=18)
    s.pill(lx + 22, ly + 18, "sidewalk-auto-labeler  (one run per city, GPU box or cluster)", fill=BLUE, size=19)
    steps = [
        ("1. Enumerate", "every panorama inside the city polygon; thin to one per ~10 m"),
        ("2. Fetch", "native-resolution tiles; archive the panoramas (23 GB to 1.2 TB per city)"),
        ("3. Detect", "RampNet on each panorama → peaks with confidence (0.30 band, 0.55 core)"),
        ("4. Place", "bearing from the pixel column; range from the ground plane and the rig's camera height"),
        ("5. Check position", "per-sequence drift on Mapillary; reposition before anything is sent"),
        ("6. Fuse", "cluster detections of one ramp across panoramas (7.5 m) → sites with a view count"),
        ("7. Submit", "POST labels per panorama to the city's server; idempotent, resumable"),
    ]
    for i, (head, body) in enumerate(steps):
        y = ly + 80 + i * 78
        s.circle(lx + 44, y + 22, 16, fill=BLUE, stroke=SURFACE)
        s.text(lx + 44, y + 29, str(i + 1), size=18, fill=SURFACE, weight="bold", anchor="middle")
        s.text(lx + 76, y + 20, head.split(". ", 1)[1], size=22, weight="bold")
        s.text(lx + 76, y + 48, body, size=17, fill=INK2)
        if i < len(steps) - 1:
            s.arrow(lx + 44, y + 40, lx + 44, y + 62, sw=2)

    # Server column.
    sx, sy, sw_, sh = 1470, 170, 410, 640
    s.rect(sx, sy, sw_, sh, fill=GOLD_LIGHT, stroke="#e6d79a", rx=18)
    s.pill(sx + 22, sy + 18, "Project Sidewalk city server", fill="#7a6210", size=19)
    s.lines(sx + 24, sy + 95, [
        "Receives AI labels per panorama",
        "Clusters them into ramps (7.5 m)",
        "Shows them in the validation queue",
        "   · volunteers: agree / disagree",
        "   · experts: tags and severity",
        "LabelMap: the city's inventory",
        "",
        "Live today",
        "   · Vancouver 64,814 AI labels",
        "   · Richmond 12,962",
        "   · Laurens 1,575",
        "   · Bayonne (Panoramax) staged",
        "",
        "Agreement per ramp cluster",
        "   · 0.97 / 0.96 / 0.90",
    ], size=19, fill=INK)
    s.arrow(lx + lw, 440, sx, 440, color=BLUE, sw=4, marker="arrow-blue")
    s.text((lx + lw + sx) // 2, 420, "labels", size=18, fill=BLUE, anchor="middle")
    s.path(f"M {sx + 200} {sy + sh} C {sx + 200} {sy + sh + 90}, {lx + 450} {sy + sh + 90}, {lx + 450} {sy + sh}",
           color=MUTED, sw=2.5, dash="8 7")
    s.text((sx + 200 + lx + 450) // 2, sy + sh + 112, "validations and misses feed the benchmark and the next model",
           size=17, fill=MUTED, anchor="middle")
    s.footer("Steps follow sidewalk-auto-labeler README and docs/production-deployment.md; counts from "
             "docs/server-agree-check.md at 66c76d6.")
    return s.save("deployment")


# ---------------------------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------------------------

def diagram_model():
    s = SVG()
    s.title("The detector: a panorama in, a heatmap out",
            "One network, one channel, points rather than boxes. rampnet/model.py, 1 epoch on "
            "the auto-labelled dataset, 16 GPUs for 3.5 hours.")

    y = 170
    stages = [
        ("Equirectangular panorama", ["3 × 2048 × 4096", "whole 360°, resized", "ImageNet-normalised"], PANEL, GRID, INK),
        ("ConvNeXt V2 base", ["timm fcmae_ft_in22k_in1k_384", "features at 1/32:", "1024 × 64 × 128"], BLUE, BLUE, SURFACE),
        ("Conv head + bilinear ×16", ["two 3×3 convs, ReLU", "upsample to 1 × 512 × 1024", "MSE loss against the targets"], BLUE, BLUE, SURFACE),
        ("Keypoint heatmap", ["Gaussian bumps, σ = 10 px", "one per curb ramp", "values in [0, 1]"], BLUE_LIGHT, "#9ec5f4", BLUE_DEEP),
        ("Peaks → detections", ["peak_local_max, min distance 10", "threshold 0.55 deployed,", "0.30 recommended"], PANEL, GRID, INK),
    ]
    bw, bh, gap = 330, 220, 42
    x = 40
    for i, (head, body, fill, stroke, hc) in enumerate(stages):
        s.rect(x, y, bw, bh, fill=fill, stroke=stroke, rx=18)
        s.text(x + 22, y + 48, head, size=24, fill=hc, weight="bold")
        s.lines(x + 22, y + 92, body, size=19, fill=SURFACE if fill == BLUE else INK2)
        if i < len(stages) - 1:
            s.arrow(x + bw + 4, y + bh // 2, x + bw + gap - 4, y + bh // 2, sw=3.5)
        x += bw + gap

    # Below: what makes it different, three notes.
    ny = 470
    notes = [
        ("Whole panorama, no tiles", "A ramp is a small thing in a big image: 2048 × 4096 in, 512 × 1024 out, so the network sees every crossing at once. The seam is handled by wrapping the scorer, not the image."),
        ("Points, not boxes", "The training labels are points (a GPS record projected to a pixel), so the model predicts points. Matching to ground truth is by distance: 0.022 of the panorama width, about 90 px."),
        ("The dataset is the result", "A YOLO trained on the same labels is 0.016 F1 behind at matched operating points. The supervision, not the architecture, is what separates RampNet from zero-shot models (0.79 vs 0.57)."),
    ]
    for i, (head, body) in enumerate(notes):
        x = 40 + i * 620
        s.rect(x, ny, 580, 300, fill=SURFACE, stroke=GRID, rx=18)
        s.text(x + 24, ny + 54, head, size=26, weight="bold")
        import textwrap
        s.lines(x + 24, ny + 100, textwrap.wrap(body, 48), size=21, fill=INK2)
    s.footer("rampnet/model.py; docs/stage2_training_cost.md; docs/seed_variance_51_135.md; "
             "docs/model_scoreboard.md.")
    return s.save("model")


# ---------------------------------------------------------------------------------------------

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def rasterize(svg_path):
    """SVG → PNG with headless Edge (what this machine has); skipped when it is absent."""
    exe = EDGE if os.path.exists(EDGE) else shutil.which("msedge") or shutil.which("chrome")
    if not exe:
        print("no browser to rasterize", svg_path)
        return None
    png = svg_path[:-4] + ".png"
    url = "file:///" + svg_path.replace("\\", "/")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={W},{H}", f"--screenshot={png}", url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", os.path.relpath(png, REPO))
    return png


DIAGRAMS = {"pipeline": diagram_pipeline, "dataflow": diagram_dataflow,
            "deployment": diagram_deployment, "model": diagram_model}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=sorted(DIAGRAMS), action="append")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    for name in (args.only or sorted(DIAGRAMS)):
        path = DIAGRAMS[name]()
        if not args.no_png:
            rasterize(path)


if __name__ == "__main__":
    main()
