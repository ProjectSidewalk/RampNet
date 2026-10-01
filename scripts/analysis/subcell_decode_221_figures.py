"""Example figures for the sub-cell decode read (#221): what argmax vs Gaussian decode
look like on real manual_gold ramps.

Two figures, both built from the committed ``analysis_out/subcell_decode_221/detections.json``
plus imagery and (for the mechanism panel) one coarse map:

1. ``examples_contact_sheet.jpg`` -- six manual_gold ramps, each a 256x256 crop of the
   2048x4096 model input (64x64 on the 512x1024 heatmap) with the argmax peak, the
   Gaussian-decoded peak and the human box plus its centre, and the 8x8-px coarse-cell grid
   drawn faintly (8 heatmap px = 32 input px).
2. ``mechanism_panel.png`` -- for one of those ramps, the 512x1024 heatmap crop next to the
   64x128 coarse crop (nearest-neighbour upscaled), a 1-D profile through the peak, and the
   manual_gold histograms of x mod 8 and y mod 8 for argmax, Gaussian and the box centres.

Selection rule (deterministic; ties broken by pano id, then row, col). Pairs are the
manual_gold argmax-matched pairs of the report step (peaks >= 0.30, radius 0.022). The
first three categories use pairs whose peak is not clipped (score <= 1), was not moved by
``climb`` and is not in a seam column (coarse col 0 or 127). d = |GT - gaussian| -
|GT - argmax| in heatmap px. At most one example per pano, in category order:

- ``large``: the two most negative d;
- ``typical``: the two whose |d| is closest to the median |d|;
- ``away``: the one most positive d (the decode moved away from the box centre);
- ``clipped``: among score > 1 pairs (not climbed, not seam) whose argmax is off the
  8i+3 / 8i+4 grid on either axis, the one whose d is closest to their median d.

The ``large`` and ``away`` picks sit near the decode's reach: their Gaussian offsets are
0.43-0.50 cell, close to ``MAX_OFFSET = 0.5``, so they show the most the decode can move a
peak, not a typical move. The mechanism panel uses the first ``typical`` example.

Usage (CPU unless --heatmap-source model)::

    python scripts/analysis/subcell_decode_221_figures.py \
        --panos-root <dir holding benchmark/manual_gold/panos> \
        --detections analysis_out/subcell_decode_221/detections.json \
        --heatmap-source model --out-dir /tmp/sc221/figures

``--panos-root`` holds ``benchmark/manual_gold/panos/<pid>.jpg`` (``fetch_manual_gold.py
--images-only``). ``--out-dir`` is required, so a run cannot overwrite the committed
figures by accident; pass ``docs/figures/subcell_decode_221`` to do it on purpose.

The mechanism panel needs the pano's 64x128 coarse map. Two sources:

- ``--coarse-dir``: an ``extract --cache-dir``. If the map's sha256 equals the one in
  ``--detections`` it is the run's own map. If not (another GPU or software stack), every
  stored 3x3 neighbourhood for that pano must agree with it to ``COARSE_TOL``.
- no ``--coarse-dir`` with ``--heatmap-source model``: the coarse map is recovered from the
  model's heatmap (``rampnet.subcell.coarse_from_heatmap``) and checked against the stored
  neighbourhoods the same way. This needs only the Hub model and the HF imagery.

``--heatmap-source model`` runs the released checkpoint on that one pano (GPU if
available) so the heatmap panel is the model's own output; ``coarse`` draws the bilinear
upsample of the coarse map instead and says so in the panel title. ``--select-only`` prints
the selection and needs no imagery.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import subcell_decode_221 as sd  # noqa: E402
from rampnet import subcell as sc  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402

IN_H, IN_W = 2048, 4096           # model input; 4 input px per heatmap px
SCALE = IN_W // sd.HM[1]
CELL_IN = sc.FACTOR * SCALE       # 32 input px per coarse cell
CROP = 256                        # input px (64 heatmap px, 8 coarse cells)
# Largest |coarse map - stored 3x3 neighbourhood| accepted when the map is not the run's own
# (sha256 differs). Cross-machine fp32 noise is about 1e-4 (doc section 3); the stored
# values are rounded to 6 decimals (5e-7).
COARSE_TOL = 2e-4

# Okabe-Ito (colour-blind safe); every mark also has its own shape.
C_ARGMAX = "#D55E00"   # vermillion, square
C_GAUSS = "#56B4E9"    # sky blue, circle
C_GT = "#F0E442"       # yellow, box + plus
CATEGORY_TITLE = {"large": "large improvement", "typical": "typical (median |d|)",
                  "away": "decode moved away", "clipped": "clipped plateau (score > 1)"}


def boxes_for(pid):
    """YOLO boxes (cx, cy, w, h), normalized, from manual_labels/<pid>.txt."""
    out = []
    with open(os.path.join(sd.REPO, "manual_labels", f"{pid}.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(tuple(float(v) for v in line.split()[1:5]))
    return out


def pair_table(D):
    """manual_gold pairs with argmax and gaussian positions and errors (heatmap px)."""
    recs = D["panos"]["manual_gold"]
    rows = sd.build_pairs(recs, sd.ground_truth("manual_gold"), radius_sq_for())
    out = []
    for pid, d, gx, gy in rows:
        ax, ay = sd.decode(d, "argmax", False)
        qx, qy = sd.decode(d, "gaussian", False)

        def err(x, y):
            return float(np.hypot(sd.wrap_dx((gx - x) * sd.HM[1]), (gy - y) * sd.HM[0]))
        ea, eg = err(ax, ay), err(qx, qy)
        out.append({"pano": pid, "det": d, "gt": (gx, gy), "argmax": (ax, ay),
                    "gaussian": (qx, qy), "e_argmax": ea, "e_gauss": eg, "d": eg - ea})
    return out


def select(pairs):
    """The six examples, per the rule in the module docstring."""
    def key(p):
        return (p["pano"], p["det"][0], p["det"][1])

    def plain(p):
        r, c, s, i, j, steps, _ = p["det"]
        return s <= 1 and steps == 0 and j not in (0, sd.COARSE[1] - 1)
    base = sorted([p for p in pairs if plain(p)], key=key)
    med_abs = float(np.median([abs(p["d"]) for p in base]))
    clipped = sorted([p for p in pairs
                      if p["det"][2] > 1 and p["det"][5] == 0
                      and p["det"][4] not in (0, sd.COARSE[1] - 1)
                      and (p["det"][0] % 8 not in (3, 4) or p["det"][1] % 8 not in (3, 4))],
                     key=key)
    med_clip = float(np.median([p["d"] for p in clipped])) if clipped else 0.0
    orders = [("large", 2, sorted(base, key=lambda p: p["d"])),
              ("typical", 2, sorted(base, key=lambda p: abs(abs(p["d"]) - med_abs))),
              ("away", 1, sorted(base, key=lambda p: -p["d"])),
              ("clipped", 1, sorted(clipped, key=lambda p: abs(p["d"] - med_clip)))]
    used, picks = set(), []
    for cat, n, cands in orders:
        k = 0
        for p in cands:        # sorted() is stable, so ties keep the pano-id order
            if k == n:
                break
            if p["pano"] in used:
                continue
            used.add(p["pano"])
            picks.append(dict(p, category=cat))
            k += 1
    return picks, {"eligible_pairs": len(base), "median_abs_d": med_abs,
                   "clipped_offgrid_pairs": len(clipped), "median_d_clipped": med_clip}


def load_input(panos_root, pid):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(os.path.join(panos_root, "benchmark", "manual_gold", "panos",
                                 f"{pid}.jpg")).convert("RGB")
    if im.size != (IN_W, IN_H):
        im = im.resize((IN_W, IN_H), Image.BILINEAR)    # what threshold_sweep.PRE does
    return np.asarray(im)


def crop_wrapped(img, cx, cy, size):
    """size x size crop centred on (cx, cy) input px; columns wrap at the seam, rows clamp.
    Returns the crop and its top-left (x0, y0) in input px (x0 may be negative)."""
    x0 = int(round(cx)) - size // 2
    y0 = int(np.clip(int(round(cy)) - size // 2, 0, img.shape[0] - size))
    cols = np.arange(x0, x0 + size) % img.shape[1]
    return img[y0:y0 + size][:, cols], x0, y0


def rel_x(x_in, x0, width):
    """Input-px x relative to a crop starting at x0, unwrapped to the nearest copy."""
    return (x_in - x0 + width / 2) % width - width / 2 if abs(x_in - x0) > width / 2 \
        else x_in - x0


def draw_example(ax, img, p):
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    gx, gy = p["gt"]
    cx, cy = gx * IN_W, gy * IN_H
    crop, x0, y0 = crop_wrapped(img, cx, cy, CROP)
    ax.imshow(crop, extent=(-0.5, CROP - 0.5, CROP - 0.5, -0.5), interpolation="lanczos")
    # coarse-cell boundaries: cell i spans heatmap px 8i-0.5 .. 8i+7.5 -> input 32i-2
    for b in range((x0 // CELL_IN) * CELL_IN - 2, x0 + CROP + CELL_IN, CELL_IN):
        if 0 <= b - x0 <= CROP:
            ax.axvline(b - x0, color="white", lw=0.5, alpha=0.35)
    for b in range((y0 // CELL_IN) * CELL_IN - 2, y0 + CROP + CELL_IN, CELL_IN):
        if 0 <= b - y0 <= CROP:
            ax.axhline(b - y0, color="white", lw=0.5, alpha=0.35)
    # human box and centre
    best = min(boxes_for(p["pano"]), key=lambda b: (b[0] - gx) ** 2 + (b[1] - gy) ** 2)
    bw, bh = best[2] * IN_W, best[3] * IN_H
    bx = rel_x(cx, x0, IN_W)
    # yellow alone is hard to see on pale concrete, so the GT marks get a black outline
    ax.add_patch(mpatches.Rectangle((bx - bw / 2, cy - y0 - bh / 2), bw, bh, fill=False,
                                    ec=C_GT, lw=1.6,
                                    path_effects=[pe.Stroke(linewidth=3.4, foreground="black"),
                                                  pe.Normal()]))
    ax.plot(bx, cy - y0, "+", ms=14, mew=2.2, color=C_GT,
            path_effects=[pe.Stroke(linewidth=4.2, foreground="black"), pe.Normal()])
    for (x, y), col, mk in ((p["argmax"], C_ARGMAX, "s"), (p["gaussian"], C_GAUSS, "o")):
        ax.plot(rel_x(x * IN_W, x0, IN_W), y * IN_H - y0, mk, ms=9, mfc=col, mec="black",
                mew=1.0)
    ax.set_xlim(-0.5, CROP - 0.5)
    ax.set_ylim(CROP - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    r, c, s = p["det"][:3]
    ax.set_title(f"{CATEGORY_TITLE[p['category']]}\n{p['pano']}  score {s:.2f}",
                 fontsize=9)
    ax.set_xlabel(f"to box centre: argmax {p['e_argmax']:.1f} px -> gaussian "
                  f"{p['e_gauss']:.1f} px  (d {p['d']:+.1f})", fontsize=9)


def contact_sheet(picks, info, panos_root, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(12, 10.4), dpi=130)
    for ax, p in zip(axes.ravel(), picks):
        draw_example(ax, load_input(panos_root, p["pano"]), p)
    handles = [Line2D([], [], ls="", marker="s", ms=9, mfc=C_ARGMAX, mec="black",
                      label="argmax (today's decode)"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=C_GAUSS, mec="black",
                      label="Gaussian sub-cell decode"),
               Line2D([], [], ls="-", color=C_GT, lw=1.6, marker="+", ms=12, mew=2,
                      path_effects=[pe.Stroke(linewidth=3.4, foreground="black"),
                                    pe.Normal()],
                      label="human box and its centre (manual_gold)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("#221: sub-cell decode on manual_gold. 256x256 crops of the 2048x4096 input; "
                 "faint grid = 8x8-px coarse cells of the 512x1024 heatmap.\n"
                 "Distances are on the 512x1024 heatmap grid (1 px = 0.35 deg); "
                 "d = gaussian - argmax, negative is closer. Over all "
                 f"{info['eligible_pairs']:,} eligible pairs, {info['frac_closer']:.0%} move "
                 f"closer; median d {info['median_d']:+.2f} px.\nMany manual_gold boxes are "
                 "only a few px across, so the box is often hidden under its centre mark.",
                 fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.87, wspace=0.04,
                        hspace=0.2)
    fig.savefig(path, pil_kwargs={"quality": 88})
    plt.close(fig)


def neighbourhood_gap(coarse, rec):
    """Largest |coarse - stored value| over every stored 3x3 neighbourhood of one pano."""
    worst = 0.0
    for r, c, s, i, j, steps, nb in rec["dets"]:
        got = sc.neighbourhood(coarse, i, j, wrap_x=True).ravel()
        for g, w in zip(got, nb):
            if w is not None:
                worst = max(worst, abs(float(g) - w))
    return worst


def check_coarse(coarse, rec, pid, source):
    """Accept ``coarse`` for ``pid`` if it is the run's own map (same sha256) or agrees with
    every stored neighbourhood to COARSE_TOL; exit otherwise. Returns it as float64."""
    got = hashlib.sha256(np.asarray(coarse, np.float32).tobytes()).hexdigest()
    if got == rec["coarse_sha256"]:
        print(f"coarse map for {pid} ({source}): sha256 equals the run's")
        return np.asarray(coarse, np.float64)
    gap = neighbourhood_gap(coarse, rec)
    if gap > COARSE_TOL:
        raise SystemExit(f"coarse map for {pid} ({source}): sha256 differs from the run's and "
                         f"it disagrees with the stored neighbourhoods by {gap:.2e} "
                         f"> {COARSE_TOL:.0e}; it is not the same model output")
    print(f"coarse map for {pid} ({source}): sha256 differs from the run's (another stack); "
          f"max |diff| to the stored neighbourhoods {gap:.2e} <= {COARSE_TOL:.0e}")
    return np.asarray(coarse, np.float64)


def model_heatmap(panos_root, pid):
    import torch
    from PIL import Image
    import threshold_sweep as ts
    Image.MAX_IMAGE_PIXELS = None
    model, _, _ = sd.load_model_at(sd.MODEL_REVISION)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    t = ts.PRE(Image.open(os.path.join(panos_root, "benchmark", "manual_gold", "panos",
                                       f"{pid}.jpg")).convert("RGB"))
    with torch.no_grad():
        h = model(t.unsqueeze(0).to(dev))
    return h[0, 0].float().cpu().numpy().astype(np.float64)


def mechanism(p, coarse, heat, heat_label, mod8, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, c, s, i, j = p["det"][:5]
    i0, j0 = i - 4, j - 4                      # 8x8 cells, cell-aligned window
    rows = np.arange(8 * i0, 8 * i0 + 64)
    cols = np.arange(8 * j0, 8 * j0 + 64) % sd.HM[1]
    hcrop = heat[np.clip(rows, 0, sd.HM[0] - 1)][:, cols]
    ccrop = coarse[np.clip(np.arange(i0, i0 + 8), 0, sd.COARSE[0] - 1)][
        :, np.arange(j0, j0 + 8) % sd.COARSE[1]]
    vmin, vmax = float(ccrop.min()), float(max(ccrop.max(), hcrop.max()))
    ext = (8 * j0 - 0.5, 8 * j0 + 63.5, 8 * i0 + 63.5, 8 * i0 - 0.5)   # heatmap px

    def marks(ax):
        ax.plot(c, r, "s", ms=8, mfc=C_ARGMAX, mec="black", label="argmax")
        gx, gy = p["gaussian"]
        ax.plot(gx * sd.HM[1], gy * sd.HM[0], "o", ms=8, mfc=C_GAUSS, mec="black",
                label="Gaussian decode")
        tx, ty = p["gt"]
        ax.plot(tx * sd.HM[1], ty * sd.HM[0], "+", ms=12, mew=2.2, color="black",
                label="box centre")

    fig = plt.figure(figsize=(13, 8.4), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hcrop, extent=ext, interpolation="nearest", cmap="viridis", vmin=vmin,
               vmax=vmax)
    marks(ax1)
    ax1.set_title(f"512x1024 heatmap ({heat_label}),\n64x64 px around the peak",
                  fontsize=10)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(ccrop, extent=ext, interpolation="nearest", cmap="viridis", vmin=vmin,
                    vmax=vmax)
    marks(ax2)
    ax2.set_title("64x128 coarse map (pre-upsample),\nsame 8x8 cells, "
                  "nearest-neighbour upscaled", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel("heatmap column (px)")
        ax.set_ylabel("heatmap row (px)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="model output")
    ax3 = fig.add_subplot(gs[0, 2])
    # 1-D profile along the axis the decode moved further on
    vertical = (abs(p["gaussian"][1] * sd.HM[0] - r)
                > abs(sd.wrap_dx(p["gaussian"][0] * sd.HM[1] - c)))
    if vertical:
        xs = np.arange(8 * i0, 8 * i0 + 64)
        ax3.plot(xs, heat[xs, c], "-", color="0.35", lw=1.5, label=f"heatmap column {c}")
        cj = int(round((c - 3.5) / 8))           # nearest coarse column to this one
        ks = np.arange(i0, i0 + 8)
        ax3.plot(8 * ks + 3.5, coarse[ks, cj % sd.COARSE[1]], "D", ms=6, color="0.1",
                 mfc="white", label=f"nearest coarse column {cj}, at 8i+3.5")
        pos = (r, p["gaussian"][1] * sd.HM[0], p["gt"][1] * sd.HM[0])
        what, unit = "column", "row"
    else:
        xs = np.arange(8 * j0, 8 * j0 + 64)
        ax3.plot(xs, heat[r, xs % sd.HM[1]], "-", color="0.35", lw=1.5,
                 label=f"heatmap row {r}")
        ci = int(round((r - 3.5) / 8))           # nearest coarse row to this one
        ks = np.arange(j0, j0 + 8)
        ax3.plot(8 * ks + 3.5, coarse[ci, ks % sd.COARSE[1]], "D", ms=6, color="0.1",
                 mfc="white", label=f"nearest coarse row {ci}, at 8j+3.5")
        pos = (c, p["gaussian"][0] * sd.HM[1], p["gt"][0] * sd.HM[1])
        what, unit = "row", "column"
    ax3.axvline(pos[0], color=C_ARGMAX, lw=2, label="argmax")
    ax3.axvline(pos[1], color=C_GAUSS, lw=2, ls="--", label="Gaussian decode")
    ax3.axvline(pos[2], color="black", lw=1.2, ls=":", label="box centre")
    ax3.set_xlabel(f"heatmap {unit} (px)")
    ax3.set_ylabel("model output")
    ax3.set_title(f"Profile down the peak {what}: piecewise linear,\nkinks only at the "
                  "8i+3.5 sample positions", fontsize=10)
    ax3.set_ylim(-0.03, 1.6 * float(heat[r, c]))     # headroom for the legend
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(alpha=0.25)
    # bars left to right in this order; hatching repeats the colour coding
    series = (("argmax", C_ARGMAX, "", "argmax (left bar)"),
              ("gaussian", C_GAUSS, "//", "Gaussian decode (middle)"),
              ("gt", "0.55", "..", "box centres, GT (right)"))
    for k, axis in enumerate(("x", "y")):
        ax = fig.add_subplot(gs[1, k])
        w = 0.27
        for n, (m, col, hatch, lab) in enumerate(series):
            v = mod8[axis][m]
            ax.bar(np.arange(8) + (n - 1) * w, v, w * 0.92, color=col, hatch=hatch,
                   edgecolor="black", linewidth=0.4, label=lab)
        ax.set_xticks(range(8))
        ax.set_xlabel(f"{'column' if axis == 'x' else 'row'} mod 8 (heatmap px)")
        ax.set_ylabel("matched pairs")
        ax.set_title(f"manual_gold, {sum(mod8[axis]['argmax']):,} pairs: {axis} mod 8",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axn = fig.add_subplot(gs[1, 2])
    axn.axis("off")
    axn.text(0, 0.95, "Reading the panel", fontsize=10, weight="bold", va="top")
    axn.text(0, 0.82,
             f"Pano {p['pano']}, peak (row {r}, col {c}),\ncoarse cell ({i}, {j}), "
             f"score {s:.2f}.\n\n"
             "The head ends in a bilinear x8 upsample, so the\n"
             "heatmap is the coarse map interpolated: along\n"
             "any row or column it is linear between coarse\n"
             "samples (at 8i+3.5), so an integer argmax can\n"
             "only land on 8i+3 or 8i+4 (the two bars in the\n"
             "argmax histograms), except on clipped plateaus.\n\n"
             "The Gaussian decode reads the 3x3 coarse\n"
             "neighbourhood instead; its x positions spread\n"
             "like the box centres. The y pile at 5-6 is where\n"
             "the ramps are (doc section 4.3), not a defect.",
             fontsize=9, va="top", family="monospace")
    fig.suptitle("#221 mechanism: the 512x1024 heatmap carries only 64x128 of resolution",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detections", default=sd.DETS)
    ap.add_argument("--panos-root", help="dir holding benchmark/manual_gold/panos/*.jpg")
    ap.add_argument("--coarse-dir",
                    help="extract --cache-dir (manual_gold/<pid>_coarse.npy); optional with "
                         "--heatmap-source model, which recovers the map from the heatmap")
    ap.add_argument("--heatmap-source", choices=("model", "coarse"), default="coarse",
                    help="model: run the released checkpoint on the mechanism pano; "
                         "coarse: bilinear upsample of the coarse map (CPU)")
    ap.add_argument("--out-dir", help="required unless --select-only; the committed figures "
                                      "are docs/figures/subcell_decode_221")
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args(argv)
    import json
    with open(a.detections, encoding="utf-8") as f:
        D = json.load(f)
    pairs = pair_table(D)
    picks, info = select(pairs)
    ds = np.array([p["d"] for p in pairs if p["det"][2] <= 1 and p["det"][5] == 0
                   and p["det"][4] not in (0, sd.COARSE[1] - 1)])
    info["median_d"] = float(np.median(ds))
    info["frac_closer"] = float(np.mean(ds < 0))
    print(f"{len(pairs)} manual_gold pairs; {info}")
    for p in picks:
        r, c, s, i, j, steps, _ = p["det"]
        print(f"  {p['category']:>8}  {p['pano']}  row {r} col {c}  score {s:.3f}  "
              f"argmax {p['e_argmax']:.2f} -> gaussian {p['e_gauss']:.2f} px (d {p['d']:+.2f})")
    if a.select_only:
        return 0
    if not (a.panos_root and a.out_dir):
        raise SystemExit("--panos-root and --out-dir are required unless --select-only")
    if not a.coarse_dir and a.heatmap_source != "model":
        raise SystemExit("--coarse-dir is required unless --heatmap-source model")
    os.makedirs(a.out_dir, exist_ok=True)
    contact_sheet(picks, info, a.panos_root,
                  os.path.join(a.out_dir, "examples_contact_sheet.jpg"))
    mp = next(p for p in picks if p["category"] == "typical")
    rec = D["panos"]["manual_gold"][mp["pano"]]
    heat = model_heatmap(a.panos_root, mp["pano"]) if a.heatmap_source == "model" else None
    if a.coarse_dir:
        coarse = check_coarse(np.load(os.path.join(a.coarse_dir, "manual_gold",
                                                   f"{mp['pano']}_coarse.npy")),
                              rec, mp["pano"], a.coarse_dir)
    else:
        coarse = check_coarse(sc.coarse_from_heatmap(heat), rec, mp["pano"],
                              "recovered from the model heatmap")
    if heat is not None:
        diff = float(np.abs(heat - sc.upsample(coarse)).max())
        print(f"mechanism pano {mp['pano']}: max |model heatmap - upsample(coarse)| = "
              f"{diff:.2e}")
        label = "model output"
    else:
        heat, label = sc.upsample(coarse), "bilinear upsample of the coarse map"
    mod8 = {ax: {} for ax in ("x", "y")}
    for m in ("argmax", "gaussian"):
        xy = np.array([p[m] for p in pairs])
        mod8["x"][m] = sd.mod8_hist(xy[:, 0] * sd.HM[1])
        mod8["y"][m] = sd.mod8_hist(xy[:, 1] * sd.HM[0])
    gt = np.array([p["gt"] for p in pairs])
    mod8["x"]["gt"] = sd.mod8_hist(gt[:, 0] * sd.HM[1])
    mod8["y"]["gt"] = sd.mod8_hist(gt[:, 1] * sd.HM[0])
    print(f"mod8: {mod8}")
    mechanism(mp, coarse, heat, label, mod8, os.path.join(a.out_dir, "mechanism_panel.png"))
    for n in sorted(os.listdir(a.out_dir)):
        print(f"-> {os.path.join(a.out_dir, n)} "
              f"({os.path.getsize(os.path.join(a.out_dir, n)) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
