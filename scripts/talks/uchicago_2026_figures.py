"""Slide figures for the RampNet section of Jon's UChicago Distinguished Lecture (Oct 2026).

Every figure re-derives from committed data, except where a constant below says otherwise and
names its source. Output goes to ``docs/talks/uchicago_2026/``. Run from the repo root:

    python scripts/talks/uchicago_2026_figures.py            # all figures
    python scripts/talks/uchicago_2026_figures.py --only comparison

Figures
-------
comparison_f1        pooled F1 over the eight US city splits, RampNet vs seven challengers
                     (``analysis_out/scoreboard.json``, the same rows as docs/model_scoreboard.md)
transfer             RampNet F1 vs the best zero-shot model on every split, grouped by imagery
                     source and labelled by country (``scoreboard.json`` plus the Bayonne rows
                     from PR #239 until that PR merges, see BAYONNE)
recall_by_distance   recall and precision by distance band on richmond + bend
                     (``docs/detection_recall_analysis.md`` §1-§2; see RECALL_BY_DISTANCE)
deployment           human-validated precision of the AI's labels on three live Project
                     Sidewalk servers (sidewalk-auto-labeler ``docs/server-agree-check.md``)

Slide figures are 16:9, large type, one hue plus neutral inks: the validated palette that
``scripts/analysis/scoreboard_figures.py`` documents, imported from there so the two families
match. Class and group ride on position and labels, never on colour alone.
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from scoreboard_figures import (  # noqa: E402
    BLUE, BLUE_DEEP, GRID, INK, INK_MUTED, INK_SECONDARY, MUTED_FILL, SURFACE,
)

SCOREBOARD = os.path.join(REPO, "analysis_out", "scoreboard.json")
OUT_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026")

# ---------------------------------------------------------------------------------------------
# Constants that are NOT re-derived here. Each names where it was read from.
# ---------------------------------------------------------------------------------------------

# Bayonne (France, Panoramax) is scored on PR #239 (branch benchmark/bayonne-verdicts-159),
# read from that branch's analysis_out/scoreboard.json on 2026-10-05. Once #239 merges,
# scoreboard.json on main carries the split and this block is ignored (see transfer_rows).
BAYONNE = {
    "rampnet": {"precision": 0.831, "recall": 0.375, "f1": 0.517},
    "best_zero_shot": {"display": "Gemini 3.1 Pro", "f1": 0.431},
    "n_gt": 301,
}

# docs/detection_recall_analysis.md §1 (recall) and §2 (precision): richmond + bend, 637
# reviewer-confirmed ramps, deployment detections at 0.55, flat-ground distance at an assumed
# 2.5 m camera height. The per-band rows cannot be re-derived from the repo (the DA3 depth
# file is not committed, §1); the 487/637 total can (analysis_out/recall_by_depth_112.json).
RECALL_BY_DISTANCE = [
    # band,     n_gt, recall, n_det, precision
    ("0–8 m",    133, 0.842, 122, 0.943),
    ("8–12 m",   173, 0.879, 132, 0.970),
    ("12–18 m",  197, 0.812, 116, 0.966),
    ("18–25 m",  101, 0.564, 104, 0.962),
    ("25 m+",     33, 0.182,  32, 1.000),
]

# sidewalk-auto-labeler docs/server-agree-check.md at 66c76d6 (measured 2026-09-30 against
# frozen server pulls, restated 2026-10-04). Per-cluster precision = majority human vote over
# the AI labels joined within 7.5 m, Wilson 95%. Mostly one rater; the validation queue is not
# a random sample (Laurens is 71% validated, Richmond 8%, Vancouver 4%).
DEPLOYMENT = [
    # city,       imagery,                      AI labels, validated, per-cluster P, lo, hi
    ("Vancouver, BC", "GSV",                       64814, 2911, 0.971, 0.964, 0.977),
    ("Richmond, VA",  "Mapillary (iSTAR Pulsar)",  12962, 1069, 0.957, 0.941, 0.969),
    ("Laurens, IA",   "Mapillary (GoPro Max)",      1575, 1112, 0.903, 0.875, 0.925),
]

# Split -> (imagery group, place label). Imagery from benchmark/README.md's rig table.
SPLIT_META = {
    "manual_gold":       ("GSV",       "NYC / Portland / Bend, US (in-distribution)"),
    "bend":              ("GSV",       "Bend, OR, US (training city)"),
    "paterson":          ("GSV",       "Paterson, NJ, US"),
    "gainesville":       ("GSV",       "Gainesville, FL, US"),
    "laurens_gsv":       ("GSV",       "Laurens, IA, US"),
    "sao_paulo":         ("GSV",       "São Paulo, Brazil"),
    "richmond":          ("Mapillary", "Richmond, VA, US"),
    "annapolis":         ("Mapillary", "Annapolis, MD, US"),
    "morgantown":        ("Mapillary", "Morgantown, WV, US"),
    "clovis":            ("Mapillary", "Clovis, CA, US"),
    "laurens_mapillary": ("Mapillary", "Laurens, IA, US"),
    "budapest_district5": ("Mapillary", "Budapest, Hungary"),
    "bayonne":           ("Panoramax", "Bayonne, France"),
}
GROUP_ORDER = ["GSV", "Mapillary", "Panoramax"]
GROUP_LABEL = {
    "GSV": "Google Street View (the training imagery)",
    "Mapillary": "Mapillary 360° (consumer and survey rigs)",
    "Panoramax": "Panoramax (French open imagery)",
}
ZERO_SHOT_CLASSES = {"chat-vlm", "pointing", "open-vocab"}

# The seven challengers for the comparison slide: best of each family, by pooled F1.
COMPARISON_MODELS = [
    "rampnet",
    "y11l_pano",
    "gemini-3.1-pro-preview",
    "claude-opus-5-effort-low",
    "gemini-3.7-flash",
    "allenai/Molmo2-8B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/owlv2-large-patch14-ensemble",
]
CLASS_LABEL = {
    "purpose-trained": "trained on our dataset",
    "supervised": "YOLO, trained on our dataset",
    "chat-vlm": "zero-shot VLM",
    "pointing": "zero-shot pointing model",
    "open-vocab": "zero-shot open-vocab detector",
}


def load_board():
    with open(SCOREBOARD, encoding="utf-8") as f:
        return json.load(f)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=13, length=0)


def _titles(fig, title, subtitle):
    """Figure-level title and subtitle, anchored at the figure's left edge. An axes title
    starts at the axes' left edge, which with wide tick labels is mid-figure, and then runs
    off the right edge."""
    fig.text(0.012, 0.955, title, fontsize=21, color=INK, fontweight="bold", va="center")
    fig.text(0.012, 0.912, subtitle, fontsize=13.5, color=INK_SECONDARY, va="center")


def _footnote(fig, text, width=175):
    import textwrap
    lines = textwrap.wrap(text, width=width)
    for i, line in enumerate(reversed(lines)):
        fig.text(0.012, 0.012 + i * 0.026, line, fontsize=10, color=INK_MUTED, va="bottom")
    return 0.02 + 0.026 * len(lines)


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", os.path.relpath(path, REPO))
    return path


# ---------------------------------------------------------------------------------------------
# 1. Comparison
# ---------------------------------------------------------------------------------------------

def fig_comparison(board, plt):
    by_id = {m["model"]: m for m in board["models"]}
    rows = [by_id[m] for m in COMPARISON_MODELS]
    rows.sort(key=lambda m: m["f1"])
    n = len(rows)
    is_ref = [m["model"] == "rampnet" for m in rows]

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.barh(range(n), [m["f1"] for m in rows], height=0.64, zorder=3,
                   color=[BLUE if r else MUTED_FILL for r in is_ref])
    for i, (bar, m) in enumerate(zip(bars, rows)):
        ax.text(bar.get_width() + 0.012, i, f"{m['f1']:.2f}", va="center", fontsize=15,
                color=INK if is_ref[i] else INK_SECONDARY,
                fontweight="bold" if is_ref[i] else "normal", zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m['display']}   ·   {CLASS_LABEL[m['class']]}" for m in rows],
                       fontsize=14)
    for tick, ref in zip(ax.get_yticklabels(), is_ref):
        tick.set_color(INK if ref else INK_SECONDARY)
        if ref:
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("F1, macro-mean over eight US benchmark cities", fontsize=12,
                  color=INK_SECONDARY)

    ref = by_id["rampnet"]["f1"]
    best_zs = max((m for m in board["models"] if m["complete"] and m["class"] in ZERO_SHOT_CLASSES),
                  key=lambda m: m["f1"])
    _titles(fig, "Purpose-trained beats zero-shot frontier models by a wide margin",
            f"RampNet {ref:.2f} vs the best zero-shot model, {best_zs['display']}, "
            f"{best_zs['f1']:.2f}. The lead holds on every one of twelve benchmark bundles.")
    bottom = _footnote(fig,
                       "Operating points differ by class: RampNet 0.55, YOLO 0.25 (its default), "
                       "OWLv2 0.05 floor, chat VLMs emit no score. At matched operating points "
                       "and across seeds the YOLO gap is 0.016 F1, 95% CI [0.008, 0.024]. "
                       "Source: docs/model_scoreboard.md, analysis_out/scoreboard.json.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, "comparison_f1.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 2. Transfer across imagery source and country
# ---------------------------------------------------------------------------------------------

def transfer_rows(board):
    """One row per split: RampNet P/R/F1 and the best zero-shot challenger's F1."""
    per = board["per_split"]
    classes = {m["model"]: m["class"] for m in board["models"]}
    display = {m["model"]: m["display"] for m in board["models"]}
    rows = []
    for split, (group, place) in SPLIT_META.items():
        if split in per["rampnet"]:
            r = per["rampnet"][split]
            best = max(((display[m], per[m][split]["f1"]) for m in per
                        if classes.get(m) in ZERO_SHOT_CLASSES and split in per[m]
                        and per[m][split].get("f1") is not None),
                       key=lambda t: t[1], default=(None, None))
            rows.append(dict(split=split, group=group, place=place, precision=r["precision"],
                             recall=r["recall"], f1=r["f1"], n_gt=r["n_gt_recall"],
                             best_display=best[0], best_f1=best[1], source="scoreboard.json"))
        elif split == "bayonne":
            b = BAYONNE
            rows.append(dict(split=split, group=group, place=place, **b["rampnet"],
                             n_gt=b["n_gt"], best_display=b["best_zero_shot"]["display"],
                             best_f1=b["best_zero_shot"]["f1"], source="PR #239"))
    return rows


def fig_transfer(board, plt):
    rows = transfer_rows(board)
    # Within a group, in-distribution first, then by F1 descending.
    order = []
    for g in GROUP_ORDER:
        grp = [r for r in rows if r["group"] == g]
        grp.sort(key=lambda r: (r["split"] != "manual_gold", -r["f1"]))
        order.append(grp)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    y = 0
    yticks, ylabels, header_y = [], [], []
    for grp in order:
        header_y.append((y, grp[0]["group"]))
        y -= 1
        for r in grp:
            ax.plot([r["best_f1"], r["f1"]], [y, y], color=GRID, lw=3, zorder=2,
                    solid_capstyle="round")
            ax.scatter([r["best_f1"]], [y], s=110, color=MUTED_FILL, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([r["f1"]], [y], s=150, color=BLUE, zorder=4, edgecolor=SURFACE,
                       linewidth=1.5)
            ax.text(r["f1"] + 0.018, y, f"{r['f1']:.2f}", va="center", fontsize=13,
                    color=INK, fontweight="bold")
            ax.text(r["best_f1"] - 0.018, y, f"{r['best_f1']:.2f}", va="center", ha="right",
                    fontsize=11.5, color=INK_SECONDARY)
            label = r["place"] + ("  (pending PR #239)" if r["source"] != "scoreboard.json" else "")
            yticks.append(y)
            ylabels.append(label)
            y -= 1
        y -= 0.6
    for hy, g in header_y:
        ax.text(-0.015, hy, GROUP_LABEL[g], fontsize=13.5, color=INK, fontweight="bold",
                ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(y + 0.6, 0.8)
    ax.set_xlabel("F1 on the split, at each model's own operating point", fontsize=12,
                  color=INK_SECONDARY)

    # Legend: two series, so a legend box plus the direct labels above.
    ax.scatter([], [], s=150, color=BLUE, label="RampNet (trained on US GSV only)")
    ax.scatter([], [], s=110, color=MUTED_FILL, label="best zero-shot model on that split")
    # Above the plot, right-aligned: inside the axes every corner meets a row.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=12.5,
              frameon=False, labelcolor=INK_SECONDARY)

    _titles(fig, "Trained on US Google Street View, it transfers to other cameras and countries",
            "RampNet keeps the top F1 on every split. What drops out of distribution is "
            "recall, not precision.")
    bottom = _footnote(fig,
                       "Ground truth is one reviewer per split; budapest and bayonne at low / "
                       "medium reviewer confidence. Bend is a training city (4 of its 110 "
                       "benchmark panoramas are in the training set). Source: "
                       "docs/model_scoreboard.md, docs/model_comparison.md, PR #239 for bayonne.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, "transfer_imagery_country.png")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------------------------
# 3. Recall by distance
# ---------------------------------------------------------------------------------------------

def fig_recall_by_distance(plt):
    import numpy as np
    bands = [r[0] for r in RECALL_BY_DISTANCE]
    rec = [r[2] for r in RECALL_BY_DISTANCE]
    prec = [r[4] for r in RECALL_BY_DISTANCE]
    x = np.arange(len(bands))
    w = 0.36

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    b1 = ax.bar(x - w / 2 - 0.01, rec, width=w, color=BLUE, zorder=3, label="recall")
    b2 = ax.bar(x + w / 2 + 0.01, prec, width=w, color=MUTED_FILL, zorder=3, label="precision")
    for bar, v in zip(b1, rec):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=14, color=INK, fontweight="bold")
    for bar, v in zip(b2, prec):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=12.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n = {r[1]} ramps)" for b, r in zip(bands, RECALL_BY_DISTANCE)],
                       fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("rate at the deployed threshold", fontsize=12, color=INK_SECONDARY)
    ax.set_xlabel("distance from the camera to the curb ramp", fontsize=12, color=INK_SECONDARY)
    # Above the plot, right-aligned, so it meets neither the 1.00 bar nor its label.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=13,
              frameon=False, labelcolor=INK_SECONDARY)
    _titles(fig, "Recall falls off with distance; precision does not",
            "When RampNet sees a distant ramp it is almost always right. It usually does "
            "not see it.")
    bottom = _footnote(fig,
                       "637 reviewer-confirmed ramps in richmond + bend, deployment detections "
                       "at 0.55. Distance is flat-ground geometry at a 2.5 m camera height; "
                       "GSV's own depth says that axis is stretched 6–40% depending on the rig. "
                       "Source: docs/detection_recall_analysis.md §1–§2.")
    fig.tight_layout(rect=(0, bottom, 1, 0.87))
    _save(fig, "recall_by_distance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 4. Deployment
# ---------------------------------------------------------------------------------------------

def fig_deployment(plt):
    rows = DEPLOYMENT
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13.33, 6.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ys = list(range(n))[::-1]
    for y, (city, imagery, n_ai, n_val, p, lo, hi) in zip(ys, rows):
        ax.barh(y, p, height=0.56, color=BLUE, zorder=3)
        ax.plot([lo, hi], [y, y], color=BLUE_DEEP, lw=2.5, zorder=4)
        ax.text(hi + 0.012, y, f"{p:.2f}", va="center", fontsize=16, color=INK,
                fontweight="bold")
        ax.text(0.012, y, f"{n_ai:,} AI labels submitted · {n_val:,} human-validated",
                va="center", fontsize=12.5, color=SURFACE, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}\n{im}" for c, im, *_ in rows], fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("share of the AI's curb ramp clusters that human validators agreed with "
                  "(Wilson 95% interval)", fontsize=12, color=INK_SECONDARY)
    _titles(fig, "Deployed in Project Sidewalk: humans agree with 90–97% of the AI's ramps",
            "Three live city servers, three cameras. Per cluster, which is roughly per "
            "physical ramp.")
    bottom = _footnote(fig,
                       "Mostly one validator per city, and the validation queue is not a random "
                       "sample (Laurens 71% validated, Richmond 8%, Vancouver 4%). Laurens's "
                       "0.90 mixes a 0.97 core (conf ≥ 0.55) with a 0.85 lower-confidence band. "
                       "Source: sidewalk-auto-labeler docs/server-agree-check.md at 66c76d6.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, "deployment_validation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 5. The Stage 1 pipeline on one real panorama
# ---------------------------------------------------------------------------------------------

EXAMPLE_PANO = "DJ8Zp111zu6KnMZz-0PHgQ"
EXAMPLE_ROW = os.path.join(OUT_DIR, f"stage1_example_{EXAMPLE_PANO}.json")
# benchmark/*/panos/ is gitignored (the imagery is on the Hub; scripts/unpack_benchmark_panos.py
# restores it, sha256-pinned by benchmark/bend/imagery_manifest.json). --pano-dir points at it.
BEND_PANO_DIR = os.path.join(REPO, "benchmark", "bend", "panos")
BEND_RECORDS = os.path.join(REPO, "benchmark", "bend", "records.jsonl")
BEND_RAMPS = os.path.join(REPO, "stage_one", "dataset_generation", "location_data", "bend.geojson")
BEND_STREETS = os.path.join(REPO, "stage_one", "dataset_generation", "street_data",
                            "Bend - Streets.min.geojson.gz")
# Stage 1 cuts a 90° perspective view at pitch -30° and keeps its middle third (download_dataset.py:
# ``equirectangular_to_perspective(equi, 90, azimuth, -30, 1024, 1024)[:, 341:682]``), so the
# ramp model sees ±15° of bearing around the government point and elevations from about +15°
# down to -75°. Drawn here as the equivalent equirectangular rectangle.
CROP_HALF_WIDTH_DEG = 15.0
CROP_TOP_ELEV_DEG, CROP_BOTTOM_ELEV_DEG = 15.0, -75.0
INCLUSION_M = 35.0   # generate_dataset_meta.INCLUSION_DISTANCE_THRESHOLD


def _local_xy(lat0, lng0, lat, lng):
    """Metres east/north of (lat0, lng0); an equirectangular approximation, fine at 100 m."""
    import math
    return ((lng - lng0) * 111320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 111320.0)


def _bearing_to_x(bearing_deg, heading_deg):
    """Equirectangular x (0-1) of a compass bearing: the panorama's centre column is the
    camera heading, bearing increases to the right (rampnet/gsv.py conventions; checked
    against the bend ground truth, the four reviewer-confirmed ramps on this panorama sit
    within 0.012 of the bearing of their government points)."""
    return (((bearing_deg - heading_deg) % 360.0) / 360.0 + 0.5) % 1.0


def fig_pipeline(plt, pano_dir=BEND_PANO_DIR):
    import gzip
    import math
    import numpy as np
    from PIL import Image
    from matplotlib.patches import Rectangle, Circle, FancyArrow

    with open(EXAMPLE_ROW, encoding="utf-8") as f:
        row = json.load(f)
    with open(BEND_RECORDS, encoding="utf-8") as f:
        rec = next(json.loads(l) for l in f if EXAMPLE_PANO in l)["pano"]
    lat0, lng0, heading = rec["lat"], rec["lng"], rec["camera_heading"]
    with open(BEND_RAMPS, encoding="utf-8") as f:
        ramps = [(ft["geometry"]["coordinates"][1], ft["geometry"]["coordinates"][0])
                 for ft in json.load(f)["features"] if ft["geometry"]]
    with gzip.open(BEND_STREETS, "rt", encoding="utf-8") as f:
        streets = [ft["geometry"]["coordinates"] for ft in json.load(f)["features"]
                   if ft["geometry"] and ft["geometry"]["type"] == "LineString"]

    # Government ramps near the panorama, with bearing and distance.
    near = []
    for lat, lng in ramps:
        ex, ny = _local_xy(lat0, lng0, lat, lng)
        d = math.hypot(ex, ny)
        if d <= 70:
            near.append(dict(x=ex, y=ny, d=d, bearing=math.degrees(math.atan2(ex, ny)) % 360,
                             used=d <= INCLUSION_M))
    used = [r for r in near if r["used"]]
    labels = row["curb_ramp_points_normalized"]
    assert len(used) == len(row["curb_ramp_coords"]) == len(labels) == 4, (len(used), len(labels))

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(os.path.join(pano_dir, f"{EXAMPLE_PANO}.jpg"))
    img.draft("RGB", (4096, 2048))
    img = img.convert("RGB").resize((4096, 2048), Image.BILINEAR)
    W, H = img.size

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1, 1, 1, 1],
                          left=0.012, right=0.988, top=0.865, bottom=0.07, hspace=0.32,
                          wspace=0.08)
    ax_pano = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_crops = [fig.add_subplot(gs[1, i]) for i in range(1, 5)]

    # --- panorama with the three stages overlaid ---------------------------------------
    ax_pano.imshow(img, extent=(0, 1, 1, 0), aspect="auto", interpolation="bilinear")
    ax_pano.set_xlim(0, 1)
    ax_pano.set_ylim(1, 0)
    ax_pano.axis("off")
    y_top = 0.5 - CROP_TOP_ELEV_DEG / 180.0
    y_bot = 0.5 - CROP_BOTTOM_ELEV_DEG / 180.0
    used.sort(key=lambda r: _bearing_to_x(r["bearing"], heading))
    for r in used:
        x = _bearing_to_x(r["bearing"], heading)
        r["x_norm"] = x
        hw = CROP_HALF_WIDTH_DEG / 360.0
        ax_pano.add_patch(Rectangle((x - hw, y_top), 2 * hw, y_bot - y_top, fill=False,
                                    edgecolor=SURFACE, lw=1.6, ls=(0, (4, 3)), zorder=3))
        ax_pano.plot([x, x], [0.0, y_top], color=SURFACE, lw=2.2, zorder=3)
        ax_pano.plot([x, x], [0.0, y_top], color=BLUE_DEEP, lw=1.2, zorder=4)
        ax_pano.scatter([x], [0.03], marker="v", s=90, color=BLUE_DEEP, edgecolor=SURFACE,
                        lw=1.2, zorder=5)
    for lx, ly in labels:
        ax_pano.scatter([lx], [ly], s=170, facecolor="none", edgecolor=BLUE, lw=3, zorder=6)
        ax_pano.scatter([lx], [ly], s=18, color=SURFACE, zorder=6)
    ax_pano.text(0.36, 0.08, "▼  bearing of each government curb-ramp point",
                 color=SURFACE, fontsize=11.5, fontweight="bold", transform=ax_pano.transAxes,
                 va="top", bbox=dict(facecolor=INK, alpha=0.55, pad=4, edgecolor="none"))
    ax_pano.text(0.36, 0.17, "▭  the crop the ramp model inspects (±15°)",
                 color=SURFACE, fontsize=11.5, fontweight="bold", transform=ax_pano.transAxes,
                 va="top", bbox=dict(facecolor=INK, alpha=0.55, pad=4, edgecolor="none"))
    ax_pano.text(0.36, 0.26, "○  the resulting label, no human involved",
                 color=SURFACE, fontsize=11.5, fontweight="bold", transform=ax_pano.transAxes,
                 va="top", bbox=dict(facecolor=INK, alpha=0.55, pad=4, edgecolor="none"))

    # --- map ----------------------------------------------------------------------------
    R = 45.0
    ax_map.set_facecolor(SURFACE)
    for line in streets:
        pts = [_local_xy(lat0, lng0, la, lo) for lo, la in line]
        if any(abs(x) < R * 1.5 and abs(y) < R * 1.5 for x, y in pts):
            ax_map.plot([p[0] for p in pts], [p[1] for p in pts], color=GRID, lw=9,
                        solid_capstyle="round", zorder=1)
    ax_map.add_patch(Circle((0, 0), INCLUSION_M, fill=False, edgecolor=MUTED_FILL, lw=1.2,
                            ls=(0, (4, 3)), zorder=2))
    other = [r for r in near if not r["used"]]
    ax_map.scatter([r["x"] for r in other], [r["y"] for r in other], s=70, color=MUTED_FILL,
                   zorder=3)
    ax_map.scatter([r["x"] for r in used], [r["y"] for r in used], s=110, color=BLUE,
                   edgecolor=SURFACE, lw=1.2, zorder=4)
    hx, hy = math.sin(math.radians(heading)) * 9, math.cos(math.radians(heading)) * 9
    ax_map.add_patch(FancyArrow(0, 0, hx, hy, width=1.6, head_width=5, head_length=4,
                                color=INK, zorder=5, length_includes_head=True))
    ax_map.scatter([0], [0], s=90, color=INK, zorder=6)
    ax_map.plot([-R + 6, -R + 16], [-R + 6, -R + 6], color=INK_SECONDARY, lw=2.5)
    ax_map.text(-R + 11, -R + 9, "10 m", ha="center", fontsize=10, color=INK_SECONDARY)
    ax_map.text(R - 4, R - 4, "N ↑", ha="right", va="top", fontsize=11, color=INK_SECONDARY)
    ax_map.set_xlim(-R, R)
    ax_map.set_ylim(-R, R)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for s in ax_map.spines.values():
        s.set_color(GRID)
    ax_map.set_title("1. Bend's curb ramp inventory\n(GPS points) around one panorama",
                     fontsize=11.5, color=INK, loc="left", pad=6)

    # --- crops: each government bearing and the label Stage 1 produced ------------------
    half_x = 9.0 / 360.0
    for ax, r in zip(ax_crops, used):
        x = r["x_norm"]
        lx, ly = min(labels, key=lambda p: min(abs(p[0] - x), 1 - abs(p[0] - x)))
        x0, x1 = x - half_x, x + half_x
        y0, y1 = 0.47, 0.75
        box = img.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        ax.imshow(box, extent=(x0, x1, y1, y0), aspect="auto", interpolation="bilinear")
        ax.axvline(x, color=SURFACE, lw=2.6, ls=(0, (4, 3)))
        ax.axvline(x, color=BLUE_DEEP, lw=1.4, ls=(0, (4, 3)))
        ax.scatter([lx], [ly], s=260, facecolor="none", edgecolor=BLUE, lw=3.2, zorder=5)
        ax.scatter([lx], [ly], s=22, color=SURFACE, zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(f"{r['d']:.0f} m away, bearing {r['bearing']:.0f}°", fontsize=11,
                     color=INK_SECONDARY, pad=5)
    ax_crops[0].text(0, 1.22, "2. Each point becomes a bearing in the panorama.   "
                     "3. A small crop model places the label.",
                     transform=ax_crops[0].transAxes, fontsize=11.5, color=INK, va="bottom")

    _titles(fig, "Stage 1: the city's GPS inventory already labels its street imagery",
            "One training panorama in Bend, OR. Four government curb-ramp points within 35 m "
            "become four pixel labels, no human in the loop.")
    _footnote(fig, f"Panorama {EXAMPLE_PANO} (GSV, Bend, OR, captured {rec['capture_date']}); "
              "inventory stage_one/dataset_generation/location_data/bend.geojson; the labels are "
              "this panorama's row in the published dataset (stage1_example_*.json beside this "
              "figure). The reviewer later confirmed all four as real ramps "
              "(benchmark/bend/verdicts.json).", width=185)
    _save(fig, "pipeline_stage1.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

FIGURES = {
    "pipeline": lambda board, plt, pano_dir: fig_pipeline(plt, pano_dir),
    "comparison": lambda board, plt, _: fig_comparison(board, plt),
    "transfer": lambda board, plt, _: fig_transfer(board, plt),
    "recall_by_distance": lambda board, plt, _: fig_recall_by_distance(plt),
    "deployment": lambda board, plt, _: fig_deployment(plt),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=sorted(FIGURES), action="append",
                    help="render only these figures (repeatable)")
    ap.add_argument("--pano-dir", default=BEND_PANO_DIR,
                    help="directory holding the bend benchmark panoramas (for 'pipeline')")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    board = load_board()
    for name in (args.only or sorted(FIGURES)):
        FIGURES[name](board, plt, args.pano_dir)


if __name__ == "__main__":
    main()
