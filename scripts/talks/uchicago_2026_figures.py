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

PROVIDER_NAME = {
    "rampnet": "RampNet (ours)", "yolo": "Ultralytics YOLO", "gemini": "Google Gemini",
    "claude": "Anthropic Claude", "qwen": "Alibaba Qwen3-VL", "molmo": "Ai2 Molmo2",
    "owlv2": "Google OWLv2", "gdino": "IDEA Grounding DINO",
}


def fig_comparison(board, plt, v2=False):
    """``v2`` keeps the chart and states the N: cities, panoramas, ramps, models, providers."""
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
    pooled = board["pooled_splits"]
    n_panos = sum(board["splits"][s]["n_panos"] for s in pooled)
    n_gt = sum(board["splits"][s]["n_gt"] for s in pooled)
    complete = [m for m in board["models"] if m["complete"]]
    providers = []
    for m in complete:
        if m["provider"] not in providers:
            providers.append(m["provider"])
    if v2:
        _titles(fig, "Purpose-trained beats zero-shot frontier models by a wide margin",
                f"N = {len(pooled)} US cities, {n_panos:,} panoramas, {n_gt:,} reviewer-confirmed "
                f"ramps. {len(complete) - 1} challengers from {len(providers) - 1} providers "
                f"tested; best of each family shown.")
        bottom = _footnote(fig,
                           "Cities (one reviewer each, ~125 panoramas per city): "
                           + ", ".join(pooled) + ". Models: "
                           + "; ".join(PROVIDER_NAME[p] for p in providers)
                           + ". YOLO was trained on the same dataset as RampNet; every other "
                           "challenger is zero-shot with a fixed prompt. Operating points: RampNet "
                           "0.55, YOLO 0.25 (its default), OWLv2 0.05 floor, chat VLMs emit no "
                           "score; at matched operating points and across seeds the YOLO gap is "
                           "0.016 F1, 95% CI [0.008, 0.024]. Source: docs/model_scoreboard.md.",
                           width=165)
        name = "comparison_f1_v2.png"
    else:
        _titles(fig, "Purpose-trained beats zero-shot frontier models by a wide margin",
                f"RampNet {ref:.2f} vs the best zero-shot model, {best_zs['display']}, "
                f"{best_zs['f1']:.2f}. The lead holds on every one of twelve benchmark bundles.")
        bottom = _footnote(fig,
                           "Operating points differ by class: RampNet 0.55, YOLO 0.25 (its default), "
                           "OWLv2 0.05 floor, chat VLMs emit no score. At matched operating points "
                           "and across seeds the YOLO gap is 0.016 F1, 95% CI [0.008, 0.024]. "
                           "Source: docs/model_scoreboard.md, analysis_out/scoreboard.json.")
        name = "comparison_f1.png"
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, name)
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


def fig_transfer(board, plt, us_only=False):
    """``us_only`` drops the non-US splits (and with them the Panoramax group), so the figure
    reads as camera transfer alone."""
    rows = transfer_rows(board)
    if us_only:
        rows = [r for r in rows if r["place"].endswith(", US") or "US (" in r["place"]]
    # Within a group, in-distribution first, then by F1 descending.
    order = []
    for g in GROUP_ORDER:
        grp = [r for r in rows if r["group"] == g]
        if not grp:
            continue
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

    if us_only:
        _titles(fig, "Trained on Google Street View, it transfers to other cameras",
                f"{len(rows)} US splits, two imagery sources, five camera rigs. RampNet keeps the "
                "top F1 on every split; what drops is recall, not precision.")
        bottom = _footnote(fig,
                           "Ground truth is one reviewer per split. Bend is a training city (4 of "
                           "its 110 benchmark panoramas are in the training set). Laurens is one "
                           "town photographed by both rigs. Source: docs/model_scoreboard.md, "
                           "docs/model_comparison.md.")
        name = "transfer_imagery_us.png"
    else:
        _titles(fig, "Trained on US Google Street View, it transfers to other cameras and countries",
                "RampNet keeps the top F1 on every split. What drops out of distribution is "
                "recall, not precision.")
        bottom = _footnote(fig,
                           "Ground truth is one reviewer per split; budapest and bayonne at low / "
                           "medium reviewer confidence. Bend is a training city (4 of its 110 "
                           "benchmark panoramas are in the training set). Source: "
                           "docs/model_scoreboard.md, docs/model_comparison.md, PR #239 for bayonne.")
        name = "transfer_imagery_country.png"
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, name)
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
# 6. Stage 2 heatmap demo (from the npz that uchicago_2026_heatmap.py wrote)
# ---------------------------------------------------------------------------------------------

HEATMAP_SPLIT, HEATMAP_PANO = "paterson", "tekhQ4HQ9pcOqs_kGGpGaw"
DEPLOYED_T, RECOMMENDED_T = 0.55, 0.30


HEAT_BAND = (0.26, 0.80)   # fraction of the equirect height shown: sky and the ground under the camera hold no ramps


def _heat_overlay(img_band, heat_band):
    """The panorama dimmed and cooled, the heatmap in jet with alpha from its value: the look of
    the 2023 RampNet talk slide. Display only; peaks are extracted from the raw heatmap."""
    import numpy as np
    from PIL import Image
    from matplotlib import cm
    base = np.asarray(img_band, dtype=np.float32)
    tint = base * 0.42 + np.array([18, 22, 90], dtype=np.float32) * 0.58
    hb = Image.fromarray((heat_band * 255).astype(np.uint8)).resize(img_band.size, Image.BICUBIC)
    hb = np.asarray(hb, dtype=np.float32) / 255.0
    rgb = cm.jet(hb)[..., :3] * 255
    alpha = np.clip(hb * 1.6, 0, 0.92)[..., None]
    return Image.fromarray((tint * (1 - alpha) + rgb * alpha).round().astype(np.uint8))


def fig_heatmap(plt, pano_dir, split=HEATMAP_SPLIT, pano=HEATMAP_PANO):
    import numpy as np
    from PIL import Image
    from skimage.feature import peak_local_max

    npz = np.load(os.path.join(OUT_DIR, f"heatmap_{split}_{pano}.npz"))
    heat = npz["heatmap"].astype(np.float32)
    meta = json.loads(str(npz["meta"]))
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        rec = next(json.loads(l) for l in f if pano in l)
    with open(os.path.join(REPO, "benchmark", split, "verdicts.json"), encoding="utf-8") as f:
        verdict = json.load(f)["panos"][pano]
    missed = verdict.get("missed", [])

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(os.path.join(pano_dir, f"{pano}.jpg"))
    img.draft("RGB", (4096, 2048))
    img = img.convert("RGB").resize((4096, 2048), Image.BILINEAR)
    b0, b1 = HEAT_BAND
    band = img.crop((0, int(b0 * 2048), 4096, int(b1 * 2048))).resize((2560, int(2560 / 2 * (b1 - b0))), Image.LANCZOS)
    H, W = heat.shape
    overlay = _heat_overlay(band, heat[int(b0 * H):int(b1 * H)])

    peaks = peak_local_max(heat, min_distance=10, threshold_abs=RECOMMENDED_T, exclude_border=False)
    conf = heat[tuple(peaks.T)] if len(peaks) else np.zeros(0)
    px = (peaks[:, 1] + 0.5) / W
    py = ((peaks[:, 0] + 0.5) / H - b0) / (b1 - b0)
    strong = conf >= DEPLOYED_T

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, left=0.012, right=0.988, top=0.855, bottom=0.085, hspace=0.14)
    ax_in = fig.add_subplot(gs[0])
    ax_out = fig.add_subplot(gs[1])
    for ax in (ax_in, ax_out):
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")
    ax_in.imshow(band, extent=(0, 1, 1, 0), aspect="auto", interpolation="bilinear")
    ax_in.set_title("input: the whole 360° panorama at 2048 × 4096 (street band shown)",
                    loc="left", fontsize=12.5, color=INK, pad=4)
    ax_out.imshow(overlay, extent=(0, 1, 1, 0), aspect="auto", interpolation="bilinear")
    ax_out.scatter(px[strong], py[strong], s=230, facecolor="none", edgecolor="#5af0ff", lw=3,
                   zorder=4)
    ax_out.scatter(px[strong], py[strong], s=22, color="#5af0ff", zorder=4)
    ax_out.scatter(px[~strong], py[~strong], s=170, facecolor="none", edgecolor="#5af0ff",
                   lw=1.8, ls=(0, (2, 2)), zorder=4)
    for m in missed:
        ax_out.scatter([m["x"]], [(m["y"] - b0) / (b1 - b0)], s=200, marker="x",
                       color="#ff7b72", lw=3, zorder=5)
    n_strong, n_weak = int(strong.sum()), int((~strong).sum())
    ax_out.set_title(f"output: a 512 × 1024 keypoint heatmap. Rings are peaks ≥ 0.55 "
                     f"({n_strong}); dashed rings sit between 0.30 and 0.55 ({n_weak})"
                     + (f"; × marks a ramp the reviewer found that the model missed "
                        f"({len(missed)})" if missed else ""),
                     loc="left", fontsize=12.5, color=INK, pad=4)

    n_gt = sum(1 for d in verdict["dets"] if d is True) + len(missed)
    _titles(fig, "Stage 2: one model, the whole panorama, points not boxes",
            f"ConvNeXt V2 backbone, one-channel heatmap head. {n_gt} confirmed ramps in this "
            f"{split} panorama, {n_strong} peaks at the deployed threshold, all of them right."
            if n_strong == n_gt and not missed else
            f"ConvNeXt V2 backbone, one-channel heatmap head. {n_gt} reviewer-confirmed ramps "
            f"in this {split} panorama.")
    cam = rec["pano"]
    _footnote(fig, f"Panorama {pano} ({split}, {cam.get('capture_date', '?')}, "
              f"{cam['width']}×{cam['height']} source), not in the training set. Checkpoint "
              f"{meta['checkpoint'][:60]}, one forward pass, no flip-TTA, run {meta['run_date']} "
              f"on {meta['device']}. Ground truth: benchmark/{split}/verdicts.json. "
              "Heatmap saved beside this figure; regenerate with uchicago_2026_heatmap.py.",
              width=185)
    _save(fig, f"heatmap_demo_{split}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 7. RampNet 2.0: find, tag, rate
# ---------------------------------------------------------------------------------------------

# Text from docs/rampnet2_plan.md §1 (goal table) and §4 (experiments), read 2026-10-05.
ROADMAP = [
    ("FIND", "Where is every curb ramp?",
     "Gold-set F1 0.91; leads every zero-shot model on all twelve benchmark bundles; "
     "live in three Project Sidewalk cities.",
     "Consumer 360° rigs (Laurens recall 0.39 per view). Merging views of one ramp: "
     "of 74 ramps no fused site recovered, 63 fired in some view.",
     "mostly done"),
    ("TAG", "What kind of ramp, and what is wrong with it?",
     "118k tagged labels in Project Sidewalk and a published DINOv2 baseline, but tags are "
     "positive-unlabeled and rater-dependent, with no rubric.",
     "A rubric and a two-rater agreement ceiling; field-of-view and positive-unlabeled training "
     "experiments; then tag channels on the keypoint head.",
     "starting"),
    ("RATE", "How severe, in numbers a city can act on?",
     "Recorded severity is a 3-level quality scale (88 / 9 / 3 %) that tags already predict "
     "(κ 0.44); two trained raters agree at κ 0.21.",
     "Width and slope from cross-view geometry and GSV depth; severity re-derived from "
     "measurements and tags; ~50 field-measured ramps as the only route to 'better than humans'.",
     "open"),
]


def fig_roadmap(plt):
    from matplotlib.patches import FancyBboxPatch
    import textwrap

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [(0.015, 0.20), (0.235, 0.355), (0.605, 0.38)]
    heads = ["", "where it stands", "what RampNet 2.0 adds"]
    for (x, w), h in zip(cols, heads):
        if h:
            ax.text(x, 0.955, h, fontsize=13, color=INK_SECONDARY, va="center")
    row_h, gap, top = 0.255, 0.035, 0.925
    for i, (name, question, state, nxt, status) in enumerate(ROADMAP):
        y1 = top - i * (row_h + gap)
        y0 = y1 - row_h
        ax.add_patch(FancyBboxPatch((0.005, y0), 0.99, row_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor="#f3f2ee", edgecolor=GRID, lw=1, zorder=1))
        x, w = cols[0]
        ax.text(x + 0.01, y1 - 0.045, name, fontsize=22, color=BLUE if i == 0 else INK,
                fontweight="bold", va="center", zorder=3)
        ax.text(x + 0.01, y1 - 0.105, "\n".join(textwrap.wrap(question, 26)), fontsize=12,
                color=INK_SECONDARY, va="top", zorder=3)
        ax.text(x + 0.01, y0 + 0.03, status, fontsize=11, color=SURFACE, va="center",
                fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BLUE if i == 0 else INK_MUTED,
                          edgecolor="none"))
        x, w = cols[1]
        ax.text(x, y1 - 0.03, "\n".join(textwrap.wrap(state, 54)), fontsize=12, color=INK,
                va="top", zorder=3, linespacing=1.35)
        x, w = cols[2]
        ax.text(x, y1 - 0.03, "\n".join(textwrap.wrap(nxt, 62)), fontsize=12, color=INK,
                va="top", zorder=3, linespacing=1.35)
    ax.plot([0.59, 0.59], [0.06, 0.93], color=GRID, lw=1, zorder=2)

    _titles(fig, "RampNet 2.0: find, then tag, then rate",
            "The goal is an AI accessibility labeller at least as good as a human. Curb ramps "
            "first: a designed object with a measurable severity.")
    _footnote(fig, "Data engine running alongside: AI proposals surfaced in Project Sidewalk's "
              "expert-validate tool, human accept / reject becomes the clean training set. "
              "Source: docs/rampnet2_plan.md §1, §4; docs/multiview_48.md.", width=185)
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0.07)
    _save(fig, "rampnet2_roadmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 8. Comparison variants: precision/recall, with manual_gold, and one city with every leg
# ---------------------------------------------------------------------------------------------

def _hbar_frame(plt, n, figsize=(13.33, 7.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n - 0.4)
    return fig, ax


def fig_comparison_pr(board, plt):
    """The same eight models, precision and recall instead of F1."""
    by_id = {m["model"]: m for m in board["models"]}
    rows = sorted((by_id[m] for m in COMPARISON_MODELS), key=lambda m: m["f1"])
    n = len(rows)
    is_ref = [m["model"] == "rampnet" for m in rows]
    fig, ax = _hbar_frame(plt, n)
    h = 0.34
    for i, m in enumerate(rows):
        ax.barh(i + h / 2 + 0.02, m["precision"], height=h, color=BLUE_DEEP, zorder=3)
        ax.barh(i - h / 2 - 0.02, m["recall"], height=h, color=BLUE, zorder=3)
        ax.text(m["precision"] + 0.01, i + h / 2 + 0.02, f"P {m['precision']:.2f}", va="center",
                fontsize=11.5, color=INK, fontweight="bold" if is_ref[i] else "normal")
        ax.text(m["recall"] + 0.01, i - h / 2 - 0.02, f"R {m['recall']:.2f}", va="center",
                fontsize=11.5, color=INK, fontweight="bold" if is_ref[i] else "normal")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m['display']}   ·   {CLASS_LABEL[m['class']]}" for m in rows], fontsize=13.5)
    for tick, ref in zip(ax.get_yticklabels(), is_ref):
        tick.set_color(INK if ref else INK_SECONDARY)
        if ref:
            tick.set_fontweight("bold")
    ax.barh([], [], color=BLUE_DEEP, label="precision")
    ax.barh([], [], color=BLUE, label="recall")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=12.5, frameon=False,
              labelcolor=INK_SECONDARY)
    ax.set_xlabel("macro-mean over eight US benchmark cities, at each model's operating point",
                  fontsize=12, color=INK_SECONDARY)
    ref = by_id["rampnet"]
    _titles(fig, "RampNet leads on precision and recall at once; challengers trade one for the other",
            f"RampNet P {ref['precision']:.2f} / R {ref['recall']:.2f}. Chat VLMs run near 0.55–0.65 "
            f"on both; OWLv2 finds {by_id['google/owlv2-large-patch14-ensemble']['recall']:.2f} of "
            f"the ramps at {by_id['google/owlv2-large-patch14-ensemble']['precision']:.2f} precision.")
    bottom = _footnote(fig,
                       "Same eight US cities, panoramas and ramps as the F1 chart (953 panoramas, "
                       "2,309 ramps). Operating points: RampNet 0.55, YOLO 0.25, OWLv2 0.05 floor, "
                       "chat VLMs emit no score. A 0.30 threshold moves RampNet to P 0.92 / R 0.80 "
                       "pooled over seven cities (docs/operating_point.md). Source: "
                       "docs/model_scoreboard.md.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, "comparison_pr.png")
    plt.close(fig)


def fig_comparison_gold(board, plt):
    """The F1 chart with a second mark per model: F1 on manual_gold, the in-distribution set."""
    by_id = {m["model"]: m for m in board["models"]}
    rows = sorted((by_id[m] for m in COMPARISON_MODELS), key=lambda m: m["f1"])
    n = len(rows)
    is_ref = [m["model"] == "rampnet" for m in rows]
    fig, ax = _hbar_frame(plt, n)
    bars = ax.barh(range(n), [m["f1"] for m in rows], height=0.6, zorder=3,
                   color=[BLUE if r else MUTED_FILL for r in is_ref])
    for i, (bar, m) in enumerate(zip(bars, rows)):
        g = m.get("manual_gold_f1")
        # The bar's value sits right of whichever mark is further out, so a diamond just past
        # the bar end never lands on the number.
        ax.text(max(bar.get_width(), (g or 0) + 0.02) + 0.012, i - 0.02, f"{m['f1']:.2f}",
                va="center", fontsize=14, color=INK if is_ref[i] else INK_SECONDARY,
                fontweight="bold" if is_ref[i] else "normal", zorder=4)
        if g is None:
            ax.text(0.012, i, "no manual_gold run", va="center", fontsize=10.5,
                    color=INK_MUTED, zorder=5, style="italic")
        else:
            ax.scatter([g], [i], marker="D", s=120, color=BLUE_DEEP, edgecolor=SURFACE,
                       linewidth=1.2, zorder=5)
            ax.text(g, i + 0.42, f"{g:.2f}", ha="center", va="bottom", fontsize=10.5,
                    color=BLUE_DEEP, zorder=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m['display']}   ·   {CLASS_LABEL[m['class']]}" for m in rows], fontsize=13.5)
    for tick, ref in zip(ax.get_yticklabels(), is_ref):
        tick.set_color(INK if ref else INK_SECONDARY)
        if ref:
            tick.set_fontweight("bold")
    ax.barh([], [], color=MUTED_FILL, label="F1, eight deployment cities (bar)")
    ax.scatter([], [], marker="D", s=120, color=BLUE_DEEP,
               label="F1, manual_gold: 1,000 in-distribution GSV panoramas, 3,919 ramps (diamond)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=11.5, frameon=False,
              labelcolor=INK_SECONDARY)
    ax.set_xlabel("F1 at each model's operating point", fontsize=12, color=INK_SECONDARY)
    _titles(fig, "In distribution the supervised models are close; deployed, RampNet holds up",
            "manual_gold is held out of the pooled headline: it is GSV from the training cities, "
            "labelled with no model in the loop.")
    bottom = _footnote(fig,
                       "manual_gold: RampNet 0.91 with flip-TTA at 0.55; YOLO11x 0.85, and level "
                       "with RampNet at matched thresholds (0.911 vs 0.905). Gemini 3.1 Pro and "
                       "Claude Opus 5 have no published manual_gold detections (the Gemini number "
                       "in the docs, 0.57, is not re-derivable). Source: docs/model_scoreboard.md "
                       "'In-distribution vs deployed'.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, "comparison_f1_with_gold.png")
    plt.close(fig)


CITY_CLASS_LABEL = dict(CLASS_LABEL, **{"supervised-transfer": "Mapillary Vistas transfer"})


def fig_comparison_city(board, plt, split="annapolis"):
    """Every leg ever scored on one split, the one with the most legs."""
    disp = {m["model"]: m for m in board["models"]}
    rows = sorted(((v[split]["f1"], m) for m, v in board["per_split"].items() if split in v),
                  key=lambda t: t[0])
    n = len(rows)
    is_ref = [m == "rampnet" for _, m in rows]
    fig, ax = _hbar_frame(plt, n)
    bars = ax.barh(range(n), [f for f, _ in rows], height=0.66, zorder=3,
                   color=[BLUE if r else MUTED_FILL for r in is_ref])
    for i, (bar, (f, m)) in enumerate(zip(bars, rows)):
        ax.text(bar.get_width() + 0.01, i, f"{f:.2f}", va="center", fontsize=11.5,
                color=INK if is_ref[i] else INK_SECONDARY,
                fontweight="bold" if is_ref[i] else "normal", zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{disp[m]['display']}  ·  {CITY_CLASS_LABEL[disp[m]['class']]}"
                        for _, m in rows], fontsize=10.5)
    for tick, ref in zip(ax.get_yticklabels(), is_ref):
        tick.set_color(INK if ref else INK_SECONDARY)
        if ref:
            tick.set_fontweight("bold")
    info = board["splits"][split]
    ax.set_xlabel(f"F1 on {split} ({info['n_panos']} panoramas, {info['n_gt']} reviewer-confirmed "
                  "ramps), at each model's operating point", fontsize=12, color=INK_SECONDARY)
    best = rows[-2]
    _titles(fig, f"One city, every model we have run: {n - 1} challengers on {split.title()}",
            f"RampNet {rows[-1][0]:.2f}; the best challenger is {disp[best[1]]['display']} at "
            f"{best[0]:.2f}. Mapillary 360° imagery from a survey rig (Trimble MX7).")
    bottom = _footnote(fig,
                       "Claude legs ran at two reasoning efforts; effort moved the operating point, "
                       "not the quality (docs/claude_legs_122.md). '(anthropic)' legs billed the "
                       "Anthropic API, the rest Vertex. The Vistas arm is Mapillary's Curb Cut class "
                       "at 1024 px input. Source: docs/model_scoreboard.md by-split table.")
    fig.tight_layout(rect=(0, bottom, 1, 0.885))
    _save(fig, f"comparison_f1_{split}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

FIGURES = {
    "pipeline": lambda board, plt, pano_dir: fig_pipeline(plt, pano_dir),
    "heatmap": lambda board, plt, pano_dir: fig_heatmap(plt, pano_dir),
    "roadmap": lambda board, plt, _: fig_roadmap(plt),
    "comparison": lambda board, plt, _: fig_comparison(board, plt),
    "comparison_v2": lambda board, plt, _: fig_comparison(board, plt, v2=True),
    "comparison_pr": lambda board, plt, _: fig_comparison_pr(board, plt),
    "comparison_gold": lambda board, plt, _: fig_comparison_gold(board, plt),
    "comparison_annapolis": lambda board, plt, _: fig_comparison_city(board, plt),
    "transfer": lambda board, plt, _: fig_transfer(board, plt),
    "transfer_us": lambda board, plt, _: fig_transfer(board, plt, us_only=True),
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
