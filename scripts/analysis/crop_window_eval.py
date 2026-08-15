"""Score crop-window sizing rules against the manual_gold boxes (issue #114).

FINDING (2026-08-14, docs/crop_window_eval.md): the manual_labels w/h are NOT
object-extent gold — ~90% are near-point marks (median 12.8 px), only ~10%
plausibly full-extent. Context-ratio outputs against them are therefore not
object-context ratios; containment means "the marked spot stays in the crop."
The harness re-runs unchanged against a proper extent box set once one exists.

Every consumer that cuts a context crop around a curb-ramp point sizes the
window with a rule, and until now no rule has ever been scored against ground
truth extent. This repo holds the only gold extent data in the ecosystem:
``manual_labels/*.txt`` — 3,919 human-drawn boxes on the 1,000 manual_gold
panos — consumed everywhere else as centers only. This script keeps the w/h and
asks, for each candidate sizing rule: *does the window it cuts actually contain
the ramp, with the ramp at a sensible scale?*

Metrics per (gold box, candidate window):

- **containment** — box fully inside the window (seam-wrap aware);
- **margin** — signed min distance from box edge to window edge, normalized by
  window side (containment is ``margin >= 0``; negative values show violation
  depth instead of a bare boolean);
- **context ratio** — box max side / window side, plus a per-axis split
  (``_h``/``_v``): aprons are strongly anisotropic in an equirect, so a square
  window sized to hold the width over-provisions the height by that factor. The
  consumer-requirements survey in sidewalk-panorama-tools converged on ~10-15% as
  the useful band (a survey, not a measurement — treated as a target band only).
- **size ratio** — predicted side / :func:`required_side`, the side that box and
  prompt actually demand. **Read the spread, not the median.** Containment and
  context ratio are both monotone in window size, so a rule can win either by
  being uniformly bigger; the size ratio's p90/p10 is the only comparison here
  that a scale constant cannot buy. ``--rescale-sweep`` (on by default) makes the
  same point end-to-end: every rule x k, re-scored with the real geometry, so the
  rules can be compared at *matched* containment.

Stratified by the box's depression angle below the horizon (a flat-ground
distance proxy: d = 2.5 m / tan(depression)).

Two prompt modes (where the window is centered):

- ``gold-center`` — at the gold box center. Isolates *sizing* error from
  placement error; every box is scored.
- ``detection`` — at the matched RampNet detection from the bundle's
  ``records.jsonl`` (greedy confidence-ordered match, standard 0.022 radius,
  1024/512 anisotropic scaling — identical to rampnet.detection_eval). The
  production-realistic number: real crops are cut at detection points. Only
  matched boxes are scored; uncovered boxes and unmatched detections are
  counted and reported.

Candidates:

- ``v1-raw`` — ``predict_crop_size`` exactly as sidewalk-panorama-tools'
  CropRunner.py runs it today (pixel-linear distance fit on native pano_y;
  constants calibrated on 6656-px-high GSV panos).
- ``v1-norm`` — the resolution-normalized port specified for SidewalkWebpage's
  CropService (ProjectSidewalk/SidewalkWebpage#4865): compute in 6656-height
  reference space, scale the result back. Bit-identical to v1-raw at
  pano_height = 6656.
- ``geo-v1.5`` — depression angle -> flat-ground distance (fixed 2.5 m camera
  height) -> metric ramp footprint -> apparent pixels -> pad to the context
  band. The naive precursor of a per-pano-height rule
  (ProjectSidewalk/sidewalk-auto-labeler#40).

All numbers need no pixels: pano dimensions come from
``benchmark/manual_gold/records.jsonl``. Imagery is only needed for
``--gallery``, which renders overlay crops for whatever subset of
``benchmark/manual_gold/panos/`` exists locally (canonical fetch:
``scripts/fetch_manual_gold.py``). ``--fetch-sample N`` can instead pull a
small sample of panos fresh from GSV via streetlevel (optional dependency) into
the gallery's own directory — view-only bytes, never written into the bundle,
not benchmark-fidelity (the bundle's canonical imagery is the HF test split).

**Real extent gold (--bundle mode, issue #116):** ``scripts/box_gallery.py`` produces
``benchmark/<city>/boxes.json`` — whole-apron boxes drawn under an explicit versioned
rule, keyed ``(pano_id, det:<i>|missed:<i>)``, pano-normalized and possibly wrapping the
equirectangular seam. ``--bundle benchmark/richmond`` scores the same rules against that
gold instead of manual_labels. Differences from the manual flow: boxes may wrap in x
(handled, never clamped); the detection prompt comes from the recorded ``point`` of each
``det:<i>`` item (adjudicated linkage — no re-matching), while ``gold:<i>`` items from
box_gallery's ``--from-manual-labels`` mode are matched to the record's operational
detections here (same greedy 0.022 match as everywhere) so that arm produces a
detection-prompted table too; ``missed:<i>`` items score in gold-center mode only
(production cuts no crop where there is no detection); and the report adds the
**directional road-context margin** — how far the window extends below the box's bottom
edge (the ramp-street junction, approximate for oblique ramps), in box-heights. Against
real extent gold, containment finally means "the whole ramp fits."

Two things about a bundle that the numbers depend on and that are reported, not assumed:
completeness (``boxes.json`` is cross-checked against ``verdicts.json``'s adjudicated
count — an abandoned session otherwise just shrinks the denominator silently) and the
resolution the gold was drawn at (``crop_px_by_pano_dims``). On richmond, 9 of 92 panos
are 4096x2048, so part of even that gold is model-resolution; ``benchmark/manual_gold``
is 4096x2048 throughout, which also means the **GSV arm scores at pano_h = 2048** — below
the 6656 calibration height, the same regime as richmond, NOT the >6656 regime where the
raw formula over-sizes. Probing that needs records carrying native GSV dimensions.

Run (from the repo root):

    python scripts/analysis/crop_window_eval.py                # manual_gold: JSON + CSV
    python scripts/analysis/crop_window_eval.py --gallery 60   # + overlay gallery
    python scripts/analysis/crop_window_eval.py --fetch-sample 60 --gallery 60
    python scripts/analysis/crop_window_eval.py --bundle benchmark/richmond --gallery 60

Outputs (``<name>`` = ``crop_window_eval`` or ``crop_window_eval_<bundle>``):

- ``analysis_out/<name>.json``        — summary (committed; see .gitignore)
- ``analysis_out/<name>_per_box.csv`` — one row per (box, rule, mode) (ignored)
- ``analysis_out/<name>_gallery/``    — overlay crops + index.html (ignored)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)
from rampnet.metrics import greedy_match  # noqa: E402
from rampnet.validation import wilson_interval  # noqa: E402

LABELS_DIR = os.path.join(REPO, "manual_labels")
BUNDLE_DIR = os.path.join(REPO, "benchmark", "manual_gold")
OUT_DIR = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

# --- v1 formula constants: verbatim from sidewalk-panorama-tools CropRunner.py
# (predict_crop_size). The distance fit is pixel-LINEAR in pano_y, calibrated on
# GSV panos 6656 px high; clamps are [50, 1500] px in that reference space.
#
# PROVENANCE — the sign convention below is the whole port, and a flipped one would
# pass every test here while inverting the conclusions, so pin it against the source
# rather than against ourselves. Upstream reads:
#
#     def predict_crop_size(pano_y, pano_height):
#         old_pano_y = pano_height / 2 - pano_y
#         distance = max(0, 19.80546390 + 0.01523952 * old_pano_y)
#         ...
#
#   https://github.com/ProjectSidewalk/sidewalk-panorama-tools/blob/master/CropRunner.py
#   (predict_crop_size; called from make_single_crop as
#    predict_crop_size(pano_y, pano_height) -> compute_crop_box(...))
#
# So ``v1-raw`` feeds ``pano_h / 2 - y_px`` straight in, exactly as production does.
# Upstream's clamps are `if crop_size > 1500 or distance == 0` then `if crop_size < 50`
# — two independent ifs, reproduced as-is (an elif would be equivalent: neither clamp
# can fire after the other).
#
# Note what the upstream docstring says about `old_pano_y`: it "converts pano_y and
# pano_height to the OLD version of pano_y that we had when this alg was written."
# That conversion is missing — the constant 6656-height space is assumed, not mapped to
# — which is precisely the defect ``v1-norm`` (ProjectSidewalk/SidewalkWebpage#4865)
# fixes, and it makes v1-norm the faithful port rather than a variant of it.
# Constant fits: sidewalk-cv-tools#2 (comment 510609873), SidewalkWebpage#633
# (comment 307283178).
V1_DIST_INTERCEPT = 19.80546390
V1_DIST_SLOPE = 0.01523952
V1_SIZE_COEF = 8725.6
V1_SIZE_EXP = -1.192
V1_MIN, V1_MAX = 50.0, 1500.0
V1_REF_HEIGHT = 6656.0

# --- geo-v1.5 constants. CAM_H matches scripts/analysis/size_analysis.py and the
# YOLO pseudo-box prep; RAMP_W is a nominal single-ramp width; TARGET_RATIO is
# the middle of the 10-15% consumer band; distance clamps bound the window when
# the flat-ground model degenerates (near the horizon, or a label above it).
#
# CAM_H is the rule's one free parameter and it absorbs everything the flat-ground
# model gets wrong: too high a value overestimates distance, so the apparent ramp is
# underestimated and the window comes out small (measured ctx above target). Overridable
# with ``--cam-height`` and recorded in the summary — geo-v1.5's target ratio cannot be
# read as calibrated without knowing which height produced it. GSV per-pano heights are
# now measurable (ProjectSidewalk/sidewalk-auto-labeler#40 measured a 2.21 m median,
# well below this 2.5), which is the path to removing the parameter instead of tuning it.
GEO_CAM_H_M = 2.5
GEO_RAMP_W_M = 1.5
GEO_TARGET_RATIO = 0.125
GEO_D_MIN_M, GEO_D_MAX_M = 2.0, 40.0

# Production operational threshold (sidewalk-auto-labeler detectors package):
# only detections at or above this ever become submitted labels, so detection-
# mode scoring defaults to the same floor.
OPERATIONAL_CONFIDENCE = 0.55

# Depression-angle strata (degrees below horizon), with flat-ground distance at
# 2.5 m camera height for the labels. Left-open, right-closed.
STRATA = [
    (0.0, 4.0, ">36 m / horizon"),
    (4.0, 8.0, "18-36 m"),
    (8.0, 15.0, "9-18 m"),
    (15.0, 25.0, "5-9 m"),
    (25.0, 90.0, "<5 m"),
]

RULES = ("v1-raw", "v1-norm", "geo-v1.5")


# ---------------------------------------------------------------------------
# Inputs

def parse_yolo_boxes(label_path):
    """Parse one YOLO label file keeping extent: [(cx, cy, w, h), ...] normalized.

    Mirrors rampnet.detection_eval.yolo_ground_truth's strictness (benchmark
    artifact: malformed lines raise), but keeps w/h — this scorer exists
    precisely because everything else drops them.
    """
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{lineno}: expected 'class cx cy w h', got {line.strip()!r}")
            try:
                cx, cy, w, h = (float(p) for p in parts[1:5])
            except ValueError:
                raise ValueError(
                    f"{label_path}:{lineno}: non-numeric field in {line.strip()!r}") from None
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError(
                    f"{label_path}:{lineno}: center ({cx}, {cy}) outside [0, 1]")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                raise ValueError(
                    f"{label_path}:{lineno}: degenerate extent ({w}, {h})")
            boxes.append((cx, cy, w, h))
    return boxes


def load_gold(labels_dir=LABELS_DIR):
    """{pano_id: [(cx, cy, w, h), ...]} for every label file (empty lists kept)."""
    gold = {}
    for name in sorted(os.listdir(labels_dir)):
        if name.endswith(".txt"):
            gold[name[:-4]] = parse_yolo_boxes(os.path.join(labels_dir, name))
    return gold


def load_records(records_path=os.path.join(BUNDLE_DIR, "records.jsonl")):
    """{pano_id: {"width", "height", "detections"}} from the bundle records."""
    recs = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pano = rec["pano"]
            recs[pano["panorama_id"]] = {
                "width": int(pano["width"]),
                "height": int(pano["height"]),
                "detections": rec.get("detections", []),
            }
    return recs


def _box_key_order(key):
    """Sort ``det:<i>``/``missed:<i>``/``gold:<i>`` keys by kind then index."""
    kind, _, idx = key.partition(":")
    return ({"det": 0, "missed": 1, "gold": 2}.get(kind, 3), int(idx) if idx.isdigit() else 0)


def load_bundle_boxes(bundle_dir):
    """Extent gold from a box_gallery export (``<bundle>/boxes.json``, #116).

    Returns ``(gold, prompts, meta)``: ``gold`` has load_gold's shape
    ({pano_id: [(cx, cy, w, h), ...]}) so the summary and gallery paths are shared;
    ``prompts[(pano_id, idx)]`` carries the item key and its recorded prompt point
    (for ``det:<i>`` the adjudicated detection position — used verbatim, no
    re-matching); ``meta`` records the embedded box_rule and status counts.
    Only ``status == "boxed"`` items score; "can't determine extent" is counted.
    """
    with open(os.path.join(bundle_dir, "boxes.json"), encoding="utf-8") as f:
        bj = json.load(f)
    gold, prompts = {}, {}
    n_cant = n_other = n_edge = 0
    for pano_id in sorted(bj.get("panos", {})):
        for key in sorted(bj["panos"][pano_id], key=_box_key_order):
            item = bj["panos"][pano_id][key]
            status = item.get("status")
            if status != "boxed":
                if status == "cant":
                    n_cant += 1
                else:
                    n_other += 1     # e.g. a note left on a ramp nobody boxed
                continue
            n_edge += 1 if item.get("edge_flag") else 0
            boxes = gold.setdefault(pano_id, [])
            prompts[(pano_id, len(boxes))] = {
                "key": key, "point": item.get("point"),
                "is_detection": key.startswith("det:"),
            }
            boxes.append((item["cx"], item["cy"], item["w"], item["h"]))
    meta = {
        "run_name": bj.get("run_name", os.path.basename(os.path.normpath(bundle_dir))),
        "box_rule": bj.get("box_rule"),
        "crop_fov_deg": bj.get("crop_fov_deg"),
        "crop_px_by_pano_dims": bj.get("crop_px_by_pano_dims"),
        "n_boxed": len(prompts), "n_cant": n_cant, "n_other": n_other,
        "n_edge_flag": n_edge,
    }
    meta.update(check_gold_complete(bundle_dir, meta))
    return gold, prompts, meta


def count_adjudicated_ramps(verdicts_path):
    """Ramps a box pass is supposed to cover: ``True`` dets + sure missed marks.

    Same population :func:`enumerate_items` in scripts/box_gallery.py enumerates, and
    the same one rampnet.validation.collect scores — including its partially-judged
    skip, so the two counts are comparable.
    """
    with open(verdicts_path, encoding="utf-8") as f:
        panos = json.load(f).get("panos", {})
    total = 0
    for entry in panos.values():
        dets = entry.get("dets", [])
        if any(d is None for d in dets):
            continue
        total += sum(1 for d in dets if d is True)
        total += sum(1 for m in entry.get("missed", []) if not m.get("unsure"))
    return total


def check_gold_complete(bundle_dir, meta):
    """{n_adjudicated, completeness_warning} — is this boxes.json the whole pass?

    Nothing else notices an abandoned annotation session: an incomplete file just
    silently shrinks the denominator, and every rate in the report is then computed
    over whatever subset got drawn. verdicts.json is in the same directory, so the
    check is free.
    """
    path = os.path.join(bundle_dir, "verdicts.json")
    if not os.path.exists(path):
        return {"n_adjudicated": None, "completeness_warning": None}
    n_adj = count_adjudicated_ramps(path)
    n_have = meta["n_boxed"] + meta["n_cant"] + meta["n_other"]
    warning = None
    if n_have != n_adj:
        warning = (f"boxes.json covers {n_have} of {n_adj} adjudicated ramps "
                   f"({n_adj - n_have} never annotated) — every rate below is over "
                   f"that subset, not over the benchmark's ramps")
    return {"n_adjudicated": n_adj, "completeness_warning": warning}


# ---------------------------------------------------------------------------
# Sizing rules (side in native pixels; y_px is the prompt row in native pixels)

def _v1_reference_size(ref_y_offset):
    """The CropRunner formula in its calibration space (6656-height reference)."""
    distance = max(0.0, V1_DIST_INTERCEPT + V1_DIST_SLOPE * ref_y_offset)
    size = V1_SIZE_COEF * distance ** V1_SIZE_EXP if distance > 0 else 0.0
    if size > V1_MAX or distance == 0:
        size = V1_MAX
    if size < V1_MIN:
        size = V1_MIN
    return size


def v1_raw_side(y_px, pano_w, pano_h):
    """predict_crop_size verbatim: native pixels fed straight into the formula."""
    return _v1_reference_size(pano_h / 2 - y_px)


def v1_norm_side(y_px, pano_w, pano_h):
    """The SidewalkWebpage CropService port: compute in reference space, scale
    back. Bit-identical to v1_raw_side at pano_h == 6656."""
    ref_offset = (pano_h / 2 - y_px) * (V1_REF_HEIGHT / pano_h)
    return _v1_reference_size(ref_offset) * (pano_h / V1_REF_HEIGHT)


def geo_v15_side(y_px, pano_w, pano_h):
    """Flat-ground geometry: depression -> distance -> apparent ramp width ->
    pad to the context band. Monotone: nearer (larger depression) -> larger."""
    depression = (y_px / pano_h - 0.5) * math.pi  # radians below horizon
    if depression <= 0:
        d = GEO_D_MAX_M
    else:
        d = min(max(GEO_CAM_H_M / math.tan(depression), GEO_D_MIN_M), GEO_D_MAX_M)
    apparent_px = (GEO_RAMP_W_M / d) * (pano_w / (2 * math.pi))
    return apparent_px / GEO_TARGET_RATIO


SIDE_FNS = {"v1-raw": v1_raw_side, "v1-norm": v1_norm_side, "geo-v1.5": geo_v15_side}


# ---------------------------------------------------------------------------
# Window geometry (mirrors CropRunner.compute_crop_box: x wraps at the seam,
# y clamps by shifting, size capped at both pano dimensions)

def crop_window(x_px, y_px, side, pano_w, pano_h):
    """(left, top, size, shifted) — integers, 0 <= left < pano_w."""
    size = min(int(round(side)), pano_w, pano_h)
    left = int(round(x_px - size / 2)) % pano_w
    ideal_top = int(round(y_px - size / 2))
    top = max(0, min(ideal_top, pano_h - size))
    return left, top, size, top != ideal_top


def box_pixels(box, pano_w, pano_h, wrap_x=False):
    """Gold box in native pixels: (x0, y0, x1, y1).

    x is clamped to the frame by default (manual_labels never wrap). With
    ``wrap_x`` (bundle extent gold, #116) x0/x1 are left unclamped — x0 may be
    negative or x1 > pano_w for a box wrapping the equirectangular seam; the
    circular margin math in :func:`box_margins` handles either form. y always
    clamps (the poles don't wrap).
    """
    cx, cy, w, h = box
    x0 = (cx - w / 2) * pano_w
    x1 = (cx + w / 2) * pano_w
    if not wrap_x:
        x0, x1 = max(0.0, x0), min(float(pano_w), x1)
    y0 = max(0.0, (cy - h / 2) * pano_h)
    y1 = min(float(pano_h), (cy + h / 2) * pano_h)
    return x0, y0, x1, y1


def box_margins(box_px, window, pano_w):
    """Signed (left, right, top, bottom) margins of the box inside the window,
    in pixels. The x offset is computed circularly (the window may cross the
    equirectangular seam), signed into [-pano_w/2, pano_w/2) so a box just left
    of the window reads as a small negative margin, not a huge one.
    """
    x0, y0, x1, y1 = box_px
    left, top, size, _ = window
    rel = ((x0 - left) + pano_w / 2) % pano_w - pano_w / 2
    left_m = rel
    right_m = size - (rel + (x1 - x0))
    top_m = y0 - top
    bottom_m = (top + size) - y1
    return left_m, right_m, top_m, bottom_m


# ---------------------------------------------------------------------------
# Scoring

def depression_deg(cy):
    return (cy - 0.5) * 180.0


def stratum_label(dep_deg):
    for lo, hi, label in STRATA:
        if lo <= dep_deg < hi:
            return label
    return STRATA[0][2] if dep_deg < STRATA[0][0] else STRATA[-1][2]


def required_side(box, prompt_xy_px, pano_w, pano_h, wrap_x=False):
    """Smallest square side that contains the box when centred on the prompt.

    A property of the (box, prompt) pair, not of any rule — which is what makes it
    the denominator that separates "wrong scale constant" from "wrong shape". A
    rule's ``predicted_side / required_side`` is scale-free: its *spread* is the
    rule's real accuracy, while its *median* is only a constant anyone can retune.
    Containment and context ratio are both monotone in window size, so without this
    the biggest rule always wins on one and loses on the other.

    Ignores the near-pole clamp-by-shift, which can only help containment, so this
    is an upper bound on what is truly required (0/227 v1 rows and 6/227 geo rows
    shift at all on richmond).
    """
    x0, y0, x1, y1 = box_pixels(box, pano_w, pano_h, wrap_x=wrap_x)
    px, py = prompt_xy_px
    dx0 = ((px - x0) + pano_w / 2) % pano_w - pano_w / 2   # circular: seam-safe
    dx1 = ((x1 - px) + pano_w / 2) % pano_w - pano_w / 2
    return 2.0 * max(dx0, dx1, py - y0, y1 - py)


def score_box(box, prompt_xy_px, rule, pano_w, pano_h, wrap_x=False):
    """One row of the per-box table (dict).

    ``road_margin_ratio`` is the directional road-context metric: how far the
    window extends below the box's bottom edge — the ramp-street junction under
    the box rule (approximate for oblique ramps, whose lowest extremity is only
    near the junction) — in units of box height. Negative = the window cuts the
    ramp's street edge off. Only meaningful against real extent gold; against
    manual_labels' near-point marks it is reported but uninterpretable (#114).
    NOTE it carries information only in ``detection`` mode: with the window centred
    on the box (``gold-center``) it collapses to ``(side / box_h - 1) / 2``, a
    restatement of the window size, not evidence about placement.

    ``context_ratio`` is the box's LONGEST side over the window side; ``context_ratio_h``
    and ``_v`` split it per axis, because curb-ramp aprons are strongly anisotropic in an
    equirect (richmond median 3.4:1) and a square window sized to hold the width
    over-provisions the height by that factor.
    """
    side = SIDE_FNS[rule](prompt_xy_px[1], pano_w, pano_h)
    window = crop_window(prompt_xy_px[0], prompt_xy_px[1], side, pano_w, pano_h)
    bpx = box_pixels(box, pano_w, pano_h, wrap_x=wrap_x)
    margins = box_margins(bpx, window, pano_w)
    size = window[2]
    box_w = bpx[2] - bpx[0]
    box_h = bpx[3] - bpx[1]
    box_side = max(box_w, box_h)
    margin_norm = min(margins) / size
    req = required_side(box, prompt_xy_px, pano_w, pano_h, wrap_x=wrap_x)
    return {
        "predicted_side": size,
        "required_side": req,
        "size_ratio": size / req if req > 0 else float("nan"),
        "contained": margin_norm >= 0.0,
        "margin_norm": margin_norm,
        "context_ratio": box_side / size,
        "context_ratio_h": box_w / size,
        "context_ratio_v": box_h / size,
        "box_side_px": box_side,
        "box_w_px": box_w,
        "box_h_px": box_h,
        "box_aspect": box_w / box_h if box_h > 0 else float("nan"),
        "road_margin_ratio": margins[3] / box_h if box_h > 0 else float("nan"),
        "shifted": window[3],
        "window": window,
        # Not written to the CSV (DictWriter ignores extras) — what rescale_sweep needs
        # to re-run the real geometry at another k instead of extrapolating.
        "raw_side": side,
        "prompt_x": prompt_xy_px[0],
        "prompt_y": prompt_xy_px[1],
        "box_px": bpx,
    }


def match_detections(detections, boxes, min_confidence):
    """Greedy confidence-ordered match of detections to gold-box centers.

    Returns (assignments, kept_detections): ``assignments[i]`` is the box index
    claimed by kept detection ``i`` or -1. Radius/scaling identical to
    rampnet.detection_eval.score_pano.
    """
    kept = [d for d in detections if float(d.get("confidence", 0.0)) >= min_confidence]
    kept.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    pred_pts = [(float(d["x_normalized"]), float(d["y_normalized"])) for d in kept]
    centers = [(cx, cy) for cx, cy, _, _ in boxes]
    assignments = greedy_match(pred_pts, centers, radius_sq_for(),
                               PANO_SCALE_X, PANO_SCALE_Y)
    return [a[0] for a in assignments], kept


def run_eval(gold, records, min_confidence=OPERATIONAL_CONFIDENCE):
    """Score every rule in both prompt modes. Returns (rows, coverage)."""
    rows = []
    coverage = {"boxes_total": 0, "boxes_covered": 0, "detections_kept": 0,
                "detections_unmatched": 0, "panos_missing_record": 0}
    for pano_id, boxes in gold.items():
        rec = records.get(pano_id)
        if rec is None:
            coverage["panos_missing_record"] += 1
            continue
        pano_w, pano_h = rec["width"], rec["height"]
        coverage["boxes_total"] += len(boxes)

        det_assign, kept = match_detections(rec["detections"], boxes, min_confidence)
        coverage["detections_kept"] += len(kept)
        coverage["detections_unmatched"] += sum(1 for a in det_assign if a < 0)
        det_for_box = {}
        for det, box_idx in zip(kept, det_assign):
            if box_idx >= 0:
                det_for_box[box_idx] = det
        coverage["boxes_covered"] += len(det_for_box)

        for box_idx, box in enumerate(boxes):
            cx, cy = box[0], box[1]
            dep = depression_deg(cy)
            base = {
                "pano_id": pano_id, "box_index": box_idx,
                "pano_width": pano_w, "pano_height": pano_h,
                "depression_deg": dep, "stratum": stratum_label(dep),
            }
            prompts = {"gold-center": (cx * pano_w, cy * pano_h)}
            det = det_for_box.get(box_idx)
            if det is not None:
                prompts["detection"] = (float(det["x_normalized"]) * pano_w,
                                        float(det["y_normalized"]) * pano_h)
            for mode, prompt in prompts.items():
                for rule in RULES:
                    row = dict(base, mode=mode, rule=rule,
                               **score_box(box, prompt, rule, pano_w, pano_h))
                    rows.append(row)
    return rows, coverage


def match_prompts_to_detections(detections, points, min_confidence):
    """{point_index: detection} — greedy confidence-ordered, same radius as everywhere.

    For items whose key carries no adjudicated detection linkage. ``det:<i>`` items
    already know their detection (point review established it); ``gold:<i>`` items from
    ``--from-manual-labels`` do not, and without this they could never be scored in
    detection mode at all — the GSV arm would silently produce gold-center-only numbers
    that are not comparable to a bundle whose keys are ``det:``.
    """
    kept = [d for d in detections if float(d.get("confidence", 0.0)) >= min_confidence]
    kept.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    pred_pts = [(float(d["x_normalized"]), float(d["y_normalized"])) for d in kept]
    assignments = greedy_match(pred_pts, points, radius_sq_for(),
                               PANO_SCALE_X, PANO_SCALE_Y)
    return {a[0]: det for det, a in zip(kept, assignments) if a[0] >= 0}


def run_bundle_eval(gold, prompts, records, min_confidence=OPERATIONAL_CONFIDENCE):
    """Score every rule against box_gallery extent gold (#116).

    ``detection`` mode needs the point production would actually cut a crop at:

    - ``det:<i>`` — the item's own recorded prompt point, the adjudicated detection
      position. No re-matching, no matching noise.
    - ``gold:<i>`` (``--from-manual-labels``) — matched here against the record's
      operational detections, exactly as the manual_labels flow does.
    - ``missed:<i>`` — no detection by definition, so gold-center only: production
      cuts no crop where the model found nothing.
    """
    rows = []
    coverage = {"boxes_total": 0, "det_prompted": 0, "gold_matched": 0,
                "gold_unmatched": 0, "missed_no_detection": 0, "det_missing_point": 0,
                "panos_missing_record": 0, "pano_heights": {}}
    for pano_id, boxes in gold.items():
        rec = records.get(pano_id)
        if rec is None:
            coverage["panos_missing_record"] += 1
            continue
        pano_w, pano_h = rec["width"], rec["height"]
        h_key = str(pano_h)
        coverage["pano_heights"][h_key] = coverage["pano_heights"].get(h_key, 0) + 1

        # Items with no adjudicated detection linkage get one greedy match pass per pano.
        needs_match = [i for i, _ in enumerate(boxes)
                       if prompts[(pano_id, i)]["key"].startswith("gold:")]
        matched = {}
        if needs_match:
            pts = [(prompts[(pano_id, i)]["point"]["x"], prompts[(pano_id, i)]["point"]["y"])
                   if prompts[(pano_id, i)].get("point")
                   else (boxes[i][0], boxes[i][1]) for i in needs_match]
            by_local = match_prompts_to_detections(rec["detections"], pts, min_confidence)
            matched = {needs_match[j]: det for j, det in by_local.items()}

        for box_idx, box in enumerate(boxes):
            coverage["boxes_total"] += 1
            info = prompts[(pano_id, box_idx)]
            dep = depression_deg(box[1])
            base = {
                "pano_id": pano_id, "box_index": box_idx, "key": info["key"],
                "pano_width": pano_w, "pano_height": pano_h,
                "depression_deg": dep, "stratum": stratum_label(dep),
            }
            modes = {"gold-center": (box[0] * pano_w, box[1] * pano_h)}
            if info["is_detection"]:
                if info.get("point"):
                    modes["detection"] = (info["point"]["x"] * pano_w,
                                          info["point"]["y"] * pano_h)
                    coverage["det_prompted"] += 1
                else:
                    # An adjudicated detection with no recorded position: it cannot be
                    # scored in detection mode, and folding it into a "no detection"
                    # bucket would misreport it as a model miss.
                    coverage["det_missing_point"] += 1
            elif box_idx in matched:
                det = matched[box_idx]
                modes["detection"] = (float(det["x_normalized"]) * pano_w,
                                      float(det["y_normalized"]) * pano_h)
                coverage["gold_matched"] += 1
            elif info["key"].startswith("gold:"):
                coverage["gold_unmatched"] += 1
            else:
                coverage["missed_no_detection"] += 1
            for mode, prompt in modes.items():
                for rule in RULES:
                    rows.append(dict(base, mode=mode, rule=rule,
                                     **score_box(box, prompt, rule, pano_w, pano_h,
                                                 wrap_x=True)))
    return rows, coverage


# ---------------------------------------------------------------------------
# Aggregation / report

def _quantiles(values, qs=(0.1, 0.5, 0.9)):
    if not values:
        return {q: float("nan") for q in qs}
    vals = sorted(values)
    out = {}
    for q in qs:
        idx = q * (len(vals) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        out[q] = vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)
    return out


def summarize(rows):
    """Nested summary: mode -> rule -> overall + per-stratum stats."""
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        summary[mode] = {}
        for rule in RULES:
            sel = [r for r in rows if r["mode"] == mode and r["rule"] == rule]
            if not sel:
                continue
            summary[mode][rule] = _stats(sel)
            strata = {}
            for _, _, label in STRATA:
                ssel = [r for r in sel if r["stratum"] == label]
                if ssel:
                    strata[label] = _stats(ssel)
            summary[mode][rule]["strata"] = strata
    return summary


def _finite(values):
    return [v for v in values if not math.isnan(v)]


def _stats(sel):
    n = len(sel)
    contained = sum(1 for r in sel if r["contained"])
    ctx = [r["context_ratio"] for r in sel]
    ctx_q = _quantiles(ctx)
    lo, hi = wilson_interval(contained, n)
    road = [r["road_margin_ratio"] for r in sel
            if not math.isnan(r.get("road_margin_ratio", float("nan")))]
    road_q = _quantiles(road)
    # Scale-free calibration: predicted / required side. The median is a constant
    # anyone can retune; the p90/p10 spread is the part a rule has to earn, and it
    # is the only comparison here that a bigger window cannot win by being bigger.
    ratio = _finite([r["size_ratio"] for r in sel])
    ratio_q = _quantiles(ratio)
    aspect_q = _quantiles(_finite([r["box_aspect"] for r in sel]))
    return {
        "n": n,
        "containment": contained / n,
        "containment_ci": [lo, hi],
        "size_ratio_p10": ratio_q[0.1],
        "size_ratio_p50": ratio_q[0.5],
        "size_ratio_p90": ratio_q[0.9],
        "size_ratio_spread": (ratio_q[0.9] / ratio_q[0.1]) if ratio_q[0.1] > 0 else float("nan"),
        "box_aspect_p50": aspect_q[0.5],
        "context_ratio_h_p50": _quantiles([r["context_ratio_h"] for r in sel])[0.5],
        "context_ratio_v_p50": _quantiles([r["context_ratio_v"] for r in sel])[0.5],
        "context_ratio_p10": ctx_q[0.1],
        "context_ratio_p50": ctx_q[0.5],
        "context_ratio_p90": ctx_q[0.9],
        "context_in_band_10_15": sum(1 for c in ctx if 0.10 <= c <= 0.15) / n,
        "context_in_band_05_20": sum(1 for c in ctx if 0.05 <= c <= 0.20) / n,
        "margin_norm_p50": _quantiles([r["margin_norm"] for r in sel])[0.5],
        "predicted_side_p50": _quantiles([float(r["predicted_side"]) for r in sel])[0.5],
        # Directional road context (window extension below the box bottom, in
        # box heights). "cut" = the window clips the ramp's street edge.
        "road_margin_p10": road_q[0.1],
        "road_margin_p50": road_q[0.5],
        "road_margin_p90": road_q[0.9],
        "road_edge_cut": (sum(1 for v in road if v < 0) / len(road)) if road else float("nan"),
    }


def format_report(summary, header_lines):
    lines = list(header_lines)
    add = lines.append
    for mode, rules in summary.items():
        add("")
        add(f"== prompt mode: {mode} ==")
        add(f"{'rule':<10} {'n':>5} {'contain':>8} {'95% CI':>15} "
            f"{'ctx p50':>8} {'ctx p10-p90':>13} {'in 10-15%':>9} {'in 5-20%':>9} "
            f"{'road p50':>8} {'edgecut':>7} {'side p50':>9}")
        for rule, s in rules.items():
            ci = f"[{s['containment_ci'][0]:.3f},{s['containment_ci'][1]:.3f}]"
            add(f"{rule:<10} {s['n']:>5} {s['containment']:>8.3f} {ci:>15} "
                f"{s['context_ratio_p50']:>8.3f} "
                f"{s['context_ratio_p10']:>6.3f}-{s['context_ratio_p90']:.3f} "
                f"{s['context_in_band_10_15']:>9.3f} {s['context_in_band_05_20']:>9.3f} "
                f"{s['road_margin_p50']:>8.2f} {s['road_edge_cut']:>7.3f} "
                f"{s['predicted_side_p50']:>9.0f}")
        add("")
        add("   scale-free calibration (predicted side / the side actually required).")
        add("   Containment and ctx are both monotone in window size, so read the")
        add("   SPREAD: the median is a constant, the spread is the rule's accuracy.")
        add(f"   {'rule':<10} {'ratio p10':>10} {'p50':>8} {'p90':>8} {'p90/p10':>9} "
            f"{'ctx_h p50':>10} {'ctx_v p50':>10} {'box aspect':>11}")
        for rule, s in rules.items():
            add(f"   {rule:<10} {s['size_ratio_p10']:>10.2f} {s['size_ratio_p50']:>8.2f} "
                f"{s['size_ratio_p90']:>8.2f} {s['size_ratio_spread']:>9.2f} "
                f"{s['context_ratio_h_p50']:>10.3f} {s['context_ratio_v_p50']:>10.3f} "
                f"{s['box_aspect_p50']:>11.2f}")
        if mode == "gold-center":
            add("   (gold-center 'road p50' is (side/box_h - 1)/2 by construction — a")
            add("    restatement of window size, not placement evidence. Use detection mode.)")
        add("")
        add("   by depression (flat-ground distance at 2.5 m camera height):")
        for rule, s in rules.items():
            for label, ss in s["strata"].items():
                add(f"   {rule:<10} {label:<16} n={ss['n']:>5}  "
                    f"contain={ss['containment']:.3f}  ctx p50={ss['context_ratio_p50']:.3f}  "
                    f"road p50={ss['road_margin_p50']:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rescale sweep

RESCALE_KS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def rescale_sweep(rows, mode="detection", ks=RESCALE_KS):
    """{rule: [{k, containment, ctx_p50, side_p50, capped}, ...]} — each rule's output
    multiplied by a constant, re-scored with the real window geometry.

    The point of the whole exercise. A rule that contains 99.6% of aprons while another
    contains 26% has not been shown to be a better rule if it is simply 4.5x larger:
    both statements are consequences of one constant. Sweeping it puts every rule on the
    same containment and asks what each then costs in window size, which is the question
    a production crop rule actually has to answer.

    Exact, not extrapolated: ``crop_window`` and ``box_margins`` re-run at each k, so the
    pano-dimension cap and the clamp-by-shift apply (a rule cannot be scaled past the
    image it cuts from — ``capped`` counts where that bites).
    """
    out = {}
    for rule in RULES:
        sel = [r for r in rows if r["mode"] == mode and r["rule"] == rule]
        if not sel:
            continue
        per_k = []
        for k in ks:
            contained = capped = 0
            ctx, sides = [], []
            for r in sel:
                pano_w, pano_h = r["pano_width"], r["pano_height"]
                window = crop_window(r["prompt_x"], r["prompt_y"],
                                     k * r["raw_side"], pano_w, pano_h)
                if k * r["raw_side"] > min(pano_w, pano_h):
                    capped += 1
                margins = box_margins(r["box_px"], window, pano_w)
                contained += 1 if min(margins) >= 0 else 0
                ctx.append(r["box_side_px"] / window[2])
                sides.append(float(window[2]))
            q_ctx, q_side = _quantiles(ctx), _quantiles(sides)
            per_k.append({"k": k, "n": len(sel), "containment": contained / len(sel),
                          "ctx_p50": q_ctx[0.5], "ctx_p90": q_ctx[0.9],
                          "side_p50": q_side[0.5], "capped": capped})
        out[rule] = per_k
    return out


def _sweep_mode(rows):
    """Sweep the production-realistic mode when it exists, else gold-center."""
    return "detection" if any(r["mode"] == "detection" for r in rows) else "gold-center"


def format_sweep(sweep, mode):
    lines = ["", f"== constant-rescale sweep ({mode} mode) ==",
             "   Every rule x k, re-scored with the real window geometry. Compare rules",
             "   at MATCHED containment: that is the comparison a scale constant cannot win.",
             f"   {'rule':<10} {'k':>5} {'contain':>8} {'ctx p50':>9} {'ctx p90':>9} "
             f"{'side p50':>9} {'capped':>7}"]
    for rule, per_k in sweep.items():
        for row in per_k:
            lines.append(f"   {rule:<10} {row['k']:>5.1f} {row['containment']:>8.3f} "
                         f"{row['ctx_p50']:>9.3f} {row['ctx_p90']:>9.3f} "
                         f"{row['side_p50']:>9.0f} {row['capped']:>7}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outputs

CSV_FIELDS = ["pano_id", "box_index", "key", "mode", "rule", "pano_width", "pano_height",
              "depression_deg", "stratum", "predicted_side", "required_side", "size_ratio",
              "contained", "margin_norm", "context_ratio", "context_ratio_h",
              "context_ratio_v", "box_side_px", "box_w_px", "box_h_px", "box_aspect",
              "road_margin_ratio", "shifted"]


def write_outputs(rows, summary, coverage, inputs, basename="crop_window_eval",
                  sweep=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    # newline="\n" on both writers: these are committed artifacts, and text mode on
    # Windows would rewrite every line as CRLF and dirty the tree on a re-run.
    csv_path = os.path.join(OUT_DIR, f"{basename}_per_box.csv")
    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["pano_id"], r["box_index"],
                                               r["mode"], r["rule"])):
            writer.writerow(row)
    with open(csv_path, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()

    json_path = os.path.join(OUT_DIR, f"{basename}.json")
    payload = {
        "issue": 114,
        "inputs": inputs,
        "coverage": coverage,
        "per_box_csv_sha256": csv_sha,
        "constants": {
            "v1": {"intercept": V1_DIST_INTERCEPT, "slope": V1_DIST_SLOPE,
                   "coef": V1_SIZE_COEF, "exp": V1_SIZE_EXP,
                   "clamp": [V1_MIN, V1_MAX], "ref_height": V1_REF_HEIGHT},
            "geo": {"cam_h_m": GEO_CAM_H_M, "ramp_w_m": GEO_RAMP_W_M,
                    "target_ratio": GEO_TARGET_RATIO,
                    "d_clamp_m": [GEO_D_MIN_M, GEO_D_MAX_M]},
        },
        "summary": summary,
    }
    if sweep:
        payload["rescale_sweep"] = sweep
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Gallery (pixels required; everything above runs without them)

GALLERY_DIR = os.path.join(OUT_DIR, "crop_window_eval_gallery")
RULE_COLORS = {"v1-raw": (255, 96, 96), "v1-norm": (255, 200, 64),
               "geo-v1.5": (96, 160, 255)}
GOLD_COLOR = (64, 255, 96)


def _find_pano_image(pano_id, dirs=None):
    if dirs is None:
        dirs = (os.path.join(BUNDLE_DIR, "panos"),
                os.path.join(GALLERY_DIR, "panos_fresh"))
    for directory in dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(directory, pano_id + ext)
            if os.path.exists(path):
                return path
    return None


def fetch_sample(gold, records, n, seed=114):
    """Fetch up to n gold panos fresh from GSV via streetlevel (optional dep)
    into the gallery's own directory. View-only bytes: never written into the
    bundle, not benchmark-fidelity (canonical imagery is the HF test split via
    scripts/fetch_manual_gold.py). Decayed panos are skipped."""
    try:
        from streetlevel import streetview
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(f"--fetch-sample needs streetlevel + pillow: {exc}")
    import random

    dest = os.path.join(GALLERY_DIR, "panos_fresh")
    os.makedirs(dest, exist_ok=True)
    candidates = [p for p in sorted(gold) if gold[p] and p in records]
    random.Random(seed).shuffle(candidates)
    fetched = skipped = 0
    for pano_id in candidates:
        if fetched >= n:
            break
        if _find_pano_image(pano_id):
            fetched += 1  # already available locally
            continue
        try:
            pano = streetview.find_panorama_by_id(pano_id)
            if pano is None:
                skipped += 1
                continue
            zoom = None
            for level, size in enumerate(pano.image_sizes or []):
                if size.x >= 4096:
                    zoom = level
                    break
            if zoom is None:
                zoom = len(pano.image_sizes) - 1 if pano.image_sizes else 2
            img = streetview.get_panorama(pano, zoom=zoom)
            img = img.convert("RGB").resize((4096, 2048), Image.LANCZOS)
            img.save(os.path.join(dest, pano_id + ".jpg"), quality=90)
            fetched += 1
        except Exception as exc:  # network fetch: skip and continue
            print(f"  fetch failed for {pano_id}: {exc}")
            skipped += 1
    print(f"fetch-sample: {fetched} panos available, {skipped} skipped/decayed")


def render_gallery(rows, gold, records, limit, gallery_dir=None, panos_dirs=None,
                   wrap_x=False):
    """Overlay crops: gold box (green) + each rule's window, centered on the
    gold-center prompt, for boxes whose pano image is available locally."""
    from PIL import Image, ImageDraw

    gallery_dir = gallery_dir or GALLERY_DIR
    os.makedirs(gallery_dir, exist_ok=True)
    by_box = {}
    for r in rows:
        if r["mode"] != "gold-center":
            continue
        by_box.setdefault((r["pano_id"], r["box_index"]), {})[r["rule"]] = r

    entries = []
    for (pano_id, box_index), per_rule in sorted(by_box.items()):
        if len(entries) >= limit:
            break
        path = _find_pano_image(pano_id, panos_dirs)
        if path is None:
            continue
        pano = Image.open(path).convert("RGB")
        pano_w, pano_h = pano.size
        rec = records[pano_id]
        if (pano_w, pano_h) != (rec["width"], rec["height"]):
            pano = pano.resize((rec["width"], rec["height"]), Image.LANCZOS)
            pano_w, pano_h = pano.size

        windows = [per_rule[rule]["window"] for rule in RULES if rule in per_rule]
        view = min(pano_h, int(1.5 * max(w[2] for w in windows)))
        any_row = next(iter(per_rule.values()))
        # Prompt = window center of any rule (all share the gold-center prompt).
        left0, top0, size0, _ = any_row["window"]
        cx_px = (left0 + size0 / 2) % pano_w
        cy_px = top0 + size0 / 2
        vleft = int(round(cx_px - view / 2)) % pano_w
        vtop = max(0, min(int(round(cy_px - view / 2)), pano_h - view))
        if vleft + view <= pano_w:
            tile = pano.crop((vleft, vtop, vleft + view, vtop + view))
        else:
            tile = Image.new("RGB", (view, view))
            first = pano_w - vleft
            tile.paste(pano.crop((vleft, vtop, pano_w, vtop + view)), (0, 0))
            tile.paste(pano.crop((0, vtop, view - first, vtop + view)), (first, 0))
        draw = ImageDraw.Draw(tile)

        def to_view_x(x):
            return (x - vleft) % pano_w

        for rule in RULES:
            if rule not in per_rule:
                continue
            left, top, size, _ = per_rule[rule]["window"]
            x0 = to_view_x(left)
            draw.rectangle([x0, top - vtop, x0 + size, top - vtop + size],
                           outline=RULE_COLORS[rule], width=3)
        b = box_pixels(gold[pano_id][box_index], pano_w, pano_h, wrap_x=wrap_x)
        gx0 = to_view_x(b[0])
        draw.rectangle([gx0, b[1] - vtop, gx0 + (b[2] - b[0]), b[3] - vtop],
                       outline=GOLD_COLOR, width=3)

        if view > 900:
            tile = tile.resize((900, 900), Image.LANCZOS)
        name = f"{pano_id}_{box_index}.jpg"
        tile.save(os.path.join(gallery_dir, name), quality=88)
        entries.append((name, pano_id, box_index, per_rule))

    index = os.path.join(gallery_dir, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>crop_window_eval gallery</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee}"
                ".card{display:inline-block;margin:8px;vertical-align:top}"
                "img{display:block;max-width:440px}"
                "table{font-size:12px;border-collapse:collapse}"
                "td,th{padding:1px 6px;text-align:right}</style>"
                "<h2>Gold box (green) vs predicted windows "
                "(red=v1-raw, orange=v1-norm, blue=geo-v1.5)</h2>")
        for name, pano_id, box_index, per_rule in entries:
            f.write(f"<div class='card'><img src='{name}'>"
                    f"<table><tr><th>rule</th><th>side</th><th>ctx</th>"
                    f"<th>margin</th><th>in</th></tr>")
            for rule in RULES:
                if rule not in per_rule:
                    continue
                r = per_rule[rule]
                f.write(f"<tr><td>{rule}</td><td>{r['predicted_side']}</td>"
                        f"<td>{r['context_ratio']:.3f}</td>"
                        f"<td>{r['margin_norm']:.3f}</td>"
                        f"<td>{'Y' if r['contained'] else 'N'}</td></tr>")
            f.write(f"</table>{pano_id} #{box_index}</div>")
    print(f"gallery: {len(entries)} overlay crops -> {index}")


# ---------------------------------------------------------------------------

def main(argv=None):
    global GEO_CAM_H_M
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bundle", metavar="DIR",
                        help="score against a box_gallery extent-gold export "
                             "(<DIR>/boxes.json + <DIR>/records.jsonl, #116) instead "
                             "of manual_labels")
    parser.add_argument("--min-confidence", type=float, default=OPERATIONAL_CONFIDENCE,
                        help="detection-match confidence floor (production operational "
                             "threshold; default %(default)s)")
    parser.add_argument("--cam-height", type=float, default=GEO_CAM_H_M, metavar="M",
                        help="geo-v1.5 camera height in metres (default %(default)s). "
                             "The rule's one free parameter: too high overestimates "
                             "distance and undersizes the window.")
    parser.add_argument("--no-sweep", action="store_true",
                        help="skip the constant-rescale sweep (on by default: it is what "
                             "separates a rule's accuracy from its scale constant)")
    parser.add_argument("--gallery", type=int, default=0, metavar="N",
                        help="render up to N overlay crops (needs local pano imagery)")
    parser.add_argument("--fetch-sample", type=int, default=0, metavar="N",
                        help="manual mode: fetch up to N gold panos fresh from GSV for "
                             "the gallery (view-only bytes; optional streetlevel dep)")
    args = parser.parse_args(argv)
    GEO_CAM_H_M = args.cam_height   # geo_v15_side reads the module global at call time

    if args.bundle:
        bundle = args.bundle.rstrip("/\\")
        gold, prompts, meta = load_bundle_boxes(bundle)
        records = load_records(os.path.join(bundle, "records.jsonl"))
        rows, coverage = run_bundle_eval(gold, prompts, records, args.min_confidence)
        summary = summarize(rows)
        sweep_mode = _sweep_mode(rows)
        sweep = None if args.no_sweep else rescale_sweep(rows, sweep_mode)
        rule = meta.get("box_rule") or {}
        basename = f"crop_window_eval_{meta['run_name']}"
        inputs = {
            "boxes": f"benchmark/{meta['run_name']}/boxes.json",
            "records": f"benchmark/{meta['run_name']}/records.jsonl",
            "box_rule_version": rule.get("version"),
            "crop_px_by_pano_dims": meta.get("crop_px_by_pano_dims"),
            "cam_height_m": GEO_CAM_H_M,
            "n_boxed": meta["n_boxed"], "n_cant": meta["n_cant"],
            "n_other": meta["n_other"], "n_edge_flag": meta["n_edge_flag"],
            "n_adjudicated": meta.get("n_adjudicated"),
        }
        prompted = (f"det-prompted: {coverage['det_prompted']}"
                    if coverage["det_prompted"] else
                    f"detection-matched: {coverage['gold_matched']} "
                    f"(unmatched: {coverage['gold_unmatched']})")
        header = [
            f"crop_window_eval --bundle {meta['run_name']} — whole-apron extent gold "
            f"(box rule v{rule.get('version')}) vs sizing rules (#114/#116)",
            f"boxes: {coverage['boxes_total']} boxed ({meta['n_cant']} can't-determine, "
            f"{meta['n_edge_flag']} edge-flagged)   {prompted}   "
            f"no detection (gold-center only): "
            f"{coverage['missed_no_detection'] + coverage['gold_unmatched']}   "
            f"pano heights: {coverage['pano_heights']}",
        ]
        if meta.get("crop_px_by_pano_dims"):
            header.append(f"gold drawn at: {meta['crop_px_by_pano_dims']} "
                          f"(pano dims -> annotation crop px)")
        if meta.get("completeness_warning"):
            header.append(f"WARNING: {meta['completeness_warning']}")
        csv_path, json_path = write_outputs(rows, summary, coverage, inputs, basename,
                                            sweep)
        print(format_report(summary, header))
        if sweep:
            print(format_sweep(sweep, sweep_mode))
        print(f"\nper-box CSV: {csv_path}\nsummary JSON: {json_path}")
        if args.gallery:
            render_gallery(rows, gold, records, args.gallery,
                           gallery_dir=os.path.join(OUT_DIR, f"{basename}_gallery"),
                           panos_dirs=(os.path.join(bundle, "panos"),), wrap_x=True)
        return

    gold = load_gold()
    records = load_records()
    rows, coverage = run_eval(gold, records, args.min_confidence)
    summary = summarize(rows)
    sweep_mode = _sweep_mode(rows)
    sweep = None if args.no_sweep else rescale_sweep(rows, sweep_mode)
    inputs = {
        "labels_dir": "manual_labels",
        "records": "benchmark/manual_gold/records.jsonl",
        "min_confidence": args.min_confidence,
        "cam_height_m": GEO_CAM_H_M,
        "n_boxes": coverage["boxes_total"],
    }
    header = [
        "crop_window_eval — manual_gold boxes vs crop-window sizing rules (issue #114)",
        f"boxes: {coverage['boxes_total']}   "
        f"covered by a >= {args.min_confidence:.2f} detection: {coverage['boxes_covered']}   "
        f"unmatched detections (would crop non-ramps): {coverage['detections_unmatched']}",
    ]
    csv_path, json_path = write_outputs(rows, summary, coverage, inputs,
                                        sweep=sweep)
    print(format_report(summary, header))
    if sweep:
        print(format_sweep(sweep, sweep_mode))
    print(f"\nper-box CSV: {csv_path}\nsummary JSON: {json_path}")

    if args.fetch_sample:
        fetch_sample(gold, records, args.fetch_sample)
    if args.gallery:
        render_gallery(rows, gold, records, args.gallery)


if __name__ == "__main__":
    main()
