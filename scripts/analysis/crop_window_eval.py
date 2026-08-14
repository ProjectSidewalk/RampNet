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
- **context ratio** — box max side / window side. The consumer-requirements
  survey in sidewalk-panorama-tools converged on ~10-15% as the useful band
  (a survey, not a measurement — treated as a target band only).

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

Run (from the repo root):

    python scripts/analysis/crop_window_eval.py                # numbers + JSON + CSV
    python scripts/analysis/crop_window_eval.py --gallery 60   # + overlay gallery
    python scripts/analysis/crop_window_eval.py --fetch-sample 60 --gallery 60

Outputs:

- ``analysis_out/crop_window_eval.json``       — summary (committed; see .gitignore)
- ``analysis_out/crop_window_eval_per_box.csv``— one row per (box, rule, mode) (ignored)
- ``analysis_out/crop_window_eval_gallery/``   — overlay crops + index.html (ignored)
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


def box_pixels(box, pano_w, pano_h):
    """Gold box in native pixels, clamped to the frame: (x0, y0, x1, y1)."""
    cx, cy, w, h = box
    x0 = max(0.0, (cx - w / 2) * pano_w)
    x1 = min(float(pano_w), (cx + w / 2) * pano_w)
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


def score_box(box, prompt_xy_px, rule, pano_w, pano_h):
    """One row of the per-box table (dict)."""
    side = SIDE_FNS[rule](prompt_xy_px[1], pano_w, pano_h)
    window = crop_window(prompt_xy_px[0], prompt_xy_px[1], side, pano_w, pano_h)
    bpx = box_pixels(box, pano_w, pano_h)
    margins = box_margins(bpx, window, pano_w)
    size = window[2]
    box_side = max(bpx[2] - bpx[0], bpx[3] - bpx[1])
    margin_norm = min(margins) / size
    return {
        "predicted_side": size,
        "contained": margin_norm >= 0.0,
        "margin_norm": margin_norm,
        "context_ratio": box_side / size,
        "box_side_px": box_side,
        "shifted": window[3],
        "window": window,
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


def _stats(sel):
    n = len(sel)
    contained = sum(1 for r in sel if r["contained"])
    ctx = [r["context_ratio"] for r in sel]
    ctx_q = _quantiles(ctx)
    lo, hi = wilson_interval(contained, n)
    return {
        "n": n,
        "containment": contained / n,
        "containment_ci": [lo, hi],
        "context_ratio_p10": ctx_q[0.1],
        "context_ratio_p50": ctx_q[0.5],
        "context_ratio_p90": ctx_q[0.9],
        "context_in_band_10_15": sum(1 for c in ctx if 0.10 <= c <= 0.15) / n,
        "context_in_band_05_20": sum(1 for c in ctx if 0.05 <= c <= 0.20) / n,
        "margin_norm_p50": _quantiles([r["margin_norm"] for r in sel])[0.5],
        "predicted_side_p50": _quantiles([float(r["predicted_side"]) for r in sel])[0.5],
    }


def format_report(summary, coverage, min_confidence):
    lines = []
    add = lines.append
    add("crop_window_eval — manual_gold boxes vs crop-window sizing rules (issue #114)")
    add(f"boxes: {coverage['boxes_total']}   "
        f"covered by a >= {min_confidence:.2f} detection: {coverage['boxes_covered']}   "
        f"unmatched detections (would crop non-ramps): {coverage['detections_unmatched']}")
    for mode, rules in summary.items():
        add("")
        add(f"== prompt mode: {mode} ==")
        add(f"{'rule':<10} {'n':>5} {'contain':>8} {'95% CI':>15} "
            f"{'ctx p50':>8} {'ctx p10-p90':>13} {'in 10-15%':>9} {'in 5-20%':>9} {'side p50':>9}")
        for rule, s in rules.items():
            ci = f"[{s['containment_ci'][0]:.3f},{s['containment_ci'][1]:.3f}]"
            add(f"{rule:<10} {s['n']:>5} {s['containment']:>8.3f} {ci:>15} "
                f"{s['context_ratio_p50']:>8.3f} "
                f"{s['context_ratio_p10']:>6.3f}-{s['context_ratio_p90']:.3f} "
                f"{s['context_in_band_10_15']:>9.3f} {s['context_in_band_05_20']:>9.3f} "
                f"{s['predicted_side_p50']:>9.0f}")
        add("")
        add("   by depression (flat-ground distance at 2.5 m camera height):")
        for rule, s in rules.items():
            for label, ss in s["strata"].items():
                add(f"   {rule:<10} {label:<16} n={ss['n']:>5}  "
                    f"contain={ss['containment']:.3f}  ctx p50={ss['context_ratio_p50']:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outputs

CSV_FIELDS = ["pano_id", "box_index", "mode", "rule", "pano_width", "pano_height",
              "depression_deg", "stratum", "predicted_side", "contained",
              "margin_norm", "context_ratio", "box_side_px", "shifted"]


def write_outputs(rows, summary, coverage, min_confidence):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "crop_window_eval_per_box.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["pano_id"], r["box_index"],
                                               r["mode"], r["rule"])):
            writer.writerow(row)
    with open(csv_path, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()

    json_path = os.path.join(OUT_DIR, "crop_window_eval.json")
    payload = {
        "issue": 114,
        "inputs": {
            "labels_dir": "manual_labels",
            "records": "benchmark/manual_gold/records.jsonl",
            "min_confidence": min_confidence,
            "n_boxes": coverage["boxes_total"],
        },
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Gallery (pixels required; everything above runs without them)

GALLERY_DIR = os.path.join(OUT_DIR, "crop_window_eval_gallery")
RULE_COLORS = {"v1-raw": (255, 96, 96), "v1-norm": (255, 200, 64),
               "geo-v1.5": (96, 160, 255)}
GOLD_COLOR = (64, 255, 96)


def _find_pano_image(pano_id):
    for directory in (os.path.join(BUNDLE_DIR, "panos"),
                      os.path.join(GALLERY_DIR, "panos_fresh")):
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


def render_gallery(rows, gold, records, limit):
    """Overlay crops: gold box (green) + each rule's window, centered on the
    gold-center prompt, for boxes whose pano image is available locally."""
    from PIL import Image, ImageDraw

    os.makedirs(GALLERY_DIR, exist_ok=True)
    by_box = {}
    for r in rows:
        if r["mode"] != "gold-center":
            continue
        by_box.setdefault((r["pano_id"], r["box_index"]), {})[r["rule"]] = r

    entries = []
    for (pano_id, box_index), per_rule in sorted(by_box.items()):
        if len(entries) >= limit:
            break
        path = _find_pano_image(pano_id)
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
        b = box_pixels(gold[pano_id][box_index], pano_w, pano_h)
        gx0 = to_view_x(b[0])
        draw.rectangle([gx0, b[1] - vtop, gx0 + (b[2] - b[0]), b[3] - vtop],
                       outline=GOLD_COLOR, width=3)

        if view > 900:
            tile = tile.resize((900, 900), Image.LANCZOS)
        name = f"{pano_id}_{box_index}.jpg"
        tile.save(os.path.join(GALLERY_DIR, name), quality=88)
        entries.append((name, pano_id, box_index, per_rule))

    index = os.path.join(GALLERY_DIR, "index.html")
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--min-confidence", type=float, default=OPERATIONAL_CONFIDENCE,
                        help="detection-mode confidence floor (production operational "
                             "threshold; default %(default)s)")
    parser.add_argument("--gallery", type=int, default=0, metavar="N",
                        help="render up to N overlay crops (needs local pano imagery)")
    parser.add_argument("--fetch-sample", type=int, default=0, metavar="N",
                        help="fetch up to N gold panos fresh from GSV for the gallery "
                             "(view-only bytes; optional streetlevel dependency)")
    args = parser.parse_args(argv)

    gold = load_gold()
    records = load_records()
    rows, coverage = run_eval(gold, records, args.min_confidence)
    summary = summarize(rows)
    csv_path, json_path = write_outputs(rows, summary, coverage, args.min_confidence)
    print(format_report(summary, coverage, args.min_confidence))
    print(f"\nper-box CSV: {csv_path}\nsummary JSON: {json_path}")

    if args.fetch_sample:
        fetch_sample(gold, records, args.fetch_sample)
    if args.gallery:
        render_gallery(rows, gold, records, args.gallery)


if __name__ == "__main__":
    main()
