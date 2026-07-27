"""Fill ``benchmark/manual_gold/records.jsonl`` with RampNet's detections (issue #58).

Runs the stage-2 model over the fetched gold panos (``scripts/fetch_manual_gold.py``
first) and writes each pano's detections into the bundle records, which is what
the comparison harness's ``rampnet`` baseline row reads (``BundleRampNetDetector``).
GPU work — run it where the other benchmark numbers came from (Hyak/makelab2);
heatmaps are cached per checkpoint+TTA, so a preempted job resumes cheaply.

The inference path is *imported from* ``stage_two/evaluate.py`` (same preprocess,
same TTA, same peak extraction), not re-implemented — that is what makes the
reproduction gate below meaningful.

Peaks are extracted down to a **0.05 floor** (not the 0.5 the city bundles were
extracted at), so RampNet gets a full PR curve / untruncated AP on this split —
the city bundles' truncated-AP caveat (docs/model_comparison.md) doesn't apply
here. The main-table operating point is still set by ``--op-threshold`` at
scoring time; the floor only decides how far down the curve extends.

**Reproduction gate:** after exporting, the script scores the detections against
the manual labels through the harness scorer at conf >= 0.55 and prints the
result next to the published gold-set numbers (P 0.949 / R 0.873, TTA — README
"corrected" row / HF model card). A match validates the fetch (no re-encode
drift), the inference config, the YOLO->point conversion, and the scorer in one
shot — before any paid VLM run on this bundle.

    python scripts/export_gold_records.py --checkpoint <stage2 .pth>
    python scripts/export_gold_records.py --checkpoint <ckpt> --limit 5   # smoke
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "stage_two"))

from evaluate import (  # noqa: E402  (stage_two/evaluate.py)
    MODEL_HEATMAP_SIZE, PEAK_MIN_DISTANCE,
    extract_peaks_from_heatmap, load_trained_model, predict_heatmap,
)
from rampnet.detection_eval import (  # noqa: E402
    aggregate, load_yolo_ground_truths, score_pano,
)
from rampnet.loading import checkpoint_fingerprint  # noqa: E402

BUNDLE_DIR = os.path.join(REPO_ROOT, "benchmark", "manual_gold")
LABELS_DIR = os.path.join(REPO_ROOT, "manual_labels")
PEAK_FLOOR = 0.05
# Published gold-set numbers the gate reproduces (README / HF model card;
# greedy 1:1 matching, radius 0.022, TTA, conf >= GATE_THRESHOLD).
GATE_THRESHOLD = 0.55
PUBLISHED_P, PUBLISHED_R = 0.949, 0.873


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def detect_pano(model, img_path, cache_path, use_tta, floor):
    if os.path.exists(cache_path):
        heatmap = np.load(cache_path)
    else:
        with Image.open(img_path) as img:
            heatmap = predict_heatmap(model, img.convert("RGB"), use_tta)
        np.save(cache_path, heatmap)
    peaks = extract_peaks_from_heatmap(heatmap, min_distance=PEAK_MIN_DISTANCE,
                                       threshold_abs=floor, heatmap_shape=MODEL_HEATMAP_SIZE)
    peaks.sort(key=lambda p: p[2], reverse=True)
    return [{"x_normalized": float(x), "y_normalized": float(y), "confidence": float(c)}
            for x, y, c in peaks]


def gate_report(records, gts, threshold):
    scores = []
    for rec in records:
        pid = rec["pano"]["panorama_id"]
        preds = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                 for d in rec["detections"] if d["confidence"] >= threshold]
        scores.append(score_pano(preds, gts[pid]))
    return aggregate(scores)


def main():
    ap = argparse.ArgumentParser(description="Export RampNet detections into the manual_gold bundle.")
    ap.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint (.pth)")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                    help="Horizontal-flip TTA (default on — the published numbers use it)")
    ap.add_argument("--peak-floor", type=float, default=PEAK_FLOOR,
                    help=f"Peak-extraction floor (default {PEAK_FLOOR}; low on purpose, "
                         "so the PR curve has a tail — NOT the operating point)")
    ap.add_argument("--cache-dir", default=os.path.join(BUNDLE_DIR, "heatmap_cache"),
                    help="Heatmap cache root (keyed by checkpoint fingerprint + TTA)")
    ap.add_argument("--limit", type=int, help="Only process the first N panos (smoke test)")
    args = ap.parse_args()

    records_path = os.path.join(BUNDLE_DIR, "records.jsonl")
    if not os.path.exists(records_path):
        raise SystemExit(f"{records_path} not found — run scripts/fetch_manual_gold.py first")
    records = load_records(records_path)
    if args.limit:
        records = records[:args.limit]

    fingerprint = checkpoint_fingerprint(args.checkpoint)
    cache_dir = os.path.join(args.cache_dir,
                             f"{fingerprint}_{'tta' if args.tta else 'notta'}")
    os.makedirs(cache_dir, exist_ok=True)
    model = load_trained_model(args.checkpoint, MODEL_HEATMAP_SIZE)

    n_dets = 0
    for i, rec in enumerate(records):
        pid = rec["pano"]["panorama_id"]
        img_path = os.path.join(BUNDLE_DIR, "panos", f"{pid}.jpg")
        if not os.path.exists(img_path):
            raise SystemExit(f"{img_path} missing — re-run scripts/fetch_manual_gold.py")
        rec["detections"] = detect_pano(
            model, img_path, os.path.join(cache_dir, f"{pid}_heatmap.npy"),
            args.tta, args.peak_floor)
        n_dets += len(rec["detections"])
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  {i + 1}/{len(records)} panos, {n_dets} detections so far")

    if not args.limit:  # a smoke run must not overwrite the real bundle records
        with open(records_path + ".tmp", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(records_path + ".tmp", records_path)
        with open(os.path.join(BUNDLE_DIR, "detections_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "exported": datetime.date.today().isoformat(),
                "checkpoint": os.path.basename(args.checkpoint),
                "checkpoint_fingerprint": fingerprint,
                "tta": args.tta,
                "peak_floor": args.peak_floor,
                "peak_min_distance": PEAK_MIN_DISTANCE,
                "n_detections": n_dets,
            }, f, indent=2)
        print(f"Wrote detections for {len(records)} panos to {records_path}")
    else:
        print("--limit smoke run: records.jsonl NOT rewritten")

    gts = load_yolo_ground_truths(LABELS_DIR)
    r = gate_report(records, gts, GATE_THRESHOLD)
    partial = f" (PARTIAL, {len(records)} panos — not comparable)" if args.limit else ""
    print(f"\nReproduction gate @ conf >= {GATE_THRESHOLD}{partial}:")
    print(f"  this export : P {r.precision:.3f} / R {r.recall:.3f} "
          f"(tp {r.tp} / fp {r.fp} / fn {r.fn})")
    print(f"  published   : P {PUBLISHED_P:.3f} / R {PUBLISHED_R:.3f} "
          "(README corrected row / HF model card, TTA)")
    if not args.limit and (abs(r.precision - PUBLISHED_P) > 0.005
                           or abs(r.recall - PUBLISHED_R) > 0.005):
        print("  MISMATCH > 0.005: check imagery source (re-encode drift — see "
              "fetch_manual_gold.py), checkpoint, and TTA before running challengers.")
    full = gate_report(records, gts, 0.0)
    ap_str = f" AP {full.ap:.3f} (floor {args.peak_floor})" if full.ap is not None else ""
    print(f"  full range  : P {full.precision:.3f} / R {full.recall:.3f}{ap_str}")


if __name__ == "__main__":
    main()
