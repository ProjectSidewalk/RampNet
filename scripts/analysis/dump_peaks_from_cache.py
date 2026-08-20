"""Turn a completed `evaluate.py` heatmap cache into per-panorama detections (#135).

`stage_two/evaluate.py` writes aggregate PR curves and nothing per-panorama, so a
paired comparison between two checkpoints -- which needs to know *which* ramps each
one found -- cannot be built from its output. It also rebuilds every heatmap from the
panorama images through the model, which needs a GPU and the 465 GB dataset.

Neither is necessary once the cache exists. `evaluate.py` caches the combined heatmap
per panorama under `<cache-dir>/heatmaps/<fingerprint>_<dataset>_<tta>/<pano>_heatmap.npy`,
and peak extraction from there is CPU-only numpy. So this reads a cache that has
already been paid for and emits the detections, **with no model, no panorama images,
no GPU and no network**.

It is a separate script rather than a flag on `evaluate.py` deliberately: that
evaluator produced every committed Stage 2 number, its heatmap cache key is
`<fingerprint>_<dataset>_<tta>` and nothing else, and the cheapest way to guarantee
this analysis cannot perturb either is to not touch it. What it does share is the
part that must not diverge -- `extract_peaks_from_heatmap`, `PEAK_MIN_DISTANCE` and
`MODEL_HEATMAP_SIZE` are imported from it, not copied, so the detections below are
extracted by the same code that produced the committed curve.

    python scripts/analysis/dump_peaks_from_cache.py \
        --cache-dir /path/to/run_a_84/evaluate_cache \
        --out-dir benchmark/model_detections --verify

Output is one file per cached checkpoint in the published-detections shape
(`{pano_id: [[x, y, confidence], ...]}`), so `benchmark_power_135.py` and every other
reader of `benchmark/model_detections/` consumes it with no special case.

**`--threshold` truncates the confidence tail, and that is deliberate.** The Run A
scoring ran `evaluate.py --threshold 0.0`, which keeps every local maximum -- about
511,000 predictions over 1,000 panoramas, most of them noise floor. The default 0.05
here matches the floor `analysis_out/op_cache` uses, keeps the files ~200 KB instead
of ~40 MB, and is far below every quantity this is used for (the protocol point is
0.30; Run A's max-F1 lands between 0.268 and 0.582). It does mean **AP is not
recoverable from these files** -- AP integrates the whole curve, so it must be read
from `docs/data/run_a_84_manual_gold/`. `--verify` checks the parts that are
recoverable against that same committed table, so a truncation that ever did bite
would fail loudly rather than shift a number quietly.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_two"))

import evaluate as ev  # noqa: E402  (stage_two/evaluate.py -- the peak extractor)
from rampnet.detection_eval import (  # noqa: E402
    load_yolo_ground_truths, radius_sq_for, score_pano,
)

#: How a cache directory names itself: <fingerprint>_<dataset>_<tta|notta>.
DEFAULT_SUMMARY = REPO / "docs" / "data" / "run_a_84_manual_gold" / "summary.csv"
#: Serialization must match export_model_cache.DUMP_KW, or two runs that agree on
#: every value still produce different bytes.
DUMP_KW = {"separators": (",", ":"), "sort_keys": True}


def parse_cache_key(name):
    """('2c1a21a7f4ba', 'manual', False) from '2c1a21a7f4ba_manual_notta'."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3 or parts[2] not in ("tta", "notta"):
        return None
    return parts[0], parts[1], parts[2] == "tta"


def fingerprint_labels(summary_csv):
    """{fingerprint: 'run_a_epoch_N'} from Run A's committed summary table.

    The cache directories are named by checkpoint fingerprint, which is the right key
    and an unreadable label. The committed summary already carries the mapping, so
    the epoch numbers come from the repo rather than from a filename someone typed.
    """
    labels = {}
    if not os.path.exists(summary_csv):
        return labels
    with open(summary_csv, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split(",")))
            fp = row.get("checkpoint_fingerprint")
            if fp:
                labels[fp] = f"run_a_epoch_{int(row['epoch'])}"
    return labels


def peaks_for_cache_dir(cache_dir, threshold):
    """{pano_id: [[x, y, confidence], ...]} for one cached checkpoint."""
    out = {}
    names = sorted(n for n in os.listdir(cache_dir) if n.endswith("_heatmap.npy"))
    for name in names:
        pano_id = name[: -len("_heatmap.npy")]
        heatmap = np.load(os.path.join(cache_dir, name))
        peaks = ev.extract_peaks_from_heatmap(
            heatmap,
            min_distance=ev.PEAK_MIN_DISTANCE,
            threshold_abs=threshold,
            heatmap_shape=ev.MODEL_HEATMAP_SIZE,
        )
        out[pano_id] = [[float(x), float(y), float(c)] for x, y, c in peaks]
    return out


def score_against_manual(detections, labels_dir, threshold):
    """(f1_at_threshold, max_f1) against the manual gold labels.

    Uses the shared model-agnostic scorer, so these numbers are directly comparable
    with `docs/data/run_a_84_manual_gold/summary.csv` and with every challenger.
    """
    gts = load_yolo_ground_truths(labels_dir)
    rsq = radius_sq_for()
    details, n_gt = [], 0
    for pano_id, gt in gts.items():
        pts = detections.get(pano_id, [])
        n_gt += len(gt.gt_points)
        details.extend(score_pano(pts, gt, radius_sq=rsq).details)

    details.sort(key=lambda d: d[0], reverse=True)
    best, at_threshold = 0.0, 0.0
    tp = 0
    for i, (conf, is_tp) in enumerate(details, start=1):
        tp += bool(is_tp)
        precision, recall = tp / i, tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Only read F1 where a threshold could actually be set -- i.e. at the last
        # detection of a block of equal confidences.
        if i == len(details) or details[i][0] < conf:
            best = max(best, f1)
            if conf >= threshold:
                at_threshold = f1
    return at_threshold, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", required=True,
                    help="An evaluate.py --cache-dir (the parent of heatmaps/).")
    # NOT benchmark/model_detections/: rampnet/roster.py asserts every file there
    # belongs to a registered challenger leg (#122), and Run A's epochs are internal
    # checkpoints of one experiment rather than entries in the RampNet-vs-VLM
    # comparison. They belong beside the rest of the #84 data.
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "data" / "run_a_84_detections"))
    ap.add_argument("--city", default="manual_gold",
                    help="Split name the detections belong to, for the filename.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Peak floor. Truncates the noise tail; AP is not recoverable "
                         "below it. See the module docstring.")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=False,
                    help="Which cache arm to read (default: the single-pass one, which "
                         "is the #54 protocol and what Run A's curve was read on).")
    ap.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY),
                    help="Committed table mapping checkpoint fingerprint -> epoch.")
    ap.add_argument("--manual-labels", default=str(REPO / "manual_labels"))
    ap.add_argument("--verify", action="store_true",
                    help="Re-score each dump against manual_labels and check it "
                         "reproduces the committed summary.csv. Do not skip this: it "
                         "is what proves the dump is the same instrument as the curve.")
    ap.add_argument("--tolerance", type=float, default=5e-4)
    args = ap.parse_args(argv)

    heatmaps = os.path.join(args.cache_dir, "heatmaps")
    if not os.path.isdir(heatmaps):
        raise SystemExit(f"{heatmaps}: not an evaluate.py cache (no heatmaps/ inside)")

    labels = fingerprint_labels(args.summary_csv)
    committed = {}
    if os.path.exists(args.summary_csv):
        with open(args.summary_csv, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split(",")
            for line in fh:
                if line.strip():
                    row = dict(zip(header, line.rstrip("\n").split(",")))
                    committed[row["checkpoint_fingerprint"]] = row

    os.makedirs(args.out_dir, exist_ok=True)
    problems, written = [], []
    for name in sorted(os.listdir(heatmaps)):
        parsed = parse_cache_key(name)
        if parsed is None:
            continue
        fingerprint, dataset, is_tta = parsed
        if is_tta != args.tta:
            continue
        cache_dir = os.path.join(heatmaps, name)
        label = labels.get(fingerprint, f"ckpt_{fingerprint}")

        detections = peaks_for_cache_dir(cache_dir, args.threshold)
        n_dets = sum(len(v) for v in detections.values())
        payload = {
            "model": label, "published_as": label, "city": args.city,
            "signature": {
                "source": "evaluate.py heatmap cache",
                "cache_key": name, "checkpoint_fingerprint": fingerprint,
                "dataset": dataset, "tta": is_tta,
                "peak_floor": args.threshold,
                "peak_min_distance": ev.PEAK_MIN_DISTANCE,
                "heatmap_size": list(ev.MODEL_HEATMAP_SIZE),
                "exclude_border": False,
            },
            "n_panos": len(detections), "n_uncached": 0,
            "detections": detections,
        }
        path = os.path.join(args.out_dir, f"{label}__{args.city}.json")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            json.dump(payload, fh, **DUMP_KW)
        written.append((label, len(detections), n_dets, os.path.getsize(path)))

        if args.verify and fingerprint in committed:
            row = committed[fingerprint]
            f1, max_f1 = score_against_manual(detections, args.manual_labels, 0.30)
            d_f1 = abs(f1 - float(row["f1_at_protocol"]))
            d_max = abs(max_f1 - float(row["max_f1"]))
            status = "ok" if max(d_f1, d_max) <= args.tolerance else "MISMATCH"
            if status != "ok":
                problems.append(
                    f"{label}: F1@0.30 {f1:.6f} vs committed {row['f1_at_protocol']} "
                    f"(d={d_f1:.6f}); max-F1 {max_f1:.6f} vs {row['max_f1']} (d={d_max:.6f})")
            print(f"  verify {label:>16}  F1@0.30 {f1:.6f} (d {d_f1:.2e})  "
                  f"max-F1 {max_f1:.6f} (d {d_max:.2e})  {status}")

    print(f"\nwrote {len(written)} file(s) -> {args.out_dir}")
    print(f"{'label':>18} {'panos':>6} {'detections':>11} {'KB':>8}")
    for label, n_panos, n_dets, size in written:
        print(f"{label:>18} {n_panos:>6} {n_dets:>11} {size / 1024:>8.0f}")

    if problems:
        print("\nVERIFICATION FAILED -- the dump is not the instrument that produced "
              "the committed curve, so nothing derived from it can be trusted:")
        for p in problems:
            print(f"  {p}")
        return 1
    if args.verify:
        print("\nEvery dump reproduces the committed summary.csv within tolerance.")
    elif written:
        print("\nNOTE: run again with --verify before using these for anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
