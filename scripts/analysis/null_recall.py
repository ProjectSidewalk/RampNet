"""Is a model's recall real detection, or just density?

A model that carpets the pano earns recall for free. At a fixed match radius,
enough scattered boxes land within radius of most GT points whether or not the
model saw anything -- and the open-vocabulary detectors emit ~70-80 boxes per
pano against RampNet's ~2. So the recall column is not measuring the same thing
down the table, and any "recall ceiling" or union-oracle claim built on it has
to be discounted before it means anything.

The null model: score each pano's ground truth against ANOTHER pano's
predictions. That preserves the detector's exact detection count and spatial
distribution -- including systematic clustering like the hood/nadir boxes --
and destroys every true correspondence, so whatever recall survives is what the
radius hands out for free at that density. Averaged over all non-identity
cyclic shifts of the pano order, which is deterministic (no seed) and gives
every pano's GT a turn against every other pano's predictions.

"Above chance" is headroom-normalized: (recall - null) / (1 - null), i.e. of
the recall a perfect detector could add over the null, how much this model
captured. It is a *generous* framing when the null is high -- read it alongside
the raw gap, not instead of it.

Reads cached detections from .model_cache, so it needs no GPU, no API key and
no model load. Models that aren't fully cached are reported and skipped rather
than run, so this is safe to point at a split whose challenger runs haven't
happened yet -- it will tell you exactly what is missing.

    python scripts/analysis/null_recall.py benchmark/annapolis
    python scripts/analysis/null_recall.py benchmark/richmond \
        --models rampnet,gemini:gemini-3.6-flash,owlv2,gdino

Results are in the "How much of a detector's recall is real?" section of
docs/model_comparison.md.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "model_comparison"))

from rampnet.detection_eval import (  # noqa: E402
    aggregate, radius_sq_for, score_pano, PANO_RADIUS_NORMALIZED,
)
import compare  # noqa: E402
from detectors import build_detector, parse_model_spec  # noqa: E402

# The full roster the city splits are scored on, in the results-table order.
DEFAULT_MODELS = ("rampnet,gemini:gemini-3.1-pro-preview,gemini:gemini-3.6-flash,"
                  "molmo:allenai/Molmo2-8B,qwen:Qwen/Qwen3-VL-32B-Instruct,"
                  "qwen:Qwen/Qwen3-VL-8B-Instruct,owlv2,gdino")


def null_recall(scored, radius_sq):
    """Mean recall over all non-identity cyclic shifts of the pano order.

    Returns (mean, max). The max matters: a high mean could in principle come
    from one pathological pairing, and it doesn't -- the spread is tight.
    """
    preds = [p for p, _ in scored]
    gts = [g for _, g in scored]
    n = len(scored)
    shifted = [
        aggregate([score_pano(preds[(i + k) % n], gts[i], radius_sq=radius_sq)
                   for i in range(n)]).recall
        for k in range(1, n)
    ]
    return sum(shifted) / len(shifted), max(shifted)


def cache_coverage(detector, label, city, gts, cache):
    """(n_cached, n_total) for one model, without instantiating anything heavy."""
    sig = detector.signature() if hasattr(detector, "signature") else None
    if sig is None:      # rampnet reads the bundle records; nothing to cache
        return len(gts), len(gts)
    hits = sum(1 for pid in gts
               if cache.get(compare.cache_key(label, sig, city, pid)) is not None)
    return hits, len(gts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", help="Bundle dir (e.g. benchmark/annapolis).")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="Comma-separated detectors, same syntax as compare.py.")
    ap.add_argument("--radius", type=float, default=PANO_RADIUS_NORMALIZED,
                    help=f"Normalized match radius (default {PANO_RADIUS_NORMALIZED}).")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / ".model_cache"))
    ap.add_argument("--allow-uncached", action="store_true",
                    help="Run detectors whose cache is incomplete. Off by default: "
                         "this is an analysis pass over runs that already happened, "
                         "and a stray GPU/API call here is never what you wanted.")
    # build_detector reads these off the namespace; the defaults must match
    # compare.py's or the detector signature -- and so the cache key -- changes.
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective")
    ap.add_argument("--gemini-model", default="gemini-3.6-flash")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--qwen-coord-space", choices=["auto", "norm1000", "pixels"], default="auto")
    ap.add_argument("--owlv2-model", default="google/owlv2-large-patch14-ensemble")
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--molmo-model", default="allenai/Molmo2-8B")
    ap.add_argument("--molmo-coord-scale", choices=["auto", "100", "1000"], default="auto")
    args = ap.parse_args()

    records, verdicts, panos_dir = compare.load_bundle(args.bundle)
    if verdicts is not None:
        compare.validate_bundle(records, verdicts)
        gts = compare.ground_truths_from_verdicts(records, verdicts)
    else:
        gts = compare.load_manual_ground_truths(args.bundle)
    radius_sq = radius_sq_for(args.radius)
    city = os.path.basename(os.path.normpath(args.bundle))
    cache = compare.DetectionCache(args.cache_dir, enabled=True)

    print(f"Bundle: {args.bundle}  ({len(gts)} scored panos)  match radius {args.radius}")
    print(f"Null model: {len(gts) - 1} cyclic shifts per model\n")

    header = (f"{'model':38s} {'preds/pano':>10s} {'recall':>8s} {'null':>8s} "
              f"{'null max':>9s} {'above chance':>13s}")
    print(header)
    print("-" * len(header))

    skipped = []
    specs = [parse_model_spec(t) for t in args.models.split(",") if t.strip()]
    for provider, model_id in specs:
        label, detector = build_detector(provider, model_id, records, args)
        hits, total = cache_coverage(detector, label, city, gts, cache)
        if hits < total and not args.allow_uncached:
            skipped.append((label, hits, total))
            continue
        run = compare.score_model(detector, records, gts, panos_dir, radius_sq,
                                  label, city, cache)
        real = run.report.recall
        mean_null, max_null = null_recall(run.scored, radius_sq)
        above = (real - mean_null) / (1 - mean_null) if mean_null < 1 else float("nan")
        per_pano = sum(len(p) for p, _ in run.scored) / len(run.scored)
        print(f"{label:38s} {per_pano:10.1f} {real:8.3f} {mean_null:8.3f} "
              f"{max_null:9.3f} {above:13.3f}")

    for label, hits, total in skipped:
        print(f"\n[skip] {label}: {hits}/{total} panos cached — run it through "
              f"compare.py first, or pass --allow-uncached to run it here.")


if __name__ == "__main__":
    main()
