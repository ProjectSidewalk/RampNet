"""Is a model's recall real detection, or just density?

A detection counts as a hit when it lands within the match radius (0.022
normalized, ~22 px) of a ground-truth ramp. That is a fair rule for a model
emitting ~2 boxes per pano. It is not fair for one emitting ~74: scatter enough
boxes and some fall within 22 px of each real ramp BY COINCIDENCE, with no
detection involved. So "recall" is not the same measurement down the table, and
any recall-ceiling or union-oracle claim built on it needs discounting first.

The question this answers: how much recall would this model get if it were not
detecting anything, but still emitting boxes the way it actually does?

THE NULL MODEL -- score pano A's ground truth against pano B's PREDICTIONS.

Both sides are real outputs of the real model on real imagery, so the box count,
the spatial distribution and any systematic clustering (hood, nadir, the
sidewalk band) are preserved exactly. This matters: uniform-random boxes would
understate the effect, because real detectors cluster where ramps also tend to
be. But pano B's boxes have nothing to do with pano A's ramps, so every match is
coincidence -- and whatever recall survives is what the radius gives away for
free at that density.

Averaged over all non-identity cyclic shifts of the pano order: deterministic
(no seed), and every pano's GT gets a turn against every other pano's
predictions. `null max` is the worst single shift; it sits close to the mean,
which is how we know the null is a property of the density and not of one
unlucky pairing.

Worked example, richmond (310 GT ramps):

    model    boxes/pano  matched      by coincidence   attributable
    rampnet     2.2      238 (0.768)   ~17 (0.055)        ~221
    owlv2      74.3      301 (0.971)  ~227 (0.733)         ~74

OWLv2 matches 63 more ramps than RampNet -- and is ~147 ramps BEHIND once the
free matches come off both sides.

"Above chance" rescales what is left:

    above chance = (recall - null) / (1 - null)

Once the null is 0.733 a model cannot score above 1.0, so only 0.267 of headroom
remains; this asks how much of that available headroom the model took. For OWLv2
on richmond, 0.238 / 0.267 = 0.891. (Same shape as Cohen's kappa: observed minus
expected, over perfect minus expected.)

Read it WITH the raw gap, never instead of it. When the null is high "above
chance" flatters -- OWLv2's 0.891 is 89% of a small remaining slice, while
RampNet's 0.754 is 75% of nearly the whole range.

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

from rampnet import roster  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    aggregate, radius_sq_for, score_pano, PANO_RADIUS_NORMALIZED,
)
import compare  # noqa: E402
from detectors import build_detector, parse_model_spec  # noqa: E402

# The full roster the city splits are scored on, from rampnet/roster.py rather than
# a fourth private copy of it.
DEFAULT_MODELS = ",".join(roster.SCORED_SPECS)


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
    # rampnet.roster.PROVIDER_DEFAULTS is where they are defined, once.
    _D = roster.PROVIDER_DEFAULTS
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective")
    ap.add_argument("--gemini-model", default=_D["gemini_model"])
    ap.add_argument("--claude-model", default=_D["claude_model"])
    ap.add_argument("--claude-effort", default=_D["claude_effort"],
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--claude-tool-choice", default=_D["claude_tool_choice"],
                    choices=["auto", "forced"])
    ap.add_argument("--qwen-model", default=_D["qwen_model"])
    ap.add_argument("--qwen-coord-space", choices=["auto", "norm1000", "pixels"],
                    default=_D["qwen_coord_space"])
    ap.add_argument("--owlv2-model", default=_D["owlv2_model"])
    ap.add_argument("--gdino-model", default=_D["gdino_model"])
    ap.add_argument("--molmo-model", default=_D["molmo_model"])
    ap.add_argument("--molmo-coord-scale", choices=["auto", "100", "1000"],
                    default=_D["molmo_coord_scale"])
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
