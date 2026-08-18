"""How much of each model's false-positive flood is real? (#46, the FP half)

``docs/model_comparison.md`` reports the challengers as FP-heavy — 119-293 false
positives against RampNet's 9 on richmond, and ~8,800 for OWLv2. #46's question is
what that number is *made of*, because several artifacts inflate it without the
model being as wrong as the count implies:

* **duplicate** — a second box on a ramp another prediction already claimed. The
  matcher charges it as a false positive (correctly — it is a redundant detection),
  but it is not a hallucinated ramp. Box-emitting models that split one ramp across
  two views (#43) pay here.
* **near_gt** — lands in ``[R, 2R)`` of a real ramp: right object, loose box. #46's
  "box->center + tight radius double-penalizes" confound — scored as an FP *and*
  the ramp as an FN, one error counted twice.
* **hood** — in the ego-vehicle / nadir band. The panoramas are shot from a car and
  the bottom of every one is vehicle, so a detection there cannot be a curb ramp.
  #46 predicted this for Grounding DINO specifically ("GDINO's top box is the
  hood"), and the y-histograms bear it out: OWLv2 and GDINO both carry a distinct
  second mode at y 0.70-0.90 that the chat VLMs do not.
* **isolated** — everything else. Not near a ramp, not on the car: the closest this
  can get to "genuine hallucination" without opening the imagery.

**The null matters more here than anywhere else in the benchmark.** OWLv2 emits
55-88 boxes per panorama; at that density, boxes land near real ramps by accident,
and ``docs/model_comparison.md``'s null-recall correction already had to discount
open-detector *recall* for exactly this reason. So ``duplicate`` and ``near_gt`` are
reported against an exact chance baseline: for a prediction at height ``y``, the
set of azimuths landing within a radius of some ground-truth ramp is a union of
arcs, and :func:`arc_union_fraction` measures it in closed form rather than by
sampling. A model whose observed near-GT rate merely matches its null is not
"nearly right" — it is dense.

Reads ``.model_cache`` (per-pano detections, keyed by model + rig + pano) and the
committed benchmark bundles. **No GPU and no model load**: ``compare.score_model``
skips ``prepare()`` when every pano is cached, and this script never calls the
detector at all — it reconstructs each detector's cache signature and reads. It
does need ``benchmark/<city>/panos`` only for the gallery, which lives elsewhere.

    python scripts/analysis/fp_taxonomy.py
    python scripts/analysis/fp_taxonomy.py --models owlv2,gdino --cities richmond

**What this does NOT do.** It cannot tell a driveway from a crosswalk from a set of
stairs — that is an appearance judgment needing the imagery, and it is why
``isolated`` is an upper bound on hallucination rather than a measurement of it.
#46's gallery half is what would split it.
"""
import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet import roster  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, prediction_confidence, radius_sq_for)
from rampnet.metrics import greedy_match  # noqa: E402

from miss_decomposition import HELD_OUT, TIER, US_SPLITS  # noqa: E402

# The roster now lives in rampnet/roster.py — one table that also carries each
# model's density class, the date it joined, and whether it is scored in the roster
# tables. Re-exported here because three analysis scripts already import it from this
# module. RampNet is read from the committed records.jsonl (it has no cache
# signature); the challengers come from .model_cache.
CHALLENGERS = roster.CHALLENGERS

# Where the ego vehicle starts. Set from the data, not assumed: across the seven
# pooled splits the 99.5th percentile of ground-truth ramp height is y=0.725 and the
# maximum is 0.756, while flat-ground geometry puts y=0.75 at 2.5 m — inside the
# car. Everything below this line is hood, roof-rack or nadir cap.
HOOD_Y = 0.75

# How far outside the match radius still counts as "the right object, loosely
# boxed". Same 2R the miss taxonomy uses for its localization bucket, so the FN and
# FP sides of a single loose box are attributed consistently.
ANNULUS_FACTOR = 2.0

BUCKETS = ("duplicate", "near_gt", "hood", "isolated")


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_fp_taxonomy.py
# --------------------------------------------------------------------------- #
def scaled(point):
    return float(point[0]) * PANO_SCALE_X, float(point[1]) * PANO_SCALE_Y


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def arc_union_fraction(py, gt_points, radius_sq):
    """Chance a uniformly random azimuth at height ``py`` lands within radius of a ramp.

    Exact, not sampled. A ramp at ``(gx, gy)`` is reachable from height ``py`` only
    if ``|py - gy| < radius``; the azimuths that reach it form an arc of half-width
    ``sqrt(radius^2 - (py-gy)^2)`` centred on ``gx``. The answer is the measure of
    the union of those arcs on a circle of circumference ``PANO_SCALE_X``.

    This is the density control for ``near_gt`` and ``duplicate``. A closed form
    beats sampling here because OWLv2 contributes ~65,000 false positives and a
    200-trial null per prediction would be 13 million distance loops for a number
    that is analytic.
    """
    if not gt_points:
        return 0.0
    spans = []
    for g in gt_points:
        gx, gy = scaled(g)
        rem = radius_sq - (py - gy) ** 2
        if rem <= 0:
            continue
        half = math.sqrt(rem)
        lo, hi = gx - half, gx + half
        # Split anything crossing the seam so the union runs on a plain interval line.
        if lo < 0:
            spans.append((lo + PANO_SCALE_X, PANO_SCALE_X))
            spans.append((0.0, hi))
        elif hi > PANO_SCALE_X:
            spans.append((lo, PANO_SCALE_X))
            spans.append((0.0, hi - PANO_SCALE_X))
        else:
            spans.append((lo, hi))
    if not spans:
        return 0.0
    spans.sort()
    total, cur_lo, cur_hi = 0.0, spans[0][0], spans[0][1]
    for lo, hi in spans[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    total += cur_hi - cur_lo
    return min(1.0, total / PANO_SCALE_X)


def classify_fp(point, gt_points, claimed, radius_sq, hood_y=HOOD_Y,
                annulus_factor=ANNULUS_FACTOR):
    """Which bucket one false positive falls in. See the module docstring.

    ``claimed`` is the set of GT indices some prediction already matched, which is
    what makes ``duplicate`` mean "a second hit on a ramp we already found" rather
    than "near a ramp". A prediction inside the radius of an *unclaimed* ramp cannot
    occur — the matcher would have assigned it — so the in-radius case is
    unambiguous.
    """
    ps = scaled(point)
    inner, outer = radius_sq, radius_sq * annulus_factor * annulus_factor
    for i, g in enumerate(gt_points):
        if i in claimed and _d2(ps, scaled(g)) < inner:
            return "duplicate"
    if any(_d2(ps, scaled(g)) < outer for g in gt_points):
        return "near_gt"
    if float(point[1]) >= hood_y:
        return "hood"
    return "isolated"


def summarize_fp(rows):
    """Bucket counts and shares for one model's false positives."""
    n = len(rows) or 1
    counts = {b: sum(1 for r in rows if r["bucket"] == b) for b in BUCKETS}
    return {"n_fp": len(rows), "counts": counts,
            "shares": {b: counts[b] / n for b in BUCKETS},
            # Expected duplicate+near_gt under chance placement, summed exactly.
            "null_near_gt": sum(r["null_2r"] for r in rows),
            "null_duplicate": sum(r["null_1r"] for r in rows)}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _compare_args(cache_dir):
    """A namespace matching ``compare.py``'s parser defaults.

    Every field here feeds ``build_detector`` and therefore the cache signature. The
    per-provider model ids come from ``rampnet.roster.PROVIDER_DEFAULTS`` so there is
    one definition rather than a copy per parser; the rest are compare.py's own
    non-provider defaults, which still have to be restated because ``compare.main``
    builds its parser inline. ``test_fp_taxonomy.py`` asserts the whole namespace
    still matches that parser, so a drift in compare.py fails a test instead of
    silently missing every cache entry and reporting zero detections.
    """
    import argparse as _a
    ns = _a.Namespace()
    for k, v in dict(
            roster.PROVIDER_DEFAULTS,
            owlv2_query=None, gdino_query=None,
            gdino_text_threshold=None, score_threshold=None,
            yolo_model=None, tiling="perspective",
            radius=0.022, op_threshold=0.0, limit=None,
            cache_dir=cache_dir, no_cache=False).items():
        setattr(ns, k, v)
    return ns


def _bucket_pano(preds, gt, city, pid, radius_sq, hood_y):
    """Bucket one pano's false positives against its ground truth."""
    confs = [prediction_confidence(p) for p in preds]
    if any(c is not None for c in confs):
        order = sorted(range(len(preds)),
                       key=lambda i: (confs[i] if confs[i] is not None
                                      else float("-inf")), reverse=True)
        preds = [preds[i] for i in order]
    assign = greedy_match([(float(p[0]), float(p[1])) for p in preds],
                          gt.gt_points, radius_sq, PANO_SCALE_X, PANO_SCALE_Y)
    claimed = {gi for gi, _ in assign if gi >= 0}
    rows = []
    for p, (gi, _) in zip(preds, assign):
        if gi >= 0:
            continue
        ps = scaled(p)
        # An ignored prediction (inside an 'unsure' mark) is neither TP nor FP in
        # the scorer, so it must not enter the taxonomy either.
        if any(_d2(ps, scaled(q)) < radius_sq for q in gt.ignore_points):
            continue
        rows.append({
            "city": city, "pano": pid, "x": float(p[0]), "y": float(p[1]),
            # Kept so a gallery can rank "worst cases" by the model's own confidence.
            # None for the chat VLMs, which emit no calibrated score.
            "confidence": prediction_confidence(p),
            "bucket": classify_fp(p, gt.gt_points, claimed, radius_sq, hood_y),
            "null_1r": arc_union_fraction(ps[1], gt.gt_points, radius_sq),
            "null_2r": arc_union_fraction(
                ps[1], gt.gt_points, radius_sq * ANNULUS_FACTOR ** 2),
        })
    return rows, len(claimed)


def rampnet_rows(city, radius_sq, hood_y, threshold, cache_dir=None):
    """RampNet's own false positives, from the committed low-floor cache.

    RampNet carries no detector signature (``compare.py`` reads its detections
    straight out of ``records.jsonl``), so it cannot be looked up in
    ``.model_cache`` like the challengers. Reading ``analysis_out/op_cache`` instead
    is the better source anyway: it is thresholded at the **deployed 0.30**
    recommendation (#79) rather than the 0.55 floor the city bundles were exported
    at, so this row and ``miss_taxonomy.py``'s FN buckets describe one operating
    point.
    """
    from miss_decomposition import CACHE_DIR
    from operating_point_curve import read_cache
    path = os.path.join(cache_dir or CACHE_DIR, f"{city}.json")
    if not os.path.exists(path):
        return None
    panos, _meta = read_cache(path)
    rows, tp = [], 0
    for pd in panos:
        preds = [p for p in pd["preds"] if p[2] >= threshold]
        pano_rows, claimed = _bucket_pano(preds, pd["gt"], city, pd["pano"],
                                          radius_sq, hood_y)
        rows.extend(pano_rows)
        tp += claimed
    return "rampnet @0.30", rows, {"tp": tp, "uncached": 0, "n_panos": len(panos)}


def model_rows(city, spec, cache, args, radius_sq, hood_y=HOOD_Y):
    """Every false positive one model made in one city, bucketed.

    Returns ``(label, rows, stats)`` or ``None`` when the model has no cache for
    this city — reported rather than skipped silently, because a missing model is
    indistinguishable from a model that made no mistakes.
    """
    import compare as C
    from export_model_cache import load_detections, spec_label

    bundle = os.path.join(REPO, "benchmark", city)
    records, verdicts, _panos = C.load_bundle(bundle)
    if verdicts is None:
        gts = C.load_manual_ground_truths(bundle)
    else:
        gts = C.ground_truths_from_verdicts(records, verdicts)

    # Prefer the PUBLISHED detections (benchmark/model_detections/) over the local
    # working cache: they are committed, so this path works from a clean clone with
    # no .model_cache and without importing the detector stack at all. The cache is
    # the fallback for runs that are still producing detections.
    label = spec_label(spec, args)
    published = load_detections(label, city)
    if published is None:
        from detectors import build_detector, parse_model_spec
        provider, model_id = parse_model_spec(spec)
        label, det = build_detector(provider, model_id, records, args)
        sig = det.signature() if hasattr(det, "signature") else None
        if sig is None:
            return None
        published = {pid: cache.get(C.cache_key(label, sig, city, pid)) for pid in gts}
        published = {k: v for k, v in published.items() if v is not None}
    rows, tp, uncached = [], 0, 0
    for pid, gt in gts.items():
        pts = published.get(pid)
        if pts is None:
            uncached += 1
            continue
        pano_rows, claimed = _bucket_pano(list(pts), gt, city, pid, radius_sq, hood_y)
        rows.extend(pano_rows)
        tp += claimed
    return label, rows, {"tp": tp, "uncached": uncached, "n_panos": len(gts)}


def gt_height_stats(cities):
    """Where ground-truth ramps actually sit in y, pooled over ``cities``.

    Printed with the results so ``HOOD_Y`` is checkable rather than asserted: the
    band is only safe to write off if essentially no real ramp lives in it.
    """
    import compare as C
    ys = []
    for city in cities:
        bundle = os.path.join(REPO, "benchmark", city)
        records, verdicts, _ = C.load_bundle(bundle)
        gts = (C.load_manual_ground_truths(bundle) if verdicts is None
               else C.ground_truths_from_verdicts(records, verdicts))
        for gt in gts.values():
            ys.extend(y for _, y in gt.gt_points)
    ys.sort()
    if not ys:
        return None
    return {"n": len(ys), "p99_5": ys[int(0.995 * len(ys))], "max": ys[-1],
            "n_below_hood": sum(1 for y in ys if y >= HOOD_Y)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default=",".join(roster.SCORED_SPECS),
                   help="Comma-separated compare.py model specs. 'rampnet' is read "
                        "from analysis_out/op_cache at the deployed threshold, not "
                        "from .model_cache (it carries no detector signature).")
    p.add_argument("--rampnet-threshold", type=float, default=0.30,
                   help="Operating point for the rampnet row (default 0.30, #79).")
    p.add_argument("--cities", default=",".join(US_SPLITS))
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--hood-y", type=float, default=HOOD_Y)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    import compare as C
    cache = C.DetectionCache(args.cache_dir, enabled=True)
    cargs = _compare_args(args.cache_dir)
    radius_sq = radius_sq_for()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    specs = [s.strip() for s in args.models.split(",") if s.strip()]

    print(f"=== False-positive taxonomy (#46) — {len(specs)} models x "
          f"{len(cities)} splits ===")
    print(f"hood boundary y >= {args.hood_y} ({'~2.5 m, inside the vehicle'}); "
          f"annulus {ANNULUS_FACTOR:.0f}R\n")
    print(f"{'model':>42} {'FP':>7} " +
          " ".join(f"{b:>15}" for b in BUCKETS) + f" {'null(near)':>11}")

    per_model, missing = {}, []
    for spec in specs:
        rows = []
        stats = {"tp": 0, "uncached": 0, "n_panos": 0}
        label = spec
        for city in cities:
            if spec == "rampnet":
                got = rampnet_rows(city, radius_sq, args.hood_y,
                                   args.rampnet_threshold)
            else:
                got = model_rows(city, spec, cache, cargs, radius_sq, args.hood_y)
            if got is None:
                continue
            label, city_rows, st = got
            rows.extend(city_rows)
            for k in stats:
                stats[k] += st[k]
        if stats["uncached"]:
            missing.append((label, stats["uncached"], stats["n_panos"]))
        s = summarize_fp(rows)
        s["stats"] = stats
        s["per_city"] = {c: summarize_fp([r for r in rows if r["city"] == c])
                         for c in cities}
        per_model[label] = s
        null_share = s["null_near_gt"] / (s["n_fp"] or 1)
        print(f"{label:>42} {s['n_fp']:>7} " +
              " ".join(f"{s['counts'][b]:>7} {s['shares'][b]:>6.1%}" for b in BUCKETS)
              + f" {null_share:>10.1%}")

    print(f"\n{'-'*100}")
    print("READING THE TABLE")
    print(f"{'-'*100}")
    print("  duplicate + near_gt = the measurement artifact #46 asked to size: a real")
    print("    ramp found and then charged as an error, or found with a loose box.")
    print("  null(near) = share of these predictions that would sit within 2R of a ramp")
    print("    by CHANCE, given this model's density and the ramps' layout. Subtract it")
    print("    before crediting a model with 'nearly right' boxes.")
    print("  hood = below the ego-vehicle line, where no curb ramp can be.")
    print("  isolated = the residue, and an UPPER BOUND on hallucination: a driveway,")
    print("    a crosswalk and a flight of stairs all land here and only imagery splits")
    print("    them (#46's gallery half, not done here).")

    print(f"\n{'-'*100}")
    print("ARTIFACT SHARE, NULL-CORRECTED")
    print(f"{'-'*100}")
    print(f"{'model':>42} {'artifact':>10} {'expected by chance':>20} {'excess':>10}")
    for label, s in per_model.items():
        art = s["counts"]["duplicate"] + s["counts"]["near_gt"]
        exp = s["null_near_gt"]
        n = s["n_fp"] or 1
        print(f"{label:>42} {art/n:>9.1%} {exp/n:>19.1%} {(art-exp)/n:>9.1%}")

    gh = gt_height_stats(cities)
    if gh:
        print(f"\n{'-'*100}")
        print(f"HOOD LINE, CHECKED AGAINST THE RAMPS ({gh['n']} pooled GT points)")
        print(f"{'-'*100}")
        print(f"  99.5th percentile of GT height  y = {gh['p99_5']:.3f}")
        print(f"  highest GT ramp                 y = {gh['max']:.3f}")
        print(f"  GT ramps at or below the line   {gh['n_below_hood']} "
              f"({gh['n_below_hood']/gh['n']:.2%}) — these are misattributed as hood")
        print(f"    only if nothing else claims them first; proximity to a ramp outranks")
        print(f"    the hood test precisely so a genuine close-range ramp is not lost.")

    for label, unc, tot in missing:
        print(f"\nNOTE: {label} had {unc}/{tot} panos with no cache entry — "
              f"those panos are excluded, not counted as zero detections.")
    if not missing:
        print(f"\nEvery model resolved every pano from cache (no GPU, no model load).")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"hood_y": args.hood_y, "annulus_factor": ANNULUS_FACTOR,
                       "cities": cities,
                       # Which models this ran over, so a reader can tell a roster
                       # change from a numbers change -- the same reasoning as the
                       # detector signature inside each published detections file.
                       "models": roster.pool_record(specs, cargs),
                       "per_model": per_model}, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
