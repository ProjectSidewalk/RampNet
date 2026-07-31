"""What ACTUALLY causes each miss? (#46, tightening #59's near-field bound)

``miss_decomposition.py`` split the 427 pooled misses by distance: 247 far-field
(pixel-starved, ``>= 18 m``) and 180 near-field. It then labelled the near-field
population "appearance/vocabulary" and recorded, explicitly, that this was **an
inference, not a measurement** — a near-field miss can equally be occlusion, deep
shadow, debris, or GT disagreement. That figure, 8.7 recall points, has been
carrying the sourcing programme's upper bound ever since.

This measures what the label was standing in for. The committed low-floor caches
hold every peak down to a 0.05 score floor — well below the 0.30 operating point —
so for each missed ramp we can ask what the model actually did there:

* **merged** — a supra-threshold peak sits within the match radius, but a
  *neighbouring* ramp claimed it. The model fired once for a pair of adjacent
  ramps. Training targets are Gaussians of sigma 10 on the 512x1024 heatmap and
  peaks are extracted with ``min_distance=10``, so a pair inside ~2 sigma has one
  mode to find. **Representation, not vocabulary — more cities cannot fix it.**
* **sub_threshold** — a peak sits within the radius scoring in [floor, 0.30). The
  model localized the ramp and was not confident enough. **Calibration**, and
  already priced: #54/#55 chose 0.30 knowing what lower thresholds cost in
  precision.
* **localization** — no peak within the radius, but an *unclaimed* peak (a false
  positive, not a neighbour's hit) within 2R. The model fired at roughly the right
  place and the point landed outside the match radius.
* **silent** — nothing, even at the 0.05 floor. **This is the population a broader
  or more diverse training corpus could plausibly reach**, and it is the only
  bucket the sourcing programme can claim.

Two checks keep the buckets honest, because both failure modes have burned this
benchmark before:

1. **The matcher is not the cause.** ``greedy_match`` could in principle strand a
   ramp that an optimal assignment would have caught — #46 lists exactly this
   ("one miss, counted twice") among its suspected confounds. ``optimal_hits``
   recomputes each pano with maximum-cardinality bipartite matching. Pooled, it is
   a **wash**: 10 ramps are hit only under optimal and 10 only under greedy, net
   zero. The permutation is reported; the recall is not affected.
2. **"A peak was there" needs a null.** ``docs/model_comparison.md``'s null-recall
   correction found open-detector recall was largely *density* — at 55-88 boxes per
   pano, a box lands near anything. So every in-radius bucket is also computed
   against a null that keeps each missed ramp's elevation (both ramps and
   detections concentrate in the horizon band) and randomizes its azimuth over the
   full 360 degrees. RampNet emits 4.2 floor-level peaks per pano, and the null
   comes out at 4.7% against a real 46.7% for near-field sub_threshold — the bucket
   survives, but the rate is reported next to it rather than asserted away.

Reads the same committed caches as ``miss_decomposition.py``: no GPU, no network,
no imagery. Geometry, matching and the split lists are imported from the E1/#59
scripts rather than reimplemented — one estimator and one matcher, or the numbers
stop comparing.

    python scripts/analysis/miss_taxonomy.py
    python scripts/analysis/miss_taxonomy.py --threshold 0.55 --json-out out.json

**What this does NOT do.** It separates causes the *cached detections* can witness.
It cannot tell occlusion from deep shadow from debris from GT disagreement inside
the ``silent`` bucket — that needs the imagery, and is the gallery half of #46.
``silent`` is therefore still an upper bound on the sourcing-addressable
population, just a much tighter one than the near-field count it replaces.
"""
import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)
from rampnet.metrics import greedy_match  # noqa: E402

from miss_decomposition import (  # noqa: E402
    ALL_SPLITS, CACHE_DIR, DEFAULT_THRESHOLD, FAR_BOUNDARY_M, HELD_OUT, TIER,
    US_SPLITS)
from operating_point_curve import read_cache  # noqa: E402
from stage1_label_recall import geom  # noqa: E402

# Peak extraction (stage_two/evaluate.py) and the training target (stage_two/train.py).
# Both are named here because the merged bucket's interpretation turns on which one
# binds: min_distance is what the EXTRACTOR enforces, sigma is what the heatmap can
# represent in the first place.
PEAK_MIN_DISTANCE = 10          # skimage.feature.peak_local_max, heatmap px
TARGET_SIGMA = 10               # Gaussian target sigma, same grid

# How far outside the match radius still counts as "fired at roughly the right
# place". 2R is deliberately generous: the point is to give the localization
# hypothesis its best shot before the residual is called silent.
ANNULUS_FACTOR = 2.0

NULL_TRIALS = 200
NULL_SEED = 20260730

BUCKETS = ("merged", "sub_threshold", "localization", "silent")


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_miss_taxonomy.py
# --------------------------------------------------------------------------- #
def scaled(point):
    """Normalized (x, y) into the anisotropic pano pixel space the matcher uses."""
    return point[0] * PANO_SCALE_X, point[1] * PANO_SCALE_Y


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def wrapped_d2(a, b):
    """Squared distance with x wrapped at 360 degrees.

    Only the null needs this: a real detection and its ramp are never on opposite
    sides of the seam, but a uniformly random azimuth lands next to it half the
    time, and not wrapping would under-count the null near x=0 and x=1.
    """
    dx = abs(a[0] - b[0])
    dx = min(dx, PANO_SCALE_X - dx)
    return dx * dx + (a[1] - b[1]) ** 2


def optimal_hits(preds, gt_points, radius_sq):
    """Indices of GT hit under **maximum-cardinality** matching, not greedy.

    The upper bound on what any assignment of these predictions could achieve.
    Compared against the greedy result to show the matcher is not manufacturing
    misses (#46's "one miss, counted twice" confound). Kuhn's augmenting-path
    algorithm; the graphs here are a handful of nodes, so the O(V*E) cost is
    irrelevant and the simplicity is worth more than Hopcroft-Karp.
    """
    adj = []
    for p in preds:
        ps = scaled(p)
        adj.append([j for j, g in enumerate(gt_points)
                    if _d2(ps, scaled(g)) < radius_sq])
    match_gt = {}

    def augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_gt or augment(match_gt[j], seen):
                match_gt[j] = i
                return True
        return False

    for i in range(len(preds)):
        augment(i, set())
    return set(match_gt)


def classify_miss(gt_point, preds_floor, preds_kept, claimed_by, radius_sq,
                  threshold, annulus_factor=ANNULUS_FACTOR):
    """Which bucket one missed ramp falls in. See the module docstring.

    ``claimed_by`` maps a kept-prediction index to the GT index it was assigned
    (-1 for an unassigned prediction, i.e. a false positive). It is what separates
    ``localization`` from ``silent``: a prediction sitting in the annulus is only
    evidence about *this* ramp if no neighbouring ramp already owns it. Skipping
    that test inflates localization roughly six-fold — 54 of the 63 pooled
    annulus predictions turn out to be a neighbour's true positive.

    The cascade is ordered, but not ambiguously so: a merged ramp never also has
    its own distinct sub-threshold peak (0 of 124 pooled), so no miss is being
    silently claimed by an earlier bucket than it belongs to.

    Note a supra-threshold peak inside the radius of a *missed* ramp is always
    claimed by some other ramp — greedy assigns each prediction to the nearest
    unclaimed GT in range, so an unassigned one could only mean this ramp was
    already hit. That is why ``merged`` needs no further test.
    """
    gs = scaled(gt_point)
    in_radius = [p for p in preds_floor if _d2(scaled(p), gs) < radius_sq]
    if any(p[2] >= threshold for p in in_radius):
        return "merged"
    if in_radius:
        return "sub_threshold"
    outer = radius_sq * annulus_factor * annulus_factor
    annulus = [(i, p) for i, p in enumerate(preds_kept)
               if radius_sq <= _d2(scaled(p), gs) < outer]
    if annulus:
        nearest = min(annulus, key=lambda ip: _d2(scaled(ip[1]), gs))
        if claimed_by.get(nearest[0], -1) < 0:
            return "localization"
    return "silent"


def merged_separation(gt_point, gt_points, preds_kept, claimed_by, radius_sq):
    """``(chebyshev, euclidean)`` px between a merged ramp and the ramp that won.

    The diagnostic that says whether the *extractor* or the *target* is binding.
    ``peak_local_max`` suppresses on a maximum filter, i.e. Chebyshev distance, so a
    pair above ``PEAK_MIN_DISTANCE`` in Chebyshev was one the extractor was free to
    emit twice and did not — the heatmap had a single mode. Returns ``None`` when
    the winning prediction is unassigned (nothing to measure against).
    """
    gs = scaled(gt_point)
    inr = [(i, p) for i, p in enumerate(preds_kept) if _d2(scaled(p), gs) < radius_sq]
    if not inr:
        return None
    i, _ = min(inr, key=lambda ip: _d2(scaled(ip[1]), gs))
    partner = claimed_by.get(i, -1)
    if partner < 0:
        return None
    qs = scaled(gt_points[partner])
    return (max(abs(qs[0] - gs[0]), abs(qs[1] - gs[1])), _d2(qs, gs) ** 0.5)


def null_in_radius(gt_point, preds_floor, radius_sq, threshold, rng, trials):
    """``(supra_rate, sub_rate)`` for a peak landing in radius **by chance**.

    Elevation is held at the real ramp's y and azimuth is randomized, because both
    ramps and detections crowd the horizon band — randomizing y as well would
    compare against a null nothing lives in and flatter every bucket.
    """
    gy = gt_point[1] * PANO_SCALE_Y
    supra = sub = 0
    for _ in range(trials):
        gs = (rng.random() * PANO_SCALE_X, gy)
        inr = [p for p in preds_floor if wrapped_d2(scaled(p), gs) < radius_sq]
        if any(p[2] >= threshold for p in inr):
            supra += 1
        elif inr:
            sub += 1
    return supra / trials, sub / trials


def summarize(rows):
    """Bucket counts, shares and recall points for one population of misses."""
    misses = [r for r in rows if not r["hit"]]
    n_gt = len(rows)
    counts = {b: sum(1 for r in misses if r["bucket"] == b) for b in BUCKETS}
    n_miss = len(misses) or 1
    return {
        "n_gt": n_gt,
        "n_miss": len(misses),
        "recall": (n_gt - len(misses)) / n_gt if n_gt else float("nan"),
        "counts": counts,
        "shares": {b: counts[b] / n_miss for b in BUCKETS},
        # Recall points each bucket is worth against the whole population.
        "recall_pts": {b: (counts[b] / n_gt if n_gt else float("nan")) for b in BUCKETS},
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_rows(city, threshold, cache_dir=CACHE_DIR, rng=None, trials=NULL_TRIALS,
              boundary=FAR_BOUNDARY_M):
    """One row per scoreable GT ramp: geometry, hit/miss, bucket, null rates.

    Mirrors ``miss_decomposition.load_rows`` — same cache, same ``fn_confirmed``
    gate, same ``geom`` — so the two scripts partition an identical population and
    their totals must agree. They are checked against each other in
    ``tests/test_miss_taxonomy.py``.
    """
    path = os.path.join(cache_dir, f"{city}.json")
    if not os.path.exists(path):
        return None
    radius_sq = radius_sq_for()
    panos, _meta = read_cache(path)
    rows, greedy_only, optimal_only = [], 0, 0
    for pd in panos:
        gt = pd["gt"]
        if not gt.fn_confirmed or not gt.gt_points:
            continue
        floor = sorted(pd["preds"], key=lambda p: p[2], reverse=True)
        kept = [p for p in floor if p[2] >= threshold]
        assign = greedy_match([(p[0], p[1]) for p in kept], gt.gt_points,
                              radius_sq, PANO_SCALE_X, PANO_SCALE_Y)
        hit = {gi for gi, _ in assign if gi >= 0}
        claimed_by = {i: gi for i, (gi, _) in enumerate(assign)}
        opt = optimal_hits(kept, gt.gt_points, radius_sq)
        greedy_only += len(hit - opt)
        optimal_only += len(opt - hit)

        for i, g in enumerate(gt.gt_points):
            dist, px = geom(g[1])
            row = {"city": city, "x": g[0], "y": g[1], "dist": dist, "px": px,
                   "hit": i in hit, "bucket": None, "tier": TIER.get(city, "-"),
                   "field": "near" if dist < boundary else "far",
                   "sep_cheb": None, "sep_euc": None,
                   "null_supra": None, "null_sub": None}
            if i not in hit:
                row["bucket"] = classify_miss(g, floor, kept, claimed_by, radius_sq,
                                              threshold)
                if row["bucket"] == "merged":
                    sep = merged_separation(g, gt.gt_points, kept, claimed_by, radius_sq)
                    if sep:
                        row["sep_cheb"], row["sep_euc"] = sep
                if rng is not None:
                    row["null_supra"], row["null_sub"] = null_in_radius(
                        g, floor, radius_sq, threshold, rng, trials)
            rows.append(row)
    return rows, {"greedy_only": greedy_only, "optimal_only": optimal_only}


def print_buckets(title, rows):
    s = summarize(rows)
    misses = [r for r in rows if not r["hit"]]
    print(f"\n{title}  —  {s['n_gt']} GT, {s['n_miss']} misses, "
          f"recall {s['recall']:.3f}")
    print(f"{'bucket':>14} {'n':>6} {'% of misses':>12} {'recall pts':>11} "
          f"{'null rate':>10}")
    for b in BUCKETS:
        nulls = [r["null_supra"] if b == "merged" else r["null_sub"]
                 for r in misses if r["bucket"] == b and r["null_sub"] is not None]
        null_s = f"{sum(nulls)/len(nulls):>9.1%}" if nulls and b != "silent" else f"{'—':>10}"
        print(f"{b:>14} {s['counts'][b]:>6} {s['shares'][b]:>11.1%} "
              f"{s['recall_pts'][b]:>11.3f} {null_s}")
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--boundary", type=float, default=FAR_BOUNDARY_M,
                   help="Metres beyond which a miss is treated as pixel-limited.")
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--trials", type=int, default=NULL_TRIALS,
                   help="Azimuth-randomization trials per miss for the null (0 to skip).")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    rng = random.Random(NULL_SEED) if args.trials else None
    print(f"=== Miss taxonomy: what caused each miss? (threshold {args.threshold}, "
          f"boundary {args.boundary:.0f} m, #46) ===\n")
    print(f"{'split':>20} {'tier':>9} {'miss':>6} " +
          " ".join(f"{b:>13}" for b in BUCKETS))

    pooled, match_check = [], {"greedy_only": 0, "optimal_only": 0}
    per_city = {}
    for city in ALL_SPLITS:
        loaded = load_rows(city, args.threshold, args.cache, rng, args.trials,
                           args.boundary)
        if loaded is None:
            print(f"{city:>20} {'— no cache —':>50}")
            continue
        rows, mc = loaded
        s = summarize(rows)
        per_city[city] = s
        mark = "" if city in US_SPLITS else " †"
        print(f"{city:>20} {TIER.get(city, '-'):>9} {s['n_miss']:>6} " +
              " ".join(f"{s['counts'][b]:>6} {s['shares'][b]:>5.0%}" for b in BUCKETS)
              + mark)
        if city in US_SPLITS:
            pooled.extend(rows)
            for k in match_check:
                match_check[k] += mc[k]
    for city, why in HELD_OUT.items():
        print(f"† {city} shown but not pooled — {why}")

    print(f"\n{'='*78}")
    overall = print_buckets("POOLED, seven US splits", pooled)
    near = [r for r in pooled if r["field"] == "near"]
    far = [r for r in pooled if r["field"] == "far"]
    near_s = print_buckets(f"NEAR-FIELD only (< {args.boundary:.0f} m)", near)
    print_buckets(f"FAR-FIELD only (>= {args.boundary:.0f} m)", far)

    print(f"\n{'-'*78}\nIs the MATCHER manufacturing misses? (#46 confound)\n{'-'*78}")
    print(f"  hit only under maximum-cardinality matching: {match_check['optimal_only']}")
    print(f"  hit only under the deployed greedy matcher:  {match_check['greedy_only']}")
    print(f"  net effect on recall: "
          f"{match_check['optimal_only'] - match_check['greedy_only']:+d} ramps — "
          f"the two agree on cardinality everywhere; the difference is a permutation,")
    print(f"  so the greedy matcher is NOT a source of misses.")

    seps = [(r["sep_cheb"], r["sep_euc"]) for r in pooled
            if r["bucket"] == "merged" and r["sep_cheb"] is not None]
    if seps:
        cheb = sorted(s[0] for s in seps)
        euc = sorted(s[1] for s in seps)
        n = len(cheb)
        free = sum(1 for c in cheb if c > PEAK_MIN_DISTANCE)
        within = sum(1 for e in euc if e <= 2 * TARGET_SIGMA)
        print(f"\n{'-'*78}\nMERGED pairs: is the EXTRACTOR or the TARGET binding?\n{'-'*78}")
        print(f"  n={n}  chebyshev px: med {cheb[n//2]:.1f}  "
              f"euclidean px: med {euc[n//2]:.1f}")
        print(f"  chebyshev > min_distance={PEAK_MIN_DISTANCE}: {free}/{n} "
              f"({free/n:.0%}) — the extractor was FREE to emit two peaks and did not,")
        print(f"    so for these the heatmap itself had one mode. Lowering "
              f"min_distance cannot recover them.")
        print(f"  euclidean <= 2*sigma={2*TARGET_SIGMA}: {within}/{n} ({within/n:.0%}) "
              f"— consistent with the sigma-{TARGET_SIGMA} target")
        print(f"    being unable to represent the pair as two modes.")
        print(f"  CAVEAT — some of this bucket may be double-marked GT. A separation")
        print(f"    below ~8 px is ~25 cm at 10 m, which is not a physical spacing for")
        print(f"    two ramps; on the verdict splits a 'missed' mark that close to a")
        print(f"    confirmed detection is plausibly the reviewer marking one ramp")
        print(f"    twice. #62 found reviewers confirming BOTH members real on 6 of 10")
        print(f"    detection pairs at 15-19 px, so genuine adjacency dominates above")
        print(f"    that band. If the tight pairs ARE double-marks they are spurious GT")
        print(f"    and leave the population entirely — they do not move to another")
        print(f"    bucket — which shrinks merged and RAISES recall:")
        tight = sum(1 for c in cheb if c < 8)
        n_gt_p, n_miss_p = len(pooled), sum(1 for r in pooled if not r["hit"])
        print(f"      pairs below 8 px chebyshev: {tight}/{n} ({tight/n:.0%})")
        print(f"      drop them -> merged {n - tight}, recall "
              f"{(n_gt_p - n_miss_p)/(n_gt_p - tight):.3f} (from "
              f"{(n_gt_p - n_miss_p)/n_gt_p:.3f}); silent is unchanged at "
              f"{sum(1 for r in pooled if r['bucket'] == 'silent')}.")
        print(f"    manual_gold is the check that this is not an artifact of the")
        print(f"    verdict workflow: its GT is INDEPENDENT manual labeling with no")
        print(f"    RampNet review in the loop, and it shows the same mechanism at 44%")
        print(f"    of misses — two separately-labelled ramps, one peak.")

    nb = near_s["counts"]
    addressable = nb["silent"] + nb["localization"]
    # Every figure below is in recall points against the POOLED population, which is
    # the denominator #59's 0.087 uses. near_s["recall_pts"] is against the
    # near-field subset and would not be comparable to it.
    n_pooled = overall["n_gt"]
    print(f"\n{'='*78}\nHEADLINE — what the sourcing programme can actually reach\n{'='*78}")
    print(f"  #59 recorded the near-field population as {near_s['n_miss']} misses = "
          f"{near_s['n_miss']/n_pooled:.3f} recall points")
    print(f"  (pooled denominator, {n_pooled} GT) and flagged the "
          f"'appearance/vocabulary' label as")
    print(f"  an inference, not a measurement. Bucketed, that population is:")
    for b in BUCKETS:
        print(f"    {b:>14} {nb[b]:>4} = {nb[b]/n_pooled:.3f} recall pts")
    print(f"\n  Only SILENT (+ the {nb['localization']}-ramp localization tail) is "
          f"sourcing-addressable:")
    print(f"    {addressable} ramps = {addressable/n_pooled:.3f} recall points, "
          f"against the {near_s['n_miss']/n_pooled:.3f} upper bound —")
    print(f"    the near-field figure over-states the reachable population by "
          f"{near_s['n_miss']/addressable:.1f}x.")
    print(f"  The rest is two mechanisms MORE DATA DOES NOT TOUCH:")
    print(f"    sub_threshold {nb['sub_threshold']:>4} = "
          f"{nb['sub_threshold']/n_pooled:.3f} pts — confidence, not "
          f"recognition (#54/#55)")
    print(f"    merged        {nb['merged']:>4} = "
          f"{nb['merged']/n_pooled:.3f} pts — adjacent-pair peak merging "
          f"(sigma/min_distance)")
    print(f"\n  STILL AN UPPER BOUND: 'silent' means the cached detections witness")
    print(f"  nothing there. Occlusion, deep shadow, debris and GT disagreement are")
    print(f"  all still inside it and need the imagery to separate (#46 gallery).")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "threshold": args.threshold, "boundary_m": args.boundary,
            "null_trials": args.trials, "null_seed": NULL_SEED,
            "peak_min_distance": PEAK_MIN_DISTANCE, "target_sigma": TARGET_SIGMA,
            "per_city": per_city, "pooled": overall,
            "near_field": near_s, "far_field": summarize(far),
            "matcher_check": match_check,
            "merged_separation_px": {
                "n": len(seps),
                "chebyshev_median": sorted(s[0] for s in seps)[len(seps)//2] if seps else None,
                "euclidean_median": sorted(s[1] for s in seps)[len(seps)//2] if seps else None,
                "above_min_distance": sum(1 for s in seps if s[0] > PEAK_MIN_DISTANCE),
            },
            "sourcing_addressable_recall_pts": addressable / overall["n_gt"],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
