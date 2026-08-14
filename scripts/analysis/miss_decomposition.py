"""Of the recall we're missing, how much can MORE DATA even reach? (#59, #38, #48)

E1 (``stage1_label_recall.py``) established that the far-field cliff is not
inherited from the labels: at 25-40 m a training label was present at 78% of gold
ramps while the model detected 49%. Distant ramps are under-detected because a
1.2 m ramp at 30 m is ~25 px in a 4096-px panorama — **more examples do not add
pixels.**

That splits the missing recall into two populations with *different* fixes, and
nobody has sized them:

* **Far-field misses** — pixel-starved. More cities cannot help. Addressed by more
  pixels on target: higher resolution (#25), or a closer viewpoint — which is what
  multi-view fusion (#48) buys, since a ramp at 30 m is at 8 m two panoramas later
  (#38). No new data collection required; those panoramas already exist.
* **Near-field misses** — the model had plenty of pixels and still failed. That is
  an *appearance/vocabulary* failure (Paterson's paired tactile surfaces,
  Gainesville's diagonal arterial ramps), and it is the population a broader,
  more diverse training corpus could plausibly fix.

This measures the split across all nine benchmark splits, so the choice between
the multi-view programme and the sourcing programme rests on a number.

    python scripts/analysis/miss_decomposition.py
    python scripts/analysis/miss_decomposition.py --threshold 0.55

Reads the committed low-floor caches (``analysis_out/op_cache/*.json``), so no
GPU, no network and no imagery. Geometry and matching are imported from E1 rather
than reimplemented — one estimator, one matcher, or the curves stop comparing.
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

from operating_point_curve import read_cache  # noqa: E402
from stage1_label_recall import (  # noqa: E402
    DIST_BUCKETS, SIZE_BUCKETS, geom, hit_indices)

CACHE_DIR = os.path.join(OUT, "op_cache")

# The seven US city splits carry verdict-grade GT and are the pooled basis, matching
# low_floor_sweep.US_SPLITS. budapest is swept but not pooled (single-rater, low
# reviewer confidence); sao_paulo is high-confidence GT held out for geography;
# manual_gold is in-distribution GSV, an in-domain reference rather than a deployment
# city. This registry must stay in step with low_floor_sweep's — a split missing here
# is silently skipped by every default in this family (export_model_cache,
# imagery_manifest, miss_taxonomy, fp_taxonomy, the galleries), which looks exactly
# like a result nobody ran. test_registries_agree_with_low_floor_sweep enforces it.
US_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
             "gainesville")

# Imagery tier per split (benchmark/README.md). Reported separately because the
# distance estimate is NOT equally trustworthy across them: flat-ground geometry
# agrees with DA3 depth at Spearman 0.95 on GSV and 0.81 on Mapillary, since
# consumer rigs are not level and the terrain is not flat. Four of the seven
# pooled splits are Mapillary, so the pooled far-field share carries that bias.
# Held-out splits are deliberately absent and print "-": tiers exist to group the
# POOLED rows, and a split that never pools has no business appearing in one.
TIER = {"bend": "gsv", "paterson": "gsv", "gainesville": "gsv",
        "richmond": "mapillary", "clovis": "mapillary", "morgantown": "mapillary",
        "annapolis": "mapillary"}
HELD_OUT = {"budapest_district5": "single-rater GT at low reviewer confidence",
            "sao_paulo": "non-US city, and the pooled basis is US deployment "
                         "(GT is high confidence; held out for geography, "
                         "not GT quality)",
            "manual_gold": "in-distribution reference, not a deployment city"}
ALL_SPLITS = US_SPLITS + tuple(HELD_OUT)

# Where the model stops being able to see. Recall is 0.90 at 18-25 m and 0.49 at
# 25-40 m on the gold set (E1), so 18 m is the last distance with adequate signal.
# Misses beyond it are pixel-limited; misses inside it are not.
FAR_BOUNDARY_M = 18.0

# The deployed recommendation (#79, PR merged). The caches hold a 0.05 floor, so
# any threshold at or above that is available without re-extracting.
DEFAULT_THRESHOLD = 0.30


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_miss_decomposition.py
# --------------------------------------------------------------------------- #
def split_misses(rows, boundary=FAR_BOUNDARY_M):
    """Partition misses into pixel-limited and appearance-limited populations.

    ``far_share`` is the headline: the fraction of all missed ramps that sit
    beyond the boundary, i.e. the share **multi-view or higher resolution could
    address and more cities could not**. Its complement is the share a broader
    training corpus might reach.
    """
    misses = [r for r in rows if not r["hit"]]
    far = [r for r in misses if r["dist"] >= boundary]
    near = [r for r in misses if r["dist"] < boundary]
    n_gt = len(rows)
    return {
        "n_gt": n_gt,
        "n_miss": len(misses),
        "recall": (n_gt - len(misses)) / n_gt if n_gt else float("nan"),
        "n_far_miss": len(far),
        "n_near_miss": len(near),
        "far_share": len(far) / len(misses) if misses else float("nan"),
        "near_share": len(near) / len(misses) if misses else float("nan"),
        # Missing recall attributable to each population, in recall points.
        "far_miss_pts": len(far) / n_gt if n_gt else float("nan"),
        "near_miss_pts": len(near) / n_gt if n_gt else float("nan"),
        "n_above_horizon": above_horizon(rows),
    }


def above_horizon(rows):
    """GT ramps the flat-ground model places at or above the horizon.

    Geometrically impossible for a ground ramp, so each one is a direct tell of an
    **unleveled rig or a hill** — and `geom()` clamps them to 150 m, which dumps
    them straight into the far-field bucket. This is the bias that matters here:
    a camera tilted up pushes a near ramp toward the horizon, so geometry calls it
    far, and the far-field share is **overstated**. `docs/detection_recall_analysis.md`
    found DA3 depth rescuing exactly these (4 in Richmond), and reports geometry
    agreeing with depth at Spearman 0.95 on GSV but only 0.81 on Mapillary.

    Reported per split so the reader can discount the tiers where it is common.
    """
    return sum(1 for r in rows if r["y"] <= 0.5)


def multiview_ceiling(rows, boundary=FAR_BOUNDARY_M):
    """Recall if every far ramp were re-observed at near range, and nothing else changed.

    An **optimistic bound**, not a forecast. It assumes a closer capture exists for
    every far ramp (true where panorama spacing is dense, false at the edge of a
    run) and that re-observation succeeds at the measured near-field rate. It
    ignores the cost of fusing views and of the extra false positives more looks
    would produce. Read it as "what is even on the table".
    """
    near = [r for r in rows if r["dist"] < boundary]
    far = [r for r in rows if r["dist"] >= boundary]
    if not rows or not near:
        return float("nan")
    near_recall = sum(1 for r in near if r["hit"]) / len(near)
    hits = sum(1 for r in near if r["hit"]) + near_recall * len(far)
    return hits / len(rows)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_rows(city, threshold, cache_dir=CACHE_DIR):
    """One row per scoreable GT ramp in a split: geometry + whether RampNet hit it."""
    path = os.path.join(cache_dir, f"{city}.json")
    if not os.path.exists(path):
        return None
    panos, _meta = read_cache(path)
    rows = []
    for pd in panos:
        gt = pd["gt"]
        if not gt.fn_confirmed:          # pano's missed-ramp check never confirmed
            continue
        preds = sorted((p for p in pd["preds"] if p[2] >= threshold),
                       key=lambda p: p[2], reverse=True)
        hit = hit_indices([(p[0], p[1]) for p in preds], gt.gt_points)
        for i, (gx, gy) in enumerate(gt.gt_points):
            dist, px = geom(gy)
            rows.append({"city": city, "x": gx, "y": gy, "dist": dist, "px": px,
                         "hit": i in hit})
    return rows


def print_pooled(rows, boundary):
    print(f"\n{'='*78}\nPOOLED — seven US splits\n{'='*78}")
    for key, buckets, unit in (("dist", DIST_BUCKETS, "m"), ("px", SIZE_BUCKETS, "px")):
        print(f"\n{'bucket':>12} {'n GT':>7} {'recall':>8} {'misses':>8} {'% of all misses':>16}")
        total_miss = sum(1 for r in rows if not r["hit"]) or 1
        for lo, hi in buckets:
            b = [r for r in rows if lo <= r[key] < hi]
            if not b:
                continue
            miss = sum(1 for r in b if not r["hit"])
            hi_s = "+" if hi > 1e8 else f"{int(hi)}"
            label = f"{int(lo)}-{hi_s} {unit}"
            print(f"{label:>12} {len(b):>7} {1 - miss/len(b):>8.3f} {miss:>8} "
                  f"{100.0*miss/total_miss:>15.1f}%")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--boundary", type=float, default=FAR_BOUNDARY_M,
                   help="Metres beyond which a miss is treated as pixel-limited.")
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    per_city, pooled = {}, []
    print(f"=== Miss decomposition: far-field vs appearance (threshold "
          f"{args.threshold}, boundary {args.boundary:.0f} m) ===\n")
    print(f"{'split':>20} {'tier':>9} {'GT':>6} {'recall':>7} {'miss':>6} "
          f"{'far':>5} {'near':>5} {'far% miss':>10} {'MV ceil':>8} {'>horiz':>7}")
    for city in ALL_SPLITS:
        rows = load_rows(city, args.threshold, args.cache)
        if rows is None:
            print(f"{city:>20} {'— no cache —':>50}")
            continue
        s = split_misses(rows, args.boundary)
        s["multiview_ceiling"] = multiview_ceiling(rows, args.boundary)
        per_city[city] = s
        if city in US_SPLITS:
            pooled.extend(rows)
        mark = "" if city in US_SPLITS else " †"
        print(f"{city:>20} {TIER.get(city, '-'):>9} {s['n_gt']:>6} {s['recall']:>7.3f} "
              f"{s['n_miss']:>6} {s['n_far_miss']:>5} {s['n_near_miss']:>5} "
              f"{s['far_share']:>9.1%} {s['multiview_ceiling']:>8.3f} "
              f"{s['n_above_horizon']:>7}{mark}")
    for city, why in HELD_OUT.items():
        print(f"† {city} shown but not pooled — {why}")

    print(f"\n{'-'*78}")
    print("By imagery tier — geometry is only trustworthy on the GSV rows "
          "(see above_horizon)")
    print("-" * 78)
    print(f"{'tier':>10} {'GT':>7} {'recall':>8} {'miss':>6} {'far% of miss':>14} {'>horiz':>8}")
    for tier in ("gsv", "mapillary"):
        tr = [r for r in pooled if TIER.get(r["city"]) == tier]
        if not tr:
            continue
        t = split_misses(tr, args.boundary)
        print(f"{tier:>10} {t['n_gt']:>7} {t['recall']:>8.3f} {t['n_miss']:>6} "
              f"{t['far_share']:>13.1%} {t['n_above_horizon']:>8}")

    ps = split_misses(pooled, args.boundary)
    ps["multiview_ceiling"] = multiview_ceiling(pooled, args.boundary)
    print_pooled(pooled, args.boundary)

    print(f"\n{'='*78}\nHEADLINE (pooled, {ps['n_gt']} GT ramps, {ps['n_miss']} misses)\n{'='*78}")
    print(f"  recall {ps['recall']:.3f}  ->  missing {1-ps['recall']:.3f}")
    print(f"  FAR-FIELD  (>= {args.boundary:.0f} m): {ps['n_far_miss']:>5} misses "
          f"= {ps['far_share']:.1%} of misses, {ps['far_miss_pts']:.3f} recall pts")
    print(f"    pixel-limited. More cities cannot reach these; multi-view (#48/#38)")
    print(f"    or higher resolution (#25) can.")
    print(f"  NEAR-FIELD (<  {args.boundary:.0f} m): {ps['n_near_miss']:>5} misses "
          f"= {ps['near_share']:.1%} of misses, {ps['near_miss_pts']:.3f} recall pts")
    print(f"    adequate pixels, still missed -> appearance/vocabulary. This is the")
    print(f"    population a broader, more diverse training corpus could address.")
    print(f"\n  Optimistic multi-view ceiling: {ps['multiview_ceiling']:.3f} "
          f"(+{ps['multiview_ceiling']-ps['recall']:.3f} recall) — assumes a closer")
    print(f"  capture exists for every far ramp and succeeds at the near-field rate.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "boundary_m": args.boundary,
                       "per_city": per_city, "pooled": ps}, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
