"""Did any OTHER model see the ramps RampNet was silent on? (#46, #35)

``miss_taxonomy.py`` left 128 pooled misses in the ``silent`` bucket — RampNet
produced nothing there, even at the 0.05 floor — and that bucket is what the sourcing
programme is sized against. It is still a mixture: a ramp behind a parked car, a ramp
in deep shadow, a ramp that is not really a ramp, and a ramp that is plainly visible
and simply not recognized are all in it, and only the last is a vocabulary failure
more training data could fix.

The gallery (``miss_gallery.py``) splits those by eye. **This splits them by
measurement first**, which is cheaper, reproducible, and not subject to a reviewer's
priors:

* **witnessed** — at least one challenger put a detection within the match radius.
  The ramp is therefore *visible in the imagery* — some model found it — so
  RampNet's failure is specific to RampNet. **This is the strongest evidence
  available for a genuine appearance/vocabulary failure**, and it needs no human.
* **unwitnessed** — no model detected anything there. Consistent with occlusion,
  deep shadow, or the ramp not being there at all; not distinguishable without
  imagery, so this is the population the gallery actually needs to work through.

**The density correction is not optional here.** OWLv2 emits 55-88 boxes per
panorama, so "OWLv2 found it" is close to meaningless on its own — the same problem
that forced the null-recall correction in ``docs/model_comparison.md`` and that showed
up again on the FP side in ``fp_taxonomy.py``. So every witness count is reported
against an **exact** chance expectation: for a prediction at height ``y``, the
azimuths that would land within the radius of our ramp form an arc, and the chance
that at least one of a model's predictions lands there is
``1 - prod(1 - p_i)`` over its predictions in that panorama. A model whose witness
rate merely matches its own null has witnessed nothing.

    python scripts/analysis/silent_witness.py
    python scripts/analysis/silent_witness.py --bucket merged --cache-dir ../.model_cache

Reads ``.model_cache`` and the committed caches. No GPU, no model load, no imagery.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402

import miss_taxonomy as mt  # noqa: E402
from miss_decomposition import US_SPLITS  # noqa: E402
from fp_taxonomy import CHALLENGERS, _compare_args, scaled  # noqa: E402

# Models sparse enough that a hit is evidence rather than coverage. Assigned from the
# measured densities in docs/model_comparison.md (1-4 boxes/pano, nulls 0.01-0.08)
# versus the open detectors' 55-88. The split is not a judgement call at the boundary:
# there is an order of magnitude between the two groups. Both are reported; only the
# sparse group is used for the headline, and the dense group's excess is printed so
# that choice is checkable rather than asserted.
SPARSE = ("gemini:gemini-3.6-flash", "gemini:gemini-3.1-pro-preview",
          "qwen:Qwen/Qwen3-VL-8B-Instruct", "qwen:Qwen/Qwen3-VL-32B-Instruct",
          "molmo:allenai/Molmo2-8B")
DENSE = ("owlv2", "gdino")


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_silent_witness.py
# --------------------------------------------------------------------------- #
def hit_chance(gt_point, preds, radius_sq):
    """P(at least one of ``preds`` lands within radius of ``gt_point`` by chance).

    Each prediction keeps its height and is given a uniformly random azimuth — the
    same null shape used elsewhere in this analysis, because both ramps and
    detections crowd the horizon band and randomizing height would compare against a
    distribution nothing lives in.

    A prediction at height ``py`` can reach the ramp only if ``|py - gy| < radius``;
    if it can, the azimuths that do form an arc of width ``2*sqrt(r^2 - dy^2)`` out of
    the panorama's ``PANO_SCALE_X`` circumference. Independence across predictions is
    the assumption; it slightly *understates* the null for a model whose boxes
    cluster, which is the conservative direction for a witness claim.
    """
    gx, gy = scaled(gt_point)
    miss_all = 1.0
    for p in preds:
        _, py = scaled(p)
        rem = radius_sq - (py - gy) ** 2
        if rem <= 0:
            continue
        p_hit = min(1.0, 2.0 * (rem ** 0.5) / PANO_SCALE_X)
        miss_all *= (1.0 - p_hit)
    return 1.0 - miss_all


def witnessed(gt_point, preds, radius_sq):
    """Did any prediction actually land within the match radius of this ramp?"""
    gx, gy = scaled(gt_point)
    for p in preds:
        px, py = scaled(p)
        if (px - gx) ** 2 + (py - gy) ** 2 < radius_sq:
            return True
    return False


def summarize(records, models):
    """Witness counts and chance expectations per model, plus the union over ``models``.

    ``records`` is one dict per miss holding ``{model: (hit, chance)}``. The union row
    is the one that matters — a ramp only needs *one* model to have seen it for the
    imagery to be shown to contain it — and its null is the chance that at least one
    model would hit by accident, which is why it cannot be summed from the rows above.
    """
    out = {}
    for m in models:
        hits = sum(1 for r in records if r["by_model"].get(m, (False, 0.0))[0])
        exp = sum(r["by_model"].get(m, (False, 0.0))[1] for r in records)
        out[m] = {"witnessed": hits, "expected": exp, "excess": hits - exp}
    union_hits, union_exp = 0, 0.0
    for r in records:
        vals = [r["by_model"][m] for m in models if m in r["by_model"]]
        if any(h for h, _ in vals):
            union_hits += 1
        miss = 1.0
        for _, c in vals:
            miss *= (1.0 - c)
        union_exp += 1.0 - miss
    out["__union__"] = {"witnessed": union_hits, "expected": union_exp,
                        "excess": union_hits - union_exp, "n": len(records)}
    return out


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def model_predictions(city, spec, cache, cargs):
    """``{pano_id: [points]}`` for one model in one city, straight from the cache."""
    import compare as C
    from detectors import build_detector, parse_model_spec

    bundle = os.path.join(REPO, "benchmark", city)
    records, verdicts, _ = C.load_bundle(bundle)
    gts = (C.load_manual_ground_truths(bundle) if verdicts is None
           else C.ground_truths_from_verdicts(records, verdicts))
    provider, model_id = parse_model_spec(spec)
    label, det = build_detector(provider, model_id, records, cargs)
    sig = det.signature() if hasattr(det, "signature") else None
    if sig is None:
        return label, None
    out = {}
    for pid in gts:
        pts = cache.get(C.cache_key(label, sig, city, pid))
        if pts is not None:
            out[pid] = pts
    return label, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bucket", default="silent", choices=list(mt.BUCKETS))
    p.add_argument("--threshold", type=float, default=mt.DEFAULT_THRESHOLD)
    p.add_argument("--cities", default=",".join(US_SPLITS))
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    import compare as C
    cache = C.DetectionCache(args.cache_dir, enabled=True)
    cargs = _compare_args(args.cache_dir)
    radius_sq = radius_sq_for()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    # The misses to explain.
    targets = []
    for city in cities:
        loaded = mt.load_rows(city, args.threshold, rng=None)
        if loaded is None:
            continue
        rows, _ = loaded
        targets.extend(r for r in rows
                       if not r["hit"] and r["bucket"] == args.bucket)
    if not targets:
        print(f"no '{args.bucket}' misses matched")
        return 0

    # Every model's predictions, once per city.
    preds = {}
    labels = {}
    for spec in CHALLENGERS:
        for city in cities:
            label, got = model_predictions(city, spec, cache, cargs)
            labels[spec] = label
            if got:
                preds[(spec, city)] = got

    records = []
    for t in targets:
        by_model = {}
        for spec in CHALLENGERS:
            pano_preds = preds.get((spec, t["city"]), {}).get(t["pano"])
            if pano_preds is None:
                continue
            pt = (t["x"], t["y"])
            by_model[spec] = (witnessed(pt, pano_preds, radius_sq),
                              hit_chance(pt, pano_preds, radius_sq))
        records.append({"city": t["city"], "pano": t["pano"], "x": t["x"],
                        "y": t["y"], "field": t["field"], "by_model": by_model})

    n = len(records)
    print(f"=== Who else saw the '{args.bucket}' misses? "
          f"({n} misses, threshold {args.threshold}) ===\n")
    print(f"{'model':>42} {'witnessed':>10} {'by chance':>10} {'excess':>9} {'':>6}")
    all_s = summarize(records, CHALLENGERS)
    for spec in CHALLENGERS:
        s = all_s[spec]
        tag = "sparse" if spec in SPARSE else "DENSE"
        print(f"{labels.get(spec, spec):>42} {s['witnessed']:>10} "
              f"{s['expected']:>10.1f} {s['excess']:>+9.1f} {tag:>6}")

    sparse_s = summarize(records, SPARSE)["__union__"]
    dense_s = summarize(records, DENSE)["__union__"]
    print(f"\n{'-'*82}")
    print(f"{'union of the 5 sparse models':>42} {sparse_s['witnessed']:>10} "
          f"{sparse_s['expected']:>10.1f} {sparse_s['excess']:>+9.1f}")
    print(f"{'union of the 2 dense detectors':>42} {dense_s['witnessed']:>10} "
          f"{dense_s['expected']:>10.1f} {dense_s['excess']:>+9.1f}")
    print(f"{'-'*82}")

    w, exp, corrected = sparse_s["witnessed"], sparse_s["expected"], sparse_s["excess"]
    print(f"\nWITNESSED BY A SPARSE MODEL: {w}/{n} raw ({w/n:.1%}) — but the raw rate")
    print(f"OVERSTATES IT. Chance placement alone would witness {exp:.1f} of these, so the")
    print(f"defensible count is the excess: ~{corrected:.0f} ({corrected/n:.1%}).")
    print(f"  A witnessed ramp is one some model put a detection on, so the imagery DOES")
    print(f"  contain a recognizable ramp and RampNet's failure is specific to RampNet.")
    print(f"  That is the strongest evidence for a genuine appearance/vocabulary failure")
    print(f"  obtainable without a human, and it is exactly what more training data targets.")
    print(f"\nUNWITNESSED: {n - w}/{n} ({(n-w)/n:.1%})")
    print(f"  No sparse model saw anything either. Consistent with occlusion, deep shadow,")
    print(f"  debris, or the ramp not being there — NOT distinguishable from the cached")
    print(f"  detections alone. THIS is the population miss_gallery.py has to work through;")
    print(f"  the witnessed ones no longer need eyes.")

    by_field = {}
    for f in ("near", "far"):
        sel = [r for r in records if r["field"] == f]
        if not sel:
            continue
        s = summarize(sel, SPARSE)["__union__"]
        by_field[f] = s
        print(f"\n  {f}-field: {s['witnessed']}/{len(sel)} raw "
              f"({s['witnessed']/len(sel):.1%}), chance {s['expected']:.1f}, "
              f"corrected ~{s['excess']:.0f} ({s['excess']/len(sel):.1%})")

    near = by_field.get("near")
    if near:
        # Pooled GT is the denominator #59's recall points are quoted against, so the
        # bracket below is directly comparable to its 0.087.
        n_pooled = sum(len(mt.load_rows(c, args.threshold, rng=None)[0])
                       for c in cities
                       if mt.load_rows(c, args.threshold, rng=None) is not None)
        lo, hi = near["excess"] / n_pooled, len(
            [r for r in records if r["field"] == "near"]) / n_pooled
        print(f"\n{'='*82}\nWHAT THIS DOES TO THE SOURCING NUMBER\n{'='*82}")
        print(f"  #59 bounded the sourcing-addressable population at 0.087 recall points.")
        print(f"  miss_taxonomy.py cut that to {hi:.3f} (the near-field 'silent' bucket).")
        print(f"  This brackets it further, against {n_pooled} pooled GT ramps:")
        print(f"    LOWER BOUND {lo:.3f} recall pts (~{near['excess']:.0f} ramps) — "
              f"CONFIRMED visible to")
        print(f"      another model, so confirmed recognizable and confirmed missed.")
        print(f"    UPPER BOUND {hi:.3f} recall pts (all "
              f"{len([r for r in records if r['field'] == 'near'])} near-field silent misses).")
        print(f"  The gap is the unwitnessed remainder: NOT shown to be unaddressable,")
        print(f"  only unproven either way. Closing it is what the gallery is for.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"bucket": args.bucket, "threshold": args.threshold,
                       "n": n, "per_model": {labels.get(k, k): v
                                             for k, v in all_s.items()},
                       "sparse_union": sparse_s, "dense_union": dense_s,
                       "by_field": by_field,
                       "unwitnessed": [
                           {"city": r["city"], "pano": r["pano"], "x": r["x"],
                            "y": r["y"], "field": r["field"]}
                           for r in records
                           if not any(r["by_model"].get(m, (False, 0))[0]
                                      for m in SPARSE)]}, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
