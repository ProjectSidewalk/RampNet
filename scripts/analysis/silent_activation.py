"""Phase 1 of the far-field `visible` anomaly study: attenuated, or absent? (#46)

The ``silent`` bucket is defined by peak extraction: no ``peak_local_max`` peak at or
above the 0.05 score floor within the match radius. That is a statement about *peaks*,
not about the heatmap — 0.04 and 0.0001 are both "silent", and they are different
failures. If the model produces a real-but-faint localized response at these ramps,
the silent bucket is the tail of the same confidence continuum ``sub_threshold``
lives on (a calibration/threshold/training story, gainesville's mechanism). If the
heatmap is flat at chance level, the model has no representation of the ramp at all —
a genuine vocabulary or scale gap, which is what Phase 2's scale counterfactual then
separates.

For every pooled silent miss (near and far — the near-field verdicts feed the 0.013
sourcing estimate just as directly), this loads the published model, runs one forward
pass per panorama, and reads:

* ``act`` — the max heatmap value within the match radius of the missed ramp (the
  radius and grid are exactly the matcher's: the scaled space *is* the 512x1024
  heatmap). Note ``act`` can exceed the 0.05 floor without contradicting ``silent``
  — a shoulder of a neighbouring peak is not a local maximum; those cases are
  counted separately rather than silently pooled.
* a per-miss **null**: the same radius-max at ``NULL_TRIALS`` random azimuths in the
  same panorama at the same elevation — the same null shape every #46 analysis uses,
  because both ramps and heatmap mass crowd the horizon band. ``act`` is reported as
  a percentile of its own panorama's null, so "there is signal here" is a claim
  against that pano's actual noise floor, not against zero.

Model: the published ``projectsidewalk/rampnet-model`` weights — the same checkpoint
every committed cache came from (``operating_point_curve.py`` extract). Single-pass,
no TTA, fp32, matching ``analysis_out/op_cache``. Needs the panorama imagery
(``benchmark/<city>/panos``, git-ignored — ``--panos-root`` from a worktree) and a
GPU-ish machine; the RTX 3070 does ~10 s/pano.

    python scripts/analysis/silent_activation.py --panos-root D:/Git/RampNet
    python scripts/analysis/silent_activation.py --json-out analysis_out/silent_activation.json
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

import miss_taxonomy as mt  # noqa: E402
from miss_decomposition import DEFAULT_THRESHOLD, US_SPLITS  # noqa: E402
from farfield_forensics import load_rated, quartiles, row_key  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)

NULL_TRIALS = 200
NULL_SEED = 20260731

# The heatmap grid IS the matcher's scaled space (PANO_SCALE_X x PANO_SCALE_Y =
# 1024 x 512), asserted at runtime so a future resolution change cannot silently
# desynchronize the two.
HEAT_W, HEAT_H = int(PANO_SCALE_X), int(PANO_SCALE_Y)


# --------------------------------------------------------------------------- #
# Pure core (no torch, no I/O) — unit-tested in tests/test_silent_activation.py
# --------------------------------------------------------------------------- #
def radius_max(heat, x, y, radius_sq=None):
    """Max heatmap value within the match radius of normalized point ``(x, y)``.

    Columns wrap at the 360-degree seam (a radius crossing x=0 continues at x=1);
    rows clamp — there is nothing above the top of a panorama. Values are clipped
    to [0, 1] exactly as ``peaks_to_dets`` clips before peak extraction, so an
    ``act`` here and a peak score there are on the same scale.
    """
    return site_profile(heat, x, y, radius_sq)[0]


def site_profile(heat, x, y, radius_sq=None):
    """``(act, off_px, center)`` for the window around normalized ``(x, y)``.

    ``off_px`` is how far from the site the in-window maximum sits, and ``center``
    is the value at the site itself — together they separate a response *at* the
    ramp from a neighbouring mode's tail reaching *into* the window, which ``act``
    alone cannot do (and 62% of silent misses turn out to need the distinction).
    """
    if radius_sq is None:
        radius_sq = radius_sq_for()
    H, W = len(heat), len(heat[0])
    r = radius_sq ** 0.5
    cx, cy = x * W, y * H
    best, off = 0.0, 0.0
    for row in range(max(0, int(cy - r)), min(H, int(cy + r) + 2)):
        dy2 = (row - cy) ** 2
        if dy2 >= radius_sq:
            continue
        span = (radius_sq - dy2) ** 0.5
        for col in range(int(cx - span), int(cx + span) + 2):
            dx = col - cx
            if dx * dx + dy2 >= radius_sq:
                continue
            v = min(float(heat[row][col % W]), 1.0)
            if v > best:
                best, off = v, (dx * dx + dy2) ** 0.5
    center = min(float(heat[min(H - 1, max(0, round(cy)))]
                       [round(cx) % W]), 1.0)
    return best, off, max(center, 0.0)


def nearest_peak(preds, x, y):
    """``(dist_px, score)`` of the closest cached floor peak, in matcher units.

    ``preds`` are the panorama's cached ``(x, y, score)`` floor peaks (>= 0.05).
    For a silent miss any such peak is by definition OUTSIDE the match radius, so
    this measures how far away the nearest thing the model actually said is —
    the difference between "a neighbouring mode's tail reaches the site" (~1-2
    radii) and "the nearest response is nowhere near" (many radii).
    """
    if not preds:
        return float("inf"), None
    best, score = float("inf"), None
    for p in preds:
        dx = abs(p[0] - x) * PANO_SCALE_X
        dx = min(dx, PANO_SCALE_X - dx)
        d = (dx * dx + ((p[1] - y) * PANO_SCALE_Y) ** 2) ** 0.5
        if d < best:
            best, score = d, p[2]
    return best, score


def null_percentile(heat, x, y, rng, trials=NULL_TRIALS, radius_sq=None):
    """``(act, percentile, null_med, null_p95)`` of the site's radius-max vs its pano.

    The null keeps the ramp's elevation and randomizes azimuth — the shape every
    other #46 null uses. Draws whose window would overlap the site's own window
    (wrapped column distance under 2R) are rejected and redrawn: the question is
    whether the site's response exceeds what the *rest* of the elevation band
    produces, and a draw that reads the site's own bump back would contaminate
    exactly the sparse-heatmap case this analysis exists to detect. The percentile
    counts ties as half, so a flat heatmap reads 0.5, not 1.0.
    """
    if radius_sq is None:
        radius_sq = radius_sq_for()
    act = radius_max(heat, x, y, radius_sq)
    W = len(heat[0])
    exclude = 2.0 * (radius_sq ** 0.5) / W  # normalized column distance
    draws = []
    while len(draws) < trials:
        nx = rng.random()
        dx = abs(nx - x)
        if min(dx, 1.0 - dx) < exclude:
            continue
        draws.append(radius_max(heat, nx, y, radius_sq))
    draws.sort()
    below = sum(1 for d in draws if d < act)
    ties = sum(1 for d in draws if d == act)
    pct = (below + 0.5 * ties) / trials
    return act, pct, draws[trials // 2], draws[int(trials * 0.95)]


def group_of(row, queue_keys, rated_by_rowkey):
    """Which selection stratum a silent miss belongs to (Phase 0's partition)."""
    key = row_key(row)
    if key not in queue_keys:
        return "witnessed"
    if key in rated_by_rowkey:
        return "rated"
    return "below_floor"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--witness", default=os.path.join(OUT, "silent_witness.json"))
    p.add_argument("--gallery", default=os.path.join(REPO, "benchmark",
                                                     "miss_taxonomy_46"))
    p.add_argument("--panos-root", default=REPO,
                   help="Checkout holding benchmark/<city>/panos (git-ignored, so "
                        "in a worktree it lives in the main checkout instead).")
    p.add_argument("--cities", default=",".join(US_SPLITS))
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many panos (smoke test).")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    import torch
    import threshold_sweep as ts
    assert (HEAT_W, HEAT_H) == (1024, 512), "heatmap grid != matcher space"

    from miss_gallery import load_queue, pano_path
    queue_keys = load_queue(args.witness)
    rated = load_rated(args.gallery, field=None)
    rated_by_rowkey = {(v["city"], v["pano"], round(float(v["x"]), 6),
                        round(float(v["y"]), 6)): v for v in rated.values()}

    from operating_point_curve import CACHE_DIR, read_cache
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    by_pano, preds_by = {}, {}
    for city in cities:
        loaded = mt.load_rows(city, args.threshold, rng=None)
        if loaded is None:
            continue
        for r in loaded[0]:
            if not r["hit"] and r["bucket"] == "silent":
                by_pano.setdefault((city, r["pano"]), []).append(r)
        panos, _ = read_cache(os.path.join(CACHE_DIR, f"{city}.json"))
        for pd in panos:
            preds_by[(city, pd["pano"])] = pd["preds"]
    n_miss = sum(len(v) for v in by_pano.values())
    print(f"=== Silent-miss activation forensics (threshold {args.threshold}, "
          f"{n_miss} misses in {len(by_pano)} panos, #46 Phase 1) ===", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    print(f"device={device} model=projectsidewalk/rampnet-model "
          f"(single-pass fp32, as op_cache)", flush=True)

    rng = random.Random(NULL_SEED)
    radius_sq = radius_sq_for()
    results, skipped = [], 0
    for i, ((city, pano), misses) in enumerate(sorted(by_pano.items()), 1):
        path = pano_path(city, pano, args.panos_root)
        if not os.path.exists(path):
            skipped += len(misses)
            continue
        heat = ts.heatmap_for(model, device, path, use_fp16=False)
        for r in misses:
            act, pct, null_med, null_p95 = null_percentile(
                heat, r["x"], r["y"], rng, radius_sq=radius_sq)
            _, off_px, center = site_profile(heat, r["x"], r["y"], radius_sq)
            npk_px, npk_score = nearest_peak(preds_by.get((city, pano), []),
                                             r["x"], r["y"])
            key = row_key(r)
            v = rated_by_rowkey.get(key)
            results.append({
                "city": city, "pano": pano, "x": r["x"], "y": r["y"],
                "field": r["field"], "dist_m": round(r["dist"], 1),
                "px": round(r["px"], 1),
                "group": group_of(r, queue_keys, rated_by_rowkey),
                "verdict": v["verdict"] if v else None,
                "act": round(act, 5), "null_pct": round(pct, 3),
                "null_med": round(null_med, 5), "null_p95": round(null_p95, 5),
                "above_own_null_p95": act > null_p95,
                "argmax_off_px": round(off_px, 1),
                "act_at_site": round(center, 5),
                "nearest_peak_px": (round(npk_px, 1)
                                    if npk_px != float("inf") else None),
                "nearest_peak_score": (round(npk_score, 3)
                                       if npk_score is not None else None),
            })
        del heat
        if i % 10 == 0:
            print(f"  {i}/{len(by_pano)} panos", flush=True)
        if args.limit and i >= args.limit:
            print(f"  --limit {args.limit} reached", flush=True)
            break
    if skipped:
        print(f"  [!] {skipped} misses skipped — panorama not on disk under "
              f"{args.panos_root}", flush=True)

    # ----------------------------------------------------------------------- #
    print(f"\n{'-'*78}\nACTIVATION AT THE MISSED RAMP, by field and stratum\n{'-'*78}")
    print(f"{'population':>34} {'n':>4} {'act q1/med/q3':>18} "
          f"{'>p95 of own null':>17} {'act>=0.01':>10}")
    groups = {}
    for field in ("near", "far"):
        for grp in ("rated", "below_floor", "witnessed"):
            sel = [r for r in results if r["field"] == field and r["group"] == grp]
            if not sel:
                continue
            name = f"{field} / {grp}"
            groups[name] = sel
    for name, sel in list(groups.items()) + [("ALL silent misses", results)]:
        q = quartiles([r["act"] for r in sel])
        n_sig = sum(1 for r in sel if r["above_own_null_p95"])
        n_01 = sum(1 for r in sel if r["act"] >= 0.01)
        print(f"{name:>34} {len(sel):>4} "
              f"{q[0]:>6.4f}/{q[1]:>6.4f}/{q[2]:>6.4f} "
              f"{n_sig:>7}/{len(sel):<7} {n_01:>10}")

    vis = [r for r in results if r["verdict"] == "visible"]
    if vis:
        q = quartiles([r["act"] for r in vis])
        n_sig = sum(1 for r in vis if r["above_own_null_p95"])
        print(f"\n  rated `visible` only (n={len(vis)}): act q1/med/q3 "
              f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}; {n_sig}/{len(vis)} above their "
              f"own pano's null p95")

    # What the in-window mass actually IS. A silent miss has no floor peak in
    # radius by definition, so act >= 0.05 can only be an outside mode's tail;
    # the argmax offset and the nearest cached peak make that checkable rather
    # than asserted.
    print(f"\n{'-'*78}\nDECOMPOSITION — what the in-window response is\n{'-'*78}")
    r_px = radius_sq ** 0.5
    cls = {"absent": [], "faint_local": [], "tail": []}
    for r in results:
        if r["act"] < 0.01:
            cls["absent"].append(r)
        elif r["act"] >= 0.05:
            cls["tail"].append(r)
        else:
            cls["faint_local"].append(r)
    for name, sel in cls.items():
        if not sel:
            continue
        med_off = quartiles([r["argmax_off_px"] for r in sel])[1]
        npks = [r["nearest_peak_px"] for r in sel if r["nearest_peak_px"]]
        med_npk = quartiles(npks)[1] if npks else float("nan")
        n_vis = sum(1 for r in sel if r["verdict"] == "visible")
        print(f"  {name:>12}: {len(sel):>3}  (rated visible {n_vis:>2})  "
              f"argmax off med {med_off:>4.1f} px  nearest floor peak med "
              f"{med_npk:>5.1f} px ({med_npk/r_px:.1f}R)")
    tail_near_edge = sum(1 for r in cls['tail']
                         if r['argmax_off_px'] > 0.75 * r_px)
    print(f"  tail cases with argmax in the window's outer quarter: "
          f"{tail_near_edge}/{len(cls['tail'])} — the mass is entering from "
          f"outside, not centred on the ramp")

    print(f"\n  Reading: 'absent' = the heatmap is genuinely flat at the site.")
    print(f"  'faint_local' = a real sub-floor response at the site itself.")
    print(f"  'tail' = a neighbouring supra-floor mode's slope reaches the window —")
    print(f"  the site contributed no mode of its own, but the model responded to")
    print(f"  something adjacent (cf. the merged bucket's sigma story). The three")
    print(f"  continue differently: absent -> Phase 2's scale counterfactual;")
    print(f"  faint_local -> threshold/calibration (the sub_threshold continuum);")
    print(f"  tail -> representation (sigma), not vocabulary.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "null_trials": NULL_TRIALS,
                       "null_seed": NULL_SEED, "n": len(results),
                       "skipped_no_imagery": skipped,
                       "model": "projectsidewalk/rampnet-model",
                       "tta": False, "results": results}, fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
