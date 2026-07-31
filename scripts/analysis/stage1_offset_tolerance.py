"""How much coordinate error can Stage 1 absorb? (issues #96, #59)

§5f measures how far a city's published coordinates sit from the physical ramp.
That number is uninterpretable without a tolerance, and the tolerance is a
property of **Stage 1**, not of the aerial imagery it was measured on.

**The load-bearing fact, from `stage_one/dataset_generation/download_dataset.py`:**

    azimuth, _, _ = geod.inv(pano_lng, pano_lat, ramp_lng, ramp_lat)
    azimuth = azimuth - pano_angle
    persp = equirectangular_to_perspective(equi, 90, azimuth, -30, 1024, 1024)
    persp = persp[0:1024, 341:341+341]          # centre third only

The government coordinate is consumed **only for its bearing from the
panorama**. The range is computed and thrown away. The crop model then localises
the ramp inside a strip cut around that bearing, so the label's position comes
from the *imagery*, not from the coordinate. Three consequences, none of which
are visible from the offset distribution alone:

1. **Tolerance is angular, not metric.** The strip is the centre 341 px of a
   1024 px, 90° FOV rendering, i.e. **±18.4°** of azimuth (arctan(170.5/512), not
   90°·341/1024 — a pinhole projection is not linear in angle).
2. **Radial error is free.** An offset along the line of sight does not move the
   bearing at all. Only the tangential component costs anything, and for an
   error of unknown direction the expected tangential fraction is 2/pi ~ 0.64.
3. **The metric tolerance scales with range**: a tangential offset survives if it
   is under ``0.332 * d``. At 3 m that is 1.0 m; at 20 m it is 6.6 m. **The same
   coordinate error is fatal next to the camera and irrelevant across the
   intersection** — so the answer depends on how far ramps actually are, which is
   why this reads real distances out of the benchmark ground truth rather than
   assuming one.

What is computed: P(the true ramp falls outside the strip cut for its own
record), by Monte Carlo over the empirical offset distribution x the empirical
ramp-range distribution x a uniformly random error direction. Exact geometry —
camera at the origin, ramp at range ``d`` on bearing 0, published point at
``d + o*cos(theta), o*sin(theta)``, so the bearing error is
``atan2(o*sin(theta), d + o*cos(theta))``.

**What it is not.** This is the *geometric* tolerance — whether the ramp is in
the strip at all. It does not model whether the crop model still localises a ramp
sitting near the strip edge, which would need the round-2 checkpoint
(`stage_one/crop_model/ps_and_manual_model/best_model.pth`, not in the repo) and
a GPU. Read it as an upper bound on what the pipeline tolerates: real degradation
begins earlier than this says, never later.

    python scripts/analysis/stage1_offset_tolerance.py \
        --verdicts analysis_out/review_denver-co/verdicts.json

Needs only committed benchmark bundles. CPU, no network.
"""
import argparse
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

# Geometry of the crop, read off download_dataset.py rather than assumed.
PERSP_W = 1024          # equirectangular_to_perspective(..., 1024, 1024)
PERSP_FOV = 90.0
CROP_LO, CROP_HI = 341, 341 + 341   # persp[0:1024, 341:341+341]

# Flat-ground range estimate, as used (and DA3-validated to 6.5-8.5%) by
# scripts/analysis/precision_by_distance.py.
CAM_H = 2.5


def crop_half_angle_deg(width=PERSP_W, fov=PERSP_FOV, lo=CROP_LO, hi=CROP_HI):
    """Half-width of the crop in degrees of azimuth. Pure.

    A pinhole projection is not linear in angle, so this is arctan of the pixel
    offset over the focal length -- NOT fov * crop_px / width, which would
    overstate it by ~8%.
    """
    f = (width / 2.0) / math.tan(math.radians(fov / 2.0))
    left = math.degrees(math.atan((lo - width / 2.0) / f))
    right = math.degrees(math.atan((hi - width / 2.0) / f))
    return min(abs(left), abs(right))


def bearing_error_deg(offset_m, range_m, theta_rad):
    """Bearing error induced by an offset of unknown direction. Pure.

    Camera at the origin, true ramp at ``range_m`` on bearing 0, published point
    displaced by ``offset_m`` at ``theta_rad``. Exact, not a small-angle
    approximation -- offsets comparable to the range do occur at close ramps.
    """
    dx = range_m + offset_m * math.cos(theta_rad)
    dy = offset_m * math.sin(theta_rad)
    return abs(math.degrees(math.atan2(dy, dx)))


def ground_range(y_normalized, cam_h=CAM_H):
    """Flat-ground range from a normalised pano row. Pure."""
    dep = (y_normalized - 0.5) * math.pi
    return cam_h / math.tan(dep) if dep > 1e-4 else float("inf")


def benchmark_ranges(cities):
    """Ranges of every ground-truth ramp in the committed benchmark bundles.

    Handles both bundle kinds. ``manual_gold`` carries independently drawn YOLO
    labels and no RampNet review, so it has ``gt_source.json`` instead of
    ``verdicts.json`` — and it is the one bundle whose ground truth was never
    anchored on RampNet's own detections, so it is the last one to drop.
    """
    from rampnet.detection_eval import build_ground_truth
    from compare import load_bundle, load_manual_ground_truths
    out, per_city = [], {}
    for city in cities:
        path = os.path.join(REPO, "benchmark", city)
        if not os.path.isdir(path):
            print("  (skipping %s -- no such bundle)" % city)
            continue
        records, verdicts, _ = load_bundle(path)
        if verdicts is None:
            gts = load_manual_ground_truths(path).values()
        else:
            gts = [build_ground_truth(records[pid]["detections"], e["dets"],
                                      e["missed"], e["no_missed"])
                   for pid, e in verdicts.items()]
        got = 0
        for gt in gts:
            for _, y in gt.gt_points:
                r = ground_range(y)
                if math.isfinite(r) and 0.5 < r < 100.0:
                    out.append(r)
                    got += 1
        per_city[city] = got
    return out, per_city


def simulate(offsets, ranges, half_angle, trials=200000, seed=20260731):
    """P(true ramp falls outside its own strip), plus the marginal by range."""
    rng = random.Random(seed)
    if not offsets or not ranges:
        return None
    outside = 0
    by_range = {}
    buckets = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 1e9)]
    for _ in range(trials):
        o = rng.choice(offsets)
        d = rng.choice(ranges)
        err = bearing_error_deg(o, d, rng.uniform(0, 2 * math.pi))
        miss = err > half_angle
        outside += miss
        for lo, hi in buckets:
            if lo <= d < hi:
                k = "%d-%s m" % (lo, "inf" if hi > 1e8 else int(hi))
                b = by_range.setdefault(k, [0, 0])
                b[0] += miss
                b[1] += 1
                break
    return {
        "p_outside": outside / trials,
        "trials": trials,
        "by_range": {k: {"outside": v[0], "n": v[1], "rate": v[0] / v[1]}
                     for k, v in sorted(by_range.items(),
                                        key=lambda kv: int(kv[0].split("-")[0]))},
    }


def sweep(ranges, half_angle, scales, offsets, trials=60000, seed=20260731):
    """Same simulation with every offset multiplied by ``s``.

    This is the reusable part: it converts "city X has median offset m" into
    "city X loses this fraction of its labels", so a future city does not need
    its own bespoke argument about whether its number is good enough.
    """
    out = []
    for s in scales:
        scaled = [o * s for o in offsets]
        med = sorted(scaled)[len(scaled) // 2]
        res = simulate(scaled, ranges, half_angle, trials=trials, seed=seed)
        out.append({"scale": s, "median_offset_m": med, "p_outside": res["p_outside"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdicts", default=os.path.join(
        OUT, "review_denver-co", "verdicts.json"))
    ap.add_argument("--cities", nargs="*", default=[
        "richmond", "bend", "morgantown", "budapest_district5", "annapolis",
        "paterson", "gainesville", "clovis", "manual_gold"])
    ap.add_argument("--trials", type=int, default=200000)
    ap.add_argument("--json", default=os.path.join(OUT, "stage1_offset_tolerance.json"))
    args = ap.parse_args(argv)

    half = crop_half_angle_deg()
    with open(args.verdicts, encoding="utf-8") as fh:
        manifest = json.load(fh)
    offsets = [r["offset_m"] for r in manifest["records"]
               if r.get("offset_m") is not None and not r.get("unreadable")]
    ranges, per_city = benchmark_ranges(args.cities)

    print("Stage 1 crop geometry")
    print("  perspective render : %d px at %.0f deg FOV" % (PERSP_W, PERSP_FOV))
    print("  crop kept          : columns %d:%d (centre third)" % (CROP_LO, CROP_HI))
    print("  => azimuth accepted: +/- %.2f deg" % half)
    print("  => tangential tolerance = %.3f x range" % math.tan(math.radians(half)))
    for d in (3, 5, 10, 20, 30):
        print("       range %2d m -> %.2f m" % (d, d * math.tan(math.radians(half))))

    ranges_sorted = sorted(ranges)
    print("\nGround-truth ramp ranges (%d ramps, %d cities, flat-ground estimate)"
          % (len(ranges), len(args.cities)))
    if ranges_sorted:
        def q(p):
            return ranges_sorted[min(len(ranges_sorted) - 1,
                                     int(p * len(ranges_sorted)))]
        print("  p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  p90 %.1f m"
              % (q(.10), q(.25), q(.50), q(.75), q(.90)))

    print("\nCity offsets: %s (n=%d, median %.2f m)"
          % (manifest.get("city"), len(offsets), sorted(offsets)[len(offsets) // 2]))

    res = simulate(offsets, ranges, half, trials=args.trials)
    print("\nP(true ramp falls OUTSIDE its own crop) = %.2f%%   [%d trials]"
          % (100 * res["p_outside"], res["trials"]))
    print("  by range to the ramp:")
    for k, v in res["by_range"].items():
        print("    %-10s %6.2f%%   (n=%d)" % (k, 100 * v["rate"], v["n"]))

    scales = [1, 2, 3, 4, 6, 8, 12, 16]
    sw = sweep(ranges, half, scales, offsets)
    print("\nTolerance curve -- this city's distribution scaled up:")
    print("   %-9s %-16s %s" % ("scale", "median offset", "P(outside crop)"))
    for row in sw:
        print("   %-9s %-16s %.2f%%"
              % ("x%g" % row["scale"], "%.2f m" % row["median_offset_m"],
                 100 * row["p_outside"]))

    payload = {
        "crop_half_angle_deg": half,
        "tangential_tolerance_per_metre_of_range": math.tan(math.radians(half)),
        "city": manifest.get("city"),
        "n_offsets": len(offsets),
        "n_benchmark_ramps": len(ranges),
        "ramps_per_bundle": per_city,
        "result": res,
        "sweep": sw,
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
