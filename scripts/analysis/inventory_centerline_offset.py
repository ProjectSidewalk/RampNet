"""Are a city's ramp coordinates shifted relative to that city's OWN streets?

The attribution half of the location-precision gate (issues #96, #59). See
``docs/curb_ramp_data_sourcing.md`` §5i.

Seattle's review sheet produced a **systematic** offset — mean vector 2.06 m of a
2.37 m mean magnitude, 87% systematic, the ramp west of the published point in 9
of 11 chips. A systematic offset is a *registration* error, not coordinate
imprecision, and it must not be quoted as a precision figure. But "registration
error" does not say **which side is wrong**, and the two answers could not be
further apart:

* **The coordinates are wrong** — Seattle's inventory is genuinely misplaced, the
  paper's Poor rating stands, and its 38,364 records are unusable for Stage 1.
* **The frame is wrong** — the coordinates are right in their own datum and the
  imagery (or a reprojection) is offset. Then the error is a *constant* and can
  simply be subtracted, and Seattle's records come back into play.

The reviewer's clicks cannot separate these, because they measure ramps against
imagery and that single comparison contains both. This script measures the
**other pair**: ramps against the city's own street centrelines, no imagery and
no reviewer involved. Combined with the two existing measurements it closes a
triangle:

    (ramps vs imagery)  =  (ramps vs centrelines)  +  (centrelines vs imagery)
     review sheet, n=11        THIS SCRIPT, n=10^4     verify_chip_georeference

Each term is measured independently, so the identity is a **check**, not an
assumption — if the three do not close, one of them is wrong and the attribution
is not yet earned.

**Why the city's own centrelines are the right reference.** For Seattle both
layers come from the same publisher, the same ArcGIS organisation
(``ZOyb2t4B0UYuYNYH``) and the same native CRS (EPSG:2926, WA State Plane), and
both are reprojected to 4326 by that same server. So a datum or reprojection
fault moves the two **together** and reads as zero here, while a defect confined
to the ramp layer reads as the full shift. That is exactly the discrimination
wanted, and it is why the centreline snapshot must come from the same org as the
inventory rather than from a national basemap.

## The estimator

A ramp sits roughly half a roadway from the centreline, on one side or the
other. Writing ``r`` for the signed perpendicular offset from centreline to ramp
along a given geographic axis, ``w`` for the half-width of the right-of-way and
``d`` for the shift being looked for:

    ramps on the  east side:  median(r) ~  +w + d
    ramps on the  west side:  median(r) ~  -w + d

so **their sum is 2d and their difference is 2w**. The half-width — which varies
by street and is not known — cancels out of the shift, and reappears as a free
sanity check: ``w`` must come out at a plausible roadway scale, and must agree
between the two axes and across cities.

Two design choices keep the estimate honest:

* **Only near-cardinal segments count**, and each is assigned to the single axis
  its perpendicular actually measures. A diagonal street constrains a diagonal
  direction; pooling it into "east" would import its north error. This is the
  same trap ``verify_chip_georeference.py`` documents, where pooling both axes in
  a grid city fills each median with structural zeros.
* **The nearest segment is chosen per axis, not overall.** Choosing the single
  nearest street biases the result toward zero: an eastward shift lengthens the
  distance to north-south streets only, so east-side ramps would preferentially
  be reassigned to the east-west street and drop out of the axis that can see the
  shift. Selecting within an axis is unaffected, because a 2 m shift never
  changes which parallel street a block away is nearest.

    python scripts/analysis/inventory_centerline_offset.py \\
        --city seattle-wa \\
        --inventory data/inventories/seattle-wa-2026-07-31.jsonl.gz \\
        --centerlines data/inventories/seattle-wa-centerlines-2026-07-31.jsonl.gz

**Run the control first.** Denver's reviewer-measured offset is 0.10 m resultant
— random, cancelling the way error must — so this test must read approximately
zero on Denver. A version of it that did not was wrong, and a Seattle number
produced without that check would not be worth reporting.

Needs no GPU, no imagery and no network: it reads two snapshots written by
``fetch_inventory.py``. The core is pure and unit-tested in
``tests/test_inventory_centerline_offset.py``; only ``load_*``, ``write_report``
and ``main`` touch disk.
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

METRES_PER_DEG_LAT = 111320.0

# How far from cardinal a segment may run and still be read as measuring one
# axis. At 20 degrees the cross-axis leak into a sample is sin(20) = 34% of a
# quantity that is itself the small residual being measured, and it is unsigned
# with respect to the axis in question, so it averages out rather than biasing.
# Reported in the output so a reader can see it was a choice; ``--max-dev-deg``
# sweeps it.
MAX_DEV_DEG = 20.0

# A ramp further than this from a centreline is not on that street. 25 m clears
# a half-width even on a wide arterial with a service road.
MAX_DIST_M = 25.0

# Plausibility band for the recovered half-width. Outside it the two clusters
# are not "the two sides of a street" and the shift they bracket means nothing.
HALF_WIDTH_BAND_M = (3.0, 20.0)

BOOTSTRAP_N = 400


def to_local_metres(points, lat0=None):
    """Project lon/lat to a local flat plane in metres, x east and y north.

    Equirectangular about the mean latitude, matching ``inventory_geometry`` —
    good to well under a metre over a city, which is an order of magnitude finer
    than anything decided here.
    """
    if not points:
        return [], 0.0
    if lat0 is None:
        lat0 = sum(p[1] for p in points) / len(points)
    mx = METRES_PER_DEG_LAT * math.cos(math.radians(lat0))
    return [(p[0] * mx, p[1] * METRES_PER_DEG_LAT) for p in points], lat0


def segments_from_paths(paths, lat0):
    """Flatten ArcGIS ``paths`` (lon/lat) into segments in local metres.

    Degenerate segments — repeated vertices — carry no direction and are dropped
    rather than defaulting to an arbitrary one.
    """
    out = []
    for path in paths:
        pts, _ = to_local_metres([(p[0], p[1]) for p in path], lat0=lat0)
        for a, b in zip(pts, pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6:
                out.append((a, b))
    return out


def segment_axis(a, b, max_dev_deg=MAX_DEV_DEG):
    """Which geographic axis this segment's perpendicular measures.

    Returns ``"east"`` for a near north-south segment (its normal points
    east-west, so the offset it measures is an east one), ``"north"`` for a near
    east-west segment, and ``None`` for anything more than ``max_dev_deg`` from
    either cardinal — a diagonal constrains a diagonal, and reading it as
    cardinal would import the other axis's error.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L <= 0:
        return None
    tol = math.sin(math.radians(max_dev_deg))
    if abs(dx) / L <= tol:          # runs north-south
        return "east"
    if abs(dy) / L <= tol:          # runs east-west
        return "north"
    return None


def perpendicular_offset(p, a, b):
    """Signed perpendicular offset from segment ``a->b`` to point ``p``.

    Returns ``(east, north, distance)`` of the vector from the foot of the
    perpendicular to the point, or ``None`` when the foot falls outside the
    segment. Rejecting the ends matters: a ramp beyond a segment's end is at an
    intersection, where the closest point is a vertex and the "perpendicular"
    offset is no longer perpendicular to anything.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return None
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    if t <= 0.0 or t >= 1.0:
        return None
    fx, fy = a[0] + t * dx, a[1] + t * dy
    ex, ny = p[0] - fx, p[1] - fy
    return ex, ny, math.hypot(ex, ny)


class SegmentIndex:
    """Uniform-grid spatial hash over segments, bucketed by measured axis.

    Kept per axis because the nearest segment is chosen **within** an axis, not
    overall — see the module docstring for why choosing the overall nearest
    attenuates the very shift this measures.
    """

    def __init__(self, segments, cell=MAX_DIST_M * 2, max_dev_deg=MAX_DEV_DEG):
        self.cell = float(cell)
        self.axes = {"east": [], "north": []}
        self.cells = {"east": defaultdict(list), "north": defaultdict(list)}
        for a, b in segments:
            axis = segment_axis(a, b, max_dev_deg)
            if axis is None:
                continue
            i = len(self.axes[axis])
            self.axes[axis].append((a, b))
            x0, x1 = sorted((a[0], b[0]))
            y0, y1 = sorted((a[1], b[1]))
            for cx in range(int(math.floor(x0 / self.cell)), int(math.floor(x1 / self.cell)) + 1):
                for cy in range(int(math.floor(y0 / self.cell)),
                                int(math.floor(y1 / self.cell)) + 1):
                    self.cells[axis][(cx, cy)].append(i)

    def nearest(self, p, axis, max_dist=MAX_DIST_M):
        """Offset from the nearest qualifying segment on ``axis``, or None."""
        cx, cy = int(math.floor(p[0] / self.cell)), int(math.floor(p[1] / self.cell))
        best = None
        seen = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.cells[axis].get((cx + dx, cy + dy), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    a, b = self.axes[axis][i]
                    off = perpendicular_offset(p, a, b)
                    if off is None or off[2] > max_dist:
                        continue
                    if best is None or off[2] < best[2]:
                        best = off
        return best


def collect_samples(ramp_xy, index, max_dist=MAX_DIST_M):
    """Signed offsets per axis: ``{"east": [...], "north": [...]}`` in metres.

    A ramp contributes at most one sample to each axis. Positive east means the
    ramp lies east of the centreline; positive north, north of it.
    """
    samples = {"east": [], "north": []}
    for p in ramp_xy:
        for axis, comp in (("east", 0), ("north", 1)):
            off = index.nearest(p, axis, max_dist=max_dist)
            if off is not None:
                samples[axis].append(off[comp])
    return samples


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _quantile(vals, q):
    s = sorted(vals)
    if not s:
        return None
    return s[min(len(s) - 1, int(q * len(s)))]


def axis_shift(values):
    """Split one axis's offsets by side and recover ``(shift, half_width)``.

    ``shift = (median_positive + median_negative) / 2`` — the roadway half-width
    cancels. ``half_width = (median_positive - median_negative) / 2`` does not
    cancel, and is returned because it is the check that the two clusters really
    are the two sides of a street.
    """
    pos = [v for v in values if v > 0]
    neg = [v for v in values if v < 0]
    if not pos or not neg:
        return None
    mp, mn = _median(pos), _median(neg)
    return {
        "n": len(values), "n_pos": len(pos), "n_neg": len(neg),
        "median_pos_m": mp, "median_neg_m": mn,
        "shift_m": 0.5 * (mp + mn),
        "half_width_m": 0.5 * (mp - mn),
        "iqr_pos_m": [_quantile(pos, 0.25), _quantile(pos, 0.75)],
        "iqr_neg_m": [_quantile(neg, 0.25), _quantile(neg, 0.75)],
    }


def bootstrap_shift(values, n=BOOTSTRAP_N, seed=0):
    """Percentile CI for one axis's shift. Resamples offsets, not ramps."""
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    out = []
    for _ in range(n):
        draw = [values[rng.randrange(k)] for _ in range(k)]
        a = axis_shift(draw)
        if a:
            out.append(a["shift_m"])
    if not out:
        return None
    out.sort()
    return [out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]]


def analyse(samples, seed=0, bootstrap=BOOTSTRAP_N):
    """Per-axis shifts plus the resultant, with the half-width sanity verdict."""
    axes = {}
    for axis in ("east", "north"):
        a = axis_shift(samples[axis])
        if a is not None and bootstrap:
            a["shift_ci95_m"] = bootstrap_shift(samples[axis], n=bootstrap, seed=seed)
        axes[axis] = a
    de = axes["east"]["shift_m"] if axes["east"] else 0.0
    dn = axes["north"]["shift_m"] if axes["north"] else 0.0
    widths = [a["half_width_m"] for a in axes.values() if a]
    lo, hi = HALF_WIDTH_BAND_M
    ok = bool(widths) and all(lo <= w <= hi for w in widths)
    return {
        "axes": axes,
        "shift_east_m": de,
        "shift_north_m": dn,
        "resultant_m": math.hypot(de, dn),
        "bearing_deg": (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0,
        "half_width_plausible": ok,
        "half_width_band_m": list(HALF_WIDTH_BAND_M),
        "sign_convention": "Positive east means the PUBLISHED RAMP POINT lies east "
                           "of where the centreline geometry puts the street, i.e. "
                           "the coordinate is east of the true ramp. This is the "
                           "same sense the review sheet reports, so the two are "
                           "directly comparable and must not be negated.",
    }


def load_points(path, lon_field="lon", lat_field="lat"):
    opener = gzip.open if path.endswith(".gz") else open
    pts = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get(lon_field) is not None and r.get(lat_field) is not None:
                    pts.append((r[lon_field], r[lat_field]))
    return pts


def load_paths(path):
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.extend(json.loads(line).get("paths") or [])
    return out


def write_report(city, report, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "centerline_offset_{}.json".format(city))
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--inventory", required=True, help="ramp snapshot (jsonl.gz)")
    ap.add_argument("--centerlines", required=True, help="polyline snapshot (jsonl.gz)")
    ap.add_argument("--max-dist-m", type=float, default=MAX_DIST_M)
    ap.add_argument("--max-dev-deg", type=float, default=MAX_DEV_DEG,
                    help="how far from cardinal a segment may run and still count")
    ap.add_argument("--sweep", action="store_true",
                    help="repeat over a range of --max-dev-deg, so the headline "
                         "number can be seen not to depend on the cutoff")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    ramps_ll = load_points(args.inventory)
    paths = load_paths(args.centerlines)
    print("{}: {} ramps, {} centreline paths".format(
        args.city, len(ramps_ll), len(paths)))
    if not ramps_ll or not paths:
        raise SystemExit("nothing to measure — check the snapshot paths")

    ramp_xy, lat0 = to_local_metres(ramps_ll)
    segments = segments_from_paths(paths, lat0)
    print("  {} segments, local plane about lat {:.4f}".format(len(segments), lat0))

    index = SegmentIndex(segments, max_dev_deg=args.max_dev_deg)
    print("  near-cardinal: {} north-south, {} east-west (within {:.0f} deg)".format(
        len(index.axes["east"]), len(index.axes["north"]), args.max_dev_deg))

    samples = collect_samples(ramp_xy, index, max_dist=args.max_dist_m)
    result = analyse(samples, seed=args.seed, bootstrap=args.bootstrap)

    print("\nRAMPS vs THE CITY'S OWN CENTRELINES")
    print("  {:>6} {:>7} {:>7} {:>12} {:>12} {:>10} {:>9}".format(
        "axis", "n+", "n-", "median +", "median -", "half-width", "shift"))
    for axis in ("east", "north"):
        a = result["axes"][axis]
        if not a:
            print("  {:>6}  no usable samples".format(axis))
            continue
        print("  {:>6} {:>7} {:>7} {:>10.2f} m {:>10.2f} m {:>8.2f} m {:>+7.2f} m".format(
            axis, a["n_pos"], a["n_neg"], a["median_pos_m"], a["median_neg_m"],
            a["half_width_m"], a["shift_m"]))
        if a.get("shift_ci95_m"):
            print("  {:>6}   95% CI [{:+.2f}, {:+.2f}] m".format(
                "", a["shift_ci95_m"][0], a["shift_ci95_m"][1]))
    print("  resultant {:.2f} m  (east {:+.2f}, north {:+.2f})".format(
        result["resultant_m"], result["shift_east_m"], result["shift_north_m"]))
    print("  half-width plausible: {}".format(
        "yes" if result["half_width_plausible"] else "NO — the clusters are not "
        "two sides of a street; the shift they bracket means nothing"))

    report = {
        "city": args.city,
        "inventory": os.path.basename(args.inventory),
        "centerlines": os.path.basename(args.centerlines),
        "ramps": len(ramps_ll),
        "centerline_paths": len(paths),
        "segments": len(segments),
        "near_cardinal": {"north_south": len(index.axes["east"]),
                          "east_west": len(index.axes["north"])},
        "max_dist_m": args.max_dist_m,
        "max_dev_deg": args.max_dev_deg,
        "result": result,
        "interpretation":
            "Measures the ramp coordinates against the SAME publisher's street "
            "geometry, so it is blind to any error the two layers share. A shift "
            "near zero means the coordinates are consistent with their own city's "
            "road network: any offset the reviewer saw against imagery is then in "
            "the frame (datum, reprojection or basemap registration) and is a "
            "constant that can be subtracted. A shift matching the reviewer's "
            "means the defect is in the ramp layer itself and no constant fixes it.",
        "limits":
            "A centreline is a cartographic construct, not a survey of the "
            "pavement midline, and this estimator assumes only that it is "
            "unbiased BETWEEN the two sides of the street -- it never uses its "
            "absolute position, which is what makes it robust to that. It cannot "
            "see an error the ramp and centreline layers share, by construction; "
            "that is what verify_chip_georeference.py is for. Cluster overlap "
            "attenuates the estimate toward zero when the half-width is small "
            "relative to the spread, so a near-zero reading is weaker evidence "
            "than a large one.",
    }
    if args.sweep:
        report["sweep"] = []
        for dev in (10.0, 15.0, 20.0, 25.0, 30.0):
            idx = SegmentIndex(segments, max_dev_deg=dev)
            s = collect_samples(ramp_xy, idx, max_dist=args.max_dist_m)
            r = analyse(s, seed=args.seed, bootstrap=0)
            report["sweep"].append({
                "max_dev_deg": dev, "shift_east_m": r["shift_east_m"],
                "shift_north_m": r["shift_north_m"], "resultant_m": r["resultant_m"],
                "n_east": r["axes"]["east"]["n"] if r["axes"]["east"] else 0,
                "n_north": r["axes"]["north"]["n"] if r["axes"]["north"] else 0,
            })
        print("\nSWEEP over the cardinal cutoff")
        for row in report["sweep"]:
            print("  {:>4.0f} deg: east {:+.2f}  north {:+.2f}  resultant {:.2f} m"
                  "  (n {} / {})".format(row["max_dev_deg"], row["shift_east_m"],
                                         row["shift_north_m"], row["resultant_m"],
                                         row["n_east"], row["n_north"]))

    path = write_report(args.city, report, out_dir=args.out_dir)
    print("\nwrote {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
