"""Measure Stage 1's registration error from its own published output (issue #96).

§5f measures a city's coordinate error against *aerial imagery*, which needs a
basemap, a reviewer and a rubric. §5g converts that into a geometric tolerance,
but explicitly cannot say whether the crop model still localises a ramp near the
strip edge. **This measures the thing both of them approximate, end to end, on
the real pipeline output — with no imagery, no reviewer, no checkpoint and no
GPU.**

**The two facts that make it possible**, both read out of
``stage_one/dataset_generation/``:

1. **The government coordinates survive into the published dataset, verbatim.**
   ``generate_dataset_meta.py`` builds ``curb_ramps_coords`` as a plain 35 m
   radius query (``INCLUSION_DISTANCE_THRESHOLD``) against ``all_locations.csv``,
   and ``download_dataset.py`` copies that list into each pano's JSON untouched.
   No model is in that loop, so the denominator is not contaminated by the thing
   we are trying to measure. **The original government files are not needed.**

2. **The output labels encode a bearing.** ``perspective_to_equirectangular``
   maps equirectangular column ``u`` to ``lon = (u/(W-1))*2*pi - pi``, and that
   ``lon`` *is* the azimuth relative to the panorama heading. So a published
   point at normalised ``x`` sits at azimuth ``x*360 - 180``.

Therefore, per record::

    bearing_gov = fwd_azimuth(pano, ramp) - pano_azimuth   # where the govt said
    bearing_obs = x_norm * 360 - 180                       # where it really was
    residual    = wrap(bearing_obs - bearing_gov)

which is the registration error **in the units §5g proved Stage 1 cares about**.
The mean catches a systematic *shift*; the spread catches *imprecision*; the
match rate is the label yield, for free.

**Four caveats, which travel with every number this prints:**

* **Censored at the strip.** A ramp outside ±18.37° was never rendered into a
  crop, so it produces no point and no residual. The distribution is truncated by
  construction — always read ``matched_frac`` beside it, because the unmatched
  remainder is where the bad tail lives.
* **Matching is greedy nearest-in-bearing.** Where adjacent corners sit a few
  degrees apart (#46 found 72% of near-field misses were adjacent-pair merges),
  assignments can swap, which biases residuals *low*. **A lower bound.**
* **``peak_local_max(min_distance=40)`` merges nearby peaks**, so a low match rate
  is partly ramp density rather than coordinate error. Do not compare match rates
  across cities of different density without controlling for it.
* **It cannot see records that never reached a panorama** — no pano within the
  10 m ``DISCOVERY_DISTANCE_THRESHOLD``, or a pano dropped wholesale by the date
  filter. This is crop-model-stage yield, not end-to-end pipeline yield.

Usage::

    python scripts/analysis/stage1_bearing_residual.py                # test split
    python scripts/analysis/stage1_bearing_residual.py --shards 8     # quick look

The first run reads four small columns over HTTP range requests (a few MB, not
the 44 GB split) and caches them; afterwards it is CPU-only and offline.
"""
import argparse
import gzip
import json
import math
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

HF_DATASET = "projectsidewalk/rampnet-dataset"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_offset_tolerance import crop_half_angle_deg  # noqa: E402

#: The crop is ``persp[0:1024, 341:341+341]`` of a 1024 px, 90° FOV rendering.
#: **Imported, not re-derived** — the strip is asymmetric about the centre column
#: (341 px left, 340 px right), so an averaged 170.5 px overstates it by 0.05°,
#: and §5g's published ±18.37° is the conservative side. One definition only.
CROP_HALF_ANGLE_DEG = crop_half_angle_deg()

#: Bounding boxes for the three cities in the published corpus
#: (``docs/data_provenance.md`` §1). Anything outside is counted and reported
#: rather than silently dropped, so a future corpus cannot be misfiled in silence.
CITY_BOXES = {
    "nyc":      (40.47, 40.95, -74.30, -73.68),
    "portland": (45.42, 45.65, -122.85, -122.45),
    "bend":     (43.98, 44.15, -121.40, -121.20),
}


# --------------------------------------------------------------------------- #
# geometry -- pure, no network, all of it unit-tested
# --------------------------------------------------------------------------- #
def wrap_deg(d):
    """Wrap an angle difference into [-180, 180).

    Note the half-open end: exactly 180 maps to -180. Irrelevant for residuals,
    which are small by construction, but pinned by a test so the convention is
    not assumed to be symmetric somewhere it matters.
    """
    return (d + 180.0) % 360.0 - 180.0


def fwd_azimuth_deg(lat1, lng1, lat2, lng2):
    """Great-circle initial bearing from point 1 to point 2, in degrees.

    Stage 1 uses ``pyproj.Geod.inv``. Over the 35 m inclusion radius the
    great-circle and geodesic azimuths differ by far less than 0.01°, which is
    two orders of magnitude below the ~3° residual being measured, so this
    dependency-free form is used instead. Pinned by a test against pyproj when
    it is installed.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x))


def equirect_x_to_azimuth_deg(x_norm):
    """Normalised equirectangular column -> azimuth relative to pano heading.

    From ``perspective_to_equirectangular``::

        lon = (u_grid / (equi_width - 1)) * 2 * np.pi - np.pi

    so ``x`` in [0, 1) maps to [-180, 180). **The -180 is load-bearing**: without
    it every residual is off by exactly half a turn, which `wrap_deg` then hides
    by folding it back into range. Pinned by a test.
    """
    return wrap_deg(x_norm * 360.0 - 180.0)


def match_bearings(gov, obs, max_sep_deg):
    """Greedy nearest-in-bearing matching of government records to output points.

    Returns ``(residuals, n_matched)`` where each residual is
    ``wrap(obs - gov)``. Pairs separated by more than ``max_sep_deg`` are never
    matched, so a record whose ramp fell outside the crop contributes nothing
    rather than being force-matched to an unrelated peak.

    Greedy rather than optimal (Hungarian) on purpose: it is the same estimator
    `rampnet.metrics.greedy_match` uses for detection matching, so the two are
    comparable. Both can swap assignments between adjacent corners, which biases
    the residual *low*.
    """
    pairs = sorted(
        (abs(wrap_deg(g - o)), i, j)
        for i, g in enumerate(gov)
        for j, o in enumerate(obs)
    )
    used_g, used_o, res = set(), set(), []
    for sep, i, j in pairs:
        if sep > max_sep_deg or i in used_g or j in used_o:
            continue
        used_g.add(i)
        used_o.add(j)
        res.append(wrap_deg(obs[j] - gov[i]))
    return res, len(used_g)


def residuals_for_pano(pano_coord, pano_azimuth, ramp_coords, points_norm,
                       max_sep_deg=40.0):
    """All angular residuals for one panorama. ``ramp_coords`` are [lat, lng]."""
    plat, plng = pano_coord[0], pano_coord[1]
    gov = [wrap_deg(fwd_azimuth_deg(plat, plng, rlat, rlng) - pano_azimuth)
           for rlat, rlng in ramp_coords]
    obs = [equirect_x_to_azimuth_deg(x) for x, _y in points_norm]
    return match_bearings(gov, obs, max_sep_deg)


def city_of(lat, lng, boxes=None):
    for name, (a, b, c, d) in (boxes or CITY_BOXES).items():
        if a <= lat <= b and c <= lng <= d:
            return name
    return "other"


def summarize(residuals, n_gov, n_matched, n_panos, median_range_m=11.1):
    """Summary statistics, including the standard error the shift test needs.

    ``median_range_m`` is §5g's measured median ramp range over 6,238 benchmark
    ground-truth ramps, used only to express the mean shift in metres.
    """
    r = sorted(residuals)
    n = len(r)
    if n < 2:
        return {"n_residuals": n, "n_gov": n_gov, "n_matched": n_matched,
                "n_panos": n_panos, "insufficient": True}
    absr = sorted(abs(v) for v in r)
    mean = st.mean(r)
    sd = st.pstdev(r)
    se = sd / math.sqrt(n)
    return {
        "n_panos": n_panos,
        "n_gov": n_gov,
        "n_matched": n_matched,
        "matched_frac": round(n_matched / n_gov, 4) if n_gov else None,
        "n_residuals": n,
        "mean_deg": round(mean, 4),
        "sd_deg": round(sd, 4),
        "se_mean_deg": round(se, 4),
        "mean_over_se": round(mean / se, 2) if se else None,
        "median_deg": round(r[n // 2], 4),
        "abs_median_deg": round(absr[n // 2], 4),
        "abs_p90_deg": round(absr[int(n * 0.90)], 4),
        "abs_p99_deg": round(absr[int(n * 0.99)], 4),
        # NOT §5g's "ramp outside its own crop". A peak further than the crop
        # half-angle from the record it was matched to cannot have been produced
        # by that record's own strip -- the combined heatmap is the max over all
        # of the panorama's crops -- so this is a **cross-assignment** rate: a
        # lower bound on how often the greedy matcher paired a peak with the
        # wrong government record.
        "frac_cross_assigned": round(
            sum(1 for v in absr if v > CROP_HALF_ANGLE_DEG) / n, 5),
        "mean_shift_m_at_median_range": round(math.tan(math.radians(mean)) * median_range_m, 4),
        "abs_median_m_at_median_range": round(math.tan(math.radians(absr[n // 2])) * median_range_m, 4),
    }


# --------------------------------------------------------------------------- #
# data access -- the only part that touches the network
# --------------------------------------------------------------------------- #
def _cache_path(split):
    return os.path.join(OUT, "stage1_bearing_cache", f"{split}.jsonl.gz")


def load_records(split="test", shards=None, refresh=False):
    """Yield ``(pano_coord, pano_azimuth, ramp_coords, points_norm)`` per pano.

    Reads only the four columns needed, via HTTP range requests, then caches.
    """
    cache = _cache_path(split)
    if os.path.exists(cache) and not refresh and shards is None:
        with gzip.open(cache, "rt") as f:
            for line in f:
                yield tuple(json.loads(line))
        return

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{HF_DATASET}/{split}/*.parquet"))
    if not paths:
        raise SystemExit(f"no {split}-split parquet found for {HF_DATASET}")
    use = paths if shards is None else paths[:shards]
    print(f"reading {len(use)}/{len(paths)} {split} shards "
          f"(4 columns only, not images)...", file=sys.stderr)

    rows = []
    for k, path in enumerate(use, 1):
        tb = pq.read_table(fs.open(path), columns=[
            "curb_ramp_coords", "curb_ramp_points_normalized",
            "pano_coord", "pano_azimuth"])
        for cc, pp, pc, az in zip(tb["curb_ramp_coords"].to_pylist(),
                                  tb["curb_ramp_points_normalized"].to_pylist(),
                                  tb["pano_coord"].to_pylist(),
                                  tb["pano_azimuth"].to_pylist()):
            if not cc:          # negative panorama: no government records
                continue
            rows.append((list(pc), az, cc, pp))
        print(f"  shard {k}/{len(use)}", file=sys.stderr)

    if shards is None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with gzip.open(cache, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    yield from rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", type=int, default=None,
                    help="limit shards (skips the cache; for a quick look)")
    ap.add_argument("--max-sep", type=float, default=40.0,
                    help="never match a pair further apart than this (deg)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "stage1_bearing_residual.json"))
    args = ap.parse_args()

    per_city = {}
    nearest_sep = []
    for pano_coord, az, ramp_coords, points_norm in load_records(
            args.split, args.shards, args.refresh):
        c = city_of(pano_coord[0], pano_coord[1])
        d = per_city.setdefault(c, {"res": [], "gov": 0, "matched": 0, "panos": 0})
        res, nm = residuals_for_pano(pano_coord, az, ramp_coords, points_norm,
                                     args.max_sep)
        d["res"].extend(res)
        d["gov"] += len(ramp_coords)
        d["matched"] += nm
        d["panos"] += 1

        # convention check: unmatched nearest separation. A wrong azimuth
        # convention would make this uniform over [0, 180].
        obs = [equirect_x_to_azimuth_deg(x) for x, _y in points_norm]
        if obs:
            for rlat, rlng in ramp_coords:
                g = wrap_deg(fwd_azimuth_deg(pano_coord[0], pano_coord[1],
                                             rlat, rlng) - az)
                nearest_sep.append(min(abs(wrap_deg(g - o)) for o in obs))

    nearest_sep.sort()
    n = len(nearest_sep)
    check = {
        "n": n,
        "median_deg": round(nearest_sep[n // 2], 3) if n else None,
        "p90_deg": round(nearest_sep[int(n * 0.9)], 3) if n else None,
        "frac_inside_crop": round(
            sum(1 for s in nearest_sep if s < CROP_HALF_ANGLE_DEG) / n, 4) if n else None,
        "note": ("A wrong azimuth convention gives median ~90 deg and "
                 "frac_inside_crop ~0.10 (uniform). Anything near that "
                 "invalidates every residual below."),
    }

    result = {
        "split": args.split,
        "shards": args.shards,
        "max_sep_deg": args.max_sep,
        "crop_half_angle_deg": round(CROP_HALF_ANGLE_DEG, 4),
        "convention_check": check,
        "cities": {c: summarize(d["res"], d["gov"], d["matched"], d["panos"])
                   for c, d in sorted(per_city.items(), key=lambda t: -t[1]["gov"])},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nconvention check: n={check['n']} median={check['median_deg']}° "
          f"inside-crop={check['frac_inside_crop']} "
          f"(uniform would be ~90° / ~0.10)")
    print(f"\n{'city':10s} {'panos':>6s} {'gov':>7s} {'match':>6s} "
          f"{'mean':>8s} {'se':>7s} {'|med|':>7s} {'p90|.|':>7s}")
    for c, s in result["cities"].items():
        if s.get("insufficient"):
            print(f"{c:10s} {s['n_panos']:6d} {s['n_gov']:7d}   (too few residuals)")
            continue
        print(f"{c:10s} {s['n_panos']:6d} {s['n_gov']:7d} {s['matched_frac']:6.3f} "
              f"{s['mean_deg']:+8.3f} {s['se_mean_deg']:7.3f} "
              f"{s['abs_median_deg']:7.2f} {s['abs_p90_deg']:7.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
