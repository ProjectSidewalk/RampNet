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

* ``act`` — the max heatmap value within the match radius of the missed ramp. The
  grid *is* the matcher's scaled space (512x1024), and the radius is the matcher's,
  **with one deliberate divergence: this window wraps at the 360-degree seam and
  the matcher's does not** (``greedy_match`` takes a plain x difference). Wrapping
  is the physically right geometry for an equirectangular panorama — the matcher is
  the approximation here — so the divergence is flagged per row (``seam``) rather
  than removed. 9 of the 128 pooled misses are seam rows; none of them changes
  bucket, but that is a property of this population, not a guarantee.
  Note ``act`` can exceed the 0.05 floor without contradicting ``silent``
  — a shoulder of a neighbouring peak is not a local maximum; those cases are
  counted separately rather than silently pooled.
* a per-miss **null**: the same radius-max at ``NULL_TRIALS`` random azimuths in the
  same panorama at the same elevation — the same null shape every #46 analysis uses,
  because both ramps and heatmap mass crowd the horizon band. ``act`` is reported as
  a percentile of its own panorama's null, so "there is signal here" is a claim
  against that pano's own distribution, not against zero.

  **Read the percentile, not the p95 flag.** With a 22.5 px radius on a 1024-wide
  grid there are only ~23 non-overlapping windows per elevation band, so the p95 of
  200 draws is effectively the band *maximum* — pooled, the median ``null_p95`` is
  0.595 against a median ``null_med`` of 0.003. ``above_own_null_p95`` therefore
  asks "is this site the strongest thing on its horizon row", which almost nothing
  sub-floor can pass (2 of 39 faint-local sites do). It is conservative twice over,
  because a draw may also land on *another GT ramp* in the same band — only the
  site's own 2 R zone is excluded. ``null_pct`` is the usable statistic and is what
  the decomposition below reports.

Model: the published ``projectsidewalk/rampnet-model`` weights — the same checkpoint
every committed cache came from (``operating_point_curve.py`` extract). Single-pass,
no TTA, fp32, matching ``analysis_out/op_cache``.

**Inputs, for someone who is not on this machine.** Unlike Phase 0 this one needs
pixels: the native-resolution panoramas at ``benchmark/<city>/panos/<pano>.jpg``,
which are git-ignored and published as the Hugging Face dataset
``projectsidewalk/rampnet-benchmark`` (the same bundle #94's imagery manifests pin by
content hash). ``--panos-root`` points at whichever checkout holds them, since a
worktree will not. Everything else — the caches, the witness list, the gallery
manifest and verdicts — is committed. A GPU-ish machine; the RTX 3070 does ~10 s/pano,
about 20 minutes for the 128 pooled misses.

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
from miss_decomposition import DEFAULT_THRESHOLD  # noqa: E402

# A FROZEN forensic study, like farfield_forensics: tests/test_silent_activation.py
# pins the committed analysis_out/silent_activation.json to the numbers the issue's
# section 0c quotes. It therefore carries its own split tuple rather than importing the
# live `US_SPLITS`, so registering a new city cannot silently restate a published
# finding. laurens is absent for that reason alone -- it is pooled everywhere else.
# To fold a split in: re-run the study, re-quote it, update the 0c expectations.
US_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
             "gainesville")
from farfield_forensics import (  # noqa: E402
    DEFAULT_RATER, load_rated, quartiles, row_key)
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)

NULL_TRIALS = 200
NULL_SEED = 20260731

# The decomposition's two cutoffs, named because the write-up quotes them.
# PEAK_FLOOR is the extractor's own score floor (the caches hold peaks >= 0.05), so
# "act >= PEAK_FLOOR on a silent miss" is definitionally an outside mode's reach.
# ABSENT_MAX is the flat-heatmap bar: below it there is no response to attenuate.
ABSENT_MAX = 0.01
PEAK_FLOOR = 0.05

# How wide a zone around the site the null refuses to draw from, in match radii.
# 2 R is the first distance at which a draw's window cannot touch the site's.
NULL_EXCLUDE_RADII = 2.0

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


def null_azimuths(x, rng, trials, exclude):
    """``trials`` random azimuths, none within ``exclude`` of ``x`` on the wrapped axis.

    Split out because the rejection is the whole point of the null and was
    otherwise untestable: a draw whose window overlaps the site's own would read
    the site's bump back as "chance", which would flatten exactly the sparse-heatmap
    case this analysis exists to detect. Rejection resampling, so the RNG is
    consumed one draw at a time and the stream is unchanged by this extraction —
    the committed artifact stays reproducible.
    """
    xs = []
    while len(xs) < trials:
        nx = rng.random()
        dx = abs(nx - x)
        if min(dx, 1.0 - dx) < exclude:
            continue
        xs.append(nx)
    return xs


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
    exclude = NULL_EXCLUDE_RADII * (radius_sq ** 0.5) / W  # normalized column dist
    draws = sorted(radius_max(heat, nx, y, radius_sq)
                   for nx in null_azimuths(x, rng, trials, exclude))
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


def seam_of(x, radius_sq=None):
    """Does this site's match window straddle the 360-degree seam?

    ``site_profile`` and ``nearest_peak`` wrap columns; ``greedy_match``, which
    produced the ``silent`` label, does not. For a seam row the two therefore read
    different windows, and roughly half of what this script scans is imagery the
    matcher never considered. Wrapping is the correct geometry for an
    equirectangular panorama, so the flag records the divergence instead of
    removing it — and makes "did any of these change bucket?" a query rather than
    a re-derivation.
    """
    if radius_sq is None:
        radius_sq = radius_sq_for()
    return min(x, 1.0 - x) * PANO_SCALE_X < radius_sq ** 0.5


def class_of(act, absent_max=ABSENT_MAX, floor=PEAK_FLOOR):
    """Which activation class a miss falls in — the decomposition's only cutoffs.

    ``absent`` the heatmap is flat at the site; ``tail`` the in-window mass is at or
    above the peak floor, which for a *silent* miss can only be an outside mode
    reaching in; ``faint_local`` everything between. Split out so the classes are
    testable and so the printed table, the JSON and the write-up cannot drift apart.
    """
    if act < absent_max:
        return "absent"
    if act >= floor:
        return "tail"
    return "faint_local"


def build_payload(results, threshold, cities, n_panos, skipped):
    """The result JSON, built in one place so a re-run is byte-comparable.

    Everything that scopes the run travels with it — ``cities`` and ``panos``
    because a subset run is not the pooled population the write-up quotes, and a
    file that does not say so is indistinguishable from one that is complete.
    """
    return {"threshold": threshold, "null_trials": NULL_TRIALS,
            "null_seed": NULL_SEED, "n": len(results),
            "cities": list(cities), "panos": n_panos,
            "skipped_no_imagery": skipped,
            "model": "projectsidewalk/rampnet-model",
            "tta": False, "results": results}


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
                   help="Stop after this many panos (smoke test). Refused with "
                        "--json-out: a truncated run must not be written as a result.")
    p.add_argument("--rater", default=DEFAULT_RATER,
                   help="Which reviewer pass to read (silent__<rater>.json).")
    p.add_argument("--json-out", default=None)
    p.add_argument("--allow-partial", action="store_true",
                   help="Permit --json-out on a city subset. The scope is recorded "
                        "in the payload either way; this only silences the refusal.")
    args = p.parse_args(argv)

    # A result file that looks complete and is not is worse than no result file.
    # analysis_out/silent_activation.json is a committed artifact and every number
    # in the write-up's 0c derives from it, so the smoke-test flags are refused
    # here rather than trusted to be remembered.
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if args.json_out and args.limit:
        p.error("--limit truncates the run; refusing to write it to --json-out")
    if args.json_out and sorted(cities) != sorted(US_SPLITS) and not args.allow_partial:
        p.error(f"--cities is a subset ({len(cities)} of {len(US_SPLITS)}); the "
                f"pooled population is what 0c quotes. Pass --allow-partial if a "
                f"scoped result file is really what you want.")

    import torch
    import threshold_sweep as ts
    assert (HEAT_W, HEAT_H) == (1024, 512), "heatmap grid != matcher space"

    from miss_gallery import load_queue, pano_path
    queue_keys = load_queue(args.witness)
    rated = load_rated(args.gallery, field=None, rater=args.rater)
    rated_by_rowkey = {(v["city"], v["pano"], round(float(v["x"]), 6),
                        round(float(v["y"]), 6)): v for v in rated.values()}

    from operating_point_curve import CACHE_DIR, read_cache
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
                "seam": seam_of(r["x"], radius_sq),
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
        cls[class_of(r["act"])].append(r)
    print(f"  {'class':>12} {'n':>4} {'vis':>4} {'argmax off':>12} "
          f"{'nearest peak':>17} {'null pct q1/med/q3':>21} {'>p95':>8}")
    for name, sel in cls.items():
        if not sel:
            continue
        med_off = quartiles([r["argmax_off_px"] for r in sel])[1]
        npks = [r["nearest_peak_px"] for r in sel if r["nearest_peak_px"]]
        med_npk = quartiles(npks)[1] if npks else float("nan")
        n_vis = sum(1 for r in sel if r["verdict"] == "visible")
        nq = quartiles([r["null_pct"] for r in sel])
        n_p95 = sum(1 for r in sel if r["above_own_null_p95"])
        print(f"  {name:>12} {len(sel):>4} {n_vis:>4} "
              f"{med_off:>9.1f} px {med_npk:>8.1f} px ({med_npk/r_px:>3.1f}R) "
              f"{nq[0]:>7.3f}/{nq[1]:.3f}/{nq[2]:.3f} {n_p95:>4}/{len(sel):<3}")
    tail_near_edge = sum(1 for r in cls['tail']
                         if r['argmax_off_px'] > 0.75 * r_px)
    print(f"  tail cases with argmax in the window's outer quarter: "
          f"{tail_near_edge}/{len(cls['tail'])} — the mass is entering from "
          f"outside, not centred on the ramp")
    # The seam divergence, surfaced rather than assumed away: this window wraps at
    # x=0/1 and the matcher's does not, so a seam row's `silent` label and its `act`
    # were read from different windows. The second count is the one that would
    # matter — a floor peak inside the radius means a wrapping matcher would have
    # called the miss `merged`, not `silent`.
    n_seam = sum(1 for r in results if r["seam"])
    seam_flip = [r for r in results if r["seam"] and r["nearest_peak_px"]
                 and r["nearest_peak_px"] < r_px]
    print(f"  seam rows (window straddles x=0/1, where this window wraps and the "
          f"matcher's\n  does not): {n_seam}/{len(results)}; of those "
          f"{len(seam_flip)} hold a floor peak inside the\n  radius, i.e. would "
          f"not be `silent` under a wrapping matcher")

    print(f"\n  Reading: 'absent' = the heatmap is genuinely flat at the site.")
    print(f"  'faint_local' = a real sub-floor response at the site itself.")
    print(f"  'tail' = a neighbouring supra-floor mode's slope reaches the window —")
    print(f"  the site contributed no mode of its own, but the model responded to")
    print(f"  something adjacent (cf. the merged bucket's sigma story). The three")
    print(f"  continue differently: absent -> Phase 2's scale counterfactual;")
    print(f"  faint_local -> threshold/calibration (the sub_threshold continuum);")
    print(f"  tail -> representation (sigma), not vocabulary.")
    print(f"\n  The null column is the check on that reading, and it separates the")
    print(f"  three cleanly: 'absent' sits at chance by construction, 'faint_local'")
    print(f"  well above it, 'tail' higher again. Read null_pct, NOT the >p95 count:")
    print(f"  p95 over ~23 non-overlapping windows per band is the band maximum, so")
    print(f"  it asks whether the site is the strongest thing on its horizon row.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        # newline="" so the artifact is LF on every platform. Python's text
        # mode would write CRLF here on Windows, which makes a regenerated
        # copy fail a byte comparison against the committed one even when
        # every number is identical — and a content hash that only holds on
        # one OS is not a content hash.
        with open(args.json_out, "w", encoding="utf-8", newline="") as fh:
            json.dump(build_payload(results, args.threshold, cities,
                                    len(by_pano), skipped), fh, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
