"""Cross-split low-floor operating-point sweep (issue #54, with #55 riding on it).

``operating_point_curve.py`` produces the low-floor extraction and the per-city PR
curve. This module is the **cross-split** layer on top of that same cache: it sweeps
every benchmark split on one threshold grid, pools splits by *imagery tier* rather
than by city, and reports the things a deployment recommendation needs that a
per-city PR curve does not give you:

1. **Detections per pano** at every threshold. A recall gain is only worth having if
   the detector stays sparse. ``docs/model_comparison.md``'s own null-recall finding
   is that an open detector's recall is largely *density* (OWLv2 at 55-88 boxes/pano),
   so reporting boxes/pano at each candidate operating point forecloses that
   objection against RampNet instead of inviting it.
2. **Per-tier curves.** Precision tracks the camera across the US Mapillary splits
   (``benchmark/README.md``), so one uniform threshold is the wrong shape of answer.
   Tiers are assigned **per pano** from ``camera_make``/``camera_model`` in
   ``records.jsonl``, not per split — richmond alone mixes iSTAR Pulsar and GoPro Max,
   so split-level grouping would smear the very effect being measured.
3. **A confidence calibration table** — empirical P(real | confidence bin) with Wilson
   intervals. That is what a multi-view promotion floor actually selects on
   (sidewalk-auto-labeler#27 stage 4).

Everything here is CPU-only and reads the cache ``operating_point_curve.py extract``
writes, so every number re-derives without a GPU.

**Two splits are swept but held out of the pooled/tier rows** (each overridable):

- ``budapest_district5`` — single-rater GT at low reviewer confidence; the merged
  ``docs/model_comparison.md`` says not to pool it into recommendations. It is still
  swept per split, because an out-of-distribution split is exactly where a threshold
  recommendation is most likely to break.
- ``manual_gold`` — in-distribution GSV from the training cities, and its GT is
  independent manual labeling rather than a RampNet review. Pooling it with the
  deployment cities would mix a train-distribution reference into a deployment
  recommendation; it is reported alongside as the in-domain reference it is.

Run order (after ``operating_point_curve.py extract``):

    python scripts/analysis/low_floor_sweep.py parity     # gate: cache reproduces records.jsonl
    python scripts/analysis/low_floor_sweep.py sweep      # per-split + pooled + per-tier curves
    python scripts/analysis/low_floor_sweep.py hist       # confidence calibration for labeler#27
    python scripts/analysis/low_floor_sweep.py distance   # where the recall gain lands
"""
import argparse
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, aggregate, radius_sq_for, score_pano)
from rampnet.metrics import greedy_match  # noqa: E402
from rampnet.validation import wilson_interval  # noqa: E402

from operating_point_curve import (  # noqa: E402
    DEPLOYED_THRESHOLD, classify_predictions, read_cache)

# The five US/VA city splits carry verdict-grade GT and are the recommendation's basis.
US_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis")
CITY_SPLITS = US_SPLITS + ("budapest_district5",)
ALL_SPLITS = CITY_SPLITS + ("manual_gold",)

# Why a split is swept but not pooled. Printed with the results so an omission can
# never be mistaken for a withheld result.
HELD_OUT = {
    "budapest_district5": "single-rater GT at low reviewer confidence "
                          "(docs/model_comparison.md: do not pool)",
    "manual_gold": "in-distribution GSV + independently-labelled GT "
                   "(in-domain reference, not a deployment city)",
}

# Splits whose records.jsonl predates the camera-provenance fields but whose imagery
# source is known from the bundle. Everything else reads its provenance per pano.
SPLIT_IMAGERY_FALLBACK = {"bend": "gsv", "manual_gold": "gsv"}

# manual_gold's committed detections were exported WITH horizontal-flip TTA and at a
# 0.05 floor (benchmark/manual_gold/detections_meta.json), unlike the city splits
# (no TTA, 0.55 floor). A 1:1 parity check against them is therefore the wrong test —
# it would be measuring TTA, not preprocessing drift. See cmd_parity.
TTA_RECORD_SPLITS = {"manual_gold"}

CACHE_DIR = os.path.join(OUT, "op_cache")

# Flat-ground distance estimate, identical to scripts/analysis/precision_by_distance.py
# (validated there against Depth-Anything-3 depth). The camera height cancels out of
# every *comparison* — it rescales all distances by one factor — so it sets the metre
# labels, not the conclusions. See benchmark/README.md's annapolis section.
CAM_H = 2.5


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_low_floor_sweep.py
# --------------------------------------------------------------------------- #
def threshold_grid(floor=0.05, top=0.90, fine_lo=0.20, fine_hi=0.50,
                   coarse_step=0.05, fine_step=0.01):
    """Sweep grid: coarse everywhere, fine through the candidate band.

    The recommendation lives in 0.20-0.50, so that band is swept at 0.01 while the
    tails stay at 0.05 — a uniform fine grid would quadruple the rows to add
    resolution where nothing is being decided. Values are rounded to 4 dp so the
    grid is exact and dedupes cleanly at the band edges.
    """
    if floor > top:
        return []
    vals = set()

    def add_range(lo, hi, step):
        if hi < lo:
            return
        n = int(round((hi - lo) / step))
        for i in range(n + 1):
            v = round(lo + i * step, 4)
            if floor - 1e-9 <= v <= top + 1e-9:
                vals.add(v)

    add_range(floor, top, coarse_step)
    add_range(max(floor, fine_lo), min(top, fine_hi), fine_step)
    return sorted(vals)


def tier_of(camera_make, camera_model, source):
    """Imagery tier for one pano, from its camera provenance.

    Grouping is by *rig class*, which is what the precision differences in
    ``benchmark/README.md`` actually track — not by city and not by vendor. A pano
    whose provenance is absent is ``unknown`` rather than being folded into a
    neighbouring tier; unknowns are reported, never quietly pooled.
    """
    src = (source or "").strip().lower()
    if src in ("launch", "gsv", "google"):
        return "gsv"
    make = (camera_make or "").strip().lower()
    model = (camera_model or "").strip().lower()
    if make in ("", "none") and model in ("", "none"):
        return "unknown"
    if "trimble" in make or "trimble" in model:
        return "survey"          # Trimble MX7, vehicle-mounted survey rig
    if "nctech" in make or "istar" in model:
        return "pro360"          # NCTECH iSTAR Pulsar
    if "fusion" in model:
        return "action-legacy"   # GoPro Fusion, 2018-era
    if "max" in model:
        return "action-modern"   # GoPro Max, 2024-era
    return "unknown"


TIER_LABEL = {
    "survey": "survey-grade (Trimble MX7)",
    "pro360": "pro 360 (NCTECH iSTAR Pulsar)",
    "action-modern": "action cam, modern (GoPro Max)",
    "action-legacy": "action cam, legacy (GoPro Fusion 2018)",
    "gsv": "Google Street View",
    "unknown": "provenance absent",
}


def sweep_rows(panos, grid, radius_sq):
    """P/R/F1 + density at each threshold over a pano list.

    ``dets_per_pano`` counts every kept prediction — TP, FP *and* the ones the
    reviewer's `unsure` marks make the scorer ignore. Ignored detections are
    invisible to P/R but a human still has to look at them, so leaving them out
    would understate exactly the review burden this column exists to measure.
    """
    rows = []
    n_panos = len(panos) or 1
    for thr in grid:
        scores, kept = [], 0
        for pd in panos:
            preds = [p for p in pd["preds"] if p[2] >= thr]
            kept += len(preds)
            scores.append(score_pano(preds, pd["gt"], radius_sq=radius_sq))
        rep = aggregate(scores)
        rows.append({
            "threshold": thr,
            "precision": rep.precision, "recall": rep.recall, "f1": rep.f1,
            "precision_lo": rep.precision_ci[0], "precision_hi": rep.precision_ci[1],
            "recall_lo": rep.recall_ci[0], "recall_hi": rep.recall_ci[1],
            "tp": rep.tp, "fp": rep.fp, "fn": rep.fn, "ignored": rep.ignored,
            "dets_per_pano": kept / n_panos,
            "n_panos": len(panos),
        })
    return rows


def best_f1_row(rows):
    """F1-optimal row; ties break toward the *higher* threshold.

    A tie broken downward would silently recommend more false positives for no F1
    gain, which is the opposite of what a tie means.
    """
    return max(rows, key=lambda r: (r["f1"], r["threshold"]))


def row_at(rows, threshold):
    """The swept row nearest a threshold (the grid may not contain it exactly)."""
    return min(rows, key=lambda r: abs(r["threshold"] - threshold))


def highest_threshold_meeting(rows, min_precision):
    """Lowest threshold whose precision still clears ``min_precision``.

    Recall-first: among the operating points that satisfy a precision floor, the
    best one is the *lowest*, because recall is monotonically non-increasing in the
    threshold. Returns None when no swept point clears the floor — a real answer
    ("this split cannot meet that bar at any threshold"), not an error.
    """
    ok = [r for r in rows if r["precision"] >= min_precision]
    return min(ok, key=lambda r: r["threshold"]) if ok else None


def confidence_calibration(panos, radius_sq, bin_edges):
    """Empirical P(real | confidence bin) — the GT-true vs GT-false histogram.

    This is the artifact sidewalk-auto-labeler#27 stage 4 consumes: a multi-view
    promotion floor chooses a confidence at which a single-view detection is
    trustworthy enough to promote, and that decision needs the *observed* hit rate
    per bin, not the score distribution.

    ``ignore``-outcome predictions (inside an `unsure` mark) are excluded from both
    counts, mirroring the scorer — the reviewer could not call them, so they can
    neither confirm nor refute a bin. Wilson intervals come along because the top
    bins are thin and a bare ratio there invites over-reading.
    """
    bins = [{"lo": lo, "hi": hi, "n_true": 0, "n_false": 0}
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    top = bin_edges[-1]
    for pd in panos:
        for _x, _y, score, outcome, _redundant in classify_predictions(
                pd["preds"], pd["gt"], radius_sq):
            if outcome == "ignore":
                continue
            for b in bins:
                # the last bin is closed on the right so a score of exactly 1.0 lands
                if b["lo"] <= score < b["hi"] or (b["hi"] == top and score >= top):
                    b["n_true" if outcome == "tp" else "n_false"] += 1
                    break
    for b in bins:
        n = b["n_true"] + b["n_false"]
        b["n"] = n
        b["precision"] = b["n_true"] / n if n else None
        b["ci"] = wilson_interval(b["n_true"], n) if n else (None, None)
    return bins


def ground_distance(y, cam_h=CAM_H):
    """Flat-ground distance for a point at normalized elevation ``y``.

    An equirectangular pano maps the vertical axis linearly to elevation, so a point
    at ``y`` sits ``(y - 0.5) * 180`` degrees below the horizon, and a camera at
    height ``cam_h`` sees flat ground there at ``cam_h / tan(theta)``. Strictly
    monotonic in ``y``, so anything expressed as a rank over distance survives any
    monotonic distance model — including a correct one.
    """
    dep = (y - 0.5) * math.pi
    return cam_h / math.tan(dep) if dep > 1e-4 else float("inf")


def matched_gt_at(panos, threshold, radius_sq):
    """Which GT points are found at ``threshold``.

    Returns ``[(pano_id, gt_index, y, matched)]`` over recall-confirmed panos only —
    the same gate ``aggregate`` applies to recall, so the strata below sum to the
    swept recall rather than to a different denominator.
    """
    out = []
    for pd in panos:
        gt = pd["gt"]
        if not gt.fn_confirmed:
            continue
        preds = sorted([p for p in pd["preds"] if p[2] >= threshold],
                       key=lambda p: p[2], reverse=True)
        assignments = greedy_match([(p[0], p[1]) for p in preds], gt.gt_points,
                                   radius_sq, PANO_SCALE_X, PANO_SCALE_Y)
        hit = {gi for gi, _ in assignments if gi >= 0}
        for i, (_gx, gy) in enumerate(gt.gt_points):
            out.append((pd["pano"], i, gy, i in hit))
    return out


DISTANCE_BANDS = ((0.0, 12.5, "near (<12.5 m)"),
                  (12.5, 25.0, "mid (12.5-25 m)"),
                  (25.0, float("inf"), "far (>25 m)"))


def recall_by_distance(panos, radius_sq, low_thr, high_thr=DEPLOYED_THRESHOLD,
                       bands=DISTANCE_BANDS):
    """Recall in each distance band at two thresholds — where the gain lands.

    ``benchmark/README.md`` establishes that RampNet's misses skew far-field. If
    lowering the floor only recovers ramps that were already near, it adds little to
    the multi-view case (labeler#27); if it recovers far ones, the two levers stack.
    """
    hi = {(p, i): m for p, i, _y, m in matched_gt_at(panos, high_thr, radius_sq)}
    lo = matched_gt_at(panos, low_thr, radius_sq)
    rows = []
    for band_lo, band_hi, label in bands:
        n = n_hi = n_lo = 0
        for pano, i, y, matched_lo in lo:
            d = ground_distance(y)
            if not (band_lo <= d < band_hi):
                continue
            n += 1
            n_hi += bool(hi.get((pano, i)))
            n_lo += bool(matched_lo)
        rows.append({
            "band": label, "n_gt": n,
            "recall_high": n_hi / n if n else 0.0,
            "recall_low": n_lo / n if n else 0.0,
            "gained": n_lo - n_hi,
        })
    return rows


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_records(city, repo=REPO):
    """{pano_id: record} from a split's committed records.jsonl."""
    path = os.path.join(repo, "benchmark", city, "records.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["pano"]["panorama_id"]] = r
    return out


def load_split(city, cache_dir=CACHE_DIR, repo=REPO):
    """Cached low-floor panos for a split, each tagged with its imagery tier."""
    panos, meta = read_cache(os.path.join(cache_dir, f"{city}.json"))
    records = load_records(city, repo)
    fallback = SPLIT_IMAGERY_FALLBACK.get(city)
    for pd in panos:
        p = records.get(pd["pano"], {}).get("pano", {})
        pd["tier"] = tier_of(p.get("camera_make"), p.get("camera_model"),
                             p.get("source") or fallback)
        pd["city"] = city
    return panos, meta


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pool_of(cities, include_budapest=False, include_gold=False):
    """Which of ``cities`` contribute to the POOLED and per-tier rows."""
    keep = []
    for c in cities:
        if c == "budapest_district5" and not include_budapest:
            continue
        if c == "manual_gold" and not include_gold:
            continue
        keep.append(c)
    return keep


# --------------------------------------------------------------------------- #
# parity — the gate everything else inherits
# --------------------------------------------------------------------------- #
# A re-extraction is "the same detections" when nothing it moved could change a
# scoring outcome. The scorer matches within one radius R, so displacement is
# measured in radii, not pixels: half a radius is well inside the tolerance that
# decides TP-vs-FP, while being far tighter than any real preprocessing break.
PARITY_TOL_RADII = 0.5
PARITY_MIN_MATCHED = 0.95   # fraction of cache detections that must land within tol
PARITY_MAX_COUNT_DELTA = 0.05


def parity_for(panos, records, threshold=DEPLOYED_THRESHOLD,
               tol_radii=PARITY_TOL_RADII, radius_sq=None):
    """Do the cache's peaks at the deployed threshold reproduce records.jsonl?

    Bit-exactness is the wrong bar and would fail for a benign reason. A split whose
    production run assembled imagery into a 4096x2048 intermediate (the GSV path —
    bend) fed the model a *different image* than this cache's native-res downsample,
    so its peaks land a heatmap cell or two away. A split whose production run
    resized once from native res (the Mapillary path) reproduces exactly.

    So the gate asks the question that actually matters downstream: **could any of
    this drift change a scoring outcome?** Displacements are therefore measured in
    match radii — the unit the scorer works in — and a detection that moved less
    than half a radius is the same detection for every number in this analysis.

    A *gross* mismatch (detections appearing or vanishing, or moving a full radius)
    means preprocessing genuinely diverged, and every number downstream inherits it.
    """
    if radius_sq is None:
        radius_sq = radius_sq_for()
    r = math.sqrt(radius_sq)
    n_cache = n_rec = matched = exact = 0
    disps = []
    per_pano_mismatch = []
    for pd in panos:
        rec = [(d["x_normalized"], d["y_normalized"])
               for d in records[pd["pano"]]["detections"]
               if d["confidence"] >= threshold]
        got = [(x, y) for x, y, s in pd["preds"] if s >= threshold]
        n_cache += len(got)
        n_rec += len(rec)
        if len(got) != len(rec):
            per_pano_mismatch.append((pd["pano"], len(rec), len(got)))
        remaining = list(rec)
        for gx, gy in got:
            if not remaining:
                continue
            j, _ = min(enumerate(remaining),
                       key=lambda t: ((gx - t[1][0]) * PANO_SCALE_X) ** 2
                       + ((gy - t[1][1]) * PANO_SCALE_Y) ** 2)
            dist = math.hypot((gx - remaining[j][0]) * PANO_SCALE_X,
                              (gy - remaining[j][1]) * PANO_SCALE_Y) / r
            if dist <= tol_radii:
                matched += 1
                exact += dist == 0.0
                disps.append(dist)
                remaining.pop(j)
    count_delta = abs(n_cache - n_rec) / n_rec if n_rec else 0.0
    matched_frac = matched / n_cache if n_cache else 0.0
    disps.sort()
    return {
        "n_records": n_rec, "n_cache": n_cache, "matched": matched, "exact": exact,
        "exact_frac": exact / n_cache if n_cache else 0.0,
        "matched_frac": matched_frac,
        "count_delta": count_delta,
        "median_displacement_r": disps[len(disps) // 2] if disps else 0.0,
        "max_displacement_r": disps[-1] if disps else 0.0,
        "panos_with_count_mismatch": per_pano_mismatch,
        "ok": matched_frac >= PARITY_MIN_MATCHED and count_delta <= PARITY_MAX_COUNT_DELTA,
    }


def cmd_parity(args):
    all_ok = True
    rows = []
    print(f"{'split':<22} {'records':>8} {'cache':>7} {'exact':>7} {'within tol':>11} "
          f"{'med R':>7} {'max R':>7}  verdict")
    print("-" * 92)
    for city in args.cities:
        panos, _meta = load_split(city, args.cache)
        res = parity_for(panos, load_records(city), args.threshold, args.tol_radii)
        rows.append((city, res))
        gated = city not in TTA_RECORD_SPLITS
        if gated:
            all_ok &= res["ok"]
            verdict = "OK" if res["ok"] else "MISMATCH"
        else:
            verdict = "n/a (TTA)"
        print(f"{city:<22} {res['n_records']:>8} {res['n_cache']:>7} "
              f"{res['exact_frac']:>6.1%} {res['matched_frac']:>10.1%} "
              f"{res['median_displacement_r']:>7.3f} {res['max_displacement_r']:>7.3f}  "
              f"{verdict}")
        if gated and not res["ok"]:
            for pano, n_rec, n_got in res["panos_with_count_mismatch"][:5]:
                print(f"    {pano}: records {n_rec} vs cache {n_got}")

    print(f"\nDisplacements are in match radii (R); tolerance {args.tol_radii} R. "
          f"'exact' = peak landed in the identical heatmap cell.")
    for city, res in rows:
        if city in TTA_RECORD_SPLITS:
            print(f"\n{city}: NOT GATED — its committed detections were exported WITH "
                  f"horizontal-flip TTA\n  (benchmark/{city}/detections_meta.json) at a "
                  f"0.05 floor, while this cache is the no-TTA\n  deployment path. Its "
                  f"row is a TTA-vs-no-TTA delta, not preprocessing drift (issue #78).")
        elif res["exact_frac"] < 0.99:
            print(f"\n{city}: reproduces within tolerance but not exactly "
                  f"({res['exact_frac']:.1%} identical cells).\n  Expected where the "
                  f"production run fed the model a different resample of the same pano "
                  f"—\n  the GSV path built a 4096x2048 intermediate, so bundle "
                  f"native-res != production input.\n  Every displacement is under "
                  f"{res['max_displacement_r']:.2f} R, so no scoring outcome changes.")
    print("\nParity gate: " + ("PASS — the cache reproduces the committed detections."
                               if all_ok else
                               "FAIL — preprocessing diverged; downstream numbers "
                               "inherit it."))
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
SWEEP_FIELDS = ["group", "threshold", "precision", "recall", "f1", "precision_lo",
                "precision_hi", "recall_lo", "recall_hi", "tp", "fp", "fn", "ignored",
                "dets_per_pano", "n_panos"]

DISPLAY_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def _print_rows(title, rows, marks):
    """Print the coarse rows plus anything notable; the CSV keeps the full grid."""
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print(f"{'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'dets/pano':>10} "
          f"{'tp/fp/fn':>18}   note")
    print("-" * 88)
    best = best_f1_row(rows)
    for r in rows:
        note = []
        if abs(r["threshold"] - DEPLOYED_THRESHOLD) < 1e-9:
            note.append("deployed")
        if r is best:
            note.append("F1-max")
        if any(abs(r["threshold"] - m) < 1e-9 for m in marks):
            note.append("candidate")
        if not note and not any(abs(r["threshold"] - d) < 1e-9
                                for d in DISPLAY_THRESHOLDS):
            continue
        counts = f'{r["tp"]}/{r["fp"]}/{r["fn"]}'
        print(f"{r['threshold']:>5.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['dets_per_pano']:>10.2f} "
              f"{counts:>18}   {' '.join(note)}")
    dep = row_at(rows, DEPLOYED_THRESHOLD)
    print(f"  deployed {dep['threshold']:.2f} -> F1-max {best['threshold']:.2f}: "
          f"recall {dep['recall']:.3f} -> {best['recall']:.3f} "
          f"({best['recall'] - dep['recall']:+.3f}), "
          f"precision {dep['precision']:.3f} -> {best['precision']:.3f} "
          f"({best['precision'] - dep['precision']:+.3f}), "
          f"density {dep['dets_per_pano']:.2f} -> {best['dets_per_pano']:.2f}/pano")


def cmd_sweep(args):
    radius_sq = radius_sq_for()
    grid = threshold_grid(args.floor, args.top)
    loaded = {city: load_split(city, args.cache)[0] for city in args.cities}
    all_rows = []

    for city, panos in loaded.items():
        rows = sweep_rows(panos, grid, radius_sq)
        for r in rows:
            r["group"] = city
        all_rows += rows
        held = HELD_OUT.get(city)
        suffix = f"   [held out of POOLED: {held}]" if held else ""
        _print_rows(f"{city.upper()}  (n={len(panos)} panos){suffix}", rows, args.mark)

    poolable = pool_of(args.cities, args.include_budapest, args.include_gold)
    pooled = [pd for c in poolable for pd in loaded[c]]
    if len(poolable) > 1:
        rows = sweep_rows(pooled, grid, radius_sq)
        for r in rows:
            r["group"] = "POOLED"
        all_rows += rows
        _print_rows(f"POOLED  ({', '.join(poolable)}; n={len(pooled)} panos)",
                    rows, args.mark)

    by_tier = {}
    for pd in pooled:
        by_tier.setdefault(pd["tier"], []).append(pd)
    for tier, panos in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        rows = sweep_rows(panos, grid, radius_sq)
        for r in rows:
            r["group"] = f"tier:{tier}"
        all_rows += rows
        cities = sorted({pd["city"] for pd in panos})
        _print_rows(f"TIER {TIER_LABEL.get(tier, tier)}  "
                    f"(n={len(panos)} panos from {', '.join(cities)})", rows, args.mark)

    path = os.path.join(args.out, "low_floor_sweep.csv")
    _write_csv(path, all_rows, SWEEP_FIELDS)
    print(f"\nwrote {path}  ({len(all_rows)} rows, full grid)")
    for city in args.cities:
        if city not in poolable:
            print(f"held out of POOLED/tier rows: {city} — {HELD_OUT[city]}")
    return 0


# --------------------------------------------------------------------------- #
# hist
# --------------------------------------------------------------------------- #
def _print_bins(title, bins):
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(f"{'bin':>14} {'GT-true':>8} {'GT-false':>9} {'P(real)':>8}  95% CI")
    print("-" * 76)
    for b in bins:
        if not b["n"]:
            continue
        label = f'{b["lo"]:.2f}-{b["hi"]:.2f}'
        ci = f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]"
        print(f"{label:>14} {b['n_true']:>8} {b['n_false']:>9} "
              f"{b['precision']:>8.3f}  {ci}")


def cmd_hist(args):
    radius_sq = radius_sq_for()
    n_bins = int(round((1.0 - args.floor) / args.bin_width))
    edges = [round(args.floor + i * args.bin_width, 4) for i in range(n_bins + 1)]
    payload = {"bin_edges": edges, "splits": {}}
    loaded = {city: load_split(city, args.cache)[0] for city in args.cities}

    for city, panos in loaded.items():
        bins = confidence_calibration(panos, radius_sq, edges)
        payload["splits"][city] = bins
        _print_bins(f"{city.upper()} — P(real | confidence bin)", bins)

    poolable = pool_of(args.cities, args.include_budapest, args.include_gold)
    pooled = [pd for c in poolable for pd in loaded[c]]
    pooled_bins = confidence_calibration(pooled, radius_sq, edges)
    payload["pooled"] = {"splits": poolable, "bins": pooled_bins}
    _print_bins(f"POOLED ({', '.join(poolable)}) — P(real | confidence bin)", pooled_bins)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "confidence_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    flat = [{"split": split, "lo": b["lo"], "hi": b["hi"], "n_true": b["n_true"],
             "n_false": b["n_false"], "precision": b["precision"],
             "ci_lo": b["ci"][0], "ci_hi": b["ci"][1]}
            for split, bins in list(payload["splits"].items()) + [("POOLED", pooled_bins)]
            for b in bins]
    _write_csv(os.path.join(args.out, "confidence_calibration.csv"), flat,
               ["split", "lo", "hi", "n_true", "n_false", "precision", "ci_lo", "ci_hi"])
    print(f"\nwrote {path} + confidence_calibration.csv")
    return 0


# --------------------------------------------------------------------------- #
# distance
# --------------------------------------------------------------------------- #
def cmd_distance(args):
    radius_sq = radius_sq_for()
    rows_out = []
    for city in args.cities:
        panos, _ = load_split(city, args.cache)
        rows = recall_by_distance(panos, radius_sq, args.low, args.high)
        print(f"\n{'=' * 78}\n{city.upper()} — recall by distance, "
              f"{args.high:.2f} -> {args.low:.2f}\n{'=' * 78}")
        print(f"{'band':>20} {'n_gt':>6} {'R@high':>8} {'R@low':>8} {'gain':>7} {'+ramps':>7}")
        print("-" * 78)
        for r in rows:
            print(f"{r['band']:>20} {r['n_gt']:>6} {r['recall_high']:>8.3f} "
                  f"{r['recall_low']:>8.3f} "
                  f"{r['recall_low'] - r['recall_high']:>7.3f} {r['gained']:>7}")
            rows_out.append({"split": city, **r})
    _write_csv(os.path.join(args.out, "recall_by_distance.csv"), rows_out,
               ["split", "band", "n_gt", "recall_high", "recall_low", "gained"])
    print(f"\nwrote {os.path.join(args.out, 'recall_by_distance.csv')}")
    print("Distances are the flat-ground estimate (camera height 2.5 m assumed). It is "
          "monotonic in y,\nso the band ordering is a rank statement; only the metre "
          "labels depend on the assumption.")
    return 0


# --------------------------------------------------------------------------- #
def _csv_list(s):
    return tuple(c.strip() for c in s.split(",") if c.strip())


def _floats(s):
    return tuple(float(v) for v in s.split(",") if v.strip())


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cities", type=_csv_list, default=ALL_SPLITS)
        sp.add_argument("--cache", default=CACHE_DIR)
        sp.add_argument("--out", default=os.path.join(OUT, "op"))
        sp.add_argument("--include-budapest", action="store_true",
                        help="pool budapest into POOLED/tier rows (default: held out — "
                             "single-rater low-confidence GT)")
        sp.add_argument("--include-gold", action="store_true",
                        help="pool manual_gold into POOLED/tier rows (default: held out — "
                             "in-distribution reference, not a deployment city)")

    sp = sub.add_parser("parity", help="gate: cache at 0.55 must reproduce records.jsonl")
    common(sp)
    sp.add_argument("--threshold", type=float, default=DEPLOYED_THRESHOLD)
    sp.add_argument("--tol-radii", type=float, default=PARITY_TOL_RADII,
                    help="displacement tolerance in match radii (default "
                         f"{PARITY_TOL_RADII}); a detection that moved less than this "
                         "cannot change a scoring outcome")
    sp.set_defaults(func=cmd_parity)

    sp = sub.add_parser("sweep", help="per-split + pooled + per-tier P/R/F1 vs threshold")
    common(sp)
    sp.add_argument("--floor", type=float, default=0.05)
    sp.add_argument("--top", type=float, default=0.90)
    sp.add_argument("--mark", type=_floats, default=(0.25, 0.30, 0.35),
                    help="candidate operating points to flag in the printed table")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("hist", help="GT-true vs GT-false confidence calibration (labeler#27)")
    common(sp)
    sp.add_argument("--floor", type=float, default=0.05)
    sp.add_argument("--bin-width", type=float, default=0.05)
    sp.set_defaults(func=cmd_hist)

    sp = sub.add_parser("distance", help="where the recall gain lands on the distance axis")
    common(sp)
    sp.add_argument("--low", type=float, default=0.30)
    sp.add_argument("--high", type=float, default=DEPLOYED_THRESHOLD)
    sp.set_defaults(func=cmd_distance)

    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
