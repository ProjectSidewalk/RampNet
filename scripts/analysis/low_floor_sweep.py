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
    python scripts/analysis/low_floor_sweep.py tta        # flip-TTA vs single-pass (#78);
                                                          # needs extract --tta for the cities
"""
import argparse
import csv
import json
import math
import os
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, aggregate, radius_sq_for, score_pano)
from rampnet.metrics import greedy_match  # noqa: E402
from rampnet.validation import wilson_interval  # noqa: E402

from operating_point_curve import (  # noqa: E402
    DEPLOYED_THRESHOLD, classify_predictions, pr_curve_and_ap, read_cache)

# The seven US city splits carry verdict-grade GT and are the recommendation's basis.
US_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
             "gainesville", "laurens")
CITY_SPLITS = US_SPLITS + ("budapest_district5", "sao_paulo")
ALL_SPLITS = CITY_SPLITS + ("manual_gold",)

# Why a split is swept but not pooled. Printed with the results so an omission can
# never be mistaken for a withheld result.
HELD_OUT = {
    "budapest_district5": "single-rater GT at low reviewer confidence "
                          "(docs/model_comparison.md: do not pool)",
    "sao_paulo": "non-US city — the pooled recommendation is a US-deployment "
                 "basis (GT is HIGH reviewer confidence; held out for "
                 "geography, not GT quality)",
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
TTA_CACHE_DIR = os.path.join(OUT, "op_cache_tta")   # the flip-TTA arm (issue #78)

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


def tta_levers(rows_single, rows_tta, deployed=DEPLOYED_THRESHOLD, candidate=0.30):
    """Issue #78's question in four rows: what each recall lever buys, alone and
    together. Flip-TTA and threshold-lowering both promote under-confident
    detections, so their gains overlap — the honest measure of TTA's production
    value is therefore not "TTA at the deployed point" but the **marginal** row:
    what TTA still adds *after* the threshold has already been dropped. That row
    is what a 2x-GPU-per-pano decision should be priced against.
    """
    s_dep, s_cand = row_at(rows_single, deployed), row_at(rows_single, candidate)
    t_dep, t_cand = row_at(rows_tta, deployed), row_at(rows_tta, candidate)

    def lever(name, frm, to):
        return {"lever": name,
                "d_recall": to["recall"] - frm["recall"],
                "d_precision": to["precision"] - frm["precision"],
                "d_f1": to["f1"] - frm["f1"],
                "dets_per_pano": to["dets_per_pano"]}

    return [
        lever(f"threshold drop alone (single {deployed:.2f}->{candidate:.2f})",
              s_dep, s_cand),
        lever(f"TTA alone (at {deployed:.2f})", s_dep, t_dep),
        lever(f"both (single {deployed:.2f} -> TTA {candidate:.2f})", s_dep, t_cand),
        lever(f"TTA after the drop (at {candidate:.2f})", s_cand, t_cand),
    ]


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


def tta_panos_from_records(single_panos, records, floor):
    """The flip-TTA arm for a split whose *committed* detections were already
    exported with TTA at a low floor (``manual_gold`` —
    ``benchmark/manual_gold/detections_meta.json``): no GPU pass needed, the
    records are the arm. Pano set and GT are taken from the single-pass cache
    entry-for-entry, so both arms score identical panos against identical GT and
    any delta is the TTA composition alone.
    """
    out = []
    for pd in single_panos:
        preds = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                 for d in records[pd["pano"]]["detections"]
                 if d["confidence"] >= floor]
        out.append({"pano": pd["pano"], "preds": preds, "gt": pd["gt"],
                    "tier": pd.get("tier"), "city": pd.get("city")})
    return out


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pool_of(cities, include_budapest=False, include_gold=False,
            include_sao_paulo=False):
    """Which of ``cities`` contribute to the POOLED and per-tier rows."""
    keep = []
    for c in cities:
        if c == "budapest_district5" and not include_budapest:
            continue
        if c == "manual_gold" and not include_gold:
            continue
        if c == "sao_paulo" and not include_sao_paulo:
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

# Splits whose parity failure has been diagnosed and ratified by the reviewer. Keyed
# like HELD_OUT, and for the same reason: a gate that fails forever on a clean clone
# teaches people to ignore it, and an exception that lives only in prose leaves a
# reproducer unable to tell a known finding from their own broken checkout. A split
# listed here still prints MISMATCH and still prints why — it just does not count as
# a NEW divergence, so the exit status keeps meaning "nothing regressed since this
# was ratified". Removing an entry is how you re-open the question.
PARITY_EXCEPTIONS = {
    "sao_paulo":
        "count arm 9.2% (23 of 251) > the 5% allowance; displacement arm passes "
        "(max 0.439 R, the GSV signature). Diagnosed and ratified by the reviewer "
        "2026-08-01: same GSV resample mechanism as every GSV split, amplified by "
        "out-of-domain scores hugging the threshold — 20 threshold-straddlers + 19 "
        "NMS pair-merges. Rows at <= 0.38 (including the recommended 0.30) are "
        "unaffected; rows at >= 0.55 understate production recall. Full decomposition "
        "in docs/operating_point.md.",
}


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
        panos, meta = load_split(city, args.cache)
        cache_tta = bool(meta.get("tta", False))
        res = parity_for(panos, load_records(city), args.threshold, args.tol_radii)
        rows.append((city, res, cache_tta))
        # Gate only when the cache and the committed records are the same arm:
        # city records are single-pass, manual_gold's are TTA (TTA_RECORD_SPLITS),
        # and a --cache pointing at the TTA extraction (#78) inverts both. A
        # cross-arm row is a TTA-vs-single delta, not preprocessing drift.
        gated = cache_tta == (city in TTA_RECORD_SPLITS)
        if gated:
            all_ok &= res["ok"] or city in PARITY_EXCEPTIONS
            verdict = ("OK" if res["ok"] else
                       "MISMATCH (ratified)" if city in PARITY_EXCEPTIONS else
                       "MISMATCH")
        else:
            verdict = "n/a (cross-arm)"
        print(f"{city:<22} {res['n_records']:>8} {res['n_cache']:>7} "
              f"{res['exact_frac']:>6.1%} {res['matched_frac']:>10.1%} "
              f"{res['median_displacement_r']:>7.3f} {res['max_displacement_r']:>7.3f}  "
              f"{verdict}")
        if gated and not res["ok"]:
            for pano, n_rec, n_got in res["panos_with_count_mismatch"][:5]:
                print(f"    {pano}: records {n_rec} vs cache {n_got}")

    print(f"\nDisplacements are in match radii (R); tolerance {args.tol_radii} R. "
          f"'exact' = peak landed in the identical heatmap cell.")
    for city, res, cache_tta in rows:
        if cache_tta != (city in TTA_RECORD_SPLITS):
            side = ("its committed detections were exported WITH horizontal-flip TTA\n"
                    f"  (benchmark/{city}/detections_meta.json) at a 0.05 floor, while "
                    "this cache is the no-TTA\n  deployment path"
                    if city in TTA_RECORD_SPLITS else
                    "this cache is the flip-TTA arm, while its committed detections "
                    "are the\n  single-pass deployment path")
            print(f"\n{city}: NOT GATED — {side}. Its row is a TTA-vs-no-TTA delta, "
                  f"not preprocessing drift (issue #78).")
        elif not res["ok"] and city in PARITY_EXCEPTIONS:
            # Deliberately ahead of the "within tolerance" branch below: that text
            # reads off the displacement arm alone and would tell you no scoring
            # outcome changes, which for a ratified count-arm failure is false.
            print(f"\n{city}: MISMATCH — RATIFIED EXCEPTION, not a new divergence.\n"
                  + textwrap.fill(PARITY_EXCEPTIONS[city], width=90,
                                  initial_indent="  ", subsequent_indent="  "))
        elif res["exact_frac"] < 0.99:
            print(f"\n{city}: reproduces within tolerance but not exactly "
                  f"({res['exact_frac']:.1%} identical cells).\n  Expected where the "
                  f"production run fed the model a different resample of the same pano "
                  f"—\n  the GSV path built a 4096x2048 intermediate, so bundle "
                  f"native-res != production input.\n  Every displacement is under "
                  f"{res['max_displacement_r']:.2f} R, so no scoring outcome changes.")
    ratified = [c for c, res, cache_tta in rows
                if not res["ok"] and cache_tta == (c in TTA_RECORD_SPLITS)
                and c in PARITY_EXCEPTIONS]
    if not all_ok:
        gate = "FAIL — preprocessing diverged; downstream numbers inherit it."
    elif ratified:
        gate = ("PASS with ratified exception(s) — " + ", ".join(ratified) +
                ". Nothing regressed, but those splits' numbers carry the caveat "
                "printed above; read it before quoting them.")
    else:
        gate = "PASS — the cache reproduces the committed detections."
    print("\nParity gate: " + gate)
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

    poolable = pool_of(args.cities, args.include_budapest, args.include_gold,
                       args.include_sao_paulo)
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
# tta — flip-TTA vs single-pass at the operating points (issue #78)
# --------------------------------------------------------------------------- #
def _load_tta_arm(city, single, tta_cache, floor):
    """``(panos, source)`` for a split's flip-TTA arm, or ``(None, why-missing)``.

    Preference order: a real ``extract --tta`` cache; else, for a split whose
    committed records ARE a TTA export at a low floor (``manual_gold``), the
    records themselves via :func:`tta_panos_from_records`.
    """
    path = os.path.join(tta_cache, f"{city}.json")
    if os.path.exists(path):
        panos, meta = load_split(city, tta_cache)
        if not meta.get("tta", False):
            raise SystemExit(f"{path}: meta says single-pass — not a TTA cache; "
                             "re-run extract --tta into its own --cache directory")
        return panos, "extract --tta cache"
    if city in TTA_RECORD_SPLITS:
        mpath = os.path.join(REPO, "benchmark", city, "detections_meta.json")
        with open(mpath, encoding="utf-8") as f:
            dm = json.load(f)
        if not dm.get("tta") or dm.get("peak_floor", 1.0) > floor:
            raise SystemExit(f"{mpath}: expected a TTA export at floor <= {floor}; "
                             f"got tta={dm.get('tta')} peak_floor={dm.get('peak_floor')}")
        return (tta_panos_from_records(single, load_records(city), dm["peak_floor"]),
                f"committed records (TTA export, floor {dm['peak_floor']})")
    return None, f"no cache at {path} — run operating_point_curve.py extract --tta"


def _print_tta(title, rows_s, rows_t, candidate, ap_s, ap_t):
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")
    print(f"{'point':<12} {'thr':>5} | {'P':>6} {'R':>6} {'F1':>6} {'d/pano':>7} | "
          f"{'P':>6} {'R':>6} {'F1':>6} {'d/pano':>7} | {'dR':>7} {'dP':>7}")
    print(f"{'':<12} {'':>5} | {'single':^28} | {'flip-TTA':^28} |")
    print("-" * 96)
    marks = [("deployed", row_at(rows_s, DEPLOYED_THRESHOLD),
              row_at(rows_t, DEPLOYED_THRESHOLD)),
             ("candidate", row_at(rows_s, candidate), row_at(rows_t, candidate)),
             ("F1-max*", best_f1_row(rows_s), best_f1_row(rows_t))]
    for label, s, t in marks:
        thr = (f"{s['threshold']:.2f}" if s["threshold"] == t["threshold"]
               else f"{s['threshold']:.2f}/{t['threshold']:.2f}")
        print(f"{label:<12} {thr:>5} | {s['precision']:>6.3f} {s['recall']:>6.3f} "
              f"{s['f1']:>6.3f} {s['dets_per_pano']:>7.2f} | {t['precision']:>6.3f} "
              f"{t['recall']:>6.3f} {t['f1']:>6.3f} {t['dets_per_pano']:>7.2f} | "
              f"{t['recall'] - s['recall']:>+7.3f} {t['precision'] - s['precision']:>+7.3f}")
    print(f"  (*each arm's own F1-max threshold — chosen on this benchmark, "
          f"quote as tune-on-test)")
    print(f"  AP: single {ap_s:.3f}  flip-TTA {ap_t:.3f}  ({ap_t - ap_s:+.3f})")
    print("  recall levers, all measured from single@%.2f:" % DEPLOYED_THRESHOLD)
    for lv in tta_levers(rows_s, rows_t, candidate=candidate):
        marginal = lv["lever"].startswith("TTA after")
        print(f"    {lv['lever']:<44} R {lv['d_recall']:+.3f}  P {lv['d_precision']:+.3f}  "
              f"F1 {lv['d_f1']:+.3f}  -> {lv['dets_per_pano']:.2f} dets/pano"
              + ("   <- the 2x-GPU decision" if marginal else ""))


def cmd_tta(args):
    radius_sq = radius_sq_for()
    grid = threshold_grid(args.floor, args.top)
    loaded = {}
    for city in args.cities:
        single, _smeta = load_split(city, args.cache)
        tta, source = _load_tta_arm(city, single, args.tta_cache, args.floor)
        if tta is None:
            print(f"[{city}] TTA arm unavailable: {source}; skipped")
            continue
        if {pd["pano"] for pd in single} != {pd["pano"] for pd in tta}:
            raise SystemExit(f"{city}: the two arms cover different pano sets — "
                             "stale cache? re-extract so both score identical panos")
        loaded[city] = (single, tta)
        print(f"[{city}] TTA arm: {source}")
    if not loaded:
        raise SystemExit("no split has both arms; run extract --tta first "
                         f"(expected caches under {args.tta_cache})")

    all_rows = []

    def compare(group, single, tta, title):
        rows_s = sweep_rows(single, grid, radius_sq)
        rows_t = sweep_rows(tta, grid, radius_sq)
        for r in rows_s:
            r.update(group=group, arm="single")
        for r in rows_t:
            r.update(group=group, arm="tta")
        all_rows.extend(rows_s + rows_t)
        _print_tta(title, rows_s, rows_t, args.candidate,
                   pr_curve_and_ap(single, radius_sq).ap,
                   pr_curve_and_ap(tta, radius_sq).ap)

    for city, (single, tta) in loaded.items():
        held = HELD_OUT.get(city)
        suffix = f"   [held out of POOLED: {held}]" if held else ""
        compare(city, single, tta,
                f"{city.upper()}  (n={len(single)} panos){suffix} — flip-TTA vs single-pass")

    poolable = pool_of(tuple(loaded), args.include_budapest, args.include_gold,
                       args.include_sao_paulo)
    if len(poolable) > 1:
        pooled_s = [pd for c in poolable for pd in loaded[c][0]]
        pooled_t = [pd for c in poolable for pd in loaded[c][1]]
        compare("POOLED", pooled_s, pooled_t,
                f"POOLED  ({', '.join(poolable)}; n={len(pooled_s)} panos) — "
                f"flip-TTA vs single-pass")

    path = os.path.join(args.out, "tta_compare.csv")
    _write_csv(path, all_rows, ["group", "arm"] + SWEEP_FIELDS[1:])
    print(f"\nwrote {path}  ({len(all_rows)} rows, full grid, both arms)")
    for city in loaded:
        if city not in poolable:
            print(f"held out of POOLED rows: {city} — {HELD_OUT[city]}")
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

    poolable = pool_of(args.cities, args.include_budapest, args.include_gold,
                       args.include_sao_paulo)
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
# tagcheck — do the committed #55 tags still resolve after a re-extraction?
# --------------------------------------------------------------------------- #
def tags_path_for(city, repo=REPO):
    return os.path.join(repo, "benchmark", city, "incremental_fp_tags.json")


def check_tag_resolution(items, tags):
    """Which committed A/B tags still point at an incremental FP in this cache.

    Tag ids are ``{pano}_{x:.5f}_{y:.5f}`` (see ``incremental_fps``), so they are
    keyed to peak *coordinates*. Re-extracting on different hardware can move a
    marginal peak by a heatmap cell and orphan its tag — silently, since an
    unresolved tag just stops contributing to the correction and quietly shrinks it.

    Returns the resolved/orphaned split plus the orphans themselves, so a
    re-extraction can never quietly discard reviewer effort.
    """
    ids = {it["id"] for it in items}
    resolved = sorted(t for t in tags if t in ids)
    orphaned = sorted(t for t in tags if t not in ids)
    return {
        "n_tags": len(tags), "n_items": len(items),
        "resolved": resolved, "orphaned": orphaned,
        "resolved_frac": len(resolved) / len(tags) if tags else 1.0,
        "untagged": sorted(ids - set(tags)),
    }


def cmd_tagcheck(args):
    from operating_point_curve import incremental_fps
    radius_sq = radius_sq_for()
    all_ok = True
    for city in args.cities:
        path = tags_path_for(city)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            tags = json.load(f)
        panos, _ = load_split(city, args.cache)
        items = incremental_fps(panos, args.op_threshold, args.upper, radius_sq)
        res = check_tag_resolution(items, tags)
        ok = res["resolved_frac"] >= args.min_resolved
        all_ok &= ok
        print(f"\n{city}: {len(res['resolved'])}/{res['n_tags']} committed tags resolve "
              f"({res['resolved_frac']:.1%})  — {res['n_items']} incremental FPs in "
              f"[{args.op_threshold}, {args.upper})   {'OK' if ok else 'DEGRADED'}")
        for t in res["orphaned"][:10]:
            print(f"    orphaned: {t}")
        if res["untagged"]:
            print(f"    {len(res['untagged'])} incremental FP(s) in this cache carry no "
                  f"tag — they widen the corrected band rather than being assumed real.")
    print("\nTag resolution: " + ("PASS — re-extraction preserved the reviewer's work."
                                  if all_ok else
                                  "DEGRADED — some tags no longer match a detection; the "
                                  "#55 correction would silently shrink. Re-tag the "
                                  "orphans or keep the cache they were made against."))
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# floor — does a 0.1 storage floor throw away recoverable ramps? (labeler#28/#27)
# --------------------------------------------------------------------------- #
def gt_best_candidate(panos, radius_sq, floor=0.0):
    """Per GT ramp, the confidence of the detection that claims it — or None.

    Greedy matching is highest-confidence-first, so dropping low-confidence
    predictions never changes what a *higher*-confidence one matched. That makes
    this one pass sufficient for every floor at once: the set of GT ramps still
    recoverable at floor ``f`` is exactly those whose best candidate scores ``>= f``.

    Restricted to recall-confirmed panos, the same gate ``aggregate`` applies, so
    these counts share a denominator with the swept recall.
    """
    out = []
    for pd in panos:
        gt = pd["gt"]
        if not gt.fn_confirmed:
            continue
        preds = sorted([p for p in pd["preds"] if p[2] >= floor],
                       key=lambda p: p[2], reverse=True)
        assignments = greedy_match([(p[0], p[1]) for p in preds], gt.gt_points,
                                   radius_sq, PANO_SCALE_X, PANO_SCALE_Y)
        best = {}
        for p, (gi, _) in zip(preds, assignments):
            if gi >= 0:
                best[gi] = p[2]
        for i in range(len(gt.gt_points)):
            out.append(best.get(i))
    return out


STORAGE_FLOOR = 0.10        # sidewalk-auto-labeler DETECTION_STORAGE_FLOOR (PR #28)
STORAGE_TOP_K = 50          # its per-pano cap


def floor_report(panos, radius_sq, bands=((0.05, 0.10), (0.10, 0.20)),
                 floors=(0.05, 0.10, 0.20, DEPLOYED_THRESHOLD)):
    """Recall ceiling at each candidate floor, plus where the marginal ramps sit.

    Two questions, one pass:

    - **Is a 0.1 storage floor safe?** ``bands`` counts GT ramps whose *best*
      candidate falls in each band. The count in ``[0.05, 0.10)`` is exactly the
      number of real ramps a 0.1 floor makes permanently unrecoverable — no
      downstream consensus policy can promote a candidate that was never stored.
    - **What is the ceiling on multi-view promotion?** ``recall_at`` is the share of
      GT ramps with any candidate at or above each floor. Stage 4 of labeler#27
      cannot exceed this, whatever k it requires.
    """
    best = gt_best_candidate(panos, radius_sq)
    n_gt = len(best)
    matched = [b for b in best if b is not None]
    return {
        "n_gt": n_gt,
        "n_unmatched": n_gt - len(matched),
        "bands": {f"[{lo:.2f},{hi:.2f})": sum(1 for b in matched if lo <= b < hi)
                  for lo, hi in bands},
        "recall_at": {f"{f:.2f}": sum(1 for b in matched if b >= f) / n_gt if n_gt else 0.0
                      for f in floors},
        "n_at": {f"{f:.2f}": sum(1 for b in matched if b >= f) for f in floors},
    }


def cap_report(panos, floor=STORAGE_FLOOR, top_k=STORAGE_TOP_K):
    """Does the labeler's per-pano top-K cap bind at this storage floor?

    The cap is the real volume bound in labeler#28, so it is worth knowing whether
    it ever actually truncates — a cap that never binds costs nothing, one that
    binds often is silently a second, harsher floor.
    """
    counts = sorted(sum(1 for p in pd["preds"] if p[2] >= floor) for pd in panos)
    n = len(counts) or 1
    return {
        "median": counts[n // 2] if counts else 0,
        "p95": counts[min(n - 1, int(0.95 * n))] if counts else 0,
        "max": counts[-1] if counts else 0,
        "n_over_cap": sum(1 for c in counts if c > top_k),
        "n_panos": len(counts),
    }


def cmd_floor(args):
    radius_sq = radius_sq_for()
    loaded = {city: load_split(city, args.cache)[0] for city in args.cities}
    rows = []
    band_keys = [f"[{lo:.2f},{hi:.2f})" for lo, hi in ((0.05, 0.10), (0.10, 0.20))]

    print(f"{'split':<22} {'GT':>5} {band_keys[0]:>13} {band_keys[1]:>13} "
          f"{'R@0.05':>7} {'R@0.10':>7} {'R@0.55':>7} {'lost@0.10':>10}")
    print("-" * 96)

    def emit(label, panos):
        rep = floor_report(panos, radius_sq)
        lost = rep["bands"][band_keys[0]]
        print(f"{label:<22} {rep['n_gt']:>5} {rep['bands'][band_keys[0]]:>13} "
              f"{rep['bands'][band_keys[1]]:>13} {rep['recall_at']['0.05']:>7.3f} "
              f"{rep['recall_at']['0.10']:>7.3f} {rep['recall_at']['0.55']:>7.3f} "
              f"{lost / rep['n_gt'] if rep['n_gt'] else 0:>9.2%}")
        rows.append({"split": label, "n_gt": rep["n_gt"],
                     "gt_best_in_005_010": rep["bands"][band_keys[0]],
                     "gt_best_in_010_020": rep["bands"][band_keys[1]],
                     **{f"recall_at_{k}": v for k, v in rep["recall_at"].items()},
                     **{f"n_at_{k}": v for k, v in rep["n_at"].items()}})
        return rep

    for city, panos in loaded.items():
        emit(city, panos)
    poolable = pool_of(args.cities, args.include_budapest, args.include_gold,
                       args.include_sao_paulo)
    pooled_panos = [pd for c in poolable for pd in loaded[c]]
    print("-" * 96)
    pooled = emit("POOLED", pooled_panos) if len(poolable) > 1 else None

    print(f"\nPer-pano candidate counts at the {STORAGE_FLOOR} storage floor "
          f"(cap = top {STORAGE_TOP_K}):")
    print(f"{'split':<22} {'median':>7} {'p95':>6} {'max':>6} {'panos over cap':>16}")
    print("-" * 62)
    for city, panos in loaded.items():
        c = cap_report(panos, args.floor, args.top_k)
        print(f"{city:<22} {c['median']:>7} {c['p95']:>6} {c['max']:>6} "
              f"{c['n_over_cap']:>16}")

    _write_csv(os.path.join(args.out, "storage_floor.csv"), rows,
               ["split", "n_gt", "gt_best_in_005_010", "gt_best_in_010_020",
                "recall_at_0.05", "recall_at_0.10", "recall_at_0.20", "recall_at_0.55",
                "n_at_0.05", "n_at_0.10", "n_at_0.20", "n_at_0.55"])
    if pooled:
        lost = pooled["bands"][band_keys[0]]
        print(f"""
{'=' * 88}
Verdict on DETECTION_STORAGE_FLOOR = {STORAGE_FLOOR} (labeler#28)
{'=' * 88}
Across the pooled US splits, {lost} of {pooled['n_gt']} ground-truth ramps
({lost / pooled['n_gt']:.2%}) have their best candidate in [0.05, 0.10) — those are the
ramps a {STORAGE_FLOOR} storage floor discards at the only point they exist.

Recall ceiling: {pooled['recall_at']['0.10']:.3f} at the {STORAGE_FLOOR} floor, against
{pooled['recall_at']['0.05']:.3f} at the 0.05 extraction floor and {pooled['recall_at']['0.55']:.3f}
at the deployed threshold. **No labeler#27 stage-4 consensus policy can exceed the
first number**, whatever k it requires, because a candidate that was never stored
cannot be promoted.""")
    print(f"\nwrote {os.path.join(args.out, 'storage_floor.csv')}")
    return 0


# --------------------------------------------------------------------------- #
# corrected — apply the #55 A/B tags, per split and pooled
# --------------------------------------------------------------------------- #
def cmd_corrected(args):
    from operating_point_curve import (corrected_precision, corrected_recall,
                                       f1_of, incremental_fps, _score_at)
    radius_sq = radius_sq_for()
    rows = []
    pooled_items, pooled_tags = [], {}
    pooled = {"tp": 0, "fp": 0, "tp_recall": 0, "n_gt_recall": 0}

    print(f"{'split':<22} {'raw P':>7} {'corr P':>7} {'band hi':>8} "
          f"{'raw R':>7} {'corr R':>7} {'raw F1':>7} {'corr F1':>8}  A/B/U")
    print("-" * 96)
    for city in args.cities:
        path = tags_path_for(city)
        if not os.path.exists(path):
            print(f"{city:<22} (no tags — not spot-checked)")
            continue
        with open(path, encoding="utf-8") as f:
            tags = json.load(f)
        panos, _ = load_split(city, args.cache)
        items = incremental_fps(panos, args.op_threshold, args.upper, radius_sq)
        rep = _score_at(panos, args.op_threshold, radius_sq)
        p = corrected_precision(rep.tp, rep.fp, items, tags)
        r = corrected_recall(rep.n_gt_recall - rep.fn, rep.n_gt_recall, items, tags)
        counts = f"{p['n_A']}/{p['n_B']}/{p['n_U']}"
        print(f"{city:<22} {p['uncorrected']:>7.3f} {p['corrected']:>7.3f} "
              f"{p['band_high']:>8.3f} {r['uncorrected']:>7.3f} {r['corrected']:>7.3f} "
              f"{f1_of(p['uncorrected'], r['uncorrected']):>7.3f} "
              f"{f1_of(p['corrected'], r['corrected']):>8.3f}  {counts}")
        rows.append({"split": city, "op_threshold": args.op_threshold,
                     "precision_raw": p["uncorrected"], "precision_corrected": p["corrected"],
                     "precision_band_high": p["band_high"],
                     "recall_raw": r["uncorrected"], "recall_corrected": r["corrected"],
                     "n_A": p["n_A"], "n_B": p["n_B"], "n_U": p["n_U"],
                     "n_A_suspect": p["n_A_suspect"], "n_incremental": p["n_incremental"]})
        if p["n_A_suspect"]:
            print(f"{'':<22} note: {p['n_A_suspect']} of {p['n_A']} A-tags sit within "
                  f"{2.0:g} R of an already-detected ramp (likely a second hit, not a "
                  f"missed ramp)")
        if city in pool_of(args.cities, args.include_budapest, args.include_gold,
                           args.include_sao_paulo):
            pooled_items += items
            pooled_tags.update(tags)
            pooled["tp"] += rep.tp
            pooled["fp"] += rep.fp
            pooled["tp_recall"] += rep.n_gt_recall - rep.fn
            pooled["n_gt_recall"] += rep.n_gt_recall

    if pooled["tp"]:
        p = corrected_precision(pooled["tp"], pooled["fp"], pooled_items, pooled_tags)
        r = corrected_recall(pooled["tp_recall"], pooled["n_gt_recall"],
                             pooled_items, pooled_tags)
        print("-" * 96)
        print(f"{'POOLED':<22} {p['uncorrected']:>7.3f} {p['corrected']:>7.3f} "
              f"{p['band_high']:>8.3f} {r['uncorrected']:>7.3f} {r['corrected']:>7.3f} "
              f"{f1_of(p['uncorrected'], r['uncorrected']):>7.3f} "
              f"{f1_of(p['corrected'], r['corrected']):>8.3f}  "
              f"{p['n_A']}/{p['n_B']}/{p['n_U']}")
        rows.append({"split": "POOLED", "op_threshold": args.op_threshold,
                     "precision_raw": p["uncorrected"], "precision_corrected": p["corrected"],
                     "precision_band_high": p["band_high"],
                     "recall_raw": r["uncorrected"], "recall_corrected": r["corrected"],
                     "n_A": p["n_A"], "n_B": p["n_B"], "n_U": p["n_U"],
                     "n_A_suspect": p["n_A_suspect"], "n_incremental": p["n_incremental"]})
        print(f"\nA-rate pooled: {p['n_A']}/{p['n_incremental']} "
              f"({p['n_A'] / p['n_incremental']:.1%}) of incremental FPs in "
              f"[{args.op_threshold}, {args.upper}) were real ramps the GT missed.")

    _write_csv(os.path.join(args.out, f"corrected_at_{args.op_threshold:g}.csv"), rows,
               ["split", "op_threshold", "precision_raw", "precision_corrected",
                "precision_band_high", "recall_raw", "recall_corrected",
                "n_A", "n_B", "n_U", "n_A_suspect", "n_incremental"])
    print("\nThe corrected column credits confirmed A tags only; 'band hi' additionally "
          "credits\nunsure and untagged items, so it is the honest upper end rather than a "
          "formality.")
    return 0


# --------------------------------------------------------------------------- #
# gtbias — why sub-0.55 precision is a lower bound, measured rather than asserted
# --------------------------------------------------------------------------- #
def gt_origins(city, repo=REPO):
    """``{pano_id: ["reviewed"|"missed", ...]}`` aligned with each pano's gt_points.

    ``build_ground_truth`` appends the reviewer-confirmed *detections* first and the
    reviewer's *missed-ramp marks* second, so the origin of each GT point is
    recoverable from the same two files in that order. Recovering it is what turns
    the GT-completeness caveat from an assertion into a measurement.

    Verdict-reviewed splits only — ``manual_gold`` has no verdicts because its GT
    was labelled independently of RampNet, which is precisely why it is the control
    (see :func:`cmd_gtbias`).
    """
    with open(os.path.join(repo, "benchmark", city, "verdicts.json"),
              encoding="utf-8") as f:
        verdicts = json.load(f)["panos"]
    out = {}
    for pid, entry in verdicts.items():
        origins = ["reviewed" for v in entry["dets"] if v is True or v == "true"]
        origins += ["missed" for m in entry["missed"] if not m.get("unsure")]
        out[pid] = origins
    return out


def _bin_lo(score, bin_width):
    """Lower edge of ``score``'s bin, immune to binary-float floor errors.

    ``0.9 // 0.1`` is 8.0, not 9.0, because 0.9/0.1 evaluates to 8.999...; nudging
    the quotient by an epsilon before flooring fixes it. A score of exactly 1.0
    belongs to the top bin, not to a phantom bin above it, so the index is clamped.
    """
    top_index = math.ceil(round(1.0 / bin_width, 6)) - 1
    index = min(math.floor(score / bin_width + 1e-9), top_index)
    return round(max(index, 0) * bin_width, 4)


def tp_origin_by_bin(panos, origins, radius_sq, bin_width=0.1):
    """True positives per confidence bin, split by where their GT point came from.

    The load-bearing observation: the city splits' GT was assembled from detections
    at or above the deployed 0.55 floor, so **below that floor a prediction can only
    score as a TP if it lands on a ramp the reviewer independently marked as
    missed**. Anything else it finds — including a real curb ramp — is counted as a
    false positive because no human ever looked there.

    That makes sub-0.55 precision a lower bound with a *known mechanism* rather than
    an unquantified worry, and it is why issue #55's A/B spot-check is the only way
    to get a real number in that band.
    """
    counts = {}
    for pd in panos:
        origin = origins.get(pd["pano"])
        if origin is None or len(origin) != len(pd["gt"].gt_points):
            continue     # re-review drift: skip rather than mis-attribute
        preds = sorted(pd["preds"], key=lambda p: -p[2])
        assignments = greedy_match([(p[0], p[1]) for p in preds], pd["gt"].gt_points,
                                   radius_sq, PANO_SCALE_X, PANO_SCALE_Y)
        for p, (gi, _) in zip(preds, assignments):
            if gi < 0:
                continue
            counts[(_bin_lo(p[2], bin_width), origin[gi])] = counts.get(
                (_bin_lo(p[2], bin_width), origin[gi]), 0) + 1
    return counts


def cmd_gtbias(args):
    radius_sq = radius_sq_for()
    rows = []
    for city in args.cities:
        if city in ("manual_gold",):
            continue
        panos, _ = load_split(city, args.cache)
        counts = tp_origin_by_bin(panos, gt_origins(city), radius_sq, args.bin_width)
        print(f"\n{'=' * 72}\n{city.upper()} — true positives by GT origin\n{'=' * 72}")
        print(f"{'bin':>12} {'from reviewed det':>18} {'from missed mark':>17}")
        print("-" * 72)
        for b in sorted({k[0] for k in counts}):
            rv = counts.get((b, "reviewed"), 0)
            ms = counts.get((b, "missed"), 0)
            rows.append({"split": city, "bin_lo": b, "from_reviewed": rv,
                         "from_missed": ms})
            flag = "  <- below the 0.55 review floor" if b < DEPLOYED_THRESHOLD else ""
            print(f"{b:>7.1f}-{b + args.bin_width:<4.1f} {rv:>18} {ms:>17}{flag}")

    _write_csv(os.path.join(args.out, "tp_origin_by_bin.csv"), rows,
               ["split", "bin_lo", "from_reviewed", "from_missed"])
    print(f"""
{'=' * 72}
Reading this table
{'=' * 72}
Every true positive below {DEPLOYED_THRESHOLD} comes from a *missed mark*, and none from a
reviewed detection. That is structural, not a coincidence: the city splits' GT was
built from detections at or above the deployed floor, so in the band this sweep
opens up, a prediction can only be credited if a human independently flagged that
ramp during the missed-ramp pass. A real curb ramp nobody marked is counted as a
false positive.

So sub-{DEPLOYED_THRESHOLD} precision on these splits is a **lower bound with a known
mechanism**, and the measured F1-optimal threshold is biased *high* — the true
optimum sits at or below it. Issue #55's A/B spot-check is what converts the bound
into a number.

benchmark/manual_gold is the control: its GT was labelled independently of RampNet
(no verdict review, no anchoring), so its curve carries none of this bias at any
threshold. Compare its precision-vs-threshold shape against the city splits to see
the anchoring effect directly.""")
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
        sp.add_argument("--include-sao-paulo", action="store_true",
                        help="pool sao_paulo into POOLED/tier rows (default: held out — "
                             "non-US city, outside the US-deployment pooled basis)")

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

    sp = sub.add_parser("tta", help="flip-TTA vs single-pass at the operating points (#78)")
    common(sp)
    sp.add_argument("--tta-cache", default=TTA_CACHE_DIR,
                    help="cache dir of the extract --tta arm (default "
                         "analysis_out/op_cache_tta); manual_gold falls back to its "
                         "committed TTA-export records when no cache is present")
    sp.add_argument("--floor", type=float, default=0.05)
    sp.add_argument("--top", type=float, default=0.90)
    sp.add_argument("--candidate", type=float, default=0.30,
                    help="the proposed operating point to compare at (default 0.30, "
                         "per docs/operating_point.md)")
    sp.set_defaults(func=cmd_tta)

    sp = sub.add_parser("hist", help="GT-true vs GT-false confidence calibration (labeler#27)")
    common(sp)
    sp.add_argument("--floor", type=float, default=0.05)
    sp.add_argument("--bin-width", type=float, default=0.05)
    sp.set_defaults(func=cmd_hist)

    sp = sub.add_parser("tagcheck",
                        help="do the committed #55 A/B tags still resolve in this cache?")
    common(sp)
    sp.add_argument("--op-threshold", type=float, default=0.25)
    sp.add_argument("--upper", type=float, default=DEPLOYED_THRESHOLD)
    sp.add_argument("--min-resolved", type=float, default=1.0,
                    help="fraction of committed tags that must still resolve (default 1.0 "
                         "— any orphan is reviewer effort silently dropped)")
    sp.set_defaults(func=cmd_tagcheck)

    sp = sub.add_parser("floor",
                        help="storage-floor validation + recall ceiling (labeler#28/#27)")
    common(sp)
    sp.add_argument("--floor", type=float, default=STORAGE_FLOOR)
    sp.add_argument("--top-k", type=int, default=STORAGE_TOP_K)
    sp.set_defaults(func=cmd_floor)

    sp = sub.add_parser("corrected",
                        help="apply the #55 A/B tags -> corrected P/R, per split and pooled")
    common(sp)
    sp.add_argument("--op-threshold", type=float, default=0.35)
    sp.add_argument("--upper", type=float, default=DEPLOYED_THRESHOLD)
    sp.set_defaults(func=cmd_corrected)

    sp = sub.add_parser("gtbias", help="measure the GT-anchoring bias below the review floor")
    common(sp)
    sp.add_argument("--bin-width", type=float, default=0.1)
    sp.set_defaults(func=cmd_gtbias)

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
