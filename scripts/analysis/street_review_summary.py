"""Turn a street-level ``verdicts.json`` into the numbers #103 asks for.

The reduction step for ``street_review_sheet.py`` — the angular sibling of
``inventory_review_summary.py``, kept column-comparable with §5j on purpose:
the angular distribution is produced by **the same ``summarize()`` the
automatic bearing residual uses**, so a candidate city's human-reviewed row
reads directly against the corpus null (NYC +0.055°, Portland −0.250°, Bend
+0.036°; |median| 2.2–3.4°).

What comes out, each with its denominator stated:

* **The gate quantity**: the fraction of measured ramps INSIDE the strip Stage
  1 would cut — against the strip's true asymmetric edges (−18.458°/+18.368°),
  with the symmetric ``crop_half_angle_deg()`` rate alongside for §5g/§5j
  comparability. This is the number the aerial sheet could only reach through
  a Monte Carlo.
* **The angular distribution** over measured records (a record marked
  unjudgeable is excluded even if a click survived somewhere — "I cannot make
  a call" and "the call is +4.5°" are contradictory claims).
* **Phantom rate** over judgeable records; **unjudgeable rate** over all
  records **with its reason breakdown** — the street instrument's selection
  bias, measured as #103 requires, and the target list for a second-vantage
  pass.
* **Systematic shift**: the mean signed offset against a SIGN-FLIP null —
  never against zero (§5i's lesson, twice): the null distribution of |mean|
  under random signs is what a shift must clear.
* **Per-stratum rows** when the sheet was built with date strata — the
  summary-side support §5l had to reconstruct by hand.
* **Paired calibration** (``--aerial-verdicts``): for records the aerial sheet
  measured, the aerial offset VECTOR (recovered from ``click_px``) is
  projected through the chosen panorama's geometry into a predicted angular
  residual and compared with the street click. For Denver most predictions sit
  BELOW the ~1–2° click floor — the pass criterion there is the city-level
  read, and the pairing screens for gross per-record disagreement.

Wilson intervals throughout (imported from the aerial summary — one
definition). A convention self-check prints loudly when the numbers look like
a sign/wrap error (|median| near 90°, inside-rate near 0.10), the same trap
§5j builds into its own output.

    python scripts/analysis/street_review_summary.py \
        analysis_out/review_denver-co-gsv/verdicts.json \
        --aerial-verdicts analysis_out/review_denver-co/verdicts.json

Pure apart from file reading; arithmetic unit-tested in
``tests/test_street_review_summary.py``.
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_review_summary import percentile, wilson  # noqa: E402
from stage1_bearing_residual import fwd_azimuth_deg, summarize, wrap_deg  # noqa: E402
from street_review_sheet import STRIP_LEFT_DEG, STRIP_RIGHT_DEG  # noqa: E402
from stage1_offset_tolerance import crop_half_angle_deg  # noqa: E402


def classify(record):
    """measured / phantom / unjudgeable / todo — unreadable tested FIRST so a
    disowned click can never re-enter the distribution (the aerial summary's
    rule, inherited; the page clears such clicks anyway — belt and braces)."""
    if record.get("unreadable"):
        return "unjudgeable"
    if record.get("no_ramp"):
        return "phantom"
    if record.get("offset_deg") is not None:
        return "measured"
    return "todo"


def inside_strip(offset_deg):
    """The gate quantity's membership test: the TRUE asymmetric edges of the
    crop ``persp[:, 341:682]``. Not ±crop_half_angle_deg(), which is the
    conservative symmetric bound §5g/§5j quote — that rate is reported
    alongside, not silently substituted."""
    return STRIP_LEFT_DEG <= offset_deg <= STRIP_RIGHT_DEG


def sign_flip_null(offsets, draws=20000, seed=20260731):
    """The null for "is the city systematically shifted clockwise?".

    Under no systematic shift, each record's sign is a coin flip, so the null
    distribution of |mean| comes from flipping signs at random while keeping
    magnitudes — the angular analogue of the aerial summary's
    direction-randomisation, and the reason nobody here tests |mean| against
    ZERO: at n=40 the null p95 of |mean| is around 0.3·mean|x|, which is very
    far from 0 (§5i earned this lesson twice in one day). Pure given a seed.
    """
    n = len(offsets)
    if n == 0:
        return {"draws": 0, "p_value": None, "null_p95_abs_mean": None,
                "observed_mean": None}
    rng = random.Random(seed)
    mags = [abs(v) for v in offsets]
    observed = abs(sum(offsets) / n)
    hits, means = 0, []
    for _ in range(draws):
        m = sum(v if rng.random() < 0.5 else -v for v in mags) / n
        means.append(abs(m))
        if abs(m) >= observed:
            hits += 1
    means.sort()
    return {"draws": draws,
            "observed_mean": round(sum(offsets) / n, 4),
            "p_value": round(hits / draws, 4),
            "null_p95_abs_mean": round(percentile(means, 0.95), 4),
            "note": "null keeps magnitudes, randomises signs; a real shift "
                    "must clear the null p95, not zero"}


def reason_breakdown(records):
    """Counts of unjudgeable reasons — the instrument's own selection bias,
    and the target list for an alternate-pano pass."""
    out = {}
    for r in records:
        if r.get("unreadable"):
            key = r.get("unreadable_reason") or "(missing)"
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def angular_block(records, n_all_records):
    """The §5j-comparable distribution plus the gate rates, over measured
    records. ``matched_frac`` in the summarize() output reads here as
    "measured / all rendered records" — the human-instrument yield."""
    measured = [r for r in records if classify(r) == "measured"]
    offsets = [r["offset_deg"] for r in measured]
    panos = {r.get("pano_id") for r in measured}
    s = summarize(offsets, n_gov=n_all_records, n_matched=len(measured),
                  n_panos=len(panos))
    n = len(offsets)
    if n:
        k_in = sum(1 for o in offsets if inside_strip(o))
        k_half = sum(1 for o in offsets if abs(o) <= crop_half_angle_deg())
        s["n_inside_strip"] = k_in
        s["frac_inside_strip"] = round(k_in / n, 4)
        s["frac_inside_strip_ci"] = [round(v, 4) for v in wilson(k_in, n)]
        s["frac_within_half_angle"] = round(k_half / n, 4)
    return s, offsets


def strata_block(records):
    """Per-stratum rows when the sheet carried them — done in the summariser
    this time, instead of §5l's after-the-fact reconstruction."""
    strata = sorted({r.get("stratum") for r in records} - {None})
    if not strata:
        return None
    out = {}
    for name in strata:
        rs = [r for r in records if r.get("stratum") == name]
        measured = [r["offset_deg"] for r in rs if classify(r) == "measured"]
        judgeable = [r for r in rs if classify(r) in ("measured", "phantom")]
        out[name] = {
            "n": len(rs),
            "measured": len(measured),
            "abs_median_deg": (None if not measured else
                               round(percentile(sorted(abs(v) for v in measured), 0.5), 2)),
            "frac_inside_strip": (None if not measured else
                                  round(sum(1 for o in measured if inside_strip(o))
                                        / len(measured), 3)),
            "phantom": sum(1 for r in rs if classify(r) == "phantom"),
            "unjudgeable": sum(1 for r in rs if classify(r) == "unjudgeable"),
            "judgeable": len(judgeable),
        }
    return out


# --------------------------------------------------------------------------- #
# paired calibration against the aerial sheet
# --------------------------------------------------------------------------- #
def aerial_offset_vector(rec, metres_per_pixel, span_px):
    """(east_m, north_m) of the aerial click relative to the published
    coordinate — the same reconstruction ``inventory_review_summary.
    systematic_shift`` does, including the north sign flip (image y grows
    southward)."""
    if rec.get("unreadable") or rec.get("click_px") is None:
        return None
    cx, cy = rec["click_px"]
    c = span_px / 2.0
    return ((cx - c) * metres_per_pixel, -(cy - c) * metres_per_pixel)


def predicted_offset_deg(street_rec, east_m, north_m):
    """Push the aerial-measured offset vector through the chosen panorama's
    geometry: displace the record by the vector, and the prediction is the
    bearing change seen from the pano. Radial error predicts ~0° — §5g's
    'radial error is free' — so this is the honest per-record expectation,
    not |offset| rescaled."""
    lat, lon = street_rec["lat"], street_rec["lon"]
    plat, plon = street_rec["pano_lat"], street_rec["pano_lon"]
    lat2 = lat + north_m / 111132.0
    lon2 = lon + east_m / (111320.0 * math.cos(math.radians(lat)) or 1e-9)
    return wrap_deg(fwd_azimuth_deg(plat, plon, lat2, lon2)
                    - fwd_azimuth_deg(plat, plon, lat, lon))


def paired_calibration(street_records, aerial, floor_deg=2.0):
    """Per-record street-vs-aerial comparison, by id.

    Returns pairs, agreement stats, and the terminal-state cross-tab. The
    cross-tab is where Seattle's question lives: records the AERIAL instrument
    could not judge (canopy) that the street instrument measures are argument
    3 of #103 working; the reverse direction measures the street instrument's
    own selection bias against a known baseline.
    """
    mpp = aerial["metres_per_pixel"]
    span_px = aerial["span_px"]
    a_by_id = {str(r["id"]): r for r in aerial["records"]}

    pairs, cross = [], {"aerial_only_unjudgeable": [], "street_only_unjudgeable": [],
                        "both_unjudgeable": [], "phantom_disagreements": []}
    for s in street_records:
        a = a_by_id.get(str(s["id"]))
        if a is None:
            continue
        s_cls, a_unj = classify(s), bool(a.get("unreadable"))
        if a_unj and s_cls == "unjudgeable":
            cross["both_unjudgeable"].append(s["id"])
        elif a_unj and s_cls != "unjudgeable":
            cross["aerial_only_unjudgeable"].append(s["id"])
        elif not a_unj and s_cls == "unjudgeable":
            cross["street_only_unjudgeable"].append(s["id"])
        if bool(a.get("no_ramp")) != (s_cls == "phantom") and \
                (a.get("no_ramp") or s_cls == "phantom"):
            cross["phantom_disagreements"].append(s["id"])

        vec = aerial_offset_vector(a, mpp, span_px)
        if vec is None or s_cls != "measured":
            continue
        pred = predicted_offset_deg(s, *vec)
        pairs.append({"id": s["id"], "predicted_deg": round(pred, 2),
                      "observed_deg": s["offset_deg"],
                      "aerial_offset_m": a.get("offset_m"),
                      "range_m": s.get("range_m")})

    above = [p for p in pairs if abs(p["predicted_deg"]) > floor_deg]
    agree = sum(1 for p in above
                if (p["predicted_deg"] > 0) == (p["observed_deg"] > 0))
    resid = sorted(abs(p["predicted_deg"] - p["observed_deg"]) for p in pairs)
    gross = [p for p in pairs if abs(p["predicted_deg"] - p["observed_deg"]) > 10.0]
    return {
        "n_pairs": len(pairs),
        "floor_deg": floor_deg,
        "n_above_floor": len(above),
        "sign_agreement_above_floor": (round(agree / len(above), 3)
                                       if above else None),
        "abs_pred_minus_obs_median_deg": (round(percentile(resid, 0.5), 2)
                                          if resid else None),
        "gross_disagreements_over_10deg": gross,
        "cross_tab": {k: {"n": len(v), "ids": v} for k, v in cross.items()},
        "note": "predictions below the ~1-2 deg click floor are expected to "
                "disagree in sign; the floor-gated agreement is the "
                "diagnostic, the city-level read is the gate",
        "pairs": pairs,
    }


# --------------------------------------------------------------------------- #
def summarise(manifest, aerial=None):
    records = manifest["records"]
    n = len(records)
    by_class = {}
    for r in records:
        c = classify(r)
        by_class[c] = by_class.get(c, 0) + 1
    judgeable = by_class.get("measured", 0) + by_class.get("phantom", 0)

    angular, offsets = angular_block(records, n)
    out = {
        "city": manifest.get("city"),
        "seed": manifest.get("seed"),
        "sheet_build": manifest.get("sheet_build"),
        "instrument": manifest.get("instrument"),
        "n_records": n,
        "classes": by_class,
        # The build's own drop accounting, restated so the yield reads next to
        # the verdict rates rather than in a different file.
        "site_status_counts": manifest.get("status_counts"),
        "angular": angular,
        "systematic": sign_flip_null(offsets),
        "phantom": {
            "k": by_class.get("phantom", 0), "n_judgeable": judgeable,
            "rate": (round(by_class.get("phantom", 0) / judgeable, 4)
                     if judgeable else None),
            "ci": [round(v, 4) for v in wilson(by_class.get("phantom", 0),
                                               judgeable)] if judgeable else None},
        "unjudgeable": {
            "k": by_class.get("unjudgeable", 0), "n": n,
            "rate": round(by_class.get("unjudgeable", 0) / n, 4) if n else None,
            "ci": [round(v, 4) for v in wilson(by_class.get("unjudgeable", 0), n)]
                  if n else None,
            "reasons": reason_breakdown(records)},
        "strata": strata_block(records),
    }
    if aerial is not None:
        out["paired_calibration"] = paired_calibration(records, aerial)

    # The §5j-style convention trap: a wrong sign/wrap convention reads as
    # |median| ~90 deg with ~10% inside the strip. Loud, not subtle.
    am = angular.get("abs_median_deg")
    fi = angular.get("frac_inside_strip")
    out["convention_check"] = {
        "suspicious": bool(am is not None and (am > 45.0 or (fi is not None and fi < 0.3))),
        "note": "a wrong azimuth convention reads as |median|~90 deg and "
                "~10% inside the strip (cf. stage1_bearing_residual §5j)"}
    return out


def render(s):
    lines = []
    a = lines.append
    a("street-level review — {} (seed {}, build {})".format(
        s["city"], s["seed"], s["sheet_build"]))
    a("records {}  classes {}".format(s["n_records"], s["classes"]))
    if s.get("site_status_counts"):
        a("build statuses {}".format(s["site_status_counts"]))
    ang = s["angular"]
    if ang.get("insufficient"):
        a("angular: insufficient measured records ({})".format(ang["n_residuals"]))
    else:
        a("angular (n={}): mean {:+.2f}° (s.e. {:.2f})  |median| {:.2f}°  "
          "p90 {:.2f}°".format(ang["n_residuals"], ang["mean_deg"],
                               ang["se_mean_deg"], ang["abs_median_deg"],
                               ang["abs_p90_deg"]))
        a("  §5j corpus null for scale: NYC +0.055° / Portland -0.250° / "
          "Bend +0.036°; |median| 2.2-3.4°")
        a("  INSIDE THE CROP STRIP: {}/{} = {:.1%}  (CI {:.1%}-{:.1%}; "
          "within ±{:.2f}°: {:.1%})".format(
              ang["n_inside_strip"], ang["n_residuals"], ang["frac_inside_strip"],
              ang["frac_inside_strip_ci"][0], ang["frac_inside_strip_ci"][1],
              crop_half_angle_deg(), ang["frac_within_half_angle"]))
    sy = s["systematic"]
    if sy["p_value"] is not None:
        a("systematic shift: mean {:+.2f}°, sign-flip p = {}  "
          "(null p95 |mean| = {}°)".format(sy["observed_mean"], sy["p_value"],
                                           sy["null_p95_abs_mean"]))
    ph, un = s["phantom"], s["unjudgeable"]
    if ph["rate"] is not None:
        a("phantom: {}/{} judgeable = {:.1%}  [{:.1%}-{:.1%}]".format(
            ph["k"], ph["n_judgeable"], ph["rate"], ph["ci"][0], ph["ci"][1]))
    a("unjudgeable: {}/{} = {:.1%}  [{:.1%}-{:.1%}]  reasons {}".format(
        un["k"], un["n"], un["rate"], un["ci"][0], un["ci"][1], un["reasons"]))
    if s.get("strata"):
        a("strata:")
        for name, row in s["strata"].items():
            a("  {:>12s}: n {}  measured {}  |median| {}°  inside {}  "
              "phantom {}  unjudgeable {}".format(
                  name, row["n"], row["measured"], row["abs_median_deg"],
                  row["frac_inside_strip"], row["phantom"], row["unjudgeable"]))
    pc = s.get("paired_calibration")
    if pc:
        a("paired vs aerial: {} pairs, {} above the {}° floor, sign agreement "
          "{}  |pred-obs| median {}°".format(
              pc["n_pairs"], pc["n_above_floor"], pc["floor_deg"],
              pc["sign_agreement_above_floor"],
              pc["abs_pred_minus_obs_median_deg"]))
        ct = pc["cross_tab"]
        a("  cross-tab: aerial-only unjudgeable {}  street-only {}  both {}  "
          "phantom disagreements {}".format(
              ct["aerial_only_unjudgeable"]["n"], ct["street_only_unjudgeable"]["n"],
              ct["both_unjudgeable"]["n"], ct["phantom_disagreements"]["n"]))
        if pc["gross_disagreements_over_10deg"]:
            a("  GROSS (>10°): {}".format(
                [(p["id"], p["predicted_deg"], p["observed_deg"])
                 for p in pc["gross_disagreements_over_10deg"]]))
    if s["convention_check"]["suspicious"]:
        a("!! CONVENTION CHECK FAILED: " + s["convention_check"]["note"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verdicts", help="street sheet verdicts.json (reviewed)")
    ap.add_argument("--aerial-verdicts", default=None,
                    help="the aerial sheet's verdicts.json for the same "
                         "records — enables the per-record paired calibration")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    with open(args.verdicts, encoding="utf-8") as fh:
        manifest = json.load(fh)
    aerial = None
    if args.aerial_verdicts:
        with open(args.aerial_verdicts, encoding="utf-8") as fh:
            aerial = json.load(fh)

    s = summarise(manifest, aerial)
    print(render(s))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=2)
            fh.write("\n")
        print("\nwrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
