"""Tests for the street-review reduction (#103).

The claims pinned, each because getting it silently wrong changes a reported
number:

* **classify** excludes a disowned click (unreadable beats offset) and the
  denominators are the stated ones (phantom over judgeable, unjudgeable over
  all).
* **The gate membership is the TRUE asymmetric strip** — an offset between
  −18.458° and −18.368° is inside the strip but outside the symmetric bound,
  and both rates must reflect that.
* **The sign-flip null is a distribution, not zero** — §5i's twice-earned
  lesson: a mean well inside the null p95 must not read as significant.
* **The paired calibration projects the aerial VECTOR through the pano
  geometry** — a purely radial aerial offset must predict ~0°, and a
  tangential one must predict atan(offset/range) with the §5j sign.
"""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import street_review_summary as srsum  # noqa: E402
from street_review_sheet import STRIP_LEFT_DEG, STRIP_RIGHT_DEG  # noqa: E402
from stage1_offset_tolerance import crop_half_angle_deg  # noqa: E402


def _rec(rid="1", offset=None, unreadable=False, reason=None, no_ramp=False,
         stratum=None, **kw):
    r = {"id": rid, "lon": -104.99, "lat": 39.74, "stratum": stratum,
         "pano_id": "P" + rid, "pano_capture": "2021-3",
         "pano_heading_deg": 100.0, "pano_lat": 39.7401, "pano_lon": -104.99,
         "range_m": 11.1, "n_candidates": 5, "az_gov_deg": 90.0,
         "theta_deg": -10.0, "offset_deg": offset, "click_px": None,
         "unreadable": unreadable, "unreadable_reason": reason,
         "no_ramp": no_ramp, "note": ""}
    r.update(kw)
    return r


# --------------------------------------------------------------------------- #
# classification and denominators
# --------------------------------------------------------------------------- #
def test_classify_unreadable_beats_a_disowned_click():
    assert srsum.classify(_rec(offset=4.5, unreadable=True)) == "unjudgeable"
    assert srsum.classify(_rec(offset=0.0)) == "measured"
    assert srsum.classify(_rec(no_ramp=True)) == "phantom"
    assert srsum.classify(_rec()) == "todo"


def test_denominators_phantom_over_judgeable_unjudgeable_over_all():
    manifest = {"city": "x", "seed": 1, "sheet_build": "b", "records": [
        _rec("1", offset=1.0), _rec("2", offset=-2.0), _rec("3", no_ramp=True),
        _rec("4", unreadable=True, reason="van_or_vehicle"), _rec("5")]}
    s = srsum.summarise(manifest)
    assert s["classes"] == {"measured": 2, "phantom": 1, "unjudgeable": 1, "todo": 1}
    assert s["phantom"]["n_judgeable"] == 3
    assert s["phantom"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["unjudgeable"]["n"] == 5
    assert s["unjudgeable"]["rate"] == pytest.approx(1 / 5, abs=1e-4)
    assert s["unjudgeable"]["reasons"] == {"van_or_vehicle": 1}


def test_missing_reason_is_surfaced_not_hidden():
    out = srsum.reason_breakdown([_rec(unreadable=True, reason=None)])
    assert out == {"(missing)": 1}


# --------------------------------------------------------------------------- #
# the gate membership — true asymmetric edges
# --------------------------------------------------------------------------- #
def test_inside_strip_is_the_asymmetric_crop_not_the_symmetric_bound():
    # A point between the left edge (-18.458) and -crop_half (-18.368) is
    # inside the actual crop but outside the symmetric bound.
    between = -(crop_half_angle_deg() + 0.05)
    assert STRIP_LEFT_DEG < between < -crop_half_angle_deg()
    assert srsum.inside_strip(between)
    assert not srsum.inside_strip(STRIP_LEFT_DEG - 0.01)
    assert not srsum.inside_strip(STRIP_RIGHT_DEG + 0.01)
    assert srsum.inside_strip(0.0)

    records = [_rec("1", offset=between), _rec("2", offset=0.0),
               _rec("3", offset=25.0)]
    ang, offsets = srsum.angular_block(records, 3)
    assert ang["n_inside_strip"] == 2
    assert ang["frac_inside_strip"] == pytest.approx(2 / 3, abs=1e-4)
    # ...while the symmetric §5g/§5j-comparable rate excludes the sliver case.
    assert ang["frac_within_half_angle"] == pytest.approx(1 / 3, abs=1e-4)


def test_outside_view_records_count_against_the_gate_bound():
    """§5o gates on the BOUND, not the conditional rate.

    A `ramp_outside_view` record is unmeasurable but *certainly* outside a
    ±18.4° strip — it is the largest coordinate error the sample can hold.
    Scoring only over measured records would censor the sample in exactly the
    direction that makes the instrument pass, so the bound counts it as a
    failure. Occlusion unjudgeables must NOT be counted either way: they are
    missing at an unknown offset.
    """
    records = [_rec("1", offset=0.0), _rec("2", offset=2.0),
               _rec("3", unreadable=True, reason=srsum.OUTSIDE_VIEW_REASON),
               _rec("4", unreadable=True, reason="van_or_vehicle")]
    ang, _ = srsum.angular_block(records, len(records))

    assert srsum.n_outside_view(records) == 1          # the van does not count
    assert ang["n_inside_strip"] == 2
    assert ang["frac_inside_strip"] == pytest.approx(1.0)      # conditional
    assert ang["n_gate_denominator"] == 3                      # 2 measured + 1
    assert ang["frac_inside_strip_bound"] == pytest.approx(2 / 3, abs=1e-4)
    # The bound is never more optimistic than the conditional rate.
    assert ang["frac_inside_strip_bound"] <= ang["frac_inside_strip"]
    lo_b, hi_b = ang["frac_inside_strip_bound_ci"]
    assert lo_b <= ang["frac_inside_strip_bound"] <= hi_b


def test_gate_bound_equals_the_plain_rate_when_nothing_is_out_of_view():
    records = [_rec("1", offset=0.0), _rec("2", offset=25.0),
               _rec("3", unreadable=True, reason="sun_or_shadow")]
    ang, _ = srsum.angular_block(records, len(records))
    assert ang["n_outside_view"] == 0
    assert ang["frac_inside_strip_bound"] == ang["frac_inside_strip"]


def test_strip_edges_come_from_the_manifest_not_this_codes_constants():
    """A verdict is only interpretable against the rule that produced it, so
    re-reducing an old verdicts.json must use ITS edges, not today's."""
    manifest = {"city": "x", "seed": 1, "sheet_build": "b",
                "projection": {"strip_left_deg": -5.0, "strip_right_deg": 5.0},
                "records": [_rec("1", offset=4.0), _rec("2", offset=10.0)]}
    s = srsum.summarise(manifest)
    assert s["strip_edges_deg"] == [-5.0, 5.0]
    assert s["strip_edges_source"] == "manifest"
    # 10.0 is inside the real crop strip but outside the manifest's edges.
    assert srsum.inside_strip(10.0)
    assert s["angular"]["n_inside_strip"] == 1

    bare = {"city": "x", "seed": 1, "sheet_build": "b",
            "records": [_rec("1", offset=4.0), _rec("2", offset=10.0)]}
    s2 = srsum.summarise(bare)
    assert s2["strip_edges_deg"] == [round(STRIP_LEFT_DEG, 4),
                                     round(STRIP_RIGHT_DEG, 4)]
    assert "constants" in s2["strip_edges_source"]
    assert s2["angular"]["n_inside_strip"] == 2


def test_a_partially_reviewed_sheet_says_so_loudly():
    manifest = {"city": "x", "seed": 1, "sheet_build": "b", "records": [
        _rec("1", offset=1.0), _rec("2"), _rec("3")]}
    s = srsum.summarise(manifest)
    assert s["incomplete_review"] is True
    assert "REVIEW INCOMPLETE" in srsum.render(s)

    done = {"city": "x", "seed": 1, "sheet_build": "b", "records": [
        _rec("1", offset=1.0), _rec("2", offset=-1.0)]}
    assert srsum.summarise(done)["incomplete_review"] is False
    assert "REVIEW INCOMPLETE" not in srsum.render(srsum.summarise(done))


def test_angular_block_uses_summarize_columns():
    records = [_rec(str(i), offset=float(i)) for i in range(-3, 4)]
    ang, _ = srsum.angular_block(records, 10)
    for col in ("mean_deg", "se_mean_deg", "abs_median_deg", "abs_p90_deg",
                "matched_frac"):
        assert col in ang                       # §5j-comparable by construction
    assert ang["mean_deg"] == pytest.approx(0.0)
    assert ang["matched_frac"] == pytest.approx(7 / 10)


# --------------------------------------------------------------------------- #
# the sign-flip null
# --------------------------------------------------------------------------- #
def test_sign_flip_null_a_small_mean_is_not_significant():
    """§5i's lesson: at n=12 with |offsets|~2°, a 0.3° mean is deep inside the
    null — the p must say so, and the null p95 must be visibly non-zero."""
    offsets = [2.1, -1.8, 2.4, -2.2, 1.9, -2.0, 2.3, -1.7, 2.2, -2.1, 1.6, -2.4]
    r = srsum.sign_flip_null(offsets, draws=4000, seed=7)
    assert r["p_value"] > 0.5
    assert r["null_p95_abs_mean"] > 0.5        # nowhere near zero


def test_sign_flip_null_a_gross_shift_is_significant():
    offsets = [3.0 + 0.1 * i for i in range(20)]      # all clockwise
    r = srsum.sign_flip_null(offsets, draws=4000, seed=7)
    assert r["p_value"] < 0.01


def test_sign_flip_null_empty():
    assert srsum.sign_flip_null([])["p_value"] is None


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #
def test_strata_block_reports_per_stratum_and_none_when_absent():
    records = [_rec("1", offset=1.0, stratum="dated_before"),
               _rec("2", unreadable=True, reason="too_far", stratum="dated_before"),
               _rec("3", offset=30.0, stratum="undated")]
    st = srsum.strata_block(records)
    assert set(st) == {"dated_before", "undated"}
    assert st["dated_before"]["measured"] == 1
    assert st["dated_before"]["unjudgeable"] == 1
    assert st["undated"]["frac_inside_strip"] == 0.0
    assert srsum.strata_block([_rec("1")]) is None


# --------------------------------------------------------------------------- #
# paired calibration — the vector projection
# --------------------------------------------------------------------------- #
def _aerial(records):
    return {"metres_per_pixel": 0.1, "span_px": 400, "records": records}


def _aerial_rec(rid, click_px=None, offset_m=None, unreadable=False,
                no_ramp=False):
    return {"id": rid, "click_px": click_px, "offset_m": offset_m,
            "unreadable": unreadable, "no_ramp": no_ramp}


def test_radial_aerial_offset_predicts_zero_tangential_predicts_atan():
    """§5g: radial error is free. The record sits 11.1 m due EAST of the pano
    (az 90°); an aerial click displaced further EAST is radial -> ~0°; a
    displacement NORTH is tangential -> anticlockwise -> NEGATIVE, with
    magnitude atan(offset/range)."""
    east_m = 11.1 / (111320.0 * math.cos(math.radians(39.74)))
    street = _rec("1", offset=0.0, lon=-104.99 + east_m, lat=39.7401,
                  pano_lat=39.7401, pano_lon=-104.99)
    pred_radial = srsum.predicted_offset_deg(street, 2.0, 0.0)   # 2 m east
    assert abs(pred_radial) < 0.05
    pred_tang = srsum.predicted_offset_deg(street, 0.0, 2.0)     # 2 m north
    assert pred_tang == pytest.approx(-math.degrees(math.atan(2.0 / 11.1)), abs=0.3)


def test_paired_calibration_pairs_gates_and_cross_tabs():
    east_m = 11.1 / (111320.0 * math.cos(math.radians(39.74)))
    street = [
        # measured both sides; aerial click 20px right of centre = 2 m east
        _rec("1", offset=0.3, lon=-104.99 + east_m, lat=39.7401,
             pano_lat=39.7401, pano_lon=-104.99),
        # street measures what aerial could not see (the canopy argument)
        _rec("2", offset=1.0),
        # street unjudgeable where aerial measured (street's own bias)
        _rec("3", unreadable=True, reason="van_or_vehicle"),
        # phantom disagreement
        _rec("4", no_ramp=True),
    ]
    aerial = _aerial([
        _aerial_rec("1", click_px=[220.0, 200.0], offset_m=2.0),
        _aerial_rec("2", unreadable=True),
        _aerial_rec("3", click_px=[210.0, 200.0], offset_m=1.0),
        _aerial_rec("4", click_px=[200.0, 200.0], offset_m=0.0),
    ])
    pc = srsum.paired_calibration(street, aerial)
    assert pc["n_pairs"] == 1                   # only id 1 measured on both sides
    p = pc["pairs"][0]
    assert p["predicted_deg"] == pytest.approx(0.0, abs=0.1)   # radial -> free
    ct = pc["cross_tab"]
    assert ct["aerial_only_unjudgeable"]["ids"] == ["2"]
    assert ct["street_only_unjudgeable"]["ids"] == ["3"]
    assert ct["phantom_disagreements"]["ids"] == ["4"]


def test_phantom_disagreement_needs_BOTH_instruments_to_have_judged():
    """An aerial-unjudgeable record has no `no_ramp` to disagree with.

    Comparing anyway reads "the aerial sheet saw a ramp" from what is really
    "the aerial sheet could not look" — and the Denver pilot deliberately
    renders all 4 aerial unjudgeables, so this would have inflated the count at
    exactly the point §5o's criterion 4 gets read.
    """
    street = [_rec("1", no_ramp=True),                       # aerial unjudgeable
              _rec("2", unreadable=True, reason="van_or_vehicle"),  # street can't
              _rec("3", no_ramp=True)]                       # a REAL disagreement
    aerial = _aerial([
        _aerial_rec("1", unreadable=True),
        _aerial_rec("2", click_px=[200.0, 200.0], offset_m=0.0),
        _aerial_rec("3", click_px=[200.0, 200.0], offset_m=0.0),
    ])
    ct = srsum.paired_calibration(street, aerial)["cross_tab"]

    assert ct["aerial_only_unjudgeable"]["ids"] == ["1"]
    assert ct["street_only_unjudgeable"]["ids"] == ["2"]
    # id 1 is NOT a phantom disagreement (aerial never judged it), and id 2 is
    # not one either (street never judged it). Only id 3 is.
    assert ct["phantom_disagreements"]["ids"] == ["3"]


def test_paired_calibration_sign_agreement_only_above_floor():
    east_m = 11.1 / (111320.0 * math.cos(math.radians(39.74)))
    base = dict(lon=-104.99 + east_m, lat=39.7401, pano_lat=39.7401,
                pano_lon=-104.99)
    # Aerial click 50 px NORTH of centre = 5 m north = tangential, predicts
    # about -24 deg... no: atan(5/11.1) ~ -24.3? atan(0.45)=24.2 deg. Street
    # observed agrees in sign.
    street = [_rec("1", offset=-20.0, **base)]
    aerial = _aerial([_aerial_rec("1", click_px=[200.0, 150.0], offset_m=5.0)])
    pc = srsum.paired_calibration(street, aerial)
    assert pc["n_above_floor"] == 1
    assert pc["sign_agreement_above_floor"] == 1.0
    # A sub-floor prediction contributes a pair but no sign vote.
    aerial2 = _aerial([_aerial_rec("1", click_px=[200.0, 199.0], offset_m=0.1)])
    pc2 = srsum.paired_calibration(street, aerial2)
    assert pc2["n_pairs"] == 1 and pc2["n_above_floor"] == 0
    assert pc2["sign_agreement_above_floor"] is None


# --------------------------------------------------------------------------- #
# the convention trap
# --------------------------------------------------------------------------- #
def test_convention_check_fires_on_wrong_convention_numbers():
    bad = {"city": "x", "seed": 1, "sheet_build": "b", "records": [
        _rec(str(i), offset=88.0 + (i % 3)) for i in range(10)]}
    s = srsum.summarise(bad)
    assert s["convention_check"]["suspicious"]
    good = {"city": "x", "seed": 1, "sheet_build": "b", "records": [
        _rec(str(i), offset=float(i - 2)) for i in range(10)]}
    assert not srsum.summarise(good)["convention_check"]["suspicious"]


def test_render_produces_the_headline_lines():
    manifest = {"city": "denver-co", "seed": 20260731, "sheet_build": "abc",
                "status_counts": {"rendered": 3},
                "records": [_rec("1", offset=1.0), _rec("2", offset=-2.0),
                            _rec("3", offset=3.0)]}
    text = srsum.render(srsum.summarise(manifest))
    assert "INSIDE THE CROP STRIP" in text
    assert "sign-flip" in text
    assert "§5j corpus null" in text
