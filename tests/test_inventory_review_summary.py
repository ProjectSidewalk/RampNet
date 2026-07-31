"""Unit tests for the review-verdict reduction (issues #96, #59).

The numbers this produces are what decide whether a city's inventory joins the
Stage 1 corpus, so the arithmetic that is easy to get quietly wrong is what gets
pinned here: which chips land in which denominator, and that a click the reviewer
disowned cannot re-enter the distribution through a side door.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_summary as irs  # noqa: E402


def _rec(rid, **kw):
    base = {"id": rid, "offset_m": None, "ramps_visible": None, "on_corner": None,
            "unreadable": False, "no_ramp": False, "note": "",
            "published_within_6m": 1, "published_within_10m": 1}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# classification -- which denominator a chip lands in
# --------------------------------------------------------------------------- #
def test_unjudgeable_beats_a_stray_click():
    """The real case this exists for: Denver chip 98816 carried a 4.54 m click AND
    unjudgeable, with the note "very hard to tell ... they look way off". Counting
    it would have put a measurement the reviewer explicitly disowned at the very
    top of the distribution, where one value moves p90 and the max."""
    assert irs.classify(_rec("98816", offset_m=4.54, unreadable=True)) == "unjudgeable"


def test_phantom_beats_a_stray_click_too():
    assert irs.classify(_rec("x", offset_m=1.0, no_ramp=True)) == "phantom"


def test_a_plain_click_is_a_measurement():
    assert irs.classify(_rec("x", offset_m=1.0)) == "measured"


def test_an_untouched_chip_is_todo():
    assert irs.classify(_rec("x")) == "todo"
    assert irs.classify(_rec("x", ramps_visible=2)) == "todo"


# --------------------------------------------------------------------------- #
# denominators
# --------------------------------------------------------------------------- #
def test_phantom_rate_excludes_unjudgeable_chips():
    """An unjudgeable chip is not evidence of a ramp being present OR absent, so
    it cannot sit in the phantom denominator. Putting it there would understate
    the rate by diluting it."""
    s = irs.summarise({"records": [
        _rec("a", no_ramp=True), _rec("b", offset_m=0.2), _rec("c", offset_m=0.3),
        _rec("d", unreadable=True), _rec("e", unreadable=True),
    ]})
    assert s["phantom"]["n"] == 1
    assert s["phantom"]["of_judgeable"] == 3          # not 5
    assert math.isclose(s["phantom"]["rate"], 1 / 3)
    assert s["unjudgeable"]["of"] == 5                # unjudgeable IS over all chips


def test_disowned_clicks_are_reported_not_silently_dropped():
    """Excluding them is right; hiding that they existed is not."""
    s = irs.summarise({"records": [
        _rec("a", offset_m=4.54, unreadable=True, note="looks way off"),
        _rec("b", offset_m=0.3),
    ]})
    assert s["offset"]["n"] == 1
    assert s["excluded_clicks"] == [
        {"id": "a", "offset_m": 4.54, "note": "looks way off"}]


def test_unfinished_chips_are_named():
    s = irs.summarise({"records": [_rec("a", offset_m=0.1), _rec("b")]})
    assert s["todo"] == ["b"] and s["reviewed"] == 1


def test_uncounted_chips_are_named_separately_from_unfinished():
    """A chip can be measured but have its count toggled off -- clicking an
    already-selected segment clears it -- which is invisible in the progress
    counter because the offset alone marks it done."""
    s = irs.summarise({"records": [_rec("a", offset_m=0.1)]})
    assert s["todo"] == [] and s["uncounted"] == ["a"]


# --------------------------------------------------------------------------- #
# per-corner comparison
# --------------------------------------------------------------------------- #
def test_a_count_inside_the_bracket_is_consistent():
    """A radius is not a corner: 6 m splits a large one and crosses a slip lane.
    Only a count outside [p6, p10] is evidence."""
    s = irs.summarise({"records": [
        _rec("a", ramps_visible=3, offset_m=0.1,
             published_within_6m=2, published_within_10m=4)]})
    assert s["per_corner"]["consistent"] == 1
    assert s["per_corner"]["disagreements"] == []


def test_counts_outside_the_bracket_are_flagged_with_direction():
    s = irs.summarise({"records": [
        _rec("hi", ramps_visible=5, offset_m=0.1,
             published_within_6m=1, published_within_10m=2),
        _rec("lo", ramps_visible=0, no_ramp=True,
             published_within_6m=2, published_within_10m=2),
    ]})
    pc = s["per_corner"]
    assert pc["more_than_published"] == 1 and pc["fewer_than_published"] == 1
    kinds = {d["id"]: d["kind"] for d in pc["disagreements"]}
    assert kinds == {"hi": "more_than_published", "lo": "fewer_than_published"}
    assert next(d for d in pc["disagreements"] if d["id"] == "lo")["phantom"] is True


# --------------------------------------------------------------------------- #
# systematic shift -- bad inventory vs bad basemap
# --------------------------------------------------------------------------- #
def _clicked(rid, dx_px, dy_px, span=698, mpp=0.0573, **kw):
    C = span / 2.0
    return _rec(rid, offset_m=math.hypot(dx_px, dy_px) * mpp,
                click_px=[C + dx_px, C + dy_px], **kw)


def test_random_directions_cancel():
    """Genuine positional imprecision points every way, so the mean VECTOR goes
    to zero while the mean MAGNITUDE does not. Denver measures 24%."""
    recs = [_clicked("a", 20, 0), _clicked("b", -20, 0),
            _clicked("c", 0, 20), _clicked("d", 0, -20)]
    s = irs.systematic_shift(recs, 0.0573, 698)
    assert s["n"] == 4
    assert abs(s["resultant_m"]) < 1e-9
    assert s["mean_magnitude_m"] > 1.0
    assert s["systematic_share"] < 1e-9


def test_a_uniform_displacement_does_not_cancel():
    """A datum or projection error moves every ramp the same way, so resultant
    and magnitude converge. Seattle's first 11 chips measure 87%."""
    recs = [_clicked(str(i), -35, 0) for i in range(6)]
    s = irs.systematic_shift(recs, 0.0573, 698)
    assert s["systematic_share"] > 0.99
    assert s["mean_east_m"] < 0          # ramp west of the published point
    assert s["east_positive"] == 0


def test_north_is_up_in_the_reported_vector():
    """Screen y grows downward. Getting this backwards would report a shift in
    exactly the wrong direction, which is worse than reporting none."""
    s = irs.systematic_shift([_clicked("a", 0, -20)], 0.0573, 698)
    assert s["mean_north_m"] > 0


def test_disowned_and_unclicked_chips_are_excluded_from_the_shift():
    recs = [_clicked("a", -35, 0),
            _clicked("b", -35, 0, unreadable=True),   # disowned
            _rec("c", offset_m=1.0)]                  # no click_px
    assert irs.systematic_shift(recs, 0.0573, 698)["n"] == 1


def test_shift_is_none_when_nothing_has_been_clicked():
    assert irs.systematic_shift([_rec("a")], 0.0573, 698) is None


def test_shift_reaches_the_summary_payload():
    s = irs.summarise({"metres_per_pixel": 0.0573, "span_px": 698,
                       "records": [_clicked("a", -35, 0), _clicked("b", -35, 0)]})
    assert s["systematic_shift"]["systematic_share"] > 0.99


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_percentiles_interpolate_and_bracket_the_data():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert irs.percentile(vals, 0.0) == 0.0
    assert irs.percentile(vals, 1.0) == 4.0
    assert math.isclose(irs.percentile(vals, 0.5), 2.0)
    assert math.isclose(irs.percentile(vals, 0.25), 1.0)


def test_wilson_interval_never_leaves_the_unit_range():
    """The reason it is used at all: at n=54 with 3 successes the normal
    approximation runs below zero, which cannot be printed honestly."""
    lo, hi = irs.wilson(3, 54)
    assert 0.0 < lo < 3 / 54 < hi < 1.0
    assert irs.wilson(0, 30)[0] == 0.0
    assert irs.wilson(30, 30)[1] == 1.0


def test_wilson_interval_tightens_as_the_sample_grows():
    narrow = irs.wilson(50, 500)
    wide = irs.wilson(5, 50)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_empty_review_does_not_divide_by_zero():
    s = irs.summarise({"records": []})
    assert s["offset"]["n"] == 0 and s["phantom"]["rate"] is None


# --------------------------------------------------------------------------- #
# the systematic-shift null (§5i)
# --------------------------------------------------------------------------- #
def test_null_share_is_not_zero_at_small_n():
    """THE correction. Random directions do not give share ~0; they give
    ~0.9/sqrt(n). Reading a raw share as though 0 were the null is what turned
    eleven Seattle chips into a 'registration error' that did not exist."""
    n = 11
    nul = irs.systematic_shift_null([1.0] * n, observed_share=0.0, draws=3000)
    assert 0.6 / math.sqrt(n) < nul["median_share"] < 1.2 / math.sqrt(n)


def test_null_median_falls_as_the_sample_grows():
    small = irs.systematic_shift_null([1.0] * 10, 0.0, draws=2000)["median_share"]
    large = irs.systematic_shift_null([1.0] * 200, 0.0, draws=2000)["median_share"]
    assert large < small / 2


def test_a_genuinely_shifted_sample_is_improbable_under_the_null():
    nul = irs.systematic_shift_null([2.0] * 20, observed_share=1.0, draws=2000)
    assert nul["p_value"] < 0.01


def test_a_share_at_the_null_median_is_unremarkable():
    mags = [1.0] * 12
    med = irs.systematic_shift_null(mags, 0.0, draws=3000)["median_share"]
    assert irs.systematic_shift_null(mags, med, draws=3000)["p_value"] > 0.3


def test_null_is_reproducible_under_its_seed():
    a = irs.systematic_shift_null([1.0, 3.0, 0.5], 0.5, draws=500, seed=7)
    b = irs.systematic_shift_null([1.0, 3.0, 0.5], 0.5, draws=500, seed=7)
    assert a["p_value"] == b["p_value"] and a["median_share"] == b["median_share"]


def test_one_huge_offset_makes_a_high_share_easy_to_reach_by_chance():
    """Why the null keeps the observed magnitudes rather than equal ones: a
    heavy tail fakes a shift far more readily, and Seattle's sample ran
    0.21 m to 8.79 m."""
    even = irs.systematic_shift_null([1.0] * 8, 0.0, draws=3000)["median_share"]
    heavy = irs.systematic_shift_null([1.0] * 7 + [20.0], 0.0, draws=3000)["median_share"]
    assert heavy > even


def test_null_refuses_degenerate_input():
    assert irs.systematic_shift_null([], 0.5) is None
    assert irs.systematic_shift_null([0.0, 0.0], 0.5) is None


def test_summary_attaches_a_null_to_every_shift_it_reports():
    recs = [{"id": str(i), "click_px": [100 + i, 100], "offset_m": 1.0}
            for i in range(6)]
    sh = irs.systematic_shift(recs, metres_per_pixel=0.1, span_px=200)
    assert sh["null"] is not None and 0.0 <= sh["null"]["p_value"] <= 1.0
