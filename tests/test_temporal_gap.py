"""Unit tests for the temporal-gap sourcing gate (issues #59, #86).

Pure logic only — no network, no imagery, no tracker snapshot on disk. The
load-bearing guarantees: the date parser refuses to guess (so the undated
fraction means something), the ordering check reproduces the pipeline's own
semantics including the #11 regression, and the two exposures stay on opposite
signs instead of quietly netting out.
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import temporal_gap as tg  # noqa: E402


# --------------------------------------------------------------------------- #
# parse_ym
# --------------------------------------------------------------------------- #
def test_parse_ym_handles_every_format_these_sources_emit():
    assert tg.parse_ym("2019-05-14") == (2019, 5)          # ISO date
    assert tg.parse_ym("2019-05-14T00:00:00Z") == (2019, 5)  # ISO datetime (Socrata)
    assert tg.parse_ym("2019-05") == (2019, 5)
    assert tg.parse_ym("2019/05/14") == (2019, 5)
    assert tg.parse_ym("2019") == (2019, 1)                # bare year
    assert tg.parse_ym(2019) == (2019, 1)


def test_parse_ym_reads_arcgis_epoch_milliseconds():
    # ArcGIS FeatureServer emits epoch ms; 1557792000000 == 2019-05-14 UTC.
    assert tg.parse_ym(1557792000000) == (2019, 5)
    assert tg.parse_ym("1557792000000") == (2019, 5)


def test_parse_ym_returns_none_rather_than_guessing():
    # The undated fraction is only meaningful if this never invents a date.
    for bad in (None, "", "   ", "n/a", "unknown", "not a date", "0000-00-00",
                "2019-13-01", [], {}):
        assert tg.parse_ym(bad) is None, bad


# --------------------------------------------------------------------------- #
# ordering_passes — mirrors generate_dataset_meta.py
# --------------------------------------------------------------------------- #
def test_ordering_regression_from_issue_11():
    # The old check compared months without years, so a ramp installed 2019-05
    # was wrongly rejected against a 2020-03 pano (5 >= 3). It predates it.
    assert tg.ordering_passes((2019, 5), (2020, 3)) is True


def test_ordering_requires_strictly_before_the_capture_month():
    assert tg.ordering_passes((2020, 2), (2020, 3)) is True
    assert tg.ordering_passes((2020, 3), (2020, 3)) is False   # same month rejected
    assert tg.ordering_passes((2020, 4), (2020, 3)) is False


def test_undated_records_follow_the_pipeline_flag():
    assert tg.ordering_passes(None, (2020, 3), undated_predates=True) is True
    assert tg.ordering_passes(None, (2020, 3), undated_predates=False) is False


# --------------------------------------------------------------------------- #
# build_hist
# --------------------------------------------------------------------------- #
def test_sentinel_counts_as_undated_and_is_reported_separately():
    hist, n_undated, n_sentinel = tg.build_hist(
        ["2019-05-01", "2000-01-01", "2000-01-01", None, "garbage"])
    assert hist == Counter({(2019, 5): 1})
    assert n_sentinel == 2
    assert n_undated == 4          # 2 sentinels + None + garbage


def test_capture_dates_do_not_apply_the_sentinel_rule():
    # A 2000-01 capture date is a real (ancient) panorama, not a lossy conversion.
    hist, _, n_sentinel = tg.build_hist(["2000-01-15"], sentinel=None)
    assert hist == Counter({(2000, 1): 1})
    assert n_sentinel == 0


# --------------------------------------------------------------------------- #
# phantom_rate — undated records admitted at empty pixels
# --------------------------------------------------------------------------- #
def test_no_undated_records_means_no_phantoms():
    ihist = Counter({(2015, 1): 100})
    chist = Counter({(2020, 1): 50})
    assert tg.phantom_rate(ihist, chist, n_undated=0) == 0.0


def test_phantoms_vanish_when_undated_records_are_discarded_instead():
    ihist = Counter({(2025, 1): 100})
    chist = Counter({(2020, 1): 50})
    assert tg.phantom_rate(ihist, chist, 100, undated_predates=False) == 0.0


def test_phantom_rate_is_bounded_by_the_undated_share():
    ihist = Counter({(2030, 1): 100})      # every dated install postdates capture
    chist = Counter({(2020, 1): 50})
    # Worst case: all undated records are phantoms -> rate == undated fraction.
    assert tg.phantom_rate(ihist, chist, 100) == 0.5
    # And it can never exceed that share.
    ihist2 = Counter({(2010, 1): 50, (2030, 1): 50})
    assert tg.phantom_rate(ihist2, chist, 100) <= 0.5


def test_phantom_rate_rises_as_installs_shift_later_than_the_imagery():
    chist = Counter({(2020, 1): 100})
    early = tg.phantom_rate(Counter({(2010, 1): 100}), chist, 100)
    late = tg.phantom_rate(Counter({(2030, 1): 100}), chist, 100)
    assert late > early


# --------------------------------------------------------------------------- #
# missing_exposure — ramps built after the snapshot
# --------------------------------------------------------------------------- #
def test_no_exposure_when_the_inventory_postdates_all_imagery():
    m = tg.missing_exposure(Counter({(2020, 1): 30}), Counter({(2015, 1): 100}),
                            snapshot_ym=(2024, 1), inventory_size=30)
    assert m["mean_gap_years"] == 0.0
    assert m["share_imagery_after_snapshot"] == 0.0


def test_dc_shaped_case_a_static_snapshot_against_far_newer_imagery():
    # DC: 2016 capture, imagery mostly 2022-23 -> a ~6 year one-directional gap.
    ihist = Counter({(2013, 1): 300, (2014, 1): 300, (2015, 1): 300})
    chist = Counter({(2022, 1): 800, (2023, 1): 200})
    m = tg.missing_exposure(ihist, chist, snapshot_ym=(2016, 1),
                            inventory_size=900, lookback=3)
    assert 6.0 <= m["mean_gap_years"] <= 7.0
    assert m["share_imagery_after_snapshot"] == 1.0
    assert m["install_rate_per_year"] == 300.0        # 900 installs / 3 years
    assert m["est_missing_ramps"] > 1800              # 300/yr x ~6.2 yr
    assert m["est_missing_pct_of_inventory"] > 100    # more missing than recorded


def test_unknown_build_rate_is_reported_as_unknown_not_zero():
    # No dated installs inside the lookback window -> the estimate is not 0.
    m = tg.missing_exposure(Counter({(1990, 1): 50}), Counter({(2022, 1): 10}),
                            snapshot_ym=(2016, 1), inventory_size=50, lookback=3)
    assert m["install_rate_per_year"] == 0.0
    assert m["est_missing_ramps"] is None
    assert m["est_missing_pct_of_inventory"] is None


# --------------------------------------------------------------------------- #
# discard_rate
# --------------------------------------------------------------------------- #
def test_discard_rate_bounds():
    chist = Counter({(2020, 1): 100})
    # Everything predates the imagery -> nothing discarded.
    assert tg.discard_rate(Counter({(2010, 1): 100}), chist, 0) == 0.0
    # Everything postdates it -> all dated pairs discarded.
    assert tg.discard_rate(Counter({(2030, 1): 100}), chist, 0) == 1.0


def test_undated_records_are_discarded_only_when_the_flag_is_off():
    chist = Counter({(2020, 1): 100})
    ihist = Counter({(2010, 1): 50})
    assert tg.discard_rate(ihist, chist, 50, undated_predates=True) == 0.0
    assert tg.discard_rate(ihist, chist, 50, undated_predates=False) == 0.5


# --------------------------------------------------------------------------- #
# median / summarize
# --------------------------------------------------------------------------- #
def test_median_ym_is_count_weighted():
    assert tg.median_ym(Counter({(2018, 1): 1, (2022, 1): 100})) == (2022, 1)
    assert tg.median_ym(Counter()) is None


def test_summarize_end_to_end_keeps_the_two_exposures_separate():
    installs = ["2013-01-01", "2014-06-01", "2015-03-01", None, "", "2000-01-01"]
    captures = ["2022-05-01"] * 8 + ["2023-01-01"] * 2
    rep = tg.summarize(installs, captures, snapshot_ym=(2016, 1))

    assert rep["inventory_size"] == 6
    assert rep["n_dated"] == 3
    assert rep["n_undated"] == 3            # None, "", and the sentinel
    assert rep["n_sentinel"] == 1
    assert abs(rep["undated_fraction"] - 0.5) < 1e-9
    assert rep["n_panos"] == 10
    assert rep["median_capture"] == (2022, 5)

    # Both exposures are live here, and neither is allowed to cancel the other.
    assert rep["missing"]["est_missing_ramps"] > 0
    assert rep["phantom_rate"] == 0.0       # all dated installs predate capture
    assert rep["ordering_discard_rate"] == 0.0
    assert "net" not in rep                 # there is deliberately no net number


def test_summarize_defaults_snapshot_to_the_newest_install():
    rep = tg.summarize(["2013-01-01", "2019-07-01"], ["2020-01-01"])
    assert rep["missing"]["inventory_snapshot"] == (2019, 7)


def test_summarize_survives_an_empty_inventory():
    rep = tg.summarize([], ["2022-01-01"])
    assert rep["inventory_size"] == 0
    assert rep["undated_fraction"] == 0.0
    assert rep["phantom_rate"] == 0.0
