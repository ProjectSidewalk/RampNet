"""Unit tests for scoring a filled location-precision review sheet (#96, #59).

Pure logic, no disk. The load-bearing guarantees: an unreviewed chip is not
silently counted as either readable or unreadable, a thin sheet is flagged
before its numbers are quoted, and no tier is ever assigned — the paper published
buckets without thresholds, and inventing one here would launder a judgment as a
measurement.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_precision_score as ips  # noqa: E402


def _sheet(records, city="x"):
    return {"city": city, "inventory": "x.jsonl.gz", "records": records}


def _judged(offset, visible=None, on_corner=True):
    return {"offset_m": offset, "ramps_visible": visible,
            "on_corner": on_corner, "unreadable": False}


# --------------------------------------------------------------------------- #
# partitioning
# --------------------------------------------------------------------------- #
def test_unscored_chips_are_counted_separately():
    """A half-finished sheet and a hard-to-read one are different problems."""
    judged, unreadable, unscored = ips.partition([
        _judged(1.0),
        {"offset_m": None, "unreadable": True},
        {"offset_m": None, "unreadable": False},
    ])
    assert (len(judged), len(unreadable), len(unscored)) == (1, 1, 1)


def test_unreadable_wins_over_a_stray_offset():
    judged, unreadable, _ = ips.partition([{"offset_m": 3.0, "unreadable": True}])
    assert len(judged) == 0 and len(unreadable) == 1


def test_completeness_is_reported():
    part = ips.score(_sheet([_judged(1.0), {"offset_m": None, "unreadable": False}]))
    assert part["complete"] is False and part["unscored"] == 1
    full = ips.score(_sheet([_judged(1.0), _judged(2.0)]))
    assert full["complete"] is True


# --------------------------------------------------------------------------- #
# readability gate
# --------------------------------------------------------------------------- #
def test_a_mostly_unreadable_sheet_is_flagged_as_insufficient():
    recs = [_judged(1.0)] + [{"offset_m": None, "unreadable": True} for _ in range(9)]
    got = ips.score(_sheet(recs))
    assert got["readable"]["share"] == 0.1
    assert got["readable"]["sufficient"] is False


def test_a_readable_sheet_passes_the_gate():
    recs = [_judged(1.0) for _ in range(8)] + [
        {"offset_m": None, "unreadable": True} for _ in range(2)]
    assert ips.score(_sheet(recs))["readable"]["sufficient"] is True


def test_an_empty_sheet_does_not_divide_by_zero():
    got = ips.score(_sheet([]))
    assert got["readable"]["share"] is None
    assert got["offset_m"]["quantiles"]["0.5"] is None


# --------------------------------------------------------------------------- #
# offsets
# --------------------------------------------------------------------------- #
def test_offset_shares_are_inclusive_of_the_radius():
    got = ips.score(_sheet([_judged(1.0), _judged(2.0), _judged(5.0), _judged(9.0)]))
    o = got["offset_m"]
    assert o["share_within_1m"] == 0.25
    assert o["share_within_2m"] == 0.5
    assert o["share_within_5m"] == 0.75


def test_offset_median_is_from_judged_chips_only():
    recs = [_judged(1.0), _judged(3.0), {"offset_m": None, "unreadable": True}]
    assert ips.score(_sheet(recs))["offset_m"]["quantiles"]["0.5"] == 2.0


# --------------------------------------------------------------------------- #
# ramps_visible — the per-ramp/per-corner evidence
# --------------------------------------------------------------------------- #
def test_ramps_visible_is_summarised():
    got = ips.score(_sheet([_judged(1.0, visible=2), _judged(1.0, visible=2),
                            _judged(1.0, visible=1)]))
    assert got["ramps_visible"]["histogram"] == {"1": 1, "2": 2}
    assert math.isclose(got["ramps_visible"]["mean"], 5 / 3.0)


def test_geometry_check_calls_out_under_recording():
    """~2 ramps seen where the inventory holds ~1.2 records means pairs merge."""
    scored = ips.score(_sheet([_judged(1.0, visible=2) for _ in range(10)]))
    got = ips.compare_to_geometry(scored, records_per_corner=1.21)
    assert got["visible_per_record"] > 1.25
    assert "under-recorded" in got["reading"]


def test_geometry_check_accepts_a_genuinely_single_ramp_city():
    scored = ips.score(_sheet([_judged(1.0, visible=1) for _ in range(10)]))
    got = ips.compare_to_geometry(scored, records_per_corner=1.05)
    assert "consistent" in got["reading"]


def test_geometry_check_is_none_without_counts():
    scored = ips.score(_sheet([_judged(1.0)]))
    assert ips.compare_to_geometry(scored, 1.2) is None


# --------------------------------------------------------------------------- #
# tiers and controls
# --------------------------------------------------------------------------- #
def test_no_tier_is_ever_assigned():
    got = ips.score(_sheet([_judged(0.1) for _ in range(50)]))
    assert got["tier"] is None
    assert "threshold" in got["tier_note"]


def test_control_comparison_reports_the_gap_not_a_verdict():
    cand = ips.score(_sheet([_judged(3.0) for _ in range(10)], city="denver"))
    ctrl = ips.score(_sheet([_judged(1.0) for _ in range(10)], city="bend"))
    got = ips.compare_to_control(cand, ctrl)
    assert got["control_city"] == "bend"
    assert math.isclose(got["median_offset_m"]["gap"], 2.0)
    assert "tier" not in got
