"""Unit tests for the per-ramp vs per-corner geometry gate (issues #96, #59).

Pure logic only — no network, no snapshot on disk. The load-bearing guarantees:
the neighbour search is exact within its block and *censors* rather than guessing
outside it, single-link clustering actually reproduces a known corner grouping
(this is what licenses running it on cities with no corner key), and the link
sweep separates "wider corner radii" from "no pairs recorded", which is the one
confound that could flip a city's verdict.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_geometry as ig  # noqa: E402


def _lonlat(x_m, y_m, lat0=40.0):
    """Inverse of to_local_metres, so tests can specify metres and get lon/lat."""
    mx = ig.METRES_PER_DEG_LAT * math.cos(math.radians(lat0))
    return (x_m / mx, y_m / ig.METRES_PER_DEG_LAT)


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def test_local_projection_round_trips_to_metres():
    pts = [_lonlat(0, 0), _lonlat(100, 0), _lonlat(0, 100)]
    xy, lat0 = ig.to_local_metres(pts, lat0=40.0)
    assert math.isclose(xy[1][0] - xy[0][0], 100.0, abs_tol=0.5)
    assert math.isclose(xy[2][1] - xy[0][1], 100.0, abs_tol=0.5)


def test_local_projection_of_empty_set_is_not_a_crash():
    assert ig.to_local_metres([]) == ([], 0.0)


# --------------------------------------------------------------------------- #
# nearest neighbour
# --------------------------------------------------------------------------- #
def test_nearest_neighbour_is_exact_within_the_block():
    xy = [(0, 0), (3, 0), (10, 0)]
    nn = ig.nearest_neighbour_distances(xy, cell=50.0)
    assert math.isclose(nn[0], 3.0, abs_tol=1e-9)
    assert math.isclose(nn[1], 3.0, abs_tol=1e-9)
    assert math.isclose(nn[2], 7.0, abs_tol=1e-9)


def test_isolated_point_is_censored_not_guessed():
    """A wrong large number would silently depress every share_within figure."""
    xy = [(0, 0), (5, 0), (10000, 10000)]
    nn = ig.nearest_neighbour_distances(xy, cell=50.0)
    assert nn[2] is None
    # ... and censoring must not inflate the share: it stays in the denominator.
    assert math.isclose(ig.share_within(nn, 6.0), 2 / 3.0)


def test_single_point_has_no_neighbour():
    assert ig.nearest_neighbour_distances([(0, 0)]) == [None]


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def test_single_link_groups_a_pair_and_separates_the_next_corner():
    # two ramps 3 m apart, next corner 25 m away with its own pair
    xy = [(0, 0), (3, 0), (25, 0), (28, 0)]
    groups = ig.single_link_clusters(xy, 6.0)
    assert sorted(len(g) for g in groups) == [2, 2]


def test_single_link_chains_transitively():
    """Documented behaviour: A-B-C at 5 m each is one group at a 6 m link."""
    xy = [(0, 0), (5, 0), (10, 0)]
    assert len(ig.single_link_clusters(xy, 6.0)) == 1
    assert len(ig.single_link_clusters(xy, 4.0)) == 3


def test_corner_recovery_scores_a_perfect_grouping():
    xy = [(0, 0), (3, 0), (25, 0), (28, 0)]
    ids = ["a", "a", "b", "b"]
    got = ig.score_corner_recovery(xy, ids, link_m=6.0)
    assert got["precision"] == 1.0 and got["recall"] == 1.0
    assert got["published_groups"] == 2 and got["geometric_groups"] == 2


def test_corner_recovery_penalises_an_over_merge():
    """One group where the publisher says two: recall holds, precision drops."""
    xy = [(0, 0), (3, 0), (6, 0)]
    got = ig.score_corner_recovery(xy, ["a", "a", "b"], link_m=6.0)
    assert got["recall"] == 1.0
    assert got["precision"] < 1.0


def test_corner_recovery_ignores_records_with_no_corner_id():
    xy = [(0, 0), (3, 0), (25, 0)]
    got = ig.score_corner_recovery(xy, ["a", "a", None], link_m=6.0)
    assert got["published_groups"] == 1


# --------------------------------------------------------------------------- #
# link sweep — the confound control
# --------------------------------------------------------------------------- #
def test_link_sweep_reveals_pairs_that_a_tight_link_misses():
    """A city whose pairs sit 9 m apart looks per-corner at 6 m and paired at 10 m.

    This is exactly the reading that would otherwise be mistaken for "records one
    point per corner", so the sweep has to make it visible.
    """
    xy = []
    for corner in range(4):
        base = corner * 200.0
        xy.extend([(base, 0.0), (base + 9.0, 0.0)])
    sweep = {row["link_m"]: row for row in ig.link_sweep(xy)}
    assert math.isclose(sweep[6.0]["records_per_group"], 1.0)
    assert math.isclose(sweep[10.0]["records_per_group"], 2.0)


def test_link_sweep_flags_when_groups_start_merging_across_the_intersection():
    """groups_per_intersection is the guard that says the ratio stopped meaning
    'per corner' — without it a rising records_per_group reads as good news."""
    # four corners of one intersection, 12 m apart, one record each
    xy = [(0, 0), (12, 0), (0, 12), (12, 12)]
    sweep = {row["link_m"]: row for row in ig.link_sweep(xy)}
    assert math.isclose(sweep[3.0]["groups_per_intersection"], 4.0)
    assert sweep[14.0]["groups_per_intersection"] < 2.0


# --------------------------------------------------------------------------- #
# summary statistics
# --------------------------------------------------------------------------- #
def test_histogram_buckets_are_half_open():
    counts = ig.histogram([0.0, 1.0, 1.999, 2.0], [0, 1, 2, 3])
    assert counts == [1, 2, 1]


def test_histogram_ignores_censored_values():
    assert ig.histogram([None, 1.5, None], [0, 1, 2]) == [0, 1]


def test_quantiles_interpolate():
    assert math.isclose(ig.quantiles([0.0, 10.0], qs=(0.5,))["0.5"], 5.0)


def test_quantiles_of_nothing_are_none_not_zero():
    assert ig.quantiles([None], qs=(0.5,))["0.5"] is None


def test_share_within_is_none_for_an_empty_set():
    assert ig.share_within([], 6.0) is None


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #
def test_epoch_ms_matches_known_arcgis_stamps():
    # the two values that actually appear in Denver's CREATEDATE column
    assert ig.epoch_ms_to_ym(1707091200000) == (2024, 2)
    assert ig.epoch_ms_to_ym(1636934400000) == (2021, 11)
    assert ig.epoch_ms_to_ym(0) == (1970, 1)


def test_epoch_ms_handles_leap_years():
    # 2020-02-29T00:00:00Z
    assert ig.epoch_ms_to_ym(1582934400000) == (2020, 2)


def test_epoch_ms_refuses_to_guess():
    assert ig.epoch_ms_to_ym(None) is None
    assert ig.epoch_ms_to_ym("not a date") is None


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_analyse_separates_a_paired_city_from_a_per_corner_one():
    paired, per_corner = [], []
    for corner in range(30):
        bx, by = (corner % 6) * 120.0, (corner // 6) * 120.0
        paired.extend([_lonlat(bx, by), _lonlat(bx + 3.0, by)])
        per_corner.append(_lonlat(bx, by))
    a = ig.analyse(paired)
    b = ig.analyse(per_corner)
    assert a["corner_clusters"]["records_per_group"] > 1.9
    assert b["corner_clusters"]["records_per_group"] == 1.0
    assert a["nearest_neighbour"]["share_within_6m"] > b["nearest_neighbour"]["share_within_6m"]


def test_analyse_reports_dates_when_a_field_is_given():
    pts = [_lonlat(0, 0), _lonlat(3, 0)]
    got = ig.analyse(pts, dates=[1707091200000, None])
    assert got["dates"]["by_year"] == {"2024": 1}
    assert got["dates"]["undated"] == 1
