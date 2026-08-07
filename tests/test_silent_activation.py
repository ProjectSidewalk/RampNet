"""Unit tests for the silent-miss activation forensics (#46, Phase 1).

Pure core only — no torch, no imagery, no GPU. The heavy path (model inference)
is exercised by running the script itself; what these protect is the geometry:
``radius_max`` must read the heatmap through exactly the matcher's coordinate
convention, or the activation numbers describe the wrong locations.
"""
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import silent_activation as sa  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
R = RSQ ** 0.5  # 22.5 heatmap px


def _heat(value=0.0):
    return [[value] * 1024 for _ in range(512)]


# --------------------------------------------------------------------------- #
# radius_max — the matcher's window, applied to the heatmap
# --------------------------------------------------------------------------- #
def test_reads_a_peak_at_the_site():
    h = _heat()
    h[256][512] = 0.7
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == pytest.approx(0.7)


def test_ignores_a_peak_outside_the_radius():
    h = _heat()
    h[256][512 + int(R) + 2] = 0.9
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == 0.0


def test_sees_a_peak_just_inside_the_radius():
    h = _heat()
    h[256][512 + int(R) - 1] = 0.9
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == pytest.approx(0.9)


def test_columns_wrap_at_the_seam():
    # A site at x~0 must see a peak stored at the right edge of the heatmap.
    h = _heat()
    h[256][1023] = 0.8
    assert sa.radius_max(h, 2 / 1024, 256 / 512) == pytest.approx(0.8)


def test_rows_clamp_at_the_top():
    # A site near the top row must not crash reaching above the panorama.
    h = _heat()
    h[0][512] = 0.6
    assert sa.radius_max(h, 512 / 1024, 0.0) == pytest.approx(0.6)


def test_values_clip_to_one_like_peak_extraction():
    h = _heat()
    h[256][512] = 1.7
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == 1.0


# --------------------------------------------------------------------------- #
# null_percentile — signal against the pano's own noise floor
# --------------------------------------------------------------------------- #
def test_flat_heatmap_reads_as_chance():
    act, pct, med, p95 = sa.null_percentile(_heat(0.003), 0.5, 0.5,
                                            random.Random(0), trials=50)
    assert act == pytest.approx(0.003)
    assert pct == pytest.approx(0.5)  # every draw ties the site
    assert med == p95 == pytest.approx(0.003)


def test_a_lone_bump_at_the_site_beats_its_null():
    h = _heat()
    h[256][512] = 0.04
    act, pct, _, p95 = sa.null_percentile(h, 512 / 1024, 256 / 512,
                                          random.Random(0), trials=100)
    assert act == pytest.approx(0.04)
    assert pct > 0.9
    assert act > p95


def test_a_site_no_better_than_the_horizon_band_fails_the_test():
    # Strong response everywhere along the site's row: the site is nothing special.
    h = _heat()
    for c in range(0, 1024, 8):
        h[256][c] = 0.5
    act, pct, _, p95 = sa.null_percentile(h, 512 / 1024, 256 / 512,
                                          random.Random(0), trials=100)
    assert act == pytest.approx(0.5)
    assert not act > p95


# --------------------------------------------------------------------------- #
# site_profile / nearest_peak — separating a site response from a neighbour's tail
# --------------------------------------------------------------------------- #
def test_site_profile_centred_bump_has_zero_offset():
    h = _heat()
    h[256][512] = 0.4
    act, off, center = sa.site_profile(h, 512 / 1024, 256 / 512)
    assert act == pytest.approx(0.4)
    assert off == pytest.approx(0.0)
    assert center == pytest.approx(0.4)


def test_site_profile_offset_bump_reports_its_distance():
    h = _heat()
    h[256][512 + 15] = 0.4
    act, off, center = sa.site_profile(h, 512 / 1024, 256 / 512)
    assert act == pytest.approx(0.4)
    assert off == pytest.approx(15.0)
    assert center == 0.0  # nothing at the ramp itself


def test_nearest_peak_measures_in_matcher_units_and_wraps():
    # A peak across the seam: x=0.999 vs site x=0.001 is ~2 px away, not ~1022.
    d, score = sa.nearest_peak([(0.999, 0.5, 0.7)], 0.001, 0.5)
    assert d == pytest.approx(0.002 * 1024, abs=0.01)
    assert score == 0.7


def test_nearest_peak_with_no_peaks_is_infinite():
    d, score = sa.nearest_peak([], 0.5, 0.5)
    assert d == float("inf") and score is None


# --------------------------------------------------------------------------- #
# group_of — Phase 0's partition, reused
# --------------------------------------------------------------------------- #
def _row(city="bend", pano="p1", x=0.25, y=0.6):
    return {"city": city, "pano": pano, "x": x, "y": y}


def test_group_partition():
    r = _row()
    key = sa.row_key(r)
    assert sa.group_of(r, set(), {}) == "witnessed"
    assert sa.group_of(r, {key}, {}) == "below_floor"
    assert sa.group_of(r, {key}, {key: {}}) == "rated"
