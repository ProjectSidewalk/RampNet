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


# --------------------------------------------------------------------------- #
# seam_of — where this window and the matcher's disagree
# --------------------------------------------------------------------------- #
def test_seam_flag_is_true_within_a_radius_of_either_edge():
    # radius_max wraps columns; rampnet.metrics.greedy_match, which produced the
    # `silent` label, takes a plain x difference. Inside R of x=0 or x=1 the two
    # therefore read different windows, and that has to be visible in the output
    # rather than inferred later from the coordinates.
    assert sa.seam_of(0.001) and sa.seam_of(0.999)
    assert sa.seam_of(R / 1024 - 1e-6)


def test_seam_flag_is_false_away_from_the_edges():
    assert not sa.seam_of(0.5)
    assert not sa.seam_of(R / 1024 + 1e-6)
    assert not sa.seam_of(1.0 - (R / 1024 + 1e-6))


# --------------------------------------------------------------------------- #
# class_of — the decomposition's only two cutoffs
# --------------------------------------------------------------------------- #
def test_class_cutoffs():
    assert sa.class_of(0.0) == "absent"
    assert sa.class_of(sa.ABSENT_MAX - 1e-9) == "absent"
    assert sa.class_of(sa.ABSENT_MAX) == "faint_local"
    assert sa.class_of(sa.PEAK_FLOOR - 1e-9) == "faint_local"
    assert sa.class_of(sa.PEAK_FLOOR) == "tail"
    assert sa.class_of(1.0) == "tail"


def test_the_class_floor_is_the_extractors_own_floor():
    # `tail` means "at or above the score floor the caches were extracted at", which
    # is what licenses reading it as an outside mode: a silent miss has no floor peak
    # inside the radius by definition. Drifting these apart would break the reading
    # without breaking anything visible.
    assert sa.PEAK_FLOOR == 0.05


# --------------------------------------------------------------------------- #
# build_payload — the result file records what it is a result OF
# --------------------------------------------------------------------------- #
def test_payload_records_the_run_scope():
    pay = sa.build_payload([{"act": 0.1}], 0.30, ["bend", "clovis"], 7, 2)
    assert pay["cities"] == ["bend", "clovis"] and pay["panos"] == 7
    assert pay["n"] == 1 and pay["skipped_no_imagery"] == 2
    assert pay["null_seed"] == sa.NULL_SEED and pay["tta"] is False


def test_json_out_refuses_a_truncated_run():
    # analysis_out/silent_activation.json is a committed artifact and every number
    # in 0c derives from it, so a smoke-test run must not be able to overwrite it
    # with something that looks complete.
    with pytest.raises(SystemExit):
        sa.main(["--limit", "3", "--json-out", "x.json"])


def test_json_out_refuses_a_city_subset_without_allow_partial():
    with pytest.raises(SystemExit):
        sa.main(["--cities", "bend", "--json-out", "x.json"])
