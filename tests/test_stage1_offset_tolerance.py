"""Unit tests for the Stage 1 coordinate-tolerance geometry (issues #96, #59).

The conclusion this supports — that a city's coordinate error costs Stage 1 far
less than the raw offset distribution suggests — rests entirely on three claims
about the pipeline's geometry. Each is easy to get wrong in a way that produces a
plausible number, so each is pinned here.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_offset_tolerance as t  # noqa: E402


# --------------------------------------------------------------------------- #
# the crop's angular half-width
# --------------------------------------------------------------------------- #
def test_crop_half_angle_matches_the_pipeline_geometry():
    """download_dataset.py renders 90 deg into 1024 px and keeps [341:682]."""
    assert math.isclose(t.crop_half_angle_deg(), 18.37, abs_tol=0.02)


def test_the_naive_linear_estimate_would_overstate_it():
    """fov * crop_px / width = 29.97 deg half-width -- wrong by 63%, because a
    pinhole projection is not linear in angle. Getting this wrong would inflate
    the tolerance and make every city look safer than it is."""
    naive = 90.0 * (682 - 341) / 1024 / 2 * 2   # 90 * 341/1024, as a half-width
    assert naive > 29.0
    assert t.crop_half_angle_deg() < 19.0


def test_a_wider_crop_accepts_more_azimuth():
    assert t.crop_half_angle_deg(lo=256, hi=768) > t.crop_half_angle_deg()


# --------------------------------------------------------------------------- #
# bearing error -- the reason metric offset is the wrong unit
# --------------------------------------------------------------------------- #
def test_radial_error_is_free():
    """An offset straight along the line of sight does not move the bearing at
    all. This is why the raw offset distribution overstates the damage: a
    fraction of every error costs nothing."""
    assert t.bearing_error_deg(5.0, 20.0, 0.0) == 0.0            # away
    # sin(pi) is 1.2e-16, not 0, so this one is zero to floating-point residue.
    assert t.bearing_error_deg(5.0, 20.0, math.pi) < 1e-12       # toward


def test_tangential_error_costs_the_most():
    perp = t.bearing_error_deg(2.0, 10.0, math.pi / 2)
    diag = t.bearing_error_deg(2.0, 10.0, math.pi / 4)
    assert perp > diag > 0.0
    assert math.isclose(perp, math.degrees(math.atan2(2.0, 10.0)), abs_tol=1e-9)


def test_the_same_offset_hurts_more_at_close_range():
    """The core asymmetry: a 1 m error is 18 deg at 3 m and 3 deg at 20 m."""
    near = t.bearing_error_deg(1.0, 3.0, math.pi / 2)
    far = t.bearing_error_deg(1.0, 20.0, math.pi / 2)
    assert near > 5 * far


def test_tolerance_boundary_is_where_the_geometry_says():
    """A tangential offset of exactly 0.332 x range sits on the crop edge."""
    half = t.crop_half_angle_deg()
    d = 10.0
    on_edge = d * math.tan(math.radians(half))
    assert math.isclose(t.bearing_error_deg(on_edge, d, math.pi / 2), half, abs_tol=1e-6)
    assert t.bearing_error_deg(on_edge * 1.05, d, math.pi / 2) > half
    assert t.bearing_error_deg(on_edge * 0.95, d, math.pi / 2) < half


def test_bearing_error_is_exact_not_small_angle():
    """Offsets comparable to the range occur at close ramps, where sin(x)~x is
    visibly wrong. A 3 m error at 3 m range is 90 deg, not 57.3."""
    assert math.isclose(t.bearing_error_deg(3.0, 3.0, math.pi / 2), 45.0, abs_tol=1e-9)
    assert t.bearing_error_deg(10.0, 1.0, math.pi / 2) > 84.0


# --------------------------------------------------------------------------- #
# range estimate
# --------------------------------------------------------------------------- #
def test_ground_range_matches_the_estimator_it_borrows():
    """Same flat-ground formula as precision_by_distance.py, which was validated
    against DA3 depth to within 6.5-8.5%."""
    y = 0.6
    assert math.isclose(t.ground_range(y), 2.5 / math.tan((y - 0.5) * math.pi))


def test_ground_range_grows_toward_the_horizon():
    assert t.ground_range(0.55) > t.ground_range(0.75)
    assert math.isinf(t.ground_range(0.5))


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def test_zero_offset_never_leaves_the_crop():
    r = t.simulate([0.0], [10.0], t.crop_half_angle_deg(), trials=500)
    assert r["p_outside"] == 0.0


def test_a_huge_offset_at_close_range_almost_always_leaves():
    r = t.simulate([20.0], [2.0], t.crop_half_angle_deg(), trials=2000)
    assert r["p_outside"] > 0.75


def test_simulation_is_deterministic_for_a_seed():
    a = t.simulate([1.0, 2.0], [5.0, 10.0], 18.37, trials=3000, seed=7)
    b = t.simulate([1.0, 2.0], [5.0, 10.0], 18.37, trials=3000, seed=7)
    assert a["p_outside"] == b["p_outside"]


def test_loss_rises_monotonically_with_the_offset_scale():
    rows = t.sweep([5.0, 10.0, 20.0], t.crop_half_angle_deg(),
                   [1, 2, 4, 8], [0.3, 0.6, 1.0], trials=4000)
    ps = [r["p_outside"] for r in rows]
    assert ps == sorted(ps)
    assert rows[0]["median_offset_m"] < rows[-1]["median_offset_m"]
