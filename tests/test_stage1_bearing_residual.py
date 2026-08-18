"""Unit tests for the Stage 1 bearing-residual estimator (issue #96).

The claim this supports — that a city's registration error can be measured from
the published dataset alone, with no government files, no imagery and no
checkpoint — rests on one convention and one estimator property:

* the published label's normalised ``x`` really is an azimuth, via
  ``lon = (u/(W-1))*2*pi - pi``; and
* a *systematic shift* is recoverable as the mean residual.

Both are easy to get wrong in ways that produce plausible numbers rather than
errors, so both are pinned here. CPU only, no network, no fixtures.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_bearing_residual as b  # noqa: E402


def destination(lat, lng, bearing_deg, range_m):
    """Point at ``range_m`` on ``bearing_deg`` from (lat, lng). Test-only."""
    R = 6371008.8
    p1, l1 = math.radians(lat), math.radians(lng)
    th, dr = math.radians(bearing_deg), range_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dr) +
                   math.cos(p1) * math.sin(dr) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(dr) * math.cos(p1),
                         math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


# --------------------------------------------------------------------------- #
# the azimuth convention
# --------------------------------------------------------------------------- #
def test_equirect_centre_column_is_straight_ahead():
    """perspective_to_equirectangular maps u/(W-1) -> lon in [-pi, pi), so the
    centre column is azimuth 0 relative to the panorama heading."""
    assert math.isclose(b.equirect_x_to_azimuth_deg(0.5), 0.0, abs_tol=1e-9)


def test_equirect_x_spans_a_full_turn_signed():
    assert math.isclose(b.equirect_x_to_azimuth_deg(0.75), 90.0, abs_tol=1e-9)
    assert math.isclose(b.equirect_x_to_azimuth_deg(0.25), -90.0, abs_tol=1e-9)


def test_dropping_the_half_turn_offset_would_invert_every_residual():
    """The -180 in ``x*360 - 180`` is load-bearing. Without it the centre column
    reads as 180 deg -- pointing backwards -- and because wrap_deg folds that
    back into range it produces a *plausible* distribution rather than an error.
    This is the mistake the convention check in the script exists to catch."""
    naive = b.wrap_deg(0.5 * 360.0)
    assert abs(naive - b.equirect_x_to_azimuth_deg(0.5)) == 180.0


def test_wrap_is_bounded_and_half_open_at_minus_180():
    assert math.isclose(b.wrap_deg(350.0), -10.0)
    assert math.isclose(b.wrap_deg(-350.0), 10.0)
    assert math.isclose(b.wrap_deg(180.0), -180.0)   # half-open, not symmetric
    for d in (-720.0, -181.0, 0.0, 181.0, 719.0):
        assert -180.0 <= b.wrap_deg(d) < 180.0


# --------------------------------------------------------------------------- #
# bearings
# --------------------------------------------------------------------------- #
def test_cardinal_bearings():
    assert math.isclose(b.fwd_azimuth_deg(40.0, -74.0, 40.001, -74.0), 0.0, abs_tol=0.01)
    assert math.isclose(b.fwd_azimuth_deg(40.0, -74.0, 40.0, -73.999), 90.0, abs_tol=0.01)
    assert math.isclose(b.fwd_azimuth_deg(40.0, -74.0, 39.999, -74.0), 180.0, abs_tol=0.01)


def test_great_circle_matches_the_geodesic_used_by_stage_1():
    """Stage 1 uses pyproj.Geod.inv. Over the 35 m inclusion radius the two must
    agree far inside the ~3 deg residual being measured, or the dependency-free
    form is not a valid substitute."""
    pyproj = __import__("importlib").util.find_spec("pyproj")
    if pyproj is None:
        import pytest
        pytest.skip("pyproj not installed")
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    for bearing in range(0, 360, 23):
        for rng in (3.0, 11.1, 35.0):
            lat, lng = destination(40.7, -74.0, bearing, rng)
            mine = b.fwd_azimuth_deg(40.7, -74.0, lat, lng)
            theirs, _, _ = geod.inv(-74.0, 40.7, lng, lat)
            assert abs(b.wrap_deg(mine - theirs)) < 0.01


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def test_matching_is_nearest_first_and_one_to_one():
    res, n = b.match_bearings([0.0, 30.0], [31.0, 1.0], max_sep_deg=40.0)
    assert n == 2
    assert sorted(round(r, 6) for r in res) == [1.0, 1.0]


def test_a_record_outside_the_crop_contributes_nothing():
    """A ramp whose true position fell outside the strip was never rendered, so
    it must not be force-matched to an unrelated peak -- that would silently
    manufacture a huge residual instead of an honest non-match."""
    res, n = b.match_bearings([0.0, 120.0], [1.0], max_sep_deg=40.0)
    assert n == 1
    assert len(res) == 1


def test_the_distribution_is_censored_not_merely_thinned():
    """Everything the estimator can see is inside max_sep by construction. This
    is why matched_frac has to be read beside the residuals."""
    res, _ = b.match_bearings([0.0, 50.0, 100.0], [0.0, 50.0, 100.0],
                              max_sep_deg=20.0)
    assert all(abs(r) <= 20.0 for r in res)


def test_matching_across_a_wraparound():
    res, n = b.match_bearings([179.0], [-179.0], max_sep_deg=10.0)
    assert n == 1
    assert math.isclose(res[0], 2.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# the property Seattle needs: a systematic shift is recoverable
# --------------------------------------------------------------------------- #
def _synthetic_pano(shift_deg, ranges=(5.0, 11.1, 20.0), bearings=range(-60, 61, 15)):
    """Government coords displaced so each sits ``shift_deg`` off the true ramp."""
    plat, plng, heading = 40.7, -74.0, 37.0
    gov_coords, points = [], []
    for br in bearings:
        for rng in ranges:
            true_abs = heading + br
            gov_coords.append(destination(plat, plng, true_abs - shift_deg, rng))
            x = ((true_abs - heading) + 180.0) / 360.0
            points.append([x % 1.0, 0.5])
    return [plat, plng], heading, gov_coords, points


def test_a_known_shift_is_recovered_as_the_mean_residual():
    """The Seattle question in a test: if a city's coordinates are rotated by a
    constant, the mean residual must report that constant."""
    for shift in (-6.0, -1.5, 0.0, 2.0, 5.0):
        pc, az, gov, pts = _synthetic_pano(shift)
        res, n = b.residuals_for_pano(pc, az, gov, pts)
        assert n == len(gov)
        assert abs(sum(res) / len(res) - shift) < 0.05


def test_an_unshifted_city_gives_a_mean_of_zero():
    pc, az, gov, pts = _synthetic_pano(0.0)
    res, _ = b.residuals_for_pano(pc, az, gov, pts)
    assert abs(sum(res) / len(res)) < 0.02


def test_shift_recovery_does_not_depend_on_range():
    """Radial error is free (§5g) -- the residual is angular, so the same shift
    must be recovered whether ramps are 5 m or 20 m away."""
    near = b.residuals_for_pano(*_synthetic_pano(4.0, ranges=(5.0,)))[0]
    far = b.residuals_for_pano(*_synthetic_pano(4.0, ranges=(20.0,)))[0]
    assert abs(sum(near) / len(near) - sum(far) / len(far)) < 0.05


# --------------------------------------------------------------------------- #
# summary statistics
# --------------------------------------------------------------------------- #
def test_standard_error_shrinks_with_n_so_a_shift_claim_is_testable():
    small = b.summarize([1.0, -1.0] * 50, 100, 100, 20)
    large = b.summarize([1.0, -1.0] * 5000, 10000, 10000, 2000)
    assert large["se_mean_deg"] < small["se_mean_deg"] / 5


def test_summary_reports_the_matched_fraction():
    s = b.summarize([0.1] * 40, n_gov=100, n_matched=40, n_panos=10)
    assert math.isclose(s["matched_frac"], 0.40)


def test_residuals_beyond_the_crop_are_counted_as_cross_assignments():
    """A peak further from its matched record than the crop half-angle cannot
    have come from that record's own strip, because the combined heatmap is the
    max over every crop in the panorama. So this counts matcher mistakes, NOT
    §5g's 'ramp fell outside its own crop' -- conflating the two would turn a
    matching artefact into a claim about coordinate error."""
    s = b.summarize([0.0] * 90 + [25.0] * 10, 100, 100, 10)
    assert math.isclose(s["frac_cross_assigned"], 0.10)
    assert 25.0 > b.CROP_HALF_ANGLE_DEG


def test_crop_half_angle_is_the_tolerance_module_s_not_a_second_copy():
    """§5g's +/-18.37 deg must have exactly one definition. Re-deriving it here
    from an averaged 170.5 px half-width silently gave 18.42 -- the strip is
    asymmetric about the centre column (341 left, 340 right) and the published
    figure is the conservative side."""
    import stage1_offset_tolerance as t
    assert b.CROP_HALF_ANGLE_DEG == t.crop_half_angle_deg()
    assert math.isclose(b.CROP_HALF_ANGLE_DEG, 18.37, abs_tol=0.01)


def test_too_few_residuals_is_reported_not_guessed():
    assert b.summarize([], 0, 0, 0)["insufficient"] is True


# --------------------------------------------------------------------------- #
# city attribution
# --------------------------------------------------------------------------- #
def test_the_three_published_cities_are_recognised():
    assert b.city_of(40.75, -73.99) == "nyc"
    assert b.city_of(45.52, -122.68) == "portland"
    assert b.city_of(44.06, -121.31) == "bend"


def test_an_unknown_city_is_labelled_not_dropped():
    """A future corpus must not be silently misfiled into an existing city."""
    assert b.city_of(47.61, -122.33) == "other"
