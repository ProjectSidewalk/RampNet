"""Unit tests for the review sheet's georeference check (issues #96, #59).

Pure logic only — no network, no imagery. These guard the checker itself, which
matters more than usual: it is the thing standing between a systematically
mis-registered sheet and a set of verdicts nobody would know to distrust.

The load-bearing guarantees: the ellipsoid maths that validates ring scale is
genuinely independent of the Web Mercator formula it validates, the road-edge
finder refuses rather than guesses when a cross-section is obstructed, and a
cross-section is only ever credited to the axis it can actually measure.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_sheet as irs  # noqa: E402
import verify_chip_georeference as vg  # noqa: E402


# --------------------------------------------------------------------------- #
# ellipsoid
# --------------------------------------------------------------------------- #
def test_radii_of_curvature_match_known_wgs84_values():
    # At the equator N == a, and M == a(1-e^2).
    m, n = vg.local_radii(0.0)
    assert math.isclose(n, vg.WGS84_A, rel_tol=1e-12)
    assert math.isclose(m, vg.WGS84_A * (1 - vg.WGS84_E2), rel_tol=1e-12)
    # At the pole both converge on a/sqrt(1-e^2).
    m, n = vg.local_radii(90.0)
    assert math.isclose(m, n, rel_tol=1e-9)


def test_offset_of_one_degree_of_latitude_is_about_111km():
    _, lat2 = vg.offset_lonlat(0.0, 40.0, 0.0, 111000.0)
    assert 0.99 < (lat2 - 40.0) < 1.01


def test_offset_east_shrinks_with_latitude():
    """The same eastward metres are more degrees of longitude further north."""
    lon_eq, _ = vg.offset_lonlat(0.0, 0.0, 1000.0, 0.0)
    lon_hi, _ = vg.offset_lonlat(0.0, 60.0, 1000.0, 0.0)
    assert lon_hi > lon_eq * 1.9


def test_offset_is_reversible():
    lon2, lat2 = vg.offset_lonlat(-105.0, 39.7, 37.0, -21.0)
    lon3, lat3 = vg.offset_lonlat(lon2, lat2, -37.0, 21.0)
    assert math.isclose(lon3, -105.0, abs_tol=1e-9)
    assert math.isclose(lat3, 39.7, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# ring scale
# --------------------------------------------------------------------------- #
def test_rings_are_true_to_well_under_a_percent():
    """The whole point: 1/2/5/10 m rings really are those radii on the ground."""
    for row in vg.ring_scale_error(39.73, 21, [1.0, 2.0, 5.0, 10.0]):
        assert row["max_rel_error"] < 0.005, row


def test_ring_error_is_the_ellipsoid_residual_not_a_bug():
    """It should be the same relative error at every radius — a scale factor, not
    an additive offset. An additive error would shrink in relative terms as the
    ring grows, which is how a real bug would look."""
    rows = vg.ring_scale_error(39.73, 21, [1.0, 10.0])
    assert math.isclose(rows[0]["max_rel_error"], rows[1]["max_rel_error"], rel_tol=0.02)


def test_ring_check_would_catch_a_missing_latitude_correction(monkeypatch):
    """If metres_per_pixel forgot cos(lat), Denver's rings would be ~23% wrong.
    The checker has to fail loudly on that, or it is not a check."""
    monkeypatch.setattr(irs, "metres_per_pixel",
                        lambda lat, zoom, tile_px=256: 156543.03392800014 / 2 ** zoom)
    rows = vg.ring_scale_error(39.73, 21, [10.0])
    assert rows[0]["max_rel_error"] > 0.2


def test_ring_check_samples_more_than_one_bearing():
    """A single east-west probe cannot see a north-south-only error."""
    rows = vg.ring_scale_error(39.73, 21, [10.0], bearings=(0, 90))
    assert rows[0]["worst_bearing_deg"] in (0, 90)


# --------------------------------------------------------------------------- #
# road-edge finding
# --------------------------------------------------------------------------- #
def _road(width_px, shift_px=0, dark=40, bright=200):
    """Luminance sampler for a synthetic road centred at ``shift_px`` from x=0."""
    def lum(x, y):
        if abs(x) > 500:
            return None
        return dark if abs(x - shift_px) <= width_px / 2 else bright
    return lum


def test_finds_the_centre_of_a_centred_road():
    got = vg.road_centre_offset(_road(60), 0, 0, 1.0, 0.0, 100, 18)
    assert abs(got) <= 1.0


def test_measures_a_shifted_road():
    """A road whose optical centre is 10 px right of the centreline reads +10."""
    got = vg.road_centre_offset(_road(60, shift_px=10), 0, 0, 1.0, 0.0, 100, 18)
    assert 9.0 <= got <= 11.0


def test_sign_follows_the_normal_direction():
    got = vg.road_centre_offset(_road(60, shift_px=10), 0, 0, -1.0, 0.0, 100, 18)
    assert -11.0 <= got <= -9.0


def test_refuses_when_an_edge_is_out_of_reach():
    """A cross-section blocked by a parked car or tree crown must return None, not
    a fabricated midpoint — obstructions are common enough to poison a mean."""
    assert vg.road_centre_offset(_road(400), 0, 0, 1.0, 0.0, 50, 18) is None


def test_ignores_a_step_smaller_than_the_threshold():
    faint = lambda x, y: 40 if abs(x) <= 30 else 45  # noqa: E731
    assert vg.road_centre_offset(faint, 0, 0, 1.0, 0.0, 100, 18) is None


def test_segment_normal_is_unit_and_perpendicular():
    ux, uy, n = vg._segment_normal((0, 0), (3, 4))
    assert math.isclose(math.hypot(ux, uy), 1.0)
    assert math.isclose(ux * 3 + uy * 4, 0.0, abs_tol=1e-12)
    assert math.isclose(n, 5.0)


def test_degenerate_segment_has_no_normal():
    assert vg._segment_normal((1.0, 1.0), (1.0, 1.0)) is None
