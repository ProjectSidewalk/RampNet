"""Unit tests for the ramps-vs-centrelines registration check (issues #96, #59).

Pure geometry — no network, no imagery, no snapshots. These matter more than
usual because the number this produces is used to **exonerate or condemn a
city's coordinates**, and two of its failure modes are silent:

* A sign error would report a shift in the wrong direction, which is worse than
  reporting none — the same hazard ``inventory_review_summary`` guards with its
  "north is up" test. The convention is pinned here against synthetic data with
  a known, deliberate displacement.
* The half-width must cancel. If it did not, the estimator would report the
  width of the road as though it were a positional error, and every city would
  look catastrophically misplaced by about 7 m.

The other load-bearing guarantee is that a near-zero reading is trustworthy:
the estimator is built so that a real shift cannot be attenuated away by the
choice of which street each ramp is compared against.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_centerline_offset as co  # noqa: E402


def ns_segment(x, y0=0.0, y1=100.0):
    """A north-south segment at easting ``x``."""
    return ((x, y0), (x, y1))


def ew_segment(y, x0=0.0, x1=100.0):
    return ((x0, y), (x1, y))


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def test_local_metres_puts_x_east_and_y_north():
    pts, lat0 = co.to_local_metres([(-122.0, 47.6), (-121.9, 47.6)])
    assert pts[1][0] > pts[0][0]          # further east is larger x
    pts, _ = co.to_local_metres([(-122.0, 47.6), (-122.0, 47.7)])
    assert pts[1][1] > pts[0][1]          # further north is larger y


def test_one_degree_of_latitude_is_about_111km():
    pts, _ = co.to_local_metres([(0.0, 0.0), (0.0, 1.0)], lat0=0.0)
    assert 110000 < pts[1][1] - pts[0][1] < 112000


def test_segments_from_paths_drops_repeated_vertices():
    """A zero-length segment has no direction and must not default to one."""
    paths = [[[-122.0, 47.6], [-122.0, 47.6], [-122.0, 47.601]]]
    segs = co.segments_from_paths(paths, lat0=47.6)
    assert len(segs) == 1


def test_segments_from_paths_splits_a_multi_vertex_path():
    paths = [[[-122.0, 47.60], [-122.0, 47.61], [-122.0, 47.62]]]
    assert len(co.segments_from_paths(paths, lat0=47.6)) == 2


# --------------------------------------------------------------------------- #
# axis assignment
# --------------------------------------------------------------------------- #
def test_north_south_street_measures_the_east_axis():
    """Its perpendicular points east-west, so it constrains east."""
    assert co.segment_axis(*ns_segment(0.0)) == "east"


def test_east_west_street_measures_the_north_axis():
    assert co.segment_axis(*ew_segment(0.0)) == "north"


def test_a_diagonal_is_refused_rather_than_pooled():
    """45 degrees constrains a diagonal; reading it as cardinal leaks the other
    axis's error into this one."""
    assert co.segment_axis((0.0, 0.0), (100.0, 100.0)) is None


def test_axis_tolerance_is_the_documented_angle():
    just_inside = math.tan(math.radians(15.0)) * 100.0
    just_outside = math.tan(math.radians(25.0)) * 100.0
    assert co.segment_axis((0.0, 0.0), (just_inside, 100.0), 20.0) == "east"
    assert co.segment_axis((0.0, 0.0), (just_outside, 100.0), 20.0) is None


def test_degenerate_segment_has_no_axis():
    assert co.segment_axis((5.0, 5.0), (5.0, 5.0)) is None


# --------------------------------------------------------------------------- #
# perpendicular offset
# --------------------------------------------------------------------------- #
def test_offset_east_of_a_north_south_street_is_positive_east():
    off = co.perpendicular_offset((7.0, 50.0), *ns_segment(0.0))
    assert off is not None
    assert math.isclose(off[0], 7.0)      # east component
    assert math.isclose(off[1], 0.0, abs_tol=1e-9)
    assert math.isclose(off[2], 7.0)


def test_offset_west_of_a_north_south_street_is_negative_east():
    off = co.perpendicular_offset((-7.0, 50.0), *ns_segment(0.0))
    assert off[0] < 0


def test_offset_north_of_an_east_west_street_is_positive_north():
    off = co.perpendicular_offset((50.0, 6.0), *ew_segment(0.0))
    assert math.isclose(off[1], 6.0)


def test_offset_is_refused_beyond_the_segment_ends():
    """Past the end the nearest point is a vertex, so the 'perpendicular' is not
    perpendicular to anything — that is an intersection, not a kerb."""
    assert co.perpendicular_offset((7.0, 150.0), *ns_segment(0.0)) is None
    assert co.perpendicular_offset((7.0, -50.0), *ns_segment(0.0)) is None


def test_offset_is_independent_of_digitisation_direction():
    """The sign must be geographic. A segment drawn south-to-north and the same
    segment drawn north-to-south must place the same ramp on the same side —
    this is the trap that makes a real shift cancel to zero."""
    a, b = ns_segment(0.0)
    fwd = co.perpendicular_offset((7.0, 50.0), a, b)
    rev = co.perpendicular_offset((7.0, 50.0), b, a)
    assert math.isclose(fwd[0], rev[0])


def test_degenerate_segment_yields_no_offset():
    assert co.perpendicular_offset((1.0, 1.0), (5.0, 5.0), (5.0, 5.0)) is None


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def test_half_width_cancels_out_of_the_shift():
    """Ramps 7 m either side of the centreline, no displacement: shift 0, and
    the 7 m reappears as the half-width rather than as an error."""
    vals = [7.0, 7.1, 6.9, -7.0, -6.9, -7.1]
    a = co.axis_shift(vals)
    assert math.isclose(a["shift_m"], 0.0, abs_tol=1e-9)
    assert math.isclose(a["half_width_m"], 7.0, abs_tol=0.05)


def test_a_known_eastward_shift_is_recovered_with_the_right_sign():
    """THE sign test. Displace every ramp 2 m east; the estimator must say
    +2.00, not -2.00 and not +9.00."""
    base = [7.0, 7.1, 6.9, -7.0, -6.9, -7.1]
    a = co.axis_shift([v + 2.0 for v in base])
    assert math.isclose(a["shift_m"], 2.0, abs_tol=0.05)
    assert math.isclose(a["half_width_m"], 7.0, abs_tol=0.05)


def test_a_westward_shift_reports_negative():
    base = [7.0, 7.1, 6.9, -7.0, -6.9, -7.1]
    assert co.axis_shift([v - 1.5 for v in base])["shift_m"] < 0


def test_shift_needs_both_sides_of_the_street():
    """One-sided data cannot separate a shift from a half-width, so refuse."""
    assert co.axis_shift([7.0, 7.1, 6.9]) is None
    assert co.axis_shift([]) is None


def test_varying_road_width_does_not_create_a_shift():
    """Half-widths differing street to street must still cancel, because it is
    the two sides of the SAME distribution that are being subtracted."""
    vals = [4.0, 6.0, 8.0, 12.0, -4.0, -6.0, -8.0, -12.0]
    assert math.isclose(co.axis_shift(vals)["shift_m"], 0.0, abs_tol=0.6)


# --------------------------------------------------------------------------- #
# end-to-end on synthetic streets
# --------------------------------------------------------------------------- #
def _grid(spacing=100.0, n=6):
    segs = []
    for i in range(n):
        segs.append(((i * spacing, 0.0), (i * spacing, (n - 1) * spacing)))
        segs.append(((0.0, i * spacing), ((n - 1) * spacing, i * spacing)))
    return segs


def _ramps_on_grid(shift_e=0.0, shift_n=0.0, half=7.0, spacing=100.0, n=6):
    """Four ramps per intersection, one per quadrant, then displaced."""
    pts = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            cx, cy = i * spacing, j * spacing
            for dx, dy in ((half, half), (-half, half), (half, -half), (-half, -half)):
                pts.append((cx + dx + shift_e, cy + dy + shift_n))
    return pts


def test_end_to_end_recovers_zero_on_an_undisplaced_grid():
    idx = co.SegmentIndex(_grid())
    res = co.analyse(co.collect_samples(_ramps_on_grid(), idx), bootstrap=0)
    assert abs(res["shift_east_m"]) < 0.2
    assert abs(res["shift_north_m"]) < 0.2
    assert res["half_width_plausible"]


def test_end_to_end_recovers_a_planted_two_metre_east_shift():
    idx = co.SegmentIndex(_grid())
    res = co.analyse(co.collect_samples(_ramps_on_grid(shift_e=2.0), idx), bootstrap=0)
    assert math.isclose(res["shift_east_m"], 2.0, abs_tol=0.3)
    assert abs(res["shift_north_m"]) < 0.3
    assert math.isclose(res["resultant_m"], 2.0, abs_tol=0.3)


def test_end_to_end_keeps_the_axes_independent():
    """A purely northward displacement must not leak into the east reading."""
    idx = co.SegmentIndex(_grid())
    res = co.analyse(co.collect_samples(_ramps_on_grid(shift_n=1.5), idx), bootstrap=0)
    assert math.isclose(res["shift_north_m"], 1.5, abs_tol=0.3)
    assert abs(res["shift_east_m"]) < 0.3


def test_a_shift_is_not_attenuated_by_which_street_is_nearest():
    """The regression this guards: picking the single nearest street biases the
    estimate toward zero, because an eastward shift lengthens the distance to
    north-south streets only, so east-side ramps get reassigned to the east-west
    street and drop out of the axis that can see the shift. Selecting the
    nearest WITHIN each axis is immune, so a large shift comes back at full
    size rather than halved."""
    idx = co.SegmentIndex(_grid())
    res = co.analyse(co.collect_samples(_ramps_on_grid(shift_e=3.0), idx), bootstrap=0)
    assert res["shift_east_m"] > 2.5


def test_each_ramp_contributes_at_most_one_sample_per_axis():
    idx = co.SegmentIndex(_grid())
    ramps = _ramps_on_grid()
    s = co.collect_samples(ramps, idx)
    assert len(s["east"]) <= len(ramps)
    assert len(s["north"]) <= len(ramps)


def test_ramps_far_from_every_street_are_dropped():
    idx = co.SegmentIndex(_grid())
    s = co.collect_samples([(250.0, 250.0)], idx, max_dist=5.0)
    assert s["east"] == [] and s["north"] == []


def test_implausible_half_width_is_flagged_not_silently_reported():
    """Clusters 0.2 m either side are not the two sides of a road, so whatever
    they bracket is not a positional shift."""
    res = co.analyse({"east": [0.2, 0.21, -0.2, -0.19], "north": []}, bootstrap=0)
    assert res["half_width_plausible"] is False


def test_bootstrap_ci_brackets_the_point_estimate():
    vals = [v + 2.0 for v in (7.0, 7.1, 6.9, 7.05, -7.0, -6.9, -7.1, -7.05)]
    lo, hi = co.bootstrap_shift(vals, n=200, seed=1)
    assert lo <= co.axis_shift(vals)["shift_m"] <= hi


def test_bootstrap_is_reproducible_under_a_seed():
    vals = [7.0, 7.1, 6.9, -7.0, -6.9, -7.1]
    assert co.bootstrap_shift(vals, n=100, seed=5) == co.bootstrap_shift(vals, n=100, seed=5)


def test_bearing_is_reported_clockwise_from_north():
    """A purely eastward shift is bearing 90, not 0 and not 270 — a bearing
    reported backwards would send a reader looking for the error on the wrong
    side of the street."""
    res = co.analyse({"east": [9.0, -5.0], "north": []}, bootstrap=0)
    assert math.isclose(res["shift_east_m"], 2.0, abs_tol=1e-9)
    assert math.isclose(res["bearing_deg"], 90.0, abs_tol=1e-6)
