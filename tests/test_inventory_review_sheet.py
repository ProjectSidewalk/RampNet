"""Unit tests for the aerial review sheet's tile math and sampling (issues #96, #59).

No network and no PIL work — only the pure half. The load-bearing guarantees: the
Web Mercator math puts the crosshair on the coordinate being judged (an offset
here would corrupt every verdict in a way no reviewer could detect), the blank
placeholder that services return past their deepest level is recognised rather
than presented as evidence, and the default sample is record-weighted so the
resulting distribution estimates label accuracy rather than area coverage.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_sheet as irs  # noqa: E402


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def test_pixel_origin_is_the_top_left_of_the_world():
    x, y = irs.lonlat_to_pixel(-180.0, 85.05112878, 0)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-3)


def test_null_island_is_the_centre_of_the_world():
    x, y = irs.lonlat_to_pixel(0.0, 0.0, 0)
    assert math.isclose(x, 128.0) and math.isclose(y, 128.0)


def test_resolution_matches_the_published_web_mercator_table():
    # z=0 at the equator is 156543.03 m/px, and each level halves it.
    assert math.isclose(irs.metres_per_pixel(0.0, 0), 156543.034, rel_tol=1e-6)
    assert math.isclose(irs.metres_per_pixel(0.0, 19), 156543.034 / 2 ** 19, rel_tol=1e-6)


def test_resolution_shrinks_with_latitude():
    """Denver's 0.23 m/px at z19 is the equatorial 0.2986 times cos(39.74)."""
    got = irs.metres_per_pixel(39.74, 19)
    assert math.isclose(got, 0.2986 * math.cos(math.radians(39.74)), rel_tol=1e-3)


def test_tile_range_centres_the_crop_on_the_coordinate():
    lon, lat, z = -104.9911615, 39.7461177, 19
    span_px = 174
    x0, y0, x1, y1, ox, oy = irs.tile_range(lon, lat, z, span_px)
    px, py = irs.lonlat_to_pixel(lon, lat, z)
    # the crosshair sits at the crop's centre, to within a pixel
    assert math.isclose(ox + span_px / 2.0, px, abs_tol=1.0)
    assert math.isclose(oy + span_px / 2.0, py, abs_tol=1.0)
    # and the tile block actually covers the crop
    assert x0 * irs.TILE_PX <= ox and (x1 + 1) * irs.TILE_PX >= ox + span_px
    assert y0 * irs.TILE_PX <= oy and (y1 + 1) * irs.TILE_PX >= oy + span_px


def test_tile_range_spans_multiple_tiles_when_the_crop_straddles_a_seam():
    # z=1 is a 2x2 world; (0, 0) sits exactly on the seam, so any crop straddles.
    x0, y0, x1, y1, _, _ = irs.tile_range(0.0, 0.0, 1, 100)
    assert (x1 - x0 + 1) * (y1 - y0 + 1) == 4


# --------------------------------------------------------------------------- #
# blank-tile detection
# --------------------------------------------------------------------------- #
def test_placeholder_tile_is_recognised():
    """'Map data not yet available' is flat mid-grey."""
    assert irs.looks_blank((200.0, 1.5)) is True


def test_real_imagery_is_not_discarded():
    assert irs.looks_blank((110.0, 42.0)) is False
    assert irs.looks_blank((200.0, 30.0)) is False


def test_a_dark_flat_chip_is_not_called_a_placeholder():
    """Deep building shadow is flat but dark — that is unreadable-for-the-reviewer,
    not a missing tile, and the two have different remedies."""
    assert irs.looks_blank((20.0, 2.0)) is False


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_uniform_sample_is_deterministic_and_sized():
    a = irs.uniform_sample(1000, 60, seed=20260731)
    b = irs.uniform_sample(1000, 60, seed=20260731)
    assert a == b and len(a) == 60 and len(set(a)) == 60


def test_uniform_sample_changes_with_the_seed():
    assert irs.uniform_sample(1000, 60, 1) != irs.uniform_sample(1000, 60, 2)


def test_uniform_sample_cannot_over_draw():
    assert len(irs.uniform_sample(10, 60, seed=1)) == 10


def test_uniform_sample_follows_record_density():
    """The point of the default: a city with 90% of its records downtown should
    yield a sample that is ~90% downtown, because that is where the labels are."""
    n = 1000
    picked = irs.uniform_sample(n, 200, seed=7)
    dense = sum(1 for i in picked if i < 900)
    assert 0.85 < dense / 200.0 < 0.95


def test_stratified_sample_spreads_across_cells():
    """The diagnostic option: 999 clustered points and 1 outlier, and the outlier
    still gets picked — which is precisely why it is not the default."""
    pts = [(0.0 + i * 1e-6, 0.0) for i in range(999)] + [(1.0, 1.0)]
    got = irs.stratified_sample(pts, 2, seed=3, grid=8)
    assert 999 in got


def test_stratified_sample_is_deterministic():
    pts = [(i * 0.01, i * 0.01) for i in range(200)]
    assert irs.stratified_sample(pts, 20, 5) == irs.stratified_sample(pts, 20, 5)


def test_stratified_sample_of_nothing_is_empty():
    assert irs.stratified_sample([], 10, 1) == []
    assert irs.stratified_sample([(0.0, 0.0)], 0, 1) == []


# --------------------------------------------------------------------------- #
# basemap registry
# --------------------------------------------------------------------------- #
def test_every_source_declares_provenance_and_a_depth_limit():
    for name, src in irs.TILE_SOURCES.items():
        assert src["attribution"] and src["note"], name
        assert src["max_zoom"] >= 1, name
        assert "{z}" in src["url"] and "{x}" in src["url"] and "{y}" in src["url"], name
