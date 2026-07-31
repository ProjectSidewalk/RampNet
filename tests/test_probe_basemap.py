"""Unit tests for the basemap probe (issue #96).

No network — the pure half only. What is pinned is the reasoning that decides
whether a city's sheet is worth a reviewer's time: the Web Mercator test has to
reject a cache that has the right CRS but a bespoke resolution ladder, and the
resolution grading has to match the thresholds §5e argues for.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import probe_basemap as pb  # noqa: E402


def _lods(n, tile_px=256):
    return [{"level": z, "resolution": pb.WEBMERC_R0 * (256.0 / tile_px) / 2 ** z}
            for z in range(n)]


def _tileinfo(wkid=3857, n=22, tile_px=256):
    return {"spatialReference": {"latestWkid": wkid}, "rows": tile_px, "lods": _lods(n, tile_px)}


# --------------------------------------------------------------------------- #
# tile scheme
# --------------------------------------------------------------------------- #
def test_standard_web_mercator_is_accepted():
    assert pb.is_web_mercator(_tileinfo())
    assert pb.is_web_mercator(_tileinfo(wkid=102100))


def test_a_state_plane_cache_is_rejected():
    """Municipal caches are often built in state plane. The sheet's tile math
    assumes 3857, and a mismatch misplaces every crosshair with no symptom the
    reviewer could see."""
    assert not pb.is_web_mercator(_tileinfo(wkid=2926))   # WA state plane


def test_right_crs_but_a_bespoke_ladder_is_rejected():
    """The half that is easy to miss: 3857 alone is not enough, because a cache
    can use its own scales and the z->resolution mapping would be wrong."""
    ti = _tileinfo()
    ti["lods"][-1]["resolution"] *= 1.5
    assert not pb.is_web_mercator(ti)


def test_a_dynamic_service_has_no_tile_info():
    assert not pb.is_web_mercator(None)
    assert not pb.is_web_mercator({})


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def test_resolution_matches_the_published_web_mercator_table():
    assert math.isclose(pb.metres_per_pixel(0, 0.0), pb.WEBMERC_R0, rel_tol=1e-9)
    assert math.isclose(pb.metres_per_pixel(1, 0.0), pb.WEBMERC_R0 / 2, rel_tol=1e-9)


def test_resolution_shrinks_with_latitude():
    assert pb.metres_per_pixel(20, 47.6) < pb.metres_per_pixel(20, 0.0)


def test_denver_and_seattle_come_out_where_they_were_measured():
    """The two real cases: Denver z21 at 39.75 is 0.057 m/px and usable; King
    County z20 at 47.61 is 0.101 and coarser."""
    assert math.isclose(pb.metres_per_pixel(21, 39.75), 0.0573, abs_tol=0.001)
    assert math.isclose(pb.metres_per_pixel(20, 47.6089), 0.1007, abs_tol=0.001)


def test_grading_matches_the_thresholds_the_doc_argues_for():
    assert pb.grade(0.057).startswith("GOOD")        # Denver
    assert pb.grade(0.101).startswith("USABLE")      # King County
    assert pb.grade(0.23).startswith("TOO COARSE")   # Denver's 2018 cache
    assert pb.grade(1.0).startswith("TOO COARSE")    # Esri World over Denver


def test_larger_tiles_at_the_same_zoom_are_finer():
    assert pb.metres_per_pixel(20, 47.6, tile_px=512) < pb.metres_per_pixel(20, 47.6)


# --------------------------------------------------------------------------- #
# blank placeholder detection
# --------------------------------------------------------------------------- #
def test_the_grey_placeholder_is_recognised():
    assert pb.looks_blank(200.0, 1.0)


def test_real_imagery_is_not_discarded():
    assert not pb.looks_blank(120.0, 45.0)


def test_a_dark_flat_subject_is_not_called_a_placeholder():
    """Both conditions have to hold, so fresh snow or a flat dark roof is not
    thrown away on low variance alone."""
    assert not pb.looks_blank(20.0, 2.0)


# --------------------------------------------------------------------------- #
# tile addressing
# --------------------------------------------------------------------------- #
def test_tile_xy_puts_null_island_at_the_centre():
    n = 2 ** 4
    assert pb.tile_xy(0.0, 0.0, 4) == (n // 2, n // 2)


def test_tile_xy_is_monotonic():
    x1, y1 = pb.tile_xy(-122.34, 47.61, 18)
    x2, y2 = pb.tile_xy(-122.33, 47.61, 18)
    assert x2 >= x1
    xn, yn = pb.tile_xy(-122.34, 47.62, 18)
    assert yn <= y1          # further north is a smaller row
