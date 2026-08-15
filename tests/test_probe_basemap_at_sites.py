"""Unit tests for the at-the-sites basemap probe (issue #96).

The point of this tool is that a basemap must be graded where it will be *used*.
Two things have to be right for its output to mean anything: the tile the probe
fetches must be the tile covering the record, and the vegetation metric must
respond to vegetation rather than to brightness. CPU only, no network.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import probe_basemap_at_sites as p  # noqa: E402


def _tile(rgb, size=32):
    import io
    import numpy as np
    from PIL import Image
    a = np.zeros((size, size, 3), dtype=np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = rgb
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def _half(rgb_a, rgb_b, size=32):
    import io
    import numpy as np
    from PIL import Image
    a = np.zeros((size, size, 3), dtype=np.uint8)
    a[: size // 2] = rgb_a
    a[size // 2:] = rgb_b
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# tile addressing -- probing the wrong tile would grade the wrong place
# --------------------------------------------------------------------------- #
def test_tile_xy_matches_the_sheet_builders_own_math():
    """If this drifts from inventory_review_sheet, the probe grades a different
    tile than the sheet will render, and a clean probe stops predicting a
    buildable sheet."""
    import inventory_review_sheet as s
    for lon, lat in [(-122.33, 47.61), (-80.83, 35.03), (-104.99, 39.74)]:
        for z in (18, 20):
            px, py = s.lonlat_to_pixel(lon, lat, z)
            assert p.tile_xy(lon, lat, z) == (int(px) // 256, int(py) // 256)


def test_tile_xy_is_monotonic_in_lon_and_lat():
    x1, y1 = p.tile_xy(-122.4, 47.7, 20)
    x2, y2 = p.tile_xy(-122.3, 47.7, 20)
    x3, y3 = p.tile_xy(-122.4, 47.6, 20)
    assert x2 > x1                      # east -> larger x
    assert y3 > y1                      # south -> larger y


def test_zoom_doubles_the_grid():
    x, y = p.tile_xy(-122.33, 47.61, 19)
    x2, y2 = p.tile_xy(-122.33, 47.61, 20)
    assert (x2 // 2, y2 // 2) == (x, y)


# --------------------------------------------------------------------------- #
# the vegetation metric
# --------------------------------------------------------------------------- #
def test_green_reads_as_vegetation_and_grey_does_not():
    veg, exg, _ = p.tile_stats(_tile((40, 140, 40)))
    assert veg == 1.0 and exg > 100
    veg, exg, _ = p.tile_stats(_tile((128, 128, 128)))
    assert veg == 0.0 and abs(exg) < 1e-6


def test_bright_grey_is_not_mistaken_for_vegetation():
    """ExG = 2G - R - B is brightness-invariant on neutral colours. A metric
    that keyed on the green CHANNEL rather than excess green would call a bright
    grey roof vegetation, and leaf-off cities would look leafy."""
    for level in (30, 90, 200, 250):
        veg, exg, _ = p.tile_stats(_tile((level, level, level)))
        assert veg == 0.0
        assert abs(exg) < 1e-6


def test_vegetation_fraction_is_a_fraction_of_pixels():
    veg, _, _ = p.tile_stats(_half((40, 140, 40), (128, 128, 128)))
    assert math.isclose(veg, 0.5, abs_tol=0.02)


def test_threshold_raises_the_bar():
    blob = _half((100, 118, 100), (128, 128, 128))   # ExG = +36 on the top half
    assert p.tile_stats(blob, exg_threshold=20.0)[0] > 0.4
    assert p.tile_stats(blob, exg_threshold=50.0)[0] == 0.0


def test_a_flat_tile_reports_no_texture_so_blanks_are_detectable():
    """Esri's grey 'not yet available' tiles return 200. The blank test is
    stddev of luma, so a flat fill must read ~0 regardless of its colour."""
    assert p.tile_stats(_tile((128, 128, 128)))[2] < 1e-6
    assert p.tile_stats(_tile((40, 140, 40)))[2] < 1e-6
    assert p.tile_stats(_half((20, 20, 20), (240, 240, 240)))[2] > 50


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_the_seattle_years_are_both_available_to_compare():
    import inventory_review_sheet as s
    assert {"seattle-2019", "seattle-2025"} <= set(s.TILE_SOURCES)
    for k in ("seattle-2019", "seattle-2025"):
        assert s.TILE_SOURCES[k]["max_zoom"] == 20   # both 404 above 20


def test_default_threshold_is_documented_as_calibrated():
    assert p.EXG_THRESHOLD == 20.0
    assert "calibrat" in p.__doc__.lower()
