"""Tests for rampnet/gsv.py — the production GSV path after its lift (#103).

Three things are pinned here, each of which would silently corrupt the
street-level review instrument if wrong:

1. **The click-to-angle map agrees with the pipeline's own geometry** —
   ``perspective_col_to_azimuth_deg`` is asserted against
   ``crop_half_angle_deg()`` from ``stage1_offset_tolerance.py``, not against
   a literal, so there stays exactly one definition of the crop half-angle.
2. **The sign convention survives the render.** A feature placed at a known
   clockwise azimuth in a synthetic equirectangular pano must come out at the
   predicted *rightward* column of the perspective view — the §5j residual
   convention, end to end through the real ``equirectangular_to_perspective``.
3. **The module stays importable in CI**, which deliberately has no cv2 or
   requests (requirements-dev.txt): heavy deps must be function-local, and the
   tests that need them skip rather than fail.

The renderer tests use a small synthetic pano; nothing here touches the
network, a GPU, or a checkpoint.
"""
import math
import os
import py_compile
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import rampnet.gsv as gsv  # noqa: E402
from stage1_offset_tolerance import crop_half_angle_deg  # noqa: E402


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #
def test_module_import_pulled_in_no_heavy_deps():
    """cv2/requests/torch must be lazy: CI has no cv2 or requests, and the
    sheet builder imports this module just for the pure angle helpers."""
    for name in ("cv2", "requests", "torch"):
        assert not hasattr(gsv, name), (
            f"rampnet.gsv exposes '{name}' at module level; keep it "
            f"function-local or CI (which lacks it) cannot import the module"
        )


def test_download_dataset_still_compiles_after_the_lift():
    """The production script cannot be *imported* here (its
    inference_isolator loads a checkpoint that is not in the repo), but it
    must at least still parse — a botched rewire would otherwise surface only
    on the next Stage 1 run."""
    path = os.path.join(REPO, "stage_one", "dataset_generation", "download_dataset.py")
    py_compile.compile(path, doraise=True)


def test_download_dataset_imports_the_lifted_functions_not_local_copies():
    """One definition each. If someone re-inlines a copy, the two paths can
    drift — the exact disease the KeypointModel consolidation cured."""
    path = os.path.join(REPO, "stage_one", "dataset_generation", "download_dataset.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "from rampnet.gsv import" in src
    assert "def fetch_panorama" not in src
    assert "def equirectangular_to_perspective" not in src
    assert "def perspective_to_equirectangular" not in src
    assert "def heading_to_azimuth" not in src


# --------------------------------------------------------------------------- #
# heading_to_azimuth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "heading,expected",
    [(0, 0), (90, 90), (179, 179), (180, -180), (270, -90), (350, -10),
     (360, 0), (540, -180), (-90, -90), (-350, 10)],
)
def test_heading_to_azimuth_known_values(heading, expected):
    assert gsv.heading_to_azimuth(heading) == expected


# --------------------------------------------------------------------------- #
# the click-to-angle map vs the pipeline's own geometry
# --------------------------------------------------------------------------- #
def test_strip_right_edge_is_exactly_the_crop_half_angle():
    """Column 682 (the exclusive right bound of persp[:, 341:682]) sits at
    +crop_half_angle_deg() — the same atan(170/512), from the same single
    definition."""
    assert gsv.perspective_col_to_azimuth_deg(682) == pytest.approx(
        crop_half_angle_deg(), abs=1e-12
    )


def test_strip_is_asymmetric_and_the_left_edge_is_wider():
    """341 px left of centre vs 340 right: the left edge is -18.4577 deg,
    strictly wider than the conservative half-angle. Symmetrising the overlay
    would misdraw the left boundary by ~0.09 deg."""
    left = gsv.perspective_col_to_azimuth_deg(341)
    assert left == pytest.approx(-math.degrees(math.atan(171 / 512)), abs=1e-12)
    assert abs(left) > crop_half_angle_deg()


def test_col_deg_roundtrip():
    for col in (0.0, 100.5, 341.0, 512.0, 682.0, 1024.0):
        deg = gsv.perspective_col_to_azimuth_deg(col)
        assert gsv.azimuth_deg_to_perspective_col(deg) == pytest.approx(col, abs=1e-9)


def test_centre_is_zero_and_signs_match_the_residual_convention():
    """Right of centre = positive = clockwise. This is the §5j sign; the
    renderer test below pins the same fact through the actual projection."""
    assert gsv.perspective_col_to_azimuth_deg(512) == 0.0
    assert gsv.perspective_col_to_azimuth_deg(700) > 0
    assert gsv.perspective_col_to_azimuth_deg(300) < 0


def test_the_map_is_not_linear_in_angle():
    """The naive linear conversion (90/1024 deg per px) overstates off-centre
    angles by up to 63%; anyone 'simplifying' this to a multiply would move
    every verdict."""
    at_edge = gsv.perspective_col_to_azimuth_deg(1024)
    assert at_edge == pytest.approx(45.0, abs=1e-9)
    halfway = gsv.perspective_col_to_azimuth_deg(768)
    assert halfway < 45.0 / 2 * 1.2
    assert halfway == pytest.approx(math.degrees(math.atan(0.5)), abs=1e-9)


def test_azimuth_beyond_ninety_degrees_has_no_column():
    with pytest.raises(ValueError):
        gsv.azimuth_deg_to_perspective_col(90.0)
    with pytest.raises(ValueError):
        gsv.azimuth_deg_to_perspective_col(-90.0)


# --------------------------------------------------------------------------- #
# the renderer, end to end on a synthetic pano
# --------------------------------------------------------------------------- #
def _synthetic_equi(width=512, height=256, stripe_az_deg=0.0, stripe_px=6):
    """Black equirect pano with a white vertical stripe at the given azimuth
    relative to the pano heading. Equirect column u maps to
    lon = (u/(W-1))*2pi - pi (the §5j fact), so azimuth az sits at
    u = (az+180)/360 * (W-1)."""
    equi = np.zeros((height, width, 3), dtype=np.uint8)
    u = int(round((stripe_az_deg + 180.0) / 360.0 * (width - 1)))
    lo = max(0, u - stripe_px // 2)
    equi[:, lo:u + stripe_px // 2 + 1, :] = 255
    return equi


def _stripe_centre_column(persp):
    """Intensity centroid, not argmax: the rendered stripe is a plateau of
    saturated columns and argmax returns its left edge, which reads as a
    systematic leftward bias that has nothing to do with the projection."""
    weights = persp.sum(axis=(0, 2)).astype(np.float64)
    cols = np.arange(weights.size)
    return float((cols * weights).sum() / weights.sum())


@pytest.mark.parametrize("stripe_az", [0.0, 10.0, -10.0, 30.0])
def test_render_puts_a_feature_at_its_predicted_column(stripe_az):
    """A stripe at clockwise azimuth `a`, rendered at theta=0, must appear at
    azimuth_deg_to_perspective_col(a) — positive azimuth to the RIGHT. This
    pins the §5j sign convention through the production renderer itself, so a
    convention error anywhere in the #103 overlay stack cannot survive CI."""
    pytest.importorskip("cv2")
    equi = _synthetic_equi(stripe_az_deg=stripe_az)
    persp = gsv.equirectangular_to_perspective(equi, 90, 0.0, 0, 256, 256)
    expected = gsv.azimuth_deg_to_perspective_col(stripe_az, width=256)
    assert abs(_stripe_centre_column(persp) - expected) <= 3
    assert persp.shape == (256, 256, 3)
    assert persp.dtype == np.uint8


def test_render_centres_the_stripe_when_theta_points_at_it():
    """theta is 'aim the view at this azimuth': rendering with theta equal to
    the stripe's azimuth recentres it — the production crop's whole premise
    (the government bearing becomes column 512 of 1024)."""
    pytest.importorskip("cv2")
    equi = _synthetic_equi(stripe_az_deg=25.0)
    persp = gsv.equirectangular_to_perspective(equi, 90, 25.0, 0, 256, 256)
    assert abs(_stripe_centre_column(persp) - 128) <= 3
