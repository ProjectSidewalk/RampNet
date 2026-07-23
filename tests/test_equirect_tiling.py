"""Correctness guards for the perspective reprojection (scripts/model_comparison/
equirect_tiling.py).

The scoring only trusts reprojection if the geometry is right, so these pin it
three ways:
  1. Round-trip identity: forward-projecting a pano point into a view and mapping
     it back recovers the original point (the inverse map is exact).
  2. De-distortion invariants: a correct gnomonic projection renders the horizon
     as a straight horizontal line and meridians as straight vertical lines. If
     the projection were wrong (e.g. still equirectangularly warped), these would
     bend.
  3. Renderer/scalar consistency: the vectorized image renderer agrees with the
     scalar point map, and a constant image reprojects to a constant image.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from equirect_tiling import (  # noqa: E402
    View, default_views,
    perspective_point_to_equirect, equirect_point_to_perspective,
    equirect_to_perspective, dedup_points,
)

PITCHED = View(yaw_deg=0.0, pitch_deg=-30.0, fov_h_deg=90.0, fov_v_deg=90.0, width=512, height=512)
LEVEL = View(yaw_deg=0.0, pitch_deg=0.0, fov_h_deg=90.0, fov_v_deg=90.0, width=512, height=512)


# --- 1. round-trip identity -------------------------------------------------

def test_roundtrip_view_point_to_pano_and_back():
    for u in (0.2, 0.5, 0.8):
        for v in (0.2, 0.5, 0.8):
            x, y = perspective_point_to_equirect(u, v, PITCHED)
            back = equirect_point_to_perspective(x, y, PITCHED)
            assert back is not None
            assert abs(back[0] - u) < 1e-6 and abs(back[1] - v) < 1e-6


def test_view_center_looks_at_yaw_pitch():
    x, y = perspective_point_to_equirect(0.5, 0.5, PITCHED)
    assert abs(x - 0.5) < 1e-9              # yaw 0 -> longitude 0 -> X 0.5
    assert abs(y - (0.5 + 30.0 / 180.0)) < 1e-9   # pitch -30 -> latitude -30 -> Y 0.6667


def test_point_behind_camera_is_not_visible():
    # A pano point at the opposite longitude is behind a yaw-0 view.
    assert equirect_point_to_perspective(0.0, 0.5, LEVEL) is None   # X=0 -> lon -180


# --- 2. de-distortion invariants (gnomonic straightness) --------------------

def test_horizon_renders_as_straight_horizontal():
    # Horizon = latitude 0 (Y=0.5). In a level view every horizon point must land
    # on the same image row (v == 0.5).
    for lon_deg in (-40, -20, 0, 20, 40):        # within the 45 deg half-FOV
        x = lon_deg / 360.0 + 0.5
        uv = equirect_point_to_perspective(x, 0.5, LEVEL)
        assert uv is not None
        assert abs(uv[1] - 0.5) < 1e-9


def test_meridian_renders_as_straight_vertical():
    # A meridian (constant longitude, a great circle) must render as a vertical
    # line: constant u as latitude varies.
    lon_deg = 20.0
    x = lon_deg / 360.0 + 0.5
    us = []
    for lat_deg in (-30, -15, 0, 15, 30):
        y = 0.5 - lat_deg / 180.0
        uv = equirect_point_to_perspective(x, y, LEVEL)
        assert uv is not None
        us.append(uv[0])
    assert max(us) - min(us) < 1e-9


# --- 3. renderer / scalar consistency ---------------------------------------

def _make_src(w, h):
    import numpy as np
    xs = (np.arange(w)[None, :] * 7) % 256
    ys = (np.arange(h)[:, None] * 13) % 256
    grey = ((xs + ys) % 256).astype("uint8")
    return np.stack([grey, grey, grey], axis=-1)


def test_renderer_size_and_scalar_agreement():
    import numpy as np
    from PIL import Image

    sw, sh = 360, 180
    src = Image.fromarray(_make_src(sw, sh))
    view = View(10.0, -20.0, 90.0, 90.0, 64, 48)
    out = np.asarray(equirect_to_perspective(src, view))
    assert out.shape == (48, 64, 3)

    src_arr = np.asarray(src)
    for col, row in [(32, 24), (5, 40), (60, 3)]:
        u = (col + 0.5) / view.width
        v = (row + 0.5) / view.height
        x, y = perspective_point_to_equirect(u, v, view)
        # The renderer samples the same (X, Y) -> source pixel this scalar map returns.
        sx = min(int((x % 1.0) * sw), sw - 1)
        sy = min(int(y * sh), sh - 1)
        assert tuple(out[row, col]) == tuple(src_arr[sy, sx])


def test_constant_image_reprojects_constant():
    import numpy as np
    from PIL import Image
    src = Image.fromarray(np.full((180, 360, 3), 123, dtype="uint8"))
    out = np.asarray(equirect_to_perspective(src, LEVEL))
    assert (out == 123).all()


# --- dedup (seam handling) --------------------------------------------------

def test_dedup_merges_near_and_wraps_seam():
    from rampnet.detection_eval import radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y
    rsq = radius_sq_for()
    # Two near points -> one; a far point survives.
    pts = [(0.500, 0.500, 0.9), (0.505, 0.500, 0.5), (0.800, 0.500, 0.7)]
    kept = dedup_points(pts, rsq, PANO_SCALE_X, PANO_SCALE_Y)
    assert len(kept) == 2
    assert kept[0][0] == 0.500   # higher-confidence of the near pair kept first
    # Seam wrap: X=0.002 and X=0.998 are ~0.004 apart across the 0/1 edge -> merge.
    wrapped = dedup_points([(0.002, 0.5, None), (0.998, 0.5, None)], rsq, PANO_SCALE_X, PANO_SCALE_Y)
    assert len(wrapped) == 1


def test_default_views_overlap_for_seam_coverage():
    views = default_views()          # 6 views, 90 deg FOV -> 60 deg apart
    assert len(views) == 6
    step = views[1].yaw_deg - views[0].yaw_deg
    assert step < views[0].fov_h_deg   # neighboring views overlap
