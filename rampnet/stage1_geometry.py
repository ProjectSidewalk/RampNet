"""Stage 1 crop geometry, in floats, plus the rig-tilt frame functions it needs (#113).

Two pieces live here because ``scripts/analysis/crop_tilt_113.py`` and its test both need them and
neither belongs in a paper-era script:

1. :func:`equirect_point_to_strip` is ``stage_one/crop_model/ps_model/data/download_data.py``'s
   ``equirectangular_point_to_perspective`` with the final ``int()`` removed. The paper-era function
   truncates, which is fine for writing a filename and useless for measuring a sub-pixel shift.
   ``tests/test_crop_tilt_113.py`` checks that ``int()`` of this function equals the original on
   random points, so the two cannot drift apart silently.

2. The tilt functions are a **vendored copy** of ``reports/scripts/tilt_geometry.py`` from
   `sidewalk-panorama-tools <https://github.com/ProjectSidewalk/sidewalk-panorama-tools>`_ at commit
   ``21d10aa3767167e67a098557f58f327def2396a5`` (the 2026-09-26 tilt error study, its PR #158).
   Copied rather than imported so that RampNet's tests run from a clean clone without the sibling
   repo. Only the functions this analysis uses are copied, unchanged apart from comments; the test
   pins them against values computed by the original module.

Sign conventions (measured in pano-tools' study, endpoint F1, not assumed): pose pitch > 0 is nose
down, roll > 0 is left side up; ``T(b) = pitch cos b + roll sin b`` is how far the gravity horizon
at bearing ``b`` sits above the rig horizon; a point stored in gravity-levelled pixels sits at
``y - T(b) * h / 180`` in the rig-frame raster that GSV actually serves (to first order).

Usage::

    from rampnet.stage1_geometry import equirect_point_to_strip, rig_pixel_from_gravity_pixel
    xr, yr = rig_pixel_from_gravity_pixel(8000.0, 5000.0, 16384, 8192, pitch_deg=2.0, roll_deg=0.0)
    sx, sy = equirect_point_to_strip(8000.0 / 16384, 5000.0 / 8192, theta_deg=0)
"""

import numpy as np

#: The fixed Stage 1 strip: a 90-degree, 2048 x 2048 perspective view at depression 30 degrees,
#: whose middle third (columns 682..1364) is kept. ``download_data.py`` hardcodes all of these.
STRIP_FOV_DEG = 90.0
STRIP_PHI_DEG = -30.0
STRIP_RENDER_SIZE = 2048
#: ``download_data.py`` subtracts the float 2048/3 from the point but slices the image at int(2048/3).
STRIP_X_OFFSET = STRIP_RENDER_SIZE / 3.0
STRIP_SLICE = (int(STRIP_RENDER_SIZE / 3), int(STRIP_RENDER_SIZE / 3 * 2))   # (682, 1365)


def equirect_point_to_perspective_float(label_x, label_y, equi_width, equi_height, fov, theta, phi,
                                        height, width):
    """``download_data.equirectangular_point_to_perspective`` without the final ``int()``.

    Returns ``(x, y)`` floats in the 2048 x 2048 render, or None behind the camera. Scalar inputs only,
    as in the original.
    """
    lon = np.deg2rad((label_x / equi_width) * 360.0 - 180.0)
    lat = np.deg2rad(90.0 - (label_y / equi_height) * 180.0)
    point = np.array([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)])
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
    forward = np.array([np.cos(phi_rad) * np.cos(theta_rad), np.sin(phi_rad),
                        np.cos(phi_rad) * np.sin(theta_rad)])
    forward /= np.linalg.norm(forward)
    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    p_cam_x, p_cam_y, p_cam_z = np.dot(point, right), np.dot(point, up), np.dot(point, forward)
    if p_cam_z <= 0:
        return None
    f = (width / 2) / np.tan(np.deg2rad(fov) / 2)
    return float(p_cam_x * f / p_cam_z + width / 2), float(-(p_cam_y * f / p_cam_z) + height / 2)


def nearest_strip_theta(x_norm):
    """``download_data.py``'s strip choice: the label's bearing from the pano centre, rounded to 30."""
    theta = x_norm * 360 - 180
    return round(theta / 30) * 30


def equirect_point_to_strip(x_norm, y_norm, theta_deg, equi_width=8192, equi_height=4096):
    """Where a normalised equirect point lands in the kept strip, in **render px** (float).

    ``download_data.py`` projects into the 2048 x 2048 render, then subtracts 2048/3 from x. The
    equirect is resized to 8192 x 4096 first; only the normalised coordinates matter here, the
    resolution is passed through to keep the arithmetic identical. None if behind the camera.
    """
    res = equirect_point_to_perspective_float(x_norm * equi_width, y_norm * equi_height, equi_width,
                                              equi_height, STRIP_FOV_DEG, theta_deg, STRIP_PHI_DEG,
                                              STRIP_RENDER_SIZE, STRIP_RENDER_SIZE)
    if res is None:
        return None
    return res[0] - STRIP_X_OFFSET, res[1]


# --- vendored from sidewalk-panorama-tools reports/scripts/tilt_geometry.py @ 21d10aa ---------------

def wrap_deg(a):
    """Wrap degrees to (-180, 180]. photometa serves roll in [0, 360), so 359.6 means -0.4."""
    a = np.asarray(a, dtype=float)
    out = -((-a + 180.0) % 360.0) + 180.0
    return float(out) if out.ndim == 0 else out


def direction_rfu(bearing_deg, elevation_deg):
    """Unit vectors (..., 3) in (right, forward, up) for bearings/elevations in degrees."""
    b = np.radians(np.asarray(bearing_deg, dtype=float))
    e = np.radians(np.asarray(elevation_deg, dtype=float))
    b, e = np.broadcast_arrays(b, e)
    return np.stack([np.cos(e) * np.sin(b), np.cos(e) * np.cos(b), np.sin(e)], axis=-1)


def bearing_elevation(v):
    """(bearing_deg, elevation_deg) of RFU vectors (..., 3); the inverse of direction_rfu."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        el = np.degrees(np.arcsin(np.clip(v[..., 2] / norm, -1.0, 1.0)))
    b = np.degrees(np.arctan2(v[..., 0], v[..., 1]))
    b = np.where(norm > 0, wrap_deg(b), np.nan)
    el = np.where(norm > 0, el, np.nan)
    if b.ndim == 0:
        return float(b), float(el)
    return b, el


def _rx(a_deg):
    a = np.radians(np.asarray(a_deg, dtype=float))
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a_deg):
    a = np.radians(np.asarray(a_deg, dtype=float))
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def rig_from_gravity(pitch_deg, roll_deg):
    """3x3 R (or (..., 3, 3)) with v_rig = R @ v_gravity, both RFU; intrinsic pitch then roll."""
    pitch_deg, roll_deg = np.broadcast_arrays(np.asarray(pitch_deg, dtype=float),
                                              np.asarray(roll_deg, dtype=float))
    axes = _rx(-pitch_deg) @ _ry(roll_deg)
    return np.swapaxes(axes, -1, -2)


def gravity_to_rig(bearing_deg, elevation_deg, pitch_deg, roll_deg):
    """Exact: where a gravity-frame direction appears in the rig frame -> (bearing_rig, el_rig)."""
    v = direction_rfu(bearing_deg, elevation_deg)
    return bearing_elevation(np.einsum('...ij,...j->...i', rig_from_gravity(pitch_deg, roll_deg), v))


def tilt_term_deg(bearing_deg, pitch_deg, roll_deg):
    """First-order T(b) = pitch cos b + roll sin b (degrees)."""
    b = np.radians(np.asarray(bearing_deg, dtype=float))
    return pitch_deg * np.cos(b) + roll_deg * np.sin(b)


def pixel_from_bearing_elevation(bearing_deg, elevation_deg, pano_width, pano_height):
    """Continuous (x, y) on a heading-centred equirectangular raster."""
    b = np.asarray(bearing_deg, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    return ((b + 180.0) / 360.0 * pano_width) % pano_width, (0.5 - el / 180.0) * pano_height


def bearing_elevation_from_pixel(x, y, pano_width, pano_height):
    """Inverse of pixel_from_bearing_elevation; bearing wrapped to (-180, 180]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return wrap_deg(x / pano_width * 360.0 - 180.0), 90.0 - y / pano_height * 180.0


def rig_pixel_from_gravity_pixel(x, y, pano_width, pano_height, pitch_deg, roll_deg):
    """Where a point stored in gravity-levelled pixels sits in a rig-frame raster (exact)."""
    b, el = bearing_elevation_from_pixel(x, y, pano_width, pano_height)
    return pixel_from_bearing_elevation(*gravity_to_rig(b, el, pitch_deg, roll_deg), pano_width,
                                        pano_height)


def xml_tilt_to_pitch_roll(pano_yaw_deg, tilt_yaw_deg, tilt_pitch_deg):
    """The 2019 XML endpoint's projection_properties -> (pitch_deg, roll_deg) in streetlevel's sign."""
    d = np.radians(np.asarray(tilt_yaw_deg, dtype=float) - np.asarray(pano_yaw_deg, dtype=float))
    m = np.asarray(tilt_pitch_deg, dtype=float)
    return -m * np.cos(d), -m * np.sin(d)
