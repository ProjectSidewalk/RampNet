"""Perspective photos <-> RampNet's equirect input (issue #218).

RampNet was trained on 2048x4096 equirect panoramas, i.e. at one angular scale:
4096 px per 360 deg. A perspective photo (phone, dashcam) can be fed to it two ways:

- **canvas embed**: reproject the photo into a 2048x4096 equirect canvas at its true
  field of view, so a ramp subtends the same angle it would in a panorama. Everything
  outside the photo is a constant fill;
- **naive stretch**: resize the photo to 2048x4096 (the strawman).

Conventions (all angles in radians unless a name says ``_deg``):

- **Camera frame**: OpenCV / OpenSfM -- x right, y down, z forward.
- **Canvas frame** ("level frame"): the same axes with the camera's pitch and roll
  removed, so z is horizontal along the camera heading and y is gravity-down.
  Canvas column ``c`` (0..W-1) has longitude ``lam = ((c + 0.5) / W) * 2 pi - pi``,
  so the photo's heading sits on the centre column; row ``r`` has latitude
  ``phi = pi / 2 - ((r + 0.5) / H) * pi`` (row 0 is straight up). A ray at (lam, phi)
  is ``(cos phi sin lam, -sin phi, cos phi cos lam)``.
- **World frame**: local east-north-up, metres.
- **Mapillary's camera model** (``camera_type == "perspective"``,
  ``camera_parameters = [f, k1, k2]``): OpenSfM's Brown model with the focal length
  normalised by ``max(width, height)``. A camera-frame ray (X, Y, Z) projects to
  ``xn = X / Z``, ``yn = Y / Z``, ``d = 1 + k1 r^2 + k2 r^4`` with ``r^2 = xn^2 + yn^2``,
  and pixel ``u = f d xn * S + w / 2 - 0.5``, ``v = f d yn * S + h / 2 - 0.5`` with
  ``S = max(w, h)``. Because f is normalised, it is the same for every resize of the
  image, including Mapillary's 2048-px thumbnails.
- **Mapillary's ``computed_rotation``** is an angle-axis world-to-camera rotation. On
  the 1,353 Richmond perspective images the heading it implies equals
  ``computed_compass_angle`` to 1e-11 deg (checked 2026-09-30), which pins the
  convention.

Example -- a 70 deg pinhole photo 1000 px wide, level; its right edge lands 35 deg
right of the canvas centre:

>>> import numpy as np
>>> cam = Camera(width=1000, height=750, focal=pinhole_focal_for_hfov(70.0, 1000, 750))
>>> u, v = project_cam(cam, level_ray(np.radians(35.0), 0.0))
>>> round(float(u), 1)
999.5
"""
import math

import numpy as np

CANVAS_H, CANVAS_W = 2048, 4096
EARTH_R = 6378137.0


# --------------------------------------------------------------------------- #
# camera model
# --------------------------------------------------------------------------- #
class Camera:
    """A Brown/OpenSfM perspective camera on an image of ``width`` x ``height`` px.

    ``focal`` is normalised by ``max(width, height)``, as Mapillary stores it."""

    def __init__(self, width, height, focal, k1=0.0, k2=0.0):
        self.width = int(width)
        self.height = int(height)
        self.focal = float(focal)
        self.k1 = float(k1)
        self.k2 = float(k2)

    @property
    def size(self):
        return max(self.width, self.height)

    def resized(self, width, height):
        """The same camera on a resized copy of the image (f is normalised)."""
        return Camera(width, height, self.focal, self.k1, self.k2)

    def hfov_deg(self):
        """Horizontal FOV edge to edge, ignoring distortion."""
        return 2 * math.degrees(math.atan(0.5 * self.width / self.size / self.focal))

    def vfov_deg(self):
        return 2 * math.degrees(math.atan(0.5 * self.height / self.size / self.focal))


def pinhole_focal_for_hfov(hfov_deg, width, height):
    """Normalised focal of a distortion-free camera with this horizontal FOV.

    >>> round(pinhole_focal_for_hfov(90.0, 1000, 500), 6)
    0.5
    """
    return 0.5 * width / max(width, height) / math.tan(math.radians(hfov_deg) / 2)


def project_cam(cam, rays):
    """Camera-frame rays (..., 3) -> pixel (u, v); NaN where the ray points backward."""
    rays = np.asarray(rays, dtype=np.float64)
    X, Y, Z = rays[..., 0], rays[..., 1], rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = np.where(Z > 1e-9, X / Z, np.nan)
        yn = np.where(Z > 1e-9, Y / Z, np.nan)
    r2 = xn * xn + yn * yn
    d = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
    u = cam.focal * d * xn * cam.size + cam.width / 2.0 - 0.5
    v = cam.focal * d * yn * cam.size + cam.height / 2.0 - 0.5
    return u, v


def unproject_cam(cam, u, v, iters=20):
    """Pixel (u, v) -> unit camera-frame ray (..., 3). Inverts the radial distortion by
    fixed-point iteration (converges for the |k| seen on Mapillary cameras)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    xd = (u + 0.5 - cam.width / 2.0) / (cam.size * cam.focal)
    yd = (v + 0.5 - cam.height / 2.0) / (cam.size * cam.focal)
    xn, yn = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = xn * xn + yn * yn
        d = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
        xn, yn = xd / d, yd / d
    ray = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def rotvec_to_matrix(rv):
    """Angle-axis -> 3x3 rotation matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def level_to_world(heading_deg):
    """3x3 matrix taking a level-frame vector to ENU, for a camera heading (deg clockwise
    from north). Level frame: x right, y down, z forward (horizontal)."""
    h = math.radians(heading_deg)
    fwd = np.array([math.sin(h), math.cos(h), 0.0])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    return np.stack([right, down, fwd], axis=1)


def world_to_cam_from_rotvec(rotvec):
    """Mapillary ``computed_rotation`` -> world(ENU)-to-camera matrix."""
    return rotvec_to_matrix(rotvec)


def heading_pitch_roll(R_wc):
    """(heading, pitch, roll) in degrees of a world-to-camera rotation. Pitch > 0 looks
    up; roll > 0 has the camera's x axis tilted up."""
    fwd = R_wc.T @ np.array([0.0, 0.0, 1.0])
    xr = R_wc.T @ np.array([1.0, 0.0, 0.0])
    head = math.degrees(math.atan2(fwd[0], fwd[1])) % 360
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, xr[2]))))
    return head, pitch, roll


def cam_from_level(R_wc=None, heading_deg=None):
    """3x3 matrix taking a level-frame ray to the camera frame.

    With ``R_wc`` (the SfM pose) this is ``R_wc @ level_to_world(heading)``, so the
    photo is placed at its true pitch and roll. With ``R_wc=None`` the camera is assumed
    level and the matrix is the identity."""
    if R_wc is None:
        return np.eye(3)
    if heading_deg is None:
        heading_deg = heading_pitch_roll(R_wc)[0]
    return R_wc @ level_to_world(heading_deg)


# --------------------------------------------------------------------------- #
# canvas
# --------------------------------------------------------------------------- #
def level_ray(lam, phi):
    """Level-frame unit ray at canvas longitude ``lam`` / latitude ``phi``."""
    lam = np.asarray(lam, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    return np.stack([np.cos(phi) * np.sin(lam), -np.sin(phi), np.cos(phi) * np.cos(lam)],
                    axis=-1)


def canvas_lam_phi(x_norm, y_norm):
    """Normalised canvas position (x in [0,1) across, y in [0,1) down) -> (lam, phi)."""
    return (np.asarray(x_norm) * 2 * math.pi - math.pi,
            math.pi / 2 - np.asarray(y_norm) * math.pi)


def ray_to_canvas_norm(ray):
    """Level-frame ray -> normalised canvas (x, y)."""
    ray = np.asarray(ray, dtype=np.float64)
    lam = np.arctan2(ray[..., 0], ray[..., 2])
    phi = np.arcsin(np.clip(-ray[..., 1] / np.linalg.norm(ray, axis=-1), -1, 1))
    return (lam + math.pi) / (2 * math.pi), (math.pi / 2 - phi) / math.pi


def canvas_sample_maps(cam, M_cam_level, H=CANVAS_H, W=CANVAS_W):
    """For every canvas pixel centre, the photo pixel (u, v) it samples, as float32
    (H, W) arrays; NaN where the ray misses the photo. ``M_cam_level`` is
    ``cam_from_level(...)``."""
    lam = ((np.arange(W) + 0.5) / W) * 2 * math.pi - math.pi
    phi = math.pi / 2 - ((np.arange(H) + 0.5) / H) * math.pi
    L, P = np.meshgrid(lam, phi)
    rays = level_ray(L, P) @ np.asarray(M_cam_level).T
    u, v = project_cam(cam, rays)
    inside = (u >= -0.5) & (u <= cam.width - 0.5) & (v >= -0.5) & (v <= cam.height - 0.5)
    # a strongly distorted model folds back on itself far outside its FOV; keep only rays
    # within the undistorted FOV plus a small margin
    xn = rays[..., 0] / np.where(rays[..., 2] > 1e-9, rays[..., 2], np.nan)
    yn = rays[..., 1] / np.where(rays[..., 2] > 1e-9, rays[..., 2], np.nan)
    lim_x = 0.5 * cam.width / cam.size / cam.focal * 1.25
    lim_y = 0.5 * cam.height / cam.size / cam.focal * 1.25
    inside &= (np.abs(xn) <= lim_x) & (np.abs(yn) <= lim_y)
    u = np.where(inside, u, np.nan).astype(np.float32)
    v = np.where(inside, v, np.nan).astype(np.float32)
    return u, v


def canvas_norm_to_cam_ray(x_norm, y_norm, M_cam_level):
    """A canvas detection -> unit camera-frame ray (the ray the canvas sampled)."""
    lam, phi = canvas_lam_phi(x_norm, y_norm)
    return level_ray(lam, phi) @ np.asarray(M_cam_level).T


def prescale_factor(cam, W=CANVAS_W):
    """Factor to shrink the photo by before canvas sampling so it is not aliased: the
    canvas has ``W / 2 pi`` px per radian, the photo ``f * S`` at its centre. Never > 1."""
    return min(1.0, (W / (2 * math.pi)) / (cam.focal * cam.size))


# --------------------------------------------------------------------------- #
# world geometry
# --------------------------------------------------------------------------- #
def enu_offset(lat0, lng0, lat, lng):
    """(east, north) metres of (lat, lng) from (lat0, lng0), local tangent plane.
    Good to millimetres over the 30 m scales used here."""
    lat = np.asarray(lat, dtype=np.float64)
    lng = np.asarray(lng, dtype=np.float64)
    n = np.radians(lat - lat0) * EARTH_R
    e = np.radians(lng - lng0) * EARTH_R * math.cos(math.radians(lat0))
    return e, n


def wrap_deg(a):
    """Wrap degrees into [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def ray_bearing_depression(world_ray):
    """ENU ray -> (bearing deg clockwise from north, depression deg below horizon)."""
    w = np.asarray(world_ray, dtype=np.float64)
    b = np.degrees(np.arctan2(w[..., 0], w[..., 1])) % 360
    dep = -np.degrees(np.arcsin(np.clip(w[..., 2] / np.linalg.norm(w, axis=-1), -1, 1)))
    return b, dep


def bearing_hit(det_bearing, det_depression, ramp_bearing, ramp_range, lateral_m=5.0,
                h_min=0.5, h_max=4.0):
    """The height-free hit test used for flat photos, whose camera heights are unknown.

    A detection hits a ramp at horizontal range ``ramp_range`` when

    - its bearing is within ``atan(lateral_m / range)`` of the ramp's (eval_sites' 5 m
      match radius, expressed as a lateral offset at the ramp's range), and
    - it is below the horizon at a depression that puts the ramp on flat ground for some
      camera height in ``[h_min, h_max]`` m: ``h = range * tan(depression)``.

    >>> bearing_hit(10.0, 8.0, 12.0, 10.0)   # 2 deg off at 10 m; h = 1.41 m
    True
    >>> bearing_hit(10.0, 8.0, 50.0, 10.0)
    False
    """
    tol = np.degrees(np.arctan2(lateral_m, ramp_range))
    ok_b = np.abs(wrap_deg(np.asarray(det_bearing) - ramp_bearing)) <= tol
    dep = np.asarray(det_depression)
    h = ramp_range * np.tan(np.radians(np.clip(dep, 1e-6, 89.0)))
    ok_h = (dep > 0) & (h >= h_min) & (h <= h_max)
    return ok_b & ok_h


def raycast_ground(world_ray, cam_height):
    """(east, north) where an ENU ray from a camera at height ``cam_height`` meets flat
    ground; NaN for rays at or above the horizon."""
    w = np.asarray(world_ray, dtype=np.float64)
    down = -w[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-9, cam_height / down, np.nan)
    return w[..., 0] * t, w[..., 1] * t
