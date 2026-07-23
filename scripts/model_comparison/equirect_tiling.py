"""Perspective reprojection of equirectangular panos for VLM detection.

VLMs are trained on ordinary rectilinear photos, not 360 equirectangular panos,
and curb ramps sit on the ground near the bottom of the pano where equirectangular
vertical stretch is worst. To give a VLM a fair input we reproject the pano into a
ring of overlapping **rectilinear virtual-camera views** (gnomonic projection,
pitched down toward the ground), detect in each undistorted view, then map each
detection's center back to pano-normalized coordinates and dedup across the
overlaps.

The load-bearing, unit-tested core here is the coordinate math:
``perspective_point_to_equirect`` (map a point in a view back to the pano) and its
inverse ``equirect_point_to_perspective`` (round-trip check + visibility test).
``equirect_to_perspective`` renders the actual view image for the live VLM call.

Conventions: world axes are right-handed with +z forward (pano longitude 0), +x
right/east, +y up. Equirectangular normalized coords ``(X, Y)`` in ``[0, 1]``:
``lon = (X-0.5)*2pi``, ``lat = (0.5-Y)*pi``.
"""
import math
from collections import namedtuple

View = namedtuple("View", ["yaw_deg", "pitch_deg", "fov_h_deg", "fov_v_deg", "width", "height"])

TWO_PI = 2.0 * math.pi


# --- vector helpers ---------------------------------------------------------

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    m = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / m, a[1] / m, a[2] / m)


# --- spherical <-> equirectangular ------------------------------------------

def _dir_from_equirect(x, y):
    lon = (x - 0.5) * TWO_PI
    lat = (0.5 - y) * math.pi
    cl = math.cos(lat)
    return (cl * math.sin(lon), math.sin(lat), cl * math.cos(lon))


def _equirect_from_dir(d):
    lon = math.atan2(d[0], d[2])
    lat = math.asin(max(-1.0, min(1.0, d[1])))
    x = (lon / TWO_PI + 0.5) % 1.0
    y = min(1.0, max(0.0, 0.5 - lat / math.pi))
    return x, y


def _camera_basis(yaw_deg, pitch_deg):
    """Right-handed, roll-free camera basis (forward, right, up) for a view
    looking at longitude=yaw, latitude=pitch."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cp = math.cos(pitch)
    forward = (cp * math.sin(yaw), math.sin(pitch), cp * math.cos(yaw))
    right = _norm(_cross((0.0, 1.0, 0.0), forward))
    up = _cross(forward, right)
    return forward, right, up


# --- point mapping (the tested core) ----------------------------------------

def perspective_point_to_equirect(u, v, view):
    """Map a point ``(u, v)`` (normalized [0,1] within the view, origin top-left)
    to pano-normalized ``(X, Y)``."""
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    a = (u * 2.0 - 1.0) * th
    b = (1.0 - v * 2.0) * tv
    ray = _norm((f[0] + a * r[0] + b * up[0],
                 f[1] + a * r[1] + b * up[1],
                 f[2] + a * r[2] + b * up[2]))
    return _equirect_from_dir(ray)


def equirect_point_to_perspective(x, y, view):
    """Inverse of ``perspective_point_to_equirect``. Returns ``(u, v)`` if the
    pano point falls inside this view, else ``None`` (behind camera or outside FOV).
    Used for round-trip tests and cross-view visibility checks."""
    d = _dir_from_equirect(x, y)
    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    zc = _dot(d, f)
    if zc <= 1e-9:
        return None
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)
    u = (_dot(d, r) / zc / th + 1.0) / 2.0
    v = (1.0 - _dot(d, up) / zc / tv) / 2.0
    if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
        return u, v
    return None


def default_views(fov_h_deg=90.0, fov_v_deg=90.0, pitch_deg=-30.0, n_yaw=6,
                   width=1024, height=1024):
    """A ring of ``n_yaw`` evenly spaced views pitched toward the ground. With the
    default 90 deg FOV and 6 yaws (60 deg apart) neighboring views overlap by 30 deg,
    so a ramp near a seam is seen whole in at least one view (dedup handles the
    double-detection)."""
    step = 360.0 / n_yaw
    return [View(i * step, pitch_deg, fov_h_deg, fov_v_deg, width, height)
            for i in range(n_yaw)]


# --- rendering (for the live VLM call) --------------------------------------

def equirect_to_perspective(pil_img, view):
    """Render one rectilinear view from an equirectangular PIL image (nearest-
    neighbor sampling — sufficient for feeding a detector). Vectorized mirror of
    ``perspective_point_to_equirect``."""
    import numpy as np
    from PIL import Image

    src = np.asarray(pil_img)
    sh, sw = src.shape[:2]
    W, H = view.width, view.height

    f, r, up = _camera_basis(view.yaw_deg, view.pitch_deg)
    th = math.tan(math.radians(view.fov_h_deg) / 2.0)
    tv = math.tan(math.radians(view.fov_v_deg) / 2.0)

    uu = ((np.arange(W) + 0.5) / W * 2.0 - 1.0) * th          # (W,)
    vv = (1.0 - (np.arange(H) + 0.5) / H * 2.0) * tv          # (H,)
    a = uu[None, :]                                           # (1,W)
    b = vv[:, None]                                           # (H,1)

    dx = f[0] + a * r[0] + b * up[0]
    dy = f[1] + a * r[1] + b * up[1]
    dz = f[2] + a * r[2] + b * up[2]
    inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx * inv, dy * inv, dz * inv

    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))
    sx = np.clip(((lon / TWO_PI + 0.5) % 1.0) * sw, 0, sw - 1).astype(np.int64)
    sy = np.clip((0.5 - lat / math.pi) * sh, 0, sh - 1).astype(np.int64)
    return Image.fromarray(src[sy, sx])


# --- cross-view dedup -------------------------------------------------------

def dedup_points(points, radius_sq, scale_x, scale_y):
    """Greedily merge points closer than the match radius (highest confidence
    kept first). X distance wraps at the 0/1 seam so a ramp split across the pano
    edge isn't double-counted."""
    ordered = sorted(points, key=lambda p: (p[2] if len(p) > 2 and p[2] is not None else float("-inf")),
                     reverse=True)
    kept = []
    for p in ordered:
        px, py = p[0], p[1]
        dup = False
        for k in kept:
            ddx = abs(px - k[0])
            ddx = min(ddx, 1.0 - ddx) * scale_x       # wrap at the seam
            ddy = (py - k[1]) * scale_y
            if ddx * ddx + ddy * ddy < radius_sq:
                dup = True
                break
        if not dup:
            kept.append(p)
    return kept
