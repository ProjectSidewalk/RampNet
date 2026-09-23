"""Crops of a Project Sidewalk label from its equirectangular panorama (#86, plan item 2b).

A crop here is a **true perspective (gnomonic) view**, not a rectangle cut out of the
equirect. The equirect stretches everything horizontally by 1/cos(latitude), and a curb
ramp 20-40 degrees below the horizon is exactly where that bites; a gnomonic view is what a
camera (and the Project Sidewalk viewer, and every crop a tag model has ever been trained
on) actually shows. A plain equirect window is offered too (:func:`equirect_window`), for a
consumer that wants CropRunner's framing.

Two ways to aim the view:

* **centered** (:func:`centered_view`): the view looks straight at the label's stored pano
  point, with a horizontal field of view you choose. The label lands at the exact centre.
* **viewport** (:func:`viewport_view`): the view the labeler was looking at when they placed
  the label: the stored ``heading``/``pitch``, the viewer's field of view for the stored
  ``zoom`` (:func:`get_3d_fov`), 3:2. The label lands at ``(canvas_x, canvas_y)`` of the
  720x480 canvas, not at the centre. This is the framing of the HF
  ``sidewalk-tagger-ai-validated`` crops (1440x960 = the canvas at 2x), which item 4
  compares against.

Conventions (identical to ``scripts/model_comparison/equirect_tiling.py``, which this
module is cross-checked against in the tests):

* Normalized pano coordinates ``X = pano_x / pano_width``, ``Y = pano_y / pano_height``.
  ``lon = (X - 0.5) * 360`` degrees, clockwise (to the right) positive; ``lat = (0.5 - Y) * 180``
  degrees, up positive. The pano's centre column looks along ``camera_heading``, so
  ``lon = heading - camera_heading`` -- the same mapping as Project Sidewalk's
  ``calculatePanoXYFromPov``. Everything is resolution-independent: a store image at a
  different resolution from the label's ``pano_width`` is sampled at the same normalized point.
* World axes: +z forward (lon 0), +x right, +y up. Views are roll-free (the horizon is level).
* Only x wraps. Column 0 and column ``pano_width`` are the same place; the poles are not.

Camera tilt. Project Sidewalk's click-to-pano mapping ignores the rig's pitch and roll
(``camera_pitch``/``camera_roll``); the GSV viewer does not. :func:`tilt_matrix` rotates the
view's rays from the level (viewer) frame into the pano image frame for a given sign
convention. The default everywhere is **no tilt**, which is the frame the stored ``pano_x``/
``pano_y`` are in. See ``docs/crop_cutter.md`` for what the validation run measured.

Pure functions of numpy arrays; the only I/O helpers are :func:`encode_jpeg` and
:func:`sha256_hex`. Usage::

    import numpy as np
    from rampnet.crops import centered_view, render_view
    pano = np.asarray(PIL.Image.open("pano.jpg"))              # (H, W, 3) uint8
    view = centered_view(13740, 4754, 16384, 8192, fov_h_deg=60, width=1440)
    crop = render_view(pano, view)                             # (960, 1440, 3) uint8
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: The Project Sidewalk labeling canvas, which ``canvas_x``/``canvas_y`` are measured on.
CANVAS_W = 720
CANVAS_H = 480
#: Production and HF crops are 3:2 (1440x960, the canvas at 2x).
DEFAULT_ASPECT = CANVAS_W / CANVAS_H
DEFAULT_WIDTH = 2 * CANVAS_W

#: Named tilt conventions for :func:`tilt_matrix`: (sign on camera_pitch, sign on camera_roll).
TILT_CONVENTIONS = {
    "pp": (1, 1),
    "pm": (1, -1),
    "mp": (-1, 1),
    "mm": (-1, -1),
}


def get_3d_fov(zoom):
    """The Project Sidewalk viewer's horizontal field of view, in degrees, for a GSV zoom.

    ``UtilitiesPanomarker.get3dFov`` in SidewalkWebpage, as ported in
    sidewalk-panorama-tools ``reports/scripts/pov_replay.py``: zoom 1 -> 89.75, 2 -> 53,
    3 -> 27.68 (the JS's "determined experimentally" branch above zoom 2). It is the angle
    across the canvas *width* (focal length ``f = (canvas_width / 2) / tan(fov / 2)``).
    """
    z = np.asarray(zoom, dtype=float)
    out = np.where(z <= 2, 126.5 - z * 36.75, 195.93 / np.power(1.92, z))
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class View:
    """A roll-free perspective camera on the pano sphere.

    ``yaw_deg``/``pitch_deg`` aim the optical axis (lon/lat, degrees); ``fov_h_deg`` is the
    angle across ``width`` pixels; the vertical angle follows from the pixel aspect (square
    pixels), ``tan(fov_v / 2) = tan(fov_h / 2) * height / width``.
    """
    yaw_deg: float
    pitch_deg: float
    fov_h_deg: float
    width: int
    height: int

    @property
    def fov_v_deg(self) -> float:
        t = math.tan(math.radians(self.fov_h_deg) / 2.0) * self.height / self.width
        return math.degrees(2.0 * math.atan(t))

    @property
    def focal_px(self) -> float:
        """Focal length in output pixels."""
        return 0.5 * self.width / math.tan(math.radians(self.fov_h_deg) / 2.0)


def wrap180(deg):
    """Wrap an angle to [-180, 180)."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def pano_px_to_lonlat(pano_x, pano_y, pano_width, pano_height):
    """Stored pano pixel -> (lon, lat) in degrees. Resolution-independent; lon wraps."""
    X = np.asarray(pano_x, dtype=float) / np.asarray(pano_width, dtype=float)
    Y = np.asarray(pano_y, dtype=float) / np.asarray(pano_height, dtype=float)
    return wrap180((X - 0.5) * 360.0), (0.5 - Y) * 180.0


def lonlat_to_pano_px(lon, lat, pano_width, pano_height):
    """Inverse of :func:`pano_px_to_lonlat` (continuous, x in [0, pano_width))."""
    X = (np.asarray(lon, dtype=float) / 360.0 + 0.5) % 1.0
    Y = 0.5 - np.asarray(lat, dtype=float) / 180.0
    return X * pano_width, Y * pano_height


def _size(width: int, aspect: float) -> Tuple[int, int]:
    width = int(width)
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    return width, max(1, int(round(width / float(aspect))))


def centered_view(pano_x, pano_y, pano_width, pano_height, fov_h_deg,
                  width=DEFAULT_WIDTH, aspect=DEFAULT_ASPECT) -> View:
    """A view looking straight at the label's stored pano point."""
    if not 0.0 < float(fov_h_deg) < 180.0:
        raise ValueError(f"fov_h_deg must be in (0, 180), got {fov_h_deg}")
    lon, lat = pano_px_to_lonlat(pano_x, pano_y, pano_width, pano_height)
    w, h = _size(width, aspect)
    return View(float(lon), float(lat), float(fov_h_deg), w, h)


def viewport_view(heading, pitch, zoom, camera_heading,
                  width=DEFAULT_WIDTH, aspect=DEFAULT_ASPECT) -> View:
    """The labeler's own view: stored POV, the viewer's FOV for the stored zoom, 3:2."""
    w, h = _size(width, aspect)
    return View(float(wrap180(float(heading) - float(camera_heading))), float(pitch),
                float(get_3d_fov(zoom)), w, h)


def label_pixel_in_viewport(canvas_x, canvas_y, view: View):
    """Where Project Sidewalk says the label sits in a viewport crop, in output pixels."""
    return (float(canvas_x) / CANVAS_W * view.width, float(canvas_y) / CANVAS_H * view.height)


def tilt_matrix(camera_pitch, camera_roll, convention: Optional[str] = None) -> Optional[np.ndarray]:
    """Rotation taking a level-frame ray into the pano image frame, or None for no tilt.

    ``convention`` picks the signs on (pitch, roll) from :data:`TILT_CONVENTIONS`, because
    GSV's is not documented where this code can check it; ``docs/crop_cutter.md`` records
    which one (if any) the validation run supports. Pitch rotates about the pano's x (right)
    axis, roll about its z (forward = lon 0) axis. To first order a level ray at relative
    azimuth ``phi`` moves ``sp * pitch * cos(phi) + sr * roll * sin(phi)`` degrees in
    elevation, the sinusoid label-latlng-estimation's ``tilt_probe`` tests.
    """
    if convention in (None, "", "none"):
        return None
    sp, sr = TILT_CONVENTIONS[convention]
    p = math.radians(sp * float(camera_pitch or 0.0))
    r = math.radians(sr * float(camera_roll or 0.0))
    cp, spn = math.cos(p), math.sin(p)
    cr, srn = math.cos(r), math.sin(r)
    # Rx(p): pitching the forward axis (0,0,1) up to (0, sin p, cos p).
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, spn], [0.0, -spn, cp]])
    # Rz(r): rolling the right axis (1,0,0) up to (cos r, sin r, 0).
    rz = np.array([[cr, -srn, 0.0], [srn, cr, 0.0], [0.0, 0.0, 1.0]])
    # R = rz @ rx is the rig's attitude (image-frame axes expressed in the level frame). A
    # level ray d is R^T d in the image frame; with row vectors that is ``d @ R``, which is
    # how :func:`rays_to_lonlat` applies it.
    return rz @ rx


def _basis(yaw_deg: float, pitch_deg: float):
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cp = math.cos(pitch)
    fwd = np.array([cp * math.sin(yaw), math.sin(pitch), cp * math.cos(yaw)])
    right = np.cross([0.0, 1.0, 0.0], fwd)
    n = np.linalg.norm(right)
    if n < 1e-12:  # looking straight up or down: any right vector is valid
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    else:
        right = right / n
    up = np.cross(fwd, right)
    return fwd, right, up


def view_rays(view: View) -> np.ndarray:
    """Unit ray (level frame) through every output pixel centre, shape (H, W, 3)."""
    fwd, right, up = _basis(view.yaw_deg, view.pitch_deg)
    f = view.focal_px
    a = (np.arange(view.width, dtype=np.float64) + 0.5 - view.width / 2.0) / f
    b = (view.height / 2.0 - (np.arange(view.height, dtype=np.float64) + 0.5)) / f
    d = fwd[None, None, :] + a[None, :, None] * right[None, None, :] + b[:, None, None] * up[None, None, :]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def rays_to_lonlat(d: np.ndarray, tilt: Optional[np.ndarray] = None):
    """Level-frame rays -> image-frame (lon, lat) in degrees."""
    if tilt is not None:
        d = d @ tilt  # row-vector form of tilt.T @ d
    lon = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    lat = np.degrees(np.arcsin(np.clip(d[..., 1], -1.0, 1.0)))
    return lon, lat


def lonlat_to_ray(lon, lat) -> np.ndarray:
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], axis=-1)


def project_lonlat(lon, lat, view: View, tilt: Optional[np.ndarray] = None):
    """Image-frame (lon, lat) -> output pixel (x, y), continuous; NaN if behind the camera.

    Pixel (0, 0) is the top-left corner of the top-left pixel, so the view centre is
    ``(width / 2, height / 2)``. Inverse of :func:`view_rays` + :func:`rays_to_lonlat`.
    """
    d = lonlat_to_ray(lon, lat)
    if tilt is not None:
        d = d @ tilt.T  # back to the level frame
    fwd, right, up = _basis(view.yaw_deg, view.pitch_deg)
    zc = d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        x = view.width / 2.0 + view.focal_px * (d @ right) / zc
        y = view.height / 2.0 - view.focal_px * (d @ up) / zc
    bad = zc <= 1e-9
    x = np.where(bad, np.nan, x)
    y = np.where(bad, np.nan, y)
    return (float(x), float(y)) if np.ndim(x) == 0 else (x, y)


def sample_bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img`` (H, W[, C]) at continuous pixel coords (centre = i + 0.5).

    x wraps (the seam), y clamps to the edge rows (the poles do not wrap).
    """
    H, W = img.shape[:2]
    fx = np.asarray(sx, dtype=np.float64) - 0.5
    fy = np.clip(np.asarray(sy, dtype=np.float64) - 0.5, 0.0, H - 1.0)
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    wx = (fx - x0)[..., None].astype(np.float32)
    wy = (fy - y0)[..., None].astype(np.float32)
    x0 = x0.astype(np.int64) % W
    x1 = (x0 + 1) % W
    y0 = y0.astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    src = img if img.ndim == 3 else img[..., None]
    top = src[y0, x0].astype(np.float32) * (1 - wx) + src[y0, x1].astype(np.float32) * wx
    bot = src[y1, x0].astype(np.float32) * (1 - wx) + src[y1, x1].astype(np.float32) * wx
    out = top * (1 - wy) + bot * wy
    out = np.clip(np.rint(out), 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out.astype(img.dtype)
    return out if img.ndim == 3 else out[..., 0]


def reduction_factor(pano_width: int, view: View) -> int:
    """Largest integer box-reduction of the pano that keeps it at least as sharp as the view.

    Compares the pano's angular resolution (``pano_width / 360`` px/deg) with the view's at
    its centre (``focal_px * pi / 180`` px/deg). Reducing by this factor before bilinear
    sampling is the anti-aliasing: without it a 90-degree view of a 16384-px pano samples
    one source pixel in three.
    """
    src_ppd = pano_width / 360.0
    out_ppd = view.focal_px * math.pi / 180.0
    return max(1, int(math.floor(src_ppd / out_ppd)))


def render_view(pano: np.ndarray, view: View, tilt: Optional[np.ndarray] = None) -> np.ndarray:
    """Render ``view`` from an equirect pano array. The caller handles anti-aliasing
    (:func:`reduction_factor`); this samples whatever resolution it is given."""
    H, W = pano.shape[:2]
    lon, lat = rays_to_lonlat(view_rays(view), tilt)
    sx, sy = lonlat_to_pano_px(lon, lat, W, H)
    return sample_bilinear(pano, sx, sy)


def equirect_window(pano: np.ndarray, pano_x, pano_y, pano_width, pano_height, fov_h_deg,
                    width=DEFAULT_WIDTH, aspect=DEFAULT_ASPECT):
    """A plain rectangle of the equirect, CropRunner's framing: ``fov_h_deg`` of azimuth wide,
    ``1/aspect`` as tall, centred on the label, x wrapping at the seam, y shifted (never
    padded) at the poles. Resized to ``width`` with Lanczos. Returns
    ``(crop, shifted_px, label_xy)``: ``shifted_px`` is how far (source pixels) the window was
    moved vertically to stay in frame (0 for every real label in sidewalk-panorama-tools'
    clamp census), and ``label_xy`` is where the label lands in the output crop.
    """
    from PIL import Image

    H, W = pano.shape[:2]
    X = float(pano_x) / float(pano_width)
    Y = float(pano_y) / float(pano_height)
    ww = max(1, int(round(float(fov_h_deg) / 360.0 * W)))
    wh = max(1, int(round(ww / float(aspect))))
    wh = min(wh, H)
    left = int(round(X * W - ww / 2.0))
    top0 = int(round(Y * H - wh / 2.0))
    top = min(max(top0, 0), H - wh)
    cols = (np.arange(left, left + ww) % W)
    win = pano[top:top + wh][:, cols]
    out_w, out_h = _size(width, aspect)
    img = Image.fromarray(win).resize((out_w, out_h), Image.LANCZOS)
    shift = top - top0
    label_xy = ((X * W - left) * out_w / ww, (Y * H - top) * out_h / wh)
    return np.asarray(img), shift, label_xy


def encode_jpeg(arr: np.ndarray, quality: int = 92) -> bytes:
    """Deterministic baseline JPEG (no EXIF, no timestamp), so the bytes hash stably."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality), optimize=False,
                              progressive=False, subsampling=0)
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
