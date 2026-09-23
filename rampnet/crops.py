"""Crops of a Project Sidewalk label from its equirectangular panorama (#86, plan item 2b).

A crop here is a **true perspective (gnomonic) view**, not a rectangle cut out of the
equirect. The equirect stretches everything horizontally by 1/cos(latitude), and a curb
ramp 20-40 degrees below the horizon is exactly where that bites; a gnomonic view is what a
camera (and the Project Sidewalk viewer, and every crop a tag model has ever been trained
on) actually shows. A plain equirect window is offered too (:func:`equirect_window`), for a
consumer that wants CropRunner's framing.

Two ways to aim the view:

* **centered** (:func:`centered_view`): the view looks straight at the label's stored pano
  point, with a horizontal field of view you choose. Rendered with the viewer's tilt (below),
  the clicked point lands at the exact centre.
* **viewport** (:func:`viewport_view`): the view the labeler was looking at when they placed
  the label: the stored ``heading``/``pitch``, the viewer's field of view for the stored
  ``zoom`` (:func:`get_3d_fov`), 3:2. Rendered with the viewer's tilt, the label lands at
  ``(canvas_x, canvas_y)`` of the 720x480 canvas, not at the centre. This is the framing of the HF
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

Camera tilt. The stored ``heading``/``pitch`` and ``pano_x``/``pano_y`` are all in the
**level (viewer) frame**: Project Sidewalk's ``povToPanoCoord`` is a linear map of the viewer's
heading and pitch that ignores the rig's ``camera_pitch``/``camera_roll``, while the GSV viewer
shows the image rotated by them (the known y-error, SidewalkWebpage#4784). :func:`tilt_matrix`
rotates the view's level-frame rays into the equirect image frame. The validation run
(``docs/crop_cutter.md`` section 3) measured that the viewer's relation is convention
:data:`VIEWER_TILT` (``"mm"``): a level-frame ray at azimuth ``phi`` (relative to the pano's
centre column) and elevation ``lat`` sits in the image at elevation, to first order::

    image_lat = lat + camera_pitch * cos(phi) + camera_roll * sin(phi)

(the roll sign is untested: ``camera_roll`` is empty in every row measured). Rendering with
that tilt puts the clicked point where the labeler saw it: at the centre of a centred view,
at ``(canvas_x, canvas_y)`` of a viewport. Rendering with no tilt treats the stored point as a
raw image pixel, which is off the click by the tilt; :func:`label_pixel` says where the click
lands for any render tilt.

Pure functions of numpy arrays; the only I/O helpers are :func:`encode_jpeg` and
:func:`sha256_hex`. Usage::

    import numpy as np
    from rampnet.crops import centered_view, render_view
    pano = np.asarray(PIL.Image.open("pano.jpg"))              # (H, W, 3) uint8
    view = centered_view(13740, 4754, 16384, 8192, fov_h_deg=60, width=1440)
    tilt = tilt_matrix(camera_pitch, camera_roll, VIEWER_TILT)   # GSV panos; None if unknown
    crop = render_view(pano, view, tilt)                         # (960, 1440, 3) uint8
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

#: The convention the GSV viewer uses, as measured against the HF viewport screenshots
#: (``docs/crop_cutter.md`` section 3): image_lat = level_lat + camera_pitch * cos(phi) +
#: camera_roll * sin(phi). The default render tilt for gnomonic crops of GSV panos.
VIEWER_TILT = "mm"


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
    """Where Project Sidewalk's canvas point sits in a viewport crop rendered in the level frame
    (i.e. with the viewer's tilt, or on a pano with none), in output pixels.

    The canvas is 720x480 and the view's focal length is set by its *width*, so both axes scale
    by ``width / 720`` about the centre; for the default 3:2 output this is ``canvas * 2``.
    """
    s = view.width / float(CANVAS_W)
    return (view.width / 2.0 + (float(canvas_x) - CANVAS_W / 2.0) * s,
            view.height / 2.0 + (float(canvas_y) - CANVAS_H / 2.0) * s)


def tilt_matrix(camera_pitch, camera_roll, convention: Optional[str] = None) -> Optional[np.ndarray]:
    """Rotation taking a level-frame ray into the pano image frame, or None for no tilt.

    ``convention`` picks the signs ``(sp, sr)`` on (pitch, roll) from :data:`TILT_CONVENTIONS`.
    Pitch rotates about the pano's x (right) axis, roll about its z (forward = lon 0) axis. To
    first order a level-frame ray at relative azimuth ``phi`` lands in the image at elevation

        image_lat = level_lat - sp * camera_pitch * cos(phi) - sr * camera_roll * sin(phi)

    so ``"mm"`` (:data:`VIEWER_TILT`, the one the validation run supports) is
    ``image_lat = level_lat + camera_pitch * cos(phi) + camera_roll * sin(phi)``: with
    ``camera_pitch = 3`` a level ray straight ahead (phi = 0) is at image lat +3, and one behind
    (phi = 180) at -3. ``tests/test_crops.py`` pins these signed values. This is the sinusoid
    label-latlng-estimation's ``tilt_probe`` fits; port the relation, not the name.
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


def label_pixel(pano_x, pano_y, pano_width, pano_height, view: View,
                render_tilt: Optional[np.ndarray] = None, true_tilt: Optional[np.ndarray] = None):
    """Where the *clicked* point lands in a crop rendered with ``render_tilt``, in output pixels.

    The stored ``pano_x``/``pano_y`` are the click in the level frame. Its content sits in the
    image at the level point rotated by ``true_tilt`` (the viewer's tilt for this pano, from
    :func:`tilt_matrix` with :data:`VIEWER_TILT`, or None where no tilt is known); the crop shows
    image content through ``render_tilt``. When the two are the same rotation the click is where
    the level-frame view puts it (the centre of a centred view); otherwise it is off by the
    difference, which is what a ``--tilt none`` crop of a tilted pano carries.

    Example: seattle-wa:9's stored point (phi = 121.9 degrees) on a pano with
    ``camera_pitch = 3``: a 30-degree centred view rendered with no tilt shows the click about
    74 px below and 31 px left of centre; rendered with the viewer's tilt, at the centre::

        v = centered_view(13740, 4754, 16384, 8192, 30)
        T = tilt_matrix(3, 0, VIEWER_TILT)
        label_pixel(13740, 4754, 16384, 8192, v, None, T)   # ~(688.6, 553.7)
        label_pixel(13740, 4754, 16384, 8192, v, T, T)      # (720.0, 480.0)
    """
    lon, lat = pano_px_to_lonlat(pano_x, pano_y, pano_width, pano_height)
    ilon, ilat = rays_to_lonlat(lonlat_to_ray(lon, lat), true_tilt)
    return project_lonlat(ilon, ilat, view, render_tilt)


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
                    width=DEFAULT_WIDTH, aspect=DEFAULT_ASPECT, point=None):
    """A plain rectangle of the equirect, CropRunner's framing: ``fov_h_deg`` of azimuth wide,
    ``1/aspect`` as tall, centred on the label, x wrapping at the seam, y shifted (never
    padded) at the poles. Resized to ``width`` with Lanczos. Returns
    ``(crop, shifted_px, label_xy)``: ``shifted_px`` is how far (source pixels) the window was
    moved vertically to stay in frame (0 for every real label in sidewalk-panorama-tools'
    clamp census), and ``label_xy`` is where the label lands in the output crop: the stored
    point, or ``point`` (another ``(x, y)`` in the same pano_x/pano_y units, e.g. where the
    viewer's tilt puts the click's image content) when given. The window is never rotated, so
    it is centred on the raw stored pixel, as CropRunner's is.
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
    if point is not None:
        px = float(point[0]) / float(pano_width)
        dx = (px - X + 0.5) % 1.0 - 0.5          # wrap: the point is near the stored one
        X, Y = X + dx, float(point[1]) / float(pano_height)
    label_xy = ((X * W - left) * out_w / ww, (Y * H - top) * out_h / wh)
    return np.asarray(img), shift, label_xy


def black_fraction(arr: np.ndarray, thresh: int = 8) -> float:
    """Share of pixels whose brightest channel is <= ``thresh``: a store-integrity signal.

    Some store panos have tiles that never downloaded and were stitched as black. Real
    imagery is almost never this dark over more than a sliver, so a crop with a visible
    black fraction is cut from a damaged pano, not a dark scene.
    """
    a = np.asarray(arr)
    m = a.max(axis=-1) if a.ndim == 3 else a
    return float((m <= thresh).mean())


def encode_jpeg(arr: np.ndarray, quality: int = 92) -> bytes:
    """Deterministic baseline JPEG (no EXIF, no timestamp), so the bytes hash stably."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality), optimize=False,
                              progressive=False, subsampling=0)
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
