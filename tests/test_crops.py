"""Geometry and bookkeeping guards for the crop cutter (rampnet/crops.py, scripts/crop_cutter.py).

CPU-only, no network, no pano store: every test runs on a synthetic equirect pano with a known
pattern, so a reader without the makelab2 store can still verify the geometry the validation
run depends on. Pinned:
  1. The label lands at the exact centre of a centred view, including across the seam and
     near the poles (marker planted in a synthetic pano, read back out of the crop).
  2. Round trip: output pixel -> sphere -> output pixel is the identity, with and without tilt.
  3. The renderer agrees with the independent point math in
     scripts/model_comparison/equirect_tiling.py.
  4. FOV scaling: an object at a known angular offset lands at f * tan(offset) from centre.
  5. Viewport mode reproduces Project Sidewalk's own click->pano math: a label synthesised
     with calculatePovIfCentered + calculatePanoXYFromPov lands at (canvas_x, canvas_y) * 2.
  6. The committed validation sample's real rows replay the same way.
  7. CLI: manifest resumability, idempotency, sha256 of the written bytes, missing panos as a
     status, corrupt panos as an error with exit 1.
"""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from rampnet import crops  # noqa: E402

SAMPLE = os.path.join(REPO_ROOT, "docs", "data", "crop_cutter", "validation_sample.csv")


# ----------------------------------------------------------------------------- helpers

def marker_pano(W, H, points, radius=3, color=(255, 0, 0)):
    """Mid-grey pano with a coloured square (wrapping in x) at each (px, py)."""
    img = np.full((H, W, 3), 128, np.uint8)
    for px, py in points:
        ys = np.arange(int(py) - radius, int(py) + radius + 1)
        ys = ys[(ys >= 0) & (ys < H)]
        xs = np.arange(int(px) - radius, int(px) + radius + 1) % W
        img[np.ix_(ys, xs)] = color
    return img


def red_centroid(arr):
    m = (arr[..., 0] > 200) & (arr[..., 1] < 60) & (arr[..., 2] < 60)
    assert m.any(), "marker not found in crop"
    ys, xs = np.nonzero(m)
    return xs.mean() + 0.5, ys.mean() + 0.5


def ps_pov_if_centered(canvas_x, canvas_y, heading, pitch, zoom, cw=720.0, ch=480.0):
    """Project Sidewalk's calculatePovIfCentered, transcribed from sidewalk-panorama-tools
    reports/scripts/pov_replay.py (x east, y north, z up) -- deliberately NOT via rampnet.crops."""
    fov = math.radians(crops.get_3d_fov(zoom))
    h0, p0 = math.radians(heading), math.radians(pitch)
    f = 0.5 * cw / math.tan(0.5 * fov)
    du, dv = canvas_x - cw / 2, ch / 2 - canvas_y
    sg = 1.0 if math.cos(p0) >= 0 else -1.0
    x = f * math.cos(p0) * math.sin(h0) + du * sg * math.cos(h0) - dv * math.sin(p0) * math.sin(h0)
    y = f * math.cos(p0) * math.cos(h0) - du * sg * math.sin(h0) - dv * math.sin(p0) * math.cos(h0)
    z = f * math.sin(p0) + dv * math.cos(p0)
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.atan2(x, y)), math.degrees(math.asin(z / r))


def ps_pano_xy(pov_heading, pov_pitch, camera_heading, W, H):
    """calculatePanoXYFromPov, continuous (no rounding)."""
    hw = (pov_heading + 360) % 360
    zero = ((camera_heading + 180) % 360 + 360) % 360
    return (W + W * (hw - zero) / 360) % W, H / 2 - (H / 2) * (pov_pitch / 90)


# ----------------------------------------------------------------------------- constants

def test_get_3d_fov_matches_the_viewer_table():
    assert crops.get_3d_fov(1) == pytest.approx(89.75)
    assert crops.get_3d_fov(2) == pytest.approx(53.0)
    assert crops.get_3d_fov(3) == pytest.approx(27.68, abs=0.01)
    assert np.allclose(crops.get_3d_fov([1, 2]), [89.75, 53.0])


def test_view_size_and_vertical_fov():
    v = crops.centered_view(100, 50, 400, 200, 60, width=300, aspect=1.5)
    assert (v.width, v.height) == (300, 200)
    assert math.tan(math.radians(v.fov_v_deg) / 2) == pytest.approx(math.tan(math.radians(30)) / 1.5)
    with pytest.raises(ValueError):
        crops.centered_view(100, 50, 400, 200, 180)


# ----------------------------------------------------------------------------- label at centre

@pytest.mark.parametrize("px,py", [
    (1000, 600),     # ordinary, below the horizon
    (0, 620),        # on the seam
    (2047, 640),     # last column
    (2048, 640),     # pano_x == pano_width: the same place as column 0 (real rows do this)
    (5, 700),        # a few columns right of the seam: the view must wrap to the right edge
    (1500, 950),     # 76 deg below the horizon
])
def test_marker_lands_at_centre(px, py):
    W, H = 2048, 1024
    pano = marker_pano(W, H, [(px % W, py)])
    view = crops.centered_view(px, py, W, H, fov_h_deg=10, width=120, aspect=1.5)
    crop = crops.render_view(pano, view)
    cx, cy = red_centroid(crop)
    # the marker is 7 source px; its centroid is at the source pixel centre (px + 0.5), which is
    # half a source pixel off the stored point -- ~1.5 output px at this magnification
    assert abs(cx - view.width / 2) < 2.5 and abs(cy - view.height / 2) < 2.5


def test_seam_crop_has_no_synthetic_black_and_uses_both_edges():
    W, H = 1024, 512
    x = np.arange(W)
    pano = np.zeros((H, W, 3), np.uint8)
    pano[:, :, 0] = np.where(x < W // 2, 200, 40)[None, :]      # left half bright red channel
    pano[:, :, 1] = 100
    pano[:, :, 2] = np.where(x < W // 2, 40, 200)[None, :]      # right half bright blue
    crop = crops.render_view(pano, crops.centered_view(0, 300, W, H, 40, width=200))
    assert crop.min() > 0
    left, right = crop[:, :90], crop[:, 110:]
    assert left[..., 2].mean() > 150 and right[..., 0].mean() > 150  # right edge left of centre, left edge right


def test_resolution_independence():
    """Same label, pano stored at 2x resolution: identical framing."""
    yy, xx = np.mgrid[0:256, 0:512]
    small = np.stack([128 + 100 * np.sin(xx / 9.0), 128 + 100 * np.cos(yy / 7.0),
                      128 + 100 * np.sin((xx + yy) / 13.0)], -1).astype(np.uint8)
    yy, xx = np.mgrid[0:512, 0:1024] / 2.0
    big = np.stack([128 + 100 * np.sin(xx / 9.0), 128 + 100 * np.cos(yy / 7.0),
                    128 + 100 * np.sin((xx + yy) / 13.0)], -1).astype(np.uint8)
    v = crops.centered_view(300, 170, 512, 256, 30, width=90)
    a = crops.render_view(small, v).astype(int)
    b = crops.render_view(big, v).astype(int)
    assert np.abs(a - b).mean() < 3  # the same smooth texture sampled at two resolutions


# ----------------------------------------------------------------------------- round trips

@pytest.mark.parametrize("tilt", [None, "pp", "mm"])
def test_pixel_sphere_pixel_round_trip(tilt):
    view = crops.View(yaw_deg=-170.0, pitch_deg=-25.0, fov_h_deg=70.0, width=64, height=40)
    T = crops.tilt_matrix(2.0, -1.5, tilt)
    lon, lat = crops.rays_to_lonlat(crops.view_rays(view), T)
    x, y = crops.project_lonlat(lon, lat, view, T)
    gx, gy = np.meshgrid(np.arange(64) + 0.5, np.arange(40) + 0.5)
    assert np.allclose(x, gx, atol=1e-7) and np.allclose(y, gy, atol=1e-7)


def test_label_projects_to_exact_centre():
    v = crops.centered_view(13740, 4754, 16384, 8192, 60)
    lon, lat = crops.pano_px_to_lonlat(13740, 4754, 16384, 8192)
    assert crops.project_lonlat(lon, lat, v) == pytest.approx((720.0, 480.0), abs=1e-6)


def test_behind_camera_is_nan():
    v = crops.View(0.0, 0.0, 60.0, 30, 20)
    x, y = crops.project_lonlat(180.0, 0.0, v)
    assert math.isnan(x) and math.isnan(y)


def test_agrees_with_equirect_tiling_point_math():
    import equirect_tiling as et
    view = crops.View(yaw_deg=40.0, pitch_deg=-30.0, fov_h_deg=80.0, width=50, height=30)
    fv = view.fov_v_deg
    etv = et.View(40.0, -30.0, 80.0, fv, 50, 30)
    for u, v in [(0.1, 0.2), (0.5, 0.5), (0.93, 0.71)]:
        X, Y = et.perspective_point_to_equirect(u, v, etv)
        x, y = crops.project_lonlat((X - 0.5) * 360, (0.5 - Y) * 180, view)
        assert (x / 50, y / 30) == pytest.approx((u, v), abs=1e-9)


def test_tilt_moves_the_horizon_by_the_sinusoid():
    T = crops.tilt_matrix(3.0, 0.0, "pp")
    lon, lat = crops.rays_to_lonlat(crops.lonlat_to_ray(0.0, 0.0)[None, :], T)
    assert abs(abs(lat[0]) - 3.0) < 1e-9          # straight ahead: the full pitch
    lon, lat = crops.rays_to_lonlat(crops.lonlat_to_ray(90.0, 0.0)[None, :], T)
    assert abs(lat[0]) < 1e-9                     # 90 deg round: pitch does not move it
    T = crops.tilt_matrix(0.0, 2.0, "pp")
    lon, lat = crops.rays_to_lonlat(crops.lonlat_to_ray(90.0, 0.0)[None, :], T)
    assert abs(abs(lat[0]) - 2.0) < 1e-9          # roll bites at 90 deg
    assert crops.tilt_matrix(3.0, 1.0, "none") is None


# ----------------------------------------------------------------------------- FOV scaling

@pytest.mark.parametrize("fov", [20.0, 45.0, 90.0])
def test_fov_scaling(fov):
    W, H = 4096, 2048
    lpx, lpy = 1000.0, 1300.0
    lon, lat = crops.pano_px_to_lonlat(lpx, lpy, W, H)
    off = fov / 5.0   # a marker off to the right by a fifth of the FOV, same elevation
    mx, my = crops.lonlat_to_pano_px(lon + off, lat, W, H)
    view = crops.centered_view(lpx, lpy, W, H, fov, width=300)
    x, y = crops.project_lonlat(lon + off, lat, view)
    # exact pinhole prediction for a point at the label's latitude, off in longitude
    d = crops.lonlat_to_ray(lon + off, lat)
    fwd, right, up = crops._basis(view.yaw_deg, view.pitch_deg)
    assert x - 150 == pytest.approx(view.focal_px * (d @ right) / (d @ fwd), abs=1e-6)
    crop = crops.render_view(marker_pano(W, H, [(mx, my)], radius=4), view)
    cx, cy = red_centroid(crop)
    assert abs(cx - x) < 1.5 + 4 * view.focal_px * math.pi / 180 * 360 / W  # within the marker size
    # and the same angular offset lands proportionally to the focal length
    assert view.focal_px == pytest.approx(150 / math.tan(math.radians(fov / 2)))


def test_reduction_factor():
    assert crops.reduction_factor(16384, crops.View(0, 0, 90.0, 1440, 960)) == 3
    assert crops.reduction_factor(16384, crops.View(0, 0, 20.0, 1440, 960)) == 1
    assert crops.reduction_factor(16384, crops.View(0, 0, 90.0, 224, 149)) == 23


# ----------------------------------------------------------------------------- viewport mode

@pytest.mark.parametrize("cx,cy,heading,pitch,zoom,cam", [
    (360, 240, 10.0, -10.0, 1, 0.0),
    (152, 153, 299.3, -17.5, 3, 180.37),
    (700, 470, 181.0, -30.0, 2, 355.0),
    (5, 10, 359.0, 5.0, 1, 90.0),
])
def test_viewport_reproduces_project_sidewalk_click_math(cx, cy, heading, pitch, zoom, cam):
    W, H = 16384, 8192
    ph, pp = ps_pov_if_centered(cx, cy, heading, pitch, zoom)
    px, py = ps_pano_xy(ph, pp, cam, W, H)
    view = crops.viewport_view(heading, pitch, zoom, cam)
    lon, lat = crops.pano_px_to_lonlat(px, py, W, H)
    x, y = crops.project_lonlat(lon, lat, view)
    assert (x, y) == pytest.approx(crops.label_pixel_in_viewport(cx, cy, view), abs=1e-6)


def _sample_rows():
    if not os.path.isfile(SAMPLE):
        pytest.skip("validation sample not committed")
    with open(SAMPLE, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_committed_sample_replays_into_its_viewport():
    """Real rows: the stored (integer) pano_x/pano_y land at canvas_x/canvas_y in the viewport
    crop, to within the stored rounding. Needs no imagery -- this is the geometry half of the
    validation that anyone can re-check from a clean clone."""
    rows = _sample_rows()
    errs = []
    for r in rows:
        view = crops.viewport_view(float(r["heading"]), float(r["pitch"]), float(r["zoom"]),
                                   float(r["camera_heading"]))
        lon, lat = crops.pano_px_to_lonlat(float(r["pano_x"]), float(r["pano_y"]),
                                           float(r["pano_width"]), float(r["pano_height"]))
        x, y = crops.project_lonlat(lon, lat, view)
        ex, ey = crops.label_pixel_in_viewport(float(r["canvas_x"]), float(r["canvas_y"]), view)
        errs.append(math.hypot(x - ex, y - ey))
    errs = np.array(errs)
    # pre-2021 rows carry parseInt-truncated POVs and camera_heading drift (pov_replay.py);
    # the bulk still replays to a few output pixels
    assert np.median(errs) < 3.0, np.median(errs)


# ----------------------------------------------------------------------------- equirect window

def test_equirect_window_wraps_and_reports_label():
    W, H = 1024, 512
    pano = marker_pano(W, H, [(2, 300)])
    crop, shift, (lx, ly) = crops.equirect_window(pano, 2, 300, W, H, 20, width=120)
    assert shift == 0 and crop.min() >= 0
    cx, cy = red_centroid(crop)
    assert abs(cx - lx) < 3 and abs(cy - ly) < 3


def test_equirect_window_shifts_at_the_pole():
    W, H = 1024, 512
    pano = marker_pano(W, H, [(500, 505)], radius=2)
    crop, shift, (lx, ly) = crops.equirect_window(pano, 500, 505, W, H, 40, width=120)
    assert shift < 0 and ly > crop.shape[0] / 2
    cx, cy = red_centroid(crop)
    assert abs(cy - ly) < 4


# ----------------------------------------------------------------------------- bytes

def test_jpeg_encoding_is_deterministic():
    arr = np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    a, b = crops.encode_jpeg(arr), crops.encode_jpeg(arr)
    assert a == b and crops.sha256_hex(a) == hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- CLI

def _write_store(root, city, pano_id, W=512, H=256, marker=(100, 150)):
    d = os.path.join(root, city, pano_id[:2])
    os.makedirs(d, exist_ok=True)
    Image.fromarray(marker_pano(W, H, [marker], radius=4)).save(os.path.join(d, pano_id + ".jpg"), quality=95)


def _labels_csv(path, rows):
    cols = ["city", "label_id", "pano_id", "pano_x", "pano_y", "pano_width", "pano_height", "heading",
            "pitch", "zoom", "camera_heading", "camera_pitch", "camera_roll", "canvas_x", "canvas_y"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _manifest(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_cli_resumable_idempotent_and_hashed(tmp_path, capsys):
    import crop_cutter
    store, out = str(tmp_path / "store"), str(tmp_path / "out")
    _write_store(store, "seattle-wa", "AAAApano")
    _write_store(store, "walla-walla-wa", "BBBBpano")            # aliased city dir
    rows = [
        # pano 16384 wide in metadata, 512 on disk: resolution-independent
        {"city": "seattle-wa", "label_id": 1, "pano_id": "AAAApano", "pano_x": 3200, "pano_y": 4800,
         "pano_width": 16384, "pano_height": 8192, "heading": 0, "pitch": -10, "zoom": 1,
         "camera_heading": 0, "canvas_x": 360, "canvas_y": 240},
        {"city": "walla-walla", "label_id": 1, "pano_id": "BBBBpano", "pano_x": 100, "pano_y": 150,
         "pano_width": 512, "pano_height": 256, "heading": 10, "pitch": -5, "zoom": 2,
         "camera_heading": 5, "canvas_x": 100, "canvas_y": 100},
        {"city": "seattle-wa", "label_id": 2, "pano_id": "ZZZZmissing", "pano_x": 1, "pano_y": 2,
         "pano_width": 512, "pano_height": 256},
        {"city": "seattle-wa", "label_id": 3, "pano_id": "AAAApano", "pano_x": 1, "pano_y": 900,
         "pano_width": 512, "pano_height": 256},                  # pano_y out of frame
    ]
    labels = str(tmp_path / "labels.csv")
    _labels_csv(labels, rows)
    argv = ["--labels", labels, "--store", store, "--out", out, "--fov", "30", "--fov", "viewport",
            "--size", "96"]
    assert crop_cutter.main(argv) == 0
    m1 = _manifest(os.path.join(out, "manifest.jsonl"))
    st = {(r["city"], r["label_id"], r["tag"]): r["status"] for r in m1}
    assert st[("seattle-wa", 1, "fov30")] == "ok"
    assert st[("walla-walla", 1, "fov30")] == "ok"
    assert st[("seattle-wa", 2, "fov30")] == "missing_pano"
    assert st[("seattle-wa", 3, "fov30")] == "out_of_frame"
    assert st[("seattle-wa", 3, "viewport")] == "no_geometry"   # no POV columns
    for r in m1:
        if r["status"] == "ok":
            with open(os.path.join(out, r["name"]), "rb") as fh:
                assert hashlib.sha256(fh.read()).hexdigest() == r["sha256"]
            assert (r["width"], r["height"]) == (96, 64)
    # the seattle label at (3200, 4800) of 16384x8192 is the marker at (100, 150) of 512x256
    arr = np.asarray(Image.open(os.path.join(out, "seattle-wa__1__fov30.jpg")))
    cx, cy = red_centroid(arr)
    assert abs(cx - 48) < 4 and abs(cy - 32) < 4

    # second run: nothing new, nothing re-cut
    assert crop_cutter.main(argv) == 0
    assert _manifest(os.path.join(out, "manifest.jsonl")) == m1

    # delete one crop: only it is re-cut, with identical bytes
    os.remove(os.path.join(out, "walla-walla__1__fov30.jpg"))
    assert crop_cutter.main(argv) == 0
    m3 = _manifest(os.path.join(out, "manifest.jsonl"))
    assert len(m3) == len(m1) + 1
    old = [r for r in m1 if r["name"] == "walla-walla__1__fov30.jpg"][0]
    assert m3[-1]["name"] == old["name"] and m3[-1]["sha256"] == old["sha256"]

    # the missing pano appears: it is picked up
    _write_store(store, "seattle-wa", "ZZZZmissing")
    assert crop_cutter.main(argv) == 0
    last = {r["name"]: r for r in _manifest(os.path.join(out, "manifest.jsonl"))}
    assert last["seattle-wa__2__fov30.jpg"]["status"] == "ok"


def test_cli_corrupt_pano_is_an_error_and_exit_1(tmp_path):
    import crop_cutter
    store, out = str(tmp_path / "store"), str(tmp_path / "out")
    os.makedirs(os.path.join(store, "x", "CC"))
    with open(os.path.join(store, "x", "CC", "CCCC.jpg"), "wb") as fh:
        fh.write(b"not a jpeg")
    labels = str(tmp_path / "l.csv")
    _labels_csv(labels, [{"city": "x", "label_id": 7, "pano_id": "CCCC", "pano_x": 1, "pano_y": 2,
                          "pano_width": 512, "pano_height": 256}])
    assert crop_cutter.main(["--labels", labels, "--store", store, "--out", out, "--fov", "40",
                             "--size", "32"]) == 1
    assert _manifest(os.path.join(out, "manifest.jsonl"))[0]["status"] == "error"


def test_cli_label_list_forms(tmp_path):
    import crop_cutter
    p = tmp_path / "ids.txt"
    p.write_text("# comment\nseattle-wa:9\ncdmx:10\n", encoding="utf-8")
    assert [(c, l) for c, l, _ in crop_cutter.read_label_list(str(p))] == [("seattle-wa", 9), ("cdmx", 10)]
    assert [(c, l) for c, l, _ in crop_cutter.read_label_list("a:1,b-c:2")] == [("a", 1), ("b-c", 2)]
    assert crop_cutter.fov_tag("22.5", "gnomonic", "none") == "fov22p5"
    assert crop_cutter.fov_tag("60", "equirect", "pp") == "fov60_eq_tiltpp"
    assert crop_cutter.store_city_dir("walla-walla", {}) == "walla-walla-wa"
    assert crop_cutter.store_city_dir("seattle-wa", {"seattle-wa": "sea"}) == "sea"
