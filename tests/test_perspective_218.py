"""Geometry of the perspective-photo arms (#218). CPU only, synthetic inputs, no network,
no checkpoint."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

from rampnet import perspective as P  # noqa: E402


def cam70(w=1000, h=750, k1=0.0, k2=0.0):
    return P.Camera(w, h, P.pinhole_focal_for_hfov(70.0, w, h), k1, k2)


def test_right_edge_of_70deg_photo_lands_35deg_right_of_centre():
    cam = cam70()
    # the level canvas column of the photo's right-edge centre pixel
    ray = P.unproject_cam(cam, cam.width - 0.5, (cam.height - 1) / 2)
    x, y = P.ray_to_canvas_norm(ray)
    col = x * P.CANVAS_W
    assert col == pytest.approx(P.CANVAS_W / 2 + 35.0 / 360.0 * P.CANVAS_W, abs=0.01)
    assert y * P.CANVAS_H == pytest.approx(P.CANVAS_H / 2, abs=0.01)


def test_canvas_width_of_70deg_photo_is_trained_scale():
    """At 4096 px / 360 deg, a 70 deg photo spans ~796 canvas columns."""
    u, v = P.canvas_sample_maps(cam70(), np.eye(3))
    cols = np.nonzero(np.isfinite(u).any(axis=0))[0]
    assert cols.max() - cols.min() + 1 == pytest.approx(70 / 360 * 4096, abs=2)


@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (-0.1, 0.01), (0.05, 0.0)])
def test_project_unproject_round_trip(k1, k2):
    cam = cam70(k1=k1, k2=k2)
    rng = np.random.default_rng(0)
    u = rng.uniform(0, cam.width - 1, 200)
    v = rng.uniform(0, cam.height - 1, 200)
    u2, v2 = P.project_cam(cam, P.unproject_cam(cam, u, v))
    assert np.allclose(u, u2, atol=1e-3) and np.allclose(v, v2, atol=1e-3)


def test_canvas_detection_maps_back_to_the_sampled_pixel():
    cam = cam70(k1=-0.08, k2=0.01)
    u, v = P.canvas_sample_maps(cam, np.eye(3))
    r, c = 1100, 2100
    assert np.isfinite(u[r, c])
    uu, vv = P.project_cam(cam, P.canvas_norm_to_cam_ray((c + 0.5) / P.CANVAS_W,
                                                         (r + 0.5) / P.CANVAS_H, np.eye(3)))
    assert float(uu) == pytest.approx(float(u[r, c]), abs=1e-2)
    assert float(vv) == pytest.approx(float(v[r, c]), abs=1e-2)


def test_windowed_maps_equal_full_canvas():
    R = P.rotvec_to_matrix([0.1, -0.2, 0.05])
    for cam, M in [(cam70(), np.eye(3)),
                   (P.Camera(2048, 1792, 0.42, -0.10, 0.007), P.cam_from_level(R))]:
        a = P.canvas_sample_maps(cam, M, window=True)
        b = P.canvas_sample_maps(cam, M, window=False)
        for x, y in zip(a, b):
            assert np.array_equal(np.isfinite(x), np.isfinite(y))
            assert np.allclose(x[np.isfinite(x)], y[np.isfinite(y)])


def test_heading_convention_and_level_frame():
    # a camera looking due east, level: forward = +E, right = -N, down = -U
    Rlw = P.level_to_world(90.0)
    assert np.allclose(Rlw @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    assert np.allclose(Rlw @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    R_wc = Rlw.T      # a level camera: world -> cam
    h, p, r = P.heading_pitch_roll(R_wc)
    assert (h, p, r) == pytest.approx((90.0, 0.0, 0.0), abs=1e-9)
    assert np.allclose(P.cam_from_level(R_wc), np.eye(3), atol=1e-12)


def test_pitched_camera_sfm_canvas_moves_the_photo_down():
    """A camera pitched 10 deg down: in the SfM canvas the photo centre sits 10 deg
    below the horizon row."""
    pitch = math.radians(-10)
    Rlw = P.level_to_world(0.0)
    # rotate the level frame about its x axis to pitch the camera down
    c, s = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])   # level -> pitched cam
    R_wc = Rx @ Rlw.T
    assert P.heading_pitch_roll(R_wc)[1] == pytest.approx(-10.0, abs=1e-9)
    M = P.cam_from_level(R_wc)
    cam = cam70()
    ray_c = P.unproject_cam(cam, (cam.width - 1) / 2, (cam.height - 1) / 2)
    x, y = P.ray_to_canvas_norm(ray_c @ M)
    assert (y - 0.5) * 180 == pytest.approx(10.0, abs=1e-6)


def test_bearing_hit_and_raycast():
    # ramp 10 m ahead; a detection 2 deg off at 8.53 deg down (h = 1.5 m)
    dep = math.degrees(math.atan(1.5 / 10))
    assert P.bearing_hit(2.0, dep, 0.0, 10.0)
    assert not P.bearing_hit(40.0, dep, 0.0, 10.0)
    assert not P.bearing_hit(0.0, -3.0, 0.0, 10.0)        # above the horizon
    assert not P.bearing_hit(0.0, 45.0, 0.0, 10.0)        # implies a 10 m camera
    w = np.array([[0.0, math.cos(math.radians(dep)), -math.sin(math.radians(dep))]])
    e, n = P.raycast_ground(w, 1.5)
    assert float(e[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(n[0]) == pytest.approx(10.0, abs=1e-9)


def test_enu_offset_small_distances():
    e, n = P.enu_offset(37.5, -77.4, 37.5 + 1e-4, -77.4)
    assert float(n) == pytest.approx(11.13, abs=0.01) and float(e) == pytest.approx(0, abs=1e-9)


def test_seoul_hfov_from_f35():
    import seoul_photos_218 as S
    assert S.hfov_from_f35(26) == pytest.approx(69.4, abs=0.1)
    cam, hfov, _ = S.seoul_camera(4032, 3024, f35=None)
    assert hfov == 70.0 and cam.hfov_deg() == pytest.approx(70.0)
    assert S.stem_key("IMG_8956.HEIC") == S.stem_key("img_8956.jpg")
    assert S.presence_label("flush") == 0 and S.presence_label("flush", True) == 1
    assert S.presence_label("cant_tell") is None


def test_stretch_mapping_is_linear():
    import perspective_photos_218 as PP
    cam = cam70(2048, 1536)
    # x = 0.5 is stretched pixel 2048 (the benchmark's x = px / W convention); its
    # centre is at photo x (2048 + 0.5) * 2048 / 4096 - 0.5
    u, v = PP.stretch_det_to_photo(0.5, 0.5, cam)
    assert (u, v) == pytest.approx((1023.75, (1024.5) * 1536 / 2048 - 0.5), abs=1e-9)
    u0, _ = PP.stretch_det_to_photo(0.0, 0.0, cam)
    assert u0 == pytest.approx(-0.25)
