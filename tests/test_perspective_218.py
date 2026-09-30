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


# --------------------------------------------------------------------------- #
# review fixes (PR #227): fold guard, chance-floor helpers, pose round trip, rating path
# --------------------------------------------------------------------------- #
def _rotvec_of(R):
    """Rotation matrix -> angle-axis (not for angles near 180 deg)."""
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-12:
        return [0.0, 0.0, 0.0]
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return list(k * th)


def _synthetic_row(cam_params, heading=0.0, lat=37.55, lng=-77.45):
    import json
    R_wc = P.level_to_world(heading).T          # a level camera looking along ``heading``
    assert np.allclose(P.rotvec_to_matrix(_rotvec_of(R_wc)), R_wc, atol=1e-12)
    return {"camera_parameters": json.dumps(cam_params),
            "computed_rotation": json.dumps(_rotvec_of(R_wc)),
            "computed_compass_angle": str(heading), "lat": str(lat), "lng": str(lng)}


def _ramp_at(lat0, lng0, bearing_deg, rng_m, uid):
    e = rng_m * math.sin(math.radians(bearing_deg))
    n = rng_m * math.cos(math.radians(bearing_deg))
    lat = lat0 + math.degrees(n / P.EARTH_R)
    lng = lng0 + math.degrees(e / (P.EARTH_R * math.cos(math.radians(lat0))))
    return (uid, lat, lng)


def test_fold_radius_matches_the_turning_point_of_r_d_r():
    cam = P.Camera(2048, 1536, 0.688, k1=-0.05, k2=-0.034)
    rf = P.fold_radius(cam)
    r = np.linspace(0, 3, 300001)
    rd = r * (1 + cam.k1 * r ** 2 + cam.k2 * r ** 4)
    assert rf == pytest.approx(r[np.argmax(rd)], abs=1e-4)
    assert P.fold_radius(P.Camera(100, 100, 0.5, k1=0.1, k2=0.01)) == math.inf


def test_negative_k2_camera_off_axis_ramp_is_not_in_view():
    """B2: with k2 < 0 a ramp 66 deg off-axis, far outside a 72 deg lens, projects back
    into the frame. The in-view rule must reject it; a ramp 20 deg off-axis stays in."""
    import perspective_photos_218 as PP
    row = _synthetic_row([0.688, 0.0, -0.034])
    lat0, lng0 = float(row["lat"]), float(row["lng"])
    ramps = [_ramp_at(lat0, lng0, 66.0, 10.0, "richmond:far"),
             _ramp_at(lat0, lng0, 20.0, 10.0, "richmond:near")]
    cam = PP.camera_of(row, 2048, 1536)
    assert cam.hfov_deg() / 2 == pytest.approx(36.0, abs=0.1)
    # the fold: the far ramp's ground point projects inside the frame
    pc = P.level_to_world(0.0).T @ np.array([10 * math.sin(math.radians(66)),
                                             10 * math.cos(math.radians(66)), -PP.VIEW_H])
    u, v = P.project_cam(cam, pc)
    assert 0 <= float(u) < cam.width and 0 <= float(v) < cam.height
    assert not P.in_distortion_domain(cam, pc)
    g = PP.image_geometry(row, 2048, 1536, ramps)
    by = {r["uid"]: r for r in g["near"]}
    assert by["richmond:far"]["folded"] and not by["richmond:far"]["in_view"]
    assert by["richmond:near"]["in_view"] and not by["richmond:near"]["folded"]


def test_unproject_is_nan_where_the_model_cannot_invert():
    cam = P.Camera(2048, 1536, 0.45, k1=0.0, k2=-0.2)   # corners beyond the fold
    ray = P.unproject_cam(cam, [1023.5, 0.0], [767.5, 0.0])
    assert np.all(np.isfinite(ray[0])) and np.all(np.isnan(ray[1]))
    u, v = P.project_cam(cam, ray[0])                  # where it can, it round-trips
    assert (float(u), float(v)) == pytest.approx((1023.5, 767.5), abs=1e-3)


def test_footprint_falls_back_to_full_canvas_when_the_border_does_not_invert():
    cam = P.Camera(2048, 1536, 0.45, k1=0.0, k2=-0.2)
    assert P.canvas_footprint(cam, np.eye(3)) == (0, P.CANVAS_H, 0, P.CANVAS_W)


def test_pose_round_trip_world_photo_canvas_sfm():
    """N4: a world direction -> photo pixel (SfM pose, distorted camera) -> both the
    scorer's photo->world path and the canvas_sfm embed give back the true bearing and
    depression. R_wc is built independently of cam_from_level: level_to_world(heading),
    then pitch about the level x axis, then roll about the camera z axis."""
    import perspective_photos_218 as PP
    rng = np.random.default_rng(218)
    cam = P.Camera(2048, 1536, 0.62, k1=-0.05, k2=0.0)
    for _ in range(200):
        head = rng.uniform(0, 360)
        pitch, roll = math.radians(rng.uniform(-12, 10)), math.radians(rng.uniform(-8, 6))
        c, s = math.cos(pitch), math.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])          # level -> pitched
        c, s = math.cos(roll), math.sin(roll)
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])          # roll about camera z
        R_wc = Rz @ Rx @ P.level_to_world(head).T
        h_rec, p_rec, _ = P.heading_pitch_roll(R_wc)
        assert float(P.wrap_deg(h_rec - head)) == pytest.approx(0, abs=1e-9)
        assert p_rec == pytest.approx(math.degrees(pitch), abs=1e-9)
        bear = head + rng.uniform(-20, 20)
        dep = rng.uniform(2, 15)
        w = np.array([math.sin(math.radians(bear)) * math.cos(math.radians(dep)),
                      math.cos(math.radians(bear)) * math.cos(math.radians(dep)),
                      -math.sin(math.radians(dep))])
        u, v = P.project_cam(cam, R_wc @ w)
        b2, d2, _ = PP.det_world([{"u": float(u), "v": float(v)}], cam, R_wc)
        assert float(P.wrap_deg(b2[0] - bear)) == pytest.approx(0, abs=1e-6)
        assert float(d2[0]) == pytest.approx(dep, abs=1e-6)
        lvl = P.unproject_cam(cam, u, v) @ P.cam_from_level(R_wc, head)
        x, y = P.ray_to_canvas_norm(lvl)
        assert float(P.wrap_deg(x * 360 - 180 - (bear - head))) == pytest.approx(0, abs=1e-6)
        assert (float(y) - 0.5) * 180 == pytest.approx(dep, abs=1e-6)


def test_chance_floor_helpers():
    import perspective_photos_218 as PP
    d = [{"u": 99.5, "v": 49.5, "score": 0.4}]
    t = PP.transplant(d, 200, 100, 400, 300)[0]
    assert ((t["u"] + 0.5) / 400, (t["v"] + 0.5) / 300) == pytest.approx((0.5, 0.5))
    near = [{"uid": "a", "bearing": 30.0, "dbear": 20.0}]
    assert PP.mirrored(near, 10.0)[0]["bearing"] == pytest.approx(350.0)
    ids = [f"i{k}" for k in range(12)]
    clusters = {i: f"c{k % 3}" for k, i in enumerate(ids)}
    a = PP.swap_donors(ids, clusters, n_null=5)
    assert a == PP.swap_donors(ids, clusters, n_null=5)            # seeded
    assert all(clusters[dr[i]] != clusters[i] for dr in a for i in ids)


def test_bearing_claims_only_near_the_ramp_bearing():
    """What a null draw can score: a detection 90 deg off never claims; 1 deg off does,
    under every bearing-family test."""
    import perspective_photos_218 as PP
    near = [{"uid": "a", "bearing": 0.0, "range": 10.0, "dbear": 0.0}]
    dets = [{"score": 0.9}]
    cl = PP.bearing_claims(dets, np.array([90.0]), np.array([8.0]), near, 0.3)
    assert all(not v for v in cl.values())
    cl = PP.bearing_claims(dets, np.array([1.0]), np.array([8.0]), near, 0.3)
    assert all(v == {"a": 0} for v in cl.values())



def test_usage_row_gpu_share():
    import perspective_photos_218 as PP
    r = PP.usage_row("x:shard1of4", 338, 996.614, "2026-09-30T19:00:50Z", "makelab2",
                     ["NVIDIA A40"], "w" + PP.shard_note(4), gpu_share=0.25,
                     concurrent_with=["y"])
    assert r["gpu_hours"] == 0.0692 and r["gpu_share"] == 0.25
    assert r["concurrent_with"] == ["y"] and r["paid"] is False
    r1 = PP.usage_row("x", 10, 3600.0, "t", "h", [], "w")
    assert r1["gpu_hours"] == 1.0 and "gpu_share" not in r1 and "concurrent_with" not in r1

