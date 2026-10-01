"""The bearing audit's pose corrections (#218, docs/bearing_audit_218.md). CPU only, no
network, no images."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

pytest.importorskip("scipy")
from rampnet import perspective as P  # noqa: E402
import bearing_audit_218 as B  # noqa: E402

ROTVEC = [0.80246883372079, -1.8527951484959, 1.7180777662461]   # a real HERO11 pose


@pytest.mark.parametrize("delta", [-35.0, -6.0, 0.0, 20.0, 179.0])
def test_rotate_heading_turns_heading_only(delta):
    h0, p0, r0 = P.heading_pitch_roll(P.world_to_cam_from_rotvec(ROTVEC))
    h1, p1, r1 = P.heading_pitch_roll(
        P.world_to_cam_from_rotvec(B.rotate_heading(ROTVEC, delta)))
    assert float(P.wrap_deg(h1 - h0 - delta)) == pytest.approx(0.0, abs=1e-9)
    assert p1 == pytest.approx(p0, abs=1e-9)
    assert r1 == pytest.approx(r0, abs=1e-9)


def test_rotate_heading_moves_detection_bearings_by_delta():
    cam = P.Camera(2048, 1536, 0.6, -0.05, 0.01)
    dets = [{"u": 300.0, "v": 900.0}, {"u": 1700.0, "v": 1000.0}]
    b0, d0 = B.det_world_tf(dets, cam, P.world_to_cam_from_rotvec(ROTVEC))
    b1, d1 = B.det_world_tf(dets, cam,
                            P.world_to_cam_from_rotvec(B.rotate_heading(ROTVEC, 12.5)))
    assert np.allclose(P.wrap_deg(b1 - b0), 12.5, atol=1e-9)
    assert np.allclose(d1, d0, atol=1e-9)


def test_mirror_reflects_bearing_about_heading_for_a_level_camera():
    cam = P.Camera(2000, 1500, 0.6)
    R = P.level_to_world(40.0).T          # a level camera heading 40 deg
    b, _ = B.det_world_tf([{"u": 1500.0, "v": 900.0}], cam, R)
    bm, _ = B.det_world_tf([{"u": 1500.0, "v": 900.0}], cam, R, mirror=True)
    assert float(P.wrap_deg(b[0] - 40.0)) == pytest.approx(-float(P.wrap_deg(bm[0] - 40.0)),
                                                         abs=0.05)


@pytest.mark.parametrize("travel,compass,want", [
    (-20.0, -22.0, -21.0),      # both device sources agree, both >= 10 off: outvoted
    (-20.0, -5.0, 0.0),         # compass close to SfM: not outvoted
    (-8.0, -9.0, 0.0),          # both under the tolerance
    (-20.0, -35.0, 0.0),        # device sources disagree with each other
    (None, -30.0, 0.0),         # no travel bearing
    (175.0, 176.0, 0.0),        # reversed frame: never turned by 180
])
def test_outvoted(travel, compass, want):
    assert B.outvoted({"d_travel_raw": travel, "d_compass": compass}, 10.0) == \
        pytest.approx(want)


def test_reversed_frame():
    assert B.reversed_frame({"d_travel_raw": -177.0, "d_compass": 179.0})
    assert not B.reversed_frame({"d_travel_raw": -177.0, "d_compass": 2.0})
    assert not B.reversed_frame({"d_travel_raw": None, "d_compass": 179.0})


def test_committed_summary_shape():
    path = os.path.join(B.OUT, "summary.json")
    if not os.path.exists(path):
        pytest.skip("summary.json not generated")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    rows = s["rescore"]["canvas_level"]["0.30"]
    assert rows[0]["name"] == "as scored"
    # the identity re-score reproduces #227's committed headline
    assert rows[0]["hits"] == 58 and rows[0]["n_pairs"] == 292
    assert rows[0]["null"] == pytest.approx(0.0928, abs=1e-4)
