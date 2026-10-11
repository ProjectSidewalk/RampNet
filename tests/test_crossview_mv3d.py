"""Multi-view 3D arms for #48 (scripts/analysis/crossview_arms/_mv3d.py): the camera model
and the pose-only transfer, pinned against the committed corner manifest. CPU only."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import crossview_align_48 as cv  # noqa: E402
from crossview_arms import _mv3d as M  # noqa: E402

MANIFEST = M.load_manifest()
PAIRS = {p["pair_id"]: p for p in cv.read_frozen_pairs()}
AUTO = cv.read_predictions("proj_height_auto")


def _oth_views():
    for c in MANIFEST["corners"]:
        for v in c["views"]:
            if v["role"] == "oth":
                yield c, v


def test_manifest_covers_every_pair_once():
    got = sorted(v["pair_id"] for _, v in _oth_views())
    assert got == sorted(PAIRS)
    assert len(MANIFEST["corners"]) == len({p["ramp_uid"] for p in PAIRS.values()})


def test_manifest_holds_no_answer():
    text = json.dumps(MANIFEST)
    for k in ("ref_x", "ref_y", "ref_conf", "ref_world_gap_m", "world_conf", "pixel_conf"):
        assert k not in text


def test_flat_transfer_reproduces_the_projection():
    worst = 0.0
    for c, v in _oth_views():
        g = M.raycast_flat(c["views"][0], c["src_x"], c["src_y"], height=M.AIM_HEIGHT_M)
        x, y = M.world_to_pano(v, g, height=M.AIM_HEIGHT_M)
        p = PAIRS[v["pair_id"]]
        worst = max(worst, float(cv.angular_error_deg(x, y, p["proj_x"], p["proj_y"])))
    assert worst < 0.01


def test_view_camera_centre_ray_is_the_view_target():
    v = MANIFEST["corners"][0]["views"][0]
    R, C = M.cam_pose_world(v)
    d = R @ np.array([0.0, 0.0, 1.0])
    want = M.pano_ray_world(v["heading"], v["cx"], v["cy"])
    assert np.allclose(d, want, atol=1e-9)


@pytest.mark.parametrize("n", [0, 7, 42])
def test_poseonly_with_the_prior_pose_is_the_auto_height_projection(n):
    """A reconstruction that agrees with the prior exactly, in an arbitrary similarity
    frame, must give back proj_height_auto."""
    c, v = list(_oth_views())[n]
    src = c["views"][0]
    Rs, Cs = M.cam_pose_world(src)
    Ro, Co = M.cam_pose_world(v)
    rng = np.random.default_rng(n)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    s, t = 0.37, rng.normal(size=3)
    r = M.poseonly_transfer(q @ Rs, s * q @ Cs + t, q @ Ro, s * q @ Co + t, src, v)
    a = AUTO[v["pair_id"]]
    assert float(cv.angular_error_deg(r["x"], r["y"], a["x"], a["y"])) < 0.01
