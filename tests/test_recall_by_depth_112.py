"""Tests for the depth-axis re-measure (#112).

CPU only. The geometry helpers are checked on closed forms; the committed
``analysis_out/recall_by_depth_112.json`` is checked to re-derive its own tables from its
per-point rows (the replication guard: the payloads are unpublished, the rows are not) and
to reproduce the doc's population. The synthetic-plane test drives the labeler's parser
through ``depth_ranges`` and is skipped when no labeler checkout is available.
"""
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import recall_by_depth_112 as rbd  # noqa: E402

COMMITTED = os.path.join(REPO, "analysis_out", "recall_by_depth_112.json")


# ---------------------------------------------------------------------------
# closed forms

def test_flat_range_is_cotangent_and_none_above_horizon():
    # 45 degrees of depression: range equals the camera height
    assert rbd.flat_range(0.75, 2.5) == pytest.approx(2.5)
    assert rbd.flat_range(0.75, 1.7) == pytest.approx(1.7)
    # the same point at a lower camera is nearer, in proportion
    assert rbd.flat_range(0.6, 1.7) / rbd.flat_range(0.6, 2.5) == pytest.approx(1.7 / 2.5)
    assert rbd.flat_range(0.5, 2.5) is None
    assert rbd.flat_range(0.4, 2.5) is None


def test_apparent_px_uses_the_4096_px_input():
    # a 1.2 m ramp at 10 m subtends 0.12 rad -> 0.12 * 4096 / 2pi px
    assert rbd.apparent_px(10.0) == pytest.approx(0.12 * 4096 / (2 * math.pi))
    # nearer is larger, in inverse proportion
    assert rbd.apparent_px(5.0) == pytest.approx(2 * rbd.apparent_px(10.0))


def test_recall_table_buckets_and_all_row():
    pts = [{"d": 3.0, "hit": True}, {"d": 9.0, "hit": False}, {"d": 9.5, "hit": True},
           {"d": None, "hit": True}]
    rows = rbd.recall_table(pts, "d", [(0, 8), (8, 12)], "m")
    assert [r["bucket"] for r in rows] == ["0-8 m", "8-12 m", "all"]
    assert rows[0]["n"] == 1 and rows[0]["recall"] == 1.0
    assert rows[1]["n"] == 2 and rows[1]["recall"] == 0.5
    assert rows[2]["n"] == 3   # the point without the axis is not counted anywhere


def test_deflation_and_thresholds():
    pts = [{"flat_2p5": 12.0, "depth_range": 10.0, "hit": True},
           {"flat_2p5": 24.0, "depth_range": 20.0, "hit": False},
           {"flat_2p5": 6.0, "depth_range": 6.0, "hit": True}]
    d = rbd.deflation(pts)
    assert d["n"] == 3
    assert d["median_ratio"] == pytest.approx(1.2)
    assert rbd.deflated_thresholds(1.2) == [15.0, 20.8]
    assert rbd.deflation(pts[:2]) is None   # fewer than three pairs: no estimate


def test_resolution_forecast_moves_points_up_the_size_curve():
    # two buckets: small (recall 0) and large (recall 1); doubling size lifts the small ones
    pts = ([{"px": 25.0, "hit": False}] * 4) + ([{"px": 60.0, "hit": True}] * 4)
    f = rbd.resolution_forecast(pts, "px", factors=(2.0,))
    assert f[0]["base"] == 0.5 and f[0]["forecast"] == 1.0 and f[0]["gain"] == 0.5


# ---------------------------------------------------------------------------
# the committed artifact

@pytest.fixture(scope="module")
def committed():
    with open(COMMITTED, encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_tables_rederive_from_the_committed_rows(committed):
    assert rbd.tables(committed) == committed["tables"]


def test_committed_artifact_reproduces_the_docs_population(committed):
    # docs/detection_recall_analysis.md: 637 reviewer-confirmed ramps, recall 0.765 at 0.55
    pr = committed["tables"]["published_reproduction"]
    assert pr["n"] == 637 and pr["hit"] == 487
    assert pr["per_city"] == {"richmond": {"n": 310, "hit": 238}, "bend": {"n": 327, "hit": 249}}


def test_committed_artifact_is_lf_and_rounded():
    with open(COMMITTED, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw
    # every float in the rows carries at most ND decimals
    data = json.loads(raw)
    for row in data["points"][:200] + data["detections"][:200]:
        for k, v in row.items():
            if isinstance(v, float) and k not in ("x", "y", "confidence"):
                assert round(v, rbd.ND) == v, (k, v)


def test_depth_axis_never_backfills_a_non_measured_pano(committed):
    for p in committed["points"]:
        if p["camera_height_status"] != "measured":
            assert p["depth_range"] is None and p["depth_source"] is None
        else:
            assert p["depth_source"] is not None


def test_every_payload_read_is_hash_verified(committed):
    read = [p for p in committed["panos"] if p["sha256"]]
    assert read and all(p["sha256_matches_index"] for p in read)
    assert all(committed["index_sha256"][c] for c in committed["constants"]["depth_splits"])


# ---------------------------------------------------------------------------
# the parser on a synthetic scene (needs the labeler checkout)

LABELER = os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler")


@pytest.mark.skipif(not os.path.exists(os.path.join(LABELER, "depth.py")),
                    reason="no sidewalk-auto-labeler checkout (LABELER_ROOT)")
def test_depth_ranges_on_a_level_plane_equal_flat_ground_at_the_measured_height():
    depthlib = rbd.load_depthlib(LABELER)
    w, h, cam_h = 512, 256, 1.8
    # plane 1: level ground at 1.8 m under every pixel below the horizon; sky above
    ground = depthlib.Plane(0.0, 0.0, -1.0, cam_h)
    indices = bytes([0] * (w * (h // 2)) + [1] * (w * (h // 2)))
    payload = depthlib.DepthPayload(w, h, [depthlib.Plane(0.0, 0.0, 0.0, 0.0), ground], indices)
    gp = depthlib.ground_plane(payload)
    assert gp.camera_height_m == pytest.approx(cam_h)
    for x, y in ((0.1, 0.6), (0.5, 0.75), (0.9, 0.55)):
        rng, ray, src = rbd.depth_ranges(depthlib, payload, gp, x, y)
        assert src == "pixel_plane"
        assert rng == pytest.approx(rbd.flat_range(y, cam_h), rel=1e-9)
        assert ray == pytest.approx(math.hypot(rng, cam_h), rel=1e-9)
    # a sky pixel falls back to the measured height over level ground, and says so
    rng, ray, src = rbd.depth_ranges(depthlib, payload, gp, 0.5, 0.45)
    assert src == "fallback_sky_none" and rng is None


@pytest.mark.skipif(not os.path.exists(os.path.join(LABELER, "depth.py")),
                    reason="no sidewalk-auto-labeler checkout (LABELER_ROOT)")
def test_depth_ranges_falls_back_under_a_wall():
    depthlib = rbd.load_depthlib(LABELER)
    w, h, cam_h = 512, 256, 2.0
    ground = depthlib.Plane(0.0, 0.0, -1.0, cam_h)
    wall = depthlib.Plane(0.0, 1.0, 0.0, 6.0)    # vertical, 6 m away
    idx = [0] * (w * (h // 2)) + [1] * (w * (h // 2))
    # the wall occupies the near-horizon rows on the right half of the image
    for row in range(h // 2, h // 2 + 8):
        for col in range(w // 2, w):
            idx[row * w + col] = 2
    payload = depthlib.DepthPayload(w, h, [depthlib.Plane(0.0, 0.0, 0.0, 0.0), ground, wall], bytes(idx))
    gp = depthlib.ground_plane(payload)
    assert gp.camera_height_m == pytest.approx(cam_h)
    # stored x maps to raw column w - col - 1, so the wall sits under stored x < 0.5
    y = (h // 2 + 2 + 0.5) / h
    rng, _, src = rbd.depth_ranges(depthlib, payload, gp, 0.25, y)
    assert src == "fallback_wall" and rng == pytest.approx(rbd.flat_range(y, cam_h))
    rng, _, src = rbd.depth_ranges(depthlib, payload, gp, 0.75, y)
    assert src == "pixel_plane" and rng == pytest.approx(rbd.flat_range(y, cam_h))
