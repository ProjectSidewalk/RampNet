"""Tests for the #48 cross-view alignment harness (scripts/analysis/crossview_align_48.py)
and its arms (scripts/analysis/crossview_arms/).

CPU only, no network, no labeler checkout, no imagery: the view geometry and the arm
plumbing are checked on toy inputs, and the committed artifacts under
``analysis_out/crossview_align_48/`` are checked for LF-pinned bytes, the frozen pair list's
hash, and re-derived where they can be (the sample from the eligible list, and every arm's
headline median and fallback rate from its committed predictions).
"""
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import crossview_align_48 as cv  # noqa: E402
from crossview_arms._registry import ANSWER_KEYS, Arm  # noqa: E402

OUT = cv.OUT


# --------------------------------------------------------------------------- #
# view geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.25, 0.6), (0.99, 0.62), (0.01, 0.45)])
def test_view_centre_maps_to_the_view_target(centre):
    x, y = cv.view_to_pano(cv.VIEW_W / 2.0, cv.VIEW_H / 2.0, *centre)
    assert float(x) == pytest.approx(centre[0], abs=1e-9)
    assert float(y) == pytest.approx(centre[1], abs=1e-9)


@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.3, 0.63), (0.995, 0.58)])
def test_view_and_pano_round_trip(centre):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, cv.VIEW_W, 50)
    v = rng.uniform(0, cv.VIEW_H, 50)
    x, y = cv.view_to_pano(u, v, *centre)
    u2, v2, front = cv.pano_to_view(x, y, *centre)
    assert front.all()
    assert np.allclose(u2, u, atol=1e-6) and np.allclose(v2, v, atol=1e-6)


def test_view_axes_right_is_clockwise_and_down_is_down():
    x_r, y_r = cv.view_to_pano(cv.VIEW_W / 2.0 + 100, cv.VIEW_H / 2.0, 0.5, 0.5)
    x_d, y_d = cv.view_to_pano(cv.VIEW_W / 2.0, cv.VIEW_H / 2.0 + 100, 0.5, 0.5)
    assert float(x_r) > 0.5 and float(y_r) == pytest.approx(0.5, abs=1e-12)
    assert float(y_d) > 0.5 and float(x_d) == pytest.approx(0.5, abs=1e-12)


def test_horizontal_fov_edge_is_half_the_fov():
    x, _ = cv.view_to_pano(cv.VIEW_W, cv.VIEW_H / 2.0, 0.5, 0.5)
    assert (float(x) - 0.5) * 360.0 == pytest.approx(cv.HFOV_DEG / 2.0, abs=1e-9)


def test_angular_and_pixel_errors_are_seam_safe():
    assert float(cv.angular_error_deg(0.5, 0.5, 0.5 + 1 / 360.0, 0.5)) == pytest.approx(1.0)
    assert float(cv.angular_error_deg(0.999, 0.5, 0.001, 0.5)) == pytest.approx(0.72, abs=1e-6)
    assert float(cv.angular_error_deg(0.1, 0.99, 0.6, 0.99)) < 3.7   # near the nadir
    assert float(cv.equirect_px_error(0.999, 0.5, 0.001, 0.5)) == pytest.approx(0.002 * 4096)
    assert float(cv.equirect_px_error(0.5, 0.5, 0.5, 0.51)) == pytest.approx(0.01 * 2048)


def test_ground_mask_keeps_below_the_margin_only():
    u = np.array([cv.VIEW_W / 2.0] * 3)
    f = cv.focal_px()
    v = cv.VIEW_H / 2.0 + f * np.tan(np.radians([-1.0, 2.0, 10.0]))
    assert list(cv.ground_mask(u, v, 0.5, 0.5, margin=0.5)) == [False, True, True]
    assert list(cv.ground_mask(u, v, 0.5, 0.5, margin=5.0)) == [False, False, True]


def test_map_point_identity_and_degenerate():
    assert cv.map_point(np.eye(3), 10.0, 20.0) == (10.0, 20.0)
    assert cv.map_point(None, 1.0, 1.0) is None
    assert cv.map_point(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.0]]), 1.0, 1.0) is None


def test_render_view_puts_a_marked_point_where_the_geometry_says():
    cv2 = pytest.importorskip("cv2")
    W = int(round(360.0 / cv.HFOV_DEG * cv.VIEW_W))
    equi = np.zeros((W // 2, W, 3), np.uint8)
    cv2.circle(equi, (int(0.55 * W), int(0.6 * W / 2)), 4, (255, 255, 255), -1)
    view = cv.render_view(equi, 0.5, 0.55)
    vv, uu = np.unravel_index(np.argmax(view[..., 0]), view.shape[:2])
    u, v, _ = cv.pano_to_view(0.55, 0.6, 0.5, 0.55)
    assert abs(uu - float(u)) < 4 and abs(vv - float(v)) < 4


# --------------------------------------------------------------------------- #
# arms: registry, answer hiding, fallback
# --------------------------------------------------------------------------- #
def test_registry_has_the_committed_arms():
    arms = cv.load_arms()
    for name in ("lg", "lg_local", "lg_band0.5", "sift", "ncc", "proj_flat_check",
                 "proj_height_perpano", "proj_height_auto", "proj_mly_gravity",
                 "proj_mly_road", "proj_mly_rawgps", "proj_gsv_depth"):
        assert name in arms, name
        assert arms[name].description


def test_run_arm_hides_the_answer_and_normalizes_fallbacks():
    pairs = [{"pair_id": "a", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7,
              "ref_conf": 0.9, "ref_world_gap_m": 1.0},
             {"pair_id": "b", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7,
              "ref_conf": 0.9, "ref_world_gap_m": 1.0}]
    seen = []

    def fn(pair, ctx):
        seen.append(set(pair))
        return None if pair["pair_id"] == "a" else {"x": 0.4, "y": None, "why": "half"}

    rows, missing = cv.run_arm(Arm("t", fn), pairs, SimpleNamespace())
    assert missing == 0
    assert all(not (s & set(ANSWER_KEYS)) for s in seen)
    assert rows[0] == {"pair_id": "a", "x": None, "y": None}
    assert rows[1] == {"pair_id": "b", "x": None, "y": None, "why": "half"}


def test_arm_errors_fall_back_to_the_projection():
    p = [{"pair_id": "a", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.5 + 2 / 360.0, "ref_y": 0.6}]
    proj = cv.arm_errors(p, None)
    fb = cv.arm_errors(p, {"a": {"x": None, "y": None}})
    hit = cv.arm_errors(p, {"a": {"x": 0.5 + 2 / 360.0, "y": 0.6}})
    assert proj[0][3] is False and fb[0][3] is True
    assert fb[0][0] == pytest.approx(proj[0][0])
    assert hit[0][0] == pytest.approx(0.0, abs=1e-6) and hit[0][3] is False


def test_homography_recovers_a_known_plane_mapping():
    pytest.importorskip("cv2")
    from crossview_arms import matching
    Hm = np.array([[1.1, 0.05, 20.0], [0.02, 0.9, -15.0], [1e-4, 2e-4, 1.0]])
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1000, (60, 2))
    b = np.array([cv.map_point(Hm, *p) for p in a])
    b[:10] += rng.uniform(50, 100, (10, 2))       # outliers
    Hf, n_in, _ = matching.fit_homography(a, b)
    assert n_in >= 50
    got, want = cv.map_point(Hf, 512, 384), cv.map_point(Hm, 512, 384)
    assert math.hypot(got[0] - want[0], got[1] - want[1]) < 0.5


def test_map_centre_falls_back_below_min_inliers():
    pytest.importorskip("cv2")
    from crossview_arms import matching
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1000, (10, 2))
    assert matching.map_centre(a, a + 5.0, (0.5, 0.6), matching.MIN_INLIERS) is None
    a = np.column_stack([rng.uniform(0, 1024, 40), rng.uniform(0, 768, 40)])
    out = matching.map_centre(a, a, (0.5, 0.6), matching.MIN_INLIERS)
    assert out["inliers"] == 40
    assert out["x"] == pytest.approx(0.5, abs=1e-6) and out["y"] == pytest.approx(0.6, abs=1e-6)


# --------------------------------------------------------------------------- #
# committed artifacts
# --------------------------------------------------------------------------- #
def _committed():
    names = ["eligible_pairs.csv", "pairs.csv", "pairs_meta.json", "reference_noise.json",
             "results.json"]
    for arm in cv.available_predictions():
        names += [f"predictions/{arm}.jsonl", f"predictions/{arm}.meta.json"]
    return names


@pytest.mark.parametrize("name", _committed())
def test_committed_artifacts_are_lf(name):
    with open(os.path.join(OUT, name), "rb") as f:
        assert b"\r\n" not in f.read()


def test_pair_list_is_the_frozen_one():
    assert cv.pairs_sha256() == cv.PAIRS_SHA256
    assert len(cv.read_frozen_pairs()) == cv.PAIRS_PER_CITY * len(cv.CITIES)


def test_committed_sample_rederives_from_the_eligible_list():
    eligible = cv.read_rows(cv.ELIGIBLE_CSV)
    committed = cv.read_rows(cv.PAIRS_CSV)
    sampled = cv.sample_pairs(eligible)
    key = lambda r: (r["pair_id"], r["ramp_uid"], r["src_pano"], r["oth_pano"])  # noqa: E731
    assert [key(r) for r in sampled] == [key(r) for r in committed]
    per_ramp = {}
    for r in committed:
        per_ramp[r["ramp_uid"]] = per_ramp.get(r["ramp_uid"], 0) + 1
    assert max(per_ramp.values()) <= cv.MAX_PAIRS_PER_RAMP
    assert all(r["ref_world_gap_m"] < cv.WORLD_HIT_M and r["oth_range_m"] <= cv.R_OTHER_M
               for r in committed)


def test_every_committed_prediction_covers_the_frozen_pairs():
    ids = {p["pair_id"] for p in cv.read_frozen_pairs()}
    for arm in cv.available_predictions():
        preds = cv.read_predictions(arm)      # refuses a different pairs_sha256
        assert set(preds) == ids, arm


def test_committed_results_rederive_from_committed_predictions():
    pairs = cv.read_frozen_pairs()
    with open(cv.RESULTS_JSON, encoding="utf-8") as f:
        res = json.load(f)
    assert res["config"]["pairs_sha256"] == cv.PAIRS_SHA256
    names = ["projection"] + cv.available_predictions()
    assert sorted(res["arms"]) == sorted(names)
    for name in names:
        errs = cv.arm_errors(pairs, None if name == "projection" else cv.read_predictions(name))
        want = res["arms"][name]["all"]
        assert round(float(np.median([e[0] for e in errs])), 4) == \
            pytest.approx(want["median_deg"], abs=1e-4), name
        assert round(float(np.mean([e[3] for e in errs])), 4) == \
            pytest.approx(want["fallback_rate"], abs=1e-4), name


def test_flat_check_arm_reproduces_the_committed_projection():
    """The geometry arms' plumbing, pinned: recomputing today's projection through
    crossview_arms.geometry lands on the pair list's proj_x / proj_y."""
    pairs = cv.read_frozen_pairs()
    preds = cv.read_predictions("proj_flat_check")
    worst = max(float(cv.angular_error_deg(preds[p["pair_id"]]["x"], preds[p["pair_id"]]["y"],
                                           p["proj_x"], p["proj_y"])) for p in pairs)
    assert worst < 0.01
