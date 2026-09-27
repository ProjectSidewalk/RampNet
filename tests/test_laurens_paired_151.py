"""Tests for the paired Laurens analysis (#151).

CPU only, no network, no depth payloads. The geometry and the counting helpers are checked on
closed forms; the committed ``analysis_out/laurens_paired_151.json`` is checked to re-derive every
row except the curb probe from committed inputs and every table from its rows, to carry the
content hash it claims, and to pin the headline numbers; the doc's tables are checked verbatim.
"""
import json
import math
import os
import sys
from collections import namedtuple

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import laurens_paired_151 as lp  # noqa: E402
from rampnet.detection_eval import GroundTruth, aggregate, score_pano  # noqa: E402

COMMITTED = os.path.join(REPO, "analysis_out", "laurens_paired_151.json")
DOC = os.path.join(REPO, "docs", "laurens_paired_151.md")


# ---------------------------------------------------------------------------
# closed forms

def test_ground_point_is_the_labelers_flat_convention():
    # centre column, 45 degrees of depression: h metres straight along the heading
    e, n, d = lp.ground_point(0.0, 0.0, 0.0, 0.5, 0.75, 2.0)
    assert (e, n, d) == pytest.approx((0.0, 2.0, 2.0))
    # a quarter turn right of a north heading is due east
    e, n, _ = lp.ground_point(10.0, 5.0, 0.0, 0.75, 0.75, 2.0)
    assert (e, n) == pytest.approx((12.0, 5.0))
    # heading 90 (east), centre column: east
    e, n, _ = lp.ground_point(0.0, 0.0, 90.0, 0.5, 0.75, 2.0)
    assert (e, n) == pytest.approx((2.0, 0.0), abs=1e-9)
    assert lp.ground_point(0, 0, 0, 0.5, 0.505, 2.0) is None        # within 0.02 rad of the horizon
    assert lp.ground_point(0, 0, 0, 0.5, 0.53, 2.6) is None         # beyond 25 m
    assert lp.ground_point(0, 0, 0, 0.5, 0.53, 2.6, max_range_m=40) is not None


def test_greedy_pairs_is_one_to_one_by_ascending_distance():
    a = [(0.0,), (1.0,)]
    b = [(0.9,), (5.0,)]
    d = lambda p, q: abs(p[0] - q[0])  # noqa: E731
    # a[1]-b[0] (0.1) claims b[0] first; a[0] is then left with nothing within 2
    assert lp.greedy_pairs(a, b, 2.0, d) == [(1, 0, pytest.approx(0.1))]
    assert lp.greedy_pairs(a, b, 5.0, d) == [(0, 1, 5.0), (1, 0, pytest.approx(0.1))]


def test_prf_matches_the_benchmark_aggregate():
    gt = GroundTruth([(0.1, 0.6), (0.3, 0.6), (0.6, 0.6)], [], True)
    s = score_pano([(0.1, 0.6, 0.9), (0.8, 0.6, 0.7)], gt)
    rep = aggregate([s])
    assert lp.prf(s.tp, s.fp, s.tp, s.n_gt) == pytest.approx((rep.precision, rep.recall, rep.f1))
    p, r, f = lp.prf_arrays(np.array([[s.tp, s.fp, s.tp, s.n_gt]], dtype=float))
    assert (p[0], r[0], f[0]) == pytest.approx((rep.precision, rep.recall, rep.f1))
    assert lp.prf(0, 0, 0, 0) == (0.0, 0.0, 0.0)


def test_bootstrap_weights_resample_every_pair_and_are_seeded():
    w = lp.bootstrap_weights(7, draws=50, seed=3)
    assert w.shape == (50, 7) and (w.sum(1) == 7).all()
    assert (w == lp.bootstrap_weights(7, draws=50, seed=3)).all()
    assert not (w == lp.bootstrap_weights(7, draws=50, seed=4)).all()


Plane = namedtuple("Plane", "nx ny nz d")
Payload = namedtuple("Payload", "width height planes indices")


def test_window_steps_reads_a_known_curb():
    """Road at 2.40 m below the camera, a sidewalk plane 0.15 m higher (2.25 m) on raw columns
    >= 256: the step at their shared boundary is 0.15 m, and a one-plane window has none."""
    sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
    import recall_by_depth_112 as rbd
    w, h = 512, 256
    road, walk = Plane(0.0, 0.0, -1.0, 2.40), Plane(0.0, 0.0, -1.0, 2.25)
    idx = bytes((0 if r < h // 2 else (2 if c >= w // 2 else 1)) for r in range(h) for c in range(w))
    p = Payload(w, h, [Plane(0, 0, 0, 0), road, walk], idx)
    n, steps = lp.window_steps(p, 0.5, 0.7, rbd.image_ray, rbd.intersect)
    assert n == 2 and steps and max(steps) == pytest.approx(0.15, abs=1e-9)
    n, steps = lp.window_steps(p, 0.25, 0.7, rbd.image_ray, rbd.intersect)
    assert n == 1 and steps == []
    # review N6: a boundary one column right of the window's last column is not counted
    x_edge = (256 - lp.PROBE_HALF_COLS - 1 + 0.5) / w   # the window's last column is raw 255
    n, steps = lp.window_steps(p, x_edge, 0.7, rbd.image_ray, rbd.intersect)
    assert n == 1 and steps == []
    assert lp.plane_z(road, 3.0, -1.0) == pytest.approx(2.40)


# ---------------------------------------------------------------------------
# the committed artifact

@pytest.fixture(scope="module")
def committed():
    with open(COMMITTED, encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_artifact_is_lf_rounded_and_hashed(committed):
    with open(COMMITTED, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert lp.rows_sha256(committed) == committed["rows_sha256"]
    assert committed["rows_sha256"] == "034e84cd18277d6099a44d2639b29dacb165a1f68cb10b1e9913d48393739fed"
    for row in committed["gt"]:
        for key, v in row.items():
            if isinstance(v, float) and key not in ("x", "y"):
                assert round(v, lp.ND) == v, (key, v)


def test_tables_rederive_from_the_committed_rows(committed):
    assert json.loads(json.dumps(lp.tables(committed))) == committed["tables"]


def test_rows_rederive_from_the_committed_inputs(committed):
    fresh = lp.build(probe=committed["curb_probe"])
    for k in lp.ROW_KEYS + ("inputs_sha256", "constants", "rows_sha256"):
        assert json.dumps(fresh.get(k), sort_keys=True) == json.dumps(committed.get(k), sort_keys=True), k


def test_whole_arm_scores_reproduce_the_2026_09_03_table(committed):
    s = {r["leg"]: r for r in committed["tables"]["scores"]}
    assert s["rampnet@0.55"]["laurens_gsv_whole_arm"]["F1"] == 0.6588
    assert s["rampnet@0.55"]["laurens_mapillary_whole_arm"]["F1"] == 0.5434
    assert s["rampnet@0.55"]["delta_whole_arm_F1"] == 0.1154
    assert s["y11x_pano_h200"]["delta_whole_arm_F1"] == 0.0387
    assert s["y26_pano"]["delta_whole_arm_F1"] == -0.0359


def test_headline_numbers(committed):
    t = committed["tables"]
    assert t["pairing"]["pairs"] == 47
    assert t["pairing"]["within_radius_of_other_arm"] == {"laurens_gsv": 51, "laurens_mapillary": 49}
    rn = next(r for r in t["scores"] if r["leg"] == "rampnet@0.55")["delta"]
    assert rn["F1"] == 0.1121 and rn["F1_ci"] == [0.0333, 0.1989]
    head = {(h["rampnet_leg"], h["yolo"]): h for h in t["headline"]}
    assert [head[("rampnet@0.55", y)]["yolo_dF1"] for y in lp.YOLO_PANO] == [0.0284, -0.0003, -0.0198]
    assert head[("rampnet@0.55", "y11x_pano_h200")]["share_draws_rampnet_larger"] == 0.973
    assert t["ramps"]["matched"] == 86
    two = t["two_by_two"]
    assert {k: two["rampnet@0.55"][k] for k in ("both", "gsv_only", "mly_only", "neither", "n")} == {
        "both": 21, "gsv_only": 23, "mly_only": 16, "neither": 26, "n": 86}
    assert two["rampnet@0.55"]["mcnemar_p"] == 0.3368


def test_b1_the_excess_over_yolo_does_not_survive_the_same_input_leg_or_the_2x2(committed):
    """PR #201 review B1, pinned: on the same-input RampNet leg at 0.55 no RampNet-minus-YOLO
    interval clears zero, and on the ramps both reviews contain the YOLO arms gain about as
    much as RampNet with no 2x2 asymmetry significant."""
    t = committed["tables"]
    head = {(h["rampnet_leg"], h["yolo"]): h for h in t["headline"]}
    assert [head[("rampnet@0.55", y)]["clears_zero"] for y in lp.YOLO_PANO] == [False, True, True]
    assert [head[("rampnet_r2048@0.55", y)]["clears_zero"] for y in lp.YOLO_PANO] == [False, False, False]
    assert [head[("rampnet_r2048@0.30", y)]["clears_zero"] for y in lp.YOLO_PANO] == [False, True, True]
    assert head[("rampnet_r2048@0.55", "y11l_pano")]["difference_ci"] == [-0.02, 0.1716]
    two = t["two_by_two"]
    assert {y: (two[y]["gsv_only"], two[y]["mly_only"]) for y in lp.YOLO_PANO} == {
        "y11x_pano_h200": (19, 14), "y11l_pano": (20, 15), "y26_pano": (18, 16)}
    assert all(c["mcnemar_p"] > 0.05 for c in two.values())
    assert two["rampnet@0.55 (3 m)"]["mcnemar_p"] == 0.1078


def test_mcnemar_exact():
    assert lp.mcnemar_exact(23, 16) == pytest.approx(0.3368, abs=1e-4)
    assert lp.mcnemar_exact(5, 5) == 1.0
    assert lp.mcnemar_exact(0, 0) == 1.0
    assert lp.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_input_hashes_ignore_line_endings():
    """PR #201 review M5: a CRLF checkout must hash like the LF one."""
    import hashlib
    raw = b'{"a": 1}\n{"b": 2}\n'
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == \
        hashlib.sha256(raw.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n")).hexdigest()
    hashes = lp.input_hashes()
    assert "analysis_out/recall_by_depth_112.json" not in hashes
    assert "analysis_out/recall_by_depth_112.json#panos[city=laurens_gsv]" in hashes


def test_near_miss_and_probe_numbers(committed):
    t = committed["tables"]
    nm = t["near_miss"]
    assert nm["laurens_mapillary"]["verdict"]["whole_arm"]["delta"] == 0.0038
    assert nm["laurens_gsv"]["verdict"]["whole_arm"]["delta"] == -0.0053
    cp = t["curb_probe"]["gt_within_8m_depth"]
    assert (cp["windows"], cp["one_ground_plane_or_none"], cp["curb_sized_step"]) == (28, 24, 0)
    far = t["curb_probe"]["by_range"][-1]
    assert far["band"] == "18 m+" and (far["gt"]["windows"], far["gt"]["curb_sized_step"]) == (40, 22)


def test_rampnet_030_has_no_gsv_side(committed):
    """Documents a current gap, not a requirement: there is no laurens_gsv op_cache yet. When one
    is produced, input_hashes() hashes it and this test should be updated with the new rows."""
    row = next(r for r in committed["tables"]["scores"] if r["leg"] == "rampnet@0.30")
    assert row["laurens_gsv"] is None and row["delta"] is None
    assert not os.path.exists(os.path.join(REPO, "analysis_out", "op_cache", "laurens_gsv.json"))


def test_the_doc_tables_are_the_committed_tables(committed):
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read().replace("\r\n", "\n")
    tabs = lp.md_tables(committed["tables"])
    assert len(tabs) == 10
    for name, tab in tabs.items():
        assert tab in doc, f"table {name!r} in docs/laurens_paired_151.md does not match the rows"
