"""Tests for pu_arm.py (crosswalk_s7 day-1 diagnostics): card merging, the Vistas box geometry, the leak
classification through train.py's own dense_ignore weight, and the refusal to touch population (b).

Synthetic inputs only; no imagery, roots or verdict files are read.
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

pytest.importorskip("numpy")
pytest.importorskip("torch")

import pu_arm as pa  # noqa: E402
from equirect_tiling import View, perspective_point_to_equirect  # noqa: E402


def test_merge_marks_groups_within_0_01_and_flags_conflicts():
    rows = [{"pano_id": "p", "x": 0.500, "y": 0.60, "verdict": "crosswalk", "src": "confirm"},
            {"pano_id": "p", "x": 0.508, "y": 0.60, "verdict": "crosswalk", "src": "s4_stepB"},
            {"pano_id": "p", "x": 0.995, "y": 0.60, "verdict": "crosswalk", "src": "confirm"},
            {"pano_id": "p", "x": 0.003, "y": 0.60, "verdict": "unsure", "src": "s4_stepB"},
            {"pano_id": "q", "x": 0.500, "y": 0.60, "verdict": "not_crosswalk", "src": "confirm"}]
    m = pa.merge_marks(rows)
    assert [(u["pano_id"], u["x"], u["verdict"]) for u in m] == [
        ("p", 0.5, "crosswalk"), ("p", 0.995, "conflict"), ("q", 0.5, "not_crosswalk")]
    assert m[0]["sources"] == "confirm;s4_stepB"


def test_vectorised_projection_matches_the_scalar_one():
    view = View(120.0, -30.0, 90.0, 90.0, 1024, 1024)
    for u, v in [(0.1, 0.2), (0.5, 0.5), (0.93, 0.81)]:
        x, y = pa.persp_to_equirect([u], [v], view)
        xs, ys = perspective_point_to_equirect(u, v, view)
        assert math.isclose(float(x[0]), xs, abs_tol=1e-9) and math.isclose(float(y[0]), ys, abs_tol=1e-9)


def test_span_and_dilation_wrap_the_seam():
    xl, xw = pa.span_x([0.995, 0.998, 0.002])
    assert math.isclose(xl, 0.995) and math.isclose(xw, 0.007, abs_tol=1e-9)
    box = pa.dilate_box({"xl": xl, "xw": xw, "yt": 0.60, "yb": 0.62})
    d = 30 / 1024
    assert box[0] > box[1]                                   # wraps, train.py's convention
    assert math.isclose(box[0], (0.995 - d) % 1, abs_tol=1e-6) and math.isclose(box[1], 0.002 + d, abs_tol=1e-6)
    assert math.isclose(box[2], 0.60 - 30 / 512, abs_tol=1e-6)
    assert pa.dilate_box({"xl": 0.1, "xw": 0.99, "yt": 0.5, "yb": 0.6})[:2] == [0.0, 1.0]


def test_view_boxes_find_a_component_and_drop_specks():
    np = pytest.importorskip("numpy")
    pytest.importorskip("skimage")
    seg = np.zeros((64, 64), dtype=np.int64)
    seg[40:50, 20:44] = 23          # a crosswalk blob
    seg[5, 5] = 8                   # one pixel: below MIN_AREA_PX
    view = View(0.0, -30.0, 90.0, 90.0, 64, 64)
    boxes = pa.view_boxes(seg, view)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["area_px"] == 240 and b["yt"] < b["y"] < b["yb"] and b["xw"] < 0.25
    assert b["xl"] <= b["x"] <= b["xl"] + b["xw"] or b["xl"] + b["xw"] > 1   # centroid inside the box


def test_background_uses_train_py_discs_and_boxes():
    pytest.importorskip("torch")
    ign = [(0.50, 0.60)]
    box = [pa.dilate_box({"xl": 0.98, "xw": 0.04, "yt": 0.60, "yb": 0.62}, 0)]
    pts = [(0.50, 0.60),             # on the ignore point
           (0.50 + 25 / 1024, 0.60),  # inside its 30 px disc
           (0.50 + 40 / 1024, 0.60),  # outside it
           (0.999, 0.61),            # inside the seam-wrapping box
           (0.25, 0.70)]             # nothing near
    assert pa.background_at(pts, ign, box) == [False, False, True, False, True]


def test_score_reports_leak_sources_and_s4_blindness(tmp_path):
    pytest.importorskip("torch")
    preds = tmp_path / "s4.jsonl"
    preds.write_text('{"pano_id": "p", "preds": [[0.30, 0.60, 0.40], [0.80, 0.60, 0.03]]}\n', encoding="utf-8")
    marks = [{"pano_id": "p", "x": 0.30, "y": 0.60},   # s4 peak there
             {"pano_id": "p", "x": 0.60, "y": 0.60},   # Vistas component there
             {"pano_id": "p", "x": 0.70, "y": 0.60},   # OSM node there, 15 m away
             {"pano_id": "p", "x": 0.80, "y": 0.60}]   # only a sub-floor s4 peak: leaks
    vistas = [{"pano_id": "p", "components": [{"xl": 0.595, "xw": 0.01, "yt": 0.595, "yb": 0.605}]}]
    osm = [{"pano_id": "p", "x": "0.70", "y": "0.60", "range_m": "15", "osm_class": "unmarked"}]
    out = pa.score(marks, ["p"], [str(preds)] * 3, vistas, osm)
    assert [m["bg_primary"] for m in out] == [0, 0, 0, 1]
    assert [m["bg_osm_marked_only"] for m in out] == [0, 0, 1, 1]      # the node is unmarked
    assert [m["covered_vistas"] for m in out] == [0, 1, 0, 0]
    assert [m["s4_blind_0.05"] for m in out] == [0, 1, 1, 1]
    res = pa.summarise("t", out, {})
    assert res["rules"]["primary"]["leaked"] == 1 and res["gate_half"]["verdict"] == "fails"


def test_population_b_refuses_without_jon_approval(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("population (b) was read")
    monkeypatch.setattr(pa, "population_b", boom)
    with pytest.raises(SystemExit, match="not approved"):
        pa.cmd_leak_check(SimpleNamespace(population="b", keep="x", jon_approved_b=False))
