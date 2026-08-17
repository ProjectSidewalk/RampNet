"""Unit tests for the crop-window sizing eval (#114).

Pure geometry — no imagery, no network, no ``.model_cache``. The guarantees are
about the things that fail silently: the v1 formula port (raw vs the
resolution-normalized variant must be bit-identical at the calibration height),
seam-wrap containment (a window crossing the equirectangular seam must still
contain a box on the far side of x=0), and clamp-by-shift (a window near a pole
shifts instead of zero-padding, and the margins must reflect the shift).
"""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import crop_window_eval as cwe  # noqa: E402


# ---------------------------------------------------------------------------
# v1 formula: raw vs normalized

def test_v1_raw_pinned_values():
    # Horizon at calibration height: distance = intercept alone.
    assert cwe.v1_raw_side(3328, 13312, 6656) == pytest.approx(248.329067, abs=1e-4)
    # Deep below horizon: distance hits 0 -> max clamp.
    assert cwe.v1_raw_side(4500, 13312, 6656) == 1500.0
    # Top of frame: smallest reachable in-frame size (the 50 px floor is
    # unreachable within a 6656-height frame).
    assert cwe.v1_raw_side(0, 13312, 6656) == pytest.approx(54.649733, abs=1e-4)


def test_v1_norm_identical_to_raw_at_calibration_height():
    for y in (0, 1000, 3328, 4000, 4500, 6656):
        assert cwe.v1_norm_side(y, 13312, 6656) == cwe.v1_raw_side(y, 13312, 6656)


def test_v1_norm_is_raw_computed_in_reference_space():
    # At any height, the normalized rule must equal: map y to 6656-reference,
    # run the raw formula there, scale the result back.
    for pano_h in (2048, 4096, 8192):
        for y_frac in (0.1, 0.5, 0.55, 0.6, 0.75):
            y = y_frac * pano_h
            expected = cwe.v1_raw_side(y * 6656 / pano_h, 2 * pano_h, 6656) * (pano_h / 6656)
            assert cwe.v1_norm_side(y, 2 * pano_h, pano_h) == pytest.approx(expected)


def test_v1_raw_resolution_dependence_is_real():
    # The defect the normalized variant exists to fix: the same world geometry
    # (same y fraction) gets a different relative window size at 2048 height.
    raw = cwe.v1_raw_side(1200, 4096, 2048)
    norm = cwe.v1_norm_side(1200, 4096, 2048)
    assert raw == pytest.approx(295.364834, abs=1e-4)
    assert norm == pytest.approx(152.554668, abs=1e-4)


# ---------------------------------------------------------------------------
# geo-v1.5

def test_geo_monotone_nearer_is_larger():
    h, w = 2048, 4096
    assert cwe.geo_v15_side(0.75 * h, w, h) > cwe.geo_v15_side(0.6 * h, w, h) \
        > cwe.geo_v15_side(0.55 * h, w, h)


def test_geo_above_horizon_falls_back_to_far_clamp():
    h, w = 2048, 4096
    far = cwe.geo_v15_side(0.4 * h, w, h)
    # d clamps to D_MAX: apparent = RAMP_W / D_MAX * px_per_rad, padded by ratio.
    import math
    expected = (cwe.GEO_RAMP_W_M / cwe.GEO_D_MAX_M) * (w / (2 * math.pi)) / cwe.GEO_TARGET_RATIO
    assert far == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Window geometry (mirrors CropRunner.compute_crop_box)

def test_crop_window_seam_wrap_and_clamps():
    assert cwe.crop_window(10, 600, 200, 4096, 2048) == (4006, 500, 200, False)
    assert cwe.crop_window(2000, 30, 200, 4096, 2048) == (1900, 0, 200, True)
    assert cwe.crop_window(2000, 2040, 200, 4096, 2048) == (1900, 1848, 200, True)


def test_crop_window_size_capped_at_pano_dims():
    left, top, size, _ = cwe.crop_window(100, 1000, 5000, 4096, 2048)
    assert size == 2048 and top == 0


def test_box_margins_across_the_seam():
    # Window [4000, 4096) U [0, 204): a box at x=[50, 150] is inside via wrap.
    margins = cwe.box_margins((50, 100, 150, 200), (4000, 80, 300, False), 4096)
    assert margins == (146.0, 54.0, 20, 180)
    assert min(margins) >= 0
    # A box on the far side of the pano is far outside, signed circularly.
    far = cwe.box_margins((2000, 100, 2100, 200), (4000, 80, 300, False), 4096)
    assert far[0] == -2000.0


def test_score_box_contained_simple_case():
    box = (0.5, 0.6, 0.02, 0.02)  # 82x41 px on 4096x2048
    prompt = (0.5 * 4096, 0.6 * 2048)
    for rule in cwe.RULES:
        row = cwe.score_box(box, prompt, rule, 4096, 2048)
        assert row["contained"], rule
        assert 0.0 < row["context_ratio"] < 0.6
        assert row["margin_norm"] >= 0.0


def test_score_box_detects_truncation():
    # A big near-field box against a deliberately tiny window: not contained,
    # negative margin shows the violation depth.
    box = (0.5, 0.75, 0.2, 0.2)
    row = cwe.score_box(box, (2048, 1536), "v1-norm", 4096, 2048)
    side = row["predicted_side"]
    if row["contained"]:
        pytest.skip("v1-norm window unexpectedly large enough")
    assert row["margin_norm"] < 0
    assert row["context_ratio"] == pytest.approx(max(0.2 * 4096, 0.2 * 2048) / side)


# ---------------------------------------------------------------------------
# YOLO parsing keeps extent

def test_parse_yolo_boxes_keeps_extent(tmp_path):
    p = tmp_path / "pano.txt"
    p.write_text("0 0.5 0.6 0.02 0.03\n\n0 0.1 0.2 0.004 0.005\n")
    boxes = cwe.parse_yolo_boxes(str(p))
    assert boxes == [(0.5, 0.6, 0.02, 0.03), (0.1, 0.2, 0.004, 0.005)]


@pytest.mark.parametrize("line", [
    "0 0.5 0.6 0.02",           # missing field
    "0 x 0.6 0.02 0.03",        # non-numeric
    "0 1.5 0.6 0.02 0.03",      # center out of range
    "0 0.5 0.6 0 0.03",         # degenerate extent
])
def test_parse_yolo_boxes_strict(tmp_path, line):
    p = tmp_path / "bad.txt"
    p.write_text(line + "\n")
    with pytest.raises(ValueError):
        cwe.parse_yolo_boxes(str(p))


# ---------------------------------------------------------------------------
# End-to-end on synthetic gold + records

def test_run_eval_modes_and_coverage():
    gold = {"pano_a": [(0.5, 0.6, 0.02, 0.02), (0.2, 0.7, 0.03, 0.03)]}
    records = {"pano_a": {
        "width": 4096, "height": 2048,
        "detections": [
            {"x_normalized": 0.501, "y_normalized": 0.601, "confidence": 0.9},
            {"x_normalized": 0.201, "y_normalized": 0.699, "confidence": 0.3},  # below floor
            {"x_normalized": 0.9, "y_normalized": 0.6, "confidence": 0.8},      # no ramp
        ],
    }}
    rows, coverage = cwe.run_eval(gold, records, min_confidence=0.55)
    gold_rows = [r for r in rows if r["mode"] == "gold-center"]
    det_rows = [r for r in rows if r["mode"] == "detection"]
    assert len(gold_rows) == 2 * len(cwe.RULES)
    assert len(det_rows) == 1 * len(cwe.RULES)  # only box 0 is covered at 0.55
    assert coverage["boxes_total"] == 2
    assert coverage["boxes_covered"] == 1
    assert coverage["detections_kept"] == 2
    assert coverage["detections_unmatched"] == 1

    summary = cwe.summarize(rows)
    for mode in ("gold-center", "detection"):
        for rule in cwe.RULES:
            s = summary[mode][rule]
            assert 0.0 <= s["containment"] <= 1.0
            assert s["containment_ci"][0] <= s["containment"] <= s["containment_ci"][1]


def test_run_eval_skips_panos_without_records():
    gold = {"pano_missing": [(0.5, 0.6, 0.02, 0.02)]}
    rows, coverage = cwe.run_eval(gold, {}, min_confidence=0.55)
    assert rows == []
    assert coverage["panos_missing_record"] == 1


# ---------------------------------------------------------------------------
# Bundle extent-gold mode (box_gallery boxes.json, #116)

BOXES_JSON = {
    "run_name": "testville",
    "box_rule": {"version": 2, "text": "rule text"},
    "crop_fov_deg": 90,
    "panos": {
        "P1": {
            "det:0": {"point": {"x": 0.501, "y": 0.601}, "status": "boxed",
                      "cx": 0.5, "cy": 0.6, "w": 0.02, "h": 0.02},
            "missed:0": {"point": {"x": 0.2, "y": 0.7}, "status": "boxed",
                         "cx": 0.2, "cy": 0.7, "w": 0.03, "h": 0.03,
                         "edge_flag": True},
            "missed:1": {"point": {"x": 0.9, "y": 0.6}, "status": "cant"},
            # A box wrapping the equirectangular seam (cx near 0).
            "det:1": {"point": {"x": 0.004, "y": 0.62}, "status": "boxed",
                      "cx": 0.004, "cy": 0.62, "w": 0.03, "h": 0.02},
        },
    },
}


def _write_bundle(tmp_path):
    import json
    d = tmp_path / "testville"
    d.mkdir()
    (d / "boxes.json").write_text(json.dumps(BOXES_JSON), encoding="utf-8")
    return str(d)


def test_load_bundle_boxes_skips_cant_and_keeps_order(tmp_path):
    gold, prompts, meta = cwe.load_bundle_boxes(_write_bundle(tmp_path))
    assert [k for k in gold] == ["P1"]
    # det items first (by index), then missed — cant excluded.
    assert [prompts[("P1", i)]["key"] for i in range(3)] == ["det:0", "det:1", "missed:0"]
    assert meta["n_boxed"] == 3 and meta["n_cant"] == 1 and meta["n_edge_flag"] == 1
    assert meta["box_rule"]["version"] == 2


def test_box_pixels_wrap_keeps_seam_boxes_unclamped():
    clamped = cwe.box_pixels((0.004, 0.62, 0.03, 0.02), 4096, 2048)
    wrapped = cwe.box_pixels((0.004, 0.62, 0.03, 0.02), 4096, 2048, wrap_x=True)
    assert clamped[0] == 0.0                      # default truncates at the seam
    assert wrapped[0] == pytest.approx(-45.056)   # wrap keeps the true left edge
    assert wrapped[2] - wrapped[0] == pytest.approx(0.03 * 4096)


def test_score_box_road_margin_ratio():
    # Window centered on the box: bottom margin = (size - box_h) / 2.
    box = (0.5, 0.6, 0.02, 0.02)
    row = cwe.score_box(box, (0.5 * 4096, 0.6 * 2048), "geo-v1.5", 4096, 2048)
    box_h = 0.02 * 2048
    expected = (row["predicted_side"] - box_h) / 2 / box_h
    assert row["road_margin_ratio"] == pytest.approx(expected, abs=0.02)


def test_run_bundle_eval_modes_and_wrap_containment(tmp_path):
    gold, prompts, meta = cwe.load_bundle_boxes(_write_bundle(tmp_path))
    records = {"P1": {"width": 4096, "height": 2048, "detections": []}}
    rows, coverage = cwe.run_bundle_eval(gold, prompts, records)
    assert coverage["boxes_total"] == 3
    assert coverage["det_prompted"] == 2 and coverage["missed_no_detection"] == 1
    # det items score in both modes, missed only in gold-center.
    assert len([r for r in rows if r["mode"] == "detection"]) == 2 * len(cwe.RULES)
    assert len([r for r in rows if r["mode"] == "gold-center"]) == 3 * len(cwe.RULES)
    # The seam-wrapping box is contained when its window wraps with it.
    seam = [r for r in rows if r["key"] == "det:1" and r["mode"] == "gold-center"]
    assert seam and all(r["contained"] for r in seam)
    summary = cwe.summarize(rows)
    for rule in cwe.RULES:
        assert summary["gold-center"][rule]["n"] == 3
        assert not math.isnan(summary["gold-center"][rule]["road_margin_p50"])


# ---------------------------------------------------------------------------
# Scale vs shape: the decomposition the rule comparison turns on

def test_required_side_is_rule_independent_and_exact():
    # required_side is a property of (box, prompt) alone. A window at exactly that
    # side contains the box; a hair under it does not.
    box, pano_w, pano_h = (0.5, 0.6, 0.04, 0.02), 4096, 2048
    prompt = (0.505 * pano_w, 0.6 * pano_h)
    req = cwe.required_side(box, prompt, pano_w, pano_h)
    bpx = cwe.box_pixels(box, pano_w, pano_h)
    at = cwe.crop_window(prompt[0], prompt[1], req + 2, pano_w, pano_h)
    under = cwe.crop_window(prompt[0], prompt[1], req - 4, pano_w, pano_h)
    assert min(cwe.box_margins(bpx, at, pano_w)) >= 0
    assert min(cwe.box_margins(bpx, under, pano_w)) < 0


def test_required_side_handles_a_seam_wrapping_box():
    box, pano_w, pano_h = (0.004, 0.62, 0.03, 0.02), 4096, 2048
    prompt = (0.004 * pano_w, 0.62 * pano_h)
    req = cwe.required_side(box, prompt, pano_w, pano_h, wrap_x=True)
    # The box is 0.03 * 4096 wide and centred on the prompt: the circular distance
    # must not read the far-side wrap as a near-full-pano requirement.
    assert req == pytest.approx(0.03 * pano_w, rel=1e-6)


def test_size_ratio_spread_is_blind_to_a_constant_rescale():
    # The whole point: two rules differing ONLY by a scale constant must score the
    # same accuracy. Containment and context ratio cannot say that; this can.
    boxes = [(0.5, 0.55 + 0.03 * i, 0.02 + 0.004 * i, 0.01) for i in range(8)]
    pano_w, pano_h = 4096, 2048
    base, scaled = [], []
    for box in boxes:
        prompt = (box[0] * pano_w, box[1] * pano_h)
        row = cwe.score_box(box, prompt, "v1-norm", pano_w, pano_h)
        base.append(row["size_ratio"])
        scaled.append(row["predicted_side"] * 4 / row["required_side"])
    spread = cwe._quantiles(base)
    spread_scaled = cwe._quantiles(scaled)
    assert (spread_scaled[0.9] / spread_scaled[0.1]) == pytest.approx(
        spread[0.9] / spread[0.1], rel=1e-6)


def test_rescale_sweep_reproduces_the_unscaled_scoring_at_k1():
    gold = {"pano_a": [(0.5, 0.62, 0.03, 0.012), (0.2, 0.7, 0.05, 0.02)]}
    records = {"pano_a": {"width": 4096, "height": 2048, "detections": [
        {"x_normalized": 0.501, "y_normalized": 0.621, "confidence": 0.9},
        {"x_normalized": 0.201, "y_normalized": 0.701, "confidence": 0.8}]}}
    rows, _ = cwe.run_eval(gold, records, min_confidence=0.55)
    sweep = cwe.rescale_sweep(rows, mode="detection", ks=(1.0, 4.0))
    for rule in cwe.RULES:
        sel = [r for r in rows if r["mode"] == "detection" and r["rule"] == rule]
        direct = sum(1 for r in sel if r["contained"]) / len(sel)
        assert sweep[rule][0]["k"] == 1.0
        assert sweep[rule][0]["containment"] == pytest.approx(direct)
        # Scaling up can only help containment, and never shrinks the window.
        assert sweep[rule][1]["containment"] >= sweep[rule][0]["containment"]
        assert sweep[rule][1]["side_p50"] >= sweep[rule][0]["side_p50"]


def test_rescale_sweep_respects_the_pano_dimension_cap():
    # A rule cannot be scaled past the image it cuts from; the sweep must not
    # pretend otherwise, or "just multiply by k" would look free at any k.
    gold = {"pano_a": [(0.5, 0.75, 0.02, 0.01)]}
    records = {"pano_a": {"width": 4096, "height": 2048, "detections": [
        {"x_normalized": 0.5, "y_normalized": 0.75, "confidence": 0.9}]}}
    rows, _ = cwe.run_eval(gold, records, min_confidence=0.55)
    sweep = cwe.rescale_sweep(rows, mode="detection", ks=(20.0,))
    for rule in cwe.RULES:
        assert sweep[rule][0]["side_p50"] <= 2048
        assert sweep[rule][0]["capped"] == 1


def test_context_ratio_splits_per_axis():
    box = (0.5, 0.6, 0.04, 0.005)          # 164 x 10 px: a wide, flat apron
    row = cwe.score_box(box, (0.5 * 4096, 0.6 * 2048), "geo-v1.5", 4096, 2048)
    assert row["context_ratio"] == pytest.approx(row["context_ratio_h"])
    assert row["context_ratio_v"] < row["context_ratio_h"]
    assert row["box_aspect"] == pytest.approx((0.04 * 4096) / (0.005 * 2048))


# ---------------------------------------------------------------------------
# Bundle mode: gold: items, completeness

GOLD_BOXES_JSON = {
    "run_name": "goldville",
    "box_rule": {"version": 2, "text": "rule text"},
    "crop_fov_deg": 90,
    "panos": {"P1": {
        "gold:0": {"point": {"x": 0.5, "y": 0.6}, "status": "boxed",
                   "cx": 0.5, "cy": 0.6, "w": 0.02, "h": 0.01},
        "gold:1": {"point": {"x": 0.2, "y": 0.7}, "status": "boxed",
                   "cx": 0.2, "cy": 0.7, "w": 0.02, "h": 0.01},
    }},
}


def test_bundle_gold_items_get_a_detection_prompt(tmp_path):
    # --from-manual-labels emits gold:<i> keys, which carry no adjudicated detection
    # linkage. Without matching them here the GSV arm would silently produce
    # gold-center-only numbers, not comparable to a det:-keyed bundle's headline.
    import json as _json
    d = tmp_path / "goldville"
    d.mkdir()
    (d / "boxes.json").write_text(_json.dumps(GOLD_BOXES_JSON), encoding="utf-8")
    gold, prompts, _ = cwe.load_bundle_boxes(str(d))
    records = {"P1": {"width": 4096, "height": 2048, "detections": [
        {"x_normalized": 0.501, "y_normalized": 0.601, "confidence": 0.9},   # matches gold:0
        {"x_normalized": 0.9, "y_normalized": 0.5, "confidence": 0.8},       # matches nothing
    ]}}
    rows, coverage = cwe.run_bundle_eval(gold, prompts, records, min_confidence=0.55)
    assert coverage["gold_matched"] == 1 and coverage["gold_unmatched"] == 1
    det = [r for r in rows if r["mode"] == "detection"]
    assert {r["key"] for r in det} == {"gold:0"}
    assert len(det) == len(cwe.RULES)


def test_check_gold_complete_warns_when_boxes_cover_a_subset(tmp_path):
    import json as _json
    d = tmp_path / "partial"
    d.mkdir()
    (d / "boxes.json").write_text(_json.dumps(BOXES_JSON), encoding="utf-8")
    (d / "verdicts.json").write_text(_json.dumps({"panos": {"P1": {
        "dets": [True, True], "missed": [{"x": 0.2, "y": 0.7}, {"x": 0.3, "y": 0.7}],
    }}}), encoding="utf-8")
    _, _, meta = cwe.load_bundle_boxes(str(d))
    assert meta["n_adjudicated"] == 4                 # 2 True dets + 2 sure missed
    assert meta["n_boxed"] + meta["n_cant"] == 4      # this one happens to line up
    assert meta["completeness_warning"] is None
    # Drop an item from the gold and the shortfall must be reported, not absorbed.
    thin = {**BOXES_JSON, "panos": {"P1": {k: v for k, v in
                                           BOXES_JSON["panos"]["P1"].items()
                                           if k != "missed:1"}}}
    (d / "boxes.json").write_text(_json.dumps(thin), encoding="utf-8")
    _, _, meta = cwe.load_bundle_boxes(str(d))
    assert "covers 3 of 4 adjudicated ramps" in meta["completeness_warning"]


def test_count_adjudicated_ramps_skips_partially_judged_panos(tmp_path):
    import json as _json
    p = tmp_path / "verdicts.json"
    p.write_text(_json.dumps({"panos": {
        "P1": {"dets": [True, False], "missed": [{"x": 0.1, "y": 0.5}]},
        "P2": {"dets": [True, None], "missed": [{"x": 0.1, "y": 0.5}]},   # unusable
        "P3": {"dets": [True], "missed": [{"x": 0.1, "y": 0.5, "unsure": True}]},
    }}), encoding="utf-8")
    assert cwe.count_adjudicated_ramps(str(p)) == 3   # P1: 1+1, P2: 0, P3: 1+0


def test_cam_height_scales_geo_windows():
    # geo-v1.5's one free parameter: a lower camera means a nearer ramp for the same
    # depression, so a bigger window. Reading the module global at call time is what
    # makes --cam-height work at all.
    h, w = 2048, 4096
    tall = cwe.geo_v15_side(0.6 * h, w, h)
    try:
        cwe.GEO_CAM_H_M = 1.7
        short = cwe.geo_v15_side(0.6 * h, w, h)
    finally:
        cwe.GEO_CAM_H_M = 2.5
    assert short == pytest.approx(tall * 2.5 / 1.7, rel=1e-9)
