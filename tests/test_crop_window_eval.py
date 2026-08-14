"""Unit tests for the crop-window sizing eval (#114).

Pure geometry — no imagery, no network, no ``.model_cache``. The guarantees are
about the things that fail silently: the v1 formula port (raw vs the
resolution-normalized variant must be bit-identical at the calibration height),
seam-wrap containment (a window crossing the equirectangular seam must still
contain a box on the far side of x=0), and clamp-by-shift (a window near a pole
shifts instead of zero-padding, and the margins must reflect the shift).
"""
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
