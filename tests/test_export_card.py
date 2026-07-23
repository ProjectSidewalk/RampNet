"""Guards for the model-card metrics rendering in scripts/export_hf_model.py.

The load-bearing correctness rule: Average Precision on the card must come from a
full confidence sweep (evaluate.py with PEAK_THRESHOLD_ABS=0.0), while precision
and recall come from the run at the operating threshold. AP taken from a
threshold-truncated run integrates only the tail of the PR curve and reads far
too low (e.g. 0.86 instead of 0.92 for the released checkpoint).
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from export_hf_model import render_eval_section  # noqa: E402

# Mirrors the committed stage_two/evaluation_results_new/ files.
THRESHOLD_METRICS = {
    "ap": 0.8615, "total_gt_points": 3919, "peak_threshold_abs": 0.55,
    "precision_at_threshold": 0.9492, "recall_at_threshold": 0.8727,
    "radius_threshold_normalized": 0.022, "tta": True,
}
FULL_SWEEP_METRICS = {
    "ap": 0.9205, "total_gt_points": 3919, "peak_threshold_abs": 0.0,
    "precision_at_threshold": 0.0322, "recall_at_threshold": 0.9408,
    "radius_threshold_normalized": 0.022, "tta": True,
}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_ap_from_full_sweep_pr_from_threshold(tmp_path):
    threshold = _write(tmp_path, "pt55.json", THRESHOLD_METRICS)
    full = _write(tmp_path, "pt0.json", FULL_SWEEP_METRICS)
    section = render_eval_section(threshold, full)
    # AP is the full-sweep 0.9205, not the truncated 0.8615.
    assert "0.9205" in section
    assert "0.8615" not in section
    # P/R are read at the operating threshold.
    assert "0.9492" in section and "0.8727" in section
    assert "threshold 0.55" in section


def test_rejects_truncated_ap_source(tmp_path):
    threshold = _write(tmp_path, "pt55.json", THRESHOLD_METRICS)
    # Passing the threshold file as the AP source is the exact mistake to catch.
    with pytest.raises(ValueError, match="full-sweep"):
        render_eval_section(threshold, threshold)


def test_ap_falls_back_to_metrics_when_it_is_full_sweep(tmp_path):
    full = _write(tmp_path, "pt0.json", FULL_SWEEP_METRICS)
    section = render_eval_section(full)  # no separate ap-json
    assert "0.9205" in section
