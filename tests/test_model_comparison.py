"""Guards for the model-comparison harness (scripts/model_comparison/).

Covers the pure box->point parsing and that the VLM detectors are safe scaffolds:
they construct without their (absent) client libraries, and only fail — with a
clear message — when a live detection is actually requested.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from detectors import (  # noqa: E402
    BundleRampNetDetector, GeminiDetector, QwenDetector, PanoSample,
    gemini_boxes_to_points, qwen_boxes_to_points,
)


def test_gemini_boxes_to_points_center_and_normalization():
    # box_2d = [ymin, xmin, ymax, xmax] scaled 0-1000 -> normalized center.
    pts = gemini_boxes_to_points([{"box_2d": [400, 200, 600, 400], "label": "curb ramp"}])
    assert pts == [(0.3, 0.5, None)]   # cx=(200+400)/2/1000, cy=(400+600)/2/1000


def test_qwen_boxes_to_points_normalizes_by_image_size():
    # bbox_2d = [x1, y1, x2, y2] in pixels of the image shown to the model.
    pts = qwen_boxes_to_points([{"bbox_2d": [100, 200, 300, 400]}], img_w=1000, img_h=2000)
    assert pts == [(0.2, 0.15, None)]  # cx=200/1000, cy=300/2000


def test_bundle_rampnet_detector_reads_records():
    records = {"p1": {"detections": [
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}]}}
    det = BundleRampNetDetector(records)
    sample = PanoSample("p1", image_path=None, width=None, height=None, meta={})
    assert det.detect(sample) == [(0.5, 0.5, 0.9)]


def test_vlm_detectors_construct_without_client_libs():
    # Constructing must not import google-genai / qwen; that only happens on detect().
    GeminiDetector(model_id="gemini-flash-latest")
    QwenDetector(model_id="Qwen/Qwen3-VL")


def test_gemini_detect_fails_clearly_without_key_or_lib():
    det = GeminiDetector(model_id="gemini-flash-latest", api_key=None)
    sample = PanoSample("p1", image_path="nope.jpg", width=100, height=100, meta={})
    try:
        det.detect(sample)
    except (ImportError, RuntimeError, NotImplementedError):
        return  # any of these is an acceptable, clear failure for a scaffold
    raise AssertionError("expected GeminiDetector.detect to fail loudly without lib/key")
