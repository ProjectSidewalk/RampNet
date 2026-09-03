"""Guards for the model-comparison harness (scripts/model_comparison/).

Covers the pure box->point parsing (both providers' box conventions), that the VLM
detectors construct without their client libraries and only fail — with a clear
message — when a live detection is actually requested, and that the detection
cache stays valid across changes (see ``test_gemini_cache_key_is_frozen``).
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

import detectors  # noqa: E402
from detectors import (  # noqa: E402
    BundleRampNetDetector, ClaudeDetector, GeminiDetector, GroundingDinoDetector,
    MolmoDetector, OwlV2Detector, QwenDetector, YoloDetector, PanoSample, _VLMDetector,
    claude_boxes_to_points, boxes_from_claude_response,
    gemini_boxes_to_points, qwen_boxes_to_points, boxes_from_gemini_response,
    boxes_from_qwen_text, infer_qwen_coord_space, parse_model_spec, build_detector,
    molmo_points_from_text, molmo_token_points_to_items, infer_molmo_mode,
    owlv2_target_size, pixel_boxes_to_points, zero_shot_results_to_boxes,
    yolo_results_to_boxes,
    CURB_RAMP_DEFINITION, DETECTION_PROMPT, GDINO_QUERY, MOLMO_PROMPT, OWLV2_QUERY,
)
from dump_detections import detections_to_view_shapes  # noqa: E402


import compare  # noqa: E402
from compare import (  # noqa: E402
    score_model, validate_bundle, validate_manual_bundle, DetectionCache, cache_key,
    ground_truths_from_verdicts, has_confidences, load_bundle,
    load_manual_ground_truths, operating_report, report_usage, rescore, sweep_rows,
    DEFAULT_USAGE_LOG,
)
import pricing  # noqa: E402
from pricing import estimate_cost, price_for  # noqa: E402
from rampnet.detection_eval import GroundTruth, radius_sq_for  # noqa: E402
from prepare_yolo_dataset import (  # noqa: E402
    parse_box_size, _ground_distance_m, _box_wh, _resolve_distances, write_data_yaml,
    Config as YoloPrepConfig,
)


def _prep_cfg(**over):
    base = dict(geometry="tiles", strategy="fixed", fixed_frac=0.03, min_frac=0.008,
                max_frac=0.12, ramp_size_m=1.5, camera_height_m=2.5, source_max_edge=4096,
                pano_w=2048, pano_h=1024, views=(), out="", overlay_dir=None, bg_keep_frac=1.0)
    base.update(over)
    return YoloPrepConfig(**base)


class _Args:
    gemini_model = "gemini-3.6-flash"
    qwen_model = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_coord_space = "auto"
    owlv2_model = "google/owlv2-large-patch14-ensemble"
    gdino_model = "IDEA-Research/grounding-dino-base"
    molmo_model = "allenai/Molmo2-8B"
    claude_model = "claude-sonnet-5"
    claude_effort = "low"
    claude_tool_choice = "auto"
    owlv2_query = None
    gdino_query = None
    gdino_text_threshold = None
    score_threshold = None
    molmo_coord_scale = "auto"
    yolo_model = None
    yolo_conf = 0.05
    yolo_iou = 0.5
    yolo_imgsz = 1024
    tiling = "perspective"


class _FlakyDetector:
    """Succeeds on every pano except ones whose id starts with 'bad'."""
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def prepare(self):
        pass

    def signature(self):
        return None  # disables caching so detect() is always exercised

    def detect(self, sample):
        self.calls += 1
        if sample.pano_id.startswith("bad"):
            raise RuntimeError("simulated transient API failure")
        return []


class _FakeBox:
    def __init__(self, box_2d, label):
        self.box_2d = box_2d
        self.label = label


class _FakeResp:
    def __init__(self, parsed=None, text=None):
        self.parsed = parsed
        self.text = text


def test_gemini_boxes_to_points_center_and_normalization():
    # box_2d = [ymin, xmin, ymax, xmax] scaled 0-1000 -> normalized center.
    pts = gemini_boxes_to_points([{"box_2d": [400, 200, 600, 400], "label": "curb ramp"}])
    assert pts == [(0.3, 0.5, None)]   # cx=(200+400)/2/1000, cy=(400+600)/2/1000


def test_qwen_boxes_to_points_pixels_normalizes_by_image_size():
    # Qwen2/2.5-VL: bbox_2d = [x1, y1, x2, y2] in pixels of the image shown to the model.
    pts = qwen_boxes_to_points([{"bbox_2d": [100, 200, 300, 400]}], img_w=1000, img_h=2000,
                               coord_space="pixels")
    assert pts == [(0.2, 0.15, None)]  # cx=200/1000, cy=300/2000


def test_qwen_boxes_to_points_norm1000_ignores_image_size():
    # Qwen3-VL (the default): bbox_2d is already normalized 0-1000, so the center is
    # /1000 regardless of the view size the processor was handed.
    boxes = [{"bbox_2d": [100, 200, 300, 400]}]
    pts = qwen_boxes_to_points(boxes, img_w=1024, img_h=1024, coord_space="norm1000")
    assert pts == [(0.2, 0.3, None)]
    assert qwen_boxes_to_points(boxes, 640, 480, coord_space="norm1000") == pts


def test_qwen_boxes_to_points_rejects_unknown_coord_space():
    try:
        qwen_boxes_to_points([], 100, 100, coord_space="normalized")
    except ValueError:
        return
    raise AssertionError("expected an unknown coord_space to raise")


def test_infer_qwen_coord_space_by_model_id():
    assert infer_qwen_coord_space("Qwen/Qwen3-VL-8B-Instruct") == "norm1000"
    assert infer_qwen_coord_space("Qwen/Qwen3-VL-32B-Instruct-FP8") == "norm1000"
    assert infer_qwen_coord_space("Qwen/Qwen2.5-VL-7B-Instruct") == "pixels"
    assert infer_qwen_coord_space("some-future-qwen") == "norm1000"  # newest convention


# --- Qwen completion parsing (an open model has no response_schema) ----------

def test_boxes_from_qwen_text_plain_json():
    assert boxes_from_qwen_text('[{"bbox_2d": [1, 2, 3, 4], "label": "curb ramp"}]') == [
        {"bbox_2d": [1.0, 2.0, 3.0, 4.0], "label": "curb ramp"}]


def test_boxes_from_qwen_text_strips_code_fence_and_prose():
    text = 'Sure! Here are the ramps:\n```json\n[{"bbox_2d": [5, 6, 7, 8], "label": "x"}]\n```\nDone.'
    assert boxes_from_qwen_text(text) == [{"bbox_2d": [5.0, 6.0, 7.0, 8.0], "label": "x"}]


def test_boxes_from_qwen_text_accepts_bare_object_and_bbox_alias():
    assert boxes_from_qwen_text('{"bbox": [1, 2, 3, 4]}') == [
        {"bbox_2d": [1.0, 2.0, 3.0, 4.0], "label": ""}]


def test_boxes_from_qwen_text_drops_malformed_items():
    text = '[{"bbox_2d": [1, 2, 3]}, "junk", {"label": "no box"}, {"bbox_2d": [1, 2, 3, 4]}]'
    assert boxes_from_qwen_text(text) == [{"bbox_2d": [1.0, 2.0, 3.0, 4.0], "label": ""}]


def test_boxes_from_qwen_text_empty_and_unparseable():
    assert boxes_from_qwen_text("") == []
    assert boxes_from_qwen_text("No curb ramps are visible in this image.") == []
    assert boxes_from_qwen_text("[{unclosed") == []
    assert boxes_from_qwen_text("[]") == []


def test_boxes_from_gemini_response_parsed_objects():
    resp = _FakeResp(parsed=[_FakeBox([400, 200, 600, 400], "curb ramp")])
    assert boxes_from_gemini_response(resp) == [{"box_2d": [400, 200, 600, 400], "label": "curb ramp"}]


def test_boxes_from_gemini_response_json_text_fallback():
    resp = _FakeResp(parsed=None, text='[{"box_2d": [1, 2, 3, 4], "label": "x"}]')
    assert boxes_from_gemini_response(resp) == [{"box_2d": [1, 2, 3, 4], "label": "x"}]


def test_boxes_from_gemini_response_empty():
    assert boxes_from_gemini_response(_FakeResp(parsed=None, text=None)) == []
    assert boxes_from_gemini_response(_FakeResp(parsed=[], text="[]")) == []


# --- open-vocabulary detectors (OWLv2 / Grounding DINO) ---------------------

class _FakeTensor:
    """Stands in for the torch tensors a live post_process returns."""
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_zero_shot_results_to_boxes_unwraps_tensors_and_scores():
    result = {"boxes": _FakeTensor([[10, 20, 30, 40], [1, 2, 3, 4]]),
              "scores": _FakeTensor([0.9, 0.11]),
              "text_labels": ["curb ramp", "curb ramp"]}
    assert zero_shot_results_to_boxes(result) == [
        {"box": [10.0, 20.0, 30.0, 40.0], "score": 0.9, "label": "curb ramp"},
        {"box": [1.0, 2.0, 3.0, 4.0], "score": 0.11, "label": "curb ramp"}]


def test_zero_shot_results_to_boxes_filters_and_survives_missing_fields():
    result = {"boxes": [[0, 0, 2, 2], [0, 0, 4, 4]], "scores": [0.5, 0.05], "labels": [0, 0]}
    kept = zero_shot_results_to_boxes(result, threshold=0.2)
    assert [it["box"] for it in kept] == [[0.0, 0.0, 2.0, 2.0]]
    assert zero_shot_results_to_boxes({}) == []                      # nothing detected
    assert zero_shot_results_to_boxes({"boxes": [[1, 2, 3]]}) == []   # malformed box dropped


def test_owlv2_target_size_is_the_padded_square():
    # OWLv2's processor pads to a square (bottom/right) before resizing, so boxes are
    # relative to that square, not to the image — which is the frame
    # pixel_boxes_to_points normalizes against.
    assert owlv2_target_size(1024, 1024) == (1024, 1024)   # square view: a no-op
    assert owlv2_target_size(2048, 1024) == (2048, 2048)   # whole-pano 2:1
    assert owlv2_target_size(600, 900) == (900, 900)


def test_pixel_boxes_to_points_carries_confidence_through():
    # The whole point of these models: the score survives to the scorer, which is
    # what makes AP / PR curves / a threshold sweep possible.
    pts = pixel_boxes_to_points([{"box": [100, 200, 300, 400], "score": 0.42}], 1000, 2000)
    assert pts == [(0.2, 0.15, 0.42)]


def test_pixel_boxes_to_points_drops_boxes_in_the_pad_region():
    # With the padded-square target size, a box OWLv2 places below a wide image has a
    # center outside the picture; it is not a detection.
    items = [{"box": [0, 0, 100, 100], "score": 0.5},        # in frame
             {"box": [0, 1200, 100, 1400], "score": 0.5}]    # below a 1000x600 image
    assert pixel_boxes_to_points(items, 1000, 600) == [(0.05, 1 / 12, 0.5)]


class _FakeYoloBoxes:
    """Stands in for ultralytics ``Results.boxes``: .xyxy / .conf as tensors (here,
    objects with .tolist()) so yolo_results_to_boxes exercises the tensor path."""
    def __init__(self, xyxy, conf):
        self.xyxy = _Tolistable(xyxy)
        self.conf = _Tolistable(conf)


class _Tolistable:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeYoloResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeYoloBoxes(xyxy, conf) if xyxy is not None else None


def test_yolo_results_to_boxes_carries_confidence_through_the_tensor_path():
    # Like the open-vocab detectors, YOLO's per-box score must survive to the scorer
    # (that is what earns it AP / a PR curve / a sweep). xyxy are absolute pixels.
    res = _FakeYoloResult([[100.0, 200.0, 300.0, 400.0]], [0.42])
    boxes = yolo_results_to_boxes(res)
    assert boxes == [{"box": [100.0, 200.0, 300.0, 400.0], "score": 0.42}]
    # And it reduces to the same normalized center the open detectors produce.
    assert pixel_boxes_to_points(boxes, 1000, 2000) == [(0.2, 0.15, 0.42)]


def test_yolo_results_to_boxes_filters_and_survives_no_detections():
    res = _FakeYoloResult([[0, 0, 10, 10], [5, 5, 15, 15]], [0.1, 0.9])
    assert yolo_results_to_boxes(res, threshold=0.5) == [{"box": [5.0, 5.0, 15.0, 15.0],
                                                          "score": 0.9}]
    assert yolo_results_to_boxes(_FakeYoloResult(None, None)) == []


def test_open_vocab_queries_are_short_and_key_the_cache():
    # These are not chat models: the query is the prompt, and it must be in the
    # signature so changing it doesn't silently reuse detections from the old one.
    owl, dino = OwlV2Detector(), GroundingDinoDetector()
    assert owl.prompt == OWLV2_QUERY and dino.prompt == GDINO_QUERY
    assert owl.signature()["query"] == OWLV2_QUERY
    assert owl.signature()["prompt"] == OWLV2_QUERY
    assert dino.signature()["text_threshold"] == dino.text_threshold
    assert OwlV2Detector(query="curb cut").signature()["query"] == "curb cut"


def test_score_threshold_is_in_the_signature():
    # It is a cache FLOOR: lowering it must invalidate cached detections (there are
    # boxes missing from them), while raising the reported operating point is free.
    low = OwlV2Detector(score_threshold=0.01).signature()
    high = OwlV2Detector(score_threshold=0.3).signature()
    assert low["score_threshold"] == 0.01 and high["score_threshold"] == 0.3
    assert cache_key("owlv2", low, "richmond", "p") != cache_key("owlv2", high, "richmond", "p")


# --- Molmo (points, not boxes) ----------------------------------------------

def test_molmo_prompt_shares_the_definition_but_asks_for_points():
    assert CURB_RAMP_DEFINITION in DETECTION_PROMPT      # one definition, every model
    assert CURB_RAMP_DEFINITION in MOLMO_PROMPT
    assert MOLMO_PROMPT.startswith("Point to every curb ramp")
    assert "bounding box" not in MOLMO_PROMPT


def test_molmo_v1_attribute_points_are_percentages():
    text = '<point x="35.4" y="61.2" alt="curb ramp">curb ramp</point>'
    assert molmo_points_from_text(text) == [{"point": [0.354, 0.612], "label": "curb ramp"}]


def test_molmo_v1_multi_point_tag():
    text = ('<points x1="10.0" y1="20.0" x2="30.0" y2="40.0" alt="curb ramps">'
            'curb ramps</points>')
    assert molmo_points_from_text(text) == [
        {"point": [0.1, 0.2], "label": "curb ramps"},
        {"point": [0.3, 0.4], "label": "curb ramps"}]


def test_molmo_v2_coords_are_image_index_then_id_x_y_triplets():
    # VERBATIM output from allenai/Molmo2-8B on a richmond view (2026-07-23).
    # The leading "1" is the IMAGE index, not a point id; consuming it as one
    # shifts every coordinate a slot left and pins all points to x~0, which is the
    # bug the dump_detections overlay caught on the first real run.
    text = '<points coords="1 1 308 305 2 752 377">curb ramp</points>'
    assert molmo_points_from_text(text) == [
        {"point": [0.308, 0.305], "label": ""},
        {"point": [0.752, 0.377], "label": ""}]


def test_molmo_v2_four_points_from_a_real_response():
    text = ('<points coords="1 1 299 338 2 532 381 3 662 446 4 932 429">'
            'curb ramp</points>')
    assert [p["point"] for p in molmo_points_from_text(text)] == [
        [0.299, 0.338], [0.532, 0.381], [0.662, 0.446], [0.932, 0.429]]


def test_molmo_v2_keeps_points_near_the_left_and_top_edges():
    # The model card's own regex demands 3-4 digits for x/y, which silently drops
    # anything in the leftmost/topmost 10% of a view. Positional chunking doesn't.
    text = '<points coords="1 1 42 7 2 500 500">curb ramp</points>'
    assert [p["point"] for p in molmo_points_from_text(text)] == [
        [0.042, 0.007], [0.5, 0.5]]


def test_molmo_v2_survives_a_generation_truncated_mid_triplet():
    # A completion cut off by max_new_tokens can end mid-triplet, shifting the
    # token count so it mimics the other index convention. The id column
    # (1, 2, 3, ...) decides the alignment, so the complete points still parse
    # instead of every pair sliding into in-frame garbage pinned near x~0 —
    # the same quiet failure the image-index fix was about.
    with_index = '<points coords="1 1 308 305 2 752"/>'       # lost y2
    assert [p["point"] for p in molmo_points_from_text(with_index)] == [[0.308, 0.305]]
    without_index = '<points coords="1 354 612 2"/>'          # lost x2 y2
    assert [p["point"] for p in molmo_points_from_text(without_index)] == [[0.354, 0.612]]


def test_molmo_v2_accepts_separators_and_a_bare_triplet_list():
    assert [p["point"] for p in molmo_points_from_text(
        '<points coords="1 1 354 612; 2 700 480"/>')] == [[0.354, 0.612], [0.7, 0.48]]
    # No leading index (token count already a multiple of 3).
    assert [p["point"] for p in molmo_points_from_text(
        '<points coords="1 354 612"/>')] == [[0.354, 0.612]]


def test_molmo_explicit_scale_overrides_the_syntax_inference():
    text = '<point x="354" y="612">x</point>'          # 0-1000 numbers in v1 syntax
    assert molmo_points_from_text(text) == []          # /100 -> out of frame, dropped
    assert molmo_points_from_text(text, coord_scale=1000.0) == [
        {"point": [0.354, 0.612], "label": ""}]


def test_molmo_points_from_text_ignores_prose_and_empty():
    assert molmo_points_from_text("There are no curb ramps in this image.") == []
    assert molmo_points_from_text("") == []
    assert molmo_points_from_text(None) == []


def test_molmo_token_points_read_the_tail_of_each_row():
    # The card documents the leading ids two different ways, so only (x, y) is read.
    rows = [[0, 0, 512.0, 256.0], [1, 0, 2048.0, 10.0]]   # second is out of frame
    assert molmo_token_points_to_items(rows, 1024, 512) == [{"point": [0.5, 0.5], "label": ""}]


def test_infer_molmo_mode():
    assert infer_molmo_mode("allenai/MolmoPoint-8B") == "point_tokens"
    assert infer_molmo_mode("allenai/Molmo2-8B") == "xml"
    assert infer_molmo_mode("allenai/Molmo-7B-D-0924") == "xml"


def test_molmo_signature_extends_without_disturbing_gemini():
    sig = MolmoDetector(model_id="allenai/MolmoPoint-8B").signature()
    assert sig["mode"] == "point_tokens" and sig["coord_scale"] is None
    assert set(sig) - set(GeminiDetector().signature()) == {
        "coord_scale", "mode", "max_new_tokens"}


# --- visual QA shapes (dump_detections) -------------------------------------

def test_detections_to_view_shapes_covers_every_provider_format():
    qwen = QwenDetector(model_id="Qwen/Qwen3-VL-8B-Instruct")
    assert detections_to_view_shapes(None, [{"box_2d": [100, 200, 300, 400]}], 1000, 1000) == [
        ("rect", 200.0, 100.0, 400.0, 300.0, None)]           # Gemini: ymin,xmin,ymax,xmax
    assert detections_to_view_shapes(qwen, [{"bbox_2d": [100, 200, 300, 400]}], 1000, 1000) == [
        ("rect", 100.0, 200.0, 300.0, 400.0, None)]           # Qwen norm1000 at a 1000px view
    assert detections_to_view_shapes(None, [{"box": [1, 2, 3, 4], "score": 0.7}], 100, 100) == [
        ("rect", 1, 2, 3, 4, 0.7)]                            # OWLv2/GDINO: pixels + score
    assert detections_to_view_shapes(None, [{"point": [0.25, 0.5]}], 800, 600) == [
        ("point", 200.0, 300.0, None)]                        # Molmo


def test_parse_model_spec():
    assert parse_model_spec("rampnet") == ("rampnet", None)
    assert parse_model_spec("gemini") == ("gemini", None)
    assert parse_model_spec("gemini:gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert parse_model_spec(" qwen : Qwen/Qwen3-VL-8B-Instruct ") == (
        "qwen", "Qwen/Qwen3-VL-8B-Instruct")


def test_build_detector_labels_variants_by_model_id():
    label, det = build_detector("rampnet", None, {}, _Args())
    assert label == "rampnet" and isinstance(det, BundleRampNetDetector)
    # A pinned variant labels by its model id, so 2.5 and 3.6 are distinct rows.
    label, det = build_detector("gemini", "gemini-2.5-flash", {}, _Args())
    assert label == "gemini-2.5-flash" and det.model_id == "gemini-2.5-flash"
    # Bare provider falls back to the args default.
    label, det = build_detector("gemini", None, {}, _Args())
    assert label == "gemini-3.6-flash"


def test_detection_cache_roundtrip(tmp_path):
    c = DetectionCache(str(tmp_path))
    k = cache_key("gemini-3.6-flash", {"tile": True}, "richmond", "pano1")
    assert c.get(k) is None
    c.put(k, [(0.1, 0.2, None), (0.3, 0.4, 0.9)])
    assert c.get(k) == [[0.1, 0.2, None], [0.3, 0.4, 0.9]]


def test_detection_cache_disabled_is_noop(tmp_path):
    c = DetectionCache(str(tmp_path), enabled=False)
    k = cache_key("m", {}, "c", "p")
    c.put(k, [(1, 2, 3)])
    assert c.get(k) is None


def test_cache_key_sensitive_and_stable():
    assert cache_key("m1", {"x": 1}, "c", "p") != cache_key("m2", {"x": 1}, "c", "p")
    assert cache_key("m", {"x": 1}, "c", "p1") != cache_key("m", {"x": 1}, "c", "p2")
    assert cache_key("m", {"x": 1}, "c", "p") == cache_key("m", {"x": 1}, "c", "p")


def test_gemini_cache_key_is_frozen():
    """Regression guard on real spend.

    The on-disk cache holds thousands of already-paid Gemini detections keyed by
    hash(label, signature, city, pano). Any drift in GeminiDetector.signature() —
    a reworded prompt, a new key, a changed default — silently misses every one of
    them and re-bills the whole run. If this fails, the change was not free: either
    revert it or accept re-paying deliberately.
    """
    det = GeminiDetector(model_id="gemini-3.6-flash")
    assert det.prompt == DETECTION_PROMPT      # provider-specific suffixes must not leak in
    assert cache_key("gemini-3.6-flash", det.signature(), "richmond", "pano1") == (
        "b4401afce834fee6bba27f9d1fbec67e86e570dd")


def test_qwen_signature_extends_without_disturbing_gemini():
    qwen = QwenDetector(model_id="Qwen/Qwen3-VL-8B-Instruct")
    gem = GeminiDetector(model_id="gemini-3.6-flash")
    sig = qwen.signature()
    assert sig["coord_space"] == "norm1000" and sig["max_new_tokens"] == 1024
    assert sig["prompt"].startswith(DETECTION_PROMPT) and sig["prompt"] != DETECTION_PROMPT
    # The extra keys live only on Qwen's signature.
    assert set(sig) - set(gem.signature()) == {"coord_space", "max_new_tokens"}


def test_build_detector_qwen_coord_space_override():
    class _Pinned(_Args):
        qwen_coord_space = "pixels"
    _, det = build_detector("qwen", "Qwen/Qwen3-VL-8B-Instruct", {}, _Pinned())
    assert det.coord_space == "pixels"          # explicit flag beats id inference
    _, det = build_detector("qwen", None, {}, _Args())
    assert det.model_id == "Qwen/Qwen3-VL-8B-Instruct" and det.coord_space == "norm1000"


def test_build_detector_wires_the_open_models():
    for token, cls, default_id in (
            ("owlv2", OwlV2Detector, "google/owlv2-large-patch14-ensemble"),
            ("gdino", GroundingDinoDetector, "IDEA-Research/grounding-dino-base"),
            ("molmo", MolmoDetector, "allenai/Molmo2-8B")):
        label, det = build_detector(token, None, {}, _Args())
        assert label == default_id and isinstance(det, cls) and det.model_id == default_id
    # A pinned variant labels by its model id, like the other providers.
    label, det = build_detector("molmo", "allenai/MolmoPoint-8B", {}, _Args())
    assert label == "allenai/MolmoPoint-8B" and det.mode == "point_tokens"


def test_build_detector_applies_query_and_threshold_overrides():
    class _Pinned(_Args):
        owlv2_query = "curb cut"
        gdino_query = "wheelchair ramp."
        gdino_text_threshold = 0.35
        score_threshold = 0.2
        molmo_coord_scale = "1000"
    _, owl = build_detector("owlv2", None, {}, _Pinned())
    assert owl.query == "curb cut" and owl.score_threshold == 0.2
    _, dino = build_detector("gdino", None, {}, _Pinned())
    assert dino.query == "wheelchair ramp." and dino.text_threshold == 0.35
    _, molmo = build_detector("molmo", None, {}, _Pinned())
    assert molmo.coord_scale == 1000.0


def test_build_detector_rejects_unknown_provider():
    try:
        build_detector("clip", None, {}, _Args())
    except ValueError as e:
        assert "owlv2" in str(e)      # the message lists what is available
        assert "yolo" in str(e)       # ...including the supervised baseline
        return
    raise AssertionError("expected an unknown provider to raise")


def test_build_detector_wires_yolo_and_labels_by_weights_stem(tmp_path):
    weights = tmp_path / "yolo11l_tiles.pt"
    weights.write_bytes(b"fake-weights")

    class _Y(_Args):
        yolo_model = str(weights)
    # Bare provider uses --yolo-model; the table label is the file STEM (not the
    # absolute path), so the cache is machine-independent.
    label, det = build_detector("yolo", None, {}, _Y())
    assert label == "yolo11l_tiles" and isinstance(det, YoloDetector)
    assert det.weights == str(weights) and det.model_id == "yolo11l_tiles"
    assert det.conf == 0.05 and det.score_threshold == 0.05 and det.tile is True
    # A pinned path via yolo:<path> works too and labels the same way.
    label2, det2 = build_detector("yolo", str(weights), {}, _Y())
    assert label2 == "yolo11l_tiles"


def test_build_detector_yolo_requires_weights():
    class _NoWeights(_Args):
        yolo_model = None
    try:
        build_detector("yolo", None, {}, _NoWeights())
    except ValueError as e:
        assert "weights" in str(e).lower()
        return
    raise AssertionError("expected yolo without weights to raise")


def test_yolo_signature_hashes_weights_and_is_machine_independent(tmp_path):
    w1 = tmp_path / "a.pt"
    w1.write_bytes(b"weights-one")
    w2 = tmp_path / "b.pt"
    w2.write_bytes(b"weights-two")
    # Same label, different weights CONTENT -> different cache identity, so a
    # re-trained checkpoint can't silently reuse the old detections.
    d1 = YoloDetector(weights=str(w1), label="m")
    d2 = YoloDetector(weights=str(w2), label="m")
    assert d1.signature()["weights_hash"] != d2.signature()["weights_hash"]
    assert cache_key("m", d1.signature(), "c", "p") != cache_key("m", d2.signature(), "c", "p")
    # Identity is the label, not a machine-specific path.
    assert d1.signature()["model_id"] == "m"
    assert d1.score_threshold == d1.conf == 0.05
    # The extra keys live only on YOLO's signature (Gemini's is undisturbed).
    extra = set(d1.signature()) - set(GeminiDetector(model_id="gemini-3.6-flash").signature())
    assert extra == {"weights_hash", "conf", "iou", "imgsz"}
    # Absent weights -> None hash (score a rsynced cache without the .pt present).
    w1.unlink()
    assert YoloDetector(weights=str(w1), label="m").signature()["weights_hash"] is None


# --- prepare_yolo_dataset.py: box-size + gps-correspondence logic ------------

def test_parse_box_size_variants():
    assert parse_box_size("fixed") == ("fixed", 0.03)
    assert parse_box_size("fixed:0.05") == ("fixed", 0.05)
    assert parse_box_size("pitch") == ("pitch", 0.0)
    assert parse_box_size("gps") == ("gps", 0.0)
    try:
        parse_box_size("bogus")
    except Exception:
        return
    raise AssertionError("expected an invalid box-size to raise")


def test_ground_distance_is_monotonic_and_diverges_at_the_horizon():
    # Lower in the frame (larger y) -> steeper down-look -> closer ground point.
    near = _ground_distance_m(0.95, 2.5)
    far = _ground_distance_m(0.55, 2.5)
    assert 0 < near < far
    # At / above the horizon the flat-ground model diverges -> inf (the min box).
    assert _ground_distance_m(0.5, 2.5) == float("inf")
    assert _ground_distance_m(0.3, 2.5) == float("inf")


def test_box_wh_fixed_ignores_distance():
    cfg = _prep_cfg(strategy="fixed", fixed_frac=0.04)
    assert _box_wh(cfg, 0.5, 0.9, 3.0, 90.0, 90.0) == (0.04, 0.04)
    assert _box_wh(cfg, 0.5, 0.6, None, 90.0, 90.0) == (0.04, 0.04)


def test_box_wh_distance_aware_shrinks_with_distance_and_clamps():
    cfg = _prep_cfg(strategy="gps")
    near_w, _ = _box_wh(cfg, 0.5, 0.8, 3.0, 90.0, 90.0)
    far_w, _ = _box_wh(cfg, 0.5, 0.8, 30.0, 90.0, 90.0)
    assert far_w < near_w                                  # farther ramp -> smaller box
    assert cfg.min_frac <= far_w <= cfg.max_frac
    assert cfg.min_frac <= near_w <= cfg.max_frac
    # A non-finite distance collapses to the min box, never a crash.
    assert _box_wh(cfg, 0.5, 0.8, float("inf"), 90.0, 90.0) == (cfg.min_frac, cfg.min_frac)


def test_resolve_distances_gps_needs_matching_point_and_coord_counts():
    cfg = _prep_cfg(strategy="gps")
    pts = [(0.1, 0.7), (0.2, 0.8)]
    dists, used = _resolve_distances(cfg, pts, [47.6, -122.3],
                                     [[47.6001, -122.3001], [47.6002, -122.3002]])
    assert used and all(d is not None and d > 0 for d in dists)
    # Count mismatch -> fall back (None triggers the pitch model downstream).
    dists2, used2 = _resolve_distances(cfg, pts, [47.6, -122.3], [[47.6001, -122.3001]])
    assert used2 is False and dists2 == [None, None]


def test_write_data_yaml_stamps_prep_provenance(tmp_path):
    # Box size / bg-keep are train-only knobs that NO downstream record captures
    # (ultralytics args.yaml never sees them) — data.yaml is where the dataset's
    # provenance must live, as comments the YAML parser ignores.
    class _P:
        box_size = ("pitch", 0.0)
        n_yaw = 6
        view_fov = 90.0
        view_pitch = -30.0
        view_size = 1024
        pano_width = 2048
        subset = None
        dataset_root = "dataset"
    cfg = _prep_cfg(strategy="pitch", ramp_size_m=1.8, bg_keep_frac=0.15, out=str(tmp_path))
    write_data_yaml(cfg, _P())
    text = (tmp_path / "data.yaml").read_text()
    assert "box-size=pitch" in text and "ramp-size-m=1.8" in text
    assert "bg-keep-frac=0.15" in text and "view-size=1024" in text
    # The stamp is comments only — the mapping ultralytics reads is untouched.
    data_lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert data_lines[0] == f"path: {tmp_path}"
    assert "train: images/train" in data_lines and "val: images/val" in data_lines


def test_open_model_detectors_construct_without_weights():
    for det in (OwlV2Detector(), GroundingDinoDetector(), MolmoDetector()):
        assert det._model is None and det._processor is None


# --- threshold sweep / re-scoring from the cache ----------------------------

def _one_pano(preds):
    """One pano with a single GT ramp at the origin, in unit coordinates."""
    return [(preds, GroundTruth([(0.0, 0.0)], [], True))]


def test_rescore_drops_predictions_below_the_threshold():
    scored = _one_pano([(0.0, 0.0, 0.9), (0.5, 0.5, 0.1)])   # one TP, one FP
    lo = rescore(scored, radius_sq_for(), 0.0)
    assert (lo.tp, lo.fp, lo.precision, lo.recall) == (1, 1, 0.5, 1.0)
    hi = rescore(scored, radius_sq_for(), 0.5)                        # the FP is gone
    assert (hi.tp, hi.fp, hi.precision, hi.recall) == (1, 0, 1.0, 1.0)


def test_rescore_never_drops_unscored_predictions():
    # A chat VLM has nothing to threshold on; filtering it would silently empty it.
    scored = _one_pano([(0.0, 0.0, None)])
    assert rescore(scored, radius_sq_for(), 0.9).tp == 1


def test_operating_report_keeps_ap_and_pr_curve_untruncated():
    # The table's P/R/F1/counts move to the operating point, but AP and the PR
    # curve are integrals over the whole confidence range — an --op-threshold must
    # not truncate them (the manual_gold bundle exports down to a 0.05 floor
    # precisely so RampNet's AP is untruncated).
    scored = _one_pano([(0.0, 0.0, 0.9), (0.5, 0.5, 0.1)])   # one TP, one low-conf FP
    full = rescore(scored, radius_sq_for(), 0.0)
    op = operating_report(full, scored, radius_sq_for(), 0.5)
    assert (op.tp, op.fp) == (1, 0)            # operating point applied to the counts
    assert op.ap == full.ap                    # ...but AP is the full-range AP
    assert op.pr_curve == full.pr_curve
    assert operating_report(full, scored, radius_sq_for(), 0.0) is full


def test_has_confidences_requires_every_prediction_to_carry_one():
    assert has_confidences(_one_pano([(0.0, 0.0, 0.9)]))
    assert not has_confidences(_one_pano([(0.0, 0.0, 0.9), (0.5, 0.5, None)]))
    assert not has_confidences(_one_pano([]))     # nothing detected: no curve to draw


def test_sweep_rows_stop_at_the_highest_score_present():
    rows = sweep_rows(_one_pano([(0.0, 0.0, 0.22)]), radius_sq_for())
    assert [t for t, _ in rows] == [0.05, 0.1, 0.15, 0.2]   # 0.25+ would be all-empty
    assert all(r.tp == 1 for _, r in rows)


def test_sweep_rows_drop_thresholds_below_the_cache_floor():
    # With a raised --score-threshold the cache holds nothing below the floor, so
    # sweep rows down there would silently repeat the floor row while reading as
    # real measurements. They must be dropped, not printed.
    scored = _one_pano([(0.0, 0.0, 0.9)])
    assert [t for t, _ in sweep_rows(scored, radius_sq_for(), floor=0.2)] == [
        0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # No floor (RampNet's bundle detections): the full range, as before.
    assert sweep_rows(scored, radius_sq_for())[0][0] == 0.05


class _UnloadableDetector:
    """Stands in for Qwen on a laptop: it can describe itself but cannot load."""
    name = "unloadable"

    def signature(self):
        return {"provider": "unloadable"}

    def prepare(self):
        raise ImportError("no GPU / weights here")

    def detect(self, sample):
        raise AssertionError("detect() must not be reached when everything is cached")


def test_score_model_skips_model_load_when_fully_cached(tmp_path):
    # A .model_cache produced on the cluster must score on a machine that cannot
    # load the model at all — otherwise the remote run is unusable locally.
    records, verdicts = _aligned()
    gts = ground_truths_from_verdicts(records, verdicts)
    det = _UnloadableDetector()
    cache = DetectionCache(str(tmp_path))
    cache.put(cache_key("unloadable", det.signature(), "richmond", "p1"), [(0.1, 0.1, None)])
    run = score_model(det, records, gts, "", radius_sq_for(),
                      "unloadable", "richmond", cache)
    assert run.report.n_panos == 1 and run.report.tp == 1 and not run.failures


def test_score_model_loads_model_on_a_cache_miss(tmp_path):
    records, verdicts = _aligned()
    gts = ground_truths_from_verdicts(records, verdicts)
    try:
        score_model(_UnloadableDetector(), records, gts, "", radius_sq_for(),
                    "unloadable", "richmond", DetectionCache(str(tmp_path)))
    except ImportError:
        return  # prepare() still fails fast when work actually has to be done
    raise AssertionError("expected prepare() to run (and fail) when a pano is uncached")


def test_score_model_isolates_pano_failures():
    records = {pid: {"detections": [], "pano": {"width": 1, "height": 1}}
               for pid in ("good", "bad")}
    gts = {pid: GroundTruth([(0.5, 0.5)], [], True) for pid in ("good", "bad")}
    det = _FlakyDetector()
    run = score_model(det, records, gts, "", radius_sq_for(),
                      "flaky", "city", DetectionCache("x", enabled=False))
    assert run.report.n_panos == 1                   # only 'good' scored
    assert len(run.failures) == 1 and run.failures[0][0] == "bad"
    assert det.calls == 2                            # both panos attempted
    assert len(run.scored) == 1                      # the failed pano isn't re-scorable


def test_bundle_rampnet_detector_reads_records():
    records = {"p1": {"detections": [
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}]}}
    det = BundleRampNetDetector(records)
    sample = PanoSample("p1", image_path=None, width=None, height=None, meta={})
    assert det.detect(sample) == [(0.5, 0.5, 0.9)]


def test_vlm_detectors_construct_without_client_libs():
    # Constructing must not import google-genai / transformers, nor download weights;
    # that only happens on prepare()/detect().
    GeminiDetector(model_id="gemini-flash-latest")
    det = QwenDetector(model_id="Qwen/Qwen3-VL-8B-Instruct")
    assert det._model is None and det._processor is None


def test_gemini_detect_fails_clearly_without_key_or_lib():
    det = GeminiDetector(model_id="gemini-flash-latest", api_key=None)
    sample = PanoSample("p1", image_path="nope.jpg", width=100, height=100, meta={})
    try:
        det.detect(sample)
    except (ImportError, RuntimeError, NotImplementedError):
        return  # any of these is an acceptable, clear failure
    raise AssertionError("expected GeminiDetector.detect to fail loudly without lib/key")


# --- cost accounting (pricing.py + usage recording) -------------------------

def test_estimate_cost_known_and_unknown_models():
    # gemini-2.5-flash: $0.30/M in, $2.50/M out (pricing.py, verified 2026-08-15)
    assert estimate_cost("gemini-2.5-flash", 2_000_000, 1_000_000) == pytest.approx(3.10)
    assert estimate_cost("not-a-model", 1, 1) is None
    assert estimate_cost(None, 1, 1) is None
    assert price_for("not-a-model") is None


def test_pricing_entries_are_complete():
    for model, p in pricing.PRICING.items():
        assert p["input_per_m"] > 0 and p["output_per_m"] > 0, model
        assert p["as_of"], f"{model}: a price without its verification date is a rumor"


def test_gemini_usage_accumulates_thinking_as_output():
    class _Usage:
        prompt_token_count = 1000
        candidates_token_count = 50
        thoughts_token_count = 200

    class _Resp:
        usage_metadata = _Usage()

    det = GeminiDetector(model_id="gemini-3.7-flash")
    # Keyed off USAGE_KEYS rather than a re-typed literal: the shape is the base
    # class's contract, and a test that hardcodes it has to be edited every time
    # a provider reports one more thing (which is the drift PROVIDERS fixed too).
    assert det.usage == dict.fromkeys(_VLMDetector.USAGE_KEYS, 0)
    det._record_usage(_Resp())
    det._record_usage(_Resp())
    assert det.usage == dict(dict.fromkeys(_VLMDetector.USAGE_KEYS, 0),
                             calls=2, input_tokens=2000,
                             output_tokens=500,   # (50 visible + 200 thinking) x 2
                             thoughts_tokens=400)
    # Gemini reports no separate cache SKUs, so those stay at zero for it.
    assert det.usage["cache_read_input_tokens"] == 0
    # A response with no usage metadata (e.g. a mocked/older client) is a no-op.
    det._record_usage(type("R", (), {"usage_metadata": None})())
    assert det.usage["calls"] == 2


def test_gemini_usage_warns_when_the_sdk_stops_adding_up(capsys):
    # The cost figure turns on candidates_token_count EXCLUDING thinking. If a
    # future SDK folds thinking in, every thinking model's output silently doubles;
    # the provider's own total is the only thing that can catch it.
    class _Usage:
        prompt_token_count = 1000
        candidates_token_count = 250     # already includes the 200 thoughts
        thoughts_token_count = 200
        total_token_count = 1250         # != 1000 + 250 + 200

    det = GeminiDetector(model_id="gemini-3.7-flash")
    det._record_usage(type("R", (), {"usage_metadata": _Usage()})())
    out = capsys.readouterr().out
    assert "WARNING" in out and "does not add up" in out
    # Warns once per run, not once per call — a full leg is ~750 calls.
    det._record_usage(type("R", (), {"usage_metadata": _Usage()})())
    assert "WARNING" not in capsys.readouterr().out


# --- Claude on Vertex (#122) ------------------------------------------------

def test_claude_boxes_are_pixels_in_the_views_own_space():
    # Deliberately NOT Gemini's 0-1000 convention: Claude maps coordinates 1:1
    # onto image pixels, so we ask for pixels and divide by the view size.
    pts = claude_boxes_to_points([{"x1": 0, "y1": 0, "x2": 512, "y2": 256}], 1024, 1024)
    assert pts == [(0.25, 0.125, None)]
    # A different view size must move the normalized point — the guard against
    # someone "simplifying" this to a fixed 1000 divisor.
    assert claude_boxes_to_points([{"x1": 0, "y1": 0, "x2": 512, "y2": 512}],
                                  512, 512) == [(0.5, 0.5, None)]


def test_claude_box_tool_forbids_a_transposable_array():
    # The whole point of named fields: a [ymin, xmin, ymax, xmax] array can be
    # silently transposed, and this repo has shipped that bug before (Molmo).
    item = detectors.CLAUDE_BOX_TOOL["input_schema"]["properties"]["boxes"]["items"]
    assert set(item["required"]) == {"x1", "y1", "x2", "y2"}
    assert item["additionalProperties"] is False


def test_claude_skips_malformed_boxes_instead_of_losing_the_panorama():
    # Without `strict: True` (org-policy blocked) the schema is a hint, not a
    # contract. Observed once in 745 calls: a list of strings where objects were
    # asked for, which raised TypeError and cost all 6 views of that panorama.
    pts = claude_boxes_to_points(
        ["curb ramp at 100,200",                       # the real failure shape
         {"x1": 0, "y1": 0, "x2": 512, "y2": 512},     # good
         {"x1": 1, "y1": 2},                           # missing keys
         {"x1": "a", "y1": "b", "x2": "c", "y2": "d"}],  # unparseable
        1024, 1024)
    assert pts == [(0.25, 0.25, None)]


def test_claude_box_tool_is_not_strict():
    # `strict: True` is implemented as structured outputs, which this project's
    # GCP org policy blocks for Anthropic partner models (measured 2026-08-15:
    # 400, disallowed feature `structured_outputs`). Forced tool_choice without
    # it passes. If someone adds strict back, every Claude call 400s.
    assert "strict" not in detectors.CLAUDE_BOX_TOOL


def test_claude_reads_boxes_from_the_tool_call():
    blk = type("B", (), {"type": "tool_use", "input": {"boxes": [
        {"x1": 1, "y1": 2, "x2": 3, "y2": 4}]}})()
    assert boxes_from_claude_response(type("R", (), {"content": [blk]})()) == [
        {"x1": 1, "y1": 2, "x2": 3, "y2": 4}]
    # Text fallback for a turn that ended without a tool call.
    txt = type("B", (), {"type": "text", "text": '{"boxes": [{"x1":0,"y1":0,"x2":2,"y2":2}]}'})()
    assert boxes_from_claude_response(type("R", (), {"content": [txt]})()) == [
        {"x1": 0, "y1": 0, "x2": 2, "y2": 2}]


def test_claude_effort_is_in_the_cache_key():
    # Effort changes how much the model thinks, which changes the detections.
    # If it were not in the signature, a cheap `low` run and an expensive `high`
    # run would collide in one cache entry and silently mix.
    low = ClaudeDetector(model_id="claude-sonnet-5", effort="low")
    high = ClaudeDetector(model_id="claude-sonnet-5", effort="high")
    assert low.signature()["effort"] == "low"
    assert cache_key("claude-sonnet-5", low.signature(), "bend", "p1") != \
           cache_key("claude-sonnet-5", high.signature(), "bend", "p1")


def test_claude_usage_does_not_double_count_thinking():
    # Anthropic's output_tokens ALREADY includes thinking (unlike google-genai's
    # candidates_token_count, which excludes it). Adding them again would double
    # the dominant cost term.
    class _Usage:
        input_tokens = 1512
        output_tokens = 800          # inclusive of the 600 thinking tokens
        output_tokens_details = type("D", (), {"thinking_tokens": 600})()

    det = ClaudeDetector(model_id="claude-sonnet-5")
    det._record_usage(type("R", (), {"usage": _Usage(), "model": "claude-sonnet-5"})())
    assert det.usage["input_tokens"] == 1512
    assert det.usage["output_tokens"] == 800      # NOT 1400
    assert det.usage["thoughts_tokens"] == 600
    assert dict(det.model_versions) == {"claude-sonnet-5": 1}


def test_claude_warns_if_output_tokens_stops_including_thinking(capsys):
    class _Usage:
        input_tokens = 10
        output_tokens = 50           # < thinking: the assumption has inverted
        output_tokens_details = type("D", (), {"thinking_tokens": 600})()

    det = ClaudeDetector(model_id="claude-sonnet-5")
    det._record_usage(type("R", (), {"usage": _Usage(), "model": "x"})())
    out = capsys.readouterr().out
    assert "WARNING" in out and "UNDER-counting" in out


def test_claude_detect_fails_clearly_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    det = ClaudeDetector(model_id="claude-sonnet-5", project=None)
    try:
        det.prepare()
    except (RuntimeError, ImportError) as e:
        assert "GOOGLE_CLOUD_PROJECT" in str(e) or "anthropic" in str(e)
        return
    raise AssertionError("expected ClaudeDetector.prepare to fail without credentials")


def test_claude_empty_response_yields_no_boxes():
    # A refusal or a max_tokens truncation before any text must read as "no
    # detections here", not raise.
    assert boxes_from_claude_response(type("R", (), {"content": []})()) == []
    blk = type("B", (), {"type": "text", "text": '{"boxes": []}'})()
    assert boxes_from_claude_response(type("R", (), {"content": [blk]})()) == []


# --- Claude: the parse layer must never cost more than the box it choked on ---
#
# Every case below was a real crash before #123's review: each one raised out of
# ``boxes_from_claude_response`` / ``claude_boxes_to_points``, propagated through
# ``_raw_detect``, and cost all six views of a panorama (which is how the
# sonnet/low leg ended up scored on 290 GT ramps instead of annapolis's 294).
# The contract is now: a malformed ANYTHING costs exactly what it is worth and
# nothing more.

# (label, response content blocks, expected boxes) for the text-fallback path.
_CLAUDE_HOSTILE_RESPONSES = [
    # A refusal in words. This is the shape `auto` tool choice invites and the
    # shape `forced` made impossible, so it arrived WITH the default change.
    ("prose refusal", "I can't identify curb ramps in this image.", []),
    ("empty text", "", []),
    ("prose then object", 'Sure! {"boxes": [{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]}',
     [{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]),
    ("fenced object", '```json\n{"boxes": [{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]}\n```',
     [{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]),
    # The model answered with the bare array instead of the wrapper object.
    # boxes_from_gemini_response has always handled this; Claude's did not.
    ("bare array", '[{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]',
     [{"x1": 0, "y1": 0, "x2": 2, "y2": 2}]),
    ("truncated json", '{"boxes": [{"x1": 0, "y1"', []),
    ("json scalar", '42', []),
]


@pytest.mark.parametrize("label,text,expected", _CLAUDE_HOSTILE_RESPONSES,
                         ids=[c[0] for c in _CLAUDE_HOSTILE_RESPONSES])
def test_claude_text_fallback_never_raises(label, text, expected):
    blk = type("B", (), {"type": "text", "text": text})()
    assert boxes_from_claude_response(type("R", (), {"content": [blk]})()) == expected


# Tool-call inputs that satisfy "there is a tool_use block" but not "boxes is a
# list of objects". Without `strict: True` — which the org policy blocks — the
# schema is a hint, so every one of these is reachable.
_CLAUDE_HOSTILE_TOOL_INPUTS = [
    ("boxes null", {"boxes": None}),
    ("boxes int", {"boxes": 3}),
    ("boxes dict", {"boxes": {"x1": 0}}),
    ("boxes string", {"boxes": "none found"}),
    ("no boxes key", {"detections": []}),
    ("input is None", None),
    ("input is a list", [{"x1": 0}]),
]


@pytest.mark.parametrize("label,payload", _CLAUDE_HOSTILE_TOOL_INPUTS,
                         ids=[c[0] for c in _CLAUDE_HOSTILE_TOOL_INPUTS])
def test_claude_tool_use_with_a_malformed_input_yields_no_boxes(label, payload):
    blk = type("B", (), {"type": "tool_use", "input": payload})()
    boxes = boxes_from_claude_response(type("R", (), {"content": [blk]})())
    # Must be list-shaped so the parse step downstream cannot trip either.
    assert isinstance(boxes, list)
    assert claude_boxes_to_points(boxes, 1024, 1024) == []


def test_claude_parse_is_total_over_the_whole_hostile_corpus():
    """End to end: response -> boxes -> points, for every bad shape at once.

    The guarantee this pins is not "the right answer" but "an answer": the
    pipeline returns well-formed points for anything a model can emit, so one
    bad response can never again cost a whole panorama."""
    responses = [type("R", (), {"content": []})()]
    for _, text, _ in _CLAUDE_HOSTILE_RESPONSES:
        responses.append(type("R", (), {"content": [
            type("B", (), {"type": "text", "text": text})()]})())
    for _, payload in _CLAUDE_HOSTILE_TOOL_INPUTS:
        responses.append(type("R", (), {"content": [
            type("B", (), {"type": "tool_use", "input": payload})()]})())
    # A thinking-only turn, which is what a mid-thought max_tokens cut leaves.
    responses.append(type("R", (), {"content": [
        type("B", (), {"type": "thinking", "thinking": "hmm"})()]})())
    for resp in responses:
        pts = claude_boxes_to_points(boxes_from_claude_response(resp), 1024, 1024)
        assert isinstance(pts, list)
        assert all(len(p) == 3 and isinstance(p[0], float) for p in pts)


def test_claude_prefers_the_tool_call_over_a_preamble():
    # With thinking on, a turn is often [thinking, text preamble, tool_use].
    blocks = [type("B", (), {"type": "thinking", "thinking": "looking..."})(),
              type("B", (), {"type": "text", "text": "I'll report what I see."})(),
              type("B", (), {"type": "tool_use", "input": {"boxes": [
                  {"x1": 1, "y1": 2, "x2": 3, "y2": 4}]}})()]
    assert boxes_from_claude_response(type("R", (), {"content": blocks})()) == [
        {"x1": 1, "y1": 2, "x2": 3, "y2": 4}]


# --- Claude: a truncated call is an ERROR, never an empty detection ----------

def _claude_resp(stop_reason="end_turn", content=None, usage=None, model="claude-sonnet-5"):
    return type("R", (), {"stop_reason": stop_reason, "content": content or [],
                          "usage": usage, "model": model})()


def test_claude_max_tokens_truncation_raises_instead_of_scoring_zero():
    """The failure mode that has no symptom otherwise.

    A response cut off mid-thinking carries no tool_use and no text, so the
    parser correctly reads it as "no boxes". If that reached the scorer it would
    become a false negative for every GT ramp on the pano AND be written to the
    detection cache, making a silent recall loss permanent. So the detector
    rejects it before it can be cached: score_model records a visible failure."""
    det = ClaudeDetector(model_id="claude-sonnet-5")
    with pytest.raises(RuntimeError, match="max_tokens"):
        det._check_stop_reason(_claude_resp(stop_reason="max_tokens"))


def test_claude_normal_stop_reasons_pass_through():
    det = ClaudeDetector(model_id="claude-sonnet-5")
    for reason in ("end_turn", "tool_use", "stop_sequence", None):
        det._check_stop_reason(_claude_resp(stop_reason=reason))   # must not raise


def test_claude_refusal_is_counted_and_announced_not_silent(capsys):
    """A refusal still scores as "found nothing" — that is the honest reading —
    but it must be COUNTED, because a leg where 20% of calls refused is not the
    same measurement as one where none did, and nothing else would show it."""
    det = ClaudeDetector(model_id="claude-sonnet-5")
    det._check_stop_reason(_claude_resp(stop_reason="refusal"))
    det._check_stop_reason(_claude_resp(stop_reason="refusal"))
    out = capsys.readouterr().out
    assert "refusal" in out and "WARNING" in out
    assert out.count("WARNING") == 1          # warn once per run, not per call
    assert det.stop_reasons["refusal"] == 2


def test_claude_stop_reasons_are_tallied_for_every_call():
    det = ClaudeDetector(model_id="claude-sonnet-5")
    for reason in ("end_turn", "tool_use", "tool_use"):
        det._check_stop_reason(_claude_resp(stop_reason=reason))
    assert dict(det.stop_reasons) == {"end_turn": 1, "tool_use": 2}


# --- Claude: the cache key is real money ------------------------------------

def test_claude_cache_key_is_frozen():
    """Regression guard on $28.82 of already-paid annapolis detections.

    Four legs (two models x two effort levels, 125 panos x 6 views each) are
    cached under hash(label, signature, city, pano). Any drift in
    ClaudeDetector.signature() misses every one of them and re-bills the run.
    Same contract as test_gemini_cache_key_is_frozen: if this fails, the change
    was not free — revert it, or re-pay deliberately and say so in the PR."""
    expected = {
        ("claude-sonnet-5", "low"): "18605fcb5c957a8181c37affe1715afe5b030e88",
        ("claude-sonnet-5", "high"): "16a9d2d5d0bdcca687145e74d4e03ff46f5bf549",
        ("claude-opus-5", "low"): "0963255957b989a32bffbcaf9de1f0bd1701b319",
        ("claude-opus-5", "high"): "3cf3ad4e5e4c95ac22116a81e708d133463c1ff0",
    }
    for (mid, effort), want in expected.items():
        det = ClaudeDetector(model_id=mid, effort=effort, tool_choice="auto")
        assert det.prompt == DETECTION_PROMPT   # provider suffixes must not leak in
        assert cache_key(mid, det.signature(), "richmond", "pano1") == want, (
            f"{mid}/{effort} signature drifted; the annapolis cache is orphaned")


def test_claude_as_run_defaults_stay_out_of_the_signature():
    """Encoding and temperature are inputs, so they belong in the key — but only
    once they DEVIATE from what the published legs ran.

    Writing them in unconditionally would change the hash for runs whose settings
    did not change, orphaning the paid cache to record a no-op. So the signature
    carries a deviation, not a description."""
    det = ClaudeDetector(model_id="claude-sonnet-5")
    assert det.image_format == detectors.CLAUDE_AS_RUN_IMAGE_FORMAT == "jpeg"
    assert det.temperature is detectors.CLAUDE_AS_RUN_TEMPERATURE is None
    sig = det.signature()
    assert "image_format" not in sig and "temperature" not in sig


@pytest.mark.parametrize("kwargs", [{"image_format": "png"}, {"temperature": 0.0}])
def test_claude_deviating_from_the_as_run_settings_invalidates_the_cache(kwargs):
    base = ClaudeDetector(model_id="claude-sonnet-5")
    other = ClaudeDetector(model_id="claude-sonnet-5", **kwargs)
    assert cache_key("claude-sonnet-5", base.signature(), "annapolis", "p1") != \
           cache_key("claude-sonnet-5", other.signature(), "annapolis", "p1")


def test_claude_tool_choice_is_in_the_cache_key():
    auto = ClaudeDetector(model_id="claude-sonnet-5", tool_choice="auto")
    forced = ClaudeDetector(model_id="claude-sonnet-5", tool_choice="forced")
    assert cache_key("claude-sonnet-5", auto.signature(), "bend", "p1") != \
           cache_key("claude-sonnet-5", forced.signature(), "bend", "p1")


# --- Claude: what pixels the model actually sees ----------------------------

def test_claude_image_encoding_is_explicit_and_round_trips():
    """The published legs sent JPEG q90 while the Gemini leg sends lossless PNG
    (google-genai's pil_to_blob only picks JPEG for an image loaded FROM a jpeg
    file, and reprojected views come from Image.fromarray). That asymmetry is
    now a named, switchable setting instead of a buried default."""
    from PIL import Image
    img = Image.new("RGB", (8, 8), (10, 200, 30))
    img.putpixel((0, 0), (255, 0, 0))

    b64, media_type = ClaudeDetector(model_id="claude-sonnet-5")._encode_image(img)
    assert media_type == "image/jpeg"

    b64, media_type = ClaudeDetector(model_id="claude-sonnet-5",
                                     image_format="png")._encode_image(img)
    assert media_type == "image/png"
    import base64 as _b64
    import io as _io
    back = Image.open(_io.BytesIO(_b64.b64decode(b64)))
    assert back.size == (8, 8)
    assert back.getpixel((0, 0)) == (255, 0, 0)      # lossless, unlike q90 JPEG


def test_claude_rejects_an_unknown_image_format():
    with pytest.raises(ValueError, match="image_format"):
        ClaudeDetector(model_id="claude-sonnet-5", image_format="webp")


# --- Claude: cost bookkeeping that cannot be reconstructed later -------------

def test_claude_regional_endpoint_warns_that_every_cost_is_low():
    """pricing.py's Claude rates are the `global` ones and regional endpoints
    bill 10% more, so a run pointed off `global` reports costs that are ~9% low
    with nothing else to show for it. GOOGLE_CLOUD_LOCATION is shared with the
    Gemini leg, so this is set by accident, not on purpose."""
    assert ClaudeDetector(model_id="claude-sonnet-5", location="global").location_warning() is None
    msg = ClaudeDetector(model_id="claude-sonnet-5", location="us-east5").location_warning()
    assert msg and "us-east5" in msg and "10%" in msg


def test_claude_records_cache_tokens_separately():
    """Anthropic bills cache reads/writes as their own SKUs and EXCLUDES them
    from input_tokens, so a run that ever enables cache_control would otherwise
    report a cost with the cached half missing."""
    class _Usage:
        input_tokens = 100
        output_tokens = 20
        cache_read_input_tokens = 900
        cache_creation_input_tokens = 50
        output_tokens_details = None

    det = ClaudeDetector(model_id="claude-sonnet-5")
    det._record_usage(_claude_resp(usage=_Usage()))
    assert det.usage["input_tokens"] == 100
    assert det.usage["cache_read_input_tokens"] == 900
    assert det.usage["cache_write_input_tokens"] == 50


def test_estimate_cost_prices_cache_tokens_when_the_model_has_rates():
    from pricing import estimate_cost, price_for
    p = price_for("claude-sonnet-5")
    assert p["cache_read_per_m"] == 0.20 and p["cache_write_per_m"] == 2.50
    # 1M plain input + 1M cached read + 1M cache write + 1M output.
    got = estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000,
                        cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert got == pytest.approx(2.00 + 10.00 + 0.20 + 2.50)
    # Back-compat: the three-arg call every other caller uses is unchanged.
    assert estimate_cost("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.00)


def test_estimate_cost_ignores_cache_tokens_for_models_without_verified_rates():
    """Never invent a rate. A model whose cache SKUs were not read off the rate
    card prices its plain tokens and says nothing about the rest."""
    from pricing import estimate_cost, price_for
    assert "cache_read_per_m" not in price_for("gemini-3.6-flash")
    assert estimate_cost("gemini-3.6-flash", 1_000_000, 0,
                         cache_read_tokens=5_000_000) == pytest.approx(0.75)


def test_usage_record_carries_stop_reasons(tmp_path):
    """A leg's abnormal terminations belong beside its cost, because that is the
    one place someone re-reading the numbers will look."""
    det = ClaudeDetector(model_id="claude-sonnet-5")
    det.accumulate_usage(10, 5, 0)
    det.stop_reasons.update(["end_turn", "refusal"])
    log = tmp_path / "usage_log.jsonl"
    report_usage(det, "claude-sonnet-5", "annapolis", 125, str(log))
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["stop_reasons"] == {"end_turn": 1, "refusal": 1}


def test_a_paid_leg_with_the_usage_log_disabled_says_so_loudly(capsys):
    """The standing rule is that spend is recorded at run time, and the token
    counts are the ONE artifact that cannot be back-filled: a re-run reads the
    detection cache, makes zero calls, and so can never reproduce them. The
    four Claude legs on #123 spent $28.82 and left no record — this is the guard
    that would have caught it while the money was being spent."""
    det = ClaudeDetector(model_id="claude-sonnet-5")
    det.accumulate_usage(1000, 50, 0)
    report_usage(det, "claude-sonnet-5", "annapolis", 125, None)
    out = capsys.readouterr().out
    assert "WARNING" in out and "not recorded" in out.lower()


# --- ... and the run stops before the first uncached call -------------------

class _PaidDetector(_UnloadableDetector):
    """A paid provider that cannot load — so any test that reaches prepare() fails
    loudly instead of quietly reaching for the network."""
    name = "claude"


def _aligned_gts():
    records, verdicts = _aligned()
    return records, ground_truths_from_verdicts(records, verdicts)


def test_a_paid_leg_stops_before_its_first_uncached_call(tmp_path):
    """The end-of-run warning fires after the tokens are bought. This fires before
    the first one, and before prepare() — _PaidDetector raises from prepare(), so a
    guard that let the run through would surface as ImportError, not as this."""
    records, gts = _aligned_gts()
    with pytest.raises(compare.UnrecordedSpend) as exc:
        score_model(_PaidDetector(), records, gts, "", radius_sq_for(),
                    "claude-opus-5", "richmond", DetectionCache(str(tmp_path)),
                    spend_needs_recording=True)
    assert "uncached pano" in str(exc.value) and "--usage-log none" in str(exc.value)


def test_a_fully_cached_paid_leg_still_runs(tmp_path):
    """It provably cannot spend: every pano is cached, so the model is never loaded
    and no call is made. Refusing it would block re-scoring the published detections
    from a clean clone, which is the path the roster exists to keep open."""
    records, gts = _aligned_gts()
    det = _PaidDetector()
    cache = DetectionCache(str(tmp_path))
    cache.put(cache_key("claude-opus-5", det.signature(), "richmond", "p1"),
              [(0.1, 0.1, None)])
    run = score_model(det, records, gts, "", radius_sq_for(), "claude-opus-5",
                      "richmond", cache, spend_needs_recording=True)
    assert run.report.n_panos == 1 and not run.failures


def test_a_free_model_may_run_unrecorded(tmp_path):
    """The rule is about spend, not about logging for its own sake."""
    records, gts = _aligned_gts()
    with pytest.raises(ImportError):        # reached prepare(), i.e. was not refused
        score_model(_UnloadableDetector(), records, gts, "", radius_sq_for(),
                    "unloadable", "richmond", DetectionCache(str(tmp_path)),
                    spend_needs_recording=True)


def test_unrecorded_spend_stays_possible_but_deliberate(tmp_path):
    """A guard with no override gets edited out instead, which is worse. With
    --allow-unrecorded-spend the run proceeds — here, as far as prepare()."""
    records, gts = _aligned_gts()
    with pytest.raises(ImportError):
        score_model(_PaidDetector(), records, gts, "", radius_sq_for(),
                    "claude-opus-5", "richmond", DetectionCache(str(tmp_path)),
                    spend_needs_recording=False)


def test_the_refusal_is_not_swallowed_as_an_unrunnable_model(monkeypatch, capsys):
    """main() skips a model that cannot run here, with a printed note. The refusal
    must NOT take that path — swallowing it would let the next paid leg in the same
    --models list spend unrecorded too."""
    def _refuse(*a, **kw):
        raise compare.UnrecordedSpend("would spend")
    monkeypatch.setattr(compare, "score_model", _refuse)
    monkeypatch.setattr(sys, "argv", ["compare.py",
                                      os.path.join(REPO_ROOT, "benchmark", "richmond"),
                                      "--models", "rampnet", "--usage-log", "none"])
    with pytest.raises(compare.UnrecordedSpend):
        compare.main()
    assert "not runnable" not in capsys.readouterr().out


def test_the_paid_provider_list_covers_every_priced_model():
    """pricing.py knows what a model costs; the roster knows which providers cost
    anything. If a priced model's provider is not in PAID_PROVIDERS, the guard has
    a hole exactly where money is being spent."""
    from rampnet import roster
    import pricing
    for model_id in pricing.PRICING:
        provider = next((c.provider for c in roster.ROSTER if c.label == model_id), None)
        if provider is None:
            continue                     # priced but not registered (e.g. retired alias)
        assert provider in roster.PAID_PROVIDERS, model_id


# --- the provider roster has ONE source of truth ----------------------------

def test_every_provider_is_listed_everywhere_a_user_looks():
    """`claude` shipped in build_detector but was missing from the --models help,
    compare.py's docstring and parse_model_spec's docstring. Rosters drift the
    moment they are duplicated, so they are now generated from PROVIDERS — and
    the prose that cannot be generated is checked here."""
    import compare
    assert detectors.PROVIDERS == ("rampnet", "gemini", "claude", "qwen", "owlv2",
                                   "gdino", "molmo", "vistas", "yolo")
    # The scored-detector list is prose that cannot be generated, and "vistas appears
    # somewhere in the docstring" is not coverage of it -- it appeared on the --models
    # line while both enumerations of who gets an AP / PR curve / sweep still omitted
    # it, in the PR whose headline result is that arm's AP.
    scored = {"rampnet", "owlv2", "gdino", "vistas", "yolo"}
    import re
    lists = re.findall(r"calibrated scores \(([^)]*)\)", compare.__doc__)
    lists += re.findall(r"carry confidences \(([^)]*)\)",
                        " ".join(a.help or "" for a in compare.build_parser()._actions))
    assert lists, "neither enumeration of the scored detectors was found"
    for text in lists:
        named = {w.strip().lower().rstrip(",") for w in text.replace("Grounding DINO", "gdino").split(",")}
        assert scored <= {n.replace(" ", "") for n in named}, text
    for provider in detectors.PROVIDERS:
        assert provider in compare.MODELS_HELP, f"{provider} missing from --models help"
        assert provider in compare.__doc__, f"{provider} missing from compare.py docstring"
        assert provider in detectors.parse_model_spec.__doc__, \
            f"{provider} missing from parse_model_spec docstring"


def test_unknown_provider_names_every_valid_one():
    with pytest.raises(ValueError) as e:
        build_detector("clyde", None, {}, _Args())
    for provider in detectors.PROVIDERS:
        assert provider in str(e.value)


def test_build_detector_wires_claude_from_the_cli_args():
    """There was no build_detector test for claude at all, which is why _Args
    was never given the flags the branch reads."""
    label, det = build_detector("claude", None, {}, _Args())
    assert label == "claude-sonnet-5" and isinstance(det, ClaudeDetector)
    assert (det.effort, det.tool_choice) == ("low", "auto")

    class _Pinned(_Args):
        claude_model = "claude-opus-5"
        claude_effort = "high"
        claude_tool_choice = "forced"
    label, det = build_detector("claude", None, {}, _Pinned())
    assert label == "claude-opus-5"
    assert (det.effort, det.tool_choice) == ("high", "forced")
    # A pinned id on the --models token beats the flag.
    label, det = build_detector("claude", "claude-sonnet-5", {}, _Pinned())
    assert label == "claude-sonnet-5"


# --- the box-mapping gate must survive what the detector survives -----------

def test_dump_detections_skips_items_that_are_not_boxes():
    """dump_detections.py is the gate that catches a transposed coordinate
    convention, so it must not be the thing that crashes on the malformed item
    the detector now tolerates."""
    shapes = detections_to_view_shapes(
        None, ["curb ramp at x1=100", None, 7,
               {"x1": 10, "y1": 20, "x2": 30, "y2": 40}], 1024, 1024)
    assert shapes == [("rect", 10, 20, 30, 40, None)]


def _gemini_resp(model_version=None, prompt=1000, candidates=50, thoughts=200):
    usage = type("U", (), {"prompt_token_count": prompt,
                           "candidates_token_count": candidates,
                           "thoughts_token_count": thoughts,
                           "total_token_count": prompt + candidates + thoughts})()
    return type("R", (), {"usage_metadata": usage, "model_version": model_version})()


def test_the_resolved_build_is_recorded_against_the_alias_we_asked_for(capsys):
    # model_id is an alias the provider is free to move; this is what answered.
    det = GeminiDetector(model_id="gemini-3.7-flash")
    assert det.model_versions is None
    for _ in range(3):
        det._record_usage(_gemini_resp("gemini-3.7-flash-001"))
    capsys.readouterr()
    assert dict(det.model_versions) == {"gemini-3.7-flash-001": 3}
    # And it stays OUT of the cache key: adding it there would miss every
    # already-paid cached detection. test_gemini_cache_key_is_frozen guards the
    # hash itself; this guards the reason.
    assert "model_version" not in det.signature()
    assert det.signature()["model_id"] == "gemini-3.7-flash"


def test_a_build_rotation_mid_run_is_loud():
    # The case nobody would otherwise notice: alias, signature and cache key are
    # all unchanged, so two models' detections land in one file indistinguishably.
    import io, contextlib
    det = GeminiDetector(model_id="gemini-3.7-flash")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        det._record_usage(_gemini_resp("gemini-3.7-flash-001"))
        det._record_usage(_gemini_resp("gemini-3.7-flash-002"))
    out = buf.getvalue()
    assert "WARNING" in out and "changed mid-run" in out
    assert dict(det.model_versions) == {"gemini-3.7-flash-001": 1,
                                        "gemini-3.7-flash-002": 1}


def test_a_provider_that_reports_no_build_stays_none(capsys):
    # None must mean "not reported", never an empty dict that reads as "asked and
    # got nothing" -- the published files predating #121 are in exactly that state.
    det = GeminiDetector(model_id="gemini-3.7-flash")
    det._record_usage(_gemini_resp(None))
    capsys.readouterr()
    assert det.model_versions is None
    assert det.usage["calls"] == 1     # the call still counted


def test_the_usage_record_carries_the_build(tmp_path):
    log = tmp_path / "usage_log.jsonl"
    det = GeminiDetector(model_id="gemini-2.5-flash")
    det._record_usage(_gemini_resp("gemini-2.5-flash-002"))
    report_usage(det, "gemini-2.5-flash", "bend", 110, str(log))
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["model_id"] == "gemini-2.5-flash"          # what we asked for
    assert rec["model_versions"] == {"gemini-2.5-flash-002": 1}   # what answered

    # A local model reports neither, and must log null rather than {}.
    class _Local:
        name = "qwen"
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        usage = {"calls": 4, "input_tokens": 1, "output_tokens": 1}
    report_usage(_Local(), "qwen", "bend", 110, str(log))
    assert json.loads(log.read_text().splitlines()[1])["model_versions"] is None


def test_gemini_usage_is_quiet_when_the_totals_agree(capsys):
    class _Usage:
        prompt_token_count = 1000
        candidates_token_count = 50
        thoughts_token_count = 200
        total_token_count = 1250         # == 1000 + 50 + 200, today's semantics

    det = GeminiDetector(model_id="gemini-3.7-flash")
    det._record_usage(type("R", (), {"usage_metadata": _Usage()})())
    assert "WARNING" not in capsys.readouterr().out
    assert det.usage["output_tokens"] == 250


def test_report_usage_appends_jsonl(tmp_path):
    class _Det:
        name = "gemini"
        model_id = "gemini-2.5-flash"
        usage = {"calls": 6, "input_tokens": 2_000_000, "output_tokens": 1_000_000,
                 "thoughts_tokens": 300}

    log = tmp_path / "usage_log.jsonl"
    report_usage(_Det(), "gemini-2.5-flash", "richmond", 124, str(log))
    report_usage(_Det(), "gemini-2.5-flash", "bend", 110, str(log))
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["bundle"] for r in recs] == ["richmond", "bend"]
    assert recs[0]["est_cost_usd"] == 3.10 and recs[0]["calls"] == 6
    assert recs[0]["pricing"]["as_of"]
    # A detector with no usage (local model) writes nothing.
    class _Local:
        model_id = "x"
        usage = None
    report_usage(_Local(), "x", "richmond", 124, str(log))
    assert len(log.read_text().splitlines()) == 2


def test_report_usage_writes_lf_on_every_platform(tmp_path):
    """The spend ledger is append-only and byte-compared in review, so a CRLF line is a
    real defect — and `read_text().splitlines()` above cannot see one, because it strips
    \\r\\n and \\n alike. Asserted on the bytes instead.

    The writer was appending CRLF on Windows and git's autocrlf was normalising it away
    on commit, so the blob looked right while every working copy was wrong. Same defect
    imagery_manifest.py was fixed for in 22dd536.
    """
    class _Det:
        name = "gemini"
        model_id = "gemini-2.5-flash"
        usage = {"calls": 1, "input_tokens": 10, "output_tokens": 10,
                 "thoughts_tokens": 0}

    log = tmp_path / "usage_log.jsonl"
    report_usage(_Det(), "gemini-2.5-flash", "richmond", 1, str(log))
    report_usage(_Det(), "gemini-2.5-flash", "bend", 1, str(log))
    raw = log.read_bytes()
    assert b"\r\n" not in raw, "usage ledger written with CRLF"
    assert raw.count(b"\n") == 2 and raw.endswith(b"\n")


def test_usage_record_carries_the_rig_that_priced_it(tmp_path):
    # Two runs of the same model on the same bundle at different tiling cost ~6x
    # different input, and without the signature the log cannot tell them apart.
    log = tmp_path / "usage_log.jsonl"
    for tile in (True, False):
        det = GeminiDetector(model_id="gemini-2.5-flash", tile=tile)
        det.usage = {"calls": 1, "input_tokens": 10, "output_tokens": 2,
                     "thoughts_tokens": 0}
        report_usage(det, "gemini-2.5-flash", "bend", 110, str(log))
    a, b = [json.loads(line) for line in log.read_text().splitlines()]
    assert a["signature"]["tile"] is True and b["signature"]["tile"] is False
    assert a["signature"]["views"] and b["signature"]["views"] is None
    # Everything else about the two records is identical apart from the timestamp,
    # which is exactly why the signature has to be there.
    assert {k: v for k, v in a.items() if k not in ("ts", "signature")} == \
           {k: v for k, v in b.items() if k not in ("ts", "signature")}


def test_report_usage_never_raises_on_a_partial_or_odd_run(tmp_path):
    # It is called from a finally, after money has been spent. Every one of these
    # would otherwise kill a run that had already paid.
    log = tmp_path / "usage_log.jsonl"

    class _NoThinkingField:          # most providers don't report thinking
        name = "someprovider"
        model_id = "gemini-2.5-flash"
        usage = {"calls": 3, "input_tokens": 1000, "output_tokens": 200}

    report_usage(_NoThinkingField(), "someprovider", "bend", None, str(log))
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["thoughts_tokens"] == 0 if "thoughts_tokens" in rec else True
    # panos_scored is None when score_model never returned: the token counts are
    # still exact, only the denominator is unknown.
    assert rec["panos_scored"] is None and rec["calls"] == 3

    class _NoModelId:               # a detector that never got as far as naming itself
        usage = {"calls": 1, "input_tokens": 5, "output_tokens": 1}

    report_usage(_NoModelId(), "mystery", "bend", 1, str(log))
    rec = json.loads(log.read_text().splitlines()[1])
    assert rec["model_id"] is None and rec["est_cost_usd"] is None


def test_an_unwritable_usage_log_does_not_abort_the_run(tmp_path, capsys):
    # The log write sits on the critical path of a paid multi-model comparison;
    # losing the file must not lose the table.
    class _Det:
        name = "gemini"
        model_id = "gemini-2.5-flash"
        usage = {"calls": 2, "input_tokens": 100, "output_tokens": 10,
                 "thoughts_tokens": 0}

    blocked = tmp_path / "a_directory_where_the_file_goes"
    blocked.mkdir()
    report_usage(_Det(), "gemini-2.5-flash", "bend", 110, str(blocked))
    out = capsys.readouterr().out
    assert "WARNING" in out and "usage record" in out
    # The numbers survive in the run log even though the file could not be written.
    assert '"calls": 2' in out


def test_the_default_usage_log_is_tracked_not_in_the_gitignored_cache():
    # The whole point of recording spend is that it outlives the machine. Defaulting
    # into .model_cache/ (gitignored) is the bug export_model_cache.py exists to undo.
    assert ".model_cache" not in DEFAULT_USAGE_LOG
    assert DEFAULT_USAGE_LOG.replace("\\", "/").endswith("analysis_out/usage_log.jsonl")
    with open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8") as fh:
        gitignore = fh.read()
    assert "!analysis_out/usage_log.jsonl" in gitignore, \
        "analysis_out/* is ignored, so the log needs an explicit re-include"


# --- tiled detect() end-to-end (no live model) ------------------------------

class _FakeTiledVLM(_VLMDetector):
    """A live-model-free _VLMDetector: _raw_detect echoes fixed per-view points and
    _parse passes them through, so detect() exercises the real tiled path — the
    view loop, per-view back-projection to pano coords, and cross-view dedup —
    without any client library."""
    name = "faketiled"

    def __init__(self, points_per_view, **kw):
        super().__init__("fake-model", **kw)
        self._ppv = points_per_view

    def _ensure_ready(self):
        pass

    def _raw_detect(self, image):
        return self._ppv

    def _parse(self, raw, img_w, img_h):
        return list(raw)


def _write_equirect(path):
    import numpy as np
    from PIL import Image
    Image.fromarray(np.zeros((128, 256, 3), dtype="uint8")).save(path)


def test_vlm_tiled_detect_maps_each_view_back_to_pano(tmp_path):
    from equirect_tiling import default_views
    pano = tmp_path / "p.jpg"
    _write_equirect(pano)
    sample = PanoSample("p", str(pano), 256, 128, {})

    # One detection at each view's center -> one mapped pano point per view.
    det = _FakeTiledVLM([(0.5, 0.5, None)], tile=True)
    pts = det.detect(sample)
    views = default_views()
    assert len(pts) == len(views)                       # 6 well-separated views, none merged
    # Every view is pitched to -30 deg, so each center maps to latitude -30 (Y=0.6667).
    assert all(abs(y - (0.5 + 30.0 / 180.0)) < 1e-6 for (_, y, _) in pts)
    assert any(abs(x - 0.5) < 1e-6 for (x, _, _) in pts)  # the yaw-0 view -> longitude 0
    assert all(conf is None for (_, _, conf) in pts)      # confidence carried through


def test_vlm_tiled_detect_dedups_overlapping_views(tmp_path):
    from equirect_tiling import View
    pano = tmp_path / "p.jpg"
    _write_equirect(pano)
    sample = PanoSample("p", str(pano), 256, 128, {})

    # Two identical views + the same center detection -> both map to one pano point,
    # which dedup must merge to a single detection (the seam-overlap contract).
    v = View(0.0, -30.0, 90.0, 90.0, 256, 256)
    det = _FakeTiledVLM([(0.5, 0.5, None)], tile=True, views=[v, v])
    assert len(det.detect(sample)) == 1


def test_vlm_tiled_detect_preserves_scores_and_keeps_the_best_duplicate(tmp_path):
    # A scored detector (OWLv2/GDINO) must come out of the tiled path with its
    # confidences intact — everything downstream (AP, PR curve, sweep) needs them —
    # and a cross-view duplicate must resolve to the higher-scoring copy.
    from equirect_tiling import View
    pano = tmp_path / "p.jpg"
    _write_equirect(pano)
    sample = PanoSample("p", str(pano), 256, 128, {})
    v = View(0.0, -30.0, 90.0, 90.0, 256, 256)
    det = _FakeTiledVLM([(0.5, 0.5, 0.42)], tile=True, views=[v])
    assert [p[2] for p in det.detect(sample)] == [0.42]

    class _Varying(_FakeTiledVLM):
        def _raw_detect(self, image):
            self._n = getattr(self, "_n", 0) + 1
            return [(0.5, 0.5, 0.3 * self._n)]      # same spot, rising score
    dets = _Varying([], tile=True, views=[v, v]).detect(sample)
    assert len(dets) == 1 and dets[0][2] == 0.6


# --- pre-flight bundle validation -------------------------------------------

def _aligned():
    records = {"p1": {"detections": [{"x_normalized": 0.1, "y_normalized": 0.1,
                                      "confidence": 0.9}], "pano": {}}}
    verdicts = {"p1": {"dets": [True], "missed": [], "no_missed": True}}
    return records, verdicts


def test_validate_bundle_passes_on_aligned():
    records, verdicts = _aligned()
    validate_bundle(records, verdicts)  # must not raise


def test_validate_bundle_flags_missing_record():
    records, verdicts = _aligned()
    verdicts["ghost"] = {"dets": [], "missed": [], "no_missed": True}
    _assert_validation_mentions(records, verdicts, "ghost")


def test_validate_bundle_flags_misaligned_lengths():
    records, verdicts = _aligned()
    verdicts["p1"]["dets"] = [True, False]  # 2 verdicts vs 1 detection
    _assert_validation_mentions(records, verdicts, "misaligned")


def test_validate_bundle_flags_missing_field():
    records, verdicts = _aligned()
    del verdicts["p1"]["no_missed"]  # legacy-style entry: rejected, not defaulted
    _assert_validation_mentions(records, verdicts, "no_missed")


def _assert_validation_mentions(records, verdicts, needle):
    try:
        validate_bundle(records, verdicts)
    except SystemExit as e:
        assert needle in str(e)
        return
    raise AssertionError(f"expected validate_bundle to reject the bundle mentioning {needle!r}")


# --- manual-GT bundles (benchmark/manual_gold, issue #58) --------------------

def _manual_bundle(tmp_path, with_detections=True):
    """A tiny gt_source.json bundle: p1 has one labeled ramp, p2 is a negative."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "p1.txt").write_text("0 0.5 0.25 0.1 0.2\n")
    (labels / "p2.txt").write_text("")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "gt_source.json").write_text(json.dumps(
        {"format": "yolo_points", "labels_dir": "../labels"}))
    rows = []
    for pid in ("p1", "p2"):
        rec = {"pano": {"panorama_id": pid, "width": 4096, "height": 2048}}
        if with_detections:
            rec["detections"] = []
        rows.append(json.dumps(rec))
    (bundle / "records.jsonl").write_text("\n".join(rows) + "\n")
    return str(bundle)


def test_load_bundle_accepts_a_manual_gt_bundle(tmp_path):
    records, verdicts, _ = load_bundle(_manual_bundle(tmp_path))
    assert verdicts is None                      # no review — GT comes from the labels
    assert set(records) == {"p1", "p2"}


def test_load_bundle_rejects_a_dir_with_neither_gt_source(tmp_path):
    (tmp_path / "records.jsonl").write_text("")
    try:
        load_bundle(str(tmp_path))
    except SystemExit as e:
        assert "neither" in str(e)
        return
    raise AssertionError("expected load_bundle to reject a bundle with no GT source")


def test_load_manual_ground_truths_builds_points_and_negatives(tmp_path):
    gts = load_manual_ground_truths(_manual_bundle(tmp_path))
    assert gts["p1"].gt_points == [(0.5, 0.25)]         # box center, w/h dropped
    assert gts["p1"].ignore_points == []                # no 'unsure' class exists
    # The negative pano still joins the recall pool (full labeling = complete scan)
    # and its false positives count — that's where VLM hallucination shows up.
    assert gts["p2"].gt_points == [] and gts["p2"].fn_confirmed is True


def test_load_manual_ground_truths_rejects_unknown_format(tmp_path):
    bundle = _manual_bundle(tmp_path)
    with open(os.path.join(bundle, "gt_source.json"), "w") as f:
        json.dump({"format": "coco", "labels_dir": "../labels"}, f)
    try:
        load_manual_ground_truths(bundle)
    except SystemExit as e:
        assert "yolo_points" in str(e)
        return
    raise AssertionError("expected an unknown gt_source format to be rejected")


def _assert_manual_validation_mentions(records, gts, needle, **kw):
    try:
        validate_manual_bundle(records, gts, **kw)
    except SystemExit as e:
        assert needle in str(e)
        return
    raise AssertionError(f"expected validate_manual_bundle to complain about {needle!r}")


def test_validate_manual_bundle_flags_label_and_record_drift(tmp_path):
    bundle = _manual_bundle(tmp_path)
    records, _, _ = load_bundle(bundle)
    gts = load_manual_ground_truths(bundle)
    validate_manual_bundle(records, gts)                      # aligned: must not raise
    _assert_manual_validation_mentions(
        {"p1": records["p1"]}, gts, "absent from records")    # labeled, no record
    _assert_manual_validation_mentions(
        dict(records, p3={"pano": {}}), gts, "no label file")  # record, no label


def test_validate_manual_bundle_requires_detections_only_for_rampnet(tmp_path):
    bundle = _manual_bundle(tmp_path, with_detections=False)
    records, _, _ = load_bundle(bundle)
    gts = load_manual_ground_truths(bundle)
    validate_manual_bundle(records, gts, need_detections=False)   # VLM-only run: fine
    _assert_manual_validation_mentions(records, gts, "export_gold_records",
                                       need_detections=True)


class _FixedDetector:
    """Returns the same point for every pano (uncached, so detect() always runs)."""
    name = "fixed"

    def prepare(self):
        pass

    def signature(self):
        return None

    def detect(self, sample):
        return [(0.5, 0.25, None)]


def test_score_model_scores_a_manual_bundle_end_to_end(tmp_path):
    bundle = _manual_bundle(tmp_path)
    records, _, _ = load_bundle(bundle)
    gts = load_manual_ground_truths(bundle)
    run = score_model(_FixedDetector(), records, gts, "", radius_sq_for(),
                      "fixed", "manual_gold", DetectionCache("x", enabled=False))
    # The fixed point hits p1's ramp (TP) and hallucinates on the negative p2 (FP);
    # both panos are recall-confirmed, so n_gt_recall counts p1's single ramp.
    r = run.report
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)
    assert r.precision == 0.5 and r.recall == 1.0
    assert r.n_recall_panos == 2


# --------------------------------------------------------------------------- #
# Mapillary Vistas supervised-transfer arm (#126)
# --------------------------------------------------------------------------- #
def _seg(rows):
    """Build a class-id map from a picture, e.g. ["..99..", "..99.."] -> ids."""
    import numpy as np
    return np.array([[int(c) if c.isdigit() else 0 for c in r] for r in rows])


def test_masks_to_points_puts_one_point_at_each_components_centroid():
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["0000000000",
                "0990000990",
                "0990000990",
                "0000000000"])
    pts = masks_to_points(seg, np.ones_like(seg, dtype=float), (9,), min_area_px=1)
    assert len(pts) == 2
    xs = sorted(round(p[0], 4) for p in pts)
    # Left blob spans cols 1-2, right blob cols 7-8; centroids at 1.5 and 7.5 of 10.
    assert xs == [0.15, 0.75]
    assert all(round(p[1], 4) == round(1.5 / 4, 4) for p in pts)


def test_masks_to_points_returns_nothing_when_the_class_is_absent():
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["0220", "0220"])
    assert masks_to_points(seg, np.ones_like(seg, dtype=float), (9,)) == []


def test_masks_to_points_drops_components_below_the_area_floor():
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["9000009900",
                "0000009900"])
    probs = np.ones_like(seg, dtype=float)
    # The 1-px speck goes, the 4-px blob stays. This floor is a CACHE floor, which
    # is why it lives in the signature.
    pts = masks_to_points(seg, probs, (9,), min_area_px=4)
    assert len(pts) == 1
    assert round(pts[0][0], 4) == 0.65


def test_masks_to_points_scores_each_component_by_its_mean_confidence():
    """Every prediction must carry a score, or detection_eval.aggregate refuses to
    compute AP and a PR curve — and the precision side is the whole question here."""
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["9900", "9900"])
    probs = np.array([[0.2, 0.4, 0.0, 0.0],
                      [0.6, 0.8, 0.0, 0.0]])
    pts = masks_to_points(seg, probs, (9,), min_area_px=1)
    assert len(pts) == 1
    assert abs(pts[0][2] - 0.5) < 1e-9
    assert all(p[2] is not None for p in pts)


def test_masks_to_points_joins_diagonally_touching_pixels():
    """A ramp at a shallow angle can be a diagonal chain; 4-connectivity would split
    it into a row of separate detections."""
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["9000", "0900", "0090"])
    pts = masks_to_points(seg, np.ones_like(seg, dtype=float), (9,), min_area_px=1)
    assert len(pts) == 1


def test_masks_to_points_unions_the_classes_it_is_given():
    """The curb-cut+curb arm: a ramp the model splits across the two adjacent classes
    is one detection, not two."""
    import numpy as np
    from detectors import masks_to_points
    seg = _seg(["9922", "9922"])
    probs = np.ones_like(seg, dtype=float)
    assert len(masks_to_points(seg, probs, (9,), min_area_px=1)) == 1
    assert len(masks_to_points(seg, probs, (9, 2), min_area_px=1)) == 1
    # ... and the union's centroid sits between them, not on either alone.
    joined = masks_to_points(seg, probs, (9, 2), min_area_px=1)[0]
    assert round(joined[0], 4) == 0.375


def test_build_detector_wires_both_vistas_arms_and_labels_them_apart():
    from detectors import build_detector, parse_model_spec, VISTAS_CHECKPOINT
    for spec, expect_ids, expect_label in [
            ("vistas", [9], "mask2former-vistas-curb-cut"),
            ("vistas:curb-cut", [9], "mask2former-vistas-curb-cut"),
            ("vistas:curb-cut+curb", [9, 2], "mask2former-vistas-curb-cut+curb"),
    ]:
        label, det = build_detector(*parse_model_spec(spec), None, _Args())
        assert label == expect_label, spec
        assert det.signature()["class_ids"] == expect_ids, spec
        # model_id is the LABEL (the published-artifact contract); the checkpoint has
        # its own field, exactly as YoloDetector does with a weights path.
        assert det.model_id == expect_label
        assert det.signature()["checkpoint"] == VISTAS_CHECKPOINT


def test_vistas_rejects_an_unknown_class_set():
    from detectors import build_detector, parse_model_spec
    with pytest.raises(ValueError):
        build_detector(*parse_model_spec("vistas:sidewalk"), None, _Args())


def test_vistas_signature_pins_everything_that_changes_the_masks():
    """dtype is in here because fp16 and fp32 do not agree, so a desktop run and a
    cluster run must not collide in one cache entry."""
    from detectors import build_detector, parse_model_spec
    _, det = build_detector(*parse_model_spec("vistas:curb-cut"), None, _Args())
    sig = det.signature()
    for key in ("class_set", "class_ids", "class_names", "min_area_px", "dtype",
                "model_id", "views", "tile"):
        assert key in sig, key
    assert sig["prompt"] == "vistas:curb-cut"


def test_vistas_arms_have_distinct_cache_keys():
    from detectors import build_detector, parse_model_spec
    from compare import cache_key
    keys = set()
    for spec in ("vistas:curb-cut", "vistas:curb-cut+curb"):
        label, det = build_detector(*parse_model_spec(spec), None, _Args())
        keys.add(cache_key(label, det.signature(), "richmond", "p1"))
    assert len(keys) == 2


def test_vistas_constructs_without_weights_or_network():
    """Same contract as every other open model: importing torch/transformers and
    downloading a checkpoint happens in _ensure_ready, never in __init__."""
    from detectors import VistasDetector
    det = VistasDetector(class_set="curb-cut")
    assert det.class_ids == (9,)
    assert det._model is None
