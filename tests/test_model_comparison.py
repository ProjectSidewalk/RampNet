"""Guards for the model-comparison harness (scripts/model_comparison/).

Covers the pure box->point parsing (both providers' box conventions), that the VLM
detectors construct without their client libraries and only fail — with a clear
message — when a live detection is actually requested, and that the detection
cache stays valid across changes (see ``test_gemini_cache_key_is_frozen``).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from detectors import (  # noqa: E402
    BundleRampNetDetector, GeminiDetector, GroundingDinoDetector, MolmoDetector,
    OwlV2Detector, QwenDetector, PanoSample, _VLMDetector,
    gemini_boxes_to_points, qwen_boxes_to_points, boxes_from_gemini_response,
    boxes_from_qwen_text, infer_qwen_coord_space, parse_model_spec, build_detector,
    molmo_points_from_text, molmo_token_points_to_items, infer_molmo_mode,
    owlv2_target_size, pixel_boxes_to_points, zero_shot_results_to_boxes,
    CURB_RAMP_DEFINITION, DETECTION_PROMPT, GDINO_QUERY, MOLMO_PROMPT, OWLV2_QUERY,
)
from dump_detections import detections_to_view_shapes  # noqa: E402


from compare import (  # noqa: E402
    score_model, validate_bundle, DetectionCache, cache_key, has_confidences, rescore,
    sweep_rows,
)
from rampnet.detection_eval import GroundTruth, radius_sq_for  # noqa: E402


class _Args:
    gemini_model = "gemini-3.6-flash"
    qwen_model = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_coord_space = "auto"
    owlv2_model = "google/owlv2-large-patch14-ensemble"
    gdino_model = "IDEA-Research/grounding-dino-base"
    molmo_model = "allenai/Molmo2-8B"
    owlv2_query = None
    gdino_query = None
    gdino_text_threshold = None
    score_threshold = None
    molmo_coord_scale = "auto"
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
        return
    raise AssertionError("expected an unknown provider to raise")


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


def test_has_confidences_requires_every_prediction_to_carry_one():
    assert has_confidences(_one_pano([(0.0, 0.0, 0.9)]))
    assert not has_confidences(_one_pano([(0.0, 0.0, 0.9), (0.5, 0.5, None)]))
    assert not has_confidences(_one_pano([]))     # nothing detected: no curve to draw


def test_sweep_rows_stop_at_the_highest_score_present():
    rows = sweep_rows(_one_pano([(0.0, 0.0, 0.22)]), radius_sq_for())
    assert [t for t, _ in rows] == [0.05, 0.1, 0.15, 0.2]   # 0.25+ would be all-empty
    assert all(r.tp == 1 for _, r in rows)


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
    det = _UnloadableDetector()
    cache = DetectionCache(str(tmp_path))
    cache.put(cache_key("unloadable", det.signature(), "richmond", "p1"), [(0.1, 0.1, None)])
    run = score_model(det, records, verdicts, "", radius_sq_for(),
                      "unloadable", "richmond", cache)
    assert run.report.n_panos == 1 and run.report.tp == 1 and not run.failures


def test_score_model_loads_model_on_a_cache_miss(tmp_path):
    records, verdicts = _aligned()
    try:
        score_model(_UnloadableDetector(), records, verdicts, "", radius_sq_for(),
                    "unloadable", "richmond", DetectionCache(str(tmp_path)))
    except ImportError:
        return  # prepare() still fails fast when work actually has to be done
    raise AssertionError("expected prepare() to run (and fail) when a pano is uncached")


def test_score_model_isolates_pano_failures():
    records = {pid: {"detections": [], "pano": {"width": 1, "height": 1}}
               for pid in ("good", "bad")}
    verdicts = {pid: {"dets": [], "missed": [{"x": 0.5, "y": 0.5}], "no_missed": True}
                for pid in ("good", "bad")}
    det = _FlakyDetector()
    run = score_model(det, records, verdicts, "", radius_sq_for(),
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
