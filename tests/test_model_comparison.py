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
    BundleRampNetDetector, GeminiDetector, QwenDetector, PanoSample, _VLMDetector,
    gemini_boxes_to_points, qwen_boxes_to_points, boxes_from_gemini_response,
    parse_model_spec, build_detector,
)


from compare import score_model, validate_bundle, DetectionCache, cache_key  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402


class _Args:
    gemini_model = "gemini-3.6-flash"
    qwen_model = "Qwen/Qwen3-VL"
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


def test_qwen_boxes_to_points_normalizes_by_image_size():
    # bbox_2d = [x1, y1, x2, y2] in pixels of the image shown to the model.
    pts = qwen_boxes_to_points([{"bbox_2d": [100, 200, 300, 400]}], img_w=1000, img_h=2000)
    assert pts == [(0.2, 0.15, None)]  # cx=200/1000, cy=300/2000


def test_boxes_from_gemini_response_parsed_objects():
    resp = _FakeResp(parsed=[_FakeBox([400, 200, 600, 400], "curb ramp")])
    assert boxes_from_gemini_response(resp) == [{"box_2d": [400, 200, 600, 400], "label": "curb ramp"}]


def test_boxes_from_gemini_response_json_text_fallback():
    resp = _FakeResp(parsed=None, text='[{"box_2d": [1, 2, 3, 4], "label": "x"}]')
    assert boxes_from_gemini_response(resp) == [{"box_2d": [1, 2, 3, 4], "label": "x"}]


def test_boxes_from_gemini_response_empty():
    assert boxes_from_gemini_response(_FakeResp(parsed=None, text=None)) == []
    assert boxes_from_gemini_response(_FakeResp(parsed=[], text="[]")) == []


def test_parse_model_spec():
    assert parse_model_spec("rampnet") == ("rampnet", None)
    assert parse_model_spec("gemini") == ("gemini", None)
    assert parse_model_spec("gemini:gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert parse_model_spec(" qwen : Qwen/Qwen3-VL ") == ("qwen", "Qwen/Qwen3-VL")


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


def test_score_model_isolates_pano_failures():
    records = {pid: {"detections": [], "pano": {"width": 1, "height": 1}}
               for pid in ("good", "bad")}
    verdicts = {pid: {"dets": [], "missed": [{"x": 0.5, "y": 0.5}], "no_missed": True}
                for pid in ("good", "bad")}
    det = _FlakyDetector()
    report, failures = score_model(det, records, verdicts, "", radius_sq_for(),
                                   "flaky", "city", DetectionCache("x", enabled=False))
    assert report.n_panos == 1                       # only 'good' scored
    assert len(failures) == 1 and failures[0][0] == "bad"
    assert det.calls == 2                            # both panos attempted


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
