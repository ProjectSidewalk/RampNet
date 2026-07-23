"""Detectors for the model-comparison harness.

A ``Detector`` turns one pano into a list of center-point detections
``(x_norm, y_norm, confidence_or_None)`` that the harness scores against the
model-agnostic ground truth (see ``rampnet/detection_eval.py``).

- ``BundleRampNetDetector`` reads RampNet's detections straight from the
  benchmark ``records.jsonl`` — free, no model load, no GPU. This is the baseline.
- ``GeminiDetector`` is **live** (google-genai; API key or Vertex+ADC): it
  reprojects the pano into rectilinear views (``equirect_tiling``), runs the model
  per view, and maps boxes back to pano coordinates.
- ``QwenDetector`` is still a **scaffold** (Qwen3-VL on Hyak): image prep,
  reprojection wiring, and box parsing are real, but the live ``_raw_detect``
  raises ``NotImplementedError``. See ``docs/model_comparison.md``.
"""
import json
import os
from collections import namedtuple

# A pano to run a detector on. image_path points at the native-res JPEG in the
# bundle's (git-ignored) panos/ dir; RampNet-from-bundle never opens it.
PanoSample = namedtuple("PanoSample", ["pano_id", "image_path", "width", "height", "meta"])


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on") if v is not None else False


def load_pano_image(path, max_edge=None):
    """Open a benchmark pano as RGB, optionally downscaling so its longest edge
    is <= ``max_edge``. Lifts PIL's decompression-bomb cap (Bend GSV panos are
    16384x8192, above the default limit)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    return img


class BundleRampNetDetector:
    """RampNet's baseline detections, read from the benchmark records.jsonl."""

    name = "rampnet"

    def __init__(self, records):
        # records: {pano_id: record_dict} from the bundle's records.jsonl.
        self.records = records

    def detect(self, sample):
        dets = self.records[sample.pano_id]["detections"]
        return [(d["x_normalized"], d["y_normalized"], d["confidence"]) for d in dets]


# --- VLM box parsing (pure, unit-tested) ------------------------------------

def boxes_from_gemini_response(resp):
    """Pull ``[{box_2d, label}, ...]`` out of a google-genai response, whether the
    SDK returned parsed schema objects (``resp.parsed``) or raw JSON text."""
    parsed = getattr(resp, "parsed", None)
    if parsed:
        return [{"box_2d": list(b.box_2d), "label": getattr(b, "label", "")} for b in parsed]
    text = getattr(resp, "text", None)
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("boxes", [])


def gemini_boxes_to_points(items):
    """Gemini returns ``box_2d = [ymin, xmin, ymax, xmax]`` normalized to 0-1000.
    Reduce each box to its normalized [0,1] center point. Confidence is None
    (Gemini bbox detection carries no calibrated score)."""
    points = []
    for it in items:
        ymin, xmin, ymax, xmax = it["box_2d"]
        cx = (xmin + xmax) / 2.0 / 1000.0
        cy = (ymin + ymax) / 2.0 / 1000.0
        points.append((cx, cy, None))
    return points


def qwen_boxes_to_points(items, img_w, img_h):
    """Qwen grounding returns absolute-pixel boxes ``bbox_2d = [x1, y1, x2, y2]``
    against the image it was shown. Normalize each box center by that image's
    width/height (the size actually sent to the model, not the native pano)."""
    points = []
    for it in items:
        x1, y1, x2, y2 = it["bbox_2d"]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        points.append((cx, cy, None))
    return points


# --- VLM detector scaffolds -------------------------------------------------

DETECTION_PROMPT = (
    "Detect every curb ramp in this street-level image. A curb ramp (curb cut) is the "
    "short sloped ramp cut into a sidewalk curb at a street corner or crossing that lets "
    "a wheelchair or stroller roll from sidewalk to street. Return one tight bounding box "
    "per curb ramp. Do not box driveways, stairs, or crosswalk paint. If there are no curb "
    "ramps, return an empty list."
)


class _VLMDetector:
    """Shared scaffold. Subclasses implement ``_raw_detect`` (the live model call)
    and ``_parse`` (provider box format -> center points, normalized within the
    image shown to the model).

    Two input modes:
      - ``tile=True`` (default, the fair input): reproject the pano into a ring of
        overlapping rectilinear views, detect in each, map centers back to pano
        coordinates, and dedup across the overlaps.
      - ``tile=False``: one downscaled whole-pano call (a lower bound; the pano is
        warped and ramps are tiny)."""

    name = "vlm"
    max_edge = 1536       # whole-pano downscale cap
    source_max_edge = 4096  # cap on the pano fed to reprojection (native can be 16k)

    def __init__(self, model_id, max_edge=None, tile=True, views=None):
        self.model_id = model_id
        if max_edge:
            self.max_edge = max_edge
        self.tile = tile
        self._views = views  # None -> equirect_tiling.default_views()

    def detect(self, sample):
        self._ensure_ready()
        if self.tile:
            return self._detect_tiled(sample)
        image = load_pano_image(sample.image_path, self.max_edge)  # whole-pano (lower bound)
        raw = self._raw_detect(image)
        return self._parse(raw, image.width, image.height)

    def _detect_tiled(self, sample):
        from equirect_tiling import (
            default_views, equirect_to_perspective, perspective_point_to_equirect, dedup_points)
        from rampnet.detection_eval import radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y

        pano = load_pano_image(sample.image_path, self.source_max_edge)
        views = self._views or default_views()
        points = []
        for view in views:
            view_img = equirect_to_perspective(pano, view)
            raw = self._raw_detect(view_img)
            # _parse returns points normalized WITHIN the view; map each back to the pano.
            for (u, v, conf) in self._parse(raw, view.width, view.height):
                x, y = perspective_point_to_equirect(u, v, view)
                points.append((x, y, conf))
        # Overlapping views see seam-straddling ramps in more than one tile; merge
        # detections closer than the match radius (with 0/1 seam wrap).
        return dedup_points(points, radius_sq_for(), PANO_SCALE_X, PANO_SCALE_Y)

    def _ensure_ready(self):
        raise NotImplementedError

    def _raw_detect(self, image):
        raise NotImplementedError

    def _parse(self, raw, img_w, img_h):
        raise NotImplementedError


class GeminiDetector(_VLMDetector):
    name = "gemini"
    max_edge = 1568  # Gemini tiles internally; a modest cap keeps token cost sane

    def __init__(self, model_id="gemini-3.6-flash", api_key=None, max_edge=None, tile=True,
                 use_vertex=None, project=None, location=None):
        super().__init__(model_id, max_edge, tile=tile)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        # Vertex AI + Application Default Credentials is the path for orgs that
        # disallow API keys. Driven by the standard google-genai env vars unless
        # overridden explicitly.
        self.use_vertex = (_truthy(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
                           if use_vertex is None else use_vertex)
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        self._client = None

    def _ensure_ready(self):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "GeminiDetector needs the `google-genai` package "
                "(pip install -r requirements-vlm.txt)") from e
        if self._client is not None:
            return
        if self.use_vertex:
            if not self.project:
                raise RuntimeError(
                    "Vertex mode needs GOOGLE_CLOUD_PROJECT (and ADC via "
                    "`gcloud auth application-default login`).")
            self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
        elif self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        else:
            raise RuntimeError(
                "No Gemini credentials. For orgs that disallow API keys, use Vertex + ADC: "
                "set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT (plus optional "
                "GOOGLE_CLOUD_LOCATION) in a git-ignored .env, and run "
                "`gcloud auth application-default login`. Otherwise set GOOGLE_API_KEY.")

    def _raw_detect(self, image):
        from google.genai import types
        from pydantic import BaseModel

        class BoundingBox(BaseModel):
            box_2d: list[int]   # [ymin, xmin, ymax, xmax], normalized 0-1000
            label: str

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=[image, DETECTION_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[BoundingBox],
                temperature=0.0,
            ),
        )
        return boxes_from_gemini_response(resp)

    def _parse(self, raw, img_w, img_h):
        return gemini_boxes_to_points(raw)


class QwenDetector(_VLMDetector):
    name = "qwen"
    max_edge = 2048  # Qwen3-VL handles large inputs; run on Hyak (A40 OOMs at native)

    def __init__(self, model_id="Qwen/Qwen3-VL", max_edge=None, tile=True):
        super().__init__(model_id, max_edge, tile=tile)

    def _ensure_ready(self):
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "QwenDetector needs `transformers` (+ qwen-vl-utils, accelerate) "
                "(pip install -r requirements-vlm.txt)") from e

    def _raw_detect(self, image):
        # TODO(increment 2, Hyak): load Qwen3-VL once (transformers AutoModelForCausalLM /
        #   Qwen3VL class + processor), run the grounding prompt, parse the JSON grounding
        #   output to [{bbox_2d:[x1,y1,x2,y2]}]. Model load belongs in _ensure_ready so it
        #   happens once per run, not per pano.
        raise NotImplementedError(
            "QwenDetector live call is scaffolded, not wired. Implement _raw_detect on Hyak "
            "(load Qwen3-VL once in _ensure_ready), then feed items to qwen_boxes_to_points "
            "via _parse. See docs/model_comparison.md.")

    def _parse(self, raw, img_w, img_h):
        return qwen_boxes_to_points(raw, img_w, img_h)


def parse_model_spec(token):
    """Parse a ``--models`` token into ``(provider, model_id_or_None)``.

    A token is either a bare provider (``rampnet`` / ``gemini`` / ``qwen``, which
    uses that provider's default model) or ``provider:model_id`` to pin a specific
    variant — e.g. ``gemini:gemini-2.5-flash`` vs ``gemini:gemini-3.6-flash`` — so
    several variants of the same provider can be compared in one run."""
    provider, _, model_id = token.partition(":")
    return provider.strip(), (model_id.strip() or None)


def build_detector(provider, model_id, records, args):
    """Instantiate a detector for one ``(provider, model_id)`` spec, returning
    ``(label, detector)``. The label is the concrete model id for VLMs (so
    variants are distinguishable in the results table) and ``rampnet`` for the
    baseline. RampNet reads from ``records``; the VLM input mode (perspective
    tiling vs whole-pano) comes from ``args``."""
    if provider == "rampnet":
        return "rampnet", BundleRampNetDetector(records)
    tile = getattr(args, "tiling", "perspective") != "none"
    if provider == "gemini":
        mid = model_id or args.gemini_model
        return mid, GeminiDetector(model_id=mid, tile=tile)
    if provider == "qwen":
        mid = model_id or args.qwen_model
        return mid, QwenDetector(model_id=mid, tile=tile)
    raise ValueError(f"unknown provider '{provider}' (choose from: rampnet, gemini, qwen)")
