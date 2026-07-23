"""Detectors for the model-comparison harness.

A ``Detector`` turns one pano into a list of center-point detections
``(x_norm, y_norm, confidence_or_None)`` that the harness scores against the
model-agnostic ground truth (see ``rampnet/detection_eval.py``).

- ``BundleRampNetDetector`` reads RampNet's detections straight from the
  benchmark ``records.jsonl`` — free, no model load, no GPU. This is the baseline.
- ``GeminiDetector`` is **live** (google-genai; API key or Vertex+ADC): it
  reprojects the pano into rectilinear views (``equirect_tiling``), runs the model
  per view, and maps boxes back to pano coordinates.
- ``QwenDetector`` is **live** (open weights via transformers; intended for a GPU
  cluster — see the Hyak runbook in ``docs/model_comparison.md``). Same tiled path
  as Gemini; the model is loaded once per run in ``_ensure_ready``.
"""
import importlib.util
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

    def prepare(self):
        pass  # nothing to load; detections come from the bundle.

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


def _first_json_blob(text):
    """Return the first balanced JSON array/object substring in ``text``.

    Qwen wraps its grounding output in a ```json fence and sometimes adds a
    sentence around it, so scan for the first ``[``/``{`` and walk to its match
    (brackets inside string literals don't count)."""
    s = text.strip()
    if "```" in s:                      # keep the body of the first fenced block
        parts = s.split("```")
        if len(parts) >= 3:
            body = parts[1].lstrip()
            s = body[4:] if body[:4].lower() == "json" else body
    start = next((i for i, ch in enumerate(s) if ch in "[{"), None)
    if start is None:
        return None
    opener = s[start]
    closer = "]" if opener == "[" else "}"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def boxes_from_qwen_text(text):
    """Pull ``[{bbox_2d, label}, ...]`` out of a Qwen grounding completion.

    Deliberately tolerant — an open model has no response-schema equivalent, so it
    may fence its JSON, wrap it in prose, return a bare object, or emit a
    malformed item. Anything without a 4-number box is dropped rather than
    crashing a 1,400-call run."""
    if not text:
        return []
    blob = _first_json_blob(text)
    if blob is None:
        return []
    try:
        data = json.loads(blob)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("boxes") or data.get("objects") or [data]
    items = []
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict):
            continue
        box = it.get("bbox_2d", it.get("bbox"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            box = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        items.append({"bbox_2d": box, "label": it.get("label", "")})
    return items


def qwen_boxes_to_points(items, img_w, img_h, coord_space="norm1000"):
    """Reduce Qwen grounding boxes ``bbox_2d = [x1, y1, x2, y2]`` to normalized
    [0,1] center points. Confidence is None (grounding carries no score).

    Two conventions exist across the family, and at a ~1000px view size their
    outputs look nearly identical, so the caller states which rather than guessing:

    - ``norm1000`` (**Qwen3-VL**): coordinates are already normalized to 0-1000
      (the cookbook rescales with ``bbox_2d[0] / 1000 * width``). Being
      resolution-independent, the processor's smart-resize cannot shift them.
    - ``pixels`` (Qwen2/2.5-VL): absolute pixels of the image the processor
      actually fed the model, so normalize by that image's width/height."""
    if coord_space not in ("norm1000", "pixels"):
        raise ValueError(f"unknown coord_space {coord_space!r} (expected norm1000 | pixels)")
    sx, sy = (1000.0, 1000.0) if coord_space == "norm1000" else (float(img_w), float(img_h))
    points = []
    for it in items:
        x1, y1, x2, y2 = it["bbox_2d"]
        cx = (x1 + x2) / 2.0 / sx
        cy = (y1 + y2) / 2.0 / sy
        points.append((cx, cy, None))
    return points


# --- VLM detectors ----------------------------------------------------------

DETECTION_PROMPT = (
    "Detect every curb ramp in this street-level image. A curb ramp (curb cut) is the "
    "short sloped ramp cut into a sidewalk curb at a street corner or crossing that lets "
    "a wheelchair or stroller roll from sidewalk to street. Return one tight bounding box "
    "per curb ramp. Do not box driveways, stairs, or crosswalk paint. If there are no curb "
    "ramps, return an empty list."
)

# Gemini gets its output shape from a response_schema; an open model has to be
# told in the prompt. Same detection task, so the two stay word-for-word identical
# up to this suffix.
QWEN_JSON_INSTRUCTION = (
    ' Respond with JSON only: a list of {"bbox_2d": [x1, y1, x2, y2], "label": "curb ramp"}.'
)
QWEN_PROMPT = DETECTION_PROMPT + QWEN_JSON_INSTRUCTION


class _VLMDetector:
    """Shared base. Subclasses implement ``_raw_detect`` (the live model call)
    and ``_parse`` (provider box format -> center points, normalized within the
    image shown to the model).

    Two input modes:
      - ``tile=True`` (default, the fair input): reproject the pano into a ring of
        overlapping rectilinear views, detect in each, map centers back to pano
        coordinates, and dedup across the overlaps.
      - ``tile=False``: one downscaled whole-pano call (a lower bound; the pano is
        warped and ramps are tiny)."""

    name = "vlm"
    prompt = DETECTION_PROMPT  # subclasses override when the provider needs more
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

    def prepare(self):
        """Build the client / load the model up front so credential, dependency,
        or not-yet-wired errors surface once (failing the model fast) instead of
        once per pano."""
        self._ensure_ready()

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

    def signature(self):
        """A stable description of everything that affects this detector's output,
        used as the detection cache key. Changing the model, tiling rig, or prompt
        invalidates cached detections."""
        from equirect_tiling import default_views
        views = self._views or (default_views() if self.tile else None)
        return {
            "provider": self.name,
            "model_id": self.model_id,
            "tile": self.tile,
            "max_edge": self.max_edge,
            "source_max_edge": self.source_max_edge,
            "views": [list(v) for v in views] if views else None,
            "prompt": self.prompt,
        }


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
        # Default to `global`, not a region: the newest flash ids (the default
        # gemini-3.6-flash) are served only there; regional endpoints lag and 404
        # on them. Override with GOOGLE_CLOUD_LOCATION only for a data-residency
        # policy (benchmark imagery is public GSV/Mapillary). See docs/model_comparison.md.
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        self._client = None

    def _ensure_ready(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "GeminiDetector needs the `google-genai` package "
                "(pip install -r requirements-vlm.txt)") from e
        if self._client is not None:
            return
        # Explicit retry policy for the ~hundreds of calls a full-city run makes:
        # exponential backoff + jitter on the transient/rate-limit status codes.
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(
            attempts=5, initial_delay=1.0, max_delay=30.0, exp_base=2.0, jitter=1.0,
            http_status_codes=[408, 429, 500, 502, 503, 504]))
        if self.use_vertex:
            if not self.project:
                raise RuntimeError(
                    "Vertex mode needs GOOGLE_CLOUD_PROJECT (and ADC via "
                    "`gcloud auth application-default login`).")
            self._client = genai.Client(vertexai=True, project=self.project,
                                        location=self.location, http_options=http_options)
        elif self.api_key:
            self._client = genai.Client(api_key=self.api_key, http_options=http_options)
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
            contents=[image, self.prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[BoundingBox],
                temperature=0.0,
            ),
        )
        return boxes_from_gemini_response(resp)

    def _parse(self, raw, img_w, img_h):
        return gemini_boxes_to_points(raw)


def infer_qwen_coord_space(model_id):
    """Which box convention a Qwen checkpoint emits (see ``qwen_boxes_to_points``).

    Qwen3-VL normalizes to 0-1000; Qwen2/2.5-VL emit absolute pixels. Unknown ids
    get the Qwen3+ convention — overridable with ``--qwen-coord-space``."""
    mid = (model_id or "").lower()
    if "qwen2" in mid:
        return "pixels"
    return "norm1000"


class QwenDetector(_VLMDetector):
    """Qwen3-VL grounding via transformers (open weights, local GPU).

    The checkpoint is loaded once per run in ``_ensure_ready`` — 8B is ~16GB in
    bf16 and 32B ~64GB, so this belongs on a cluster GPU (see the Hyak runbook in
    docs/model_comparison.md), not the dev box. Detections it produces are written
    to the same ``.model_cache`` as every other model, and that cache key contains
    nothing machine-specific, so a cache produced on the cluster scores locally."""

    name = "qwen"
    prompt = QWEN_PROMPT
    max_edge = 2048  # whole-pano mode only; tiled views are rendered at 1024

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", max_edge=None, tile=True,
                 coord_space=None, max_new_tokens=1024):
        super().__init__(model_id, max_edge, tile=tile)
        self.coord_space = coord_space or infer_qwen_coord_space(model_id)
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_ready(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "QwenDetector needs `torch` and `transformers>=4.57` "
                "(pip install -r requirements-vlm.txt)") from e
        try:
            from transformers import AutoModelForImageTextToText as model_cls
        except ImportError:  # older transformers: reach for the Qwen3-VL class directly
            from transformers import Qwen3VLForConditionalGeneration as model_cls

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        # device_map="auto" shards a checkpoint too big for one GPU (32B needs two),
        # but it needs accelerate; without it fall back to a single device.
        if importlib.util.find_spec("accelerate") is not None:
            model = model_cls.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model_cls.from_pretrained(self.model_id, dtype="auto").to(device)
        self._model = model.eval()

    def _raw_detect(self, image):
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt},
        ]}]
        # Qwen3-VL's chat template accepts PIL images directly, so qwen-vl-utils
        # isn't needed. Greedy decoding mirrors Gemini's temperature=0.
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        # generate() returns prompt + completion; keep only the completion.
        completion = out[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(completion, skip_special_tokens=True)[0]
        return boxes_from_qwen_text(text)

    def _parse(self, raw, img_w, img_h):
        return qwen_boxes_to_points(raw, img_w, img_h, coord_space=self.coord_space)

    def signature(self):
        # Extends the base signature. Gemini's stays byte-identical, so the
        # detections already paid for keep hitting the cache.
        sig = super().signature()
        sig.update({"coord_space": self.coord_space, "max_new_tokens": self.max_new_tokens})
        return sig


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
        coord_space = getattr(args, "qwen_coord_space", "auto")
        return mid, QwenDetector(model_id=mid, tile=tile,
                                 coord_space=None if coord_space == "auto" else coord_space)
    raise ValueError(f"unknown provider '{provider}' (choose from: rampnet, gemini, qwen)")
