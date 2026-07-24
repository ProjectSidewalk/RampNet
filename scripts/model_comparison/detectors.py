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
- ``OwlV2Detector`` / ``GroundingDinoDetector`` are **live** open-vocabulary
  *detectors* (not chat models): text query in, boxes **with calibrated scores**
  out. That confidence is carried through the whole harness, which is what makes
  AP / PR curves and threshold sweeps possible for a non-RampNet model.
- ``MolmoDetector`` is **live** and emits **points**, not boxes — RampNet's native
  output format, so it avoids the box->center reduction every other VLM needs.
"""
import importlib.util
import json
import os
import re
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


# --- open-vocabulary detector parsing (pure, unit-tested) -------------------

def zero_shot_results_to_boxes(result, threshold=None):
    """Normalize a transformers ``post_process_grounded_object_detection`` result
    (one image) into ``[{"box": [x1, y1, x2, y2], "score": float, "label": str}]``.

    The result's values are torch tensors in a live run and plain lists in tests,
    so everything goes through ``_as_list``. Boxes are absolute pixels in the frame
    named by the ``target_sizes`` that was passed (see ``owlv2_target_size``)."""
    boxes = _as_list(result.get("boxes"))
    scores = _as_list(result.get("scores"))
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels")
    labels = _as_list(labels)
    items = []
    for i, box in enumerate(boxes):
        box = [float(v) for v in _as_list(box)]
        if len(box) != 4:
            continue
        score = float(scores[i]) if i < len(scores) else None
        if threshold is not None and score is not None and score < threshold:
            continue
        label = labels[i] if i < len(labels) else ""
        items.append({"box": box, "score": score, "label": str(label)})
    return items


def _as_list(v):
    """Tensor / ndarray / sequence -> plain list (empty for None)."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v) if isinstance(v, (list, tuple)) else [v]


def owlv2_target_size(img_w, img_h):
    """The frame OWLv2's boxes live in: ``(side, side)`` with ``side = max(w, h)``.

    OWLv2's image processor **pads the image to a square before resizing**, adding
    the padding at the bottom and right, so the model's boxes are relative to that
    square and the original image sits in its top-left corner. Dividing by the
    *original* width/height (``pixel_boxes_to_points``) is therefore what recovers
    normalized in-image coordinates — and a box may legitimately land outside them,
    in the pad.

    Current transformers already scales OWLv2 boxes by ``max(h, w)`` on both axes
    inside ``_scale_boxes`` ("for owlv2 image is padded to max size"), so passing
    the square and passing the image's own ``(h, w)`` are equivalent there —
    verified empirically on a 2:1 crop, where both put the top box at y 0.815 (true
    position 0.817). Passing the square is still what this returns: it is also
    correct under the older per-axis scaling the model card's workaround was
    written for, and it states the frame the caller is normalizing against instead
    of relying on a library internal. Square views (the default rig) are
    unaffected either way; whole-pano mode (2:1) is the only place it could bite."""
    side = max(int(img_w), int(img_h))
    return (side, side)


def pixel_boxes_to_points(items, img_w, img_h):
    """Reduce ``{"box": [x1, y1, x2, y2] (pixels), "score": s}`` to normalized [0,1]
    center points ``(cx, cy, score)``.

    Unlike the chat VLMs, the score is a real per-box confidence and is **carried
    through** — it is what lets ``score_pano`` rank predictions and the harness
    report AP / PR curves. Centers outside the image are dropped: OWLv2 can place a
    box in the padded region, which is not part of the picture."""
    points = []
    for it in items:
        x1, y1, x2, y2 = it["box"]
        cx = (x1 + x2) / 2.0 / float(img_w)
        cy = (y1 + y2) / 2.0 / float(img_h)
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            points.append((cx, cy, it.get("score")))
    return points


# --- Molmo point parsing (pure, unit-tested) --------------------------------

# Molmo emits points as XML-ish tags, and the two generations disagree on both the
# tag shape and the coordinate scale:
#   Molmo 1:  <point x="35.4" y="61.2" alt="...">...</point>
#             <points x1="10.5" y1="20.0" x2="30.1" y2="40.2" ...>...</points>
#             -> coordinates are PERCENTAGES of the image (0-100).
#   Molmo 2:  <points coords="0 354 612; 1 700 480"/>  (triplets: object_id x y)
#             -> coordinates are scaled by 1000, per the model card's own regex.
# The two are distinguishable by syntax (a `coords` attribute vs `x`/`y` attributes),
# which is why the scale can be inferred here — unlike Qwen's two box conventions,
# which were syntactically identical and had to be chosen by model id.
_MOLMO_TAG_RE = re.compile(r"<(point|points)\b([^>]*)>", re.IGNORECASE)
_MOLMO_ATTR_RE = re.compile(r'([A-Za-z_]\w*)\s*=\s*"([^"]*)"')
_MOLMO_XY_RE = re.compile(r"^([xy])(\d*)$")
_MOLMO_TRIPLE_RE = re.compile(r"([0-9]+) ([0-9.]+) ([0-9.]+)")

MOLMO_ATTR_SCALE = 100.0    # Molmo 1: percent of the image
MOLMO_COORDS_SCALE = 1000.0  # Molmo 2: the card's "coordinates are scaled by 1000"


def molmo_points_from_text(text, coord_scale=None):
    """Parse a Molmo completion into ``[{"point": [x, y], "label": str}]`` with
    **normalized [0,1]** coordinates.

    ``coord_scale=None`` (default) infers the divisor from the tag syntax, as
    documented above; pass 100.0 / 1000.0 to force one. Points outside [0,1] after
    scaling are dropped — the reference implementation on the model card does the
    same, and it makes a wrong scale show up as "almost nothing detected" instead
    of a silent systematic offset."""
    if not text:
        return []
    items = []
    for tag, attr_text in _MOLMO_TAG_RE.findall(text):
        attrs = dict(_MOLMO_ATTR_RE.findall(attr_text))
        label = attrs.get("alt", "")
        if "coords" in attrs:                       # Molmo 2: "<id> <x> <y>" triplets
            scale = coord_scale or MOLMO_COORDS_SCALE
            pairs = [(m.group(2), m.group(3)) for m in _MOLMO_TRIPLE_RE.finditer(attrs["coords"])]
        else:                                       # Molmo 1: x/y (or x1/y1, x2/y2 ...)
            scale = coord_scale or MOLMO_ATTR_SCALE
            xs, ys = {}, {}
            for key, val in attrs.items():
                m = _MOLMO_XY_RE.match(key)
                if m:
                    (xs if m.group(1) == "x" else ys)[m.group(2)] = val
            # Suffixes are "", "1", "2", ...; sort numerically so a 10+-point tag
            # doesn't come back as 1, 10, 2 (order is cosmetic — pairing is by key).
            order = sorted(xs, key=lambda k: (len(k), k))
            pairs = [(xs[k], ys[k]) for k in order if k in ys]
        for xs_, ys_ in pairs:
            try:
                x, y = float(xs_) / scale, float(ys_) / scale
            except ValueError:
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                items.append({"point": [x, y], "label": label})
    return items


def molmo_token_points_to_items(points, img_w, img_h):
    """MolmoPoint's decoded points -> the same ``{"point": [x, y]}`` items.

    ``model.extract_image_points`` returns rows whose **last two** values are pixel
    coordinates in the input image; the leading ids are documented inconsistently
    on the model card (``[object_id, image_num, x, y]`` in the code comment,
    ``(image_id, object_id, x, y)`` in the prose), so only the tail is read."""
    items = []
    for row in points or []:
        row = _as_list(row)
        if len(row) < 2:
            continue
        try:
            x, y = float(row[-2]) / float(img_w), float(row[-1]) / float(img_h)
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            items.append({"point": [x, y], "label": ""})
    return items


def points_to_center_points(items):
    """Molmo point items -> ``(x, y, None)`` triples. Molmo carries no per-point
    score, so these models get an operating point but no PR curve."""
    return [(it["point"][0], it["point"][1], None) for it in items]


# --- VLM detectors ----------------------------------------------------------

# The one definition of the target class, shared verbatim by every prompted model so
# they are asked for the same thing. Changing it re-bills every cached detection.
CURB_RAMP_DEFINITION = (
    "A curb ramp (curb cut) is the short sloped ramp cut into a sidewalk curb at a street "
    "corner or crossing that lets a wheelchair or stroller roll from sidewalk to street."
)

DETECTION_PROMPT = (
    "Detect every curb ramp in this street-level image. " + CURB_RAMP_DEFINITION +
    " Return one tight bounding box per curb ramp. Do not box driveways, stairs, or "
    "crosswalk paint. If there are no curb ramps, return an empty list."
)

# Gemini gets its output shape from a response_schema; an open model has to be
# told in the prompt. Same detection task, so the two stay word-for-word identical
# up to this suffix.
QWEN_JSON_INSTRUCTION = (
    ' Respond with JSON only: a list of {"bbox_2d": [x1, y1, x2, y2], "label": "curb ramp"}.'
)
QWEN_PROMPT = DETECTION_PROMPT + QWEN_JSON_INSTRUCTION

# Open-vocabulary detectors take a text *query*, not an instruction. They are not
# chat models: OWLv2 is CLIP-based and responds to a "a photo of a ..." template,
# Grounding DINO expects lowercase, period-terminated category text. The paragraph
# above would be truncated by their text encoders, so the class name is the prompt.
OWLV2_QUERY = "a photo of a curb ramp"
GDINO_QUERY = "curb ramp."

# Molmo points instead of boxing, so the same definition gets a pointing verb.
MOLMO_PROMPT = (
    "Point to every curb ramp in this street-level image. " + CURB_RAMP_DEFINITION +
    " Put one point at the center of each curb ramp. Do not point at driveways, stairs, "
    "or crosswalk paint. If there are no curb ramps, say so."
)


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


class _ZeroShotDetector(_VLMDetector):
    """Open-vocabulary *detector* (OWLv2, Grounding DINO) via transformers.

    The important difference from the chat VLMs: these are trained to detect, and
    every box carries a **calibrated score**. The harness threads that score all the
    way through (``pixel_boxes_to_points`` -> ``dedup_points`` -> ``score_pano``),
    so these models get AP, a PR curve, and a threshold sweep — the tunable
    operating range that a chat VLM pinned at one point cannot offer.

    ``score_threshold`` is a **cache floor**, not the operating point: detections are
    computed once down to a low score and every higher threshold is then a free
    re-score of the cache. Lowering it invalidates the cache (it is in the
    signature); raising the *reported* threshold does not (``--op-threshold``)."""

    name = "zeroshot"
    query = "object"
    score_threshold = 0.05
    max_edge = 1536       # whole-pano mode only; tiled views render at 1024

    def __init__(self, model_id, query=None, score_threshold=None, max_edge=None,
                 tile=True, views=None):
        super().__init__(model_id, max_edge, tile=tile, views=views)
        self.query = query or self.query
        # The text query *is* the prompt for these models, so the base signature's
        # "prompt" key keys the cache on it.
        self.prompt = self.query
        if score_threshold is not None:
            self.score_threshold = float(score_threshold)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_ready(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} needs `torch` and `transformers` "
                "(pip install -r requirements-vlm.txt)") from e
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(self._device).eval()

    def _raw_detect(self, image):
        import torch
        inputs = self._processor(images=image, text=self._text_input(),
                                 return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._post_process(outputs, inputs, image)
        return zero_shot_results_to_boxes(results[0])

    def _parse(self, raw, img_w, img_h):
        return pixel_boxes_to_points(raw, img_w, img_h)

    def _text_input(self):
        raise NotImplementedError

    def _post_process(self, outputs, inputs, image):
        raise NotImplementedError

    def _post_process_fn(self):
        """``post_process_grounded_object_detection`` is the current name; older
        transformers only had ``post_process_object_detection`` for OWLv2."""
        fn = getattr(self._processor, "post_process_grounded_object_detection", None)
        return fn or self._processor.post_process_object_detection

    def signature(self):
        sig = super().signature()
        sig.update({"query": self.query, "score_threshold": self.score_threshold})
        return sig


class OwlV2Detector(_ZeroShotDetector):
    """OWLv2 (``google/owlv2-large-patch14-ensemble``) — text-prompted detection."""

    name = "owlv2"
    query = OWLV2_QUERY

    def __init__(self, model_id="google/owlv2-large-patch14-ensemble", **kw):
        super().__init__(model_id, **kw)

    def _text_input(self):
        return [[self.query]]           # batch of 1 image, 1 query

    def _post_process(self, outputs, inputs, image):
        # target_sizes must be the PADDED SQUARE, not the image — see owlv2_target_size.
        return self._post_process_fn()(
            outputs=outputs, threshold=self.score_threshold,
            target_sizes=[owlv2_target_size(image.width, image.height)])


class GroundingDinoDetector(_ZeroShotDetector):
    """Grounding DINO (``IDEA-Research/grounding-dino-base``).

    ``text_threshold`` gates how strongly a box must align with the query tokens;
    ``score_threshold`` gates box confidence. Both are in the signature."""

    name = "gdino"
    query = GDINO_QUERY
    text_threshold = 0.2

    def __init__(self, model_id="IDEA-Research/grounding-dino-base", text_threshold=None, **kw):
        super().__init__(model_id, **kw)
        if text_threshold is not None:
            self.text_threshold = float(text_threshold)

    def _text_input(self):
        return self.query               # "a. b." category text, lowercase

    def _post_process(self, outputs, inputs, image):
        # Grounding DINO does not pad to square, so the image's own (h, w) is right.
        return self._post_process_fn()(
            outputs=outputs, input_ids=inputs["input_ids"],
            threshold=self.score_threshold, text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)])

    def signature(self):
        sig = super().signature()
        sig["text_threshold"] = self.text_threshold
        return sig


def infer_molmo_mode(model_id):
    """Which decoding path a Molmo checkpoint needs.

    ``MolmoPoint`` emits points as **special tokens** that only the model can decode
    (``extract_image_points``, with metadata from the processor); every other Molmo
    writes them as XML in plain text. Unknown ids get the text path."""
    return "point_tokens" if "molmopoint" in (model_id or "").lower() else "xml"


class MolmoDetector(_VLMDetector):
    """Ai2 Molmo — the one model here whose native output is **points**, not boxes.

    Every other VLM in this harness is scored by the center of a box it drew, a
    documented reduction (``docs/model_comparison.md``). Molmo removes it: it points
    where RampNet points, so the comparison is like-for-like. There is no per-point
    score, so Molmo gets an operating point but no PR curve.

    8B in bf16 is ~16 GB — a cluster model, like Qwen (see the Hyak runbook)."""

    name = "molmo"
    prompt = MOLMO_PROMPT
    max_edge = 2048       # whole-pano mode only; tiled views render at 1024

    def __init__(self, model_id="allenai/Molmo2-8B", max_edge=None, tile=True,
                 coord_scale=None, mode=None, max_new_tokens=512):
        super().__init__(model_id, max_edge, tile=tile)
        self.coord_scale = float(coord_scale) if coord_scale else None
        self.mode = mode or infer_molmo_mode(model_id)
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_ready(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (imported for the same clear error as Qwen)
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "MolmoDetector needs `torch` and `transformers` "
                "(pip install -r requirements-vlm.txt)") from e
        # Molmo ships custom modeling/processing code on the Hub; both classes need
        # trust_remote_code. padding_side="left" is what the MolmoPoint card uses.
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, padding_side="left")
        kw = dict(trust_remote_code=True, dtype="auto")
        if importlib.util.find_spec("accelerate") is not None:
            kw["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw)
        if "device_map" not in kw:
            import torch
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.eval()

    def _messages(self, image):
        return [{"role": "user", "content": [
            {"type": "text", "text": self.prompt},
            {"type": "image", "image": image},
        ]}]

    def _raw_detect(self, image):
        import torch
        want_meta = self.mode == "point_tokens"
        template_kw = {"return_pointing_metadata": True} if want_meta else {}
        inputs = self._processor.apply_chat_template(
            self._messages(image), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **template_kw)
        metadata = inputs.pop("metadata", None) if want_meta else None
        device = getattr(self._model, "device", "cpu")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kw = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if want_meta:
            # Constrains decoding so point tokens can only be emitted validly.
            gen_kw["logits_processor"] = self._model.build_logit_processor_from_inputs(inputs)
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kw)
        completion = out[:, inputs["input_ids"].shape[1]:]

        if not want_meta:
            text = self._processor.tokenizer.decode(completion[0], skip_special_tokens=True)
            return molmo_points_from_text(text, coord_scale=self.coord_scale)
        # Point tokens survive only with skip_special_tokens=False.
        text = self._processor.post_process_image_text_to_text(
            completion, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        points = self._model.extract_image_points(
            text, metadata["token_pooling"], metadata["subpatch_mapping"],
            metadata["image_sizes"])
        return molmo_token_points_to_items(points, image.width, image.height)

    def _parse(self, raw, img_w, img_h):
        # Both modes already produce normalized in-view points.
        return points_to_center_points(raw)

    def signature(self):
        sig = super().signature()
        sig.update({"coord_scale": self.coord_scale, "mode": self.mode,
                    "max_new_tokens": self.max_new_tokens})
        return sig


def parse_model_spec(token):
    """Parse a ``--models`` token into ``(provider, model_id_or_None)``.

    A token is either a bare provider (``rampnet`` / ``gemini`` / ``qwen`` /
    ``owlv2`` / ``gdino`` / ``molmo``, which uses that provider's default model) or
    ``provider:model_id`` to pin a specific variant — e.g. ``gemini:gemini-2.5-flash``
    vs ``gemini:gemini-3.6-flash`` — so several variants of the same provider can be
    compared in one run."""
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
    if provider == "owlv2":
        mid = model_id or getattr(args, "owlv2_model", None) or "google/owlv2-large-patch14-ensemble"
        return mid, OwlV2Detector(model_id=mid, tile=tile,
                                  query=getattr(args, "owlv2_query", None),
                                  score_threshold=getattr(args, "score_threshold", None))
    if provider == "gdino":
        mid = model_id or getattr(args, "gdino_model", None) or "IDEA-Research/grounding-dino-base"
        return mid, GroundingDinoDetector(model_id=mid, tile=tile,
                                          query=getattr(args, "gdino_query", None),
                                          score_threshold=getattr(args, "score_threshold", None),
                                          text_threshold=getattr(args, "gdino_text_threshold", None))
    if provider == "molmo":
        mid = model_id or getattr(args, "molmo_model", None) or "allenai/Molmo2-8B"
        scale = getattr(args, "molmo_coord_scale", "auto")
        return mid, MolmoDetector(model_id=mid, tile=tile,
                                  coord_scale=None if scale in (None, "auto") else float(scale))
    raise ValueError(f"unknown provider '{provider}' (choose from: rampnet, gemini, qwen, "
                     "owlv2, gdino, molmo)")
