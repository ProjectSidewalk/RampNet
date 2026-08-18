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
- ``YoloDetector`` is the one **supervised** model here: an Ultralytics YOLO box
  detector *trained on the RampNet dataset* (the architecture-vs-data baseline,
  issue #51). ``--yolo-model`` points at trained weights, not an HF id, so its
  signature also hashes the weights file. Boxes carry a calibrated score, so it
  gets AP / PR / sweep like the open-vocab detectors; tiled by default, with
  ``--tiling none`` as the whole-pano ablation.
"""
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, namedtuple

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


def masks_to_points(seg, prob, class_ids, min_area_px=16):
    """Reduce a semantic segmentation map to normalized center points (#126).

    ``seg`` is an ``(H, W)`` array of class ids and ``prob`` the matching ``(H, W)``
    per-pixel confidence for whichever class won there. One point per **connected
    component** of the selected classes, at its centroid, scored by the mean
    confidence over the component.

    Three choices worth stating, because each is a place this could silently be
    wrong rather than fail:

    * **Connected components, not one blob per class.** A semantic segmenter has no
      notion of instances, and a panorama view routinely contains several ramps. One
      point per class per view would cap recall at 1 and put that point in the empty
      space between two real ramps.
    * **The score is carried through**, like OWLv2's and YOLO's and unlike the chat
      VLMs'. ``aggregate`` only computes AP and a PR curve when *every* prediction
      has a confidence, and the open question this arm exists to answer (#126) is
      about the precision side, which is exactly what a curve shows.
    * **``min_area_px`` is a cache floor, not the operating point** — the same
      doctrine as ``score_threshold`` on the open-vocabulary detectors. It exists to
      stop single-pixel speckle becoming detections; the operating point is a
      re-score of the cache (``--op-threshold``, ``--sweep``).

    Components are found over the union of ``class_ids``, so a ramp the model splits
    between "Curb Cut" and the adjacent "Curb" yields one point rather than two when
    both classes are selected. That union is the second, separately-labelled arm.

    Returns points normalized to the segmentation map's own frame, which the caller
    post-processes to the view size — so these are already view-normalized, the same
    contract every other ``_parse`` obeys.
    """
    import numpy as np
    from skimage.measure import label as cc_label, regionprops

    seg = np.asarray(seg)
    prob = np.asarray(prob, dtype=float)
    h, w = seg.shape[:2]

    selected = np.isin(seg, list(class_ids))
    if not selected.any():
        return []
    points = []
    # connectivity=2 (8-connected): a ramp seen at a shallow angle can be a single
    # diagonal chain of pixels, and 4-connectivity would split it into several.
    for region in regionprops(cc_label(selected, connectivity=2),
                              intensity_image=prob):
        if region.area < min_area_px:
            continue
        cy, cx = region.centroid          # regionprops is (row, col)
        # `mean_intensity` is deprecated for removal in scikit-image 2.0 and
        # `intensity_mean` does not exist before 0.19; requirements.txt pins neither.
        score = (region.intensity_mean if hasattr(region, "intensity_mean")
                 else region.mean_intensity)
        points.append((cx / w, cy / h, float(score)))
    return points


def yolo_results_to_boxes(result, threshold=None):
    """Normalize one Ultralytics ``Results`` into ``[{"box": [x1, y1, x2, y2]
    (pixels), "score": float}]`` — the same shape ``pixel_boxes_to_points`` consumes.

    ``result.boxes.xyxy`` are absolute pixels in the frame the model was shown and
    ``result.boxes.conf`` are calibrated per-box confidences; both are torch tensors
    in a live run and plain lists in tests, so they go through ``_as_list``. The
    score is **carried through** — it is what lets YOLO get AP / PR / a sweep."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _as_list(getattr(boxes, "xyxy", None))
    conf = _as_list(getattr(boxes, "conf", None))
    items = []
    for i, box in enumerate(xyxy):
        box = [float(v) for v in _as_list(box)]
        if len(box) != 4:
            continue
        score = float(conf[i]) if i < len(conf) else None
        if threshold is not None and score is not None and score < threshold:
            continue
        items.append({"box": box, "score": score})
    return items


# --- Molmo point parsing (pure, unit-tested) --------------------------------

# Molmo emits points as XML-ish tags, and the two generations disagree on both the
# tag shape and the coordinate scale:
#   Molmo 1:  <point x="35.4" y="61.2" alt="...">...</point>
#             <points x1="10.5" y1="20.0" x2="30.1" y2="40.2" ...>...</points>
#             -> coordinates are PERCENTAGES of the image (0-100).
#   Molmo 2:  <points coords="1 1 308 305 2 752 377">curb ramp</points>
#             -> a leading IMAGE INDEX, then (point_id, x, y) triplets, scaled by 1000.
#             The leading index is easy to miss and costly: consuming it as a point
#             id shifts every coordinate one slot left, which pins every point to
#             x~0. That is what the first real Molmo run did until the
#             dump_detections overlay showed a column of crosshairs on the left
#             edge. Verified against Molmo2-8B on 2026-07-23; the token count is
#             1 mod 3 (7 tokens for 2 points, 13 for 4) and the ids run 1,2,3,...
# The two are distinguishable by syntax (a `coords` attribute vs `x`/`y` attributes),
# which is why the scale can be inferred here — unlike Qwen's two box conventions,
# which were syntactically identical and had to be chosen by model id.
_MOLMO_TAG_RE = re.compile(r"<(point|points)\b([^>]*)>", re.IGNORECASE)
_MOLMO_ATTR_RE = re.compile(r'([A-Za-z_]\w*)\s*=\s*"([^"]*)"')
_MOLMO_XY_RE = re.compile(r"^([xy])(\d*)$")


def _molmo_coord_pairs(coords):
    """``"1 1 308 305 2 752 377"`` -> ``[("308", "305"), ("752", "377")]``.

    Chunk into ``(id, x, y)`` triplets, dropping the leading image index when one
    is present. Positional chunking rather than the model card's
    ``([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})`` regex: that pattern happens to
    resynchronize past the leading index only because it demands 3-4 digits, so it
    silently drops any point in the leftmost/topmost 10% of a view (x or y < 100).

    Whether the leading index is there is decided by the point-id column (the ids
    run 1, 2, 3, ... — model card, confirmed on real output), not by token count
    alone: a generation truncated mid-triplet by max_new_tokens shifts the count
    by one or two, and misaligned pairs would divide small point ids by 1000 into
    in-frame garbage pinned near x~0 — a quiet failure, exactly what this parser
    must never produce. Trying the with-index alignment first is safe: for the
    without-index alignment to be mistaken for it, a real x coordinate would have
    to equal the next expected id (x of 0.001-0.009 of a view), which does not
    occur. The token-count heuristic remains as the fallback for id sequences
    that don't read 1..k.

    Single images only — a multi-frame video response interleaves several frame
    ids, which this harness never requests."""
    nums = re.split(r"[\s;,:\t]+", coords.strip())
    nums = [n for n in nums if n]
    for lead in (1, 0):         # with the image index, then without
        tail = nums[lead:]
        n_full = len(tail) // 3
        if n_full and tail[0::3][:n_full] == [str(i + 1) for i in range(n_full)]:
            return [(tail[i + 1], tail[i + 2]) for i in range(0, 3 * n_full, 3)]
    if len(nums) % 3 == 1:      # leading image/frame index
        nums = nums[1:]
    return [(nums[i + 1], nums[i + 2]) for i in range(0, len(nums) - 2, 3)]

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
        if "coords" in attrs:                       # Molmo 2: index + (id, x, y) triplets
            scale = coord_scale or MOLMO_COORDS_SCALE
            pairs = _molmo_coord_pairs(attrs["coords"])
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
        # Paid providers accumulate per-call token usage here so every run's
        # cost is recorded (compare.py --usage-log); None = provider doesn't
        # report usage (local GPU models are free in API terms).
        self.usage = None
        # {resolved build -> calls}. `model_id` is what we ASKED for, and for a
        # hosted model that is an alias the provider is free to move; this is what
        # actually answered. Deliberately NOT in signature() -- that feeds
        # cache_key, so adding it would miss every already-paid cached detection
        # and re-bill the run (test_gemini_cache_key_is_frozen guards exactly
        # that). None = provider reports no build. See #121.
        self.model_versions = None

    # The key contract for `usage`, in one place. Every paid provider reports
    # input and output; almost none report thinking separately, so a reader must
    # never assume that key exists (compare.py uses .get). Subclasses start the
    # dict with init_usage() and add to it with accumulate_usage() rather than
    # inventing their own shape.
    # cache_* are their own billed SKUs and are EXCLUDED from input_tokens on at
    # least the Anthropic path, so a run that enables prompt caching would report
    # a cost with the cached half missing. Zero for every provider that does not
    # report them, which is all of them today.
    USAGE_KEYS = ("calls", "input_tokens", "output_tokens", "thoughts_tokens",
                  "cache_read_input_tokens", "cache_write_input_tokens")

    def init_usage(self):
        self.usage = dict.fromkeys(self.USAGE_KEYS, 0)

    def record_model_version(self, version):
        """Note which build answered this call, and say so if it changes mid-run.

        A rotation partway through a leg is the case nobody would otherwise
        notice: the alias, the signature and the cache key are all unchanged, so
        two models' detections land in one published file indistinguishably."""
        if not version:
            return
        if self.model_versions is None:
            self.model_versions = Counter()
        seen = set(self.model_versions)
        self.model_versions[version] += 1
        if seen and version not in seen:
            print(f"[{self.model_id}] WARNING: the resolved model build changed "
                  f"mid-run ({', '.join(sorted(seen))} -> {version}). This leg's "
                  f"detections come from more than one model, and nothing in the "
                  f"cache key or the published signature distinguishes them.")

    def accumulate_usage(self, input_tokens, output_tokens, thoughts_tokens=0,
                         cache_read_tokens=0, cache_write_tokens=0):
        """Add one paid call to the running total. ``output_tokens`` must ALREADY
        include thinking, because that is how it bills; ``thoughts_tokens`` is
        carried alongside only because it is invisible in the response text and
        dominates output cost. ``cache_*`` are separate SKUs, not a subset of
        ``input_tokens`` -- pass them only if the provider reports them that way."""
        self.usage["calls"] += 1
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["thoughts_tokens"] += thoughts_tokens
        self.usage["cache_read_input_tokens"] += cache_read_tokens
        self.usage["cache_write_input_tokens"] += cache_write_tokens

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
        self.init_usage()
        self._usage_warned = False   # warn once per run, not once per call

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
        self._record_usage(resp)
        return boxes_from_gemini_response(resp)

    def _record_usage(self, resp):
        """Accumulate this call's token counts so the run's cost is a recorded
        fact, not a reconstruction. Thinking tokens are billed as output, so
        output_tokens already includes them; thoughts_tokens is kept separately
        because it is invisible in the response text and dominates output cost.
        Cached panos make no call, so this reflects what THIS run actually paid."""
        # Before the usage_metadata guard: a response can carry the resolved build
        # without carrying usage, and the build is the half we cannot back-fill.
        self.record_model_version(getattr(resp, "model_version", None))
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return
        thoughts = getattr(um, "thoughts_token_count", None) or 0
        prompt = um.prompt_token_count or 0
        candidates = um.candidates_token_count or 0
        # The whole cost figure turns on candidates_token_count EXCLUDING thinking,
        # which is what google-genai reports today and what makes
        # total = prompt + candidates + thoughts hold. If a future SDK folds
        # thinking into candidates instead, every thinking model's output silently
        # doubles (~$20 on the 3.6-flash leg alone) with no other symptom -- and a
        # unit test against a hand-built fake cannot catch that. So check it
        # against the provider's own total, once, and say so loudly if it breaks.
        total = getattr(um, "total_token_count", None)
        if total and not self._usage_warned:
            expected = prompt + candidates + thoughts
            if abs(total - expected) > max(4, 0.02 * total):
                self._usage_warned = True
                print(f"[{self.model_id}] WARNING: usage_metadata does not add up "
                      f"(total {total} vs prompt {prompt} + candidates {candidates} "
                      f"+ thoughts {thoughts} = {expected}). The cost estimate may "
                      f"double-count thinking; check the SDK's field semantics.")
        self.accumulate_usage(prompt, candidates + thoughts, thoughts)

    def _parse(self, raw, img_w, img_h):
        return gemini_boxes_to_points(raw)


# --- Claude box parsing (pure, unit-tested) ---------------------------------

# What we ask Claude for, constrained by the API rather than by prompting.
#
# Delivered as a FORCED TOOL CALL, not via output_config.format. Both give a
# schema-shaped answer, but this project's GCP org policy
# (constraints/vertexai.allowedPartnerModelFeatures) does not allow the
# `structured_outputs` feature for Anthropic partner models -- measured
# 2026-08-15: output_config.format returns 400, and so does a tool marked
# `strict: True` (which is implemented as structured outputs underneath). A
# plain tool + tool_choice passes and returns `{"boxes": [...]}` in the
# tool_use block. If the org ever allow-lists
# `publishers/anthropic/models/<model>:structured_outputs`, `strict: True`
# becomes available and would add hard schema validation on top of this.
#
# Named fields, NOT a 4-element array: Gemini's [ymin, xmin, ymax, xmax]
# ordering is a convention the model can silently transpose, and this repo has
# already been bitten by exactly that class of bug (see the Molmo
# triplet-alignment fix, 61c52d0). x1/y1/x2/y2 cannot be mis-ordered without
# being obviously wrong.
CLAUDE_BOX_TOOL = {
    "name": "report_curb_ramps",
    "description": ("Report every curb ramp visible in the image, as tight pixel "
                    "bounding boxes in the image's own coordinate space. Report an "
                    "empty list if there are none."),
    "input_schema": {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer"},
                        "y1": {"type": "integer"},
                        "x2": {"type": "integer"},
                        "y2": {"type": "integer"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["boxes"],
        "additionalProperties": False,
    },
}


def claude_boxes_to_points(items, img_w, img_h):
    """Claude returns boxes in the view's OWN PIXEL space; reduce each to its
    normalized [0,1] center point.

    Deliberately different from Gemini's 0-1000 convention: Claude 4.7+ maps
    returned coordinates 1:1 onto actual image pixels, so asking for pixels is
    asking for what the model natively produces rather than making it rescale.
    Confidence is None -- like the other chat VLMs, Claude carries no calibrated
    per-box score.

    Malformed items are SKIPPED, not fatal. Without ``strict: True`` on the tool --
    which the org policy blocks -- the schema is a strong hint rather than an
    enforced contract, and the model does occasionally return something else.
    Measured over a full annapolis split: 1 malformed item in 745 calls (0.13%),
    a list of strings where a list of objects was asked for, which raised
    "string indices must be integers" and cost the whole panorama. One bad box
    should cost one box, not six views' work."""
    points = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            x1, y1, x2, y2 = (float(it["x1"]), float(it["y1"]),
                              float(it["x2"]), float(it["y2"]))
        except (KeyError, TypeError, ValueError):
            continue
        cx = (x1 + x2) / 2.0 / float(img_w)
        cy = (y1 + y2) / 2.0 / float(img_h)
        points.append((cx, cy, None))
    return points


def _claude_box_list(data):
    """The ``boxes`` array out of a tool input (or a parsed text object), or [].

    Without ``strict: True`` -- which the org policy blocks -- the schema is a
    strong hint, not a contract, so every one of ``{"boxes": null}``,
    ``{"boxes": 3}``, a bare array and a missing key is reachable. Each returns
    an empty list rather than raising, because the alternative costs all six
    views of the panorama (see ``claude_boxes_to_points``)."""
    if isinstance(data, list):        # the model emitted the bare array
        return data
    if not isinstance(data, dict):
        return []
    boxes = data.get("boxes")
    return boxes if isinstance(boxes, list) else []


def boxes_from_claude_response(resp):
    """Pull the box list out of a tool-call response, tolerating anything else.

    The tool_use block's ``input`` is already a parsed object, so the happy path
    needs no JSON handling at all.

    The TEXT FALLBACK carries the weight now that ``tool_choice`` defaults to
    ``auto``: forcing the call made a text-only turn impossible, and allowing
    thinking made it ordinary. So a turn can end in prose -- a refusal, a
    preamble, a fenced JSON block -- and none of those may raise. This reuses
    ``_first_json_blob`` (the Qwen path's scanner) to lift the first balanced
    JSON value out of the text instead of assuming the whole block parses; a
    refusal in words simply yields no boxes, which is the honest reading for
    scoring. Truncation is NOT handled here: an empty result from a cut-off turn
    is indistinguishable from "no ramps" at this layer, so
    ``ClaudeDetector._check_stop_reason`` rejects it one level up, before it can
    be cached."""
    blocks = getattr(resp, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "tool_use":
            return _claude_box_list(getattr(block, "input", None))
    for block in blocks:
        if getattr(block, "type", None) != "text" or not getattr(block, "text", None):
            continue
        blob = _first_json_blob(block.text)
        if blob is None:
            continue
        try:
            return _claude_box_list(json.loads(blob))
        except ValueError:            # a truncated or malformed blob, not an answer
            continue
    return []


# What the published annapolis legs actually ran with (2026-08-15). These are
# INPUTS -- they change the detections -- so they belong in the cache key, but
# only once a run DEVIATES from them: writing them into every signature would
# change the hash for runs whose settings did not change, orphaning $28.82 of
# already-paid detections to record a no-op. See ClaudeDetector.signature and
# test_claude_cache_key_is_frozen.
#
# `jpeg` is the as-run value rather than the preferred one. GeminiDetector hands
# google-genai a PIL image, and its `pil_to_blob` encodes that as lossless PNG
# (its JPEG branch needs `image.format == "JPEG"` AND a filename, and a
# reprojected view is an in-memory Image.fromarray with neither) -- so the two
# paid legs are NOT seeing identical pixels today. `--claude-image-format png`
# closes that gap at the price of re-running the legs; see docs/model_comparison.md.
CLAUDE_AS_RUN_IMAGE_FORMAT = "jpeg"
CLAUDE_AS_RUN_TEMPERATURE = None      # i.e. the provider default, NOT greedy
CLAUDE_IMAGE_FORMATS = {"jpeg": ("JPEG", "image/jpeg"), "png": ("PNG", "image/png")}


class ClaudeDetector(_VLMDetector):
    """Claude via Google Cloud Vertex AI (#122).

    Runs on the SAME credential path as the Gemini legs -- Vertex + ADC, same
    project, same ``global`` location -- so it needs no new secret. Vertex's
    Claude rates match Anthropic's first-party rates on ``global`` (regional
    endpoints cost 10% more), verified 2026-08-15; see pricing.py.
    """

    name = "claude"
    # Claude's own vision cap is 2576px on the long edge, but the rig feeds it
    # 1024x1024 perspective views, so this only bounds the untiled whole-pano
    # path. Matched to Gemini's so the two legs see the same pixels there.
    max_edge = 1568

    def __init__(self, model_id="claude-sonnet-5", max_edge=None, tile=True,
                 project=None, location=None, effort="low", tool_choice="auto",
                 views=None, image_format=CLAUDE_AS_RUN_IMAGE_FORMAT,
                 temperature=CLAUDE_AS_RUN_TEMPERATURE):
        super().__init__(model_id, max_edge, tile=tile, views=views)
        if image_format not in CLAUDE_IMAGE_FORMATS:
            raise ValueError(f"unknown image_format {image_format!r} "
                             f"(choose from: {', '.join(sorted(CLAUDE_IMAGE_FORMATS))})")
        self.image_format = image_format
        # None = send no temperature at all and take the provider default, which
        # is what the published legs did. GeminiDetector pins temperature=0.0, so
        # the two paid legs currently differ in decoding as well as encoding;
        # both are caveated in docs/model_comparison.md.
        self.temperature = temperature
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        # `global` for the same reason as Gemini, plus a second one: Vertex prices
        # regional endpoints 10% above global for Claude.
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        # Effort drives how much the model thinks, thinking bills as OUTPUT, and
        # output is the entire cost variance on this leg -- so it belongs in the
        # signature (below) as much as the prompt does. `low` by default: reading
        # a 1024x1024 view and emitting a short box list is not intelligence-
        # sensitive work, and the default `high` costs several times more for it.
        self.effort = effort
        # FORCING the tool call suppresses thinking entirely, which makes `effort`
        # inert. Measured 2026-08-15 on one view: forced gives 60 output tokens
        # and 0 thinking at BOTH low and high, while auto gives 0 / 42 / 237
        # thinking at low / high / max. So `auto` is the default -- otherwise the
        # effort knob silently does nothing. `forced` guarantees the answer comes
        # back as a tool call (no text fallback needed) and is the better choice
        # at effort=low, where there is no thinking to lose.
        self.tool_choice = tool_choice
        self._client = None
        self.init_usage()
        self._usage_warned = False
        self._not_found_warned = False   # warn once per run, not once per retry
        self._refusal_warned = False
        # How each call ended, tallied. A leg where 12% of calls refused is not
        # the same measurement as one where none did, and without this nothing
        # would ever say so -- a refusal scores as "found nothing", which is
        # indistinguishable from a confident empty answer. Rides along in the
        # usage-log record (compare.report_usage).
        self.stop_reasons = Counter()

    def signature(self):
        """Effort and the tool schema change the detections, so they change the key.

        Everything else is inherited. Omitting effort would let a cheap `low` run
        and an expensive `high` run collide in one cache entry and silently mix;
        the tool definition is this provider's equivalent of the prompt, since it
        is what constrains the answer's shape.

        Image encoding and temperature are inputs too, but they appear ONLY when
        they deviate from what the published legs ran (see
        CLAUDE_AS_RUN_IMAGE_FORMAT). Recording an unchanged default would change
        the hash without changing the run and orphan the paid annapolis cache."""
        sig = super().signature()
        sig["effort"] = self.effort
        sig["tool_choice"] = self.tool_choice
        sig["box_tool"] = json.dumps(CLAUDE_BOX_TOOL, sort_keys=True)
        if self.image_format != CLAUDE_AS_RUN_IMAGE_FORMAT:
            sig["image_format"] = self.image_format
        if self.temperature != CLAUDE_AS_RUN_TEMPERATURE:
            sig["temperature"] = self.temperature
        return sig

    def location_warning(self):
        """The message to print when this run's endpoint is not priced by the table.

        pricing.py's Claude rates are the ``global`` ones; Vertex bills regional
        endpoints 10% above them. GOOGLE_CLOUD_LOCATION is shared with the Gemini
        legs, so a region set for Gemini's sake silently makes every Claude cost
        figure ~9% low. Returns None when there is nothing to say."""
        if self.location == "global":
            return None
        return (f"[{self.model_id}] WARNING: {self.location!r} is a REGIONAL Vertex "
                f"endpoint. Claude bills 10% ABOVE the `global` rates recorded in "
                f"pricing.py there, so every cost figure from this run is ~9% low. "
                f"Unset GOOGLE_CLOUD_LOCATION (or set it to 'global') to match the "
                f"table.")

    def _ensure_ready(self):
        try:
            from anthropic import AnthropicVertex
        except ImportError as e:
            raise ImportError(
                "ClaudeDetector needs the `anthropic[vertex]` package "
                "(pip install -r requirements-vlm.txt)") from e
        if self._client is not None:
            return
        if not self.project:
            raise RuntimeError(
                "Claude on Vertex needs GOOGLE_CLOUD_PROJECT (and ADC via "
                "`gcloud auth application-default login`) -- the same credentials "
                "the Gemini legs use. Each Claude model must also be enabled "
                "separately in Vertex Model Garden; an un-enabled model 404s with "
                "'was not found or your project does not have access to it'.")
        # max_retries above the SDK default of 2: a full-city run is ~750 calls and
        # a transient 429/5xx mid-leg is expensive to redo. Matches the Gemini rig's
        # 5 attempts.
        warning = self.location_warning()
        if warning:
            print(warning)
        self._client = AnthropicVertex(project_id=self.project, region=self.location,
                                       max_retries=5)

    # Vertex intermittently answers a perfectly good request with 404 "Publisher
    # model ... was not found or your project does not have access to it", most
    # visibly in the hours after a model is enabled -- the entitlement appears to
    # propagate unevenly across serving backends. Measured 2026-08-15: 12/12
    # identical calls succeeded in one burst while 3 of 5 panos 404'd minutes
    # later. The SDK does not retry 404 (it is a 4xx, normally permanent), so
    # without this a leg silently loses panos to a transient lie. A genuinely
    # un-enabled model still fails, just after this many tries.
    _NOT_FOUND_RETRIES = 4
    _NOT_FOUND_BACKOFF = 2.0   # seconds, doubled each attempt

    def _encode_image(self, image):
        """``(base64 payload, media type)`` for one view, in the configured format.

        Explicit rather than hardcoded because it decides what the model actually
        sees: `jpeg` (q90, what the published legs sent) re-quantizes every view,
        while `png` is lossless and is what the Gemini leg receives. Neither
        changes the token bill -- image tokens are a function of dimensions -- so
        the only cost of matching Gemini is re-running the legs."""
        import base64
        import io

        pil_format, media_type = CLAUDE_IMAGE_FORMATS[self.image_format]
        buf = io.BytesIO()
        if pil_format == "JPEG":
            image.save(buf, format=pil_format, quality=90)
        else:
            image.save(buf, format=pil_format)
        return base64.standard_b64encode(buf.getvalue()).decode(), media_type

    def _raw_detect(self, image):
        import time

        from anthropic import NotFoundError

        b64, media_type = self._encode_image(image)

        delay = self._NOT_FOUND_BACKOFF
        for attempt in range(self._NOT_FOUND_RETRIES):
            try:
                return self._call(b64, media_type)
            except NotFoundError:
                if attempt == self._NOT_FOUND_RETRIES - 1:
                    raise
                if not self._not_found_warned:
                    self._not_found_warned = True
                    print(f"[{self.model_id}] transient 404 from Vertex; retrying. "
                          f"If every call 404s, the model is not enabled for this "
                          f"project in Model Garden.")
                time.sleep(delay)
                delay *= 2

    # Thinking bills against max_tokens, so a high-effort turn can be cut off
    # mid-thought. That is the one failure with NO symptom: the response carries
    # no tool_use and no text, the parser reads it as "no boxes", every ground
    # truth ramp on the pano becomes a false negative, and compare.score_model
    # writes it to the detection cache -- making a silent recall loss permanent.
    # So it is raised, not returned: score_model records a visible failure and
    # caches nothing.
    _TRUNCATED_STOP_REASON = "max_tokens"

    def _call(self, b64, media_type):
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            # effort is allowed by the org policy; structured_outputs is not.
            output_config={"effort": self.effort},
            tools=[CLAUDE_BOX_TOOL],
            tool_choice=({"type": "tool", "name": CLAUDE_BOX_TOOL["name"]}
                         if self.tool_choice == "forced" else {"type": "auto"}),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media_type, "data": b64}},
                {"type": "text", "text": self.prompt},
            ]}],
            **kwargs,
        )
        self._record_usage(resp)
        self._check_stop_reason(resp)
        return boxes_from_claude_response(resp)

    def _check_stop_reason(self, resp):
        """Tally how this call ended, and refuse to let a truncation score as zero.

        ``max_tokens`` raises (see above). ``refusal`` does not -- a model that
        declined genuinely found nothing, as far as scoring is concerned -- but
        it is counted and announced once, because a leg with a refusal rate is a
        different measurement from one without and nothing else would reveal it."""
        reason = getattr(resp, "stop_reason", None)
        if reason:
            self.stop_reasons[reason] += 1
        if reason == self._TRUNCATED_STOP_REASON:
            raise RuntimeError(
                f"[{self.model_id}] response hit max_tokens before answering "
                f"(effort={self.effort}). Thinking bills against max_tokens, so a "
                f"high-effort call can be cut off mid-thought; scoring that as "
                f"'no curb ramps' would be a silent recall loss AND would be "
                f"cached. Raise max_tokens or lower --claude-effort.")
        if reason == "refusal" and not self._refusal_warned:
            self._refusal_warned = True
            print(f"[{self.model_id}] WARNING: a call ended in `refusal`, which "
                  f"scores as 'found nothing'. The per-run tally is reported with "
                  f"this leg's usage; a non-trivial refusal rate makes the leg's "
                  f"recall incomparable with the others.")

    def _record_usage(self, resp):
        """Accumulate this call's token counts.

        NOTE the difference from Gemini: Anthropic's ``output_tokens`` ALREADY
        includes thinking, whereas google-genai's ``candidates_token_count``
        excludes it. So thinking is passed through for reporting but must NOT be
        added again -- doing so would double-count the dominant cost term. The
        cross-check below is what catches that assumption inverting."""
        self.record_model_version(getattr(resp, "model", None))
        u = getattr(resp, "usage", None)
        if u is None:
            return
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        details = getattr(u, "output_tokens_details", None)
        thinking = (getattr(details, "thinking_tokens", None) or 0) if details else 0
        if thinking > out and not self._usage_warned:
            self._usage_warned = True
            print(f"[{self.model_id}] WARNING: thinking_tokens ({thinking}) exceeds "
                  f"output_tokens ({out}), so output_tokens is evidently NOT "
                  f"inclusive of thinking on this SDK. The cost estimate is now "
                  f"UNDER-counting output; check the field semantics.")
        # Separate SKUs, and NOT part of input_tokens. Zero unless a run turns on
        # cache_control (nothing does today -- the tool definition renders below
        # Sonnet 5's 1,024-token minimum cacheable prefix), but recorded so the
        # first run that does is priced whole rather than silently low.
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.accumulate_usage(inp, out, thinking, cache_read, cache_write)

    def _parse(self, raw, img_w, img_h):
        return claude_boxes_to_points(raw, img_w, img_h)


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


def _weights_fingerprint(path):
    """Content hash of a YOLO weights file, so retraining to the same path/label
    invalidates the detection cache — the weights *are* the model here, unlike a
    stable HF id. Machine-independent (hashes bytes, not the path). Returns ``None``
    if the file is absent (e.g. scoring a rsynced cache without the weights present),
    falling back to the label for identity."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class YoloDetector(_VLMDetector):
    """Ultralytics YOLO — the one **supervised** detector here, trained on the
    RampNet dataset (issue #51: is RampNet's keypoint architecture doing the work,
    or would any detector trained on the auto-generated data also beat the zero-shot
    field?). Boxes carry objectness x class confidence, so it gets AP / PR / a sweep
    like the open-vocabulary detectors, via the same box->center->score path.

    ``model_id`` here is a **weights path** (``.pt``), not an HF id. Its identity for
    the results table and the cache is the file **stem** (machine-independent), and
    ``signature`` additionally hashes the weights bytes so a re-trained checkpoint at
    the same path invalidates cached detections. Tiled by default (perspective
    views, matching how the VLMs are scored); ``--tiling none`` runs the whole-pano
    ablation with pano-geometry weights.

    Cache-gap caveat (shared with the other providers): the key does not include the
    box->point parser version. If ``_parse`` / ``yolo_results_to_boxes`` changes,
    bump something in ``signature()`` to bust ``.model_cache``."""

    name = "yolo"
    prompt = None            # not a prompted model; the base "prompt" cache key stays None
    score_threshold = 0.05   # cache floor, like the zero-shot detectors

    def __init__(self, weights, label=None, conf=None, iou=0.5, imgsz=1024,
                 tile=True, views=None):
        # Identity is the file stem, not the absolute path, so the cache is
        # machine-independent (score a rsynced .model_cache without re-running).
        stem = label or os.path.splitext(os.path.basename(str(weights)))[0]
        super().__init__(stem, tile=tile, views=views)
        self.weights = weights
        self.conf = self.score_threshold if conf is None else float(conf)
        self.score_threshold = self.conf   # attribute read to feed the --sweep floor
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self._model = None
        self._device = "cpu"

    def _ensure_ready(self):
        if self._model is not None:
            return
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YoloDetector needs `ultralytics` "
                "(pip install ultralytics; see requirements-vlm.txt)") from e
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(self.weights)

    def _raw_detect(self, image):
        # conf is a low cache floor; higher operating points are free re-scores
        # (--op-threshold / --sweep), exactly like the open-vocab detectors.
        results = self._model.predict(image, conf=self.conf, iou=self.iou,
                                      imgsz=self.imgsz, device=self._device, verbose=False)
        return yolo_results_to_boxes(results[0])

    def _parse(self, raw, img_w, img_h):
        return pixel_boxes_to_points(raw, img_w, img_h)

    def signature(self):
        sig = super().signature()
        sig.update({"weights_hash": _weights_fingerprint(self.weights),
                    "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz})
        return sig


# Every provider ``build_detector`` knows, in the order they are documented. The
# ONE source of truth: this list previously lived re-typed in four places, and
# `claude` shipped in build_detector while three of them still said it did not
# exist. compare.py builds its --models help from this; the tests check the
# prose that cannot be generated (test_every_provider_is_listed_everywhere).
PROVIDERS = ("rampnet", "gemini", "claude", "qwen", "owlv2", "gdino", "molmo",
             "vistas", "yolo")

# --------------------------------------------------------------------------- #
# Mapillary Vistas supervised-transfer baseline (#126)
# --------------------------------------------------------------------------- #
VISTAS_CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Class ids in that checkpoint's 65-class head (the Vistas v1.2 label set), verified
# 2026-08-18 against its published config.json id2label. They are pinned here rather
# than read off the loaded model because `signature()` has to work without weights —
# export_model_cache reconstructs signatures on a laptop with no GPU. _ensure_ready
# checks them against the loaded id2label anyway, so a checkpoint change fails loudly
# instead of silently segmenting the wrong class.
VISTAS_CLASS_IDS = {"curb-cut": 9, "curb": 2}

#: The two arms. `curb-cut` is the class this benchmark is about; the union is a
#: separate arm because Vistas draws the ramp/curb boundary somewhere we do not, and
#: whether our recall is hiding on the other side of it is a measurable question.
VISTAS_CLASS_SETS = {
    "curb-cut": ("curb-cut",),
    "curb-cut+curb": ("curb-cut", "curb"),
}


class VistasDetector(_VLMDetector):
    """A Mapillary-Vistas-supervised segmenter, scored through the point protocol.

    **The one class of baseline the roster was missing.** Every other challenger is
    zero-shot — a prompted general VLM or an open-vocabulary detector — so the roster
    answers "can a general model be prompted to do this?", and #51 answers
    "architecture versus data *within* our dataset". Neither answers "do somebody
    else's supervised curb-cut labels transfer to deployment panoramas?", which is
    what this arm is for. It is also the only member that natively produces masks.

    **This is a baseline, never a supervision source.** The RampNet paper
    (arXiv 2508.09415) already reviewed this exact class and rejected it as a data
    source: *"their categorization was overly broad and included driveways labeled as
    curb cuts."* That prior assessment stands and is cited rather than rediscovered.
    It also makes a prediction — driveway aprons should show up as a characteristic
    false-positive mode — which `fp_taxonomy.py` can name directly.

    Two things to expect, recorded up front so they are findings rather than
    surprises. Vistas is perspective imagery, and so is the input here (the same
    six-view rig every tiled leg uses), but the checkpoint has still never seen a
    reprojected 360 panorama. And the rig's ``pitch_deg=-30`` puts the capture
    vehicle's hood in the bottom of every view; Vistas has classes for it, and a
    segmenter is likely to react to that more strongly than a box detector did (#47).
    """

    name = "vistas"

    def __init__(self, class_set="curb-cut", label=None, checkpoint=None,
                 min_area_px=16, max_edge=None, tile=True, views=None,
                 dtype="float16", input_size=None, revision=None):
        # ``model_id`` is the LABEL, not the checkpoint -- the same trick YoloDetector
        # uses for a weights path. Two arms share one checkpoint and differ only by
        # which classes they read, so the checkpoint cannot identify a row; and the
        # published-artifact contract is that signature["model_id"] equals the file's
        # model name (tests/test_export_model_cache.py). The checkpoint keeps its own
        # signature field below.
        super().__init__(label or f"mask2former-vistas-{class_set}", max_edge,
                         tile=tile, views=views)
        self.checkpoint = checkpoint or VISTAS_CHECKPOINT
        if class_set not in VISTAS_CLASS_SETS:
            raise ValueError(
                f"unknown vistas class set {class_set!r} "
                f"(choose from: {', '.join(VISTAS_CLASS_SETS)})")
        self.class_set = class_set
        self.class_names = VISTAS_CLASS_SETS[class_set]
        self.class_ids = tuple(VISTAS_CLASS_IDS[n] for n in self.class_names)
        # A cache floor, not the operating point -- see masks_to_points.
        self.min_area_px = int(min_area_px)
        # fp16 and fp32 do not give identical masks, so a desktop run and a cluster
        # run must not share a cache key. Hence this is in the signature.
        self.dtype = dtype
        # The class names ARE this model's prompt: it is not told what to look for at
        # inference, it was supervised on it. Putting them in the base signature's
        # "prompt" slot keys the cache on the arm, exactly as the text query does for
        # the open-vocabulary detectors.
        self.prompt = "vistas:" + "+".join(self.class_names)
        # What the model ACTUALLY sees. The checkpoint's own preprocessor_config
        # says {"height": 384, "width": 384} with do_resize, so a 1024x1024 view is
        # downsized to about 1/7 the pixel area and the masks come back at 96x96 to
        # be upsampled 10.67x. That is a real handicap against every other tiled leg
        # and it was invisible: not pinned, not overridable, not in the signature.
        # None keeps the processor's default -- i.e. exactly what was published.
        self.input_size = tuple(input_size) if input_size else None
        # Nor was the checkpoint pinned, so a re-download could silently change every
        # mask. Same treatment: recorded when set, absent when it is whatever HF
        # served, which is the honest description of the published run.
        self.revision = revision
        self._model = None
        self._processor = None
        self._device = "cpu"

    def signature(self):
        """The cache key for this arm.

        `class_set`, `class_ids`, `class_names` and the base `prompt` are four
        spellings of one datum, all derivable from `class_set` plus the pinned
        VISTAS_CLASS_IDS. That is redundant and it is deliberately left alone:
        cache_key hashes the WHOLE signature, so dropping a key to tidy up would
        orphan both arms' published detections and force a re-run for no gain.
        Anything ADDED here has the same cost, which is why the two knobs below
        appear only when they deviate from what the published run used.
        """
        sig = super().signature()
        sig.update({"checkpoint": self.checkpoint,
                    "class_set": self.class_set,
                    "class_ids": list(self.class_ids),
                    "class_names": list(self.class_names),
                    "min_area_px": self.min_area_px,
                    "dtype": self.dtype})
        # Deviation-only, so the published richmond detections keep their key: the
        # arm as published took the processor's own 384x384 and an unpinned revision.
        if self.input_size is not None:
            sig["input_size"] = list(self.input_size)
        if self.revision is not None:
            sig["revision"] = self.revision
        return sig

    def _ensure_ready(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as e:
            raise ImportError(
                "VistasDetector needs `torch` and `transformers` "
                "(pip install -r requirements-vlm.txt)") from e
        kw = {"revision": self.revision} if self.revision else {}
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint, **kw)
        if self.input_size is not None:
            h, w = self.input_size
            self._processor.size = {"height": int(h), "width": int(w)}
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.checkpoint, **kw)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype == "float16" and self._device != "cuda":
            # dtype is in the signature; the device is not. Silently running fp32
            # here would put CPU-derived and GPU-derived masks under ONE cache key,
            # which is the collision the signature field exists to prevent. Refuse
            # instead: --vistas-dtype float32 is a different key, correctly.
            raise RuntimeError(
                "vistas dtype float16 needs CUDA, and this machine has none. Running "
                "fp32 under the fp16 cache key would mix two different sets of masks "
                "in one cache entry. Pass --vistas-dtype float32 (a distinct key).")
        if self.dtype == "float16":
            model = model.half()
        model = model.to(self._device).eval()
        # Validate BEFORE publishing self._model. The other order left a half-ready
        # detector behind on failure: _ensure_ready returns early when self._model is
        # set, so one caught RuntimeError turned the check into a no-op for the rest
        # of the run -- and this check exists precisely because segmenting class 9 of
        # a different label set would not raise, it would quietly score the wrong
        # object.
        self._check_class_ids(model)
        self._model = model

    def _check_class_ids(self, model):
        """Fail loudly if the checkpoint's label set is not the one we pinned.

        The ids are constants so the signature works without weights. That is only
        safe if a mismatch is caught: segmenting class 9 of a *different* label set
        would not raise, it would quietly score the wrong object."""
        id2label = getattr(model.config, "id2label", None) or {}
        for name, cid in zip(self.class_names, self.class_ids):
            actual = id2label.get(cid, id2label.get(str(cid)))
            expected = name.replace("-", " ").title()          # curb-cut -> Curb Cut
            if actual is None or actual.strip().lower() != expected.lower():
                raise RuntimeError(
                    f"{self.checkpoint} class {cid} is {actual!r}, expected "
                    f"{expected!r}. VISTAS_CLASS_IDS is pinned to the 65-class "
                    f"Vistas v1.2 head; this checkpoint has a different label set, "
                    f"so the ids must be re-derived before it can be scored.")

    def _raw_detect(self, image):
        """``(seg, prob)`` at the view's resolution: winning class id per pixel, and
        that class's normalized score.

        The semantic map is assembled here rather than taken from
        ``post_process_semantic_segmentation`` because that helper returns the argmax
        only, and a per-pixel confidence is what gives this arm a PR curve. The
        einsum below is the same combination the helper performs internally:
        per-query class probabilities (with the no-object column dropped) against
        per-query mask probabilities.
        """
        import torch
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if self.dtype == "float16":
            inputs = {k: (v.half() if v.is_floating_point() else v)
                      for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # NOTE, deliberately not changed: this upsamples all ~100 query masks to the
        # full view before reducing over classes, which for a 1024x1024 view is about
        # 1.1 GB of transient tensors (419 MB interpolated + 419 MB sigmoid + 272 MB
        # einsum output) to carry information that only exists at 96x96. Reducing
        # first -- einsum and max at 96x96, then interpolate the two resulting maps --
        # is ~1/50th the memory and agrees everywhere except boundary pixels. It is
        # left alone because "except boundary pixels" means different points, which
        # means re-scoring the arm and republishing both files; the win is memory, not
        # a number. Worth doing the next time this arm runs anyway.
        masks = torch.nn.functional.interpolate(
            outputs.masks_queries_logits.float(),
            size=(image.height, image.width), mode="bilinear", align_corners=False)
        mask_probs = masks.sigmoid()                              # (1, Q, H, W)
        class_probs = outputs.class_queries_logits.float().softmax(-1)[..., :-1]
        semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)[0]
        # Normalize across classes so the score is comparable between pixels; the
        # einsum output is a weighted mask sum, not a distribution.
        total = semantic.sum(0, keepdim=True).clamp_min(1e-6)
        normalized = semantic / total
        prob, seg = normalized.max(0)
        return seg.cpu().numpy(), prob.cpu().numpy()

    def _parse(self, raw, img_w, img_h):
        seg, prob = raw
        return masks_to_points(seg, prob, self.class_ids,
                               min_area_px=self.min_area_px)


def parse_model_spec(token):
    """Parse a ``--models`` token into ``(provider, model_id_or_None)``.

    A token is either a bare provider (``rampnet`` / ``gemini`` / ``claude`` /
    ``qwen`` / ``owlv2`` / ``gdino`` / ``molmo`` / ``vistas`` / ``yolo``, which uses that
    provider's default model) or ``provider:model_id`` to pin a variant — e.g.
    ``gemini:gemini-2.5-flash`` vs ``gemini:gemini-3.6-flash``, or
    ``yolo:runs/detect/train/weights/best.pt`` for a trained checkpoint — so several
    variants of the same provider can be compared in one run."""
    provider, _, model_id = token.partition(":")
    return provider.strip(), (model_id.strip() or None)


def _D(key):
    """One provider default, from the registry.

    build_detector is reached from namespaces that are not compare.py's parser --
    null_recall and dump_detections build their own, and two analysis scripts hand
    it a private Args -- so every `getattr(args, k, None) or <literal>` here was a
    fifth copy of a value that feeds the cache signature. A copy that drifts does
    not crash; it changes the key and silently misses every already-paid detection.
    Imported lazily so detectors.py keeps importing without the rampnet package on
    sys.path.
    """
    from rampnet.roster import PROVIDER_DEFAULTS
    return PROVIDER_DEFAULTS[key]


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
    if provider == "claude":
        mid = model_id or args.claude_model
        return mid, ClaudeDetector(
            model_id=mid, tile=tile,
            effort=getattr(args, "claude_effort", None) or _D("claude_effort"),
            tool_choice=getattr(args, "claude_tool_choice", None) or _D("claude_tool_choice"),
            image_format=(getattr(args, "claude_image_format", None)
                          or CLAUDE_AS_RUN_IMAGE_FORMAT),
            temperature=getattr(args, "claude_temperature", CLAUDE_AS_RUN_TEMPERATURE))
    if provider == "qwen":
        mid = model_id or args.qwen_model
        coord_space = getattr(args, "qwen_coord_space", "auto")
        return mid, QwenDetector(model_id=mid, tile=tile,
                                 coord_space=None if coord_space == "auto" else coord_space)
    if provider == "owlv2":
        mid = model_id or getattr(args, "owlv2_model", None) or _D("owlv2_model")
        return mid, OwlV2Detector(model_id=mid, tile=tile,
                                  query=getattr(args, "owlv2_query", None),
                                  score_threshold=getattr(args, "score_threshold", None))
    if provider == "gdino":
        mid = model_id or getattr(args, "gdino_model", None) or _D("gdino_model")
        return mid, GroundingDinoDetector(model_id=mid, tile=tile,
                                          query=getattr(args, "gdino_query", None),
                                          score_threshold=getattr(args, "score_threshold", None),
                                          text_threshold=getattr(args, "gdino_text_threshold", None))
    if provider == "molmo":
        mid = model_id or getattr(args, "molmo_model", None) or _D("molmo_model")
        scale = getattr(args, "molmo_coord_scale", "auto")
        return mid, MolmoDetector(model_id=mid, tile=tile,
                                  coord_scale=None if scale in (None, "auto") else float(scale))
    if provider == "yolo":
        # model_id (from yolo:<path>) or --yolo-model is a trained weights path, not
        # an HF id; the table label + cache identity is its machine-independent stem.
        weights = model_id or getattr(args, "yolo_model", None)
        if not weights:
            raise ValueError("yolo needs a trained weights path: "
                             "--models yolo:<path.pt> or --yolo-model <path.pt>")
        iou = getattr(args, "yolo_iou", None)
        imgsz = getattr(args, "yolo_imgsz", None)
        iou = _D("yolo_iou") if iou is None else float(iou)
        imgsz = _D("yolo_imgsz") if imgsz is None else int(imgsz)
        label = os.path.splitext(os.path.basename(str(weights)))[0]
        return label, YoloDetector(weights=weights, label=label, tile=tile,
                                   conf=getattr(args, "yolo_conf", None),
                                   iou=iou, imgsz=imgsz)
    if provider == "vistas":
        # The model_id slot carries the CLASS SET, not a model id -- the checkpoint
        # comes from --vistas-model, because the arm varies by which Vistas classes
        # are read out, not by which checkpoint reads them. So the label cannot be
        # derived the usual way and is declared in rampnet/roster.py instead.
        from rampnet import roster   # lazy, like every other rampnet import here
        class_set = (model_id or getattr(args, "vistas_class_set", None)
                     or _D("vistas_class_set"))
        min_area = getattr(args, "vistas_min_area_px", None)
        label = roster.label_for(f"vistas:{class_set}")
        det = VistasDetector(
            class_set=class_set, label=label,
            checkpoint=getattr(args, "vistas_model", None) or _D("vistas_model"),
            min_area_px=_D("vistas_min_area_px") if min_area is None else int(min_area),
            tile=tile,
            dtype=getattr(args, "vistas_dtype", None) or _D("vistas_dtype"),
            input_size=getattr(args, "vistas_input_size", None),
            revision=getattr(args, "vistas_revision", None))
        return label, det
    raise ValueError(f"unknown provider '{provider}' "
                     f"(choose from: {', '.join(PROVIDERS)})")
