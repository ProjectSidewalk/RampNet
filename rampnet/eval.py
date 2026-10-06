"""The RampNet benchmark scoring protocol, as code (#150).

Score a prediction file against a committed benchmark bundle **exactly as the committed
numbers were scored**::

    python -m rampnet.eval score --bundle benchmark/richmond \\
        --predictions benchmark/model_detections/y11l_pano__richmond.json \\
        --op-threshold 0.25 --floor 0.05
    python -m rampnet.eval score --bundle benchmark/manual_gold --predictions rampnet \\
        --op-threshold 0.55 --floor 0.05
    python -m rampnet.eval pins --verify       # split files unchanged since pinned?
    python -m rampnet.eval loco --model y11l_pano --op-threshold 0.25
    python -m rampnet.eval protocol            # the rule, the constants, the fingerprint

This module adds no scoring logic of its own. Matching and aggregation are
:func:`rampnet.detection_eval.score_pano` and :func:`rampnet.detection_eval.aggregate`,
the same functions behind ``analysis_out/scoreboard.json``, ``docs/model_comparison.md``
and ``scripts/model_comparison/yolo_baseline/benchmark_eval/``.
``scripts/analysis/eval_protocol_150.py --check`` proves it by re-deriving every one of
those committed numbers through :func:`score_split`. See ``docs/eval_protocol_150.md``.

**The protocol** (the erratum's rule, ``README.md``):

* Ground truth per pano: verdict-derived for the city splits (reviewer-confirmed
  detections plus missed marks; ``unsure`` ones become *ignore* points), independent YOLO
  box centres for ``manual_gold``.
* Matching: greedy one-to-one, predictions in descending confidence order, each claims
  the nearest unclaimed GT point **strictly** within radius 0.022 of the panorama width,
  in a 1024 x 512 scaled space, with x wrapping across the 0/1 seam. An unmatched
  prediction within radius of an ignore point is neither TP nor FP.
* Precision counts every pano; recall counts only panos whose missed-ramp check is
  confirmed (``fn_confirmed``). Wilson 95% intervals on both.
* AP: VOC all-point interpolated, over the recall-confirmed panos, computed from the
  **full** confidence range of the predictions as given. An operating threshold
  truncates P/R/F1 only, never AP. AP is therefore truncated at whatever floor the
  predictions were exported at, and the output says so (``ap_note``).

**Prediction format** -- the ``benchmark/model_detections/`` file shape, and no other::

    {"model": "my-detector",
     "city": "richmond",
     "detections": {
       "1273933840289887": [[0.2851, 0.5859, 0.95], [0.8154, 0.5703, 0.93]],
       "934739365184374": []}}

``x`` and ``y`` are normalised to ``[0, 1]`` on the equirectangular panorama (x wraps).
The third element is a confidence, or ``null`` for a detector that emits none (such a
model gets no AP and no sweep). A pano absent from ``detections`` is scored as zero
predictions, and the output reports how many there were. ``--predictions rampnet``
scores the bundle's own ``records.jsonl`` detections.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from numbers import Real

from rampnet.bundles import ground_truths, rampnet_predictions, split_pins
from rampnet.detection_eval import (
    PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y, aggregate, prediction_confidence,
    radius_sq_for, score_pano,
)
from rampnet.roster import slug

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
#: The checkout this module lives in. The module-level defaults below point here; the
#: CLI prefers the current directory when it is a checkout (see :func:`cli_root`), so an
#: editable install shared between worktrees verifies the worktree you are in.
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
BENCHMARK_DIR = os.path.join(REPO_ROOT, "benchmark")
PUBLISHED_DIR = os.path.join(BENCHMARK_DIR, "model_detections")
PINS_PATH = os.path.join(BENCHMARK_DIR, "split_pins.json")

#: The files every committed benchmark number is a function of, fingerprinted the way
#: ``rescore_benchmark_eval.scorer_fingerprint`` does it (and must agree with it; a test
#: checks). This module is deliberately NOT in the list: it only packages the scorer, and
#: adding it would invalidate every stamped ``benchmark_eval/`` file.
SCORER_SOURCES = ("geometry.py", "metrics.py", "detection_eval.py", "validation.py")

#: The scored splits whose files are pinned in ``benchmark/split_pins.json``: the twelve
#: bundles on the scoreboard. Restates ``scripts/analysis/low_floor_sweep.ALL_SPLITS``
#: (a package cannot import a script); ``tests/test_eval_protocol_150.py`` asserts the
#: two agree. Bundles still being built (bayonne, vancouver, the #48 neighbourhood
#: bundles) are left out on purpose until their files settle.
PINNED_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
                 "gainesville", "laurens_mapillary", "laurens_gsv", "budapest_district5",
                 "sao_paulo", "manual_gold")

#: The in-distribution (pooled) splits, ``low_floor_sweep.US_SPLITS`` restated (tested).
#: The ``loco`` report holds each one out in turn against the rest of this pool.
POOLED_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
                 "gainesville", "laurens_mapillary")

MATCHING_RULE = "greedy 1:1, confidence-descending, strict radius"
RECALL_OVER = "fn_confirmed panos"

SWEEP_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------------- #
# Re-scoring helpers (moved from scripts/model_comparison/compare.py, which
# re-exports them). ``wrap_x`` defaults to score_pano's own default, so a caller
# that does not pass it gets byte-identical behaviour to the pre-move code.
# --------------------------------------------------------------------------- #
def rescore(scored, radius_sq, min_confidence=0.0, wrap_x=True):
    """Re-aggregate a finished run with predictions below ``min_confidence`` dropped.

    Detections are cached with their scores, so every operating point of a
    confidence-carrying detector is a free re-score — no second model run. A
    prediction with no confidence (chat VLMs) is never dropped: there is nothing to
    threshold on."""
    return aggregate([
        score_pano([p for p in preds
                    if prediction_confidence(p) is None
                    or prediction_confidence(p) >= min_confidence],
                   gt, radius_sq=radius_sq, wrap_x=wrap_x)
        for preds, gt in scored])


def operating_report(report, scored, radius_sq, op_threshold, wrap_x=True):
    """The table row for one model: P/R/F1/counts at the operating threshold, but
    AP and the PR curve kept from the full-range ``report``. Those two are
    integrals over the whole confidence range — rescore()'s filtered aggregate
    would silently truncate them at the operating point, exactly the caveat the
    manual_gold bundle's 0.05 export floor exists to avoid."""
    if op_threshold <= 0:
        return report
    return rescore(scored, radius_sq, op_threshold, wrap_x=wrap_x)._replace(
        ap=report.ap, pr_curve=report.pr_curve)


def has_confidences(scored):
    """True when every prediction in the run carries a score (so AP / a sweep mean
    something). An empty run counts as no confidences."""
    preds = [p for ps, _ in scored for p in ps]
    return bool(preds) and all(prediction_confidence(p) is not None for p in preds)


def sweep_rows(scored, radius_sq, thresholds=SWEEP_THRESHOLDS, floor=None, wrap_x=True):
    """(threshold, ScoreReport) for each threshold that still keeps a prediction.

    ``floor`` is the detector's cache floor (--score-threshold): the cache holds
    no detections below it, so a sweep row under the floor would silently repeat
    the floor row while reading as a real measurement. Those rows are dropped
    (with the default floor, 0.05, nothing is — it equals the lowest threshold)."""
    top = max((prediction_confidence(p) for ps, _ in scored for p in ps
               if prediction_confidence(p) is not None),
              default=0.0)
    return [(t, rescore(scored, radius_sq, t, wrap_x=wrap_x)) for t in thresholds
            if t <= top and (floor is None or t >= floor)]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _lf_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def scorer_fingerprint(package_dir=PACKAGE_DIR):
    """sha256 (first 12 hex) over the LF-normalised bytes of the four scorer files.

    The same value ``rescore_benchmark_eval.scorer_fingerprint`` stamps into every
    ``benchmark_eval/`` file, so a result from :func:`score_split` names the scorer the
    committed numbers came from."""
    h = hashlib.sha256()
    for name in SCORER_SOURCES:
        with open(os.path.join(package_dir, name), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def eval_sha256():
    """sha256 (first 12 hex, LF-normalised) of this module: the packaging layer."""
    return _lf_sha256(os.path.abspath(__file__))[:12]


def protocol_constants(radius=PANO_RADIUS_NORMALIZED, wrap_x=True):
    """The protocol block every :func:`score_split` result carries."""
    return {"radius": radius, "scale_x": PANO_SCALE_X, "scale_y": PANO_SCALE_Y,
            "wrap_x": wrap_x, "matching": MATCHING_RULE, "recall_over": RECALL_OVER,
            "ap": "VOC all-point interpolated over fn_confirmed panos, full confidence "
                  "range of the predictions as given (not truncated by op_threshold)"}


# --------------------------------------------------------------------------- #
# Prediction files
# --------------------------------------------------------------------------- #
class PredictionFormatError(ValueError):
    """A prediction file that does not match the documented format."""


def _is_number(v):
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)


def validate_predictions(obj, source="predictions"):
    """Check a parsed prediction file; return it unchanged or raise PredictionFormatError.

    The message names the pano id and the index of the offending point, so a malformed
    file is fixable from the error alone. Confidences must be non-negative (not capped at
    1: RampNet heatmap peaks can exceed it); ``model`` and ``city``, when present, must be
    strings."""
    if not isinstance(obj, dict):
        raise PredictionFormatError(f"{source}: top level must be an object, got "
                                    f"{type(obj).__name__}")
    for key in ("model", "city"):
        if key in obj and obj[key] is not None and not isinstance(obj[key], str):
            raise PredictionFormatError(f"{source}: '{key}' must be a string, got "
                                        f"{type(obj[key]).__name__}")
    dets = obj.get("detections")
    if not isinstance(dets, dict):
        raise PredictionFormatError(f"{source}: 'detections' must be an object mapping "
                                    "pano id -> list of [x, y, conf|null]")
    for pid, points in dets.items():
        if not isinstance(points, list):
            raise PredictionFormatError(f"{source}: pano {pid}: expected a list of points, "
                                        f"got {type(points).__name__}")
        for i, p in enumerate(points):
            where = f"{source}: pano {pid}, point {i}"
            if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
                raise PredictionFormatError(f"{where}: expected [x, y] or [x, y, conf], "
                                            f"got {p!r}")
            x, y = p[0], p[1]
            if not (_is_number(x) and _is_number(y)):
                raise PredictionFormatError(f"{where}: x and y must be finite numbers, "
                                            f"got {p!r}")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise PredictionFormatError(f"{where}: ({x}, {y}) outside [0, 1]")
            if len(p) == 3 and p[2] is not None and not _is_number(p[2]):
                raise PredictionFormatError(f"{where}: confidence must be a number or "
                                            f"null, got {p[2]!r}")
            # Not capped at 1: RampNet's heatmap peaks can exceed it (the committed
            # op_cache holds a 1.011), and ranking is all AP needs.
            if len(p) == 3 and p[2] is not None and p[2] < 0.0:
                raise PredictionFormatError(f"{where}: confidence {p[2]} is negative")
    return obj


def op_cache_predictions(bundle_dir):
    """RampNet's low-floor extraction (``analysis_out/op_cache/<split>.json``, #54) for a
    bundle, in the prediction-file format: the same no-TTA run as a city bundle's
    ``records.jsonl``, down to 0.05 instead of 0.55. This is what the published RampNet
    city-split AP is read from (``scoreboard.uses_low_floor_cache``). Found relative to
    the bundle (``<bundle>/../../analysis_out/op_cache``), so it works from any cwd."""
    split = os.path.basename(os.path.abspath(bundle_dir))
    path = os.path.normpath(os.path.join(os.path.abspath(bundle_dir), os.pardir, os.pardir,
                                         "analysis_out",
                                         "op_cache", f"{split}.json"))
    if not os.path.exists(path):
        raise PredictionFormatError(f"{path}: no low-floor cache for split {split!r}")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("meta", {}).get("gt") == "unreviewed":
        raise PredictionFormatError(f"{path}: an unreviewed cache; it cannot be scored")
    return {"model": "rampnet (op_cache)", "city": split,
            "detections": {p["pano"]: [list(t) for t in p["preds"]]
                           for p in payload["panos"]}}


def load_predictions(path):
    """Read and validate one prediction file."""
    with open(path, encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
        except json.JSONDecodeError as e:
            raise PredictionFormatError(f"{path}: not valid JSON ({e})") from None
    return validate_predictions(obj, source=path)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _report_dict(r):
    return {"precision": r.precision, "recall": r.recall, "f1": r.f1,
            "precision_ci": list(r.precision_ci), "recall_ci": list(r.recall_ci),
            "tp": r.tp, "fp": r.fp, "fn": r.fn, "ignored": r.ignored}


def score_split(bundle_dir, predictions, *, radius=PANO_RADIUS_NORMALIZED, wrap_x=True,
                op_threshold=0.0, floor=None, sweep=False, pr_curve=False, model=None):
    """Score one prediction set against one benchmark bundle.

    ``predictions`` is a parsed prediction file (the documented format) or the string
    ``"rampnet"`` for the bundle's own ``records.jsonl`` detections. ``op_threshold``
    drops predictions below it for P/R/F1 and the counts; AP and the PR curve are always
    taken from the full range. ``floor`` declares the confidence the predictions were
    exported down to: anything below it is dropped before scoring (the result says how
    many) and sweep rows below it are omitted. Returns a JSON-ready dict.

    Example::

        >>> res = score_split("benchmark/richmond", "rampnet", op_threshold=0.55)
        >>> res["tp"], res["fp"], res["fn"]
        (238, 9, 72)
    """
    records, gts, gt_kind = ground_truths(bundle_dir)
    split = os.path.basename(os.path.abspath(bundle_dir))
    warnings = []
    if isinstance(predictions, str):
        if predictions == "rampnet":
            dets = rampnet_predictions(records, gts)
            model = model or "rampnet"
        elif predictions == "rampnet-op-cache":
            predictions = op_cache_predictions(bundle_dir)
            # The cache is a stand-in only where the bundle's own detections are
            # truncated (scoreboard.uses_low_floor_cache, turned around). manual_gold's
            # bundle is already at 0.05 with flip-TTA; its op_cache is a no-TTA run that
            # was never published, so scoring it there would quietly swap the model config.
            own = [c for pid in gts for c in (prediction_confidence(d) for d in
                                              records[pid].get("detections", []))
                   if c is not None]
            cache = [prediction_confidence(p) for pts in predictions["detections"].values()
                     for p in pts if prediction_confidence(p) is not None]
            if own and cache and min(own) - min(cache) <= 0.1:
                raise PredictionFormatError(
                    f"{split}: the bundle's own RampNet detections already reach "
                    f"{min(own):.4f}, within 0.1 of the op_cache floor {min(cache):.4f}, so "
                    "the op_cache is not the published source here (it is a separate no-TTA "
                    "extraction); use --predictions rampnet")
        else:
            raise ValueError("predictions must be a parsed prediction file, 'rampnet' or "
                             "'rampnet-op-cache'")
    if not isinstance(predictions, str):
        validate_predictions(predictions)
        dets = predictions["detections"]
        model = model or predictions.get("model")
        city = predictions.get("city")
        if city is not None and city != split:
            warnings.append(f"the predictions say city {city!r} but the bundle is {split!r}")
        if dets and not any(pid in gts for pid in dets):
            raise PredictionFormatError(
                f"none of the {len(dets)} prediction panos is in {split!r}; wrong split?"
                + (f" (the file says city {city!r})" if city else ""))
    if floor is not None and 0 < op_threshold < floor:
        warnings.append(f"op_threshold {op_threshold} is below the declared floor {floor}: "
                        "nothing between them exists, so the operating point is the floor")

    dropped = 0
    if floor is not None:
        kept = {}
        for pid, pts in dets.items():
            keep = [p for p in pts if prediction_confidence(p) is None
                    or prediction_confidence(p) >= floor]
            dropped += len(pts) - len(keep)
            kept[pid] = keep
        dets = kept

    radius_sq = radius_sq_for(radius)
    scored = [(dets.get(pid, []), gt) for pid, gt in gts.items()]
    full = aggregate([score_pano(p, g, radius_sq=radius_sq, wrap_x=wrap_x)
                      for p, g in scored])
    rep = operating_report(full, scored, radius_sq, op_threshold, wrap_x=wrap_x)

    confs = [prediction_confidence(p) for ps, _ in scored for p in ps]
    lowest = min((c for c in confs if c is not None), default=None)
    if rep.ap is None:
        ap_note = ("no AP: at least one scored prediction carries no confidence, or "
                   "there are no predictions")
    else:
        # Always the lowest confidence actually present: that, not a declared floor, is
        # where the curve AP integrates is cut off.
        ap_note = (f"AP over predictions as given; truncated at the lowest confidence "
                   f"present, {lowest:.4f}")
        if floor is not None:
            ap_note += f" (declared floor {floor})"
            # The same test scoreboard.uses_low_floor_cache applies before it swaps in
            # the low-floor cache for RampNet's AP.
            if lowest - floor > 0.1:
                warnings.append(
                    f"the predictions start at {lowest:.4f}, more than 0.1 above the "
                    f"declared floor {floor}: AP is truncated there, not at the floor"
                    + (" (for RampNet's published city-split AP use --predictions "
                       "rampnet-op-cache)" if model == "rampnet" else ""))
        else:
            ap_note += " (declare --floor to state the export floor)"

    out = {
        "split": split, "model": model, "protocol": protocol_constants(radius, wrap_x),
        "gt_kind": gt_kind, "op_threshold": op_threshold, "floor": floor,
        "n_below_floor_dropped": dropped,
        **_report_dict(rep),
        "ap": rep.ap, "ap_note": ap_note,
        "n_panos": rep.n_panos, "n_recall_panos": rep.n_recall_panos,
        "n_gt_recall": rep.n_gt_recall,
        "fp_per_pano": rep.fp / rep.n_panos if rep.n_panos else None,
        "n_panos_without_predictions": sum(1 for pid in gts if pid not in dets),
        "n_prediction_panos_not_in_bundle": sum(1 for pid in dets if pid not in gts),
        "warnings": warnings,
    }
    if pr_curve and rep.pr_curve is not None:
        out["pr_curve"] = {"recalls": list(rep.pr_curve[0]),
                           "precisions": list(rep.pr_curve[1])}
    if sweep:
        out["sweep"] = ([{"threshold": t, **_report_dict(r)}
                         for t, r in sweep_rows(scored, radius_sq, floor=floor,
                                                wrap_x=wrap_x)]
                        if has_confidences(scored) else [])
    out["pins"] = split_pins(bundle_dir)
    out["scorer_fingerprint"] = scorer_fingerprint()
    out["eval_sha256"] = eval_sha256()
    return out


# --------------------------------------------------------------------------- #
# Split pins
# --------------------------------------------------------------------------- #
def all_pins(benchmark_dir=BENCHMARK_DIR, splits=PINNED_SPLITS):
    """``{split: split_pins(...)}`` for every pinned split, in registry order."""
    return {s: split_pins(os.path.join(benchmark_dir, s)) for s in splits}


def pins_payload(benchmark_dir=BENCHMARK_DIR):
    """The text of ``benchmark/split_pins.json`` (LF, sorted keys, trailing newline)."""
    body = {"what": "Content hashes of the files each scored benchmark split is "
                    "scored from (records.jsonl; verdicts.json or the manual labels; the "
                    "imagery manifest digest). LF-normalised sha256. Regenerate with "
                    "`python -m rampnet.eval pins --write`; `--verify` fails on drift. "
                    "See docs/eval_protocol_150.md.",
            "splits": all_pins(benchmark_dir)}
    return json.dumps(body, indent=1, sort_keys=True) + "\n"


def verify_pins(path=PINS_PATH, benchmark_dir=BENCHMARK_DIR):
    """List of human-readable drift lines; empty when every pinned file is unchanged."""
    if not os.path.exists(path):
        return [f"{path}: missing (run `python -m rampnet.eval pins --write`)"]
    with open(path, encoding="utf-8") as fh:
        committed = json.load(fh)["splits"]
    live = all_pins(benchmark_dir)
    drift = []
    for split in sorted(set(committed) | set(live)):
        a, b = committed.get(split), live.get(split)
        if a is None or b is None:
            drift.append(f"{split}: {'not pinned' if a is None else 'no longer pinned'}")
            continue
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                drift.append(f"{split}.{key}: pinned {a.get(key)!r}, now {b.get(key)!r}")
    return drift


# --------------------------------------------------------------------------- #
# Leave-one-city-out reporting
# --------------------------------------------------------------------------- #
def loco(model, *, predictions_dir=PUBLISHED_DIR, benchmark_dir=BENCHMARK_DIR,
         op_threshold=0.0, floor=None, splits=POOLED_SPLITS):
    """Leave-one-city-out **reporting** over the pooled in-distribution splits.

    No model is retrained here (there is no LOCO training runner in this repo). For each
    pooled split ``s`` the model's score on ``s`` is set beside the micro-pooled (summed
    tp/fp/fn) and macro-mean scores over the *other* pooled splits it has predictions
    for. ``model`` is a published stem (``<model>__<split>.json`` in ``predictions_dir``)
    or ``"rampnet"``. Splits without a predictions file are listed in ``missing``.
    """
    cells, missing = {}, []
    for s in splits:
        bundle = os.path.join(benchmark_dir, s)
        if model == "rampnet":
            preds = "rampnet"
        else:
            path = os.path.join(predictions_dir, f"{slug(model)}__{s}.json")
            if not os.path.exists(path):
                missing.append(s)
                continue
            preds = load_predictions(path)
        r = score_split(bundle, preds, op_threshold=op_threshold, floor=floor, model=model)
        cells[s] = {k: r[k] for k in ("precision", "recall", "f1", "ap", "tp", "fp", "fn",
                                      "n_gt_recall")}

    def micro(names):
        # Same gating as aggregate(): precision over every pano, recall over the
        # recall-confirmed GT only (tp on unconfirmed panos is not in the recall pool).
        tp = sum(cells[n]["tp"] for n in names)
        fp = sum(cells[n]["fp"] for n in names)
        fn = sum(cells[n]["fn"] for n in names)
        n_gt = sum(cells[n]["n_gt_recall"] for n in names)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = (n_gt - fn) / n_gt if n_gt else 0.0
        return {"precision": p, "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn,
                "n_gt_recall": n_gt}

    def macro(names):
        out = {}
        for k in ("precision", "recall", "f1", "ap"):
            vals = [cells[n][k] for n in names if cells[n][k] is not None]
            out[k] = sum(vals) / len(vals) if len(vals) == len(names) and vals else None
        return out

    rows = []
    for s in cells:
        rest = [n for n in cells if n != s]
        rows.append({"held_out": s, "held_out_score": cells[s], "rest": rest,
                     "rest_micro": micro(rest) if rest else None,
                     "rest_macro": macro(rest) if rest else None})
    return {"model": model, "op_threshold": op_threshold, "floor": floor,
            "pool": list(splits), "missing": missing, "rows": rows,
            "note": "Reporting convention only: no model was retrained without the "
                    "held-out city. Micro = counts summed over the other splits (precision "
                    "over all panos, recall over recall-confirmed GT, as in aggregate()); "
                    "macro = mean of their per-split metrics.",
            "scorer_fingerprint": scorer_fingerprint(), "eval_sha256": eval_sha256()}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
PROTOCOL_TEXT = """\
RampNet benchmark scoring protocol (rampnet.eval; docs/eval_protocol_150.md)

  matching      {matching}; each prediction claims the nearest unclaimed GT point
                strictly within the radius; unmatched predictions within radius of an
                ignore point are neither TP nor FP
  radius        {radius} of panorama width, in a {scale_x} x {scale_y} scaled space
  wrap_x        True (x is cyclic across the panorama seam)
  precision     over all scored panos
  recall        over {recall_over} only
  AP            VOC all-point, recall-confirmed panos, full confidence range as given;
                an operating threshold never truncates it; the export floor does
  scorer        rampnet/{sources}
  fingerprint   {fingerprint} (the stamp in benchmark_eval/*.txt)

Where committed numbers come from:
  analysis_out/scoreboard.json, docs/model_scoreboard.md, docs/model_comparison.md
      rampnet.detection_eval via scripts/analysis/scoreboard.py (this protocol)
  scripts/model_comparison/yolo_baseline/benchmark_eval/
      rampnet.detection_eval via rescore_benchmark_eval.py (this protocol)
  README erratum manual_gold precision 0.949 (and stage_two/evaluate.py output)
      stage_two/evaluate.py: same 1:1 matcher, no fn gating, own decode; NOT this path
  RampNet verdict cross-check blocks
      rampnet/validation.py: per-detection verdicts, RampNet only
"""


def cli_root(cwd=None):
    """The checkout the CLI's default paths resolve against.

    The nearest of the current directory and its parents that holds
    ``benchmark/split_pins.json`` (so a shared editable install run from inside a
    worktree verifies that worktree), else the checkout this module was imported from. A
    non-editable install has neither; pass explicit paths."""
    here = os.path.abspath(cwd or os.getcwd())
    while True:
        if os.path.exists(os.path.join(here, "benchmark", "split_pins.json")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return REPO_ROOT
        here = parent


NAMED_PREDICTIONS = ("rampnet", "rampnet-op-cache")


def _cmd_score(args):
    named = args.predictions in NAMED_PREDICTIONS
    preds = args.predictions if named else load_predictions(args.predictions)
    res = score_split(args.bundle, preds, radius=args.radius, wrap_x=not args.no_wrap_x,
                      op_threshold=args.op_threshold, floor=args.floor, sweep=args.sweep,
                      pr_curve=args.pr_curve)
    if not named:
        res["predictions_sha256"] = _lf_sha256(args.predictions)
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    text = json.dumps(res, indent=1) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    ap = f"{res['ap']:.4f}" if res["ap"] is not None else "-"
    print(f"{res['split']} / {res['model']} @ op {res['op_threshold']}: "
          f"P {res['precision']:.4f}  R {res['recall']:.4f}  F1 {res['f1']:.4f}  AP {ap}  "
          f"tp/fp/fn/ign {res['tp']}/{res['fp']}/{res['fn']}/{res['ignored']}  "
          f"({res['n_panos']} panos, {res['n_gt_recall']} GT in recall pool)")
    print(f"  {res['n_panos_without_predictions']} bundle panos had no predictions; "
          f"{res['n_prediction_panos_not_in_bundle']} prediction panos are not in the bundle")
    print(f"  {res['ap_note']}; scorer {res['scorer_fingerprint']}"
          + (f"; written to {args.out}" if args.out else ""))
    return 0


def _cmd_pins(args):
    bench = os.path.join(cli_root(), "benchmark")
    if args.write:
        path = os.path.join(bench, "split_pins.json") if args.write is True else args.write
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(pins_payload(bench))
        print(f"wrote {path} ({len(PINNED_SPLITS)} splits)")
        return 0
    if args.verify:
        path = args.path or os.path.join(bench, "split_pins.json")
        drift = verify_pins(path, bench)
        if drift:
            print("split pins drifted:")
            for line in drift:
                print("  " + line)
            return 1
        print(f"{path}: all {len(PINNED_SPLITS)} pinned splits unchanged")
        return 0
    print(json.dumps(all_pins(bench), indent=1, sort_keys=True))
    return 0


def _cmd_loco(args):
    bench = os.path.join(cli_root(), "benchmark")
    res = loco(args.model, predictions_dir=args.predictions_dir
               or os.path.join(bench, "model_detections"), benchmark_dir=bench,
               op_threshold=args.op_threshold, floor=args.floor)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, indent=1) + "\n")
    print(f"LOCO report for {res['model']} @ op {res['op_threshold']} "
          "(reporting only, no retraining)")
    print(f"  {'held out':<20} {'F1':>6} {'rest F1 micro':>14} {'rest F1 macro':>14}")
    for row in res["rows"]:
        micro = row["rest_micro"]["f1"] if row["rest_micro"] else float("nan")
        macro = row["rest_macro"]["f1"] if row["rest_macro"] else float("nan")
        print(f"  {row['held_out']:<20} {row['held_out_score']['f1']:>6.3f} "
              f"{micro:>14.3f} {macro:>14.3f}")
    if res["missing"]:
        print(f"  no predictions for: {', '.join(res['missing'])}")
    return 0


def _cmd_protocol(_args):
    print(PROTOCOL_TEXT.format(
        matching=MATCHING_RULE, radius=PANO_RADIUS_NORMALIZED, scale_x=PANO_SCALE_X,
        scale_y=PANO_SCALE_Y, recall_over=RECALL_OVER,
        sources=", rampnet/".join(SCORER_SOURCES), fingerprint=scorer_fingerprint()))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m rampnet.eval",
                                 description="The RampNet benchmark scoring protocol (#150).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Score one prediction file against one bundle.")
    s.add_argument("--bundle", required=True, help="benchmark/<split> directory")
    s.add_argument("--predictions", required=True,
                   help="Prediction file (benchmark/model_detections format); 'rampnet' "
                        "for the bundle's own records.jsonl detections; or "
                        "'rampnet-op-cache' for RampNet's 0.05-floor no-TTA extraction in "
                        "analysis_out/op_cache/, the source of its published AP on the city "
                        "splits whose bundles stop at 0.55 (refused where the bundle itself "
                        "already reaches the floor, e.g. manual_gold).")
    s.add_argument("--op-threshold", type=float, default=0.0,
                   help="Drop predictions below this for P/R/F1 (AP is never truncated "
                        "by it). Default 0: score everything.")
    s.add_argument("--floor", type=float, default=None,
                   help="The confidence the predictions were exported down to; anything "
                        "below is dropped, and a warning is printed when the lowest "
                        "confidence present sits more than 0.1 above it.")
    s.add_argument("--radius", type=float, default=PANO_RADIUS_NORMALIZED)
    s.add_argument("--no-wrap-x", action="store_true",
                   help="Do not wrap x across the seam (pre-#140 audits only).")
    s.add_argument("--sweep", action="store_true", help="Add a threshold sweep.")
    s.add_argument("--pr-curve", action="store_true", help="Include the PR curve points.")
    s.add_argument("--out", help="Write the full result JSON here.")
    s.set_defaults(func=_cmd_score)

    p = sub.add_parser("pins", help="Show, write or verify benchmark/split_pins.json.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", metavar="PATH", nargs="?", const=True,
                   help="Write the pins file (default benchmark/split_pins.json).")
    g.add_argument("--verify", action="store_true", help="Exit 1 if any pinned file drifted.")
    p.add_argument("--path", default=None,
                   help="Pins file to verify against (default benchmark/split_pins.json).")
    p.epilog = ("Paths resolve against the current directory when it is a RampNet checkout, "
                "else against the checkout rampnet was imported from.")
    p.set_defaults(func=_cmd_pins)

    lo = sub.add_parser("loco", help="Leave-one-city-out report (no retraining).")
    lo.add_argument("--model", required=True,
                    help="Published stem in --predictions-dir, or 'rampnet'.")
    lo.add_argument("--predictions-dir", default=None,
                    help="Default: benchmark/model_detections (see `pins --help` on paths).")
    lo.add_argument("--op-threshold", type=float, default=0.0)
    lo.add_argument("--floor", type=float, default=None)
    lo.add_argument("--out")
    lo.set_defaults(func=_cmd_loco)

    pr = sub.add_parser("protocol", help="Print the rule, constants and fingerprint.")
    pr.set_defaults(func=_cmd_protocol)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PredictionFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
