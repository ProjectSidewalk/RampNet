"""Model-agnostic detection scoring against the human-verified benchmark.

`rampnet/validation.py` scores RampNet against the reviewers' *per-detection*
verdicts, which are specific to RampNet's own detections and so can't score a
different model. This module derives a **model-agnostic ground truth** from the
same review and scores *any* detector's points against it identically, so
RampNet and VLMs (Gemini, Qwen, ...) are compared on equal footing.

Per pano, the ground truth is:
  - **GT ramp points** — detections the reviewer confirmed real (`True`) plus the
    ramps they marked as missed (non-`unsure`). This is the reviewer's complete
    enumeration of real curb ramps in the pano.
  - **Ignore points** — detections the reviewer marked `unsure`, and `unsure`
    missed marks. A prediction landing here is scored as neither TP nor FP
    (mirrors `validation.collect`'s `unsure` abstention): the reviewer couldn't
    tell from the imagery, so no model should be rewarded or penalized there.
  - `False`/`duplicate` detections contribute to neither set. A duplicate is a
    second hit on a ramp already in GT, so it becomes a false positive naturally
    when a model's own points are matched greedily 1:1.

Every detector's output is reduced to center points `(x, y[, confidence])` and
matched to the GT points by `rampnet.metrics.greedy_match` — the same matcher
both other evaluators use — within a normalized radius (the pano value 0.022)
and the same anisotropic x/y scaling (1024/512) the rest of the pipeline uses.
Precision is over all panos; recall is over panos whose missed-ramp check is
confirmed (so un-scanned panos can't bias it), exactly as `validation.collect`
gates recall.

When every prediction carries a confidence — RampNet, and the open-vocabulary
detectors (OWLv2 / Grounding DINO), but not the chat VLMs — `aggregate` also
returns **AP and a PR curve** (via `rampnet.metrics`), so those models can be
compared across their whole operating range instead of at one arbitrary point.

Caveat: the GT was assembled during a RampNet review, so it is "RampNet-anchored"
— a reviewer scanning fresh for another model might catch a few more ramps. The
complete-scan attestation (`no_missed`) mitigates this; it is documented in
`docs/model_comparison.md`.
"""
import os
from collections import namedtuple

from rampnet.metrics import calculate_ap_and_pr_curve, greedy_match
from rampnet.validation import wilson_interval

# Pano coordinate space and matching radius — identical to rampnet/metrics.py and
# stage_two/evaluate.py so numbers stay comparable across the codebase.
PANO_SCALE_X = 1024
PANO_SCALE_Y = 512
PANO_RADIUS_NORMALIZED = 0.022

GroundTruth = namedtuple("GroundTruth", ["gt_points", "ignore_points", "fn_confirmed"])
# details: one (confidence, is_true_positive) per *scored* prediction, in match
# order (ignored predictions are excluded — they are neither). Feeds AP.
PanoScore = namedtuple("PanoScore", ["tp", "fp", "ignored", "n_gt", "fn_confirmed", "details"],
                       defaults=((),))
ScoreReport = namedtuple("ScoreReport", [
    "precision", "recall", "f1",
    "precision_ci", "recall_ci",
    "tp", "fp", "fn", "ignored",
    "n_gt_recall", "n_panos", "n_recall_panos",
    "ap", "pr_curve",
], defaults=(None, None))


def radius_sq_for(radius_normalized=PANO_RADIUS_NORMALIZED, scale_x=PANO_SCALE_X):
    """Squared match radius in the scaled coordinate space (x-axis units)."""
    return (radius_normalized * scale_x) ** 2


def _xy(point):
    """Accept a detection as a dict (x_normalized/y_normalized) or an (x, y[, ...]) tuple."""
    if isinstance(point, dict):
        return float(point["x_normalized"]), float(point["y_normalized"])
    return float(point[0]), float(point[1])


def _is_true(verdict):
    # Raw verdicts.json carries Python bools; the parquet/encoded form uses strings.
    return verdict is True or verdict == "true"


def _is_unsure(verdict):
    return verdict == "unsure"


def build_ground_truth(detections, det_verdicts, missed, no_missed):
    """Derive the model-agnostic ground truth for one pano.

    detections: list of the reviewed RampNet detections (dicts or (x, y) tuples),
        positionally aligned to det_verdicts.
    det_verdicts: list of True / False / "unsure" / "duplicate" (or "true"/"false").
    missed: list of {"x", "y", "unsure"?} marks the reviewer added.
    no_missed: bool — the reviewer attested a complete scan with no missed ramps.

    Returns a GroundTruth. ``fn_confirmed`` is True when the pano's missed-ramp
    check is trustworthy (an explicit no-missed attestation, or at least one
    missed mark), which is the gate for including the pano in recall.
    """
    if len(detections) != len(det_verdicts):
        raise ValueError(
            f"detections ({len(detections)}) and det_verdicts ({len(det_verdicts)}) misaligned")

    gt_points, ignore_points = [], []
    for det, verdict in zip(detections, det_verdicts):
        if _is_true(verdict):
            gt_points.append(_xy(det))
        elif _is_unsure(verdict):
            ignore_points.append(_xy(det))
        # False / duplicate -> neither.

    for mark in missed:
        pt = (float(mark["x"]), float(mark["y"]))
        if mark.get("unsure"):
            ignore_points.append(pt)
        else:
            gt_points.append(pt)

    fn_confirmed = bool(no_missed) or len(missed) > 0
    return GroundTruth(gt_points, ignore_points, fn_confirmed)


def yolo_ground_truth(label_path):
    """Parse one YOLO-format manual label file into a GroundTruth.

    The 1,000-pano manual gold set (``manual_labels/*.txt``, issue #58) is
    labeled independently of any model, so unlike :func:`build_ground_truth`
    there is no verdict structure to derive from: each ``class cx cy w h`` box
    becomes its center point in ``gt_points`` (the same box->center reduction
    the VLM challengers get), ``ignore_points`` is empty (the labels carry no
    'unsure' class — no abstention for any model), and ``fn_confirmed`` is True
    (full manual labeling is a complete scan, so an empty file is a genuine
    negative pano that joins the recall pool).

    Strict on purpose: these labels are a benchmark artifact, so a malformed or
    out-of-range line raises instead of being skipped (stage_two/evaluate.py's
    tolerant reader would silently shrink the GT).
    """
    points = []
    with open(label_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{lineno}: expected 'class cx cy w h', got {line.strip()!r}")
            try:
                cx, cy = float(parts[1]), float(parts[2])
                float(parts[3]), float(parts[4])
            except ValueError:
                raise ValueError(
                    f"{label_path}:{lineno}: non-numeric field in {line.strip()!r}") from None
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError(
                    f"{label_path}:{lineno}: center ({cx}, {cy}) outside [0, 1]")
            points.append((cx, cy))
    return GroundTruth(points, [], True)


def load_yolo_ground_truths(labels_dir):
    """{pano_id: GroundTruth} for every ``.txt`` label file in ``labels_dir``."""
    gts = {}
    for name in sorted(os.listdir(labels_dir)):
        if name.endswith(".txt"):
            gts[name[:-4]] = yolo_ground_truth(os.path.join(labels_dir, name))
    return gts


def prediction_confidence(point):
    """The confidence a prediction carries, or None (chat VLMs emit none).

    Shared with the comparison CLI's re-scoring/sweep helpers so "does this
    prediction carry a score" is decided one way everywhere."""
    if isinstance(point, dict):
        return point.get("confidence")
    return point[2] if len(point) > 2 else None


def score_pano(pred_points, gt, radius_sq=None, scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """Score one pano's predictions against its GroundTruth.

    Greedy one-to-one matching via :func:`rampnet.metrics.greedy_match`,
    highest-confidence first when confidences are present (else input order):
    each prediction claims the nearest unclaimed GT point strictly within
    ``radius``. A prediction with no *unclaimed* GT in range that falls within
    radius of an ignore point is *ignored* (neither TP nor FP); otherwise it is a
    false positive — including a redundant second hit on an already-claimed ramp,
    matching ``match_predictions``. GT always takes priority over an ignore point.

    Returns a PanoScore. False negatives are ``n_gt - tp`` (computed in aggregate,
    and only for panos whose recall is confirmed).
    """
    if radius_sq is None:
        radius_sq = radius_sq_for(scale_x=scale_x)

    gt_pts = gt.gt_points
    ignore_pts = gt.ignore_points

    confs = [prediction_confidence(p) for p in pred_points]
    if any(c is not None for c in confs):
        order = sorted(range(len(pred_points)),
                       key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                       reverse=True)
        preds = [pred_points[i] for i in order]
    else:
        preds = list(pred_points)

    # GT matching is the shared core (rampnet.metrics); only the ignore-point
    # fallback is specific to this scorer, and it applies solely to predictions
    # the matcher left unassigned — which is what gives GT priority over ignores.
    assignments = greedy_match([_xy(p) for p in preds], gt_pts,
                               radius_sq, scale_x, scale_y)

    tp = fp = ignored = 0
    details = []
    for p, (gt_index, _) in zip(preds, assignments):
        if gt_index >= 0:
            tp += 1
            details.append((prediction_confidence(p), True))
            continue
        px_n, py_n = _xy(p)
        px, py = px_n * scale_x, py_n * scale_y
        in_ignore = any(
            (px - ix * scale_x) ** 2 + (py - iy * scale_y) ** 2 < radius_sq
            for ix, iy in ignore_pts)
        if in_ignore:
            ignored += 1        # neither TP nor FP, so it stays out of the PR curve too
        else:
            fp += 1
            details.append((prediction_confidence(p), False))

    return PanoScore(tp=tp, fp=fp, ignored=ignored, n_gt=len(gt_pts),
                     fn_confirmed=gt.fn_confirmed, details=details)


def aggregate(pano_scores):
    """Combine per-pano PanoScores into precision/recall/F1 with Wilson CIs.

    Precision uses detections from every pano; recall uses only panos whose
    missed-ramp check is confirmed (``fn_confirmed``), so panos that weren't
    fully scanned don't deflate recall. Returns a ScoreReport.

    **AP** is filled in only when every scored prediction carries a confidence —
    RampNet always, and the open-vocabulary detectors (OWLv2, Grounding DINO); chat
    VLMs emit no calibrated score, so they get ``ap=None`` and an operating point
    only. It is computed over the **recall-confirmed subset** alone, because AP
    needs one consistent denominator for recall (``n_gt_recall``) and predictions
    from unscanned panos have no GT to be measured against. That makes AP a
    slightly different slice than the operating-point precision above it, which
    counts every pano — the tradeoff is noted in docs/model_comparison.md.
    """
    tp_all = sum(s.tp for s in pano_scores)
    fp_all = sum(s.fp for s in pano_scores)
    ignored_all = sum(s.ignored for s in pano_scores)

    recall_scores = [s for s in pano_scores if s.fn_confirmed]
    tp_recall = sum(s.tp for s in recall_scores)
    n_gt_recall = sum(s.n_gt for s in recall_scores)

    n_preds = tp_all + fp_all
    precision = tp_all / n_preds if n_preds else 0.0
    recall = tp_recall / n_gt_recall if n_gt_recall else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    ap, pr_curve = None, None
    details = [d for s in recall_scores for d in s.details]
    if details and n_gt_recall and all(c is not None for c, _ in details):
        ap, plot_recalls, plot_precisions, _, _ = calculate_ap_and_pr_curve(details, n_gt_recall)
        pr_curve = (plot_recalls, plot_precisions)

    return ScoreReport(
        precision=precision,
        recall=recall,
        f1=f1,
        precision_ci=wilson_interval(tp_all, n_preds),
        recall_ci=wilson_interval(tp_recall, n_gt_recall),
        tp=tp_all,
        fp=fp_all,
        fn=n_gt_recall - tp_recall,
        ignored=ignored_all,
        n_gt_recall=n_gt_recall,
        n_panos=len(pano_scores),
        n_recall_panos=len(recall_scores),
        ap=ap,
        pr_curve=pr_curve,
    )
