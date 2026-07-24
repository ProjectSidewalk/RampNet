"""Shared detection metrics for RampNet evaluators.

The matching here replaces the pre-release per-evaluator logic, which had two
defects that biased reported numbers upward (issue #9): a single prediction
within radius of two ground-truth points could be counted as two true
positives, and duplicate detections of an already-matched ground-truth point
were dropped entirely instead of counted as false positives.
"""


def _greedy_match(pred_points, gt_points, radius_sq, scale_x, scale_y):
    """Assign each prediction, in the order given, to the nearest unclaimed GT.

    The single matching core shared by every RampNet evaluator — keep it that
    way. Returns one ``(gt_index, saw_in_range)`` pair per prediction:

    - ``gt_index`` is the claimed ground-truth index, or ``-1`` when nothing
      unclaimed was within radius (i.e. the prediction is a false positive).
    - ``saw_in_range`` says whether *any* ground truth, claimed or not, was
      within radius. That distinguishes a **redundant** second hit on a ramp
      already counted from a detection with no ramp anywhere near it — a
      distinction some callers report separately, though both are false
      positives.
    """
    gt_scaled = [(gx * scale_x, gy * scale_y) for gx, gy in gt_points]
    claimed = [False] * len(gt_scaled)
    assignments = []
    for x_norm, y_norm in pred_points:
        pred_x, pred_y = x_norm * scale_x, y_norm * scale_y
        best_k, best_dist_sq, saw_in_range = -1, radius_sq, False
        for k, (gt_x, gt_y) in enumerate(gt_scaled):
            dist_sq = (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2
            if dist_sq < radius_sq:
                saw_in_range = True
                if not claimed[k] and dist_sq < best_dist_sq:
                    best_dist_sq, best_k = dist_sq, k
        if best_k >= 0:
            claimed[best_k] = True
        assignments.append((best_k, saw_in_range))
    return assignments


def match_predictions(pred_peaks, gt_points, radius_sq, scale_x, scale_y):
    """Greedy one-to-one matching of predicted peaks to ground-truth points.

    pred_peaks: iterable of (x_norm, y_norm, confidence).
    gt_points: iterable of (x_norm, y_norm).
    radius_sq: squared matching radius, in the scaled coordinate space.
    scale_x/scale_y: factors mapping normalized coords into that space.

    Predictions are processed in descending confidence order; each claims the
    nearest unclaimed ground-truth point strictly within the radius (a true
    positive). A prediction with no unclaimed ground truth in range — including
    a duplicate detection of an already-claimed point — is a false positive.

    Returns a list of (confidence, is_true_positive), one entry per prediction.
    """
    preds_sorted = sorted(pred_peaks, key=lambda p: p[2], reverse=True)
    assignments = _greedy_match([(p[0], p[1]) for p in preds_sorted],
                                gt_points, radius_sq, scale_x, scale_y)
    return [(p[2], gt_index >= 0) for p, (gt_index, _) in zip(preds_sorted, assignments)]


def match_points(pred_points, gt_points, radius_sq, scale_x, scale_y):
    """Match unscored points (no confidence) and summarise as counts.

    The Stage 1 dataset evaluator's entry point: generated label points carry no
    confidence, so they are matched in input order rather than by score, but the
    geometry is identical to :func:`match_predictions` — nearest unclaimed
    ground truth within radius.

    Returns ``(tp, fp, n_redundant)``. ``n_redundant`` is the subset of ``fp``
    that landed on an already-claimed ramp; it is a reported diagnostic, **not**
    a third outcome — redundant points are false positives, matching
    ``match_predictions`` and ``rampnet.validation``'s duplicate handling
    (issue #18).
    """
    assignments = _greedy_match(pred_points, gt_points, radius_sq, scale_x, scale_y)
    tp = sum(1 for gt_index, _ in assignments if gt_index >= 0)
    n_redundant = sum(1 for gt_index, saw in assignments if gt_index < 0 and saw)
    return tp, len(assignments) - tp, n_redundant


def calculate_ap_and_pr_curve(all_predictions_details, total_gt_points):
    """All-point interpolated AP (VOC-style precision envelope) and PR curve.

    Returns (ap, plot_recalls, plot_precisions, sorted_confidences,
    sorted_tp_flags). The plotted curve uses the same monotone precision
    envelope the AP integrates, so the two are consistent.
    """
    if total_gt_points == 0 or not all_predictions_details:
        return 0.0, [0.0], [0.0], [], []

    sorted_preds = sorted(all_predictions_details, key=lambda x: x[0], reverse=True)
    sorted_confidences = [p[0] for p in sorted_preds]
    sorted_tp_flags = [p[1] for p in sorted_preds]

    tp_count = 0
    fp_count = 0
    raw_recalls_list = []
    raw_precisions_list = []
    for is_tp in sorted_tp_flags:
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1
        raw_precisions_list.append(tp_count / (tp_count + fp_count))
        raw_recalls_list.append(tp_count / total_gt_points)

    # Monotone (max-envelope) interpolation, computed right-to-left.
    interp_precisions = list(raw_precisions_list)
    for k in range(len(interp_precisions) - 2, -1, -1):
        interp_precisions[k] = max(interp_precisions[k], interp_precisions[k + 1])

    ap = 0.0
    last_recall_val = 0.0
    for i, is_tp in enumerate(sorted_tp_flags):
        if is_tp:
            ap += interp_precisions[i] * (raw_recalls_list[i] - last_recall_val)
            last_recall_val = raw_recalls_list[i]

    plot_recalls = [0.0] + raw_recalls_list
    plot_precisions = [1.0] + interp_precisions
    if plot_recalls[-1] < 1.0:
        plot_recalls.append(plot_recalls[-1])
        plot_precisions.append(0.0)
    return ap, plot_recalls, plot_precisions, sorted_confidences, sorted_tp_flags


def calculate_pr_rc_confidence_curves(sorted_confidences, sorted_tp_flags, total_gt_points):
    """Precision and recall as functions of the confidence threshold."""
    if not sorted_confidences:
        conf_thresholds_for_plot = [0.0, 1.0]
        if total_gt_points > 0:
            precisions_at_thresholds_for_plot = [1.0, 1.0]
        else:
            precisions_at_thresholds_for_plot = [0.0, 0.0]
        recalls_at_thresholds_for_plot = [0.0, 0.0]
        return conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot

    conf_thresholds_unique_desc = []
    precisions_at_thresholds_desc = []
    recalls_at_thresholds_desc = []
    tp_count_cumulative = 0
    for i in range(len(sorted_confidences)):
        if sorted_tp_flags[i]:
            tp_count_cumulative += 1
        num_preds_cumulative = i + 1
        current_precision = tp_count_cumulative / num_preds_cumulative
        current_recall = tp_count_cumulative / total_gt_points if total_gt_points > 0 else 0.0
        current_confidence_threshold = sorted_confidences[i]
        if i == len(sorted_confidences) - 1 or sorted_confidences[i + 1] < current_confidence_threshold:
            conf_thresholds_unique_desc.append(current_confidence_threshold)
            precisions_at_thresholds_desc.append(current_precision)
            recalls_at_thresholds_desc.append(current_recall)

    conf_thresholds_for_plot = list(reversed(conf_thresholds_unique_desc))
    precisions_at_thresholds_for_plot = list(reversed(precisions_at_thresholds_desc))
    recalls_at_thresholds_for_plot = list(reversed(recalls_at_thresholds_desc))

    if not conf_thresholds_for_plot or conf_thresholds_for_plot[0] > 0.0:
        conf_thresholds_for_plot.insert(0, 0.0)
        precisions_at_thresholds_for_plot.insert(0, precisions_at_thresholds_for_plot[0] if precisions_at_thresholds_for_plot else 0.0)
        recalls_at_thresholds_for_plot.insert(0, recalls_at_thresholds_for_plot[0] if recalls_at_thresholds_for_plot else 0.0)

    if conf_thresholds_for_plot[-1] < 1.0:
        conf_thresholds_for_plot.append(1.0)
        precisions_at_thresholds_for_plot.append(1.0 if total_gt_points > 0 else 0.0)
        recalls_at_thresholds_for_plot.append(0.0)
    elif recalls_at_thresholds_for_plot[-1] == 0.0:
        precisions_at_thresholds_for_plot[-1] = 1.0 if total_gt_points > 0 else 0.0

    return conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot
