"""Tests for the corrected matching and AP computation (issue #9).

Pure-Python: these run anywhere, no torch needed.
    pytest tests/test_metrics.py -v
"""
from rampnet.metrics import calculate_ap_and_pr_curve, match_predictions

RADIUS_SQ = 10.0 ** 2
SCALE = 100.0  # normalized coords x100 -> pixel-ish space


def test_one_prediction_cannot_claim_two_gts():
    # One high-confidence prediction sits within radius of two GT points.
    preds = [(0.5, 0.5, 0.9)]
    gts = [(0.52, 0.5), (0.48, 0.5)]
    details = match_predictions(preds, gts, RADIUS_SQ, SCALE, SCALE)
    assert details == [(0.9, True)]  # exactly one TP, never two


def test_duplicate_detection_is_false_positive():
    # Two predictions on the same single GT: best one matches, the duplicate
    # must be a false positive (pre-release code dropped it entirely).
    preds = [(0.5, 0.5, 0.9), (0.51, 0.5, 0.8)]
    gts = [(0.5, 0.5)]
    details = match_predictions(preds, gts, RADIUS_SQ, SCALE, SCALE)
    assert sorted(details, reverse=True) == [(0.9, True), (0.8, False)]


def test_higher_confidence_matches_first():
    # The high-confidence prediction claims the GT even if listed second.
    preds = [(0.51, 0.5, 0.3), (0.5, 0.5, 0.9)]
    gts = [(0.5, 0.5)]
    details = match_predictions(preds, gts, RADIUS_SQ, SCALE, SCALE)
    assert (0.9, True) in details
    assert (0.3, False) in details


def test_prediction_outside_radius_is_false_positive():
    preds = [(0.5, 0.5, 0.9)]
    gts = [(0.9, 0.9)]
    details = match_predictions(preds, gts, RADIUS_SQ, SCALE, SCALE)
    assert details == [(0.9, False)]


def test_nearest_unclaimed_gt_wins():
    # A prediction between two GTs claims the nearer one, leaving the farther
    # for the next prediction.
    preds = [(0.50, 0.5, 0.9), (0.60, 0.5, 0.8)]
    gts = [(0.49, 0.5), (0.58, 0.5)]
    details = match_predictions(preds, gts, RADIUS_SQ, SCALE, SCALE)
    assert details == [(0.9, True), (0.8, True)]


def test_ap_perfect_detector():
    details = [(0.9, True), (0.8, True), (0.7, True)]
    ap, _, _, _, _ = calculate_ap_and_pr_curve(details, total_gt_points=3)
    assert abs(ap - 1.0) < 1e-9


def test_ap_interpolation_uses_max_envelope():
    # TP, FP, TP over 2 GTs. Raw precisions: 1.0, 0.5, 2/3.
    # Interpolated envelope at the first TP is max(1.0, 2/3)=1.0, at the
    # second TP 2/3. AP = 0.5*1.0 + 0.5*(2/3).
    details = [(0.9, True), (0.8, False), (0.7, True)]
    ap, _, _, _, _ = calculate_ap_and_pr_curve(details, total_gt_points=2)
    assert abs(ap - (0.5 + 0.5 * (2.0 / 3.0))) < 1e-9


def test_ap_no_predictions_or_no_gt():
    assert calculate_ap_and_pr_curve([], 5)[0] == 0.0
    assert calculate_ap_and_pr_curve([(0.9, False)], 0)[0] == 0.0
