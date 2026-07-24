"""Tests for the corrected matching and AP computation (issue #9).

Pure-Python: these run anywhere, no torch needed.
    pytest tests/test_metrics.py -v
"""
from rampnet.metrics import calculate_ap_and_pr_curve, match_points, match_predictions

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


# --- match_points: the unscored/counted variant used by the Stage 1 evaluator ---
# Same geometry as match_predictions (nearest unclaimed GT within radius); the
# only difference is input order instead of confidence order, and count output.
# The load-bearing case is issue #18: a redundant point on an already-matched
# ramp is a FALSE POSITIVE, not an ignored point.

UNIT = dict(scale_x=1, scale_y=1)
R2 = 0.25  # radius 0.5 at unit scale -> plain Euclidean, easy to place points


def test_match_points_perfect_one_to_one():
    assert match_points([(0.0, 0.0), (10.0, 10.0)],
                        [(0.0, 0.0), (10.0, 10.0)], R2, **UNIT) == (2, 0, 0)


def test_match_points_redundant_point_is_false_positive():
    # Two predictions on one GT: the first claims it, the second finds only an
    # already-claimed point -> false positive, flagged redundant (issue #18).
    assert match_points([(0.0, 0.0), (0.1, 0.0)], [(0.0, 0.0)], R2, **UNIT) == (1, 1, 1)


def test_match_points_far_prediction_is_plain_fp_not_redundant():
    assert match_points([(5.0, 5.0)], [(0.0, 0.0)], R2, **UNIT) == (0, 1, 0)


def test_match_points_falls_through_to_an_unclaimed_gt():
    # Both GTs are in range of the second prediction; it must take the unclaimed
    # one rather than be written off as redundant.
    assert match_points([(0.0, 0.0), (0.1, 0.0)],
                        [(0.0, 0.0), (0.2, 0.0)], R2, **UNIT) == (2, 0, 0)


def test_match_points_claims_nearest_not_first_in_list():
    # Regression guard on the rule itself, not just the outcome.
    #
    #   gtA=0.40   gtB=0.80        preds: P=0.75 (in range of BOTH)
    #                                     Q=0.10 (in range of gtA ONLY)
    #
    # Nearest-unclaimed: P takes gtB (0.05 away, not gtA at 0.35), leaving gtA
    # for Q -> 2 TP. First-in-list-order would give gtA to P, stranding Q with
    # nothing unclaimed in range -> (1, 1, 1). The old Stage 1 evaluator did the
    # latter; this pins the corrected, shared behaviour.
    assert match_points([(0.75, 0.5), (0.10, 0.5)],
                        [(0.40, 0.5), (0.80, 0.5)], R2, **UNIT) == (2, 0, 0)


def test_match_points_unmatched_gt_is_not_a_false_positive():
    # A GT with nothing near it is a false negative (recall side); match_points
    # only accounts for the prediction side.
    assert match_points([], [(0.0, 0.0)], R2, **UNIT) == (0, 0, 0)


def test_match_points_redundancy_lifts_the_precision_denominator():
    tp, fp, redundant = match_points([(0.0, 0.0), (0.1, 0.0)], [(0.0, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (1, 1, 1)
    assert tp / (tp + fp) == 0.5  # pre-#18 this was 1/1 -- the optimism being fixed


def test_match_points_scaling_is_anisotropic():
    # The pano is 2:1, so the SAME normalized offset is a different distance in x
    # vs y. At the real 1024x512 scales with radius 0.022, an offset of 0.03:
    #   x -> 30.7 units (outside the 22.5 radius) : no match
    #   y -> 15.4 units (inside)                  : match
    r2 = (0.022 * 1024) ** 2
    assert match_points([(0.53, 0.5)], [(0.5, 0.5)], r2, 1024, 512) == (0, 1, 0)
    assert match_points([(0.5, 0.53)], [(0.5, 0.5)], r2, 1024, 512) == (1, 0, 0)
