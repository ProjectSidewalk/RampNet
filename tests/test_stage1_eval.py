"""Guards for the Stage 1 dataset evaluator's point matching
(stage_one/dataset_evaluation/evaluate.py).

The load-bearing case is #18: a redundant generated point that lands on an
already-matched ground-truth ramp must be scored as a false positive, not
silently ignored (which made the published Stage 1 agreement precision
optimistic). These tests exercise ``match_points`` directly with unit scales so
the geometry is trivial to reason about.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "stage_one", "dataset_evaluation"))
from evaluate import match_points  # noqa: E402

# Unit scales + radius 0.5 (radius_sq 0.25): distance is plain Euclidean on the
# raw coordinates, so "within 0.5" is easy to place points around.
UNIT = dict(x_scale=1, y_scale=1)
R2 = 0.25


def test_perfect_one_to_one():
    tp, fp, redundant = match_points([(0.0, 0.0), (10.0, 10.0)],
                                     [(0.0, 0.0), (10.0, 10.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (2, 0, 0)


def test_redundant_point_on_claimed_gt_is_false_positive():
    # Two predictions on a single GT: the first claims it (TP), the second finds
    # only the already-claimed GT in range -> false positive, flagged redundant.
    tp, fp, redundant = match_points([(0.0, 0.0), (0.1, 0.0)],
                                     [(0.0, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (1, 1, 1)


def test_far_prediction_is_plain_false_positive_not_redundant():
    tp, fp, redundant = match_points([(5.0, 5.0)], [(0.0, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (0, 1, 0)


def test_prediction_skips_claimed_gt_for_an_unclaimed_one():
    # Both GTs sit within radius of the second prediction. The first prediction
    # claims gtA; the second must fall through to the unclaimed gtB (TP), not be
    # marked redundant.
    tp, fp, redundant = match_points([(0.0, 0.0), (0.1, 0.0)],
                                     [(0.0, 0.0), (0.2, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (2, 0, 0)


def test_missed_gt_does_not_create_false_positive():
    # A GT with no prediction nearby is a false negative (recall side), never a
    # false positive; match_points only accounts for the prediction side.
    tp, fp, redundant = match_points([], [(0.0, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (0, 0, 0)


def test_redundant_counts_lift_fp_denominator():
    # Precision = TP / (TP + FP). Two predictions land on the one ramp; the fix
    # makes the extra one a FP, so precision is 1/2. Pre-fix (redundant ignored)
    # it would have been 1/1 -- the optimism issue #18 is about.
    tp, fp, redundant = match_points([(0.0, 0.0), (0.1, 0.0)],
                                     [(0.0, 0.0)], R2, **UNIT)
    assert (tp, fp, redundant) == (1, 1, 1)
    assert tp / (tp + fp) == 0.5


def test_default_scales_are_anisotropic():
    # With the real 1024x512 scales, a prediction exactly on a GT still matches.
    tp, fp, redundant = match_points([(0.5, 0.5)], [(0.5, 0.5)],
                                     (0.022 * 1024) ** 2)
    assert (tp, fp, redundant) == (1, 0, 0)
