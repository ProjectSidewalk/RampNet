"""Guards for the model-agnostic detection scorer (rampnet/detection_eval.py).

Two layers: unit tests that pin the ground-truth construction and matching
semantics with trivial unit-scale geometry, and an integration test that scores
RampNet's own committed detections against the derived ground truth and checks it
reproduces the published verdict-based numbers (the harness self-validation).
"""
import json
import os

from rampnet.detection_eval import (
    GroundTruth, PanoScore,
    build_ground_truth, score_pano, aggregate,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unit scale + radius 0.5 (radius_sq 0.25): plain Euclidean on raw coords.
UNIT = dict(scale_x=1, scale_y=1, radius_sq=0.25)


# --- build_ground_truth -----------------------------------------------------

def test_ground_truth_partitions_verdicts():
    dets = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)]
    verdicts = [True, False, "unsure", "duplicate"]
    missed = [{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6, "unsure": True}]
    gt = build_ground_truth(dets, verdicts, missed, no_missed=False)
    assert gt.gt_points == [(0.1, 0.1), (0.5, 0.5)]        # true det + confident miss
    assert gt.ignore_points == [(0.3, 0.3), (0.6, 0.6)]    # unsure det + unsure miss
    assert gt.fn_confirmed is True                          # a missed mark exists


def test_ground_truth_accepts_dicts_and_string_verdicts():
    dets = [{"x_normalized": 0.1, "y_normalized": 0.2},
            {"x_normalized": 0.3, "y_normalized": 0.4}]
    gt = build_ground_truth(dets, ["true", "false"], missed=[], no_missed=True)
    assert gt.gt_points == [(0.1, 0.2)]
    assert gt.ignore_points == []
    assert gt.fn_confirmed is True   # explicit complete-scan attestation


def test_ground_truth_unconfirmed_when_no_missed_false_and_no_marks():
    gt = build_ground_truth([(0.1, 0.1)], [True], missed=[], no_missed=False)
    assert gt.fn_confirmed is False


def test_ground_truth_rejects_misaligned_verdicts():
    try:
        build_ground_truth([(0.1, 0.1)], [True, False], missed=[], no_missed=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError on misaligned detections/verdicts")


# --- score_pano -------------------------------------------------------------

def _gt(gt_points, ignore_points=(), fn_confirmed=True):
    return GroundTruth(list(gt_points), list(ignore_points), fn_confirmed)


def test_score_true_positive_and_false_positive():
    gt = _gt([(0.0, 0.0)])
    assert score_pano([(0.0, 0.0)], gt, **UNIT)[:3] == (1, 0, 0)   # tp
    assert score_pano([(5.0, 5.0)], gt, **UNIT)[:3] == (0, 1, 0)   # fp


def test_score_ignore_zone_drops_prediction():
    gt = _gt([], ignore_points=[(0.0, 0.0)])
    tp, fp, ignored = score_pano([(0.1, 0.0)], gt, **UNIT)[:3]
    assert (tp, fp, ignored) == (0, 0, 1)   # neither TP nor FP


def test_score_duplicate_is_false_positive():
    gt = _gt([(0.0, 0.0)])
    tp, fp, ignored = score_pano([(0.0, 0.0, 0.9), (0.1, 0.0, 0.5)], gt, **UNIT)[:3]
    assert (tp, fp, ignored) == (1, 1, 0)   # second hit on the same ramp -> FP


def test_score_gt_takes_priority_over_ignore():
    gt = _gt([(0.0, 0.0)], ignore_points=[(0.1, 0.0)])
    tp, fp, ignored = score_pano([(0.05, 0.0)], gt, **UNIT)[:3]
    assert (tp, fp, ignored) == (1, 0, 0)   # claims the GT, not ignored


def test_score_redundant_hit_inside_ignore_zone_is_ignored():
    # A second hit on an already-claimed ramp is normally an FP, but if it also
    # lands in an ignore zone the abstention wins: the ignore fallback applies to
    # every prediction the matcher left unassigned, redundant ones included.
    gt = _gt([(0.0, 0.0)], ignore_points=[(0.2, 0.0)])
    tp, fp, ignored = score_pano([(0.0, 0.0, 0.9), (0.2, 0.0, 0.5)], gt, **UNIT)[:3]
    assert (tp, fp, ignored) == (1, 0, 1)


def test_score_claims_nearest_gt_not_first_in_list():
    # Shared-matcher semantics reaching score_pano: the first prediction is in
    # range of both ramps and takes the *nearer* one, leaving the far ramp for
    # the second prediction. First-in-list matching would score this (1 TP, 1 FP).
    gt = _gt([(0.4, 0.0), (0.0, 0.0)])
    tp, fp, ignored = score_pano([(0.05, 0.0), (0.4, 0.0)], gt, **UNIT)[:3]
    assert (tp, fp, ignored) == (2, 0, 0)


# --- aggregate --------------------------------------------------------------

def test_aggregate_gates_recall_on_confirmed_panos():
    scores = [
        PanoScore(tp=1, fp=0, ignored=0, n_gt=2, fn_confirmed=True),
        PanoScore(tp=1, fp=1, ignored=0, n_gt=1, fn_confirmed=False),
    ]
    r = aggregate(scores)
    assert r.precision == 2 / 3          # all detections count for precision
    assert r.recall == 1 / 2             # only the confirmed pano's GT counts
    assert r.fn == 1                     # 2 GT - 1 TP on the confirmed pano
    assert r.n_recall_panos == 1 and r.n_panos == 2


# --- integration: RampNet reproduces its published numbers ------------------

def _load_bundle(city):
    cdir = os.path.join(REPO_ROOT, "benchmark", city)
    records = {}
    with open(os.path.join(cdir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    verdicts = json.load(open(os.path.join(cdir, "verdicts.json"), encoding="utf-8"))["panos"]
    return records, verdicts


def _score_rampnet(city):
    records, verdicts = _load_bundle(city)
    pano_scores = []
    for pid, entry in verdicts.items():
        dets = records[pid]["detections"]
        gt = build_ground_truth(dets, entry["dets"], entry["missed"], entry["no_missed"])
        preds = [(d["x_normalized"], d["y_normalized"], d["confidence"]) for d in dets]
        pano_scores.append(score_pano(preds, gt))
    return aggregate(pano_scores)


def test_rampnet_reproduces_published_numbers():
    # Published verdict-based numbers (benchmark/README.md). The point-matching
    # scorer should land within a small tolerance; the only source of drift is a
    # RampNet "false" detection occasionally falling within radius of a real GT
    # point, which the per-detection human verdict scored differently.
    expected = {"richmond": (0.960, 0.765), "bend": (0.954, 0.758)}
    for city, (exp_p, exp_r) in expected.items():
        r = _score_rampnet(city)
        assert abs(r.precision - exp_p) <= 0.03, f"{city} precision {r.precision:.3f} vs {exp_p}"
        assert abs(r.recall - exp_r) <= 0.03, f"{city} recall {r.recall:.3f} vs {exp_r}"
