"""Pure helpers of the #258 step 3 fusion read-out, plus one end-to-end split (CPU only)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))

import validator_fusion_258 as vf  # noqa: E402


def test_logit_clips():
    v = vf.logit([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(v)) and v[1] == 0.0 and v[0] == pytest.approx(-v[2])


def test_fit_logistic_recovers_known_weights():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(size=20000), rng.normal(size=20000)])
    w_true = np.array([-0.5, 1.5, 0.7])
    y = rng.random(20000) < 1 / (1 + np.exp(-(X @ w_true)))
    w = vf.fit_logistic(X, y)
    assert w == pytest.approx(w_true, abs=0.08)


def test_best_at_precision_picks_highest_recall_meeting_the_bar():
    # rows: (tp, fp, recall tp, recall n_gt) at three thresholds
    c = np.array([[8, 8, 8, 10],     # P 0.50, R 0.8
                  [7, 2, 7, 10],     # P 0.78, R 0.7
                  [5, 0, 5, 10]])    # P 1.00, R 0.5
    assert vf.best_at_precision(c, 0.75) == 1
    assert vf.best_at_precision(c, 0.40) == 0
    assert vf.best_at_precision(c, 1.01) == -1


def test_best_at_precision_is_batched():
    c = np.array([[[8, 8, 8, 10], [5, 0, 5, 10]],
                  [[9, 1, 9, 10], [5, 0, 5, 10]]])
    assert vf.best_at_precision(c, np.array([[0.9], [0.9]])).tolist() == [1, 0]


def test_bootstrap_weights_sum_to_n():
    w = vf.bootstrap_weights(7, b=20, seed=1)
    assert w.shape == (20, 7) and np.all(w.sum(1) == 7)


def test_dev_split_rescoring_matches_labels_and_the_operating_point():
    """sao_paulo end to end from committed files: re-scored RampNet@0.30 equals the
    step-1 labels (asserted inside split_tables' caller), and the fit converges."""
    panos, rsq = vf.load_panos("sao_paulo")
    s = np.concatenate([p["s"] for p in panos])
    v = np.concatenate([p["v"] for p in panos])
    lab = np.concatenate([np.array(p["label"]) for p in panos])
    keep = lab != "ignored"
    w = vf.fit_logistic(vf.features(s[keep], v[keep]), lab[keep] == "tp")
    t = vf.split_tables(panos, rsq, w)
    vf.check_against_labels(panos, t["base"])
    # the threshold-only sweep at 0.30 is RampNet@0.30
    i = vf.THR_SWEEP.index(0.30)
    assert np.array_equal(t["thr"][i], t["base"])
