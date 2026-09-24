"""Unit tests for the numbers behind docs/pu_training_86.md (#86 item 5).

Synthetic fixtures only: the real inputs live on the #178 and #183 branches. What is tested is
the part a wrong reading would silently bias: the pi_U formula (the prior among *untagged*
labels, not the marginal prior) and the exclusion, which must drop HF train rows on a panorama
of either test set and rows of the same city within 10 m of any test label, and nothing else.
"""
import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import pu_training_86_plan as pu  # noqa: E402


def test_pi_unlabeled_censoring_formula():
    # pi = P(t), o = P(tagged t); among the untagged, P(t) = (pi - o) / (1 - o).
    assert pu.pi_unlabeled(0.389, 0.169) == pytest.approx(0.2647, abs=1e-4)
    assert pu.pi_unlabeled(0.5, 0.0) == pytest.approx(0.5)      # nothing tagged: U is the marginal
    assert pu.pi_unlabeled(0.01, 0.02) == 0.0                   # prior below the tag rate: clamped


def _row(uid, split, city, pano, lat, lng):
    d = dict(label_uid=uid, split=split, city=city, pano_id=pano, lat=lat, lng=lng)
    d.update({c: 0 for c in pu.TAGS})
    return d


def test_survivors_excludes_test_panos_and_10m_neighbours_only():
    lab = pd.DataFrame([
        _row("a:1", "test", "a", "P1", 47.0, -122.0),
        _row("a:2", "train", "a", "P1", 47.001, -122.0),       # on an HF test pano
        _row("a:3", "train", "a", "P2", 47.0, -122.00005),     # ~4 m from a:1, same city
        _row("b:3", "train", "b", "P3", 47.0, -122.00005),     # same spot, other city: kept
        _row("a:4", "train", "a", "P4", 47.01, -122.0),        # re-split test pano
        _row("a:5", "train", "a", "P5", 47.02, -122.0),        # far from everything: kept
    ])
    rs = pd.DataFrame({"label_uid": lab.label_uid,
                       "split": ["train", "train", "train", "train", "test", "train"]})
    surv, counts = pu.survivors(lab, rs)
    assert sorted(surv.label_uid) == ["a:5", "b:3"]
    assert counts["survivors"] == counts["survivors_trainable"] == 2
    assert counts["hf_train_on_test_pano"] == 2               # a:2 (HF test pano), a:4 (re-split)
    assert counts["hf_train_within_10m_of_test"] == 1         # a:3
    assert counts["hf_split_x_resplit"]["train/test"] == 1
