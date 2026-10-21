"""Tests for the #258 verifier candidate set (``verifier_candidates_258.py``).

These read the committed low-floor caches (``analysis_out/op_cache/``) and
``benchmark/*/records.jsonl`` / ``verdicts.json``; no GPU, no network, no imagery.

The load-bearing guarantees:

* per-peak labels sum to ``score_pano``'s counts at every threshold, not only the three
  the script asserts at, because greedy matching in score order is prefix-stable;
* a second hit on an already-claimed ramp is an ``fp`` marked ``redundant``;
* ``--check`` re-derives the committed files byte for byte.
"""
import contextlib
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import verifier_candidates_258 as vc  # noqa: E402
from operating_point_curve import read_cache  # noqa: E402
from rampnet.detection_eval import GroundTruth, radius_sq_for, score_pano  # noqa: E402


def test_labels_match_score_pano_at_every_threshold():
    panos, meta = read_cache(os.path.join(vc.CACHE_DIR, "juneau.json"))
    radius_sq = radius_sq_for(meta["radius_normalized"])
    for t in (0.05, 0.12, 0.25, 0.30, 0.41, 0.55, 0.8):
        tp = fp = ign = 0
        exp = [0, 0, 0]
        for pd in panos:
            rows = vc.label_peaks(pd["preds"], pd["gt"], radius_sq)
            kept = [r for r in rows if r[2] >= t]
            tp += sum(r[3] == "tp" for r in kept)
            fp += sum(r[3] == "fp" for r in kept)
            ign += sum(r[3] == "ignored" for r in kept)
            ps = score_pano([p for p in pd["preds"] if p[2] >= t], pd["gt"], radius_sq)
            exp[0] += ps.tp
            exp[1] += ps.fp
            exp[2] += ps.ignored
        assert (tp, fp, ign) == tuple(exp), t


def test_second_hit_on_a_claimed_ramp_is_a_redundant_fp():
    gt = GroundTruth([(0.5, 0.5)], [], True)
    rows = vc.label_peaks([(0.5, 0.5, 0.9), (0.501, 0.5, 0.4), (0.1, 0.5, 0.3)], gt,
                          radius_sq_for())
    assert [(r[3], r[4]) for r in rows] == [("tp", False), ("fp", True), ("fp", False)]


def test_unmatched_peak_near_an_ignore_point_is_ignored():
    gt = GroundTruth([], [(0.3, 0.5)], True)
    rows = vc.label_peaks([(0.3, 0.5, 0.7)], gt, radius_sq_for())
    assert rows[0][3] == "ignored"


def test_check_reproduces_committed_files():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = vc.main(["--check"])
    assert rc == 0, buf.getvalue()
