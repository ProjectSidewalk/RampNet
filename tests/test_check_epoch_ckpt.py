"""The epoch-label guard for renamed Ultralytics checkpoints (#51, #135).

Pins the one fact the guard exists for: ``epochN.pt`` is ``results.csv`` row N+1, so a
file renamed to ``_ep44`` must hold ``ckpt["epoch"] == 43``. The 2026-09-15 scoring
copied ``epoch44.pt`` as ``_ep44`` and scored one epoch late; this test builds that
exact mismatch and asserts the guard rejects it.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model_comparison", "yolo_baseline"))

import check_epoch_ckpt as g  # noqa: E402


def _ckpt(stored_epoch, n_rows):
    return {"epoch": stored_epoch,
            "train_results": {"epoch": list(range(1, n_rows + 1)),
                              "metrics/mAP50-95(B)": [0.4] * n_rows}}


def test_claimed_epoch_parses_the_stem_only():
    assert g.claimed_epoch("seedvar_ckpts/y11x_tiles_s1_ep44.pt") == 44
    assert g.claimed_epoch("/x/y11x_tiles_s3_ep42.pt") == 42
    assert g.claimed_epoch("y11x_tiles_s1_best.pt") is None
    assert g.claimed_epoch("ep44_something.pt") is None


def test_epoch43_pt_is_results_csv_row_44():
    ok, _ = g.check_epoch(_ckpt(43, 44), 44)
    assert ok


def test_the_2026_09_15_mismatch_is_rejected():
    # epoch44.pt copied as _ep44: stored epoch 44, 45 rows -- one epoch late.
    ok, detail = g.check_epoch(_ckpt(44, 45), 44)
    assert not ok
    assert "ckpt.epoch=44" in detail and "rows=45" in detail


def test_row_count_alone_is_not_enough():
    ok, _ = g.check_epoch(_ckpt(43, 45), 44)
    assert not ok


def test_best_pt_shape_is_rejected_if_a_label_is_claimed():
    ok, _ = g.check_epoch({"epoch": -1, "train_results": {"epoch": list(range(1, 61))}}, 44)
    assert not ok


def test_cli_exit_status(tmp_path):
    good = tmp_path / "y11x_tiles_s1_ep44.pt"
    bad = tmp_path / "y11x_tiles_s2_ep44.pt"
    best = tmp_path / "y11x_tiles_s1_best.pt"
    torch.save(_ckpt(43, 44), good)
    torch.save(_ckpt(44, 45), bad)
    torch.save({"epoch": -1}, best)
    assert g.main([str(good), str(best)]) == 0
    assert g.main([str(good), str(bad)]) == 1
