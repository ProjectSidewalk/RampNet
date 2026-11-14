"""Tests for recall_check.py fa-sheet (#285 A4): spot grouping and the seam-wrapping crop.

Pure logic on synthetic inputs; no imagery or predictions on disk.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274"))

import recall_check as rc  # noqa: E402


def test_spots_group_across_models_and_the_seam():
    dets = [("crosswalk_s6full", 0.995, 0.6, 0.9), ("crosswalk_s4_klone", 0.005, 0.6, 0.4),
            ("crosswalk_s6half", 0.5, 0.7, 0.3)]
    spots = rc._spots(dets)
    assert [len(s["dets"]) for s in spots] == [2, 1]
    assert spots[0]["x"] == 0.995          # a spot sits at its strongest detection


def test_family_keeps_the_step_control_apart_from_s4():
    assert rc._family("crosswalk_s4_steps1397") == "s4_steps1397"
    assert rc._family("crosswalk_s4_seed2") == rc._family("crosswalk_s4_klone") == "s4"


def test_crop_wraps_across_the_seam():
    Image = pytest.importorskip("PIL.Image")
    w, h = 1000, 500
    img = Image.new("RGB", (w, h), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 10, h))          # a red strip at the left edge only
    crop, (left, top, cw, ch) = rc._crop(img, 0.995, 0.5)
    assert left < w and left + cw > w              # the window crosses the seam
    px = crop.load()
    red = sum(1 for x in range(crop.size[0]) if px[x, crop.size[1] // 2][0] > 200)
    assert red > 0                                 # the part past the seam came from the left edge
