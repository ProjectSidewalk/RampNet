"""Tests for scripts/analysis/input_res_sweep_25.py (issue #25, frozen-model input sweep).

CPU only, no network, committed fixtures only: the model is built with
``pretrained_backbone=False`` and the report tests read the committed caches under
analysis_out/input_res_sweep_25/cache/.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import input_res_sweep_25 as irs  # noqa: E402


# (a) arm preprocessing -------------------------------------------------------------
def test_arm_sizes_and_resize_order():
    assert irs.resize_steps("r2048", (11000, 5500)) == [((2048, 4096), "bilinear")]
    assert irs.resize_steps("r3072", (8000, 4000)) == [((3072, 6144), "bilinear")]
    assert irs.resize_steps("r4096", (16384, 8192)) == [((4096, 8192), "bilinear")]
    # the upsample control: bilinear down to the model size FIRST, then bicubic up
    assert irs.resize_steps("u4096", (11000, 5500)) == [((2048, 4096), "bilinear"),
                                                        ((4096, 8192), "bicubic")]
    assert irs.resize_steps("r4096_hm1024", (8000, 4000)) == [((4096, 8192), "bilinear")]


def test_rnative_is_native_floored_and_capped():
    assert irs.arm_input_size("rnative", (11000, 5500)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (5660, 2830)) == (2830, 5660)
    assert irs.arm_input_size("rnative", (16384, 8192)) == (5500, 11000)   # the cap
    assert irs.arm_input_size("rnative", (13312, 6656)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (12288, 6144)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (16384, 8192), (6144, 12288)) == (6144, 12288)
    assert irs.arm_input_size("rnative", (3328, 1664)) == (2048, 4096)     # the floor
    assert irs.arm_input_size("rnative", (4096, 2048)) == (2048, 4096)
    assert irs.arm_input_size("rnative", (16384, 8192), (4096, 8192)) == (4096, 8192)


def test_r2048_is_the_committed_instrument():
    """r2048 must go through threshold_sweep.PRE itself, not a copy of it."""
    torch = pytest.importorskip("torch")
    from PIL import Image
    import threshold_sweep as ts
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (60, 120, 3), dtype=np.uint8))
    a = irs.arm_tensor(img, "r2048")
    assert tuple(a.shape) == (3, 2048, 4096)
    assert torch.equal(a, ts.PRE(img))


def test_u4096_is_bicubic_of_the_2048_derivative():
    torch = pytest.importorskip("torch")
    from PIL import Image
    from torchvision import transforms
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8))
    got = irs.arm_tensor(img, "u4096")
    mid = transforms.Resize((2048, 4096),
                            interpolation=transforms.InterpolationMode.BILINEAR)(img)
    up = transforms.Resize((4096, 8192),
                           interpolation=transforms.InterpolationMode.BICUBIC)(mid)
    want = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(transforms.ToTensor()(up))
    assert tuple(got.shape) == (3, 4096, 8192)
    assert torch.equal(got, want)


# (b) the sensitivity arm's head loads the released weights strictly -----------------
def test_hm1024_head_loads_512_state_dict_strictly():
    torch = pytest.importorskip("torch")
    from rampnet.model import KeypointModel
    base = KeypointModel(pretrained_backbone=False)
    big = KeypointModel(heatmap_size=(1024, 2048), pretrained_backbone=False)
    big.load_state_dict(base.state_dict(), strict=True)
    x = torch.zeros(1, 3, 64, 128)
    with torch.no_grad():
        assert tuple(big.eval()(x).shape) == (1, 1, 1024, 2048)
        assert tuple(base.eval()(x).shape) == (1, 1, 512, 1024)


# (c) headroom classes from the committed records ----------------------------------
def test_headroom_class_from_records():
    rich = irs.native_sizes_from_records("richmond")
    widths = [w for w, _ in rich.values()]
    cls = [irs.headroom_class(w, 8192) for w in widths]
    assert cls.count("none") == 20            # the 4096-wide built-in null
    assert cls.count("full") == 77 + 14       # 11000 and 12288 wide
    assert cls.count("partial") == 13         # 5760 wide
    ann = irs.native_sizes_from_records("annapolis")
    assert {irs.headroom_class(w, 8192) for w, _ in ann.values()} == {"partial"}
    assert {irs.headroom_class(w, 6144) for w, _ in ann.values()} == {"full"}
    morg = irs.native_sizes_from_records("morgantown")
    assert sum(irs.headroom_class(w, 8192) == "none" for w, _ in morg.values()) == 122


# (e) usage rows --------------------------------------------------------------------
def test_usage_rows_are_free_and_timed():
    rows = irs.usage_rows({"r4096": {"elapsed_s": 100.0, "panos_scored": 50, "fp16": False}},
                          decode_s=40.0, host="makelab2.cs.washington.edu",
                          gpus=["NVIDIA A40"], cities=["annapolis"],
                          started="2026-09-26T00:00:00Z")
    assert len(rows) == 2
    for r in rows:
        assert r["paid"] is False
        assert r["est_cost_usd"] == 0.0
        assert r["hardware"] == {"host": "makelab2.cs.washington.edu", "gpus": ["NVIDIA A40"]}
        assert r["elapsed_s"] > 0
        assert r["run_id"].startswith("input-res-sweep-25:")
    assert rows[0]["s_per_pano"] == 2.0
    assert len({r["run_id"] for r in rows}) == 2


def test_extract_refuses_unrecorded_spend():
    with pytest.raises(SystemExit):
        irs.main(["extract", "--usage-log", "none", "--cities", "richmond"])


# verdict rule ------------------------------------------------------------------------
def _d(p, r, f):
    return {"precision": {"observed": p[0], "ci_lo": p[1], "ci_hi": p[2]},
            "recall": {"observed": r[0], "ci_lo": r[1], "ci_hi": r[2]},
            "f1": {"observed": f[0], "ci_lo": f[1], "ci_hi": f[2]}}


def test_verdict_rule():
    helps = _d((-0.01, -0.02, 0.0), (0.05, 0.02, 0.08), (0.02, -0.01, 0.05))
    beats_up = _d((0, 0, 0), (0.03, 0.01, 0.05), (0, 0, 0))
    ties_up = _d((0, 0, 0), (0.01, -0.01, 0.03), (0, 0, 0))
    assert irs.verdict(helps, beats_up)[0] == "helps"
    assert irs.verdict(helps, ties_up)[0] == "tolerates"     # object scale, not pixels
    hurts = _d((-0.1, -0.15, -0.05), (0.0, -0.02, 0.02), (-0.05, -0.08, -0.02))
    assert irs.verdict(hurts, beats_up)[0] == "hurts"
    flat = _d((0.0, -0.01, 0.01), (0.0, -0.01, 0.01), (0.0, -0.01, 0.01))
    assert irs.verdict(flat, None)[0] == "tolerates"


def test_check_compares_peak_sets():
    ref = {"panos": [{"pano": "a", "preds": [[0.1, 0.6, 0.9], [0.5, 0.7, 0.4]]}]}
    same = [{"pano": "a", "preds": [(0.5, 0.7, 0.40000004), (0.1, 0.6, 0.9)]}]
    assert irs.compare_to_op_cache(same, ref)[0] == 0
    off = [{"pano": "a", "preds": [(0.5, 0.7, 0.41), (0.1, 0.6, 0.9)]}]
    assert irs.compare_to_op_cache(off, ref)[0] == 1
    fewer = [{"pano": "a", "preds": [(0.1, 0.6, 0.9)]}]
    assert irs.compare_to_op_cache(fewer, ref)[0] == 1
