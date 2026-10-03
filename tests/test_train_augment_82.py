"""Stage 2 augmentation flags (#82): off means off, and arms stay paired by seed.

Three properties are load-bearing, and each fails silently on the cluster:

1. **With no ``--aug`` flag the dataset output is bit-identical to the published
   recipe's.** ``GOLDEN_OFF`` below is the sha256 of the lifted ``EquiHeatmapDataset``'s
   output on synthetic PNGs, computed from ``stage_two/train.py`` *before* the
   augmentation hook was added (origin/main 459ea9e). If a later edit changes what the
   dataset returns with augmentation off -- even by one pixel -- this hash moves.
2. **Augmentation draws never touch the global ``random`` stream.** The horizontal flip
   uses ``random.random()``; if an augmentation consumed the same stream, an
   augmented arm would see a different flip sequence from its same-seed control and the
   arms would no longer be paired.
3. **An augmentation draw is a pure function of (aug seed, epoch, sample index)**, so a
   job that resumes from ``latest_checkpoint.pth`` re-draws exactly what an
   uninterrupted run would have drawn for the same sample.

CPU only, synthetic images, no network, no checkpoint.
"""
import hashlib
import io
import json
import math
import os
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from torchvision import transforms  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402

from rampnet import augment as A  # noqa: E402

#: sha256 over (image bytes, heatmap bytes) for two passes over the synthetic set with
#: random.seed(42), computed with the ORIGINAL EquiHeatmapDataset (origin/main 459ea9e,
#: before #82 touched train.py). Do not regenerate this to make a failure go away: a
#: changed hash means the published recipe's data path changed.
GOLDEN_OFF = "3d3cbe4949b938374c0a00cfc068c3bf58c72430be57bf2931c85f9e451526ef"


def _lift():
    return load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)


def _make_split(root, n=6, size=(128, 64)):
    """Deterministic synthetic PNG panos (PNG, not JPEG: no decoder-version drift)."""
    split = os.path.join(root, "train")
    os.makedirs(split)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(split, f"p{i}.png"))
        pts = [[float(rng.random()), float(rng.random())] for _ in range(i % 3)]
        with open(os.path.join(split, f"p{i}.json"), "w") as f:
            json.dump({"curb_ramp_points_normalized": pts}, f)
    return root


def _digest(ds, passes=2, seed=42):
    random.seed(seed)
    h = hashlib.sha256()
    for _ in range(passes):
        for i in range(len(ds)):
            img, hm = ds[i]
            h.update(img.numpy().tobytes())
            h.update(hm.numpy().tobytes())
    return h.hexdigest()


def _dataset(mod, root, **kw):
    return mod.EquiHeatmapDataset(
        root_dir=root, split="train", target_heatmap_shape=(16, 32),
        transform_input=transforms.ToTensor(),
        points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
        apply_horizontal_flip=True, **kw)


def test_augmentation_off_is_bit_identical_to_the_published_dataset(tmp_path):
    mod = _lift()
    root = _make_split(str(tmp_path))
    assert _digest(_dataset(mod, root)) == GOLDEN_OFF


def test_flips_are_the_same_stream_with_augmentation_on(tmp_path):
    """Heatmaps depend only on the points and the flip draw, so equal heatmaps across the
    two arms mean equal flips; and the global `random` stream ends in the same state."""
    mod = _lift()
    root = _make_split(str(tmp_path))
    aug = A.Augmenter(A.parse_specs(["blur=1:0.5:2", "jpeg=1:30:60", "noise=1:2:4",
                                     "brightness=1:0.7:0.9"]), seed=3)
    out = {}
    for name, kw in (("off", {}), ("on", {"augment": aug})):
        ds = _dataset(mod, root, **kw)
        random.seed(7)
        imgs, hms = [], []
        for _ in range(3):
            for i in range(len(ds)):
                img, hm = ds[i]
                imgs.append(img.numpy().copy())
                hms.append(hm.numpy().copy())
        out[name] = (imgs, hms, random.random())
    assert all(np.array_equal(a, b) for a, b in zip(out["off"][1], out["on"][1]))
    assert out["off"][2] == out["on"][2]
    # ... and augmentation really did change the pixels
    assert not any(np.array_equal(a, b) for a, b in zip(out["off"][0], out["on"][0]))


def test_draws_are_a_pure_function_of_seed_epoch_index():
    aug = A.Augmenter(A.parse_specs(["downscale=0.5:0.4:0.8", "gamma=0.5:0.8:1.2"]), seed=1)
    first = [aug.params(i, 0)[0] for i in (5, 3, 9)]
    again = [aug.params(i, 0)[0] for i in (9, 3, 5)][::-1]
    assert first == again                      # order of access does not matter (resume)
    assert aug.params(5, 0)[0] != aug.params(5, 1)[0] or aug.params(3, 0)[0] != aug.params(3, 1)[0]
    other = A.Augmenter(aug.specs, seed=2)
    assert any(aug.params(i, 0)[0] != other.params(i, 0)[0] for i in range(10))


def test_augmentation_never_touches_the_global_streams():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    before = (np.random.random(), random.random(), torch.rand(1).item())
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    img = Image.fromarray(np.full((32, 64, 3), 120, np.uint8))
    aug = A.Augmenter(A.parse_specs([f"{op}=1:{lo}:{hi}" for op, lo, hi in (
        ("downscale", 0.5, 0.7), ("blur", 0.5, 1), ("brightness", 0.8, 1.2),
        ("contrast", 0.8, 1.2), ("saturation", 0.8, 1.2), ("gamma", 0.8, 1.2),
        ("wb", -0.2, 0.2), ("hue", -10, 10), ("noise", 1, 3), ("jpeg", 40, 80))]), seed=0)
    for i in range(5):
        aug(img, i, 0)
    after = (np.random.random(), random.random(), torch.rand(1).item())
    assert before == after


@pytest.mark.parametrize("op", sorted(A.OPS))
def test_every_op_keeps_size_and_mode_and_neutral_is_identity(op):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8))
    level = {"downscale": 0.5, "blur": 1.0, "jpeg": 50, "brightness": 1.2,
             "contrast": 0.8, "saturation": 0.5, "gamma": 1.3, "wb": 0.3, "hue": 20,
             "noise": 3.0, "unsharp": 100, "clahe": 0.02}[op]
    out = A.apply_op(img, op, level, rng=np.random.default_rng(1))
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    if A.NEUTRAL[op] is not None:
        same = A.apply_op(img, op, A.NEUTRAL[op], rng=np.random.default_rng(1))
        assert np.array_equal(np.asarray(same), np.asarray(img))


def test_parse_specs():
    s = A.parse_specs(["jpeg=1:40:80", "blur=0.5:0.3:1.5"])
    assert [x.op for x in s] == ["blur", "jpeg"]       # applied in TRAIN_ORDER
    assert str(s[0]) == "blur=0.5:0.3:1.5"
    for bad in (["blur=2:0:1"], ["blur=0.5:2:1"], ["sharpen=1:0:1"], ["blur=1"],
                ["blur=1:0:1", "blur=1:0:2"], ["clahe=1:0:0.1"]):
        with pytest.raises(ValueError):
            A.parse_specs(bad)
    assert A.parse_specs([]) == ()


def test_train_py_defaults_are_the_recipe():
    import argparse
    mod = load_from_train_py("parse_args", "PRESET_LR", "LR_SCHEDULES", argparse=argparse,
                             HISTORICAL_SEED=42, A=A)
    argv = sys.argv
    try:
        sys.argv = ["train.py"]
        a = mod.parse_args()
        assert a.aug_specs == () and a.grad_accum == 1 and a.max_steps is None
        sys.argv = ["train.py", "--aug", "blur=0.5:0.3:1.5", "--grad-accum", "4",
                    "--checkpoint-interval-steps", "400", "--max-steps", "2000"]
        a = mod.parse_args()
        assert a.aug_specs == (A.AugSpec("blur", 0.5, 0.3, 1.5),) and a.grad_accum == 4
        for bad in (["--grad-accum", "3"],                      # 1000 % 3 != 0
                    ["--aug", "blur=2:0:1"], ["--max-steps", "0"], ["--grad-accum", "0"]):
            sys.argv = ["train.py"] + bad
            with pytest.raises(SystemExit):
                mod.parse_args()
    finally:
        sys.argv = argv


def _module_call(name):
    import ast
    tree = ast.parse(open(os.path.join(REPO, "stage_two", "train.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name
                                                for t in node.targets):
            return node.value
    raise AssertionError(name)


def test_only_the_train_split_is_augmented_and_only_when_asked():
    import ast
    train = {k.arg: k.value for k in _module_call("train_dataset").keywords}
    val = {k.arg: k.value for k in _module_call("val_dataset").keywords}
    assert "augment" not in val
    assert ast.unparse(train["augment"]) == (
        "A.Augmenter(args.aug_specs, args.seed) if args.aug_specs else None")


def test_run_train_slurm_is_untouched():
    """stage_two/run_train.slurm is the preserved record of the published run."""
    with open(os.path.join(REPO, "stage_two", "run_train.slurm"), "rb") as f:
        assert "--aug" not in f.read().decode("utf-8")


def test_final_micro_batch_skips_the_periodic_save():
    """#82 review S3: at the last micro-batch of a --max-steps run the periodic save must not
    run, or a preemption before save_final leaves a finished resume file with no
    final_step_N.pth. The recipe (no --max-steps) keeps step % interval == 0 exactly."""
    mod = load_from_train_py("periodic_save_due")
    due = mod["periodic_save_due"] if isinstance(mod, dict) else mod.periodic_save_due
    # the launcher's defaults: 2,000 optimizer steps x accumulation 4, interval 100 x 4
    assert not due(8000, 400, 8000)
    assert due(7600, 400, 8000) and not due(7601, 400, 8000)
    assert not due(8400, 400, 8000)
    for step in range(1, 5001):
        assert due(step, 250, None) == (step % 250 == 0)


def test_the_loop_uses_periodic_save_due():
    src = open(os.path.join(REPO, "stage_two", "train.py"), encoding="utf-8").read()
    assert "if periodic_save_due(current_total_step, checkpoint_interval_steps, max_micro_steps):" in src
    assert "current_total_step % checkpoint_interval_steps == 0" not in src
