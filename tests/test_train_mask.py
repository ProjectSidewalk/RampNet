"""Stage 2 masked heatmap loss (#263): weight maps, the loss, and off-means-off.

What is load-bearing, and fails silently on the cluster if wrong:

1. **The weight map is built from the same point list as the target**, so a flipped
   pano gets a flipped disc, and the disc is clipped at the 0/360 seam exactly as the
   target is (no wrap). A disc that drifted from its target would train the model to
   predict a ramp where the loss cannot see it, and zero where it can.
2. **The loss is the published MSE when nothing is masked**, and ignores masked pixels
   completely when something is.
3. **Without ``--loss-mask`` nothing about the published path moves**: the dataset output
   still hashes to ``GOLDEN_OFF`` (the #82 constant, imported, not copied), and the
   training loop still calls ``nn.MSELoss`` on that path.

CPU only, synthetic images, no network, no checkpoint.
"""
import ast
import hashlib
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402
from test_train_augment_82 import GOLDEN_OFF, _make_split  # noqa: E402

from rampnet import augment as A  # noqa: E402

TRAIN_PY = os.path.join(REPO, "stage_two", "train.py")
SHAPE = (512, 1024)                      # the recipe's heatmap shape
RADIUS = 30                              # ceil(3 * sigma), sigma 10


def _lift():
    return load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points", "generate_weight_from_points",
        "supervision_mode", "masked_mse", "SUPERVISION_MODES",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)


def _digest(ds, passes=2, seed=42):
    """As test_train_augment_82._digest, but over the first two elements only, so a
    mask-on dataset can be hashed against the same constant."""
    random.seed(seed)
    h = hashlib.sha256()
    for _ in range(passes):
        for i in range(len(ds)):
            img, hm = ds[i][:2]
            h.update(img.numpy().tobytes())
            h.update(hm.numpy().tobytes())
    return h.hexdigest()


def _dataset(mod, root, shape=(16, 32), **kw):
    return mod.EquiHeatmapDataset(
        root_dir=root, split="train", target_heatmap_shape=shape,
        transform_input=transforms.ToTensor(),
        points_to_heatmap_transform_fn=mod.generate_heatmap_from_points, **kw)


def _one_pano(root, record, size=(64, 32)):
    """A single-pano train split whose JSON is ``record``."""
    split = os.path.join(root, "train")
    os.makedirs(split, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 90, np.uint8)).save(
        os.path.join(split, "p.png"))
    with open(os.path.join(split, "p.json"), "w") as f:
        json.dump(record, f)
    return root


def _sparse(points):
    return {"curb_ramp_points_normalized": points,
            "supervision": {"mode": "sparse", "sources": ["strong_site"] * len(points),
                            "site_ids": ["x:1"] * len(points), "run_sha": "0" * 64}}


def _disc(cx, cy, shape=SHAPE, r=RADIUS):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)


# ------------------------------------------------------------------ off means off

def test_mask_off_dataset_still_hashes_to_golden_off(tmp_path):
    mod = _lift()
    root = _make_split(str(tmp_path))
    assert _digest(_dataset(mod, root, apply_horizontal_flip=True)) == GOLDEN_OFF


def test_mask_on_changes_nothing_but_adds_a_weight_on_dense_data(tmp_path):
    """Dense panos under --loss-mask: identical image and target (same flip stream), and
    an all-ones weight -- which is what makes the masked loss reduce to MSE on them."""
    mod = _lift()
    root = _make_split(str(tmp_path))
    ds = _dataset(mod, root, apply_horizontal_flip=True, loss_mask=True)
    assert _digest(ds) == GOLDEN_OFF
    for i in range(len(ds)):
        img, hm, w = ds[i]
        assert w.shape == hm.shape and w.dtype == torch.float32
        assert torch.equal(w, torch.ones_like(hm))


def test_mask_off_dataset_returns_the_published_two_tuple(tmp_path):
    mod = _lift()
    ds = _dataset(mod, _make_split(str(tmp_path)))
    assert all(len(ds[i]) == 2 for i in range(len(ds)))
    assert len(ds[len(ds)]) == 2                       # the out-of-range dummy too


def test_mask_off_refuses_a_sparse_pano(tmp_path):
    """Without the flag a sparse pano would train as dense -- 'no ramp' everywhere off its
    few positives. That has to stop the run, not pass silently."""
    mod = _lift()
    ds = _dataset(mod, _one_pano(str(tmp_path), _sparse([[0.5, 0.5]])),
                  apply_horizontal_flip=False)
    with pytest.raises(ValueError, match="--loss-mask"):
        ds[0]


# ---------------------------------------------------------------- the weight maps

def test_sparse_weight_is_the_union_of_3_sigma_discs():
    mod = _lift()
    pts = [(0.25, 0.5), (0.7, 0.3)]
    w = mod.generate_weight_from_points(pts, heatmap_shape=SHAPE).numpy()[0]
    want = np.maximum(_disc(256, 256), _disc(717, 154))   # round(0.7*1024)=717, 0.3*512=153.6
    assert np.array_equal(w, want)
    assert w.sum() == 2 * _disc(256, 256).sum()           # disjoint, ~2,821 px each


def test_disc_is_exactly_where_the_target_is_at_least_exp_minus_4_5():
    """Same rounding, same clamp, same radius as the target: the disc is the target's
    level set at d = 3 sigma, not an approximation of it."""
    mod = _lift()
    for pt in ((0.31, 0.62), (0.5, 0.05), (0.0004, 0.7), (0.9996, 0.2)):
        hm = mod.generate_heatmap_from_points([pt], heatmap_shape=SHAPE).numpy()[0]
        w = mod.generate_weight_from_points([pt], heatmap_shape=SHAPE).numpy()[0]
        assert np.array_equal(w == 1, hm >= math.exp(-4.5) * (1 - 1e-6)), pt


@pytest.mark.parametrize("x, far_col", [(0.9996, 0), (0.0004, -1)])
def test_seam_is_clipped_not_wrapped_in_both_maps(x, far_col):
    mod = _lift()
    hm = mod.generate_heatmap_from_points([(x, 0.5)], heatmap_shape=SHAPE).numpy()[0]
    w = mod.generate_weight_from_points([(x, 0.5)], heatmap_shape=SHAPE).numpy()[0]
    assert hm[:, far_col].max() == 0 and w[:, far_col].max() == 0
    near_col = -1 if far_col == 0 else 0
    assert hm[:, near_col].max() == 1 and w[:, near_col].max() == 1
    # clipped: the disc keeps only the half on its own side of the seam (+ the centre column)
    assert w.sum() < 0.55 * _disc(512, 256).sum()


def test_all_three_modes_through_the_dataset(tmp_path):
    mod = _lift()
    cases = {
        "dense": {"curb_ramp_points_normalized": [[0.5, 0.5]]},
        "sparse": _sparse([[0.5, 0.5]]),
        "verified_empty": {"curb_ramp_points_normalized": [],
                           "supervision": {"mode": "verified_empty", "sources": [],
                                           "site_ids": [], "run_sha": "0" * 64}},
    }
    out = {}
    for name, rec in cases.items():
        ds = _dataset(mod, _one_pano(str(tmp_path / name), rec), shape=SHAPE,
                      apply_horizontal_flip=False, loss_mask=True)
        out[name] = ds[0]
    assert torch.equal(out["dense"][2], torch.ones(1, *SHAPE))
    assert np.array_equal(out["sparse"][2].numpy()[0], _disc(512, 256))
    assert torch.equal(out["sparse"][1], out["dense"][1])       # same target as dense
    assert torch.equal(out["verified_empty"][1], torch.zeros(1, *SHAPE))
    assert torch.equal(out["verified_empty"][2], torch.ones(1, *SHAPE))


def test_flip_moves_the_disc_with_the_target(tmp_path):
    mod = _lift()
    ds = _dataset(mod, _one_pano(str(tmp_path), _sparse([[0.2, 0.4]])), shape=SHAPE,
                  apply_horizontal_flip=True, loss_mask=True)
    seen = set()
    random.seed(0)
    for _ in range(12):
        img, hm, w = ds[0]
        hm, w = hm.numpy()[0], w.numpy()[0]
        cx = int(np.unravel_index(hm.argmax(), hm.shape)[1])
        seen.add(cx)
        assert np.array_equal(w == 1, hm >= math.exp(-4.5) * (1 - 1e-6))
        assert np.array_equal(w, _disc(cx, 205))
    assert seen == {205, 819}           # round(0.2*1024)=205; round(0.8*1024)=819


def test_unknown_and_inconsistent_modes_are_errors(tmp_path):
    mod = _lift()
    with pytest.raises(ValueError, match="mode"):
        mod.supervision_mode({"supervision": {"mode": "partial"}}, [])
    with pytest.raises(ValueError, match="mode"):
        mod.supervision_mode({"supervision": {}}, [])
    with pytest.raises(ValueError, match="verified_empty"):
        mod.supervision_mode({"supervision": {"mode": "verified_empty"}}, [(0.1, 0.2)])
    assert mod.SUPERVISION_MODES == ("sparse", "verified_empty")
    bad = _sparse([[0.5, 0.5]])
    bad["supervision"]["mode"] = "Sparse"
    ds = _dataset(mod, _one_pano(str(tmp_path), bad), loss_mask=True,
                  apply_horizontal_flip=False)
    with pytest.raises(ValueError):
        ds[0]


def test_unreadable_json_teaches_nothing_under_the_mask(tmp_path):
    mod = _lift()
    root = _one_pano(str(tmp_path), {})
    with open(os.path.join(root, "train", "p.json"), "w") as f:
        f.write("{not json")
    _, hm, w = _dataset(mod, root, loss_mask=True, apply_horizontal_flip=False)[0]
    assert w.sum() == 0 and hm.sum() == 0


def test_dummy_samples_carry_a_zero_weight(tmp_path):
    mod = _lift()
    ds = _dataset(mod, _make_split(str(tmp_path)), loss_mask=True)
    img, hm, w = ds[len(ds)]
    assert w.shape == hm.shape and w.sum() == 0


# ------------------------------------------------------------------------ the loss

def test_masked_loss_with_all_ones_is_mse():
    mod = _lift()
    g = torch.Generator().manual_seed(0)
    out = torch.rand(2, 1, 64, 128, generator=g).half()       # autocast emits fp16
    tgt = torch.rand(2, 1, 64, 128, generator=g)
    got = mod.masked_mse(out, tgt, torch.ones_like(tgt))
    want = torch.nn.MSELoss()(out.float(), tgt)
    assert got.dtype == torch.float32
    assert torch.allclose(got, want, rtol=1e-6, atol=0)


def test_masked_loss_ignores_masked_pixels():
    mod = _lift()
    g = torch.Generator().manual_seed(1)
    tgt = torch.rand(1, 1, 64, 128, generator=g)
    w = torch.zeros_like(tgt)
    w[..., 10:30, 40:90] = 1
    out = torch.rand(1, 1, 64, 128, generator=g, requires_grad=True)
    loss = mod.masked_mse(out, tgt, w)
    inside = ((out - tgt) ** 2)[..., 10:30, 40:90].mean()
    assert torch.allclose(loss, inside, rtol=1e-6)
    loss.backward()
    assert out.grad[w == 0].abs().max() == 0 and out.grad[w == 1].abs().max() > 0
    # garbage where w == 0 changes nothing
    out2 = out.detach().clone()
    out2[w == 0] = 1e6
    assert torch.equal(mod.masked_mse(out2, tgt, w), loss.detach())


def test_fully_masked_batch_is_zero_not_nan():
    mod = _lift()
    out = torch.rand(1, 1, 8, 16, requires_grad=True)
    loss = mod.masked_mse(out, torch.rand(1, 1, 8, 16), torch.zeros(1, 1, 8, 16))
    assert loss.item() == 0
    loss.backward()
    assert out.grad.abs().max() == 0


# ------------------------------------------------- the training loop, read as source

def _tree():
    with open(TRAIN_PY, encoding="utf-8") as f:
        return ast.parse(f.read())


def _is_loss_mask_test(node):
    return isinstance(node, ast.If) and ast.unparse(node.test) == "args.loss_mask"


def test_default_path_still_calls_nn_mseloss():
    """criterion is still nn.MSELoss(), every criterion(...) call sits in the else of an
    `if args.loss_mask`, and every masked_mse(...) call sits in its body."""
    tree = _tree()
    crit = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "criterion" for t in n.targets)]
    assert len(crit) == 1 and ast.unparse(crit[0].value) == "nn.MSELoss()"
    criterion_calls, masked_calls = [], []
    for node in ast.walk(tree):
        if _is_loss_mask_test(node):
            for branch, sink in ((node.orelse, criterion_calls), (node.body, masked_calls)):
                for stmt in branch:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                            sink.append(sub.func.id)
    all_calls = [n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert all_calls.count("criterion") == criterion_calls.count("criterion") == 2
    # one call per loop, plus none outside the guarded bodies
    assert all_calls.count("masked_mse") == masked_calls.count("masked_mse") == 2
    src = open(TRAIN_PY, encoding="utf-8").read()
    assert "loss = criterion(outputs, target_heatmaps)" in src
    assert "loss_val = criterion(outputs_val, target_heatmaps_val)" in src


def test_flag_defaults_off_and_both_datasets_follow_it():
    import argparse
    mod = load_from_train_py("parse_args", "PRESET_LR", "LR_SCHEDULES", argparse=argparse,
                             HISTORICAL_SEED=42, A=A)
    argv = sys.argv
    try:
        sys.argv = ["train.py"]
        assert mod.parse_args().loss_mask is False
        sys.argv = ["train.py", "--loss-mask"]
        assert mod.parse_args().loss_mask is True
    finally:
        sys.argv = argv
    for name in ("train_dataset", "val_dataset"):
        call = next(n.value for n in _tree().body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == name for t in n.targets))
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw["loss_mask"] == "args.loss_mask", name


def test_every_dataset_unpack_tolerates_the_third_element():
    src = open(TRAIN_PY, encoding="utf-8").read()
    assert "for i, (images, target_heatmaps, *weights) in enumerate(train_loader):" in src
    assert "for images_val, target_heatmaps_val, *weights_val in val_pbar:" in src
    assert "image_vis, gt_heatmap_vis, *_ = val_dataset[rand_idx]" in src
    assert "image_vis, gt_heatmap_vis = val_dataset" not in src


def test_no_existing_launcher_turns_the_mask_on():
    """The mask is opt-in per experiment. The preserved launchers and the #263 control arm
    must not pass it."""
    for name in ("run_train.slurm", "run_train_seed.slurm", "run_finetune_aug82.slurm",
                 "run_finetune_control.slurm"):
        with open(os.path.join(REPO, "stage_two", name), encoding="utf-8") as f:
            code = [ln for ln in f.read().splitlines()
                    if not ln.lstrip().startswith(("#", "echo "))]
        assert not any("--loss-mask" in ln for ln in code), name
