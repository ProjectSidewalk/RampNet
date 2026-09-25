"""Tests for the tag trainer's infrastructure (scripts/analysis/tag_benchmark_86.py ``prep``,
``train --prep``, ``train --resume``; #86, the trainer prerequisites of the PU plan, PR #182
decision 9).

CPU only, synthetic crops, no network, no checkpoint. What must hold:

- the ``prep`` memmap holds exactly the bytes the in-memory decode holds, in the trainer's row
  order, and ``train --prep`` refuses an array prepared for a different table;
- the factored training loop is the old loop, statement for statement (same losses, same
  weights), and 1 epoch + resume + 1 epoch gives the same weights as 2 straight epochs. The real
  backbone needs a GPU and a 350 MB checkpoint, so a tiny stand-in model goes through the loop;
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import tag_benchmark_86 as tb  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _crops(tmp_path, n=7, tags=("t-a", "t-b")):
    """n synthetic crops of assorted sizes and formats + a labels table; returns (labels, dir)."""
    from PIL import Image
    rng = np.random.default_rng(0)
    d = tmp_path / "crops"
    d.mkdir()
    rows = []
    for i in range(n):
        w, h = [(640, 640), (300, 200), (97, 511)][i % 3]
        fn = f"gsv-city-{i}-CurbRamp." + ("png" if i % 2 else "jpg")
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(d / fn)
        rows.append({"split": "test" if i == 3 else "train", "filename": fn, "city": "city",
                     "label_id": i, "label_uid": f"city:{i}",
                     **{t: int(rng.random() < 0.4) for t in tags}})
    lab = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(lab, index=False)
    return str(lab), str(d)


def _old_decode(path):
    """cmd_train's in-memory decode before the prep change, copied line for line."""
    from torchvision import io as tvio, transforms
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((tb.IMAGE_DIMENSION, tb.IMAGE_DIMENSION))])
    img = tf(tvio.read_image(path, mode=tvio.ImageReadMode.RGB))
    pw = (tb.PATCH_MULTIPLE - img.width % tb.PATCH_MULTIPLE) % tb.PATCH_MULTIPLE
    ph = (tb.PATCH_MULTIPLE - img.height % tb.PATCH_MULTIPLE) % tb.PATCH_MULTIPLE
    img = transforms.Pad((pw // 2, ph // 2, pw - pw // 2, ph - ph // 2))(img)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def _prep(lab, d, out, workers=1):
    tb.cmd_prep(argparse.Namespace(labels=lab, split_csv=None, images=[d], workers=workers, out=out))


# --------------------------------------------------------------------------- #
# 1. prep
# --------------------------------------------------------------------------- #
def test_decode_crop_is_the_old_in_memory_decode(tmp_path):
    lab, d = _crops(tmp_path)
    for fn in pd.read_csv(lab).filename:
        new = torch.from_numpy(tb.decode_crop(os.path.join(d, fn)))
        old = _old_decode(os.path.join(d, fn))
        assert new.dtype == torch.uint8 and new.shape == (3, tb.PREP_SIDE, tb.PREP_SIDE) == old.shape
        assert torch.equal(new, old)


@pytest.mark.parametrize("workers", [1, 2])
def test_prep_memmap_equals_the_in_memory_array(tmp_path, workers):
    lab, d = _crops(tmp_path)
    out = str(tmp_path / "prep" / "train.npy")
    _prep(lab, d, out, workers=workers)
    tr, tags = tb.train_table(lab)
    X_mem = torch.stack([_old_decode(os.path.join(d, fn)) for fn in tr.filename])
    X, meta = tb.load_prep(out, tr, lab, verify=True)
    assert isinstance(X, np.memmap) and not X.flags.writeable
    assert np.array_equal(X, X_mem.numpy())
    assert meta["label_uids"] == tr.label_uid.tolist() and "city:3" not in meta["label_uids"]  # train rows only
    assert meta["array_sha256"] == hashlib.sha256(X_mem.numpy().tobytes()).hexdigest()
    assert meta["labels_sha256"] == tb.sha256_file(lab) and meta["tags"] == tags
    assert not os.path.exists(out + ".partial")
    # a batch read from the memmap is the batch read from the tensor, in idx order
    idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(1))[:4]
    assert torch.equal(tb.batch_pixels(X, idx), tb.batch_pixels(X_mem, idx))


def test_load_prep_refuses_an_array_for_another_table(tmp_path):
    lab, d = _crops(tmp_path)
    out = str(tmp_path / "train.npy")
    _prep(lab, d, out)
    tr, _ = tb.train_table(lab)
    with pytest.raises(SystemExit, match="row order"):
        tb.load_prep(out, tr.iloc[::-1].reset_index(drop=True), lab)
    df = pd.read_csv(lab)
    df.loc[0, "t-a"] = 1 - df.loc[0, "t-a"]
    lab2 = str(tmp_path / "labels2.csv")
    df.to_csv(lab2, index=False)
    with pytest.raises(SystemExit, match="sha256"):
        tb.load_prep(out, tr, lab2)
    meta_path = out + ".meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["array_sha256"] = "0" * 64
    json.dump(meta, open(meta_path, "w", encoding="utf-8"))
    tb.load_prep(out, tr, lab)                                   # not re-hashed by default
    with pytest.raises(SystemExit, match="array sha256"):
        tb.load_prep(out, tr, lab, verify=True)


def test_train_cli_keeps_images_as_the_default_path():
    """The committed runbooks pass --images and no --prep; that must still parse."""
    import unittest.mock as um
    with um.patch.object(tb, "cmd_train") as ct:
        tb.main(["train", "--tagger-repo", "x", "--images", "a", "b", "--backbone", "bb", "--out-dir", "o"])
    args = ct.call_args[0][0]
    assert args.images == ["a", "b"] and args.prep is None and not args.verify_prep


# --------------------------------------------------------------------------- #
# 2. the factored loop and resume
# --------------------------------------------------------------------------- #
class _Tiny(torch.nn.Module):
    """A stand-in for the DINOv2 classifier: same input (normalised N x 3 x H x W), T logits."""
    def __init__(self, nc):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                        torch.nn.Flatten(), torch.nn.Linear(4, nc))

    def forward(self, x):
        return self.body(x)


def _data(n=10, t=3, side=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 256, (n, 3, side, side), generator=g, dtype=torch.uint8)
    Y = (torch.rand((n, t), generator=g) < 0.4).float()
    return X, Y


def _model(t=3, seed=86):
    torch.manual_seed(seed)
    return _Tiny(t)


def _old_loop(model, X, Y, epochs, batch, lr, seed, out_dir):
    """cmd_train's loop before it was factored into run_training, copied statement for statement
    (dev = cpu; X[idx] on the in-memory tensor). Returns the per-step loss values."""
    from torch import nn, optim
    from sklearn.metrics import accuracy_score
    dev = torch.device("cpu")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    os.makedirs(out_dir, exist_ok=True)
    log_rows, best_acc, best_loss = [], 0.0, 100.0
    g = torch.Generator().manual_seed(seed)
    steps = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        losses, accs = [], []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = (X[idx].to(dev).float() / 255.0 - mean) / std
            yb = Y[idx].to(dev)
            opt.zero_grad()
            out = model(xb).squeeze(dim=1)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pred = (torch.sigmoid(out) > 0.5).float()
            accs.append(accuracy_score(yb.cpu().numpy(), pred.detach().cpu().numpy()))
        steps += losses
        el, ea = float(np.mean(losses)), float(np.mean(accs))
        if ea > best_acc or (ea == best_acc and el < best_loss):
            best_acc, best_loss = ea, el
        log_rows.append({"epoch": epoch, "loss": el, "train_exact_match_acc": ea})
    return log_rows, steps


KW = dict(batch=4, lr=1e-2, seed=86, device=torch.device("cpu"))


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _strip(rows):
    return [{k: v for k, v in r.items() if k != "epoch_s"} for r in rows]


def test_factored_loop_is_the_old_loop(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    m_old, m_new = _model(), _model()
    old_rows, _ = _old_loop(m_old, X, Y, epochs=3, batch=4, lr=1e-2, seed=86, out_dir=str(tmp_path / "old"))
    new_rows, resumed = tb.run_training(m_new, X, Y, epochs=3, out_dir=str(tmp_path / "new"), **KW)
    assert resumed is None
    assert _same_weights(m_old, m_new)
    for o, n in zip(old_rows, new_rows):
        assert o["loss"] == n["loss"] and o["train_exact_match_acc"] == n["train_exact_match_acc"]
    assert os.path.exists(tmp_path / "new" / "best.pth") and os.path.exists(tmp_path / "new" / "checkpoint.pth")


def test_resume_gives_the_straight_run_bit_for_bit(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    fp = {"n_train": 10, "tags": ["a", "b", "c"]}
    straight = _model()
    rows_a, _ = tb.run_training(straight, X, Y, epochs=2, out_dir=str(tmp_path / "a"), fingerprint=fp, **KW)
    first = _model()
    tb.run_training(first, X, Y, epochs=1, out_dir=str(tmp_path / "b"), fingerprint=fp, **KW)
    # a new process: different init and a disturbed global RNG, so all state must come from the checkpoint
    torch.manual_seed(12345)
    requeued = _Tiny(3)
    rows_b, resumed = tb.run_training(requeued, X, Y, epochs=2, out_dir=str(tmp_path / "b"), resume=True,
                                      fingerprint=fp, **KW)
    assert resumed == 1
    assert _same_weights(straight, requeued)
    assert _strip(rows_a) == _strip(rows_b)
    la = pd.read_csv(tmp_path / "a" / "train_log.csv").drop(columns="epoch_s")
    lb = pd.read_csv(tmp_path / "b" / "train_log.csv").drop(columns="epoch_s")
    pd.testing.assert_frame_equal(la, lb)
    ca = torch.load(tmp_path / "a" / "best.pth")
    cb = torch.load(tmp_path / "b" / "best.pth")
    assert ca["epoch"] == cb["epoch"]
    # resuming a finished run trains nothing and changes nothing
    again = _Tiny(3)
    rows_c, resumed = tb.run_training(again, X, Y, epochs=2, out_dir=str(tmp_path / "b"), resume=True,
                                      fingerprint=fp, **KW)
    assert resumed == 2 and _same_weights(straight, again) and _strip(rows_c) == _strip(rows_a)


def test_resume_without_a_checkpoint_starts_fresh_and_refuses_other_settings(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    fresh, plain = _model(), _model()
    _, resumed = tb.run_training(fresh, X, Y, epochs=1, out_dir=str(tmp_path / "r"), resume=True,
                                 fingerprint={"lr": 0.01}, **KW)
    tb.run_training(plain, X, Y, epochs=1, out_dir=str(tmp_path / "p"), fingerprint={"lr": 0.01}, **KW)
    assert resumed is None and _same_weights(fresh, plain)
    with pytest.raises(SystemExit, match="lr"):
        tb.run_training(_model(), X, Y, epochs=2, out_dir=str(tmp_path / "r"), resume=True,
                        fingerprint={"lr": 0.02}, **KW)
