"""Tests for the tag trainer's infrastructure (scripts/analysis/tag_benchmark_86.py ``prep`` and
``train --prep``; #86, the trainer prerequisites of the PU plan, PR #182 decision 9).

CPU only, synthetic crops, no network, no checkpoint. What must hold:

- the ``prep`` memmap holds exactly the bytes the in-memory decode holds, in the trainer's row
  order, and ``train --prep`` refuses an array prepared for a different table;
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
