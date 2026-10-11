"""Tests for the tag trainer's infrastructure (scripts/analysis/tag_benchmark_86.py ``prep``,
``train --prep``, ``train --resume``; #86, the trainer prerequisites of the PU plan, PR #182
decision 9).

CPU only, synthetic crops, no network, no checkpoint. What must hold:

- the ``prep`` memmap holds exactly the bytes the in-memory decode holds, in the trainer's row
  order, and ``train --prep`` refuses an array prepared for a different table;
- the factored training loop is the old loop, statement for statement (same losses, same
  weights), and 1 epoch + resume + 1 epoch gives the same weights as 2 straight epochs. The real
  backbone needs a GPU and a 350 MB checkpoint, so a tiny stand-in model goes through the loop;
- only ``--resume`` writes ``checkpoint.pth``; a resume refuses changed settings (the fingerprint
  ``cmd_train`` builds covers tag values, ``affirmed``, pixels and backbone) and a smaller
  ``--epochs``, and repairs a ``best.pth`` a kill left behind;
- ``affirmed`` is never a tag, in any reader, and conflicting ``train`` flags are refused;
- the loss switch matches hand-computed values.
"""
import argparse
import hashlib
import math
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
    assert os.path.exists(tmp_path / "new" / "best.pth")
    # the default path (no --resume) writes nothing beyond what it wrote before the change
    assert sorted(os.listdir(tmp_path / "new")) == ["best.pth", "train_log.csv"]


def test_resume_gives_the_straight_run_bit_for_bit(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    fp = {"n_train": 10, "tags": ["a", "b", "c"]}
    straight = _model()
    rows_a, _ = tb.run_training(straight, X, Y, epochs=2, out_dir=str(tmp_path / "a"), fingerprint=fp, **KW)
    first = _model()
    tb.run_training(first, X, Y, epochs=1, out_dir=str(tmp_path / "b"), resume=True, fingerprint=fp, **KW)
    assert os.path.exists(tmp_path / "b" / "checkpoint.pth")
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


def test_resume_refuses_fewer_epochs_than_the_checkpoint_has(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    tb.run_training(_model(), X, Y, epochs=3, out_dir=str(tmp_path), resume=True, fingerprint={}, **KW)
    with pytest.raises(SystemExit, match="3 epochs already finished but --epochs is 1"):
        tb.run_training(_model(), X, Y, epochs=1, out_dir=str(tmp_path), resume=True, fingerprint={}, **KW)


def test_resume_repairs_a_best_pth_left_behind_by_a_kill(tmp_path):
    """A kill between the checkpoint's rename and best.pth's rename: the checkpoint says epoch 0
    is the best, best.pth is missing (or older) and best.pth.pending is lying there. The resume
    re-saves best.pth from the checkpoint and ends where a straight run ends."""
    pytest.importorskip("sklearn")
    X, Y = _data()
    straight = _model()
    tb.run_training(straight, X, Y, epochs=2, out_dir=str(tmp_path / "a"), **KW)
    b = tmp_path / "b"
    tb.run_training(_model(), X, Y, epochs=1, out_dir=str(b), resume=True, fingerprint={}, **KW)
    ck = torch.load(b / "checkpoint.pth")
    assert ck["epoch"] == ck["best_epoch"] == 0
    os.replace(b / "best.pth", b / "best.pth.pending")          # the rename that never happened
    stats = {}
    requeued = _Tiny(3)
    tb.run_training(requeued, X, Y, epochs=2, out_dir=str(b), resume=True, fingerprint={}, stats=stats, **KW)
    assert stats["best_restored_from"] is None                   # there was no best.pth at all
    assert not os.path.exists(b / "best.pth.pending") and _same_weights(straight, requeued)
    ba, bb = torch.load(tmp_path / "a" / "best.pth"), torch.load(b / "best.pth")
    assert ba["epoch"] == bb["epoch"] and ba["loss"] == bb["loss"]
    assert all(torch.equal(ba["model_state_dict"][k], bb["model_state_dict"][k]) for k in ba["model_state_dict"])
    assert stats["checkpoint_write_s"] > 0


def test_restore_best_rule(tmp_path):
    """_restore_best: agree -> nothing; checkpoint's own epoch is the best -> re-save from it;
    anything else -> refuse."""
    m = _model()
    ck = {"epoch": 3, "best_epoch": 3, "best_loss": 0.5, "model_state_dict": m.state_dict()}
    torch.save({"epoch": 1, "model_state_dict": {}, "loss": 0.9}, tmp_path / "best.pth")
    assert tb._restore_best(str(tmp_path), ck) == 1
    assert torch.load(tmp_path / "best.pth")["epoch"] == 3
    assert tb._restore_best(str(tmp_path), ck) is None           # now consistent
    torch.save({"epoch": 2, "model_state_dict": {}, "loss": 0.9}, tmp_path / "best.pth")
    with pytest.raises(SystemExit, match="refusing to resume"):
        tb._restore_best(str(tmp_path), dict(ck, best_epoch=1))


# --------------------------------------------------------------------------- #
# 2b. cmd_train's own wiring (CUDA steps stubbed), the fingerprint and the flags
# --------------------------------------------------------------------------- #
class _NoMove(_Tiny):
    def to(self, *a, **k):                                        # stand-in for .to(cuda)
        return self


def _run_cmd_train(monkeypatch, tmp_path, lab, d, extra, out="o"):
    """Run the real cmd_train through main() with only the CUDA-bound calls stubbed; returns the
    kwargs it passed to run_training and the train_meta.json it wrote."""
    seen = {}

    def fake_run_training(model, X, Y, **kw):
        seen.update(kw, X=X, Y=Y)
        os.makedirs(kw["out_dir"], exist_ok=True)
        if kw["stats"] is not None:
            kw["stats"].update(checkpoint_write_s=0.0, best_restored_from=None)
        return [{"epoch": 0, "loss": 0.5, "train_exact_match_acc": 0.5, "epoch_s": 1.0, "saved": "best"}], None
    monkeypatch.setattr(tb, "run_training", fake_run_training)
    monkeypatch.setattr(tb, "build_model", lambda repo, nc, backbone=None: _NoMove(nc))
    monkeypatch.setattr(tb, "check_tagger", lambda repo, sha: "tagger-sha-under-test")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "stub")
    bb = tmp_path / "backbone.pth"
    if not bb.exists():
        bb.write_bytes(b"backbone v1")
    tb.main(["train", "--tagger-repo", "x", "--labels", lab, "--backbone", str(bb),
             "--out-dir", str(tmp_path / out), "--epochs", "1"] + extra)
    meta = json.load(open(tmp_path / out / "train_meta.json", encoding="utf-8"))
    return seen, meta


def test_cmd_train_fingerprint_covers_tag_values_affirmed_pixels_and_backbone(tmp_path, monkeypatch):
    lab, d = _crops(tmp_path)
    fp = lambda extra, lab=lab, out="o": _run_cmd_train(monkeypatch, tmp_path, lab, d, extra, out)[0]["fingerprint"]
    base = fp(["--images", d, "--resume"])
    assert base["tagger_sha"] == "tagger-sha-under-test" and base["affirmed_sha256"] is None
    assert base["backbone_sha256"] == tb.sha256_file(str(tmp_path / "backbone.pth"))

    def changed(other):
        return sorted(k for k in base if base[k] != other[k])
    # one tag value flipped on a training row: same row ids, different targets
    df = pd.read_csv(lab)
    df.loc[0, "t-a"] = 1 - df.loc[0, "t-a"]
    flipped = str(tmp_path / "flipped.csv")
    df.to_csv(flipped, index=False)
    assert changed(fp(["--images", d, "--resume"], lab=flipped)) == ["targets_sha256"]
    # an affirmed column, then one affirmed value flipped
    df = pd.read_csv(lab)
    df["affirmed"] = 0
    a0 = str(tmp_path / "aff0.csv")
    df.to_csv(a0, index=False)
    df.loc[0, "affirmed"] = 1
    a1 = str(tmp_path / "aff1.csv")
    df.to_csv(a1, index=False)
    f0, f1 = fp(["--images", d, "--resume"], lab=a0), fp(["--images", d, "--resume"], lab=a1)
    assert f0["tags"] == base["tags"] == ["t-a", "t-b"]            # affirmed is not a tag
    assert f0["affirmed_sha256"] != f1["affirmed_sha256"] and f0["targets_sha256"] == f1["targets_sha256"]
    # the same pixels through --prep give the same fingerprint; other pixels do not
    out = str(tmp_path / "prep" / "train.npy")
    _prep(lab, d, out)
    assert changed(fp(["--prep", out, "--resume"])) == []
    meta_path = out + ".meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["array_sha256"] = "f" * 64
    json.dump(meta, open(meta_path, "w", encoding="utf-8"))
    assert changed(fp(["--prep", out, "--resume"])) == ["pixels_sha256"]
    # another backbone file
    (tmp_path / "backbone.pth").write_bytes(b"backbone v2")
    assert changed(fp(["--images", d, "--resume"])) == ["backbone_sha256"]


def test_cmd_train_default_path_builds_no_fingerprint_and_new_meta_fields_only_with_resume(tmp_path, monkeypatch):
    lab, d = _crops(tmp_path)
    seen, meta = _run_cmd_train(monkeypatch, tmp_path, lab, d, ["--images", d])
    assert seen["resume"] is False and seen["fingerprint"] is None and seen["loss_fn"] is None
    assert not {"fingerprint", "epoch_s_sum_all_runs", "checkpoint_write_s", "best_pth_rule"} & set(meta)
    seen, meta = _run_cmd_train(monkeypatch, tmp_path, lab, d, ["--images", d, "--resume"], out="r")
    assert {"epoch_s_sum_all_runs", "checkpoint_write_s", "timing_scope", "resumed_from_epoch"} <= set(meta)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"t-a": 0.4, "t-b": 0.1}))
    seen, meta = _run_cmd_train(monkeypatch, tmp_path, lab, d, ["--images", d, "--loss", "nnpu",
                                                                "--prior", str(prior)], out="pu")
    assert meta["best_pth_rule"] == tb.PLACEHOLDER_BEST_RULE and meta["prior_sha256"] == tb.sha256_file(str(prior))


@pytest.mark.parametrize("extra, msg", [
    (["--images", "a", "--prep", "p.npy"], "--prep and --images both given"),
    ([], "needs --images"),
    (["--images", "a", "--verify-prep"], "--verify-prep needs --prep"),
    (["--images", "a", "--prior", "p.json"], "--prior is not used by --loss bce"),
    (["--images", "a", "--loss", "soft", "--prior", "p.json", "--nnpu-beta", "0.1"], "only used by --loss nnpu"),
])
def test_train_refuses_conflicting_flags(extra, msg):
    ap_args = ["train", "--tagger-repo", "x", "--backbone", "bb", "--out-dir", "o"] + extra
    import unittest.mock as um
    with um.patch.object(tb, "cmd_train") as ct:
        tb.main(ap_args)
    with pytest.raises(SystemExit, match=msg):
        tb.check_train_flags(ct.call_args[0][0])


def test_prep_rerun_never_pairs_an_old_sidecar_with_a_new_array(tmp_path, monkeypatch):
    lab, d = _crops(tmp_path)
    out = str(tmp_path / "train.npy")
    _prep(lab, d, out)
    assert os.path.exists(out + ".meta.json")
    calls = []
    real = tb.decode_crop

    def dies_on_row_3(path):
        calls.append(path)
        if len(calls) == 3:
            raise RuntimeError("killed")
        return real(path)
    monkeypatch.setattr(tb, "decode_crop", dies_on_row_3)
    with pytest.raises(RuntimeError, match="killed"):
        _prep(lab, d, out)
    assert not os.path.exists(out + ".meta.json")                  # the old sidecar is gone
    tr, _ = tb.train_table(lab)
    with pytest.raises(FileNotFoundError):
        tb.load_prep(out, tr, lab)


# --------------------------------------------------------------------------- #
# 2c. one non-tag column list everywhere
# --------------------------------------------------------------------------- #
def test_affirmed_is_never_a_tag_in_any_reader(tmp_path):
    assert tb.LABEL_META_COLS is tb.NON_TAG_COLUMNS and "affirmed" in tb.NON_TAG_COLUMNS
    lab = tmp_path / "labels.csv"
    pd.DataFrame({"split": ["train", "test"], "filename": ["a.png", "b.png"], "label_uid": ["c:1", "c:2"],
                  "normalized_x": [0.5, 0.5], "normalized_y": [0.5, 0.5], "narrow": [1, 0], "steep": [0, 1],
                  "affirmed": [1, 0]}).to_csv(lab, index=False)
    df = pd.read_csv(lab)
    assert tb.tag_columns(df) == ["narrow", "steep"]                              # infer
    assert tb.load_labels(str(lab))[1] == ["narrow", "steep"]                     # score, test-only, context_fov_86
    assert tb.train_table(str(lab))[1] == ["narrow", "steep"]                     # train, prep


# --------------------------------------------------------------------------- #
# 3. the loss / mask switch
# --------------------------------------------------------------------------- #
def lp(z):
    """l(z, +1) = log(1 + e^-z), by hand."""
    return math.log1p(math.exp(-z))


def ln(z):
    """l(z, -1) = log(1 + e^z), by hand."""
    return math.log1p(math.exp(z))


def sig(z):
    return 1 / (1 + math.exp(-z))


# Table of N = 4 rows, T = 2 tags. tag 0: n_P = 2, n_U = 4, o = 0.5; tag 1: n_P = 1, n_U = 4, o = 0.25.
Y4 = np.array([[1, 0], [0, 1], [0, 0], [1, 0]], np.float32)
IDX = torch.tensor([0, 2])                    # batch b = 2: row 0 (tag 0 positive) and row 2 (untagged)
Z = [[0.3, -1.2], [-0.5, 0.8]]


def _call(loss, z=Z, idx=IDX):
    out = torch.tensor(z, dtype=torch.float32, requires_grad=True)
    obj, val, fired = loss(out, torch.from_numpy(Y4)[idx], idx)
    return out, obj, val, fired


def test_make_loss_is_none_for_plain_bce_and_that_path_is_the_old_loop(tmp_path):
    """--loss bce without --mask-csv must be the recipe's own path: make_loss returns None, and
    run_training with loss_fn=None gives the pre-change loop's per-step losses exactly."""
    pytest.importorskip("sklearn")
    tr = pd.DataFrame({"label_uid": ["c:0", "c:1"], "a": [0, 1]})
    assert tb.make_loss("bce", tr, ["a"]) is None
    X, Y = _data()
    m_old, m_new = _model(), _model()
    _, old_steps = _old_loop(m_old, X, Y, epochs=2, batch=4, lr=1e-2, seed=86, out_dir=str(tmp_path / "o"))
    steps = []
    orig = torch.nn.BCEWithLogitsLoss.forward

    def spy(self, a, b):
        v = orig(self, a, b)
        steps.append(v.item())
        return v
    torch.nn.BCEWithLogitsLoss.forward = spy
    try:
        tb.run_training(m_new, X, Y, epochs=2, out_dir=str(tmp_path / "n"), loss_fn=None, **KW)
    finally:
        torch.nn.BCEWithLogitsLoss.forward = orig
    assert len(steps) == len(old_steps) == 6
    assert steps == old_steps and _same_weights(m_old, m_new)


def test_masked_bce_by_hand_and_unmasked_equals_the_recipe():
    mask = np.ones((4, 2), np.float32)
    mask[2, 0] = 0                                             # row 2, tag 0 excluded
    loss = tb.TagLoss("bce", Y4, mask=mask)
    out, obj, val, fired = _call(loss)
    want = (lp(0.3) + ln(-1.2) + 0 + ln(0.8)) / (2 * 2)       # row 0: y=(1,0); row 2: (masked, 0)
    assert val.item() == pytest.approx(want, abs=1e-6) and obj is val and fired is None
    plain = tb.TagLoss("bce", Y4)
    out = torch.tensor(Z)
    ref = torch.nn.BCEWithLogitsLoss()(out, torch.from_numpy(Y4)[IDX])
    assert plain(out, torch.from_numpy(Y4)[IDX], IDX)[1].item() == pytest.approx(ref.item(), abs=1e-7)


def test_nnpu_by_hand_positive_unlabeled_and_clamp():
    # tag 0: prior 0.7 > o 0.5; tag 1: prior 0.1 < o 0.25 -> pi' = 0.25
    loss = tb.TagLoss("nnpu", Y4, prior=[0.7, 0.1])
    assert loss.pi_prime.tolist() == pytest.approx([0.7, 0.25])
    cP0, cU = 4 / (2 * 2), 4 / (2 * 4)                         # N / (b n_P), N / (b n_U)
    e_p_pos, e_p_neg = cP0 * lp(0.3), cP0 * ln(0.3)             # tag 0's batch positive: row 0
    e_u_neg0 = cU * (ln(0.3) + ln(-0.5))                        # U* = every PU cell, tagged or not
    br0 = e_u_neg0 - 0.7 * e_p_neg
    assert br0 > 0
    r0 = 1.0 * (0.7 * e_p_pos + br0)                            # weight n_U / N = 1
    r1 = 1.0 * (cU * (ln(-1.2) + ln(0.8)))                      # no tag-1 positive in the batch: E_P = 0
    out, obj, val, fired = _call(loss)
    assert val.item() == pytest.approx((r0 + r1) / 2, abs=1e-6)
    assert obj.item() == pytest.approx(val.item(), abs=1e-7) and not fired.any()
    # the censoring form (U = untagged only, prior pi_U = (pi - o) / (1 - o)) is the same risk
    n_uu, pi_u = 2, (0.7 - 0.5) / (1 - 0.5)
    cens = 0.7 * e_p_pos + (1 - 0.5) * max(0.0, 4 / (2 * n_uu) * ln(-0.5) - pi_u * e_p_neg)
    assert cens == pytest.approx(r0, abs=1e-12)

    # prior 0.9 on tag 0: the bracket goes negative -> clamped value, gradient ascent on the bracket
    loss = tb.TagLoss("nnpu", Y4, prior=[0.9, 0.1])
    br0 = e_u_neg0 - 0.9 * e_p_neg
    assert br0 < 0
    out, obj, val, fired = _call(loss)
    assert fired.tolist() == [True, False]
    assert val.item() == pytest.approx((0.9 * e_p_pos + r1) / 2, abs=1e-6)       # max(0, .) = 0
    assert obj.item() == pytest.approx((-1.0 * br0 + r1) / 2, abs=1e-6)          # -gamma * bracket
    obj.backward()
    # d/dz of -B_0 / 2 at row 0 (a positive): -(cU - 0.9 cP) sigmoid(0.3) / 2; row 2: -cU sigmoid(-0.5) / 2
    assert out.grad[0, 0].item() == pytest.approx(-(cU - 0.9 * cP0) * sig(0.3) / 2, abs=1e-6)
    assert out.grad[1, 0].item() == pytest.approx(-cU * sig(-0.5) / 2, abs=1e-6)
    # beta above |bracket|: no ascent, and the clamp holds the bracket's gradient at 0
    loss = tb.TagLoss("nnpu", Y4, prior=[0.9, 0.1], beta=1.0)
    out, obj, val, fired = _call(loss)
    assert not fired.any() and obj.item() == pytest.approx(val.item(), abs=1e-7)
    obj.backward()
    assert out.grad[1, 0].item() == 0.0
    assert out.grad[0, 0].item() == pytest.approx(-0.9 * cP0 * sig(-0.3) / 2, abs=1e-6)   # d l(z,+1) / dz


def test_nnpu_with_pi_prime_equal_o_is_naive_with_mask_and_affirmed_rows():
    """pi' = o (here: prior 0 everywhere, so pi' = max(0, o) = o) must give the naive masked BCE
    term for term, on a random table with masked cells and affirmed rows."""
    rng = np.random.default_rng(3)
    Y = (rng.random((40, 3)) < 0.3).astype(np.float32)
    mask = (rng.random((40, 3)) > 0.2).astype(np.float32)
    aff = (rng.random(40) < 0.25).astype(np.float32)
    naive = tb.TagLoss("bce", Y, mask=mask, affirmed=aff)
    pu = tb.TagLoss("nnpu", Y, mask=mask, affirmed=aff, prior=[0.0, 0.0, 0.0])
    assert np.allclose(pu.pi_prime, pu.o)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        idx = torch.randperm(40, generator=g)[:4]
        z = torch.randn((4, 3), generator=g)
        yb = torch.from_numpy(Y)[idx]
        a = naive(z, yb, idx)[1].item()
        obj, val, fired = pu(z, yb, idx)
        assert val.item() == pytest.approx(a, abs=1e-6) and obj.item() == pytest.approx(a, abs=1e-6)
        assert not fired.any()


@pytest.mark.parametrize("kind", ["bce", "nnpu", "soft"])
def test_a_masked_cell_contributes_nothing(kind):
    mask = np.ones((4, 2), np.float32)
    mask[2, 0] = 0
    loss = tb.TagLoss(kind, Y4, mask=mask, prior=[0.7, 0.1])
    out, obj, val, _ = _call(loss)
    obj.backward()
    assert out.grad[1, 0].item() == 0.0                         # row 2 is batch row 1
    z2 = [list(Z[0]), [5.0, Z[1][1]]]
    assert _call(loss, z=z2)[2].item() == pytest.approx(val.item(), abs=1e-7)


def test_soft_target_by_hand():
    loss = tb.TagLoss("soft", Y4, prior=[0.7, 0.1])
    assert loss.pi_U.tolist() == pytest.approx([0.4, 0.0])      # (0.7 - 0.5) / (1 - 0.5); tag 1 clamped
    out, obj, val, fired = _call(loss)

    def bce(z, t):
        return t * lp(z) + (1 - t) * ln(z)
    want = (bce(0.3, 1) + bce(-1.2, 0) + bce(-0.5, 0.4) + bce(0.8, 0)) / 4
    assert val.item() == pytest.approx(want, abs=1e-6) and fired is None
    # an affirmed row's untagged cell is a hard 0, and affirmed rows leave o (PU cells only)
    aff = np.array([0, 0, 1, 0], np.float32)
    loss = tb.TagLoss("soft", Y4, affirmed=aff, prior=[0.7, 0.1])
    assert loss.o.tolist() == pytest.approx([2 / 3, 1 / 3])
    want = (bce(0.3, 1) + bce(-1.2, 0) + bce(-0.5, 0) + bce(0.8, 0)) / 4
    assert _call(loss)[2].item() == pytest.approx(want, abs=1e-6)


def test_mask_and_prior_files(tmp_path):
    tr = pd.DataFrame({"label_uid": ["c:0", "c:1", "c:2"], "a": [1, 0, 0], "b": [0, 0, 1]})
    p = tmp_path / "m.csv"
    pd.DataFrame({"label_uid": ["c:2", "c:0", "c:1"], "a": [0, 1, 1]}).to_csv(p, index=False)
    m = tb.load_mask(str(p), tr, ["a", "b"])
    assert m.tolist() == [[1, 1], [1, 1], [0, 1]]                # joined on label_uid; b absent = unmasked
    pd.DataFrame({"label_uid": ["c:0", "c:1"], "a": [1, 1]}).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="no row"):
        tb.load_mask(str(p), tr, ["a", "b"])
    pd.DataFrame({"label_uid": ["c:0", "c:1", "c:2"], "typo": [1, 1, 1]}).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="typo"):
        tb.load_mask(str(p), tr, ["a", "b"])
    q = tmp_path / "prior.json"
    q.write_text(json.dumps({"a": 0.4}))
    with pytest.raises(SystemExit, match="missing"):
        tb.load_prior(str(q), ["a", "b"])
    q.write_text(json.dumps({"a": 0.4, "b": 0.1}))
    assert tb.load_prior(str(q), ["a", "b"]).tolist() == [0.4, 0.1]
    with pytest.raises(SystemExit, match="--prior"):
        tb.make_loss("nnpu", tr, ["a", "b"])


def test_nnpu_through_the_loop_logs_the_clamp_rate(tmp_path):
    pytest.importorskip("sklearn")
    X, Y = _data()
    loss = tb.TagLoss("nnpu", Y.numpy(), prior=[0.5, 0.5, 0.5])
    rows, _ = tb.run_training(_model(), X, Y, epochs=2, out_dir=str(tmp_path), loss_fn=loss,
                              tags=["a", "b", "c"], **KW)
    assert all(0.0 <= r[f"clamp:{t}"] <= 1.0 for r in rows for t in "abc")
    assert np.isfinite([r["loss"] for r in rows]).all()
