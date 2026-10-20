"""scripts/mix_training_dirs.py (#263): a sparse set plus a seeded GSV replay sample, as links.

CPU only, no network. Needs symlinks; skipped where the OS will not create them (Windows
without developer mode). CI runs on Linux.
"""
import json
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import mix_training_dirs as mix  # noqa: E402


@pytest.fixture(autouse=True)
def _needs_symlinks(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this OS/user cannot create symlinks")
    (tmp_path / "probe").unlink()


def _pano(d, pid, sup=False):
    (d / f"{pid}.jpg").write_bytes(b"jpg" + pid.encode())
    rec = {"curb_ramp_points_normalized": []}
    if sup:
        rec["supervision"] = {"mode": "sparse"}
    (d / f"{pid}.json").write_text(json.dumps(rec), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    sparse, gsv = tmp_path / "sparse", tmp_path / "gsv"
    for d in (sparse, gsv / "train", gsv / "val"):
        d.mkdir(parents=True)
    for pid in ("S1", "S2", "S3"):
        _pano(sparse, pid, sup=True)
    (sparse / "labels_manifest.json").write_text(json.dumps(
        {"pano_ids": ["S1", "S2", "S3"], "labels_sha256": "ab" * 32, "city": "lyon"}),
        encoding="utf-8")
    for i in range(10):
        _pano(gsv / "train", f"G{i}")
    (gsv / "train" / "lonely.jpg").write_bytes(b"x")              # no json: not a candidate
    _pano(gsv / "val", "V1")
    return sparse, gsv


def _argv(inputs, out, *extra):
    sparse, gsv = inputs
    return ["--sparse", str(sparse), "--gsv-root", str(gsv), "--out", str(out), *extra]


def test_mix_has_every_sparse_pano_and_a_seeded_gsv_sample(inputs, tmp_path):
    out = tmp_path / "mix"
    assert mix.main(_argv(inputs, out, "--ratio", "2", "--seed", "4")) == 0
    m = json.loads((out / mix.MANIFEST).read_text(encoding="utf-8"))
    cands = [f"G{i}" for i in range(10)]
    assert m["gsv_ids"] == sorted(random.Random(4).sample(cands, 6))
    assert m["counts"] == {"sparse": 3, "gsv": 6, "train": 9}
    assert m["gsv"]["n_candidates"] == 10 and m["ratio"] == 2.0 and m["seed"] == 4
    names = sorted(os.listdir(out / "train"))
    assert len(names) == 18
    assert (out / "train" / "S1.jpg").read_bytes() == b"jpgS1"
    assert os.path.islink(out / "val") and os.listdir(out / "val") == ["V1.jpg", "V1.json"]
    assert mix.main(_argv(inputs, out, "--ratio", "2", "--seed", "4", "--check")) == 0


def test_check_catches_drift(inputs, tmp_path):
    out = tmp_path / "mix"
    mix.main(_argv(inputs, out, "--ratio", "1"))
    assert mix.main(_argv(inputs, out, "--ratio", "1", "--seed", "1", "--check")) == 1
    os.unlink(out / "train" / "S2.json")
    assert mix.main(_argv(inputs, out, "--ratio", "1", "--check")) == 1


def test_same_seed_same_mix(inputs, tmp_path):
    shas = []
    for name in ("a", "b"):
        mix.main(_argv(inputs, tmp_path / name, "--ratio", "1", "--seed", "7"))
        shas.append(json.loads((tmp_path / name / mix.MANIFEST).read_text())["train_sha256"])
    assert shas[0] == shas[1]


def test_refusals(inputs, tmp_path):
    sparse, gsv = inputs
    with pytest.raises(SystemExit, match="needs 33"):
        mix.main(_argv(inputs, tmp_path / "m1", "--ratio", "11"))
    out = tmp_path / "m2"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        mix.main(_argv(inputs, out, "--ratio", "1"))
    _pano(gsv / "train", "S1")                                      # an id in both sets
    with pytest.raises(SystemExit, match="in both"):
        mix.main(_argv(inputs, tmp_path / "m3", "--ratio", "1"))
    (gsv / "train" / "S1.jpg").unlink()
    (gsv / "train" / "S1.json").unlink()
    (sparse / "S3.jpg").unlink()
    with pytest.raises(SystemExit, match="no jpg"):
        mix.main(_argv(inputs, tmp_path / "m4", "--ratio", "1"))
