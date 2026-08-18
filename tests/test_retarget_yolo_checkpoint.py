"""Unit tests for the cross-cluster checkpoint retarget (#51, #108).

Tiny synthetic checkpoints, no ultralytics — the thing under test is which keys get
rewritten and which are left alone.

What these protect: the failure mode this script exists to prevent is silent. A resumed
run whose ``save_dir`` still points at the ORIGINAL cluster's run directory writes its
results over the source run, and nothing errors. Equally silent is the opposite slip —
rewriting a hyperparameter along with the paths, which turns a continuation into a
different config (``epochs`` is the LR-decay denominator, not a label). So the tests
assert both directions: the six path keys change, and everything else does not.
"""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import retarget_yolo_checkpoint as rt  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts", "model_comparison", "retarget_yolo_checkpoint.py")

# A stand-in for the klone-trained arm: the six cluster-absolute paths, plus the
# hyperparameters that define the pre-registered schedule (#71).
KLONE_ARGS = {
    "data": "/gscratch/scrubbed/jonf/yolo/pano/data.yaml",
    "project": "/gscratch/makelab/jonf/yolo_runs",
    "name": "y11x_pano",
    "save_dir": "/gscratch/makelab/jonf/yolo_runs/y11x_pano",
    "model": "/gscratch/makelab/jonf/yolo_runs/y11x_pano/weights/last.pt",
    "resume": "/gscratch/makelab/jonf/yolo_runs/y11x_pano/weights/last.pt",
    "epochs": 60,
    "patience": 20,
    "lr0": 0.01,
    "batch": 6,
    "imgsz": 1280,
    "optimizer": "auto",
    "save_period": -1,
    "seed": 0,
}


def _ckpt(tmp_path, name="last.pt", **over):
    ta = dict(KLONE_ARGS)
    ta.update(over)
    p = tmp_path / name
    torch.save({"epoch": 37, "best_fitness": 0.55959, "train_args": ta}, p)
    return p


def _run(*argv):
    return subprocess.run([sys.executable, SCRIPT, *argv],
                          capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# retarget_paths — the ultralytics project/name layout
# --------------------------------------------------------------------------- #
def test_paths_follow_the_project_name_layout():
    new = rt.retarget_paths("/d/data.yaml", "/p/runs", "arm")
    assert new["save_dir"] == "/p/runs/arm"
    # model and resume both point at the run's own last.pt, not at the source file.
    assert new["model"] == new["resume"]
    assert new["model"] == "/p/runs/arm/weights/last.pt"
    assert set(new) == set(rt.PATH_KEYS)


def test_paths_are_idempotent():
    once = rt.retarget_paths("/d/data.yaml", "/p/runs", "arm")
    assert rt.retarget_paths(once["data"], once["project"], once["name"]) == once


def test_target_separators_do_not_follow_the_host():
    """Staging a Linux-cluster checkpoint from Windows must still write POSIX paths."""
    new = rt.retarget_paths("/gpfs/scrubbed/jonf/pano/data.yaml",
                            "/gpfs/projects/makelab/jonf/yolo_runs", "y11x_pano_h200")
    for k in ("save_dir", "model", "resume", "data"):
        assert "\\" not in new[k], (k, new[k])
    assert new["save_dir"] == "/gpfs/projects/makelab/jonf/yolo_runs/y11x_pano_h200"
    # A genuinely Windows target keeps Windows separators.
    win = rt.retarget_paths(r"D:\yolo\data.yaml", r"D:\yolo\runs", "arm")
    assert win["save_dir"] == r"D:\yolo\runs\arm"


# --------------------------------------------------------------------------- #
# dry run — the default must never write
# --------------------------------------------------------------------------- #
def test_dry_run_leaves_the_checkpoint_alone(tmp_path):
    ck = _ckpt(tmp_path)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")
    before = ck.read_bytes()

    r = _run(str(ck), "--data", str(yml), "--project", str(tmp_path / "runs"),
             "--name", "y11x_pano_h200")

    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert ck.read_bytes() == before
    assert not (tmp_path / "last.pt.preretarget").exists()


def test_missing_data_is_fatal_but_no_check_data_proceeds(tmp_path):
    ck = _ckpt(tmp_path)
    missing = tmp_path / "nope" / "data.yaml"

    r = _run(str(ck), "--data", str(missing), "--project", str(tmp_path / "runs"),
             "--name", "arm")
    assert r.returncode == 2
    assert "does not exist on this host" in r.stderr

    r = _run(str(ck), "--data", str(missing), "--project", str(tmp_path / "runs"),
             "--name", "arm", "--no-check-data")
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


# --------------------------------------------------------------------------- #
# apply — rewrite the six, keep the rest
# --------------------------------------------------------------------------- #
def test_apply_rewrites_only_the_path_keys(tmp_path):
    ck = _ckpt(tmp_path)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")
    project = tmp_path / "runs"

    r = _run(str(ck), "--data", str(yml), "--project", str(project),
             "--name", "y11x_pano_h200", "--apply")
    assert r.returncode == 0, r.stderr
    assert "verified" in r.stdout

    ta = torch.load(ck, map_location="cpu", weights_only=False)["train_args"]
    assert ta["save_dir"] == str(project / "y11x_pano_h200")
    assert ta["data"] == str(yml)
    assert ta["resume"] == str(project / "y11x_pano_h200" / "weights" / "last.pt")
    # The schedule is untouched: epochs is the LR denominator, so rewriting it would
    # change what every remaining epoch does rather than just relabelling the run.
    for k in ("epochs", "patience", "lr0", "batch", "imgsz", "optimizer",
              "save_period", "seed"):
        assert ta[k] == KLONE_ARGS[k], k
    # And the training state survives the round trip.
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 37 and ckpt["best_fitness"] == pytest.approx(0.55959)


def test_apply_backs_up_the_original_once(tmp_path):
    ck = _ckpt(tmp_path)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")
    original = ck.read_bytes()

    args = [str(ck), "--data", str(yml), "--project", str(tmp_path / "runs"),
            "--name", "arm", "--apply"]
    assert _run(*args).returncode == 0
    backup = tmp_path / "last.pt.preretarget"
    assert backup.read_bytes() == original

    # Second run is a no-op that must not overwrite the backup with retargeted bytes.
    r = _run(*args)
    assert r.returncode == 0
    assert "Already retargeted" in r.stdout
    assert backup.read_bytes() == original


def test_apply_stamps_provenance(tmp_path):
    ck = _ckpt(tmp_path)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")

    assert _run(str(ck), "--data", str(yml), "--project", str(tmp_path / "runs"),
                "--name", "arm", "--apply").returncode == 0

    info = torch.load(ck, map_location="cpu", weights_only=False)["retarget_info"]
    assert info["from"]["save_dir"] == KLONE_ARGS["save_dir"]
    assert info["to"]["save_dir"] == str(tmp_path / "runs" / "arm")
    assert info["script"] == "retarget_yolo_checkpoint.py"
    # Outside train_args, so ultralytics never sees an argument it does not know.
    assert "retarget_info" not in torch.load(
        ck, map_location="cpu", weights_only=False)["train_args"]


def test_out_writes_elsewhere_and_leaves_the_input(tmp_path):
    ck = _ckpt(tmp_path)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")
    out = tmp_path / "retargeted.pt"
    before = ck.read_bytes()

    assert _run(str(ck), "--data", str(yml), "--project", str(tmp_path / "runs"),
                "--name", "arm", "--out", str(out), "--apply").returncode == 0

    assert ck.read_bytes() == before
    assert not (tmp_path / "last.pt.preretarget").exists()
    ta = torch.load(out, map_location="cpu", weights_only=False)["train_args"]
    assert ta["save_dir"] == str(tmp_path / "runs" / "arm")


def test_bad_checkpoint_shape_is_rejected(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"epoch": 1, "train_args": "not-a-dict"}, p)
    yml = tmp_path / "data.yaml"
    yml.write_text("path: .\n")

    r = _run(str(p), "--data", str(yml), "--project", str(tmp_path / "runs"),
             "--name", "arm", "--apply")
    assert r.returncode == 2
    assert "expected dict" in r.stderr
