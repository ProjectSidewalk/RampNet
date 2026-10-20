"""Every .slurm launcher must at least PARSE.

This exists because of a real loss. `stage_two/run_train_seed.slurm` carried an
apostrophe inside a `${SEED:?...}` message; bash parses the word for quoting even
inside double quotes, so the closing brace was never found. The three klone Stage 2
seed replicates (39515025/26/27, #51 / #135) therefore died at job start with exit 2
in about one second each, printed nothing to stdout, and sat unnoticed for a day while
the paid Tillicum half of the same campaign ran normally.

Nothing else in the suite reads these files as shell. `test_seeding.py` asserts on
their CONTENT with regexes, which a syntactically broken script passes happily. A
launcher is the one artifact whose failure is invisible locally -- you find out on the
cluster, hours later, from an empty log -- so it is worth the two seconds `bash -n`
costs. CPU-only, no network, no cluster.
"""

import hashlib
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# rglob, but never descend into dot-directories: `.claude/worktrees/` holds whole
# nested checkouts of this repo, and collecting their launchers would multiply this
# test by the number of branches anyone happens to have on disk.
SLURM_SCRIPTS = sorted(
    p
    for p in REPO.rglob("*.slurm")
    if not any(part.startswith(".") for part in p.relative_to(REPO).parts)
)

BASH = shutil.which("bash")

requires_bash = pytest.mark.skipif(
    BASH is None,
    reason="bash not on PATH (CI runs ubuntu-latest, where it always is)",
)


def test_the_launchers_were_actually_found():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(SLURM_SCRIPTS) >= 15, [str(p) for p in SLURM_SCRIPTS]


@requires_bash
@pytest.mark.parametrize(
    "script", SLURM_SCRIPTS, ids=[str(p.relative_to(REPO)) for p in SLURM_SCRIPTS]
)
def test_slurm_script_parses(script):
    """`bash -n` reads the script and checks syntax without running a line of it."""
    proc = subprocess.run(
        [BASH, "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{script.relative_to(REPO)} is not valid bash -- sbatch would accept it and "
        f"the job would die in about a second with exit 2:\n{proc.stderr}"
    )


@requires_bash
def test_the_seed_guard_still_rejects_an_unset_seed():
    """The apostrophe fix must not have cost the guard its job.

    SEED is required rather than defaulted because a replicate that silently ran at 42
    would be a duplicate of the published run wearing a new name (#51 / #135). Sourcing
    the whole launcher would try to reach Slurm, so this evaluates the one expansion.
    """
    line = next(
        ln
        for ln in (REPO / "stage_two" / "run_train_seed.slurm").read_text().splitlines()
        if ln.startswith("SEED=")
    )
    proc = subprocess.run(
        [BASH, "-c", line], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
    )
    assert proc.returncode != 0, "an unset SEED must abort the job, not default"
    assert "SEED" in proc.stderr


def _seed_launcher_line(prefix):
    return next(
        ln
        for ln in (REPO / "stage_two" / "run_train_seed.slurm").read_text().splitlines()
        if ln.startswith(prefix)
    )


@requires_bash
def test_the_env_guard_rejects_an_unset_prefix():
    """RAMPNET_ENV used to fall back to `source activate sidewalkcv2`, which has nothing
    to resolve on klone (prefixes only, and sbatch inherits a PATH without conda). Jobs
    39583887/90/91 died that way on 2026-09-04. The guard must abort and name the
    variable, the same way the SEED guard does."""
    proc = subprocess.run(
        [BASH, "-c", _seed_launcher_line("RAMPNET_ENV=")],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0, "an unset RAMPNET_ENV must abort the job, not default"
    assert "RAMPNET_ENV" in proc.stderr


@requires_bash
def test_repo_defaults_to_the_submit_directory():
    """Slurm exports the directory sbatch ran in as SLURM_SUBMIT_DIR; USAGE requires that
    to be the repo root. Evaluated with the variable set by hand -- Slurm's own behaviour
    is documented, not tested here."""
    line = _seed_launcher_line("REPO=")
    proc = subprocess.run(
        [BASH, "-c", line + '; printf %s "$REPO"'],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "SLURM_SUBMIT_DIR": "/some/checkout", "HOME": "/h"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "/some/checkout"


@requires_bash
@pytest.mark.parametrize("has_seed_flag", [True, False], ids=["current", "stale"])
def test_the_checkout_preflight_rejects_a_train_py_without_seed(tmp_path, has_seed_flag):
    """39619547/48/49 ran a stale $HOME/RampNet whose train.py predates --seed, and the
    error surfaced 20 s in under a torch traceback. The preflight has to fail BEFORE
    torchrun, with a message that names the checkout, and pass on a current one."""
    src = (REPO / "stage_two" / "run_train_seed.slurm").read_text()
    m = re.search(r"^if ! grep -q -- '--seed'.*?^fi$", src, re.S | re.M)
    assert m, "no preflight block"
    (tmp_path / "stage_two").mkdir()
    train_py = tmp_path / "stage_two" / "train.py"
    train_py.write_text(
        'parser.add_argument("--seed", type=int)\n' if has_seed_flag
        else 'parser.add_argument("--epochs", type=int)\n'
    )
    proc = subprocess.run(
        [BASH, "-c", "REPO=" + tmp_path.as_posix() + "\n" + m.group(0)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    if has_seed_flag:
        assert proc.returncode == 0, proc.stderr
    else:
        assert proc.returncode == 1
        assert "stale checkout" in proc.stderr and tmp_path.as_posix() in proc.stderr


@requires_bash
def test_the_commit_line_survives_a_checkout_that_is_not_a_repo():
    """The banner runs under `set -euo pipefail`. A checkout rsync'd without .git, or a
    node without git on PATH, must print `unknown` for BOTH fields -- `dirty files: 0`
    beside an unknown commit reads as a clean checkout when nothing was measured -- and
    must not kill the job before torchrun."""
    src = (REPO / "stage_two" / "run_train_seed.slurm").read_text()
    m = re.search(r'^if commit=\$\(git -C "\$REPO" rev-parse.*?^echo "Repo commit: .*?$',
                  src, re.S | re.M)
    assert m, "no commit/dirty status block in the banner"
    proc = subprocess.run(
        [BASH, "-c", "set -euo pipefail\nREPO=/nonexistent\n" + m.group(0)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Repo commit: unknown   dirty files: unknown"


# ------------------------------------------------ #263 control arm (run_finetune_control)

CONTROL = REPO / "stage_two" / "run_finetune_control.slurm"
AUG82 = REPO / "stage_two" / "run_finetune_aug82.slurm"


def _block(path, start, end):
    src = path.read_text()
    i = src.index(start)
    return src[i:src.index(end, i)]


def test_control_keeps_every_aug82_preflight_verbatim():
    """The env-tar unpack, the sha-checked conversion of the released weights, the
    world-size check and the shared-library warm-up are copied, not rewritten: each one is
    a fix for a klone failure (see the aug82 header). Only the per-campaign paths differ."""
    renames = (("aug82env_", "ctrl263env_"),
               ("$USER/aug82/sidewalkcv2_env.tar", "$USER/control263/sidewalkcv2_env.tar"),
               ("eight jobs starting together", "three seeds starting together"))
    for start, end in (("# Run the env from node-local NVMe", 'if commit=$(git -C "$REPO"'),
                       ("# Read the env's big shared libraries once", 'cd "$RUNDIR"')):
        want = _block(AUG82, start, end)
        for old, new in renames:
            want = want.replace(old, new)
        assert _block(CONTROL, start, end) == want, start
    src = CONTROL.read_text()
    assert "HF_SHA256=f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5" in src
    assert 'sha256sum -c --quiet' in src


def test_control_is_the_recipe_with_a_step_budget_and_nothing_else():
    src = CONTROL.read_text()
    call = src[src.index('             "$REPO/stage_two/train.py"'):
               src.index('echo "--- Slurm job finished')]
    args = shlex.split(call.replace("\\n", " "))
    assert "--aug" not in args and "--loss-mask" not in args and "--lr-schedule" not in args
    for flag, value in (("--seed", "$SEED"), ("--init-weights", "$INIT"),
                        ("--max-steps", "$STEPS"), ("--grad-accum", "$ACCUM"),
                        ("--lr", "$LR"), ("--data-root", "$DATA_ROOT")):
        assert args[args.index(flag) + 1] == value, flag
    for line in ('STEPS="${STEPS:-2000}"', 'GLOBAL_BATCH="${GLOBAL_BATCH:-16}"',
                 'LR="${LR:-1e-5}"',
                 'RUNDIR="${RUNDIR:-/gscratch/scrubbed/$USER/control263/s${SEED}}"',
                 'DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}"',
                 'INIT="${INIT:-/gscratch/scrubbed/$USER/aug82/released_rampnet_state_dict.pth}"',
                 'COPY_DEST="${COPY_DEST:-/gscratch/makelab/jonf/control263/s${SEED}}"'):
        assert any(ln.split('  #')[0].rstrip() == line for ln in src.splitlines()), line


@requires_bash
def test_control_seed_guard_rejects_an_unset_seed():
    line = next(ln for ln in CONTROL.read_text().splitlines() if ln.startswith("SEED="))
    proc = subprocess.run([BASH, "-c", line], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode != 0 and "SEED" in proc.stderr


@requires_bash
@pytest.mark.parametrize("finished", [True, False], ids=["finished", "unfinished"])
def test_control_copy_step(tmp_path, finished):
    """The copy step is the one new piece of shell here: it must copy the final weights
    with a sha256 beside them, and fail the job when the final weights do not exist."""
    m = re.search(r"^# --- copy step.*?^# --- end copy step ---$", CONTROL.read_text(),
                  re.S | re.M)
    assert m, "no copy-step block"
    run, dest, repo = tmp_path / "run", tmp_path / "dest" / "s1", tmp_path / "repo"
    (run / "checkpoints").mkdir(parents=True)
    (run / "runs" / "experiment_1").mkdir(parents=True)
    (run / "runs" / "experiment_1" / "events.out").write_text("ev")
    (repo / "logs").mkdir(parents=True)
    (repo / "logs" / "control263_77.out").write_text("log")
    if finished:
        (run / "checkpoints" / "final_step_5.pth").write_bytes(b"weights")
    script = "\n".join(["set -euo pipefail", f"RUNDIR={run.as_posix()}",
                        f"COPY_DEST={dest.as_posix()}", f"REPO={repo.as_posix()}",
                        "STEPS=5", "SLURM_JOB_ID=77", m.group(0)])
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    if not finished:
        assert proc.returncode == 1 and "final_step_5.pth" in proc.stderr
        assert not dest.exists()
        return
    assert proc.returncode == 0, proc.stderr
    assert (dest / "final_step_5.pth").read_bytes() == b"weights"
    sha = (dest / "final_step_5.pth.sha256").read_text().split()
    # Git Bash on Windows marks binary mode with a leading "*"; coreutils on Linux does not
    assert [sha[0], sha[1].lstrip("*")] == [hashlib.sha256(b"weights").hexdigest(),
                                            "final_step_5.pth"]
    assert (dest / "runs" / "experiment_1" / "events.out").read_text() == "ev"
    # review L7: the job log stays in logs/ on lab storage; nothing else is copied
    assert sorted(x.name for x in dest.iterdir()) == ["final_step_5.pth",
                                                      "final_step_5.pth.sha256", "runs"]


def _run_block(script_path, start, end, prelude, **kw):
    m = re.search(rf"^{re.escape(start)}.*?^{re.escape(end)}.*?$", script_path.read_text(),
                  re.S | re.M)
    assert m, (start, end)
    return subprocess.run([BASH, "-c", "set -euo pipefail\n" + prelude + "\n" + m.group(0)],
                          capture_output=True, text=True, **kw)


@requires_bash
@pytest.mark.parametrize("case", ["missing", "present", "other_name"])
def test_control_rebuilds_a_missing_env_tarball(tmp_path, case):
    """The #82 tarball was purged; the control rebuilds it once (atomically) rather than
    running four ranks off the GPFS env, and leaves an existing tar or an env it cannot
    unpack by name alone."""
    if case == "missing" and sys.platform == "win32":
        pytest.skip("GNU tar reads C:/... as host:file; klone paths have no colon")
    name = "sidewalkcv2" if case != "other_name" else "otherenv"
    env = tmp_path / "envs" / name
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").write_text("py")
    tar = tmp_path / "cache" / "sidewalkcv2_env.tar"
    if case == "present":
        tar.parent.mkdir()
        tar.write_text("existing")
    proc = _run_block(CONTROL, "# --- env tarball rebuild", "# --- end env tarball rebuild",
                      f"RAMPNET_ENV={env.as_posix()}\nENV_TAR={tar.as_posix()}\n"
                      "SLURM_JOB_ID=9")
    assert proc.returncode == 0, proc.stderr
    if case == "present":
        assert tar.read_text() == "existing"
    elif case == "other_name":
        assert not tar.exists()
    else:
        import tarfile
        with tarfile.open(tar) as tf:
            assert "sidewalkcv2/bin/python" in tf.getnames()
        assert not list(tar.parent.glob("*.tmp.*"))
        assert "Built env tarball" in proc.stdout


def _fake_tar(tmp_path, body):
    """A `tar` first on PATH: `tar cf <out> ...` runs ``body`` with $out set."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "tar"
    script.write_text("#!/bin/bash\nout=\"$2\"\n" + body + "\n", newline="\n")
    script.chmod(0o755)
    return bin_dir


def _path_entry(path):
    """A directory as a PATH entry: Git Bash on Windows splits PATH on ':', so C:/x is
    written /c/x there."""
    posix = path.as_posix()
    if sys.platform == "win32" and len(posix) > 1 and posix[1] == ":":
        posix = "/" + posix[0].lower() + posix[2:]
    return posix


def _tar_block(tmp_path, fake_body):
    env = tmp_path / "envs" / "sidewalkcv2"
    env.mkdir(parents=True)
    tar = tmp_path / "cache" / "sidewalkcv2_env.tar"
    bin_dir = _fake_tar(tmp_path, fake_body.replace("@TAR@", tar.as_posix()))
    proc = _run_block(CONTROL, "# --- env tarball rebuild", "# --- end env tarball rebuild",
                      f"export PATH={_path_entry(bin_dir)}:$PATH\n"
                      f"RAMPNET_ENV={env.as_posix()}\nENV_TAR={tar.as_posix()}\n"
                      "SLURM_JOB_ID=9")
    return proc, tar


@requires_bash
def test_control_tar_rebuild_that_loses_the_race_says_so(tmp_path):
    """Review L6: when another job's tar lands first, `mv -n` keeps it, this job's temp
    file is removed, and the log does not claim this job built anything."""
    proc, tar = _tar_block(tmp_path, 'echo mine > "$out"; echo theirs > "@TAR@"')
    assert proc.returncode == 0, proc.stderr
    assert tar.read_text().strip() == "theirs"
    assert "finished first by another job" in proc.stdout
    assert "Built env tarball" not in proc.stdout
    assert not list(tar.parent.glob("*.tmp.*"))


@requires_bash
def test_control_tar_rebuild_cleans_up_when_preempted(tmp_path):
    """Review L6: a SIGTERM during `tar cf` (a ckpt-all preemption) must not leave the
    multi-GB partial file behind."""
    proc, tar = _tar_block(tmp_path, 'echo partial > "$out"; kill -TERM $PPID; sleep 1')
    assert proc.returncode == 143, (proc.returncode, proc.stderr)
    assert not tar.exists()
    assert not list(tar.parent.glob("*.tmp.*"))


@requires_bash
@pytest.mark.parametrize("n_train, lone_jpg, ok", [(3, False, True), (2, False, False),
                                                    (2, True, False)])
def test_control_refuses_a_thinned_dataset(tmp_path, n_train, lone_jpg, ok):
    """The last staged copy lost most of train/ and all of val/ to the scrubbed purge
    without anything failing. The preflight counts jpg+json PAIRS before any GPU time: a
    jpg without its json (review L1) is not a pano train.py will read."""
    for split, n in (("train", n_train), ("val", 1)):
        (tmp_path / split).mkdir()
        for i in range(n):
            (tmp_path / split / f"p{i}.jpg").write_text("x")
            (tmp_path / split / f"p{i}.json").write_text("{}")
    if lone_jpg:
        (tmp_path / "train" / "lone.jpg").write_text("x")
    line = next(ln for ln in CONTROL.read_text().splitlines() if ln.startswith("DATA_ROOT="))
    proc = _run_block(CONTROL, "EXPECT_TRAIN=", "done",
                      f"DATA_ROOT={tmp_path.as_posix()}\nEXPECT_TRAIN=3\nEXPECT_VAL=1\n"
                      + line)
    assert (proc.returncode == 0) == ok, proc.stderr
    if not ok:
        assert "has 2 jpg+json pairs, expected 3" in proc.stderr


# --------------------------------------------- #263 dataset re-stage (run_restage_263)

RESTAGE = REPO / "stage_two" / "run_restage_263.slurm"


def _assignment(path, var):
    line = next(ln for ln in path.read_text().splitlines() if ln.startswith(var + "="))
    return line.split("=", 1)[1].split("  #")[0].strip()


def test_restage_writes_where_the_control_reads():
    """The re-stage target, the control arm's default DATA_ROOT and the counts the control
    checks must agree, or the control refuses a correct dataset (or reads a stale one)."""
    out = _assignment(RESTAGE, "OUT")
    assert out == "/gscratch/scrubbed/jfroehli/rampnet_dataset_v2"
    data_root = _assignment(CONTROL, "DATA_ROOT")
    assert data_root == '"${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}"'
    src = RESTAGE.read_text()
    train = _assignment(CONTROL, "EXPECT_TRAIN").strip('"').split("-")[1].rstrip("}")
    val = _assignment(CONTROL, "EXPECT_VAL").strip('"').split("-")[1].rstrip("}")
    assert f"{int(train):,} / {int(val):,}" in src


def test_restage_runs_download_dataset_the_way_it_writes():
    """download_dataset.py takes no arguments and writes ./dataset relative to its cwd, so
    the job must cd into a dir whose `dataset` is the target. Its scratch must not be /tmp
    (node-local on klone) or $HOME (a 10 GB quota)."""
    dl = (REPO / "download_dataset.py").read_text()
    assert 'OUTPUT_DIR = "./dataset"' in dl
    src = RESTAGE.read_text()
    assert 'ln -sfn "$OUT" "$STAGE/dataset"' in src and 'cd "$STAGE"' in src
    assert src.index('cd "$STAGE"') < src.index('"$PY" "$REPO/download_dataset.py"')
    assert "set -euo pipefail" in src
    hf_home = _assignment(RESTAGE, "export HF_HOME")
    assert hf_home.startswith("/gscratch/") and "/tmp" not in src


@requires_bash
@pytest.mark.skipif(sys.platform == "win32", reason="Git Bash ln -s copies instead of linking")
def test_restage_symlink_lands_the_files_in_out(tmp_path):
    src = RESTAGE.read_text()
    body = src[src.index('mkdir -p "$OUT" "$STAGE"'):src.index('echo "host')]
    out, stage = tmp_path / "v2", tmp_path / "stage"
    script = "\n".join(["set -euo pipefail", f"OUT={out.as_posix()}",
                        f"STAGE={stage.as_posix()}", body,
                        "mkdir -p dataset/train && touch dataset/train/a.jpg"])
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "train" / "a.jpg").exists()
