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
               ("$USER/aug82/released_rampnet_state_dict.pth",
                "$USER/control263/released_rampnet_state_dict.pth"),
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
    assert (dest / "control263_77.out").read_text() == "log"
