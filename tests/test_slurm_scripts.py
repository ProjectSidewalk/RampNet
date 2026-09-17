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
