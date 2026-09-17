"""Every .slurm launcher must at least PARSE.

This exists because of a real loss. `stage_two/run_train_seed.slurm` carried an
apostrophe inside a `${SEED:?...}` message; bash parses the word for quoting even
inside double quotes, so the closing brace was never found. The three klone Stage 2
seed replicates (39515025/26/27, #51 / #135) therefore died at submit time with exit 2
in about one second each, printed nothing to stdout, and sat unnoticed for a day while
the paid Tillicum half of the same campaign ran normally.

Nothing else in the suite reads these files as shell. `test_seeding.py` asserts on
their CONTENT with regexes, which a syntactically broken script passes happily. A
launcher is the one artifact whose failure is invisible locally -- you find out on the
cluster, hours later, from an empty log -- so it is worth the two seconds `bash -n`
costs. CPU-only, no network, no cluster.
"""

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
