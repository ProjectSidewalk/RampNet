"""Tests for the Stage 2 / YOLO seed plumbing (docs/seed_variance_51_135.md).

These are cheap source-level assertions rather than training runs, and they exist
because every failure mode here is SILENT: a sweep whose arms share a data order, a
positional argument list that shifts by one, or a default that quietly stops
reproducing the published run all produce a job that finishes green with the wrong
number in it.
"""

import re
from pathlib import Path

import pytest

from rampnet.seeding import (
    HISTORICAL_SAMPLER_SEED,
    HISTORICAL_SEED,
    sampler_seed_for,
)

REPO = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO / "stage_two" / "train.py"
SEED_SLURM = REPO / "stage_two" / "run_train_seed.slurm"
TILLICUM_SLURM = REPO / "scripts" / "model_comparison" / "run_yolo_train_tillicum.slurm"


# --------------------------------------------------------------------------------
# sampler_seed_for
# --------------------------------------------------------------------------------

def test_historical_seed_maps_to_the_historical_sampler_seed():
    """The default must reproduce published runs, which paired manual_seed(42) with
    DistributedSampler's own default of 0. If this ever returns 42, every reproduction
    silently gets a different data order."""
    assert sampler_seed_for(HISTORICAL_SEED) == HISTORICAL_SAMPLER_SEED
    assert HISTORICAL_SEED != HISTORICAL_SAMPLER_SEED  # or the mapping is vacuous


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 41, 43, 1234])
def test_every_other_seed_maps_to_itself(seed):
    """A sweep has to move the data order too; identity is what guarantees that."""
    assert sampler_seed_for(seed) == seed


def test_distinct_seeds_give_distinct_sampler_seeds():
    """The campaign's three replicates must not collide with each other or with the
    published run's data order."""
    campaign = [1, 2, 3]
    sampler_seeds = [sampler_seed_for(s) for s in campaign]
    assert len(set(sampler_seeds)) == len(campaign)
    assert HISTORICAL_SAMPLER_SEED not in sampler_seeds


# --------------------------------------------------------------------------------
# stage_two/train.py -- source assertions (importing it runs argparse and DDP setup)
# --------------------------------------------------------------------------------

def test_train_py_declares_seed_defaulting_to_the_historical_value():
    src = TRAIN_PY.read_text(encoding="utf-8")
    assert "'--seed'" in src
    assert re.search(r"add_argument\(\s*'--seed',\s*type=int,\s*default=HISTORICAL_SEED",
                     src), "the --seed default must be the named constant, not a literal"


def test_train_py_seeds_all_three_rngs_from_args():
    """torch, numpy and random were all hardcoded to 42; all three have to follow the
    flag or the sweep varies only part of the state."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    for call in ("torch.manual_seed(args.seed)",
                 "random.seed(args.seed)",
                 "np.random.seed(args.seed)"):
        assert call in src, f"{call} missing"
    assert "torch.manual_seed(42)" not in src
    assert "np.random.seed(42)" not in src


def test_train_sampler_seed_is_derived_not_hardcoded():
    """The regression this guards: dropping the sampler seed (back to a constant 0) and
    leaving a sweep that varies initialization only."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    sampler_line = next(l for l in src.splitlines()
                        if "DistributedSampler(train_dataset" in l or
                        (l.strip().startswith("seed=sampler_seed_for")))
    assert "sampler_seed_for(args.seed)" in src
    assert "shuffle=True" in src
    # and the val sampler must NOT be shuffled, so it needs no seed
    val_line = next(l for l in src.splitlines() if "val_sampler = DistributedSampler" in l)
    assert "shuffle=False" in val_line


def test_train_py_logs_both_seeds():
    """A replicate whose log cannot identify its own seed is not reproducible."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    assert 'f"Seed: {args.seed} (sampler seed: {sampler_seed_for(args.seed)})' in src


# --------------------------------------------------------------------------------
# stage_two/run_train_seed.slurm
# --------------------------------------------------------------------------------

def test_seed_launcher_requires_a_seed():
    """Defaulting SEED would make an un-set replicate a silent duplicate of the
    published run."""
    src = SEED_SLURM.read_text(encoding="utf-8")
    assert 'SEED="${SEED:?' in src, "SEED must be required (:?), never defaulted"


def test_seed_launcher_isolates_each_replicate_in_its_own_directory():
    """train.py writes best_model.pth and latest_checkpoint.pth to the CWD, so shared
    directories cross-contaminate resume state between replicates."""
    src = SEED_SLURM.read_text(encoding="utf-8")
    assert 'RUNDIR="${RUNDIR:-' in src
    assert "rampnet_s${SEED}" in src, "the default RUNDIR must be per-seed"
    assert re.search(r'^cd "\$RUNDIR"$', src, re.M), "must cd into RUNDIR before torchrun"
    assert "--seed \"$SEED\"" in src


def test_seed_launcher_does_not_write_checkpoints_to_home():
    """klone home is a separate 10 GB quota; per-epoch checkpoints blow it."""
    src = SEED_SLURM.read_text(encoding="utf-8")
    rundir = next(l for l in src.splitlines() if l.startswith('RUNDIR='))
    assert "$HOME" not in rundir and "~/" not in rundir


# --------------------------------------------------------------------------------
# scripts/model_comparison/run_yolo_train_tillicum.slurm
# --------------------------------------------------------------------------------

def test_tillicum_launcher_defaults_seed_to_the_published_value():
    """Every #51 arm ran seed=0, so the default has to stay 0 or reproducing an arm
    silently trains a different model."""
    src = TILLICUM_SLURM.read_text(encoding="utf-8")
    assert 'SEED="${SEED:-0}"' in src


def test_tillicum_launcher_positional_args_line_up():
    """The training call passes bare positionals into a heredoc, so an added argument
    that is not also unpacked shifts every later field by one -- imgsz becoming epochs,
    and so on. Nothing would raise; the run would just be wrong."""
    src = TILLICUM_SLURM.read_text(encoding="utf-8")

    call = src.split("run_train() {", 1)[1].split("<<'PY'", 1)[0]
    n_passed = len(re.findall(r'"\$[A-Z_]+"', call))

    unpack = re.search(r"\)\s*=\s*sys\.argv\[1:(\d+)\]", src)
    assert unpack, "could not find the sys.argv unpack"
    n_unpacked = int(unpack.group(1)) - 1

    assert n_passed == n_unpacked, (
        f"{n_passed} shell positionals but sys.argv[1:{n_unpacked + 1}] unpacks "
        f"{n_unpacked}")

    names = re.search(r"\(ckpt, data, .*?\)\s*=\s*sys\.argv", src, re.S).group(0)
    assert len([n for n in names.split("=")[0].strip("() ").split(",") if n.strip()]) == n_unpacked


def test_tillicum_launcher_forwards_seed_to_ultralytics():
    src = TILLICUM_SLURM.read_text(encoding="utf-8")
    assert "seed=int(seed)" in src, "seed must reach YOLO.train(), not just the shell"
    assert '"$SEED"' in src.split("run_train() {", 1)[1].split("<<'PY'", 1)[0]


def test_tillicum_launcher_echoes_the_seed():
    """The as-run seed has to be recoverable from the job log, not only from args.yaml
    inside a checkpoint that may be purged."""
    src = TILLICUM_SLURM.read_text(encoding="utf-8")
    assert "seed: ${SEED}" in src
