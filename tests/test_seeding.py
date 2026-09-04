"""Tests for the Stage 2 / YOLO seed plumbing (docs/seed_variance_51_135.md).

These are cheap source-level assertions rather than training runs, and they exist
because every failure mode here is SILENT: a sweep whose arms share a data order, a
positional argument list that shifts by one, or a default that quietly stops
reproducing the published run all produce a job that finishes green with the wrong
number in it.
"""

import ast
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


def _assigned_call(target_name):
    """Return the ``ast.Call`` assigned to ``target_name`` at module level in train.py.

    Substring assertions over the whole file are NOT good enough here. ``train.py`` also
    logs ``sampler_seed_for(args.seed)`` in a print(), so ``"sampler_seed_for(args.seed)"
    in src`` stays true even after the kwarg is deleted from the sampler itself -- which
    is the one regression this file exists to catch. Parsing the actual statement is the
    only assertion that distinguishes them.
    """
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == target_name for t in node.targets):
            continue
        value = node.value
        # `x = Call(...) if cond else None` -- the val sampler's shape
        if isinstance(value, ast.IfExp):
            value = value.body
        if isinstance(value, ast.Call):
            return value
    raise AssertionError(f"no module-level `{target_name} = <call>(...)` in train.py")


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


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


def test_seed_zero_shares_the_published_data_order():
    """The mapping is deliberately NOT injective, and seed 0 is where that bites.

    ``sampler_seed_for(0) == sampler_seed_for(42) == 0``, so a replicate at ``--seed 0``
    gets a different initialization from the published run but the SAME data order. That
    is a legal thing to want; it is not a legal thing to do by accident, because it makes
    two arms less independent than the seed column suggests. Campaigns A and B both use
    1/2/3, which is why this is documented rather than forbidden -- extend with 4, 5, ...,
    never with 0.
    """
    assert sampler_seed_for(0) == sampler_seed_for(HISTORICAL_SEED)


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
    flag or the sweep varies only part of the state.

    Matched line-anchored, not as substrings: ``"random.seed(args.seed)" in src`` is
    satisfied by ``np.random.seed(args.seed)`` alone, so the stdlib ``random`` call --
    the one that drives the horizontal-flip augmentation in EquiHeatmapDataset -- could
    be deleted without failing anything.
    """
    src = TRAIN_PY.read_text(encoding="utf-8")
    for call in ("torch.manual_seed(args.seed)",
                 "random.seed(args.seed)",
                 "np.random.seed(args.seed)"):
        assert re.search(rf"^{re.escape(call)}$", src, re.M), f"{call} missing"
    assert "torch.manual_seed(42)" not in src
    assert "np.random.seed(42)" not in src


def _unwrap_to(call, func_name):
    """Descend through wrapper calls to the ``func_name`` call inside.

    ``train_sampler`` is ``ResumeSkipSampler(DistributedSampler(...))`` since #135's
    resume-skip landed. The seed belongs on the INNER sampler -- putting it on the
    wrapper would be inert -- so this walks in rather than asserting on the outer call,
    and fails loudly if the target is not found at all.
    """
    seen = []
    while isinstance(call, ast.Call):
        name = call.func.id if isinstance(call.func, ast.Name) else None
        seen.append(name)
        if name == func_name:
            return call
        nested = [a for a in call.args if isinstance(a, ast.Call)]
        assert len(nested) == 1, (
            f"cannot find {func_name}(...) in the assignment; walked {seen} and the last "
            f"call has {len(nested)} nested calls, so which one to follow is ambiguous")
        call = nested[0]
    raise AssertionError(f"no {func_name}(...) call found; walked {seen}")


def test_train_sampler_seed_is_derived_not_hardcoded():
    """The regression this guards: dropping the sampler seed (back to its constant 0
    default) and leaving a sweep that varies initialization only.

    Asserted on the parsed ``train_sampler = DistributedSampler(...)`` statement, not on
    the file text. Deleting the kwarg passes any whole-file substring check, because the
    same expression appears in the seed log line.
    """
    call = _unwrap_to(_assigned_call("train_sampler"), "DistributedSampler")

    shuffle = _kwarg(call, "shuffle")
    assert isinstance(shuffle, ast.Constant) and shuffle.value is True, (
        "the train sampler must shuffle, or the seed is inert")

    seed = _kwarg(call, "seed")
    assert seed is not None, (
        "train_sampler has no seed= -- every replicate would reuse DistributedSampler's "
        "default of 0, i.e. one data order across the whole sweep, silently")
    assert ast.unparse(seed) == "sampler_seed_for(args.seed)", (
        f"train_sampler seed= is {ast.unparse(seed)!r}; it must be derived from --seed "
        f"via sampler_seed_for, not a literal")


def test_val_sampler_is_unshuffled_and_therefore_needs_no_seed():
    call = _assigned_call("val_sampler")
    assert _kwarg(call, "shuffle").value is False
    assert _kwarg(call, "seed") is None


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
    # Both spellings, "$VAR" and "${VAR}". Counting only the bare form left the check
    # blind to a braced positional: inserting "${LR0}" mid-list shifts data -> imgsz ->
    # epochs by one and used to pass green.
    n_passed = len(re.findall(r'"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"', call))

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
