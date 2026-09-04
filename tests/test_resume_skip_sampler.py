"""Unit tests for the resume fast-forward fix (#135).

Resuming used to fast-forward with `if i < batch_idx_in_epoch: continue` *inside* the
training loop, which pulls every skipped batch through the DataLoader -- decoding a
2048x4096 panorama per skipped step -- and discards it. Measured on the cosine rung:
4.6 min from job start to first training step on a fresh start, 23-30 min on a resume.
On a preemptible partition that is a livelock, not a slow start: once resume cost
exceeds the gap between preemptions, no incarnation ever reaches a checkpoint and the
run advances by nothing. That is what job 38640313 did for 8h54m at step 9,000.

`ResumeSkipSampler` drops the *indices* instead, so the workers never fetch them.

The property that has to hold is **equivalence**: the batches that remain, and their
order, must be exactly what the discard loop would have processed. If that ever breaks,
a resumed run silently trains on a different sample of the epoch than an uninterrupted
one -- which no metric downstream would reveal. `test_skipping_matches_the_old_discard_
loop` is the test that matters; the rest guard the bookkeeping around it.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Sampler  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from check_lr_schedule_135 import load_from_train_py  # noqa: E402

from rampnet.seeding import HISTORICAL_SEED

import itertools  # noqa: E402

LIFTED = load_from_train_py("ResumeSkipSampler", itertools=itertools, Sampler=Sampler)
ResumeSkipSampler = LIFTED.ResumeSkipSampler


class FakeDistributedSampler:
    """Stands in for DistributedSampler: a permutation that depends on (seed, epoch)."""

    def __init__(self, n, seed=0):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.iter_calls = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.iter_calls += 1
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n


# --------------------------------------------------------------------------- #
# the property that matters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("skip", [0, 1, 9, 500, 9377])
def test_skipping_matches_the_old_discard_loop(skip):
    """Same remaining indices, same order, as `if i < skip: continue` would have given."""
    base = FakeDistributedSampler(9378)
    base.set_epoch(3)
    old_way = [idx for i, idx in enumerate(iter(base)) if i >= skip]

    wrapped = ResumeSkipSampler(FakeDistributedSampler(9378))
    wrapped.set_epoch(3)
    wrapped.skip = skip
    assert list(wrapped) == old_way


def test_a_different_epoch_reshuffles_but_still_matches():
    """The skip must compose with set_epoch, not fight it."""
    for epoch in (0, 1, 7):
        base = FakeDistributedSampler(500)
        base.set_epoch(epoch)
        old_way = [idx for i, idx in enumerate(iter(base)) if i >= 123]

        wrapped = ResumeSkipSampler(FakeDistributedSampler(500))
        wrapped.set_epoch(epoch)
        wrapped.skip = 123
        assert list(wrapped) == old_way


def test_skipping_does_not_consume_the_dataset():
    """The point of the fix: skipped indices are never handed downstream at all."""
    wrapped = ResumeSkipSampler(FakeDistributedSampler(9378))
    wrapped.skip = 9000
    assert len(list(wrapped)) == 378


# --------------------------------------------------------------------------- #
# bookkeeping the training loop depends on
# --------------------------------------------------------------------------- #
def test_epoch_length_is_the_unskipped_count():
    """The LR horizon reads this. If it moved on resume, the cosine would change shape.

    That is the subtle one: `total_train_steps = num_epochs * epoch_length`, so a
    resumed run whose horizon shrank would decay faster than the run it is continuing,
    and the schedule under test would silently not be the pre-registered one.
    """
    wrapped = ResumeSkipSampler(FakeDistributedSampler(9378))
    assert wrapped.epoch_length == 9378
    wrapped.skip = 9000
    assert wrapped.epoch_length == 9378, "epoch_length must not move when skip is set"


def test_len_is_what_remains():
    """`num_batches_processed_this_epoch = len(train_loader)` relies on this."""
    wrapped = ResumeSkipSampler(FakeDistributedSampler(9378))
    assert len(wrapped) == 9378
    wrapped.skip = 9000
    assert len(wrapped) == 378
    wrapped.skip = 0
    assert len(wrapped) == 9378


def test_len_never_goes_negative():
    wrapped = ResumeSkipSampler(FakeDistributedSampler(10))
    wrapped.skip = 25
    assert len(wrapped) == 0
    assert list(wrapped) == []


def test_zero_skip_passes_the_iterator_straight_through():
    """A fresh start must be byte-for-byte the old behaviour, not islice(…, 0, None)."""
    base = FakeDistributedSampler(50)
    wrapped = ResumeSkipSampler(base)
    assert list(wrapped) == [idx for idx in iter(FakeDistributedSampler(50))]


def test_set_epoch_reaches_the_base_sampler():
    base = FakeDistributedSampler(10)
    wrapped = ResumeSkipSampler(base)
    wrapped.set_epoch(5)
    assert base.epoch == 5


# --------------------------------------------------------------------------- #
# the loop's index arithmetic, reproduced
# --------------------------------------------------------------------------- #
def test_checkpoint_interval_default_is_still_the_paper_recipe():
    """1000 produced every committed Stage 2 number; only the rung opts down to 200.

    The rung passes `--checkpoint-interval-steps 200` from its slurm launcher. If the
    *default* ever drifted, every other Stage 2 run would silently change its
    checkpointing granularity along with it.
    """
    import argparse
    mod = load_from_train_py("parse_args", "PRESET_LR", "LR_SCHEDULES",
                             argparse=argparse, HISTORICAL_SEED=HISTORICAL_SEED)
    argv = sys.argv
    try:
        sys.argv = ["train.py"]
        assert mod.parse_args().checkpoint_interval_steps == 1000
        sys.argv = ["train.py", "--checkpoint-interval-steps", "200"]
        assert mod.parse_args().checkpoint_interval_steps == 200
        sys.argv = ["train.py", "--checkpoint-interval-steps", "0"]
        with pytest.raises(SystemExit):
            mod.parse_args()
    finally:
        sys.argv = argv

    slurm = os.path.join(REPO, "stage_two", "run_train_cosine_rung.slurm")
    with open(slurm, encoding="utf-8") as fh:
        src = fh.read()
    assert "--checkpoint-interval-steps" in src, (
        "the rung must pass the interval explicitly -- inheriting the 1000 default is "
        "what let 8h54m of ckpt-all slices bank nothing")


def test_resumed_epoch_lands_on_the_same_global_step():
    """resume_offset + processed == epoch_length, so global_step is unchanged by a resume.

    Mirrors the training loop: `step_index = global_step + i`, checkpoints record
    `resume_offset + i + 1`, and after the epoch `global_step += len(train_loader)`.
    """
    epoch_length, global_step, resume_offset = 9378, 9000, 9000
    wrapped = ResumeSkipSampler(FakeDistributedSampler(epoch_length))
    wrapped.skip = resume_offset

    steps, batch_idxs = [], []
    for i, _idx in enumerate(wrapped):
        steps.append(global_step + i + 1)          # current_total_step
        batch_idxs.append(resume_offset + i + 1)   # checkpointed batch_idx_in_epoch

    assert steps[0] == 9001 and steps[-1] == 9378
    assert batch_idxs[-1] == epoch_length
    assert global_step + len(wrapped) == epoch_length
