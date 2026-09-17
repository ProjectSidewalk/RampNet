"""Unit tests for Stage 2's learning-rate schedule (#135 cosine rung).

`stage_two/train.py` produced the released model, so the schedule was added in a way
that cannot change anything unless it is asked for. The three things that would each
be a silent, expensive failure:

- **The default must still be the paper recipe.** Any drift here re-trains every
  future run on a different recipe than the committed numbers describe.
- **The schedule must be a pure function of the step index.** Stage 2 runs on a
  preemptible partition and resumes from `latest_checkpoint.pth`; a scheduler holding
  its own state restarts the decay from the peak on every requeue, turning a cosine
  into a sawtooth. The run still completes and the loss curve still looks plausible,
  so nothing catches it downstream.
- **The endpoints must be right.** A cosine that never reaches its floor, or that
  starts below the peak, is a different experiment than the one pre-registered.

Importing train.py executes it, so the function is loaded from source rather than
imported — the module is a script, not a library, and running it here would try to
build a model and open a dataset.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO, "stage_two", "train.py")
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

# The AST lift lives with the script that checks the *applied* schedule against it, so
# there is one copy: this file tests the formula, that one tests where it was applied.
from check_lr_schedule_135 import load_schedule  # noqa: E402

SCHED = load_schedule()
TOTAL = 8 * 9378          # the rung: 8 epochs at world size 16
PEAK = 1e-5


def test_default_is_the_paper_recipe():
    """`constant` must be flat, and it must be what train.py defaults to."""
    for step in (0, 1, TOTAL // 3, TOTAL - 1, TOTAL, TOTAL * 2):
        assert SCHED.lr_at_step(step, TOTAL, PEAK, "constant") == PEAK

    with open(TRAIN_PY, encoding="utf-8") as fh:
        src = fh.read()
    assert "'--lr-schedule', choices=LR_SCHEDULES, default='constant'" in src, (
        "the default schedule must stay 'constant' -- it is the paper recipe and "
        "every committed Stage 2 number was produced under it")


def test_cosine_endpoints():
    assert SCHED.lr_at_step(0, TOTAL, PEAK, "cosine") == pytest.approx(PEAK)
    assert SCHED.lr_at_step(TOTAL, TOTAL, PEAK, "cosine") == pytest.approx(0.0, abs=1e-18)
    # Half way through a cosine is half the peak.
    assert SCHED.lr_at_step(TOTAL // 2, TOTAL, PEAK, "cosine") == pytest.approx(PEAK / 2, rel=1e-3)


def test_cosine_is_monotone_decreasing():
    prev = float("inf")
    for step in range(0, TOTAL + 1, TOTAL // 200):
        lr = SCHED.lr_at_step(step, TOTAL, PEAK, "cosine")
        assert lr <= prev + 1e-18
        prev = lr


def test_final_frac_sets_the_floor():
    lr = SCHED.lr_at_step(TOTAL, TOTAL, PEAK, "cosine", final_frac=0.1)
    assert lr == pytest.approx(PEAK * 0.1)


def test_schedule_survives_preemption_by_construction():
    """The property the whole design exists for.

    Simulate a run chopped into arbitrary segments by requeues. Because the rate is a
    function of the absolute step index — which `latest_checkpoint.pth` already
    carries — the sequence of learning rates must be identical to an uninterrupted
    run. A stateful scheduler restarted per segment would produce a sawtooth here.
    """
    uninterrupted = [SCHED.lr_at_step(s, TOTAL, PEAK, "cosine") for s in range(0, TOTAL, 977)]

    # Three requeues at awkward points; each segment resumes from the saved step.
    resumed, cuts = [], [0, 3_211, 40_000, 61_117, TOTAL]
    for start, end in zip(cuts, cuts[1:]):
        global_step = start                       # what the checkpoint carried
        for offset in range(0, end - start):
            step_index = global_step + offset     # exactly train.py's reconstruction
            if step_index % 977 == 0:
                resumed.append(SCHED.lr_at_step(step_index, TOTAL, PEAK, "cosine"))

    assert resumed == uninterrupted


def test_beyond_the_horizon_clamps_rather_than_rising():
    """Past total_steps the cosine would turn back UP; it must clamp at the floor.

    Reachable if a run is ever extended by raising --epochs on a resume, which is
    exactly the kind of thing that happens late at night.
    """
    end = SCHED.lr_at_step(TOTAL, TOTAL, PEAK, "cosine")
    for over in (TOTAL + 1, TOTAL * 2, TOTAL * 10):
        assert SCHED.lr_at_step(over, TOTAL, PEAK, "cosine") == pytest.approx(end, abs=1e-18)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        SCHED.lr_at_step(0, TOTAL, PEAK, "linear")
    assert SCHED.LR_SCHEDULES == ("constant", "cosine")
