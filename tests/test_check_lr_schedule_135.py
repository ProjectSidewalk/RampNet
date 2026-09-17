"""Unit tests for the applied-LR check (#135 cosine rung).

The script this covers exists to catch one failure: a requeue restarting the learning-rate
decay from the peak, turning the pre-registered cosine into a sawtooth. That failure is
silent everywhere else -- the job completes, the loss curve looks fine -- so a checker
that passed vacuously would be worse than no checker at all, because it would be read as
evidence.

So the tests here are mostly *negative*: they build event files with each defect
deliberately present and assert the script fails on them. The one that matters most is
`test_a_sawtooth_resume_fails` -- if that ever goes green while the script prints PASS on
a real run, the real run's PASS means nothing.

TFRecord framing is `test_stage2_train_cost._write_events`; the parsing under test is
covered there, not re-tested here.
"""
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import check_lr_schedule_135 as clr  # noqa: E402
from test_stage2_train_cost import _event, _write_events  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts", "analysis", "check_lr_schedule_135.py")

TOTAL = clr.RUNG_TOTAL_STEPS
PEAK = clr.RUNG_PEAK_LR


def _cosine(step):
    """The rung's rate at an absolute step index -- written out rather than imported.

    This is the one place a reimplementation is worth having: it is what makes the
    fixtures independent of the function the script lifts out of train.py.
    """
    return PEAK * 0.5 * (1.0 + math.cos(math.pi * step / TOTAL))


def _lr_events(steps, lr_of_step=_cosine, wall0=1000.0):
    """Events as train.py writes them: `lr_at_step(n)` logged under step `n + 1`."""
    return [_event(wall0 + i, n + clr.LOG_STEP_OFFSET, clr.LR_TAG, lr_of_step(n))
            for i, n in enumerate(steps)]


def _run(events_dir):
    return subprocess.run([sys.executable, SCRIPT, "--events-dir", str(events_dir)],
                          capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# the healthy case
# --------------------------------------------------------------------------- #
def test_a_single_incarnation_passes(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", _lr_events(range(0, 4003)))
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout
    assert "has not been requeued" in result.stdout


def test_a_clean_resume_passes(tmp_path):
    # Preempted after logging through step 4003; checkpoint was at 4000, so the second
    # incarnation rewinds and re-runs 4000-4002 before going on.
    _write_events(tmp_path / "events.out.tfevents.1000", _lr_events(range(0, 4003)))
    _write_events(tmp_path / "events.out.tfevents.2000", _lr_events(range(4000, 8000)))
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout
    assert "2 incarnation(s)" in result.stdout
    # The rewind and overlap are reported, not silently merged away.
    assert "rewound 3 step(s), 3 overlapping" in result.stdout


# --------------------------------------------------------------------------- #
# the failure this exists for
# --------------------------------------------------------------------------- #
def test_a_sawtooth_resume_fails(tmp_path):
    """A stateful scheduler restarting its decay from the peak on requeue.

    Note what makes this hard to see by eye: 4,000 steps into a 75,024-step cosine the
    rate has fallen only 0.7%, so the sawtooth's teeth are tiny. It is caught because the
    check is a ratio against the peak and a monotonicity test, not an eyeball.
    """
    _write_events(tmp_path / "events.out.tfevents.1000", _lr_events(range(0, 4003)))
    # Second incarnation logs under the right absolute steps but computes the rate from a
    # scheduler that thinks it is at step 0 again.
    restarted = [_event(2000.0 + i, n + clr.LOG_STEP_OFFSET, clr.LR_TAG, _cosine(n - 4000))
                 for i, n in enumerate(range(4000, 8000))]
    _write_events(tmp_path / "events.out.tfevents.2000", restarted)

    result = _run(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "FAIL" in result.stdout
    assert "restarted rather than continuing" in result.stdout
    # and it is caught twice over -- by the boundary ratio and by monotonicity
    assert "Non-decreasing violations: 0" not in result.stdout


def test_disagreeing_incarnations_fail(tmp_path):
    """The same step index must give the same rate, whichever incarnation ran it."""
    _write_events(tmp_path / "events.out.tfevents.1000", _lr_events(range(0, 4003)))
    _write_events(tmp_path / "events.out.tfevents.2000",
                  _lr_events(range(4000, 8000), lr_of_step=lambda n: _cosine(n) * 0.9))
    result = _run(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "not a function of the step index" in result.stdout


def test_a_wrong_step_offset_is_named_not_just_flagged(tmp_path):
    """A whole-series shift is a logging-convention change, and the script says so."""
    shifted = [_event(1000.0 + i, n, clr.LR_TAG, _cosine(n))       # logged at n, not n+1
               for i, n in enumerate(range(1, 4003))]
    _write_events(tmp_path / "events.out.tfevents.1000", shifted)
    result = _run(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "logging step convention" in result.stdout


def test_an_empty_directory_refuses(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000",
                  [_event(1.0, 1, "Loss/train_step", 0.5)])
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "no 'LR' scalars" in result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# the lift out of train.py
# --------------------------------------------------------------------------- #
def test_the_lifted_function_is_the_one_train_py_will_apply(tmp_path):
    lr_at_step = clr.load_lr_at_step()
    for step in (0, 1, 4000, TOTAL // 2, TOTAL):
        assert lr_at_step(step, TOTAL, PEAK, "cosine") == _cosine(step)
    # and the default is still the paper recipe
    assert lr_at_step(4000, TOTAL, PEAK) == PEAK
