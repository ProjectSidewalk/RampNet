"""Unit tests for the Stage 2 epoch-curve extraction (#84).

The risky parts of this script are not the arithmetic, they are the three places where a
silent wrong answer is possible:

- **step -> epoch.** A hardcoded steps/epoch that does not match the run would relabel
  every row with plausible-looking wrong epoch numbers. It must refuse, not round.
- **resume dedupe.** A requeued run re-emits steps it already wrote; the later file has to
  win, or the curve reports a value the run did not finish with.
- **integrity.** The event files are checksummed replication artifacts. A mismatch has to
  stop the run, otherwise "the script prints a sha256" proves nothing.

The TFRecord/protobuf reading itself is `stage2_train_cost.read_scalars` and is covered by
tests/test_stage2_train_cost.py -- deliberately not re-tested here, because there is only
one copy of it.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage2_epoch_curve as ec  # noqa: E402
from test_stage2_train_cost import _event, _write_events  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts", "analysis", "stage2_epoch_curve.py")
RUN_A = os.path.join(REPO, "stage_two", "run_a_84_events")
PAPER = os.path.join(REPO, "docs", "data", "rampnet1_stage2_run")

SPE = ec.STEPS_PER_EPOCH


def _run(*argv):
    return subprocess.run([sys.executable, SCRIPT, *argv], capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# step -> epoch: refuse rather than round
# --------------------------------------------------------------------------- #
def test_epoch_of_maps_exact_multiples():
    assert ec.epoch_of(SPE) == 1
    assert ec.epoch_of(5 * SPE) == 5


def test_epoch_of_refuses_a_step_off_the_grid():
    # The old code did round(step / STEPS_PER_EPOCH), which silently turns a run with a
    # different train-set size or world size into confidently mislabelled epochs.
    with pytest.raises(SystemExit) as excinfo:
        ec.epoch_of(5 * SPE + 17)
    assert "not a multiple" in str(excinfo.value)


def test_epoch_of_honours_a_different_steps_per_epoch():
    assert ec.epoch_of(400, steps_per_epoch=100) == 4


def test_epoch_curve_is_one_row_per_epoch():
    curve = ec.epoch_curve({SPE: 0.5, 2 * SPE: 0.4}, SPE)
    assert curve == {1: 0.5, 2: 0.4}


# --------------------------------------------------------------------------- #
# resume dedupe: the later file wins
# --------------------------------------------------------------------------- #
def test_a_resumed_step_takes_the_later_files_value(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.11)])
    _write_events(tmp_path / "events.out.tfevents.2000", [_event(2.0, SPE, ec.VAL_TAG, 0.22)])
    assert ec.read_scalars(tmp_path)[SPE] == pytest.approx(0.22, rel=1e-6)


def test_repeat_measurements_keeps_both_incarnations(tmp_path):
    """The duplicate is the free noise-floor measurement, so it must not be dropped."""
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, 5 * SPE, ec.VAL_TAG, 0.10)])
    _write_events(tmp_path / "events.out.tfevents.2000", [_event(2.0, 5 * SPE, ec.VAL_TAG, 0.11)])
    repeats = ec.repeat_measurements(tmp_path)
    assert list(repeats) == [5]
    assert len(repeats[5]) == 2
    assert ec.spread_pct(repeats[5]) == pytest.approx(10.0, rel=1e-3)


def test_an_epoch_seen_once_is_not_a_repeat(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.10)])
    assert ec.repeat_measurements(tmp_path) == {}


# --------------------------------------------------------------------------- #
# integrity: a manifest that does not match must stop the run
# --------------------------------------------------------------------------- #
def test_a_mismatched_manifest_is_fatal(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.10)])
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  events.out.tfevents.1000\n")
    with pytest.raises(SystemExit) as excinfo:
        ec.read_scalars(tmp_path)
    assert "does not match" in str(excinfo.value)


def test_a_file_listed_but_absent_is_fatal(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.10)])
    digest = ec.read_scalars_by_file(tmp_path, verify=False)[0][1]
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  events.out.tfevents.1000\n{'1' * 64}  events.out.tfevents.9999\n")
    with pytest.raises(SystemExit) as excinfo:
        ec.read_scalars(tmp_path)
    assert "absent" in str(excinfo.value)


def test_a_matching_manifest_passes(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.10)])
    digest = ec.read_scalars_by_file(tmp_path, verify=False)[0][1]
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  events.out.tfevents.1000\n")
    assert ec.read_scalars(tmp_path) == pytest.approx({SPE: 0.10}, rel=1e-6)


def test_no_manifest_is_allowed(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1000", [_event(1.0, SPE, ec.VAL_TAG, 0.10)])
    assert ec.read_manifest(tmp_path) == {}
    assert list(ec.read_scalars(tmp_path)) == [SPE]


# --------------------------------------------------------------------------- #
# the paper column degrades rather than crashing
# --------------------------------------------------------------------------- #
def test_missing_paper_events_give_an_empty_column_not_a_crash(tmp_path):
    assert ec.read_paper_curve(tmp_path / "nope") == {}


def test_empty_directory_is_a_message(tmp_path):
    r = _run("--events-dir", str(tmp_path))
    assert r.returncode != 0
    assert "no event files" in (r.stdout + r.stderr)
    assert "Traceback" not in r.stderr


# --------------------------------------------------------------------------- #
# the committed runs still say what the doc says
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(RUN_A), reason="Run A telemetry not present")
def test_run_a_matches_its_committed_manifest_and_bottoms_at_epoch_5():
    curve = ec.read_curve(ec.RUN_A_EVENTS)          # raises if SHA256SUMS disagrees
    assert sorted(curve) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert min(curve, key=curve.get) == 5
    assert curve[5] == pytest.approx(0.00045976, abs=5e-9)


@pytest.mark.skipif(not os.path.isdir(PAPER), reason="paper-run telemetry not present")
def test_the_paper_curve_is_re_derivable_and_also_bottoms_at_epoch_5():
    """The gap this doc used to claim: the paper events ARE committed (#104)."""
    paper = ec.read_paper_curve()
    assert sorted(paper) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert min(paper, key=paper.get) == 5
    assert paper[5] == pytest.approx(0.00045825, abs=5e-9)


@pytest.mark.skipif(not (os.path.isdir(RUN_A) and os.path.isdir(PAPER)),
                    reason="telemetry not present")
def test_the_published_replication_bound():
    """docs/stage2_epoch_curve_84.md quotes these two numbers; they are derived, not typed."""
    curve, paper = ec.read_curve(ec.RUN_A_EVENTS), ec.read_paper_curve()
    deltas = [abs(curve[e] / paper[e] - 1.0) * 100.0 for e in sorted(curve) if e in paper]
    assert len(deltas) == 8
    assert max(deltas) == pytest.approx(1.755, abs=5e-4)
    assert sum(deltas) / len(deltas) == pytest.approx(0.579, abs=5e-4)


@pytest.mark.skipif(not os.path.isdir(RUN_A), reason="Run A telemetry not present")
def test_the_measurement_floor_is_derived_from_the_requeue():
    repeats = ec.repeat_measurements()
    assert list(repeats) == [5], "a requeue landed mid-epoch-5; only that epoch is doubled"
    assert ec.spread_pct(repeats[5]) == pytest.approx(0.0090, abs=5e-5)
