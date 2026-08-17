"""Unit tests for the Stage 2 training-cost measurement (#59, #84).

The script reads TensorBoard event files with a hand-written TFRecord + protobuf
reader, so that it needs neither TensorFlow nor the network. That reader is the risk:
a wire-format slip would not raise, it would silently yield a different set of scalars
and a different s/step — and every cost number in docs/stage2_training_cost.md is
derived from that one figure. So the tests encode events themselves, check the framing
edge cases (truncated tail, header-only file), and then re-derive the published numbers
from the committed run.
"""
import os
import struct
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage2_train_cost as tc  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts", "analysis", "stage2_train_cost.py")
EVENTS = os.path.join(REPO, "docs", "data", "rampnet1_stage2_run")


# --------------------------------------------------------------------------- #
# a minimal protobuf/TFRecord writer, so the tests do not depend on tensorflow
# --------------------------------------------------------------------------- #
def _uvarint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(fno, wire):
    return _uvarint((fno << 3) | wire)


def _len_field(fno, data):
    return _key(fno, 2) + _uvarint(len(data)) + data


def _event(wall, step, tag, value):
    """Event{wall_time=1, step=2, summary=5{value=1{tag=1, simple_value=2}}}."""
    val = _len_field(1, tag.encode("utf8")) + _key(2, 5) + struct.pack("<f", value)
    summary = _len_field(1, val)
    return (_key(1, 1) + struct.pack("<d", wall)
            + _key(2, 0) + _uvarint(step)
            + _len_field(5, summary))


def _framed(payload):
    """uint64 length + masked crc32 + payload + masked crc32. CRCs are not checked."""
    return struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0"


def _write_events(path, events):
    with open(path, "wb") as fh:
        # Real files open with a file_version header event carrying no summary.
        fh.write(_framed(_key(1, 1) + struct.pack("<d", 0.0)
                         + _len_field(3, b"brain.Event:2")))
        for e in events:
            fh.write(_framed(e))
    return path


def _run(*argv):
    return subprocess.run([sys.executable, SCRIPT, *argv], capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# varint / field decoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 127, 128, 300, 9378, 112_434, 2**35])
def test_varint_round_trip(n):
    assert tc._varint(_uvarint(n), 0) == (n, len(_uvarint(n)))


def test_fields_reads_each_wire_type():
    buf = (_key(1, 0) + _uvarint(7)
           + _key(2, 1) + struct.pack("<d", 1.5)
           + _key(3, 2) + _uvarint(3) + b"abc"
           + _key(4, 5) + struct.pack("<f", 2.5))
    got = list(tc._fields(buf))
    assert got[0] == (1, 0, 7)
    assert struct.unpack("<d", got[1][2]) == (1.5,)
    assert got[2] == (3, 2, b"abc")
    assert struct.unpack("<f", got[3][2]) == (2.5,)


def test_fields_stops_on_an_unknown_wire_type():
    # Wire type 3/4 (deprecated groups) never appear in Event; bailing out beats
    # walking off into the middle of a record.
    assert list(tc._fields(_key(1, 3) + b"\x00")) == []


# --------------------------------------------------------------------------- #
# TFRecord framing
# --------------------------------------------------------------------------- #
def test_records_round_trip(tmp_path):
    p = tmp_path / "events.out.tfevents.1"
    with open(p, "wb") as fh:
        fh.write(_framed(b"one") + _framed(b"two"))
    assert list(tc._records(p)) == [b"one", b"two"]


def test_records_ignores_a_truncated_tail(tmp_path):
    """A preempted job dies mid-write; the last frame must not poison the file."""
    p = tmp_path / "events.out.tfevents.1"
    with open(p, "wb") as fh:
        fh.write(_framed(b"complete"))
        fh.write(struct.pack("<Q", 999) + b"\0\0\0\0" + b"partial")
    assert list(tc._records(p)) == [b"complete"]


def test_read_scalars_extracts_tag_step_and_value(tmp_path):
    p = _write_events(tmp_path / "events.out.tfevents.1",
                      [_event(100.0, 1, "Loss/train_step", 0.5),
                       _event(101.5, 2, "Loss/train_step", 0.25),
                       _event(102.0, 9378, "Loss/val_epoch", 0.00052)])
    got = tc.read_scalars(p)
    assert [(s, t) for _, s, t, _ in got] == [
        (1, "Loss/train_step"), (2, "Loss/train_step"), (9378, "Loss/val_epoch")]
    assert got[1][0] == pytest.approx(101.5)
    assert got[2][3] == pytest.approx(0.00052, rel=1e-5)


def test_read_scalars_skips_the_header_event(tmp_path):
    p = _write_events(tmp_path / "events.out.tfevents.1", [])
    assert tc.read_scalars(p) == []


# --------------------------------------------------------------------------- #
# end to end on a synthetic run
# --------------------------------------------------------------------------- #
def test_preemption_gaps_are_excluded_from_the_step_time(tmp_path):
    # Ten steps 2 s apart, with one 1-hour hole where the job was requeued. The
    # median must be the step time, not something averaged with the outage.
    walls, events = 0.0, []
    for i in range(1, 11):
        walls += 3600.0 if i == 6 else 2.0
        events.append(_event(walls, i, "Loss/train_step", 0.1))
    _write_events(tmp_path / "events.out.tfevents.1", events)

    r = _run("--events-dir", str(tmp_path), "--train-panos", "160", "--world-size", "16")
    assert r.returncode == 0, r.stderr
    assert "median s/step (rank 0) : 2.000" in r.stdout
    assert "9,378" not in r.stdout            # steps/epoch comes from the args
    assert "10 steps x 2.000 s" in r.stdout


def test_no_training_scalars_is_a_message_not_a_traceback(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1",
                  [_event(1.0, 1, "Loss/val_epoch", 0.1)])
    r = _run("--events-dir", str(tmp_path))
    assert r.returncode != 0
    assert "no usable step gaps" in (r.stdout + r.stderr)
    assert "Traceback" not in r.stderr


def test_empty_directory_is_a_message(tmp_path):
    r = _run("--events-dir", str(tmp_path))
    assert r.returncode != 0
    assert "no tfevents files" in (r.stdout + r.stderr)


def test_single_segment_reports_a_median_without_quartiles(tmp_path):
    _write_events(tmp_path / "events.out.tfevents.1",
                  [_event(0.0, 1, "Loss/train_step", 0.1),
                   _event(3.0, 2, "Loss/train_step", 0.1)])
    r = _run("--events-dir", str(tmp_path), "--train-panos", "32", "--world-size", "16")
    assert r.returncode == 0, r.stderr
    assert "median s/step (rank 0) : 3.000  [n=1]" in r.stdout


def test_hms_switches_to_days_past_two():
    assert tc.hms(3.49) == "3.49 h"
    assert tc.hms(74.6) == "74.6 h (3.1 d)"


# --------------------------------------------------------------------------- #
# the committed run still says what the doc says
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(EVENTS), reason="stage 2 telemetry not present")
def test_committed_telemetry_reproduces_the_published_cost():
    r = _run()
    assert r.returncode == 0, r.stderr
    # 1.341 s/step median -> 9,378 steps = 3.49 h/epoch on 16 GPUs (~56 GPU-h), and
    # the run reached 11.99 epochs. These are the numbers docs/stage2_training_cost.md
    # is written from.
    assert "median s/step (rank 0) : 1.341" in r.stdout
    assert "max global_step        : 112,434 = 11.99 epochs" in r.stdout
    assert "9,378 steps x 1.341 s = 3.49 h  (56 GPU-h)" in r.stdout
    # Selection on auto-label val loss would have picked epoch 5, not the released epoch 1.
    assert "epoch  5.00  step  46,890  val 0.000458  <- best" in r.stdout
