"""The compute half of what an experiment cost (#143).

The fixtures below are shaped like the two Tillicum jobs whose cost is already
written down in docs/tillicum.md, so the parser is checked against numbers that
were arrived at independently of it rather than against itself.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from slurm_usage import (  # noqa: E402
    gpus_from_tres, is_terminal, latest_rows, new_rows, parse_sacct, print_by_name,
    row_key, sacct_command, summarize, SACCT_FIELDS, COLUMNS,
)
from rampnet import ledger  # noqa: E402


def _line(job_id, name, cluster, part, qos, state, start, end, elapsed, tres,
          nodes=1, exit_code="0:0", submit="2026-07-31T00:00:00"):
    return "|".join([job_id, name, cluster, part, qos, state, submit, start, end,
                     str(elapsed), tres, str(nodes), exit_code])


# 4.67 GPU-hours on one H200 at `normal` = the $4.20 data-prep job in docs/tillicum.md.
PREP = _line("198910", "yolo_data_prep", "tillicum", "", "normal", "COMPLETED",
             "2026-07-31T01:00:00", "2026-07-31T05:40:12", 16812,
             "billing=8,cpu=8,gres/gpu=1,mem=200G,node=1")
# The 2-minute smoke job on `debug`, which Slurm bills at UsageFactor 0.
SMOKE = _line("198638", "tillicum_smoke", "tillicum", "", "debug", "COMPLETED",
              "2026-07-31T00:10:00", "2026-07-31T00:12:00", 120,
              "billing=8,cpu=8,gres/gpu=1,mem=100G,node=1")


def test_gpu_hours_and_dollars_match_the_documented_tillicum_job():
    """docs/tillicum.md: 'It cost $4.20 — 4.67 GPU-hours at normal QoS.'"""
    rec, = parse_sacct(PREP)
    assert rec["gpus"] == 1
    assert rec["gpu_hours"] == pytest.approx(4.67, abs=0.005)
    assert rec["est_cost_usd"] == pytest.approx(4.20, abs=0.005)
    assert rec["cluster"] == "tillicum" and rec["qos"] == "normal"


def test_the_debug_qos_is_priced_at_slurms_usage_factor_not_wall_clock():
    """`debug` is UsageFactor 0.0 in `sacctmgr show qos`. hyakusage disagrees and
    charged this job $0.03; that conflict is unresolved and recorded in
    pricing.py's note, so a reader is never handed a bare 'free' with no source."""
    rec, = parse_sacct(SMOKE)
    assert rec["est_cost_usd"] == 0.0
    # The row carries the rate it was priced at and when that was checked; the
    # caveat itself lives in the versioned table, not repeated in every row.
    assert rec["rate_usd_per_gpu_hour"] == 0.90 and rec["rate_as_of"] == "2026-07-30"
    from pricing import compute_price_for
    assert "hyakusage" in compute_price_for("tillicum")["note"]


def test_gpu_hours_multiply_by_gpu_count():
    """Tillicum's unit is elapsed x N GPUs — an idle GPU in a 2-GPU job bills
    exactly like a busy one, which is the whole 2-GPU trap in docs/tillicum.md."""
    two = _line("1", "two_gpu", "tillicum", "", "normal", "COMPLETED",
                "2026-08-01T00:00:00", "2026-08-01T01:00:00", 3600,
                "billing=16,cpu=16,gres/gpu=2,mem=400G,node=1")
    rec, = parse_sacct(two)
    assert rec["gpu_hours"] == 2.0 and rec["est_cost_usd"] == pytest.approx(1.80)


def test_typed_and_generic_gpu_tres_are_not_double_counted():
    """Slurm reports the same GPUs twice, generically and by type. Summing both
    would double every GPU-hour in the ledger."""
    assert gpus_from_tres("cpu=8,gres/gpu=2,gres/gpu:a40=2,mem=200G") == (2, "a40")
    assert gpus_from_tres("cpu=8,gres/gpu:l40s=4,mem=200G") == (4, "l40s")
    assert gpus_from_tres("cpu=8,mem=200G") == (0, None)
    assert gpus_from_tres(None) == (0, None)
    assert gpus_from_tres("cpu=8,gres/gpu:a40=1,gres/gpu:l40s=1") == (2, "a40,l40s")


def test_a_cpu_only_klone_job_is_free_but_still_recorded():
    """klone is free at the point of use; the GPU-hours are still a real cost of
    the science and the #51 baseline's 496.5 of them belong in the paper."""
    rec, = parse_sacct(_line("42", "prep", "klone", "ckpt-all", "ckpt", "COMPLETED",
                             "2026-07-24T00:00:00", "2026-07-24T02:00:00", 7200,
                             "cpu=12,mem=100G,node=1"))
    assert rec["gpus"] == 0 and rec["gpu_hours"] == 0.0
    assert rec["est_cost_usd"] == 0.0  # priced at zero, not unpriced


def test_an_unpriced_cluster_is_visibly_unpriced_not_silently_free():
    rec, = parse_sacct(_line("7", "j", "somewhere_else", "gpu", "normal", "COMPLETED",
                             "2026-08-01T00:00:00", "2026-08-01T01:00:00", 3600,
                             "gres/gpu=1"))
    assert rec["est_cost_usd"] is None and rec["rate_usd_per_gpu_hour"] is None
    assert summarize([rec])["somewhere_else"]["unpriced"] == 1


def test_overriding_the_cluster_name_says_so(capsys):
    """A dump can legitimately span clusters, and restamping one silently would
    price its jobs at the wrong rate and attribute its GPU-hours to the wrong
    machine. Nothing downstream can detect that, so it has to be loud here."""
    rows = parse_sacct(PREP, cluster="klone")
    assert rows[0]["cluster"] == "klone" and rows[0]["rate_usd_per_gpu_hour"] == 0.0
    out = capsys.readouterr().out
    assert "WARNING" in out and "tillicum" in out
    # ...and no warning when the override agrees with sacct.
    parse_sacct(PREP, cluster="tillicum")
    assert "WARNING" not in capsys.readouterr().out


def test_lines_that_are_not_job_records_are_skipped():
    text = "\n".join(["", "sacct: warning: something", "a|b|c", PREP])
    assert len(parse_sacct(text)) == 1


def test_requeued_incarnations_are_separate_rows_not_one():
    """klone's ckpt partition preempts and requeues: the paper's Stage 2 run was 15
    preemptions. sacct -D reports each incarnation under the SAME job id, so keying
    on the id alone would throw away all but the last one's compute."""
    a = _line("999", "train", "klone", "ckpt-all", "ckpt", "PREEMPTED",
              "2026-08-01T00:00:00", "2026-08-01T03:00:00", 10800, "gres/gpu=4")
    b = _line("999", "train", "klone", "ckpt-all", "ckpt", "COMPLETED",
              "2026-08-01T04:00:00", "2026-08-01T09:00:00", 18000, "gres/gpu=4")
    rows = parse_sacct(a + "\n" + b)
    assert len({row_key(r) for r in rows}) == 2
    assert summarize(rows)["klone"]["gpu_hours"] == pytest.approx(32.0)


def test_appending_twice_does_not_double_count():
    """Someone will run this again next week. It has to add what is new and
    nothing else, or the ledger inflates every time anyone checks it."""
    rows = parse_sacct(PREP + "\n" + SMOKE)
    assert len(new_rows(rows, [])) == 2
    assert new_rows(rows, rows) == []


def test_a_job_recorded_while_running_is_re_recorded_once_it_finishes():
    """Its elapsed time was still growing when we first saw it, so the first row
    understates the cost. Readers take the last row per key."""
    running = parse_sacct(_line("500", "train", "tillicum", "", "normal", "RUNNING",
                                "2026-08-01T00:00:00", "Unknown", 3600, "gres/gpu=1"))
    done = parse_sacct(_line("500", "train", "tillicum", "", "normal", "COMPLETED",
                             "2026-08-01T00:00:00", "2026-08-01T05:00:00", 18000,
                             "gres/gpu=1"))
    assert len(new_rows(done, running)) == 1
    assert new_rows(done, done) == []          # ...but only once


def test_a_re_recorded_job_is_counted_once_not_twice(capsys):
    """The other half of re-appending: the ledger then holds both the RUNNING row
    and the finished one, and totalling every row bills the job for both. 5.0
    GPU-hours, not 6.0."""
    running = parse_sacct(_line("500", "train", "tillicum", "", "normal", "RUNNING",
                                "2026-08-01T00:00:00", "Unknown", 3600, "gres/gpu=1"))
    done = parse_sacct(_line("500", "train", "tillicum", "", "normal", "COMPLETED",
                             "2026-08-01T00:00:00", "2026-08-01T05:00:00", 18000,
                             "gres/gpu=1"))
    ledger_rows = running + new_rows(done, running)
    assert len(ledger_rows) == 2
    assert [r["state"] for r in latest_rows(ledger_rows)] == ["COMPLETED"]
    agg = summarize(ledger_rows)["tillicum"]
    assert agg["jobs"] == 1
    assert agg["gpu_hours"] == pytest.approx(5.0)
    assert agg["usd"] == pytest.approx(4.50)
    print_by_name(ledger_rows)
    assert "5.0" in capsys.readouterr().out


def test_terminal_states_include_the_ones_with_a_suffix():
    assert is_terminal("CANCELLED by 12345") and is_terminal("COMPLETED")
    assert is_terminal("PREEMPTED") and not is_terminal("RUNNING")
    assert not is_terminal("PENDING") and not is_terminal("")
    # A requeued incarnation is finished: it has an End time and its elapsed will
    # not grow, and the next incarnation carries a different start.
    assert is_terminal("REQUEUED")


def test_the_sacct_command_asks_for_duplicates_and_the_pinned_columns():
    """-D is not optional: without it Slurm reports only the last incarnation of a
    requeued job, and most of a preempted run's compute silently disappears."""
    cmd = sacct_command("jfroehli", "2026-07-01")
    assert "-D" in cmd and "-X" in cmd and "-P" in cmd
    assert "--format=" + ",".join(SACCT_FIELDS) in cmd
    # The parser indexes by position, so the two lists must stay in step.
    assert len(COLUMNS) == len(SACCT_FIELDS)


def test_the_ledger_round_trips_through_the_shared_writer(tmp_path):
    log = tmp_path / "compute_log.jsonl"
    ledger.append_rows(str(log), parse_sacct(PREP + "\n" + SMOKE))
    assert log.read_bytes().count(b"\r\n") == 0        # LF, on Windows too
    back = ledger.read_rows(str(log))
    assert [r["job_id"] for r in back] == ["198910", "198638"]
    # Both ledgers name the duration the same, so one reader can total them.
    rows, usd, hours, recovered = ledger.ledger_totals(str(log))
    assert rows == 2 and usd == pytest.approx(4.20, abs=0.005)
    assert hours == pytest.approx((16812 + 120) / 3600.0)
    assert recovered == 0        # sacct rows are measured, never reconstructed


def test_the_compute_ledger_is_re_included_in_gitignore():
    """analysis_out/* is ignored wholesale, so a new committed artifact under it
    needs an explicit re-include or it is silently never committed — which is the
    class of failure this whole issue is about."""
    with open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8") as fh:
        assert "!analysis_out/compute_log.jsonl" in fh.read()
