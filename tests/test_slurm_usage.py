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

import slurm_usage  # noqa: E402
from slurm_usage import (  # noqa: E402
    gpu_hours_as_of, gpus_from_tres, is_terminal, latest_rows, new_rows, parse_sacct,
    print_by_name, row_key, sacct_command, summarize, SACCT_FIELDS, COLUMNS,
)
import gpu_hours_as_of as as_of_script  # noqa: E402
from rampnet import ledger  # noqa: E402

KLONE_DUMP = os.path.join(REPO_ROOT, "docs", "data", "compute", "sacct_klone_2026-08-19.txt")


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


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["slurm_usage.py", *argv])
    return slurm_usage.main()


def test_a_dump_must_name_the_account_it_came_from(tmp_path, monkeypatch, capsys):
    """The committed klone dump is jfroehli's. Parsed on a laptop whose login is
    jonf, the old default stamped every row -- and the printed regeneration
    command -- with jonf. Nothing in a dump says whose it is, so the caller has to."""
    dump = tmp_path / "sacct.txt"
    dump.write_text(PREP + "\n", encoding="utf-8")
    monkeypatch.setenv("USER", "jonf")
    monkeypatch.setenv("USERNAME", "jonf")
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, "--from-file", str(dump), "--dry-run")
    assert exc.value.code == 2
    assert "--user is required with --from-file" in capsys.readouterr().err
    # With the account named, the rows carry it and not the local login.
    out = tmp_path / "compute_log.jsonl"
    assert _run_main(monkeypatch, "--from-file", str(dump), "--user", "jfroehli",
                     "--out", str(out)) == 0
    assert [r["user"] for r in ledger.read_rows(str(out))] == ["jfroehli"]


def test_print_command_never_substitutes_the_local_login(monkeypatch, capsys):
    """The printed command is run ON the cluster, where sacct defaults to the
    caller; -u <laptop login> there is the wrong account or no account."""
    monkeypatch.setenv("USER", "jonf")
    monkeypatch.setenv("USERNAME", "jonf")
    assert _run_main(monkeypatch, "--print-command") == 0
    printed = capsys.readouterr().out
    assert "jonf" not in printed and " -u " not in printed
    assert _run_main(monkeypatch, "--print-command", "--user", "jfroehli") == 0
    assert " -u jfroehli " in capsys.readouterr().out


def test_no_user_means_sacct_default_not_a_none_in_argv():
    """With $USER unset and no --user, the live path used to put None in the argv
    list and die inside subprocess.run with a TypeError nothing caught."""
    cmd = sacct_command(None, "2026-07-01")
    assert "-u" not in cmd and all(isinstance(a, str) for a in cmd)
    assert "-D" in cmd


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


def test_the_shared_reader_counts_a_re_recorded_job_once(tmp_path):
    """The other half of 'one reader can total both ledgers': the compute ledger
    holds a RUNNING row and its finished replacement for any job that spanned two
    pulls, and ledger_totals used to add them -- 6.0 h and $5.40 for a 5.0 h job.
    The moment a second pull landed, the number compare.py prints after every leg
    would have been wrong for every job that was in flight at the first one."""
    running = parse_sacct(_line("500", "train", "tillicum", "", "normal", "RUNNING",
                                "2026-08-01T00:00:00", "Unknown", 3600, "gres/gpu=1"))
    done = parse_sacct(_line("500", "train", "tillicum", "", "normal", "COMPLETED",
                             "2026-08-01T00:00:00", "2026-08-01T05:00:00", 18000,
                             "gres/gpu=1"))
    log = tmp_path / "compute_log.jsonl"
    ledger.append_rows(str(log), running)
    ledger.append_rows(str(log), new_rows(done, running))
    assert len(ledger.read_rows(str(log))) == 2          # both rows are on disk...
    rows, usd, hours, _ = ledger.ledger_totals(str(log))
    assert rows == 1                                     # ...one job is counted
    assert hours == pytest.approx(5.0) and usd == pytest.approx(4.50)


def test_api_rows_are_never_collapsed_by_the_shared_reader(tmp_path):
    """usage_log.jsonl rows carry no job_id: two legs on the same bundle with the
    same model are two spends, and nothing about them is a re-record."""
    leg = {"ts": "2026-08-18T14:54:02+00:00", "bundle": "annapolis",
           "model_id": "claude-sonnet-5", "est_cost_usd": 1.5, "elapsed_s": 1800}
    log = tmp_path / "usage_log.jsonl"
    ledger.append_rows(str(log), [leg, dict(leg)])
    rows, usd, hours, _ = ledger.ledger_totals(str(log))
    assert rows == 2 and usd == pytest.approx(3.0) and hours == pytest.approx(1.0)
    assert ledger.latest_rows([leg, dict(leg)]) == [leg, leg]


def test_a_pending_row_is_not_a_job_allocation():
    """A PENDING record has no start and no elapsed, and its key (cluster, id,
    Unknown) is never superseded once the job starts under a real start -- so it
    would sit in the ledger for good as a zero-hour job. The 2026-08-19 klone pull
    had one (38640313, the #135 cosine rung, then 6 h into a requeue wait)."""
    pending = _line("38640313", "rampnet_cosine_rung_135", "klone", "ckpt-all",
                    "ckpt-gpu", "PENDING", "Unknown", "Unknown", 0, "gres/gpu=8")
    started = _line("38640313", "rampnet_cosine_rung_135", "klone", "ckpt-all",
                    "ckpt-gpu", "RUNNING", "2026-08-20T01:00:00", "Unknown", 3600,
                    "gres/gpu=8")
    assert parse_sacct(pending) == []
    assert len(parse_sacct(pending + "\n" + started)) == 1
    # A job cancelled before it ever started is terminal: a final, zero-hour
    # record with a stable key, and it stays.
    never_ran = _line("7", "j", "klone", "ckpt-all", "ckpt", "CANCELLED by 1",
                      "None", "2026-08-01T00:00:00", 0, "gres/gpu=1")
    assert parse_sacct(never_ran)[0]["gpu_hours"] == 0.0


def test_the_committed_ledger_is_exactly_what_the_committed_dump_parses_to():
    """docs/compute_cost.md's numbers are claimed re-derivable from a clean clone.
    That is only true if the ledger is the dump's parse and nothing else: same
    rows, same order, differing only in the recorded_at stamp."""
    with open(KLONE_DUMP, encoding="utf-8") as fh:
        parsed = parse_sacct(fh.read(), cluster="klone", user="jfroehli")
    committed = ledger.read_rows(os.path.join(REPO_ROOT, "analysis_out",
                                              "compute_log.jsonl"))
    assert len(committed) == len(parsed) == 3990
    for have, want in zip(committed, parsed):
        have = dict(have)
        assert have.pop("recorded_at").startswith("2026-08-19T")
        assert have == want
    # ...and the headline figures in the doc, from the ledger as committed.
    agg = summarize(committed)["klone"]
    assert agg["jobs"] == 3990 and round(agg["gpu_hours"], 1) == 2684.4
    assert agg["usd"] == 0.0 and agg["unpriced"] == 0
    base = [r for r in committed if r["job_name"] == "yolo_curb_ramp_train"]
    assert len(base) == 3857 and round(sum(r["gpu_hours"] for r in base), 1) == 2046.9
    assert len({r["job_id"] for r in base}) == 27
    assert sum(r["state"].startswith("PREEMPTED") for r in committed) == 3780
    assert sum(r["state"] == "REQUEUED" for r in committed) == 59
    running = [r for r in committed if r["state"] == "RUNNING"]
    assert len(running) == 3 and round(sum(r["gpu_hours"] for r in running), 1) == 158.0


def test_the_compute_ledger_is_re_included_in_gitignore():
    """analysis_out/* is ignored wholesale, so a new committed artifact under it
    needs an explicit re-include or it is silently never committed — which is the
    class of failure this whole issue is about."""
    with open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8") as fh:
        assert "!analysis_out/compute_log.jsonl" in fh.read()


def test_a_snapshot_counts_elapsed_so_far_not_finished_jobs():
    """A figure copied from a live sacct counts a running job's hours up to that
    moment. A by-end running sum counts nothing of it until it ends -- which is how
    the 496.5 check was first done, and why it landed on the wrong instant."""
    from datetime import datetime
    rows = parse_sacct("\n".join([
        # 4 GPUs, started 2 h before the query and ran 3 h more: 8 GPU-h so far.
        _line("1", "train", "klone", "ckpt-all", "ckpt", "PREEMPTED",
              "2026-07-30T05:00:00", "2026-07-30T10:00:00", 18000, "gres/gpu=4"),
        # Ended before the window: sacct -S excludes it entirely.
        _line("2", "train", "klone", "ckpt-all", "ckpt", "COMPLETED",
              "2026-07-20T00:00:00", "2026-07-23T00:00:00", 259200, "gres/gpu=4"),
        # Alive across the window's start: counted in full, as -S does.
        _line("3", "train", "klone", "ckpt-all", "ckpt", "COMPLETED",
              "2026-07-23T23:00:00", "2026-07-24T01:00:00", 7200, "gres/gpu=1"),
        # Not started yet at query time.
        _line("4", "train", "klone", "ckpt-all", "ckpt", "COMPLETED",
              "2026-07-30T08:00:00", "2026-07-30T09:00:00", 3600, "gres/gpu=1"),
        # Another job name: out of scope when one is asked for.
        _line("5", "other", "klone", "ckpt-all", "ckpt", "COMPLETED",
              "2026-07-29T00:00:00", "2026-07-29T01:00:00", 3600, "gres/gpu=1"),
    ]))
    at, since = datetime(2026, 7, 30, 7), datetime(2026, 7, 24)
    assert gpu_hours_as_of(rows, at, since, "train") == (pytest.approx(10.0), 2)
    assert gpu_hours_as_of(rows, at, since) == (pytest.approx(11.0), 3)
    # No -S: the 288 GPU-h job that ended before the window counts too.
    assert gpu_hours_as_of(rows, at, None, "train") == (pytest.approx(298.0), 3)


def test_the_tillicum_496_5_reproduces_from_the_committed_dump_as_a_baseline_snapshot():
    """docs/compute_cost.md: the baseline-only snapshot at 07:00 on 2026-07-30 is
    497.5 GPU-hours (all job names: 553.2; last incarnation per job id: 85.8), and
    the by-end running sum crosses 496.5 only at 11:03 that day. Pinned so the doc's
    numbers cannot drift from the script that produces them."""
    from datetime import datetime
    with open(KLONE_DUMP, encoding="utf-8") as fh:
        rows = parse_sacct(fh.read(), cluster="klone", user="jfroehli")
    at, since = datetime(2026, 7, 30, 7), datetime(2026, 7, 24)
    hours, n = gpu_hours_as_of(rows, at, since, "yolo_curb_ramp_train")
    assert round(hours, 1) == 497.5 and n == 400
    assert round(gpu_hours_as_of(rows, at, since)[0], 1) == 553.2
    # 496.5 itself is passed at about 06:50, before the 07:07 commit that wrote it.
    assert gpu_hours_as_of(rows, datetime(2026, 7, 30, 6, 50), since,
                           "yolo_curb_ramp_train")[0] > 496.5
    assert gpu_hours_as_of(rows, datetime(2026, 7, 30, 6, 45), since,
                           "yolo_curb_ramp_train")[0] < 496.5
    # Without -D: the last incarnation per job id that had started by then.
    last = {}
    for r in sorted(rows, key=lambda r: r["start"]):
        if r["job_name"] == "yolo_curb_ramp_train" and r["start"] < "2026-07-30T07":
            last[r["job_id"]] = r
    assert round(gpu_hours_as_of(list(last.values()), at, since)[0], 1) == 85.8
    # The by-end sum the doc warns against: baseline crosses at 11:03, not 21:17.
    total, crossed = 0.0, None
    for r in sorted((r for r in rows if r["job_name"] == "yolo_curb_ramp_train"
                     and r["end"] >= "2026-07-24"), key=lambda r: r["end"]):
        total += r["gpu_hours"]
        if total >= 496.5:
            crossed = r["end"]
            break
    assert crossed == "2026-07-30T11:03:20"


def test_the_snapshot_script_prints_the_documented_figure(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "gpu_hours_as_of.py", "--from-file", KLONE_DUMP, "--since", "2026-07-24",
        "--at", "2026-07-30T07:00", "--job-name", "yolo_curb_ramp_train"])
    assert as_of_script.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("497.5 GPU-hours as of 2026-07-30T07:00:00: 400 incarnation(s)")
