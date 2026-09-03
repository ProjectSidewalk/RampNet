"""Reconciling the committed ledger against what the provider actually billed (#143).

The numbers in these fixtures are the real #139 incident: a claude-opus-5 leg billed
11,988,993 input tokens (~$70.41) and wrote no row to analysis_out/usage_log.jsonl.
This is the check that would have said so on the day, while the metric was still
inside its ~6-week retention window.

No network and no cloud credentials: only the pure comparison functions are exercised,
because `fetch_token_series` imports google-auth lazily inside itself.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))

from rampnet import ledger  # noqa: E402
from vertex_usage import (  # noqa: E402
    ledger_totals_by_model, print_reconciliation, reconcile,
)

OPUS_BILLED = {"input": 11_988_993, "output": 418_503}
SONNET_BILLED = {"input": 12_594, "output": 480}


def _row(model, ts="2026-08-18T14:54:02+00:00", **kw):
    return dict({"ts": ts, "model_id": model}, **kw)


def test_a_leg_that_billed_and_logged_nothing_is_flagged_missing():
    """The #139 failure exactly: $70.41 of opus with no ledger row. #119's guard
    cannot see it — that one proves a log path was accepted, never that the file it
    wrote survived — so this comparison is the only thing that catches it."""
    logged = ledger_totals_by_model([_row("claude-sonnet-5", input_tokens=12_594,
                                          output_tokens=480)])
    rows = reconcile({"claude-opus-5": OPUS_BILLED,
                      "claude-sonnet-5": SONNET_BILLED}, logged)
    verdicts = {r["model_id"]: r["verdict"] for r in rows}
    assert verdicts["claude-opus-5"].startswith("MISSING")
    assert verdicts["claude-sonnet-5"] == "ok"


def test_a_ledger_that_matches_reconciles_clean():
    logged = ledger_totals_by_model([
        _row("claude-opus-5", input_tokens=11_000_000, output_tokens=400_000),
        _row("claude-opus-5", input_tokens=988_993, output_tokens=18_503),
    ])
    rows = reconcile({"claude-opus-5": OPUS_BILLED}, logged)
    assert rows[0]["verdict"] == "ok" and rows[0]["logged_rows"] == 2


def test_a_short_ledger_is_flagged_under_and_an_over_one_is_not_alarming():
    """Asymmetric on purpose: billed > logged is spend with no record, which is the
    failure this exists for. logged > billed is odd but harmless — a re-run, or a
    row stamped just outside the queried window."""
    under = reconcile({"m": {"input": 1_000_000}},
                      ledger_totals_by_model([_row("m", input_tokens=500_000)]))
    over = reconcile({"m": {"input": 1_000_000}},
                     ledger_totals_by_model([_row("m", input_tokens=1_500_000)]))
    assert under[0]["verdict"].startswith("UNDER")
    assert over[0]["verdict"].startswith("over")


def test_small_drift_is_within_tolerance():
    """Cloud Monitoring's daily rows are 24 h windows ending at query time-of-day,
    not calendar days, so an exact match is not the bar — a leg straddling the
    boundary moves a little either way."""
    rows = reconcile({"m": {"input": 1_000_000}},
                     ledger_totals_by_model([_row("m", input_tokens=990_000)]))
    assert rows[0]["verdict"] == "ok"


def test_free_legs_are_not_reconciled_against_a_bill():
    """A row with no token keys is a GPU leg (#143). It has no bill to compare to,
    and counting it as a model with zero tokens would invent a MISSING verdict."""
    logged = ledger_totals_by_model([_row("owlv2-large-patch14-ensemble", paid=False,
                                          elapsed_s=900.0)])
    assert logged == {}


def test_rows_outside_the_window_are_not_counted_against_it():
    """The ledger is append-only and covers all time; the metric query covers
    --days. Comparing the two without trimming would report a spurious 'over'."""
    rows = [_row("m", ts="2026-06-01T00:00:00+00:00", input_tokens=9_000_000),
            _row("m", ts="2026-08-18T00:00:00+00:00", input_tokens=1_000_000)]
    assert ledger_totals_by_model(rows, since="2026-08-01")["m"]["input"] == 1_000_000
    assert ledger_totals_by_model(rows)["m"]["input"] == 10_000_000


def test_a_recorded_recovery_explains_the_gap_instead_of_repeating_the_alarm():
    """The #139 gap is closed: it was found, priced, and written into the ledger as
    a recovery. The check must still refuse to count that row as a measurement — it
    was read off this same bill — but reporting the same 11,940,249 tokens as
    unaccounted for every run teaches an operator to ignore the one check that
    catches a silent no-write."""
    logged = ledger_totals_by_model([
        _row("claude-opus-5", input_tokens=48_744, output_tokens=2_752),
        _row("claude-opus-5", kind=ledger.RECOVERED, input_tokens=11_940_249,
             output_tokens=415_751),
    ])
    assert logged["claude-opus-5"]["input"] == 48_744        # not the recovered row
    assert logged["claude-opus-5"]["recovered_input"] == 11_940_249
    row = reconcile({"claude-opus-5": OPUS_BILLED}, logged)[0]
    assert row["verdict"] == "ok (1 recovered)"
    assert row["logged_rows"] == 1 and row["unexplained_input"] == 0


def test_a_partial_recovery_still_flags_the_part_nobody_has_looked_at():
    """A recovery explains only what it covers. The remainder is spend with no
    record, which is exactly what the check exists to say."""
    logged = ledger_totals_by_model([
        _row("m", kind=ledger.RECOVERED, input_tokens=400_000)])
    row = reconcile({"m": {"input": 1_000_000}}, logged)[0]
    assert row["verdict"].startswith("UNDER")
    assert row["unexplained_input"] == 600_000


def test_the_report_prints_the_recovered_column_and_does_not_cry_emergency(capsys):
    logged = ledger_totals_by_model([
        _row("claude-opus-5", input_tokens=48_744),
        _row("claude-opus-5", kind=ledger.RECOVERED, input_tokens=11_940_249),
    ])
    worst = print_reconciliation(reconcile({"claude-opus-5": OPUS_BILLED}, logged))
    out = capsys.readouterr().out
    assert worst == []
    assert "recovered in" in out and "11,940,249" in out
    assert "emergency with a deadline" not in out
    assert "not a measurement" in out       # still not evidence the leg was logged


def test_the_report_says_what_to_do_about_a_gap(capsys):
    """A missing row is recoverable for ~6 weeks and then not at all, so the output
    has to carry the deadline rather than just the discrepancy."""
    worst = print_reconciliation(reconcile({"claude-opus-5": OPUS_BILLED},
                                           ledger_totals_by_model([])))
    out = capsys.readouterr().out
    assert len(worst) == 1
    assert "emergency with a deadline" in out and "6 weeks" in out
    assert "per-model per-DAY" in out       # per-split attribution is already gone
    assert "11,988,993" in out
