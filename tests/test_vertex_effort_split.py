"""Guards on the two-leg effort split (#139, #143).

The cloud query needs credentials and is not testable here. The *decision* logic is,
and it is the part that can hand back a confident wrong number: a mixture solver
pointed at a flat series will cheerfully report that high effort cost less than low.
These pin the changepoint detector and the separability rule against series whose
right answer is known by construction.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

ves = pytest.importorskip("vertex_effort_split")
pricing = pytest.importorskip("pricing")

SERIES_DIR = os.path.join(REPO, "docs", "data", "vertex_minute_series")


def _series(spec):
    """[(n_minutes, input_per_min, ratio)] -> the (ts, input, output) rows."""
    rows, minute = [], 0
    for n, inp, ratio in spec:
        for _ in range(n):
            rows.append((f"2026-08-15T{minute // 60:02d}:{minute % 60:02d}:00Z",
                         inp, round(inp * ratio)))
            minute += 1
    return rows


def test_changepoint_finds_the_throughput_cliff():
    """Fast leg finishes, slow leg runs on alone: throughput drops, and the index
    returned must be the first minute of the slow phase, not somewhere in the middle."""
    rows = _series([(30, 60_000, 0.035), (30, 20_000, 0.127)])
    cut, drop = ves.find_changepoint(rows)
    assert cut == 30
    assert drop == pytest.approx(3.0, abs=0.01)


def test_a_flat_series_is_reported_as_not_separable():
    """Sonnet's real shape: both legs at effectively one ratio.

    The detector still returns *a* changepoint -- there is always a largest drop --
    so the separability rule, not the detector, is what has to refuse. If this ever
    starts passing the lift threshold, the split it produces is noise.
    """
    rows = _series([(30, 100_000, 0.036), (25, 60_000, 0.035)])
    cut, _ = ves.find_changepoint(rows)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_tail < r_head * ves.MIN_RATIO_LIFT


def test_a_real_two_component_day_clears_the_threshold():
    """The Opus shape, so the guard is not simply refusing everything."""
    rows = _series([(40, 58_000, 0.0675), (35, 21_000, 0.1203)])
    cut, _ = ves.find_changepoint(rows)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_tail >= r_head * ves.MIN_RATIO_LIFT


def test_the_input_split_is_geometry_not_inference():
    """Input per leg comes from the pano count, so it must not depend on the
    output at all -- that is what makes the mixture solvable with one unknown."""
    per_pano, legs = 12_186, 2
    total_in = 251 * per_pano
    per_leg = (round(total_in / per_pano) // legs) * per_pano
    assert per_leg == 125 * per_pano       # the odd pano is not silently halved


def test_rate_anchor_and_tail_anchor_bracket_the_published_opus_split():
    """The 2026-08-15 Opus day, as recovered. Both anchors must stay on the same
    side of the story: high effort costs more, and the two agree within ~5%."""
    total_in, total_out = 3_058_702, 247_222
    per_leg_in = 125 * 12_186
    for out_high in (0.1203 * per_leg_in,                 # tail anchor
                     total_out - 0.034908 * per_leg_in):  # rate anchor
        out_low = total_out - out_high
        assert out_high > out_low
        # Priced through pricing.py, not a hand-copied rate: if the claude-opus-5
        # row ever moves, the script's output moves with it and so must this test,
        # rather than passing on numbers the script no longer prints.
        cost_low = pricing.estimate_cost("claude-opus-5", per_leg_in, out_low)
        cost_high = pricing.estimate_cost("claude-opus-5", per_leg_in, out_high)
        assert cost_low == pytest.approx(9.1, abs=0.3)
        assert cost_high == pytest.approx(12.3, abs=0.3)
        assert cost_low + cost_high == pytest.approx(21.41, abs=0.05)


# --- the committed minute series (F1) ---------------------------------------
#
# Cloud Monitoring keeps this metric about six weeks, so every figure in
# docs/model_comparison.md section "Splitting a two-leg day by effort" was, until these
# files were committed, derivable only from one cloud project inside one month. These
# replay the committed series and pin the published answers to them, which is what makes
# that section reproducible from a clean clone with no credentials.

def _replay(name):
    return ves.load_series(os.path.join(SERIES_DIR, name))


def test_the_committed_opus_series_replays_the_published_effort_split():
    """The 2026-08-15 Opus day, from the committed rows: 251 panos, the 18:32
    changepoint, and the two anchors that bracket $8.94 / $12.46."""
    rows, model, _, _ = _replay("claude-opus-5_2026-08-15.json")
    assert model == "claude-opus-5"
    assert len(rows) == 76
    tin = sum(r[1] for r in rows)
    tout = sum(r[2] for r in rows)
    assert (tin, tout) == (3_058_702, 247_222)      # == the billed daily row

    per_pano = 12_186
    assert tin / per_pano == pytest.approx(251.0, abs=0.02)

    cut, drop = ves.find_changepoint(rows)
    assert rows[cut][0] == "2026-08-15T18:32:00Z"
    assert drop == pytest.approx(2.54, abs=0.01)

    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_head == pytest.approx(0.0675, abs=0.0001)
    assert r_tail == pytest.approx(0.1203, abs=0.0001)
    assert r_tail >= r_head * ves.MIN_RATIO_LIFT     # separable

    # Both anchors, priced off the rate card, straddle the run-time console figures.
    per_leg_in = (round(tin / per_pano) // 2) * per_pano
    for out_high in (r_tail * per_leg_in, tout - 0.034908 * per_leg_in):
        out_low = tout - out_high
        low = pricing.estimate_cost("claude-opus-5", per_leg_in, out_low)
        high = pricing.estimate_cost("claude-opus-5", per_leg_in, out_high)
        assert low == pytest.approx(9.1, abs=0.3)    # console: $8.94
        assert high == pytest.approx(12.3, abs=0.3)  # console: $12.46


def test_the_committed_sonnet_series_still_refuses_to_separate():
    """Sonnet is the negative result, and it has to stay negative: a future change
    that made this series look separable would publish a wrong split."""
    rows, model, _, _ = _replay("claude-sonnet-5_2026-08-15.json")
    assert model == "claude-sonnet-5"
    assert sum(r[1] for r in rows) == 3_300_368      # == the billed daily row
    cut, drop = ves.find_changepoint(rows)
    assert drop == pytest.approx(1.63, abs=0.01)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_tail < r_head                            # the ratio moves the wrong way
    assert r_tail < r_head * ves.MIN_RATIO_LIFT       # NOT SEPARABLE


def test_the_committed_139_series_carries_the_leg_wall_clock():
    """The $70.41 leg. The money was recovered at the time; the wall-clock is only in
    this series, which is one reason it is committed."""
    from datetime import datetime

    rows, model, _, _ = _replay("claude-opus-5_2026-08-18.json")
    assert model == "claude-opus-5"
    tin = sum(r[1] for r in rows)
    tout = sum(r[2] for r in rows)
    assert (tin, tout) == (11_988_993, 418_503)       # == the billed daily row
    assert pricing.estimate_cost("claude-opus-5", tin, tout) == pytest.approx(
        70.41, abs=0.01)

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    first = datetime.strptime(rows[0][0], fmt)
    last = datetime.strptime(rows[-1][0], fmt)
    assert rows[0][0] == "2026-08-18T23:29:00Z"
    assert rows[-1][0] == "2026-08-19T01:15:00Z"
    assert len(rows) == 83                            # active minutes
    assert (last - first).total_seconds() / 60 == 106.0
    assert tout / 984 == pytest.approx(425.3, abs=0.1)  # output tokens per pano


def test_every_committed_series_round_trips_through_save_and_load(tmp_path):
    """save_series/load_series are the committed artifacts' only writer and reader, so
    a change to either must not silently reshape the files already in git."""
    for name in ("claude-opus-5_2026-08-15.json",
                 "claude-sonnet-5_2026-08-15.json",
                 "claude-opus-5_2026-08-18.json"):
        src = os.path.join(SERIES_DIR, name)
        rows, model, start, end = ves.load_series(src)
        out = tmp_path / name
        ves.save_series(out, model, start, end, rows)
        with open(src, encoding="utf-8", newline="") as f:
            original = f.read()
        with open(out, encoding="utf-8", newline="") as f:
            written = f.read()
        # fetched_utc is provenance and moves; every other byte must not.
        keep = lambda t: [x for x in t.splitlines() if "fetched_utc" not in x]
        assert keep(written) == keep(original)
        # LF-only, and one row per line rather than one integer per line -- which is
        # what keeps a real change to a series visible in a diff.
        assert "\r" not in written
        assert sum(1 for x in written.splitlines()
                   if x.startswith('    ["')) == len(rows)


def test_an_unpriced_model_reports_tokens_instead_of_raising():
    """F3: estimate_cost returns None for a model that is not in the rate card, and
    None cannot be formatted with :7.2f. The script has to decide that before it
    formats anything, or its "no verified price" branch is unreachable code."""
    assert pricing.price_for("claude-opus-5") is not None
    unpriced = "no-such-model-9"
    assert pricing.price_for(unpriced) is None
    assert pricing.estimate_cost(unpriced, 1_000_000, 1_000) is None
    with pytest.raises(TypeError):
        # the shape of the bug: this is what report() used to do unconditionally
        "{:7.2f}".format(pricing.estimate_cost(unpriced, 1_000_000, 1_000))


def test_replaying_a_series_under_the_wrong_model_is_refused(monkeypatch):
    """A series file is per-model. Replaying Sonnet's series as Opus would price the
    wrong rate card against it and print a confident wrong number."""
    monkeypatch.setattr(sys, "argv", [
        "vertex_effort_split.py", "--model", "claude-opus-5", "--from-series",
        os.path.join(SERIES_DIR, "claude-sonnet-5_2026-08-15.json")])
    with pytest.raises(SystemExit) as e:
        ves.main()
    assert "claude-sonnet-5" in str(e.value)


def test_a_cloud_query_still_needs_a_window(monkeypatch):
    """--start/--end stopped being argparse-required so --from-series could omit them.
    The check has to survive by hand, or a windowless query reaches the API."""
    monkeypatch.setattr(sys, "argv", [
        "vertex_effort_split.py", "--model", "claude-opus-5", "--project", "p"])
    with pytest.raises(SystemExit) as e:
        ves.main()
    assert "--start and --end" in str(e.value)
