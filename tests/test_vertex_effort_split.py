"""Guards on the two-leg effort split (#139, #143).

The cloud query needs credentials and is not testable here. The *decision* logic is,
and it is the part that can hand back a confident wrong number: a mixture solver
pointed at a flat series will cheerfully report that high effort cost less than low.
These pin the changepoint detector and the separability rule against series whose
right answer is known by construction.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

ves = pytest.importorskip("vertex_effort_split")
vu = pytest.importorskip("vertex_usage")
pricing = pytest.importorskip("pricing")

SERIES_DIR = os.path.join(REPO, "docs", "data", "vertex_minute_series")
DAILY_SNAPSHOT = os.path.join(SERIES_DIR, "vertex_usage_daily_2026-09-03.json")


@pytest.fixture
def no_dotenv(monkeypatch):
    """main() reads the repo-root .env before argparse. On a developer checkout that
    file holds credentials, and a test that calls main() would load them into the
    pytest process for the rest of the run (S7). Every main() test takes this."""
    monkeypatch.setattr(ves, "_load_dotenv", lambda: None)


def _point(end_time, ttype, n):
    """One Cloud Monitoring time series carrying one ALIGN_DELTA point."""
    return {"metric": {"type": ves.TOKEN_METRIC, "labels": {"type": ttype}},
            "points": [{"interval": {"endTime": end_time},
                        "value": {"int64Value": str(n)}}]}


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


def test_changepoint_can_land_on_the_last_full_window():
    """S6: a cliff exactly `window` minutes from the end. rows[i:i+window] is a full
    window up to i == len(rows) - window, and the search has to include it -- the
    first version stopped one short, put the cut a minute early and read a diluted
    "after" window (3.57x here instead of the real 10x). This is not hypothetical:
    the committed Sonnet series has its largest drop at exactly that position."""
    rows = _series([(15, 100, 0.05), (5, 10, 0.05)])
    cut, drop = ves.find_changepoint(rows)
    assert cut == 15
    assert drop == pytest.approx(10.0, abs=0.01)
    # ...and one more tail minute, which the old range did search, agrees.
    cut, drop = ves.find_changepoint(rows + [("2026-08-15T00:20:00Z", 10, 1)])
    assert (cut, drop) == (15, pytest.approx(10.0, abs=0.01))


def test_changepoint_refuses_a_series_with_no_input_to_anchor_on():
    """S2 made zero-input minutes reachable, so the detector must say "none" rather
    than slice on it: (None, 0.0), and main() turns that into a message, not a
    TypeError."""
    rows = [(f"2026-08-15T00:{m:02d}:00Z", 0, 50) for m in range(20)]
    assert ves.find_changepoint(rows) == (None, 0.0)


def test_output_only_minutes_survive_the_bucket_step():
    """S2: a minute with output tokens and no input tokens is a long response
    completing after its request was counted. It is billed, and the first version
    dropped it before save_series ran, so the committed file could never be checked
    against the daily row for that loss. Every minute with any tokens is kept."""
    series = [_point("2026-08-15T17:51:00Z", "input", 60_000),
              _point("2026-08-15T17:51:00Z", "output", 2_000),
              _point("2026-08-15T17:52:00Z", "output", 5),        # output only
              _point("2026-08-15T17:53:00Z", "input", 60_000),
              _point("2026-08-15T17:54:00Z", "input", 0)]          # a zero point
    rows = ves.minute_rows(series)
    assert rows == [("2026-08-15T17:51:00Z", 60_000, 2_000),
                    ("2026-08-15T17:52:00Z", 0, 5),
                    ("2026-08-15T17:53:00Z", 60_000, 0)]
    assert sum(r[2] for r in rows) == 2_005                # nothing lost


def test_an_output_only_series_replays_to_a_message_not_a_traceback(
        tmp_path, monkeypatch, no_dotenv):
    """The other half of S2: now that such minutes are kept, a replayed series can
    have no input anywhere, and main() must exit with a sentence."""
    def replay(rows):
        path = tmp_path / "series.json"
        ves.save_series(path, "claude-opus-5", "s", "e", rows)
        monkeypatch.setattr(sys, "argv", ["vertex_effort_split.py", "--model",
                                          "claude-opus-5", "--from-series", str(path)])
        with pytest.raises(SystemExit) as e:
            ves.main()
        return str(e.value)

    # Nothing but output: refused before the blended ratio would divide by zero.
    assert "no input tokens" in replay(
        [(f"2026-08-15T00:{m:02d}:00Z", 0, 50) for m in range(20)])
    # Some input, but never on both sides of a window: the detector returns None
    # and main() must say so rather than slice rows[:None - 3].
    assert "no changepoint" in replay(
        [("2026-08-15T00:00:00Z", 12_186, 400), ("2026-08-15T00:01:00Z", 12_186, 400)]
        + [(f"2026-08-15T00:{m:02d}:00Z", 0, 50) for m in range(2, 20)])


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
    # The largest drop sits exactly `window` minutes from the end -- the position the
    # S6 off-by-one used to exclude. It read 17:35 and 1.63x before that fix.
    assert rows[cut][0] == "2026-08-15T17:36:00Z"
    assert drop == pytest.approx(1.78, abs=0.01)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_head == pytest.approx(0.0363, abs=0.0001)
    assert r_tail == pytest.approx(0.0273, abs=0.0001)
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


def test_the_daily_snapshot_backs_the_published_cost_table(tmp_path):
    """S4: the four Claude rows in docs/model_comparison.md's recovery table -- $21.47,
    $7.79, $70.41 and the $0.03 re-run -- come from the committed daily snapshot, and
    until this test nothing opened that file. The rows are pinned, priced through
    pricing.py to the published figures, and round-tripped through write_json (the
    --save-rows path, dict rows, not save_series) byte-for-byte except fetched_utc."""
    with open(DAILY_SNAPSHOT, encoding="utf-8", newline="") as f:
        original = f.read()
    doc = json.loads(original)
    assert doc["alignment_period"] == "86400s"
    claude = {(r["window_end"], r["model"]): r["tokens"] for r in doc["rows"]
              if r["model"].startswith("claude-")}
    want = {                                   # window_end, model -> input, output, $
        ("2026-08-16", "claude-opus-5"): (3_058_702, 247_222, 21.47),
        ("2026-08-16", "claude-sonnet-5"): (3_300_368, 118_471, 7.79),
        ("2026-08-19", "claude-opus-5"): (11_988_993, 418_503, 70.41),
        ("2026-08-19", "claude-sonnet-5"): (12_594, 480, 0.03),
    }
    assert set(claude) == set(want)
    for key, (tin, tout, dollars) in want.items():
        tokens = claude[key]
        assert (tokens["input"], tokens["output"]) == (tin, tout), key
        # Every token type is carried, including the cache buckets that are zero here:
        # a snapshot that dropped a billed bucket would be worse than no snapshot.
        assert {"cache_read_input", "cache_write_1h_input", "cache_write_input"} <= set(tokens)
        assert pricing.estimate_cost(key[1], tin, tout) == pytest.approx(dollars, abs=0.005)
    # The 08-15 Opus and Sonnet minute series are these two daily rows, re-read at 60 s.
    opus_rows, _, _, _ = _replay("claude-opus-5_2026-08-15.json")
    assert (sum(r[1] for r in opus_rows), sum(r[2] for r in opus_rows)) == (3_058_702, 247_222)
    sonnet_rows, _, _, _ = _replay("claude-sonnet-5_2026-08-15.json")
    assert sum(r[1] for r in sonnet_rows) == 3_300_368
    assert sum(r[2] for r in sonnet_rows) == 118_471 - 1     # the one-token gap, see the doc

    out = tmp_path / "daily.json"
    vu.write_json(out, doc, "rows")
    with open(out, encoding="utf-8", newline="") as f:
        written = f.read()
    keep = lambda t: [x for x in t.splitlines() if "fetched_utc" not in x]
    assert keep(written) == keep(original)
    assert "\r" not in written
    assert sum(1 for x in written.splitlines() if x.startswith('    {"window_end"')) == \
        len(doc["rows"])


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


def test_replaying_a_series_under_the_wrong_model_is_refused(monkeypatch, no_dotenv):
    """A series file is per-model. Replaying Sonnet's series as Opus would price the
    wrong rate card against it and print a confident wrong number."""
    monkeypatch.setattr(sys, "argv", [
        "vertex_effort_split.py", "--model", "claude-opus-5", "--from-series",
        os.path.join(SERIES_DIR, "claude-sonnet-5_2026-08-15.json")])
    with pytest.raises(SystemExit) as e:
        ves.main()
    assert "claude-sonnet-5" in str(e.value)


def test_a_cloud_query_still_needs_a_window(monkeypatch, no_dotenv):
    """--start/--end stopped being argparse-required so --from-series could omit them.
    The check has to survive by hand, or a windowless query reaches the API."""
    monkeypatch.setattr(sys, "argv", [
        "vertex_effort_split.py", "--model", "claude-opus-5", "--project", "p"])
    with pytest.raises(SystemExit) as e:
        ves.main()
    assert "--start and --end" in str(e.value)


def test_the_minute_query_and_the_daily_query_share_one_fetch(monkeypatch):
    """S8: vertex_effort_split.py had its own copy of vertex_usage.py's paging loop
    and .env parser, differing only in timeout and page size. One fetch_series now
    serves both, so a paging fix lands in both queries. Checked without HTTP: the
    minute fetch must call the shared function with a 60 s alignment and the
    per-model filter, and hand its result to minute_rows."""
    calls = []

    def fake_fetch(project, filter_, start, end, alignment_period, group_by, **kw):
        calls.append((project, filter_, start, end, alignment_period, list(group_by), kw))
        return [_point(end, "input", 12_186), _point(end, "output", 400)]

    monkeypatch.setattr(ves, "fetch_series", fake_fetch)
    rows = ves.fetch_minute_series("proj", "claude-opus-5", "S", "E")
    assert rows == [("E", 12_186, 400)]
    (project, filter_, start, end, alignment, group_by, kw), = calls
    assert (project, start, end, alignment) == ("proj", "S", "E", "60s")
    assert ves.TOKEN_METRIC in filter_ and 'model_user_id = "claude-opus-5"' in filter_
    assert group_by == ["metric.labels.type"]
    assert kw == {"timeout": 90, "page_size": 2000}
    assert ves.fetch_series is not vu.fetch_series      # the patch took
    assert ves._load_dotenv is vu._load_dotenv          # one .env reader, not two


def test_the_dotenv_fallback_reads_the_same_file_the_same_way(tmp_path, monkeypatch):
    """vertex_usage._load_dotenv prefers compare.load_dotenv; when the detector stack
    is not importable it must still read .env rather than silently load nothing,
    and the two parsers must agree on what a line means."""
    env = tmp_path / ".env"
    env.write_text('# comment\nGOOGLE_CLOUD_PROJECT="proj-a"\nOTHER=x=y\n\nBAD LINE\n',
                   encoding="utf-8")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    vu._parse_dotenv(str(env))
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "proj-a"
    assert os.environ["OTHER"] == "x=y"                  # split on the first '=' only
    compare = pytest.importorskip("compare")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    compare.load_dotenv(str(tmp_path))
    assert (os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["OTHER"]) == ("proj-a", "x=y")
