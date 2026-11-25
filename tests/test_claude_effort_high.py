"""The effort=high write-up's numbers, re-derived from committed files.

``docs/claude_effort_high.md`` pairs the Opus 5.5 effort=high leg against the effort=low leg.
CPU only, no network, no ``.model_cache``:

1. every per-split P / R / F1 and boxes-per-pano figure in ``results.json`` re-scores from the
   published detections and the bundles, and every low-leg cell equals the scoreboard's (the
   full bootstrap is skipped: it takes about a minute;
   ``python scripts/analysis/claude_effort_high.py --check`` re-derives everything);
2. the bootstrap machinery (``interval``, ``paired``, ``pooled``) on small hand-made inputs whose
   answer is known: identical legs give 0 [0, 0], a fixed gap gives that gap with the right
   sign, pools resample within each split, and the macro and micro differences are distinct;
3. the ledger figures: the high leg's per-split costs and ``high_total_usd`` are the sum of its
   ledger rows, every pano called once;
4. every table row the doc prints is exactly what ``--markdown`` renders from ``results.json``,
   and the prose figures (counts, cost range, totals) in the doc, the roster note and the
   ``docs/README.md`` row are the ones ``results.json`` and the ledger give.
"""
import json
import os
import random
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import claude_effort_high as M  # noqa: E402
from rampnet import roster  # noqa: E402
from rampnet.eval import PINNED_SPLITS, POOLED_SPLITS  # noqa: E402

DOC = os.path.join(REPO, "docs", "claude_effort_high.md")
README = os.path.join(REPO, "docs", "README.md")


@pytest.fixture(scope="module")
def results():
    with open(M.OUT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def board():
    with open(M.SCOREBOARD, encoding="utf-8") as fh:
        return json.load(fh)["per_split"]


def _flat(text):
    """Whitespace collapsed, so a figure that wraps across lines still matches."""
    return " ".join(text.split())


def _iv(d):
    return f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


# --------------------------------------------------------------------------- #
# 1. Re-scoring from the published detections
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", M.SPLITS)
def test_scores_and_box_counts_rescore_from_published_detections(results, board, split):
    rows, panos, emitted = M.load_split(split)
    want = results["per_split"][split]
    assert len(panos) == want["n_panos"]
    cells = {m: M.cell(rows[m], panos) for m in rows}
    for m in rows:
        assert cells[m] == want["scores"][m], (split, m)
    for leg in M.LEGS:
        assert emitted[leg] == want["boxes_emitted"][leg], (split, leg)
        assert round(emitted[leg] / len(panos), 3) == want["boxes_per_pano"][leg], (split, leg)
    M.assert_low_matches_scoreboard(split, cells[M.LOW], board)


def test_the_scoreboard_check_fails_when_the_low_leg_drifts(board):
    """The low leg is the scoreboard's leg; a drift between the two must stop the build."""
    cell = dict(board[M.LOW]["clovis"])
    M.assert_low_matches_scoreboard("clovis", cell, board)
    drifted = {**cell, "f1": cell["f1"] + 0.001}
    with pytest.raises(AssertionError):
        M.assert_low_matches_scoreboard("clovis", drifted, board)
    with pytest.raises(AssertionError):
        M.assert_low_matches_scoreboard("clovis", {**cell, "fp": cell["fp"] + 1}, board)


def test_the_split_lists_cover_every_city_split_once(results):
    assert M.CITY_SPLITS == tuple(POOLED_SPLITS) + M.OTHER_SPLITS
    assert len(set(M.CITY_SPLITS)) == len(M.CITY_SPLITS) == 13
    assert set(M.CITY_SPLITS) == set(PINNED_SPLITS) - {"manual_gold"}
    assert results["splits"] == list(M.SPLITS)
    assert results["pooled_other"]["splits"] == list(M.OTHER_SPLITS)
    assert results["pooled_us"]["splits"] == list(POOLED_SPLITS)


def test_the_bootstrap_settings_are_the_committed_ones(results):
    """A different seed or draw count moves every interval; --check catches it, and so
    does this, without the minute-long rebuild."""
    b = results["bootstrap"]
    assert (b["seed"], b["n_draws"]) == (M.SEED, M.N_BOOT) == (0, 10_000)


# --------------------------------------------------------------------------- #
# 2. The bootstrap on hand-made inputs
# --------------------------------------------------------------------------- #
HIT = (1, 0, 1, True)      # one ground truth, found
MISS = (0, 1, 1, True)     # one ground truth, missed, one false positive
EMPTY = (0, 0, 0, True)    # nothing there, nothing emitted


def test_interval_takes_the_2_5_and_97_5_percentiles():
    diffs = [i / 1000 for i in range(1000)]
    iv = M.interval(diffs, 0.5)
    assert iv["ci95"] == [0.025, 0.974]
    assert iv["diff"] == 0.5 and iv["ci_excludes_zero"] is True
    assert iv["share_draws_above_zero"] == 0.999


def test_paired_gives_zero_for_identical_legs():
    """Heterogeneous panos, so an unpaired bootstrap (an independent draw per leg) would
    give a non-zero spread; a paired one gives exactly 0 [0, 0]."""
    panos = [f"p{i}" for i in range(12)]
    rows = {p: (HIT, MISS, EMPTY, (2, 1, 3, True))[i % 4] for i, p in enumerate(panos)}
    iv = M.paired(rows, dict(rows), panos, random.Random(0))
    assert iv["diff"] == 0 and iv["ci95"] == [0, 0]
    assert iv["ci_excludes_zero"] is False


def test_paired_gives_a_fixed_gap_with_its_sign():
    panos = [f"p{i}" for i in range(10)]
    perfect = {p: HIT for p in panos}
    useless = {p: MISS for p in panos}
    up = M.paired(perfect, useless, panos, random.Random(0))
    assert up["diff"] == 1.0 and up["ci95"] == [1.0, 1.0]
    down = M.paired(useless, perfect, panos, random.Random(0))
    assert down["diff"] == -1.0 and down["ci95"] == [-1.0, -1.0]


def _two_strata():
    """Split ``a`` (3 panos): both legs perfect. Split ``b`` (5 panos): high perfect, low
    misses everything. A stratified draw keeps 3 + 5 panos, so the micro F1s are
    constant: high 1.0, low 2*(3/8)*(3/8)/(3/4) = 0.375, a gap of 0.625. The macro gap is
    the mean of the per-split gaps, (0 + 1) / 2 = 0.5. An unstratified draw changes the
    mix of a and b panos, and so the micro gap, on every draw."""
    per = {}
    for split, n, low in (("a", 3, HIT), ("b", 5, MISS)):
        panos = [f"{split}{i}" for i in range(n)]
        per[split] = ({M.HIGH: {p: HIT for p in panos}, M.LOW: {p: low for p in panos},
                       M.RAMPNET: {p: HIT for p in panos}}, panos)
    return per


def test_pooled_resamples_within_each_split_and_reports_micro_and_macro():
    p = M.pooled(_two_strata(), ("a", "b"), 0)
    assert p["n_panos"] == 8
    assert p["high_minus_low"]["diff"] == 0.625
    assert p["high_minus_low"]["ci95"] == [0.625, 0.625]
    assert p["macro_f1"] == {M.HIGH: 1.0, M.LOW: 0.5, M.RAMPNET: 1.0}
    assert p["high_minus_low_macro"]["diff"] == 0.5
    assert p["high_minus_low_macro"]["ci95"] == [0.5, 0.5]


# --------------------------------------------------------------------------- #
# 3. The ledger figures
# --------------------------------------------------------------------------- #
def _high_rows():
    rows = [r for r in M.read_rows(M.LEDGER) or []
            if r.get("model_id") == "claude-opus-5-5"
            and (r.get("signature") or {}).get("effort") == "high"
            and (r.get("signature") or {}).get("max_tokens") == 20000]
    assert rows
    return rows


def test_the_high_legs_costs_are_its_ledger_rows(results):
    """Every pano of every city split called once, and each split's cost the sum of its
    rows. ``high_total_usd`` is the unrounded sum, rounded once."""
    rows = [r for r in _high_rows() if r["bundle"] in M.SPLITS]
    cost = M.ledger_cost()
    total = 0.0
    for s in M.SPLITS:
        mine = [r for r in rows if r["bundle"] == s]
        usd = sum(r["est_cost_usd"] for r in mine)
        total += usd
        assert sum(r["panos_called"] for r in mine) == results["per_split"][s]["n_panos"], s
        assert cost[(M.HIGH, s)]["panos"] == results["per_split"][s]["n_panos"], s
        assert cost[(M.HIGH, s)]["usd"] == pytest.approx(usd), s
        assert results["per_split"][s]["cost"][M.HIGH]["usd"] == round(usd, 2), s
    assert M.total_usd(cost, M.HIGH) == results["high_total_usd"] == round(total, 2)
    assert {r["api_account"] for r in rows} == {"claude20"}


def test_only_boundary_intervals_get_a_seed_check(results):
    """An interval end within BOUNDARY of zero is re-drawn at several seeds, so the doc
    does not count it as excluding zero (or not) off one seed. clovis is the only one."""
    assert M.on_boundary([-0.0964, 0.0008]) and M.on_boundary([0.0015, 0.05])
    assert not M.on_boundary([-0.059, -0.006]) and not M.on_boundary([0.017, 0.090])
    checked = {s for s in M.SPLITS if "seed_check" in results["per_split"][s]["high_minus_low"]}
    assert checked == {s for s in M.SPLITS
                       if M.on_boundary(results["per_split"][s]["high_minus_low"]["ci95"])}
    assert checked == {"clovis"}
    sc = results["per_split"]["clovis"]["high_minus_low"]["seed_check"]
    assert sc["seeds"] == list(M.SEED_CHECK_SEEDS)
    assert sc["ci95_by_seed"][0] == results["per_split"]["clovis"]["high_minus_low"]["ci95"]
    assert sc["n_excluding_zero"] == sum(lo > 0 or hi < 0 for lo, hi in sc["ci95_by_seed"])


# --------------------------------------------------------------------------- #
# 4. The doc, the roster note and the README row
# --------------------------------------------------------------------------- #
def test_every_table_row_in_the_doc_matches_results_json(results):
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    rows = [line for line in M.markdown(results).splitlines() if line.startswith("| ")]
    assert len(rows) > 30
    missing = [r for r in rows if r not in doc]
    assert not missing, missing


def _prose_figures(results):
    """The figures the prose quotes, each derived from results.json or the ledger."""
    ps = results["per_split"]
    us, other, every = results["pooled_us"], results["pooled_other"], results["pooled_all_cities"]
    d = {k: ps[k]["high_minus_low"] for k in M.SPLITS}
    f = {}
    f["us_macro"] = f"{us['macro_f1'][M.LOW]:.3f} -> {us['macro_f1'][M.HIGH]:.3f}"
    f["us_macro_iv"] = _iv(us["high_minus_low_macro"])
    f["us_micro_iv"] = _iv(us["high_minus_low"])
    f["other_macro"] = f"{other['macro_f1'][M.LOW]:.3f} -> {other['macro_f1'][M.HIGH]:.3f}"
    f["other_macro_iv"] = _iv(other["high_minus_low_macro"])
    f["other_micro_iv"] = _iv(other["high_minus_low"])
    f["all_micro_iv"] = _iv(every["high_minus_low"])
    f["n_lower_us"] = f"lower on {sum(d[s]['diff'] < 0 for s in POOLED_SPLITS)} of the 9"
    f["gainesville"] = _iv(d["gainesville"])
    f["budapest"] = _iv(d["budapest_district5"])
    seeds = d["clovis"]["seed_check"]
    f["clovis_seeds"] = (f"excludes zero at {seeds['n_excluding_zero']} of "
                         f"{len(seeds['seeds'])} bootstrap seeds")
    f["clovis_upper"] = f"{d['clovis']['ci95'][1]:+.3f}"
    sc = {s: (ps[s]["scores"][M.LOW], ps[s]["scores"][M.HIGH]) for s in M.SPLITS}
    f["clovis_precision"] = (f"{sc['clovis'][0]['precision']:.3f} -> "
                             f"{sc['clovis'][1]['precision']:.3f}")
    rm = {s: ps[s]["rampnet_minus_high"] for s in M.SPLITS}
    f["n_rampnet_ahead"] = f"ahead on {sum(rm[s]['diff'] > 0 for s in M.SPLITS)} of the 13"
    f["rm_laurens_mapillary"] = f"laurens_mapillary, RampNet minus high " \
                                f"{rm['laurens_mapillary']['diff']:+.3f}"
    f["rm_bayonne"] = f"bayonne {rm['bayonne']['diff']:+.3f}"
    c = {s: ps[s]["cost"] for s in M.SPLITS}
    lo_pp = [c[s][M.LOW]["usd_per_pano"] for s in M.SPLITS]
    hi_pp = [c[s][M.HIGH]["usd_per_pano"] for s in M.SPLITS]
    f["cost_range"] = (f"${min(hi_pp):.3f}-{max(hi_pp):.3f} per pano against "
                       f"${min(lo_pp):.3f}-{max(lo_pp):.3f}")
    ratio = (sum(c[s][M.HIGH]["usd"] for s in M.SPLITS) / sum(c[s][M.HIGH]["panos_called"]
             for s in M.SPLITS)) / (sum(c[s][M.LOW]["usd"] for s in M.SPLITS)
                                    / sum(c[s][M.LOW]["panos_called"] for s in M.SPLITS))
    f["pct"] = f"about {round((ratio - 1) * 100)}% more"
    th = {leg: sum(c[s][leg]["thinking_per_call"] for s in M.SPLITS) / len(M.SPLITS)
          for leg in M.LEGS}
    f["thinking"] = (f"about {round(th[M.HIGH], -1):.0f} thinking tokens per call against "
                     f"about {round(th[M.LOW], -1):.0f}")
    # The prose that restates a table cell, anchored to its sentence, so a prose edit cannot
    # hide behind the same figure in a table row (review 2 of #299).
    f["us_line"] = (f"macro F1 {f['us_macro']}, {f['us_macro_iv']}; micro {f['us_micro_iv']}")
    f["other_line"] = (f"macro {f['other_macro']}, {f['other_macro_iv']}; "
                       f"micro {f['other_micro_iv']}")
    f["all_line"] = (f"wash**: macro {_iv(every['high_minus_low_macro'])}, "
                     f"micro {f['all_micro_iv']}")
    worst = min(POOLED_SPLITS, key=lambda s: d[s]["diff"])
    f["worst_us"] = f"{worst} drops the most ({d[worst]['diff']:+.3f})"
    f["gainesville_line"] = f"gainesville's own interval excludes zero ({f['gainesville']})"
    f["budapest_line"] = (f"budapest_district5 ({f['budapest']}, the one split where high is "
                          f"clearly ahead)")
    f["clovis_precision_line"] = f"almost all of it precision ({f['clovis_precision']})"
    bpp = {s: {leg: ps[s]["boxes_emitted"][leg] / ps[s]["n_panos"] for leg in M.LEGS}
           for s in M.SPLITS}
    f["boxes_line"] = (f"(clovis {bpp['clovis'][M.LOW]:.2f} -> {bpp['clovis'][M.HIGH]:.2f} "
                       f"per pano, annapolis {bpp['annapolis'][M.LOW]:.2f} -> "
                       f"{bpp['annapolis'][M.HIGH]:.2f})")
    f["total"] = f"The 13 city splits cost ${results['high_total_usd']:.2f} at list price"
    f["total_source"] = f"| $/pano, thinking/call, ${results['high_total_usd']:.2f} |"
    return f, sc


def test_the_docs_prose_figures_are_what_results_json_says(results):
    with open(DOC, encoding="utf-8") as fh:
        doc = _flat(fh.read())
    f, sc = _prose_figures(results)
    missing = {k: v for k, v in f.items() if v not in doc}
    assert not missing, missing
    # Who gains or loses on precision and recall, named in the mechanism bullet.
    both_up = [s for s in M.SPLITS if sc[s][1]["precision"] > sc[s][0]["precision"]
               and sc[s][1]["recall"] > sc[s][0]["recall"]]
    both_down = [s for s in M.SPLITS if sc[s][1]["precision"] < sc[s][0]["precision"]
                 and sc[s][1]["recall"] < sc[s][0]["recall"]]
    same_recall = [s for s in M.SPLITS if sc[s][1]["recall"] == sc[s][0]["recall"]]
    trade = [s for s in M.SPLITS if sc[s][1]["precision"] < sc[s][0]["precision"]
             and sc[s][1]["recall"] > sc[s][0]["recall"]]
    assert f"recall rises and precision falls on {len(trade)} of the 13" in doc
    assert f"{' and '.join(both_down)} lose on both" in doc
    assert f"{' and '.join(both_up)} gains on both" in doc
    assert f"{' and '.join(same_recall)} keep their recall and lose precision" in doc
    assert all(sc[s][1]["precision"] < sc[s][0]["precision"] for s in same_recall)
    # Every split emits more boxes at high effort.
    ps = results["per_split"]
    assert all(ps[s]["boxes_emitted"][M.HIGH] > ps[s]["boxes_emitted"][M.LOW] for s in M.SPLITS)
    assert "more boxes on every split" in doc
    # The two splits where Claude beats RampNet with an interval that excludes zero.
    beats = {s for s in M.SPLITS if ps[s]["rampnet_minus_high"]["ci95"][1] < 0}
    assert beats == {"laurens_mapillary", "bayonne"}


def test_the_docs_run_to_run_caveat_is_stated_and_true(results):
    """The Result section's opening caveat (the fix for finding 1 of review 1), checked
    against the code and the ledger it describes, and the #276 scale figure against the
    committed replicate table it cites."""
    import detectors
    with open(DOC, encoding="utf-8") as fh:
        doc = _flat(fh.read())
    assert "**Every interval here resamples panos only.**" in doc
    assert detectors.CLAUDE_AS_RUN_TEMPERATURE is None
    assert "(`CLAUDE_AS_RUN_TEMPERATURE = None` in `detectors.py`, not greedy)" in doc
    # The two legs' run dates, from their ledger rows.
    from datetime import date
    rows = M.read_rows(M.LEDGER)
    low = sorted({r["ts"][:10] for r in rows if r.get("model_id") == "claude-opus-5-5"
                  and (r.get("signature") or {}).get("effort") == "low"
                  and r.get("api_account") and r["ts"] < "2026-10-21"
                  and r["bundle"] in M.SPLITS})
    high = sorted({r["ts"][:10] for r in _high_rows() if r["bundle"] in M.SPLITS})
    assert low == ["2026-10-19", "2026-10-20"] and high == ["2026-11-17"]
    gap = (date.fromisoformat(high[0]) - date.fromisoformat(low[-1])).days
    assert 25 <= gap <= 31, gap
    assert "ran four weeks apart (low on 2026-10-19/20, high on 2026-11-17)" in doc
    # The US result is worded as one run's result, not as settled.
    assert ("**High effort gives no gain on the nine pooled US cities, and is lower in this "
            "run**") in doc
    assert "is worse" not in doc and "worse on the nine" not in doc
    # The #276 replicate range: three runs of the zero-shot leg on a two-split pool.
    with open(os.path.join(REPO, "analysis_out", "prompt_opt_276", "rewrite_replicates.json"),
              encoding="utf-8") as fh:
        rep = json.load(fh)
    spread = rep["spread"]["zs"]["range"]
    assert len(rep["runs"]) == 3 and len(rep["pool"]) == 2
    assert (f"three runs of one zero-shot Gemini leg on a two-split pool spanned "
            f"{spread:.3f} in macro F1") in doc
    # "larger than either pooled difference below"
    assert spread > max(abs(results[k]["high_minus_low_macro"]["diff"])
                        for k in ("pooled_us", "pooled_other"))
    assert "larger than either pooled difference below" in doc


def test_the_docs_timing_is_what_the_ledger_says():
    """'21 s per pano per stream in practice, against 24.7 s in the 5-pano calibration':
    the full-split rows' elapsed over their panos, and the one 5-pano high-effort row."""
    with open(DOC, encoding="utf-8") as fh:
        doc = _flat(fh.read())
    rows = [r for r in _high_rows() if r["bundle"] in M.SPLITS]
    cal = [r for r in rows if r["panos_called"] == 5]
    full = [r for r in rows if r["panos_called"] > 5]
    assert len(cal) == 1 and len(full) == 13
    per = sum(r["elapsed_s"] for r in full) / sum(r["panos_called"] for r in full)
    assert (f"{per:.0f} s per pano per stream in practice, against "
            f"{cal[0]['s_per_pano']:.1f} s in the 5-pano calibration") in doc


def test_the_figure_draws_the_committed_micro_intervals(results, tmp_path, monkeypatch):
    """The forest plot, drawn into a temporary file, with its axes captured: every split's
    dot and bar are that split's high-minus-low interval, and the three pooled diamonds
    are the MICRO intervals their "(micro)" labels promise, in run order."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captured = {}
    real = plt.subplots

    def spy(*a, **k):
        fig, ax = real(*a, **k)
        captured["fig"], captured["ax"] = fig, ax
        return fig, ax
    monkeypatch.setattr(plt, "subplots", spy)
    monkeypatch.setattr(M, "FIGURE", str(tmp_path / "forest.png"))
    # figures() prints the path relative to the repo, which raises on Windows when the
    # temporary directory is on another drive.
    monkeypatch.setattr(M.os.path, "relpath", lambda p, start=None: p)
    M.figures()
    ax = captured["ax"]
    try:
        assert (tmp_path / "forest.png").exists()
        dots = [ln.get_xdata()[0] for ln in ax.lines if ln.get_marker() == "o"]
        diamonds = [ln.get_xdata()[0] for ln in ax.lines if ln.get_marker() == "D"]
        bars = [list(ln.get_xdata()) for ln in ax.lines
                if len(ln.get_xdata()) == 2 and ln.get_marker() in (None, "None", "")
                and ln.get_ydata()[0] == ln.get_ydata()[1]]
        pools = ("pooled_us", "pooled_other", "pooled_all_cities")
        ivs = [results["per_split"][s]["high_minus_low"] for s in results["splits"]]
        ivs += [results[k]["high_minus_low"] for k in pools]
        assert dots == [iv["diff"] for iv in ivs[:-3]]
        assert diamonds == [iv["diff"] for iv in ivs[-3:]]
        assert bars == [iv["ci95"] for iv in ivs]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels[-3:] == ["Pooled (micro): 9 US cities", "Pooled (micro): 4 other cities",
                               "Pooled (micro): all 13 cities"]
    finally:
        plt.close(captured["fig"])


def test_the_days_other_ledger_rows_are_what_the_doc_says():
    """The manual_gold slice and the effort=max calibration, read off the ledger."""
    with open(DOC, encoding="utf-8") as fh:
        doc = _flat(fh.read())
    mg = [r for r in _high_rows() if r["bundle"] == "manual_gold"]
    assert len(mg) == 1 and mg[0]["panos_scored"] == 150
    assert f"${mg[0]['est_cost_usd']:.2f} for a 150-pano `manual_gold` slice" in doc
    assert f"about ${850 * mg[0]['est_cost_usd'] / 150:.0f}" in doc
    mx = [r for r in M.read_rows(M.LEDGER)
          if (r.get("signature") or {}).get("effort") == "max"
          and r.get("model_id") == "claude-opus-5-5"]
    assert len(mx) == 1 and mx[0]["panos_called"] == 5
    assert f"${mx[0]['est_cost_usd']:.2f} for an effort=max calibration" in doc
    assert f"took {mx[0]['s_per_pano']:.0f} s per pano" in doc


def test_the_roster_note_and_readme_row_quote_the_same_figures(results):
    leg = next(c for c in roster.ROSTER if roster.published_name(c) == M.HIGH)
    note = _flat(leg.note)
    us = results["pooled_us"]
    f, _ = _prose_figures(results)
    assert (f"{us['macro_f1'][M.HIGH]:.3f} against low's {us['macro_f1'][M.LOW]:.3f}") in note
    assert f"micro high minus low {f['us_micro_iv']}" in note
    assert f"micro {f['other_micro_iv']}" in note
    assert f"macro {f['other_macro_iv']}" in note
    assert "one run per arm" in note.lower() and "resample panos only" in note
    pct = f["pct"].split()[1]                      # "18%"
    thinking = f["thinking"].split()[1]            # "150"
    assert f"~{thinking} thinking tokens/call" in note
    assert f"~{pct} more cost per pano" in note
    with open(README, encoding="utf-8") as fh:
        row = next(line for line in fh if line.startswith("| [`claude_effort_high.md`]"))
    assert f"{f['us_macro']} ({us['high_minus_low_macro']['diff']:+.3f})" in row
    assert "one run per arm" in row
    bud = results["per_split"]["budapest_district5"]["high_minus_low"]["diff"]
    assert f"budapest_district5 rises {bud:+.3f}" in row
    assert f"about {pct} more cost per pano" in row
