"""The newest-ids write-up's derived numbers, re-derived from committed files (PR #261).

``docs/claude_newest_ids.md`` quotes more than the plain P / R / F1 table that
``test_claude_published_legs.py`` pins: paired F1 intervals against RampNet, a radius check
on all 13 splits, emitted-point counts, per-pano cost ranges, per-call token and thinking
figures, and a cost estimate for the juneau legs that were not run. Each of those is checked
here, from ``benchmark/model_detections/``, the committed bundles,
``analysis_out/claude_newest_ids_261/results.json`` and ``analysis_out/usage_log.jsonl``.
CPU only, no network, no ``.model_cache``.

The full bootstrap over all 13 splits takes about a minute, so the paired intervals are
re-derived here only on the three splits where a Claude leg scores above RampNet (the claims
the docs make); the radius table is re-derived on every split with the bootstrap skipped.
``python scripts/analysis/claude_newest_ids_261.py --check`` re-derives everything.
"""
import json
import os
import re
import statistics as st
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import claude_newest_ids_261 as M  # noqa: E402

DOC = os.path.join(REPO, "docs", "claude_newest_ids.md")
USAGE_LOG = os.path.join(REPO, "analysis_out", "usage_log.jsonl")
CLAIM_SPLITS = ("bayonne", "laurens_gsv", "laurens_mapillary")
NAMES = {M.OPUS: "Opus 5.5", M.FABLE: "Fable 5.1"}


def _signed(x):
    """The docs' signed format: a real minus sign, an explicit plus."""
    return f"{x:+.3f}".replace("-", "\N{MINUS SIGN}")


@pytest.fixture(scope="module")
def committed():
    with open(M.OUT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def board():
    return M.load_board()


@pytest.fixture(scope="module")
def doc():
    with open(DOC, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def ledger():
    with open(USAGE_LOG, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("api_account")]


# --------------------------------------------------------------------------- #
# the committed artifact re-derives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", CLAIM_SPLITS)
def test_paired_intervals_reproduce_where_a_claude_leg_scores_above_rampnet(split, board, committed):
    assert M.split_result(split, board) == committed["per_split"][split]


@pytest.mark.parametrize("split", M.SPLITS)
def test_radius_check_and_point_counts_reproduce_on_every_split(split, board, committed):
    got = M.split_result(split, board, bootstrap=False)
    want = {k: v for k, v in committed["per_split"][split].items() if k != "paired_f1_vs_rampnet"}
    assert got == want


def test_the_summary_fields_follow_from_the_per_split_rows(committed):
    shares = [v["gap_share_closed_at_0.044"] for v in committed["per_split"].values()]
    assert committed["gap_share_closed_at_0.044"] == {
        "min": min(shares), "median": round(st.median(shares), 4), "max": max(shares)}
    blob = json.dumps(committed["per_split"], sort_keys=True).encode()
    import hashlib
    assert committed["per_split_sha256"] == hashlib.sha256(blob).hexdigest()
    assert committed["splits"] == list(M.SPLITS)


# --------------------------------------------------------------------------- #
# the write-up quotes the artifact
# --------------------------------------------------------------------------- #
def _claims(committed):
    """(split, leg) cells where the leg's F1 is above RampNet's."""
    out = []
    for split, v in committed["per_split"].items():
        for leg in M.LEGS:
            if v["paired_f1_vs_rampnet"][leg]["rampnet_minus_leg_f1"] < 0:
                out.append((split, leg))
    return out


def test_the_four_cells_where_claude_scores_above_rampnet_are_the_ones_the_doc_names(committed):
    assert sorted(_claims(committed)) == sorted([
        ("bayonne", M.OPUS), ("bayonne", M.FABLE),
        ("laurens_gsv", M.OPUS), ("laurens_mapillary", M.OPUS)])


def test_the_paired_table_matches_and_reads_each_interval_correctly(committed, doc):
    for split, leg in _claims(committed):
        v = committed["per_split"][split]
        p = v["paired_f1_vs_rampnet"][leg]
        lo, hi = p["ci95"]
        reading = (f"{NAMES[leg]} wins" if p["ci_excludes_zero"]
                   else "point estimate only; the interval includes zero")
        row = (f"| `{split}` | {NAMES[leg]} | {v['scores'][M.RAMPNET]['0.022']['f1']:.3f} "
               f"| {v['scores'][leg]['0.022']['f1']:.3f} | {_signed(p['rampnet_minus_leg_f1'])} "
               f"[{_signed(lo)}, {_signed(hi)}] | {reading} |")
        assert row in doc, row
    # Which of them are wins is the point of the table.
    wins = {(s, leg) for s, leg in _claims(committed)
            if committed["per_split"][s]["paired_f1_vs_rampnet"][leg]["ci_excludes_zero"]}
    assert wins == {("bayonne", M.OPUS), ("laurens_mapillary", M.OPUS)}


def test_rampnets_ten_leads_over_opus_all_clear_zero_and_the_narrowest_is_quoted(committed, doc):
    leads = {s: v["paired_f1_vs_rampnet"][M.OPUS] for s, v in committed["per_split"].items()
             if v["paired_f1_vs_rampnet"][M.OPUS]["rampnet_minus_leg_f1"] > 0}
    assert len(leads) == 10
    assert all(p["ci95"][0] > 0 for p in leads.values())
    s, p = min(leads.items(), key=lambda kv: kv[1]["ci95"][0])
    assert f"the narrowest is {s}, {_signed(p['rampnet_minus_leg_f1'])} " \
           f"[{_signed(p['ci95'][0])}, {_signed(p['ci95'][1])}]" in doc
    fable = {s: v["paired_f1_vs_rampnet"][M.FABLE] for s, v in committed["per_split"].items()
             if v["paired_f1_vs_rampnet"][M.FABLE]["rampnet_minus_leg_f1"] > 0}
    above = [s for s, p in fable.items() if p["ci95"][0] > 0]
    assert len(above) == 11
    (odd,) = set(fable) - set(above)
    q = fable[odd]["ci95"]
    assert f"on `{odd}` it is {_signed(fable[odd]['rampnet_minus_leg_f1'])} " \
           f"[{_signed(q[0])}, {_signed(q[1])}]" in doc


def test_the_radius_table_is_the_artifact(committed, doc):
    rows = re.findall(r"^\| (\w+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| "
                      r"([\N{MINUS SIGN}-]?\d+)% \| ([\d.]+) \| ([\d.]+) \|$", doc, flags=re.M)
    assert [r[0] for r in rows] == list(M.SPLITS)
    for split, g1, g2, g3, share, fo, ff in rows:
        v = committed["per_split"][split]
        g = v["opus_minus_fable_gap"]
        assert (g1, g2, g3) == tuple(f"{g[k]:+.3f}" for k in ("0.022", "0.033", "0.044")), split
        assert int(share.replace("\N{MINUS SIGN}", "-")) == round(100 * v["gap_share_closed_at_0.044"])
        assert fo == f"{v['scores'][M.OPUS]['0.044']['f1']:.3f}"
        assert ff == f"{v['scores'][M.FABLE]['0.044']['f1']:.3f}"


def test_the_radius_prose_is_the_artifact(committed, doc):
    summ = committed["gap_share_closed_at_0.044"]
    lo = round(100 * summ["min"])
    assert f"closes by **{_signed(lo / 100)[0]}{abs(lo)}% to {round(100 * summ['max'])}% of its " \
           f"0.022 value, median {round(100 * summ['median'])}%**" in doc.replace("\n", " ")
    shrink = [s for s, v in committed["per_split"].items()
              if v["opus_minus_fable_gap"]["0.044"] < v["opus_minus_fable_gap"]["0.022"]]
    assert len(shrink) == 12
    (grows,) = set(M.SPLITS) - set(shrink)
    assert grows == "clovis"
    g = committed["per_split"]["clovis"]["opus_minus_fable_gap"]
    assert f"the gap grows from {_signed(g['0.022'])} to {_signed(g['0.044'])}" in doc.replace("\n", " ")
    g44 = {s: v["opus_minus_fable_gap"]["0.044"] for s, v in committed["per_split"].items()}
    lo_s, hi_s = min(g44, key=g44.get), max(g44, key=g44.get)
    assert all(x > 0 for x in g44.values())
    assert f"by {g44[lo_s]:.3f} ({lo_s}) to {g44[hi_s]:.3f} ({hi_s}), median " \
           f"{st.median(g44.values()):.3f}" in doc.replace("\n", " ")


def test_the_emitted_and_scored_point_counts(committed, doc):
    pts = committed["per_split"]["annapolis"]["points"]
    o, f = pts[M.OPUS], pts[M.FABLE]
    text = doc.replace("\n", " ")
    assert f"{o['emitted']} Opus 5.5 points against {f['emitted']} Fable 5.1 points " \
           f"(+{round(100 * (o['emitted'] / f['emitted'] - 1))}%)" in text
    assert f"{o['scored']} and {f['scored']} of them are scored" in text


# --------------------------------------------------------------------------- #
# the ledger figures the write-up quotes
# --------------------------------------------------------------------------- #
def _by_model(ledger, model):
    return [r for r in ledger if r["model_id"] == model]


def test_per_pano_cost_ranges_and_bayonne_costs(ledger, doc):
    text = doc.replace("\n", " ")
    opus = _by_model(ledger, "claude-opus-5-5")
    fable = _by_model(ledger, "claude-fable-5-1")
    per = lambda r: r["est_cost_usd"] / r["panos_called"]  # noqa: E731
    assert f"Opus 5.5 **${min(map(per, opus)):.4f}–{max(map(per, opus)):.4f}**" in text
    # "its first 11 legs of this campaign": every Fable 5.1 row but bayonne, which ran last
    # (ten city splits plus manual_gold; annapolis is #156's leg and has no tagged row).
    first = [r for r in fable if r["bundle"] != "bayonne"]
    assert len(first) == 11
    assert f"Fable 5.1 **${min(map(per, first)):.3f}–{max(map(per, first)):.3f}** on its first 11 " \
           f"legs" in text
    bay = {r["model_id"]: r for r in ledger if r["bundle"] == "bayonne"}
    assert f"**${per(bay['claude-fable-5-1']):.3f}** on bayonne" in text
    assert f"bayonne's ${bay['claude-opus-5-5']['est_cost_usd']:.2f} on 2026-10-20" in text
    assert f"bayonne's ${bay['claude-fable-5-1']['est_cost_usd']:.2f} on 2026-10-20" in text


def test_per_call_token_and_thinking_figures(ledger, doc):
    text = doc.replace("\n", " ")
    assert {r["input_tokens"] / r["calls"] for r in ledger} == {2033.0}
    assert "2,033 input tokens per call" in text
    out_per_leg = [r["output_tokens"] / r["calls"] for r in ledger]
    assert f"is {round(min(out_per_leg))}–{round(max(out_per_leg))} tokens per call on every leg" in text
    stats = {}
    for model in ("claude-opus-5-5", "claude-fable-5-1"):
        rows = _by_model(ledger, model)
        calls = sum(r["calls"] for r in rows)
        th = [r["thoughts_tokens"] / r["calls"] for r in rows]
        stats[model] = {
            "out": sum(r["output_tokens"] for r in rows) / calls,
            "think": sum(r["thoughts_tokens"] for r in rows) / calls,
            "lo": min(th), "hi": max(th), "median": st.median(th),
            "share": sum(r["thoughts_tokens"] for r in rows) / sum(r["output_tokens"] for r in rows)}
    o, f = stats["claude-opus-5-5"], stats["claude-fable-5-1"]
    assert f"{round(o['out'])} per call for Opus 5.5 and {round(f['out'])} for Fable 5.1" in text
    assert f"is {round(o['think'])} per call for Opus 5.5 ({round(o['lo'])}–{round(o['hi'])} by leg, " \
           f"about {round(100 * o['share'])}% of its output)" in text
    assert f"{f['think']:.1f} per call for Fable 5.1 ({f['lo']:.1f}–{round(f['hi'])} by leg, median " \
           f"{f['median']:.1f}, about {round(100 * f['share'])}% of its output), about " \
           f"{round(o['think'] / f['think'])}x less" in text
    ends = sorted((r["model_id"], r["bundle"]) for r in ledger
                  if r["stop_reasons"].get("end_turn"))
    assert ends == [("claude-opus-5-5", "clovis"), ("claude-opus-5-5", "manual_gold")]
    assert all(r["stop_reasons"].get("end_turn", 0) <= 1 for r in ledger)


def test_the_juneau_estimate_follows_from_the_measured_per_pano_cost(ledger, doc):
    """The legs were not run; the estimate is 125 panos at the campaign's per-pano range."""
    with open(os.path.join(REPO, "benchmark", "juneau", "records.jsonl"), encoding="utf-8") as fh:
        n = sum(1 for line in fh if line.strip())
    assert n == 125
    per = lambda r: r["est_cost_usd"] / r["panos_called"]  # noqa: E731
    o = [per(r) for r in _by_model(ledger, "claude-opus-5-5")]
    f = [per(r) for r in _by_model(ledger, "claude-fable-5-1")]
    text = doc.replace("\n", " ")
    assert f"about **${n * min(o):.2f}–{n * max(o):.2f}** for Opus 5.5" in text
    assert f"**${n * min(f):.2f}–{n * max(f):.2f}** for Fable 5.1" in text
    assert f"**${n * (min(o) + min(f)):.2f}–{n * (max(o) + max(f)):.2f}** together" in text
    assert not [r for r in ledger if r["bundle"] == "juneau"], "the juneau legs have run; update the doc"
