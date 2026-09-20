"""Guards on the aggregated scoreboard (docs/model_scoreboard.md).

A summary table is a special kind of liability: it is the page people quote, and it is
the page furthest from the data that produced it. So the assertions here are about the
ways a summary goes wrong rather than about arithmetic —

- it silently disagrees with the detailed log it summarizes (spot-checked against the
  numbers committed in docs/model_comparison.md),
- it pools a split the log says must not be pooled,
- it grows a private copy of the split registry that drifts from everyone else's,
- it drops a model that was actually run, or reports a partial run as a complete one,
- it goes stale after a re-export and nothing fails.

Pure: reads only committed bundles, committed published detections, and manual_labels.
No cache, no GPU, no credentials, no network.
"""
import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import low_floor_sweep as lfs  # noqa: E402
import scoreboard as sb  # noqa: E402
import scoreboard_render as sr  # noqa: E402
from rampnet import roster  # noqa: E402


@pytest.fixture(scope="module")
def board():
    """Score every (model, split) pair once; ~3 s, shared by the whole module."""
    return sb.build()


def _cell(board, model, split):
    return board["per_split"][model][split]


def _summary(board, model):
    return next(m for m in board["models"] if m["model"] == model)


# --------------------------------------------------------------------------- #
# the registry — one copy, not three
# --------------------------------------------------------------------------- #
def test_split_registry_is_imported_not_copied():
    """A private split list here would drift from low_floor_sweep's without failing.

    test_registries_agree_with_low_floor_sweep already ties miss_decomposition to that
    registry; this extends the same contract to the scoreboard, which is the module most
    likely to be tempted into 'just the nine cities'.
    """
    assert sb.US_SPLITS is lfs.US_SPLITS
    assert sb.ALL_SPLITS is lfs.ALL_SPLITS
    assert sb.HELD_OUT is lfs.HELD_OUT


def test_budapest_is_never_pooled(board):
    """docs/model_comparison.md: budapest's numbers 'must not be pooled with the US
    splits or averaged into a headline'. The headline row must obey that literally."""
    for model in board["models"]:
        assert "budapest_district5" not in model["pooled_splits"]
        assert "sao_paulo" not in model["pooled_splits"]
        assert "manual_gold" not in model["pooled_splits"]
    assert board["pooled_splits"] == list(lfs.US_SPLITS)


def test_every_held_out_split_states_why(board):
    """Same contract HELD_OUT carries everywhere else: an omission must be explained,
    because an unexplained one is indistinguishable from a withheld result."""
    for split in set(board["all_splits"]) - set(board["pooled_splits"]):
        assert board["held_out"].get(split), f"{split} held out with no documented reason"


# --------------------------------------------------------------------------- #
# agreement with the detailed log
# --------------------------------------------------------------------------- #
# (model, split) -> (P, R, F1) as committed in docs/model_comparison.md.
#
# This is a hand-copy, and test_every_number_matches_model_comparison below reads the log
# itself — so this dict is deliberately NOT the contract, it is a regression pin. Its job
# is to name a handful of numbers explicitly, in a form that is readable in a diff and
# that does not depend on the parser. When the #132 seam wrap moved the challengers, this
# dict went stale exactly as a hand-copy does; the full check caught the same thing across
# all 88 rows. Keep it small, and update it from the log, never from the scorer.
PUBLISHED = {
    ("rampnet", "richmond"): (0.964, 0.768, 0.855),
    ("rampnet", "annapolis"): (0.973, 0.738, 0.839),
    ("rampnet", "budapest_district5"): (0.874, 0.510, 0.644),
    ("gemini-3.1-pro-preview", "paterson"): (0.852, 0.567, 0.681),
    ("Qwen/Qwen3-VL-32B-Instruct", "budapest_district5"): (0.433, 0.043, 0.079),
    # These two moved with the #132 seam wrap (0.462/0.457/0.460 and 0.025/0.908/0.049
    # before it): one detection each sat across the 360 seam from its ground truth.
    ("allenai/Molmo2-8B", "morgantown"): (0.466, 0.461, 0.463),
    ("google/owlv2-large-patch14-ensemble", "clovis"): (0.025, 0.913, 0.049),
    # The two arms that arrived with #126 and #122, at the same no-floor operating
    # point their own write-ups report.
    ("mask2former-vistas-curb-cut", "richmond"): (0.411, 0.697, 0.517),
    ("mask2former-vistas-curb-cut+curb", "richmond"): (0.126, 0.648, 0.210),
    # The #126 resolution-parity arm, published under #163: the numbers the
    # "Resolution parity" section reports for the 1024x1024 re-run (274/442/36).
    ("mask2former-vistas-curb-cut-1024x1024", "richmond"): (0.383, 0.884, 0.534),
}


@pytest.mark.parametrize("key,expected", sorted(PUBLISHED.items(), key=lambda kv: str(kv[0])))
def test_matches_the_numbers_committed_in_model_comparison(board, key, expected):
    model, split = key
    cell = _cell(board, model, split)
    got = (cell["precision"], cell["recall"], cell["f1"])
    for name, want, have in zip(("P", "R", "F1"), expected, got):
        assert have == pytest.approx(want, abs=0.0006), \
            f"{model} on {split}: {name} {have:.4f} vs published {want}"


def test_rampnet_manual_gold_row_is_the_published_gate(board):
    """manual_gold is exported at the 0.05 floor, so RampNet's headline row only lands on
    the published gold-set numbers if the deployed 0.55 threshold is actually applied.
    Drop the operating point and this reads 0.723/0.935 instead."""
    cell = _cell(board, "rampnet", "manual_gold")
    assert cell["precision"] == pytest.approx(0.947, abs=0.0006)
    assert cell["recall"] == pytest.approx(0.873, abs=0.0006)
    assert cell["f1"] == pytest.approx(0.908, abs=0.0006)


def test_yolo_rows_are_at_the_preregistered_threshold(board):
    """The #71 protocol fixes the YOLO headline at conf 0.25 before any benchmark
    contact. At the 0.05 export floor these same rows read 0.660 and 0.786."""
    assert _cell(board, "y11l_pano", "richmond")["f1"] == pytest.approx(0.595, abs=0.0006)
    assert _cell(board, "y11x_pano_h200", "manual_gold")["f1"] == pytest.approx(0.851, abs=0.0006)


def test_ap_is_read_full_range_not_truncated_at_the_operating_point(board):
    """AP integrates the whole sweep, so it must come from the unthresholded run.

    RampNet's manual_gold bundle is exported at 0.05, so its AP is already untruncated
    (0.917); if the operating point leaked into the AP column it would fall to the
    0.55-and-above slice.
    """
    assert _cell(board, "rampnet", "manual_gold")["ap"] == pytest.approx(0.917, abs=0.0006)


def test_rampnet_city_ap_comes_from_the_low_floor_cache(board):
    """The city bundles stop at 0.55, so an AP computed from them is a truncated curve.

    Read that way richmond is 0.763 and the pooled figure is 0.677 — which puts RampNet
    BELOW the YOLO arms on AP, an artifact of the floor rather than a result. The #54
    re-extraction carries the same run down to 0.05, which is where every other scored
    model is exported.

    The pooled figure was 0.720 over seven splits; laurens_mapillary is the eighth, and
    it is the split RampNet does worst on, so the truncated pooled AP falls further.
    """
    cell = _cell(board, "rampnet", "richmond")
    assert cell["ap_source"] == "op_cache (0.05 floor)"
    assert cell["ap"] == pytest.approx(0.876, abs=0.002)
    assert cell["ap_bundle"] == pytest.approx(0.763, abs=0.0006)
    # ...and the pooled truncated figure is the one that inverts the ordering.
    assert _summary(board, "rampnet")["ap_bundle"] == pytest.approx(0.677, abs=0.0006)


def test_the_substitution_is_scoped_to_actual_truncation(board):
    """manual_gold's bundle is already at 0.05, so it must keep its own AP.

    Swapping the cache in there would trade a flip-TTA export for a no-TTA one (0.917 ->
    0.904) — a different change from un-truncating a curve, and it would leave one row's
    AP and its P/R/F1 describing two different inference configurations.
    """
    assert _cell(board, "rampnet", "manual_gold")["ap_source"] == "bundle"


def test_ap_ordering_is_not_an_artifact_of_the_floor(board):
    """With both read at a 0.05 floor, RampNet's AP leads the supervised arms clearly."""
    rampnet = _summary(board, "rampnet")["ap"]
    best_yolo = max(_summary(board, m)["ap"]
                    for m in ("y11l_pano", "y11x_pano_h200", "y26_pano"))
    assert rampnet > best_yolo + 0.1, f"RampNet AP {rampnet:.3f} vs best YOLO {best_yolo:.3f}"


def test_threshold_marks_reproduce_the_published_operating_point_table(board):
    """The PR figure's marked points must agree with docs/operating_point.md.

    That document's pooled row is P 0.9594 / R 0.6864 / F1 0.8003 at the deployed 0.55
    and 0.8991 raw precision at 0.30. Computed here from the same committed cache by a
    different code path, so a drift in either is a real disagreement.

    These were 0.964 / 0.722 / 0.826 and 0.900 over seven splits. Registering
    laurens_mapillary as the eighth moved them: precision is untouched (0.964 -> 0.959),
    the recall drop is laurens' own 0.390 entering the pool. Both documents are on the
    eight-split basis; if one is ever re-pooled without the other, this fails.
    """
    marks = board["curves"]["rampnet"]["marks"]
    assert marks["0.55"]["precision"] == pytest.approx(0.9594, abs=0.0006)
    assert marks["0.55"]["recall"] == pytest.approx(0.6864, abs=0.0006)
    assert marks["0.55"]["f1"] == pytest.approx(0.8003, abs=0.0006)
    assert marks["0.30"]["precision"] == pytest.approx(0.8991, abs=0.0006)


def test_only_score_carrying_models_get_a_curve(board):
    """A chat VLM has one operating point, not a curve — it must not get a fake one."""
    curves = board["curves"]
    assert "rampnet" in curves and "google/owlv2-large-patch14-ensemble" in curves
    for scoreless in ("gemini-3.1-pro-preview", "Qwen/Qwen3-VL-32B-Instruct",
                      "allenai/Molmo2-8B"):
        assert scoreless not in curves
    # ...and a leg that has not run every pooled split cannot be pooled into one.
    assert "mask2former-vistas-curb-cut" not in curves


# --------------------------------------------------------------------------- #
# the whole log, not a spot-check
# --------------------------------------------------------------------------- #
# model_comparison.md's row labels -> roster published names. Both documents score the
# same committed detections, so a disagreement is a bug in one of them, never a choice.
_LOG_ROW_NAMES = {
    "rampnet": "rampnet",
    "rampnet @0.55": "rampnet",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.7-flash": "gemini-3.7-flash",
    "molmo2-8B (points)": "allenai/Molmo2-8B",
    "Qwen3-VL-32B-Instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "owlv2-large-patch14-ensemble": "google/owlv2-large-patch14-ensemble",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "mask2former-vistas-curb-cut": "mask2former-vistas-curb-cut",
    "mask2former-vistas-curb-cut+curb": "mask2former-vistas-curb-cut+curb",
    # Effort-pinned legs carry the effort in the row label, because one model id is
    # two legs and the bare id would name whichever ran last (see roster.published_as).
    "claude-opus-5 (effort low)": "claude-opus-5-effort-low",
}
# (model, split) pairs the log prints and this page deliberately does not carry, each with
# the reason. Anything else missing is a failure, not an exemption.
_LOG_ROWS_NOT_ON_THE_BOARD = {
    ("gemini-3.1-pro-preview", "manual_gold"):
        "manual_gold detections never published (docs/model_scoreboard.md, 'What is missing')",
    ("gemini-3.6-flash", "manual_gold"):
        "manual_gold detections never published (docs/model_scoreboard.md, 'What is missing')",
}
_LOG_TABLE_HEADER = "| model | P | R | F1 | AP | tp/fp/fn |"
_LOG_SPLIT_HEADING = re.compile(
    r"^(?:\*\*(\w+)\*\*|#+ Result: (\w+)) \(\d[\d,]* (?:reviewed )?panos")


def _parse_model_comparison():
    """Every (split, model, P, R, F1, AP) row in docs/model_comparison.md's result tables.

    Reads the log rather than a transcription of it, so this test cannot pass by agreeing
    with a stale copy of the numbers it is supposed to be checking.
    """
    path = os.path.join(REPO, "docs", "model_comparison.md")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    def value(cell):
        cell = cell.replace("**", "").replace("*", "").strip()
        return None if cell in ("–", "-", "—", "") else float(cell)

    rows, split, in_table = [], None, False
    for lineno, line in enumerate(lines, 1):
        heading = _LOG_SPLIT_HEADING.match(line.strip())
        if heading:
            split = heading.group(1) or heading.group(2)
            continue
        if line.strip() == _LOG_TABLE_HEADER:
            in_table = True
            continue
        if not in_table:
            continue
        cells = line.split("|")[1:-1]
        if not line.startswith("|") or len(cells) != 6:
            in_table = False
            continue
        label = cells[0].replace("**", "").replace("*", "").strip()
        if set(label) <= set("-: "):
            continue
        assert label in _LOG_ROW_NAMES, (
            f"docs/model_comparison.md:{lineno}: unrecognized model row {label!r}. Add it "
            f"to _LOG_ROW_NAMES so this row is checked rather than skipped.")
        rows.append((split, _LOG_ROW_NAMES[label], value(cells[1]), value(cells[2]),
                     value(cells[3]), value(cells[4]), lineno))
    return rows


def test_the_log_parser_actually_finds_the_tables():
    """A parser that silently matches nothing would make the check below vacuous.

    docs/model_comparison.md gets restructured often; if a heading or column layout
    changes, this fails loudly instead of the real test passing on an empty list.
    """
    rows = _parse_model_comparison()
    assert len(rows) >= 85, f"only parsed {len(rows)} rows out of model_comparison.md"
    assert {s for s, *_ in rows} == set(lfs.ALL_SPLITS)


def test_every_number_matches_model_comparison(board):
    """Every P/R/F1/AP in the log's per-split tables, against this page's scorer.

    The spot-check above pins nine hand-picked cells. This pins all of them, in both
    directions: a number edited in either document without re-running fails here.

    RampNet's AP is the one deliberate difference — the log prints the bundle AP, which is
    truncated at the deployed 0.55, and this page substitutes the low-floor cache. That is
    checked too, against ``ap_bundle``, so the exception cannot quietly widen into
    "RampNet's AP does not have to agree with anything".
    """
    mismatches = []
    for split, model, P, R, F1, AP, lineno in _parse_model_comparison():
        cell = board["per_split"].get(model, {}).get(split)
        if cell is None:
            if (model, split) in _LOG_ROWS_NOT_ON_THE_BOARD:
                continue
            mismatches.append(f"model_comparison.md:{lineno} {model}/{split}: in the log, "
                              "absent from the scoreboard")
            continue
        for metric, want, have in (("P", P, cell["precision"]), ("R", R, cell["recall"]),
                                   ("F1", F1, cell["f1"])):
            if want is not None and abs(want - have) > 0.0006:
                mismatches.append(f"model_comparison.md:{lineno} {model}/{split} {metric}: "
                                  f"log {want} vs scoreboard {have:.4f}")
        if AP is not None:
            # The log always prints the bundle AP. So must ap_bundle -- including on the
            # rows where the page then substitutes the low-floor cache.
            if cell["ap_bundle"] is None or abs(AP - cell["ap_bundle"]) > 0.0011:
                mismatches.append(f"model_comparison.md:{lineno} {model}/{split} AP: "
                                  f"log {AP} vs bundle {cell['ap_bundle']}")
    assert not mismatches, "\n".join(mismatches)


def test_only_rampnets_ap_is_allowed_to_differ_from_the_log(board):
    """Scope the exception: every other model's AP must be the bundle AP unchanged.

    Without this, a future substitution applied to some other arm would sail through the
    test above, because that test compares the log to ``ap_bundle`` rather than to what
    the page actually prints.
    """
    for model, cells in board["per_split"].items():
        for split, cell in cells.items():
            if model == "rampnet":
                continue
            assert cell["ap_source"] == "bundle", \
                f"{model}/{split} AP was substituted; only RampNet's may be"
            assert cell["ap"] == cell["ap_bundle"]


def test_the_ap_provenance_table_shows_the_logs_number(board):
    """The page's reconciliation table has to quote the log, not a rounded memory of it."""
    table = sr.ap_provenance_table(board)
    assert "AP in `model_comparison.md`" in table
    for split, log_ap in (("richmond", "0.763"), ("paterson", "0.681"),
                          ("manual_gold", "0.917")):
        row = next(l for l in table.splitlines() if l.startswith(f"| `{split}` |"))
        assert log_ap in row, f"{split} row does not carry the log's AP {log_ap}: {row}"
    # manual_gold is the row that must NOT be substituted, and must say so.
    gold = next(l for l in table.splitlines() if l.startswith("| `manual_gold` |"))
    assert "bundle" in gold and "0.05" in gold


# --------------------------------------------------------------------------- #
# coverage — a run that happened must appear; one that didn't must not be invented
# --------------------------------------------------------------------------- #
def test_the_provenance_table_does_not_claim_an_untruncated_bundle_it_has_not_got(board):
    """Three AP provenances, not two — and the third one used to be mislabelled.

    The table had a binary: substituted means "op_cache, was truncated", anything else
    means "bundle, already at 0.05, flip-TTA export". That held while manual_gold was
    the only unsubstituted row. `laurens_gsv` is the third case — a held-out split that
    never went through the #54 re-extraction, so there is no cache to swap in and its
    bundle is truncated at the deployed 0.55 like every other city's. The old table
    printed "already at 0.05" against it, which is a false provenance claim on the one
    page whose job is provenance, and no test would have caught it.
    """
    table = sr.ap_provenance_table(board)

    gold = next(l for l in table.splitlines() if l.startswith("| `manual_gold` |"))
    assert "already at 0.05" in gold
    assert _cell(board, "rampnet", "manual_gold")["bundle_floor"] < 0.4

    gsv = next(l for l in table.splitlines() if l.startswith("| `laurens_gsv` |"))
    assert "already at 0.05" not in gsv,         "laurens_gsv has no op_cache and a 0.55 bundle; it is truncated, not low-floor"
    assert "truncated" in gsv and "no `op_cache`" in gsv
    cell = _cell(board, "rampnet", "laurens_gsv")
    assert cell["ap_source"] == "bundle"
    assert cell["bundle_floor"] >= 0.4,         "a city bundle is exported at the deployed 0.55, so its floor cannot be low"
    assert cell["ap"] == cell["ap_bundle"]


def test_scores_every_registered_leg(board):
    """The roster is the source of truth for who has been run, so every entry is scored.

    Hardcoding a roster here is how gemini-3.7-flash sat published-but-unscored for a
    while. Reading rampnet.roster instead means a new leg reaches this page by being
    registered, which is the same act that publishes its detections.
    """
    scored = {m["model"] for m in board["models"]}
    registered = {roster.published_name(c) for c in roster.ROSTER}
    assert scored == registered, f"registered but not scored: {registered - scored}"


def test_no_published_file_is_left_unscored(board):
    """A detections file no roster entry claims would be silently invisible here.

    test_roster.py asserts the same thing from the registry side; this is the scoreboard
    refusing to present a partial view as a complete one if the two ever drift.
    """
    assert board["unregistered_exports"] == []


def test_a_pinned_leg_loads_its_own_detections(board):
    """claude-opus-5 at two efforts is two legs sharing one label.

    They publish under different stems and must not resolve to the same file — loading by
    label alone would give both rows whichever file won, and the two would read as
    identical results rather than as the effort comparison they are.
    """
    low = _cell(board, "claude-opus-5-effort-low", "annapolis")
    high = _cell(board, "claude-opus-5-effort-high", "annapolis")
    assert low["f1"] != high["f1"]
    assert low["f1"] == pytest.approx(0.588, abs=0.0006)


def test_the_annapolis_displacement_does_not_survive_pooling(board):
    """#139's whole result, pinned: the one split that flattered Opus was the one split.

    The doc-currency tests below catch a *forgotten* regeneration. They pass happily on a
    regenerated wrong number, because the doc and the JSON would both move together. This
    is the assertion that has to be edited deliberately, so it names the claim instead of
    the artifact: on annapolis Opus leads by +0.021; pooled over the eight city splits the
    two are within 0.01 of each other (0.568 vs 0.575, Gemini ahead); and the split-to-split
    spread is what swamps the lead.

    History, because the numbers moved once already: on the seven-split board this was
    0.588 vs 0.608, a -0.021 deficit with 3 wins of 7. Registering laurens_mapillary as the
    eighth pooled split (#151) -- the split where Opus leads Gemini by the most, +0.086 --
    pulled the gap to -0.007 and the wins to 4 each. The claim that survives both boards is
    "the annapolis lead was noise", not "Opus trails"; the win count is deliberately not
    pinned, because it is the number most likely to churn with the next split.
    """
    opus = _summary(board, "claude-opus-5-effort-low")
    gem = _summary(board, "gemini-3.1-pro-preview")

    assert opus["coverage"] == "8/8" and opus["complete"] is True
    assert opus["n_splits_run"] == 11         # 8 pooled + laurens_gsv + budapest + sao_paulo
    assert opus["f1"] == pytest.approx(0.568, abs=0.0006)
    assert opus["precision"] == pytest.approx(0.562, abs=0.0006)
    assert opus["recall"] == pytest.approx(0.586, abs=0.0006)
    assert gem["f1"] == pytest.approx(0.575, abs=0.0006)

    # The ranking claim itself, not just the two numbers behind it: Gemini still tops the
    # table, and by less than the 0.01 the doc calls a tie.
    assert opus["f1"] < gem["f1"], "gemini-3.1-pro is no longer the best challenger"
    assert abs(opus["f1"] - gem["f1"]) < 0.01
    assert _cell(board, "claude-opus-5-effort-low", "annapolis")["f1"] > \
        _cell(board, "gemini-3.1-pro-preview", "annapolis")["f1"]

    # ...and the reason the lead did not generalise: the per-split gaps span a range far
    # wider than the annapolis margin, and both signs occur. The doc quotes the range and
    # the largest single gap as multiples of the lead, so both statistics are named here.
    deltas = {s: _cell(board, "claude-opus-5-effort-low", s)["f1"] - _cell(board, gem["model"], s)["f1"]
              for s in opus["pooled_splits"]}
    assert any(d > 0 for d in deltas.values()) and any(d < 0 for d in deltas.values()), deltas
    spread = max(deltas.values()) - min(deltas.values())
    assert spread == pytest.approx(0.156, abs=0.001), deltas
    assert spread > 7 * abs(deltas["annapolis"])
    assert max(abs(d) for d in deltas.values()) == pytest.approx(0.086, abs=0.001)
    assert max(deltas, key=deltas.get) == "laurens_mapillary"
    # "smaller than six of the other seven": every pooled gap but morgantown's exceeds it.
    assert sum(abs(d) > abs(deltas["annapolis"])
               for s, d in deltas.items() if s != "annapolis") == 6, deltas

    # The pooled scope matters: the held-out laurens_gsv is a wider gap still, and the
    # doc must say "of the eight pooled splits" for 0.086 and name laurens_gsv (+0.158)
    # as the largest on the whole board -- not call 0.086 the board's largest.
    city_deltas = {s: _cell(board, "claude-opus-5-effort-low", s)["f1"]
                   - _cell(board, gem["model"], s)["f1"]
                   for s in board["city_splits"]}
    assert max(city_deltas, key=lambda s: abs(city_deltas[s])) == "laurens_gsv"
    assert city_deltas["laurens_gsv"] == pytest.approx(0.158, abs=0.001)
    assert sum(d > 0 for d in city_deltas.values()) == 6      # six of eleven overall
    with open(sb.DEFAULT_DOC, encoding="utf-8") as f:
        prose = re.sub(r"\s+", " ", f.read())          # the doc wraps at 90 columns
    assert "+0.086, the largest gap of the eight pooled splits" in prose
    assert "laurens_gsv, by +0.158, the largest gap on the whole board" in prose
    assert "the largest gap on the board)" not in prose
    assert "smaller in magnitude than six of the other seven" in prose

    # Opus is the highest-recall chat VLM that has run the full pool -- the axis
    # operating_point.md says to optimize, and the reason the near-tie is not a wash.
    pooled_vlms = [m for m in board["models"]
                   if m["class"] == "chat-vlm" and m["complete"]]
    assert max(pooled_vlms, key=lambda m: m["recall"])["model"] == "claude-opus-5-effort-low"


def test_partial_coverage_is_reported_not_averaged_away(board):
    """The two Gemini legs have city detections but no published manual_gold.

    The failure this guards is the quiet one: an aggregate over four cities printed in
    the same column as one over seven. Coverage travels with the row instead.
    """
    for model in ("gemini-3.1-pro-preview", "gemini-3.6-flash"):
        summary = _summary(board, model)
        assert summary["coverage"] == "8/8"
        assert summary["complete"] is True
        assert summary["manual_gold_f1"] is None
        assert "manual_gold" not in board["per_split"][model]


def test_single_split_legs_stay_out_of_the_pooled_tables(board):
    """Vistas ran richmond only; three of the four Claude legs ran annapolis only.

    claude-opus-5-effort-low is deliberately NOT in this set any more: #139 took it to
    nine splits and #151 to both Laurens arms, so it is a complete leg and belongs in
    the pooled tables. If it ever reappears here, a leg lost coverage rather than a
    test needing a nudge.

    A one-city macro-mean in the pooled column would be read as an eight-city one. It
    is computed (the number is real, for that city) but must not reach the headline
    table or the pooled column of the matrix.

    Coverage is pinned per leg rather than as one number, so a leg that gains a split
    without reaching the full pool is noticed here rather than averaged in. Partial
    coverage still means excluded; the point of this test is the exclusion, not the
    number.
    """
    want_coverage = {
        "mask2former-vistas-curb-cut": "1/8",
        "mask2former-vistas-curb-cut+curb": "1/8",
        "mask2former-vistas-curb-cut-1024x1024": "1/8",     # #163, richmond only
        "claude-opus-5-effort-high": "1/8",
        "claude-sonnet-5-effort-low": "1/8",
        "claude-sonnet-5-effort-high": "1/8",
        # #156, annapolis only. Both clear the expansion gate, so these may well
        # grow past 1/8 later; until they do, partial coverage means excluded.
        "claude-fable-5-1-effort-low-anthropic": "1/8",
        "claude-fable-5-effort-low-anthropic": "1/8",
    }
    single = [m for m in board["models"] if not m["complete"]]
    assert {m["model"] for m in single} == set(want_coverage)
    for m in single:
        assert m["coverage"] == want_coverage[m["model"]], m["model"]

    headline = sr.headline_table(board)
    matrix = sr.by_split_table(board)
    for m in single:
        assert m["display"] not in headline, f"{m['display']} is in the pooled headline"
        assert m["display"] in matrix, f"{m['display']} vanished from the matrix"
        # ...and its POOLED cell is blank rather than a one-city mean. Checked by column
        # position, not by substring: a one-split leg's pooled mean equals its single
        # city cell, so searching the row for that value matches the legitimate one.
        row = next(l for l in matrix.splitlines() if l.startswith(f"| {m['display']} |"))
        cells = [c.strip() for c in row.split("|")]
        pooled = cells[2 + len(lfs.US_SPLITS)]
        assert pooled == "–", f"{m['display']} pooled cell is {pooled!r}, expected a dash"


def test_partial_table_names_the_split_every_number_came_from(board):
    table = sr.partial_table(board)
    assert "`richmond`" in table and "`annapolis`" in table
    assert "Claude Opus 5 (high)" in table
    assert "Mask2Former Vistas (curb cut)" in table
    # The low-effort leg went to nine splits in #139, so it is pooled now, not partial.
    assert "Claude Opus 5 (low)" not in table


_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
          9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}


def test_the_prose_beside_the_partial_table_agrees_with_it(board):
    """The generator rewrites the tables and nothing else. The sentences around them
    are hand-written, and when the Fable legs landed (#156) the table gained two rows
    that beat the leg the prose still called the strongest challenger, while the leg
    count it quoted stayed at six. (S2 on PR #157.)

    Two checks, both against the board rather than against any number typed here:
    the count of legs outside the pooled tables, and which leg is the best challenger
    on annapolis -- the split every Claude leg has run, and the one the prose singles
    out.
    """
    with open(sb.DEFAULT_DOC, encoding="utf-8", newline="") as fh:
        doc = fh.read()
    partial = [m for m in board["models"] if not m["complete"]]
    one_split = [m for m in partial if m["n_splits_run"] == 1]
    more = [m for m in partial if m["n_splits_run"] > 1]
    if not more:
        # Every partial leg is one-split (the state since #139 completed Opus-low),
        # and the sentence says exactly that.
        want = f"{_WORDS[len(partial)]} legs have run one split each"
    else:
        # A breakdown, which has to add up. The first version of this test pinned
        # the total only, and the sentence said "six of them one split each"
        # beside a three-split leg: six plus one is seven.
        want = (f"{_WORDS[len(partial)]} legs have not run the pooled splits — "
                f"{_WORDS[len(one_split)].lower()} of them one split each")
    assert doc.count(want) == 2, (
        f"expected {want!r} twice in docs/model_scoreboard.md (the section opener and "
        f"the 'What is missing' bullet); the board has {len(partial)} partial legs, "
        f"{len(one_split)} of them one-split, {[m['model'] for m in more]} with more")

    best = max((m for m in board["models"] if m["model"] != "rampnet"
                and "annapolis" in board["per_split"][m["model"]]),
               key=lambda m: board["per_split"][m["model"]]["annapolis"]["f1"])
    claim = re.search(r"\*\*(.+?) is the strongest challenger measured on annapolis\*\*", doc)
    assert claim, "the strongest-challenger sentence is gone"
    # Whole model name, not a prefix: "Claude Fable 5" is a prefix of "Claude Fable
    # 5.1 (low, anthropic)", and the two are 0.001 F1 apart, so a prefix match
    # would stay green through the one flip that is actually plausible.
    assert best["display"].split(" (")[0] == claim.group(1).split(" at ")[0], (
        f"the doc names {claim.group(1)!r}; the board says {best['display']!r} "
        f"(F1 {board['per_split'][best['model']]['annapolis']['f1']:.3f})")


def test_a_leg_from_an_unmapped_provider_is_classified_not_dropped():
    """A roster entry whose provider predates this module still has to reach the board."""
    leg = roster.Challenger(spec="newthing:x", label="x", provider="newthing",
                            density=None, standing=False, added="2026-01-01", note="")
    assert sb.class_of(leg) == "unclassified"
    assert sb.display_of(leg) == "x"


def test_every_roster_provider_has_a_class_and_an_operating_point():
    """A provider missing from PROVIDER_CLASS falls to 'unclassified' silently, which is
    the safe failure but the wrong one to ship — catch it here instead."""
    providers = {c.provider for c in roster.ROSTER}
    missing = providers - set(sb.PROVIDER_CLASS)
    assert not missing, f"roster providers with no model class: {sorted(missing)}"
    for klass in sb.CLASS_ORDER:
        assert klass in sb.OPERATING_POINT
        assert klass in sb.OPERATING_POINT_NOTE
        assert klass in sb.CLASS_LABEL
    for klass in sb.PROVIDER_CLASS.values():
        assert klass in sb.CLASS_ORDER


# --------------------------------------------------------------------------- #
# the splice — prose survives, numbers are replaced
# --------------------------------------------------------------------------- #
def test_splice_replaces_only_the_generated_block():
    doc = ("prose above\n\n"
           + sr.BEGIN.format(name="headline") + "\nstale\n" + sr.END.format(name="headline")
           + "\n\nprose below\n")
    out = sr.splice(doc, {"headline": "| fresh |"})
    assert "prose above" in out and "prose below" in out
    assert "stale" not in out and "| fresh |" in out
    assert sr.splice(out, {"headline": "| fresh |"}) == out    # idempotent


def test_splice_ignores_a_block_the_doc_does_not_want():
    doc = "just prose\n"
    assert sr.splice(doc, {"headline": "| x |"}) == doc


def test_committed_doc_is_current(board):
    """The whole point of generating the tables: staleness is a test failure.

    Re-run `python scripts/analysis/scoreboard.py` if this fails.
    """
    with open(sb.DEFAULT_DOC, encoding="utf-8", newline="") as fh:
        current = fh.read()
    assert sr.splice(current, sr.render_tables(board)) == current, \
        "docs/model_scoreboard.md is stale — re-run scripts/analysis/scoreboard.py"


# --------------------------------------------------------------------------- #
# the committed JSON — the artifact nothing used to check
# --------------------------------------------------------------------------- #
def test_committed_json_is_current(board):
    """analysis_out/scoreboard.json is committed, so it can go stale like the doc can.

    Compared as bytes, which also pins the LF endings: a value-level compare passes
    happily on a Windows-written CRLF file, which is the trap json_artifacts keep hitting
    (imagery_manifest, the usage ledger).
    """
    with open(sb.DEFAULT_JSON, "rb") as fh:
        on_disk = fh.read()
    assert on_disk == sr.json_payload(board).encode("utf-8"), \
        "analysis_out/scoreboard.json is stale — re-run scripts/analysis/scoreboard.py"


def test_the_committed_json_has_no_crlf():
    """Stated separately from the byte-compare so a failure says which defect it is."""
    with open(sb.DEFAULT_JSON, "rb") as fh:
        assert b"\r\n" not in fh.read()


def test_the_json_is_reproducible_across_environments():
    """No float in the committed JSON may carry more digits than the writer rounds to.

    A byte-compare of full-precision floats is not a portable check: AP comes out of
    numpy, and a different numpy build shifts the last bits, so the file differed between
    CI's 3.10 and 3.12 while every value was the same. Rounding at the writer is what
    makes the artifact reproducible; this asserts the file on disk actually went through
    it, because a hand-edit or an older writer would not show up in any value comparison.
    """
    def deepest(value):
        if isinstance(value, float):
            text = repr(value)
            return len(text.split(".")[1]) if "." in text and "e" not in text else 0
        if isinstance(value, dict):
            return max((deepest(v) for v in value.values()), default=0)
        if isinstance(value, list):
            return max((deepest(v) for v in value), default=0)
        return 0

    with open(sb.DEFAULT_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert deepest(payload) <= sr.JSON_PRECISION, \
        "a full-precision float reached the committed JSON; it will not survive a " \
        "different numpy build"


def test_write_json_pins_lf_on_the_writer_not_on_git(tmp_path, board):
    """Round-trip through the actual writer, checked on bytes.

    Reading the file back and comparing values would pass on a CRLF write — that is how
    this class of defect keeps surviving (imagery_manifest, the usage ledger). The
    committed artifact has to be right regardless of the contributor's autocrlf setting.
    """
    out = tmp_path / "nested" / "scoreboard.json"
    sr.write_json(str(out), board)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw)["pooled_splits"] == list(lfs.US_SPLITS)


def test_the_json_does_not_carry_the_curve_point_arrays(board):
    """The plot-only arrays are ~120k points; serialized they were 7.7 MB of committed
    artifact, 98% of the file, that no reader diffs and the figures rebuild in seconds.

    What the page cites — AP, RampNet's marked thresholds, the point count — stays.
    """
    payload = json.loads(sr.json_payload(board))
    assert payload["curves"], "the curves block itself must survive"
    for name, curve in payload["curves"].items():
        assert "recalls" not in curve and "precisions" not in curve, \
            f"{name}: point arrays leaked back into the committed JSON"
        assert curve["ap"] is not None and curve["n_points"] > 0
    assert payload["curves"]["rampnet"]["marks"]["0.55"]["f1"] == pytest.approx(0.8003,
                                                                               abs=0.0006)
    # In-memory, the figures still get the full curves.
    assert len(board["curves"]["rampnet"]["recalls"]) == \
        payload["curves"]["rampnet"]["n_points"]
    assert len(sr.json_payload(board)) < 200_000, "committed JSON is bloated again"


# --------------------------------------------------------------------------- #
# the CLI — --check has to cover what it claims, --models must not clobber
# --------------------------------------------------------------------------- #
def _run_scoreboard(*args):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "analysis", "scoreboard.py"), *args],
        capture_output=True, text=True, cwd=REPO)


def test_check_fails_on_a_stale_json(tmp_path):
    """--check used to verify only the doc while its help said 'doc and JSON'.

    A falsified headline F1 in the committed JSON passed with exit 0, which made the one
    artifact the .gitignore re-include exists for the one artifact nothing validated.
    """
    payload = json.loads(open(sb.DEFAULT_JSON, encoding="utf-8").read())
    payload["models"][0]["f1"] = 0.111
    stale = tmp_path / "scoreboard.json"
    stale.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")
    done = _run_scoreboard("--check", "--json-out", str(stale))
    assert done.returncode != 0, "a falsified JSON passed --check"
    assert "stale" in done.stdout


def test_a_models_subset_does_not_touch_the_committed_page(tmp_path):
    """`--models y11x_pano_h200` used to splice a ONE-ROW headline into the committed doc.

    That left YOLO11x bolded as the winner in every column, directly above prose reading
    "RampNet wins by 0.221 F1" — and then --check reported the real page as stale, so the
    signal inverted. It also crashed in the figures afterwards, half-written.
    """
    doc_before = open(sb.DEFAULT_DOC, "rb").read()
    json_before = open(sb.DEFAULT_JSON, "rb").read()
    done = _run_scoreboard("--models", "y11x_pano_h200", "--no-figures")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "left alone" in done.stdout
    assert open(sb.DEFAULT_DOC, "rb").read() == doc_before
    assert open(sb.DEFAULT_JSON, "rb").read() == json_before
    # ...but naming a destination explicitly still writes a partial board there.
    out = tmp_path / "subset.json"
    done = _run_scoreboard("--models", "y11x_pano_h200", "--no-figures",
                           "--json-out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    assert [m["model"] for m in json.loads(out.read_text())["models"]] == ["y11x_pano_h200"]


def test_check_refuses_a_subset():
    """--check on a subset would compare a partial board against the full page and
    report the page as stale. That is a false alarm, so it is refused instead."""
    done = _run_scoreboard("--check", "--models", "rampnet")
    assert done.returncode != 0
    assert "drop --models" in done.stdout


def test_a_figure_helper_survives_a_board_without_rampnet(board, tmp_path):
    """The subset guard protects the committed files; these two protect the process.

    Both used to raise StopIteration on any board with no RampNet row, after main() had
    already rewritten the doc and the JSON.
    """
    from scoreboard_figures import fig_by_split, fig_headline
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib
    matplotlib.use("Agg")

    trimmed = dict(board)
    trimmed["models"] = [m for m in board["models"] if m["model"] != "rampnet"]
    trimmed["per_split"] = {k: v for k, v in board["per_split"].items() if k != "rampnet"}
    out = str(tmp_path / "fig.png")
    fig_headline(trimmed, out, plt)
    fig_by_split(trimmed, out, plt)
