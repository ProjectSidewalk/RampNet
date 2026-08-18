"""Guards on the aggregated scoreboard (docs/scoreboard.md).

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
import os
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
# (model, split) -> (P, R, F1) exactly as committed in docs/model_comparison.md. If the
# scoreboard and the log disagree, one of them is wrong and both are quoted.
PUBLISHED = {
    ("rampnet", "richmond"): (0.964, 0.768, 0.855),
    ("rampnet", "annapolis"): (0.973, 0.738, 0.839),
    ("rampnet", "budapest_district5"): (0.874, 0.510, 0.644),
    ("gemini-3.1-pro-preview", "paterson"): (0.852, 0.567, 0.681),
    ("Qwen/Qwen3-VL-32B-Instruct", "budapest_district5"): (0.433, 0.043, 0.079),
    ("allenai/Molmo2-8B", "morgantown"): (0.462, 0.457, 0.460),
    ("google/owlv2-large-patch14-ensemble", "clovis"): (0.025, 0.908, 0.049),
    # The two arms that arrived with #126 and #122, at the same no-floor operating
    # point their own write-ups report.
    ("mask2former-vistas-curb-cut", "richmond"): (0.411, 0.697, 0.517),
    ("mask2former-vistas-curb-cut+curb", "richmond"): (0.126, 0.648, 0.210),
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

    Read that way richmond is 0.763 and the pooled figure is 0.720 — which puts RampNet
    BELOW the YOLO arms on AP, an artifact of the floor rather than a result. The #54
    re-extraction carries the same run down to 0.05, which is where every other scored
    model is exported.
    """
    cell = _cell(board, "rampnet", "richmond")
    assert cell["ap_source"] == "op_cache (0.05 floor)"
    assert cell["ap"] == pytest.approx(0.876, abs=0.002)
    assert cell["ap_truncated_at_operating_point"] == pytest.approx(0.763, abs=0.0006)


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

    That document's pooled row is P 0.964 / R 0.722 / F1 0.826 at the deployed 0.55 and
    0.900 raw precision at 0.30. Computed here from the same committed cache by a
    different code path, so a drift in either is a real disagreement.
    """
    marks = board["curves"]["rampnet"]["marks"]
    assert marks["0.55"]["precision"] == pytest.approx(0.964, abs=0.0006)
    assert marks["0.55"]["recall"] == pytest.approx(0.722, abs=0.0006)
    assert marks["0.55"]["f1"] == pytest.approx(0.826, abs=0.0006)
    assert marks["0.30"]["precision"] == pytest.approx(0.900, abs=0.0006)


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
# coverage — a run that happened must appear; one that didn't must not be invented
# --------------------------------------------------------------------------- #
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


def test_partial_coverage_is_reported_not_averaged_away(board):
    """The two Gemini legs have city detections but no published manual_gold.

    The failure this guards is the quiet one: an aggregate over four cities printed in
    the same column as one over seven. Coverage travels with the row instead.
    """
    for model in ("gemini-3.1-pro-preview", "gemini-3.6-flash"):
        summary = _summary(board, model)
        assert summary["coverage"] == "7/7"
        assert summary["complete"] is True
        assert summary["manual_gold_f1"] is None
        assert "manual_gold" not in board["per_split"][model]


def test_single_split_legs_stay_out_of_the_pooled_tables(board):
    """Vistas ran richmond only; the Claude legs ran annapolis only.

    A one-city macro-mean in the pooled column would be read as a seven-city one. It is
    computed (the number is real, for that one city) but must not reach the headline
    table or the pooled column of the matrix.
    """
    single = [m for m in board["models"] if not m["complete"]]
    assert {m["model"] for m in single} == {
        "mask2former-vistas-curb-cut", "mask2former-vistas-curb-cut+curb",
        "claude-opus-5-effort-low", "claude-opus-5-effort-high",
        "claude-sonnet-5-effort-low", "claude-sonnet-5-effort-high",
    }
    for m in single:
        assert m["coverage"] == "1/7"

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
    assert "Claude Opus 5 (low)" in table
    assert "Mask2Former Vistas (curb cut)" in table


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
        "docs/scoreboard.md is stale — re-run scripts/analysis/scoreboard.py"
