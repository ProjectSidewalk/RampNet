"""Tests for the matched-operating-point read (scripts/analysis/operating_point_parity_51.py).

Two things here are load-bearing and everything else is arithmetic.

The first is the CONTROL. RampNet is swept from ``op_cache`` while its published row is
bundle-derived, and the two extractions disagree slightly by construction. The script is
only trustworthy while that disagreement stays small, so the control is tested in both
directions: it holds on the committed data, and it actually fails when the published
value moves.

The second is that SELECTION NEVER TOUCHES A REPORTED SPLIT. That is the entire claim to
fairness; if a dev split ever leaks into ``US_SPLITS`` the numbers become tune-on-test
and mean nothing. It is asserted rather than trusted to review.
"""

import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "analysis", "operating_point_parity_51.py")
JSON_PATH = os.path.join(ROOT, "docs", "data", "operating_point_parity_51.json")

REPORT = """\
Bundle: benchmark/fake  (7 scored panos)  match radius 0.022
model                                     P          95% CI      R          95% CI     F1     AP       tp/fp/fn/ign
-------------------------------------------------------------------------------------
faux_leg                              0.920   (0.871-0.952)  0.523   (0.467-0.578)  0.667  0.773       162/14/148/8

[faux_leg] threshold sweep (re-scored from cached detections)
    thr      P      R     F1         tp/fp/fn
   0.05  0.789  0.823  0.806        255/68/55 <- best F1
   0.10  0.855  0.758  0.803        235/40/75
   0.25  0.920  0.523  0.667       162/14/148

PR curves written to /somewhere/else

--- RampNet verdict-based cross-check ---
threshold   kept  precision   recall
     0.55    247      0.960    0.765
"""


def _load():
    spec = importlib.util.spec_from_file_location("operating_point_parity_51", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def built(mod):
    return mod.artifact()


@pytest.fixture(scope="module")
def committed():
    with open(JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_sweep_row_regex_accepts_the_best_f1_marker(mod):
    m = mod.SWEEP_ROW_RE.match("   0.05  0.789  0.823  0.806        255/68/55 <- best F1")
    assert m and m.group("thr") == "0.05" and m.group("f1") == "0.806"


def test_sweep_row_regex_accepts_a_plain_row(mod):
    m = mod.SWEEP_ROW_RE.match("   0.25  0.920  0.523  0.667       162/14/148")
    assert m and m.group("tp") == "162" and m.group("fn") == "148"


def test_sweep_row_regex_rejects_the_headline_row(mod):
    """The operating-point table has a 4-part tp/fp/fn/ign and must not be read as a sweep."""
    line = ("faux_leg                              0.920   (0.871-0.952)  "
            "0.523   (0.467-0.578)  0.667  0.773       162/14/148/8")
    assert mod.SWEEP_ROW_RE.match(line) is None


def test_parse_sweeps_reads_every_row_not_just_the_best(mod, tmp_path):
    p = tmp_path / "fake_tiles.txt"
    p.write_text(REPORT, encoding="utf-8")
    sweeps = mod.parse_sweeps(str(p))
    assert set(sweeps) == {"faux_leg"}
    assert sorted(sweeps["faux_leg"]) == [0.05, 0.10, 0.25]
    assert sweeps["faux_leg"][0.05]["f1"] == 0.806


def test_parse_sweeps_stops_at_the_next_section(mod, tmp_path):
    """The RampNet cross-check block below the sweep has a similar shape; if the
    section boundary is not honoured its rows land in the leg's curve."""
    p = tmp_path / "fake_tiles.txt"
    p.write_text(REPORT, encoding="utf-8")
    sweeps = mod.parse_sweeps(str(p))
    assert 0.55 not in sweeps["faux_leg"], "cross-check rows leaked into the sweep"


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_select_threshold_picks_the_f1_argmax(mod):
    per_split = {"dev": {0.05: {"f1": 0.70}, 0.10: {"f1": 0.81}, 0.25: {"f1": 0.66}}}
    assert mod.select_threshold(per_split, "dev") == 0.10


def test_select_threshold_breaks_ties_toward_the_higher_threshold(mod):
    per_split = {"dev": {0.10: {"f1": 0.80}, 0.30: {"f1": 0.80}}}
    assert mod.select_threshold(per_split, "dev") == 0.30


def test_select_threshold_is_none_for_an_unswept_split(mod):
    assert mod.select_threshold({"dev": {0.1: {"f1": 1.0}}}, "absent") is None


def test_macro_at_is_a_macro_mean(mod):
    """Each split weighted equally -- a split with 100x the panos must not dominate."""
    per_split = {s: {0.1: {"p": 1.0, "r": 1.0, "f1": 1.0}} for s in mod.US_SPLITS}
    per_split[mod.US_SPLITS[0]][0.1] = {"p": 0.0, "r": 0.0, "f1": 0.0}
    out = mod.macro_at(per_split, 0.1)
    assert out["f1"] == pytest.approx(6 / 7)


def test_macro_at_refuses_a_partial_pool(mod):
    per_split = {s: {0.1: {"p": 1.0, "r": 1.0, "f1": 1.0}} for s in mod.US_SPLITS[:-1]}
    assert mod.macro_at(per_split, 0.1) is None


def test_rampnet_grid_spans_the_cache_floor_to_the_top(mod):
    grid = mod.rampnet_grid()
    assert grid[0] == 0.05 and grid[-1] == 0.95
    assert all(round(b - a, 10) == 0.05 for a, b in zip(grid, grid[1:]))


# --------------------------------------------------------------------------- #
# the fairness invariant
# --------------------------------------------------------------------------- #
def test_no_candidate_dev_split_is_ever_reported_over(mod):
    assert set(mod.NON_POOLED).isdisjoint(mod.US_SPLITS)
    assert mod.DEFAULT_DEV in mod.NON_POOLED


def test_every_non_pooled_split_is_a_candidate(mod):
    """A split quietly dropped from the candidate list would make the sensitivity
    table look more stable than it is."""
    assert set(mod.NON_POOLED) == set(mod.ALL_SPLITS) - set(mod.US_SPLITS)


def test_build_reports_over_the_pooled_splits_only(built, mod):
    assert built["pool"] == list(mod.US_SPLITS)
    assert built["dev_split"] not in built["pool"]


# --------------------------------------------------------------------------- #
# the control, both directions
# --------------------------------------------------------------------------- #
def test_control_agrees_on_the_committed_data(built):
    c = built["control"]
    assert c["agrees"], (
        f"op_cache re-score {c['f1_op_cache']} has drifted from the published "
        f"{c['f1_published_bundle']} by {c['delta']} (tolerance {c['tolerance']})")


def test_control_delta_is_the_known_extraction_difference(built):
    """Documented in the module docstring as -0.0025. If this moves, the docstring's
    explanation is stale and the number in it is wrong."""
    assert built["control"]["delta"] == pytest.approx(-0.0025, abs=0.0003)


def test_control_fails_when_the_published_value_moves(mod, monkeypatch):
    monkeypatch.setattr(mod, "PUBLISHED_RAMPNET_F1", 0.900)
    assert not mod.build()["control"]["agrees"]


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def test_rampnet_and_all_three_yolo_legs_are_scored(built, mod):
    assert set(built["models"]) == {mod.RAMPNET, *mod.LEGS}


def test_every_leg_gains_from_being_given_its_own_operating_point(built):
    """The finding, asserted: nobody is made worse off by parity, so the shrinking gap
    is not an artifact of handicapping RampNet."""
    for name, row in built["models"].items():
        gain = row["pooled"]["f1"] - row["published_point"]["f1"]
        assert gain >= 0, f"{name} scored worse at its selected threshold"


def test_rampnet_still_leads_at_parity(built, mod):
    """The gap shrinks but does not close. If this ever fails the write-up's headline
    is wrong, not this test."""
    rn = built["models"][mod.RAMPNET]["pooled"]["f1"]
    best_yolo = max(built["models"][m]["pooled"]["f1"] for m in mod.LEGS)
    assert rn > best_yolo


def test_at_floor_is_flagged_so_a_lower_bound_is_never_read_as_an_optimum(built, mod):
    for name, row in built["models"].items():
        if row["at_floor"]:
            assert row["selected_threshold"] == 0.05


def test_committed_artifact_is_current(built, committed, mod):
    assert mod.rnd(built) == committed, (
        "docs/data/operating_point_parity_51.json is stale -- re-run the script")
