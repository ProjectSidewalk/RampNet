"""Tests for #135's Run B gate evaluation (scripts/analysis/run_b_gate_135.py).

The gate is a decision rule written before the numbers existed. These tests check that
the implementation is the rule as written — including the branches that did NOT fire,
since an implementation that only ever produces the observed verdict is untestable by
its own output.
"""

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "analysis" / "run_b_gate_135.py"
ARTIFACT = REPO / "docs" / "data" / "run_b_gate_135.json"


def _load():
    spec = importlib.util.spec_from_file_location("run_b_gate_135", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def result(mod):
    return mod.evaluate()


# --------------------------------------------------------------------------------
# the observed outcome
# --------------------------------------------------------------------------------

def test_primary_is_not_significant_at_either_end_of_the_bracket(result):
    """The whole reading rests on this being robust to which s.e. is picked: if the
    favourable end cleared 1.96 the conclusion would depend on an unmeasurable choice."""
    p = result["primary"]
    assert p["delta"] > 0                       # cosine is nominally ahead at ep8
    assert p["z_at_se_lo"] < result["z_critical"]   # ... but not significantly
    assert p["significant"] is False
    assert p["reading"] == "no effect at 8 epochs"


def test_secondary_does_not_confirm_an_arrest(result):
    s = result["secondary"]
    assert s["cosine_change"] < 0, "the cosine arm still declines ep3->ep8"
    assert s["arrested"] is False
    assert s["significantly_smaller"] is False
    assert s["confirms_arrest"] is False


def test_verdict_is_the_judgment_call_branch(result):
    assert result["verdict"] == "JUDGMENT CALL"


def test_epoch_7_is_flagged_as_the_post_hoc_trap(mod, result):
    """Epoch 7 has the largest delta and would clear the threshold at the favourable end
    of the bracket. It is not the pre-registered comparison, and the artifact has to say
    so or someone will quote it as the result."""
    lg = result["largest_absolute_delta"]
    assert lg["epoch"] == 7
    assert lg["epoch"] != mod.PRIMARY_EPOCH
    assert lg["would_reach_significance_at_favourable_se"] is True


# --------------------------------------------------------------------------------
# the branches that did not fire -- the rule, not this dataset
# --------------------------------------------------------------------------------

def test_significant_positive_primary_would_proceed(mod, monkeypatch):
    monkeypatch.setattr(mod, "_max_f1_by_epoch", lambda p: (
        {e: 0.90 for e in range(1, 9)} if "run_a" in str(p)
        else {**{e: 0.90 for e in range(1, 9)}, 8: 0.95}))
    assert mod.evaluate()["verdict"] == "PROCEED"


def test_tie_with_an_arrested_decline_would_proceed(mod, monkeypatch):
    """Primary tie, but the cosine arm's ep3->ep8 change is >= 0 -> the mechanism fired
    even though the endpoint did not move."""
    run_a = {e: 0.92 for e in range(1, 9)}
    run_a[3], run_a[8] = 0.9200, 0.9130          # Run A declines
    cos = dict(run_a)
    cos[8] = 0.9130 + 0.0001                     # tie at ep8 ...
    cos[3] = 0.9120                              # ... but cosine RISES ep3 -> ep8
    monkeypatch.setattr(mod, "_max_f1_by_epoch",
                        lambda p: run_a if "run_a" in str(p) else cos)
    r = mod.evaluate()
    assert r["primary"]["significant"] is False
    assert r["secondary"]["arrested"] is True
    assert r["verdict"] == "PROCEED"


def test_significant_negative_primary_without_arrest_would_not_proceed(mod, monkeypatch):
    run_a = {e: 0.92 for e in range(1, 9)}
    cos = {e: 0.92 for e in range(1, 9)}
    cos[8] = 0.90                                # cosine clearly worse at ep8
    monkeypatch.setattr(mod, "_max_f1_by_epoch",
                        lambda p: run_a if "run_a" in str(p) else cos)
    r = mod.evaluate()
    assert r["primary"]["significant"] is True
    assert r["primary"]["delta"] < 0
    assert r["verdict"] == "DO NOT PROCEED"


# --------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------

def test_standard_error_is_the_measured_bracket_not_an_assumption(result):
    assert "MEASURED" in result["se_source"]
    lo, hi = result["se_bracket"]
    assert 0 < lo < hi

def test_committed_artifact_matches_a_fresh_run(mod):
    """--check is what CI and reviewers run; this is the same assertion."""
    assert ARTIFACT.exists()
    fresh = json.dumps(mod.evaluate(), indent=2, sort_keys=True) + "\n"
    assert ARTIFACT.read_bytes() == fresh.encode(), "artifact drifted from the inputs"


def test_artifact_bytes_are_lf_only(mod):
    """Committed JSON must not pick up CRLF on Windows -- the repo asserts on bytes."""
    assert b"\r\n" not in ARTIFACT.read_bytes()
