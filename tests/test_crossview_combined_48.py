"""The combined cross-family table (#48) re-derives from committed predictions.

Checks medians, fallback rates, point gains, the post hoc column, both screens and the doc
table (fast, no bootstrap); the bootstrap CIs are re-derived by running
``scripts/analysis/crossview_combined_48.py``.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import crossview_align_48 as H  # noqa: E402
import crossview_combined_48 as C  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(C.OUT_JSON), reason="no committed table")


def _table():
    with open(C.OUT_JSON, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw
    return json.loads(raw)


def test_combined_table_rederives():
    tab = _table()
    assert tab["config"]["pairs_sha256"] == H.PAIRS_SHA256
    assert [r["arm"] for r in tab["rows"]] == [a for a, *_ in C.ROWS]
    pairs = H.read_frozen_pairs()
    strata = {"all": list(range(len(pairs))),
              "gsv": [i for i, p in enumerate(pairs) if p["imagery"] == "gsv"],
              "mapillary": [i for i, p in enumerate(pairs) if p["imagery"] == "mapillary"]}
    for row in tab["rows"]:
        errs = H.arm_errors(pairs, C.load(row["arm"]))
        for s, idx in strata.items():
            if s not in row:
                continue
            assert round(float(np.median([errs[i][0] for i in idx])), 4) == \
                pytest.approx(row[s]["median_deg"], abs=1e-4), (row["arm"], s)
            assert float(np.mean([errs[i][3] for i in idx])) == \
                pytest.approx(row[s]["fallback_rate"], abs=1e-4), (row["arm"], s)


def test_richmond_only_arms_are_never_ranked_on_all_pairs():
    """A Richmond-only arm scored over all 300 would fall back on the 240 GSV pairs and be
    ranked as if it had run everywhere; the table must carry the Richmond stratum only."""
    for row in _table()["rows"]:
        if row["arm"] in C.RICHMOND_ONLY:
            assert row["richmond_only"] and set(row) & {"all", "gsv"} == set(), row["arm"]
            assert row["mapillary"]["n"] == 60


# --------------------------------------------------------------------------- #
# Review of #210 (A7): a planted gain change, a fake screen entry or a flipped post hoc flag
# in combined_table.json must fail a test. Each check below is point-estimate only (no
# bootstrap), so it runs in seconds.
# --------------------------------------------------------------------------- #
def _setup():
    pairs = H.read_frozen_pairs()
    auto = H.arm_errors(pairs, H.read_predictions(C.AUTO))
    return pairs, auto, C.strata_for(pairs)


def _med(v):
    return float(np.median(v))


def test_combined_gains_rederive_per_stratum():
    tab = _table()
    pairs, auto, strata = _setup()
    pos = {p["pair_id"]: i for i, p in enumerate(pairs)}
    for row in tab["rows"]:
        preds = C.load(row["arm"])
        errs = H.arm_errors(pairs, preds)
        for s, idx in strata.items():
            if s not in row:
                continue
            b = row[s]
            assert _med([auto[i][0] - errs[i][0] for i in idx]) == \
                pytest.approx(b["gain_vs_auto"], abs=1e-4), (row["arm"], s)
            used = [i for i in idx if not errs[i][3]]
            if "aligned" in b:
                assert b["aligned"]["n"] == len(used)
                assert _med([auto[i][0] - errs[i][0] for i in used]) == \
                    pytest.approx(b["aligned"]["gain_vs_auto"], abs=1e-4), (row["arm"], s)
            if row["arm"] == C.AUTO:
                assert "else_auto" not in row
                continue
            raw = None if preds is None else {pos[k]: v for k, v in preds.items()}
            ea, is_auto = C.else_auto(errs, auto, raw)
            e = row["else_auto"][s]
            assert _med([ea[i][0] for i in idx]) == pytest.approx(e["median_deg"], abs=1e-4)
            assert _med([auto[i][0] - ea[i][0] for i in idx]) == \
                pytest.approx(e["gain_vs_auto"], abs=1e-4), (row["arm"], s)
            assert float(np.mean([is_auto[i] for i in idx])) == \
                pytest.approx(e["auto_prior_share"], abs=1e-4), (row["arm"], s)


def test_hybrids_auto_share_is_counted():
    """roma_warp_hyb never falls back, but 126 of its 300 answers are the auto prior; the
    common-rule column must show it, and put it level with roma_warp's fallback rate."""
    rows = {r["arm"]: r for r in _table()["rows"]}
    hyb, warp = rows["roma_warp_hyb"], rows["roma_warp"]
    assert hyb["all"]["fallback_rate"] == 0
    assert hyb["else_auto"]["all"]["auto_prior_share"] == pytest.approx(126 / 300, abs=1e-4)
    assert warp["else_auto"]["all"]["auto_prior_share"] == pytest.approx(126 / 300, abs=1e-4)
    assert hyb["else_auto"]["all"]["median_deg"] == warp["else_auto"]["all"]["median_deg"]


def test_post_hoc_column_matches_the_roster():
    tab = _table()
    for row, (arm, _, post_hoc, _) in zip(tab["rows"], C.ROWS):
        assert row["arm"] == arm
        assert row["post_hoc"] is bool(post_hoc), arm
        assert row["post_hoc_why"] == (post_hoc or None), arm
    # found post hoc from git history by the review of #210; must not silently revert
    must = {"mapa_posed_pair", "mapa_posed_poseonly", "mapa_mono_depthonly", "sfm_colmap",
            "mono_da3_hcal", "mono_unidepth_point", "roma_local", "roma_warp", "roma_warp_hyb"}
    assert must <= {r["arm"] for r in tab["rows"] if r["post_hoc"]}


def test_gsv_screen_entries_rederive():
    """Every uncorrected screen entry must have its recorded (positive) GSV point gain; a
    planted entry for an arm that loses to auto on GSV (e.g. lg) fails here."""
    tab = _table()
    pairs, auto, strata = _setup()
    idx = strata["gsv"]
    for arm, v in tab["gsv_ci_clear_vs_auto"].items():
        errs = H.arm_errors(pairs, H.read_predictions(arm))
        g = _med([auto[i][0] - errs[i][0] for i in idx])
        assert g > 0 and v["ci"][0] > 0, arm
        assert g == pytest.approx(v["gain_vs_auto"], abs=1e-4), arm


def test_multiplicity_screen_is_consistent():
    tab = _table()
    mt = tab["multiplicity"]
    cfg = mt["config"]
    arms = H.available_predictions()
    assert cfg["n_arms"] == len(arms), "a new predictions/*.jsonl grew the screened family"
    assert cfg["percentile"] == pytest.approx(100 * cfg["alpha"] / cfg["n_arms"], abs=1e-4)
    assert cfg["seed"] == H.SEED and cfg["n_boot"] >= 20000
    pairs, auto, strata = _setup()
    for s in ("gsv", "all"):
        for arm, v in mt[s].items():
            errs = H.arm_errors(pairs, H.read_predictions(arm))
            g = _med([auto[i][0] - errs[i][0] for i in strata[s]])
            assert g == pytest.approx(v["gain_vs_auto"], abs=1e-4), (s, arm)
            assert v["survives"] is (v["bonferroni_lower"] > 0), (s, arm)
        assert mt[f"{s}_survivors"] == sorted(a for a, v in mt[s].items() if v["survives"])
    assert set(mt["gsv_survivors"]) <= set(tab["gsv_ci_clear_vs_auto"])


def test_doc_table_is_the_committed_table():
    doc = os.path.join(os.path.dirname(HERE), "docs", "crossview_align_48.md")
    with open(doc, encoding="utf-8") as f:
        assert C.markdown(_table()) in f.read()
