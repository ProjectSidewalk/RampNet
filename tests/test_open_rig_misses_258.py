"""Tests for ``miss_taxonomy.py --cities`` and the open-rig miss report (#263, #258).

These read the committed low-floor caches (``analysis_out/op_cache/``) and
``benchmark/*/records.jsonl``; no GPU, no network, no imagery.

The load-bearing guarantees:

* ``--cities`` with the seven original US splits reproduces the committed #46 pooled
  table (``analysis_out/miss_taxonomy.json``: 124 / 166 / 9 / 128 of 427), so the open-rig
  doc's reference row is the same number #46 published;
* the default run (no ``--cities``) still covers every split with a cache, in registry
  order, and writes no ``cities`` key, so its output is unchanged;
* ``open_rig_misses_258.py --check`` re-derives the committed JSON byte for byte.
"""
import contextlib
import io
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import miss_taxonomy as mt  # noqa: E402
import open_rig_misses_258 as orm  # noqa: E402

SEVEN = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
         "gainesville")
COMMITTED_46 = os.path.join(REPO, "analysis_out", "miss_taxonomy.json")


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mt.main(argv)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------- #
# --cities
# --------------------------------------------------------------------------- #
def test_parse_cities_accepts_commas_and_spaces_in_registry_order():
    assert mt.parse_cities(["gainesville,bend", "richmond"]) == (
        "richmond", "bend", "gainesville")


def test_parse_cities_rejects_an_unknown_split():
    with pytest.raises(SystemExit):
        mt.parse_cities(["richmond,richmnd"])


def test_cities_reproduces_the_committed_46_pool(tmp_path):
    out = tmp_path / "seven.json"
    rc, text = _run(["--cities", ",".join(SEVEN), "--json-out", str(out)])
    assert rc == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    with open(COMMITTED_46, encoding="utf-8") as f:
        want = json.load(f)
    assert got["pooled"]["counts"] == {"merged": 124, "sub_threshold": 166,
                                       "localization": 9, "silent": 128}
    for key in ("pooled", "near_field", "far_field", "matcher_check",
                "merged_separation_px", "sourcing_addressable_recall_pts"):
        assert got[key] == want[key], key
    for city in SEVEN:
        assert got["per_city"][city] == want["per_city"][city], city
    assert got["cities"] == list(SEVEN)
    assert "POOLED, 7 US splits" in text


def test_default_run_is_unrestricted_and_unlabelled(tmp_path):
    out = tmp_path / "default.json"
    rc, text = _run(["--trials", "0", "--json-out", str(out)])
    assert rc == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    assert "cities" not in got
    have_cache = [c for c in mt.ALL_SPLITS
                  if os.path.exists(os.path.join(mt.CACHE_DIR, f"{c}.json"))]
    assert list(got["per_city"]) == have_cache
    # The title predates laurens_mapillary and juneau joining US_SPLITS; it is kept
    # verbatim so the default output stays byte-stable (see the module docstring).
    assert "POOLED, seven US splits" in text


def test_held_out_only_run_has_no_pool(tmp_path):
    out = tmp_path / "held.json"
    rc, text = _run(["--cities", "bayonne", "--trials", "0", "--json-out", str(out)])
    assert rc == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["pooled"] is None and list(got["per_city"]) == ["bayonne"]
    assert "No pooled split" in text


def test_threshold_055_runs_and_only_moves_misses_one_way():
    lo, _ = mt.load_rows("juneau", 0.30)
    hi, _ = mt.load_rows("juneau", 0.55)
    assert len(lo) == len(hi)
    # Raising the threshold can only lose hits.
    assert sum(r["hit"] for r in hi) <= sum(r["hit"] for r in lo)


# --------------------------------------------------------------------------- #
# open_rig_misses_258 pure helpers
# --------------------------------------------------------------------------- #
def test_reading_rule():
    assert orm.reading((0.01, 0.2)).startswith("sub_threshold leads")
    assert orm.reading((-0.3, -0.02)).startswith("silent leads")
    assert orm.reading((-0.1, 0.1)) == "neither leads"
    assert orm.reading((None, None)) == "no misses"


def _row(hit, bucket=None, pano="p"):
    return {"hit": hit, "bucket": bucket, "pano": pano}


def test_bootstrap_is_deterministic_and_brackets_a_pure_population():
    units = {"a": [[_row(False, "silent", "p1")], [_row(False, "silent", "p2")],
                   [_row(True, None, "p3")]]}
    b1 = orm.bootstrap(units, reps=200)
    b2 = orm.bootstrap(units, reps=200)
    assert b1 == b2
    assert b1["silent"] == (1.0, 1.0)
    assert b1["sub_minus_silent"] == (-1.0, -1.0)


def test_separation_stats_without_merged_misses():
    s = orm.separation_stats([_row(True)])
    assert s["n"] == 0 and s["chebyshev_median"] is None


# --------------------------------------------------------------------------- #
# the committed artifact
# --------------------------------------------------------------------------- #
def test_committed_report_rederives_byte_for_byte():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = orm.main(["--check"])
    assert rc == 0, buf.getvalue()


def test_committed_report_reference_row_is_the_46_pool():
    with open(orm.DEFAULT_JSON, encoding="utf-8") as f:
        d = json.load(f)
    ref = d["by_threshold"]["0.30"]["groups"]["reference_7"]
    assert {b: ref["buckets"][b]["n"] for b in mt.BUCKETS} == {
        "merged": 124, "sub_threshold": 166, "localization": 9, "silent": 128}
    # Every open split is either reported or listed as a gap, never dropped.
    for th in d["by_threshold"].values():
        for city in d["open_splits"]:
            assert (city in th["splits"]) != (city in d["gaps"]), city
