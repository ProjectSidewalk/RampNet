"""Juneau (#254): the reviewed bundle, its provenance, and its registration as a pooled
US split.

CPU only, committed inputs only. The byte check against the panos runs only where
``benchmark/juneau/panos/`` exists (a developer checkout); on CI the record-level
checks still run.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import juneau_254 as J  # noqa: E402

BUNDLE = os.path.join(REPO, "benchmark", "juneau")


def test_the_bundle_verifies():
    problems, _, facts = J.verify_bundle(BUNDLE)
    assert problems == [], problems
    assert facts == {"panos": 125, "strata": {"top": 5, "random": 95, "empty": 25},
                     "detections_055": 217}


def test_the_gpu_checks_rederive_from_committed_caches():
    """checks.json is the replication control + parity record of the klone run; it must stay
    what the committed op_caches say (CPU only: both caches are committed)."""
    import argparse
    assert J.cmd_checks(argparse.Namespace(write=False, print=False)) == 0
    res = J.build_checks()
    rc, par = res["replication_control"], res["juneau_parity"]
    assert rc["peaks_only_new"] == rc["peaks_only_committed"] == 0
    assert rc["peaks_same_cell"] == 374
    assert par["records"] == par["cache"] == 217 and par["panos_differing"] == []


def test_the_review_says_who_made_it_and_how_sure_they_were():
    with open(os.path.join(BUNDLE, "verdicts.json"), encoding="utf-8") as f:
        v = json.load(f)
    notes = v["review_notes"]
    assert notes["reviewer"] == "jonf" and notes["confidence"] == "medium"
    assert notes["reviewed_at"] == "2026-10-19"
    assert any(c.startswith("Distant ramps.") for c in notes["caveats"])
    assert any(c.startswith("Diagonal corner ramps.") for c in notes["caveats"])
    assert len(v["panos"]) == 125


def test_the_diagonal_ramp_ruling_is_in_the_rubric_and_says_when_it_was_made():
    with open(os.path.join(REPO, "benchmark", "RUBRICS.md"), encoding="utf-8") as f:
        rubric = f.read()
    row = next(line for line in rubric.splitlines() if "diagonal corner ramp" in line)
    assert "2026-10-19" in row and "not re-checked" in row


def test_juneau_is_pooled_everywhere_the_registries_are_kept():
    import low_floor_sweep as L
    import miss_decomposition as M
    from export_benchmark import BENCHMARK_SPLITS
    from rampnet import eval as E
    assert "juneau" in L.US_SPLITS and "juneau" not in L.HELD_OUT
    assert "juneau" in M.US_SPLITS and M.TIER["juneau"] == "mapillary"
    assert "juneau" in E.POOLED_SPLITS and "juneau" in E.PINNED_SPLITS
    assert "juneau" in BENCHMARK_SPLITS


def test_every_pano_lands_in_a_known_imagery_tier():
    import low_floor_sweep as L
    tiers = {}
    for r in J.load_records("juneau"):
        p = r["pano"]
        t = L.tier_of(p.get("camera_make"), p.get("camera_model"), p.get("source"))
        tiers[t] = tiers.get(t, 0) + 1
    assert tiers == {"survey": 120, "action-modern": 5}


def test_provenance_points_at_the_durable_archive_and_the_run_hash():
    with open(os.path.join(BUNDLE, "bundle_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["run"]["results_jsonl_sha256"] == (
        "5f7822fb3f977d741c6b6f5d2139bdf1e54c7b5052dd73f1e061ad57c5446ee4")
    assert prov["run"]["results_record_count"] == 4999


def test_every_record_is_mapillary_with_a_credit():
    for r in J.load_records("juneau"):
        p = r["pano"]
        assert p["source"] == "mapillary"
        assert p["copyright"] and p["license"] == "CC-BY-SA-4.0"
