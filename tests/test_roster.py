"""The challenger registry (#122): its invariants, and the freeze it exists to hold.

The tests that matter here are the ones about ``WITNESS_POOL_46``. Everything else
in this file is ordinary consistency checking; those are a guard on a finished human
pass, and the failure they prevent is silent — the numbers simply change.
"""
import json
import pathlib

import pytest

from rampnet import roster

REPO = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# The freeze
# --------------------------------------------------------------------------- #
def test_the_46_witness_pool_is_exactly_what_the_human_pass_ran_against():
    """The #46 tagging verdicts are only meaningful against the item list that
    produced them, and that list is a function of this pool. It is written out
    literally in roster.py rather than derived, so that adding a challenger cannot
    move it; this pins the literal."""
    assert roster.WITNESS_POOL_46 == (
        "gemini:gemini-3.6-flash",
        "gemini:gemini-3.1-pro-preview",
        "qwen:Qwen/Qwen3-VL-8B-Instruct",
        "qwen:Qwen/Qwen3-VL-32B-Instruct",
        "molmo:allenai/Molmo2-8B",
        "owlv2",
        "gdino",
    )


def test_the_witness_pool_is_not_derived_from_the_standing_roster():
    """The point of the freeze is that these two can diverge. They happen to be
    equal today; a test that asserted equality would re-create exactly the coupling
    #122 removed, so assert the weaker, permanent property instead."""
    assert set(roster.WITNESS_POOL_46) <= {c.spec for c in roster.ROSTER}


def test_the_committed_silent_witness_artifact_records_the_frozen_pool():
    """An analysis whose item list depends on which models ran has to say which
    models ran, or a verdict file cannot be matched to the pool that made its items."""
    data = json.loads((REPO / "analysis_out" / "silent_witness.json").read_text("utf-8"))
    assert data["models"]["pool"] == "WITNESS_POOL_46"
    assert tuple(data["models"]["specs"]) == roster.WITNESS_POOL_46


def test_the_committed_verdicts_still_match_the_gallery_they_were_made_on():
    """The end of the chain the freeze protects: 50 tagged items, one manifest
    digest. If a roster change ever reaches this, these stop agreeing."""
    base = REPO / "benchmark" / "miss_taxonomy_46"
    verdicts = json.loads((base / "silent__jonf.json").read_text("utf-8"))
    manifest = json.loads((base / "silent_gallery" / "manifest.json").read_text("utf-8"))
    assert verdicts["manifest_digest"] == manifest["digest"]
    assert verdicts["n_items"] == verdicts["n_tagged"] == manifest["n"] == 50
    assert set(verdicts["verdicts"]) == set(manifest["items"])


# --------------------------------------------------------------------------- #
# Registry consistency
# --------------------------------------------------------------------------- #
def test_specs_and_labels_are_unique():
    specs = [c.spec for c in roster.ROSTER]
    labels = [c.label for c in roster.ROSTER]
    assert len(specs) == len(set(specs))
    # Labels are filenames in benchmark/model_detections; a collision would have two
    # models overwriting each other's published detections.
    assert len(labels) == len(set(labels))


def test_every_entry_label_is_what_label_for_resolves():
    """`label` is data used to render the docs table; `label_for` is what the export
    path actually calls. They must not be two answers to one question."""
    for c in roster.ROSTER:
        assert roster.label_for(c.spec) == c.label, c.spec


def test_challengers_are_the_standing_roster_without_rampnet():
    assert "rampnet" not in roster.CHALLENGERS
    assert roster.SCORED_SPECS == ("rampnet",) + roster.CHALLENGERS
    assert all(roster.BY_SPEC[s].standing for s in roster.CHALLENGERS)


def test_sparse_and_dense_partition_the_standing_challengers():
    assert set(roster.SPARSE) | set(roster.DENSE) == set(roster.CHALLENGERS)
    assert not set(roster.SPARSE) & set(roster.DENSE)


def test_the_open_vocabulary_detectors_are_the_dense_ones():
    """Density is measured boxes/pano, not a preference: the open-vocab detectors sit
    at 55-88 against everything else's 1-4."""
    assert set(roster.DENSE) == {"owlv2", "gdino"}


def test_off_roster_entries_are_published_but_not_scored():
    assert roster.OFF_ROSTER, "gemini-3.7-flash should be registered (#120)"
    for c in roster.OFF_ROSTER:
        assert not c.standing
        assert c.spec not in roster.SCORED_SPECS


def test_every_entry_records_when_it_joined():
    for c in roster.ROSTER:
        assert c.added and len(c.added) == 10, c.spec
        assert c.note, c.spec


# --------------------------------------------------------------------------- #
# density_of / partition_by_density refuse to guess
# --------------------------------------------------------------------------- #
def test_density_of_rejects_an_unregistered_model():
    with pytest.raises(KeyError):
        roster.density_of("gemini:not-a-real-model")


def test_density_of_rejects_an_unmeasured_model():
    """The old code filed anything outside SPARSE as dense, silently. Only the sparse
    union feeds a headline, so a wrong guess here moves a published number."""
    unmeasured = roster.Challenger(
        spec="vistas:curb-cut", label="x", provider="vistas", density=None,
        standing=False, added="2026-08-18", note="n/a")
    saved = dict(roster.BY_SPEC)
    roster.BY_SPEC[unmeasured.spec] = unmeasured
    try:
        with pytest.raises(ValueError):
            roster.density_of(unmeasured.spec)
    finally:
        roster.BY_SPEC.clear()
        roster.BY_SPEC.update(saved)


def test_partition_by_density_keeps_the_pools_own_order():
    pool = ("gdino", "gemini:gemini-3.6-flash", "owlv2", "molmo:allenai/Molmo2-8B")
    sparse, dense = roster.partition_by_density(pool)
    assert sparse == ("gemini:gemini-3.6-flash", "molmo:allenai/Molmo2-8B")
    assert dense == ("gdino", "owlv2")


# --------------------------------------------------------------------------- #
# pool_record — the provenance block embedded in analysis artifacts
# --------------------------------------------------------------------------- #
def test_pool_record_names_the_frozen_pool_only_when_it_is_that_pool():
    assert roster.pool_record(roster.WITNESS_POOL_46)["pool"] == "WITNESS_POOL_46"
    # Same members, different order: not the same pool, because the pool is a
    # sequence and the artifact must not claim provenance it does not have.
    shuffled = tuple(reversed(roster.WITNESS_POOL_46))
    assert roster.pool_record(shuffled)["pool"] is None
    assert roster.pool_record(roster.WITNESS_POOL_46 + ("gemini:gemini-3.7-flash",))["pool"] is None


def test_pool_record_is_json_serializable_and_complete():
    rec = roster.pool_record(roster.WITNESS_POOL_46)
    json.dumps(rec)
    assert set(rec) == {"pool", "specs", "labels", "sparse", "dense"}
    assert len(rec["labels"]) == len(rec["specs"])
    assert len(rec["sparse"]) + len(rec["dense"]) == len(rec["specs"])


# --------------------------------------------------------------------------- #
# label_for's override path and the cargs override
# --------------------------------------------------------------------------- #
def test_label_for_follows_an_overridden_provider_model():
    """A run with --gemini-model set must publish under the model actually used, not
    under the registry default, or its detections land in the wrong file."""
    class _Args:
        gemini_model = "gemini-2.5-flash"
    assert roster.label_for("gemini", _Args()) == "gemini-2.5-flash"
    # An explicit model id in the spec still wins over the namespace.
    assert roster.label_for("gemini:gemini-3.6-flash", _Args()) == "gemini-3.6-flash"


def test_label_overrides_win_for_providers_whose_model_id_slot_is_not_a_model():
    saved = dict(roster.LABEL_OVERRIDES)
    roster.LABEL_OVERRIDES["vistas:curb-cut"] = "mask2former-vistas-curb-cut"
    try:
        assert roster.label_for("vistas:curb-cut") == "mask2former-vistas-curb-cut"
    finally:
        roster.LABEL_OVERRIDES.clear()
        roster.LABEL_OVERRIDES.update(saved)


# --------------------------------------------------------------------------- #
# The registry vs. the parsers that consume it
# --------------------------------------------------------------------------- #
def test_provider_defaults_match_compare_pys_parser():
    """The defaults feed build_detector and so the cache key. A drift does not crash;
    it silently misses every already-paid cached detection."""
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "model_comparison"))
    import compare  # noqa: E402

    parser = compare.build_parser() if hasattr(compare, "build_parser") else None
    if parser is None:
        pytest.skip("compare.py builds its parser inline inside main()")
    defaults = {a.dest: a.default for a in parser._actions}
    for key, value in roster.PROVIDER_DEFAULTS.items():
        assert defaults[key] == value, key
