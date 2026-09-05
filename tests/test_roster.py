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
# The docs say what the registry says
# --------------------------------------------------------------------------- #
def test_the_roster_table_in_the_docs_matches_the_registry():
    """The anti-drift mechanism for the docs is this test, not anyone's discipline.
    Before #122 the roster count was hand-written in four files and two of them
    disagreed — "all 8" in model_comparison.md against "seven-model roster" in
    replication.md — while a ninth leg sat published in the repo."""
    doc = (REPO / "docs" / "model_comparison.md").read_text("utf-8")
    assert roster.TABLE_MARKER in doc, (
        "the generated roster table is gone from docs/model_comparison.md; "
        "regenerate it with `python -m rampnet.roster`")
    after = doc.split(roster.TABLE_MARKER, 1)[1].lstrip("\n")
    table = "\n".join(_table_lines(after))
    assert table == roster.markdown_table(), (
        "docs/model_comparison.md's roster table is stale; regenerate with "
        "`python -m rampnet.roster`")


def _table_lines(text):
    for line in text.splitlines():
        if not line.startswith("|"):
            return
        yield line


def test_no_doc_still_hardcodes_the_old_roster_count():
    """These exact phrases were the drift. Catch them coming back.

    Whitespace is collapsed first, deliberately: every one of these was wrapped
    across a line break in the prose, so a naive substring check finds none of them
    and passes while the docs are still wrong.
    """
    import re
    # Each entry is a phrase that was actually in the docs and wrong. Two properties
    # matter and neither is obvious:
    #  * No entry may be a prefix of another -- "all 8" and "all 8 model groups"
    #    both shipped, and the shorter can never fail independently, so the longer
    #    reads as coverage it does not add.
    #  * The count has to be bound to the roster, or the guard fires on perfectly
    #    good prose. "all 8" alone would reject a future "all 8 splits"; the splits
    #    are a different axis and there are ten of them.
    stale = (r"all 8 (?:model|challenger|zero-shot)", r"all 8(?! splits| cities)",
             r"8-model roster", r"seven-model roster", r"8 model groups")
    docs = ("model_comparison.md", "replication.md", "curb_ramp_data_sourcing.md")
    for name in docs:
        text = re.sub(r"\s+", " ", (REPO / "docs" / name).read_text("utf-8"))
        for phrase in stale:
            hit = re.search(phrase, text)
            assert hit is None, f"docs/{name} still says {hit.group(0)!r}"
    # The analysis README carries per-model prose and this PR edits it, so it is in
    # scope for the same rot even though it is not under docs/.
    readme = REPO / "scripts" / "analysis" / "README.md"
    if readme.exists():
        text = re.sub(r"\s+", " ", readme.read_text("utf-8"))
        for phrase in stale:
            hit = re.search(phrase, text)
            assert hit is None, f"scripts/analysis/README.md still says {hit.group(0)!r}"


# --------------------------------------------------------------------------- #
# Registry consistency
# --------------------------------------------------------------------------- #
def test_a_bare_spec_names_exactly_one_leg():
    """One spec can name several legs once a pin is involved, but only one of them is
    what running that spec with no extra flags reproduces."""
    default = [c.spec for c in roster.ROSTER if roster.is_default_leg(c)]
    assert len(default) == len(set(default))
    assert set(roster.BY_SPEC) == set(default)


def test_published_names_are_unique():
    """These are filenames. A collision does not raise -- the second leg overwrites
    the first, leaving a file that still looks complete and still passes --verify.
    That is exactly how the two claude-sonnet-5 legs collided (#122)."""
    names = [roster.published_name(c) for c in roster.ROSTER]
    assert len(names) == len(set(names))


def test_a_pinned_leg_is_published_under_a_name_that_says_so():
    """If a pin changes the detections, the filename has to change with it, or the
    directory claims one set of detections is the whole model."""
    for c in roster.ROSTER:
        if not c.pins:
            continue
        assert c.published_as, c.spec
        for _, value in c.pins:
            assert str(value) in c.published_as, (c.published_as, c.pins)


def test_every_leg_of_a_pinned_model_is_qualified():
    """Half-qualified is worse than unqualified: a bare `claude-sonnet-5__annapolis`
    sitting next to `claude-sonnet-5-effort-high__annapolis` reads as the model
    rather than as one of its legs."""
    pinned = {c.label for c in roster.ROSTER if c.pins}
    for c in roster.ROSTER:
        if c.label in pinned:
            assert c.published_as, (c.label, "sibling leg is pinned")


def test_every_published_filename_round_trips_to_its_model_and_city():
    """Published detections are ``{slug(label)}__{city}.json``, read back by splitting
    on the LAST ``__``. Most labels are HF ids whose ``/`` already slugs to ``__``
    (``Qwen__Qwen3-VL-8B-Instruct__richmond.json``), so containing the separator is
    the convention, not a bug — what must hold is that the split still recovers the
    right city, which fails the moment a *city* name gains a ``__``."""
    import re
    cities = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
              "gainesville", "budapest_district5", "sao_paulo", "manual_gold")
    for c in roster.ROSTER:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", c.label)
        for city in cities:
            model_part, _, city_part = f"{slug}__{city}".rpartition("__")
            assert city_part == city, (c.label, city)
            assert model_part == slug, (c.label, city)


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
# The registry and the published detections are the same statement
# --------------------------------------------------------------------------- #
DETECTIONS = REPO / "benchmark" / "model_detections"


def _published_files():
    return sorted(p.name for p in DETECTIONS.glob("*__*.json"))


def test_every_published_detections_file_belongs_to_a_registered_leg():
    """The registry claims to be every model the benchmark knows about. This is that
    claim, checked against the directory rather than against prose.

    It caught the gap it was written for: the registry covered 78 of 112 files. The
    three supervised YOLO arms (30 files, #51) and the four annapolis Claude legs (4
    files, #122) were published, scored and written up while the one place that is
    supposed to list them said nothing about them.
    """
    known = {roster.slug(roster.published_name(c)) for c in roster.ROSTER}
    orphans = sorted({f.rsplit("__", 1)[0] for f in _published_files()} - known)
    assert not orphans, (
        "published detections with no roster entry: " + ", ".join(orphans) +
        " -- add them to rampnet/roster.py")


def test_every_registered_leg_has_published_detections():
    """The other direction. An entry with no files is a model someone meant to run,
    or a name that drifted from the filename it is supposed to predict."""
    missing = [roster.published_name(c) for c in roster.PUBLISHED
               if not list(DETECTIONS.glob(roster.published_filename(c, "*")))]
    assert not missing, "registered but nothing published: " + ", ".join(missing)


def test_rampnet_is_the_only_roster_member_without_detections():
    """It is read from each bundle's committed records.jsonl and carries no detector
    signature, which is why it is the one row with no file here."""
    assert {c.spec for c in roster.ROSTER} - {c.spec for c in roster.PUBLISHED} == {"rampnet"}


def test_each_published_file_names_the_leg_it_says_it_is():
    """The filename, the `published_as` recorded inside, and the registry all have to
    agree, or a file can be renamed into a different leg's identity."""
    for name in _published_files():
        city = name.rpartition("__")[2][:-len(".json")]
        entry = next((c for c in roster.ROSTER
                      if roster.published_filename(c, city) == name), None)
        assert entry is not None, name
        payload = json.loads((DETECTIONS / name).read_text("utf-8"))
        assert payload["model"] == entry.label, name
        assert payload.get("published_as", entry.label) == roster.published_name(entry), name
        assert payload["signature"]["model_id"] == entry.label, name
        for key, value in entry.pins:
            sig_key = key.split("_", 1)[1]          # claude_effort -> effort
            if sig_key in payload["signature"]:
                assert payload["signature"][sig_key] == value, (name, key)
            else:
                # A pin that is NOT a signature field (claude_serving_path, #156:
                # it changes who bills, not what was asked, so it is deliberately
                # kept out of the cache key). The file still has to name it, or
                # nothing in the published artifact distinguishes a Vertex-served
                # leg from a first-party one. `pins` is absent on files published
                # before that field existed — those legs pin only signature
                # fields, so they never reach this branch.
                assert payload.get("pins", {}).get(key) == value, (name, key)


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

    # Deliberately not guarded by hasattr(): this test spent its whole life skipping
    # because compare.py built its parser inline, so it read as coverage of the most
    # load-bearing invariant here while asserting nothing. If build_parser goes away,
    # this must fail, not skip.
    defaults = {a.dest: a.default for a in compare.build_parser()._actions}
    for key, value in roster.PROVIDER_DEFAULTS.items():
        assert defaults[key] == value, key
