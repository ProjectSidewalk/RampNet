"""Unit tests for the published challenger detections (#46, replication).

Pure logic — no ``.model_cache``, no torch.

The point of publishing is that a clean clone can score the challengers with neither a
GPU nor the detector stack. So the guarantees are: the label a spec resolves to must be
derivable *without* importing detectors, filenames must survive model ids containing
slashes, and a missing export must be reported as missing rather than as "this model
found nothing".
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import export_model_cache as em  # noqa: E402
from fp_taxonomy import CHALLENGERS, _compare_args  # noqa: E402

ARGS = _compare_args("/tmp/cache")


# --------------------------------------------------------------------------- #
# slug — model ids contain slashes; filenames cannot
# --------------------------------------------------------------------------- #
def test_slashes_become_filesystem_safe():
    assert em.slug("IDEA-Research/grounding-dino-base") == "IDEA-Research__grounding-dino-base"


def test_plain_ids_are_unchanged():
    assert em.slug("gemini-3.6-flash") == "gemini-3.6-flash"


def test_distinct_models_never_collide():
    slugs = {em.slug(em.spec_label(s, ARGS)) for s in CHALLENGERS}
    assert len(slugs) == len(CHALLENGERS)


# --------------------------------------------------------------------------- #
# spec_label — resolvable without importing the detector stack
# --------------------------------------------------------------------------- #
def test_an_explicit_model_id_wins():
    assert em.spec_label("gemini:gemini-3.1-pro-preview", ARGS) == "gemini-3.1-pro-preview"


def test_a_bare_provider_uses_the_documented_default():
    assert em.spec_label("owlv2", ARGS) == "google/owlv2-large-patch14-ensemble"
    assert em.spec_label("gdino", ARGS) == "IDEA-Research/grounding-dino-base"


def test_every_challenger_resolves_to_a_label():
    for spec in CHALLENGERS:
        label = em.spec_label(spec, ARGS)
        assert label and label != spec.split(":")[0] or ":" not in spec


def test_whitespace_is_tolerated():
    assert em.spec_label(" gemini : gemini-3.6-flash ", ARGS) == "gemini-3.6-flash"


# --------------------------------------------------------------------------- #
# load_detections — absent must mean absent
# --------------------------------------------------------------------------- #
def test_a_missing_export_returns_none_not_empty(tmp_path):
    # Returning {} would read as "this model detected nothing anywhere", which would
    # silently zero a model's recall instead of reporting it as unavailable.
    assert em.load_detections("nope", "bend", str(tmp_path)) is None


def test_an_export_round_trips(tmp_path):
    path = tmp_path / f"{em.slug('a/b')}__bend.json"
    path.write_text(json.dumps({"model": "a/b", "city": "bend",
                                "detections": {"p1": [[0.5, 0.6, 0.9]]}}))
    got = em.load_detections("a/b", "bend", str(tmp_path))
    assert got == {"p1": [[0.5, 0.6, 0.9]]}


# --------------------------------------------------------------------------- #
# the committed artifact itself
# --------------------------------------------------------------------------- #
def test_the_published_detections_are_committed_and_readable():
    # This is the replication guarantee, asserted against the real files: without
    # them, fp_taxonomy / silent_witness / complementarity / null_recall only run
    # where an unpublished .model_cache happens to exist.
    if not os.path.isdir(em.PUBLISHED_DIR):
        import pytest
        pytest.skip("published detections not present in this checkout")
    got = em.load_detections("google/owlv2-large-patch14-ensemble", "bend")
    assert got is not None and len(got) > 50
    pts = next(iter(got.values()))
    assert isinstance(pts, list)


def test_every_published_file_records_its_provenance():
    if not os.path.isdir(em.PUBLISHED_DIR):
        import pytest
        pytest.skip("published detections not present in this checkout")
    files = em.published_files(em.PUBLISHED_DIR)
    assert files
    # Replicates included (PR #167 m3c): the one committed so far sat outside every
    # form check because they all globbed the top level.
    assert any(os.sep + "replicates" + os.sep in f for f in files)
    # EVERY file, not a slice. This used to check files[:5], which the directory
    # outgrew: at 77 files the slice was five Molmo entries, so a newly published
    # leg was never opened by any test and the suite passed without looking at it.
    for f in files:
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)
        # The signature is what makes a published file traceable back to the exact
        # detector configuration that produced it.
        for k in ("model", "city", "signature", "detections"):
            assert k in p, (f, k)
        # Filename and contents must agree — a mislabelled export would attribute
        # one model's detections to another with nothing to catch it. The stem is
        # the PUBLICATION name, which is the model id unless the leg needed a
        # distinguishing one (Claude: one model id, two effort levels, two legs —
        # see published_path), so compare against that and require it to be
        # recorded in the file rather than inferred from the name.
        base = os.path.basename(f)[:-len(".json")]
        stem, _, city = base.rpartition("__")
        published_as = p.get("published_as", p["model"])
        assert em.slug(published_as) == stem, (f, published_as)
        assert p["city"] == city, (f, p["city"])
        assert p["signature"].get("model_id", p["model"]) == p["model"], f
        # A distinct publication name must still say which model produced it.
        if published_as != p["model"]:
            assert p["model"] in published_as, (f, p["model"], published_as)


def test_published_detections_are_structurally_sound():
    """The invariants a downstream reader relies on, asserted on the real files.

    These held for every file when checked by hand during the #120 review; nothing
    kept them holding. A published point outside [0,1], or an n_panos that disagrees
    with the payload, would corrupt recall silently rather than raise."""
    if not os.path.isdir(em.PUBLISHED_DIR):
        import pytest
        pytest.skip("published detections not present in this checkout")
    for f in em.published_files(em.PUBLISHED_DIR):
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)
        dets = p["detections"]
        assert p["n_panos"] == len(dets), (f, p["n_panos"], len(dets))
        # A partial export looks exactly like a complete one downstream, so the
        # committed artifacts must not be partial.
        assert p.get("n_uncached", 0) == 0, (f, p.get("n_uncached"))
        for pid, pts in dets.items():
            for pt in pts:
                assert len(pt) == 3, (f, pid, pt)
                x, y, conf = pt
                assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, (f, pid, pt)
                assert conf is None or 0.0 <= conf <= 1.0, (f, pid, pt)


def test_published_panos_match_the_committed_bundles():
    """Every published split covers exactly the panos in its bundle — no more, no
    fewer. A missing pano scores as "the model found nothing there" rather than as
    missing data, which understates recall with no error anywhere."""
    if not os.path.isdir(em.PUBLISHED_DIR):
        import pytest
        pytest.skip("published detections not present in this checkout")
    bundles = {}
    for f in em.published_files(em.PUBLISHED_DIR):
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)
        city = p["city"]
        recs = os.path.join(REPO, "benchmark", city, "records.jsonl")
        if not os.path.exists(recs):      # manual_gold has no records.jsonl
            continue
        if city not in bundles:
            with open(recs, encoding="utf-8") as fh:
                bundles[city] = {json.loads(line)["pano"]["panorama_id"]
                                 for line in fh if line.strip()}
        assert set(p["detections"]) == bundles[city], (
            f, len(p["detections"]), len(bundles[city]))


# --------------------------------------------------------------------------- #
# publish_as / collisions — one model id can be several legs
# --------------------------------------------------------------------------- #
def test_the_filename_defaults_to_the_model_id(tmp_path):
    assert em.published_path("gemini-3.7-flash", "annapolis", str(tmp_path)) == \
        os.path.join(str(tmp_path), "gemini-3.7-flash__annapolis.json")


def test_publish_as_renames_only_the_file_not_the_leg(tmp_path):
    """The cache LABEL has to stay the bare model id — it is baked into keys that
    have already been paid for — so the distinguishing name lives at publication
    time and nowhere else."""
    low = em.published_path("claude-sonnet-5", "annapolis", str(tmp_path),
                            publish_as="claude-sonnet-5-effort-low")
    high = em.published_path("claude-sonnet-5", "annapolis", str(tmp_path),
                             publish_as="claude-sonnet-5-effort-high")
    assert low != high
    assert os.path.basename(low) == "claude-sonnet-5-effort-low__annapolis.json"


def test_two_legs_of_one_model_id_collide_without_publish_as():
    """The bug this guards: claude-sonnet-5 at effort low and at effort high are
    different detections with different cache signatures, and both resolve to
    claude-sonnet-5__annapolis.json."""
    assert em.published_path("claude-sonnet-5", "annapolis") == \
        em.published_path("claude-sonnet-5", "annapolis")


def test_load_detections_reads_back_what_publish_as_wrote(tmp_path):
    path = em.published_path("claude-sonnet-5", "bend", str(tmp_path),
                             publish_as="claude-sonnet-5-effort-high")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"detections": {"p1": [[0.5, 0.5, None]]}}, fh)
    # Found under the publication name...
    assert em.load_detections("claude-sonnet-5", "bend", str(tmp_path),
                              publish_as="claude-sonnet-5-effort-high") is not None
    # ...and correctly ABSENT under the bare id, rather than silently reading the
    # other leg's file.
    assert em.load_detections("claude-sonnet-5", "bend", str(tmp_path)) is None


def test_export_refuses_to_overwrite_a_different_leg(tmp_path, monkeypatch):
    """A silent overwrite is the worst available outcome: the file still looks
    complete, --verify still passes against whichever leg was written last, and
    the other leg's numbers are simply gone."""
    out = tmp_path / "out"
    out.mkdir()
    # The name the low-effort leg publishes under, per the registry -- not the bare
    # model id, which no longer collides because publication_name resolves it.
    existing = out / "claude-sonnet-5-effort-low__annapolis.json"
    existing.write_text(json.dumps({
        "model": "claude-sonnet-5", "city": "annapolis",
        "signature": {"provider": "claude", "effort": "high"},
        "detections": {"p1": []}}), encoding="utf-8")
    before = existing.read_text(encoding="utf-8")

    written, skipped, partial, collisions = _fake_export(
        monkeypatch, out, sig={"provider": "claude", "effort": "low"},
        detections={"p1": [[0.1, 0.2, None]]})

    assert written == [] and collisions and collisions[0][1] == "annapolis"
    assert existing.read_text(encoding="utf-8") == before   # untouched


def test_export_overwrites_the_same_leg_happily(tmp_path, monkeypatch):
    """Re-exporting an unchanged leg must stay a no-op-shaped success, or every
    routine re-run would look like a collision. Unchanged means the DETECTIONS
    too: this test used to overwrite `{"p1": []}` with a real point and call that
    a no-op, which is the hole the next test closes (PR #167 M1)."""
    out = tmp_path / "out"
    out.mkdir()
    sig = {"provider": "claude", "effort": "low"}
    dets = {"p1": [[0.1, 0.2, None]]}
    (out / "claude-sonnet-5-effort-low__annapolis.json").write_text(json.dumps({
        "model": "claude-sonnet-5", "city": "annapolis", "signature": sig,
        "detections": dets}), encoding="utf-8")

    written, skipped, partial, collisions = _fake_export(
        monkeypatch, out, sig=sig, detections=dets)
    assert collisions == [] and len(written) == 1
    assert json.loads((out / "claude-sonnet-5-effort-low__annapolis.json")
                      .read_text("utf-8"))["detections"] == dets


def test_export_refuses_the_same_leg_with_different_detections(tmp_path, monkeypatch):
    """A file with THIS run's signature but other detections is a re-run of the
    same leg. A replicate is exactly that by construction, so exporting the #163
    control's cache at the default --out replaced 114 of 124 panos' detections in
    the published 384 file with `collisions == []`, and --verify then passed
    against the replaced file (PR #167 M1). Refuse it; --replace is the override,
    and it has to be a decision."""
    out = tmp_path / "out"
    out.mkdir()
    sig = {"provider": "claude", "effort": "low"}
    path = out / "claude-sonnet-5-effort-low__annapolis.json"
    path.write_text(json.dumps({
        "model": "claude-sonnet-5", "city": "annapolis", "signature": sig,
        "detections": {"p1": []}}), encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    written, skipped, partial, collisions = _fake_export(
        monkeypatch, out, sig=sig, detections={"p1": [[0.1, 0.2, None]]})
    assert written == []
    assert [(c[1], c[3]) for c in collisions] == [("annapolis", "detections")]
    assert path.read_text(encoding="utf-8") == before        # untouched

    written, skipped, partial, collisions = _fake_export(
        monkeypatch, out, sig=sig, detections={"p1": [[0.1, 0.2, None]]}, replace=True)
    assert collisions == [] and len(written) == 1
    assert json.loads(path.read_text("utf-8"))["detections"] == {"p1": [[0.1, 0.2, None]]}
    # --replace does NOT reach across to another leg: a different signature is
    # still a name collision, whatever the flag says.
    written, skipped, partial, collisions = _fake_export(
        monkeypatch, out, sig={"provider": "claude", "effort": "high"},
        detections={"p1": []}, replace=True)
    assert written == [] and [c[3] for c in collisions] == ["signature"]


def test_export_refuses_an_unregistered_value_of_an_opt_in_knob(tmp_path):
    """`--vistas-input-size 512 512` is neither the bare 384 leg (its signature
    carries no input_size) nor the 1024 one. It used to publish as the bare name
    with `input_size: [512, 512]` inside and `pins: {}` (PR #167 M2)."""
    cargs = em._compare_args(".model_cache")
    assert cargs.vistas_input_size is None
    assert em.publication_name("vistas:curb-cut", cargs) == "mask2former-vistas-curb-cut"
    cargs.vistas_input_size = [1024, 1024]
    assert em.publication_name("vistas:curb-cut", cargs) == "mask2former-vistas-curb-cut-1024x1024"
    cargs.vistas_input_size = [512, 512]
    with pytest.raises(ValueError) as err:
        em.publication_name("vistas:curb-cut", cargs)
    msg = str(err.value)
    assert "vistas_input_size=[512, 512]" in msg and "not the bare leg" in msg
    assert "--publish-as" in msg
    # The whole run refuses, not just the name, and nothing is written first.
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="not the bare leg"):
        em.export(str(tmp_path / "cache"), str(out), ["richmond"], ["vistas:curb-cut"],
                  overrides={"vistas_input_size": [512, 512]})
    assert not any(out.iterdir())
    # And --publish-as remains the way to name a genuinely new leg.
    assert em.publication_name("vistas:curb-cut", cargs, "vistas-512") == "vistas-512"


def test_a_pinned_leg_publishes_under_its_registry_name_without_being_told():
    """--publish-as used to be the only thing standing between a pinned leg and the
    bare model id. Forgetting it wrote claude-opus-5__annapolis.json, which collides
    with no existing file, so the overwrite guard stayed quiet and it surfaced only
    later as a file belonging to no registered leg. The registry knows the name."""
    cargs = em._compare_args(".model_cache")
    cargs.claude_effort = "high"
    assert em.publication_name("claude:claude-opus-5", cargs) == "claude-opus-5-effort-high"
    cargs.claude_effort = "low"
    assert em.publication_name("claude:claude-opus-5", cargs) == "claude-opus-5-effort-low"
    # An explicit flag still wins -- it is how an unregistered leg gets a name.
    assert em.publication_name("claude:claude-opus-5", cargs, "other") == "other"
    # And an unregistered spec falls back to its plain label rather than raising:
    # naming a file is not the place to enforce registration.
    assert em.publication_name("gemini:gemini-9-turbo", cargs) == "gemini-9-turbo"


@pytest.mark.parametrize("spec", ["claude:claude-fable-5", "claude:claude-fable-5-1"])
def test_a_registered_spec_whose_pins_all_miss_refuses_the_bare_label(spec):
    """Both Fable legs pin claude_serving_path=anthropic; the default is vertex. The
    path is not in the cache key, so an export at the defaults found every pano,
    named the file `claude-fable-5__annapolis.json`, wrote `pins: {}`, collided with
    nothing, and reported success. The effort pin never had this hole because its
    default (`low`) matches the published legs; the serving-path default matches
    neither. (S1 on PR #157.)"""
    cargs = em._compare_args(".model_cache")
    assert cargs.claude_serving_path == "vertex"          # the trap, as shipped
    with pytest.raises(ValueError) as err:
        em.publication_name(spec, cargs)
    msg = str(err.value)
    # The message names what the registry knows and what the run had, so the fix is
    # readable off the error rather than off the roster.
    assert "claude_serving_path=anthropic" in msg
    assert "claude_serving_path=vertex" in msg
    assert "--publish-as" in msg
    cargs.claude_serving_path = "anthropic"
    assert em.publication_name(spec, cargs).endswith("-effort-low-anthropic")


def test_the_same_hole_on_the_effort_pin_is_closed_too():
    """Pre-existing shape of the same bug: `claude:claude-sonnet-5` has legs at
    effort low and high only, so `medium` used to publish as the bare
    `claude-sonnet-5`."""
    cargs = em._compare_args(".model_cache")
    cargs.claude_effort = "medium"
    with pytest.raises(ValueError, match="claude_effort=medium"):
        em.publication_name("claude:claude-sonnet-5", cargs)
    # --publish-as remains the way to name a leg that is genuinely new.
    assert em.publication_name("claude:claude-sonnet-5", cargs, "sonnet-medium") == "sonnet-medium"


def test_publish_as_refuses_more_than_one_spec(tmp_path):
    with pytest.raises(ValueError, match="ONE leg"):
        em.export("/nope", str(tmp_path), ["annapolis"],
                  ["claude:claude-sonnet-5", "gemini:gemini-3.6-flash"],
                  publish_as="both")


def _fake_stack(monkeypatch, sig, detections, label="claude-sonnet-5",
                provider="claude"):
    """Stub the detector stack and the cache so export() runs without torch.

    export() imports compare/detectors lazily inside the function, so the stubs go
    into sys.modules; the bundle's ``records.jsonl`` genuinely exists, so the only
    thing faked is the detector stack and the cache lookup. Every pano gets the
    same points (``detections["p1"]``)."""
    import types

    gts = dict.fromkeys(detections)
    fake_compare = types.SimpleNamespace(
        DetectionCache=lambda root, enabled: types.SimpleNamespace(
            get=lambda key: detections["p1"]),
        load_bundle=lambda bundle: ({}, {}, None),
        load_manual_ground_truths=lambda bundle: gts,
        ground_truths_from_verdicts=lambda records, verdicts: gts,
        cache_key=lambda label, s, city, pid: "k",
    )
    fake_detectors = types.SimpleNamespace(
        build_detector=lambda provider, mid, records, cargs: (
            label, types.SimpleNamespace(signature=lambda: sig)),
        parse_model_spec=lambda spec: (provider, label),
    )
    monkeypatch.setitem(sys.modules, "compare", fake_compare)
    monkeypatch.setitem(sys.modules, "detectors", fake_detectors)


def _fake_export(monkeypatch, out, sig, detections, replace=False):
    """Drive export() over the real annapolis bundle path without torch or a cache.
    The point is to exercise the overwrite guard, not to re-test scoring."""
    _fake_stack(monkeypatch, sig, detections)
    return em.export("/nope", str(out), ["annapolis"], ["claude:claude-sonnet-5"],
                     replace=replace)


# --------------------------------------------------------------------------- #
# --replicate: the directory comes from the registry, never from the keyboard
# --------------------------------------------------------------------------- #
REPLICATE_TAG = "makelab2-a40-2026-09-20"


def _replicate_argv(*extra):
    return ["--cache-dir", "/nope", "--models", "vistas:curb-cut",
            "--splits", "richmond", "--replicate", REPLICATE_TAG, *extra]


def test_replicate_flag_derives_out_from_the_registry(tmp_path, monkeypatch, capsys):
    """Exporting the control with a typed --out was how it could land on the file
    it replicates. With --replicate the directory is roster.replicate_dir(tag), so
    the file cannot be anywhere but under replicates/<tag>/ (PR #167 M1)."""
    from rampnet import roster
    monkeypatch.setattr(em, "PUBLISHED_DIR", str(tmp_path))
    sig = {"provider": "vistas", "model_id": "mask2former-vistas-curb-cut"}
    _fake_stack(monkeypatch, sig, {"p1": [[0.1, 0.2, None]]},
                label="mask2former-vistas-curb-cut", provider="vistas")
    assert em.main(_replicate_argv()) == 0
    rep = roster.REPLICATES_BY_TAG[REPLICATE_TAG]
    expected = tmp_path / "replicates" / REPLICATE_TAG / roster.replicate_filename(rep, "richmond")
    assert expected.exists()
    assert not (tmp_path / roster.replicate_filename(rep, "richmond")).exists()
    payload = json.loads(expected.read_text("utf-8"))
    assert payload["published_as"] == rep.of and payload["signature"] == sig
    assert str(expected.parent) in capsys.readouterr().out


def test_replicate_flag_refuses_an_explicit_out_and_an_unknown_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "PUBLISHED_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        em.main(_replicate_argv("--out", str(tmp_path / "elsewhere")))
    with pytest.raises(SystemExit):
        em.main(["--cache-dir", "/nope", "--models", "vistas:curb-cut",
                 "--splits", "richmond", "--replicate", "no-such-tag"])
    assert not (tmp_path / "replicates").exists()


def test_replicate_flag_refuses_a_leg_or_split_the_tag_does_not_replicate(
        tmp_path, monkeypatch):
    """A tag's directory holds only (rep.of, rep.splits) files -- test_roster.py
    asserts that on the committed tree -- so the exporter refuses to put anything
    else there rather than leave a stray for that test to find."""
    monkeypatch.setattr(em, "PUBLISHED_DIR", str(tmp_path))
    sig = {"provider": "vistas", "model_id": "mask2former-vistas-curb-cut"}
    _fake_stack(monkeypatch, sig, {"p1": []},
                label="mask2former-vistas-curb-cut", provider="vistas")
    with pytest.raises(SystemExit):                 # the 1024 leg is not what it replicates
        em.main(_replicate_argv("--vistas-input-size", "1024", "1024"))
    with pytest.raises(SystemExit):                 # a split it is not registered for
        em.main(["--cache-dir", "/nope", "--models", "vistas:curb-cut",
                 "--splits", "richmond,bend", "--replicate", REPLICATE_TAG])
    assert not (tmp_path / "replicates").exists()


# --------------------------------------------------------------------------- #
# the two #163 files round-trip through the real exporter, byte for byte
# --------------------------------------------------------------------------- #
PARITY_FILE = os.path.join(em.PUBLISHED_DIR, "mask2former-vistas-curb-cut-1024x1024__richmond.json")
PUBLISHED_384_FILE = os.path.join(em.PUBLISHED_DIR, "mask2former-vistas-curb-cut__richmond.json")


def _cache_from_published(tmp_path, path):
    """A ``.model_cache``-shaped directory holding exactly one published file's
    detections, keyed the way the exporter will look them up."""
    from compare import DetectionCache, cache_key
    with open(path, encoding="utf-8") as fh:
        published = json.load(fh)
    cache_dir = tmp_path / "cache"
    cache = DetectionCache(str(cache_dir))
    for pid, pts in published["detections"].items():
        cache.put(cache_key(published["model"], published["signature"],
                            published["city"], pid), pts)
    return str(cache_dir)


def test_the_parity_file_round_trips_through_the_exporter(tmp_path):
    """Rebuild a cache from the committed parity file, export it with the flags
    the doc records, and require the bytes back. This is the review's own check
    (PR #167), kept: it proves the committed file is what the exporter writes for
    `--vistas-input-size 1024 1024`, with no GPU and no cache of its own."""
    cache_dir = _cache_from_published(tmp_path, PARITY_FILE)
    out = tmp_path / "out"
    written, skipped, partial, collisions = em.export(
        cache_dir, str(out), ["richmond"], ["vistas:curb-cut"],
        overrides={"vistas_input_size": [1024, 1024]})
    assert [w[:2] for w in written] == [("mask2former-vistas-curb-cut", "richmond")]
    assert partial == [] and collisions == []
    got = out / os.path.basename(PARITY_FILE)
    assert got.read_bytes() == open(PARITY_FILE, "rb").read()


def test_the_replicate_round_trips_and_cannot_land_on_the_file_it_replicates(tmp_path):
    """The other half of M1 on the real files. The replicate's cache re-exports
    byte-identically into its registered directory; the same cache aimed at the
    published directory, where the 384 file already sits with the SAME signature
    and different detections, is refused and the published file is untouched --
    which is the overwrite the review reproduced on a scratch copy."""
    import shutil
    from rampnet import roster
    rep = roster.REPLICATES_BY_TAG["makelab2-a40-2026-09-20"]
    committed = os.path.join(roster.replicate_dir(rep, em.PUBLISHED_DIR),
                             roster.replicate_filename(rep, "richmond"))
    cache_dir = _cache_from_published(tmp_path, committed)
    out = tmp_path / "out"
    out.mkdir()
    rep_dir = roster.replicate_dir(rep, str(out))

    written, _, partial, collisions = em.export(
        cache_dir, rep_dir, ["richmond"], ["vistas:curb-cut"])
    assert len(written) == 1 and partial == [] and collisions == []
    got = os.path.join(rep_dir, roster.replicate_filename(rep, "richmond"))
    assert open(got, "rb").read() == open(committed, "rb").read()

    # Now the published 384 file sits where a hand-typed --out would aim.
    target = out / os.path.basename(PUBLISHED_384_FILE)
    shutil.copyfile(PUBLISHED_384_FILE, target)
    before = target.read_bytes()
    written, _, partial, collisions = em.export(
        cache_dir, str(out), ["richmond"], ["vistas:curb-cut"])
    assert written == [] and partial == []
    assert [(c[1], c[3]) for c in collisions] == [("richmond", "detections")]
    assert target.read_bytes() == before
    # The same signature is what makes it a replicate, so the two files agree on
    # the header and disagree only in the detections -- which is why the old
    # guard (signature only) waved it through.
    a, b = json.loads(before), json.loads(open(committed, encoding="utf-8").read())
    assert a["signature"] == b["signature"] and a["detections"] != b["detections"]


# --------------------------------------------------------------------------- #
# the ledger count is a test, not a promise
# --------------------------------------------------------------------------- #
def test_the_ledger_count_matches_the_directory():
    """docs/replication.md quotes the size of the published detection corpus.

    That number drifted three times (61 -> 68 -> 78 -> 108) before anyone noticed,
    in the one document whose entire job is keeping the repo honest about what a
    stranger can actually obtain. Prose cannot hold a count; this can.

    The count is of LEGS' files, i.e. the top level: a replicate is not a leg and
    the ledger says "+ one replicate" beside the number rather than folding it in,
    so this deliberately does not use ``published_files`` (PR #167 m3c)."""
    import re

    published = [f for f in os.listdir(em.PUBLISHED_DIR) if f.endswith(".json")]
    assert set(published) == {os.path.basename(f) for f in
                              em.published_files(em.PUBLISHED_DIR, replicates=False)}
    doc = os.path.join(REPO, "docs", "replication.md")
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()

    claimed = re.search(r"\*\*(\d+) files, [\d.]+ MB\*\*", text)
    assert claimed, "docs/replication.md no longer states a '**N files, X MB**' count"
    assert int(claimed.group(1)) == len(published), (
        f"docs/replication.md claims {claimed.group(1)} published detection files, "
        f"{em.PUBLISHED_DIR} holds {len(published)}. Re-measure and update the ledger.")

    row = re.search(r"model_detections/`[^|]*\|[^|]*\((\d+) files\)", text)
    assert row and int(row.group(1)) == len(published), (
        "the 'Status by input' table's file count disagrees with the directory")

    # The per-leg breakdown is the half a reader uses to find a given file's write-up,
    # and it is the half that drifts: #139 published eight files and updated only the
    # total, leaving rows that summed to 114 under a heading that said 122. A total
    # nobody can decompose is not a ledger, so check the decomposition too.
    block = re.search(r"^\| what \| files \|.*?\n\n", text, re.S | re.M)
    assert block, "docs/replication.md no longer has a '| what | files |' breakdown table"
    rows = re.findall(r"^\|[^|\n]+\|\s*(\d+)\s*\|", block.group(0), re.M)
    assert rows, "the breakdown table has no countable rows"
    assert sum(int(n) for n in rows) == len(published), (
        f"the per-leg breakdown in docs/replication.md sums to "
        f"{sum(int(n) for n in rows)} ({'+'.join(rows)}), but {em.PUBLISHED_DIR} holds "
        f"{len(published)}. A row is missing or stale — the total alone is not the ledger.")


def test_every_published_file_is_in_canonical_form():
    """A regenerated copy must be provably identical, not merely equivalent.

    This is the strong version of that promise and it costs nothing to check: every
    committed file must equal its own canonical re-dump, so the corpus cannot drift
    from what the exporter writes. It caught the drift it was written for -- 108 of
    114 files predated the `published_as` field, so re-exporting any of them would
    have produced a diff with identical detections inside.
    """
    stale = em.canonicalize(em.PUBLISHED_DIR, write=False)[0]
    # canonicalize walks replicates/<tag>/ too; make sure this run actually did.
    assert any(os.sep + "replicates" + os.sep in f
               for f in em.published_files(em.PUBLISHED_DIR))
    assert not stale, (
        f"{len(stale)} published file(s) are not what the exporter would write, "
        f"e.g. {stale[:3]} — run `python scripts/analysis/export_model_cache.py "
        f"--canonicalize --write`")


def test_canonicalize_never_touches_detections():
    """It exists to edit the metadata envelope. If it could rewrite a detection it
    would be a way to silently alter published results without a cache."""
    before = {}
    for path in em.published_files(em.PUBLISHED_DIR):
        with open(path, encoding="utf-8") as fh:
            before[path] = json.load(fh)["detections"]
    changed, unfixable = em.canonicalize(em.PUBLISHED_DIR, write=False)
    assert not changed and not unfixable          # already canonical, so a no-op
    for path, dets in before.items():
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh)["detections"] == dets, path


def test_every_published_file_declares_the_name_it_is_published_under():
    """`model` is the cache label and `published_as` is the filename. They differ
    exactly when a pin is involved, so a file that omits the second is a file whose
    identity has to be inferred from its own filename."""
    for path in em.published_files(em.PUBLISHED_DIR):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert "published_as" in payload, path
        assert em.slug(payload["published_as"]) == name.rpartition("__")[0], path


def test_every_published_leg_is_named_in_the_ledger():
    """A published artifact nobody documented is indistinguishable from a stray
    file. Each distinct model stem under model_detections/ must appear by name in
    docs/replication.md."""
    stems = {f.rsplit("__", 1)[0] for f in os.listdir(em.PUBLISHED_DIR)
             if f.endswith(".json")}
    text = ""
    for rel in ("docs/replication.md", "docs/model_comparison.md",
                "scripts/model_comparison/yolo_baseline/README.md"):
        path = os.path.join(REPO, *rel.split("/"))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                text += fh.read()
    # The Claude legs are written as one brace expansion rather than four literal
    # stems, which is how a person would write it; accept that form too.
    brace = "claude-{sonnet,opus}-5-effort-{low,high}"

    def documented(stem):
        # Docs name models by their real id, while the filename carries the slug,
        # so check every spelling slug() can produce. It maps any run of unsafe
        # characters to "__", so the same stem can stand for a "/" (a HF repo id,
        # IDEA-Research/grounding-dino-base) or a "+" (a class set,
        # mask2former-vistas-curb-cut+curb).
        if any(spelling in text for spelling in
               (stem, stem.replace("__", "/"), stem.replace("__", "+"))):
            return True
        return stem.startswith("claude-") and brace in text

    undocumented = sorted(s for s in stems if not documented(s))
    assert not undocumented, (
        f"published but named in no doc a reader would find: {undocumented}")
