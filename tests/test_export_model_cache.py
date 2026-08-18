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
    import glob
    files = sorted(glob.glob(os.path.join(em.PUBLISHED_DIR, "*.json")))
    assert files
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
        # one model's detections to another with nothing to catch it.
        base = os.path.basename(f)[:-len(".json")]
        model, _, city = base.rpartition("__")
        assert em.slug(p["model"]) == model, (f, p["model"])
        assert p["city"] == city, (f, p["city"])
        assert p["signature"].get("model_id", p["model"]) == p["model"], f


def test_published_detections_are_structurally_sound():
    """The invariants a downstream reader relies on, asserted on the real files.

    These held for every file when checked by hand during the #120 review; nothing
    kept them holding. A published point outside [0,1], or an n_panos that disagrees
    with the payload, would corrupt recall silently rather than raise."""
    if not os.path.isdir(em.PUBLISHED_DIR):
        import pytest
        pytest.skip("published detections not present in this checkout")
    import glob
    for f in sorted(glob.glob(os.path.join(em.PUBLISHED_DIR, "*.json"))):
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
    import glob
    bundles = {}
    for f in sorted(glob.glob(os.path.join(em.PUBLISHED_DIR, "*.json"))):
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
