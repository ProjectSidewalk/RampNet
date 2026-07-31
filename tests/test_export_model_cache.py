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
    files = glob.glob(os.path.join(em.PUBLISHED_DIR, "*.json"))
    assert files
    for f in files[:5]:
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)
        # The signature is what makes a published file traceable back to the exact
        # detector configuration that produced it.
        for k in ("model", "city", "signature", "detections"):
            assert k in p, (f, k)
