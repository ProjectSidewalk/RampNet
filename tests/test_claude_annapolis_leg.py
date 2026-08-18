"""The published Claude/annapolis numbers, re-derived from committed files (#122).

Every figure in the "First results: annapolis" table of ``docs/model_comparison.md``
is recomputed here from ``benchmark/model_detections/claude-*__annapolis.json`` plus
the committed annapolis bundle. No ``.model_cache``, no network, no API key, no GPU —
which is the whole point: the four legs cost $28.82 and nobody should have to spend
that again to check the arithmetic, or to notice when someone edits a number in the
doc without re-running anything.

All four legs now cover the whole split, and that is asserted rather than assumed.
It was not always true: the sonnet/low leg originally scored 124 panos / 290 GT ramps
because one panorama (annapolis:1528518111324684) was lost to the malformed-tool-result
crash, so its recall was computed against a different denominator from the other three
rows while the table said "Full 125-pano annapolis split". The parse hardening in
``boxes_from_claude_response`` fixed the cause and the pano was re-run (6 calls, $0.03,
2026-08-18). A denominator that silently differs per row is exactly the kind of defect
that hides in prose, so ``test_every_leg_covers_the_whole_split`` keeps it out.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import compare as C  # noqa: E402
from export_model_cache import load_detections, published_path  # noqa: E402
from rampnet.detection_eval import radius_sq_for, score_pano  # noqa: E402

BUNDLE = os.path.join(REPO, "benchmark", "annapolis")
PUBLISHED = os.path.join(REPO, "benchmark", "model_detections")

# The panorama the sonnet/low leg originally lost to the parse crash, recovered
# 2026-08-18. Named here because "one pano" is not a checkable claim and this is.
RECOVERED_PANO = "1528518111324684"

# (model, effort) -> (panos, n_gt, tp, fp, fn, P, R, F1) exactly as published.
PUBLISHED_LEGS = {
    ("claude-sonnet-5", "low"): (125, 294, 112, 78, 182, 0.589, 0.381, 0.463),
    ("claude-sonnet-5", "high"): (125, 294, 122, 119, 172, 0.506, 0.415, 0.456),
    ("claude-opus-5", "low"): (125, 294, 178, 133, 116, 0.572, 0.605, 0.588),
    ("claude-opus-5", "high"): (125, 294, 193, 256, 101, 0.430, 0.656, 0.520),
}


def _ground_truths():
    records, verdicts, _ = C.load_bundle(BUNDLE)
    return C.ground_truths_from_verdicts(records, verdicts)


def _score(model, effort, gts):
    pub = load_detections(model, "annapolis", PUBLISHED,
                          publish_as=f"{model}-effort-{effort}")
    assert pub is not None, f"no published export for {model}/{effort}"
    tp = fp = n_gt = panos = 0
    for pid, gt in gts.items():
        if pid not in pub:          # a pano this leg never produced
            continue
        s = score_pano(pub[pid], gt, radius_sq=radius_sq_for())
        tp += s.tp
        fp += s.fp
        n_gt += s.n_gt
        panos += 1
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return panos, n_gt, tp, fp, fn, precision, recall, f1


def test_the_annapolis_bundle_is_the_one_the_numbers_were_read_against():
    gts = _ground_truths()
    assert len(gts) == 125
    assert sum(len(g.gt_points) for g in gts.values()) == 294


@pytest.mark.parametrize("model,effort", sorted(PUBLISHED_LEGS))
def test_published_claude_numbers_reproduce_from_the_committed_detections(model, effort):
    want = PUBLISHED_LEGS[(model, effort)]
    got = _score(model, effort, _ground_truths())
    assert got[:5] == want[:5], (
        f"{model}/{effort}: (panos, n_gt, tp, fp, fn) = {got[:5]}, published {want[:5]}")
    for name, g, w in zip(("precision", "recall", "f1"), got[5:], want[5:]):
        assert round(g, 3) == pytest.approx(w, abs=0.0005), f"{model}/{effort} {name}"


def test_every_leg_covers_the_whole_split():
    """All four rows must share one denominator, or the table compares nothing.

    The sonnet/low leg once did not: a single lost panorama put its recall over
    290 GT ramps while the other three used 294, inside a table captioned "Full
    125-pano annapolis split". Comparing 0.372 against 0.415 across different
    denominators is the error that produces; this is the guard that makes it
    impossible to reintroduce quietly."""
    gts = _ground_truths()
    for model, effort in PUBLISHED_LEGS:
        pub = load_detections(model, "annapolis", PUBLISHED,
                              publish_as=f"{model}-effort-{effort}")
        missing = set(gts) - set(pub)
        assert not missing, f"{model}/{effort} is missing {len(missing)} pano(s)"
        assert RECOVERED_PANO in pub, f"{model}/{effort} lacks the recovered pano"
    assert {v[0] for v in PUBLISHED_LEGS.values()} == {125}
    assert {v[1] for v in PUBLISHED_LEGS.values()} == {294}


def test_each_leg_is_published_under_its_own_name():
    """Two effort levels of one model id are two legs. Publishing them under the
    bare id would put both on one filename and silently keep only the last."""
    paths = {published_path(m, "annapolis", PUBLISHED, publish_as=f"{m}-effort-{e}")
             for m, e in PUBLISHED_LEGS}
    assert len(paths) == len(PUBLISHED_LEGS)
    for path in paths:
        assert os.path.exists(path), f"missing published export: {path}"


def test_every_export_records_the_signature_that_produced_it():
    """Provenance has to survive in the file, because the cache it came from is
    git-ignored and the effort level is not visible in the detections."""
    for (model, effort) in PUBLISHED_LEGS:
        path = published_path(model, "annapolis", PUBLISHED,
                              publish_as=f"{model}-effort-{effort}")
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        sig = blob["signature"]
        assert blob["model"] == model
        assert blob["published_as"] == f"{model}-effort-{effort}"
        assert sig["provider"] == "claude"
        assert sig["model_id"] == model
        assert sig["effort"] == effort
        assert sig["tool_choice"] == "auto"
        # As-run encoding/temperature are absent by construction — see
        # ClaudeDetector.signature and CLAUDE_AS_RUN_IMAGE_FORMAT.
        assert "image_format" not in sig and "temperature" not in sig
        assert blob["n_uncached"] == 0   # every leg is complete; see the module docstring
