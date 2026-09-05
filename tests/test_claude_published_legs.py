"""The published Claude numbers, re-derived from committed files (#122, #151).

Every Claude figure in ``docs/model_comparison.md`` is recomputed here from
``benchmark/model_detections/claude-*.json`` plus the committed bundles. No
``.model_cache``, no network, no API key, no GPU -- which is the whole point: these
legs cost real money and nobody should have to spend it again to check the
arithmetic, or to notice when someone edits a number in the doc without re-running
anything.

Three splits are covered, and they are not symmetric:

* ``annapolis`` (#122) -- all four legs, two model ids x two effort levels.
* ``laurens_mapillary`` / ``laurens_gsv`` (#151) -- ``claude-opus-5`` at effort low
  only. Those two arms were run to answer whether Laurens is hard because it is
  rural or because of the imagery rig, and that test wanted the strongest zero-shot
  model in the benchmark on both arms. The other three legs were not run there.

**Whole-split coverage is asserted, never assumed**, because it has silently broken
twice. On annapolis the sonnet/low leg originally scored 124 panos / 290 GT ramps --
one panorama lost to a malformed-tool-result crash -- so its recall used a different
denominator from the other three rows while the table said "Full 125-pano annapolis
split". On ``laurens_mapillary`` two panoramas died on Vertex's transient 404 even
after the detector's four retries; ``compare.py`` isolated them and scored the other
92, which would have published a recall against 247 GT ramps under a caption saying
249. Both were recovered by re-running. A denominator that silently differs is
exactly the kind of defect that hides in prose, so it is a test rather than a habit.
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

PUBLISHED = os.path.join(REPO, "benchmark", "model_detections")

# Panoramas that were lost and recovered. Named because "one pano" / "two panos" is
# not a checkable claim and these are.
RECOVERED = {
    "annapolis": ("1528518111324684",),                      # parse crash, 2026-08-18
    "laurens_mapillary": ("2102336717175440",                # Vertex transient 404,
                          "2281219182305735"),               # both 2026-09-04
}

# split -> (panos, GT ramps) the bundle must hold for these numbers to mean anything.
SPLITS = {
    "annapolis": (125, 294),
    "laurens_mapillary": (94, 249),
    "laurens_gsv": (86, 220),
}

# (model, effort, serving_path, split) -> (panos, n_gt, tp, fp, fn, P, R, F1) exactly
# as published. The serving path is part of the key because it is part of the leg's
# published NAME (#156): it does not enter the detection signature -- it changes who
# bills, not what was asked -- but a Fable leg cannot run on Vertex at all, so two
# rows of this table now come from a different account than the other six, and a
# fixture that could not say which would be hiding that.
PUBLISHED_LEGS = {
    ("claude-sonnet-5", "low", "vertex", "annapolis"): (125, 294, 112, 78, 182, 0.589, 0.381, 0.463),
    ("claude-sonnet-5", "high", "vertex", "annapolis"): (125, 294, 122, 119, 172, 0.506, 0.415, 0.456),
    ("claude-opus-5", "low", "vertex", "annapolis"): (125, 294, 178, 133, 116, 0.572, 0.605, 0.588),
    ("claude-opus-5", "high", "vertex", "annapolis"): (125, 294, 193, 256, 101, 0.430, 0.656, 0.520),
    ("claude-opus-5", "low", "vertex", "laurens_mapillary"): (94, 249, 96, 102, 153, 0.485, 0.386, 0.430),
    ("claude-opus-5", "low", "vertex", "laurens_gsv"): (86, 220, 87, 91, 133, 0.489, 0.395, 0.437),
    ("claude-fable-5-1", "low", "anthropic", "annapolis"): (125, 294, 172, 98, 122, 0.637, 0.585, 0.610),
    ("claude-fable-5", "low", "anthropic", "annapolis"): (125, 294, 190, 138, 104, 0.579, 0.646, 0.611),
}


def _publish_as(model, effort, serving):
    """The filename stem this leg publishes under.

    Mirrors ``roster.published_name`` without importing the registry, so this file
    stays an independent check on the published numbers rather than a second reading
    of the same source. `vertex` is elided because it is the default and the six
    Vertex legs were published before the path was a pin.
    """
    stem = f"{model}-effort-{effort}"
    return stem if serving == "vertex" else f"{stem}-{serving}"


def _bundle(split):
    return os.path.join(REPO, "benchmark", split)


def _ground_truths(split):
    records, verdicts, _ = C.load_bundle(_bundle(split))
    return C.ground_truths_from_verdicts(records, verdicts)


def _score(model, effort, serving, split, gts):
    pub = load_detections(model, split, PUBLISHED,
                          publish_as=_publish_as(model, effort, serving))
    assert pub is not None, f"no published export for {model}/{effort}/{split}"
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


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_the_bundles_are_the_ones_the_numbers_were_read_against(split):
    want_panos, want_gt = SPLITS[split]
    gts = _ground_truths(split)
    assert len(gts) == want_panos
    assert sum(len(g.gt_points) for g in gts.values()) == want_gt


@pytest.mark.parametrize("model,effort,serving,split", sorted(PUBLISHED_LEGS))
def test_published_claude_numbers_reproduce_from_the_committed_detections(
        model, effort, serving, split):
    want = PUBLISHED_LEGS[(model, effort, serving, split)]
    got = _score(model, effort, serving, split, _ground_truths(split))
    assert got[:5] == want[:5], (
        f"{model}/{effort}/{split}: (panos, n_gt, tp, fp, fn) = {got[:5]}, "
        f"published {want[:5]}")
    for name, g, w in zip(("precision", "recall", "f1"), got[5:], want[5:]):
        assert round(g, 3) == pytest.approx(w, abs=0.0005), f"{model}/{effort}/{split} {name}"


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_every_leg_covers_the_whole_split(split):
    """Rows in one table must share one denominator, or the table compares nothing.

    Broken once per split family so far -- a parse crash on annapolis, two transient
    Vertex 404s on laurens_mapillary -- and in both cases the run still exited 0 and
    printed a plausible table. This is the guard that makes it impossible to
    reintroduce quietly."""
    gts = _ground_truths(split)
    legs = [k for k in PUBLISHED_LEGS if k[3] == split]
    assert legs, f"no published legs for {split}"
    for model, effort, serving, _ in legs:
        pub = load_detections(model, split, PUBLISHED,
                              publish_as=_publish_as(model, effort, serving))
        missing = set(gts) - set(pub)
        assert not missing, f"{model}/{effort}/{split} is missing {len(missing)} pano(s)"
        for pano in RECOVERED.get(split, ()):
            assert pano in pub, f"{model}/{effort}/{split} lacks recovered pano {pano}"
    want_panos, want_gt = SPLITS[split]
    assert {PUBLISHED_LEGS[k][0] for k in legs} == {want_panos}
    assert {PUBLISHED_LEGS[k][1] for k in legs} == {want_gt}


def test_each_leg_is_published_under_its_own_name():
    """Two effort levels of one model id are two legs. Publishing them under the
    bare id would put both on one filename and silently keep only the last."""
    paths = {published_path(m, split, PUBLISHED, publish_as=_publish_as(m, e, sp))
             for m, e, sp, split in PUBLISHED_LEGS}
    assert len(paths) == len(PUBLISHED_LEGS)
    for path in paths:
        assert os.path.exists(path), f"missing published export: {path}"


@pytest.mark.parametrize("model,effort,serving,split", sorted(PUBLISHED_LEGS))
def test_every_export_records_the_signature_that_produced_it(
        model, effort, serving, split):
    """Provenance has to survive in the file, because the cache it came from is
    git-ignored and the effort level is not visible in the detections."""
    path = published_path(model, split, PUBLISHED,
                          publish_as=_publish_as(model, effort, serving))
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    sig = blob["signature"]
    assert blob["model"] == model
    assert blob["published_as"] == _publish_as(model, effort, serving)
    # The serving path is NOT in the signature, by design. On the legs where it is
    # not the default it must therefore be legible from `pins`, or the published
    # file cannot say which account produced it.
    if serving != "vertex":
        assert blob["pins"]["claude_serving_path"] == serving
    assert sig["provider"] == "claude"
    assert sig["model_id"] == model
    assert sig["effort"] == effort
    assert sig["tool_choice"] == "auto"
    # As-run encoding/temperature are absent by construction -- see
    # ClaudeDetector.signature and CLAUDE_AS_RUN_IMAGE_FORMAT.
    assert "image_format" not in sig and "temperature" not in sig
    assert blob["n_uncached"] == 0   # every leg is complete; see the module docstring


def test_the_laurens_arms_are_the_pair_the_rig_comparison_rests_on():
    """#151 compares one model across two rigs, so both arms must be the same leg.

    A comparison whose two halves came from different effort levels (or different
    model ids) would attribute to the imagery a difference that is really a
    configuration change -- the exact confound the second arm exists to remove."""
    arms = [k for k in PUBLISHED_LEGS if k[3].startswith("laurens_")]
    assert {k[3] for k in arms} == {"laurens_mapillary", "laurens_gsv"}
    # Serving path is pinned here too: both arms must come from the same account,
    # for the same reason they must share an effort level.
    assert {(k[0], k[1], k[2]) for k in arms} == {("claude-opus-5", "low", "vertex")}
    f1 = {k[3]: PUBLISHED_LEGS[k][7] for k in arms}
    # The published reading: the strongest zero-shot model is FLAT across the rigs
    # (+0.007) where RampNet gains +0.115. If a re-run ever moves this materially,
    # the write-up's argument changes and should be re-read, not silently updated.
    assert abs(f1["laurens_gsv"] - f1["laurens_mapillary"]) < 0.01
