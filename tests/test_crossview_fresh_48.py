"""Fresh-pair confirmation of the #48 cross-view winners (docs/crossview_align_48.md,
"Fresh-pair confirmation"): the pinned fresh pair list re-derives from committed inputs, is
disjoint from the frozen 300, and the committed confirmation.json re-derives from the
committed fresh predictions. CPU only, committed files only."""
import collections
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import crossview_align_48 as H  # noqa: E402
import crossview_fresh_48 as F  # noqa: E402
from crossview_arms import _mv3d as M  # noqa: E402

FRESH_CSV = os.path.join(H.FRESH_DIR, "pairs.csv")
FRESH_PRED_DIR = os.path.join(H.FRESH_DIR, "predictions")
# The fresh predictions and confirmation.json are committed (0f820f9, 0da7360), so a missing
# file is a failure, not a skip: the skipif markers these tests carried while the GPU runs
# were in flight were removed after the review of #220, which showed that moving
# mapa_k_pair.jsonl aside left the suite green (3 passed, 3 skipped).

#: code_fingerprint of crossview_arms/ff3d.py, the code of all three MapAnything arms, at
#: c84dc74 -- the commit the 300's mapa_posed_pair predictions were committed with (the run is
#: timestamped slightly earlier; the diffs in between are no-ops for these arms), and the code
#: cd06387's mapa_k_pair predictions ran (same fingerprint). Equal at every commit since.
#: mapa_posed_corner's 300 predictions (committed with 0428bb7) predate two code edits, both no-ops for it:
#: 8cdf52f passes revision=MODEL_REVISIONS[...] (the snapshot that run had loaded) and
#: c84dc74 adds a "mono" branch that corner_mode=True never takes; see the doc's
#: "Settings and code unchanged".
FF3D_FINGERPRINT_300 = "43958e582f4041c3ff70d652d2eff63749dd2dd84e5eef60575d2092d8b37b19"
MAPANYTHING_WEIGHTS = "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def fresh():
    prev = H.use_pair_set("fresh")
    try:
        yield H.read_frozen_pairs()
    finally:
        H.use_pair_set(prev)


def test_default_pair_set_is_the_frozen_300():
    assert H.PAIR_SET == "frozen300"
    assert H.PAIRS_SHA256 == "a85a11bceb57e7d4db4bc5914c35cb17b5fdaf9bc8c9189957574a13260db388"
    assert H.PRED_DIR == os.path.join(H.OUT, "predictions")


def test_fresh_pairs_are_pinned_and_disjoint_from_the_300(fresh):
    assert _sha(FRESH_CSV) == H.FRESH_PAIRS_SHA256
    assert len(fresh) == 807
    assert [p["pair_id"] for p in fresh] == [f"f{i:03d}" for i in range(807)]
    frozen = {(p["ramp_uid"], p["oth_pano"]) for p in H.read_rows(os.path.join(H.OUT, "pairs.csv"))}
    assert not frozen & {(p["ramp_uid"], p["oth_pano"]) for p in fresh}
    per_ramp = collections.Counter(p["ramp_uid"] for p in fresh)
    assert max(per_ramp.values()) <= H.MAX_PAIRS_PER_RAMP
    assert len(per_ramp) == 477
    assert sum(p["ramp_uid"] not in F.old_ramps() for p in fresh) == 686


def test_fresh_pairs_rederive_from_eligible_pairs(tmp_path):
    """The pinned file is exactly what the committed rule draws from the committed lists."""
    assert _sha(H.ELIGIBLE_CSV) == H.ELIGIBLE_SHA256
    rows = F.draw_fresh(H.read_rows(H.ELIGIBLE_CSV), H.read_rows(os.path.join(H.OUT, "pairs.csv")))
    out = H.write_rows(str(tmp_path / "pairs.csv"), rows)
    assert _sha(out) == H.FRESH_PAIRS_SHA256


def test_fresh_manifest_and_predictions_belong_to_the_fresh_set(fresh):
    m = M.load_manifest()
    assert m["pairs_sha256"] == H.FRESH_PAIRS_SHA256
    assert len(m["corners"]) == 477
    assert m["instrument_check_worst_deg"] < 0.01
    ids = {p["pair_id"] for p in fresh}
    for arm in (F.BASELINE,) + F.FRESH_ARMS:
        preds = H.read_predictions(arm)          # refuses a meta with another pairs hash
        assert set(preds) == ids, arm
        with open(H.prediction_paths(arm)[1], encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["pair_set"] == "fresh" and meta["pairs"] == 807, arm


def test_fresh_arms_ran_the_300s_code_settings_and_weights(fresh):
    """Settings AND code unchanged since the 300 (review of #220): the registered config,
    description and inputs match the 300-pair metas; ff3d.py's code (comments and docstrings
    ignored) is the code the 300 ran; the weights pin is the one both runs loaded; and each
    fresh meta names the committed fresh corner manifest."""
    frozen_dir = os.path.join(H.OUT, "predictions")
    with open(M.__file__.replace("_mv3d.py", "ff3d.py"), encoding="utf-8") as f:
        assert F.code_fingerprint(f.read()) == FF3D_FINGERPRINT_300
    from crossview_arms import ff3d
    assert ff3d.MODEL_REVISIONS["mapanything"] == MAPANYTHING_WEIGHTS
    manifest_sha = _sha(os.path.join(H.FRESH_DIR, "mv3d_corners.json"))
    for arm in F.FRESH_ARMS:
        with open(H.prediction_paths(arm)[1], encoding="utf-8") as f:
            new = json.load(f)
        with open(os.path.join(frozen_dir, f"{arm}.meta.json"), encoding="utf-8") as f:
            old = json.load(f)
        for key in ("config", "description", "needs"):
            assert new[key] == old[key], (arm, key)
        assert new["manifest_sha256"] == manifest_sha, arm


def test_code_fingerprint_ignores_comments_and_docstrings_only():
    base = 'def f(x):\n    """Doc."""\n    return x + 1  # add\n'
    assert F.code_fingerprint(base) == F.code_fingerprint(
        '# header\n\ndef f(x):\n    """Other\n    doc."""\n    return x + 1\n')
    assert F.code_fingerprint(base) != F.code_fingerprint(base.replace("x + 1", "x + 2"))


def test_harness_pairs_refuses_under_the_fresh_set():
    """crossview_align_48.py pairs builds the frozen 300 and writes eligible_pairs.csv and
    pairs_meta.json under OUT; under the fresh set it must refuse before touching anything."""
    import argparse
    before = {p: _sha(p) for p in (FRESH_CSV, H.ELIGIBLE_CSV, os.path.join(H.OUT, "pairs_meta.json"))}
    prev = H.use_pair_set("fresh")
    try:
        with pytest.raises(SystemExit, match="frozen300 set only"):
            H.cmd_pairs(argparse.Namespace(force=True, labeler_root="unused", runs_root=None,
                                           results_root=None))
    finally:
        H.use_pair_set(prev)
    assert before == {p: _sha(p) for p in before}


def test_fresh_pairs_check_mode_passes_and_writes_nothing(capsys):
    before = (_sha(FRESH_CSV), _sha(F.FRESH_META_JSON))
    F.main(["pairs", "--check"])
    assert "pairs.csv identical" in capsys.readouterr().out
    assert before == (_sha(FRESH_CSV), _sha(F.FRESH_META_JSON))


def test_confirmation_prespecified_part_is_the_original_bytes():
    """The post hoc ``sensitivity`` key was added after the pre-specified test ran. Removing
    it and re-serializing must give the original file (sha bb8ead92...), so no pre-specified
    number moved."""
    with open(F.CONFIRMATION_JSON, encoding="utf-8") as f:
        committed = json.load(f)
    assert set(committed) == {"config", "primary", "secondary", "per_city", "sensitivity"}
    committed.pop("sensitivity")
    body = json.dumps(H.rnd(committed), indent=1, sort_keys=True) + "\n"
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == F.PRESPECIFIED_CONFIRMATION_SHA256


def test_sensitivity_counts_and_the_fragile_cell():
    """The post hoc reads (review of #220): 55 primary pairs repeat a 300 pano pair, 190 share
    a pano with the 300; mapa_k_pair on GSV is the only cell any read takes to <= 0."""
    with open(F.CONFIRMATION_JSON, encoding="utf-8") as f:
        sens = json.load(f)["sensitivity"]
    assert sens["primary_pairs_identical_pano_pair"] == 55
    assert sens["primary_pairs_any_shared_pano"] == 190
    for read in F.SENSITIVITY_READS:
        for stratum in ("all", "gsv", "mapillary"):
            for arm in F.FRESH_ARMS:
                cell = sens[read][stratum][arm]
                fragile = (arm, stratum) == ("mapa_k_pair", "gsv")
                assert cell["confirmed"] is not fragile, (read, stratum, arm)
                if not fragile:
                    assert cell["bonferroni_lower"] >= 0.105, (read, stratum, arm)


def test_confirmation_rederives_from_committed_predictions():
    with open(F.CONFIRMATION_JSON, encoding="utf-8") as f:
        committed = json.load(f)
    got = json.loads(json.dumps(H.rnd(F.score_fresh())))
    assert got == committed


#: sha256 of the doc's "### Fixed before running" subsection, as committed in the plan commit
#: cd65024 before any prediction: the plan must never be edited after the fact.
PLAN_SECTION_SHA256 = "80f1cb32c95a548428a98c69f5b203cf77229966a7832cc4a846db2e3e04e90f"


def test_plan_section_is_byte_identical_to_the_plan_commit():
    doc = os.path.join(os.path.dirname(HERE), "docs", "crossview_align_48.md")
    with open(doc, "rb") as f:
        text = f.read().decode("utf-8").replace("\r\n", "\n")
    a = text.index("### Fixed before running")
    b = text.index("### Result", a)
    assert hashlib.sha256(text[a:b].encode("utf-8")).hexdigest() == PLAN_SECTION_SHA256


def test_package_version_never_guesses_a_checkout_for_a_module_without_a_file():
    """A module with no ``__file__`` must not pick up the git HEAD of the current directory
    (re-review of #220, N3): it records only its version."""
    import types
    fake = types.ModuleType("fake_no_file")
    fake.__version__ = "9.9"
    assert M._package_version("fake_no_file", fake) == "9.9"
