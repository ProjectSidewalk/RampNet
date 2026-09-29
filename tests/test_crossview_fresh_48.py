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
#: the GPU arms' predictions come back from makelab2 after the plan commit; until they are
#: committed, only the tests that read them skip (named, not a blanket skip)
_MISSING_ARMS = [a for a in F.FRESH_ARMS
                 if not os.path.exists(os.path.join(FRESH_PRED_DIR, f"{a}.jsonl"))]
needs_fresh_predictions = pytest.mark.skipif(
    bool(_MISSING_ARMS), reason=f"fresh predictions not committed yet: {_MISSING_ARMS}")
needs_confirmation = pytest.mark.skipif(
    not os.path.exists(F.CONFIRMATION_JSON), reason="fresh/confirmation.json not committed yet")


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


@needs_fresh_predictions
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


@needs_fresh_predictions
def test_fresh_arms_use_their_committed_300_pair_settings(fresh):
    frozen_dir = os.path.join(H.OUT, "predictions")
    for arm in F.FRESH_ARMS:
        with open(H.prediction_paths(arm)[1], encoding="utf-8") as f:
            new = json.load(f)
        with open(os.path.join(frozen_dir, f"{arm}.meta.json"), encoding="utf-8") as f:
            old = json.load(f)
        assert new["config"] == old["config"], arm


@needs_fresh_predictions
@needs_confirmation
def test_confirmation_rederives_from_committed_predictions():
    with open(F.CONFIRMATION_JSON, encoding="utf-8") as f:
        committed = json.load(f)
    got = json.loads(json.dumps(H.rnd(F.score_fresh())))
    assert got == committed
