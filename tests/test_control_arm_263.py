"""#263 control arm: the committed caches and report.

CPU only; reads analysis_out/control_arm_263/ and the committed op_cache. It does not re-run
control_arm_263.py (about five minutes of bootstrap); `control_arm_263.py --check` does that.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
C = pytest.importorskip("control_arm_263")

OUT = os.path.join(REPO, "analysis_out", "control_arm_263")
DOC = os.path.join(REPO, "docs", "control_arm_263.md")
SM = "control (mean of s1-s3) - released"


def _cache(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def _rep():
    with open(C.RESULTS, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("split", C.SPLITS)
def test_every_checkpoint_scored_every_pano_of_every_split(split):
    """extract scores only the panos present on disk, so a short pano directory would give a
    short cache silently; every cache must have the committed op_cache's pano set."""
    want = {p["pano"] for p in _cache(REPO, "analysis_out", "op_cache", f"{split}.json")["panos"]}
    for lab in ("released",) + C.CONTROLS:
        d = _cache(OUT, "caches", lab, f"{split}.json")
        assert {p["pano"] for p in d["panos"]} == want, (lab, split)
        assert d["meta"]["fp16"] is False and d["meta"]["tta"] is False
        assert d["meta"]["score_floor"] == 0.05 and d["meta"]["min_distance"] == 10
        assert d["meta"]["model"] == ("projectsidewalk/rampnet-model" if lab == "released"
                                      else f"control263_{lab}")


@pytest.mark.parametrize("split", sorted(set(C.SPLITS) - {"bayonne", "juneau"}))
def test_released_rescore_is_bit_identical_to_the_82_klone_caches(split):
    """Same scorer, same checkpoint, three weeks apart on klone: the same peaks to the bit."""
    now = _cache(OUT, "caches", "released", f"{split}.json")["panos"]
    then = _cache(REPO, "analysis_out", "aug_transfer_82", "finetune", "released",
                  f"{split}.json")["panos"]
    assert {p["pano"]: p["preds"] for p in now} == {p["pano"]: p["preds"] for p in then}


def test_report_has_no_missing_caches_and_every_label():
    rep = _rep()
    assert rep["protocol"]["missing_caches"] == []
    assert rep["protocol"]["labels"] == list(C.LABELS)
    for sub in C.SUBSETS:
        assert set(rep["per_split"][sub]) == set(C.SPLITS)
        assert set(rep["pooled"][sub]) == set(C.POOLS)


def _cell(d):
    return f"{d['observed']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


@pytest.mark.parametrize("pool,thr,metric", [
    ("US pooled basis (miss_decomposition.US_SPLITS)", "0.30", "f1"),
    ("US pooled basis (miss_decomposition.US_SPLITS)", "0.55", "f1"),
    ("US pooled basis (miss_decomposition.US_SPLITS)", "0.30", "max_f1"),
    ("open-rig 4 (laurens_mapillary+bayonne+juneau+budapest_district5)", "0.30", "f1"),
    ("open-rig 4 (laurens_mapillary+bayonne+juneau+budapest_district5)", "0.55", "f1"),
    ("open-rig 4 (laurens_mapillary+bayonne+juneau+budapest_district5)", "0.30", "max_f1"),
    ("manual_gold", "0.30", "f1"),
    ("manual_gold", "0.30", "max_f1"),
])
def test_doc_quotes_the_committed_seed_mean_contrasts(pool, thr, metric):
    d = _rep()["pooled"]["all"][pool]["seed_mean_contrasts"][SM][thr][metric]
    with open(DOC, encoding="utf-8") as f:
        assert _cell(d) in f.read()


def test_doc_quotes_the_committed_laurens_rig_effect_change():
    lp = _rep()["laurens_paired"]["rig_effect_vs_control_seed_mean"][
        "control (seed mean) - released"]
    with open(DOC, encoding="utf-8") as f:
        doc = f.read()
    assert _cell(lp["0.30"]["f1"]) in doc and _cell(lp["max_f1"]["f1"]) in doc


def test_every_fine_tune_is_below_released_on_laurens_mapillary_max_f1():
    """The #82 finding, now on five fine-tunes of the same recipe."""
    m = _rep()["per_split"]["all"]["laurens_mapillary"]["max_f1"]
    assert all(v < m["released"] for k, v in m.items() if k != "released")
