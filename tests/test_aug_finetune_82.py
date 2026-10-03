"""#82 Step 3 read: the scoring instrument and the committed report.

CPU only; reads the committed caches under analysis_out/aug_transfer_82/finetune/.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
P = pytest.importorskip("aug_probe_82")
F = pytest.importorskip("aug_finetune_82")

FT = os.path.join(REPO, "analysis_out", "aug_transfer_82", "finetune")
R2048 = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "cache", "r2048")


def _preds(path):
    with open(path, encoding="utf-8") as f:
        return {p["pano"]: [tuple(t) for t in p["preds"]] for p in json.load(f)["panos"]}


@pytest.mark.parametrize("split", sorted(os.path.splitext(f)[0] for f in os.listdir(R2048)))
def test_klone_scorer_reproduces_r2048_for_the_released_checkpoint(split):
    """The released checkpoint scored on klone (job 41103600) is the #25 instrument:
    same peaks on every pano, scores within cross-machine fp32 noise."""
    mine = _preds(os.path.join(FT, "released", f"{split}.json"))
    bad, worst, det = P.compare_preds(mine, _preds(os.path.join(R2048, f"{split}.json")))
    assert bad == 0, det
    assert worst < 1e-4


def test_manual_gold_scored_in_full():
    with open(os.path.join(FT, "released", "manual_gold.json"), encoding="utf-8") as f:
        d = json.load(f)
    assert len(d["panos"]) == 1000 and d["meta"]["fp16"] is False and d["meta"]["tta"] is False


RESULTS = os.path.join(REPO, "analysis_out", "aug_transfer_82", "finetune_results.json")
DOC = os.path.join(REPO, "docs", "aug_transfer_82.md")


def _rep():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def test_all_eight_finetunes_scored_on_all_twelve_bundles():
    rep = _rep()
    labels = rep["protocol"]["labels"]
    assert labels == ["released"] + [f"{a}_s{s}" for a in F.ARMS for s in F.SEEDS]
    for c in F.SPLITS:
        assert set(rep["per_split"][c]["metrics"]) == set(labels), c


def _cell(d):
    return f"{d['observed']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]".replace("-", "−")


@pytest.mark.parametrize("name,read", [
    ("spread: control_s2 - control_s1", "max_f1"),
    ("res - control (seed mean)", "max_f1"),
    ("photo - control (seed mean)", "max_f1"),
    ("both - control (seed mean)", "max_f1"),
    ("control (seed mean) - released", "max_f1"),
])
def test_doc_quotes_the_committed_rig_effect_did(name, read):
    """The difference-in-differences numbers in docs/aug_transfer_82.md are the committed ones."""
    lp = _rep()["laurens_paired"]
    table = (lp["rig_effect_vs_control"] if name.startswith("spread")
             else lp["rig_effect_vs_control_seed_mean"])
    with open(DOC, encoding="utf-8") as f:
        doc = f.read()
    assert _cell(table[name][read]["f1"]) in doc


@pytest.mark.parametrize("name", ["control (seed mean) - released", "res - control (seed mean)",
                                  "photo - control (seed mean)", "both - control (seed mean)"])
def test_doc_quotes_the_committed_transfer_pool_max_f1(name):
    pool = _rep()["pooled"]["transfer (laurens_mapillary+clovis+richmond)"]
    with open(DOC, encoding="utf-8") as f:
        doc = f.read()
    assert _cell(pool["seed_mean_contrasts"][name]["0.30"]["max_f1"]) in doc


def test_no_finetune_beats_released_on_laurens_mapillary_max_f1():
    m = _rep()["per_split"]["laurens_mapillary"]["max_f1"]
    assert all(v < m["released"] for k, v in m.items() if k != "released")
