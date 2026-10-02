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
