"""Tests for s8_prep.py (crosswalk_s8 pre-registration prep): seam-wrapped distance, and that the held-out slice
and the primary's pano sample reproduce from the committed inputs exactly as committed.

Reads only committed files under analysis_out/crosswalk_s8/inputs/ and analysis_out/crosswalk_s8/slice/.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274"))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import s8_prep as sp  # noqa: E402

SLICE = os.path.join(sp.OUT, "slice")


def test_wrap_dx_crosses_the_seam():
    assert sp.wrap_dx(0.99, 0.01) == pytest.approx(0.02)
    assert sp.wrap_dx(0.30, 0.10) == pytest.approx(0.20)


@pytest.mark.skipif(not os.path.exists(os.path.join(SLICE, "panos.csv")), reason="slice not built")
def test_slice_and_sample_reproduce_from_committed_inputs():
    pt, lab, _ = sp.load_inputs()
    committed = pd.read_csv(os.path.join(SLICE, "panos.csv"), dtype={"pano_id": str})
    summ = __import__("json").load(open(os.path.join(SLICE, "summary.json")))
    p = summ["params"]
    in_slice, areas, _ = sp.grow_slice(pt, lab.groupby("pano_id").size(), p["seed"], p["areas"],
                                       p["target_label_share"], p["min_sep_m"], p["cell_m"])
    got = set(pt.pano_id[in_slice >= 0])
    assert got == set(committed.pano_id[committed.role == "slice"])
    assert [a["centre_pano"] for a in areas] == [a["centre_pano"] for a in summ["areas"]]
    sample = open(os.path.join(SLICE, "primary_sample.txt")).read().split()
    reserve = open(os.path.join(SLICE, "primary_reserve.txt")).read().split()
    order = np.random.default_rng(p["seed"] + 1).permutation(pt.pano_id[in_slice >= 0].to_numpy())
    assert list(order[:len(sample)]) == sample and list(order[len(sample):len(sample) + len(reserve)]) == reserve
    assert p["buffer_m"] == sp.BUFFER_M
    # The buffer: every non-slice pano within buffer_m of the slice, and nothing farther.
    buf = committed[committed.role == "buffer"]
    assert (buf.dist_to_slice_m <= p["buffer_m"]).all()
    assert (committed[committed.role == "train"].dist_to_slice_m > p["buffer_m"]).all()


POOLS = os.path.join(sp.OUT, "pools")


@pytest.mark.skipif(not os.path.exists(os.path.join(POOLS, "candidates.csv")), reason="pools not built")
def test_pool_invariants():
    import json
    c = pd.read_csv(os.path.join(POOLS, "candidates.csv"), dtype={"pano_id": str, "seq": str})
    summ = json.load(open(os.path.join(POOLS, "summary.json")))
    planned = c[c.planned.isin(["positive", "negative", "mined_for_jon_page"])]
    # No planned point on a slice, buffer, recall285, PS-labelled or root_v5 pano.
    assert (planned.role == "train").all() and planned.eligible.all()
    assert not planned.on_recall285_or_ps_pano.any() and not planned.on_root_v5_pano.any()
    panos = pd.read_csv(os.path.join(sp.OUT, "slice", "panos.csv"), dtype={"pano_id": str})
    bad = set(panos.pano_id[panos.role != "train"]) | set(panos.pano_id[panos.recall285 | panos.ps_labelled])
    assert not set(planned.pano_id) & bad
    # Caps.
    negs = planned[planned.planned.isin(["negative", "mined_for_jon_page"])]
    assert negs.groupby("pano_id").size().max() <= sp.NEG_PER_PANO
    assert negs.groupby("seq").size().max() <= sp.NEG_PER_SEQ
    assert negs[negs.rear_seam].groupby("seq").size().max() <= sp.SEAM_PER_SEQ
    assert (planned.planned == "mined_for_jon_page").sum() <= sp.MINED_CAP
    assert not negs.neg_near_crosswalk.any()
    # Mined candidates are validator-sourced and never train without Jon's verdict; validator positives are GoPro only.
    assert (planned[planned.planned == "mined_for_jon_page"].source == "validator_negative_candidate").all()
    vpos = planned[(planned.planned == "positive") & (planned.source == "validator_positive_candidate")]
    assert (vpos.camera_model == sp.GOPRO).all()
    # Juneau positives <= Juneau negatives in every branch; GoPro positives >= GoPro negatives.
    br = summ["planned"]["branches"]
    assert all(b["rule_holds"] for k, b in br.items() if isinstance(b, dict))
    assert summ["planned"]["gopro_positives"] >= summ["planned"]["gopro_negatives_after_balance"]
