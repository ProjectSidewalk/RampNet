"""Tests for the #48 multi-view evidence scripts.

CPU only, no network, no labeler checkout: the pure functions are checked on toy inputs,
and the committed artifacts under ``analysis_out/multiview_48/`` are checked for LF-pinned
bytes and re-derived from the committed per-capture table where they can be.
"""
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import multiview_evidence_48 as mv  # noqa: E402
from rampnet.detection_eval import GroundTruth  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "multiview_48")


# --------------------------------------------------------------------------- #
# per-capture claims
# --------------------------------------------------------------------------- #
def test_one_detection_between_two_dual_ramps_claims_only_the_nearer():
    claims = mv.claim_by_confidence([(1.0, 0.0, 0.9)], [("a", 0.0, 0.0), ("b", 3.0, 0.0)],
                                    mv.world_d2, 25.0)
    assert claims == {"a": 0.9}


def test_claims_are_made_in_confidence_order():
    # the confident detection takes the ramp it is nearest to; the weak one gets the other
    dets = [(0.5, 0.0, 0.3), (0.4, 0.0, 0.8)]
    claims = mv.claim_by_confidence(dets, [("a", 0.0, 0.0), ("b", 2.0, 0.0)], mv.world_d2, 25.0)
    assert claims == {"a": 0.8, "b": 0.3}


def test_claims_at_a_floor_do_not_depend_on_lower_detections():
    targets = [("a", 0.0, 0.0), ("b", 2.0, 0.0)]
    dets = [(0.5, 0.0, 0.3), (1.9, 0.0, 0.9), (0.2, 0.0, 0.6)]
    full = mv.claim_by_confidence(dets, targets, mv.world_d2, 25.0)
    top = mv.claim_by_confidence([d for d in dets if d[2] >= 0.55], targets, mv.world_d2, 25.0)
    assert {k: v for k, v in full.items() if v >= 0.55} == top


def test_pixel_distance_wraps_at_the_seam():
    assert mv.pixel_d2(0.999, 0.5, 0.001, 0.5) < mv.pixel_radius_sq()


# --------------------------------------------------------------------------- #
# B.1 / B.2 on toy ramps
# --------------------------------------------------------------------------- #
def _cap(pid, d, hit, source=False, e=0.0, n=0.0):
    return {"pano_id": pid, "dist_m": d, "is_source": source, "world_conf": 0.9 if hit else None,
            "pixel_conf": None, "cam_e": e, "cam_n": n}


def test_k_nearest_uses_the_nearest_other_views_first():
    ramps = [{"captures": [_cap("s", 5, True, source=True), _cap("a", 4, False),
                           _cap("b", 9, True), _cap("c", 30, True)]}]
    rows = mv.recall_k_nearest(ramps, 0.55, "world", 18.0, kmax=2, fixed_population=True)
    assert [r["recall"] for r in rows] == [0.0, 1.0]         # c is beyond 18 m, s is the source
    assert mv.recall_k_nearest(ramps, 0.55, "world", 18.0, 3, True)[0]["ramps"] == 0


def test_recall_by_capture_count_bins_on_other_views():
    ramps = [{"captures": [_cap("s", 5, True, source=True), _cap("a", 4, False)]},
             {"captures": [_cap("s2", 5, False, source=True), _cap("b", 4, True), _cap("c", 6, True)]}]
    rows = {r["n_bin"]: r for r in mv.recall_by_capture_count(ramps, 0.55, "world", 18.0)}
    assert rows["1"]["ramps"] == 1 and rows["1"]["recall_other"] == 0.0
    assert rows["1"]["recall_union"] == 1.0
    assert rows["2"]["ramps"] == 1 and rows["2"]["recall_other"] == 1.0


def test_failure_correlation_flags_perfectly_correlated_misses():
    # ten ramps each seen by two views at the same range: five all-hit, five all-miss.
    ramps = []
    for i in range(10):
        hit = i < 5
        ramps.append({"captures": [_cap(f"a{i}", 8, hit, e=0, n=0), _cap(f"b{i}", 8, hit, e=4, n=0)]})
    fc = mv.failure_correlation(ramps, 0.55, "world", 18.0)
    assert fc["all_missed"]["observed"] == 5
    assert fc["all_missed"]["predicted_independent"] == pytest.approx(10 * 0.25)
    row = fc["by_separation"][0]
    assert row["p_miss_given_miss"] == 1.0 and row["p_miss_marginal"] == 0.5


def test_failure_correlation_is_one_under_independence():
    # every miss pattern equally often -> joint miss rate equals the product of marginals
    ramps = []
    for a in (True, False):
        for b in (True, False):
            ramps.append({"captures": [_cap("x", 8, a, e=0), _cap("y", 8, b, e=4)]})
    fc = mv.failure_correlation(ramps, 0.55, "world", 18.0)
    assert fc["by_separation"][0]["ratio"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# per-pano classification and the world scorer
# --------------------------------------------------------------------------- #
def test_classify_preds_follows_score_pano():
    gt = GroundTruth([(0.5, 0.5)], [(0.2, 0.5)], True)
    out = mv.classify_preds([("hi", 0.5, 0.5, 0.9), ("dup", 0.501, 0.5, 0.8),
                             ("ign", 0.2, 0.5, 0.7), ("far", 0.8, 0.5, 0.6)], gt)
    assert out == {"hi": ("tp", 0), "dup": ("fp", None), "ign": ("ignored", None),
                   "far": ("fp", None)}


def test_match_one_to_one_is_greedy_by_distance():
    assert mv.match_one_to_one([(0, 0), (1, 0)], [("s", 0.9, 0)], 5.0) == {1: "s"}


def test_score_world_self_claim_needs_an_accepted_site():
    pool = [{"e": 0.0, "n": 0.0, "gt_refs": [("p", 0)]}]
    sites = [{"id": 1, "e": 20.0, "n": 0.0, "members": [("p", 7, 0.4)]}]
    pc = {("p", 7): ("tp", 0)}
    assert mv.score_world(pool, sites, {1}, pc)["recall"] == 1.0      # self, site far away
    assert mv.score_world(pool, sites, set(), pc)["recall"] == 0.0    # its site not submitted


def test_score_world_precision_rule():
    pool = [{"e": 0.0, "n": 0.0, "gt_refs": []}]
    sites = [{"id": 1, "e": 0.0, "n": 0.0, "members": [("p", 1, 0.9), ("q", 1, 0.9)]},
             {"id": 2, "e": 50.0, "n": 0.0, "members": [("p", 2, 0.9)]},
             {"id": 3, "e": 90.0, "n": 0.0, "members": [("p", 3, 0.9)]},
             {"id": 4, "e": 70.0, "n": 0.0, "members": [("z", 1, 0.9)]}]
    pc = {("p", 1): ("fp", None), ("q", 1): ("tp", 0), ("p", 2): ("fp", None),
          ("p", 3): ("ignored", None)}
    r = mv.score_world(pool, sites, {1, 2, 3, 4}, pc)
    assert (r["tp"], r["fp"], r["unsure_only"]) == (1, 1, 1)         # site 4 has no judged member
    assert r["recall"] == 1.0                                         # matched by site 1


def test_kofn_keeps_operational_sites_and_counts_distinct_panos():
    sites = [{"id": 1, "members": [("a", 0, 0.9)]},
             {"id": 2, "members": [("a", 0, 0.4), ("b", 0, 0.35)]},
             {"id": 3, "members": [("a", 0, 0.4), ("b", 0, 0.2)]}]
    assert mv.policy_kofn(sites, 2) == {1, 2}
    assert mv.policy_kofn(sites, 1) == {1, 2, 3}


# --------------------------------------------------------------------------- #
# evidence model
# --------------------------------------------------------------------------- #
def test_evidence_score_counts_misses_in_range_as_negative():
    real = [(True, [(5, 0.9), (8, 0.8)]) for _ in range(20)]
    false = [(False, [(5, None), (8, None), (9, 0.35)]) for _ in range(20)]
    model = mv.fit_evidence_model(real + false)
    assert mv.capture_llr(model, 8, None) < 0              # a miss where real ramps are seen
    assert mv.capture_llr(model, 8, 0.9) > 0
    assert mv.evidence_score(model, [(5, 0.9), (8, 0.8)]) > \
        mv.evidence_score(model, [(5, 0.9), (8, 0.8), (7, None), (9, None)])
    assert mv.capture_llr(model, 40, None) == 0.0         # outside every range bin


def test_residual_class_precedence():
    caps = [_cap("s", 8, False, source=True)]
    assert mv.residual_class({"captures": caps}, True) == "coverage_gap"
    caps.append(_cap("o", 10, False))
    assert mv.residual_class({"captures": caps}, True) == "never_fired"
    assert mv.residual_class({"captures": caps}, False) == "never_fired_unknown_below_055"
    caps[1]["world_conf"] = 0.2
    assert mv.residual_class({"captures": caps}, True) == "sub_threshold_only"
    caps[0]["pixel_conf"] = 0.6
    assert mv.residual_class({"captures": caps}, True) == "association_placement"


# --------------------------------------------------------------------------- #
# the committed artifacts
# --------------------------------------------------------------------------- #
def test_write_json_is_lf_and_rounded(tmp_path):
    p = mv.write_json(str(tmp_path / "x.json"), {"a": 1 / 3, "b": [2 / 3]})
    raw = open(p, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert json.loads(raw) == {"a": 0.3333, "b": [0.6667]}


COMMITTED = ["meta.json", "recall_vs_captures.json", "evidence_vs_kofn.json",
             "residual_misses.json", "captures_R25.csv"]


@pytest.mark.parametrize("name", COMMITTED)
def test_committed_outputs_are_lf_pinned(name):
    path = os.path.join(OUT, name)
    # A committed artifact that has gone missing is a failure, not a skip.
    assert os.path.exists(path), f"{name} is missing from analysis_out/multiview_48/"
    assert b"\r" not in open(path, "rb").read()


def test_recall_tables_rederive_from_the_committed_capture_table():
    csv_path = os.path.join(OUT, "captures_R25.csv")
    rv_path = os.path.join(OUT, "recall_vs_captures.json")
    meta_path = os.path.join(OUT, "meta.json")
    for p in (csv_path, rv_path, meta_path):
        assert os.path.exists(p), f"{p} is missing (a committed #48 artifact)"
    meta = json.load(open(meta_path, encoding="utf-8"))
    ramps = mv.ramps_from_capture_csv(csv_path)
    for city, info in meta["city_info"].items():
        assert len(ramps[city]) == info["pool_ramps"], city
    got = mv.rnd(mv.b1_b2(ramps, meta["sub_threshold_cities"]))
    want = json.load(open(rv_path, encoding="utf-8"))
    for key in ("pooled|0.55|world|R18", "gsv|0.30|world|R18", "mapillary|0.10|either|R12"):
        assert got[key] == want[key], key


# --------------------------------------------------------------------------- #
# re-run vs published agreement (multiview_challengers_48.detection_agreement)
# --------------------------------------------------------------------------- #
def test_agreement_is_not_fooled_by_export_rounding_across_a_4dp_boundary():
    import multiview_challengers_48 as mc
    published = 0.12344996                  # full precision, as benchmark/model_detections keeps it
    exported = round(published, 5)          # 0.12345, as ``export`` writes it
    assert round(published, 4) != round(exported, 4)   # the old 4-dp buckets split these
    mine = {"a": [[exported, 0.5, 0.9]], "b": []}
    pub = {"a": [[published, 0.5, 0.9]], "b": []}
    assert mc.detection_agreement(mine, pub, ["a", "b"]) == {"same_count": 1.0,
                                                             "same_detections": 1.0}


def test_agreement_separates_count_from_position():
    import multiview_challengers_48 as mc
    pub = {"a": [[0.10, 0.5]], "b": [[0.30, 0.5]], "c": [[0.7, 0.5]]}
    mine = {"a": [[0.10, 0.5]],                       # same
            "b": [[0.30 + 2 * mc.AGREE_TOL, 0.5]],    # same count, moved box
            "c": [[0.7, 0.5], [0.9, 0.5]]}            # extra box
    got = mc.detection_agreement(mine, pub, ["a", "b", "c"])
    assert got == {"same_count": pytest.approx(2 / 3), "same_detections": pytest.approx(1 / 3)}
    assert mc.detection_agreement(mine, pub, []) is None
