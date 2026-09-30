"""rampnet.cluster_review (issue #224): the loader, every validate() refusal, the
pre-registered inter-rater agreement on a hand-built pair, summary and the verdicts.json
consistency check. Synthetic files only.

    pytest tests/test_cluster_review.py -v
"""
import json

import pytest

from rampnet import cluster_review as cr

SHA = "a" * 64
LAT, LNG = 45.6, -122.6
M = 1 / 111320.0            # ~1 m of latitude, in degrees


def corner(cid="c:sig:000001", keys=("1", "2", "3", "4"), window=30.0):
    return {"corner_id": cid, "type": "signalised", "has_labels": bool(keys),
            "centre": {"lat": LAT, "lng": LNG}, "window_m": window,
            "labels": [{"key": k, "pano_id": f"P{k}", "pano_x": 100 * int(k), "pano_y": 50,
                        "lat": LAT + int(k) * M, "lng": LNG} for k in keys]}


def unit(labels, ramps=None, uncovered=(), complete=True, seed="deployed", elapsed=10.0):
    if ramps is None:
        ramps = {r: {"lat": LAT, "lng": LNG + 0.0002} for r in set(labels.values())
                 if cr.is_ramp(r)}
    return {"seed_arm": seed, "stratum": {"city": "c", "type": "signalised", "has_labels": True},
            "labels": labels, "ramps": ramps, "uncovered": list(uncovered),
            "complete": complete, "elapsed_s": elapsed, "note": ""}


def assignments(corners_map, rater=None, rubric=1, sha=SHA):
    return {"schema": cr.SCHEMA, "city": "c", "snapshot_sha256": sha, "rubric_version": rubric,
            "seed_arm": "deployed", "rater": rater, "corners": corners_map}


SNAP = {"schema": cr.SNAPSHOT_SCHEMA, "city": "c", "labels": {"sha256": SHA}}
GOOD = unit({"1": "r1", "2": "r1", "3": "r2", "4": "not_ramp"})


def test_load_bundle_round_trip(tmp_path):
    (tmp_path / "snapshot.json").write_text(json.dumps(SNAP), encoding="utf-8")
    (tmp_path / "corners.jsonl").write_text(json.dumps(corner()) + "\n", encoding="utf-8")
    snap, corners, files = cr.load_bundle(tmp_path)
    assert files == {} and corners[0]["corner_id"] == "c:sig:000001"
    a = assignments({"c:sig:000001": GOOD})
    (tmp_path / "assignments.json").write_text(json.dumps(a), encoding="utf-8")
    (tmp_path / "assignments__mikey.json").write_text(json.dumps(a), encoding="utf-8")
    _s, _c, files = cr.load_bundle(tmp_path)
    assert sorted(files) == ["assignments.json", "assignments__mikey.json"]
    assert cr.validate(files["assignments.json"], corners, snap) == []
    assert cr.rater_file_name(None) == "assignments.json"
    assert cr.rater_file_name("mikey") == "assignments__mikey.json"
    with pytest.raises(ValueError):
        cr.rater_file_name("../x")


@pytest.mark.parametrize("mutate, needle", [
    (lambda a: a.update(schema="x"), "schema"),
    (lambda a: a.update(snapshot_sha256="b" * 64), "snapshot_sha256"),
    (lambda a: a.update(rubric_version="1"), "rubric_version"),
    (lambda a: a["corners"].update({"c:zzz:000009": GOOD}), "not a unit"),
    (lambda a: a["corners"]["c:sig:000001"]["labels"].update({"99": "r1"}), "not in the unit"),
    (lambda a: a["corners"]["c:sig:000001"]["labels"].update({"4": "ramp!"}), "has value"),
    (lambda a: a["corners"]["c:sig:000001"]["labels"].pop("4"), "unassigned"),
    (lambda a: a["corners"]["c:sig:000001"]["ramps"].pop("r2"), "not in ramps"),
    (lambda a: a["corners"]["c:sig:000001"]["ramps"].update({"r7": {"lat": LAT, "lng": LNG}}),
     "holds no label"),
    (lambda a: a["corners"]["c:sig:000001"]["uncovered"].append(
        {"lat": LAT + 40 * M, "lng": LNG, "unsure": False}), "window"),
    (lambda a: a["corners"]["c:sig:000001"]["uncovered"].append(
        {"lat": LAT, "lng": LNG + 0.0002, "unsure": False}), "sits on ramp"),
    (lambda a: a["corners"]["c:sig:000001"].update(elapsed_s=-1), "elapsed_s"),
])
def test_validate_refuses(mutate, needle):
    a = assignments({"c:sig:000001": json.loads(json.dumps(GOOD))})
    assert cr.validate(a, [corner()], SNAP) == []
    mutate(a)
    problems = cr.validate(a, [corner()], SNAP)
    assert problems and any(needle in p for p in problems), problems
    with pytest.raises(ValueError):
        cr.require_valid(a, [corner()], SNAP)


def test_incomplete_unit_may_leave_labels_unassigned():
    u = unit({"1": "r1"}, complete=False)
    assert cr.validate(assignments({"c:sig:000001": u}), [corner()], SNAP) == []


def test_pairs_only_between_ramp_labels():
    p = cr.pairs(GOOD)
    assert p == {("1", "2"): True, ("1", "3"): False, ("2", "3"): False}


def test_agreement_on_a_hand_built_pair():
    # unit 1: A = {1,2}{3}, 4 not_ramp; B = {1}{2,3}, 4 not_ramp.
    #   pairs (1,2): A same, B diff -> disagree; (1,3): diff/diff agree; (2,3): diff/same no.
    # unit 2: A and B identical: 1,2,3 one ramp, 4 unsure in B -> kappa skips it.
    #   pairs (1,2),(1,3),(2,3) all agree.
    a = assignments({"u1": unit({"1": "r1", "2": "r1", "3": "r2", "4": "not_ramp"},
                                uncovered=[{"lat": LAT, "lng": LNG, "unsure": False}]),
                     "u2": unit({"1": "r1", "2": "r1", "3": "r1", "4": "not_ramp"}),
                     "u3": unit({"1": "r1"}, complete=False)}, rater="a")
    b = assignments({"u1": unit({"1": "r1", "2": "r2", "3": "r2", "4": "not_ramp"}, seed="fusion"),
                     "u2": unit({"1": "r1", "2": "r1", "3": "r1", "4": "unsure"}),
                     "u3": unit({"1": "r1"})}, rater="b")
    rep = cr.agreement(a, b)
    assert rep["units"]["both"] == 2
    assert (rep["pairwise"]["agree"], rep["pairwise"]["pairs"]) == (4, 6)
    assert rep["pairwise"]["rate"] == pytest.approx(4 / 6)
    assert (rep["pairwise_different_seed"]["agree"], rep["pairwise_different_seed"]["pairs"]) == (1, 3)
    assert (rep["pairwise_same_seed"]["agree"], rep["pairwise_same_seed"]["pairs"]) == (3, 3)
    # kappa over labels u1:1-4, u2:1-3 (u2:4 is unsure in B): both raters say not_ramp on
    # u1:4 only -> perfect agreement on a non-constant vector
    assert rep["not_ramp_kappa"]["labels"] == 7 and rep["not_ramp_kappa"]["kappa"] == pytest.approx(1.0)
    assert rep["uncovered"] == {"total_a": 1, "total_b": 0, "abs_diff_per_unit": {0: 1, 1: 1}}
    assert rep["pilot"]["pass"] is False           # 0.667 < 0.90


def test_agreement_refuses_different_rubric_or_snapshot():
    a = assignments({"u1": GOOD})
    with pytest.raises(ValueError, match="rubric_version"):
        cr.agreement(a, assignments({"u1": GOOD}, rubric=2))
    with pytest.raises(ValueError, match="snapshot"):
        cr.agreement(a, assignments({"u1": GOOD}, sha="c" * 64))


def test_summary_counts_only_complete_units():
    a = assignments({"u1": GOOD, "u2": unit({"1": "unsure"}, elapsed=30.0),
                     "u3": unit({"1": "r1"}, complete=False, elapsed=99.0)})
    s = cr.summary(a)
    assert s["units"] == 3 and s["complete"] == 2
    assert s["labels"] == {"ramp": 3, "not_ramp": 1, "unsure": 1}
    assert s["ramps"] == 2 and s["elapsed_s_median"] == 20.0


def test_verdict_consistency_flags_disagreements():
    corners = [corner()]
    a = assignments({"c:sig:000001": GOOD})
    # label 1 at pixel (100, 50) and label 4 at (400, 50) on their panos
    records = {"P1": {"pano": {"width": 1000, "height": 100},
                      "detections": [{"x_normalized": 0.1, "y_normalized": 0.5}]},
               "P4": {"pano": {"width": 1000, "height": 100},
                      "detections": [{"x_normalized": 0.4, "y_normalized": 0.5}]}}
    verdicts = {"P1": {"dets": [True]}, "P4": {"dets": [True]}}
    out = cr.verdict_consistency(a, corners, verdicts, records)
    assert out["checked"] == 2
    assert [d["key"] for d in out["disagree"]] == ["4"]     # judged a ramp, reviewed not_ramp
