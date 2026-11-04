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
    (lambda a: a["corners"]["c:sig:000001"].update(inventory_seen="yes"), "not a boolean"),
    (lambda a: a["corners"]["c:sig:000001"].update(edited_after_inventory=True),
     "without inventory_seen"),
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
    # no corners: the pilot units are unknown, so the gate decides nothing (Amendment 5, S5)
    assert rep["pilot"]["pass"] is None


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


# --- can't judge (protocol, Amendment 3) ------------------------------------------------

def cj_unit(reason="under an overpass; the aerial shows the deck", complete=False, cj=True):
    u = unit({"1": "r1"}, complete=complete)
    u["cant_judge"], u["cant_judge_reason"] = cj, reason
    return u


def test_cant_judge_valid_with_a_reason_and_partial_labels():
    assert cr.validate(assignments({"c:sig:000001": cj_unit()}), [corner()], SNAP) == []


@pytest.mark.parametrize("u, needle", [
    (cj_unit(complete=True), "cant_judge and complete"),
    (cj_unit(reason="   "), "without a cant_judge_reason"),
    (cj_unit(reason=""), "without a cant_judge_reason"),
    (cj_unit(cj="yes"), "is not a boolean"),
])
def test_cant_judge_refusals(u, needle):
    problems = cr.validate(assignments({"c:sig:000001": u}), [corner()], SNAP)
    assert any(needle in p for p in problems), problems


def test_cant_judge_is_not_counted_and_is_reported():
    a = assignments({"c:sig:000001": GOOD, "c:sig:000002": cj_unit()})
    assert list(cr.complete_units(a)) == ["c:sig:000001"]
    s = cr.summary(a)
    assert s["complete"] == 1 and s["cant_judge"] == 1 and s["cant_judge_by_type"] == {"signalised": 1}
    assert cr.cant_judge_units(a) == {"c:sig:000002": "under an overpass; the aerial shows the deck"}


def test_agreement_counts_cant_judge_disagreements():
    a = assignments({"c:sig:000001": GOOD, "c:sig:000002": cj_unit(), "c:sig:000003": cj_unit()})
    b = assignments({"c:sig:000001": cj_unit(), "c:sig:000002": GOOD, "c:sig:000003": cj_unit()},
                    rater="retest")
    r = cr.agreement(a, b)
    assert r["units"]["both"] == 0
    assert r["cant_judge"] == {"a": 2, "b": 2, "either": 3, "both": 1,
                               "complete_a_cant_judge_b": ["c:sig:000001"],
                               "cant_judge_a_complete_b": ["c:sig:000002"]}


# --- Amendment 5: pair F1 gate (B1), kappa (S5), washout (S3), kind (S4), 1 m rule (S9) ---

def test_pair_f1_on_a_hand_built_pair_and_pilot_needs_corners():
    a = assignments({"u1": unit({"1": "r1", "2": "r1", "3": "r2"}),
                     "u2": unit({"1": "r1", "2": "r1", "3": "r1"})}, rater="a")
    b = assignments({"u1": unit({"1": "r1", "2": "r2", "3": "r2"}, seed="fusion"),
                     "u2": unit({"1": "r1", "2": "r1", "3": "r1"})}, rater="b")
    rep = cr.agreement(a, b)
    # u1: A {1,2} vs B {2,3} -> same pairs 1, 1, both 0; u2: 3, 3, 3 -> pooled 6/8,
    # per-unit mean (0 + 1) / 2
    assert rep["pair_f1"]["pooled"] == pytest.approx(0.75)
    assert rep["pair_f1"]["per_unit_mean"] == pytest.approx(0.5)
    assert rep["pair_f1_different_seed"]["pooled"] == pytest.approx(0.0)
    assert rep["pilot"]["pass"] is None and "corners" in rep["pilot"]["why_none"]
    cs = [dict(corner("u1", keys=("1", "2", "3")), pilot=True),
          dict(corner("u2", keys=("1", "2", "3")), pilot=True)]
    rep = cr.agreement(a, b, cs)
    assert rep["pilot"]["full_pilot"] is True and rep["pilot"]["pass"] is False


def _seed_pass(c, arm):
    """A pass that accepts the unit's seed grouping in ``arm`` as it is (a label without a
    seed in that arm becomes a ramp of its own) and completes the unit."""
    labels, name = {}, {}
    for lab in c["labels"]:
        g = (lab.get("seed_group") or {}).get(arm) or ("solo", lab["key"])
        labels[lab["key"]] = name.setdefault(g, f"r{len(name) + 1}")
    return unit(labels, ramps={r: {"lat": LAT, "lng": LNG} for r in name.values()}, seed=arm)


def test_new_pilot_gate_fails_a_rater_who_accepts_every_seed():
    # Review B1, on the committed Vancouver bundle: pass 1 accepts the deployed seed, the
    # retest accepts each unit's rater_b_seed. The old Rand-style rule passes this rater
    # (about 0.96); the Amendment 5 pair-F1 gate on different-seed units must not.
    corners = cr.load_corners("benchmark/vancouver/cluster_review/corners.jsonl")
    pilot = [c for c in corners if c.get("pilot")]
    assert len(pilot) == 30
    a = assignments({c["corner_id"]: _seed_pass(c, "deployed") for c in pilot}, rater="jon")
    b = assignments({c["corner_id"]: _seed_pass(c, c["rater_b_seed"]) for c in pilot},
                    rater="jon-retest")
    rep = cr.agreement(a, b, corners)
    assert rep["agreement_kind"] == "intra"
    assert rep["pairwise"]["rate"] >= cr.PILOT_MIN_PAIRWISE         # the old gate: passed
    p = rep["pilot"]
    assert p["full_pilot"] is True and p["kappa_leg"] is True       # undefined, identical
    assert p["pair_f1_leg"] is False and p["pass"] is False
    d = p["pair_f1_different_seed"]
    assert d["pooled"] < 0.85 and d["per_unit_mean"] < cr.PILOT_MIN_PAIR_F1
    # accepting every seed scores the seed arms' own agreement on those units (not exactly:
    # the one label with no deployed seed is a ramp of its own here, absent from the reference)
    assert d["pooled"] == pytest.approx(p["seed_reference"]["pooled"], abs=0.005)
    lo, hi = d["pooled_ci95"]
    assert lo <= d["pooled"] <= hi
    # ... while a rater who reads every unit the same way twice passes
    same = assignments({c["corner_id"]: dict(a["corners"][c["corner_id"]],
                                              seed_arm=c["rater_b_seed"]) for c in pilot},
                       rater="jon-retest")
    assert cr.agreement(a, same, corners)["pilot"]["pass"] is True


def test_identical_passes_without_not_ramp_pass_the_kappa_leg():
    # S5: kappa is undefined when both passes are constant and identical: full agreement
    u = unit({"1": "r1", "2": "r1", "3": "r2"})
    a = assignments({"u1": u}, rater="x")
    b = assignments({"u1": dict(u, seed_arm="fusion")}, rater="x-retest")
    cs = [dict(corner("u1", keys=("1", "2", "3")), pilot=True)]
    rep = cr.agreement(a, b, cs)
    assert rep["not_ramp_kappa"]["kappa"] is None
    assert rep["not_ramp_kappa"]["undefined_identical"] is True
    assert rep["pilot"]["kappa_leg"] is True and rep["pilot"]["pass"] is True
    assert cr.kappa_leg([True, False], [True, True])[1] is False    # defined and low


def test_pilot_pass_is_none_for_a_partial_pilot():
    # S5: the gate reads the pilot units from corners, and decides only on the full pilot
    u = unit({"1": "r1", "2": "r1"})
    cs = [dict(corner("u1", keys=("1", "2")), pilot=True),
          dict(corner("u2", keys=("1", "2")), pilot=True)]
    rep = cr.agreement(assignments({"u1": u, "u2": u}), assignments({"u1": u}, rater="x-retest"),
                       cs)
    assert rep["pilot"]["full_pilot"] is False and rep["pilot"]["pass"] is None


def test_washout_days_flag_units_under_seven():
    ua = dict(GOOD, completed_at="2026-11-01T10:00:00.000Z")
    a = assignments({"u1": ua, "u2": ua, "u3": GOOD})
    b = assignments({"u1": dict(GOOD, opened_at="2026-11-05T10:00:00.000Z",
                                completed_at="2026-11-09T10:00:00.000Z"),
                     "u2": dict(GOOD, opened_at="2026-11-08T22:00:00.000Z"),
                     "u3": dict(GOOD, opened_at="2026-11-20T00:00:00.000Z")}, rater="r-retest")
    w = cr.agreement(a, b)["washout"]
    assert w["under_min"] == ["u1"] and w["missing"] == ["u3"]
    assert w["units_with_times"] == 2 and w["shortest_days"] == pytest.approx(4.0)
    assert w["pair_f1_without_under_min"]["units"] == 2          # u2, u3: reported without u1


def test_validate_refuses_a_malformed_time():
    a = assignments({"c:sig:000001": dict(GOOD, completed_at="yesterday")})
    assert any("completed_at" in p for p in cr.validate(a, [corner()], SNAP))
    a = assignments({"c:sig:000001": dict(GOOD, completed_at="2026-11-01T10:00:00.000Z",
                                          opened_at="2026-11-01T09:00:00Z")})
    assert cr.validate(a, [corner()], SNAP) == []


def test_agreement_kind():
    assert cr.infer_kind({"rater": None}, {"rater": "jon-retest"}) == "intra"
    assert cr.infer_kind({"rater": "jon"}, {"rater": "jon-retest"}) == "intra"
    assert cr.infer_kind({"rater": "ann"}, {"rater": "jon-retest"}) is None
    assert cr.infer_kind({"rater": None}, {"rater": "ann"}) is None
    a, b = assignments({"u1": GOOD}), assignments({"u1": GOOD}, rater="ann")
    assert cr.agreement(a, b)["agreement_kind"] is None
    assert cr.agreement(a, b, kind="inter")["agreement_kind"] == "inter"
    with pytest.raises(ValueError, match="kind"):
        cr.agreement(a, b, kind="both")


def test_uncovered_on_a_ramp_is_refused_only_in_a_complete_unit():
    # S9, as the gallery's Export: an incomplete unit may hold a mark under a moved mean
    near = {"lat": LAT, "lng": LNG + 0.0002 + 0.3 * M, "unsure": False}
    u = unit({"1": "r1", "2": "r1", "3": "r2", "4": "not_ramp"}, uncovered=[near],
             complete=False)
    assert cr.validate(assignments({"c:sig:000001": u}), [corner()], SNAP) == []
    u["complete"] = True
    problems = cr.validate(assignments({"c:sig:000001": u}), [corner()], SNAP)
    assert any("sits on ramp" in p for p in problems)


def test_verdict_consistency_reads_the_label_in_its_own_unit():
    # N4: label key 1 is in two units, on different panos; the reviewed unit's own label
    # must be the one looked up, not whichever corner came last
    c1, c2 = corner("c:sig:000001"), corner("c:sig:000002")
    for lab in c2["labels"]:
        lab["pano_id"] = "Q" + lab["key"]
    a = assignments({"c:sig:000001": GOOD})
    records = {"P1": {"pano": {"width": 1000, "height": 100},
                      "detections": [{"x_normalized": 0.1, "y_normalized": 0.5}]}}
    out = cr.verdict_consistency(a, [c1, c2], {"P1": {"dets": [True]}}, records)
    assert out["checked"] == 1 and out["disagree"] == []


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rows = [(1, 1, 0), (3, 3, 3), (2, 1, 1), (0, 0, 0)]
    assert cr.bootstrap_pair_f1(rows) == cr.bootstrap_pair_f1(rows)
    p, m, k = cr.pair_f1(rows)
    (plo, phi), (mlo, mhi) = cr.bootstrap_pair_f1(rows)
    assert plo <= p <= phi and mlo <= m <= mhi and k == 3


def test_agreement_script_names_the_kind(tmp_path, capsys):
    # S4: the output the numbers are copied from says intra or inter, in the headline and JSON
    import importlib.util
    spec = importlib.util.spec_from_file_location("cra", "scripts/cluster_review_agreement.py")
    cra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cra)
    (tmp_path / "snapshot.json").write_text(json.dumps(SNAP), encoding="utf-8")
    (tmp_path / "corners.jsonl").write_text(json.dumps(corner()) + "\n", encoding="utf-8")
    a = assignments({"c:sig:000001": GOOD})
    (tmp_path / "assignments.json").write_text(json.dumps(a), encoding="utf-8")
    (tmp_path / "assignments__ann.json").write_text(
        json.dumps(assignments({"c:sig:000001": GOOD}, rater="ann")), encoding="utf-8")
    (tmp_path / "assignments__jon-retest.json").write_text(
        json.dumps(assignments({"c:sig:000001": GOOD}, rater="jon-retest")), encoding="utf-8")
    with pytest.raises(SystemExit, match="--kind"):
        cra.main([str(tmp_path), "--b", "assignments__ann.json"])
    out = tmp_path / "out.json"
    assert cra.main([str(tmp_path), "--b", "assignments__jon-retest.json", "--json", str(out)]) == 0
    assert "INTRA-rater agreement" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["agreement"]["agreement_kind"] == "intra"
    assert cra.main([str(tmp_path), "--b", "assignments__ann.json", "--kind", "inter"]) == 0
    assert "INTER-rater agreement" in capsys.readouterr().out
