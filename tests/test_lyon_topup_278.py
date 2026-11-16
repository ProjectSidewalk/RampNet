"""The Lyon audit top-up for other_cameras (#278): the draw, the never-train lists, the
pre-registered plan, the pooled rule and the stop rule.

CPU only, committed inputs only. The stop rule is exercised on synthetic look exports; the
real review has not happened. The byte checks against the panos run only where
``benchmark/lyon_topup278/panos/`` exists (a developer checkout).
"""
import json
import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import lyon_review_159 as L  # noqa: E402
import lyon_topup_278 as T  # noqa: E402

BUNDLE = T.BUNDLE
FIRST = T.FIRST


def _recs():
    return L.load_records(BUNDLE)


# --------------------------------------------------------------------------- #
# the draw and the never-train lists
# --------------------------------------------------------------------------- #
def test_the_draw_is_what_the_plan_says():
    recs = _recs()
    assert len(recs) == T.N_MAX + T.RESERVE == 80
    assert T.draw_problems(recs) == []        # random / other_cameras / a 0.55 detection,
    #                                           > 30 m from the 200, pairwise >= 30 m
    problems, _ = T.verify_bundle(require_imagery=False)
    assert problems == []


def test_the_never_train_lists():
    ids = set(L._read_lines(os.path.join(BUNDLE, "ids.txt")))
    first_ids = set(L._read_lines(os.path.join(FIRST, "ids.txt")))
    first_excl = set(L._read_lines(os.path.join(FIRST, "exclude_ids.txt")))
    assert ids == {r["pano"]["panorama_id"] for r in _recs()}
    assert not ids & first_excl
    assert set(L._read_lines(os.path.join(BUNDLE, "never_train_ids.txt"))) == first_ids | ids
    excl = set(L._read_lines(os.path.join(BUNDLE, "exclude_ids.txt")))
    assert first_excl | ids <= excl
    assert len(excl) > len(first_excl)


def test_the_group_mapping_reproduces_the_first_draw_strata():
    assert T.check_mapping(T.load_group_map()) == []


def test_the_top_up_is_not_a_scored_split():
    from rampnet.bundles import is_scored_split
    assert not is_scored_split(BUNDLE)


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_rebuilds_byte_for_byte_and_pins_the_first_draw():
    with open(os.path.join(BUNDLE, T.PLAN_NAME), "rb") as f:
        assert f.read() == L._dumps(T.build_plan()).encode("utf-8")
    plan = json.load(open(os.path.join(BUNDLE, T.PLAN_NAME), encoding="utf-8"))
    assert plan["first_draw_pins"] == T.first_pins()
    assert plan["looks"] == [25, 45, 70] and plan["z"] == 2.35
    assert len(plan["consequences"]) == 4 and plan["deviations"]
    assert plan["review_order"] == T.review_order([r["pano"]["panorama_id"] for r in _recs()])
    oc = json.load(open(T.DESIGN_OC, encoding="utf-8"))
    assert plan["design_oc"] == oc                # read from disk, never recomputed
    have = {(r["z"], r["rho"], r["p"]) for r in oc["rows"]}
    for z in (2.35, 2.2):
        for rho in (0.0, 0.3):
            for p in (0.70, 0.75, 0.78, 0.80):
                assert (z, rho, p) in have


def test_the_pre_registration_hashes_match():
    prov = T.load_provenance()
    pre = prov["pre_registered"]
    assert pre["audit_plan_sha256"] == L._sha256(os.path.join(BUNDLE, T.PLAN_NAME))
    assert pre["records_jsonl_sha256"] == L._sha256(os.path.join(BUNDLE, "records.jsonl"))


# --------------------------------------------------------------------------- #
# the pooled rule
# --------------------------------------------------------------------------- #
def test_first_draw_units_and_the_bootstrap_match_the_posted_scorer():
    fd = T.first_draw()
    assert (fd["panos"], fd["correct"], fd["decided"]) == (30, 40, 46)
    assert T.bootstrap_ci_level(fd["units"], 1.96, reps=10000, seed=159) == \
        L.bootstrap_ci(fd["units"])
    score = json.load(open(os.path.join(REPO, "analysis_out", "lyon_review_159",
                                        "score.json"), encoding="utf-8"))
    lo, hi = T.bootstrap_ci_level(fd["units"], 1.96, reps=10000, seed=159)
    assert [L._r(lo), L._r(hi)] == score["groups"]["other_cameras"]["precision_055"]["bootstrap95"]


def test_unsure_is_dropped_and_duplicates_are_false():
    assert T.pano_unit(["unsure", True, "duplicate", False]) == (1, 3)
    assert T.pano_unit(["unsure"]) == (0, 0)


# --------------------------------------------------------------------------- #
# the stop rule on synthetic looks
# --------------------------------------------------------------------------- #
STAGED = [f"P{i:02d}" for i in range(70)]
FIRST_UNITS = T.first_draw()["units"]


def _looks(pattern, n_dets=1):
    """A look's panos map: ``pattern`` is a list of per-pano verdict lists."""
    return {pid: {"group": "random", "dets": dets, "missed": [], "no_missed": False}
            for pid, dets in zip(STAGED, pattern)}


def _ndets(pattern):
    nd = {pid: 1 for pid in STAGED}
    nd.update({pid: len(d) for pid, d in zip(STAGED, pattern)})
    return nd


def _run(*looks, pattern):
    return T.evaluate_looks(list(looks), STAGED, _ndets(pattern), FIRST_UNITS)


def test_settled_eligible_at_look_1():
    pat = [[True, True, True]] * 25
    res = _run(_looks(pat), pattern=pat)
    assert res["status"] == "final" and res["outcome"] == "eligible" and res["look"] == 1
    assert T.consequence_row(res["outcome"], res["numbers"]["estimate"]) == T.CONSEQUENCES[0]


def test_continue_then_settled_at_look_2():
    pat = [[True]] * 20 + [[False]] * 5 + [[True, True, True]] * 20
    res1 = _run(_looks(pat[:25]), pattern=pat)
    assert res1["status"] == "continue" and res1["next"] == 45
    assert "numbers" not in res1 and res1["looks"] == [
        {"look": 1, "panos_counted": 25, "outcome": "CONTINUE"}]
    res2 = _run(_looks(pat[:25]), _looks(pat), pattern=pat)
    assert res2["status"] == "final" and res2["outcome"] == "eligible" and res2["look"] == 2


def test_settled_not_eligible():
    pat = [[False, False, False]] * 25
    res = _run(_looks(pat), pattern=pat)
    assert res["status"] == "final" and res["outcome"] == "not eligible"
    assert T.consequence_row(res["outcome"], res["numbers"]["estimate"]) == T.CONSEQUENCES[3]


def _single(correct_at_25, correct_at_45, correct_at_70):
    """70 single-detection panos with the given cumulative correct counts."""
    out = []
    for size, good in ((25, correct_at_25), (20, correct_at_45 - correct_at_25),
                       (25, correct_at_70 - correct_at_45)):
        out += [[True]] * good + [[False]] * (size - good)
    return out


def test_final_eligible_not_decisive_at_look_3():
    pat = _single(21, 37, 57)
    looks = [_looks(pat[:25]), _looks(pat[:45]), _looks(pat)]
    res = _run(*looks, pattern=pat)
    assert res["status"] == "final" and res["outcome"] is None and res["look"] == 3
    assert res["numbers"]["estimate"] >= 0.80
    assert T.consequence_row(None, res["numbers"]["estimate"]) == T.CONSEQUENCES[1]


def test_final_not_eligible_at_look_3():
    pat = _single(18, 32, 50)
    res = _run(_looks(pat[:25]), _looks(pat[:45]), _looks(pat), pattern=pat)
    assert res["status"] == "final" and res["outcome"] is None
    assert res["numbers"]["estimate"] < 0.80
    assert T.consequence_row(None, res["numbers"]["estimate"]) == T.CONSEQUENCES[2]


def test_a_look_reads_exactly_its_prefix():
    pat = [[True]] * 20 + [[False]] * 5
    more_good = pat + [[True, True, True]] * 10
    more_bad = pat + [[False, False, False]] * 10
    a = _run(_looks(more_good), pattern=more_good)
    b = _run(_looks(more_bad), pattern=more_bad)
    assert a == b and a["status"] == "continue"


def test_waiting_until_the_look_is_fully_judged():
    pat = [[True]] * 25
    look = _looks(pat)
    look["P10"]["dets"] = [None]
    res = _run(look, pattern=pat)
    assert res == {"status": "waiting", "look": 1, "need": 25, "done": 24, "looks": []}


def test_a_changed_verdict_on_a_counted_pano_is_refused():
    pat = [[True]] * 20 + [[False]] * 5 + [[True]] * 20
    look2 = _looks(pat)
    look2["P03"] = {**look2["P03"], "dets": [False]}
    res = _run(_looks(pat[:25]), look2, pattern=pat)
    assert res["status"] == "refuse" and "P03" in res["problems"][0]


def test_a_look_after_a_settlement_is_refused():
    pat = [[True, True, True]] * 45
    res = _run(_looks(pat[:25]), _looks(pat), pattern=pat)
    assert res["status"] == "refuse" and "look 1 read SETTLED" in res["problems"][0]


def test_an_unstaged_pano_early_in_the_order_does_not_block(tmp_path):
    order = [f"Q{i}" for i in range(6)]
    index = {}
    for pid in order:
        (tmp_path / f"{pid}.jpg").write_bytes(pid.encode())
        index[pid] = {"sha256": L._sha256(tmp_path / f"{pid}.jpg")}
    os.remove(tmp_path / "Q1.jpg")                       # never fetched
    index["Q2"]["sha256"] = "0" * 64                     # wrong bytes
    staged, dropped, unused = T.choose_staged(order, index, str(tmp_path), n_max=3)
    assert [s[0] for s in staged] == ["Q0", "Q3", "Q4"]
    assert [d[0] for d in dropped] == ["Q1", "Q2"] and unused == ["Q5"]
    assert T.staged_order_problems(order, [s[0] for s in staged],
                                   [{"id": d[0]} for d in dropped]) == []
    assert T.staged_order_problems(order, ["Q0", "Q3"], []) != []


def test_score_end_to_end_interim_output_is_blind(tmp_path, monkeypatch, capsys):
    b = tmp_path / "lyon_topup278"
    b.mkdir()
    for n in ("records.jsonl", "sample.json", T.PLAN_NAME):
        shutil.copyfile(os.path.join(BUNDLE, n), b / n)
    plan = json.load(open(b / T.PLAN_NAME, encoding="utf-8"))
    staged = plan["review_order"][:70]
    (b / T.PROVENANCE).write_text(json.dumps({
        "pre_registered": {"audit_plan_sha256": L._sha256(b / T.PLAN_NAME),
                           "records_jsonl_sha256": L._sha256(b / "records.jsonl"),
                           "comment": None},
        "imagery": {"staged_ids": staged}}), encoding="utf-8")
    monkeypatch.setattr(T, "LOOKS_JSON", str(tmp_path / "looks.json"))
    monkeypatch.setattr(T, "SCORE", str(tmp_path / "score.json"))
    ndet = {r["pano"]["panorama_id"]: len(r["detections"]) for r in _recs()}
    (b / T.LOOK_DIR).mkdir()

    def export(k, verdict_of):
        panos = {pid: {"group": "random", "dets": [verdict_of(i, j) for j in range(ndet[pid])],
                       "missed": [], "no_missed": False}
                 for i, pid in enumerate(staged[:T.LOOKS[k - 1]])}
        (b / T.LOOK_DIR / f"look{k}.json").write_text(json.dumps(
            {"run_key": "lyon_topup278", "panos": panos}), encoding="utf-8")

    export(1, lambda i, j: i % 10 != 0)                  # mixed: not settled at look 1
    assert T.main(["--bundle", str(b), "score", "--write"]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith("CONTINUE: rate on to pano 45")
    assert "0." not in out                               # no estimate, no interval
    looks = json.load(open(tmp_path / "looks.json", encoding="utf-8"))
    assert [r["outcome"] for r in looks["looks"]] == ["CONTINUE"]
    for rec in looks["looks"]:                           # no number beyond the look's size
        assert set(rec) == {"look", "panos_counted", "outcome", "file", "sha256"}
    assert not (tmp_path / "score.json").exists()
    # a later export that flips a counted verdict is refused
    export(2, lambda i, j: (i % 10 != 0) if i != 1 else False)
    assert T.main(["--bundle", str(b), "score", "--write"]) == 1
    assert "changes the verdicts" in capsys.readouterr().out
    # the honest look 2 settles, and only now are numbers printed and written
    export(2, lambda i, j: (i % 10 != 0) if i < 25 else True)
    assert T.main(["--bundle", str(b), "score", "--write"]) == 0
    out = capsys.readouterr().out
    assert "SETTLED eligible at look 2" in out and "pooled P@0.55" in out
    assert "plan was not posted" in out
    score = json.load(open(tmp_path / "score.json", encoding="utf-8"))
    assert score["settled"] and score["plan_comment_posted"] is False
    assert (b / "verdicts.json").read_bytes() == (b / T.LOOK_DIR / "look2.json").read_bytes()
    # without --write: a check against what was written
    assert T.main(["--bundle", str(b), "score"]) == 0


def test_score_refuses_a_changed_plan(tmp_path, capsys):
    b = tmp_path / "lyon_topup278"
    b.mkdir()
    for n in ("records.jsonl", "sample.json", T.PLAN_NAME):
        shutil.copyfile(os.path.join(BUNDLE, n), b / n)
    (b / T.PROVENANCE).write_text(json.dumps({"pre_registered": {
        "audit_plan_sha256": "0" * 64, "records_jsonl_sha256": "0" * 64}}), encoding="utf-8")
    assert T.main(["--bundle", str(b), "score"]) == 1
    assert "refusing to score" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the detections-only page
# --------------------------------------------------------------------------- #
def test_default_gallery_is_unchanged_and_the_variant_is_a_post_process():
    from gt_gallery import build_html, apply_detections_only, DETECTIONS_ONLY_EDITS
    page = build_html([], {}, "k", "k", "records.jsonl")
    assert "This is a two-sided check." in page
    assert "function fnChecked(s) { return s.missed.length > 0 || !!s.noMissed; }" in page
    assert "Detections-only" not in page and "detections-only" not in page
    variant = apply_detections_only(page)
    assert "Do not scan for missed ramps in this review." in variant
    assert "function fnChecked(s) { return true; }" in variant
    assert "two-sided check" not in variant
    for _, new in DETECTIONS_ONLY_EDITS:
        assert new in variant
    with pytest.raises(ValueError, match="expected once"):
        apply_detections_only(variant)                 # its snippets are gone now


# --------------------------------------------------------------------------- #
# imagery (developer checkout only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(os.path.join(BUNDLE, "panos")),
                    reason="benchmark/lyon_topup278/panos/ is git-ignored and not staged here")
def test_the_staged_imagery_verifies():
    problems, _ = T.verify_bundle(require_imagery=True)
    assert problems == []
