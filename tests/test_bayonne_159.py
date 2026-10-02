"""Bayonne staged ahead of its review (#159): the bundle, the GT-free reads, and the
unreviewed paths through the shared tools.

CPU only, committed inputs only. The byte check against the panos runs only where
``benchmark/bayonne/panos/`` exists (a developer checkout); on CI the record-level
checks still run.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import bayonne_159 as B  # noqa: E402

BUNDLE = os.path.join(REPO, "benchmark", "bayonne")


# --------------------------------------------------------------------------- #
# the staged bundle
# --------------------------------------------------------------------------- #
def test_the_staged_bundle_verifies():
    problems, notes, facts = B.verify_bundle(BUNDLE)
    assert problems == [], problems
    assert facts == {"panos": 125, "strata": {"top": 5, "random": 95, "empty": 25},
                     "detections_055": 147}


def test_the_bundle_is_not_reviewed():
    # The whole point of this stage: nothing may claim a review that did not happen.
    assert not os.path.exists(os.path.join(BUNDLE, "verdicts.json"))
    assert not os.path.exists(os.path.join(BUNDLE, "gt_source.json"))


def test_provenance_records_the_run_hash_the_labeler_doc_records():
    with open(os.path.join(BUNDLE, "bundle_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["run"]["results_jsonl_sha256"] == (
        "4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3")
    assert prov["run"]["results_record_count"] == 28524
    assert "NOT reviewed" in prov["status"]


def test_every_record_is_panoramax_gopro_max_with_a_credit():
    for r in B.load_records("bayonne"):
        p = r["pano"]
        assert p["source"] == "panoramax"
        assert p["camera_make"] == "GoPro" and p["camera_model"] == "Max"
        assert p["copyright"] and p["license"]


def test_bayonne_lands_in_the_modern_action_cam_tier_without_a_new_branch():
    # The plan proposed a Panoramax branch in tier_of(). Tiers are by RIG, not by
    # source, and the records carry GoPro / Max, so the existing branch already
    # classifies every pano; a source branch would split one rig across two tiers.
    from low_floor_sweep import tier_of
    tiers = {tier_of(r["pano"]["camera_make"], r["pano"]["camera_model"],
                     r["pano"]["source"]) for r in B.load_records("bayonne")}
    assert tiers == {"action-modern"}


# --------------------------------------------------------------------------- #
# the nadir band
# --------------------------------------------------------------------------- #
def test_band_top_walks_up_from_inside_the_band():
    white = [0.0] * 512
    for r in range(405, 512):
        white[r] = 1.0
    assert B.band_top_from_rows(white) == 405 / 512


def test_no_white_band_is_none():
    assert B.band_top_from_rows([0.2] * 512) is None


def test_committed_band_file_summary():
    with open(B.NADIR_BAND, encoding="utf-8") as f:
        band = json.load(f)
    assert band["n_with_band"] == 123 and band["n_without_band"] == 2
    assert band["band_top_y_median"] == 0.791
    assert 0.77 <= band["band_top_y_min"] <= band["band_top_y_max"] <= 0.80
    no_band = {p["copyright"] for p in band["panos"].values() if p["band_top_y"] is None}
    assert no_band == {"Arretche"}


def test_band_file_reproduces_from_the_panos():
    if not os.path.isdir(os.path.join(BUNDLE, "panos")):
        pytest.skip("benchmark/bayonne/panos/ is git-ignored and absent here")
    assert B.cmd_band(type("A", (), {"bundle": BUNDLE, "write": False})()) == 0


# --------------------------------------------------------------------------- #
# GT-free reads
# --------------------------------------------------------------------------- #
def test_border_ring_matches_skimage_exclude_border_at_min_distance_10():
    assert B.in_border_ring(9 / 1024, 0.5)
    assert not B.in_border_ring(10 / 1024, 0.5)
    assert B.in_border_ring(1014 / 1024, 0.5)
    assert not B.in_border_ring(1013 / 1024, 0.5)
    assert B.in_border_ring(0.5, 502 / 512)
    assert not B.in_border_ring(0.5, 501 / 512)


def test_firing_rows_counts_by_stratum_and_drops_the_ring_only_when_asked():
    peaks = {"a": [(0.5, 0.5, 0.6), (0.001, 0.5, 0.9)], "b": [(0.5, 0.5, 0.2)], "c": []}
    strata = {"a": "random", "b": "empty", "c": "empty"}
    raw = B.firing_rows(peaks, strata, thresholds=(0.1, 0.55))
    inner = B.firing_rows(peaks, strata, thresholds=(0.1, 0.55), interior_only=True)
    assert raw["all"]["rows"][0] == {"threshold": 0.1, "peaks_per_pano": 1.0,
                                     "share_panos_firing": 0.6667}
    assert raw["random"]["rows"][1]["peaks_per_pano"] == 2.0
    assert inner["random"]["rows"][1]["peaks_per_pano"] == 1.0
    assert raw["empty"]["panos"] == 2 and raw["empty"]["rows"][0]["peaks_per_pano"] == 0.5


def _committed(name):
    path = os.path.join(B.OUT_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not written yet")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_firing_json_rederives_from_committed_caches():
    want = _committed("firing.json")
    assert B._dumps(B.build_firing()) == want


def test_frame_json_rederives_from_committed_caches():
    want = _committed("frame.json")
    assert B._dumps(B.build_frame()) == want


def test_the_bayonne_cache_is_marked_unreviewed_and_covers_the_bundle():
    if not os.path.exists(B.BAYONNE_CACHE):
        pytest.skip("cache not extracted yet")
    peaks, meta = B.load_peaks(B.BAYONNE_CACHE)
    assert meta["gt"] == "unreviewed" and meta["score_floor"] == 0.05
    assert set(peaks) == {r["pano"]["panorama_id"] for r in B.load_records("bayonne")}


# --------------------------------------------------------------------------- #
# the unreviewed paths through shared tools
# --------------------------------------------------------------------------- #
def _tiny_bundle(tmp_path, name="stageville", verdicts=False):
    d = tmp_path / "benchmark" / name
    (d / "panos").mkdir(parents=True)
    (d / "records.jsonl").write_text(
        json.dumps({"pano": {"panorama_id": "P1"}, "detections": []}) + "\n"
        + json.dumps({"pano": {"panorama_id": "P2"}, "detections": []}) + "\n",
        encoding="utf-8")
    if verdicts:
        (d / "verdicts.json").write_text(json.dumps({"panos": {}}), encoding="utf-8")
    return d


def test_compare_load_bundle_refuses_an_unreviewed_bundle_without_the_flag(tmp_path):
    import compare as C
    d = _tiny_bundle(tmp_path)
    with pytest.raises(SystemExit):
        C.load_bundle(str(d))
    records, verdicts, _ = C.load_bundle(str(d), unreviewed=True)
    assert set(records) == {"P1", "P2"} and verdicts == {}


def test_compare_load_bundle_unreviewed_flag_does_not_hide_a_review(tmp_path):
    import compare as C
    d = _tiny_bundle(tmp_path, verdicts=True)
    _, verdicts, _ = C.load_bundle(str(d), unreviewed=True)
    assert verdicts == {}   # the file's (empty) review is returned, not skipped
    d2 = tmp_path / "benchmark" / "reviewed"
    (d2 / "panos").mkdir(parents=True)
    (d2 / "records.jsonl").write_text(
        json.dumps({"pano": {"panorama_id": "P1"}, "detections": []}) + "\n", encoding="utf-8")
    (d2 / "verdicts.json").write_text(json.dumps({"panos": {"P1": {
        "group": "random", "dets": [], "missed": [], "no_missed": True}}}), encoding="utf-8")
    _, verdicts, _ = C.load_bundle(str(d2), unreviewed=True)
    assert "P1" in verdicts


def test_extract_placeholder_gt_and_attach_gt(tmp_path):
    import operating_point_curve as O
    from rampnet.detection_eval import GroundTruth
    _tiny_bundle(tmp_path)
    gts, _ = O.unreviewed_ground_truths("stageville", repo=str(tmp_path))
    assert set(gts) == {"P1", "P2"}
    assert all(g == GroundTruth([], [], False) for g in gts.values())
    panos = [{"pano": p, "preds": [(0.5, 0.5, 0.9)], "gt": g} for p, g in gts.items()]
    real = {"P1": GroundTruth([(0.5, 0.5)], [], True), "P2": GroundTruth([], [], True)}
    out, meta = O.attach_ground_truth(panos, {"gt": "unreviewed", "score_floor": 0.05}, real)
    assert "gt" not in meta and meta["score_floor"] == 0.05
    assert out[0]["preds"] == [(0.5, 0.5, 0.9)] and out[0]["gt"] == real[out[0]["pano"]]
    with pytest.raises(ValueError):
        O.attach_ground_truth(panos, {}, {"P1": real["P1"]})


def test_extract_unreviewed_refuses_a_reviewed_bundle(tmp_path):
    import operating_point_curve as O
    _tiny_bundle(tmp_path, verdicts=True)
    with pytest.raises(SystemExit):
        O.unreviewed_ground_truths("stageville", repo=str(tmp_path))


def test_a_staged_bundle_is_not_a_cascade_split():
    import cascade_cost_35 as CC
    splits = CC.all_benchmark_splits()
    assert "bayonne" not in splits and "richmond" in splits


# --------------------------------------------------------------------------- #
# the review instrument
# --------------------------------------------------------------------------- #
def test_gallery_reads_the_band_and_links_panoramax():
    import gt_gallery as G
    bands = G.load_nadir_band(BUNDLE)
    assert len(bands) == 123
    rec = B.load_records("bayonne")[0]
    pid = rec["pano"]["panorama_id"]
    e = G.entry_meta(rec, "random", bands.get(pid))
    assert e["url"] == f"https://panoramax.ign.fr/#focus=pic&pic={pid}"
    assert e["band"] == bands[pid]
    assert e["credit"] == "sig_bayonne / etalab-2.0"


def test_gallery_without_a_band_file_renders_as_before(tmp_path):
    import gt_gallery as G
    assert G.load_nadir_band(tmp_path) == {}
    e = G.entry_meta({"pano": {"panorama_id": "X", "source": "mapillary"},
                      "detections": []}, "random")
    assert "band" not in e and "credit" not in e
    assert e["url"] == "https://www.mapillary.com/app/?pKey=X&focus=photo"


def test_gallery_html_carries_the_band_note_and_the_in_band_confirm():
    import gt_gallery as G
    html = G.build_html([], {}, "k", "n", "src")
    assert 'id="bandnote"' in html
    assert "inside the nadir logo band" in html


# --------------------------------------------------------------------------- #
# the AI pre-read: a model's labels, kept apart from ground truth
# --------------------------------------------------------------------------- #
def test_the_preread_is_labelled_as_a_model_and_carries_its_rubric():
    with open(B.PREREAD_LABELS_FILE, encoding="utf-8") as f:
        pre = json.load(f)
    assert pre["rater"]["kind"] == "model" and pre["rater"]["human"] is False
    assert "NOT ground truth" in pre["what"] and "rubric" in pre["rubric"].lower()
    assert {v["label"] for v in pre["labels"].values()} <= set(B.PREREAD_LABELS)


def test_the_preread_covers_exactly_the_detections_the_review_will_judge():
    with open(os.path.join(B.PREREAD_DIR, "items.json"), encoding="utf-8") as f:
        items = json.load(f)["items"]
    assert [i["item"] for i in items] == [i["item"] for i in B.preread_items(B.load_records("bayonne"))]
    assert len(items) == 147


def test_preread_summary_rederives():
    want = _committed("ai_preread/summary.json")
    assert B._dumps(B.build_preread_summary()) == want


# --------------------------------------------------------------------------- #
# the replication control, the parity detail, the candidate list
# --------------------------------------------------------------------------- #
def test_checks_json_rederives_and_says_what_it_found():
    want = _committed("checks.json")
    assert B._dumps(B.build_checks()) == want
    res = json.loads(want)
    rc = res["replication_control"]
    assert rc["peaks_only_new"] == 0 and rc["peaks_only_committed"] == 0
    bp = res["bayonne_parity"]
    assert bp["records"] == 147
    assert all(e["in_border_ring"] for p in bp["panos_differing"] for e in p["extra"])


def test_repro_check_counts_cells():
    a = {"p": [(0.5, 0.5, 0.9), (0.001, 0.5, 0.6)]}
    b = {"p": [(0.5, 0.5, 0.90001)]}
    r = B.repro_check(a, b)
    assert r["peaks_same_cell"] == 1 and r["peaks_only_new"] == 1
    assert r["peaks_only_new_in_border_ring"] == 1 and r["peaks_only_committed"] == 0


def test_candidate_needs_two_legs_and_a_silent_rampnet():
    from rampnet.detection_eval import radius_sq_for
    rsq = radius_sq_for()
    legs = {"a": {"p": [(0.30, 0.6), (0.70, 0.6)]},
            "b": {"p": [(0.305, 0.6), (0.70, 0.6)]},
            "c": {"p": [(0.90, 0.6)]}}
    rampnet = {"p": [(0.70, 0.6, 0.8), (0.30, 0.6, 0.2)]}
    out = B.candidate_misses(rampnet, legs, rsq, band={"p": 0.79})
    # (0.30, 0.6): RampNet's peak there is 0.2 < 0.30 -> silent, two legs agree.
    # (0.70, 0.6): RampNet fires -> not a candidate. (0.90, 0.6): one leg only.
    assert [(c["x"], c["legs"], c["rampnet_best_peak_within_radius"]) for c in out] == [
        (0.3, ["a", "b"], 0.2)]
    assert out[0]["in_nadir_band"] is False


def test_candidates_json_rederives():
    want = _committed("candidates.json")
    assert B._dumps(B.build_candidates()) == want


def test_paid_estimate_rederives_from_the_ledger_before_its_cutoff():
    want = _committed("paid_legs_estimate.json")
    assert B._dumps(B.build_paid_estimate()) == want


def test_paid_estimate_ignores_recovered_rows_and_rows_after_the_cutoff():
    rows = [{"provider": "gemini", "label": "g", "est_cost_usd": 1.0, "panos_scored": 10,
             "ts": "2026-09-01T00:00:00Z"},
            {"provider": "gemini", "label": "g", "est_cost_usd": 9.0, "panos_scored": 10,
             "ts": "2026-10-05T00:00:00Z"},
            {"provider": "gemini", "label": "g", "est_cost_usd": 50.0, "kind": "recovered",
             "ts": "2026-09-01T00:00:00Z"}]
    [leg] = B.paid_estimate(rows, 125, legs=(("gemini", "g", None),))
    assert leg["usd_per_pano"] == 0.1 and leg["expected_usd"] == 12.5
