"""Tests for ``rampnet.eval``, the benchmark scoring protocol as code (#150).

CPU-only, offline, committed inputs only. The full reproduction (every scoreboard cell and
every ``benchmark_eval/`` number) is ``scripts/analysis/eval_protocol_150.py --check``;
here a few representative cells are re-derived so the suite stays fast.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison", "yolo_baseline"))

import compare as C  # noqa: E402
import eval_protocol_150 as EP  # noqa: E402
import low_floor_sweep as LFS  # noqa: E402
import rescore_benchmark_eval as RBE  # noqa: E402
from rampnet import bundles  # noqa: E402
from rampnet import detection_eval as DE  # noqa: E402
from rampnet import eval as E  # noqa: E402

BENCH = os.path.join(REPO, "benchmark")


# --------------------------------------------------------------------------- #
# (a) a synthetic bundle with known answers, scored through the CLI
# --------------------------------------------------------------------------- #
def _det(x, y, c):
    return {"x_normalized": x, "y_normalized": y, "confidence": c}


@pytest.fixture
def toy_bundle(tmp_path):
    """Three panos covering every verdict kind.

    p1: A True, B False, C unsure, D duplicate; one missed mark M  -> GT {A, M}, ignore {C},
        recall-confirmed (it has a missed mark).
    p2: E True, attested no_missed                                  -> GT {E}, confirmed.
    p3: F True, not attested and no missed marks                    -> GT {F}, NOT confirmed.
    """
    records = [
        {"pano": {"panorama_id": "p1"}, "detections": [
            _det(0.10, 0.50, 0.9), _det(0.30, 0.50, 0.8), _det(0.50, 0.50, 0.7),
            _det(0.105, 0.50, 0.6)]},
        {"pano": {"panorama_id": "p2"}, "detections": [_det(0.20, 0.40, 0.9)]},
        {"pano": {"panorama_id": "p3"}, "detections": [_det(0.60, 0.60, 0.9)]},
    ]
    verdicts = {"panos": {
        "p1": {"dets": [True, False, "unsure", "duplicate"],
               "missed": [{"x": 0.70, "y": 0.50}], "no_missed": False},
        "p2": {"dets": [True], "missed": [], "no_missed": True},
        "p3": {"dets": [True], "missed": [], "no_missed": False},
    }}
    bdir = tmp_path / "toy"
    bdir.mkdir()
    (bdir / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (bdir / "verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")
    preds = {"model": "toy", "city": "toy", "detections": {
        # TP on A, TP on M, lands on the unsure C (ignored), and a clear FP.
        "p1": [[0.10, 0.50, 0.95], [0.70, 0.50, 0.5], [0.50, 0.50, 0.4], [0.30, 0.30, 0.3]],
        "p2": [],                                  # E is missed
        # TP on F (outside the recall pool) and an FP.
        "p3": [[0.60, 0.60, 0.9], [0.90, 0.10, 0.8]],
    }}
    pfile = tmp_path / "toy_preds.json"
    pfile.write_text(json.dumps(preds), encoding="utf-8")
    return str(bdir), str(pfile)


def test_synthetic_bundle_scores_known_counts(toy_bundle, tmp_path, capsys):
    bdir, pfile = toy_bundle
    out = tmp_path / "res.json"
    assert E.main(["score", "--bundle", bdir, "--predictions", pfile,
                   "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert (res["tp"], res["fp"], res["fn"], res["ignored"]) == (3, 2, 1, 1)
    assert res["precision"] == pytest.approx(3 / 5)
    assert res["recall"] == pytest.approx(2 / 3)       # A, M found of A, M, E
    assert res["n_panos"] == 3 and res["n_recall_panos"] == 2 and res["n_gt_recall"] == 3
    assert res["ap"] is not None
    assert res["pins"]["n_records"] == 3 and res["pins"]["gt_kind"] == "verdicts"
    assert res["scorer_fingerprint"] == E.scorer_fingerprint()
    assert "toy / toy @ op 0.0" in capsys.readouterr().out


def test_op_threshold_truncates_counts_but_not_ap(toy_bundle):
    bdir, pfile = toy_bundle
    preds = E.load_predictions(pfile)
    full = E.score_split(bdir, preds)
    op = E.score_split(bdir, preds, op_threshold=0.45)
    # 0.4 (ignored) and 0.3 (FP) on p1 drop out; p3's two survive.
    assert (op["tp"], op["fp"], op["fn"], op["ignored"]) == (3, 1, 1, 0)
    assert op["ap"] == full["ap"]


def test_floor_drops_and_reports(toy_bundle):
    bdir, pfile = toy_bundle
    res = E.score_split(bdir, E.load_predictions(pfile), floor=0.45)
    assert res["n_below_floor_dropped"] == 2
    assert "lowest confidence present, 0.5000 (declared floor 0.45)" in res["ap_note"]
    assert res["warnings"] == []


def test_floor_far_below_the_predictions_warns(toy_bundle):
    """Declaring 0.05 for detections that start at 0.6 must not read as 'truncated at
    0.05' (review of #245): the note gives the real cut and a warning says so."""
    bdir, _ = toy_bundle
    res = E.score_split(bdir, "rampnet", floor=0.05)
    assert "lowest confidence present, 0.6000" in res["ap_note"]
    assert any("more than 0.1 above the declared floor" in w for w in res["warnings"])


def test_op_threshold_below_floor_warns(toy_bundle):
    bdir, pfile = toy_bundle
    res = E.score_split(bdir, E.load_predictions(pfile), floor=0.5, op_threshold=0.25)
    assert any("below the declared floor" in w for w in res["warnings"])


def test_city_mismatch_warns_and_no_overlap_is_an_error(toy_bundle, capsys):
    bdir, pfile = toy_bundle
    preds = E.load_predictions(pfile)
    preds["city"] = "elsewhere"
    res = E.score_split(bdir, preds)
    assert any("city 'elsewhere'" in w for w in res["warnings"])
    with pytest.raises(E.PredictionFormatError, match="wrong split"):
        E.score_split(bdir, {"city": "bend", "detections": {"zz": [[0.1, 0.1, 0.9]]}})
    wrong = os.path.join(BENCH, "model_detections", "y11l_pano__bend.json")
    assert E.main(["score", "--bundle", os.path.join(BENCH, "richmond"),
                   "--predictions", wrong]) == 2
    assert "wrong split" in capsys.readouterr().err


def test_rampnet_predictions_score_own_detections(toy_bundle):
    bdir, _ = toy_bundle
    res = E.score_split(bdir, "rampnet")
    # A, E, F are GT-true detections -> TP; B false -> FP; C unsure -> ignored; D is a
    # duplicate of A -> FP under 1:1 matching. M is a miss.
    assert (res["tp"], res["fp"], res["fn"], res["ignored"]) == (3, 2, 1, 1)


# --------------------------------------------------------------------------- #
# (b) the loaders moved, not copied; every scored bundle loads
# --------------------------------------------------------------------------- #
def test_compare_reexports_the_moved_functions():
    for name in ("load_bundle", "ground_truths_from_verdicts", "load_manual_ground_truths",
                 "validate_bundle", "validate_manual_bundle", "verdicts_from_spec"):
        assert getattr(C, name) is getattr(bundles, name), name
    for name in ("rescore", "operating_report", "sweep_rows", "has_confidences",
                 "SWEEP_THRESHOLDS"):
        assert getattr(C, name) is getattr(E, name), name


@pytest.mark.parametrize("split", E.PINNED_SPLITS)
def test_every_pinned_bundle_loads(split):
    records, gts, kind = bundles.ground_truths(os.path.join(BENCH, split))
    assert gts and set(gts) <= set(records)
    assert kind == ("manual_labels" if split == "manual_gold" else "verdicts")


def test_registries_agree_with_low_floor_sweep():
    assert E.PINNED_SPLITS == LFS.ALL_SPLITS
    assert E.POOLED_SPLITS == LFS.US_SPLITS


# --------------------------------------------------------------------------- #
# (c) split pins
# --------------------------------------------------------------------------- #
def test_split_pins_match_committed_file():
    assert E.verify_pins() == [], "benchmark/split_pins.json drifted; see `pins --verify`"


def test_pins_file_is_byte_identical_to_a_regeneration():
    with open(E.PINS_PATH, "rb") as fh:
        have = fh.read().replace(b"\r\n", b"\n").decode("utf-8")
    assert have == E.pins_payload()


def test_verify_pins_reports_drift(tmp_path):
    with open(E.PINS_PATH, encoding="utf-8") as fh:
        body = json.load(fh)
    body["splits"]["richmond"]["records_sha256"] = "0" * 64
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    drift = E.verify_pins(str(path))
    assert len(drift) == 1 and drift[0].startswith("richmond.records_sha256")


# --------------------------------------------------------------------------- #
# (d) representative cells of the reproduction check
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scoreboard_cells():
    with open(EP.SCOREBOARD_JSON, encoding="utf-8") as fh:
        return json.load(fh)["per_split"]


def _round(v):
    return EP._r(v)


def test_rampnet_city_cell_reproduces(scoreboard_cells):
    stored = scoreboard_cells["rampnet"]["richmond"]
    res = E.score_split(os.path.join(BENCH, "richmond"), "rampnet", op_threshold=0.55)
    for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_panos", "n_gt_recall"):
        assert _round(res[k]) == stored[k], k
    assert _round(res["ap"]) == stored["ap_bundle"]
    assert stored["ap_source"].startswith("op_cache")
    low = E.score_split(os.path.join(BENCH, "richmond"), EP.op_cache_predictions("richmond"))
    assert _round(low["ap"]) == stored["ap"]


def test_challenger_cell_reproduces(scoreboard_cells):
    stored = scoreboard_cells["gemini-3.6-flash"]["richmond"]
    preds = E.load_predictions(
        os.path.join(BENCH, "model_detections", "gemini-3.6-flash__richmond.json"))
    res = E.score_split(os.path.join(BENCH, "richmond"), preds, op_threshold=0.0)
    for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_panos", "n_gt_recall"):
        assert _round(res[k]) == stored[k], k
    assert res["ap"] is None and stored["ap"] is None


def test_yolo_arm_reproduces_benchmark_eval():
    heads, sweeps = EP.parse_benchmark_txt(os.path.join(RBE.BENCHMARK_EVAL, "richmond.txt"))
    assert set(heads) == set(RBE.ARMS)
    preds = E.load_predictions(
        os.path.join(BENCH, "model_detections", "y11l_pano__richmond.json"))
    res = E.score_split(os.path.join(BENCH, "richmond"), preds, op_threshold=0.25,
                        floor=0.05, sweep=True)
    h = heads["y11l_pano"]
    assert (h["p"], h["r"], h["f1"], h["ap"]) == tuple(
        f"{res[k]:.3f}" for k in ("precision", "recall", "f1", "ap"))
    assert (int(h["tp"]), int(h["fp"]), int(h["fn"]), int(h["ign"])) == (
        res["tp"], res["fp"], res["fn"], res["ignored"])
    assert len(sweeps["y11l_pano"]) == len(res["sweep"])
    with open(os.path.join(RBE.BENCHMARK_EVAL, "pr_richmond", "pr_y11l_pano.json"),
              encoding="utf-8") as fh:
        assert json.load(fh)["ap"] == res["ap"]


def test_op_cache_predictions_give_the_published_rampnet_ap(scoreboard_cells):
    stored = scoreboard_cells["rampnet"]["richmond"]
    res = E.score_split(os.path.join(BENCH, "richmond"), "rampnet-op-cache",
                        op_threshold=0.55, floor=0.05)
    assert _round(res["ap"]) == stored["ap"]
    assert (res["tp"], res["fp"], res["fn"]) == (stored["tp"], stored["fp"], stored["fn"])
    assert res["warnings"] == []


def test_op_cache_is_refused_where_the_bundle_already_reaches_the_floor():
    """manual_gold's bundle is a 0.05 flip-TTA export; its op_cache is a never-published
    no-TTA run, so substituting it would swap the model config (re-review of #245)."""
    with pytest.raises(E.PredictionFormatError, match="use --predictions rampnet"):
        E.score_split(os.path.join(BENCH, "manual_gold"), "rampnet-op-cache")


def test_bundle_dot_names_the_split(monkeypatch):
    monkeypatch.chdir(os.path.join(BENCH, "richmond"))
    res = E.score_split(".", "rampnet", op_threshold=0.55)
    assert res["split"] == "richmond"
    assert E.op_cache_predictions(".")["city"] == "richmond"


def test_borrowed_verdict_bundle_scores_and_pins():
    """A #48 bundle.json bundle used to crash in split_pins (review of #245)."""
    bdir = os.path.join(BENCH, "richmond_neighbourhood")
    if not os.path.exists(os.path.join(bdir, "bundle.json")):
        pytest.skip("no bundle.json bundle in this checkout")
    res = E.score_split(bdir, "rampnet")
    pins = res["pins"]
    assert pins["gt_kind"] == "borrowed_verdicts"
    assert pins["verdicts_sha256"] == bundles.split_pins(
        os.path.join(BENCH, "richmond"))["verdicts_sha256"]


def test_cli_root_prefers_a_checkout_cwd(tmp_path):
    assert E.cli_root(REPO) == os.path.abspath(REPO)
    assert E.cli_root(os.path.join(REPO, "docs")) == os.path.abspath(REPO)
    assert E.cli_root(str(tmp_path)) == E.REPO_ROOT


def test_reproduction_rederives_in_ci():
    """The whole proof, re-derived (about 20 s): every scoreboard and benchmark_eval cell
    equal, none unchecked, and the committed reproduction.json byte-identical to a
    regeneration. Without this, a scorer, bundle or eval.py change could leave
    `eval_protocol_150.py --check` failing while CI stayed green (review of #245)."""
    result = EP.build()
    for src, summ in result["summary"].items():
        assert summ["differs"] == 0 and summ["unchecked"] == 0, (src, summ)
    with open(EP.OUT_JSON, "rb") as fh:
        have = fh.read().replace(b"\r\n", b"\n").decode("utf-8")
    assert have == EP.payload(result), "reproduction.json is stale: re-run the script"


def test_committed_reproduction_reports_all_equal():
    with open(EP.OUT_JSON, encoding="utf-8") as fh:
        rep = json.load(fh)
    for src, s in rep["summary"].items():
        assert s["differs"] == 0 and s["unchecked"] == 0, src
        assert s["equal"] == s["cells"] > 0, src
    assert rep["scorer_fingerprint"] == E.scorer_fingerprint()


# --------------------------------------------------------------------------- #
# (e) the fingerprint guard
# --------------------------------------------------------------------------- #
def test_scorer_fingerprint_unchanged():
    assert E.scorer_fingerprint() == RBE.scorer_fingerprint()
    assert E.scorer_fingerprint() == "f4aef67aba03", (
        "One of rampnet/{geometry,metrics,detection_eval,validation}.py changed. Every "
        "committed benchmark number names this fingerprint: re-run "
        "rescore_benchmark_eval.py, scoreboard.py and eval_protocol_150.py, commit their "
        "outputs, and update this constant in the same PR (#148, #150).")


# --------------------------------------------------------------------------- #
# (f) the prediction-file validator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad, needle", [
    ({"detections": {"pX": [[0.1, 0.2, 0.3], [0.5, "a", 0.1]]}}, "pano pX, point 1"),
    ({"detections": {"pX": [[1.5, 0.2, 0.3]]}}, "pano pX, point 0"),
    ({"detections": {"pX": [[0.1, 0.2, "high"]]}}, "pano pX, point 0"),
    ({"detections": {"pX": [[0.1]]}}, "pano pX, point 0"),
    ({"detections": {"pX": {"x": 0.1}}}, "pano pX"),
    ({"detections": []}, "'detections' must be an object"),
    ({"detections": {"pX": [[0.1, 0.2, -0.1]]}}, "pano pX, point 0: confidence -0.1"),
    ({"model": 3, "detections": {}}, "'model' must be a string"),
])
def test_validator_names_the_pano_and_index(bad, needle):
    with pytest.raises(E.PredictionFormatError, match=needle):
        E.validate_predictions(bad)


def test_cli_rejects_malformed_file(tmp_path, toy_bundle, capsys):
    bdir, _ = toy_bundle
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"detections": {"p1": [[0.1, None, 0.5]]}}), encoding="utf-8")
    assert E.main(["score", "--bundle", bdir, "--predictions", str(bad)]) == 2
    assert "pano p1, point 0" in capsys.readouterr().err


def test_null_confidence_is_valid_and_gets_no_ap(toy_bundle):
    bdir, _ = toy_bundle
    res = E.score_split(bdir, {"detections": {"p1": [[0.10, 0.50, None]]}})
    assert res["tp"] == 1 and res["ap"] is None
    assert res["n_panos_without_predictions"] == 2


# --------------------------------------------------------------------------- #
# (g) the protocol constants
# --------------------------------------------------------------------------- #
def test_protocol_constants_match_detection_eval(capsys):
    c = E.protocol_constants()
    assert (c["radius"], c["scale_x"], c["scale_y"], c["wrap_x"]) == (
        DE.PANO_RADIUS_NORMALIZED, DE.PANO_SCALE_X, DE.PANO_SCALE_Y, True)
    assert E.main(["protocol"]) == 0
    out = capsys.readouterr().out
    assert str(DE.PANO_RADIUS_NORMALIZED) in out and E.scorer_fingerprint() in out
    assert tuple(f"rampnet/{s}" for s in E.SCORER_SOURCES) == RBE.SCORER_SOURCES


def test_loco_reports_each_pooled_split():
    res = E.loco("y11l_pano", op_threshold=0.25, splits=("richmond", "bend", "clovis"))
    assert [r["held_out"] for r in res["rows"]] == ["richmond", "bend", "clovis"]
    row = res["rows"][0]
    assert row["rest"] == ["bend", "clovis"]
    tp = sum(res["rows"][i]["held_out_score"]["tp"] for i in (1, 2))
    assert row["rest_micro"]["tp"] == tp
    # Micro recall is gated like aggregate(): recall-confirmed GT only.
    m = row["rest_micro"]
    assert m["recall"] == pytest.approx((m["n_gt_recall"] - m["fn"]) / m["n_gt_recall"])
