"""Unit tests for the complementarity gate (#35, generalized past Gemini in #126).

Pure logic plus two drift guards — no GPU, no ``.model_cache``, no network. Every
detection these read comes from ``benchmark/model_detections/``, the published export,
so a clean clone runs them.

What they protect:

* **``matched_gt`` and ``score_pano`` must agree.** This script prints the four
  complementarity cells (from ``matched_gt``) and the false-positive counts and union
  P/R/F1 (from ``score_pano``) in one table. ``score_pano`` wraps the 360° seam by
  default since #132; ``matched_gt`` re-derived the distance inline and did not, which
  put two matchers in one output. It now calls the shared matcher, and the wrap test
  below fails if the inline form comes back.
* **``model_spec`` must reject a mistyped provider.** A bad spec does not raise: it
  addresses a cache entry nothing ever wrote, and the script reports a model with zero
  detections, which reads as a missing run.
* **``compare_args`` must reconstruct the signature ``compare.py`` cached under.**
  Same silent failure. Checked against the signature recorded inside the published
  export rather than against a copy of the defaults.
* **``complementary_null``** is the discount every attributable-gain number in
  ``docs/model_comparison.md`` is quoted after, so it is checked against cases whose
  answer is arithmetic rather than measurement.
* **``partition_cells``** is the one loop behind this script's table, the cascade
  gate's partition and the regression tests below — the tests call it rather than
  re-implementing it, so a bug in the threshold filter or the op_cache fallback fails
  here instead of being copied into a third place.
* the published 384 richmond column, cell for cell, at the shipped point and at the
  two op_cache thresholds the doc quotes.
"""
import argparse
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import complementarity as cx  # noqa: E402
from compare import DetectionCache, cache_key, load_bundle  # noqa: E402
from detectors import PROVIDERS, build_detector  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, build_ground_truth, radius_sq_for, score_pano)

RSQ = radius_sq_for()
R_NORM = RSQ ** 0.5 / PANO_SCALE_X          # match radius in normalized x, ~0.022
PUBLISHED = os.path.join(REPO, "benchmark", "model_detections",
                         "mask2former-vistas-curb-cut__richmond.json")
PARITY = os.path.join(REPO, "benchmark", "model_detections",
                      "mask2former-vistas-curb-cut-1024x1024__richmond.json")


def _args(**kw):
    """The namespace ``compare_args`` reads, with this script's own CLI defaults."""
    ns = argparse.Namespace(tiling="perspective", radius=0.022,
                            cache_dir=os.path.join(REPO, ".model_cache"),
                            vistas_input_size=None, vistas_revision=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# --------------------------------------------------------------------------- #
# matched_gt — the same matcher score_pano uses, wrap included
# --------------------------------------------------------------------------- #
def test_a_prediction_on_the_ramp_claims_it():
    assert cx.matched_gt([(0.5, 0.5, 0.9)], [(0.5, 0.5)], RSQ) == {0}


def test_a_prediction_out_of_radius_claims_nothing():
    assert cx.matched_gt([(0.5 + 2 * R_NORM, 0.5, 0.9)], [(0.5, 0.5)], RSQ) == set()


def test_two_predictions_cannot_both_claim_one_ramp():
    # The 1:1 rule: the second is a false positive, not a second hit, so the covered
    # set stays one ramp. score_pano counts it as an FP on the same input.
    covered = cx.matched_gt([(0.5, 0.5, 0.9), (0.5, 0.5, 0.8)], [(0.5, 0.5)], RSQ)
    assert covered == {0}


def test_the_highest_confidence_prediction_claims_first():
    # Two ramps, both in range of both predictions: the order decides which pairs with
    # which, and score_pano orders by confidence. Both end up covered either way, so
    # the guard is that the ordering path runs at all rather than raising.
    gt = [(0.5, 0.5), (0.5 + R_NORM / 2, 0.5)]
    assert cx.matched_gt([(0.5, 0.5, 0.2), (0.5 + R_NORM / 2, 0.5, 0.9)], gt, RSQ) == {0, 1}


def test_predictions_without_confidence_are_matched_in_input_order():
    # The chat VLMs emit boxes with no score. Nothing to sort by, so input order.
    assert cx.matched_gt([(0.5, 0.5, None)], [(0.5, 0.5)], RSQ) == {0}


def test_a_prediction_across_the_seam_claims_the_ramp():
    # x=0.998 and x=0.002 are ~4 px apart the short way round and ~1020 px apart the
    # long way. score_pano wraps (#132); an inline non-wrapping distance here would
    # score this as a miss AND as a false positive in the same table.
    assert cx.matched_gt([(0.998, 0.5, 0.9)], [(0.002, 0.5)], RSQ) == {0}


def test_the_cells_and_the_fp_counts_come_from_one_matcher():
    # The seam case, read through both halves of the script's output: matched_gt says
    # the ramp is covered, so score_pano must call the same prediction a true positive
    # rather than a false one.
    gt = build_ground_truth([{"x_normalized": 0.002, "y_normalized": 0.5,
                              "confidence": 0.9}], [True], [], True)
    preds = [(0.998, 0.5, 0.9)]
    assert cx.matched_gt(preds, gt.gt_points, RSQ) == {0}
    assert score_pano(preds, gt, RSQ).fp == 0


def _one_pano(rampnet_x, challenger_x):
    """A single recall-eligible pano with one GT ramp at x=0.5, RampNet's shipped
    detection at ``rampnet_x`` and the challenger's at ``challenger_x``."""
    records = {"p": {"detections": [{"x_normalized": rampnet_x, "y_normalized": 0.5,
                                     "confidence": 0.9}]}}
    verdicts = {"p": {"dets": [False], "missed": [{"x": 0.5, "y": 0.5}],
                      "no_missed": False}}
    return records, verdicts, {"p": [(challenger_x, 0.5, 0.9)]}


def test_a_non_default_radius_reaches_the_cells_and_the_fp_counts_alike():
    # Both models' detections sit 1.5 default radii from the ramp. At the default
    # radius each is a miss AND a false positive; at twice the radius each is a hit
    # and no false positive. `--radius` used to reach matched_gt only, so the cells
    # moved to "both" while score_pano, still at its default, kept billing two FPs.
    records, verdicts, challenger = _one_pano(0.5 + 1.5 * R_NORM, 0.5 - 1.5 * R_NORM)
    tight = cx.partition_cells(records, verdicts, challenger.get, RSQ)
    assert tight.counts["neither"] == 1 and (tight.r_fp, tight.c_fp) == (1, 1)
    wide = cx.partition_cells(records, verdicts, challenger.get, radius_sq_for(0.044))
    assert wide.counts["both"] == 1 and (wide.r_fp, wide.c_fp) == (0, 0)


# --------------------------------------------------------------------------- #
# partition_cells — the loop three places used to carry a copy of
# --------------------------------------------------------------------------- #
def test_the_threshold_filters_the_floor_peaks_and_nothing_else():
    records, verdicts, challenger = _one_pano(0.5, 0.5)
    floor = {"p": [(0.5, 0.5, 0.20)]}                       # a peak below 0.30
    at_030 = cx.partition_cells(records, verdicts, challenger.get, RSQ, 0.30, floor)
    at_005 = cx.partition_cells(records, verdicts, challenger.get, RSQ, 0.05, floor)
    assert at_030.counts["challenger_only"] == 1
    assert at_005.counts["both"] == 1
    # The shipped detection in records.jsonl is not consulted under the flag.
    assert at_030.sites == [{"pano": "p", "x": 0.5, "y": 0.5, "cell": "challenger_only"}]


def test_the_threshold_and_the_floor_peaks_go_together():
    records, verdicts, challenger = _one_pano(0.5, 0.5)
    with pytest.raises(ValueError):
        cx.partition_cells(records, verdicts, challenger.get, RSQ, 0.30, None)
    with pytest.raises(ValueError):
        cx.partition_cells(records, verdicts, challenger.get, RSQ, None, {"p": []})


def test_a_pano_the_challenger_never_cached_is_skipped_and_counted():
    records, verdicts, _ = _one_pano(0.5, 0.5)
    part = cx.partition_cells(records, verdicts, lambda pid: None, RSQ)
    assert part.missing == 1 and part.panos == 0 and sum(part.counts.values()) == 0


def test_a_pano_missing_from_the_op_cache_is_counted_not_absorbed():
    # Under --rampnet-op-threshold a pano the op_cache does not list scores as
    # RampNet-blank: every GT ramp on it becomes a miss. It used to do so silently in
    # this script (cascade_gate.py warned; this one did not).
    records, verdicts, challenger = _one_pano(0.5, 0.5)
    part = cx.partition_cells(records, verdicts, challenger.get, RSQ, 0.30, {})
    assert part.no_floor == 1
    assert part.counts["challenger_only"] == 1          # RampNet's shipped hit is gone
    assert "1 pano(s) are absent" in cx.floor_gap_warning(part.no_floor, "richmond")
    assert cx.floor_gap_warning(0, "richmond") is None


def test_the_missed_gt_handed_to_the_null_is_what_rampnet_missed():
    records, verdicts, challenger = _one_pano(0.5 + 2 * R_NORM, 0.5)
    part = cx.partition_cells(records, verdicts, challenger.get, RSQ)
    assert part.shift_rows == [([(0.5, 0.5, 0.9)], [(0.5, 0.5)])]


def test_a_split_without_an_op_cache_exits_with_a_message_not_a_traceback(
        tmp_path, monkeypatch):
    # --rampnet-op-threshold on a split whose op_cache is missing used to be a raw
    # FileNotFoundError. cascade_gate.py already exits with a message; this mirrors it.
    monkeypatch.setattr(cx, "CACHE_DIR", str(tmp_path / "no_op_cache"))
    monkeypatch.setattr(sys, "argv", [
        "complementarity.py", "vistas:curb-cut", "richmond",
        "--rampnet-op-threshold", "0.3", "--cache-dir", str(tmp_path / "no_model_cache")])
    with pytest.raises(SystemExit) as e:
        cx.main()
    assert "op_cache" in str(e.value) and "richmond" in str(e.value)


# --------------------------------------------------------------------------- #
# model_spec — the legacy positional form, and the typo that used to be silent
# --------------------------------------------------------------------------- #
def test_a_bare_provider_uses_that_provider_default_model():
    assert cx.model_spec("vistas") == ("vistas", None)


def test_provider_colon_model_id_is_passed_through():
    assert cx.model_spec("vistas:curb-cut") == ("vistas", "curb-cut")
    assert cx.model_spec("gemini:gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")


def test_a_bare_non_provider_token_is_still_a_gemini_model_id():
    # The #35 gate's committed invocation. Keep it working.
    assert cx.model_spec("gemini-3.1-pro-preview") == ("gemini", "gemini-3.1-pro-preview")


def test_an_unknown_provider_with_a_colon_is_rejected():
    # Used to be read as the Gemini model id "foo:bar", which builds a detector whose
    # signature nothing cached: the run then reports zero detections instead of a typo.
    with pytest.raises(SystemExit) as e:
        cx.model_spec("foo:bar")
    assert "foo" in str(e.value)


def test_every_provider_name_survives_a_round_trip():
    for provider in PROVIDERS:
        assert cx.model_spec(provider) == (provider, None)


# --------------------------------------------------------------------------- #
# compare_args — the cache signature this script has to reconstruct
# --------------------------------------------------------------------------- #
def test_the_vistas_signature_matches_the_published_export():
    # The published richmond arm records the signature it was cached under. If
    # compare_args drifts from compare.py's defaults, this reconstruction stops
    # matching and every cache lookup silently misses.
    published = json.load(open(PUBLISHED, encoding="utf-8"))
    label, det = build_detector("vistas", "curb-cut", {}, cx.compare_args(_args()))
    assert label == published["model"]
    assert det.signature() == published["signature"]


def test_the_input_size_override_addresses_a_different_cache_entry():
    published = json.load(open(PUBLISHED, encoding="utf-8"))
    _, parity = build_detector("vistas", "curb-cut", {},
                               cx.compare_args(_args(vistas_input_size=[1024, 1024])))
    sig = parity.signature()
    assert sig["input_size"] == [1024, 1024]
    assert sig != published["signature"]
    # Deviation-only: the key the 384 arm was paid for is untouched by the flag existing.
    assert "input_size" not in published["signature"]
    assert (cache_key("mask2former-vistas-curb-cut", sig, "richmond", "p")
            != cache_key("mask2former-vistas-curb-cut", published["signature"],
                         "richmond", "p"))


# --------------------------------------------------------------------------- #
# complementary_null — the chance discount every "attributable" number is quoted after
# --------------------------------------------------------------------------- #
def _row(pred_xy, missed_xy):
    return ([(pred_xy[0], pred_xy[1], 1.0)], [missed_xy])


def test_no_missed_ramps_means_no_null():
    # Nothing for the challenger to recover, so there is no coincidence rate to report.
    rows = [([(0.2, 0.2, 1.0)], []), ([(0.5, 0.5, 1.0)], [])]
    assert cx.complementary_null(rows, RSQ) == (0.0, 0.0)


def test_one_pano_has_no_shift_to_take():
    # Every shift is the identity with n=1, so there is nothing to measure.
    assert cx.complementary_null([_row((0.2, 0.2), (0.2, 0.2))], RSQ) == (0.0, 0.0)


def test_boxes_that_never_line_up_give_a_zero_null():
    rows = [_row((0.1, 0.1), (0.1, 0.1)),
            _row((0.5, 0.5), (0.5, 0.5)),
            _row((0.9, 0.9), (0.9, 0.9))]
    assert cx.complementary_null(rows, RSQ) == (0.0, 0.0)


def test_identical_boxes_on_every_pano_give_a_null_of_one():
    # The degenerate high-density case: every shift matches every missed ramp, so all
    # of the "recovery" is what the radius gives away.
    rows = [_row((0.3, 0.3), (0.3, 0.3)) for _ in range(3)]
    assert cx.complementary_null(rows, RSQ) == (1.0, 1.0)


def test_the_null_averages_over_every_non_identity_shift():
    # 3 panos, 3 missed ramps, so shifts k=1 and k=2. k=1 lands 2 of 3 (panos 0 and 2
    # both miss a ramp at (0.2, 0.2), and panos 1 and 0 both predict there); k=2 lands
    # 1 of 3. Mean 0.5, worst shift 2/3.
    rows = [_row((0.2, 0.2), (0.2, 0.2)),
            _row((0.2, 0.2), (0.7, 0.7)),
            _row((0.9, 0.1), (0.2, 0.2))]
    mean, worst = cx.complementary_null(rows, RSQ)
    assert mean == pytest.approx(0.5)
    assert worst == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# regression — the published 384 richmond column, read from committed files only
# --------------------------------------------------------------------------- #
def _published_cells(tmp_path, rampnet_op_threshold=None, path=PUBLISHED, **overrides):
    """The four cells for a published Vistas arm on richmond (384 by default).

    Rebuilds a ``.model_cache``-shaped directory from the published export and reads it
    back through ``DetectionCache``/``cache_key``, i.e. the path the script itself
    takes. That is deliberate: the lookup only succeeds if the signature reconstructed
    from ``compare_args`` is the one the export was written under. The partition is
    the script's own ``partition_cells``, not a copy of it, so the threshold filter
    and the op_cache fallback are under test here too.
    """
    published = json.load(open(path, encoding="utf-8"))
    cache = DetectionCache(str(tmp_path / "cache"))
    for pid, points in published["detections"].items():
        cache.put(cache_key(published["model"], published["signature"],
                            published["city"], pid), points)

    label, det = build_detector("vistas", "curb-cut", {},
                                cx.compare_args(_args(**overrides)))
    sig = det.signature()
    records, verdicts, _ = load_bundle(os.path.join(REPO, "benchmark", "richmond"))
    floor = (cx.load_floor_peaks("richmond") if rampnet_op_threshold is not None
             else None)
    part = cx.partition_cells(
        records, verdicts, lambda pid: cache.get(cache_key(label, sig, "richmond", pid)),
        RSQ, rampnet_op_threshold, floor)
    assert part.missing == 0, "signature drifted from the published export"
    assert part.no_floor == 0
    return part.counts


def test_the_published_384_column_reproduces(tmp_path):
    # docs/model_comparison.md, "Complementarity" table, the vistas @384 column.
    # RampNet's side is the bundle's shipped detections (>= 0.5519 on richmond).
    assert _published_cells(tmp_path) == {"both": 194, "rampnet_only": 44,
                                          "challenger_only": 22, "neither": 50}


def test_the_384_column_adds_up_to_richmond_s_recall_eligible_ground_truth(tmp_path):
    counts = _published_cells(tmp_path)
    assert sum(counts.values()) == 310
    # 22 of RampNet's 72 misses recovered, 31%; and 216 of 310 for the challenger,
    # which is the recall its published row reports.
    assert counts["challenger_only"] + counts["neither"] == 72
    assert counts["both"] + counts["challenger_only"] == 216


def test_the_384_column_at_the_two_op_cache_thresholds_reproduces(tmp_path):
    # docs/model_comparison.md, the seam-exposure table: the published 384 arm with
    # RampNet re-sourced from analysis_out/op_cache/richmond.json at the recommended
    # 0.30 and at the 0.05 floor. Both sides committed, so a clean clone checks them.
    assert _published_cells(tmp_path, 0.30) == {"both": 202, "rampnet_only": 55,
                                                "challenger_only": 14, "neither": 39}
    assert _published_cells(tmp_path, 0.05) == {"both": 213, "rampnet_only": 66,
                                                "challenger_only": 3, "neither": 28}


# --------------------------------------------------------------------------- #
# regression — the published 1024 (parity) richmond column, committed since #163
# --------------------------------------------------------------------------- #
def _parity_cells(tmp_path, rampnet_op_threshold=None):
    """Same path as ``_published_cells``, addressed with the parity pin: the lookup
    only succeeds if ``--vistas-input-size 1024 1024`` rebuilds the signature the
    parity export records, which is what selects this arm over the 384 one."""
    return _published_cells(tmp_path, rampnet_op_threshold, path=PARITY,
                            vistas_input_size=[1024, 1024])


def test_the_parity_1024_column_reproduces(tmp_path):
    # docs/model_comparison.md, "Complementarity" table, the vistas @1024 column, and
    # the cells of analysis_out/cascade_gate.json. Until #163 this column rested on
    # the table alone; the detections are published now, so a clean clone checks it.
    assert _parity_cells(tmp_path) == {"both": 220, "rampnet_only": 18,
                                       "challenger_only": 54, "neither": 18}


def test_the_parity_column_at_the_recommended_point_reproduces(tmp_path):
    # The operating-point-correction table: rampnet re-sourced from the op_cache at
    # 0.30, so 16 of the 54 recovered ramps are ones RampNet already has (54 -> 38),
    # and the cells of analysis_out/cascade_gate_op030.json.
    assert _parity_cells(tmp_path, 0.30) == {"both": 236, "rampnet_only": 21,
                                             "challenger_only": 38, "neither": 15}


def test_the_parity_column_adds_up_to_its_published_row(tmp_path):
    counts = _parity_cells(tmp_path)
    assert sum(counts.values()) == 310
    assert counts["challenger_only"] + counts["neither"] == 72          # rampnet's misses
    assert counts["both"] + counts["challenger_only"] == 274            # recall 0.884


def test_the_parity_file_is_the_pinned_leg_and_the_384_file_is_not(tmp_path):
    # The two files share model and label; the parity one carries the input_size
    # key and the 384 one does not, which is the whole naming argument in #163.
    published = json.load(open(PUBLISHED, encoding="utf-8"))
    parity = json.load(open(PARITY, encoding="utf-8"))
    assert parity["model"] == published["model"] == "mask2former-vistas-curb-cut"
    assert parity["published_as"] == "mask2former-vistas-curb-cut-1024x1024"
    assert parity["signature"]["input_size"] == [1024, 1024]
    assert "input_size" not in published["signature"]
    assert parity["pins"] == {"vistas_input_size": [1024, 1024]}
    # ...and addressing the parity file at the 384 signature finds nothing.
    cache = DetectionCache(str(tmp_path / "cache"))
    for pid, points in parity["detections"].items():
        cache.put(cache_key(parity["model"], parity["signature"], parity["city"], pid),
                  points)
    label, det = build_detector("vistas", "curb-cut", {}, cx.compare_args(_args()))
    assert cache.get(cache_key(label, det.signature(), "richmond",
                               next(iter(parity["detections"])))) is None

