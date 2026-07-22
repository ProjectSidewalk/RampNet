"""Tests for human-verdict precision/recall scoring (rampnet.validation).

Ported from the deployment pipeline's score_validation suite, which owned this
logic before it moved here. Pure-Python: no torch.
    pytest tests/test_validation.py -v
"""
from rampnet.validation import (
    collect, format_report, total_ramps, wilson_interval, THRESHOLDS,
)


def test_wilson_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert 0.0 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0


def test_wilson_interval_contains_p_and_narrows():
    lo_s, hi_s = wilson_interval(5, 10)
    lo_l, hi_l = wilson_interval(500, 1000)
    assert lo_s < 0.5 < hi_s and lo_l < 0.5 < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)  # more data, tighter interval


CONFS = {
    "P_TWO_DETS": [0.91, 0.63],
    "P_EMPTY": [],
    "P_ONE_DET": [0.7],
}


def _panos(no_missed_flags=True):
    """Three panos: fully judged w/ a missed mark, empty+affirmed, judged-unaffirmed."""
    panos = {
        "P_TWO_DETS": {"group": "random", "dets": [True, True],
                       "missed": [{"x": 0.2, "y": 0.3}], "no_missed": False},
        "P_EMPTY": {"group": "empty", "dets": [], "missed": [], "no_missed": True},
        "P_ONE_DET": {"group": "random", "dets": [False], "missed": [], "no_missed": False},
    }
    if not no_missed_flags:  # old-schema export: flag absent everywhere
        for entry in panos.values():
            del entry["no_missed"]
    return panos


def test_collect_unconfirmed_pano_counts_for_precision_not_recall():
    pools = collect(_panos(), CONFS)
    assert (pools.n_seen, pools.n_judged, pools.n_unconfirmed) == (3, 3, 1)
    assert (pools.n_unsure, pools.missed_unsure) == (0, 0)
    # P_ONE_DET's crop verdict is valid for precision even though its missed-ramp
    # check was never confirmed...
    assert sorted(pools.judged) == [(0.63, True), (0.7, False), (0.91, True)]
    # ...but it is excluded from the recall pool.
    assert sorted(pools.recall_judged) == [(0.63, True), (0.91, True)]
    assert pools.missed_total == 1
    assert total_ramps(pools) == 3  # 2 correct in recall pool + 1 missed


def test_collect_assume_scanned_includes_unconfirmed_panos():
    # Reviewer attests they scanned every pano: P_ONE_DET (no missed mark, not
    # affirmed) must now enter the recall pool instead of being held out.
    pools = collect(_panos(), CONFS, assume_scanned=True)
    assert (pools.n_seen, pools.n_judged, pools.n_unconfirmed) == (3, 3, 0)
    assert sorted(pools.recall_judged) == [(0.63, True), (0.7, False), (0.91, True)]
    assert pools.missed_total == 1


def test_collect_old_schema_keeps_legacy_behavior():
    pools = collect(_panos(no_missed_flags=False), CONFS)
    # Legacy entries (no no_missed key) are trusted for recall, as before.
    assert (pools.n_seen, pools.n_judged, pools.n_unconfirmed) == (3, 3, 0)
    assert len(pools.judged) == 3 and len(pools.recall_judged) == 3 and pools.missed_total == 1


def test_collect_mixed_schema_gates_per_entry():
    # A legacy entry hand-merged into a new-schema file must stay trusted; the
    # new-schema unconfirmed entry must still be excluded from recall.
    panos = _panos()
    del panos["P_TWO_DETS"]["no_missed"]  # legacy entry, missed mark
    pools = collect(panos, CONFS)
    assert pools.n_unconfirmed == 1  # P_ONE_DET only
    assert sorted(pools.recall_judged) == [(0.63, True), (0.91, True)]
    assert pools.missed_total == 1


def test_collect_unsure_abstains_from_both_metrics():
    # A pano with one confident-correct det, one 'unsure' det, and one 'unsure'
    # missed mark. The unsure crop leaves precision/recall; the unsure missed mark
    # leaves the recall denominator; both are counted separately. The unsure missed
    # mark still confirms the pano was scanned (it enters the recall pool).
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True, "unsure"],
                            "missed": [{"x": 0.1, "y": 0.1, "unsure": True}],
                            "no_missed": False}}
    pools = collect(panos, {"P_TWO_DETS": [0.91, 0.63]})
    assert (pools.n_seen, pools.n_judged, pools.n_unconfirmed) == (1, 1, 0)
    assert pools.judged == [(0.91, True)]          # 'unsure' det dropped from precision
    assert pools.recall_judged == [(0.91, True)]   # ...and from recall
    assert (pools.n_unsure, pools.missed_total, pools.missed_unsure) == (1, 0, 1)


def test_collect_duplicate_is_false_positive_by_default():
    # Two dets on one physical ramp: the reviewer keeps the first 'correct' and
    # marks the second 'duplicate'. By default the duplicate scores as a false
    # positive (matching rampnet.metrics' one-to-one matching), but stays counted
    # distinctly from a plain 'incorrect'.
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True, "duplicate"],
                            "missed": [], "no_missed": True}}
    pools = collect(panos, {"P_TWO_DETS": [0.91, 0.63]})
    assert pools.n_judged == 1 and pools.n_duplicate == 1 and pools.n_unsure == 0
    # Folded to False in the pools, so precision sees 1 correct of 2 (an FP).
    assert sorted(pools.judged) == [(0.63, False), (0.91, True)]
    assert sorted(pools.recall_judged) == [(0.63, False), (0.91, True)]
    assert total_ramps(pools) == 1  # only the one correct det is a real ramp


def test_collect_duplicate_lenient_abstains():
    # The lenient variant drops duplicates from both metrics entirely (like unsure),
    # so precision/recall are computed as if the redundant hit never fired.
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True, "duplicate"],
                            "missed": [], "no_missed": True}}
    pools = collect(panos, {"P_TWO_DETS": [0.91, 0.63]}, lenient_duplicates=True)
    assert pools.n_duplicate == 1
    assert pools.judged == [(0.91, True)] and pools.recall_judged == [(0.91, True)]
    assert total_ramps(pools) == 1


def test_collect_skips_partially_judged():
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True, None],
                            "missed": [], "no_missed": True}}
    pools = collect(panos, CONFS)
    assert pools.n_seen == 1 and pools.n_judged == 0
    assert pools.judged == [] and pools.recall_judged == [] and pools.missed_total == 0


def test_collect_mismatched_counts_warns_not_prints():
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True],  # results has 2
                            "missed": [], "no_missed": True}}
    pools = collect(panos, CONFS)
    assert pools.n_seen == 0 and pools.n_judged == 0 and pools.judged == []
    assert any("don't match" in w for w in pools.warnings)  # surfaced as data, not stdout


def test_collect_exclude_top():
    panos = _panos()
    panos["P_TWO_DETS"]["group"] = "top"
    pools = collect(panos, CONFS, exclude_top=True)
    assert pools.n_seen == 2 and pools.n_judged == 2
    # P_EMPTY (affirmed, no dets) and P_ONE_DET (unconfirmed) remain.
    assert pools.judged == [(0.7, False)] and pools.recall_judged == [] and pools.missed_total == 0
    assert pools.n_unconfirmed == 1


def test_format_report_headline_and_sweep():
    pools = collect(_panos(), CONFS)
    text = format_report("All reviewed panos", pools)
    assert "Precision: 0.667" in text          # 2 correct of 3 judged
    assert "Recall:    0.667" in text           # 2 correct of 3 ramps
    # Threshold sweep present, and raising the bar drops the low-confidence FP:
    # at 0.75 only the two 0.91/0.63->0.91 correct dets survive on the precision side.
    assert "threshold" in text and "precision" in text
    prec_at_75 = [ln for ln in text.splitlines() if ln.strip().startswith("0.75")]
    assert prec_at_75 and "1.000" in prec_at_75[0]  # 0.7 FP excluded -> precision 1.0


def test_format_report_nothing_judged():
    pools = collect({"P": {"group": "random", "dets": [None], "missed": [], "no_missed": True}},
                    {"P": [0.9]})
    assert "Nothing judged yet." in format_report("Empty", pools)
