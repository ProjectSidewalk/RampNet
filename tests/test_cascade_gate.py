"""Unit tests for the cascade gate (#126).

Pure logic plus a drift guard on the committed artifacts — no GPU, no imagery, no
``.model_cache``. The heavy half (one RampNet forward per pano) is exercised by running
the script; what these protect is the bookkeeping the write-up quotes.

``summarize`` is the whole reporting surface: every number in the cascade tables in
``docs/model_comparison.md`` is one of its keys, and it is re-derivable from the
``sites`` list committed alongside it. So the guard here is that ``summarize`` applied
to a committed run's ``sites`` reproduces that run's ``cells`` exactly — which also
pins which *subset* each figure is a median of, the thing a hand-copied number gets
wrong (the cell's ``act_median`` is not the no-peak rows' median).

The parity (1024x1024) detections behind those two artifacts are not published, so a
clean clone cannot regenerate these files. It can still check them against themselves,
which is what the second half of this file does.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import cascade_gate as cg  # noqa: E402
from farfield_forensics import quartiles  # noqa: E402

OUT = os.path.join(REPO, "analysis_out")
SHIPPED = os.path.join(OUT, "cascade_gate.json")
OP030 = os.path.join(OUT, "cascade_gate_op030.json")


def _row(cell, **kw):
    """One site row, with the fields ``summarize`` reads and neutral defaults."""
    row = {"cell": cell, "act": 0.5, "center": 0.5, "argmax_off_px": 0.0,
           "nearest_peak_px": 0.0, "nearest_peak_score": 0.5, "class": "tail",
           "peak_in_radius": True, "seam": False,
           "null_pct": None, "null_med": None, "null_p95": None}
    row.update(kw)
    return row


# --------------------------------------------------------------------------- #
# cell_of — the 2x2 that names the recoverable set
# --------------------------------------------------------------------------- #
def test_the_four_cells():
    assert cg.cell_of(True, True) == "both"
    assert cg.cell_of(True, False) == "rampnet_only"
    assert cg.cell_of(False, True) == "challenger_only"
    assert cg.cell_of(False, False) == "neither"


def test_only_the_two_rampnet_miss_cells_get_a_null():
    # The null is meaningless where RampNet found the ramp: the activation is high by
    # construction there, which is why those cells are the positive control. So the
    # cells a null is reported for are exactly the two cell_of returns on a miss.
    assert set(cg.CELLS) == {"both", "rampnet_only", "challenger_only", "neither"}
    assert set(cg.MISS_CELLS) == {cg.cell_of(False, True), cg.cell_of(False, False)}


# --------------------------------------------------------------------------- #
# summarize — the reporting surface
# --------------------------------------------------------------------------- #
def test_an_empty_cell_reports_n_zero_and_nothing_else():
    out = cg.summarize([_row("both")], "neither")
    assert out == {"cell": "neither", "n": 0}


def test_it_selects_only_its_own_cell():
    rows = [_row("both", act=0.9), _row("challenger_only", act=0.1),
            _row("challenger_only", act=0.3), _row("challenger_only", act=0.4)]
    out = cg.summarize(rows, "challenger_only")
    assert out["n"] == 3
    assert out["act_median"] == 0.3


def test_peak_in_radius_is_counted_and_shared():
    rows = [_row("challenger_only", peak_in_radius=True),
            _row("challenger_only", peak_in_radius=True),
            _row("challenger_only", peak_in_radius=False, nearest_peak_px=40.0),
            _row("challenger_only", peak_in_radius=False, nearest_peak_px=60.0)]
    out = cg.summarize(rows, "challenger_only")
    assert out["peak_in_radius"] == 2
    assert out["peak_in_radius_share"] == 0.5


def test_the_peak_score_median_covers_only_the_rows_that_have_a_peak():
    # This is the distinction a hand-copied number loses: a per-row column's median is
    # over the rows that have that column, not over the cell.
    rows = [_row("challenger_only", peak_in_radius=True, nearest_peak_score=0.10),
            _row("challenger_only", peak_in_radius=True, nearest_peak_score=0.20),
            _row("challenger_only", peak_in_radius=True, nearest_peak_score=0.30),
            _row("challenger_only", peak_in_radius=False, nearest_peak_score=0.99)]
    out = cg.summarize(rows, "challenger_only")
    # The 0.99 belongs to a row with no peak in radius, so it is out of this median.
    assert out["peak_in_radius_score_median"] == 0.2


def test_a_cell_with_no_peak_anywhere_reports_no_peak_score():
    rows = [_row("neither", peak_in_radius=False, nearest_peak_px=None,
                 nearest_peak_score=None)]
    out = cg.summarize(rows, "neither")
    assert out["peak_in_radius"] == 0
    assert "peak_in_radius_score_median" not in out
    assert out["nearest_peak_px_median"] is None


def test_the_hit_cells_carry_no_null_statistics():
    out = cg.summarize([_row("both"), _row("both")], "both")
    for key in ("null_pct_median", "above_null_p95", "null_med_median"):
        assert key not in out


def test_above_null_p95_counts_sites_over_their_own_shifted_p95():
    rows = [_row("challenger_only", act=0.5, null_pct=0.99, null_med=0.01, null_p95=0.2),
            _row("challenger_only", act=0.1, null_pct=0.40, null_med=0.02, null_p95=0.2),
            _row("challenger_only", act=0.3, null_pct=0.70, null_med=0.03, null_p95=0.2)]
    out = cg.summarize(rows, "challenger_only")
    # act > this site's own p95 on the 0.5 and 0.3 rows, not on the 0.1 one.
    assert out["above_null_p95"] == 2
    assert out["null_pct_median"] == 0.7
    assert out["null_med_median"] == 0.02


def test_seam_sites_are_counted_per_cell():
    rows = [_row("challenger_only", seam=True), _row("challenger_only", seam=False),
            _row("both", seam=True)]
    assert cg.summarize(rows, "challenger_only")["seam"] == 1
    assert cg.summarize(rows, "both")["seam"] == 1


def test_the_class_shares_sum_to_one():
    rows = [_row("neither", **{"class": "absent"}),
            _row("neither", **{"class": "faint_local"}),
            _row("neither", **{"class": "tail"}),
            _row("neither", **{"class": "tail"})]
    out = cg.summarize(rows, "neither")
    assert out["classes"] == {"absent": 1, "faint_local": 1, "tail": 2}
    assert sum(out["class_share"].values()) == 1.0


# --------------------------------------------------------------------------- #
# the committed artifacts — every cell figure re-derives from the sites list
# --------------------------------------------------------------------------- #
def _payload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_committed_cells_re_derive_from_the_committed_sites():
    for path in (SHIPPED, OP030):
        payload = _payload(path)
        assert [cg.summarize(payload["sites"], c) for c in cg.CELLS] == payload["cells"], (
            f"{os.path.basename(path)}: cells no longer match summarize(sites)")


def test_the_two_artifacts_partition_the_same_310_richmond_ramps():
    for path in (SHIPPED, OP030):
        payload = _payload(path)
        assert payload["split"] == "richmond"
        assert payload["challenger"] == "mask2former-vistas-curb-cut"
        assert payload["vistas_input_size"] == [1024, 1024]
        assert payload["n_sites"] == 310
        assert sum(c["n"] for c in payload["cells"]) == 310
        assert payload["skipped_sites"] == 0


def test_only_the_op030_artifact_records_the_threshold_key():
    # cascade_gate.json was written before --rampnet-op-threshold entered the payload,
    # so a regeneration would add "rampnet_op_threshold": null and change its bytes
    # even with identical results. Stated in docs/model_comparison.md beside it.
    assert "rampnet_op_threshold" not in _payload(SHIPPED)
    assert _payload(OP030)["rampnet_op_threshold"] == 0.3


def test_the_cascade_ceiling_is_nineteen_promotable_ramps():
    # docs/model_comparison.md, "The cascade gate": of the 38 genuinely-complementary
    # ramps at rampnet@0.30, 19 carry a floor peak in radius scoring 0.05-0.30 (the
    # promotable set), 4 carry one at >= 0.30 that the greedy matcher gave to an
    # adjacent GT, and 15 carry none.
    sites = [s for s in _payload(OP030)["sites"] if s["cell"] == "challenger_only"]
    assert len(sites) == 38
    peaked = [s for s in sites if s["peak_in_radius"]]
    no_peak = [s for s in sites if not s["peak_in_radius"]]
    assert len(no_peak) == 15
    assert sum(1 for s in peaked if 0.05 <= s["nearest_peak_score"] < 0.30) == 19
    assert sum(1 for s in peaked if s["nearest_peak_score"] >= 0.30) == 4


def test_the_no_peak_rows_have_their_own_activation_median():
    # The row in the cascade table is about the 15 sites with nothing to promote, so
    # its activation figure is those 15 sites' median (0.2723) -- not the whole
    # challenger_only cell's (0.2152, which is what cells[].act_median reports).
    payload = _payload(OP030)
    cell = next(c for c in payload["cells"] if c["cell"] == "challenger_only")
    sites = [s for s in payload["sites"] if s["cell"] == "challenger_only"]
    no_peak = [s["act"] for s in sites if not s["peak_in_radius"]]
    assert round(quartiles(no_peak)[1], 4) == 0.2723
    assert cell["act_median"] == 0.2152


def test_moving_to_the_recommended_threshold_takes_sixteen_from_the_recovered_cell():
    # docs/model_comparison.md: RampNet gains 19 hits going 0.55 -> 0.30 (72 -> 53
    # misses), of which 16 come out of challenger_only and 3 out of neither. The
    # complementary-gain headline is the challenger_only figure, so it falls by 16.
    shipped = {c["cell"]: c["n"] for c in _payload(SHIPPED)["cells"]}
    op030 = {c["cell"]: c["n"] for c in _payload(OP030)["cells"]}
    assert shipped["challenger_only"] == 54 and op030["challenger_only"] == 38
    assert shipped["neither"] == 18 and op030["neither"] == 15
    gained = ((shipped["challenger_only"] + shipped["neither"])
              - (op030["challenger_only"] + op030["neither"]))
    assert gained == 19
    assert shipped["challenger_only"] - op030["challenger_only"] == 16
    assert shipped["neither"] - op030["neither"] == 3


def test_the_seam_exposure_is_one_ramp_in_the_recovered_cell():
    # The bound on the pre-#132 non-wrapping match: 6 of richmond's 310 GT ramps
    # straddle the seam, and only 1 of them is in challenger_only.
    payload = _payload(OP030)
    assert sum(1 for s in payload["sites"] if s["seam"]) == 6
    by_cell = {c["cell"]: c["seam"] for c in payload["cells"]}
    assert by_cell["challenger_only"] == 1
    assert by_cell["both"] == 5
