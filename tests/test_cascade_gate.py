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
which is what the second half of this file does — including what the "no floor peak in
radius" row *is*: the committed ``op_cache`` says where each such site's nearest peak
sits and whether the matcher already gave it to a neighbour, and the doc's reading of
that row is pinned to those columns here.
"""
import json
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import cascade_gate as cg  # noqa: E402
import silent_activation as sa  # noqa: E402
from farfield_forensics import quartiles  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, radius_sq_for  # noqa: E402

OUT = os.path.join(REPO, "analysis_out")
SHIPPED = os.path.join(OUT, "cascade_gate.json")
OP030 = os.path.join(OUT, "cascade_gate_op030.json")
RSQ = radius_sq_for()
R_PX = RSQ ** 0.5                            # 22.5 heatmap px
R_NORM = R_PX / PANO_SCALE_X                 # the same radius in normalized x


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
# no_peak_profile — what the "no floor peak in radius" row is made of
# --------------------------------------------------------------------------- #
def test_the_profile_covers_only_the_rows_with_no_peak_in_radius():
    rows = [_row("challenger_only", peak_in_radius=True, nearest_peak_px=5.0),
            _row("challenger_only", peak_in_radius=False, nearest_peak_px=30.0,
                 argmax_off_px=22.4, act=0.3),
            _row("challenger_only", peak_in_radius=False, nearest_peak_px=60.0,
                 argmax_off_px=3.0, act=0.1, **{"class": "faint_local"}),
            _row("neither", peak_in_radius=False, nearest_peak_px=30.0)]
    out = cg.no_peak_profile(rows, "challenger_only", R_PX)
    assert out["n"] == 2
    assert out["nearest_peak_px_median"] == 60.0       # quartiles: v[n // 2]
    assert out["peak_within_2r"] == 1 and out["peak_beyond_2r"] == 1
    assert out["argmax_on_edge"] == 1                  # 22.4 is within 0.5 px of R
    assert out["classes"] == {"absent": 0, "faint_local": 1, "tail": 1}
    assert "claimed" not in out                        # rows predate the column


def test_the_profile_reports_the_claimed_count_only_when_every_row_carries_it():
    rows = [_row("neither", peak_in_radius=False, nearest_peak_px=30.0,
                 nearest_peak_claimed=True),
            _row("neither", peak_in_radius=False, nearest_peak_px=30.0,
                 nearest_peak_claimed=False)]
    assert cg.no_peak_profile(rows, "neither", R_PX)["claimed"] == 1


def test_an_empty_profile_reports_n_zero():
    assert cg.no_peak_profile([_row("both")], "neither", R_PX) == {"cell": "neither",
                                                                   "n": 0}


def test_a_row_with_no_peak_anywhere_counts_as_beyond_two_radii():
    rows = [_row("neither", peak_in_radius=False, nearest_peak_px=None,
                 nearest_peak_score=None)]
    out = cg.no_peak_profile(rows, "neither", R_PX)
    assert out["nearest_peak_px_median"] is None
    assert out["peak_within_2r"] == 0 and out["peak_beyond_2r"] == 1


# --------------------------------------------------------------------------- #
# claimed_by_adjacent — the #130 mechanism, per site
# --------------------------------------------------------------------------- #
def test_a_neighbours_detection_between_two_ramps_is_claimed_by_the_nearer_one():
    # Two ramps 1.5 R apart, one peak on the left ramp. The left ramp's nearest peak
    # is its own (not claimed by another); the right ramp's nearest peak is that same
    # one, and the matcher gave it to the left ramp.
    sites = [{"x": 0.5, "y": 0.5}, {"x": 0.5 + 1.5 * R_NORM, "y": 0.5}]
    preds = [(0.5, 0.5, 0.9)]
    assert cg.claimed_by_adjacent(sites, preds, 0.30, RSQ) == [False, True]


def test_a_peak_below_the_threshold_claims_nothing():
    sites = [{"x": 0.5, "y": 0.5}, {"x": 0.5 + 1.5 * R_NORM, "y": 0.5}]
    assert cg.claimed_by_adjacent(sites, [(0.5, 0.5, 0.2)], 0.30, RSQ) == [False, False]


def test_no_peaks_means_nothing_is_claimed():
    assert cg.claimed_by_adjacent([{"x": 0.5, "y": 0.5}], [], 0.30, RSQ) == [False]


def test_nearest_peak_index_agrees_with_nearest_peak_across_the_seam():
    preds = [(0.5, 0.5, 0.9), (0.998, 0.5, 0.4)]
    x, y = 0.002, 0.5
    i = cg.nearest_peak_index(preds, x, y)
    assert i == 1
    assert sa.nearest_peak(preds, x, y)[1] == preds[i][2]
    assert cg.nearest_peak_index([], x, y) is None


# --------------------------------------------------------------------------- #
# site_rng — a site's null must not depend on which sites came before it
# --------------------------------------------------------------------------- #
def _heat_with_bump():
    # A ramp along the site's row, so every azimuth draws a different value and the
    # null's median and p95 depend on which azimuths were drawn.
    h = [[0.0] * 1024 for _ in range(512)]
    for c in range(1024):
        h[256][c] = 0.03 * c / 1024
    h[256][512] = 0.04
    return h


def test_a_sites_null_is_the_same_whatever_ran_before_it():
    h = _heat_with_bump()
    x, y = 512 / 1024, 256 / 512
    first = sa.null_percentile(h, x, y, cg.site_rng("p", x, y), trials=50)
    # Consume a different amount of "other sites" work, then ask again.
    for other in range(3):
        sa.null_percentile(h, 0.1 * (other + 1), y, cg.site_rng("p", 0.1 * (other + 1), y),
                           trials=50)
    again = sa.null_percentile(h, x, y, cg.site_rng("p", x, y), trials=50)
    assert first == again


def test_one_stream_shared_across_sites_did_depend_on_order():
    # The shape being replaced: the same site, read second instead of first from one
    # stream, draws different azimuths. Guarded so the reason for site_rng stays
    # demonstrable rather than remembered.
    h = _heat_with_bump()
    x, y = 512 / 1024, 256 / 512
    rng = random.Random(sa.NULL_SEED)
    alone = sa.null_percentile(h, x, y, rng, trials=50)[2:]
    rng = random.Random(sa.NULL_SEED)
    sa.null_percentile(h, 0.1, y, rng, trials=50)
    after_another = sa.null_percentile(h, x, y, rng, trials=50)[2:]
    assert alone != after_another


def test_different_sites_get_different_streams():
    a = cg.site_rng("p", 0.5, 0.5).random()
    assert a != cg.site_rng("p", 0.5, 0.6).random()
    assert a != cg.site_rng("q", 0.5, 0.5).random()
    assert a == cg.site_rng("p", 0.5, 0.5).random()


# --------------------------------------------------------------------------- #
# panos_without_floor — the probe path's op_cache gap, which used to be silent
# --------------------------------------------------------------------------- #
def test_a_pano_the_op_cache_does_not_list_is_named():
    floor = {"a": [(0.5, 0.5, 0.2)], "b": []}
    assert cg.panos_without_floor(["a", "b", "c"], floor) == ["c"]
    # Listed with zero floor peaks is an answer, not a gap.
    assert cg.panos_without_floor(["b"], floor) == []


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
    # adjacent GT, and 15 carry none inside the radius.
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


def _claimed_at_030(payload):
    """``{(pano, x, y): bool}`` -- is the site's nearest floor peak one the greedy
    match at 0.30 gave to a different GT? From the committed op_cache, the same
    source the artifact's peak columns were read from."""
    floor = cg.load_floor_peaks(payload["split"])
    rsq = radius_sq_for(payload["radius"])
    by_pano = {}
    for s in payload["sites"]:
        by_pano.setdefault(s["pano"], []).append(s)
    out = {}
    for pid, sites in by_pano.items():
        flags = cg.claimed_by_adjacent(sites, floor.get(pid, []),
                                       payload["rampnet_op_threshold"], rsq)
        for s, flag in zip(sites, flags):
            out[(pid, s["x"], s["y"])] = flag
    return out


def test_the_no_peak_row_is_mostly_a_neighbours_shoulder_not_unpeaked_mass():
    # docs/model_comparison.md, the "no floor peak in radius" row of the cascade
    # table. The first write-up read these 15 as "unpeaked heatmap mass
    # peak_local_max never called a maximum". The artifact's own columns say
    # otherwise: for 11 of the 15 the nearest floor peak is 1-2 R away (median 35.0
    # px against R = 22.5), and for 11 the in-window maximum sits on the window edge
    # (median argmax_off_px 22.4) -- a neighbouring mode's shoulder reaching in, #46
    # Phase 1's `tail`, which is where class_of puts 12 of the 15. Only 4 have no
    # floor peak within 2 R, and one of those is the seam site whose peak the
    # pre-f4c71c8 op_cache dropped.
    payload = _payload(OP030)
    r_px = radius_sq_for(payload["radius"]) ** 0.5
    prof = cg.no_peak_profile(payload["sites"], "challenger_only", r_px)
    assert prof["n"] == 15
    assert prof["nearest_peak_px_median"] == 35.0
    assert prof["argmax_off_px_median"] == 22.4
    assert prof["argmax_on_edge"] == 11
    assert prof["peak_within_2r"] == 11
    assert prof["peak_beyond_2r"] == 4
    assert prof["classes"] == {"absent": 0, "faint_local": 3, "tail": 12}
    assert prof["seam"] == 1
    assert prof["act_median"] == 0.2723


def test_the_no_peak_rows_nearest_peaks_are_mostly_claimed_by_an_adjacent_ramp():
    # ...and that neighbouring peak is, for 11 of the 15, one the matcher already
    # gave to another GT at 0.30 (7 of them within 2 R) -- the same #130 mechanism
    # the table's "4 in radius at >= 0.30" row names, so those two rows are one
    # cause, not two. All 4 of that row are claimed too, which is what "unmatched"
    # there meant.
    payload = _payload(OP030)
    r_px = radius_sq_for(payload["radius"]) ** 0.5
    claimed = _claimed_at_030(payload)
    co = [s for s in payload["sites"] if s["cell"] == "challenger_only"]
    no_peak = [s for s in co if not s["peak_in_radius"]]
    assert sum(claimed[(s["pano"], s["x"], s["y"])] for s in no_peak) == 11
    assert sum(claimed[(s["pano"], s["x"], s["y"])] for s in no_peak
               if s["nearest_peak_px"] <= 2 * r_px) == 7
    in_r_high = [s for s in co if s["peak_in_radius"] and s["nearest_peak_score"] >= 0.30]
    assert len(in_r_high) == 4
    assert all(claimed[(s["pano"], s["x"], s["y"])] for s in in_r_high)
    # The re-cut of the 38 the doc now prints, exhaustive and disjoint.
    promotable = sum(1 for s in co if s["peak_in_radius"] and s["nearest_peak_score"] < 0.30)
    within_2r_unclaimed = sum(1 for s in no_peak if s["nearest_peak_px"] <= 2 * r_px
                              and not claimed[(s["pano"], s["x"], s["y"])])
    beyond_2r = sum(1 for s in no_peak if s["nearest_peak_px"] > 2 * r_px)
    assert (promotable, len(in_r_high) + 7, within_2r_unclaimed, beyond_2r) == (19, 11, 4, 4)
    assert promotable + 11 + within_2r_unclaimed + beyond_2r == 38


def test_the_seam_site_in_the_recovered_cell_has_a_peak_the_op_cache_lacks():
    # 723487737079243 at x = 0.0069: act 0.946 at 7.4 px from the ramp, centre 0.78,
    # and the committed op_cache's nearest peak 117 px away. That is the f4c71c8 seam
    # dropout made concrete -- a regenerated op_cache would list this peak, making
    # the site a RampNet hit at 0.30 and taking it out of challenger_only (38 -> 37).
    payload = _payload(OP030)
    site = next(s for s in payload["sites"]
                if s["cell"] == "challenger_only" and s["seam"])
    assert site["pano"] == "723487737079243"
    assert not site["peak_in_radius"] and site["nearest_peak_px"] > 100
    assert site["act"] > 0.9 and site["center"] > 0.7 and site["argmax_off_px"] < 10


def test_the_committed_nulls_came_from_one_stream_and_say_so():
    # Both artifacts predate site_rng: their nulls were drawn from one stream in pano
    # order, so a site carries a different draw in each file (43 of the 53 sites with
    # a null in both differ, by up to 0.075) while act and nearest_peak_px agree on
    # every one. A regeneration with per-site seeding moves those values without any
    # change in the heatmap, which is why neither file records "null_rng".
    shipped, op030 = _payload(SHIPPED), _payload(OP030)
    assert "null_rng" not in shipped and "null_rng" not in op030
    key = lambda s: (s["pano"], s["x"], s["y"])  # noqa: E731
    a = {key(s): s for s in shipped["sites"]}
    both = [(a[key(s)], s) for s in op030["sites"]
            if key(s) in a and a[key(s)]["null_pct"] is not None
            and s["null_pct"] is not None]
    assert len(both) == 53
    differ = [(x, y) for x, y in both if x["null_pct"] != y["null_pct"]]
    assert len(differ) == 43
    assert max(abs(x["null_pct"] - y["null_pct"]) for x, y in differ) == pytest.approx(0.075)
    assert all(x["act"] == y["act"] and x["nearest_peak_px"] == y["nearest_peak_px"]
               for x, y in both)


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
