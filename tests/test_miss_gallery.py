"""Unit tests for the miss gallery's instrument (#46).

Pure geometry and bookkeeping — no imagery, no PIL, no panos on disk.

This file exists because the gallery's failure mode is not a crash, it is a
*confidently wrong verdict*. The guarantees that prevent that:

* **native sampling** — a panel is never rendered wider than the imagery supports,
  so upsampled mush can never be mistaken for detail;
* **parity classification** — a pano stored at the model's own width gives the
  reviewer no advantage, and that has to be detected per pano (richmond mixes
  4096-px and 12288-px panos in one split);
* **the judgeability floor** — items too small for anyone to call are excluded from
  rates rather than labelled;
* **centred views** — every subject sits dead centre, so framing cannot bias a
  judgment between one crop and another;
* **tag keys** matching ``incremental_fp_tags.json`` exactly, so reviewer verdicts
  stay joinable with the existing human-tag corpus.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import miss_gallery as mg  # noqa: E402


# --------------------------------------------------------------------------- #
# native_panel_width — never invent detail, never discard it
# --------------------------------------------------------------------------- #
def test_panel_width_is_the_arc_of_imagery_the_fov_covers():
    # 60 deg of a 4096-px-around panorama is 4096/6 px of real imagery.
    assert mg.native_panel_width(60.0, 4096, max_px=10000) == round(4096 / 6)


def test_a_wider_fov_takes_more_imagery():
    assert (mg.native_panel_width(60.0, 8192, max_px=10000)
            > mg.native_panel_width(16.0, 8192, max_px=10000))


def test_a_bigger_pano_gives_a_bigger_panel():
    assert (mg.native_panel_width(16.0, 16384, max_px=10000)
            == 4 * mg.native_panel_width(16.0, 4096, max_px=10000))


def test_the_cap_is_respected():
    assert mg.native_panel_width(60.0, 16384, max_px=1400) == 1400


def test_a_tiny_pano_still_yields_a_usable_panel():
    # Floor exists so a degenerate source cannot produce a 3-px image.
    assert mg.native_panel_width(1.0, 512, max_px=1400) == 64


# --------------------------------------------------------------------------- #
# source_px / parity — the reviewer's pixels vs the model's
# --------------------------------------------------------------------------- #
def test_source_pixels_scale_with_the_stored_width():
    assert mg.source_px(20.0, 16384) == 80.0        # 4x pano -> 4x the pixels
    assert mg.source_px(20.0, 4096) == 20.0         # stored at model width -> same


def test_a_pano_at_the_model_width_is_parity():
    assert mg.parity_class(4096) == "parity"


def test_a_larger_pano_is_advantaged():
    for w in (5760, 8000, 11000, 16384):
        assert mg.parity_class(w) == "advantaged", w


def test_a_pano_smaller_than_the_model_input_is_still_parity():
    # paterson's smallest pano is 3328 px: the model upsamples it, so the reviewer
    # has no advantage. Treating it as 'advantaged' would be backwards.
    assert mg.parity_class(3328) == "parity"


def test_the_parity_band_tolerates_a_negligible_excess():
    assert mg.parity_class(4096 * 1.04) == "parity"
    assert mg.parity_class(4096 * 1.20) == "advantaged"


# --------------------------------------------------------------------------- #
# judgeable — the floor that stops confident labels on nothing
# --------------------------------------------------------------------------- #
def test_the_floor_excludes_ramps_too_small_to_call():
    assert not mg.judgeable(17.0)
    assert mg.judgeable(30.0)
    assert mg.judgeable(200.0)


def test_the_floor_is_configurable():
    assert mg.judgeable(40.0, floor=50.0) is False
    assert mg.judgeable(60.0, floor=50.0) is True


# --------------------------------------------------------------------------- #
# views_for — centring, and the convention it must match
# --------------------------------------------------------------------------- #
def test_the_subject_lands_dead_centre_in_both_panels():
    from equirect_tiling import equirect_point_to_perspective
    x, y = 0.31, 0.62
    for view in mg.views_for(x, y, 16384):
        uv = equirect_point_to_perspective(x, y, view)
        assert uv is not None
        assert abs(uv[0] - 0.5) < 1e-6 and abs(uv[1] - 0.5) < 1e-6


def test_centring_holds_across_the_pano_including_the_seam():
    from equirect_tiling import equirect_point_to_perspective
    for x in (0.001, 0.25, 0.5, 0.75, 0.999):
        for y in (0.52, 0.60, 0.74):
            for view in mg.views_for(x, y, 8192):
                uv = equirect_point_to_perspective(x, y, view)
                assert uv is not None, (x, y)
                assert abs(uv[0] - 0.5) < 1e-6 and abs(uv[1] - 0.5) < 1e-6


def test_yaw_and_pitch_follow_the_equirect_convention():
    # equirect_tiling: lon = (x-0.5)*2pi, lat = (0.5-y)*pi.
    ctx, _ = mg.views_for(0.75, 0.60, 4096)
    assert abs(ctx.yaw_deg - 90.0) < 1e-9
    assert abs(ctx.pitch_deg - (-18.0)) < 1e-9


def test_the_detail_panel_is_tighter_than_the_context_panel():
    ctx, det = mg.views_for(0.5, 0.6, 8192)
    assert det.fov_h_deg < ctx.fov_h_deg


def test_panels_are_square_so_display_scaling_cannot_distort():
    for view in mg.views_for(0.5, 0.6, 8192):
        assert view.width == view.height
        assert view.fov_h_deg == view.fov_v_deg


def test_the_model_budget_view_is_coarser_than_the_source_view():
    # The third panel is rendered from a 4096-px-wide pano; on a 4x split it must
    # carry a quarter of the linear detail, which is the comparison it exists for.
    _, det = mg.views_for(0.5, 0.6, 16384)
    _, model_det = mg.views_for(0.5, 0.6, mg.MODEL_WIDTH)
    assert model_det.width * 4 == det.width


def test_the_model_budget_view_equals_the_source_view_at_parity():
    _, det = mg.views_for(0.5, 0.6, 4096)
    _, model_det = mg.views_for(0.5, 0.6, mg.MODEL_WIDTH)
    assert model_det.width == det.width


# --------------------------------------------------------------------------- #
# tag_key — must match the existing human-tag corpus
# --------------------------------------------------------------------------- #
def test_tag_key_matches_the_incremental_fp_tag_format():
    # Real key from benchmark/gainesville/incremental_fp_tags.json.
    assert mg.tag_key("b-azHLq5uJSlywVqKYbgHA", 0.68262, 0.52344) == \
        "b-azHLq5uJSlywVqKYbgHA_0.68262_0.52344"


def test_tag_key_is_stable_for_equal_coordinates():
    assert mg.tag_key("p", 0.5, 0.6) == mg.tag_key("p", 0.50000, 0.60000)


def test_tag_key_separates_distinct_ramps_in_one_pano():
    assert mg.tag_key("p", 0.24609, 0.55273) != mg.tag_key("p", 0.60547, 0.55273)


# --------------------------------------------------------------------------- #
# summarize_feasibility — what the gallery is allowed to conclude
# --------------------------------------------------------------------------- #
def _item(parity, judgeable_flag, spx=100.0):
    return {"parity": parity, "judgeable": judgeable_flag, "source_px": spx}


def test_summary_counts_each_class():
    items = [_item("parity", True), _item("parity", False),
             _item("advantaged", True), _item("advantaged", True)]
    s = mg.summarize_feasibility(items)
    assert s["total"] == 4 and s["judgeable"] == 3
    assert s["parity"]["n"] == 2 and s["parity"]["judgeable"] == 1
    assert s["advantaged"]["n"] == 2 and s["advantaged"]["judgeable"] == 2


def test_the_model_budget_panel_licenses_every_judgeable_item():
    # Without the third panel only parity items would support an appearance verdict;
    # with it, the reviewer compares budgets directly on advantaged panos too.
    items = [_item("advantaged", True), _item("parity", True), _item("advantaged", False)]
    s = mg.summarize_feasibility(items)
    assert s["appearance_licensed"] == s["judgeable"] == 2
    assert s["parity_only"] == 1


def test_unjudgeable_items_never_license_a_verdict():
    s = mg.summarize_feasibility([_item("parity", False), _item("advantaged", False)])
    assert s["appearance_licensed"] == 0 and s["parity_only"] == 0


def test_summary_survives_an_empty_population():
    s = mg.summarize_feasibility([])
    assert s["total"] == 0 and s["judgeable"] == 0
    assert s["parity"]["median_source_px"] is None


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_the_model_width_matches_the_architecture():
    from rampnet.model import PANO_INPUT_SIZE
    # PANO_INPUT_SIZE is (height, width); the parity maths is all in width.
    assert mg.MODEL_WIDTH == PANO_INPUT_SIZE[1]


def test_the_buckets_offered_are_the_taxonomy_s_own():
    import miss_taxonomy as mt
    assert set(mt.BUCKETS) >= {"silent", "merged"}
    assert mg.mt.BUCKETS is mt.BUCKETS
