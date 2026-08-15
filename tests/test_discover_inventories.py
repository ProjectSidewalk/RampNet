"""Unit tests for the inventory discovery sweep (issue #96 §3, §5m).

The sweep's job is to stop a *supply* question being answered by a *search*
artefact — §3 records a pass that searched one phrase, concluded supply was thin,
and was wrong. Its failure modes are now the opposite: "ramp" is overloaded, and
the dangerous false positive is a planned-work layer, which has the same wrong
polarity as Atlanta's *Missing ADA Ramps* — a list of places a ramp is NEEDED is
not a list of ramps. CPU only, no network.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import discover_inventories as d  # noqa: E402


def keep(name):
    return d.looks_like_ramps({"name": name})


# --------------------------------------------------------------------------- #
# what should be kept
# --------------------------------------------------------------------------- #
def test_the_synonym_set_is_the_instrument_not_one_phrase():
    """§3: a title search for 'curb ramp' does not match NYC's own dataset."""
    assert "curb ramp" in d.QUERIES
    assert "pedestrian ramp" in d.QUERIES
    assert "curb cut" in d.QUERIES
    assert len(d.QUERIES) >= 6


def test_real_inventories_survive_the_filter():
    for n in ("Curb Ramps", "DVRPC Pedestrian Ramps", "ADA Curb Ramps",
              "Ped Ramps", "Curb Cut", "Sidewalk ADA Ramps", "sCurbRamps",
              "Pedestrian Ramp Inventory", "Ramp Inventory 2019"):
        assert keep(n), n


# --------------------------------------------------------------------------- #
# what must be rejected -- each of these was produced by a real run
# --------------------------------------------------------------------------- #
def test_boat_and_rail_ramps_are_rejected():
    assert not keep("FWC Florida Boat Ramp Inventory")
    assert not keep("CSX Ramp Address Points")


def test_planned_work_layers_are_rejected_for_polarity_not_noise():
    """The dangerous class. Atlanta's 'Missing ADA Ramps' is confirmed ABSENCE
    (§3) -- counting these as supply would inflate the pool with records that
    assert a ramp does NOT exist, which is exactly backwards for Stage 1."""
    for n in ("MAF Missing ADA Ramps Draft", "Barrier Free Ramp Projects",
              "ADA Ramp Needs", "Ramp Improvements View", "No Curb Ramp",
              "Proposed Curb Ramps", "Planned Ped Ramps",
              "KYTC - Sidewalk or Ramp Obstructions - Points"):
        assert not keep(n), n


def test_maintenance_layers_are_rejected():
    """Work orders and inspections track ACTIVITY, not the asset -- one ramp can
    have many, so counting them would double-count supply."""
    assert not keep("Curb Ramp Work Orders")


def test_a_layer_with_no_ramp_word_at_all_is_rejected():
    """Hub matches on description too, so a sidewalk-segment layer surfaces on a
    ramp query. Counting segments as ramps would be a category error."""
    assert not keep("Sidewalk Lines Inspections Most Recent Insp")
    assert not keep("Sidewalk Inventory")


def test_matching_is_case_insensitive_both_ways():
    assert keep("PROWAG CURB RAMP ASSESSMENT VL")
    assert not keep("FLORIDA BOAT RAMP INVENTORY")


# --------------------------------------------------------------------------- #
# known-city bookkeeping
# --------------------------------------------------------------------------- #
def test_cities_already_in_the_doc_are_flagged_not_dropped():
    """Known cities are reported separately rather than hidden, so 'known' stays
    auditable -- otherwise a filter bug silently shrinks the candidate pool."""
    assert d.is_known({"orgName": "City of Seattle ArcGIS Online", "name": "Curb Ramps"})
    assert d.is_known({"orgName": "City of Charlotte", "name": "ADA Curb Ramps"})
    assert not d.is_known({"orgName": "DVRPC-GIS", "name": "DVRPC Pedestrian Ramps"})


def test_known_check_reads_both_org_and_name():
    assert d.is_known({"orgName": "", "name": "Denver Curb Ramps"})
    assert d.is_known({"orgName": "BostonMaps", "name": ""})


def test_point_geometry_is_what_stage_1_can_use():
    assert "esriGeometryPoint" in d.POINT_TYPES
    assert "esriGeometryPolyline" not in d.POINT_TYPES


# --------------------------------------------------------------------------- #
# pagination — a short page is not the end
# --------------------------------------------------------------------------- #
def _paged(monkeypatch, pages):
    """Serve `pages` (a list of row-count ints) to search(), and record the URLs."""
    seen = []

    def fake_fetch(url, **kw):
        seen.append(url)
        i = len(seen) - 1
        if i >= len(pages):
            return {"data": []}
        return {"data": [{"id": f"{i}-{k}"} for k in range(pages[i])]}

    monkeypatch.setattr(d, "fetch", fake_fetch)
    return seen


def test_a_short_page_does_not_terminate_the_sweep(monkeypatch):
    """The Hub filters after paging, so page 2 can be short and page 3 full.
    Stopping on the short one silently truncates a supply count."""
    seen = _paged(monkeypatch, [100, 80, 100])
    rows = d.search("curb ramp", pages=3, page_size=100)
    assert len(rows) == 280
    assert len(seen) == 3


def test_an_empty_page_does_terminate_it(monkeypatch):
    seen = _paged(monkeypatch, [100, 0, 100])
    rows = d.search("curb ramp", pages=3, page_size=100)
    assert len(rows) == 100
    assert len(seen) == 2


def test_the_page_cap_still_bounds_the_cost(monkeypatch):
    seen = _paged(monkeypatch, [100, 100, 100, 100])
    d.search("curb ramp", pages=2, page_size=100)
    assert len(seen) == 2
