"""OSM crossing nodes as a Stage 1 location source (#274 Phase 1 item 3). Stdlib, no network."""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "stage_one", "dataset_generation"))

import crossing_source as cs  # noqa: E402


def node(i, lat, lng, **tags):
    return {"type": "node", "id": i, "lat": lat, "lon": lng, "tags": {"highway": "crossing", **tags}}


def test_osm_class_matches_the_phase0_census_rule():
    assert cs.osm_class({"crossing": "zebra"}) == "marked"
    assert cs.osm_class({"crossing": "uncontrolled"}) == "untagged"
    assert cs.osm_class({"crossing": "uncontrolled", "crossing:markings": "zebra"}) == "marked"
    assert cs.osm_class({"crossing": "marked", "crossing:markings": "no"}) == "unmarked"
    assert cs.osm_class({"crossing": "informal"}) == "unmarked"
    assert cs.osm_class({}) == "untagged"


def test_positive_rule_marked_kept_unmarked_dropped_untagged_needs_crowd():
    els = [node(3, 47.6000, -122.3, crossing="marked"),
           node(1, 47.6010, -122.3, crossing="unmarked"),
           node(2, 47.6020, -122.3),                       # untagged, crowd label ~5.6 m away
           node(4, 47.6030, -122.3),                       # untagged, nearest crowd label ~100 m away
           {"type": "way", "id": 9, "nodes": [1, 2]}]
    crowd = [(47.60205, -122.3), (47.6039, -122.3)]
    rows = cs.crossing_locations(els, crowd_points=crowd)
    assert [r["source_id"] for r in rows] == ["osm:node/2", "osm:node/3"]   # sorted by node id
    assert {r["osm_class"] for r in rows} == {"untagged", "marked"}
    # without crowd points an untagged node is never kept
    assert [r["source_id"] for r in cs.crossing_locations(els)] == ["osm:node/3"]
    # the rule is a parameter
    rows = cs.crossing_locations(els, positive=("marked", "unmarked"), positive_with_crowd=())
    assert [r["source_id"] for r in rows] == ["osm:node/1", "osm:node/3"]


def test_rows_extend_the_curb_ramp_location_schema_without_changing_it():
    """The first three columns are exactly what combine_location_data.py writes for curb ramps, and the date is
    empty (undated), so a CurbRamp all_locations.csv is untouched and a Crosswalk row reads like an undated ramp."""
    src = open(os.path.join(REPO, "stage_one", "dataset_generation", "combine_location_data.py"), encoding="utf-8").read()
    fieldnames = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", "") == "fieldnames" for t in n.targets)]
    assert len(fieldnames) == 1
    assert ast.literal_eval(fieldnames[0].value) == cs.FIELDS[:3]
    rows = cs.crossing_locations([node(1, 1.0, 2.0, crossing="marked")])
    assert list(rows[0]) == cs.FIELDS and rows[0]["date"] == "" and rows[0]["label_type"] == "Crosswalk"


def test_duplicate_nodes_from_overlapping_tiles_appear_once():
    els = [node(5, 1.0, 2.0, crossing="zebra"), node(5, 1.0, 2.0, crossing="zebra")]
    assert len(cs.crossing_locations(els)) == 1
