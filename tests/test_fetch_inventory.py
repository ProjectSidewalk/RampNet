"""Unit tests for inventory snapshot fetching (issues #96, #59).

No network — every test drives the pure parsing/paging core with canned payloads
shaped like the ones these publishers actually return. The load-bearing
guarantees: a feature with no geometry is dropped rather than becoming a ramp at
(0, 0), ID paging terminates instead of looping, and the snapshot's own sha256 is
stable across re-fetches of identical data, because an unstable digest makes the
manifest useless as a provenance record (§9).
"""
import gzip
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import fetch_inventory as fi  # noqa: E402


# --------------------------------------------------------------------------- #
# query construction
# --------------------------------------------------------------------------- #
def test_query_url_requests_wgs84_and_all_fields():
    url = fi.arcgis_query_url("https://x/FeatureServer/0")
    assert "outSR=4326" in url
    assert "outFields=%2A" in url or "outFields=*" in url
    assert "returnGeometry=true" in url


def test_query_url_pages_by_object_id_not_offset():
    """Offset paging is unstable against a layer that refreshes weekly."""
    url = fi.arcgis_query_url("https://x/FeatureServer/0", min_oid=500)
    assert "OBJECTID+%3E+500" in url or "OBJECTID%20%3E%20500" in url
    assert "resultOffset" not in url


def test_query_url_keeps_a_user_where_clause_when_paging():
    url = fi.arcgis_query_url("https://x/FeatureServer/0", where="STATUS='A'", min_oid=7)
    assert "STATUS" in url and "%3E+7" in url.replace("%20", "+")


def test_count_only_query_asks_for_no_geometry():
    url = fi.arcgis_query_url("https://x/FeatureServer/0", count_only=True)
    assert "returnCountOnly=true" in url
    assert "outFields" not in url


# --------------------------------------------------------------------------- #
# ArcGIS parsing
# --------------------------------------------------------------------------- #
def test_parse_flattens_geometry_onto_the_record():
    payload = {"features": [{"attributes": {"OBJECTID": 1, "UPDATE_STATUS": "NC"},
                             "geometry": {"x": -105.0, "y": 39.7}}]}
    recs, exceeded = fi.parse_arcgis_page(payload)
    assert recs == [{"OBJECTID": 1, "UPDATE_STATUS": "NC", "lon": -105.0, "lat": 39.7}]
    assert exceeded is False


def test_parse_drops_attribute_only_rows():
    """ArcGIS returns these happily; a point inventory row with no point is not
    a ramp location, and defaulting it to 0,0 would put labels in the Atlantic."""
    payload = {"features": [
        {"attributes": {"OBJECTID": 1}, "geometry": None},
        {"attributes": {"OBJECTID": 2}, "geometry": {"x": None, "y": 39.7}},
        {"attributes": {"OBJECTID": 3}, "geometry": {"x": -105.0, "y": 39.7}},
    ]}
    recs, _ = fi.parse_arcgis_page(payload)
    assert [r["OBJECTID"] for r in recs] == [3]


def test_parse_raises_on_a_server_error_rather_than_returning_empty():
    """An empty page is the pagination stop condition, so a silently-empty error
    response would truncate a city and look like a complete fetch."""
    try:
        fi.parse_arcgis_page({"error": {"code": 400, "message": "Invalid where"}})
    except RuntimeError as exc:
        assert "400" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_reports_the_transfer_limit_flag():
    _, exceeded = fi.parse_arcgis_page({"features": [], "exceededTransferLimit": True})
    assert exceeded is True


def test_max_oid_ignores_non_integer_ids():
    assert fi.max_oid([{"OBJECTID": 3}, {"OBJECTID": 9}, {"OBJECTID": None}]) == 9
    assert fi.max_oid([{"OBJECTID": "abc"}]) is None


# --------------------------------------------------------------------------- #
# Socrata parsing — both coordinate shapes appear across the candidates
# --------------------------------------------------------------------------- #
def test_socrata_reads_a_geojson_point_column():
    rows = [{"rampid": "1", "the_geom": {"type": "Point", "coordinates": [-73.9, 40.8]}}]
    got = fi.parse_socrata_page(rows, point_field="the_geom")
    assert got[0]["lon"] == -73.9 and got[0]["lat"] == 40.8


def test_socrata_reads_flat_columns():
    got = fi.parse_socrata_page([{"longitude": "-73.9", "latitude": "40.8"}])
    assert got[0]["lon"] == -73.9 and got[0]["lat"] == 40.8


def test_socrata_drops_unparseable_coordinates():
    rows = [{"longitude": "", "latitude": "40.8"},
            {"the_geom": {"type": "Point", "coordinates": []}},
            {"longitude": "-73.9", "latitude": "40.8"}]
    assert len(fi.parse_socrata_page(rows, point_field="the_geom")) == 1


# --------------------------------------------------------------------------- #
# snapshot writing — the provenance contract from §9
# --------------------------------------------------------------------------- #
def test_snapshot_round_trips_and_records_its_own_digest(tmp_path):
    recs = [{"OBJECTID": 1, "lon": -105.0, "lat": 39.7},
            {"OBJECTID": 2, "lon": -105.1, "lat": 39.8}]
    manifest = {"city": "x", "fetched": "2026-07-31", "endpoint": "https://x"}
    payload, man = fi.write_snapshot("x", recs, manifest, out_dir=str(tmp_path))
    assert os.path.basename(payload) == "x-2026-07-31.jsonl.gz"
    with gzip.open(payload, "rt") as fh:
        back = [json.loads(line) for line in fh if line.strip()]
    assert back == recs
    saved = json.load(open(man))
    assert saved["records"] == 2
    assert len(saved["sha256"]) == 64
    assert saved["payload"] == os.path.basename(payload)


def test_identical_data_hashes_identically(tmp_path):
    """gzip stamps mtime by default, which would make every re-fetch of unchanged
    data look like a change and destroy the digest's value as a drift signal."""
    recs = [{"OBJECTID": 1, "lon": -105.0, "lat": 39.7}]
    a = json.load(open(fi.write_snapshot(
        "a", recs, {"city": "a", "fetched": "2026-07-31"}, out_dir=str(tmp_path))[1]))
    b = json.load(open(fi.write_snapshot(
        "b", recs, {"city": "b", "fetched": "2026-07-31"}, out_dir=str(tmp_path))[1]))
    assert a["sha256"] == b["sha256"]


def test_record_order_is_preserved_in_the_payload(tmp_path):
    recs = [{"OBJECTID": i, "lon": 0.0, "lat": 0.0} for i in (5, 1, 3)]
    payload, _ = fi.write_snapshot("x", recs, {"city": "x", "fetched": "2026-07-31"},
                                   out_dir=str(tmp_path))
    with gzip.open(payload, "rt") as fh:
        assert [json.loads(l)["OBJECTID"] for l in fh if l.strip()] == [5, 1, 3]


# --------------------------------------------------------------------------- #
# polyline geometry -- the centreline reference (§5i)
# --------------------------------------------------------------------------- #
def test_polyline_mode_keeps_paths_instead_of_a_point():
    payload = {"features": [{"attributes": {"OBJECTID": 3, "SND_ID": 9},
                             "geometry": {"paths": [[[-122.0, 47.6], [-122.0, 47.61]]]}}]}
    recs, _ = fi.parse_arcgis_page(payload, geometry="polyline")
    assert recs == [{"OBJECTID": 3, "SND_ID": 9,
                     "paths": [[[-122.0, 47.6], [-122.0, 47.61]]]}]
    assert "lon" not in recs[0]


def test_polyline_mode_drops_a_single_vertex_path():
    """One vertex carries no direction, so it can support no perpendicular."""
    payload = {"features": [{"attributes": {"OBJECTID": 1},
                             "geometry": {"paths": [[[-122.0, 47.6]]]}}]}
    recs, _ = fi.parse_arcgis_page(payload, geometry="polyline")
    assert recs == []


def test_polyline_mode_drops_geometry_free_rows():
    payload = {"features": [{"attributes": {"OBJECTID": 1}, "geometry": None}]}
    assert fi.parse_arcgis_page(payload, geometry="polyline")[0] == []


def test_polyline_mode_keeps_every_part_of_a_multipart_line():
    payload = {"features": [{"attributes": {"OBJECTID": 4}, "geometry": {"paths": [
        [[-122.0, 47.6], [-122.0, 47.61]], [[-122.1, 47.7], [-122.1, 47.71]]]}}]}
    recs, _ = fi.parse_arcgis_page(payload, geometry="polyline")
    assert len(recs[0]["paths"]) == 2


def test_point_mode_is_unchanged_and_stays_the_default():
    """The centreline work must not disturb how every existing inventory parses."""
    payload = {"features": [{"attributes": {"OBJECTID": 1},
                             "geometry": {"x": -105.0, "y": 39.7}}]}
    assert fi.parse_arcgis_page(payload) == fi.parse_arcgis_page(payload,
                                                                 geometry="point")
    assert fi.parse_arcgis_page(payload)[0][0]["lon"] == -105.0


def test_a_point_payload_read_as_polyline_yields_nothing_rather_than_zeros():
    """Mismatching the mode must fail loudly-empty, not invent geometry."""
    payload = {"features": [{"attributes": {"OBJECTID": 1},
                             "geometry": {"x": -105.0, "y": 39.7}}]}
    assert fi.parse_arcgis_page(payload, geometry="polyline")[0] == []
