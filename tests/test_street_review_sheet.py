"""Tests for the street-level review sheet's pure half (#103).

What is pinned, and why it matters:

* **The strip edges come from the one crop definition** — asymmetric, and
  asserted against ``crop_half_angle_deg()`` rather than literals.
* **The pano-pick rule** is the instrument's sampling-within-a-record; every
  branch (band, temporal eligibility, tie-breaks) is exercised because a wrong
  pick silently changes what the reviewer judges.
* **The caches tell the truth about absence** — §5h's zero-byte trap is the
  named enemy: absences are readable marker files, ``--refetch-absent``
  actually refetches, and a failed search is never cached as a result.
* **The shared-field construction site** — chips and verdict templates extend
  one base dict whose keys must equal ``SHARED_FIELDS``; the JS export copies
  the same list (asserted end-to-end in the page-logic test).

No network, no GPU: GSV calls are stubbed by injecting a fake ``search_panos``
module / monkeypatching ``rampnet.gsv.fetch_panorama``.
"""
import json
import math
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import street_review_sheet as srs  # noqa: E402
from stage1_offset_tolerance import crop_half_angle_deg  # noqa: E402


# --------------------------------------------------------------------------- #
# strip geometry — one definition
# --------------------------------------------------------------------------- #
def test_strip_edges_come_from_the_crop_definition():
    assert srs.STRIP_RIGHT_DEG == pytest.approx(crop_half_angle_deg(), abs=1e-12)
    assert srs.STRIP_LEFT_DEG == pytest.approx(-math.degrees(math.atan(171 / 512)), abs=1e-12)
    assert abs(srs.STRIP_LEFT_DEG) > srs.STRIP_RIGHT_DEG  # asymmetric, wider left


# --------------------------------------------------------------------------- #
# haversine
# --------------------------------------------------------------------------- #
def test_haversine_known_values():
    # One degree of latitude ~111.2 km, anywhere.
    assert srs.haversine_m(39.0, -105.0, 40.0, -105.0) == pytest.approx(111195, rel=0.01)
    # ~11 m east at Denver's latitude.
    d = srs.haversine_m(39.7392, -104.9903, 39.7392, -104.99017)
    assert d == pytest.approx(11.1, abs=0.5)
    assert srs.haversine_m(39.7, -105.0, 39.7, -105.0) == 0.0


# --------------------------------------------------------------------------- #
# the pano-pick rule
# --------------------------------------------------------------------------- #
LAT, LON = 39.7392, -104.9903


def _pano(pid, dlat_m=0.0, dlon_m=10.0, date="2020-6"):
    """A candidate a given metric offset from the record."""
    return {"pano_id": pid, "lat": LAT + dlat_m / 111132.0,
            "lon": LON + dlon_m / (111320.0 * math.cos(math.radians(LAT))),
            "date": date}


def test_pick_empty_and_out_of_band():
    chosen, status, stats = srs.choose_pano([], LAT, LON, None)
    assert (chosen, status) == (None, "no_panos")
    far = [_pano("far", dlon_m=200.0), _pano("close", dlon_m=1.0)]
    chosen, status, stats = srs.choose_pano(far, LAT, LON, None)
    assert (chosen, status) == (None, "no_pano_in_band")
    assert stats == {"n_panos": 2, "n_in_band": 0, "n_eligible": 0}


def test_pick_nearest_in_band():
    cands = [_pano("a", dlon_m=25.0), _pano("b", dlon_m=8.0), _pano("c", dlon_m=15.0)]
    chosen, status, _ = srs.choose_pano(cands, LAT, LON, None)
    assert status == "ok"
    assert chosen["pano_id"] == "b"
    assert chosen["range_m"] == pytest.approx(8.0, abs=0.2)


def test_temporal_eligibility_is_per_record():
    """#103 argument 4: a pano captured before the record's date cannot show
    the ramp, so it is ineligible — per record, not per city."""
    cands = [_pano("old", dlon_m=6.0, date="2014-3"),
             _pano("new", dlon_m=20.0, date="2016-1")]
    chosen, status, stats = srs.choose_pano(cands, LAT, LON, record_ym=(2015, 1))
    assert status == "ok"
    assert chosen["pano_id"] == "new"          # nearer one is too old
    assert stats["n_in_band"] == 2 and stats["n_eligible"] == 1

    chosen, status, _ = srs.choose_pano(
        [_pano("old", dlon_m=6.0, date="2014-3")], LAT, LON, record_ym=(2015, 1))
    assert (chosen, status) == (None, "no_dated_pano_in_band")

    # Capture in the record's own month counts — ">= record ym".
    chosen, _, _ = srs.choose_pano(
        [_pano("same", dlon_m=6.0, date="2015-1")], LAT, LON, record_ym=(2015, 1))
    assert chosen["pano_id"] == "same"


def test_undated_record_accepts_any_pano_and_undated_pano_needs_a_dated_record_rule():
    """record_ym None -> everything in band is eligible (flagged in the
    manifest, not silently); an UNDATED PANO is ineligible for a dated record
    because 'captured after' cannot be established."""
    cands = [_pano("undated", dlon_m=6.0, date=None)]
    chosen, status, _ = srs.choose_pano(cands, LAT, LON, record_ym=None)
    assert status == "ok" and chosen["pano_id"] == "undated"
    chosen, status, _ = srs.choose_pano(cands, LAT, LON, record_ym=(2015, 1))
    assert (chosen, status) == (None, "no_dated_pano_in_band")


def test_tie_break_is_newest_then_id_deterministic():
    a = _pano("aaa", dlon_m=10.0, date="2019-5")
    b = _pano("bbb", dlon_m=10.0, date="2023-8")
    chosen, _, _ = srs.choose_pano([a, b], LAT, LON, None)
    assert chosen["pano_id"] == "bbb"          # same range -> newest capture
    c = dict(b, pano_id="ccc")
    chosen, _, _ = srs.choose_pano([c, b], LAT, LON, None)
    assert chosen["pano_id"] == "bbb"          # same range+date -> lowest id
    # GSV's non-padded dates order correctly through parse_ym: 2019-10 > 2019-9.
    d1 = _pano("d1", dlon_m=10.0, date="2019-9")
    d2 = _pano("d2", dlon_m=10.0, date="2019-10")
    chosen, _, _ = srs.choose_pano([d1, d2], LAT, LON, None)
    assert chosen["pano_id"] == "d2"


# --------------------------------------------------------------------------- #
# neighbour bearings
# --------------------------------------------------------------------------- #
def test_neighbour_offsets_signs_membership_and_labels():
    """A record east of the crosshair bearing must come out POSITIVE
    (clockwise/right — the §5j sign); membership is within 35 m of the PANO
    (the production inclusion rule); the label is distance from the RECORD."""
    pano_lat, pano_lon = LAT, LON
    rec = _pano("rec", dlon_m=11.0)            # record 11 m east of pano
    rows = [
        {"OBJECTID": "self", "lon": rec["lon"], "lat": rec["lat"]},
        # 10 m north of the pano: bearing 0 vs az_gov 90 -> offset -90 (out of view)
        {"OBJECTID": "north", "lon": pano_lon, "lat": pano_lat + 10.0 / 111132.0},
        # 4 m south of the record: still ~east of the pano, a few deg clockwise
        {"OBJECTID": "south_of_rec", "lon": rec["lon"],
         "lat": rec["lat"] - 4.0 / 111132.0},
        # 100 m east: outside the 35 m pano radius entirely
        {"OBJECTID": "far", "lon": pano_lon + 100.0 / (111320.0 * math.cos(math.radians(LAT))),
         "lat": pano_lat},
    ]
    az_gov = 90.0  # record is due east of the pano
    drawable, n_out = srs.neighbour_offsets(
        rows, rec["lon"], rec["lat"], pano_lon, pano_lat, az_gov,
        self_id="self", id_field="OBJECTID")
    assert n_out == 1                            # "north" is at -90°
    assert len(drawable) == 1                    # far excluded, self excluded
    off, d_m = drawable[0]
    assert off > 0                               # south of an east-pointing view = clockwise
    assert d_m == pytest.approx(4.0, abs=0.3)    # labelled from the RECORD


def test_neighbour_offsets_excludes_self_even_at_zero_offset():
    rows = [{"OBJECTID": "self", "lon": LON, "lat": LAT}]
    drawable, n_out = srs.neighbour_offsets(
        rows, LON, LAT, LON, LAT + 1e-4, 180.0, self_id="self")
    assert drawable == [] and n_out == 0


# --------------------------------------------------------------------------- #
# shared-field construction — the anti-§5l mechanism
# --------------------------------------------------------------------------- #
def _base():
    site = {"id": "42", "lon": LON, "lat": LAT, "stratum": "dated_before"}
    chosen = {"pano_id": "P", "lat": LAT + 1e-4, "lon": LON, "date": "2021-3",
              "range_m": 11.1}
    return srs.make_base_record(site, chosen, heading=123.4, az_gov=90.0,
                                theta=-33.4, n_candidates=7)


def test_base_record_keys_are_exactly_the_shared_fields():
    assert set(_base().keys()) == set(srs.SHARED_FIELDS)


def test_main_source_writes_every_verdict_field():
    """The server-side template is the second of the three paths; a verdict
    field missing from it would resurface §5l's dropped-stratum bug. String-
    level, because main() needs network to run."""
    with open(os.path.join(REPO, "scripts", "analysis", "street_review_sheet.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    tail = src[src.index("verdicts.append"):]
    block = tail[:tail.index("))") + 2]
    for field in srs.VERDICT_FIELDS:
        assert field in block, "verdict template misses {!r}".format(field)


# --------------------------------------------------------------------------- #
# build_sheet
# --------------------------------------------------------------------------- #
def _chips(n=2):
    out = []
    for i in range(n):
        base = dict(_base(), id=chr(65 + i))
        out.append(dict(base, uri="", ctx_uri="",
                        neighbors=[[10.0, 5.2], [-30.5, 12.0]],
                        n_neighbors_out_of_view=1))
    return out


def _meta():
    return {"city": "denver-co", "seed": 20260731,
            "inventory": "denver-co-2026-07-31.jsonl.gz", "sites_desc": "59 records"}


def test_build_sheet_substitutes_every_token_and_namespaces_storage():
    html = srs.build_sheet(_meta(), _chips(), {"city": "denver-co",
                                               "rubric": srs.RUBRIC})
    assert "__CHIPS__" not in html and "__META__" not in html
    assert "__CITY__" not in html and "__BUILD__" not in html
    # The localStorage key MUST be namespaced: the aerial Denver sheet shares
    # city AND seed with this one, and an unnamespaced key merges their state.
    assert '"rampnet-gsv-verdicts-" + META.city' in html
    assert srs.sheet_build_id() in html
    meta = json.loads(html.split("const META = ", 1)[1].split(";\n", 1)[0])
    assert meta["shared_fields"] == list(srs.SHARED_FIELDS)
    assert meta["strip_left_deg"] == pytest.approx(srs.STRIP_LEFT_DEG)
    assert meta["strip_right_deg"] == pytest.approx(srs.STRIP_RIGHT_DEG)
    assert meta["ctx_w"] == srs.CTX_W and meta["ctx_h"] == srs.CTX_H
    assert [tuple(r) for r in meta["reasons"]] == list(srs.UNREADABLE_REASONS)


def test_build_id_tracks_template_and_rubric(monkeypatch):
    before = srs.sheet_build_id()
    monkeypatch.setattr(srs, "RUBRIC", dict(srs.RUBRIC, extra="clause"))
    assert srs.sheet_build_id() != before


# --------------------------------------------------------------------------- #
# caches — honest absence
# --------------------------------------------------------------------------- #
def test_search_cache_hit_miss_and_failure_semantics(tmp_path, monkeypatch):
    calls = []

    fake = types.ModuleType("search_panos")

    class _P:
        def __init__(s):
            s.pano_id, s.lat, s.lon, s.heading, s.date = "X", 1.0, 2.0, 90.0, "2020-1"

    def search_panoramas(lat, lon):
        calls.append((lat, lon))
        if lat < 0:
            raise RuntimeError("endpoint drift")
        return [] if lat > 50 else [_P()]

    fake.search_panoramas = search_panoramas
    monkeypatch.setitem(sys.modules, "search_panos", fake)

    d = str(tmp_path)
    # miss -> network -> cached; hit -> no network
    assert srs.cached_search(40.0, -105.0, d)[0]["pano_id"] == "X"
    assert srs.cached_search(40.0, -105.0, d)[0]["pano_id"] == "X"
    assert len(calls) == 1
    # an EMPTY result is a result, and caches
    assert srs.cached_search(60.0, -105.0, d) == []
    assert srs.cached_search(60.0, -105.0, d) == []
    assert len(calls) == 2
    # a FAILURE is not a result, and never caches — a transient must not
    # masquerade as "no coverage here" (§5h's trap, by construction)
    with pytest.raises(RuntimeError):
        srs.cached_search(-1.0, -105.0, d)
    with pytest.raises(RuntimeError):
        srs.cached_search(-1.0, -105.0, d)
    assert len(calls) == 4


def test_search_retries_transient_http_but_not_schema_drift(tmp_path, monkeypatch):
    """The first Denver probe failed 15/59 sites on GetMetadata HTTP 502 —
    rate limiting mid-burst, all recoverable. A transient status is retried
    with backoff before it is believed (§5h's rule); genuine schema drift
    still raises immediately, because retrying THAT would hide a broken
    parser behind four slow attempts."""
    calls = {"n": 0}
    fake = types.ModuleType("search_panos")

    class _P:
        pano_id, lat, lon, heading, date = "X", 1.0, 2.0, 90.0, "2020-1"

    def search_panoramas(lat, lon):
        calls["n"] += 1
        if lat == 1.0 and calls["n"] < 3:
            raise RuntimeError("GetMetadata returned HTTP 502 for pano X")
        if lat == 2.0:
            raise RuntimeError("date path [1][0][6][7] not found — schema drift?")
        return [_P()]

    fake.search_panoramas = search_panoramas
    monkeypatch.setitem(sys.modules, "search_panos", fake)
    monkeypatch.setattr(srs.time, "sleep", lambda s: None)

    d = str(tmp_path)
    assert srs.cached_search(1.0, -105.0, d)[0]["pano_id"] == "X"
    assert calls["n"] == 3                       # two 502s, then success
    calls["n"] = 0
    with pytest.raises(RuntimeError, match="schema drift"):
        srs.cached_search(2.0, -105.0, d)
    assert calls["n"] == 1                       # drift is NOT retried


def test_pano_cache_success_and_absence_marker(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    import rampnet.gsv as gsv

    calls = []

    def fake_fetch(pano_id):
        calls.append(pano_id)
        if pano_id == "gone":
            return None
        return np.full((8, 16, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(gsv, "fetch_panorama", fake_fetch)
    d = str(tmp_path)

    # success: cached as jpg, fetcher not called again, BGR round-trips
    equi, reason = srs.fetch_panorama_cached("ok", d)
    assert reason is None and equi.shape == (8, 16, 3)
    equi2, _ = srs.fetch_panorama_cached("ok", d)
    assert equi2 is not None and calls == ["ok"]

    # absence: retried within the call, then a READABLE marker — never a
    # zero-byte sentinel (§5h)
    equi, reason = srs.fetch_panorama_cached("gone", d, retry_sleep_s=0.0)
    assert equi is None and reason == "fetch_returned_none"
    assert calls == ["ok", "gone", "gone"]
    marker = os.path.join(d, "gone.absent.json")
    with open(marker, encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["reason"] == "fetch_returned_none" and m["attempts"] == 2

    # cached absence answers without the network...
    equi, reason = srs.fetch_panorama_cached("gone", d, retry_sleep_s=0.0)
    assert equi is None and reason.startswith("absent_cached")
    assert len(calls) == 3

    # ...and --refetch-absent actually refetches: the §5h retry fix was inert
    # precisely because a cached absence short-circuited it.
    equi, reason = srs.fetch_panorama_cached("gone", d, refetch_absent=True,
                                             retry_sleep_s=0.0)
    assert len(calls) == 5                      # the fetcher genuinely ran again
    assert equi is None and reason == "fetch_returned_none"
    assert os.path.exists(marker)               # still absent -> marker rewritten


def test_pano_cache_network_exception_is_not_cached(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    import rampnet.gsv as gsv

    def boom(pano_id):
        raise OSError("network down")

    monkeypatch.setattr(gsv, "fetch_panorama", boom)
    with pytest.raises(OSError):
        srs.fetch_panorama_cached("p", str(tmp_path))
    assert os.listdir(str(tmp_path)) == []      # no marker, no jpg


# --------------------------------------------------------------------------- #
# sites from a built aerial sheet — the committed Denver fixture
# --------------------------------------------------------------------------- #
DENVER_VERDICTS = os.path.join(REPO, "analysis_out", "review_denver-co",
                               "verdicts.json")


@pytest.mark.skipif(not os.path.exists(DENVER_VERDICTS),
                    reason="committed Denver verdicts not present")
def test_sites_from_the_committed_denver_verdicts():
    sites, source = srs.load_sites_from_verdicts(DENVER_VERDICTS)
    assert len(sites) == 59                     # the aerial sheet as built
    assert source["seed"] == 20260731
    assert source["mode"] == "verdicts"
    assert all(isinstance(s["id"], str) for s in sites)
    # ALL records travel — including aerial-unjudgeable ones; looking under
    # the canopy is #103's argument 3, so filtering them would defeat it.
    with open(DENVER_VERDICTS, encoding="utf-8") as fh:
        assert len(json.load(fh)["records"]) == len(sites)
