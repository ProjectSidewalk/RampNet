"""Tests for the GSV pano probe (#103) — the pure aggregation and the
record-date plumbing.

The probe's whole justification is §5h's rule made structural: check the
fetcher's claims against the sample, per site, with reasons. So the tests pin
that the denominators are what the docstrings say, that failures stay listed
individually beside the aggregate, and that record dates go through the ONE
parser (sentinels count as undated — Boston's 18991230 must not become a
temporal constraint from the year 1899).
"""
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import probe_panos_at_sites as probe  # noqa: E402


# --------------------------------------------------------------------------- #
# record dates through the one parser
# --------------------------------------------------------------------------- #
def test_record_ym_parses_epoch_ms_and_rejects_sentinels():
    row = {"CREATEDATE": 1420070400000}          # 2015-01-01 in ArcGIS epoch ms
    assert probe.record_ym_of(row, "CREATEDATE") == (2015, 1)
    # Sentinels are undated, not ancient: the 2000-01 placeholder (#11) and the
    # OLE zero date Boston publishes as 18991230.
    assert probe.record_ym_of({"D": "2000-01"}, "D") is None
    assert probe.record_ym_of({"D": "18991230"}, "D") is None
    assert probe.record_ym_of({"D": None}, "D") is None
    assert probe.record_ym_of(row, None) is None  # no field configured


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _site(sid, status="ok", n_panos=5, rng=11.0, date="2021-3",
          record_ym=(2015, 1), dated_after=3, **kw):
    r = {"id": sid, "stratum": None, "status": status, "n_panos": n_panos,
         "n_in_band": 3, "n_eligible": 2,
         "record_ym": None if record_ym is None else list(record_ym),
         "n_dated_after_record": dated_after}
    if status == "ok":
        r.update(chosen_pano="P" + sid, chosen_date=date, chosen_range_m=rng)
    r.update(kw)
    return r


def test_summarise_rates_have_the_stated_denominators():
    results = [
        _site("1", rng=8.0, date="2019-7"),
        _site("2", rng=14.0, date="2021-3"),
        _site("3", status="no_pano_in_band", dated_after=1),
        _site("4", status="no_panos", n_panos=0, dated_after=0),
        _site("5", status="search_failed", n_panos=0, dated_after=0,
              detail="GoogleEndpointSchemaError: drift"),
    ]
    s = probe.summarise(results)
    assert s["n_sites"] == 5
    assert s["pick_rate"] == pytest.approx(2 / 5)
    # coverage = any pano at all, regardless of band/date
    assert s["coverage"] == pytest.approx(3 / 5)
    # date coverage = a pano postdating the record exists (or record undated)
    assert s["date_coverage"] == pytest.approx(3 / 5)
    assert s["chosen_range_m"] == {"min": 8.0, "median": 11.0, "max": 14.0}
    assert s["chosen_year_hist"] == {"2019": 1, "2021": 1}
    # failures stay individually listed beside the aggregate — a drop count is
    # a claim about the fetcher until it can be audited against the sample
    assert [f["id"] for f in s["failures"]] == ["3", "4", "5"]
    assert s["status_counts"] == {"ok": 2, "no_pano_in_band": 1,
                                  "no_panos": 1, "search_failed": 1}


def test_summarise_undated_record_counts_as_date_covered():
    s = probe.summarise([_site("1", record_ym=None, dated_after=0)])
    assert s["date_coverage"] == 1.0


def test_summarise_empty():
    s = probe.summarise([])
    assert s["n_sites"] == 0 and s["coverage"] is None
    assert s["chosen_range_m"] is None and s["failures"] == []


def test_summarise_even_median():
    results = [_site("1", rng=8.0), _site("2", rng=12.0)]
    assert probe.summarise(results)["chosen_range_m"]["median"] == 10.0


# --------------------------------------------------------------------------- #
# probe_site through a stubbed search
# --------------------------------------------------------------------------- #
def test_probe_site_runs_the_sheets_own_pick_rule(tmp_path, monkeypatch):
    fake = types.ModuleType("search_panos")

    class _P:
        def __init__(self, pid, lat, lon, date):
            self.pano_id, self.lat, self.lon, self.date = pid, lat, lon, date
            self.heading = 100.0

    def search_panoramas(lat, lon):
        e = 10.0 / (111320.0 * 0.77)
        return [_P("old", lat, lon + e, "2014-6"), _P("new", lat, lon + e, "2016-2")]

    fake.search_panoramas = search_panoramas
    monkeypatch.setitem(sys.modules, "search_panos", fake)

    site = {"id": "42", "lat": 39.74, "lon": -104.99, "stratum": "dated_before"}
    row = {"OBJECTID": 42, "CREATEDATE": 1420070400000}   # (2015, 1)
    r = probe.probe_site(site, row, "CREATEDATE", (4.0, 30.0),
                         str(tmp_path), sleep_s=0.0)
    assert r["status"] == "ok"
    assert r["chosen_pano"] == "new"       # 2014 pano ineligible for a 2015 record
    assert r["record_ym"] == [2015, 1]
    assert r["n_dated_after_record"] == 1
    # ...and the search landed in the cache the sheet build will reuse.
    assert any(f.startswith("search_") for f in os.listdir(str(tmp_path)))


def test_probe_site_records_a_search_failure_with_its_reason(tmp_path, monkeypatch):
    fake = types.ModuleType("search_panos")

    def search_panoramas(lat, lon):
        raise RuntimeError("endpoint drift")

    fake.search_panoramas = search_panoramas
    monkeypatch.setitem(sys.modules, "search_panos", fake)

    site = {"id": "42", "lat": 39.74, "lon": -104.99}
    r = probe.probe_site(site, {}, None, (4.0, 30.0), str(tmp_path), sleep_s=0.0)
    assert r["status"] == "search_failed"
    assert "endpoint drift" in r["detail"]
    assert os.listdir(str(tmp_path)) == []     # a failure is never cached
