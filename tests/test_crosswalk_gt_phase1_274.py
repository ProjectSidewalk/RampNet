"""Crosswalk GT Phase 1 (#274): city config, seeded-draw determinism, Seattle page parity, page blindness.

No network, no imagery: the draw runs on a synthetic census cache written to tmp_path.
"""
import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1 = os.path.join(ROOT, "scripts", "analysis", "crosswalk_gt_phase1_274")
D0 = os.path.join(ROOT, "scripts", "analysis", "crosswalk_phase0_274")
sys.path.insert(0, D1)
sys.path.insert(0, D0)

import cities  # noqa: E402
import draw  # noqa: E402
import gt_page  # noqa: E402


def test_city_table_is_valid():
    assert cities.validate()
    assert set(cities.DRAW_CITIES) == {"amsterdam", "columbus-oh", "chicago-il", "sao-paulo-brazil"}
    assert [cities.CITIES[c]["n"] for c in cities.DRAW_CITIES] == [120, 120, 40, 40]
    for c in ("tucson-az", "madison-wi"):
        assert cities.CITIES[c]["status"] == "config_only"


def test_tile_urls_are_square_mercator_and_respect_min_extent():
    for cid, c in cities.CITIES.items():
        for src in c["imagery"]:
            u = cities.tile_url(src, 47.6, -122.3, ground_m=30, px=800)
            assert ("3857" in u) and ("800" in u), (cid, u)
    kc25 = cities.CITIES["seattle-wa"]["imagery"][1]
    # a 10 Mercator-metre request is widened to King County 2025's 36 m floor
    u = cities.tile_url(kc25, 47.6, -122.3, merc_m=10, px=720)
    x0, y0, x1, y1 = map(float, u.split("bbox=")[1].split("&")[0].split(","))
    assert abs((x1 - x0) - 36) < 1e-6 and abs((y1 - y0) - 36) < 1e-6
    # ground metres -> Mercator metres scales by 1 / cos(lat)
    u = cities.tile_url(cities.CITIES["amsterdam"]["imagery"][0], 60.0, 5.0, ground_m=30, px=800)
    x0, _, x1, _ = map(float, u.split("BBOX=")[1].split("&")[0].split(","))
    assert abs((x1 - x0) - 60) < 1e-6


def test_phase0_template_parity_for_seattle():
    """The city-generic Phase 0 template, filled from the Seattle row of the table, is the Phase 0 template."""
    got = gt_page.phase0_template("seattle-wa")
    want = open(os.path.join(D0, "aerial_spotcheck_template.html"), encoding="utf-8", newline="").read()
    assert got == want


def test_phase0_page_parity_when_inputs_are_present():
    """Full page parity (needs the census rawLabels cache, which is not in the repo; skipped without it)."""
    raw = os.environ.get("CROSSWALK_SEATTLE_RAWLABELS", "")
    if not os.path.exists(raw):
        pytest.skip("set CROSSWALK_SEATTLE_RAWLABELS to the census seattle-wa_Crosswalk.csv to run")
    import types
    import aerial_spotcheck as p0
    o = os.path.join(ROOT, "analysis_out", "crosswalk_phase0_274")
    ns = types.SimpleNamespace(sheet=os.path.join(o, "aerial_sample_seattle-wa.rated_ai.csv"), rawlabels=raw,
                               review=os.path.join(o, "aerial_spotcheck_review_rows.json"),
                               verdicts=os.path.join(o, "aerial_verdicts_seattle-wa.json"), archive=None,
                               out=os.path.join(o, "x.html"), server=cities.CITIES["seattle-wa"]["server"])
    html = p0.render(ns, tpl=gt_page.phase0_template("seattle-wa"))
    assert html == open(os.path.join(o, "aerial_spotcheck_seattle-wa.html"), encoding="utf-8", newline="").read()


def _fake_census(tmp_path, city):
    """A small census cache: 40 nodes on a line, labels near some of them, a few far away."""
    raw = tmp_path / "rawlabels"
    raw.mkdir()
    osm = tmp_path / "osm" / city / "Crosswalk"
    osm.mkdir(parents=True)
    els, labels = [], []
    lat0, lng0 = 52.0, 4.0
    agree = json.dumps([{"validator_type": "Human", "validation": "Agree"}] * 2)
    for k in range(40):
        lat = lat0 + k * 0.0006   # ~67 m apart
        tags = [{"crossing": "marked"}, {"crossing": "unmarked"}, {}][k % 3]
        els.append({"type": "node", "id": 1000 + k, "lat": lat, "lon": lng0, "tags": {"highway": "crossing", **tags}})
        if k % 4 != 3:   # three of four nodes carry a label ~3 m away; the rest are 'unlabeled'
            labels.append({"label_id": str(k + 1), "label_type": "Crosswalk", "latitude": str(lat + 0.00003),
                           "longitude": str(lng0), "validations": agree if k % 2 else "[]",
                           "pano_id": f"p{k}", "pano_source": "gsv", "image_capture_date": "2024-05",
                           "time_created": "2025-01-01T00:00:00Z", "pano_url": ""})
    (osm / "tile_0_0.json").write_text(json.dumps({"elements": els}))
    with open(raw / f"{city}_Crosswalk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0]))
        w.writeheader()
        w.writerows(labels)
    return str(tmp_path)


def test_draw_is_deterministic_and_stratified(tmp_path):
    census = _fake_census(tmp_path, "amsterdam")
    r1, m1 = draw.draw_city(census, "amsterdam", 16)
    r2, m2 = draw.draw_city(census, "amsterdam", 16)
    assert r1 == r2 and m1 == m2
    assert sum(m1["quota"].values()) == 16
    assert set(m1["quota"]) == set(draw.STRATA)
    # no node twice, and labeled strata carry a label, osm_only never does
    assert len({r["node_id"] for r in r1}) == len(r1)
    for r in r1:
        assert bool(r["label_uid"]) == (r["stratum"] != "osm_only")
        if r["label_uid"]:
            assert r["label_uid"] == f"amsterdam:{r['label_id']}"
            assert float(r["dist_m"]) <= draw.MATCH_M
        else:
            assert draw.NOLABEL_MIN_M < float(r["nearest_label_m"]) <= draw.AUDITED_MAX_M
    # confirmed labels are drawn before undecided ones within a stratum
    for s in ("marked", "untagged", "unmarked"):
        tiers = [r["label_tier"] for r in r1 if r["stratum"] == s]
        assert tiers == sorted(tiers)   # "confirmed" < "undecided"


def test_allocate_redistributes_shortfall():
    q = draw.allocate(120, {"marked": 900, "untagged": 11, "unmarked": 12, "osm_only": 300})
    assert q == {"marked": 49, "untagged": 11, "unmarked": 12, "osm_only": 48}
    assert draw.allocate(40, {s: 1000 for s in draw.STRATA}) == {s: 10 for s in draw.STRATA}
    assert sum(draw.allocate(40, {"marked": 3, "untagged": 0, "unmarked": 0, "osm_only": 5}).values()) == 8


def test_page_payload_is_blind(tmp_path):
    """The page rows carry no stratum, OSM class/tags, label tier or AI field."""
    sheet = [{"gt_id": "r000", "city": "amsterdam", "node_id": "1", "node_lat": "52.0", "node_lng": "4.0",
              "label_id": "7", "label_uid": "amsterdam:7", "label_lat": "52.00001", "label_lng": "4.0", "dist_m": "1.1",
              "image_capture_date": "2024-05", "time_created": "2025-01-01", "pano_url": "",
              "stratum": "unmarked", "osm_class": "unmarked", "crossing_tag": "unmarked", "label_tier": "confirmed"}]
    log = {"cities": {"amsterdam": {"accepted": {"1": {"tiles": {"2025": {"file": "ortho/amsterdam/1_2025.jpg"}},
                                                       "blank_years": {"2024": "blank: uniform"}, "crop": None}}}}}
    rows = gt_page.page_rows(sheet, log, str(tmp_path), str(tmp_path / "page" / "x.html"))
    assert tuple(rows[0]) == gt_page.ROW_KEYS
    text = json.dumps(rows)
    for leak in ("stratum", "unmarked", "confirmed", "osm_class", "crossing", "ai_", "painted"):
        assert leak not in text, leak
    tpl = open(os.path.join(D1, "gt_page_template.html"), encoding="utf-8").read()
    assert "ai_p" not in tpl and "AI said" not in tpl and "prerate" not in tpl
