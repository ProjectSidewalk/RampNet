"""City-config table for the multi-city crosswalk GT round (RampNet#274 Phase 1 item 2).

One entry per city: its Project Sidewalk server, its orthoimagery sources (one per year, each with service
type, URL, layer, GSD, the minimum export extent the service fills, and the licence / attribution string),
whether its terms are confirmed, and its draw size. Everything that used to be hard-coded for Seattle in
`scripts/analysis/crosswalk_phase0_274/aerial_spotcheck_template.html` (title, provider line, service URL,
years, `MIN_TILE_M`, the per-panel credit, the crop server) is read from here, and
`gt_page.py build-phase0 --city seattle-wa` reproduces the Phase 0 page byte for byte from this table (the
parity check; see `tests/test_crosswalk_gt_phase1_274.py`).

Service kinds, all requested in Web Mercator (EPSG:3857) with a square bbox centred on the OSM node:

  arcgis_export   ArcGIS MapServer `export`          (King County)
  arcgis_image    ArcGIS ImageServer `exportImage`   (OGRIP Columbus, Cook County)
  wms13           OGC WMS 1.3.0 `GetMap`, CRS=EPSG:3857 (PDOK Amsterdam, GeoSampa Sao Paulo; GeoSampa's 2020
                  layers advertise EPSG:31983 only, but GeoServer reprojects a 3857 request; probed 2026-11-02)

Usage example::

    >>> from cities import CITIES, tile_url
    >>> src = CITIES["amsterdam"]["imagery"][0]
    >>> tile_url(src, 52.3731, 4.8926, ground_m=30, px=800)[:60]
    'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?SERVICE=WM'

Probe log (one request per service before any bulk fetch, 2026-11-02; 720 px, 36 Mercator m): PDOK
2025_orthoHR 200 image/jpeg 51 KB (Dam square, people visible); OGRIP osip_most_current 200 image/png 579 KB;
Cook County CookOrtho2025 200 image/png 309 KB (lane lines visible, soft at 15 cm); GeoSampa ORTO_RGB_2020
200 image/png 106 KB via Python (curl on Windows fails the TLS check because its store lacks the Sectigo
R46 root; Python's verified TLS works, so nothing is fetched with verification off).
"""
import math

R = 6378137.0
UA = "ProjectSidewalk-RampNet crosswalk-gt-phase1 (research; contact jonf@cs.uw.edu)"

KC = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_{y}/MapServer/export"
PDOK = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
OGRIP = "https://maps.ohio.gov/image/rest/services/osip_most_current/ImageServer/exportImage"
COOK = "https://gis.cookcountyil.gov/imagery/rest/services/CookOrtho{y}/ImageServer/exportImage"
GEOSAMPA = "https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms"
TUCSON = "https://images.tucsonaz.gov/images/rest/services/AerialImagery/ORTHO_2022_New/ImageServer/exportImage"
DANE = "https://dcimapapps.danecounty.gov/arcgisimg/rest/services/ColorOrtho3Inch{y}Web/ImageServer/exportImage"

PDOK_CREDIT = "Beeldmateriaal Nederland / PDOK, CC BY 4.0"
GEOSAMPA_CREDIT = "PMSP / GeoSampa, CC BY-SA 4.0"

# status: "draw" = sampled and rated in this round; "parity" = Phase 0 city kept for the parity check;
# "config_only" = an entry so a later round can draw it, terms not yet confirmed, nothing fetched.
CITIES = {
    "seattle-wa": {
        "name": "Seattle", "server": "https://sidewalk-sea.cs.washington.edu", "status": "parity",
        "provider_line": "King County ortho, 3 in/px", "service_name": "King County",
        "terms": "view only (EagleView licence to King County); never redistributed",
        "imagery": [
            {"year": 2023, "kind": "arcgis_export", "url": KC.format(y=2023), "gsd_cm": 7.5, "min_tile_m": 0,
             "credit": "King County GIS 2023, view only"},
            # KingCo_Aerial_2025 returns a blank 632-byte PNG at <= 24 Mercator m (720 px); imagery from 28 m.
            {"year": 2025, "kind": "arcgis_export", "url": KC.format(y=2025), "gsd_cm": 7.5, "min_tile_m": 36,
             "credit": "King County GIS 2025, view only"},
        ],
    },
    "amsterdam": {
        "name": "Amsterdam", "server": "https://sidewalk-amsterdam.cs.washington.edu", "status": "draw", "n": 120,
        "provider_line": "PDOK Luchtfoto HR, 8 cm/px",
        "terms": "CC BY 4.0 (PDOK: free for all uses under CC BY); attribution to Beeldmateriaal Nederland",
        "imagery": [
            {"year": 2024, "kind": "wms13", "url": PDOK, "layer": "2024_orthoHR", "format": "image/jpeg",
             "gsd_cm": 8, "min_tile_m": 0, "credit": PDOK_CREDIT + " (2024)"},
            {"year": 2025, "kind": "wms13", "url": PDOK, "layer": "2025_orthoHR", "format": "image/jpeg",
             "gsd_cm": 8, "min_tile_m": 0, "credit": PDOK_CREDIT + " (2025)"},
        ],
    },
    "columbus-oh": {
        "name": "Columbus", "server": "https://sidewalk-columbus.cs.washington.edu", "status": "draw", "n": 120,
        "provider_line": "OGRIP OSIP (COLUMBUS_2025), 7.6 cm/px",
        "terms": "public use at the user's own risk (OGRIP); no named licence; imagery kept out of the repo",
        "imagery": [
            {"year": 2025, "kind": "arcgis_image", "url": OGRIP, "gsd_cm": 7.62, "min_tile_m": 0,
             "credit": "OGRIP / Ohio DAS, COLUMBUS_2025"},
        ],
    },
    "chicago-il": {
        "name": "Chicago", "server": "https://sidewalk-chicago.cs.washington.edu", "status": "draw", "n": 40,
        "provider_line": "Cook County ortho, 15 cm/px (presence test)",
        "terms": "Cook County: data provided 'as is'; no named licence; imagery kept out of the repo",
        "imagery": [
            {"year": 2024, "kind": "arcgis_image", "url": COOK.format(y=2024), "gsd_cm": 15.24, "min_tile_m": 0,
             "credit": "Cook County GIS, CookOrtho2024"},
            {"year": 2025, "kind": "arcgis_image", "url": COOK.format(y=2025), "gsd_cm": 15.24, "min_tile_m": 0,
             "credit": "Cook County GIS, CookOrtho2025"},
        ],
    },
    "sao-paulo-brazil": {
        "name": "São Paulo", "server": "https://sidewalk-sao-paulo.cs.washington.edu", "status": "draw", "n": 40,
        "provider_line": "GeoSampa Ortofotos, 10 cm/px (2020; 2017 at 12 cm)",
        "terms": "CC BY-SA 4.0 (share-alike applies to derived rasters, so none are committed)",
        "imagery": [
            {"year": 2017, "kind": "wms13", "url": GEOSAMPA, "layer": "Orto_PMD_RGB_2017", "format": "image/png",
             "gsd_cm": 12, "min_tile_m": 0, "credit": GEOSAMPA_CREDIT + " (2017)"},
            {"year": 2020, "kind": "wms13", "url": GEOSAMPA, "layer": "ORTO_RGB_2020", "format": "image/png",
             "gsd_cm": 10, "min_tile_m": 0, "credit": GEOSAMPA_CREDIT + " (2020)"},
        ],
    },
    # Config only: the survey found open services but no licence text (Tucson: every licence field empty;
    # Dane County: disclaimer only). Confirm terms before drawing.
    "tucson-az": {
        "name": "Tucson", "server": "https://sidewalk-tucson.cs.washington.edu", "status": "config_only",
        "provider_line": "City of Tucson ORTHO_2022_New, 5.1 cm/px",
        "terms": "UNCONFIRMED: licence fields empty; ask GIS_IT@tucsonaz.gov",
        "imagery": [
            {"year": 2022, "kind": "arcgis_image", "url": TUCSON, "gsd_cm": 5.08, "min_tile_m": 0,
             "credit": "City of Tucson, ORTHO_2022"},
        ],
    },
    "madison-wi": {
        "name": "Madison", "server": "https://sidewalk-madison.cs.washington.edu", "status": "config_only",
        "provider_line": "Dane County LIO ColorOrtho3Inch, 7.6 cm/px",
        "terms": "UNCONFIRMED: disclaimer and credit line only",
        "imagery": [
            {"year": 2024, "kind": "arcgis_image", "url": DANE.format(y=2024), "gsd_cm": 7.62, "min_tile_m": 0,
             "credit": "Dane County LIO / Fly Dane, 2024"},
            {"year": 2026, "kind": "arcgis_image", "url": DANE.format(y=2026), "gsd_cm": 7.25, "min_tile_m": 0,
             "credit": "Dane County LIO / Fly Dane, 2026"},
        ],
    },
}

DRAW_CITIES = [c for c, v in CITIES.items() if v["status"] == "draw"]


def merc(lat, lng):
    """Web Mercator (EPSG:3857) metres."""
    return R * lng * math.pi / 180, R * math.log(math.tan(math.pi / 4 + lat * math.pi / 360))


def merc_per_ground(lat):
    """Web Mercator metres per ground metre at a latitude (1 / cos(lat))."""
    return 1 / math.cos(math.radians(lat))


def tile_url(src, lat, lng, ground_m=None, px=800, merc_m=None):
    """URL of a square tile centred on (lat, lng), `ground_m` GROUND metres wide (or `merc_m` Mercator metres),
    `px` pixels square, for one imagery source of the table. A source's `min_tile_m` (Mercator metres) widens a
    narrower request to the extent the service actually fills (the caller then scales the image up)."""
    if merc_m is None:
        merc_m = ground_m * merc_per_ground(lat)
    merc_m = max(merc_m, src.get("min_tile_m") or 0)
    cx, cy = merc(lat, lng)
    h = merc_m / 2
    bbox = f"{cx - h},{cy - h},{cx + h},{cy + h}"
    if src["kind"] == "arcgis_export":
        return f"{src['url']}?bbox={bbox}&bboxSR=3857&imageSR=3857&size={px},{px}&format=png&f=image"
    if src["kind"] == "arcgis_image":
        return f"{src['url']}?bbox={bbox}&bboxSR=3857&imageSR=3857&size={px},{px}&format=png&f=image"
    if src["kind"] == "wms13":
        return (f"{src['url']}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={src['layer']}&STYLES="
                f"&CRS=EPSG:3857&BBOX={bbox}&WIDTH={px}&HEIGHT={px}&FORMAT={src['format']}")
    raise ValueError(f"unknown service kind {src['kind']!r}")


def validate(cities=CITIES):
    """Raise AssertionError on a malformed entry (the pytest runs this)."""
    for cid, c in cities.items():
        assert c["status"] in ("draw", "parity", "config_only"), cid
        assert c["server"].startswith("https://"), cid
        assert c["imagery"], cid
        years = [s["year"] for s in c["imagery"]]
        assert years == sorted(set(years)), f"{cid}: years must be unique and ascending"
        for s in c["imagery"]:
            assert s["kind"] in ("arcgis_export", "arcgis_image", "wms13"), cid
            assert s["credit"] and s["gsd_cm"] > 0, cid
            if s["kind"] == "wms13":
                assert s.get("layer") and s.get("format", "").startswith("image/"), cid
        if c["status"] == "draw":
            assert c.get("n", 0) > 0 and c.get("terms"), cid
    return True
