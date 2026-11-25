"""Seattle SDOT marked crosswalks as a Stage 1 location source (RampNet#274).

Stdlib only; the crosswalk twin of `crossing_source.py`. Turns features of SDOT's "Marked Crosswalks" layer
(ArcGIS FeatureServer, pulled as GeoJSON by `scripts/analysis/crosswalk_inventory_274/sdot_inventory.py fetch`)
into Stage 1 location rows: the three columns `combine_location_data.py` writes for curb ramps
(`latitude, longitude, date`) plus `label_type`, `source_id`, `marking`, `midblock`.

Why SDOT and not OSM: under the amended rule in `sdot_inventory.py` (coverage mask, 2026-11-03) the layer agrees
with the Phase 0 human verdicts at sensitivity 0.961 inside its coverage, and it is the city's own record, the
role the government curb-ramp inventories played for the paper's Stage 1.

`date` is SDOT's `INSTALL_DATE` as `YYYY-MM` when present, which lets `generate_dataset_meta.py`'s
install-before-capture rule work. Only ~6% of features carry one (397 of 6,689 on the 2026-11-03 pull); the
rest are "" and `TREAT_UNDATED_AS_PREDATING` decides, as for undated ramps.

The status filter is the layer's own recommended one, CURRENT_STATUS in (INSVC, PLNRECON) or empty.

Usage example::

    >>> f = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.3, 47.6]},
    ...       "properties": {"OBJECTID": 7, "CURRENT_STATUS": "INSVC", "MARKING_TYPE": "LADER",
    ...                      "MIDBLOCK_CROSSWALK": "N", "INSTALL_DATE": 1421798400000}},
    ...      {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.3, 47.7]},
    ...       "properties": {"OBJECTID": 8, "CURRENT_STATUS": "PLANNED"}}]
    >>> [(r["source_id"], r["date"], r["marking"]) for r in sdot_locations(f)]
    [('sdot:7', '2015-01', 'LADER')]
"""
import datetime

LABEL_TYPE = "Crosswalk"
CURRENT = {"INSVC", "PLNRECON", ""}
FIELDS = ["latitude", "longitude", "date", "label_type", "source_id", "marking", "midblock"]


def install_month(ms):
    """ArcGIS epoch milliseconds -> 'YYYY-MM' (UTC), or '' when missing.

    >>> install_month(1421798400000), install_month(None)
    ('2015-01', '')
    """
    if ms in (None, ""):
        return ""
    # epoch + timedelta, not fromtimestamp: some SDOT dates predate 1970 and Windows refuses negative stamps
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return (epoch + datetime.timedelta(milliseconds=int(ms))).strftime("%Y-%m")


def sdot_locations(features, keep=None, statuses=CURRENT):
    """Stage 1 location rows for the current SDOT crosswalks in `features` (GeoJSON features).

    `keep(lat, lng) -> bool` restricts the area (e.g. the Project Sidewalk Seattle regions). Features without a
    point geometry are dropped. Rows come out sorted by OBJECTID, so the output is deterministic.
    """
    rows = []
    for f in sorted(features, key=lambda f: f["properties"]["OBJECTID"]):
        g, p = f.get("geometry"), f["properties"]
        if not g or not g.get("coordinates"):
            continue
        if (p.get("CURRENT_STATUS") or "").strip() not in statuses:
            continue
        lng, lat = g["coordinates"][:2]
        if keep is not None and not keep(lat, lng):
            continue
        rows.append({"latitude": lat, "longitude": lng, "date": install_month(p.get("INSTALL_DATE")),
                     "label_type": LABEL_TYPE, "source_id": f"sdot:{p['OBJECTID']}",
                     "marking": (p.get("MARKING_TYPE") or "").strip(),
                     "midblock": (p.get("MIDBLOCK_CROSSWALK") or "").strip()})
    return rows
