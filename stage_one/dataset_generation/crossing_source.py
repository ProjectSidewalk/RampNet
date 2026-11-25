"""OSM crossing nodes as a Stage 1 location source (RampNet#274 Phase 1 item 3).

Stdlib only. Turns an Overpass JSON answer for `node["highway"="crossing"]` into Stage 1 location rows, the
shape `combine_location_data.py` writes for curb ramps (`latitude, longitude, date`), plus the columns a
second label type needs (`label_type`, `source_id`, `osm_class`). It does NOT wire those rows into
`generate_dataset_meta.py` / `download_dataset.py`: where `label_type` lives in the dataset schema is an
open design fork, laid out in `docs/stage1_label_type_274.md`, and this module is the part that does not
depend on how it is resolved.

The positive rule is Phase 0's Seattle reading (docs/crosswalk_phase0_274.md; marked 40/40 painted,
untagged-with-a-crowd-label 39/40, `unmarked` 28/40):

  marked     positive
  untagged   positive only when a crowd Crosswalk label lies within `match_m` (default 10 m) of the node
  unmarked   excluded: neither a positive (12 of 40 were not painted) nor a negative (28 of 40 were)

Both sets are arguments, so the Phase 1 multi-city GT (docs/crosswalk_gt_phase1_274.md) can change the rule
per city without touching the reader.

`date` is always "": OSM nodes carry no install date, so `generate_dataset_meta.py`'s
`TREAT_UNDATED_AS_PREDATING` decides, as it does for undated curb ramps. Paint is repainted and removed in
a way a ramp is not; the doc lists that as a known gap.

Usage example::

    >>> els = [{"type": "node", "id": 1, "lat": 47.6, "lon": -122.3, "tags": {"highway": "crossing", "crossing": "marked"}},
    ...        {"type": "node", "id": 2, "lat": 47.7, "lon": -122.3, "tags": {"highway": "crossing", "crossing": "unmarked"}}]
    >>> [r["source_id"] for r in crossing_locations(els)]
    ['osm:node/1']
"""
import math

LABEL_TYPE = "Crosswalk"
# The same two sets the Phase 0 census used (scripts/analysis/crosswalk_phase0_274/crossing_census.py).
MARKED_VALUES = {"marked", "zebra", "lines", "ladder", "dashes", "dots", "surface", "pedestrian_signals"}
UNMARKED_VALUES = {"unmarked", "no", "informal"}
FIELDS = ["latitude", "longitude", "date", "label_type", "source_id", "osm_class"]


def osm_class(tags):
    """'marked' | 'unmarked' | 'untagged', by `crossing:markings` first, then `crossing` (Phase 0 `is_marked`)."""
    m = tags.get("crossing:markings")
    if m is not None:
        return "unmarked" if m == "no" else "marked"
    c = tags.get("crossing")
    if c in MARKED_VALUES:
        return "marked"
    if c in UNMARKED_VALUES:
        return "unmarked"
    return "untagged"


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class _Grid:
    """Bucketed lat/lng points for a within-radius test (cells of ~0.001 deg, about 110 m)."""

    def __init__(self, pts, cell=0.001):
        self.cell, self.b = cell, {}
        for lat, lng in pts:
            self.b.setdefault((int(lat // cell), int(lng // cell)), []).append((lat, lng))

    def any_within(self, lat, lng, radius_m):
        ci, cj = int(lat // self.cell), int(lng // self.cell)
        k = int(radius_m / (self.cell * 111000 * 0.5)) + 1
        for i in range(ci - k, ci + k + 1):
            for j in range(cj - k, cj + k + 1):
                for p in self.b.get((i, j), ()):
                    if haversine_m(lat, lng, p[0], p[1]) <= radius_m:
                        return True
        return False


def crossing_locations(elements, crowd_points=None, match_m=10.0,
                       positive=("marked",), positive_with_crowd=("untagged",)):
    """Stage 1 location rows for the positive OSM crossing nodes in an Overpass `elements` list.

    `crowd_points`: (lat, lng) of crowd Crosswalk labels (e.g. human-confirmed ones from v3 rawLabels); a node
    of a class in `positive_with_crowd` is kept only when one lies within `match_m`. Without crowd points
    those classes are dropped, never kept. Rows come out sorted by node id, so the output is deterministic.
    """
    grid = _Grid(crowd_points or [])
    rows = []
    nodes = sorted((el for el in elements if el.get("type") == "node"
                    and (el.get("tags") or {}).get("highway") == "crossing"), key=lambda el: el["id"])
    seen = set()
    for el in nodes:
        if el["id"] in seen:
            continue
        seen.add(el["id"])
        cls = osm_class(el.get("tags") or {})
        keep = cls in positive or (cls in positive_with_crowd and crowd_points
                                   and grid.any_within(el["lat"], el["lon"], match_m))
        if keep:
            rows.append({"latitude": el["lat"], "longitude": el["lon"], "date": "", "label_type": LABEL_TYPE,
                         "source_id": f"osm:node/{el['id']}", "osm_class": cls})
    return rows
