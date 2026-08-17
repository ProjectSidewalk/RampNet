"""Is a curb-ramp inventory recorded per *ramp* or per *corner*?

The automatable half of the location-precision gate (issues #96, #59). See
``docs/curb_ramp_data_sourcing.md`` §5.

Positional offset — *is the coordinate on the ramp?* — needs aerial imagery and a
human, and this script does not attempt it. But one of §5's six checks is pure
geometry and can be settled from the point set alone:

    **Per-ramp vs per-corner.** If a city records one point per *corner* rather
    than per ramp, paired ramps collapse to a single label — **the exact
    supervision gap behind Paterson's failure** (#46: 72% of paterson's near-field
    misses are adjacent-pair merges).

A corner in a modern build carries **two** ramps, one per crossing direction, a
few metres apart. So the two recording conventions separate cleanly on the
**nearest-neighbour distance distribution**: a per-ramp inventory has a strong
mode at the within-corner spacing (a few metres), a per-corner inventory does not,
because its nearest neighbour is the *next corner* across a crosswalk (tens of
metres).

**Why this is calibrated rather than asserted.** A bare "68% of Denver's points
have a neighbour within 6 m" means nothing without knowing what a known-per-ramp
inventory scores. NYC publishes both ``rampid`` **and** ``cornerid``, so it is
ground truth for this question — it fixes the reference value *and* lets the
geometric corner-recovery be scored against the publisher's own grouping. Run NYC
with ``--corner-field cornerid`` first; every other city is read against it.

    python scripts/analysis/inventory_geometry.py \
        --city nyc --inventory data/inventories/nyc-ny-2026-07-31.jsonl.gz \
        --corner-field cornerid

    python scripts/analysis/inventory_geometry.py \
        --city denver-co --inventory data/inventories/denver-co-2026-07-31.jsonl.gz \
        --date-field CREATEDATE

Needs no GPU, no imagery and no network — it reads a snapshot written by
``fetch_inventory.py``. The core is pure and unit-tested in
``tests/test_inventory_geometry.py``; only ``load_inventory``, ``write_report``
and ``main`` touch disk.
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

# Two ramps on one corner are separated by the corner radius — a few metres. 6 m
# is wide enough to catch a standard pair and narrow enough to exclude the next
# corner across a crosswalk, which is a full roadway width away. Calibrated
# against NYC's published cornerid grouping; see ``score_corner_recovery``.
CORNER_LINK_M = 6.0

# Corner-to-corner across a typical urban intersection. Used only for the
# records-per-intersection figure that §5 quotes for NYC (~1.8), which is a
# coarser statistic than the corner test and is reported for continuity.
INTERSECTION_LINK_M = 30.0

# Below this, two records are the same physical ramp entered twice — a data
# defect, and one that would put two identical labels in the same panorama.
COINCIDENT_M = 0.5

# Link distances for ``link_sweep``. Stops at 14 m because a US residential
# roadway is ~9-12 m kerb to kerb: past that the link bridges the crossing and
# corner groups start merging into intersections, which the sweep reports as
# ``groups_per_intersection`` falling away from 4.
LINK_SWEEP_M = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0)

# Nearest-neighbour histogram edges, in metres. Dense below 10 m because that is
# where the per-ramp/per-corner signal lives.
NN_BINS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, float("inf")]

METRES_PER_DEG_LAT = 111320.0


def to_local_metres(points, lat0=None):
    """Project lon/lat to a local flat plane in metres.

    Equirectangular about the set's mean latitude. Good to well under a metre
    over a city, which is an order of magnitude finer than the distances that
    decide any question here, and it avoids a projection dependency.
    """
    if not points:
        return [], 0.0
    if lat0 is None:
        lat0 = sum(p[1] for p in points) / len(points)
    mx = METRES_PER_DEG_LAT * math.cos(math.radians(lat0))
    return [(p[0] * mx, p[1] * METRES_PER_DEG_LAT) for p in points], lat0


class GridIndex:
    """Uniform-grid spatial hash over points in metres.

    A k-d tree would be tidier but pulls in scipy; the point sets here are
    city-scale and roughly uniform along streets, so a grid at the query radius
    keeps every lookup to nine cells.
    """

    def __init__(self, xy, cell):
        self.cell = float(cell)
        self.xy = xy
        self.cells = defaultdict(list)
        for i, (x, y) in enumerate(xy):
            self.cells[(int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))].append(i)

    def neighbours(self, i):
        """Indices in the 3x3 cell block around point ``i``, excluding ``i``."""
        x, y = self.xy[i]
        cx, cy = int(math.floor(x / self.cell)), int(math.floor(y / self.cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in self.cells.get((cx + dx, cy + dy), ()):
                    if j != i:
                        yield j

    def within(self, i, radius):
        """Indices within ``radius`` of point ``i``. Requires ``cell >= radius``."""
        x, y = self.xy[i]
        r2 = radius * radius
        out = []
        for j in self.neighbours(i):
            jx, jy = self.xy[j]
            if (jx - x) ** 2 + (jy - y) ** 2 <= r2:
                out.append(j)
        return out


def nearest_neighbour_distances(xy, cell=None):
    """Distance from each point to its nearest other point, in metres.

    Returns ``None`` for a point with no neighbour inside the search block rather
    than silently reporting a wrong large value — a grid search is exact only
    within its 3x3 block, and an isolated point in a sparse suburb genuinely has
    no neighbour within reach. Callers report the censored count.
    """
    if len(xy) < 2:
        return [None] * len(xy)
    cell = cell or 50.0
    idx = GridIndex(xy, cell)
    out = []
    for i, (x, y) in enumerate(xy):
        best = None
        for j in idx.neighbours(i):
            jx, jy = xy[j]
            d2 = (jx - x) ** 2 + (jy - y) ** 2
            if best is None or d2 < best:
                best = d2
        out.append(None if best is None else math.sqrt(best))
    return out


def single_link_clusters(xy, link_m):
    """Group points by single-link connectivity at ``link_m``.

    Returns a list of clusters, each a list of point indices. Single-link is the
    right join for this: two ramps on a corner are near each other, and the
    question is only whether the inventory separates them at all.
    """
    n = len(xy)
    idx = GridIndex(xy, max(link_m, 1e-6))
    seen = [False] * n
    clusters = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, members = [start], [start]
        while stack:
            i = stack.pop()
            for j in idx.within(i, link_m):
                if not seen[j]:
                    seen[j] = True
                    members.append(j)
                    stack.append(j)
        clusters.append(members)
    return clusters


def histogram(values, edges):
    """Count values into ``[edges[k], edges[k+1])`` buckets. Nones are ignored."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        if v is None:
            continue
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1]:
                counts[k] += 1
                break
    return counts


def quantiles(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {str(q): None for q in qs}
    out = {}
    for q in qs:
        pos = q * (len(vals) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[str(q)] = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return out


def share_within(nn, radius):
    """Share of points whose nearest neighbour is within ``radius`` metres.

    **The headline statistic.** Censored points (no neighbour in the search
    block) count in the denominator — they are genuinely isolated, which is
    evidence against pairing, not missing data.
    """
    if not nn:
        return None
    return sum(1 for d in nn if d is not None and d <= radius) / float(len(nn))


def link_sweep(xy, links=LINK_SWEEP_M):
    """Records-per-group as the corner link distance grows.

    **The control that stops the headline being an artifact of NYC.** A single
    threshold calibrated on Manhattan is not obviously transferable: NYC's corner
    radii are tight (its within-corner mode sits at 2-3 m), and a city built to
    suburban geometry would space the *same* pair of ramps further apart, so a
    6 m link would score it as per-corner purely for being wide.

    Sweeping separates the two readings. If a city's records-per-group climbs
    toward NYC's ~1.6 as the link widens — and does so *before* the link reaches
    the roadway width where groups start merging across the intersection — the
    pairs are there and only the spacing differs. If it stays flat and then jumps
    straight to intersection-scale merging, the pairs are genuinely absent.

    ``merge_ratio`` is the guard for that second clause: groups per intersection
    cluster. It starts near 4 (four corners) and collapses toward 1 once the link
    is wide enough to bridge crossings, which is the point past which the
    records-per-group figure means nothing.
    """
    inter = single_link_clusters(xy, INTERSECTION_LINK_M)
    n_inter = len(inter) or 1
    out = []
    for link in links:
        groups = single_link_clusters(xy, link)
        sizes = Counter(len(g) for g in groups)
        out.append({
            "link_m": link,
            "groups": len(groups),
            "records_per_group": len(xy) / float(len(groups)) if groups else None,
            "share_singleton": sizes[1] / float(len(groups)) if groups else None,
            "groups_per_intersection": len(groups) / float(n_inter),
        })
    return out


def score_corner_recovery(xy, corner_ids, link_m=CORNER_LINK_M):
    """How well does geometric clustering reproduce a published corner grouping?

    Only NYC can answer this, and the answer is what licenses running the same
    clustering on cities that publish no corner key. Reported as pair-counting
    precision/recall over co-membership: of the point pairs the clustering puts
    together, how many share a ``cornerid``, and vice versa.

    Pairs are counted within clusters and within corner groups only — never
    across the whole set — so this stays linear in the number of *pairs that
    either side proposes*, not quadratic in the city.
    """
    clusters = single_link_clusters(xy, link_m)
    by_corner = defaultdict(list)
    for i, cid in enumerate(corner_ids):
        if cid is not None:
            by_corner[cid].append(i)

    def pairs(groups):
        out = set()
        for g in groups:
            g = sorted(g)
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    out.add((g[a], g[b]))
        return out

    pred = pairs(clusters)
    true = pairs(by_corner.values())
    tp = len(pred & true)
    return {
        "link_m": link_m,
        "cluster_pairs": len(pred),
        "corner_pairs": len(true),
        "shared_pairs": tp,
        "precision": (tp / len(pred)) if pred else None,
        "recall": (tp / len(true)) if true else None,
        "published_groups": len(by_corner),
        "geometric_groups": len(clusters),
    }


def epoch_ms_to_ym(ms):
    """ArcGIS dates are epoch milliseconds. Returns ``(year, month)`` or None."""
    if ms is None:
        return None
    try:
        secs = float(ms) / 1000.0
    except (TypeError, ValueError):
        return None
    # Pure arithmetic rather than datetime, so the core stays timezone-free:
    # these are calendar stamps, not instants, and a UTC/local slip would move a
    # record across a month boundary.
    days = int(math.floor(secs / 86400.0))
    y = 1970
    while True:
        leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        n = 366 if leap else 365
        if days < n:
            break
        days -= n
        y += 1
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    m = 1
    for ln in lengths:
        if days < ln:
            break
        days -= ln
        m += 1
    return (y, m)


def bbox(points):
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return {"lon_min": min(lons), "lon_max": max(lons),
            "lat_min": min(lats), "lat_max": max(lats)}


def analyse(points, corner_ids=None, dates=None, corner_link_m=CORNER_LINK_M,
            intersection_link_m=INTERSECTION_LINK_M):
    """Full geometric report for one inventory. Pure."""
    xy, lat0 = to_local_metres(points)
    nn = nearest_neighbour_distances(xy)
    censored = sum(1 for d in nn if d is None)

    corner_clusters = single_link_clusters(xy, corner_link_m)
    corner_sizes = Counter(len(c) for c in corner_clusters)
    inter_clusters = single_link_clusters(xy, intersection_link_m)

    report = {
        "records": len(points),
        "bbox": bbox(points),
        "mean_latitude": lat0,
        "nearest_neighbour": {
            "censored": censored,
            "censored_note": "no neighbour within the 50 m grid block; counted in "
                             "the denominator of every share below",
            "quantiles_m": quantiles(nn),
            "histogram_edges_m": NN_BINS[:-1] + ["inf"],
            "histogram": histogram(nn, NN_BINS),
            "share_within_0.5m": share_within(nn, COINCIDENT_M),
            "share_within_2m": share_within(nn, 2.0),
            "share_within_6m": share_within(nn, corner_link_m),
            "share_within_10m": share_within(nn, 10.0),
        },
        "corner_clusters": {
            "link_m": corner_link_m,
            "groups": len(corner_clusters),
            "records_per_group": len(points) / float(len(corner_clusters)) if corner_clusters else None,
            "size_histogram": {str(k): corner_sizes[k] for k in sorted(corner_sizes)},
            "share_singleton": corner_sizes[1] / float(len(corner_clusters)) if corner_clusters else None,
        },
        "intersection_clusters": {
            "link_m": intersection_link_m,
            "groups": len(inter_clusters),
            "records_per_group": len(points) / float(len(inter_clusters)) if inter_clusters else None,
        },
        "coincident": {
            "threshold_m": COINCIDENT_M,
            "records": sum(1 for d in nn if d is not None and d <= COINCIDENT_M),
        },
        "link_sweep": link_sweep(xy),
    }
    if corner_ids is not None:
        report["corner_recovery"] = score_corner_recovery(xy, corner_ids, corner_link_m)
        by_corner = Counter(c for c in corner_ids if c is not None)
        sizes = Counter(by_corner.values())
        report["published_corners"] = {
            "groups": len(by_corner),
            "records_per_group": sum(by_corner.values()) / float(len(by_corner)) if by_corner else None,
            "size_histogram": {str(k): sizes[k] for k in sorted(sizes)},
            "missing_corner_id": sum(1 for c in corner_ids if c is None),
        }
    if dates is not None:
        yms = [epoch_ms_to_ym(d) for d in dates]
        years = Counter(ym[0] for ym in yms if ym is not None)
        report["dates"] = {
            "undated": sum(1 for ym in yms if ym is None),
            "by_year": {str(y): years[y] for y in sorted(years)},
        }
    return report


def composite_key(rows, fields):
    """Build a corner key from one or more columns. Pure.

    Only NYC publishes a single ready-made corner id. Charlotte and Minneapolis
    publish the same information split in two — an intersection id plus which
    corner of it (`RP_LocInInt` = NE/SE/…, `quadrant`) — so the key has to be
    composed.

    A record missing *any* part yields ``None``. The alternative, substituting an
    empty string, would collapse every incomplete record into one enormous
    pseudo-corner and wreck the recovery score in a direction that looks like
    over-merging.
    """
    out = []
    for r in rows:
        parts = [r.get(f) for f in fields]
        if any(p is None or p == "" for p in parts):
            out.append(None)
        else:
            out.append("|".join(str(p) for p in parts))
    return out


def load_inventory(path, lon_field="lon", lat_field="lat"):
    """Read a snapshot written by ``fetch_inventory.py``."""
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    points = [(r[lon_field], r[lat_field]) for r in rows]
    return rows, points


def write_report(city, report, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "inventory_geometry_{}.json".format(city))
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--inventory", required=True, help="jsonl(.gz) from fetch_inventory.py")
    ap.add_argument("--corner-field", default=None,
                    help="published corner key, if the city has one. Comma-separate to "
                         "compose one from several columns: NYC 'cornerid'; Charlotte "
                         "'RP_IntID,RP_LocInInt' (intersection + which corner of it); "
                         "Minneapolis 'intersection_id,quadrant'. A record missing any "
                         "part is treated as having no corner key rather than being "
                         "silently grouped with every other incomplete record.")
    ap.add_argument("--date-field", default=None,
                    help="epoch-ms date field to histogram by year (Denver: CREATEDATE)")
    ap.add_argument("--corner-link-m", type=float, default=CORNER_LINK_M)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args(argv)

    rows, points = load_inventory(args.inventory)
    corner_ids = composite_key(rows, args.corner_field.split(",")) if args.corner_field else None
    dates = [r.get(args.date_field) for r in rows] if args.date_field else None
    report = analyse(points, corner_ids=corner_ids, dates=dates,
                     corner_link_m=args.corner_link_m)
    report["city"] = args.city
    report["inventory"] = os.path.basename(args.inventory)

    nnr = report["nearest_neighbour"]
    print("{}: {} records".format(args.city, report["records"]))
    print("  nearest neighbour median {:.1f} m".format(nnr["quantiles_m"]["0.5"]))
    print("  share within {:.0f} m: {:.3f}   <-- per-ramp signal".format(
        args.corner_link_m, nnr["share_within_6m"]))
    print("  corner clusters: {} groups, {:.2f} records/group, {:.3f} singleton".format(
        report["corner_clusters"]["groups"],
        report["corner_clusters"]["records_per_group"],
        report["corner_clusters"]["share_singleton"]))
    print("  link sweep (link_m: rec/group, groups/intersection)")
    for row in report["link_sweep"]:
        print("    {:>5.0f} m: {:.3f}   {:.2f}".format(
            row["link_m"], row["records_per_group"], row["groups_per_intersection"]))
    if "corner_recovery" in report:
        cr = report["corner_recovery"]
        print("  corner recovery vs published: P {:.3f} / R {:.3f} "
              "({} geometric vs {} published groups)".format(
                  cr["precision"], cr["recall"],
                  cr["geometric_groups"], cr["published_groups"]))
    print("wrote {}".format(write_report(args.city, report, out_dir=args.out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
