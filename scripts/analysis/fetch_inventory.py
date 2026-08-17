"""Fetch a government curb-ramp point inventory and pin it as a dated snapshot.

Sourcing gate for a larger Stage 1 corpus (issues #59, #96). See
``docs/curb_ramp_data_sourcing.md`` §9 for why the snapshot matters:

    The source inventories are **not in this repo**. ``location_data/`` and
    ``street_data/`` are neither present nor tracked — the README tells you to
    download them from live portal links, which serve *current* data. Bend has
    drifted **+8.7%** since the paper. Anyone re-running Stage 1 from those links
    today builds a measurably different dataset and has no way to detect the
    difference.

So every inventory this project assesses gets written as gzipped JSONL plus a
sidecar manifest recording the endpoint, the exact query, the fetch date, the
record count and a sha256 of the payload — the way ``benchmark/*/records.jsonl``
already pins benchmark inputs. The manifest is what makes a later count
re-derivable; the payload is what makes it reproducible when the publisher has
moved on.

Two publisher APIs cover every candidate in §3:

* **ArcGIS FeatureServer / MapServer** — Denver, SF, Charlotte, Boston, Sioux
  Falls, Minneapolis, Arlington, VDOT, WisDOT.
* **Socrata** — NYSDOT, CDOT.

Coordinates are always requested in **EPSG:4326**. Several of these layers are
published in a state-plane CRS (Denver's native is EPSG:2877, NAD83 / Colorado
Central in *US survey feet*), and silently mixing those is the "wrong projection
shifts a whole city" failure in §5. Asking the server to reproject keeps the
datum shift on the publisher's side, where it is authoritative — but the
round-trip is still worth checking, so ``--keep-native`` fetches a second copy in
the layer's own CRS for comparison.

    python scripts/analysis/fetch_inventory.py \
        --city denver-co \
        --arcgis https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services/ODC_TRANS_CURBRAMPS_P/FeatureServer/228

Network is the only thing this needs — no GPU, no imagery, no pipeline run. The
parsing core is pure and unit-tested in ``tests/test_fetch_inventory.py``; only
``fetch_*``, ``write_snapshot`` and ``main`` touch the network or disk.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.environ.get(
    "RAMPNET_INVENTORY_DIR", os.path.join(REPO, "data", "inventories")
)

# ArcGIS caps a single page server-side (``maxRecordCount``, commonly 1000-2000)
# and signals truncation with ``exceededTransferLimit``. We page by object ID
# rather than resultOffset: offset paging is not stable across concurrent edits,
# and several of these layers refresh weekly.
DEFAULT_PAGE_SIZE = 2000

# A fetch that pages forever is a bug, not a big city. NYC is the largest
# inventory in the programme at ~218k records.
MAX_PAGES = 1000

USER_AGENT = "RampNet-sourcing/1.0 (+https://github.com/ProjectSidewalk/RampNet)"


def _get_json(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def arcgis_query_url(layer_url, where="1=1", out_sr=4326, page_size=DEFAULT_PAGE_SIZE,
                     min_oid=None, oid_field="OBJECTID", count_only=False):
    """Build one ArcGIS ``/query`` URL. Pure — no network."""
    params = {"f": "json", "where": where}
    if count_only:
        params["returnCountOnly"] = "true"
    else:
        params.update({
            "outFields": "*",
            "returnGeometry": "true",
            "resultRecordCount": str(page_size),
            "orderByFields": oid_field,
        })
        if out_sr is not None:
            params["outSR"] = str(out_sr)
        if min_oid is not None:
            params["where"] = "{} > {}".format(oid_field, min_oid)
            if where not in ("1=1", ""):
                params["where"] = "({}) AND {}".format(where, params["where"])
    return layer_url.rstrip("/") + "/query?" + urllib.parse.urlencode(params)


def parse_arcgis_page(payload, geometry="point"):
    """Extract ``(records, exceeded_limit)`` from one ArcGIS query response.

    Point records are flattened to ``{**attributes, "lon": x, "lat": y}`` so
    downstream analysis never has to care which API served them. A feature with
    null geometry is dropped and counted by the caller — ArcGIS happily returns
    attribute-only rows, and a point inventory row without a point is not a ramp
    location.

    ``geometry="polyline"`` keeps ``paths`` instead, which is what a **street
    centreline** layer serves. Centrelines are not an inventory, but they are
    fetched by this tool on purpose: they are the independent municipal geometry
    the registration check in ``inventory_centerline_offset.py`` measures the ramp
    coordinates against, so they need the same snapshot discipline — same digest,
    same manifest, same refusal to write a truncated fetch. An analysis that
    attributes a city's positional error to its coordinates must not depend on a
    live endpoint for the reference it attributed against.
    """
    if "error" in payload:
        raise RuntimeError("ArcGIS error: {}".format(payload["error"]))
    out = []
    for feat in payload.get("features", []):
        geom = feat.get("geometry") or {}
        rec = dict(feat.get("attributes") or {})
        if geometry == "polyline":
            paths = geom.get("paths")
            # A path of one vertex has no direction, so it can carry no
            # perpendicular and would be dropped downstream anyway.
            paths = [p for p in (paths or []) if len(p) >= 2]
            if not paths:
                continue
            rec["paths"] = paths
        else:
            x, y = geom.get("x"), geom.get("y")
            if x is None or y is None:
                continue
            rec["lon"] = x
            rec["lat"] = y
        out.append(rec)
    return out, bool(payload.get("exceededTransferLimit"))


def parse_socrata_page(payload, lon_field="longitude", lat_field="latitude",
                       point_field=None):
    """Extract records from one Socrata page.

    Socrata publishes coordinates either as flat columns or as a GeoJSON-ish
    ``{"type": "Point", "coordinates": [lon, lat]}`` blob, depending on the
    dataset. Both shapes appear across the §3 candidates, so both are handled.
    """
    out = []
    for row in payload:
        rec = dict(row)
        lon = lat = None
        if point_field and isinstance(row.get(point_field), dict):
            coords = row[point_field].get("coordinates") or []
            if len(coords) >= 2:
                lon, lat = coords[0], coords[1]
        if lon is None and lon_field in row and lat_field in row:
            try:
                lon, lat = float(row[lon_field]), float(row[lat_field])
            except (TypeError, ValueError):
                lon = lat = None
        if lon is None or lat is None:
            continue
        rec["lon"] = lon
        rec["lat"] = lat
        out.append(rec)
    return out


def max_oid(records, oid_field="OBJECTID"):
    """Highest object ID in a page, or None. Drives ID-based pagination."""
    vals = [r[oid_field] for r in records if isinstance(r.get(oid_field), int)]
    return max(vals) if vals else None


def fetch_arcgis(layer_url, where="1=1", out_sr=4326, page_size=DEFAULT_PAGE_SIZE,
                 oid_field="OBJECTID", geometry="point", log=print):
    """Page an ArcGIS layer to exhaustion. Returns ``(records, pages, queries)``."""
    records, queries, min_oid = [], [], None
    for page in range(MAX_PAGES):
        url = arcgis_query_url(layer_url, where=where, out_sr=out_sr,
                               page_size=page_size, min_oid=min_oid,
                               oid_field=oid_field)
        queries.append(url)
        batch, _exceeded = parse_arcgis_page(_get_json(url), geometry=geometry)
        if not batch:
            break
        records.extend(batch)
        nxt = max_oid(batch, oid_field)
        if nxt is None:
            log("  ! no {} on page {} — cannot page by ID, stopping".format(
                oid_field, page))
            break
        min_oid = nxt
        log("  page {:>3}: +{:<5} total {}".format(page + 1, len(batch), len(records)))
    else:
        raise RuntimeError("hit MAX_PAGES={} — pagination is not terminating".format(MAX_PAGES))
    return records, len(queries), queries


def find_oid_field(layer_meta, fallback="OBJECTID"):
    """Read the layer's object-ID field name from its metadata. Pure.

    **Do not assume ``OBJECTID``.** San Francisco's curb-ramp layer names it
    something else, and since ID paging keys on that field, assuming the name
    made ``max_oid`` return None, which stopped the fetch after one page —
    2,000 of 50,096 records, written out as though complete. The count guard
    caught it, but the fix belongs here: ask the layer what its key is.
    """
    uniq = (layer_meta.get("uniqueIdField") or {}).get("name")
    if uniq:
        return uniq
    named = layer_meta.get("objectIdField")
    if named:
        return named
    for f in layer_meta.get("fields") or []:
        if f.get("type") == "esriFieldTypeOID":
            return f["name"]
    return fallback


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def write_snapshot(city, records, manifest, out_dir=DEFAULT_OUT):
    """Write ``<city>-<date>.jsonl.gz`` plus ``<...>.manifest.json``.

    The payload is written first and hashed, so the manifest can record the
    digest of exactly the bytes on disk. Two gzip header fields have to be pinned
    or the digest stops meaning "these records": ``mtime=0``, otherwise every
    re-fetch of unchanged data looks like a change, and ``filename=""``, because
    GzipFile otherwise reads the name off ``fileobj`` and embeds it — which would
    make the digest depend on what the file is called rather than what is in it.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = "{}-{}".format(city, manifest["fetched"])
    payload_path = os.path.join(out_dir, stem + ".jsonl.gz")
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    with open(payload_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(body)
    with open(payload_path, "rb") as fh:
        manifest["sha256"] = sha256_bytes(fh.read())
    manifest["records"] = len(records)
    manifest["payload"] = os.path.basename(payload_path)
    manifest_path = os.path.join(out_dir, stem + ".manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return payload_path, manifest_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True,
                    help="slug for the snapshot filename, e.g. denver-co")
    ap.add_argument("--arcgis", help="ArcGIS FeatureServer/MapServer layer URL")
    ap.add_argument("--socrata", help="Socrata resource URL (…/resource/abcd-1234.json)")
    ap.add_argument("--point-field", default=None,
                    help="Socrata GeoJSON point column, e.g. the_geom (NYC)")
    ap.add_argument("--lon-field", default="longitude")
    ap.add_argument("--lat-field", default="latitude")
    ap.add_argument("--where", default="1=1", help="ArcGIS where clause")
    ap.add_argument("--oid-field", default=None,
                    help="object-ID field to page on; read from the layer if omitted")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a snapshot even when it is short of the server's "
                         "own count (only for a layer with genuinely null geometry)")
    ap.add_argument("--geometry", choices=("point", "polyline"), default="point",
                    help="'polyline' freezes a street-centreline layer, the "
                         "reference the registration check measures against")
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    ap.add_argument("--out-sr", type=int, default=4326,
                    help="output CRS; 4326 unless you are checking the round-trip")
    ap.add_argument("--fetched", required=True,
                    help="fetch date YYYY-MM-DD (explicit, so the snapshot name "
                         "is not a function of the machine clock)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--note", default="", help="free text recorded in the manifest")
    args = ap.parse_args(argv)

    if bool(args.arcgis) == bool(args.socrata):
        ap.error("pass exactly one of --arcgis / --socrata")

    if args.arcgis:
        count_url = arcgis_query_url(args.arcgis, where=args.where, count_only=True)
        declared = _get_json(count_url).get("count")
        oid_field = args.oid_field or find_oid_field(_get_json(args.arcgis + "?f=json"))
        print("declared count: {} | paging on {}".format(declared, oid_field))
        records, pages, queries = fetch_arcgis(
            args.arcgis, where=args.where, out_sr=args.out_sr,
            page_size=args.page_size, oid_field=oid_field, geometry=args.geometry)
        manifest = {
            "city": args.city,
            "fetched": args.fetched,
            "api": "arcgis",
            "endpoint": args.arcgis,
            "where": args.where,
            "geometry": args.geometry,
            "out_sr": args.out_sr,
            "declared_count": declared,
            "pages": pages,
            "count_query": count_url,
            "first_query": queries[0] if queries else None,
            "oid_field": oid_field,
            "note": args.note,
        }
        if declared is not None and declared != len(records):
            manifest["count_mismatch"] = {
                "declared": declared, "fetched": len(records),
                "note": "server count vs rows retained after dropping null geometry",
            }
            # A short fetch that writes a normal-looking snapshot is the worst
            # outcome here: every downstream count silently inherits the
            # truncation. Refuse by default and make the operator say otherwise.
            print("! declared {} but retained {}".format(declared, len(records)))
            if not args.allow_partial:
                print("  refusing to write a truncated snapshot. If the shortfall is "
                      "genuinely null geometry, re-run with --allow-partial.")
                return 2
    else:
        rows, offset = [], 0
        while True:
            url = args.socrata + ("&" if "?" in args.socrata else "?") + \
                urllib.parse.urlencode({"$limit": args.page_size, "$offset": offset,
                                        "$order": ":id"})
            payload = _get_json(url)
            if not payload:
                break
            page = parse_socrata_page(payload, lon_field=args.lon_field,
                                      lat_field=args.lat_field,
                                      point_field=args.point_field)
            rows.extend(page)
            offset += args.page_size
            print("  offset {:>6}: kept {} of {} rows, total {}".format(
                offset, len(page), len(payload), len(rows)))
        records = rows
        manifest = {
            "city": args.city, "fetched": args.fetched, "api": "socrata",
            "endpoint": args.socrata, "out_sr": 4326, "note": args.note,
            "point_field": args.point_field,
        }

    payload_path, manifest_path = write_snapshot(args.city, records, manifest,
                                                 out_dir=args.out_dir)
    print("\nwrote {} ({} records)".format(payload_path, len(records)))
    print("wrote {}".format(manifest_path))
    print("sha256 {}".format(manifest["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
