"""Sweep ArcGIS Hub for curb-ramp inventories, by synonym (issues #96, #59).

§3 of `docs/curb_ramp_data_sourcing.md` records the trap this automates: an
earlier pass searched dataset titles for **"curb ramp"** only, concluded supply
was thin, and recommended verifying rather than searching further. That was
wrong — a title search for "curb ramp" does not match NYC's own *Pedestrian Ramp
Locations*. Re-running across the synonym set surfaced ~195k ramps in one pass.

That correction was applied by hand once. This makes it repeatable, so a future
pass cannot regress to one phrase, and so the candidate list can be refreshed as
cities publish. Hub's search API returns ``recordCount`` per layer, which is the
number that decides whether a candidate is worth pursuing at all.

    python scripts/analysis/discover_inventories.py --min-records 2000
    python scripts/analysis/discover_inventories.py --org Dallas Spokane Tacoma

**What this does not do.** ``recordCount`` is the layer's row count, not a count
of *ramps*: Charlotte's 40,601 included 5,505 ``RP_Type=NoRamp`` assertions, and
San Francisco's 50,096 rows held 7,553 distinct points because they are
intersection centroids (§6). A hit here is a candidate to *read*, never a number
to add to a total. Sidewalk-segment layers in particular will match "ramp"
queries and count segments.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))

HUB_API = "https://hub.arcgis.com/api/v3/datasets"

#: The synonym set from §3. Naming is wildly inconsistent across publishers —
#: *Curb Ramps* (Seattle, Portland), *Pedestrian Ramp Locations* (NYC), *ADA Ped
#: Ramps* (Minneapolis), *Pedestrian Ramp Inventory* (Boston), *Access Ramps*
#: (LA), *Sidewalk ADA Ramps* (Arlington), *sCurbRamps* (Bend) — so the query
#: set, not any single phrase, is the instrument.
QUERIES = (
    "curb ramp", "pedestrian ramp", "ADA ramp", "ped ramp",
    "access ramp", "curb cut", "sidewalk ramp", "ADA curb",
)

#: Layers already recorded in §3/§5, by a distinctive substring of org or name.
#: Reported separately rather than hidden, so "known" stays auditable.
KNOWN = (
    "seattle", "portland", "bend", "new york", "nyc", "denver", "san francisco",
    "charlotte", "boston", "sioux falls", "minneapolis", "arlington",
    "vdot", "virginia", "wisconsin", "wisdot", "colorado", "cdot",
    "nysdot", "austin", "nashville", "los angeles", "washington",
)

#: Layers whose geometry is lines/polygons are sidewalk or corridor inventories,
#: not ramp points. Kept but flagged — a few publishers store ramps as polygons.
POINT_TYPES = ("esriGeometryPoint", "esriGeometryMultipoint")


def fetch(url, timeout=45, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "RampNet-inventory-discovery/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as exc:                       # noqa: BLE001
            if attempt == retries - 1:
                print(f"  !! {type(exc).__name__} on {url[:90]}", file=sys.stderr)
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def search(query, pages=3, page_size=100):
    """All Hub datasets matching ``query``. Pages until exhausted or ``pages``.

    Stops on an **empty** page rather than a short one. A short page is not proof
    of the end: the API applies its own filtering after paging, so a page that
    comes back with 80 of 100 rows can still be followed by a full one, and
    treating it as terminal silently truncates the sweep. The `pages` cap bounds
    the cost either way, and every total this feeds is reported as a floor.
    """
    out = []
    for page in range(1, pages + 1):
        url = (f"{HUB_API}?q={urllib.parse.quote(query)}"
               f"&page[size]={page_size}&page[number]={page}")
        d = fetch(url)
        if not d or not d.get("data"):
            break
        out.extend(d["data"])
    return out


def is_known(row):
    hay = ((row.get("orgName") or "") + " " + (row.get("name") or "")).lower()
    return any(k in hay for k in KNOWN)


#: "Ramp" is badly overloaded. These are the false positives this sweep actually
#: produced: boat ramps (Florida FWC, 2,631), railroad ramps (CSX, 16,319), and
#: — subtler and more dangerous — **planned-work layers**, which have the same
#: wrong polarity as Atlanta's *Missing ADA Ramps* (§3): a list of places a ramp
#: is needed is not a list of ramps. Counting those would inflate supply with
#: records that are confirmed ABSENCE.
NOT_RAMPS = re.compile(
    r"boat ramp|csx|obstruction|\bneeds?\b|improvement|work order|"
    r"project|missing|planned|proposed|no curb ramp|clearing", re.I)


def looks_like_ramps(row):
    """Name mentions a pedestrian ramp or curb cut, and is not a known
    false-positive class. Guards against sidewalk-segment layers that match the
    query only through their description."""
    name = row.get("name") or ""
    if NOT_RAMPS.search(name):
        return False
    return bool(re.search(r"\bramp|curb ?cut|curb ?ramp", name, re.I))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-records", type=int, default=1000)
    ap.add_argument("--org", nargs="*", default=None,
                    help="only report orgs whose name contains one of these")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--include-known", action="store_true")
    ap.add_argument("--all-geometry", action="store_true",
                    help="keep line/polygon layers too (default: points only)")
    ap.add_argument("--json", default=os.path.join(OUT, "inventory_discovery.json"))
    args = ap.parse_args(argv)

    seen, rows = set(), []
    for q in QUERIES:
        hits = search(q, pages=args.pages)
        print(f"  {q!r}: {len(hits)} hits", file=sys.stderr)
        for h in hits:
            a = h.get("attributes", {})
            key = (a.get("url") or "") + "|" + (a.get("name") or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": a.get("name"), "org": a.get("orgName"),
                "records": a.get("recordCount"), "geometry": a.get("geometryType"),
                "url": a.get("url"), "type": a.get("type"),
                "region": a.get("region"), "modified": a.get("modified"),
                "matched_query": q,
            })

    cand = [r for r in rows
            if (r["records"] or 0) >= args.min_records and looks_like_ramps(r)]
    if not args.all_geometry:
        cand = [r for r in cand if r["geometry"] in POINT_TYPES]
    if args.org:
        low = [o.lower() for o in args.org]
        cand = [r for r in cand
                if any(o in ((r["org"] or "") + " " + (r["name"] or "")).lower()
                       for o in low)]
    known = [r for r in cand if is_known(r)]
    fresh = sorted((r for r in cand if not is_known(r)),
                   key=lambda r: -(r["records"] or 0))

    print(f"\n{len(rows)} unique layers seen; {len(cand)} look like ramp points "
          f"with >= {args.min_records} records\n")
    print(f"{'records':>9}  {'org':38s} {'name':44s} geometry")
    print("-" * 108)
    for r in fresh:
        print(f"{r['records']:>9}  {(r['org'] or '')[:38]:38s} "
              f"{(r['name'] or '')[:44]:44s} {(r['geometry'] or '')[13:]}")
    if known and args.include_known:
        print(f"\n-- already in §3/§5 ({len(known)}) --")
        for r in sorted(known, key=lambda r: -(r["records"] or 0)):
            print(f"{r['records']:>9}  {(r['org'] or '')[:38]:38s} {(r['name'] or '')[:44]}")
    elif known:
        print(f"\n({len(known)} hits matched cities already in §3/§5; "
              f"--include-known to list them)")

    payload = {"queries": list(QUERIES), "min_records": args.min_records,
               "n_seen": len(rows), "candidates": fresh, "known": known}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.json}")
    print("\n!! recordCount is ROWS, not ramps. Charlotte's 40,601 held 5,505 "
          "NoRamp assertions;\n   San Francisco's 50,096 rows were 7,553 "
          "intersection centroids. Read before counting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
