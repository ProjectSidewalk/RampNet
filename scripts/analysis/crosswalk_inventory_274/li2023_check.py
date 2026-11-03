"""Li et al. 2023 street-view crosswalk counts as a second outside opinion (RampNet#274, Phase 1 side work).

Li, Sheng, Irvin, Chung, Ying, Sun, Ng & Rodriguez, "Marked crosswalks in US transit-oriented station areas,
2007-2020: a computer vision approach using street view imagery", Environment and Planning B 50(2), 2023,
doi:10.1177/23998083221112157. Supplementary data: Zenodo 6803956 (`inf_crosswalk.csv`, CC BY 4.0): for every
OSM intersection within 250 m of a US TOD station and every year 2007-2020, the number of parallel-line
(`plain`) and high-visibility (`zebra`) crosswalk boxes their detector found in four 90-degree Street View crops
(`both`), with `impute` = 1 when the year had no imagery. It is a MODEL's output, so it is an opinion, not ground
truth, and it counts per intersection rather than locating each crossing.

  python li2023_check.py fetch --out analysis_out/crosswalk_inventory_274
  python li2023_check.py score --out analysis_out/crosswalk_inventory_274 \
      [--phase1-sheet <crosswalk-gt-phase1 worktree>/analysis_out/crosswalk_gt_phase1_274/sheet.csv]

`fetch` downloads the CSV (142 MB, untracked) and adds its sha256 to `inputs.json`. `score` needs
`sdot_inventory.py fetch` to have run first (it reuses the SDOT pull and the Seattle region polygons) and writes
`li2023_report.md` + `li2023_seattle.csv`. Descriptive only: no rule, no verdict.

Seattle comparison: each Li intersection inside the Seattle regions, at its latest non-imputed year, is
"marked" when `both` > 0. SDOT and OSM are counted within LEG_M = 30 m of the intersection point (a crosswalk
sits on a leg, ~5-20 m from the junction centre). Li's year is <= 2020 and SDOT's is 2026, so a real change in
between reads as a disagreement. `--phase1-sheet` only COUNTS how many Phase 1 GT nodes lie within LEG_M of a Li
intersection, per city; it never reads Li's answer for them, so nothing here can unblind the rating round.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sdot_inventory import (CURRENT, Nearest, frac, haversine_m, load_inputs, osm_nodes, sdot_points,
                            sha256)  # noqa: E402

LI_URL = "https://zenodo.org/api/records/6803956/files/inf_crosswalk.csv/content"
LI_FILE = "inputs/li2023_inf_crosswalk.csv"
LEG_M = 30.0


def fetch(a):
    p = os.path.join(a.out, LI_FILE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with urllib.request.urlopen(LI_URL, timeout=600) as r, open(p, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    rp = os.path.join(a.out, "inputs.json")
    rec = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
    rec["li2023"] = {"url": LI_URL, "zenodo_record": 6803956, "license": "CC-BY-4.0", "sha256": sha256(p),
                     "bytes": os.path.getsize(p), "file": LI_FILE,
                     "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"{LI_FILE}: {rec['li2023']['bytes']} bytes, sha256 {rec['li2023']['sha256'][:16]}")


def latest_li(path):
    """intersection_id -> (lat, lng, year, both, plain, zebra, agency) at its latest non-imputed year."""
    best = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["impute"] == "1":
                continue
            i, y = r["intersection_id"], int(r["year"])
            if i not in best or y > best[i][2]:
                best[i] = (float(r["lat_intersection"]), float(r["lon_intersection"]), y, float(r["both"]),
                           float(r["plain"]), float(r["zebra"]), r["Agency"].strip())
    return best


def count_within(grid, lat, lng, r):
    """Number of points in a `Nearest` grid within r metres of (lat, lng)."""
    ci, cj = int(lat // grid.dlat), int(lng // grid.dlng)
    k = int(r // grid.cell_m) + 1
    n = 0
    for i in range(ci - k, ci + k + 1):
        for j in range(cj - k, cj + k + 1):
            for it in grid.b.get((i, j), ()):
                if haversine_m(lat, lng, it[0], it[1]) <= r:
                    n += 1
    return n


def score(a):
    rec, sd, regions = load_inputs(a)
    assert sha256(os.path.join(a.out, LI_FILE)) == rec["li2023"]["sha256"], "Li CSV changed since fetch"
    li = latest_li(os.path.join(a.out, LI_FILE))
    sea = {i: v for i, v in li.items() if regions.contains(v[0], v[1])}
    cur = [p for p in sdot_points(sd) if p[2]["status"] in CURRENT]
    g_sdot = Nearest(cur)
    nodes = osm_nodes(a.census)
    g_osm_m = Nearest([(v[0], v[1], {"id": k}) for k, v in nodes.items() if v[2] == "marked"])
    rows = []
    for i, (lat, lng, y, both, plain, zebra, ag) in sorted(sea.items(), key=lambda t: int(t[0])):
        rows.append({"intersection_id": i, "year": y, "li_both": both, "li_plain": plain, "li_zebra": zebra,
                     "agency": ag, "sdot_within": count_within(g_sdot, lat, lng, LEG_M),
                     "osm_marked_within": count_within(g_osm_m, lat, lng, LEG_M)})
    with open(os.path.join(a.out, "li2023_seattle.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    L = []
    w = L.append
    w("# Li et al. 2023 street-view crosswalk counts vs SDOT and OSM, Seattle (RampNet#274)\n")
    w(f"Generated by `scripts/analysis/crosswalk_inventory_274/li2023_check.py score`. Li CSV sha256 "
      f"`{rec['li2023']['sha256'][:16]}…` (Zenodo 6803956, CC BY 4.0); SDOT and regions as in `report.md`. "
      f"Counts within {LEG_M:.0f} m of each Li intersection; Li at its latest non-imputed year.\n")
    yrs = sorted({r["year"] for r in rows})
    w(f"{len(li)} Li intersections with a non-imputed year nationally; {len(rows)} inside the Seattle regions "
      f"(latest year {yrs[0]}–{yrs[-1]}; {sum(r['year'] == 2020 for r in rows)} at 2020).\n")
    w("| | SDOT ≥ 1 within 30 m | SDOT none | OSM-marked ≥ 1 | OSM-marked none |\n|---|---|---|---|---|")
    for lab, sel in (("Li: crosswalk (both > 0)", lambda r: r["li_both"] > 0),
                     ("Li: none", lambda r: r["li_both"] == 0)):
        s = [r for r in rows if sel(r)]
        w(f"| {lab} ({len(s)}) | {sum(r['sdot_within'] > 0 for r in s)} | {sum(r['sdot_within'] == 0 for r in s)} | "
          f"{sum(r['osm_marked_within'] > 0 for r in s)} | {sum(r['osm_marked_within'] == 0 for r in s)} |")
    w("")
    sdot_pos = [r for r in rows if r["sdot_within"] > 0]
    sdot_neg = [r for r in rows if r["sdot_within"] == 0]
    w(f"Taking SDOT as the reference: Li finds a crosswalk at "
      f"{frac(sum(r['li_both'] > 0 for r in sdot_pos), len(sdot_pos))} of intersections SDOT has one at, and "
      f"none at {frac(sum(r['li_both'] == 0 for r in sdot_neg), len(sdot_neg))} of those SDOT has none at.\n")
    agree = sum((r["li_both"] > 0) == (r["sdot_within"] > 0) for r in rows)
    w(f"Presence agreement Li vs SDOT: {frac(agree, len(rows))}; "
      f"OSM-marked vs SDOT: {frac(sum((r['osm_marked_within'] > 0) == (r['sdot_within'] > 0) for r in rows), len(rows))}.\n")

    if a.phase1_sheet:
        sheet = list(csv.DictReader(open(a.phase1_sheet, encoding="utf-8", newline="")))
        allli = list(li.values())
        g = Nearest([(v[0], v[1], {}) for v in allli], lat0=40.0)
        w("## Phase 1 coverage (counts only; Li's answers for these rows are NOT read)\n")
        w("| city | GT rows | within 30 m of a Li intersection |\n|---|---|---|")
        for c in dict.fromkeys(r["city"] for r in sheet):
            cs = [r for r in sheet if r["city"] == c]
            k = sum(1 for r in cs if g.query(float(r["node_lat"]), float(r["node_lng"]), max_m=LEG_M)[0] is not None)
            w(f"| {c} | {len(cs)} | {k} |")
        w("")
    with open(os.path.join(a.out, "li2023_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}/li2023_report.md and li2023_seattle.csv ({len(rows)} Seattle intersections)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--census", required=True, help="Phase 0 census OUT dir (osm/ tiles)")
    s.add_argument("--phase1-sheet")
    a = ap.parse_args()
    {"fetch": fetch, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
