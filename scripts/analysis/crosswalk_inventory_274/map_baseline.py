"""Map-only crosswalk baseline, scored against Seattle's SDOT inventory (RampNet#274, Phase 1 side work).

Question: how well can "is there a marked crosswalk here?" be answered from free map data alone, with no
pixels? Whatever a street-level or aerial detector adds has to be measured against this bar.

  python map_baseline.py --out analysis_out/crosswalk_inventory_274 \
      --census D:/Git/sal-scratch-keep/phase0-census-2026-10-28/census/out

Reads only what `sdot_inventory.py fetch` pulled (hash-checked) plus the Phase 0 census cache (OSM tiles and
crowd labels, 2026-10-28). Writes `baseline_report.md`, `baseline_nodes.csv` (one row per OSM node in the
scored universe) and `baseline_sdot.csv` (one row per current SDOT crosswalk). No network, no GPU.

Truth is SDOT under the AMENDED rule (sdot_inventory.py, `AMENDMENT`): a current SDOT point within MATCH_M =
10 m means painted, and a place with no SDOT point of ANY status within HOLE_M = 100 m is outside the
inventory's coverage and is left out, because there SDOT's silence says nothing.

Two scoring frames, because the two ways a map can be wrong are seen from different places:
  * NODES: every OSM `highway=crossing` node inside the Seattle regions and inside SDOT coverage. A predictor
    says yes or no per node; truth is an SDOT point within 10 m. Precision, recall, F1.
  * SDOT: every current SDOT crosswalk inside the regions (all of them are covered by definition). A predictor
    "finds" one when it has a positive within 10 m: a yes-node, or for the crowd predictors a label. This is
    the only frame that sees a crosswalk OSM has no node for.

Predictors (all from free data, fixed before scoring):
  osm_marked         Phase 0's `is_marked(tags) == "marked"`.
  osm_marked_signal  osm_marked, or a node not tagged unmarked whose `crossing=traffic_signals` or
                     `crossing:signals=yes` (signalised crossings in the US are nearly always painted).
  crowd_any          a crowd Crosswalk label within 10 m, unless human votes judged it incorrect.
  crowd_correct      a crowd Crosswalk label within 10 m judged correct by human votes (agree - disagree >= 2).
  osm_or_crowd       osm_marked_signal OR crowd_any.
Crowd predictors are limited to where Project Sidewalk users have audited, which is reported beside them.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sdot_inventory import (CITY, CURRENT, HOLE_M, MATCH_M, Nearest, crowd_labels, frac, fmt_m, is_marked,
                            load_inputs, sdot_points, wilson)  # noqa: E402

PREDICTORS = ("osm_marked", "osm_marked_signal", "crowd_any", "crowd_correct", "osm_or_crowd")


def osm_nodes_tagged(census, regions):
    """node_id -> (lat, lng, tags) for every highway=crossing node inside the regions."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(census, "osm", CITY, "Crosswalk", "*.json"))):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            t = e.get("tags") or {}
            if e.get("type") == "node" and t.get("highway") == "crossing" and regions.contains(e["lat"], e["lon"]):
                out[e["id"]] = (e["lat"], e["lon"], t)
    return out


def signalised(t):
    return t.get("crossing") == "traffic_signals" or t.get("crossing:signals") == "yes"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return p, r, f


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]"


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--census", required=True, help="Phase 0 census OUT dir (rawlabels/ + osm/)")
    a = ap.parse_args()

    rec, sd, regions = load_inputs(a)
    allp = sdot_points(sd)
    cur = [p for p in allp if p[2]["status"] in CURRENT]
    near_cur, near_all = Nearest(cur), Nearest(allp)
    labs = [x for x in crowd_labels(a.census) if regions.contains(x[0], x[1])]
    near_lab_any = Nearest([x for x in labs if x[2]["verdict"] != "incorrect"])
    near_lab_ok = Nearest([x for x in labs if x[2]["verdict"] == "correct"])
    near_lab_all = Nearest(labs)

    def within(grid, lat, lng, r=MATCH_M):
        return grid.query(lat, lng, max_m=r)[0] is not None

    # ---- NODES frame
    nodes = osm_nodes_tagged(a.census, regions)
    nrows = []
    n_uncovered = 0
    for nid, (lat, lng, t) in sorted(nodes.items()):
        da, _ = near_all.query(lat, lng, max_m=HOLE_M)
        if da is None:
            n_uncovered += 1
            continue
        d, it = near_cur.query(lat, lng)
        cls = is_marked(t)
        pred = {"osm_marked": cls == "marked",
                "osm_marked_signal": cls == "marked" or (cls != "unmarked" and signalised(t)),
                "crowd_any": within(near_lab_any, lat, lng),
                "crowd_correct": within(near_lab_ok, lat, lng)}
        pred["osm_or_crowd"] = pred["osm_marked_signal"] or pred["crowd_any"]
        nrows.append({"node_id": nid, "osm_class": cls, "signalised": int(signalised(t)),
                      "audited": int(within(near_lab_all, lat, lng, 100.0)),
                      "d_sdot_current_m": fmt_m(d), "sdot_marking": it[2]["marking"] if it else "",
                      "truth": int(d is not None and d <= MATCH_M), **{k: int(v) for k, v in pred.items()}})
    write_csv(os.path.join(a.out, "baseline_nodes.csv"), nrows)

    # ---- SDOT frame
    cur_in = [p for p in cur if regions.contains(p[0], p[1])]
    pos_nodes = {k: Nearest([(nodes[int(r["node_id"])][0], nodes[int(r["node_id"])][1], {}) for r in nrows
                             if r[k]]) for k in ("osm_marked", "osm_marked_signal")}
    srows = []
    for lat, lng, p in cur_in:
        found = {"osm_marked": within(pos_nodes["osm_marked"], lat, lng),
                 "osm_marked_signal": within(pos_nodes["osm_marked_signal"], lat, lng),
                 "crowd_any": within(near_lab_any, lat, lng),
                 "crowd_correct": within(near_lab_ok, lat, lng)}
        found["osm_or_crowd"] = found["osm_marked_signal"] or found["crowd_any"]
        srows.append({"objectid": p["objectid"], "marking": p["marking"], "midblock": p["midblock"],
                      "audited": int(within(near_lab_all, lat, lng, 100.0)),
                      **{k: int(v) for k, v in found.items()}})
    write_csv(os.path.join(a.out, "baseline_sdot.csv"), srows)

    # ---- report
    L = []
    w = L.append
    w("# Map-only crosswalk baseline vs SDOT, Seattle (RampNet#274)\n")
    w(f"Generated by `scripts/analysis/crosswalk_inventory_274/map_baseline.py`. SDOT pulled "
      f"{rec['sdot']['fetched_utc']} (sha256 `{rec['sdot']['sha256'][:16]}…`); OSM tiles and crowd labels from "
      f"the Phase 0 census cache (2026-10-28). Truth: a current SDOT point within {MATCH_M:.0f} m, scored only "
      f"inside SDOT coverage (some SDOT point of any status within {HOLE_M:.0f} m; the amended rule in "
      f"`sdot_inventory.py`). No pixels are used anywhere here.\n")
    w("## 1. Per OSM crossing node\n")
    npos = sum(r["truth"] for r in nrows)
    w(f"{len(nodes)} OSM crossing nodes inside the Seattle regions; {n_uncovered} outside SDOT coverage (left "
      f"out); {len(nrows)} scored, {npos} of them with an SDOT crosswalk within {MATCH_M:.0f} m.\n")
    w("| predictor | yes | precision | recall | F1 |\n|---|---|---|---|---|")
    for k in PREDICTORS:
        tp = sum(1 for r in nrows if r[k] and r["truth"])
        fp = sum(1 for r in nrows if r[k] and not r["truth"])
        fn = sum(1 for r in nrows if not r[k] and r["truth"])
        p, rc, f = prf(tp, fp, fn)
        w(f"| `{k}` | {tp + fp} | {p:.3f} {ci(tp, tp + fp)} | {rc:.3f} {ci(tp, tp + fn)} | {f:.3f} |")
    w("")
    w("Where the best map predictor goes wrong, by OSM class (`osm_marked_signal`):\n")
    w("| OSM class | nodes | SDOT yes | predictor says yes | missed (SDOT yes, map no) | over-claimed (map yes, "
      "SDOT no) |\n|---|---|---|---|---|---|")
    for c in ("marked", "unknown", "unmarked"):
        cs = [r for r in nrows if r["osm_class"] == c]
        w(f"| {c} | {len(cs)} | {sum(r['truth'] for r in cs)} | {sum(r['osm_marked_signal'] for r in cs)} | "
          f"{sum(1 for r in cs if r['truth'] and not r['osm_marked_signal'])} | "
          f"{sum(1 for r in cs if r['osm_marked_signal'] and not r['truth'])} |")
    w("")
    aud = [r for r in nrows if r["audited"]]
    w(f"Crowd coverage: {len(aud)} of {len(nrows)} scored nodes ({len(aud) / len(nrows):.3f}) have any crowd "
      f"Crosswalk label within 100 m. On those nodes only:\n")
    w("| predictor | precision | recall | F1 |\n|---|---|---|---|")
    for k in PREDICTORS:
        tp = sum(1 for r in aud if r[k] and r["truth"])
        fp = sum(1 for r in aud if r[k] and not r["truth"])
        fn = sum(1 for r in aud if not r[k] and r["truth"])
        p, rc, f = prf(tp, fp, fn)
        w(f"| `{k}` | {p:.3f} | {rc:.3f} | {f:.3f} |")
    w("")
    w("## 2. Per SDOT crosswalk (recall, including crosswalks OSM has no node for)\n")
    w(f"{len(srows)} current SDOT crosswalks inside the regions.\n")
    w("| predictor | found within 10 m | found, audited areas only |\n|---|---|---|")
    sa = [r for r in srows if r["audited"]]
    for k in PREDICTORS:
        w(f"| `{k}` | {frac(sum(r[k] for r in srows), len(srows))} | {frac(sum(r[k] for r in sa), len(sa))} |")
    w("")
    w("By SDOT marking type (`osm_marked_signal`):\n")
    w("| marking | crosswalks | found |\n|---|---|---|")
    for m in sorted({r["marking"] for r in srows}, key=lambda m: -sum(r["marking"] == m for r in srows)):
        ms = [r for r in srows if r["marking"] == m]
        w(f"| `{m or '(empty)'}` | {len(ms)} | {frac(sum(r['osm_marked_signal'] for r in ms), len(ms))} |")
    mb = [r for r in srows if r["midblock"] == "Y"]
    w(f"\nMid-block crosswalks: {frac(sum(r['osm_marked_signal'] for r in mb), len(mb))} found by "
      f"`osm_marked_signal`.\n")
    w("## Reading\n")
    w("A detector scored the same way has to beat the best row of each table. The cells it can win are the "
      "\"missed\" and \"over-claimed\" counts in section 1 and the unfound share in section 2; everything else the "
      "map already gets right. Truth is SDOT, which itself missed 4 of 102 covered human-painted rows in Phase 0, "
      "so a few percent of disagreement here is SDOT's, not the map's.\n")
    with open(os.path.join(a.out, "baseline_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}/baseline_report.md, baseline_nodes.csv ({len(nrows)}), baseline_sdot.csv ({len(srows)})")


if __name__ == "__main__":
    main()
