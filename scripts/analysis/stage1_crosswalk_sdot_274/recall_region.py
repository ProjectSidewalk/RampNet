"""Region matching for the missed-crosswalk check (#285, exploratory; added after the radius results were seen).

A crosswalk is long and thin, so the reviewer's one click and the detector's peak can both be on the same
crosswalk and still be further apart than the match radius. This re-scores the same marks with the Vistas
segmenter: the mark's *region* is the segmented crosswalk component nearest the click (within pilot.SNAP_R,
the placer's own rule), after a morphological closing that joins zebra stripes. A mark is found if a
detection lies on that region or within TOL of it. Marks with no region fall back to the radius rule.
Matching stays one-to-one (a detection can serve one mark).

    python recall_region.py pairs --verdicts <export>.json --preds <city>.jsonl ... --cities laurens-ia juneau-ak \
        --native "laurens-ia=<dir>" "juneau-ak=<dir>" --label crosswalk_s4 --threshold 0.23    # GPU, minutes
    python recall_region.py score --label crosswalk_s4                                          # CPU, seconds

Outputs: analysis_out/crosswalk_followups_285/recall/score/<label>/{region_pairs.csv,region_marks.csv,
region_recall.json}.
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

import recall_check as rc

TOL = 0.022          # the detector matcher's radius, pano-width units: how far off the region a peak may sit
CLOSE_PX = 15        # closing kernel (px in the 1024 px, 90 degree view) that joins zebra stripes
RADIUS = 0.05        # the page's primary radius; fallback where the segmenter sees no region
BANDS = ((0.532, 0.58, "10 to 25 m"), (0.58, 0.63, "6 to 10 m"), (0.63, 1.0, "nearer than 6 m"))


def cmd_pairs(a):
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label as cc_label
    import crowd_arm
    import pilot
    import vistas_crosswalk_probe as probe
    from equirect_tiling import equirect_point_to_perspective, perspective_point_to_equirect, View

    v = json.load(open(a.verdicts, encoding="utf-8"))["panos"]
    native = dict(s.split("=", 1) for s in a.native)
    preds = {}
    for path in a.preds:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            preds[r["pano_id"]] = [(x, y, c) for x, y, c in r["preds"] if c >= a.threshold]
    info = {}
    for city in a.cities:
        for r in csv.DictReader(open(os.path.join(rc.OUT, city, "intersections.csv"), encoding="utf-8")):
            if r["in_draw"] == "1":
                info[r["pano_id"]] = {"city": city, "camera_model": r["camera_model"], "flagged": r["flagged"]}
    seg = probe.Segmenter()
    struct = np.ones((CLOSE_PX, CLOSE_PX), bool)
    marks, pairs = [], []
    todo = [(pid, e) for pid, e in sorted(v.items()) if any(not m.get("unsure") for m in e.get("missed", []))]
    for n, (pid, e) in enumerate(todo):
        pano = crowd_arm.load_pano(os.path.join(native[info[pid]["city"]], pid + ".jpg"))
        sure = [(m["x"], m["y"]) for m in e["missed"] if not m.get("unsure")]
        for i, (x, y) in enumerate(sure):
            o = pilot.vistas_at(seg, pano, x, y)
            row = {"pano_id": pid, **info[pid], "mark": i, "x": round(x, 4), "y": round(y, 4),
                   "within_25m": int(y >= rc.Y_25M), "on_paint": o["on_paint"], "region_found": 0,
                   "region_area_px": "", "region_width": ""}
            region = None
            if o["comp_found"]:
                closed = ndimage.binary_closing(o["sel"], structure=struct) | o["sel"]
                lab = cc_label(closed, connectivity=2)
                _, (ii, jj) = ndimage.distance_transform_edt(~o["sel"], return_indices=True)
                ni, nj = int(ii[o["pv"], o["pu"]]), int(jj[o["pv"], o["pu"]])   # the pixel the placer snapped to
                region = lab == lab[ni, nj]
                cols = np.nonzero(region.any(axis=0))[0]
                view = View((x - 0.5) * 360.0, -(y - 0.5) * 180.0, 90.0, 90.0, pilot.VIEW_PX, pilot.VIEW_PX)
                # angular width of the region across the view, pano-width units (90 degrees = 0.25)
                ex = [perspective_point_to_equirect((c + 0.5) / pilot.VIEW_PX, 0.5, view)[0] for c in (cols[0], cols[-1])]
                row.update({"region_found": 1, "region_area_px": int(region.sum()),
                            "region_width": round(min(abs(ex[1] - ex[0]), 1 - abs(ex[1] - ex[0])), 4)})
                _, (ri, rj) = ndimage.distance_transform_edt(~region, return_indices=True)
            marks.append(row)
            for j, (dx, dy, c) in enumerate(preds[pid]):
                p = {"pano_id": pid, "mark": i, "det": j, "conf": round(c, 4),
                     "click_dist": round(rc.wrap_dist(x, y, dx, dy), 5), "region_dist": ""}
                if region is not None:
                    uv = equirect_point_to_perspective(dx, dy, view)
                    if uv is not None:
                        pu = min(pilot.VIEW_PX - 1, int(uv[0] * pilot.VIEW_PX))
                        pv = min(pilot.VIEW_PX - 1, int(uv[1] * pilot.VIEW_PX))
                        if region[pv, pu]:
                            p["region_dist"] = 0.0
                        else:
                            qx, qy = perspective_point_to_equirect((int(rj[pv, pu]) + 0.5) / pilot.VIEW_PX,
                                                                   (int(ri[pv, pu]) + 0.5) / pilot.VIEW_PX, view)
                            p["region_dist"] = round(rc.wrap_dist(dx, dy, qx, qy), 5)
                pairs.append(p)
        if n % 10 == 0:
            pilot.log(f"  {n}/{len(todo)} panos")
    out = os.path.join(rc.OUT, "score", a.label)
    os.makedirs(out, exist_ok=True)
    for name, rows in (("region_marks_raw.csv", marks), ("region_pairs.csv", pairs)):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(marks)} marks, {len(pairs)} mark-detection pairs -> {out}")


def greedy(cands):
    """One-to-one, nearest first. cands: (distance, mark key, det key). Returns the hit mark keys.

    >>> sorted(greedy([(0.0, 'a', 1), (0.01, 'b', 1), (0.02, 'b', 2)]))
    ['a', 'b']
    >>> sorted(greedy([(0.0, 'a', 1), (0.01, 'b', 1)]))
    ['a']
    """
    hit, used = set(), set()
    for d, m, j in sorted(cands):
        if m not in hit and j not in used:
            hit.add(m)
            used.add(j)
    return hit


def cmd_score(a):
    out = os.path.join(rc.OUT, "score", a.label)
    marks = list(csv.DictReader(open(os.path.join(out, "region_marks_raw.csv"), encoding="utf-8")))
    pairs = list(csv.DictReader(open(os.path.join(out, "region_pairs.csv"), encoding="utf-8")))
    has_region = {(m["pano_id"], m["mark"]): m["region_found"] == "1" for m in marks}
    rules = {"radius": [], "region": [], "region_tol0": []}
    for p in pairs:
        k, j = (p["pano_id"], p["mark"]), (p["pano_id"], p["det"])
        cd = float(p["click_dist"])
        if cd <= RADIUS:
            rules["radius"].append((cd, k, j))
        if has_region[k]:
            if p["region_dist"] != "":
                rd = float(p["region_dist"])
                if rd <= TOL:
                    rules["region"].append((rd, cd, k, j))
                if rd == 0.0:
                    rules["region_tol0"].append((rd, cd, k, j))
        elif cd <= RADIUS:                       # no region: the radius rule
            rules["region"].append((cd, cd, k, j))
            rules["region_tol0"].append((cd, cd, k, j))
    hits = {}
    for name, c in rules.items():
        hits[name] = greedy([(t[:-2], t[-2], t[-1]) for t in c])
    for m in marks:
        k = (m["pano_id"], m["mark"])
        for name in rules:
            m["hit_" + name] = int(k in hits[name])
    with open(os.path.join(out, "region_marks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(marks[0]))
        w.writeheader()
        w.writerows(marks)

    def recall(sub, key):
        by = defaultdict(list)
        for m in sub:
            by[m["pano_id"]].append(m[key])
        k, n = sum(map(sum, by.values())), sum(map(len, by.values()))
        if not n:
            return {"marks": 0}
        rng, keys, boots = random.Random(rc.SEED), sorted(by), []
        for _ in range(rc.BOOT_N):
            pick = rng.choices(keys, k=len(keys))
            boots.append(sum(sum(by[p]) for p in pick) / sum(len(by[p]) for p in pick))
        boots.sort()
        return {"marks": n, "hit": k, "recall": round(k / n, 3), "panos": len(by),
                "ci95": [round(boots[int(0.025 * (rc.BOOT_N - 1))], 3), round(boots[int(0.975 * (rc.BOOT_N - 1))], 3)]}

    near = [m for m in marks if m["within_25m"] == "1"]
    groups = {"ALL": near}
    for m in near:
        groups.setdefault(m["city"], []).append(m)
        groups.setdefault(f'{m["city"]} | {m["camera_model"]}', []).append(m)
        groups.setdefault(f'{m["city"]} | near mining pano={m["flagged"]}', []).append(m)
        for lo, hi, name in BANDS:
            if lo <= float(m["y"]) < hi:
                groups.setdefault(f'{m["city"]} | {name}', []).append(m)
        groups.setdefault(f'{m["city"]} | region_found={m["region_found"]}', []).append(m)
    groups["beyond 25 m"] = [m for m in marks if m["within_25m"] == "0"]
    res = {"label": a.label, "tol": TOL, "close_px": CLOSE_PX, "radius": RADIUS, "groups": {
        g: {name: recall(sub, "hit_" + name) for name in rules} for g, sub in groups.items()}}
    json.dump(res, open(os.path.join(out, "region_recall.json"), "w", encoding="utf-8"), indent=1)
    for g, r in res["groups"].items():
        print(f'{g:48s} n={r["radius"].get("marks", 0):4d}  ' + "  ".join(
            f'{name} {x.get("hit", 0)} ({x.get("recall", "")}, {x.get("ci95", "")})' for name, x in r.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--verdicts", required=True)
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--native", nargs="+", required=True, help="city=<native pano dir>")
    p.add_argument("--label", required=True)
    p.add_argument("--threshold", type=float, required=True)
    s = sub.add_parser("score")
    s.add_argument("--label", required=True)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
