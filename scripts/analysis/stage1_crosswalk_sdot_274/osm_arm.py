#!/usr/bin/env python3
"""OSM marked crossings as a crosswalk label source for many cities (#285): placer checks.

The #285 placer is B2 from the Seattle SDOT pilot: project a crossing into a pano, then snap it
to the nearest Mapillary Vistas crosswalk component within 0.05 of the pano width
(``pilot.vistas_at``). A crossing with no component becomes an ignore region. So the placer can
only ever label crosswalks that the Vistas model sees.

``style-check`` asks which crosswalks those are. It runs the same snap at human-confirmed
crosswalk clicks, where the answer should be "yes, paint here", and reports the share with a
component, by crossing style (brick or paver, fading paint, other) and by imagery source. A
style with a low share cannot be supplied by OSM + Vistas, however well a city is tagged.

Usage (GPU; about 1 s per label on an RTX 3070)::

    python scripts/analysis/stage1_crosswalk_sdot_274/osm_arm.py style-check \
        --keep D:/Git/sal-scratch-keep
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402
import crowd_arm  # noqa: E402  (also puts the Phase 0 probe directory on sys.path)
import vistas_crosswalk_probe as probe  # noqa: E402

REPO = pilot.REPO
HARDNEG = os.path.join(REPO, "analysis_out", "stage1_crosswalk_detector_274", "hardneg")
OUT = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "osm_arm")
TOL_R = pilot.TOL_R     # 0.022, RampNet's matcher radius
SNAP_R = pilot.SNAP_R   # 0.05, the B2 snap radius

# label set -> (labels csv under hardneg/, 4096-px pano dir under --keep, imagery source)
LABEL_SETS = {
    "columbia_gt": ("s4_inputs/stepA/columbia_gt.csv", "s4-stepA/prep_4096", "gsv"),
    "richmond_positives": ("s4_inputs/stepA/richmond_positives.csv", "s4-stepA/prep_4096", "mapillary"),
    "ps_labels": ("confirm_s3/ps_labels/labels.csv", "crosswalk274-confirm/ps_label_4096", "mapillary"),
    "bayonne_labels": ("s4_bayonne/labels.csv", "s4-bayonne/prep_4096", "panoramax"),
}
TAG_FILES = ["s4_inputs/stepA/ps_s4_labels_2026-11-08.csv",
             "confirm_s3/ps_labels/ps_crosswalk_labels_2026-11-08.csv",
             "s4_bayonne/ps_bayonne_labels_2026-11-09.csv"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    if not n:
        return ["", ""]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def style_of(tags):
    """One style per label from its PS tags: brick wins over fading, since brick has no paint at all."""
    t = (tags or "").lower()
    if "brick" in t or "cobble" in t:
        return "brick_or_paver"
    if "fading" in t:
        return "paint_fading"
    return "other_or_untagged"


def load_tags():
    tags = {}
    for rel in TAG_FILES:
        path = os.path.join(HARDNEG, rel)
        if not os.path.exists(path):
            continue
        for r in read_csv(path):
            if "tags" in r and r.get("label_id"):
                tags[f'{r["city"]}:{r["label_id"]}'] = r["tags"]
    return tags


def cmd_style_check(a):
    tags = load_tags()
    labels = []
    for name, (rel, pano_dir, source) in LABEL_SETS.items():
        for r in read_csv(os.path.join(HARDNEG, rel)):
            uid = r["label_uid"]
            labels.append({"set": name, "source": source, "label_uid": uid, "city": uid.split(":")[0],
                           "pano_id": r["pano_id"], "x": float(r["x"]), "y": float(r["y"]),
                           "tags": tags.get(uid, ""), "tags_known": int(uid in tags),
                           "path": os.path.join(a.keep, pano_dir, r["pano_id"] + ".jpg")})
    labels.sort(key=lambda r: r["path"])
    seg = probe.Segmenter()
    rows, pano, pano_path = [], None, None
    for i, lab in enumerate(labels):
        if not os.path.exists(lab["path"]):
            rows.append({**lab, "pano_found": 0})
            continue
        if lab["path"] != pano_path:
            pano, pano_path = crowd_arm.load_pano(lab["path"]), lab["path"]
        v = pilot.vistas_at(seg, pano, lab["x"], lab["y"])
        near = v["near_px_dist"]
        rows.append({**lab, "pano_found": 1, "on_paint": v["on_paint"], "near_px_dist": near,
                     "paint_within_tol": int(near != "" and near <= TOL_R), "comp_found": v["comp_found"],
                     "shift": v["shift"], "comp_area_px": v["comp_area_px"]})
        if i % 25 == 0:
            pilot.log(f"  {i}/{len(labels)}")
    os.makedirs(OUT, exist_ok=True)
    cols = ["set", "source", "city", "label_uid", "pano_id", "x", "y", "tags", "tags_known", "style", "pano_found",
            "on_paint", "near_px_dist", "paint_within_tol", "comp_found", "shift", "comp_area_px"]
    for r in rows:
        r["style"] = style_of(r["tags"]) if r["tags_known"] else "tags_unknown"
    with open(os.path.join(OUT, "style_check_labels.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        if r["pano_found"]:
            groups[(r["set"], r["city"], r["source"], r["style"])].append(r)
            groups[(r["set"], r["city"], r["source"], "ALL")].append(r)
    summary = []
    for (s, city, source, style), rs in sorted(groups.items()):
        n = len(rs)
        row = {"set": s, "city": city, "source": source, "style": style, "labels": n}
        for key in ("on_paint", "paint_within_tol", "comp_found"):
            k = sum(int(r[key]) for r in rs)
            row[key] = k
            row[key + "_share"] = round(k / n, 3)
            row[key + "_ci95"] = json.dumps(wilson(k, n))
        summary.append(row)
    with open(os.path.join(OUT, "style_check_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    missing = sum(1 for r in rows if not r["pano_found"])
    print(f"{len(rows)} labels, {missing} with no local pano")
    for r in summary:
        print(f'  {r["set"]:20s} {r["city"]:12s} {r["style"]:18s} n={r["labels"]:3d}  on paint {r["on_paint_share"]:.2f}'
              f'  within {TOL_R} {r["paint_within_tol_share"]:.2f}  component within {SNAP_R} {r["comp_found_share"]:.2f} {r["comp_found_ci95"]}')


# ---- geometry check: B1 + B2 from OSM nodes against human clicks, on non-GSV panos --------------

NON_GSV_CAMERA_M = 2.6      # sidewalk-auto-labeler geo.DEFAULT_CAMERA_HEIGHT_M
NODE_RANGE_M = 25.0         # project crossing nodes this close to the pano (the labeler's raycast limit)
SAME_R = 0.10               # a node whose B1 point is this close to a click is taken to be that click's crossing


def osm_class(tags):
    """marked / unmarked / unknown, as pilot.osm_marked_nodes reads the tags."""
    m, c = tags.get("crossing:markings"), tags.get("crossing")
    if m is not None:
        return "unmarked" if m == "no" else "marked"
    if c in pilot.OSM_MARKED or c == "traffic_signals" or tags.get("crossing:signals") == "yes":
        return "marked"
    return "unmarked" if c in pilot.OSM_UNMARKED else "unknown"


def load_osm_nodes(census, city):
    out = {}
    d = os.path.join(census, "osm", city, "Crosswalk")
    if not os.path.isdir(d):        # a city the Phase 0 census did not pull (Juneau): no nodes, and the summary shows it
        pilot.log(f"no cached OSM crossings for {city}")
        return []
    for fn in sorted(os.listdir(d)):
        for e in json.load(open(os.path.join(d, fn), encoding="utf-8")).get("elements", []):
            t = e.get("tags") or {}
            if e.get("type") == "node" and t.get("highway") == "crossing":
                out[e["id"]] = (e["lat"], e["lon"], e["id"], osm_class(t))
    return list(out.values())


def cmd_geometry_check(a):
    """For labelled panos: every OSM crossing node within NODE_RANGE_M -> B1 -> B2, compared with the clicks.

    Clicks are not exhaustive, so "a node with no click near it" is not an error; the readings are
    conditional on a click being within SAME_R of the node's B1 point (the same crossing).
    """
    panos = {}
    for path in a.records:
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line).get("pano") or {}
                if p.get("lat") is not None and p.get("camera_heading") is not None:
                    panos[str(p["panorama_id"])] = p
    clicks = defaultdict(list)
    meta = {}
    for name, (rel, pano_dir, source) in LABEL_SETS.items():
        if source == "gsv":
            continue
        for r in read_csv(os.path.join(HARDNEG, rel)):
            city = r["label_uid"].split(":")[0]
            clicks[r["pano_id"]].append((float(r["x"]), float(r["y"]), r["label_uid"]))
            meta[r["pano_id"]] = (city, source, os.path.join(a.keep, pano_dir, r["pano_id"] + ".jpg"))
    nodes_by_city = {}
    seg = probe.Segmenter()
    node_rows, click_rows, coverage = [], [], defaultdict(lambda: [0, 0])
    for pid in sorted(clicks):
        city, source, path = meta[pid]
        coverage[city][1] += 1
        if pid not in panos or not os.path.exists(path):
            continue
        coverage[city][0] += 1
        if city not in nodes_by_city:
            nodes_by_city[city] = load_osm_nodes(a.census, city)
        p = panos[pid]
        pano = crowd_arm.load_pano(path)
        placed = []
        for lat, lng, nid, cls in nodes_by_city[city]:
            if abs(lat - p["lat"]) > 0.0005:
                continue
            x, y, d, _ = pilot.project(p["lat"], p["lng"], p["camera_heading"], lat, lng, camera_h=NON_GSV_CAMERA_M)
            if d > NODE_RANGE_M:
                continue
            v = pilot.vistas_at(seg, pano, x, y)
            b2 = (v["b2_x"], v["b2_y"]) if v["comp_found"] else None
            d1, lab = min((pilot.pano_dist(x, y, cx, cy), uid) for cx, cy, uid in clicks[pid])
            d2 = min(pilot.pano_dist(b2[0], b2[1], cx, cy) for cx, cy, _ in clicks[pid]) if b2 else ""
            placed.append((x, y, b2))
            node_rows.append({"city": city, "source": source, "pano_id": pid, "node_id": nid, "osm_class": cls,
                              "range_m": round(d, 1), "b1_x": round(x, 5), "b1_y": round(y, 5),
                              "comp_found": v["comp_found"], "b2_x": v["b2_x"], "b2_y": v["b2_y"],
                              "nearest_click": lab, "b1_to_click": round(d1, 5),
                              "b2_to_click": round(d2, 5) if d2 != "" else ""})
        for cx, cy, uid in clicks[pid]:
            b1 = min((pilot.pano_dist(x, y, cx, cy) for x, y, _ in placed), default="")
            b2 = min((pilot.pano_dist(b[0], b[1], cx, cy) for _, _, b in placed if b), default="")
            click_rows.append({"city": city, "source": source, "pano_id": pid, "label_uid": uid,
                               "nodes_in_range": len(placed),
                               "nearest_b1": round(b1, 5) if b1 != "" else "",
                               "nearest_b2": round(b2, 5) if b2 != "" else ""})
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("geometry_nodes.csv", node_rows), ("geometry_clicks.csv", click_rows)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def share(k, n):
        return {"k": k, "n": n, "share": round(k / n, 3) if n else "", "ci95": wilson(k, n)}
    summary = {"camera_height_m": NON_GSV_CAMERA_M, "node_range_m": NODE_RANGE_M, "same_crossing_r": SAME_R,
               "panos_with_record_and_image": {c: f"{k} of {n}" for c, (k, n) in coverage.items()}, "cities": {}}
    for city in sorted({r["city"] for r in click_rows}):
        cr = [r for r in click_rows if r["city"] == city]
        nr = [r for r in node_rows if r["city"] == city]
        same = [r for r in nr if r["b1_to_click"] <= SAME_R]
        snapped = [r for r in same if r["comp_found"]]
        summary["cities"][city] = {
            "clicks": len(cr),
            "clicks_with_an_osm_node_b1_within_0.10": share(sum(r["nearest_b1"] != "" and r["nearest_b1"] <= SAME_R for r in cr), len(cr)),
            "clicks_with_a_b2_within_0.05": share(sum(r["nearest_b2"] != "" and r["nearest_b2"] <= SNAP_R for r in cr), len(cr)),
            "clicks_with_a_b2_within_0.022": share(sum(r["nearest_b2"] != "" and r["nearest_b2"] <= TOL_R for r in cr), len(cr)),
            "nodes_projected": len(nr),
            "nodes_on_a_clicked_crossing": len(same),
            "of_those_b1_within_0.022_of_click": share(sum(r["b1_to_click"] <= TOL_R for r in same), len(same)),
            "of_those_component_found": share(len(snapped), len(same)),
            "of_snapped_b2_within_0.022_of_click": share(sum(r["b2_to_click"] <= TOL_R for r in snapped), len(snapped)),
            "of_snapped_b2_within_0.05_of_click": share(sum(r["b2_to_click"] <= SNAP_R for r in snapped), len(snapped)),
            "median_b1_to_click": round(sorted(r["b1_to_click"] for r in same)[len(same) // 2], 4) if same else "",
            "median_b2_to_click": round(sorted(r["b2_to_click"] for r in snapped)[len(snapped) // 2], 4) if snapped else "",
        }
    with open(os.path.join(OUT, "geometry_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry-check", help="B1 + B2 from OSM nodes vs human clicks on Mapillary / Panoramax panos")
    g.add_argument("--keep", required=True)
    g.add_argument("--census", required=True, help="Phase 0 census out/ dir (osm/<city>/Crosswalk/*.json)")
    g.add_argument("--records", nargs="+", required=True, help="labeler records with lat, lng, camera_heading")
    s = sub.add_parser("style-check", help="Vistas component-exists at human-confirmed crosswalk clicks, by style")
    s.add_argument("--keep", required=True, help="the local scratch root holding the 4096-px pano dirs")
    a = ap.parse_args()
    {"style-check": cmd_style_check, "geometry-check": cmd_geometry_check}[a.cmd](a)


if __name__ == "__main__":
    main()
