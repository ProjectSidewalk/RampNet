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
import math
import os
import random
import sys
from collections import Counter, defaultdict

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


# ---- city pilot: place OSM marked crossings on fetched open-imagery panos (inputs from osm_city.py) ----

PLACE_RANGE_M = 20.0        # the plan's rule: a marked crossing within 20 m of the pano
COLLIDE_R = crowd_arm.COLLIDE_R
A3_N = 100                  # the per-city precision sample
GALLERY_N = 60
SEED = 285
PLACE_COLS = ["city", "pano_id", "node_id", "basis", "range_m", "b1_x", "b1_y", "on_paint", "near_px_dist",
              "comp_found", "b2_x", "b2_y", "shift", "comp_area_px", "capture_date", "camera_make", "camera_model",
              "sequence_id"]


def city_dirs(a):
    return os.path.join(OUT, "cities", a.city), os.path.join(a.keep, "osm285", a.city)


def cmd_city_place(a):
    """B1 (2.6 m camera) + B2 for every marked, at-grade crossing within PLACE_RANGE_M of each fetched pano."""
    out, keep = city_dirs(a)
    views = os.path.join(keep, "views")
    os.makedirs(views, exist_ok=True)
    nodes = defaultdict(list)       # 0.001-degree latitude band -> crossings
    for r in read_csv(os.path.join(out, "crossings.csv")):
        if r["osm_class"] == "marked" and r["grade_separated"] == "0":
            nodes[int(float(r["lat"]) * 1000)].append((float(r["lat"]), float(r["lng"]), r["node_id"], r["basis"]))
    panos = [json.loads(line)["pano"] for line in open(os.path.join(keep, "records.jsonl"), encoding="utf-8") if line.strip()]
    panos.sort(key=lambda p: p["panorama_id"])
    done_path, pl_path = os.path.join(out, "placed_panos.txt"), os.path.join(out, "placements.csv")
    done = {l.strip() for l in open(done_path, encoding="utf-8")} if os.path.exists(done_path) else set()
    new = not os.path.exists(pl_path)
    seg = probe.Segmenter()
    with open(pl_path, "a", encoding="utf-8", newline="") as fh, open(done_path, "a", encoding="utf-8") as dh:
        w = csv.DictWriter(fh, fieldnames=PLACE_COLS, lineterminator="\n")
        if new:
            w.writeheader()
        for k, p in enumerate(panos):
            pid = p["panorama_id"]
            path = os.path.join(a.keep, "pano_store", a.city, pid[:2], pid + ".jpg")   # osm_city.py store
            if pid in done or not os.path.exists(path):
                continue
            band = int(p["lat"] * 1000)
            near = []
            for lat, lng, nid, basis in nodes[band - 1] + nodes[band] + nodes[band + 1]:
                x, y, d, _ = pilot.project(p["lat"], p["lng"], p["camera_heading"], lat, lng, camera_h=NON_GSV_CAMERA_M)
                if d <= PLACE_RANGE_M:
                    near.append((x, y, d, nid, basis))
            pano = crowd_arm.load_pano(path) if near else None
            for x, y, d, nid, basis in near:
                v = pilot.vistas_at(seg, pano, x, y)
                w.writerow({"city": a.city, "pano_id": pid, "node_id": nid, "basis": basis, "range_m": round(d, 1),
                            "b1_x": round(x, 6), "b1_y": round(y, 6), "on_paint": v["on_paint"],
                            "near_px_dist": v["near_px_dist"], "comp_found": v["comp_found"], "b2_x": v["b2_x"],
                            "b2_y": v["b2_y"], "shift": v["shift"], "comp_area_px": v["comp_area_px"],
                            "capture_date": p["capture_date"], "camera_make": p.get("camera_make") or "",
                            "camera_model": p.get("camera_model") or "", "sequence_id": p.get("sequence_id") or ""})
                pilot.thumbnail(v).save(os.path.join(views, f"{pid}__{nid}.jpg"), quality=85)
            fh.flush()
            dh.write(pid + "\n")
            dh.flush()
            if k % 25 == 0:
                pilot.log(f"  pano {k + 1}/{len(panos)}")


def city_kinds(rows):
    """Each placement's kind: `positive` (snapped, and the first claim on its paint), `ignore_collided` (snapped
    onto paint another crossing on the same pano already labels) or `ignore_no_component`."""
    by_pano = defaultdict(list)
    for r in rows:
        by_pano[r["pano_id"]].append(r)
    for rs in by_pano.values():
        hit = set(crowd_arm.sdot_collisions(
            [(r["b2_x"], r["b2_y"], r["shift"]) if r["comp_found"] == "1" else None for r in rs], [], [], r=COLLIDE_R))
        for i, r in enumerate(rs):
            r["kind"] = "ignore_no_component" if r["comp_found"] != "1" else \
                "ignore_collided" if i in hit else "positive"
    return rows


def cmd_city_report(a):
    """Counts, the seeded A3 precision sample (a `score_274.py cards` candidates file) and an eyeball gallery."""
    import html
    import random
    out, keep = city_dirs(a)
    rows = city_kinds(read_csv(os.path.join(out, "placements.csv")))
    with open(os.path.join(out, "placements_kinds.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLACE_COLS + ["kind"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    placed = sum(1 for _ in open(os.path.join(out, "placed_panos.txt"), encoding="utf-8"))

    def split(key):
        g = defaultdict(lambda: [0, 0])
        for r in rows:
            g[key(r)][1] += 1
            g[key(r)][0] += r["kind"] == "positive"
        return {k: {"positive": p, "placements": n, "share": round(p / n, 3)} for k, (p, n) in sorted(g.items())}
    pos = [r for r in rows if r["kind"] == "positive"]
    summary = {"city": a.city, "place_range_m": PLACE_RANGE_M, "camera_height_m": NON_GSV_CAMERA_M,
               "panos_placed": placed, "panos_with_a_crossing_in_range": len({r["pano_id"] for r in rows}),
               "placements": len(rows), "kinds": {k: sum(r["kind"] == k for r in rows)
                                                  for k in ("positive", "ignore_collided", "ignore_no_component")},
               "panos_with_a_positive": len({r["pano_id"] for r in pos}),
               "crossings_with_a_positive": len({r["node_id"] for r in pos}),
               "by_basis": split(lambda r: r["basis"].split("=")[0]),
               "by_range": split(lambda r: "under 10 m" if float(r["range_m"]) < 10 else "10 to 20 m"),
               "by_camera": split(lambda r: (r["camera_make"] + " " + r["camera_model"]).strip() or "unknown"),
               "by_capture_year": split(lambda r: r["capture_date"][:4])}
    with open(os.path.join(out, "placements_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    pos.sort(key=lambda r: (r["pano_id"], r["node_id"]))
    a3 = random.Random(SEED).sample(pos, min(A3_N, len(pos)))
    with open(os.path.join(out, "a3_candidates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pano_id", "x", "y", "conf", "city", "node_id", "basis"], lineterminator="\n")
        w.writeheader()
        for r in a3:
            w.writerow({"pano_id": r["pano_id"], "x": r["b2_x"], "y": r["b2_y"], "conf": 1.0, "city": a.city,
                        "node_id": r["node_id"], "basis": r["basis"]})
    allr = sorted(rows, key=lambda r: (r["pano_id"], r["node_id"]))
    pick = sorted(random.Random(SEED + 2).sample(allr, min(GALLERY_N, len(allr))), key=lambda r: float(r["range_m"]))
    cards = []
    for r in pick:
        cap = (f"node {r['node_id']} · {r['basis']} · {float(r['range_m']):.1f} m · {r['capture_date']} · "
               f"{html.escape(r['camera_model'] or '?')} · {r['kind']}")
        cards.append(f'<figure><img src="views/{r["pano_id"]}__{r["node_id"]}.jpg" alt="View toward crossing node '
                     f'{r["node_id"]} from pano {html.escape(r["pano_id"])}" loading="lazy">'
                     f'<figcaption>{cap}</figcaption></figure>')
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OSM crossing placements: {html.escape(a.city)}</title>
  <style>
    :root {{ --bg: #fafafa; --fg: #222; --muted: #555; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #161616; --fg: #e8e8e8; --muted: #aaa; }} }}
    body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(256px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>OSM crossing placements: {html.escape(a.city)}</h1>
  <p>{len(pick)} seeded placements of every kind, nearest first. Red ring: the OSM crossing node projected into the
  pano. Cyan cross: the centroid of the segmenter's crosswalk component it snaps to (the training label). Yellow
  tint: the segmenter's crosswalk pixels. Each view is 90° wide, centred on the red ring.</p>
  <div class="grid">
    {chr(10).join(cards)}
  </div>
</body>
</html>
"""
    with open(os.path.join(keep, "gallery.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    pilot.log(f"A3 sample: {len(a3)} of {len(pos)} positives -> a3_candidates.csv; gallery -> {os.path.join(keep, 'gallery.html')}")


def cmd_city_score(a):
    """Score a rated A3 page: correct = verdict 'crosswalk'; unsure counts as not correct (the written gate)."""
    out = os.path.join(OUT, "cities", a.city)
    key = lambda r: (r["pano_id"], round(float(r["x"]), 4), round(float(r["y"]), 4))  # noqa: E731
    cand = {key(r): r for r in read_csv(os.path.join(out, "a3_candidates.csv"))}
    place = {(r["pano_id"], r["node_id"]): r for r in read_csv(os.path.join(out, "placements_kinds.csv"))}
    rated = read_csv(a.ratings)
    assert len({key(r) for r in rated}) == len(rated) == len(cand), "ratings do not cover the sample one-to-one"
    splits, wrong = defaultdict(lambda: [0, 0]), []
    for r in rated:
        c = cand[key(r)]
        p = place[(c["pano_id"], c["node_id"])]
        good = r["verdict"] == "crosswalk"
        for g in ("all", "basis: " + c["basis"],
                  "range: under 10 m" if float(p["range_m"]) < 10 else "range: 10 to 20 m",
                  "projected point on paint" if p["on_paint"] == "1" else "projected point off paint"):
            splits[g][0] += good
            splits[g][1] += 1
        if not good:
            wrong.append({"pano_id": c["pano_id"], "node_id": c["node_id"], "verdict": r["verdict"],
                          "basis": c["basis"], "range_m": p["range_m"], "comp_area_px": p["comp_area_px"],
                          "note": r["note"]})
    k, n = splits["all"]
    lo, hi = wilson(k, n)
    res = {"city": a.city, "ratings": os.path.basename(a.ratings), "correct": k, "n": n,
           "wilson": [lo, hi], "bar": a.bar, "passes": k >= a.bar,
           "verdicts": {v: sum(r["verdict"] == v for r in rated) for v in sorted({r["verdict"] for r in rated})},
           "splits": {g: v for g, v in sorted(splits.items())}, "not_correct": wrong}
    with open(os.path.join(out, "a3_score.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


MIN_REGION_PX = 100         # crosswalk_s6: a positive's snapped region must be larger than this (4096-wide pano)
MAX_DIP_DEG = 40.0          # ... and the label at most this far below the horizon
MERGE_R = 0.05              # two positives on one pano closer than this are one crossing (s5's rule)


def dip_deg(y):
    """Degrees below the horizon of a normalized image height.

    >>> dip_deg(0.5), round(dip_deg(0.75), 1)
    (0.0, 45.0)
    """
    return (y - 0.5) * 180.0


def cmd_city_export(a):
    """The `crosswalk_s6` label rules (docs/crosswalk_followups_285.md, section A4) applied to each city's
    placements: positives, ignore points, pano positions and the seeded half dose, as inputs for
    `export_stage2.py add-panos`. Reads only committed files plus each city's records.jsonl (positions)."""
    out = os.path.join(OUT, "root_v6")
    os.makedirs(out, exist_ok=True)
    key = lambda r: (r["pano_id"], round(float(r["x"]), 4), round(float(r["y"]), 4))  # noqa: E731
    pos_rows, ign_rows, position_rows, summary, half = [], [], [], {}, set()
    n_collided_skipped = Counter()
    for city in a.cities:
        cdir = os.path.join(OUT, "cities", city)
        rows = read_csv(os.path.join(cdir, "placements_kinds.csv"))
        cand = {key(r): (r["pano_id"], r["node_id"]) for r in read_csv(os.path.join(cdir, "a3_candidates.csv"))}
        ratings = sorted(f for f in os.listdir(os.path.join(cdir, "ratings")) if f.endswith(".csv"))
        assert len(ratings) == 1, f"{city}: expected one ratings CSV, found {ratings}"
        wrong = {cand[key(r)] for r in read_csv(os.path.join(cdir, "ratings", ratings[0]))
                 if r["verdict"] != "crosswalk"}
        n = Counter()
        by_pano = defaultdict(list)
        for r in rows:
            n["placements"] += 1
            why = None
            if r["kind"] != "positive":
                why = r["kind"]
            else:
                n["positive"] += 1
                small = float(r["comp_area_px"]) <= MIN_REGION_PX
                low = dip_deg(float(r["b2_y"])) > MAX_DIP_DEG
                n["region_small"] += small
                n["too_low"] += low
                n["region_small_and_too_low"] += small and low
                if small or low:
                    why = "cut"
                elif (r["pano_id"], r["node_id"]) in wrong:
                    n["rated_not_correct"] += 1
                    why = "rated_not_correct"
            r["_why"] = why
            by_pano[r["pano_id"]].append(r)
        n["kept_before_merge"] = sum(r["_why"] is None for r in rows)
        n["panos_before_merge"] = sum(any(r["_why"] is None for r in rs) for rs in by_pano.values())
        kept_panos = []
        for pid in sorted(by_pano):
            keep = []
            for r in sorted((r for r in by_pano[pid] if r["_why"] is None),
                            key=lambda r: (-float(r["comp_area_px"]), r["node_id"])):
                x, y = float(r["b2_x"]), float(r["b2_y"])
                dx = lambda q: min(abs(x - q), 1.0 - abs(x - q))  # noqa: E731
                if any(math.hypot(dx(kx), y - ky) < MERGE_R for kx, ky in keep):
                    r["_why"] = "merged"
                    n["merged"] += 1
                else:
                    keep.append((x, y))
            if not keep:
                continue
            kept_panos.append(pid)
            for r in by_pano[pid]:
                uid = f'osm:{city}:{r["node_id"]}@{pid}'
                if r["_why"] is None:
                    pos_rows.append({"label_uid": uid, "city": city, "pano_id": pid, "x": r["b2_x"], "y": r["b2_y"]})
                elif r["_why"] == "ignore_collided":
                    # Amendment 2026-11-13: no ignore point. A collided placement sits on paint that the first
                    # claim on this pano already covers (a positive, or that claim's own ignore point); 109 of
                    # the 249 were within 30 px of a kept positive and zeroed its loss weight.
                    n_collided_skipped[city] += 1
                else:
                    snapped = r["comp_found"] == "1"
                    ign_rows.append({"ignore_id": uid, "pano_id": pid, "why": r["_why"],
                                     "x": r["b2_x"] if snapped else r["b1_x"], "y": r["b2_y"] if snapped else r["b1_y"]})
        n["positives"] = sum(1 for r in pos_rows if r["city"] == city)
        n["panos"] = len(kept_panos)
        n["ignore_points"] = sum(1 for r in ign_rows if r["ignore_id"].startswith(f"osm:{city}:"))
        n["collided_no_ignore_point"] = n_collided_skipped[city]
        picked = set(random.Random(SEED).sample(kept_panos, len(kept_panos) // 2))
        half |= picked
        n["half_panos"] = len(picked)
        n["half_positives"] = sum(1 for r in pos_rows if r["pano_id"] in picked)
        keep_dir = os.path.join(a.keep, "osm285", city)
        want = set(kept_panos)
        for line in open(os.path.join(keep_dir, "records.jsonl"), encoding="utf-8"):
            if line.strip():
                q = json.loads(line)["pano"]
                if q["panorama_id"] in want:
                    position_rows.append({"pano_id": q["panorama_id"], "lat": q["lat"], "lng": q["lng"],
                                          "capture_date": q.get("capture_date") or ""})
        summary[city] = {"ratings": ratings[0], **dict(n)}

    def write(name, rows_, cols):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows_)

    pcols = ["label_uid", "city", "pano_id", "x", "y"]
    write("positives_full.csv", pos_rows, pcols)
    write("positives_half.csv", [r for r in pos_rows if r["pano_id"] in half], pcols)
    write("ignore.csv", ign_rows, ["ignore_id", "pano_id", "why", "x", "y"])
    write("positions.csv", position_rows, ["pano_id", "lat", "lng", "capture_date"])
    panos = sorted({r["pano_id"] for r in pos_rows})
    assert len(position_rows) == len(panos), "a kept pano has no record"
    for name, ids in (("panos_full.txt", panos), ("panos_half.txt", sorted(half))):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{pid}.jpg\n" for pid in ids)
    tot = Counter()
    for c in summary.values():
        tot.update({k: v for k, v in c.items() if k != "ratings"})
    res = {"rules": {"min_region_px_exclusive": MIN_REGION_PX, "max_dip_deg": MAX_DIP_DEG, "merge_r": MERGE_R,
                     "half_seed": SEED}, "cities": summary, "all": dict(tot)}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("city-place", "B1 + B2 for OSM marked crossings on a city's fetched panos (GPU)"),
                        ("city-report", "placement counts, the A3 precision sample and an eyeball gallery")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--city", required=True)
        c.add_argument("--keep", required=True, help="scratch root; panos are in <keep>/osm285/<city>/")
    c = sub.add_parser("city-score", help="score a rated A3 precision page against the written bar")
    c.add_argument("--city", required=True)
    c.add_argument("--ratings", required=True, help="the page's exported CSV")
    c.add_argument("--bar", type=int, default=91)
    g = sub.add_parser("geometry-check", help="B1 + B2 from OSM nodes vs human clicks on Mapillary / Panoramax panos")
    g.add_argument("--keep", required=True)
    g.add_argument("--census", required=True, help="Phase 0 census out/ dir (osm/<city>/Crosswalk/*.json)")
    g.add_argument("--records", nargs="+", required=True, help="labeler records with lat, lng, camera_heading")
    s = sub.add_parser("style-check", help="Vistas component-exists at human-confirmed crosswalk clicks, by style")
    s.add_argument("--keep", required=True, help="the local scratch root holding the 4096-px pano dirs")
    e = sub.add_parser("city-export", help="crosswalk_s6: positives, ignore points and the half dose for add-panos")
    e.add_argument("--cities", nargs="+", required=True)
    e.add_argument("--keep", required=True, help="scratch root; positions come from <keep>/osm285/<city>/records.jsonl")
    a = ap.parse_args()
    {"style-check": cmd_style_check, "geometry-check": cmd_geometry_check,
     "city-place": cmd_city_place, "city-report": cmd_city_report,
     "city-score": cmd_city_score, "city-export": cmd_city_export}[a.cmd](a)


if __name__ == "__main__":
    main()
