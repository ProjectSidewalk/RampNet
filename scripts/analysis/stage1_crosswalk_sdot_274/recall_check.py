#!/usr/bin/env python3
"""Missed-crosswalk check for the crosswalk detector on open imagery (#285 follow-up C).

Question (per view): of the marked crosswalks a person can see in a pano within 25 m, what share
does the detector find? Look-alike rates are measured on random panos; this measures misses.

``draw`` fixes the sample before any page is built or any pixel fetched:

- Intersections come from the OSM ROAD GRAPH (the labeler's cached street query), never from OSM
  crossing tags, so the sample does not depend on how well a city's crossings are mapped.
  A node is an intersection when at least 3 street segments meet there; nodes within MERGE_M are
  merged, so a divided road or a staggered junction counts once.
- One pano per intersection: the nearest pano within NEAR_M that is not in an earlier draw.
  Exclusion is by draw membership (lists of pano ids), never by "has a rated card": cards exist
  only where a model fired, and excluding those panos would bias recall down.
- ``--take all`` keeps every intersection; ``--take N --all-of "GoPro Max"`` keeps every
  intersection whose pano has that camera model and a seeded sample of the rest up to N.

Usage::

    python scripts/analysis/stage1_crosswalk_sdot_274/recall_check.py draw --city laurens-ia \
        --streets <labeler>/runs/laurens/osm_streets.json --records <labeler>/runs/laurens/results.raw.jsonl \
        --exclude <hardneg>/confirm_s3/cities_draw.txt <hardneg>/s4_inputs/stepA/laurens_mine_draw.txt --take all
"""
import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "recall")

SEED = 285
MERGE_M = 20.0     # intersection nodes this close are one intersection
NEAR_M = 15.0      # a pano must be this close to the (merged) intersection
MIN_DEGREE = 3


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def read_ids(path):
    """Pano ids from a draw list: one per line, or a CSV with a pano_id column; '.jpg' is stripped."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines and lines[0].split(",")[0] == "pano_id":
        lines = [ln.split(",")[0] for ln in lines[1:]]
    return {ln[:-4] if ln.endswith(".jpg") else ln for ln in lines}


def intersections(streets_path):
    """Merged intersections of the cached street query: [{lat, lng, degree, n_nodes}]."""
    ways = [e for e in json.load(open(streets_path, encoding="utf-8"))["elements"] if e.get("type") == "way"]
    degree, pos = Counter(), {}
    for w in ways:
        nodes, geom = w["nodes"], w["geometry"]
        closed = nodes[0] == nodes[-1]
        for i, (n, g) in enumerate(zip(nodes, geom)):
            pos[n] = (g["lat"], g["lon"])
            if closed and i == len(nodes) - 1:
                continue
            ends = i in (0, len(nodes) - 1) and not closed
            degree[n] += 1 if ends else 2
    nodes = sorted(n for n, d in degree.items() if d >= MIN_DEGREE)
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if haversine_m(*pos[a], *pos[b]) <= MERGE_M:
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    out = []
    for root in sorted(groups):
        ns = groups[root]
        out.append({"intersection_id": f"osm:{min(ns)}", "lat": sum(pos[n][0] for n in ns) / len(ns),
                    "lng": sum(pos[n][1] for n in ns) / len(ns), "degree": max(degree[n] for n in ns),
                    "n_nodes": len(ns)})
    return out


def load_panos(records):
    panos = {}
    for path in records:
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line).get("pano") or {}
                if p.get("lat") is None:
                    continue
                panos[str(p["panorama_id"])] = {
                    "pano_id": str(p["panorama_id"]), "lat": p["lat"], "lng": p["lng"],
                    "camera_model": p.get("camera_model") or "", "sequence_id": p.get("sequence_id") or "",
                    "capture_date": p.get("capture_date") or ""}
    return panos


def cmd_draw(a):
    inter = intersections(a.streets)
    panos = load_panos(a.records)
    excluded = set()
    for path in a.exclude or []:
        excluded |= read_ids(path)
    flagged = set()
    for path in a.flag or []:
        flagged |= read_ids(path)
    plist = list(panos.values())
    rows, used = [], set()
    for it in inter:
        near = sorted((haversine_m(it["lat"], it["lng"], p["lat"], p["lng"]), p["pano_id"]) for p in plist
                      if abs(p["lat"] - it["lat"]) < 0.0005)
        near = [(d, pid) for d, pid in near if d <= NEAR_M]
        ok = [(d, pid) for d, pid in near if pid not in excluded and pid not in used]
        row = {**it, "lat": round(it["lat"], 7), "lng": round(it["lng"], 7), "panos_within": len(near),
               "panos_excluded": sum(pid in excluded for _, pid in near)}
        if ok:
            d, pid = ok[0]
            used.add(pid)
            p = panos[pid]
            row.update({"status": "pano", "pano_id": pid, "dist_m": round(d, 1), "rank_taken": near.index((d, pid)) + 1,
                        "camera_model": p["camera_model"], "sequence_id": p["sequence_id"],
                        "capture_date": p["capture_date"], "flagged": int(pid in flagged)})
        else:
            row["status"] = "no_pano_within" if not near else "all_near_panos_excluded"
        rows.append(row)

    cand = [r for r in rows if r["status"] == "pano"]
    if a.take == "all":
        chosen = {r["intersection_id"] for r in cand}
    else:
        n = int(a.take)
        must = [r for r in cand if a.all_of and r["camera_model"] == a.all_of]
        rest = [r for r in cand if r not in must]
        random.Random(SEED).shuffle(rest)
        chosen = {r["intersection_id"] for r in must + rest[:max(0, n - len(must))]}
    for r in rows:
        r["in_draw"] = int(r["intersection_id"] in chosen)

    out = os.path.join(OUT, a.city)
    os.makedirs(out, exist_ok=True)
    cols = ["intersection_id", "lat", "lng", "degree", "n_nodes", "panos_within", "panos_excluded", "status",
            "in_draw", "pano_id", "dist_m", "rank_taken", "camera_model", "sequence_id", "capture_date", "flagged"]
    with open(os.path.join(out, "intersections.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    drawn = [r for r in rows if r["in_draw"]]
    with open(os.path.join(out, "draw.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(r["pano_id"] + "\n" for r in drawn))
    meta = {
        "city": a.city, "seed": SEED, "merge_m": MERGE_M, "near_m": NEAR_M, "min_degree": MIN_DEGREE,
        "take": a.take, "all_of": a.all_of or "", "streets": os.path.basename(a.streets),
        "records": [os.path.basename(p) for p in a.records], "exclude": [os.path.basename(p) for p in a.exclude or []],
        "flag": [os.path.basename(p) for p in a.flag or []], "panos_in_records": len(panos),
        "panos_excluded_by_draw": len(excluded & set(panos)), "intersections": len(rows),
        "status": dict(Counter(r["status"] for r in rows)), "drawn": len(drawn),
        "drawn_by_camera": dict(Counter(r["camera_model"] or "none" for r in drawn)),
        "drawn_not_nearest": sum(r["rank_taken"] > 1 for r in drawn),
        "drawn_flagged": sum(r["flagged"] for r in drawn),
        "drawn_sequences": len({r["sequence_id"] for r in drawn}),
    }
    with open(os.path.join(out, "draw_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


# ---- the blind marking page ------------------------------------------------------------------

BANNER = """<div style="background:#fff8c5;border:2px solid #9a6700;padding:10px 14px;margin:8px 0;font-size:15px;line-height:1.45">
<b>Missed-crosswalk check (#285). There are no detections on this page.</b><br>
For each panorama: click the <b>centre of every marked crosswalk you can see</b> (painted, brick or paver all count).
If there are none, press <b>m</b>. Click a mark again to flag it unsure, and once more to remove it.
Mark what you can recognise as a crosswalk without zooming all the way in; do not hunt for specks at the horizon.
Please note the time after the first 10 panoramas.
</div>"""

# The gallery's visible wording is about curb ramps; this page asks about crosswalks. Applied in order.
REWORD = [("curb ramps", "marked crosswalks"), ("curb ramp", "marked crosswalk"),
          ("missed-ramp", "crosswalk"), ("missed ramps", "crosswalks"), ("missed ramp", "crosswalk"),
          ("ramps", "crosswalks"), ("ramp", "crosswalk")]


def cmd_page(a):
    """Bundle (records with detections stripped + panos) -> gt_gallery page -> crosswalk wording."""
    import shutil
    import subprocess
    import sys
    os.makedirs(os.path.join(a.bundle, "panos"), exist_ok=True)
    n = 0
    with open(os.path.join(a.bundle, "records.jsonl"), "w", encoding="utf-8", newline="\n") as out:
        for spec in a.draw:
            city, rest = spec.split("=", 1)
            records, native = rest.split("::")
            files = {os.path.splitext(fn)[0]: fn for fn in os.listdir(native)}
            with open(records, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    pid = str(r["pano"]["panorama_id"])
                    if pid not in files:
                        raise SystemExit(f"{city}: no image for {pid} in {native}")
                    dst = os.path.join(a.bundle, "panos", files[pid])
                    if not os.path.exists(dst):
                        try:
                            os.link(os.path.join(native, files[pid]), dst)
                        except OSError:
                            shutil.copy2(os.path.join(native, files[pid]), dst)
                    r["detections"] = []          # blind: the page shows no model output
                    r["benchmark_group"] = city
                    out.write(json.dumps(r) + "\n")
                    n += 1
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "gt_gallery.py"), a.bundle, "--out", a.out,
                    "--hide-credit"], check=True)
    index = os.path.join(a.out, "index.html")
    html = open(index, encoding="utf-8").read()
    for old, new in REWORD:
        html = html.replace(old, new).replace(old.capitalize(), new.capitalize())
    anchor = '<div id="intro">'      # the gallery's own two-sided instructions, which do not apply here
    if anchor not in html:
        raise SystemExit("gt_gallery's page has no intro block to put the instructions before")
    html = html.replace(anchor, BANNER + "<style>#intro{display:none}</style>" + anchor, 1)
    with open(index, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{n} panos -> {index}")


# ---- scoring ---------------------------------------------------------------------------------

RADII = (0.05, 0.03, 0.08)      # the first is the primary
Y_25M = 0.532                   # 25 m away on flat ground from a 2.5 m camera; the labeler's raycast limit
BOOT_N = 2000


def wrap_dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    return math.hypot(min(dx, 1.0 - dx), y1 - y2)


def match_one_to_one(marks, dets, radius):
    """Greedy nearest-first one-to-one matching; returns the set of hit mark indices.

    >>> sorted(match_one_to_one([(0.10, 0.6), (0.12, 0.6)], [(0.11, 0.6)], 0.05))
    [0]
    >>> sorted(match_one_to_one([(0.99, 0.6)], [(0.01, 0.6)], 0.05))
    [0]
    >>> sorted(match_one_to_one([(0.2, 0.6)], [(0.3, 0.6)], 0.05))
    []
    """
    pairs = sorted((wrap_dist(mx, my, dx, dy), i, j) for i, (mx, my) in enumerate(marks)
                   for j, (dx, dy) in enumerate(dets))
    hit, used = set(), set()
    for d, i, j in pairs:
        if d > radius:
            break
        if i not in hit and j not in used:
            hit.add(i)
            used.add(j)
    return hit


def cmd_score(a):
    v = json.load(open(a.verdicts, encoding="utf-8"))["panos"]
    preds = {}
    for path in a.preds:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                preds[r["pano_id"]] = r["preds"]
    info = {}
    for city in a.cities:
        for r in csv.DictReader(open(os.path.join(OUT, city, "intersections.csv"), encoding="utf-8")):
            if r["in_draw"] == "1":
                info[r["pano_id"]] = {"city": city, "camera_model": r["camera_model"], "flagged": r["flagged"]}
    rows, skipped = [], Counter()
    for pid, e in v.items():
        marks = e.get("missed", [])
        if not marks and not e.get("no_missed"):
            skipped["seen_but_not_confirmed"] += 1
            continue
        if pid not in preds or pid not in info:
            skipped["no_preds_or_not_in_draw"] += 1
            continue
        dets = [(x, y) for x, y, c in preds[pid] if c >= a.threshold]
        sure = [(m["x"], m["y"]) for m in marks if not m.get("unsure")]
        hits = {r: match_one_to_one(sure, dets, r) for r in RADII}
        for i, (x, y) in enumerate(sure):
            rows.append({"pano_id": pid, **info[pid], "x": round(x, 4), "y": round(y, 4),
                         "within_25m": int(y >= Y_25M), **{f"hit_{r}": int(i in hits[r]) for r in RADII}})
        if not sure:
            rows.append({"pano_id": pid, **info[pid], "x": "", "y": "", "within_25m": "",
                         **{f"hit_{r}": "" for r in RADII}})
    if not rows:
        raise SystemExit(f"nothing to score ({dict(skipped)})")
    out = os.path.join(OUT, "score", a.label)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "marks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def recall(sub, key):
        by = defaultdict(list)
        for r in sub:
            by[r["pano_id"]].append(r[key])
        k, n = sum(map(sum, by.values())), sum(map(len, by.values()))
        if not n:
            return {"marks": 0}
        rng, keys, boots = random.Random(SEED), sorted(by), []
        for _ in range(BOOT_N):
            pick = rng.choices(keys, k=len(keys))
            boots.append(sum(sum(by[p]) for p in pick) / sum(len(by[p]) for p in pick))
        boots.sort()
        return {"marks": n, "hit": k, "recall": round(k / n, 3), "panos_with_marks": len(by),
                "ci95": [round(boots[int(0.025 * (BOOT_N - 1))], 3), round(boots[int(0.975 * (BOOT_N - 1))], 3)]}

    marks = [r for r in rows if r["x"] != ""]
    res = {"label": a.label, "threshold": a.threshold, "panos_scored": len({r["pano_id"] for r in rows}),
           "panos_with_no_crosswalk": len({r["pano_id"] for r in rows if r["x"] == ""}), "skipped": dict(skipped),
           "groups": {}}
    groups = {"ALL": marks}
    for r in marks:
        groups.setdefault(r["city"], []).append(r)
        groups.setdefault(f'{r["city"]} | {r["camera_model"]}', []).append(r)
        if r["flagged"] == "0":
            groups.setdefault(f'{r["city"]} | not within 30 m of a mining pano', []).append(r)
    for g, rs in groups.items():
        near = [r for r in rs if r["within_25m"]]
        res["groups"][g] = {"within_25m": {f"r{rad}": recall(near, f"hit_{rad}") for rad in RADII},
                            "beyond_25m": recall([r for r in rs if not r["within_25m"]], f"hit_{RADII[0]}")}
    with open(os.path.join(out, "recall.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


RIG_MASK_Y = 0.5 + 49.0 / 180.0   # the labeler's NADIR_MASK_DEG: a detection below this is on the capture vehicle


def _model_spec(spec):
    """'label@threshold' -> (label, threshold); predictions are read from recall/preds/<label>/<city>.jsonl."""
    label, thr = spec.rsplit("@", 1)
    return label, float(thr)


def _pano_stats(verdicts, cities, label, threshold, primary_city):
    """Per pano, for one model: (marks, hits) on the primary population, and false alarms on confirmed-empty panos.

    The primary population is the sure marks within 25 m in ``primary_city``, matched one to one at RADII[0],
    exactly as ``score`` does. A confirmed-empty pano has no mark at all (sure or unsure) and the reviewer's
    "none here"; every detection on it is a false alarm.
    """
    preds = {}
    for city in cities:
        with open(os.path.join(OUT, "preds", label, f"{city}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                preds[r["pano_id"]] = r["preds"]
    city_of = {}
    for city in cities:
        for r in csv.DictReader(open(os.path.join(OUT, city, "intersections.csv"), encoding="utf-8")):
            if r["in_draw"] == "1":
                city_of[r["pano_id"]] = city
    marks, empty = {}, {}
    for pid, e in verdicts.items():
        ms = e.get("missed", [])
        if (not ms and not e.get("no_missed")) or pid not in city_of:
            continue
        if pid not in preds:
            raise SystemExit(f"{label}: no predictions for pano {pid}")
        dets = [(x, y) for x, y, c in preds[pid] if c >= threshold]
        if not ms:
            empty[pid] = (city_of[pid], len(dets), sum(1 for _, y in dets if y <= RIG_MASK_Y))
            continue
        if city_of[pid] != primary_city:
            continue
        sure = [(m["x"], m["y"]) for m in ms if not m.get("unsure")]
        hit = match_one_to_one(sure, dets, RADII[0])
        near = [i for i, (_, y) in enumerate(sure) if y >= Y_25M]
        if near:
            marks[pid] = (len(near), sum(1 for i in near if i in hit))
    return marks, empty


def cmd_compare(a):
    """Seed-averaged recall and confirmed-empty false alarms for one model, or new minus base (#285 A4).

    With only ``--base`` it is the calibration: per-seed numbers and the seed spread (the largest difference
    between the first seed and any other). With ``--new`` it adds the paired difference of the seed means,
    with a pano bootstrap that resamples the same panos for every model.
    """
    verdicts = json.load(open(a.verdicts, encoding="utf-8"))["panos"]
    sides = {"base": [_model_spec(s) for s in a.base], "new": [_model_spec(s) for s in (a.new or [])]}
    stats = {side: [_pano_stats(verdicts, a.cities, lab, thr, a.primary_city) for lab, thr in specs]
             for side, specs in sides.items()}
    panos = sorted(stats["base"][0][0])
    empties = sorted(stats["base"][0][1])

    def mean_recall(side, pick):
        return sum(sum(m[p][1] for p in pick) / sum(m[p][0] for p in pick) for m, _ in stats[side]) / len(stats[side])

    def mean_fa(side, col):
        return sum(sum(e[p][col] for p in empties) for _, e in stats[side]) / len(stats[side])

    res = {"primary_city": a.primary_city, "radius": RADII[0], "marks": sum(stats["base"][0][0][p][0] for p in panos),
           "panos_with_marks": len(panos), "empty_panos": len(empties),
           "empty_panos_by_city": dict(Counter(stats["base"][0][1][p][0] for p in empties)), "models": {}}
    for side, specs in sides.items():
        for (lab, thr), (m, e) in zip(specs, stats[side]):
            res["models"][f"{lab}@{thr}"] = {
                "side": side, "hit": sum(v[1] for v in m.values()),
                "recall": round(sum(v[1] for v in m.values()) / sum(v[0] for v in m.values()), 4),
                "empty_false_alarms": sum(v[1] for v in e.values()),
                "empty_false_alarms_rig_masked": sum(v[2] for v in e.values()),
                "empty_panos_with_one": sum(1 for v in e.values() if v[1])}
    for side, specs in sides.items():
        if not specs:
            continue
        ms = [res["models"][f"{lab}@{thr}"] for lab, thr in specs]
        res[side] = {"seeds": [f"{lab}@{thr}" for lab, thr in specs],
                     "mean_recall": round(mean_recall(side, panos), 4),
                     "mean_empty_false_alarms": round(mean_fa(side, 1), 3),
                     "mean_empty_false_alarms_rig_masked": round(mean_fa(side, 2), 3),
                     "seed_spread": {k: max([abs(ms[0][k] - x[k]) for x in ms[1:]] or [0])
                                     for k in ("hit", "empty_false_alarms", "empty_false_alarms_rig_masked")}}
    if sides["new"]:
        rng, boots = random.Random(SEED), []
        for _ in range(BOOT_N):
            pick = rng.choices(panos, k=len(panos))
            boots.append(mean_recall("new", pick) - mean_recall("base", pick))
        boots.sort()
        first = lambda side: res["models"]["%s@%s" % sides[side][0]]["recall"]
        seed1 = first("new") - first("base")
        diff = mean_recall("new", panos) - mean_recall("base", panos)
        lo, hi = boots[int(0.025 * (BOOT_N - 1))], boots[int(0.975 * (BOOT_N - 1))]
        res["new_minus_base"] = {
            "mean_recall_diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "first_seed_recall_diff": round(seed1, 4), "primary_met": bool(lo > 0 and seed1 > 0),
            "empty_false_alarms_diff": round(mean_fa("new", 1) - mean_fa("base", 1), 3),
            "empty_false_alarms_rig_masked_diff": round(mean_fa("new", 2) - mean_fa("base", 2), 3)}
    out = os.path.join(OUT, "compare", a.out_label)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


# ---- false-alarm contact sheet (guard 6, looked at rather than counted) -------------------------

SPOT_RADIUS = 0.03      # detections from different models this close (wrap distance) are one spot
CROP_W = 0.07           # crop width as a share of the pano width (about 25 degrees)
CROP_ASPECT = 0.75      # crop height / width, in pixels
CROP_PX = 360           # crop width on the sheet
THUMB_PX = 960          # whole-pano thumbnail width on the sheet


def _family(label):
    """Model family for grouping seeds on the sheet.

    >>> [_family(x) for x in ("crosswalk_s4_klone", "crosswalk_s4_seed3", "crosswalk_s4_steps1397",
    ...                       "crosswalk_s6full_seed2", "crosswalk_s6half")]
    ['s4', 's4', 's4_steps1397', 's6full', 's6half']
    """
    name = label.replace("crosswalk_", "")
    if name.startswith("s4_steps"):
        return name
    return name.split("_")[0]


def _spots(dets, radius=SPOT_RADIUS):
    """Greedy grouping of (model, x, y, conf) detections on one pano, strongest first.

    A detection joins the first spot whose seed detection is within ``radius`` (wrap distance), so a
    spot's position is its strongest detection's.

    >>> s = _spots([("a", 0.10, 0.6, 0.9), ("b", 0.11, 0.6, 0.5), ("a", 0.99, 0.6, 0.4), ("c", 0.01, 0.6, 0.3)])
    >>> [(round(sp["x"], 2), [d[0] for d in sp["dets"]]) for sp in s]
    [(0.1, ['a', 'b']), (0.99, ['a', 'c'])]
    """
    spots = []
    for d in sorted(dets, key=lambda d: -d[3]):
        for sp in spots:
            if wrap_dist(sp["x"], sp["y"], d[1], d[2]) <= radius:
                sp["dets"].append(d)
                break
        else:
            spots.append({"x": d[1], "y": d[2], "dets": [d]})
    return spots


def _crop(img, x, y):
    """A CROP_W-wide window centred on normalized (x, y), wrapping across the 360-degree seam."""
    from PIL import Image
    w, h = img.size
    cw = int(CROP_W * w)
    ch = int(cw * CROP_ASPECT)
    left = int(x * w) - cw // 2
    top = min(max(int(y * h) - ch // 2, 0), h - ch)
    out = Image.new("RGB", (cw, ch))
    dx = 0
    while dx < cw:                      # at most two pastes: the part before the seam and the part after
        sx = (left + dx) % w
        run = min(cw - dx, w - sx)
        out.paste(img.crop((sx, top, sx + run, top + ch)), (dx, 0))
        dx += run
    return out.resize((CROP_PX, int(CROP_PX * CROP_ASPECT))), (left, top, cw, ch)


def cmd_fa_sheet(a):
    """Local contact sheet of every detection guard 6 counts: model output on panos Jon confirmed empty.

    Reads the thresholds from the committed ``compare.json`` files, so the sheet shows exactly what the
    guard scored, and refuses to write anything if its own per-model counts differ from theirs.
    Imagery (crops, thumbnails, page) goes to ``--out`` outside the repo; the spot table goes to
    ``recall/fa_sheet/<out-label>/`` in the repo.
    """
    import html as H
    from PIL import Image, ImageDraw
    verdicts = json.load(open(a.verdicts, encoding="utf-8"))["panos"]
    models, guard = {}, {}
    for path in a.compare:
        for spec, m in json.load(open(path, encoding="utf-8"))["models"].items():
            label, thr = _model_spec(spec)
            if label in models and models[label] != thr:
                raise SystemExit(f"{label}: threshold {models[label]} in one compare file, {thr} in {path}")
            models[label], guard[label] = thr, m["empty_false_alarms_rig_masked"]
    native = dict(s.split("=", 1) for s in a.native)
    info = {}
    for city in a.cities:
        for r in csv.DictReader(open(os.path.join(OUT, city, "intersections.csv"), encoding="utf-8")):
            if r["in_draw"] == "1":
                info[r["pano_id"]] = {"city": city, **{k: r[k] for k in ("camera_model", "capture_date", "sequence_id")}}
    empties = sorted(pid for pid, e in verdicts.items()
                     if pid in info and not e.get("missed") and e.get("no_missed"))
    dets, counts = defaultdict(list), Counter()
    for label, thr in models.items():
        preds = {}
        for city in a.cities:
            with open(os.path.join(OUT, "preds", label, f"{city}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    preds[r["pano_id"]] = r["preds"]
        for pid in empties:
            for x, y, c in preds[pid]:
                if c >= thr and y <= RIG_MASK_Y:
                    dets[pid].append((label, x, y, c))
                    counts[label] += 1
        check = sum(v[2] for v in _pano_stats(verdicts, a.cities, label, thr, a.cities[0])[1].values())
        if not (counts[label] == guard[label] == check):
            raise SystemExit(f"{label}@{thr}: sheet {counts[label]}, compare.json {guard[label]}, _pano_stats {check}")
    families = sorted({_family(l) for l in models}, key=lambda f: (not f.startswith("s4"), f))
    seeds = {f: sorted(l for l in models if _family(l) == f) for f in families}
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    rows, cards = [], []
    order = sorted(dets, key=lambda p: (-sum(1 for d in dets[p] if not _family(d[0]).startswith("s4")), p))
    for pid in order:
        meta = info[pid]
        fns = [fn for fn in os.listdir(native[meta["city"]]) if os.path.splitext(fn)[0] == pid]
        if not fns:
            raise SystemExit(f"{meta['city']}: no image for {pid} in {native[meta['city']]}")
        img = Image.open(os.path.join(native[meta["city"]], fns[0])).convert("RGB")
        w, h = img.size
        thumb = img.resize((THUMB_PX, int(THUMB_PX * h / w)))
        draw, k = ImageDraw.Draw(thumb), THUMB_PX / w
        mask_y = int(RIG_MASK_Y * thumb.size[1])
        draw.line([(0, mask_y), (THUMB_PX, mask_y)], fill=(255, 255, 255), width=1)
        spot_html = []
        for n, sp in enumerate(_spots(dets[pid]), 1):
            crop, (left, top, cw, ch) = _crop(img, sp["x"], sp["y"])
            cd = ImageDraw.Draw(crop)
            s = CROP_PX / cw
            for _, x, y, _ in sp["dets"]:
                px = ((x * w - left) % w) * s
                py = (y * h - top) * s
                cd.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(255, 0, 255), width=2)
            name = f"{pid}_{n}.jpg"
            crop.save(os.path.join(a.out, "img", name), quality=85)
            x0, y0, y1 = (left * k) % THUMB_PX, int(top * k), int((top + ch) * k)
            x1 = x0 + cw * k
            for shift in (0, -THUMB_PX):        # the second pass draws the part that wraps past the seam
                if x1 + shift > 0 and x0 + shift < THUMB_PX:
                    draw.rectangle([int(x0 + shift), y0, int(x1 + shift), y1], outline=(255, 0, 255), width=2)
            draw.text((int(sp["x"] * THUMB_PX) + 3, int(top * k) + 2), str(n), fill=(255, 0, 255))
            fired = {f: sorted({seeds[f].index(d[0]) + 1 for d in sp["dets"] if _family(d[0]) == f}) for f in families}
            s6 = sum(len(v) for f, v in fired.items() if not f.startswith("s4"))
            rows.append({"pano_id": pid, **meta, "spot": n, "x": round(sp["x"], 4), "y": round(sp["y"], 4),
                         "max_conf": round(max(d[3] for d in sp["dets"]), 3), "n_models": len({d[0] for d in sp["dets"]}),
                         **{f"seeds_{f}": " ".join(map(str, fired[f])) for f in families}, "crop": f"img/{name}"})
            chips = " ".join(f'<span class="chip {"base" if f.startswith("s4") else "new"}">{H.escape(f)}: '
                             f'{",".join(map(str, fired[f])) or "&ndash;"}</span>' for f in families)
            spot_html.append(
                f'<figure class="spot" data-s4="{int(bool(fired.get("s4")))}" data-new="{int(s6 > 0)}">'
                f'<img src="img/{name}" width="{CROP_PX}" alt="Spot {n} on pano {pid}: area around the detection">'
                f'<figcaption><strong>Spot {n}</strong>, max conf {rows[-1]["max_conf"]}<br>{chips}</figcaption></figure>')
        tname = f"{pid}_thumb.jpg"
        thumb.save(os.path.join(a.out, "img", tname), quality=80)
        n_new = sum(1 for d in dets[pid] if not _family(d[0]).startswith("s4"))
        cards.append(
            f'<section class="card"><h2>{H.escape(meta["city"])} &middot; {pid}</h2>'
            f'<p class="meta">{H.escape(meta["camera_model"])}, captured {H.escape(meta["capture_date"])}, '
            f'sequence {H.escape(meta["sequence_id"])}; {len(dets[pid])} detections '
            f'({n_new} from the new models)</p>'
            f'<img class="thumb" src="img/{tname}" width="{THUMB_PX}" alt="Whole pano {pid}; magenta boxes mark the '
            f'spots below; the white line is the rig mask, below which nothing is counted">'
            f'<div class="spots">{"".join(spot_html)}</div></section>')
    csv_dir = os.path.join(OUT, "fa_sheet", a.out_label)
    os.makedirs(csv_dir, exist_ok=True)
    fields = list(rows[0]) if rows else ["pano_id"]
    with open(os.path.join(csv_dir, "spots.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    by_city = Counter()
    for pid, ds in dets.items():
        for d in ds:
            by_city[(info[pid]["city"], _family(d[0]))] += 1
    summary = {"empty_panos": len(empties), "panos_with_a_detection": len(dets), "spots": len(rows),
               "thresholds": models, "detections": dict(counts),
               "detections_by_city_family": {f"{c} | {f}": n for (c, f), n in sorted(by_city.items())},
               "spots_new_only": sum(1 for r in rows if not r.get("seeds_s4")),
               "spot_radius": SPOT_RADIUS, "rig_mask_y": round(RIG_MASK_Y, 4)}
    with open(os.path.join(csv_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    trs = "".join(f"<tr><td>{H.escape(l)}</td><td>{models[l]}</td><td>{counts[l]}</td></tr>" for l in sorted(models))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guard 6 false alarms</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#555; --line:#ddd; --base:#e6f0ff; --new:#ffe9e0; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161616; --fg:#eee; --muted:#aaa; --line:#333;
    --base:#1f2f4a; --new:#4a2a1f; }} }}
  body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 16px 48px; }}
  h1 {{ font-size:1.4em; }} h2 {{ font-size:1.05em; margin:0 0 4px; }}
  .meta {{ color:var(--muted); margin:0 0 8px; }}
  table {{ border-collapse:collapse; margin:8px 0 16px; }} td, th {{ border:1px solid var(--line); padding:3px 8px; }}
  .card {{ border-top:1px solid var(--line); padding:16px 0; }}
  .thumb {{ max-width:100%; height:auto; display:block; }}
  .spots {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; }}
  .spot {{ margin:0; max-width:{CROP_PX}px; }} .spot img {{ max-width:100%; height:auto; display:block; }}
  .chip {{ display:inline-block; border-radius:4px; padding:0 5px; margin:2px 2px 0 0; font-size:.85em; }}
  .chip.base {{ background:var(--base); }} .chip.new {{ background:var(--new); }}
  fieldset {{ border:1px solid var(--line); }}
</style></head><body>
<h1>Guard 6 false alarms: detections on the {len(empties)} panos Jon confirmed have no crosswalk</h1>
<p>Every detection here counts in guard 6 (at the model's matched threshold, above the rig mask). Nothing to rate.
Spots group detections within {SPOT_RADIUS} of the pano width across models; chips list which seeds fired there
(s4 = the adopted model, blue; new models, orange). {len(dets)} panos have at least one, in {len(rows)} spots.
Cards are ordered by how many new-model detections the pano has.</p>
<table><caption>Detections per model (equal to compare.json, checked)</caption>
<tr><th scope="col">Model</th><th scope="col">Threshold</th><th scope="col">Detections</th></tr>{trs}</table>
<fieldset><legend>Show spots</legend>
<label><input type="radio" name="f" value="all" checked> all</label>
<label><input type="radio" name="f" value="new"> fired by a new model but no s4 seed</label>
<label><input type="radio" name="f" value="s4"> fired by an s4 seed</label></fieldset>
{"".join(cards)}
<script>
  document.querySelectorAll('input[name=f]').forEach(function (r) {{
    r.addEventListener('change', function () {{
      document.querySelectorAll('.spot').forEach(function (s) {{
        var show = r.value === 'all' || (r.value === 'new' ? s.dataset.new === '1' && s.dataset.s4 === '0'
                                                           : s.dataset.s4 === '1');
        s.hidden = !show;
      }});
      document.querySelectorAll('.card').forEach(function (c) {{
        c.hidden = !c.querySelector('.spot:not([hidden])');
      }});
    }});
  }});
</script></body></html>
"""
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(json.dumps(summary, indent=1))
    print(f"-> {os.path.join(a.out, 'index.html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="the seeded intersection sample, fixed before any page or fetch")
    d.add_argument("--city", required=True)
    d.add_argument("--streets", required=True, help="the labeler's cached osm_streets.json for the run area")
    d.add_argument("--records", nargs="+", required=True, help="labeler results.jsonl file(s) with pano positions")
    d.add_argument("--exclude", nargs="*", help="draw lists whose panos may not be used")
    d.add_argument("--flag", nargs="*", help="pano lists to flag (not exclude), e.g. within 30 m of a mining pano")
    d.add_argument("--take", default="all", help="'all' or a number of intersections")
    d.add_argument("--all-of", help="with --take N: keep every intersection whose pano has this camera model")
    p = sub.add_parser("page", help="the blind marking page (gt_gallery over a detection-free bundle)")
    p.add_argument("--draw", nargs="+", required=True, help="city=<draw_records.jsonl>::<native pano dir>")
    p.add_argument("--bundle", required=True, help="bundle dir to (re)build; outside the repo, it holds imagery")
    p.add_argument("--out", required=True, help="page dir; outside the repo")
    sc = sub.add_parser("score", help="recall of a model's detections against the exported marks")
    sc.add_argument("--verdicts", required=True, help="the page's exported <run>_verdicts.json")
    sc.add_argument("--preds", nargs="+", required=True, help="score_274.py infer output (.jsonl), one per city")
    sc.add_argument("--cities", nargs="+", required=True)
    sc.add_argument("--label", required=True, help="model label, e.g. crosswalk_s4")
    sc.add_argument("--threshold", type=float, required=True)
    cp = sub.add_parser("compare", help="seed-averaged recall and confirmed-empty false alarms; new minus base")
    cp.add_argument("--verdicts", required=True, help="the page's exported <run>_verdicts.json")
    cp.add_argument("--cities", nargs="+", required=True)
    cp.add_argument("--primary-city", default="juneau-ak", help="the city whose marks are the recall population")
    cp.add_argument("--base", nargs="+", required=True, metavar="LABEL@THRESHOLD",
                    help="one per seed, first seed first; reads recall/preds/<label>/<city>.jsonl")
    cp.add_argument("--new", nargs="*", metavar="LABEL@THRESHOLD", help="the new model's seeds, in the same order")
    cp.add_argument("--out-label", required=True, help="results go to recall/compare/<out-label>/compare.json")
    fs = sub.add_parser("fa-sheet", help="local contact sheet of the detections guard 6 counts on confirmed-empty panos")
    fs.add_argument("--verdicts", required=True, help="the page's exported <run>_verdicts.json")
    fs.add_argument("--cities", nargs="+", required=True)
    fs.add_argument("--compare", nargs="+", required=True, help="compare.json files whose models and thresholds to show")
    fs.add_argument("--native", nargs="+", required=True, metavar="CITY=DIR", help="native pano dir per city")
    fs.add_argument("--out", required=True, help="page dir with crops; outside the repo")
    fs.add_argument("--out-label", required=True, help="spot table goes to recall/fa_sheet/<out-label>/")
    a = ap.parse_args()
    {"draw": cmd_draw, "page": cmd_page, "score": cmd_score, "compare": cmd_compare,
     "fa-sheet": cmd_fa_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
