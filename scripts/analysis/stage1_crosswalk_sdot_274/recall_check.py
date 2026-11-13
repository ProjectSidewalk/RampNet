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
    a = ap.parse_args()
    {"draw": cmd_draw, "page": cmd_page, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
