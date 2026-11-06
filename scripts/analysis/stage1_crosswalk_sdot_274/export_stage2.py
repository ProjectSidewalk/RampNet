"""Crowd-label arm -> a Stage 2 training root for a first crosswalk detector (RampNet#274).

Stage 2 (`stage_two/train.py`) reads `<root>/{train,val}/<id>.jpg + <id>.json` with the points under
`curb_ramp_points_normalized` (a plain points key: the model is single-channel, so a crosswalk root needs no code
change) and, under `--loss-mask`, a `supervision` block (docs/sparse_supervision.md). Per pano of an arm's
`crowd_labels.jsonl`:

  complete (no unaccounted OSM paint / non-current SDOT point)   -> `dense_ignore`: trained everywhere (it teaches
                                                                    background) except its ignore points and boxes
  incomplete, with >= 1 positive                                  -> `sparse`: trained only near the positives;
                                                                    every ignore region becomes an ignore point
  incomplete, no positive                                         -> dropped (it would teach nothing)
plus VERIFIED-EMPTY panos (the crosswalk analogue of `generate_negative_panos.py`): random Seattle store panos
inside the regions with no SDOT point of any status, OSM crossing node or crowd Crosswalk label within NEG_M, and
no Vistas crosswalk component on a 6-view pass -> `verified_empty`.

Split: `stage_one/dataset_generation/split_dataset.py`'s rule re-implemented (that file is a script that creates
directories and runs on import, so it cannot be imported): panos within 60 m are one connected group, groups are
shuffled with seed 42, and filled train 70 / val 20 / rest test. Every pano within PHASE1_M of a Phase 1 GT node
(any city) is dropped first, so Jon's blind GT stays a held-out test.

  python export_stage2.py neg-pick  --store-listing <ls of the Seattle store> --inventory <inv> --census <census>
                                     --crowd <arm C out> --out OUT [--n 3000]         # network: PS metadata
  (copy OUT/neg_files.txt from makelab2 read-only, as for arm C)
  python export_stage2.py neg-check --panos <copied negatives> --out OUT               # GPU: Vistas
  python export_stage2.py build --arm <arm C out>=<its panos dir> [--arm ...] --negatives OUT=<negatives dir>
                                 --phase1-sheet <sheet.csv> --root <Stage 2 root> --out OUT [--keep-split]
Writes `manifest.json` (counts per split / mode / city, input sha256s) into OUT; the root itself is untracked.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402
from pilot import Grid, all_sdot_points, haversine_m, load_inventory, log, read_jsonl  # noqa: E402
import crowd_arm  # noqa: E402

NEG_M = 60.0             # generate_negative_panos.py's distance, here from any crossing evidence
PHASE1_M = 35.0
SPLIT_M = 60.0           # split_dataset.py DISTANCE_THRESHOLD_METERS
SPLIT_SEED = 42
OUT_W, OUT_H = 4096, 2048
SEED = 274


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ps_metadata(pid):
    url = f"{crowd_arm.PS_SERVER}/backupImage/{pid}/metadata"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": crowd_arm.UA}),
                                        timeout=30) as r:
                j = json.loads(r.read().decode("utf-8"))
            return {"pano_id": pid, "found": True, "lat": j["lat"], "lon": j["lng"],
                    "heading_deg": j["camera_heading"], "date": j.get("capture_date")}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After") or 30))
                continue
            return {"pano_id": pid, "found": False, "http": e.code}
        except Exception as e:  # noqa: BLE001
            log(f"PS metadata {pid}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------- negatives

def cmd_neg_pick(a):
    _, feats, regions = load_inventory(a.inventory)
    g_sdot = Grid(all_sdot_points(feats))
    import glob
    osm = []
    for fp in glob.glob(os.path.join(a.census, "osm", "seattle-wa", "Crosswalk", "*.json")):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            if e.get("type") == "node" and (e.get("tags") or {}).get("highway") == "crossing":
                osm.append((e["lat"], e["lon"], e["id"]))
    g_osm = Grid(osm)
    crowd = []
    with open(os.path.join(a.census, "rawlabels", "seattle-wa_Crosswalk.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("latitude"):
                crowd.append((float(r["latitude"]), float(r["longitude"]), r["label_id"]))
    g_crowd = Grid(crowd)
    labelled = {r["pano_id"] for r in csv.DictReader(open(os.path.join(a.crowd, "selection.csv"), encoding="utf-8"))}
    store = sorted(l.split()[0][:-4] for l in open(a.store_listing, encoding="utf-8") if l.strip())
    cand = [p for p in store if p not in labelled]
    pick = random.Random(SEED).sample(cand, min(a.n, len(cand)))
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, "neg_meta.jsonl")
    meta = {m["pano_id"]: m for m in read_jsonl(mp)}
    todo = [p for p in pick if p not in meta]
    log(f"{len(store)} store panos, {len(cand)} without a crowd Crosswalk label; PS metadata for {len(todo)} of {len(pick)}")
    for i, pid in enumerate(todo):
        m = ps_metadata(pid)
        if m is not None:
            meta[pid] = m
            with open(mp, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(m) + "\n")
        time.sleep(crowd_arm.PS_SPACING_S)
        if i and i % 500 == 0:
            log(f"  {i}/{len(todo)}")
    keep, why = [], {}
    for pid in pick:
        m = meta.get(pid)
        reason = None
        if not m or not m.get("found"):
            reason = "no_metadata"
        elif not regions(m["lat"], m["lon"]):
            reason = "outside_regions"
        elif g_sdot.within(m["lat"], m["lon"], NEG_M):
            reason = "sdot_within_60m"
        elif g_osm.within(m["lat"], m["lon"], NEG_M):
            reason = "osm_crossing_within_60m"
        elif g_crowd.within(m["lat"], m["lon"], NEG_M):
            reason = "crowd_crosswalk_within_60m"
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            keep.append(pid)
    with open(os.path.join(a.out, "neg_files.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"./{p[:2]}/{p}.jpg\n" for p in sorted(keep)))
    json.dump({"store_listing_sha256": sha256(a.store_listing), "sampled": len(pick), "kept": len(keep),
               "dropped": why, "neg_m": NEG_M, "seed": SEED},
              open(os.path.join(a.out, "neg_pick.json"), "w", encoding="utf-8"), indent=1)
    log(f"kept {len(keep)} of {len(pick)} candidates ({why}) -> neg_files.txt")


def cmd_neg_check(a):
    import vistas_crosswalk_probe as probe
    from equirect_tiling import default_views, equirect_to_perspective, perspective_point_to_equirect
    ids = [l.strip().split("/")[-1][:-4] for l in open(os.path.join(a.out, "neg_files.txt"), encoding="utf-8")
           if l.strip()]
    outp = os.path.join(a.out, "neg_check.csv")
    done = set()
    if os.path.exists(outp):
        done = {r["pano_id"] for r in csv.DictReader(open(outp, encoding="utf-8", newline=""))}
    new = not os.path.exists(outp)
    fh = open(outp, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=["pano_id", "n_components", "max_area_px", "verified_empty"],
                       lineterminator="\n")
    if new:
        w.writeheader()
    seg = probe.Segmenter()
    for k, pid in enumerate(ids):
        if pid in done or not os.path.exists(crowd_arm.pano_path(a.panos, pid)):
            continue
        pano = crowd_arm.load_pano(crowd_arm.pano_path(a.panos, pid))
        comps = []
        for view in default_views():
            sg, pr = seg.seg_prob(equirect_to_perspective(pano, view))
            comps += probe.view_components(sg, pr, view, perspective_point_to_equirect)
        w.writerow({"pano_id": pid, "n_components": len(comps),
                    "max_area_px": max((c["area_px"] for c in comps), default=0), "verified_empty": int(not comps)})
        fh.flush()
        if k % 100 == 0:
            log(f"  {k + 1}/{len(ids)}")
    fh.close()


# ---------------------------------------------------------------- build

def split_groups(coords):
    """split_dataset.py's rule: 60 m connected groups, shuffled with seed 42, filled 70/20/rest."""
    ids = sorted(coords)
    g = Grid([(coords[i][0], coords[i][1], i) for i in ids])
    seen, groups = set(), []
    for i in ids:
        if i in seen:
            continue
        q, grp = deque([i]), []
        seen.add(i)
        while q:
            c = q.popleft()
            grp.append(c)
            for _, it in g.within(coords[c][0], coords[c][1], SPLIT_M):
                if it[2] not in seen:
                    seen.add(it[2])
                    q.append(it[2])
        groups.append(grp)
    random.Random(SPLIT_SEED).shuffle(groups)
    total = len(ids)
    tgt = {"train": int(0.7 * total), "val": int(0.2 * total)}
    cnt = {"train": 0, "val": 0, "test": 0}
    out = {}
    for grp in groups:
        s = "train" if cnt["train"] < tgt["train"] else "val" if cnt["val"] < tgt["val"] else "test"
        cnt[s] += len(grp)
        for i in grp:
            out[i] = s
    return out


def pano_json(rec, coord):
    pts = [[p["x"], p["y"]] for p in rec["points"]]
    sup = {"city": rec.get("city"), "label_type": "Crosswalk", "sources": [p["source"] for p in rec["points"]],
           "point_ids": [p["id"] for p in rec["points"]]}
    if rec["complete"]:
        sup.update({"mode": "dense_ignore",
                    "ignore_points": [[i["x"], i["y"]] for i in rec["ignore"] if "box" not in i],
                    "ignore_boxes": [i["box"] for i in rec["ignore"] if "box" in i]})
    else:
        sup.update({"mode": "sparse", "ignore_points": [[i["x"], i["y"]] for i in rec["ignore"]]})
    return {"pano_id": rec["pano_id"], "curb_ramp_points_normalized": pts, "pano_coord": coord,
            "capture_date": rec.get("capture_date"), "supervision": sup}


def write_image(src, dst):
    img = crowd_arm.load_pano(src, max_edge=OUT_W)
    if img.size != (OUT_W, OUT_H):
        img = img.resize((OUT_W, OUT_H))
    img.save(dst, quality=92)


def cmd_build(a):
    p1 = []
    if a.phase1_sheet:
        for r in csv.DictReader(open(a.phase1_sheet, encoding="utf-8", newline="")):
            p1.append((float(r["node_lat"]), float(r["node_lng"]), r["gt_id"]))   # positions only, never answers
    g_p1 = Grid(p1, lat0=40.0) if p1 else None
    items, coords, dropped = {}, {}, {}

    def drop(k):
        dropped[k] = dropped.get(k, 0) + 1

    inputs = {}
    for spec in a.arm:
        arm_dir, panos = spec.split("=", 1)
        inputs[arm_dir] = sha256(os.path.join(arm_dir, crowd_arm.EXPORT))
        meta, _, _ = crowd_arm.merged_meta(arm_dir)
        for rec in read_jsonl(os.path.join(arm_dir, crowd_arm.EXPORT)):
            m = meta.get(rec["pano_id"]) or {}
            if not m.get("found"):
                drop("no_position")
                continue
            if g_p1 and g_p1.within(m["lat"], m["lon"], PHASE1_M):
                drop("near_phase1_gt")
                continue
            if not rec["complete"] and not rec["points"]:
                drop("incomplete_no_positive")
                continue
            items[rec["pano_id"]] = (pano_json(rec, [m["lat"], m["lon"]]), crowd_arm.pano_path(panos, rec["pano_id"]))
            coords[rec["pano_id"]] = (m["lat"], m["lon"])
    if a.negatives:
        nd, npanos = a.negatives.split("=", 1)
        nm = {m["pano_id"]: m for m in read_jsonl(os.path.join(nd, "neg_meta.jsonl"))}
        inputs[nd] = sha256(os.path.join(nd, "neg_check.csv"))
        for r in csv.DictReader(open(os.path.join(nd, "neg_check.csv"), encoding="utf-8", newline="")):
            if r["verified_empty"] != "1":
                drop("negative_has_vistas_paint")
                continue
            m = nm[r["pano_id"]]
            if g_p1 and g_p1.within(m["lat"], m["lon"], PHASE1_M):
                drop("near_phase1_gt")
                continue
            items[r["pano_id"]] = ({"pano_id": r["pano_id"], "curb_ramp_points_normalized": [],
                                    "pano_coord": [m["lat"], m["lon"]], "capture_date": m.get("date"),
                                    "supervision": {"mode": "verified_empty", "city": "seattle-wa",
                                                    "label_type": "Crosswalk", "sources": []}},
                                   crowd_arm.pano_path(npanos, r["pano_id"]))
            coords[r["pano_id"]] = (m["lat"], m["lon"])
    split = split_groups(coords)
    kept = {}
    if a.keep_split:
        # A relabel must not reshuffle: split_groups fills groups in shuffled order, so one pano more or less moves
        # every later group. Keep each pano's split as the root already has it; only panos new to the root take the
        # rule's split, and panos no longer built are removed from the root.
        for s in ("train", "val", "test"):
            d = os.path.join(a.root, s)
            for fn in (os.listdir(d) if os.path.isdir(d) else []):
                if fn.endswith(".json"):
                    kept[fn[:-5]] = s
        moved = {pid: kept[pid] for pid in items if pid in kept and kept[pid] != split[pid]}
        split.update(moved)
        gone = sorted(set(kept) - set(items))
        for pid in gone:
            for ext in (".json", ".jpg"):
                p = os.path.join(a.root, kept[pid], pid + ext)
                if os.path.exists(p):
                    os.remove(p)
        log(f"--keep-split: {len(kept)} panos already in the root; {len(moved)} keep a split the rule would move; "
            f"{sum(1 for p in items if p not in kept)} new; {len(gone)} removed ({','.join(gone[:20])})")
    counts = {}
    for pid, (js, src) in sorted(items.items()):
        s = split[pid]
        d = os.path.join(a.root, s)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, pid + ".jpg")):
            write_image(src, os.path.join(d, pid + ".jpg"))
        with open(os.path.join(d, pid + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(js, f)
        key = f"{s}/{js['supervision']['mode']}/{js['supervision'].get('city')}"
        counts[key] = counts.get(key, 0) + 1
    os.makedirs(a.out, exist_ok=True)
    man = {"root": a.root, "panos": len(items), "counts": dict(sorted(counts.items())), "dropped": dropped,
           "inputs_sha256": inputs, "split": {"rule": "split_dataset.py: 60 m groups, seed 42, 70/20/rest",
                                              "phase1_exclusion_m": PHASE1_M,
                                              "kept_from_root": bool(a.keep_split)},
           "points": sum(len(js["curb_ramp_points_normalized"]) for js, _ in items.values()),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), indent=1)
    log(f"{len(items)} panos -> {a.root}; {counts}; dropped {dropped}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("neg-pick")
    for k in ("--store-listing", "--inventory", "--census", "--crowd", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--n", type=int, default=3000)
    s = sub.add_parser("neg-check")
    s.add_argument("--panos", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("build")
    s.add_argument("--arm", action="append", required=True, help="<arm out dir>=<its panos dir>")
    s.add_argument("--negatives", help="<neg out dir>=<negatives panos dir>")
    s.add_argument("--phase1-sheet")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--keep-split", action="store_true",
                   help="relabel an existing root: keep each pano's split, remove panos no longer built")
    a = ap.parse_args()
    {"neg-pick": cmd_neg_pick, "neg-check": cmd_neg_check, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
