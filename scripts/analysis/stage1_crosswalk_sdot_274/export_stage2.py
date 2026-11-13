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
  python export_stage2.py add-negatives --ratings <page CSVs, in order> --drop <disc-rule CSV> --src-root <root_v2>
                                 --root <root_v3> --out OUT     # crosswalk_s3: Jon's negatives on train panos
  python export_stage2.py add-panos --src-root <root_v3> --root <root_v4> --positives <csv>=<jpgs> ...
                                 --negatives <ratings>=<jpgs> --positions <prod_s4_pano_positions.sh out>
                                 --phase1-sheet <sheet.csv> --census-rawlabels <census out/rawlabels>
                                 --exclude <draw lists> --out OUT   # crosswalk_s4: new sparse train panos
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


# ---------------------------------------------------------------- hard negatives (crosswalk_s3)

NEGATIVE_MIN_DIST = 0.05   # stage_two/train.py's NEGATIVE_MIN_DIST: it refuses a negative this near a positive


def neg_dist(a, b):
    """stage_two/train.py's distance: x in pano widths with seam wrap, y halved."""
    dx = abs(a[0] - b[0])
    return math.hypot(min(dx, 1.0 - dx), (a[1] - b[1]) / 2)


def confirmed_negatives(ratings, drops):
    """Jon's verdicts -> the negatives a root trains. `ratings` are read in order and a later file's verdict on a
    card replaces an earlier one (so a re-review listed after its page overrides it); a card is a negative when its
    last verdict is not_crosswalk and its card_id is not in a `drops` file (the disc rule). Returns
    (rows, counts)."""
    last = {}
    for path in ratings:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("verdict"):
                    last[r["card_id"]] = r
    dropped = set()
    for path in drops:
        with open(path, encoding="utf-8", newline="") as f:
            dropped |= {r["card_id"] for r in csv.DictReader(f)}
    verdicts = {}
    for r in last.values():
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    rows = [r for cid, r in sorted(last.items()) if r["verdict"] == "not_crosswalk" and cid not in dropped]
    unknown = dropped - set(last)
    if unknown:
        raise SystemExit(f"drop list names cards no ratings file has: {sorted(unknown)}")
    return rows, {"rated_cards": len(last), "verdicts": verdicts,
                  "dropped_disc_rule": sum(1 for c in dropped if last[c]["verdict"] == "not_crosswalk"),
                  "negatives": len(rows)}


def cmd_add_negatives(a):
    """A new root = --src-root plus `supervision.negative_points` on train panos (#274 hard negatives). JPEGs are
    hardlinked (no second copy); every JSON is rewritten, unchanged except on the panos that gain negatives. A
    negative on a val/test pano, or on a pano the source root lacks, stops the build: val and test stay as they
    were, so s3 scores on the same data as s1. A negative within NEGATIVE_MIN_DIST of one of its pano's positives
    is left out and counted (the trainer would refuse it)."""
    rows, counts = confirmed_negatives(a.ratings, a.drop)
    where = {}
    for s in ("train", "val", "test"):
        d = os.path.join(a.src_root, s)
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                where[fn[:-5]] = s
    per_pano, near_pos, out_rows = {}, [], []
    for r in rows:
        pid, xy = r["pano_id"], (float(r["x"]), float(r["y"]))
        s = where.get(pid)
        if s != "train":
            raise SystemExit(f"{r['card_id']}: pano is in {s or 'no split of the source root'}; negatives go on "
                             f"train panos only")
        with open(os.path.join(a.src_root, s, pid + ".json"), encoding="utf-8") as f:
            pos = json.load(f)["curb_ramp_points_normalized"]
        if any(neg_dist(xy, p) < NEGATIVE_MIN_DIST for p in pos):
            near_pos.append(r["card_id"])
            continue
        per_pano.setdefault(pid, []).append((xy, r["card_id"]))
        out_rows.append({"card_id": r["card_id"], "pano_id": pid, "x": r["x"], "y": r["y"], "city": r["city"],
                         "mode": r["mode"], "stratum": r["stratum"], "split": s, "note": r.get("note", "")})
    counts["left_out_near_positive"] = near_pos
    by = {}
    for s in ("train", "val", "test"):
        src, dst = os.path.join(a.src_root, s), os.path.join(a.root, s)
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(src)):
            sp, dp = os.path.join(src, fn), os.path.join(dst, fn)
            if fn.endswith(".jpg"):
                if not os.path.exists(dp):
                    os.link(sp, dp)
            elif fn.endswith(".json"):
                with open(sp, encoding="utf-8") as f:
                    js = json.load(f)
                negs = per_pano.get(fn[:-5])
                if negs:
                    sup = js.setdefault("supervision", {})
                    sup["negative_points"] = [[x, y] for (x, y), _ in negs]
                    sup["negative_ids"] = [cid for _, cid in negs]
                    key = f"{s}/{sup.get('mode')}/{sup.get('city')}"
                    by[key] = by.get(key, 0) + len(negs)
                with open(dp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(js, f)
    labels = hashlib.sha256()
    for s in ("train", "val", "test"):
        for fn in sorted(os.listdir(os.path.join(a.root, s))):
            if fn.endswith(".json"):
                labels.update(f"{s}/{fn} {sha256(os.path.join(a.root, s, fn))}\n".encode())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "negatives.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    man = {"src_root": a.src_root, "root": a.root, **counts, "trained_negatives": len(out_rows),
           "panos_with_negatives": len(per_pano), "negatives_by_split_mode_city": dict(sorted(by.items())),
           "inputs_sha256": {p: sha256(p) for p in [*a.ratings, *a.drop]},
           "labels_digest": labels.hexdigest(),
           "labels_digest_rule": "sha256 over sorted lines '<split>/<id>.json <sha256 of the file>\\n'",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    log(f"{len(out_rows)} negatives on {len(per_pano)} train panos -> {a.root}; {len(near_pos)} left out near a "
        f"positive; labels digest {labels.hexdigest()[:12]}")


# ---------------------------------------------------------------- new train panos (crosswalk_s4)

def _ids(path):
    """Pano ids from a draw list (`<id>.jpg` per line) or any CSV with a pano_id column."""
    with open(path, encoding="utf-8", newline="") as f:
        if path.endswith(".csv"):
            return {r["pano_id"] for r in csv.DictReader(f)}
        return {ln.strip()[:-4] if ln.strip().endswith(".jpg") else ln.strip() for ln in f if ln.strip()}


def census_crosswalk_points(rawlabels_dir, panos):
    """{pano_id: [(label_uid, x, y)]}: every Crosswalk label in the Phase 0 census on `panos`, any vote."""
    out = {}
    for path in sorted(os.listdir(rawlabels_dir)):
        if not path.endswith("_Crosswalk.csv"):
            continue
        city = path[:-len("_Crosswalk.csv")]
        with open(os.path.join(rawlabels_dir, path), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["pano_id"] in panos:
                    out.setdefault(r["pano_id"], []).append(
                        (f"{city}:{r['label_id']}", float(r["pano_x"]) / float(r["pano_width"]),
                         float(r["pano_y"]) / float(r["pano_height"])))
    return out


def cmd_add_panos(a):
    """A new root = --src-root plus new TRAIN panos (crosswalk_s4, pre-registered in docs/stage1_crosswalk_sdot_274.md):
    Jon-confirmed positives as `sparse` panos, and car-top negatives as `sparse` panos with only
    `supervision.negative_points`. Every source-root file is carried over byte for byte (JPEGs hardlinked, JSONs
    copied), so val, test and the s3 train set are unchanged. Every other census Crosswalk label on a new positive
    pano becomes an ignore point (it is unconfirmed, and sparse training would otherwise teach background near it).
    Stops on: a new pano already in the source root or in an --exclude list, a pano with no prod position, a
    negative within NEGATIVE_MIN_DIST of a positive. Drops (and counts) a pano within PHASE1_M of a Phase 1 GT node,
    measured from the pano's own position in prod (prod_s4_pano_positions.sh), never from a label's."""
    where = {}
    for s in ("train", "val", "test"):
        for fn in os.listdir(os.path.join(a.src_root, s)):
            if fn.endswith(".json"):
                where[fn[:-5]] = s
    excluded = set()
    for p in a.exclude:
        excluded |= _ids(p)
    pos_by, img = {}, {}
    for spec in a.positives:
        path, d = spec.split("=", 1)
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                pos_by.setdefault(r["pano_id"], {"city": r["city"], "pts": []})["pts"].append(
                    (r["label_uid"], float(r["x"]), float(r["y"])))
                img[r["pano_id"]] = os.path.join(d, r["pano_id"] + ".jpg")
    neg_by = {}
    for spec in a.negatives:
        path, d = spec.split("=", 1)
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["verdict"] != "not_crosswalk":
                    continue
                neg_by.setdefault(r["pano_id"], {"city": r["city"], "pts": []})["pts"].append(
                    (r["card_id"], float(r["x"]), float(r["y"])))
                img.setdefault(r["pano_id"], os.path.join(d, r["pano_id"] + ".jpg"))
    new = set(pos_by) | set(neg_by)
    clash = sorted(p for p in new if p in where)
    if clash:
        raise SystemExit(f"{len(clash)} new panos are already in the source root: {clash[:10]}")
    clash = sorted(new & excluded)
    if clash:
        raise SystemExit(f"{len(clash)} new panos are in an --exclude list: {clash[:10]}")
    pos_at = {}
    with open(a.positions, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["pano_id"] in new:
                pos_at[r["pano_id"]] = (float(r["lat"]), float(r["lng"]), r["capture_date"] or None)
    missing = sorted(new - set(pos_at))
    if missing:
        raise SystemExit(f"{len(missing)} new panos have no prod position: {missing[:10]}")
    p1 = [(float(r["node_lat"]), float(r["node_lng"]), r["gt_id"])
          for r in csv.DictReader(open(a.phase1_sheet, encoding="utf-8", newline=""))]
    g_p1 = Grid(p1, lat0=40.0)
    near_p1 = sorted(p for p in new if g_p1.within(pos_at[p][0], pos_at[p][1], PHASE1_M))
    new -= set(near_p1)
    others = census_crosswalk_points(a.census_rawlabels, new) if a.census_rawlabels else {}
    for path in a.ignore:          # crosswalk_s5: extra ignore points (pano_id, x, y), e.g. Jon's unsure verdicts
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["pano_id"] in new:
                    others.setdefault(r["pano_id"], []).append((r["ignore_id"], float(r["x"]), float(r["y"])))
    # copy the source root unchanged
    for s in ("train", "val", "test"):
        src, dst = os.path.join(a.src_root, s), os.path.join(a.root, s)
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(src)):
            dp = os.path.join(dst, fn)
            if fn.endswith(".jpg") and not os.path.exists(dp):
                os.link(os.path.join(src, fn), dp)
            elif fn.endswith(".json"):
                with open(os.path.join(src, fn), "rb") as fi, open(dp, "wb") as fo:
                    fo.write(fi.read())
    counts, rows = {}, []
    for pid in sorted(new):
        p, n = pos_by.get(pid), neg_by.get(pid)
        city = (p or n)["city"]
        pts = [(x, y) for _, x, y in (p["pts"] if p else [])]
        mine = {u for u, _, _ in (p["pts"] if p else [])}
        ignore = [[round(x, 6), round(y, 6)] for u, x, y in others.get(pid, []) if u not in mine]
        sup = {"city": city, "label_type": "Crosswalk", "mode": "sparse",
               "sources": [a.source] * len(pts),
               "point_ids": [u for u, _, _ in p["pts"]] if p else [], "ignore_points": ignore}
        if n:
            for cid, nx, ny in n["pts"]:
                if any(neg_dist((nx, ny), q) < NEGATIVE_MIN_DIST for q in pts):
                    raise SystemExit(f"{cid}: negative within {NEGATIVE_MIN_DIST} of a positive on {pid}")
            sup["negative_points"] = [[x, y] for _, x, y in n["pts"]]
            sup["negative_ids"] = [c for c, _, _ in n["pts"]]
        lat, lng, date = pos_at[pid]
        js = {"pano_id": pid, "curb_ramp_points_normalized": [[x, y] for x, y in pts], "pano_coord": [lat, lng],
              "capture_date": date, "supervision": sup}
        d = os.path.join(a.root, "train")
        jp = os.path.join(d, pid + ".jpg")
        if not os.path.exists(jp):
            os.link(img[pid], jp)          # already written by bprime-prep = write_image, as the root's JPEGs
        with open(os.path.join(d, pid + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(js, f)
        kind = "positives" if p else "negatives_only"
        key = f"train/{kind}/{city}"
        counts[key] = counts.get(key, 0) + 1
        rows.append({"pano_id": pid, "city": city, "kind": kind, "positives": len(pts),
                     "negatives": len(n["pts"]) if n else 0, "ignore_points": len(ignore)})
    labels = hashlib.sha256()
    for s in ("train", "val", "test"):
        for fn in sorted(os.listdir(os.path.join(a.root, s))):
            if fn.endswith(".json"):
                labels.update(f"{s}/{fn} {sha256(os.path.join(a.root, s, fn))}\n".encode())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "new_panos.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    man = {"src_root": a.src_root, "root": a.root, "new_panos": len(rows), "panos_by_kind_city": dict(sorted(counts.items())),
           "positives": sum(r["positives"] for r in rows), "negatives": sum(r["negatives"] for r in rows),
           "ignore_points": sum(r["ignore_points"] for r in rows),
           "dropped_near_phase1_gt": near_p1, "phase1_exclusion_m": PHASE1_M,
           "inputs_sha256": {p.split("=", 1)[0]: sha256(p.split("=", 1)[0])
                             for p in [*a.positives, *a.negatives, a.positions, a.phase1_sheet, *a.exclude,
                                       *a.ignore]},
           "labels_digest": labels.hexdigest(),
           "labels_digest_rule": "sha256 over sorted lines '<split>/<id>.json <sha256 of the file>\\n'",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    log(f"{len(rows)} new train panos -> {a.root}; {counts}; {len(near_p1)} dropped near Phase 1 GT; "
        f"labels digest {labels.hexdigest()[:12]}")


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
    s = sub.add_parser("add-negatives", help="a new root with Jon's confirmed hard negatives on train panos")
    s.add_argument("--ratings", nargs="+", required=True, help="ratings CSVs in order; a later verdict wins")
    s.add_argument("--drop", nargs="*", default=[], help="CSVs of card_ids left out (the disc rule)")
    s.add_argument("--src-root", required=True, help="the root to extend (root_v2)")
    s.add_argument("--root", required=True, help="the new root (JPEGs hardlinked from --src-root)")
    s.add_argument("--out", required=True, help="writes negatives.csv + manifest.json here")
    s = sub.add_parser("add-panos", help="crosswalk_s4: a new root with new sparse train panos")
    s.add_argument("--src-root", required=True, help="the root to extend (root_v3)")
    s.add_argument("--root", required=True, help="the new root (root_v4)")
    s.add_argument("--positives", nargs="+", default=[],
                   help="<csv with label_uid,city,pano_id,x,y>=<dir of 4096x2048 JPEGs>")
    s.add_argument("--negatives", nargs="*", default=[],
                   help="<ratings CSV; not_crosswalk rows become negatives>=<dir of 4096x2048 JPEGs>")
    s.add_argument("--positions", required=True, help="prod_s4_pano_positions.sh output")
    s.add_argument("--phase1-sheet", required=True)
    s.add_argument("--census-rawlabels", help="Phase 0 census out/rawlabels; omit for panos PS has no labels on "
                                              "(crosswalk_s6's OSM cities)")
    s.add_argument("--source", default="ps_crowd_jon_confirmed",
                   help="supervision.sources value for the new positives (crosswalk_s6: osm_vistas_auto)")
    s.add_argument("--exclude", nargs="*", default=[], help="draw lists / panos CSVs that must not be added")
    s.add_argument("--ignore", nargs="*", default=[],
                   help="CSVs of ignore_id,pano_id,x,y: extra ignore points on new panos (crosswalk_s5)")
    s.add_argument("--out", required=True, help="writes new_panos.csv + manifest.json here")
    a = ap.parse_args()
    {"neg-pick": cmd_neg_pick, "neg-check": cmd_neg_check, "build": cmd_build,
     "add-negatives": cmd_add_negatives, "add-panos": cmd_add_panos}[a.cmd](a)


if __name__ == "__main__":
    main()
