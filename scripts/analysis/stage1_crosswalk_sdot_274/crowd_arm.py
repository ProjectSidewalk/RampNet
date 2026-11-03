"""Arm C of the Stage 1 crosswalk pilot: SDOT-confirmed crowd Crosswalk labels on stored Seattle panos (RampNet#274).

The SDOT pilot (`pilot.py`, docs/stage1_crosswalk_sdot_274.md) showed that a projected inventory point is a
bearing, not a placement (B1 exact-pixel 0.371), that Vistas snapping places it (B2 passes), and that old captures
can predate the paint. Project Sidewalk's own panos are backed up at native resolution on makelab2
(`/projects/makeabilitylab/sidewalk_panos/Panoramas/seattle-wa/<id[:2]>/<id>.jpg`), and 3,292 of the 3,301
Seattle panos carrying a crowd Crosswalk label are there. A crowd label is a person clicking paint in THAT pano,
so date and placement are settled; SDOT then confirms it and supplies the crosswalks nobody clicked.

Label classes (crowd labels keyed `seattle-wa:<label_id>`; verdict = Phase 0's human-only margin rule):
  crowd_pos       verdict not incorrect AND a current SDOT point within 10 m of the label    -> a positive point
  crowd_neg       verdict incorrect AND no current SDOT point within 10 m                    -> a hard negative point
  crowd_conflict  everything else (crowd and SDOT disagree)                                  -> an ignore region
  sdot_b2         a current SDOT crosswalk within INCLUSION_M of the pano with no crowd_pos label on this pano
                  within 10 m of it, placed by B2 (pilot.vistas_at)                          -> a positive point
  sdot_ignore     the same, but Vistas finds no component within SNAP_R of B1               -> an ignore region
  vistas_unexplained  a crosswalk component of a full-pano Vistas pass (the Phase 0 probe's 6-view rig) that no
                  mark above explains (`explained`)                                          -> an ignore region
A pano is `complete: false` when, within INCLUSION_M, there is OSM paint (marked or signalised node) that neither
SDOT (10 m) nor a crowd_pos label (10 m) accounts for, or a non-current SDOT point (PLANNED etc.): unlabelled
paint may be in view. Incomplete panos are kept and flagged, never silently dropped.

Subcommands:
  select   classes per label + pano metadata (streetlevel find_panorama_by_id, cached) -> selection.csv
  place    Vistas at every crowd click (the metric's human ceiling) and B1/B2 for SDOT-only crosswalks
           -> placements.csv, thumbs/ (untracked)
  export   crowd_labels.jsonl, one record per pano: points, ignore, negatives, complete, capture_date
  report   report.md (calibration of the exact-pixel metric against human clicks, label mix, by year)
  gallery  gallery.html, a local contact sheet of GALLERY_N seeded panos with every mark (never an Artifact)

  python crowd_arm.py select --inventory <crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274 \
      --census <Phase 0 census OUT> --panos D:/Git/sal-scratch-keep/arm-c-seattle-store/panos --out OUT
  python crowd_arm.py place --inventory ... --census ... --panos ... --out OUT [--limit N]
  python crowd_arm.py export --out OUT && python crowd_arm.py report --out OUT && python crowd_arm.py gallery --out OUT

Ignore regions are points with radius SNAP_R (pano-width units, RampNet's matcher metric), and the
vistas_unexplained ones also carry an approximate box [x_left, x_right, y_top, y_bottom] (x may wrap the seam):
a trainer should neither reward nor penalise detections inside them.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402  (path setup, helpers, project, vistas_at, thumbnail, constants)
from pilot import (CURRENT, INCLUSION_M, SNAP_R, TOL_R, Grid, all_sdot_points, frac, haversine_m,  # noqa: E402
                   load_inventory, log, osm_marked_nodes, pano_dist, pct, project, read_jsonl, vistas_at)
sys.path.insert(0, os.path.join(pilot.REPO, "scripts", "analysis", "crosswalk_phase0_274"))
from crossing_census import verdict  # noqa: E402  (human-only margin rule, as in Phase 0 and the inventory branch)

CITY = "seattle-wa"
MATCH_M = 10.0
GALLERY_N = 60
SEED = 274
SELECTION, META, PLACEMENTS, EXPORT = "selection.csv", "crowd_meta.jsonl", "placements.csv", "crowd_labels.jsonl"
PS_META = "ps_meta.jsonl"
PS_SERVER = "https://sidewalk-sea.cs.washington.edu"
PS_SPACING_S = 0.25
BOX_PAD_Y = 0.02     # vertical pad on an unexplained component's approximate box (pano-height units)
UA = "ProjectSidewalk-RampNet-crosswalk-274 (research)"


def pano_path(panos, pid):
    return os.path.join(panos, pid[:2], pid + ".jpg")


def load_pano(path, max_edge=4096):
    """Native-res store JPEG -> RGB PIL at <= max_edge, decoded with JPEG draft scaling (much faster at 13k px)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path)
    img.draft("RGB", (max_edge, max_edge // 2))
    img = img.convert("RGB")
    if max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    return img


# ---------------------------------------------------------------- select

def cmd_select(a):
    from streetlevel import streetview as sv
    raw = {}
    with open(os.path.join(a.census, "rawlabels", f"{CITY}_Crosswalk.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            raw[f"{CITY}:{r['label_id']}"] = r
    inv = {r["label_uid"]: r for r in csv.DictReader(open(os.path.join(a.inventory, "crowd_labels.csv"),
                                                          encoding="utf-8", newline=""))}
    rows = []
    for uid, c in sorted(inv.items()):
        r = raw[uid]
        if not os.path.exists(pano_path(a.panos, r["pano_id"])):
            continue
        assert verdict(r.get("validations")) == c["verdict"], uid
        d = float(c["d_current_m"]) if c["d_current_m"] else None
        near = d is not None and d <= MATCH_M
        cls = ("crowd_pos" if near else "crowd_conflict") if c["verdict"] != "incorrect" else \
            ("crowd_conflict" if near else "crowd_neg")
        try:
            x = float(r["pano_x"]) / float(r["pano_width"])
            y = float(r["pano_y"]) / float(r["pano_height"])
        except (ValueError, ZeroDivisionError):
            continue
        rows.append({"label_uid": uid, "pano_id": r["pano_id"], "class": cls, "verdict": c["verdict"],
                     "d_sdot_m": c["d_current_m"], "x": round(x, 6), "y": round(y, 6),
                     "pano_width": r["pano_width"], "pano_height": r["pano_height"],
                     "ps_camera_heading": r["camera_heading"], "label_lat": r["latitude"], "label_lng": r["longitude"],
                     "image_capture_date": r["image_capture_date"]})
    assert len({r["label_uid"] for r in rows}) == len(rows)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, SELECTION), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    # pano metadata (position for the SDOT arm, heading, capture date), cached and resumable
    mp = os.path.join(a.out, META)
    meta = {m["pano_id"]: m for m in read_jsonl(mp)}
    todo = sorted({r["pano_id"] for r in rows} - set(meta))
    lock = threading.Lock()

    def one(pid):
        try:
            p = sv.find_panorama_by_id(pid)
        except Exception as e:  # noqa: BLE001 -- retried on the next run
            log(f"metadata {pid} failed: {e}")
            return
        m = {"pano_id": pid, "found": p is not None}
        if p is not None:
            m.update({"lat": p.lat, "lon": p.lon, "heading_deg": math.degrees(p.heading) % 360.0,
                      "date": f"{p.date.year}-{p.date.month:02d}" if p.date else None})
        with lock:
            meta[pid] = m
            with open(mp, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(m) + "\n")

    log(f"{len(rows)} labels on {len({r['pano_id'] for r in rows})} stored panos; metadata for {len(todo)}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i and i % 500 == 0:
                log(f"  {i}/{len(todo)}")
    # Project Sidewalk's own pano record (GET /backupImage/<id>/metadata): the frame the crowd labelled in, and the
    # only source for the ~half of stored panos GSV no longer serves. Sequential, spaced, 429 honoured, cached.
    import urllib.error
    import urllib.request
    pp = os.path.join(a.out, PS_META)
    psm = {m["pano_id"]: m for m in read_jsonl(pp)}
    todo = sorted({r["pano_id"] for r in rows} - set(psm))
    log(f"PS metadata for {len(todo)} panos from {PS_SERVER}")
    for i, pid in enumerate(todo):
        url = f"{PS_SERVER}/backupImage/{pid}/metadata"
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    j = json.loads(resp.read().decode("utf-8"))
                m = {"pano_id": pid, "found": True, "lat": j["lat"], "lon": j["lng"],
                     "heading_deg": j["camera_heading"], "date": j.get("capture_date"),
                     "width": j.get("width"), "height": j.get("height")}
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(int(e.headers.get("Retry-After") or 30))
                    continue
                m = {"pano_id": pid, "found": False, "http": e.code}
                break
            except Exception as e:  # noqa: BLE001 -- transient: retry, then leave uncached
                log(f"PS metadata {pid}: {e}")
                time.sleep(5 * (attempt + 1))
                m = None
        if m is not None:
            psm[pid] = m
            with open(pp, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(m) + "\n")
        time.sleep(PS_SPACING_S)
        if i and i % 500 == 0:
            log(f"  {i}/{len(todo)}")
    c = {}
    for r in rows:
        c[r["class"]] = c.get(r["class"], 0) + 1
    log(f"classes {c}; GSV metadata for {sum(m.get('found', False) for m in meta.values())} of {len(meta)}, "
        f"PS metadata for {sum(m.get('found', False) for m in psm.values())} of {len(psm)}")


def merged_meta(out):
    """pano_id -> metadata: Project Sidewalk's record first (the labelling frame), else streetlevel's."""
    sl = {m["pano_id"]: m for m in read_jsonl(os.path.join(out, META))}
    ps = {m["pano_id"]: m for m in read_jsonl(os.path.join(out, PS_META))}
    res = {}
    for pid in set(sl) | set(ps):
        if ps.get(pid, {}).get("found"):
            res[pid] = {**ps[pid], "source": "ps"}
        elif sl.get(pid, {}).get("found"):
            res[pid] = {**sl[pid], "source": "streetlevel"}
        else:
            res[pid] = {"pano_id": pid, "found": False}
    return res, sl, ps


# ---------------------------------------------------------------- place

def cmd_place(a):
    import vistas_crosswalk_probe as probe
    from equirect_tiling import default_views, equirect_to_perspective, perspective_point_to_equirect
    _, feats, _ = load_inventory(a.inventory)
    allp = all_sdot_points(feats)
    g_cur = Grid([p for p in allp if p[2]["status"] in CURRENT])
    g_other = Grid([p for p in allp if p[2]["status"] not in CURRENT])
    g_osm = Grid(osm_marked_nodes(a.census))
    sel = list(csv.DictReader(open(os.path.join(a.out, SELECTION), encoding="utf-8", newline="")))
    meta, _, _ = merged_meta(a.out)
    by = {}
    for r in sel:
        by.setdefault(r["pano_id"], []).append(r)
    pids = sorted(by)
    gal = set(random.Random(SEED).sample(pids, min(GALLERY_N, len(pids))))
    if a.limit:
        pids = pids[:a.limit]
    out_p = os.path.join(a.out, PLACEMENTS)
    done = set()
    if os.path.exists(out_p):
        done = {r["pano_id"] for r in csv.DictReader(open(out_p, encoding="utf-8", newline=""))}
    fields = ["pano_id", "kind", "id", "capture_date", "x", "y", "range_m", "on_paint", "near_px_dist", "comp_found",
              "b2_x", "b2_y", "shift", "comp_area_px", "box", "incomplete_reasons"]
    new = not os.path.exists(out_p)
    fh = open(out_p, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    if new:
        w.writeheader()
    thumbs = os.path.join(a.out, "thumbs")
    os.makedirs(thumbs, exist_ok=True)
    seg = probe.Segmenter()
    log(f"{len(pids)} panos ({len(done)} done); segmenter {seg.backend}, revision {seg.revision}")
    t0 = time.time()
    for k, pid in enumerate(pids):
        if pid in done:
            continue
        m = meta.get(pid) or {}
        pano = load_pano(pano_path(a.panos, pid))
        labs = by[pid]
        date = m.get("date") or (labs[0]["image_capture_date"] or "")[:7]
        out = []
        # crowd clicks: Vistas at the human pixel
        for r in labs:
            x, y = float(r["x"]), float(r["y"])
            v = vistas_at(seg, pano, x, y)
            out.append(({"kind": r["class"], "id": r["label_uid"], "x": x, "y": y, "range_m": ""}, v))
        # SDOT crosswalks nobody clicked on this pano
        reasons = []
        if m.get("found"):
            pos = [(float(r["label_lat"]), float(r["label_lng"])) for r in labs if r["class"] == "crowd_pos"]
            for d, it in g_cur.within(m["lat"], m["lon"], INCLUSION_M):
                if any(haversine_m(it[0], it[1], la, ln) <= MATCH_M for la, ln in pos):
                    continue
                x, y, rng, _ = project(m["lat"], m["lon"], m["heading_deg"], it[0], it[1])
                v = vistas_at(seg, pano, x, y)
                kind = "sdot_b2" if v["comp_found"] else "sdot_ignore"
                out.append(({"kind": kind, "id": f"sdot:{it[2]['objectid']}", "x": x, "y": y,
                             "range_m": round(rng, 2)}, v))
            for d, it in g_osm.within(m["lat"], m["lon"], INCLUSION_M):
                if not g_cur.within(it[0], it[1], MATCH_M) and \
                        not any(haversine_m(it[0], it[1], la, ln) <= MATCH_M for la, ln in pos):
                    reasons.append(f"osm_paint_unaccounted:osm:{it[2]}")
            for d, it in g_other.within(m["lat"], m["lon"], INCLUSION_M):
                reasons.append(f"sdot_{it[2]['status'] or 'empty'}_nearby:sdot:{it[2]['objectid']}")
        else:
            reasons.append("no_pano_metadata")
        # Completeness: segment the whole pano (the Phase 0 probe's 6-view rig) and turn every crosswalk component
        # no mark explains into an ignore region, so paint that neither the crowd nor SDOT recorded is never
        # trained as background (found on the smoke test: SDOT had no point for a leg right in front of a camera).
        comps = []
        for view in default_views():
            sg, pr = seg.seg_prob(equirect_to_perspective(pano, view))
            comps += probe.view_components(sg, pr, view, perspective_point_to_equirect)
        marks = [(row["x"], row["y"]) for row, _ in out]
        marks += [(v["b2_x"], v["b2_y"]) for _, v in out if v["comp_found"]]
        # Unexplained components stay per view (not merged): a near crosswalk splits across overlapping views, and
        # each keeps an approximate pano-space box from its centroid and left/right extreme pixels, so the ignore
        # region covers its extent. Overlapping ignore boxes are harmless.
        unexpl = [c for c in comps if not explained(c, marks)]
        for i, c in enumerate(unexpl):
            ys = (c["y"], c["y_left"], c["y_right"])
            box = (round(c["x_left"], 5), round(c["x_right"], 5), round(max(0.0, min(ys) - BOX_PAD_Y), 5),
                   round(min(1.0, max(ys) + BOX_PAD_Y), 5))
            out.append(({"kind": "vistas_unexplained", "id": f"vistas:{c['view_yaw']:.0f}:{i}", "x": c["x"],
                         "y": c["y"], "range_m": "", "box": ";".join(map(str, box))},
                        {"on_paint": 1, "near_px_dist": 0.0, "comp_found": 1, "b2_x": round(c["x"], 6),
                         "b2_y": round(c["y"], 6), "shift": 0.0, "comp_area_px": c["area_px"]}))
        for row, v in out:
            w.writerow({"pano_id": pid, "capture_date": date, "box": "", **row, "x": round(row["x"], 6), "y": round(row["y"], 6),
                        "on_paint": v["on_paint"], "near_px_dist": v["near_px_dist"], "comp_found": v["comp_found"],
                        "b2_x": v["b2_x"], "b2_y": v["b2_y"], "shift": v["shift"], "comp_area_px": v["comp_area_px"],
                        "incomplete_reasons": "|".join(reasons)})
        fh.flush()
        if pid in gal:
            draw_pano(pano, out, os.path.join(thumbs, pid + ".jpg"))
        if k % 100 == 0:
            el = time.time() - t0
            log(f"  pano {k + 1}/{len(pids)} ({el / 60:.1f} min)")
    fh.close()
    log(f"placements -> {out_p}")


def explained(c, marks, r=SNAP_R, pad_x=0.01, pad_y=0.04):
    """A per-view Vistas component is explained by a mark within r of its centroid (matcher metric), or by a mark
    inside its horizontal extent (x_left..x_right, seam-aware, padded) and within pad_y of it vertically: a long
    near crosswalk's centroid can sit far from the crowd click on one end of it."""
    for mx, my in marks:
        if pano_dist(mx, my, c["x"], c["y"]) <= r:
            return True
        lo, hi = c["x_left"] - pad_x, c["x_right"] + pad_x
        inside = (lo <= mx <= hi) if lo <= hi else (mx >= lo or mx <= hi)
        if inside and abs(my - c["y"]) <= pad_y:
            return True
    return False


def draw_pano(pano, out, path, width=1600):
    """Full-pano thumbnail with every mark: crowd_pos green ring, crowd_neg red X, conflicts and sdot_ignore orange
    ring (ignore), sdot_b2 cyan cross (with a thin line from B1)."""
    from PIL import ImageDraw
    th = pano.resize((width, width // 2))
    dr = ImageDraw.Draw(th)
    W, H = th.size
    for row, v in out:
        x, y = row["x"] * W, row["y"] * H
        if row["kind"] == "crowd_pos":
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 230, 0), width=3)
        elif row["kind"] == "crowd_neg":
            dr.line([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 0), width=3)
            dr.line([x - 8, y + 8, x + 8, y - 8], fill=(255, 0, 0), width=3)
        elif row["kind"] in ("crowd_conflict", "sdot_ignore"):
            dr.ellipse([x - 11, y - 11, x + 11, y + 11], outline=(255, 150, 0), width=2)
        elif row["kind"] == "vistas_unexplained":
            xl, xr, y0, y1 = (float(t) for t in row["box"].split(";"))
            for a0, a1 in (((xl, xr),) if xl <= xr else ((xl, 1.0), (0.0, xr))):
                dr.rectangle([a0 * W, y0 * H, a1 * W, y1 * H], outline=(255, 0, 255), width=2)
        elif row["kind"] == "sdot_b2":
            bx, by = v["b2_x"] * W, v["b2_y"] * H
            dr.line([x, y, bx, by], fill=(0, 200, 255), width=1)
            dr.line([bx - 9, by, bx + 9, by], fill=(0, 255, 255), width=3)
            dr.line([bx, by - 9, bx, by + 9], fill=(0, 255, 255), width=3)
    th.save(path, quality=85)


# ---------------------------------------------------------------- export / report / gallery

def load_rows(out):
    rows = list(csv.DictReader(open(os.path.join(out, PLACEMENTS), encoding="utf-8", newline="")))
    for r in rows:
        r["on_paint"] = int(r["on_paint"])
        r["comp_found"] = int(r["comp_found"])
        r["near"] = float(r["near_px_dist"]) if r["near_px_dist"] else None
        r["year"] = int(r["capture_date"][:4]) if r["capture_date"][:4].isdigit() else None
    return rows


def cmd_export(a):
    rows = load_rows(a.out)
    by = {}
    for r in rows:
        by.setdefault(r["pano_id"], []).append(r)
    n = 0
    with open(os.path.join(a.out, EXPORT), "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(by):
            rs = by[pid]
            rec = {"pano_id": pid, "city": CITY, "label_type": "Crosswalk", "capture_date": rs[0]["capture_date"],
                   "points": [], "ignore": [], "negatives": [], "ignore_radius": SNAP_R,
                   "complete": not rs[0]["incomplete_reasons"],
                   "incomplete_reasons": [x for x in rs[0]["incomplete_reasons"].split("|") if x]}
            for r in rs:
                if r["kind"] == "crowd_pos":
                    rec["points"].append({"x": float(r["x"]), "y": float(r["y"]), "source": "crowd", "id": r["id"]})
                elif r["kind"] == "sdot_b2":
                    rec["points"].append({"x": float(r["b2_x"]), "y": float(r["b2_y"]), "source": "sdot_b2",
                                          "id": r["id"]})
                elif r["kind"] == "crowd_neg":
                    rec["negatives"].append({"x": float(r["x"]), "y": float(r["y"]), "id": r["id"]})
                else:
                    ig = {"x": float(r["x"]), "y": float(r["y"]), "kind": r["kind"], "id": r["id"]}
                    if r.get("box"):
                        ig["box"] = [float(v) for v in r["box"].split(";")]   # x_left, x_right, y_top, y_bottom
                    rec["ignore"].append(ig)
            f.write(json.dumps(rec) + "\n")
            n += 1
    log(f"{n} pano records -> {EXPORT}")


def cmd_report(a):
    rows = load_rows(a.out)
    recs = read_jsonl(os.path.join(a.out, EXPORT))
    meta, slm, psm = merged_meta(a.out)
    sel = list(csv.DictReader(open(os.path.join(a.out, SELECTION), encoding="utf-8", newline="")))
    pilot_pl = pilot.load_placements(os.path.join(pilot.REPO, "analysis_out", "stage1_crosswalk_sdot_274"))
    b1 = [r for r in pilot_pl if r["range_m"] <= 20]
    L = []
    w = L.append
    w("# Arm C: SDOT-confirmed crowd Crosswalk labels on stored Seattle panos (RampNet#274)\n")
    w(f"Generated by `scripts/analysis/stage1_crosswalk_sdot_274/crowd_arm.py report`. {len(sel)} crowd Crosswalk "
      f"labels inside the Seattle regions on {len({r['pano_id'] for r in sel})} panos held in Project Sidewalk's "
      f"store (makelab2, native resolution, copied read-only); SDOT and crowd verdicts as on branch "
      f"crosswalk-inventory-274; {len(recs)} panos placed.\n")
    w("## 1. The label set\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    w("| kind | count | becomes |\n|---|---|---|")
    for k, becomes in (("crowd_pos", "positive (the click)"), ("sdot_b2", "positive (Vistas-snapped SDOT point)"),
                       ("crowd_neg", "hard negative"), ("crowd_conflict", "ignore region"),
                       ("sdot_ignore", "ignore region"),
                       ("vistas_unexplained", "ignore region (Vistas paint no mark explains)")):
        w(f"| `{k}` | {kinds.get(k, 0)} | {becomes} |")
    npts = [len(r["points"]) for r in recs]
    comp = [r for r in recs if r["complete"]]
    w(f"\nPer pano: median {pct(npts, .5):.0f} positive points (p90 {pct(npts, .9):.0f}); {sum(n == 0 for n in npts)} "
      f"panos with none (only negatives or ignore). Complete panos (no unaccounted OSM paint or non-current SDOT "
      f"point within {INCLUSION_M:.0f} m): {frac(len(comp), len(recs))}.\n")
    reasons = {}
    for r in recs:
        for k in {x.split(":")[0] for x in r["incomplete_reasons"]}:
            reasons[k] = reasons.get(k, 0) + 1
    if reasons:
        w("Incomplete because: " + ", ".join(f"`{k}` {n}" for k, n in sorted(reasons.items(), key=lambda t: -t[1])) + "\n")
    yrs = {}
    for r in recs:
        y = (r["capture_date"] or "?")[:4]
        yrs[y] = yrs.get(y, 0) + 1
    w("Panos by capture year: " + ", ".join(f"{y} {n}" for y, n in sorted(yrs.items())) + "\n")

    w("## 2. Calibrating the pilot's paint metric against human clicks\n")
    w("Vistas run in a 90° view centred on the crowd click, exactly as `pilot.vistas_at` does for B1. A person "
      "clicked paint here, so this is the ceiling the exact-pixel metric can reach; it is a calibration of the "
      "metric, not a re-score of the pilot's pre-registered verdict.\n")
    w("| points | n | exact pixel on paint | paint within 0.022 | component within 0.05 |\n|---|---|---|---|---|")

    def line(name, s, near_key="near"):
        w(f"| {name} | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} | "
          f"{frac(sum(1 for r in s if r[near_key] is not None and r[near_key] <= TOL_R), len(s))} | "
          f"{frac(sum(r['comp_found'] for r in s), len(s))} |")

    cp = [r for r in rows if r["kind"] == "crowd_pos"]
    line("crowd_pos clicks (all)", cp)
    sl = {r["label_uid"]: r for r in sel}
    line("crowd_pos, validated correct", [r for r in cp if sl[r["id"]]["verdict"] == "correct"])
    line("crowd_pos, undecided", [r for r in cp if sl[r["id"]]["verdict"] == "undecided"])
    line("crowd_neg clicks (voted incorrect, no SDOT)", [r for r in rows if r["kind"] == "crowd_neg"])
    line("crowd_conflict clicks", [r for r in rows if r["kind"] == "crowd_conflict"])
    line("pilot B1, within 20 m (for comparison)", b1)
    line("this arm's SDOT-only B1, within 20 m", [r for r in rows if r["kind"] in ("sdot_b2", "sdot_ignore")
                                                  and r["range_m"] and float(r["range_m"]) <= 20])
    w("")
    w("crowd_pos clicks by capture year:\n")
    w("| year | n | exact pixel on paint | component within 0.05 |\n|---|---|---|---|")
    for y in sorted({r["year"] for r in cp if r["year"]}):
        s = [r for r in cp if r["year"] == y]
        w(f"| {y} | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} | "
          f"{frac(sum(r['comp_found'] for r in s), len(s))} |")
    w("")
    w("## 3. Heading agreement\n")
    diffs = []
    for r in sel:
        m = slm.get(r["pano_id"])   # rawLabels' camera_heading vs streetlevel (PS metadata is the same record)
        if m and m.get("found") and r["ps_camera_heading"]:
            dd = abs((float(r["ps_camera_heading"]) - m["heading_deg"] + 180) % 360 - 180)
            diffs.append(dd)
    w(f"rawLabels `camera_heading` vs streetlevel's heading, per label: median {pct(diffs, .5):.3f}°, "
      f"p99 {pct(diffs, .99):.3f}°, max {max(diffs) if diffs else float('nan'):.3f}° over {len(diffs)} labels. "
      "Pano metadata source: " + ", ".join(f"{k} {sum(1 for m in meta.values() if m.get('source') == k)}"
                                           for k in ("ps", "streetlevel")) +
      f"; none for {sum(1 for m in meta.values() if not m.get('found'))} (SDOT arm skipped, marked incomplete). "
      f"GSV no longer serves {sum(1 for m in slm.values() if not m.get('found'))} of {len(slm)} stored panos.\n")
    both = [(psm[p], slm[p]) for p in psm if psm[p].get("found") and slm.get(p, {}).get("found")]
    dpos = [haversine_m(x["lat"], x["lon"], y["lat"], y["lon"]) for x, y in both]
    dhd = [abs((x["heading_deg"] - y["heading_deg"] + 180) % 360 - 180) for x, y in both]
    w(f"Where both exist ({len(both)} panos), PS vs streetlevel: position median {pct(dpos, .5):.2f} m (max "
      f"{max(dpos) if dpos else float('nan'):.2f}), heading median {pct(dhd, .5):.3f}° (max "
      f"{max(dhd) if dhd else float('nan'):.3f}°).\n")
    w("## Replication\n")
    w("```bash\nC=scripts/analysis/stage1_crosswalk_sdot_274/crowd_arm.py; OUT=analysis_out/stage1_crosswalk_crowd_274\n"
      "INV=<crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274; CEN=<Phase 0 census OUT>\n"
      "PANOS=<copy of makelab2 sidewalk_panos/Panoramas/seattle-wa for the selected ids>\n"
      "python $C select --inventory $INV --census $CEN --panos $PANOS --out $OUT   # network: streetlevel metadata\n"
      "python $C place --inventory $INV --census $CEN --panos $PANOS --out $OUT    # GPU: Vistas\n"
      "python $C export --out $OUT && python $C report --out $OUT && python $C gallery --out $OUT\n```\n")
    w("The pano copy: one read-only `tar cf - --verbatim-files-from -T -` stream from makelab2 over `./<id[:2]>/<id>.jpg` "
      "paths (ids may start with `-`).\n")
    w("| number | lives in |\n|---|---|")
    w("| label classes, verdicts, crowd pixels | `selection.csv` |")
    w("| pano position, heading, date | `crowd_meta.jsonl` |")
    w("| every mark, on-paint, B2 | `placements.csv` (one row per pano × label or SDOT crosswalk) |")
    w("| the training set | `crowd_labels.jsonl` (one record per pano) |")
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log(f"wrote {a.out}/report.md")


def cmd_gallery(a):
    import html
    recs = {r["pano_id"]: r for r in read_jsonl(os.path.join(a.out, EXPORT))}
    ids = sorted(p[:-4] for p in os.listdir(os.path.join(a.out, "thumbs")) if p.endswith(".jpg") and p[:-4] in recs)
    cards = []
    for pid in ids:
        r = recs[pid]
        src = sum(1 for p in r["points"] if p["source"] == "crowd"), sum(1 for p in r["points"] if p["source"] == "sdot_b2")
        cap = (f"{html.escape(pid)} · {html.escape(r['capture_date'] or '?')} · crowd {src[0]} · SDOT/B2 {src[1]} · "
               f"ignore {len(r['ignore'])} · neg {len(r['negatives'])}" + ("" if r["complete"] else " · incomplete"))
        cards.append(f'<figure><img src="thumbs/{html.escape(pid)}.jpg" alt="Pano {html.escape(pid)} with crosswalk '
                     f'marks" loading="lazy"><figcaption>{cap}</figcaption></figure>')
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crowd crosswalk labels</title>
  <style>
    :root {{ --bg: #fafafa; --fg: #222; --muted: #666; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #161616; --fg: #e8e8e8; --muted: #aaa; }} }}
    body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; margin: 16px; }}
    figure {{ margin: 0 0 18px; }}
    img {{ width: 100%; max-width: 1600px; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Arm C: SDOT-confirmed crowd crosswalk labels</h1>
  <p>{len(ids)} seeded panos. Green ring: crowd click (positive). Cyan cross: SDOT crosswalk nobody clicked, placed
  by the Vistas snap (thin line from the projected point). Orange ring: ignore region (crowd and SDOT disagree,
  or Vistas found nothing at an SDOT point). Magenta square: Vistas crosswalk paint that no label explains (ignore
  region). Red X: crowd click voted incorrect with no SDOT crosswalk (hard negative).</p>
  {chr(10).join(cards)}
</body>
</html>
"""
    with open(os.path.join(a.out, "gallery.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    log(f"gallery -> {os.path.join(a.out, 'gallery.html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "place", "export", "report", "gallery"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        if name in ("select", "place"):
            s.add_argument("--inventory", required=True)
            s.add_argument("--census", required=True)
            s.add_argument("--panos", required=True)
        if name == "place":
            s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"select": cmd_select, "place": cmd_place, "export": cmd_export, "report": cmd_report,
     "gallery": cmd_gallery}[a.cmd](a)


if __name__ == "__main__":
    main()
