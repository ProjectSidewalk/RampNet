"""Arm C for a city with NO crosswalk inventory: crowd labels confirmed by the Seattle-calibrated rule (RampNet#274).

Seattle's arm (`crowd_arm.py`) uses SDOT to confirm crowd labels and to supply the crosswalks nobody clicked. Other
cities in Project Sidewalk's store (Chicago 28k crowd Crosswalk labels, Teaneck, Amsterdam, Burnaby, ...) have
no inventory, so here:
  * a crowd label not voted incorrect is a POSITIVE when it passes the confirmer rule `confirm_rule.py` adopted in
    Seattle (signals V validated, O an OSM marked-or-signalised node within 10 m, P Vistas paint at the click's
    exact pixel, T paint within 0.022 width); otherwise an IGNORE point (`crowd_unconfirmed`);
  * a label voted incorrect with no OSM marked node within 10 m and no Vistas paint within 0.022 is a hard
    NEGATIVE point; otherwise an ignore point (`crowd_conflict`);
  * every pano is incomplete by construction (no inventory says what else is in view), so it exports as `sparse`;
    the full-pano Vistas pass still adds `vistas_unexplained` ignore points, which matter only near positives;
  * pano position/heading/date come from the city's own PS server (`/backupImage/<id>/metadata`), resolved from
    `/v3/api/cities`.
Labels are keyed `<city>:<label_id>`.

  python crowd_city.py select --city teaneck-nj --census <Phase 0 census OUT> --panos <store copy> --out OUT
  python crowd_city.py place  --city teaneck-nj --census ... --panos ... --out OUT [--limit N]
  python crowd_city.py export --city teaneck-nj --confirm "<rule adopted by confirm_rule.py>" --out OUT
The output layout matches crowd_arm.py's (selection.csv, ps_meta.jsonl, placements.csv, crowd_labels.jsonl), so
`export_stage2.py build --arm OUT=<panos>` takes it unchanged.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crowd_arm  # noqa: E402
from crowd_arm import draw_pano, explained, load_pano, pano_path, verdict  # noqa: E402
from pilot import TOL_R, Grid, log, read_jsonl, vistas_at  # noqa: E402
from confirm_rule import evaluate  # noqa: E402
import export_stage2  # noqa: E402

CITIES_URL = "https://sidewalk-sea.cs.washington.edu/v3/api/cities"
MATCH_M = 10.0
GALLERY_N = 40
OSM_MARKED = crowd_arm.pilot.OSM_MARKED
OSM_UNMARKED = crowd_arm.pilot.OSM_UNMARKED


def city_server(city, out):
    p = os.path.join(out, "city.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))["url"]
    with urllib.request.urlopen(urllib.request.Request(CITIES_URL, headers={"User-Agent": crowd_arm.UA}),
                                timeout=60) as r:
        cities = json.loads(r.read().decode("utf-8"))["cities"]
    c = next(c for c in cities if c["city_id"] == city)
    json.dump({"city": city, "url": c["url"], "visibility": c.get("visibility")},
              open(p, "w", encoding="utf-8"), indent=1)
    return c["url"]


def osm_marked(census, city):
    out = []
    for fp in glob.glob(os.path.join(census, "osm", city, "Crosswalk", "*.json")):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            t = e.get("tags") or {}
            if e.get("type") != "node" or t.get("highway") != "crossing":
                continue
            m, c = t.get("crossing:markings"), t.get("crossing")
            cls = ("unmarked" if m == "no" else "marked") if m is not None else \
                ("marked" if c in OSM_MARKED else "unmarked" if c in OSM_UNMARKED else "unknown")
            if cls == "marked" or (cls == "unknown" and (c == "traffic_signals" or t.get("crossing:signals") == "yes")):
                out.append((e["lat"], e["lon"], e["id"]))
    return out


def cmd_select(a):
    os.makedirs(a.out, exist_ok=True)
    server = city_server(a.city, a.out)
    g_osm = Grid(osm_marked(a.census, a.city), lat0=45.0)
    rows = []
    with open(os.path.join(a.census, "rawlabels", f"{a.city}_Crosswalk.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("latitude") or not os.path.exists(pano_path(a.panos, r["pano_id"])):
                continue
            try:
                x = float(r["pano_x"]) / float(r["pano_width"])
                y = float(r["pano_y"]) / float(r["pano_height"])
            except (ValueError, ZeroDivisionError):
                continue
            v = verdict(r.get("validations"))
            lat, lng = float(r["latitude"]), float(r["longitude"])
            rows.append({"label_uid": f"{a.city}:{r['label_id']}", "pano_id": r["pano_id"],
                         "class": "crowd_incorrect" if v == "incorrect" else "crowd_candidate", "verdict": v,
                         "osm_marked_10m": int(bool(g_osm.within(lat, lng, MATCH_M))), "x": round(x, 6),
                         "y": round(y, 6), "label_lat": lat, "label_lng": lng,
                         "image_capture_date": r["image_capture_date"]})
    assert len({r["label_uid"] for r in rows}) == len(rows), "duplicate label_uid"
    with open(os.path.join(a.out, "selection.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    mp = os.path.join(a.out, crowd_arm.PS_META)
    meta = {m["pano_id"]: m for m in read_jsonl(mp)}
    todo = sorted({r["pano_id"] for r in rows} - set(meta))
    log(f"{len(rows)} labels on {len({r['pano_id'] for r in rows})} stored panos; PS metadata for {len(todo)} from {server}")
    crowd_arm.PS_SERVER = server                     # export_stage2.ps_metadata reads it from crowd_arm
    for i, pid in enumerate(todo):
        m = export_stage2.ps_metadata(pid)
        if m is not None:
            meta[pid] = m
            with open(mp, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(m) + "\n")
        time.sleep(crowd_arm.PS_SPACING_S)
        if i and i % 500 == 0:
            log(f"  {i}/{len(todo)}")
    open(os.path.join(a.out, crowd_arm.META), "a").close()  # crowd_arm.merged_meta reads both caches


def cmd_place(a):
    import vistas_crosswalk_probe as probe
    from equirect_tiling import default_views, equirect_to_perspective, perspective_point_to_equirect
    sel = list(csv.DictReader(open(os.path.join(a.out, "selection.csv"), encoding="utf-8", newline="")))
    by = {}
    for r in sel:
        by.setdefault(r["pano_id"], []).append(r)
    pids = sorted(by)
    gal = set(random.Random(crowd_arm.SEED).sample(pids, min(GALLERY_N, len(pids))))
    if a.limit:
        pids = pids[:a.limit]
    out_p = os.path.join(a.out, crowd_arm.PLACEMENTS)
    done = set()
    if os.path.exists(out_p):
        done = {r["pano_id"] for r in csv.DictReader(open(out_p, encoding="utf-8", newline=""))}
    fields = ["pano_id", "kind", "id", "capture_date", "x", "y", "range_m", "on_paint", "near_px_dist",
              "comp_found", "b2_x", "b2_y", "shift", "comp_area_px", "box", "incomplete_reasons"]
    new = not os.path.exists(out_p)
    fh = open(out_p, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    if new:
        w.writeheader()
    thumbs = os.path.join(a.out, "thumbs")
    os.makedirs(thumbs, exist_ok=True)
    meta = {m["pano_id"]: m for m in read_jsonl(os.path.join(a.out, crowd_arm.PS_META))}
    seg = probe.Segmenter()
    log(f"{a.city}: {len(pids)} panos ({len(done)} done); segmenter {seg.backend}")
    t0 = time.time()
    for k, pid in enumerate(pids):
        if pid in done:
            continue
        pano = load_pano(pano_path(a.panos, pid))
        date = (meta.get(pid) or {}).get("date") or (by[pid][0]["image_capture_date"] or "")[:7]
        out = []
        for r in by[pid]:
            v = vistas_at(seg, pano, float(r["x"]), float(r["y"]))
            out.append(({"kind": r["class"], "id": r["label_uid"], "x": float(r["x"]), "y": float(r["y"]),
                         "range_m": ""}, v))
        comps = []
        for view in default_views():
            sg, pr = seg.seg_prob(equirect_to_perspective(pano, view))
            comps += probe.view_components(sg, pr, view, perspective_point_to_equirect)
        marks = [(row["x"], row["y"]) for row, _ in out]
        for i, c in enumerate(cc for cc in comps if not explained(cc, marks)):
            ys = (c["y"], c["y_left"], c["y_right"])
            box = (round(c["x_left"], 5), round(c["x_right"], 5), round(max(0.0, min(ys) - crowd_arm.BOX_PAD_Y), 5),
                   round(min(1.0, max(ys) + crowd_arm.BOX_PAD_Y), 5))
            out.append(({"kind": "vistas_unexplained", "id": f"vistas:{c['view_yaw']:.0f}:{i}", "x": c["x"],
                         "y": c["y"], "range_m": "", "box": ";".join(map(str, box))},
                        {"on_paint": 1, "near_px_dist": 0.0, "comp_found": 1, "b2_x": round(c["x"], 6),
                         "b2_y": round(c["y"], 6), "shift": 0.0, "comp_area_px": c["area_px"]}))
        for row, v in out:
            w.writerow({"pano_id": pid, "capture_date": date, "box": "", **row, "x": round(row["x"], 6),
                        "y": round(row["y"], 6), "on_paint": v["on_paint"], "near_px_dist": v["near_px_dist"],
                        "comp_found": v["comp_found"], "b2_x": v["b2_x"], "b2_y": v["b2_y"], "shift": v["shift"],
                        "comp_area_px": v["comp_area_px"], "incomplete_reasons": "no_inventory"})
        fh.flush()
        if pid in gal:
            draw_pano(pano, [(dict(row, kind="crowd_pos" if row["kind"] == "crowd_candidate" else row["kind"]), v)
                             for row, v in out], os.path.join(thumbs, pid + ".jpg"))
        if k % 100 == 0:
            log(f"  pano {k + 1}/{len(pids)} ({(time.time() - t0) / 60:.1f} min)")
    fh.close()


def cmd_export(a):
    sel = {r["label_uid"]: r for r in csv.DictReader(open(os.path.join(a.out, "selection.csv"), encoding="utf-8"))}
    rows = crowd_arm.load_rows(a.out)
    by = {}
    for r in rows:
        by.setdefault(r["pano_id"], []).append(r)
    counts = {}
    with open(os.path.join(a.out, crowd_arm.EXPORT), "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(by):
            rec = {"pano_id": pid, "city": a.city, "label_type": "Crosswalk", "capture_date": by[pid][0]["capture_date"],
                   "points": [], "ignore": [], "negatives": [], "ignore_radius": crowd_arm.SNAP_R, "complete": False,
                   "incomplete_reasons": ["no_inventory"], "confirm_rule": a.confirm}
            for r in by[pid]:
                kind = r["kind"]
                if kind in ("crowd_candidate", "crowd_incorrect"):
                    s = sel[r["id"]]
                    sig = {"V": s["verdict"] == "correct", "O": s["osm_marked_10m"] == "1", "P": r["on_paint"] == 1,
                           "T": r["near"] is not None and r["near"] <= TOL_R}
                    if kind == "crowd_candidate":
                        kind = "crowd_pos" if evaluate(a.confirm, sig) else "crowd_unconfirmed"
                    else:
                        kind = "crowd_conflict" if (sig["O"] or sig["T"]) else "crowd_neg"
                counts[kind] = counts.get(kind, 0) + 1
                pt = {"x": float(r["x"]), "y": float(r["y"]), "id": r["id"]}
                if kind == "crowd_pos":
                    rec["points"].append({**pt, "source": "crowd"})
                elif kind == "crowd_neg":
                    rec["negatives"].append(pt)
                else:
                    ig = {**pt, "kind": kind}
                    if r.get("box"):
                        ig["box"] = [float(v) for v in r["box"].split(";")]
                    rec["ignore"].append(ig)
            f.write(json.dumps(rec) + "\n")
    json.dump({"city": a.city, "confirm_rule": a.confirm, "counts": counts},
              open(os.path.join(a.out, "export_counts.json"), "w", encoding="utf-8"), indent=1)
    log(f"{a.city}: {len(by)} panos; {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "place", "export"):
        s = sub.add_parser(name)
        s.add_argument("--city", required=True)
        s.add_argument("--out", required=True)
        if name in ("select", "place"):
            s.add_argument("--census", required=True)
            s.add_argument("--panos", required=True)
        if name == "place":
            s.add_argument("--limit", type=int, default=0)
        if name == "export":
            s.add_argument("--confirm", required=True, help="the rule confirm_rule.py adopted, e.g. 'V|(O&T)'")
    a = ap.parse_args()
    {"select": cmd_select, "place": cmd_place, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
