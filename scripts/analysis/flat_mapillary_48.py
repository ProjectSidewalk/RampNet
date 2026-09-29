"""Flat (non-panoramic) Mapillary imagery around curb ramps (#48): census, fetch, and the
inputs of the per-corner 3D reconstruction.

In Richmond, VA most Mapillary imagery is ordinary phone / dashcam frames taken every few
metres along a drive, not 360 panoramas. This script counts it around each Richmond pool
ramp (the census) and fetches it for the corners that get reconstructed.

Subcommands, in order:

    # 1. ramp positions (desktop CPU; the committed captures table + the labeler's richmond
    #    results.jsonl for each capture's SfM position and heading)
    python scripts/analysis/flat_mapillary_48.py ramps \\
        --results ../sidewalk-auto-labeler/runs/richmond/results.jsonl
    #    -> analysis_out/flat_mapillary_3d/census/ramps.csv (committed)
    # 2. census (Mapillary Graph API, metadata only; the token is read from an .env file)
    python scripts/analysis/flat_mapillary_48.py census --env ../sidewalk-auto-labeler/.env
    #    -> census/images.csv, census/ramp_images.csv, census/per_ramp.csv, census/summary.json

**Ramp positions.** A pool ramp's world position is eval_sites' merged GT point
(``multiview_evidence_48.world_gt``). It is not committed as lat/lng, but the committed
capture table ``analysis_out/multiview_48/captures_R25.csv`` has, for every capture within
25 m of the ramp, the horizontal distance ``dist_m`` and the equirect column ``x_proj`` at
which ``geo.ground_point_to_pano`` places the ramp. With the capture's SfM position and
heading from the labeler's results.jsonl, bearing = heading + (x_proj - 0.5) * 360 and the
ramp is ``dist_m`` along it: exact up to the labeler's linearised frame. Every capture
gives an estimate; the median is kept and the spread is recorded (it should be ~0).

**The token** is read from the .env file at run time, sent in an ``Authorization``
header (never in a URL), and never printed, logged or written anywhere.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import crossview_align_48 as H  # noqa: E402

OUT = os.path.join(H.OUT_ROOT, "flat_mapillary_3d")
CENSUS = os.path.join(OUT, "census")
CAPTURES_CSV = os.path.join(H.OUT_ROOT, "multiview_48", "captures_R25.csv")
RAMPS_CSV = os.path.join(CENSUS, "ramps.csv")
CITY = "richmond"
M_PER_DEG_LAT = 111_320.0     # geo.METERS_PER_DEG_LAT in the labeler
RADIUS_M = 30.0               # census radius; 25 m is reported too
RADII = (15.0, 25.0, 30.0)
GRAPH = "https://graph.mapillary.com/images"
FIELDS = ["id", "is_pano", "camera_type", "captured_at", "make", "model", "sequence",
          "computed_geometry", "geometry", "computed_compass_angle", "compass_angle",
          "computed_rotation", "computed_altitude", "altitude", "merge_cc",
          "atomic_scale", "camera_parameters", "width", "height", "quality_score",
          "exif_orientation"]
# ``sfm_cluster`` is NOT in the bbox query: asking for it on a busy box answers HTTP 500
# "Service temporarily unavailable" every time (measured 2026-09-28, richmond:75, the same
# box answers 200 without it; id batches of 5-50 answer 500 too). It is fetched one image
# per call by ``cmd_census``, for the harness ramps' flat images only, as the sub-field
# sfm_cluster{id} (no signed URL).
MIN_INTERVAL_S = 0.2          # <= 5 requests / s, far under the documented limits
PAGE_LIMIT = 2000


# --------------------------------------------------------------------------- #
# small geometry
# --------------------------------------------------------------------------- #


def enu(lat0, lng0, lat, lng):
    """East, north metres of (lat, lng) about (lat0, lng0), the labeler's LocalFrame."""
    return ((lng - lng0) * M_PER_DEG_LAT * math.cos(math.radians(lat0)),
            (lat - lat0) * M_PER_DEG_LAT)


def offset(lat, lng, e, n):
    return (lat + n / M_PER_DEG_LAT,
            lng + e / (M_PER_DEG_LAT * math.cos(math.radians(lat))))


# --------------------------------------------------------------------------- #
# ramps
# --------------------------------------------------------------------------- #


def read_labeler_panos(path, want):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["pano"]
            if p["panorama_id"] in want:
                out[p["panorama_id"]] = p
    return out


def cmd_ramps(args):
    rows = [r for r in csv.DictReader(open(CAPTURES_CSV, encoding="utf-8", newline=""))
            if r["city"] == CITY]
    panos = read_labeler_panos(args.results, {r["pano_id"] for r in rows})
    missing = {r["pano_id"] for r in rows} - set(panos)
    if missing:
        raise SystemExit(f"{len(missing)} capture panos not in {args.results}")
    harness = defaultdict(list)
    for p in H.read_frozen_pairs():
        if p["city"] == CITY:
            harness[p["ramp_uid"]].append(p["pair_id"])
    by_ramp = defaultdict(list)
    for r in rows:
        p = panos[r["pano_id"]]
        b = math.radians(float(p["camera_heading"]) + (float(r["x_proj"]) - 0.5) * 360.0)
        d = float(r["dist_m"])
        # geo.detection_ground_point's linearised step, from the camera's own frame
        by_ramp[r["ramp_uid"]].append(offset(p["lat"], p["lng"], d * math.sin(b),
                                             d * math.cos(b)))
    out = []
    for uid in sorted(by_ramp, key=lambda u: int(u.split(":")[1])):
        pts = np.array(by_ramp[uid])
        lat, lng = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        spread = max(math.hypot(*enu(lat, lng, a, b)) for a, b in pts)
        out.append({"ramp_uid": uid, "lat": round(lat, 8), "lng": round(lng, 8),
                    "n_captures_run": len(pts), "estimate_spread_m": round(spread, 3),
                    "harness_pairs": ";".join(sorted(harness.get(uid, [])))})
    os.makedirs(CENSUS, exist_ok=True)
    write_csv(RAMPS_CSV, out, list(out[0]))
    spreads = [r["estimate_spread_m"] for r in out]
    print(f"{len(out)} ramps ({sum(bool(r['harness_pairs']) for r in out)} in harness pairs); "
          f"estimate spread max {max(spreads):.3f} m, median {np.median(spreads):.3f} m "
          f"-> {RAMPS_CSV}")


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def read_ramps():
    return list(csv.DictReader(open(RAMPS_CSV, encoding="utf-8", newline="")))


# --------------------------------------------------------------------------- #
# Mapillary Graph API (metadata)
# --------------------------------------------------------------------------- #


def read_token(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("MAPILLARY_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"MAPILLARY_ACCESS_TOKEN not found in {env_path}")


class Graph:
    """Throttled Graph API client. The token only ever travels in a header."""

    def __init__(self, token):
        import requests
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"OAuth {token}"
        self.last = 0.0
        self.calls = 0
        self.retries = 0

    def get(self, url, params, attempts=8, soft=None):
        """JSON of a GET. Retries 429 / 5xx with exponential back-off; after ``attempts``
        it exits, or returns None when ``soft`` (default: soft iff attempts < 8)."""
        soft = attempts < 8 if soft is None else soft
        delay = 2.0
        for _ in range(attempts):
            wait = MIN_INTERVAL_S - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.calls += 1
            try:
                r = self.s.get(url, params=params, timeout=60)
            except Exception as e:     # network error: message never includes the header
                print(f"  network error {type(e).__name__}; retry in {delay:.0f} s", flush=True)
                self.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                self.retries += 1
                print(f"  HTTP {r.status_code}; back off {delay:.0f} s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise SystemExit(f"Graph API HTTP {r.status_code}: {r.text[:300]}")
        if soft:
            return None
        raise SystemExit("Graph API: out of retries")

    def images_in_bbox(self, lat, lng, half_m):
        dlat = half_m / M_PER_DEG_LAT
        dlng = half_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
        bbox = f"{lng - dlng:.7f},{lat - dlat:.7f},{lng + dlng:.7f},{lat + dlat:.7f}"
        js = self.get(GRAPH, {"bbox": bbox, "fields": ",".join(FIELDS), "limit": PAGE_LIMIT})
        data = js.get("data", [])
        if len(data) >= PAGE_LIMIT:    # never silently truncate: split the box
            out = {}
            for se in (-1, 1):
                for sn in (-1, 1):
                    la, ln = offset(lat, lng, se * half_m / 2, sn * half_m / 2)
                    for d in self.images_in_bbox(la, ln, half_m / 2):
                        out[d["id"]] = d
            return list(out.values())
        return data


def flatten(d):
    cg = (d.get("computed_geometry") or {}).get("coordinates")
    g = (d.get("geometry") or {}).get("coordinates")
    sc = d.get("sfm_cluster") or {}
    cp = d.get("camera_parameters") or []
    ts = d.get("captured_at")
    return {
        "image_id": d["id"], "is_pano": int(bool(d.get("is_pano"))),
        "camera_type": d.get("camera_type") or "",
        "captured_at": ts if ts is not None else "",
        "capture_date": time.strftime("%Y-%m-%d", time.gmtime(ts / 1000)) if ts else "",
        "make": (d.get("make") or "").strip(), "model": (d.get("model") or "").strip(),
        "sequence": d.get("sequence") or "",
        "lat": cg[1] if cg else "", "lng": cg[0] if cg else "",
        "raw_lat": g[1] if g else "", "raw_lng": g[0] if g else "",
        "has_computed_geometry": int(bool(cg)),
        "computed_compass_angle": d.get("computed_compass_angle", ""),
        "compass_angle": d.get("compass_angle", ""),
        "has_computed_compass_angle": int(d.get("computed_compass_angle") is not None),
        "computed_rotation": json.dumps(d["computed_rotation"]) if d.get("computed_rotation")
        else "",
        "has_computed_rotation": int(bool(d.get("computed_rotation"))),
        "computed_altitude": d.get("computed_altitude", ""),
        # "" = not queried (see SFM_CLUSTER_BATCH); 0/1 only for the harness ramps' images
        "has_sfm_cluster": int(bool(sc)) if "sfm_cluster" in d else "",
        "sfm_cluster_id": sc.get("id", ""),
        "merge_cc": d.get("merge_cc", ""), "atomic_scale": d.get("atomic_scale", ""),
        "camera_parameters": json.dumps(cp) if cp else "",
        "width": d.get("width", ""), "height": d.get("height", ""),
        "quality_score": d.get("quality_score", ""),
        "exif_orientation": d.get("exif_orientation", ""),
    }


IMAGE_COLS = list(flatten({"id": "0"}).keys())


def cmd_census(args):
    token = read_token(args.env)
    g = Graph(token)
    del token
    ramps = read_ramps()
    # harness ramps first, so a partial run still covers them
    ramps.sort(key=lambda r: (not r["harness_pairs"], int(r["ramp_uid"].split(":")[1])))
    if args.limit:
        ramps = ramps[:args.limit]
    raw_dir = args.raw_cache
    os.makedirs(raw_dir, exist_ok=True)
    images, links = {}, []
    t0 = time.time()
    for i, r in enumerate(ramps):
        lat, lng = float(r["lat"]), float(r["lng"])
        cache = os.path.join(raw_dir, r["ramp_uid"].replace(":", "_") + ".json")
        if os.path.exists(cache):
            data = json.load(open(cache, encoding="utf-8"))
        else:
            data = g.images_in_bbox(lat, lng, RADIUS_M + 2.0)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f)
        for d in data:
            row = flatten(d)
            if row["has_computed_geometry"]:
                e, n = enu(lat, lng, row["lat"], row["lng"])
            elif row["raw_lat"] != "":
                e, n = enu(lat, lng, row["raw_lat"], row["raw_lng"])
            else:
                continue
            dist = math.hypot(e, n)
            if dist > RADIUS_M:
                continue
            images[row["image_id"]] = row
            links.append({"ramp_uid": r["ramp_uid"], "image_id": row["image_id"],
                          "dist_m": round(dist, 2),
                          "position": "sfm" if row["has_computed_geometry"] else "raw"})
        if (i + 1) % 20 == 0 or i + 1 == len(ramps):
            print(f"{i + 1}/{len(ramps)} ramps, {len(images)} images, {g.calls} calls, "
                  f"{time.time() - t0:.0f} s", flush=True)
    # sfm_cluster, one image per call (id batches answer HTTP 500 too), for the flat images
    # of the harness ramps; answers are cached so a re-run makes no calls
    hr = {r["ramp_uid"] for r in ramps if r["harness_pairs"]}
    want = sorted({l["image_id"] for l in links if l["ramp_uid"] in hr
                   and not images[l["image_id"]]["is_pano"]})
    sc_cache = os.path.join(raw_dir, "sfm_cluster.json")
    sc_ans = json.load(open(sc_cache, encoding="utf-8")) if os.path.exists(sc_cache) else {}
    unanswered = 0
    for k, i in enumerate(want):
        if i not in sc_ans:
            one = g.get(f"https://graph.mapillary.com/{i}", {"fields": "sfm_cluster{id}"},
                        attempts=3)
            if one is not None:
                sc_ans[i] = (one.get("sfm_cluster") or {}).get("id", "")
            if (k + 1) % 100 == 0:
                print(f"  sfm_cluster {k + 1}/{len(want)}, {g.calls} calls", flush=True)
                with open(sc_cache, "w", encoding="utf-8") as f:
                    json.dump(sc_ans, f)
        if i not in sc_ans:
            unanswered += 1     # stays "" (unknown), counted in the summary
            continue
        images[i]["has_sfm_cluster"] = int(bool(sc_ans[i]))
        images[i]["sfm_cluster_id"] = sc_ans[i]
    with open(sc_cache, "w", encoding="utf-8") as f:
        json.dump(sc_ans, f)
    print(f"sfm_cluster queried for {len(want)} flat images of harness ramps "
          f"({unanswered} unanswered), {g.calls} calls", flush=True)
    elapsed = time.time() - t0
    imgs = sorted(images.values(), key=lambda x: x["image_id"])
    links.sort(key=lambda x: (int(x["ramp_uid"].split(":")[1]), x["dist_m"], x["image_id"]))
    write_csv(os.path.join(CENSUS, "images.csv"), imgs, IMAGE_COLS)
    write_csv(os.path.join(CENSUS, "ramp_images.csv"), links,
              ["ramp_uid", "image_id", "dist_m", "position"])
    summary = summarize(read_ramps(), images, links)
    summary["api"] = {"endpoint": GRAPH + " (bbox search)", "calls": g.calls,
                      "retries": g.retries, "elapsed_s": round(elapsed, 1),
                      "run_date": time.strftime("%Y-%m-%d"), "fields": FIELDS,
                      "radius_m": RADIUS_M,
                      "note": "calls = 0 when every ramp was read from the raw cache"}
    H.write_json(os.path.join(CENSUS, "summary.json"), summary)
    print(json.dumps(summary["overall"], indent=1))


def summarize(ramps, images, links):
    per = defaultdict(list)
    for l in links:
        per[l["ramp_uid"]].append(l)
    harness = {r["ramp_uid"] for r in ramps if r["harness_pairs"]}
    rows = []
    for r in ramps:
        uid = r["ramp_uid"]
        row = {"ramp_uid": uid, "harness": int(uid in harness)}
        for R in RADII:
            ls = [l for l in per[uid] if l["dist_m"] <= R]
            ims = [images[l["image_id"]] for l in ls]
            flat = [x for x in ims if not x["is_pano"]]
            pano = [x for x in ims if x["is_pano"]]
            tag = f"{int(R)}m"
            row[f"n_{tag}"] = len(ims)
            row[f"flat_{tag}"] = len(flat)
            row[f"pano_{tag}"] = len(pano)
            row[f"flat_seq_{tag}"] = len({x["sequence"] for x in flat})
            row[f"flat_sfm_{tag}"] = sum(1 for x in flat if x["has_computed_geometry"]
                                         and x["has_computed_compass_angle"]
                                         and x["has_computed_rotation"])
            if R == RADIUS_M:
                dates = sorted(x["capture_date"] for x in flat if x["capture_date"])
                row["flat_first_date"] = dates[0] if dates else ""
                row["flat_last_date"] = dates[-1] if dates else ""
                row["flat_n_years"] = len({d[:4] for d in dates})
                row["flat_n_clusters"] = len({x["sfm_cluster_id"] for x in flat
                                              if x["sfm_cluster_id"]})
                mm = Counter(f"{x['make']} {x['model']}".strip() or "(none)" for x in flat)
                row["flat_top_camera"] = mm.most_common(1)[0][0] if mm else ""
        rows.append(row)
    write_csv(os.path.join(CENSUS, "per_ramp.csv"), rows, list(rows[0]))

    def dist(vals):
        a = np.array(vals, float)
        return {"n": len(a), "zero": int((a == 0).sum()),
                "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max())}

    def block(sel):
        out = {}
        for R in RADII:
            tag = f"{int(R)}m"
            out[tag] = {k: dist([r[f"{k}_{tag}"] for r in sel])
                        for k in ("n", "flat", "pano", "flat_seq", "flat_sfm")}
        return out

    in_census = {l["image_id"] for l in links}
    flat = [images[i] for i in in_census if not images[i]["is_pano"]]
    pano = [images[i] for i in in_census if images[i]["is_pano"]]

    def share(xs, k):
        xs = [x for x in xs if x[k] != ""]
        return {"share": round(sum(int(x[k]) for x in xs) / max(1, len(xs)), 4),
                "of": len(xs)}

    cams = Counter(f"{x['make']} {x['model']}".strip() or "(none)" for x in flat)
    ctype = Counter(x["camera_type"] or "(none)" for x in flat)
    years = Counter(x["capture_date"][:4] for x in flat if x["capture_date"])
    return {
        "overall": {
            "ramps": len(rows), "harness_ramps": len(harness),
            "unique_images_30m": len(in_census), "flat": len(flat), "pano": len(pano),
            "flat_sequences": len({x["sequence"] for x in flat}),
            "pano_sequences": len({x["sequence"] for x in pano}),
            "flat_has_computed_geometry": share(flat, "has_computed_geometry"),
            "flat_has_computed_compass_angle": share(flat, "has_computed_compass_angle"),
            "flat_has_computed_rotation": share(flat, "has_computed_rotation"),
            "flat_has_sfm_cluster": share(flat, "has_sfm_cluster"),
            "flat_has_camera_parameters": round(sum(1 for x in flat if x["camera_parameters"])
                                                / max(1, len(flat)), 4),
            "pano_has_computed_geometry": share(pano, "has_computed_geometry"),
        },
        "all_ramps": block(rows),
        "harness_ramps": block([r for r in rows if r["harness"]]),
        "flat_camera_make_model_top15": cams.most_common(15),
        "flat_camera_type": dict(ctype),
        "flat_capture_year": dict(sorted(years.items())),
    }


# --------------------------------------------------------------------------- #
# select: per-corner image manifest for the reconstruction (no answers read)
# --------------------------------------------------------------------------- #

MANIFEST = os.path.join(OUT, "manifest.json")
AIM_HEIGHT_M = 2.6            # the harness projection height; aims views, centres corners
SELECT_RADIUS_M = 30.0
FOV_MARGIN_DEG = 15.0         # a flat image is kept if the corner is inside hfov/2 + this
DEFAULT_FOCAL_NORM = 0.85     # OpenSfM's default when an image has no camera_parameters
MAX_FLAT = 150                # cap per corner, nearest first (recorded per corner)
PANO_Z_PRIOR_M, FLAT_Z_PRIOR_M = 2.6, 1.5


def rodrigues(rvec):
    r = np.asarray(rvec, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def flat_hfov_deg(row):
    """Horizontal FOV of a flat Mapillary image from its OpenSfM focal (normalised by the
    longer side)."""
    try:
        cp = json.loads(row["camera_parameters"]) if row["camera_parameters"] else []
    except ValueError:
        cp = []
    f = float(cp[0]) if cp else DEFAULT_FOCAL_NORM
    w, h = float(row["width"] or 1), float(row["height"] or 1)
    return 2.0 * math.degrees(math.atan(0.5 * (w / max(w, h)) / f))


def ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def cmd_select(args):
    from crossview_arms._registry import ANSWER_KEYS
    pairs = [{k: v for k, v in p.items() if k not in ANSWER_KEYS}
             for p in H.read_frozen_pairs() if p["city"] == CITY]
    by_ramp = defaultdict(list)
    for p in pairs:
        by_ramp[p["ramp_uid"]].append(p)
    ramps = {r["ramp_uid"]: r for r in read_ramps()}
    images = {r["image_id"]: r for r in csv.DictReader(
        open(os.path.join(CENSUS, "images.csv"), encoding="utf-8", newline=""))}
    caps = defaultdict(list)
    for r in csv.DictReader(open(CAPTURES_CSV, encoding="utf-8", newline="")):
        if r["city"] == CITY and r["ramp_uid"] in by_ramp:
            caps[r["ramp_uid"]].append(r["pano_id"])
    want = {p["src_pano"] for p in pairs} | {p["oth_pano"] for p in pairs} | \
        {pid for v in caps.values() for pid in v}
    panos = read_labeler_panos(args.results, want)
    corners = []
    for uid in sorted(by_ramp, key=lambda u: int(u.split(":")[1])):
        ps = sorted(by_ramp[uid], key=lambda p: p["pair_id"])
        s0 = ps[0]
        sp = panos[s0["src_pano"]]
        # corner centre: the source click raycast onto flat ground at 2.6 m (the harness's
        # own aim), never the reference
        el = (0.5 - s0["src_y"]) * math.pi
        rng = AIM_HEIGHT_M / math.tan(-el)
        b = math.radians(sp["camera_heading"] + (s0["src_x"] - 0.5) * 360.0)
        lat0, lng0 = offset(sp["lat"], sp["lng"], rng * math.sin(b), rng * math.cos(b))
        imgs = []

        def pano_entry(pid, kind, view, cx, cy, pair_id=""):
            p = panos[pid]
            e, n = enu(lat0, lng0, p["lat"], p["lng"])
            sm = p.get("source_metadata") or {}
            return {"name": view, "kind": kind, "id": pid, "pair_id": pair_id,
                    "e": round(e, 3), "n": round(n, 3), "z_prior": PANO_Z_PRIOR_M,
                    "heading": p["camera_heading"], "rotvec": sm.get("computed_rotation"),
                    "cx": cx, "cy": cy, "date": p.get("capture_date") or "",
                    "sequence": p.get("sequence_id") or "", "dist_m": round(math.hypot(e, n), 2),
                    "make": p.get("camera_make") or "", "model": p.get("camera_model") or ""}

        imgs.append(pano_entry(s0["src_pano"], "pano_src", f"{s0['pair_id']}_src.jpg",
                               s0["src_x"], s0["src_y"]))
        for p in ps:
            imgs.append(pano_entry(p["oth_pano"], "pano_oth", f"{p['pair_id']}_oth.jpg",
                                   p["proj_x"], p["proj_y"], p["pair_id"]))
        used = {s0["src_pano"]} | {p["oth_pano"] for p in ps}
        for pid in sorted(set(caps[uid]) - used):
            pp = panos[pid]
            e, n = enu(lat0, lng0, pp["lat"], pp["lng"])
            # aim at the corner centre, flat ground at 2.6 m (as _mv3d on crossview-sfm-48)
            bearing = math.degrees(math.atan2(-e, -n))
            d = math.hypot(e, n)
            cx = (0.5 + ((bearing - pp["camera_heading"] + 180.0) % 360.0 - 180.0) / 360.0) % 1.0
            cy = 0.5 + math.degrees(math.atan2(AIM_HEIGHT_M, max(d, 0.5))) / 180.0
            imgs.append(pano_entry(pid, "pano_extra", f"{uid.replace(':', '_')}_{pid}.jpg",
                                   round(cx, 6), round(cy, 6)))
        # flat images within SELECT_RADIUS_M whose view contains the corner
        cand = []
        n_flat_radius = 0
        for iid, r in images.items():
            if r["is_pano"] == "1" or not r["has_computed_geometry"] == "1":
                continue
            e, n = enu(lat0, lng0, float(r["lat"]), float(r["lng"]))
            d = math.hypot(e, n)
            if d > SELECT_RADIUS_M:
                continue
            n_flat_radius += 1
            if r["computed_compass_angle"] == "":
                continue
            bearing = math.degrees(math.atan2(-e, -n)) % 360.0
            if d > 2.0 and ang_diff(bearing, float(r["computed_compass_angle"])) > \
                    flat_hfov_deg(r) / 2.0 + FOV_MARGIN_DEG:
                continue
            cp = json.loads(r["camera_parameters"]) if r["camera_parameters"] else []
            cand.append({"name": f"{iid}.jpg", "kind": "flat", "id": iid,
                         "e": round(e, 3), "n": round(n, 3), "z_prior": FLAT_Z_PRIOR_M,
                         "heading": float(r["computed_compass_angle"]),
                         "rotvec": json.loads(r["computed_rotation"])
                         if r["computed_rotation"] else None,
                         "camera_parameters": cp or [DEFAULT_FOCAL_NORM, 0.0, 0.0],
                         "camera_type": r["camera_type"], "width": int(r["width"] or 0),
                         "height": int(r["height"] or 0),
                         "date": r["capture_date"], "sequence": r["sequence"],
                         "dist_m": round(d, 2), "make": r["make"], "model": r["model"],
                         "exif_orientation": r["exif_orientation"]})
        cand.sort(key=lambda x: (x["dist_m"], x["id"]))
        flat = cand[:MAX_FLAT]
        imgs.extend(flat)
        corners.append({
            "ramp_uid": uid, "origin": [round(lat0, 9), round(lng0, 9)],
            "origin_is": "source click raycast at 2.6 m from the source pano",
            "pool_position": [float(ramps[uid]["lat"]), float(ramps[uid]["lng"])],
            "pairs": [{k: p[k] for k in ("pair_id", "src_pano", "src_x", "src_y", "oth_pano",
                                         "proj_x", "proj_y", "src_range_m", "oth_range_m",
                                         "baseline_m")} for p in ps],
            "n_flat_within_radius": n_flat_radius, "n_flat_facing": len(cand),
            "n_flat_kept": len(flat), "images": imgs})
        print(f"{uid}: {len(ps)} pairs, panos {sum(i['kind'] != 'flat' for i in imgs)}, flat "
              f"{n_flat_radius} within {SELECT_RADIUS_M:g} m, {len(cand)} facing, "
              f"{len(flat)} kept", flush=True)
    H.write_json(MANIFEST, {"pairs_sha256": H.PAIRS_SHA256, "radius_m": SELECT_RADIUS_M,
                            "fov_margin_deg": FOV_MARGIN_DEG, "max_flat": MAX_FLAT,
                            "z_prior_m": {"pano": PANO_Z_PRIOR_M, "flat": FLAT_Z_PRIOR_M},
                            "frame": "per corner ENU metres about 'origin' (labeler "
                                     "LocalFrame linearisation); z up",
                            "corners": corners})
    print(f"{len(corners)} corners -> {MANIFEST}")


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    if m["pairs_sha256"] != H.PAIRS_SHA256:
        raise SystemExit("manifest built on a different pair list")
    return m


# --------------------------------------------------------------------------- #
# fetch: flat thumbnails (desktop; the token never leaves this machine)
# --------------------------------------------------------------------------- #

FETCHED_CSV = os.path.join(OUT, "fetched_images.csv")
THUMB_FIELD = "thumb_2048_url"
URL_BATCH = 10


def cmd_fetch(args):
    import hashlib
    import requests
    from PIL import Image
    m = load_manifest()
    corners = [c for c in m["corners"] if not args.corners or c["ramp_uid"] in args.corners]
    ids = sorted({i["id"] for c in corners for i in c["images"] if i["kind"] == "flat"})
    os.makedirs(args.out, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(args.out, f"{i}.jpg"))]
    print(f"{len(ids)} flat images for {len(corners)} corners; {len(todo)} to fetch", flush=True)
    token = read_token(args.env)
    g = Graph(token)
    del token
    plain = requests.Session()           # the image CDN needs no token: never send it there
    t0 = time.time()
    n_bytes, failed = 0, []
    for k in range(0, len(todo), URL_BATCH):
        batch = todo[k:k + URL_BATCH]
        js = g.get("https://graph.mapillary.com/", {"ids": ",".join(batch),
                                                    "fields": THUMB_FIELD}, attempts=3) or {}
        for i in batch:
            url = (js.get(i) or {}).get(THUMB_FIELD)
            if not url:
                one = g.get(f"https://graph.mapillary.com/{i}", {"fields": THUMB_FIELD},
                            attempts=3) or {}
                url = one.get(THUMB_FIELD)
            if not url:
                failed.append(i)
                continue
            r = plain.get(url, timeout=60)
            if r.status_code != 200:
                failed.append(i)
                continue
            with open(os.path.join(args.out, f"{i}.jpg"), "wb") as f:
                f.write(r.content)
            n_bytes += len(r.content)
        print(f"  {min(k + URL_BATCH, len(todo))}/{len(todo)}, {n_bytes / 1e6:.0f} MB, "
              f"{g.calls} API calls, {time.time() - t0:.0f} s", flush=True)
    rows = []
    prev = {}
    if os.path.exists(FETCHED_CSV):
        prev = {r["image_id"]: r for r in csv.DictReader(open(FETCHED_CSV, encoding="utf-8"))}
    for i in ids:
        p = os.path.join(args.out, f"{i}.jpg")
        if not os.path.exists(p):
            continue
        data = open(p, "rb").read()
        w, h = Image.open(p).size
        rows.append({"image_id": i, "bytes": len(data), "width": w, "height": h,
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "fetched": prev.get(i, {}).get("fetched") or time.strftime("%Y-%m-%d")})
    write_csv(FETCHED_CSV, rows, ["image_id", "bytes", "width", "height", "sha256", "fetched"])
    log = {"step": "fetch", "date": time.strftime("%Y-%m-%d"), "corners": len(corners),
           "requested": len(todo), "failed": len(failed), "api_calls": g.calls,
           "api_retries": g.retries, "cdn_bytes": n_bytes,
           "elapsed_s": round(time.time() - t0, 1), "thumb": THUMB_FIELD}
    with open(os.path.join(OUT, "api_log.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(log, sort_keys=True) + "\n")
    print(json.dumps(log))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ramps")
    a.add_argument("--results", required=True, help="the labeler's richmond results.jsonl")
    a.set_defaults(fn=cmd_ramps)
    a = sub.add_parser("census")
    a.add_argument("--env", required=True, help=".env file holding MAPILLARY_ACCESS_TOKEN")
    a.add_argument("--raw-cache", default=os.path.join(OUT, "raw_api_cache"),
                   help="raw Graph responses per ramp (not committed)")
    a.add_argument("--limit", type=int, default=0)
    a.set_defaults(fn=cmd_census)
    a = sub.add_parser("select")
    a.add_argument("--results", required=True, help="the labeler's richmond results.jsonl")
    a.set_defaults(fn=cmd_select)
    a = sub.add_parser("fetch")
    a.add_argument("--env", required=True, help=".env file holding MAPILLARY_ACCESS_TOKEN")
    a.add_argument("--out", required=True, help="image directory (NOT in the repo)")
    a.add_argument("--corners", nargs="*", default=[], help="ramp uids (default: all)")
    a.set_defaults(fn=cmd_fetch)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
