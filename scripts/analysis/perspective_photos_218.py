"""RampNet on perspective (flat) photos -- the controlled Richmond test (issue #218).

Question: can the released RampNet checkpoint, trained on 2048x4096 equirect panoramas,
find curb ramps in ordinary perspective photos? Richmond's flat Mapillary images change
only the input (same city, same ramps, camera on a vehicle), so they are the controlled
test; the Seoul pedestrian photos (``seoul_photos_218.py``) are the stretch test.

Arms (every arm runs the released checkpoint unchanged, fp32, peaks at the 0.10 floor
with the benchmark's ``min_distance=10`` and ``exclude_border=False``):

    canvas_level  the photo reprojected into a 2048x4096 equirect canvas at its true FOV
                  (Mapillary's SfM focal + k1/k2), heading on the centre column, camera
                  ASSUMED LEVEL. A ramp subtends the angle it would in a panorama.
    canvas_sfm    the same canvas, but the photo placed at its SfM pitch and roll
                  (Mapillary ``computed_rotation``) instead of level.
    stretch       the photo resized straight to 2048x4096 (the strawman; this is
                  ``threshold_sweep.PRE`` itself).
    canvas_x2     arm (c): the level canvas at twice the angular scale (4096x8192 input,
                  8192 px per 360 deg), i.e. close to the 2048-px thumbnails' own scale.

Outside the photo the canvas is the ImageNet mean colour (zero after normalisation);
peaks there are dropped and counted. Every detection is mapped back to the photo pixel
it came from, so every arm is scored with the same geometry.

Subcommands::

    # desktop, CPU: the image list (committed census, no network)
    python scripts/analysis/perspective_photos_218.py select
    # desktop: fetch the 2048-px thumbnails (needs a Mapillary token; images not committed)
    python scripts/analysis/perspective_photos_218.py fetch --env ../sidewalk-auto-labeler/.env \\
        --out IMGDIR
    # GPU (makelab2): detections per arm -> analysis_out/perspective_photos_218/dets_<arm>.jsonl
    python scripts/analysis/perspective_photos_218.py infer --images IMGDIR \\
        --arms canvas_level,canvas_sfm,stretch
    # CPU: tables + paired bootstrap -> results.json / results.md
    python scripts/analysis/perspective_photos_218.py score
    # CPU: the precision gallery for a human rater (needs IMGDIR)
    python scripts/analysis/perspective_photos_218.py gallery --images IMGDIR

See docs/perspective_photos_218.md for the method, the numbers and the caveats.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import perspective as P  # noqa: E402

OUT = os.path.join(REPO, "analysis_out", "perspective_photos_218")
CENSUS = os.path.join(REPO, "analysis_out", "flat_mapillary_3d", "census")
CAPTURES = os.path.join(REPO, "analysis_out", "multiview_48", "captures_R25.csv")
NEIGHBOURHOOD = os.path.join(REPO, "benchmark", "richmond_neighbourhood", "records.jsonl")
IMAGES_CSV = os.path.join(OUT, "images.csv")
FETCHED_CSV = os.path.join(OUT, "fetched.csv")

FLOOR = 0.10
MIN_DISTANCE = 10
THRESHOLDS = (0.30, 0.55)
PRIMARY_THR = 0.30

ARMS = {
    "canvas_level": {"kind": "canvas", "pose": "level", "scale": 1},
    "canvas_sfm": {"kind": "canvas", "pose": "sfm", "scale": 1},
    "stretch": {"kind": "stretch", "pose": None, "scale": 1},
    "canvas_x2": {"kind": "canvas", "pose": "level", "scale": 2},
}

# ---- in-view / hit geometry (docs/perspective_photos_218.md section 3) ----
RANGE_MIN, RANGE_MAX = 3.0, 18.0       # a pool ramp counts as "in view" in this band
RANGE_BINS = ((3.0, 6.0), (6.0, 12.0), (12.0, 18.0))
EDGE_MARGIN_FRAC = 0.03                # ... and projects >= 3% of the width inside the frame
VIEW_H = 1.5                           # height used for the in-view vertical check (m)
CANDIDATE_MAX = 30.0                   # ramps a detection may claim
NEG_RANGE = 40.0                       # "pool-negative": no pool ramp this close ...
NEG_MARGIN_DEG = 10.0                  # ... within the FOV widened by this much each side
LATERAL_M = 5.0                        # eval_sites' match radius
H_MIN, H_MAX = 0.5, 4.0                # heights the bearing test accepts
WORLD_HEIGHTS = (1.5, 2.6)             # flat-ground raycast sensitivity (2.6 = labeler's)
N_REPS = 2000
SEED = 218
ND = 4


# --------------------------------------------------------------------------- #
# small io
# --------------------------------------------------------------------------- #
def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def rnd(v, nd=ND):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), nd)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
IMAGE_COLS = ["image_id", "sequence", "capture_date", "make", "model", "width", "height",
              "lat", "lng", "computed_compass_angle", "computed_rotation",
              "camera_parameters", "nearest_ramp", "nearest_ramp_m"]


def cmd_select(args):
    """Every perspective flat image of the #216 census, upright only.

    The census holds every Mapillary image within 30 m of a Richmond pool ramp. Fisheye
    frames are dropped (69; a different camera model), and so would be any image with an
    EXIF orientation other than 1 (the thumbnail's pixel frame vs the SfM pose was not
    checked for those; on 2026-09-30 all 38 such images were fisheye, so this drops
    nothing further)."""
    ims = read_csv(os.path.join(CENSUS, "images.csv"))
    ramps = read_csv(os.path.join(CENSUS, "ramps.csv"))
    rl = np.array([[float(r["lat"]), float(r["lng"])] for r in ramps])
    rows, n_fish, n_rot = [], 0, 0
    for im in ims:
        if im["is_pano"] != "0":
            continue
        if im["camera_type"] != "perspective":
            n_fish += 1
            continue
        if im["exif_orientation"] != "1":
            n_rot += 1
            continue
        lat, lng = float(im["lat"]), float(im["lng"])
        e, n = P.enu_offset(lat, lng, rl[:, 0], rl[:, 1])
        d = np.hypot(e, n)
        k = int(np.argmin(d))
        rows.append({**{c: im.get(c, "") for c in IMAGE_COLS},
                     "nearest_ramp": ramps[k]["ramp_uid"], "nearest_ramp_m": f"{d[k]:.2f}"})
    rows.sort(key=lambda r: r["image_id"])
    write_csv(IMAGES_CSV, rows, IMAGE_COLS)
    print(f"{len(rows)} images (dropped {n_fish} non-perspective, {n_rot} EXIF-rotated) "
          f"-> {IMAGES_CSV}")


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
THUMB_FIELD = "thumb_2048_url"


def cmd_fetch(args):
    """Mapillary 2048-px thumbnails of ``images.csv`` into ``--out``; sha256 -> fetched.csv.

    The token is read from the .env at run time and only ever sent to the Graph API in a
    header (``flat_mapillary_48.Graph``); the image CDN gets no token."""
    import requests
    from PIL import Image
    import flat_mapillary_48 as F
    ids = [r["image_id"] for r in read_csv(IMAGES_CSV)]
    os.makedirs(args.out, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(args.out, f"{i}.jpg"))]
    print(f"{len(ids)} images; {len(todo)} to fetch", flush=True)
    t0 = time.time()
    failed, n_bytes, calls = [], 0, 0
    if todo:
        token = F.read_token(args.env)
        g = F.Graph(token)
        del token
        plain = requests.Session()
        for k in range(0, len(todo), 10):
            batch = todo[k:k + 10]
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
            if (k // 10) % 10 == 0:
                print(f"  {min(k + 10, len(todo))}/{len(todo)} {n_bytes / 1e6:.0f} MB "
                      f"{time.time() - t0:.0f} s", flush=True)
        calls = g.calls
    prev = {r["image_id"]: r for r in read_csv(FETCHED_CSV)} if os.path.exists(FETCHED_CSV) else {}
    rows = []
    for i in ids:
        p = os.path.join(args.out, f"{i}.jpg")
        if not os.path.exists(p):
            continue
        w, h = Image.open(p).size
        rows.append({"image_id": i, "bytes": os.path.getsize(p), "width": w, "height": h,
                     "sha256": sha256_file(p),
                     "fetched": prev.get(i, {}).get("fetched") or time.strftime("%Y-%m-%d")})
    write_csv(FETCHED_CSV, rows, ["image_id", "bytes", "width", "height", "sha256", "fetched"])
    log = {"step": "fetch", "date": time.strftime("%Y-%m-%d"), "requested": len(todo),
           "failed": len(failed), "failed_ids": failed, "api_calls": calls,
           "cdn_bytes": n_bytes, "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "fetch_log.jsonl"), "a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(log, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in log.items() if k != "failed_ids"}))


# --------------------------------------------------------------------------- #
# inputs (pure; unit-tested)
# --------------------------------------------------------------------------- #
def camera_of(row, width, height):
    """The Mapillary camera on an image of ``width`` x ``height`` (the thumbnail)."""
    f, k1, k2 = json.loads(row["camera_parameters"])
    return P.Camera(width, height, f, k1, k2)


def pose_of(row):
    """World-to-camera rotation from ``computed_rotation``."""
    return P.world_to_cam_from_rotvec(json.loads(row["computed_rotation"]))


def arm_M(arm, R_wc):
    """Level-to-camera matrix the canvas for ``arm`` is built with."""
    spec = ARMS[arm]
    if spec["kind"] != "canvas":
        return None
    return P.cam_from_level(R_wc if spec["pose"] == "sfm" else None)


def canvas_size(arm):
    s = ARMS[arm]["scale"]
    return P.CANVAS_H * s, P.CANVAS_W * s


def build_canvas_tensor(img, cam, M, H, W):
    """PIL photo -> normalised (3, H, W) float tensor of the canvas, plus the (u, v)
    sampling maps (in the ORIGINAL photo's pixels). Outside the photo the tensor is 0,
    i.e. the ImageNet mean colour."""
    import torch
    import torch.nn.functional as TF
    from PIL import Image
    from torchvision import transforms
    s = P.prescale_factor(cam, W)
    if s < 1.0:
        w2, h2 = max(1, round(cam.width * s)), max(1, round(cam.height * s))
        small = img.resize((w2, h2), Image.BILINEAR)
    else:
        small = img
    cam_s = cam.resized(*small.size)
    u, v = P.canvas_sample_maps(cam_s, M, H, W)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(small)
    gx = (u + 0.5) / cam_s.width * 2 - 1
    gy = (v + 0.5) / cam_s.height * 2 - 1
    gx = np.where(np.isfinite(gx), gx, -3.0).astype(np.float32)
    gy = np.where(np.isfinite(gy), gy, -3.0).astype(np.float32)
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1))[None]
    out = TF.grid_sample(t[None], grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0]
    return out, np.isfinite(u)


def canvas_det_to_photo(x, y, cam, M):
    """Canvas detection (normalised) -> pixel (u, v) in the photo ``cam`` describes."""
    ray = P.canvas_norm_to_cam_ray(x, y, M)
    return P.project_cam(cam, ray)


def stretch_det_to_photo(x, y, cam):
    """Detection on the 2048x4096 stretch (normalised) -> photo pixel (u, v)."""
    return (x * P.CANVAS_W + 0.5) * cam.width / P.CANVAS_W - 0.5, \
        (y * P.CANVAS_H + 0.5) * cam.height / P.CANVAS_H - 0.5


def peaks(h, floor=FLOOR, md=MIN_DISTANCE):
    """Same extraction as threshold_sweep.peaks_to_dets (exclude_border=False)."""
    from skimage.feature import peak_local_max
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=md, threshold_abs=floor,
                        exclude_border=False)
    Hh, Wh = h.shape
    return [(float(c / Wh), float(r / Hh), float(h[r][c])) for r, c in pk]


def run_arm(model, device, img, cam, arm, M):
    """One forward pass for ``arm`` -> (dets, n_fill_peaks, max_in_photo).

    ``dets`` rows are dicts with the model-input position (x, y), score, and the photo
    pixel (u, v) in ``cam``'s frame."""
    import torch
    import threshold_sweep as ts
    spec = ARMS[arm]
    if spec["kind"] == "stretch":
        t = ts.PRE(img)
        inside = None
    else:
        H, W = canvas_size(arm)
        t, inside = build_canvas_tensor(img, cam, M, H, W)
    with torch.no_grad():
        hm = model(t[None].to(device)).squeeze().float().cpu().numpy()
    raw = peaks(hm)
    dets, n_fill = [], 0
    for x, y, s in raw:
        if spec["kind"] == "stretch":
            u, v = stretch_det_to_photo(x, y, cam)
        else:
            Hc, Wc = inside.shape
            r = min(Hc - 1, int(y * Hc)), min(Wc - 1, int(x * Wc))
            if not inside[r]:
                n_fill += 1
                continue
            u, v = canvas_det_to_photo(x, y, cam, M)
            u, v = float(u), float(v)
        dets.append({"x": rnd(x, 6), "y": rnd(y, 6), "score": rnd(s, 6),
                     "u": rnd(u, 2), "v": rnd(v, 2)})
    max_in = max((d["score"] for d in dets), default=0.0)
    return dets, n_fill, max_in


# --------------------------------------------------------------------------- #
# infer
# --------------------------------------------------------------------------- #
def dets_path(arm, out=OUT):
    return os.path.join(out, f"dets_{arm}.jsonl")


def usage_row(label, n, elapsed_s, started, host, gpus, what, issue=218,
              script="scripts/analysis/perspective_photos_218.py", bundle=None):
    return {"provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
            "hardware": {"host": host, "gpus": gpus}, "status": "ok", "est_cost_usd": 0.0,
            "pricing": None, "issue": issue, "ts": started,
            "bundle": bundle or "analysis_out/perspective_photos_218/images.csv",
            "label": label, "run_id": f"{label}:{host}:{started}", "panos_scored": n,
            "elapsed_s": round(elapsed_s, 3), "s_per_pano": round(elapsed_s / n, 4) if n else None,
            "gpu_hours": round(elapsed_s / 3600.0, 4), "what": what, "script": script}


def load_model(device):
    import threshold_sweep as ts
    return ts.load_model().to(device)


def cmd_infer(args):
    import torch
    from PIL import Image
    from rampnet import ledger
    arms = args.arms.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv(IMAGES_CSV)
    fetched = {r["image_id"]: r for r in read_csv(FETCHED_CSV)}
    if args.limit:
        rows = rows[:args.limit]
    out = args.out or OUT
    model = load_model(device)
    host = socket.gethostname().split(".")[0]
    gpus = [torch.cuda.get_device_name(0)] if device.type == "cuda" else []
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_arm = {a: 0.0 for a in arms}
    recs = {a: [] for a in arms}
    n_bad = 0
    t_all = time.time()
    for k, r in enumerate(rows, 1):
        iid = r["image_id"]
        p = os.path.join(args.images, f"{iid}.jpg")
        if iid not in fetched or not os.path.exists(p):
            continue
        if args.verify_sha and sha256_file(p) != fetched[iid]["sha256"]:
            n_bad += 1
            print(f"  sha256 mismatch {iid}: skipped", flush=True)
            continue
        img = Image.open(p).convert("RGB")
        cam = camera_of(r, *img.size)
        R_wc = pose_of(r)
        for a in arms:
            t0 = time.time()
            dets, n_fill, mx = run_arm(model, device, img, cam, a, arm_M(a, R_wc))
            t_arm[a] += time.time() - t0
            recs[a].append({"image_id": iid, "width": img.size[0], "height": img.size[1],
                            "max_score": rnd(mx, 6), "n_fill_peaks": n_fill, "dets": dets})
        if k % 50 == 0:
            print(f"  {k}/{len(rows)} {time.time() - t_all:.0f} s", flush=True)
    for a in arms:
        path = dets_path(a, out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            for rec in recs[a]:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        meta = {"arm": a, "spec": ARMS[a], "n_images": len(recs[a]), "floor": FLOOR,
                "min_distance": MIN_DISTANCE, "host": host, "gpus": gpus,
                "device": device.type, "fp16": False, "started": started,
                "elapsed_s": round(t_arm[a], 3), "sha256_mismatch": n_bad,
                "torch": torch.__version__}
        write_json(path[:-len(".jsonl")] + ".meta.json", meta)
        print(f"{a}: {len(recs[a])} images, {t_arm[a]:.0f} s -> {path}")
    if args.usage_log != "none" and not args.limit:
        ul = args.usage_log or os.path.join(ledger.canonical_repo_root(REPO) or REPO,
                                            "analysis_out", "usage_log.jsonl")
        rows_u = [usage_row(f"perspective-218:richmond:{a}", len(recs[a]), t_arm[a], started,
                            host, gpus,
                            f"perspective_photos_218.py infer, arm {a}: per-image seconds "
                            f"(canvas build + forward + peaks), fp32, run wall "
                            f"{time.time() - t_all:.0f} s for all arms")
                  for a in arms]
        ledger.append_rows(ul, rows_u)
        print(f"usage rows -> {ul}")


# --------------------------------------------------------------------------- #
# score: geometry (pure; unit-tested)
# --------------------------------------------------------------------------- #
def ramp_table():
    """[(uid, lat, lng)] of the 253 Richmond pool ramps (census/ramps.csv)."""
    return [(r["ramp_uid"], float(r["lat"]), float(r["lng"]))
            for r in read_csv(os.path.join(CENSUS, "ramps.csv"))]


def image_geometry(row, width, height, ramps):
    """Per image: the pool ramps near it with range, bearing, and whether each is in view.

    In view means: horizontal range in [RANGE_MIN, RANGE_MAX]; a flat-ground point at the
    ramp, seen from VIEW_H m with the SfM pose, projects inside the frame at least
    EDGE_MARGIN_FRAC of the width from the left/right edges and above the bottom edge.
    ``pool_negative`` is True when no pool ramp within NEG_RANGE has a bearing inside the
    horizontal FOV widened by NEG_MARGIN_DEG on each side."""
    cam = camera_of(row, width, height)
    R_wc = pose_of(row)
    heading = float(row["computed_compass_angle"])
    lat, lng = float(row["lat"]), float(row["lng"])
    uids = [u for u, _, _ in ramps]
    e, n = P.enu_offset(lat, lng, np.array([a for _, a, _ in ramps]),
                        np.array([b for _, _, b in ramps]))
    d = np.hypot(e, n)
    bearing = np.degrees(np.arctan2(e, n)) % 360
    near = []
    half = cam.hfov_deg() / 2
    neg = True
    for i in np.nonzero(d <= max(NEG_RANGE, CANDIDATE_MAX))[0]:
        dbear = float(P.wrap_deg(bearing[i] - heading))
        if d[i] <= NEG_RANGE and abs(dbear) <= half + NEG_MARGIN_DEG:
            neg = False
        in_view = False
        if RANGE_MIN <= d[i] <= RANGE_MAX:
            pc = R_wc @ np.array([e[i], n[i], -VIEW_H])
            if pc[2] > 0:
                u, v = P.project_cam(cam, pc)
                m = EDGE_MARGIN_FRAC * cam.width
                in_view = bool(m <= u <= cam.width - 1 - m and v <= cam.height - 1)
        if d[i] <= CANDIDATE_MAX:
            near.append({"uid": uids[i], "range": float(d[i]), "bearing": float(bearing[i]),
                         "dbear": dbear, "e": float(e[i]), "n": float(n[i]),
                         "in_view": in_view})
    return {"cam": cam, "R_wc": R_wc, "near": near, "pool_negative": neg,
            "positive": any(r["in_view"] for r in near)}


def det_world(dets, cam, R_wc):
    """Detections' photo pixels -> (bearing, depression) arrays with the SfM pose."""
    if not dets:
        return np.zeros(0), np.zeros(0), np.zeros((0, 3))
    ray = P.unproject_cam(cam, [d["u"] for d in dets], [d["v"] for d in dets])
    w = ray @ R_wc              # R_wc.T @ ray for each row
    b, dep = P.ray_bearing_depression(w)
    return b, dep, w


def claim_bearing(dets, b, dep, near, thr):
    """Greedy one-to-one claims in descending score: each detection >= thr claims the
    unclaimed candidate ramp (range <= CANDIDATE_MAX) with the smallest bearing error that
    passes ``bearing_hit``. Returns {ramp uid: det index}."""
    order = sorted((i for i, d in enumerate(dets) if d["score"] >= thr),
                   key=lambda i: -dets[i]["score"])
    claimed = {}
    for i in order:
        best, best_err = None, None
        for r in near:
            if r["uid"] in claimed:
                continue
            if not P.bearing_hit(b[i], dep[i], r["bearing"], r["range"], LATERAL_M,
                                 H_MIN, H_MAX):
                continue
            err = abs(float(P.wrap_deg(b[i] - r["bearing"])))
            if best is None or err < best_err:
                best, best_err = r["uid"], err
        if best is not None:
            claimed[best] = i
    return claimed


def claim_world(dets, w, near, thr, h):
    """As ``claim_bearing`` but with a flat-ground raycast at camera height ``h`` and the
    5 m radius; nearest unclaimed ramp wins."""
    order = sorted((i for i, d in enumerate(dets) if d["score"] >= thr),
                   key=lambda i: -dets[i]["score"])
    claimed = {}
    if not order:
        return claimed
    ge, gn = P.raycast_ground(w, h)
    for i in order:
        if not np.isfinite(ge[i]):
            continue
        best, best_d = None, None
        for r in near:
            if r["uid"] in claimed:
                continue
            dd = math.hypot(ge[i] - r["e"], gn[i] - r["n"])
            if dd <= LATERAL_M and (best is None or dd < best_d):
                best, best_d = r["uid"], dd
        if best is not None:
            claimed[best] = i
    return claimed


def range_bin(r):
    for lo, hi in RANGE_BINS:
        if lo <= r < hi or (hi == RANGE_BINS[-1][1] and r == hi):
            return f"{lo:g}-{hi:g}"
    return None


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def load_dets(arm, out=OUT):
    recs = {}
    with open(dets_path(arm, out), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["image_id"]] = r
    return recs


def per_image_table(arm_recs, rows, ramps, thresholds=THRESHOLDS):
    """Per (arm, image): positive / pool_negative / fired, and per (arm, image, in-view
    ramp): hit under each test. Returns (images, pairs) lists of dicts."""
    images, pairs = [], []
    by_id = {r["image_id"]: r for r in rows}
    ids = sorted(set.intersection(*[set(v) for v in arm_recs.values()]))
    for iid in ids:
        row = by_id[iid]
        any_rec = next(iter(arm_recs.values()))[iid]
        g = image_geometry(row, any_rec["width"], any_rec["height"], ramps)
        cluster = row["nearest_ramp"]
        for arm, recs in arm_recs.items():
            rec = recs[iid]
            dets = rec["dets"]
            b, dep, w = det_world(dets, g["cam"], g["R_wc"])
            im = {"arm": arm, "image_id": iid, "cluster": cluster, "positive": g["positive"],
                  "pool_negative": g["pool_negative"], "max_score": rec["max_score"],
                  "n_in_view": sum(r["in_view"] for r in g["near"])}
            for thr in thresholds:
                cl = claim_bearing(dets, b, dep, g["near"], thr)
                in_view_uids = {r["uid"] for r in g["near"] if r["in_view"]}
                im[f"fired@{thr}"] = any(d["score"] >= thr for d in dets)
                im[f"n_dets@{thr}"] = sum(d["score"] >= thr for d in dets)
                im[f"n_matched@{thr}"] = len(cl)
                im[f"loc_hit@{thr}"] = bool(in_view_uids & set(cl))
                wcl = {h: claim_world(dets, w, g["near"], thr, h) for h in WORLD_HEIGHTS}
                for r in g["near"]:
                    if not r["in_view"]:
                        continue
                    pairs.append({"arm": arm, "image_id": iid, "ramp": r["uid"],
                                  "range": r["range"], "bin": range_bin(r["range"]),
                                  "thr": thr, "hit_bearing": r["uid"] in cl,
                                  **{f"hit_world_{h}": r["uid"] in wcl[h]
                                     for h in WORLD_HEIGHTS}})
            images.append(im)
    return images, pairs


def boot_indices(clusters, n_reps=N_REPS, seed=SEED):
    """Cluster bootstrap: list of arrays of cluster labels drawn with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(clusters)))
    return uniq, [rng.integers(0, len(uniq), len(uniq)) for _ in range(n_reps)]


def cluster_rate(values, clusters, uniq, draws):
    """Point estimate and 95% percentile CI of mean(values) under a cluster bootstrap."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if len(values) == 0:
        return None, None, None, 0
    idx = {c: np.nonzero(clusters == c)[0] for c in uniq}
    s = np.array([values[idx[c]].sum() if len(idx[c]) else 0.0 for c in uniq])
    n = np.array([len(idx[c]) for c in uniq], dtype=float)
    est = s.sum() / n.sum()
    reps = []
    for dr in draws:
        nn = n[dr].sum()
        if nn > 0:
            reps.append(s[dr].sum() / nn)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, lo, hi, len(values)


def paired_diff(va, vb, clusters, uniq, draws):
    """mean(va) - mean(vb) over the same units, cluster bootstrap CI."""
    return cluster_rate(np.asarray(va, float) - np.asarray(vb, float), clusters, uniq, draws)


def fmt_ci(t):
    est, lo, hi, n = t
    if est is None:
        return "n/a"
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def pano_reference(ramps_in_view):
    """RampNet on the 360 panos: the world-test hit rate of every non-source Richmond
    capture of a ramp in the flat set, from multiview_48/captures_R25.csv (world_conf is
    the best claiming detection's confidence, 2.6 m raycast within 5 m)."""
    out = []
    for r in read_csv(CAPTURES):
        if r["city"] != "richmond" or r["is_source"] != "0":
            continue
        d = float(r["dist_m"])
        if not (RANGE_MIN <= d <= RANGE_MAX):
            continue
        wc = float(r["world_conf"]) if r["world_conf"] not in ("", "nan") else 0.0
        out.append({"ramp": r["ramp_uid"], "pano_id": r["pano_id"], "range": d,
                    "bin": range_bin(d), "world_conf": wc,
                    "same_ramps": r["ramp_uid"] in ramps_in_view})
    return out


def pano_bearing_check(thr=0.55):
    """Test comparability: the flat photos' bearing test applied to the same pano captures
    whose world test is in ``pano_reference``. Only detections >= 0.55 are stored in the
    neighbourhood records, so this runs at 0.55 only. The ramp's bearing in the pano is
    its projected column (``x_proj``); the detection's depression is read off a level
    equirect, ``(y - 0.5) * 180``. Returns per-capture rows with both tests."""
    dets = {}
    with open(NEIGHBOURHOOD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                dets[r["pano"]["panorama_id"]] = [
                    d for d in r["detections"] if d["confidence"] >= thr]
    rows = []
    by_pano = {}
    for r in read_csv(CAPTURES):
        if r["city"] != "richmond" or r["is_source"] != "0":
            continue
        d = float(r["dist_m"])
        if not (RANGE_MIN <= d <= RANGE_MAX) or r["pano_id"] not in dets:
            continue
        by_pano.setdefault(r["pano_id"], []).append(r)
    for pid, caps in by_pano.items():
        ds = sorted(dets[pid], key=lambda x: -x["confidence"])
        claimed = set()
        hits = {}
        for dd in ds:
            best, be = None, None
            for c in caps:
                if c["ramp_uid"] in claimed:
                    continue
                db = float(P.wrap_deg((dd["x_normalized"] - float(c["x_proj"])) * 360.0))
                dep = (dd["y_normalized"] - 0.5) * 180.0
                if P.bearing_hit(db, dep, 0.0, float(c["dist_m"]), LATERAL_M, H_MIN, H_MAX):
                    if best is None or abs(db) < be:
                        best, be = c["ramp_uid"], abs(db)
            if best:
                claimed.add(best)
        for c in caps:
            wc = float(c["world_conf"]) if c["world_conf"] not in ("", "nan") else 0.0
            rows.append({"ramp": c["ramp_uid"], "bearing_hit": c["ramp_uid"] in claimed,
                         "world_hit": wc >= thr})
    return rows


def cmd_score(args):
    arms = args.arms.split(",")
    rows = read_csv(IMAGES_CSV)
    ramps = ramp_table()
    arm_recs = {a: load_dets(a, args.dets_dir or OUT) for a in arms}
    images, pairs = per_image_table(arm_recs, rows, ramps)
    res_dir = args.results_dir or OUT
    res = {"config": {"arms": arms, "thresholds": list(THRESHOLDS), "floor": FLOOR,
                      "range": [RANGE_MIN, RANGE_MAX], "bins": [list(b) for b in RANGE_BINS],
                      "edge_margin_frac": EDGE_MARGIN_FRAC, "view_h": VIEW_H,
                      "neg_range": NEG_RANGE, "neg_margin_deg": NEG_MARGIN_DEG,
                      "lateral_m": LATERAL_M, "h_accept": [H_MIN, H_MAX],
                      "world_heights": list(WORLD_HEIGHTS), "n_reps": N_REPS, "seed": SEED},
           "counts": {}, "presence": {}, "points": {}, "paired": {}, "pano_reference": {},
           "fill_peaks": {}}
    a0 = arms[0]
    im0 = [i for i in images if i["arm"] == a0]
    res["counts"] = {"images": len(im0),
                     "positive": sum(i["positive"] for i in im0),
                     "pool_negative": sum(i["pool_negative"] for i in im0),
                     "neither": sum(not i["positive"] and not i["pool_negative"] for i in im0),
                     "in_view_pairs": sum(1 for p in pairs if p["arm"] == a0
                                          and p["thr"] == THRESHOLDS[0]),
                     "in_view_ramps": len({p["ramp"] for p in pairs if p["arm"] == a0}),
                     "clusters_positive": len({i["cluster"] for i in im0 if i["positive"]})}
    for a in arms:
        recs = arm_recs[a].values()
        res["fill_peaks"][a] = {"images_with_fill_peaks": sum(r["n_fill_peaks"] > 0 for r in recs),
                                "fill_peaks": sum(r["n_fill_peaks"] for r in recs)}
    # --- image-level
    pos = [i for i in im0 if i["positive"]]
    neg = [i for i in im0 if i["pool_negative"]]
    upos, dpos = boot_indices([i["cluster"] for i in pos])
    uneg, dneg = boot_indices([i["cluster"] for i in neg], seed=SEED + 1)
    for a in arms:
        ia = {i["image_id"]: i for i in images if i["arm"] == a}
        for thr in THRESHOLDS:
            P_ = [ia[i["image_id"]] for i in pos]
            N_ = [ia[i["image_id"]] for i in neg]
            res["presence"][f"{a}@{thr}"] = {
                "presence_recall": cluster_rate([x[f"fired@{thr}"] for x in P_],
                                                [x["cluster"] for x in P_], upos, dpos),
                "localized_recall": cluster_rate([x[f"loc_hit@{thr}"] for x in P_],
                                                 [x["cluster"] for x in P_], upos, dpos),
                "fire_rate_pool_negative": cluster_rate([x[f"fired@{thr}"] for x in N_],
                                                        [x["cluster"] for x in N_], uneg, dneg),
                "dets_per_image_all": float(np.mean([ia[i["image_id"]][f"n_dets@{thr}"]
                                                      for i in im0])),
                "matched_frac_of_dets": (sum(ia[i["image_id"]][f"n_matched@{thr}"] for i in im0)
                                         / max(1, sum(ia[i["image_id"]][f"n_dets@{thr}"]
                                                      for i in im0))),
            }
    # --- point-level (in-view image-ramp pairs), cluster = ramp
    p0 = [p for p in pairs if p["arm"] == a0 and p["thr"] == THRESHOLDS[0]]
    upr, dpr = boot_indices([p["ramp"] for p in p0], seed=SEED + 2)
    key = lambda p: (p["image_id"], p["ramp"])  # noqa: E731
    pa = {(p["arm"], p["thr"], key(p)): p for p in pairs}
    order = [key(p) for p in p0]
    cl = [p["ramp"] for p in p0]
    for a in arms:
        for thr in THRESHOLDS:
            rows_a = [pa[(a, thr, k)] for k in order]
            ent = {}
            for test in ["hit_bearing"] + [f"hit_world_{h}" for h in WORLD_HEIGHTS]:
                ent[test] = cluster_rate([r[test] for r in rows_a], cl, upr, dpr)
            for lo, hi in RANGE_BINS:
                b = f"{lo:g}-{hi:g}"
                sel = [i for i, r in enumerate(rows_a) if r["bin"] == b]
                ent[f"hit_bearing[{b}]"] = cluster_rate([rows_a[i]["hit_bearing"] for i in sel],
                                                        [cl[i] for i in sel], upr, dpr)
            res["points"][f"{a}@{thr}"] = ent
    # --- paired contrasts vs canvas_level
    ref = "canvas_level" if "canvas_level" in arms else a0
    for a in arms:
        if a == ref:
            continue
        for thr in THRESHOLDS:
            ra = [pa[(a, thr, k)]["hit_bearing"] for k in order]
            rr = [pa[(ref, thr, k)]["hit_bearing"] for k in order]
            ia = {i["image_id"]: i for i in images if i["arm"] == a}
            ir = {i["image_id"]: i for i in images if i["arm"] == ref}
            res["paired"][f"{a}-{ref}@{thr}"] = {
                "point_hit_bearing": paired_diff(ra, rr, cl, upr, dpr),
                "presence_recall": paired_diff([ia[i["image_id"]][f"fired@{thr}"] for i in pos],
                                               [ir[i["image_id"]][f"fired@{thr}"] for i in pos],
                                               [i["cluster"] for i in pos], upos, dpos),
                "fire_rate_pool_negative": paired_diff(
                    [ia[i["image_id"]][f"fired@{thr}"] for i in neg],
                    [ir[i["image_id"]][f"fired@{thr}"] for i in neg],
                    [i["cluster"] for i in neg], uneg, dneg)}
    # --- pano reference on the same ramps
    ramps_iv = {p["ramp"] for p in p0}
    pr = pano_reference(ramps_iv)
    for scope in ("same_ramps", "all_pool"):
        sel = [x for x in pr if x["same_ramps"]] if scope == "same_ramps" else pr
        u, dr = boot_indices([x["ramp"] for x in sel], seed=SEED + 3)
        ent = {"n_captures": len(sel), "n_ramps": len(u)}
        for thr in THRESHOLDS:
            ent[f"hit@{thr}"] = cluster_rate([x["world_conf"] >= thr for x in sel],
                                             [x["ramp"] for x in sel], u, dr)
            for lo, hi in RANGE_BINS:
                b = f"{lo:g}-{hi:g}"
                s2 = [x for x in sel if x["bin"] == b]
                ent[f"hit@{thr}[{b}]"] = cluster_rate([x["world_conf"] >= thr for x in s2],
                                                      [x["ramp"] for x in s2], u, dr)
        res["pano_reference"][scope] = ent
    pb = pano_bearing_check()
    u, dr = boot_indices([x["ramp"] for x in pb], seed=SEED + 5)
    res["pano_reference"]["test_check@0.55"] = {
        "n_captures": len(pb), "n_ramps": len(u),
        "bearing_test": cluster_rate([x["bearing_hit"] for x in pb], [x["ramp"] for x in pb], u, dr),
        "world_test": cluster_rate([x["world_hit"] for x in pb], [x["ramp"] for x in pb], u, dr),
        "agree": rnd(np.mean([x["bearing_hit"] == x["world_hit"] for x in pb]))}
    # --- flat vs pano, ramp-level paired: per ramp, flat pair-hit rate minus pano capture
    # hit rate, over ramps that have both
    for a in arms:
        for thr in THRESHOLDS:
            fr, pr_ = {}, {}
            for k in order:
                p = pa[(a, thr, k)]
                fr.setdefault(p["ramp"], []).append(p["hit_bearing"])
            for x in pr:
                if x["same_ramps"]:
                    pr_.setdefault(x["ramp"], []).append(x["world_conf"] >= thr)
            both = sorted(set(fr) & set(pr_))
            diff = [np.mean(fr[r]) - np.mean(pr_[r]) for r in both]
            u, dr = boot_indices(both, seed=SEED + 4)
            res["paired"][f"{a}-pano@{thr}"] = {
                "ramp_mean_hit_diff": cluster_rate(diff, both, u, dr),
                "n_ramps": len(both)}
    res = _round(res)
    write_json(os.path.join(res_dir, "results.json"), res)
    with open(os.path.join(res_dir, "images_scored.csv"), "w", encoding="utf-8", newline="") as f:
        cols = sorted(images[0])
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for i in images:
            w.writerow(i)
    md = markdown(res, arms)
    with open(os.path.join(res_dir, "results.md"), "w", encoding="utf-8", newline="") as f:
        f.write(md)
    print(md)


def _round(o):
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return rnd(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def markdown(res, arms):
    L = ["# Perspective photos (#218): Richmond results", "",
         "Generated by `scripts/analysis/perspective_photos_218.py score`. "
         "95% CIs: cluster bootstrap (images clustered by nearest pool ramp; pairs by ramp), "
         f"{N_REPS} reps.", "", "## Counts", ""]
    for k, v in res["counts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Image level", "",
          "| arm @ thr | presence recall | localized recall | fire rate, pool-negative | "
          "dets / image | matched fraction |", "|---|---|---|---|---|---|"]
    for a in arms:
        for thr in THRESHOLDS:
            e = res["presence"][f"{a}@{thr}"]
            L.append(f"| {a} @ {thr} | {fmt_ci(e['presence_recall'])} | "
                     f"{fmt_ci(e['localized_recall'])} | {fmt_ci(e['fire_rate_pool_negative'])} | "
                     f"{e['dets_per_image_all']:.2f} | {e['matched_frac_of_dets']:.3f} |")
    L += ["", "## Point hits (in-view image-ramp pairs)", "",
          "| arm @ thr | bearing test | world 1.5 m | world 2.6 m | "
          + " | ".join(f"bearing {lo:g}-{hi:g} m" for lo, hi in RANGE_BINS) + " |",
          "|---|---|---|---|" + "---|" * len(RANGE_BINS)]
    for a in arms:
        for thr in THRESHOLDS:
            e = res["points"][f"{a}@{thr}"]
            L.append(f"| {a} @ {thr} | {fmt_ci(e['hit_bearing'])} | "
                     f"{fmt_ci(e['hit_world_1.5'])} | {fmt_ci(e['hit_world_2.6'])} | "
                     + " | ".join(f"{fmt_ci(e[f'hit_bearing[{lo:g}-{hi:g}]'])} "
                                  f"(n={e[f'hit_bearing[{lo:g}-{hi:g}]'][3]})"
                                  for lo, hi in RANGE_BINS) + " |")
    L += ["", "## 360 pano reference (world test, non-source captures, 3-18 m)", ""]
    for scope, e in res["pano_reference"].items():
        if scope.startswith("test_check"):
            L.append(f"- {scope} (same captures, both tests): {e['n_captures']} captures of "
                     f"{e['n_ramps']} ramps; bearing test {fmt_ci(e['bearing_test'])}, world "
                     f"test {fmt_ci(e['world_test'])}, per-capture agreement {e['agree']}")
            continue
        L.append(f"- {scope}: {e['n_captures']} captures of {e['n_ramps']} ramps")
        for thr in THRESHOLDS:
            L.append(f"  - @ {thr}: {fmt_ci(e[f'hit@{thr}'])}; "
                     + "; ".join(f"{lo:g}-{hi:g} m {fmt_ci(e[f'hit@{thr}[{lo:g}-{hi:g}]'])}"
                                 for lo, hi in RANGE_BINS))
    L += ["", "## Paired contrasts", ""]
    for k, e in res["paired"].items():
        L.append(f"- {k}: " + "; ".join(f"{m} {fmt_ci(v) if isinstance(v, list) else v}"
                                         for m, v in e.items()))
    L += ["", "## Peaks in the canvas fill (dropped)", ""]
    for a, e in res["fill_peaks"].items():
        L.append(f"- {a}: {e}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# gallery: detections for a human rater (precision; nothing is rated here)
# --------------------------------------------------------------------------- #
FP_DIR = os.path.join(REPO, "benchmark", "richmond_flat_fp_218")
FP_REL = "benchmark/richmond_flat_fp_218/gallery.html"
FP_QUESTION = "Is there a curb ramp at the ring?"
FP_RUBRIC = [
    ("yes", "Yes",
     "A curb ramp is at the ring or touching it: a sloped section that takes the sidewalk "
     "down through the curb to street level (Project Sidewalk: 'a curb ramp connecting "
     "sidewalk to street'). It may be partly hidden or far away, as long as you can see "
     "that it is a ramp."),
    ("no", "No",
     "No curb ramp at the ring: plain curb, sidewalk, street, a driveway (Project Sidewalk "
     "puts no Curb Ramp label on driveways), a level crossing with no curb, or something "
     "else; and the nearest ramp, if any, is more than roughly one ramp width away."),
    ("cant_tell", "Can't tell",
     "The crop does not let you decide (too dark, blocked, too blurry, too far). Excluded "
     "from every rate."),
]
FP_RULES = [
    "Judge the spot under the ring, not whether a ramp exists somewhere in the crop.",
    "The cards are shuffled and mix detections that matched a known ramp with ones that did "
    "not; nothing on the card says which.",
    "Add a note for anything worth recording, e.g. 'ramp 1 m left of ring'.",
]
FP_MAX_UNMATCHED = 150
FP_MATCHED_CONTROL = 40
CROP_W, CROP_H = 720, 480


def gallery_items(arm, thr=PRIMARY_THR, seed=SEED):
    """The detections to rate: every ``arm`` detection >= thr that no pool ramp claimed
    under the bearing test (a random FP_MAX_UNMATCHED if there are more), plus a random
    FP_MATCHED_CONTROL of the claimed ones as a blind control. Shuffled."""
    rows = read_csv(IMAGES_CSV)
    by_id = {r["image_id"]: r for r in rows}
    ramps = ramp_table()
    recs = load_dets(arm)
    unmatched, matched = [], []
    for iid in sorted(recs):
        rec = recs[iid]
        dets = rec["dets"]
        if not any(d["score"] >= thr for d in dets):
            continue
        g = image_geometry(by_id[iid], rec["width"], rec["height"], ramps)
        b, dep, _ = det_world(dets, g["cam"], g["R_wc"])
        cl = claim_bearing(dets, b, dep, g["near"], thr)
        claimed_idx = {i: u for u, i in cl.items()}
        for i, d in enumerate(dets):
            if d["score"] < thr:
                continue
            it = {"image_id": iid, "det": i, "u": d["u"], "v": d["v"], "score": d["score"],
                  "width": rec["width"], "height": rec["height"],
                  "matched_ramp": claimed_idx.get(i), "pool_negative": g["pool_negative"]}
            (matched if i in claimed_idx else unmatched).append(it)
    rng = np.random.default_rng(seed)
    if len(unmatched) > FP_MAX_UNMATCHED:
        unmatched = [unmatched[k] for k in sorted(rng.choice(len(unmatched), FP_MAX_UNMATCHED,
                                                             replace=False))]
    ctrl = [matched[k] for k in sorted(rng.choice(len(matched), min(FP_MATCHED_CONTROL,
                                                                   len(matched)), replace=False))]
    items = unmatched + ctrl
    order = rng.permutation(len(items))
    items = [items[k] for k in order]
    for k, it in enumerate(items, 1):
        it["item"] = f"d{k:03d}"
    return items, {"unmatched_total": len(unmatched), "matched_total": len(matched)}


def crop_box(u, v, w, h, cw=CROP_W, ch=CROP_H):
    """A cw x ch box centred on (u, v), shifted to stay inside a w x h image (smaller if
    the image is)."""
    cw, ch = min(cw, w), min(ch, h)
    x0 = int(round(min(max(u - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(v - ch / 2, 0), h - ch)))
    return x0, y0, x0 + cw, y0 + ch


def cmd_gallery(args):
    from PIL import Image
    import rating_page_218 as RP
    items, totals = gallery_items(args.arm)
    fetched = {r["image_id"]: r for r in read_csv(FETCHED_CSV)}
    img_dir = os.path.join(FP_DIR, "img")
    os.makedirs(img_dir, exist_ok=True)
    cards = []
    for it in items:
        box = crop_box(it["u"], it["v"], it["width"], it["height"])
        dst = os.path.join(img_dir, f"{it['item']}.jpg")
        src = os.path.join(args.images, f"{it['image_id']}.jpg")
        if sha256_file(src) != fetched[it["image_id"]]["sha256"]:
            raise SystemExit(f"{src}: sha256 differs from fetched.csv")
        Image.open(src).convert("RGB").crop(box).save(dst, quality=90)
        it["crop_box"] = list(box)
        w, h = box[2] - box[0], box[3] - box[1]
        cards.append({"name": it["item"], "img": f"img/{it['item']}.jpg", "w": w, "h": h,
                      "ring": ((it["u"] - box[0]) / w, (it["v"] - box[1]) / h),
                      "alt": f"Crop of a Richmond flat photo, ring on detection {it['item']}"})
    digest = hashlib.sha256("\n".join(
        f"{it['item']} {it['image_id']} {fetched[it['image_id']]['sha256']} {it['crop_box']}"
        for it in items).encode()).hexdigest()[:16]
    write_json(os.path.join(FP_DIR, "manifest.json"), {
        "arm": args.arm, "threshold": PRIMARY_THR, "items": items, "totals": totals,
        "manifest_digest": digest, "question": FP_QUESTION,
        "rubric": [{"key": k, "label": lab, "definition": d} for k, lab, d in FP_RUBRIC],
        "rules": FP_RULES, "image_sha256": {it["image_id"]: fetched[it["image_id"]]["sha256"]
                                            for it in items},
        "note": "img/ is not committed; `perspective_photos_218.py fetch` then `gallery` "
                "rebuilds it byte-for-byte from the sha256-checked thumbnails"})
    page = RP.render(cards, digest, {
        "title": "Richmond flat detections", "h1": "Richmond flat photos (#218): detections",
        "intro": ("<p>Each card is a crop of one Richmond flat (non-360) Mapillary photo, with a "
                  "ring on a RampNet detection (canvas-embed arm, score at least 0.30). Answer "
                  f"one question: <strong>{html_escape(FP_QUESTION)}</strong></p>"),
        "question": FP_QUESTION, "rubric": FP_RUBRIC, "rules": FP_RULES,
        "keys": {"y": "yes", "n": "no", "c": "cant_tell"},
        "task": "RampNet #218 Richmond flat photos, detections: " + FP_QUESTION,
        "export_prefix": "richmond_flat_fp__", "storage_prefix": "rflat218_",
        "gallery_rel": FP_REL, "commit_dir": "benchmark/richmond_flat_fp_218/"})
    with open(os.path.join(FP_DIR, "gallery.html"), "w", encoding="utf-8", newline="") as f:
        f.write(page)
    print(f"{len(items)} cards ({totals}), digest {digest} -> {FP_DIR}/gallery.html")


def html_escape(s):
    import html
    return html.escape(s)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    f = sub.add_parser("fetch")
    f.add_argument("--env", required=True)
    f.add_argument("--out", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--images", required=True)
    i.add_argument("--arms", default="canvas_level,canvas_sfm,stretch")
    i.add_argument("--out", default=None)
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--verify-sha", action="store_true")
    i.add_argument("--usage-log", default=None,
                   help="default: the main checkout's analysis_out/usage_log.jsonl; 'none' skips")
    s = sub.add_parser("score")
    s.add_argument("--arms", default="canvas_level,canvas_sfm,stretch")
    s.add_argument("--dets-dir", default=None, help="default analysis_out/perspective_photos_218")
    s.add_argument("--results-dir", default=None, help="default analysis_out/perspective_photos_218")
    g = sub.add_parser("gallery")
    g.add_argument("--images", required=True)
    g.add_argument("--arm", default="canvas_level")
    args = ap.parse_args(argv)
    {"select": cmd_select, "fetch": cmd_fetch, "infer": cmd_infer, "score": cmd_score,
     "gallery": cmd_gallery}[args.cmd](args)


if __name__ == "__main__":
    main()
