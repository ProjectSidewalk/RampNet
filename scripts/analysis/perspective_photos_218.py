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
    # after a rating pass: precision (strata weighted back), agreement for 2+ raters
    python scripts/analysis/perspective_photos_218.py rates \\
        --verdicts benchmark/richmond_flat_fp_218/richmond_flat_fp__<rater>.json
    # ledger rows from the rows infer wrote (see docs section 9)
    python scripts/analysis/perspective_photos_218.py reledger --raw RAW --replace-in LEDGER

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
LATERAL_LOOSE_M = 10.0                 # sensitivity: how much could pose error explain?
H_MIN, H_MAX = 0.5, 4.0                # heights the bearing test accepts
WORLD_HEIGHTS = (1.5, 2.6)             # flat-ground raycast sensitivity (2.6 = labeler's)
# chance floor of the bearing test (docs section 3, "Chance floor")
N_NULL = 20                            # swap-null draws
NULL_TESTS = ("hit_bearing", "hit_bearing_loose", "hit_bearing_no_hgate")
PANO_NULL_SHIFTS = (0.25, 0.5, 0.75)   # pano rotation null: 90 / 180 / 270 deg
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


def build_canvas_tensor(img, cam, M, H, W, device=None):
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
    if device is not None:        # sampling on the GPU; bilinear either way
        t, grid = t.to(device), grid.to(device)
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
        t, inside = build_canvas_tensor(img, cam, M, H, W, device)
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
def dets_path(arm, out=OUT, shard=None):
    suffix = f".shard{shard[0]}of{shard[1]}" if shard else ""
    return os.path.join(out, f"dets_{arm}{suffix}.jsonl")


def parse_shard(s):
    """'2/4' -> (2, 4): this process takes rows[2::4]."""
    if not s:
        return None
    k, n = (int(x) for x in s.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {s}: need 0 <= k < n")
    return k, n


def cmd_merge(args):
    """Concatenate the shard files of each arm into dets_<arm>.jsonl (sorted by image id)
    and one meta; the shard files are removed."""
    import glob
    out = args.out or OUT
    for a in args.arms.split(","):
        parts = sorted(glob.glob(os.path.join(out, f"dets_{a}.shard*of*.jsonl")))
        if not parts:
            continue
        recs, metas = [], []
        for pth in parts:
            with open(pth, encoding="utf-8") as f:
                recs += [json.loads(x) for x in f if x.strip()]
            metas.append(json.load(open(pth[:-len(".jsonl")] + ".meta.json", encoding="utf-8")))
        ids = [r["image_id"] for r in recs]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{a}: duplicate images across shards")
        recs.sort(key=lambda r: r["image_id"])
        with open(dets_path(a, out), "w", encoding="utf-8", newline="") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        meta = dict(metas[0])
        meta.update({"n_images": len(recs), "shards": len(parts),
                     "elapsed_s": round(sum(m["elapsed_s"] for m in metas), 3),
                     "sha256_mismatch": sum(m["sha256_mismatch"] for m in metas),
                     "started": min(m["started"] for m in metas)})
        write_json(dets_path(a, out)[:-len(".jsonl")] + ".meta.json", meta)
        for pth in parts:
            os.remove(pth)
            os.remove(pth[:-len(".jsonl")] + ".meta.json")
        print(f"{a}: {len(parts)} shards, {len(recs)} images -> {dets_path(a, out)}")


def shard_note(n_shards):
    """The sentence a sharded run adds to a ledger row's ``what``."""
    return (f"; one of {n_shards} shard processes sharing the GPU, so gpu_hours = elapsed / "
            f"{n_shards}")


def usage_row(label, n, elapsed_s, started, host, gpus, what, issue=218,
              script="scripts/analysis/perspective_photos_218.py", bundle=None,
              gpu_share=1.0, concurrent_with=None):
    """One ``paid: false`` ledger row. ``gpu_share`` < 1 is for a process that shared the
    GPU with ``1 / gpu_share`` - 1 others of the same run (``--shard``): its GPU-hours are
    its elapsed time times its share, so the shards of one run sum to about the run's
    wall-clock rather than to N times it. ``concurrent_with`` lists other jobs on the same
    GPU (so the GPU-hours are upper bounds)."""
    row = {"provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
           "hardware": {"host": host, "gpus": gpus}, "status": "ok", "est_cost_usd": 0.0,
           "pricing": None, "issue": issue, "ts": started,
           "bundle": bundle or "analysis_out/perspective_photos_218/images.csv",
           "label": label, "run_id": f"{label}:{host}:{started}", "panos_scored": n,
           "elapsed_s": round(elapsed_s, 3), "s_per_pano": round(elapsed_s / n, 4) if n else None,
           "gpu_hours": round(elapsed_s * gpu_share / 3600.0, 4), "what": what, "script": script}
    if gpu_share != 1.0:
        row["gpu_share"] = gpu_share
    if concurrent_with:
        row["concurrent_with"] = list(concurrent_with)
    return row


def cmd_reledger(args):
    """Rebuild ledger rows from the rows ``infer`` wrote before ``usage_row`` knew about
    shards (``--raw``, committed as ``usage_rows_raw*.jsonl``): same label, run id, times and
    counts; ``gpu_share`` from the label's ``:shard<k>of<n>`` tag; ``concurrent_with`` from
    the command line. Writes the rows to ``--out`` and, with ``--replace-in``, replaces the
    rows with the same ``run_id`` in that ledger in place, leaving every other line as it
    was."""
    import re
    rows = []
    with open(args.raw, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = re.search(r":shard\d+of(\d+)$", r["label"])
            share = 1.0 / int(m.group(1)) if m else 1.0
            what = r["what"] + (shard_note(int(m.group(1))) if m else "")
            rows.append(usage_row(r["label"], r["panos_scored"], r["elapsed_s"], r["ts"],
                                  r["hardware"]["host"], r["hardware"]["gpus"], what,
                                  r["issue"], r["script"], r["bundle"], share,
                                  args.concurrent_with))
            assert rows[-1]["run_id"] == r["run_id"]
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if args.replace_in:
        new = {r["run_id"]: r for r in rows}
        with open(args.replace_in, encoding="utf-8", newline="") as f:
            lines = f.readlines()
        n_rep = 0
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            rid = json.loads(line).get("run_id")
            if rid in new:
                lines[k] = json.dumps(new.pop(rid)) + "\n"
                n_rep += 1
        if new:
            raise SystemExit(f"{len(new)} rows not found in {args.replace_in}: {sorted(new)}")
        with open(args.replace_in, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        print(f"replaced {n_rep} rows in {args.replace_in}")
    print(f"{len(rows)} rows, {sum(r['gpu_hours'] for r in rows):.4f} GPU-h")


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
    shard = parse_shard(args.shard)
    if shard:
        rows = rows[shard[0]::shard[1]]
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
        path = dets_path(a, out, shard)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            for rec in recs[a]:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        meta = {"arm": a, "spec": ARMS[a], "n_images": len(recs[a]), "floor": FLOOR,
                "min_distance": MIN_DISTANCE, "host": host, "gpus": gpus,
                "device": device.type, "fp16": False, "started": started,
                "elapsed_s": round(t_arm[a], 3), "sha256_mismatch": n_bad,
                "torch": torch.__version__, "shard": args.shard or None}
        write_json(path[:-len(".jsonl")] + ".meta.json", meta)
        print(f"{a}: {len(recs[a])} images, {t_arm[a]:.0f} s -> {path}")
    if args.usage_log != "none" and not args.limit:
        ul = args.usage_log or os.path.join(ledger.canonical_repo_root(REPO) or REPO,
                                            "analysis_out", "usage_log.jsonl")
        tag = f":shard{shard[0]}of{shard[1]}" if shard else ""
        rows_u = [usage_row(f"perspective-218:richmond:{a}{tag}", len(recs[a]), t_arm[a], started,
                            host, gpus,
                            f"perspective_photos_218.py infer, arm {a}: per-image seconds "
                            f"(canvas build + forward + peaks), fp32, run wall "
                            f"{time.time() - t_all:.0f} s for all arms"
                            + (shard_note(shard[1]) if shard else ""),
                            gpu_share=1.0 / shard[1] if shard else 1.0,
                            concurrent_with=args.concurrent_with)
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
    ramp, seen from VIEW_H m with the SfM pose, is inside the distortion model's monotonic
    range (``P.in_distortion_domain``) and projects inside the frame at least
    EDGE_MARGIN_FRAC of the width from the left/right edges and above the bottom edge.
    Without the domain check, cameras with k2 < 0 fold rays from 60-70 deg off-axis back
    into the frame; those pairs are flagged ``folded`` and are not in view.
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
        in_view = folded = False
        if RANGE_MIN <= d[i] <= RANGE_MAX:
            pc = R_wc @ np.array([e[i], n[i], -VIEW_H])
            if pc[2] > 0:
                u, v = P.project_cam(cam, pc)
                m = EDGE_MARGIN_FRAC * cam.width
                in_frame = bool(m <= u <= cam.width - 1 - m and v <= cam.height - 1)
                folded = in_frame and not bool(P.in_distortion_domain(cam, pc))
                in_view = in_frame and not folded
        if d[i] <= CANDIDATE_MAX:
            near.append({"uid": uids[i], "range": float(d[i]), "bearing": float(bearing[i]),
                         "dbear": dbear, "e": float(e[i]), "n": float(n[i]),
                         "in_view": in_view, "folded": folded})
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


def claim_bearing(dets, b, dep, near, thr, lateral=LATERAL_M, h_min=H_MIN, h_max=H_MAX):
    """Greedy one-to-one claims in descending score: each detection >= thr claims the
    unclaimed candidate ramp (range <= CANDIDATE_MAX) with the smallest bearing error that
    passes ``bearing_hit``. Returns {ramp uid: det index}. ``h_min=0, h_max=inf`` drops the
    height gate (the detection need only be below the horizon)."""
    order = sorted((i for i, d in enumerate(dets) if d["score"] >= thr),
                   key=lambda i: -dets[i]["score"])
    claimed = {}
    for i in order:
        best, best_err = None, None
        for r in near:
            if r["uid"] in claimed:
                continue
            if not P.bearing_hit(b[i], dep[i], r["bearing"], r["range"], lateral,
                                 h_min, h_max):
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


def bearing_claims(dets, b, dep, near, thr):
    """The bearing-family claims: {test name: {ramp uid: det index}}."""
    return {"hit_bearing": claim_bearing(dets, b, dep, near, thr),
            "hit_bearing_loose": claim_bearing(dets, b, dep, near, thr, LATERAL_LOOSE_M),
            "hit_bearing_no_hgate": claim_bearing(dets, b, dep, near, thr, LATERAL_M,
                                                  0.0, math.inf)}


def mirrored(near, heading):
    """Chance-floor helper: every candidate ramp's bearing reflected about the camera
    heading (a ramp 20 deg right of the heading moves to 20 deg left)."""
    return [dict(r, bearing=float((heading - r["dbear"]) % 360)) for r in near]


def transplant(dets, w_from, h_from, w_to, h_to):
    """Chance-floor helper: detections moved to another image at the same normalised pixel
    position (u / width, v / height)."""
    return [dict(d, u=(d["u"] + 0.5) / w_from * w_to - 0.5,
                 v=(d["v"] + 0.5) / h_from * h_to - 0.5) for d in dets]


def count_bucket(n):
    """Detection-count bucket for the count-matched swap null: 0 / 1 / 2 / 3+."""
    return min(int(n), 3)


def swap_donors(ids, clusters, n_null=N_NULL, seed=SEED, buckets=None):
    """For each draw, a donor for every image in ``ids``: an image drawn uniformly (with
    replacement) from ``ids`` whose cluster (nearest pool ramp) differs from the
    receiver's, so a donor never shows the receiver's own corner. Without ``buckets`` the
    donors are the same for every arm and threshold, so the null is paired across arms.

    With ``buckets`` ({image: bucket}, e.g. ``count_bucket`` of its detections >= 0.30)
    the donor must also share the receiver's bucket: the count-matched null, which keeps
    how often and how much the model fires on the receiving image and randomises only
    where. If no donor shares both, the bucket condition is dropped for that receiver.
    Seeded separately (``seed + 8``) so the unmatched null is unchanged by it."""
    rng = np.random.default_rng(seed + (8 if buckets is not None else 7))
    ids = list(ids)
    cl = np.array([clusters[i] for i in ids])
    bk = np.array([buckets[i] for i in ids]) if buckets is not None else None
    out = []
    for _ in range(n_null):
        dr = {}
        for k, i in enumerate(ids):
            ok = cl != cl[k]
            if bk is not None and np.any(ok & (bk == bk[k])):
                ok = ok & (bk == bk[k])
            pool = np.nonzero(ok)[0]
            dr[i] = ids[int(pool[rng.integers(0, len(pool))])]
        out.append(dr)
    return out


def per_image_table(arm_recs, rows, ramps, thresholds=THRESHOLDS, n_null=N_NULL):
    """Per (arm, image): positive / pool_negative / fired, and per (arm, image, in-view
    ramp): hit under each test. Returns (images, pairs, matches) lists of dicts.

    Chance floor (docs section 3): on positive images every pair also carries
    - ``null_swap_<test>``: per draw (``n_null`` of them), whether the ramp is hit when the
      image's own detections are replaced by those of an unrelated positive image (a
      different nearest pool ramp) at the same normalised pixel positions, projected
      through THIS image's camera and SfM pose;
    - ``null_swapc_<test>``: the same with count-matched donors (``swap_donors`` with
      ``buckets``): the donor also has the receiver's number of detections >= 0.30
      (0 / 1 / 2 / 3+), per arm;
    - ``null_mirror_<test>``: whether it is hit when every candidate ramp's bearing is
      mirrored about the camera heading.
    ``matches`` lists every bearing-test claim (real and swap-null) with the camera height
    it implies, for the height diagnostic."""
    images, pairs, matches = [], [], []
    by_id = {r["image_id"]: r for r in rows}
    ids = sorted(set.intersection(*[set(v) for v in arm_recs.values()]))
    any_recs = next(iter(arm_recs.values()))
    geo = {iid: image_geometry(by_id[iid], any_recs[iid]["width"], any_recs[iid]["height"],
                               ramps) for iid in ids}
    pos_ids = [i for i in ids if geo[i]["positive"]]
    donors = swap_donors(pos_ids, {i: by_id[i]["nearest_ramp"] for i in pos_ids}, n_null) \
        if n_null else []
    # count-matched swap null (S6 of the PR #227 re-review): per arm, donors from the same
    # detections->=PRIMARY_THR bucket as the receiver
    donors_c = {arm: swap_donors(
        pos_ids, {i: by_id[i]["nearest_ramp"] for i in pos_ids}, n_null,
        buckets={i: count_bucket(sum(d["score"] >= PRIMARY_THR for d in recs[i]["dets"]))
                 for i in pos_ids}) if n_null else []
        for arm, recs in arm_recs.items()}
    for iid in ids:
        row = by_id[iid]
        g = geo[iid]
        cluster = row["nearest_ramp"]
        heading = float(row["computed_compass_angle"])
        near_m = mirrored(g["near"], heading)
        cam_model = f'{row["make"]} {row["model"]}'.strip()
        for arm, recs in arm_recs.items():
            rec = recs[iid]
            dets = rec["dets"]
            b, dep, w = det_world(dets, g["cam"], g["R_wc"])
            im = {"arm": arm, "image_id": iid, "cluster": cluster, "positive": g["positive"],
                  "pool_negative": g["pool_negative"], "max_score": rec["max_score"],
                  "n_in_view": sum(r["in_view"] for r in g["near"]),
                  "n_folded": sum(r["folded"] for r in g["near"])}
            swap, swapc = [], []
            if g["positive"]:
                for dlist, out in ((donors, swap), (donors_c[arm], swapc)):
                    for dr in dlist:
                        dd = recs[dr[iid]]
                        sd = transplant(dd["dets"], dd["width"], dd["height"], rec["width"],
                                        rec["height"])
                        sb, sdep, _ = det_world(sd, g["cam"], g["R_wc"])
                        out.append((sd, sb, sdep))
            in_view_uids = {r["uid"] for r in g["near"] if r["in_view"]}
            rng_of = {r["uid"]: r["range"] for r in g["near"]}
            for thr in thresholds:
                bc = bearing_claims(dets, b, dep, g["near"], thr)
                cl = bc["hit_bearing"]
                im[f"fired@{thr}"] = any(d["score"] >= thr for d in dets)
                im[f"n_dets@{thr}"] = sum(d["score"] >= thr for d in dets)
                im[f"n_matched@{thr}"] = len(cl)
                im[f"loc_hit@{thr}"] = bool(in_view_uids & set(cl))
                im[f"implied_h@{thr}"] = [
                    rng_of[u] * math.tan(math.radians(dep[i])) for u, i in cl.items()]
                for u, i in cl.items():
                    matches.append({"arm": arm, "thr": thr, "kind": "real", "draw": None,
                                    "image_id": iid, "sequence": row["sequence"],
                                    "camera": cam_model, "ramp": u,
                                    "h": rng_of[u] * math.tan(math.radians(dep[i]))})
                wcl = {h: claim_world(dets, w, g["near"], thr, h) for h in WORLD_HEIGHTS}
                null_m, null_s, null_c = {}, [], []
                if g["positive"]:
                    null_c = [bearing_claims(sd, sb, sdep, g["near"], thr)
                              for sd, sb, sdep in swapc]
                    im[f"null_swapc_loc_hit@{thr}"] = [
                        bool(in_view_uids & set(sc["hit_bearing"])) for sc in null_c]
                    null_m = bearing_claims(dets, b, dep, near_m, thr)
                    for k, (sd, sb, sdep) in enumerate(swap):
                        sc = bearing_claims(sd, sb, sdep, g["near"], thr)
                        null_s.append(sc)
                        for u, i in sc["hit_bearing"].items():
                            matches.append({"arm": arm, "thr": thr, "kind": "swap", "draw": k,
                                            "image_id": iid, "sequence": row["sequence"],
                                            "camera": cam_model, "ramp": u,
                                            "h": rng_of[u] * math.tan(math.radians(sdep[i]))})
                    im[f"null_swap_loc_hit@{thr}"] = [bool(in_view_uids & set(sc["hit_bearing"]))
                                                      for sc in null_s]
                    im[f"null_mirror_loc_hit@{thr}"] = bool(
                        in_view_uids & set(null_m["hit_bearing"]))
                for r in g["near"]:
                    if not r["in_view"]:
                        continue
                    pr = {"arm": arm, "image_id": iid, "ramp": r["uid"], "camera": cam_model,
                          "range": r["range"], "bin": range_bin(r["range"]), "thr": thr,
                          **{t: r["uid"] in bc[t] for t in NULL_TESTS},
                          **{f"hit_world_{h}": r["uid"] in wcl[h] for h in WORLD_HEIGHTS}}
                    for t in NULL_TESTS:
                        pr[f"null_mirror_{t}"] = r["uid"] in null_m[t]
                        pr[f"null_swap_{t}"] = [r["uid"] in sc[t] for sc in null_s]
                        pr[f"null_swapc_{t}"] = [r["uid"] in sc[t] for sc in null_c]
                    pairs.append(pr)
            images.append(im)
    return images, pairs, matches


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


def pano_bearing_check(thr=0.55, shift=0.0):
    """Test comparability: the flat photos' bearing test applied to the same pano captures
    whose world test is in ``pano_reference``. Only detections >= 0.55 are stored in the
    neighbourhood records, so this runs at 0.55 only. The ramp's bearing in the pano is
    its projected column (``x_proj``); the detection's depression is read off a level
    equirect, ``(y - 0.5) * 180``. Returns per-capture rows with both tests.

    ``shift`` (a fraction of 360 deg) rotates every detection's column before matching:
    the chance floor of the bearing test on the panos (0.25 / 0.5 / 0.75)."""
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
                db = float(P.wrap_deg((dd["x_normalized"] + shift - float(c["x_proj"]))
                                      * 360.0))
                dep = (dd["y_normalized"] - 0.5) * 180.0
                if P.bearing_hit(db, dep, 0.0, float(c["dist_m"]), LATERAL_M, H_MIN, H_MAX):
                    if best is None or abs(db) < be:
                        best, be = c["ramp_uid"], abs(db)
            if best:
                claimed.add(best)
        for c in caps:
            wc = float(c["world_conf"]) if c["world_conf"] not in ("", "nan") else 0.0
            rows.append({"ramp": c["ramp_uid"], "pano_id": pid, "range": float(c["dist_m"]),
                         "bin": range_bin(float(c["dist_m"])),
                         "bearing_hit": c["ramp_uid"] in claimed, "world_hit": wc >= thr})
    return rows


def null_draw_rates(rows, key):
    """Per swap-null draw, the mean of ``key`` (a per-row list of per-draw hits) over
    ``rows``. Returns {mean, p5, p95, n_draws}; None without draws."""
    if not rows or not rows[0][key]:
        return None
    m = np.array([r[key] for r in rows], dtype=float)      # rows x draws
    per_draw = m.mean(axis=0)
    return {"mean": float(per_draw.mean()), "p5": float(np.percentile(per_draw, 5)),
            "p95": float(np.percentile(per_draw, 95)), "n_draws": int(m.shape[1])}


def above_chance(rows, test, clusters, uniq, draws, kind="swap"):
    """Real hit minus its per-row swap-null expectation (mean over draws), cluster
    bootstrap CI. The null mean is treated as fixed: the spread across draws is reported
    separately (``null_draw_rates``). ``kind``: "swap" or "swapc" (count-matched)."""
    if not rows or not rows[0][f"null_{kind}_{test}"]:
        return None
    return paired_diff([r[test] for r in rows],
                       [float(np.mean(r[f"null_{kind}_{test}"])) for r in rows],
                       clusters, uniq, draws)


def covisible_components(pairs):
    """{ramp uid: component label}: ramps joined whenever they are in view of the same
    photo (union-find). Pairs from one photo share one detection set, so clustering the
    bootstrap by component instead of by ramp is the conservative choice (N9 of the
    PR #227 re-review)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_img = {}
    for p in pairs:
        by_img.setdefault(p["image_id"], []).append(p["ramp"])
    for rs in by_img.values():
        for r in rs:
            find(r)
        for r in rs[1:]:
            parent[find(r)] = find(rs[0])
    return {r: find(r) for r in parent}


def camera_scale(row, width, height):
    """Angular sampling of one photo (docs section 2, S5): px/deg at the thumbnail's
    centre; how much faster the photo is sampled at the horizontal edge than at its centre
    (the canvas downsamples the edge by this factor more than the centre, with no
    antialiasing: ``(1 + 3 k1 t^2 + 5 k2 t^4)(1 + t^2)`` at ``t = tan(edge angle)``, None
    where the edge does not invert); and the stretch arm's px/deg across and down (nominal
    FOV) against the canvas's 4096 / 360."""
    cam = camera_of(row, width, height)
    ray = P.unproject_cam(cam, cam.width - 0.5, (cam.height - 1) / 2.0)
    t = float(ray[0] / ray[2]) if np.all(np.isfinite(ray)) else None
    edge = ((1 + 3 * cam.k1 * t * t + 5 * cam.k2 * t ** 4) * (1 + t * t)) if t is not None \
        else None
    return {"hfov": cam.hfov_deg(), "centre_px_per_deg": cam.focal * cam.size * math.pi / 180,
            "edge_over_centre": edge,
            "stretch_px_per_deg_x": P.CANVAS_W / cam.hfov_deg(),
            "stretch_px_per_deg_y": P.CANVAS_H / cam.vfov_deg()}


def cmd_score(args):
    arms = args.arms.split(",")
    rows = read_csv(IMAGES_CSV)
    by_id = {r["image_id"]: r for r in rows}
    ramps = ramp_table()
    arm_recs = {a: load_dets(a, args.dets_dir or OUT) for a in arms}
    images, pairs, matches = per_image_table(arm_recs, rows, ramps, n_null=args.n_null)
    res_dir = args.results_dir or OUT
    res = {"config": {"arms": arms, "thresholds": list(THRESHOLDS), "floor": FLOOR,
                      "range": [RANGE_MIN, RANGE_MAX], "bins": [list(b) for b in RANGE_BINS],
                      "edge_margin_frac": EDGE_MARGIN_FRAC, "view_h": VIEW_H,
                      "neg_range": NEG_RANGE, "neg_margin_deg": NEG_MARGIN_DEG,
                      "lateral_m": LATERAL_M, "lateral_loose_m": LATERAL_LOOSE_M,
                      "h_accept": [H_MIN, H_MAX], "world_heights": list(WORLD_HEIGHTS),
                      "n_null": args.n_null, "pano_null_shifts": list(PANO_NULL_SHIFTS),
                      "in_view_rule": "in frame and inside the distortion model's monotonic "
                                      "range (fold guard)",
                      "n_reps": N_REPS, "seed": SEED},
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
                     "clusters_positive": len({i["cluster"] for i in im0 if i["positive"]}),
                     "pairs_excluded_by_fold_guard": sum(i["n_folded"] for i in im0)}
    for a in arms:
        recs = arm_recs[a].values()
        hs = [h for i in images if i["arm"] == a for h in i[f"implied_h@{PRIMARY_THR}"]]
        res.setdefault("implied_height", {})[a] = {
            "n": len(hs), "p10_p50_p90": [rnd(x) for x in np.percentile(hs, [10, 50, 90])]
            if hs else None}
        res["fill_peaks"][a] = {"images_with_fill_peaks": sum(r["n_fill_peaks"] > 0 for r in recs),
                                "fill_peaks": sum(r["n_fill_peaks"] for r in recs)}
    # --- implied camera height by camera model (S1): real matches vs swap-null matches
    for a in arms:
        real = [m for m in matches if m["arm"] == a and m["thr"] == PRIMARY_THR
                and m["kind"] == "real"]
        null = [m for m in matches if m["arm"] == a and m["thr"] == PRIMARY_THR
                and m["kind"] == "swap"]
        # a chance-level match is only defined on positive images, where the null ran
        pos_ids = {i["image_id"] for i in im0 if i["positive"]}
        by_cam = {}
        for cam_name in sorted({m["camera"] for m in real}):
            rc = [m for m in real if m["camera"] == cam_name]
            rcp = [m for m in rc if m["image_id"] in pos_ids]
            nc = [m for m in null if m["camera"] == cam_name]
            by_cam[cam_name] = {
                "n_matches": len(rc), "n_sequences": len({m["sequence"] for m in rc}),
                "h_p50": float(np.median([m["h"] for m in rc])),
                "n_matches_positive_images": len(rcp),
                "null_matches_per_draw": len(nc) / max(1, args.n_null),
                "null_h_p50": float(np.median([m["h"] for m in nc])) if nc else None}
        seq_med = {}
        for m in real:
            seq_med.setdefault(m["sequence"], []).append(m["h"])
        res["implied_height"][a].update({
            "n_sequences": len(seq_med),
            "sequences_median_below_2.2m": sum(np.median(v) < 2.2 for v in seq_med.values()),
            "null_swap": {"matches_per_draw": len(null) / max(1, args.n_null),
                          "h_p10_p50_p90": [float(x) for x in np.percentile(
                              [m["h"] for m in null], [10, 50, 90])] if null else None},
            "by_camera": by_cam})
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
            ent = {
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
            if args.n_null:
                ent["localized_recall_null_swap"] = null_draw_rates(
                    P_, f"null_swap_loc_hit@{thr}")
                ent["localized_recall_null_mirror"] = float(np.mean(
                    [x[f"null_mirror_loc_hit@{thr}"] for x in P_]))
                ent["localized_recall_above_swap"] = paired_diff(
                    [x[f"loc_hit@{thr}"] for x in P_],
                    [float(np.mean(x[f"null_swap_loc_hit@{thr}"])) for x in P_],
                    [x["cluster"] for x in P_], upos, dpos)
                ent["localized_recall_null_swapc"] = null_draw_rates(
                    P_, f"null_swapc_loc_hit@{thr}")
                ent["localized_recall_above_swapc"] = paired_diff(
                    [x[f"loc_hit@{thr}"] for x in P_],
                    [float(np.mean(x[f"null_swapc_loc_hit@{thr}"])) for x in P_],
                    [x["cluster"] for x in P_], upos, dpos)
            res["presence"][f"{a}@{thr}"] = ent
    # --- point-level (in-view image-ramp pairs), cluster = ramp
    p0 = [p for p in pairs if p["arm"] == a0 and p["thr"] == THRESHOLDS[0]]
    upr, dpr = boot_indices([p["ramp"] for p in p0], seed=SEED + 2)
    key = lambda p: (p["image_id"], p["ramp"])  # noqa: E731
    pa = {(p["arm"], p["thr"], key(p)): p for p in pairs}
    order = [key(p) for p in p0]
    cl = [p["ramp"] for p in p0]
    comp = covisible_components(p0)
    clc = [comp[r] for r in cl]
    ucc, dcc = boot_indices(clc, seed=SEED + 9)
    res["counts"]["covisible_components"] = len(ucc)
    for a in arms:
        for thr in THRESHOLDS:
            rows_a = [pa[(a, thr, k)] for k in order]
            ent = {}
            for test in list(NULL_TESTS) + [f"hit_world_{h}" for h in WORLD_HEIGHTS]:
                ent[test] = cluster_rate([r[test] for r in rows_a], cl, upr, dpr)
            if args.n_null:
                for test in NULL_TESTS:
                    ent[f"{test}__null_swap"] = null_draw_rates(rows_a, f"null_swap_{test}")
                    ent[f"{test}__null_mirror"] = float(np.mean(
                        [r[f"null_mirror_{test}"] for r in rows_a]))
                    ent[f"{test}__above_swap"] = above_chance(rows_a, test, cl, upr, dpr)
                    ent[f"{test}__null_swapc"] = null_draw_rates(rows_a, f"null_swapc_{test}")
                    ent[f"{test}__above_swapc"] = above_chance(rows_a, test, cl, upr, dpr,
                                                               "swapc")
                # clustering sensitivity: pairs clustered by co-visible component
                ent["hit_bearing__by_component"] = cluster_rate(
                    [r["hit_bearing"] for r in rows_a], clc, ucc, dcc)
                for kind in ("swap", "swapc"):
                    ent[f"hit_bearing__above_{kind}__by_component"] = above_chance(
                        rows_a, "hit_bearing", clc, ucc, dcc, kind)
            for lo, hi in RANGE_BINS:
                b = f"{lo:g}-{hi:g}"
                sel = [i for i, r in enumerate(rows_a) if r["bin"] == b]
                rb, cb = [rows_a[i] for i in sel], [cl[i] for i in sel]
                ent[f"hit_bearing[{b}]"] = cluster_rate([r["hit_bearing"] for r in rb],
                                                        cb, upr, dpr)
                if args.n_null:
                    ent[f"hit_bearing[{b}]__null_swap"] = null_draw_rates(
                        rb, "null_swap_hit_bearing")
                    ent[f"hit_bearing[{b}]__above_swap"] = above_chance(
                        rb, "hit_bearing", cb, upr, dpr)
                    ent[f"hit_bearing[{b}]__null_swapc"] = null_draw_rates(
                        rb, "null_swapc_hit_bearing")
                    ent[f"hit_bearing[{b}]__above_swapc"] = above_chance(
                        rb, "hit_bearing", cb, upr, dpr, "swapc")
            res["points"][f"{a}@{thr}"] = ent
    # --- paired contrasts vs canvas_level
    ref = "canvas_level" if "canvas_level" in arms else a0
    for a in arms:
        if a == ref:
            continue
        for thr in THRESHOLDS:
            ra = [pa[(a, thr, k)] for k in order]
            rr = [pa[(ref, thr, k)] for k in order]
            ia = {i["image_id"]: i for i in images if i["arm"] == a}
            ir = {i["image_id"]: i for i in images if i["arm"] == ref}
            ent = {
                "point_hit_bearing": paired_diff([x["hit_bearing"] for x in ra],
                                                 [x["hit_bearing"] for x in rr], cl, upr, dpr),
                "presence_recall": paired_diff([ia[i["image_id"]][f"fired@{thr}"] for i in pos],
                                               [ir[i["image_id"]][f"fired@{thr}"] for i in pos],
                                               [i["cluster"] for i in pos], upos, dpos),
                "fire_rate_pool_negative": paired_diff(
                    [ia[i["image_id"]][f"fired@{thr}"] for i in neg],
                    [ir[i["image_id"]][f"fired@{thr}"] for i in neg],
                    [i["cluster"] for i in neg], uneg, dneg)}
            if args.n_null:
                # the arms' floors differ (the stretch fires more), so compare them above
                # their own floors too
                ent["point_hit_bearing_above_swap"] = paired_diff(
                    [x["hit_bearing"] - np.mean(x["null_swap_hit_bearing"]) for x in ra],
                    [x["hit_bearing"] - np.mean(x["null_swap_hit_bearing"]) for x in rr],
                    cl, upr, dpr)
                ent["point_hit_bearing_above_swapc"] = paired_diff(
                    [x["hit_bearing"] - np.mean(x["null_swapc_hit_bearing"]) for x in ra],
                    [x["hit_bearing"] - np.mean(x["null_swapc_hit_bearing"]) for x in rr],
                    cl, upr, dpr)
            res["paired"][f"{a}-{ref}@{thr}"] = ent
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
    # chance floor on the panos: every detection rotated by 90 / 180 / 270 deg
    pb_rot = {sh: pano_bearing_check(shift=sh) for sh in (PANO_NULL_SHIFTS if args.n_null
                                                           else ())}
    for k, x in enumerate(pb):
        x["null_rot"] = [pb_rot[sh][k]["bearing_hit"] for sh in pb_rot]
        assert all(pb_rot[sh][k]["ramp"] == x["ramp"] for sh in pb_rot)
    u, dr = boot_indices([x["ramp"] for x in pb], seed=SEED + 5)
    ent = {"n_captures": len(pb), "n_ramps": len(u),
           "bearing_test": cluster_rate([x["bearing_hit"] for x in pb], [x["ramp"] for x in pb],
                                        u, dr),
           "world_test": cluster_rate([x["world_hit"] for x in pb], [x["ramp"] for x in pb], u, dr),
           "agree": rnd(np.mean([x["bearing_hit"] == x["world_hit"] for x in pb]))}
    if pb_rot:
        ent["bearing_null_rotation"] = {f"{round(sh * 360)}deg": float(np.mean(
            [x["bearing_hit"] for x in pb_rot[sh]])) for sh in pb_rot}
        ent["bearing_above_rotation"] = paired_diff(
            [x["bearing_hit"] for x in pb], [float(np.mean(x["null_rot"])) for x in pb],
            [x["ramp"] for x in pb], u, dr)
    res["pano_reference"]["test_check@0.55"] = ent
    # like-for-like range mix (S2): the pano bearing test on the flat set's ramps, per range
    # bin, re-weighted to the flat pairs' range mix
    flat_ramps = {p["ramp"] for p in p0}
    pbs = [x for x in pb if x["ramp"] in flat_ramps]
    mix = {f"{lo:g}-{hi:g}": sum(p["bin"] == f"{lo:g}-{hi:g}" for p in p0) / len(p0)
           for lo, hi in RANGE_BINS}
    by_bin = {b: [x for x in pbs if x["bin"] == b] for b in mix}
    res["pano_reference"]["range_mix@0.55"] = {
        "flat_pairs_by_bin": {b: sum(p["bin"] == b for p in p0) for b in mix},
        "pano_captures_by_bin": {b: len(v) for b, v in by_bin.items()},
        "flat_median_range": float(np.median([p["range"] for p in p0])),
        "pano_median_range": float(np.median([x["range"] for x in pbs])),
        "pano_bearing_by_bin": {b: float(np.mean([x["bearing_hit"] for x in v]))
                                for b, v in by_bin.items() if v},
        "pano_bearing_unweighted": float(np.mean([x["bearing_hit"] for x in pbs])),
        "pano_bearing_reweighted_to_flat_mix": float(sum(
            mix[b] * np.mean([x["bearing_hit"] for x in v]) for b, v in by_bin.items() if v))}
    # --- flat vs pano, ramp-level paired: per ramp, flat pair-hit rate minus pano capture
    # hit rate, over ramps that have both
    for a in arms:
        for thr in THRESHOLDS:
            fr, frn, frc, pr_ = {}, {}, {}, {}
            for k in order:
                p = pa[(a, thr, k)]
                fr.setdefault(p["ramp"], []).append(p["hit_bearing"])
                if args.n_null:
                    frn.setdefault(p["ramp"], []).append(
                        p["hit_bearing"] - np.mean(p["null_swap_hit_bearing"]))
                    frc.setdefault(p["ramp"], []).append(
                        p["hit_bearing"] - np.mean(p["null_swapc_hit_bearing"]))
            for x in pr:
                if x["same_ramps"]:
                    pr_.setdefault(x["ramp"], []).append(x["world_conf"] >= thr)
            both = sorted(set(fr) & set(pr_))
            diff = [np.mean(fr[r]) - np.mean(pr_[r]) for r in both]
            u, dr = boot_indices(both, seed=SEED + 4)
            res["paired"][f"{a}-pano@{thr}"] = {
                "ramp_mean_hit_diff": cluster_rate(diff, both, u, dr),
                "n_ramps": len(both)}
            if thr == 0.55:
                # like for like: the pano side under the same bearing test
                pbr, pbrn = {}, {}
                for x in pb:
                    if x["ramp"] in fr:
                        pbr.setdefault(x["ramp"], []).append(x["bearing_hit"])
                        if pb_rot:
                            pbrn.setdefault(x["ramp"], []).append(
                                x["bearing_hit"] - np.mean(x["null_rot"]))
                both2 = sorted(set(fr) & set(pbr))
                diff2 = [np.mean(fr[r]) - np.mean(pbr[r]) for r in both2]
                u2, dr2 = boot_indices(both2, seed=SEED + 6)
                ent = {"ramp_mean_hit_diff": cluster_rate(diff2, both2, u2, dr2),
                       "n_ramps": len(both2),
                       "flat_ramp_mean": rnd(np.mean([np.mean(fr[r]) for r in both2])),
                       "pano_ramp_mean": rnd(np.mean([np.mean(pbr[r]) for r in both2]))}
                if args.n_null and pb_rot:
                    # both sides above their own chance floor (flat: swap null; pano:
                    # rotation null), per ramp
                    diff3 = [np.mean(frn[r]) - np.mean(pbrn[r]) for r in both2]
                    ent.update({
                        "above_chance_diff": cluster_rate(diff3, both2, u2, dr2),
                        "flat_above_chance_ramp_mean": rnd(np.mean([np.mean(frn[r])
                                                                    for r in both2])),
                        "pano_above_chance_ramp_mean": rnd(np.mean([np.mean(pbrn[r])
                                                                    for r in both2])),
                        "above_chance_diff_count_matched": cluster_rate(
                            [np.mean(frc[r]) - np.mean(pbrn[r]) for r in both2], both2, u2,
                            dr2),
                        "flat_above_count_matched_ramp_mean": rnd(np.mean(
                            [np.mean(frc[r]) for r in both2]))})
                res["paired"][f"{a}-pano_bearing@{thr}"] = ent
    # --- angular sampling by camera model (S2 / S5)
    by_cam = {}
    rec0 = arm_recs[a0]
    for iid, rec in rec0.items():
        row = by_id[iid]
        by_cam.setdefault(f'{row["make"]} {row["model"]}'.strip(), []).append(
            camera_scale(row, rec["width"], rec["height"]))
    scale = {"canvas_px_per_deg": P.CANVAS_W / 360.0, "by_camera": {}}
    for cam_name, v in sorted(by_cam.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 10:
            continue
        edge = [x["edge_over_centre"] for x in v if x["edge_over_centre"] is not None]
        scale["by_camera"][cam_name] = {
            "n_images": len(v), "hfov_p50": float(np.median([x["hfov"] for x in v])),
            "thumb_centre_px_per_deg_p50": float(np.median([x["centre_px_per_deg"] for x in v])),
            "edge_over_centre_p50": float(np.median(edge)) if edge else None,
            "edge_not_invertible": len(v) - len(edge),
            "stretch_px_per_deg_x_p50": float(np.median([x["stretch_px_per_deg_x"] for x in v])),
            "stretch_px_per_deg_y_p50": float(np.median([x["stretch_px_per_deg_y"] for x in v]))}
    allv = [x for v in by_cam.values() for x in v]
    scale["all"] = {"n_images": len(allv),
                    "stretch_px_per_deg_x_p10_p50_p90": [float(x) for x in np.percentile(
                        [x["stretch_px_per_deg_x"] for x in allv], [10, 50, 90])],
                    "stretch_px_per_deg_y_p10_p50_p90": [float(x) for x in np.percentile(
                        [x["stretch_px_per_deg_y"] for x in allv], [10, 50, 90])]}
    res["angular_scale"] = scale
    # --- geometry checks: detections the distortion model cannot place (N3, B2)
    chk = {}
    for a in arms:
        n_det = n_nan = n_fold = 0
        for iid, rec in arm_recs[a].items():
            row = by_id[iid]
            cam = camera_of(row, rec["width"], rec["height"])
            dets = [d for d in rec["dets"] if d["score"] >= PRIMARY_THR]
            if not dets:
                continue
            n_det += len(dets)
            ray = P.unproject_cam(cam, [d["u"] for d in dets], [d["v"] for d in dets])
            n_nan += int((~np.isfinite(ray[:, 0])).sum())
            M = arm_M(a, pose_of(row))
            if M is not None:
                cr = P.canvas_norm_to_cam_ray(np.array([d["x"] for d in dets]),
                                              np.array([d["y"] for d in dets]), M)
                n_fold += int((~P.in_distortion_domain(cam, cr)).sum())
        chk[a] = {f"dets@{PRIMARY_THR}": n_det, "photo_pixel_not_invertible": n_nan,
                  "canvas_ray_beyond_fold": n_fold if ARMS[a]["kind"] == "canvas" else None}
    res["geometry_checks"] = chk
    res = _round(res)
    write_json(os.path.join(res_dir, "results.json"), res)
    with open(os.path.join(res_dir, "images_scored.csv"), "w", encoding="utf-8", newline="") as f:
        cols = sorted(k for k in images[0] if not k.startswith(("implied_h", "null_")))
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for i in images:
            w.writerow({k: i[k] for k in cols})
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


def fmt_null(e):
    if not e:
        return "n/a"
    return f"{e['mean']:.3f} ({e['p5']:.3f}-{e['p95']:.3f})"


def markdown(res, arms):
    L = ["# Perspective photos (#218): Richmond results", "",
         "Generated by `scripts/analysis/perspective_photos_218.py score`. "
         "95% CIs: cluster bootstrap (images clustered by nearest pool ramp; pairs by ramp), "
         f"{N_REPS} reps. Chance floors: swap null = the image's detections replaced by an "
         f"unrelated positive image's, {res['config']['n_null']} draws, mean (p5-p95 over "
         "draws); count-matched swap null = the same with the donor's number of detections "
         ">= 0.30 in the receiver's bucket (0/1/2/3+); mirror null = candidate ramp bearings mirrored about the camera heading; "
         "pano rotation null = detections rotated 90/180/270 deg. 'Above chance' = real minus "
         "the per-pair swap-null mean, paired cluster bootstrap.", "", "## Counts", ""]
    for k, v in res["counts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Image level", "",
          "| arm @ thr | presence recall | localized recall | localized, swap null | "
          "localized, above chance | localized, above count-matched | "
          "fire rate, pool-negative | dets / image | matched fraction |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for thr in THRESHOLDS:
            e = res["presence"][f"{a}@{thr}"]
            L.append(f"| {a} @ {thr} | {fmt_ci(e['presence_recall'])} | "
                     f"{fmt_ci(e['localized_recall'])} | "
                     f"{fmt_null(e.get('localized_recall_null_swap'))} | "
                     f"{fmt_ci(e['localized_recall_above_swap']) if e.get('localized_recall_above_swap') else 'n/a'} | "
                     f"{fmt_ci(e['localized_recall_above_swapc']) if e.get('localized_recall_above_swapc') else 'n/a'} | "
                     f"{fmt_ci(e['fire_rate_pool_negative'])} | "
                     f"{e['dets_per_image_all']:.2f} | {e['matched_frac_of_dets']:.3f} |")
    L += ["", "## Point hits (in-view image-ramp pairs)", "",
          "| arm @ thr | bearing test | bearing, 10 m lateral | bearing, no height gate | "
          "world 1.5 m | world 2.6 m | "
          + " | ".join(f"bearing {lo:g}-{hi:g} m" for lo, hi in RANGE_BINS) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(RANGE_BINS)]
    for a in arms:
        for thr in THRESHOLDS:
            e = res["points"][f"{a}@{thr}"]
            L.append(f"| {a} @ {thr} | {fmt_ci(e['hit_bearing'])} | "
                     f"{fmt_ci(e['hit_bearing_loose'])} | {fmt_ci(e['hit_bearing_no_hgate'])} | "
                     f"{fmt_ci(e['hit_world_1.5'])} | {fmt_ci(e['hit_world_2.6'])} | "
                     + " | ".join(f"{fmt_ci(e[f'hit_bearing[{lo:g}-{hi:g}]'])} "
                                  f"(n={e[f'hit_bearing[{lo:g}-{hi:g}]'][3]})"
                                  for lo, hi in RANGE_BINS) + " |")
    if res["config"]["n_null"]:
        L += ["", "## Chance floor of the bearing tests (point hits)", "",
              "| arm @ thr | test | real | swap null (p5-p95) | mirror null | above chance "
              "(real - swap null) | count-matched swap null | above count-matched |",
              "|---|---|---|---|---|---|---|---|"]
        for a in arms:
            for thr in THRESHOLDS:
                e = res["points"][f"{a}@{thr}"]
                for t in NULL_TESTS:
                    L.append(f"| {a} @ {thr} | {t} | {e[t][0]:.3f} | "
                             f"{fmt_null(e[f'{t}__null_swap'])} | {e[f'{t}__null_mirror']:.3f} | "
                             f"{fmt_ci(e[f'{t}__above_swap'])} | "
                             f"{fmt_null(e[f'{t}__null_swapc'])} | "
                             f"{fmt_ci(e[f'{t}__above_swapc'])} |")
        L += ["", "Clustering sensitivity (bearing test): pairs clustered by co-visible "
              f"component ({res['counts'].get('covisible_components')} components) instead "
              "of by ramp:", "",
              "| arm @ thr | real | above swap null | above count-matched swap null |",
              "|---|---|---|---|"]
        for a in arms:
            for thr in THRESHOLDS:
                e = res["points"][f"{a}@{thr}"]
                L.append(f"| {a} @ {thr} | {fmt_ci(e['hit_bearing__by_component'])} | "
                         f"{fmt_ci(e['hit_bearing__above_swap__by_component'])} | "
                         f"{fmt_ci(e['hit_bearing__above_swapc__by_component'])} |")
        L += ["", "By range (bearing test):", "",
              "| arm @ thr | " + " | ".join(f"{lo:g}-{hi:g} m: real / swap null / above / "
                                            "count-matched null, above"
                                            for lo, hi in RANGE_BINS) + " |",
              "|---|" + "---|" * len(RANGE_BINS)]
        for a in arms:
            for thr in THRESHOLDS:
                e = res["points"][f"{a}@{thr}"]
                cells = []
                for lo, hi in RANGE_BINS:
                    b = f"hit_bearing[{lo:g}-{hi:g}]"
                    cells.append(f"{e[b][0]:.3f} (n={e[b][3]}) / {fmt_null(e[b + '__null_swap'])}"
                                 f" / {fmt_ci(e[b + '__above_swap'])} / count-matched "
                                 f"{fmt_null(e[b + '__null_swapc'])}, "
                                 f"{fmt_ci(e[b + '__above_swapc'])}")
                L.append(f"| {a} @ {thr} | " + " | ".join(cells) + " |")
    L += ["", "## 360 pano reference (world test, non-source captures, 3-18 m)", ""]
    for scope, e in res["pano_reference"].items():
        if scope.startswith("test_check"):
            L.append(f"- {scope} (same captures, both tests): {e['n_captures']} captures of "
                     f"{e['n_ramps']} ramps; bearing test {fmt_ci(e['bearing_test'])}, world "
                     f"test {fmt_ci(e['world_test'])}, per-capture agreement {e['agree']}")
            if "bearing_null_rotation" in e:
                L.append(f"  - bearing test, rotation null: {e['bearing_null_rotation']}; "
                         f"above chance {fmt_ci(e['bearing_above_rotation'])}")
            continue
        if scope.startswith("range_mix"):
            L.append(f"- {scope}: flat pairs by bin {e['flat_pairs_by_bin']}, pano captures of "
                     f"the same ramps by bin {e['pano_captures_by_bin']} (median range "
                     f"{e['flat_median_range']} vs {e['pano_median_range']} m); pano bearing "
                     f"by bin {e['pano_bearing_by_bin']}; unweighted "
                     f"{e['pano_bearing_unweighted']}, re-weighted to the flat range mix "
                     f"{e['pano_bearing_reweighted_to_flat_mix']}")
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
    L += ["", "## Camera height implied by bearing-matched detections @ 0.3 (range x tan depression)",
          "", "Real matches include chance matches (see the swap null row); a chance match's "
          "depression is arbitrary inside the 0.5-4 m gate.", ""]
    for a, e in res.get("implied_height", {}).items():
        L.append(f"- {a}: n={e['n']}, p10/p50/p90 = {e['p10_p50_p90']} m; "
                 f"{e.get('n_sequences')} sequences, {e.get('sequences_median_below_2.2m')} with "
                 f"a median below 2.2 m; swap null: {e.get('null_swap')}")
        for c, x in (e.get("by_camera") or {}).items():
            L.append(f"  - {c}: {x['n_matches']} matches in {x['n_sequences']} sequences, "
                     f"h p50 {x['h_p50']} m; on positive images {x['n_matches_positive_images']}"
                     f" matches vs {x['null_matches_per_draw']} per swap-null draw (null h p50 "
                     f"{x['null_h_p50']} m)")
    if "angular_scale" in res:
        sc = res["angular_scale"]
        L += ["", "## Angular sampling by camera model (cameras with >= 10 images)", "",
              f"Canvas: {sc['canvas_px_per_deg']} px/deg everywhere. Stretch, all images: "
              f"px/deg across p10/p50/p90 {sc['all']['stretch_px_per_deg_x_p10_p50_p90']}, down "
              f"{sc['all']['stretch_px_per_deg_y_p10_p50_p90']}.", "",
              "| camera | images | HFOV p50 | thumbnail px/deg at centre | edge / centre "
              "sampling (canvas extra downsampling at the side edge) | edge not invertible | "
              "stretch px/deg across | stretch px/deg down |", "|---|---|---|---|---|---|---|---|"]
        for c, x in sc["by_camera"].items():
            L.append(f"| {c} | {x['n_images']} | {x['hfov_p50']} | "
                     f"{x['thumb_centre_px_per_deg_p50']} | {x['edge_over_centre_p50']} | "
                     f"{x['edge_not_invertible']} | {x['stretch_px_per_deg_x_p50']} | "
                     f"{x['stretch_px_per_deg_y_p50']} |")
    if "geometry_checks" in res:
        L += ["", "## Geometry checks", ""]
        for a, e in res["geometry_checks"].items():
            L.append(f"- {a}: {e}")
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
    totals = {"unmatched_total": len(unmatched), "matched_total": len(matched)}
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
    totals.update({"unmatched_rated": len(unmatched), "matched_rated": len(ctrl)})
    return items, totals


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


FP_EXPORT_PREFIX = "richmond_flat_fp__"
FP_ANSWERS = ("yes", "no", "cant_tell")


def fp_reference():
    """What a Richmond verdict file must match: the committed manifest, with its digest
    re-derived from its items (so a hand-edited digest is caught), and this module's
    question, rubric and rules (so a manifest built under an older rubric is caught)."""
    with open(os.path.join(FP_DIR, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    digest = hashlib.sha256("\n".join(
        f"{it['item']} {it['image_id']} {man['image_sha256'][it['image_id']]} {it['crop_box']}"
        for it in man["items"]).encode()).hexdigest()[:16]
    if digest != man["manifest_digest"]:
        raise SystemExit(f"manifest.json: digest {man['manifest_digest']} does not re-derive "
                         f"from its items ({digest})")
    rubric = [{"key": k, "label": lab, "definition": d} for k, lab, d in FP_RUBRIC]
    if (man["question"], man["rubric"], man["rules"]) != (FP_QUESTION, rubric, FP_RULES):
        raise SystemExit("manifest.json: question/rubric/rules differ from this module's")
    return man, {"manifest_digest": digest, "items": [it["item"] for it in man["items"]],
                 "question": FP_QUESTION, "rubric": rubric, "rules": FP_RULES}


def fp_precision(verdicts, man, n_reps=N_REPS, seed=SEED):
    """Precision of the gallery's arm at its threshold from one rater's verdicts.

    The gallery is stratified: ``unmatched_rated`` of the ``unmatched_total`` detections no
    pool ramp claimed, and ``matched_rated`` of the ``matched_total`` claimed ones (the
    blind control). Each stratum's precision is Yes / (Yes + No) (Can't tell and
    unanswered excluded, and counted); the overall precision weights the two by their
    totals, with a stratified bootstrap CI (items resampled within each stratum)."""
    tot = man["totals"]
    strata = {"unmatched": [it for it in man["items"] if it["matched_ramp"] is None],
              "matched": [it for it in man["items"] if it["matched_ramp"] is not None]}
    w = {"unmatched": tot["unmatched_total"], "matched": tot["matched_total"]}
    out, ys = {}, {}
    for k, its in strata.items():
        ans = [((verdicts.get(it["item"]) or {}).get("answer")) for it in its]
        y = np.array([a == "yes" for a in ans if a in ("yes", "no")], dtype=float)
        ys[k] = y
        out[k] = {"n_items": len(its), "yes": int(y.sum()), "no": int(len(y) - y.sum()),
                  "cant_tell": sum(a == "cant_tell" for a in ans),
                  "unanswered": sum(a is None for a in ans),
                  "precision": float(y.mean()) if len(y) else None,
                  "of_total": w[k]}
    if all(len(y) for y in ys.values()):
        est = sum(w[k] * ys[k].mean() for k in ys) / sum(w.values())
        rng = np.random.default_rng(seed)
        reps = [sum(w[k] * rng.choice(ys[k], len(ys[k])).mean() for k in ys) / sum(w.values())
                for _ in range(n_reps)]
        out["weighted"] = [float(est), *[float(x) for x in np.percentile(reps, [2.5, 97.5])],
                           int(sum(len(y) for y in ys.values()))]
    else:
        out["weighted"] = None
    return out


def cmd_rates(args):
    """Richmond flat-photo detection precision from rater exports
    (``benchmark/richmond_flat_fp_218/richmond_flat_fp__<rater>.json``); with two or more,
    also pairwise agreement (Cohen's kappa, Can't tell excluded)."""
    import rating_page_218 as RP
    man, ref = fp_reference()
    files = [RP.load_verdicts(p, ref, FP_EXPORT_PREFIX) for p in args.verdicts]
    res = {"arm": man["arm"], "threshold": man["threshold"], "totals": man["totals"]}
    for v in files:
        res[v["rater"]] = fp_precision(v["verdicts"], man)
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            res[f"agreement:{files[i]['rater']}-{files[j]['rater']}"] = RP.agreement(
                files[i]["verdicts"], files[j]["verdicts"], ref["items"],
                lambda a: a if a in ("yes", "no") else None)
    print(json.dumps(_round(res), indent=1))


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
    i.add_argument("--shard", default=None, help="k/n: run rows[k::n] (parallel processes)")
    i.add_argument("--usage-log", default=None,
                   help="default: the main checkout's analysis_out/usage_log.jsonl; 'none' skips")
    i.add_argument("--concurrent-with", action="append", default=None,
                   help="another job sharing the GPU, for the ledger row (repeatable)")
    rl = sub.add_parser("reledger")
    rl.add_argument("--raw", required=True)
    rl.add_argument("--out", default=None)
    rl.add_argument("--replace-in", default=None)
    rl.add_argument("--concurrent-with", action="append", default=None)
    s = sub.add_parser("score")
    s.add_argument("--arms", default="canvas_level,canvas_sfm,stretch")
    s.add_argument("--dets-dir", default=None, help="default analysis_out/perspective_photos_218")
    s.add_argument("--results-dir", default=None, help="default analysis_out/perspective_photos_218")
    s.add_argument("--n-null", type=int, default=N_NULL,
                   help="swap-null draws for the chance floor (0 skips every null)")
    mg = sub.add_parser("merge")
    mg.add_argument("--arms", default="canvas_level,canvas_sfm,stretch")
    mg.add_argument("--out", default=None)
    g = sub.add_parser("gallery")
    g.add_argument("--images", required=True)
    g.add_argument("--arm", default="canvas_level")
    rt = sub.add_parser("rates")
    rt.add_argument("--verdicts", required=True, nargs="+",
                    help="benchmark/richmond_flat_fp_218/richmond_flat_fp__<rater>.json")
    args = ap.parse_args(argv)
    {"select": cmd_select, "fetch": cmd_fetch, "infer": cmd_infer, "score": cmd_score,
     "gallery": cmd_gallery, "merge": cmd_merge, "reledger": cmd_reledger,
     "rates": cmd_rates}[args.cmd](args)


if __name__ == "__main__":
    main()
