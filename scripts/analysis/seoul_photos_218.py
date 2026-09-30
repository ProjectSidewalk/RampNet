"""RampNet on the Seoul pedestrian-level iPhone photos -- the stretch test of issue #218.

The Seoul Sidewalk Accessibility Image Dataset (Lieu et al., Zenodo
10.5281/zenodo.22699523, CC0) is 514 iPhone photos taken about 1.0 m above the walking
path, looking along it. It has **no curb ramp annotations**, so this script only prepares
the test: it fetches the photos, runs RampNet's canvas and stretch arms on them, and builds
a blind presence gallery for a human rater. **Nothing here rates anything.**

Images are not redistributed. ``fetch`` pulls named members out of the three Zenodo zips
with HTTP range reads (no full download), checks each member's zip CRC, and records its
sha256 in ``seoul/fetched.csv`` the first time. A later fetch with that file present
refuses any image whose sha256 differs.

Camera assumptions (docs/perspective_photos_218.md section 6):
- level horizon (the capture protocol holds the phone level along the path);
- horizontal FOV from the EXIF 35 mm-equivalent focal length when present
  (``2 atan(18 / f35)``), else 70 deg; pinhole (iPhone JPEGs are distortion-corrected
  in camera);
- camera height 1.0 m (the protocol), used only for the range column of each detection.

Subcommands::

    python scripts/analysis/seoul_photos_218.py manifest            # zip directories -> seoul/files.csv
    python scripts/analysis/seoul_photos_218.py fetch --out DIR      # all 514 (or --names a,b)
    python scripts/analysis/seoul_photos_218.py infer --images DIR   # GPU; arms canvas_level, stretch
    python scripts/analysis/seoul_photos_218.py gallery --images DIR # blind presence gallery
    python scripts/analysis/seoul_photos_218.py rates --verdicts FILE  # after rating (not run yet)
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import socket
import sys
import time
import zipfile
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import perspective as P  # noqa: E402
import perspective_photos_218 as PP  # noqa: E402
import rating_page_218 as RP  # noqa: E402

RECORD = "22699523"
API = f"https://zenodo.org/api/records/{RECORD}"
ZIPS = ("imagery_1.zip", "imagery_2.zip", "imagery_3.zip")
OUT = os.path.join(PP.OUT, "seoul")
FILES_CSV = os.path.join(OUT, "files.csv")
FETCHED_CSV = os.path.join(OUT, "fetched.csv")
SUMMARY_CSV = os.path.join(OUT, "summary_attributes.csv")
GALLERY_DIR = os.path.join(REPO, "benchmark", "seoul_presence_218")
GALLERY_REL = "benchmark/seoul_presence_218/gallery.html"
DEFAULT_HFOV = 70.0
CAM_H = 1.0
ARMS = ("canvas_level", "stretch")
DISPLAY_LONG = 1400
RATER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


# --------------------------------------------------------------------------- #
# HTTP range file
# --------------------------------------------------------------------------- #
class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, enough for ``zipfile``."""

    def __init__(self, url, session=None, block=1 << 16):
        import requests
        self.url = url
        self.s = session or requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
        if r.status_code != 206:
            raise SystemExit(f"{url}: server did not honour a range request ({r.status_code})")
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.requests = 1
        self.bytes = 1
        self.block = block
        self._cache = (None, b"")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start, data = self._cache
        if start is not None and start <= self.pos and self.pos + n <= start + len(data):
            out = data[self.pos - start:self.pos - start + n]
        else:
            end = min(self.size, self.pos + max(n, self.block)) - 1
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"},
                                   timeout=120)
                    if r.status_code == 206:
                        break
                except Exception:
                    pass
                time.sleep(2 ** attempt)
            else:
                raise SystemExit(f"range read failed at {self.pos}")
            self.requests += 1
            self.bytes += len(r.content)
            self._cache = (self.pos, r.content)
            out = r.content[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def zip_url(name):
    return f"{API}/files/{name}/content"


# --------------------------------------------------------------------------- #
# manifest / fetch
# --------------------------------------------------------------------------- #
def cmd_manifest(args):
    """The Zenodo file list and md5s, summary_attributes.csv (md5-checked), and each
    zip's member list (name, size, CRC) -> seoul/files.csv."""
    import requests
    s = requests.Session()
    rec = s.get(API, timeout=60).json()
    md5 = {f["key"]: f["checksum"].split(":")[1] for f in rec["files"]}
    data = s.get(zip_url("summary_attributes.csv"), timeout=60).content
    if hashlib.md5(data).hexdigest() != md5["summary_attributes.csv"]:
        raise SystemExit("summary_attributes.csv md5 mismatch")
    os.makedirs(OUT, exist_ok=True)
    with open(SUMMARY_CSV, "wb") as f:
        f.write(data)
    rows = []
    for z in ZIPS:
        hf = HttpRangeFile(zip_url(z), s)
        with zipfile.ZipFile(hf) as zf:
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if info.is_dir() or info.filename.startswith("__MACOSX/") or base.startswith("._"):
                    continue
                rows.append({"member": info.filename, "filename": os.path.basename(info.filename),
                             "zip": z, "zip_md5": md5[z], "file_size": info.file_size,
                             "compress_size": info.compress_size,
                             "crc32": f"{info.CRC:08x}"})
        print(f"{z}: {sum(r['zip'] == z for r in rows)} members, {hf.requests} range requests")
    PP.write_csv(FILES_CSV, rows, ["filename", "member", "zip", "zip_md5", "file_size",
                                   "compress_size", "crc32"])
    print(f"{len(rows)} members -> {FILES_CSV}")


def stem_key(name):
    """summary_attributes.csv names some photos .HEIC / .JPG; the zips hold .jpg. They
    match on the case-folded stem."""
    return os.path.splitext(os.path.basename(name))[0].lower()


def local_name(name):
    """The file name a photo is stored under locally (its stem + .jpg)."""
    return os.path.splitext(os.path.basename(name))[0] + ".jpg"


def summary_rows():
    with open(SUMMARY_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cmd_fetch(args):
    """Named images (default: every row of summary_attributes.csv) -> ``--out``."""
    import requests
    files = {stem_key(r["filename"]): r for r in PP.read_csv(FILES_CSV)}
    want = args.names.split(",") if args.names else [r["filename"] for r in summary_rows()]
    known = ({r["filename"]: r for r in PP.read_csv(FETCHED_CSV)}
             if os.path.exists(FETCHED_CSV) else {})
    os.makedirs(args.out, exist_ok=True)
    s = requests.Session()
    handles, zfs = {}, {}
    t0 = time.time()
    rows, bad, missing = dict(known), [], []
    for k, name in enumerate(want, 1):
        fr = files.get(stem_key(name))
        if fr is None:
            missing.append(name)
            continue
        dst = os.path.join(args.out, local_name(name))
        if not os.path.exists(dst):
            z = fr["zip"]
            if z not in zfs:
                handles[z] = HttpRangeFile(zip_url(z), s, block=1 << 22)
                zfs[z] = zipfile.ZipFile(handles[z])
            data = zfs[z].read(fr["member"])   # zipfile checks the CRC
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
        sha = PP.sha256_file(dst)
        if name in known and known[name]["sha256"] != sha:
            bad.append(name)
            os.remove(dst)
            continue
        rows[name] = {"filename": name, "member": fr["member"], "bytes": os.path.getsize(dst), "sha256": sha,
                      "fetched": known.get(name, {}).get("fetched") or time.strftime("%Y-%m-%d")}
        if k % 25 == 0:
            got = sum(h.bytes for h in handles.values())
            print(f"  {k}/{len(want)} {got / 1e6:.0f} MB {time.time() - t0:.0f} s", flush=True)
    PP.write_csv(FETCHED_CSV, [rows[n] for n in sorted(rows)],
                 ["filename", "member", "bytes", "sha256", "fetched"])
    log = {"step": "seoul-fetch", "date": time.strftime("%Y-%m-%d"), "requested": len(want),
           "missing_from_zips": missing, "sha256_mismatch": bad,
           "range_requests": sum(h.requests for h in handles.values()),
           "bytes": sum(h.bytes for h in handles.values()),
           "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "fetch_log.jsonl"), "a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(log, sort_keys=True) + "\n")
    print(json.dumps(log))
    if bad:
        raise SystemExit(f"{len(bad)} images differ from the recorded sha256: {bad[:5]}")


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def hfov_from_f35(f35):
    """Horizontal FOV (deg) of a 36 mm-wide frame at 35 mm-equivalent focal ``f35``.

    >>> round(hfov_from_f35(26), 1)
    69.4
    """
    return 2 * math.degrees(math.atan(18.0 / float(f35)))


def open_photo(path):
    """(PIL RGB image, EXIF 35 mm focal or None), EXIF orientation applied."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    f35 = None
    try:
        exif = im.getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405)   # FocalLengthIn35mmFilm
    except Exception:
        f35 = None
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im, (float(f35) if f35 else None)


def seoul_camera(width, height, f35=None, hfov_deg=None):
    """Pinhole camera; the HFOV applies to the long side of a 4:3 photo held landscape.
    For a portrait photo the 35 mm-equivalent still refers to the long side."""
    if hfov_deg is None:
        hfov_deg = hfov_from_f35(f35) if f35 else DEFAULT_HFOV
    long_ = max(width, height)
    f = 0.5 / math.tan(math.radians(hfov_deg) / 2)      # normalised by the long side
    return P.Camera(width, height, f), hfov_deg, long_


def det_ground(cam, u, v, h=CAM_H):
    """Level camera at height h: (lateral m, forward m) of the flat-ground point, or None
    at/above the horizon."""
    ray = P.unproject_cam(cam, u, v)
    if ray[1] <= 1e-6:
        return None
    t = h / ray[1]
    return float(ray[0] * t), float(ray[2] * t)


# --------------------------------------------------------------------------- #
# infer
# --------------------------------------------------------------------------- #
def cmd_infer(args):
    import torch
    from rampnet import ledger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = summary_rows()
    if args.limit:
        rows = rows[:args.limit]
    fetched = {r["filename"]: r for r in PP.read_csv(FETCHED_CSV)}
    model = PP.load_model(device)
    host = socket.gethostname().split(".")[0]
    gpus = [torch.cuda.get_device_name(0)] if device.type == "cuda" else []
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {a: [] for a in ARMS}
    t_arm = {a: 0.0 for a in ARMS}
    t_all = time.time()
    for k, r in enumerate(rows, 1):
        name = r["filename"]
        p = os.path.join(args.images, local_name(name))
        if not os.path.exists(p):
            continue
        if args.verify_sha and PP.sha256_file(p) != fetched[name]["sha256"]:
            raise SystemExit(f"sha256 mismatch: {name}")
        img, f35 = open_photo(p)
        cam, hfov, _ = seoul_camera(*img.size, f35=f35)
        for a in ARMS:
            t0 = time.time()
            M = P.cam_from_level(None) if a == "canvas_level" else None
            dets, n_fill, mx = PP.run_arm(model, device, img, cam, a, M)
            t_arm[a] += time.time() - t0
            for d in dets:
                g = det_ground(cam, d["u"], d["v"])
                d["lateral_m"], d["forward_m"] = ((PP.rnd(g[0], 2), PP.rnd(g[1], 2))
                                                  if g else (None, None))
            out[a].append({"filename": name, "width": img.size[0], "height": img.size[1],
                           "f35": f35, "hfov_deg": PP.rnd(hfov, 3), "max_score": PP.rnd(mx, 6),
                           "n_fill_peaks": n_fill, "dets": dets})
        if k % 50 == 0:
            print(f"  {k}/{len(rows)} {time.time() - t_all:.0f} s", flush=True)
    for a in ARMS:
        path = os.path.join(args.out or OUT, f"dets_{a}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            for rec in out[a]:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        PP.write_json(path[:-len(".jsonl")] + ".meta.json",
                      {"arm": a, "n_images": len(out[a]), "floor": PP.FLOOR,
                       "min_distance": PP.MIN_DISTANCE, "host": host, "gpus": gpus,
                       "device": device.type, "fp16": False, "started": started,
                       "elapsed_s": round(t_arm[a], 3), "cam_h_m": CAM_H,
                       "default_hfov_deg": DEFAULT_HFOV, "torch": torch.__version__})
        print(f"{a}: {len(out[a])} images, {t_arm[a]:.0f} s -> {path}")
    if args.usage_log != "none" and not args.limit:
        ul = args.usage_log or os.path.join(ledger.canonical_repo_root(REPO) or REPO,
                                            "analysis_out", "usage_log.jsonl")
        ledger.append_rows(ul, [PP.usage_row(
            f"perspective-218:seoul:{a}", len(out[a]), t_arm[a], started, host, gpus,
            f"seoul_photos_218.py infer, arm {a}: per-image seconds (decode excluded), fp32, "
            f"run wall {time.time() - t_all:.0f} s for both arms",
            script="scripts/analysis/seoul_photos_218.py",
            bundle="analysis_out/perspective_photos_218/seoul/fetched.csv") for a in ARMS])


# --------------------------------------------------------------------------- #
# gallery (blind presence rating)
# --------------------------------------------------------------------------- #
QUESTION = "Is there a curb ramp in this photo?"
RUBRIC = [
    ("ramp", "Curb ramp",
     "A curb ramp you can make out anywhere in the photo: a sloped section that takes the "
     "walkway down through a curb to street level (Project Sidewalk: 'a curb ramp connecting "
     "sidewalk to street'). Count it even if partly hidden or far away, as long as you can "
     "see that it is a ramp. A small lip at the bottom still counts."),
    ("flush", "Flush crossing only",
     "No curb ramp, but a crossing point where the walkway meets the street LEVEL, with no "
     "curb to ramp over (often marked by yellow tactile paving and bollards). Project "
     "Sidewalk's guide gives a level crossing no Curb Ramp label, so the primary score "
     "counts this answer as 'no ramp'; it is recorded separately so the result can also be "
     "scored the other way."),
    ("no", "No",
     "Neither a curb ramp nor a flush crossing is visible. Driveways do not count (Project "
     "Sidewalk labels no Curb Ramp on driveways)."),
    ("cant_tell", "Can't tell",
     "The photo does not let you decide (too dark, blocked, too blurry, or a possible ramp "
     "too far away to judge). Excluded from every rate."),
]
RULES = [
    "Rate the whole photo, not a spot: the page shows no model output on purpose.",
    "If the photo holds both a curb ramp and a flush crossing, answer Curb ramp.",
    "Add a note for anything worth recording, e.g. 'ramp at far left, 20 m' or 'crossing is "
    "a raised table'.",
]
EXPORT_PREFIX = "seoul_presence__"
EXPORT_SUFFIX = ".json"


def manifest_digest(names, sha):
    lines = [f"{n} {sha[n]}" for n in names]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def render_gallery(items, digest):
    """``items``: [{"name", "img" (relative path), "w", "h"}]."""
    return RP.render(items, digest, {
        "title": "Seoul presence check", "h1": "Seoul photos (#218): curb ramp presence",
        "intro": ("<p>Each card is one photo from the Seoul Sidewalk Accessibility Image "
                  "Dataset (iPhone, about 1 m above the walking path). Answer one question per "
                  f"photo: <strong>{html.escape(QUESTION)}</strong> No model output is shown, "
                  "so the answer is independent of RampNet.</p>"),
        "question": QUESTION, "rubric": RUBRIC, "rules": RULES,
        "keys": {"r": "ramp", "f": "flush", "n": "no", "c": "cant_tell"},
        "task": "RampNet #218 Seoul photos, presence: " + QUESTION,
        "export_prefix": EXPORT_PREFIX, "storage_prefix": "seoul218_",
        "gallery_rel": GALLERY_REL,
        "commit_dir": "analysis_out/perspective_photos_218/seoul/"})


def cmd_gallery(args):
    """Display copies (long side DISPLAY_LONG, EXIF-upright) into the gitignored
    ``img/`` beside the page, the page itself, and manifest.json (source sha256s)."""
    from PIL import Image
    fetched = {r["filename"]: r for r in PP.read_csv(FETCHED_CSV)}
    names = [r["filename"] for r in summary_rows() if r["filename"] in fetched]
    img_dir = os.path.join(GALLERY_DIR, "img")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for name in names:
        dst = os.path.join(img_dir, os.path.splitext(name)[0] + ".jpg")
        src = os.path.join(args.images, local_name(name)) if args.images else None
        if not os.path.exists(dst):
            if not src or not os.path.exists(src):
                raise SystemExit(f"{name}: no display copy and no source in --images")
            im, _ = open_photo(src)
            s = DISPLAY_LONG / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
            im.save(dst, quality=85)
        w, h = Image.open(dst).size
        items.append({"name": name, "img": "img/" + os.path.basename(dst), "w": w, "h": h})
    digest = manifest_digest(names, {n: fetched[n]["sha256"] for n in names})
    PP.write_json(os.path.join(GALLERY_DIR, "manifest.json"),
                  {"items": names, "sha256": {n: fetched[n]["sha256"] for n in names},
                   "manifest_digest": digest, "question": QUESTION,
                   "rubric": [{"key": k, "label": lab, "definition": d}
                              for k, lab, d in RUBRIC], "rules": RULES,
                   "display_long_px": DISPLAY_LONG,
                   "note": "img/ is not committed (images are not redistributed); "
                           "`seoul_photos_218.py fetch` then `gallery` rebuilds it"})
    with open(os.path.join(GALLERY_DIR, "gallery.html"), "w", encoding="utf-8", newline="") as f:
        f.write(render_gallery(items, digest))
    print(f"{len(items)} cards, digest {digest} -> {GALLERY_DIR}/gallery.html")


# --------------------------------------------------------------------------- #
# rates (for after the rating pass; not run in #218's first PR)
# --------------------------------------------------------------------------- #
def presence_label(answer, flush_counts=False):
    """Rater answer -> 1 / 0 / None (excluded)."""
    if answer == "ramp":
        return 1
    if answer == "flush":
        return 1 if flush_counts else 0
    if answer == "no":
        return 0
    return None


def cmd_rates(args):
    v = json.load(open(args.verdicts, encoding="utf-8"))
    man = json.load(open(os.path.join(GALLERY_DIR, "manifest.json"), encoding="utf-8"))
    if v["manifest_digest"] != man["manifest_digest"]:
        raise SystemExit("verdicts were made on a different image set")
    res = {}
    for a in ARMS:
        recs = {}
        with open(os.path.join(OUT, f"dets_{a}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[r["filename"]] = r
        for flush in (False, True):
            for thr in PP.THRESHOLDS:
                tp = fp = fn = tn = 0
                for name, ans in v["verdicts"].items():
                    y = presence_label(ans.get("answer"), flush)
                    if y is None or name not in recs:
                        continue
                    pred = recs[name]["max_score"] >= thr
                    tp += y and pred
                    fp += (not y) and pred
                    fn += y and not pred
                    tn += (not y) and not pred
                res[f"{a}@{thr}{'+flush' if flush else ''}"] = {
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": PP.rnd(tp / (tp + fp)) if tp + fp else None,
                    "recall": PP.rnd(tp / (tp + fn)) if tp + fn else None}
    print(json.dumps(res, indent=1))


def cmd_summary(args):
    """Unscored description of the model's output on Seoul (no labels exist yet):
    per arm, the share of photos with a detection >= each threshold, detections per photo,
    the HFOV used, and the canvas fill peaks -> seoul/summary.json."""
    out = {}
    for a in ARMS:
        recs = []
        with open(os.path.join(OUT, f"dets_{a}.jsonl"), encoding="utf-8") as f:
            recs = [json.loads(x) for x in f if x.strip()]
        e = {"n_images": len(recs),
             "hfov_deg_p10_p50_p90": [PP.rnd(x) for x in np.percentile(
                 [r["hfov_deg"] for r in recs], [10, 50, 90])],
             "n_with_exif_f35": sum(r["f35"] is not None for r in recs),
             "fill_peaks": sum(r["n_fill_peaks"] for r in recs)}
        for thr in PP.THRESHOLDS:
            e[f"share_fired@{thr}"] = PP.rnd(np.mean([r["max_score"] >= thr for r in recs]))
            e[f"dets_per_image@{thr}"] = PP.rnd(np.mean(
                [sum(d["score"] >= thr for d in r["dets"]) for r in recs]))
            fw = [d["forward_m"] for r in recs for d in r["dets"]
                  if d["score"] >= thr and d["forward_m"] is not None]
            e[f"forward_m_at_1m_p10_p50_p90@{thr}"] = (
                [PP.rnd(x) for x in np.percentile(fw, [10, 50, 90])] if fw else None)
        out[a] = e
    PP.write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--names", default="")
    i = sub.add_parser("infer")
    i.add_argument("--images", required=True)
    i.add_argument("--out", default=None)
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--verify-sha", action="store_true")
    i.add_argument("--usage-log", default=None)
    g = sub.add_parser("gallery")
    g.add_argument("--images", default=None)
    sub.add_parser("summary")
    r = sub.add_parser("rates")
    r.add_argument("--verdicts", required=True)
    args = ap.parse_args(argv)
    {"manifest": cmd_manifest, "fetch": cmd_fetch, "infer": cmd_infer, "gallery": cmd_gallery,
     "rates": cmd_rates, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
