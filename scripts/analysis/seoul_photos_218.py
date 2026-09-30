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
    radios = lambda key: "".join(  # noqa: E731
        f'<label><input type="radio" name="v_{key}" value="{k}"> {html.escape(lab)}</label>'
        for k, lab, _ in RUBRIC)
    cards = []
    for n, it in enumerate(items, 1):
        key = re.sub(r"[^A-Za-z0-9_]", "_", it["name"])
        cards.append(
            f'<section class="card" data-uid="{html.escape(it["name"])}" aria-labelledby="h_{key}">'
            f'<h2 id="h_{key}">{n}. {html.escape(it["name"])}</h2>'
            f'<img src="{html.escape(it["img"])}" width="{it["w"]}" height="{it["h"]}" '
            f'loading="lazy" alt="Seoul sidewalk photo {html.escape(it["name"])}">'
            f'<fieldset><legend>{html.escape(QUESTION)}</legend><div class="opts">'
            f'{radios(key)}</div><label class="note">Note (optional) '
            f'<textarea rows="2" name="n_{key}"></textarea></label></fieldset></section>')
    rubric_html = "".join(f"<dt>{html.escape(lab)}</dt><dd>{html.escape(d)}</dd>"
                          for _, lab, d in RUBRIC)
    rules_html = "".join(f"<li>{html.escape(x)}</li>" for x in RULES)
    meta = json.dumps({"question": QUESTION,
                       "rubric": [{"key": k, "label": lab, "definition": d}
                                  for k, lab, d in RUBRIC],
                       "rules": RULES, "items": [it["name"] for it in items],
                       "manifest_digest": digest, "gallery": GALLERY_REL})
    meta_js = meta.replace("</", "<" + chr(92) + "/")
    keys = {"r": "ramp", "f": "flush", "n": "no", "c": "cant_tell"}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seoul presence check</title>
<style>
:root {{ color-scheme:light dark; --bg:#ffffff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --focus:#0969da; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --focus:#4493f8; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 64px; max-width:1100px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin:0 0 8px; }}
.muted {{ color:var(--muted); font-size:13px; }}
.intro {{ max-width:75ch; }}
.card {{ border-top:1px solid var(--line); padding:14px 0; }}
.card.done h2::after {{ content:" \\2713"; color:var(--muted); }}
img {{ max-width:100%; height:auto; display:block; }}
fieldset {{ border:1px solid var(--line); border-radius:6px; background:var(--panel); margin:8px 0 0; padding:8px 10px; }}
legend {{ font-weight:600; padding:0 4px; }}
.opts {{ display:flex; flex-wrap:wrap; gap:4px 18px; }}
.opts label {{ cursor:pointer; padding:2px 0; }}
.note {{ display:block; margin-top:6px; font-size:13px; color:var(--muted); }}
textarea {{ display:block; width:100%; box-sizing:border-box; font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; }}
:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
button {{ font:inherit; padding:6px 12px; }}
input[type=text] {{ font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:4px 6px; }}
dt {{ font-weight:600; }} dd {{ margin:0 0 6px 16px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }}
</style></head><body>
<h1>Seoul photos (#218): curb ramp presence</h1>
<div class="intro">
<p>Each card is one photo from the Seoul Sidewalk Accessibility Image Dataset (iPhone, about
1 m above the walking path). Answer one question per photo: <strong>{html.escape(QUESTION)}</strong>
No model output is shown, so the answer is independent of RampNet.</p>
</div>
<dl>{rubric_html}</dl>
<ul>{rules_html}</ul>
<p class="muted">Enter your rater id first (lower-case letters, digits, "_" or "-"; or
<code>?rater=yourid</code> in the address). Answers are saved in this browser per rater as you
go; Export writes <code>{EXPORT_PREFIX}&lt;rater&gt;{EXPORT_SUFFIX}</code>, to be committed under
<code>analysis_out/perspective_photos_218/seoul/</code>. Keys: R, F, N or C answer the card that
has focus or else the topmost card on screen (not inside a text field).</p>
<div class="bar"><label for="rater">Rater id</label>
<input id="rater" type="text" size="10" autocomplete="off" spellcheck="false" aria-describedby="msg">
<button type="button" id="export">Export verdicts JSON</button>
<button type="button" id="next">Next unanswered</button>
<span id="count" aria-live="polite"></span> <span id="msg" role="status"></span></div>
{"".join(cards)}
<script id="meta" type="application/json">{meta_js}</script>
<script>
const META = JSON.parse(document.getElementById('meta').textContent);
const RATER_RE = new RegExp({json.dumps(RATER_RE.pattern)});
const LAST_RATER = "seoul218_last_rater";
function getItem(k) {{ try {{ return localStorage.getItem(k); }} catch (e) {{ return null; }} }}
function setItem(k, v) {{ try {{ localStorage.setItem(k, v); }} catch (e) {{}} }}
const CARDS = [...document.querySelectorAll('.card')];
const raterInput = document.getElementById('rater');
let rater = (new URLSearchParams(location.search).get('rater') || getItem(LAST_RATER) || "").trim().toLowerCase();
if (!RATER_RE.test(rater)) rater = "";
raterInput.value = rater;
let saved = {{}};
function storageKey() {{ return "seoul218_" + META.manifest_digest + "__" + rater; }}
function load() {{ saved = {{}}; if (!rater) return; try {{ saved = JSON.parse(getItem(storageKey()) || "{{}}") || {{}}; }} catch (e) {{ saved = {{}}; }} }}
function persist() {{ if (rater) setItem(storageKey(), JSON.stringify(saved)); }}
function say(t) {{ document.getElementById('msg').textContent = t; }}
function needRater() {{ if (rater) return false; say("Enter your rater id first."); raterInput.focus(); return true; }}
function answered() {{ return META.items.filter(u => saved[u] && saved[u].answer).length; }}
function update() {{
  document.getElementById('count').textContent = (rater ? rater + ": " : "") + answered() + " of " + META.items.length + " answered";
  CARDS.forEach(c => c.classList.toggle('done', !!(saved[c.dataset.uid] || {{}}).answer));
}}
function render() {{
  CARDS.forEach(card => {{
    const cur = saved[card.dataset.uid] || {{}};
    card.querySelectorAll('input[type=radio]').forEach(inp => {{ inp.checked = cur.answer === inp.value; }});
    card.querySelector('textarea').value = cur.note || "";
  }});
  update();
}}
raterInput.addEventListener('change', () => {{
  const v = raterInput.value.trim().toLowerCase();
  if (!RATER_RE.test(v)) {{ raterInput.value = rater; say("Rater id " + JSON.stringify(v) + " refused: use lower-case letters, digits, _ or -, up to 32 characters."); return; }}
  rater = v; setItem(LAST_RATER, v); load(); render(); say("Rating as " + v + ".");
}});
CARDS.forEach(card => {{
  const uid = card.dataset.uid;
  card.querySelectorAll('input[type=radio]').forEach(inp => {{
    inp.addEventListener('change', () => {{
      if (needRater()) {{ inp.checked = false; return; }}
      saved[uid] = Object.assign(saved[uid] || {{}}, {{answer: inp.value}}); persist(); update();
    }});
  }});
  const ta = card.querySelector('textarea');
  ta.addEventListener('input', () => {{
    if (needRater()) {{ ta.value = ""; return; }}
    saved[uid] = Object.assign(saved[uid] || {{}}, {{note: ta.value}}); persist();
  }});
}});
function topCard() {{
  const barBottom = document.querySelector('.bar').getBoundingClientRect().bottom;
  return CARDS.find(c => c.getBoundingClientRect().bottom > barBottom + 40);
}}
document.addEventListener('keydown', ev => {{
  const t = ev.target;
  if (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type === 'text')) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = {json.dumps(keys)}[(ev.key || "").toLowerCase()];
  if (!k) return;
  const card = (t.closest && t.closest('.card')) || topCard();
  if (!card) return;
  ev.preventDefault();
  if (needRater()) return;
  const inp = card.querySelector('input[value="' + k + '"]');
  inp.checked = true; inp.focus(); inp.dispatchEvent(new Event('change'));
}});
load(); render();
document.getElementById('next').addEventListener('click', () => {{
  const card = CARDS.find(c => !(saved[c.dataset.uid] || {{}}).answer);
  if (card) {{ card.scrollIntoView({{block: 'start'}}); card.querySelector('input[type=radio]').focus({{preventScroll: true}}); }}
}});
document.getElementById('export').addEventListener('click', () => {{
  if (needRater()) return;
  const verdicts = {{}};
  META.items.forEach(u => {{
    const v = saved[u];
    if (!v || (!v.answer && !(v.note || "").trim())) return;
    verdicts[u] = {{answer: v.answer || null, note: (v.note || "").trim()}};
  }});
  const out = {{task: "RampNet #218 Seoul photos, presence: " + META.question,
    question: META.question, rubric: META.rubric, rules: META.rules, rater: rater,
    items: META.items, manifest_digest: META.manifest_digest, n_items: META.items.length,
    n_answered: answered(), gallery: META.gallery, exported_at: new Date().toISOString(),
    verdicts: verdicts}};
  const blob = new Blob([JSON.stringify(out, null, 1) + "\\n"], {{type: "application/json"}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "{EXPORT_PREFIX}" + rater + "{EXPORT_SUFFIX}";
  a.click();
  say("Exported {EXPORT_PREFIX}" + rater + "{EXPORT_SUFFIX}.");
}});
</script></body></html>
"""


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
    r = sub.add_parser("rates")
    r.add_argument("--verdicts", required=True)
    args = ap.parse_args(argv)
    {"manifest": cmd_manifest, "fetch": cmd_fetch, "infer": cmd_infer, "gallery": cmd_gallery,
     "rates": cmd_rates}[args.cmd](args)


if __name__ == "__main__":
    main()
