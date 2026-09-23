"""Cut label crops from the makelab2 pano store (#86, RampNet 2.0 plan item 2b).

Only Project Sidewalk labels placed since 2023-10-12 have a production crop. This cuts one
for any ``(city, label_id, field of view)`` from the equirectangular pano archive, as a true
perspective view (``rampnet/crops.py``). Geometry convention, validation and caveats:
``docs/crop_cutter.md``.

Inputs
  --labels   a CSV with ``city,label_id`` columns (plus, optionally, the label geometry
             columns of ``/v3/api/rawLabels``), a text file of ``city:label_id`` lines, or an
             inline comma-separated list ``seattle-wa:9,seattle-wa:10``.
  --raw-cache  where to look up geometry the labels file does not carry: a directory of
             ``<city>__rawLabels__<LabelType>.csv`` as written by
             ``scripts/analysis/ps_supervision_audit.py fetch`` (PR #175), i.e. the
             deployment's ``/v3/api/rawLabels?filetype=csv`` response.
  --store    the pano store root, laid out ``<city_dir>/<pano_id[:2]>/<pano_id>.jpg``. On
             makelab2 that is /projects/makeabilitylab/sidewalk_panos/Panoramas. It is an
             unpublished local input; the script never writes to it.

Outputs
  <out>/<city>__<label_id>__<tag>.jpg, where <tag> is ``fov<deg>`` (label-centred view with
  that horizontal field of view), ``viewport`` (the labeler's own view, the HF
  sidewalk-tagger-ai-validated framing), with ``_eq`` appended for ``--projection equirect``
  and ``_tilt<conv>`` for a tilt-corrected render; and one manifest JSONL row per attempt
  (the latest row for a crop wins).

Resumable: a crop whose latest manifest row is ``ok`` and whose file exists is skipped; a
``missing_pano`` is re-checked and only re-logged if its status changes, so re-running over
an unchanged store appends nothing. Exit status is 1 only on real errors (``error``), never on
``missing_pano``/``no_geometry``/``out_of_frame``/``bad_aspect``.

Example (makelab2)::

    python3 scripts/crop_cutter.py --labels docs/data/crop_cutter/validation_sample.csv \\
        --store /projects/makeabilitylab/sidewalk_panos/Panoramas \\
        --fov viewport --fov 30 --fov 60 --fov 90 --size 1440 \\
        --out /homes/gws/jonf/nobackup/crop_cutter/validation --workers 8
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rampnet import crops  # noqa: E402

CROP_CUTTER_VERSION = "1"

#: Audit city ids whose store directory has a different name. Anything else resolves to the
#: directory of the same name. Checked against the 58 directories of the makelab2 store on
#: 2026-09-22 (``docs/crop_cutter.md``). Override with ``--city-dir city=dir``.
STORE_CITY_DIRS = {
    "columbia": "columbia-sc",
    "kaohsiung": "kaohsiung-tw",
    "keelung": "keelung-tw",
    "la": "la-ca",
    "new-taipei": "new-taipei-tw",
    "taichung": "taichung-tw",
    "tainan": "tainan-tw",
    "walla-walla": "walla-walla-wa",
    "west-chester": "west-chester-pa",
}

GEOMETRY_COLUMNS = ("pano_id", "pano_x", "pano_y", "pano_width", "pano_height", "heading", "pitch",
                    "zoom", "camera_heading", "camera_pitch", "camera_roll", "canvas_x", "canvas_y")

STATUSES = ("ok", "missing_pano", "no_geometry", "out_of_frame", "bad_aspect", "error")


# ----------------------------------------------------------------------------- inputs

def _f(v):
    """Float or None for a CSV cell."""
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    return float(v)


def read_label_list(spec):
    """``[(city, label_id, row_or_None)]`` from a CSV, a city:label_id file, or an inline list."""
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8", newline="") as fh:
            head = fh.readline()
            fh.seek(0)
            if "," in head and "city" in head and "label_id" in head:
                return [(r["city"], int(float(r["label_id"])), r) for r in csv.DictReader(fh)]
            items = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    else:
        items = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for it in items:
        city, _, lid = it.rpartition(":")
        if not city:
            raise SystemExit(f"--labels entry {it!r} is not city:label_id")
        out.append((city, int(lid), None))
    return out


def has_geometry(row):
    return row is not None and all(k in row for k in ("pano_id", "pano_x", "pano_y", "pano_width", "pano_height"))


def lookup_geometry(labels, raw_cache):
    """Fill rows lacking geometry from ``<raw_cache>/<city>__rawLabels__*.csv``."""
    need = defaultdict(set)
    for city, lid, row in labels:
        if not has_geometry(row):
            need[city].add(lid)
    found = {}
    for city, ids in need.items():
        if not raw_cache:
            continue
        for path in sorted(glob.glob(os.path.join(raw_cache, f"{city}__rawLabels__*.csv"))):
            with open(path, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        lid = int(r["label_id"])
                    except (KeyError, ValueError):
                        continue
                    if lid in ids:
                        found[(city, lid)] = r
    return [(c, l, row if has_geometry(row) else found.get((c, l))) for c, l, row in labels]


def store_city_dir(city, overrides):
    return overrides.get(city) or STORE_CITY_DIRS.get(city) or city


def store_relpath(city_dir, pano_id):
    return f"{city_dir}/{pano_id[:2]}/{pano_id}.jpg"


# ----------------------------------------------------------------------------- jobs

def fov_tag(fov, projection, tilt):
    if fov == "viewport":
        tag = "viewport"
    else:
        v = float(fov)
        tag = "fov" + (str(int(v)) if v == int(v) else f"{v:g}".replace(".", "p"))
    if projection == "equirect":
        tag += "_eq"
    if tilt not in (None, "none"):
        tag += f"_tilt{tilt}"
    return tag


def crop_name(city, label_id, tag):
    return f"{city}__{label_id}__{tag}.jpg"


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def make_job(city, label_id, row, fov, args):
    tag = fov_tag(fov, args.projection, args.tilt)
    job = {"city": city, "label_id": int(label_id), "tag": tag, "fov": fov,
           "name": crop_name(city, label_id, tag)}
    if row is None or not has_geometry(row) or not str(row.get("pano_id") or "").strip():
        job["status"] = "no_geometry"
        return job
    g = {k: row.get(k) for k in GEOMETRY_COLUMNS}
    job["pano_id"] = str(g["pano_id"]).strip()
    for k in GEOMETRY_COLUMNS[1:]:
        try:
            job[k] = _f(g[k])
        except ValueError:
            job[k] = None
    return job


def job_view(job, args):
    if job["fov"] == "viewport":
        if any(job.get(k) is None for k in ("heading", "pitch", "zoom", "camera_heading")):
            return None, "no_geometry"
        return crops.viewport_view(job["heading"], job["pitch"], job["zoom"], job["camera_heading"],
                                   args.size, args.aspect), None
    if any(job.get(k) is None for k in ("pano_x", "pano_y", "pano_width", "pano_height")) \
            or not job["pano_width"] or not job["pano_height"]:
        return None, "no_geometry"
    if not 0 <= job["pano_y"] < job["pano_height"]:
        return None, "out_of_frame"
    return crops.centered_view(job["pano_x"], job["pano_y"], job["pano_width"], job["pano_height"],
                               float(job["fov"]), args.size, args.aspect), None


# ----------------------------------------------------------------------------- worker

def _process_pano(payload):
    """Cut every job on one pano. Returns manifest rows. Never raises."""
    store_path, rel, jobs, cfg = payload
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    class A:  # a tiny args shim so job_view works in a worker
        size, aspect = cfg["size"], cfg["aspect"]

    rows = []

    def base(job, status, **kw):
        r = {"city": job["city"], "label_id": job["label_id"], "tag": job["tag"], "name": job["name"],
             "status": status, "pano_id": job.get("pano_id"), "store_path": rel,
             "pano_x": _r(job.get("pano_x"), 3), "pano_y": _r(job.get("pano_y"), 3),
             "pano_width": _r(job.get("pano_width"), 1), "pano_height": _r(job.get("pano_height"), 1),
             "crop_cutter_version": CROP_CUTTER_VERSION}
        r.update(kw)
        return r

    if not os.path.isfile(store_path):
        return [base(j, "missing_pano") for j in jobs]
    try:
        im = Image.open(store_path)
        full_w, full_h = im.size
        if abs(full_w / float(full_h) - 2.0) > 0.01:
            return [base(j, "bad_aspect", store_width=full_w, store_height=full_h) for j in jobs]
        views = []
        for j in jobs:
            v, why = job_view(j, A)
            views.append((j, v, why))
        live = [(j, v) for j, v, why in views if v is not None]
        for j, v, why in views:
            if v is None:
                rows.append(base(j, why, store_width=full_w, store_height=full_h))
        if not live:
            return rows
        kmin = min(crops.reduction_factor(full_w, v) for _, v in live)
        if kmin > 1 and cfg["draft"]:
            im.draft("RGB", (full_w // kmin, full_h // kmin))
        dec = im.convert("RGB")
        dec_w, dec_h = dec.size
        reduced = {}
        for j, v in live:
            k = crops.reduction_factor(dec_w, v)
            if k not in reduced:
                reduced[k] = np.asarray(dec.reduce(k) if k > 1 else dec)
            src = reduced[k]
            tilt = crops.tilt_matrix(j.get("camera_pitch"), j.get("camera_roll"), cfg["tilt"])
            shifted = None
            if cfg["projection"] == "equirect":
                if j["fov"] == "viewport":
                    rows.append(base(j, "error", error="viewport is gnomonic-only"))
                    continue
                arr, shifted, label_px = crops.equirect_window(
                    src, j["pano_x"], j["pano_y"], j["pano_width"], j["pano_height"], float(j["fov"]),
                    cfg["size"], cfg["aspect"])
            else:
                arr = crops.render_view(src, v, tilt)
                if j["fov"] == "viewport":
                    cx, cy = j.get("canvas_x"), j.get("canvas_y")
                    label_px = crops.label_pixel_in_viewport(np.nan if cx is None else cx,
                                                             np.nan if cy is None else cy, v)
                else:
                    label_px = (v.width / 2.0, v.height / 2.0)
            data = crops.encode_jpeg(arr, cfg["quality"])
            out_path = os.path.join(cfg["out"], j["name"])
            tmp = out_path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, out_path)
            rows.append(base(
                j, "ok", store_width=full_w, store_height=full_h,
                dims_match=bool(j.get("pano_width") == full_w and j.get("pano_height") == full_h),
                projection=cfg["projection"], tilt=cfg["tilt"],
                camera_pitch=_r(j.get("camera_pitch"), 4), camera_roll=_r(j.get("camera_roll"), 4),
                yaw_deg=_r(v.yaw_deg), pitch_deg=_r(v.pitch_deg), fov_h_deg=_r(v.fov_h_deg),
                fov_v_deg=_r(v.fov_v_deg), width=v.width, height=v.height,
                decode_width=dec_w, reduce=int(k), jpeg_quality=cfg["quality"],
                label_px=[_r(label_px[0], 3), _r(label_px[1], 3)],
                equirect_shift_px=shifted,
                bytes=len(data), sha256=crops.sha256_hex(data)))
        return rows
    except Exception as e:  # a corrupt pano or a failed write: counted, never fatal to the run
        return rows + [base(j, "error", error=f"{type(e).__name__}: {e}") for j in jobs
                       if not any(r["name"] == j["name"] for r in rows)]


# ----------------------------------------------------------------------------- manifest

def read_manifest(path):
    latest = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    latest[r["name"]] = r
    return latest


def dump_row(r):
    return json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--raw-cache", help="dir of <city>__rawLabels__*.csv for geometry lookup")
    ap.add_argument("--store", required=True, help="pano store root (<city_dir>/<id[:2]>/<id>.jpg)")
    ap.add_argument("--fov", action="append", required=True,
                    help="horizontal FOV in degrees (label-centred), or 'viewport'. Repeatable.")
    ap.add_argument("--size", type=int, default=crops.DEFAULT_WIDTH, help="output width px (default 1440)")
    ap.add_argument("--aspect", type=float, default=crops.DEFAULT_ASPECT, help="width/height (default 1.5)")
    ap.add_argument("--projection", choices=("gnomonic", "equirect"), default="gnomonic")
    ap.add_argument("--tilt", choices=("none",) + tuple(crops.TILT_CONVENTIONS), default="none",
                    help="rotate by camera_pitch/camera_roll under this sign convention (default none)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", help="manifest JSONL (default <out>/manifest.jsonl)")
    ap.add_argument("--summary", help="also write the run summary JSON here")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-draft", dest="draft", action="store_false",
                    help="decode full-resolution JPEGs even when a DCT-scaled decode would do")
    ap.add_argument("--city-dir", action="append", default=[], help="city=store_dir override")
    ap.add_argument("--limit", type=int, help="only the first N labels (smoke tests)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    for f in args.fov:
        if f != "viewport":
            float(f)
    if args.projection == "equirect" and "viewport" in args.fov:
        raise SystemExit("--projection equirect cannot cut a viewport crop")
    os.makedirs(args.out, exist_ok=True)
    manifest = args.manifest or os.path.join(args.out, "manifest.jsonl")
    overrides = dict(s.split("=", 1) for s in args.city_dir)
    t0 = time.time()

    labels = read_label_list(args.labels)
    if args.limit:
        labels = labels[: args.limit]
    labels = lookup_geometry(labels, args.raw_cache)
    latest = read_manifest(manifest)

    cfg = {"size": args.size, "aspect": args.aspect, "projection": args.projection, "tilt": args.tilt,
           "quality": args.quality, "out": args.out, "draft": args.draft}
    by_pano = defaultdict(list)
    immediate = []
    skipped = 0
    seen = set()
    for city, lid, row in labels:
        for fov in args.fov:
            j = make_job(city, lid, row, fov, args)
            if j["name"] in seen:
                continue
            seen.add(j["name"])
            prev = latest.get(j["name"])
            if prev and prev["status"] == "ok" and os.path.isfile(os.path.join(args.out, j["name"])):
                skipped += 1
                continue
            if j.get("status") == "no_geometry":
                immediate.append({"city": city, "label_id": int(lid), "tag": j["tag"], "name": j["name"],
                                  "status": "no_geometry", "crop_cutter_version": CROP_CUTTER_VERSION})
                continue
            rel = store_relpath(store_city_dir(city, overrides), j["pano_id"])
            by_pano[rel].append(j)

    payloads = [(os.path.join(args.store, rel), rel, js, cfg) for rel, js in by_pano.items()]
    counts = Counter()
    n_new = 0
    with open(manifest, "a", encoding="utf-8", newline="") as mf:
        def emit(rows):
            nonlocal n_new
            for r in rows:
                counts[r["status"]] += 1
                prev = latest.get(r["name"])
                if prev and prev["status"] == r["status"] and r["status"] != "ok" and r["status"] != "error":
                    continue  # unchanged non-ok status: keep the manifest append-only and idempotent
                mf.write(dump_row(r))
                n_new += 1
            mf.flush()

        emit(immediate)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for rows in ex.map(_process_pano, payloads, chunksize=1):
                    emit(rows)
        else:
            for p in payloads:
                emit(_process_pano(p))

    wall = time.time() - t0
    from PIL import Image
    summary = {"labels": len(labels), "crops_requested": len(seen), "skipped_existing": skipped,
               "panos_opened": len(payloads), "status": {s: counts.get(s, 0) for s in STATUSES},
               "manifest_rows_appended": n_new, "wall_s": round(wall, 2),
               "crops_per_s": round(counts.get("ok", 0) / wall, 3) if wall > 0 else None,
               "workers": args.workers, "fov": args.fov, "size": args.size, "aspect": _r(args.aspect),
               "projection": args.projection, "tilt": args.tilt, "host": platform.node(),
               "python": platform.python_version(), "numpy": np.__version__,
               "pillow": Image.__version__, "crop_cutter_version": CROP_CUTTER_VERSION,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(summary, indent=1, sort_keys=True))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
