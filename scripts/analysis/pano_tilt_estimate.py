"""Image-based camera tilt (pitch, roll) for equirectangular panos: validate on GSV, run on Lyon.

Why (#159): some Lyon review panos are visibly angled and their metadata cannot say which. The
estimator is ``rampnet/pano_tilt.py`` (the zenith vanishing point of near-vertical line
segments, CPU only). It is first checked against Google's measured pose on the GSV benchmark
splits, and only then run on Lyon. ``docs/pano_tilt_estimate.md`` is the write-up.

Conventions are #113's (``rampnet/stage1_geometry.py``): pitch > 0 is nose down, roll > 0 is left
side up, and the gravity horizon at bearing ``b = x / w * 360 - 180`` sits
``T(b) = pitch cos b + roll sin b`` degrees above the image's middle row.

Subcommands, in order:

    # the stable list of GSV validation panos (split, pano_id, dev|test)
    python scripts/analysis/pano_tilt_estimate.py ids --write

    # estimate every listed pano and compare with Google's pose (needs benchmark/<split>/panos/)
    python scripts/analysis/pano_tilt_estimate.py validate --panos-root benchmark --workers 12

    # the Lyon review bundle's 200 panos, read-only from wherever they are staged
    python scripts/analysis/pano_tilt_estimate.py lyon --panos-dir <dir with the 200 jpgs>

    # re-derive both summaries from the committed per-pano rows (no images needed)
    python scripts/analysis/pano_tilt_estimate.py summarize           # compare, exit 1 on drift
    python scripts/analysis/pano_tilt_estimate.py summarize --write

    # visual check: one view unlevelled, levelled by the record's pose (Google's on GSV) and
    # levelled by the estimate, plumb lines every 40 px (writes a JPEG wherever --out says;
    # nothing is committed). --panos-root holds <split>/panos/<pano_id>.jpg
    python scripts/analysis/pano_tilt_estimate.py inspect --split gainesville \\
        --pano-id DSID5BsZ19qz5BuvC_ZDNA --view 6 --out inspect.jpg

``validate`` and ``lyon`` write per-pano rows; every float is rounded and every file is LF with
sorted keys. LSD's output can differ slightly between OpenCV builds, so ``validate`` and
``lyon`` reproduce the rows closely, not byte for byte, on another machine (the doc says how
closely); ``summarize`` is exact given the rows.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet import pano_tilt as pt  # noqa: E402
from rampnet import stage1_geometry as sg  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "pano_tilt")
IDS = os.path.join(OUT_DIR, "validation_ids.tsv")
VAL_ROWS = os.path.join(OUT_DIR, "validation_gsv.json")
VAL_SUMMARY = os.path.join(OUT_DIR, "validation_summary.json")
LYON_ROWS = os.path.join(OUT_DIR, "lyon.json")
LYON_SUMMARY = os.path.join(OUT_DIR, "lyon_summary.json")
LYON_BUNDLE = os.path.join(REPO, "benchmark", "lyon")

#: The GSV benchmark splits whose records carry Google's measured camera_pitch / camera_roll for
#: every pano (checked by `ids`). annapolis, clovis, morgantown, richmond, budapest_district5 and
#: laurens_mapillary are Mapillary (no pose); manual_gold's records carry no pose at all.
VALIDATION_SPLITS = ("bend", "gainesville", "laurens_gsv", "paterson", "sao_paulo")
#: A quarter of the panos, chosen by hash, were used to develop and tune the method; the rest
#: are the held-out test set. The rule is fixed so the split is reproducible from the ids alone.
DEV_MOD = 4


def role_of(pano_id):
    """'dev' for the development quarter (sha256(pano_id) % 4 == 0), else 'test'."""
    return "dev" if int(hashlib.sha256(pano_id.encode("utf-8")).hexdigest(), 16) % DEV_MOD == 0 else "test"


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

def rnd(v, nd=4):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def check_or_write(obj, path, write):
    """Write ``obj`` (LF, sorted keys) or compare the committed file's bytes to it."""
    data = dumps(obj).encode("utf-8")
    rel = os.path.relpath(path, REPO) if os.path.splitdrive(path)[0] == os.path.splitdrive(REPO)[0] else path
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {rel}")
        return 0
    if not os.path.exists(path):
        print(f"MISSING {rel}")
        return 1
    with open(path, "rb") as f:
        same = f.read() == data
    print(f"{'OK' if same else 'DIFFERS'}: {rel}")
    return 0 if same else 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tilt_magnitude_deg(pitch, roll):
    """Angle between gravity up and the rig's up axis: acos(cos pitch cos roll)."""
    c = math.cos(math.radians(pitch)) * math.cos(math.radians(roll))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def estimate_row(est, method_version=pt.METHOD_VERSION):
    """The estimator's fields of a per-pano row, rounded."""
    d = est.to_dict()
    row = {"status": d["status"], "reason": d["reason"],
           "pitch_deg": rnd(d["pitch_deg"]), "roll_deg": rnd(d["roll_deg"]),
           "se_pitch_deg": rnd(d["se_pitch_deg"]), "se_roll_deg": rnd(d["se_roll_deg"]),
           "n_segments": d["n_segments"], "n_inliers": d["n_inliers"], "n_sectors": d["n_sectors"],
           "sector_inliers": d["sector_inliers"], "residual_rms_deg": rnd(d["residual_rms_deg"]),
           "ambiguity": rnd(d["ambiguity"]), "method_version": method_version}
    row["tilt_deg"] = (rnd(tilt_magnitude_deg(d["pitch_deg"], d["roll_deg"]))
                       if d["pitch_deg"] is not None else None)
    return row


def _estimate_file(path):
    """Worker: (sha256, estimate row) for one image file."""
    gray = pt.load_equirect_gray(path, pt.TiltParams().work_width)
    return sha256_file(path), estimate_row(pt.estimate_tilt(gray))


def _run_files(paths, workers):
    if workers <= 1:
        return [_estimate_file(p) for p in paths]
    with Pool(workers) as pool:
        out = []
        for i, res in enumerate(pool.imap(_estimate_file, paths, chunksize=1)):
            out.append(res)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(paths)}", flush=True)
        return out


def params_block():
    return {"method_version": pt.METHOD_VERSION, "params": pt.TiltParams().__dict__}


# --------------------------------------------------------------------------------------------
# ids
# --------------------------------------------------------------------------------------------

def validation_ids(bench_root):
    """[(split, pano_id, role)] for every record with a numeric pitch and roll, sorted."""
    out = []
    for split in VALIDATION_SPLITS:
        for rec in load_records(os.path.join(bench_root, split, "records.jsonl")):
            p = rec["pano"]
            if not isinstance(p.get("camera_pitch"), (int, float)) or not isinstance(p.get("camera_roll"), (int, float)):
                raise SystemExit(f"{split}/{p['panorama_id']}: no numeric camera_pitch/camera_roll")
            out.append((split, p["panorama_id"], role_of(p["panorama_id"])))
    return sorted(out)


def ids_bytes(rows):
    return ("split\tpano_id\trole\n" + "".join(f"{s}\t{i}\t{r}\n" for s, i, r in rows)).encode("utf-8")


def read_ids(path=IDS):
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    assert lines[0] == ["split", "pano_id", "role"], lines[0]
    return [tuple(x) for x in lines[1:]]


def cmd_ids(args):
    data = ids_bytes(validation_ids(os.path.join(REPO, "benchmark")))
    if args.write:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(IDS, "wb") as f:
            f.write(data)
        print(f"wrote {os.path.relpath(IDS, REPO)}")
        return 0
    same = os.path.exists(IDS) and open(IDS, "rb").read() == data
    print(f"{'OK' if same else 'DIFFERS'}: {os.path.relpath(IDS, REPO)}")
    return 0 if same else 1


# --------------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------------

def cmd_validate(args):
    ids = read_ids()
    if args.limit:
        ids = ids[:args.limit]
    poses, manifests = {}, {}
    for split in VALIDATION_SPLITS:
        for rec in load_records(os.path.join(REPO, "benchmark", split, "records.jsonl")):
            p = rec["pano"]
            poses[(split, p["panorama_id"])] = p
        manifests[split] = load_json(os.path.join(REPO, "benchmark", split, "imagery_manifest.json"))["panos"]
    paths = [os.path.join(args.panos_root, s, "panos", f"{i}.jpg") for s, i, _ in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} panos missing, e.g. {missing[0]}")
    print(f"estimating {len(paths)} panos with {args.workers} workers")
    results = _run_files(paths, args.workers)
    rows = []
    for (split, pid, role), (sha, est) in zip(ids, results):
        p = poses[(split, pid)]
        row = {"split": split, "pano_id": pid, "role": role, "width": p["width"], "height": p["height"],
               "image_sha256": sha, "manifest_match": sha == manifests[split][pid]["sha256"],
               "google_pitch_deg": rnd(p["camera_pitch"]),
               "google_roll_deg": rnd(sg.wrap_deg(p["camera_roll"]))}
        row.update(est)
        rows.append(row)
    obj = dict(params_block(), rows=rows)
    rc = check_or_write(obj, VAL_ROWS, True)
    return rc | check_or_write(validation_summary(obj), VAL_SUMMARY, True)


def _err_block(rows):
    """Error of the estimate against Google's pose over ``rows`` (all must carry an estimate)."""
    if not rows:
        return {"n": 0}
    ep = np.array([r["pitch_deg"] - r["google_pitch_deg"] for r in rows])
    er = np.array([r["roll_deg"] - r["google_roll_deg"] for r in rows])
    both = np.concatenate([np.abs(ep), np.abs(er)])
    worst = np.maximum(np.abs(ep), np.abs(er))
    return {"n": len(rows),
            "mae_pitch_deg": rnd(np.mean(np.abs(ep))), "mae_roll_deg": rnd(np.mean(np.abs(er))),
            "mae_deg": rnd(np.mean(both)),
            "median_abs_deg": rnd(np.median(both)), "p90_abs_deg": rnd(np.percentile(both, 90)),
            "max_abs_deg": rnd(both.max()),
            "median_signed_pitch_deg": rnd(np.median(ep)), "median_signed_roll_deg": rnd(np.median(er)),
            "share_worst_above_1deg": rnd(np.mean(worst > 1.0)),
            "share_worst_above_3deg": rnd(np.mean(worst > 3.0))}


def _slope(x, y):
    """OLS slope and intercept of y on x (the sign test: +1 if the conventions agree)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    A = np.stack([x, np.ones_like(x)], 1)
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"slope": rnd(b), "intercept_deg": rnd(a), "n": len(x)}


def _subset_block(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    est = [r for r in rows if r["pitch_deg"] is not None]
    reasons = {}
    for r in rows:
        if r["status"] != "ok":
            for part in r["reason"].split("; "):
                key = part.split(" ")[0]
                reasons[key] = reasons.get(key, 0) + 1
    return {"n": len(rows), "n_ok": len(ok), "decline_share": rnd(1 - len(ok) / len(rows)) if rows else None,
            "decline_reasons": reasons,
            "accepted": _err_block(ok),
            "all_estimates_ungated": _err_block(est),
            "declined_with_estimate": _err_block([r for r in est if r["status"] != "ok"]),
            "sign_test_accepted": {"pitch": _slope([r["google_pitch_deg"] for r in ok], [r["pitch_deg"] for r in ok]),
                                   "roll": _slope([r["google_roll_deg"] for r in ok], [r["roll_deg"] for r in ok])}}


def validation_summary(obj):
    rows = obj["rows"]
    out = {"method_version": obj["method_version"],
           "n_manifest_mismatch": sum(not r["manifest_match"] for r in rows),
           "by_role": {k: _subset_block([r for r in rows if r["role"] == k]) for k in ("dev", "test")},
           "all": _subset_block(rows),
           "by_split_test": {s: _subset_block([r for r in rows if r["split"] == s and r["role"] == "test"])
                             for s in VALIDATION_SPLITS},
           "by_split_all": {s: _subset_block([r for r in rows if r["split"] == s]) for s in VALIDATION_SPLITS}}
    # Google's own tilt over the accepted vs declined test panos: does the gate select easy panos?
    test = [r for r in rows if r["role"] == "test"]
    out["google_tilt_by_status_test"] = {
        st: {"n": len(g), "median_tilt_deg": rnd(np.median([tilt_magnitude_deg(r["google_pitch_deg"], r["google_roll_deg"]) for r in g])) if g else None}
        for st, g in (("ok", [r for r in test if r["status"] == "ok"]), ("declined", [r for r in test if r["status"] != "ok"]))}
    # Does the error grow with tilt? Accepted test panos binned by Google's tilt magnitude.
    ok = [r for r in test if r["status"] == "ok"]
    gtilt = [tilt_magnitude_deg(r["google_pitch_deg"], r["google_roll_deg"]) for r in ok]
    out["google_tilt_max_deg"] = rnd(max(tilt_magnitude_deg(r["google_pitch_deg"], r["google_roll_deg"]) for r in rows))
    out["accepted_test_by_google_tilt"] = {
        f"{lo}-{hi}": _err_block([r for r, t in zip(ok, gtilt) if lo <= t < hi])
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 90))}
    return out


# --------------------------------------------------------------------------------------------
# lyon
# --------------------------------------------------------------------------------------------

def group_of(pano, group_map):
    """The Lyon audit group, by benchmark/lyon/group_map.json's first matching 'model|producer' rule."""
    model = pano.get("camera_model") or ""
    prod = pano.get("copyright") or ""
    for pattern, group in group_map:
        if pattern == "*":
            return group
        pm, pp = pattern.split("|", 1)
        if pm == model and (pp == "*" or pp == prod):
            return group
    raise ValueError(f"no group for {model}|{prod}")


def metadata_pose(pano):
    """(source, pitch, roll) of whatever pose the Lyon record carries.

    ``source`` is 'absent' (no camera_pitch/camera_roll), 'zero' (exactly 0.0 / 0.0),
    'panoflat' (Panoramax's own image-based estimate: the record's source_metadata names
    ``computed_by[pitch] = panoflat/...``) or 'other' (a non-zero value of unstated origin, e.g.
    a camera's XMP pose).
    """
    p, r = pano.get("camera_pitch"), pano.get("camera_roll")
    if p is None or r is None:
        return "absent", None, None
    if p == 0 and r == 0:
        return "zero", 0.0, 0.0
    sem = (pano.get("source_metadata") or {}).get("semantics") or []
    if any(s.get("key") == "computed_by[pitch]" and str(s.get("value", "")).startswith("panoflat") for s in sem):
        return "panoflat", float(p), float(r)
    return "other", float(p), float(r)


def cmd_lyon(args):
    recs = load_records(os.path.join(LYON_BUNDLE, "records.jsonl"))
    group_map = load_json(os.path.join(LYON_BUNDLE, "group_map.json"))
    manifest = load_json(os.path.join(LYON_BUNDLE, "imagery_manifest.json"))["panos"]
    pids = [r["pano"]["panorama_id"] for r in recs]
    paths = [os.path.join(args.panos_dir, manifest[pid]["file"]) for pid in pids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} panos missing, e.g. {missing[0]}")
    print(f"estimating {len(paths)} Lyon panos with {args.workers} workers (read-only)")
    results = _run_files(paths, args.workers)
    rows = []
    for rec, (sha, est) in zip(recs, results):
        p = rec["pano"]
        src, mp, mr = metadata_pose(p)
        row = {"pano_id": p["panorama_id"], "image_sha256": sha,
               "manifest_match": sha == manifest[p["panorama_id"]]["sha256"],
               "width": p["width"], "height": p["height"],
               "camera_make": p.get("camera_make"), "camera_model": p.get("camera_model"),
               "producer": p.get("copyright"), "group": group_of(p, group_map),
               "metadata_pose_source": src, "metadata_pitch_deg": rnd(mp), "metadata_roll_deg": rnd(mr)}
        row.update(est)
        rows.append(row)
    rows.sort(key=lambda r: r["pano_id"])
    obj = dict(params_block(), rows=rows,
               note="Tilt is a covariate for after Jon's review of #159. These rows are not joined "
                    "to any verdict; precision and recall by tilt wait until the review is final.")
    rc = check_or_write(obj, LYON_ROWS, True)
    return rc | check_or_write(lyon_summary(obj), LYON_SUMMARY, True)


def _tilt_block(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    t = np.array([r["tilt_deg"] for r in ok]) if ok else np.zeros(0)
    out = {"n": len(rows), "n_ok": len(ok), "decline_share": rnd(1 - len(ok) / len(rows)) if rows else None}
    if ok:
        out.update({"median_tilt_deg": rnd(np.median(t)), "p90_tilt_deg": rnd(np.percentile(t, 90)),
                    "max_tilt_deg": rnd(t.max()),
                    "median_abs_pitch_deg": rnd(np.median([abs(r["pitch_deg"]) for r in ok])),
                    "median_abs_roll_deg": rnd(np.median([abs(r["roll_deg"]) for r in ok])),
                    "n_tilt_ge_2deg": int((t >= 2).sum()), "n_tilt_ge_5deg": int((t >= 5).sum()),
                    "n_tilt_ge_10deg": int((t >= 10).sum())})
    return out


def panoflat_agreement(rows):
    """Estimate vs Panoramax's panoflat pitch/roll, under both pitch-sign hypotheses.

    panoflat is another image-based estimator, so this is agreement between two instruments,
    not error against a measurement. Which pitch sign Panoramax's ``pers:pitch`` uses is not
    stated in the record; the rows decide it (``pitch_flipped`` means ``pers:pitch = -pitch``).
    """
    pf = [r for r in rows if r["metadata_pose_source"] == "panoflat" and r["status"] == "ok"]
    if not pf:
        return {"n": 0}
    out = {"n": len(pf)}
    for name, sign in (("same_sign", 1.0), ("pitch_flipped", -1.0)):
        dp = np.array([r["pitch_deg"] - sign * r["metadata_pitch_deg"] for r in pf])
        dr = np.array([r["roll_deg"] - r["metadata_roll_deg"] for r in pf])
        out[name] = {"mae_pitch_deg": rnd(np.mean(np.abs(dp))), "mae_roll_deg": rnd(np.mean(np.abs(dr))),
                     "median_abs_deg": rnd(np.median(np.abs(np.concatenate([dp, dr])))),
                     "max_abs_deg": rnd(np.max(np.abs(np.concatenate([dp, dr]))))}
    return out


def lyon_summary(obj):
    rows = obj["rows"]
    groups = sorted({r["group"] for r in rows})
    models = sorted({r["camera_model"] or "(none)" for r in rows})
    out = {"method_version": obj["method_version"],
           "n_manifest_mismatch": sum(not r["manifest_match"] for r in rows),
           "all": _tilt_block(rows),
           "by_group": {g: _tilt_block([r for r in rows if r["group"] == g]) for g in groups},
           "by_camera_model": {m: _tilt_block([r for r in rows if (r["camera_model"] or "(none)") == m]) for m in models},
           "by_metadata_pose_source": {s: _tilt_block([r for r in rows if r["metadata_pose_source"] == s])
                                       for s in sorted({r["metadata_pose_source"] for r in rows})}}
    out["metadata_nonzero"] = [
        {k: r[k] for k in ("pano_id", "camera_model", "group", "metadata_pose_source", "metadata_pitch_deg",
                           "metadata_roll_deg", "status", "pitch_deg", "roll_deg", "tilt_deg", "reason")}
        for r in rows if r["metadata_pose_source"] in ("panoflat", "other")]
    out["panoflat_agreement"] = panoflat_agreement(rows)
    out["declined_reasons"] = {}
    for r in rows:
        if r["status"] != "ok":
            for part in r["reason"].split("; "):
                key = part.split(" ")[0]
                out["declined_reasons"][key] = out["declined_reasons"].get(key, 0) + 1
    return out


# --------------------------------------------------------------------------------------------
# summarize / inspect
# --------------------------------------------------------------------------------------------

def cmd_summarize(args):
    rc = 0
    if os.path.exists(VAL_ROWS):
        rc |= check_or_write(validation_summary(load_json(VAL_ROWS)), VAL_SUMMARY, args.write)
    if os.path.exists(LYON_ROWS):
        rc |= check_or_write(lyon_summary(load_json(LYON_ROWS)), LYON_SUMMARY, args.write)
    return rc


def cmd_inspect(args):
    from PIL import Image, ImageDraw
    rec = next(r["pano"] for r in load_records(os.path.join(REPO, "benchmark", args.split, "records.jsonl"))
               if r["pano"]["panorama_id"] == args.pano_id)
    params = pt.TiltParams()
    gray = pt.load_equirect_gray(os.path.join(args.panos_root, args.split, "panos", f"{args.pano_id}.jpg"),
                                 params.work_width)
    est = pt.estimate_tilt(gray, params)
    view = pt.make_views(params, gray.shape[1])[args.view]
    panels = []
    poses = [("none", 0.0, 0.0), ("record pose", rec.get("camera_pitch") or 0.0, float(sg.wrap_deg(rec.get("camera_roll") or 0.0))),
             ("estimate", est.pitch_deg, est.roll_deg)]
    for name, p, r in poses:
        if p is None:
            continue
        im = pt.render_view(pt.level_equirect(gray, p, r), view)
        img = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(img)
        for x in range(0, img.width, 40):
            d.line([(x, 0), (x, img.height)], fill=(255, 0, 0), width=1)
        d.text((8, 8), f"levelled by {name}: pitch {p:.2f} roll {r:.2f}", fill=(255, 255, 0))
        panels.append(img)
    out = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1), panels[0].height))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 8
    out.save(args.out, quality=90)
    print(f"estimate {est.status} {est.reason}: pitch {est.pitch_deg} roll {est.roll_deg}; wrote {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ids", help="write/check the validation id list")
    a.add_argument("--write", action="store_true")
    a = sub.add_parser("validate", help="estimate the GSV validation panos and compare with Google's pose")
    a.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"),
                   help="directory holding <split>/panos/<pano_id>.jpg")
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--limit", type=int, default=0, help="first N ids only (smoke test; overwrites the outputs)")
    a = sub.add_parser("lyon", help="estimate the 200 Lyon review panos")
    a.add_argument("--panos-dir", required=True, help="directory holding the 200 bundle jpgs (read-only)")
    a.add_argument("--workers", type=int, default=1)
    a = sub.add_parser("summarize", help="re-derive the summaries from the committed rows")
    a.add_argument("--write", action="store_true")
    a = sub.add_parser("inspect", help="render a view levelled by the record's pose vs by the estimate")
    a.add_argument("--split", required=True)
    a.add_argument("--pano-id", required=True)
    a.add_argument("--view", type=int, default=0, help="which of the 8 views (0 = bearing 0, 45 deg steps)")
    a.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"))
    a.add_argument("--out", required=True)
    sub.add_parser("check", help="compare the id list and both summaries with a re-derivation (no images)")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        args.write = False
        return cmd_ids(args) | cmd_summarize(args)
    return {"ids": cmd_ids, "validate": cmd_validate, "lyon": cmd_lyon, "summarize": cmd_summarize,
            "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
