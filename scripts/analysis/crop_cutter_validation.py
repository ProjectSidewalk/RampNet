"""Validation of the crop cutter against the HF validated crops, plus pano-store coverage (#86 item 2b).

Four steps, in order. Steps 1, 3 and 4 run on any machine with the PR #175 audit cache and
network access to Hugging Face; step 2 (and the cutting between 3 and 4) needs the makelab2
pano store, which is an unpublished local input. Everything a step writes that a number in
``docs/crop_cutter.md`` depends on goes to ``docs/data/crop_cutter/`` and is committed.

1. ``sample``   (local) -- read the HF CurbRamp index and the audit's rawLabels cache; write
   ``coverage_input.csv``: every HF validated CurbRamp label, plus a seeded 5,000-label sample
   of tag-era human CurbRamp labels (placed on or after 2018-04-29), with the geometry columns
   the cutter needs, in a seeded random order.
2. ``coverage`` (makelab2, stdlib only) -- stat ``<store>/<city_dir>/<id[:2]>/<id>.jpg`` for
   every row; write ``coverage.csv`` (one flag per row) and ``coverage.json`` (per set, per
   city, per crop era).
3. ``pick``     (local) -- the first ``--n`` HF rows, in the seeded order, whose pano is in the
   store -> ``validation_sample.csv`` (the ``--labels`` input for ``scripts/crop_cutter.py``).
4. ``compare``  (local) -- fetch just those HF crops by HTTP range (never the 30 GB zip), compare
   each to the cutter's ``viewport`` crop of the same label, and write ``validation.json``,
   ``validation_per_label.csv`` and the contact sheet ``docs/assets/crop_cutter_contact_sheet.jpg``.

Exact commands: ``docs/crop_cutter.md`` section "Reproduce".
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

DATA = os.path.join(REPO, "docs", "data", "crop_cutter")
ASSETS = os.path.join(REPO, "docs", "assets")
HF_CACHE = os.path.join(REPO, "analysis_out", "crop_cutter", "hf_crops")

SEED = 86
TAG_ERA_START = "2018-04-29"      # tags entered the schema (SidewalkWebpage evolution 14)
CROP_DATE = "2023-10-12"          # production stores a crop for every label placed since
SIDEWALK_AI_USER = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"
HF_VALIDATED_ZIP = ("https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/"
                    "resolve/main/Validated/CurbRamp.zip")
#: HF filename city -> audit city id (from the audit's hf_validated_join.json city_map).
HF_CITY = {"amsterdam": "amsterdam", "cdmx": "cdmx", "chicago": "chicago-il", "columbus": "columbus-oh",
           "newberg": "newberg-or", "oradell": "oradell-nj", "pittsburgh": "pittsburgh-pa",
           "seattle": "seattle-wa", "spgg": "spgg", "walla-walla": "walla-walla"}

COLS = ["set", "order", "city", "label_id", "pano_id", "pano_source", "time_created", "hf_split",
        "hf_filename", "pano_x", "pano_y", "pano_width", "pano_height", "heading", "pitch", "zoom",
        "camera_heading", "camera_pitch", "camera_roll", "canvas_x", "canvas_y"]


def _write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _rnd(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ----------------------------------------------------------------------------- 1. sample

def cmd_sample(args):
    hf = _read_csv(args.hf_index)
    want = defaultdict(dict)
    for r in hf:
        want[HF_CITY[r["city"]]][int(r["label_id"])] = r
    rows, tag_era = [], []
    for path in sorted(os.listdir(args.raw_cache)):
        if not path.endswith("__rawLabels__CurbRamp.csv"):
            continue
        city = path.split("__")[0]
        for r in _read_csv(os.path.join(args.raw_cache, path)):
            lid = int(r["label_id"])
            g = {k: r.get(k, "") for k in COLS if k in r}
            g.update(city=city, label_id=lid)
            if lid in want.get(city, {}):
                h = want[city][lid]
                g.update(set="hf_validated", hf_split=h["split"], hf_filename=h["filename"])
                rows.append(g)
            if r.get("time_created", "") >= args.tag_era_start and r.get("user_id") != SIDEWALK_AI_USER:
                t = dict(g)
                t.update(set="tag_era_5k", hf_split="", hf_filename="")
                tag_era.append(t)
    rng = random.Random(args.seed)
    tag_era.sort(key=lambda r: (r["city"], r["label_id"]))
    sample = rng.sample(tag_era, args.n_tag_era)
    rows.sort(key=lambda r: (r["city"], r["label_id"]))
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["order"] = i
    for i, r in enumerate(sample):
        r["order"] = i
    missing = sum(len(v) for v in want.values()) - len(rows)
    _write_csv(args.out, rows + sample, COLS)
    print(f"wrote {args.out}: {len(rows)} HF rows ({missing} HF labels not in the cache), "
          f"{len(sample)} of {len(tag_era)} tag-era labels")


# ----------------------------------------------------------------------------- 2. coverage

def cmd_coverage(args):
    from crop_cutter import STORE_CITY_DIRS
    rows = _read_csv(args.input)
    out, agg = [], defaultdict(lambda: [0, 0])
    for r in rows:
        d = STORE_CITY_DIRS.get(r["city"], r["city"])
        pid = r["pano_id"].strip()
        present = bool(pid) and os.path.isfile(os.path.join(args.store, d, pid[:2], pid + ".jpg"))
        era = "has_prod_crop" if r["time_created"] >= CROP_DATE else "needs_recut"
        out.append({"set": r["set"], "city": r["city"], "label_id": r["label_id"], "pano_id": pid,
                    "present": int(present)})
        for key in [(r["set"], "all"), (r["set"], "city", r["city"]), (r["set"], "era", era),
                    (r["set"], "source", r["pano_source"] or "unknown")]:
            agg[key][0] += int(present)
            agg[key][1] += 1
    summary = defaultdict(lambda: defaultdict(dict))
    for key, (p, n) in sorted(agg.items()):
        cell = {"present": p, "n": n, "pct": round(100.0 * p / n, 2)}
        if key[1] == "all":
            summary[key[0]]["all"] = cell
        else:
            summary[key[0]]["by_" + key[1]][key[2]] = cell
    # a pano may carry several labels: also count distinct panos
    for s in {r["set"] for r in out}:
        seen = {}
        for r in out:
            if r["set"] == s:
                seen[(r["city"], r["pano_id"])] = r["present"]
        summary[s]["distinct_panos"] = {"present": sum(seen.values()), "n": len(seen),
                                        "pct": round(100.0 * sum(seen.values()) / len(seen), 2)}
    res = {"store": args.store, "probed_utc": args.probed, "input": os.path.relpath(args.input, REPO),
           "sets": summary}
    _write_csv(args.out_csv, out, ["set", "city", "label_id", "pano_id", "present"])
    _write_json(args.out_json, res)
    for s, v in summary.items():
        print(s, v["all"], "panos", v["distinct_panos"])


# ----------------------------------------------------------------------------- 3. pick

def cmd_pick(args):
    rows = [r for r in _read_csv(args.input) if r["set"] == "hf_validated"]
    cov = {(r["city"], r["label_id"]): r["present"] == "1"
           for r in _read_csv(args.coverage) if r["set"] == "hf_validated"}
    rows.sort(key=lambda r: int(r["order"]))
    pick = [r for r in rows if cov.get((r["city"], r["label_id"]))][: args.n]
    _write_csv(args.out, pick, COLS)
    print(f"wrote {args.out}: {len(pick)} labels, cities "
          f"{dict(sorted(defaultdict(int, {c: sum(1 for r in pick if r['city'] == c) for c in {r['city'] for r in pick}}).items()))}")


# ----------------------------------------------------------------------------- 4. compare

def _hf_zip():
    import zipfile
    sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
    from _range_file import RangeFile
    return zipfile.ZipFile(io.BufferedReader(RangeFile(HF_VALIDATED_ZIP), buffer_size=1 << 20))


def fetch_hf_crops(filenames, cache=HF_CACHE):
    """Fetch only the named members of the HF zip (HTTP range), caching them as PNG files."""
    os.makedirs(cache, exist_ok=True)
    todo = [f for f in filenames if not os.path.isfile(os.path.join(cache, f))]
    if todo:
        z = _hf_zip()
        for f in todo:
            data = z.read("CurbRamp/crops/" + f)
            with open(os.path.join(cache, f), "wb") as fh:
                fh.write(data)
    return {f: os.path.join(cache, f) for f in filenames}


def _gray(img, size):
    import numpy as np
    from PIL import Image
    return np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=np.float64)


def ncc(a, b):
    import numpy as np
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def compare_pair(hf_img, cut_img, size=(360, 240)):
    """NCC and SSIM at ``size`` (grey), and the residual shift by phase correlation, reported in
    pixels of the 1440x960 frame and as degrees at the view centre."""
    from skimage.metrics import structural_similarity
    from skimage.registration import phase_cross_correlation
    a, b = _gray(hf_img, size), _gray(cut_img, size)
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=4)
    scale = 1440.0 / size[0]
    return {"ncc": ncc(a, b), "ssim": float(structural_similarity(a, b, data_range=255.0)),
            "shift_x_px": float(shift[1]) * scale, "shift_y_px": float(shift[0]) * scale}


def _q(vals, q):
    import numpy as np
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return round(float(np.quantile(v, q)), 4) if v else None


def cmd_compare(args):
    import numpy as np
    from PIL import Image
    from rampnet import crops

    sample = _read_csv(args.sample)
    manifests = {}
    for tag, path in (s.split("=", 1) for s in args.manifest):
        latest = {}
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    latest[r["name"]] = r
        manifests[tag] = latest
    hf_paths = fetch_hf_crops([r["hf_filename"] for r in sample])
    rng = random.Random(args.seed)

    per, fovs = [], {}
    for r in sample:
        hf_img = Image.open(hf_paths[r["hf_filename"]]).convert("RGB")
        row = {"city": r["city"], "label_id": r["label_id"], "pano_id": r["pano_id"],
               "zoom": r["zoom"], "time_created": r["time_created"],
               "hf_width": hf_img.size[0], "hf_height": hf_img.size[1]}
        for tag, latest in manifests.items():
            name = f"{r['city']}__{r['label_id']}__{tag}.jpg"
            m = latest.get(name)
            if not m or m["status"] != "ok":
                row[f"{tag}_status"] = m["status"] if m else "absent"
                continue
            cut = Image.open(os.path.join(args.crops, name)).convert("RGB")
            c = compare_pair(hf_img, cut)
            # the residual shift as an angle at the view centre, so zooms are comparable
            ppd = m["width"] / 2.0 / math.tan(math.radians(m["fov_h_deg"]) / 2.0) * math.pi / 180.0
            c["shift_deg"] = math.hypot(c["shift_x_px"], c["shift_y_px"]) / ppd
            c["black_frac"] = crops.black_fraction(np.asarray(cut))
            for k, v in c.items():
                row[f"{tag}_{k}"] = _rnd(v)
            row[f"{tag}_status"] = "ok"
        per.append(row)

    # null: the same metric between a label's viewport crop and a *different* label's HF crop
    # of the same city (a shuffled pairing), so "0.6" has something to be compared with
    base = args.null_tag
    null = []
    by_city = defaultdict(list)
    for r in sample:
        by_city[r["city"]].append(r)
    for r in sample:
        others = [o for o in by_city[r["city"]] if o["label_id"] != r["label_id"]]
        if not others:
            continue
        o = rng.choice(others)
        name = f"{r['city']}__{r['label_id']}__{base}.jpg"
        if manifests[base].get(name, {}).get("status") != "ok":
            continue
        a = Image.open(hf_paths[o["hf_filename"]]).convert("RGB")
        b = Image.open(os.path.join(args.crops, name)).convert("RGB")
        null.append(compare_pair(a, b))

    summ = {}
    for tag in manifests:
        ok = [p for p in per if p.get(f"{tag}_status") == "ok"]
        sh = [math.hypot(p[f"{tag}_shift_x_px"], p[f"{tag}_shift_y_px"]) for p in ok]
        summ[tag] = {"n": len(ok),
                     "ncc_median": _q([p[f"{tag}_ncc"] for p in ok], 0.5),
                     "ncc_p10": _q([p[f"{tag}_ncc"] for p in ok], 0.1),
                     "ncc_p90": _q([p[f"{tag}_ncc"] for p in ok], 0.9),
                     "ssim_median": _q([p[f"{tag}_ssim"] for p in ok], 0.5),
                     "shift_px_median": _q(sh, 0.5), "shift_px_p90": _q(sh, 0.9),
                     "shift_deg_median": _q([p[f"{tag}_shift_deg"] for p in ok], 0.5),
                     "shift_deg_p90": _q([p[f"{tag}_shift_deg"] for p in ok], 0.9),
                     "n_black_frac_gt_1pct": sum(p[f"{tag}_black_frac"] > 0.01 for p in ok),
                     "shift_y_px_median": _q([p[f"{tag}_shift_y_px"] for p in ok], 0.5),
                     "shift_x_px_median": _q([p[f"{tag}_shift_x_px"] for p in ok], 0.5),
                     "frac_ncc_ge_0p5": round(sum(p[f"{tag}_ncc"] >= 0.5 for p in ok) / len(ok), 4) if ok else None}
    summ["null_shuffled_same_city"] = {"n": len(null),
                                       "ncc_median": _q([c["ncc"] for c in null], 0.5),
                                       "ncc_p90": _q([c["ncc"] for c in null], 0.9),
                                       "ssim_median": _q([c["ssim"] for c in null], 0.5)}
    by_zoom = {}
    for z in sorted({p["zoom"] for p in per}):
        ok = [p for p in per if p["zoom"] == z and p.get(f"{base}_status") == "ok"]
        by_zoom[z] = {"n": len(ok), "ncc_median": _q([p[f"{base}_ncc"] for p in ok], 0.5),
                      "shift_deg_median": _q([p[f"{base}_shift_deg"] for p in ok], 0.5),
                      "shift_px_median": _q([math.hypot(p[f"{base}_shift_x_px"], p[f"{base}_shift_y_px"])
                                             for p in ok], 0.5)}
    res = {"sample": os.path.relpath(args.sample, REPO), "n_sample": len(sample),
           "metric": "grey 360x240 after bilinear resize; NCC = zero-mean normalized cross-correlation; "
                     "SSIM = skimage structural_similarity; shift = phase correlation, in 1440x960 px",
           "hf_zip": HF_VALIDATED_ZIP, "by_tag": summ, "viewport_by_zoom": by_zoom,
           "cut_run": args.cut_run}
    if args.cut_summary and os.path.isfile(args.cut_summary):
        with open(args.cut_summary, encoding="utf-8") as fh:
            res["cut_summary"] = json.load(fh)
    cols = sorted({k for p in per for k in p}, key=lambda k: (k not in ("city", "label_id"), k))
    _write_csv(os.path.join(DATA, "validation_per_label.csv"), per, cols)
    _write_json(os.path.join(DATA, "validation.json"), res)
    print(json.dumps(summ, indent=1))

    # contact sheet: HF crop | viewport cut | the label-centred cuts, for a seeded handful
    tags = [t for t in args.sheet_tags]
    pick = random.Random(args.seed).sample([r for r in sample], min(args.sheet_n, len(sample)))
    tw, th = 240, 160
    sheet = Image.new("RGB", (tw * (1 + len(tags)), th * len(pick)), (255, 255, 255))
    for i, r in enumerate(pick):
        ims = [Image.open(hf_paths[r["hf_filename"]]).convert("RGB")]
        for t in tags:
            p = os.path.join(args.crops, f"{r['city']}__{r['label_id']}__{t}.jpg")
            ims.append(Image.open(p).convert("RGB") if os.path.isfile(p) else Image.new("RGB", (tw, th)))
        for j, im in enumerate(ims):
            im = im.resize((tw, th), Image.LANCZOS)
            arr = np.asarray(im).copy()
            # mark where the label is: canvas point for HF/viewport, centre for label-centred cuts
            if j == 0 or tags[j - 1] == "viewport":
                lx, ly = float(r["canvas_x"]) / 720 * tw, float(r["canvas_y"]) / 480 * th
            else:
                lx, ly = tw / 2, th / 2
            for dx in range(-6, 7):
                for x, y in ((lx + dx, ly), (lx, ly + dx)):
                    if 0 <= int(x) < tw and 0 <= int(y) < th and abs(dx) > 2:
                        arr[int(y), int(x)] = (255, 255, 0)
            sheet.paste(Image.fromarray(arr), (j * tw, i * th))
    os.makedirs(ASSETS, exist_ok=True)
    sheet.save(os.path.join(ASSETS, "crop_cutter_contact_sheet.jpg"), quality=80)
    print("wrote", os.path.join(ASSETS, "crop_cutter_contact_sheet.jpg"), "columns: HF |", " | ".join(tags))


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--hf-index", required=True, help="audit hf_validated_curbramp_index.csv (PR #175 hf-index)")
    s.add_argument("--raw-cache", required=True, help="audit rawLabels cache (PR #175 fetch)")
    s.add_argument("--tag-era-start", default=TAG_ERA_START)
    s.add_argument("--n-tag-era", type=int, default=5000)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", default=os.path.join(DATA, "coverage_input.csv"))
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("coverage")
    c.add_argument("--store", required=True)
    c.add_argument("--input", default=os.path.join(DATA, "coverage_input.csv"))
    c.add_argument("--probed", required=True, help="UTC date of the probe, recorded in the JSON")
    c.add_argument("--out-csv", default=os.path.join(DATA, "coverage.csv"))
    c.add_argument("--out-json", default=os.path.join(DATA, "coverage.json"))
    c.set_defaults(func=cmd_coverage)

    p = sub.add_parser("pick")
    p.add_argument("--input", default=os.path.join(DATA, "coverage_input.csv"))
    p.add_argument("--coverage", default=os.path.join(DATA, "coverage.csv"))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default=os.path.join(DATA, "validation_sample.csv"))
    p.set_defaults(func=cmd_pick)

    m = sub.add_parser("compare")
    m.add_argument("--sample", default=os.path.join(DATA, "validation_sample.csv"))
    m.add_argument("--crops", required=True, help="the cutter's --out directory, copied back")
    m.add_argument("--manifest", action="append", required=True,
                   help="tag=manifest.jsonl, e.g. viewport=<crops>/manifest.jsonl (repeatable)")
    m.add_argument("--null-tag", default="viewport")
    m.add_argument("--sheet-tags", nargs="*", default=["viewport", "fov30", "fov60", "fov90"])
    m.add_argument("--sheet-n", type=int, default=8)
    m.add_argument("--seed", type=int, default=SEED)
    m.add_argument("--cut-summary", help="the cutter's --summary JSON, embedded in validation.json")
    m.add_argument("--cut-run", help="free text: where/when the cut ran")
    m.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
