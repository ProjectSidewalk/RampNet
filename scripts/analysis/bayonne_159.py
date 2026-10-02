"""Bayonne (#159), staged ahead of its ground-truth review: checks and GT-free reads.

Bayonne is the first Panoramax split: 125 GoPro Max panoramas, 123 of them from the
municipal account, exported by sidewalk-auto-labeler (PR 125) and staged in
``benchmark/bayonne/``. **Nothing here uses ground truth, and nothing here is a
precision or recall.** The review is a human pass that has not happened; these reads
exist so that the review starts with context and so that every number that needs a
GPU is already cached the moment ``verdicts.json`` lands.

Subcommands, in the order ``docs/bayonne_split_159.md`` runs them:

    # bundle integrity (needs benchmark/bayonne/panos/ for the byte check)
    python scripts/analysis/bayonne_159.py verify

    # the nadir logo band, measured per pano -> benchmark/bayonne/nadir_band.json
    python scripts/analysis/bayonne_159.py band --write

    # detections per pano vs threshold, by sampler stratum, against every split's
    # committed op_cache -> analysis_out/bayonne_159/firing.json
    python scripts/analysis/bayonne_159.py firing --write

    # where the peaks sit in the frame (normalized y), GoPro Max splits only
    # -> analysis_out/bayonne_159/frame.json
    python scripts/analysis/bayonne_159.py frame --write

    # where RampNet is silent and >= 2 challenger legs agree (reviewer attention list)
    # -> analysis_out/bayonne_159/candidates.json
    python scripts/analysis/bayonne_159.py candidates --write

Every ``--write`` output is LF, sorted, with floats rounded, so a re-run on any OS is
byte-identical; ``--check`` (the default when ``--write`` is absent) recomputes and
compares against the committed file. ``tests/test_bayonne_159.py`` runs the CPU ones.

**The extractor difference that the firing read controls for.** Ten of the eleven
committed ``analysis_out/op_cache/*.json`` were built before f4c71c8, when
``peaks_to_dets`` left ``skimage``'s ``exclude_border`` at its default and so dropped
every peak within ``min_distance`` (10 heatmap cells) of the array edge; they carry 0
border-ring peaks. ``laurens_mapillary``'s was built after the fix (2026-08-31) and
carries 4. Bayonne's cache is built with the fixed extractor. So ``firing`` reports two columns per split: ``raw`` (as cached)
and ``interior`` (every split with peaks in the 10-cell border ring removed), and
cross-split comparisons are read off ``interior``, where all splits are measured the
same way.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPLIT = "bayonne"
BUNDLE = os.path.join(REPO, "benchmark", SPLIT)
OUT_DIR = os.path.join(REPO, "analysis_out", "bayonne_159")
BAYONNE_CACHE = os.path.join(OUT_DIR, "op_cache", f"{SPLIT}.json")
OP_CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
NADIR_BAND = os.path.join(BUNDLE, "nadir_band.json")

# The sampler's strata (export_benchmark.py / gt_gallery.choose_panos).
STRATA = ("top", "random", "empty")
EXPECTED_STRATA = {"top": 5, "random": 95, "empty": 25}
EXPECTED_DETECTIONS = 147          # >= 0.55 in records.jsonl, from the labeler hand-off
DEPLOYED = 0.55

# Heatmap geometry and the pre-f4c71c8 border ring (see the module docstring).
HEAT_W, HEAT_H = 1024, 512
MIN_DISTANCE = 10

# Thresholds the firing read reports. 0.05 is the cache floor, 0.30 the recommended
# operating point (docs/operating_point.md), 0.55 the deployed one.
FIRING_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                     0.60, 0.70, 0.80, 0.90)

# Splits whose committed op_cache the firing read compares against. laurens_gsv has
# no op_cache (never extracted) and manual_gold is not a city bundle drawn by the
# sampler, so neither is here.
COMPARISON_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
                     "gainesville", "laurens_mapillary", "budapest_district5", "sao_paulo")

# The band as the labeler measured it on two municipal panos
# (sidewalk-auto-labeler docs/panoramax-bayonne.md, LOGO_BAND_Y), and the rig mask
# the labeler applies at inference (NADIR_MASK_DEG = 49 deg -> y >= 0.772).
LABELER_BAND_Y = 0.791
RIG_MASK_Y = 0.772

# Band measurement. In the municipal panos the band is pure white across the centre
# of the frame (the logo text sits near x 0.2-0.3 and 0.7-0.85), so the band's top is
# the first row, scanning up from near the bottom, at which the logo-free centre
# columns stop being white. Measured on a 1024x512 bilinear downsample.
BAND_X0, BAND_X1 = 420, 600        # logo-free columns at 1024 wide (x 0.41-0.59)
BAND_WHITE_MIN = 235               # every channel above this counts as white
BAND_ROW_WHITE_FRAC = 0.98         # share of those columns that must be white
BAND_START_ROW = int(0.95 * HEAT_H)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def _check_or_write(obj, path, write):
    text = _dumps(obj)
    if write:
        _dump(obj, path)
        print(f"wrote {os.path.relpath(path, REPO)}")
        return 0
    if not os.path.exists(path):
        print(f"MISSING {os.path.relpath(path, REPO)} (run with --write)")
        return 1
    with open(path, encoding="utf-8") as f:
        same = f.read() == text
    print(f"{'OK' if same else 'DIFFERS'}: {os.path.relpath(path, REPO)}")
    return 0 if same else 1


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_records(split, repo=REPO):
    path = os.path.join(repo, "benchmark", split, "records.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strata_of(split, repo=REPO):
    """``{pano_id: stratum}`` from records' ``benchmark_group``, falling back to the
    verdicts' ``group`` for the hand-built bundles that predate the field."""
    out = {}
    vpath = os.path.join(repo, "benchmark", split, "verdicts.json")
    vgroups = {}
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            vgroups = {pid: e.get("group") for pid, e in json.load(f)["panos"].items()}
    for r in load_records(split, repo):
        pid = r["pano"]["panorama_id"]
        out[pid] = r.get("benchmark_group") or vgroups.get(pid) or "random"
    return out


def load_peaks(path):
    """``{pano_id: [(x, y, score), ...]}`` from an op_cache file, and its meta."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return ({p["pano"]: [tuple(t) for t in p["preds"]] for p in payload["panos"]},
            payload.get("meta", {}))


def in_border_ring(x, y, md=MIN_DISTANCE, w=HEAT_W, h=HEAT_H):
    """True where skimage's ``exclude_border=True`` would have dropped a peak: within
    ``md`` cells of any edge of the heatmap. ``x``/``y`` are ``col/W`` and ``row/H``,
    exactly as ``peaks_to_dets`` writes them."""
    c, r = round(x * w), round(y * h)
    return c < md or r < md or c >= w - md or r >= h - md


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify_bundle(bundle=BUNDLE):
    """List of problems with the staged bundle (empty = OK).

    The committed pin is ``imagery_manifest.json`` (sha256 + bytes per pano), written
    at staging time. The exporter's ``index.csv`` is git-ignored by repo convention
    (it travels with the imagery), so it is cross-checked against the manifest only
    where it exists. The byte check runs only for panos present on disk; a missing
    ``panos/`` is reported, not failed, so the record-level checks still run on a
    clean clone."""
    problems, notes = [], []
    with open(os.path.join(bundle, "imagery_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)["panos"]
    recs = load_records(os.path.basename(bundle), os.path.dirname(os.path.dirname(bundle)))
    rec_ids = [r["pano"]["panorama_id"] for r in recs]
    if len(set(rec_ids)) != len(rec_ids):
        problems.append("duplicate panorama_id in records.jsonl")
    if set(rec_ids) != set(manifest):
        problems.append(f"records.jsonl and imagery_manifest.json disagree: "
                        f"{len(set(rec_ids) - set(manifest))} only in records, "
                        f"{len(set(manifest) - set(rec_ids))} only in the manifest")
    index_path = os.path.join(bundle, "index.csv")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8", newline="") as f:
            index = {row["panorama_id"]: row for row in csv.DictReader(f)}
        bad = [pid for pid in set(index) | set(manifest)
               if pid not in index or pid not in manifest
               or index[pid]["sha256"] != manifest[pid]["sha256"]
               or int(index[pid]["bytes"]) != manifest[pid]["bytes"]]
        if bad:
            problems.append(f"index.csv and imagery_manifest.json disagree on {len(bad)} panos")
        else:
            notes.append(f"exporter index.csv agrees with imagery_manifest.json "
                         f"({len(index)} panos)")
    else:
        notes.append("index.csv absent (git-ignored): exporter cross-check skipped")
    groups = {}
    for r in recs:
        groups[r.get("benchmark_group")] = groups.get(r.get("benchmark_group"), 0) + 1
    if groups != EXPECTED_STRATA:
        problems.append(f"strata {groups} != {EXPECTED_STRATA}")
    n_det = sum(1 for r in recs for d in r["detections"] if d["confidence"] >= DEPLOYED)
    if n_det != EXPECTED_DETECTIONS:
        problems.append(f"{n_det} detections >= {DEPLOYED}, expected {EXPECTED_DETECTIONS}")
    for r in recs:
        if r["benchmark_group"] == "empty" and r["detections"]:
            problems.append(f"{r['pano']['panorama_id']}: 'empty' pano carries detections")
    panos_dir = os.path.join(bundle, "panos")
    if not os.path.isdir(panos_dir):
        notes.append("panos/ absent: byte check skipped")
    else:
        checked = 0
        for pid, row in sorted(manifest.items()):
            path = os.path.join(panos_dir, row["file"])
            if not os.path.exists(path):
                problems.append(f"{pid}: missing from panos/")
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != row["sha256"] or os.path.getsize(path) != row["bytes"]:
                problems.append(f"{pid}: bytes differ from imagery_manifest.json")
            checked += 1
        notes.append(f"{checked} panos byte-checked against imagery_manifest.json")
    return problems, notes, {"panos": len(recs), "strata": groups, "detections_055": n_det}


def cmd_verify(args):
    problems, notes, facts = verify_bundle(args.bundle)
    print(json.dumps(facts, sort_keys=True))
    for n in notes:
        print("  " + n)
    for p in problems:
        print("  PROBLEM: " + p)
    print("verify: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# band
# --------------------------------------------------------------------------- #
#: Bands the automatic method cannot see. The OSM-FR producer's green-and-white band
#: has map graphics where the method looks for white (the centre columns at y 0.95),
#: so it is read by hand: on the 1024x512 downsample rows 0-408 are photo (row mean
#: ~100), row 409 is the transition (154), and rows 410 onward are all white across the
#: centre columns, so the band starts at row 410 = y 0.8008. Read 2026-10-02 by
#: claude-opus-5-5 with the same downsample; an overlay at y 0.80 sits on the edge
#: (PR #234 review, M6).
MANUAL_BANDS = {"f8759625-2874-4b43-832f-a3aa2669af15": 0.8008}


def band_top_from_rows(white_frac, start_row=BAND_START_ROW, need=BAND_ROW_WHITE_FRAC,
                       h=HEAT_H):
    """Normalized y of the band's top edge from per-row white fractions, or None.

    Pure. Starts at ``start_row`` (inside the band, above the bottom ornament line)
    and walks up while the row is still white. A pano whose ``start_row`` is not white
    has no white band (the two non-municipal panos)."""
    if white_frac[start_row] < need:
        return None
    r = start_row
    while r > 0 and white_frac[r - 1] >= need:
        r -= 1
    return r / h


def measure_band(path):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((HEAT_W, HEAT_H), Image.BILINEAR))
    centre = a[:, BAND_X0:BAND_X1].astype(int)
    white = (centre.min(axis=2) > BAND_WHITE_MIN).mean(axis=1)
    return band_top_from_rows(list(white))


def build_band(bundle=BUNDLE):
    recs = load_records(os.path.basename(bundle), os.path.dirname(os.path.dirname(bundle)))
    panos = {}
    for r in recs:
        pid = r["pano"]["panorama_id"]
        y = measure_band(os.path.join(bundle, "panos", f"{pid}.jpg"))
        entry = {"band_top_y": _r(y), "copyright": r["pano"].get("copyright"),
                 "method": "auto"}
        if y is None and pid in MANUAL_BANDS:
            entry.update(band_top_y=MANUAL_BANDS[pid], method="manual")
        panos[pid] = entry
    ys = sorted(v["band_top_y"] for v in panos.values() if v["band_top_y"] is not None)
    return {
        "what": "top edge of the white nadir logo band, normalized y (0 = top of the "
                "equirect), measured per pano. null = no white band found.",
        "method": {
            "downsample": f"{HEAT_W}x{HEAT_H} bilinear",
            "columns": f"x {BAND_X0}-{BAND_X1} of {HEAT_W} (logo-free centre)",
            "white": f"every channel > {BAND_WHITE_MIN}",
            "row_is_band": f">= {BAND_ROW_WHITE_FRAC} of those columns white",
            "scan": f"up from row {BAND_START_ROW} while rows stay white",
            "script": "scripts/analysis/bayonne_159.py band",
        },
        "caveat": "114 municipal panos measure 0.791; 9 measure 0.771-0.789, and an "
                  "overlay of the measured line on those 9 (2026-10-02) shows the band "
                  "itself starting higher (two-wheeler captures, and one with no vehicle in "
                  "frame), not a measurement error. "
                  "One non-municipal pano carries a "
                  "different, green-and-white band that this method does not detect "
                  "(f8759625-...); its edge is read by hand (method 'manual', see "
                  "MANUAL_BANDS). The other non-municipal pano has none.",
        "labeler_band_y": LABELER_BAND_Y,
        "rig_mask_y": RIG_MASK_Y,
        "n_with_band": len(ys),
        "n_measured_by_hand": sum(v["method"] == "manual" for v in panos.values()),
        "n_without_band": len(panos) - len(ys),
        "band_top_y_min": ys[0] if ys else None,
        "band_top_y_median": ys[len(ys) // 2] if ys else None,
        "band_top_y_max": ys[-1] if ys else None,
        "panos": panos,
    }


def cmd_band(args):
    return _check_or_write(build_band(args.bundle), NADIR_BAND, args.write)


def load_band(path=NADIR_BAND):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["panos"]


# --------------------------------------------------------------------------- #
# firing
# --------------------------------------------------------------------------- #
def firing_rows(peaks_by_pano, strata, thresholds=FIRING_THRESHOLDS, interior_only=False):
    """Per stratum (+ ``all``): panos, peaks per pano and share of panos with >= 1
    peak, at each threshold. Pure."""
    groups = {"all": list(peaks_by_pano)}
    for s in STRATA:
        groups[s] = [p for p in peaks_by_pano if strata.get(p) == s]
    out = {}
    for g, pids in groups.items():
        rows = []
        for t in thresholds:
            counts = []
            for p in pids:
                pk = [q for q in peaks_by_pano[p] if q[2] >= t
                      and not (interior_only and in_border_ring(q[0], q[1]))]
                counts.append(len(pk))
            n = len(counts)
            rows.append({"threshold": t,
                         "peaks_per_pano": _r(sum(counts) / n) if n else None,
                         "share_panos_firing": _r(sum(c > 0 for c in counts) / n) if n else None})
        out[g] = {"panos": len(pids), "rows": rows}
    return out


def build_firing(bayonne_cache=BAYONNE_CACHE, op_cache_dir=OP_CACHE_DIR,
                 splits=COMPARISON_SPLITS):
    result = {"note": "GT-free. Peaks per pano from each split's low-floor cache, by "
                      "sampler stratum. 'interior' drops peaks within 10 heatmap cells of "
                      "any edge from EVERY split, which is what the pre-f4c71c8 extractor "
                      "behind ten of the eleven committed op_caches did (laurens_mapillary's "
                      "was built after the fix); read cross-split comparisons "
                      "off 'interior'. The 'random' stratum is conditioned on >= 1 "
                      "detection at 0.55 and 'empty' on none, so neither is a city rate.",
              "thresholds": list(FIRING_THRESHOLDS), "splits": {}}
    sources = [(SPLIT, bayonne_cache)] + [(s, os.path.join(op_cache_dir, f"{s}.json"))
                                          for s in splits]
    for split, path in sources:
        peaks, meta = load_peaks(path)
        strata = strata_of(split)
        result["splits"][split] = {
            "cache": os.path.relpath(path, REPO).replace(os.sep, "/"),
            "extractor_excludes_border": split != SPLIT,
            "raw": firing_rows(peaks, strata),
            "interior": firing_rows(peaks, strata, interior_only=True),
        }
    return result


def cmd_firing(args):
    res = build_firing()
    if args.print:
        _print_firing(res)
    return _check_or_write(res, os.path.join(OUT_DIR, "firing.json"), args.write)


def _row_at(block, t):
    return next(r for r in block["rows"] if abs(r["threshold"] - t) < 1e-9)


def _print_firing(res, ts=(0.10, 0.30, 0.55)):
    for stratum in ("random", "empty"):
        print(f"\n{stratum} stratum, interior peaks per pano (share of panos with >= 1)")
        print(f"{'split':<20} {'n':>4} " + " ".join(f"{'@' + str(t):>14}" for t in ts))
        for split, d in res["splits"].items():
            b = d["interior"][stratum]
            cells = []
            for t in ts:
                r = _row_at(b, t)
                cells.append(f"{r['peaks_per_pano']:>6.3f} ({r['share_panos_firing']:.2f})"
                             if r["peaks_per_pano"] is not None else f"{'-':>14}")
            print(f"{split:<20} {b['panos']:>4} " + " ".join(f"{c:>14}" for c in cells))


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #
#: GoPro Max splits (records camera_model contains "max"); richmond is mixed, so only
#: its GoPro Max panos are taken.
FRAME_SPLITS = ("bayonne", "laurens_mapillary", "morgantown", "budapest_district5",
                "richmond")
FRAME_THRESHOLDS = (0.30, 0.55)


def _quantiles(vals, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    if not vals:
        return {}
    v = sorted(vals)
    return {f"p{int(q * 100)}": _r(v[min(len(v) - 1, int(q * len(v)))]) for q in qs}


def frame_summary(ys, band_y=LABELER_BAND_Y, rig_y=RIG_MASK_Y):
    """Distribution of peak rows plus the shares in the rig mask and in the band. Pure."""
    n = len(ys)
    return {"n_peaks": n, "y": _quantiles(ys),
            "share_y_ge_rig_mask": _r(sum(y >= rig_y for y in ys) / n) if n else None,
            "share_y_ge_band": _r(sum(y >= band_y for y in ys) / n) if n else None,
            "share_y_ge_0_70": _r(sum(y >= 0.70 for y in ys) / n) if n else None}


def build_frame(bayonne_cache=BAYONNE_CACHE, op_cache_dir=OP_CACHE_DIR):
    out = {"note": "GT-free. Normalized y of every interior peak (border ring dropped, as "
                   "in firing) at each threshold, GoPro Max panos only. The band and rig "
                   "shares use y >= 0.791 (labeler LOGO_BAND_Y) and y >= 0.772 (labeler "
                   "NADIR_MASK_DEG 49); only bayonne's municipal panos actually carry the "
                   "band, so for the other splits that column is a counterfactual: the "
                   "share of their peaks a band like Bayonne's would have covered.",
           "splits": {}}
    for split in FRAME_SPLITS:
        path = bayonne_cache if split == SPLIT else os.path.join(op_cache_dir, f"{split}.json")
        peaks, _ = load_peaks(path)
        recs = {r["pano"]["panorama_id"]: r["pano"] for r in load_records(split)}
        keep = [p for p in peaks if "max" in str(recs.get(p, {}).get("camera_model", "")).lower()]
        entry = {"panos": len(keep)}
        for t in FRAME_THRESHOLDS:
            ys = [q[1] for p in keep for q in peaks[p]
                  if q[2] >= t and not in_border_ring(q[0], q[1])]
            entry[f"t{t:.2f}"] = frame_summary(ys)
        out["splits"][split] = entry
    return out


def cmd_frame(args):
    res = build_frame()
    if args.print:
        for split, e in res["splits"].items():
            for t in FRAME_THRESHOLDS:
                s = e[f"t{t:.2f}"]
                print(f"{split:<20} panos {e['panos']:>4}  t {t:.2f}  n {s['n_peaks']:>4}  "
                      f"y p50 {s['y'].get('p50')}  p90 {s['y'].get('p90')}  "
                      f">=0.70 {s['share_y_ge_0_70']}  rig {s['share_y_ge_rig_mask']}  "
                      f"band {s['share_y_ge_band']}")
    return _check_or_write(res, os.path.join(OUT_DIR, "frame.json"), args.write)


# --------------------------------------------------------------------------- #
# AI pre-read: the crops it is made on, and its summary
# --------------------------------------------------------------------------- #
PREREAD_DIR = os.path.join(OUT_DIR, "ai_preread")
#: Square crop side as a share of the NATIVE pano width: 0.09 x 5760 = 518 px,
#: 0.09 x 5376 = 484 px, shown 1:1. At the 0.022 match radius that frames about +/-2 R
#: horizontally, wide enough to see the kerb line on both sides of a lowering.
PREREAD_CROP_FRAC = 0.09
PREREAD_PER_SHEET = 6
PREREAD_LABELS = ("ramp", "not_ramp", "cant_tell")


def preread_items(recs, threshold=DEPLOYED):
    """Every detection the review will judge, in records order, with a stable id."""
    items = []
    for r in recs:
        pid = r["pano"]["panorama_id"]
        for i, d in enumerate(r["detections"]):
            if d["confidence"] < threshold:
                continue
            items.append({"item": f"{pid[:8]}_d{i}", "pano": pid, "det": i,
                          "x": _r(d["x_normalized"], 5), "y": _r(d["y_normalized"], 5),
                          "conf": _r(d["confidence"]),
                          "group": r.get("benchmark_group")})
    return items


def render_preread(bundle=BUNDLE, out=PREREAD_DIR):
    """Native-resolution crops (git-ignored) plus contact sheets of six, for viewing."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    items = preread_items(load_records(os.path.basename(bundle),
                                       os.path.dirname(os.path.dirname(bundle))))
    crops_dir = os.path.join(out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    opened, tiles = {}, []
    for it in items:
        if it["pano"] not in opened:
            opened.clear()
            opened[it["pano"]] = Image.open(os.path.join(bundle, "panos",
                                                         it["pano"] + ".jpg")).convert("RGB")
        im = opened[it["pano"]]
        W, H = im.size
        side = int(round(PREREAD_CROP_FRAC * W))
        cx, cy = it["x"] * W, it["y"] * H
        left = int(min(max(cx - side / 2, 0), W - side))
        top = int(min(max(cy - side / 2, 0), H - side))
        crop = im.crop((left, top, left + side, top + side))
        dr = ImageDraw.Draw(crop)
        px, py, rr = cx - left, cy - top, max(6, side // 40)
        dr.ellipse((px - rr, py - rr, px + rr, py + rr), outline=(255, 230, 0), width=2)
        crop.save(os.path.join(crops_dir, it["item"] + ".jpg"), quality=92)
        it["native_px"] = [W, H]
        it["crop_px"] = side
        tiles.append((it, crop))
    for s in range(0, len(tiles), PREREAD_PER_SHEET):
        chunk = tiles[s:s + PREREAD_PER_SHEET]
        side = max(c.size[0] for _, c in chunk)
        sheet = Image.new("RGB", (3 * side, 2 * (side + 22)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for k, (it, c) in enumerate(chunk):
            x0, y0 = (k % 3) * side, (k // 3) * (side + 22)
            sheet.paste(c, (x0, y0 + 22))
            dr.text((x0 + 4, y0 + 4), f"{s + k + 1}. {it['item']}  conf {it['conf']:.2f}  "
                                      f"y {it['y']:.3f}", fill=(0, 0, 0))
        sheet.save(os.path.join(out, f"sheet_{s // PREREAD_PER_SHEET + 1:02d}.jpg"), quality=90)
    return items


def cmd_preread_crops(args):
    items = render_preread()
    _dump({"note": "The 147 detections >= 0.55 the review will judge, as the AI pre-read "
                   "saw them: native-resolution square crops, side = "
                   f"{PREREAD_CROP_FRAC} x native width, marker at the peak. Crops and "
                   "sheets are git-ignored; this list is not.",
           "items": items}, os.path.join(PREREAD_DIR, "items.json"))
    print(f"{len(items)} crops -> {os.path.relpath(PREREAD_DIR, REPO)}")
    return 0


def preread_summary(labels, items):
    """Counts of the pre-read's labels, overall and by confidence band. Pure."""
    by_item = {it["item"]: it for it in items}
    missing = sorted(set(by_item) - set(labels))
    extra = sorted(set(labels) - set(by_item))
    if missing or extra:
        raise ValueError(f"labels/items disagree: {len(missing)} missing, {len(extra)} extra")
    bands = (("0.55-0.70", 0.55, 0.70), ("0.70-0.85", 0.70, 0.85), ("0.85-1.00", 0.85, 1.01))
    out = {"n": len(labels), "counts": {k: 0 for k in PREREAD_LABELS}, "by_conf": {}}
    for name, lo, hi in bands:
        out["by_conf"][name] = {k: 0 for k in PREREAD_LABELS}
    for item, lab in labels.items():
        out["counts"][lab["label"]] += 1
        c = by_item[item]["conf"]
        for name, lo, hi in bands:
            if lo <= c < hi:
                out["by_conf"][name][lab["label"]] += 1
    return out


# --------------------------------------------------------------------------- #
# checks: the pipeline reproduces a committed cache, and Bayonne's cache
# reproduces its committed records
# --------------------------------------------------------------------------- #
REPRO_CACHE = os.path.join(OUT_DIR, "repro_op_cache", "laurens_mapillary.json")


def _cells(peaks):
    return {(round(x * HEAT_W), round(y * HEAT_H)): s for x, y, s in peaks}


def repro_check(new, old):
    """Peak-cell agreement between a re-extracted cache and a committed one. Pure."""
    if set(new) != set(old):
        raise ValueError("the two caches cover different panos")
    same = only_new = only_old = ring_new = ring_old = 0
    max_ds = 0.0
    for p in sorted(new):
        a, b = _cells(new[p]), _cells(old[p])
        for k, s in a.items():
            if k in b:
                same += 1
                max_ds = max(max_ds, abs(s - b[k]))
            else:
                only_new += 1
                ring_new += in_border_ring(k[0] / HEAT_W, k[1] / HEAT_H)
        for k in b:
            if k not in a:
                only_old += 1
                ring_old += in_border_ring(k[0] / HEAT_W, k[1] / HEAT_H)
    return {"panos": len(new), "peaks_same_cell": same, "peaks_only_new": only_new,
            "peaks_only_new_in_border_ring": ring_new, "peaks_only_committed": only_old,
            "peaks_only_committed_in_border_ring": ring_old,
            "max_score_diff_same_cell": float(f"{max_ds:.2g}")}


def parity_extras(peaks, recs, threshold=DEPLOYED):
    """Panos whose cache count at the deployed threshold differs from records.jsonl,
    with where the extra peaks sit. Pure."""
    out = []
    for r in recs:
        pid = r["pano"]["panorama_id"]
        got = [q for q in peaks[pid] if q[2] >= threshold]
        rec = {(round(d["x_normalized"] * HEAT_W), round(d["y_normalized"] * HEAT_H))
               for d in r["detections"] if d["confidence"] >= threshold}
        extra = [q for q in got if (round(q[0] * HEAT_W), round(q[1] * HEAT_H)) not in rec]
        missing = len(rec) - (len(got) - len(extra))
        if extra or missing:
            out.append({"pano": pid, "group": r.get("benchmark_group"),
                        "records": len(rec), "cache": len(got), "missing_from_cache": missing,
                        "extra": [{"x": _r(q[0], 5), "y": _r(q[1], 5), "score": _r(q[2]),
                                   "in_border_ring": in_border_ring(q[0], q[1])}
                                  for q in extra]})
    return out


def build_checks():
    new, meta_new = load_peaks(REPRO_CACHE)
    old, _ = load_peaks(os.path.join(OP_CACHE_DIR, "laurens_mapillary.json"))
    peaks, meta = load_peaks(BAYONNE_CACHE)
    recs = load_records(SPLIT)
    extras = parity_extras(peaks, recs)
    n_rec = sum(1 for r in recs for d in r["detections"] if d["confidence"] >= DEPLOYED)
    n_cache = sum(1 for p in peaks.values() for q in p if q[2] >= DEPLOYED)
    return {
        "replication_control": {
            "what": "laurens_mapillary re-extracted on makelab2 with this branch's code "
                    "(scripts/analysis/bayonne_159_gpu.sh, step repro) against its committed "
                    "analysis_out/op_cache/laurens_mapillary.json",
            "re_extracted_on": meta_new.get("device"),
            **repro_check(new, old)},
        "bayonne_parity": {
            "what": "Bayonne cache peaks >= 0.55 against the 147 committed records.jsonl "
                    "detections (low_floor_sweep.py parity reads the same)",
            "records": n_rec, "cache": n_cache,
            "panos_differing": extras,
            "reading": "every extra cache peak sits in the 10-cell border ring at the 360 "
                       "seam (x = 0 or ~1). The labeler's production extractor "
                       "(sidewalk-auto-labeler detectors/curb_ramp.py) calls peak_local_max "
                       "without exclude_border=False, so it drops peaks within 10 cells of "
                       "the heatmap edge -- the extractor defect RampNet fixed in f4c71c8 "
                       "(#132). Off the seam the two agree cell for cell."
                       if extras and all(e["in_border_ring"] for x in extras
                                         for e in x["extra"])
                       and not any(x["missing_from_cache"] for x in extras) else
                       "differences are not all seam peaks; read panos_differing",
        },
    }


def cmd_checks(args):
    res = build_checks()
    if args.print:
        print(json.dumps(res, indent=1, sort_keys=True))
    return _check_or_write(res, os.path.join(OUT_DIR, "checks.json"), args.write)


# --------------------------------------------------------------------------- #
# candidates: where RampNet is silent and >= 2 challenger legs agree
# --------------------------------------------------------------------------- #
DETECTIONS_DIR = os.path.join(OUT_DIR, "model_detections")
#: Each leg's operating point for this list. YOLO: the #71 protocol's headline conf.
#: OWLv2 / Grounding DINO: the thresholds their best-F1 sweeps land on across the
#: published splits (docs/model_comparison.md: OWLv2 0.25 on most, Grounding DINO
#: 0.15). Qwen and Molmo emit no score, so every point counts.
CANDIDATE_LEGS = (
    ("y11l_pano", "y11l_pano", 0.25, "yolo"),
    ("y26_pano", "y26_pano", 0.25, "yolo"),
    ("y11x_pano_h200", "y11x_pano_h200", 0.25, "yolo"),
    ("google__owlv2-large-patch14-ensemble", "owlv2", 0.25, "open-vocab"),
    ("IDEA-Research__grounding-dino-base", "gdino", 0.15, "open-vocab"),
    ("Qwen__Qwen3-VL-8B-Instruct", "qwen3-vl-8b", None, "chat-vlm"),
    ("allenai__Molmo2-8B", "molmo2-8b", None, "chat-vlm"),
)
RAMPNET_SILENT_BELOW = 0.30     # the recommended operating point (docs/operating_point.md)
MIN_LEGS = 2
#: Legs that do NOT count toward agreement: the open-vocabulary detectors, which the
#: roster classes as dense (55-88 boxes/pano on the published splits; 5.5 and 27.4
#: points/pano here at their thresholds). Two dense legs land within one radius of
#: each other almost anywhere, so their agreement says nothing; with them counted,
#: most candidates are OWLv2 + Grounding DINO alone (``candidates.json``,
#: ``if_dense_legs_vote``). They are
#: reported beside each candidate as support, never as a vote.
SUPPORT_ONLY = ("owlv2", "gdino")


def _d2(a, b):
    from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y
    return ((a[0] - b[0]) * PANO_SCALE_X) ** 2 + ((a[1] - b[1]) * PANO_SCALE_Y) ** 2


def candidate_misses(rampnet_peaks, legs, radius_sq, band=None, min_legs=MIN_LEGS,
                     silent_below=RAMPNET_SILENT_BELOW, support_only=()):
    """Locations where RampNet has no peak >= ``silent_below`` within the match radius
    and at least ``min_legs`` distinct challenger legs put a point within it. Pure.

    ``legs`` is ``{leg: {pano: [(x, y), ...]}}`` already cut at each leg's operating
    point. Seeds are every challenger point in turn (in leg order, then point order);
    a seed's cluster is every leg with a point within one radius of it; clusters are
    reported once (a later seed within a radius of a reported one is skipped). Legs in
    ``support_only`` neither seed nor vote; they are listed as ``support``. The scorer
    does not wrap the seam, so neither does this."""
    out = []
    panos = sorted(set(rampnet_peaks) | {p for d in legs.values() for p in d})
    for pano in panos:
        rn = [(x, y) for x, y, s in rampnet_peaks.get(pano, []) if s >= silent_below]
        reported = []
        voters = [l for l in legs if l not in support_only]
        for leg in voters:
            for pt in legs[leg].get(pano, []):
                if any(_d2(pt, q) <= radius_sq for q in rn):
                    continue
                if any(_d2(pt, c) <= radius_sq for c in reported):
                    continue
                near = [l2 for l2 in legs
                        if any(_d2(pt, q) <= radius_sq for q in legs[l2].get(pano, []))]
                agree = sorted(l2 for l2 in near if l2 not in support_only)
                support = sorted(l2 for l2 in near if l2 in support_only)
                if len(agree) < min_legs:
                    continue
                reported.append(pt)
                bmax = [s for x, y, s in rampnet_peaks.get(pano, []) if _d2(pt, (x, y)) <= radius_sq]
                out.append({"pano": pano, "x": _r(pt[0], 4), "y": _r(pt[1], 4), "legs": agree,
                            "n_legs": len(agree), "support": support,
                            "rampnet_best_peak_within_radius": _r(max(bmax)) if bmax else None,
                            "in_nadir_band": bool(band is not None and band.get(pano) is not None
                                                  and pt[1] >= band[pano])})
    return out


def load_leg_points(stem, threshold, split=SPLIT, d=DETECTIONS_DIR):
    path = os.path.join(d, f"{stem}__{split}.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    pts = {}
    for pano, dets in payload["detections"].items():
        pts[pano] = [(t[0], t[1]) for t in dets
                     if threshold is None or (len(t) > 2 and t[2] is not None and t[2] >= threshold)]
    return pts, payload


def build_candidates():
    from rampnet.detection_eval import radius_sq_for
    peaks, _ = load_peaks(BAYONNE_CACHE)
    legs, ran, not_run = {}, [], []
    for stem, name, thr, family in CANDIDATE_LEGS:
        path = os.path.join(DETECTIONS_DIR, f"{stem}__{SPLIT}.json")
        if not os.path.exists(path):
            not_run.append(name)
            continue
        pts, payload = load_leg_points(stem, thr)
        legs[name] = pts
        ran.append({"leg": name, "family": family, "threshold": thr,
                    "n_panos": payload["n_panos"], "n_uncached": payload["n_uncached"],
                    "points_per_pano": _r(sum(len(v) for v in pts.values()) / len(pts))})
    band = {p: e["band_top_y"] for p, e in load_band().items()}
    strata = strata_of(SPLIT)
    cands = candidate_misses(peaks, legs, radius_sq_for(), band, support_only=SUPPORT_ONLY)
    for c in cands:
        c["group"] = strata.get(c["pano"])
    # The counterfactual that justifies SUPPORT_ONLY: let the dense legs vote too.
    voting = candidate_misses(peaks, legs, radius_sq_for(), band)
    dense_only = sum(1 for c in voting if set(c["legs"]) <= set(SUPPORT_ONLY))
    lt2_sparse = sum(1 for c in voting if len(set(c["legs"]) - set(SUPPORT_ONLY)) < MIN_LEGS)
    return {
        "what": "Reviewer-attention list, NOT ground truth: locations where RampNet has no "
                f"peak >= {RAMPNET_SILENT_BELOW} within the 0.022 match radius and >= "
                f"{MIN_LEGS} challenger legs agree within it, not counting the dense "
                "open-vocabulary legs (OWLv2, Grounding DINO), which are listed as support "
                "only. Read after judging a pano, as a second sweep, so it does not steer "
                "the first look.",
        "legs_run": ran, "legs_not_run": not_run,
        "n_candidates": len(cands),
        "support_only": list(SUPPORT_ONLY),
        "if_dense_legs_vote": {"n_candidates": len(voting),
                               "n_from_dense_legs_only": dense_only,
                               "n_with_fewer_than_2_non_dense_legs": lt2_sparse},
        "n_in_nadir_band": sum(c["in_nadir_band"] for c in cands),
        "candidates": cands,
    }


def cmd_candidates(args):
    res = build_candidates()
    if args.print:
        print({k: v for k, v in res.items() if k != "candidates"})
    return _check_or_write(res, os.path.join(OUT_DIR, "candidates.json"), args.write)


# --------------------------------------------------------------------------- #
# paid legs: NOT run; their expected cost from the ledger's measured rows
# --------------------------------------------------------------------------- #
LEDGER = os.path.join(REPO, "analysis_out", "usage_log.jsonl")
#: The paid legs a full Bayonne scoreboard row would need: the two scored Gemini legs,
#: the published-but-unscored 3.7-flash, and the one complete Claude leg (opus-5 at
#: effort low, eleven splits). (provider, label, effort or None)
PAID_LEGS = (("gemini", "gemini-3.6-flash", None), ("gemini", "gemini-3.1-pro-preview", None),
             ("gemini", "gemini-3.7-flash", None), ("claude", "claude-opus-5", "low"))


#: Rows written after this are ignored, so a later paid run of the same legs does not
#: silently move a committed estimate.
LEDGER_CUTOFF = "2026-10-02T00:00:00Z"


def paid_estimate(rows, n_panos, legs=PAID_LEGS, cutoff=LEDGER_CUTOFF):
    """Expected dollars per leg = measured dollars per pano x ``n_panos``. Pure.

    Uses only measured rows (``kind`` absent; recovered rows carry no pano count). A
    row's denominator is the panos that actually reached the API: ``panos_called``
    when the row has it, else ``calls`` / the number of perspective views in its
    signature (one call per view). Never ``panos_scored``: on a partly cached re-run
    it counts cache hits that cost nothing (one claude-opus-5 row says 94 panos and
    made 12 calls), which understated the per-pano rate (PR #234 review, S1). A row
    with neither is skipped and counted in ``rows_without_denominator``."""
    out = []
    for provider, label, effort in legs:
        n = usd = 0.0
        k = skipped = 0
        for r in rows:
            if r.get("kind") == "recovered" or r.get("provider") != provider:
                continue
            if (r.get("ts") or "") >= cutoff:
                continue
            if r.get("label") != label or not r.get("est_cost_usd"):
                continue
            if effort is not None and (r.get("signature") or {}).get("effort") != effort:
                continue
            views = (r.get("signature") or {}).get("views") or []
            p = r.get("panos_called") or (r["calls"] / len(views)
                                          if r.get("calls") and views else 0)
            if not p:
                skipped += 1
                continue
            n += p
            usd += r["est_cost_usd"]
            k += 1
        out.append({"provider": provider, "label": label, "effort": effort,
                    "ledger_rows": k, "ledger_panos": _r(n, 2),
                    "rows_without_denominator": skipped,
                    "usd_per_pano": _r(usd / n, 5) if n else None,
                    "expected_usd": _r(usd / n * n_panos, 2) if n else None})
    return out


def build_paid_estimate():
    with open(LEDGER, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    legs = paid_estimate(rows, 125)
    return {"what": "Paid legs NOT run on Bayonne (no spend today). Expected cost = the "
                    "ledger's measured dollars per pano for that leg (rows before "
                    f"{LEDGER_CUTOFF}) x 125 panos. Dollars are estimates; the billing "
                    "console is authoritative. A row's panos are the ones that reached the API "
                    "(panos_called, else calls / views), never panos_scored, which "
                    "counts cache hits. The rates rest on 155-184 called panos per leg, "
                    "and Gemini's thinking spend varies by imagery.",
            "n_panos": 125, "legs": legs,
            "expected_usd_total": _r(sum(l["expected_usd"] or 0 for l in legs), 2)}


def cmd_paid_estimate(args):
    res = build_paid_estimate()
    if args.print:
        print(json.dumps(res, indent=1))
    return _check_or_write(res, os.path.join(OUT_DIR, "paid_legs_estimate.json"), args.write)


PREREAD_LABELS_FILE = os.path.join(PREREAD_DIR, "preread__claude-opus-5-5.json")


def build_preread_summary():
    with open(os.path.join(PREREAD_DIR, "items.json"), encoding="utf-8") as f:
        items = json.load(f)["items"]
    with open(PREREAD_LABELS_FILE, encoding="utf-8") as f:
        pre = json.load(f)
    s = preread_summary(pre["labels"], items)
    s["rater"] = pre["rater"]
    s["note"] = ("Counts of a MODEL's labels on native-resolution crops. Not ground truth "
                 "and not a precision; see the rubric in the labels file. Reviewer: do "
                 "not read this before your verdicts.json is exported.")
    return s


def cmd_preread_summary(args):
    return _check_or_write(build_preread_summary(),
                           os.path.join(PREREAD_DIR, "summary.json"), args.write)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="bundle integrity against index.csv")
    v.add_argument("--bundle", default=BUNDLE)
    v.set_defaults(func=cmd_verify)
    b = sub.add_parser("band", help="measure the nadir band per pano (needs panos/)")
    b.add_argument("--bundle", default=BUNDLE)
    b.add_argument("--write", action="store_true")
    b.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    b.set_defaults(func=cmd_band)
    f = sub.add_parser("firing", help="peaks per pano vs threshold, by stratum")
    f.add_argument("--write", action="store_true")
    f.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    f.add_argument("--print", action="store_true")
    f.set_defaults(func=cmd_firing)
    fr = sub.add_parser("frame", help="where peaks sit in the frame, GoPro Max splits")
    fr.add_argument("--write", action="store_true")
    fr.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    fr.add_argument("--print", action="store_true")
    fr.set_defaults(func=cmd_frame)
    cd = sub.add_parser("candidates", help="RampNet silent, >= 2 challenger legs agree")
    cd.add_argument("--write", action="store_true")
    cd.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    cd.add_argument("--print", action="store_true")
    cd.set_defaults(func=cmd_candidates)
    pe = sub.add_parser("paid-estimate", help="expected cost of the paid legs not run")
    pe.add_argument("--write", action="store_true")
    pe.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    pe.add_argument("--print", action="store_true")
    pe.set_defaults(func=cmd_paid_estimate)
    ck = sub.add_parser("checks", help="replication control + Bayonne parity detail")
    ck.add_argument("--write", action="store_true")
    ck.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    ck.add_argument("--print", action="store_true")
    ck.set_defaults(func=cmd_checks)
    pc = sub.add_parser("preread-crops", help="render the AI pre-read's crops (needs panos/)")
    pc.set_defaults(func=cmd_preread_crops)
    ps = sub.add_parser("preread-summary", help="counts of the AI pre-read's labels")
    ps.add_argument("--write", action="store_true")
    ps.add_argument("--check", action="store_true",
                   help="compare against the committed file (the default without --write)")
    ps.set_defaults(func=cmd_preread_summary)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
