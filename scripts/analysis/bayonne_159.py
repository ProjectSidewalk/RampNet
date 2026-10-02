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

**The extractor difference that the firing read controls for.** Every committed
``analysis_out/op_cache/*.json`` was built before f4c71c8, when ``peaks_to_dets`` left
``skimage``'s ``exclude_border`` at its default and so dropped every peak within
``min_distance`` (10 heatmap cells) of the array edge. Bayonne's cache is built with
the fixed extractor. So ``firing`` reports two columns per split: ``raw`` (as cached)
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
    """List of problems with the staged bundle (empty = OK). The byte check runs only
    for panos present on disk; a missing ``panos/`` is reported, not failed, so the
    record-level checks still run on a clean clone."""
    problems, notes = [], []
    with open(os.path.join(bundle, "index.csv"), encoding="utf-8", newline="") as f:
        index = {row["panorama_id"]: row for row in csv.DictReader(f)}
    recs = load_records(os.path.basename(bundle), os.path.dirname(os.path.dirname(bundle)))
    rec_ids = [r["pano"]["panorama_id"] for r in recs]
    if len(set(rec_ids)) != len(rec_ids):
        problems.append("duplicate panorama_id in records.jsonl")
    if set(rec_ids) != set(index):
        problems.append(f"records.jsonl and index.csv disagree: "
                        f"{len(set(rec_ids) - set(index))} only in records, "
                        f"{len(set(index) - set(rec_ids))} only in index")
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
        for pid, row in sorted(index.items()):
            path = os.path.join(panos_dir, row["filename"])
            if not os.path.exists(path):
                problems.append(f"{pid}: missing from panos/")
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != row["sha256"] or os.path.getsize(path) != int(row["bytes"]):
                problems.append(f"{pid}: bytes differ from index.csv")
            checked += 1
        notes.append(f"{checked} panos byte-checked against index.csv")
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
        panos[pid] = {"band_top_y": _r(y), "copyright": r["pano"].get("copyright")}
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
                  "(f8759625-...); the other non-municipal pano has none.",
        "labeler_band_y": LABELER_BAND_Y,
        "rig_mask_y": RIG_MASK_Y,
        "n_with_band": len(ys),
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
                      "behind the committed op_caches did; read cross-split comparisons "
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
    b.set_defaults(func=cmd_band)
    f = sub.add_parser("firing", help="peaks per pano vs threshold, by stratum")
    f.add_argument("--write", action="store_true")
    f.add_argument("--print", action="store_true")
    f.set_defaults(func=cmd_firing)
    fr = sub.add_parser("frame", help="where peaks sit in the frame, GoPro Max splits")
    fr.add_argument("--write", action="store_true")
    fr.add_argument("--print", action="store_true")
    fr.set_defaults(func=cmd_frame)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
