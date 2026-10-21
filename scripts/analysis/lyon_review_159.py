"""Lyon label audit (#159, #263): stage the review bundle, pin the plan, score it per group.

Lyon is a **training-only** candidate city, never a benchmark split. On 2026-10-06 Jon
decided that its self-labels train only after a quality audit: about 150-200 panos
from the 5 m run of record, stratified by rig x producer, judged the same way as a
benchmark session, with the pass bar posted on #159 before anyone reviews. The audited
panos, and every run pano within 30 m of one, are held out of every training pool.
``docs/lyon_review_159.md`` is the write-up.

**Nothing here produces a verdict.** The review is Jon's. ``plan`` pins the design and
the pass bar before the review; ``score`` refuses to run until ``verdicts.json`` exists,
and refuses a plan that changed after it was posted.

Subcommands, in the order the doc runs them:

    # copy the exporter's bundle in (masking email-shaped tokens, #149), and write
    # ids.txt, exclude_ids.txt (the 30 m buffer), and the run's stored peaks for the
    # audited panos (analysis_out/lyon_review_159/run_peaks.json, outside the bundle)
    python scripts/analysis/lyon_review_159.py stage --from D:/Git/labeler-wt/lyon-bundle \\
        --results <runs/lyon/results.jsonl>

    # the audit plan: strata, group table and pass bar -> benchmark/lyon/audit_plan.json
    python scripts/analysis/lyon_review_159.py plan --write

    # the known-artifact reads from #159 on the bundle's stored peaks
    # -> analysis_out/lyon_review_159/artifacts.json
    python scripts/analysis/lyon_review_159.py artifacts --results <runs/lyon/results.jsonl> --write

    # bottom-of-frame overlays measured per pano (needs panos/) -> benchmark/lyon/nadir_band.json
    python scripts/analysis/lyon_review_159.py band --write

    # bundle integrity; exits non-zero until the imagery is staged and pinned
    python scripts/analysis/lyon_review_159.py verify

    # after the review only: per-group precision / recall at 0.55 and 0.30 against the bar
    python scripts/analysis/lyon_review_159.py score --write

Every ``--write`` output is LF, sorted, with floats rounded, so a re-run on any OS is
byte-identical; without ``--write`` the command recomputes and compares the committed
file's **bytes**. ``tests/test_lyon_review_159.py`` covers the CPU logic.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

BUNDLE = os.path.join(REPO, "benchmark", "lyon")
OUT_DIR = os.path.join(REPO, "analysis_out", "lyon_review_159")
PLAN_NAME = "audit_plan.json"
NADIR_BAND_NAME = "nadir_band.json"
ARTIFACTS = os.path.join(OUT_DIR, "artifacts.json")
#: The run's stored peaks for the audited panos. Kept out of benchmark/lyon/ so that the
#: bundle the reviewer opens holds only what the gallery shows (re-review N4).
RUN_PEAKS = os.path.join(OUT_DIR, "run_peaks.json")
SCORE = os.path.join(OUT_DIR, "score.json")

STRATA = ("top", "random", "empty")
#: The four audit groups of Jon's 2026-10-06 decision on #159, in reading order.
GROUPS = ("grand_lyon_ladybug", "ecartip_gopro_max", "other_gopro_max", "other_cameras")
GROUP_TITLES = {
    "grand_lyon_ladybug": "Grand Lyon 8192x4096 (Point Grey Ladybug, incl. the make-less "
                          "batches; both account spellings)",
    "ecartip_gopro_max": "ecartip, GoPro Max",
    "other_gopro_max": "other GoPro Max accounts",
    "other_cameras": "long tail of other cameras",
}
DEPLOYED = 0.55          # the shipped threshold; the pass bar is read here
LOW_TIER = 0.30          # the recommended operating point; reported alongside
PASS_PRECISION = 0.80    # the #158 build line
HEAT_W, HEAT_H = 1024, 512
EXCLUDE_BUFFER_M = 30    # no training pano within this distance of an audited one
BOOT_REPS = 10000
BOOT_SEED = 159

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
#: An email-shaped token inside JSON or plain text: no whitespace, quote, bracket or
#: separator on either side of the @, and a dot in the domain. Non-ASCII local parts are
#: covered (sidewalk-auto-labeler PR 150's census regex is the looser
#: ``[^@\\s]+@[^@\\s]+\\.[^@\\s]+``, which would swallow the JSON quotes around a value).
_TOK = r"""[^\s@"'<>,;:()\[\]{}|]"""
EMAIL_RE = re.compile(_TOK + r"+@" + _TOK + r"+\." + _TOK + r"+")


def mask_email(text):
    """Mask every email-shaped token, as the labeler's census does (issue #149).

    Panoramax serves producer names publicly and a few accounts use an email address.
    The repo never republishes one: the token keeps its first character only.

    Example:
        >>> mask_email('{"copyright": "jane.doe@example.org"}')
        '{"copyright": "j***@***"}'
        >>> mask_email('grand lyon, rampnet-model@606a11956743')
        'grand lyon, rampnet-model@606a11956743'
    """
    return EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@***", text)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def _check_or_write(obj, path, write):
    """Write ``obj`` (LF, sorted) or compare the committed file's bytes to it."""
    data = _dumps(obj).encode("utf-8")
    try:
        rel = os.path.relpath(path, REPO)
    except ValueError:          # another drive (a test's tmp dir on Windows)
        rel = path
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {rel}")
        return 0
    if not os.path.exists(path):
        print(f"MISSING {rel} (run with --write)")
        return 1
    with open(path, "rb") as f:
        same = f.read() == data
    print(f"{'OK' if same else 'DIFFERS'}: {rel}")
    return 0 if same else 1


def _write_lines(path, lines):
    with open(path, "wb") as f:
        f.write("".join(line + "\n" for line in lines).encode("utf-8"))


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_records(bundle=BUNDLE):
    with open(os.path.join(bundle, "records.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run_peaks(path=RUN_PEAKS):
    """``{pano_id: [peak, ...]}``: every peak the run stored (>= 0.10) for the bundle."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["panos"]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k of n, or (None, None) when n is 0."""
    from rampnet.validation import wilson_interval
    return wilson_interval(k, n, z) if n else (None, None)


def rig_of(pano):
    """'make model WxH' with absent EXIF written as '(no make/model)'."""
    mm = " ".join(str(pano.get(k)) for k in ("camera_make", "camera_model") if pano.get(k))
    return f"{mm or '(no make/model)'} {pano.get('width')}x{pano.get('height')}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_results(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _order(g):
    return (GROUPS + (g,)).index(g)


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
STAGED_FILES = ("records.jsonl", "sample.json", "group_map.json")
#: Every committed text file of the bundle that ``verify`` scans for email-shaped tokens.
TEXT_FILES = STAGED_FILES + ("ids.txt", "exclude_ids.txt", PLAN_NAME,
                             "bundle_provenance.json", NADIR_BAND_NAME)


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def exclusion_ids(run_points, audited, buffer_m=EXCLUDE_BUFFER_M):
    """Ids of every run pano within ``buffer_m`` of an audited pano (the audited included).

    ``run_points``: iterable of (pano_id, lat, lng); ``audited``: {pano_id: (lat, lng)}.
    A grid of ``buffer_m`` cells keeps it to a handful of distance checks per pano. A run
    pano without a position is excluded only if it is itself audited. Pure."""
    cell = buffer_m / 111320.0
    grid = collections.defaultdict(list)
    for lat, lng in audited.values():
        grid[(math.floor(lat / cell), math.floor(lng / cell))].append((lat, lng))
    out = set(audited)
    for pid, lat, lng in run_points:
        if lat is None or lng is None:
            continue
        # lng cells are narrower in metres than lat cells, so look two lng cells out.
        ky, kx = math.floor(lat / cell), math.floor(lng / cell)
        span = max(1, math.ceil(1 / max(0.05, math.cos(math.radians(lat)))))
        for dy in (-1, 0, 1):
            for dx in range(-span, span + 1):
                if any(haversine_m(lat, lng, a, b) <= buffer_m
                       for a, b in grid.get((ky + dy, kx + dx), ())):
                    out.add(pid)
                    break
            else:
                continue
            break
    return out


def stage(src, results, dst=BUNDLE, buffer_m=EXCLUDE_BUFFER_M, peaks_path=RUN_PEAKS):
    """Copy the exporter's bundle files into ``dst``, masking email-shaped tokens in every
    file written, and derive the never-train lists and the run's stored peaks.

    Writes ``records.jsonl``, ``sample.json``, ``group_map.json`` (masked copies),
    ``ids.txt`` (the audited ids, sorted), ``exclude_ids.txt`` (every run pano within
    ``buffer_m`` of an audited one, sorted), and ``peaks_path`` (each audited pano's
    stored peaks, >= 0.10, for the 0.30-tier read; outside the bundle). Returns a summary dict. ``panos/``
    and ``index.csv`` are not copied: they are git-ignored and travel with the imagery."""
    os.makedirs(dst, exist_ok=True)
    masked = {}
    for name in STAGED_FILES:
        with open(os.path.join(src, name), encoding="utf-8", newline="") as f:
            text = f.read()
        new = mask_email(text)
        masked[name] = sum(a != b for a, b in zip(text.splitlines(), new.splitlines()))
        with open(os.path.join(dst, name), "wb") as f:
            f.write(new.encode("utf-8"))
    recs = load_records(dst)
    audited = {r["pano"]["panorama_id"]: (r["pano"]["lat"], r["pano"]["lng"]) for r in recs}
    ids = sorted(audited)
    _write_lines(os.path.join(dst, "ids.txt"), ids)
    points, peaks = [], {}
    for rec in _iter_results(results):
        p = rec["pano"]
        points.append((p["panorama_id"], p.get("lat"), p.get("lng")))
        if p["panorama_id"] in audited:
            peaks[p["panorama_id"]] = [
                {"x_normalized": d["x_normalized"], "y_normalized": d["y_normalized"],
                 "confidence": round(d["confidence"], 6)} for d in rec.get("detections") or []]
    missing = set(ids) - set(peaks)
    if missing:
        raise ValueError(f"{len(missing)} audited panos are not in {results}")
    excl = sorted(exclusion_ids(points, audited, buffer_m))
    _write_lines(os.path.join(dst, "exclude_ids.txt"), excl)
    run_sha = _sha256(results)
    os.makedirs(os.path.dirname(peaks_path), exist_ok=True)
    with open(peaks_path, "wb") as f:
        f.write(_dumps({"what": "every peak the run stored (>= 0.10; seam ring dropped by "
                                "production) for each audited pano, for the 0.30-tier read. "
                                "Do not open before verdicts.json is exported.",
                        "results_jsonl_sha256": run_sha,
                        "panos": {pid: peaks[pid] for pid in ids}}).encode("utf-8"))
    return {"masked_lines": masked, "ids": len(ids), "exclude_ids": len(excl),
            "buffer_m": buffer_m, "run_panos": len(points), "results_jsonl_sha256": run_sha}


def cmd_stage(args):
    out = stage(args.src, args.results, args.bundle, args.buffer)
    out["run_peaks_sha256"] = _sha256(RUN_PEAKS)
    print(json.dumps(out, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def group_table(recs, run_peaks):
    """Per audit group and stratum: panos, record detections (>= 0.55) and the run's
    stored peaks >= 0.30. Pure."""
    rows = {}
    for r in recs:
        g, s = r["benchmark_stratum"], r["benchmark_group"]
        row = rows.setdefault(g, {f"{k}_{s2}": 0 for s2 in STRATA
                                  for k in ("panos", "det055", "peaks030")})
        row[f"panos_{s}"] += 1
        row[f"det055_{s}"] += sum(d["confidence"] >= DEPLOYED for d in r["detections"])
        row[f"peaks030_{s}"] += sum(d["confidence"] >= LOW_TIER
                                    for d in run_peaks[r["pano"]["panorama_id"]])
    return {g: rows[g] for g in sorted(rows, key=_order)}


def rig_breakdown(recs):
    """Panos per (group, rig, producer, capture year), the descriptive rows. Pure."""
    c = collections.Counter()
    for r in recs:
        p = r["pano"]
        c[(r["benchmark_stratum"], rig_of(p), str(p.get("copyright")),
           (p.get("capture_date") or "")[:4])] += 1
    return [{"group": g, "rig": rig, "producer": prod, "year": y, "panos": n}
            for (g, rig, prod, y), n in sorted(c.items(), key=lambda kv: (
                _order(kv[0][0]), -kv[1], kv[0]))]


PASS_BAR = {
    "metric": "precision of the detections at confidence >= 0.55 (the records the "
              "reviewer judges)",
    "threshold": PASS_PRECISION,
    "read_on": "each audit group's unbiased subset: the random and empty strata, the top "
               "stratum excluded as for every split",
    "rule": "a group is eligible for a Lyon training pool when its point estimate is "
            ">= 0.80; below 0.80 it is not eligible",
    "intervals": [f"95% percentile interval of a pano bootstrap ({BOOT_REPS} resamples of the "
                  f"group's random panos with replacement, seed {BOOT_SEED})",
                  "95% Wilson interval over detections"],
    "decisive": "information for Jon, not a separate outcome: a reading is decisive when "
                "both intervals sit on the same side of 0.80 as the point estimate, and "
                "'not decisive' when either contains 0.80. score then prints the random "
                "panos a top-up draw would need, capped at the group's remaining pool",
    "decisive_fail": "both upper bounds < 0.80: the group is excluded from any Lyon "
                     "training pool",
    "all_row": "reported, no verdict: the floors over-sample the small groups, so the pool "
               "is not a city rate",
    "source": "the #158 build line (>= 0.80 build, 0.50-0.80 visibility, < 0.50 drop)",
    "reported_alongside": ["recall at 0.55", "precision and recall at 0.30, read the "
                           "standard way (the run's stored peaks matched to the GT, so a "
                           "sub-threshold peak on a ramp the reviewer marked missed is a TP)",
                           "unsure and duplicate counts", "per-rig rows inside each group (descriptive, no bar)"],
    "duplicates": "of two detections on one ramp, the lower-confidence one is the duplicate; "
                  "scored as a false positive (rampnet.validation default)",
}


def build_plan(bundle=BUNDLE):
    recs = load_records(bundle)
    peaks = load_run_peaks()
    with open(os.path.join(bundle, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    strata = sample["stratified"]["strata"]
    table = group_table(recs, peaks)
    for g, row in table.items():
        # The run's own panos per group: with >= 1 detection at 0.55 (the sampler's pool
        # plus the top panos it drew first) and without.
        row["run_panos_with_det055"] = strata[g]["pool_with_detections"] + strata[g]["top"]
        row["run_panos_empty"] = strata[g]["pool_empty"]
    return {
        "what": "Lyon label audit (#159 decision of 2026-10-06, #263 track 2). A working "
                "draft, posted on #159 before the review; Jon may amend it before the review "
                "opens, and score refuses a plan whose hash differs from the posted one. "
                "Lyon is not a benchmark split, and these panos never train.",
        "question": "Are the RampNet self-labels on each Lyon rig x producer group good "
                    "enough to train on?",
        "pass_bar": PASS_BAR,
        "groups": {g: GROUP_TITLES.get(g, g) for g in table},
        "group_table": table,
        "rig_breakdown": rig_breakdown(recs),
        "sampler": {k: sample[k] for k in ("seed", "sample", "empty_sample", "min_spacing_m",
                                           "min_confidence", "record_confidence", "top_n",
                                           "selected", "groups", "source_record_count")
                    if k in sample},
        "sampler_floors": {k: sample["stratified"][k] for k in ("group_min", "empty_group_min")},
        "records_show_detections_down_to": sample.get("record_confidence", DEPLOYED),
        "never_train": f"Every id in ids.txt is held out of every training pool, and so is "
                       f"every run pano within {EXCLUDE_BUFFER_M} m of one (exclude_ids.txt), "
                       f"whatever the audit reads.",
    }


def cmd_plan(args):
    return _check_or_write(build_plan(args.bundle), os.path.join(args.bundle, PLAN_NAME),
                           args.write)


def print_table(plan):
    """The group table as markdown (what the doc and the #159 comment quote)."""
    head = ("| group | run panos with a 0.55 detection / without | top | random | empty "
            "| detections ≥ 0.55 (random) | stored peaks ≥ 0.30 (random + empty) |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|"]
    tot = collections.Counter()
    table = plan["group_table"]
    for g in sorted(table, key=_order):
        row = table[g]
        cells = [row["run_panos_with_det055"], row["run_panos_empty"], row["panos_top"],
                 row["panos_random"], row["panos_empty"], row["det055_random"],
                 row["peaks030_random"] + row["peaks030_empty"]]
        tot.update(dict(enumerate(cells)))
        lines.append(f"| {g} | {cells[0]:,} / {cells[1]:,} | " +
                     " | ".join(str(c) for c in cells[2:]) + " |")
    lines.append(f"| **all** | {tot[0]:,} / {tot[1]:,} | " +
                 " | ".join(f"**{tot[i]}**" for i in range(2, 7)) + " |")
    return "\n".join(lines)


def cmd_table(args):
    with open(os.path.join(args.bundle, PLAN_NAME), encoding="utf-8") as f:
        print(print_table(json.load(f)))
    return 0


# --------------------------------------------------------------------------- #
# artifacts (#159's known artifacts, read on the bundle's stored peaks)
# --------------------------------------------------------------------------- #
FIXED_CELL_MIN_PANOS = 5     # a 1024x512 cell repeating in >= 5 panos of one sequence


def cell_of(det):
    """The heatmap cell (column, row) of a detection on the 1024x512 grid."""
    return (min(HEAT_W - 1, int(round(det["x_normalized"] * HEAT_W))),
            min(HEAT_H - 1, int(round(det["y_normalized"] * HEAT_H))))


def sequence_cells(results_lines, sequences, tier=LOW_TIER):
    """{sequence_id: Counter(cell -> panos with a >= tier peak there)}, streamed. Pure
    over an iterable of results.jsonl records (dicts)."""
    out = {s: collections.Counter() for s in sequences}
    for rec in results_lines:
        seq = rec["pano"].get("sequence_id")
        if seq not in out:
            continue
        cells = {cell_of(d) for d in rec.get("detections") or [] if d["confidence"] >= tier}
        out[seq].update(cells)
    return out


def build_artifacts(recs, seq_cells, run_sha, peaks_by_pid=None):
    """Counts per audit group of the two artifacts #159 named. Pure.

    Read on each pano's stored peaks (``peaks_by_pid``; default the record detections):
    * fixed-pixel cells: peaks >= 0.30 in a cell that repeats in >= 5 panos of the pano's
      own sequence (the Grand Lyon cluster: x 0.503, y 0.756, just above the black nadir);
    * above the horizon: peaks with y < 0.5 (ecartip: 10.3% at 0.55 in the first slice,
      with pose reported as exactly 0/0, so possibly unlevelled panos).
    """
    per, items = {}, []
    for r in recs:
        g, pid = r["benchmark_stratum"], r["pano"]["panorama_id"]
        dets = peaks_by_pid[pid] if peaks_by_pid is not None else r["detections"]
        cnt = seq_cells.get(r["pano"].get("sequence_id"), collections.Counter())
        row = per.setdefault(g, {"peaks030": 0, "peaks055": 0, "fixed_cell_030": 0,
                                 "fixed_cell_055": 0, "above_horizon_030": 0,
                                 "above_horizon_055": 0})
        for i, d in enumerate(dets):
            if d["confidence"] < LOW_TIER:
                continue
            hi = d["confidence"] >= DEPLOYED
            fixed = cnt[cell_of(d)] >= FIXED_CELL_MIN_PANOS
            above = d["y_normalized"] < 0.5
            row["peaks030"] += 1
            row["peaks055"] += hi
            row["fixed_cell_030"] += fixed
            row["fixed_cell_055"] += fixed and hi
            row["above_horizon_030"] += above
            row["above_horizon_055"] += above and hi
            if fixed or above:
                items.append({"pano": pid, "peak": i, "group": g,
                              "confidence": _r(d["confidence"]), "x": _r(d["x_normalized"]),
                              "y": _r(d["y_normalized"]), "fixed_cell": fixed,
                              "above_horizon": above,
                              "cell_panos_in_sequence": cnt[cell_of(d)]})
    return {"what": "GT-free counts of the #159 known artifacts on the bundle's stored peaks. "
                    "DO NOT open `items` before verdicts.json is exported: it names "
                    "places you are about to judge.",
            "definitions": {"fixed_cell": f"peak in a {HEAT_W}x{HEAT_H} cell that holds "
                                          f"a >= {LOW_TIER} peak in >= {FIXED_CELL_MIN_PANOS} "
                                          f"panos of its own sequence (the pano included)",
                            "above_horizon": "y_normalized < 0.5"},
            "results_jsonl_sha256": run_sha,
            "per_group": {g: per[g] for g in sorted(per, key=_order)},
            "items": sorted(items, key=lambda it: (it["pano"], it["peak"]))}


def cmd_artifacts(args):
    recs = load_records(args.bundle)
    seqs = {r["pano"].get("sequence_id") for r in recs}
    cells = sequence_cells(_iter_results(args.results), seqs)
    return _check_or_write(build_artifacts(recs, cells, _sha256(args.results),
                                           load_run_peaks()),
                           ARTIFACTS, args.write)


# --------------------------------------------------------------------------- #
# band: opaque regions at the bottom of the frame
# --------------------------------------------------------------------------- #
#: A row belongs to the overlay when this share of its pixels qualifies.
BAND_ROW_FRAC = 0.98
BLACK_MAX = 24           # every channel at or below this counts as black (JPEG noise)
WHITE_MIN = 235          # every channel above this counts as white (Bayonne's logo band)
BAND_START_ROW = int(0.97 * HEAT_H)
LABELS = {"black": "black nadir (no camera)", "white": "white nadir logo band"}


def band_top_from_rows(frac, start_row=BAND_START_ROW, need=BAND_ROW_FRAC, h=HEAT_H):
    """Normalized y of an overlay's top edge from per-row qualifying fractions, or None.

    Pure. Starts at ``start_row`` (near the bottom) and walks up while rows qualify; a
    pano whose ``start_row`` does not qualify has no such overlay."""
    if frac[start_row] < need:
        return None
    r = start_row
    while r > 0 and frac[r - 1] >= need:
        r -= 1
    return r / h


def measure_band(path):
    """``(kind, band_top_y)`` for a pano's bottom overlay, or ``(None, None)``.

    Full-width rows of a 1024x512 bilinear downsample. Black is tried first (the Grand
    Lyon Ladybug rig has no downward camera), then white (a Bayonne-style logo band)."""
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((HEAT_W, HEAT_H), Image.BILINEAR)).astype(int)
    black = (a.max(axis=2) <= BLACK_MAX).mean(axis=1)
    y = band_top_from_rows(list(black))
    if y is not None:
        return "black", y
    white = (a.min(axis=2) > WHITE_MIN).mean(axis=1)
    y = band_top_from_rows(list(white))
    return ("white", y) if y is not None else (None, None)


#: Burned-in logo bands the black/white method cannot see, found by eye on a contact sheet
#: of the bottom 31% of all 200 panos (2026-10-20, claude-opus-5-5; the sheet is not
#: committed, it regenerates from the panos). Two graphics, each identical across its
#: panos: the "La Voie Bleue" bike-route band (white with cyan waves, account
#: veloroutelavoiebleue) and a navy "la ville a velo" band (accounts hverlin and
#: trucbidule; three other trucbidule panos have none). Their top edge is measured per
#: pano by ``logo_band_top``.
LOGO_BANDS = {
    "1e003752-8f9f-4265-9bfa-d65e4f477c3c": "La Voie Bleue bike-route logo",
    "1fa0c0f1-7964-431b-b599-9103459b89e8": "La Voie Bleue bike-route logo",
    "3a17f053-ff4a-4178-923f-e42ab2c5ea60": "La Voie Bleue bike-route logo",
    "41e43f3b-5a3c-4633-b445-c5d976e13356": "La Voie Bleue bike-route logo",
    "6bbf88af-ed13-4bba-8905-cc7031d7a279": "La Voie Bleue bike-route logo",
    "9c6dfd07-8e0a-432d-b485-5d994fcc4905": "La Voie Bleue bike-route logo",
    "637f5103-6333-4ddf-bab9-3309ce52b33a": "navy 'la ville a velo' logo",
    "0c90b917-a4ff-4489-aceb-58440e3fed58": "navy 'la ville a velo' logo",
    "2b8c9798-08e1-4010-b9cd-cdceb655a7e3": "navy 'la ville a velo' logo",
    "a3a03099-0daf-471f-9095-4a2be694a727": "navy 'la ville a velo' logo",
}
LOGO_ROW_FRAC = 0.10     # share of a row in the logo's blue for the row to be in the band
LOGO_SCAN_FROM = 380     # rows above this (y < 0.742) are never band


def logo_band_top(frac, start=LOGO_SCAN_FROM, need=LOGO_ROW_FRAC, run=8, h=HEAT_H):
    """Normalized y of a logo band's top edge from per-row blue fractions, or None. Pure.

    The first row at or below ``start`` where the blue share reaches ``need`` and stays
    there for the next ``run`` rows: the crest of the Voie Bleue waves, the straight top
    of the navy band."""
    for r in range(start, h - run):
        if all(frac[r + i] >= need for i in range(run + 1)):
            return r / h
    return None


def measure_logo_band(path):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((HEAT_W, HEAT_H), Image.BILINEAR)).astype(int)
    blue = ((a[..., 2] > a[..., 0] + 40) & (a[..., 0] < 90)).mean(axis=1)
    return logo_band_top(list(blue))


#: Overlays read entirely by hand; empty, since every overlay found has a rule.
MANUAL_BANDS = {}


def build_band(bundle=BUNDLE):
    recs = load_records(bundle)
    panos = {}
    for r in recs:
        pid = r["pano"]["panorama_id"]
        kind, y = measure_band(os.path.join(bundle, "panos", f"{pid}.jpg"))
        entry = {"band_top_y": _r(y), "kind": kind, "label": LABELS.get(kind),
                 "group": r["benchmark_stratum"], "rig": rig_of(r["pano"]),
                 "method": "auto"}
        if pid in LOGO_BANDS:
            entry.update(band_top_y=_r(measure_logo_band(
                os.path.join(bundle, "panos", f"{pid}.jpg"))), kind="logo",
                label="nadir logo band", overlay=LOGO_BANDS[pid],
                method="found by eye on a contact sheet; edge by logo_band_top")
        if pid in MANUAL_BANDS:
            entry.update(MANUAL_BANDS[pid], method="manual")
        panos[pid] = {k: v for k, v in entry.items() if v is not None}
    by = collections.defaultdict(list)
    for e in panos.values():
        by[(e["rig"], e.get("kind"))].append(e.get("band_top_y"))
    summary = []
    for (rig, kind), ys in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        yy = sorted(y for y in ys if y is not None)
        summary.append({"rig": rig, "kind": kind, "panos": len(ys),
                        "band_top_y_min": yy[0] if yy else None,
                        "band_top_y_median": yy[len(yy) // 2] if yy else None,
                        "band_top_y_max": yy[-1] if yy else None})
    return {
        "what": "top edge of an opaque region at the bottom of the equirect, normalized y "
                "(0 = top), per pano. kind black = solid black (no downward camera); "
                "white = a burned-in logo band. Absent band_top_y = none found. "
                "gt_gallery.py hatches it, labels it, and asks before a missed-ramp mark "
                "lands inside it.",
        "found_by_eye": "all 200 panos' bottom 31% were looked at on a contact sheet "
                        "(2026-10-20, claude-opus-5-5); the black nadirs are found by the "
                        "rule, the two logo graphics by eye (LOGO_BANDS); rig parts such as "
                        "roof racks and car bodies are imagery and not marked",
        "method": {"downsample": f"{HEAT_W}x{HEAT_H} bilinear", "columns": "all",
                   "logo": f"rows >= {LOGO_SCAN_FROM}: first row whose share of logo blue "
                           f"(b > r + 40, r < 90) is >= {LOGO_ROW_FRAC} for 9 rows running",
                   "black": f"every channel <= {BLACK_MAX}",
                   "white": f"every channel > {WHITE_MIN}",
                   "row_is_band": f">= {BAND_ROW_FRAC} of the row qualifies",
                   "scan": f"up from row {BAND_START_ROW} while rows qualify",
                   "script": "scripts/analysis/lyon_review_159.py band"},
        "n_with_band": sum("band_top_y" in e for e in panos.values()),
        "n_without_band": sum("band_top_y" not in e for e in panos.values()),
        "by_rig": summary,
        "panos": panos,
    }


def cmd_band(args):
    return _check_or_write(build_band(args.bundle),
                           os.path.join(args.bundle, NADIR_BAND_NAME), args.write)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify_bundle(bundle=BUNDLE, require_imagery=True):
    """(problems, notes, facts) for the staged bundle; empty problems = OK.

    With ``require_imagery`` (the default, the pre-review gate) a missing
    ``imagery_manifest.json``, ``nadir_band.json`` or ``panos/`` is a problem, so
    "verify: OK" means the imagery is staged and byte-checked."""
    problems, notes = [], []
    recs = load_records(bundle)
    ids = [r["pano"]["panorama_id"] for r in recs]
    if len(ids) != len(set(ids)):
        problems.append("duplicate panorama_id in records.jsonl")
    with open(os.path.join(bundle, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    strata = collections.Counter(r["benchmark_group"] for r in recs)
    if dict(strata) != sample["groups"]:
        problems.append(f"strata {dict(strata)} != sample.json {sample['groups']}")
    for r in recs:
        n55 = sum(d["confidence"] >= DEPLOYED for d in r["detections"])
        if (r["benchmark_group"] == "empty") != (n55 == 0):
            problems.append(f"{r['pano']['panorama_id']}: stratum {r['benchmark_group']} "
                            f"with {n55} detections >= {DEPLOYED}")
        if any(d["confidence"] < sample.get("record_confidence", DEPLOYED)
               for d in r["detections"]):
            problems.append(f"{r['pano']['panorama_id']}: detection below the record tier")
        if "benchmark_stratum" not in r:
            problems.append(f"{r['pano']['panorama_id']}: no benchmark_stratum")
    # never-train lists
    if _read_lines(os.path.join(bundle, "ids.txt")) != sorted(ids):
        problems.append("ids.txt is not the sorted record ids")
    excl = set(_read_lines(os.path.join(bundle, "exclude_ids.txt")))
    if not set(ids) <= excl:
        problems.append("exclude_ids.txt does not contain every audited id")
    peaks = load_run_peaks()
    if set(peaks) != set(ids):
        problems.append("run_peaks.json and records.jsonl list different panos")
    for r in recs:      # the records are the run's >= 0.55 peaks, in order
        hi = [d for d in peaks.get(r["pano"]["panorama_id"], []) if d["confidence"] >= DEPLOYED]
        if [(d["x_normalized"], d["y_normalized"]) for d in hi] != \
                [(d["x_normalized"], d["y_normalized"]) for d in r["detections"]]:
            problems.append(f"{r['pano']['panorama_id']}: records and run_peaks disagree at 0.55")
    for name in TEXT_FILES:
        path = os.path.join(bundle, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                if EMAIL_RE.search(f.read()):
                    problems.append(f"{name} carries an email-shaped token")
    # imagery
    man_path = os.path.join(bundle, "imagery_manifest.json")
    panos_dir = os.path.join(bundle, "panos")
    manifest = None
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)["panos"]
        if set(manifest) != set(ids):
            problems.append("records.jsonl and imagery_manifest.json list different panos")
    else:
        (problems if require_imagery else notes).append(
            "imagery not staged: imagery_manifest.json absent")
    if not os.path.exists(os.path.join(bundle, NADIR_BAND_NAME)):
        (problems if require_imagery else notes).append(
            f"imagery not staged: {NADIR_BAND_NAME} absent")
    index_path = os.path.join(bundle, "index.csv")
    if os.path.exists(index_path) and manifest is not None:
        with open(index_path, encoding="utf-8", newline="") as f:
            index = {row["panorama_id"]: row for row in csv.DictReader(f)}
        bad = [p for p in set(index) | set(manifest) if p not in index or p not in manifest
               or index[p]["sha256"] != manifest[p]["sha256"]]
        if bad:
            problems.append(f"index.csv and imagery_manifest.json disagree on {len(bad)} panos")
        else:
            notes.append(f"index.csv agrees with imagery_manifest.json ({len(index)} panos)")
    else:
        (problems if require_imagery else notes).append(
            "imagery not staged: index.csv or manifest absent, exporter cross-check skipped")
    if manifest is not None and os.path.isdir(panos_dir):
        checked = 0
        for pid, row in sorted(manifest.items()):
            path = os.path.join(panos_dir, row["file"])
            if not os.path.exists(path):
                problems.append(f"{pid}: missing from panos/")
                continue
            if _sha256(path) != row["sha256"]:
                problems.append(f"{pid}: bytes differ from imagery_manifest.json")
            checked += 1
        notes.append(f"{checked} panos byte-checked against imagery_manifest.json")
    else:
        (problems if require_imagery else notes).append(
            "imagery not staged: panos/ or manifest absent, byte check skipped")
    facts = {"panos": len(recs), "strata": dict(sorted(strata.items())),
             "detections_055": sum(d["confidence"] >= DEPLOYED for r in recs
                                   for d in r["detections"]),
             "run_peaks_030": sum(d["confidence"] >= LOW_TIER for v in peaks.values()
                                  for d in v),
             "exclude_ids": len(excl)}
    return problems, notes, facts


def cmd_verify(args):
    problems, notes, facts = verify_bundle(args.bundle, not args.records_only)
    print(json.dumps(facts, sort_keys=True))
    for n in notes:
        print("  " + n)
    for p in problems:
        print("  PROBLEM: " + p)
    print("verify: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# score (after the review only)
# --------------------------------------------------------------------------- #
def bootstrap_ci(units, reps=BOOT_REPS, seed=BOOT_SEED):
    """95% percentile interval of sum(k)/sum(n) over pano units resampled with replacement.

    ``units`` is a list of (k, n) per pano (panos with n = 0 still count as draws). Pure
    and deterministic for a seed. Returns (lo, hi), or (None, None) without any n."""
    if not units or not sum(n for _, n in units):
        return None, None
    rng = random.Random(seed)
    m = len(units)
    vals = []
    for _ in range(reps):
        k = n = 0
        for _ in range(m):
            a, b = units[rng.randrange(m)]
            k += a
            n += b
        if n:
            vals.append(k / n)
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


TOP_UP_MIN_GAP = 0.01    # nearer the bar than this, no finite top-up is suggested


def bar_reading(p, intervals, n_random, pool_left=None, rate=None, bar=PASS_PRECISION):
    """The bar reading for one group. Pure.

    ``p`` is the point estimate, ``intervals`` the (lo, hi) pairs reported beside it
    (bootstrap and Wilson), ``n_random`` the group's judged random panos, ``pool_left``
    the random-eligible panos the run still has for this group (None: no cap), ``rate``
    detections per random pano. Eligible = ``p >= bar``. Decisive = both intervals on
    p's side of the bar. A non-decisive reading carries a top-up size: the random panos
    that would shrink the widest half-width on the bar's side to ``|p - bar|`` at the
    same p (half-width ~ 1/sqrt(n)); none when p is within 0.01 of the bar, and
    "not reachable" when it exceeds the pool.

    Example:
        >>> bar_reading(0.9, [(0.82, 0.96), (0.81, 0.95)], 50)["reading"]
        'eligible (decisive)'
    """
    if p is None:
        return {"eligible": None, "decisive": None, "reading": "no detections judged"}
    eligible = p >= bar
    ivs = [(lo, hi) for lo, hi in intervals if lo is not None]
    if eligible:
        decisive = all(lo >= bar for lo, _ in ivs)
        reading = ("eligible (decisive)" if decisive else
                   "eligible, not decisive: an interval contains 0.80")
    else:
        decisive = all(hi < bar for _, hi in ivs)
        reading = ("decisive fail: excluded from any Lyon training pool" if decisive else
                   "not eligible (point estimate below 0.80), not decisive: an interval "
                   "contains 0.80")
    out = {"eligible": eligible, "decisive": decisive, "reading": reading}
    if decisive:
        return out
    gap = abs(p - bar)
    if gap < TOP_UP_MIN_GAP:
        out["top_up"] = (f"no finite top-up: the point estimate is within {TOP_UP_MIN_GAP} "
                         f"of the bar")
        return out
    half = max((p - lo) if eligible else (hi - p) for lo, hi in ivs)
    extra = max(0, math.ceil(round(n_random * (half / gap) ** 2, 6)) - n_random)
    if pool_left is not None and extra > pool_left:
        out["top_up"] = (f"not reachable: needs about {extra} more random panos, the group's "
                         f"pool has {pool_left} left")
        return out
    out["top_up"] = {"extra_random_panos": extra}
    if rate is not None:
        out["top_up"]["extra_detections_at_current_rate"] = round(extra * rate)
    return out


def _scan_entries(recs, verdict_panos):
    """(problems, warnings) for verdict entries against the records: every non-top record
    must have a fully judged entry whose length matches its detections."""
    problems = []
    for r in recs:
        if r["benchmark_group"] == "top":
            continue
        pid = r["pano"]["panorama_id"]
        e = verdict_panos.get(pid)
        if e is None:
            problems.append(f"{pid}: no verdict entry")
        elif len(e.get("dets", [])) != len(r["detections"]):
            problems.append(f"{pid}: {len(e.get('dets', []))} verdicts for "
                            f"{len(r['detections'])} detections")
        elif any(d is None for d in e["dets"]):
            problems.append(f"{pid}: partly judged")
    return problems


def score_groups(recs, verdict_panos, run_peaks, lenient_duplicates=False, pool_left=None):
    """Per audit group (+ ``all``), unbiased subset: precision / recall at 0.55 (from the
    verdicts, rampnet.validation) and at 0.30 (the run's stored peaks matched to the GT,
    rampnet.detection_eval), with pano-bootstrap and Wilson intervals and the bar reading.
    Pure: ``verdict_panos`` is verdicts.json's ``panos``; ``pool_left`` maps a group to the
    random-eligible panos its run pool still has (caps the top-up). The 0.55 precision is
    read on the random panos (empty panos carry no 0.55 detection); recall and the 0.30
    tier on random + empty."""
    from rampnet.detection_eval import build_ground_truth, score_pano
    from rampnet.validation import collect, total_ramps
    confs = {r["pano"]["panorama_id"]: [d["confidence"] for d in r["detections"]]
             for r in recs}
    stratum = {r["pano"]["panorama_id"]: r["benchmark_stratum"] for r in recs}
    group_of = {r["pano"]["panorama_id"]: r["benchmark_group"] for r in recs}
    rec_of = {r["pano"]["panorama_id"]: r for r in recs}
    out, warnings = {}, []
    present = [g for g in GROUPS if g in set(stratum.values())]
    for g in present + ["all"]:
        pids = sorted(pid for pid in verdict_panos
                      if pid in confs and group_of[pid] != "top"
                      and (g == "all" or stratum[pid] == g))
        panos = {pid: {**verdict_panos[pid], "group": group_of[pid]} for pid in pids}
        pools = collect(panos, confs, exclude_top=True, lenient_duplicates=lenient_duplicates)
        warnings += [f"{g}: {w}" for w in pools.warnings]
        row = {"panos_reviewed": pools.n_judged,
               "recall_pool_panos": pools.n_judged - pools.n_unconfirmed,
               "unsure_detections": pools.n_unsure, "duplicate_detections": pools.n_duplicate,
               "missed_marks": pools.missed_total, "missed_unsure": pools.missed_unsure}
        # 0.55: per-pano (tp, n) units from the verdicts, the same folding as collect()
        units55, rec_units55 = [], []
        for pid in pids:
            e = panos[pid]
            if any(d is None for d in e["dets"]) or len(e["dets"]) != len(confs[pid]):
                continue
            ok = []
            for d in e["dets"]:
                if d == "unsure" or (d == "duplicate" and lenient_duplicates):
                    continue
                ok.append(d is True)
            if group_of[pid] == "random":
                units55.append((sum(ok), len(ok)))
            fn_ok = (e["no_missed"] or e["missed"]) if "no_missed" in e else True
            if fn_ok:
                miss = sum(1 for m in e["missed"] if not m.get("unsure"))
                rec_units55.append((sum(ok), sum(ok) + miss))
        tp55, n55 = sum(k for k, _ in units55), sum(n for _, n in units55)
        assert n55 == len(pools.judged), "0.55 detections outside the random stratum"
        denom = total_ramps(pools)
        rtp55 = sum(k for k, _ in rec_units55)
        assert rtp55 == denom - pools.missed_total, "recall pool disagrees with collect()"
        # 0.30: the run's stored peaks against the GT built from the verdicts
        units30, rec_units30 = [], []
        for pid in pids:
            e = panos[pid]
            if any(d is None for d in e["dets"]) or len(e["dets"]) != len(confs[pid]):
                continue
            gt = build_ground_truth(rec_of[pid]["detections"], e["dets"], e["missed"],
                                    e.get("no_missed", True))
            preds = [d for d in run_peaks[pid] if d["confidence"] >= LOW_TIER]
            s = score_pano(preds, gt)
            units30.append((s.tp, s.tp + s.fp))
            if s.fn_confirmed:
                rec_units30.append((s.tp, s.n_gt))
        for tag, units, runits in (("055", units55, rec_units55), ("030", units30, rec_units30)):
            tp, n = sum(k for k, _ in units), sum(b for _, b in units)
            rtp, rn = sum(k for k, _ in runits), sum(b for _, b in runits)
            blo, bhi = bootstrap_ci(units)
            wlo, whi = wilson(tp, n)
            rlo, rhi = bootstrap_ci(runits)
            row[f"precision_{tag}"] = {"tp": tp, "n": n, "p": _r(tp / n) if n else None,
                                       "bootstrap95": [_r(blo), _r(bhi)],
                                       "wilson95": [_r(wlo), _r(whi)]}
            row[f"recall_{tag}"] = {"tp": rtp, "n": rn, "r": _r(rtp / rn) if rn else None,
                                    "bootstrap95": [_r(rlo), _r(rhi)]}
        assert (tp55, n55) == (row["precision_055"]["tp"], row["precision_055"]["n"])
        if g == "all":
            row["bar"] = "reported, no verdict (the groups are not sampled in proportion)"
        else:
            p = row["precision_055"]
            reading = bar_reading(p["p"], [tuple(p["bootstrap95"]), tuple(p["wilson95"])],
                                  len(units55), (pool_left or {}).get(g),
                                  n55 / len(units55) if units55 else None)
            row["bar"] = reading.pop("reading")
            row.update(reading)
        out[g] = row
    return out, warnings


def posted_mismatches(bundle=BUNDLE):
    """Why the bundle is not the version posted on #159 (empty list = it is).

    Checks the posted sha256 of audit_plan.json and records.jsonl, and that the current
    code still rebuilds the committed plan byte for byte (the bar's constants live in this
    script, so a code edit could change the reading while the plan file still matched)."""
    out = []
    with open(os.path.join(bundle, "bundle_provenance.json"), encoding="utf-8") as f:
        posted = json.load(f).get("posted") or {}
    for name, key in ((PLAN_NAME, "audit_plan_sha256"), ("records.jsonl", "records_jsonl_sha256")):
        sha = _sha256(os.path.join(bundle, name))
        if sha != posted.get(key):
            out.append(f"{name} hashes to {sha[:12]}, the posted value is "
                       f"{(posted.get(key) or 'none')[:12]}")
    with open(os.path.join(bundle, PLAN_NAME), "rb") as f:
        if f.read() != _dumps(build_plan(bundle)).encode("utf-8"):
            out.append("the current code does not rebuild the committed audit_plan.json")
    return out


def cmd_score(args):
    vpath = os.path.join(args.bundle, "verdicts.json")
    if not os.path.exists(vpath):
        print(f"no {os.path.relpath(vpath, REPO)}: the review has not happened, "
              f"nothing to score")
        return 1
    problems = posted_mismatches(args.bundle)
    for p in problems:
        print("  PROBLEM: " + p)
    if problems and not args.allow_unposted_plan:
        print("refusing to score against a plan or sample other than the one posted on #159 "
              "(--allow-unposted-plan overrides, and is recorded)")
        return 1
    plan_sha = _sha256(os.path.join(args.bundle, PLAN_NAME))
    with open(vpath, encoding="utf-8") as f:
        verdicts = json.load(f)
    recs = load_records(args.bundle)
    problems_v = _scan_entries(recs, verdicts.get("panos", {}))
    for p in problems_v:
        print("  PROBLEM: " + p)
    if problems_v and not args.partial:
        print(f"{len(problems_v)} non-top panos are missing, misaligned or partly judged: "
              f"refusing to score (--partial scores the rest, and is recorded)")
        return 1
    with open(os.path.join(args.bundle, "sample.json"), encoding="utf-8") as f:
        strata = json.load(f)["stratified"]["strata"]
    pool_left = {g: v["pool_with_detections"] - v["random"] for g, v in strata.items()}
    groups, warnings = score_groups(recs, verdicts.get("panos", {}), load_run_peaks(),
                                    args.lenient_duplicates, pool_left)
    for w in warnings:
        print("  WARNING: " + w)
    res = {"pass_bar": PASS_BAR, "audit_plan_sha256": plan_sha,
           "plan_posted": not posted_mismatches(args.bundle), "partial": bool(problems_v),
           "unscored_panos": problems_v, "lenient_duplicates": args.lenient_duplicates,
           "review_notes": verdicts.get("review_notes") or {}, "groups": groups}
    for g, row in groups.items():
        p, p3 = row["precision_055"], row["precision_030"]
        print(f"{g:>20}  P@0.55 {p['p']} ({p['tp']}/{p['n']}, boot {p['bootstrap95']})  "
              f"P@0.30 {p3['p']} ({p3['tp']}/{p3['n']})  R@0.55 {row['recall_055']['r']}  "
              f"R@0.30 {row['recall_030']['r']}  -> {row['bar']}"
              + (f"; top-up: {row['top_up']}" if row.get("top_up") else ""))
    return _check_or_write(res, SCORE, args.write)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=BUNDLE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--from", dest="src", required=True, help="the exporter's bundle dir")
    s.add_argument("--results", required=True, help="the run's results.jsonl")
    s.add_argument("--buffer", type=float, default=EXCLUDE_BUFFER_M,
                   help="never-train radius around each audited pano, metres "
                        "(default: %(default)s)")
    s.set_defaults(fn=cmd_stage)
    for name, fn in (("plan", cmd_plan), ("band", cmd_band)):
        s = sub.add_parser(name)
        s.add_argument("--write", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("table")
    s.set_defaults(fn=cmd_table)
    s = sub.add_parser("artifacts")
    s.add_argument("--results", required=True, help="the run's results.jsonl")
    s.add_argument("--write", action="store_true")
    s.set_defaults(fn=cmd_artifacts)
    s = sub.add_parser("verify")
    s.add_argument("--records-only", action="store_true",
                   help="check the committed files only; imagery absence is a note")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("score")
    s.add_argument("--write", action="store_true")
    s.add_argument("--lenient-duplicates", action="store_true")
    s.add_argument("--partial", action="store_true",
                   help="score despite missing / misaligned / partly judged panos")
    s.add_argument("--allow-unposted-plan", action="store_true",
                   help="score although audit_plan.json differs from the posted plan")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
