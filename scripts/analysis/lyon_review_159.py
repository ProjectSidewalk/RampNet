"""Lyon label audit (#159, #263): stage the review bundle, pin the plan, score it per group.

Lyon is a **training-only** candidate city, never a benchmark split. On 2026-10-06 Jon
decided that its self-labels train only after a quality audit: about 150-200 panos
from the 5 m run of record, stratified by rig x producer, judged the same way as a
benchmark session, with the pass bar posted on #159 before anyone reviews. The audited
panos are held out of every training pool. ``docs/lyon_review_159.md`` is the write-up.

**Nothing here produces a verdict.** The review is Jon's. ``plan`` pins the design and
the pass bar before the review; ``score`` refuses to run until ``verdicts.json`` exists.

Subcommands, in the order the doc runs them:

    # copy the exporter's bundle in, masking email-shaped producer names (#149)
    python scripts/analysis/lyon_review_159.py stage --from D:/Git/labeler-wt/lyon-bundle

    # the audit plan: strata, group table and pass bar -> benchmark/lyon/audit_plan.json
    python scripts/analysis/lyon_review_159.py plan --write

    # the known-artifact reads from #159 on the bundle's detections (needs the run's
    # results.jsonl, which is not in this repo) -> analysis_out/lyon_review_159/artifacts.json
    python scripts/analysis/lyon_review_159.py artifacts --results <runs/lyon/results.jsonl> --write

    # bottom-of-frame overlays measured per pano (needs panos/) -> benchmark/lyon/nadir_band.json
    python scripts/analysis/lyon_review_159.py band --write

    # bundle integrity (needs panos/ for the byte check)
    python scripts/analysis/lyon_review_159.py verify

    # after the review only: per-group precision / recall at 0.55 and 0.30 against the bar
    python scripts/analysis/lyon_review_159.py score --write

Every ``--write`` output is LF, sorted, with floats rounded, so a re-run on any OS is
byte-identical; without ``--write`` the command recomputes and compares against the
committed file. ``tests/test_lyon_review_159.py`` covers the CPU logic.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

BUNDLE = os.path.join(REPO, "benchmark", "lyon")
OUT_DIR = os.path.join(REPO, "analysis_out", "lyon_review_159")
PLAN = os.path.join(BUNDLE, "audit_plan.json")
NADIR_BAND = os.path.join(BUNDLE, "nadir_band.json")
ARTIFACTS = os.path.join(OUT_DIR, "artifacts.json")
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

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask_email(text):
    """Mask every email-shaped token, as the labeler's census does (issue #149).

    Panoramax serves producer names publicly and a few accounts use an email address.
    The repo never republishes one: the token keeps its first character only.

    Example:
        >>> mask_email('{"copyright": "jane.doe@example.org"}')
        '{"copyright": "j***@***"}'
        >>> mask_email('grand lyon')
        'grand lyon'
    """
    return EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@***", text)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def _check_or_write(obj, path, write):
    text = _dumps(obj)
    rel = os.path.relpath(path, REPO)
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"wrote {rel}")
        return 0
    if not os.path.exists(path):
        print(f"MISSING {rel} (run with --write)")
        return 1
    with open(path, encoding="utf-8") as f:
        same = f.read() == text
    print(f"{'OK' if same else 'DIFFERS'}: {rel}")
    return 0 if same else 1


def load_records(bundle=BUNDLE):
    with open(os.path.join(bundle, "records.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k of n, or (None, None) when n is 0."""
    from rampnet.validation import wilson_interval
    return wilson_interval(k, n, z) if n else (None, None)


def rig_of(pano):
    """'make model WxH' with absent EXIF written as '(no make/model)'."""
    mm = " ".join(str(pano.get(k)) for k in ("camera_make", "camera_model") if pano.get(k))
    return f"{mm or '(no make/model)'} {pano.get('width')}x{pano.get('height')}"


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
STAGED_FILES = ("records.jsonl", "sample.json", "group_map.json")


def stage(src, dst=BUNDLE):
    """Copy the exporter's bundle files into ``dst``, masking emails in records.jsonl.

    Returns the number of records whose text changed. ``panos/`` and ``index.csv`` are
    not copied here: they are git-ignored and travel with the imagery."""
    os.makedirs(dst, exist_ok=True)
    masked = 0
    with open(os.path.join(src, "records.jsonl"), encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        new = mask_email(line)
        masked += new != line
        out.append(new)
    with open(os.path.join(dst, "records.jsonl"), "w", encoding="utf-8", newline="") as f:
        f.write("".join(line + "\n" for line in out))
    for name in STAGED_FILES[1:]:
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    return masked


def cmd_stage(args):
    n = stage(args.src, args.bundle)
    print(f"staged {', '.join(STAGED_FILES)} into {os.path.relpath(args.bundle, REPO)}; "
          f"{n} record(s) had an email-shaped token masked")
    return 0


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def group_table(recs):
    """Per audit group and stratum: panos and detections at 0.55 and 0.30. Pure."""
    rows = {}
    for r in recs:
        g, s = r["benchmark_stratum"], r["benchmark_group"]
        row = rows.setdefault(g, {f"{k}_{s2}": 0 for s2 in STRATA
                                  for k in ("panos", "det055", "det030")})
        row[f"panos_{s}"] += 1
        row[f"det055_{s}"] += sum(d["confidence"] >= DEPLOYED for d in r["detections"])
        row[f"det030_{s}"] += sum(d["confidence"] >= LOW_TIER for d in r["detections"])
    return {g: rows[g] for g in sorted(rows, key=lambda g: (GROUPS + (g,)).index(g))}


def rig_breakdown(recs):
    """Panos per (group, rig, producer, capture year), the descriptive rows. Pure."""
    c = collections.Counter()
    for r in recs:
        p = r["pano"]
        c[(r["benchmark_stratum"], rig_of(p), str(p.get("copyright")),
           (p.get("capture_date") or "")[:4])] += 1
    return [{"group": g, "rig": rig, "producer": prod, "year": y, "panos": n}
            for (g, rig, prod, y), n in sorted(c.items(), key=lambda kv: (
                (GROUPS + (kv[0][0],)).index(kv[0][0]), -kv[1], kv[0]))]


def build_plan(bundle=BUNDLE):
    recs = load_records(bundle)
    with open(os.path.join(bundle, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    strata = sample["stratified"]["strata"]
    table = group_table(recs)
    for g, row in table.items():
        # The run's own panos per group: with >= 1 detection at 0.55 (the sampler's pool
        # plus the top panos it drew first) and without.
        row["run_panos_with_det055"] = strata[g]["pool_with_detections"] + strata[g]["top"]
        row["run_panos_empty"] = strata[g]["pool_empty"]
    return {
        "what": "Lyon label audit (#159 decision of 2026-10-06, #263 track 2). A working "
                "draft posted before the review; Jon may amend it before the review opens. "
                "Lyon is not a benchmark split, and these panos never train.",
        "question": "Are the RampNet self-labels on each Lyon rig x producer group good "
                    "enough to train on?",
        "pass_bar": {
            "metric": "precision of the detections at confidence >= 0.55",
            "threshold": PASS_PRECISION,
            "read_on": "each audit group's unbiased subset (random + empty strata; the "
                       "top stratum is excluded, as for every split)",
            "rule": "a group passes if its point estimate is >= 0.80. The 95% Wilson "
                    "interval is reported beside it; an interval that straddles 0.80 is "
                    "reported as not decisive. A group that fails is excluded from any "
                    "Lyon training pool.",
            "source": "the #158 build line (>= 0.80 build, 0.50-0.80 visibility, < 0.50 drop)",
            "reported_alongside": ["recall at 0.55", "precision and recall at 0.30",
                                   "unsure and duplicate counts", "per-rig breakdown "
                                   "inside each group (descriptive, no bar)"],
            "duplicates": "scored as false positives (rampnet.validation default)",
        },
        "groups": {g: GROUP_TITLES.get(g, g) for g in table},
        "group_table": table,
        "rig_breakdown": rig_breakdown(recs),
        "sampler": {k: sample[k] for k in ("seed", "sample", "empty_sample", "min_spacing_m",
                                           "min_confidence", "record_confidence", "top_n",
                                           "selected", "groups", "source_record_count")
                    if k in sample},
        "records_show_detections_down_to": sample.get("record_confidence", DEPLOYED),
        "never_train": "Every panorama_id in benchmark/lyon/records.jsonl is held out of "
                       "every training pool, whatever the audit reads.",
    }


def cmd_plan(args):
    return _check_or_write(build_plan(args.bundle), PLAN, args.write)


def print_table(plan):
    """The group table as markdown (what the doc and the #159 comment quote)."""
    head = ("| group | run panos with a 0.55 detection / without | top | random | empty "
            "| detections ≥ 0.55 (random) | detections ≥ 0.30 (random + empty) |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|"]
    tot = collections.Counter()
    table = plan["group_table"]
    for g in sorted(table, key=lambda g: (GROUPS + (g,)).index(g)):
        row = table[g]
        cells = [row["run_panos_with_det055"], row["run_panos_empty"], row["panos_top"],
                 row["panos_random"], row["panos_empty"], row["det055_random"],
                 row["det030_random"] + row["det030_empty"]]
        tot.update(dict(enumerate(cells)))
        lines.append(f"| {g} | {cells[0]:,} / {cells[1]:,} | " +
                     " | ".join(str(c) for c in cells[2:]) + " |")
    lines.append(f"| **all** | {tot[0]:,} / {tot[1]:,} | " +
                 " | ".join(f"**{tot[i]}**" for i in range(2, 7)) + " |")
    return "\n".join(lines)


def cmd_table(args):
    with open(PLAN, encoding="utf-8") as f:
        print(print_table(json.load(f)))
    return 0


# --------------------------------------------------------------------------- #
# artifacts (#159's known artifacts, read on the bundle's own detections)
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


def build_artifacts(recs, seq_cells, run_sha):
    """Counts per audit group of the two artifacts #159 named. Pure.

    * fixed-pixel cells: detections >= 0.30 in a cell that repeats in >= 5 panos of the
      pano's own sequence (the Grand Lyon cluster: x 0.503, y 0.756, just above the
      black nadir, at the 0.30 tier only);
    * above the horizon: detections with y < 0.5 (ecartip: 10.3% at 0.55 in the first
      slice, with pose reported as exactly 0/0, so possibly unlevelled panos).
    """
    per = {}
    items = []
    for r in recs:
        g, pid = r["benchmark_stratum"], r["pano"]["panorama_id"]
        cnt = seq_cells.get(r["pano"].get("sequence_id"), collections.Counter())
        row = per.setdefault(g, {"det030": 0, "det055": 0, "fixed_cell_030": 0,
                                 "fixed_cell_055": 0, "above_horizon_030": 0,
                                 "above_horizon_055": 0})
        for i, d in enumerate(r["detections"]):
            if d["confidence"] < LOW_TIER:
                continue
            hi = d["confidence"] >= DEPLOYED
            fixed = cnt[cell_of(d)] >= FIXED_CELL_MIN_PANOS
            above = d["y_normalized"] < 0.5
            row["det030"] += 1
            row["det055"] += hi
            row["fixed_cell_030"] += fixed
            row["fixed_cell_055"] += fixed and hi
            row["above_horizon_030"] += above
            row["above_horizon_055"] += above and hi
            if fixed or above:
                items.append({"pano": pid, "det": i, "group": g,
                              "confidence": _r(d["confidence"]), "x": _r(d["x_normalized"]),
                              "y": _r(d["y_normalized"]), "fixed_cell": fixed,
                              "above_horizon": above,
                              "cell_panos_in_sequence": cnt[cell_of(d)]})
    return {"what": "GT-free counts of the #159 known artifacts on the bundle's detections. "
                    "DO NOT open `items` before verdicts.json is exported: it names "
                    "detections you are about to judge.",
            "definitions": {"fixed_cell": f"detection in a {HEAT_W}x{HEAT_H} cell that holds "
                                          f"a >= {LOW_TIER} peak in >= {FIXED_CELL_MIN_PANOS} "
                                          f"panos of its own sequence (the pano included)",
                            "above_horizon": "y_normalized < 0.5"},
            "results_jsonl_sha256": run_sha,
            "per_group": {g: per[g] for g in sorted(per, key=lambda g: (GROUPS + (g,)).index(g))},
            "items": sorted(items, key=lambda it: (it["pano"], it["det"]))}


def _iter_results(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_artifacts(args):
    recs = load_records(args.bundle)
    seqs = {r["pano"].get("sequence_id") for r in recs}
    cells = sequence_cells(_iter_results(args.results), seqs)
    return _check_or_write(build_artifacts(recs, cells, _sha256(args.results)),
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


#: Overlays the automatic method cannot see, read by eye; empty until a pano needs one.
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
        "method": {"downsample": f"{HEAT_W}x{HEAT_H} bilinear", "columns": "all",
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
    return _check_or_write(build_band(args.bundle), NADIR_BAND, args.write)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify_bundle(bundle=BUNDLE):
    """(problems, notes, facts) for the staged bundle; empty problems = OK."""
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
    with open(os.path.join(bundle, "records.jsonl"), encoding="utf-8") as f:
        if EMAIL_RE.search(f.read()):
            problems.append("records.jsonl carries an email-shaped token (stage masks them)")
    man_path = os.path.join(bundle, "imagery_manifest.json")
    manifest = None
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)["panos"]
        if set(manifest) != set(ids):
            problems.append("records.jsonl and imagery_manifest.json list different panos")
    else:
        notes.append("imagery_manifest.json absent: imagery not pinned yet")
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
        notes.append("index.csv or manifest absent: exporter cross-check skipped")
    panos_dir = os.path.join(bundle, "panos")
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
        notes.append("panos/ or manifest absent: byte check skipped")
    facts = {"panos": len(recs), "strata": dict(sorted(strata.items())),
             "detections_055": sum(d["confidence"] >= DEPLOYED for r in recs
                                   for d in r["detections"]),
             "detections_030": sum(d["confidence"] >= LOW_TIER for r in recs
                                   for d in r["detections"])}
    return problems, notes, facts


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
# score (after the review only)
# --------------------------------------------------------------------------- #
def score_groups(recs, verdict_panos, lenient_duplicates=False):
    """Per audit group (+ ``all``), unbiased subset: precision / recall at 0.55 and 0.30
    and the pass-bar reading. Pure: ``verdict_panos`` is verdicts.json's ``panos``."""
    from rampnet.validation import _precision_at, _recall_tp_at, collect, total_ramps
    confs = {r["pano"]["panorama_id"]: [d["confidence"] for d in r["detections"]]
             for r in recs}
    stratum = {r["pano"]["panorama_id"]: r["benchmark_stratum"] for r in recs}
    group_of = {r["pano"]["panorama_id"]: r["benchmark_group"] for r in recs}
    out = {}
    for g in [g for g in GROUPS if g in set(stratum.values())] + ["all"]:
        panos = {pid: {**v, "group": group_of.get(pid, v.get("group"))}
                 for pid, v in verdict_panos.items()
                 if pid in confs and (g == "all" or stratum[pid] == g)}
        pools = collect(panos, confs, exclude_top=True, lenient_duplicates=lenient_duplicates)
        denom = total_ramps(pools)
        row = {"panos_reviewed": pools.n_judged, "recall_pool_panos":
               pools.n_judged - pools.n_unconfirmed, "unsure_detections": pools.n_unsure,
               "duplicate_detections": pools.n_duplicate, "missed_marks": pools.missed_total,
               "missed_unsure": pools.missed_unsure, "ramps": denom}
        for t, tag in ((DEPLOYED, "055"), (LOW_TIER, "030")):
            tp, kept = _precision_at(pools.judged, t)
            lo, hi = wilson(tp, kept)
            rtp = _recall_tp_at(pools.recall_judged, t)
            rlo, rhi = wilson(rtp, denom)
            row[f"precision_{tag}"] = {"tp": tp, "n": kept, "p": _r(tp / kept) if kept else None,
                                       "ci95": [_r(lo), _r(hi)]}
            row[f"recall_{tag}"] = {"tp": rtp, "n": denom, "r": _r(rtp / denom) if denom else None,
                                    "ci95": [_r(rlo), _r(rhi)]}
        p = row["precision_055"]
        if p["p"] is None:
            row["bar"] = "no detections judged"
        else:
            row["bar"] = "PASS" if p["p"] >= PASS_PRECISION else "FAIL"
            if p["ci95"][0] < PASS_PRECISION <= p["ci95"][1]:
                row["bar"] += " (not decisive: the interval straddles 0.80)"
        out[g] = row
    return out


def cmd_score(args):
    vpath = os.path.join(args.bundle, "verdicts.json")
    if not os.path.exists(vpath):
        print(f"no {os.path.relpath(vpath, REPO)}: the review has not happened, "
              f"nothing to score")
        return 1
    with open(vpath, encoding="utf-8") as f:
        verdicts = json.load(f)
    res = {"pass_bar": f"precision at {DEPLOYED} >= {PASS_PRECISION} per group, "
                       f"unbiased subset (random + empty)",
           "review_notes": verdicts.get("review_notes") or {},
           "groups": score_groups(load_records(args.bundle), verdicts.get("panos", {}),
                                  args.lenient_duplicates)}
    for g, row in res["groups"].items():
        p, r3 = row["precision_055"], row["precision_030"]
        print(f"{g:>20}  P@0.55 {p['p']} ({p['tp']}/{p['n']}, CI {p['ci95']})  "
              f"P@0.30 {r3['p']} ({r3['tp']}/{r3['n']})  R@0.55 {row['recall_055']['r']}  "
              f"R@0.30 {row['recall_030']['r']}  -> {row['bar']}")
    return _check_or_write(res, SCORE, args.write)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=BUNDLE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--from", dest="src", required=True, help="the exporter's bundle dir")
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
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("score")
    s.add_argument("--write", action="store_true")
    s.add_argument("--lenient-duplicates", action="store_true")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
