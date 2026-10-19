"""Juneau (#254): bundle integrity and the GPU run's two checks.

Juneau is a pooled US split: 125 Mapillary panoramas, 120 of them from contributor
rking's Trimble MX7 (the Annapolis rig) and 5 GoPro Max, exported by
sidewalk-auto-labeler (PR 164) and reviewed by Jon on 2026-10-19
(``docs/juneau_split_254.md``). Unlike Bayonne it was reviewed *before* its GPU work,
so the extraction wrote the reviewed cache straight to ``analysis_out/op_cache/`` and
there is no unreviewed cache or ``attach-gt`` step.

Subcommands:

    # bundle integrity (needs benchmark/juneau/panos/ for the byte check)
    python scripts/analysis/juneau_254.py verify

    # replication control + Juneau parity detail -> analysis_out/juneau_254/checks.json
    python scripts/analysis/juneau_254.py checks --write

``checks`` without ``--write`` recomputes and compares against the committed file. The
pure helpers are bayonne's (``scripts/analysis/bayonne_159.py``), imported, not copied.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bayonne_159 import (  # noqa: E402
    DEPLOYED, _check_or_write, load_peaks, load_records, parity_extras, repro_check)

SPLIT = "juneau"
BUNDLE = os.path.join(REPO, "benchmark", SPLIT)
OUT_DIR = os.path.join(REPO, "analysis_out", "juneau_254")
OP_CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
JUNEAU_CACHE = os.path.join(OP_CACHE_DIR, f"{SPLIT}.json")
REPRO_CACHE = os.path.join(OUT_DIR, "repro_op_cache", "laurens_mapillary.json")

EXPECTED_STRATA = {"top": 5, "random": 95, "empty": 25}
EXPECTED_DETECTIONS = 217          # >= 0.55 in records.jsonl: 211 judged + 6 unsure
EXPECTED_RIGS = {("Trimble", "Trimble mx7"): 120, ("GoPro", "GoPro Max"): 5}


def verify_bundle(bundle=BUNDLE):
    """``(problems, notes, facts)`` for the bundle; ``problems`` empty means OK.

    The byte check against ``imagery_manifest.json`` runs only where ``panos/`` exists
    (a developer checkout); on a clean clone the record-level checks still run.
    """
    problems, notes = [], []
    with open(os.path.join(bundle, "imagery_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)["panos"]
    recs = load_records(os.path.basename(bundle), os.path.dirname(os.path.dirname(bundle)))
    ids = [r["pano"]["panorama_id"] for r in recs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate panorama_id in records.jsonl")
    if set(ids) != set(manifest):
        problems.append("records.jsonl and imagery_manifest.json cover different panos")
    with open(os.path.join(bundle, "verdicts.json"), encoding="utf-8") as f:
        verdicts = json.load(f)
    if set(verdicts["panos"]) != set(ids):
        problems.append("verdicts.json and records.jsonl cover different panos")
    groups = dict(collections.Counter(r.get("benchmark_group") for r in recs))
    if groups != EXPECTED_STRATA:
        problems.append(f"strata {groups} != {EXPECTED_STRATA}")
    n_det = sum(1 for r in recs for d in r["detections"] if d["confidence"] >= DEPLOYED)
    if n_det != EXPECTED_DETECTIONS:
        problems.append(f"{n_det} detections >= {DEPLOYED}, expected {EXPECTED_DETECTIONS}")
    rigs = dict(collections.Counter((r["pano"]["camera_make"], r["pano"]["camera_model"])
                                    for r in recs))
    if rigs != EXPECTED_RIGS:
        problems.append(f"rigs {rigs} != {EXPECTED_RIGS}")
    for r in recs:
        if r["benchmark_group"] == "empty" and r["detections"]:
            problems.append(f"{r['pano']['panorama_id']}: 'empty' pano carries detections")
    panos_dir = os.path.join(bundle, "panos")
    if not os.path.isdir(panos_dir):
        notes.append("panos/ absent: byte check skipped")
    else:
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
        notes.append(f"{len(manifest)} panos byte-checked against imagery_manifest.json")
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


def build_checks():
    new, meta_new = load_peaks(REPRO_CACHE)
    old, _ = load_peaks(os.path.join(OP_CACHE_DIR, "laurens_mapillary.json"))
    peaks, _ = load_peaks(JUNEAU_CACHE)
    recs = load_records(SPLIT)
    extras = parity_extras(peaks, recs)
    n_rec = sum(1 for r in recs for d in r["detections"] if d["confidence"] >= DEPLOYED)
    n_cache = sum(1 for p in peaks.values() for q in p if q[2] >= DEPLOYED)
    seam_only = (bool(extras) and all(e["in_border_ring"] for x in extras for e in x["extra"])
                 and not any(x["missing_from_cache"] for x in extras))
    return {
        "replication_control": {
            "what": "laurens_mapillary re-extracted on klone with this branch's code "
                    "(scripts/analysis/juneau_254_gpu.sh, step repro) against its committed "
                    "analysis_out/op_cache/laurens_mapillary.json",
            "re_extracted_on": meta_new.get("device"),
            **repro_check(new, old)},
        "juneau_parity": {
            "what": f"Juneau cache peaks >= {DEPLOYED} against the {n_rec} "
                    "committed records.jsonl detections (low_floor_sweep.py parity reads "
                    "the same)",
            "records": n_rec, "cache": n_cache,
            "panos_differing": extras,
            "reading": ("identical: every record reproduced, no extra cache peak" if not extras
                        else "every extra cache peak sits in the 10-cell border ring at the "
                             "360 seam, the labeler extractor defect RampNet fixed in f4c71c8 "
                             "(#132); off the seam the two agree cell for cell" if seam_only
                        else "differences are not all seam peaks; read panos_differing"),
        },
    }


def cmd_checks(args):
    res = build_checks()
    if args.print:
        print(json.dumps(res, indent=1, sort_keys=True))
    return _check_or_write(res, os.path.join(OUT_DIR, "checks.json"), args.write)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="bundle integrity against imagery_manifest.json")
    v.add_argument("--bundle", default=BUNDLE)
    v.set_defaults(fn=cmd_verify)
    ck = sub.add_parser("checks", help="replication control + Juneau parity detail")
    ck.add_argument("--write", action="store_true")
    ck.add_argument("--print", action="store_true")
    ck.set_defaults(fn=cmd_checks)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
