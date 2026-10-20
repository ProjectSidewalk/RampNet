#!/usr/bin/env python3
"""Build a mixed Stage 2 training directory: sparse-labeled panos + a GSV replay sample (#263).

``stage_two/train.py`` trains on every jpg/json pair in ``<data-root>/train``, sampled
uniformly, so a "1:1 Lyon/GSV" arm needs a train directory that holds exactly that mix. This
builds one out of symlinks, copying nothing:

    <out>/train/<id>.jpg, <id>.json   -> every pano of a build_sparse_labels.py output, plus
                                         a seeded uniform sample of the GSV train split,
                                         round(--ratio x n_sparse) panos
    <out>/val                         -> the GSV val directory, unchanged, so every arm is
                                         validated on the same panos
    <out>/mix_manifest.json           -> inputs, ratio, seed, counts, the chosen GSV ids, and
                                         train_sha256 over the sorted (name, source) list

Train with ``--data-root <out> --loss-mask``. ``--check`` re-derives the selection from the
inputs and compares it with the manifest and the links on disk. Nothing here touches the
network.

Usage:

    python scripts/mix_training_dirs.py --sparse /gscratch/scrubbed/$USER/sparse/lyon \\
        --gsv-root /gscratch/scrubbed/$USER/rampnet_dataset_v2 --ratio 1.0 --seed 0 \\
        --out /gscratch/scrubbed/$USER/mix263/lyon_5k_1to1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

MANIFEST = "mix_manifest.json"
SPARSE_MANIFEST = "labels_manifest.json"


def pairs(directory):
    """Sorted stems in ``directory`` that have both a .jpg and a .json."""
    names = os.listdir(directory)
    jpg = {n[:-4] for n in names if n.endswith(".jpg")}
    js = {n[:-5] for n in names if n.endswith(".json")}
    return sorted(jpg & js)


def plan(args):
    """(links {name: target}, manifest). Pure in the inputs and --seed."""
    sparse = Path(args.sparse).resolve()
    gsv_train = Path(args.gsv_root).resolve() / "train"
    gsv_val = Path(args.gsv_root).resolve() / "val"
    for d in (gsv_train, gsv_val):
        if not d.is_dir():
            raise SystemExit(f"no {d}")
    smpath = sparse / SPARSE_MANIFEST
    if not smpath.is_file():
        raise SystemExit(f"{smpath} missing: --sparse must be a build_sparse_labels.py output")
    sm = json.loads(smpath.read_text(encoding="utf-8"))
    sparse_ids = sorted(sm["pano_ids"])
    on_disk = set(pairs(sparse))
    missing = [pid for pid in sparse_ids if pid not in on_disk]
    if missing:
        raise SystemExit(f"{len(missing)} panos in {smpath} have no jpg+json, e.g. "
                         f"{missing[:3]}; run its --check")
    candidates = pairs(gsv_train)
    clash = sorted(set(candidates) & set(sparse_ids))
    if clash:
        raise SystemExit(f"{len(clash)} pano ids are in both the sparse set and the GSV "
                         f"train split, e.g. {clash[:3]}")
    n_gsv = int(round(args.ratio * len(sparse_ids)))
    if n_gsv > len(candidates):
        raise SystemExit(f"--ratio {args.ratio} needs {n_gsv} GSV panos; "
                         f"{gsv_train} has {len(candidates)}")
    gsv_ids = sorted(random.Random(args.seed).sample(candidates, n_gsv))

    links, rows = {}, []
    for source, base, ids in (("sparse", sparse, sparse_ids), ("gsv", gsv_train, gsv_ids)):
        for pid in ids:
            for ext in ("jpg", "json"):
                links[f"{pid}.{ext}"] = str(base / f"{pid}.{ext}")
            rows.append(f"{pid}\t{source}\n")
    h = hashlib.sha256("".join(sorted(rows)).encode("utf-8")).hexdigest()
    manifest = {
        "tool": "scripts/mix_training_dirs.py",
        "issue": "https://github.com/ProjectSidewalk/RampNet/issues/263",
        "sparse": {"dir": str(sparse), "labels_sha256": sm.get("labels_sha256"),
                   "city": sm.get("city"), "n": len(sparse_ids)},
        "gsv": {"train": str(gsv_train), "n_candidates": len(candidates), "n": n_gsv},
        "val": str(gsv_val),
        "ratio": args.ratio, "seed": args.seed,
        "counts": {"sparse": len(sparse_ids), "gsv": n_gsv, "train": len(sparse_ids) + n_gsv},
        "gsv_ids": gsv_ids,
        "train_sha256": h,
    }
    return links, manifest


def build(args):
    links, manifest = plan(args)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a mix is built once, into a new directory")
    (out / "train").mkdir(parents=True)
    for name, target in links.items():
        os.symlink(target, out / "train" / name)
    os.symlink(manifest["val"], out / "val", target_is_directory=True)
    with open(out / MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    c = manifest["counts"]
    print(f"{out}: {c['sparse']} sparse + {c['gsv']} GSV = {c['train']} train panos; "
          f"val -> {manifest['val']}; train_sha256 {manifest['train_sha256']}")
    return 0


def check(args):
    out = Path(args.out)
    mpath = out / MANIFEST
    if not mpath.is_file():
        raise SystemExit(f"--check: no {mpath}")
    have = json.loads(mpath.read_text(encoding="utf-8"))
    links, want = plan(args)
    problems = [f"manifest '{k}' differs from a rebuild" for k in want if want[k] != have.get(k)]
    train = out / "train"
    names = set(os.listdir(train)) if train.is_dir() else set()
    for name in sorted(names - set(links)):
        problems.append(f"train/{name} is not in this mix")
    for name, target in links.items():
        p = train / name
        if not p.is_symlink() or os.readlink(p) != target:
            problems.append(f"train/{name} does not link to {target}")
    if not (out / "val").is_symlink() or os.readlink(out / "val") != want["val"]:
        problems.append(f"val does not link to {want['val']}")
    for p in problems[:20]:
        print("FAIL", p)
    if problems:
        print(f"--check: {len(problems)} problem(s)")
        return 1
    print(f"--check OK: {want['counts']['train']} train panos, "
          f"train_sha256 {want['train_sha256']}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sparse", required=True, help="a build_sparse_labels.py --out directory")
    ap.add_argument("--gsv-root", required=True,
                    help="the Stage 2 dataset root (holds train/ and val/)")
    ap.add_argument("--ratio", type=float, required=True,
                    help="GSV panos per sparse pano, e.g. 1.0 for a 1:1 mix")
    ap.add_argument("--seed", type=int, default=0, help="seed for the GSV sample (default 0)")
    ap.add_argument("--out", required=True, help="new directory to build the mix in")
    ap.add_argument("--check", action="store_true",
                    help="re-derive the mix and compare with --out; writes nothing")
    args = ap.parse_args(argv)
    if args.ratio < 0:
        ap.error("--ratio must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    return check(args) if args.check else build(args)


if __name__ == "__main__":
    sys.exit(main())
