"""Build the ``benchmark/manual_gold`` bundle's imagery + records (issue #58).

The 1,000-pano manual gold set's labels live in-repo (``manual_labels/*.txt``);
its images live in the **test split** of ``projectsidewalk/rampnet-dataset`` on
Hugging Face (21,441 panos — the gold panos are a subset). This script fetches
those 1,000 images into ``benchmark/manual_gold/panos/`` (git-ignored, like every
bundle's imagery) and writes the bundle's ``records.jsonl`` (pano metadata only —
RampNet's detections are added by ``scripts/export_gold_records.py``).

    # id-only audit: no image download. Verifies every gold id is in the HF test
    # split and none leaked into train/validation, and re-checks city-bundle overlap.
    python scripts/fetch_manual_gold.py --audit

    # full fetch (downloads the HF test split, ~44 GB -> run it on Hyak/makelab2):
    python scripts/fetch_manual_gold.py

    # or copy from an existing download_dataset.py output instead of the Hub:
    python scripts/fetch_manual_gold.py --source local --local-dataset ./dataset/test

Byte fidelity matters: a past gold-set re-eval moved P +2.2 / R -1.8 on JPEG
re-encoding alone (see docs/model_comparison.md). ``--source hf`` therefore
writes the parquet's **raw image bytes** untouched (``decode=False``), and
``--source local`` copies files byte-for-byte. The two sources are NOT
byte-identical to each other — ``download_dataset.py`` re-encodes at quality 95 —
so the bundle records which one built it (``bundle_meta.json``), and the
exporter's reproduction gate against the published gold-set numbers is the
arbiter of whether the difference matters.

Stage-1's auto-generated labels (``curb_ramp_points_normalized`` etc.) are
deliberately NOT copied into the records: the bundle's ground truth is the
manual labels, and carrying the weaker auto labels alongside invites mixups.
"""
import argparse
import datetime
import io
import json
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_DATASET = "projectsidewalk/rampnet-dataset"
LABELS_DIR = os.path.join(REPO_ROOT, "manual_labels")
BUNDLE_DIR = os.path.join(REPO_ROOT, "benchmark", "manual_gold")
CITY_BUNDLES = ("bend", "richmond", "clovis")
# HF row keys copied into each record's "pano" dict (verbatim, no reinterpretation).
PANO_META_KEYS = ("record_creation_time", "pano_coord", "pano_azimuth")


def gold_ids():
    ids = sorted(n[:-4] for n in os.listdir(LABELS_DIR) if n.endswith(".txt"))
    if not ids:
        raise SystemExit(f"no .txt label files in {LABELS_DIR}")
    return ids


def split_of(parquet_path):
    """Infer the split from a Hub parquet path. This dataset nests shards under a
    split directory (``.../test/data-00000-of-00128.parquet``); the basename
    pattern (``train-00000-of-00300.parquet``) is the other common Hub layout."""
    parts = parquet_path.replace("\\", "/").split("/")
    for token, split in (("train", "train"), ("validation", "validation"),
                         ("val", "validation"), ("test", "test")):
        if token in parts[:-1] or re.search(rf"(^|[-_.]){token}([-_.]|$)", parts[-1]):
            return split
    return None


def audit(ids):
    """Id-only membership + overlap checks; no image download.

    Reads just the ``pano_id`` column of every parquet shard via HTTP range
    requests (a few KB per shard instead of ~500 MB), so the whole 460 GB
    dataset audits in minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob(f"datasets/{HF_DATASET}/**/*.parquet")
    by_split = {"train": [], "validation": [], "test": []}
    unknown = []
    for f in files:
        (by_split[split_of(f)] if split_of(f) in by_split else unknown).append(f)
    if unknown:
        print(f"warning: {len(unknown)} parquet file(s) with unrecognized split name, "
              f"e.g. {unknown[0]}")
    for split, fl in by_split.items():
        if not fl:
            raise SystemExit(f"no parquet files found for split {split!r} — "
                             "has the dataset layout changed?")

    def read_ids(path):
        return set(pq.read_table(path, columns=["pano_id"], filesystem=fs)["pano_id"]
                   .to_pylist())

    gold = set(ids)
    ok = True
    for split, fl in by_split.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            split_ids = set().union(*ex.map(read_ids, fl))
        inter = gold & split_ids
        print(f"{split}: {len(split_ids)} panos; gold overlap {len(inter)}")
        if split == "test":
            missing = sorted(gold - split_ids)
            if missing:
                ok = False
                print(f"  MISSING from test split ({len(missing)}): {missing[:10]}"
                      + (" ..." if len(missing) > 10 else ""))
        elif inter:
            ok = False
            print(f"  LEAKED gold ids in {split} ({len(inter)}): {sorted(inter)[:10]}"
                  + (" ..." if len(inter) > 10 else ""))

    for city in CITY_BUNDLES:
        vpath = os.path.join(REPO_ROOT, "benchmark", city, "verdicts.json")
        with open(vpath, encoding="utf-8") as f:
            city_ids = set(json.load(f)["panos"])
        inter = sorted(gold & city_ids)
        print(f"benchmark/{city}: {len(city_ids)} panos; gold overlap {len(inter)}"
              + (f" -> {inter}" if inter else ""))

    print("\nAudit " + ("PASSED: gold set is fully in the test split, absent from "
                        "train/validation." if ok else "FAILED (see above)."))
    return 0 if ok else 1


def image_dims(jpeg_bytes):
    from PIL import Image
    with Image.open(io.BytesIO(jpeg_bytes)) as img:  # header read only, no full decode
        return img.width, img.height


def make_record(pano_id, width, height, meta):
    pano = {"panorama_id": pano_id, "width": width, "height": height}
    pano.update({k: meta[k] for k in PANO_META_KEYS if k in meta})
    return {"pano": pano}


def fetch_hf(ids, panos_dir):
    """Yield (pano_id, record) writing raw parquet image bytes to panos_dir."""
    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(HF_DATASET, split="test").cast_column("image", HFImage(decode=False))
    wanted = set(ids)
    for row in ds:
        pid = row["pano_id"]
        if pid not in wanted:
            continue
        data = row["image"]["bytes"]
        if not data:
            raise SystemExit(f"{pid}: test-split row has no embedded image bytes")
        with open(os.path.join(panos_dir, f"{pid}.jpg"), "wb") as f:
            f.write(data)
        w, h = image_dims(data)
        yield pid, make_record(pid, w, h, row)


def fetch_local(ids, panos_dir, local_dataset):
    """Yield (pano_id, record) copying byte-for-byte from a download_dataset.py
    output directory (``dataset/test``: <pid>.jpg + <pid>.json)."""
    for pid in ids:
        src = os.path.join(local_dataset, f"{pid}.jpg")
        if not os.path.exists(src):
            continue
        shutil.copyfile(src, os.path.join(panos_dir, f"{pid}.jpg"))
        with open(os.path.join(local_dataset, f"{pid}.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(src, "rb") as f:
            w, h = image_dims(f.read())
        yield pid, make_record(pid, w, h, meta)


def main():
    ap = argparse.ArgumentParser(description="Fetch the manual gold set's imagery (issue #58).")
    ap.add_argument("--audit", action="store_true",
                    help="Id-only split-membership + overlap audit; downloads no images.")
    ap.add_argument("--source", choices=["hf", "local"], default="hf",
                    help="'hf' = raw bytes from the Hub parquet (canonical); 'local' = "
                         "byte-for-byte copies from an existing dataset/test directory "
                         "(download_dataset.py output, which re-encoded at quality 95).")
    ap.add_argument("--local-dataset", default=os.path.join(REPO_ROOT, "dataset", "test"),
                    help="Source directory for --source local.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing records.jsonl (this DISCARDS any detections "
                         "an earlier export_gold_records.py run wrote into it).")
    args = ap.parse_args()

    ids = gold_ids()
    print(f"{len(ids)} gold label files in {LABELS_DIR}")
    if args.audit:
        sys.exit(audit(ids))

    records_path = os.path.join(BUNDLE_DIR, "records.jsonl")
    if os.path.exists(records_path) and not args.force:
        raise SystemExit(f"{records_path} already exists; re-fetching would discard any "
                         "exported detections in it. Pass --force to overwrite.")

    panos_dir = os.path.join(BUNDLE_DIR, "panos")
    os.makedirs(panos_dir, exist_ok=True)
    rows = (fetch_hf(ids, panos_dir) if args.source == "hf"
            else fetch_local(ids, panos_dir, args.local_dataset))

    records = dict(rows)
    missing = sorted(set(ids) - set(records))

    with open(records_path + ".tmp", "w", encoding="utf-8") as f:
        for pid in sorted(records):
            f.write(json.dumps(records[pid], ensure_ascii=False) + "\n")
    os.replace(records_path + ".tmp", records_path)

    meta = {
        "built": datetime.date.today().isoformat(),
        "source": args.source,
        "hf_dataset": HF_DATASET,
        "hf_split": "test",
        "n_panos": len(records),
        "imagery": "gsv",
    }
    with open(os.path.join(BUNDLE_DIR, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(records)} panos + records.jsonl to {BUNDLE_DIR}")
    if missing:
        raise SystemExit(f"{len(missing)} gold pano(s) NOT found in the {args.source} source "
                         f"(bundle is incomplete): {missing[:10]}"
                         + (" ..." if len(missing) > 10 else ""))
    print("All gold panos fetched. Next: scripts/export_gold_records.py (GPU).")


if __name__ == "__main__":
    main()
