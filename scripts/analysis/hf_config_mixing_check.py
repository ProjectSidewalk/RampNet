"""Can one HF dataset repo serve a loose-JPEG config AND a Parquet config? Measured answer: no.

This exists because the plan in `docs/replication.md` §4 used to be "put the round-1 crop set into
`rampnet-crop-model-dataset` (since renamed `-round2`) as Parquet under `round1_ps/`, leaving the
1,212 round-2 JPEGs loose".
That plan is impossible, and it fails **silently** -- the Parquet config's files resolve correctly
and then yield zero rows, with no error until you ask for a split.

The mechanism: `datasets` infers ONE builder module per repository, from the default config, and
applies it to every config in that repo. So:

  - jpgs default  -> both configs get Imagefolder; the Parquet config finds no images -> 0 rows
  - parquet default -> both configs get Parquet; the JPEG config would break instead

There is no arrangement that works, which is why round 1 ships as its own all-Parquet repo.

Run it yourself (needs a writable HF namespace; creates two tiny repos and deletes them):

    python scripts/analysis/hf_config_mixing_check.py --namespace <your-hf-user>
    python scripts/analysis/hf_config_mixing_check.py --namespace <your-hf-user> --keep

Verified 2026-08-05 with datasets 5.0.0 / huggingface_hub 1.24.0. If a later `datasets` gains
per-config module inference this check is how you would find out.
"""
import argparse
import io
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# A miniature of the round-1 export's schema -- deliberately inline rather than imported from
# scripts/export_crop_dataset.py, so this check keeps working if that schema changes.
SCHEMA = pa.schema([
    pa.field("crop_id", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
]).with_metadata({b"huggingface": b'{"info": {"features": {"crop_id": {"dtype": "string", '
                                 b'"_type": "Value"}, "image": {"_type": "Image"}}}}'})

MIXED_CARD = """---
license: mit
configs:
- config_name: round2
  default: true
  data_files:
  - split: train
    path: train/**
- config_name: round1
  data_files:
  - split: train
    path: round1/data/train/train-*.parquet
---
Throwaway, created by scripts/analysis/hf_config_mixing_check.py.
"""

PARQUET_ONLY_CARD = """---
license: mit
configs:
- config_name: round1
  data_files:
  - split: train
    path: round1/data/train/train-*.parquet
---
Throwaway control, created by scripts/analysis/hf_config_mixing_check.py.
"""


def jpeg_bytes(shade=90):
    buf = io.BytesIO()
    Image.new("RGB", (683, 2048), (10, 20, shade)).save(buf, format="JPEG", quality=60)
    return buf.getvalue()


def write_parquet(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(
        [{"crop_id": "a_-_11_22", "image": {"bytes": jpeg_bytes(), "path": "a.jpg"}}],
        schema=SCHEMA), path, compression="zstd")


def build_tree(root, card, with_jpegs):
    os.makedirs(root, exist_ok=True)
    write_parquet(os.path.join(root, "round1", "data", "train", "train-00000.parquet"))
    if with_jpegs:
        d = os.path.join(root, "train")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "aaaa1111_-_100_200.jpg"), "wb") as fh:
            fh.write(jpeg_bytes(140))
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)


def probe(api, repo_id, root, card, with_jpegs, configs):
    from datasets import load_dataset, load_dataset_builder
    build_tree(root, card, with_jpegs)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=root,
                      commit_message="hf_config_mixing_check")
    results = {}
    for config in configs:
        builder = type(load_dataset_builder(repo_id, config)).__name__.split("Tmp-")[0]
        try:
            rows = load_dataset(repo_id, config, split="train",
                                download_mode="force_redownload").num_rows
        except Exception as err:                    # noqa: BLE001 - the failure IS the result
            rows = "{}: {}".format(type(err).__name__, str(err)[:60])
        results[config] = (builder, rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True, help="HF user or org you can write to")
    parser.add_argument("--keep", action="store_true", help="leave the throwaway repos behind")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="hf_config_mixing_")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(workdir, "cache")
    from huggingface_hub import HfApi
    api = HfApi()

    mixed = "{}/tmp-hf-config-mixing-mixed".format(args.namespace)
    control = "{}/tmp-hf-config-mixing-control".format(args.namespace)
    failures = []
    try:
        print("== mixed repo: loose JPEGs (default config) + Parquet config ==")
        got = probe(api, mixed, os.path.join(workdir, "mixed"), MIXED_CARD, True,
                    ["round2", "round1"])
        for config, (builder, rows) in got.items():
            print("  {:<8} builder={:<12} rows={}".format(config, builder, rows))
        if got["round1"][0] != "Imagefolder":
            failures.append("expected the Parquet config to be built by Imagefolder here")
        if got["round1"][1] == 1:
            failures.append("the Parquet config returned data -- mixing may now be supported")

        print("\n== control: the identical Parquet config, no JPEGs in the repo ==")
        got = probe(api, control, os.path.join(workdir, "control"), PARQUET_ONLY_CARD, False,
                    ["round1"])
        for config, (builder, rows) in got.items():
            print("  {:<8} builder={:<12} rows={}".format(config, builder, rows))
        if got["round1"] != ("Parquet", 1):
            failures.append("the control failed: a parquet-only repo should just work")
    finally:
        if not args.keep:
            for repo in (mixed, control):
                try:
                    api.delete_repo(repo, repo_type="dataset")
                except Exception:                   # noqa: BLE001 - cleanup is best-effort
                    print("  note: could not delete {}".format(repo))

    print("\n" + "-" * 70)
    if failures:
        print("FINDING CHANGED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("CONFIRMED: one builder module per repo. A Parquet config inside a JPEG repo\n"
          "silently yields 0 rows, so formats cannot be mixed in one dataset repo.")


if __name__ == "__main__":
    main()
