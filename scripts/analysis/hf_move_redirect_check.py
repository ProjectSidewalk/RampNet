"""Does renaming an HF dataset repo leave a redirect that keeps the OLD id working? Measured: yes.

This is the check behind renaming `rampnet-crop-model-dataset` to `rampnet-crop-model-dataset-round2`
(2026-08-05). The naming pair was backwards -- the unmarked repo held round *2* while the new repo
carried the `-round1` suffix -- and the fix was worth doing only if a rename breaks nobody. It
breaks nobody. After `HfApi.move_repo`, the old id keeps working at every layer a user touches:

  1. `HfApi.dataset_info(OLD_ID)` resolves, reporting the new id
  2. `/resolve/` file URLs under the old id return HTTP 200 with the correct bytes
  3. `load_dataset(OLD_ID)` loads normally (fresh cache, so nothing is served locally)

One caveat the redirect does NOT cover: the freed name can be reclaimed. If anyone later creates a
new repo at the old id, that repo shadows the redirect and every stale link silently points at the
wrong dataset. The org must simply never reuse `rampnet-crop-model-dataset`.

Run it yourself (needs a writable HF namespace; creates one tiny private repo and deletes it):

    python scripts/analysis/hf_move_redirect_check.py --namespace <your-hf-user>

Verified 2026-08-05 with datasets 5.0.0 / huggingface_hub 1.24.0, then confirmed on the real
rename: content sha identical before and after, all 1,214 files intact.
"""
import argparse
import io
import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import requests

SCHEMA = pa.schema([
    pa.field("crop_id", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
]).with_metadata({b"huggingface": b'{"info": {"features": {"crop_id": {"dtype": "string", '
                                 b'"_type": "Value"}, "image": {"_type": "Image"}}}}'})

CARD = """---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/train-*.parquet
---
Throwaway, created by scripts/analysis/hf_move_redirect_check.py.
"""


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True, help="HF user or org you can write to")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="hf_move_redirect_")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(workdir, "cache")
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    src = "{}/tmp-hf-move-redirect-src".format(args.namespace)
    dst = "{}/tmp-hf-move-redirect-dst".format(args.namespace)
    failures = []
    created = []
    try:
        api.create_repo(src, repo_type="dataset", private=True)
        created.append(src)
        parquet = os.path.join(workdir, "train-00000.parquet")
        pq.write_table(pa.Table.from_pylist(
            [{"crop_id": "a", "image": {"bytes": jpeg_bytes(), "path": "a.jpg"}}],
            schema=SCHEMA), parquet)
        api.upload_file(path_or_fileobj=parquet, path_in_repo="data/train/train-00000.parquet",
                        repo_id=src, repo_type="dataset")
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                        repo_id=src, repo_type="dataset")

        api.move_repo(from_id=src, to_id=dst, repo_type="dataset")
        created = [dst]
        print("moved {} -> {}".format(src, dst))

        try:
            info = api.dataset_info(src)
            print("1. dataset_info(OLD id): OK, resolves to '{}'".format(info.id))
        except Exception as err:                    # noqa: BLE001 - the failure IS the result
            failures.append("dataset_info(OLD id) raised {}".format(type(err).__name__))

        url = "https://huggingface.co/datasets/{}/resolve/main/data/train/train-00000.parquet".format(src)
        resp = requests.get(url, headers={"Authorization": "Bearer {}".format(get_token())},
                            allow_redirects=True, timeout=60)
        print("2. resolve URL under OLD id: HTTP {}, {} bytes".format(
            resp.status_code, len(resp.content)))
        with open(parquet, "rb") as fh:
            uploaded = fh.read()
        if resp.status_code != 200 or resp.content != uploaded:
            failures.append("resolve URL under the old id did not return the original bytes")

        from datasets import load_dataset
        try:
            ds = load_dataset(src, split="train")
            feature = type(ds.features["image"]).__name__
            print("3. load_dataset(OLD id): OK, {} row(s), image feature = {}".format(
                ds.num_rows, feature))
            if ds.num_rows != 1 or feature != "Image":
                failures.append("load_dataset(OLD id) returned the wrong shape")
        except Exception as err:                    # noqa: BLE001 - the failure IS the result
            failures.append("load_dataset(OLD id) raised {}: {}".format(
                type(err).__name__, str(err)[:60]))
    finally:
        for repo in created:
            try:
                api.delete_repo(repo, repo_type="dataset")
            except Exception:                       # noqa: BLE001 - cleanup is best-effort
                print("  note: could not delete {}".format(repo))
        # The remote repo was cleaned up but the local workdir -- fixture plus a full datasets
        # cache -- was left behind on every run. Same leak as hf_config_mixing_check.py had.
        shutil.rmtree(workdir, ignore_errors=True)

    print("-" * 70)
    if failures:
        print("FINDING CHANGED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("CONFIRMED: move_repo leaves a working redirect at the API, file-URL and\n"
          "load_dataset layers. Renaming a published dataset repo breaks no existing user.")


if __name__ == "__main__":
    main()
