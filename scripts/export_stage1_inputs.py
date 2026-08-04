"""Package the Stage 1 *inputs* as a HuggingFace dataset repo.

`rampnet-dataset` is what Stage 1 produced. This is what went into it: the three government curb
ramp inventories, the street centrelines used to place negative panoramas, and the manifests that
record exactly which panoramas the paper's run consumed.

The manifests are the reason this repo matters more than it looks. `generate_negative_panos.py`
samples with an unseeded RNG, so `negativepanosSHORTENED.jsonl` is the *only* record of which
negatives the paper used -- they cannot be regenerated, only downloaded.

`location_data/` is read from the training repo, where it is committed, and its sha256 values are
**asserted against the pinned values in docs/data_provenance.md** so this exporter cannot publish
inventories that drifted from the ones the paper ran on.

Build locally:

    python scripts/export_stage1_inputs.py \
        --manifests   <dir with the 5 manifest files> \
        --street-data <dir with the 3 raw "<City> - Streets.geojson"> \
        --out         dist/rampnet-stage1-inputs

Push:

    python scripts/export_stage1_inputs.py --manifests ... --street-data ... \
        --out dist/rampnet-stage1-inputs --push --repo-id projectsidewalk/rampnet-stage1-inputs
"""

import argparse
import datetime
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "hf_package" / "README.stage1_inputs_card.template.md"
COMMITTED_LOCATION_DATA = REPO_ROOT / "stage_one" / "dataset_generation" / "location_data"

# Pinned in docs/data_provenance.md section 3. Publishing anything else would be a silent lie
# about which inventories the paper ran on.
LOCATION_DATA_SHA256 = {
    "bend.geojson": "a0da4e016474c2c8fddcc6f77a7dd4a3aa5caaea455c839fad762d66a7af948e",
    "nyc.csv": "beea2b323d00d82192dd18ace3f257cef30ce3b579544d4e607fe7abe5e57f8c",
    "portland.geojson": "d5366a7e0d18f09f9ba49f1cbf7a26b99ee90633689dbe94cbde2a21bd395dbe",
}

MANIFESTS = [
    "finaldataset.jsonl",
    "dataset.jsonl",
    "negativepanosSHORTENED.jsonl",
    "negativepanos.jsonl",
    "all_locations.csv",
]

STREET_FILES = [
    "Bend - Streets.geojson",
    "New York - Streets.geojson",
    "Portland - Streets.geojson",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                # noqa: BLE001 - provenance is best-effort
        return "unknown"


def stage(src, dst, rows, folder):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    digest = sha256(dst)
    if digest != sha256(src):
        sys.exit("error: {} changed during copy".format(src.name))
    rows.append((folder + "/" + src.name, src.stat().st_size, digest))
    return digest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--street-data", required=True, type=Path)
    parser.add_argument("--location-data", type=Path, default=COMMITTED_LOCATION_DATA,
                        help="defaults to the copy committed in this repo")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-stage1-inputs")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    rows = []

    print("location_data/  (verifying against docs/data_provenance.md)")
    for name, expected in sorted(LOCATION_DATA_SHA256.items()):
        src = args.location_data / name
        if not src.is_file():
            sys.exit("error: missing {}".format(src))
        digest = stage(src, args.out / "location_data" / name, rows, "location_data")
        if digest != expected:
            sys.exit("error: {} sha256 {}\n       expected {}\n"
                     "       this is NOT the inventory the paper ran on".format(name, digest, expected))
        print("  {:<24} {:>13,}  verified".format(name, src.stat().st_size))

    print("manifests/")
    for name in MANIFESTS:
        src = args.manifests / name
        if not src.is_file():
            sys.exit("error: missing {}".format(src))
        stage(src, args.out / "manifests" / name, rows, "manifests")
        print("  {:<32} {:>13,}".format(name, src.stat().st_size))

    print("street_data/")
    for name in STREET_FILES:
        src = args.street_data / name
        if not src.is_file():
            sys.exit("error: missing {}".format(src))
        stage(src, args.out / "street_data" / name, rows, "street_data")
        print("  {:<32} {:>13,}".format(name, src.stat().st_size))

    table = ["| file | bytes | sha256 |", "| :--- | ---: | :--- |"]
    for path, size, digest in rows:
        table.append("| `{}` | {:,} | `{}` |".format(path, size, digest))
    total = sum(size for _, size, _ in rows)
    table.append("| **total** | **{:,}** | |".format(total))

    card = TEMPLATE.read_text(encoding="utf-8").format(
        contents_table="\n".join(table),
        git_commit=git_commit(),
        export_date=datetime.date.today().isoformat(),
        repo_id=args.repo_id,
    )
    (args.out / "README.md").write_text(card, encoding="utf-8")

    print("\n{} files, {:,} bytes ({:.2f} GB) -> {}".format(
        len(rows), total, total / 1e9, args.out))

    if not args.push:
        print("Not pushed. Re-run with --push --repo-id <org/name> to upload.")
        return

    from huggingface_hub import HfApi                # imported late: not needed for a local build
    api = HfApi()
    print("\nPushing to {}".format(args.repo_id))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(args.out),
                      commit_message="Add the paper's Stage 1 inputs: inventories, street data, manifests")
    print("Done: https://huggingface.co/datasets/{}".format(args.repo_id))


if __name__ == "__main__":
    main()
