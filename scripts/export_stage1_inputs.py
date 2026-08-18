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

`all_locations.csv` gets the same treatment, and it needs it more. Current `main` produces a
*different* file from the paper's: `combine_location_data.py` now seeds its shuffle
(`random.seed(42)`, added after publication) and `convert_date` now returns `""` where the
paper-era version returned `"2000-01-01"`, which changes 23,088 of 276,071 rows -- 8.36% of the
corpus (docs/data_provenance.md §3.2). So a `--manifests` directory holding a present-day
regeneration looks exactly like the paper's, and this exporter would publish it under a card that
calls it "the exact manifest the paper consumed". The pinned hash is what makes that impossible.

The other four manifests have no published hash to pin -- they are large and were rescued rather
than regenerated -- so `--expect-sha256` / `MANIFEST_SHA256` is the hook for adding one as it
becomes known, and every staged file's hash is printed and written into the card either way.

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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_export_common import git_commit, sha256_file  # noqa: E402

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

# Pinned in docs/data_provenance.md section 3.1. `all_locations.csv` is the one manifest that a
# present-day `combine_location_data.py` will happily regenerate into a different file (seeded
# shuffle, changed date semantics), so publishing an unverified copy under a paper-era claim is a
# live failure mode rather than a hypothetical one. Add entries here as other hashes are recorded.
MANIFEST_SHA256 = {
    "all_locations.csv": "06fec4e9a8077582deac12c3c303b89c8a2396ce3d78e7e923b0960a2c091a3b",
}

STREET_FILES = [
    "Bend - Streets.geojson",
    "New York - Streets.geojson",
    "Portland - Streets.geojson",
]


def stage(src, dst, rows, folder, expected=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    digest = sha256_file(dst)
    if digest != sha256_file(src):
        sys.exit("error: {} changed during copy".format(src.name))
    if expected and digest != expected:
        sys.exit("error: {} sha256 {}\n       expected {}\n"
                 "       this is NOT the file the paper ran on".format(src.name, digest, expected))
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
    parser.add_argument("--expect-sha256", action="append", metavar="FILE=SHA256",
                        help="fail unless FILE hashes to SHA256; repeatable. Adds to the pins "
                             "already in MANIFEST_SHA256/LOCATION_DATA_SHA256")
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-stage1-inputs")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Add the paper's Stage 1 inputs: inventories, street data, manifests",
                        help="the Hub commit message; say what this push actually carried")
    args = parser.parse_args()

    rows = []

    # One table of expected hashes for every staged file, so --expect-sha256 works uniformly and
    # a pin can be added for any of them without a second code path.
    expected_sha256 = dict(LOCATION_DATA_SHA256)
    expected_sha256.update(MANIFEST_SHA256)
    for pair in args.expect_sha256 or []:
        name, _, digest = pair.partition("=")
        if not digest:
            sys.exit("error: --expect-sha256 takes <filename>=<sha256>, got {!r}".format(pair))
        expected_sha256[name] = digest

    def stage_group(label, names, source_dir, folder):
        print("{}/".format(label))
        for name in names:
            src = source_dir / name
            if not src.is_file():
                sys.exit("error: missing {}".format(src))
            expected = expected_sha256.get(name)
            stage(src, args.out / folder / name, rows, folder, expected)
            print("  {:<32} {:>13,}{}".format(
                name, src.stat().st_size, "  verified" if expected else "  (no pinned hash)"))

    stage_group("location_data", sorted(LOCATION_DATA_SHA256), args.location_data, "location_data")
    stage_group("manifests", MANIFESTS, args.manifests, "manifests")
    stage_group("street_data", STREET_FILES, args.street_data, "street_data")

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
    # Unchanged files are skipped by the preupload check, so re-running this to correct the card
    # uploads only README.md -- at which point the first publication's fixed message is wrong.
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(args.out),
                      commit_message=args.message)
    print("Done: https://huggingface.co/datasets/{}".format(args.repo_id))


if __name__ == "__main__":
    main()
