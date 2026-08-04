"""Package the benchmark panoramas as a HuggingFace dataset repo, in Parquet.

Parquet rather than loose image folders because that is what Hugging Face asked us for when they
approved hosting `rampnet-dataset` in July 2025 (see docs/replication.md), it is what makes the
dataset viewer work, and loose folders do not survive many more cities.

Three configs:

  native       the panoramas exactly as fetched -- 4096 to 16384 px wide depending on city/source
  4096x2048    the same panoramas at the model's input size, which is what GT reviewers actually
               saw (gt_gallery.py renders at 4096x2048 and never native), so this is the config a
               second rater needs
  galleries    the #55 incremental-FP crops the A/B reviewers saw

Configs are named by resolution, not by consumer: "model resolution" is a relative label that goes
wrong the moment the model's input size changes, and a published folder name cannot be fixed later
without replacing large blobs.

**The image bytes are embedded verbatim.** Rows carry `image` as {{bytes, path}}, so Parquet stores
the exact source bytes with no re-encode, and each row also carries the `sha256` of those bytes.
`verify` reads every Parquet back and re-hashes, so "the round trip preserved the pixels" is
checked rather than assumed -- which is the whole point of publishing what reviewers saw.

Labels are deliberately NOT here. `records.jsonl` and `verdicts.json` live in git, where they can
be revised; imagery is immutable once fetched, which is what makes this repo safe to grow one city
at a time.

Build locally:

    python scripts/export_benchmark.py build \
        --benchmark benchmark --panos-4096 <rendered dir> --galleries analysis_out/op \
        --out dist/rampnet-benchmark

Verify, then push:

    python scripts/export_benchmark.py verify --out dist/rampnet-benchmark
    python scripts/export_benchmark.py push   --out dist/rampnet-benchmark \
        --repo-id projectsidewalk/rampnet-benchmark
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # benchmark natives reach 16384x8192; not a decompression bomb

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "hf_package" / "README.benchmark_card.template.md"

NATIVE, MODEL_RES, GALLERIES = "native", "4096x2048", "galleries"
ROWS_PER_BATCH = 4                     # large images: keep the writer's working set small

SCHEMA = pa.schema([
    pa.field("pano_id", pa.string()),
    pa.field("city", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("sha256", pa.string()),
])

# The metadata key `datasets` reads to recover column semantics -- without it the viewer shows an
# opaque struct instead of an image.
HF_FEATURES = {
    "pano_id": {"dtype": "string", "_type": "Value"},
    "city": {"dtype": "string", "_type": "Value"},
    "image": {"_type": "Image"},
    "width": {"dtype": "int32", "_type": "Value"},
    "height": {"dtype": "int32", "_type": "Value"},
    "sha256": {"dtype": "string", "_type": "Value"},
}


def schema_with_metadata():
    return SCHEMA.with_metadata(
        {b"huggingface": json.dumps({"info": {"features": HF_FEATURES}}).encode()})


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                # noqa: BLE001 - provenance is best-effort
        return "unknown"


def write_parquet(dst, records):
    """Stream (pano_id, city, path) triples into one Parquet, embedding bytes verbatim."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    schema = schema_with_metadata()
    n = 0
    with pq.ParquetWriter(str(dst), schema, compression="zstd") as writer:
        batch = []
        for pano_id, city, src in records:
            data = src.read_bytes()
            with Image.open(src) as im:
                width, height = im.size
            batch.append({
                "pano_id": pano_id, "city": city,
                "image": {"bytes": data, "path": src.name},
                "width": width, "height": height, "sha256": sha256_bytes(data),
            })
            if len(batch) >= ROWS_PER_BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                n += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            n += len(batch)
    return n, dst.stat().st_size


def collect(benchmark, panos_4096, galleries):
    """Yield (config, city, [(id, city, path), ...]) for everything that exists on disk."""
    for panos_dir in sorted(Path(benchmark).glob("*/panos")):
        city = panos_dir.parent.name
        items = [(p.stem, city, p) for p in sorted(panos_dir.iterdir()) if p.is_file()]
        if items:
            yield NATIVE, city, items

    if panos_4096:
        for city_dir in sorted(Path(panos_4096).iterdir()):
            if not city_dir.is_dir():
                continue
            items = [(p.stem, city_dir.name, p) for p in sorted(city_dir.glob("*.jpg"))]
            if items:
                yield MODEL_RES, city_dir.name, items

    if galleries:
        for city_dir in sorted(Path(galleries).glob("*_incremental_fp")):
            city = city_dir.name[: -len("_incremental_fp")]
            items = [(p.stem, city, p) for p in sorted(city_dir.glob("*.png"))]
            if items:
                yield GALLERIES, city, items


def build(args):
    out = Path(args.out)
    index = {}
    print("{:<12} {:<20} {:>6} {:>14} {:>14}".format("config", "city", "rows", "bytes in", "parquet"))
    print("-" * 72)
    for config, city, items in collect(args.benchmark, args.panos_4096, args.galleries):
        dst = out / "data" / config / "{}.parquet".format(city)
        src_bytes = sum(p.stat().st_size for _, _, p in items)
        rows, size = write_parquet(dst, items)
        index.setdefault(config, []).append(city)
        print("{:<12} {:<20} {:>6,} {:>14,} {:>14,}".format(config, city, rows, src_bytes, size))

    if not index:
        sys.exit("error: nothing found to package -- check --benchmark / --panos-4096 paths")

    configs_yaml = []
    for config in (NATIVE, MODEL_RES, GALLERIES):
        if config not in index:
            continue
        configs_yaml.append("- config_name: {}".format(config))
        configs_yaml.append("  data_files:")
        for city in sorted(index[config]):
            configs_yaml.append("  - split: {}".format(city))
            configs_yaml.append("    path: data/{}/{}.parquet".format(config, city))

    total = sum(f.stat().st_size for f in out.rglob("*.parquet"))
    card = TEMPLATE.read_text(encoding="utf-8").format(
        configs_yaml="\n".join(configs_yaml),
        git_commit=git_commit(),
        export_date=datetime.date.today().isoformat(),
        repo_id=args.repo_id,
        n_cities=len(set(sum(index.values(), []))),
        total_gb="{:.2f}".format(total / 1e9),
    )
    (out / "README.md").write_text(card, encoding="utf-8")
    print("-" * 72)
    print("{:,} parquet bytes ({:.2f} GB) -> {}".format(total, total / 1e9, out))
    print("Now run: export_benchmark.py verify --out {}".format(out))


def verify(args):
    """Re-hash every embedded image straight out of the Parquet."""
    out = Path(args.out)
    files = sorted(out.rglob("*.parquet"))
    if not files:
        sys.exit("error: no parquet files under {}".format(out))
    bad = rows = 0
    for path in files:
        # Stream in small batches: a native split is >2 GB, and reading it whole would hold the
        # entire file (plus its Python copy) in memory just to hash it.
        n = 0
        parquet = pq.ParquetFile(str(path))
        for batch in parquet.iter_batches(batch_size=ROWS_PER_BATCH,
                                          columns=["pano_id", "image", "sha256"]):
            for row in batch.to_pylist():
                rows += 1
                n += 1
                if sha256_bytes(row["image"]["bytes"]) != row["sha256"]:
                    bad += 1
                    print("  MISMATCH {} in {}".format(row["pano_id"], path.name))
        print("  {:<44} {:>6,} rows OK".format(str(path.relative_to(out)), n))
    print("-" * 72)
    if bad:
        sys.exit("FAIL: {} of {:,} embedded images do not match their recorded sha256".format(bad, rows))
    print("PASS: {:,} embedded images round-trip byte-identical".format(rows))

    verify_against_manifests(out, Path(args.benchmark))


def verify_against_manifests(out, benchmark):
    """Tie the packaged native bytes to the sha256 pinned when each split was reviewed.

    The round-trip check above proves Parquet preserved whatever we put in. This proves what we
    put in is what the ground-truth reviewers actually judged -- a different claim, and the one
    that matters for anyone redoing a review. `benchmark/<city>/imagery_manifest.json` is committed
    and was written at review time.
    """
    native = out / "data" / NATIVE
    if not native.is_dir():
        return
    print()
    print("Cross-check: packaged native bytes vs review-time imagery_manifest.json")
    print("-" * 72)
    total = matched = absent = wrong = 0
    for path in sorted(native.glob("*.parquet")):
        city = path.stem
        manifest = benchmark / city / "imagery_manifest.json"
        if not manifest.is_file():
            print("  {:<20} no imagery_manifest.json -- skipped".format(city))
            continue
        pinned = json.loads(manifest.read_text(encoding="utf-8")).get("panos", {})
        table = pq.read_table(str(path), columns=["pano_id", "sha256"])   # skips the image bytes
        ids = table.column("pano_id").to_pylist()
        shas = table.column("sha256").to_pylist()
        ok = miss = bad = 0
        for pano_id, digest in zip(ids, shas):
            total += 1
            entry = pinned.get(pano_id)
            if entry is None:
                miss += 1
                absent += 1
            elif entry.get("sha256") == digest:
                ok += 1
                matched += 1
            else:
                bad += 1
                wrong += 1
        print("  {:<20} {:>4,} pinned  {:>3} not in manifest  {:>3} MISMATCH".format(
            city, ok, miss, bad))
    print("-" * 72)
    if wrong:
        sys.exit("FAIL: {} panoramas differ from the bytes their split was reviewed against".format(wrong))
    print("PASS: {:,}/{:,} native panoramas are the exact bytes reviewers judged{}".format(
        matched, total, " ({} not in any manifest)".format(absent) if absent else ""))


def card(args):
    """Re-render README.md from the template against an already-built package.

    Cards get revised far more often than 11 GB of Parquet does, and rebuilding the whole package
    to fix a sentence would rewrite every blob. This recomputes the template values by scanning the
    existing data/ tree and rewrites only the card; --push uploads only that file.
    """
    out = Path(args.out)
    index = {}
    for path in sorted((out / "data").glob("*/*.parquet")):
        index.setdefault(path.parent.name, []).append(path.stem)
    if not index:
        sys.exit("error: no built package under {} -- run build first".format(out))

    configs_yaml = []
    for config in (NATIVE, MODEL_RES, GALLERIES):
        if config not in index:
            continue
        configs_yaml.append("- config_name: {}".format(config))
        configs_yaml.append("  data_files:")
        for city in sorted(index[config]):
            configs_yaml.append("  - split: {}".format(city))
            configs_yaml.append("    path: data/{}/{}.parquet".format(config, city))

    total = sum(f.stat().st_size for f in out.rglob("*.parquet"))
    (out / "README.md").write_text(TEMPLATE.read_text(encoding="utf-8").format(
        configs_yaml="\n".join(configs_yaml),
        git_commit=git_commit(),
        export_date=datetime.date.today().isoformat(),
        repo_id=args.repo_id,
        n_cities=len(set(sum(index.values(), []))),
        total_gb="{:.2f}".format(total / 1e9),
    ), encoding="utf-8")
    print("Re-rendered {}".format(out / "README.md"))

    if not args.push:
        print("Not pushed. Add --push to upload just the card.")
        return
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out / "README.md"), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message="Clarify that this benchmark is post-publication, not the paper's evaluation")
    print("Card updated: https://huggingface.co/datasets/{}".format(args.repo_id))


def push(args):
    from huggingface_hub import HfApi                # imported late: not needed to build or verify
    out = Path(args.out)
    if not (out / "README.md").exists():
        sys.exit("error: build first")
    api = HfApi()
    print("Pushing {} to {}".format(out, args.repo_id))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out),
                      commit_message="Add benchmark panoramas (native + 4096x2048) and A/B galleries")
    print("Done: https://huggingface.co/datasets/{}".format(args.repo_id))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["build", "verify", "card", "push"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "benchmark",
                        help="repo benchmark/ dir; reads <city>/panos/")
    parser.add_argument("--panos-4096", type=Path, default=None,
                        help="directory of <city>/<pano>.jpg rendered at 4096x2048")
    parser.add_argument("--galleries", type=Path, default=None,
                        help="directory holding <city>_incremental_fp/ PNG crops")
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-benchmark")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="with `card`: upload only README.md")
    args = parser.parse_args()
    {"build": build, "verify": verify, "card": card, "push": push}[args.mode](args)


if __name__ == "__main__":
    main()
