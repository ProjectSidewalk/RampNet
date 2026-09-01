"""Package the benchmark panoramas as a HuggingFace dataset repo, in Parquet.

Parquet rather than loose image folders because that is what Hugging Face asked us for when they
approved hosting `rampnet-dataset` in July 2025 (see docs/replication.md), it is what makes the
dataset viewer work, and loose folders do not survive many more cities.

Four configs:

  records      the ground truth itself -- per-pano metadata, detections with their human verdict,
               and reviewer-marked missed ramps. A few MB, so a verdict fix never touches imagery
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

Labels are deliberately NOT in the imagery configs. `records.jsonl` and `verdicts.json` live in
git, where they can be revised; imagery is immutable once fetched, which is what makes this repo
safe to grow one city at a time.

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
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_export_common import (  # noqa: E402
    clear_build_dir, git_commit, hf_features_metadata, hf_value, sha256_bytes)

Image.MAX_IMAGE_PIXELS = None          # benchmark natives reach 16384x8192; not a decompression bomb

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "hf_package" / "README.benchmark_card.template.md"

NATIVE, MODEL_RES, GALLERIES, RECORDS = "native", "4096x2048", "galleries", "records"
CONFIG_ORDER = (NATIVE, MODEL_RES, GALLERIES, RECORDS)
IMAGERY_CONFIGS = (NATIVE, MODEL_RES, GALLERIES)
ROWS_PER_BATCH = 4                     # large images: keep the writer's working set small

# What `build` records about itself, so `card` does not have to re-derive the published config set
# from whatever happens to be on the local disk. See load_index().
MANIFEST_NAME = "build_manifest.json"

# `panos/` is populated by the fetchers and is not in git, so a directory listing is not a
# statement of intent. `benchmark/manual_gold/panos` holds the paper's 1,000-panorama gold set --
# imagery already published inside `rampnet-dataset`, with no `imagery_manifest.json` here and no
# `verdicts.json`, so it is neither a benchmark split nor cross-checkable. Globbing `benchmark/*`
# swept it into `native` as a 10th split; this list is what stops that.
# tests/test_export_benchmark.py::test_split_allowlist_matches_the_analysis_registry keeps it in
# step with scripts/analysis/miss_decomposition.py.
BENCHMARK_SPLITS = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville",
                    "laurens_gsv", "laurens_mapillary", "morgantown", "paterson",
                    "richmond", "sao_paulo")
EXCLUDED_SPLITS = {"manual_gold": "the paper's gold set -- published in rampnet-dataset already"}

# A `panos/` directory can pick up .DS_Store, Thumbs.db, a leftover .json sidecar or a
# partly-downloaded .jpg.tmp. Any of those reaches PIL and kills an 11 GB build mid-run, after
# multi-GB parquets are already written and with no resume.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# low_floor_sweep.py:852 names a gallery crop `{pano}_{x:.5f}_{y:.5f}`, so the *stem* is a
# detection tag, not a panorama id. Panorama ids themselves contain underscores, so the split has
# to come off the right-hand end and both trailing fields must parse as fixed-precision floats.
GALLERY_ID_RE = re.compile(r"^(?P<pano>.+)_(?P<x>-?\d+\.\d+)_(?P<y>-?\d+\.\d+)$")

# verdicts.json stores per-detection judgments as a mix of bool and string. Normalise to readable
# strings; the mapping is documented in the card so nothing is lost.
VERDICT_LABELS = {True: "correct", False: "incorrect"}

RECORDS_SCHEMA = pa.schema([
    pa.field("pano_id", pa.string()),
    pa.field("city", pa.string()),
    pa.field("source", pa.string()),
    pa.field("capture_date", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("lat", pa.float64()),
    pa.field("lng", pa.float64()),
    pa.field("camera_heading", pa.float64()),
    pa.field("copyright", pa.string()),
    pa.field("label_type", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("model_training_date", pa.string()),
    pa.field("detections", pa.list_(pa.struct([
        pa.field("x_normalized", pa.float64()),
        pa.field("y_normalized", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("verdict", pa.string()),
    ]))),
    pa.field("missed", pa.list_(pa.struct([
        pa.field("x_normalized", pa.float64()),
        pa.field("y_normalized", pa.float64()),
        pa.field("unsure", pa.bool_()),
    ]))),
    pa.field("no_missed", pa.bool_()),
    pa.field("review_group", pa.string()),
])

RECORDS_FEATURES = {
    "pano_id": hf_value("string"), "city": hf_value("string"), "source": hf_value("string"),
    "capture_date": hf_value("string"), "width": hf_value("int32"), "height": hf_value("int32"),
    "lat": hf_value("float64"), "lng": hf_value("float64"),
    "camera_heading": hf_value("float64"),
    "copyright": hf_value("string"), "label_type": hf_value("string"),
    "model_id": hf_value("string"), "model_training_date": hf_value("string"),
    "detections": [{"x_normalized": hf_value("float64"), "y_normalized": hf_value("float64"),
                    "confidence": hf_value("float64"), "verdict": hf_value("string")}],
    "missed": [{"x_normalized": hf_value("float64"), "y_normalized": hf_value("float64"),
                "unsure": hf_value("bool")}],
    "no_missed": hf_value("bool"), "review_group": hf_value("string"),
}

SCHEMA = pa.schema([
    pa.field("pano_id", pa.string()),
    pa.field("city", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("sha256", pa.string()),
])

HF_FEATURES = {
    "pano_id": hf_value("string"), "city": hf_value("string"), "image": {"_type": "Image"},
    "width": hf_value("int32"), "height": hf_value("int32"), "sha256": hf_value("string"),
}

# Galleries are crops of a detection, not panoramas, so they get their own schema. `crop_id` is the
# file stem; `pano_id` is parsed back out of it so the documented join against `records` actually
# returns rows -- naming the stem `pano_id` (as the first version did) made that join silently
# empty. Same trap tests/test_export_crop_dataset.py guards for the crop set.
GALLERY_SCHEMA = pa.schema([
    pa.field("crop_id", pa.string()),
    pa.field("pano_id", pa.string()),
    pa.field("city", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("x_normalized", pa.float64()),
    pa.field("y_normalized", pa.float64()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("sha256", pa.string()),
])

GALLERY_FEATURES = {
    "crop_id": hf_value("string"), "pano_id": hf_value("string"), "city": hf_value("string"),
    "image": {"_type": "Image"},
    "x_normalized": hf_value("float64"), "y_normalized": hf_value("float64"),
    "width": hf_value("int32"), "height": hf_value("int32"), "sha256": hf_value("string"),
}


def schema_for(config):
    """(pyarrow schema with HF metadata, id column name) for an imagery config."""
    if config == GALLERIES:
        return GALLERY_SCHEMA.with_metadata(hf_features_metadata(GALLERY_FEATURES)), "crop_id"
    return SCHEMA.with_metadata(hf_features_metadata(HF_FEATURES)), "pano_id"


def parse_gallery_id(stem):
    """`abc123_0.51234_0.44321` -> ("abc123", 0.51234, 0.44321); (None, None, None) if it is not
    a tag in that shape, so a renamed or hand-added crop is reported rather than mis-parsed."""
    match = GALLERY_ID_RE.match(stem)
    if not match:
        return None, None, None
    return match.group("pano"), float(match.group("x")), float(match.group("y"))


def write_parquet(dst, records, config):
    """Stream (id, city, path) triples into one Parquet, embedding bytes verbatim."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    schema, _ = schema_for(config)
    is_gallery = config == GALLERIES
    n = unparsed = 0
    with pq.ParquetWriter(str(dst), schema, compression="zstd") as writer:
        batch = []
        for item_id, city, src in records:
            data = src.read_bytes()
            with Image.open(src) as im:
                width, height = im.size
            row = {"city": city, "image": {"bytes": data, "path": src.name},
                   "width": width, "height": height, "sha256": sha256_bytes(data)}
            if is_gallery:
                pano_id, x, y = parse_gallery_id(item_id)
                unparsed += pano_id is None
                row.update({"crop_id": item_id, "pano_id": pano_id,
                            "x_normalized": x, "y_normalized": y})
            else:
                row["pano_id"] = item_id
            batch.append(row)
            if len(batch) >= ROWS_PER_BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                n += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            n += len(batch)
    if unparsed:
        print("  note: {:,} gallery crops are not `<pano>_<x>_<y>` tags; their pano_id is null"
              .format(unparsed))
    return n, dst.stat().st_size


def has_reviewed_splits(benchmark):
    """True when at least one allowlisted split has both records.jsonl and verdicts.json."""
    return any((Path(benchmark) / city / "records.jsonl").is_file()
               and (Path(benchmark) / city / "verdicts.json").is_file()
               for city in BENCHMARK_SPLITS)


def build_records(benchmark, out):
    """Join records.jsonl + verdicts.json into one Parquet per city -- the ground truth itself.

    Issue #21 asks for the *ground truth* as a dataset, with per-pano `source`, capture date,
    camera heading and source attribution, not just pixels. All of it already exists in the two
    committed files; this only reshapes them, so git stays the source of truth and the config is
    regenerable rather than a second original.

    Kept as its own config on purpose: it is a few MB against 11.41 GB of imagery, so labels can be
    corrected -- and verdicts do get revised -- without replacing a single image blob.
    """
    written = []
    schema = RECORDS_SCHEMA.with_metadata(hf_features_metadata(RECORDS_FEATURES))
    for records_path in sorted(Path(benchmark).glob("*/records.jsonl")):
        city = records_path.parent.name
        if city not in BENCHMARK_SPLITS:
            continue
        verdicts_path = records_path.parent / "verdicts.json"
        if not verdicts_path.is_file():
            continue
        judged = json.loads(verdicts_path.read_text(encoding="utf-8")).get("panos", {})

        rows = []
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pano = record.get("pano", {})
            pano_id = pano.get("panorama_id")
            verdict = judged.get(pano_id)
            if verdict is None:            # only panoramas that were actually reviewed
                continue
            dets = record.get("detections", [])
            marks = verdict.get("dets", [])
            rows.append({
                "pano_id": pano_id,
                "city": city,
                "source": pano.get("source"),
                "capture_date": pano.get("capture_date"),
                "width": pano.get("width"),
                "height": pano.get("height"),
                "lat": pano.get("lat"),
                "lng": pano.get("lng"),
                "camera_heading": pano.get("camera_heading"),
                "copyright": pano.get("copyright"),
                "label_type": record.get("label_type"),
                "model_id": record.get("model_id"),
                "model_training_date": record.get("model_training_date"),
                "detections": [{
                    "x_normalized": d.get("x_normalized"),
                    "y_normalized": d.get("y_normalized"),
                    "confidence": d.get("confidence"),
                    # marks is index-aligned with detections; anything unjudged stays None
                    "verdict": VERDICT_LABELS.get(marks[i], marks[i]) if i < len(marks) else None,
                } for i, d in enumerate(dets)],
                "missed": [{
                    "x_normalized": m.get("x"),
                    "y_normalized": m.get("y"),
                    "unsure": bool(m.get("unsure", False)),
                } for m in verdict.get("missed", [])],
                "no_missed": bool(verdict.get("no_missed", False)),
                "review_group": verdict.get("group"),
            })

        if not rows:
            continue
        dst = out / "data" / RECORDS / "{}.parquet".format(city)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(dst), compression="zstd")
        n_det = sum(len(r["detections"]) for r in rows)
        n_missed = sum(len(r["missed"]) for r in rows)
        written.append((city, len(rows), n_det, n_missed, dst.stat().st_size))
    return written


def image_items(directory, city):
    """Sorted (stem, city, path) for the image files in one directory, ignoring anything else."""
    return [(p.stem, city, p) for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def collect(benchmark, panos_4096, galleries):
    """Yield (config, city, [(id, city, path), ...]) for everything that exists on disk.

    Only `BENCHMARK_SPLITS` are considered -- see the constant for why a bare glob is wrong here.
    """
    for panos_dir in sorted(Path(benchmark).glob("*/panos")):
        city = panos_dir.parent.name
        if city not in BENCHMARK_SPLITS:
            if city in EXCLUDED_SPLITS:
                print("  skipping {}/panos -- {}".format(city, EXCLUDED_SPLITS[city]))
            continue
        items = image_items(panos_dir, city)
        if items:
            yield NATIVE, city, items

    if panos_4096:
        for city_dir in sorted(Path(panos_4096).iterdir()):
            if not city_dir.is_dir() or city_dir.name not in BENCHMARK_SPLITS:
                continue
            items = image_items(city_dir, city_dir.name)
            if items:
                yield MODEL_RES, city_dir.name, items

    if galleries:
        for city_dir in sorted(Path(galleries).glob("*_incremental_fp")):
            city = city_dir.name[: -len("_incremental_fp")]
            if city not in BENCHMARK_SPLITS:
                continue
            items = image_items(city_dir, city)
            if items:
                yield GALLERIES, city, items


def configs_yaml(index):
    """The `configs:` block the Hub reads. One definition, used by both build and card.

    It used to be built twice, character-for-character, and the copy in `card` is the one that
    reaches the Hub on a cheap re-push -- so a divergence would surface as a published README that
    disagrees with the repo's own data/ tree.
    """
    lines = []
    for config in CONFIG_ORDER:
        if config not in index:
            continue
        lines.append("- config_name: {}".format(config))
        lines.append("  data_files:")
        for city in sorted(index[config]):
            lines.append("  - split: {}".format(city))
            lines.append("    path: data/{}/{}.parquet".format(config, city))
    return "\n".join(lines)


def scan_index(out):
    """{config: [city, ...]} from the Parquet actually present under out/data."""
    index = {}
    for path in sorted((out / "data").glob("*/*.parquet")):
        index.setdefault(path.parent.name, []).append(path.stem)
    return index


def config_bytes(out):
    """{config: total parquet bytes} for what is on this disk right now."""
    sizes = {}
    for path in (Path(out) / "data").glob("*/*.parquet"):
        sizes[path.parent.name] = sizes.get(path.parent.name, 0) + path.stat().st_size
    return sizes


def write_manifest(out, configs, sizes):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / MANIFEST_NAME).write_text(json.dumps(
        {"configs": configs, "sizes": sizes, "git_commit": git_commit(),
         "written_at": datetime.date.today().isoformat()}, indent=2), encoding="utf-8")


def save_index(out, index):
    """Record what this package contains, so a later partial rebuild cannot forget the rest."""
    merged = load_manifest(out)
    merged.update(dict((config, sorted(set(cities))) for config, cities in index.items()))
    # Sizes travel with the config list for the same reason: the card advertises a total, and a
    # partial rebuild only holds its own configs on disk. Record each rebuilt config's byte total
    # so the next card can add back the ones it did not touch. See package_bytes().
    sizes = load_sizes(out)
    on_disk = config_bytes(out)
    sizes.update(dict((config, on_disk[config]) for config in index if config in on_disk))
    write_manifest(out, merged, sizes)


def _manifest_field(out, field):
    path = Path(out) / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(field, {})
    except ValueError:
        return {}


def load_manifest(out):
    return _manifest_field(out, "configs")


def load_sizes(out):
    return _manifest_field(out, "sizes")


def package_bytes(out, index):
    """Total parquet bytes of the *published* package, not merely of what this run rebuilt.

    render_card summed out.rglob("*.parquet"), which is right for a full build and wrong for every
    partial one: re-pushing one config advertised that config as the whole dataset -- `galleries`
    alone would have published "0.40 GB" over the 11.41 GB actually on the Hub. That is the same
    hole as the config list, one field over, so it closes the same way. Configs present on this
    disk are authoritative for their own size; the rest come from the manifest.
    """
    sizes = load_sizes(out)
    sizes.update(config_bytes(out))
    return sum(size for config, size in sizes.items() if config in index)


def load_index(out, allow_partial=False):
    """The config set the *published* package has, not merely what this machine holds.

    `records` mode exists so a verdict fix costs a few MB instead of re-uploading 11.41 GB, which
    means it gets run against an --out holding only `data/records/`. Deriving the card's config
    list from that directory alone emitted a one-config README and `--push` uploaded it, leaving
    the image parquets on the Hub present but unreferenced: `load_dataset(..., "4096x2048")` then
    fails and the viewer loses three configs. So the manifest written at build time is unioned in,
    and a card that would still drop an imagery config is refused rather than rendered.
    """
    index = dict((config, sorted(set(cities))) for config, cities in load_manifest(out).items())
    for config, cities in scan_index(out).items():
        index[config] = sorted(set(index.get(config, [])) | set(cities))
    if not index:
        sys.exit("error: no built package under {} -- run build first".format(out))
    missing = [c for c in IMAGERY_CONFIGS if c not in index]
    if missing and not allow_partial:
        sys.exit(
            "error: this would publish a card declaring only {}, dropping {}.\n"
            "       The image parquets stay on the Hub but become unreferenced, so\n"
            "       load_dataset(repo, \"{}\") starts failing and the viewer loses them.\n"
            "       Fix: point --out at the package built by `build` -- its {} records\n"
            "       every config -- or pass --allow-partial if this repo really is\n"
            "       records-only.".format(
                ", ".join(sorted(index)), ", ".join(missing), missing[0], MANIFEST_NAME))
    return index


def split_date_range(benchmark, cities):
    """"between <first> and <last>" over the packaged splits' review dates, from verdicts.json.

    Hardcoded in the template until São Paulo landed on 2026-08-01 and silently fell outside the
    stated range. Nothing recomputed it, because the card's counts are interpolated and its dates
    were prose.
    """
    dates = []
    for city in cities:
        path = Path(benchmark) / city / "verdicts.json"
        if not path.is_file():
            continue
        stamp = json.loads(path.read_text(encoding="utf-8")).get("exported_at")
        if stamp:
            dates.append(stamp[:10])
    if not dates:
        return "on dates recorded in each split's verdicts.json"
    if min(dates) == max(dates):
        return "on {}".format(min(dates))
    return "between {} and {}".format(min(dates), max(dates))


def render_card(out, benchmark, repo_id, index):
    cities = sorted(set(sum(index.values(), [])))
    total = package_bytes(out, index)
    card_text = TEMPLATE.read_text(encoding="utf-8").format(
        configs_yaml=configs_yaml(index),
        git_commit=git_commit(),
        export_date=datetime.date.today().isoformat(),
        repo_id=repo_id,
        n_cities=len(cities),
        split_date_range=split_date_range(benchmark, cities),
        total_gb="{:.2f}".format(total / 1e9),
    )
    (out / "README.md").write_text(card_text, encoding="utf-8")
    return total


def build(args):
    out = Path(args.out)
    index = {}
    print("{:<12} {:<20} {:>6} {:>14} {:>14}".format("config", "city", "rows", "bytes in", "parquet"))
    print("-" * 72)
    cleared = set()
    for config, city, items in collect(args.benchmark, args.panos_4096, args.galleries):
        if config not in cleared:
            # Only the configs this run rebuilds: clearing all of data/ would delete a config
            # produced by an earlier invocation that passed different --panos-4096/--galleries.
            clear_build_dir(out, "data/{}".format(config))
            cleared.add(config)
        dst = out / "data" / config / "{}.parquet".format(city)
        src_bytes = sum(p.stat().st_size for _, _, p in items)
        rows, size = write_parquet(dst, items, config)
        index.setdefault(config, []).append(city)
        print("{:<12} {:<20} {:>6,} {:>14,} {:>14,}".format(config, city, rows, src_bytes, size))

    # Only clear the records config if this run can actually rebuild it -- a --benchmark path with
    # no reviewed splits must not silently delete a records config an earlier run produced.
    if has_reviewed_splits(args.benchmark):
        clear_build_dir(out, "data/{}".format(RECORDS))
    for city, n_panos, n_det, n_missed, size in build_records(args.benchmark, out):
        index.setdefault(RECORDS, []).append(city)
        print("{:<12} {:<20} {:>6,} {:>14} {:>14,}".format(
            RECORDS, city, n_panos, "{} det/{} miss".format(n_det, n_missed), size))

    if not index:
        sys.exit("error: nothing found to package -- check --benchmark / --panos-4096 paths")

    save_index(out, index)
    total = render_card(out, args.benchmark, args.repo_id, load_index(out, args.allow_partial))
    print("-" * 72)
    print("{:,} parquet bytes ({:.2f} GB) -> {}".format(total, total / 1e9, out))
    print("Now run: export_benchmark.py verify --out {}".format(out))


def verify(args):
    """Re-hash every embedded image straight out of the Parquet."""
    out = Path(args.out)
    # The `records` config carries labels, not pixels -- nothing to re-hash there.
    files = [p for p in sorted(out.rglob("*.parquet")) if p.parent.name != RECORDS]
    if not files:
        sys.exit("error: no image parquet files under {}".format(out))
    bad = rows = 0
    for path in files:
        # Stream in small batches: a native split is >2 GB, and reading it whole would hold the
        # entire file (plus its Python copy) in memory just to hash it.
        n = 0
        parquet = pq.ParquetFile(str(path))
        id_column = "crop_id" if "crop_id" in parquet.schema_arrow.names else "pano_id"
        for batch in parquet.iter_batches(batch_size=ROWS_PER_BATCH,
                                          columns=[id_column, "image", "sha256"]):
            for row in batch.to_pylist():
                rows += 1
                n += 1
                if sha256_bytes(row["image"]["bytes"]) != row["sha256"]:
                    bad += 1
                    print("  MISMATCH {} in {}".format(row[id_column], path.name))
        print("  {:<44} {:>6,} rows OK".format(str(path.relative_to(out)), n))
    print("-" * 72)
    if bad:
        sys.exit("FAIL: {} of {:,} embedded images do not match their recorded sha256".format(bad, rows))
    print("PASS: {:,} embedded images round-trip byte-identical".format(rows))

    verify_against_manifests(out, Path(args.benchmark), args.allow_unpinned)


def verify_against_manifests(out, benchmark, allow_unpinned=False):
    """Tie the packaged native bytes to the sha256 pinned when each split was reviewed.

    The round-trip check above proves Parquet preserved whatever we put in. This proves what we
    put in is what the ground-truth reviewers actually judged -- a different claim, and the one
    that matters for anyone redoing a review. `benchmark/<city>/imagery_manifest.json` is committed
    and was written at review time.

    A split with no manifest, and a panorama absent from one, both used to print a note, increment
    a counter and still exit PASS -- so "these are the exact bytes reviewers judged" was printed
    over panoramas nothing had pinned. Both now fail, because an unpinned panorama is precisely the
    case this check exists to catch. `--allow-unpinned` downgrades them for a split still being
    assembled.
    """
    native = out / "data" / NATIVE
    if not native.is_dir():
        return
    print()
    print("Cross-check: packaged native bytes vs review-time imagery_manifest.json")
    print("-" * 72)
    total = matched = absent = wrong = 0
    unpinned_splits = []
    for path in sorted(native.glob("*.parquet")):
        city = path.stem
        manifest = benchmark / city / "imagery_manifest.json"
        if not manifest.is_file():
            n_rows = pq.ParquetFile(str(path)).metadata.num_rows
            total += n_rows
            absent += n_rows
            unpinned_splits.append(city)
            print("  {:<20} {:>4} rows  NO imagery_manifest.json".format(city, n_rows))
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
    if absent and not allow_unpinned:
        sys.exit(
            "FAIL: {} of {:,} native panoramas are pinned by no imagery_manifest.json{}, so\n"
            "      nothing ties them to the verdicts they are published alongside.\n"
            "      Fix: run `python scripts/analysis/imagery_manifest.py` for those splits,\n"
            "      or pass --allow-unpinned to publish them as unpinned anyway.".format(
                absent, total,
                " (whole splits: {})".format(", ".join(unpinned_splits)) if unpinned_splits else ""))
    print("PASS: {:,}/{:,} native panoramas are the exact bytes reviewers judged{}".format(
        matched, total, " ({} unpinned, allowed)".format(absent) if absent else ""))


def card(args):
    """Re-render README.md from the template against an already-built package.

    Cards get revised far more often than 11 GB of Parquet does, and rebuilding the whole package
    to fix a sentence would rewrite every blob. This recomputes the template values from the build
    manifest plus the existing data/ tree and rewrites only the card; --push uploads only that file.
    """
    out = Path(args.out)
    index = load_index(out, args.allow_partial)
    render_card(out, args.benchmark, args.repo_id, index)
    print("Re-rendered {} ({} configs: {})".format(
        out / "README.md", len(index), ", ".join(c for c in CONFIG_ORDER if c in index)))

    if not args.push:
        print("Not pushed. Add --push to upload just the card.")
        return
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out / "README.md"), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message="Clarify that this benchmark is post-publication, not the paper's evaluation")
    print("Card updated: https://huggingface.co/datasets/{}".format(args.repo_id))


def records(args):
    """Build (or rebuild) just the `records` config, and refresh the card to match.

    Verdicts get corrected; imagery does not. This exists so a label fix costs a few MB of upload
    instead of replacing 11.41 GB of image blobs.
    """
    out = Path(args.out)
    if not has_reviewed_splits(args.benchmark):
        sys.exit("error: no records.jsonl + verdicts.json pairs under {}".format(args.benchmark))
    clear_build_dir(out, "data/{}".format(RECORDS))
    written = build_records(args.benchmark, out)
    if not written:
        sys.exit("error: no records.jsonl + verdicts.json pairs under {}".format(args.benchmark))
    print("{:<20} {:>7} {:>7} {:>7} {:>12}".format("city", "panos", "dets", "missed", "parquet"))
    print("-" * 58)
    tot = [0, 0, 0, 0]
    for city, n_panos, n_det, n_missed, size in written:
        print("{:<20} {:>7,} {:>7,} {:>7,} {:>12,}".format(city, n_panos, n_det, n_missed, size))
        tot = [a + b for a, b in zip(tot, (n_panos, n_det, n_missed, size))]
    print("-" * 58)
    print("{:<20} {:>7,} {:>7,} {:>7,} {:>12,}".format("TOTAL", *tot))

    save_index(out, {RECORDS: [city for city, _, _, _, _ in written]})
    wanted_push, args.push = args.push, False
    card(args)                                   # re-render so the config lands in the YAML
    args.push = wanted_push
    if not args.push:
        print("\nNot pushed. Add --push to upload the records config and the card.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out),
                      allow_patterns=["data/{}/*".format(RECORDS), "README.md"],
                      commit_message="Add the `records` config: ground truth, per-pano metadata and attribution (#21)")
    print("Pushed: https://huggingface.co/datasets/{}".format(args.repo_id))


def push(args):
    from huggingface_hub import HfApi                # imported late: not needed to build or verify
    out = Path(args.out)
    if not (out / "README.md").exists():
        sys.exit("error: build first")
    api = HfApi()
    print("Pushing {} to {}".format(out, args.repo_id))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    # upload_folder adds and overwrites; it never deletes, so a partial --out re-pushes only the
    # configs it holds and leaves the rest of the repo alone. The message must then say what this
    # push actually carried, rather than the first publication's fixed wording.
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out),
                      commit_message=args.message)
    print("Done: https://huggingface.co/datasets/{}".format(args.repo_id))


def adopt_index(siblings):
    """(configs, sizes) from a published repo's file listing. Split out so it is testable."""
    configs, sizes = {}, {}
    for name, size in siblings:
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "data" or not parts[2].endswith(".parquet"):
            continue
        config = parts[1]
        configs.setdefault(config, []).append(parts[2][: -len(".parquet")])
        sizes[config] = sizes.get(config, 0) + (size or 0)
    return dict((c, sorted(set(s))) for c, s in configs.items()), sizes


def adopt(args):
    """Write a manifest for a package that is already on the Hub.

    build_manifest.json postdates the first publication of rampnet-benchmark, so the earliest
    partial rebuild has nothing to union against and `card` refuses -- correctly, but with no way
    forward short of rebuilding 11.41 GB to change one config. This reads the published repo's own
    listing, which is the only authority on what is actually up there, and writes the manifest a
    later `build` merges into. Run it once against an --out you are about to rebuild into.
    """
    from huggingface_hub import HfApi                # imported late: not needed to build or verify
    out = Path(args.out)
    info = HfApi().repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    configs, sizes = adopt_index([(s.rfilename, s.size) for s in info.siblings])
    if not configs:
        sys.exit("error: {} publishes no data/<config>/<split>.parquet".format(args.repo_id))
    write_manifest(out, configs, sizes)
    for config in sorted(configs):
        print("{:<12} {:>2} splits {:>15,} bytes".format(
            config, len(configs[config]), sizes[config]))
    print("{:,} bytes total -> {}".format(sum(sizes.values()), out / MANIFEST_NAME))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["build", "verify", "records", "card", "push", "adopt"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "benchmark",
                        help="repo benchmark/ dir; reads <city>/panos/")
    parser.add_argument("--panos-4096", type=Path, default=None,
                        help="directory of <city>/<pano>.jpg rendered at 4096x2048")
    parser.add_argument("--galleries", type=Path, default=None,
                        help="directory holding <city>_incremental_fp/ PNG crops")
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-benchmark")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Add benchmark panoramas (native + 4096x2048) and A/B galleries",
                        help="push: the Hub commit message; say what this push carried")
    parser.add_argument("--push", action="store_true",
                        help="with `card`: upload only README.md")
    parser.add_argument("--allow-partial", action="store_true",
                        help="render a card even though it declares no imagery config")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="verify: pass panoramas that no imagery_manifest.json pins")
    args = parser.parse_args()
    {"build": build, "verify": verify, "records": records,
     "card": card, "push": push, "adopt": adopt}[args.mode](args)


if __name__ == "__main__":
    main()
