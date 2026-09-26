"""Which reviewed benchmark panoramas are also in RampNet's training data? (#127)

The benchmark measures the published model on cities and imagery it was not trained on. A
benchmark panorama whose id is also in `projectsidewalk/rampnet-dataset`'s train or validation
split breaks that for the panorama in question. The first check (2026-07-22, done inside the
since-removed `scripts/build_benchmark_dataset.py`) found 4 such panoramas, all in `bend`, and
none anywhere else it looked. This script re-runs that check for **every** split in
`export_benchmark.BENCHMARK_SPLITS` -- the Mapillary splits too, so "different id space" is a
measured zero rather than an assumption -- and writes the answer to a committed file:

    python scripts/analysis/train_overlap_check.py --benchmark benchmark \\
        --out benchmark/train_overlap.json

`benchmark/train_overlap.json` is what `scripts/export_benchmark.py` reads to fill the
`train_overlap` column of the published `records` config, and it refuses to export a split that
has no entry there, so a new city cannot be published unchecked. `scripts/score_validation.py
--exclude-train-overlap` reads it too.

Method: exact panorama-id membership. Only the `pano_id` column of each parquet shard is read,
over HTTP range requests (the same approach as `scripts/fetch_manual_gold.py --audit`), so the
~460 GB dataset is checked in about 10 minutes (11 m 05 s on 2026-09-26) without downloading
imagery. Network read-only; it never writes to the Hub.

Output is written with sorted keys, 2-space indent, LF line endings and a trailing newline, so a
re-run that finds the same answer produces the same bytes (apart from `checked_at`).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_benchmark import BENCHMARK_SPLITS  # noqa: E402
from fetch_manual_gold import HF_DATASET, split_of  # noqa: E402

METHOD = ("exact panorama-id membership; pano_id column of every parquet shard read over "
          "HTTP range")
SCRIPT = "scripts/analysis/train_overlap_check.py"


def reviewed_ids(benchmark):
    """{split: set of reviewed pano ids} -- the keys of each split's verdicts.json["panos"]."""
    out = {}
    for split in BENCHMARK_SPLITS:
        path = Path(benchmark) / split / "verdicts.json"
        if not path.is_file():
            sys.exit("error: {} has no verdicts.json; every split in BENCHMARK_SPLITS must be "
                     "checked".format(path.parent))
        out[split] = set(json.loads(path.read_text(encoding="utf-8"))["panos"])
    return out


def dataset_ids(splits):
    """{dataset split: set of pano ids} for the requested rampnet-dataset splits."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob("datasets/{}/**/*.parquet".format(HF_DATASET))
    by_split = dict((s, []) for s in splits)
    for f in files:
        split = split_of(f)
        if split in by_split:
            by_split[split].append(f)
    for split, shards in by_split.items():
        if not shards:
            sys.exit("error: no parquet shards found for split {!r} of {} -- has the layout "
                     "changed?".format(split, HF_DATASET))

    def read_ids(path):
        return set(pq.read_table(path, columns=["pano_id"], filesystem=fs)["pano_id"]
                   .to_pylist())

    ids = {}
    for split, shards in by_split.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            ids[split] = set().union(*ex.map(read_ids, shards))
        print("{}: {:,} shards, {:,} panos".format(split, len(shards), len(ids[split])))
    return ids


def overlap_report(reviewed, training, checked_at, dataset_splits):
    """The JSON-ready result. Pure, so the tests can build one without the network."""
    union = set().union(*training.values()) if training else set()
    return {
        "checked_at": checked_at,
        "dataset": HF_DATASET,
        "dataset_split_sizes": dict((s, len(training[s])) for s in dataset_splits),
        "dataset_splits_checked": list(dataset_splits),
        "method": METHOD,
        "overlap": dict((split, sorted(ids & union)) for split, ids in reviewed.items()),
        "reviewed_counts": dict((split, len(ids)) for split, ids in reviewed.items()),
        "script": SCRIPT,
    }


def dump(report):
    """The exact bytes written to disk: sorted keys, indent 2, trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=REPO_ROOT / "benchmark")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "benchmark" / "train_overlap.json")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"],
                    help="rampnet-dataset splits to check against (default: train validation)")
    args = ap.parse_args()

    reviewed = reviewed_ids(args.benchmark)
    training = dataset_ids(args.splits)
    report = overlap_report(reviewed, training, datetime.date.today().isoformat(), args.splits)

    print()
    print("{:<20} {:>8} {:>8}".format("benchmark split", "reviewed", "overlap"))
    print("-" * 38)
    for split in BENCHMARK_SPLITS:
        print("{:<20} {:>8} {:>8}".format(split, report["reviewed_counts"][split],
                                          len(report["overlap"][split])))
    for split, ids in report["overlap"].items():
        for pano_id in ids:
            print("  {}: {}".format(split, pano_id))

    try:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write(dump(report))
    except OSError as exc:
        sys.exit("error: could not write {}: {}".format(args.out, exc))
    print("\nwrote {}".format(args.out))


if __name__ == "__main__":
    main()
