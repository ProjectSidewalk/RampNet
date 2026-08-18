"""Rebuild the row-level provenance that `combine_location_data.py` throws away.

Stage 1 starts from three open-government curb ramp inventories (Bend, Portland, NYC).
`combine_location_data.py` merges them into `all_locations.csv` with only three columns --
`latitude`, `longitude`, `date` -- and then shuffles. That discards:

  * NYC's `RampID` / `CornerID` and the geojson `OBJECTID` / `FacilityID` / `NonAssetID`,
    i.e. every government primary key;
  * the 24 NYC attribute columns (slopes, widths, conditions);
  * **which city each row came from** -- there is no source column at all;
  * and, via the shuffle, any chance of recovering the above from row order.

So `all_locations.csv` on its own cannot answer "which government records are in the paper's
training set?". This script answers it.

Why a coordinate join and not a replay of the shuffle
-----------------------------------------------------
The obvious approach -- re-run the same concatenation and shuffle with the same seed, carrying a
provenance payload -- **does not work for the paper's data**, and the reason is worth recording:
the paper-era `combine_location_data.py` called `random.shuffle(all_data)` with **no seed**.
`random.seed(42)` was added later. The published row *order* is therefore unreproducible, which is
the Stage 1 counterpart of the split-seeding caveat in `docs/data_provenance.md` section 4.

The row *contents* survive intact, so provenance is recovered by joining each `all_locations.csv`
row back to its source record on the (latitude, longitude) pair. Government inventories store
high-precision coordinates, so these are effectively unique keys; the script verifies that rather
than assuming it, and reports any collision or unmatched row instead of silently guessing.

Date handling is imported from `combine_location_data.py` rather than copied, so it cannot drift.
Note that the paper-era `convert_date` mapped unknown dates to `"2000-01-01"` while the current one
returns `""` -- pass `--repo-root` pointing at the checkout whose logic you want, and read the
reported date-agreement figure to see how far the two diverge.

Optionally joins forward to `dataset.jsonl` to mark which ramps were actually consumed by a
generated panorama.

Usage
-----
    python scripts/analysis/gov_provenance.py \
        --location-data  stage_one/dataset_generation/location_data \
        --all-locations  stage_one/dataset_generation/all_locations.csv \
        --dataset-jsonl  stage_one/dataset_generation/dataset.jsonl \
        --out            analysis_out/gov_provenance.csv
"""

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_export_common import sha256_file as sha256  # noqa: E402

# combine_location_data.py hardcodes these three files, in this order, with these date fields.
SOURCES = [
    # (filename, kind, date_field, id_fields)
    ("bend.geojson", "geojson", "InstallDate", ("OBJECTID", "FacilityID", "GISOBJID")),
    ("portland.geojson", "geojson", "InstallDate", ("OBJECTID", "NonAssetID")),
    ("nyc.csv", "csv", "GeoCyclora", ("RampID", "CornerID")),
]

ID_COLUMNS = ["OBJECTID", "FacilityID", "GISOBJID", "NonAssetID", "RampID", "CornerID"]

# combine_location_data.py -- only rows whose the_geom parses as a POINT are kept.
POINT_RE = re.compile(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)")


def load_convert_date(repo_root):
    """Import `convert_date` from the real combine script so the date logic cannot drift."""
    path = repo_root / "stage_one" / "dataset_generation" / "combine_location_data.py"
    if not path.exists():
        sys.exit("error: cannot find {} -- pass --repo-root".format(path))
    spec = importlib.util.spec_from_file_location("_combine_location_data", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is __main__-guarded, so nothing runs
    return module.convert_date


def parse_source(directory, filename, kind, date_field, id_fields, convert_date):
    """Parse one inventory, mirroring combine_location_data.py but carrying provenance.

    Returns (records, n_source_rows). The difference is the count combine would have dropped.
    """
    path = directory / filename
    records = []

    if kind == "geojson":
        with open(path) as fh:
            data = json.load(fh)
        features = data.get("features", [])
        for source_row, feature in enumerate(features):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            records.append({
                "latitude": lat,
                "longitude": lon,
                "date": convert_date(props.get(date_field)),
                "source_file": filename,
                "source_row": source_row,
                "source_ids": dict((k, props.get(k)) for k in id_fields),
            })
        return records, len(features)

    n_source_rows = 0
    with open(path, newline="") as fh:
        for source_row, row in enumerate(csv.DictReader(fh)):
            n_source_rows += 1
            match = POINT_RE.search(row.get("the_geom", ""))
            if not match:
                continue
            records.append({
                "latitude": float(match.group(2)),
                "longitude": float(match.group(1)),
                "date": convert_date(row.get(date_field, "")),
                "source_file": filename,
                "source_row": source_row,
                "source_ids": dict((k, row.get(k)) for k in id_fields),
            })
    return records, n_source_rows


def coord_key(lat, lng):
    """Join key. all_locations.csv stores repr(float); float(repr(x)) == x, so this is exact."""
    return "{!r},{!r}".format(lat, lng)


def load_consumed_coords(dataset_jsonl):
    """Scan dataset.jsonl once.

    Returns (consumed, stats) where `consumed` maps coord_key -> one pano_id that used it (a ramp
    is typically visible from several panoramas, so the choice is arbitrary and only the *fact* of
    consumption is meaningful), and `stats` records the totals that the map itself cannot show.
    """
    consumed = {}
    panos = set()
    ramp_instances = 0
    with open(str(dataset_jsonl)) as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            pano_id = record.get("pano_id")
            panos.add(pano_id)
            for lat, lng in record.get("curb_ramps_coords", []):
                ramp_instances += 1
                consumed.setdefault(coord_key(lat, lng), pano_id)
    return consumed, {"panos": len(panos), "ramp_instances": ramp_instances}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--location-data", required=True, type=Path,
                        help="directory holding bend.geojson, portland.geojson, nyc.csv")
    parser.add_argument("--all-locations", required=True, type=Path,
                        help="the published all_locations.csv to attach provenance to")
    parser.add_argument("--dataset-jsonl", type=Path, default=None,
                        help="optional dataset.jsonl, to mark which ramps a panorama consumed")
    parser.add_argument("--out", required=True, type=Path, help="output provenance CSV")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repo root (to import combine_location_data.convert_date)")
    args = parser.parse_args()

    convert_date = load_convert_date(args.repo_root)

    print("Source inventories")
    print("-" * 78)
    index = defaultdict(list)
    per_source = []
    for filename, kind, date_field, id_fields in SOURCES:
        records, n_source_rows = parse_source(
            args.location_data, filename, kind, date_field, id_fields, convert_date
        )
        dropped = n_source_rows - len(records)
        per_source.append((filename, n_source_rows, len(records), dropped))
        print("  {:<20} {:>7,} records  ->  {:>7,} kept  ({:,} dropped)  date field {!r}".format(
            filename, n_source_rows, len(records), dropped, date_field))
        print("  {:<20} sha256 {}".format("", sha256(args.location_data / filename)))
        for record in records:
            index[coord_key(record["latitude"], record["longitude"])].append(record)

    n_source_total = sum(len(v) for v in index.values())
    collisions = sum(1 for v in index.values() if len(v) > 1)
    print("-" * 78)
    print("  combined: {:,} records, {:,} distinct coordinates".format(n_source_total, len(index)))
    print("  coordinates shared by more than one government record: {:,}".format(collisions))

    consumed, dataset_stats = (
        load_consumed_coords(args.dataset_jsonl) if args.dataset_jsonl else (None, None)
    )

    columns = ["all_locations_row", "source_file", "source_row",
               "latitude", "longitude", "date"] + ID_COLUMNS
    if consumed is not None:
        columns += ["in_dataset", "pano_id"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matched = unmatched = ambiguous = date_agree = n_consumed = 0
    by_source = defaultdict(lambda: [0, 0])

    with open(str(args.all_locations), newline="") as src, \
            open(str(args.out), "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=columns)
        writer.writeheader()
        for i, original in enumerate(csv.DictReader(src)):
            key = "{},{}".format(original["latitude"], original["longitude"])
            candidates = index.get(key, [])
            if not candidates:
                unmatched += 1
                continue
            if len(candidates) > 1:
                ambiguous += 1
            record = candidates[0]
            matched += 1
            if record["date"] == original["date"]:
                date_agree += 1

            row = {
                "all_locations_row": i,
                "source_file": record["source_file"],
                "source_row": record["source_row"],
                "latitude": original["latitude"],
                "longitude": original["longitude"],
                "date": original["date"],
            }
            row.update(dict((k, record["source_ids"].get(k, "")) for k in ID_COLUMNS))

            counts = by_source[record["source_file"]]
            counts[1] += 1
            if consumed is not None:
                pano_id = consumed.get(key)
                row["in_dataset"] = int(pano_id is not None)
                row["pano_id"] = pano_id or ""
                if pano_id is not None:
                    n_consumed += 1
                    counts[0] += 1
            writer.writerow(row)

    total = matched + unmatched
    print()
    print("Join against {}".format(args.all_locations.name))
    print("  sha256 {}".format(sha256(args.all_locations)))
    print("  matched to a government record: {:,} / {:,}".format(matched, total))
    print("  unmatched:                      {:,}".format(unmatched))
    print("  ambiguous (shared coordinate):  {:,}".format(ambiguous))
    if matched:
        print("  date reproduced by this checkout's convert_date: {:,} / {:,}  ({:.2f}%)".format(
            date_agree, matched, 100.0 * date_agree / matched))
    if unmatched:
        sys.exit("error: {:,} rows could not be traced to a source record".format(unmatched))

    print()
    print("Wrote {}  ({:,} rows)".format(args.out, matched))
    if consumed is not None:
        print("  government records consumed by a panorama: {:,} / {:,}  ({:.2f}%)".format(
            n_consumed, matched, 100.0 * n_consumed / matched))
        print("  panoramas in dataset.jsonl:               {:,}".format(dataset_stats["panos"]))
        print("  ramp instances across those panoramas:    {:,}  ({:.2f} per consumed record)".format(
            dataset_stats["ramp_instances"],
            dataset_stats["ramp_instances"] / float(n_consumed) if n_consumed else 0.0))
        print()
        print("  Per-source consumption")
        for filename, _, _, _ in per_source:
            hit, seen = by_source[filename]
            share = 100.0 * hit / seen if seen else 0.0
            print("    {:<20} {:>7,} / {:>7,}  ({:.2f}%)".format(filename, hit, seen, share))


if __name__ == "__main__":
    main()
