"""Shrink the Stage 1 street centrelines to the part the pipeline actually reads.

`street_data/` as downloaded is **801.6 MB** -- "New York - Streets.geojson" alone is 669 MB, past
GitHub's 100 MB hard limit, so it has never been publishable from git. But its only consumer,
`generate_negative_panos.py`, reads just two things from each feature:

  * the LineString geometry, and
  * **one** name property, used solely as an emptiness test -- `FULLNAME` (Bend),
    `FULL_NAME` (Portland), `Street` (New York). See `load_city_streets`.

Every other column -- route numbers, ZIP, MSAG, ESN, one-way direction, road class, county --
is never read. Dropping them takes the three files from 801.6 MB to 92.6 MB, or **18.7 MB
gzipped**, which git carries comfortably.

The name field is kept rather than dropped: Portland has 4,192 features whose `FULL_NAME` is the
empty string, and `load_city_streets` skips exactly those. A pure-geometry file would silently
re-admit them and change the sampled street network.

Behaviour preservation is *proved*, not asserted. `verify` computes a **consumer fingerprint** --
a sha256 over the ordered sequence of (name value, geometry) for every feature that survives the
name filter, which is precisely the input `load_city_streets` builds its length-weighted sampling
index from. Identical fingerprints mean the derivative and the original produce the same network.
The boundary clip that follows is a pure function of that geometry, so it cannot diverge either.

Usage
-----
    python scripts/build_street_derivative.py build  --src <street_data/> --out <dir>
    python scripts/build_street_derivative.py verify --src <street_data/> --out <dir>

`verify` exits non-zero if any city's fingerprint differs.

Caveat worth keeping attached to these files: `generate_negative_panos.py` calls `random.uniform`
and `random.choice` with **no seed**, so the negative panoramas cannot be regenerated identically
from any street file, derivative or original. The set actually used is recorded in
`negativepanosSHORTENED.jsonl`. This derivative is for *re-running* the sampling on new cities,
not for reproducing the paper's negatives.
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_export_common import sha256_file  # noqa: E402

CITIES = ["Bend", "New York", "Portland"]

# The name properties load_city_streets tests, in the order it tests them.
NAME_FIELDS = ["FULL_NAME", "FULLNAME", "Street"]

SRC_NAME = "{city} - Streets.geojson"
OUT_NAME = "{city} - Streets.min.geojson.gz"


def open_maybe_gzip(path):
    """Open .gz transparently, mirroring generate_negative_panos.open_street_file.

    `encoding="utf-8"` is not cosmetic here. GeoJSON is UTF-8 by spec (RFC 7946), but bare
    `open()` and `gzip.open(..., "rt")` decode with the *platform's* locale codec -- cp1252 on a
    default Windows box, UTF-8 on the Linux cluster. The fingerprint this whole equivalence proof
    rests on hashes decoded street *names*, so an unpinned codec makes it machine-dependent: the
    same correct derivative would read MATCH on one box and `*** DIFFERS ***` on another, or
    `build` would bake mojibake names into a committed artifact. The three current cities are
    ASCII-only, which is why the cross-platform check passed; a fourth city need not be.
    """
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def kept_features(features):
    """Yield (name_props, geometry) for features load_city_streets would keep.

    Mirrors its filter exactly: a name field that is present *and* empty means skip.
    """
    for feature in features:
        props = feature.get("properties", {}) or {}
        present = [f for f in NAME_FIELDS if f in props]
        if any(props[f] == "" for f in present):
            continue
        yield dict((f, props[f]) for f in present), feature["geometry"]


def fingerprint(path):
    """sha256 over exactly what load_city_streets consumes, in order."""
    with open_maybe_gzip(path) as fh:
        data = json.load(fh)
    digest = hashlib.sha256()
    n = 0
    for name_props, geometry in kept_features(data["features"]):
        digest.update(json.dumps(name_props, sort_keys=True, separators=(",", ":")).encode())
        digest.update(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode())
        n += 1
    return digest.hexdigest(), n


def build(src_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    total_src = total_out = 0
    print("{:<26} {:>12} {:>12} {:>7}  {}".format("file", "original", "derivative", "ratio", "kept field"))
    print("-" * 78)
    for city in CITIES:
        src = src_dir / SRC_NAME.format(city=city)
        out = out_dir / OUT_NAME.format(city=city)
        with open_maybe_gzip(src) as fh:
            data = json.load(fh)

        features = []
        fields_seen = set()
        for feature in data["features"]:
            props = feature.get("properties", {}) or {}
            present = [f for f in NAME_FIELDS if f in props]
            fields_seen.update(present)
            features.append({
                "type": "Feature",
                "properties": dict((f, props[f]) for f in present),
                "geometry": feature["geometry"],
            })

        blob = json.dumps({"type": "FeatureCollection", "features": features},
                          separators=(",", ":")).encode()
        # mtime=0 so the same input always produces a byte-identical .gz
        with gzip.GzipFile(str(out), "wb", compresslevel=9, mtime=0) as fh:
            fh.write(blob)

        src_size, out_size = src.stat().st_size, out.stat().st_size
        total_src += src_size
        total_out += out_size
        print("{:<26} {:>12,} {:>12,} {:>6.1f}x  {}".format(
            src.name, src_size, out_size, src_size / float(out_size),
            ",".join(sorted(fields_seen)) or "(none)"))

    print("-" * 78)
    print("{:<26} {:>12,} {:>12,} {:>6.1f}x".format(
        "TOTAL", total_src, total_out, total_src / float(total_out)))
    print()
    print("Derivative sha256 (the committed artifacts):")
    for city in CITIES:
        out = out_dir / OUT_NAME.format(city=city)
        print("  {:<34} {}".format(out.name, sha256_file(out)))


def verify(src_dir, out_dir):
    print("Consumer fingerprint -- sha256 over (name, geometry) for every feature kept")
    print("-" * 78)
    failed = False
    for city in CITIES:
        src = src_dir / SRC_NAME.format(city=city)
        out = out_dir / OUT_NAME.format(city=city)
        if not out.exists():
            print("  {:<12} MISSING derivative {}".format(city, out.name))
            failed = True
            continue
        src_fp, src_n = fingerprint(src)
        out_fp, out_n = fingerprint(out)
        ok = src_fp == out_fp and src_n == out_n
        failed |= not ok
        print("  {:<12} {}  {:>7,} features kept".format(
            city, "MATCH  " + src_fp[:16] if ok else "*** DIFFERS ***", src_n))
        if not ok:
            print("      original   {}  ({:,} features)".format(src_fp, src_n))
            print("      derivative {}  ({:,} features)".format(out_fp, out_n))
    print("-" * 78)
    if failed:
        sys.exit("FAIL: derivative would change the sampled street network")
    print("PASS: derivative and original yield an identical sampling network")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--src", required=True, type=Path, help="original street_data/ directory")
    parser.add_argument("--out", required=True, type=Path, help="derivative output directory")
    args = parser.parse_args()
    (build if args.mode == "build" else verify)(args.src, args.out)


if __name__ == "__main__":
    main()
