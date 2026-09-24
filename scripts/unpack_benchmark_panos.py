"""Unpack the published benchmark panoramas into ``benchmark/<city>/panos/``.

The imagery behind every benchmark split is git-ignored (``.gitignore``: ``benchmark/*/panos/``)
and published as the Hugging Face dataset ``projectsidewalk/rampnet-benchmark`` in Parquet, one
file per city per config (``export_benchmark.py``). Everything in this repo that needs pixels --
``silent_activation.py``, ``seam_response.py``, ``cascade_gate.py``, the gallery renderers --
reads the loose-file layout, and until this script existed nothing committed turned the one
into the other (#179 recorded the gap; #131 needed it closed to re-run Phase 1 elsewhere).

Each Parquet row carries the source bytes verbatim plus their sha256, and each split commits
``benchmark/<city>/imagery_manifest.json`` with the same hash per pano. Both are checked on the
way out: a byte that does not hash to the Parquet's own column is a corrupt download; one that
hashes differently from the committed manifest means the Hub holds a different image from the
one the split was reviewed on. Either is a hard failure, never a warning.

    # the seven US splits Phase 1 pools, native resolution, into a fresh root
    python scripts/unpack_benchmark_panos.py --out /scratch/benchmark_root \
        --cities richmond,bend,clovis,morgantown,annapolis,paterson,gainesville

    # then point a consumer at that root
    python scripts/analysis/silent_activation.py --panos-root /scratch/benchmark_root

``--out`` is a *checkout-shaped* root: files land at ``<out>/benchmark/<city>/panos/<id>.jpg``
so ``--panos-root <out>`` works unchanged. ``--revision`` pins the dataset commit (default
``main``; the resolved commit is printed and written to ``<out>/unpack_manifest.json`` with every
file's hash, so a run is a record and not just a directory). Network only for the download;
``--parquet`` takes local Parquet files instead, which is what the tests use.
"""
import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "projectsidewalk/rampnet-benchmark"
NATIVE = "native"
MODEL_RES = "4096x2048"
CONFIGS = (NATIVE, MODEL_RES)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(benchmark_dir, city):
    """The committed ``imagery_manifest.json`` for one split, or ``None`` if absent.

    The manifest maps pano id -> {file, bytes, sha256, width, height}. Only ``native`` was
    reviewed at these hashes; the 4096x2048 config is a re-render and has no manifest."""
    path = os.path.join(benchmark_dir, city, "imagery_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["panos"]


def unpack_parquet(path, out_root, city, manifest=None, config=NATIVE):
    """Write every row of one city's Parquet to ``<out_root>/benchmark/<city>/panos/``.

    Returns ``(written, skipped, entries)`` where ``entries`` is ``{pano_id: {file, bytes,
    sha256}}`` for the unpack manifest. An existing file whose bytes already hash to the
    expected value is left alone and counted as skipped, so a re-run after an interrupted
    download does the remaining work only.

    Raises ``ValueError`` on the first hash mismatch, naming the pano and both hashes.
    """
    import pyarrow.parquet as pq

    dest = os.path.join(out_root, "benchmark", city, "panos")
    os.makedirs(dest, exist_ok=True)
    written = skipped = 0
    entries = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4):
        rows = batch.to_pylist()
        for row in rows:
            pano = row["pano_id"]
            if row["city"] != city:
                raise ValueError(f"{path}: row {pano} says city={row['city']!r}, expected {city!r}")
            data = row["image"]["bytes"]
            digest = sha256_bytes(data)
            if digest != row["sha256"]:
                raise ValueError(f"{city}/{pano}: bytes hash to {digest}, Parquet says "
                                 f"{row['sha256']} -- corrupt download or file")
            if manifest is not None:
                expect = manifest.get(pano)
                if expect is None:
                    raise ValueError(f"{city}/{pano}: not in the committed imagery_manifest.json")
                if config == NATIVE and expect["sha256"] != digest:
                    raise ValueError(f"{city}/{pano}: Hub bytes hash to {digest}, the committed "
                                     f"manifest says {expect['sha256']} -- a different image "
                                     f"from the one this split was reviewed on")
            name = os.path.basename(row["image"]["path"] or f"{pano}.jpg")
            target = os.path.join(dest, name)
            if os.path.isfile(target):
                with open(target, "rb") as fh:
                    if sha256_bytes(fh.read()) == digest:
                        skipped += 1
                        entries[pano] = {"file": name, "bytes": len(data), "sha256": digest}
                        continue
            tmp = target + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
            written += 1
            entries[pano] = {"file": name, "bytes": len(data), "sha256": digest}
    return written, skipped, entries


def check_complete(manifest, entries, city):
    """Every manifest pano must have been unpacked; a Parquet short of the manifest is an error."""
    missing = sorted(set(manifest) - set(entries))
    if missing:
        raise ValueError(f"{city}: {len(missing)} manifest panos absent from the Parquet, "
                         f"e.g. {missing[:3]}")


def download(repo_id, revision, config, city, cache_dir):
    """Fetch one city's Parquet from the Hub; returns (local path, resolved commit sha)."""
    from huggingface_hub import HfApi, hf_hub_download
    filename = f"data/{config}/{city}.parquet"
    path = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision,
                           cache_dir=cache_dir)
    info = HfApi().dataset_info(repo_id, revision=revision)
    return path, info.sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--out", required=True,
                   help="Checkout-shaped root: files go to <out>/benchmark/<city>/panos/")
    p.add_argument("--cities", required=True, help="Comma-separated split names")
    p.add_argument("--config", default=NATIVE, choices=CONFIGS,
                   help="Which imagery config (default native, the reviewed bytes)")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default="main",
                   help="Dataset commit / tag / branch to fetch (resolved sha is recorded)")
    p.add_argument("--cache-dir", default=None,
                   help="huggingface_hub cache (defaults to HF_HOME); put it on scratch on a "
                        "cluster whose home is quota-capped")
    p.add_argument("--benchmark", default=os.path.join(REPO, "benchmark"),
                   help="Where the committed imagery_manifest.json files are (for the "
                        "content-hash check); pass a directory without them to skip it")
    p.add_argument("--parquet", nargs="*", default=None,
                   help="Local Parquet files (one per city, in --cities order) instead of "
                        "downloading; for tests and offline copies")
    args = p.parse_args(argv)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if args.parquet is not None and len(args.parquet) != len(cities):
        p.error(f"--parquet gives {len(args.parquet)} files for {len(cities)} cities")

    record = {"repo_id": args.repo_id, "revision": args.revision, "resolved_sha": None,
              "config": args.config, "out": os.path.abspath(args.out), "cities": {}}
    t0 = time.time()
    for i, city in enumerate(cities):
        manifest = load_manifest(args.benchmark, city)
        if args.parquet is not None:
            path, sha = args.parquet[i], None
        else:
            path, sha = download(args.repo_id, args.revision, args.config, city, args.cache_dir)
            record["resolved_sha"] = sha
        written, skipped, entries = unpack_parquet(path, args.out, city, manifest, args.config)
        if manifest is not None:
            check_complete(manifest, entries, city)
        record["cities"][city] = {"parquet": os.path.basename(path), "n": len(entries),
                                  "written": written, "already_present": skipped,
                                  "checked_against_manifest": manifest is not None,
                                  "panos": entries}
        print(f"[{city}] {len(entries)} panos: {written} written, {skipped} already present"
              f"{', hashes match the committed manifest' if manifest is not None else ''}",
              flush=True)
    record["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    out_manifest = os.path.join(args.out, "unpack_manifest.json")
    with open(out_manifest, "w", encoding="utf-8", newline="") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    print(f"resolved {args.repo_id}@{record['resolved_sha'] or args.revision}; "
          f"wrote {out_manifest} in {record['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
