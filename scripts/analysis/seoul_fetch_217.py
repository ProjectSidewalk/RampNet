"""Fetch the Seoul Sidewalk Accessibility Image Dataset (#217) and verify it.

Source: Lieu et al., arXiv 2609.17882; Zenodo record 22699523
(https://doi.org/10.5281/zenodo.22699523), CC0. 514 iPhone photos taken 1.0 m above
the ground from the centre of the walking path, with laser-measured effective width
in ``summary_attributes.csv``.

The images are NOT committed. What is committed (``data/seoul_sidewalk_217/``):

* ``summary_attributes.csv`` -- the ground-truth table, verbatim (CC0).
* ``manifest.json`` -- every extracted image's relative path, byte size and sha256,
  plus the md5 of each Zenodo archive as the Zenodo API reported it.

Usage::

    # download the three imagery zips + CSV into DEST, check md5, extract, then
    # check every image against the committed manifest
    python scripts/analysis/seoul_fetch_217.py fetch --dest /homes/gws/jonf/seoul_sidewalk

    # an existing copy (someone else's, or an old one): check it against the manifest,
    # and the committed summary_attributes.csv against Zenodo's md5
    python scripts/analysis/seoul_fetch_217.py verify --dest /homes/gws/jonf/seoul_sidewalk

    # (first fetch only) write the manifest that later verifies check against
    python scripts/analysis/seoul_fetch_217.py fetch --dest DIR \
        --write-manifest data/seoul_sidewalk_217/manifest.json

An archive whose md5 already matches Zenodo's is not downloaded again.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile

RECORD = 22699523
API = f"https://zenodo.org/api/records/{RECORD}"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".heic")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MANIFEST = os.path.join(REPO, "data", "seoul_sidewalk_217", "manifest.json")
COMMITTED_CSV = os.path.join(REPO, "data", "seoul_sidewalk_217", "summary_attributes.csv")


def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    out = []
    for f in rec["files"]:
        algo, _, digest = f["checksum"].partition(":")
        out.append({"key": f["key"], "size": f["size"], "algo": algo, "digest": digest,
                    "url": f["links"]["self"]})
    return out


def _download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    os.replace(tmp, dest)


def _images(dest):
    root = os.path.join(dest, "images")
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(IMAGE_EXT) and not n.startswith("._"):
                p = os.path.join(dirpath, n)
                yield os.path.relpath(p, root).replace(os.sep, "/"), p


def fetch(args):
    dest = args.dest
    zips = os.path.join(dest, "zips")
    os.makedirs(zips, exist_ok=True)
    archives = {}
    for f in _record_files():
        target = (os.path.join(zips, f["key"]) if f["key"].endswith(".zip")
                  else os.path.join(dest, f["key"]))
        if not (os.path.exists(target) and _hash(target, f["algo"]) == f["digest"]):
            print(f"downloading {f['key']} ({f['size'] / 1e9:.2f} GB)", flush=True)
            _download(f["url"], target)
        got = _hash(target, f["algo"])
        if got != f["digest"]:
            sys.exit(f"{f['key']}: {f['algo']} {got} != Zenodo's {f['digest']}")
        archives[f["key"]] = {"size": f["size"], f["algo"]: f["digest"]}
        print(f"ok {f['key']}", flush=True)
        if f["key"].startswith("imagery_") and f["key"].endswith(".zip"):
            with zipfile.ZipFile(target) as z:
                z.extractall(os.path.join(dest, "images"))
    if args.write_manifest:
        manifest = {"source": f"https://doi.org/10.5281/zenodo.{RECORD}",
                    "archives": archives,
                    "images": {rel: {"size": os.path.getsize(p), "sha256": _hash(p, "sha256")}
                               for rel, p in _images(dest)}}
        os.makedirs(os.path.dirname(os.path.abspath(args.write_manifest)), exist_ok=True)
        with open(args.write_manifest, "w", encoding="utf-8", newline="") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"{len(manifest['images'])} images; wrote {args.write_manifest}")
    else:
        verify(args)


def verify_committed_csv(manifest=DEFAULT_MANIFEST, csv_path=COMMITTED_CSV):
    """md5 of the committed GT table against the Zenodo md5 recorded in the manifest.
    Returns (ok, got, want)."""
    with open(manifest, encoding="utf-8") as f:
        want = json.load(f)["archives"]["summary_attributes.csv"]["md5"]
    got = _hash(csv_path, "md5")
    return got == want, got, want


def verify(args):
    ok, got, want_md5 = verify_committed_csv(args.manifest)
    print(f"committed summary_attributes.csv md5 {got} "
          f"{'==' if ok else '!='} Zenodo's {want_md5}")
    if not ok:
        sys.exit("FAILED: the committed GT table is not the Zenodo file")
    with open(args.manifest, encoding="utf-8") as f:
        want = json.load(f)["images"]
    have = dict(_images(args.dest))
    missing = sorted(set(want) - set(have))
    bad = [rel for rel in sorted(set(want) & set(have))
           if _hash(have[rel], "sha256") != want[rel]["sha256"]]
    extra = sorted(set(have) - set(want))
    print(f"{len(want)} in manifest, {len(have)} on disk; missing {len(missing)}, "
          f"sha256 mismatch {len(bad)}, extra {len(extra)}")
    if missing or bad:
        sys.exit(f"FAILED: missing {missing[:5]} bad {bad[:5]}")
    print("verify ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "verify"])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--write-manifest", default=None)
    args = ap.parse_args()
    (fetch if args.cmd == "fetch" else verify)(args)


if __name__ == "__main__":
    main()
