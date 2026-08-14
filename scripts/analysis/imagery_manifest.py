"""Pin the imagery every human review was made against (#46 replication, #21).

The judgments in ``benchmark/<city>/verdicts.json`` and
``benchmark/<city>/incremental_fp_tags.json`` are committed, and the detections they
were made against are committed. **The pixels are not.** The panoramas are 9 GB of
git-ignored files heading for Hugging Face, and nothing currently records *which bytes*
a reviewer actually looked at.

That gap matters in both directions:

* **After publishing**, someone who downloads the archive has no way to confirm they
  received the same imagery the ground truth was built on. A re-fetch from Mapillary or
  GSV can silently return re-processed pixels — different stitching, different
  compression, occasionally a different capture — and a verdict re-paired with
  different imagery is worse than no verdict.
* **Before publishing**, we cannot tell whether the local copies have drifted from what
  was reviewed months ago.

So this writes ``benchmark/<city>/imagery_manifest.json``: a sha256 and pixel size per
panorama, keyed by pano id. It is ~100 KB per city — committed, unlike the 9 GB it
describes — and it is the integrity check for the HF archive when #21 lands.

    python scripts/analysis/imagery_manifest.py --write
    python scripts/analysis/imagery_manifest.py --verify    # do local files still match?

``--verify`` is the check a second rater (or a future us) runs after downloading the
published panos, before trusting any committed verdict.

This is the same guarantee ``miss_gallery.py`` already gives the #46 crops, extended to
the two older review passes that predate it.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miss_decomposition import ALL_SPLITS  # noqa: E402

MANIFEST_NAME = "imagery_manifest.json"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def digest_of(entries):
    """One hash over a city's whole imagery set — a single value to quote or compare."""
    spec = ";".join(f"{k}|{v['sha256']}" for k, v in sorted(entries.items()))
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def scan(panos_dir, with_size=True):
    """``{pano_id: {sha256, bytes, width, height}}`` for every image in a bundle."""
    entries = {}
    if not os.path.isdir(panos_dir):
        return entries
    for name in sorted(os.listdir(panos_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        path = os.path.join(panos_dir, name)
        rec = {"sha256": sha256_file(path), "bytes": os.path.getsize(path),
               "file": name}
        if with_size:
            try:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(path) as im:
                    rec["width"], rec["height"] = im.size
            except Exception:
                pass
        entries[stem] = rec
    return entries


def manifest_path(city, repo=REPO):
    return os.path.join(repo, "benchmark", city, MANIFEST_NAME)


def load(city, repo=REPO):
    path = manifest_path(city, repo)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare(entries, recorded):
    """``(ok, missing, extra, changed)`` between a fresh scan and a recorded manifest.

    ``changed`` is the one that matters: a pano present under the same id with
    different bytes means the imagery moved under a verdict that still claims to
    describe it.
    """
    have, want = set(entries), set(recorded)
    changed = [k for k in have & want
               if entries[k]["sha256"] != recorded[k]["sha256"]]
    return (not changed and not (want - have)), sorted(want - have), sorted(have - want), sorted(changed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cities", default=",".join(ALL_SPLITS))
    p.add_argument("--panos-root", default=REPO,
                   help="Checkout holding benchmark/<city>/panos (git-ignored, so in "
                        "a worktree it lives in the main checkout).")
    p.add_argument("--write", action="store_true", help="Write/refresh the manifests.")
    p.add_argument("--verify", action="store_true",
                   help="Check local imagery against the committed manifests.")
    args = p.parse_args(argv)
    if not (args.write or args.verify):
        p.error("choose --write or --verify")

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    failures = 0
    print(f"{'split':>20} {'panos':>6} {'digest':>18}  status")
    for city in cities:
        panos_dir = os.path.join(args.panos_root, "benchmark", city, "panos")
        entries = scan(panos_dir)
        recorded = load(city)

        if args.write:
            if not entries:
                print(f"{city:>20} {'—':>6} {'—':>18}  no panos on disk, skipped")
                continue
            payload = {"city": city, "n": len(entries),
                       "digest": digest_of(entries), "panos": entries}
            os.makedirs(os.path.dirname(manifest_path(city)), exist_ok=True)
            with open(manifest_path(city), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
            print(f"{city:>20} {len(entries):>6} {payload['digest']:>18}  written")
            continue

        # --verify
        if recorded is None:
            print(f"{city:>20} {'—':>6} {'—':>18}  NO MANIFEST — nothing to verify")
            continue
        if not entries:
            print(f"{city:>20} {'—':>6} {recorded['digest']:>18}  "
                  f"imagery absent locally (expected {recorded['n']} panos)")
            continue
        ok, missing, extra, changed = compare(entries, recorded["panos"])
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"{city:>20} {len(entries):>6} {digest_of(entries):>18}  {status}")
        for k in changed[:10]:
            print(f"    CHANGED BYTES: {k}")
        for k in missing[:10]:
            print(f"    missing locally: {k}")
        for k in extra[:10]:
            print(f"    not in manifest: {k}")

    if args.verify:
        print()
        print("Every reviewed panorama matches the bytes recorded at review time."
              if not failures else
              f"{failures} split(s) MISMATCH — verdicts for those describe different pixels.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
