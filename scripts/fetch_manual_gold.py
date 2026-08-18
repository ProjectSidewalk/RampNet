"""Build the ``benchmark/manual_gold`` bundle's imagery + records (issue #58).

The 1,000-pano manual gold set's labels live in-repo (``manual_labels/*.txt``);
its images live in the **test split** of ``projectsidewalk/rampnet-dataset`` on
Hugging Face (21,441 panos — the gold panos are a subset). This script fetches
those 1,000 images into ``benchmark/manual_gold/panos/`` (git-ignored, like every
bundle's imagery) and writes the bundle's ``records.jsonl`` (pano metadata only —
RampNet's detections are added by ``scripts/export_gold_records.py``).

    # id-only audit: no image download. Verifies every gold id is in the HF test
    # split and none leaked into train/validation, and re-checks city-bundle overlap.
    python scripts/fetch_manual_gold.py --audit

    # imagery for THIS machine, committed bundle untouched — the usual re-fetch.
    # records.jsonl (with its exported detections) and bundle_meta.json are left
    # exactly as committed; refuses a --source that contradicts bundle_meta.json.
    # Panos already on disk that match records.jsonl are skipped, so a preempted
    # run resumes rather than starting the whole fetch over.
    python scripts/fetch_manual_gold.py --images-only

    # full bundle build (first time, or rebuild with --force — DISCARDS exported
    # detections in records.jsonl):
    python scripts/fetch_manual_gold.py

    # or copy from an existing download_dataset.py output instead of the Hub. This
    # is a FULL BUILD option only: the committed bundle records source="hf", and
    # --images-only refuses a source that contradicts it (below), so there is no
    # --source local re-fetch for the committed bundle — by design, since the two
    # sources are not byte-identical.
    python scripts/fetch_manual_gold.py --source local --local-dataset ./dataset/test

Cost note, measured 2026-08-14 on makelab2: the ``--source hf`` path goes through
``load_dataset``, which downloaded and arrow-materialized **all three splits** (~2.5 h
end to end) despite ``split="test"`` — not the ~44 GB test-split-only fetch this
docstring used to promise. A shard-scoped fetch via ``HfFileSystem`` + pyarrow (the
``--audit`` path already works that way) would cut that substantially; deliberately
not done yet, to keep this change from touching the byte-fidelity-sensitive read path.

Byte fidelity matters: a past gold-set re-eval moved P +2.2 / R -1.8 on JPEG
re-encoding alone (see docs/model_comparison.md). ``--source hf`` therefore
writes the parquet's **raw image bytes** untouched (``decode=False``), and
``--source local`` copies files byte-for-byte. The two sources are NOT
byte-identical to each other — ``download_dataset.py`` re-encodes at quality 95 —
so the bundle records which one built it (``bundle_meta.json``), and the
exporter's reproduction gate against the published gold-set numbers is the
arbiter of whether the difference matters.

After any fetch the imagery is checked against
``benchmark/manual_gold/imagery_manifest.json`` — a sha256 per pano, the same content
hash the nine city splits carry, written and verified by
``scripts/analysis/imagery_manifest.py``. **That file does not exist yet**: nobody has run
the writer on a machine holding all 1,000 panos, so today the check prints the command
that would create it instead of verifying anything, and manual_gold is the one split whose
imagery has no committed hash. Until it exists, ``bundle_meta.json``'s recorded source and
the per-pano pixel size in ``records.jsonl`` are the only evidence that the imagery under
the committed records is the imagery those records describe.

Stage-1's auto-generated labels (``curb_ramp_points_normalized`` etc.) are
deliberately NOT copied into the records: the bundle's ground truth is the
manual labels, and carrying the weaker auto labels alongside invites mixups.
"""
import argparse
import datetime
import io
import json
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_DATASET = "projectsidewalk/rampnet-dataset"
LABELS_DIR = os.path.join(REPO_ROOT, "manual_labels")
BUNDLE_DIR = os.path.join(REPO_ROOT, "benchmark", "manual_gold")
CITY_BUNDLES = ("bend", "richmond", "clovis")
# HF row keys copied into each record's "pano" dict (verbatim, no reinterpretation).
PANO_META_KEYS = ("record_creation_time", "pano_coord", "pano_azimuth")


def gold_ids():
    ids = sorted(n[:-4] for n in os.listdir(LABELS_DIR) if n.endswith(".txt"))
    if not ids:
        raise SystemExit(f"no .txt label files in {LABELS_DIR}")
    return ids


def split_of(parquet_path):
    """Infer the split from a Hub parquet path. This dataset nests shards under a
    split directory (``.../test/data-00000-of-00128.parquet``); the basename
    pattern (``train-00000-of-00300.parquet``) is the other common Hub layout."""
    parts = parquet_path.replace("\\", "/").split("/")
    for token, split in (("train", "train"), ("validation", "validation"),
                         ("val", "validation"), ("test", "test")):
        if token in parts[:-1] or re.search(rf"(^|[-_.]){token}([-_.]|$)", parts[-1]):
            return split
    return None


def audit(ids):
    """Id-only membership + overlap checks; no image download.

    Reads just the ``pano_id`` column of every parquet shard via HTTP range
    requests (a few KB per shard instead of ~500 MB), so the whole 460 GB
    dataset audits in minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob(f"datasets/{HF_DATASET}/**/*.parquet")
    by_split = {"train": [], "validation": [], "test": []}
    unknown = []
    for f in files:
        split = split_of(f)
        (by_split[split] if split in by_split else unknown).append(f)
    if unknown:
        print(f"warning: {len(unknown)} parquet file(s) with unrecognized split name, "
              f"e.g. {unknown[0]}")
    for split, fl in by_split.items():
        if not fl:
            raise SystemExit(f"no parquet files found for split {split!r} — "
                             "has the dataset layout changed?")

    def read_ids(path):
        return set(pq.read_table(path, columns=["pano_id"], filesystem=fs)["pano_id"]
                   .to_pylist())

    gold = set(ids)
    ok = True
    for split, fl in by_split.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            split_ids = set().union(*ex.map(read_ids, fl))
        inter = gold & split_ids
        print(f"{split}: {len(split_ids)} panos; gold overlap {len(inter)}")
        if split == "test":
            missing = sorted(gold - split_ids)
            if missing:
                ok = False
                print(f"  MISSING from test split ({len(missing)}): {missing[:10]}"
                      + (" ..." if len(missing) > 10 else ""))
        elif inter:
            ok = False
            print(f"  LEAKED gold ids in {split} ({len(inter)}): {sorted(inter)[:10]}"
                  + (" ..." if len(inter) > 10 else ""))

    for city in CITY_BUNDLES:
        vpath = os.path.join(REPO_ROOT, "benchmark", city, "verdicts.json")
        with open(vpath, encoding="utf-8") as f:
            city_ids = set(json.load(f)["panos"])
        inter = sorted(gold & city_ids)
        print(f"benchmark/{city}: {len(city_ids)} panos; gold overlap {len(inter)}"
              + (f" -> {inter}" if inter else ""))

    print("\nAudit " + ("PASSED: gold set is fully in the test split, absent from "
                        "train/validation." if ok else "FAILED (see above)."))
    return 0 if ok else 1


def _dims(src):
    from PIL import Image
    with Image.open(src) as img:  # header read only, no full decode
        return img.width, img.height


def image_dims(jpeg_bytes):
    """(width, height) from JPEG bytes already in memory (the Hub path has them)."""
    return _dims(io.BytesIO(jpeg_bytes))


def image_dims_path(path):
    """(width, height) straight off a file's header — never reads the pixels.

    The gold panos are 4096x2048 and up, so slurping a whole one to read twenty
    bytes of header is GBs of pointless I/O across the 1,000-pano set.
    """
    return _dims(path)


def make_record(pano_id, width, height, meta):
    pano = {"panorama_id": pano_id, "width": width, "height": height}
    pano.update({k: meta[k] for k in PANO_META_KEYS if k in meta})
    return {"pano": pano}


def fetch_hf(ids, panos_dir):
    """Yield (pano_id, record) writing raw parquet image bytes to panos_dir."""
    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(HF_DATASET, split="test").cast_column("image", HFImage(decode=False))
    wanted = set(ids)
    for row in ds:
        pid = row["pano_id"]
        if pid not in wanted:
            continue
        data = row["image"]["bytes"]
        if not data:
            raise SystemExit(f"{pid}: test-split row has no embedded image bytes")
        with open(os.path.join(panos_dir, f"{pid}.jpg"), "wb") as f:
            f.write(data)
        w, h = image_dims(data)
        yield pid, make_record(pid, w, h, row)


def fetch_local(ids, panos_dir, local_dataset, need_meta=True):
    """Yield (pano_id, record) copying byte-for-byte from a download_dataset.py
    output directory (``dataset/test``: <pid>.jpg + <pid>.json).

    ``need_meta`` is False on the --images-only path, which throws the records away:
    the sidecars are then not read at all, so imagery rsync'd to a machine without
    them still fetches. When they ARE needed, a missing or unparseable sidecar exits
    with guidance rather than a traceback partway through the copy.
    """
    for pid in ids:
        src = os.path.join(local_dataset, f"{pid}.jpg")
        if not os.path.exists(src):
            continue
        meta = {}
        if need_meta:
            meta_path = os.path.join(local_dataset, f"{pid}.json")
            if not os.path.exists(meta_path):
                raise SystemExit(
                    f"{meta_path} is missing; a full build needs every pano's sidecar "
                    "metadata. Point --local-dataset at a complete download_dataset.py "
                    "output, or use --images-only, which never reads the sidecars.")
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except ValueError as e:
                raise SystemExit(f"{meta_path} is not valid JSON ({e}); a full build "
                                 "cannot read this pano's metadata.")
        dst = os.path.join(panos_dir, f"{pid}.jpg")
        shutil.copyfile(src, dst)
        w, h = image_dims_path(dst)
        yield pid, make_record(pid, w, h, meta)


def record_panos(records_path):
    """``{panorama_id: pano dict}`` from an existing records.jsonl.

    Every failure here means the file is not a gold bundle's records.jsonl, so each
    one exits with the same "rebuild it" guidance the surrounding guards give, rather
    than a bare KeyError out of dict indexing. Duplicate ids are an error rather than
    a silent set-collapse: they would let the drift check below pass on a bundle that
    is internally inconsistent.
    """
    panos = {}
    with open(records_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                pano = json.loads(line)["pano"]
                pid = pano["panorama_id"]
            except (ValueError, KeyError, TypeError) as e:
                raise SystemExit(
                    f"{records_path}:{n} is not a gold record ({type(e).__name__}: {e}); "
                    "the bundle needs a full rebuild (--force), not an image fetch.")
            if pid in panos:
                raise SystemExit(
                    f"{records_path}:{n} repeats panorama_id {pid!r}; the bundle needs a "
                    "full rebuild (--force), not an image fetch.")
            panos[pid] = pano
    return panos


def committed_dims(pano):
    """(width, height) a record claims for its pano, or None if it does not say."""
    w, h = pano.get("width"), pano.get("height")
    return (w, h) if isinstance(w, int) and isinstance(h, int) else None


def usable_on_disk(panos_dir, ids, expected):
    """Ids already fetched AND consistent with the size records.jsonl claims.

    Skipping these is what makes a preempted fetch resumable — the hf path measures in
    hours (see the module docstring), so restarting from zero is expensive. A file that
    is absent, unreadable (a truncated write from a killed job) or the wrong size is
    deliberately left out, so the fetch overwrites it instead of trusting it.
    """
    ok = []
    for pid in ids:
        path = os.path.join(panos_dir, f"{pid}.jpg")
        if not os.path.exists(path):
            continue
        try:
            dims = image_dims_path(path)
        except Exception:
            continue
        if pid in expected and dims != expected[pid]:
            continue
        ok.append(pid)
    return ok


def manifest_hint(manifest_path):
    return (f"note: {manifest_path} does not exist, so this fetch was checked against "
            "records.jsonl's pixel sizes and bundle_meta.json's source, NOT against a "
            "content hash. The nine city splits all carry an imagery_manifest.json; "
            "write this split's from a machine holding all 1,000 panos with:\n"
            "  python scripts/analysis/imagery_manifest.py --write --cities manual_gold")


def check_manifest(panos_dir, manifest_path):
    """``(ok, message)`` for the committed sha256 imagery manifest.

    Reuses scripts/analysis/imagery_manifest.py — the tool that already writes and
    verifies this file for the nine city splits — so "the right bytes" keeps one
    definition across every split instead of gaining a second, special-cased one here.
    ``ok`` is False only on a real mismatch; an absent manifest is reported rather than
    failed, because manual_gold has none committed yet.
    """
    if not os.path.exists(manifest_path):
        return True, manifest_hint(manifest_path)
    analysis_dir = os.path.join(REPO_ROOT, "scripts", "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from imagery_manifest import compare, digest_of, scan

    with open(manifest_path, encoding="utf-8") as f:
        recorded = json.load(f)
    if not isinstance(recorded, dict) or "panos" not in recorded:
        return False, f"{manifest_path} has no 'panos' section; it is not an imagery manifest."
    entries = scan(panos_dir)
    ok, missing, extra, changed = compare(entries, recorded["panos"])
    if ok:
        return True, (f"imagery_manifest.json: {len(entries)} pano(s) match the committed "
                      f"hashes (digest {digest_of(entries)}).")
    lines = [f"fetched imagery does NOT match {manifest_path} — do not score this bundle:"]
    if changed:
        lines.append(f"  {len(changed)} pano(s) present with different bytes, "
                     f"e.g. {changed[:5]}")
    if missing:
        lines.append(f"  {len(missing)} pano(s) in the manifest but not on disk, "
                     f"e.g. {missing[:5]}")
    if extra:
        lines.append(f"  {len(extra)} pano(s) on disk but not in the manifest, "
                     f"e.g. {extra[:5]}")
    return False, "\n".join(lines)


def report_manifest(panos_dir, manifest_path, enabled):
    if not enabled:
        return
    ok, message = check_manifest(panos_dir, manifest_path)
    if not ok:
        raise SystemExit(message)
    print(message)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch the manual gold set's imagery (issue #58).")
    ap.add_argument("--audit", action="store_true",
                    help="Id-only split-membership + overlap audit; downloads no images.")
    ap.add_argument("--source", choices=["hf", "local"], default="hf",
                    help="'hf' = raw bytes from the Hub parquet (canonical); 'local' = "
                         "byte-for-byte copies from an existing dataset/test directory "
                         "(download_dataset.py output, which re-encoded at quality 95).")
    ap.add_argument("--local-dataset", default=os.path.join(REPO_ROOT, "dataset", "test"),
                    help="Source directory for --source local.")
    ap.add_argument("--images-only", action="store_true",
                    help="Fetch the imagery into benchmark/manual_gold/panos/ and touch "
                         "nothing committed: records.jsonl and bundle_meta.json stay exactly "
                         "as they are. This is the per-machine re-fetch path — imagery is "
                         "git-ignored while the records (and the detections exported into "
                         "them) are committed, so a fresh clone wants exactly this. Panos "
                         "already on disk that match records.jsonl are skipped.")
    ap.add_argument("--refetch", action="store_true",
                    help="With --images-only, re-fetch panos that are already on disk "
                         "instead of skipping them.")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild an existing records.jsonl (this DISCARDS any detections "
                         "an earlier export_gold_records.py run wrote into it).")
    ap.add_argument("--no-manifest-check", action="store_true",
                    help="Skip the post-fetch check against imagery_manifest.json (that "
                         "check re-reads every fetched file to hash it).")
    args = ap.parse_args(argv)
    if args.images_only and args.force:
        raise SystemExit("--images-only and --force contradict each other: one promises to "
                         "leave records.jsonl alone, the other rebuilds it. Pick one.")
    if args.audit and (args.images_only or args.force):
        raise SystemExit("--audit downloads nothing, so it cannot also fetch: it would "
                         "silently ignore --images-only/--force. Run the audit, then the "
                         "fetch.")
    if args.refetch and not args.images_only:
        raise SystemExit("--refetch only means anything with --images-only; a full build "
                         "fetches every pano regardless.")

    ids = gold_ids()
    print(f"{len(ids)} gold label files in {LABELS_DIR}")
    if args.audit:
        sys.exit(audit(ids))

    records_path = os.path.join(BUNDLE_DIR, "records.jsonl")
    manifest_path = os.path.join(BUNDLE_DIR, "imagery_manifest.json")
    if os.path.exists(records_path) and not (args.force or args.images_only):
        raise SystemExit(f"{records_path} already exists. Use --images-only to fetch the "
                         "imagery for this machine without touching the committed records "
                         "(the usual case), or --force to rebuild records.jsonl — which "
                         "DISCARDS any detections exported into it.")

    expected = {}
    if args.images_only:
        # Fail fast, before any download. The bundle records which source built it, and
        # the two sources are NOT byte-identical (a JPEG re-encode alone moved the gold
        # numbers by P +2.2 / R -1.8 — see the module docstring), so imagery from the
        # other source under the committed records would silently change what is scored.
        if not os.path.exists(records_path):
            raise SystemExit("--images-only needs an existing records.jsonl to be "
                             "consistent with; run the full fetch (no flags) instead.")
        meta_path = os.path.join(BUNDLE_DIR, "bundle_meta.json")
        built_source = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    built_source = json.load(f).get("source")
            except ValueError as e:
                raise SystemExit(f"{meta_path} is not valid JSON ({e}); restore it with "
                                 "git checkout, or rebuild the bundle with --force.")
        # An unknown source is not a pass. Waving the check through exactly when the
        # built source cannot be established is how the wrong pixels get in underneath
        # committed records — the case this guard exists for.
        if not built_source:
            raise SystemExit(
                f"{meta_path} does not record which source built this bundle, so imagery "
                "fetched now cannot be shown to belong under the committed records — and "
                "the two sources are not byte-identical. Restore the committed file "
                "(git checkout benchmark/manual_gold/bundle_meta.json), or rebuild the "
                "whole bundle with --force.")
        if built_source != args.source:
            raise SystemExit(
                f"bundle_meta.json records source={built_source!r} but this fetch would "
                f"use --source {args.source}; the two are not byte-identical. Fetch with "
                f"--source {built_source}, or rebuild the whole bundle with --force.")

        committed = record_panos(records_path)
        drift = sorted(set(committed) ^ set(ids))
        if drift:
            raise SystemExit(f"records.jsonl and manual_labels/ disagree on {len(drift)} "
                             f"id(s) (e.g. {drift[:5]}); the bundle needs a full rebuild "
                             "(--force), not an image fetch.")
        for pid, pano in committed.items():
            dims = committed_dims(pano)
            if dims:
                expected[pid] = dims
        if not expected:
            print("note: records.jsonl carries no width/height, so the fetched imagery "
                  "cannot be cross-checked against it.")

    panos_dir = os.path.join(BUNDLE_DIR, "panos")
    os.makedirs(panos_dir, exist_ok=True)

    already = set()
    if args.images_only and not args.refetch:
        already = set(usable_on_disk(panos_dir, ids, expected))
        if already:
            print(f"{len(already)} pano(s) already on disk and consistent with "
                  "records.jsonl — skipping them (--refetch fetches them anyway).")
    todo = [pid for pid in ids if pid not in already]

    records = {}
    if todo:
        rows = (fetch_hf(todo, panos_dir) if args.source == "hf"
                else fetch_local(todo, panos_dir, args.local_dataset,
                                 need_meta=not args.images_only))
        records = dict(rows)
    else:
        print("nothing to fetch — every gold pano is already on disk.")
    missing = sorted(set(ids) - already - set(records))
    # The one free integrity check --images-only has: the committed records already
    # state each pano's pixel size, so imagery that disagrees is different imagery under
    # the same ids. Only meaningful here — a full build writes the records FROM these
    # dimensions, so there is nothing independent to compare them against.
    wrong = sorted(pid for pid, rec in records.items()
                   if pid in expected
                   and (rec["pano"]["width"], rec["pano"]["height"]) != expected[pid])

    if args.images_only:
        print(f"Fetched {len(records)} pano(s) into {panos_dir}"
              + (f" ({len(already)} already present)" if already else "")
              + "; records.jsonl and bundle_meta.json left untouched.")
        if wrong:
            detail = "; ".join(
                f"{pid}: records say {expected[pid][0]}x{expected[pid][1]}, fetched "
                f"{records[pid]['pano']['width']}x{records[pid]['pano']['height']}"
                for pid in wrong[:3])
            raise SystemExit(
                f"{len(wrong)} fetched pano(s) are not the size the committed records "
                f"claim ({detail}). That is different imagery under the same ids, and "
                "scoring it would silently move the gold numbers. Check --source, or "
                "rebuild the whole bundle with --force.")
        if missing:
            raise SystemExit(f"{len(missing)} gold pano(s) NOT found in the {args.source} "
                             f"source (imagery is incomplete): {missing[:10]}"
                             + (" ..." if len(missing) > 10 else ""))
        report_manifest(panos_dir, manifest_path, not args.no_manifest_check)
        print("All gold panos fetched.")
        return

    with open(records_path + ".tmp", "w", encoding="utf-8") as f:
        for pid in sorted(records):
            f.write(json.dumps(records[pid], ensure_ascii=False) + "\n")
    os.replace(records_path + ".tmp", records_path)

    meta = {
        "built": datetime.date.today().isoformat(),
        "source": args.source,
        "hf_dataset": HF_DATASET,
        "hf_split": "test",
        "n_panos": len(records),
        "imagery": "gsv",
    }
    with open(os.path.join(BUNDLE_DIR, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(records)} panos + records.jsonl to {BUNDLE_DIR}")
    if missing:
        raise SystemExit(f"{len(missing)} gold pano(s) NOT found in the {args.source} source "
                         f"(bundle is incomplete): {missing[:10]}"
                         + (" ..." if len(missing) > 10 else ""))
    report_manifest(panos_dir, manifest_path, not args.no_manifest_check)
    print("All gold panos fetched. Next: scripts/export_gold_records.py (GPU).")


if __name__ == "__main__":
    main()
