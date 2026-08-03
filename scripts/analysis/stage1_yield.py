"""What did Stage 1 generation cost, and what did it lose? 97.91% yield, all of it Google's fault.

`docs/curb_ramp_data_sourcing.md` §7 calls Stage 1 generation "the long pole and
entirely unmeasured". The paper run's logs survived on a lab scratch volume, so
it is measurable now. Two results, from the committed evidence in
docs/data/rampnet1_stage1_run/:

  - **Yield: 214,599 of 219,170 intended panoramas were written (97.91%).**
    4,571 were never written, and **4,570 of those 4,571 are exactly the panos
    Google refused to serve** ("Failed to fetch panorama for pano_id ..."). One
    is unexplained. Fetch failure is the entire loss mechanism.

  - **The disk-quota wall cost nothing.** The final Slurm incarnation logged
    11,438 `[Errno 122] Disk quota exceeded` errors, which looks like a
    truncated dataset -- but every one of those 11,438 line indices appears in
    progress.txt, i.e. all were completed on a later pass. Quota was an
    operational incident, not data loss. (`download_dataset.py` marks an index
    done only after both the .jpg and .json are written, and re-reads
    progress.txt on restart, so a quota-failed index is simply retried.)

Timing is from Slurm accounting (`sacct -u jsomeara -S 2025-03-01 -E 2025-07-01`)
and is a **lower bound**: `--requeue` overwrites the accounting record, and the
run was finished off in interactive jobs whose stdout never reached the log.

  run_download_dataset.slurm   >=49.1 h across 26 jobs   (fetch + crop-model inference)
  run_generate_meta.slurm       >=13.5 h across  7 jobs
  run_generate_negatives.slurm   >=1.5 h across  2 jobs

So <=4,370 panoramas/hour, against Google's undocumented endpoints at 32 tiles
each. See docs/stage1_generation_cost.md for what that implies for a rebuild.

The 219,170-line finaldataset.jsonl is NOT committed (64 MB); it lives on Hyak
at /gscratch/makelab/jonf/rescue_jsomeara_rampnet/RampNet/stage_one/
dataset_generation/. Everything below reproduces without it, because
missing_panos.csv is committed. Pass --finaldataset to regenerate that CSV from
the manifest and prove it identical.

    python scripts/analysis/stage1_yield.py
    python scripts/analysis/stage1_yield.py --finaldataset /path/to/finaldataset.jsonl
"""
import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(REPO, "docs", "data", "rampnet1_stage1_run")


def load_progress(path):
    """Completed line indices. download_dataset.py appends one per written pano."""
    done = set()
    dupes = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.isdigit():
                idx = int(line)
                dupes += idx in done
                done.add(idx)
    return done, dupes


def load_log(path):
    """Unique pano_ids Google refused, and line indices that hit the quota wall."""
    failed, quota = set(), set()
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith("Failed to fetch panorama for pano_id "):
                failed.add(line.strip().split()[-1])
            elif "Disk quota exceeded" in line and "line index" in line:
                try:
                    quota.add(int(line.split("line index")[1].split(":")[0]))
                except (IndexError, ValueError):
                    pass
    return failed, quota


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence-dir", default=DEFAULT_DIR,
                    help="committed Stage 1 evidence (default: docs/data/rampnet1_stage1_run)")
    ap.add_argument("--finaldataset", default=None,
                    help="optional 64 MB finaldataset.jsonl: regenerate missing_panos.csv "
                         "from it and verify it matches the committed copy")
    args = ap.parse_args()

    done, dupes = load_progress(os.path.join(args.evidence_dir, "progress.txt"))
    failed, quota = load_log(os.path.join(args.evidence_dir, "download_dataset.out"))

    with open(os.path.join(args.evidence_dir, "missing_panos.csv")) as handle:
        missing = [(int(r["line_index"]), r["pano_id"]) for r in csv.DictReader(handle)]
    missing_ids = {pid for _, pid in missing if pid}

    intended = len(done) + len(missing)
    print("== yield ==")
    print(f"  intended (finaldataset.jsonl lines) : {intended:,}")
    print(f"  written  (progress.txt, unique)     : {len(done):,}   "
          f"(duplicate lines: {dupes})")
    print(f"  never written                       : {len(missing):,}  "
          f"({len(missing) / intended:.2%})")
    print(f"  yield                               : {len(done) / intended:.2%}")

    print("\n== why they are missing ==")
    hit = len(missing_ids & failed)
    print(f"  unique 'Failed to fetch' in last log: {len(failed):,}")
    print(f"  never-written that Google refused   : {hit:,} / {len(missing):,} "
          f"({hit / len(missing):.2%})")
    print(f"  unexplained                         : {len(missing) - hit:,}")

    print("\n== the disk-quota scare ==")
    still = quota - done
    print(f"  quota-failed line indices in log    : {len(quota):,}")
    print(f"  ...still missing                    : {len(still):,}")
    print("  => quota cost nothing; every quota-failed index completed on a later pass"
          if not still else f"  => {len(still):,} genuinely lost to quota")

    if args.finaldataset:
        regen = []
        with open(args.finaldataset) as handle:
            for i, line in enumerate(handle):
                if i not in done:
                    try:
                        regen.append((i, json.loads(line)["pano_id"]))
                    except (ValueError, KeyError):
                        regen.append((i, ""))
        same = regen == missing
        print(f"\n== regenerated from {os.path.basename(args.finaldataset)} ==")
        print(f"  lines: {i + 1:,}   missing: {len(regen):,}   "
              f"matches committed CSV: {'YES' if same else 'NO'}")
        if not same:
            raise SystemExit("regenerated missing-pano list does NOT match the committed CSV")


if __name__ == "__main__":
    main()
