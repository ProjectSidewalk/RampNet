"""Re-derive a GPU-hour figure that was transcribed from a live `sacct` (#143).

A number copied out of `sacct` on a given morning is a snapshot: every incarnation
alive after `-S` counts, and a job still running at that moment counts the hours it
has run *so far*. Checking such a figure against a dump pulled weeks later is not a
running sum over finished jobs -- that method crosses the same figure hours later,
because it waits for each incarnation to end before counting any of it. This script
does the snapshot: each incarnation's elapsed, truncated at the query instant.

    python scripts/analysis/gpu_hours_as_of.py \\
        --from-file docs/data/compute/sacct_klone_2026-08-19.txt \\
        --since 2026-07-24 --at 2026-07-30T07:00 --job-name yolo_curb_ramp_train

reproduces docs/tillicum.md's "496.5 GPU-hours consumed on the baseline since
2026-07-24", written 2026-07-30 at 07:07: 497.5 at 07:00, and 496.5 itself about
seventeen minutes before the line was written (its commit landed at 08:15, when the
snapshot reads 503.9). Without `--job-name` the same
snapshot is 553.2, which is what that sentence would have said had it been an
account-wide figure. Timestamps are the cluster's own clock, as sacct prints them.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slurm_usage import gpu_hours_as_of, parse_sacct  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--from-file", required=True,
                    help="A saved `sacct -X -D -P -n` dump (see slurm_usage.py).")
    ap.add_argument("--at", required=True, type=datetime.fromisoformat,
                    help="The instant the figure was read, e.g. 2026-07-30T07:00.")
    ap.add_argument("--since", type=datetime.fromisoformat,
                    help="The -S the original query used, e.g. 2026-07-24.")
    ap.add_argument("--job-name", help="Count only this job name.")
    args = ap.parse_args()

    rows = parse_sacct(Path(args.from_file).read_text(encoding="utf-8"))
    hours, n = gpu_hours_as_of(rows, args.at, args.since, args.job_name)
    scope = f"job name {args.job_name}" if args.job_name else "every job name"
    window = f" started or alive after {args.since.isoformat()}" if args.since else ""
    print(f"{hours:,.1f} GPU-hours as of {args.at.isoformat()}: {n:,} incarnation(s) "
          f"of {scope}{window}, elapsed truncated at that instant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
