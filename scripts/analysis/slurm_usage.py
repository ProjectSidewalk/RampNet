"""What our cluster jobs cost, in GPU-hours and dollars, from Slurm's own accounting.

The API half of an experiment's cost has been recorded per run since #119
(`compare.py --usage-log` -> `analysis_out/usage_log.jsonl`). The compute half was
not recorded anywhere: Tillicum's $0.90/GPU-hour and klone's GPU-hours lived only
as prose in `docs/tillicum.md`, written by hand, per job, when someone remembered.
This is the missing ledger (#143).

Unlike API tokens, this half is **partly back-fillable** — `sacct` retains job
records — which is the whole reason to run it now rather than later.

    # on a login node (klone, tillicum):
    python scripts/analysis/slurm_usage.py --cluster tillicum --since 2026-07-01

    # from a machine that cannot reach the cluster: save the dump there, parse here
    ssh klone 'sacct -X -D -P -n -u $USER -S 2026-07-01 \\
        --format=JobID,JobName%60,Cluster,Partition,QOS,State,Submit,Start,End,\\
ElapsedRaw,AllocTRES,NNodes,ExitCode' > sacct_klone.txt
    python scripts/analysis/slurm_usage.py --cluster klone --from-file sacct_klone.txt

`--print-command` prints that command for the current settings rather than guessing.

**Replication:** `sacct` output is retrievable only by someone with an account on the
cluster, so `--save-raw` writes the exact dump a run parsed. Commit it next to the
numbers it produced (`docs/data/compute/`) and the ledger becomes re-derivable from a
clean clone with no cluster access — the same reason `usage_log.jsonl` is committed
while `vertex_usage.py` needs cloud credentials.

Three things this gets right that a hand tally does not:

- **`-D` (duplicates).** Slurm shows only the last incarnation of a requeued job by
  default. Our klone runs live on the preemptable `ckpt` partition — the paper's
  Stage 2 run was 15 preemptions and 44.7 h of active compute across 74.6 h of
  calendar — so without `-D` the compute is undercounted by whatever the earlier
  incarnations burned.
- **GPU-hours = elapsed x N GPUs**, which is how Tillicum bills: an idle GPU in a
  2-GPU job costs exactly as much as a busy one.
- **Idempotent.** Rows are keyed by (cluster, job id, start), so re-running after
  more jobs finish appends only what is new. A job first recorded while RUNNING is
  re-appended once it reaches a terminal state; readers take the last row per key.
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from rampnet import ledger  # noqa: E402
from pricing import compute_price_for, estimate_compute_cost  # noqa: E402

DEFAULT_COMPUTE_LOG_REL = os.path.join("analysis_out", "compute_log.jsonl")

# Pinned, because the parser indexes by position: a caller who changes one must
# change the other. %60 widens JobName, which sacct otherwise truncates to 8 chars
# and would make every job name in the ledger useless.
SACCT_FIELDS = ("JobID", "JobName%60", "Cluster", "Partition", "QOS", "State",
                "Submit", "Start", "End", "ElapsedRaw", "AllocTRES", "NNodes",
                "ExitCode")
COLUMNS = [f.split("%")[0] for f in SACCT_FIELDS]

# Both spellings Slurm uses: `gres/gpu=2` and the typed `gres/gpu:a40=2`. A job
# usually reports both, which is why the generic count wins rather than being added.
GPU_TRES = re.compile(r"gres/gpu(?::([^=,]+))?=(\d+)")

# Terminal from the accounting point of view: the elapsed time will not grow.
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED",
                   "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED")


def sacct_command(user, since, until=None):
    cmd = ["sacct", "-X", "-D", "-P", "-n", "-u", user, "-S", since,
           "--format=" + ",".join(SACCT_FIELDS)]
    if until:
        cmd += ["-E", until]
    return cmd


def gpus_from_tres(tres):
    """(GPU count, GPU type or None) from an AllocTRES string.

    The generic ``gres/gpu=N`` wins over the typed ``gres/gpu:a40=N`` because
    Slurm reports both for the same GPUs; summing them would double every job."""
    generic, typed = 0, defaultdict(int)
    for gpu_type, count in GPU_TRES.findall(tres or ""):
        if gpu_type:
            typed[gpu_type] += int(count)
        else:
            generic = max(generic, int(count))
    return (generic or sum(typed.values())), (",".join(sorted(typed)) or None)


def is_terminal(state):
    # "CANCELLED by 12345" is one state with a suffix, so match on the head.
    return (state or "").split()[0].upper() in TERMINAL_STATES if state else False


def parse_sacct(text, cluster=None, user=None):
    """Rows for one ledger, from `sacct -P -n` output.

    ``cluster`` overrides sacct's own Cluster column, which reports the Slurm
    cluster name — that is what we want when they agree and a lie when a site
    names its cluster something other than how we price it."""
    rows, overridden = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != len(COLUMNS):
            continue  # a header, a warning, a wrapped line: not a job record
        rec = dict(zip(COLUMNS, parts))
        elapsed_s = int(rec["ElapsedRaw"]) if rec["ElapsedRaw"].isdigit() else 0
        gpus, gpu_type = gpus_from_tres(rec["AllocTRES"])
        gpu_hours = elapsed_s / 3600.0 * gpus
        name = (cluster or rec["Cluster"] or "").lower()
        if cluster and rec["Cluster"] and rec["Cluster"].lower() != cluster.lower():
            overridden.add(rec["Cluster"])
        cost = estimate_compute_cost(name, gpu_hours, rec["QOS"])
        price = compute_price_for(name)
        rows.append({
            "cluster": name,
            "job_id": rec["JobID"],
            "job_name": rec["JobName"],
            "user": user,
            "partition": rec["Partition"] or None,
            "qos": rec["QOS"] or None,
            "state": rec["State"],
            "submit": rec["Submit"] or None,
            "start": rec["Start"] or None,
            "end": rec["End"] or None,
            # Named to match usage_log.jsonl so one reader can total both ledgers.
            "elapsed_s": elapsed_s,
            "nodes": int(rec["NNodes"]) if rec["NNodes"].isdigit() else None,
            "gpus": gpus,
            "gpu_type": gpu_type,
            "gpu_hours": round(gpu_hours, 4),
            "exit_code": rec["ExitCode"] or None,
            "est_cost_usd": round(cost, 4) if cost is not None else None,
            # The rate and the date it was checked, not the whole pricing entry:
            # this ledger runs to thousands of rows (3,991 on klone alone, because
            # ckpt requeues), and embedding the full table in each one was 1.7 MB of
            # the 2.6 MB file. The table itself, caveats included, is versioned in
            # scripts/model_comparison/pricing.py.
            "rate_usd_per_gpu_hour": (price or {}).get("usd_per_gpu_hour"),
            "rate_as_of": (price or {}).get("as_of"),
        })
    if overridden:
        # Silently restamping another cluster's jobs would price them at the wrong
        # rate and attribute their GPU-hours to the wrong machine -- and a dump can
        # legitimately span clusters. Loud, because nothing downstream can detect it.
        print(f"WARNING: --cluster {cluster} overrides the Cluster column on rows "
              f"reporting {', '.join(sorted(overridden))}. Those jobs are now priced "
              f"and attributed as {cluster}. Drop --cluster to trust sacct's own "
              f"names.")
    return rows


def row_key(rec):
    """(cluster, job id, start) — not the job id alone.

    With `-D` a requeued job appears once per incarnation under the same id, and
    on `ckpt` that is routine: collapsing them on job id would throw away most of
    a preempted run's compute."""
    return (rec.get("cluster"), rec.get("job_id"), rec.get("start"))


def new_rows(parsed, existing):
    """The rows worth appending: ones never seen, plus ones whose stored state was
    still non-terminal and has since finished."""
    seen = {}
    for rec in existing or []:
        seen[row_key(rec)] = rec
    out = []
    for rec in parsed:
        prior = seen.get(row_key(rec))
        if prior is None:
            out.append(rec)
        elif not is_terminal(prior.get("state")) and is_terminal(rec.get("state")):
            out.append(rec)
    return out


def summarize(rows):
    """Per-cluster totals: jobs, GPU-hours, dollars. What a paper's methods
    section quotes, and what `hyakusage` is checked against."""
    by_cluster = defaultdict(lambda: {"jobs": 0, "gpu_hours": 0.0, "usd": 0.0,
                                      "elapsed_h": 0.0, "unpriced": 0})
    for rec in rows:
        agg = by_cluster[rec.get("cluster") or "?"]
        agg["jobs"] += 1
        agg["gpu_hours"] += rec.get("gpu_hours") or 0
        agg["elapsed_h"] += (rec.get("elapsed_s") or 0) / 3600.0
        if rec.get("est_cost_usd") is None:
            agg["unpriced"] += 1
        else:
            agg["usd"] += rec["est_cost_usd"]
    return dict(by_cluster)


def print_by_name(rows, top=15):
    """GPU-hours per job name.

    The ledger records every job on the account, including other projects' -- a
    complete measurement, with attribution left to the reader rather than baked
    into the durable artifact by a filter chosen once. This is how the reader sees
    the composition immediately instead of summing the wrong thing."""
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for rec in rows:
        a = agg[rec.get("job_name") or "?"]
        a[0] += 1
        a[1] += rec.get("gpu_hours") or 0
        a[2] += rec.get("est_cost_usd") or 0
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"\n{'job name':<34} {'jobs':>6} {'GPU-h':>10} {'est $':>9}")
    print("-" * 62)
    for name, (jobs, gpu_h, usd) in ranked[:top]:
        print(f"{name:<34.34} {jobs:>6,} {gpu_h:>10,.1f} {usd:>9,.2f}")
    if len(ranked) > top:
        rest = ranked[top:]
        print(f"{f'... {len(rest)} more job name(s)':<34} "
              f"{sum(a[0] for _, a in rest):>6,} {sum(a[1] for _, a in rest):>10,.1f} "
              f"{sum(a[2] for _, a in rest):>9,.2f}")


def print_summary(rows, title):
    print(f"\n{title}")
    print(f"{'cluster':<12} {'jobs':>6} {'GPU-h':>10} {'wall-h':>9} {'est $':>9}")
    print("-" * 50)
    for cluster, agg in sorted(summarize(rows).items()):
        unpriced = f"  ({agg['unpriced']} unpriced)" if agg["unpriced"] else ""
        print(f"{cluster:<12} {agg['jobs']:>6,} {agg['gpu_hours']:>10,.2f} "
              f"{agg['elapsed_h']:>9,.2f} {agg['usd']:>9,.2f}{unpriced}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cluster", help="Cluster name for pricing (tillicum, klone). "
                                      "Defaults to sacct's own Cluster column.")
    ap.add_argument("--user", default=os.environ.get("USER") or os.environ.get("USERNAME"),
                    help="Slurm user to query (default: $USER).")
    ap.add_argument("--since", default="2026-07-01",
                    help="sacct -S start date (default: 2026-07-01, before the first "
                         "RampNet cluster run this ledger covers).")
    ap.add_argument("--until", help="sacct -E end date (default: now).")
    ap.add_argument("--from-file", help="Parse a saved `sacct -P -n` dump instead of "
                                        "running sacct (for machines off the cluster).")
    ap.add_argument("--save-raw", help="Write the sacct output parsed by this run, so "
                                       "the numbers stay re-derivable without cluster "
                                       "access. Commit it beside the results.")
    ap.add_argument("--out", help=f"Ledger to append to (default: "
                                  f"{DEFAULT_COMPUTE_LOG_REL} in the main checkout).")
    ap.add_argument("--by-name", action="store_true",
                    help="Also break the summary down by job name — the ledger holds "
                         "every job on the account, including other projects'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be appended; write nothing.")
    ap.add_argument("--print-command", action="store_true",
                    help="Print the sacct command for these settings and exit.")
    args = ap.parse_args()

    if args.print_command:
        print(" ".join(sacct_command(args.user or "$USER", args.since, args.until)))
        return 0

    if args.from_file:
        text = Path(args.from_file).read_text(encoding="utf-8")
    else:
        cmd = sacct_command(args.user, args.since, args.until)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"could not run sacct ({type(e).__name__}: {e}).\nRun it on a login "
                  f"node and pass the output with --from-file:\n  "
                  + " ".join(sacct_command(args.user or "$USER", args.since, args.until)),
                  file=sys.stderr)
            return 2
        if proc.returncode != 0:
            print(f"sacct failed ({proc.returncode}): {proc.stderr.strip()}", file=sys.stderr)
            return 2
        text = proc.stdout

    if args.save_raw:
        raw = Path(args.save_raw)
        raw.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so the dump keeps the cluster's own LF endings on Windows: it is
        # committed as a replication input and has to stay byte-identical to what
        # sacct emitted.
        raw.write_text(text, encoding="utf-8", newline="")
        print(f"raw sacct output saved to {args.save_raw}")

    rows = parse_sacct(text, cluster=args.cluster, user=args.user)
    if not rows:
        print("no job records parsed — check the sacct format and the date range.")
        return 1
    print_summary(rows, f"Parsed {len(rows):,} job record(s)")
    if args.by_name:
        print_by_name(rows)

    # The ledger belongs in the MAIN checkout: a scratch worktree is deleted when
    # its session ends, which is how #139's $70.41 vanished (#143).
    out = args.out or str(ledger.canonical_repo_root(REPO) / DEFAULT_COMPUTE_LOG_REL)
    existing = ledger.read_rows(out) or []
    fresh = new_rows(rows, existing)
    stamped = [dict(r, recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
               for r in fresh]
    if args.dry_run:
        print(f"\n[dry run] {len(fresh):,} new row(s) would be appended to "
              f"{os.path.abspath(out)}")
        return 0
    if not fresh:
        print(f"\nnothing new: all {len(rows):,} record(s) already in "
              f"{os.path.abspath(out)}")
        return 0
    ledger.append_rows(out, stamped)
    print(f"\nappended {len(fresh):,} new row(s) to {os.path.abspath(out)}")
    print_summary(existing + stamped, "Ledger now holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
