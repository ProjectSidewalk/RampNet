"""Where a cost record goes, and how to total one.

Shared by both ledgers — the per-run API/time record (`compare.py --usage-log`
-> `analysis_out/usage_log.jsonl`) and the per-job compute record
(`scripts/analysis/slurm_usage.py` -> `analysis_out/compute_log.jsonl`) — because
the way to lose a cost record is the same for both, and it has already happened
twice.

**A ledger is not a commit.** #123's four Claude legs spent $28.82 and left no
record. #139's claude-opus-5 leg spent $70.41 and left no record: it ran from a
scratch worktree, so the default log path resolved *inside that worktree*, and the
worktree was deleted. Neither loss was visible to the #119 guard, which proves a
log path was accepted and says nothing about whether the file survives.

Hence `canonical_repo_root`: git's *common* dir is shared by every worktree of a
repo, so its parent is the one checkout that outlives them all, and every worktree
appends to a single ledger in the main checkout.
"""
import json
import subprocess
from pathlib import Path

#: A row's provenance. Absent means ``MEASURED``: every row written before this
#: convention existed was taken at run time.
MEASURED = "measured"
#: Reconstructed after the fact from the provider's billing telemetry, because layer 1
#: failed to write and the spend would otherwise be absent from every total.
RECOVERED = "recovered"


def row_kind(rec):
    """``MEASURED`` or ``RECOVERED`` for one ledger row.

    **A recovered row is not a measurement, and the difference matters for
    reconciliation.** It
    carries real money and real token counts, but it was read *off the bill*, so:

    - it has **no per-split attribution** — recovery is per-model per-day, and which
      split spent what is permanently gone;
    - it must **never be fed back into reconciliation**. ``vertex_usage.reconcile``
      compares the ledger against that same bill, so counting a recovered row as
      "logged" makes the bill agree with itself and reports ``ok`` for precisely the
      gap the check exists to find.

    Totals still include it — a cost table that omits recovered spend is wrong by the
    whole amount, which for #139 was a factor of about 200.
    """
    return rec.get("kind") or MEASURED


def canonical_repo_root(start):
    """The MAIN checkout for ``start``, even when ``start`` is a linked worktree.

    Falls back to ``start`` whenever git cannot answer — a tarball, an HF clone,
    no git on PATH. Bookkeeping must never be the reason a run refuses to start."""
    start = Path(start)
    try:
        proc = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                              cwd=str(start), capture_output=True, text=True,
                              timeout=10)
    except (OSError, subprocess.SubprocessError):
        return start
    if proc.returncode != 0 or not proc.stdout.strip():
        return start
    # git returns a bare ".git" when run from the root of a normal checkout and an
    # absolute path from a worktree; Path handles both.
    root = (start / proc.stdout.strip()).resolve().parent
    return root if root.is_dir() else start


def read_rows(path):
    """Every parseable row in a JSONL ledger, or None if it cannot be read.

    Tolerant by construction: these files are append-only and committed, so they
    hold rows written by older versions of the code, and one unparseable line must
    never cost a reader the rest of the record."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return None
    return rows


def append_rows(path, rows):
    """Append rows to a JSONL ledger, creating it if needed.

    newline="" so a Windows run appends LF, not CRLF: these ledgers are
    append-only and byte-compared in review, and a CRLF line silently breaks that
    (the same defect imagery_manifest.py was fixed for)."""
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def ledger_totals(path):
    """(rows, total USD, wall-clock hours, recovered USD) in an existing ledger.

    Printed after every logged leg so a run that wrote somewhere unexpected is
    visible *while someone is still watching*, rather than weeks later when the
    provider's usage telemetry has aged out. Rows predating #143 carry no timing
    keys and free rows carry no cost; both are counted as the zero they are.

    ``recovered`` is the part of the USD total that came from :data:`RECOVERED` rows
    rather than from measurement. It is reported separately, not subtracted: the
    money was really spent, but nobody timed it and no split can claim it, so a
    reader should never mistake it for an as-run number."""
    rows = read_rows(path)
    if rows is None:
        return None
    usd = sum(r.get("est_cost_usd") or 0 for r in rows)
    seconds = sum(r.get("elapsed_s") or 0 for r in rows)
    recovered = sum(r.get("est_cost_usd") or 0
                    for r in rows if row_kind(r) == RECOVERED)
    return len(rows), usd, seconds / 3600.0, recovered
