#!/usr/bin/env python
"""Run every analysis check in the repo, together, against the current tree.

Why this exists: several analysis scripts pin *another* analysis' output (e.g.
``da3_calibration_101.py`` pins ``analysis_out/recall_by_depth_112.json``, which
``recall_by_depth_112.py --only <split>`` appends to). Two PRs that each pass their own
check can break each other's pins at merge time, and nothing ran all of them together.
This is that gate. **Run it with ``--all`` after merging any analysis PR, and after
merging main into one.** ``docs/replication.md`` ("Running every check") has the table of
what each entry proves and the known gaps.

The registry below is explicit on purpose -- no discovery. A new script with a check mode
gets an entry here; ``tests/test_check_all.py`` proves every entry's argv is accepted by
that script's argparse. Each step runs as ``python <argv>`` in a subprocess with the repo
root as cwd, so the scripts' own relative paths resolve exactly as their docstrings say.

Two placeholders may appear in an argv:

  * ``{tmp}``        -- a fresh temporary directory for that entry (deleted afterwards),
                        for scripts whose check is "regenerate into scratch, then compare"
  * ``{local_root}`` -- ``--local-root`` (default: the repo root), the checkout that holds
                        the git-ignored local data a ``local-cache`` entry reads

Requirements an entry may declare (none == committed inputs only, CPU, offline):

  * ``local-cache``   -- git-ignored local data (``.model_cache/``, ``benchmark/*/panos/``)
                         that a clean clone does not have
  * ``labeler-root``  -- a sidewalk-auto-labeler checkout (external repo)
  * ``network``       -- Hugging Face / Google / any network beyond pip and GitHub
  * ``gpu``           -- CUDA

An entry with requirements is SKIPped, with the reason, unless ``--allow <req>`` is given
(``--ci`` ignores ``--allow``). Entries marked slow run only with ``--all``.

After every entry the runner compares ``git status --porcelain`` with what it was before:
a check that rewrites a tracked file (or leaves an untracked one) FAILs, because a check
must write nothing.

Usage::

    python scripts/check_all.py --all        # the full gate (after merging analysis PRs)
    python scripts/check_all.py --ci         # what CI runs: no requirements, not slow
    python scripts/check_all.py --list       # the registry; nothing run
    python scripts/check_all.py --only da3_calibration_101 --only recall_by_depth_112
    python scripts/check_all.py --all --allow local-cache --local-root D:/Git/RampNet
    python scripts/check_all.py --all --json analysis_out/check_all/latest.json

Exit code: 1 if any entry FAILs (SKIPs do not fail the run), 2 on a usage error. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_REQUIREMENTS = ("local-cache", "labeler-root", "network", "gpu")


@dataclass(frozen=True)
class Entry:
    name: str
    steps: tuple           # tuple of argv tuples; each argv = (script relative to repo root, *args)
    verifies: str          # one line: what a PASS proves
    requires: tuple = ()   # subset of KNOWN_REQUIREMENTS; () == committed inputs, CPU, offline
    slow: bool = False     # True: runs only with --all (and so is not in CI)
    note: str = ""         # known gap / what unblocks it; printed on SKIP

    @property
    def scripts(self) -> list:
        return sorted({argv[0] for argv in self.steps})


def E(name, argv, verifies, **kw):
    """One-step entry."""
    return Entry(name, (tuple(argv),), verifies, **kw)


AN = "scripts/analysis/"

# Order is the run order: cheap first, so a broken tree fails fast.
REGISTRY: list = [
    E("sourcing_tables", (AN + "sourcing_tables.py", "--check"),
      "generated tables in the data-sourcing docs re-render identical from committed files (#145)"),
    E("yolo_warmup_dip_72", (AN + "yolo_warmup_dip_72.py", "--check"),
      "pinned facts of the YOLO warm-up LR dip hold on the committed curves (#72)"),
    E("yolo_geometry_51", (AN + "yolo_geometry_51.py", "--check"),
      "YOLO control leg reproduces its scoreboard row; committed JSON matches a fresh read (#51)"),
    E("run_b_gate_135", (AN + "run_b_gate_135.py", "--check"),
      "Run-B gate decision artifact re-derives from the committed CSVs + power JSON (#135)"),
    E("seed_variance_read_51_135", (AN + "seed_variance_read_51_135.py", "--check"),
      "seed-variance read JSON matches a fresh build from docs/data/seed_variance_51_135/"),
    E("operating_point_parity_51", (AN + "operating_point_parity_51.py", "--check"),
      "operating-point parity JSON matches a fresh build over every committed report (#51)"),
    E("recall_by_depth_112", (AN + "recall_by_depth_112.py", "--check"),
      "recall-by-depth tables re-derive from the committed rows (no payloads) (#112)"),
    E("da3_calibration_101", (AN + "da3_calibration_101.py", "--check"),
      "DA3 rows re-derive from raw files + bundles + #112 JSON; tables, md, SHA256SUMS hold (#101)"),
    E("laurens_paired_151", (AN + "laurens_paired_151.py", "--check"),
      "Laurens paired rows (all but the curb probe) and tables re-derive from committed inputs (#151)"),
    E("crossview_fresh_48", (AN + "crossview_fresh_48.py", "pairs", "--check"),
      "fresh cross-view pair list + meta re-derive byte-identical (FRESH_PAIRS_SHA256) (#48)"),
    E("two_scale_197", (AN + "two_scale_197.py", "report", "--check"),
      "two-scale results.json/.md regenerate byte-identical from the committed #196 caches (#197)"),
    Entry("input_res_sweep_25", (
        (AN + "input_res_sweep_25.py", "sums"),
        (AN + "input_res_sweep_25.py", "check", "--out", "{tmp}/instrument_check.json"),
        (AN + "input_res_sweep_25.py", "report", "--out", "{tmp}/results.json"),
        (AN + "input_res_sweep_25.py", "sums", "--root", "{tmp}", "--partial"),
    ), "SHA256SUMS holds; r2048 reproduces op_cache; check + report regenerate to the pinned hashes (#25)"),
    E("yolo_rescore_benchmark_eval",
      ("scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py", "--check"),
      "yolo_baseline/benchmark_eval/ matches the current scorer over the committed detections"),
    E("cascade_transfer_35", (AN + "cascade_transfer_35.py", "--check"),
      "cascade transfer table regenerates byte-identical from committed GT, op_cache, detections (#35)"),
    E("scoreboard", (AN + "scoreboard.py", "--check"),
      "scoreboard doc tables, its prose counts, the log tables and scoreboard.json match a fresh scoring"),
    E("cascade_cost_35", (AN + "cascade_cost_35.py", "--check"),
      "every per-pair cascade file regenerates byte-identical from its recorded args (#35)",
      slow=True),
    # -- need git-ignored local data: never in CI, run with --allow local-cache ---------------
    E("export_model_cache_verify",
      ("scripts/analysis/export_model_cache.py", "--verify", "--cache-dir", "{local_root}/.model_cache"),
      "published benchmark/model_detections score identically to the local .model_cache",
      requires=("local-cache",),
      note=".model_cache/ is local; the published detections it is compared with are committed"),
    E("imagery_manifest_verify",
      ("scripts/analysis/imagery_manifest.py", "--verify", "--panos-root", "{local_root}"),
      "local benchmark/*/panos/ JPEGs match the committed imagery manifests",
      requires=("local-cache",),
      note="benchmark/*/panos/ is git-ignored; it comes from the HF dataset"),
]


def by_name() -> dict:
    return {e.name: e for e in REGISTRY}


def skip_reason(entry: Entry, *, ci: bool, run_all: bool, allowed: set):
    """None if ``entry`` should run, else the reason it is skipped."""
    missing = [r for r in entry.requires if ci or r not in allowed]
    if missing:
        reason = "requires " + ",".join(missing)
        return reason + (f" -- {entry.note}" if entry.note else "")
    if entry.slow and not run_all:
        return "slow; runs only with --all"
    return None


def expand(argv, tmp: str, local_root: str) -> list:
    return [a.replace("{tmp}", tmp).replace("{local_root}", local_root) for a in argv]


def git_status():
    try:
        p = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def run_entry(entry: Entry, timeout: float, local_root: str) -> tuple:
    """Run every step; stop at the first failure. Returns (ok, detail, output, seconds)."""
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    tmp = tempfile.mkdtemp(prefix=f"check_all_{entry.name}_")
    out_all, t0 = [], time.monotonic()
    ok, detail = True, ""
    try:
        for i, argv in enumerate(entry.steps, 1):
            cmd = [sys.executable, *expand(argv, tmp, local_root)]
            out_all.append(f"$ python {' '.join(cmd[1:])}")
            try:
                p = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                rc, out = p.returncode, p.stdout.decode("utf-8", "replace")
            except subprocess.TimeoutExpired as ex:
                rc, out = -1, (ex.stdout or b"").decode("utf-8", "replace")
                out += f"\n[check_all] timed out after {timeout:.0f}s"
            out_all.append(out.rstrip())
            if rc != 0:
                ok = False
                detail = f"step {i}/{len(entry.steps)} exit {rc}" if len(entry.steps) > 1 else f"exit {rc}"
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok, detail, "\n".join(out_all), time.monotonic() - t0


def print_table(rows) -> None:
    w = max([len(r["name"]) for r in rows] + [4])
    print(f"\n{'name':<{w}}  {'status':<6}  {'seconds':>8}  detail")
    print(f"{'-' * w}  {'-' * 6}  {'-' * 8}  ------")
    for r in rows:
        secs = "" if r["seconds"] is None else f"{r['seconds']:.1f}"
        print(f"{r['name']:<{w}}  {r['status']:<6}  {secs:>8}  {r['detail']}")


def list_registry() -> None:
    for e in REGISTRY:
        req = ",".join(e.requires) or "none"
        print(f"{e.name}  [requires={req}{'; slow' if e.slow else ''}]")
        print(f"    {e.verifies}")
        for argv in e.steps:
            print(f"    $ python {' '.join(argv)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring and docs/replication.md.")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="run only this entry (repeatable); requirements and --all still apply")
    ap.add_argument("--ci", action="store_true",
                    help="committed inputs only: skip every entry with a requirement, ignoring --allow")
    ap.add_argument("--all", dest="run_all", action="store_true",
                    help="also run entries marked slow (the full gate)")
    ap.add_argument("--allow", action="append", default=[], choices=KNOWN_REQUIREMENTS,
                    help="treat this requirement as satisfied (repeatable; ignored with --ci)")
    ap.add_argument("--local-root", default=REPO,
                    help="checkout holding the git-ignored data local-cache entries read "
                         "(default: this repo; from a worktree, point it at the main checkout)")
    ap.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-step timeout in seconds (default 3600)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every entry's output, not only failures'")
    args = ap.parse_args(argv)
    try:   # child output is UTF-8 (deltas, Greek letters); a cp1252 console must not crash on it
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.list:
        list_registry()
        return 0

    names = by_name()
    bad = [n for n in args.only if n not in names]
    if bad:
        ap.error(f"unknown entry {bad}; see --list")
    entries = [names[n] for n in args.only] if args.only else REGISTRY
    local_root = os.path.abspath(args.local_root)

    rows = []
    for e in entries:
        skip = skip_reason(e, ci=args.ci, run_all=args.run_all, allowed=set(args.allow))
        if skip:
            rows.append({"name": e.name, "status": "SKIP", "seconds": None, "detail": skip})
            print(f"[SKIP] {e.name}: {skip}", flush=True)
            continue
        print(f"[run ] {e.name}", flush=True)
        before = git_status()
        ok, detail, out, secs = run_entry(e, args.timeout, local_root)
        after = git_status()
        if ok and before is not None and after is not None and before != after:
            changed = sorted(set(after.splitlines()) ^ set(before.splitlines()))
            ok, detail = False, "modified the working tree: " + "; ".join(changed[:5])
        status = "PASS" if ok else "FAIL"
        rows.append({"name": e.name, "status": status, "seconds": round(secs, 1), "detail": detail})
        if not ok or args.verbose:
            lines = out.splitlines()
            shown = lines if args.verbose else lines[-40:]
            print(f"---- {e.name} output{'' if args.verbose else ' (last 40 lines)'} ----")
            print("\n".join(shown))
            print(f"---- end {e.name} ----", flush=True)

    print_table(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "SKIP")}
    total = sum(r["seconds"] or 0 for r in rows)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"in {total:.1f}s")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"mode": {"ci": args.ci, "all": args.run_all, "allow": sorted(args.allow)},
                       "counts": counts, "results": rows}, f, indent=1)
            f.write("\n")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
