#!/usr/bin/env python
"""Run every analysis check in the repo, together, against the current tree.

Why this exists: several analysis scripts pin *another* analysis' output (e.g.
``da3_calibration_101.py`` pins ``analysis_out/recall_by_depth_112.json``, which
``recall_by_depth_112.py --only <split>`` appends to). Two PRs that each pass their own
check can break each other's pins at merge time, and nothing ran all of them together.
This is that gate. **Run it with ``--all`` after merging any analysis PR, and after
merging main into one.** Running ``--only`` on the analysis you changed is NOT enough: a
change to a file can pass its owner's check and fail a dependent's (a one-byte change to
``recall_by_depth_112.json``'s ``labeler_commit`` passes ``recall_by_depth_112`` and fails
``da3_calibration_101``). ``--touching <path>`` lists and runs every entry that pins a path.
``docs/replication.md`` ("Running every check") has what each entry proves and the gaps.

The registry below is explicit on purpose -- no discovery. A new script with a check mode
gets an entry here; ``tests/test_check_all.py`` proves every entry's argv is accepted by
that script's argparse and every pin exists. Each step runs as ``python <argv>`` in a
subprocess with the repo root as cwd, so the scripts' own relative paths resolve exactly as
their docstrings say.

Entry fields:

  * ``pins``        -- committed files / directories / globs (repo-relative, ``/``) the check
                       compares or re-derives from. Maintained by hand from each script's
                       check code; ``--list`` shows them and ``--touching`` selects on them.
  * ``requires``    -- none == committed inputs only, CPU, offline. Otherwise a subset of:
                       ``local-cache`` (git-ignored local data: ``.model_cache/``,
                       ``benchmark/*/panos/``), ``labeler-root`` (a sidewalk-auto-labeler
                       checkout), ``network`` (anything beyond pip and GitHub), ``gpu``.
                       Such an entry is SKIPped, with the reason, unless ``--allow <req>``
                       (``--ci`` ignores ``--allow``).
  * ``needs_paths`` -- globs (``{local_root}`` expanded) that must match before the entry
                       runs; no match is a SKIP naming what is absent, so a local-data entry
                       never runs against nothing.
  * ``nothing_verified`` -- a function of the entry's output returning a reason when the
                       script exited 0 without comparing anything; that is a FAIL.
  * ``slow``        -- runs only with ``--all``. One entry is slow: ``benchmark_power_135``
                       (about 11 minutes; see the doc).

Placeholders in an argv: ``{tmp}`` (a fresh scratch directory per entry, deleted
afterwards) and ``{local_root}`` (``--local-root``, default the repo root).

**A check must write nothing.** Before the first entry the runner hashes every file
``git ls-files --cached --others --exclude-standard`` lists (tracked plus untracked,
not ignored); after each entry it re-stats them and re-hashes **only the files whose size
or mtime changed** (so a write that preserves both is not seen), and FAILs the entry if any
content changed or a file appeared or disappeared. This
is content-based, so it works on an already-dirty tree: rewriting a file that was already
modified is still caught. **Writes to git-ignored paths are not detected** (most of
``analysis_out/*`` is ignored), nor are writes outside the repo. Without git the detection
is off and the run says so.

Usage::

    python scripts/check_all.py --all        # the full gate (after merging analysis PRs)
    python scripts/check_all.py --ci         # what CI runs: no requirements, not slow
    python scripts/check_all.py --list       # the registry with pins; nothing run
    python scripts/check_all.py --touching analysis_out/recall_by_depth_112.json --list
    python scripts/check_all.py --only da3_calibration_101 --only recall_by_depth_112
    python scripts/check_all.py --all --allow local-cache --local-root D:/Git/RampNet
    python scripts/check_all.py --all --json analysis_out/check_all/latest.json

Exit code: 1 if any entry FAILs; 3 if an entry named with ``--only`` was SKIPped (it was
asked for and did not run) or a ``--touching`` path selects no entry; 2 on a usage error;
otherwise 0. Stdlib only.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_REQUIREMENTS = ("local-cache", "labeler-root", "network", "gpu")


@dataclass(frozen=True)
class Entry:
    name: str
    steps: tuple           # tuple of argv tuples; each argv = (script relative to repo root, *args)
    verifies: str          # one line: what a PASS proves
    pins: tuple = ()       # committed paths / dirs (trailing /) / globs the check reads or compares
    requires: tuple = ()   # subset of KNOWN_REQUIREMENTS; () == committed inputs, CPU, offline
    slow: bool = False     # True: runs only with --all
    note: str = ""         # known gap / what unblocks it; printed on SKIP
    needs_paths: tuple = ()          # globs ({local_root} expanded); none matching -> SKIP
    absent_reason: str = ""          # the SKIP reason when needs_paths match nothing
    nothing_verified: Optional[Callable] = None   # output -> reason if exit 0 compared nothing

    @property
    def scripts(self) -> list:
        return sorted({argv[0] for argv in self.steps})


def E(name, argv, verifies, **kw):
    """One-step entry."""
    return Entry(name, (tuple(argv),), verifies, **kw)


def imagery_nothing_verified(out: str):
    """imagery_manifest.py --verify exits 0 when every split's imagery is absent."""
    verified = re.search(r"\s(OK|MISMATCH)\s*$", out, re.M)
    absent = re.search(r"imagery absent locally", out)
    if absent and not verified:
        return "every split reported 'imagery absent locally': nothing was verified"
    return None


AN = "scripts/analysis/"
GT = ("benchmark/*/records.jsonl", "benchmark/*/verdicts.json")   # a benchmark bundle's GT

# Order is the run order: cheap first, so a broken tree fails fast.
REGISTRY: list = [
    E("manifests_sha256", ("scripts/verify_sha256sums.py",),
      "every committed SHA256SUMS / *.sha256 manifest matches the files present (absent ones counted)",
      pins=("analysis_out/context_fov_86/", "analysis_out/da3_calibration_101/",
            "analysis_out/input_res_sweep_25/", "analysis_out/recall_by_depth_112.json",
            "docs/data/rampnet1_stage1_run/", "docs/data/rampnet1_stage2_run/",
            "docs/data/seed_variance_51_135/", "stage_two/cosine_rung_135_events/",
            "stage_two/run_a_84_events/")),
    E("sourcing_tables", (AN + "sourcing_tables.py", "--check"),
      "generated tables in the data-sourcing docs re-render identical from committed files (#145)",
      pins=("docs/location_precision_assessment_96.md", "docs/curb_ramp_data_sourcing.md",
            "analysis_out/stage1_offset_tolerance.json", "data/inventories/")),
    E("yolo_warmup_dip_72", (AN + "yolo_warmup_dip_72.py", "--check"),
      "pinned facts of the YOLO warm-up LR dip hold on the committed curves (#72)",
      pins=("scripts/model_comparison/yolo_baseline/runs/", "docs/data/seed_variance_51_135/")),
    E("yolo_geometry_51", (AN + "yolo_geometry_51.py", "--check"),
      "YOLO control leg reproduces its scoreboard row; committed JSON matches a fresh read (#51)",
      pins=("docs/data/yolo_geometry_51/", "docs/data/yolo_geometry_51.json")),
    E("run_b_gate_135", (AN + "run_b_gate_135.py", "--check"),
      "Run-B gate decision artifact re-derives from the committed CSVs + power JSON (#135)",
      pins=("docs/data/run_a_84_manual_gold/summary.csv",
            "docs/data/cosine_rung_135_manual_gold/summary.csv",
            "docs/data/benchmark_power_135.json", "docs/data/run_b_gate_135.json")),
    E("seed_variance_read_51_135", (AN + "seed_variance_read_51_135.py", "--check"),
      "seed-variance read JSON matches a fresh build from docs/data/seed_variance_51_135/",
      pins=("docs/data/seed_variance_51_135/", "docs/data/seed_variance_51_135.json",
            "docs/data/run_a_84_detections/run_a_epoch_1__manual_gold.json",
            "docs/data/operating_point_parity_51.json", "docs/data/yolo_geometry_51/",
            "analysis_out/op_cache/")),
    E("operating_point_parity_51", (AN + "operating_point_parity_51.py", "--check"),
      "operating-point parity JSON matches a fresh build over every committed report (#51)",
      pins=("analysis_out/op_cache/", "docs/data/yolo_geometry_51/",
            "docs/data/operating_point_parity_51.json")),
    E("recall_by_depth_112", (AN + "recall_by_depth_112.py", "--check"),
      "recall-by-depth tables re-derive from the committed rows (no payloads) (#112)",
      pins=("analysis_out/recall_by_depth_112.json",)),
    E("da3_calibration_101", (AN + "da3_calibration_101.py", "--check"),
      "DA3 rows re-derive from raw files + bundles + #112 JSON; tables, md, SHA256SUMS hold (#101)",
      pins=("analysis_out/da3_calibration_101/", "analysis_out/recall_by_depth_112.json") + GT),
    E("laurens_paired_151", (AN + "laurens_paired_151.py", "--check"),
      "Laurens paired rows (all but the curb probe) and tables re-derive from committed inputs (#151)",
      pins=("analysis_out/laurens_paired_151.json", "benchmark/laurens_gsv/records.jsonl",
            "benchmark/laurens_gsv/verdicts.json", "benchmark/laurens_mapillary/records.jsonl",
            "benchmark/laurens_mapillary/verdicts.json",
            "analysis_out/input_res_sweep_25/cache/r2048/", "analysis_out/op_cache/",
            "benchmark/model_detections/", "analysis_out/recall_by_depth_112.json")),
    E("crossview_fresh_48", (AN + "crossview_fresh_48.py", "pairs", "--check"),
      "fresh pair list + meta re-derive byte-identical; frozen-300 and eligible lists hash to their pins (#48)",
      pins=("analysis_out/crossview_align_48/fresh/pairs.csv",
            "analysis_out/crossview_align_48/fresh/pairs_meta.json",
            "analysis_out/crossview_align_48/pairs.csv",
            "analysis_out/crossview_align_48/eligible_pairs.csv")),
    E("per_ramp_recall_38", (AN + "per_ramp_recall_38.py", "check"),
      "per-ramp recall results.json + ramps_other_views.csv re-derive byte-identical from #48 captures (#38)",
      pins=("analysis_out/per_ramp_recall_38/", "analysis_out/multiview_48/captures_R25.csv")),
    E("two_scale_197", (AN + "two_scale_197.py", "report", "--check"),
      "two-scale results.json/.md regenerate byte-identical from the committed #196 caches (#197)",
      pins=("analysis_out/two_scale_197/results.json", "analysis_out/two_scale_197/results.md",
            "analysis_out/input_res_sweep_25/cache/r2048/",
            "analysis_out/input_res_sweep_25/cache/u4096/", "analysis_out/usage_log.jsonl")),
    Entry("input_res_sweep_25", (
        (AN + "input_res_sweep_25.py", "sums"),
        (AN + "input_res_sweep_25.py", "check", "--out", "{tmp}/instrument_check.json"),
        (AN + "input_res_sweep_25.py", "report", "--out", "{tmp}/results.json"),
        (AN + "input_res_sweep_25.py", "sums", "--root", "{tmp}", "--partial"),
    ), "SHA256SUMS holds; r2048 reproduces op_cache; check + report regenerate to the pinned hashes (#25)",
        pins=("analysis_out/input_res_sweep_25/", "analysis_out/op_cache/",
              "benchmark/*/records.jsonl")),
    E("yolo_rescore_benchmark_eval",
      ("scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py", "--check"),
      "yolo_baseline/benchmark_eval/ matches the current scorer over the committed detections",
      pins=("scripts/model_comparison/yolo_baseline/benchmark_eval/",
            "benchmark/model_detections/") + GT),
    E("cascade_transfer_35", (AN + "cascade_transfer_35.py", "--check"),
      "cascade transfer table regenerates byte-identical from committed GT, op_cache, detections (#35)",
      pins=("analysis_out/cascade_cost_35/transfer/transfer.json", "analysis_out/op_cache/",
            "analysis_out/input_res_sweep_25/cache/r2048/", "benchmark/model_detections/") + GT),
    E("scoreboard", (AN + "scoreboard.py", "--check"),
      "scoreboard doc tables, its prose counts, the log tables and scoreboard.json match a fresh scoring",
      pins=("docs/model_scoreboard.md", "docs/model_comparison.md", "analysis_out/scoreboard.json",
            "benchmark/model_detections/", "manual_labels/") + GT),
    E("cascade_cost_35", (AN + "cascade_cost_35.py", "--check"),
      "every per-pair cascade file regenerates byte-identical from its recorded args (#35)",
      # ~400 s on the desktop, ~5 min on a GitHub runner: in --ci, since the job runs in
      # parallel with pytest (8-9 min) and stays well inside its 30 min guard.
      pins=("analysis_out/cascade_cost_35/", "analysis_out/op_cache/",
            "analysis_out/cascade_gate_op030.json", "benchmark/model_detections/") + GT),
    # Slow: 676 s of bootstrap on the Windows desktop CPU, so --all only, not --ci (#236).
    E("benchmark_power_135", (AN + "benchmark_power_135.py", "--check"),
      "the default command regenerates docs/data/benchmark_power_135.json byte-identical (#135, #236)",
      pins=("docs/data/benchmark_power_135.json", "manual_labels/", "benchmark/model_detections/",
            "docs/data/run_a_84_detections/", "docs/data/run_a_84_manual_gold/summary.csv",
            "analysis_out/op_cache/") + GT,
      slow=True),
    # -- need git-ignored local data: never in CI, run with --allow local-cache ---------------
    E("export_model_cache_verify",
      ("scripts/analysis/export_model_cache.py", "--verify", "--cache-dir", "{local_root}/.model_cache"),
      "published benchmark/model_detections score identically to the local .model_cache",
      pins=("benchmark/model_detections/",), requires=("local-cache",),
      needs_paths=("{local_root}/.model_cache",), absent_reason=".model_cache absent",
      note=".model_cache/ is local; the published detections it is compared with are committed"),
    E("imagery_manifest_verify",
      ("scripts/analysis/imagery_manifest.py", "--verify", "--panos-root", "{local_root}"),
      "local benchmark/*/panos/ JPEGs match the committed imagery manifests",
      pins=("benchmark/*/imagery_manifest.json",), requires=("local-cache",),
      needs_paths=("{local_root}/benchmark/*/panos",), absent_reason="panos absent",
      nothing_verified=imagery_nothing_verified,
      note="benchmark/*/panos/ is git-ignored; it comes from the HF dataset"),
]


def by_name() -> dict:
    return {e.name: e for e in REGISTRY}


def expand(argv, tmp: str, local_root: str) -> list:
    return [a.replace("{tmp}", tmp).replace("{local_root}", local_root) for a in argv]


def skip_reason(entry: Entry, *, ci: bool, run_all: bool, allowed: set, local_root: str = None):
    """None if ``entry`` should run, else the reason it is skipped."""
    local_root = local_root or REPO
    missing = [r for r in entry.requires if ci or r not in allowed]
    if missing:
        reason = "requires " + ",".join(missing)
        return reason + (f" -- {entry.note}" if entry.note else "")
    if entry.slow and not run_all:
        return "slow; runs only with --all"
    for pattern in entry.needs_paths:
        p = expand((pattern,), "", local_root)[0]
        if not glob.glob(p):
            return f"{entry.absent_reason or 'local data absent'}: nothing matches {p}"
    return None


# --------------------------------------------------------------------------- #
# --touching: which entries pin a path
# --------------------------------------------------------------------------- #
def _norm(path: str, repo: str = None) -> str:
    repo = repo or REPO
    p = path.replace("\\", "/")
    if os.path.isabs(path):
        try:
            p = os.path.relpath(path, repo).replace("\\", "/")
        except ValueError:
            pass
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/")


def path_matches(path: str, pin: str) -> bool:
    """Component-wise glob match on the common prefix: true when ``path`` is the pin, lies
    under a pinned directory, or is a directory holding the pin (or something it globs)."""
    a = [c for c in _norm(path).split("/") if c]
    b = [c for c in pin.replace("\\", "/").rstrip("/").split("/") if c]
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return all(fnmatch.fnmatchcase(a[i], b[i]) for i in range(n))


def touching(path: str) -> list:
    """Every entry whose pins (or its own scripts) match ``path``."""
    return [e for e in REGISTRY
            if any(path_matches(path, pin) for pin in e.pins + tuple(e.scripts))]


# --------------------------------------------------------------------------- #
# write detection (content-based)
# --------------------------------------------------------------------------- #
class TreeWatch:
    """Content snapshot of tracked + untracked-not-ignored files under ``repo``.

    Hashes every file once; afterwards only files whose (size, mtime) moved are re-hashed,
    so a check costs one ``git ls-files`` and one stat per file."""

    def __init__(self, repo: str = None):
        self.repo = repo or REPO
        files = self._list()
        self.enabled = files is not None
        self.state = {f: self._sig(f) for f in files} if self.enabled else {}

    def _list(self):
        try:
            p = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                               cwd=self.repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if p.returncode != 0:
            return None
        return sorted({f for f in p.stdout.decode("utf-8", "replace").split("\0") if f})

    def _stat(self, f):
        try:
            st = os.stat(os.path.join(self.repo, f))
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _sig(self, f, stat=None):
        stat = stat or self._stat(f)
        if stat is None:
            return None                       # tracked but deleted on disk
        h = hashlib.sha256()
        try:
            with open(os.path.join(self.repo, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except (OSError, IsADirectoryError):
            return None
        return (stat, h.hexdigest())

    def changes(self) -> list:
        """['M path' | 'A path' | 'D path'] since the last call; updates the snapshot."""
        if not self.enabled:
            return []
        files = self._list()
        if files is None:
            return []
        out, new = [], {}
        for f in files:
            old = self.state.get(f)
            stat = self._stat(f)
            if old is not None and stat is not None and old[0] == stat:
                new[f] = old
                continue
            sig = self._sig(f, stat) if stat is not None else None
            new[f] = sig
            if old is None and sig is not None and f not in self.state:
                out.append(f"A {f}")
            elif old is None and sig is not None:
                out.append(f"A {f}")          # was deleted, now back
            elif old is not None and sig is None:
                out.append(f"D {f}")
            elif old is not None and sig is not None and old[1] != sig[1]:
                out.append(f"M {f}")
        for f, old in self.state.items():
            if f not in new and old is not None:
                out.append(f"D {f}")
        self.state = new
        return sorted(out)


def git_head(repo: str = None):
    repo = repo or REPO
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.decode().strip() if p.returncode == 0 else None


def git_dirty(repo: str = None):
    repo = repo or REPO
    try:
        p = subprocess.run(["git", "status", "--porcelain"], cwd=repo, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bool(p.stdout.strip()) if p.returncode == 0 else None


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_entry(entry: Entry, timeout: float, local_root: str, repo: str = None) -> dict:
    """Run every step; stop at the first failure."""
    repo = repo or REPO
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    tmp = tempfile.mkdtemp(prefix=f"check_all_{entry.name}_")
    out_all, steps, t0 = [], [], time.monotonic()
    ok, detail = True, ""
    try:
        for i, argv in enumerate(entry.steps, 1):
            cmd = [sys.executable, *expand(argv, tmp, local_root)]
            out_all.append(f"$ python {' '.join(cmd[1:])}")
            try:
                p = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                rc, out = p.returncode, p.stdout.decode("utf-8", "replace")
            except subprocess.TimeoutExpired as ex:
                rc, out = -1, (ex.stdout or b"").decode("utf-8", "replace")
                out += f"\n[check_all] timed out after {timeout:.0f}s"
            out_all.append(out.rstrip())
            steps.append({"argv": list(argv), "exit": rc})
            if rc != 0:
                ok = False
                detail = f"step {i}/{len(entry.steps)} exit {rc}" if len(entry.steps) > 1 else f"exit {rc}"
                break
            if entry.nothing_verified:
                why = entry.nothing_verified(out)
                if why:
                    ok, detail = False, f"nothing verified: {why}"
                    break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": ok, "detail": detail, "output": "\n".join(out_all),
            "seconds": time.monotonic() - t0, "steps": steps}


def recorded_argv(argv) -> list:
    """argv for the --json record without machine-local paths: the --json and --local-root
    values become repo-relative when inside the repo, else a placeholder."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        for flag in ("--json", "--local-root"):
            if a == flag and i + 1 < len(argv):
                out += [a, _local(argv[i + 1], flag)]
                i += 2
                break
            if a.startswith(flag + "="):
                out.append(f"{flag}={_local(a.split('=', 1)[1], flag)}")
                i += 1
                break
        else:
            out.append(a)
            i += 1
    return out


def _local(path: str, flag: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), REPO) if os.path.splitdrive(os.path.abspath(path))[0].lower() \
        == os.path.splitdrive(REPO)[0].lower() else ".."
    if rel == ".":
        return "."
    if rel.startswith(".."):
        return "<main checkout>" if flag == "--local-root" else "<outside the repo>"
    return rel.replace(os.sep, "/")


def print_table(rows) -> None:
    w = max([len(r["name"]) for r in rows] + [4])
    print(f"\n{'name':<{w}}  {'status':<6}  {'seconds':>8}  detail")
    print(f"{'-' * w}  {'-' * 6}  {'-' * 8}  ------")
    for r in rows:
        secs = "" if r["seconds"] is None else f"{r['seconds']:.1f}"
        print(f"{r['name']:<{w}}  {r['status']:<6}  {secs:>8}  {r['detail']}")


def list_registry(entries) -> None:
    for e in entries:
        req = ",".join(e.requires) or "none"
        print(f"{e.name}  [requires={req}{'; slow' if e.slow else ''}]")
        print(f"    {e.verifies}")
        for argv in e.steps:
            print(f"    $ python {' '.join(argv)}")
        if e.needs_paths:
            print(f"    needs: {', '.join(e.needs_paths)}")
        print(f"    pins: {', '.join(e.pins) or '-'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring and docs/replication.md.")
    ap.add_argument("--list", action="store_true", help="print the (selected) registry and exit")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="run only this entry (repeatable). A named entry that is skipped "
                         "makes the run exit 3")
    ap.add_argument("--touching", action="append", default=[], metavar="PATH",
                    help="select every entry that pins PATH (file, directory prefix, or a "
                         "directory holding a pin); repeatable, combines with --only")
    ap.add_argument("--ci", action="store_true",
                    help="committed inputs only: skip every entry with a requirement, ignoring --allow")
    ap.add_argument("--all", dest="run_all", action="store_true",
                    help="also run entries marked slow (the full gate)")
    ap.add_argument("--allow", action="append", default=[], choices=KNOWN_REQUIREMENTS,
                    help="treat this requirement as satisfied (repeatable; ignored with --ci)")
    ap.add_argument("--local-root", default=None,
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

    names = by_name()
    bad = [n for n in args.only if n not in names]
    if bad:
        ap.error(f"unknown entry {bad}; see --list")
    entries = REGISTRY
    untouched = []
    if args.only or args.touching:
        chosen = {n for n in args.only}
        for path in args.touching:
            hit = touching(path)
            print(f"--touching {path}: {', '.join(e.name for e in hit) or 'no entry pins it'}")
            if not hit:
                untouched.append(path)
            chosen |= {e.name for e in hit}
        entries = [e for e in REGISTRY if e.name in chosen]

    if args.list:
        list_registry(entries)
        for path in untouched:
            print(f"--touching {path} selects no entry; exit 3")
        return 3 if untouched else 0

    local_root = os.path.abspath(args.local_root or REPO)
    dirty = git_dirty()
    watch = TreeWatch()
    if not watch.enabled:
        print("[warn] git unavailable: write detection is OFF for this run", flush=True)
    elif dirty:
        print("[note] the working tree is dirty; write detection compares content, so it "
              "still applies", flush=True)

    rows, skipped_named = [], []
    for e in entries:
        skip = skip_reason(e, ci=args.ci, run_all=args.run_all, allowed=set(args.allow),
                           local_root=local_root)
        if skip:
            rows.append({"name": e.name, "status": "SKIP", "seconds": None, "detail": skip,
                         "steps": []})
            print(f"[SKIP] {e.name}: {skip}", flush=True)
            if e.name in args.only:
                skipped_named.append((e.name, skip))
            continue
        print(f"[run ] {e.name}", flush=True)
        res = run_entry(e, args.timeout, local_root)
        ok, detail = res["ok"], res["detail"]
        changed = watch.changes()
        if changed:
            what = "; ".join(changed[:5]) + (f"; ... (+{len(changed) - 5})" if len(changed) > 5 else "")
            ok = False
            detail = (detail + "; " if detail else "") + "modified the working tree: " + what
        status = "PASS" if ok else "FAIL"
        rows.append({"name": e.name, "status": status, "seconds": round(res["seconds"], 1),
                     "detail": detail, "steps": res["steps"]})
        if not ok or args.verbose:
            lines = res["output"].splitlines()
            shown = lines if args.verbose else lines[-40:]
            print(f"---- {e.name} output{'' if args.verbose else ' (last 40 lines)'} ----")
            print("\n".join(shown))
            print(f"---- end {e.name} ----", flush=True)

    if rows:
        print_table(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "SKIP")}
    total = sum(r["seconds"] or 0 for r in rows)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"in {total:.1f}s")
    for name, why in skipped_named:
        print(f"--only {name} was requested but skipped ({why}); exit 3")
    for path in untouched:
        print(f"--touching {path} selects no entry; exit 3")

    code = 1 if counts["FAIL"] else (3 if skipped_named or untouched else 0)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"git_head": git_head(), "tree_dirty_at_start": dirty,
                       "write_detection": watch.enabled,
                       "python": sys.version.split()[0], "platform": sys.platform,
                       "argv": recorded_argv(sys.argv[1:] if argv is None else list(argv)),
                       "mode": {"ci": args.ci, "all": args.run_all, "allow": sorted(args.allow)},
                       "exit": code, "counts": counts, "total_seconds": round(total, 1),
                       "results": rows}, f, indent=1)
            f.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
