#!/usr/bin/env python
"""Verify every committed sha256sum-style manifest against the files on disk.

A committed ``SHA256SUMS`` (or ``*.sha256``) is a promise that the listed bytes are the ones
an analysis read. Some manifests have an owning script that checks them
(``da3_calibration_101.py --check``, ``input_res_sweep_25.py sums``); others are only ever
checked by hand with ``sha256sum -c`` (``docs/data/rampnet1_stage1_run/``,
``docs/data/rampnet1_stage2_run/``, ``stage_two/run_a_84_events/``,
``stage_two/cosine_rung_135_events/``, ``docs/data/seed_variance_51_135/``). This checks all of
them, so a manifest nobody owns cannot rot silently. It is the ``manifests_sha256`` entry of
``scripts/check_all.py``.

Manifests are found with ``git ls-files`` (``**/SHA256SUMS*`` and ``**/*.sha256``), so a new
one is covered without editing this file. Each line is ``<64 hex>  <name>`` or
``<64 hex> *<name>`` (GNU ``sha256sum`` text / binary mode). ``<name>`` resolves against the
manifest's directory first, then the repo root (``da3_calibration_101``'s manifest lists
repo-relative paths).

* A manifest whose listed files are **all absent** is skipped and counted, never failed:
  several manifests pin files that live only on the cluster (``.pth`` checkpoints, the crops
  tarball) and are recorded so a copy can be proven identical when it turns up.
* A manifest that is **partly present** (some listed files exist, some do not) fails: that is
  a committed file deleted or renamed out from under its pin.
* A file that is present and **differs** fails the run.
* If the raw bytes differ but the file contains CRLF and its LF-normalized bytes match, it
  counts as a match (reported as ``eol``): a ``core.autocrlf=true`` checkout must not fail a
  pin when no content changed. This repo's own checkouts are LF (``core.autocrlf=input``).
* If no listed file at all is present, the run fails: a check that compared nothing has not
  passed.

Usage::

    python scripts/verify_sha256sums.py            # every manifest; exit 1 on any mismatch
    python scripts/verify_sha256sums.py --list     # the manifests found, nothing hashed

Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = ("SHA256SUMS*", "*/SHA256SUMS*", "*.sha256")
LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


def manifests(repo=None):
    repo = repo or REPO
    p = subprocess.run(["git", "ls-files", "-z", "--", *PATTERNS], cwd=repo,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    names = [n for n in p.stdout.decode("utf-8").split("\0") if n]
    return sorted({n for n in names if os.path.basename(n).startswith("SHA256SUMS")
                   or n.endswith(".sha256")})


def parse(path):
    """[(digest, name)] from a manifest; a malformed line is an error, not a skip."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                raise SystemExit(f"{path}:{i}: not a sha256sum line: {line!r}")
            out.append((m.group(1).lower(), m.group(2)))
    return out


def resolve(manifest, name, repo=None):
    repo = repo or REPO
    if os.path.isabs(name) or name.startswith("/"):
        return name if os.path.isfile(name) else None
    for base in (os.path.dirname(os.path.join(repo, manifest)), repo):
        p = os.path.normpath(os.path.join(base, name))
        if os.path.isfile(p):
            return p
    return None


def digests(path):
    with open(path, "rb") as fh:
        data = fh.read()
    raw = hashlib.sha256(data).hexdigest()
    lf = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() if b"\r\n" in data else raw
    return raw, lf


def verify(repo=None, only=None):
    repo = repo or REPO
    """Returns (rows, problems): one row per manifest, and every mismatch."""
    rows, problems = [], []
    for man in only or manifests(repo):
        ok = eol = absent = bad = 0
        for want, name in parse(os.path.join(repo, man)):
            p = resolve(man, name, repo)
            if p is None:
                absent += 1
                continue
            raw, lf = digests(p)
            if raw == want:
                ok += 1
            elif lf == want:
                eol += 1
            else:
                bad += 1
                problems.append(f"{man}: {name}: sha256 {raw} != pinned {want}")
        if absent and (ok + eol + bad):
            # some listed files exist and some do not: a committed file was deleted or renamed
            problems.append(f"{man}: partly present -- {absent} of {absent + ok + eol + bad} "
                            "listed files are absent while the rest exist")
        rows.append({"manifest": man, "ok": ok, "eol": eol, "absent": absent, "bad": bad})
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print the manifests found and exit")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.list:
        for m in manifests(REPO):
            print(m)
        return 0
    rows, problems = verify(REPO)
    w = max(len(r["manifest"]) for r in rows)
    print(f"{'manifest':<{w}}  {'ok':>4}  {'eol':>4}  {'absent':>6}  {'bad':>4}")
    for r in rows:
        print(f"{r['manifest']:<{w}}  {r['ok']:>4}  {r['eol']:>4}  {r['absent']:>6}  {r['bad']:>4}")
    tot = {k: sum(r[k] for r in rows) for k in ("ok", "eol", "absent", "bad")}
    print(f"\n{len(rows)} manifests: {tot['ok']} match, {tot['eol']} match after CRLF->LF, "
          f"{tot['absent']} listed files absent, {tot['bad']} MISMATCH")
    for p in problems:
        print("FAIL " + p)
    if tot["ok"] + tot["eol"] == 0:
        print("NOTHING VERIFIED: no listed file is present")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
