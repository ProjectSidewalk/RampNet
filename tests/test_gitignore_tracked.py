"""No tracked file may match a ``.gitignore`` pattern (#251).

A tracked file that ``.gitignore`` matches stays tracked, so nothing is lost on the day it
happens. Two things break quietly later: ``git add`` of a regenerated copy is refused
without an error a script would notice, and ``scripts/check_all.py``'s write detection
(``git ls-files --cached --others --exclude-standard``) is the only place that still sees the
file. The fix for a hit is a ``!path`` carve-out in ``.gitignore`` beside the bundle it
belongs to, not ``git add -f``.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120)


def test_no_tracked_file_is_gitignored():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout (e.g. an sdist)")
    p = _git("ls-files", "-z", "--cached", "--ignored", "--exclude-standard")
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    hits = sorted(f for f in p.stdout.decode("utf-8", "replace").split("\0") if f)
    assert not hits, ("tracked files matched by .gitignore; add a `!path` carve-out for each:\n  "
                      + "\n  ".join(hits))
