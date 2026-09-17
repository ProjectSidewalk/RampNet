"""Lint for the committed docs: the things a careful read does not catch.

Every check here walks the tracked Markdown and fails loudly with the file and
line named. None of them touches the network, a checkpoint, or a GPU; the one
external dependency is ``git ls-files``, which needs a checkout and ``git`` on
PATH (both true under CI's ``actions/checkout``).
"""
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]


def _tracked_docs():
    """Tracked .md paths. git ls-files, not a walk: the checkout can contain
    nested worktrees and virtualenvs that a walk would descend into."""
    listing = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "*.md"],
                             capture_output=True, text=True, check=True)
    return [REPO / p for p in listing.stdout.split("\0") if p]


# --------------------------------------------------------------------------- #
# Collapsed line continuations in shell snippets
# --------------------------------------------------------------------------- #
def _blank_quoted(line):
    """Replace the contents of "..." and '...' spans with x, keeping the width.

    Quoted text is data, not command structure: a run of spaces or a ``#``
    inside it says nothing about a lost continuation, and left alone the first
    produces a false positive and the second hides everything after it."""
    out = []
    quote = None
    for ch in line:
        if quote is None:
            out.append(ch)
            if ch in "\"'":
                quote = ch
        else:
            out.append(ch if ch == quote else "x")
            if ch == quote:
                quote = None
    return "".join(out)


def _collapsed_continuations(text):
    """Yield (line_no, line) for every shell line that looks like a lost backslash.

    Three spaces is the threshold because these runbooks indent continuations by
    four, so a swallowed trailing backslash leaves at least that indent inside
    one line; a flush-left or two-space continuation would collapse below three
    and is not used here.
    """
    in_shell = False
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            # lstrip, so a fence nested in a list item is entered and left.
            in_shell = stripped.startswith("```bash") or stripped.startswith("```sh")
            continue
        if not in_shell or stripped.startswith("#"):
            continue
        scan = _blank_quoted(line)
        # An aligned trailing comment legitimately uses run-on spaces.
        scan = scan.split(" #", 1)[0].rstrip()
        # A line that still HAS its continuation is correct by definition,
        # whatever spacing it uses to align its arguments.
        if scan.endswith("\\"):
            continue
        # Three or more spaces mid-command is what a swallowed trailing
        # backslash plus newline leaves behind.
        if re.search(r"\S {3,}\S", scan):
            yield n, line


def test_the_collapsed_continuation_scanner_catches_the_shape_it_claims():
    """The scanner is the guard; these fixtures are the guard on the guard.

    Planting a defect in a committed doc is the only other way to check that it
    still catches the class, and that is not a thing to leave lying around."""
    collapsed = "```bash\npython x.py --a b     --c d\n```\n"
    assert list(_collapsed_continuations(collapsed)) == [
        (2, "python x.py --a b     --c d")]

    aligned_comment = "```bash\npython x.py --a b        # what b is for\n```\n"
    assert list(_collapsed_continuations(aligned_comment)) == []

    kept_continuation = "```bash\npython x.py --a b   \\\n    --c d\n```\n"
    assert list(_collapsed_continuations(kept_continuation)) == []


def test_no_runbook_snippet_has_a_collapsed_line_continuation():
    """A shell snippet whose trailing backslash was lost still *looks* fine -- the
    wrap becomes a run of spaces inside one long line -- and it still runs, so
    nothing catches it. But the exact-commands-in-order rule is what these blocks
    exist to satisfy, and a reader copying the wrapped form gets a broken command.

    Two of them shipped in #129 (for #126) because a patch script ate the backslashes.

    Every tracked .md is walked, not the four runbooks the two defects happened
    to be in: a collapsed snippet in README.md would be just as invisible.
    """
    hits = []
    for path in _tracked_docs():
        rel = path.relative_to(REPO).as_posix()
        hits += [f"{rel}:{n} {line.strip()!r}"
                 for n, line in _collapsed_continuations(path.read_text("utf-8"))]
    assert hits == [], (
        "these lines look like collapsed line continuations:\n" + "\n".join(hits))


def test_no_doc_still_hardcodes_the_old_roster_count():
    """These exact phrases were the drift. Catch them coming back.

    Whitespace is collapsed first, deliberately: every one of these was wrapped
    across a line break in the prose, so a naive substring check finds none of them
    and passes while the docs are still wrong.
    """
    # Each entry is a phrase that was actually in the docs and wrong. Two properties
    # matter and neither is obvious:
    #  * No entry may be a prefix of another -- "all 8" and "all 8 model groups"
    #    both shipped, and the shorter can never fail independently, so the longer
    #    reads as coverage it does not add.
    #  * The count has to be bound to the roster, or the guard fires on perfectly
    #    good prose. "all 8" alone would reject a future "all 8 splits"; the splits
    #    are a different axis and there are ten of them.
    stale = (r"all 8 (?:model|challenger|zero-shot)", r"all 8\b(?! splits| cities)",
             r"8-model roster", r"seven-model roster", r"8 model groups")
    docs = ("model_comparison.md", "replication.md", "curb_ramp_data_sourcing.md")
    for name in docs:
        text = re.sub(r"\s+", " ", (REPO / "docs" / name).read_text("utf-8"))
        for phrase in stale:
            hit = re.search(phrase, text)
            assert hit is None, f"docs/{name} still says {hit.group(0)!r}"
    # The analysis README carries per-model prose and this PR edits it, so it is in
    # scope for the same rot even though it is not under docs/.
    readme = REPO / "scripts" / "analysis" / "README.md"
    if readme.exists():
        text = re.sub(r"\s+", " ", readme.read_text("utf-8"))
        for phrase in stale:
            hit = re.search(phrase, text)
            assert hit is None, f"scripts/analysis/README.md still says {hit.group(0)!r}"
