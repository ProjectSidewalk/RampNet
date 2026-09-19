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
# Fence tags that mark a block as shell. Untyped fences are deliberately NOT
# scanned: in this tree they hold tree listings, program output and formulas,
# whose aligned columns are exactly the shape the scanner rejects. A shell
# snippet has to say it is one to be guarded.
_SHELL_FENCES = ("bash", "sh", "shell", "zsh")


def _blank_quoted(text):
    """Replace the contents of "..." and '...' spans with x, keeping the width.

    Quoted text is data, not command structure: a run of spaces or a ``#``
    inside it says nothing about a lost continuation, and left alone the first
    produces a false positive and the second hides everything after it.

    Takes a whole block, not a line, so a string that spans lines stays masked
    on every line it covers. Follows bash's rules: outside quotes and inside
    double quotes a backslash protects the next character, so ``\\"`` neither
    opens nor closes a span; inside single quotes a backslash is literal; and a
    ``#`` that starts a word outside quotes begins a comment, which is copied
    through untouched to the end of its line -- an apostrophe in ``# don't``
    must not open a span that masks every line after it. Newlines are always
    kept so the caller can still split on them.
    """
    out = []
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is None:
            if ch == "#" and (i == 0 or text[i - 1] in " \t\n"):
                end = text.find("\n", i)
                end = len(text) if end < 0 else end
                out.append(text[i:end])
                i = end
                continue
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch in "\"'":
                quote = ch
        elif ch == quote:
            out.append(ch)
            quote = None
        elif ch == "\n":
            out.append(ch)
        elif ch == "\\" and quote == '"' and i + 1 < len(text):
            out.append("x")
            out.append("\n" if text[i + 1] == "\n" else "x")
            i += 2
            continue
        else:
            out.append("x")
        i += 1
    return "".join(out)


def _shell_blocks(text):
    """Yield (first_line_no, [lines]) for every fenced block tagged as shell.

    ``lstrip`` before the fence test, so a fence nested in a list item is
    entered and left. A fence inside a blockquote (``> ```bash``) is not
    recognised and its block is not scanned."""
    start, block = None, []
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if start is not None:
                yield start, block
                start, block = None, []
                continue
            tag = stripped[3:].strip().split(None, 1)
            if tag and tag[0] in _SHELL_FENCES:
                start = n + 1
            continue
        if start is not None:
            block.append(line)


def _collapsed_continuations(text):
    """Yield (line_no, line) for every shell line that looks like a lost backslash.

    Two shapes are reported. A run of three or more spaces mid-command: a
    swallowed ``␠\\`` + newline leaves one space plus the next line's indent, and
    every continuation in these docs is indented by at least two, so the run is
    at least three. (The two flush-left ``VAR=x \\`` env-prefix continuations in
    docs/tillicum.md would collapse to a single space and are not covered; the
    run-on form executes identically, so nothing is lost there.) And a
    backslash followed by trailing whitespace: bash reads that as an escaped
    space, not a continuation, so the next line runs as a separate command.

    A line that still ends in ``\\`` is scanned up to the backslash, not skipped:
    losing the first of two continuations leaves ``a     b \\`` and that is a
    collapse too. An aligned trailing comment (``cmd      # why``) legitimately
    uses run-on spaces and is cut off before the scan; so is anything quoted.
    """
    for start, lines in _shell_blocks(text):
        masked = _blank_quoted("\n".join(lines)).split("\n")
        for offset, (line, scan) in enumerate(zip(lines, masked)):
            n = start + offset
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"\\[ \t]+$", scan):
                yield n, line
                continue
            # A blank of either kind starts a comment, as bash reads it.
            scan = re.split(r"[ \t]#", scan, 1)[0].rstrip()
            if scan.endswith("\\"):
                scan = scan[:-1].rstrip()
            if re.search(r"\S {3,}\S", scan):
                yield n, line


def test_the_collapsed_continuation_scanner_catches_the_shape_it_claims():
    """The scanner is the guard; these fixtures are the guard on the guard.

    Planting a defect in a committed doc is the only other way to check that it
    still catches the class, and that is not a thing to leave lying around."""
    def hits(block):
        return list(_collapsed_continuations(block))

    collapsed = "```bash\npython x.py --a b     --c d\n```\n"
    assert hits(collapsed) == [(2, "python x.py --a b     --c d")]

    aligned_comment = "```bash\npython x.py --a b        # what b is for\n```\n"
    assert hits(aligned_comment) == []

    kept_continuation = "```bash\npython x.py --a b   \\\n    --c d\n```\n"
    assert hits(kept_continuation) == []

    # A two-space indent still collapses to three spaces and is caught.
    two_space_indent = "```bash\npython x.py --a b   --c d\n```\n"
    assert hits(two_space_indent) == [(2, "python x.py --a b   --c d")]

    # Only a fence tagged as shell is entered; untyped and other-language
    # fences are not, whatever they contain.
    body = "python x.py --a b     --c d\n"
    for tag in _SHELL_FENCES:
        assert hits(f"```{tag}\n{body}```\n") == [(2, body.rstrip("\n"))], tag
    for tag in ("", "text", "python", "powershell"):
        assert hits(f"```{tag}\n{body}```\n") == [], tag

    # Backslash then trailing whitespace is an escaped space, not a
    # continuation: the next line runs as its own command.
    escaped_space = "```bash\npython x.py --a b \\ \n    --c d\n```\n"
    assert hits(escaped_space) == [(2, "python x.py --a b \\ ")]

    # A line that keeps its LAST continuation can still have lost an earlier one.
    collapse_before_backslash = (
        "```bash\npython tool.py --foo bar     --baz qux \\\n    --c d\n```\n")
    assert hits(collapse_before_backslash) == [
        (2, "python tool.py --foo bar     --baz qux \\")]

    # An escaped quote does not end the quoted span, so the spaces after it
    # are still data.
    escaped_quote = '```bash\necho "Starting \\"   Phase 2\\""\n```\n'
    assert hits(escaped_quote) == []

    # A quoted string that spans lines is data on every line it covers.
    multi_line_string = (
        "```bash\ncurl -d '{\n    \"key\":   \"value\",\n}' https://x\n```\n")
    assert hits(multi_line_string) == []

    # An apostrophe in a comment -- full-line or trailing -- is not a quote,
    # and must not mask the lines after it.
    comment_apostrophe = (
        "```bash\n# don't\ncmd     --x\ncmd --a   # it's fine\ncmd2     --y\n```\n")
    assert hits(comment_apostrophe) == [(3, "cmd     --x"), (5, "cmd2     --y")]

    # A tab before the # starts a comment just as a space does.
    tab_comment = "```bash\ncmd --a b\t#   why this\n```\n"
    assert hits(tab_comment) == []

    # Lines are numbered within the file, not within the block.
    later_block = "text\n\n```python\nx = 1\n```\n\n```bash\ncmd     --x\n```\n"
    assert hits(later_block) == [(8, "cmd     --x")]


def test_blank_quoted_keeps_width_and_newlines():
    """Line numbers and column positions are read off the masked text, so the
    mask must be a one-for-one substitution."""
    src = "a \"b c\" 'd\ne' \\\" f \"g\\\"h\"\n"
    out = _blank_quoted(src)
    assert len(out) == len(src)
    assert [i for i, c in enumerate(src) if c == "\n"] == \
        [i for i, c in enumerate(out) if c == "\n"]
    assert out == "a \"xxx\" 'x\nx' \\\" f \"xxxx\"\n"


def test_no_runbook_snippet_has_a_collapsed_line_continuation():
    """A shell snippet whose trailing backslash was lost still *looks* fine -- the
    wrap becomes a run of spaces inside one long line -- and it still runs, so
    nothing catches it. But the exact-commands-in-order rule is what these blocks
    exist to satisfy, and a reader copying the wrapped form gets a broken command.

    Two of them shipped in #129 (for #126) because a patch script ate the
    backslashes, and eight more in #152, in the Laurens replication runbook.

    Every tracked .md is walked, not the four runbooks the first two defects
    happened to be in; within each file only fenced blocks tagged as shell are
    scanned (see ``_SHELL_FENCES``), so a shell snippet in an untyped fence is
    not covered until it is tagged.
    """
    hits = []
    for path in _tracked_docs():
        rel = path.relative_to(REPO).as_posix()
        hits += [f"{rel}:{n} {line.strip()!r}"
                 for n, line in _collapsed_continuations(path.read_text("utf-8"))]
    assert hits == [], (
        "these lines look like collapsed line continuations:\n" + "\n".join(hits))


# --------------------------------------------------------------------------- #
# Hand-written numbers that the registry now owns
# --------------------------------------------------------------------------- #
# Each entry is a phrase that was actually in the docs and wrong. Two properties
# matter and neither is obvious:
#  * No entry may be a prefix of another -- "all 8" and "all 8 model groups"
#    both shipped, and the shorter can never fail independently, so the longer
#    reads as coverage it does not add.
#  * The count has to be bound to the roster, or the guard fires on perfectly
#    good prose. The second pattern is bound by exclusion: it rejects every
#    "all 8" EXCEPT the other eights these docs count -- splits, cities, epochs,
#    seeds, checkpoints -- so a new non-roster axis has to be added here before
#    "all 8 <axis>" will pass. That is deliberate: a false hit costs one edit to
#    this tuple; a miss is a stale number in a published doc.
# The word boundary after the 8 is load-bearing: without it "all 83 far-field
# silent misses" is a hit. The copy this was moved from carried a literal
# backspace byte there instead of \b (a heredoc ate the backslash), so the
# second pattern could never match anything and the guard was inert;
# test_stale_roster_patterns_match_what_they_claim pins it against reverting.
STALE_ROSTER_PHRASES = (
    r"all 8 (?:model|challenger|zero-shot)",
    r"all 8\b(?! splits| cities| epochs| seeds| checkpoints)",
    r"8-model roster", r"seven-model roster", r"8 model groups")


def test_stale_roster_patterns_match_what_they_claim():
    """The guard on the guard. The second pattern shipped inert once -- a
    literal 0x08 where \b should be -- and every test stayed green, because
    nothing asserted what the patterns match."""
    for pattern in STALE_ROSTER_PHRASES:
        assert chr(8) not in pattern, "a backspace byte where a word boundary was meant"
    all_8 = STALE_ROSTER_PHRASES[1]
    for hit in ("all 8 of them", "all 8 models", "all 8, and", "all 8 epoch groups"):
        assert re.search(all_8, hit), hit
    for miss in ("all 83 far-field silent misses", "all 8 splits", "all 8 cities",
                 "all 8 epochs", "all 8 seeds", "all 8 checkpoints"):
        assert not re.search(all_8, miss), miss


def test_no_doc_still_hardcodes_the_old_roster_count():
    """These exact phrases were the drift. Catch them coming back.

    Whitespace is collapsed first, deliberately: every one of these was wrapped
    across a line break in the prose, so a naive substring check finds none of them
    and passes while the docs are still wrong.
    """
    stale = STALE_ROSTER_PHRASES
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
