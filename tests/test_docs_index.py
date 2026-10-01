"""docs/README.md is the index of every document in docs/. These checks keep it from going stale.

Sits beside tests/test_docs.py (which lints the docs' shell snippets and roster counts); nothing
there checks links, so nothing here duplicates it. Offline and filesystem-only: no network, no
git, no `gh`. Issue numbers are checked for shape only; whether each one is real and on topic was
verified with `gh issue view` when the row was written, and cannot be re-checked without network.

What the hook-number check guarantees, and what it does not
-----------------------------------------------------------
``test_every_number_in_a_hook_appears_in_its_document`` reads each number a hook quotes and
requires the same number to appear in the document the row links, under these rules:

* **Token boundaries.** The number must stand alone in the document: not preceded by a digit,
  a decimal point or a thousands comma, and not followed by a digit. ``0.16`` does not match
  ``0.160``, and ``0.018`` does not match ``10.0180``.
* **Sign.** If the hook writes a sign (``+``, ``±``, or a minus as ``-`` or ``−``), the document
  must carry the same sign; the two minus glyphs are interchangeable. An unsigned hook number
  matches either.
* **Skipped tokens.** ISO dates (``2026-09-03``) are removed before tokenising, and ``#n``
  issue references are not numbers. A plain integer
  below 100 (``3 GSV splits``, ``n=9``, ``epochs 2–6``) is skipped unless the hook writes it with
  a decimal point or a unit directly after it (``%``, ``°``, ``×``, ``m``, ``px``, ``pt``/``pts``,
  ``h``/``hours``), because a bare small integer matches almost any document.
* ``data:`` URIs are stripped from the document first (the HTML report embeds base64 images).

The limit: this guards against a hook number **disappearing** from its document. A headline
number often appears several times in its document, so an edit to one occurrence while the
others stay will not trip it. It is a staleness alarm, not a proof that the hook is the
document's current headline; the accuracy of a hook is still a reviewer's job.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
INDEX = DOCS / "README.md"

# The only document in docs/ that the index does not list: the index itself.
NOT_INDEXED = {"README.md"}
# Subdirectories of docs/ that hold supporting files, not documents. The index mentions each in
# one line; their contents are not rows.
SUPPORT_DIRS = {"assets", "data", "figures"}

HEADER = ["file", "issue(s)", "kind", "status", "hook", "reproduce"]
ARRIVING_HEADER = ["file", "PR", "what it is"]
KINDS = {"result", "negative result", "protocol/rubric", "plan/proposal", "ledger", "how-to",
         "report/index"}
# Highest issue/PR number the index may cite. Generous on purpose: this only catches typos like
# an extra digit, not numbers that are merely new.
MAX_ISSUE = 2000

LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
SCRIPT = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh|slurm)")
# A script in the reproduce column, in backticks, and the parenthesised label right after it.
LABELLED_SCRIPT = re.compile(r"`(scripts/[^`\s]*[^`\s/])`(?:\s*\(([^)]*)\))?")

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
HOOK_NUMBER = re.compile(
    r"(?<![\w.,#])(?P<sign>[+±\-−])?(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>%|°|×|\s?(?:m|px|pts?|h|hours)\b)?")
MINUS = "-−"

# A script "has a check" if it declares a --check flag or a `check` subcommand.
HAS_CHECK = re.compile(r"""add_argument\(\s*["']--check["']|add_parser\(\s*["']check["']""")
# ... and "has a verify mode" if it declares a `verify` subcommand/choice or a --verify flag.
HAS_VERIFY = re.compile(r"""["'](?:--)?verify["']""")


def _documents():
    """Every .md and .html under docs/, outside the supporting directories, relative to docs/."""
    out = set()
    for p in DOCS.rglob("*"):
        rel = p.relative_to(DOCS)
        if p.suffix not in (".md", ".html") or not p.is_file():
            continue
        if rel.parts[0] in SUPPORT_DIRS:
            continue
        out.add(rel.as_posix())
    return out


def _split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(header_cells):
    """Yield (line_no, {column: cell}) for every row of every table with this header."""
    header = None
    for n, line in enumerate(INDEX.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = _split_row(line)
        if cells == header_cells:
            header = cells
            continue
        if header is None or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        assert len(cells) == len(header), (
            f"docs/README.md:{n}: expected {len(header)} cells, got {len(cells)}")
        yield n, dict(zip(header, cells))


def _rows():
    return _table(HEADER)


def _row_doc(row):
    links = LINK.findall(row["file"])
    assert len(links) == 1, f"file cell should hold exactly one link: {row['file']!r}"
    return links[0]


def _doc_text(path):
    text = path.read_text(encoding="utf-8")
    return re.sub(r"data:[^\s\"')]+", "", text)


def _hook_numbers(hook):
    """Yield (sign or '', number) for every hook number the check applies to (rules above)."""
    for m in HOOK_NUMBER.finditer(ISO_DATE.sub(" ", hook)):
        num, unit = m.group("num"), m.group("unit")
        plain_small = ("." not in num and "," not in num and int(num) < 100)
        if plain_small and not unit:
            continue
        yield (m.group("sign") or ""), num


def _number_in(body, sign, num):
    if sign in MINUS:
        sign_re = "[" + MINUS + "]" if sign else ""
    else:
        sign_re = re.escape(sign)
    if sign:
        pattern = sign_re + re.escape(num) + r"(?!\d)"
    else:
        pattern = r"(?<![\d.])(?<!\d,)" + re.escape(num) + r"(?!\d)"
    return re.search(pattern, body) is not None


def _labelled_scripts(cell):
    """Yield (script, label or None) for each script in a reproduce cell. A script named inside
    another script's label (``checked via `scripts/...` ``) is part of that label."""
    for m in LABELLED_SCRIPT.finditer(cell):
        yield m.group(1), m.group(2)


def test_the_index_parses_into_rows():
    rows = list(_rows())
    assert len(rows) >= 40, f"only {len(rows)} index rows parsed; the table format may have changed"


def test_every_document_in_docs_has_a_row():
    indexed = {_row_doc(row) for _, row in _rows()}
    missing = sorted(_documents() - indexed - NOT_INDEXED)
    assert not missing, (
        "documents in docs/ with no row in docs/README.md (add one to the right group): "
        + ", ".join(missing))


def test_no_document_has_two_rows():
    seen = {}
    for n, row in _rows():
        doc = _row_doc(row)
        assert doc not in seen, f"docs/README.md:{n}: {doc} already has a row at line {seen[doc]}"
        seen[doc] = n


def test_every_link_in_the_index_resolves():
    text = INDEX.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (DOCS / target).exists(), f"docs/README.md links {target}, which does not exist"


def test_every_script_in_the_reproduce_column_exists():
    for n, row in _rows():
        for path in SCRIPT.findall(row["reproduce"]):
            assert (REPO / path).is_file(), f"docs/README.md:{n}: {path} does not exist"


def test_every_check_label_matches_the_script():
    """Each script in the reproduce column carries its own label, and the label is true:
    "(check…)" means the script declares --check or a `check` subcommand, "(no check)" means it
    declares no check and no verify mode, "(verify)" means it has a `verify` subcommand or a
    --verify flag, and "(checked via `X`…)" means
    script X has a check."""
    seen = 0
    for n, row in _rows():
        for script, label in _labelled_scripts(row["reproduce"]):
            seen += 1
            where = f"docs/README.md:{n}: {script}"
            assert label is not None, f"{where} has no (check)/(no check) label"
            src = (REPO / script).read_text(encoding="utf-8", errors="replace")
            label = label.strip()
            if label == "no check":
                assert not HAS_CHECK.search(src), f"{where} is labelled 'no check' but has a check mode"
                assert not HAS_VERIFY.search(src), f"{where} is labelled 'no check' but has a verify mode; label it (verify)"
            elif label.startswith("checked via"):
                via = SCRIPT.findall(label)
                assert via, f"{where}: 'checked via' must name the checking script by path"
                for v in via:
                    vsrc = (REPO / v).read_text(encoding="utf-8", errors="replace")
                    assert HAS_CHECK.search(vsrc), f"{where}: {v} has no check mode"
            elif label == "check" or label.startswith("check:"):
                assert HAS_CHECK.search(src), f"{where} is labelled '{label}' but has no check mode"
            elif label == "verify":
                assert HAS_VERIFY.search(src), f"{where} is labelled 'verify' but has no verify mode"
            else:
                raise AssertionError(f"{where}: unknown label {label!r}")
    assert seen >= 40, f"only {seen} labelled scripts parsed; the reproduce format may have changed"


def test_kind_is_from_the_fixed_vocabulary():
    for n, row in _rows():
        assert row["kind"] in KINDS, f"docs/README.md:{n}: kind {row['kind']!r} is not one of {sorted(KINDS)}"


def test_status_is_from_the_fixed_vocabulary():
    for n, row in _rows():
        assert re.match(r"(final|in progress|proposed)\b", row["status"]), (
            f"docs/README.md:{n}: status {row['status']!r}")
        if "superseded" in row["status"]:
            assert row["status"].startswith("final; part superseded by"), (
                f"docs/README.md:{n}: write a partial supersession as 'final; part superseded by <file>'")


def test_issue_numbers_are_plausible():
    text = INDEX.read_text(encoding="utf-8")
    for m in re.finditer(r"#(\d+)\b", text):
        n = int(m.group(1))
        assert 1 <= n <= MAX_ISSUE, f"docs/README.md cites #{m.group(1)}, not a plausible issue number"
    for n, row in _rows():
        cell = row["issue(s)"]
        assert cell == "—" or re.fullmatch(r"#\d+(, #\d+)*", cell), (
            f"docs/README.md:{n}: issue cell {cell!r} should be '#n, #m' or '—'")


def test_every_number_in_a_hook_appears_in_its_document():
    """See the module docstring for the matching rules and their limit."""
    for n, row in _rows():
        doc = DOCS / _row_doc(row)
        body = _doc_text(doc)
        for sign, num in _hook_numbers(row["hook"]):
            assert _number_in(body, sign, num), (
                f"docs/README.md:{n}: hook for {doc.name} quotes {sign}{num}, "
                f"which {doc.name} no longer contains as a standalone number")


def test_hook_number_rules():
    """The matcher is the guard; these fixtures are the guard on the guard."""
    assert list(_hook_numbers("ΔF1 −0.018 on 2026-09-03 over 3 splits, 90° and 7%, 1,422 of 12,150")) == [
        ("−", "0.018"), ("", "90"), ("", "7"), ("", "1,422"), ("", "12,150")]
    assert _number_in("ΔF1 -0.018 [x]", "−", "0.018")
    assert not _number_in("ΔF1 +0.018 [x]", "−", "0.018")
    assert not _number_in("value 10.0180 here", "", "0.018")
    assert not _number_in("residual 0.160 F1", "", "0.16")
    assert _number_in("residual 0.160 F1", "", "0.160")
    assert not _number_in("n 21,422 panos", "", "1,422")
    assert _number_in("(1,422 of 12,150)", "", "1,422")


def test_arriving_docs_are_not_yet_on_this_branch():
    """A doc listed under "Arriving on open PRs" must not exist yet. When its PR merges this
    fails, which is the prompt to move the line into a real row."""
    rows = list(_table(ARRIVING_HEADER))
    assert rows, "the 'Arriving on open PRs' table did not parse"
    for n, row in rows:
        paths = re.findall(r"`(docs/[^`]+)`", row["file"])
        assert len(paths) == 1, f"docs/README.md:{n}: arriving row should name one docs/ path"
        assert not (REPO / paths[0]).exists(), (
            f"docs/README.md:{n}: {paths[0]} is now on this branch; move it from 'Arriving on "
            "open PRs' into a row of the right group")
        assert re.fullmatch(r"#\d+", row["PR"]), f"docs/README.md:{n}: PR cell {row['PR']!r}"
