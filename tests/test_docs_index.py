"""docs/README.md is the index of every document in docs/. These checks keep it from going stale.

Sits beside tests/test_docs.py (which lints the docs' shell snippets and roster counts); nothing
there checks links, so nothing here duplicates it. Offline and filesystem-only: no network, no
git, no `gh`. Issue numbers are checked for shape only; whether each one is real and on topic was
verified with `gh issue view` when the row was written, and cannot be re-checked without network.
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
KINDS = {"result", "negative result", "protocol/rubric", "plan/proposal", "ledger", "how-to",
         "report/index"}
# Highest issue/PR number the index may cite. Generous on purpose: this only catches typos like
# an extra digit, not numbers that are merely new.
MAX_ISSUE = 2000

LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
SCRIPT = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh|slurm)")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


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


def _rows():
    """Yield (line_no, {column: cell}) for every row of every six-column index table."""
    header = None
    for n, line in enumerate(INDEX.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = _split_row(line)
        if cells == HEADER:
            header = cells
            continue
        if header is None or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        assert len(cells) == len(HEADER), f"docs/README.md:{n}: expected {len(HEADER)} cells, got {len(cells)}"
        yield n, dict(zip(HEADER, cells))


def _row_doc(row):
    links = LINK.findall(row["file"])
    assert len(links) == 1, f"file cell should hold exactly one link: {row['file']!r}"
    return links[0]


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


def test_kind_is_from_the_fixed_vocabulary():
    for n, row in _rows():
        assert row["kind"] in KINDS, f"docs/README.md:{n}: kind {row['kind']!r} is not one of {sorted(KINDS)}"


def test_status_is_from_the_fixed_vocabulary():
    for n, row in _rows():
        assert re.match(r"(final|superseded by|in progress|proposed)\b", row["status"]), (
            f"docs/README.md:{n}: status {row['status']!r}")


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
    """A hook may only quote numbers its document states. If the document is edited so a number
    no longer appears, the hook is stale and this says which."""
    for n, row in _rows():
        doc = DOCS / _row_doc(row)
        body = doc.read_text(encoding="utf-8")
        for num in NUMBER.findall(row["hook"]):
            if len(num.replace(",", "")) < 2:
                continue  # a lone digit ("3 GSV splits") matches almost any document
            assert num in body, (
                f"docs/README.md:{n}: hook for {doc.name} quotes {num}, which {doc.name} no longer contains")
