#!/usr/bin/env python
"""Render the Hugging Face dataset cards for `rampnet-dataset` and `rampnet-benchmark` (#150).

The cards are filled from two committed sources and nothing else:

* ``croissant/rampnet-dataset.json`` and ``croissant/rampnet-benchmark.json`` -- the
  hand-authored Croissant records (``docs/fair_metadata_150.md``), which are the record of truth
  for every count, split, file, revision, date, licence and creator the cards print;
* ``scripts/hf_package/cards_facts.json`` -- the few facts Croissant does not carry as a
  structured field. Each has a ``source``; one marked ``in_croissant`` must also appear, as
  printed, in that card's Croissant file, or rendering stops.

The benchmark card also reads the committed bundles (``benchmark/<split>/verdicts.json`` and
``benchmark/train_overlap.json``) through the same helpers ``scripts/export_benchmark.py`` uses,
restricted to the splits Croissant says are published, so its review-confidence table and
training-overlap listing are the ones the next export would print.

**Nothing is uploaded, and there is no flag that would.** The output is a pair of README files
for review; uploading is a separate decision (``docs/dataset_cards_150.md``, "Uploading later").

Usage::

    python scripts/export_dataset_cards.py                 # -> hf_export/cards/README.<repo>.md
    python scripts/export_dataset_cards.py --out DIR
    python scripts/export_dataset_cards.py --check         # render to a temp dir, diff against
                                                           # docs/data/hf_cards/; exit 1 on drift
    python scripts/export_dataset_cards.py --update        # rewrite docs/data/hf_cards/ (review
                                                           # the diff before committing)

Exit codes: 0 ok; 1 ``--check`` found a difference or a missing render; 2 a fact or template
could not be rendered.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from hf_export_common import size_category  # noqa: E402

PACKAGE = SCRIPTS / "hf_package"
CROISSANT = REPO / "croissant"
BENCHMARK = REPO / "benchmark"
FACTS = PACKAGE / "cards_facts.json"
COMMITTED = REPO / "docs" / "data" / "hf_cards"
DEFAULT_OUT = REPO / "hf_export" / "cards"

# card key -> (Croissant file, template, rendered file name)
CARDS = {
    "dataset": ("rampnet-dataset.json", "README.dataset_card.template.md",
                "README.rampnet-dataset.md"),
    "benchmark": ("rampnet-benchmark.json", "README.benchmark_card.template.md",
                  "README.rampnet-benchmark.md"),
}

# The section headings of the Hugging Face dataset card template, in order, with their Markdown
# level. Read from huggingface_hub's src/huggingface_hub/templates/datasetcard_template.md on
# 2026-11-05 (main); "[optional]" suffixes dropped, as filled cards drop them. The H1 is
# "Dataset Card for <pretty_name>". tests/test_dataset_cards_150.py requires every one, at this
# level and in this order, in both rendered cards.
HF_TEMPLATE_URL = ("https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/"
                   "templates/datasetcard_template.md")
HF_TEMPLATE_HEADINGS = (
    (1, "Dataset Card for"),
    (2, "Dataset Details"),
    (3, "Dataset Description"),
    (3, "Dataset Sources"),
    (2, "Uses"),
    (3, "Direct Use"),
    (3, "Out-of-Scope Use"),
    (2, "Dataset Structure"),
    (2, "Dataset Creation"),
    (3, "Curation Rationale"),
    (3, "Source Data"),
    (4, "Data Collection and Processing"),
    (4, "Who are the source data producers?"),
    (3, "Annotations"),
    (4, "Annotation process"),
    (4, "Who are the annotators?"),
    (4, "Personal and Sensitive Information"),
    (2, "Bias, Risks, and Limitations"),
    (3, "Recommendations"),
    (2, "Citation"),
    (2, "Glossary"),
    (2, "More Information"),
    (2, "Dataset Card Authors"),
    (2, "Dataset Card Contact"),
)

# Croissant licence URL -> the Hub's licence identifier
# (https://huggingface.co/docs/hub/repositories-licenses). Anything else is refused, not guessed.
HUB_LICENSES = {"https://spdx.org/licenses/MIT.html": "mit"}

PLACEHOLDER_RE = re.compile(r"\{[a-z_][a-z0-9_]*\}")
REVISION_RE = re.compile(r"revision ([0-9a-f]{40})", re.IGNORECASE)


class RenderError(Exception):
    """A fact, Croissant field or template value the cards cannot be rendered without."""


# --------------------------------------------------------------------------------- inputs

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_template(name):
    """The template text with LF line endings, whatever the checkout's core.autocrlf did."""
    return (PACKAGE / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def record_set(doc, rs_id):
    found = next((r for r in doc.get("recordSet", []) if r.get("@id") == rs_id), None)
    if found is None:
        raise RenderError("Croissant {!r} has no record set {!r}".format(doc.get("name"), rs_id))
    return found


def distribution(doc, dist_id):
    found = next((d for d in doc.get("distribution", []) if d.get("@id") == dist_id), None)
    if found is None:
        raise RenderError("Croissant {!r} has no distribution {!r}".format(doc.get("name"), dist_id))
    return found


def pinned_revision(doc):
    """The 40-hex Hub revision the ``repo`` description names (the same rule as the validator)."""
    found = {r.lower() for r in REVISION_RE.findall(distribution(doc, "repo").get("description", ""))}
    if len(found) != 1:
        raise RenderError("Croissant {!r}: expected one pinned revision, found {}".format(
            doc.get("name"), sorted(found)))
    return found.pop()


def repo_id(doc):
    prefix = "https://huggingface.co/datasets/"
    url = doc.get("url", "")
    if not url.startswith(prefix):
        raise RenderError("Croissant url {!r} is not a Hub dataset URL".format(url))
    return url[len(prefix):]


def hub_license(doc):
    lic = doc.get("license")
    if lic not in HUB_LICENSES:
        raise RenderError("no Hub licence identifier known for {!r}; add it to HUB_LICENSES".format(lic))
    return HUB_LICENSES[lic]


def manifest_rows(doc):
    return [(r["file_manifest/path"], r["file_manifest/bytes"])
            for r in record_set(doc, "file_manifest")["data"]]


def fmt(value):
    """How a fact is printed: integers with thousands separators, everything else verbatim."""
    if isinstance(value, bool):
        raise RenderError("boolean facts are not printable: {!r}".format(value))
    if isinstance(value, int):
        return "{:,}".format(value)
    return str(value)


def appears_bounded(value, text):
    """True if `value` occurs in `text` as a whole token: not inside a longer number, word or
    hash. "12" does not match "12,345", "123", "0.12", "2026-01-12" or a sha; it does match
    "12 other cities". The limit: a short value can still match an unrelated occurrence of the
    same token elsewhere in the file (#150 review)."""
    pattern = r"(?<![\w.,\-/])" + re.escape(value) + r"(?![\w]|[.,]\d)"
    return re.search(pattern, text) is not None


def load_facts(card, croissant_text, facts_path=FACTS):
    """{key: printed value} for one card, after checking each in_croissant fact is there."""
    facts = load_json(facts_path).get(card, {})
    printed = {}
    for key, entry in facts.items():
        if "value" not in entry or not str(entry.get("source", "")).strip():
            raise RenderError("fact {}.{} needs a value and a source".format(card, key))
        value = entry["value"]
        values = list(value.values()) if isinstance(value, dict) else [value]
        if entry.get("in_croissant"):
            for v in values:
                if not appears_bounded(fmt(v), croissant_text):
                    raise RenderError(
                        "fact {}.{} = {!r} is marked in_croissant but does not appear in the "
                        "Croissant file; fix the fact or the Croissant record".format(card, key, fmt(v)))
        printed[key] = value if isinstance(value, dict) else fmt(value)
    return printed


# --------------------------------------------------------------------------------- helpers

def apa_authors(creators):
    """'John S. O'Meara' -> "O'Meara, J. S.", joined APA style."""
    names = []
    for person in creators:
        parts = person["name"].split()
        initials = " ".join(p[0] + "." for p in parts[:-1])
        names.append("{}, {}".format(parts[-1], initials))
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def bibtex_field(bibtex, field):
    m = re.search(r"^\s*" + field + r"\s*=\s*\{+(.*?)\}+,?\s*$", bibtex, re.M)
    if not m:
        raise RenderError("citeAs has no {} field".format(field))
    return m.group(1)


def md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def field_type(field):
    names = {"sc:ImageObject": "image", "sc:Text": "string", "cr:Int64": "int64",
             "cr:Int32": "int32", "cr:Float32": "float32", "cr:Float64": "float64",
             "sc:Boolean": "bool"}
    dtype = field.get("dataType")
    base = names.get(dtype, str(dtype))
    if field.get("isArray"):
        return "{}, shape ({})".format(base, field.get("arrayShape", "-1").replace("-1", "n"))
    return base


def fields_table(record):
    """Markdown table of a record set's fields, from the Croissant field descriptions."""
    lines = ["| column | type | meaning |", "| :--- | :--- | :--- |"]
    for f in record["field"]:
        if f.get("source", {}).get("extract", {}).get("fileProperty"):
            continue                           # parsed from the file path, not a Parquet column
        lines.append("| `{}` | {} | {} |".format(f["name"], field_type(f),
                                                 md_cell(f.get("description", ""))))
    return "\n".join(lines)


# --------------------------------------------------------------------------------- contexts

def dataset_context(doc, facts):
    """Template values for README.dataset_card.template.md."""
    splits = dict((r["splits/name"], r["splits/num_rows"]) for r in record_set(doc, "splits")["data"])
    if sorted(splits) != ["test", "train", "val"]:
        raise RenderError("rampnet-dataset splits are {}, expected train/val/test".format(sorted(splits)))
    total = sum(splits.values())
    per_split_fake = int(facts["placeholder_rows_per_split"])
    usable = dict((k, v - per_split_fake) for k, v in splits.items())
    usable_total = sum(usable.values())
    # The live card's total; the Croissant description quotes it. If the placeholder rows are
    # ever removed on the Hub, this arithmetic stops matching and the card must be revisited.
    if "{:,} panoramas".format(usable_total) not in doc["description"]:
        raise RenderError("Hub rows minus placeholder rows ({:,}) is not the card total the "
                          "Croissant description quotes".format(usable_total))
    files = manifest_rows(doc)
    by_dir = {}
    for path, _ in files:
        by_dir[path.split("/")[0]] = by_dir.get(path.split("/")[0], 0) + 1
    if len(set(by_dir.values())) != 1:
        raise RenderError("shards per split differ: {}".format(by_dir))

    cities = []
    for place in doc["spatialCoverage"]:
        name = place["name"]
        if name not in facts["gov_records_by_city"]:
            raise RenderError("no government record count for {!r} in cards_facts.json".format(name))
        cities.append("| {} | {} | `{}` |".format(
            name, fmt(facts["gov_records_by_city"][name]), place["geo"]["box"]))
    city_table = "\n".join(["| city | government curb ramp records (committed) | extent of those records "
                            "(S W N E, WGS 84) |", "| :--- | ---: | :--- |"] + cities)

    cite = doc["citeAs"]
    rev = pinned_revision(doc)
    ctx = dict(facts)
    ctx.pop("gov_records_by_city")
    ctx.update(
        license=hub_license(doc),
        pretty_name=doc["alternateName"][1],
        size_category=size_category(total),
        repo_id=repo_id(doc),
        repo_url=doc["url"],
        code_url=doc["isBasedOn"][0]["codeRepository"],
        revision=rev,
        revision_short=rev[:7],
        version=doc["version"],
        date_created=doc["dateCreated"],
        date_published=doc["datePublished"],
        date_modified=doc["dateModified"],
        creators=", ".join(p["name"] for p in doc["creator"]),
        creators_bibtex=" and ".join(p["name"] for p in doc["creator"]),
        publisher=doc["publisher"]["name"],
        rows_train=fmt(splits["train"]), rows_val=fmt(splits["val"]),
        rows_test=fmt(splits["test"]), rows_total=fmt(total),
        usable_train=fmt(usable["train"]), usable_val=fmt(usable["val"]),
        usable_test=fmt(usable["test"]), usable_total=fmt(usable_total),
        pct_train="{:.0f}%".format(100 * usable["train"] / usable_total),
        pct_val="{:.0f}%".format(100 * usable["val"] / usable_total),
        pct_test="{:.0f}%".format(100 * usable["test"] / usable_total),
        placeholder_total=fmt(per_split_fake * len(splits)),
        n_shards=fmt(len(files)),
        shards_per_split=fmt(next(iter(by_dir.values()))),
        total_gb="{:.2f}".format(sum(b for _, b in files) / 1e9),
        fields_table=fields_table(record_set(doc, "panoramas")),
        city_table=city_table,
        cite_as=cite,
        apa_authors=apa_authors(doc["creator"]),
        paper_title=bibtex_field(cite, "title"),
        paper_venue=bibtex_field(cite, "booktitle"),
        paper_doi=bibtex_field(cite, "doi"),
        identifier=doc["identifier"],
    )
    return ctx


def benchmark_index(doc):
    """{config: [split, ...]} as published, from the Croissant file manifest."""
    index = {}
    for path, _ in manifest_rows(doc):
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "data" or not parts[2].endswith(".parquet"):
            raise RenderError("unexpected benchmark file {!r}".format(path))
        index.setdefault(parts[1], []).append(parts[2][:-len(".parquet")])
    return dict((config, sorted(cities)) for config, cities in index.items())


def benchmark_context(doc, facts, benchmark=BENCHMARK):
    """Template values for README.benchmark_card.template.md -- the same keys
    export_benchmark.render_card fills, sourced from Croissant + facts instead of a build."""
    import export_benchmark as eb                # pyarrow + PIL at import; only needed here

    index = benchmark_index(doc)
    declared = sorted(r["splits/name"] for r in record_set(doc, "splits")["data"])
    cities = sorted(index.get(eb.RECORDS, []))
    if cities != declared:
        raise RenderError("records files cover {} but splits declares {}".format(cities, declared))
    imagery = sorted(set(sum((index.get(c, []) for c in eb.IMAGERY_CONFIGS), [])))
    overlap_md, overlap_counts = eb.overlap_listing(benchmark, cities)
    extents = [dict((k.split("/", 1)[1], v) for k, v in row.items())
               for row in record_set(doc, "split_extents")["data"]]
    ctx = dict(
        configs_yaml=eb.configs_yaml(index),
        git_commit=facts["export_git_commit"],
        export_date=facts["export_date"],
        repo_id=repo_id(doc),
        n_cities=len(cities),
        n_imagery_splits=len(imagery),
        records_only_note=eb.records_only_note(eb.records_only_splits(index)),
        overlap_checked_at=eb.overlap_checked_at(benchmark),
        confidence_table=eb.confidence_table(benchmark, cities),
        budapest_confidence=eb.split_confidences(benchmark, ["budapest_district5"])["budapest_district5"],
        overlap_listing=overlap_md,
        bend_unseen_rows=eb.reviewed_rows(benchmark, "bend") - overlap_counts.get("bend", 0),
        split_date_range=eb.split_date_range(benchmark, cities),
        total_gb="{:.2f}".format(sum(b for _, b in manifest_rows(doc)) / 1e9),
        split_table=eb.split_table(extents),
        strata_table=eb.strata_table(benchmark, cities),
        paper_cite_as=eb.croissant_citations(CROISSANT)[0],
        cite_as=doc["citeAs"],
        size_category=size_category(sum(r["num_panoramas"] for r in extents)),
    )
    return ctx


# --------------------------------------------------------------------------------- render

def render(card, facts_path=FACTS, benchmark=BENCHMARK, croissant_dir=CROISSANT):
    """The rendered card text (LF) for card in CARDS."""
    croissant_name, template, _ = CARDS[card]
    croissant_path = Path(croissant_dir) / croissant_name
    croissant_text = croissant_path.read_text(encoding="utf-8")
    doc = json.loads(croissant_text)
    facts = load_facts(card, croissant_text, facts_path)
    if card == "dataset":
        ctx = dataset_context(doc, facts)
    else:
        ctx = benchmark_context(doc, facts, benchmark)
    text = read_template(template)
    try:
        out = text.format(**ctx)
    except (KeyError, IndexError) as exc:
        raise RenderError("{}: template needs {} which nothing supplies".format(template, exc))
    left = PLACEHOLDER_RE.findall(out)
    if left:
        raise RenderError("{}: unfilled placeholders {}".format(template, sorted(set(left))))
    return out


def write_cards(out_dir, **kw):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for card, (_, _, name) in CARDS.items():
        path = out_dir / name
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(card, **kw))
        written.append(path)
    return written


def check(committed=COMMITTED, **kw):
    """Render to a temp dir and diff against the committed renders. Returns a list of problems."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        for path in write_cards(tmp, **kw):
            ref = Path(committed) / path.name
            fresh = path.read_text(encoding="utf-8")
            if not ref.is_file():
                problems.append("{} is not committed (run --update)".format(ref))
                continue
            old = ref.read_text(encoding="utf-8").replace("\r\n", "\n")
            if old != fresh:
                diff = "".join(list(difflib.unified_diff(
                    old.splitlines(True), fresh.splitlines(True),
                    fromfile=str(ref), tofile="fresh render"))[:60])
                problems.append("{} differs from a fresh render:\n{}".format(ref.name, diff))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="render to a temp dir and diff against docs/data/hf_cards/")
    mode.add_argument("--update", action="store_true",
                      help="rewrite the committed renders in docs/data/hf_cards/")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory for a plain render (default: hf_export/cards, git-ignored)")
    ap.add_argument("--facts", default=str(FACTS), help=argparse.SUPPRESS)
    ap.add_argument("--committed", default=str(COMMITTED), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    kw = dict(facts_path=Path(args.facts))
    try:
        if args.check:
            problems = check(Path(args.committed), **kw)
            for p in problems:
                print("FAIL:", p)
            if problems:
                return 1
            print("ok: {} cards re-render identical to {}".format(
                len(CARDS), Path(args.committed).resolve().relative_to(REPO).as_posix()
                if Path(args.committed).resolve().is_relative_to(REPO) else args.committed))
            return 0
        target = Path(args.committed) if args.update else Path(args.out)
        for path in write_cards(target, **kw):
            print("wrote", path)
        print("Nothing uploaded. See docs/dataset_cards_150.md for the upload step.")
        return 0
    except RenderError as exc:
        print("error:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
