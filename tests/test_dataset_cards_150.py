"""Guards for the #150 dataset cards (scripts/export_dataset_cards.py).

The cards for `projectsidewalk/rampnet-dataset` and `projectsidewalk/rampnet-benchmark` are
rendered from the committed Croissant records plus scripts/hf_package/cards_facts.json, and the
reviewed renders are committed under docs/data/hf_cards/. These tests pin:

- every section heading of the Hugging Face dataset card template, at its level and in order;
- the YAML front matter parses, and its licence is the Croissant licence;
- every split and record count the cards print equals the Croissant value;
- no `{placeholder}` survives rendering;
- `--check` passes on the committed renders, and fails when a fact changes;
- a fact marked in_croissant that the Croissant file does not carry is refused;
- the benchmark card's split table is the same whether it is filled from Croissant (this
  renderer) or from the committed bundles (export_benchmark.render_card);
- nothing in the renderer can upload.

CPU only, no network: reads committed files and writes only to tmp_path.
"""
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import export_dataset_cards as cards  # noqa: E402

CROISSANT = REPO_ROOT / "croissant"
COMMITTED = REPO_ROOT / "docs" / "data" / "hf_cards"


@pytest.fixture(scope="module")
def rendered():
    return dict((card, cards.render(card)) for card in cards.CARDS)


@pytest.fixture(scope="module")
def croissant():
    return dict((card, json.loads((CROISSANT / spec[0]).read_text(encoding="utf-8")))
                for card, spec in cards.CARDS.items())


def headings(text):
    """(level, text) of every Markdown heading outside fenced code blocks."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        m = re.match(r"^(#{1,6}) (.+?)\s*$", line)
        if m and not fenced:
            out.append((len(m.group(1)), m.group(2)))
    return out


def front_matter(text):
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---\n", 2)[1])


def table_rows(text, header):
    """The body rows of the first Markdown table whose header line starts with `header`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


# ------------------------------------------------------------------------ the HF template

@pytest.mark.parametrize("card", sorted(cards.CARDS))
def test_every_hf_template_heading_is_present_in_order(rendered, card):
    found = headings(rendered[card])
    pos = 0
    for level, title in cards.HF_TEMPLATE_HEADINGS:
        match = next((i for i in range(pos, len(found))
                      if found[i][0] == level and (found[i][1] == title or
                                                   (level == 1 and found[i][1].startswith(title)))),
                     None)
        assert match is not None, "{}: missing or out of order: {} {}".format(card, "#" * level, title)
        pos = match + 1


@pytest.mark.parametrize("card", sorted(cards.CARDS))
def test_the_issue_150_topics_are_covered(rendered, card):
    """The second checkbox of #150 names these; each must be on the card, not just implied."""
    text = rendered[card]
    for needle in ("Splits and their rationale", "city-held-out", "Known biases" if card == "dataset"
                   else "Provider, camera and city are confounded", "Direct Use", "Capture years",
                   "a major version on any change to a data file",
                   "a minor version on a metadata-only change", "DOI: forthcoming",
                   "@inproceedings", "@misc", "Personal and Sensitive Information"):
        assert needle.lower() in text.lower(), "{} card lacks {!r}".format(card, needle)
    if card == "dataset":
        assert "Seam duplicates" in text and "v1.1-corrected-eval" in text
    else:
        assert "Reviewer caveats travel with the rows" in text and "Training overlap" in text


# ------------------------------------------------------------------------ front matter

@pytest.mark.parametrize("card", sorted(cards.CARDS))
def test_front_matter_parses_and_licence_is_croissants(rendered, croissant, card):
    meta = front_matter(rendered[card])
    assert meta["license"] == cards.HUB_LICENSES[croissant[card]["license"]]
    assert meta["pretty_name"]
    assert meta["task_categories"] and meta["size_categories"]
    for value in meta["size_categories"]:
        assert re.match(r"^(n<1K|\d+[KM]<n<\d+[KM]|n>10M)$", value), value
    title = next(t for level, t in headings(rendered[card]) if level == 1)
    assert title == "Dataset Card for " + meta["pretty_name"]


def test_benchmark_front_matter_declares_every_published_file(rendered, croissant):
    meta = front_matter(rendered["benchmark"])
    declared = sorted(f["path"] for c in meta["configs"] for f in c["data_files"])
    published = sorted(r["file_manifest/path"]
                       for r in cards.record_set(croissant["benchmark"], "file_manifest")["data"])
    assert declared == published


def test_size_categories_follow_the_counts(rendered):
    assert front_matter(rendered["dataset"])["size_categories"] == ["100K<n<1M"]
    assert front_matter(rendered["benchmark"])["size_categories"] == ["1K<n<10K"]


# ------------------------------------------------------------------------ counts

def test_dataset_split_counts_equal_croissant(rendered, croissant):
    text = rendered["dataset"]
    rows = dict((r[0], r) for r in table_rows(text, "| split | Hub directory"))
    splits = dict((r["splits/name"], r["splits/num_rows"])
                  for r in cards.record_set(croissant["dataset"], "splits")["data"])
    names = {"train": "train", "val": "validation", "test": "test"}
    for split, n in splits.items():
        row = rows[names[split]]
        assert row[4] == "{:,}".format(n)
        assert row[5].startswith("{:,} (".format(n - 3))     # minus the placeholder rows
    total = sum(splits.values())
    assert rows["**total**"][4] == "**{:,}**".format(total)
    assert rows["**total**"][5] == "**{:,}**".format(total - 9)
    assert "{:,} Google Street View panoramas".format(total - 9) in text
    n_files = len(cards.record_set(croissant["dataset"], "file_manifest")["data"])
    assert rows["**total**"][3] == "**{:,}**".format(n_files)


def test_benchmark_split_counts_equal_croissant(rendered, croissant):
    text = rendered["benchmark"]
    rows = dict((r[0].strip("`"), r) for r in table_rows(text, "| split | imagery |"))
    extents = cards.record_set(croissant["benchmark"], "split_extents")["data"]
    for row in extents:
        got = rows[row["split_extents/split"]]
        assert got[1] == row["split_extents/imagery_source"]
        assert got[2] == "{:,}".format(row["split_extents/num_panoramas"])
        assert got[3] == "{} to {}".format(row["split_extents/capture_date_min"],
                                           row["split_extents/capture_date_max"])
    total = sum(r["split_extents/num_panoramas"] for r in extents)
    assert rows["**total**"][2] == "**{:,}**".format(total)
    assert "These {} splits were built".format(len(extents)) in text


def test_benchmark_split_table_agrees_with_the_bundles(croissant):
    """export_benchmark.render_card fills {split_table} from the committed bundles; this renderer
    fills it from Croissant. For the published splits the two must print the same table."""
    import export_benchmark as eb
    from validate_croissant import derive_split_extents
    extents = cards.record_set(croissant["benchmark"], "split_extents")["data"]
    from_croissant = [dict((k.split("/", 1)[1], v) for k, v in r.items()) for r in extents]
    splits = [r["split"] for r in from_croissant]
    assert eb.split_table(from_croissant) == eb.split_table(
        derive_split_extents(splits, REPO_ROOT / "benchmark"))


# ------------------------------------------------------------------------ rendering

@pytest.mark.parametrize("card", sorted(cards.CARDS))
def test_no_placeholder_left(rendered, card):
    assert not cards.PLACEHOLDER_RE.findall(rendered[card])
    assert "\r" not in rendered[card]


def test_check_passes_on_the_committed_renders(capsys):
    assert cards.main(["--check"]) == 0
    assert "re-render identical" in capsys.readouterr().out


def test_committed_renders_are_lf():
    for _, _, name in cards.CARDS.values():
        assert b"\r\n" not in (COMMITTED / name).read_bytes()


def _facts_with(tmp_path, card, key, value):
    facts = json.loads(cards.FACTS.read_text(encoding="utf-8"))
    facts[card][key]["value"] = value
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts), encoding="utf-8")
    return path


def test_a_changed_fact_fails_check(tmp_path, capsys):
    path = _facts_with(tmp_path, "dataset", "crop_pretrain_crops", 20699)
    assert cards.main(["--check", "--facts", str(path)]) == 1
    out = capsys.readouterr().out
    assert "README.rampnet-dataset.md differs" in out and "20,699" in out


def test_a_missing_committed_render_fails_check(tmp_path):
    assert cards.main(["--check", "--committed", str(tmp_path)]) == 1


def test_an_in_croissant_fact_the_record_lacks_is_refused(tmp_path):
    path = _facts_with(tmp_path, "dataset", "seam_pairs", 8362)
    with pytest.raises(cards.RenderError, match="in_croissant"):
        cards.render("dataset", facts_path=path)
    assert cards.main(["--out", str(tmp_path / "out"), "--facts", str(path)]) == 2


def test_a_fact_without_a_source_is_refused(tmp_path):
    facts = json.loads(cards.FACTS.read_text(encoding="utf-8"))
    facts["dataset"]["gold_marks"]["source"] = " "
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts), encoding="utf-8")
    with pytest.raises(cards.RenderError, match="source"):
        cards.render("dataset", facts_path=path)


def test_an_unknown_licence_is_refused(croissant):
    doc = dict(croissant["dataset"], license="https://spdx.org/licenses/CC-BY-4.0.html")
    with pytest.raises(cards.RenderError, match="licence"):
        cards.hub_license(doc)


def test_the_renderer_cannot_upload():
    source = (REPO_ROOT / "scripts" / "export_dataset_cards.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+huggingface_hub", source, re.M)
    assert "HfApi" not in source
    assert "--push" not in source and "--upload" not in source
    with pytest.raises(SystemExit):
        cards.main(["--push"])


# ------------------------------------------------------------------------ review fixes (#282)

def test_a_short_in_croissant_value_must_match_a_whole_token(tmp_path):
    """"13" occurs inside 13,357, dates and shas; none of those may satisfy the guard."""
    path = _facts_with(tmp_path, "dataset", "crop_model_cities", "13")
    with pytest.raises(cards.RenderError, match="crop_model_cities"):
        cards.render("dataset", facts_path=path)
    assert cards.appears_bounded("12", "from 12 other cities")
    for text in ("13,357", "a 2026-01-13 date", "0.13", "x13y", "sha 4f13c2", "13.5"):
        assert not cards.appears_bounded("13", text), text


@pytest.mark.parametrize("card", sorted(cards.CARDS))
def test_each_card_prints_its_own_croissant_citation(rendered, croissant, card):
    assert croissant[card]["citeAs"] in rendered[card]
    # the paper entry, on both cards, is the dataset record's citeAs, not a hand-kept copy
    assert croissant["dataset"]["citeAs"] in rendered[card]


def test_strata_table_matches_the_bundles_and_the_split_counts(rendered, croissant):
    rows = dict((r[0].strip("`"), r) for r in table_rows(rendered["benchmark"], "| split | `top` |"))
    extents = cards.record_set(croissant["benchmark"], "split_extents")["data"]
    for row in extents:
        split = row["split_extents/split"]
        verdicts = json.loads((REPO_ROOT / "benchmark" / split / "verdicts.json")
                              .read_text(encoding="utf-8"))["panos"]
        want = [sum(1 for v in verdicts.values() if v.get("group") == g)
                for g in ("top", "random", "empty")]
        assert [int(x) for x in rows[split][1:4]] == want
        assert int(rows[split][4]) == row["split_extents/num_panoramas"]
    assert rows["bend"][1:4] == ["5", "95", "10"] and rows["richmond"][1:4] == ["5", "94", "25"]


def test_the_doc_states_the_fact_counts_correctly():
    facts = json.loads(cards.FACTS.read_text(encoding="utf-8"))
    counts = dict((card, (len(facts[card]), sum(1 for v in facts[card].values()
                                                 if v.get("in_croissant"))))
                  for card in cards.CARDS)
    doc = " ".join((REPO_ROOT / "docs" / "dataset_cards_150.md").read_text(encoding="utf-8").split())
    (n_ds, k_ds), (n_b, k_b) = counts["dataset"], counts["benchmark"]
    assert "{} of the {} dataset facts, and {} of the {} benchmark facts".format(
        k_ds, n_ds, k_b, n_b) in doc
