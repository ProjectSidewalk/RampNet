# Hugging Face dataset cards for `rampnet-dataset` and `rampnet-benchmark`

**Status: written 2026-11-05 for the second and sixth checkboxes of
[#150](https://github.com/ProjectSidewalk/RampNet/issues/150) (dataset cards that meet the Hugging
Face template fully; carry the #127 review notes into the published benchmark card). Nothing has
been uploaded to the Hub.** The cards are repo-managed templates, rendered from the committed
Croissant records by a script with a `--check` mode, and the reviewed renders are committed:

| card | template | committed render | Hub repository |
| :--- | :--- | :--- | :--- |
| `rampnet-dataset` | [`scripts/hf_package/README.dataset_card.template.md`](../scripts/hf_package/README.dataset_card.template.md) (new) | [`docs/data/hf_cards/README.rampnet-dataset.md`](data/hf_cards/README.rampnet-dataset.md) | [`projectsidewalk/rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) |
| `rampnet-benchmark` | [`scripts/hf_package/README.benchmark_card.template.md`](../scripts/hf_package/README.benchmark_card.template.md) (extended) | [`docs/data/hf_cards/README.rampnet-benchmark.md`](data/hf_cards/README.rampnet-benchmark.md) | [`projectsidewalk/rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) |

The DOI, the leaderboard and any upload are Jon's decisions and are not part of this work.

## 1. How the cards are built

`scripts/export_dataset_cards.py` fills each template with Python's `str.format`, the same way
every other card template in `scripts/hf_package/` is filled, from three committed inputs:

1. **`croissant/rampnet-dataset.json` and `croissant/rampnet-benchmark.json`**, the record of truth
   ([`fair_metadata_150.md`](fair_metadata_150.md)). Every split count, file count and size,
   pinned revision, date, version, creator, licence, citation, city box and field description on
   the cards is read from these files, so a card cannot quote a count that the Croissant record
   does not.
2. **`scripts/hf_package/cards_facts.json`**, for the facts Croissant does not carry as a
   structured field (for example the label count, the seam-duplicate count, or the Stage 1
   agreement figures, which Croissant has only inside prose). Every entry has a `source`. An entry marked
   `in_croissant` must also appear, exactly as the card prints it, somewhere in that card's
   Croissant file; if it does not, rendering stops with an error naming the fact. 27 of the 39
   dataset facts, and 1 of the 2 benchmark facts, are marked this way. The match is on a whole
   token (not inside a longer number, word, date or hash), so "12" is not satisfied by "12,345" or
   a sha. The remaining limit: a short value can still match an unrelated occurrence of the same
   token elsewhere in the Croissant file, so the guard proves presence, not meaning.
3. **For the benchmark card only, the committed bundles**, read through the helpers
   `scripts/export_benchmark.py` already uses for its card: the review-confidence table and
   Budapest's value from `benchmark/<split>/verdicts.json`, and the training-overlap listing and
   check date from `benchmark/train_overlap.json`. They are restricted to the splits Croissant
   declares, so the render shows what the published splits would carry.

```bash
python scripts/export_dataset_cards.py            # write hf_export/cards/README.<repo>.md (git-ignored)
python scripts/export_dataset_cards.py --check    # render to a temp dir and diff against docs/data/hf_cards/
python scripts/export_dataset_cards.py --update   # rewrite docs/data/hf_cards/ after a reviewed change
```

`--check` exits 1 on any difference and prints the diff. It is registered in
`scripts/check_all.py` as `dataset_cards_150` (committed inputs only, about 0.2 s), with pins on
`docs/data/hf_cards/`, `croissant/`, `cards_facts.json`, both templates,
`benchmark/train_overlap.json` and `benchmark/*/verdicts.json`. The script has no upload flag and
does not import `huggingface_hub`; a test asserts both.

**The benchmark template is still rendered by `scripts/export_benchmark.py` too**, which is what
publishes it with a `records` or `card` push. That renderer fills the same placeholders from the
built package and the committed bundles, plus two new ones added here: `{split_table}` (from
`validate_croissant.derive_split_extents`, the function the Croissant validator re-derives the
`split_extents` record set with), `{size_category}`, `{strata_table}` (per-split `top` / `random` /
`empty` counts from each split's `verdicts.json` `group`, the value written to `review_group`),
and `{paper_cite_as}` / `{cite_as}` (the `citeAs` of the dataset and benchmark Croissant records,
so neither card keeps a hand-copied BibTeX). A test renders the split table both ways
and requires the same table. `tests/test_train_overlap.py::test_rendered_card_agrees_with_the_built_records`
renders the template through `export_benchmark.render_card` and still passes.

## 2. Coverage of the Hugging Face template

Checked against `huggingface_hub`'s
[`datasetcard_template.md`](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md)
and the metadata spec in `hub-docs/datasetcard.md`, both read on 2026-11-05. The template has 24
headings; both cards carry all 24 at the template's level and in its order
(`HF_TEMPLATE_HEADINGS` in the script; `test_every_hf_template_heading_is_present_in_order`).
"[optional]" is dropped from the heading text, as filled cards do. Both cards also keep the older
template's subsections where #150 names them (supported tasks, data instances and fields, splits
and their rationale, social impact, known biases, licensing, maintenance, contributions).

| #150 asks for | `rampnet-dataset` | `rampnet-benchmark` |
| :--- | :--- | :--- |
| composition | Dataset Structure: fields table (from Croissant), splits table, 384 shards | Dataset Structure: per-split table, configs, `records` columns |
| collection process | Source Data: records, panorama selection, imagery, label placement, what was lost | Source Data: auto-labeler run, stratified sample, review |
| splits and their rationale | random panorama-level split, not city-held-out; the Hub copy is the split of record | city-held-out by design; the three sampling strata |
| seam-duplicate note | Limitations: 8,361 pairs from `docs/seam.md` §3 | not applicable (benchmark scoring wraps x) |
| known biases | city mix, provider, capture years, selection, model-anchored labels | model-anchored GT, provider/camera/city confounded, capture years, reviewer confidence |
| intended uses | Direct Use, Out-of-Scope Use | Direct Use, Out-of-Scope Use |
| citation | paper BibTeX (Croissant `citeAs`), dataset `@misc` with `DOI: forthcoming`, APA | paper BibTeX, benchmark `@misc` with `DOI: forthcoming`, APA |
| maintenance and versioning | major version on any data-file change, minor on metadata-only | the same |
| #127 review notes | not applicable | already in the template from #193; kept, and the Training overlap section kept |

**YAML front matter.** `license` is the Hub identifier for the Croissant licence (`mit`; any other
licence is refused rather than guessed). `pretty_name` is Croissant's `alternateName`.
`size_categories` is computed from the counts (`100K<n<1M` for 214,385 rows; `1K<n<10K` for the
benchmark's 1,109). `task_categories` adds `keypoint-detection` to the dataset's existing
`object-detection`. `annotations_creators` is `machine-generated` for the dataset and
`expert-generated` for the benchmark. `language: en` is declared only for the benchmark, whose
`note`, `review_summary` and `review_caveats` fields are English text; the dataset has no text
field. No `configs` block is added to the dataset card: the Hub already reads its `train/`,
`val/` and `test/` directories, and a new block could change what `load_dataset` returns. No
`dataset_info` block is written for either card, because its `num_bytes` and `dataset_size` are
Arrow sizes that no committed source records.

## 3. Where every number on the cards comes from

| number | value | source |
| :--- | :--- | :--- |
| dataset rows per split | 150,066 / 42,878 / 21,441 (214,385) | Croissant `splits` record set |
| usable panoramas per split | 150,063 / 42,875 / 21,438 (214,376) | rows minus 3 placeholder rows per split, from [`restage_263.md`](restage_263.md); the renderer refuses to render unless the total is the one the Croissant description quotes |
| labels | 849,895 (card) / 849,904 (Parquet scan) | facts; both in the Croissant description, the scan in [`seam.md`](seam.md) §3 |
| shards, size | 384 shards, 462.47 GB | Croissant `file_manifest` (count and summed bytes) |
| split percentages | 70% / 20% / 10% | computed from the usable counts |
| revision, dates, version | `ee882e3` (2025-08-15), created 2025-07-12, published 2025-07-15, version 1.0.0 | Croissant `repo` description, `dateCreated`, `datePublished`, `dateModified`, `version` |
| creators, citation | five authors; the paper BibTeX | Croissant `creator`, `citeAs` |
| government records | 276,071 committed (NYC 217,679, Portland 45,035, Bend 13,357); 276,615 in Table 1; NYC 78.8% | facts, all `in_croissant`; [`data_provenance.md`](data_provenance.md) §3 |
| city boxes | three `S W N E` boxes | Croissant `spatialCoverage` |
| records never labelled | 119,359 (43.23%) | facts, `in_croissant`; `data_provenance.md` §3.1 |
| panoramas never written | 4,571 of 219,170 (2.09%); 214,599 written | facts, `in_croissant`; [`stage1_generation_cost.md`](stage1_generation_cost.md), [`rampnet1_findings.md`](rampnet1_findings.md) §1 |
| selection and placement constants | 10 m, 35 m, about 20% negatives, `min_distance 40 px, threshold 0.4` | facts, `in_croissant`; the Stage 1 scripts |
| Stage 1 agreement | P 0.9152 / R 0.9275 corrected; 0.9403 / 0.9245 published | facts, `in_croissant`; README erratum, `rampnet1_findings.md` §1 |
| gold set | 1,000 panoramas, 3,919 marks | facts; `manual_labels/`, `seam.md` §2 |
| seam duplicates | 8,361 pairs, 0.98% of labels, 7,987 panoramas (3.7%); 10 in the gold set | facts; `seam.md` §2–3 |
| crop-model training sets | 20,698 crops / 27,704 labels; 312 crops / 1,212 labels; 12 cities | facts; the live dataset card's summary table, `data_provenance.md` §1 |
| crop decode | 3.961 to 2.964 heatmap px | facts; [`crop_decode_221.md`](crop_decode_221.md) §1 |
| benchmark splits, counts, sources, capture months | 9 splits, 1,109 panoramas | Croissant `split_extents` |
| benchmark configs and size | 4 configs x 9 splits, 11.41 GB | Croissant `file_manifest` |
| benchmark review dates | 2026-07-22 to 2026-08-01 | `verdicts.json` `exported_at` (the template's existing `{split_date_range}`) |
| review confidence, overlap | Budapest `low`; Bend's 4 panoramas, 106 unseen rows; checked 2026-10-19 | `verdicts.json`, `benchmark/train_overlap.json` |
| benchmark export provenance | `d7aeda4-dirty`, 2026-08-17 | facts; the Hub's `build_manifest.json` at `63d5ffd`, as quoted in `fair_metadata_150.md` §1 |
| Bend with and without the overlap | 0.954 / 0.758 … 0.976 / 0.731 | hardcoded in the template since #193, pinned by `tests/test_train_overlap.py` |

## 4. What changed in the benchmark card

- **New sections** for every template heading it lacked: Dataset Details (description, curators,
  funding, language, licence, sources), Uses (direct, supported tasks, out of scope), the per-split
  table with the city-held-out rationale and the sampling strata, Dataset Creation (rationale,
  collection, producers, annotation process, annotators, personal information), Bias, Risks, and
  Limitations with Recommendations, a benchmark `@misc` citation and APA, Glossary, Licensing,
  Maintenance and versioning, Contributions, Card Authors and Contact.
- **Existing sections kept** with their text and placeholders: the warning banner, the intro, the
  Configs table, the `records` columns, "Reviewer caveats travel with the rows", "Training
  overlap" (with the Bend table), Usage, and "Verifying you have the pixels the reviewers
  judged". They moved one heading level down, under Dataset Structure; their anchors are
  unchanged. The Provenance table moved under More Information.
- **The #127 notes were already carried** by #193 (the `records` columns, the confidence table, the
  overlap listing); nothing was duplicated. The new Bias and Recommendations sections point at them.
- **Two fixes from [`fair_metadata_150.md`](fair_metadata_150.md) §6 applied:** the `source` row now
  says `launch` (Google Street View) or `mapillary`, which is what the data holds, rather than
  `gsv`; and the paper BibTeX is no longer a hand-kept copy (which held a URL in `doi`): both
  cards print the dataset Croissant record's `citeAs`, and the benchmark card prints its own
  record's `citeAs` for the benchmark entry, so the citations cannot drift from Croissant. A line
  under each paper entry explains `note = {DOI: forthcoming}` beside the arXiv DOI.
- **Title** is now "Dataset Card for RampNet Benchmark Imagery", the template's H1 with the
  existing name.

## 5. What the committed renders describe, and the #127 push

The committed dataset render describes the Hub as it is: the files at `ee882e3`.

The committed benchmark render takes its splits, files and counts from Croissant, which pins Hub
revision `63d5ffd` (9 splits). Its text, however, comes from the template on `main`, which since
#193 describes the seven `records` columns #127 added, and those columns are **not** yet on the
Hub. So this render belongs on the Hub **with** the #127 `records` push, not before it. That push
runs `export_benchmark.py`, which renders the card itself from the built package. That render will
differ from the committed one in three expected ways: the provenance commit and date (that run's,
not `d7aeda4-dirty` / 2026-08-17), the split count (11 with the two Laurens splits, and the
records-only note under the Configs table), and the split table (11 rows). After the push,
Croissant is updated as `fair_metadata_150.md` §2 lists, and `--update` here re-renders from it.

## 6. TODO(Jon)

Each of these is written as `TODO(Jon)` on the card where it applies:

1. **Funding** acknowledgement for each card (no committed document records it).
2. **Imagery terms.** Both cards declare MIT, as the published cards always have. Whether the
   cards should also state the terms Google Street View imagery carries (and, for the benchmark,
   restate Mapillary's CC BY-SA 4.0 beyond the per-record `copyright`) is a licensing decision.
3. **Who labelled** the manual crop set and the 1,000-panorama gold set, if the dataset card should
   name them.
4. **Benchmark reviewers** for the five splits whose `verdicts.json` has no `review_notes`
   (`annapolis`, `bend`, `clovis`, `morgantown`, `richmond`); their `reviewer` column is null.
5. **Benchmark creators** for the DOI record (already open in `fair_metadata_150.md` §3).
6. **Card authors**, if individuals should be listed.
7. **A capture-year histogram** for the dataset, which would need a metadata lookup for every
   panorama id; the rows carry no capture date.

Decisions carried from earlier #150 work and not changed here: the DOI, the version label, and
tagging the Hub revisions ([`fair_metadata_150.md`](fair_metadata_150.md) "Decisions for Jon").

## 7. Uploading later (not run here)

The card upload is a metadata-only change (a minor version under the policy above). It also moves
each Hub repository's `main`, so `scripts/validate_croissant.py --hub` will fail until the
Croissant pins are moved to the new revision (`fair_metadata_150.md` §5); doing both in one sitting
keeps the record and the Hub in step. The data files stay byte-identical, so `file_manifest` does
not change.

**`rampnet-dataset`** (card only):

```bash
python scripts/export_dataset_cards.py --check
python scripts/export_dataset_cards.py
python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='hf_export/cards/README.rampnet-dataset.md', path_in_repo='README.md', repo_id='projectsidewalk/rampnet-dataset', repo_type='dataset', commit_message='Dataset card: full Hugging Face template, rendered from croissant/rampnet-dataset.json (#150)')"
```

**`rampnet-benchmark`**, with the #127 records push (section 5; the precondition and the exact
command are in [#193](https://github.com/ProjectSidewalk/RampNet/pull/193)):

```bash
python scripts/export_benchmark.py adopt --out dist/rampnet-benchmark
python scripts/export_benchmark.py records --out dist/rampnet-benchmark --push --message "records: #127 columns and Laurens ground truth; card: full Hugging Face template (#127, #150)"
```

Then move the Croissant pins (`fair_metadata_150.md` §2 and §5), run
`python scripts/export_dataset_cards.py --update`, and commit the re-rendered cards.

## 8. Decisions made in this work

- The cards are filled from Croissant rather than duplicating its numbers, and facts outside it
  carry a source and, where possible, an `in_croissant` cross-check.
- The benchmark template gained two placeholders (`{split_table}`, `{size_category}`), filled by
  both renderers; no existing placeholder was removed or renamed.
- The nine-row gap between the Hub's 214,385 rows and the card's 214,376 panoramas is explained on
  the card by the placeholder rows `restage_263.md` found. Croissant still calls it "recorded, not
  explained"; that text should be updated with the next Croissant revision, which is not changed
  here. The 9-label gap (849,904 against 849,895) has the same size but was not checked to be the
  same rows, and the card says so.
- The model card's "heatmap size is nominal" note (#221) is carried into the dataset card's
  limitations in its crop-model form, because the crop model has the same head and places every
  label. Whether the 8-pixel grid survives Stage 1's resize and reprojection was not measured
  ([`crop_decode_221.md`](crop_decode_221.md) §6), and the card says so.
- The round-1 and round-2 crop datasets are linked by their current names
  (`rampnet-crop-model-dataset-round1` / `-round2`), not the pre-rename id the live card uses.
