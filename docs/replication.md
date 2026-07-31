# Replication: can someone else reproduce our numbers from a clean clone?

**The standing rule is in `CLAUDE.md`.** This document is the ledger that keeps it honest: for
every experiment, what a new student would actually need, and — where they cannot get it yet —
exactly what is blocking them.

The test is not "is the script committed". It is: *clone the repo, follow written instructions,
obtain every input, re-run, get our numbers.* An experiment that fails that test is a result that
lives on one machine.

## Status by input

| input | size | where it is | replicable? |
| :--- | ---: | :--- | :--- |
| `benchmark/*/records.jsonl`, `verdicts.json` | small | **committed** | ✅ |
| `manual_labels/*.txt` (gold set) | small | **committed** | ✅ |
| `analysis_out/op_cache/*.json` (RampNet low-floor detections) | ~KB | **committed** | ✅ |
| `analysis_out/*.json` (derived results) | ~100 KB | **committed** | ✅ |
| `benchmark/miss_taxonomy_46/*.json` (human verdicts) | small | **committed** | ✅ |
| RampNet model weights | — | HF `projectsidewalk/rampnet-model` | ✅ |
| Stage 1 dataset | 44 GB | HF `projectsidewalk/rampnet-dataset` | ✅ |
| `benchmark/model_detections/` (challenger detections) | 18.0 MB | **committed** ✅ | ✅ |
| **`benchmark/*/panos/` (native-res panoramas)** | **9.0 GB** | **git-ignored; HF #21 pending** | ❌ **blocker** |

### ✅ Resolved — the challenger detections are published

`fp_taxonomy.py`, `silent_witness.py`, `complementarity.py` and `null_recall.py` all read the
challenger detections (Gemini ×2, Qwen ×2, Molmo, OWLv2, Grounding DINO). Those cost real GPU-hours
on Hyak and paid API spend, and until 2026-07-31 they lived only in a git-ignored `.model_cache/`,
so every number they produced was reproducible on exactly one machine.

`.model_cache` is fine as a *working* cache and hostile as a published artifact: 12,951
single-panorama shards keyed by an opaque SHA-1 of (label, signature, city, pano), unreadable
without reconstructing detector signatures. `scripts/analysis/export_model_cache.py` consolidates
it into **61 human-readable files, 18.0 MB**, one per (model, split), keyed by panorama id with the
detector signature recorded inside.

```bash
python scripts/analysis/export_model_cache.py --out benchmark/model_detections
python scripts/analysis/export_model_cache.py --verify     # exported == cached
```

`--verify` re-scores every split from both sources and asserts identical TP/FP counts — **61/61
pairs verified identical** — because a published artifact that silently differs from what produced
the paper's numbers is worse than none.

Downstream scripts prefer the published files over `.model_cache`, and the label a `--models` spec
resolves to is derived *without* building a detector, so **a clean clone reproduces these numbers
with no cache, no GPU and no torch installed.** Verified by running against a nonexistent
`--cache-dir`.

### Blocker 2 — the panoramas are 9.0 GB and only HF can carry them

`benchmark/README.md` already says the native-resolution panos "are archived separately and
published to HF (#21); they are intentionally not in git". **#21 is still open**, so the imagery
half of the benchmark is currently unobtainable by anyone else. That blocks `miss_gallery.py`,
`fp_gallery.py`, `gt_gallery.py` and any re-rating of the #46 tagging task.

Nothing in this repo can fix that; it needs #21 to land.

## Every manual-review task, and what it would take to redo it

Three distinct human passes exist. All three produce committed judgments; they differ in whether
someone else can **redo the pass**.

| task | judgments | what the reviewer saw | redoable by someone else? |
| :--- | :--- | :--- | :--- |
| **GT verification** (9 splits) | `benchmark/<city>/verdicts.json` ✅ | whole panoramas at **model resolution** via `scripts/gt_gallery.py` | ❌ needs the panos |
| **#55 incremental-FP A/B** (8 splits) | `benchmark/<city>/incremental_fp_tags.json` ✅ | crops from `low_floor_sweep.py gallery` (244 MB PNG, git-ignored) | ❌ needs the panos |
| **#46 miss taxonomy** (1 split-set) | `benchmark/miss_taxonomy_46/silent__jonf.json` ✅ | **crops committed** (15 MB) | ✅ **yes, today** |

### What makes #46 redoable, and what the other two need

The insight is that **a rating task does not need the 9 GB of panoramas — it needs the crops the
rater actually saw**, and those are small. #46's are committed at
`benchmark/miss_taxonomy_46/silent_gallery/` (50 crops + manifest, 15 MB), so a second rater needs
no imagery, no `.model_cache` and no GPU: open the committed crops, tag, commit a second file.

Applying the same to the other two is a size question, and here are the measured numbers:

- **#55 A/B galleries: 244 MB as PNG** across the six cities still on disk (two cities' galleries
  have already been deleted, though their tags survive — which is the failure mode this section
  exists to prevent). Re-encoded as JPEG they would be far smaller. **Decision needed:** commit
  re-encoded crops, or publish the galleries to HF.
- **GT verification is the hard one.** Reviewers scan *whole panoramas* for missed ramps, so crops
  are not enough. But `gt_gallery.py` renders at the model's **4096×2048**, never native (the #26
  fairness note), so what is actually required is a **model-resolution derivative — roughly
  1–2 GB, not the 9 GB native archive.** That is a much easier thing to publish, and it is worth
  scoping HF #21 to include it rather than native-res alone.

Both rubrics also currently live in code rather than beside the data: `gt_gallery.py`'s docstring
for the verdict schema, and `low_floor_sweep.py` for what `A` and `B` mean. A second rater should
not have to read the source to learn the rubric — #46's scheme travels inside its verdict files,
and the older two should be given the same treatment.

## Human-rated tasks

A human judgment is the least reproducible thing we produce, so it gets the most structure:

- **Per-rater files** — `benchmark/miss_taxonomy_46/<task>__<rater>.json`, never a single blob.
- **The rubric travels in the file.** Every export carries the verdict scheme it was made under,
  because a verdict whose definition is unknown cannot be reused or compared. `make_tagger.py` also
  keys its browser storage on the scheme, so changing a verdict's meaning starts a clean session
  instead of silently averaging two rubrics into one rate.
- **The item list is hashed.** The manifest records a `sha256` per crop and one `digest` over the
  whole set; a verdict file records that digest. `tag_results.py` refuses to compare raters whose
  digests disagree.
- **Rendering is deterministic, and this was tested rather than assumed** — re-rendering the #46
  gallery reproduced all **50/50 crops byte-identically**, which is what lets committed verdicts
  re-pair with a regenerated gallery instead of merely being presumed to.

## Run-book: the #46 miss-taxonomy rating task

Reproduces the tagging task exactly. **Requires the panos (blocker 2) and `.model_cache`
(blocker 1).** Everything else is committed.

```bash
# 1. Which silent misses did another model already explain? (needs .model_cache)
python scripts/analysis/silent_witness.py --json-out analysis_out/silent_witness.json

# 2. Render the crops for the ones nobody explained. (needs benchmark/<city>/panos)
python scripts/analysis/miss_gallery.py --bucket silent \
    --queue analysis_out/silent_witness.json --render analysis_out/gallery46_silent

# 3. Confirm you are looking at the same images the first rater saw.
#    The digest must equal the one in benchmark/miss_taxonomy_46/silent__manifest.json.
python -c "import json;print(json.load(open('analysis_out/gallery46_silent/manifest.json'))['digest'])"
#    -> 360b5ddf8751dcd0

# 4. Build the tagging page and rate all 50 crops.
python scripts/analysis/make_tagger.py analysis_out/gallery46_silent
#    open analysis_out/gallery46_silent/tagger.html ; export when done

# 5. Commit the export as a new rater, then compare.
cp ~/Downloads/verdicts.json benchmark/miss_taxonomy_46/silent__<name>.json
python scripts/analysis/tag_results.py
```

Step 5 prints per-rater distributions, pairwise agreement and Cohen's kappa, the specific
disagreements, and the resulting sourcing figure.

### What the first pass produced

One rater (`jonf`), 50/50 tagged, manifest digest `360b5ddf8751dcd0`.

| | near-field (n=13) | far-field (n=37) |
| :--- | ---: | ---: |
| visible | 7 | 34 |
| context-only | 1 | 2 |
| definition | 2 | — |
| unclear | 3 | 1 |

**No second rater has done this, so no agreement statistic exists** and the numbers below rest on
one person's judgment. That is the single largest caveat on the result, and running step 5 above
with a second rater is what would retire it.
