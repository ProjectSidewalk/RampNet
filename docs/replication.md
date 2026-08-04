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
| Stage 1 dataset | **463 GB** (test split ~44 GB) | HF `projectsidewalk/rampnet-dataset` | ✅ |
| `benchmark/model_detections/` (challenger detections) | 18.0 MB | **committed** ✅ | ✅ |
| **`location_data/` (the paper's government inventories)** | 71.8 MB | **committed** ✅ | ✅ |
| **`street_data/` derivative (what the pipeline actually reads)** | 18.7 MB | **committed** ✅ | ✅ |
| `street_data/` raw downloads (NY file alone is 669 MB) | 801 MB | git-ignored; HF #21 pending | ⚠️ superseded by the derivative |
| Stage 1 manifests (`all_locations.csv`, `dataset.jsonl`, `finaldataset.jsonl`, `negativepanos*.jsonl`) | 151 MB | **git-ignored; HF #21 pending** | ❌ **blocker** |
| Crop-model checkpoints (round 1 + round 2) | 720 MB | lab storage only | ❌ **blocker** |
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

### ✅ Resolved — the paper's government inventories are published

Stage 1 begins with three open-government curb ramp inventories. Until now none of them were in
git: `stage_one/dataset_generation/.gitignore` carried a `location_data/*` line, so `v1.0-iccv2025`
shipped the code that consumes them and not the files themselves. That is the worst class of
replication gap, because **it is not fixable later by re-downloading** — the city portals serve
current data and it drifts (§9 measures Bend at +8.7%), so a fresh download is a different
experiment, not a reproduction.

All three are now committed (71.8 MB), hash-pinned in
[`data_provenance.md` §3](data_provenance.md), and marked `binary` in `.gitattributes` so no
contributor's `core.autocrlf` can silently invalidate those hashes.

`scripts/analysis/gov_provenance.py` closes the second half of the gap — *which* government records
ended up in training. `combine_location_data.py` reduces the inventories to three columns and
shuffles, discarding every government primary key and even the city label, so `all_locations.csv`
alone cannot answer that. The script rebuilds the mapping by coordinate join and verifies it:
**276,071 / 276,071 rows resolved, 0 unmatched**, of which **156,712 (56.77%) were consumed by a
generated panorama**.

Two things it cannot recover, both documented beside the numbers rather than papered over: the
paper-era shuffle was **unseeded**, so the published row *order* is gone for good; and 8 coordinates
are shared by two records, making 16 rows ambiguous.

The street centrelines came along too, by a different route. At 801.6 MB they could never live in
git, but `generate_negative_panos.py` reads only the geometry and one name field, so
`scripts/build_street_derivative.py` cuts them **42.9× to 18.7 MB** and proves the sampled network
is unchanged with a consumer fingerprint. See [`data_provenance.md` §3](data_provenance.md).

**Priority note for #21.** Street data is *not* on the critical path for reproducing the paper:
`generate_negative_panos.py` is unseeded, so the negatives cannot be regenerated from any street
file. What reproduces them is the manifest — `negativepanosSHORTENED.jsonl` (5.2 MB, the 43,834
negatives actually used) and `finaldataset.jsonl` (64 MB, the 219,170-panorama manifest consumed by
`download_dataset.py`). Those are the higher-value, smaller publish and should go first.

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

✅ **Rubrics now live beside the data** — `benchmark/RUBRICS.md` documents all three passes
(GT verdict schema, the A/B spot-check, the miss taxonomy), extracted from the tools that produced
them. #46's rubric additionally travels *inside* each verdict file, which is the pattern the older
two should adopt.

### ✅ The "deleted" #55 galleries were never lost

Two cities (`richmond`, `bend`) had no gallery on disk. They are **derived artifacts**, not
originals: regenerating them from the committed `op_cache` plus the panoramas took one command
each, and `low_floor_sweep.py tagcheck` then confirmed **8/8 cities, 100% of committed tags still
resolve** against the regenerated crops.

```bash
python scripts/analysis/operating_point_curve.py gallery --city richmond \
    --op-threshold 0.25 --upper 0.55 --panos benchmark/richmond/panos \
    --out analysis_out/op/richmond_incremental_fp
python scripts/analysis/low_floor_sweep.py tagcheck        # 8/8 PASS
```

The irreplaceable half — the human tags — was committed all along. That is the design working:
**commit the judgments, regenerate the pixels.**

## Publishing plan for Hugging Face

Two artifacts are too large for git and belong on HF (issue #21).

| artifact | size | why |
| :--- | ---: | :--- |
| **model-resolution panos** (4096×2048) | ~1–2 GB | what GT verification actually needs — reviewers scan whole panoramas at the model's resolution, never native |
| **native-res panos** | 9.0 GB | the resolution experiment (#25) and any future re-render at higher fidelity |
| **#55 A/B galleries** | 244 MB | the crops the A/B reviewers saw (also regenerable, so belt-and-braces) |

**Package all three as Parquet or WebDataset** — see the HF correspondence below. Embed the exact
bytes the reviewers saw (PNG stays PNG); `imagery_manifest.json` is what proves the round trip
preserved them.

### Is churn actually a problem? Mostly no — publish sooner

The concern is reasonable but the mechanics are milder than they look. From HF's own repository
guidance:

- **Commits: no hard limit**, though the Hub's UX "starts to degrade after a few thousand commits."
  This project would add tens, not thousands.
- **Old LFS versions do keep consuming quota** after you replace a file — that is the one real
  churn cost. It is reclaimable with `super_squash_history` (destructive: history is lost), and
  note that deleting LFS *pointers* alone frees nothing.
- **`upload_folder` / `hf upload` skip files whose hash already matches**, so re-running an upload
  after adding one city re-transfers only that city.
- **Hard limit: 10k entries per folder** (and <100k files per repo recommended) — so use
  `panos/<city>/` subdirectories rather than one flat folder. With ~1,100 panos we are far under,
  but the per-city shape is right anyway.

**So the cost of churn is concentrated in one behaviour: repeatedly replacing the same large
blobs.** Structuring additively — a new folder per city, never a rewrite of existing ones — makes
updates nearly free, and means waiting to publish buys very little.

**Recommendation:** settle the directory layout once (`panos/<city>/`, `panos_model_res/<city>/`,
`galleries/<city>/`), then publish incrementally as splits land. The thing genuinely worth getting
right up front is the *layout*, not the timing.

### Storage capacity is not a constraint

`projectsidewalk` already hosts **`rampnet-dataset` at 463 GB / 200,000+ panoramas** — the ~44 GB
figure quoted elsewhere in this repo is the *test split*, not the whole dataset. Another ~9 GB is
under 2% of what is already published, and hosting at that scale was arranged with Hugging Face in
July 2025.

The conditions agreed then still apply and should shape how #21 is packaged:

1. **Use Parquet or WebDataset** for large datasets. `push_to_hub` on a `datasets.Dataset` handles
   the conversion and partitioning.
2. **Stay under the per-file / per-folder limits** — notably the hard limit of **10k entries per
   folder**, a real constraint for loose image files and a non-issue for Parquet.
3. **Host under an organisation**, not an individual account — hence `projectsidewalk`.
4. **Document it properly in the dataset card**, especially for a non-standard layout: if
   `load_dataset('org/repo')` does not just work, spell out how to download and load it.

### What this changes about our plan

**Publish the benchmark imagery as Parquet or WebDataset, not as loose JPEG/PNG folders.** That is
what HF asked for, it is what `rampnet-dataset` already uses, and it sidesteps the
10k-entries-per-folder limit entirely. A folder of 1,109 loose panos would work today but is the
shape they explicitly steered us away from, and it would not survive many more cities.

This supersedes the PNG-vs-JPEG framing above: inside Parquet or WebDataset the images are stored
as bytes, so the question becomes **what encoding to embed**, and the answer is unchanged — embed
the exact bytes the reviewers saw, whose hashes are pinned in
`benchmark/<city>/imagery_manifest.json`. That manifest is what lets a downloader confirm the
round trip through Parquet returned the same pixels.

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
- **The imagery every review was made against is pinned by hash.**
  `benchmark/<city>/imagery_manifest.json` records a sha256 and pixel size per panorama plus one
  digest per split — 206 KB describing 9 GB. This is what makes a verdict verifiable *after* the
  panos go to HF and come back: same pano id and filename is **not** evidence that the bytes are
  the ones the reviewer judged, because a re-fetch from Mapillary or GSV can return re-stitched or
  re-compressed pixels.

  ```bash
  python scripts/analysis/imagery_manifest.py --verify   # run this after downloading the archive
  ```

  *(All 8 reviewed splits verified OK on 2026-07-31 — 984 panoramas unchanged since review.)*

### Integrity coverage per pass

| pass | detections pinned | imagery pinned | rubric in the file |
| :--- | :--- | :--- | :--- |
| GT verification | ✅ `records.jsonl` (committed) | ✅ `imagery_manifest.json` | ⚠️ in `benchmark/RUBRICS.md`, not the verdict file |
| #55 A/B | ✅ `op_cache` + `tagcheck` re-resolves ids | ✅ `imagery_manifest.json` | ⚠️ in `benchmark/RUBRICS.md`, not the tag file |
| #46 miss taxonomy | ✅ committed caches | ✅ per-crop sha256 + manifest digest | ✅ inside every verdict file |

`tagcheck` and the imagery manifest answer different questions and both are needed: `tagcheck`
confirms the *detection coordinates* a tag refers to still exist, the manifest confirms the
*pixels* have not moved underneath them.

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
