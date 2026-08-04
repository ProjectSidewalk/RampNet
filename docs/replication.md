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

Everything RampNet publishes lives under the [`projectsidewalk`
organisation](https://huggingface.co/projectsidewalk) and is collected at
[huggingface.co/collections/projectsidewalk/rampnet](https://huggingface.co/collections/projectsidewalk/rampnet).
Three repos exist today; three more are planned, and the collection is the index that makes them
findable together.

| repo | type | size | status |
| :--- | :--- | ---: | :--- |
| [`rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model) | model | — | ✅ published |
| [`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) | dataset | 463 GB | ✅ published — the Stage 1 *output*, 214k panoramas |
| [`rampnet-crop-model-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset) | dataset | 507 MB → 15.5 GB | ✅ published (1,212 round-2 crops); ⬜ round-1 crop set to add |
| **`rampnet-crop-model`** | model | 720.7 MB | ⬜ **planned — highest priority** |
| **`rampnet-stage1-inputs`** | dataset | 1.06 GB | ⬜ planned |
| **`rampnet-benchmark`** | dataset | 12.1 GB | ⬜ planned (#21) |

Total new upload is under 29 GB against the 463 GB already hosted — about +6%.

### Do we hit any HF limit? No, and it is not close

Checked against the [current limits](https://huggingface.co/docs/hub/repositories-recommendations)
rather than assumed:

| characteristic | HF limit | our maximum | headroom |
| :--- | :--- | :--- | ---: |
| single file size | 500 GB hard, <200 GB recommended | 669 MB (`New York - Streets.geojson`) | ~300× |
| entries per folder | **10,000 hard** | 125 (a per-city pano folder) | 80× |
| files per repo | <100k recommended | ~2,550 (`rampnet-benchmark`) | 39× |
| commit size | <100 files recommended | auto-split by `upload_folder` / `hf upload` | n/a |

Nothing in the plan is within an order of magnitude of a limit. The per-city subdirectory layout is
what keeps the folder count trivial, which is the one decision that could have gone wrong.

**The consideration that is real is the storage quota, not file sizes.** `projectsidewalk` is a
*free* organisation, and free organisations get **"best-effort"** public storage — HF's page notes
they run "mitigations in place to prevent abuse of free public storage" and ask that anything
beyond the first few gigabytes be of genuine community value. The existing 463 GB sits on a
human approval from July 2025 rather than an entitlement. +6% on top of that is noise and these
artifacts are squarely the kind of thing that approval was for, but the arrangement is informal,
and HF's own guidance points academic groups at Academia Hub or a Team plan for *guaranteed*
limits. Worth a courtesy note to `datasets@huggingface.co` on the existing thread when we publish —
they also asked to be told about public comms.

### 1. `rampnet-crop-model` — 720.7 MB, and it blocks everything else

**Stage 1 cannot run at all without this, published inputs or not.**
`dataset_generation/inference_isolator.py` hardcodes a path to the round-2 crop checkpoint; it is
the model that converts every government GPS coordinate into a pixel keypoint. It exists only on
lab storage. Publishing `location_data/` without it hands someone the inputs to a pipeline they
still cannot execute.

| file | bytes | sha256 |
| :--- | ---: | :--- |
| round 1, `ps_model/model/best_model.pth` | 360,358,458 | `00dba3948298a313…` |
| round 2, `ps_and_manual_model/best_model.pth` | 360,358,458 | `3fc00ad6b9ac2768…` |

Both rounds go up, not just round 2: round 1 is the initialisation for round 2, so without it the
second stage of crop training cannot be reproduced either. Note that
`ps_and_manual_model/ps_model.pth` is **byte-identical** to the round-1 checkpoint
(`00dba394…` both) — that is the "copy it here, renamed" step in the README, now verified by hash
rather than assumed, and it means only two distinct files need uploading.

`scripts/export_crop_model.py` builds and pushes the package, mirroring the Stage 2 exporter.
Nothing uploads without `--push`, the expected hashes are asserted before anything is copied, and
the copies are re-hashed afterwards so a push cannot ship bytes that were never verified:

```bash
python scripts/export_crop_model.py \
    --round1 <...>/ps_model/model/best_model.pth \
    --round2 <...>/ps_and_manual_model/best_model.pth \
    --out dist/rampnet-crop-model \
    --expect-round1-sha256 00dba3948298a313435b7c1955a2d4fccde43bc98c199e384ef197bf8b8cff49 \
    --expect-round2-sha256 3fc00ad6b9ac2768787b0262588b9bfa71ddd01d9f51109974e6ae377b9b520a
```

The card it renders is `scripts/hf_package/README.crop_model_card.template.md`.

### 2. `rampnet-stage1-inputs` — 1.06 GB

The inputs half. Order within it matters, because the manifests are the reproduction path and the
source files are the archive:

| contents | size | why HF rather than git |
| :--- | ---: | :--- |
| **manifests** — `finaldataset.jsonl` (64.3 MB), `dataset.jsonl` (59.1 MB), `all_locations.csv` (13.5 MB), `negativepanos.jsonl` (10.5 MB), `negativepanosSHORTENED.jsonl` (5.2 MB) | 152.5 MB | **upload first.** `finaldataset.jsonl` is the exact 219,170-panorama manifest `download_dataset.py` consumed, and `negativepanosSHORTENED.jsonl` the 43,834 negatives actually used — which is the *only* way to reproduce them, since the sampler is unseeded |
| `street_data/` raw downloads | 801.6 MB | the pristine originals behind the committed 18.7 MB derivative |
| `location_data/` originals | 71.8 MB | mirror of the committed copy, same sha256 — belt and braces against a git accident |
| `gov_provenance.csv` | 29.4 MB | optional; regenerates from the committed script, hash in §3 |

### 3. `rampnet-benchmark` — ~12 GB (#21)

| folder | size | why |
| :--- | ---: | :--- |
| `panos_4096x2048/<city>/` | **1.02 GB** | **what GT reviewers actually saw.** `gt_gallery.py` renders at the model's 4096×2048 and never native, so this — not the native archive — is what lets a second rater redo the pass |
| `panos_native/<city>/` | **10.89 GB** | the resolution experiment (#25) and any future re-render |
| `galleries/<city>/` | 244 MB | the #55 A/B crops reviewers saw (also regenerable, so belt-and-braces) |

Both pano figures are **measured**, by rendering all 1,109 panoramas across the 9 splits at
4096×2048 / JPEG q82 / BILINEAR — byte-for-byte the transform `gt_gallery.py` applies, deliberately
*not* using PIL's faster `draft()` DCT downscale, which would produce different pixels and defeat
the point.

**The derivative is a review-fidelity artifact, not a compression trick, and the per-split spread
shows it:**

| split | native | 4096×2048 | ratio | native size |
| :--- | ---: | ---: | ---: | :--- |
| gainesville, paterson, sao_paulo, bend, clovis | 1.5–2.3 GB each | 85–136 MB | 16–18× | 13312–16384 px |
| annapolis, richmond | 365, 381 MB | 103, 100 MB | 3.6–3.8× | 8000–12288 px |
| budapest_district5 | 217 MB | 136 MB | 1.6× | 4096–5760 px |
| **morgantown** | **96 MB** | **102 MB** | **0.9×** | 4096–5760 px |

Morgantown's derivative is *larger* than its native archive: those panoramas are already at model
resolution and heavily compressed, so re-encoding costs a JPEG generation and buys nothing. That is
not an argument to skip it — `gt_gallery.py` re-encodes at q82 regardless, so this **is** what the
reviewer saw — but it does mean the folder should not be described as "the small one". Publish
both; they answer different questions.

**Publish incrementally, don't wait for more cities.** Benchmark imagery is immutable once
fetched — `imagery_manifest.py --verify` confirmed all 984 reviewed panoramas unchanged since
review — and the things that *do* get revised, `records.jsonl` and `verdicts.json`, live in git.
So the set only ever grows, a new city is a new folder, and `hf upload` skips files whose hash
already matches. An 11th city costs one folder. What waiting *does* cost is real: the panoramas are
what block GT re-verification and the second-rater pass that Budapest's LOW-confidence ground truth
needs.

**Name the folders by resolution, not by consumer.** `panos_model_res/` was the earlier proposal
and it is a poor public name twice over: "res" is unexplained, and "model resolution" is a
*relative* label that silently becomes wrong when the model's input size changes — live risk given
#25 and #20. `panos_4096x2048/` states the fact, cannot rot, and a future input size just adds a
sibling folder instead of invalidating this one.

Layout is the one thing worth settling before the first upload, because restructuring means
*replacing* large blobs, which is the single expensive operation on HF (see churn, below).

### 4. Round-1 crop training data — 15 GB, into `rampnet-crop-model-dataset`

The Project Sidewalk crop set behind round 1 of the crop model — **15 GB, 27,710 files** — joins
the round-2 manual crops in
[`rampnet-crop-model-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset),
as **Parquet** under `round1_ps/`. Three reasons, none of which is "it would be illegal otherwise":
HF requires well-integrated formats for large datasets, Parquet is what makes the dataset viewer
work, and 27,710 loose files would consume a quarter of the `<100k files per repo` recommendation
for no benefit. The 10,000 limit is **per folder, not per repo**, so the existing
train/val/test × class subdirectory structure would in fact keep loose files legal — that is worth
stating plainly, because the constraint is a strong recommendation here rather than a wall.

The 1,212 round-2 crops stay exactly as they are — loose JPEGs under `test/`, which is fine at that
count. Converting them would orphan LFS versions and break existing paths on an already-published
repo for a cosmetic gain, so the repo will hold two shapes and the card should say why in a line.

Publishing this makes the paper's crop-model training set **available** but not **reproducible**:
`download_data.py` reads live from Project Sidewalk servers whose databases keep growing, so a
re-run builds a different set (§1). It can be downloaded, never regenerated — which is the reason
it needs publishing, not an argument against.

**Package the imagery as Parquet or WebDataset** — see the HF correspondence below. Embed the exact
bytes the reviewers saw (PNG stays PNG); `imagery_manifest.json` is what proves the round trip
preserved them. The handful of loose source files in `rampnet-stage1-inputs` are a different case:
they are documents to download and use as-is, not rows to iterate, so loose files are right there —
but the card must say plainly that `load_dataset()` will not work and give the `hf download`
command instead.

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

**Recommendation:** settle the directory layout once (`panos_native/<city>/`,
`panos_4096x2048/<city>/`, `galleries/<city>/`), then publish incrementally as splits land. The
thing genuinely worth getting right up front is the *layout*, not the timing.

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
