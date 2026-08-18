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
| `analysis_out/usage_log.jsonl` (paid-run spend) | ~KB | **committed** ✅ | ⚠️ starts 2026-08-18; see the Claude section |
| `benchmark/miss_taxonomy_46/*.json` (human verdicts) | small | **committed** | ✅ |
| RampNet model weights | — | HF `projectsidewalk/rampnet-model` | ✅ |
| Stage 1 dataset | **463 GB** (test split ~44 GB) | HF `projectsidewalk/rampnet-dataset` | ✅ |
| `benchmark/model_detections/` (challenger detections) | 23.1 MB (114 files) | **committed** ✅ | ✅ |
| **`location_data/` (the paper's government inventories)** | 71.8 MB | **committed** ✅ | ✅ |
| **`street_data/` derivative (what the pipeline actually reads)** | 18.7 MB | **committed** ✅ | ✅ |
| `street_data/` raw downloads (NY file alone is 669 MB) | 801 MB | git-ignored; HF #21 pending | ⚠️ superseded by the derivative |
| Stage 1 manifests (`all_locations.csv`, `dataset.jsonl`, `finaldataset.jsonl`, `negativepanos*.jsonl`) | 152 MB | HF [`rampnet-stage1-inputs`](https://huggingface.co/datasets/projectsidewalk/rampnet-stage1-inputs) | ✅ |
| **Crop-model checkpoints** (rounds 1 + 2) | 720.7 MB | HF [`rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model) | ✅ |
| Round-1 crop training set (Project Sidewalk crops) | 13.4 GB | [`rampnet-crop-model-dataset-round1`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1) | ✅ published 2026-08-05 (§4) |
| **`benchmark/*/panos/` (benchmark panoramas)** | 11.41 GB | HF [`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) | ✅ |

### ✅ Resolved — the challenger detections are published

`fp_taxonomy.py`, `silent_witness.py`, `complementarity.py` and `null_recall.py` all read the
challenger detections. Those cost real GPU-hours on Hyak and paid API spend, and until
2026-07-31 they lived only in a git-ignored `.model_cache/`, so every number they produced was
reproducible on exactly one machine. Which models those are is `rampnet/roster.py`, not a list
in this sentence — the list here was one of the things that drifted.

`.model_cache` is fine as a *working* cache and hostile as a published artifact: 12,951
single-panorama shards keyed by an opaque SHA-1 of (label, signature, city, pano), unreadable
without reconstructing detector signatures. `scripts/analysis/export_model_cache.py` consolidates
it into human-readable files, one per (model, split), keyed by panorama id with the detector
signature recorded inside. As of 2026-08-18 that is **114 files, 23.1 MB**, and every one of
them belongs to a registered leg:

| what | files | where it is written up |
|---|---:|---|
| the standing zero-shot roster, ten splits each (two Gemini legs are absent on `manual_gold`) | 68 | the roster tables in [`model_comparison.md`](model_comparison.md) |
| `gemini-3.7-flash`, ten splits, published ahead of its write-up (#120) | 10 | §below |
| the supervised YOLO pano trio, ten splits each (#51) | 30 | [`model_comparison.md` §supervised baseline](model_comparison.md), and the [training record](../scripts/model_comparison/yolo_baseline/README.md) |
| the four annapolis Claude legs (#122) | 4 | [`model_comparison.md` §Claude](model_comparison.md) |
| the two Mapillary Vistas class-set arms, richmond only (#126) | 2 | [`model_comparison.md` §Vistas](model_comparison.md) |

`rampnet` is a row in every results table and has no file here: it is read from each bundle's
committed `records.jsonl` and carries no detector signature.

Every one of those files is in **canonical form** — byte-identical to what
`export_model_cache.py` would write today — which is the difference between a corpus that
*is* reproducible and one that merely scores the same. It was not, until 2026-08-18: the
`published_as` field arrived with the Claude legs, so the 108 files exported before it lacked
it and re-exporting any of them produced a diff with identical detections inside. Bringing
them up to date needed no cache, because the serialization is deterministic and each file's
published name was recoverable from the file itself:

```bash
python scripts/analysis/export_model_cache.py --canonicalize          # report
python scripts/analysis/export_model_cache.py --canonicalize --write  # apply
```

That is worth knowing for the next envelope change: **it is not a reason to go find the
machine that produced each leg.** Only a file whose published name cannot be derived from its
own contents needs a real re-export, and `--canonicalize` names those rather than guessing.
`test_every_published_file_is_in_canonical_form` now fails if the corpus drifts again.

Three tests hold that table up rather than trust it.
`test_the_ledger_count_matches_the_directory` asserts the count, and
`test_every_published_detections_file_belongs_to_a_registered_leg` asserts the stronger
property that nothing in the directory is unaccounted for; the third is the canonical-form
check above. The second one is why this table
exists: **the registry covered 78 of the then 112 files.** The YOLO arms and the Claude legs had
been run, scored, verified and written up, and the one place that is supposed to enumerate
every model said nothing about them — which is also what the unremarked 78 → 108 jump in the
drift list below actually was.

```bash
python scripts/analysis/export_model_cache.py --out benchmark/model_detections
python scripts/analysis/export_model_cache.py --verify     # exported == cached
```

`--verify` re-scores every split from both sources and asserts identical per-pano (TP, FP)
— **68/68 pairs on the default roster verified identical** — because a published artifact
that silently differs from what produced the paper's numbers is worse than none. It exits
non-zero when it had nothing to compare, so a green run cannot mean "found no cache".

The other 44 files were each verified the same way when they were published, but by their own
invocation against their own cache rather than by the default-roster one above — the YOLO
trio on makelab2 (30 files, identical to the producing cache; see the
[training record](../scripts/model_comparison/yolo_baseline/README.md)) and the four Claude
legs one at a time under `--publish-as` (see
[`model_comparison.md`](model_comparison.md)). Re-running `--verify` on any of them needs the
cache that produced it, which is machine-local by construction. What a clean clone can check
without any cache is that each file's recorded signature still matches the leg the registry
says it is: `test_each_published_file_names_the_leg_it_says_it_is`.

**Keep this count current when a leg is added.** It drifted three times (61 → 68 at the São
Paulo split, 68 → 78 at gemini-3.7-flash, 78 → 108 unremarked — that one was the YOLO trio, see the table above — and 108 → 112 at the Claude legs) before anyone noticed, and a
ledger that exists to keep the repo honest is the wrong document to let rot. It is now a
test rather than a promise.

#### The gemini-3.7-flash leg is published but off the default roster

`benchmark/model_detections/gemini-3.7-flash__*.json` covers **all ten splits**, including
`manual_gold` (1,000 panoramas, 0 uncached — that leg finished 2026-08-15 08:04 UTC, after the
first nine were exported). It is currently the only Gemini with a `manual_gold` file: the other
two are absent for a different reason, their `manual_gold` detections not being in the cache
under current keys (#20). Produced and verified with the model named explicitly:

```bash
python scripts/analysis/export_model_cache.py --models gemini:gemini-3.7-flash
python scripts/analysis/export_model_cache.py --verify --models gemini:gemini-3.7-flash
# -> compared 10 (model, split) pair(s); published detections score IDENTICALLY
```

`gemini:gemini-3.7-flash` is registered in `rampnet/roster.py` as published-but-off-roster,
so it is not in `CHALLENGERS` and not what `--models` defaults to. The two commands above are
therefore not optional detail: the default `--verify` skips these ten files entirely and
reports a clean pass without opening them, and `fp_taxonomy.py` needs the same `--models`
flag. Since #122, `silent_witness.py` takes `--models` as well, so it can be pointed at any
pool without editing a tuple.

Promoting the leg to the standing roster is one field in the registry (`standing=True`) plus a
re-run of `fp_taxonomy.py` and `null_recall.py`, whose committed JSON would change — a re-run,
not an edit, open until the write-up lands in `docs/model_comparison.md`. **What promotion no
longer touches is the #46 human pass.** `silent_witness.py` defaults to
`roster.WITNESS_POOL_46`, the pool frozen at the state the pass was rated under, and both
committed artifacts record the pool they ran over.

#### The four Claude legs are published, annapolis only, one file per effort level

`benchmark/model_detections/claude-{sonnet,opus}-5-effort-{low,high}__annapolis.json` — four
files, 125 panoramas each, 0 uncached (#122). Only annapolis was run; the other nine splits
are a stated gap.

These need one flag the other legs do not. **Effort is part of the cache signature, so one
model id is two different legs**, and both would export to `claude-sonnet-5__annapolis.json`
— the second silently overwriting the first, leaving a file that still looks complete and
still passes `--verify` against whichever leg was written last. `--publish-as` names the
published file without touching the cache label (which has to stay the bare model id,
because it is baked into keys that have already been paid for), and `export_model_cache.py`
now refuses outright to overwrite a file whose recorded signature differs from the run's:

```bash
for m in claude-sonnet-5 claude-opus-5; do for e in low high; do
  python scripts/analysis/export_model_cache.py --splits annapolis \
      --models claude:$m --claude-effort $e --publish-as $m-effort-$e
  python scripts/analysis/export_model_cache.py --verify --splits annapolis \
      --models claude:$m --claude-effort $e --publish-as $m-effort-$e
done; done
# -> 4 x "compared 1 (model, split) pair(s); published detections score IDENTICALLY"
```

Neither Claude spec is in `CHALLENGERS`, so the same caveat as `gemini-3.7-flash` applies:
the default `--verify` never opens these files, and `fp_taxonomy.py` / `silent_witness.py`
cannot reach them without the explicit `--models`.

Unlike every other published leg, these four can also be checked with **no cache and no
credentials at all** — `tests/test_claude_annapolis_leg.py` recomputes the entire published
result table from the committed detections plus the committed annapolis bundle, and runs in
CI. That is the strongest form this ledger's promise can take, and it is the pattern worth
copying to the other legs.

**Known gap, unrecoverable: the four legs' token counts were never written to
`analysis_out/usage_log.jsonl`.** The $28.82 figure and the per-leg costs quoted in
`docs/model_comparison.md` come from the runs' console output. They cannot be back-filled,
because a re-run reads the detection cache, makes zero API calls and therefore has no usage
to record — the cost record is the one artifact here that is write-once. Only the
2026-08-18 single-panorama re-run ($0.03) is in the log. `compare.report_usage` now prints
a loud warning when a leg that spent money finishes with no log destination.

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

### ✅ Resolved — the benchmark panoramas are published (#21)

This was the last blocker: the imagery half of the benchmark existed only on lab machines, so
`gt_gallery.py`, `miss_gallery.py`, `fp_gallery.py` and any re-rating were unobtainable by anyone
else. It is now
[`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) — 11.41 GB,
Parquet, **four** configs: `records` (the ground truth, 9 splits), `native` (9), `4096x2048` (9)
and `galleries` (**8** — Budapest was not part of the #55 A/B). Each config is one Parquet per
split at `data/<config>/<city>.parquet`; `load_dataset` reads them by config name, so the paths are
an implementation detail rather than something to navigate.

```python
from datasets import load_dataset
ds = load_dataset("projectsidewalk/rampnet-benchmark", "4096x2048", split="budapest_district5")
```

**The second-rater pass on Budapest is therefore unblocked.** Its ground truth was reviewed at LOW
confidence, and the imagery a reviewer needs is the `4096x2048` config — `gt_gallery.py` renders at
that size and never native. What remains blocking a second rater is a *person*, not an artifact.

> **Outstanding: `galleries` needs one re-push.** As first published, a gallery row's `pano_id`
> column held the *crop tag* — `low_floor_sweep.py` names each crop `{pano}_{x:.5f}_{y:.5f}` — so
> the join to `records` documented on the card silently returned zero rows. The exporter now
> writes `crop_id` (the tag), `pano_id` (parsed back out of it) and the detection's
> `x_normalized` / `y_normalized`, which makes that join work. **The fix is in the code but not yet
> on the Hub**: re-running `export_benchmark.py build --galleries ...` and pushing the 244 MB
> `galleries` config is what lands it. `records`, `native` and `4096x2048` are unaffected.

## Every manual-review task, and what it would take to redo it

Three distinct human passes exist. All three produce committed judgments; they differ in whether
someone else can **redo the pass**.

| task | judgments | what the reviewer saw | redoable by someone else? |
| :--- | :--- | :--- | :--- |
| **GT verification** (9 splits) | `benchmark/<city>/verdicts.json` ✅ | whole panoramas at 4096×2048 via `scripts/gt_gallery.py` | ✅ **yes** — `rampnet-benchmark`, config `4096x2048` |
| **#55 incremental-FP A/B** (8 splits) | `benchmark/<city>/incremental_fp_tags.json` ✅ | crops from `low_floor_sweep.py gallery` | ✅ **yes** — `rampnet-benchmark`, config `galleries` (the exact 314 crops) |
| **#46 miss taxonomy** (1 split-set) | `benchmark/miss_taxonomy_46/silent__jonf.json` ✅ | **crops committed** (15 MB) | ✅ **yes, from git alone** |

**All three human passes are now redoable by someone else.** That was not true this morning: two of
the three needed imagery that existed only on lab machines. The judgments and rubrics were already
committed; publishing the pixels is what closed the loop.

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
organisation](https://huggingface.co/projectsidewalk) and is indexed by the
[RampNet collection](https://huggingface.co/collections/projectsidewalk/rampnet), which is what
makes the six artifacts findable together.

> **Creating a repo does not add it to the collection.** Collection membership is a separate API
> call and none of the exporters do it — a repo published without this step is reachable only by
> direct URL or org listing, which is exactly the discoverability gap publishing was meant to
> close. After any `--push`, run:
>
> ```python
> from huggingface_hub import HfApi
> HfApi().add_collection_item(
>     collection_slug="projectsidewalk/rampnet-6871b77b1add07bdfecfcd5c",
>     item_id="projectsidewalk/<new-repo>", item_type="dataset",   # or "model"
>     note="<one line on what it is>", exists_ok=True)
> ```
>
> The slug carries an id suffix; the bare `projectsidewalk/rampnet` URL will not resolve in the
> API. Find it with `HfApi().list_collections(owner="projectsidewalk")`.

| repo | type | size | status |
| :--- | :--- | ---: | :--- |
| [`rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model) | model | — | ✅ published |
| [`rampnet-dataset`](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) | dataset | 463 GB | ✅ published — the Stage 1 *output*, 214k panoramas |
| [`rampnet-crop-model-dataset-round2`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2) | dataset | 507 MB | ✅ published (1,212 round-2 crops) — renamed from `rampnet-crop-model-dataset` 2026-08-05, old id redirects; round 1 cannot join it (§4) |
| [`rampnet-crop-model-dataset-round1`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1) | dataset | 13.37 GB | ✅ **published 2026-08-05** (§4) |
| [`rampnet-crop-model`](https://huggingface.co/projectsidewalk/rampnet-crop-model) | model | 720.7 MB | ✅ **published 2026-08-04** |
| [`rampnet-stage1-inputs`](https://huggingface.co/datasets/projectsidewalk/rampnet-stage1-inputs) | dataset | 1.06 GB | ✅ **published 2026-08-04** |
| [`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) | dataset | 11.41 GB | ✅ **published 2026-08-04** (#21) |

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

### 1. ✅ `rampnet-crop-model` — 720.7 MB, published

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

### 2. ✅ `rampnet-stage1-inputs` — 1.06 GB, published

The inputs half. Order within it matters, because the manifests are the reproduction path and the
source files are the archive:

| contents | size | why HF rather than git |
| :--- | ---: | :--- |
| **manifests** — `finaldataset.jsonl` (64.3 MB), `dataset.jsonl` (59.1 MB), `all_locations.csv` (13.5 MB), `negativepanos.jsonl` (10.5 MB), `negativepanosSHORTENED.jsonl` (5.2 MB) | 152.5 MB | **upload first.** `finaldataset.jsonl` is the exact 219,170-panorama manifest `download_dataset.py` consumed, and `negativepanosSHORTENED.jsonl` the 43,834 negatives actually used — which is the *only* way to reproduce them, since the sampler is unseeded |
| `street_data/` raw downloads | 801.6 MB | the pristine originals behind the committed 18.7 MB derivative |
| `location_data/` originals | 71.8 MB | mirror of the committed copy, same sha256 — belt and braces against a git accident |
| `gov_provenance.csv` | 29.4 MB | optional; regenerates from the committed script, hash in §3 |

### 3. ✅ `rampnet-benchmark` — 11.41 GB, published (#21)

| config | size | why |
| :--- | ---: | :--- |
| `records` | a few MB | **the ground truth** — per-pano metadata, detections with their human verdict, reviewer-marked misses. Its own config so a verdict correction never rewrites an image blob |
| `4096x2048` | **1.02 GB** | **what GT reviewers actually saw.** `gt_gallery.py` renders at the model's 4096×2048 and never native, so this — not the native archive — is what lets a second rater redo the pass |
| `native` | **10.89 GB** | the resolution experiment (#25) and any future re-render |
| `galleries` | 244 MB | the #55 A/B crops reviewers saw, 8 splits (also regenerable, so belt-and-braces) |

(The plan below was written as loose `panos_native/<city>/` folders; it shipped as Parquet, one
file per split at `data/<config>/<city>.parquet`, for the reasons in "Packaged as Parquet" further
down. The naming argument survived the change intact — it is now the *config* names that are
resolutions rather than consumers.)

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

**Name them by resolution, not by consumer.** `panos_model_res/` was the earlier proposal and it is
a poor public name twice over: "res" is unexplained, and "model resolution" is a *relative* label
that silently becomes wrong when the model's input size changes — live risk given #25 and #20.
`4096x2048` states the fact, cannot rot, and a future input size just adds a sibling config instead
of invalidating this one. This is the one design decision here that shipped exactly as written.

Layout is the one thing worth settling before the first upload, because restructuring means
*replacing* large blobs, which is the single expensive operation on HF (see churn, below).

**Packaged as Parquet, not loose image folders**, which is what HF asked for in July 2025 and what
makes the dataset viewer work. `scripts/export_benchmark.py` streams each split into one Parquet
per (config, city), embedding the **exact source bytes** — rows carry `image` as `{bytes, path}`,
so nothing is re-encoded on write, and each row also carries the `sha256` of its own bytes.

```bash
python scripts/export_benchmark.py build  --benchmark benchmark \
    --panos-4096 <rendered dir> --galleries analysis_out/op --out dist/rampnet-benchmark
python scripts/export_benchmark.py verify --out dist/rampnet-benchmark
python scripts/export_benchmark.py push   --out dist/rampnet-benchmark \
    --repo-id projectsidewalk/rampnet-benchmark
```

`verify` re-hashes every embedded image straight out of the Parquet, so "the round trip preserved
the pixels" is checked rather than assumed — the point of the whole artifact is that it is what a
reviewer's eyes were on. That is a *different* check from
`benchmark/<city>/imagery_manifest.json`, which pins the bytes as they were at review time; both
are needed, and the card says so.

**Labels stay in git.** `records.jsonl` and `verdicts.json` are revisable and belong under version
control; imagery is immutable once fetched. Keeping them apart is what makes this repo purely
additive.

### 4. Round-1 crop training data — 13.4 GB, as its own repo

The Project Sidewalk crop set behind round 1 of the crop model — **27,704 crops, 13.37 GB as
Parquet** — ships as its own dataset repo, `rampnet-crop-model-dataset-round1`. Parquet because HF
asks for it on large datasets and it is what makes the viewer work; not because loose files would
be illegal (the 10,000 limit is **per folder**, so train/val/test subdirectories would have kept
loose JPEGs legal — a strong recommendation here, not a wall).

**This section used to say the opposite**, and the correction is the useful part.

The plan was to put round 1 into the existing round-2 repo (then named
`rampnet-crop-model-dataset`; see the rename below)
under `round1_ps/`, leaving the 1,212 round-2 JPEGs loose, so both training rounds lived at one
address. **That is impossible, and it fails silently.** `datasets` infers **one builder module per
repository**, from the default config, and applies it to every config in the repo:

| default config | round-2 (JPEGs) | round-1 (Parquet) |
| :--- | :--- | :--- |
| round 2 | `Imagefolder`, works | `Imagefolder` — **0 rows**, no error until you request a split |
| round 1 | `Parquet` | `Parquet` — the JPEG config breaks instead |

There is no arrangement that works. The Parquet config's `data_files` patterns resolve perfectly —
the shards are found — and then the imagefolder builder looks inside them for images, finds none,
and reports an empty split. `scripts/analysis/hf_config_mixing_check.py` demonstrates both
directions plus a parquet-only control that works, against throwaway repos it deletes afterwards
(verified 2026-08-05, `datasets` 5.0.0 / `huggingface_hub` 1.24.0).

Unifying would therefore mean converting the 1,212 round-2 JPEGs to Parquet and deleting them —
replacing a published layout and breaking every existing direct file path. This project has
already drawn that line, in the safetensors commit: *the "already up there, live with it" argument
applies to replacing blobs, not to adding a file.* Adding a repo is additive; rewriting the
round-2 repo is not.

Separate repos initially left a genuine naming wart — the unmarked `rampnet-crop-model-dataset`
was round **2**, while the *marked* repo was round 1. **Resolved 2026-08-05 by renaming the
round-2 repo to
[`rampnet-crop-model-dataset-round2`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2).**
A rename was worth doing only if it breaks nobody, so that was measured first:
`scripts/analysis/hf_move_redirect_check.py` proves `HfApi.move_repo` leaves a redirect that keeps
the old id working at every layer a user touches — `dataset_info`, `/resolve/` file URLs, and
`load_dataset(OLD_ID)` — and the real rename was then verified content-identical (same git sha,
all 1,214 files, downloads/likes carried). Both cards additionally carry the same two-row table
naming which repo is which round.

One obligation the redirect leaves behind: **the org must never create a new repo named
`rampnet-crop-model-dataset`** — a new repo at the freed name would shadow the redirect, and every
stale link would silently point at the wrong dataset.

**Two things the export found that the card now states, because both would mislead a user:**

- The token each filename starts with is **not a panorama id**. `download_data.py:277` builds it
  with `random.choices(alphabet, k=8)`, so the source panorama is not recoverable from this
  artifact at all. Published as `crop_uid`; naming it `pano_id` would have invited a join against
  `rampnet-dataset` that silently returns nothing.
- The labels carry a **3.1% x-axis mismatch**. `train.py` scales both keypoint axes by `0.5`, but
  `transforms.Resize((1024, 352))` scales the image by 0.5 on y and 352/683 = 0.5154 on x — so a
  label drifts left of its ramp in proportion to x, up to ~10.5 px at the right edge. Coordinates
  are published verbatim rather than normalised precisely so this stays visible and the reader
  chooses what to do about it.

**Published 2026-08-05** at
[`rampnet-crop-model-dataset-round1`](https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1),
from the artifact built by `scripts/export_crop_dataset.py` at `6a5b554`, and verified at every
hop rather than assumed:

- the source tar's sha256 (`7fd446c1…8ec9`) matches on klone
  (`/gscratch/makelab/jonf/round1_ps_crops.tar`, job 38137571) and after transfer;
- pre-upload, `verify` re-hashed all 27,704 crops out of the shards against their recorded
  sha256 — byte-identical, 35,757 keypoints cross-checked by an independent filename re-parse;
- post-upload, each of the 11 shards matches its **Hub LFS sha256** against the local file (no
  re-download needed — HF stores every LFS object under its content hash), the README round-trips
  byte-identical, the remote parquet footers count 19,392 / 4,155 / 4,157 = 27,704 rows, and a
  live-streamed row's image bytes re-hash to its stored `sha256`.

The repo is in the RampNet collection (7 items), and the dataset viewer is live.

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

**Recommendation, as shipped:** settle the layout once — one Parquet per split at
`data/<config>/<city>.parquet`, configs `records` / `native` / `4096x2048` / `galleries` — then
publish incrementally as splits land. The thing genuinely worth getting right up front is the
*layout*, not the timing. An 11th city adds four files and rewrites none.

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

Reproduces the tagging task exactly. **Everything it needs is now obtainable**: the challenger
detections are committed under `benchmark/model_detections/`, and the panoramas are on HF at
[`rampnet-benchmark`](https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark) (use the
`4096x2048` config — `miss_gallery.py` crops at model resolution). The #46 crops themselves are
committed, so steps 3–5 need no imagery at all.

```bash
# 1. Which silent misses did another model already explain? (needs .model_cache)
python scripts/analysis/silent_witness.py --json-out analysis_out/silent_witness.json

# 2. Render the crops for the ones nobody explained. (needs benchmark/<city>/panos)
python scripts/analysis/miss_gallery.py --bucket silent \
    --queue analysis_out/silent_witness.json --render analysis_out/gallery46_silent

# 3. Confirm you are looking at the same images the first rater saw.
#    It must equal the digest in benchmark/miss_taxonomy_46/silent_gallery/manifest.json,
#    which is also recorded as `manifest_digest` inside every per-rater verdict file.
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
