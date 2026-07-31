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
| **`.model_cache/` (challenger detections)** | **18.8 MB** | **git-ignored, unpublished** | ❌ **blocker** |
| **`benchmark/*/panos/` (native-res panoramas)** | **9.0 GB** | **git-ignored; HF #21 pending** | ❌ **blocker** |

### Blocker 1 — `.model_cache` is 18.8 MB and blocks three scripts for no good reason

`fp_taxonomy.py`, `silent_witness.py`, `complementarity.py` and `null_recall.py` all read the
cached challenger detections (Gemini ×2, Qwen ×2, Molmo, OWLv2, Grounding DINO). Those detections
cost real GPU-hours and API spend to produce, and **nobody outside this machine can obtain them.**

It is 12,951 files totalling **18.8 MB** — the same size class as the `op_cache` we already commit,
and the same argument applies: it is image-free, it makes several documented numbers re-derivable
on CPU, and re-creating it requires a GPU cluster and paid API calls.

The file *count* is the only real objection (12,951 tiny shards make for slow checkouts).
**Recommended fix: consolidate to one JSON per (model, split) — about 60 files — and commit.**
Until that happens, every number in `docs/model_comparison.md`'s FP taxonomy and every witness
figure in §0b of `docs/curb_ramp_data_sourcing.md` is reproducible **only on this machine.** That
is stated here rather than left to be discovered.

### Blocker 2 — the panoramas are 9.0 GB and only HF can carry them

`benchmark/README.md` already says the native-resolution panos "are archived separately and
published to HF (#21); they are intentionally not in git". **#21 is still open**, so the imagery
half of the benchmark is currently unobtainable by anyone else. That blocks `miss_gallery.py`,
`fp_gallery.py`, `gt_gallery.py` and any re-rating of the #46 tagging task.

Nothing in this repo can fix that; it needs #21 to land.

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
