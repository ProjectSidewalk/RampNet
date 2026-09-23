# Curb-ramp tag benchmark of record (#86, plan item 2)

Item 2 of [`rampnet2_plan.md`](rampnet2_plan.md) §4. The question: what does the ASSETS'24
curb-ramp tagger actually score on its own test set, and how much of that comes from the
test set sharing panoramas with train?

Everything here re-derives from a clean clone: the numbers in §2 and §3 come from committed
per-label predictions through `scripts/analysis/tag_benchmark_86.py score` on CPU, and the
GPU steps that produced those predictions are in §6 as exact commands. The one exception is
stated where it applies: the §5 interim numbers also need the arms' epoch-4 checkpoints, which
are unpublished and live on makelab2.

## 1. Result

| test set | n labels (panos) | mAP | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|
| Published (ASSETS'24, DINOv2-B, validated) | 2,183 | 0.34 | 0.67 | 0.31 |
| **Reproduction, tagger's `evaluate.py` unmodified** | 2,183 | **0.34** | **0.67** | **0.31** |
| Same checkpoint, all of `test.csv` (this repo's scorer) | 2,183 (1,869) | 0.341 [0.324, 0.371] | 0.665 [0.648, 0.682] | 0.315 [0.289, 0.340] |
| **Leak-free**: pano not in train | 957 (889) | **0.341** [0.314, 0.386] | 0.672 [0.646, 0.696] | 0.296 [0.256, 0.336] |
| Leaked: pano also in train | 1,226 (980) | 0.363 [0.336, 0.414] | 0.659 [0.634, 0.681] | 0.329 [0.296, 0.358] |
| Leak-free and no train label within 10 m | 759 (717) | 0.339 [0.311, 0.398] | 0.683 [0.658, 0.707] | 0.306 [0.256, 0.357] |

Brackets are 95 % bootstrap intervals, resampling panoramas (1,000 draws, seed 86). Every row
averages the same eight tags (the tags with at least 10 positives in the full test set, the
tagger's rule); F1 is at the tagger's threshold of 0.3.

**Not scored: expert-validate as a second test set** (2,432 labels, plan item 2's last clause).
Those labels have no crop in the HF framing. PR #177's crop cutter has a `viewport` mode that
reproduces that framing (median NCC 0.780 against 200 HF crops); the remaining step is to merge
#177, cut those labels with `--fov viewport`, run `crop.py`'s 640 px box over them, and score
them here (§8).

- **The published numbers reproduce exactly.** The released checkpoint through the tagger's
  own `evaluate.py` prints mAP 0.34 / micro-F1 0.67 / macro-F1 0.31, and the per-tag AP from
  its stats JSON matches this repo's scorer to within 4e-6 on every tag (mAP 0.3408 both ways).
- **The label-level split leaks heavily, and there is no measurable inflation from it.**
  1,226 of 2,183 test labels (56 %) sit on a panorama that also carries train labels. The
  direct test compares the published number with its own leak-free part on the same bootstrap
  draws (paired, 1,000 draws, seed 86). Full minus leak-free: mAP −0.000 [−0.028, +0.023], micro-F1
  −0.007 [−0.025, +0.010], macro-F1 +0.019 [−0.007, +0.048]. The 95 % interval of the inflation of the published mAP is −0.03 to +0.02:
  no measurable inflation, and at most about 0.02 mAP. The secondary contrast, leaked minus
  leak-free on disjoint panoramas: mAP +0.022 [−0.028, +0.074], macro-F1 +0.033 [−0.016,
  +0.078], micro-F1 −0.013 [−0.048, +0.019]. None of these intervals excludes zero.
- **So within that bound the published 0.34 is also this checkpoint's leak-free number**, with
  the caveats in §4. The weak tags are weak on both sides of the split: outside "missing tactile warning"
  (AP 0.98) and "surface problem" (0.62), every averaged tag is at AP 0.30 or below.

Why a leak might not matter much here: a crop is a 640 px box around one label, so two labels
on one panorama usually show different ramps, and the tags are judgments about each ramp. The
retrained arms in §5 test this directly: if the leak were doing work, a model trained on a
pano-grouped split would score lower on its held-out panos than the control does on its own.

## 2. Per-tag AP (released checkpoint)

| tag | full n+ | full AP | leak-free n+ | leak-free AP | leaked n+ | leaked AP | leak-free, >10 m n+ | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| missing tactile warning | 872 | 0.982 | 453 | 0.985 | 419 | 0.978 | 379 | 0.986 |
| surface problem | 267 | 0.624 | 104 | 0.599 | 163 | 0.642 | 74 | 0.602 |
| points into traffic | 297 | 0.295 | 145 | 0.303 | 152 | 0.304 | 113 | 0.272 |
| narrow | 152 | 0.284 | 74 | 0.275 | 78 | 0.304 | 57 | 0.226 |
| pooled water | 42 | 0.175 | 22 | 0.183 | 20 | 0.264 | 13 | 0.239 |
| not level with street | 65 | 0.162 | 32 | 0.206 | 33 | 0.127 | 27 | 0.229 |
| not enough landing space | 84 | 0.156 | 44 | 0.100 | 40 | 0.239 | 34 | 0.099 |
| steep | 29 | 0.049 | 15 | 0.075 | 14 | 0.043 | 13 | 0.060 |
| parallel lines | 0 | — | 0 | — | 0 | — | 0 | — |
| tactile warning | 0 | — | 0 | — | 0 | — | 0 | — |

Per-tag AP on a subset carries that subset's base rate: AP is not comparable across columns
the way a recall is, and on a tag with 13–22 positives a single ramp moves AP by several
points. Read the per-tag columns as "no tag collapses when the leak is removed", not as a
per-tag leak effect. The test set has **no** positives for "tactile warning" and "parallel
lines" (train has 471 and 0), so the benchmark says nothing about either. The 471 train
positives for "tactile warning" (Chicago 383, Oradell 88) are the same labels as the audit's 471
of 543 removals (`docs/ps_supervision_audit.md` §8): in the live `rawLabels` pulled 2026-09-21,
0 of the 471 still carry the tag. The released model was trained on a tag that has since been
fully retracted.

## 3. How the leak was measured

- **Label to panorama.** The HF CSVs carry no panorama id. `tag_benchmark_86.py labels` parses
  (city, label_id) out of each crop filename (`gsv-walla_walla-124-CurbRamp.png` →
  `walla-walla:124`; the filenames use an underscore), maps the HF city token to the
  deployment id, and joins each label to its `pano_id` and lat/lng from Project Sidewalk's
  public `rawLabels` API (`/v3/api/rawLabels?filetype=csv&labelType=CurbRamp`, pulled
  2026-09-21 by PR #175's `ps_supervision_audit.py fetch`). The join is on the composite
  `(city, label_id)` with `validate="one_to_one"`. Result, committed:
  `analysis_out/tag_benchmark_86/hf_curbramp_labels.csv` (10,857 rows).
- **Four train labels have no live row** (`seattle-wa:275934`, `:496`, `:99143`, `:499`, deleted
  since the 2024 freeze). All four are train, so no test label is unresolved; they cannot mark
  a test label as leaked, which could only under-count the leak by at most four panoramas.
- **Leaked** = the test label's panorama carries at least one train label. **Leak-free** = it
  does not. This reproduces PR #175's count: 980 of 6,295 panoramas are on both sides.
- **The stricter row** also drops leak-free labels with a train label of the same city within
  10 m (198 of 957), which catches the same ramp or the same corner seen from a neighbouring
  panorama. The median test label is 9.5 m from its nearest train label.

## 4. Caveats that travel with these numbers

- **Test-set subsets, not a retrain.** §1's leak-free row is the *same* checkpoint on a subset
  of its test set. That answers "is 0.34 inflated by near-duplicates of training crops"; it
  does not answer "what would this recipe score if trained without the leak" (§5 does).
- **Subsets differ in composition, not only in leakage.** Leaked labels come from panoramas
  with several labels (busy corners), and the share varies by city: 65–77 % of Chicago,
  Columbus, Newberg and Oradell test labels are leaked, against 35 % of Seattle's. The gap in §1 mixes the leak with that.
- **One threshold, the tagger's.** F1 is at 0.3 for every tag, as published; per-tag best-F1
  thresholds are in `released_scores.json` but not quoted.
- **Numerics.** The run used torch 2.4.1+cu121 without xformers; the paper environment pinned
  torch 2.0.0 and xformers 0.0.18. The headline and per-tag AP match to 4e-6, so the attention
  kernel does not matter at the reported precision.
- **Stored scores are logits.** The first scoring pass stored rounded sigmoid scores and came
  out 0.002 mAP low, lower on every tag: at 6 decimals, 867 of 2,183 "missing tactile warning" scores
  rounded to exactly 0 and tied. The committed predictions are logits (5 dp); the scorer applies
  the sigmoid in float32, as `evaluate.py` does. A batch-1 vs batch-64 check ruled out batching
  (largest score difference 2.9e-5).

## 5. Retraining the recipe on leak-free splits (interim: epoch index 4, i.e. 5 of 100 epochs)

The tagger's DINOv2 recipe (`notebooks/dino-trainer.ipynb`: full fine-tune of DINOv2-B with 4
registers, Adam lr 1e-6, batch 4, BCE, no augmentation, 100 epochs, checkpoint kept by best
*training* exact-match accuracy) is running on three splits of the same 10,857 labels:

| arm | split | train / test labels | committed split file |
|---|---|---:|---|
| control | the published HF split | 8,674 / 2,183 | (HF `csv/`) |
| pano | seeded pano-grouped re-split: no panorama on both sides | 8,660 / 2,197 | `resplit_pano_grouped_seed86.csv` |
| cell | seeded ~100 m block-grouped re-split: a panorama goes wherever its block goes | 8,638 / 2,219 | `resplit_cell100m_seed86.csv` |

Both re-splits draw panoramas (or blocks) per city until each city's test count reaches its
count in the published split, so city mix and test size match (seed 86). The pano-grouped split
still leaves 692 of its 2,197 test labels within 10 m of a train label (a neighbouring panorama
of the same corner); the block-grouped split cuts that to 70 of 2,219. For comparison, the published
split has 1,128 of its 2,183 test labels within 10 m of a train label (930 of them also on a
shared panorama).

Epochs are counted from 0 in the logs and file names: "epoch index 4" / `ep4` is the checkpoint
after 5 epochs. **Status:** all three arms finished epoch index 9 (10 epochs) at about
2026-09-23 03:45:42 UTC, 6,407–6,410 s after launch (from the mtime of `train_<arm>/train_log.csv`
and the launch line of `train_<arm>.log`). An epoch takes ~605 s with the three arms sharing the
A40 (630–690 s while the epoch-4 inference also ran), so 100 epochs is ~17 h per arm and the runs
should end around 2026-09-23 19:00 UTC, followed by ~6 min per arm of inference on all 10,857
crops. The numbers below are each arm's `best.pth` as of epoch index 4, scored on its own
held-out labels, same eight tags as §1:

| arm, epoch index 4 | test n (panos) | mAP | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|
| released checkpoint, for reference | 2,183 (1,869) | 0.341 [0.324, 0.371] | 0.665 [0.648, 0.682] | 0.315 [0.289, 0.340] |
| control, all of `test.csv` | 2,183 (1,869) | 0.337 [0.319, 0.372] | 0.661 [0.644, 0.678] | 0.304 [0.282, 0.325] |
| control, leak-free subset | 957 (889) | 0.353 [0.329, 0.395] | 0.678 [0.654, 0.699] | 0.294 [0.263, 0.323] |
| control, leaked subset | 1,226 (980) | 0.337 [0.311, 0.387] | 0.645 [0.617, 0.668] | 0.314 [0.280, 0.342] |
| pano-grouped, its held-out panos | 2,197 (1,283) | 0.360 [0.341, 0.388] | 0.655 [0.635, 0.674] | 0.276 [0.258, 0.294] |
| block-grouped, its held-out blocks | 2,219 (1,256) | 0.373 [0.350, 0.406] | 0.655 [0.637, 0.673] | 0.301 [0.281, 0.319] |

- **Interim, epoch index 4, unpaired: no leak-free arm scores below the control.** The control
  arm is within 0.004 mAP of the released checkpoint, and the two leak-free arms score 0.360
  and 0.373 on held-out panoramas and blocks. Whether a model trained without the leak scores
  no lower is the question the epoch-100 read answers; this read cannot settle it, for the two
  reasons below. Inside the control arm, full minus leak-free is mAP −0.016 [−0.045, +0.010] (paired, as
  in §1); leaked minus leak-free is mAP −0.017 [−0.068, +0.032] and micro-F1 −0.032
  [−0.067, −0.001]. That micro-F1 interval sits on zero, and whether it excludes it depends on
  the bootstrap seed (seed 12345 gives [−0.067, +0.002]), so it supports no reading.
- **Caveat: the arms' test sets are different labels.** The re-splits hold out different ramps,
  with different tag base rates (e.g. "narrow" has 152 test positives in the published split,
  178 and 177 in the re-splits), so 0.360 vs 0.337 is not a paired comparison. Labels that are
  test in all three splits and leak-free in the published one number only 39, too few to score.
- **Caveat: epoch index 4 is not the recipe.** Training exact-match accuracy is climbing fast
  (0.67 at epoch index 4, 0.92 at index 9), so by epoch 100 the model will be close to
  memorising its train set, which is exactly when a panorama-level leak would pay off most. The
  released checkpoint's epoch is not recorded anywhere we can find.

Per-tag AP at epoch index 4 (full held-out set of each arm): missing tactile warning 0.974 / 0.973 /
0.972, surface problem 0.574 / 0.593 / 0.564, points into traffic 0.320 / 0.340 / 0.385, narrow
0.276 / 0.353 / 0.399, not enough landing space 0.115 / 0.240 / 0.210, pooled water 0.202 /
0.125 / 0.175, not level with street 0.182 / 0.178 / 0.185, steep 0.056 / 0.077 / 0.092
(control / pano / block).

**Where the interim numbers come from.** `tag_benchmark_86_snap.sh` copied each arm's
`best.pth` to `best_after_ep4.pth` right after epoch index 4; `infer` scored each copy on all
10,857 crops (`snap_ep4_<arm>_predictions.csv`); `test-only` kept each arm's test rows, byte for
byte, as the committed `train_<arm>_ep4_test_predictions.csv`; `score` gave the table (the
`snapshots` stage of the runbook does all of this). Each committed `.meta.json` describes the
file beside it (rows, sha256) and carries the 10,857-row file's own meta and sha256. The CPU half
re-derives from the repo, and `tests/test_tag_benchmark_86.py` re-derives it on every run. The
GPU half needs the three checkpoints, which are **not published**: they are on makelab2 at
`/homes/gws/jonf/nobackup/tagbench86/train_<arm>/best_after_ep4.pth`, 347,189,218 bytes each,
sha256 control `604432509a9420efad12f20fb16aed2d63426bb5da6928db2f877f265f045de9`, pano
`4241acf12860b36693768733e02b83dc0a291d0250ada5652cbfd0de7dc34e1c`, cell
`8e3d77ef28a41ffa0a9af38253a391c8409262e479b82b7fc2022e8cd5998437`. Without them, `tag_benchmark_86.sh
train` regenerates them with the same seed, but GPU nondeterminism means not bit for bit.

Two deliberate differences from the notebook, both inside `train`: a fixed seed (86; the
notebook sets none), and the deterministic preprocessing (read, resize to 256, pad to 266) is
done once and held in memory as uint8, which produces the same tensors and removes the per-epoch
PNG decode.

**Assumption: the arms train on the same `crop.py` 640 px box the test set is evaluated on.**
The HF zip ships 1440×960 crops and the training notebook does not crop, so the notebook alone
does not say which framing the released model was trained on. The evidence for the 640 px box:
`crop.py` was added to the tagger repo on 2024-12-30 (`1617e0b`), after the ASSETS'24 paper,
together with `download_and_process_test_dataset.sh` (`c17bb73`), which fetches only the four
test splits and then runs `crop.py`. That reads as the HF upload storing uncropped originals
and the paper pipeline having used 640 px crops, which fits the exact reproduction of the
published numbers on `crop.py` output (§1). All 10,857 prepared crops were checked in review to be single
`crop.py` passes of a 1440×960 original. **Stated in advance:** if the control arm's final
checkpoint scores near 0.341 mAP on `test.csv`, the framing assumption holds; if it lands well
away from it, the train framing is the first suspect.

### Finishing the 100-epoch runs

Once all three `train_<arm>.log` files end in `EXIT 0` (each run's wrapper has then also scored
its final `best.pth` on all 10,857 crops), from a checkout of this branch on makelab2:

```bash
WORK=/homes/gws/jonf/nobackup/tagbench86 PY=/homes/gws/jonf/envs/tagger/bin/python \
  bash scripts/analysis/tag_benchmark_86.sh finish
```

`finish` refuses to start unless every arm ended with `EXIT 0`. It runs GPU inference for each
snapshot not yet scored (epoch indices 9, 19 and 49; ~6 min each on a free A40), then
`tag_benchmark_86.py collect --final`, which for each arm:

- writes `train_<arm>_ep{4,9,19,49}_{test_predictions.csv,scores.json}`, an epoch curve with one
  point per snapshot, and `train_<arm>_final_{test_predictions.csv,scores.json}`;
- reads the final checkpoint's epoch from `best.pth` itself and checks it against
  `train_meta.json`'s `best_epoch` (selection is by training accuracy, so it need not be the
  last epoch; it is recorded in the `_final_` scores), and checks that the post-train
  predictions name that checkpoint's sha256;
- copies `train_<arm>/train_log.csv` and `train_meta.json` as `train_<arm>_log.csv` and
  `train_<arm>_meta.json`, so the arms cannot overwrite each other;
- appends usage rows: one per inference, the post-train one included, and the final training
  row, which replaces the `in_progress` row because it carries the same `run_id`.

`collect` is safe to re-run: outputs are rewritten and a re-written usage row replaces its
predecessor. Then commit `analysis_out/` and update §5 and §7.

## 6. Reproduce

Inputs:

| input | identifier | hash |
|---|---|---|
| tagger code | `ProjectSidewalk/sidewalk-tagger-ai` @ `3b7405cd3206ece631cb7a65e22b1ab219df4b75` | git sha; `--tagger-sha` refuses any other HEAD |
| crops + `csv/{train,test}.csv` | HF dataset `projectsidewalk/sidewalk-tagger-ai-validated` @ `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`, `Validated/CurbRamp.zip`, 30,792,993,257 bytes | sha256 `5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d` |
| released checkpoint | HF model `projectsidewalk/sidewalk-tagger-ai-models` @ `65959dbc80b87e4f39385204c4b639cbcf58e1a8`, `validated-dino-cls-b-curbramp-tags-best.pth`, 347,207,242 bytes | sha256 `4d00193aed73fc199049f31cebade51f236bfca92ad9f76adf08a9d08a272833` |
| DINOv2 backbone (training only) | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`, 346,393,545 bytes | sha256 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| label → pano | public `rawLabels` API, 10 deployments | committed result: `hf_curbramp_labels.csv` |
| retrain checkpoints (§5 interim only) | `best_after_ep4.pth` per arm, **unpublished**, makelab2 | sha256 in §5 |

The `fetch` stage downloads the two HF files from `resolve/<revision>/…` at the revisions above,
not from `main`, and checks all three downloads with `sha256sum -c`, stopping on a mismatch.
The zip, crops and checkpoints are a local cache (on makelab2 under
`/homes/gws/jonf/nobackup/tagbench86/`, never committed); the first three are re-downloadable
from the identifiers above. The HF crops are 1440×960; the tagger's `crop.py` cuts a 640 px box
around the label point (clamped at the edges) before evaluation, and `prepare` runs that
function from the pinned checkout rather than a re-typed copy. `prepare` refuses a directory
that an interrupted earlier run left half-cropped.

GPU steps (Linux, one CUDA GPU; makelab2 A40 for the committed run):

```bash
WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/tag_benchmark_86.sh fetch prepare eval infer
# the retrain arms: launch (background, ~17 h), interim reads while they run, and the finish
WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/tag_benchmark_86.sh train
WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/tag_benchmark_86.sh snapshots
WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/tag_benchmark_86.sh finish
```

`train` launches the three arms at once plus the snapshot watcher
(`scripts/analysis/tag_benchmark_86_snap.sh`), which is how the committed run was made. The
committed run was launched by hand with the same commands; verbatim copies of the two helper
scripts that ran beside it are in `analysis_out/tag_benchmark_86/as_run/` (`snap.sh`, and
`infer_snap.sh`, whose job `snapshots` now does). The per-arm launch, read from `ps` on
makelab2 and run from a checkout at `533924c` (identical `train` code to HEAD), was
`bash -c "date -u +%FT%TZ; $PY scripts/analysis/tag_benchmark_86.py train … --out-dir $W/train_<arm> && $PY scripts/analysis/tag_benchmark_86.py infer … --checkpoint $W/train_<arm>/best.pth --out $W/train_<arm>_predictions.csv; echo EXIT \$?; date -u +%FT%TZ" > $W/train_<arm>.log 2>&1`,
which is what the `train` stage writes.

CPU steps (any machine, from the committed files):

```bash
python scripts/analysis/tag_benchmark_86.py score --pred analysis_out/tag_benchmark_86/released_test_predictions.csv --out analysis_out/tag_benchmark_86/released_scores.json --per-label-out analysis_out/tag_benchmark_86/released_test_per_label.csv
python scripts/analysis/tag_benchmark_86.py resplit --out analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv
python scripts/analysis/tag_benchmark_86.py resplit --group cell --out analysis_out/tag_benchmark_86/resplit_cell100m_seed86.csv
# §5, per arm (control: no --split-csv; cell: resplit_cell100m_seed86.csv)
python scripts/analysis/tag_benchmark_86.py score --pred analysis_out/tag_benchmark_86/train_pano_ep4_test_predictions.csv --split-csv analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv --fixed-tags missing-tactile-warning,narrow,not-enough-landing-space,not-level-with-street,points-into-traffic,pooled-water,steep,surface-problem --out analysis_out/tag_benchmark_86/train_pano_ep4_scores.json
```

With the 10,857-row snapshot predictions in `$WORK` (makelab2), `tag_benchmark_86.py collect
--work $WORK --epochs 4` regenerates the committed test-only files byte for byte, their metas
and the §5 scores. `pytest -q tests/test_tag_benchmark_86.py` re-derives the §1 and §5 point
estimates from the committed predictions without the bootstrap.

The label table is rebuilt with
`python scripts/analysis/tag_benchmark_86.py labels --raw-dir <dir> --fetch` (pulls the ten
deployments' `rawLabels` and reads the two CSVs out of the HF zip by HTTP range request, at the
pinned revision). The live API drifts as labels are edited or deleted, so a rebuild is not
guaranteed byte-identical to the committed table; the committed table is the input of record.

Committed outputs (`analysis_out/tag_benchmark_86/`, pinned LF in `.gitattributes` so the hashes
hold on any clone):

| file | what | sha256 |
|---|---|---|
| `hf_curbramp_labels.csv` | 10,857 HF labels with split, `(city, label_id)`, pano, lat/lng, tags | `20731670b66e09df8b6b1e0facb8b55155ba6a4b5f76d616b4a19fdcd6cfa5ec` |
| `released_tagger_evaluate_py.json` | what the tagger's `evaluate.py` reported | `21abe6e03f6ac7f2416a539a26ebc3c2dd30a22eaabad60d91457db2ed114150` |
| `released_test_predictions.csv` | per-label logits, released checkpoint, 2,183 test crops | `4d0333d434c94960474f31c24a048d565dbdc6b8a8e2568d0c33b6e8d0b15904` |
| `released_scores.json` | §1–§2 numbers, per subset, with bootstrap CIs and the paired full-minus-leak-free contrast | `c55f525098146d9a201d68d006c6d0b8404f115a07f737bb316122ebd1777c3c` |
| `released_test_per_label.csv` | per-label labels, leak flags, nearest-train distance, probabilities | `ecda7f08af051a081fa515ddb04b212b953b3feb663d42788f10113d1df84c7f` |
| `resplit_pano_grouped_seed86.csv` | seeded pano-grouped re-split | `c54aa284da4b8d15aff79f2fd447cc5c4578d41fb45434cce497116a733d9222` |
| `resplit_cell100m_seed86.csv` | seeded 100 m block-grouped re-split | `4651dcc79dcc8f7afb180413039a847c6bcf4655c64bac4e114d8eca2521ab9f` |
| `train_{control,pano,cell}_ep4_test_predictions.csv` | per-label logits of each arm's epoch-index-4 checkpoint on its own test labels (+ `.meta.json`) | `45d76552…`, `c4d4da33…`, `e2abcc70…` |
| `train_{control,pano,cell}_ep4_scores.json` | §5 numbers | `63d3dca3…`, `358a2f8e…`, `3cde5799…` |
| `as_run/snap.sh`, `as_run/infer_snap.sh` | the helper scripts as they ran on makelab2 | provenance only |

## 7. Cost

makelab2 (1x A40), no Slurm, so every run is a `paid: false` row in
`analysis_out/usage_log.jsonl` (provider `tagger-86`). Another lab job held ~9–11 GB of the same
A40 throughout.

| run | wall-clock | `gpu_share` |
|---|---:|---:|
| tagger `evaluate.py`, released checkpoint | 139.8 s | 1 |
| `infer`, released checkpoint (first pass, sigmoid scores; superseded) | 64.0 s | 1 |
| `infer`, batch-1 diagnostic | 72.7 s | 1 |
| `infer`, released checkpoint (logits; committed) | 74.5 s | 1 |
| `infer`, epoch-index-4 snapshots, 10,857 crops each, run one after another | 521.4 s, 488.0 s, 509.6 s | 0.25 |
| `train`, three arms concurrently (in progress; 10 epochs in 6,407–6,410 s each) | ~17 h projected per arm, same wall-clock | 0.333 |

**Rows share one GPU, so their wall-clock does not add up to GPU time.** The three training rows
each carry the full wall-clock of an A40 they shared three ways, and the snapshot inference ran
on the same GPU at the same time. Every `tagger-86` row therefore carries `concurrent_with` (our
other runs on that GPU for at least half of this run, read from the run logs) and `gpu_share` =
1 / (1 + that count). Summing `elapsed_s` over the training rows overstates GPU occupancy about
3×; sum `elapsed_s × gpu_share` instead. The other lab job is not counted in `gpu_share`.

The training rows are `status: in_progress`, with the elapsed time measured from the logs at
10 epochs. Each carries a `run_id`; `finish` writes the final row with the same `run_id`, and
`rampnet.ledger.latest_rows` (which every ledger total reads through) keeps only the last row per
`run_id`, so the interim time is not counted twice. At three arms sharing the A40, the full
recipe is ~17 A40-hours of wall-clock for all three together.

Plus CPU: the zip download (~16 min), hashing, extraction and cropping of 10,857 crops (~58 min on NFS),
and `score` (~5.5 min per prediction file, dominated by the bootstrap).

## 8. Not run

- **The 100-epoch retrain results.** In flight at the time of writing (§5); only the
  epoch-index-4 snapshot is scored. The epoch-index-9 snapshots exist on makelab2 and are scored
  by `finish`.
- **Expert-validate as a second test set (plan item 2, last clause).** The 2,432
  expert-validated labels have no crops in the HF format. Production stores a browser crop only
  for labels placed since 2023-10-12, served through a signed, referer-checked route, and those
  crops are framed by the auditor's view, not the 640 px box around the label point the
  checkpoint was evaluated on, so scoring them would test a different instrument. Item 2b's crop
  cutter (PR #177, open) has a `viewport` mode that reproduces the HF framing (median NCC 0.780
  against 200 HF crops; 99.95 % of HF CurbRamp labels have their pano in the makelab2 store).
  What remains: merge #177, cut the 2,432 labels with `--fov viewport` at the HF crop size, run
  `crop.py`'s 640 px box over them, then `infer` and `score` them here. Store coverage for those
  particular labels is not yet measured.
