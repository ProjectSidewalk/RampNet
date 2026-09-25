# Curb-ramp tag benchmark of record (#86, plan item 2)

Item 2 of [`rampnet2_plan.md`](rampnet2_plan.md) §4. The question: what does the ASSETS'24
curb-ramp tagger actually score on its own test set, and how much of that comes from the
test set sharing panoramas with train?

Everything here re-derives from a clean clone: the numbers in §2 and §3 come from committed
per-label predictions through `scripts/analysis/tag_benchmark_86.py score` on CPU, and the
GPU steps that produced those predictions are in §6 as exact commands. The one exception is
stated where it applies: the GPU half of §5 (the retrained arms) needs the arms' checkpoints,
which are unpublished and live on makelab2; the CPU half re-derives from committed predictions.

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
reproduces that framing (median NCC 0.780 against 200 HF crops). #177 merged on 2026-09-24 and
is on this branch; the remaining step is to cut those labels with `--fov viewport`, run
`crop.py`'s 640 px box over them, and score them here (§8).

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

- **Retraining without the leak does not lower the score either (§5).** The recipe retrained
  for 100 epochs on a pano-grouped and a ~100 m block-grouped re-split scores mAP 0.381 and
  0.412 on its own held-out labels, against 0.354 for the same recipe retrained on the published
  split. Unpaired, pano − control is +0.027 [−0.014, +0.064] and block − control +0.058 [+0.010,
  +0.099]; paired on the labels both splits hold out, +0.049 [−0.006, +0.104] (425 labels) and
  +0.002 [−0.051, +0.088] (418). The arms' test sets are different labels, the checkpoint is kept
  by training accuracy, and each arm is one run; §5.1 has the caveats.

Why a leak might not matter much here: a crop is a 640 px box around one label, so two labels
on one panorama usually show different ramps, and the tags are judgments about each ramp. The
retrained arms in §5 test this directly: if the leak were doing work, a model trained on a
pano-grouped split would score lower on its held-out panos than the control does on its own. It
does not.

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

## 5. Retraining the recipe on leak-free splits (100 epochs)

The tagger's DINOv2 recipe (`notebooks/dino-trainer.ipynb`: full fine-tune of DINOv2-B with 4
registers, Adam lr 1e-6, batch 4, BCE, no augmentation, 100 epochs, checkpoint kept by best
*training* exact-match accuracy) was run on three splits of the same 10,857 labels:

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
after 5 epochs.

**Status: final.** The first launch (2026-09-23 01:58:54 UTC) died with a makelab2 reboot at
15:15 UTC, 47,766 s in. Its last epoch lines were at 12:53 UTC (control and block, epoch index
66) and 12:25 UTC (pano, index 62), while the box was at load 60 from another job; the recipe has
no resume (§5.3, §5.4). All three arms were relaunched from scratch at 2026-09-24 04:09:18 UTC
with `tag_benchmark_86.sh train`, unchanged recipe and seed, from the branch at `7b66ad8`, and all
three ended `EXIT 0` between 20:30 and 20:33 UTC
(`train_<arm>_meta.json`: 58,869–59,034 s each, three arms sharing the A40, ~587 s per epoch).
`tag_benchmark_86.sh finish` then scored the snapshots and the final checkpoints (commit
`05fec46`). Every number in §5.1–§5.3 is from the relaunch.

### 5.1 Final result: training without the leak does not lower the score

Each arm's final `best.pth`, scored on its own held-out labels, same eight tags as §1. Brackets
are 95 % pano-clustered bootstrap intervals (1,000 draws, seed 86), from
`train_<arm>_final_scores.json`:

| arm, final `best.pth` (epoch index) | test n (panos) | mAP | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|
| released checkpoint, for reference | 2,183 (1,869) | 0.341 [0.324, 0.371] | 0.665 [0.648, 0.682] | 0.315 [0.289, 0.340] |
| control (89), all of `test.csv` | 2,183 (1,869) | **0.354** [0.335, 0.391] | 0.660 [0.642, 0.677] | 0.315 [0.285, 0.344] |
| control (89), leak-free subset | 957 (889) | 0.370 [0.341, 0.415] | 0.676 [0.654, 0.699] | 0.304 [0.269, 0.339] |
| control (89), leaked subset | 1,226 (980) | 0.358 [0.329, 0.411] | 0.644 [0.618, 0.668] | 0.326 [0.282, 0.369] |
| pano-grouped (90), its held-out panos | 2,197 (1,283) | **0.381** [0.357, 0.417] | 0.660 [0.641, 0.680] | 0.330 [0.298, 0.364] |
| block-grouped (91), its held-out blocks | 2,219 (1,256) | **0.412** [0.381, 0.451] | 0.653 [0.634, 0.671] | 0.367 [0.326, 0.400] |

The contrasts, from `final_contrasts.json` (`tag_benchmark_86.py contrast`, §6). **Unpaired**
compares each arm's headline number on its own test set, the two sets resampled independently
(the re-split arm with seed 86, the other with seed 87), as §1's leaked-minus-leak-free
contrast does. **Paired** keeps only the labels that are test in both splits and scores both
models on the same pano-clustered resamples (seed 86), as §1's full-minus-leak-free contrast
does; it removes the label-set difference at the cost of sample size.

| contrast | unpaired mAP | paired: n labels (panos) | paired mAP | paired micro-F1 | paired macro-F1 |
|---|---:|---:|---:|---:|---:|
| pano-grouped − control | +0.027 [−0.014, +0.064] | 425 (374) | +0.049 [−0.006, +0.104] | +0.021 [−0.010, +0.052] | +0.003 [−0.064, +0.074] |
| block-grouped − control | +0.058 [+0.010, +0.099] | 418 (364) | +0.002 [−0.051, +0.088] | +0.003 [−0.027, +0.033] | +0.003 [−0.062, +0.074] |
| block-grouped − pano-grouped | +0.031 [−0.014, +0.075] | 488 (267) | +0.001 [−0.050, +0.047] | +0.029 [−0.003, +0.059] | +0.057 [+0.010, +0.096] |

Unpaired micro-F1 and macro-F1: pano − control +0.001 [−0.025, +0.025] and +0.015 [−0.029,
+0.063]; block − control −0.007 [−0.031, +0.018] and +0.052 [+0.000, +0.095]; block − pano
−0.007 [−0.036, +0.020] and +0.036 [−0.011, +0.081].

- **The headline question: does a model trained without the leak score lower? No.** Both
  leak-free arms score *higher* than the control on every point estimate of mAP, unpaired and
  paired. The one mAP interval that excludes zero (block − control, unpaired, +0.058) points
  the other way from a leak effect. The lower ends of the intervals bound how much lower a
  leak-free model could score: 0.014 mAP (pano, unpaired) and 0.006 (pano, paired on 425
  labels). The block-grouped paired interval is wide (−0.051) because 418 labels is a small set.
- **The block arm's unpaired lead does not show that block-grouped training is better.** On the
  418 labels both splits hold out, block − control is +0.002 mAP [−0.051, +0.088], against
  +0.058 unpaired. That paired interval contains both 0 and +0.058, so it cannot distinguish a
  model effect from a label-set effect (the two test sets hold different ramps with different
  base rates); it only fails to confirm the unpaired lead.
- **Inside the control arm, still no measurable inflation.** Full minus leak-free (paired, as in
  §1): mAP −0.016 [−0.046, +0.014], micro-F1 −0.016 [−0.034, +0.000], macro-F1 +0.011 [−0.018,
  +0.045]. Leaked minus leak-free (unpaired): mAP −0.012 [−0.064, +0.045], micro-F1 −0.032
  [−0.066, −0.000], macro-F1 +0.022 [−0.036, +0.075]. As at epoch index 4, the micro-F1
  interval sits on zero (its upper end is −0.00005) and supports no reading. After 100 epochs,
  at 99.98 % training exact-match accuracy, the leaked labels still do not score higher than
  the leak-free ones.
- **The framing check stated in advance passes.** The control's final mAP is 0.354 [0.335,
  0.391]; the released checkpoint's 0.341 is inside that interval, 0.013 below the point
  estimate. That is consistent with the train framing being the `crop.py` 640 px box (see the
  assumption below). It does not prove it.
- **Across these 18 intervals, expect about one to exclude zero by chance alone.** Three do, all
  favouring the block arm: block − control unpaired mAP (above), block − pano paired macro-F1
  +0.057 [+0.010, +0.096], and block − control unpaired macro-F1, whose lower end is +0.0001.
  The first is not confirmed by the paired comparison (three bullets up); treat all three as
  leads for a second seed, not results.

**Caveats that travel with §5.1:**

- **The arms' test sets are different labels.** Unpaired contrasts mix the model difference with
  the difference between the held-out sets (base rates and difficulty differ; "narrow" has 152
  test positives in the published split, 178 and 177 in the re-splits). The paired contrasts
  remove that but use only 418–488 labels, and those labels are not a random sample of either
  test set: they are the labels that fall in the test side of both splits.
- **The final checkpoint is chosen by training accuracy, not held-out accuracy.** The recipe
  keeps `best.pth` by best training exact-match accuracy, ties broken by lower training loss
  (`train_<arm>_log.csv`, column `saved`). Training accuracy saturates early: it first reaches
  its maximum (0.99977 / 0.99954 / 0.99965 for control / pano / block) at epoch index 29 / 51 /
  45, and the kept checkpoints (89 / 90 / 91) are later epochs tied at that accuracy with lower
  loss. So "final" means "late in training", chosen by a rule that never sees held-out data;
  the snapshot curve (§5.2) shows how much the held-out score moves meanwhile.
- **One run per arm.** §5.3 measures how much a re-run with the same seed moves the epoch-4
  numbers (about 0.001 mAP), but that is not seed variance: a different seed changes the head's
  initialisation and the shuffle order, which the re-run did not. No second seed was run (§8).

Per-tag AP of the final checkpoints (full held-out set of each arm, control / pano / block):
missing tactile warning 0.973 / 0.970 / 0.969, surface problem 0.597 / 0.607 / 0.575, points into
traffic 0.323 / 0.378 / 0.348, narrow 0.234 / 0.339 / 0.326, not enough landing space 0.182 /
0.241 / 0.299, pooled water 0.323 / 0.210 / 0.522, not level with street 0.141 / 0.161 / 0.180,
steep 0.062 / 0.145 / 0.079. Same caveat as §2: per-tag AP carries each test set's base rate,
and the rare tags (13–42 positives) move several points per ramp.

### 5.2 Epoch curve (relaunch)

mAP of each arm's `best.pth` as of each snapshot, on its own held-out labels (from
`train_<arm>_ep<E>_scores.json` and `train_<arm>_final_scores.json`); training exact-match
accuracy at that epoch from `train_<arm>_log.csv`:

| epoch index | control mAP | pano mAP | block mAP | train acc (control / pano / block) |
|---:|---:|---:|---:|---|
| 4 | 0.338 [0.320, 0.373] | 0.361 [0.342, 0.390] | 0.373 [0.350, 0.406] | 0.669 / 0.678 / 0.681 |
| 9 | 0.327 [0.310, 0.364] | 0.365 [0.343, 0.401] | 0.397 [0.366, 0.435] | 0.917 / 0.913 / 0.906 |
| 19 | 0.339 [0.321, 0.376] | 0.379 [0.355, 0.415] | 0.398 [0.368, 0.434] | 0.995 / 0.997 / 0.995 |
| 49 | 0.352 [0.333, 0.388] | 0.373 [0.351, 0.405] | 0.416 [0.385, 0.453] | 0.999 / 0.998 / 0.998 |
| final (89 / 90 / 91) | 0.354 [0.335, 0.391] | 0.381 [0.357, 0.417] | 0.412 [0.381, 0.451] | 0.9998 / 0.9995 / 0.9997 |

The ordering control < pano < block holds at every snapshot, and each arm moves by at most
~0.04 mAP from epoch index 4 to the end. The epoch-index-4 prediction that
memorisation would make a panorama-level leak "pay off most" late in training did not come true:
the control's leaked-minus-leak-free mAP was −0.017 at epoch index 4 and is −0.012 at the end.

### 5.3 The first launch's epoch-4 snapshot, and what the re-run says about run-to-run noise

The first launch's epoch-index-4 predictions were committed before the reboot (they are what the
superseded interim reading in §5.4 quotes). `finish` rewrote `train_<arm>_ep4_*` with the
relaunch's epoch-4 snapshot, so the first launch's files are kept, byte for byte as committed at
`81e768f`, in `analysis_out/tag_benchmark_86/dead_run_2026-09-23/`. Their checkpoints
(`best_after_ep4.pth`, sha256 control `60443250…`, pano `4241acf1…`, block `8e3d77ef…`, full
values in the files' `.meta.json` and in `tests/test_tag_benchmark_86.py`) were moved before the
relaunch, with the first launch's other snapshots and logs, to
`/homes/gws/jonf/nobackup/tagbench86/dead_2026-09-23/train_<arm>/` on makelab2 (unpublished; per
the #178 run-status comment of 2026-09-24 04:15 UTC, not re-checked here). The relaunch's
epoch-4 checkpoints are different files (their metas name `ff7b8ce9…`, `452b6486…`,
`6c235d32…`).

Both launches ran the same `train` code (`cmd_train` and `build_model` are identical at `533924c`,
the first launch, and `7b66ad8`, the relaunch) with seed 86, which fixes the head's initialisation and
the shuffle order. They still differ. The likely cause is that `train` does not force deterministic CUDA
kernels, so some GPU reductions can sum in a different order on each run and the rounding
differences grow over thousands of steps; the runs did not record enough to confirm which
kernels differed. `ep4_relaunch_minus_dead_run.json`
(`contrast`, relaunch minus first launch, same test labels, so the paired contrast is on every
label):

| arm, epoch index 4 | first launch mAP | relaunch mAP | paired difference, mAP | micro-F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|
| control | 0.3372 | 0.3381 | +0.0008 [−0.0006, +0.0028] | +0.0006 [−0.0026, +0.0039] | −0.0009 [−0.0041, +0.0024] |
| pano-grouped | 0.3598 | 0.3609 | +0.0011 [−0.0017, +0.0060] | +0.0005 [−0.0035, +0.0044] | −0.0003 [−0.0039, +0.0032] |
| block-grouped | 0.3727 | 0.3733 | +0.0006 [−0.0029, +0.0045] | −0.0016 [−0.0060, +0.0028] | −0.0036 [−0.0103, +0.0033] |

What this shows, and what it does not: at epoch index 4, re-running the recipe with the same seed
moved mAP by 0.0006–0.0011, and no interval excludes zero. That is small beside the gaps
between arms (0.02–0.06) and beside each arm's own bootstrap interval (about ±0.035). It is
n = 2 runs per arm, at 5 of 100 epochs, and with the seed held fixed. It is therefore not an
estimate of seed variance, which also varies the initialisation and the data order and is
usually larger (for RampNet's detector it was the binding limit, `docs/seed_variance_51_135.md`).
Nor does it say how far two runs have drifted apart by epoch 100.

**A consumer of the first launch:** #180's interim control (`docs/context_fov_86.md` on its branch)
is the first launch's `best_after_ep49.pth`, inferred on 2026-09-24 before the relaunch reached
epoch 49. The relaunch's epoch-49 snapshot is a different checkpoint (control mAP 0.352 on its full
test set here). #180 swapped to this section's final control per its own §6 (2026-09-24); its §4.1 keeps the interim read and lists what changed.

### 5.4 Superseded: the interim reading at epoch index 4 (first launch, 2026-09-23)

*Kept as history. Superseded by §5.1–§5.3. The numbers below are the first launch's epoch-4
snapshot, now in `dead_run_2026-09-23/`; `tests/test_tag_benchmark_86.py` still re-derives them.*

**Status at the time (superseded):** all three arms finished epoch index 9 (10 epochs) at about
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

### 5.5 Where the §5 numbers come from

`tag_benchmark_86_snap.sh` copied each arm's `best.pth` to `best_after_ep<E>.pth` right after
epoch indices 4, 9, 19 and 49; `infer` scored each copy on all 10,857 crops
(`snap_ep<E>_<arm>_predictions.csv`); `test-only` kept each arm's test rows, byte for byte, as the
committed `train_<arm>_ep<E>_test_predictions.csv`; `score` gave the scores JSON. The final
checkpoint went the same way from `train_<arm>/best.pth` (inferred by each run's wrapper right
after training) to `train_<arm>_final_*`. `finish` / `collect --final` did all of this for the
relaunch (§"How the runs were finished"). Each committed `.meta.json` describes the file beside it
(rows, sha256) and carries the 10,857-row file's own meta, including the checkpoint's sha256. The
contrasts are `contrast` over the committed test predictions (commands in §6).

The CPU half re-derives from the repo, and `tests/test_tag_benchmark_86.py` re-derives every §5
point estimate (snapshots, finals, the first launch's epoch 4, and the contrasts) on every run.
The GPU half needs the checkpoints, which are **not published** (347 MB each). The relaunch's
are on makelab2 under `/homes/gws/jonf/nobackup/tagbench86/train_<arm>/`; the final `best.pth`
sha256 are control `de59f2d38511332376d79c694ede00b5ef547ea6a2a32824e800fac7db2ce24c`, pano
`d821d2eafe3fe545287657dbecd617049e98648a3eb2716101a533f2a192f239`, block
`d77cfb2e423a47605969e34217fe0dba6bcac452943670e31ad2bef97ea667d8` (also in
`train_<arm>_final_scores.json`), and each snapshot's is in its `.meta.json`. The first launch's
are under `dead_2026-09-23/` beside them (§5.3). Without the checkpoints, `tag_benchmark_86.sh train` regenerates
them with the same seed, but GPU nondeterminism means not bit for bit; §5.3 measures how far
apart two such runs were at epoch index 4.

### 5.6 Recipe notes

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
away from it, the train framing is the first suspect. **Result:** the control's final
checkpoint scores 0.354 [0.335, 0.391], which contains 0.341, so the check passes (§5.1).

### How the runs were finished

This is the procedure that produced commit `05fec46`, kept as the runbook for a re-run. Once all three `train_<arm>.log` files end in `EXIT 0` (each run's wrapper has then also scored
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
  row, which replaces an `in_progress` row with the same `run_id` if the ledger holds one.

`collect` is safe to re-run: outputs are rewritten and a re-written usage row replaces its
predecessor. Then commit `analysis_out/`, move any overwritten earlier record aside (as §5.3 did
for the first launch's epoch 4), and update §5 and §7.

## 6. Reproduce

Inputs:

| input | identifier | hash |
|---|---|---|
| tagger code | `ProjectSidewalk/sidewalk-tagger-ai` @ `3b7405cd3206ece631cb7a65e22b1ab219df4b75` | git sha; `--tagger-sha` refuses any other HEAD |
| crops + `csv/{train,test}.csv` | HF dataset `projectsidewalk/sidewalk-tagger-ai-validated` @ `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`, `Validated/CurbRamp.zip`, 30,792,993,257 bytes | sha256 `5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d` |
| released checkpoint | HF model `projectsidewalk/sidewalk-tagger-ai-models` @ `65959dbc80b87e4f39385204c4b639cbcf58e1a8`, `validated-dino-cls-b-curbramp-tags-best.pth`, 347,207,242 bytes | sha256 `4d00193aed73fc199049f31cebade51f236bfca92ad9f76adf08a9d08a272833` |
| DINOv2 backbone (training only) | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`, 346,393,545 bytes | sha256 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| label → pano | public `rawLabels` API, 10 deployments | committed result: `hf_curbramp_labels.csv` |
| retrain checkpoints (GPU half of §5 only) | `best.pth` and `best_after_ep{4,9,19,49}.pth` per arm, **unpublished**, makelab2 | sha256 in §5.5 and each `.meta.json` |

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
which is what the `train` stage writes. That describes the first launch; the relaunch of
2026-09-24 04:09 UTC ran `tag_benchmark_86.sh train` itself, from the branch at `7b66ad8`
(its `train_<arm>_meta.json` records the same tagger sha, torch version and seed).

CPU steps (any machine, from the committed files):

```bash
python scripts/analysis/tag_benchmark_86.py score --pred analysis_out/tag_benchmark_86/released_test_predictions.csv --out analysis_out/tag_benchmark_86/released_scores.json --per-label-out analysis_out/tag_benchmark_86/released_test_per_label.csv
python scripts/analysis/tag_benchmark_86.py resplit --out analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv
python scripts/analysis/tag_benchmark_86.py resplit --group cell --out analysis_out/tag_benchmark_86/resplit_cell100m_seed86.csv
# §5, per arm and per stem (ep4, ep9, ep19, ep49, final; control: no --split-csv; cell: resplit_cell100m_seed86.csv)
python scripts/analysis/tag_benchmark_86.py score --pred analysis_out/tag_benchmark_86/train_pano_final_test_predictions.csv --split-csv analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv --fixed-tags missing-tactile-warning,narrow,not-enough-landing-space,not-level-with-street,points-into-traffic,pooled-water,steep,surface-problem --out analysis_out/tag_benchmark_86/train_pano_final_scores.json
# §5.1 contrasts (~15 min) and §5.3's relaunch-minus-first-launch (~20 min)
O=analysis_out/tag_benchmark_86
python scripts/analysis/tag_benchmark_86.py contrast --pair pano=$O/train_pano_final_test_predictions.csv control=$O/train_control_final_test_predictions.csv --pair cell=$O/train_cell_final_test_predictions.csv control=$O/train_control_final_test_predictions.csv --pair cell=$O/train_cell_final_test_predictions.csv pano=$O/train_pano_final_test_predictions.csv --out $O/final_contrasts.json
python scripts/analysis/tag_benchmark_86.py contrast --pair control=$O/train_control_ep4_test_predictions.csv control=$O/dead_run_2026-09-23/train_control_ep4_test_predictions.csv --pair pano=$O/train_pano_ep4_test_predictions.csv pano=$O/dead_run_2026-09-23/train_pano_ep4_test_predictions.csv --pair cell=$O/train_cell_ep4_test_predictions.csv cell=$O/dead_run_2026-09-23/train_cell_ep4_test_predictions.csv --out $O/ep4_relaunch_minus_dead_run.json
```

The contrast JSONs carry the sha256 of every prediction file and of the label table they read.

With the 10,857-row snapshot predictions in `$WORK` (makelab2), `tag_benchmark_86.py collect
--work $WORK --final` regenerates the committed test-only files byte for byte, their metas
and the §5 scores (and appends usage rows, which replace their predecessors by `run_id`). `pytest -q tests/test_tag_benchmark_86.py` re-derives the §1 and §5 point
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
| `train_{control,pano,cell}_ep{4,9,19,49}_test_predictions.csv` | relaunch: per-label logits of each arm's snapshot checkpoint on its own test labels (+ `.meta.json`) | ep4 `0d87f312…`, `ecb840bd…`, `a1d7074f…`; ep9 `c8046a6d…`, `d14bacd0…`, `4336c604…`; ep19 `64b5c7c6…`, `d072aa16…`, `2c333103…`; ep49 `01b82c5f…`, `989b0186…`, `b98f6355…` |
| `train_{control,pano,cell}_final_test_predictions.csv` | relaunch: the same for the final `best.pth` (+ `.meta.json`) | `4e97dbea…`, `e9f5932a…`, `d577216b…` |
| `train_{control,pano,cell}_{ep4,ep9,ep19,ep49,final}_scores.json` | §5.1–§5.2 numbers, with bootstrap CIs; `_final_` also carries `best_epoch` and the checkpoint sha256 | final `06aa2b69…`, `369ca379…`, `30954eeb…` |
| `train_{control,pano,cell}_log.csv`, `train_{control,pano,cell}_meta.json` | relaunch: per-epoch loss, training exact-match accuracy, `saved`; run metadata | provenance |
| `dead_run_2026-09-23/train_{control,pano,cell}_ep4_{test_predictions.csv,scores.json}` (+ `.meta.json`) | the first launch's epoch-index-4 record, byte for byte as committed at `81e768f` (§5.3, §5.4) | predictions `45d76552…`, `c4d4da33…`, `e2abcc70…`; scores `63d3dca3…`, `358a2f8e…`, `3cde5799…` |
| `final_contrasts.json` | §5.1 arm contrasts, unpaired and paired, with the sha256 of every input | re-derived by the tests (point estimates) |
| `ep4_relaunch_minus_dead_run.json` | §5.3 relaunch minus first launch at epoch index 4 | re-derived by the tests (point estimates) |
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
| `infer`, first launch's epoch-index-4 snapshots, 10,857 crops each, one after another | 521.4 s, 488.0 s, 509.6 s | 0.25 |
| `train`, first launch, three arms concurrently, **failed** (killed by the reboot; row `status: failed`) | 47,766 s each, same wall-clock | 0.333 |
| `train`, relaunch, three arms concurrently (control / pano / block) | 59,034 s, 58,982 s, 58,869 s | 0.333 |
| `infer`, relaunch final `best.pth`, 10,857 crops (control / pano / block; the three overlapped each other for ~5 min) | 315.3 s, 327.6 s, 358.9 s | 1 |
| `infer`, relaunch snapshots at epoch indices 4 / 9 / 19 / 49, 12 runs one after another, after training | 284.1–287.6 s each, 3,422.1 s total | 1 |

**Rows share one GPU, so their wall-clock does not add up to GPU time.** The three training rows
each carry the full wall-clock of an A40 they shared three ways, and the first launch's snapshot
inference ran on the same GPU at the same time. Every `tagger-86` row therefore carries
`concurrent_with` (our other training runs on that GPU for at least half of this run, read from
the run logs) and `gpu_share` = 1 / (1 + that count). Summing `elapsed_s` over the training rows
overstates GPU occupancy about 3×; sum `elapsed_s × gpu_share` instead. The other lab job is not
counted in `gpu_share`, and neither is overlap between inference runs (only training runs are
checked), so the three final inferences, which overlapped each other, each carry `gpu_share` 1.

Totals, `elapsed_s × gpu_share` over `rampnet.ledger.latest_rows` of the `tagger-86:` rows:

| part | A40-hours |
|---|---:|
| released checkpoint (evaluate + 3 inference passes) | 0.10 |
| first launch: training (failed at 47,766 s) + epoch-4 inference | 13.27 + 0.11 |
| relaunch: training | 16.38 |
| relaunch: 12 snapshot + 3 final inferences | 0.95 + 0.28 |
| **total** | **31.08** (of which 13.37 is the failed first launch) |

All `paid: false`; $0. The first launch's time is real GPU time spent and stays in the ledger
as `status: failed` rows. Each training row carries a `run_id`. The first launch's
`in_progress` rows were replaced by its `failed` rows, because `rampnet.ledger.latest_rows`
(which every ledger total reads through) keeps only the last row per `run_id`. The relaunch
never had `in_progress` rows in the committed ledger; `finish` wrote its final rows directly.

Plus CPU: the zip download (~16 min), hashing, extraction and cropping of 10,857 crops (~58 min on NFS),
and `score` (~5.5 min per prediction file, dominated by the bootstrap).

## 8. Not run

- **A second seed per arm.** Each arm is one run of the recipe at seed 86 (§5.1). §5.3's re-run
  held the seed fixed, so it is not a seed-variance estimate. A second seed would take ~16 A40-hours
  for all three arms together on makelab2 (§7).
- **The first launch's later snapshots (epoch indices 9, 19, 49).** Their checkpoints are kept on
  makelab2 (§5.3) but were not scored here; only #180 used one (control, epoch index 49), on its
  own common rows.
- **Expert-validate as a second test set (plan item 2, last clause).** The 2,432
  expert-validated labels have no crops in the HF format. Production stores a browser crop only
  for labels placed since 2023-10-12, served through a signed, referer-checked route, and those
  crops are framed by the auditor's view, not the 640 px box around the label point the
  checkpoint was evaluated on, so scoring them would test a different instrument. Item 2b's crop
  cutter (PR #177, merged 2026-09-24, on this branch) has a `viewport` mode that reproduces the HF framing (median NCC 0.780
  against 200 HF crops; 99.95 % of HF CurbRamp labels have their pano in the makelab2 store).
  What remains: cut the 2,432 labels with `--fov viewport` at the HF crop size, run
  `crop.py`'s 640 px box over them, then `infer` and `score` them here. Store coverage for those
  particular labels is not yet measured.
