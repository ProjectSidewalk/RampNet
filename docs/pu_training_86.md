# Positive-unlabeled training of the tag head (#86, RampNet 2.0 plan item 5)

**Status: PROPOSED, 2026-09-24. Nothing here has run.** This is the plan for item 5 of
[`rampnet2_plan.md`](rampnet2_plan.md) §4, written for review before any crop is cut or any job
is submitted. Every number is taken from a committed document or a committed table, named at the
point of use; the few that are arithmetic on those numbers say so. Decisions Jon has not made are
marked PROPOSED in the text and collected in §8.

Umbrella issue: [#86](https://github.com/ProjectSidewalk/RampNet/issues/86). Inputs it builds on:
the supervision audit (item 1, `ps_supervision_audit.md`, PR #175), the benchmark of record
(item 2, `tag_benchmark_86.md`, PR #178), the crop cutter (item 2b, `crop_cutter.md`, PR #177)
and the context experiment (item 4, `context_fov_86.md`, PR #180).

## 1. Question, and the decision it makes

The plan's wording (item 5): *same head on the 118k tagged plus the ~236k untagged tag-era labels
with a PU loss, vs the clean 8.7k. ASSETS found noisy data hurt under naive training; PU is
untested. Needs item 2b for every label without a production crop. ~2 GPU-days.*

The question in one line: **does the tag head get better when it is trained on every human
curb-ramp label of the tag era, with the untagged ones treated as unlabeled rather than
negative, than when it is trained on the 8,674 reviewed crops the ASSETS'24 tagger used?**

Why it is not obvious. The ASSETS'24 paper ran exactly the naive version of this experiment on
its whole 33-class dataset (a ~24k cleaned set against a ~87k "larger but noisier crowdsourced"
set) and reports that the larger set was *detrimental* (`sidewalk-tagger-ai` README, abstract).
The audit explains why that should happen for curb ramps: an untagged label is not a negative.
Tags are positive-unlabeled (plan §2.2; audit §4, §7): the two validation-study raters tagged the
same ramps at 61 % vs 14 %, and when both tagged they agreed on *which* tag, so the difference is
the threshold of reporting, not perception. Training with "no tag = negative" teaches the model
that threshold. A PU loss is the standard tool for exactly this label structure; whether it
recovers what the naive run loses is the thing nobody has measured.

**What "beats" means.** Per-tag average precision and mAP on the benchmark's eight scored tags
(`tag_benchmark_86.md` §1: the tags with at least 10 test positives), on three test sets, all
pano-clustered bootstrap CIs as the benchmark reports them:

1. the committed HF `test.csv` rows (2,183 labels), the only split comparable to the published
   0.34 mAP;
2. the pano-grouped re-split's held-out panos (`resplit_pano_grouped_seed86.csv`, 2,197 labels),
   the leak-free read;
3. the expert-validate corpus (2,452 labels, 22 deployments, audit §6) as a second, independent
   test set, once its crops exist (§2.4; this is also the unfinished last clause of item 2,
   `tag_benchmark_86.md` §8).

The comparison is paired on test labels: every arm is scored on the same rows. A gain counts if
the paired difference's 95 % interval excludes zero on (1) *and* (2); (3) is reported beside them.
The benchmark's seed variance is unmeasured (`context_fov_86.md` §6), so a difference smaller than
a seed's worth is not a result; §7 says what that costs to fix.

**Decision it makes.** If PU beats the clean set: the 2.0 tag head trains on the full label
store, and the review pass (item 3) is a test set, not a training set. If naive beats clean:
the reporting-threshold reading is wrong somewhere and §7 says what to look at. If neither beats
clean: the tag head is data-limited by *reviewed* labels, and the effort goes into the review
pass and the close-the-loop item (plan item 11), not into more crops.

## 2. The label universe

All counts are human CurbRamp labels (SidewalkAI's account excluded by user id), from
`analysis_out/ps_audit/tiers.csv` on the audit branch at `fdd73c6` (PR #175's post-merge
timestamp fix, which keeps 354 labels whose timestamps lack fractional seconds).
**`main` still carries the pre-fix table** (tag era 361,863, tagged 118,293) until that commit
lands in a follow-up PR; the differences are 0.1 % and change nothing below, but the plan cites
the fixed numbers so the scripts written from it match.

### 2.1 Tiers

| tier | tag-era labels (placed ≥ 2018-04-29) | tagged | untagged | with a production crop |
|---|---:|---:|---:|---:|
| tier 1: every human label | 362,217 | 118,402 | 243,815 | 196,742 |
| tier 2: crowd-validated correct (`correct == true`) | 295,957 | 89,514 | 206,443 | 168,952 |
| tier 3: placed by an Owner | 8,063 | 3,277 | 4,786 | 1,564 |

`untagged` is arithmetic on the two columns beside it. The production-crop column counts labels
placed since 2023-10-12, which is inside the tag era, so it is the same number in every era.

**PROPOSED: the universe is tier 2, not tier 1.** Tier 2 is a *position* tier: an Agree from
`/validate` or `/mobile` says the ramp is there, not that its tags are right (audit §2). That is
exactly what a PU loss wants: every crop really shows a curb ramp, and only the tag set is
uncertain. Tier 1 adds 66,260 labels that nobody confirmed, and a crop of a non-ramp with
"unlabeled" tags is noise a PU loss cannot model (its unlabeled term assumes the class prior of
the positive-labeled population). The cost is 28,888 tagged labels (118,402 − 89,514). Tier 1 is
recorded as a not-run arm in §7.

### 2.2 Which tags

The eight tags the benchmark scores, and no others: *missing tactile warning, points into
traffic, narrow, surface problem, not level with street, not enough landing space, debris /
pooled water, steep* (`tag_benchmark_86.md` §2). The other four in the union vocabulary are out:
`tactile warning` is retired (0 of its 471 HF train positives still carry it, `tag_benchmark_86.md`
§2), `parallel lines` and `not aligned with crosswalk` are city-specific (1 and 3 cities with
labels, audit §4), `not visible` has 5 labels.

Observed positive rate in the tag era, tier 1 (audit §4, `tags.csv`): missing tactile warning
16.9 %, points into traffic 9.2 %, narrow 5.3 %, surface problem 4.2 %, not level with street
4.0 %, not enough landing space 3.6 %, debris / pooled water 1.3 %, steep 1.3 %. These are the
*labeled* frequencies the PU loss sees, not the class priors (§3.2).

### 2.3 Excluded tags are masked cells, not negatives

Each deployment lists its own tag subset (`config.excluded_tags`; audit §4 and
`analysis_out/ps_audit/excluded_tags.csv`). Where a tag is not offered, a label cannot carry it,
so its absence there says nothing. Of the eight target tags, two deployments hide some: amsterdam
hides *missing tactile warning* and *points into traffic* (yet 771 and 809 of its tag-era labels
carry them anyway, placed before the exclusion, audit §4); chandigarh-india lists only 3 tags and hides six of the
eight (119 labels). Everywhere else all eight are listed.

Rule: the loss is computed per (label, tag) cell, and a cell whose tag is not in the label's
deployment tag list at fetch time is **masked** out of both the positive and the unlabeled term
(a tag carried anyway, as in amsterdam, stays a positive). The mask comes from the audit's
`labelTags` pull, keyed on deployment, so it is reproducible from the committed table. This is a
few lines in the loss and must not be skipped: with amsterdam's 12,246 tag-era labels treated as
missing-tactile negatives, the second-strongest tag in the benchmark acquires 12k wrong
negatives.

### 2.4 Nothing in a test set, or near it, is in training

Keys are the composite `(city, label_id)` (`label_uid`), never `label_id` alone. Removed from
every training arm, by panorama, not by label:

- every panorama carrying an HF `test.csv` label (1,869 panos, `tag_benchmark_86.md` §1) and every
  panorama in the pano-grouped re-split's test side (1,283 panos, §5 there), plus every other
  label of the same city within 10 m of any of those test labels, the benchmark's stricter rule
  (`tag_benchmark_86.md` §3: the same corner seen from a neighbouring panorama);
- every panorama carrying one of the 2,452 expert-validate labels, and the same 10 m rule;
- the validation slice of §5.3.

The exclusion list is written once, committed as `analysis_out/pu_training_86/excluded_panos.csv`
with the reason per row, and asserted against every arm's training table by the test suite.
The four HF train labels with no live row (`tag_benchmark_86.md` §3) are simply absent.

## 3. The PU formulation

### 3.1 Arms

Every arm is the benchmark's recipe (`tag_benchmark_86.py train`: DINOv2-B/14 with registers,
full fine-tune, Adam 1e-6, batch 4, BCE-with-logits, no augmentation, seed 86) with only the
training table and the loss changed. The control already exists.

| arm | training labels | loss on an untagged label | what it isolates |
|---|---|---|---|
| **clean** (control, exists) | HF train, 8,674 reviewed crops, absence affirmed | negative (correct: the ASSETS'24 pass affirmed empty tag sets, audit §8) | the published benchmark, #178 `train_control`, re-scored on the shared test rows |
| **naive** | tier 2 tag era minus §2.4 | negative for every tag | the ASSETS'24 "Experiment 2" arm on curb ramps; the thing PU is supposed to fix |
| **nnPU** | same table as naive | unlabeled: non-negative PU risk per tag with a class prior per tag (§3.2) | the experiment |
| soft-negative (PROPOSED, only if the budget allows a fourth arm) | same table | negative with target 1 − c_t instead of 0, where c_t is the tag's labeling frequency | the cheapest fix; tells whether nnPU's unlabeled term is doing anything a soft target does not |

Both `naive` and `nnPU` include the HF train crops in their table, framed the same way as every
other label (§4), so the PU arms are supersets of the control's data, not disjoint from it. On the
HF train rows the absence *is* affirmed, so their cells stay hard negatives in every arm
(they are the only labels in the universe for which that is true, plus the expert-validate rows,
which are test).

**nnPU per tag.** For tag t with prior π_t, positives P_t (labels carrying t) and unlabeled U_t
(every other, unmasked label), the non-negative PU risk (Kiryo et al., 2017) is

    R_t = π_t · E_P[ℓ(f_t, +1)] + max(0, E_U[ℓ(f_t, −1)] − π_t · E_P[ℓ(f_t, −1)])

with ℓ the sigmoid loss, summed over the eight tags, per mini-batch, with the paper's
gradient-ascent step when the bracket is negative. Batch 4 is small for the unlabeled-term
estimate; §5.4 makes batch size a decision.

**PROPOSED: a tagged label that lacks tag t is unlabeled for t, not negative.** A rater who
tagged *narrow* looked at the ramp and did not tag *steep*; that is stronger evidence than an
untagged label, but the validation study says the reporting threshold varies by rater and the
audit says it varies by year, so it is not an affirmed absence either. Treating it as unlabeled
is the conservative reading; the alternative is a cheap ablation on `nnPU` and is listed in §7.

### 3.2 The class prior per tag

nnPU needs π_t, the fraction of curb ramps that *truly* carry tag t, which the labeled frequency
under-reports by the labeling rate c_t (π_t = observed / c_t). Two candidate sources exist; both
are committed.

| source | n labels | missing tactile | points into traffic | surface problem | narrow | landing | not level | steep | pooled water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSETS'24 validated set, all 10,857 rows (positives from #86, 2026-09-22T12:50 comment; rate is arithmetic on 10,857) | 10,857 | 38.9 % | 12.7 % | 9.9 % | 8.5 % | 5.8 % | 4.1 % | 1.4 % | 1.4 % |
| jonfroehlich, own labels 2024–2026 (`rater_drift.csv`, rate weighted by labels per year) | 1,133 | 38.4 % | 20.4 % | 9.4 % | 16.1 % | 11.7 % | 9.0 % | 4.4 % | — |
| mikey, own labels 2024–2026 (same) | 363 | 30.3 % | 15.2 % | 8.6 % | 1.7 % | 2.7 % | 1.7 % | 0.0 % | — |

`rater_drift.csv` does not carry pooled water. The two trusted raters disagree by 2× or more on
five of the seven tags they share (narrow 16.1 vs 1.7, landing 11.7 vs 2.7, not level 9.0 vs 1.7,
steep 4.4 vs 0.0), which is the reporting-threshold effect again, now inside the trusted tier.
Against the tag-era observed rates (§2.2) the ASSETS priors imply labeling frequencies c_t of
about 0.43 for missing tactile warning and 0.72 for points into traffic, and near 1 for the
rare tags, which is plausible: the rarer the defect the more likely it was tagged when seen.

**PROPOSED: π_t from the ASSETS'24 validated set**, because it is the only source where absence
was affirmed over a full tag set by more than one reviewer (five editor accounts, 10 cities,
audit §6 and §8), and it is 7× the size of the trusted-rater 2024+ set; the trusted-rater rates are the
sensitivity check (one extra `nnPU` run with Jon's priors, §7, if the main result is close).
Its known bias travels with it: the validated set is 10 cities and the reviewers' own threshold.
Everything about the prior is a CLI argument (`--prior-csv`), never a constant edited in a
session.

## 4. Cut plan

### 4.1 What gets cut, and at which field of view

**Every training label is re-cut from the makelab2 pano store, the production crops are not
used.** Production stores a browser crop only for labels placed since 2023-10-12 (168,952 of
tier 2), framed by the labeler's viewport at their zoom, and served through a signed,
referer-checked route (`tag_benchmark_86.md` §8); mixing those with cutter crops for the older
127,005 labels would put two framings in one arm. One cutter, one field of view, one JPEG encode
for every crop, including the HF train and test rows (whose re-cut at the control's framing is
the `viewport` arm of item 4).

**The field of view comes from item 4's results: PROPOSED `fov25`.** `context_fov_86.md` §4
(PR #180, on `exp/context-fov-86`) scored four framings on 2,182 common test rows, paired
against an interim control (the #178 control's epoch-index-49 snapshot) and against `viewport`:

| arm minus reference | mAP, paired 95 % interval |
|---|---|
| `viewport` − control | −0.010 [−0.035, +0.017] |
| `fov25` − control | −0.006 [−0.035, +0.020] |
| `fov25` − `viewport` | +0.004 [−0.024, +0.025] |
| `fov50` − control | **−0.036 [−0.064, −0.009]** |
| `fov50` − `viewport` | **−0.026 [−0.053, −0.002]** |
| `fov90` − control | **−0.075 [−0.105, −0.048]** |

So `fov50` and `fov90` are measurably worse (*surface problem*, *steep* and *missing tactile
warning* all drop at 50°), and `fov25` and `viewport` tie the control and each other. The
choice between the two tied framings is a trade-off:

- **`fov25` (proposed):** label-centred, one field of view for every label, one JPEG encode, and
  independent of the labeler's zoom. Zoom varies by rater, and rater is the axis the PU loss is
  trying not to learn, so a framing that does not depend on it is the safer default. Its one hint
  (*points into traffic* +0.045 [−0.003, +0.089] against the control) needs seeds before it is
  anything.
- **`viewport`:** reproduces the control's framing (the labeler's zoom, 12.5°–48°, ramp
  off-centre), so a PU arm would differ from the HF-framed reference rows (§3.1) only in data. It
  needs each label's zoom, re-saves the 640 px box as a second JPEG encode, and had one tag off 0
  against the control (*surface problem* −0.054 [−0.102, −0.009]).

Item 4's control column is interim until the #178 re-run finishes (`context_fov_86.md` §6); if
the final control changes this reading, the default is revisited before the cut. The cut command
is the cutter's, unchanged (`scripts/crop_cutter.py --fov 25 --size 640 --aspect 1.0 --tilt mm`,
PR #177).

Crops to cut: 295,957 (tier 2 tag era) plus the expert-validate 2,452, minus overlap. Store
coverage on a seeded 5,000-label tag-era sample was 99.16 % (`crop_cutter.md` §4); the misses
are infra3d cities (0 / 26, no directory in the store) and 16 GSV panos. Expect about 2,500
labels to drop with status `missing_pano`, recorded per row in the manifest as item 4 did
(`dropped_labels.csv`).

### 4.2 Throughput, and the worker cap

Measured on item 4's cut (`analysis_out/context_fov_86/cut_summary_*.json`, makelab2, 48 cores):

| pass | crops | panos opened | workers | wall | crops / s | labels / s |
|---|---:|---:|---:|---:|---:|---:|
| three fields of view in one pass | 32,559 | 6,295 | 12 | 1,767 s | 18.4 | 6.1 |
| viewport (1440×960) | 10,853 | 6,295 | 12 | 1,122 s | 9.7 | 9.7 |

The pano decode dominates (one decode serves all three crops of a label in the first row), so a
single-field-of-view pass over 296k labels runs at roughly 6–10 labels/s at 12 workers, **8–14 h**,
and about twice that at 6 workers. That is an estimate from two passes over 6,295 panos; the first
thing the cut does is time the same 5,000-label coverage sample and write the measured rate into
this section before the full run.

**12 workers is not allowed again.** The item 4 cut at 12 workers beside the three #178 training
arms drove makelab2's load to 60, its sshd refused connections from about 12:30 UTC, and the box
was rebooted at 15:15 UTC on 2026-09-23, killing all three arms at epoch 62–66 of 100 with no
resume (PR #178 comments of 2026-09-23 13:23 UTC and 2026-09-24). PROPOSED rule: **at most 6
workers, `nice 10`, and never while a GPU job of ours is training on makelab2**; the #178 rerun
launched 2026-09-24 04:09 UTC ends about 20:30 UTC the same day, and the cut waits for it. The
alternative, cutting from makelab1 over NFS, keeps makelab2's CPUs free but reads 15 TB of store
over the network and its throughput is unmeasured; it is the fallback if makelab2 is busy, after a
timed sample run.

### 4.3 Size and transfer

Bytes per crop for item 4's crops, **observed on klone and not committed** (2026-09-24,
`find -printf %s` over the 10,848 crops per arm): fov25 155.0 KB, fov50 161.2 KB, fov90 133.6 KB,
viewport after the 640 px box 123.1 KB. At fov25's 155.0 KB, 295,957 crops are about **46 GB**
(arithmetic). The cutter's manifest records every crop's `bytes` and `sha256`, so the real total
is summed from it and committed with the cut summary rather than projected.

Neither makelab nor klone holds a key for the other, so item 4's tar went makelab → this
desktop → klone in two hops; the 6.3 GB push needed one resume (observed during item 4's
transfer, not recorded in a committed file). Eight times that is a day of
relay. PROPOSED: split the tar per city (`tar` one archive per deployment, sha256 each), push with
`rsync --partial` from the desktop, and verify the per-city sums on klone before unpacking; the
manifest already carries a per-crop sha256, so an incomplete city is detected, not guessed. A
direct makelab → klone path (a key on makelab1 for klone, or Globus) is Jon's call and would
remove the relay entirely; it is not assumed.

### 4.4 The crops are an unpublished input

The makelab2 pano store is the Project Sidewalk scraper's archive, not a published dataset
(`crop_cutter.md`, top). So the training crops for `naive` and `nnPU` cannot be re-derived by
someone without the store, and every number from those arms says so beside it. What a reader
without the store can still check: the committed training tables (label ids, tags, priors, mask,
exclusion list), the per-crop sha256 manifests, and the CPU scoring of the committed per-label
predictions. What would publish the input: a Hugging Face dataset of the 640 px crops (~45 GB, the
same shape as `sidewalk-tagger-ai-validated`, which ships 30.8 GB of crops of the same labels'
subset today). Whether to publish is a decision for Jon (§8); the doc records the gap until then.

## 5. Training on klone

### 5.1 Measured cost of the recipe

| where | train crops | s / epoch | source |
|---|---:|---:|---|
| klone L40S, one arm per GPU (item 4, `gpu-l40s-makelab`) | 8,666 | 111.5–113.3 mean over 100 epochs (arms `fov25`, `fov50`, `viewport`) | `analysis_out/context_fov_86/train_<arm>_train_log.csv` on klone, to be committed with #180 |
| makelab2 A40, three arms sharing | 8,674 | ~605 | `tag_benchmark_86.md` §5 |

So the recipe processes **~77 crops/s on a dedicated L40S** (8,666 / 112 s), and one epoch over
tier 2's 295,957 crops is **1.06 h**; over tier 1's 362,217 it would be 1.30 h. Item 4's jobs
ran 3:58–4:57 h wall for 100 epochs including preprocessing and inference (`sacct`, jobs
40486691–40486693).

### 5.2 Epoch budget

The recipe is 100 epochs over 8,674 crops, 867k sample presentations. On tier 2 that is
**2.9 epochs**. Two budgets, both PROPOSED, run in this order:

1. **Presentation-matched, 3 epochs** (~3.2 h per arm on one L40S, plus prep). The same number
   of optimizer steps as the control saw, so a difference is data, not compute. Cheap enough to
   run `naive`, `nnPU` and the soft-negative arm all three: ~10 GPU-hours.
2. **30 epochs** (~32 h per arm) for `naive` and `nnPU`, ~64 GPU-hours, with a checkpoint scored
   on the validation slice every 5 epochs so the curve says whether 30 was enough. This is the
   arm the headline comes from.

The plan's "~2 GPU-days" is budget 1 plus one 30-epoch arm; both 30-epoch arms are ~3 GPU-days.
Stated so the ledger is not a surprise.

### 5.3 Selection without reading the test set

The recipe keeps the checkpoint with the best *training* exact-match accuracy
(`tag_benchmark_86.md` §5). Under a PU loss that number is meaningless (the targets are not
labels), and on 296k crops the last epoch is not necessarily the best. PROPOSED: hold out a
**10 % pano-grouped slice of the HF train rows** (about 870 reviewed labels, absence affirmed,
same seed-86 grouping as the re-split) from every new arm, score every 5-epoch checkpoint on it,
and select by its mAP. It is the only place in the universe with affirmed negatives that is not a
test set. The control keeps its published selection (best training accuracy, epoch read from
`best.pth`), which is a stated asymmetry: the control was selected by a rule that reads only its
training set, the new arms by a reviewed slice they never trained on. If that worries the read,
the control's epoch-4/9/19/49 snapshots (PR #178) can be scored on the same slice, but they are
training labels for the control, so that read is optimistic and is reported as such.

### 5.4 What the trainer needs before a 30-hour job is safe

`tag_benchmark_86.py train` was written for 8.7k crops and one sitting. Three changes, each a
small PR against the benchmark script with its own test, before any budget-2 job is submitted:

- **Preprocess once, outside the job.** The trainer decodes every crop into a uint8
  3×266×266 tensor in memory at start: 212,268 bytes per crop, 62.8 GB for tier 2, and the
  decode ran at 3.3–24.8 crops/s from `/gscratch` in item 4's jobs (`prep_s` 349–2,634 s for
  8,666 crops, `train_<arm>_train_meta.json`), i.e. **3–25 h of serial decode per job**. A
  `prep` stage writes the uint8 array once as a `.npy` memmap on `/gscratch/makelab` (one file
  per arm framing, sha256 in the manifest), in parallel on a CPU allocation; the trainer maps it.
  The 48 GB the item 4 jobs requested becomes `--mem=96G` for the memmap's page cache, well
  inside an L40S node's 1.5 TB.
- **Resume.** Save model, optimizer, epoch and the shuffle generator's state every epoch; on
  start, load if present. The #178 arms were lost to a reboot for want of this, and it is the
  precondition for `ckpt-all`, where `PreemptMode=REQUEUE` (`scontrol show partition`). The
  stateless-schedule rule from `stage2_cosine_rung_135.md` applies: anything that depends on the
  step index is recomputed from the index, never carried in optimizer state.
- **The PU loss and the mask** as a `--loss {bce,nnpu,soft}` switch with `--prior-csv` and
  `--mask-csv`, unit-tested against a hand-computed batch (positive term, unlabeled term, the
  non-negative clamp, a masked cell contributing zero to both).

Batch size stays at 4 (PROPOSED). Raising it would cut wall-clock and improve the unlabeled-term
estimate, and would also change the recipe the control was trained under; if the 3-epoch read
looks promising, a batch-16 `nnPU` run is the first follow-up, not part of this comparison.

### 5.5 Where, and the rows it writes

`gpu-l40s-makelab` (`MaxTime=UNLIMITED`, no preemption, one L40S per job, 8 CPUs) for every arm,
one Slurm job per arm as `context_fov_86.slurm` does it; `ckpt-makelab` only once resume exists
and only for budget 2, since a requeue there costs the epoch in flight. Each job writes an
`in_progress` row at start and an `ok` row at the end with the same `run_id` to
`analysis_out/usage_log.jsonl` (`tag_benchmark_86.py log-usage --host --gpu`, as #180 does), and
`scripts/analysis/slurm_usage.py` adds the `sacct` record to `analysis_out/compute_log.jsonl`
after each job (`compute_cost.md`). Both ledgers live in the main checkout and are committed with
the results, never left in the run directory. The cut and the prep stage record their wall-clock
in the cut summaries and the prep manifest respectively.

## 6. What we learn under each outcome

| outcome on tests (1) and (2), paired, 30 epochs | reading | next |
|---|---|---|
| `nnPU` > `clean`, `naive` ≤ `clean` | the store is usable supervision once absence is treated as unknown; ASSETS'24's negative result was the loss, not the data | train the 2.0 tag head on the full store; the review pass is a test set; item 6 (one-pass detect + tag) inherits the PU loss |
| `naive` > `clean` too | the untagged labels are closer to negatives than the two-rater study suggests, or the volume outweighs the label noise at this size | check which tags gained: if only *missing tactile warning*, the gain is the easy tag and the read is "more data helps the visible tag"; report per tag |
| neither > `clean` (and the 3-epoch read agrees) | the head is limited by reviewed labels, not by crops | stop cutting; the effort is the review pass and the closed loop (plan items 10–11); keep the arms' checkpoints for item 6's ablation |
| `nnPU` < `naive` | the priors are wrong or the unlabeled term at batch 4 is too noisy | the sensitivity run with Jon's priors and a batch-16 run, before any conclusion about PU |

Per-tag results are the content either way. The street-dependent tags (*points into traffic, not
level, landing space*) gained nothing from wider context at any field of view in item 4
(`context_fov_86.md` §4), which left two readings: they are limited by the labels
(positive-unlabeled, rater-dependent), or they need context and resolution together. This
experiment tests the first. A gain concentrated on those tags supports it; no gain on them, with
gains elsewhere, points to the second, which a wider single crop cannot test and a larger input or
a two-crop model would.

## 7. Not run, and caveats that travel with the numbers

- **One seed per arm.** Same caveat as item 4; the benchmark's seed variance is unmeasured. A
  second seed of `nnPU` at budget 1 (~3 h) is the cheapest way to put a number on it and is
  PROPOSED as part of the run if the paired difference at budget 2 is under ~0.02 mAP, the
  size of the leak bound in `tag_benchmark_86.md` §1.
- **Prior mis-estimation is confounded with label noise.** If `nnPU` fails, the priors are the
  first suspect (§3.2); the sensitivity run with Jon's 2024+ rates separates the two only
  partly, because those rates are one rater's threshold. A prior estimated from the data (e.g.
  the KM or TIcE estimators) is a further arm, not run.
- **Rater × era drift** (audit §7). The universe spans 2018–2026 and the reporting threshold moved
  over it; a PU loss with one prior per tag assumes a single labeling frequency. A per-era prior
  is the natural refinement and is not run.
- **Tier 1 not run.** The 66,260 unvalidated tag-era labels are excluded (§2.1).
- **"Tagged but not this tag" as negative, not run.** The ablation of §3.1's PROPOSED rule.
- **The expert-validate test set depends on its crops existing**, which is the same cutter run;
  until then tests (1) and (2) carry the result.
- **Training crops are an unpublished input** (§4.4). The control's crops are public.
- **The framing decision inherits item 4's result**, including its one-seed caveat and its
  interim control column (the #178 epoch-index-49 snapshot, until the re-run finishes).
- **The control is selected by a different rule** than the new arms (§5.3).
- **The 8 scored tags are the benchmark's**; a tag with no test positives cannot be scored no
  matter how many training labels it gains (`parallel lines`, `tactile warning`).

## 8. Decisions for Jon

1. **Universe:** tier 2 (validated correct, 295,957 tag-era labels) as proposed, or tier 1
   (362,217)?
2. **Framing:** cut at `fov25` (proposed: ties the control and `viewport` in item 4, centred,
   independent of the labeler's zoom) or `viewport` (ties too, and reproduces the control's
   framing, at the cost of a second JPEG encode and a per-label zoom); `fov50` is out, measured
   worse (−0.036 [−0.064, −0.009] mAP against the control). Re-cut the HF rows the same way, with
   no production crops in any arm?
3. **Priors:** ASSETS'24 validated base rates as the main prior, trusted-rater 2024+ as the
   sensitivity check?
4. **Arms:** `naive` + `nnPU` at both budgets, soft-negative at budget 1 only?
5. **Budgets:** 3 epochs (presentation-matched) then 30 epochs; ~3 GPU-days on the lab L40S
   allocation for the full set, against the plan's ~2?
6. **Selection:** a 10 % pano-grouped slice of HF train held out from the new arms, control
   left as published?
7. **Cut host and cap:** makelab2 at ≤ 6 workers, `nice`, after the #178 rerun finishes; makelab1
   over NFS only after a timed sample?
8. **Transfer:** per-city tars relayed through the desktop, or set up a direct makelab → klone
   path?
9. **Trainer changes first:** the prep memmap, resume and the loss switch as small PRs against
   `tag_benchmark_86.py` before any 30-epoch job?
10. **Tagged-but-not-this-tag** as unlabeled (proposed) or negative?
11. **Publish the cut crops** as a Hugging Face dataset so the arms are reproducible without the
    store, or record them as an unpublished input?
12. **Batch size** stays 4 for the comparison, batch 16 as a follow-up?
