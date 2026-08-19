# The 360° seam

**Issues:** [#130](https://github.com/ProjectSidewalk/RampNet/issues/130), [#132](https://github.com/ProjectSidewalk/RampNet/issues/132). **Status:** findings settled; two fixes landed, two still open.

A panorama wraps: normalized `x = 0` and `x = 1` address the same column of pixels. Several
things in this codebase measured horizontal distance arithmetically instead, and each had
reimplemented that distance inline. This documents what that cost, what it did **not** cost,
and how to re-derive every number here from a clean clone.

---

## 1. Summary

**Three real defects:**

| defect | scale | where | status |
| :--- | :--- | :--- | :--- |
| Stage 1 double-labels ramps on the seam | 8,361 label pairs, ~1% of the published training set | `stage_one/dataset_generation/download_dataset.py` | open — fix targeted at RampNet 2.0 |
| Ground truth double-marks ramps on the seam | **10** confirmed duplicate pairs in `manual_gold` | `manual_labels/*.txt` | adjudicated; merge not yet applied |
| Cached detections dropped near the seam | a 3.5° blind strip in every committed `op_cache` | `scripts/analysis/threshold_sweep.py` | **fixed** (`f4c71c8`); caches need regenerating |

**Two claims made during this investigation and later retracted.** Both are recorded because a
retracted claim that nobody can find gets re-proposed:

| retracted claim | what it actually was |
| :--- | :--- |
| "RampNet finds under half the ramps on the seam" | our own extractor bug — `peak_local_max`'s `exclude_border` was left at skimage's default of `True`, discarding every peak within 10 columns of the array edge. Production (`stage_two/evaluate.py`) always passed `exclude_border=False` and was never affected. The model responds at **24 of 25** seam-band ramps; recall at the production setting is **0.96**, not 0.44. |
| "Stage 1 drops ~72% of labels near the seam" | the along-street axis. The identical density profile appears at the **anti-seam** (x=0.5), where there is no seam: 0.278× vs 0.283× at the centre bin, 0.413× vs 0.411× one bin out. `x=0` is north with the panorama's own heading removed, so both dips sit straight ahead of and behind the vehicle, where ramps are distant and sparse. |

**What was never affected:** the published evaluation path and the paper's numbers.

**The model is affected, but far less than the labels are** — and less than the retracted claim
said. Its *detections* at the deployed threshold are essentially unchanged (24 of 25 seam-band
ramps found, recall 0.96), but its *response* is measurably reduced for about a third of them.
See §4a; this is a correction to an earlier version of this document, which said the model had no
seam defect at all.

---

## 2. Ground truth: 10 ramps were marked twice

`manual_gold`'s GT comes from the independent 1,000-pano labelling pass (3,919 marks), which never
had a written rubric and never had [#43](https://github.com/ProjectSidewalk/RampNet/issues/43)'s
reviewer rule applied. `yolo_ground_truth()` does no dedup, so a ramp straddling the seam was
labelled once per edge and the second mark is an unclaimable false negative.

**Why the labeller could not see it:** `scripts/gt_gallery.py` — the point-adjudication viewer —
*clamped* its crop window (`left = min(max(px - half, 0), W - CROP_SIZE)`) instead of wrapping it.
The two halves of a seam ramp landed in separate 512 px crops taken 3,584 px apart. No reviewer
could have seen them as one object.

### The adjudication

14 candidate pairs (11 inside the 22.53 px match radius, 3 at 23.8–25.1 px — the radius is a
*scoring* boundary and duplication does not respect it). Adjudicated by Jon Froehlich under
`benchmark/RUBRICS.md` §4, verdicts in `benchmark/manual_gold/seam_verdicts__jon.json`:

| verdict | n | separation |
| :--- | ---: | :--- |
| **one ramp** | 10 | 2.95 – 12.81 px (1.04° – 4.50°) |
| **two ramps** | 4 | 17.28 – 25.06 px (6.08° – 8.81°) |

**Perfectly rank-ordered by separation**, with an empty band between 12.81 and 17.28 px.

This matters beyond the eleven: an automatic merge would have been wrong. `MJbCTDJV5advy8eYbL0V5Q`
sits *inside* the match radius at 17.28 px and is **two genuine ramps** — merging it would have
deleted a real ramp, in the direction that flatters our own recall. Conversely `manual_gold` holds
**234 within-radius pairs away from the seam**, 87 of them at near-identical elevation on the
horizon, which are overwhelmingly genuine adjacent far-field ramps. Adjacency is not duplication.

### Corroboration from the model

On the three pairs examined in detail, RampNet's own behaviour is informative and not uniform:

| pano | adjudicated | model output |
| :--- | :--- | :--- |
| `JrR9wwG_ynJrSP9ov3Rxrw` | one ramp | **1 detection** (0.67) — correct, and penalised for it: GT says two, so one mark is a phantom FN |
| `2NgKmkIoU9nUwvjk6K5wtw` | one ramp | **2 detections** (0.88, 0.35) — the model split it too. GT wrong *and* model wrong, so scoring counts 2 TP and looks perfect |
| `MJbCTDJV5advy8eYbL0V5Q` | two ramps | **2 detections** (0.90, 0.76) — both right |

The middle row is the reason GT dedup shifts *precision* as well as recall: merging that pair turns
the model's second detection from a TP into an FP.

---

## 3. Stage 1 double-labels ~1% of the published training set

`stage_one/dataset_generation/download_dataset.py` runs one 90°-FOV perspective view per government
ramp record, projects each crop-model heatmap back with `perspective_to_equirectangular`, and
max-combines into a 4096×2048 composite. That projection inverse-maps over every equirectangular
pixel, so it wraps correctly and a seam ramp's response lands properly split across both edges.

Then:

```python
peak_local_max(combined_heatmap, min_distance=40, threshold_abs=0.4*255, exclude_border=False)
```

`exclude_border` is correct here, but **`min_distance` cannot suppress across the seam** — the two
fragments are ~4,085 columns apart in array coordinates. One ramp, two labels.

Measured across all 384 parquet files of `projectsidewalk/rampnet-dataset`
(`analysis_out/stage1_seam_scan.json`):

| | value |
| :--- | ---: |
| panoramas / labels | 214,385 / 849,904 |
| seam-crossing label pairs | **8,361** |
| expected under a uniform-azimuth null | 485.0 |
| enrichment | **17.24×** (z ≈ 358σ, normal approximation) |
| share of all labels | **0.98%** |
| panoramas affected | 7,987 (3.7%) — train 5,664 of the duplicates |

**This does not rest on the null.** Independently: panoramas with a seam pair carry exactly
**+1.000** more labels than they have government ramp records (mean +0.591 against −0.408 without;
z = +14.0). If those pairs were two distinct ramps there would be two records and no excess.

⚠️ **Do not index-pair `curb_ramp_points_normalized` with `curb_ramp_coords`.** They are not
parallel arrays — only 65% of rows have equal lengths, and matching coord *i* to point *i* gives a
median azimuth residual of 113°. The generator max-combines per-ramp heatmaps into one composite
before extracting peaks, so the record→label correspondence is destroyed by construction.

### What threshold should the fix use?

The adjudication gives an empirical boundary rather than an inherited constant. In generator
pixels (×4 from matcher units): **one ramp ≤ 51.2, two ramps ≥ 69.1.**

| threshold | Stage 1 pairs merged | against the adjudicated 14 |
| :--- | ---: | :--- |
| current `min_distance=40` (10.0 matcher px) | 4,203 (50.3%) | 0 false merges, 5 duplicates left unmerged |
| **15.0 matcher px (midpoint of the gap)** | **6,312 (75.5%)** | **0 false merges, 0 missed** |
| 17.0 matcher px | 7,004 (83.8%) | 0 false merges, 0 missed |

Simply wrapping the existing `min_distance=40` is **safe but incomplete** — it makes no false merges
on the adjudicated set, and leaves half the duplicates. A threshold in the empirical gap catches all
of them and still merges none of the genuine pairs.

Distribution of the 8,361 Stage 1 pairs against the adjudicated boundary: **65.4%** in the
all-one-ramp zone (≤12.81 px), **20.1%** in the unevidenced gap, **14.5%** in the all-two-ramps zone.
So roughly one in seven is likely a genuine adjacent pair that must not be merged.

### RampNet 1.0 is not being corrected

Decision (Jon Froehlich, 2026-08-18): **do not modify the published 1.0 dataset.** It is the artifact
the paper's numbers were computed on; changing it means a downloader gets different data than the
paper used, which is worse for reproducibility than a documented defect. The fix belongs in the
generator, before RampNet 2.0's generation run — which regenerates from scratch anyway, so no
special rerun is needed.

---

## 4. Our cached detections dropped peaks beside the seam

`scripts/analysis/threshold_sweep.py::peaks_to_dets` called `peak_local_max` without
`exclude_border`. **skimage defaults it to `True`**, discarding every peak within `min_distance`=10
of the array edge — a 3.5° blind strip either side of the seam on a 1024-wide heatmap.
`operating_point_curve.py:487` builds every committed `op_cache` through that function.

| seam-band GT (n=25) | recall |
| :--- | ---: |
| cached extractor | 0.44 |
| `exclude_border=False` (production) | **0.96** |
| heatmap already ≥ 0.30 at the ramp | 24 / 25 |

`stage_two/evaluate.py:78` always passed `exclude_border=False`, so production and the paper were
never affected. Fixed in `f4c71c8` and pinned by
`tests/test_geometry.py::test_peaks_to_dets_keeps_peaks_next_to_the_seam`.

**The caches still need regenerating.** Until they are, RampNet's recall in every cache-derived
analysis is slightly understated. The corrected `manual_gold` row therefore depends on *both* the
GT merge and the cache regeneration, and is not quoted here until both land.

---

## 4a. The seam costs the detector response, for about a third of ramps

The retracted claim in §1 said the model was blind at the seam. It is not. But "the model is
unaffected" is also wrong, and this section is what replaces both.

The measurement is paired and within-ramp, and deliberately avoids peak extraction entirely — the
thing that produced the retracted result. For each ground-truth ramp, the **maximum raw heatmap
response inside the match radius**, taken twice: with the panorama as stored, and with it rolled
180° so the same ramp sits half a world from the seam. Ramps far from the seam *in the same
panoramas* are the control.

| | n | seam through it | rolled away | paired diff | **gained > 0.05** | lost > 0.05 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| seam-band ramps | 25 | 0.785 | 0.818 | +0.033 (t=1.20) | **9 / 25** | 3 / 25 |
| control, same panoramas | 111 | 0.848 | 0.845 | −0.003 (t=−1.69) | **0 / 111** | 3 / 111 |

**Read the rate, not the mean.** The mean shift is small and not individually significant, because
the effect is heterogeneous — it depends on how much of a ramp falls on each side of the split, so
averaging the badly-affected together with the unaffected dilutes it. The sharp statistic is that
rolling the seam away moves **no** control ramp by more than 0.05 and **9 of 25** seam ramps, with
individual gains reaching +0.32, +0.29 and +0.24.

**What this does and does not mean.** In this sample the lost response rarely pushed a ramp below
the 0.30 detection threshold — 24 of 25 were still detected — because most had headroom. Where a
ramp's response is already marginal, it would. So the seam is a real but second-order effect on
detection, and a first-order one only for faint ramps.

Reproduce with `python scripts/analysis/seam_response.py --panos-root benchmark`;
result in `analysis_out/seam_response.json`. n = 25 is small, and the seam-band ramps were not
stratified by how much of each ramp falls on either side of the split, which is the variable the
effect most plausibly depends on.

---

## 5. The code audit

The defect class was never "the seam" — it was cyclic distance re-derived inline at every site.
`rampnet/geometry.py` now holds one definition (`fold` is the primitive) and the callers use it.

**Fixed** (`eccadda`): `rampnet/metrics.py::greedy_match` (the shared matcher, via `wrap_x`),
`rampnet/detection_eval.py::score_pano` and its ignore-point fallback, `stage_two/evaluate.py`,
`stage_one/dataset_evaluation/evaluate.py`. **Fixed** (`f4c71c8`): `peaks_to_dets`.

**Was already correct:** `miss_taxonomy.py`, `silent_activation.py` (all three sites),
`equirect_tiling.dedup_points`, `crop_window_eval.py`, `box_gallery.py`, and `miss_gallery.py`
(which reprojects, so it is structurally immune).

**Still open:** `gt_gallery.py` clamps its crop window (the cause of §2's defect);
`operating_point_curve.py`'s crop renderer clamps; `size_analysis.py` builds a crop box with no
clamp at all and lets PIL pad.

### The one caller that must NOT wrap

`stage_one/crop_model/ps_and_manual_model/evaluate.py:206` matches at `scale_x=341/4` — **crop**
space, where the two ends of the x axis are different places. An unconditional wrap would have fixed
the panorama scorer and silently broken the crop model's numbers. Wrapping is therefore opt-in on
the generic matcher, and `tests/test_geometry.py::test_match_predictions_does_not_wrap_by_default`
pins it so a later "make it consistent" cleanup cannot undo the distinction.

---

## 6. Replication

Every command below runs from a clean clone. Inputs are committed, or published and named.

**Committed inputs:** `manual_labels/*.txt`, `analysis_out/op_cache/*.json`,
`benchmark/manual_gold/seam_verdicts__jon.json`, `benchmark/RUBRICS.md`.
**Published inputs:** panoramas — `projectsidewalk/rampnet-benchmark`; model —
`projectsidewalk/rampnet-model`; Stage 1 labels — `projectsidewalk/rampnet-dataset`.
**Local-only:** `benchmark/*/panos/` is the local copy of the published benchmark bundle.

```bash
# 1. The seam audit and the fixes -- CPU only, no network, ~80 s
pytest -q tests/test_geometry.py

# 2. Scan the published Stage 1 dataset (network; reads ~12 MB of label
#    columns via parquet column projection, NOT the 463 GB of images)
python scripts/analysis/stage1_seam_scan.py
#    -> analysis_out/stage1_seam_scan.json

# 3. Rebuild the adjudication deck and re-judge it
python scripts/analysis/seam_review.py --panos-root benchmark --rater <you> --blind
#    -> analysis_out/seam_review/index_<you>.html
#    judge, Export, save to benchmark/manual_gold/seam_verdicts__<you>.json

# 4. The gallery behind the figures (needs a GPU and the benchmark panoramas)
python scripts/analysis/seam_gallery.py --panos-root benchmark \
    --out analysis_out/seam_gallery
```

The deck's `manifest_digest` (`022a8686d324868d` for the 14-item list) is written into every export,
so two raters cannot be compared across different item lists without it being obvious, and the
rubric text and version travel in the file itself.

---

## 7. Caveats, and what is not done

- **The first pass was not blind.** Each card displayed the pair's separation in pixels and degrees.
  The verdicts came out perfectly rank-ordered by separation, which is either the real signal or
  anchoring — **that pass cannot distinguish the two.** `--blind` was added afterwards and any
  second rater must use it. Until then, treat the 12.81 / 17.28 px boundary as provisional.
- **n = 14, one rater.** The empty band between 12.81 and 17.28 px means the boundary is bracketed,
  not located. Nothing was observed inside it, and 20.1% of the Stage 1 pairs fall there.
- **The rubric is retroactive.** The original 1,000-pano pass had none; `benchmark/RUBRICS.md` §4 was
  written on 2026-08-18, after the defect was found. A second pass measures labelling disagreement
  only if it uses the same version.
- **Not done:** the GT merge itself; the Stage 1 generator fix; regenerating the `op_caches`; the
  three remaining clamping viewers in §5; the 234 non-seam within-radius pairs, which have never
  been adjudicated and have no tool.
- **§4a is small.** n = 25, and unstratified by how much of each ramp straddles the seam — the
  variable the effect most likely depends on. It establishes that the seam costs response, not how
  much, nor for which ramps.
- **Two retracted claims** are in §1 rather than deleted. Both survived internally-consistent
  measurement — one survived a designed falsification test — and were caught only by a rendering that
  disagreed with the numbers, and by a control that should have been run first. The lesson is in the
  order: verify the instrument before the subject.


---

## Appendix: the note published on the Hugging Face dataset card

Kept here so the published wording is version-controlled and cannot drift from this report.
The card is at `projectsidewalk/rampnet-dataset`.

> ### Known limitation: duplicate labels at the 360° seam
>
> Panoramas wrap — the left and right edges of an equirectangular image are the same place — but
> this dataset's label generator extracts peaks without suppressing across that wrap. A curb ramp
> sitting on the seam can therefore be labelled **twice**, once per edge.
>
> Measured across the full dataset: **8,361 duplicate label pairs among 849,904 labels (0.98%)**,
> affecting **7,987 of 214,385 panoramas (3.7%)**. Panoramas containing such a pair carry exactly
> +1.000 more labels than they have source government ramp records.
>
> **If you are training on this dataset**, the practical effect is a small number of near-duplicate
> targets close to `x ≈ 0` / `x ≈ 1`. Duplicate pairs sit roughly 1–9° apart. On a
> 14-pair human adjudication, separations up to ~4.5° were always one physical ramp and those
> from ~6° up were two genuine adjacent ramps — but that sample is small and the rater could
> see each pair's separation, so treat the boundary as indicative rather than settled.
>
> **This dataset is not being changed.** It is the exact artifact the ICCV'25 paper's numbers were
> computed on, and replacing it would break reproduction from the published inputs.
>
> The released [model](https://huggingface.co/projectsidewalk/rampnet-model) is affected less than
> the labels are, but not unaffected: rolling a panorama so the seam falls elsewhere raises the
> model's response at the ramp by more than 0.05 for 9 of 25 seam-adjacent gold-set ramps, against
> 0 of 111 controls. In that sample it rarely changed whether the ramp was detected, but it would
> where the response is already near threshold.
>
> Full analysis: [`docs/seam.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/seam.md)
> and [issue #132](https://github.com/ProjectSidewalk/RampNet/issues/132).
