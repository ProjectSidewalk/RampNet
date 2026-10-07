# Seam-wrapped NMS in `detect_peaks`

**Issue:** [#248](https://github.com/ProjectSidewalk/RampNet/issues/248). **Status:** the option is
shipped, default **off**. Whether to make it the default is still open (section 4).

## 1. What changed

`rampnet.subcell.detect_peaks` finds peaks with
`peak_local_max(..., exclude_border=False)`. Its non-maximum suppression does not wrap in x, so a
ramp sitting on the 360° seam can come back as two peaks, one at each edge of the panorama (see
[`seam.md`](seam.md)). A new keyword, `wrap_nms=True`, runs the same suppression on a cylinder:

- the maximum filter (window `2 * min_distance + 1`) uses mode `('nearest', 'wrap')`, so x is
  cyclic and y is padded as before;
- the greedy spacing pass rejects a peak within **wrapped** Chebyshev distance `< min_distance`
  of a stronger one already kept.

Everything else is what scikit-image does: candidates are strictly `> threshold`, a constant map
has no peak, and candidates are sorted by a stable descending sort, so ties break in row-major order.
It is a port of sidewalk-auto-labeler's `detectors/decode.py::_cylinder_peaks` (its PR #138,
`--border wrap`), without the labeler's 50-peak cap, because RampNet's extractor has no cap.

`wrap_nms` is independent of `wrap_x`, which only wraps the sub-cell **decode**.
`wrap_nms=True, wrap_x=True` is the labeler's `wrap` combination. With `wrap_nms=False` (the default)
the output is bit-identical to the bare `peak_local_max` call, so no published number changes.
`wrap_nms=True` together with `exclude_border=True` raises `ValueError`. The scipy import is lazy, inside a
`try` block, so the Hub's remote-code loader still requires only numpy to load the shipped
`rampnet_subcell.py`.

| path | flag / API | default |
|---|---|---|
| `rampnet.subcell.detect_peaks` | `wrap_nms=False` | off |
| `stage_two/evaluate.py` | `--wrap-nms` / `--no-wrap-nms`; result filenames gain `_wrapnms` and `metrics.json` gains `wrap_nms`. The heatmap cache key is unchanged (the heatmaps do not depend on it) | off |
| HF package | `RampNetModel.detect(..., wrap_x=False, wrap_nms=False)` | off; **not re-exported to the Hub in this change** |
| `stage_two/demo.py` | not wired | keeps skimage's `exclude_border=True` on purpose ([`subcell_decode_221.md`](subcell_decode_221.md) section 10) |
| `scripts/analysis/decode_e2e_221.py` and the analysis scripts that call `peak_local_max` directly | not wired | they reproduce committed numbers |

## 2. How often the default splits a seam ramp

No heatmaps are committed, so this is measured on committed **peaks**:
`docs/data/run_a_84_detections/*.json`, the peaks of the 8 Run A epoch checkpoints
([#84](https://github.com/ProjectSidewalk/RampNet/issues/84)) on the 1,000 `manual_gold`
panoramas. They were extracted with no TTA, a floor of 0.05, `peak_min_distance` 10 and
`exclude_border=False`, which the script checks against each file's `signature`. A
**straddling pair** is two peaks of one panorama that are ≥ 10 px apart (Chebyshev, on the
1024 x 512 grid) measured straight across the image, but < 10 px apart once x is wrapped. The
default spacing keeps both peaks of such a pair, and a wrapped spacing would not. A pair counts
at a threshold when both of its peaks are at or above it. All 8 epochs are pooled, so one
panorama counts up to 8 times.

| threshold | peaks | straddling pairs | panos with a pair | peaks `wrap_nms` would drop (estimate) |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 36,050 | 114 | 113 | 113 |
| 0.3 | 30,689 | 72 | 72 | 72 |
| 0.55 | 28,708 | 58 | 58 | 58 |

Here "panos" means panorama × checkpoint (8,000 in all). The split rate is about 0.2% of peaks at
every threshold, and stays roughly constant across epochs:

| checkpoint | pairs >= 0.05 | pairs >= 0.3 | pairs >= 0.55 |
| :--- | ---: | ---: | ---: |
| run_a_epoch_1 | 18 | 6 | 4 |
| run_a_epoch_2 | 20 | 10 | 8 |
| run_a_epoch_3 | 17 | 11 | 8 |
| run_a_epoch_4 | 10 | 7 | 6 |
| run_a_epoch_5 | 14 | 8 | 7 |
| run_a_epoch_6 | 12 | 10 | 8 |
| run_a_epoch_7 | 12 | 12 | 9 |
| run_a_epoch_8 | 11 | 8 | 8 |

**The pairs recur on a few panoramas, and most of those are panoramas whose ground truth is itself
split.** The table below joins the panoramas holding a pair to the GT seam adjudication
(`benchmark/manual_gold/seam_verdicts__jon.json`, [`seam.md`](seam.md) section 2). `one` means a
seam ramp that was marked twice and judged one ramp; that merge has **not** been applied to the GT.
`two` means two real ramps.

| threshold | distinct panos with a pair | GT seam verdict `one` | `two` | not adjudicated |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 31 | 8 | 2 | 21 |
| 0.3 | 15 | 8 | 2 | 5 |
| 0.55 | 11 | 8 | 2 | 1 |

The join is by panorama, but the pairs are the adjudicated ramps. On every adjudicated panorama,
every peak of every straddling pair lies within **6.36 px** (wrapped Chebyshev, on the heatmap grid)
of one of that panorama's two adjudicated GT marks, at all three thresholds. That is well inside the
0.022 match radius (about 22 px at 1024 wide).

At 0.55, 10 of the 11 panoramas are ones the seam adjudication already looked at. Eight of those
ten were judged one ramp, double-marked in the GT. In that case the model's split currently scores
as **two true positives** (the `2NgKmkIoU9nUwvjk6K5wtw` row in [`seam.md`](seam.md) section 2).
So, against the current, unmerged `manual_gold` GT, turning `wrap_nms` on should cost recall on
those panoramas rather than gain precision. Scoring `wrap_nms` fairly needs the GT merge first.

### Caveats (they travel with the numbers)

- **Run A checkpoints, not the published model.** `projectsidewalk/rampnet-model`'s own peaks would
  need a GPU re-dump of its heatmaps, which was not done.
- **The join is by panorama, checked by geometry.** Every pair peak on an adjudicated panorama lies
  within 6.36 px of an adjudicated GT mark (pinned by `--check`), so the pairs are the adjudicated
  ramps. The 21 / 5 / 1 unadjudicated panoramas have no verdict to compare with.
- **The "drop" column is an estimate.** It runs a wrapped greedy pass over the stored peaks. The
  real `wrap_nms` also runs a wrapped maximum filter over the heatmap. That filter can drop a peak
  whose stronger neighbour across the seam was never stored as a peak, and on the 4-px edge
  plateau (see the next caveat) it can pick a different pixel. Neither effect is visible without
  the heatmap.
- **Edge plateaus.** Under `align_corners=False`, the x8 upsample is constant beyond the outermost
  coarse centres, so hi-res columns 0-3 and 1020-1023 form an exact 4-px plateau in every row.
  When a seam peak suppresses the default's pick on such a plateau, `wrap_nms` can keep another
  pixel of the same plateau, in the same row and with the same score. This is the one case where a
  `wrap_nms` peak is not also a default peak (`tests/test_seam_nms_248.py`).
- **Not measured: any detection metric under `wrap_nms`.** No manual_gold AP, P or R was re-run.

## 3. Reproduce

```bash
python scripts/analysis/seam_pairs_248.py --markdown   # the tables above
python scripts/analysis/seam_pairs_248.py --check      # exits 1 unless the pinned counts hold
pytest -q tests/test_seam_nms_248.py                   # the option's contract
```

Both commands are stdlib only, run on CPU, and read only committed files. `--check` is registered in
`scripts/check_all.py` as `seam_pairs_248`.

## 4. Open decisions (not taken here)

1. **Make `wrap_nms` the default.** This needs a `manual_gold` re-evaluation, after the GT seam
   merge in [`seam.md`](seam.md) section 2 (otherwise the comparison is biased against it, as shown
   above), and an HF re-export.
2. **Should `wrap_x=True` imply `wrap_nms=True` in `RampNetModel.detect()`?** Today they are
   independent.
3. **The [#132](https://github.com/ProjectSidewalk/RampNet/issues/132) dataset seam duplicates.**
   Stage 1 double-labels about 1% of the training set at the seam ([`seam.md`](seam.md) section 3).
   Training on those duplicates teaches the model the very split that `wrap_nms` removes at
   inference.
4. **A GPU re-dump of the published model's peaks**, to replace the Run A estimate above.
