# Crop cutter: label crops from the makelab2 pano store

**Issue:** [#86](https://github.com/ProjectSidewalk/RampNet/issues/86), RampNet 2.0 plan item 2b
(`docs/rampnet2_plan.md` §4). **Run date:** 2026-09-22; re-run the same day with crop cutter
v3 after the [PR #177 review](https://github.com/ProjectSidewalk/RampNet/pull/177), which changed
the default tilt (section 3 keeps the v2 numbers as a labelled arm). **Code:** `rampnet/crops.py` (geometry),
`scripts/crop_cutter.py` (CLI), `scripts/analysis/crop_cutter_validation.py` (validation and
coverage), tests in `tests/test_crops.py`. **Committed outputs:** `docs/data/crop_cutter/`.

Production stores a crop only for labels placed since 2023-10-12 (196,556 human CurbRamp labels,
per the PR #175 audit). Items 4 (context experiment) and 5 (PU training) need crops for labels
older than that, at fields of view we choose. This cuts one for any `(city, label_id, field of
view)` from the pano archive on makelab2.

> **The pano store is an unpublished local input.** It is the Project Sidewalk scraper's archive,
> `/projects/makeabilitylab/sidewalk_panos/Panoramas` on makelab2 (55 city directories,
> `<city>/<pano_id[:2]>/<pano_id>.jpg`, beside `old_scrapes/`, `oldest_scrapes/` and
> `scrape_queue.log`). It is listed as an open input in `docs/replication.md`. Nobody outside the lab can re-cut the crops or re-derive
> the similarity numbers below. What a reader **without** the store can still check: all of
> the geometry, because every test runs on a synthetic pano and on the committed label rows
> (`pytest tests/test_crops.py`, including the replay of the 200 real validation labels into
> their viewports); the coverage input and per-row result (`coverage_input.csv`,
> `coverage.csv`); and the per-crop SHA-256 of every crop the validation run wrote
> (`cut_run/manifest_*.jsonl`), which proves a re-cut identical or not (for the same `--fov`
> set; section 1, Sampling).

## 1. What a crop is

A crop is a **perspective (gnomonic) view**, rendered from the equirectangular pano, not a
rectangle cut out of it. The equirect stretches a scene horizontally by 1/cos(latitude), and a
curb ramp sits 10 to 40 degrees below the horizon, where that stretch is 1.02 to 1.31 and varies
across the crop. A gnomonic view is what a camera shows, what the Project Sidewalk viewer
shows, and what every crop in the HF tag set is. A plain equirect window (CropRunner's framing)
is available with `--projection equirect` for anyone who needs to compare against those crops.

Two ways to aim the view:

| `--fov` value | aim | label lands at | field of view | use |
|---|---|---|---|---|
| a number, e.g. `60` | straight at the label's stored `pano_x`, `pano_y` | the exact centre (with the default tilt) | that many degrees across the width | items 4 and 5 |
| `viewport` | the labeler's stored `heading`, `pitch` | `(canvas_x, canvas_y)` × 2 (with the default tilt) | the viewer's FOV for the stored `zoom` | reproducing the HF crops |

Output is 3:2 and 1440 px wide by default (`--size`, `--aspect`), the shape of production and HF
crops. The named file is `<city>__<label_id>__fov<deg>.jpg` (`fov22p5` for 22.5°,
`__viewport.jpg` for the viewport, `_eq` for `--projection equirect`, and `_tilt<conv>` only
when `--tilt` is not the default, e.g. `fov60_tiltnone`).

### Geometry convention

Normalized pano coordinates are `X = pano_x / pano_width`, `Y = pano_y / pano_height`.

```
lon = (X - 0.5) * 360      degrees, positive to the right (clockwise)
lat = (0.5 - Y) * 180      degrees, positive up
```

The pano's centre column looks along `camera_heading`, so `lon = heading - camera_heading`.
That is exactly Project Sidewalk's `calculatePanoXYFromPov`, inverted. It is the same
convention as `scripts/model_comparison/equirect_tiling.py` (the test suite cross-checks the
two), and it is what sidewalk-panorama-tools' `CropRunner` uses: x wraps at the seam, y does
not. Everything is resolution-independent. A store JPEG at a different resolution from the
label's `pano_width` is sampled at the same normalized point; 16 of 800 validation crops came
from such a pano (`dims_match: false` in the manifest), and CropRunner would have skipped
them as a `dims_mismatch`.

The view is a roll-free pinhole camera with focal length `f = (width / 2) / tan(fov_h / 2)`
output pixels, so `fov` is always the **horizontal** angle and the vertical one follows from
the aspect (`tan(fov_v / 2) = tan(fov_h / 2) / aspect`). The viewport FOV is SidewalkWebpage's
`get3dFov(zoom)`: 89.75° at zoom 1, 53° at zoom 2, 27.68° at zoom 3, as ported in
sidewalk-panorama-tools `reports/scripts/pov_replay.py`.

```
   equirect pano (16384 x 8192)                       gnomonic crop (1440 x 960)
  +-----------------------------------------+        +---------------------------+
  |                                         |        |                           |
  |---------------- horizon ----------------|        |      label at centre      |
  |                            *  label     |  --->  |            +              |
  |                    (13740, 4754)        |        |                           |
  +-----------------------------------------+        +---------------------------+
     lon = +121.90 deg, lat = -14.46 deg              f = 2687 px for fov 30
```

**Worked example** (seattle-wa:9, the first row of the Seattle rawLabels export): pano
16384×8192, `pano_x = 13740`, `pano_y = 4754`, so `lon = (13740/16384 − 0.5) × 360 = 121.90°`
and `lat = (0.5 − 4754/8192) × 180 = −14.46°`. A `--fov 30` crop looks there with
`f = 720 / tan 15° = 2687 px`; the vertical FOV is 20.26°. The labeler's viewport was
`heading 299.32`, `pitch −17.54`, zoom 3, on a pano whose `camera_heading` is 180.37, so the
viewport view looks at `lon = 118.95°`, `lat = −17.54°` with a 27.68° FOV, and the label
projects to (866.0, 323.6) in it, against Project Sidewalk's `canvas_x, canvas_y = 433, 162`
× 2 = (866, 324). For another `--aspect`, both canvas axes scale by `width / 720` about the
centre, because the view's focal length is set by its width. `tests/test_crops.py` pins this for synthetic rows built with Project
Sidewalk's own click math and for all 200 real validation rows.

**Sampling.** Bilinear, with x wrapping and y clamped. Before sampling, the pano is box-reduced
by the largest integer factor that keeps it at least as sharp as the view's centre (a 90° view
at 1440 px reduces a 16384-px pano 3×, 60° reduces 2×, 30° not at all), which is the
anti-aliasing. For JPEG panos the first power-of-two part of that reduction happens in the
decoder (`Image.draft`), which is most of the speed. Output JPEGs are baseline, quality 92,
4:4:4, no metadata, so their bytes are deterministic for a given Pillow/libjpeg build **and a
given decode**: the two v2 validation runs (before and after the version bump that added
`black_frac`) wrote 800 of 800 byte-identical crops. The decode is shared by every crop of one
pano in one run and is drafted for the sharpest of them, so a crop's bytes depend on which other
FOVs were cut with it: of the 200 viewport crops, the v3 run (cut beside 30°/60°/90°) and the v2
`--tilt mm` run (viewport alone) agree byte for byte on exactly the 106 whose `decode_width` and
`reduce` match, and on none of the other 94. Both are recorded in the manifest, so a re-cut is
provably identical when it uses the same `--fov` set.

**Camera tilt (`--tilt`).** The stored `heading`/`pitch` **and** `pano_x`/`pano_y` are in the
**level (viewer) frame**. Project Sidewalk's `povToPanoCoord` is a linear map of the viewer's
heading and pitch with no rig pitch or roll, while the GSV viewer shows the image rotated by the
rig's `camera_pitch`/`camera_roll` (sidewalk-panorama-tools `docs/cropper.md` names this as the
known y-error, [SidewalkWebpage#4784](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4784)).
So the stored point is where the labeler clicked in the viewer, not a pixel of the equirect
image, and the view has to be rotated into the image frame to show what they saw.
`--tilt <conv>` does that under one of four sign conventions (`pp`, `pm`, `mp`, `mm`: sign on
pitch, sign on roll). Section 3 measured that the viewer's convention is `mm`, which is, for a
level-frame direction at azimuth φ from the pano's centre column (φ = `heading − camera_heading`)
and elevation `lat`, to first order:

```
image_lat = lat + camera_pitch · cos φ + camera_roll · sin φ
```

(a direction straight ahead of the pano's centre column sits `camera_pitch` higher in the image
than in the viewer, one behind it `camera_pitch` lower; the roll term is untested because
`camera_roll` is empty in every GSV row measured). **The default for gnomonic
crops is `--tilt mm`** (`crops.VIEWER_TILT`), so the clicked point is at the centre of a
label-centred crop and at the canvas point of a viewport crop. `--tilt none` renders the raw
image around the stored pixel, which is off the click by the tilt; its manifest `label_px`
says where the click is. Tilt is applied only to GSV panos (`pano_source` `gsv`, or blank when
the labels file does not carry it): the measured convention is GSV's, and in several infra3d
rows `camera_pitch` equals the view pitch rather than a rig tilt. The manifest records
`pano_source` and `tilt_applied`. `--projection equirect` is a rectangle of the raw image and
cannot be rotated; it refuses `--tilt`, is centred on the stored pixel as CropRunner's windows
are, and reports in `label_px` where the click's image content is.

## 2. Reconciliation with prior art

| source | framing | what this cutter does about it |
|---|---|---|
| HF `sidewalk-tagger-ai-validated` crops (ASSETS'24) | 1440×960 PNG **screenshots of the labeling viewport**: the stored heading, pitch and zoom, label *off-centre* at `canvas_x/720, canvas_y/480` (the HF CSV's `normalized_x`/`normalized_y` equal those to 1e-7 on all 10,853 rows) | `--fov viewport` reproduces this framing from the stored geometry (section 3). The HF crops are not label-centred, so no centred FOV can match them pixel for pixel |
| sidewalk-tagger-ai `crop.py` | ±320 px square cut around the label from those screenshots, for the test split only | not reproduced; a consumer that wants it can crop the viewport output the same way |
| production crops since 2023-10-12 | browser canvas captures, 1440×960, no marker | same shape and size as the default output; same viewport framing |
| sidewalk-panorama-tools `CropRunner.py` (`docs/cropper.md`, sizing rule v2, 2026-08-19) | 3:2 **equirect** rectangle, width from a distance regression × 2.5 clamped to 8–90°, x wraps, y shifts instead of padding, capped at 1440 px stored | the wrap, the no-padding rule and the 3:2 aspect are kept; the equirect cut is `--projection equirect` (y shift reported as `equirect_shift_px`). The distance-adaptive width is **not** implemented: items 4 and 5 need fixed FOVs, and v2's median window, 24.9°, sits inside the 30–90° range cut here. That repo's `CLAUDE.md` says CropRunner is being replaced |
| RampNet `equirect_tiling.py` / `dump_views.py` | gnomonic views for VLM and depth runs, nearest-neighbour sampling | same axis convention and point math (cross-tested); bilinear sampling with a box pre-reduction instead of nearest-neighbour, because these crops are model inputs at up to 3× downsampling |
| `docs/crop_window_eval.md` (#114) | scores window-sizing rules against gold boxes; found `manual_labels` w/h are tactile-pad marks, not apron extents | not re-scored here: this item fixes the FOV per experiment, it does not choose one. The gold-extent results are the input to that choice (item 4) |

Which FOVs item 4 should use is item 4's decision. The validation run cut 30°, 60° and 90° as
a bracket: 90° is about the zoom-1 viewport (54 % of HF labels), 30° about zoom 3 (18 %), and
v2's distance rule lands between 8° and 90° with median 24.9°.

## 3. Validation against the HF crops

**Sample.** 200 HF validated CurbRamp labels, the first 200 in a seeded shuffle (seed 86) whose
pano is in the store (`validation_sample.csv`): seattle-wa 57, chicago-il 46, oradell-nj 26,
columbus-oh 24, newberg-or 17, cdmx 10, pittsburgh-pa 10, spgg 7, amsterdam 2, walla-walla 1;
zoom 1/2/3 = 97/50/53; 136 placed in 2021 or later, 64 before. Only these 200 HF crops were
fetched (by HTTP range from the zip, about 600 MB, never the 30 GB archive).

**Metric.** Each HF crop against the cutter's viewport crop of the same label, both converted
to grey and resized to 360×240: zero-mean normalized cross-correlation (NCC), SSIM, and the
residual translation by phase correlation, reported as an angle at the view centre. The null
pairs each viewport crop with a *different* label's HF crop from the same city. The HF crops are
read from dataset revision `6e3a116` (pinned in the script; each cached crop is checked against
that revision's zip CRC-32, and its sha256 is in `validation_per_label.csv`).

**v3 (current, crop cutter v3, default `--tilt mm`)**: `validation.json`,
`validation_per_label.csv`, `cut_run/manifest_*.jsonl`.

| crop | n | NCC median | NCC p10 / p90 | NCC ≥ 0.5 | SSIM median | residual shift, median / p90 |
|---|---:|---:|---:|---:|---:|---:|
| viewport, default (`--tilt mm`) | 200 | **0.780** | 0.487 / 0.948 | 88.5 % | 0.458 | 0.71° / 2.17° |
| viewport, `--tilt none` | 200 | 0.745 | 0.394 / 0.904 | 84.5 % | 0.411 | 0.95° / 2.94° |
| viewport, `--tilt pp` | 200 | 0.671 | 0.317 / 0.866 | 76.0 % | 0.389 | 1.36° / 5.10° |
| null: another label, same city | 199 | 0.098 | — / 0.348 | — | 0.275 | — |

By zoom (default): NCC 0.822 / 0.790 / 0.715 and residual 0.79° / 0.74° / 0.61° at zoom
1 / 2 / 3. By era: 0.800 (2021+, n = 136) against 0.741 (pre-2021, n = 64); residual 0.65°
against 0.86°. Paired, the default beats `--tilt none` on 71 % of labels (1 tie), median gain
+0.026.

**v2 (superseded, kept as the record of the first run, default was `--tilt none`)**:
`validation_v2.json`, `validation_per_label_v2.csv`, `cut_run/v2/`,
`assets/crop_cutter_contact_sheet_v2.jpg`. There `viewport` meant no tilt.

| crop (v2 names) | n | NCC median | NCC p10 / p90 | NCC ≥ 0.5 | SSIM median | residual shift, median / p90 |
|---|---:|---:|---:|---:|---:|---:|
| `viewport` (no tilt, the v2 default) | 200 | 0.746 | 0.394 / 0.904 | 84.5 % | 0.411 | 0.87° / 3.02° |
| `viewport_tiltmm` | 200 | 0.782 | 0.486 / 0.949 | 88.5 % | 0.455 | 0.70° / 2.17° |
| `viewport_tiltpp` | 200 | 0.671 | 0.317 / 0.866 | 76.0 % | 0.389 | 1.36° / 5.10° |
| null | 199 | 0.113 | — / 0.354 | — | 0.275 | — |

The v2 and v3 rows for the same tilt differ only in the third decimal, and only where the decode
differs (section 1, Sampling: 94 of the 200 viewport crops were decoded at a different draft
scale because the v2 `mm` and v3 `none` runs cut the viewport alone). The `pp` arm and the 60°
equirect arm are byte-identical between v2 and v3 (200 of 200 each). The null moved (0.113 to
0.098) because it is drawn against the default arm, which is now the tilted one.

![HF crop, then the cutter's default viewport, fov30, fov60 and fov90 crops, for 8 seeded labels](assets/crop_cutter_contact_sheet.jpg)

*Columns: the HF crop; the cutter's viewport crop; label-centred 30°, 60°, 90°, all v3 defaults.
Yellow cross: the canvas point on the HF crop, the manifest's `label_px` on the cuts (the
centre, for the label-centred ones).*

What this says:

- **The viewport crop reproduces the HF framing.** Median NCC 0.78 against 0.10 for an
  unrelated crop of the same city, and the contact sheet is visually the same picture. The
  residual is a sub-degree to few-degree translation, the size of the 1–3° tilt residual
  sidewalk-panorama-tools reports.
- **The viewer's tilt is `mm`: image_lat = level_lat + camera_pitch · cos φ.** `mm` beats no
  tilt on 71 % of labels and cuts the median residual from 0.95° to 0.71°; `pp` is worse than no
  tilt on every summary. The gain grows with the displacement the tilt predicts,
  |camera_pitch · cos φ| (v2 per-label data, joined to `validation_sample.csv`; the PR #177
  review's table, re-derived here):

  | predicted displacement | n | median NCC gain, mm over none | mm wins | median residual drop |
  |---|---:|---:|---:|---:|
  | ≤ 0.25° | 72 | +0.004 | 57 % | 0.02° |
  | 0.25–0.75° | 61 | +0.039 | 72 % | 0.24° |
  | 0.75–1.5° | 36 | +0.045 | 69 % | 0.27° |
  | > 1.5° | 31 | +0.097 | 90 % | 1.23° |

  Spearman(displacement, gain) = 0.37. `camera_roll` is empty on every one of the 200 rows (and
  on 10,853 of 10,853 HF rows; only the two Mapillary rows of the tag-era sample carry it), so
  `mm` and `mp` are the same run and **the roll sign is untested**.
- **Why this is the default, and what it means for items 4 and 5.** Because the stored
  `pano_x`/`pano_y` are the click in the level frame, an untilted label-centred crop is centred
  on the raw image pixel at those coordinates, not on what the labeler clicked. Projected over
  all 15,751 GSV rows of `coverage_input.csv`, the click sits median 23 px, p90 83 px, p99 205 px
  and at most 661 px from the centre of an untilted `--fov 30` 1440-px crop (11 / 38 / 95 /
  307 px at `--fov 60`). **Item 4 (and item 5) inherit `--tilt mm` as the default** unless they
  decide otherwise; `--tilt none` is kept for parity with earlier crops, and its manifest
  `label_px` then gives the click's true position (`crops.label_pixel`), not the centre.
- **What the remaining mismatch is.** The 9 labels with NCC below 0.3 under the default (12
  without tilt) include: one pano in the store with missing tiles stitched as black
  (seattle-wa:147235); pre-2021 rows whose viewport is displaced in heading or pitch, the known
  legacy-era POV truncation and `camera_heading` drift (pov_replay.py in
  sidewalk-panorama-tools); and one store pano with a visible stitching break.
- **Not measured:** whether the label-centred crops put the *ramp* at the centre. That is the
  placement question (the stored point is a click at ground contact), sidewalk-panorama-tools
  #54's to answer, and it needs gold, not an HF comparison.

**Damaged panos (`black_frac`).** Some store panos have tiles that never downloaded and were
stitched as black. seattle-wa:147235 is 59 % black in its viewport crop and 60–100 % across
its FOVs, while ordinary deep shadow reaches a few percent (walla-walla:480 at 4 %). **Exclude a
crop with `black_frac > 0.5`**; treat 0.01–0.5 as "look at it". At about one damaged pano in 195
this is ~0.5 % of item 5's crops; the status stays `ok` so the rule is the consumer's.

## 4. Store coverage

Probed 2026-09-22 by file existence on makelab2 (`coverage.json`, per-row `coverage.csv`),
re-run the same day after adding the `la-piedad-old → la-piedad` directory mapping.

| set | labels | pano in store | distinct panos | in store |
|---|---:|---:|---:|---:|
| HF validated CurbRamp (all rows matched to the audit cache) | 10,853 | **10,848 (99.95 %)** | 6,295 | 6,291 (99.94 %) |
| tag-era sample: 5,000 human CurbRamp labels placed ≥ 2018-04-29, seeded from 362,217 | 5,000 | **4,958 (99.16 %)** | 4,876 | 4,835 (99.16 %) |

HF by city: 100 % in nine cities; newberg-or 1,226 / 1,231 (99.59 %). Four of the 10,857 HF rows
have no rawLabels row today and are not in the 10,853.

Tag-era sample by source: GSV 4,956 / 4,972 (99.68 %), Mapillary 2 / 2 (richmond-va, both
11000×5500 and present: the store holds them, so they will be cut, untilted), infra3d 0 / 26
(zurich-infra3d 25, winterthur-infra3d 1; the store has no infra3d directory). The 16 GSV misses
are single labels in kaohsiung (5 of 268), seattle-wa (4 of 1,400), dc (2 of 72), cdmx,
chicago-il, newberg-or, spgg and walla-walla (1 each). Looked at one by one: 3 of the Seattle
misses have a `.jpg` only in `old_scrapes/scrapes_dump_seattle_badsize/` (not used; the name
says why), and newberg-or:10546 and walla-walla:9585 have a `.depth.npz` beside no `.jpg`
(so do 3 of the 5 HF newberg misses, on 2 panos). Labels without a production crop (placed
before 2023-10-12) are covered at 99.61 % (2,270 / 2,279), those with one at 98.79 %
(2,688 / 2,721): the newest labels are the likeliest to be on a pano the scraper has not
reached yet. Per-city rows for all 51 cities in the sample are in `coverage.json`.

**Why the tag-era frame is 362,217, not #175's 361,863 or the plan's ~354k.** The frame here is
every non-SidewalkAI CurbRamp row of the audit's rawLabels cache whose `time_created` string is
≥ `2018-04-29` (a string comparison of ISO-8601 UTC timestamps). The PR #175 audit parses the
timestamps with `pandas.to_datetime(errors="coerce")`, which turns the 499 timestamps written
without fractional seconds (`2022-03-05T14:58:59Z`) into NaT and so drops them; 354 of those are
in the tag era, which is exactly 362,217 − 361,863 (seattle-wa 91, chicago-il 74, columbus-oh 18,
…). Both see 505,293 human labels. The plan's ~354k (118k tagged + ~236k untagged, §4 item 5)
predates the audit: it is the rounded scratch census on #86 (505,193 human labels), and is not
re-derived here.

A file that exists is not a file that decodes. In the 200-label validation all 195 panos opened
and one was damaged (above); the whole-store decode rate is not measured.

## 5. Throughput

makelab2, CPU only, 8 worker processes, Python 3.9, numpy 1.23.5, Pillow 10.0.1 (the system
python; no conda env needed):

| run (crop cutter v3) | crops | wall-clock | crops/s |
|---|---:|---:|---:|
| viewport + 30° + 60° + 90°, 1440×960, default tilt | 800 (195 panos) | 100.8 s | 7.9 |
| re-run of the first, nothing to do | 0 (800 skipped) | 0.2 s | — |
| viewport, `--tilt none` / `--tilt pp` | 200 each | 40.7 s / 38.0 s | 4.9 / 5.3 |
| 60° equirect window | 200 | 17.5 s | 11.5 |

The v2 runs (default no tilt) took 111.2 s for the first row, 45.0 s / 44.0 s for the `pp` /
`mm` viewports, 21.2 s for the equirect window and 0.6 s for the empty re-run
(`cut_run/v2/summary_*.json`); tilt adds no measurable cost. The first row cuts four crops per
decode; a single-FOV run decodes once per crop and is slower per crop. At ~8 crops/s, item 4's
10,857 labels × 3 FOVs is about 1.1 h; item 5's ~362k tag-era labels at one FOV is one to two
days at 8 workers, less with more workers. makelab2 has 48 cores but was short of free memory
on the day (~24 GB); one full-resolution decode holds ~0.4 GB as a PIL image and another ~0.4 GB
as its numpy copy, so a worker peaks near 0.8–1 GB at FOVs up to ~40° (no reduction) and 8
workers need ~7–8 GB. No GPU and no billed compute, so there is no `compute_log.jsonl` row; the
per-run summaries with host, versions and timing are committed in `cut_run/summary_*.json`.

## 6. Manifest and resumability

One JSONL row per attempt, keyed by the output name; the latest row wins. `ok` rows carry the
pano id and `pano_source`, the store path used, the label's `pano_x`/`pano_y`/`pano_width`/
`pano_height`, the store image's real size, the view (`yaw_deg`, `pitch_deg`, `fov_h_deg`,
`fov_v_deg`, `width`, `height`), `tilt` (requested) and `tilt_applied` (what was used: `none`
for a non-GSV pano), `camera_pitch`/`camera_roll`, the decode width and reduction factor, where
the clicked point lands in the crop (`label_px`: the centre of a default label-centred crop, the
canvas point of a default viewport, and offset by the tilt otherwise), `black_frac`, byte count
and `sha256`. Other statuses: `missing_pano`, `no_geometry` (label not in the geometry source, or
the viewport columns are empty), `out_of_frame` (`pano_y` outside the image; x is never out of
frame), `bad_aspect` (store image not 2:1), and `error` (a corrupt pano or failed write: the only
status that makes the exit code 1). A re-run skips an `ok` crop only if its file exists and it
was cut with the same `width`, `height`, `jpeg_quality`, `projection`, `tilt` and
`crop_cutter_version`; otherwise it is re-cut and the new row wins (`recut_changed_params` in the
summary). `missing_pano` rows are re-checked and logged again only if their status changes, so
re-running over an unchanged store appends nothing. A malformed manifest line (the tail of a
killed run) is skipped with a warning, its crop is re-cut, and the next row starts on a new
line. `--fov` is checked to lie in (0, 180) before anything runs, and `--projection equirect
--tilt <conv>` is refused. Floats are rounded and every file is written with `newline=""`.

## 7. Reproduce

The geometry, from a clean clone, with no store and no network:

```bash
pytest -q tests/test_crops.py
```

Everything else, in order, from a clean clone plus the store. Paths are relative to the repo
root; steps marked makelab2 run in a checkout there (`git pull` the same commit first).

**Step 0, the label geometry** (local). The rawLabels cache and the HF index come from the PR #175
audit script, `scripts/analysis/ps_supervision_audit.py` (on `main` once PR #175 merges; until
then `git checkout origin/audit/ps-supervision-86 -- scripts/analysis/ps_supervision_audit.py`).
The cache is the deployments' live `/v3/api/rawLabels?filetype=csv` responses, so a re-fetch
moves as labels accrue: the committed `coverage_input.csv` is the frozen output of step 1, and
steps 2 on can start from it without steps 0–1.

```bash
python scripts/analysis/ps_supervision_audit.py fetch      # -> analysis_out/ps_audit/raw/
python scripts/analysis/ps_supervision_audit.py hf-index   # -> analysis_out/ps_audit/hf_validated_curbramp_index.csv

# 1. local: the coverage input (HF rows + seeded 5k tag-era sample)
python scripts/analysis/crop_cutter_validation.py sample \
    --hf-index analysis_out/ps_audit/hf_validated_curbramp_index.csv --raw-cache analysis_out/ps_audit/raw

# 2. makelab2: store coverage (stdlib only), then copy coverage.csv/.json back
python3 scripts/analysis/crop_cutter_validation.py coverage \
    --store /projects/makeabilitylab/sidewalk_panos/Panoramas --probed 2026-09-22

# 3. local: the 200-label validation sample
python scripts/analysis/crop_cutter_validation.py pick

# 4. makelab2: cut the validation run (crop cutter v3), including the empty re-run
S=/projects/makeabilitylab/sidewalk_panos/Panoramas
L=docs/data/crop_cutter/validation_sample.csv
O=/homes/gws/jonf/nobackup/crop_cutter/validation_v3
python3 scripts/crop_cutter.py --labels $L --store $S --fov viewport --fov 30 --fov 60 --fov 90 --out $O --manifest $O/manifest_main.jsonl --workers 8 --summary $O/summary_main.json
python3 scripts/crop_cutter.py --labels $L --store $S --fov viewport --fov 30 --fov 60 --fov 90 --out $O --manifest $O/manifest_main.jsonl --workers 8 --summary $O/summary_rerun.json
python3 scripts/crop_cutter.py --labels $L --store $S --fov viewport --tilt none --out $O --manifest $O/manifest_tiltnone.jsonl --workers 8 --summary $O/summary_tiltnone.json
python3 scripts/crop_cutter.py --labels $L --store $S --fov viewport --tilt pp --out $O --manifest $O/manifest_tiltpp.jsonl --workers 8 --summary $O/summary_tiltpp.json
python3 scripts/crop_cutter.py --labels $L --store $S --fov 60 --projection equirect --out $O --manifest $O/manifest_eq.jsonl --workers 8 --summary $O/summary_eq.json
cd $(dirname $O) && tar cf validation_v3.tar validation_v3

# 5. local: bring the crops back, commit the manifests and summaries
mkdir -p analysis_out/crop_cutter && scp makelab2:/homes/gws/jonf/nobackup/crop_cutter/validation_v3.tar analysis_out/crop_cutter/
tar xf analysis_out/crop_cutter/validation_v3.tar -C analysis_out/crop_cutter
V=analysis_out/crop_cutter/validation_v3
cp $V/manifest_*.jsonl $V/summary_*.json docs/data/crop_cutter/cut_run/

# 6. local: compare against the HF crops (fetches the 200 by HTTP range from revision 6e3a116)
python scripts/analysis/crop_cutter_validation.py compare --crops $V \
    --manifest viewport=$V/manifest_main.jsonl --manifest viewport_tiltnone=$V/manifest_tiltnone.jsonl \
    --manifest viewport_tiltpp=$V/manifest_tiltpp.jsonl --cut-summary $V/summary_main.json \
    --sheet-manifest $V/manifest_main.jsonl
```

Step 6 writes `validation.json`, `validation_per_label.csv` and the contact sheet. To check a
re-cut against the committed one, compare the `sha256` column of the new manifests with
`docs/data/crop_cutter/cut_run/`. The v2 run is reproduced the same way at commit `118ff28`
(its commands are in that commit's copy of this section); its outputs are in `cut_run/v2/` and
the `_v2` files.

Cutting for another label set only needs a `--labels` file: a CSV with `city,label_id` and the
geometry columns (as `validation_sample.csv`, including `pano_source`), or bare `city:label_id`
lines plus `--raw-cache <dir of <city>__rawLabels__*.csv>`. Audit city ids whose store directory
is named differently (`walla-walla` → `walla-walla-wa`, `la` → `la-ca`, `la-piedad-old` →
`la-piedad`, the Taiwan cities, `columbia`, `west-chester`) are mapped in `STORE_CITY_DIRS`;
`--city-dir city=dir` overrides.

## 8. Gaps

- **The store is unpublished**, so the crops and the similarity numbers cannot be re-derived
  outside the lab (see the note at the top for what can be).
- **Roll sign untested**; `camera_roll` is empty in the GSV data (section 3).
- **The tilt convention is measured on GSV only.** Non-GSV panos are cut untilted.
- **Placement of the ramp in a label-centred crop is not measured** (section 3, last bullet).
- **Whole-store decodability not measured**; existence only.
- **Only CurbRamp** labels were sampled; nothing in the cutter is label-type specific.
- **infra3d panos** are not in this store; those labels come back `missing_pano`. Mapillary
  panos are (both sampled ones are present).
- **The HF comparison is on 200 labels**, 64 of them from before 2021 where stored POVs are
  known to drift; the per-era split above is small-n.
