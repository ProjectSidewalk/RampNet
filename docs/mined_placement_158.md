# Image-based placement of mined targets ([#158](https://github.com/ProjectSidewalk/RampNet/issues/158) phase 2)

This doc covers the RampNet side of #158 phase 2: running the #48 placement arms on the
labeler's mined candidates. The measurement itself, its tables and its caveats are in the
labeler's `docs/mined-precision.md`
([sidewalk-auto-labeler#110](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/110)).
That is where a placed pixel is adjudicated against the benchmark verdicts. Nothing in
this repo reads a verdict for this.

## What runs here

`scripts/analysis/mined_placement_158.py` turns the labeler's `sources.csv` into a
harness-shaped pair list. That is one row per mined candidate:

- the **source view** is the member the pre-registered source rule picked (nearest member
  camera to the target);
- the **other view** is the target pano, centred on the step-2 flat projection.

The script then drives registered #48 arms through the harness's own `Context` and
`run_arm`. The pair list has no answer columns at all, and the harness withholds
detections as it does for #48. The multi-view arms get a corner manifest in `_mv3d`'s
format, with one two-view corner per pair and `_mv3d.build_manifest`'s pose priors.

Outputs are in `analysis_out/mined_placement_158/`:

- `pairs.csv`, `keys.csv` (pair_id → city, site_id, pano_id) and `corners.json`;
- `predictions/<arm>.jsonl` + `.meta.json` for `mapa_posed_pair`, `roma` and
  `roma_local`.

## Result (details and caveats in the labeler doc)

The comparison is paired, candidate by candidate, against the step-2 flat projection on 127
candidates:

- **Richmond (Mapillary).**
  - Under the pre-registered 5 m world match, `roma_local` moves 9 false positives onto a
    detected ramp within 5 m and breaks none (sign test p = 0.004; the test itself was
    chosen after the predictions existed, and it survives Bonferroni over 3 arms). All 9
    become `already_detected`, so there are no new hard positives.
  - all-mined 0.431 → 0.627 [0.49, 0.75]; hard-only 0.310 → 0.406.
  - `roma` gives 8 fixed / 0 broken, and `mapa_posed_pair` 4 / 2.
- **The 9 are not shown to be the mined ramp** (post hoc, added after review). The detection
  each one lands on belongs to *another* fused site in 6 of the 9: a multi-pano site 5.4–11 m
  away. At least 2 are demonstrably a different ramp. Site 1287's own source pano detects
  both ramps separately, about 45° apart, and RoMa carried the click from one onto the
  target's detection of the other. Site 350 lands on the same detection. The other 3 land
  on singleton sites whose identity is undetermined. An earlier version of this doc (and the
  #158 comment) said `roma_local` "fixes" richmond's placement; **that is retracted**. What
  holds: the all-mined numbers, under the rubric, where a label on any real ramp within 5 m
  is correct. Read as "placed on the mined ramp", the paired count lies between 0 / 0
  (no discordant pair, so the sign test is undefined) and 7 / 0 (p = 0.016). It is 3 / 0
  (p = 0.25) only if the 3 singleton landings, 8.1 m from the site and of undetermined
  identity, are *taken* to be the mined ramp; that is a reading, not a lower bound. The
  labeler's `scripts/mined_placement_attribution.py` produces the per-candidate table.
- **Pooled.** all-mined 0.605 → 0.702 [0.61, 0.78] and hard-only 0.483 → 0.534 with
  `roma_local`. But pooled `tp` falls 42 → 39. On GSV the same pull turns 7 true hard
  positives into `already_detected`. In 5 of the 7 the landed detection belongs to a site
  that the source pano also detects separately, 0.9–3.8 m away on the same corner.
- **GSV.** No arm helps.
- **Where the arms stand in #48.** [#220](https://github.com/ProjectSidewalk/RampNet/pull/220)
  (open; [results](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5895553631))
  re-tested the MapAnything arms against `proj_height_auto` on fresh pairs. It confirmed
  `mapa_posed_pair`, the arm run here, which stays robust under #220's sensitivity reads
  (GSV α/3 lower bound ≥ +0.105°; Mapillary +1.32° [0.73, 2.34]), and `mapa_posed_corner`.
  `mapa_k_pair` is confirmed as pre-specified but borderline on GSV: under #220's review and
  [sensitivity reads](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5896570175)
  its GSV lower bound is 0.000 / −0.009 / −0.048, so that cell is not confirmed under any of
  them. `mapa_k_pair` was best on Mapillary (+1.89°); it was not run in phase 2 (step 3 runs it; see below). `roma` and
  `roma_local` were **not** in that re-test. In #48 no matching arm beats auto on GSV, and
  their Mapillary gains come from a stratum #48 did not screen for multiplicity. `roma_local` is post hoc there. `roma`
  inherits `lg`'s post hoc 5° ground band, so it is not fully pre-specified either.
- **For step 3.** On a dense corner, the 5 m nearest-point world match cannot tell the mined
  ramp from a neighbour. Any later `already_detected` gain should carry the own / other
  attribution beside it.

## Step 3 ([#222](https://github.com/ProjectSidewalk/RampNet/pull/222))

Step 3 of #158 (peak-anchored targets) is in the labeler:
[sidewalk-auto-labeler#112](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/112),
`docs/mined-precision.md` § Step 3. On the RampNet side it adds two things:

- **`mapa_k_pair` on the same 127 mined pairs.** It was named as an arm before scoring. It
  fell back on 1 pair and took 50 s on the makelab2 A40.
  - Under the rubric, richmond reads 11 fixed / 0 broken. All 11 become `already_detected`,
    so, as with `roma_local`, they are not shown to land on the mined ramp.
  - Its #48 status is confirmed as pre-specified but borderline on GSV (above).
- **`paid: false` ledger rows** for that run and for the labeler's richmond and bend floor
  passes. The bend pass failed its instrument gate and was not used. The likely cause, found
  after the review of #222 and #112, is the pixel source: bend's production run fed the model
  Google's zoom-3 rendition, while the floor pass read the archive's max-zoom JPEGs (labeler
  doc, step 3 "Post hoc").
- **`mapa_k_pair.meta.json` provenance.** As written by the run, it says
  `"pre_specified_for_158": false`, because `PLANNED_ARMS` then listed only the phase-2 arms.
  The step-3 plan named the arm before the run and before scoring, so the file now also
  carries `"pre_specified_in": "step 3"` and a dated `provenance_correction`. `PLANNED_ARMS`
  records the plan per arm, and new metas write `pre_specified_in`. No prediction changed.

## Cost

| step | where | wall-clock | GPU-h |
|---|---|---|---|
| build | desktop CPU | 8 s | 0 |
| cut-views (254 views, 211 panos) | makelab2 CPU | 40 s | 0 |
| `mapa_posed_pair` | makelab2 A40 (shared, free memory checked first) | 134 s | 0.037 |
| `roma` + `roma_local` | desktop RTX 3070 | 204 s | 0.057 |
| step 3: `mapa_k_pair` (ended 18:13:07Z) | makelab2 A40 (shared, free memory checked first) | 50 s | 0.014 |
| step 3: labeler floor passes, richmond + bend | makelab2 A40 | 214 s + 245 s | 0.059 + 0.068 |

Every GPU run has a `paid: false` row in `analysis_out/usage_log.jsonl`; the RoMa row
(`mined-placement-158:roma+roma_local`) covers both RoMa arms. The two floor-pass rows'
`ts` are the end of each pass rounded to the minute, so they are approximate. Nothing ran
on Tillicum and no paid API was called.

## Reproduce

The commands are in the labeler doc, "Reproduce (phase 2)". Environments are as in #48:

- MapAnything used makelab2's `crossview48_sfm/venv`, with `HF_HOME` / `TORCH_HOME` from
  `crossview48_sfm`. That holds for step 3's `mapa_k_pair` too: its meta records torch
  2.6.0+cu124, which is that venv's, and its log reads the DINOv2 weights from
  `crossview48_sfm/torch_home`. The command is in the labeler doc, "Reproduce (step 3)".
- RoMa ran on the desktop RTX 3070 against RampNet's `.venv` (torch 2.6.0+cu126, numpy
  2.5.1, opencv 5.0.0, python 3.12.10), plus these packages, installed `--no-deps` into a
  scratch directory put first on `PYTHONPATH`:
  - `romatch==0.1.2`
  - `kornia==0.8.3`, with `kornia_rs` 0.2.0
  - `loguru==0.7.3`
  - `win32_setctime` 1.2.0 (Windows only)

  As one command (Windows; drop `win32_setctime` elsewhere):

  ```bash
  pip install --no-deps --target <dir> romatch==0.1.2 kornia==0.8.3 kornia_rs==0.2.0 loguru==0.7.3 win32_setctime==1.2.0
  ```

  `docs/crossview_align_48/matching.md` §7 lists the same packages but not `kornia_rs`. The scratch
  directory was a Claude Code session scratchpad (`%TEMP%\claude\...\scratchpad\pkgs`). It is
  not durable, so reinstall from those lines.
- **Match cache.** `roma` ran first with `--extra match_cache=CACHE`. It filled `CACHE/roma/`
  (7.7 MB, 17:07–17:10Z, including first-use weight downloads) in 192.8 s.
  `roma_local` then read the same cache and took 4.2 s. So:
  - `roma_local` alone on an empty cache takes about as long as `roma`;
  - GPU nondeterminism can change a pair or two of the match set (#48 `matching.md` §2).

  The cache was in the same session scratchpad and is not kept.
- **Recorded versions.** The committed `roma*.meta.json` record cv2, kornia, numpy, python
  and torch. They name romatch 0.1.2 only inside `config`. Since the review fix,
  `predict` also records the installed `romatch` / `loguru` / `lightglue` versions and the
  `--extra` arguments (the match-cache path) in each new meta. The committed files are
  unchanged, because the labeler holds byte-identical copies.

The views are not published. They are on makelab2 at `/homes/gws/jonf/mined158/views`
(`views.tar` sha256 `8a8bd69029736ffa1207de69484826b453109046485de028fadf2cabd7cccad6`) and
regenerate from the archive with `cut-views`.
