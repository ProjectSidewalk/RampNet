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
  - `roma_local` fixes 9 false positives and breaks none (sign test p = 0.004). All 9
    become `already_detected`, so there are no new hard positives.
  - all-mined 0.431 → 0.627 [0.49, 0.75]; hard-only 0.310 → 0.406.
  - `roma` gives 8 fixed / 0 broken. `mapa_posed_pair` gives 4 / 2 and is **provisional**:
    it is post hoc in #48, and its fresh-pair re-test is running.
- **Pooled.** all-mined 0.605 → 0.702 [0.61, 0.78] and hard-only 0.483 → 0.534 with
  `roma_local`.
- **GSV.** No arm helps.

## Cost

| step | where | wall-clock | GPU-h |
|---|---|---|---|
| build | desktop CPU | 8 s | 0 |
| cut-views (254 views, 211 panos) | makelab2 CPU | 40 s | 0 |
| `mapa_posed_pair` | makelab2 A40 (shared, free memory checked first) | 134 s | 0.037 |
| `roma` + `roma_local` | desktop RTX 3070 | 204 s | 0.057 |

Both GPU runs have `paid: false` rows in `analysis_out/usage_log.jsonl`. Nothing ran on
Tillicum and no paid API was called.

## Reproduce

The commands are in the labeler doc, "Reproduce (phase 2)". Environments are as in #48:

- MapAnything used makelab2's `crossview48_sfm/venv`, with `HF_HOME` / `TORCH_HOME` from
  `crossview48_sfm`.
- RoMa used `romatch` 0.1.2 + `kornia` 0.8.3, installed `--no-deps` into a scratch directory
  on `PYTHONPATH` beside `.venv`.

The views are not published. They are on makelab2 at `/homes/gws/jonf/mined158/views`
(`views.tar` sha256 `8a8bd69029736ffa1207de69484826b453109046485de028fadf2cabd7cccad6`) and
regenerate from the archive with `cut-views`.
