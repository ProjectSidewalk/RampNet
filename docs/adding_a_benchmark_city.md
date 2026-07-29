# Protocol: adding a city to the validation benchmark

End-to-end runbook for taking a city from "not in the benchmark" to "fully integrated into
every analysis that quotes benchmark numbers." Written because the expensive half is not the
part people expect: the ~125-pano ground-truth review is one afternoon, but a new split
invalidates roughly a dozen committed numbers, two figures and four documents, and there is no
mechanism that tells you which.

**Use the checklist at the bottom.** The prose explains why each step exists; the checklist is
what stops a split from landing half-integrated.

## The two-repo boundary

RampNet and [`sidewalk-auto-labeler`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler)
split **by question, not by city** (see `benchmark/README.md`):

- *"What are the curb ramps in this city?"* → **auto-labeler**. It is the only repo that
  fetches pixels, so it also hands RampNet the imagery bundle.
- *"How good is the model, and is it improving?"* → **RampNet**. Ground truth, scoring, and
  the benchmark itself.

Phase 1 is auto-labeler work; phases 2–6 are RampNet.

## Phase 0 — decide what the city is *for*, before spending the review

Every split in the benchmark answers a question the others cannot, and that intent determines
how it is scored later. Decide it up front and write it down, because it is the thing that
gets lost:

| Split | What it exists to answer |
|---|---|
| richmond | OOD deployment on mixed Mapillary rigs |
| bend | in-domain GSV reference (a Stage-2 training city) |
| clovis | worst-case imagery — 2018 GoPro Fusion |
| morgantown | cleanest imagery; the control |
| annapolis | survey-grade rig; the far-field distance finding |
| budapest_district5 | non-US infrastructure + rubric transfer |
| manual_gold | in-distribution reference with **un-anchored** GT |

A city that duplicates an existing split's answer costs a day of review and buys a row in a
table. The most valuable additions right now are a **second GSV city** (bend's operating point
is confounded between "GSV" and "in-domain" — see `docs/operating_point.md`) and a **second
rater on budapest**, which is a review, not a new city.

Also decide now, and record it in the PR:

- **Does this split enter the pooled recommendation?** Default yes for a US/VA deployment city
  with verdict-grade GT. No for out-of-distribution or rubric-uncertain splits.
- **Which imagery tier is it?** If the rig is new, `tier_of()` needs a branch (see phase 5).

## Phase 1 — auto-labeler: detect the city and export the bundle

```bash
# in sidewalk-auto-labeler
python main.py <city.geojson> --source <mapillary|gsv>      # enumerate → thin → detect
# results land in runs/<name>/ — <name> defaults to the geojson basename
python scripts/export_benchmark.py runs/<city>/results.jsonl \
    --bundle <RampNet>/benchmark/<city>
```

`--bundle` samples the run and fetches native-res pixels. The sample is **stratified**, and the
strata matter downstream:

| stratum | n | why |
|---|---|---|
| `top` | 5 | the densest panos, always included — they make precision look good, so scoring excludes them from the *unbiased* column |
| `random` | 95 | spatially de-clustered detection panos — the honest sample |
| `empty` | 25 | zero-detection panos, so a silent model is measurable, not invisible |

125 panos total in the four newest splits; richmond has 124 and bend 110 (only 10 `empty`).
Keep 5/95/25 unless there is a reason not to; changing it makes the split non-comparable on
the unbiased column.

The exporter reconciles the archive against the records and writes `index.csv` (per-pano
sha256) and `decayed.txt`. **It exits non-zero on anything that means the archive is not
trustworthy** — a pano with no matching record, or a fetch that failed rather than a source
that dropped the image. Do not proceed past a non-zero exit.

Deliverable: `benchmark/<city>/{records.jsonl, panos/}`, with `panos/` git-ignored.

## Phase 2 — RampNet: the ground-truth review (~125 panos)

```bash
python scripts/gt_gallery.py benchmark/<city>
# open the printed index.html, review, "Export verdicts",
# save over benchmark/<city>/verdicts.json
```

Per pano the reviewer judges **every detection** (correct / false positive / duplicate /
unsure) *and* **scans the whole pano for ramps the model missed**. Both halves are required —
the missed-ramp pass is the only thing that makes recall measurable, and the `no_missed`
attestation on a clean pano is what lets that pano into the recall denominator at all.

Three things that are easy to get wrong:

- **Review at model resolution.** The gallery renders at 4096×2048 with pan/zoom, never the
  native image, on purpose. Mapillary panos are often 11000×5500; showing the reviewer more
  than the model saw inflates the miss count and deflates recall. Richmond and bend had to be
  **re-reviewed** for this reason (recall corrected 0.895 → 0.765 and 0.831 → 0.758).
- **Write the review notes.** The gallery's *Review notes* panel and the per-pano note box feed
  `review_notes`, which `score_validation.py` prints **above** the numbers. This is where "the
  rubric did not fit" goes. Budapest exists as a usable split only because its reviewer
  recorded low confidence in their own pass; without that the 0.873/0.503 would have been
  quoted as a clean number.
- **Use `unsure` freely.** It abstains from both metrics rather than forcing a guess.
  Abstention rates run 1.9% (bend) to 8.5% (budapest) of detections and are themselves a
  reported signal about imagery legibility.

Deliverable: `benchmark/<city>/verdicts.json` — committed, image-free, self-contained.

## Phase 3 — score and sanity-check

```bash
python scripts/score_validation.py benchmark/<city>
```

Reports precision/recall with Wilson intervals, both all-panos and on the **unbiased subset**
(random + empty, dropping the 5 hand-picked `top` panos). Quote the unbiased column for
between-city comparison.

Gates before going further:

- Do the `empty`-group panos behave? All 25 clean means the model's "nothing here" is right.
  The stratum is zero-detection by construction, so what can appear there is **missed marks** —
  real ramps the model was silent on, worth reading (annapolis's negative check: 4 of its 25
  held 10 of them).
- Does the `duplicate` count look sane? 0–3 is normal; budapest's 7 flagged a genuine rubric
  ambiguity (diagonal corner aprons) worth ~4 precision points.
- Does the recall gate include most panos? Panos with neither a `no_missed` attestation nor a
  missed mark are excluded from recall entirely.

## Phase 4 — regenerate the operating-point analysis

**This is the step that gets skipped.** A new split changes pooled numbers, per-tier numbers,
the GT-completeness correction, and both figures. See `docs/operating_point.md`.

```bash
# 1. GPU, once, on Hyak (~1.5 s/pano; the launcher skips splits already cached)
mkdir -p logs   # the launcher's #SBATCH output paths live under logs/; the job dies without it
CITIES=<city> sbatch -A <account> scripts/analysis/run_low_floor_extract.slurm

# 2. THE GATE — run before trusting anything downstream
python scripts/analysis/low_floor_sweep.py parity --cities <city>
```

**Parity must pass.** Peaks at ≥0.55 must reproduce the committed `records.jsonl`, measured in
match radii. Expect **100% bit-exact for a Mapillary split**. A GSV split will not be exact —
the GSV production path builds a 4096×2048 intermediate, so production saw a different resample
than the native-res bundle — but should still land inside 0.5 R (bend's max is 0.439 R). The
gate itself is slightly looser than that: **≥95% of detections within 0.5 R and a count delta
≤5%** (`PARITY_MIN_MATCHED` / `PARITY_MAX_COUNT_DELTA` in `low_floor_sweep.py`). Anything worse
means preprocessing diverged and every number downstream inherits it.

```bash
# 3. CPU — the analyses
python scripts/analysis/low_floor_sweep.py sweep      # P/R/F1 + density, per split / pooled / per tier
python scripts/analysis/low_floor_sweep.py hist       # calibration (feeds auto-labeler#27 stage 4)
python scripts/analysis/low_floor_sweep.py gtbias     # confirm the anchoring pattern holds
python scripts/analysis/low_floor_sweep.py floor      # storage-floor cost + recall ceiling
python scripts/analysis/low_floor_sweep.py distance   # where the recall gain lands
```

### The #55 GT-completeness correction (do not skip)

The new split's sub-0.55 precision is a **lower bound by construction** — its GT was assembled
from detections at or above 0.55, so a real ramp nobody marked scores as a false positive.
`gtbias` will show it: below the review floor, every true positive comes from a missed mark and
none from a reviewed detection. Correct it:

```bash
python scripts/analysis/operating_point_curve.py gallery --city <city> --op-threshold 0.25
# review index.html, tag A / B / unsure, Download <city>_tags.json, then:
cp <city>_tags.json benchmark/<city>/incremental_fp_tags.json
python scripts/analysis/low_floor_sweep.py corrected --op-threshold 0.30
python scripts/analysis/low_floor_sweep.py tagcheck --cities <city>
```

Use **op-threshold 0.25** so the band is `[0.25, 0.55)`, identical to every existing split —
otherwise the A-rates are not comparable. Expect 23–30 items for a US city (budapest had 89).
A-rates so far span 13%–30% and **do not order by imagery quality**, so do not extrapolate one
from a neighbouring city; measure it.

Two flags to not get wrong: `corrected`'s `--op-threshold` defaults to **0.35, not the
recommended 0.30**, so the flag above is load-bearing. And each `corrected` run reports a single
operating point — the multi-point corrected tables in `docs/operating_point.md` take one run
per row, not one run total.

`tagcheck` exists because tag ids are keyed to peak *coordinates*: re-extracting can move a
marginal peak and orphan a tag silently, shrinking the correction with no error.

### Figures

```bash
python scripts/analysis/plot_operating_point.py    # docs/figures/operating_point_pr.png
python scripts/analysis/plot_storage_floor.py      # docs/figures/storage_floor_ceiling.png
```

Both read the committed caches and need no GPU. **Look at the output**, don't just regenerate
it — an eighth series changes the legend layout and can collide with the curves.

## Phase 5 — code that must be told the split exists

Grep-able touchpoints. None of these fail loudly if missed; they fail by quietly omitting the
city or mislabelling it.

| File | What to change |
|---|---|
| `scripts/analysis/low_floor_sweep.py` → `US_SPLITS` / `CITY_SPLITS` / `ALL_SPLITS` | add the split; `US_SPLITS` membership is what puts it in the pooled recommendation |
| `low_floor_sweep.py` → `HELD_OUT` | **required** if it is not pooled — every held-out split must carry a stated reason, and a test enforces this |
| `low_floor_sweep.py` → `tier_of()` | add a branch if the rig is new, or the split lands in `unknown` |
| `low_floor_sweep.py` → `SPLIT_IMAGERY_FALLBACK` | only if `records.jsonl` lacks camera provenance |
| `low_floor_sweep.py` → `TTA_RECORD_SPLITS` | only if the committed detections used TTA |
| `scripts/analysis/plot_operating_point.py` → `SERIES` | add a colour **from the validated palette, in slot order** — never a made-up hue. Past 8 slots, fold to "Other" or facet |
| `scripts/build_benchmark_dataset.py` | ⚠️ still hardcoded to bend + richmond — the HF dataset lags the repo (issue #21) |

## Phase 6 — documentation (where a split actually becomes real)

| Document | What to update |
|---|---|
| `benchmark/README.md` | both split tables (all-panos and unbiased), and a **prose section** for the city: what it is for, what its imagery is, what fought the rubric, negative-check result |
| `docs/model_comparison.md` | the coverage matrix at the top — including the `#55` column with the new A-rate — and the anchoring caveat's list of measured cities |
| `docs/operating_point.md` | per-split table, per-tier table, pooled rows if it is pooled, the corrected-results tables, the storage-floor table |
| `docs/adding_a_benchmark_city.md` | this file, if the protocol changed |

Per `CLAUDE.md`: **an omission is indistinguishable from a withheld result.** If a step was not
run, say so explicitly and say why, rather than leaving a blank.

## Rough cost

| Phase | Cost |
|---|---|
| 1 — detect + export | hours of compute, mostly unattended; native-res fetch is the slow part |
| 2 — **GT review** | **the dominant human cost**, ~125 panos judged detection-by-detection plus a full-pano miss scan |
| 3 — score | seconds |
| 4 — extraction | ~3 min GPU for 125 panos; everything else is CPU-seconds |
| 4 — **#55 tagging** | second human pass, 23–30 items for a US city |
| 5–6 — code + docs | the part that gets rushed, and the reason this file exists |

## Traps

- **A new split changes pooled numbers everywhere.** Every pooled figure in
  `docs/operating_point.md` and the recommendation itself are computed over `US_SPLITS`. Adding
  a US city silently moves them; re-run and update the prose, don't just add a row.
- **Do not pool a split whose GT you do not trust.** Budapest is swept and reported but held out
  of every recommendation. That is a scientific judgement, recorded in `HELD_OUT`, not a bug.
- **`analysis_out/` is mostly git-ignored.** The detection caches (`op_cache/*.json`) and the
  derived tables (`op/*.csv`, `*.json`) are committed exceptions so results survive without a
  GPU. The gallery crops are not — but the **A/B tags are**, under
  `benchmark/<city>/incremental_fp_tags.json`. Losing those means redoing the human pass.
- **`panos/` is git-ignored and irreplaceable if the source decays.** Mapillary images can
  disappear. `index.csv` records sha256 per pano; archive the bundle somewhere durable (the HF
  dataset, issue #21) rather than relying on a laptop.
- **Never review at native resolution.** It is the one methodological error the benchmark has
  already made once, and it cost a re-review of two cities.

## Checklist

Copy into the PR description and tick. Anything not done gets a line saying so and why.

**Phase 1 — bundle (auto-labeler)**
- [ ] City detected (`main.py --source <mapillary|gsv>`)
- [ ] `export_benchmark.py --bundle` exited **zero**; `index.csv` written, `decayed.txt` reviewed
- [ ] `benchmark/<city>/records.jsonl` committed; `panos/` present locally and archived durably
- [ ] Strata are 5 `top` / 95 `random` / 25 `empty` (or the deviation is justified)

**Phase 2–3 — ground truth**
- [ ] Reviewed with `gt_gallery.py` **at model resolution**
- [ ] `review_notes` written (reviewer confidence + what fought the rubric)
- [ ] `verdicts.json` committed
- [ ] `score_validation.py` run; unbiased column recorded
- [ ] Camera provenance present in `records.jsonl` (`camera_make` / `camera_model`)

**Phase 4 — operating point**
- [ ] Low-floor extraction run (`run_low_floor_extract.slurm`)
- [ ] **`low_floor_sweep.py parity` PASSES** — Mapillary bit-exact, GSV within 0.5 R
- [ ] `sweep` / `hist` / `gtbias` / `floor` / `distance` re-run
- [ ] #55 gallery tagged at **op-threshold 0.25**; `incremental_fp_tags.json` committed
- [ ] `corrected` re-run; `tagcheck` passes 100%
- [ ] `op_cache/<city>.json` and the refreshed `op/*.csv` committed
- [ ] Both figures regenerated **and visually inspected**

**Phase 5 — code**
- [ ] Added to `US_SPLITS` / `CITY_SPLITS` / `ALL_SPLITS`
- [ ] `HELD_OUT` reason recorded if not pooled
- [ ] `tier_of()` recognises the rig (not `unknown`)
- [ ] `SERIES` colour added from the validated palette in slot order
- [ ] `pytest -q` green

**Phase 6 — docs**
- [ ] `benchmark/README.md`: both tables + a prose section for the city
- [ ] `docs/model_comparison.md`: coverage matrix incl. the #55 A-rate, and the anchoring caveat
- [ ] `docs/operating_point.md`: per-split, per-tier, pooled, corrected and storage-floor tables
- [ ] **Pooled numbers and the recommendation re-checked** — a new US city moves them
- [ ] Anything deliberately not run is stated explicitly, with the reason
