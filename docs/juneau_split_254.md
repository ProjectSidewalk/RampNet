# Juneau (#254): the Annapolis rig on new streets, reviewed and pooled

**Status (2026-10-19): reviewed and pooled.** `benchmark/juneau/` holds 125 panoramas, their
detections and Jon's `verdicts.json` (single rater, confidence **medium**). The split is a pooled
US split, so it is in `low_floor_sweep.US_SPLITS` and in `rampnet/eval.py::PINNED_SPLITS` and
`POOLED_SPLITS`. Issue [#254](https://github.com/ProjectSidewalk/RampNet/issues/254); bundle in PR
[#250](https://github.com/ProjectSidewalk/RampNet/pull/250); archive pointer in PR
[#257](https://github.com/ProjectSidewalk/RampNet/pull/257). The labeler run is
[sidewalk-auto-labeler PR 164](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/164).
Results are in `benchmark/README.md` (Juneau section), `docs/model_comparison.md` and
`docs/operating_point.md`; this file is the run record.

## 1. What the split is for (Phase 0)

- **The rig is Annapolis's rig.** 120 of 125 panos are 8000x4000 Trimble MX7 captures by the
  Mapillary contributor `rking`, who shot all 125 Annapolis panos (Annapolis 2020 and 2023,
  Juneau 2024-05). The other five are GoPro Max from two other contributors, captured
  2025-08/09 (four at 4096x2048 from `tdbowthorpe_AKDOT`, one at 5760x2880 from `kinneyeng`).
  Juneau is therefore **not a new imagery provider**; what it can be is a replication of
  Annapolis's far-field distance finding on different streets, a different climate and a later
  capture date, at the highest latitude on the board (58.3 N).
- **Deployment target.** Juneau is a Project Sidewalk city (SidewalkWebpage#5737, downtown minus
  Douglas Island, about 4.8 km2), and the deployment runs on Mapillary, the same imagery this
  split samples. So this split measures the detector on the deployment's own imagery.
- **Imagery tier:** survey-grade for the 120 Trimble panos (`tier_of` returns `survey`), and
  `action-modern` for the 5 GoPro Max panos.
- **Pooled.** Decided by Jon on 2026-10-19 after the review: a US deployment city with
  verdict-grade GT, so it joins the pooled recommendation. It is a different city from
  Annapolis, not a second imagery arm of it, so `CITY_OF` does not pair them.

## 2. The bundle

| | |
|---|---|
| source run | `runs/juneau/results.jsonl` in the labeler: 6,098 Mapillary panos, 4,999 after 5 m thinning; detection code `c30fbff`; model `606a1195`; klone job 41691449 |
| sample | 5 top-detection / 95 random / 25 empty, seed 0, min confidence 0.55, 30 m spacing (`sample.json`) |
| detections | 217 over the 100 non-empty panos, min confidence 0.5501 |
| imagery | 125 native panos, sha256 pinned in `imagery_manifest.json` (digest `f37ab52d9d1165cf`); `panos/`, `index.csv` and `gallery/` git-ignored as for every split |
| archive | `makelab2:/projects/makeabilitylab/sidewalk-auto-labeler/runs/juneau/` (`results.jsonl`) and its `benchmark_bundle_254/panos/` (the 125 sampled panos); `benchmark_bundle/` beside it is the superseded pre-resample export |
| provenance | `bundle_provenance.json`: run, export, resample and staging, with the hashes |

## 3. The spacing bug and the re-sample

The first export (2026-10-09, RampNet `d6fee0f`) declared 30 m spacing but had **7 pano pairs
under 30 m, the closest 15.8 m** (two panos of one Mapillary sequence). The Opus 5.5 review of
#250 traced it to the labeler's `geo.LatLngSpacingIndex._key`, which sized the longitude grid
cell with each point's own cos(lat): at 58.3 N, 134.4 W the longitude index is about 264,000
cells from the meridian, and a 30 m change in latitude moved it by about two cells, past the
3x3 neighbourhood the index searches, so those pairs were never compared. The error grows with
|longitude| x tan(latitude), which is why Bayonne (1.5 W) is clean and Juneau is the worst case
on the board. The same bug left `laurens_mapillary` with 6 pairs under 30 m (min 26.2 m); that
split has verdicts and stands, with the cause now noted in `docs/laurens_paired_151.md`.

Fixed in [sidewalk-auto-labeler PR 165](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/165)
(one longitude cell size per index, from a reference latitude, with a 2% margin; a brute-force
test at Juneau's latitude). Re-exported 2026-10-11 from a worktree of that fix with the same
`results.jsonl`, seed and strata: **119 of 125 panos unchanged, 6 dropped, 6 added, 0 pairs
under 30 m, closest 30.7 m.** The ids are in `bundle_provenance.json`. No verdicts existed, so
nothing was discarded.

## 4. The review (2026-10-19)

On the Windows desktop, from the panos copied from the makelab2 archive and checked with
`imagery_manifest.py --verify` (digest `f37ab52d9d1165cf`, OK):

```bash
python scripts/gt_gallery.py benchmark/juneau       # review, Export, save as benchmark/juneau/verdicts.json
python scripts/score_validation.py benchmark/juneau
```

- **No `nadir_band.json`.** The 120 Trimble panos are pure black below y ≈ 0.825 on every pano
  (the vehicle mask: 117 at 0.826, 3 at 0.825, measured on the gallery's 4096x2048 renders),
  and the 5 GoPro Max panos show the vehicle's hood with no mask. A ramp cannot be hidden in
  pure black, so unlike Bayonne's logo band there is nothing for the gallery to hatch.
- **Verdicts are committed as exported**, then `review_notes` were added with every verdict
  unchanged (the script that added them refuses unless a re-dump reproduces the export byte for
  byte). Confidence **medium**: many distant ramps had to be marked unsure (50 of 143 missed
  marks). One pano note asks whether a diagonal corner ramp takes one label or two; Jon ruled
  after the review that it takes two only when it is large enough to serve both crossing
  directions (`benchmark/RUBRICS.md` §1). The verdicts were not re-checked against the ruling.
- **Score:** unbiased subset (120 panos) P **0.978** [0.944, 0.991], R **0.664** [0.605, 0.719];
  all 125 panos P 0.967 [0.933, 0.984], R 0.687 [0.632, 0.737].

## 5. The GPU work: makelab2 first, then klone

The GPU half (`scripts/analysis/juneau_254_gpu.sh`) differs from Bayonne's because it ran
**after** the review: `extract --unreviewed` refuses a reviewed bundle, so the extraction wrote
the reviewed cache straight to `analysis_out/op_cache/juneau.json` and there is no `attach-gt`.

**makelab2, abandoned.** It started on makelab2 (A40) on 2026-10-19 at 06:36 PDT. The host was
wedged on a stuck NFS mount (load 70-90, almost all of it threads in D state) and the run was
stopped during the OWLv2 + Grounding DINO step. Its outputs are not used. The compute it spent
is in `analysis_out/usage_log.jsonl`: the three YOLO legs' rows, recovered after the host's
reboot, and back-filled repro (107 s) and extract (138 s) rows from the run log's step stamps,
all marked abandoned in their `note`. The killed OWLv2 leg has no end stamp and is not recorded.
Log: `analysis_out/juneau_254/logs/makelab2_abandoned_gpu_run.log`.

**klone, the run of record.** Tree `fa8426d` (`git archive`), workdir
`/gscratch/scrubbed/jfroehli/wt-juneau254`, partition `ckpt-g2` (free) because
`gpu-l40s-makelab` was at its `AssocGrpCpuLimit`. Jobs 42034543-545 landed on H200s; the
Molmo and Qwen re-runs (42034817, 42035326-327) on L40s. The scrubbed purge had deleted
the old conda envs' interpreters and every cached HF weight, so the envs were rebuilt with uv:

| env | python | torch | transformers | used for |
|---|---|---|---|---|
| `rampnet-eval` | system 3.12.15 | 2.14.1+cu126 | 5.15.1 | repro, extract, parity, YOLO, OWLv2, Grounding DINO |
| `rampnet-eval2` | uv-managed 3.12.13 | 2.14.1+cu126 | 5.15.1 | Qwen3-VL-8B, Qwen3-VL-32B |
| `molmo3` | uv-managed 3.12.13 | 2.14.1+cu126 | 4.57.1 | Molmo2-8B |

The first Molmo and Qwen3-VL-8B attempts ran on klone's system Python, which has no `Python.h`,
so Triton could not compile its CUDA launcher and every pano failed (job 42034545 and the qwen
step of 42034543). Both were re-run on the uv-managed interpreter, which has the headers. The
failed attempts' ledger rows are kept: the GPU time was spent.

| job | steps | result |
|---|---|---|
| 42034543 | repro, extract, parity, OWLv2 + Grounding DINO (Qwen3-VL-8B failed) | repro: 374 of 374 `laurens_mapillary` peaks in the identical cell, max score difference 7e-05; parity: 217 of 217 records reproduced exactly, no seam extras (`checks.json`) |
| 42034544 | YOLO y11l_pano, y26_pano, y11x_pano_h200 | checkpoints are `rampnet_yolo_baseline_51/<arm>/weights/best.pt`, sha1 equal to the published signatures |
| 42034817 | Molmo2-8B | 125/125 |
| 42035326 | Qwen3-VL-8B | 125/125 |
| 42035327 | Qwen3-VL-32B, 2 GPUs (`run_qwen.slurm`) | 125/125 |

The replication control ran on an H200 and reproduced the committed (A40/L40S-built) cache
exactly, so the hardware change moves nothing. GPU-hours: `slurm_usage.py --from-file` over the
eleven jobs' `sacct` dump (`docs/data/compute/sacct_klone_2026-10-19_juneau254.txt`) into
`analysis_out/compute_log.jsonl`: 2.75 GPU-h at $0.00, per job in `docs/compute_cost.md`.

**Paid legs, desktop.** Authorized by Jon (estimate ~$5.26 from Bayonne's measured rates):
Gemini 3.6 Flash $1.92, Gemini 3.1 Pro $2.14, Gemini 3.7 Flash $1.54, **$5.60 total**
(list-price estimates; the billing console is authoritative). `vertex_usage.py --reconcile
--days 1`: every billed model has a ledger row within tolerance.

## 6. Commands, in order

```bash
# imagery and integrity
python scripts/analysis/imagery_manifest.py --verify --cities juneau --panos-root <checkout>
python scripts/analysis/juneau_254.py verify

# GPU (klone): build the envs on a login node, then one submission per env
# (exact commands, env per step and the YOLO_DIR links: the header of the .slurm file)
bash scripts/analysis/juneau_254_klone_env.sh          # versions: analysis_out/juneau_254/envs/
sbatch -A ckpt-makelab -p ckpt-g2 scripts/analysis/juneau_254_klone.slurm   # STEPS / PY / MOLMO_PY / YOLO_DIR
BUNDLE=benchmark/juneau QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct \
    sbatch -A ckpt-makelab -p ckpt-g2 --gpus-per-node=2 scripts/model_comparison/run_qwen.slurm
python scripts/analysis/juneau_254.py checks --write

# paid (desktop)
python scripts/model_comparison/compare.py benchmark/juneau --models gemini:gemini-3.6-flash,gemini:gemini-3.1-pro-preview,gemini:gemini-3.7-flash --cache-dir D:/Git/RampNet/.model_cache
python scripts/analysis/vertex_usage.py --reconcile --days 1

# publish detections (YOLO on klone, where the checkpoints are), then --verify each
python scripts/analysis/export_model_cache.py --splits juneau --tiling none --yolo-imgsz 1280 --models yolo:yolo_ckpts/y11l_pano.pt,yolo:yolo_ckpts/y26_pano.pt,yolo:yolo_ckpts/y11x_pano_h200.pt --write
python scripts/analysis/export_model_cache.py --splits juneau --models owlv2,gdino,qwen:Qwen/Qwen3-VL-8B-Instruct,qwen:Qwen/Qwen3-VL-32B-Instruct,molmo:allenai/Molmo2-8B --write
python scripts/analysis/export_model_cache.py --cache-dir D:/Git/RampNet/.model_cache --splits juneau --models gemini:gemini-3.6-flash,gemini:gemini-3.1-pro-preview,gemini:gemini-3.7-flash --write

# CPU analyses
C=richmond,bend,clovis,morgantown,annapolis,paterson,gainesville,laurens_mapillary,juneau,budapest_district5,sao_paulo,bayonne,manual_gold
for s in sweep hist gtbias floor distance; do python scripts/analysis/low_floor_sweep.py $s --cities $C; done
python scripts/analysis/operating_point_curve.py gallery --city juneau --op-threshold 0.25 --upper 0.55 --panos benchmark/juneau/panos --out analysis_out/op/juneau_incremental_fp
python scripts/analysis/low_floor_sweep.py tagcheck --cities juneau
for t in 0.25 0.30 0.35; do python scripts/analysis/low_floor_sweep.py corrected --op-threshold $t; done
python -m rampnet.eval pins --write
python scripts/analysis/train_overlap_check.py --benchmark benchmark --out benchmark/train_overlap.json
python scripts/analysis/scoreboard.py
```

## 7. The #55 tags

21 incremental FPs in [0.25, 0.55), 5 flagged "dup?" (within 2 R of an already-detected ramp).
The gallery has no duplicate key. Jon stated that of the five "dup?" items only
`2549721381871423_0.32422_0.69336` is a separate ramp (A); the other four are second hits on a
counted ramp, which the rubric scores B (`RUBRICS.md` §2). So two exported tags were changed:
`1529294995153246_0.69824_0.52344` A -> B and `2163717303998475_0.59668_0.55469` U -> B. The
file as downloaded is kept in `analysis_out/juneau_254/incremental_fp_tags_as_exported.json`.
Final tags A 4 / B 15 / U 2; `tagcheck` 21/21 resolve. `corrected`'s note that one A tag sits
within 2 R of a detected ramp refers to `2549721381871423`, the one Jon judged separate.

## 8. Not done, and why

- **claude-opus-5 (low)**: not run. Jon is having the plan API-credits route set up separately
  (`serving_path` is not in the Claude cache key, so a first-party run is the same leg). Until it
  runs, that leg covers 8 of 9 pooled cities and drops out of the complete pooled rows. The two
  newest Claude ids did run here, off-roster, on 2026-10-20 (PR #261, `claude20` org):
  `claude-opus-5-5` F1 0.727 ($8.40) and `claude-fable-5-1` 0.673 ($20.66) at effort low, both
  below RampNet's 0.803 with paired intervals that exclude zero
  ([`claude_newest_ids.md`](claude_newest_ids.md)).
- **Null-recall** (`scripts/analysis/null_recall.py`) and the **#35 cascade read**: not run,
  as for Bayonne; out of scope for this pass.
- **HF publish of the panos**: Jon's call. Until then the 125 panos are an input another person
  cannot obtain from the repo; they are in the makelab2 archive and pinned by
  `imagery_manifest.json`.
- **A second rater**: none. The review is single-rater at medium confidence.
- **`docs/figures/operating_point_pr.png` and `docs/figures/storage_floor_ceiling.png`**: not
  regenerated. Both plotters read every split in `ALL_SPLITS` and take no split list, and
  `laurens_gsv` has no op_cache, so they exit on it (the same blocker Bayonne hit; the
  storage-floor figure was last regenerated at sao_paulo). For `operating_point_pr.png`, juneau
  would also be a ninth pooled series past the palette's eight categorical slots, which needs a
  design decision first. The CSVs behind both are current.
- **The `dup?` tags of other splits**: `corrected` flags one A tag within 2 R of a detected ramp
  on each of clovis, morgantown, gainesville and budapest. If those raters also had no duplicate
  key, some may be duplicates scored A. Not re-checked here.
- **The labeler's detection commit**: sidewalk-auto-labeler PR 164 is unmerged, so `c30fbff` is
  reachable only from its branch until it merges.
