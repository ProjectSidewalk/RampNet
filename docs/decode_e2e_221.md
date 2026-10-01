# End-to-end check of the shipped sub-cell decode (#221)

PR #229 shipped `rampnet.subcell.detect_peaks`, `stage_two/evaluate.py --decode`, and
`RampNetModel.detect()` in the Hugging Face package. Its section 10 in
[`subcell_decode_221.md`](subcell_decode_221.md) lists three things that were never run:
`evaluate.py --decode gaussian` on a GPU, decoding under flip TTA, and an export from real
weights. This document records those runs. Run 2026-10-01 on branch
`analysis/decode-e2e-221`, which is stacked on #229 (`feat/subcell-decode-ship-221` at b119790).

## Summary

| check | result |
|---|---|
| `--decode argmax` vs `main`'s evaluate.py, same cache | **byte-identical** CSVs at thresholds 0.0, 0.30 and 0.55, TTA and single pass. The metrics JSON differs only by the new `decode` key |
| `--decode argmax` vs the committed `stage_two/evaluation_results_new/` | **not identical**, and the gap comes from the heatmaps, not the code. See [below](#the-committed-evaluation_results_new-files) |
| gaussian vs argmax detection metrics | **one more true positive** at 0.30 and at 0.55 (TTA and single pass). AP +0.0006 to +0.0016 |
| gaussian position error, single pass | **5.080 → 4.353 px** on 3,517 pairs. This reproduces #226 to the last digit |
| gaussian position error, flip TTA | **5.066 → 4.366 px** (-0.700 [-0.755, -0.645]) on 3,571 pairs. The shipped branch-select rule works |
| `StaleCoarseCache` guard | fires on all 3 stale pairings tested. It stays quiet below `COARSE_ATOL` and on branch order, which does not matter. `--fresh` on an argmax run clears `coarse/` |
| HF export round-trip, no upload | `detect()` from a local export equals evaluate.py: argmax exact, gaussian within 1.5e-9 (normalized) |
| bug found | `export_hf_model.py --from-hub-revision` could not load the current Hub weights. **Fixed** here |

Weights: `projectsidewalk/rampnet-model` at commit `606a11956743f7eb328d9207769034752f6191f4`
(`model.safetensors` sha256 `f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5`, the
same file #226 used). evaluate.py takes a `.pth`, so the safetensors keys were written to one
with the `model.` prefix stripped. On makelab2 that file's `checkpoint_fingerprint` is
`f7f255c586ba`; on the Windows desktop the same tensors fingerprint as `1c67c4b24091`, because
`torch.save` bytes differ between torch builds. The cache key therefore names the file, not the
weights. Split: **manual_gold only** (1,000 panos, 3,919 GT points, read from
`benchmark/manual_gold/panos/` on makelab2. These are the files #226 hash-checked against
`imagery_manifest.json`). evaluate.py supports only `--dataset manual` and `--dataset test`;
see [Not run](#not-run) for `test`.

## 1. evaluate.py on the GPU

Two fresh `--decode gaussian` passes, one with `--tta` and one with `--no-tta`. Each wrote the
heatmap cache and the coarse cache together. Every other run read those caches on the CPU:
argmax and gaussian at thresholds 0.0, 0.30 and 0.55, `main`'s evaluate.py
(origin/main a45bb91) on the same caches, and the committed-era evaluate.py (a9ed8a5, the commit
that wrote `evaluation_results_new/`). Results: `analysis_out/decode_e2e_221/metrics/`
(`<run>__metrics_*.json`). Every CSV's sha256 is in `csv_sha256.txt`; the CSVs themselves stay on
makelab2 (71 MB).

### 1a. argmax is bit-identical to main

On every one of the 12 (TTA x threshold x CSV) pairs, `cmp` finds #229's `--decode argmax`
output identical to `main`'s. The metrics JSONs are equal once the new `decode: "argmax"` key is
removed. The gaussian run read back from the cache equals the fresh GPU run byte for byte (the
`pr_rc_vs_c` CSV at 0.0, both TTA settings).

### 1b. Gaussian vs argmax detection metrics, manual_gold

| TTA | threshold | decode | predictions | P | R | F1 | AP |
|---|---|---|---:|---:|---:|---:|---:|
| flip | 0.30 | argmax | 3,981 | 0.89701 | 0.91120 | 0.90405 | 0.89722 |
| flip | 0.30 | gaussian | 3,981 | 0.89726 | 0.91146 | 0.90430 | 0.89808 |
| flip | 0.55 | argmax | 3,610 | 0.94737 | 0.87267 | 0.90849 | 0.86153 |
| flip | 0.55 | gaussian | 3,610 | 0.94765 | 0.87293 | 0.90875 | 0.86239 |
| flip | 0.0 (full sweep) | argmax | 113,876 | | 0.94106 | | **0.92077** |
| flip | 0.0 (full sweep) | gaussian | 113,876 | | 0.94106 | | **0.92140** |
| none | 0.30 | argmax | 3,868 | 0.90926 | 0.89742 | 0.90330 | 0.88452 |
| none | 0.30 | gaussian | 3,868 | 0.90951 | 0.89768 | 0.90356 | 0.88592 |
| none | 0.55 | argmax | 3,507 | 0.95438 | 0.85404 | 0.90143 | 0.84390 |
| none | 0.55 | gaussian | 3,507 | 0.95466 | 0.85430 | 0.90170 | 0.84528 |
| none | 0.0 (full sweep) | argmax | 82,353 | | 0.93366 | | **0.91407** |
| none | 0.0 (full sweep) | gaussian | 82,353 | | 0.93417 | | **0.91567** |

The decode finds the same peaks with the same scores, so the prediction counts are identical.
At 0.30 and at 0.55 it gains exactly **one** true positive in each setting: a peak that was just
outside the 0.022 radius at its argmax pixel and is inside it once decoded. Over the full
single-pass sweep it gains 2. That is +0.00026 in P and R. The AP gains (+0.0006 TTA,
+0.0016 single pass at the full sweep) are larger than the TP counts suggest, because a gained
match also moves where TPs fall in the confidence order. Caveat: one split, no CI. A +1 TP
difference is not a resolvable effect at this n. "Detection metrics are unchanged" is the
right reading, as #226 said. No decode loses a TP.

### The committed `evaluation_results_new/` files

`--decode argmax` does **not** reproduce the committed files byte for byte. At 0.55 with TTA the
committed run has P 0.94921 / R 0.87267 / AP 0.86151 from 3,603 predictions. This run has
P 0.94737 / R 0.87267 / AP 0.86153 from 3,610: the same 3,420 TPs plus 7 more FPs. Over the full
sweep the AP is 0.92051 committed against 0.92077 here.

The difference is in the **heatmaps, not the code**. The committed-era evaluate.py (a9ed8a5, run
from a `git archive` of that commit) on *this run's* cached heatmaps gives 3,610 predictions,
P 0.94737 and AP 0.92076, the same as `main`. The only movement is AP at the 1e-5 level, from the
#132 seam matcher. So nothing between a9ed8a5 and #229 changed these numbers. The committed run
read `../dataset/test/*.jpg` (the HF dataset re-saved as JPEG quality 95 by `download_dataset.py`)
on an unrecorded GPU. This run read `benchmark/manual_gold/panos/`. Which of the two (image bytes
or hardware) accounts for the 7 FPs was **not determined**. The published P/R/AP figures in the
README erratum stay as they are. They are within 0.002 of this run, and no code path changed them.

## 2. Position error against the manual_gold box centres

`scripts/analysis/decode_e2e_221.py positions` reads evaluate.py's own caches and extracts peaks
through `evaluate.extract_peaks_from_heatmap`, the shipped path with its guard. It then repeats
#226's protocol, reusing that script's GT loader, residuals and bootstrap: peaks >= 0.30, pairs
matched once on argmax positions (greedy by confidence, radius 0.022, x wrapped), every decode
scored on the same pairs, and a pano-cluster bootstrap (2,000 reps, seed 221).
`analysis_out/decode_e2e_221/positions_{notta,tta}.json`.

| setting | pairs (panos) | argmax mean px | decode | mean px | d mean px [95% CI] | mean deg |
|---|---|---:|---|---:|---|---:|
| single pass | 3,517 (787) | 5.080 | gaussian | 4.353 | **-0.727 [-0.782, -0.673]** | 1.726 → 1.473 |
| flip TTA | 3,571 (790) | 5.066 | **gaussian (shipped: branch-select)** | 4.366 | **-0.700 [-0.755, -0.645]** | 1.722 → 1.478 |
| flip TTA | | | original branch only | 4.440 | -0.626 [-0.698, -0.552] | |
| flip TTA | | | flipped branch only | 4.425 | -0.641 [-0.710, -0.576] | |
| flip TTA | | | mean of the two branches | 4.372 | -0.693 [-0.758, -0.629] | |

- **Single pass reproduces #226 exactly**: 3,517 pairs on 787 panos, 5.080 → 4.353, and the
  same CI to three decimals. The extraction now goes through evaluate.py's float32 coarse cache
  rather than #226's in-memory head output, so this is a real check of the shipped path.
- **TTA decoding works.** The shipped rule decodes each peak from the branch with the higher
  upsampled value at that pixel. The two branches are chosen about equally often (2,004 original
  and 1,977 flipped of 3,981 peaks >= 0.30). The rule gives the same gain as single pass. It
  beats decoding from either branch alone by about 0.07 px, and it ties the branch mean
  (4.366 vs 4.372; the CIs overlap completely), which would cost a second decode per peak. No
  reason to change the rule.
- TTA does not make positions better than single pass (4.366 vs 4.353 px under gaussian). Its
  value is recall (54 more pairs), as #78 found.
- Re-matching each decode afresh at 0.30 / 0.55 gives the +1 TP of section 1b in every setting.
  None of the TTA alternatives changes a TP count by more than 5.

Caveats from #226 apply unchanged. A box centre is not the Stage 1 point the model trained on,
so only the paired change is interpretable. This is manual_gold GSV imagery only. The TTA rows
are new and have no replication.

## 3. The stale-cache guard and `--fresh`

A 3-pano subset with its own cache root (`--tta`, decode gaussian unless noted). The first
pano's cached coarse stack was edited between runs:
`analysis_out/decode_e2e_221/stale_guard_summary.txt` and `stale_guard_messages.txt`.

| case | expected | got |
|---|---|---|
| 1. fresh gaussian run | ok | ok |
| 2. +2e-3 added to the coarse map (2x `COARSE_ATOL`) | raise | `StaleCoarseCache` (mismatch 0.002) |
| 3. +5e-4 added (below `COARSE_ATOL`) | no raise | ok. Note: a stale cache this close would pass |
| 4. another pano's coarse stack | raise | `StaleCoarseCache` (0.843) |
| 5. single-pass stack (B=1) under the TTA key | raise | `StaleCoarseCache` (0.0888) |
| 6. branches swapped | no raise | ok. Max-combine and branch-select are both order-invariant, so this is harmless |
| 7. stale coarse present, then `--decode argmax --fresh` | `coarse/` cleared | 0 coarse files left (#229 review S1 fix confirmed) |
| 8. gaussian after 7 (heatmap cached, coarse missing) | coarse rebuilt, guard passes | ok, 3 coarse files |

## 4. Hugging Face export round-trip (nothing uploaded)

**Bug fixed: `export_hf_model.py --from-hub-revision` could not load the current Hub weights.**
It loaded `model.safetensors` straight into a bare `KeypointModel`. The Hub's `main` (606a119)
stores the HF wrapper's keys (`model.feature_extractor...`), which is the layout
`assemble_package` itself writes, so the load failed with every key missing. That is the
re-export #229 leaves for Jon to run. The new `hub_state_dict_to_keypoint()` strips the prefix
only when every key has it, so the load is still strict. Tests:
`tests/test_export_hub_keys.py` (CPU, no network). With the fix:

```bash
HF_HUB_OFFLINE=1 python scripts/export_hf_model.py \
    --from-hub-revision 606a11956743f7eb328d9207769034752f6191f4 \
    --source-fingerprint b0c3ff7a10fc --output-dir <tmp>/hf_export      # no --push
```

- The exported `model.safetensors` is **byte-identical** to the Hub's (sha256 `f2119e3b…`).
  `rampnet_model.py`, `configuration_rampnet.py` and `config.json` are also byte-identical to
  the Hub's. Only `modeling_rampnet.py` differs, which is expected: it adds `detect()`. The new
  `rampnet_subcell.py` is added too.
- The exporter's own verification passed.
- `decode_e2e_221.py roundtrip` loaded that directory with
  `AutoModel.from_pretrained(dir, trust_remote_code=True)` (transformers 5.14.1, torch
  2.6.0+cu126, RTX 3070). It ran `detect(pixel_values, threshold=0.30, decode=...)` on the first
  8 manual_gold panos (40 detections at >= 0.30) and compared the output with evaluate.py's single-pass
  extraction from a `KeypointModel` loaded from the same weights. **argmax: exact. gaussian:
  max difference 1.5e-9 normalized (1.1e-6 px).** The gaussian residual is expected:
  evaluate.py stores the coarse stack as float32 so that a cached decode equals a fresh one,
  and `detect()` decodes from the float64 recovery.
  `analysis_out/decode_e2e_221/roundtrip.json`.
- All four package code files are byte-identical to their sources: `rampnet_model.py` ==
  `rampnet/model.py`, `rampnet_subcell.py` == `rampnet/subcell.py`, and the two
  `scripts/hf_package/` files. `modeling_rampnet.py` is still in sync with `rampnet/model.py`.
- **Cross-machine.** The same 8 panos decoded from makelab2's A40 cache (`--no-tta`) and from
  the desktop's `detect()` give the same peak pixels. Gaussian positions agree within 4.2e-4 px,
  and scores and heatmaps within 3.5e-5.

## Recommendation on the two defaults (Jon's call)

- **`RampNetModel.detect()` default `gaussian`: keep.** It is verified end to end: the export
  path works, the output equals evaluate.py's, the error vs human boxes drops 0.73 px, and
  detection metrics do not get worse.
- **evaluate.py / demo.py default: flipping to `gaussian` is now safe, but it is not urgent.**
  The decode never loses a TP. It gains one at each operating point, and it moves AP by
  <= 0.0016, so no published comparison changes. The cost is a second cache (`coarse/`, 64 KB
  per pano with TTA) and new `_dgaussian` file names. The reason to keep `argmax` is that the
  committed `evaluation_results*/` and every published number used it. A middle course: flip
  `demo.py` (it draws positions, where the decode helps) and keep `evaluate.py` on `argmax`
  until the next full re-evaluation.

## Not run

- **`--dataset test`** (the auto-labelled Stage 1 test split). It is on makelab2 only as the HF
  dataset's parquet shards, and the `.venv` there has no `pyarrow`/`datasets` to unpack them into
  `dataset/test/`. Its GT is Stage 1 points, which the model was trained to reproduce, so it says
  less about position error than manual_gold does.
- **The other benchmark splits** (annapolis, paterson, richmond, sao_paulo, …). evaluate.py does
  not read them; they go through `scripts/model_comparison/`, which #229 did not touch. #226
  measured them single pass.
- **demo.py `--decode gaussian`** was not launched (gradio). It calls the same `detect_peaks`.
- **The upload.** Nothing was pushed to the Hub.

## Reproduction

On makelab2, from a checkout of this branch at `$WT`, with the RampNet `.venv`:

```bash
# released weights -> .pth (strip the HF wrapper's "model." prefix)
python -c "from safetensors.torch import load_file; import torch; \
  sd=load_file('<hf cache>/models--projectsidewalk--rampnet-model/snapshots/606a119.../model.safetensors'); \
  torch.save({k[6:]: v for k, v in sd.items()}, '$W/released_606a119.pth')"
mkdir -p $W/data && ln -s $REPO/benchmark/manual_gold/panos $W/data/test
bash scripts/analysis/decode_e2e_221_gpu.sh    # GPU: two fresh --decode gaussian passes
bash scripts/analysis/decode_e2e_221_cpu.sh    # CPU: all other runs, positions, stale-guard cases
```

The two scripts hold the exact evaluate.py command lines. Edit `WT`/`WM`/`W`/`PY` at their tops,
and `FP` to the fingerprint your `.pth` gets. Then, on a machine with transformers and the panos:

```bash
python scripts/analysis/decode_e2e_221.py roundtrip --checkpoint <released .pth> \
    --panos-dir benchmark/manual_gold/panos --export-dir <tmp>/hf_export --reuse-export \
    --n 8 --tol 1e-8 [--compare-cache <makelab2 cache copy>] \
    --out analysis_out/decode_e2e_221/roundtrip.json
```

The committed-era comparison ran `git archive a9ed8a5 stage_two/evaluate.py rampnet manual_labels`
with `--cache-dir` pointed at a directory whose `heatmaps/f7f255c586ba_tta` is a symlink to this
run's `heatmaps/f7f255c586ba_manual_tta` (that commit's key has no dataset id).

**Cost.** makelab2 A40, 4,603 s wall for the two GPU passes (TTA 3,000 s, single pass 1,603 s),
**1.28 GPU-h upper bound, $0**. The GPU was shared with another user's job the whole time, so the
s/pano figures are not a speed measurement. The CPU phase took 1,134 s. One `paid: false` row is
in `analysis_out/usage_log.jsonl` (`decode-e2e-221:evaluate`). The desktop round-trip runs took
about 30 s each and are not logged.
