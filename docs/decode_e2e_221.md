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
| `--decode argmax` vs the committed `stage_two/evaluation_results_new/` | on the raw HF image bytes: P 0.9474, 7 more FPs at 0.55, a gap already on record. **On quality-95 re-encodes (`download_dataset.py`'s path) it reproduces the committed 0.55 numbers exactly**: 3,603 predictions, P 0.94921, R 0.87267. See [below](#the-committed-evaluation_results_new-files) |
| gaussian vs argmax detection metrics | **one more true positive** at 0.30 and at 0.55 (TTA and single pass). AP +0.0006 to +0.0016 |
| gaussian position error, single pass | **5.080 → 4.353 px** on 3,517 pairs. This reproduces #226 to the last digit |
| gaussian position error, flip TTA | **5.066 → 4.366 px** (-0.700 [-0.755, -0.645]) on 3,571 pairs. The shipped branch-select rule beats either single branch in a direct paired test, and ties the branch mean |
| `StaleCoarseCache` guard | fires on all 3 stale pairings tested. It stays quiet below `COARSE_ATOL` and on branch order, which does not matter. `--fresh` on an argmax run clears `coarse/` |
| HF export round-trip, no upload | `detect()` from a local export equals evaluate.py: argmax exact, gaussian within 1.5e-9 (normalized) |
| `demo.py --decode gaussian`, run headless | same peaks as evaluate.py's TTA path. Positions within 1.1e-6 px on 8 panos (39 marks) |
| bug found | `export_hf_model.py --from-hub-revision` could not load the current Hub weights. **Fixed** here |

Weights: `projectsidewalk/rampnet-model` at commit `606a11956743f7eb328d9207769034752f6191f4`
(`model.safetensors` sha256 `f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5`, the
same file #226 used). evaluate.py takes a `.pth`, so the safetensors keys were written to one
with the `model.` prefix stripped. On makelab2 that file's `checkpoint_fingerprint` is
`f7f255c586ba`; on the Windows desktop the same tensors fingerprint as `1c67c4b24091`, because
`torch.save` bytes differ between torch builds. The cache key therefore names the file, not the
weights. Split: **manual_gold only** (1,000 panos, 3,919 GT points). Image source for every run
except the q95 one: `benchmark/manual_gold/panos/` on makelab2. These are the raw JPEG bytes
of the HF test split's parquet, written by `scripts/fetch_manual_gold.py`, and they are the
files #226 hash-checked against `imagery_manifest.json`. The q95 run read re-encodes of those
files (see [below](#the-committed-evaluation_results_new-files)). evaluate.py supports only `--dataset manual` and `--dataset test`;
see [Not run](#not-run) for `test`.

## 1. evaluate.py on the GPU

Two fresh `--decode gaussian` passes, one with `--tta` and one with `--no-tta`. Each wrote the
heatmap cache and the coarse cache together. Every other run read those caches on the CPU:
argmax and gaussian at thresholds 0.0, 0.30 and 0.55, `main`'s evaluate.py
(origin/main a45bb91) on the same caches, and the committed-era evaluate.py (a9ed8a5, the commit
that wrote `evaluation_results_new/`). Results: `analysis_out/decode_e2e_221/metrics/`
(`<run>__metrics_*.json`). Those JSONs are evaluate.py's output copied verbatim. They keep
evaluate.py's unrounded floats and absolute makelab2 paths, which is an exception to this repo's
rounded-JSON rule, because they are a tool's raw output. Every CSV's sha256 is in
`csv_sha256.txt`; the CSVs themselves stay on makelab2 (71 MB).

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

**This gap is not new.** `docs/model_comparison.md` (lines 1889-1891, recorded 2026-07-25)
already has RampNet at P 0.947 / R 0.873 against the published 0.949 / 0.873, on the same
`benchmark/manual_gold/panos` raw bytes. The "drift vs published: env, JPEG re-encode" row of
`docs/eval_protocol_verification.html` names JPEG re-encoding as a candidate. `model_comparison.md`
(line 1814) also records that re-encoding alone once moved P +2.2 / R -1.8 on this split.

| run (TTA, argmax) | image source | preds @0.55 | P @0.55 | R @0.55 | preds, full sweep | R, full sweep | AP, full sweep |
|---|---|---:|---:|---:|---:|---:|---:|
| committed `evaluation_results_new/` (a9ed8a5, `checkpoints/epoch_1_step_9378.pth`) | `../dataset/test/*.jpg` from `download_dataset.py`, i.e. PIL re-encode at quality 95 (inferred from evaluate.py's default `--data-root`) | 3,603 | 0.94921 | 0.87267 | 114,473 | 0.94080 | 0.92051 |
| this PR, main and #229 (identical) | raw HF bytes, `benchmark/manual_gold/panos` | 3,610 | 0.94737 | 0.87267 | 113,876 | 0.94106 | 0.92077 |
| a9ed8a5's evaluate.py on this PR's cached heatmaps | raw HF bytes | 3,610 | 0.94737 | 0.87267 | 113,876 | 0.94106 | 0.92076 |
| **this PR, `decode_e2e_221_q95.sh`** | **raw HF bytes → PIL decode → JPEG quality 95 (Pillow 11.3.0), as `download_dataset.py:save_example` does** | **3,603** | **0.94921** | **0.87267** | 113,816 | **0.94080** | 0.92052 |

- **Raw bytes vs committed.** At 0.55 there are 7 more predictions and the same 3,420 TPs. Over
  the full sweep there are 597 fewer peaks and 1 more TP. The committed-era code gives the same
  numbers on these heatmaps, so the cause is upstream of peak extraction.
- **Hardware is close to ruled out by this PR's own data** (an argument from the #233 review).
  A40 and RTX 3070 heatmaps differ by at most 3.5e-5 (`roundtrip.json`). About 370 peaks lie in
  [0.30, 0.55), so about 0.1 peaks are expected within 3.5e-5 of 0.55. That is far from 7
  crossings, all in one direction.
- **Weights** were a third candidate. The committed run used the training `.pth` (b0c3ff7a10fc),
  which is not on makelab2. This run used the Hub safetensors. The two Hub revisions (1078bcd, 606a119)
  are tensor-identical to each other. They were not compared with the training `.pth`.
- **Settled by the q95 run: the gap is the JPEG re-encode.** Re-encoding the raw bytes at quality
  95 and running the same evaluate.py with the same weights reproduces the committed 0.55 row
  exactly (3,603 / P 0.94921 / R 0.87267). It also reproduces the committed full-sweep recall
  (0.94080), and AP to 1e-5 (0.920524 vs 0.920511). With those numbers matching, a weights
  difference is implausible. The README's headline P 0.949 / R 0.873 therefore depends on
  evaluating `download_dataset.py`'s re-encoded images, not the raw HF bytes. On raw bytes it is
  P 0.947 / R 0.873.
- **Not byte-identical, and why that is expected.** The CSVs still differ from the committed
  ones (`q95_vs_committed.txt`). The full sweep has 657 fewer peaks (113,816 vs 114,473), all
  far below any operating point: the 0.55 counts match. AP differs by 1.3e-5. The remaining
  differences in low-score peaks and AP are consistent with float noise in near-zero heatmap
  values across GPU, driver and Pillow/libjpeg versions, plus the #132 seam matcher. The
  committed run's GPU, driver and Pillow version are unrecorded, so this residual cannot be
  closed further.
- Neither decode change nor any code path between a9ed8a5 and #229 moved these numbers.

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
  and 1,977 flipped of 3,981 peaks >= 0.30). The rule gives the same gain as single pass.
- **Direct paired comparison of the branch rules** (`shipped_minus_alternative` in
  `positions_tta.json`). Same 3,571 pairs, pano-cluster bootstrap, 2,000 reps, seed 221 per
  comparison. d = shipped minus alternative, mean px; negative means the shipped rule is closer:

  | alternative | d mean px [95% CI] |
  |---|---|
  | original branch only | **-0.074 [-0.129, -0.024]** |
  | flipped branch only | **-0.058 [-0.097, -0.017]** |
  | mean of the two branches | -0.006 [-0.038, +0.026] |

  The shipped rule beats either single branch, and both CIs exclude 0. It is indistinguishable
  from the branch mean, which would cost a second decode per peak. No reason to change the rule.
- TTA does not make positions better than single pass (4.366 vs 4.353 px under gaussian). Its
  value is recall (54 more pairs), as #78 found.
- Re-matching each decode afresh at 0.30 / 0.55 gives the +1 TP of section 1b in every setting.
  Against the shipped rule at 0.30 (3,572 TPs), the alternatives lose TPs: original branch 3,566
  (-6), flipped 3,568 (-4), mean 3,569 (-3). At 0.55 they are within -3/+1 of it (3,421).

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
python scripts/export_hf_model.py \
    --from-hub-revision 606a11956743f7eb328d9207769034752f6191f4 \
    --source-fingerprint b0c3ff7a10fc --output-dir <tmp>/hf_export      # no --push
```

- The exported `model.safetensors` is **byte-identical** to the Hub's (sha256 `f2119e3b…`).
  `rampnet_model.py`, `configuration_rampnet.py` and `config.json` are also byte-identical to
  the Hub's. Only `modeling_rampnet.py` differs, which is expected: it adds `detect()`. The new
  `rampnet_subcell.py` is added too.
- The exporter's own verification passed. (Run here with `HF_HUB_OFFLINE=1` against a warm HF
  cache; drop it on a machine that has never downloaded the model.) The export was re-run
  after this PR's docstring edits to `rampnet/subcell.py` and `modeling_rampnet.py`, so the
  results below are for the final code.
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

## 5. demo.py `--decode gaussian`, run headless

The #233 review pointed out that demo.py does not share evaluate.py's path. It builds its own
coarse stack in-process (`coarse_from_heatmap` on the raw original and the flipped-oriented
outputs), runs no `coarse_mismatch` guard, crops and LANCZOS-resizes its input, draws peaks
>= 0.4, and keeps skimage's default `exclude_border=True`. `decode_e2e_221.py demo` imports
demo.py with a stub `gradio` module. It calls `predict_and_visualize()` on the first 8
manual_gold panos with `--decode gaussian` and `--decode argmax`, and records the peaks
demo.py passes to the drawing code. It then compares those with
`detect_peaks(h, 0.4, exclude_border=True, coarse=<stack>)` on evaluate.py's
`predict_heatmap(..., use_tta=True, return_coarse=True)` for the same image and weights (RTX
3070).

- Same peaks on every pano (39 marks under each decode).
- argmax: exact.
- gaussian: max difference 1.1e-6 heatmap px, from evaluate.py's float32 coarse cast.

`analysis_out/decode_e2e_221/demo_headless.json`. The crop and resize change nothing on full
2:1 GSV panos (PIL returns a copy when the size already matches), so this does not exercise
demo.py on cropped or odd-sized uploads. The Gradio UI itself was not launched.

## Recommendation on the two defaults (Jon's call)

- **`RampNetModel.detect()` default `gaussian`: keep.** It is verified end to end: the export
  path works, the output equals evaluate.py's, the error vs human boxes drops 0.73 px, and
  detection metrics do not get worse.
- **evaluate.py / demo.py default: flipping to `gaussian` is now safe, but it is not urgent.**
  The decode never loses a TP. It gains one at each operating point, and it moves AP by
  <= 0.0016, so no published comparison changes. The cost is a second cache (`coarse/`, 64 KB
  per pano with TTA) and new `_dgaussian` file names. The reason to keep `argmax` is that the
  committed `evaluation_results*/` and every published number used it. A middle course: flip
  `demo.py` (it draws positions, where the decode helps; its gaussian path is checked in
  section 5 on full-size GSV panos) and keep `evaluate.py` on `argmax` until the next full
  re-evaluation.

## Not run

- **`--dataset test`** (the auto-labelled Stage 1 test split). It is on makelab2 only as the HF
  dataset's parquet shards, and the `.venv` there has no `pyarrow`/`datasets` to unpack them into
  `dataset/test/`. Its GT is Stage 1 points, which the model was trained to reproduce, so it says
  less about position error than manual_gold does.
- **The other benchmark splits** (annapolis, paterson, richmond, sao_paulo, …). evaluate.py does
  not read them; they go through `scripts/model_comparison/`, which #229 did not touch. #226
  measured them single pass.
- **The demo's Gradio UI**, and demo.py on cropped or non-2:1 uploads (section 5).
- **Comparing the Hub weights with the training `.pth`** (b0c3ff7a10fc). That file is not on
  makelab2. The q95 run makes a weights difference implausible, but it was not checked directly.
- **The upload.** Nothing was pushed to the Hub.

## Reproduction

From a clean clone, in this order. `$REPO` is the clone. `$W` is any scratch directory with
about 10 GB free; on makelab2 it was `/homes/gws/jonf/nobackup/e2e221`. `$PY` is a Python with
`requirements-dev.txt` installed (makelab2: the RampNet `.venv`). Steps 1-6 need a GPU.

```bash
REPO=$(pwd); W=/path/to/scratch; PY=python; mkdir -p $W
# 1. imagery: the raw HF test-split bytes for the 1,000 gold panos, checked against
#    benchmark/manual_gold/imagery_manifest.json after the fetch
$PY scripts/fetch_manual_gold.py --images-only
mkdir -p $W/data && ln -sfn $REPO/benchmark/manual_gold/panos $W/data/test
# 2. released weights -> .pth (hub_state_dict_to_keypoint strips the wrapper's "model."
#    prefix, and only when every key has it)
$PY -c "import sys, torch; sys.path.insert(0, 'scripts')
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from export_hf_model import hub_state_dict_to_keypoint as k
p = hf_hub_download('projectsidewalk/rampnet-model', 'model.safetensors',
                    revision='606a11956743f7eb328d9207769034752f6191f4')
torch.save(k(load_file(p)), '$W/released_606a119.pth')"
$PY -c "from rampnet.loading import checkpoint_fingerprint as f; print(f('$W/released_606a119.pth'))"
#    -> put this value in FP= in the scripts below (makelab2: f7f255c586ba)
# 3. worktrees the CPU script compares against
git worktree add --detach $W/wt-main a45bb91
# 4. edit WT=$REPO, WM=$W/wt-main, W, PY (and FP) at the top of each script, then:
bash scripts/analysis/decode_e2e_221_gpu.sh     # GPU: two fresh --decode gaussian passes
bash scripts/analysis/decode_e2e_221_cpu.sh     # CPU: argmax/gaussian x 0.0/0.30/0.55, main's
                                                # evaluate.py, positions, stale-guard cases
# 5. committed-era evaluate.py on the same heatmaps (needs a9ed8a5: not in a shallow clone)
bash scripts/analysis/decode_e2e_221_a9ed8a5.sh
# 6. S1: quality-95 re-encode, the path evaluation_results_new/ used
bash scripts/analysis/decode_e2e_221_q95.sh
# 7. export round-trip and headless demo (any CUDA machine with transformers; no upload)
$PY scripts/export_hf_model.py --from-hub-revision 606a11956743f7eb328d9207769034752f6191f4 \
    --source-fingerprint b0c3ff7a10fc --output-dir $W/hf_export
$PY scripts/analysis/decode_e2e_221.py roundtrip --checkpoint $W/released_606a119.pth \
    --panos-dir benchmark/manual_gold/panos --export-dir $W/hf_export --reuse-export \
    --n 8 --tol 1e-8 [--compare-cache <copy of the GPU machine's $W/cache>] \
    --out analysis_out/decode_e2e_221/roundtrip.json
$PY scripts/analysis/decode_e2e_221.py demo --checkpoint $W/released_606a119.pth \
    --panos-dir benchmark/manual_gold/panos --n 8 \
    --out analysis_out/decode_e2e_221/demo_headless.json
```

The `positions_*.json` files are written by `decode_e2e_221_cpu.sh` (its `positions` step). They
were re-run once after the review to add the direct branch-rule comparison; every earlier number
in them is unchanged.

**Cost.** makelab2 A40.

- Main run: 4,603 s wall for the two GPU passes (TTA 3,000 s, single pass 1,603 s), **1.28 GPU-h
  upper bound**. The GPU was shared with another user's job the whole time, so these s/pano
  figures are not a speed measurement. The CPU phase took 1,134 s.
- q95 run: 1,426 s, **0.40 GPU-h**. No other process was on the GPU at launch or at the end.
- Total **1.67 GPU-h, $0**, in two `paid: false` rows in `analysis_out/usage_log.jsonl`
  (`decode-e2e-221:evaluate`, `decode-e2e-221:q95-reencode`).
- The desktop round-trip and demo runs took under a minute each and are not logged.
