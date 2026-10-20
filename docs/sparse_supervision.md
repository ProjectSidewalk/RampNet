# Sparse supervision for Stage 2: schema, masked loss, label builder (#263)

**Status: plumbing only.** Nothing has been trained with it, and the builder has not been run
on any city. This is Track 3 of [#263](https://github.com/ProjectSidewalk/RampNet/issues/263):
- the masked heatmap loss (`stage_two/train.py --loss-mask`);
- the builder that turns a sidewalk-auto-labeler run into sparse-labeled training panos
  (`scripts/build_sparse_labels.py`);
- the mixer that builds a sparse + GSV-replay training directory
  (`scripts/mix_training_dirs.py`);
- the GSV-only control arm that every later arm is read against
  (`stage_two/run_finetune_control.slurm`).

Everything is **off by default**. Without `--loss-mask`, the published training path is
unchanged, and `tests/test_train_augment_82.py`'s `GOLDEN_OFF` hash still passes. The
first review of PR #266 shaped most of what follows; each rule below says which finding it
answers.

## Why a mask

A self-labeled open-imagery pano knows where some ramps are (strong fused sites, mined misses)
and nothing about the rest of the image. Training it the dense way, with every unlabeled pixel
as a zero target, teaches "no ramp" wherever the labeler was silent. On the open rigs that is
exactly where RampNet 1.0 misses (the recall gap in #151 and #159). The mask trains such a pano
only near its verified positives. It trains a pano everywhere only when there is a reason to
believe the pano is empty, and that reason is still a proposal (below).

## The pano JSON

Each training pano is `<id>.jpg` + `<id>.json` in a split directory, as in the Stage 2
dataset. A sparse-labeled JSON is the dataset format plus a `supervision` block:

```json
{
 "pano_id": "1000248907732708",
 "curb_ramp_points_normalized": [[0.25, 0.6], [0.9, 0.61]],
 "pano_coord": [45.75, 4.85],
 "pano_azimuth": -37.716926,
 "pano_meta": {"source": "mapillary", "camera_make": "GoPro", "camera_model": "GoPro Max",
               "capture_date": "2024-05", "native_width": 5760, "native_height": 2880,
               "native_sha256": "…"},
 "supervision": {
  "mode": "sparse",
  "sources": ["strong_site", "mined"],
  "site_ids": ["lyon:1042", "lyon:mined:877"],
  "ignore_points": [[0.31, 0.62]],
  "run_sha": "…",
  "city": "lyon"
 }
}
```

| field | meaning |
|---|---|
| `curb_ramp_points_normalized` | The verified positives, as `[x, y]` fractions of width and height. `[]` for `verified_empty`. |
| `supervision.mode` | `"sparse"` or `"verified_empty"`. **Any other value is an error** (train.py raises). So is a `sparse` pano with no positives, and a `verified_empty` pano that lists positives or ignore points. |
| `supervision.sources` | One entry per point, parallel to the point list: `"strong_site"` or `"mined"`. An arm can drop one source without a rebuild. |
| `supervision.site_ids` | One entry per point, the site the point came from: `"<city>:<site_id>"` for a `sites.jsonl` site, `"<city>:mined:<site_id>"` for a mined row. `mine_city.py` runs its own fuse, so its ids are a different serial. A site id means nothing without its run. |
| `supervision.ignore_points` | The pano's other stored peaks, i.e. every detection in `results.jsonl` that is not a positive (#266 review M4). A peak within 2 heatmap px of a positive counts as that positive. Their discs are carved out of the positives' discs; see below. Optional in train.py (missing means `[]`). |
| `supervision.run_sha` | The sha256 of the run's `results.jsonl`, which identifies the labeler run. |
| `supervision.city` | The city slug. |
| `pano_coord`, `pano_azimuth` | Carried from the run: `[lat, lng]`, and the run's `camera_heading` wrapped to [-180, 180) as Stage 1 does (`rampnet.gsv.heading_to_azimuth`). For GSV that is the quantity Stage 1 stored; for other sources it is the labeler's heading convention. Nothing in training reads either field. |
| `pano_meta` | The native image's provenance, including its sha256, so the JSON pins the exact input image. |

A JSON with **no** `supervision` block is **dense**, which is every published Stage 2 pano.
Stage 1's `curb_ramp_coords` and `record_creation_time` are not written. The first has no
counterpart in a labeler run, and the second would change the output on every rebuild.

Positions are normalized, so resizing the native pano (4,096 to 16,384 px wide depending on
the rig) to the dataset's 4096x2048 changes no label.

## What `--loss-mask` does

`EquiHeatmapDataset.__getitem__` returns `(image, heatmap, weight)`, where `weight` has the
heatmap's shape, (1, 512, 1024), and is float32:

| pano | target | weight |
|---|---|---|
| dense (no block) | Gaussians at the points, as now | 1 everywhere |
| `sparse` | Gaussians at the points, as now | 1 inside a disc of radius ceil(3σ) = 30 heatmap px around each positive, minus the ignore points' share (below), 0 elsewhere |
| `verified_empty` | all zero | 1 everywhere |
| unreadable JSON, dummy sample | all zero | 0 everywhere (teaches nothing) |

**The weight follows the target exactly.**
- It is built from the **same** point list as the target, after the horizontal flip, so a
  flipped pano gets a flipped disc. The ignore points are mirrored with the positives.
- The centre is rounded and clamped as the target's is.
- The disc is clipped at the image edge with **no wrap across the 0/360 seam**, the same as
  the target.
- The disc is exactly the set of pixels where the target is at least exp(-4.5) = 0.011. Only
  the corners of the target's square window, which fall below that, lie outside it.

**Ignore points (#266 review M4).** Without them, the annulus of a positive's disc trains
"no ramp" over 30 px around every positive. That includes any second ramp at the same corner,
or a sub-threshold detection the pano made itself. So inside a positive's disc, a pixel
**strictly nearer an ignore point than every positive** gets weight 0. Positives win ties, and
a pixel nearer its own positive keeps weight 1, so an ignore point next to a positive never
erases that positive's peak. An ignore point outside every positive disc changes nothing,
because the weight there is already 0. With no ignore points, the weight is the plain union of
the discs. The rule is a nearest-point split rather than "zero every ignore disc", which would
blank the positive whenever a peak sits within 30 px of it. The literal reading of "positives
win where discs overlap" (the overlap goes wholly to the positive) would make ignore points a
no-op, because outside the positive discs the weight is already 0.

**The loss**, for train and validation alike, with `--mask-norm` choosing the denominator:

```python
def masked_mse(outputs, target, weight, norm='pixels'):
    total = ((outputs.float() - target) ** 2 * weight).sum()
    if norm == 'pixels':
        return total / weight.numel()
    if norm == 'weight':
        return total / weight.sum().clamp(min=1)
```

With `weight` all ones, both norms equal `nn.MSELoss()` (checked to rtol 1e-6). On the dense
GSV validation set the logged validation loss therefore means what it always has. With the
mask on, a validation sample whose weight is all zero (an unreadable JSON or image) is left
out of the mean instead of counted as loss 0 (#266 review L2). An all-masked batch is 0 under
either norm, never NaN.

**Without `--loss-mask`**, `criterion = nn.MSELoss()` is called exactly as before. A JSON that
**does** carry a `supervision` block is an error, so it cannot silently train as dense.
`--mask-norm` without `--loss-mask` is an error too.

**Resume.** `latest_checkpoint.pth` records `loss_mask` and `mask_norm` when the mask is on.
A resume under different flags is refused (#266 review L3), because it would mix two
objectives in one run and `best_val_loss` would compare different quantities. With the mask
off, the checkpoint carries exactly the keys it always has, and a checkpoint without the keys
counts as mask-off.

**Logging.** Each epoch, and a `--max-steps` run at its end, prints the share of masked pixels
on rank 0's shard and logs it as `Mask/masked_pixel_share`. A sparse-only pool reads about
0.99; a run whose sparse panos never arrived reads 0.0.

### The normalization is a hyperparameter

train.py runs one pano per rank per micro-batch, so either norm normalizes **per pano**. The
two differ in how hard a sparse positive pulls against a dense one. The table below is for a
silent miss on one ramp (model output 0), from `scripts/analysis/mask_norm_263.py`, which runs
the functions in train.py:

| normalization | loss | gradient vs a dense pano with the same miss |
|---|---|---|
| dense `nn.MSELoss` | 0.000599 | 1x |
| sparse, `--mask-norm pixels` (default) | 0.000599 | 0.99x |
| sparse, `--mask-norm weight` | 0.1114 | 184.6x |

`pixels` trains a sparse positive **exactly like** a dense positive. Once the dense pano's
background is near zero, its gradient also sits around its ramps, so masking that background
removes almost nothing.

`weight` up-weights a sparse positive by N / sum(w), which is about 186/k for k positives on
the pano:
- 185.9x for one positive (2,821 disc px);
- 37.2x for five;
- 258.9x for one positive whose disc is clipped at y = 0.02 (2,025 px).

That k-dependence means a pano with more ramps counts each ramp less. In a 1:1 mix, the sparse
panos would also dominate the update, and the GSV replay, which is the no-regression anchor,
would contribute about 1/186 of it.

`pixels` is the default because it is parity with the dense path, not because it is known to
train better. The earlier draft of this document claimed the per-pano `weight` choice was what
#263 specified; the issue says nothing about normalization. Which norm trains better on a Lyon
mix is **a hyperparameter for the label-budget curve to test**. Neither has been run.

## Building sparse labels from a labeler run

```bash
python scripts/build_sparse_labels.py \
    --results runs/lyon/results.jsonl --sites runs/lyon/sites.jsonl \
    --mined runs/lyon/mined/labels.jsonl --panos /data/lyon/index.csv \
    --benchmark-ids benchmark/lyon/ids.txt --exclude-ids benchmark/lyon/exclude_ids.txt \
    --city lyon --out /gscratch/scrubbed/$USER/sparse/lyon --limit 20000 --seed 0

python scripts/build_sparse_labels.py ...the same arguments... --check
```

Inputs:

| flag | file | from |
|---|---|---|
| `--results` | `results.jsonl`: one record per processed pano, with `pano.{panorama_id, lat, lng, camera_heading, width, height, capture_date, source, camera_make, camera_model}` and `detections[] = {x_normalized, y_normalized, confidence}` | the labeler run |
| `--manifest` | the run's `manifest.json` (default: beside `results.jsonl`). Its `detection_storage_floor` must be **<= 0.1**, else the build is refused (#266 review H1). A manifest without the key predates the floor (labeler #28) and stored only peaks >= 0.55, which is what labeler `main.py` assumes too. On such a run, "zero stored peaks" and the ignore points would only cover >= 0.55. | the labeler run |
| `--sites` | `sites.jsonl`: `site_id, lat, lng, members[] = {pano_id, x_normalized, y_normalized, confidence, in_refit, lat, lng, …}` | labeler `scripts/fuse_sites.py` |
| `--sites-meta` | `<sites stem>_meta.json` (default: beside `--sites`). Its `params` are recorded in the manifest as `fuse_params`. The build is refused when `params.min_confidence` is above the member threshold, because the recount would then be missing members. | labeler `scripts/fuse_sites.py` |
| `--mined` | `labels.jsonl`: `pano_id, site_id, status, emit, x, y, peak_conf, band, benchmark_pano` | labeler `scripts/mine_city.py` (sidewalk-auto-labeler PR #114, not merged as of 2026-10-20) |
| `--panos` | a directory of native `<id>.jpg`, or a labeler archive `index.csv` (`panorama_id, filename, bytes, sha256`, with the files in `panos/` beside it). The index's sha256 is checked when each image is read. | labeler `scripts/export_benchmark.py` |
| `--benchmark-ids` | pano ids never to emit, one per line. **Required whenever `benchmark/<city>/` exists in this repo** (#266 review M2). | the benchmark bundle |
| `--exclude-ids` | more pano ids never to emit: the spatial buffer around the benchmark, e.g. the 30 m `benchmark/lyon/exclude_ids.txt` that #264 will write | the bundle's build |

Rules, in order:

1. **sparse, `strong_site`** (#266 review H2).
   - A member is a positive only at confidence >= `--member-min-confidence` (default 0.55,
     the operational threshold) and with `in_refit: true` when the field is present.
     `fuse_sites.py` defaults to `min_confidence` 0.30 and attaches 0.1–0.30 detections as
     `in_refit: false` support members, so neither of those trains.
   - A site is strong when its qualifying members span **>= 3 panos**. That count is redone
     here; the file's `n_operational_panos` is not trusted.
   - Each qualifying member is a positive on its own pano, at the member's own
     `x_normalized, y_normalized`.
2. **sparse, `mined`.**
   - Every `labels.jsonl` row with `emit == true` is a positive at its `x, y`.
   - A point both sources name (the same position to 6 decimals) is kept once, as
     `strong_site`.
3. **Exclusion** (#266 review M2).
   - Benchmark panos (`--benchmark-ids`, plus any mined row with `benchmark_pano: true`) and
     `--exclude-ids` panos are never emitted.
   - **Every site with a member among them is dropped whole**: a `sites.jsonl` site checked
     against all its members, a mined site against all its rows. The other members of such a
     site see the same ramp from the same street, often from the same capture, and would
     leak the benchmark's ground truth into training.
   - The manifest counts each kind: `excluded_benchmark_panos`, `excluded_buffer_panos`,
     `dropped_sites_benchmark`, `dropped_sites_excluded` and `dropped_mined_sites`.
4. **verified_empty: proposed, unreviewed, and off unless `--emit-verified-empty`** (#266
   review H1).
   - The rule: a pano with zero stored peaks, **no site of any tier** within 25 m (the site's
     position or any member's ground point), and at least two *other* panos within 25 m that
     do carry peaks.
   - The first draft vetoed only strong sites. Its test fixture emitted a pano 5 m from a
     2-pano site, which is far more likely a silent miss than an empty street.
   - Even this version has no measured precision. 25 m is also inside the range at which the
     detector fires, so a site 26–60 m away may well be visible in the pano. A reviewed
     sample has to come before any arm uses the flag.
5. **Every other pano is skipped**, including panos whose peaks belong to no strong site.
6. **Missing images.** Panos with no native image are dropped and listed in the manifest.
7. **`--limit N`** keeps a seeded uniform sample of what is left (`--seed`, default 0). It
   does not balance modes or sources; the manifest records what the sample came out as.

**Output.**
- Per pano, `<id>.jpg` resized to 4096x2048 (bilinear, JPEG quality 95, from the native
  file) and `<id>.json` as above.
- The image is written first and the JSON second, each atomically, so a JSON on disk always
  has its finished image beside it.
- A **non-empty `--out` is refused unless `--prune` is passed** (#266 review M3). With
  `--prune`, files the new build does not list are deleted, and panos whose JSON is already
  byte-identical are kept, which is how a build resumes.
- `--prune` refuses outright if `--out` holds a dense pano JSON, or any file this tool does
  not write. So it can never empty a GSV split.

`labels_manifest.json` records:
- the inputs (name and sha256), the parameters and the fuse parameters;
- counts per mode and per source, the exclusion counts, and the number of ignore points;
- the missing and emitted ids;
- `labels_sha256`: the sha256 over `"<id>.json\n"` plus that file's bytes, in sorted id
  order.

The JSON is a pure function of the inputs: sorted keys, floats rounded to 6 decimals, LF line
ends. So `--check` re-derives every JSON from the inputs without writing anything, compares it
with the manifest and the files on disk, and reports any file in `--out` the build does not
account for. The re-encoded JPEGs are not hashed: their bytes depend on the libjpeg version,
and the native image's sha256 in each JSON is the durable fact.

Not yet run on any city. The Lyon pool waits on Track 2's review (#263, #159).

## Mixing sparse panos with GSV replay

`train.py` trains on every pair in `<data-root>/train`, sampled uniformly, so a "1:1 Lyon/GSV"
arm needs a training directory holding exactly that mix (#266 review L8).
`scripts/mix_training_dirs.py` builds one out of symlinks:
- every pano of a `build_sparse_labels.py` output;
- a seeded uniform sample of round(`--ratio` x n_sparse) GSV training panos;
- `val` linked to the GSV val directory unchanged, so every arm validates on the same panos.

`mix_manifest.json` records the sparse set's `labels_sha256`, the ratio, the seed, the chosen
GSV ids and `train_sha256` over the sorted (id, source) list. `--check` re-derives the
selection and compares it with the manifest and the links. A pano id present in both sets is
an error. Train on the result with `--data-root <out> --loss-mask`.

## The control arm and the dataset re-stage

`stage_two/run_finetune_control.slurm` fine-tunes the released weights on the GSV training set
only:
- 2,000 optimizer steps at global batch 16, constant LR 1e-5;
- horizontal flip, no augmentation, **no `--loss-mask`**;
- one job per `SEED`.

It is `run_finetune_aug82.slurm`'s `control` arm with these changes:
- **Run directory.** It has its own.
- **Data.** It reads the re-staged dataset, `/gscratch/scrubbed/$USER/rampnet_dataset_v2`.
  At job start it checks for 150,063 train and 42,875 val **jpg+json pairs** (a jpg without
  its JSON is not a pano train.py reads; #266 review L1).
- **Env tarball.** It rebuilds the purged node-local env tarball. It says "Built" only when
  its own copy won the `mv -n` race, and removes its partial temp file if preempted
  mid-tar. After changing the conda env, delete the tar.
- **Copy step.** The final weights (with a sha256) and the TensorBoard events go to
  `/gscratch/makelab/jonf/control263/s<SEED>`.

Its preflight blocks are asserted verbatim against the aug82 launcher in
`tests/test_slurm_scripts.py`. Not yet run.

`stage_two/run_restage_263.slurm` is kept byte-identical to the file that ran on klone as job
42083469, which re-stages the HF dataset into `rampnet_dataset_v2`. One nit, left in place for
that reason: under `set -e`, its `echo "download_dataset.py exit $?"` line runs only when the
download succeeded, so it always prints 0, and a failure leaves no exit line at all. Read the
split counts it prints, and the control's pair check, instead.
