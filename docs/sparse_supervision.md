# Sparse supervision for Stage 2: schema, masked loss, label builder (#263)

**Status: plumbing only.** Nothing has been trained with it. This is Track 3 of
[#263](https://github.com/ProjectSidewalk/RampNet/issues/263): the masked heatmap loss
(`stage_two/train.py --loss-mask`), the builder that turns a sidewalk-auto-labeler run into
sparse-labeled training panos (`scripts/build_sparse_labels.py`), and the GSV-only control
arm every later arm is read against (`stage_two/run_finetune_control.slurm`). Everything is
**off by default**: without `--loss-mask` the published training path is unchanged, and
`tests/test_train_augment_82.py`'s `GOLDEN_OFF` hash still passes.

## Why a mask

A self-labeled open-imagery pano knows where some ramps are (strong fused sites, mined misses)
and nothing about the rest of the image. Training it the dense way, with every unlabeled pixel
as a zero target, teaches "no ramp" wherever the labeler was silent, which on the open rigs is
exactly where RampNet 1.0 misses (the recall gap in #151 and #159). The mask trains such a pano
only near its verified positives, and trains a pano everywhere only when there is a reason to
believe it is empty.

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
  "site_ids": ["lyon:1042", "lyon:877"],
  "run_sha": "…",
  "city": "lyon"
 }
}
```

| field | meaning |
|---|---|
| `curb_ramp_points_normalized` | the verified positives, `[x, y]` as fractions of width and height. For `verified_empty` it is `[]`. |
| `supervision.mode` | `"sparse"` or `"verified_empty"`. **Any other value is an error** (train.py raises), as is a `verified_empty` pano that lists positives. |
| `supervision.sources` | one entry per point, parallel to the point list: `"strong_site"` or `"mined"` (see below). Lets an arm drop one source without rebuilding. |
| `supervision.site_ids` | one entry per point: the labeler site the point came from, as `"<city>:<site_id>"`. A site id is a per-run serial and means nothing without its run. |
| `supervision.run_sha` | sha256 of the run's `results.jsonl`, which identifies the labeler run. |
| `supervision.city` | the city slug. |
| `pano_coord`, `pano_azimuth` | carried from the run: `[lat, lng]`, and the run's `camera_heading` wrapped to [-180, 180) as Stage 1 does (`rampnet.gsv.heading_to_azimuth`). For GSV that is the same quantity Stage 1 stored; for other sources it is the labeler's heading convention. Nothing in training reads either. |
| `pano_meta` | the native image's provenance, including its sha256, so the JSON pins the exact input image. |

A JSON with **no** `supervision` block is **dense**: every published Stage 2 pano. Stage 1's
`curb_ramp_coords` and `record_creation_time` are not written: the first has no counterpart
in a labeler run, and the second would make the output differ on every rebuild.

Positions are normalized, so resizing the native pano (4,096 to 16,384 px wide depending on
the rig) to the dataset's 4096x2048 changes no label.

## What `--loss-mask` does

`EquiHeatmapDataset.__getitem__` returns `(image, heatmap, weight)`; `weight` has the
heatmap's shape, (1, 512, 1024), float32:

| pano | target | weight |
|---|---|---|
| dense (no block) | Gaussians at the points, as now | 1 everywhere |
| `sparse` | Gaussians at the points, as now | 1 inside a disc of radius ceil(3σ) = 30 heatmap px around each point, 0 elsewhere |
| `verified_empty` | all zero | 1 everywhere |
| unreadable JSON, dummy sample | all zero | 0 everywhere (teaches nothing) |

The weight is built from the **same** point list as the target, after the horizontal flip, so
a flipped pano gets a flipped disc. The centre is rounded and clamped exactly as the target's
is, and the disc is clipped at the image edge with **no wrap across the 0/360 seam**, the
same as the target. The disc is exactly the set of pixels where the target is at least
exp(-4.5) = 0.011; only the corners of the target's square window, below that, fall outside
it. `tests/test_train_mask.py` checks each of these.

The loss, for train and validation alike:

```python
def masked_mse(outputs, target, weight):
    return ((outputs.float() - target) ** 2 * weight).sum() / weight.sum().clamp(min=1)
```

With `weight` all ones this is `nn.MSELoss()` (checked to rtol 1e-6), so on the dense GSV
validation set the logged validation loss means what it always has. The clamp keeps an
all-masked batch at 0 rather than NaN.

Without `--loss-mask`, `criterion = nn.MSELoss()` is called exactly as before, and a JSON
that **does** carry a `supervision` block is an error rather than silently trained as dense.

With the flag on, each epoch (and a `--max-steps` run at its end) prints the share of masked
pixels on rank 0's shard and logs it as `Mask/masked_pixel_share`. A sparse-only pool reads
about 0.99; a run whose sparse panos never arrived reads 0.0.

### The normalization is per batch, and that is a decision

The loss divides by the weighted pixel count of the batch. train.py runs one pano per rank
per micro-batch, so in practice the normalization is **per pano**: a sparse pano with one
positive contributes the mean error over its 2,821 disc pixels, at the same total weight as a
dense pano's mean over 524,288 pixels. Per pixel, that is about 186x the weight. That is what
#263 specified, and it makes every pano count once regardless of how much of it is labeled.
The alternative, dividing by the full pixel count so that a masked pixel simply contributes
nothing, would keep per-pixel weights equal to the dense recipe's but make a sparse pano's
gradient about 186x smaller than a dense pano's. Which one trains better on a 1:1 Lyon/GSV
mix is an empirical question; it has not been tested, and it is listed as an open question on
the PR.

## Building sparse labels from a labeler run

```bash
python scripts/build_sparse_labels.py \
    --results runs/lyon/results.jsonl --sites runs/lyon/sites.jsonl \
    --mined runs/lyon/mined/labels.jsonl --panos /data/lyon/index.csv \
    --benchmark-ids benchmark/lyon/ids.txt --city lyon \
    --out /gscratch/scrubbed/$USER/sparse/lyon --limit 20000 --seed 0

python scripts/build_sparse_labels.py ...the same arguments... --check
```

Inputs:

| flag | file | from |
|---|---|---|
| `--results` | `results.jsonl`: one record per processed pano, `pano.{panorama_id, lat, lng, camera_heading, width, height, capture_date, source, camera_make, camera_model}` and `detections[] = {x_normalized, y_normalized, confidence}` (every peak >= 0.1, at most 50) | the labeler run |
| `--sites` | `sites.jsonl`: `site_id, lat, lng, n_operational_panos, members[] = {pano_id, x_normalized, y_normalized, confidence, …}` | labeler `scripts/fuse_sites.py` |
| `--mined` | `labels.jsonl`: `pano_id, site_id, status, emit, x, y, peak_conf, band, benchmark_pano` | labeler `scripts/mine_city.py` (sidewalk-auto-labeler PR #114, not merged as of 2026-10-20) |
| `--panos` | a directory of native `<id>.jpg`, or a labeler archive `index.csv` (`panorama_id, filename, bytes, sha256`, files in `panos/` beside it); the index's sha256 is checked when each image is read | labeler `scripts/export_benchmark.py` |
| `--benchmark-ids` | pano ids never to emit, one per line | the benchmark bundle |

Rules, in order:

1. **sparse, `strong_site`.** A strong site has `n_operational_panos >= 3` (operational =
   confidence >= 0.55, the miners' fuse). Every member of a strong site is a positive on its
   own pano, at the member's own `x_normalized, y_normalized`.
2. **sparse, `mined`.** Every `labels.jsonl` row with `emit == true` is a positive at its
   `x, y`. A point both sources name (same position to 6 decimals) is kept once, as
   `strong_site`.
3. **verified_empty (proposed rule, not validated).** A pano with zero stored peaks, no strong
   site within 25 m (haversine from the pano to the site), and at least two *other* panos
   within 25 m that do carry peaks. The intent: the detector was active on comparable imagery
   right there and still found nothing in this pano. Whether such panos really are empty has
   not been measured; that needs a reviewed sample before any arm relies on them.
4. **Every other pano is skipped**, including panos whose peaks belong to no strong site.
5. Benchmark panos, from `--benchmark-ids` or any mined row with `benchmark_pano: true`, are
   excluded; so are panos with no native image (listed in the manifest).
6. `--limit N` keeps a seeded uniform sample of what is left (`--seed`, default 0). It does not
   balance modes or sources; the manifest records what the sample came out as.

Output, per pano: `<id>.jpg` resized to 4096x2048 (bilinear, JPEG quality 95, from the native
file) and `<id>.json` as above, written image first, JSON second, each atomically, so a JSON on
disk always has its finished image beside it. A rerun skips panos whose JSON is already
byte-identical. The builder refuses to overwrite a JSON without a `supervision` block, so
pointing `--out` at a split that already holds dense panos cannot clobber them.

`labels_manifest.json` records the inputs (name and sha256), the parameters, counts per mode
and per source, the excluded and missing ids, the emitted ids, and `labels_sha256`: sha256
over `"<id>.json\n"` + that file's bytes, in sorted id order. The JSON is a pure function of
the inputs (sorted keys, floats rounded to 6 decimals, LF line ends), so `--check` re-derives
every JSON from the inputs without writing anything and compares it with the manifest and the
files on disk, and reports any sparse-label JSON in `--out` that the build does not account
for. The re-encoded JPEGs are not hashed: their bytes depend on the libjpeg version, and the
native image's sha256 in each JSON is the durable fact.

Not yet run on any city. The Lyon pool waits on Track 2's review (#263, #159).

## The control arm

`stage_two/run_finetune_control.slurm` fine-tunes the released weights on the GSV training set
only: 2,000 optimizer steps at global batch 16, constant LR 1e-5, horizontal flip, no
augmentation, **no `--loss-mask`**, one job per `SEED`. It is `run_finetune_aug82.slurm`'s
`control` arm with its own run directory, the re-staged dataset
(`/gscratch/scrubbed/$USER/rampnet_dataset_v2`, written by `stage_two/run_restage_263.slurm`,
checked at job start for 150,063 train and 42,875 val panos), a rebuild of the purged
node-local env tarball, and a copy of the final weights (with sha256), events and log to
`/gscratch/makelab/jonf/control263/s<SEED>`. Its preflight blocks are asserted verbatim
against the aug82 launcher in `tests/test_slurm_scripts.py`. Not yet run.
