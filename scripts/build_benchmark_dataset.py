"""Build the RampNet deployment-validation benchmark as a HuggingFace dataset (issue #21).

Packs the committed benchmark bundles (``benchmark/{bend,richmond}/`` — native-res
panoramas + model detections + human verdicts) into a ``DatasetDict`` with one
split per city, writes a domain-labeled dataset card, and verifies that scoring
the reloaded dataset reproduces the documented precision/recall. Does NOT push
(review first, then push with the printed command).

    python scripts/build_benchmark_dataset.py --output-dir hf_benchmark_export

Two splits, two very different measurements — kept separate on purpose:

* ``bend``     — Google Street View, one of the three training cities. An
  **in-distribution** reference point (same city distribution, same imagery
  source as training).
* ``richmond`` — Mapillary, a city never seen in training. The **out-of-
  distribution** deployment test (unseen city AND unseen imagery source).

Four Bend benchmark panoramas whose ids appear in the training split of
``projectsidewalk/rampnet-dataset`` are dropped (see ``LEAKED_BEND_IDS``); their
removal shifts Bend precision/recall by <=0.5 pt (inside the Wilson CIs).
Richmond has zero training overlap.
"""
import argparse
import glob
import json
import os
import sys

import PIL.Image
from datasets import Dataset, Features, Image, Value, load_dataset

# Bend GSV panoramas run up to 16384x8192 (134 MP), above Pillow's default
# decompression-bomb ceiling. These are trusted local files, so lift the cap.
PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from rampnet.validation import collect, format_report  # noqa: E402

BENCHMARK_DIR = os.path.join(REPO_ROOT, "benchmark")

# Bend benchmark panos whose ids are in projectsidewalk/rampnet-dataset train/val
# (exact-id overlap check, 2026-07-22). Dropped from the published benchmark.
LEAKED_BEND_IDS = {
    "6WC0hdAYRsSAcluKSs5iRg", "9kW9cxpuj7q8DMzf-ClrQQ",
    "DJ8Zp111zu6KnMZz-0PHgQ", "VgWpqFkTwCIROvM0z-DkOw",
}

CITY_IMAGERY = {"bend": "gsv", "richmond": "mapillary"}

FEATURES = Features({
    "pano_id": Value("string"),
    "city": Value("string"),
    "imagery_source": Value("string"),      # gsv | mapillary
    # True iff this exact pano id is in a projectsidewalk/rampnet-dataset training
    # split. Nothing is dropped — filter on this to score the clean subset.
    "train_overlap": Value("bool"),
    "group": Value("string"),               # top | random | empty (sampling stratum)
    "image": Image(),
    "lat": Value("float64"),
    "lng": Value("float64"),
    "capture_date": Value("string"),
    "width": Value("int64"),
    "height": Value("int64"),
    "camera_heading": Value("float64"),
    "copyright": Value("string"),
    # Model detections at the panorama, normalized coords + confidence
    # (list-of-structs; the [{...}] wrapper keeps list-of-dict input, not columnar).
    "detections": [{
        "x_normalized": Value("float64"),
        "y_normalized": Value("float64"),
        "confidence": Value("float64"),
    }],
    # Human verdict per detection, aligned 1:1 with `detections`:
    # "true" (correct) | "false" (false positive) | "unsure" (abstain) | "duplicate".
    "det_verdicts": [Value("string")],
    # Ground-truth ramps the model missed (false negatives); `unsure` abstains.
    "missed": [{
        "x": Value("float64"),
        "y": Value("float64"),
        "unsure": Value("bool"),
    }],
    "no_missed": Value("bool"),
    # Full raw panorama metadata (superset of the surfaced columns), as JSON.
    "pano_metadata_json": Value("string"),
})


def encode_verdict(v):
    """bool/str verdict -> homogeneous string for a parquet list column."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)  # "unsure" | "duplicate"


def load_city(city):
    cdir = os.path.join(BENCHMARK_DIR, city)
    records = {}
    with open(os.path.join(cdir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    verdicts = json.load(open(os.path.join(cdir, "verdicts.json"), encoding="utf-8"))["panos"]
    return records, verdicts, os.path.join(cdir, "panos")


def build_rows(city, read_images=True):
    """Build one split's row dicts. ``read_images=False`` skips the ~GB of JPEG
    reads (image set to None) for fast, image-free checks/tests. Nothing is
    dropped; panos with training-split id overlap are flagged via ``train_overlap``.
    Returns ``(rows, flagged_ids)``."""
    records, verdicts, panos_dir = load_city(city)
    flagged = []
    rows = []
    for pid, ver in verdicts.items():
        overlap = city == "bend" and pid in LEAKED_BEND_IDS
        if overlap:
            flagged.append(pid)
        rec = records[pid]
        pano = rec["pano"]
        img_path = os.path.join(panos_dir, f"{pid}.jpg")
        # Embed the raw JPEG bytes: a bare path is stored as an unresolved
        # reference (to_parquet does not read it), so the Hub parquet would ship
        # imageless. bytes-in, and the image travels with the row.
        image = None
        if read_images:
            with open(img_path, "rb") as fh:
                image = {"path": f"{pid}.jpg", "bytes": fh.read()}
        rows.append({
            "pano_id": pid,
            "city": city,
            "imagery_source": CITY_IMAGERY[city],
            "train_overlap": overlap,
            "group": ver["group"],
            "image": image,
            "lat": pano.get("lat"),
            "lng": pano.get("lng"),
            "capture_date": pano.get("capture_date"),
            "width": pano.get("width"),
            "height": pano.get("height"),
            "camera_heading": pano.get("camera_heading"),
            "copyright": pano.get("copyright"),
            "detections": [
                {"x_normalized": d["x_normalized"], "y_normalized": d["y_normalized"],
                 "confidence": d["confidence"]}
                for d in rec["detections"]
            ],
            "det_verdicts": [encode_verdict(v) for v in ver["dets"]],
            "missed": [
                {"x": m["x"], "y": m["y"], "unsure": bool(m.get("unsure", False))}
                for m in ver.get("missed", [])
            ],
            "no_missed": ver["no_missed"],
            "pano_metadata_json": json.dumps(pano, ensure_ascii=False),
        })
    return rows, flagged


def decode_verdicts_for_scoring(split_ds):
    """Reconstruct the (verdicts, confs) inputs rampnet.validation.collect expects
    from a built split — the round-trip used to verify the published numbers."""
    def decode(v):
        return {"true": True, "false": False}.get(v, v)  # keep unsure/duplicate strings

    verdicts, confs = {}, {}
    for row in split_ds:
        pid = row["pano_id"]
        verdicts[pid] = {
            "group": row["group"],
            "dets": [decode(v) for v in row["det_verdicts"]],
            "missed": row["missed"],
            "no_missed": row["no_missed"],
        }
        confs[pid] = [d["confidence"] for d in row["detections"]]
    return verdicts, confs


def score_split(split_ds, title):
    verdicts, confs = decode_verdicts_for_scoring(split_ds)
    # Default flags: the splits bake complete-scan attestation into `no_missed`,
    # so the numbers reproduce without --assume-scanned (see benchmark/README.md).
    pools = collect(verdicts, confs)
    return format_report(title, pools)


def pr_from_report(report):
    """Extract ('0.960', '0.765') from a format_report block."""
    import re
    p = re.search(r"Precision:\s*([\d.]+)", report).group(1)
    r = re.search(r"Recall:\s*([\d.]+)", report).group(1)
    return p, r


def render_card(counts, reports, bend_clean):
    """bend_clean = (precision, recall) strings for Bend with train_overlap rows
    excluded — the numbers reproduced by filtering `train_overlap == False`."""
    leaked = ", ".join(f"`{i}`" for i in sorted(LEAKED_BEND_IDS))
    bend_clean_p, bend_clean_r = bend_clean
    return f"""---
license: cc-by-sa-4.0
task_categories:
  - keypoint-detection
tags:
  - curb-ramp-detection
  - accessibility
  - street-view
  - benchmark
  - deployment-validation
pretty_name: RampNet Deployment-Validation Benchmark
configs:
  - config_name: default
    data_files:
      - split: bend
        path: bend/*.parquet
      - split: richmond
        path: richmond/*.parquet
---

# RampNet Deployment-Validation Benchmark

Human-verified ground truth for evaluating the
[projectsidewalk/rampnet-model](https://huggingface.co/projectsidewalk/rampnet-model)
curb-ramp detector **in deployment**, on real street-view panoramas. Each panorama carries
the model's detections, a human verdict per detection (correct / false-positive / unsure /
duplicate), and any ground-truth ramps the model **missed** — enough to reproduce precision and
recall exactly (see [Scoring](#scoring)).

This complements the training-distribution gold set reported with the model. It measures the two
things a deployment actually cares about, kept in **separate splits** because they answer different
questions:

| Split | Imagery | City vs. training | Measures | n |
| :--- | :--- | :--- | :--- | ---: |
| `bend` | Google Street View | **in-distribution** (a training city, same imagery source) | in-domain reference point | {counts['bend']} |
| `richmond` | Mapillary | **out-of-distribution** (unseen city **and** unseen imagery source) | true deployment generalization | {counts['richmond']} |

Do **not** pool these into one number: `bend` tells you how the model does on data like what it was
trained on; `richmond` tells you how it transfers to a new city and a new camera. That the OOD
Richmond split scores on par with the in-distribution Bend split is the headline result.

## Results (recommended operating threshold 0.55, flip-TTA)

{reports['bend']}

{reports['richmond']}

Precision = correct detections / all detections; recall = matched ground-truth ramps / all visible
ground-truth ramps in the recall pool. `unsure` verdicts abstain (excluded from both); `duplicate`
detections count as false positives by default. 95% Wilson confidence intervals are shown because
the splits are small. These supersede all earlier figures (the original 1600 px review overstated
recall; these were re-reviewed at model input resolution).

## Training overlap (important)

Bend is one of RampNet's three training cities, so its imagery distribution is **not held out** —
treat the `bend` split as an in-distribution reference, not a generalization test. Beyond the
distribution overlap, four Bend panoramas have exact ids present in the `train`/`val` split of
[projectsidewalk/rampnet-dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset):
{leaked}. They are **kept** in the split but flagged with `train_overlap = true` — nothing is
dropped, so you can score either way. Excluding them (`train_overlap == False`, n={counts['bend'] - 4})
gives precision {bend_clean_p} / recall {bend_clean_r} — a shift of <=0.5 pt, within the confidence
intervals. **Richmond has zero id overlap** with any training split (`train_overlap` all false) and is
a clean out-of-distribution test.

The headline Bend numbers above are on the full split (n={counts['bend']}), matching the in-repo
`benchmark/bend` bundle.

## Columns

- `pano_id`, `city`, `imagery_source` (`gsv` | `mapillary`), `train_overlap` (bool; see
  [Training overlap](#training-overlap-important)), `group` (`top` | `random` | `empty` sampling
  stratum), `image` (the panorama), plus panorama metadata (`lat`, `lng`, `capture_date`, `width`,
  `height`, `camera_heading`, `copyright`, and the full raw `pano_metadata_json`).
- `detections`: model detections — normalized `x`/`y` + `confidence`.
- `det_verdicts`: aligned 1:1 with `detections` — `"true"` | `"false"` | `"unsure"` | `"duplicate"`.
- `missed`: ground-truth ramps the model missed (false negatives), normalized `x`/`y`; `unsure`
  abstains.
- `no_missed`: reviewer attestation that the panorama was fully scanned for misses.

## Scoring

The scorer lives in the training repo
([`rampnet/validation.py`](https://github.com/ProjectSidewalk/RampNet/blob/main/rampnet/validation.py)):

```python
from datasets import load_dataset
from rampnet.validation import collect, format_report

ds = load_dataset("projectsidewalk/rampnet-benchmark", split="richmond")

def decode(v):
    return {{"true": True, "false": False}}.get(v, v)  # keep unsure/duplicate

verdicts, confs = {{}}, {{}}
for row in ds:
    verdicts[row["pano_id"]] = {{
        "group": row["group"], "dets": [decode(v) for v in row["det_verdicts"]],
        "missed": row["missed"], "no_missed": row["no_missed"],
    }}
    confs[row["pano_id"]] = [d["confidence"] for d in row["detections"]]

print(format_report("richmond", collect(verdicts, confs)))
```

(The splits bake complete-scan attestation into `no_missed`, so scoring needs no
`assume_scanned` override — see the [results](#results-recommended-operating-threshold-055-flip-tta).)

## Provenance

Detections are from `projectsidewalk/rampnet-model` (`model_training_date` 2025-08-21). Imagery ©
Google (Bend, GSV) and © the respective Mapillary contributors (Richmond, CC BY-SA 4.0); see each
panorama's `copyright`. Built by
[`scripts/build_benchmark_dataset.py`](https://github.com/ProjectSidewalk/RampNet/blob/main/scripts/build_benchmark_dataset.py).
"""


def main():
    ap = argparse.ArgumentParser(description="Build the RampNet deployment-validation benchmark dataset (#21).")
    ap.add_argument("--output-dir", default="hf_benchmark_export",
                    help="Where to write the dataset (save_to_disk) + README.md")
    ap.add_argument("--repo-id", default="projectsidewalk/rampnet-benchmark",
                    help="Target Hub dataset id (used only in the printed push instructions)")
    args = ap.parse_args()

    # ~500 MB/shard target; Bend native-res GSV is the large split.
    SHARDS = {"bend": 4, "richmond": 1}

    counts, flagged_all, reports = {}, {}, {}
    bend_clean = None
    os.makedirs(args.output_dir, exist_ok=True)
    for city in ("bend", "richmond"):
        rows, flagged = build_rows(city)
        ds = Dataset.from_list(rows, features=FEATURES)
        counts[city] = len(ds)
        flagged_all[city] = flagged
        reports[city] = score_split(ds, f"{city} (n={len(ds)}, {CITY_IMAGERY[city]})")
        print(f"[{city}] {len(ds)} panos"
              + (f"; {len(flagged)} flagged train_overlap: {flagged}" if flagged else ""))

        if city == "bend":  # numbers with the training-overlap panos excluded
            clean = ds.filter(lambda r: not r["train_overlap"])
            bend_clean = pr_from_report(score_split(clean, "bend (clean)"))

        # Write the exact repo layout the card's `configs` block declares:
        # <out>/<city>/data-<i>-of-<n>.parquet (self-contained, image bytes embedded).
        city_dir = os.path.join(args.output_dir, city)
        os.makedirs(city_dir, exist_ok=True)
        n = SHARDS[city]
        for i in range(n):
            shard = ds.shard(num_shards=n, index=i, contiguous=True)
            shard.to_parquet(os.path.join(city_dir, f"data-{i:05d}-of-{n:05d}.parquet"))

    card = render_card(counts, reports, bend_clean)
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"\nSaved parquet splits + README.md to {args.output_dir}")

    # Verify: scoring the reloaded parquet reproduces the just-computed reports.
    for city in ("bend", "richmond"):
        reloaded = load_dataset("parquet",
                                data_files=os.path.join(args.output_dir, city, "*.parquet"),
                                split="train")
        again = score_split(reloaded, f"{city} (reloaded)")
        line = next(l for l in reports[city].splitlines() if l.strip().startswith("Precision"))
        line2 = next(l for l in again.splitlines() if l.strip().startswith("Precision"))
        assert line == line2, f"{city} round-trip mismatch:\n{line}\n{line2}"
    print("Round-trip verified: reloaded parquet re-scores to the documented numbers.")

    print("\nNOT pushed. To publish (after review), upload the staged folder as-is:")
    print(f'  from huggingface_hub import HfApi')
    print(f'  api = HfApi(); api.create_repo("{args.repo_id}", repo_type="dataset", exist_ok=True)')
    print(f'  api.upload_folder(folder_path="{args.output_dir}", repo_id="{args.repo_id}", '
          f'repo_type="dataset")')


if __name__ == "__main__":
    main()
