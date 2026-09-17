"""Scan the PUBLISHED Stage 1 training dataset for the 360-seam defect (#132).

Result, all 384 parquet files (analysis_out/stage1_seam_scan.json):

  849,904 labels over 214,385 panoramas
  8,361 seam-crossing label pairs against 485.0 expected -- 17.24x, ~358 sigma
  after removing them, the two seam-adjacent azimuth bins hold 0.315x and 0.293x
  of mean density, recovering monotonically to baseline by ~17 degrees

so BOTH failure modes are present and measured: ramps on the seam are labelled
twice, and the seam region is under-labelled overall.

Reads only the label columns from the HF parquet (column projection: ~0.003 MB per
row group vs 203 MB for images), so this touches ~12 MB, not 463 GB.

Two failure modes, opposite signs, both from peak_local_max not wrapping in
stage_one/dataset_generation/download_dataset.py:
  * DOUBLING  - one seam ramp emits a peak on each edge -> two labels
  * DROPOUT   - the split response falls under threshold_abs -> zero labels
Doubling shows as seam-crossing label pairs above the uniform-azimuth null.
Dropout shows as a deficit of labels in the columns nearest x=0/x=1.
"""
import argparse
import json, math, os, sys
from collections import Counter
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# The repo checkout, for default output paths -- NOT to be confused with REPO,
# which is the Hugging Face dataset id.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "projectsidewalk/rampnet-dataset"
SX, SY = 1024, 512                 # matcher units, as in rampnet.detection_eval
R = 0.022 * SX
COLS = ["pano_id", "curb_ramp_points_normalized", "pano_azimuth"]
NBINS = 64                         # azimuth histogram, 5.625 deg per bin

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="Scan only the first N parquet files (smoke test).")
    ap.add_argument("--json-out", default=os.path.join(REPO_DIR, "analysis_out",
                                                       "stage1_seam_scan.json"),
                    help="Where to write the result (default: analysis_out/).")
    args = ap.parse_args()
    limit = args.limit
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".parquet")]
    files.sort()
    if limit:
        files = files[:limit]
    print(f"scanning {len(files)} parquet files", flush=True)

    hist = [0] * NBINS
    hist_dd = [0] * NBINS
    panos_with_seam_pair = 0
    seam_by_split = Counter()
    n_panos = n_labels = 0
    seam_obs = 0
    seam_exp = 0.0
    near_pairs = 0
    seam_examples = []
    seps = []
    by_split = Counter()

    for i, fp in enumerate(files, 1):
        split = fp.split("/")[0]
        t = pq.ParquetFile(f"hf://datasets/{REPO}/{fp}").read(columns=COLS)
        pids = t.column("pano_id").to_pylist()
        pts_col = t.column("curb_ramp_points_normalized").to_pylist()
        for pid, pts in zip(pids, pts_col):
            n_panos += 1
            by_split[split] += 1
            pts = [(float(p[0]), float(p[1])) for p in (pts or [])]
            n_labels += len(pts)
            for x, _y in pts:
                hist[min(NBINS - 1, int(x * NBINS))] += 1
            keep, dropped = [], 0
            for g in pts:
                dup = False
                for k in keep:
                    dxk = abs(g[0] - k[0]) * SX
                    if dxk <= SX / 2:
                        continue
                    if (SX - dxk) ** 2 + ((g[1] - k[1]) * SY) ** 2 < R * R:
                        dup = True
                        break
                if dup:
                    dropped += 1
                else:
                    keep.append(g)
            if dropped:
                panos_with_seam_pair += 1
                seam_by_split[split] += dropped
            for x, _y in keep:
                hist_dd[min(NBINS - 1, int(x * NBINS))] += 1
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    dx = abs(pts[a][0] - pts[b][0]) * SX
                    wdx = min(dx, SX - dx)
                    dy = abs(pts[a][1] - pts[b][1]) * SY
                    if (wdx * wdx + dy * dy) ** 0.5 < R:
                        near_pairs += 1
                        seam_exp += wdx / SX
                        if dx > SX / 2:
                            seam_obs += 1
                            seps.append(round((wdx * wdx + dy * dy) ** 0.5, 2))
                            if len(seam_examples) < 40:
                                seam_examples.append(
                                    {"split": split, "pano_id": pid,
                                     "a": [round(v, 5) for v in pts[a]],
                                     "b": [round(v, 5) for v in pts[b]]})
        if i % 16 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  panos={n_panos} labels={n_labels} "
                  f"seam_pairs={seam_obs} (exp {seam_exp:.1f})", flush=True)

    out = {"repo": REPO, "files": len(files), "panos": n_panos, "labels": n_labels,
           "by_split": dict(by_split), "within_radius_pairs": near_pairs,
           "seam_crossing_pairs": seam_obs, "seam_expected_uniform_null": seam_exp,
           "seam_pair_separations_px": seps, "azimuth_hist_64": hist, "azimuth_hist_64_seam_deduped": hist_dd,
           "panos_with_seam_pair": panos_with_seam_pair,
           "seam_dupes_by_split": dict(seam_by_split),
           "examples": seam_examples}
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", newline="") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.json_out}")

    mean = n_labels / NBINS
    print(f"\nlabels={n_labels} over {n_panos} panos")
    print(f"seam-crossing label pairs within R: {seam_obs}   expected under null: {seam_exp:.2f}")
    if seam_exp > 0:
        z = (seam_obs - seam_exp) / math.sqrt(seam_exp)
        print(f"  enrichment {seam_obs / seam_exp:.2f}x   z = {z:.0f} sigma")
    print(f"\nazimuth bins touching the seam (uniform => {mean:.0f} each):")
    print(f"  bin 0  (x in [0,{1/NBINS:.4f})) = {hist[0]}   ratio {hist[0]/mean:.3f}")
    print(f"  bin 63 (x in [{63/NBINS:.4f},1)) = {hist[63]}   ratio {hist[63]/mean:.3f}")
    mid = sorted(hist[8:56])
    print(f"  interior bins 8-55: median {mid[len(mid)//2]}, min {mid[0]}, max {mid[-1]}")

main()
