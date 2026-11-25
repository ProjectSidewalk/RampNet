"""Check a Stage 2 root's hard negatives through the trainer's own code (RampNet#274, crosswalk_s3).

Every label JSON is parsed with `stage_two/train.py`'s `supervision_mode` and `supervision_negative_points` (which
refuse a negative within NEGATIVE_MIN_DIST of a positive); a negative outside train stops the check. Then every
train pano that carries negatives goes through `EquiHeatmapDataset.__getitem__` (loss mask on, no flip), and at
each negative's heatmap pixel the weight must be 1 and the target about 0. The functions are lifted from train.py
as the tests do (it parses arguments and sets up CUDA on import). CPU only.

  python scripts/analysis/stage1_crosswalk_sdot_274/check_root_negatives.py <root> [<new_panos.csv>]

With `new_panos.csv` (export_stage2.py add-panos, crosswalk_s4) every pano it lists also goes through the dataset,
negatives or not, and at each of its positives the weight must be 1 and the target at least 0.9.
"""
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402
from rampnet import augment as A  # noqa: E402

SHAPE = (512, 1024)


def main():
    root = sys.argv[1]
    new = set()
    if len(sys.argv) > 2:
        import csv
        with open(sys.argv[2], encoding="utf-8", newline="") as f:
            new = {r["pano_id"] for r in csv.DictReader(f)}
    mod = load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points", "generate_weight_from_points",
        "generate_dense_ignore_weight", "supervision_ignore_boxes", "supervision_negative_points",
        "add_negative_weight", "NEGATIVE_MIN_DIST", "supervision_mode", "SUPERVISION_MODES", "_point_list",
        "_nearest_dist_sq", os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F, random=random,
        rank=1, A=A, Dataset=torch.utils.data.Dataset)
    n = nneg = 0
    for s in ("train", "val", "test"):
        for p in sorted(glob.glob(os.path.join(root, s, "*.json"))):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            pts = [tuple(q) for q in d["curb_ramp_points_normalized"]]
            mod.supervision_mode(d, pts)
            neg = mod.supervision_negative_points(d, pts)
            if neg and s != "train":
                raise SystemExit(f"negative outside train: {p}")
            n += 1
            nneg += len(neg)
    print(f"{n} label files parse; {nneg} negatives, all on train")
    ds = mod.EquiHeatmapDataset(root_dir=root, split="train", target_heatmap_shape=SHAPE,
                                transform_input=transforms.ToTensor(),
                                points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
                                apply_horizontal_flip=False, loss_mask=True)
    checked = checked_pos = seen_new = 0
    for i, jp in enumerate(ds.json_paths):
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        negs = (d.get("supervision") or {}).get("negative_points")
        is_new = os.path.basename(jp)[:-5] in new
        if not negs and not is_new:
            continue
        _, hm, w = ds[i]
        hm, w = hm.numpy()[0], w.numpy()[0]
        if is_new:
            seen_new += 1
            for x, y in d["curb_ramp_points_normalized"]:
                r, c = min(int(y * SHAPE[0]), SHAPE[0] - 1), min(int(x * SHAPE[1]), SHAPE[1] - 1)
                if not (w[r, c] == 1 and hm[r, c] >= 0.9):
                    raise SystemExit(f"{jp}: positive ({x}, {y}) weight {w[r, c]} target {hm[r, c]}")
                checked_pos += 1
        negs = negs or []
        for x, y in negs:
            r, c = min(int(y * SHAPE[0]), SHAPE[0] - 1), min(int(x * SHAPE[1]), SHAPE[1] - 1)
            if not (w[r, c] == 1 and hm[r, c] < 0.05):
                raise SystemExit(f"{jp}: negative ({x}, {y}) weight {w[r, c]} target {hm[r, c]}")
            checked += 1
    print(f"{checked} negatives through the dataset: weight 1, target ~0 at each")
    if new:
        if seen_new != len(new):
            raise SystemExit(f"{len(new) - seen_new} panos of new_panos.csv are not in {root}/train")
        print(f"{seen_new} new panos through the dataset; {checked_pos} positives: weight 1, target >= 0.9 at each")


if __name__ == "__main__":
    main()
