"""Sub-cell decode on the Stage 1 crop model, measured against manual labels (#221).

The crop model has the pano model's head (``Conv3x3 -> ReLU -> Upsample(bilinear) ->
Conv1x1`` on a stride-32 ConvNeXt map), so its 256x88 heatmap is exactly a bilinear x8
upsample of a 32x11 coarse map, and every ``peak_local_max`` peak sits on an 8-px grid.
``rampnet/subcell.py`` recovers the sub-cell position from the 3x3 coarse neighbourhood.
``docs/subcell_decode_221.md`` measured that on the pano model; this script measures it
on the crop model, whose training target is wider (sigma 12 heatmap px = 1.5 coarse
cells, against 10 px = 1.25 cells on the pano).

Subcommands (run from the repo root with PYTHONPATH set to it)::

    # network: the pinned checkpoints and the round-2 crop dataset's test/ and val/
    python scripts/analysis/crop_decode_221.py fetch --cache .hf_cache \
        --out analysis_out/crop_decode_221/inputs.json

    # GPU (or CPU): two fp32 forwards per crop (straight + mirrored, as the committed
    # crop evaluator does), capturing the 32x11 coarse map of each branch
    python scripts/analysis/crop_decode_221.py extract --split test --checkpoint round2 \
        --cache .hf_cache --out analysis_out/crop_decode_221/extract_test.json \
        --usage-out analysis_out/crop_decode_221/usage_test.json

    # CPU, no model, no images: rebuild the heatmaps from the committed coarse maps,
    # decode, match, bootstrap
    python scripts/analysis/crop_decode_221.py report \
        --extract analysis_out/crop_decode_221/extract_test.json \
        --out analysis_out/crop_decode_221/results_test.json \
        --md analysis_out/crop_decode_221/results_test.md

    # re-derive every committed results_*.json / .md from the committed extracts and
    # fail on any byte difference
    python scripts/analysis/crop_decode_221.py --check

Protocol (what differs from the pano read in ``subcell_decode_221.py``): no x wrap (a crop
is not a panorama); matching uses the committed crop evaluator's geometry verbatim
(``stage_one/crop_model/ps_and_manual_model/evaluate.py``: radius 0.132 normalized,
``scale_x = 341/4``, ``scale_y = 1024/4``, so 11.25 units); the flip-TTA arm combines the
two branches the way that evaluator does (clip each to [0, 1], flip the mirrored one back,
elementwise max). Pairs are fixed once per arm on the argmax peaks >= 0.30 (greedy by
confidence) and every decode is scored on the same pairs, so the comparison is paired.
Residual = GT - detection, in heatmap pixels of the 256x88 grid (x: 1 px = 682/88 = 7.75
source px = 4 model-input px; y: 1 px = 8 source px = 4 model-input px). The bootstrap
resamples distinct images (sha256), 2,000 reps, seed 221: the round-2 splits store a
multi-point crop once per point, so one image can appear as several files, and those
files must move together. A deduplicated read (one file per image) is reported beside
the all-files one.
"""
import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet import subcell as sc  # noqa: E402
from rampnet.metrics import (calculate_ap_and_pr_curve, greedy_match,  # noqa: E402
                             match_predictions)

OUT_DIR = os.path.join(REPO, "analysis_out", "crop_decode_221")
HM = (256, 88)                     # crop heatmap (H, W)
COARSE = (32, 11)
INPUT = (1024, 352)                # model input (H, W), evaluate.py MODEL_INPUT_SIZE
FACTOR = sc.FACTOR
# Verbatim from stage_one/crop_model/ps_and_manual_model/evaluate.py.
RADIUS_NORM = 0.132
SCALE_X = 341 / 4
SCALE_Y = 1024 / 4
RADIUS_SQ = (RADIUS_NORM * SCALE_X) ** 2          # 11.253 units, squared
MIN_DISTANCE = 10
PAIR_THRESHOLD = 0.30
THRESHOLDS = (0.30, 0.0, 0.55)
SIGMA = 12.0                       # training target sigma, heatmap px (both train.py)
N_REPS = 2000
SEED = 221
ND = 4                             # rounding of reported statistics
COARSE_ND = 7                      # rounding of stored coarse values
METHODS = list(sc.METHODS)
ARMS = ("single", "tta")

MODEL_REPO = "projectsidewalk/rampnet-crop-model"
MODEL_REVISION = "7aa79b8edb10b384ed69c2e99f74945e9c527fd3"
#: sha256 of each checkpoint file at MODEL_REVISION, read 2026-10-05 from the Hub API's
#: LFS metadata (``model_info(..., files_metadata=True)``) and re-hashed on download.
#: The .pth values match the model card; the safetensors values are not on the card.
CHECKPOINTS = {
    "round2": ("round2_ps_and_manual_best_model.safetensors",
               "d129c0c6beffbb565633042f41598e44874fd27012f1c98ea5eb326851a75239"),
    "round1": ("round1_ps_best_model.safetensors",
               "23e40b5926bf377d1b1065fe4adaca08e23284926fd8d751c685b64ac92509b7"),
}
PTH_SHA256 = {
    "round2_ps_and_manual_best_model.pth":
        "3fc00ad6b9ac2768787b0262588b9bfa71ddd01d9f51109974e6ae377b9b520a",
    "round1_ps_best_model.pth":
        "00dba3948298a313435b7c1955a2d4fccde43bc98c199e384ef197bf8b8cff49",
}
DATA_REPO = "projectsidewalk/rampnet-crop-model-dataset-round2"
DATA_REVISION = "9e902acf3bf23bb38122d3a7ebd0d9b9dcd5cfce"
SPLITS = ("test", "val")

#: The committed extracts and the results each one produces; ``--check`` walks this list.
COMMITTED = (
    ("extract_test.json", "results_test"),
    ("extract_val.json", "results_val"),
    ("extract_test_round1.json", "results_test_round1"),
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def rnd(v, nd=ND):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=False) + "\n"


def write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_gt(filename, width, height):
    """GT points from a crop filename, normalized exactly as evaluate.py does.

    ``{uid}_-_{x}_{y}[_-_{x}_{y}...].jpg`` -> ``[(x / width, y / height), ...]``.
    Unparseable point strings are skipped (evaluate.py prints a warning and skips them).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_-_")
    pts = []
    for point_str in parts[1:]:
        try:
            xs, ys = point_str.split("_")
            pts.append((int(xs) / width, int(ys) / height))
        except ValueError:
            continue
    return pts


# --------------------------------------------------------------------------- #
# fetch (network)
# --------------------------------------------------------------------------- #
def cmd_fetch(args):
    from huggingface_hub import hf_hub_download, snapshot_download
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {}
    for key, (fname, want) in CHECKPOINTS.items():
        p = hf_hub_download(MODEL_REPO, fname, revision=MODEL_REVISION, cache_dir=args.cache)
        got = sha256_file(p)
        if got != want:
            raise SystemExit(f"{fname}: sha256 {got} != pinned {want}")
        files[f"{MODEL_REPO}/{fname}"] = {"sha256": got, "bytes": os.path.getsize(p)}
    root = snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REVISION,
                             cache_dir=args.cache,
                             allow_patterns=[f"{s}/*" for s in SPLITS])
    data = {}
    for s in SPLITS:
        d = os.path.join(root, s)
        names = sorted(n for n in os.listdir(d) if n.lower().endswith(".jpg"))
        data[s] = {n: sha256_file(os.path.join(d, n)) for n in names}
    digest = {s: hashlib.sha256("".join(f"{n}:{h}\n" for n, h in v.items()).encode())
              .hexdigest() for s, v in data.items()}
    out = {"read": started,
           "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": files,
                     "pth_sha256_on_card": PTH_SHA256},
           "dataset": {"repo": DATA_REPO, "revision": DATA_REVISION,
                       "counts": {s: len(v) for s, v in data.items()},
                       "digest_of_split": digest, "files": data},
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    write_text(args.out, dumps(out))
    print(f"fetched -> {root}; {out['dataset']['counts']}; {out['elapsed_s']} s")


def dataset_dir(cache):
    from huggingface_hub import snapshot_download
    return snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REVISION,
                             cache_dir=cache, allow_patterns=[f"{s}/*" for s in SPLITS],
                             local_files_only=True)


# --------------------------------------------------------------------------- #
# extract (GPU)
# --------------------------------------------------------------------------- #
def cmd_extract(args):
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from torchvision import transforms

    from rampnet.model import CROP_HEATMAP_SIZE, KeypointModel

    # evaluate.py's preprocess_transform, verbatim.
    pre = transforms.Compose([
        transforms.Resize(INPUT, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    fname, want = CHECKPOINTS[args.checkpoint]
    ckpt = hf_hub_download(MODEL_REPO, fname, revision=MODEL_REVISION, cache_dir=args.cache,
                           local_files_only=True)
    if sha256_file(ckpt) != want:
        raise SystemExit(f"{fname}: sha256 does not match the pin")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = KeypointModel(heatmap_size=CROP_HEATMAP_SIZE)
    model.load_state_dict(load_file(ckpt))
    model = model.to(device).eval()
    head = model.head
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda" else [])

    d = os.path.join(dataset_dir(args.cache), args.split)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(".jpg"))
    if args.limit:
        names = names[:args.limit]
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_run = time.perf_counter()
    gpu_s = 0.0
    crops = []
    for k, n in enumerate(names, 1):
        path = os.path.join(d, n)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        rec = {"crop": n, "sha256": sha256_file(path), "width": w, "height": h,
               "gt": [[round(x, 9), round(y, 9)] for x, y in parse_gt(n, w, h)],
               "coarse": [], "recon_rel": [], "numpy_recon_rel": []}
        t0 = time.perf_counter()
        for branch_img in (img, ImageOps.mirror(img)):
            with torch.no_grad():
                f = model.feature_extractor(pre(branch_img).unsqueeze(0).to(device))
                z = head[1](head[0](f))
                coarse_t = head[3](z)                     # 1x1 conv at 32x11
                h_t = head[3](head[2](z))                 # == model.head(f)
                up = torch.nn.functional.interpolate(coarse_t, size=HM, mode="bilinear",
                                                     align_corners=False)
                scale = max(float(h_t.abs().max()), 1e-6)
                recon = float((h_t - up).abs().max()) / scale
            hh = h_t[0, 0].float().cpu().numpy()
            cc = coarse_t[0, 0].float().cpu().numpy()
            if cc.shape != COARSE or hh.shape != HM:
                raise SystemExit(f"unexpected shapes {cc.shape} {hh.shape}")
            rec["coarse"].append([[round(float(v), COARSE_ND) for v in row] for row in cc])
            rec["recon_rel"].append(float(f"{recon:.3e}"))
            rec["numpy_recon_rel"].append(float(f"{sc.upsample_residual(hh, cc):.3e}"))
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_s += time.perf_counter() - t0
        crops.append(rec)
        if k % 50 == 0:
            print(f"  {args.split}: {k}/{len(names)}", flush=True)
    wall = time.perf_counter() - t_run
    out = {"meta": {"model": MODEL_REPO, "model_revision": MODEL_REVISION,
                    "checkpoint": fname, "checkpoint_sha256": want,
                    "dataset": DATA_REPO, "dataset_revision": DATA_REVISION,
                    "split": args.split, "crops": len(crops),
                    "preprocess": "evaluate.py preprocess_transform: Resize((1024, 352), "
                                  "bilinear), ToTensor, ImageNet Normalize",
                    "branches": ["straight", "mirrored (ImageOps.mirror), coarse map in "
                                 "the mirrored orientation, not flipped back"],
                    "coarse_rounding_decimals": COARSE_ND,
                    "recon_rel": "max|head output - torch bilinear upsample(coarse)| / "
                                 "max|head output|, per branch",
                    "fp16": False, "device": device.type, "gpus": gpus,
                    "torch": torch.__version__, "started": started},
           "crops": crops}
    write_text(args.out, json.dumps(out, separators=(",", ":")) + "\n")
    n = len(crops)
    label = f"crop-decode-221:extract:{args.split}" + (
        "" if args.checkpoint == "round2" else f":{args.checkpoint}")
    row = {"ts": started, "bundle": f"crop-round2-{args.split}", "label": label,
           "crops_scored": n, "forwards": 2 * n, "elapsed_s": round(wall, 3),
           "s_per_crop": round(wall / n, 4) if n else None,
           "gpu_forward_s": round(gpu_s, 3),
           "gpu_hours": round(wall / 3600 * (1 if gpus else 0), 6),
           "what": (f"crop_decode_221.py extract: {args.checkpoint} crop checkpoint, two fp32 "
                    "forwards per crop (straight + mirrored) at 1024x352, 32x11 coarse map "
                    "captured from the head; elapsed_s is wall-clock after model load"),
           "run_id": f"{label}:{started}", "provider": "rampnet", "model_id": MODEL_REPO,
           "model_revision": MODEL_REVISION, "paid": False,
           "hardware": {"host": socket.gethostname(), "gpus": gpus}, "status": "ok",
           "est_cost_usd": 0.0, "pricing": None, "gpu_share": 1.0,
           "script": "scripts/analysis/crop_decode_221.py", "issue": 221}
    if args.usage_out:
        write_text(args.usage_out, json.dumps(row, sort_keys=True) + "\n")
    worst = max(max(c["recon_rel"]) for c in crops) if crops else float("nan")
    print(f"done: {n} crops, wall {wall:.1f}s, forward {gpu_s:.1f}s, "
          f"max recon_rel {worst:.2e} -> {args.out}")


# --------------------------------------------------------------------------- #
# report (CPU)
# --------------------------------------------------------------------------- #
def load_extract(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_UP = {}


def up(c):
    """``sc.upsample`` with its two bilinear matrices built once per shape (same matrices,
    same product order, so the same floats)."""
    h, w = c.shape
    if (h, w) not in _UP:
        _UP[(h, w)] = (sc.bilinear_upsample_matrix(h, h * FACTOR),
                       sc.bilinear_upsample_matrix(w, w * FACTOR).T)
    a, b = _UP[(h, w)]
    return a @ c @ b


def arm_maps(crop, arm):
    """(heatmap to find peaks on, coarse stack oriented like it, per-branch upsamples).

    ``single``: the straight branch's raw head output (peaks found on its clip, as
    ``detect_peaks(clip=True)`` does). ``tta``: evaluate.py's combine -- clip each branch
    to [0, 1], flip the mirrored one back, elementwise max."""
    c0 = np.asarray(crop["coarse"][0], dtype=np.float64)
    h0 = up(c0)
    if arm == "single":
        return h0, c0[None], [h0]
    c1 = np.fliplr(np.asarray(crop["coarse"][1], dtype=np.float64))
    h1 = np.fliplr(up(np.asarray(crop["coarse"][1], dtype=np.float64)))
    combined = np.maximum(np.clip(h0, 0, 1), np.clip(h1, 0, 1))
    return combined, np.stack([c0, c1]), [h0, h1]


def flip_commutes(crop):
    """max |upsample(fliplr(c)) - fliplr(upsample(c))| for the mirrored branch: the
    reverted branch's coarse map is only valid for decoding if this is float noise."""
    c1 = np.asarray(crop["coarse"][1], dtype=np.float64)
    return float(np.abs(up(np.fliplr(c1)) - np.fliplr(up(c1))).max())


def peaks_all(crop, arm, threshold):
    """{method: (N, 3) row, col, score} for every decode; same peaks, same order."""
    hm, stack, _ = arm_maps(crop, arm)
    out = {}
    for m in METHODS:
        out[m] = sc.detect_peaks(hm, threshold, min_distance=MIN_DISTANCE, decode=m,
                                 clip=True, coarse=stack, wrap_x=False)
    base = out["argmax"]
    for m, v in out.items():
        assert np.array_equal(v[:, 2], base[:, 2]), m
    return out


def ordered(peaks):
    """Indices of peaks by descending score, stable (match_predictions' order)."""
    return sorted(range(len(peaks)), key=lambda k: peaks[k][2], reverse=True)


def build_pairs(crops, arm, threshold):
    """Fixed argmax matching per crop. Returns (pairs, peaks cache).

    pairs: list of (crop index, peak index, gt index)."""
    pairs, cache = [], []
    for ci, crop in enumerate(crops):
        pk = peaks_all(crop, arm, threshold)
        cache.append(pk)
        am = pk["argmax"]
        order = ordered(am)
        pred = [(am[k, 1] / HM[1], am[k, 0] / HM[0]) for k in order]
        for k, (g, _) in zip(order, greedy_match(pred, crop["gt"], RADIUS_SQ,
                                                  SCALE_X, SCALE_Y, False)):
            if g >= 0:
                pairs.append((ci, k, g))
    return pairs, cache


#: The training-target x mismatch (both train.py files): keypoints are scaled by 0.5
#: while the image is resized from ``width`` to 352 px, so on the 88-px heatmap a point
#: at true column t gets its target at ``k * t`` with ``k = width / 704`` (0.96875 at
#: width 682). Training also mirrors half the crops (``apply_horizontal_flip=True``) and
#: places the mirrored target at ``351 - 0.5 x`` input px, which in the mirrored view is
#: ``k * t' + 87.75 * (1 - k)``. A model that cannot tell the two views apart learns the
#: average, ``k * t + 87.75 * (1 - k) / 2``: a contraction toward the middle of the crop
#: (zero bias near column 44), not a uniform shift left.
X_FIXES = ("label_scale", "flip_average")


def fix_x(c, width, how):
    k = width / 704.0
    if how == "label_scale":          # undo k only (the no-flip reading)
        return c / k
    if how == "flip_average":         # undo k and the mirrored half's offset
        return (c - 87.75 * (1 - k) / 2) / k
    return c


def residuals(crops, pairs, cache, method, x_fix=None):
    """dx, dy (GT - detection) in heatmap px over pairs. ``x_fix`` (one of ``X_FIXES``)
    first corrects the detection's x for the training-target x mismatch (``fix_x``)."""
    dx, dy = np.empty(len(pairs)), np.empty(len(pairs))
    for n, (ci, k, g) in enumerate(pairs):
        r, c, _ = cache[ci][method][k]
        if x_fix:
            c = fix_x(c, crops[ci]["width"], x_fix)
        gx, gy = crops[ci]["gt"][g]
        dx[n] = gx * HM[1] - c
        dy[n] = gy * HM[0] - r
    return dx, dy


STAT_NAMES = ("mean_px", "mean_abs_x_px", "mean_abs_y_px", "sd_x_px", "sd_y_px")


def _sums(dx, dy):
    """Per-pair sufficient statistics for STAT_NAMES (column 0 is the count)."""
    return np.column_stack([np.ones_like(dx), np.hypot(dx, dy), np.abs(dx), np.abs(dy),
                            dx, dx * dx, dy, dy * dy])


def _from_sums(S):
    """STAT_NAMES from (summed) sufficient statistics; works row-wise on a 2-D array."""
    S = np.atleast_2d(S)
    n = S[:, 0]
    mx, my = S[:, 4] / n, S[:, 6] / n
    sdx = np.sqrt(np.maximum(S[:, 5] / n - mx * mx, 0))
    sdy = np.sqrt(np.maximum(S[:, 7] / n - my * my, 0))
    return np.column_stack([S[:, 1] / n, S[:, 2] / n, S[:, 3] / n, sdx, sdy])


def stats(dx, dy):
    e = np.hypot(dx, dy)
    return {"n": int(len(dx)), "mean_px": e.mean(), "median_px": float(np.median(e)),
            "mean_abs_x_px": np.abs(dx).mean(), "mean_abs_y_px": np.abs(dy).mean(),
            "bias_x_px": dx.mean(), "bias_y_px": dy.mean(),
            "sd_x_px": dx.std(), "sd_y_px": dy.std(),
            "rms_x_px": float(np.sqrt((dx ** 2).mean())),
            "rms_y_px": float(np.sqrt((dy ** 2).mean())),
            "mean_over_sigma": e.mean() / SIGMA}


class Boot:
    """Cluster bootstrap as a (reps x clusters) count matrix (clusters: distinct images).

    Each replicate draws ``G`` clusters with replacement from the ``G`` that hold a
    pair; a replicate statistic is computed from per-cluster sums weighted by those counts,
    which is the same as concatenating the drawn clusters' pairs (a mean or an SD over the
    concatenation depends only on the summed sufficient statistics)."""

    def __init__(self, cluster_ids, rng, n_reps):
        self.groups, self.gidx = np.unique(cluster_ids, return_inverse=True)
        G = len(self.groups)
        self.C = np.zeros((n_reps, G))
        for k in range(n_reps):
            self.C[k] = np.bincount(rng.integers(0, G, G), minlength=G)

    def per_group(self, per_pair):
        out = np.zeros((len(self.groups), per_pair.shape[1]))
        np.add.at(out, self.gidx, per_pair)
        return out

    def replicate(self, per_pair):
        """(reps, k) replicate sums of a (pairs, k) per-pair array."""
        return self.C @ self.per_group(per_pair)


def paired_boot(boot, a, b):
    """Cluster bootstrap of b - a for STAT_NAMES; a, b = (dx, dy)."""
    sa, sb = _sums(*a), _sums(*b)
    obs = _from_sums(sb.sum(0))[0] - _from_sums(sa.sum(0))[0]
    d = _from_sums(boot.replicate(sb)) - _from_sums(boot.replicate(sa))
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {f"d_{nm}": {"obs": float(o), "ci95": [float(l), float(h)]}
            for nm, o, l, h in zip(STAT_NAMES, obs, lo, hi)}


def subcell_fit(crops, pairs, cache, arm, method):
    """OLS slope and Pearson r of the GT's offset from the decoding coarse centre on the
    decoded offset, per axis, in coarse cells."""
    out = {}
    u = {"x": [], "y": []}
    dd = {"x": [], "y": []}
    for ci, k, g in pairs:
        _, stack, ups = arm_maps(crops[ci], arm)
        r, c, _ = cache[ci]["argmax"][k]
        r, c = int(r), int(c)
        b = int(np.argmax([hb[r, c] for hb in ups]))
        i, j = sc.coarse_cell(r, c)
        i, j, _ = sc.climb(stack[b], i, j, False)
        rr, cc, _ = cache[ci][method][k]
        gx, gy = crops[ci]["gt"][g]
        cx, cy = FACTOR * j + (FACTOR - 1) / 2, FACTOR * i + (FACTOR - 1) / 2
        u["x"].append((gx * HM[1] - cx) / FACTOR)
        u["y"].append((gy * HM[0] - cy) / FACTOR)
        dd["x"].append((cc - cx) / FACTOR)
        dd["y"].append((rr - cy) / FACTOR)
    for ax in ("x", "y"):
        uu, d = np.array(u[ax]), np.array(dd[ax])
        if len(d) < 3 or d.std() < 1e-9:
            out[ax] = {"slope": None, "pearson_r": None, "sd_decoded_cells": float(d.std())
                       if len(d) else None}
            continue
        out[ax] = {"slope": float(np.cov(d, uu)[0, 1] / d.var(ddof=1)),
                   "pearson_r": float(np.corrcoef(d, uu)[0, 1]),
                   "sd_decoded_cells": float(d.std())}
    return out


def mod8(vals):
    return np.bincount(np.floor(np.asarray(vals) + 1e-9).astype(int) % FACTOR,
                       minlength=FACTOR).tolist()


def _ols_boot(boot, x, y):
    """OLS slope and intercept of y on x, with 95% bootstrap CIs (distinct images)."""
    slope, icpt = np.polyfit(x, y, 1)
    R = boot.replicate(np.column_stack([np.ones_like(x), x, y, x * x, x * y]))
    n, sx, sy, sxx, sxy = R.T
    b = (sxy - sx * sy / n) / (sxx - sx * sx / n)
    a = (sy - b * sx) / n
    return {"slope": float(slope),
            "slope_ci95": [float(v) for v in np.percentile(b, [2.5, 97.5])],
            "intercept_px": float(icpt),
            "intercept_ci95": [float(v) for v in np.percentile(a, [2.5, 97.5])]}


#: What the training-target geometry predicts for dx = GT - det against the true column t
#: at width 682. Both rounds train with ``apply_horizontal_flip=True`` (round 1:
#: ``ps_model/model/train.py`` L118; round 2: ``ps_and_manual_model/train.py``), so both
#: predict the flip average of ``X_FIXES``: dx = (1 - k) t - 87.75 (1 - k) / 2, k = 682/704.
_K682 = 682 / 704
PREDICTED_X_BIAS = {"slope": 1 - _K682, "intercept_px": -87.75 * (1 - _K682) / 2}


def x_bias(crops, pairs, cache, method, boot):
    """dx (GT - det, heatmap px) regressed on x.

    ``on_gt_x`` is the estimate to read: the regressor is the label's own column, measured
    without the detection's error. ``on_det_x`` is kept as a comparison only. Its regressor
    carries the detection error e = det - GT, which also sits (negated) in dx, so its slope
    is pulled down by about var(e) / var(det x) (errors in variables); ``dilution`` reports
    that ratio. Thirds are by GT x."""
    dx, _ = residuals(crops, pairs, cache, method)
    if len(dx) < 3:
        return None
    xg = np.array([crops[ci]["gt"][g][0] * HM[1] for ci, _, g in pairs])
    xd = np.array([cache[ci][method][k][1] for ci, k, _ in pairs])
    thirds = [dx[(xg >= HM[1] * t / 3) & (xg < HM[1] * (t + 1) / 3)] for t in range(3)]
    return {"on_gt_x": _ols_boot(boot, xg, dx), "on_det_x": _ols_boot(boot, xd, dx),
            "dilution": float(np.var(xd - xg) / np.var(xd)),
            "mean_dx_by_gt_third": [float(t.mean()) if len(t) else None for t in thirds],
            "n_by_gt_third": [int(len(t)) for t in thirds]}


def detection_flags(crops, cache0, method):
    """Per-peak (score, is_tp) lists per crop under evaluate.py's matching, threshold 0.0."""
    out = []
    for ci, crop in enumerate(crops):
        pk = cache0[ci][method]
        peaks = [(c / HM[1], r / HM[0], s) for r, c, s in pk]
        out.append(match_predictions(peaks, crop["gt"], RADIUS_SQ, SCALE_X, SCALE_Y,
                                     wrap_x=False))
    return out


def detection_metrics(crops, flags):
    """evaluate.py's protocol at threshold 0.0: AP; plus P/R/F1 at 0.30 and 0.55."""
    preds = [p for f in flags for p in f]
    n_gt = sum(len(c["gt"]) for c in crops)
    ap = calculate_ap_and_pr_curve(preds, n_gt)[0]
    out = {"ap_at_0.0": ap, "n_gt": n_gt, "n_pred_at_0.0": len(preds)}
    for t in (0.30, 0.55):
        sel = [tp for s, tp in preds if s >= t]
        tp = sum(sel)
        p = tp / len(sel) if sel else 0.0
        r = tp / n_gt if n_gt else 0.0
        out[f"at_{t:.2f}"] = {"tp": tp, "fp": len(sel) - tp, "precision": p, "recall": r,
                              "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out


def tp_flips(crops, cache0, fa, fg):
    """Peaks whose TP flag differs between argmax and gaussian, at 0.30 and 0.55.

    Peaks, scores and match order are identical under both decodes (only positions move),
    so the flags pair up one to one. ``argmax_units_*`` is the range of the flipped peaks'
    argmax distance to their nearest label, in the evaluator's units (radius 11.25)."""
    out = {}
    for t in (0.30, 0.55):
        gained, lost, units = 0, 0, []
        for ci, (a, g) in enumerate(zip(fa, fg)):
            am = cache0[ci]["argmax"]
            for k, (sa, ta), (sg, tg) in zip(ordered(am), a, g):
                assert sa == sg
                if sa < t or ta == tg:
                    continue
                gained += int(tg)
                lost += int(ta)
                r, c, _ = am[k]
                units.append(min(math.hypot((c / HM[1] - gx) * SCALE_X,
                                            (r / HM[0] - gy) * SCALE_Y)
                                 for gx, gy in crops[ci]["gt"]))
        out[f"at_{t:.2f}"] = {"gained": gained, "lost": lost,
                              "argmax_units_min": min(units) if units else None,
                              "argmax_units_max": max(units) if units else None}
    return out


def unique_image_indices(crops):
    """One crop per distinct image (sha256), the first by filename.

    The round-2 splits store a multi-point crop once per point, with the points permuted
    in the name, so several files can hold the same bytes and the same point set."""
    first = {}
    for ci in sorted(range(len(crops)), key=lambda i: crops[i]["crop"]):
        first.setdefault(crops[ci]["sha256"], ci)
    return sorted(first.values())


def analyse(crops, arm, thresholds, n_reps, seed, full=True):
    """One arm on one crop list. The bootstrap resamples distinct images (sha256), so
    duplicate files of one image move together."""
    cache_by_thr = {}
    a_res = {"thresholds": {}}
    order = [PAIR_THRESHOLD] + [t for t in THRESHOLDS if t != PAIR_THRESHOLD]
    for thr in order:
        if thr not in thresholds and thr != 0.0:
            continue
        pairs, cache = build_pairs(crops, arm, thr)
        cache_by_thr[thr] = cache
        if thr not in thresholds:
            continue
        cluster = np.array([crops[ci]["sha256"] for ci, _, _ in pairs])
        rng = np.random.default_rng(seed)
        boot = Boot(cluster, rng, n_reps) if len(pairs) else None
        base = residuals(crops, pairs, cache, "argmax")
        t_res = {"n_pairs": len(pairs),
                 "n_crops_with_pairs": len({ci for ci, _, _ in pairs}),
                 "n_images_with_pairs": int(len(np.unique(cluster))),
                 "n_peaks": int(sum(len(c["argmax"]) for c in cache)),
                 "stats": {}, "paired_vs_argmax": {}}
        for m in (METHODS if full else ("argmax", "gaussian")):
            r = residuals(crops, pairs, cache, m)
            t_res["stats"][m] = stats(*r) if len(pairs) else None
            if m != "argmax" and len(pairs):
                t_res["paired_vs_argmax"][m] = paired_boot(boot, base, r)
        if thr == PAIR_THRESHOLD and len(pairs):
            t_res["x_bias"] = {m: x_bias(crops, pairs, cache, m, boot)
                               for m in ("argmax", "gaussian")}
            if full:
                t_res["x_fixed"] = {}
                for how in X_FIXES:
                    fa = residuals(crops, pairs, cache, "argmax", x_fix=how)
                    fg = residuals(crops, pairs, cache, "gaussian", x_fix=how)
                    t_res["x_fixed"][how] = {
                        "argmax": stats(*fa), "gaussian": stats(*fg),
                        "paired_gaussian_vs_argmax": paired_boot(boot, fa, fg),
                        "paired_vs_uncorrected": {
                            m: paired_boot(boot, residuals(crops, pairs, cache, m), f)
                            for m, f in (("argmax", fa), ("gaussian", fg))}}
                t_res["subcell_fit"] = {m: subcell_fit(crops, pairs, cache, arm, m)
                                        for m in ("centre", "quarter", "parabola",
                                                  "gaussian", "dark", "centroid")}
                gt_x = [crops[ci]["gt"][g][0] * HM[1] for ci, _, g in pairs]
                gt_y = [crops[ci]["gt"][g][1] * HM[0] for ci, _, g in pairs]
                t_res["mod8"] = {
                    "gt": {"x": mod8(gt_x), "y": mod8(gt_y)},
                    **{m: {"x": mod8([cache[ci][m][k][1] for ci, k, _ in pairs]),
                           "y": mod8([cache[ci][m][k][0] for ci, k, _ in pairs])}
                       for m in ("argmax", "gaussian")}}
        a_res["thresholds"][f"{thr:.2f}"] = t_res
    fl = {m: detection_flags(crops, cache_by_thr[0.0], m) for m in ("argmax", "gaussian")}
    a_res["detection"] = {m: detection_metrics(crops, f) for m, f in fl.items()}
    a_res["tp_flips_gaussian_vs_argmax"] = tp_flips(crops, cache_by_thr[0.0],
                                                    fl["argmax"], fl["gaussian"])
    return a_res


def report(ex, n_reps=N_REPS, seed=SEED):
    crops = ex["crops"]
    meta = ex["meta"]
    if not crops:
        raise ValueError("extract holds no crops")
    dedup = [crops[i] for i in unique_image_indices(crops)]
    res = {"meta": {"extract_meta": {k: meta[k] for k in
                                     ("model", "model_revision", "checkpoint",
                                      "checkpoint_sha256", "dataset", "dataset_revision",
                                      "split", "crops")},
                    "protocol": {"pair_threshold": PAIR_THRESHOLD, "min_distance": MIN_DISTANCE,
                                 "radius_norm": RADIUS_NORM, "scale_x": SCALE_X,
                                 "scale_y": SCALE_Y, "radius_units": RADIUS_NORM * SCALE_X,
                                 "wrap_x": False, "reps": n_reps, "seed": seed,
                                 "bootstrap_unit": "distinct image (sha256)",
                                 "sigma_px": SIGMA, "residual_units": "heatmap px (256x88)",
                                 "predicted_x_bias": PREDICTED_X_BIAS},
                    "n_crops": len(crops),
                    "n_gt": sum(len(c["gt"]) for c in crops),
                    "n_unique_images": len(dedup),
                    "n_unique_points": sum(len(c["gt"]) for c in dedup),
                    "widths": sorted({c["width"] for c in crops}),
                    "heights": sorted({c["height"] for c in crops})},
           # strings, so the ND rounding applied to every other float does not zero them
           "mechanism": {
               "max_recon_rel_torch": f'{max(max(c["recon_rel"]) for c in crops):.2e}',
               "max_recon_rel_numpy": f'{max(max(c["numpy_recon_rel"]) for c in crops):.2e}',
               "max_flip_commute_abs": f"{max(flip_commutes(c) for c in crops):.2e}"},
           "arms": {}, "dedup": {}}
    for arm in ARMS:
        res["arms"][arm] = analyse(crops, arm, THRESHOLDS, n_reps, seed, full=True)
        res["dedup"][arm] = analyse(dedup, arm, (PAIR_THRESHOLD,), n_reps, seed, full=False)
    return rnd(res)


def fmt_ci(d):
    if not d:
        return "n/a"
    return f"{d['obs']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def fv(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def _fmt_ols(o):
    return (f"slope {o['slope']:+.4f} [{o['slope_ci95'][0]:+.4f}, {o['slope_ci95'][1]:+.4f}], "
            f"intercept {o['intercept_px']:+.3f} [{o['intercept_ci95'][0]:+.3f}, "
            f"{o['intercept_ci95'][1]:+.3f}]")


def _stats_table(L, t):
    L.append("| decode | mean px | mean abs x | mean abs y | bias x | bias y | "
             "sd x | sd y | change in mean px [95% CI] | change in sd x | "
             "change in sd y |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for meth, s in t["stats"].items():
        if s is None:
            L.append(f"| {meth} | n/a | | | | | | | | | |")
            continue
        p = t["paired_vs_argmax"].get(meth)
        extra = ("| | | |" if p is None else
                 f"| {fmt_ci(p['d_mean_px'])} | {fmt_ci(p['d_sd_x_px'])} | "
                 f"{fmt_ci(p['d_sd_y_px'])} |")
        L.append(f"| {meth} | {s['mean_px']:.3f} | {s['mean_abs_x_px']:.3f} | "
                 f"{s['mean_abs_y_px']:.3f} | {s['bias_x_px']:+.3f} | "
                 f"{s['bias_y_px']:+.3f} | {s['sd_x_px']:.3f} | {s['sd_y_px']:.3f} "
                 + extra)
    L.append("")


def _x_bias_lines(L, t):
    L.append("x bias (dx = GT - det, heatmap px, regressed on x; predicted from the training "
             "geometry at width 682: slope +0.0312, intercept -1.371):")
    L.append("")
    for meth, xb in t["x_bias"].items():
        if not xb:
            continue
        L.append(f"- {meth}, on GT x: {_fmt_ols(xb['on_gt_x'])}; mean dx by GT-x third "
                 f"{xb['mean_dx_by_gt_third']} (n {xb['n_by_gt_third']})")
        L.append(f"- {meth}, on detection x (comparison only, errors in variables; "
                 f"var(e)/var(det x) = {xb['dilution']:.4f}): {_fmt_ols(xb['on_det_x'])}")
    L.append("")


def _detection_table(L, a):
    L.append("Detection metrics (evaluate.py protocol: radius 0.132, threshold 0.0 for AP; "
             "every file counted, so a duplicated image counts once per copy):")
    L.append("")
    L.append("| decode | AP | preds | TP@0.30 | FP@0.30 | R@0.30 | F1@0.30 | TP@0.55 | "
             "FP@0.55 | F1@0.55 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for meth, dm in a["detection"].items():
        a3, a5 = dm["at_0.30"], dm["at_0.55"]
        L.append(f"| {meth} | {dm['ap_at_0.0']:.4f} | {dm['n_pred_at_0.0']} | {a3['tp']} | "
                 f"{a3['fp']} | {a3['recall']:.4f} | {a3['f1']:.4f} | {a5['tp']} | "
                 f"{a5['fp']} | {a5['f1']:.4f} |")
    L.append("")
    for t, fl in a["tp_flips_gaussian_vs_argmax"].items():
        L.append(f"- TP flips gaussian vs argmax {t.replace('at_', 'at ')}: "
                 f"+{fl['gained']} gained, -{fl['lost']} lost (argmax distance of the flipped "
                 f"peaks {fv(fl['argmax_units_min'], '.2f')} to "
                 f"{fv(fl['argmax_units_max'], '.2f')} units; radius 11.25)")
    L.append("")


def render_md(res, title):
    m = res["meta"]
    em = m["extract_meta"]
    L = [f"# {title}", "",
         "Generated by `scripts/analysis/crop_decode_221.py report`; do not edit by hand.",
         "",
         f"- checkpoint `{em['checkpoint']}` @ `{em['model_revision'][:8]}`; data "
         f"`{em['dataset']}` @ `{em['dataset_revision'][:8]}`, split `{em['split']}`",
         f"- {m['n_crops']} files with {m['n_gt']} GT points; {m['n_unique_images']} distinct "
         f"images (sha256) with {m['n_unique_points']} distinct points; widths {m['widths']}, "
         f"heights {m['heights']}",
         f"- mechanism: max relative |head - upsample(coarse)| torch "
         f"{res['mechanism']['max_recon_rel_torch']}, numpy "
         f"{res['mechanism']['max_recon_rel_numpy']}; flip/upsample commute "
         f"{res['mechanism']['max_flip_commute_abs']}",
         "- residual = GT - detection, heatmap px on the 256x88 grid; CIs are 95% bootstrap "
         "of the paired change vs argmax, resampling distinct images (sha256)", ""]
    for arm, a in res["arms"].items():
        L.append(f"## Arm: {arm} (all files)")
        L.append("")
        for thr, t in a["thresholds"].items():
            L.append(f"### Pairs at threshold {thr}: {t['n_pairs']} pairs in "
                     f"{t['n_crops_with_pairs']} files / {t['n_images_with_pairs']} images "
                     f"({t['n_peaks']} peaks)")
            L.append("")
            _stats_table(L, t)
            if "x_bias" in t:
                _x_bias_lines(L, t)
            if "x_fixed" in t:
                for how, xf in t["x_fixed"].items():
                    pu = xf["paired_vs_uncorrected"]
                    L.append(f"- x corrected ({how}): argmax mean {xf['argmax']['mean_px']:.3f} "
                             f"(bias x {xf['argmax']['bias_x_px']:+.3f}; vs uncorrected "
                             f"{fmt_ci(pu['argmax']['d_mean_px'])}), gaussian "
                             f"{xf['gaussian']['mean_px']:.3f} (bias x "
                             f"{xf['gaussian']['bias_x_px']:+.3f}; vs uncorrected "
                             f"{fmt_ci(pu['gaussian']['d_mean_px'])}); gaussian vs argmax "
                             f"{fmt_ci(xf['paired_gaussian_vs_argmax']['d_mean_px'])}")
                L.append("")
            if "subcell_fit" in t:
                L.append("Sub-cell fit (GT offset from the coarse centre on decoded offset, "
                         "cells):")
                L.append("")
                for meth, f in t["subcell_fit"].items():
                    L.append(f"- {meth}: x slope {f['x']['slope']}, r {f['x']['pearson_r']}; "
                             f"y slope {f['y']['slope']}, r {f['y']['pearson_r']}")
                L.append("")
                L.append("mod-8 histograms of position (floor(px) % 8):")
                L.append("")
                for k, v in t["mod8"].items():
                    L.append(f"- {k}: x {v['x']}, y {v['y']}")
                L.append("")
        _detection_table(L, a)
    L.append(f"## Deduplicated: one file per distinct image ({m['n_unique_images']} images, "
             f"{m['n_unique_points']} points)")
    L.append("")
    for arm, a in res["dedup"].items():
        t = a["thresholds"][f"{PAIR_THRESHOLD:.2f}"]
        L.append(f"### Arm: {arm}, pairs at {PAIR_THRESHOLD:.2f}: {t['n_pairs']} pairs in "
                 f"{t['n_images_with_pairs']} images")
        L.append("")
        _stats_table(L, t)
        if "x_bias" in t:
            _x_bias_lines(L, t)
        _detection_table(L, a)
    return "\n".join(L) + "\n"


def run_report(extract_path, n_reps=N_REPS, seed=SEED):
    ex = load_extract(extract_path)
    res = report(ex, n_reps, seed)
    title = (f"Crop-model decode (#221): {ex['meta']['split']} split, "
             f"{ex['meta']['checkpoint'].split('_best')[0]}")
    return dumps(res), render_md(res, title)


def cmd_report(args):
    js, md = run_report(args.extract, args.reps, args.seed)
    write_text(args.out, js)
    if args.md:
        write_text(args.md, md)
    print(f"wrote {args.out}" + (f" and {args.md}" if args.md else ""))


def cmd_check(out_dir=OUT_DIR, only=None):
    """Re-derive committed results from committed extracts; ``only`` limits it to the
    listed extract file names (the test suite checks one, the CLI all three)."""
    bad = []
    n = 0
    todo = [c for c in COMMITTED if only is None or c[0] in only]
    for ex_name, stem in todo:
        ex_path = os.path.join(out_dir, ex_name)
        if not os.path.exists(ex_path):
            bad.append(f"missing {ex_name}")
            continue
        js, md = run_report(ex_path)
        for text, ext in ((js, ".json"), (md, ".md")):
            p = os.path.join(out_dir, stem + ext)
            if not os.path.exists(p):
                bad.append(f"missing {stem}{ext}")
                continue
            with open(p, encoding="utf-8", newline="") as f:
                if f.read() != text:
                    bad.append(f"{stem}{ext} differs")
            n += 1
    if bad:
        print("CHECK FAILED: " + "; ".join(bad))
        return 1
    print(f"check ok: {n} files re-derived byte for byte from {len(todo)} extracts")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="re-derive every committed results file and fail on drift")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch")
    f.add_argument("--cache", default=os.path.join(REPO, ".hf_cache"))
    f.add_argument("--out", default=os.path.join(OUT_DIR, "inputs.json"))
    e = sub.add_parser("extract")
    e.add_argument("--split", choices=SPLITS, required=True)
    e.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="round2")
    e.add_argument("--cache", default=os.path.join(REPO, ".hf_cache"))
    e.add_argument("--out", required=True)
    e.add_argument("--usage-out")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--cpu", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--extract", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--md")
    r.add_argument("--reps", type=int, default=N_REPS)
    r.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.check:
        return cmd_check()
    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
