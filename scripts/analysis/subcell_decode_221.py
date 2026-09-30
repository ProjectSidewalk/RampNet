"""Sub-cell decode of RampNet peaks, measured against independent box centres (#221).

The pano head is ``Conv3x3 -> ReLU -> Upsample(x8 bilinear) -> Conv1x1``, so the
512x1024 heatmap is a bilinear upsample of a 64x128 map and every ``peak_local_max``
peak sits on an 8-px grid. ``rampnet/subcell.py`` refines each peak from the 3x3 coarse
neighbourhood. This script measures whether that refinement moves detections toward
human-drawn box centres.

Three subcommands:

    # GPU (makelab2 A40, shared: 1.2 s/pano forward, 1.72 s/pano wall-clock including
    # single-threaded JPEG decode of the native panos). Runs the released checkpoint,
    # pinned by --model-revision, once per pano,
    # captures the pre-upsample 64x128 map from the head, checks the head output is its
    # bilinear upsample, extracts peaks exactly as analysis_out/op_cache does
    # (threshold_sweep.peaks_to_dets: clip [0,1], min_distance 10, exclude_border=False)
    # at a 0.30 floor, and writes each peak with its 3x3 coarse neighbourhood to
    # analysis_out/subcell_decode_221/detections.json. Usage rows go to --usage-out
    # (appended to analysis_out/usage_log.jsonl by hand in the main checkout, because
    # the GPU host's clone is not the ledger's home).
    # The committed detections.json was made with (from /homes/gws/jonf/wt-subcell221
    # on makelab2, panos from Jon's makelab checkout, model at Hub main = MODEL_REVISION):
    python scripts/analysis/subcell_decode_221.py extract \
        --panos-root /homes/gws/jonf/RampNet --cache-dir /homes/gws/jonf/subcell221_cache

    # CPU: sha256 every pano the extraction reads against the committed
    # benchmark/<split>/imagery_manifest.json, and write imagery_check.json. extract also
    # runs this and records the result in detections.json's meta (runs after 2026-09-30).
    python scripts/analysis/subcell_decode_221.py verify-imagery \
        --panos-root /homes/gws/jonf/RampNet

    # CPU, no model, no images: decode every committed neighbourhood with
    # rampnet.subcell, match to GT, and write results.json + results.md.
    python scripts/analysis/subcell_decode_221.py report

Ground truth:

- ``manual_gold``: centres of the YOLO boxes in ``manual_labels/`` -- labelled without
  any model, so not anchored to the model's grid. The primary read.
- ``annapolis``, ``paterson``, ``richmond``, ``sao_paulo``: centres of the reviewer boxes
  in ``benchmark/<split>/boxes.json`` (status ``boxed``). Those boxes were drawn in a
  crop around a *shown* detection point (``det:k``) or a reviewer-placed miss point
  (``missed:k``), so they are human extents but not blind to the model. ``cant``
  entries (extent undeterminable) are kept as match decoys and dropped from residuals.

Pairs are fixed once, on the argmax positions: detections at >= 0.30 (the #79
recommended operating point), greedy by confidence, radius 0.022 (the benchmark's), x
wrapped. Every decode is then scored on the same pairs, so the comparison is paired.
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import subcell as sc  # noqa: E402
from rampnet.detection_eval import load_yolo_ground_truths, radius_sq_for  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "subcell_decode_221")
DETS = os.path.join(OUT_DIR, "detections.json")
OP_CACHE = os.path.join(REPO, "analysis_out", "op_cache")
BOX_SPLITS = ("annapolis", "paterson", "richmond", "sao_paulo")
SPLITS = ("manual_gold",) + BOX_SPLITS
HM = (512, 1024)
COARSE = (64, 128)
FLOOR = 0.30            # the #79 recommended operating point; also the score floor stored
MIN_DISTANCE = 10
N_REPS = 2000
SEED = 221
ND = 4
DEG_PER_PX = 360.0 / HM[1]        # 0.3516; the same in y (180 / 512)
METHODS = [m for m in sc.METHODS]
CHECK_TOL = 2e-4                  # input_res_sweep_25.CHECK_TOL: cross-machine fp32 noise
MODEL_REPO = "projectsidewalk/rampnet-model"
#: Hub commit of MODEL_REPO the committed detections.json was extracted with. Hub ``main``
#: has pointed here since 2026-07-24 (``model_info().last_modified``), and makelab2's HF
#: cache ``refs/main`` held it on the run date; model.safetensors sha256 is
#: MODEL_WEIGHTS_SHA256.
MODEL_REVISION = "606a11956743f7eb328d9207769034752f6191f4"
MODEL_WEIGHTS_SHA256 = "f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5"
IMAGERY_CHECK = os.path.join(OUT_DIR, "imagery_check.json")
#: Row bands (argmax row, hi-res px, half-open) for the y-profile read of section 4.3.
#: Row 256 is the horizon; 256-290 is the far-field band just below it.
Y_BANDS = (256, 290, 330)
#: Uniform-quantization variance models, px^2 per axis, for an 8-px cell (S2 of the #226
#: review). Snapping to the coarse centre leaves u ~ U[-4, 4]: 64/12. The argmax already
#: leans 0.5 px toward the true side, a = 0.5 sign(u): E[(u - a)^2] = 64/12 - 2(0.5)E|u|
#: + 0.25 with E|u| = 2, i.e. 64/12 - 1.75.
QVAR_CENTRE = 64.0 / 12.0
QVAR_ARGMAX = QVAR_CENTRE - 1.75


def rnd(v, nd=ND):
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    if isinstance(v, (np.floating,)):
        return rnd(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def split_pano_ids(split):
    """Every pano in the bundle (the extraction set)."""
    ids = []
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["pano"]["panorama_id"])
    return ids


def ground_truth(split):
    """``{pid: (points, is_scored)}``: GT points in normalized (x, y) and, per point,
    whether its residual is scored (False for a ``cant`` decoy)."""
    if split == "manual_gold":
        gts = load_yolo_ground_truths(os.path.join(REPO, "manual_labels"))
        ids = set(split_pano_ids(split))
        return {p: (list(g.gt_points), [True] * len(g.gt_points))
                for p, g in gts.items() if p in ids}
    with open(os.path.join(REPO, "benchmark", split, "boxes.json"), encoding="utf-8") as f:
        panos = json.load(f)["panos"]
    out = {}
    for pid, entries in panos.items():
        pts, scored = [], []
        for _, e in sorted(entries.items()):
            if e.get("status") == "boxed":
                pts.append((e["cx"], e["cy"]))
                scored.append(True)
            else:
                pts.append((e["point"]["x"], e["point"]["y"]))
                scored.append(False)
        out[pid] = (pts, scored)
    return out


# --------------------------------------------------------------------------- #
# extract (GPU)
# --------------------------------------------------------------------------- #
def peaks_from(h):
    """(row, col) peaks exactly as threshold_sweep.peaks_to_dets finds them."""
    from skimage.feature import peak_local_max
    return peak_local_max(np.clip(h, 0, 1), min_distance=MIN_DISTANCE, threshold_abs=FLOOR,
                          exclude_border=False)


def pano_record(h, coarse):
    """Per-pano record: peaks with their coarse cell and 3x3 neighbourhood (x wrapped;
    y off-map is null)."""
    dets = []
    for r, c in peaks_from(h):
        i, j = sc.coarse_cell(r, c)
        i, j, steps = sc.climb(coarse, i, j)
        nb = sc.neighbourhood(coarse, i, j, wrap_x=True)
        dets.append([int(r), int(c), rnd(float(h[r, c]), 6), int(i), int(j), int(steps),
                     [rnd(float(v), 6) if np.isfinite(v) else None for v in nb.ravel()]])
    dets.sort(key=lambda d: -d[2])
    return dets


def verify_imagery(panos_root, splits, limit=0):
    """sha256 of every pano ``extract`` reads, against ``benchmark/<split>/imagery_manifest.json``.

    Checks exactly the ids ``extract`` iterates (``records.jsonl``), at
    ``<panos_root>/benchmark/<split>/panos/<id>.jpg``, against the manifest committed in
    *this* checkout (not the one under ``panos_root``, which may be older). Returns a
    JSON-able dict with a per-split status; ``ok`` needs every pano present and equal.
    """
    import imagery_manifest as im
    out = {"panos_root": panos_root, "splits": {}}
    for split in splits:
        pids = split_pano_ids(split)[:limit or None]
        man = im.load(split)
        if man is None:
            out["splits"][split] = {"status": "NO MANIFEST", "panos": len(pids)}
            continue
        pdir = os.path.join(panos_root, "benchmark", split, "panos")
        got, missing, changed, unlisted = {}, [], [], []
        for pid in pids:
            path = os.path.join(pdir, f"{pid}.jpg")
            if not os.path.exists(path):
                missing.append(pid)
                continue
            got[pid] = {"sha256": sha256_file(path)}
            want = man["panos"].get(pid)
            if want is None:
                unlisted.append(pid)
            elif want["sha256"] != got[pid]["sha256"]:
                changed.append(pid)
        ok = not (missing or changed or unlisted)
        out["splits"][split] = {
            "status": "ok" if ok else "MISMATCH", "panos": len(pids), "hashed": len(got),
            "match": len(got) - len(changed) - len(unlisted), "missing": missing,
            "changed": changed, "not_in_manifest": unlisted,
            "manifest_digest": man["digest"], "manifest_n": man["n"],
            "digest_of_panos_read": im.digest_of(got)}
    out["status"] = ("ok" if all(s["status"] == "ok" for s in out["splits"].values())
                     else "MISMATCH")
    return out


def cmd_verify_imagery(args):
    splits = [s for s in args.splits.split(",") if s]
    rep = verify_imagery(args.panos_root, splits, args.limit)
    rep["checked"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rep["host"] = socket.getfqdn()
    write_json(args.out, rep)
    for s, r in rep["splits"].items():
        print(f"{s:>12}: {r['status']} ({r.get('match', 0)}/{r['panos']} match)")
    print(f"-> {args.out}")
    return 0 if rep["status"] == "ok" else 1


def load_model_at(revision):
    """``threshold_sweep.load_model``, pinned to a Hub commit. Returns the model, the
    resolved commit and the sha256 of the weights file actually loaded."""
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download
    from rampnet.model import KeypointModel
    path = hf_hub_download(MODEL_REPO, "model.safetensors", revision=revision)
    sd = st.load_file(path)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    m = KeypointModel()
    m.load_state_dict(sd)
    # the HF cache stores files at .../snapshots/<commit>/<file>
    commit = os.path.basename(os.path.dirname(path))
    return m.eval(), commit, sha256_file(path)


def cmd_extract(args):
    import torch
    from PIL import Image
    import threshold_sweep as ts
    Image.MAX_IMAGE_PIXELS = None

    splits_req = [s for s in args.splits.split(",") if s]
    imagery = verify_imagery(args.panos_root, splits_req, args.limit)
    if imagery["status"] != "ok" and not args.allow_imagery_mismatch:
        raise SystemExit(f"imagery does not match the committed manifests: "
                         f"{ {s: r['status'] for s, r in imagery['splits'].items()} } "
                         f"(--allow-imagery-mismatch to run anyway)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_commit, weights_sha = load_model_at(args.model_revision)
    model = model.to(device).eval()
    head = model.head
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda" else [])
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_run = time.perf_counter()
    splits = splits_req
    out = {"meta": {"model": MODEL_REPO, "model_revision": model_commit,
                    "model_weights_sha256": weights_sha,
                    "imagery_check": {s: {k: r.get(k) for k in
                                          ("status", "panos", "match", "digest_of_panos_read")}
                                      for s, r in imagery["splits"].items()},
                    "floor": FLOOR,
                    "min_distance": MIN_DISTANCE, "heatmap": list(HM), "coarse": list(COARSE),
                    "preprocess": "threshold_sweep.PRE (resize to 2048x4096 bilinear)",
                    "fp16": False, "tta": False, "device": device.type, "gpus": gpus,
                    "torch": torch.__version__, "started": started,
                    "det_fields": ["row", "col", "score", "coarse_i", "coarse_j",
                                   "climb_steps", "nb3x3_rowmajor_xwrapped"]},
           "panos": {}}
    gpu_s, cpu_s, n = 0.0, 0.0, 0
    per_split = {}
    for split in splits:
        pids = split_pano_ids(split)
        if args.limit:
            pids = pids[:args.limit]
        pdir = os.path.join(args.panos_root, "benchmark", split, "panos")
        cdir = os.path.join(args.cache_dir, split)
        os.makedirs(cdir, exist_ok=True)
        recs, t_split = {}, time.perf_counter()
        for k, pid in enumerate(pids, 1):
            t0 = time.perf_counter()
            t = ts.PRE(Image.open(os.path.join(pdir, f"{pid}.jpg")).convert("RGB"))
            t1 = time.perf_counter()
            with torch.no_grad():
                f = model.feature_extractor(t.unsqueeze(0).to(device))
                z = head[1](head[0](f))
                coarse_t = head[3](z)                       # 1x1 conv at 64x128
                h_t = head[3](head[2](z))                   # == model.head(f)
                up_t = torch.nn.functional.interpolate(coarse_t, size=HM, mode="bilinear",
                                                       align_corners=False)
                recon = float((h_t - up_t).abs().max())
            h = h_t[0, 0].float().cpu().numpy()
            coarse = coarse_t[0, 0].float().cpu().numpy()
            t2 = time.perf_counter()
            np.save(os.path.join(cdir, f"{pid}_coarse.npy"), coarse.astype(np.float32))
            rec = {"coarse_sha256": hashlib.sha256(coarse.astype(np.float32).tobytes())
                   .hexdigest(),
                   "recon_max_abs": rnd(recon, 9),
                   "numpy_recon_max_abs": rnd(float(np.abs(sc.upsample(coarse) - h).max()), 9),
                   "dets": pano_record(h, coarse)}
            recs[pid] = rec
            gpu_s += t2 - t1
            cpu_s += t1 - t0
            n += 1
            if k % 100 == 0:
                print(f"  {split}: {k}/{len(pids)}", flush=True)
        per_split[split] = {"panos": len(pids),
                            "elapsed_s": round(time.perf_counter() - t_split, 3)}
        out["panos"][split] = recs
        print(f"{split}: {len(pids)} panos, {sum(len(r['dets']) for r in recs.values())} "
              f"peaks >= {FLOOR}", flush=True)
    wall = time.perf_counter() - t_run
    write_json(args.out, out)
    row = {"ts": started, "bundle": ",".join(splits), "label": "subcell-decode-221:extract",
           "panos_scored": n, "elapsed_s": round(wall, 3),
           "s_per_pano": round(wall / n, 4) if n else None,
           "gpu_forward_s": round(gpu_s, 3), "cpu_decode_s": round(cpu_s, 3),
           "per_split": per_split,
           "what": ("subcell_decode_221.py extract: one fp32 forward per pano at 2048x4096, "
                    "coarse map captured from the head, peaks >= 0.30; elapsed_s is the "
                    "run's wall-clock after model load (JPEG decode not overlapped)"),
           "run_id": f"subcell-decode-221:extract:{started}",
           "provider": "rampnet", "model_id": MODEL_REPO, "model_revision": model_commit,
           "paid": False,
           "hardware": {"host": socket.getfqdn(), "gpus": gpus}, "status": "ok",
           "est_cost_usd": 0.0, "pricing": None, "gpu_share": 1.0,
           "script": "scripts/analysis/subcell_decode_221.py", "issue": 221}
    if args.note:
        row["note"] = args.note
    write_json(args.usage_out, row)
    print(f"done: {n} panos, wall {wall:.0f}s, forward {gpu_s:.0f}s -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #
# report (CPU)
# --------------------------------------------------------------------------- #
def decode(det, method, wrap_x):
    """Normalized (x, y) of one stored detection under ``method``."""
    r, c, _, i, j, _, nb = det
    if method == "argmax":
        return c / HM[1], r / HM[0]
    n = np.array([np.nan if v is None else v for v in nb], dtype=float).reshape(3, 3)
    if not wrap_x and j in (0, COARSE[1] - 1):
        n[:, 0 if j == 0 else 2] = np.nan
    dy, dx = sc.refine_offset(n, method)
    x = (sc.FACTOR * (j + dx) + (sc.FACTOR - 1) / 2) % HM[1]
    y = sc.FACTOR * (i + dy) + (sc.FACTOR - 1) / 2
    return x / HM[1], y / HM[0]


def wrap_dx(dx_px):
    return (dx_px + HM[1] / 2) % HM[1] - HM[1] / 2


def great_circle_deg(x1, y1, x2, y2):
    lon1, lon2 = x1 * 2 * np.pi, x2 * 2 * np.pi
    lat1, lat2 = (0.5 - y1) * np.pi, (0.5 - y2) * np.pi
    s = (np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def build_pairs(dets_by_pano, gts, rsq):
    """Fixed argmax matching. Returns rows (pano, det, gt_x, gt_y) for scored pairs."""
    rows = []
    for pid, (pts, scored) in gts.items():
        dets = dets_by_pano.get(pid, {"dets": []})["dets"]
        pred = [(d[1] / HM[1], d[0] / HM[0]) for d in dets]    # already by confidence
        for d, (g, _) in zip(dets, greedy_match(pred, pts, rsq, HM[1], HM[0], True)):
            if g >= 0 and scored[g]:
                rows.append((pid, d, pts[g][0], pts[g][1]))
    return rows


def residuals(rows, method, wrap_x):
    """Arrays over pairs: dx, dy (GT - det, hi-res px), great-circle deg, pano index."""
    xy = np.array([decode(d, method, wrap_x) for _, d, _, _ in rows])
    gt = np.array([(gx, gy) for _, _, gx, gy in rows])
    dx = wrap_dx((gt[:, 0] - xy[:, 0]) * HM[1])
    dy = (gt[:, 1] - xy[:, 1]) * HM[0]
    gc = great_circle_deg(xy[:, 0], xy[:, 1], gt[:, 0], gt[:, 1])
    return dx, dy, gc


def stats(dx, dy, gc):
    e = np.hypot(dx, dy)
    return {"mean_px": float(e.mean()), "median_px": float(np.median(e)),
            "rms_x_px": float(np.sqrt((dx ** 2).mean())),
            "rms_y_px": float(np.sqrt((dy ** 2).mean())),
            "bias_x_px": float(dx.mean()), "bias_y_px": float(dy.mean()),
            "sd_x_px": float(dx.std()), "sd_y_px": float(dy.std()),
            "mean_deg": float(gc.mean()), "median_deg": float(np.median(gc))}


def paired_boot(pano_idx, a, b, rng, n_reps=N_REPS):
    """Cluster (pano) bootstrap of the paired difference in four statistics, b - a.

    ``a`` / ``b`` = (dx, dy, gc). Statistics: mean euclidean px, mean great-circle deg,
    and the per-axis SD (bias removed within each resample) in px."""
    groups = [np.flatnonzero(pano_idx == u) for u in np.unique(pano_idx)]

    def f(ix, r):
        dx, dy, gc = (v[ix] for v in r)
        return np.array([np.hypot(dx, dy).mean(), gc.mean(), dx.std(), dy.std()])

    allix = np.arange(len(pano_idx))
    obs = f(allix, b) - f(allix, a)
    draws = np.empty((n_reps, 4))
    for k in range(n_reps):
        pick = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([groups[p] for p in pick])
        draws[k] = f(ix, b) - f(ix, a)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    names = ("d_mean_px", "d_mean_deg", "d_sd_x_px", "d_sd_y_px")
    return {nm: {"obs": float(o), "ci95": [float(l), float(h)]}
            for nm, o, l, h in zip(names, obs, lo, hi)}


def subcell_fit(rows, method, wrap_x):
    """How well the decoded sub-cell offset predicts the GT's own offset from the same
    coarse centre, per axis: OLS slope of GT offset on decoded offset, and Pearson r."""
    out = {}
    for ax in ("x", "y"):
        u, d = [], []
        for _, det, gx, gy in rows:
            r, c, _, i, j, _, _ = det
            x, y = decode(det, method, wrap_x)
            if ax == "x":
                centre = sc.FACTOR * j + 3.5
                u.append(wrap_dx(gx * HM[1] - centre) / sc.FACTOR)
                d.append(wrap_dx(x * HM[1] - centre) / sc.FACTOR)
            else:
                centre = sc.FACTOR * i + 3.5
                u.append((gy * HM[0] - centre) / sc.FACTOR)
                d.append((y * HM[0] - centre) / sc.FACTOR)
        u, d = np.array(u), np.array(d)
        if d.std() < 1e-9:
            out[ax] = {"slope": None, "pearson_r": None, "sd_decoded_cells": float(d.std())}
            continue
        slope = float(np.cov(d, u)[0, 1] / d.var(ddof=1))
        out[ax] = {"slope": slope, "pearson_r": float(np.corrcoef(d, u)[0, 1]),
                   "sd_decoded_cells": float(d.std())}
    return out


def mod8_hist(vals_px):
    return np.bincount(np.floor(np.asarray(vals_px)).astype(int) % 8, minlength=8).tolist()


def in_border_band(r, c, md=MIN_DISTANCE):
    """Within ``md`` px of the heatmap edge: the band ``exclude_border=True`` drops."""
    return r < md or c < md or r > HM[0] - 1 - md or c > HM[1] - 1 - md


def instrument_check(split, recs):
    """Our argmax peaks >= 0.30 vs the committed analysis_out/op_cache/<split>.json.

    The op_caches hold no peak within 10 px of the heatmap edge (they were extracted with
    skimage's default ``exclude_border=True``, the #132 defect; production and this
    script pass False). Peaks in that band are counted, not compared. Outside it, a peak
    agrees if the op_cache has one at the same pixel or 1 px away (Chebyshev): a near-tie
    between the two pixels flanking one coarse centre (8i+3 vs 8i+4) flips on
    cross-machine fp32 noise. Both pixels lie in the same coarse cell, so every refined
    decode places such a peak identically. (That 1-px tie is not #221's 7-px flip, which
    is a change in *which* coarse cell is the maximum, 8i+4 -> 8(i+1)+3.) Peaks within
    ``CHECK_TOL`` of the 0.30 floor may be on either side of it in either run and are
    not counted."""
    path = os.path.join(OP_CACHE, f"{split}.json")
    if not os.path.exists(path):
        return {"status": "no op_cache"}
    with open(path, encoding="utf-8") as f:
        op = {p["pano"]: p["preds"] for p in json.load(f)["panos"]}
    n_ours = n_op = n_exact = n_1px = n_band_ours = n_band_op = 0
    max_d = 0.0
    keep = lambda k, v: not in_border_band(*k) and v >= FLOOR + CHECK_TOL  # noqa: E731
    for pid, rec in recs.items():
        ours = {(d[0], d[1]): d[2] for d in rec["dets"]}
        theirs = {(round(y * HM[0]), round(x * HM[1])): s
                  for x, y, s in op.get(pid, []) if s >= FLOOR}
        n_band_ours += sum(in_border_band(*k) for k in ours)
        n_band_op += sum(in_border_band(*k) for k in theirs)
        ours = {k: v for k, v in ours.items() if keep(k, v)}
        theirs = {k: v for k, v in theirs.items() if keep(k, v)}
        n_ours += len(ours)
        n_op += len(theirs)
        for (r, c), v in ours.items():
            if (r, c) in theirs:
                n_exact += 1
                max_d = max(max_d, abs(v - theirs[(r, c)]))
                continue
            near = [theirs[(r + a, c + b)] for a in (-1, 0, 1) for b in (-1, 0, 1)
                    if (r + a, c + b) in theirs]
            if near:
                n_1px += 1
                max_d = max(max_d, min(abs(v - t) for t in near))
    ok = n_exact + n_1px == n_ours == n_op and max_d <= CHECK_TOL
    return {"status": "ok" if ok else "MISMATCH", "peaks_ours": n_ours, "peaks_op_cache": n_op,
            "same_position": n_exact, "within_1px": n_1px, "max_score_diff": max_d,
            "border_band_ours": n_band_ours, "border_band_op_cache": n_band_op}


NOISE_JSON = os.path.join(REPO, "analysis_out", "crossview_align_48", "reference_noise.json")
OPERATIONAL = 0.55


def floor_read(recs, gts, wrap_x, rng, n_reps):
    """The #48 cross-view harness's reference-noise floor, re-read per decode.

    ``crossview_align_48.py noise`` measures how far a manual_gold detection >= 0.55 sits
    from the box centre it matches (median 1.51 deg, committed in reference_noise.json);
    docs/crossview_align_48.md calls ~2 deg the floor its placement errors can resolve.
    Same protocol here (>= 0.55, greedy one-to-one within 0.022, great-circle degrees),
    matched once on argmax. One difference: the committed file read
    ``benchmark/manual_gold/records.jsonl``, which was exported with flip TTA
    (``detections_meta.json``: ``"tta": true``), and TTA raises scores, so more peaks
    clear 0.55 there. This extraction is single-pass (the deployed decode), so the argmax
    row is expected to land near, not on, the committed numbers; both are reported."""
    rsq = radius_sq_for()
    hi = {pid: {"dets": [d for d in r["dets"] if d[2] >= OPERATIONAL]} for pid, r in recs.items()}
    rows = build_pairs(hi, gts, rsq)
    _, pano_idx = np.unique([p for p, _, _, _ in rows], return_inverse=True)
    groups = [np.flatnonzero(pano_idx == u) for u in range(pano_idx.max() + 1)]
    gc = {m: residuals(rows, m, wrap_x)[2] for m in METHODS}
    out = {"n_matched": len(rows), "methods": {}}
    for m in METHODS:
        e = gc[m]
        out["methods"][m] = {"median_deg": float(np.median(e)),
                             "p90_deg": float(np.percentile(e, 90)), "mean_deg": float(e.mean())}
        if m == "argmax":
            continue
        draws = np.empty(n_reps)
        for k in range(n_reps):
            ix = np.concatenate([groups[p] for p in rng.integers(0, len(groups), len(groups))])
            draws[k] = np.median(e[ix]) - np.median(gc["argmax"][ix])
        out["methods"][m]["d_median_deg_vs_argmax"] = {
            "obs": float(np.median(e) - np.median(gc["argmax"])),
            "ci95": [float(v) for v in np.percentile(draws, [2.5, 97.5])]}
    if os.path.exists(NOISE_JSON):
        with open(NOISE_JSON, encoding="utf-8") as f:
            ref = json.load(f)
        a = out["methods"]["argmax"]
        out["committed_reference_noise"] = {
            "n_matched": ref["n_matched"], "median_deg": ref["median_deg"],
            "p90_deg": ref["p90_deg"], "mean_deg": ref["mean_deg"],
            "input": "records.jsonl, flip TTA", "argmax_here_median_deg": a["median_deg"]}
    return out


def build_report(dets_path=DETS, wrap_x=False, n_reps=N_REPS, y_bands=Y_BANDS):
    with open(dets_path, encoding="utf-8") as f:
        D = json.load(f)
    rsq = radius_sq_for()
    rng = np.random.default_rng(SEED)
    try:
        shown = os.path.relpath(dets_path, REPO).replace(os.sep, "/")
    except ValueError:          # another drive (Windows scratch dir)
        shown = dets_path
    rep = {"inputs": {"detections": shown,
                      "detections_sha256": sha256_file(dets_path)},
           "protocol": {"floor": FLOOR, "radius_normalized": 0.022, "wrap_x_decode": wrap_x,
                        "pairs": "fixed on argmax positions, greedy by confidence",
                        "bootstrap": f"pano-cluster, {n_reps} reps, seed {SEED}",
                        "deg_per_px": DEG_PER_PX, "y_bands": list(y_bands)},
           "mechanism": {}, "instrument_check": {}, "splits": {}}
    if os.path.exists(IMAGERY_CHECK):
        with open(IMAGERY_CHECK, encoding="utf-8") as f:
            ic = json.load(f)
        rep["inputs"]["imagery_check"] = {
            "file": os.path.relpath(IMAGERY_CHECK, REPO).replace(os.sep, "/"),
            "status": ic["status"], "panos_root": ic["panos_root"],
            "splits": {s: {"status": r["status"], "match": r.get("match"),
                           "panos": r["panos"]} for s, r in ic["splits"].items()}}
    rep["inputs"]["model_revision_as_run"] = MODEL_REVISION
    rep["inputs"]["model_weights_sha256"] = MODEL_WEIGHTS_SHA256
    recon = {}
    pooled_rows = {"boxes4": [], "all5": []}
    for split in SPLITS:
        recs = D["panos"].get(split)
        if not recs:
            rep["splits"][split] = {"status": "not extracted"}
            continue
        alld = [d for r in recs.values() for d in r["dets"]]
        rep["mechanism"][split] = {
            "panos": len(recs), "peaks": len(alld),
            "col_mod8_in_34": sum(d[1] % 8 in (3, 4) for d in alld),
            "row_mod8_in_34": sum(d[0] % 8 in (3, 4) for d in alld),
            "climbed": sum(d[5] > 0 for d in alld),
            "score_over_1": sum(d[2] > 1 for d in alld),
            "off_grid_and_score_over_1": sum((d[1] % 8 not in (3, 4) or d[0] % 8 not in (3, 4))
                                             and d[2] > 1 for d in alld),
            "off_grid": sum(d[1] % 8 not in (3, 4) or d[0] % 8 not in (3, 4) for d in alld),
            # the other off-grid source: bilinear upsampling clamps at the edges, so hi-res
            # cols 0-3 all equal coarse col 0 (a plateau; peak_local_max returns col 0)
            "off_grid_other_than_plateau_or_col0": sum(
                (d[1] % 8 not in (3, 4) or d[0] % 8 not in (3, 4)) and d[2] <= 1 and d[1] != 0
                for d in alld),
            # climb census (#226 review S4): clipped peaks never needed it here; every
            # climb was an on-grid peak pulled into the next cell by a diagonal neighbour
            "climbed_and_score_over_1": sum(d[5] > 0 and d[2] > 1 for d in alld),
            "climbed_on_grid": sum(d[5] > 0 and d[1] % 8 in (3, 4) and d[0] % 8 in (3, 4)
                                   for d in alld),
            "peak_score_max": max((d[2] for d in alld), default=None)}
        # fp32-scale values: kept out of rnd(), which would print them as 0 (review S1)
        recon[split] = {
            "recon_max_abs_torch": max(r["recon_max_abs"] for r in recs.values()),
            "recon_min_abs_torch": min(r["recon_max_abs"] for r in recs.values()),
            "recon_max_abs_numpy": max(r["numpy_recon_max_abs"] for r in recs.values()),
            "recon_panos_exactly_zero": sum(r["recon_max_abs"] == 0 for r in recs.values())}
        rep["instrument_check"][split] = instrument_check(split, recs)
        gts = ground_truth(split)
        rows = build_pairs(recs, gts, rsq)
        pooled_rows["all5"] += [(f"{split}/{p}", d, gx, gy) for p, d, gx, gy in rows]
        if split in BOX_SPLITS:
            pooled_rows["boxes4"] += [(f"{split}/{p}", d, gx, gy) for p, d, gx, gy in rows]
        rep["splits"][split] = summarize(rows, gts, wrap_x, rng, n_reps)
        rep["splits"][split]["tp_at_floor"] = tp_by_method(recs, gts, rsq, wrap_x)
        if split == "manual_gold":
            rep["crossview_floor"] = floor_read(recs, gts, wrap_x, rng, n_reps)
    for name, rows in pooled_rows.items():
        rep["splits"][f"pooled:{name}"] = summarize(rows, None, wrap_x, rng, n_reps)
    # Deterministic extras added after the #226 review. They draw no random numbers, so
    # every bootstrap above is unchanged by their presence.
    for split in SPLITS:
        recs = D["panos"].get(split)
        if not recs:
            continue
        rows = build_pairs(recs, ground_truth(split), rsq)
        s = rep["splits"][split]
        s["quantization_variance"] = quantization_variance(s["methods"])
        s["climbed_pairs"] = climbed_pairs(rows, wrap_x)
        if split == "manual_gold" and y_bands:
            s["y_profile"] = y_profile(rows, y_bands, wrap_x)
    for name in pooled_rows:
        s = rep["splits"][f"pooled:{name}"]
        s["quantization_variance"] = quantization_variance(s["methods"])
    rep = rnd(rep)
    for split, v in recon.items():
        rep["mechanism"][split].update({k: float(f"{x:.3g}") if isinstance(x, float) else x
                                        for k, x in v.items()})
    return rep


def quantization_variance(methods):
    """Per-axis variance removed by the Gaussian decode, against both quantization models.

    ``methods`` is a split's per-decode ``stats`` (SDs are bias-removed). The measured
    ``centre - argmax`` variance is the check on the argmax model: it predicts
    ``QVAR_CENTRE - QVAR_ARGMAX`` = 1.75 px^2 per axis."""
    out = {"model_centre_snap_px2": QVAR_CENTRE, "model_argmax_px2": QVAR_ARGMAX,
           "model_centre_minus_argmax_px2": QVAR_CENTRE - QVAR_ARGMAX}
    for ax in ("x", "y"):
        v = {m: methods[m][f"sd_{ax}_px"] ** 2 for m in ("argmax", "centre", "gaussian")}
        removed = v["argmax"] - v["gaussian"]
        out[ax] = {"var_argmax_px2": v["argmax"], "var_centre_px2": v["centre"],
                   "var_gaussian_px2": v["gaussian"],
                   "measured_centre_minus_argmax_px2": v["centre"] - v["argmax"],
                   "removed_vs_argmax_px2": removed,
                   "removed_frac_of_argmax_model": removed / QVAR_ARGMAX,
                   "removed_vs_centre_px2": v["centre"] - v["gaussian"],
                   "removed_frac_of_centre_model": (v["centre"] - v["gaussian"]) / QVAR_CENTRE}
    return out


def _euclid(rows, method, wrap_x):
    dx, dy, _ = residuals(rows, method, wrap_x)
    return np.hypot(dx, dy)


def climbed_pairs(rows, wrap_x):
    """Pairs whose peak was climbed to a neighbouring coarse cell: count, and the mean
    residual under argmax (the pixel peak_local_max returned) and under gaussian."""
    sel = [r for r in rows if r[1][5] > 0]
    if not sel:
        return {"pairs": 0}
    return {"pairs": len(sel), "argmax_mean_px": float(_euclid(sel, "argmax", wrap_x).mean()),
            "gaussian_mean_px": float(_euclid(sel, "gaussian", wrap_x).mean())}


def y_profile(rows, bands, wrap_x):
    """Where the non-uniform decoded-y mod-8 histogram (doc section 4.3) comes from.

    Per argmax-row band ``[lo, hi)``: pair count, decoded-y mod-8 histogram, the mean
    offset from the coarse centre (cells) of the gaussian decode and of the GT, and the
    mean |y residual| under argmax and gaussian. Plus the shape of the coarse profile over
    all pairs (share of peaks whose lower / right neighbour is the higher one, median
    log-curvature per axis; sigma 1.25 cells gives -1/1.25^2 = -0.64) and a calibration
    table: GT offset by decoded-offset bin."""
    r_arg = np.array([d[0] for _, d, _, _ in rows], dtype=float)
    cen = sc.FACTOR * np.array([d[3] for _, d, _, _ in rows]) + (sc.FACTOR - 1) / 2
    y_dec = np.array([decode(d, "gaussian", wrap_x)[1] for _, d, _, _ in rows]) * HM[0]
    y_gt = np.array([gy for _, _, _, gy in rows]) * HM[0]
    off_dec, off_gt = (y_dec - cen) / sc.FACTOR, (y_gt - cen) / sc.FACTOR
    edges = [0] + list(bands) + [HM[0]]
    out = {"band_on": "argmax row, hi-res px, [lo, hi)", "bands": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r_arg >= lo) & (r_arg < hi)
        b = {"rows": [lo, hi], "pairs": int(m.sum())}
        if m.any():
            b.update({"mod8_gaussian": mod8_hist(y_dec[m]),
                      "mean_offset_decoded_cells": float(off_dec[m].mean()),
                      "mean_offset_gt_cells": float(off_gt[m].mean()),
                      "mean_abs_dy_argmax_px": float(np.abs(y_gt - r_arg)[m].mean()),
                      "mean_abs_dy_gaussian_px": float(np.abs(y_gt - y_dec)[m].mean())})
        out["bands"].append(b)
    nb = np.array([[np.nan if v is None else v for v in d[6]] for _, d, _, _ in rows])
    L = np.log(np.clip(nb, 1e-6, None))
    out["shape"] = {"lower_neighbour_higher": float(np.mean(nb[:, 7] > nb[:, 1])),
                    "right_neighbour_higher": float(np.mean(nb[:, 5] > nb[:, 3])),
                    "median_log_curvature_y": float(np.nanmedian(L[:, 7] - 2 * L[:, 4] + L[:, 1])),
                    "median_log_curvature_x": float(np.nanmedian(L[:, 5] - 2 * L[:, 4] + L[:, 3])),
                    "sd_offset_decoded_cells": float(off_dec.std()),
                    "sd_offset_gt_cells": float(off_gt.std())}
    cal = []
    e = np.linspace(-0.5, 0.5, 6)
    for k, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        m = (off_dec >= lo) & ((off_dec <= hi) if k == len(e) - 2 else (off_dec < hi))
        if m.any():
            cal.append({"decoded_offset_bin": [round(float(lo), 2), round(float(hi), 2)],
                        "pairs": int(m.sum()),
                        "mean_decoded": float(off_dec[m].mean()),
                        "mean_gt": float(off_gt[m].mean())})
    out["calibration"] = cal
    return out


def tp_by_method(recs, gts, rsq, wrap_x, methods=("argmax", "gaussian")):
    """Matched detections >= 0.30 when each decode's positions are matched afresh.

    The benchmark radius (0.022, ~22.5 px) is wide next to a <= 4 px move, so this is
    expected to barely change; it is here so that "the decode does not move detection
    metrics" is a measured statement. Box splits count only scored (boxed) GT."""
    out = {}
    for m in methods:
        tp = 0
        for pid, (pts, scored) in gts.items():
            dets = recs.get(pid, {"dets": []})["dets"]
            pred = [decode(d, m, wrap_x) for d in dets]
            tp += sum(g >= 0 and scored[g]
                      for g, _ in greedy_match(pred, pts, rsq, HM[1], HM[0], True))
        out[m] = tp
    return out


def summarize(rows, gts, wrap_x, rng, n_reps):
    if not rows:
        return {"pairs": 0}
    _, pano_idx = np.unique([p for p, _, _, _ in rows], return_inverse=True)
    res = {m: residuals(rows, m, wrap_x) for m in METHODS}
    out = {"pairs": len(rows), "panos_with_pairs": int(pano_idx.max() + 1),
           "methods": {m: stats(*res[m]) for m in METHODS},
           "vs_argmax": {m: paired_boot(pano_idx, res["argmax"], res[m], rng, n_reps)
                         for m in METHODS if m != "argmax"},
           "subcell_fit": {m: subcell_fit(rows, m, wrap_x) for m in METHODS
                           if m not in ("argmax", "centre")}}
    if gts is not None:
        out["gt_panos"] = len(gts)
        out["gt_points_scored"] = sum(sum(s) for _, s in gts.values())
    # mod-8 histograms of the detection position (hi-res px) before and after, and of the
    # matched GT, per axis
    xy = {m: np.array([decode(d, m, wrap_x) for _, d, _, _ in rows])
          for m in ("argmax", "gaussian", "dark")}
    gt = np.array([(gx, gy) for _, _, gx, gy in rows])
    out["mod8"] = {"x": {m: mod8_hist(v[:, 0] * HM[1]) for m, v in xy.items()},
                   "y": {m: mod8_hist(v[:, 1] * HM[0]) for m, v in xy.items()}}
    out["mod8"]["x"]["gt"] = mod8_hist(gt[:, 0] * HM[1])
    out["mod8"]["y"]["gt"] = mod8_hist(gt[:, 1] * HM[0])
    return out


def markdown(rep):
    L = ["# Sub-cell decode (#221): results", "",
         f"Generated by `scripts/analysis/subcell_decode_221.py report` from "
         f"`{rep['inputs']['detections']}` (sha256 `{rep['inputs']['detections_sha256'][:16]}`).",
         f"Pairs: {rep['protocol']['pairs']}; floor {rep['protocol']['floor']}; radius "
         f"{rep['protocol']['radius_normalized']}; bootstrap {rep['protocol']['bootstrap']}. "
         f"1 px = {DEG_PER_PX:.4f} deg on the 512x1024 grid.", "",
         f"Model `{MODEL_REPO}` at commit `{rep['inputs']['model_revision_as_run'][:12]}` "
         f"(weights sha256 `{rep['inputs']['model_weights_sha256'][:16]}`)."]
    ic = rep["inputs"].get("imagery_check")
    if ic:
        L.append(f"Imagery vs committed `imagery_manifest.json` (`{ic['file']}`, panos root "
                 f"`{ic['panos_root']}`): **{ic['status']}** — " + ", ".join(
                     f"{s} {r['match']}/{r['panos']}" for s, r in ic["splits"].items()) + ".")
    else:
        L.append("Imagery check: not run (no `imagery_check.json`).")
    L += ["", "## Mechanism", "",
          "| split | panos | peaks >= 0.30 | col mod 8 in {3,4} | row mod 8 in {3,4} | "
          "peaks > 1 (clipped plateau) | climbed (on-grid / score > 1) | "
          "max abs(head - upsample(coarse)), torch / numpy |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for s, m in rep["mechanism"].items():
        L.append(f"| {s} | {m['panos']} | {m['peaks']} | {m['col_mod8_in_34']} | "
                 f"{m['row_mod8_in_34']} | {m['score_over_1']} | {m['climbed']} "
                 f"({m['climbed_on_grid']} / {m['climbed_and_score_over_1']}) | "
                 f"{m['recon_max_abs_torch']:.2g} / {m['recon_max_abs_numpy']:.2g} |")
    L += ["", "The reconstruction difference is non-zero on "
          + ", ".join(f"{s} {m['panos'] - m['recon_panos_exactly_zero']}/{m['panos']}"
                      for s, m in rep["mechanism"].items())
          + " panos, at fp32 rounding scale (torch min "
          + f"{min(m['recon_min_abs_torch'] for m in rep['mechanism'].values()):.2g})."]
    L += ["", "## Instrument check (argmax peaks vs analysis_out/op_cache)", "",
          "Outside the 10-px border band the op_caches drop (`exclude_border`, #132), and "
          "excluding peaks within 2e-4 of the 0.30 floor:", "",
          "| split | status | ours | op_cache | same pixel | 1 px away | max score diff | "
          "border-band peaks ours / op_cache |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for s, c in rep["instrument_check"].items():
        if "peaks_ours" in c:
            L.append(f"| {s} | {c['status']} | {c['peaks_ours']} | {c['peaks_op_cache']} | "
                     f"{c['same_position']} | {c['within_1px']} | {c['max_score_diff']:.2g} | "
                     f"{c['border_band_ours']} / {c['border_band_op_cache']} |")
        else:
            L.append(f"| {s} | {c['status']} | | | | | | |")
    fl = rep.get("crossview_floor")
    if fl:
        ck = fl.get("committed_reference_noise", {})
        L += ["", "## The #48 cross-view reference floor, re-read (manual_gold, >= 0.55)", "",
              f"{fl['n_matched']} matched detections (single pass). The committed "
              f"`analysis_out/crossview_align_48/reference_noise.json` read the flip-TTA "
              f"records: n {ck.get('n_matched')}, median {ck.get('median_deg')} deg, p90 "
              f"{ck.get('p90_deg')} deg.", "",
              "| decode | median deg | p90 deg | mean deg | d median vs argmax [95% CI] |",
              "|---|---:|---:|---:|---|"]
        for m, st in fl["methods"].items():
            d = st.get("d_median_deg_vs_argmax")
            ds = ("--" if d is None else
                  f"{d['obs']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
            L.append(f"| {m} | {st['median_deg']:.3f} | {st['p90_deg']:.3f} | "
                     f"{st['mean_deg']:.3f} | {ds} |")
    for s, r in rep["splits"].items():
        if not r.get("pairs"):
            continue
        L += ["", f"## {s}: {r['pairs']} matched pairs in {r['panos_with_pairs']} panos", ""]
        if "tp_at_floor" in r:
            L += ["Matched detections >= 0.30 when each decode is matched afresh: " + ", ".join(
                f"{m} {v}" for m, v in r["tp_at_floor"].items()) + ".", ""]
        L += [
              "| decode | mean px | median px | mean deg | bias x | bias y | SD x | SD y | "
              "d mean px vs argmax [95% CI] | d SD x | d SD y |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for m, st in r["methods"].items():
            if m == "argmax":
                d = "--"
                dsx = dsy = "--"
            else:
                b = r["vs_argmax"][m]
                d = (f"{b['d_mean_px']['obs']:+.3f} [{b['d_mean_px']['ci95'][0]:+.3f}, "
                     f"{b['d_mean_px']['ci95'][1]:+.3f}]")
                dsx = (f"{b['d_sd_x_px']['obs']:+.3f} [{b['d_sd_x_px']['ci95'][0]:+.3f}, "
                       f"{b['d_sd_x_px']['ci95'][1]:+.3f}]")
                dsy = (f"{b['d_sd_y_px']['obs']:+.3f} [{b['d_sd_y_px']['ci95'][0]:+.3f}, "
                       f"{b['d_sd_y_px']['ci95'][1]:+.3f}]")
            L.append(f"| {m} | {st['mean_px']:.3f} | {st['median_px']:.3f} | "
                     f"{st['mean_deg']:.3f} | {st['bias_x_px']:+.2f} | {st['bias_y_px']:+.2f} | "
                     f"{st['sd_x_px']:.3f} | {st['sd_y_px']:.3f} | {d} | {dsx} | {dsy} |")
        L += ["", "Sub-cell fit (GT offset from the coarse centre regressed on the decoded "
              "offset; slope 1 = unbiased, r = how much of the GT's sub-cell position the "
              "decode explains):", "", "| decode | slope x | r x | slope y | r y |",
              "|---|---:|---:|---:|---:|"]
        for m, f in r["subcell_fit"].items():
            fx, fy = f["x"], f["y"]
            fmt = lambda v: "--" if v is None else f"{v:.3f}"  # noqa: E731
            L.append(f"| {m} | {fmt(fx['slope'])} | {fmt(fx['pearson_r'])} | "
                     f"{fmt(fy['slope'])} | {fmt(fy['pearson_r'])} |")
        q = r.get("quantization_variance")
        if q:
            L += ["", f"Variance removed by `gaussian` (bias-removed SDs, px^2). Models: "
                  f"centre snap {q['model_centre_snap_px2']:.2f}, argmax "
                  f"{q['model_argmax_px2']:.2f}, so centre - argmax is predicted at "
                  f"{q['model_centre_minus_argmax_px2']:.2f}:", "",
                  "| axis | var argmax | var centre | var gaussian | centre - argmax "
                  "(measured) | removed vs argmax | / argmax model | removed vs centre | "
                  "/ centre model |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for ax in ("x", "y"):
                a = q[ax]
                L.append(f"| {ax} | {a['var_argmax_px2']:.2f} | {a['var_centre_px2']:.2f} | "
                         f"{a['var_gaussian_px2']:.2f} | "
                         f"{a['measured_centre_minus_argmax_px2']:.2f} | "
                         f"{a['removed_vs_argmax_px2']:.2f} | "
                         f"{a['removed_frac_of_argmax_model']:.0%} | "
                         f"{a['removed_vs_centre_px2']:.2f} | "
                         f"{a['removed_frac_of_centre_model']:.0%} |")
        cp = r.get("climbed_pairs")
        if cp and cp.get("pairs"):
            L += ["", f"Climbed pairs (peak re-anchored to a neighbouring coarse cell): "
                  f"{cp['pairs']}, mean residual argmax {cp['argmax_mean_px']:.2f} px -> "
                  f"gaussian {cp['gaussian_mean_px']:.2f} px."]
        yp = r.get("y_profile")
        if yp:
            L += ["", f"y profile by band ({yp['band_on']}):", "",
                  "| rows | pairs | mean decoded offset (cells) | mean GT offset (cells) | "
                  "mean abs dy argmax px | mean abs dy gaussian px | decoded y mod 8 |",
                  "|---|---:|---:|---:|---:|---:|---|"]
            for b in yp["bands"]:
                if not b["pairs"]:
                    L.append(f"| {b['rows'][0]}-{b['rows'][1]} | 0 | | | | | |")
                    continue
                L.append(f"| {b['rows'][0]}-{b['rows'][1]} | {b['pairs']} | "
                         f"{b['mean_offset_decoded_cells']:+.3f} | "
                         f"{b['mean_offset_gt_cells']:+.3f} | {b['mean_abs_dy_argmax_px']:.3f} | "
                         f"{b['mean_abs_dy_gaussian_px']:.3f} | {b['mod8_gaussian']} |")
            sh = yp["shape"]
            L += ["", f"Coarse profile shape: lower neighbour higher in "
                  f"{sh['lower_neighbour_higher']:.1%} of pairs, right neighbour in "
                  f"{sh['right_neighbour_higher']:.1%}; median log-curvature y "
                  f"{sh['median_log_curvature_y']:.3f}, x {sh['median_log_curvature_x']:.3f} "
                  f"(sigma 1.25 cells: -0.640); SD of offset decoded "
                  f"{sh['sd_offset_decoded_cells']:.3f} vs GT {sh['sd_offset_gt_cells']:.3f} "
                  f"cells.", "", "| decoded y offset bin (cells) | pairs | mean decoded | "
                  "mean GT |", "|---|---:|---:|---:|"]
            for c in yp["calibration"]:
                L.append(f"| [{c['decoded_offset_bin'][0]:+.1f}, "
                         f"{c['decoded_offset_bin'][1]:+.1f}) | {c['pairs']} | "
                         f"{c['mean_decoded']:+.3f} | {c['mean_gt']:+.3f} |")
        L += ["", "Position mod 8 (hi-res px, bins 0..7):", ""]
        for ax in ("x", "y"):
            for m, h in r["mod8"][ax].items():
                L.append(f"- {ax} {m}: {h}")
    return "\n".join(L) + "\n"


def cmd_report(args):
    bands = tuple(int(v) for v in args.y_bands.split(",") if v.strip())
    rep = build_report(args.detections, wrap_x=args.wrap_x, n_reps=args.reps, y_bands=bands)
    write_json(args.out, rep)
    md = os.path.splitext(args.out)[0] + ".md"
    with open(md, "w", encoding="utf-8", newline="") as f:
        f.write(markdown(rep))
    print(f"-> {args.out}\n-> {md}")
    bad = [s for s, c in rep["instrument_check"].items() if c.get("status") == "MISMATCH"]
    if bad:
        print(f"instrument check MISMATCH on {bad}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--panos-root", required=True,
                   help="dir holding benchmark/<split>/panos/*.jpg")
    e.add_argument("--cache-dir", required=True, help="where the 64x128 coarse maps go")
    e.add_argument("--splits", default=",".join(SPLITS))
    e.add_argument("--out", default=DETS)
    e.add_argument("--usage-out", default=os.path.join(OUT_DIR, "usage_row.json"))
    e.add_argument("--limit", type=int, default=0, help="smoke test: panos per split")
    e.add_argument("--note", default="")
    e.add_argument("--model-revision", default=MODEL_REVISION,
                   help="Hub commit of projectsidewalk/rampnet-model (default: the one the "
                        "committed detections.json used)")
    e.add_argument("--allow-imagery-mismatch", action="store_true",
                   help="run even if a pano's sha256 differs from imagery_manifest.json")
    v = sub.add_parser("verify-imagery")
    v.add_argument("--panos-root", required=True,
                   help="dir holding benchmark/<split>/panos/*.jpg")
    v.add_argument("--splits", default=",".join(SPLITS))
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--out", default=IMAGERY_CHECK)
    r = sub.add_parser("report")
    r.add_argument("--detections", default=DETS)
    r.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    r.add_argument("--wrap-x", action="store_true",
                   help="use the neighbour across the 360 seam at coarse cols 0 / 127")
    r.add_argument("--reps", type=int, default=N_REPS)
    r.add_argument("--y-bands", default=",".join(str(v) for v in Y_BANDS),
                   help="argmax-row band edges for the manual_gold y-profile read "
                        "(doc section 4.3); empty string to skip it")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        cmd_extract(a)
        return 0
    if a.cmd == "verify-imagery":
        return cmd_verify_imagery(a)
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
