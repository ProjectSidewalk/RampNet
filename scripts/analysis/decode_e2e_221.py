"""End-to-end checks of the shipped sub-cell decode (#221, PR #229).

PR #229 shipped ``rampnet.subcell.detect_peaks``, ``stage_two/evaluate.py --decode`` and
``RampNetModel.detect()`` without running any of them on real model output. This script
holds the parts of that verification that are not a plain ``evaluate.py`` invocation; the
full command sequence is in ``docs/decode_e2e_221.md``.

Subcommands
-----------
``positions``  Reads the heatmap and coarse caches a ``evaluate.py --decode gaussian`` run
               left behind, for one TTA setting, and measures position error against the
               manual_gold box centres the way #226 did (peaks >= 0.30, matched once on
               their argmax positions, radius 0.022, x wrapped; every decode scored on the
               same pairs). With TTA it also scores the alternatives to the shipped
               branch-select rule (original branch only, flipped branch only, mean of the
               two branches' decodes). CPU only.

``roundtrip``  Exports a checkpoint with ``scripts/export_hf_model.py`` (never ``--push``)
               to a local directory, loads it the way a Hub user would
               (``AutoModel.from_pretrained(dir, trust_remote_code=True)``), runs
               ``detect()`` with both decodes on a few manual_gold panos, and compares the
               result with ``stage_two/evaluate.py``'s own extraction (single pass) for the
               same panos and weights. Needs a GPU or a lot of patience, transformers and
               the panos.

Usage::

    python scripts/analysis/decode_e2e_221.py positions --cache-dir <evaluate cache root> \
        --fingerprint f7f255c586ba --tta --out analysis_out/decode_e2e_221/positions_tta.json
    python scripts/analysis/decode_e2e_221.py roundtrip --checkpoint <released .pth> \
        --panos-dir benchmark/manual_gold/panos --export-dir <tmp> \
        --out analysis_out/decode_e2e_221/roundtrip.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "stage_two"))

from rampnet import subcell as sc  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# stage_two/evaluate.py, loaded by path: a pip package called "evaluate" can shadow it.
ev = _load("rampnet_stage_two_evaluate", os.path.join(REPO, "stage_two", "evaluate.py"))
# The #226 analysis script: same GT loader, residual and bootstrap code, so the numbers
# here are computed exactly the way section 4 of docs/subcell_decode_221.md computed them.
s221 = _load("subcell_decode_221", os.path.join(REPO, "scripts", "analysis",
                                                 "subcell_decode_221.py"))

HM = (512, 1024)
FLOOR = 0.30
OPS = (0.30, 0.55)
RSQ = (ev.RADIUS_THRESHOLD_NORMALIZED * HM[1]) ** 2
SEED = 221
N_REPS = 2000
SIG = 6


def rnd(v):
    if isinstance(v, dict):
        return {k: rnd(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x) for x in v]
    if isinstance(v, (float, np.floating)):
        # significant figures, not decimals: a 1e-9 difference must not print as 0
        return float(f"{float(v):.{SIG}g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(rnd(obj), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #
def _xy(rcs):
    """detect_peaks rows -> normalized (x, y)."""
    return np.column_stack([rcs[:, 1] / HM[1], rcs[:, 0] / HM[0]])


def _mean_xy(a, b):
    """Mean of two normalized positions, x on the circle (they are never > 0.5 apart)."""
    dx = (b[:, 0] - a[:, 0] + 0.5) % 1.0 - 0.5
    return np.column_stack([(a[:, 0] + dx / 2) % 1.0, (a[:, 1] + b[:, 1]) / 2])


def pano_decodes(h, coarse, threshold):
    """Every decode of one pano's peaks >= threshold, all on the same peaks.

    Returns ``(scores, {name: (N, 2) normalized xy}, branch)``. ``argmax`` and
    ``gaussian`` go through ``evaluate.extract_peaks_from_heatmap`` -- the shipped code
    path, stale-cache guard included. With a TTA stack (B = 2) the alternatives to the
    shipped branch-select rule are added: decode from the original branch only, from the
    flipped branch only, and the mean of the two."""
    out = {}
    pa = ev.extract_peaks_from_heatmap(h, ev.PEAK_MIN_DISTANCE, threshold, HM, "argmax")
    pg = ev.extract_peaks_from_heatmap(h, ev.PEAK_MIN_DISTANCE, threshold, HM, "gaussian",
                                       coarse=coarse)
    sa = np.array([p[2] for p in pa], dtype=float)
    sg = np.array([p[2] for p in pg], dtype=float)
    if len(sa) != len(sg) or not np.array_equal(sa, sg):
        raise AssertionError("argmax and gaussian found different peaks or scores")
    out["argmax"] = np.array([(p[0], p[1]) for p in pa], dtype=float).reshape(-1, 2)
    out["gaussian"] = np.array([(p[0], p[1]) for p in pg], dtype=float).reshape(-1, 2)
    branch = np.zeros(len(sa), dtype=int)
    if coarse.ndim == 3 and len(coarse) == 2 and len(sa):
        _, pk = sc.detect_peaks(h, threshold, ev.PEAK_MIN_DISTANCE, decode="argmax",
                                return_pixels=True)
        branch = np.array([int(np.argmax([sc._value_at(cb.astype(np.float64), r, c, HM)
                                          for cb in coarse])) for r, c in pk])
        for b, name in ((0, "orig_branch"), (1, "flip_branch")):
            out[name] = _xy(sc.detect_peaks(h, threshold, ev.PEAK_MIN_DISTANCE,
                                            decode="gaussian", coarse=coarse[b]))
        out["branch_mean"] = _mean_xy(out["orig_branch"], out["flip_branch"])
    return sa, out, branch


def residuals(xy, gt):
    dx = s221.wrap_dx((gt[:, 0] - xy[:, 0]) * HM[1])
    dy = (gt[:, 1] - xy[:, 1]) * HM[0]
    gc = s221.great_circle_deg(xy[:, 0], xy[:, 1], gt[:, 0], gt[:, 1])
    return dx, dy, gc


def cmd_positions(args):
    tag = "tta" if args.tta else "notta"
    key = f"{args.fingerprint}_manual_{tag}"
    hdir = os.path.join(args.cache_dir, "heatmaps", key)
    cdir = os.path.join(args.cache_dir, "coarse", key)
    gts = s221.ground_truth("manual_gold")
    pids = sorted(gts)
    if args.limit:
        pids = pids[:args.limit]
    t0 = time.time()
    rows = {}           # decode -> list of (pano_idx, x, y, gx, gy)
    tp = {op: {} for op in OPS}       # each decode re-matched afresh (detection metric)
    branch_counts = np.zeros(2, dtype=int)
    branch_by_tp = np.zeros(2, dtype=int)
    n_peaks = 0
    for k, pid in enumerate(pids):
        h = np.load(os.path.join(hdir, f"{pid}_heatmap.npy"))
        coarse = np.load(os.path.join(cdir, f"{pid}_coarse.npy"))
        scores, dec, branch = pano_decodes(h, coarse, FLOOR)
        n_peaks += len(scores)
        order = np.argsort(-scores, kind="stable")
        pts, scored = gts[pid]
        # pairs: fixed on argmax positions, as in #226
        match = greedy_match([tuple(v) for v in dec["argmax"][order]], pts, RSQ, HM[1], HM[0],
                             True)
        for oi, (g, _) in zip(order, match):
            if g >= 0 and scored[g]:
                for name, xy in dec.items():
                    rows.setdefault(name, []).append((k, xy[oi, 0], xy[oi, 1],
                                                      pts[g][0], pts[g][1]))
                branch_by_tp[branch[oi]] += 1
        branch_counts += np.bincount(branch, minlength=2)
        for op in OPS:
            keep = order[scores[order] >= op]
            for name, xy in dec.items():
                m = greedy_match([tuple(v) for v in xy[keep]], pts, RSQ, HM[1], HM[0], True)
                tp[op][name] = tp[op].get(name, 0) + sum(g >= 0 and scored[g] for g, _ in m)
    rng = np.random.default_rng(SEED)
    res = {}
    base = None
    for name in ["argmax", "gaussian"] + [n for n in rows if n not in ("argmax", "gaussian")]:
        r = np.array(rows[name])
        pano_idx = r[:, 0].astype(int)
        d = residuals(r[:, 1:3], r[:, 3:5])
        res[name] = {"stats": s221.stats(*d)}
        if name == "argmax":
            base = (pano_idx, d)
        else:
            res[name]["vs_argmax"] = s221.paired_boot(pano_idx, base[1], d, rng, args.reps)
    n_gt = sum(sum(s) for p, (_, s) in gts.items() if p in set(pids))
    out = {
        "what": "position error of evaluate.py's decodes vs manual_gold box centres, from "
                "evaluate.py's own caches (#221 end-to-end check)",
        "cache_key": key, "tta": args.tta, "panos": len(pids), "gt_points": n_gt,
        "peaks_ge_0.30": n_peaks, "pairs": len(rows["argmax"]),
        "pair_panos": len({r[0] for r in rows["argmax"]}),
        "protocol": "peaks >= 0.30 via evaluate.extract_peaks_from_heatmap; pairs matched once "
                    "on argmax positions (greedy by confidence, radius 0.022, x wrapped); "
                    f"pano-cluster bootstrap, {args.reps} reps, seed {SEED}",
        "decodes": res,
        "tp_rematched": {f"{op:.2f}": tp[op] for op in OPS},
        "elapsed_s": time.time() - t0,
    }
    if args.tta:
        out["branch_select"] = {"all_peaks": branch_counts.tolist(),
                                "paired_peaks": branch_by_tp.tolist(),
                                "note": "[original, flipped] branch chosen by the shipped rule"}
    write_json(args.out, out)
    for name, v in res.items():
        line = f"{name:12s} mean {v['stats']['mean_px']:.3f} px"
        if "vs_argmax" in v:
            d = v["vs_argmax"]["d_mean_px"]
            line += f"  d {d['obs']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
        print(line)
    print("TP rematched:", out["tp_rematched"])


# --------------------------------------------------------------------------- #
# roundtrip
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_roundtrip(args):
    import torch
    from PIL import Image
    from transformers import AutoModel

    t0 = time.time()
    exp = os.path.abspath(args.export_dir)
    if not args.reuse_export:
        cmd = [sys.executable, os.path.join(REPO, "scripts", "export_hf_model.py"),
               "--checkpoint", args.checkpoint, "--output-dir", exp]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO)
    # The package's verbatim copies must equal the canonical modules byte for byte.
    sync = {}
    for dst, src in (("rampnet_model.py", "rampnet/model.py"),
                     ("rampnet_subcell.py", "rampnet/subcell.py"),
                     ("modeling_rampnet.py", "scripts/hf_package/modeling_rampnet.py"),
                     ("configuration_rampnet.py", "scripts/hf_package/configuration_rampnet.py")):
        sync[dst] = sha256_file(os.path.join(exp, dst)) == sha256_file(os.path.join(REPO, src))
    hub = AutoModel.from_pretrained(exp, trust_remote_code=True).eval()
    ref = ev.load_trained_model(args.checkpoint, ev.MODEL_HEATMAP_SIZE)
    hub.to(ev.DEVICE)
    pids = sorted(p[:-4] for p in os.listdir(args.panos_dir) if p.endswith(".jpg"))[:args.n]
    per = []
    worst = {"argmax": 0.0, "gaussian": 0.0}
    all_ok = True
    for pid in pids:
        img = Image.open(os.path.join(args.panos_dir, pid + ".jpg")).convert("RGB")
        x = ev.preprocess_transform(img).unsqueeze(0)
        h, coarse = ev.predict_heatmap(ref, img, use_tta=False, return_coarse=True)
        rec = {"pano": pid}
        for dec in ("argmax", "gaussian"):
            mine = np.array(ev.extract_peaks_from_heatmap(
                h, ev.PEAK_MIN_DISTANCE, args.threshold, ev.MODEL_HEATMAP_SIZE, dec,
                coarse=None if dec == "argmax" else coarse), dtype=float).reshape(-1, 3)
            theirs = hub.detect(x, threshold=args.threshold, decode=dec)[0]
            same_n = len(mine) == len(theirs)
            d = float(np.max(np.abs(mine - theirs))) if same_n and len(mine) else 0.0
            dpx = (float(np.max(np.abs((mine[:, :2] - theirs[:, :2]) * [HM[1], HM[0]])))
                   if same_n and len(mine) else 0.0)
            rec[dec] = {"n_eval": len(mine), "n_detect": len(theirs), "max_abs_diff": d,
                        "max_abs_diff_px": dpx}
            ok = same_n and d <= args.tol
            all_ok &= ok
            worst[dec] = max(worst[dec], d if same_n else float("inf"))
        per.append(rec)
        print(pid, {k: v for k, v in rec.items() if k != "pano"})
    out = {
        "what": "HF export round-trip (#221): RampNetModel.detect() from a locally exported "
                "package vs stage_two/evaluate.py's single-pass extraction, same weights",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "export_dir_files": sorted(os.listdir(exp)),
        "verbatim_sync": sync,
        "threshold": args.threshold, "tol": args.tol, "panos": per,
        "worst_max_abs_diff": worst, "all_match": bool(all_ok),
        "torch": torch.__version__, "device": str(ev.DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_s": time.time() - t0,
        "tol_note": "evaluate.py casts the coarse stack to float32 before decoding (so a "
                    "cached decode equals a fresh one); detect() decodes from the float64 "
                    "recovery. That moves a gaussian position by ~1e-6 px, so the gaussian "
                    "comparison needs a tolerance of order 1e-8 (normalized); argmax is exact.",
    }
    import transformers
    out["transformers"] = transformers.__version__
    write_json(args.out, out)
    print("verbatim sync:", sync, " all_match:", all_ok)
    if not all_ok or not all(sync.values()):
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions")
    p.add_argument("--cache-dir", required=True, help="evaluate.py --cache-dir root")
    p.add_argument("--fingerprint", required=True, help="checkpoint_fingerprint of the weights")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=N_REPS)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N panos")
    r = sub.add_parser("roundtrip")
    r.add_argument("--checkpoint", required=True, help="local .pth of the weights to export")
    r.add_argument("--panos-dir", required=True)
    r.add_argument("--export-dir", required=True, help="local directory; nothing is uploaded")
    r.add_argument("--reuse-export", action="store_true")
    r.add_argument("--n", type=int, default=8, help="number of panos (sorted by id)")
    r.add_argument("--threshold", type=float, default=FLOOR)
    r.add_argument("--tol", type=float, default=0.0,
                   help="max abs difference allowed in (x, y, score); 0 = exact")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"positions": cmd_positions, "roundtrip": cmd_roundtrip}[args.cmd](args)


if __name__ == "__main__":
    main()
