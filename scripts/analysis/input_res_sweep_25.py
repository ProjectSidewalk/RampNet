"""Frozen-model input-size sweep for RampNet (issue #25, arm 1: no retraining).

RampNet was trained, and is deployed, at a fixed 2048x4096 input. The benchmark panoramas
are stored at native resolution, which for most splits is larger than that (annapolis
8000 px wide, richmond mostly 11000-12288, the GSV splits 16384). This script asks the
cheap question issue #25 lists first: **does the released checkpoint, unchanged, tolerate
or benefit from being fed more pixels?** It does not answer whether a model *retrained*
at higher resolution would gain (the issue's +10-recall-point forecast is for that arm).

Arms (input H x W; every resize is torchvision ``Resize`` on the PIL image, the same call
``threshold_sweep.PRE`` makes):

    r2048         native -> 2048x4096 BILINEAR          the control; must equal op_cache
    r3072         native -> 3072x6144 BILINEAR          1.5x
    r4096         native -> 4096x8192 BILINEAR          2x
    rnative       native size, floored at 2048x4096 and capped at --native-cap
    u4096         native -> 2048x4096 BILINEAR -> 4096x8192 BICUBIC
                  (the upsample control: object scale grows, no new information)
    r4096_hm1024  as r4096, but the head's Upsample targets 1024x2048 and peaks are
                  extracted at min_distance 20 (the same normalized radius as 10 on
                  512x1024): the extraction-parameter sensitivity arm

The model as constructed resamples its /32 feature map to a fixed 512x1024 heatmap
whatever the input size, so every arm except r4096_hm1024 extracts peaks on the same
grid with the same ``min_distance`` in normalized units.

Subcommands:

    # GPU (makelab2). Per-city/per-arm skip-if-exists; decode each native jpg once and
    # run every missing arm on it. Appends one usage_log row per arm + one decode row.
    python scripts/analysis/input_res_sweep_25.py extract --arms r2048 \
        --panos-root /homes/gws/jonf/RampNet

    # CPU. The instrument check: r2048 must reproduce analysis_out/op_cache. Exit 1 on
    # mismatch. Run before any other arm's number is read.
    python scripts/analysis/input_res_sweep_25.py check

    # CPU. Every table + the paired pano-level bootstrap -> results.json + results.md
    python scripts/analysis/input_res_sweep_25.py report

The launcher ``input_res_sweep_25.sh`` runs extract(r2048) -> check -> extract(rest) ->
report, so a failed check stops the run before any other arm is extracted.
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from queue import Queue

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import ledger  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402
from operating_point_curve import (  # noqa: E402
    _gt_from_json, _gt_to_json, _score_at, bundle_ground_truths, pr_curve_and_ap)
from recall_by_depth_112 import (  # noqa: E402
    CAM_H, M_BUCKETS, PX_BUCKETS, apparent_px, bucket_label, bucket_of, flat_range)
from miss_decomposition import US_SPLITS  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "input_res_sweep_25")
CACHE_ROOT = os.path.join(OUT_DIR, "cache")
OP_CACHE = os.path.join(REPO, "analysis_out", "op_cache")
USAGE_LOG = os.path.join(REPO, "analysis_out", "usage_log.jsonl")

#: Every benchmark split with a verdict review. manual_gold is excluded: its panos are
#: stored at the model's own 2048x4096, so there is no headroom to test, and it is the
#: in-distribution split.
SPLITS = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville", "laurens_gsv",
          "laurens_mapillary", "morgantown", "paterson", "richmond", "sao_paulo")
HEADLINE = ("annapolis", "richmond", "laurens_mapillary")   # Mapillary, with headroom
GSV_SPLITS = ("bend", "paterson", "gainesville", "sao_paulo", "laurens_gsv")
POOLS = {
    "headline (annapolis+richmond+laurens_mapillary)": HEADLINE,
    "US pool (miss_decomposition.US_SPLITS)": tuple(US_SPLITS),
    "GSV z5 (bend+paterson+gainesville+sao_paulo+laurens_gsv)": GSV_SPLITS,
}

BASE_SIZE = (2048, 4096)          # (H, W) the model was trained at
BASE_HEATMAP = (512, 1024)
DEFAULT_NATIVE_CAP = (6144, 12288)
SCORE_FLOOR = 0.05
THRESHOLDS = (0.30, 0.55)         # the #79 recommended operating point; the shipped one
N_REPS = 2000
SEED = 25
ND = 4                            # decimals in every committed metric
COORD_ND = 6                      # decimals for cached peak x / y / score
CHECK_TOL = 1e-4
FAR_M = 18.0                      # far band starts here (docs/detection_recall_analysis.md)

#: arm -> spec. ``size`` None means per-pano native (floored at BASE_SIZE, capped).
ARMS = {
    "r2048": {"size": BASE_SIZE, "via": None, "heatmap": BASE_HEATMAP, "min_distance": 10},
    "r3072": {"size": (3072, 6144), "via": None, "heatmap": BASE_HEATMAP, "min_distance": 10},
    "r4096": {"size": (4096, 8192), "via": None, "heatmap": BASE_HEATMAP, "min_distance": 10},
    "rnative": {"size": None, "via": None, "heatmap": BASE_HEATMAP, "min_distance": 10},
    "u4096": {"size": (4096, 8192), "via": BASE_SIZE, "heatmap": BASE_HEATMAP,
              "min_distance": 10},
    "r4096_hm1024": {"size": (4096, 8192), "via": None, "heatmap": (1024, 2048),
                     "min_distance": 20},
}
CONTROL = "r2048"
UPSAMPLE_CONTROL = "u4096"
RESOLUTION_ARMS = ("r3072", "r4096", "rnative")   # the arms the verdict rule is applied to


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested, no torch)
# --------------------------------------------------------------------------- #
def parse_hw(s):
    """'6144x12288' -> (6144, 12288)  (H x W)."""
    h, w = s.lower().split("x")
    return int(h), int(w)


def arm_input_size(arm, native_wh, native_cap=DEFAULT_NATIVE_CAP):
    """(H, W) the model sees for this arm and a pano whose native size is (w, h).

    ``rnative`` keeps the native size, never below the model's own 2048x4096 (two
    paterson panos are 3328 wide; feeding them smaller than the control would be a
    different question) and never above ``native_cap`` (a GPU-memory guard; the GSV
    splits are 16384 wide). Aspect ratio is preserved when capping."""
    spec = ARMS[arm]
    if spec["size"] is not None:
        return tuple(spec["size"])
    w, h = native_wh
    cap_h, cap_w = native_cap
    if w <= BASE_SIZE[1]:
        return BASE_SIZE
    if w > cap_w:
        return (int(round(h * cap_w / w)), cap_w)
    return (h, w)


def resize_steps(arm, native_wh, native_cap=DEFAULT_NATIVE_CAP):
    """The ordered (size, interpolation-name) resizes applied to the native image."""
    spec = ARMS[arm]
    target = arm_input_size(arm, native_wh, native_cap)
    steps = []
    if spec["via"] is not None:
        steps.append((tuple(spec["via"]), "bilinear"))
        steps.append((target, "bicubic"))
    else:
        steps.append((target, "bilinear"))
    return steps


HEADROOM_CLASSES = ("full", "partial", "none")


def headroom_class(native_w, arm_w, base_w=BASE_SIZE[1]):
    """How many of an arm's input pixels are real, for one pano.

    - ``full``: native >= arm width -- the arm only downsamples, every input pixel is
      real image content;
    - ``partial``: base < native < arm width -- the arm sees more real detail than the
      2048x4096 control but is then upsampled the last stretch (annapolis's 8000-wide
      panos at r4096);
    - ``none``: native <= 4096 -- nothing beyond what the control already sees, so any
      change is object scale against the receptive field, not resolution (richmond's
      20 and morgantown's 122 4096-wide panos).

    The plan's two-way split (native > arm width vs not) is ``full`` vs
    ``partial + none``; the three-way split keeps annapolis out of the null class."""
    if native_w <= base_w:
        return "none"
    return "full" if native_w >= arm_w else "partial"


def native_sizes_from_records(city, repo=REPO):
    """{pano_id: (width, height)} from the bundle's records.jsonl."""
    out = {}
    with open(os.path.join(repo, "benchmark", city, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["pano"]["panorama_id"]] = (int(r["pano"]["width"]),
                                                  int(r["pano"]["height"]))
    return out


def rnd(v, nd=ND):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(float(v), nd)


def write_json(path, obj):
    """LF-pinned, sorted-free (insertion order), rounded by the caller."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def cache_path(cache_root, arm, city):
    return os.path.join(cache_root, arm, f"{city}.json")


def write_arm_cache(path, city, panos, meta):
    """op_cache schema plus per-pano ``native`` [w, h], ``input`` [H, W], ``fp16``.

    ``operating_point_curve.read_cache`` ignores the extra keys, so every existing
    reader works on these files unchanged."""
    payload = {"city": city, "meta": meta, "panos": [
        {"pano": p["pano"],
         "preds": [[round(x, COORD_ND), round(y, COORD_ND), round(s, COORD_ND)]
                   for (x, y, s) in p["preds"]],
         "gt": _gt_to_json(p["gt"]),
         "native": list(p["native"]), "input": list(p["input"]), "fp16": bool(p["fp16"])}
        for p in panos]}
    write_json(path, payload)


def read_arm_cache(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    panos = [{"pano": p["pano"], "preds": [tuple(t) for t in p["preds"]],
              "gt": _gt_from_json(p["gt"]),
              "native": tuple(p.get("native") or ()), "input": tuple(p.get("input") or ()),
              "fp16": p.get("fp16")}
             for p in payload["panos"]]
    return panos, payload.get("meta", {})


def usage_rows(arm_stats, decode_s, host, gpus, cities, started, extra_note=None):
    """One ``paid: false`` usage_log row per arm plus one for the shared JPEG decode.

    ``arm_stats`` = {arm: {"elapsed_s", "panos_scored", "fp16"}}. makelab2 has no Slurm,
    so this is the only ledger the GPU time can go in (docs/compute_cost.md)."""
    rows = []
    hw = {"host": host, "gpus": gpus}
    n_total = 0
    for arm, st in arm_stats.items():
        n = st["panos_scored"]
        n_total = max(n_total, n)
        rows.append({
            "ts": started, "bundle": ",".join(cities), "label": f"input-res-25:{arm}",
            "provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
            "panos_scored": n, "elapsed_s": round(st["elapsed_s"], 3),
            "s_per_pano": round(st["elapsed_s"] / n, 4) if n else None,
            "hardware": hw, "status": "ok",
            "what": (f"input_res_sweep_25.py extract, arm {arm} (resize + forward + peak "
                     f"extraction; native decode is the separate decode row), fp16="
                     f"{st['fp16']}"),
            "est_cost_usd": 0.0, "pricing": None,
            "run_id": f"input-res-sweep-25:{arm}:{started}",
            "concurrent_with": [], "gpu_share": 1.0,
            **({"note": extra_note} if extra_note else {}),
        })
    rows.append({
        "ts": started, "bundle": ",".join(cities), "label": "input-res-25:decode",
        "provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
        "panos_scored": n_total, "elapsed_s": round(decode_s, 3),
        "s_per_pano": round(decode_s / n_total, 4) if n_total else None,
        "hardware": hw, "status": "ok",
        "what": ("input_res_sweep_25.py extract: native JPEG decode, done once per pano and "
                 "shared by every arm in the run (CPU, overlapped with GPU work by a "
                 "prefetch thread, so this is not additive wall-clock)"),
        "est_cost_usd": 0.0, "pricing": None,
        "run_id": f"input-res-sweep-25:decode:{started}",
        "concurrent_with": [], "gpu_share": 1.0,
        **({"note": extra_note} if extra_note else {}),
    })
    return rows


# --------------------------------------------------------------------------- #
# extract (GPU)
# --------------------------------------------------------------------------- #
def _interp(name):
    from torchvision import transforms
    return {"bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC}[name]


def arm_tensor(img, arm, native_cap=DEFAULT_NATIVE_CAP):
    """PIL image (native) -> normalized (3, H, W) tensor for ``arm``.

    r2048 goes through ``threshold_sweep.PRE`` itself, so the control is the committed
    instrument by construction rather than by a copy of it."""
    from torchvision import transforms
    import threshold_sweep as ts
    if arm == CONTROL:
        return ts.PRE(img)
    x = img
    for size, interp in resize_steps(arm, img.size, native_cap):
        x = transforms.Resize(size, interpolation=_interp(interp))(x)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(x)


def load_models(device, arms):
    """{heatmap_size: model}. The (1024, 2048) head loads the released (512, 1024)
    weights strictly: Upsample has no parameters, so no key depends on its size."""
    import threshold_sweep as ts
    from rampnet.model import KeypointModel
    base = ts.load_model()
    models = {BASE_HEATMAP: base.to(device)}
    for arm in arms:
        hm = tuple(ARMS[arm]["heatmap"])
        if hm not in models:
            m = KeypointModel(heatmap_size=hm)
            m.load_state_dict(base.state_dict(), strict=True)
            models[hm] = m.eval().to(device)
    return models


def _forward(model, t, device, fp16):
    import torch
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        if fp16 and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(t)
        else:
            h = model(t)
    out = h.squeeze().float().cpu().numpy()
    del t, h
    return out


def _prefetch(paths):
    """Decode native jpgs one ahead in a thread (PIL releases the GIL while decoding)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None   # 16384x8192 = 134 MP trips PIL's bomb guard
    q = Queue(maxsize=2)

    def work():
        for p in paths:
            t0 = time.perf_counter()
            try:
                img = Image.open(p).convert("RGB")
                q.put((p, img, time.perf_counter() - t0, None))
            except Exception as e:  # noqa: BLE001 -- surfaced to the caller
                q.put((p, None, time.perf_counter() - t0, e))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def cmd_extract(args):
    import torch
    import threshold_sweep as ts

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {', '.join(ARMS)}")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if args.usage_log.lower() == "none" and not args.allow_unrecorded_spend:
        raise SystemExit("--usage-log none drops the GPU-time record; pass "
                         "--allow-unrecorded-spend if that is really intended")
    native_cap = parse_hw(args.native_cap)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, arms)
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda" else [])
    host = socket.getfqdn()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fp16 = {a: args.fp16 == "on" for a in arms}
    rsq = radius_sq_for()
    print(f"device={device} gpus={gpus} arms={arms} native_cap={native_cap} "
          f"torch={torch.__version__}", flush=True)

    arm_stats = {a: {"elapsed_s": 0.0, "panos_scored": 0, "fp16": False} for a in arms}
    decode_s = 0.0
    done_cities = []
    for city in cities:
        todo = [a for a in arms
                if args.force or not os.path.exists(cache_path(args.cache_root, a, city))]
        if not todo:
            print(f"{city}: every requested arm cached -> skip", flush=True)
            continue
        gts, _ = bundle_ground_truths(city)
        panos_dir = os.path.join(args.panos_root, "benchmark", city, "panos")
        recs = native_sizes_from_records(city)
        pids = list(gts)
        if args.limit:
            pids = pids[:args.limit]
        paths = [os.path.join(panos_dir, f"{pid}.jpg") for pid in pids]
        results = {a: [] for a in todo}
        for i, (pid, (path, img, dec_s, err)) in enumerate(zip(pids, _prefetch(paths)), 1):
            if err is not None:
                raise SystemExit(f"{path}: {err}")
            decode_s += dec_s
            native = img.size
            if tuple(native) != recs[pid]:
                print(f"  WARNING {city}/{pid}: jpg is {native}, records.jsonl says "
                      f"{recs[pid]}", flush=True)
            for a in todo:
                spec = ARMS[a]
                t0 = time.perf_counter()
                size = arm_input_size(a, native, native_cap)
                model = models[tuple(spec["heatmap"])]
                while True:
                    try:
                        t = arm_tensor(img, a, native_cap)
                        h = _forward(model, t, device, fp16[a])
                        break
                    except torch.cuda.OutOfMemoryError:
                        if fp16[a] or args.fp16 == "off":
                            raise SystemExit(
                                f"{city}/{pid} arm {a} input {size}: OOM even with fp16 "
                                "autocast -- lower --native-cap and record it (tiling is "
                                "deliberately not implemented: it changes context)")
                        torch.cuda.empty_cache()
                        fp16[a] = True
                        print(f"  {a}: OOM at {size} -> fp16 autocast from here on",
                              flush=True)
                preds = ts.peaks_to_dets(h, SCORE_FLOOR, spec["min_distance"])
                arm_stats[a]["elapsed_s"] += time.perf_counter() - t0
                arm_stats[a]["panos_scored"] += 1
                arm_stats[a]["fp16"] = arm_stats[a]["fp16"] or fp16[a]
                results[a].append({"pano": pid, "preds": preds, "gt": gts[pid],
                                   "native": native, "input": size, "fp16": fp16[a]})
                del t, h
            del img
            if i % 25 == 0:
                print(f"  {city}: {i}/{len(pids)}", flush=True)
        for a in todo:
            ps = results[a]
            meta = {"arm": a, "arm_spec": {k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in ARMS[a].items()},
                    "heatmap_size": list(ARMS[a]["heatmap"]),
                    "score_floor": SCORE_FLOOR, "min_distance": ARMS[a]["min_distance"],
                    "radius_normalized": 0.022, "native_cap": list(native_cap),
                    "fp16": any(p["fp16"] for p in ps), "tta": False, "n_panos": len(ps),
                    "model": "projectsidewalk/rampnet-model", "device": device.type,
                    "gpus": gpus, "torch": torch.__version__}
            write_arm_cache(cache_path(args.cache_root, a, city), city, ps, meta)
            # sanity print at the two operating points
            s30 = _score_at(ps, 0.30, rsq)
            print(f"{city} {a}: {len(ps)} panos  P/R/F1@0.30 = {s30.precision:.3f}/"
                  f"{s30.recall:.3f}/{s30.f1:.3f}", flush=True)
        done_cities.append(city)
        print(f"{city} done; elapsed so far "
              + ", ".join(f"{a}={arm_stats[a]['elapsed_s']:.0f}s" for a in arms)
              + f", decode={decode_s:.0f}s"
              + (f", peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                 if device.type == "cuda" else ""), flush=True)

    if args.usage_log.lower() != "none" and done_cities:
        rows = usage_rows({a: s for a, s in arm_stats.items() if s["panos_scored"]},
                          decode_s, host, gpus, done_cities, started, args.note)
        ledger.append_rows(args.usage_log, rows)
        print(f"usage_log: +{len(rows)} rows -> {args.usage_log}", flush=True)
    print("extract done", flush=True)


# --------------------------------------------------------------------------- #
# check (CPU): the instrument check
# --------------------------------------------------------------------------- #
def compare_to_op_cache(arm_panos, op_payload, tol=CHECK_TOL):
    """(n_panos_mismatched, max_abs_diff, details) between two peak lists per pano."""
    ref = {p["pano"]: p["preds"] for p in op_payload["panos"]}
    bad, max_diff, details = 0, 0.0, []
    if set(ref) != {p["pano"] for p in arm_panos}:
        return len(ref), float("inf"), ["pano sets differ"]
    for p in arm_panos:
        a = sorted(p["preds"])
        b = sorted(tuple(t) for t in ref[p["pano"]])
        if len(a) != len(b):
            bad += 1
            details.append(f"{p['pano']}: {len(a)} peaks vs {len(b)}")
            continue
        d = max((abs(u - v) for pa, pb in zip(a, b) for u, v in zip(pa, pb)), default=0.0)
        max_diff = max(max_diff, d)
        if d > tol:
            bad += 1
            details.append(f"{p['pano']}: max |diff| {d:.2e}")
    return bad, max_diff, details


def cmd_check(args):
    rsq = radius_sq_for()
    ok = True
    rows = []
    for city in SPLITS:
        cp = cache_path(args.cache_root, CONTROL, city)
        op = os.path.join(args.op_cache, f"{city}.json")
        if not os.path.exists(cp):
            print(f"{city:>20}: no r2048 cache -- NOT CHECKED")
            ok = False
            continue
        panos, _ = read_arm_cache(cp)
        recs = native_sizes_from_records(city)
        size_bad = [p["pano"] for p in panos if tuple(p["native"]) != recs[p["pano"]]]
        if size_bad:
            print(f"{city:>20}: {len(size_bad)} panos whose jpg size != records.jsonl")
        if not os.path.exists(op):
            print(f"{city:>20}: no committed op_cache -- no single-pass reference, "
                  "unchecked (reported, not failed)")
            rows.append({"city": city, "checked": False})
            continue
        with open(op, encoding="utf-8") as f:
            payload = json.load(f)
        bad, md, details = compare_to_op_cache(panos, payload)
        ref_panos = [{"pano": p["pano"], "preds": [tuple(t) for t in p["preds"]],
                      "gt": _gt_from_json(p["gt"])} for p in payload["panos"]]
        prf_ok = True
        prf = {}
        for thr in THRESHOLDS:
            a, b = _score_at(panos, thr, rsq), _score_at(ref_panos, thr, rsq)
            same = (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
            prf_ok &= same
            prf[str(thr)] = {"sweep": [a.tp, a.fp, a.fn], "op_cache": [b.tp, b.fp, b.fn],
                             "P": rnd(a.precision), "R": rnd(a.recall), "F1": rnd(a.f1)}
        city_ok = bad == 0 and prf_ok
        ok &= city_ok
        rows.append({"city": city, "checked": True, "ok": city_ok, "panos_mismatched": bad,
                     "max_abs_diff": md, "prf": prf})
        print(f"{city:>20}: {'PASS' if city_ok else 'FAIL'}  panos mismatched {bad}/"
              f"{len(panos)}  max|diff| {md:.2e}  "
              + "  ".join(f"@{t}: tp/fp/fn {v['sweep']} vs {v['op_cache']}"
                          for t, v in prf.items()))
        for d in details[:10]:
            print(f"{'':>22}{d}")
    print("INSTRUMENT CHECK:", "PASS" if ok else "FAIL")
    if args.out:
        write_json(args.out, {"tolerance": CHECK_TOL, "pass": ok, "cities": [
            {k: (rnd(v, 8) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows]})
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# report (CPU)
# --------------------------------------------------------------------------- #
def scored_from_panos(split, panos, rsq):
    """A ``benchmark_power_135.Scored`` over these panos, via its own ``score_model``.

    score_model's ``rampnet`` source reads ``records[pid]["detections"]``, so handing it
    the cache's peaks in that shape reuses the power script's matcher, its score_pano
    agreement asserts, and its bootstrap unchanged."""
    records = {p["pano"]: {"detections": [list(t) for t in p["preds"]]} for p in panos}
    gts = {p["pano"]: p["gt"] for p in panos}
    return bp.score_model(REPO, split, "rampnet", records, gts, rsq)


def gt_points_in_scored_order(panos):
    """(split-local) GT (x, y) in exactly the order ``Scored.hit_gt`` enumerates them:
    sorted pano id, recall-confirmed panos only, gt_points order."""
    by = {p["pano"]: p for p in panos}
    pts, pano_of = [], []
    for i, pid in enumerate(sorted(by)):
        gt = by[pid]["gt"]
        if gt.fn_confirmed:
            for gx, gy in gt.gt_points:
                pts.append((gx, gy))
                pano_of.append(i)
    return pts, np.asarray(pano_of, dtype=np.int64)


def flat_m(y):
    r = flat_range(y, CAM_H)
    return r


def band_of_point(y):
    """(metre band label, apparent-px band label) for a GT point at normalized y."""
    r = flat_m(y)
    if r is None:
        return "above horizon", "above horizon"
    ray = math.hypot(r, CAM_H)
    mb = bucket_of(r, M_BUCKETS)
    pb = bucket_of(apparent_px(ray), PX_BUCKETS)
    return bucket_label(*mb, "m"), bucket_label(*pb, "px")


def paired_subset_recall(hit_a, hit_b, pano_of, mask, weights, threshold):
    """Paired bootstrap difference in recall over the GT points in ``mask``.

    ``weights`` (B, n_panos); a GT point carries its pano's multiplicity."""
    w = weights[:, pano_of[mask]]
    n = w.sum(axis=1)
    ra = (w * (hit_a[mask] >= threshold)).sum(axis=1)
    rb = (w * (hit_b[mask] >= threshold)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (ra - rb) / np.where(n > 0, n, 1), np.nan)


def _ci(obs, draws):
    d = draws[~np.isnan(draws)]
    if len(d) == 0:
        return {"observed": rnd(obs), "ci_lo": None, "ci_hi": None}
    return {"observed": rnd(obs), "ci_lo": rnd(np.percentile(d, 2.5)),
            "ci_hi": rnd(np.percentile(d, 97.5))}


def _contrast(s_arm, s_ref, sizes, threshold):
    rng = np.random.default_rng(SEED)
    r = bp.observed_and_se(s_arm, sizes, threshold, rng, N_REPS, paired=s_ref)
    return {k: {"observed": rnd(v["observed"]), "ci_lo": rnd(v["ci_lo"]),
                "ci_hi": rnd(v["ci_hi"])} for k, v in r.items() if k != "max_f1"}


def verdict(d_arm, d_up):
    """Apply the pre-stated rule (plan section 1) to one split x arm at 0.30.

    ``d_arm`` = contrast arm - r2048; ``d_up`` = contrast arm - u4096 (None for u4096
    itself). Returns (label, reasons)."""
    dr, dp, df = d_arm["recall"], d_arm["precision"], d_arm["f1"]
    reasons = {
        "recall_up_ci_excludes_0": dr["ci_lo"] is not None and dr["ci_lo"] > 0,
        "precision_drop_not_larger_than_recall_gain": dp["observed"] >= -dr["observed"],
        "f1_ci_not_below_0": df["ci_hi"] is not None and df["ci_hi"] >= 0,
        "recall_gain_exceeds_u4096_ci": (d_up is not None and d_up["recall"]["ci_lo"]
                                         is not None and d_up["recall"]["ci_lo"] > 0),
    }
    if all(reasons.values()):
        label = "helps"
    elif df["ci_hi"] is not None and df["ci_hi"] < 0:
        label = "hurts"
    elif df["ci_lo"] is not None and df["ci_lo"] <= 0 <= df["ci_hi"]:
        label = "tolerates"
    else:
        label = "F1 up, rule not met"
    return label, reasons


def _metrics(panos, rsq):
    out = {}
    for thr in THRESHOLDS:
        s = _score_at(panos, thr, rsq)
        out[f"{thr:.2f}"] = {"P": rnd(s.precision), "R": rnd(s.recall), "F1": rnd(s.f1),
                             "tp": s.tp, "fp": s.fp, "fn": s.fn}
    ap = pr_curve_and_ap(panos, rsq).ap
    out["AP"] = rnd(ap)
    out["n_panos"] = len(panos)
    return out


def load_all(cache_root, arms, cities):
    data = {}
    for arm in arms:
        for city in cities:
            p = cache_path(cache_root, arm, city)
            if os.path.exists(p):
                data[(arm, city)] = read_arm_cache(p)[0]
    return data


def build_report(cache_root, arms=tuple(ARMS), cities=SPLITS):
    rsq = radius_sq_for()
    data = load_all(cache_root, arms, cities)
    arms = [a for a in arms if any((a, c) in data for c in cities)]
    cities = [c for c in cities if (CONTROL, c) in data]
    rep = {"protocol": {
        "thresholds": list(THRESHOLDS), "score_floor": SCORE_FLOOR, "n_reps": N_REPS,
        "seed": SEED, "bootstrap": "pano-level paired cluster bootstrap, stratified by "
        "split (benchmark_power_135.observed_and_se)", "cam_h_m": CAM_H,
        "far_band_m": FAR_M, "arms": {a: ARMS[a] for a in arms},
        "native_cap_hw": list(DEFAULT_NATIVE_CAP)},
        "per_split": {}, "pooled": {}, "headroom": {}, "bands": {}, "verdicts": {}}

    scored = {}
    for (arm, city), panos in data.items():
        scored[(arm, city)] = scored_from_panos(city, panos, rsq)

    # per split: metrics per arm + paired contrasts vs r2048 (and vs u4096)
    for city in cities:
        ent = {"metrics": {}, "vs_r2048": {}, "vs_u4096": {}}
        for arm in arms:
            if (arm, city) not in data:
                continue
            ent["metrics"][arm] = _metrics(data[(arm, city)], rsq)
            if arm == CONTROL:
                continue
            n = len(scored[(arm, city)].pids)
            ent["vs_r2048"][arm] = {f"{t:.2f}": _contrast(
                scored[(arm, city)], scored[(CONTROL, city)], [n], t) for t in THRESHOLDS}
            if arm in RESOLUTION_ARMS and (UPSAMPLE_CONTROL, city) in data:
                ent["vs_u4096"][arm] = {"0.30": _contrast(
                    scored[(arm, city)], scored[(UPSAMPLE_CONTROL, city)], [n], 0.30)}
        rep["per_split"][city] = ent
        rep["verdicts"][city] = {}
        for arm in RESOLUTION_ARMS + (UPSAMPLE_CONTROL,):
            if arm not in ent["vs_r2048"]:
                continue
            lab, why = verdict(ent["vs_r2048"][arm]["0.30"],
                               ent["vs_u4096"].get(arm, {}).get("0.30"))
            rep["verdicts"][city][arm] = {"verdict": lab, "criteria": why}

    # pooled
    for name, members in POOLS.items():
        members = [c for c in members if c in cities]
        if not members:
            continue
        ent = {"members": members, "metrics": {}, "vs_r2048": {}, "vs_u4096": {}}
        sizes = None
        stacked = {}
        for arm in arms:
            if not all((arm, c) in data for c in members):
                continue
            panos = [p for c in members for p in data[(arm, c)]]
            ent["metrics"][arm] = _metrics(panos, rsq)
            stacked[arm] = bp.stack([scored[(arm, c)] for c in members])
            sizes = [len(scored[(arm, c)].pids) for c in members]
        for arm in stacked:
            if arm == CONTROL:
                continue
            ent["vs_r2048"][arm] = {f"{t:.2f}": _contrast(stacked[arm], stacked[CONTROL],
                                                          sizes, t) for t in THRESHOLDS}
            if arm in RESOLUTION_ARMS and UPSAMPLE_CONTROL in stacked:
                ent["vs_u4096"][arm] = {"0.30": _contrast(
                    stacked[arm], stacked[UPSAMPLE_CONTROL], sizes, 0.30)}
        rep["pooled"][name] = ent
        rep["verdicts"][name] = {}
        for arm in RESOLUTION_ARMS + (UPSAMPLE_CONTROL,):
            if arm in ent["vs_r2048"]:
                lab, why = verdict(ent["vs_r2048"][arm]["0.30"],
                                   ent["vs_u4096"].get(arm, {}).get("0.30"))
                rep["verdicts"][name][arm] = {"verdict": lab, "criteria": why}

    # headroom classes (see headroom_class): every split pooled, per arm
    for arm in arms:
        if arm == CONTROL:
            continue
        ent = {}
        for cls in HEADROOM_CLASSES:
            sub_a, sub_c, sizes = [], [], []
            for city in cities:
                if (arm, city) not in data:
                    continue
                keep = {p["pano"] for p in data[(arm, city)]
                        if headroom_class(p["native"][0], p["input"][1]) == cls}
                if not keep:
                    continue
                pa = [p for p in data[(arm, city)] if p["pano"] in keep]
                pc = [p for p in data[(CONTROL, city)] if p["pano"] in keep]
                sub_a.append(scored_from_panos(city, pa, rsq))
                sub_c.append(scored_from_panos(city, pc, rsq))
                sizes.append(len(keep))
            if not sizes:
                ent[cls] = {"n_panos": 0}
                continue
            sa, sc = bp.stack(sub_a), bp.stack(sub_c)
            ent[cls] = {"n_panos": int(sum(sizes)),
                        "splits": {s.split: len(s.pids) for s in sub_a},
                        "vs_r2048": {"0.30": _contrast(sa, sc, sizes, 0.30)}}
        rep["headroom"][arm] = ent

    # distance / apparent-size bands (0.30), per pool, with a paired far-band CI
    for name, members in POOLS.items():
        members = [c for c in members if c in cities]
        if not members:
            continue
        ent = {"members": members, "recall_by_m": {}, "recall_by_px": {}, "far_vs_r2048": {}}
        pts, pano_of, sizes = [], [], []
        offset = 0
        for c in members:
            p, po = gt_points_in_scored_order(data[(CONTROL, c)])
            pts.extend(p)
            pano_of.append(po + offset)
            offset += len(data[(CONTROL, c)])
            sizes.append(len(data[(CONTROL, c)]))
        pano_of = np.concatenate(pano_of) if pano_of else np.zeros(0, dtype=np.int64)
        mb = [band_of_point(y)[0] for _, y in pts]
        pb = [band_of_point(y)[1] for _, y in pts]
        far = np.array([(flat_m(y) is not None and flat_m(y) >= FAR_M) for _, y in pts])
        hits = {}
        for arm in arms:
            if not all((arm, c) in data for c in members):
                continue
            s = bp.stack([scored[(arm, c)] for c in members])
            assert len(s.hit_gt) == len(pts), (name, arm)
            hits[arm] = s.hit_gt
            for key, labels in (("recall_by_m", mb), ("recall_by_px", pb)):
                tab = {}
                for lab in dict.fromkeys(labels):
                    m = np.array([x == lab for x in labels])
                    tab[lab] = {"n": int(m.sum()),
                                "recall": rnd(float((s.hit_gt[m] >= 0.30).mean()))}
                ent[key][arm] = tab
        rng = np.random.default_rng(SEED)
        w = bp.bootstrap_weights(rng, sizes, N_REPS)
        ones = np.ones((1, sum(sizes)))
        for arm in hits:
            if arm == CONTROL:
                continue
            obs = paired_subset_recall(hits[arm], hits[CONTROL], pano_of, far, ones, 0.30)[0]
            dr = paired_subset_recall(hits[arm], hits[CONTROL], pano_of, far, w, 0.30)
            ent["far_vs_r2048"][arm] = {"n_far_gt": int(far.sum()), **_ci(obs, dr)}
        rep["bands"][name] = ent
    return rep


def _fmt_d(d):
    if d["ci_lo"] is None:
        return f"{d['observed']:+.3f}"
    return f"{d['observed']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def markdown(rep):
    L = []
    arms = [a for a in ARMS if a != CONTROL]

    def contrast_table(title, entries, thr):
        L.append(f"**{title} — paired change vs r2048 at {thr} (95% CI)**\n")
        L.append("| split | arm | P | R | F1 | ΔP | ΔR | ΔF1 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, ent in entries:
            c = ent["metrics"].get(CONTROL)
            if c:
                m = c[thr]
                L.append(f"| {name} | r2048 | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} "
                         "| — | — | — |")
            for arm in arms:
                if arm not in ent["vs_r2048"]:
                    continue
                m = ent["metrics"][arm][thr]
                d = ent["vs_r2048"][arm][thr]
                L.append(f"| {name} | {arm} | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} | "
                         f"{_fmt_d(d['precision'])} | {_fmt_d(d['recall'])} | "
                         f"{_fmt_d(d['f1'])} |")
        L.append("")

    head = [(c, rep["per_split"][c]) for c in HEADLINE if c in rep["per_split"]]
    gsv = [(c, rep["per_split"][c]) for c in GSV_SPLITS if c in rep["per_split"]]
    other = [(c, rep["per_split"][c]) for c in rep["per_split"]
             if c not in HEADLINE and c not in GSV_SPLITS]
    pooled = list(rep["pooled"].items())
    for thr in ("0.30", "0.55"):
        contrast_table("Headline (Mapillary with headroom)", head, thr)
        contrast_table("GSV z5 splits", gsv, thr)
        contrast_table("Other splits", other, thr)
        contrast_table("Pooled", pooled, thr)

    L.append("**Verdict per split (rule applied at 0.30)**\n")
    L.append("| split | " + " | ".join(RESOLUTION_ARMS + (UPSAMPLE_CONTROL,)) + " |")
    L.append("|---|" + "---|" * (len(RESOLUTION_ARMS) + 1))
    for name, v in rep["verdicts"].items():
        L.append(f"| {name} | " + " | ".join(v.get(a, {}).get("verdict", "—")
                                             for a in RESOLUTION_ARMS + (UPSAMPLE_CONTROL,))
                 + " |")
    L.append("")

    L.append("**Resolution arm vs the upsample control u4096 at 0.30 (paired, 95% CI)**\n")
    L.append("| split | arm | ΔP | ΔR | ΔF1 |")
    L.append("|---|---|---|---|---|")
    for name, ent in head + gsv + other + pooled:
        for arm, d in ent["vs_u4096"].items():
            d = d["0.30"]
            L.append(f"| {name} | {arm} − u4096 | {_fmt_d(d['precision'])} | "
                     f"{_fmt_d(d['recall'])} | {_fmt_d(d['f1'])} |")
    L.append("")

    L.append("**Headroom class (all splits pooled, 0.30): full = arm only downsamples; "
             "partial = more real pixels than r2048, then upsampled; none = native <= "
             "4096 wide, nothing new**\n")
    L.append("| arm | class | panos | ΔP | ΔR | ΔF1 |")
    L.append("|---|---|---|---|---|---|")
    for arm, ent in rep["headroom"].items():
        for cls, e in ent.items():
            if not e.get("n_panos"):
                L.append(f"| {arm} | {cls} | 0 | — | — | — |")
                continue
            d = e["vs_r2048"]["0.30"]
            L.append(f"| {arm} | {cls} | {e['n_panos']} | {_fmt_d(d['precision'])} | "
                     f"{_fmt_d(d['recall'])} | {_fmt_d(d['f1'])} |")
    L.append("")

    for name, ent in rep["bands"].items():
        for key, unit in (("recall_by_m", "flat-ground range"), ("recall_by_px",
                                                                  "apparent size")):
            tabs = ent[key]
            if not tabs:
                continue
            labels = list(tabs[CONTROL].keys()) if CONTROL in tabs else []
            order = [bucket_label(lo, hi, "m" if key == "recall_by_m" else "px")
                     for lo, hi in (M_BUCKETS if key == "recall_by_m" else PX_BUCKETS)]
            labels = [lb for lb in order if lb in labels] + [lb for lb in labels
                                                               if lb not in order]
            L.append(f"**{name}: recall at 0.30 by {unit} (CAM_H {CAM_H} m; px at the "
                     "2048x4096 input)**\n")
            L.append("| band | n | " + " | ".join(tabs) + " |")
            L.append("|---|---|" + "---|" * len(tabs))
            for lb in labels:
                n = tabs[CONTROL][lb]["n"]
                L.append(f"| {lb} | {n} | " + " | ".join(
                    f"{tabs[a][lb]['recall']:.3f}" for a in tabs) + " |")
            L.append("")
        if ent["far_vs_r2048"]:
            L.append(f"**{name}: far-band (≥{FAR_M:g} m) recall change vs r2048 at 0.30**\n")
            L.append("| arm | n far GT | ΔR far |")
            L.append("|---|---|---|")
            for arm, d in ent["far_vs_r2048"].items():
                L.append(f"| {arm} | {d['n_far_gt']} | {_fmt_d(d)} |")
            L.append("")
    return "\n".join(L)


def cmd_report(args):
    rep = build_report(args.cache_root)
    write_json(args.out, rep)
    md = markdown(rep)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8", newline="") as f:
        f.write(md + "\n")
    try:
        sys.stdout.reconfigure(errors="replace")   # a cp1252 console cannot print the deltas
    except AttributeError:
        pass
    print(md)
    print(f"\n-> {args.out}\n-> {md_path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="GPU: per-arm low-floor peak caches")
    e.add_argument("--arms", default=",".join(ARMS))
    e.add_argument("--cities", default=",".join(SPLITS))
    e.add_argument("--panos-root", default=REPO,
                   help="dir holding benchmark/<city>/panos/*.jpg (native bytes)")
    e.add_argument("--cache-root", default=CACHE_ROOT)
    e.add_argument("--native-cap", default="x".join(map(str, DEFAULT_NATIVE_CAP)),
                   help="H x W cap for the rnative arm (GPU-memory guard)")
    e.add_argument("--fp16", choices=("auto", "on", "off"), default="auto",
                   help="auto = fp32, switching to fp16 autocast per arm on OOM")
    e.add_argument("--force", action="store_true")
    e.add_argument("--limit", type=int, default=0, help="smoke test: first N panos/city")
    e.add_argument("--usage-log", default=USAGE_LOG,
                   help="ledger to append the paid:false GPU-time rows to; 'none' to skip")
    e.add_argument("--allow-unrecorded-spend", action="store_true")
    e.add_argument("--note", default=None, help="free-text note carried on usage rows")
    c = sub.add_parser("check", help="CPU: r2048 must reproduce analysis_out/op_cache")
    c.add_argument("--cache-root", default=CACHE_ROOT)
    c.add_argument("--op-cache", default=OP_CACHE)
    c.add_argument("--out", default=os.path.join(OUT_DIR, "instrument_check.json"))
    r = sub.add_parser("report", help="CPU: tables + paired bootstrap")
    r.add_argument("--cache-root", default=CACHE_ROOT)
    r.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        return cmd_extract(args) or 0
    if args.cmd == "check":
        return cmd_check(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
