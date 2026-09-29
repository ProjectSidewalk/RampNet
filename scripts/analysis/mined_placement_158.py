"""Image-based placement of MINED targets (#158 phase 2), through the #48 harness's arms.

#158 mines a training target wherever a strong fused site stands near a pano that did not
detect it. Step 2 placed that target by flat-ground projection of the site. Phase 2 asks
whether the #48 placement arms (``mapa_posed_pair``, ``roma``, ``roma_local``) place it
better. The labeler's ``scripts/mined_precision.py`` decides that: it adjudicates whatever
pixel this script writes against the benchmark verdicts, paired candidate by candidate with
the flat run. This script never reads a verdict.

It reuses the harness rather than copying it: the same view cutter (``H._cut_pano``), the
same ``Context`` and ``run_arm`` (answer columns stripped, detections withheld), the same
registered arm functions, and a corner manifest in ``_mv3d``'s format so the multi-view
arms find their pose priors where they expect them.

**What a "pair" is here.** One row per mined candidate (city, site_id, target pano):

* source = the view the labeler's SOURCE RULE picked (``sources.csv``, written by
  ``mined_precision.py --emit-sources``): the site's operational member from another pano
  whose camera is nearest the target camera. Its detection pixel is the click.
* other  = the target pano, its view centred on the step-2 flat projection of the site
  (``proj_x`` / ``proj_y``: the prior the miner has).

There is no answer column at all: the pair list has no ``ref_*``. ``ramp_uid`` is
``<city>:<site_id>:<target pano>``, one "corner" per pair, holding just [src, oth].

**Pose priors** (for the posed MapAnything arm) are exactly ``_mv3d.build_manifest``'s:
position and heading from ``fuse_sites.pano_pose(p, 'off')``, height from the labeler's
``auto`` resolver, flat; ENU about the source camera.

Usage:
    # 1. pairs + manifest (desktop CPU; the labeler checkout and the FROZEN runs of step 2)
    python scripts/analysis/mined_placement_158.py build \\
        --sources-root ../labeler-wt/mined158/docs/figures/mined-precision/data/frozen/perrig_perpano \\
        --labeler-root ../labeler-wt/mined158 --runs-root FROZEN_RUNS
    # 2. views (makelab2 CPU, the native-res archive)
    python scripts/analysis/mined_placement_158.py cut-views \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out VIEWS
    # 3. an arm (GPU)
    python scripts/analysis/mined_placement_158.py predict --arm mapa_posed_pair --views VIEWS
    # -> analysis_out/mined_placement_158/predictions/<arm>.jsonl, the labeler's --placement
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import crossview_align_48 as H  # noqa: E402

CITIES = H.CITIES
OUT = os.path.join(H.OUT_ROOT, "mined_placement_158")
PAIRS_CSV = os.path.join(OUT, "pairs.csv")
KEYS_CSV = os.path.join(OUT, "keys.csv")
MANIFEST = os.path.join(OUT, "corners.json")
PRED_DIR = os.path.join(OUT, "predictions")
#: the harness's pair columns without the answer ones (there is no answer here)
COLUMNS = [c for c in H.PAIR_COLUMNS if not c.startswith("ref_")]
#: arms this script is meant to drive; any registered arm runs, these are the pre-stated ones
PLANNED_ARMS = ("mapa_posed_pair", "roma", "roma_local")


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_sources(root, cities=CITIES):
    rows = []
    for city in cities:
        path = os.path.join(root, city, "sources.csv")
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["city"] != city:
                    raise SystemExit(f"{path}: row for {r['city']}")
                rows.append(r)
    keys = [(r["city"], int(r["site_id"]), r["pano_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate (city, site_id, pano_id) in sources.csv")
    return rows


def build_pairs(sources, date_of):
    """Harness-shaped pair rows (no answer columns) and the pair_id -> candidate key map.
    ``date_of(city, pano_id)`` gives the capture date string."""
    pairs, keys = [], []
    ordered = sorted(sources, key=lambda r: (CITIES.index(r["city"]), int(r["site_id"]),
                                             r["pano_id"]))
    for i, r in enumerate(ordered):
        pid = f"m{i:03d}"
        sd, od = date_of(r["city"], r["src_pano"]), date_of(r["city"], r["pano_id"])
        pairs.append({
            "pair_id": pid, "city": r["city"], "imagery": H.IMAGERY[r["city"]],
            "ramp_uid": f"{r['city']}:{r['site_id']}:{r['pano_id']}",
            "src_pano": r["src_pano"], "src_x": float(r["src_x"]), "src_y": float(r["src_y"]),
            "src_range_m": float(r["src_range_m"]), "src_date": sd,
            "oth_pano": r["pano_id"], "oth_date": od, "same_date": int(sd[:7] == od[:7]),
            "oth_range_m": float(r["oth_range_m"]), "baseline_m": float(r["baseline_m"]),
            "proj_x": float(r["proj_x"]), "proj_y": float(r["proj_y"])})
        keys.append({"pair_id": pid, "city": r["city"], "site_id": int(r["site_id"]),
                     "pano_id": r["pano_id"], "src_det_index": int(r["src_det_index"]),
                     "src_conf": float(r["src_conf"])})
    return pairs, keys


def cmd_build(args):
    """pairs.csv, keys.csv and corners.json from the labeler's sources.csv."""
    from pathlib import Path
    import multiview_evidence_48 as mv
    L = mv.import_labeler(args.labeler_root)
    sources = read_sources(args.sources_root)
    slim, prov = {}, {}
    for city in CITIES:
        want = {p for r in sources if r["city"] == city for p in (r["src_pano"], r["pano_id"])}
        results = Path(args.runs_root) / city / "results.jsonl"
        idx = Path(args.runs_root) / city / "depth" / "index.csv"
        panos, _, height, auto = L.fs.load_at_height(
            results, L.fs.HEIGHT_AUTO, depth_index=idx if idx.exists() else None)
        got = {p.pano_id: p for p in panos if p.pano_id in want}
        if want - set(got):
            raise SystemExit(f"{city}: {len(want - set(got))} panos not in {results}")
        slim[city] = (got, height)
        prov[city] = {"results_sha256": _sha256(results), "height": str(height),
                      "auto": (auto or {}).get("resolved")}
    pairs, keys = build_pairs(
        sources, lambda c, p: slim[c][0][p].capture_date or "")

    corners = []
    for p in pairs:
        got, height = slim[p["city"]]
        pose_src = L.fs.pano_pose(got[p["src_pano"]], "off")
        frame = L.geo.LocalFrame(pose_src.lat, pose_src.lng)

        def cam(pid):
            # _mv3d.build_manifest's cam(), verbatim in effect
            pose = L.fs.pano_pose(got[pid], "off")
            e, n = frame.to_enu(pose.lat, pose.lng)
            h = L.geo.camera_height_for(pose, camera_height=height)[0]
            return {"pano": pid, "e": e, "n": n, "h": h, "heading": pose.heading_deg,
                    "date": got[pid].capture_date or "", "source": pose.source}

        views = [dict(cam(p["src_pano"]), role="src", view=f"{p['pair_id']}_src.jpg",
                      cx=p["src_x"], cy=p["src_y"]),
                 dict(cam(p["oth_pano"]), role="oth", view=f"{p['pair_id']}_oth.jpg",
                      cx=p["proj_x"], cy=p["proj_y"], pair_id=p["pair_id"])]
        corners.append({"ramp_uid": p["ramp_uid"], "city": p["city"], "imagery": p["imagery"],
                        "src_pano": p["src_pano"], "src_x": p["src_x"], "src_y": p["src_y"],
                        "origin": [pose_src.lat, pose_src.lng], "pairs": [p["pair_id"]],
                        "views": views})
    os.makedirs(OUT, exist_ok=True)
    H.write_rows(PAIRS_CSV, pairs, COLUMNS)
    with open(KEYS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(keys[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(keys)
    manifest = {"pairs_sha256": _sha256(PAIRS_CSV), "view": [H.VIEW_W, H.VIEW_H, H.HFOV_DEG],
                "pose_prior": "fuse_sites.pano_pose(p, 'off'); height by HEIGHT_AUTO",
                "labeler": L.prov, "cities": prov,
                "sources": {c: _sha256(os.path.join(args.sources_root, c, "sources.csv"))
                            for c in CITIES},
                "corners": corners}
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(H.rnd(manifest, 8), indent=1, sort_keys=True) + "\n")
    by = defaultdict(int)
    for p in pairs:
        by[p["city"]] += 1
    print(f"{len(pairs)} pairs ({dict(by)}) -> {PAIRS_CSV}, {MANIFEST}")


def read_pairs(path=None):
    """pairs.csv with the harness's numeric columns typed (H.read_rows expects the answer
    columns, which this list does not have)."""
    with open(path or PAIRS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and any(k.startswith("ref_") for k in rows[0]):
        raise SystemExit(f"{PAIRS_CSV} carries answer columns")
    for r in rows:
        for c in H.FLOAT_COLS & set(r):
            r[c] = float(r[c])
        r["same_date"] = int(r["same_date"])
    return rows


def cmd_cut_views(args):
    """The harness's cut-views on this pair list (source view on the click, target view on
    the step-2 projection)."""
    from multiprocessing import Pool
    pairs = read_pairs()
    os.makedirs(args.out, exist_ok=True)
    jobs = defaultdict(list)
    for r in pairs:
        jobs[(r["city"], r["src_pano"])].append((f"{r['pair_id']}_src.jpg", r["src_x"], r["src_y"]))
        jobs[(r["city"], r["oth_pano"])].append((f"{r['pair_id']}_oth.jpg", r["proj_x"], r["proj_y"]))
    work = [(os.path.join(args.archive_root, c, "panos", f"{p}.jpg"), items, args.out)
            for (c, p), items in sorted(jobs.items())]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = [x for chunk in pool.imap_unordered(H._cut_pano, work) for x in chunk]
    miss = [p for s, p in res if s == "missing"]
    print(f"{sum(1 for s, _ in res if s == 'ok')} views from {len(work)} panos in "
          f"{time.time() - t0:.1f} s; {len(miss)} panos missing")
    for m in miss:
        print("  missing", m)


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    if m["pairs_sha256"] != _sha256(PAIRS_CSV):
        raise SystemExit("corners.json was built on a different pairs.csv")
    return m


def cmd_predict(args):
    """Run one registered #48 arm over the mined pairs; write predictions/<arm>.jsonl (one
    row per candidate, keyed by city / site_id / pano_id, x / y null on fallback) and
    <arm>.meta.json."""
    registry = H.load_arms()
    if args.arm not in registry:
        raise SystemExit(f"unknown arm {args.arm!r}")
    arm = registry[args.arm]
    pairs = read_pairs()
    if args.cities:
        pairs = [p for p in pairs if p["city"] in args.cities]
    with open(KEYS_CSV, encoding="utf-8", newline="") as f:
        keys = {r["pair_id"]: r for r in csv.DictReader(f)}
    ctx = H.Context(args, pairs)
    # the multi-view arms find their corners here (_mv3d.corner_for), keyed by ramp_uid
    ctx.cache["mv3d_manifest"] = {c["ramp_uid"]: c for c in load_manifest()["corners"]}
    t0 = time.time()
    rows, errors = H.run_arm(arm, pairs, ctx)
    elapsed = time.time() - t0
    os.makedirs(PRED_DIR, exist_ok=True)
    name = args.arm + (f"__{'_'.join(args.cities)}" if args.cities else "")
    pred = os.path.join(PRED_DIR, f"{name}.jsonl")
    with open(pred, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            k = keys[r["pair_id"]]
            out = {"city": k["city"], "site_id": int(k["site_id"]), "pano_id": k["pano_id"],
                   **r}
            f.write(json.dumps(H.rnd(out, 6), sort_keys=True) + "\n")
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    meta = {"arm": args.arm, "description": arm.description, "config": arm.config,
            "pairs_sha256": _sha256(PAIRS_CSV), "manifest_sha256": _sha256(MANIFEST),
            "pairs": len(pairs), "cities": args.cities or list(CITIES),
            "elapsed_s": round(elapsed, 2), "host": platform.node(), "gpu_visible": gpu,
            "missing_inputs": errors, "fallback": sum(1 for r in rows if r["x"] is None),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": H._versions(), "pre_specified_for_158": args.arm in PLANNED_ARMS}
    H.write_json(pred.replace(".jsonl", ".meta.json"), meta)
    print(f"{name}: {len(rows)} pairs in {elapsed:.1f} s, fallback {meta['fallback']}, "
          f"missing inputs {errors} -> {pred}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources-root", required=True)
    b.add_argument("--labeler-root", required=True)
    b.add_argument("--runs-root", required=True)
    c = sub.add_parser("cut-views")
    c.add_argument("--archive-root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("predict")
    p.add_argument("--arm", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--labeler-root", default=None)
    p.add_argument("--runs-root", default=None)
    p.add_argument("--results-root", default=None)
    args = ap.parse_args(argv)
    {"build": cmd_build, "cut-views": cmd_cut_views, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
