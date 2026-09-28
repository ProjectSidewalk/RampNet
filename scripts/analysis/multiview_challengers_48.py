"""Does multi-view agreement narrow the RampNet-vs-challenger gap? (#48, Richmond)

#48 claims that cross-capture agreement "would kill much of the chat-VLMs'
false-positive flood" and so could narrow the RampNet-vs-VLM gap, making "best
single-image model" and "best model in a multi-view system" different rankings. That
was never measured, because every benchmark bundle is thinned to 30 m so no ramp is seen
twice. This measures it on Richmond with the free challengers only (Jon's scope
decision: no paid API legs):

1. ``bundle`` -- a *neighbourhood bundle*, ``benchmark/richmond_neighbourhood/``: every
   run pano whose camera is within 20 m of a world GT ramp of eval_sites' Richmond pool,
   plus the 124 judged panos. Records are copied from the labeler run (the judged ones
   verbatim from ``benchmark/richmond/records.jsonl``) and tagged with the ramps they
   qualify for. ``bundle.json`` borrows ``benchmark/richmond``'s verdicts rather than
   copying them, so only the judged panos are scored and ``compare.py
   --detect-unjudged`` runs the rest detect-only. Imagery is linked from the labeler's
   native-res archive on makelab2 (``multiview_challengers_48.sh``), never committed.
2. ``multiview_challengers_48.sh`` (makelab2) runs the legs through ``compare.py``.
3. ``export`` (where the cache is; no labeler import) writes each leg's pano-normalized
   detections for every bundle pano to ``analysis_out/multiview_48/challengers/
   detections/``, in ``export_model_cache.py``'s format.
4. ``score`` (local; labeler import, no GPU) fuses each leg with the production
   FuseParams and scores it in world space with eval_sites' definitions -- next to its
   per-pano score on the same 124 judged panos, and RampNet's own control, which must
   first reproduce ``runs/richmond/fusion_eval/report.md`` (0.941 / 0.959).

    python scripts/analysis/multiview_challengers_48.py bundle --labeler-root ../sidewalk-auto-labeler
    bash scripts/analysis/multiview_challengers_48.sh            # on makelab2
    python scripts/analysis/multiview_challengers_48.py export --cache-dir DIR   # on makelab2
    python scripts/analysis/multiview_challengers_48.py score --labeler-root ../sidewalk-auto-labeler
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import multiview_evidence_48 as mv  # noqa: E402
from rampnet.detection_eval import aggregate, score_pano  # noqa: E402

REPO = mv.REPO
CITY = "richmond"
HOOD = "richmond_neighbourhood"
HOOD_DIR = os.path.join(REPO, "benchmark", HOOD)
HOOD_RADIUS_M = 20.0
#: The 20 m ramp neighbourhood was widened (additively: the first 1,560 panos kept their
#: cache) to every pano within 30 m -- the 25 m raycast envelope plus the 5 m match
#: radius -- of a pool ramp OR of a judged camera, so the sites precision is scored on
#: keep the captures that support or contradict them (review of PR 200). See
#: covered_sites for the part of that problem a radius cannot close.
HOOD_RADIUS_WIDE_M = 30.0
OUT = os.path.join(mv.OUT, "challengers")
DET_DIR = os.path.join(OUT, "detections")
PUBLISHED = os.path.join(REPO, "benchmark", "model_detections")

#: The free legs, their compare.py overrides, and the per-pano operating point the
#: scoreboard reports them at (docs/model_scoreboard.md, "Operating points differ by
#: model class"): open-vocab at the 0.05 export floor, YOLO at the pre-registered 0.25,
#: chat VLMs / Molmo unthresholded. Qwen3-VL-32B is the scoreboard's other Qwen but does
#: not fit makelab2's A40 in bf16 (run_laurens_open.sh), so only the 8B runs here.
LEGS = (
    {"spec": "owlv2", "slug": "google__owlv2-large-patch14-ensemble", "overrides": {},
     "headline": 0.05, "sweep": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)},
    {"spec": "gdino", "slug": "IDEA-Research__grounding-dino-base", "overrides": {},
     "headline": 0.05, "sweep": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)},
    {"spec": "qwen:Qwen/Qwen3-VL-8B-Instruct", "slug": "Qwen__Qwen3-VL-8B-Instruct",
     "overrides": {}, "headline": None, "sweep": (None,)},
    {"spec": "molmo:allenai/Molmo2-8B", "slug": "allenai__Molmo2-8B", "overrides": {},
     "headline": None, "sweep": (None,)},
    {"spec": "yolo:yolo_ckpts/y11l_pano.pt", "slug": "y11l_pano",
     "overrides": {"tiling": "none", "yolo_imgsz": 1280}, "headline": 0.25,
     "sweep": (0.05, 0.1, 0.25, 0.4, 0.55)},
    {"spec": "yolo:yolo_ckpts/y11x_pano_h200.pt", "slug": "y11x_pano_h200",
     "overrides": {"tiling": "none", "yolo_imgsz": 1280}, "headline": 0.25,
     "sweep": (0.05, 0.1, 0.25, 0.4, 0.55)},
    {"spec": "yolo:yolo_ckpts/y26_pano.pt", "slug": "y26_pano",
     "overrides": {"tiling": "none", "yolo_imgsz": 1280}, "headline": 0.25,
     "sweep": (0.05, 0.1, 0.25, 0.4, 0.55)},
)
RAMPNET = {"slug": "rampnet", "headline": 0.55, "sweep": (0.30, 0.40, 0.55, 0.70)}
KOFN = (1, 2, 3)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #


def neighbourhood(pool, cams, judged, radius=HOOD_RADIUS_M, judged_radius=None):
    """{pano_id: [ramp uids it qualifies for]} for panos within ``radius`` of a pool
    ramp or within ``judged_radius`` of a judged pano's camera, plus every judged pano.
    A pano qualifies for a ramp when it is within ``radius`` of it (maybe none)."""
    out = defaultdict(list)
    for r in pool:
        for pid, (ce, cn) in cams.items():
            if math.hypot(ce - r["e"], cn - r["n"]) <= radius:
                out[pid].append(r["uid"])
    if judged_radius:
        for j in judged:
            je, jn = cams[j]
            for pid, (ce, cn) in cams.items():
                if math.hypot(ce - je, cn - jn) <= judged_radius:
                    out.setdefault(pid, [])
    for pid in judged:
        out.setdefault(pid, [])
    return {pid: sorted(v, key=lambda u: int(u.split(":")[1])) for pid, v in out.items()}


def cmd_bundle(args):
    L = mv.import_labeler(args.labeler_root)
    runs_root = args.runs_root or os.path.join(args.labeler_root, "runs")
    verdicts, bundle_ops, run_panos, info = mv.load_city(L, CITY, runs_root)
    params = mv.fuse_params(L)
    gtw = mv.world_gt(L, CITY, verdicts, bundle_ops, run_panos, params)
    ok, got, want = mv.check_reproduction(
        CITY, gtw.result, os.path.join(runs_root, CITY, "fusion_eval", "report.md"))
    if not ok:
        raise SystemExit(f"eval_sites does not reproduce the committed report: {got} vs {want}")
    _, cams = mv.camera_index(L, gtw.frame, run_panos)
    hood = neighbourhood(gtw.pool, cams, set(gtw.judged_gt), args.radius, args.radius)
    first_pass = neighbourhood(gtw.pool, cams, set(gtw.judged_gt), HOOD_RADIUS_M)
    judged_recs = mv.read_bundle_records(CITY)
    raw = {}
    with open(os.path.join(runs_root, CITY, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["pano"]["panorama_id"] in hood:
                    raw[r["pano"]["panorama_id"]] = r
    os.makedirs(HOOD_DIR, exist_ok=True)
    rows = []
    for pid in sorted(hood):
        rec = dict(judged_recs[pid] if pid in judged_recs else raw[pid])
        rec["multiview_48"] = {"qualifies_for": hood[pid], "judged": pid in judged_recs}
        rows.append(json.dumps(rec, sort_keys=True))
    path = os.path.join(HOOD_DIR, "records.jsonl")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(rows) + "\n")
    spec = {"kind": "neighbourhood", "verdicts_from": "../richmond",
            "issue": "ProjectSidewalk/RampNet#48",
            "rule": f"every labeler-run pano whose camera is within {args.radius:g} m of a "
                    "world GT ramp in eval_sites' Richmond recall pool or of a judged pano's "
                    "camera, plus every judged pano; judged records verbatim from "
                    "benchmark/richmond",
            "radius_m": args.radius, "n_panos": len(hood),
            "first_pass": {"radius_m": HOOD_RADIUS_M, "rule": "pool ramps only",
                           "n_panos": len(first_pass)},
            "n_judged": sum(1 for p in hood if p in judged_recs),
            "pool_ramps": len(gtw.pool),
            "labeler_results_sha256": info["results_sha256"],
            "labeler": L.prov,
            "records_sha256": mv._sha256(path),
            "imagery": "native-res labeler archive on makelab2: "
                       "/projects/makeabilitylab/sidewalk-auto-labeler/runs/richmond/panos/"
                       "<pano_id>.jpg (linked into panos/ by multiview_challengers_48.sh)"}
    mv.write_json(os.path.join(HOOD_DIR, "bundle.json"), spec)
    print(f"neighbourhood bundle: {len(hood)} panos ({spec['n_judged']} judged, "
          f"{len(hood) - spec['n_judged']} detect-only) for {len(gtw.pool)} pool ramps -> {HOOD_DIR}")


# --------------------------------------------------------------------------- #
# export (runs where the cache is; imports the detector stack, never the labeler)
# --------------------------------------------------------------------------- #


def cmd_export(args):
    sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
    import compare as C
    from detectors import build_detector, parse_model_spec
    from fp_taxonomy import _compare_args
    records, _, _ = C.load_bundle(HOOD_DIR)
    cache = C.DetectionCache(args.cache_dir, enabled=True)
    os.makedirs(DET_DIR, exist_ok=True)
    for leg in LEGS:
        cargs = _compare_args(args.cache_dir)
        for k, v in leg["overrides"].items():
            setattr(cargs, k, v)
        provider, model_id = parse_model_spec(leg["spec"])
        label, det = build_detector(provider, model_id, records, cargs)
        sig = det.signature()
        dets, missing = {}, []
        for pid in sorted(records):
            pts = cache.get(C.cache_key(label, sig, HOOD, pid))
            if pts is None:
                missing.append(pid)
                continue
            # 5 decimals of normalized x is 0.1 px on an 11,000 px pano; rounding keeps
            # the committed file small and its bytes platform-independent.
            dets[pid] = [[round(p[0], 5), round(p[1], 5),
                          None if len(p) < 3 or p[2] is None else round(p[2], 4)] for p in pts]
        if not dets:
            print(f"[{label}] nothing cached; skipped")
            continue
        payload = {"model": label, "published_as": leg["slug"], "city": HOOD,
                   "signature": sig, "n_panos": len(dets), "n_uncached": len(missing),
                   "detections": dets}
        path = os.path.join(DET_DIR, f"{leg['slug']}__{HOOD}.json")
        with open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        print(f"[{label}] {len(dets)} panos, {len(missing)} uncached -> {path}")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def load_leg_detections(slug):
    path = os.path.join(DET_DIR, f"{slug}__{HOOD}.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["detections"], d


def load_published(slug):
    path = os.path.join(PUBLISHED, f"{slug}__{CITY}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["detections"]


def as_dets(points):
    """[(det_index, x, y, conf)] with a chat VLM's missing confidence read as 1.0 -- it
    has one operating point, and every one of its outputs is 'operational'."""
    return [(i, p[0], p[1], 1.0 if (len(p) < 3 or p[2] is None) else p[2])
            for i, p in enumerate(points)]


def per_pano_score(dets_by_pid, judged_gt, tier):
    scores = []
    for pid, gt in judged_gt.items():
        preds = [(x, y, c) for _, x, y, c in dets_by_pid.get(pid, ())
                 if tier is None or c >= tier - 1e-12]
        scores.append(score_pano(preds, gt))
    rep = aggregate(scores)
    return {"precision": rep.precision, "recall": rep.recall, "f1": rep.f1, "tp": rep.tp,
            "fp": rep.fp, "fn": rep.fn, "n_panos": rep.n_panos}


#: A detection reaches a site from a camera at most R_MAX from its ground point, and the
#: point is at most max_match_m (8 m) from the site it joins.
COVER_RADIUS_M = mv.R_MAX + 8.0


def covered_sites(sites, frame, run_cams_ll, bundle_pids, radius=COVER_RADIUS_M):
    """Site ids whose whole capture neighbourhood is in the bundle: every run pano (the
    full run, not the bundle) with its camera within ``radius`` of the site. Only these
    sites are scored for precision, so a site at the bundle's edge cannot look more (or
    less) supported than it would with every capture run. ``run_cams_ll``:
    [(pano_id, lat, lng)] over the full run."""
    cell = radius
    grid = defaultdict(list)
    for pid, lat, lng in run_cams_ll:
        e, n = frame.to_enu(lat, lng)
        grid[(math.floor(e / cell), math.floor(n / cell))].append((pid, e, n))
    out = set()
    for s in sites:
        kx, ky = math.floor(s["e"] / cell), math.floor(s["n"] / cell)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for pid, e, n in grid.get((kx + dx, ky + dy), ()):
                    if pid not in bundle_pids and math.hypot(e - s["e"], n - s["n"]) <= radius:
                        ok = False
        if ok:
            out.add(s["id"])
    return out


def world_eval(L, panos, dets_by_pid, tier, floor, pool_ll, judged_gt, run_cams_ll,
               bundle_pids, capture_radius=mv.R_DEFAULT):
    """Fuse one leg's detections over ``panos`` and score it in world space.

    Returns per-policy P/R (flat and k-of-n at ``tier``; precision over covered_sites
    only), the per-pano score on the judged panos at ``tier``, recall vs number of
    qualifying captures, and the sites."""
    fs = L.fs
    tier_ = 0.0 if tier is None else tier
    params = mv.fuse_params(L, min_confidence=max(tier_, 1e-9), floor=min(floor, tier_) if tier else 0.0)
    mp = [replace(p, detections=list(dets_by_pid.get(p.pano_id, ()))) for p in panos]
    sites, frame, stats = fs.fuse(mp, params)
    sd = [s for s in mv.site_dicts(sites) if (mv.site_max_conf(s) or 0) >= tier_ - 1e-12]
    pool = []
    for r in pool_ll:
        e, n = frame.to_enu(r["lat"], r["lng"])
        pool.append({"e": e, "n": n, "gt_refs": r["gt_refs"], "uid": r["uid"],
                     "source_panos": r["source_panos"]})
    pc = mv.pano_class_for(judged_gt, {pid: dets_by_pid.get(pid, ()) for pid in judged_gt}, tier_)
    scope = covered_sites(sd, frame, run_cams_ll, bundle_pids)
    policies = {}
    for k in KOFN:
        acc = {s["id"] for s in sd if mv.site_tier_panos(s, tier_) >= k}
        policies[f"kofn_{k}"] = mv.score_world(pool, sd, acc, pc, precision_scope=scope)
    # recall vs qualifying captures, world 5 m test at the tier, other views only
    ground = defaultdict(list)
    dets, _, _ = fs.project(mp, params)
    for d in dets:
        if d.conf >= tier_ - 1e-12:
            ground[d.pano_id].append((d.e, d.n, d.conf))
    # the same one-to-one per-capture claims as multiview_evidence_48.capture_table
    reach = mv.R_MAX + mv.MATCH_RADIUS_M
    ramps = [{"captures": []} for _ in pool]
    for p in panos:
        ce, cn = frame.to_enu(p.lat, p.lng)
        cand = [i for i, r in enumerate(pool) if math.hypot(ce - r["e"], cn - r["n"]) <= reach]
        if not cand:
            continue
        claims = mv.claim_by_confidence(ground.get(p.pano_id, ()),
                                        [(i, pool[i]["e"], pool[i]["n"]) for i in cand],
                                        mv.world_d2, mv.MATCH_RADIUS_M ** 2)
        for i in cand:
            dd = math.hypot(ce - pool[i]["e"], cn - pool[i]["n"])
            if dd <= mv.R_MAX:
                ramps[i]["captures"].append({
                    "pano_id": p.pano_id, "dist_m": dd, "cam_e": ce, "cam_n": cn,
                    "is_source": p.pano_id in pool[i]["source_panos"],
                    "world_conf": claims.get(i), "pixel_conf": None})
    return {
        "per_pano": per_pano_score(dets_by_pid, judged_gt, tier),
        "world": {name: mv.pr_counts(res) for name, res in policies.items()},
        "fused": {"n_sites": len(sd), "n_sites_precision_scope": len(scope), "n_multi_pano": sum(1 for s in sd if mv.site_tier_panos(s, tier_) > 1),
                  "detections_projected": stats["n_projected"], "drops": stats["drops"]},
        "recall_by_capture_count": mv.recall_by_capture_count(ramps, tier_, "world", capture_radius),
        "k_nearest_fixed": mv.recall_k_nearest(ramps, tier_, "world", capture_radius, mv.KMAX, True),
        "k_nearest_all": mv.recall_k_nearest(ramps, tier_, "world", capture_radius, mv.KMAX, False),
    }, (sd, frame)


def write_sites(path, sites, frame):
    rows = []
    for s in sites:
        lat, lng = frame.to_latlng(s["e"], s["n"])
        rows.append(json.dumps({"id": s["id"], "lat": round(lat, 7), "lng": round(lng, 7),
                                "max_conf": mv.rnd(mv.site_max_conf(s)),
                                "members": [[pid, k, mv.rnd(c)] for pid, k, c in s["members"]]},
                               sort_keys=True))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))


#: Largest |dx| or |dy| (pano-normalized) at which a re-run detection counts as the same
#: detection as the published one. ``export`` rounds x, y to 5 dp (error <= 5e-6), and the
#: published files keep full precision, so identical detections differ by at most 5e-6.
#: 1e-4 is 20x that and about 0.4 px on a 4096-px-wide pano: rounding can never exceed
#: it, and a real re-run change (a moved box) almost always does. An earlier version
#: compared both sides rounded to 4 dp, so a value on either side of a 4-dp boundary read
#: as "different" and the share measured rounding, not re-run drift (review of PR 200).
AGREE_TOL = 1e-4


def detection_agreement(mine, published, judged, tol=AGREE_TOL):
    """How closely a re-run leg reproduces the published richmond leg on the judged panos.

    Returns ``{"same_count": share of judged panos with the same number of detections,
    "same_detections": share with the same count AND, after sorting both lists by (x, y),
    every pair within ``tol`` in x and in y}``, or None when nothing is judged.
    Confidence is not compared: only whether the same boxes were found.

    >>> detection_agreement({"a": [[0.123455, 0.5, 0.9]]},
    ...                     {"a": [[0.1234549, 0.5, 0.9]]}, ["a"])
    {'same_count': 1.0, 'same_detections': 1.0}
    """
    if not judged:
        return None
    same_count = same = 0
    for pid in judged:
        a = sorted((p[0], p[1]) for p in mine.get(pid, []))
        b = sorted((p[0], p[1]) for p in published.get(pid, []))
        if len(a) != len(b):
            continue
        same_count += 1
        same += all(abs(ax - bx) <= tol and abs(ay - by) <= tol
                    for (ax, ay), (bx, by) in zip(a, b))
    n = len(judged)
    return {"same_count": same_count / n, "same_detections": same / n}


def cmd_score(args):
    L = mv.import_labeler(args.labeler_root)
    runs_root = args.runs_root or os.path.join(args.labeler_root, "runs")
    verdicts, bundle_ops, run_panos, info = mv.load_city(L, CITY, runs_root)
    params = mv.fuse_params(L)
    exact = L.es.evaluate_city(verdicts, bundle_ops, run_panos, params,
                               match_radius_m=mv.MATCH_RADIUS_M, gt_merge_m=mv.GT_MERGE_M)
    ok, got, want = mv.check_reproduction(
        CITY, exact, os.path.join(runs_root, CITY, "fusion_eval", "report.md"))
    if not ok:
        raise SystemExit(f"STOP: the RampNet control does not reproduce "
                         f"runs/richmond/fusion_eval/report.md: {got} vs {want}")
    print(f"control reproduces the committed report: recall {got['world_recall']} "
          f"precision {got['precision']}", flush=True)
    mv.apply_reinfer(CITY, runs_root, run_panos, info)
    gtw = mv.world_gt(L, CITY, verdicts, bundle_ops, run_panos, params)
    with open(os.path.join(HOOD_DIR, "bundle.json"), encoding="utf-8") as f:
        spec = json.load(f)
    hood = set()
    with open(os.path.join(HOOD_DIR, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hood.add(json.loads(line)["pano"]["panorama_id"])
    hood_panos = [p for p in run_panos if p.pano_id in hood]
    run_cams_ll = [(p.pano_id, p.lat, p.lng) for p in run_panos]
    judged = sorted(gtw.judged_gt)
    os.makedirs(OUT, exist_ok=True)
    out = {"control": {"eval_sites_full_run": got, "committed_report": want},
           "bundle": {k: spec[k] for k in ("n_panos", "n_judged", "radius_m", "pool_ramps")},
           "legs": {}}

    # RampNet: the run's own detections (>= 0.55 from results.jsonl, below it from the
    # 0.10-floor re-inference), on the full run and on the neighbourhood only.
    rn = {p.pano_id: list(p.detections) for p in run_panos}
    leg_out = {"tiers": {}}
    for tier in RAMPNET["sweep"]:
        for scope, panos in (("neighbourhood", hood_panos), ("full_run", run_panos)):
            res, (sd, frame) = world_eval(L, panos, rn, tier, mv.STORAGE_FLOOR, gtw.pool,
                                          gtw.judged_gt, run_cams_ll, hood)
            leg_out["tiers"][f"{tier:.2f}|{scope}"] = res
            if scope == "neighbourhood" and tier in (0.30, 0.55):
                write_sites(os.path.join(OUT, f"sites__rampnet__{tier:.2f}.jsonl"), sd, frame)
    out["legs"]["rampnet"] = leg_out

    for leg in LEGS:
        pts, meta = load_leg_detections(leg["slug"])
        if pts is None:
            print(f"[{leg['slug']}] no exported detections; skipped", flush=True)
            out["legs"][leg["slug"]] = {"missing": True}
            continue
        if meta["n_uncached"] and not args.allow_partial:
            raise SystemExit(f"[{leg['slug']}] {meta['n_uncached']} bundle panos have no "
                             "detections in the export; a missing pano would read as 'no "
                             "detections' and lower k-of-n support and recall. Finish the leg, "
                             "or pass --allow-partial deliberately.")
        dets = {pid: as_dets(v) for pid, v in pts.items()}
        leg_out = {"n_panos": meta["n_panos"], "n_uncached": meta["n_uncached"],
                   "signature": meta["signature"], "tiers": {}}
        pub = load_published(leg["slug"])
        if pub is not None:
            pdets = {pid: as_dets(v) for pid, v in pub.items()}
            leg_out["published_richmond"] = {
                "per_pano_at_headline": per_pano_score(pdets, gtw.judged_gt, leg["headline"]),
                "judged_panos_agreement": detection_agreement(pts, pub, judged)}
        for tier in leg["sweep"]:
            floor = min(t for t in leg["sweep"] if t is not None) if tier is not None else 0.0
            res, (sd, frame) = world_eval(L, hood_panos, dets, tier, floor, gtw.pool,
                                          gtw.judged_gt, run_cams_ll, hood)
            leg_out["tiers"]["none" if tier is None else f"{tier:.2f}"] = res
            if tier == leg["headline"]:
                write_sites(os.path.join(OUT, f"sites__{leg['slug']}__"
                                              f"{'none' if tier is None else f'{tier:.2f}'}.jsonl"),
                            sd, frame)
        out["legs"][leg["slug"]] = leg_out
        print(f"[{leg['slug']}] scored", flush=True)
    mv.write_json(os.path.join(OUT, "scores.json"), out)
    print(f"wrote {OUT}/scores.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--labeler-root", required=True)
    b.add_argument("--runs-root", default=None)
    b.add_argument("--radius", type=float, default=HOOD_RADIUS_WIDE_M)
    e = sub.add_parser("export")
    e.add_argument("--cache-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--labeler-root", required=True)
    s.add_argument("--runs-root", default=None)
    s.add_argument("--allow-partial", action="store_true",
                   help="score a leg whose export is missing some bundle panos")
    args = ap.parse_args(argv)
    {"bundle": cmd_bundle, "export": cmd_export, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
