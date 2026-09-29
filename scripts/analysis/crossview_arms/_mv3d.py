"""Shared plumbing for the multi-view 3D arms (#48): per-corner capture manifest, corner
views, camera models and the lift-and-project transfer.

The multi-view 3D arms (``sfm.py``, ``ff3d.py``) need more than the pair: every capture of
the corner, rendered toward it, with a pose prior for each. That is built in two CLI steps
and then read by the arms through ``--extra``:

    # 1. manifest (desktop CPU, labeler checkout + its runs; the same inputs as `pairs`)
    python scripts/analysis/crossview_arms/_mv3d.py manifest \\
        --labeler-root ../sidewalk-auto-labeler --runs-root ../sidewalk-auto-labeler/runs
    #    -> analysis_out/crossview_align_48/mv3d_corners.json (committed)
    # 2. corner views (makelab2 CPU, where the native-res panos are)
    python scripts/analysis/crossview_arms/_mv3d.py render \\
        --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out CORNER_VIEWS
    # 3. an arm
    python scripts/analysis/crossview_align_48.py predict --arm sfm_colmap --views VIEWS \\
        --extra corner_views=CORNER_VIEWS

**What a corner is.** One per ramp in the frozen pair list (174). Its captures are the run
panos within 25 m of the ramp's pool position (``analysis_out/multiview_48/captures_R25.csv``,
membership only: its detection-confidence columns are never read). The source view and the
pair's other views are the harness's own views (``<pair>_src.jpg``, ``<pair>_oth.jpg``);
every other capture gets a view of the same size and FOV (1024 x 768, 75 deg) aimed at the
source click raycast onto flat ground at 2.6 m -- the same aim the harness uses for the
other view, so no capture is aimed with information the projection does not have.

**Pose prior.** Position, heading and camera height as the labeler loads them
(``fuse_sites.load_at_height(..., 'auto')`` then ``fuse_sites.pano_pose(p, 'off')``): GSV
position and heading from the pano metadata, Mapillary SfM ``computed_geometry`` /
``computed_compass_angle``, flat (no pitch / roll), height by the labeler's 'auto' rule.

**World frame.** Per corner, east-north-up metres in the labeler's ``LocalFrame`` about the
source camera; the ground is z = 0 and a camera sits at z = its height.

**Nothing here reads the answer.** The manifest is built from the pair list with the
``ref_*`` columns dropped, the capture list's membership, and pano metadata.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import crossview_align_48 as H  # noqa: E402

R_CORNER_M = 25.0               # captures_R25.csv's radius
AIM_HEIGHT_M = 2.6              # the harness's projection height, used only to aim views
MANIFEST = os.path.join(H.OUT, "mv3d_corners.json")
CAPTURES_CSV = os.path.join(H.OUT_ROOT, "multiview_48", "captures_R25.csv")


# --------------------------------------------------------------------------- #
# camera geometry (pure; pano frame axes are the harness's (right, down, forward))
# --------------------------------------------------------------------------- #


def pano_to_world(heading_deg):
    """3x3 rotation taking a pano-frame vector (right, down, forward) to ENU for a flat
    camera with this heading (x = 0.5 looks along the heading, clockwise from north)."""
    p = math.radians(heading_deg)
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [-s, 0.0, c], [0.0, -1.0, 0.0]])


def view_cam_to_world(heading_deg, cx, cy):
    """Rotation from an OpenCV view camera (right, down, forward) aimed at pano (cx, cy)
    to ENU."""
    return pano_to_world(heading_deg) @ H.view_rotation(cx, cy)


def intrinsics(w=H.VIEW_W, h=H.VIEW_H, hfov_deg=H.HFOV_DEG):
    f = H.focal_px(w, hfov_deg)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def pano_ray_world(heading_deg, x, y):
    return pano_to_world(heading_deg) @ H.pano_dir(x, y)


def raycast_flat(cam, x, y, height=None):
    """ENU ground point (z = 0) hit by pano pixel (x, y) from ``cam`` (a manifest view
    dict with e, n, heading and h). None above the horizon."""
    h = cam["h"] if height is None else height
    d = pano_ray_world(cam["heading"], x, y)
    if d[2] >= -1e-9:
        return None
    t = h / -d[2]
    return np.array([cam["e"], cam["n"], h]) + t * d


def world_to_pano(cam, X, height=None):
    """Equirect (x, y) of ENU point X in ``cam``."""
    h = cam["h"] if height is None else height
    d = np.asarray(X, float) - np.array([cam["e"], cam["n"], h])
    v = pano_to_world(cam["heading"]).T @ d
    x, y = H.dir_to_pano(v)
    return float(x), float(y)


def view_pixel_to_pano(view, u, v):
    """A pixel of a manifest view (1024 x 768, 75 deg, centred on view['cx'], view['cy'])
    to equirect (x, y) in that view's pano."""
    x, y = H.view_to_pano(u, v, view["cx"], view["cy"])
    return float(x), float(y)


def cam_pose_world(view, height=None):
    """(R_cw, C) for a manifest view under its pose prior: camera-to-world rotation and
    camera centre in the corner's ENU frame."""
    h = view["h"] if height is None else height
    return (view_cam_to_world(view["heading"], view["cx"], view["cy"]),
            np.array([view["e"], view["n"], h]))


def project_cam(R_cw, C, K, X):
    """Pixel (u, v) of world point X in a pinhole camera, and whether it is in front."""
    p = R_cw.T @ (np.asarray(X, float) - C)
    if p[2] <= 1e-6:
        return None
    q = K @ (p / p[2])
    return float(q[0]), float(q[1])


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def _pairs_without_answers():
    from crossview_arms._registry import ANSWER_KEYS
    return [{k: v for k, v in r.items() if k not in ANSWER_KEYS} for r in H.read_frozen_pairs()]


def build_manifest(args):
    import multiview_evidence_48 as mv
    from pathlib import Path
    L = mv.import_labeler(args.labeler_root)
    runs_root = args.runs_root or os.path.join(args.labeler_root, "runs")
    pairs = _pairs_without_answers()
    by_ramp = defaultdict(list)
    for p in pairs:
        by_ramp[p["ramp_uid"]].append(p)
    members = defaultdict(list)
    with open(CAPTURES_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["ramp_uid"] in by_ramp:
                members[r["ramp_uid"]].append((float(r["dist_m"]), r["pano_id"],
                                               r["capture_date"]))
    corners, prov = [], {}
    for city in H.CITIES:
        want = {pid for u, ps in by_ramp.items() if ps[0]["city"] == city
                for pid in [ps[0]["src_pano"]] + [pid for _, pid, _ in members[u]]}
        results = os.path.join(mv.results_dir(city, runs_root, args.results_root),
                               "results.jsonl")
        idx = Path(runs_root) / city / "depth" / "index.csv"
        panos, _, height, auto = L.fs.load_at_height(
            Path(results), L.fs.HEIGHT_AUTO, depth_index=idx if idx.exists() else None)
        slim = {p.pano_id: p for p in panos if p.pano_id in want}
        prov[city] = {"results": os.path.basename(os.path.dirname(results)),
                      "height": height if isinstance(height, (int, float)) else str(height),
                      "auto": (auto or {}).get("resolved")}
        missing = want - set(slim)
        if missing:
            raise SystemExit(f"{city}: {len(missing)} capture panos not in {results}")
        for uid in sorted((u for u, ps in by_ramp.items() if ps[0]["city"] == city),
                          key=lambda u: int(u.split(":")[1])):
            ps = sorted(by_ramp[uid], key=lambda p: p["pair_id"])
            s0 = ps[0]
            assert all(p["src_pano"] == s0["src_pano"] and p["src_x"] == s0["src_x"]
                       and p["src_y"] == s0["src_y"] for p in ps)
            pose_src = L.fs.pano_pose(slim[s0["src_pano"]], "off")
            frame = L.geo.LocalFrame(pose_src.lat, pose_src.lng)

            def cam(pid):
                pose = L.fs.pano_pose(slim[pid], "off")
                e, n = frame.to_enu(pose.lat, pose.lng)
                h = L.geo.camera_height_for(pose, camera_height=height)[0]
                return {"pano": pid, "e": e, "n": n, "h": h, "heading": pose.heading_deg,
                        "date": slim[pid].capture_date or "", "source": pose.source}

            src = cam(s0["src_pano"])
            aim = raycast_flat(src, s0["src_x"], s0["src_y"], height=AIM_HEIGHT_M)
            views = [dict(src, role="src", view=f"{s0['pair_id']}_src.jpg",
                          cx=s0["src_x"], cy=s0["src_y"])]
            oth = {p["oth_pano"]: p for p in ps}
            for pid, p in sorted(oth.items()):
                views.append(dict(cam(pid), role="oth", view=f"{p['pair_id']}_oth.jpg",
                                  cx=p["proj_x"], cy=p["proj_y"], pair_id=p["pair_id"]))
            for d, pid, _ in sorted(members[uid]):
                if pid == s0["src_pano"] or pid in oth:
                    continue
                c = cam(pid)
                cx, cy = world_to_pano(c, aim, height=AIM_HEIGHT_M) if aim is not None \
                    else (0.5, 0.6)
                views.append(dict(c, role="extra", view=f"{uid.replace(':', '_')}_{pid}.jpg",
                                  cx=cx, cy=cy, dist_m=d))
            corners.append({"ramp_uid": uid, "city": city, "imagery": s0["imagery"],
                            "src_pano": s0["src_pano"], "src_x": s0["src_x"],
                            "src_y": s0["src_y"], "origin": [pose_src.lat, pose_src.lng],
                            "aim_enu": None if aim is None else [float(a) for a in aim],
                            "pairs": [p["pair_id"] for p in ps], "views": views})
        print(f"{city}: {sum(c['city'] == city for c in corners)} corners, "
              f"{sum(len(c['views']) for c in corners if c['city'] == city)} views", flush=True)
    # instrument check: the flat 2.6 m transfer in this frame must reproduce proj_x / proj_y
    worst = 0.0
    pmap = {p["pair_id"]: p for p in pairs}
    for c in corners:
        src = c["views"][0]
        g = raycast_flat(src, c["src_x"], c["src_y"], height=AIM_HEIGHT_M)
        for v in c["views"]:
            if v["role"] == "oth":
                x, y = world_to_pano(v, g, height=AIM_HEIGHT_M)
                p = pmap[v["pair_id"]]
                worst = max(worst, float(H.angular_error_deg(x, y, p["proj_x"], p["proj_y"])))
    print(f"instrument check: flat 2.6 m transfer vs proj_x/proj_y, worst {worst:.4f} deg")
    out = {"pairs_sha256": H.PAIRS_SHA256, "radius_m": R_CORNER_M, "aim_height_m": AIM_HEIGHT_M,
           "view": [H.VIEW_W, H.VIEW_H, H.HFOV_DEG], "labeler": L.prov, "cities": prov,
           "instrument_check_worst_deg": worst, "corners": corners}
    # 8 decimals, not write_json's 4: a 1e-4 rounding of a pixel coordinate is 0.036 deg,
    # which at grazing depression moves a flat-ground transfer by up to ~0.3 deg
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(H.rnd(out, 8), indent=1, sort_keys=True) + "\n")
    print(f"{len(corners)} corners, {sum(len(c['views']) for c in corners)} views -> {args.out}")


def load_manifest(path=None):
    with open(path or MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("pairs_sha256") != H.PAIRS_SHA256:
        raise SystemExit("corner manifest was built on a different pair list")
    return m


# --------------------------------------------------------------------------- #
# render (makelab2)
# --------------------------------------------------------------------------- #


def cmd_render(args):
    from multiprocessing import Pool
    m = load_manifest(args.manifest)
    jobs = defaultdict(list)
    for c in m["corners"]:
        for v in c["views"]:
            if v["role"] == "extra":
                jobs[(c["city"], v["pano"])].append((v["view"], v["cx"], v["cy"]))
    os.makedirs(args.out, exist_ok=True)
    work = [(os.path.join(args.archive_root, c, "panos", f"{p}.jpg"), items, args.out)
            for (c, p), items in sorted(jobs.items())]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = [x for chunk in pool.imap_unordered(H._cut_pano, work) for x in chunk]
    miss = [p for s, p in res if s == "missing"]
    print(f"{sum(1 for s, _ in res if s == 'ok')} corner views from {len(work)} panos in "
          f"{time.time() - t0:.1f} s; {len(miss)} panos missing")
    for p in miss[:20]:
        print("  missing", p)


# --------------------------------------------------------------------------- #
# arm-side helpers
# --------------------------------------------------------------------------- #


def extra(ctx, key, default=None):
    for kv in getattr(ctx.args, "extra", None) or []:
        k, _, v = kv.partition("=")
        if k == key:
            return v
    return os.environ.get(f"MV3D_{key.upper()}", default)


def corner_for(ctx, pair):
    """The manifest corner holding this pair (cached)."""
    if "mv3d_manifest" not in ctx.cache:
        m = load_manifest(extra(ctx, "manifest"))
        ctx.cache["mv3d_manifest"] = {c["ramp_uid"]: c for c in m["corners"]}
    return ctx.cache["mv3d_manifest"][pair["ramp_uid"]]


def view_path(ctx, view):
    """Harness views live in --views, corner extras in --extra corner_views=DIR."""
    if view["role"] == "extra":
        d = extra(ctx, "corner_views")
        if not d:
            raise SystemExit("this arm needs --extra corner_views=DIR (from `_mv3d.py render`)")
        return os.path.join(d, view["view"])
    return os.path.join(ctx.args.views, view["view"])


def read_view(ctx, view):
    import cv2
    img = cv2.imread(view_path(ctx, view), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(view_path(ctx, view))
    return img


def select_views(corner, pair, max_views):
    """The source view, this pair's other view, then the other captures nearest first,
    up to ``max_views`` (None = all)."""
    views = corner["views"]
    src = views[0]
    oth = next(v for v in views if v["role"] == "oth" and v["pair_id"] == pair["pair_id"])
    rest = [v for v in views[1:] if v is not oth]
    rest.sort(key=lambda v: (v.get("dist_m", 0.0), v["pano"]))    # nearest the ramp first
    chosen = [src, oth] + rest
    return chosen if max_views is None else chosen[:max_views]


def poseonly_transfer(R_s, C_s, R_o, C_o, vs, vo, u=H.VIEW_W / 2.0, v=H.VIEW_H / 2.0):
    """Today's flat-ground transfer with only the other camera re-posed by a reconstruction.

    ``(R_s, C_s)``, ``(R_o, C_o)``: camera-to-world rotation and centre of the source and
    other view in ANY reconstruction frame (scale-free). The source camera keeps its prior
    pose; the other camera gets the reconstruction's relative rotation and baseline
    direction, with the prior's baseline length. The click is raycast onto flat ground at
    the source's 'auto' height and projected into the re-posed other camera. Isolates what a
    reconstruction's POSE is worth, independent of its depth. Returns the arm dict or None.
    """
    Rsp, Csp = cam_pose_world(vs)
    Rop, Cop = cam_pose_world(vo)
    R_new = Rsp @ R_s.T @ R_o
    b = R_s.T @ (C_o - C_s)
    nb = np.linalg.norm(b)
    if nb < 1e-12:
        return None
    C_new = Csp + np.linalg.norm(Cop - Csp) * (Rsp @ (b / nb))
    ray = Rsp @ np.linalg.solve(intrinsics(), np.array([u, v, 1.0]))
    if ray[2] >= -1e-9:
        return None
    X = Csp + (Csp[2] / -ray[2]) * ray
    uv = project_cam(R_new, C_new, intrinsics(), X)
    if uv is None:
        return None
    x, y = view_pixel_to_pano(vo, *uv)
    return {"x": x, "y": y, "u": uv[0], "v": uv[1],
            "oth_shift_m": float(np.linalg.norm(C_new - Cop))}


def pixel_in_view(u, v, margin=0):
    return margin <= u < H.VIEW_W - margin and margin <= v < H.VIEW_H - margin


PILOT_PER_CITY = 6


def pilot_pairs(pairs, per_city=PILOT_PER_CITY):
    """The pilot subset: the first ``per_city`` pairs of each city in pair-id order (30)."""
    out, seen = [], defaultdict(int)
    for p in sorted(pairs, key=lambda p: p["pair_id"]):
        if seen[p["city"]] < per_city:
            out.append(p)
            seen[p["city"]] += 1
    return out


def cmd_pilot(args):
    """Run one arm on the pilot subset (answers hidden, as in `predict`), write
    mv3d_pilot_<arm>.jsonl, then score it against the reference next to the projection and
    the committed proj_height_auto. A pilot decides whether an arm is worth all 300."""
    import platform
    registry = H.load_arms()
    arm = registry[args.arm]
    pairs = H.read_frozen_pairs()
    sub = pilot_pairs(pairs, args.per_city)
    ctx = H.Context(args, pairs)
    t0 = time.time()
    rows, errors = H.run_arm(arm, sub, ctx)
    el = time.time() - t0
    path = os.path.join(H.OUT, f"mv3d_pilot_{args.arm}.jsonl")
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(H.rnd(r, 6), sort_keys=True) + "\n")
    preds = {r["pair_id"]: r for r in rows}
    e_arm = H.arm_errors(sub, preds)
    e_proj = H.arm_errors(sub, None)
    auto = H.arm_errors(sub, H.read_predictions("proj_height_auto"))
    used = [i for i, e in enumerate(e_arm) if not e[3]]
    med = lambda e, ii: float(np.median([e[i][0] for i in ii])) if ii else float("nan")  # noqa
    print(f"{args.arm}: {len(sub)} pilot pairs in {el:.1f} s on {platform.node()}, "
          f"fallback {len(sub) - len(used)}, missing inputs {errors}")
    print(f"  all: arm {med(e_arm, range(len(sub))):.2f}  projection "
          f"{med(e_proj, range(len(sub))):.2f}  auto {med(auto, range(len(sub))):.2f}")
    if used:
        g = [e_proj[i][0] - e_arm[i][0] for i in used]
        ga = [auto[i][0] - e_arm[i][0] for i in used]
        print(f"  non-fallback n={len(used)}: arm {med(e_arm, used):.2f} projection "
              f"{med(e_proj, used):.2f} auto {med(auto, used):.2f}; paired gain vs proj "
              f"{np.median(g):.2f}, vs auto {np.median(ga):.2f}; closer than proj "
              f"{np.mean([x > 0 for x in g]):.2f}")
    reasons = defaultdict(int)
    for r in rows:
        if r["x"] is None:
            reasons[r.get("reason") or r.get("error") or "none"] += 1
    print("  fallback reasons:", dict(reasons))
    print(f"-> {path}")


def cmd_predict_many(args):
    """``crossview_align_48.py predict`` for several arms in ONE process with one shared
    Context, so arms that are different readings of the same model run (e.g. mast3r_pair
    and mast3r_poseonly) run the model once. Writes the same predictions/<arm>.jsonl and
    .meta.json as ``predict``; each meta's elapsed_s is that arm's own wall-clock, so the
    first arm of a group carries the model cost and the rest only their reading, and
    ``shared_context_with`` names the group."""
    import platform
    registry = H.load_arms()
    names = args.arms.split(",")
    pairs = H.read_frozen_pairs()
    ctx = H.Context(args, pairs)
    for name in names:
        arm = registry[name]
        t0 = time.time()
        rows, errors = H.run_arm(arm, pairs, ctx)
        elapsed = time.time() - t0
        pred_path, meta_path = H.prediction_paths(name)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w", encoding="utf-8", newline="") as f:
            for r in rows:
                f.write(json.dumps(H.rnd(r, 6), sort_keys=True) + "\n")
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        meta = {"arm": name, "description": arm.description, "needs": list(arm.needs),
                "pairs_sha256": H.PAIRS_SHA256, "pairs": len(pairs), "elapsed_s": elapsed,
                "host": platform.node(), "gpu_visible": gpu, "missing_inputs": errors,
                "fallback": sum(1 for r in rows if r["x"] is None),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": arm.config, "versions": _versions(),
                "shared_context_with": names,
                "manifest_sha256": _sha256(extra(ctx, "manifest") or MANIFEST)}
        H.write_json(meta_path, meta)
        print(f"{name}: {len(rows)} pairs in {elapsed:.1f} s, fallback {meta['fallback']}, "
              f"missing inputs {errors} -> {pred_path}", flush=True)


def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _versions():
    v = H._versions()
    for mod in ("pycolmap", "vggt", "mapanything", "mast3r", "dust3r"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            pass
    return v


REPORT_JSON = os.path.join(H.OUT, "mv3d_results.json")
BASELINES = ("projection", "proj_height_auto")


def _summ(pairs, e, idx, base):
    """Median [CI], within-2 deg, fallback, and the paired median gain over each baseline
    [CI] on pairs ``idx``; CIs resample ramps (the harness's cluster bootstrap)."""
    by_ramp = defaultdict(list)
    for i in idx:
        by_ramp[pairs[i]["ramp_uid"]].append(i)
    groups = [v for _, v in sorted(by_ramp.items())]

    def med(ii):
        return float(np.median([e[i][0] for i in ii])) if ii else float("nan")

    out = {"n_pairs": len(idx), "n_ramps": len(groups), "median_deg": med(idx),
           "median_ci": H.cluster_bootstrap(groups, med) if idx else None,
           "within_2deg": float(np.mean([e[i][0] <= 2.0 for i in idx])) if idx else None,
           "fallback_rate": float(np.mean([e[i][3] for i in idx])) if idx else None}
    for b, eb in base.items():
        def gain(ii, eb=eb):
            return float(np.median([eb[i][0] - e[i][0] for i in ii])) if ii else float("nan")

        def closer(ii, eb=eb):
            return float(np.mean([e[i][0] < eb[i][0] - 1e-9 for i in ii])) if ii else float("nan")

        out[f"gain_vs_{b}"] = gain(idx)
        out[f"gain_vs_{b}_ci"] = H.cluster_bootstrap(groups, gain) if idx else None
        out[f"closer_than_{b}"] = closer(idx)
        out[f"{b}_median_deg"] = float(np.median([eb[i][0] for i in idx])) if idx else None
    return out


def cmd_report(args):
    """Score the multi-view 3D arms against BOTH baselines -- today's projection and the
    labeler's 'auto' height (proj_height_auto), the cheapest known improvement -- on all
    pairs and on the pairs each arm did not fall back on, overall and by imagery.
    Reads committed predictions only; writes mv3d_results.json."""
    pairs = H.read_frozen_pairs()
    base = {"projection": H.arm_errors(pairs, None),
            "proj_height_auto": H.arm_errors(pairs, H.read_predictions("proj_height_auto"))}
    res = {"config": {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT, "seed": H.SEED,
                      "ci": "2.5-97.5 percentile, ramps resampled",
                      "baselines": list(BASELINES)}, "arms": {}}
    strata = {"all": list(range(len(pairs)))}
    for i, p in enumerate(pairs):
        strata.setdefault(f"imagery={p['imagery']}", []).append(i)
    for name in ["proj_height_auto"] + args.arms.split(","):
        e = base["proj_height_auto"] if name == "proj_height_auto" else \
            H.arm_errors(pairs, H.read_predictions(name))
        res["arms"][name] = {}
        for sk, idx in strata.items():
            used = [i for i in idx if not e[i][3]]
            res["arms"][name][sk] = {"all_pairs": _summ(pairs, e, idx, base),
                                     "not_fallen_back": _summ(pairs, e, used, base)}
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(H.rnd(res, 4), indent=1, sort_keys=True) + "\n")
    fmt = lambda v, ci: f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"  # noqa
    for sk in strata:
        print(f"\n### {sk}\n")
        print("| arm | fallback | median deg, all pairs [CI] | within 2 deg | n used | used: arm "
              "vs auto vs projection | used: paired gain vs projection [CI] | used: paired "
              "gain vs auto [CI] | used: closer than auto |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, r in res["arms"].items():
            a, u = r[sk]["all_pairs"], r[sk]["not_fallen_back"]
            if u["n_pairs"]:
                used = (f"{u['n_pairs']} | {u['median_deg']:.2f} vs "
                        f"{u['proj_height_auto_median_deg']:.2f} vs "
                        f"{u['projection_median_deg']:.2f} | "
                        f"{fmt(u['gain_vs_projection'], u['gain_vs_projection_ci'])} | "
                        f"{fmt(u['gain_vs_proj_height_auto'], u['gain_vs_proj_height_auto_ci'])}"
                        f" | {u['closer_than_proj_height_auto']:.2f}")
            else:
                used = "0 | - | - | - | -"
            print(f"| {name} | {a['fallback_rate']:.2f} | {fmt(a['median_deg'], a['median_ci'])}"
                  f" | {a['within_2deg']:.2f} | {used} |")
    print(f"\n-> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="multi-view 3D arms: manifest and corner views")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--labeler-root", required=True)
    p.add_argument("--runs-root")
    p.add_argument("--results-root")
    p.add_argument("--out", default=MANIFEST)
    p.set_defaults(fn=build_manifest)
    p = sub.add_parser("render")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("pilot", help="run an arm on the 30-pair pilot subset and score it")
    p.add_argument("--arm", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--per-city", type=int, default=PILOT_PER_CITY)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--labeler-root")
    p.add_argument("--runs-root")
    p.add_argument("--results-root")
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("predict-many", help="several arms, one process, shared model runs")
    p.add_argument("--arms", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--labeler-root")
    p.add_argument("--runs-root")
    p.add_argument("--results-root")
    p.set_defaults(fn=cmd_predict_many)
    p = sub.add_parser("report", help="score arms against projection AND proj_height_auto")
    p.add_argument("--arms", required=True, help="comma-separated committed predictions")
    p.add_argument("--out", default=REPORT_JSON)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
