"""Viewer bundle for a reconstructed corner (#214): the scene files plus JSON a 3D viewer
can load to inspect where each arm put the ramp.

Run AFTER scoring. Unlike the arms, this reads the reference (the answer), because its
purpose is inspection: the reference detection is drawn as a ray from the other camera.

Per corner, into ``--out/<ramp_uid>/`` (``richmond_150`` etc.):

* ``splat.ply`` -- the Gaussian splat (standard 3DGS .ply, SH degree 0) and
  ``points.ply`` -- the sparse SfM points (at most 200k, randomly decimated). Copied only
  when under ``--max-mb``; otherwise left on makelab2 and only their sha256 is recorded.
* ``cameras.json`` -- every registered camera: pose (camera-to-world rotation, centre),
  pinhole/RADIAL intrinsics, image id, flat or pano view, capture date, sequence.
* ``points.json`` -- the GT click lifted to 3D by each lift (sparse / gs / mvs), and per
  harness pair: each 3D arm's point, and each arm's predicted pixel in the other pano as a
  ray from the other camera (2D arms place a direction, not a depth); the reference as a
  ray; today's projection also as its flat-ground 3D point.
* ``README.md`` -- the frame.

Frame: the SfM model's frame. Position priors (each image's Mapillary SfM position in the
corner's ENU frame) make it metric and approximately east-north-up about the corner origin
(the source click raycast at 2.6 m); ``cameras.json`` records each camera's prior next to
its reconstructed centre, so the residual is visible.

    python scripts/analysis/flat3d/viewer_bundle.py --corners-root CORNERS \\
        --corner richmond:150 --out analysis_out/flat_mapillary_3d/scenes
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import crossview_align_48 as H  # noqa: E402
import flat_mapillary_48 as F  # noqa: E402

ARMS_2D = ["projection", "proj_height_auto", "lg", "flat_sfm", "flat_gs", "flat_mvs",
           "noflat_sfm", "noflat_gs"]
EXTERNAL = {"roma_magsac": "origin/analysis/crossview-matching-48",
            "sfm_colmap": "origin/analysis/crossview-sfm-48",
            "mast3r_pair": "origin/analysis/crossview-sfm-48"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ray_of(cam, cx, cy, x, y):
    """World ray (origin, unit direction) through equirect (x, y) of the pano a view
    camera was cut from."""
    R = np.asarray(cam["R_cw"])
    d = R @ (H.view_rotation(cx, cy).T @ H.pano_dir(x, y))
    return {"origin": cam["C"], "dir": [round(float(v), 6) for v in d / np.linalg.norm(d)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners-root", required=True)
    ap.add_argument("--corner", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mb", type=float, default=30.0)
    args = ap.parse_args(argv)
    pairs = {p["pair_id"]: p for p in H.read_frozen_pairs()}
    preds = {}
    for a in ARMS_2D[1:]:
        try:
            preds[a], _ = F.read_preds_any(a)
        except (FileNotFoundError, SystemExit):
            pass
    for a, ref in EXTERNAL.items():
        try:
            preds[a], _ = F.read_preds_any(a, ref)
        except Exception:
            pass
    for uid in args.corner:
        tag = uid.replace(":", "_")
        src_dir = os.path.join(args.corners_root, tag)
        res = json.load(open(os.path.join(src_dir, "result.json"), encoding="utf-8"))
        out = os.path.join(args.out, tag)
        os.makedirs(out, exist_ok=True)
        files = {}
        for fn in ("splat.ply", "points.ply"):
            p = os.path.join(src_dir, fn)
            if not os.path.exists(p):
                continue
            mb = os.path.getsize(p) / 1e6
            files[fn] = {"sha256": sha(p), "mb": round(mb, 2),
                         "committed": mb <= args.max_mb,
                         "makelab2": f"/homes/gws/jonf/flat3d/scenes/{tag}/{fn}"}
            if mb <= args.max_mb:
                shutil.copyfile(p, os.path.join(out, fn))
        cams = res.get("cameras", [])
        H.write_json(os.path.join(out, "cameras.json"), {
            "ramp_uid": uid, "frame": "SfM model frame, metric, ~ENU about the corner origin "
            "(x east, y north, z up, metres)", "R_cw": "camera-to-world rotation; camera axes "
            "x right, y down, z forward (OpenCV)", "intrinsics": "COLMAP RADIAL [f, cx, cy, "
            "k1, k2] in pixels of width x height; pano views have k1 = k2 = 0",
            "cameras": cams, "files": files, "model": res.get("model")})
        by = {c["name"]: c for c in cams}
        src = next((c for c in cams if c["kind"] == "pano_src"), None)
        pts = {"ramp_uid": uid, "gt_click_3d": {m: v.get("X") for m, v in
                                                 res.get("lifts", {}).items()},
               "pairs": {}}
        for pid, pr in res.get("pairs", {}).items():
            p = pairs[pid]
            oth = by.get(f"{pid}_oth.jpg")
            row = {"arm_points_3d": {m: pts["gt_click_3d"].get(m) for m in pr},
                   "oth_camera": f"{pid}_oth.jpg" if oth else None}
            if oth:
                cx, cy = p["proj_x"], p["proj_y"]
                row["reference_ray"] = ray_of(oth, cx, cy, p["ref_x"], p["ref_y"])
                row["arm_rays"] = {"projection": ray_of(oth, cx, cy, p["proj_x"], p["proj_y"])}
                for a, pp in preds.items():
                    r = pp.get(pid)
                    if r and r.get("x") is not None:
                        row["arm_rays"][a] = ray_of(oth, cx, cy, float(r["x"]) % 1.0,
                                                    float(r["y"]))
            if src:
                # today's projection as a 3D point: the click ray from the source camera
                # meets flat ground 2.6 m below it (gravity from the level source view)
                R = np.asarray(src["R_cw"])
                g = R @ H.view_rotation(p["src_x"], p["src_y"]).T @ np.array([0, 1.0, 0])
                ray = R @ np.array([0, 0, 1.0])
                t = 2.6 / float(ray @ g) if ray @ g > 1e-6 else None
                row["projection_point_3d_at_2.6m"] = None if t is None else \
                    [round(float(v), 4) for v in np.asarray(src["C"]) + t * ray]
            pts["pairs"][pid] = row
        H.write_json(os.path.join(out, "points.json"), pts)
        with open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# {uid}\n\nFrame: the SfM model frame, metric and approximately "
                    "east-north-up (x east, y north, z up, metres) about the corner origin, the "
                    "source click raycast at 2.6 m. Camera poses in cameras.json are "
                    "camera-to-world (OpenCV axes). points.json holds the GT click in 3D per "
                    "lift, and per pair each arm's prediction and the reference as rays from "
                    "the other camera. Scene files listed in cameras.json -> files; those not "
                    "committed are on makelab2 with their sha256.\n")
        print(uid, "->", out, {k: v["mb"] for k, v in files.items()})


if __name__ == "__main__":
    main()
