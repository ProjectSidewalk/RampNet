"""3D viewer bundles for a few corners (#48): scene point cloud, cameras, points.

    # pick corners (desktop, committed predictions; reads the reference to rank them)
    python scripts/analysis/crossview_arms/_scenes.py pick --arm mapa_posed_pair
    # export (GPU host with the views; MapAnything env)
    python scripts/analysis/crossview_arms/_scenes.py export --ramps richmond:180,... \\
        --views VIEWS --extra corner_views=CORNER_VIEWS --out SCENES
    # finish (desktop): add every committed arm's point / ray to points.json, hash the scenes
    python scripts/analysis/crossview_arms/_scenes.py points --scenes SCENES

A bundle is ``<out>/<ramp_uid with : -> _>/``:

* ``scene.ply`` -- MapAnything's metric point cloud of the corner (``mapa_posed_corner``'s
  run: up to 12 captures, given intrinsics and pose priors), carried into the corner frame
  through the source camera, confidence-filtered and subsampled to ``MAX_POINTS``; binary
  PLY, x y z float32 + red green blue uint8.
* ``cameras.json`` -- every capture used: pano id, role (``src`` / ``oth`` / ``extra``), its
  harness view file, pinhole intrinsics of that 1024 x 768 view, and camera-to-world pose (the
  PRIOR pose; MapAnything's output pose alongside for the views it saw).
* ``points.json`` -- the GT click in 3D (MapAnything's point at the click), the click's ray,
  and for each pair of the corner every committed arm's predicted point in the other view as
  a ray from the other camera (``ref`` is the answer, the other view's RampNet detection).
* the frame: **east-north-up metres** about the source camera's ground point (the labeler's
  LocalFrame at the source pano; z = 0 is the flat ground, cameras sit at their 'auto'
  height). Camera axes are OpenCV (x right, y down, z forward).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import crossview_align_48 as H  # noqa: E402
from crossview_arms import _mv3d as M  # noqa: E402

MAX_POINTS = 400_000
CONF_PCT = 30            # drop the lowest-confidence 30% of pointmap pixels
README = ("Frame: east-north-up metres (x east, y north, z up), origin on the ground below "
          "the source camera (labeler LocalFrame at the source pano); z = 0 is flat ground "
          "and each camera sits at its 'auto' height. Camera poses are camera-to-world 4x4 "
          "with OpenCV camera axes (x right, y down, z forward). Rays: origin + t * dir, t >= 0.")


def scene_dir(out, uid):
    return os.path.join(out, uid.replace(":", "_"))


def cmd_pick(args):
    """Rank corners by the arm's mean error over the corner's pairs: the best ``--n-best``
    and worst ``--n-worst`` (reads the reference: this is for choosing what to look at)."""
    pairs = H.read_frozen_pairs()
    e = H.arm_errors(pairs, H.read_predictions(args.arm))
    auto = H.arm_errors(pairs, H.read_predictions("proj_height_auto"))
    by = {}
    for p, a, b in zip(pairs, e, auto):
        if a[3]:
            continue
        by.setdefault(p["ramp_uid"], []).append((a[0], b[0], p["pair_id"], p["city"]))
    ranked = sorted(by.items(), key=lambda kv: np.mean([x[0] for x in kv[1]]))
    # best: the arm is accurate AND beats auto on both of the corner's pairs (a showcase of
    # what the arm adds, not of corners where everything already works)
    good = [kv for kv in ranked if len(kv[1]) == 2 and all(a < b for a, b, _, _ in kv[1])]
    good = good[:args.n_best]
    bad = [kv for kv in ranked if len(kv[1]) == 2][-args.n_worst:]
    for tag, rows in (("best", good), ("worst", bad)):
        for uid, xs in rows:
            print(tag, uid, " ".join(f"{pid} arm {a:.2f} auto {b:.2f}" for a, b, pid, _ in xs))
    print(",".join(uid for uid, _ in good + bad))


def _write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


def _cam_entry(v, T_out=None):
    R, C = M.cam_pose_world(v)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, C
    e = {"pano_id": v["pano"], "role": v["role"], "is_source": v["role"] == "src",
         "view": v["view"], "pair_id": v.get("pair_id"), "capture_date": v["date"],
         "width": H.VIEW_W, "height": H.VIEW_H, "K": M.intrinsics().tolist(),
         "view_centre_equirect": [v["cx"], v["cy"]], "heading_deg": v["heading"],
         "camera_height_m": v["h"], "cam_to_world_prior": T.tolist()}
    if T_out is not None:
        e["cam_to_world_mapanything"] = T_out.tolist()
    return e


def cmd_export(args):
    """MapAnything (posed, corner) per chosen corner -> scene.ply + cameras.json + a first
    points.json with the click."""
    import torch
    from crossview_arms import ff3d
    pairs = {p["pair_id"]: p for p in H.read_frozen_pairs()}
    ctx = H.Context(args, list(pairs.values()))
    for uid in args.ramps.split(","):
        corner = M.corner_for(ctx, {"ramp_uid": uid})
        pair = pairs[corner["pairs"][0]]
        views = ff3d._views_for(ctx, pair, True)
        size = ff3d.SIZES["mapanything"]
        # the same run as mapa_posed_corner, keeping the dense output
        from mapanything.utils.image import preprocess_inputs
        K = torch.from_numpy(ff3d._K_at(size)).float()
        inputs = []
        for v in views:
            R, C = M.cam_pose_world(v)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, C
            inputs.append({"img": torch.from_numpy(ff3d._rgb(ctx, v, size) * 255.0).float(),
                           "intrinsics": K, "camera_poses": torch.from_numpy(T).float(),
                           "is_metric_scale": torch.tensor([True])})
        model = ff3d._model(ctx, "mapanything")
        with torch.no_grad():
            preds = model.infer(preprocess_inputs(inputs), memory_efficient_inference=False,
                                use_amp=True, amp_dtype="bf16", apply_mask=False,
                                mask_edges=False, apply_confidence_mask=False)
        Tsp = np.eye(4)
        Tsp[:3, :3], Tsp[:3, 3] = M.cam_pose_world(views[0])
        Tso = preds[0]["camera_poses"][0].float().cpu().numpy()
        A = Tsp @ np.linalg.inv(Tso)                   # MapAnything frame -> corner ENU
        xyz, rgb, conf, cams = [], [], [], []
        for v, p in zip(views, preds):
            P = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
            xyz.append(P @ A[:3, :3].T + A[:3, 3])
            rgb.append((p["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3) * 255)
                       .clip(0, 255).astype(np.uint8))
            conf.append(p["conf"][0].float().cpu().numpy().ravel())
            cams.append(_cam_entry(v, A @ p["camera_poses"][0].float().cpu().numpy()))
        xyz, rgb, conf = np.concatenate(xyz), np.concatenate(rgb), np.concatenate(conf)
        keep = conf >= np.percentile(conf, CONF_PCT)
        idx = np.flatnonzero(keep)
        rng = np.random.default_rng(H.SEED)
        if len(idx) > MAX_POINTS:
            idx = rng.choice(idx, MAX_POINTS, replace=False)
        d = scene_dir(args.out, uid)
        os.makedirs(d, exist_ok=True)
        _write_ply(os.path.join(d, "scene.ply"), xyz[idx].astype(np.float32), rgb[idx])
        # the click, lifted by the same run
        sx, sy = ff3d._scale(size)
        Xc = ff3d._bilinear(preds[0]["pts3d"][0].float().cpu().numpy(),
                            ff3d.CLICK[0] * sx, ff3d.CLICK[1] * sy)
        X = A[:3, :3] @ Xc + A[:3, 3]
        src = views[0]
        _, Cs = M.cam_pose_world(src)
        # every capture of the corner, the ones MapAnything did not see without its pose
        seen = {v["view"] for v in views}
        for v in corner["views"]:
            if v["view"] not in seen:
                cams.append(_cam_entry(v))
        meta = {"ramp_uid": uid, "city": corner["city"], "imagery": corner["imagery"],
                "frame": README, "scene": "scene.ply",
                "scene_source": f"MapAnything {ff3d.MODEL_IDS['mapanything']}@"
                                f"{ff3d.MODEL_REVISIONS['mapanything'][:7]}, given intrinsics "
                                f"and pose priors, {len(views)} views; lowest {CONF_PCT}% "
                                f"confidence dropped; {len(idx)} points",
                "n_points": int(len(idx))}
        with open(os.path.join(d, "cameras.json"), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(H.rnd({**meta, "cameras": cams}, 6), indent=1) + "\n")
        ray = M.pano_ray_world(src["heading"], corner["src_x"], corner["src_y"])
        pts = {**meta, "gt_click": {
            "pano_id": src["pano"], "equirect": [corner["src_x"], corner["src_y"]],
            "ray": {"origin": Cs.tolist(), "dir": ray.tolist()},
            "point_mapanything": X.tolist(),
            "point_flat_auto": M.raycast_flat(src, corner["src_x"], corner["src_y"]).tolist()}}
        with open(os.path.join(d, "points.json"), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(H.rnd(pts, 6), indent=1) + "\n")
        print(f"{uid}: {len(views)} views, {len(idx)} points -> {d}", flush=True)


def cmd_points(args):
    """Add every committed arm's prediction for the corner's pairs to points.json (a ray from
    the other camera, and the arm's own 3D point where it has one), plus the reference; and
    write scenes_manifest.json with each scene file's sha256."""
    pairs = {p["pair_id"]: p for p in H.read_frozen_pairs()}
    arms = H.available_predictions()
    preds = {a: H.read_predictions(a) for a in arms}
    manifest = {"frame": README, "scenes": {}}
    for d in sorted(os.listdir(args.scenes)):
        pj = os.path.join(args.scenes, d, "points.json")
        if not os.path.exists(pj):
            continue
        pts = json.load(open(pj, encoding="utf-8"))
        corner = next(c for c in M.load_manifest()["corners"] if c["ramp_uid"] == pts["ramp_uid"])
        pts["pairs"] = {}
        for v in corner["views"]:
            if v["role"] != "oth":
                continue
            p = pairs[v["pair_id"]]
            _, Co = M.cam_pose_world(v)

            def ray(x, y, v=v, Co=Co):
                return {"equirect": [x, y], "origin": Co.tolist(),
                        "dir": M.pano_ray_world(v["heading"], x, y).tolist()}

            entry = {"other_pano_id": v["pano"], "reference_rampnet_detection": ray(p["ref_x"], p["ref_y"]),
                     "projection_2p6m": ray(p["proj_x"], p["proj_y"]), "arms": {}}
            for a in arms:
                r = preds[a].get(v["pair_id"])
                if r and r.get("x") is not None:
                    entry["arms"][a] = ray(r["x"], r["y"])
                    entry["arms"][a]["error_deg"] = float(
                        H.angular_error_deg(r["x"], r["y"], p["ref_x"], p["ref_y"]))
            pts["pairs"][v["pair_id"]] = entry
        with open(pj, "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(H.rnd(pts, 6), indent=1) + "\n")
        files = {}
        for fn in sorted(os.listdir(os.path.join(args.scenes, d))):
            with open(os.path.join(args.scenes, d, fn), "rb") as f:
                b = f.read()
            files[fn] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
        manifest["scenes"][pts["ramp_uid"]] = files
    with open(os.path.join(args.scenes, "scenes_manifest.json"), "w", encoding="utf-8",
              newline="") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1)[:2000])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--arm", default="mapa_posed_pair")
    p.add_argument("--n-best", type=int, default=4)
    p.add_argument("--n-worst", type=int, default=2)
    p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("export")
    p.add_argument("--ramps", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("points")
    p.add_argument("--scenes", required=True)
    p.set_defaults(fn=cmd_points)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
