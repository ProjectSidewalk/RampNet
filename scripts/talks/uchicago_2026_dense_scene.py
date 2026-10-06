"""Dense MapAnything reconstruction of one benchmark corner, from the native panoramas.

Runs on makelab2 in the cross-view sweep's environment (MapAnything installed, the harness
checkout importable), reads the corner's cameras from the #48 viewer scene and the native
panoramas from the labeler archive, cuts several posed perspective faces per panorama, runs
MapAnything once over all of them with intrinsics and pose priors, and writes everything the
two renderers need:

    <out>/<slug>/views/<pano>_<k>.jpg      the faces (1024 x 768, --hfov degrees)
    <out>/<slug>/cameras.json              per face: K, prior and refined cam-to-world (ENU)
    <out>/<slug>/pointmaps.npz             per face: xyz (ENU, float16), conf, rgb at model res
    <out>/<slug>/dense.ply                 merged, confidence-filtered, subsampled
    <out>/<slug>/colmap/{images,sparse/0}  COLMAP text model for a Gaussian-splatting trainer

    B=/homes/gws/jonf/crossview48_sfm
    HF_HOME=$B/hf $B/venv/bin/python uchicago_2026_dense_scene.py --slug richmond_99 \\
        --harness $B/RampNet/scripts/analysis \\
        --scenes /homes/gws/jonf/crossview48/scenes \\
        --archive /projects/makeabilitylab/sidewalk-auto-labeler/runs --out /homes/gws/jonf/talk_dense

Frame: the viewer scene's east-north-up metres, origin on the ground under the source camera.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

VIEW_W, VIEW_H = 1024, 768
MODEL_SIZE = (518, 392)          # MapAnything's working resolution, as the harness used it
MODEL_ID = "facebook/map-anything"
MODEL_REV = "a1d87e9"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slug", required=True, help="viewer scene slug, e.g. richmond_99")
    ap.add_argument("--harness", required=True, help="scripts/analysis of a checkout with crossview_arms")
    ap.add_argument("--scenes", required=True, help="directory holding <slug>/cameras.json")
    ap.add_argument("--archive", required=True, help="labeler runs root with <city>/panos/<id>.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--faces", type=int, default=6, help="azimuth faces per panorama")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--pitch", type=float, default=-12.0, help="face pitch, degrees (negative = down)")
    ap.add_argument("--max-panos", type=int, default=None)
    ap.add_argument("--conf-pct", type=float, default=20.0, help="drop this % lowest-confidence points")
    ap.add_argument("--ply-points", type=int, default=3_000_000)
    ap.add_argument("--colmap-points", type=int, default=400_000)
    args = ap.parse_args(argv)

    sys.path.insert(0, args.harness)
    import cv2
    import torch
    import crossview_align_48 as H
    from crossview_arms import _mv3d as M

    with open(os.path.join(args.scenes, args.slug, "cameras.json"), encoding="utf-8") as f:
        scene = json.load(f)
    city = scene["city"]
    panos = []
    seen = set()
    for c in scene["cameras"]:
        if c["pano_id"] in seen:
            continue
        seen.add(c["pano_id"])
        T = np.array(c["cam_to_world_prior"], dtype=np.float64)
        panos.append(dict(pano_id=c["pano_id"], role=c["role"], heading=float(c["heading_deg"]),
                          h=float(c["camera_height_m"]), C=T[:3, 3].copy(),
                          date=c.get("capture_date")))
    # Source first, then the rest nearest the source camera first.
    panos.sort(key=lambda p: (p["role"] != "src", float(np.linalg.norm(p["C"][:2]))))
    if args.max_panos:
        panos = panos[:args.max_panos]
    log(f"{args.slug}: {len(panos)} panoramas, {args.faces} faces each")

    out = os.path.join(args.out, args.slug)
    vdir = os.path.join(out, "views")
    os.makedirs(vdir, exist_ok=True)
    cy = 0.5 - args.pitch / 180.0
    f_px = (VIEW_W / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
    K = np.array([[f_px, 0, VIEW_W / 2.0], [0, f_px, VIEW_H / 2.0], [0, 0, 1.0]])
    tw = int(round(360.0 / args.hfov * VIEW_W))

    faces = []
    for p in panos:
        path = os.path.join(args.archive, city, "panos", f"{p['pano_id']}.jpg")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"missing panorama {path}")
        img = cv2.resize(img, (tw, tw // 2), interpolation=cv2.INTER_AREA)
        for k in range(args.faces):
            cx = (k + 0.5) / args.faces
            face = H.render_view(img, cx, cy, VIEW_W, VIEW_H, args.hfov)
            name = f"{p['pano_id']}_{k}.jpg"
            cv2.imwrite(os.path.join(vdir, name), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            R = M.view_cam_to_world(p["heading"], cx, cy)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, p["C"]
            faces.append(dict(view=name, pano_id=p["pano_id"], face=k, cx=cx, cy=cy,
                              role=p["role"], date=p["date"], T_prior=T,
                              rgb_small=cv2.cvtColor(cv2.resize(face, MODEL_SIZE, interpolation=cv2.INTER_AREA),
                                                     cv2.COLOR_BGR2RGB).astype(np.float32)))
    log(f"cut {len(faces)} faces")

    # ---- MapAnything over every face, with intrinsics and pose priors --------------------
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from crossview_arms import ff3d
    rev = ff3d.MODEL_REVISIONS["mapanything"]          # the full commit the sweep pinned
    assert rev.startswith(MODEL_REV), rev
    dev = "cuda"
    model = MapAnything.from_pretrained(MODEL_ID, revision=rev).to(dev).eval()
    sx, sy = MODEL_SIZE[0] / VIEW_W, MODEL_SIZE[1] / VIEW_H
    Ks = K.copy()
    Ks[0] *= sx
    Ks[1] *= sy
    inputs = [{"img": torch.from_numpy(fc["rgb_small"]).float(),
               "intrinsics": torch.from_numpy(Ks).float(),
               "camera_poses": torch.from_numpy(fc["T_prior"]).float(),
               "is_metric_scale": torch.tensor([True])} for fc in faces]
    t0 = time.time()
    with torch.no_grad():
        preds = model.infer(preprocess_inputs(inputs), memory_efficient_inference=True,
                            use_amp=True, amp_dtype="bf16", apply_mask=False, mask_edges=False,
                            apply_confidence_mask=False)
    log(f"MapAnything on {len(faces)} views in {time.time() - t0:.0f} s; "
        f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    T_src_prior = faces[0]["T_prior"]
    T_src_out = preds[0]["camera_poses"][0].float().cpu().numpy().astype(np.float64)
    A = T_src_prior @ np.linalg.inv(T_src_out)            # model frame -> ENU
    n = len(faces)
    hh, ww = preds[0]["pts3d"][0].shape[:2]
    xyz = np.zeros((n, hh, ww, 3), np.float32)
    conf = np.zeros((n, hh, ww), np.float32)
    rgb = np.zeros((n, hh, ww, 3), np.uint8)
    cams = []
    for i, (fc, p) in enumerate(zip(faces, preds)):
        P = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        xyz[i] = (P @ A[:3, :3].T + A[:3, 3]).reshape(hh, ww, 3)
        conf[i] = p["conf"][0].float().cpu().numpy()
        rgb[i] = (p["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        T_ref = A @ p["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        cams.append(dict(view=fc["view"], pano_id=fc["pano_id"], face=fc["face"], role=fc["role"],
                         capture_date=fc["date"], cx=fc["cx"], cy=fc["cy"], width=VIEW_W,
                         height=VIEW_H, hfov_deg=args.hfov, K=K.tolist(),
                         cam_to_world_prior=fc["T_prior"].tolist(),
                         cam_to_world_refined=T_ref.tolist()))
    np.savez_compressed(os.path.join(out, "pointmaps.npz"), xyz=xyz.astype(np.float16),
                        conf=conf.astype(np.float16), rgb=rgb, model_size=np.array(MODEL_SIZE))
    gt = None
    pj = os.path.join(args.scenes, args.slug, "points.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as f:
            gt = json.load(f).get("gt_click")
    meta = dict(slug=args.slug, ramp_uid=scene.get("ramp_uid"), city=city, imagery=scene.get("imagery"),
                frame=scene.get("frame"), n_panos=len(panos), faces_per_pano=args.faces,
                hfov_deg=args.hfov, pitch_deg=args.pitch, model=f"{MODEL_ID}@{MODEL_REV}",
                model_size=list(MODEL_SIZE), inference="memory_efficient, bf16, intrinsics + pose priors, metric",
                gt_click=gt, run_date=time.strftime("%Y-%m-%d"), cameras=cams)
    with open(os.path.join(out, "cameras.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)

    # ---- merged cloud --------------------------------------------------------------------
    flat_xyz = xyz.reshape(-1, 3)
    flat_rgb = rgb.reshape(-1, 3)
    flat_conf = conf.ravel()
    keep = np.flatnonzero(flat_conf >= np.percentile(flat_conf, args.conf_pct))
    rng = np.random.default_rng(48)
    sel = keep if len(keep) <= args.ply_points else rng.choice(keep, args.ply_points, replace=False)
    write_ply(os.path.join(out, "dense.ply"), flat_xyz[sel], flat_rgb[sel])
    log(f"dense.ply: {len(sel):,} of {len(keep):,} confident points ({len(flat_conf):,} total)")

    # ---- COLMAP text model for a splatting trainer ---------------------------------------
    cdir = os.path.join(out, "colmap")
    sdir = os.path.join(cdir, "sparse", "0")
    os.makedirs(sdir, exist_ok=True)
    idir = os.path.join(cdir, "images")
    if not os.path.lexists(idir):
        os.symlink("../views", idir)          # relative: the tree is copied to other hosts
    with open(os.path.join(sdir, "cameras.txt"), "w", newline="\n") as f:
        f.write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {VIEW_W} {VIEW_H} {f_px:.6f} {f_px:.6f} {VIEW_W / 2:.1f} {VIEW_H / 2:.1f}\n")
    with open(os.path.join(sdir, "images.txt"), "w", newline="\n") as f:
        f.write("# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[]\n")
        for i, c in enumerate(cams, 1):
            T = np.array(c["cam_to_world_refined"])
            Rwc = T[:3, :3].T
            t = -Rwc @ T[:3, 3]
            q = rot_to_quat(Rwc)
            f.write(f"{i} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {c['view']}\n\n")
    sel2 = sel if len(sel) <= args.colmap_points else rng.choice(sel, args.colmap_points, replace=False)
    with open(os.path.join(sdir, "points3D.txt"), "w", newline="\n") as f:
        f.write("# 3D point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for j, k in enumerate(sel2, 1):
            x, y, z = flat_xyz[k]
            r, g, b = flat_rgb[k]
            f.write(f"{j} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} 0\n")
    log(f"colmap model: {len(cams)} images, {len(sel2):,} points -> {cdir}")


def rot_to_quat(R):
    """Rotation matrix -> (w, x, y, z), COLMAP's convention."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def write_ply(path, xyz, rgb):
    import struct
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        buf = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        buf["r"], buf["g"], buf["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(buf.tobytes())


if __name__ == "__main__":
    main()
