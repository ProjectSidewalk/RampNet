"""Fuse a dense MapAnything scene into one textured surface and render a camera path.

Input is what ``uchicago_2026_dense_scene.py`` wrote on makelab2 (``pointmaps.npz`` +
``cameras.json``), copied under ``docs/talks/uchicago_2026/flyaround/dense/<slug>/``.
Per-view point maps become depth images in each refined camera, are integrated into a
truncated signed distance volume (Open3D), and marching cubes gives one mesh with vertex
colours: no z-fighting between overlapping views, holes only where nothing was seen.

    python scripts/talks/uchicago_2026_mesh.py fuse   --slug richmond_99 [--voxel 0.04]
    python scripts/talks/uchicago_2026_mesh.py path   --slug richmond_99        # writes path.json
    python scripts/talks/uchicago_2026_mesh.py render --slug richmond_99        # frames + mp4

The path is also what the Gaussian-splatting renderer on klone follows (``path.json``:
cam-to-world 4x4 per frame plus K), so the two videos are comparable shot for shot.
Shots: ``drive`` follows the real capture line past the corner, looking at the ramp;
``orbit`` circles the ramp at a modest radius and height.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FA_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "flyaround")
FRAME_W, FRAME_H = 1920, 1080
FPS = 30


def dense_dir(slug):
    return os.path.join(FA_DIR, "dense", slug)


def load_scene(slug):
    with open(os.path.join(dense_dir(slug), "cameras.json"), encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(os.path.join(dense_dir(slug), "pointmaps.npz"))
    return meta, z


# ---------------------------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------------------------

def cmd_fuse(args):
    """TSDF fusion with Open3D's tensor VoxelBlockGrid (the legacy ScalableTSDFVolume returns
    an empty mesh in the 0.20 Windows wheel, even on synthetic input). Depth goes in as uint16
    millimetres and colour as uint8, the one input pairing that kernel accepts."""
    import open3d as o3d
    import open3d.core as o3c
    meta, z = load_scene(args.slug)
    xyz = z["xyz"].astype(np.float32)
    conf = z["conf"].astype(np.float32)
    rgb = z["rgb"]
    n, h, w, _ = xyz.shape
    mw, mh = [int(v) for v in z["model_size"]]
    assert (w, h) == (mw, mh)
    gt = np.array(meta["gt_click"]["point_mapanything"], np.float32) if meta.get("gt_click") else np.zeros(3)
    thr = np.percentile(conf, args.conf_pct)
    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=args.voxel, block_resolution=8,
        block_count=args.blocks, device=dev)
    for i, cam in enumerate(meta["cameras"]):
        T = np.array(cam["cam_to_world_refined"], np.float64)
        K = np.array(cam["K"], np.float64)
        sx, sy = w / cam["width"], h / cam["height"]
        Km = K.copy()
        Km[0] *= sx
        Km[1] *= sy
        P = xyz[i].reshape(-1, 3)
        depth = ((P - T[:3, 3]) @ T[:3, :3])[:, 2].reshape(h, w)
        ok = (conf[i] >= thr) & (depth > 0.5) & (depth < args.max_depth)
        ok &= np.linalg.norm(P - gt, axis=1).reshape(h, w) < args.radius
        # the capture vehicle: anything within --vehicle-radius of the camera, below it
        ok &= ~((np.linalg.norm(P - T[:3, 3], axis=1).reshape(h, w) < args.vehicle_radius)
                & (P[:, 2].reshape(h, w) < T[2, 3] - 0.3))
        dimg = o3d.t.geometry.Image(o3c.Tensor(np.where(ok, depth * 1000.0, 0).astype(np.uint16)))
        Kt = o3c.Tensor(Km, dtype=o3c.float64)
        Kc = Kt
        if args.color_views:
            # colour from the full-resolution face rather than the model-resolution copy
            from PIL import Image
            face = np.asarray(Image.open(os.path.join(dense_dir(args.slug), "views", cam["view"])).convert("RGB"))
            cimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(face)))
            Kc = o3c.Tensor(K, dtype=o3c.float64)
        else:
            cimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb[i])))
        ext = o3c.Tensor(np.linalg.inv(T), dtype=o3c.float64)
        blocks = vbg.compute_unique_block_coordinates(dimg, Kt, ext, 1000.0, args.max_depth)
        vbg.integrate(blocks, dimg, cimg, Kt, Kc, ext, 1000.0, args.max_depth)
        if i % 20 == 0:
            print("integrated", i, "/", n, flush=True)
    tm = vbg.extract_triangle_mesh(weight_threshold=args.min_weight)
    mesh = tm.to_legacy()
    mesh.compute_vertex_normals()
    tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n = np.asarray(cluster_n)
    keep = cluster_n[tri_clusters] >= args.min_cluster
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    out = os.path.join(dense_dir(args.slug), f"mesh_v{int(args.voxel * 100):02d}.ply")
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles -> {out}")


# ---------------------------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------------------------

def look_at(eye, target, up=(0, 0, 1)):
    """Cam-to-world with OpenCV axes (x right, y down, z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                            # down
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, eye
    return T


def smooth(points, n_out):
    """Catmull-Rom-ish resampling of a polyline to n_out points."""
    from scipy.interpolate import make_interp_spline
    pts = np.asarray(points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    k = min(3, len(pts) - 1)
    spl = make_interp_spline(s, pts, k=k)
    return spl(np.linspace(0, s[-1], n_out))


def cmd_path(args):
    meta, _ = load_scene(args.slug)
    gt = np.array(meta["gt_click"]["point_mapanything"], float)
    target = gt + np.array([0, 0, 0.3])
    cams = {}
    for c in meta["cameras"]:
        cams.setdefault(c["pano_id"], np.array(c["cam_to_world_refined"], float)[:3, 3])
    C = np.array(list(cams.values()))
    # Drive shot: order the capture positions along their principal axis, keep the ones
    # within 35 m of the ramp, pass through them at a slightly raised height.
    d = np.linalg.norm(C[:, :2] - gt[:2], axis=1)
    C = C[d < 35]
    u = C[:, :2] - C[:, :2].mean(0)
    axis = np.linalg.svd(u, full_matrices=False)[2][0]
    order = np.argsort(u @ axis)
    line = C[order].copy()
    line[:, 2] = line[:, 2].mean() + args.drive_lift
    n_drive = int(args.seconds_drive * FPS)
    drive = smooth(line, n_drive) if len(line) >= 2 else np.repeat(line, n_drive, axis=0)
    frames = [dict(shot="drive", T=look_at(p, target).tolist()) for p in drive]
    # Orbit shot: a circle around the ramp, elevated, starting where the drive ended.
    n_orbit = int(args.seconds_orbit * FPS)
    start = drive[-1] - target
    a0 = math.atan2(start[0], start[1])
    for i in range(n_orbit):
        a = a0 + 2 * math.pi * i / n_orbit
        eye = target + np.array([args.orbit_r * math.sin(a), args.orbit_r * math.cos(a), args.orbit_h])
        frames.append(dict(shot="orbit", T=look_at(eye, target).tolist()))
    f_px = (FRAME_W / 2) / math.tan(math.radians(args.fov) / 2)
    path = dict(slug=args.slug, width=FRAME_W, height=FRAME_H, fps=FPS,
                K=[[f_px, 0, FRAME_W / 2], [0, f_px, FRAME_H / 2], [0, 0, 1]],
                target=target.tolist(), frames=frames)
    out = os.path.join(dense_dir(args.slug), "path.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(path, f)
    print(f"{len(frames)} frames ({len(drive)} drive, {n_orbit} orbit) -> {out}")


# ---------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------

def cmd_render(args):
    """pyrender (OpenGL) rather than Open3D's Filament renderer: on this machine the latter
    returns a black frame for any mesh larger than a few hundred thousand triangles, while
    pyrender draws the full 9M-triangle fusion in about 3 s."""
    import open3d as o3d
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw
    meta, _ = load_scene(args.slug)
    with open(os.path.join(dense_dir(args.slug), "path.json"), encoding="utf-8") as f:
        path = json.load(f)
    mesh_path = args.mesh or sorted(p for p in os.listdir(dense_dir(args.slug)) if p.startswith("mesh_v"))[0]
    om = o3d.io.read_triangle_mesh(os.path.join(dense_dir(args.slug), mesh_path))
    V = np.asarray(om.vertices)
    C = (np.clip(np.asarray(om.vertex_colors), 0, 1) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(vertices=V, faces=np.asarray(om.triangles), vertex_colors=C, process=False)
    gt = np.array(path["target"]) - np.array([0, 0, 0.3])
    ring = trimesh.creation.torus(0.9, 0.06)
    ring.apply_translation(gt + np.array([0, 0, 0.08]))
    ring.visual.vertex_colors = np.tile(np.array([255, 211, 77, 255], np.uint8), (len(ring.vertices), 1))
    W, H = path["width"], path["height"]
    K = np.array(path["K"])
    scene = pyrender.Scene(bg_color=[0.07, 0.08, 0.11, 1.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    scene.add(pyrender.Mesh.from_trimesh(ring, smooth=False))
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.1, zfar=300.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])          # OpenCV (y down, z forward) -> OpenGL camera
    rend = pyrender.OffscreenRenderer(W, H)
    frames_dir = os.path.join(FA_DIR, "frames", f"mesh_{args.slug}")
    os.makedirs(frames_dir, exist_ok=True)
    label = f"{meta['city'].title()}, {meta['imagery']} · {meta['n_panos']} panoramas, fused surface"
    for i, fr in enumerate(path["frames"]):
        if i % args.step:
            continue
        scene.set_pose(cam_node, np.array(fr["T"]) @ flip)
        color, _ = rend.render(scene, flags=pyrender.RenderFlags.FLAT)
        im = Image.fromarray(color)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, H - 110, W, H), fill=(12, 12, 14, 200))
        d.text((40, H - 92), label, font=_font(38), fill=(240, 240, 236))
        d.text((40, H - 44), "ring: the ramp's ground-truth point · " + fr["shot"] + " shot",
               font=_font(22), fill=(160, 160, 154))
        im.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        if i % 60 == 0:
            print("frame", i, flush=True)
    rend.delete()
    if args.step == 1:
        out = os.path.join(FA_DIR, f"flyaround_mesh_{args.slug}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(frames_dir, "frame_%04d.png"), "-vf", "format=yuv420p",
                        "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", out], check=True)
        print("wrote", os.path.relpath(out, REPO), f"({os.path.getsize(out) / 1e6:.1f} MB)")


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fuse")
    p.add_argument("--slug", required=True)
    p.add_argument("--voxel", type=float, default=0.04)
    p.add_argument("--conf-pct", type=float, default=25.0)
    p.add_argument("--max-depth", type=float, default=40.0)
    p.add_argument("--radius", type=float, default=45.0, help="keep points within this of the ramp")
    p.add_argument("--min-cluster", type=int, default=2000)
    p.add_argument("--min-weight", type=float, default=2.0, help="voxels seen fewer times are dropped")
    p.add_argument("--color-views", action="store_true", help="colour from views/*.jpg at full res")
    p.add_argument("--vehicle-radius", type=float, default=3.5, help="mask the capture vehicle around each camera")
    # Pre-size the block table: letting it rehash as views arrive segfaults in the 0.20
    # Windows wheel (richmond_99 at 5 cm needs ~380k blocks; 1.5M costs ~15 GB RAM).
    p.add_argument("--blocks", type=int, default=1_500_000)
    p = sub.add_parser("path")
    p.add_argument("--slug", required=True)
    p.add_argument("--seconds-drive", type=float, default=8.0)
    p.add_argument("--seconds-orbit", type=float, default=8.0)
    p.add_argument("--drive-lift", type=float, default=1.0)
    p.add_argument("--orbit-r", type=float, default=11.0)
    p.add_argument("--orbit-h", type=float, default=5.0)
    p.add_argument("--fov", type=float, default=60.0)
    p = sub.add_parser("render")
    p.add_argument("--slug", required=True)
    p.add_argument("--mesh", default=None)
    p.add_argument("--step", type=int, default=1)
    args = ap.parse_args(argv)
    {"fuse": cmd_fuse, "path": cmd_path, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
