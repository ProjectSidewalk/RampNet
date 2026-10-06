"""Orbit video of a MapAnything reconstruction of one corner from several panoramas.

The point clouds are the ones behind the cross-view placement viewer
(https://claude.ai/artifact/LxtRmaT7HR7i43zYYH9sJ3, built for #48 from PR #210's
MapAnything arm: 12 Mapillary / GSV views with GPS + compass pose priors). They are committed
here as ``flyaround/<corner>.npz`` (float16 xyz in east-north-up metres, uint8 rgb, metadata with
the camera poses and the ramp's ground-truth point) so the video regenerates without the artifact.

    python scripts/talks/uchicago_2026_flyaround.py render --corner richmond_99
    python scripts/talks/uchicago_2026_flyaround.py video  --corner richmond_99

Frames and the mp4 are gitignored (``flyaround/.gitignore``); ``video`` needs ffmpeg.
"""
import argparse
import json
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FA_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "flyaround")
N_FRAMES = 240            # 8 s at 30 fps
ELEV_DEG = 30.0
RADIUS_M = 30.0           # orbit radius around the ramp, metres
FOV_DEG = 48.0
POINT_PX = 4


def load(corner):
    import numpy as np
    z = np.load(os.path.join(FA_DIR, f"{corner}.npz"))
    return z["xyz"].astype(np.float32), z["rgb"], json.loads(str(z["meta"]))


def _project(pts, cam_pos, target, fov_deg, w, h):
    """Pinhole projection of Nx3 world points for a camera at cam_pos looking at target, z up.
    Returns pixel x, y and depth (positive = in front)."""
    import numpy as np
    f = np.asarray(target, np.float32) - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0, 0, 1], np.float32))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = pts - cam_pos
    x, y, z = d @ r, d @ u, d @ f
    focal = (w / 2) / np.tan(np.radians(fov_deg / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        px = w / 2 + focal * x / z
        py = h / 2 - focal * y / z
    return px, py, z


def _splat(canvas, px, py, z, colors, size):
    """Draw points far-to-near as size x size squares; nearer points overwrite farther ones."""
    import numpy as np
    h, w = canvas.shape[:2]
    ok = (z > 0.3) & (px > -size) & (px < w + size) & (py > -size) & (py < h + size)
    order = np.argsort(-z[ok])
    xi = px[ok][order].astype(np.int32)
    yi = py[ok][order].astype(np.int32)
    col = colors[ok][order]
    half = size // 2
    for dy in range(-half, size - half):
        for dx in range(-half, size - half):
            xx, yy = xi + dx, yi + dy
            m = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            canvas[yy[m], xx[m]] = col[m]


def cmd_render(args):
    import numpy as np
    from PIL import Image, ImageDraw

    xyz, rgb, meta = load(args.corner)
    gt = np.array(meta["gt"]["point_mapa"], dtype=np.float32)
    keep = ((np.abs(xyz[:, 0] - gt[0]) < 40) & (np.abs(xyz[:, 1] - gt[1]) < 40)
            & (xyz[:, 2] > -1.5) & (xyz[:, 2] < 14))
    xyz, rgb = xyz[keep], rgb[keep]
    # Pavement is dark; lift the mid-tones so the scene reads on a black background.
    colors = (255 * (rgb.astype(np.float32) / 255.0) ** 0.6).astype(np.uint8)
    cams = [(np.array(c["T_prior"], np.float32)[:3, 3], c["role"]) for c in meta["cameras"]]
    frames_dir = os.path.join(FA_DIR, "frames", args.corner)
    os.makedirs(frames_dir, exist_ok=True)
    with open(os.path.join(FA_DIR, ".gitignore"), "w", newline="\n") as f:
        f.write("frames/\n*.mp4\n")

    W, H = 1920, 1080
    target = np.array([gt[0], gt[1], 0.8], np.float32)
    for i in range(0, N_FRAMES, args.step):
        az = np.radians(360.0 * i / N_FRAMES)
        el = np.radians(ELEV_DEG)
        cam_pos = target + RADIUS_M * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az),
                                                np.sin(el)], np.float32)
        canvas = np.full((H, W, 3), 12, np.uint8)
        px, py, z = _project(xyz, cam_pos, target, FOV_DEG, W, H)
        _splat(canvas, px, py, z, colors, POINT_PX)
        frame = Image.fromarray(canvas)
        d = ImageDraw.Draw(frame, "RGBA")
        d.rectangle((0, 960, W, H), fill=(12, 12, 14, 215))
        for pos, role in cams:
            cx, cy, cz = _project(pos[None, :], cam_pos, target, FOV_DEG, W, H)
            gx, gy, gz = _project(np.array([[pos[0], pos[1], 0.0]], np.float32), cam_pos, target, FOV_DEG, W, H)
            if cz[0] > 0.3 and gz[0] > 0.3:
                r = 11 if role == "src" else 8
                d.line([(gx[0], gy[0]), (cx[0], cy[0])], fill=(90, 240, 255), width=2)
                d.polygon([(cx[0], cy[0] - r), (cx[0] - r, cy[0] + r), (cx[0] + r, cy[0] + r)],
                          fill=(90, 240, 255) if role == "src" else (150, 215, 255))
        rx, ry, rz = _project(gt[None, :] + np.array([0, 0, 0.05], np.float32), cam_pos, target, FOV_DEG, W, H)
        if rz[0] > 0.3:
            rad = max(10, int(1200 / rz[0]))
            d.ellipse((rx[0] - rad, ry[0] - rad, rx[0] + rad, ry[0] + rad), outline=(255, 211, 77), width=5)
        d.text((40, 985), f"{meta['city'].title()}, {meta['imagery']} · {len(cams)} panoramas "
               "reconstructed together", font=_font(40), fill=(240, 240, 236))
        d.text((40, 1038), "ring: the curb ramp's ground-truth point, placed in 3D from two views · "
               "triangles: camera positions", font=_font(24), fill=(160, 160, 154))
        frame.save(os.path.join(frames_dir, f"frame_{i:03d}.png"))
        if i % 30 == 0:
            print("frame", i, flush=True)


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def cmd_video(args):
    frames_dir = os.path.join(FA_DIR, "frames", args.corner)
    out = os.path.join(FA_DIR, f"flyaround_{args.corner}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                    "-i", os.path.join(frames_dir, "frame_%03d.png"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    print("wrote", os.path.relpath(out, REPO), f"({os.path.getsize(out) / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "video"):
        p = sub.add_parser(name)
        p.add_argument("--corner", default="richmond_99")
        if name == "render":
            p.add_argument("--step", type=int, default=1, help="render every Nth frame (for a quick look)")
    args = ap.parse_args(argv)
    {"render": cmd_render, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
