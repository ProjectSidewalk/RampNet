#!/usr/bin/env python3
"""Build an Ultralytics YOLO detection dataset from the RampNet point dataset.

The supervised baseline of issue #51: RampNet's training data is per-pano *points*
(``curb_ramp_points_normalized``), not boxes, so this synthesizes boxes to train a
generic detector on the same data. Two geometries (pick with ``--geometry``):

- ``tiles`` (the headline baseline): reproject each pano into the SAME ring of
  rectilinear views the harness scores with (``equirect_tiling.default_views``),
  project every GT point into each view with ``equirect_point_to_perspective``, and
  write a box per in-view point. Train and inference geometry match by construction.
- ``pano``: one whole-pano (downscaled equirect) image per pano, boxes placed at the
  points directly. Matches RampNet's own full-pano input; the ``--tiling none`` eval.

**Box size is a train-only knob** — evaluation reduces every prediction back to a
center point and matches at a fixed radius, so box w/h never enter the metric; they
only shape YOLO's assignment / NMS / confidence. ``--box-size`` picks the strategy:

- ``fixed:<frac>`` (default): constant fraction of the view/pano. Simplest.
- ``pitch``: apparent size proportional to 1/distance, with distance from the point's
  ground-plane pitch (its vertical position). Self-contained, no extra data.
- ``gps``: exact ground distance via haversine of ``pano_coord`` <-> ``curb_ramp_coords``
  (the JSON already carries both). Falls back to ``pitch`` per-pano when the point
  count and coord count disagree (the placed points are not guaranteed 1:1 with the
  GPS list); the summary reports how often that happened, so you can trust or drop it.

Output is an Ultralytics dataset: ``<out>/images/{train,val}/*.jpg`` +
``<out>/labels/{train,val}/*.txt`` + ``<out>/data.yaml`` (test is reserved, not used
for training). ``tiles`` produces ~6x as many files as panos; use ``--subset`` while
iterating and ``--bg-keep-frac`` to thin the many label-less (background) tiles.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict, namedtuple
from glob import glob
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from equirect_tiling import (  # noqa: E402
    default_views, equirect_to_perspective, equirect_point_to_perspective)

CLASS_NAME = "curb_ramp"
SPLIT_MAP = {"train": "train", "val": "val"}   # RampNet split -> YOLO split

Config = namedtuple("Config", [
    "geometry", "strategy", "fixed_frac", "min_frac", "max_frac",
    "ramp_size_m", "camera_height_m", "source_max_edge", "pano_w", "pano_h",
    "views", "out", "overlay_dir", "bg_keep_frac",
])

_CFG = None  # set per worker via the Pool initializer (avoids re-pickling cfg per task)


# --- geometry / distance helpers --------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _ground_distance_m(y_norm, camera_height_m):
    """Ground distance to a pano point under a flat-ground, fixed-camera-height
    model. ``y`` is 0 (top/up) -> 1 (bottom/down); below the horizon (y>0.5) the
    downward pitch is ``(y-0.5)*pi``. At/above the horizon -> inf (the min box)."""
    pitch_below = (y_norm - 0.5) * math.pi
    if pitch_below <= 1e-3:
        return float("inf")
    return camera_height_m / math.tan(pitch_below)


def _clamp_frac(f, cfg):
    return max(cfg.min_frac, min(cfg.max_frac, f))


def _resolve_distances(cfg, points, pano_coord, coords):
    """Per-point distance in meters, or None when it should come from the point's
    own y (fixed ignores it; pitch derives it; gps uses it when correspondence is
    clean). Returns ``(dists, gps_used)``."""
    if (cfg.strategy == "gps" and points and pano_coord and coords
            and len(coords) == len(points)):
        try:
            return ([_haversine_m(pano_coord[0], pano_coord[1], c[0], c[1]) for c in coords],
                    True)
        except (TypeError, IndexError):
            pass
    return [None] * len(points), False


def _box_wh(cfg, x, y, dist_m, fov_h_deg, fov_v_deg):
    """(w, h) box size normalized to the target frame (the square tile, or the pano).

    ``fixed`` returns a constant; the distance-aware strategies convert a physical
    ramp size at ``dist_m`` into an angular size and then a fraction of the frame."""
    if cfg.strategy == "fixed":
        return cfg.fixed_frac, cfg.fixed_frac
    if dist_m is None:                       # pitch, or gps that fell back to pitch
        dist_m = _ground_distance_m(y, cfg.camera_height_m)
    if not math.isfinite(dist_m) or dist_m <= 0:
        return cfg.min_frac, cfg.min_frac
    alpha = cfg.ramp_size_m / dist_m         # small-angle subtended size (radians)
    if cfg.geometry == "tiles":
        return (_clamp_frac(alpha / math.radians(fov_h_deg), cfg),
                _clamp_frac(alpha / math.radians(fov_v_deg), cfg))
    # pano (equirect): a full turn is 2*pi horizontally, pi vertically; longitude is
    # stretched by 1/cos(lat), so widen the box by that (clamped away from the poles).
    lat = (0.5 - y) * math.pi
    return (_clamp_frac(alpha / (2 * math.pi) / max(0.2, math.cos(lat)), cfg),
            _clamp_frac(alpha / math.pi, cfg))


def _keep_background(cfg, stem):
    """Deterministically thin label-less tiles to ``--bg-keep-frac`` (stable across
    processes/runs — a salted hash() would differ per worker)."""
    if cfg.bg_keep_frac >= 1.0:
        return True
    h = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % 1000
    return h < cfg.bg_keep_frac * 1000


# --- IO helpers -------------------------------------------------------------

def _valid_pt(p):
    return (isinstance(p, (list, tuple)) and len(p) == 2
            and all(isinstance(v, (int, float)) for v in p)
            and 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)


def _downscale(img, max_edge):
    from PIL import Image
    if max_edge and max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    return img


def _write_pair(out, yolo_split, stem, image, lines):
    image.save(os.path.join(out, "images", yolo_split, stem + ".jpg"), quality=90)
    with open(os.path.join(out, "labels", yolo_split, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _save_overlay(overlay_dir, stem, image, lines):
    """Draw the synthesized boxes on the tile so a human can eyeball that they land
    on ramps (the label-gen analog of dump_detections' inference overlays)."""
    from PIL import ImageDraw
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for line in lines:
        _, cx, cy, w, h = (float(t) for t in line.split())
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    img.save(os.path.join(overlay_dir, stem + ".jpg"), quality=90)


# --- per-pano worker --------------------------------------------------------

def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def process_pano(task):
    """Render one pano's tiles (or its whole-pano image) + YOLO labels. Returns a
    stats dict; workers write distinct files (keyed by pano id) so there is no race."""
    from PIL import Image
    split, jpg_path, json_path, overlay = task
    cfg = _CFG
    st = defaultdict(int)
    st["panos"] = 1
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        st["read_errors"] = 1
        return st

    points = [tuple(p) for p in meta.get("curb_ramp_points_normalized", []) if _valid_pt(p)]
    st["points"] = len(points)
    dists, gps_used = _resolve_distances(cfg, points, meta.get("pano_coord"),
                                         meta.get("curb_ramp_coords"))
    if cfg.strategy == "gps" and points and not gps_used:
        st["gps_mismatch"] = 1

    try:
        img = Image.open(jpg_path).convert("RGB")
    except OSError:
        st["read_errors"] = 1
        return st
    pid = str(meta.get("pano_id") or os.path.splitext(os.path.basename(jpg_path))[0])
    yolo_split = SPLIT_MAP[split]

    if cfg.geometry == "tiles":
        src = _downscale(img, cfg.source_max_edge)
        for k, view in enumerate(cfg.views):
            lines = []
            for (x, y), d in zip(points, dists):
                uv = equirect_point_to_perspective(x, y, view)
                if uv is None:
                    continue
                u, v = uv
                w, h = _box_wh(cfg, x, y, d, view.fov_h_deg, view.fov_v_deg)
                lines.append(f"0 {u:.6f} {v:.6f} {w:.6f} {h:.6f}")
            stem = f"{pid}_v{k}"
            if not lines and not _keep_background(cfg, stem):
                st["bg_skipped"] += 1
                continue
            _write_pair(cfg.out, yolo_split, stem, equirect_to_perspective(src, view), lines)
            st["tiles"] += 1
            st["boxes"] += len(lines)
            st["bg_tiles"] += 1 if not lines else 0
            if overlay and cfg.overlay_dir:
                _save_overlay(cfg.overlay_dir, stem, equirect_to_perspective(src, view), lines)
    else:  # pano
        pimg = img.resize((cfg.pano_w, cfg.pano_h), Image.BILINEAR)
        lines = []
        for (x, y), d in zip(points, dists):
            w, h = _box_wh(cfg, x, y, d, None, None)
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        _write_pair(cfg.out, yolo_split, pid, pimg, lines)
        st["tiles"] += 1
        st["boxes"] += len(lines)
        st["bg_tiles"] += 1 if not lines else 0
        if overlay and cfg.overlay_dir:
            _save_overlay(cfg.overlay_dir, pid, pimg, lines)
    return st


# --- driver -----------------------------------------------------------------

def parse_box_size(s):
    s = s.strip().lower()
    if s.startswith("fixed"):
        _, _, val = s.partition(":")
        return "fixed", (float(val) if val else 0.03)
    if s in ("pitch", "gps"):
        return s, 0.0
    raise argparse.ArgumentTypeError("box-size must be fixed[:frac], pitch, or gps")


def write_data_yaml(out):
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"# Generated by prepare_yolo_dataset.py\npath: {os.path.abspath(out)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"names:\n  0: {CLASS_NAME}\n")


def build_config(args):
    strategy, fixed_frac = args.box_size
    views = default_views(fov_h_deg=args.view_fov, fov_v_deg=args.view_fov,
                          pitch_deg=args.view_pitch, n_yaw=args.n_yaw,
                          width=args.view_size, height=args.view_size)
    return Config(
        geometry=args.geometry, strategy=strategy, fixed_frac=fixed_frac,
        min_frac=args.min_frac, max_frac=args.max_frac, ramp_size_m=args.ramp_size_m,
        camera_height_m=args.camera_height_m, source_max_edge=args.source_max_edge,
        pano_w=args.pano_width, pano_h=args.pano_width // 2, views=views,
        out=os.path.abspath(args.out), overlay_dir=os.path.abspath(args.overlay_dir)
        if args.overlay_dir else None, bg_keep_frac=args.bg_keep_frac)


def gather_tasks(args, cfg):
    tasks = []
    for split in SPLIT_MAP:
        d = os.path.join(args.dataset_root, split)
        jpgs = sorted(glob(os.path.join(d, "*.jpg")))
        if args.subset:
            jpgs = jpgs[:args.subset]
        for i, jpg in enumerate(jpgs):
            js = jpg[:-4] + ".json"
            if os.path.exists(js):
                overlay = bool(cfg.overlay_dir) and i < args.overlay_n
                tasks.append((split, jpg, js, overlay))
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Build a YOLO dataset from the RampNet point dataset.")
    ap.add_argument("--dataset-root", default="dataset",
                    help="Root with train/ val/ (test/) of <pano_id>.jpg + .json (default dataset).")
    ap.add_argument("--out", required=True, help="Output Ultralytics dataset dir.")
    ap.add_argument("--geometry", choices=["tiles", "pano"], default="tiles")
    ap.add_argument("--box-size", type=parse_box_size, default=("fixed", 0.03),
                    help="fixed[:frac] (default fixed:0.03), pitch, or gps. Train-only knob.")
    ap.add_argument("--subset", type=int, help="Use at most N panos PER SPLIT (smoke tests).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--bg-keep-frac", type=float, default=1.0,
                    help="Fraction of label-less (background) tiles to keep, thinned "
                         "deterministically (default 1.0). ~0.1-0.2 tames the tile skew.")
    # Box-size model constants (distance-aware strategies only).
    ap.add_argument("--min-frac", type=float, default=0.008)
    ap.add_argument("--max-frac", type=float, default=0.12)
    ap.add_argument("--ramp-size-m", type=float, default=1.5)
    ap.add_argument("--camera-height-m", type=float, default=2.5)
    # Geometry — defaults MATCH equirect_tiling.default_views() so train == inference.
    ap.add_argument("--n-yaw", type=int, default=6)
    ap.add_argument("--view-fov", type=float, default=90.0)
    ap.add_argument("--view-pitch", type=float, default=-30.0)
    ap.add_argument("--view-size", type=int, default=1024)
    ap.add_argument("--source-max-edge", type=int, default=4096,
                    help="Downscale the pano to this longest edge before tiling "
                         "(matches the harness's source_max_edge).")
    ap.add_argument("--pano-width", type=int, default=2048,
                    help="Width of the whole-pano image for --geometry pano (height = width/2).")
    ap.add_argument("--overlay-dir", help="Also render boxes-on-tiles for QA.")
    ap.add_argument("--overlay-n", type=int, default=20, help="Panos to overlay (default 20).")
    args = ap.parse_args()

    cfg = build_config(args)
    if (args.n_yaw, args.view_fov, args.view_pitch, args.view_size) != (6, 90.0, -30.0, 1024):
        print("WARNING: view geometry differs from default_views(); tiles won't match "
              "the harness's inference rig unless the 'yolo' provider uses the same --views.")
    for sp in set(SPLIT_MAP.values()):
        os.makedirs(os.path.join(cfg.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(cfg.out, "labels", sp), exist_ok=True)
    if cfg.overlay_dir:
        os.makedirs(cfg.overlay_dir, exist_ok=True)

    tasks = gather_tasks(args, cfg)
    print(f"{len(tasks)} panos | geometry={cfg.geometry} | box-size={args.box_size[0]}"
          f"{':' + str(cfg.fixed_frac) if cfg.strategy == 'fixed' else ''} | "
          f"workers={args.workers} | bg-keep={cfg.bg_keep_frac} -> {cfg.out}")

    agg = defaultdict(int)
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for n, st in enumerate(pool.imap_unordered(process_pano, tasks, chunksize=8), 1):
            for k, v in st.items():
                agg[k] += v
            if n % 2000 == 0:
                print(f"  {n}/{len(tasks)} panos | {agg['tiles']} imgs | {agg['boxes']} boxes")

    write_data_yaml(cfg.out)
    print(f"\nDone. panos={agg['panos']} images={agg['tiles']} boxes={agg['boxes']} "
          f"bg_images={agg['bg_tiles']} bg_skipped={agg['bg_skipped']} "
          f"points={agg['points']} read_errors={agg['read_errors']}")
    if args.box_size[0] == "gps":
        print(f"gps: fell back to pitch on {agg['gps_mismatch']} panos "
              f"(point/coord count mismatch). Trust gps only if this is near 0.")
    print(f"data.yaml -> {os.path.join(cfg.out, 'data.yaml')}")


if __name__ == "__main__":
    main()
