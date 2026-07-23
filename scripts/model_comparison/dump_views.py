"""Visual QA for the perspective reprojection.

Renders a pano's rectilinear views and overlays a graticule. A correct gnomonic
projection renders every great circle (all meridians of constant longitude, and
the equator/horizon) as a **straight line**; if the reprojection were still
equirectangularly warped these would visibly bend. So: straight graticule lines +
natural-looking buildings/poles = the de-distortion is working.

    python scripts/model_comparison/dump_views.py benchmark/richmond --out <dir>
    python scripts/model_comparison/dump_views.py benchmark/bend --pano <id> --no-grid

Produces one PNG per view plus a contact sheet.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equirect_tiling import default_views, equirect_to_perspective, equirect_point_to_perspective  # noqa: E402
from detectors import load_pano_image  # noqa: E402


def _first_pano_id(bundle_dir):
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)["pano"]["panorama_id"]
    raise SystemExit("no records in bundle")


def _polyline(view, x, lat_range):
    """Project a meridian (constant pano X, varying latitude) into a view; return
    the in-view pixel points."""
    pts = []
    for i in range(lat_range[0], lat_range[1] + 1, 2):
        y = 0.5 - i / 180.0
        uv = equirect_point_to_perspective(x, y, view)
        if uv is not None:
            pts.append((uv[0] * view.width, uv[1] * view.height))
    return pts


def _equator(view):
    pts = []
    for lon_deg in range(-180, 181, 2):
        uv = equirect_point_to_perspective(lon_deg / 360.0 + 0.5, 0.5, view)
        if uv is not None:
            pts.append((uv[0] * view.width, uv[1] * view.height))
    return pts


def _overlay_graticule(img, view):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for lon_deg in range(0, 360, 15):                       # meridians (great circles)
        pts = _polyline(view, lon_deg / 360.0 + 0.5, (-80, 80))
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 90, 90), width=1)
    eq = _equator(view)                                     # equator / horizon
    if len(eq) >= 2:
        draw.line(eq, fill=(90, 160, 255), width=2)
    return img


def _contact_sheet(images, cols=3):
    from PIL import Image
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def main():
    ap = argparse.ArgumentParser(description="Render + graticule-overlay a pano's perspective views.")
    ap.add_argument("bundle", help="Bundle dir (benchmark/<city>).")
    ap.add_argument("--pano", help="Pano id (default: first in the bundle).")
    ap.add_argument("--out", default="view_dump", help="Output directory.")
    ap.add_argument("--no-grid", action="store_true", help="Skip the graticule overlay.")
    ap.add_argument("--source-max-edge", type=int, default=4096)
    args = ap.parse_args()

    pano_id = args.pano or _first_pano_id(args.bundle)
    pano_path = os.path.join(args.bundle, "panos", f"{pano_id}.jpg")
    if not os.path.exists(pano_path):
        raise SystemExit(f"pano image not found: {pano_path} (bundle panos/ are git-ignored; "
                         "they must be present locally)")

    os.makedirs(args.out, exist_ok=True)
    pano = load_pano_image(pano_path, args.source_max_edge)
    views = default_views()
    rendered = []
    for i, view in enumerate(views):
        im = equirect_to_perspective(pano, view)
        if not args.no_grid:
            im = _overlay_graticule(im, view)
        name = f"view_{i}_yaw{int(view.yaw_deg)}_pitch{int(view.pitch_deg)}.png"
        im.save(os.path.join(args.out, name))
        rendered.append(im)

    sheet_path = os.path.join(args.out, f"contact_{pano_id}.png")
    _contact_sheet(rendered).save(sheet_path)
    print(f"Wrote {len(rendered)} views + contact sheet to {args.out}/ (pano {pano_id})")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
