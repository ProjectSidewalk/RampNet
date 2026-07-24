"""Visual QA for a VLM detector: overlay its boxes on the perspective views.

`dump_views.py` proves the *reprojection* is right. This proves the *box mapping*
is right — the other half of the pipeline, and the one with a silent failure mode:
box coordinate conventions differ across providers and even across Qwen
generations (Qwen3-VL normalizes to 0-1000, Qwen2.5-VL emits absolute pixels), and
at a ~1000px view size a wrong choice does not crash, it just slides every
detection toward the top-left by a constant factor. That is invisible in a P/R
table and obvious in one overlay.

For one pano it renders each rectilinear view with the model's raw predictions
(red) and the pano's ground-truth ramps (green) / ignore points (amber) projected
into the same view. Predictions should sit on the ramps.

Three prediction shapes are drawn, because the models disagree on what they emit:
**boxes** (Gemini, Qwen), **scored boxes** (OWLv2, Grounding DINO — the score is
printed, since these are the models whose threshold we tune), and **points**
(Molmo, whose scale convention is exactly the kind of thing this script exists to
catch: Molmo 1 emits percentages, Molmo 2 emits 0-1000).

    python scripts/model_comparison/dump_detections.py benchmark/richmond \
        --model qwen:Qwen/Qwen3-VL-8B-Instruct --out view_dump/qwen
    python scripts/model_comparison/dump_detections.py benchmark/richmond \
        --model owlv2 --out view_dump/owlv2

Needs the model's credentials/weights — it makes real calls (one per view).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rampnet.detection_eval import build_ground_truth  # noqa: E402
from equirect_tiling import (  # noqa: E402
    default_views, equirect_to_perspective, equirect_point_to_perspective)
from detectors import build_detector, load_pano_image, parse_model_spec  # noqa: E402
from dump_views import _contact_sheet  # noqa: E402
from compare import load_bundle, load_dotenv  # noqa: E402

GT_COLOR = (60, 220, 90)
IGNORE_COLOR = (240, 190, 60)
BOX_COLOR = (255, 70, 70)


def detections_to_view_shapes(detector, raw, width, height):
    """Provider raw item -> a drawable shape in view pixels.

    ``("rect", x1, y1, x2, y2, score_or_None)`` or ``("point", x, y, score_or_None)``.
    Mirrors what ``detector._parse`` does to get centers, but keeps the full box (or
    the bare point) so a coordinate-space mistake is visible as a whole shape in the
    wrong place rather than as a slightly-off center."""
    shapes = []
    for it in raw:
        if "box_2d" in it:                      # Gemini: [ymin, xmin, ymax, xmax], 0-1000
            ymin, xmin, ymax, xmax = it["box_2d"]
            shapes.append(("rect", xmin / 1000.0 * width, ymin / 1000.0 * height,
                           xmax / 1000.0 * width, ymax / 1000.0 * height, None))
        elif "bbox_2d" in it:                   # Qwen: [x1, y1, x2, y2]
            x1, y1, x2, y2 = it["bbox_2d"]
            if getattr(detector, "coord_space", "norm1000") == "norm1000":
                sx = sy = 1000.0
            else:
                sx, sy = width, height
            shapes.append(("rect", x1 / sx * width, y1 / sy * height,
                           x2 / sx * width, y2 / sy * height, None))
        elif "box" in it:                       # OWLv2 / Grounding DINO: pixels + score
            x1, y1, x2, y2 = it["box"]
            shapes.append(("rect", x1, y1, x2, y2, it.get("score")))
        elif "point" in it:                     # Molmo: already normalized to the view
            x, y = it["point"]
            shapes.append(("point", x * width, y * height, it.get("score")))
    return shapes


def _marker(draw, x, y, color, r=14):
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    draw.line([x - r, y, x + r, y], fill=color, width=1)
    draw.line([x, y - r, x, y + r], fill=color, width=1)


def _crosshair(draw, cx, cy, color, arm=10, width=2):
    draw.line([cx - arm, cy, cx + arm, cy], fill=color, width=width)
    draw.line([cx, cy - arm, cx, cy + arm], fill=color, width=width)


def overlay(image, shapes, gt_uv, ignore_uv):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for (x, y) in ignore_uv:
        _marker(draw, x, y, IGNORE_COLOR)
    for (x, y) in gt_uv:
        _marker(draw, x, y, GT_COLOR)
    for shape in shapes:
        if shape[0] == "rect":
            _, x1, y1, x2, y2, score = shape
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=4)
            _crosshair(draw, (x1 + x2) / 2, (y1 + y2) / 2, BOX_COLOR)
            if score is not None:
                draw.text((x1 + 3, max(0, y1 - 12)), f"{score:.2f}", fill=BOX_COLOR)
        else:
            _, x, y, score = shape
            # A point prediction has no extent, so draw the marker style used for GT
            # (in red) — same visual weight, so a scale error is as obvious as a box's.
            _marker(draw, x, y, BOX_COLOR, r=10)
            _crosshair(draw, x, y, BOX_COLOR, arm=16, width=1)
            if score is not None:
                draw.text((x + 12, y + 12), f"{score:.2f}", fill=BOX_COLOR)
    return image


def _project(points, view):
    """Pano-normalized points -> in-view pixel coords, dropping those out of frame."""
    out = []
    for (x, y) in points:
        uv = equirect_point_to_perspective(x, y, view)
        if uv is not None:
            out.append((uv[0] * view.width, uv[1] * view.height))
    return out


def _pick_pano(verdicts, panos_dir, requested):
    if requested:
        return requested
    for pid in verdicts:
        if os.path.exists(os.path.join(panos_dir, f"{pid}.jpg")):
            return pid
    raise SystemExit(f"no reviewed pano image found under {panos_dir} "
                     "(bundle panos/ are git-ignored; they must be present locally)")


def main():
    ap = argparse.ArgumentParser(
        description="Overlay a detector's boxes on a pano's perspective views (visual QA).")
    ap.add_argument("bundle", help="Bundle dir (benchmark/<city>).")
    ap.add_argument("--model", default="qwen",
                    help="One detector spec: provider or provider:model_id "
                         "(e.g. qwen:Qwen/Qwen3-VL-8B-Instruct, gemini:gemini-3.6-flash, "
                         "owlv2, gdino, molmo:allenai/Molmo2-8B).")
    ap.add_argument("--pano", help="Pano id (default: first reviewed pano with an image).")
    ap.add_argument("--out", default="view_dump/detections", help="Output directory.")
    ap.add_argument("--source-max-edge", type=int, default=4096)
    # Consumed by build_detector.
    ap.add_argument("--gemini-model", default="gemini-3.6-flash")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--qwen-coord-space", choices=["auto", "norm1000", "pixels"], default="auto")
    ap.add_argument("--owlv2-model", default="google/owlv2-large-patch14-ensemble")
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--molmo-model", default="allenai/Molmo2-8B")
    ap.add_argument("--owlv2-query")
    ap.add_argument("--gdino-query")
    ap.add_argument("--gdino-text-threshold", type=float)
    ap.add_argument("--score-threshold", type=float,
                    help="Score floor for owlv2/gdino (default 0.05). Raise it to declutter "
                         "the overlay; the harness's own floor is unaffected.")
    ap.add_argument("--molmo-coord-scale", choices=["auto", "100", "1000"], default="auto")
    ap.add_argument("--tiling", choices=["perspective"], default="perspective",
                    help="Only the tiled path is worth eyeballing.")
    args = ap.parse_args()

    load_dotenv(str(REPO_ROOT))
    records, verdicts, panos_dir = load_bundle(args.bundle)
    pano_id = _pick_pano(verdicts, panos_dir, args.pano)
    pano_path = os.path.join(panos_dir, f"{pano_id}.jpg")
    if not os.path.exists(pano_path):
        raise SystemExit(f"pano image not found: {pano_path}")

    provider, model_id = parse_model_spec(args.model)
    label, detector = build_detector(provider, model_id, records, args)
    if not hasattr(detector, "_raw_detect"):
        raise SystemExit(f"'{label}' has no boxes to draw (it reads points from the bundle)")
    detector.prepare()

    entry = verdicts[pano_id]
    gt = build_ground_truth(records[pano_id]["detections"], entry["dets"],
                            entry["missed"], entry["no_missed"])

    os.makedirs(args.out, exist_ok=True)
    pano = load_pano_image(pano_path, args.source_max_edge)
    views = detector._views or default_views()
    rendered, total_preds = [], 0
    for i, view in enumerate(views):
        view_img = equirect_to_perspective(pano, view)
        raw = detector._raw_detect(view_img)
        shapes = detections_to_view_shapes(detector, raw, view.width, view.height)
        total_preds += len(shapes)
        gt_uv = _project(gt.gt_points, view)
        ignore_uv = _project(gt.ignore_points, view)
        overlay(view_img, shapes, gt_uv, ignore_uv)
        name = f"view_{i}_yaw{int(view.yaw_deg)}_preds{len(shapes)}.png"
        view_img.save(os.path.join(args.out, name))
        rendered.append(view_img)
        print(f"  view {i} (yaw {int(view.yaw_deg)}): {len(shapes)} prediction(s), "
              f"{len(gt_uv)} GT / {len(ignore_uv)} ignore point(s) in frame")

    sheet_path = os.path.join(args.out, f"contact_{label.replace('/', '_')}_{pano_id}.png")
    _contact_sheet(rendered).save(sheet_path)
    print(f"\npano {pano_id}: {total_preds} raw prediction(s) across {len(views)} views vs "
          f"{len(gt.gt_points)} GT ramp(s) ({len(gt.ignore_points)} ignored)")
    print(f"Contact sheet: {sheet_path}")
    print("Predictions (red) should sit on ramps; a constant offset toward the top-left means "
          "the coordinate space is wrong (see --qwen-coord-space / --molmo-coord-scale).")


if __name__ == "__main__":
    main()
