"""A short video of RampNet heatmaps on benchmark panoramas where it was near perfect.

Each frame: the raw panorama on top, and below it the same panorama dimmed with the model's
heatmap and a ring on every peak at the deployed threshold, with the city, imagery source and
the reviewer's verdict in a caption. The selection is mechanical, from the committed verdicts:
panoramas where every detection was confirmed and the reviewer found nothing missed, plus a few
with no ramps and no detections (true negatives).

    python scripts/talks/uchicago_2026_showcase.py select
    python scripts/talks/uchicago_2026_showcase.py infer  --pano-root benchmark      # GPU
    python scripts/talks/uchicago_2026_showcase.py render --pano-root benchmark
    python scripts/talks/uchicago_2026_showcase.py video                              # ffmpeg

Committed: the manifest (which panoramas, why) and one 8-bit heatmap PNG per panorama, so the
frames and the mp4 regenerate on CPU without the checkpoint. Frames and the mp4 are not
committed (``.gitignore`` in the showcase folder); ``video`` rebuilds them in a minute.
"""
import argparse
import datetime
import json
import os
import random
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHOW_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "showcase")
MANIFEST = os.path.join(SHOW_DIR, "manifest.json")
FRAMES_DIR = os.path.join(SHOW_DIR, "frames")
HF_MODEL = "projectsidewalk/rampnet-model"
DEPLOYED_T = 0.55

# Imagery source and place per split, from benchmark/README.md.
SPLITS = {
    "bend":               ("Bend, OR, US",            "Google Street View (training city)"),
    "paterson":           ("Paterson, NJ, US",        "Google Street View"),
    "gainesville":        ("Gainesville, FL, US",     "Google Street View"),
    "sao_paulo":          ("São Paulo, Brazil",       "Google Street View"),
    "laurens_gsv":        ("Laurens, IA, US",         "Google Street View"),
    "richmond":           ("Richmond, VA, US",        "Mapillary 360° (iSTAR Pulsar / GoPro Max)"),
    "annapolis":          ("Annapolis, MD, US",       "Mapillary 360° (Trimble MX7)"),
    "morgantown":         ("Morgantown, WV, US",      "Mapillary 360° (GoPro Max)"),
    "clovis":             ("Clovis, CA, US",          "Mapillary 360° (GoPro Fusion)"),
    "laurens_mapillary":  ("Laurens, IA, US",         "Mapillary 360° (GoPro Max)"),
    "budapest_district5": ("Budapest, Hungary",       "Mapillary 360° (GoPro Max)"),
}
PER_SPLIT = 3          # top-2 by ramp count plus one seeded random, for variety of scenes
NEGATIVES = 4          # true-negative panoramas, spread across splits
SEED = 2026


def load_verdicts(split):
    with open(os.path.join(REPO, "benchmark", split, "verdicts.json"), encoding="utf-8") as f:
        return json.load(f)["panos"]


def load_records(split):
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        return {json.loads(l)["pano"]["panorama_id"]: json.loads(l) for l in f}


def cmd_select(args):
    rng = random.Random(SEED)
    items, negatives = [], []
    for split in SPLITS:
        v = load_verdicts(split)
        perfect = sorted(((sum(1 for d in vv["dets"] if d is True), pid) for pid, vv in v.items()
                          if vv["dets"] and all(d is True for d in vv["dets"])
                          and not vv.get("missed")), reverse=True)
        perfect = [(n, pid) for n, pid in perfect if n >= 3]
        pick = perfect[:2]
        rest = perfect[2:]
        if rest:
            pick.append(rng.choice(rest))
        for n, pid in pick[:PER_SPLIT]:
            items.append(dict(split=split, pano_id=pid, n_ramps=n, kind="all_found",
                              why=f"{n} ramps, every detection confirmed, nothing missed"))
        tn = sorted(pid for pid, vv in v.items() if not vv["dets"] and not vv.get("missed"))
        if tn:
            negatives.append(dict(split=split, pano_id=rng.choice(tn), n_ramps=0,
                                  kind="true_negative",
                                  why="no curb ramps, and the model fired nowhere"))
    rng.shuffle(negatives)
    items += negatives[:NEGATIVES]
    # Order: alternate imagery sources so the video does not run city by city.
    rng.shuffle(items)
    os.makedirs(SHOW_DIR, exist_ok=True)
    manifest = dict(selected=datetime.date.today().isoformat(), rule=(
        f"per split: the two 'all_found' panoramas with the most confirmed ramps (>= 3) plus one "
        f"seeded-random other; then {NEGATIVES} true negatives (no ramps, no detections), "
        f"seed {SEED}; from benchmark/<split>/verdicts.json"), items=items)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(SHOW_DIR, ".gitignore"), "w", newline="\n") as f:
        f.write("frames/\n*.mp4\n")
    print(f"{len(items)} panoramas -> {os.path.relpath(MANIFEST, REPO)}")


def _heat_path(it):
    return os.path.join(SHOW_DIR, f"heat_{it['split']}_{it['pano_id']}.png")


def _pano_path(root, it):
    return os.path.join(root, it["split"], "panos", f"{it['pano_id']}.jpg")


def cmd_infer(args):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    ckpt = f"{args.checkpoint} @ {getattr(model.config, '_commit_hash', 'unknown')}"
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    Image.MAX_IMAGE_PIXELS = None
    for it in manifest["items"]:
        out = _heat_path(it)
        if os.path.exists(out) and not args.force:
            continue
        img = Image.open(_pano_path(args.pano_root, it))
        img.draft("RGB", (4096, 2048))
        img = img.convert("RGB").resize((4096, 2048), Image.LANCZOS)
        x = norm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    y = model(x)
            else:
                y = model(x)
        heat = np.clip(y.float().squeeze().cpu().numpy(), 0, 1)
        Image.fromarray((heat * 255).round().astype(np.uint8)).save(out)
        it["max"] = round(float(heat.max()), 3)
        print(it["split"], it["pano_id"], "max", it["max"], flush=True)
    manifest["inference"] = dict(checkpoint=ckpt, device=device.type, autocast_fp16=device.type == "cuda",
                                 input_size=[2048, 4096], tta=False,
                                 run_date=datetime.date.today().isoformat())
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


# Frame geometry: 1920x1080. Two panels of the panorama's street band (the sky above and the
# vehicle roof below carry no ramps), each 1920 x 442, with a 196 px caption block.
FRAME_W, FRAME_H = 1920, 1080
BAND = (0.30, 0.76)       # fraction of the equirect height kept
PANEL_H = int(FRAME_W / 2 * (BAND[1] - BAND[0]))
CAPTION_H = FRAME_H - 2 * PANEL_H


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(it, pano_path, heat_path, index, total):
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import cm
    from skimage.feature import peak_local_max

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(pano_path)
    img.draft("RGB", (4096, 2048))
    img = img.convert("RGB").resize((4096, 2048), Image.BILINEAR)
    y0, y1 = int(BAND[0] * 2048), int(BAND[1] * 2048)
    band = img.crop((0, y0, 4096, y1)).resize((FRAME_W, PANEL_H), Image.LANCZOS)

    heat = np.asarray(Image.open(heat_path), dtype=np.float32) / 255.0      # 512 x 1024
    peaks = peak_local_max(heat, min_distance=10, threshold_abs=DEPLOYED_T, exclude_border=False)

    # Bottom panel: the panorama dimmed and cooled, the heatmap in jet, alpha from its value.
    base = np.asarray(band, dtype=np.float32)
    tint = base * 0.42 + np.array([18, 22, 90], dtype=np.float32) * 0.58
    hb = heat[int(BAND[0] * 512):int(BAND[1] * 512)]
    hb_img = Image.fromarray((hb * 255).astype(np.uint8)).resize((FRAME_W, PANEL_H), Image.BICUBIC)
    hb = np.asarray(hb_img, dtype=np.float32) / 255.0
    rgba = cm.jet(hb)[..., :3] * 255
    alpha = np.clip(hb * 1.6, 0, 0.92)[..., None]
    comp = tint * (1 - alpha) + rgba * alpha
    bottom = Image.fromarray(comp.round().astype(np.uint8))
    draw = ImageDraw.Draw(bottom)
    for r, c in peaks:
        x = (c + 0.5) / 1024 * FRAME_W
        y = ((r + 0.5) / 512 - BAND[0]) / (BAND[1] - BAND[0]) * PANEL_H
        if 0 <= y <= PANEL_H:
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline=(90, 240, 255), width=5)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(90, 240, 255))

    frame = Image.new("RGB", (FRAME_W, FRAME_H), (12, 12, 14))
    frame.paste(band, (0, 0))
    frame.paste(bottom, (0, PANEL_H))
    d = ImageDraw.Draw(frame)
    place, imagery = SPLITS[it["split"]]
    n_peaks = len(peaks)
    if it["kind"] == "true_negative":
        verdict = "no curb ramps here, and no detections"
    else:
        verdict = (f"{it['n_ramps']} curb ramps, {n_peaks} detected, every one confirmed by the "
                   f"reviewer, none missed")
    y_cap = 2 * PANEL_H
    d.text((40, y_cap + 42), place, font=_font(54), fill=(245, 245, 240))
    d.text((40, y_cap + 112), imagery, font=_font(30), fill=(160, 160, 155))
    d.text((FRAME_W - 40, y_cap + 44), verdict, font=_font(34), fill=(130, 215, 255), anchor="ra")
    d.text((FRAME_W - 40, y_cap + 112), f"{index + 1} / {total}", font=_font(28),
           fill=(130, 130, 125), anchor="ra")
    return frame


def cmd_render(args):
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    items = manifest["items"]
    for i, it in enumerate(items):
        frame = render_frame(it, _pano_path(args.pano_root, it), _heat_path(it), i, len(items))
        frame.save(os.path.join(FRAMES_DIR, f"frame_{i:03d}.png"))
        print("frame", i, it["split"], it["pano_id"], flush=True)


def cmd_video(args):
    with open(MANIFEST, encoding="utf-8") as f:
        items = json.load(f)["items"]
    lst = os.path.join(FRAMES_DIR, "frames.txt")
    with open(lst, "w", newline="\n") as f:
        for i in range(len(items)):
            f.write(f"file 'frame_{i:03d}.png'\nduration {args.seconds}\n")
        f.write(f"file 'frame_{len(items) - 1:03d}.png'\n")
    out = os.path.join(SHOW_DIR, "showcase.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    print("wrote", os.path.relpath(out, REPO), f"({os.path.getsize(out) / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    p = sub.add_parser("infer")
    p.add_argument("--pano-root", required=True, help="directory holding <split>/panos/")
    p.add_argument("--checkpoint", default=HF_MODEL)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("--pano-root", required=True)
    p = sub.add_parser("video")
    p.add_argument("--seconds", type=float, default=2.5, help="seconds per panorama")
    args = ap.parse_args(argv)
    {"select": cmd_select, "infer": cmd_infer, "render": cmd_render, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
