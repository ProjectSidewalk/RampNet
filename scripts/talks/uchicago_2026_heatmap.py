"""Run the released RampNet checkpoint on one benchmark panorama and save its heatmap.

The figure script (``uchicago_2026_figures.py --only heatmap``) renders from the saved ``.npz``
so the slide regenerates on CPU with no checkpoint; this script is the one GPU step, and the
npz records what produced it (checkpoint snapshot, panorama sha256, date).

    python scripts/talks/uchicago_2026_heatmap.py --split paterson --pano tekhQ4HQ9pcOqs_kGGpGaw \
        --pano-dir benchmark/paterson/panos

Inference is the deployed path: resize to 2048x4096, ImageNet normalisation, one forward pass,
output clipped to [0, 1] (``stage_two/demo.py``), no flip-TTA (the city benchmark records carry
none). Peaks are extracted at render time with ``skimage.peak_local_max`` (min_distance 10) so
the threshold is a figure choice, not baked into the data.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026")
MODEL_INPUT_SIZE = (2048, 4096)
HF_MODEL = "projectsidewalk/rampnet-model"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--pano", required=True, help="panorama id (file stem in --pano-dir)")
    ap.add_argument("--pano-dir", required=True)
    ap.add_argument("--checkpoint", default=HF_MODEL,
                    help="local .pth or a Hub repo id (default: the released weights)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if os.path.exists(args.checkpoint):
        sys.path.insert(0, REPO)
        from rampnet.loading import checkpoint_fingerprint, load_checkpoint
        from rampnet.model import KeypointModel
        model = KeypointModel(heatmap_size=(512, 1024))
        load_checkpoint(model, args.checkpoint, map_location=device)
        ckpt_desc = f"{args.checkpoint} ({checkpoint_fingerprint(args.checkpoint)})"
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
        ckpt_desc = f"{args.checkpoint} @ {getattr(model.config, '_commit_hash', 'unknown')}"
    model.to(device).eval()

    pano_path = os.path.join(args.pano_dir, f"{args.pano}.jpg")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(pano_path)
    img.draft("RGB", MODEL_INPUT_SIZE[::-1])
    img = img.convert("RGB").resize(MODEL_INPUT_SIZE[::-1], Image.LANCZOS)
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
        else:
            out = model(x)
    heat = np.clip(out.float().squeeze().cpu().numpy(), 0, 1).astype(np.float16)
    assert heat.shape == (512, 1024), heat.shape

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"heatmap_{args.split}_{args.pano}.npz")
    meta = dict(split=args.split, pano_id=args.pano, pano_sha256=sha256_of(pano_path),
                checkpoint=ckpt_desc, device=device.type, autocast_fp16=device.type == "cuda",
                input_size=list(MODEL_INPUT_SIZE), tta=False,
                run_date=datetime.date.today().isoformat())
    np.savez_compressed(out_path, heatmap=heat, meta=json.dumps(meta))
    from skimage.feature import peak_local_max
    peaks = peak_local_max(heat.astype(np.float32), min_distance=10, threshold_abs=0.30,
                           exclude_border=False)
    print(f"wrote {os.path.relpath(out_path, REPO)}  max {float(heat.max()):.3f}  "
          f"peaks>=0.30 {len(peaks)}  >=0.55 {int((heat[tuple(peaks.T)] >= 0.55).sum()) if len(peaks) else 0}")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
