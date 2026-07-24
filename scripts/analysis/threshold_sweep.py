"""Recall-first operating-point sweep for RampNet on the benchmark.

Re-runs RampNet inference on every benchmark pano ONCE, then re-extracts peaks at a
grid of (threshold_abs, min_distance) and re-scores each against the same derived
ground truth. Tests the two untested cheap recall levers:
  - threshold below the deployment floor of 0.55 (never measured)
  - min_distance below 10 (10 heatmap px = 40 pano px, vs a 90 px match radius, so
    adjacent ramps can collapse into one peak)

Inference replicates sidewalk-auto-labeler/detectors/curb_ramp.py exactly
(resize 2048x4096 bilinear, ImageNet norm, no TTA) so (0.55, 10) should reproduce
the committed records.jsonl numbers.
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image
from skimage.feature import peak_local_max
from torchvision import transforms

from rampnet.detection_eval import build_ground_truth, score_pano, aggregate, radius_sq_for
from rampnet.model import KeypointModel

Image.MAX_IMAGE_PIXELS = None
THRESHOLDS = [0.55, 0.45, 0.35, 0.25, 0.15]
MIN_DISTS = [10, 5, 3]
RSQ = radius_sq_for()

PRE = transforms.Compose([
    transforms.Resize((2048, 4096), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    from huggingface_hub import hf_hub_download
    try:
        import safetensors.torch as st
        sd = st.load_file(hf_hub_download("projectsidewalk/rampnet-model", "model.safetensors"))
    except Exception:
        sd = torch.load(hf_hub_download("projectsidewalk/rampnet-model", "pytorch_model.bin"),
                        map_location="cpu")
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    m = KeypointModel()
    m.load_state_dict(sd)
    return m.eval()


def load_bundle(city):
    import json
    cdir = os.path.join(REPO, "benchmark", city)
    records = {}
    with open(os.path.join(cdir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    with open(os.path.join(cdir, "verdicts.json"), encoding="utf-8") as f:
        verdicts = json.load(f)["panos"]
    return records, verdicts, os.path.join(cdir, "panos")


def heatmap_for(model, device, path, use_fp16):
    img = Image.open(path).convert("RGB")
    t = PRE(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(t)
        else:
            h = model(t)
    h = h.squeeze().float().cpu().numpy()
    del t
    return h


def peaks_to_dets(h, thr, md):
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=md, threshold_abs=thr)
    H, W = h.shape
    return [(float(c / W), float(r / H), float(h[r][c])) for r, c in pk]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    use_fp16 = False
    print(f"device={device}", flush=True)

    results = {}   # (city, thr, md) -> list[PanoScore]
    for city in ("richmond", "bend"):
        records, verdicts, panos_dir = load_bundle(city)
        for i, (pid, entry) in enumerate(verdicts.items(), 1):
            gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                    entry["missed"], entry["no_missed"])
            path = os.path.join(panos_dir, f"{pid}.jpg")
            try:
                h = heatmap_for(model, device, path, use_fp16)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                use_fp16 = True
                print("  OOM -> switching to fp16 autocast", flush=True)
                h = heatmap_for(model, device, path, use_fp16)
            for thr in THRESHOLDS:
                for md in MIN_DISTS:
                    dets = peaks_to_dets(h, thr, md)
                    results.setdefault((city, thr, md), []).append(
                        score_pano(dets, gt, radius_sq=RSQ))
            if i % 20 == 0:
                print(f"  {city}: {i}/{len(verdicts)}", flush=True)
            del h
        print(f"{city} done ({len(verdicts)} panos)", flush=True)

    print(f"\nfp16={'yes' if use_fp16 else 'no (fp32)'}")
    for city in ("richmond", "bend"):
        print(f"\n{'='*74}\n{city.upper()}  operating-point sweep "
              f"(baseline = thr 0.55 / md 10)\n{'='*74}")
        print(f"{'thr':>5} {'min_d':>6} {'P':>7} {'R':>7} {'F1':>7}   {'tp/fp/fn':>16}")
        print("-" * 74)
        for thr in THRESHOLDS:
            for md in MIN_DISTS:
                r = aggregate(results[(city, thr, md)])
                tag = "  <-- baseline" if (thr == 0.55 and md == 10) else ""
                print(f"{thr:>5.2f} {md:>6} {r.precision:>7.3f} {r.recall:>7.3f} "
                      f"{r.f1:>7.3f}   {f'{r.tp}/{r.fp}/{r.fn}':>16}{tag}")


if __name__ == "__main__":
    main()
