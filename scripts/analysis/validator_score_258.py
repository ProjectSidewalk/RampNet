"""Score every #258 candidate with the production CurbRamp validator, as Project Sidewalk serves it.

Step 2 of the two-pass verifier (#258, docs/verifier_258.md). For each candidate in
``analysis_out/verifier_258/candidates/<split>.json`` this cuts the crop the deployed
validator would see and records the validator's P(correct).

**The crop path is the serving path**, sidewalk-ai-api ``main.py`` /process at
``745e0c5`` (``SERVING_COMMIT``), not the training path. Step by step:

1. ``cv2.imread`` the panorama and resize it to 8192x4096 with ``INTER_LINEAR``, as
   ``Panorama`` does for a cached scrape (``sidewalk_ai_api/panorama.py``).
2. A 90-degree, 2048x2048 perspective centred on the candidate
   (``equirectangular_to_perspective`` below, copied verbatim from ``panorama.py``).
3. Depth-Anything-V2 metric, ``vitb`` / ``vkitti`` / max depth 80
   (``sidewalk_ai_api/depthanything.py``), on the BGR perspective; depth at the centre pixel.
4. Crop half-size ``int(6100 / depth)``, clipped to the image, resized to 640 on the long
   side with ``INTER_CUBIC``, converted to RGB.
5. ``AutoImageProcessor(backend="pil")`` + ``AutoModelForImageClassification`` at the
   ``MODEL_MAP`` revision, float32 (``sidewalk_ai_api/validator.py``).

The one deliberate difference: serving returns the argmax label and its confidence; this
records ``p_correct``, the softmax of the ``correct`` class, so candidates can be ranked.
The argmax label is kept too.

The panorama is read once per pano and reused for its candidates; serving re-reads it per
request, which gives the same array.

    python scripts/analysis/validator_score_258.py \\
        --panos-root /gscratch/makelab/jonf/rampnet_benchmark \\
        --dav2-dir <Depth-Anything-V2 checkout @ a561b84> \\
        --dav2-ckpt <depth_anything_v2_metric_vkitti_vitb.pth>

Writes ``analysis_out/verifier_258/validator_scores/<split>.json``, one row per candidate
in the candidate file's order. A split in progress is appended to ``<split>.partial.jsonl``
one pano at a time, so a preempted job resumes where it stopped (klone ``ckpt``). GPU
required. HF weights load from the cache when ``HF_HUB_OFFLINE=1``.
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)

from rampnet.eval import PINNED_SPLITS  # noqa: E402

CAND_DIR = os.path.join(REPO, "analysis_out", "verifier_258", "candidates")
OUT_DIR = os.path.join(OUT, "verifier_258", "validator_scores")

SERVING_REPO = "https://github.com/ProjectSidewalk/sidewalk-ai-api"
SERVING_COMMIT = "745e0c5"
VALIDATOR_REPO = "projectsidewalk/sidewalk-validator-ai-curbramp"
VALIDATOR_REVISION = "969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a"   # MODEL_MAP, trained 08-16-2025
DAV2_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"          # serving's submodule pin
# depth_anything_v2_metric_vkitti_vitb.pth at HF depth-anything/Depth-Anything-V2-Metric-VKITTI-Base
# @e96fd0c8 (download_models.sh); the LFS sha256 read from the Hub on 2026-10-22.
DAV2_SHA256 = "4dad67a7cc10b462bca48e6b8569c762b8eb3c1adada170a3851a6d3ba37bb3e"
DAV2_CONFIG = {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}
MAX_DEPTH = 80
PANO_W, PANO_H = 8192, 4096
FOV = 90
CROP_K = 6100
LONG_SIDE = 640
ROUND = 6


# --------------------------------------------------------------------------- #
# Verbatim from sidewalk-ai-api sidewalk_ai_api/panorama.py @745e0c5 (Apache-2.0).
# Do not edit: the point is that these crops are the ones production cuts.
# --------------------------------------------------------------------------- #
def equirectangular_to_perspective(equi_img, fov, theta, phi, height, width):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the image for PyTorch
    img = torch.tensor(equi_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    h, w = equi_img.shape[:2]

    # Compute the tangent of horizontal and vertical FOV
    hFOV = float(height) / width * fov
    w_len = torch.tan(torch.deg2rad(torch.tensor(fov / 2.0, device=device)))
    h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=device)))

    # Generate normalized 3D coordinates in the perspective image space
    x_map = torch.ones((height, width), dtype=torch.float32, device=device)
    y_map = torch.linspace(-w_len, w_len, width, device=device).repeat(height, 1)
    z_map = -torch.linspace(-h_len, h_len, height, device=device).unsqueeze(1).repeat(1, width)

    # Normalize vectors
    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1) / D.unsqueeze(-1)

    # Compute rotation matrices
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))

    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)

    # Rotate xyz coordinates
    xyz = xyz.view(-1, 3).T
    xyz = torch.matmul(R1, xyz)
    xyz = torch.matmul(R2, xyz).T
    xyz = xyz.view(height, width, 3)

    # Convert 3D coordinates to spherical coordinates
    lat = torch.asin(xyz[:, :, 2])
    lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])

    # Normalize longitude and latitude to pixel coordinates
    lon = lon / np.pi * (w - 1) / 2.0 + (w - 1) / 2.0
    lat = lat / (np.pi / 2.0) * (h - 1) / 2.0 + (h - 1) / 2.0

    # Flip latitude to correct the upside-down issue
    lat = h - lat

    # Normalize to range [-1, 1] for PyTorch grid sampling
    lon = (lon / ((w - 1) / 2.0)) - 1
    lat = (lat / ((h - 1) / 2.0)) - 1

    grid = torch.stack((lon, lat), dim=-1).unsqueeze(0)

    # Sample perspective image from equirectangular image
    persp = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert back to numpy image
    return (persp[0].permute(1, 2, 0) * 255).byte().cpu().numpy()


def get_perspective_center_params_from_equi_point(label_x, label_y, equi_width, equi_height):
    import numpy as np
    # Convert equirectangular coordinates to spherical coordinates
    lon = (label_x / equi_width - 0.5) * 2 * np.pi
    lat = (0.5 - label_y / equi_height) * np.pi

    # Convert spherical angles to degrees for yaw and pitch
    theta = np.degrees(lon) # Yaw (theta) is longitude
    phi = np.degrees(lat)   # Pitch (phi) is latitude

    return theta, phi
# --------------------------------------------------------------------------- #


def crop_box(depth_center, img_w, img_h):
    """``(start_x, start_y, end_x, end_y, half)`` exactly as ``main.py`` computes them."""
    center_x, center_y = img_w // 2, img_h // 2
    half = int((1 / depth_center) * CROP_K)
    return (max(0, center_x - half), max(0, center_y - half),
            min(img_w, center_x + half), min(img_h, center_y + half), half)


def resize_dims(crop_w, crop_h):
    """640 on the long side, as ``main.py`` sizes it."""
    if crop_w > crop_h:
        return LONG_SIDE, int(LONG_SIDE * crop_h / crop_w)
    return int(LONG_SIDE * crop_w / crop_h), LONG_SIDE


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Scorer:
    """The depth model and the validator, loaded once."""

    def __init__(self, dav2_dir, dav2_ckpt):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        sys.path.append(os.path.join(dav2_dir, "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth = DepthAnythingV2(**{**DAV2_CONFIG, "max_depth": MAX_DEPTH})
        self.depth.load_state_dict(torch.load(dav2_ckpt, map_location=self.device))
        self.depth.to(self.device).eval()

        self.processor = AutoImageProcessor.from_pretrained(
            VALIDATOR_REPO, revision=VALIDATOR_REVISION, backend="pil")
        assert type(self.processor).__name__.endswith("Pil"), type(self.processor).__name__
        self.model = AutoModelForImageClassification.from_pretrained(
            VALIDATOR_REPO, revision=VALIDATOR_REVISION, dtype=torch.float32).to(self.device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.correct_index = next(i for i, v in self.id2label.items() if v == "correct")

    def score(self, pano_bgr, x, y):
        """One candidate (normalized x, y) on an 8192x4096 BGR pano -> result dict."""
        import cv2
        import torch
        import torch.nn.functional as F
        from PIL import Image

        height, width = pano_bgr.shape[:2]
        theta, phi = get_perspective_center_params_from_equi_point(
            x * width, y * height, width, height)
        persp = equirectangular_to_perspective(pano_bgr, FOV, theta, phi,
                                               width // 4, width // 4)
        depth = self.depth.infer_image(persp)
        img_h, img_w = persp.shape[:2]
        d = float(depth[img_h // 2, img_w // 2])
        sx, sy, ex, ey, half = crop_box(depth[img_h // 2, img_w // 2], img_w, img_h)
        crop = persp[sy:ey, sx:ex]
        ch, cw = crop.shape[:2]
        rw, rh = resize_dims(cw, ch)
        crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_CUBIC)
        crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        inputs = self.processor(crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(-1).item())
        return {"p_correct": round(float(probs[self.correct_index]), ROUND),
                "pred": self.id2label[pred],
                "depth_m": round(d, 4), "crop_half": half, "crop_wh": [cw, ch]}


def load_candidates(split):
    with open(os.path.join(CAND_DIR, f"{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def read_partial(path):
    """``{pano: [rows]}`` already scored, from a partial file (last line may be torn)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break   # a line torn by preemption; that pano is re-scored
            done[rec["pano"]] = rec["rows"]
    return done


def score_split(split, scorer, panos_root, run_meta):
    import cv2
    payload = load_candidates(split)
    cands = payload["candidates"]
    by_pano = {}
    for i, c in enumerate(cands):
        by_pano.setdefault(c["pano"], []).append(i)

    os.makedirs(OUT_DIR, exist_ok=True)
    partial = os.path.join(OUT_DIR, f"{split}.partial.jsonl")
    done = read_partial(partial)
    rows = [None] * len(cands)
    for pid, prow in done.items():
        for r in prow:
            rows[r["index"]] = r
    todo = [p for p in by_pano if p not in done]
    print(f"{split}: {len(cands)} candidates on {len(by_pano)} panos; "
          f"{len(done)} panos resumed, {len(todo)} to score", flush=True)

    t0 = time.time()
    with open(partial, "a", encoding="utf-8", newline="") as out:
        for n, pid in enumerate(todo, 1):
            path = os.path.join(panos_root, split, "panos", f"{pid}.jpg")
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(path)
            native = [img.shape[1], img.shape[0]]
            img = cv2.resize(img, (PANO_W, PANO_H), interpolation=cv2.INTER_LINEAR)
            prow = []
            for i in by_pano[pid]:
                c = cands[i]
                r = {"index": i, "pano": pid, "x": c["x"], "y": c["y"], "native_wh": native,
                     **scorer.score(img, c["x"], c["y"])}
                rows[i] = r
                prow.append(r)
            out.write(json.dumps({"pano": pid, "rows": prow}, sort_keys=True) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {split} {n}/{len(todo)} panos  {time.time() - t0:.0f}s", flush=True)

    assert all(r is not None for r in rows), f"{split}: unscored candidates"
    result = {"split": split, "n_candidates": len(cands),
              "candidates_sha256": sha256_file(os.path.join(CAND_DIR, f"{split}.json")),
              "run": run_meta, "rows": rows}
    with open(os.path.join(OUT_DIR, f"{split}.json"), "w", encoding="utf-8", newline="") as f:
        head = {k: v for k, v in result.items() if k != "rows"}
        body = json.dumps(head, indent=1, sort_keys=True)[:-2]
        lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows]
        f.write(body + ',\n "rows": [\n  ' + ",\n  ".join(lines) + "\n ]\n}\n")
    os.remove(partial)
    return time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="*", default=list(PINNED_SPLITS))
    ap.add_argument("--panos-root", required=True,
                    help="directory holding <split>/panos/<pano>.jpg")
    ap.add_argument("--dav2-dir", required=True,
                    help=f"Depth-Anything-V2 checkout (serving pins {DAV2_COMMIT[:7]})")
    ap.add_argument("--dav2-ckpt", required=True,
                    help="depth_anything_v2_metric_vkitti_vitb.pth")
    ap.add_argument("--redo", action="store_true",
                    help="re-score splits whose output already exists")
    args = ap.parse_args(argv)

    import cv2
    import PIL
    import torch
    import transformers

    ckpt_sha = sha256_file(args.dav2_ckpt)
    if ckpt_sha != DAV2_SHA256:
        sys.exit(f"depth checkpoint sha256 {ckpt_sha} != serving's {DAV2_SHA256}")
    dav2_head = git_head(args.dav2_dir)
    if dav2_head != DAV2_COMMIT:
        sys.exit(f"Depth-Anything-V2 checkout at {dav2_head}, serving pins {DAV2_COMMIT}")
    if not torch.cuda.is_available():
        sys.exit("no GPU")

    run_meta = {
        "script": "scripts/analysis/validator_score_258.py",
        "serving": f"{SERVING_REPO}@{SERVING_COMMIT}",
        "validator": f"{VALIDATOR_REPO}@{VALIDATOR_REVISION}",
        "depth": {"model": "Depth-Anything-V2 metric vkitti vitb", "commit": DAV2_COMMIT,
                  "ckpt_sha256": DAV2_SHA256, "max_depth": MAX_DEPTH},
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "opencv": cv2.__version__,
                     "pillow": PIL.__version__},
        "date": datetime.date.today().isoformat(),
    }
    print(json.dumps(run_meta, indent=1), flush=True)
    scorer = Scorer(args.dav2_dir, args.dav2_ckpt)
    print(f"id2label {scorer.id2label}", flush=True)

    for split in args.splits:
        if os.path.exists(os.path.join(OUT_DIR, f"{split}.json")) and not args.redo:
            print(f"{split}: done, skipping", flush=True)
            continue
        secs = score_split(split, scorer, args.panos_root, run_meta)
        print(f"{split}: wrote in {secs:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
