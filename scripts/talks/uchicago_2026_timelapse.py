"""A corner over time: the same Bend intersection across Google Street View captures, with the
city's inventory saying when the curb ramps were installed and RampNet's heatmap on each capture.

    python scripts/talks/uchicago_2026_timelapse.py candidates          # rank corners (CPU, offline)
    python scripts/talks/uchicago_2026_timelapse.py fetch  --pano 0CvMs02xHlg3mCMHJdc6Xg   # GSV metadata + tiles
    python scripts/talks/uchicago_2026_timelapse.py infer  --pano 0CvMs02xHlg3mCMHJdc6Xg   # GPU
    python scripts/talks/uchicago_2026_timelapse.py render --pano 0CvMs02xHlg3mCMHJdc6Xg

Inputs: ``benchmark/bend/records.jsonl`` (each panorama's GSV history: prior pano ids and
dates), ``stage_one/dataset_generation/location_data/bend.geojson`` (the inventory, with
``InstallDate``), the GSV metadata endpoint and tile server in ``stage_one/dataset_generation/
search_panos.py`` and ``rampnet/gsv.py`` (the production Stage 1 path), and the released
checkpoint. Committed: the manifest (pano ids, dates, headings, positions, tile sha256) and one
8-bit heatmap PNG per capture. The historical panoramas themselves are cached under
``timelapse/panos/`` and gitignored; ``fetch`` restores them and checks the sha256.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "stage_one", "dataset_generation"))
TL_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "timelapse")
PANO_DIR = os.path.join(TL_DIR, "panos")
RECORDS = os.path.join(REPO, "benchmark", "bend", "records.jsonl")
INVENTORY = os.path.join(REPO, "stage_one", "dataset_generation", "location_data", "bend.geojson")
HF_MODEL = "projectsidewalk/rampnet-model"
RADIUS_M = 25.0
DEPLOYED_T = 0.55


def _dist(lat1, lng1, lat2, lng2):
    return math.hypot((lng2 - lng1) * 111320.0 * math.cos(math.radians(lat1)), (lat2 - lat1) * 111320.0)


def _bearing(lat1, lng1, lat2, lng2):
    dx = (lng2 - lng1) * 111320.0 * math.cos(math.radians(lat1))
    dy = (lat2 - lat1) * 111320.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


def load_inventory():
    with open(INVENTORY, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    out = []
    for ft in feats:
        if not ft["geometry"]:
            continue
        ms = ft["properties"].get("InstallDate")
        if not ms or ms < 0:
            continue
        d = datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc)
        out.append(dict(lat=ft["geometry"]["coordinates"][1], lng=ft["geometry"]["coordinates"][0],
                        installed=d.strftime("%Y-%m"), facility_id=ft["properties"].get("FacilityID")))
    return out


def load_records():
    with open(RECORDS, encoding="utf-8") as f:
        return {json.loads(l)["pano"]["panorama_id"]: json.loads(l)["pano"] for l in f}


def cmd_candidates(args):
    """Benchmark panoramas whose GSV history brackets at least one inventory install date."""
    inv = load_inventory()
    rows = []
    for pid, p in load_records().items():
        hist = p.get("history", [])
        if not hist:
            continue
        dates = sorted([h["date"] for h in hist] + [p["capture_date"]])
        near = [r for r in inv if _dist(p["lat"], p["lng"], r["lat"], r["lng"]) <= RADIUS_M]
        flips = [r for r in near if dates[0] < r["installed"] <= dates[-1]]
        if flips:
            rows.append((len(flips), pid, dates, sorted({r["installed"] for r in flips})))
    rows.sort(reverse=True)
    for n, pid, dates, inst in rows[:args.top]:
        print(f"{pid}  {n:2d} ramps installed {inst}  captures {dates}")


def _manifest_path(pano):
    return os.path.join(TL_DIR, f"manifest_{pano}.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args):
    import cv2
    from search_panos import _get_metadata, _parse_pano_date, _parse_pano_heading
    from rampnet.gsv import fetch_panorama

    p = load_records()[args.pano]
    captures = [dict(pano_id=args.pano, date=p["capture_date"], heading=p["camera_heading"],
                     lat=p["lat"], lng=p["lng"], source="benchmark record")]
    for h in p.get("history", []):
        m = _get_metadata(h["pano_id"])
        node = m[1][0][5][0][1][0]        # [None, None, lat, lng], observed 2026-10-05
        captures.append(dict(pano_id=h["pano_id"], date=_parse_pano_date(m).replace("-", "-0")[:7]
                             if len(_parse_pano_date(m)) == 6 else _parse_pano_date(m),
                             heading=_parse_pano_heading(m), lat=float(node[2]), lng=float(node[3]),
                             source="GetMetadata"))
    captures.sort(key=lambda c: c["date"])
    os.makedirs(PANO_DIR, exist_ok=True)
    for c in captures:
        out = os.path.join(PANO_DIR, f"{c['pano_id']}.jpg")
        if not os.path.exists(out):
            img = fetch_panorama(c["pano_id"])          # 2048x4096 BGR, the production fetch
            if img is None:
                raise SystemExit(f"fetch failed for {c['pano_id']}")
            cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        c["sha256"] = _sha256(out)
        print(c["date"], c["pano_id"], "heading", round(c["heading"], 1), flush=True)
    inv = load_inventory()
    ramps = [dict(r, dist_m=round(_dist(p["lat"], p["lng"], r["lat"], r["lng"]), 1))
             for r in inv if _dist(p["lat"], p["lng"], r["lat"], r["lng"]) <= RADIUS_M]
    manifest = dict(pano=args.pano, fetched=datetime.date.today().isoformat(), radius_m=RADIUS_M,
                    captures=captures, ramps=sorted(ramps, key=lambda r: r["installed"]))
    os.makedirs(TL_DIR, exist_ok=True)
    with open(_manifest_path(args.pano), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    with open(os.path.join(TL_DIR, ".gitignore"), "w", newline="\n") as f:
        f.write("panos/\n")
    print(f"{len(captures)} captures, {len(ramps)} inventory ramps within {RADIUS_M:.0f} m")


def _heat_path(pano_id):
    return os.path.join(TL_DIR, f"heat_{pano_id}.png")


def cmd_infer(args):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    with open(_manifest_path(args.pano), encoding="utf-8") as f:
        manifest = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True).to(device).eval()
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for c in manifest["captures"]:
        img = Image.open(os.path.join(PANO_DIR, f"{c['pano_id']}.jpg")).convert("RGB")
        assert img.size == (4096, 2048), img.size
        x = norm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    y = model(x)
            else:
                y = model(x)
        heat = np.clip(y.float().squeeze().cpu().numpy(), 0, 1)
        Image.fromarray((heat * 255).round().astype(np.uint8)).save(_heat_path(c["pano_id"]))
        c["heat_max"] = round(float(heat.max()), 3)
        print(c["date"], c["pano_id"], "max", c["heat_max"], flush=True)
    manifest["inference"] = dict(checkpoint=f"{HF_MODEL} @ {getattr(model.config, '_commit_hash', 'unknown')}",
                                 device=device.type, autocast_fp16=device.type == "cuda", tta=False,
                                 run_date=datetime.date.today().isoformat())
    with open(_manifest_path(args.pano), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


def cmd_render(args):
    import numpy as np
    from PIL import Image
    from skimage.feature import peak_local_max
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO, "scripts", "talks"))
    from uchicago_2026_figures import (_footnote, _heat_overlay, _save, _titles, INK, INK_SECONDARY,
                                       SURFACE, BLUE)

    with open(_manifest_path(args.pano), encoding="utf-8") as f:
        manifest = json.load(f)
    caps = manifest["captures"]
    if args.dates:
        caps = [c for c in caps if c["date"] in args.dates]
    ramps = manifest["ramps"]
    b0, b1 = 0.33, 0.72
    n = len(caps)
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(n, 1, left=0.012, right=0.988, top=0.86, bottom=0.075, hspace=0.12)
    for i, c in enumerate(caps):
        img = Image.open(os.path.join(PANO_DIR, f"{c['pano_id']}.jpg")).convert("RGB")
        band = img.crop((0, int(b0 * 2048), 4096, int(b1 * 2048))).resize((2560, int(1280 * (b1 - b0))), Image.LANCZOS)
        heat = np.asarray(Image.open(_heat_path(c["pano_id"])), dtype=np.float32) / 255.0
        # Rotate every capture so that north is at the centre column: the rigs were driven in
        # both directions, and the corner should not jump left-right between rows.
        shift = int(round(((c["heading"]) % 360.0) / 360.0 * band.width))
        band = Image.fromarray(np.roll(np.asarray(band), shift, axis=1))
        heat_r = np.roll(heat, int(round((c["heading"] % 360.0) / 360.0 * heat.shape[1])), axis=1)
        over = _heat_overlay(band, heat_r[int(b0 * 512):int(b1 * 512)])
        ax = fig.add_subplot(gs[i])
        ax.imshow(over, extent=(0, 1, 1, 0), aspect="auto", interpolation="bilinear")
        peaks = peak_local_max(heat_r, min_distance=10, threshold_abs=DEPLOYED_T, exclude_border=False)
        for r, col in peaks:
            y = ((r + 0.5) / 512 - b0) / (b1 - b0)
            if 0 <= y <= 1:
                ax.scatter([(col + 0.5) / 1024], [y], s=150, facecolor="none", edgecolor="#5af0ff", lw=2.4)
        installed = sorted(r["installed"] for r in ramps if r["installed"] <= c["date"])
        new = [r for r in ramps if (caps[i - 1]["date"] if i else "0000") < r["installed"] <= c["date"]]
        label = f"{c['date']}   ·   {len(peaks)} detected"
        if new:
            label += f"   ·   {len(new)} ramps built {sorted({r['installed'] for r in new})[0]} per the city"
        ax.text(0.008, 0.08, label, transform=ax.transAxes, fontsize=13, color=SURFACE,
                fontweight="bold", va="top",
                bbox=dict(facecolor=INK, alpha=0.6, pad=4, edgecolor="none"))
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")
    n_ramps = len(ramps)
    _titles(fig, "The same corner over the years: the city builds ramps, the model sees them appear",
            f"Bend, OR. {n_ramps} inventory ramps within {RADIUS_M:.0f} m, install dates from the "
            f"city; Street View captures {caps[0]['date']} to {caps[-1]['date']}.")
    _footnote(fig, f"Panorama {args.pano} and its GSV history (benchmark/bend/records.jsonl); "
              "inventory stage_one/dataset_generation/location_data/bend.geojson (InstallDate). "
              "Each capture rotated so north is the centre column; detections are peaks ≥ 0.55 of "
              "the released model, no TTA. Manifest and heatmaps in docs/talks/uchicago_2026/timelapse/.",
              width=185)
    _save(fig, os.path.join("timelapse", f"timelapse_{args.pano}.png"))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidates")
    p.add_argument("--top", type=int, default=15)
    for name in ("fetch", "infer", "render"):
        p = sub.add_parser(name)
        p.add_argument("--pano", required=True)
        if name == "infer":
            p.add_argument("--cpu", action="store_true")
        if name == "render":
            p.add_argument("--dates", nargs="*", help="subset of capture dates (YYYY-MM) to show")
    args = ap.parse_args(argv)
    {"candidates": cmd_candidates, "fetch": cmd_fetch, "infer": cmd_infer, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
