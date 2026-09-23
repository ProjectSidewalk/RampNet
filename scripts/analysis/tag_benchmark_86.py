"""The curb-ramp tag benchmark of record (#86, RampNet 2.0 plan item 2).

Reproduces the ASSETS'24 curb-ramp tag baseline (released DINOv2-B checkpoint
``validated-dino-cls-b-curbramp-tags-best.pth``, HF ``projectsidewalk/sidewalk-tagger-ai-models``)
on the committed ``test.csv`` of HF ``projectsidewalk/sidewalk-tagger-ai-validated``
(``Validated/CurbRamp.zip``), then reports the same checkpoint on the part of that test set
whose panorama never appears in train, because the published split is by label, not by
panorama (980 of 6,295 panoramas are in both splits; PR #175 / ``docs/ps_supervision_audit.md``).

Subcommands, in run order (``docs/tag_benchmark_86.md`` has the exact commands):

``labels``   CPU. Join the HF train/test CSVs (read from the zip over HTTP range requests, or
             from a local copy) to each label's ``pano_id`` and lat/lng from Project Sidewalk's
             public ``rawLabels`` API. Writes the committed ``hf_curbramp_labels.csv``.
``prepare``  CPU. Extract the crops from the zip and apply the tagger's own ``crop.py``
             preprocessing (a 640 px box around the label point), executed from the pinned
             tagger checkout, not re-typed.
``infer``    GPU. Run a checkpoint over prepared crops with the tagger's preprocessing and
             model class; writes per-label sigmoid scores.
``score``    CPU. The tagger's metrics (mAP, micro/macro F1 at 0.3, per-tag AP; tags with
             fewer than 10 positives dropped) on the full test set and on the pano-disjoint
             ("leak-free") and pano-shared ("leaked") subsets, with pano-clustered bootstrap CIs.
``resplit``  CPU. A seeded, pano-grouped, per-city re-split of all 10,857 labels.
``train``    GPU. The tagger's DINOv2 training recipe (``notebooks/dino-trainer.ipynb``),
             on a split CSV.

Every tagger-code dependency is imported from a checkout of ``sidewalk-tagger-ai`` passed as
``--tagger-repo``; its HEAD sha is checked against ``--tagger-sha`` so a run on a different
commit fails loudly instead of quietly scoring something else.
"""
import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "analysis_out", "tag_benchmark_86")

HF_ZIP_URL = ("https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/"
              "resolve/main/Validated/CurbRamp.zip")
#: sidewalk-tagger-ai commit every number in docs/tag_benchmark_86.md was produced with.
TAGGER_SHA = "3b7405cd3206ece631cb7a65e22b1ab219df4b75"
UA = {"User-Agent": "Mozilla/5.0 (RampNet research; #86 tag benchmark)"}

#: HF filename city token (underscore -> hyphen) -> Project Sidewalk deployment id.
#: Same mapping PR #175's audit derived (analysis_out/ps_audit/hf_validated_join.json).
HF_CITY_TO_DEPLOYMENT = {
    "amsterdam": "amsterdam", "cdmx": "cdmx", "chicago": "chicago-il",
    "columbus": "columbus-oh", "newberg": "newberg-or", "oradell": "oradell-nj",
    "pittsburgh": "pittsburgh-pa", "seattle": "seattle-wa", "spgg": "spgg",
    "walla-walla": "walla-walla",
}
CITIES_API = "https://sidewalk-sea.cs.washington.edu/v3/api/cities?filetype=csv"

#: Tagger evaluation defaults (notebooks/evaluate.py at TAGGER_SHA).
MIN_INSTANCES = 10
MIN_THRESHOLD = 0.3
IMAGE_DIMENSION = 256
PATCH_MULTIPLE = 14


# ----------------------------------------------------------------------------- helpers

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(df, path, float_digits=6):
    """LF-pinned CSV with rounded floats, so a re-run on another box is byte-identical."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(float_digits)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def _round(o, nd=6):
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, nd)
    if isinstance(o, (np.floating,)):
        return _round(float(o), nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(_round(obj), fh, indent=1, sort_keys=True)
        fh.write("\n")


def parse_filename(fn):
    """``gsv-walla_walla-124-CurbRamp.png`` -> (``walla-walla``, 124)."""
    stem = os.path.basename(str(fn))
    assert stem.startswith("gsv-") and stem.endswith("-CurbRamp.png"), stem
    body = stem[len("gsv-"):-len("-CurbRamp.png")]
    city, label_id = body.rsplit("-", 1)
    return city.replace("_", "-"), int(label_id)


def tag_columns(df):
    """The tag columns of a tagger CSV: everything after ``validated_by`` or ``normalized_y``,
    the same rule as ``get_labels_ref_for_run`` in notebooks/evaluate.py."""
    cols = list(df.columns)
    anchor = "validated_by" if "validated_by" in cols else "normalized_y"
    return cols[cols.index(anchor) + 1:]


def check_tagger(repo, want_sha):
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    if want_sha and sha != want_sha:
        raise SystemExit(f"tagger checkout {repo} is at {sha}, expected {want_sha} "
                         f"(pass --tagger-sha '' to override, and say so in the doc)")
    return sha


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- zip access

class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (same approach as PR #175's
    ``ps_supervision_audit.py hf-index``): zipfile reads the central directory and the two
    CSVs without fetching the 30 GB archive."""

    def __init__(self, url):
        import requests
        self._rq = requests
        self.url = requests.head(url, headers=UA, allow_redirects=True, timeout=60).url
        self.size = int(requests.head(self.url, headers=UA, timeout=60).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n <= 0 or self.pos >= self.size:
            return 0
        r = self._rq.get(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"},
                         timeout=120)
        r.raise_for_status()
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def open_zip(src):
    import zipfile
    if src.startswith("http"):
        return zipfile.ZipFile(io.BufferedReader(_RangeFile(src), buffer_size=1 << 20))
    return zipfile.ZipFile(src)


def read_split_csvs(z):
    """{'train': df, 'test': df} from the zip's ``csv/`` members."""
    out = {}
    for n in z.namelist():
        base = os.path.basename(n)
        if n.lower().endswith(".csv") and base in ("train.csv", "test.csv"):
            out[base[:-4]] = pd.read_csv(io.BytesIO(z.read(n)))
    assert set(out) == {"train", "test"}, f"expected train.csv and test.csv, found {sorted(out)}"
    return out


# ----------------------------------------------------------------------------- labels

def fetch_raw_labels(city, dest):
    """rawLabels (CurbRamp) for one deployment from the public API."""
    import requests
    cities = pd.read_csv(io.StringIO(requests.get(CITIES_API, headers=UA, timeout=60).text))
    row = cities[cities.city_id == city]
    base = (row.url.iloc[0] if len(row) and isinstance(row.url.iloc[0], str)
            else f"https://sidewalk-{city}.cs.washington.edu").rstrip("/")
    r = requests.get(base + "/v3/api/rawLabels?filetype=csv&labelType=CurbRamp", headers=UA, timeout=900)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)


def build_label_table(splits, raw):
    """One row per HF crop: split, filename, city, label_id, pano_id, lat, lng, tag columns.

    ``raw`` maps deployment id -> rawLabels DataFrame (label_id, pano_id, latitude, longitude).
    A label with no live row (deleted since the 2024 freeze) keeps an empty pano_id and is
    reported, never dropped silently."""
    frames = []
    tags = None
    for split, df in splits.items():
        t = tag_columns(df)
        assert tags is None or t == tags, f"tag columns differ between splits: {t} vs {tags}"
        tags = t
        d = df[["filename", "normalized_x", "normalized_y"] + t].copy()
        d.insert(0, "split", split)
        parsed = d.filename.map(parse_filename)
        d.insert(2, "hf_city", [c for c, _ in parsed])
        d.insert(3, "label_id", [l for _, l in parsed])
        frames.append(d)
    lab = pd.concat(frames, ignore_index=True)
    unknown = set(lab.hf_city) - set(HF_CITY_TO_DEPLOYMENT)
    assert not unknown, f"unmapped HF cities: {unknown}"
    lab.insert(3, "city", lab.hf_city.map(HF_CITY_TO_DEPLOYMENT))
    lab = lab.drop(columns="hf_city")
    lab["label_uid"] = lab.city + ":" + lab.label_id.astype(str)
    assert lab.label_uid.is_unique, "duplicate (city, label_id) in the HF CSVs"
    live = []
    for city, r in raw.items():
        r = r[["label_id", "pano_id", "latitude", "longitude"]].copy()
        r.insert(0, "city", city)
        live.append(r)
    live = pd.concat(live, ignore_index=True).drop_duplicates(["city", "label_id"])
    lab = lab.merge(live, on=["city", "label_id"], how="left", validate="one_to_one")
    lab = lab.rename(columns={"latitude": "lat", "longitude": "lng"})
    front = ["split", "filename", "city", "label_id", "label_uid", "pano_id", "lat", "lng",
             "normalized_x", "normalized_y"]
    return lab[front + tags], tags


def cmd_labels(args):
    z = open_zip(args.zip)
    splits = read_split_csvs(z)
    raw = {}
    os.makedirs(args.raw_dir, exist_ok=True)
    for city in sorted(set(HF_CITY_TO_DEPLOYMENT.values())):
        p = os.path.join(args.raw_dir, f"{city}__rawLabels__CurbRamp.csv")
        if not os.path.exists(p):
            if not args.fetch:
                raise SystemExit(f"missing {p}; pass --fetch to pull it from the public API")
            print(f"fetching {city} rawLabels ...")
            fetch_raw_labels(city, p)
        raw[city] = pd.read_csv(p, usecols=["label_id", "pano_id", "latitude", "longitude"])
    lab, tags = build_label_table(splits, raw)
    write_csv(lab, args.out, float_digits=7)
    miss = lab[lab.pano_id.isna()]
    print(f"wrote {args.out}: {len(lab)} rows ({lab.split.value_counts().to_dict()}), "
          f"{len(tags)} tags, {len(miss)} without a live pano_id: {miss.label_uid.tolist()}")


# ----------------------------------------------------------------------------- leak

def leak_table(lab, test_split="test", train_split="train"):
    """Per test label: does its pano appear in train, and how far is the nearest train label
    of the same city. Labels with no pano_id are flagged ``pano_unknown`` and kept out of both
    subsets."""
    tr = lab[lab.split == train_split]
    te = lab[lab.split == test_split].copy()
    train_panos = set(tr.pano_id.dropna())
    te["pano_unknown"] = te.pano_id.isna()
    te["pano_in_train"] = te.pano_id.isin(train_panos) & ~te.pano_unknown
    near = np.full(len(te), np.nan)
    for city, g in te.groupby("city"):
        t = tr[(tr.city == city) & tr.lat.notna()]
        if not len(t):
            continue
        idx = np.where(te.city.values == city)[0]
        for i in idx:
            la, ln = te.lat.values[i], te.lng.values[i]
            if np.isnan(la):
                continue
            near[i] = float(np.min(haversine_m(la, ln, t.lat.values, t.lng.values)))
    te["nearest_train_m"] = near
    return te


# ----------------------------------------------------------------------------- metrics

def tagger_metrics(y_true, y_score, tags, min_instances=MIN_INSTANCES, threshold=MIN_THRESHOLD,
                   selected=None):
    """The tagger's reported metrics (notebooks/evaluate.py at TAGGER_SHA), from arrays.

    - per tag: AP = ``average_precision_score``; best-F1 threshold on the PR curve, floored
      at ``threshold``; F1 at ``threshold``.
    - ``mAP`` = mean AP over tags with >= ``min_instances`` positives.
    - micro / macro / weighted F1 over the same tags, predictions ``score >= threshold``.

    ``selected`` overrides which tags are averaged (e.g. the tags selected on the full test
    set, so a subset's mAP averages the same tags). Returns a dict."""
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, auc, \
        precision_score, recall_score
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = y_true.sum(axis=0)
    if selected is None:
        sel = n_pos >= min_instances
    else:
        sel = np.array([t in set(selected) for t in tags])
    per_tag = {}
    for i, t in enumerate(tags):
        yt, ys = y_true[:, i], y_score[:, i]
        row = {"n_pos": int(n_pos[i]), "n": int(len(yt)), "selected": bool(sel[i])}
        if 0 < n_pos[i] < len(yt):
            p, r, th = precision_recall_curve(yt, ys, pos_label=1)
            row["ap"] = float(average_precision_score(yt, ys, average="weighted"))
            row["pr_auc"] = float(auc(r, p))
            den = p + r
            f1s = np.where(den != 0, 2 * p * r / np.where(den == 0, 1, den), 0)
            bt = float(max(th[int(np.argmax(f1s))], threshold))
            yb = (ys >= bt).astype(int)
            row.update(best_threshold=bt,
                       precision_at_best=float(precision_score(yt, yb, zero_division=0)),
                       recall_at_best=float(recall_score(yt, yb, zero_division=0)),
                       f1_at_best=float(f1_score(yt, yb, zero_division=0)))
            row["f1_at_threshold"] = float(f1_score(yt, (ys >= threshold).astype(int), zero_division=0))
        else:
            row["ap"] = None
        per_tag[t] = row
    usable = [t for t, i in zip(tags, range(len(tags))) if sel[i] and per_tag[t]["ap"] is not None]
    cols = np.array([t in usable for t in tags])
    yt, yb = y_true[:, cols], (y_score[:, cols] >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "tags_averaged": usable,
        "mAP": float(np.mean([per_tag[t]["ap"] for t in usable])) if usable else None,
        "micro_f1": float(f1_score(yt, yb, average="micro", zero_division=0)) if usable else None,
        "macro_f1": float(f1_score(yt, yb, average="macro", zero_division=0)) if usable else None,
        "weighted_f1": float(f1_score(yt, yb, average="weighted", zero_division=0)) if usable else None,
        "per_tag": per_tag,
        "min_instances": min_instances,
        "threshold": threshold,
    }
    return out


def bootstrap_ci(y_true, y_score, groups, tags, selected, n_boot=1000, seed=86, alpha=0.05):
    """Pano-clustered bootstrap CI for mAP and micro/macro F1 on a fixed tag set.

    Labels on one panorama are not independent (same image, same rater session), so the
    resampling unit is the panorama."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    stats = {"mAP": [], "micro_f1": [], "macro_f1": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        m = tagger_metrics(y_true[idx], y_score[idx], tags, selected=selected)
        for k in stats:
            if m[k] is not None:
                stats[k].append(m[k])
    return {k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
            for k, v in stats.items() if v}


# ----------------------------------------------------------------------------- prepare

def load_tagger_function(tagger_repo, relpath, name):
    """Execute ONE top-level function definition from a tagger file (its imports included),
    without running the file's module-level code. ``crop.py`` crops all four label types at
    import time, so importing it is not an option; this runs its own ``crop_image`` source."""
    path = os.path.join(tagger_repo, relpath)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name == name)]
    assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in keep), f"{name} not in {path}"
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)  # noqa: S102
    return ns[name]


def cmd_prepare(args):
    """Extract crops for the requested splits and run the tagger's crop_image over them.

    Layout: ``<out>/<split>/<filename>`` plus ``<out>/<split>/<split>.csv`` (the zip's CSV
    verbatim), i.e. the directory the tagger's evaluate.py / trainer expects."""
    check_tagger(args.tagger_repo, args.tagger_sha)
    crop_image = load_tagger_function(args.tagger_repo, "crop.py", "crop_image")
    z = open_zip(args.zip)
    names = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".png")}
    splits = read_split_csvs(z)
    for split in args.splits:
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        df = splits[split]
        csv_path = os.path.join(d, f"{split}.csv")
        with open(csv_path, "wb") as fh:
            fh.write(z.read([n for n in z.namelist() if n.endswith(f"csv/{split}.csv")][0]))
        missing = 0
        for fn in df.filename:
            if fn not in names:
                missing += 1
                continue
            dest = os.path.join(d, fn)
            if not os.path.exists(dest):
                with open(dest, "wb") as fh:
                    fh.write(z.read(names[fn]))
        print(f"{split}: extracted {len(df) - missing} crops, {missing} listed but absent from the zip")
        marker = os.path.join(d, ".cropped")
        if os.path.exists(marker):
            print(f"{split}: already cropped ({marker}); not cropping twice")
            continue
        crop_image(d, csv_path)  # the tagger's own function, in place, as its pipeline does
        with open(marker, "w") as fh:
            fh.write(dt.datetime.now(dt.timezone.utc).isoformat())


# ----------------------------------------------------------------------------- model

def build_model(tagger_repo, nc, backbone=None):
    """The tagger's DinoVisionTransformerClassifier (base, 4 registers, img_size 526), built
    from its dinov2 package. ``backbone`` loads the pretrain weights (training); inference
    loads a full fine-tuned state dict over it, so it is optional there."""
    import torch
    from torch import nn
    sys.path.insert(0, tagger_repo)
    from dinov2.models.vision_transformer import vit_base  # noqa: E402

    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            model = vit_base(patch_size=14, img_size=526, init_values=1.0,
                             num_register_tokens=4, block_chunks=0)
            if backbone:
                model.load_state_dict(torch.load(backbone, map_location="cpu"))
            self.transformer = model
            self.embedding_size = 768
            self.classifier = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, nc))

        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            return self.classifier(x)

    return DinoVisionTransformerClassifier()


def eval_transform():
    """evaluate.py: PIL resize to 256x256, then ResizeAndPad(256, 14), ToTensor, Normalize."""
    from torchvision import transforms

    def resize_and_pad(img):
        img = transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION))(img)
        pw = (PATCH_MULTIPLE - img.width % PATCH_MULTIPLE) % PATCH_MULTIPLE
        ph = (PATCH_MULTIPLE - img.height % PATCH_MULTIPLE) % PATCH_MULTIPLE
        return transforms.Pad((pw // 2, ph // 2, pw - pw // 2, ph - ph // 2))(img)

    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def f(img):
        img = img.convert("RGB").resize((IMAGE_DIMENSION, IMAGE_DIMENSION))
        return norm(resize_and_pad(img))
    return f


def cmd_infer(args):
    import torch
    from PIL import Image
    sha = check_tagger(args.tagger_repo, args.tagger_sha)
    df = pd.read_csv(args.csv)
    tags = tag_columns(df)
    t0 = time.time()
    model = build_model(args.tagger_repo, len(tags))
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    load_s = time.time() - t0
    tf = eval_transform()
    where = {}
    for d in args.images:
        for f in df.filename:
            if f not in where and os.path.exists(os.path.join(d, f)):
                where[f] = os.path.join(d, f)
    present = [f for f in df.filename if f in where]
    print(f"{len(present)}/{len(df)} crops present; {len(tags)} tags; device {dev}")
    scores = []
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(present), args.batch):
            batch = torch.stack([tf(Image.open(where[f])) for f in present[i:i + args.batch]])
            scores.append(torch.sigmoid(model(batch.to(dev))).float().cpu().numpy())
    infer_s = time.time() - t1
    s = np.concatenate(scores) if scores else np.zeros((0, len(tags)))
    out = pd.DataFrame({"filename": present})
    for j, t in enumerate(tags):
        out[f"score:{t}"] = s[:, j]
    write_csv(out, args.out, float_digits=6)
    meta = {"checkpoint": os.path.basename(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
            "tagger_sha": sha, "csv": os.path.basename(args.csv), "n_scored": len(present),
            "n_listed": int(len(df)), "tags": tags, "load_s": load_s, "infer_s": infer_s,
            "elapsed_s": time.time() - t0, "host": socket.getfqdn(),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "predictions_sha256": sha256_file(args.out)}
    write_json(meta, args.out + ".meta.json")
    print(json.dumps(_round(meta, 3), indent=1))


# ----------------------------------------------------------------------------- tagger-eval

_TAGGER_EVAL_BOOT = """
import os, sys, runpy
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep the title as text so it can be read back
os.chdir(sys.argv[1])
sys.argv = ["evaluate.py"] + sys.argv[2:]
runpy.run_path("evaluate.py", run_name="__main__")
"""


def parse_tagger_title(svg_text):
    """The headline numbers evaluate.py prints only into its figure title (2 dp)."""
    import re
    m = re.search(r"mAP: ([\d.]+) \| Micro F1: ([\d.]+) \| Macro F1: ([\d.]+) \| "
                  r"Weighted F1: ([\d.]+) \| Manual avg\.: ([\d.]+) \| Threshold: ([\d.]+)", svg_text)
    if not m:
        return None
    k = ("mAP", "micro_f1", "macro_f1", "weighted_f1", "manual_avg_f1", "threshold")
    return dict(zip(k, map(float, m.groups())))


def cmd_tagger_eval(args):
    """Run the tagger's notebooks/evaluate.py UNMODIFIED on the prepared test crops and read
    back what it reports: per-tag AP from its stats JSON, headline numbers from its figure
    title. This is the reproduction of record; ``infer`` + ``score`` must agree with it."""
    sha = check_tagger(args.tagger_repo, args.tagger_sha)
    nb = os.path.join(args.tagger_repo, "notebooks")
    need = [os.path.join(args.tagger_repo, "dinov2_vitb14_reg4_pretrain.pth"),
            os.path.join(nb, "models", "validated-dino-cls-b-curbramp-tags-best.pth"),
            os.path.join(args.tagger_repo, "datasets", "crops-curbramp-tags", "test", "test.csv")]
    for p in need:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (see docs/tag_benchmark_86.md for the layout)")
    t0 = time.time()
    with open(args.log, "w", encoding="utf-8") as log:
        rc = subprocess.run([sys.executable, "-c", _TAGGER_EVAL_BOOT, nb, "--label-type", "curbramp",
                             "--model", "DINO", "--dataset-type", "validated"],
                            stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise SystemExit(f"evaluate.py exited {rc}; see {args.log}")
    res = os.path.join(args.tagger_repo, "results", "curbramp")
    with open(os.path.join(res, "validated-dino-inference-stats.json"), encoding="utf-8") as fh:
        stats = json.load(fh)["category_to_prediction_stats"]
    with open(os.path.join(res, "validated-dino-pr-curve.svg"), encoding="utf-8") as fh:
        title = parse_tagger_title(fh.read())
    per_tag = {t: {"n_pos": v["n_instances"], "ap": v["average_precision_val"], "pr_auc": v["pr_auc"]}
               for t, v in stats.items()}
    sel = [t for t, v in per_tag.items() if v["n_pos"] >= MIN_INSTANCES]
    out = {"tagger_sha": sha, "title_2dp": title, "per_tag": per_tag, "tags_averaged": sorted(sel),
           "mAP_from_stats_json": float(np.mean([per_tag[t]["ap"] for t in sel])),
           "elapsed_s": elapsed, "host": socket.getfqdn(),
           "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json(out, args.out)
    print(json.dumps(_round(out, 4), indent=1))


# ----------------------------------------------------------------------------- score

def _arrays(pred, lab, tags):
    m = lab.merge(pred, on="filename", how="inner", validate="one_to_one")
    y = m[tags].to_numpy(float)
    s = m[[f"score:{t}" for t in tags]].to_numpy(float)
    return m, y, s


def score_subsets(pred, lab, tags, test_split="test", train_split="train", near_m=10.0,
                  n_boot=1000, fixed_tags=None):
    """Full / leak-free / leaked metrics for one prediction file. Returns (summary, per_label)."""
    te = leak_table(lab, test_split, train_split)
    m, y, s = _arrays(pred, te, tags)
    full = tagger_metrics(y, s, tags)
    fixed = list(fixed_tags) if fixed_tags else full["tags_averaged"]
    groups = m.pano_id.fillna(m.label_uid).to_numpy()
    subsets = {
        "full": np.ones(len(m), bool),
        "leak_free": (~m.pano_in_train & ~m.pano_unknown).to_numpy(),
        "leaked": m.pano_in_train.to_numpy(),
        f"leak_free_no_train_within_{int(near_m)}m": (~m.pano_in_train & ~m.pano_unknown
                                                      & (m.nearest_train_m > near_m)).to_numpy(),
    }
    out = {"n_scored": int(len(m)), "tags_fixed": fixed, "subsets": {}}
    for name, mask in subsets.items():
        if mask.sum() == 0:
            continue
        own = tagger_metrics(y[mask], s[mask], tags)
        same = tagger_metrics(y[mask], s[mask], tags, selected=fixed)
        out["subsets"][name] = {
            "n": int(mask.sum()), "n_panos": int(len(set(groups[mask]))),
            "cities": m.city[mask].value_counts().sort_index().to_dict(),
            "tagger_rule": {k: own[k] for k in ("mAP", "micro_f1", "macro_f1", "weighted_f1", "tags_averaged")},
            "fixed_tags": {k: same[k] for k in ("mAP", "micro_f1", "macro_f1", "weighted_f1")},
            "ci95_fixed_tags": bootstrap_ci(y[mask], s[mask], groups[mask], tags, fixed, n_boot=n_boot) if n_boot else None,
            "per_tag": same["per_tag"],
        }
    per_label = m[["filename", "city", "label_id", "pano_id", "pano_in_train", "nearest_train_m"] + tags
                  + [f"score:{t}" for t in tags]]
    return out, per_label


def cmd_score(args):
    lab = pd.read_csv(args.labels)
    if args.split_csv:
        sp = pd.read_csv(args.split_csv)[["label_uid", "split"]]
        lab = lab.drop(columns="split").merge(sp, on="label_uid", how="inner", validate="one_to_one")
    tags = [c for c in lab.columns if c not in ("split", "filename", "city", "label_id", "label_uid",
                                                 "pano_id", "lat", "lng", "normalized_x", "normalized_y")]
    pred = pd.read_csv(args.pred)
    fixed = args.fixed_tags.split(",") if args.fixed_tags else None
    summary, per_label = score_subsets(pred, lab, tags, near_m=args.near_m, n_boot=args.n_boot,
                                       fixed_tags=fixed)
    summary["pred_file"] = os.path.basename(args.pred)
    summary["pred_sha256"] = sha256_file(args.pred)
    summary["labels_sha256"] = sha256_file(args.labels)
    if args.split_csv:
        summary["split_csv_sha256"] = sha256_file(args.split_csv)
    write_json(summary, args.out)
    if args.per_label_out:
        write_csv(per_label, args.per_label_out)
    for k, v in summary["subsets"].items():
        f = v["fixed_tags"]
        print(f"{k:32s} n={v['n']:5d} panos={v['n_panos']:5d} mAP={f['mAP']:.4f} "
              f"microF1={f['micro_f1']:.4f} macroF1={f['macro_f1']:.4f}  CI(mAP)={v['ci95_fixed_tags'] and v['ci95_fixed_tags']['mAP']}")


# ----------------------------------------------------------------------------- resplit

def cell_groups(lab, cell_m=100.0):
    """A group key per label from a ~``cell_m`` square lat/lng grid (a city block), so a ramp
    seen from two neighbouring panoramas lands on one side. A panorama is placed by the mean
    location of its labels, so every label of one panorama shares a cell and the split stays
    pano-disjoint. Labels with no location or pano are their own group."""
    lab = lab.copy()
    lab["_g"] = lab.pano_id.fillna("nopano:" + lab.label_uid)
    loc = lab.groupby("_g")[["lat", "lng"]].mean()
    dy = cell_m / 111_320.0
    dx = cell_m / (111_320.0 * np.cos(np.radians(loc.lat.fillna(0).to_numpy())))
    iy, ix = np.floor(loc.lat.to_numpy() / dy), np.floor(loc.lng.to_numpy() / dx)
    city = lab.groupby("_g").city.first().reindex(loc.index)
    key = {g: (f"cell:{c}:{int(a)}:{int(b)}" if not (np.isnan(a) or np.isnan(b)) else g)
           for g, c, a, b in zip(loc.index, city, iy, ix)}
    return lab["_g"].map(key)


def pano_grouped_split(lab, seed=86, target=None, group="pano", cell_m=100.0):
    """Seeded pano-grouped re-split, per city: panos are shuffled and moved to test until the
    city's test count reaches its count in the original split (so city mix and test size
    match the published split). No pano ends up on both sides. Labels without a pano_id are
    each their own group."""
    rng = np.random.default_rng(seed)
    lab = lab.copy()
    if group == "pano":
        lab["group"] = lab.pano_id.fillna("nopano:" + lab.label_uid)
    elif group == "cell":
        lab["group"] = cell_groups(lab, cell_m)
    else:
        raise ValueError(group)
    if target is None:
        target = lab[lab.split == "test"].city.value_counts().to_dict()
    test_groups = set()
    for city in sorted(lab.city.unique()):
        g = lab[lab.city == city].groupby("group").size()
        order = sorted(g.index)
        rng.shuffle(order)
        n = 0
        for grp in order:
            if n >= target.get(city, 0):
                break
            test_groups.add(grp)
            n += int(g[grp])
    lab["resplit"] = np.where(lab.group.isin(test_groups), "test", "train")
    return lab[["label_uid", "filename", "city", "label_id", "pano_id", "resplit"]].rename(
        columns={"resplit": "split"})


def cmd_resplit(args):
    lab = pd.read_csv(args.labels)
    sp = pano_grouped_split(lab, seed=args.seed, group=args.group, cell_m=args.cell_m)
    both = set(sp[sp.split == "train"].pano_id.dropna()) & set(sp[sp.split == "test"].pano_id.dropna())
    assert not both, f"{len(both)} panos on both sides"
    write_csv(sp, args.out)
    print(f"wrote {args.out} sha256={sha256_file(args.out)}: {sp.split.value_counts().to_dict()}")


# ----------------------------------------------------------------------------- train

def cmd_train(args):
    """The tagger's DINOv2 recipe (notebooks/dino-trainer.ipynb at TAGGER_SHA): full
    fine-tune, Adam lr 1e-6, batch 4, shuffle, BCEWithLogitsLoss, 100 epochs, no
    augmentation, checkpoint kept by best *training* exact-match accuracy (ties -> lower
    loss). Two deliberate differences, both stated in the doc: a fixed seed, and the
    deterministic preprocessing is computed once and cached in memory (identical tensors,
    much faster epochs)."""
    import torch
    from torch import nn, optim
    from torchvision import io as tvio, transforms
    from sklearn.metrics import accuracy_score
    sha = check_tagger(args.tagger_repo, args.tagger_sha)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lab = pd.read_csv(args.labels)
    if args.split_csv:
        sp = pd.read_csv(args.split_csv)[["label_uid", "split"]]
        lab = lab.drop(columns="split").merge(sp, on="label_uid", how="inner", validate="one_to_one")
    tags = [c for c in lab.columns if c not in ("split", "filename", "city", "label_id", "label_uid",
                                                 "pano_id", "lat", "lng", "normalized_x", "normalized_y")]
    tr = lab[lab.split == "train"].reset_index(drop=True)
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION))])
    imgs = []
    t0 = time.time()
    def locate(fn):
        for d in args.images:
            if os.path.exists(os.path.join(d, fn)):
                return os.path.join(d, fn)
        raise FileNotFoundError(fn)

    for fn in tr.filename:
        img = tf(tvio.read_image(locate(fn), mode=tvio.ImageReadMode.RGB))
        pw = (PATCH_MULTIPLE - img.width % PATCH_MULTIPLE) % PATCH_MULTIPLE
        ph = (PATCH_MULTIPLE - img.height % PATCH_MULTIPLE) % PATCH_MULTIPLE
        img = transforms.Pad((pw // 2, ph // 2, pw - pw // 2, ph - ph // 2))(img)
        imgs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
    X = torch.stack(imgs)  # uint8 N,3,266,266 -- ToTensor is /255, pad pixels stay 0
    Y = torch.tensor(tr[tags].to_numpy(np.float32))
    prep_s = time.time() - t0
    dev = torch.device("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model = build_model(args.tagger_repo, len(tags), backbone=args.backbone).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    os.makedirs(args.out_dir, exist_ok=True)
    log_rows, best_acc, best_loss = [], 0.0, 100.0
    g = torch.Generator().manual_seed(args.seed)
    t_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        te = time.time()
        perm = torch.randperm(len(X), generator=g)
        losses, accs = [], []
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = (X[idx].to(dev).float() / 255.0 - mean) / std
            yb = Y[idx].to(dev)
            opt.zero_grad()
            out = model(xb).squeeze(dim=1)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pred = (torch.sigmoid(out) > 0.5).float()
            accs.append(accuracy_score(yb.cpu().numpy(), pred.detach().cpu().numpy()))
        el, ea = float(np.mean(losses)), float(np.mean(accs))
        saved = ""
        if ea > best_acc or (ea == best_acc and el < best_loss):
            best_acc, best_loss = ea, el
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": el},
                       os.path.join(args.out_dir, "best.pth"))
            saved = "best"
        row = {"epoch": epoch, "loss": el, "train_exact_match_acc": ea, "epoch_s": time.time() - te, "saved": saved}
        log_rows.append(row)
        print(json.dumps(_round(row, 5)), flush=True)
        write_csv(pd.DataFrame(log_rows), os.path.join(args.out_dir, "train_log.csv"))
    torch.save({"epoch": args.epochs - 1, "model_state_dict": model.state_dict()},
               os.path.join(args.out_dir, "last.pth"))
    meta = {"tagger_sha": sha, "n_train": int(len(tr)), "tags": tags, "epochs": args.epochs, "lr": args.lr,
            "batch": args.batch, "seed": args.seed, "prep_s": prep_s, "train_s": time.time() - t_train,
            "elapsed_s": time.time() - t0, "best_epoch": int(max(
                (r for r in log_rows if r["saved"]), key=lambda r: r["epoch"])["epoch"]),
            "host": socket.getfqdn(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "split_csv": os.path.basename(args.split_csv) if args.split_csv else "hf test.csv/train.csv",
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json(meta, os.path.join(args.out_dir, "train_meta.json"))
    print(json.dumps(_round(meta, 3)))


# ----------------------------------------------------------------------------- usage

def usage_row(label, elapsed_s, status, what, n=None, host="makelab2.cs.washington.edu",
              gpu="NVIDIA A40", ts=None, extra=None):
    """One ``paid: false`` row for analysis_out/usage_log.jsonl (docs/compute_cost.md: GPU
    time on a host without Slurm goes there). Failed runs get a row too."""
    row = {"ts": ts or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "bundle": "hf-tagger-validated-curbramp", "label": label, "provider": "tagger-86",
           "model_id": label, "paid": False, "panos_scored": n, "elapsed_s": round(float(elapsed_s), 3),
           "hardware": {"host": host, "gpus": [gpu] if gpu else []}, "status": status, "what": what,
           "est_cost_usd": 0.0, "pricing": None}
    if extra:
        row.update(extra)
    return row


def cmd_log_usage(args):
    sys.path.insert(0, REPO)
    from rampnet import ledger
    row = usage_row(args.label, args.elapsed_s, args.status, args.what, n=args.n, ts=args.ts)
    ledger.append_rows(args.log, [row])
    print(json.dumps(row))


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tagger(p):
        p.add_argument("--tagger-repo", required=True, help="checkout of ProjectSidewalk/sidewalk-tagger-ai")
        p.add_argument("--tagger-sha", default=TAGGER_SHA, help="required HEAD of --tagger-repo ('' = any)")

    p = sub.add_parser("labels", help="HF CSVs + rawLabels -> hf_curbramp_labels.csv")
    p.add_argument("--zip", default=HF_ZIP_URL, help="local CurbRamp.zip or its URL (range-read)")
    p.add_argument("--raw-dir", required=True, help="dir of <city>__rawLabels__CurbRamp.csv")
    p.add_argument("--fetch", action="store_true", help="pull missing rawLabels from the public API")
    p.add_argument("--out", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("prepare", help="extract crops and apply the tagger's crop.py")
    tagger(p)
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "test"])
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("infer", help="score crops with a checkpoint (GPU)")
    tagger(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True, help="a tagger-format CSV listing the crops and tag columns")
    p.add_argument("--images", required=True, nargs="+", help="prepared crop dir(s), searched in order")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("tagger-eval", help="run the tagger's evaluate.py unmodified (GPU)")
    tagger(p)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_tagger_eval)

    p = sub.add_parser("score", help="tagger metrics on full / leak-free / leaked subsets")
    p.add_argument("--pred", required=True)
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--split-csv", default=None, help="score against this split instead of the HF one")
    p.add_argument("--near-m", type=float, default=10.0)
    p.add_argument("--fixed-tags", default=None,
                   help="comma list of tags every subset averages (default: the tagger rule on the full set)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--per-label-out", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("resplit", help="seeded pano-grouped re-split")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--group", default="pano", choices=["pano", "cell"],
                   help="pano: no panorama on both sides; cell: no ~--cell-m grid cell on both sides")
    p.add_argument("--cell-m", type=float, default=100.0)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "resplit_pano_grouped_seed86.csv"))
    p.set_defaults(func=cmd_resplit)

    p = sub.add_parser("train", help="the tagger's DINOv2 recipe on a split (GPU)")
    tagger(p)
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--split-csv", default=None)
    p.add_argument("--images", required=True, nargs="+", help="prepared crop dir(s), searched in order")
    p.add_argument("--backbone", required=True, help="dinov2_vitb14_reg4_pretrain.pth")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("log-usage", help="append a paid:false GPU-time row to the usage ledger")
    p.add_argument("--label", required=True, help="e.g. tagger-eval, infer-released, train-control")
    p.add_argument("--elapsed-s", type=float, required=True)
    p.add_argument("--status", default="ok", choices=["ok", "failed", "killed"])
    p.add_argument("--what", required=True, help="one line: what ran")
    p.add_argument("--n", type=int, default=None, help="crops scored / trained on")
    p.add_argument("--ts", default=None, help="run start (UTC ISO); default now")
    p.add_argument("--log", default=os.path.join(REPO, "analysis_out", "usage_log.jsonl"))
    p.set_defaults(func=cmd_log_usage)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
