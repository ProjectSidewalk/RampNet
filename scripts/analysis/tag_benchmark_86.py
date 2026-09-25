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
``contrast`` CPU. One arm's predictions minus another's: unpaired (each on its own test set,
             independent pano-clustered draws) and paired (the labels test in both splits,
             same draws).
``resplit``  CPU. A seeded, pano-grouped, per-city re-split of all 10,857 labels.
``prep``     CPU. Optional: decode a split's training crops once into a uint8 ``.npy``
             memmap that ``train --prep`` maps instead of decoding in memory.
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

#: HF revisions every number in docs/tag_benchmark_86.md was produced from. The URLs resolve
#: at these commits, not at ``main``, so a later upload cannot change what a re-run reads.
HF_DATASET_REV = "6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50"   # sidewalk-tagger-ai-validated
HF_MODEL_REV = "65959dbc80b87e4f39385204c4b639cbcf58e1a8"     # sidewalk-tagger-ai-models
HF_ZIP_URL = ("https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/"
              f"resolve/{HF_DATASET_REV}/Validated/CurbRamp.zip")

#: The three retrain arms and the split each trains on (None = the published HF split).
ARM_SPLITS = {"control": None,
              "pano": "resplit_pano_grouped_seed86.csv",
              "cell": "resplit_cell100m_seed86.csv"}
#: The eight tags every committed number averages (the tagger's >= 10-positive rule on the
#: published test set); the re-split arms are scored on the same eight.
FIXED_TAGS = ["missing-tactile-warning", "narrow", "not-enough-landing-space", "not-level-with-street",
              "points-into-traffic", "pooled-water", "steep", "surface-problem"]
#: Epoch indices (0-based) after which the live run copied best.pth
#: (scripts/analysis/tag_benchmark_86_snap.sh).
SNAPSHOT_EPOCHS = (4, 9, 19, 49)
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


#: The one list of columns that are never tags, shared by ``train`` (``train_table``), ``infer``
#: (``tag_columns``), ``score`` / ``test-only`` (``load_labels``) and ``context_fov_86.py`` (which
#: calls ``load_labels``). Every other column of a label table is a tag, in file order.
#: ``affirmed`` (optional, 0/1) marks rows whose untagged cells are affirmed absences (the HF
#: ASSETS'24 rows); ``train --loss nnpu|soft`` treats them as ordinary BCE rows (see ``TagLoss``).
#: It is here so that no reader ever takes it for a tag. Example::
#:
#:     >>> [c for c in ["label_uid", "split", "steep", "affirmed"] if c not in NON_TAG_COLUMNS]
#:     ['steep']
NON_TAG_COLUMNS = ("split", "filename", "city", "label_id", "label_uid", "pano_id", "lat", "lng",
                   "normalized_x", "normalized_y", "affirmed")


def tag_columns(df):
    """The tag columns of a tagger CSV: everything after ``validated_by`` or ``normalized_y``,
    the same rule as ``get_labels_ref_for_run`` in notebooks/evaluate.py, less any
    ``NON_TAG_COLUMNS`` (so a PU training table's ``affirmed`` is not read as a tag)."""
    cols = list(df.columns)
    anchor = "validated_by" if "validated_by" in cols else "normalized_y"
    return [c for c in cols[cols.index(anchor) + 1:] if c not in NON_TAG_COLUMNS]


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
    live = pd.concat(live, ignore_index=True)
    # assert rather than drop_duplicates: a duplicated live row must stop the join, not be
    # silently resolved to whichever copy came first (0 duplicates on 2026-09-21)
    dup = live[live.duplicated(["city", "label_id"], keep=False)]
    assert dup.empty, f"duplicate (city, label_id) in rawLabels: {dup.head().to_dict('records')}"
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


def bootstrap_samples(y_true, y_score, groups, tags, selected, n_boot=1000, seed=86):
    """Pano-clustered bootstrap draws of mAP and micro/macro F1 on a fixed tag set.

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
            stats[k].append(np.nan if m[k] is None else m[k])
    return {k: np.array(v) for k, v in stats.items()}


def paired_subset_diff_samples(y_true, y_score, groups, sub_mask, tags, selected, n_boot=1000, seed=86):
    """Bootstrap draws of metric(whole set) - metric(subset), both computed on the SAME
    pano-clustered resample. ``sub_mask`` must be constant within a panorama (leak-free is a
    pano-level property), so the subset of a resample is the resample of the subset.

    This is the direct test of "does the leak inflate the published number": the published
    number is the whole set, its leak-free counterpart is the subset, and they share
    panoramas, so independent draws would overstate the noise."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    sub_mask = np.asarray(sub_mask, bool)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    stats = {"mAP": [], "micro_f1": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        sub = idx[sub_mask[idx]]
        a = tagger_metrics(y_true[idx], y_score[idx], tags, selected=selected)
        b = tagger_metrics(y_true[sub], y_score[sub], tags, selected=selected) if len(sub) else None
        for k in stats:
            ok = b is not None and a[k] is not None and b[k] is not None
            stats[k].append(a[k] - b[k] if ok else np.nan)
    return {k: np.array(v) for k, v in stats.items()}


def _ci(v, alpha=0.05):
    v = np.asarray(v)[~np.isnan(v)]
    return [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))] if len(v) else None


def bootstrap_ci(y_true, y_score, groups, tags, selected, n_boot=1000, seed=86, alpha=0.05):
    """95 % pano-clustered bootstrap CI for mAP and micro/macro F1 (see bootstrap_samples)."""
    d = bootstrap_samples(y_true, y_score, groups, tags, selected, n_boot, seed)
    return {k: _ci(v, alpha) for k, v in d.items()}


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


#: (width, height) of every crop in the HF zip at HF_DATASET_REV, before crop.py.
HF_CROP_SIZE = (1440, 960)


def uncropped_violations(d, filenames, size=HF_CROP_SIZE):
    """Filenames in ``d`` whose image is not ``size`` (reads headers only)."""
    from PIL import Image
    bad = []
    for fn in filenames:
        with Image.open(os.path.join(d, fn)) as im:
            if im.size != tuple(size):
                bad.append(fn)
    return bad


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
        # crop_image crops in place and the marker is written only after it returns, so an
        # interrupted run leaves a mix of cropped and uncropped files that a resumed run would
        # crop again, silently changing the framing. Refuse unless every crop is still the
        # HF original size.
        bad = uncropped_violations(d, [f for f in df.filename if f in names])
        if bad:
            raise SystemExit(f"{split}: {len(bad)} crops are not {HF_CROP_SIZE[0]}x{HF_CROP_SIZE[1]} "
                             f"(e.g. {bad[:3]}) but {marker} is absent: an earlier prepare was "
                             f"interrupted mid-crop. Delete {d} and run prepare again.")
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
            scores.append(model(batch.to(dev)).float().cpu().numpy())   # logits
    infer_s = time.time() - t1
    s = np.concatenate(scores) if scores else np.zeros((0, len(tags)))
    out = pd.DataFrame({"filename": present})
    # Logits, not sigmoid scores: rounded to a fixed number of decimals, sigmoid scores
    # collapse into ties near 0 and 1 (867 of 2,183 missing-tactile-warning scores rounded to
    # 0.000000 at 6 dp), and ties lower AP. score_probs() applies the sigmoid in float32, as
    # evaluate.py does.
    for j, t in enumerate(tags):
        out[f"logit:{t}"] = s[:, j]
    write_csv(out, args.out, float_digits=5)
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

def score_probs(pred, tags):
    """Sigmoid scores per tag from a predictions frame: ``logit:<tag>`` columns go through a
    float32 sigmoid (what evaluate.py's ``torch.sigmoid`` does on its float32 logits);
    ``score:<tag>`` columns are taken as they are."""
    if all(f"logit:{t}" in pred for t in tags):
        z = pred[[f"logit:{t}" for t in tags]].to_numpy(np.float32)
        return (1.0 / (1.0 + np.exp(-z, dtype=np.float32))).astype(np.float32).astype(float)
    return pred[[f"score:{t}" for t in tags]].to_numpy(float)


def _arrays(pred, lab, tags):
    m = lab.merge(pred, on="filename", how="inner", validate="one_to_one")
    y = m[tags].to_numpy(float)
    s = score_probs(m, tags)
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
    draws = {}
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
            "ci95_fixed_tags": None,
            "per_tag": same["per_tag"],
        }
        if n_boot:
            draws[name] = bootstrap_samples(y[mask], s[mask], groups[mask], tags, fixed, n_boot=n_boot)
            out["subsets"][name]["ci95_fixed_tags"] = {k: _ci(v) for k, v in draws[name].items()}
    if n_boot and "leaked" in draws and "leak_free" in draws:
        # the two subsets share no panorama, so independent draws are the right null
        out["leaked_minus_leak_free"] = {
            k: {"point": out["subsets"]["leaked"]["fixed_tags"][k] - out["subsets"]["leak_free"]["fixed_tags"][k],
                "ci95": _ci(draws["leaked"][k] - draws["leak_free"][k])}
            for k in ("mAP", "micro_f1", "macro_f1")}
    if n_boot and "leaked" in out["subsets"] and "leak_free" in out["subsets"]:
        # the published number against its own leak-free part, paired on the same draws: an
        # upper bound on how much the leak can be inflating the whole-test-set number
        d = paired_subset_diff_samples(y, s, groups, subsets["leak_free"], tags, fixed, n_boot=n_boot)
        out["full_minus_leak_free"] = {
            k: {"point": out["subsets"]["full"]["fixed_tags"][k] - out["subsets"]["leak_free"]["fixed_tags"][k],
                "ci95": _ci(d[k])}
            for k in ("mAP", "micro_f1", "macro_f1")}
    per_label = m[["filename", "city", "label_id", "pano_id", "pano_in_train", "nearest_train_m"] + tags].copy()
    for j, t in enumerate(tags):
        per_label[f"prob:{t}"] = s[:, j]
    return out, per_label


#: Kept as a name for older callers; it is ``NON_TAG_COLUMNS``, not a second list.
LABEL_META_COLS = NON_TAG_COLUMNS


def load_labels(labels_path, split_csv=None):
    """(label table, tag columns), with ``split`` replaced by ``split_csv``'s when given."""
    lab = pd.read_csv(labels_path)
    if split_csv:
        sp = pd.read_csv(split_csv)[["label_uid", "split"]]
        lab = lab.drop(columns="split").merge(sp, on="label_uid", how="inner", validate="one_to_one")
    return lab, [c for c in lab.columns if c not in NON_TAG_COLUMNS]


def score_file(pred_path, labels_path, split_csv=None, fixed=None, n_boot=1000, near_m=10.0):
    """``score`` on files: (summary with input hashes, per-label frame)."""
    lab, tags = load_labels(labels_path, split_csv)
    summary, per_label = score_subsets(pd.read_csv(pred_path), lab, tags, near_m=near_m, n_boot=n_boot,
                                       fixed_tags=fixed)
    summary["pred_file"] = os.path.basename(pred_path)
    summary["pred_sha256"] = sha256_file(pred_path)
    summary["labels_sha256"] = sha256_file(labels_path)
    if split_csv:
        summary["split_csv_sha256"] = sha256_file(split_csv)
    return summary, per_label


def cmd_score(args):
    fixed = args.fixed_tags.split(",") if args.fixed_tags else None
    summary, per_label = score_file(args.pred, args.labels, args.split_csv, fixed, args.n_boot, args.near_m)
    write_json(summary, args.out)
    if args.per_label_out:
        write_csv(per_label, args.per_label_out)
    for k, v in summary["subsets"].items():
        f = v["fixed_tags"]
        print(f"{k:32s} n={v['n']:5d} panos={v['n_panos']:5d} mAP={f['mAP']:.4f} "
              f"microF1={f['micro_f1']:.4f} macroF1={f['macro_f1']:.4f}  CI(mAP)={v['ci95_fixed_tags'] and v['ci95_fixed_tags']['mAP']}")


# ----------------------------------------------------------------------------- contrast

def _arm_frame(pred_path, labels_path, split_csv):
    """(test rows of this arm's split joined to its predictions, tags)."""
    lab, tags = load_labels(labels_path, split_csv)
    te = lab[lab.split == "test"]
    m, y, s = _arrays(pd.read_csv(pred_path), te, tags)
    return m.reset_index(drop=True), y, s, tags


def contrast_files(pred_a, split_a, pred_b, split_b, labels_path, fixed=None, n_boot=1000, seed=86):
    """metric(A) - metric(B) for two prediction files, each on its own split's test labels.

    - ``unpaired``: each file on ALL of its own test labels, resampled independently
      (pano-clustered, A with ``seed``, B with ``seed + 1``). This is the comparison of the
      arms' headline numbers; when the two test sets are different labels it mixes the model
      difference with the difference between the label sets.
    - ``paired``: only the labels that are test in BOTH splits, both files scored on the SAME
      pano-clustered resample, so label-set differences cancel. Fewer labels, same ramps.

    Point estimates and 95 % percentile intervals on the fixed tag set, for mAP and micro/macro
    F1. Returns a dict."""
    fixed = list(fixed or FIXED_TAGS)
    ma, ya, sa, tags = _arm_frame(pred_a, labels_path, split_a)
    mb, yb, sb, tags_b = _arm_frame(pred_b, labels_path, split_b)
    assert tags == tags_b
    keys = ("mAP", "micro_f1", "macro_f1")
    def rel(q):   # path under OUT_DIR, "/"-separated; a file elsewhere is named by its basename
        q = os.path.abspath(q)
        inside = os.path.commonpath([q, os.path.abspath(OUT_DIR)]) == os.path.abspath(OUT_DIR) \
            if os.path.splitdrive(q)[0].lower() == os.path.splitdrive(os.path.abspath(OUT_DIR))[0].lower() else False
        return os.path.relpath(q, OUT_DIR).replace(os.sep, "/") if inside else os.path.basename(q)
    out = {"a": {"pred_file": rel(pred_a), "pred_sha256": sha256_file(pred_a), "n": int(len(ma))},
           "b": {"pred_file": rel(pred_b), "pred_sha256": sha256_file(pred_b), "n": int(len(mb))},
           "tags_fixed": fixed, "n_boot": n_boot, "seed": seed}
    fa = tagger_metrics(ya, sa, tags, selected=fixed)
    fb = tagger_metrics(yb, sb, tags, selected=fixed)
    ga = ma.pano_id.fillna(ma.label_uid).to_numpy()
    gb = mb.pano_id.fillna(mb.label_uid).to_numpy()
    da = bootstrap_samples(ya, sa, ga, tags, fixed, n_boot=n_boot, seed=seed)
    db = bootstrap_samples(yb, sb, gb, tags, fixed, n_boot=n_boot, seed=seed + 1)
    out["unpaired"] = {k: {"a": fa[k], "b": fb[k], "point": fa[k] - fb[k], "ci95": _ci(da[k] - db[k])}
                       for k in keys}
    # paired: the common test labels, in one order, same draws for both
    common = ma[["label_uid"]].reset_index().merge(mb[["label_uid"]].reset_index(), on="label_uid",
                                                    how="inner", validate="one_to_one",
                                                    suffixes=("_a", "_b"))
    ia, ib = common.index_a.to_numpy(), common.index_b.to_numpy()
    assert np.array_equal(ya[ia], yb[ib]), "the same label carries different tags in the two frames"
    y, s1, s2 = ya[ia], sa[ia], sb[ib]
    groups = ma.pano_id.fillna(ma.label_uid).to_numpy()[ia]
    pa = tagger_metrics(y, s1, tags, selected=fixed)
    pb = tagger_metrics(y, s2, tags, selected=fixed)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    d = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        a = tagger_metrics(y[idx], s1[idx], tags, selected=fixed)
        b = tagger_metrics(y[idx], s2[idx], tags, selected=fixed)
        for k in keys:
            d[k].append(np.nan if a[k] is None or b[k] is None else a[k] - b[k])
    out["paired"] = {"n": int(len(common)), "n_panos": int(len(uniq)),
                     **{k: {"a": pa[k], "b": pb[k], "point": pa[k] - pb[k], "ci95": _ci(np.array(d[k]))}
                        for k in keys}}
    return out


def _resolve_arm_file(spec):
    """``<arm>=<predictions path>`` -> (arm, path, the arm's split CSV or None)."""
    arm, path = spec.split("=", 1)
    split = os.path.join(OUT_DIR, ARM_SPLITS[arm]) if ARM_SPLITS[arm] else None
    return arm, path, split


def cmd_contrast(args):
    labels = args.labels
    res = {"labels_sha256": sha256_file(labels), "contrasts": []}
    for a_spec, b_spec in args.pair:
        arm_a, pa, sa = _resolve_arm_file(a_spec)
        arm_b, pb, sb = _resolve_arm_file(b_spec)
        c = contrast_files(pa, sa, pb, sb, labels, n_boot=args.n_boot, seed=args.seed)
        c["name"] = f"{arm_a} ({c['a']['pred_file']}) minus {arm_b} ({c['b']['pred_file']})"
        c["a"]["arm"], c["b"]["arm"] = arm_a, arm_b
        res["contrasts"].append(c)
        u, p = c["unpaired"]["mAP"], c["paired"]["mAP"]
        print(f"{c['name']}: unpaired mAP {u['point']:+.4f} {u['ci95']}; paired on "
              f"{c['paired']['n']} common labels {p['point']:+.4f} {p['ci95']}", flush=True)
    write_json(res, args.out)


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

#: Side of a prepared crop: IMAGE_DIMENSION padded up to the patch multiple (256 -> 266).
PREP_SIDE = IMAGE_DIMENSION + (PATCH_MULTIPLE - IMAGE_DIMENSION % PATCH_MULTIPLE) % PATCH_MULTIPLE


def train_table(labels_path, split_csv=None):
    """The rows ``train`` trains on, in the order it trains on them, and the tag columns.

    ``split_csv`` (``label_uid``, ``split``) replaces the labels file's own ``split`` column.
    ``prep`` and ``train`` both call this, so a prepared array's row order is the trainer's."""
    lab = pd.read_csv(labels_path)
    if split_csv:
        sp = pd.read_csv(split_csv)[["label_uid", "split"]]
        lab = lab.drop(columns="split").merge(sp, on="label_uid", how="inner", validate="one_to_one")
    tags = [c for c in lab.columns if c not in NON_TAG_COLUMNS]
    tr = lab[lab.split == "train"].reset_index(drop=True)
    return tr, tags


def locate_crop(fn, dirs):
    """First ``dir/fn`` that exists, searching ``dirs`` in order."""
    for d in dirs:
        if os.path.exists(os.path.join(d, fn)):
            return os.path.join(d, fn)
    raise FileNotFoundError(fn)


def decode_crop(path):
    """One crop as the trainer sees it before normalisation: uint8, C-contiguous, 3 x 266 x 266.

    The recipe's deterministic preprocessing: decode to RGB, ``Resize((256, 256))`` on the PIL
    image, then zero-pad to the patch multiple (14), split as evenly as possible. ``ToTensor``'s
    /255 and the mean/std normalisation happen per batch on the GPU, so pad pixels stay 0 here.
    The in-memory path of ``train`` and the ``prep`` memmap both call this one function, so the
    two hold the same bytes."""
    from torchvision import io as tvio, transforms
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION))])
    img = tf(tvio.read_image(path, mode=tvio.ImageReadMode.RGB))
    pw = (PATCH_MULTIPLE - img.width % PATCH_MULTIPLE) % PATCH_MULTIPLE
    ph = (PATCH_MULTIPLE - img.height % PATCH_MULTIPLE) % PATCH_MULTIPLE
    img = transforms.Pad((pw // 2, ph // 2, pw - pw // 2, ph - ph // 2))(img)
    return np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1))


def cmd_prep(args):
    """Decode every training crop once into a uint8 ``.npy`` memmap (N x 3 x 266 x 266).

    Same rows, same order and same bytes as the in-memory decode at the top of ``train``
    (``train_table`` + ``decode_crop``). Rows are written one at a time into a memmap, so memory
    stays flat however large N is (tier 2, ~300k crops, is ~64 GB at 212,268 bytes per crop).
    The array is written to ``<out>.partial`` and renamed only when complete, and the sidecar
    ``<out>.meta.json`` is written last, so a killed prep never looks finished; a re-run
    deletes any existing sidecar before it starts, so an old sidecar can never describe a new
    array. The sidecar
    carries the row order (``label_uids``), the labels and split files' sha256, the image dirs
    and the sha256 of the array's data bytes in row order (not of the ``.npy`` file, whose
    header is not data).

    Usage::

        python scripts/analysis/tag_benchmark_86.py prep --labels L.csv --split-csv S.csv \\
            --images /gscratch/.../crops --out /gscratch/.../train_fov25.npy --workers 8
        python scripts/analysis/tag_benchmark_86.py train ... --prep /gscratch/.../train_fov25.npy
    """
    import torch
    import torchvision
    tr, tags = train_table(args.labels, args.split_csv)
    t0 = time.time()
    paths = [locate_crop(fn, args.images) for fn in tr.filename]   # fail before writing anything
    shape = (len(tr), 3, PREP_SIDE, PREP_SIDE)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # A re-run onto an existing --out: drop the old sidecar first, so a kill anywhere below
    # leaves no sidecar at all (train --prep then refuses) rather than an old sidecar paired
    # with a new array.
    if os.path.exists(args.out + ".meta.json"):
        os.remove(args.out + ".meta.json")
    partial = args.out + ".partial"
    arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
    h = hashlib.sha256()
    pool = None
    if args.workers > 1:
        import multiprocessing
        pool = multiprocessing.Pool(args.workers)
        rows = pool.imap(decode_crop, paths, chunksize=32)     # imap keeps input order
    else:
        rows = map(decode_crop, paths)
    try:
        for i, a in enumerate(rows):
            if a.shape != shape[1:]:
                raise SystemExit(f"{paths[i]}: decoded to {a.shape}, expected {shape[1:]}")
            arr[i] = a
            h.update(a.tobytes())
            if (i + 1) % 10000 == 0:
                arr.flush()
                print(f"{i + 1}/{len(paths)} crops, {time.time() - t0:.0f} s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    arr.flush()
    del arr                      # close the map before the rename (Windows refuses otherwise)
    os.replace(partial, args.out)
    done = np.load(args.out, mmap_mode="r")         # the sidecar describes the finished file
    if done.shape != shape or done.dtype != np.uint8:
        raise SystemExit(f"{args.out}: finished array is {done.dtype} {done.shape}, expected uint8 {shape}")
    del done
    meta = {"shape": list(shape), "dtype": "uint8", "array_sha256": h.hexdigest(),
            "label_uids": tr.label_uid.tolist(), "n": int(len(tr)), "tags": tags,
            "labels": os.path.basename(args.labels), "labels_sha256": sha256_file(args.labels),
            "split_csv": os.path.basename(args.split_csv) if args.split_csv else None,
            "split_csv_sha256": sha256_file(args.split_csv) if args.split_csv else None,
            "images": list(args.images), "workers": args.workers, "prep_s": time.time() - t0,
            "host": socket.getfqdn(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json(meta, args.out + ".meta.json")
    print(json.dumps(_round({k: v for k, v in meta.items() if k != "label_uids"}, 3)))


def array_sha256(X, chunk_rows=1024):
    """sha256 of an array's data bytes in row order, read ``chunk_rows`` rows at a time."""
    h = hashlib.sha256()
    for i in range(0, len(X), chunk_rows):
        h.update(np.ascontiguousarray(X[i:i + chunk_rows]).tobytes())
    return h.hexdigest()


def load_prep(prep_path, tr, labels_path, verify=False):
    """Map a ``prep`` array read-only and check it is this training table's.

    Refuses (SystemExit) if the sidecar's row order is not ``tr.label_uid``, the labels file's
    sha256 differs from the one it was prepared from, or the shape is not N x 3 x 266 x 266.
    ``verify`` also re-hashes the array (one full sequential read) against ``array_sha256``.
    Returns ``(X, meta)``; ``X`` is an ``np.memmap``, so indexing it reads only those rows."""
    with open(prep_path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta["label_uids"] != tr.label_uid.tolist():
        raise SystemExit(f"{prep_path}: row order is not this training table's (prepared from "
                         f"{meta['labels']} / {meta['split_csv']}); re-run prep")
    if meta["labels_sha256"] != sha256_file(labels_path):
        raise SystemExit(f"{prep_path}: prepared from a different {meta['labels']} (sha256 differs)")
    X = np.load(prep_path, mmap_mode="r")
    if X.dtype != np.uint8 or X.shape != (len(tr), 3, PREP_SIDE, PREP_SIDE):
        raise SystemExit(f"{prep_path}: {X.dtype} {X.shape}, expected uint8 {(len(tr), 3, PREP_SIDE, PREP_SIDE)}")
    if verify and array_sha256(X) != meta["array_sha256"]:
        raise SystemExit(f"{prep_path}: array sha256 does not match its sidecar")
    return X, meta


def batch_pixels(X, idx):
    """Rows ``idx`` (a LongTensor, any order) of the crop array as a uint8 tensor, in ``idx`` order.

    ``X`` is either the in-memory uint8 tensor or a ``prep`` memmap; for the memmap only these
    rows are read from disk (page cache), never the whole array."""
    import torch
    if isinstance(X, np.ndarray):
        return torch.from_numpy(np.asarray(X[idx.numpy()]))
    return X[idx]


# ----------------------------------------------------------------------------- loss

def load_mask(mask_csv, tr, tags):
    """Per-cell loss mask for ``tr`` as an N x T float32 array (1 = the cell counts, 0 = excluded).

    Convention: a CSV with a ``label_uid`` column and one 0/1 column per tag it masks, named as
    the tag. A tag with no column is unmasked (all 1). Every training ``label_uid`` must have a
    row (a missing row is an error, not a default), and a column that is not a tag is an error,
    so a typo cannot silently unmask a tag. Built from the audit's deployment tag lists (plan
    §2.3): a cell whose tag the label's deployment does not offer is 0.

    Example: amsterdam hides points-into-traffic, so every amsterdam row has
    ``points-into-traffic = 0``; its cells contribute no loss in any ``--loss`` mode."""
    m = pd.read_csv(mask_csv)
    extra = [c for c in m.columns if c != "label_uid" and c not in tags]
    if "label_uid" not in m.columns or extra:
        raise SystemExit(f"{mask_csv}: needs label_uid plus tag columns only (unknown: {extra})")
    j = tr[["label_uid"]].merge(m, on="label_uid", how="left", validate="one_to_one", indicator=True)
    if (j["_merge"] != "both").any():
        raise SystemExit(f"{mask_csv}: {(j['_merge'] != 'both').sum()} training labels have no row")
    out = np.ones((len(tr), len(tags)), np.float32)
    for k, t in enumerate(tags):
        if t in m.columns:
            v = j[t].to_numpy()
            if not np.isin(v, [0, 1]).all():
                raise SystemExit(f"{mask_csv}: column {t} is not 0/1")
            out[:, k] = v
    return out


def load_prior(prior_json, tags):
    """``{tag: pi_t}`` from a JSON object; every tag needs a prior in [0, 1] and no key may be
    a non-tag. pi_t = 0 is allowed and makes that tag's nnPU term exactly naive (pi' = o_t)."""
    with open(prior_json, encoding="utf-8") as fh:
        pr = json.load(fh)
    missing, extra = [t for t in tags if t not in pr], [k for k in pr if k not in tags]
    if missing or extra:
        raise SystemExit(f"{prior_json}: missing priors for {missing}; unknown keys {extra}")
    if not all(0.0 <= float(pr[t]) <= 1.0 for t in tags):
        raise SystemExit(f"{prior_json}: priors must be in [0, 1]")
    return np.array([float(pr[t]) for t in tags], np.float64)


class TagLoss:
    """The ``--loss`` switch: masked BCE (``bce``), non-negative PU risk (``nnpu``) or BCE toward a
    soft target on unlabeled cells (``soft``); per (label, tag) cell, with an optional mask.

    Construct once from the whole training table, call per batch as ``loss(out, yb, idx)`` with
    the batch's logits (b x T), targets and row indices into the table. Returns
    ``(objective, value, fired)``: ``objective`` is what is back-propagated, ``value`` the loss
    that is logged, ``fired`` a length-T bool tensor (nnPU's per-tag gradient-ascent flag) or
    None. ``cmd_train`` uses no ``TagLoss`` at all for ``--loss bce`` without a mask, so that
    path is the recipe's ``nn.BCEWithLogitsLoss()`` call, unchanged.

    Cells. ``mask`` (N x T, 0/1) removes a cell from every term. ``affirmed`` (N, 0/1) marks
    rows whose absences are affirmed (the HF rows): their cells are ordinary BCE in every mode.
    Every other row is a PU row. For tag t, over the whole table's unmasked PU cells:
    n_P = cells tagged t, n_U = all cells (tagged or not; the case-control U* sample),
    o = n_P / n_U, and pi' = max(pi_t, o) with pi_t from ``prior`` (plan §3.1: pi' = o makes the
    tag exactly naive, and a prior below the observed rate would push tagged positives negative).

    ``bce``  value = (1 / (b T)) * sum over unmasked cells of l(z, y); with no mask this is
             BCEWithLogitsLoss's mean (up to float summation order).
    ``soft`` the same, toward target 1 on a tagged cell, 0 on an affirmed untagged cell, and
             pi_U = max(0, (pi' - o) / (1 - o)) on an untagged PU cell: P(t | untagged), plan §3.2.
    ``nnpu`` the non-negative PU risk of Kiryo, Niu, du Plessis and Sugiyama, "Positive-Unlabeled
             Learning with Non-Negative Risk Estimator", NeurIPS 2017, case-control form, with the
             logistic loss l(z, +1) = log(1 + e^-z), l(z, -1) = log(1 + e^z) (the recipe's BCE; the
             paper's experiments used the sigmoid loss), per tag:

                 R_t = pi' E_P[l(z, +1)] + max(0, E_U[l(z, -1)] - pi' E_P[l(z, -1)])

             Estimated with global normalisation so any batch is defined (plan §3.1): with N table
             rows and batch size b, E_X[g] = (N / (b n_X)) * sum over the batch's unmasked PU cells
             in X of g (0 if n_X = 0). The tag's term is (n_U / N) * R_t plus (1 / b) * the BCE sum
             over the batch's unmasked affirmed cells; value = mean of the T terms. With pi' = o
             this equals ``bce`` term for term. Algorithm 1 of the paper: when the bracket
             B_t = E_U[l(z,-1)] - pi' E_P[l(z,-1)] < -beta, the step for that tag is gradient
             *ascent* on B_t (the tag's objective term is (n_U / N) * (-gamma * B_t) plus its
             affirmed BCE) instead of descent on R_t; ``value`` still reports R_t.

    Example (one tag, no mask, no affirmed rows, table of N = 4 with 1 tagged row, so
    n_P = 1, n_U = 4, o = 0.25; prior 0.5 -> pi' = 0.5; batch rows 0 (tagged, z0) and 1 (z1)):
    E_P[l+] = 4/(2*1) l(z0,+1), E_U[l-] = 4/(2*4) (l(z0,-1) + l(z1,-1)),
    value = (4/4) * (0.5 E_P[l+] + max(0, E_U[l-] - 0.5 * 4/(2*1) l(z0,-1))).
    """

    KINDS = ("bce", "nnpu", "soft")

    def __init__(self, kind, Y, mask=None, affirmed=None, prior=None, beta=0.0, gamma=1.0):
        import torch
        if kind not in self.KINDS:
            raise ValueError(kind)
        if kind != "bce" and prior is None:
            raise SystemExit(f"--loss {kind} needs --prior")
        Y = np.asarray(Y, np.float64)
        n, t = Y.shape
        mask = np.ones((n, t)) if mask is None else np.asarray(mask, np.float64)
        aff = np.zeros(n) if affirmed is None else np.asarray(affirmed, np.float64)
        if not np.isin(aff, [0, 1]).all():
            raise SystemExit("affirmed must be 0/1")
        pu = mask * (1.0 - aff)[:, None]
        self.kind, self.N, self.T, self.beta, self.gamma = kind, n, t, float(beta), float(gamma)
        self.n_P = (pu * Y).sum(0)
        self.n_U = pu.sum(0)
        self.o = np.divide(self.n_P, self.n_U, out=np.zeros(t), where=self.n_U > 0)
        self.prior = None if prior is None else np.asarray(prior, np.float64)
        self.pi_prime = None if prior is None else np.maximum(self.prior, self.o)
        self.pi_U = None if prior is None else np.divide(
            np.maximum(self.pi_prime - self.o, 0.0), 1.0 - self.o, out=np.zeros(t), where=self.o < 1)
        # nnPU coefficients (float64, fixed for the run): 1/n_P, 1/n_U (0 where the count is 0) and
        # k_P = 1/n_U - pi'/n_P, set to exactly 0 where pi' = o (the naive case).
        self.inv_n_P = np.divide(1.0, self.n_P, out=np.zeros(t), where=self.n_P > 0)
        self.inv_n_U = np.divide(1.0, self.n_U, out=np.zeros(t), where=self.n_U > 0)
        self.k_P = None if prior is None else np.where(
            (self.pi_prime == self.o) | (self.n_P == 0), 0.0, self.inv_n_U - self.pi_prime * self.inv_n_P)
        self.mask = torch.from_numpy(mask.astype(np.float32))
        self.affirmed = torch.from_numpy(aff.astype(np.float32))

    def summary(self):
        """Per-tag counts and priors as run, for the run meta."""
        d = {"loss": self.kind, "n_rows": self.N, "n_P": self.n_P.tolist(), "n_U": self.n_U.tolist(),
             "o": self.o.tolist()}
        if self.prior is not None:
            d.update(prior=self.prior.tolist(), pi_prime=self.pi_prime.tolist(), pi_U=self.pi_U.tolist())
        if self.kind == "nnpu":
            d.update(nnpu_beta=self.beta, nnpu_gamma=self.gamma)
        return d

    def __call__(self, out, yb, idx):
        import torch
        import torch.nn.functional as F
        dev, b = out.device, out.shape[0]
        m = self.mask[idx].to(dev)
        a = self.affirmed[idx].to(dev)[:, None]
        if self.kind == "bce":
            v = (m * F.binary_cross_entropy_with_logits(out, yb, reduction="none")).sum() / (b * self.T)
            return v, v, None
        if self.kind == "soft":
            pi_u = torch.tensor(self.pi_U, dtype=out.dtype, device=dev)
            target = yb + (1 - yb) * (1 - a) * pi_u
            v = (m * F.binary_cross_entropy_with_logits(out, target, reduction="none")).sum() / (b * self.T)
            return v, v, None

        def f(x):
            return torch.tensor(x, dtype=out.dtype, device=dev)
        pu, hf = m * (1 - a), m * a
        l_pos, l_neg = F.softplus(-out), F.softplus(out)               # l(z, +1), l(z, -1)
        hf_term = (hf * F.binary_cross_entropy_with_logits(out, yb, reduction="none")).sum(0) / b
        e_p_pos = (self.N / b) * f(self.inv_n_P) * (pu * yb * l_pos).sum(0)
        # B_t = E_U[l-] - pi' E_P[l-], regrouped per cell so that pi' = o gives k_P = 0 exactly:
        # (N / b) * (sum_untagged l- / n_U + (1 / n_U - pi' / n_P) * sum_tagged l-). Computed as the
        # plain difference, float rounding made B_t = -1e-8 on an all-positive batch and fired the
        # ascent step where the maths says B_t >= 0.
        bracket = (self.N / b) * (f(self.inv_n_U) * (pu * (1 - yb) * l_neg).sum(0)
                                  + f(self.k_P) * (pu * yb * l_neg).sum(0))
        pi = f(self.pi_prime)
        w = f(self.n_U) / self.N
        risk = w * (pi * e_p_pos + bracket.clamp(min=0))
        fired = bracket.detach() < -self.beta
        obj = torch.where(fired, w * (-self.gamma * bracket), risk)
        value = (risk + hf_term).sum() / self.T
        objective = (obj + hf_term).sum() / self.T
        return objective, value, fired.cpu()


def make_loss(kind, tr, tags, mask_csv=None, prior_json=None, beta=0.0, gamma=1.0):
    """The ``TagLoss`` for ``train``'s flags, or None for ``--loss bce`` with no ``--mask-csv``
    (the recipe's own ``nn.BCEWithLogitsLoss()`` path, bit for bit)."""
    if kind == "bce" and not mask_csv:
        return None
    mask = load_mask(mask_csv, tr, tags) if mask_csv else None
    aff = tr["affirmed"].to_numpy() if "affirmed" in tr.columns else None
    prior = load_prior(prior_json, tags) if prior_json else None
    return TagLoss(kind, tr[tags].to_numpy(np.float32), mask=mask, affirmed=aff, prior=prior,
                   beta=beta, gamma=gamma)


# ----------------------------------------------------------------------------- train loop

#: Written every epoch by ``train --resume`` (only then); a requeued ``--resume`` continues from it.
CHECKPOINT_NAME = "checkpoint.pth"
#: The recipe's kept checkpoint (best training exact-match accuracy, ties -> lower loss).
BEST_NAME = "best.pth"


def _restore_best(out_dir, ck):
    """Make ``best.pth`` agree with the checkpoint a resume is about to continue from.

    Under ``--resume`` an epoch's writes go in this order: the new best weights (if any) to
    ``best.pth.pending``, then ``checkpoint.pth`` (temp file + rename; it records
    ``best_epoch``), then ``best.pth.pending`` is renamed onto ``best.pth``. So after a kill at
    any point, either the checkpoint is the previous epoch's and ``best.pth`` still agrees with
    it (a leftover ``.pending`` is from the dead epoch and is deleted), or the checkpoint says
    this epoch was the best and ``best.pth`` may still hold the older one. In that second case
    the best weights ARE the checkpoint's model state, so ``best.pth`` is re-saved from it.
    Any other disagreement (``best.pth`` copied in from elsewhere, or edited by hand) cannot be
    repaired from the checkpoint and is refused.

    Returns the epoch the stale ``best.pth`` held (None if nothing was restored).

    Example: killed after epoch 5's checkpoint rename but before its ``best.pth`` rename ->
    checkpoint ``epoch == best_epoch == 5``, ``best.pth`` says 3 -> ``best.pth`` becomes epoch
    5's weights and the function returns 3."""
    import torch
    best_path = os.path.join(out_dir, BEST_NAME)
    if os.path.exists(best_path + ".pending"):
        os.remove(best_path + ".pending")
    want = ck.get("best_epoch")
    if want is None:
        return None
    have = torch.load(best_path, map_location="cpu")["epoch"] if os.path.exists(best_path) else None
    if have == want:
        return None
    if want != ck["epoch"]:
        raise SystemExit(f"{best_path} holds epoch {have} but {CHECKPOINT_NAME} (epoch {ck['epoch']}) "
                         f"says the best is epoch {want}, whose weights it does not carry; "
                         "refusing to resume (restore that best.pth or start a fresh --out-dir)")
    torch.save({"epoch": want, "model_state_dict": ck["model_state_dict"], "loss": ck["best_loss"]},
               best_path + ".pending")
    os.replace(best_path + ".pending", best_path)
    print(f"{best_path}: held epoch {have}, restored to epoch {want} from {CHECKPOINT_NAME}", flush=True)
    return have


def run_training(model, X, Y, *, epochs, batch, lr, seed, out_dir, device, resume=False, fingerprint=None,
                 loss_fn=None, tags=None, stats=None):
    """The recipe's training loop, factored out of ``cmd_train`` so a small stand-in model can be
    driven through it on CPU (``tests/test_tag_trainer_86.py``). Behaviour is the pre-factoring
    loop's, statement for statement: Adam(lr), a ``torch.Generator`` seeded with ``seed`` whose
    ``randperm`` sets each epoch's order, batches of ``batch`` rows, pixels /255 then ImageNet
    mean/std on ``device``, BCE-with-logits, ``best.pth`` kept by best training exact-match
    accuracy (ties -> lower loss), ``train_log.csv`` rewritten every epoch. Without ``resume``
    nothing else is written, exactly as before.

    Resume. With ``resume=True`` the loop also writes ``checkpoint.pth`` after every epoch, to
    a temp name and then an atomic rename, BEFORE the epoch's line is printed (so a watcher
    that reacts to the line finds that epoch's checkpoint): model and optimizer state, the
    shuffle generator's state, torch's global CPU/CUDA RNG state, the last finished epoch,
    ``best_acc``/``best_loss``/``best_epoch``, the log rows and ``fingerprint``. If a checkpoint
    is present, all of that is loaded and the loop starts at the next epoch; without one it
    starts fresh, so a requeued job can always pass ``--resume``. Everything the next epoch
    depends on comes from the checkpoint, never from the process (the rule
    ``docs/stage2_cosine_rung_135.md`` states for the stateless LR schedule): there is no
    scheduler, and the epoch's order is the generator's next ``randperm``.

    A resume is refused when ``fingerprint`` (a JSON-able dict of the settings a resume must not
    change; ``cmd_train`` builds it with ``train_fingerprint``) differs from the checkpoint's,
    or when ``epochs`` is smaller than the number of epochs the checkpoint has finished.
    ``epochs`` is deliberately not in the fingerprint, so a finished run can be extended.
    ``best.pth`` is made consistent with the checkpoint first (``_restore_best``), and
    ``train_log.csv`` is rewritten from the checkpoint's log rows.

    ``loss_fn`` None is the recipe's ``nn.BCEWithLogitsLoss()``; otherwise a ``TagLoss``, called
    as ``loss_fn(logits, yb, idx)``. For nnPU the log row also carries, per tag in ``tags``, the
    fraction of the epoch's steps on which the non-negative correction fired (``clamp:<tag>``).

    ``stats``, if a dict, receives ``checkpoint_write_s`` (seconds this process spent writing
    ``checkpoint.pth`` and renaming ``best.pth``; 0 without ``resume``; not in any ``epoch_s``)
    and ``best_restored_from`` (``_restore_best``'s return).

    Returns ``(log_rows, resumed_from_epoch)``, the latter None for a fresh start.

    Usage (the resume test, abridged)::

        run_training(m1, X, Y, epochs=2, ..., out_dir=a)                 # straight
        run_training(m2, X, Y, epochs=1, ..., out_dir=b, resume=True)    # killed after epoch 0
        run_training(m3, X, Y, epochs=2, ..., out_dir=b, resume=True)    # requeued
        # m3's weights == m1's, bit for bit (CPU)
    """
    import torch
    from torch import nn, optim
    from sklearn.metrics import accuracy_score
    dev = device
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    os.makedirs(out_dir, exist_ok=True)
    log_rows, best_acc, best_loss = [], 0.0, 100.0
    g = torch.Generator().manual_seed(seed)
    ckpt_path = os.path.join(out_dir, CHECKPOINT_NAME)
    best_path = os.path.join(out_dir, BEST_NAME)
    start, resumed_from, best_epoch, ckpt_write_s, restored = 0, None, None, 0.0, None
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        if ck["fingerprint"] != fingerprint:
            diff = sorted(k for k in set(ck["fingerprint"] or {}) | set(fingerprint or {})
                          if (ck["fingerprint"] or {}).get(k) != (fingerprint or {}).get(k))
            raise SystemExit(f"{ckpt_path}: written under different settings ({', '.join(diff)}); "
                             "refusing to resume")
        if ck["epoch"] + 1 > epochs:
            raise SystemExit(f"{ckpt_path}: {ck['epoch'] + 1} epochs already finished but --epochs is "
                             f"{epochs}; pass --epochs {ck['epoch'] + 1} or more (or a fresh --out-dir)")
        restored = _restore_best(out_dir, ck)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        g.set_state(ck["generator_state"])
        torch.set_rng_state(ck["torch_rng_state"])
        if ck.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng_state"])
        log_rows, best_acc, best_loss = ck["log_rows"], ck["best_acc"], ck["best_loss"]
        best_epoch = ck.get("best_epoch")
        write_csv(pd.DataFrame(log_rows), os.path.join(out_dir, "train_log.csv"))
        start = resumed_from = ck["epoch"] + 1
        print(f"resuming at epoch {start} from {ckpt_path}", flush=True)
    for epoch in range(start, epochs):
        model.train()
        te = time.time()
        perm = torch.randperm(len(X), generator=g)
        losses, accs, fires = [], [], []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = (batch_pixels(X, idx).to(dev).float() / 255.0 - mean) / std
            yb = Y[idx].to(dev)
            opt.zero_grad()
            out = model(xb).squeeze(dim=1)
            if loss_fn is None:
                loss = objective = crit(out, yb)
            else:
                objective, loss, fired = loss_fn(out, yb, idx)
                if fired is not None:
                    fires.append(fired.numpy())
            objective.backward()
            opt.step()
            losses.append(loss.item())
            pred = (torch.sigmoid(out) > 0.5).float()
            accs.append(accuracy_score(yb.cpu().numpy(), pred.detach().cpu().numpy()))
        el, ea = float(np.mean(losses)), float(np.mean(accs))
        saved = ""
        if ea > best_acc or (ea == best_acc and el < best_loss):
            best_acc, best_loss, best_epoch = ea, el, epoch
            # without --resume: best.pth directly, as before; with it: renamed after the checkpoint
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": el},
                       best_path + ".pending" if resume else best_path)
            saved = "best"
        row = {"epoch": epoch, "loss": el, "train_exact_match_acc": ea, "epoch_s": time.time() - te, "saved": saved}
        if fires:
            row.update({f"clamp:{t}": float(v) for t, v in zip(tags, np.mean(fires, axis=0))})
        log_rows.append(row)
        if resume:
            tc = time.time()
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(), "generator_state": g.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "best_acc": best_acc, "best_loss": best_loss, "best_epoch": best_epoch,
                        "log_rows": log_rows, "fingerprint": fingerprint}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            if saved:
                os.replace(best_path + ".pending", best_path)
            ckpt_write_s += time.time() - tc
        print(json.dumps(_round(row, 5)), flush=True)
        write_csv(pd.DataFrame(log_rows), os.path.join(out_dir, "train_log.csv"))
    if stats is not None:
        stats.update(checkpoint_write_s=ckpt_write_s, best_restored_from=restored)
    return log_rows, resumed_from


def check_train_flags(args):
    """Refuse ``train`` flag combinations that would otherwise be silently ignored.

    Example: ``--prep a.npy --images crops/`` -> SystemExit naming both flags, instead of
    training on ``a.npy`` without saying so."""
    errs = []
    if args.prep and args.images:
        errs.append("--prep and --images both given; pass one (--prep maps a prepared array, "
                    "--images decodes in memory)")
    if not args.prep and not args.images:
        errs.append("train needs --images (decode in memory) or --prep (a prep array)")
    if args.verify_prep and not args.prep:
        errs.append("--verify-prep needs --prep")
    if args.prior and args.loss == "bce":
        errs.append("--prior is not used by --loss bce; drop it or pick --loss nnpu|soft")
    if args.loss != "nnpu" and (args.nnpu_beta != 0.0 or args.nnpu_gamma != 1.0):
        errs.append("--nnpu-beta / --nnpu-gamma are only used by --loss nnpu")
    if errs:
        raise SystemExit("train: " + "; ".join(errs))


def train_fingerprint(args, tr, tags, tagger_sha, pixels_sha256):
    """The settings a ``train --resume`` must not change, as a JSON-able dict.

    ``cmd_train`` builds it with this function and stores it in ``checkpoint.pth``; a resume
    whose fingerprint differs in any key is refused, naming the keys. It covers the row
    identity (``label_uids_sha256``), the targets (``targets_sha256``: the tag matrix as
    float32, row-major, in training order; ``affirmed_sha256``: the ``affirmed`` column, or
    None if the table has none), the pixels (``pixels_sha256``: the uint8 crop array's data
    bytes in row order -- the ``prep`` sidecar's ``array_sha256`` under ``--prep``, or the same
    hash computed over the in-memory decode, so the two paths agree for the same bytes), the
    model (``backbone_sha256`` of the ``--backbone`` file, ``tagger_sha``), the recipe (lr,
    batch, seed) and the loss (``--loss``, β, γ, and the sha256 of the mask CSV and prior
    JSON). Needs no GPU. ``--epochs`` is deliberately not in it (a run may be extended).

    Example::

        fp = train_fingerprint(args, tr, tags, "3b7405cd...", prep_meta["array_sha256"])
        fp["targets_sha256"]   # changes if any tag value in tr flips
    """
    y = np.ascontiguousarray(tr[tags].to_numpy(np.float32))
    aff = (np.ascontiguousarray(tr["affirmed"].to_numpy(np.float32)) if "affirmed" in tr.columns else None)
    return {"n_train": int(len(tr)), "tags": list(tags), "lr": args.lr, "batch": args.batch, "seed": args.seed,
            "label_uids_sha256": hashlib.sha256("\n".join(tr.label_uid.astype(str)).encode()).hexdigest(),
            "targets_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "affirmed_sha256": hashlib.sha256(aff.tobytes()).hexdigest() if aff is not None else None,
            "pixels_sha256": pixels_sha256,
            "backbone_sha256": sha256_file(args.backbone), "tagger_sha": tagger_sha,
            "loss": args.loss, "nnpu_beta": args.nnpu_beta, "nnpu_gamma": args.nnpu_gamma,
            "mask_sha256": sha256_file(args.mask_csv) if args.mask_csv else None,
            "prior_sha256": sha256_file(args.prior) if args.prior else None}


#: Recorded in ``train_meta.json`` for any run with a ``TagLoss`` (plan §5.3): exact-match
#: training accuracy counts masked cells and treats unlabeled cells as negatives, so it is not a
#: selection rule there.
PLACEHOLDER_BEST_RULE = ("placeholder: best.pth is kept by training exact-match accuracy, which is "
                         "not a valid selection rule under --loss nnpu/soft or --mask-csv (plan 5.3); "
                         "score last.pth or a retained checkpoint instead")


def cmd_train(args):
    """The tagger's DINOv2 recipe (notebooks/dino-trainer.ipynb at TAGGER_SHA): full
    fine-tune, Adam lr 1e-6, batch 4, shuffle, BCEWithLogitsLoss, 100 epochs, no
    augmentation, checkpoint kept by best *training* exact-match accuracy (ties -> lower
    loss). Two deliberate differences, both stated in the doc: a fixed seed, and the
    deterministic preprocessing is computed once and cached in memory (identical tensors,
    much faster epochs). ``--prep`` maps a ``prep`` array instead of decoding (same bytes).

    The steps that need CUDA (``torch.device("cuda")``, the model on the GPU,
    ``get_device_name``) are not under the CPU test suite; ``run_training``,
    ``train_fingerprint`` and ``check_train_flags`` are. ``train_meta.json`` timing fields:
    ``prep_s``, ``train_s`` and ``elapsed_s`` cover this process only. With ``--resume`` it
    also has ``resumed_from_epoch``, ``epoch_s_sum_all_runs`` (Σ ``epoch_s`` over every log
    row, from every process that ran this out-dir; excludes checkpoint writes and any epoch
    lost to a kill), ``checkpoint_write_s`` (this process's checkpoint writes, which ARE in its
    ``train_s``) and ``timing_scope`` restating this."""
    import torch
    check_train_flags(args)
    sha = check_tagger(args.tagger_repo, args.tagger_sha)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr, tags = train_table(args.labels, args.split_csv)
    t0 = time.time()
    prep_meta = None
    if args.prep:
        X, prep_meta = load_prep(args.prep, tr, args.labels, verify=args.verify_prep)
    else:
        # uint8 N,3,266,266 -- ToTensor is /255, pad pixels stay 0
        X = torch.stack([torch.from_numpy(decode_crop(locate_crop(fn, args.images))) for fn in tr.filename])
    Y = torch.tensor(tr[tags].to_numpy(np.float32))
    prep_s = time.time() - t0
    dev = torch.device("cuda")
    model = build_model(args.tagger_repo, len(tags), backbone=args.backbone).to(dev)
    loss_fn = make_loss(args.loss, tr, tags, mask_csv=args.mask_csv, prior_json=args.prior,
                        beta=args.nnpu_beta, gamma=args.nnpu_gamma)
    if loss_fn is not None:
        print(f"note: {PLACEHOLDER_BEST_RULE}", flush=True)
    fingerprint = None
    if args.resume:        # hashing only when a checkpoint will carry it; no RNG is consumed
        fingerprint = train_fingerprint(args, tr, tags, sha, prep_meta["array_sha256"] if prep_meta
                                        else array_sha256(X.numpy()))
    stats = {}
    t_train = time.time()
    log_rows, resumed_from = run_training(model, X, Y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                                          seed=args.seed, out_dir=args.out_dir, device=dev,
                                          resume=args.resume, fingerprint=fingerprint, loss_fn=loss_fn, tags=tags,
                                          stats=stats)
    torch.save({"epoch": args.epochs - 1, "model_state_dict": model.state_dict()},
               os.path.join(args.out_dir, "last.pth"))
    meta = {"tagger_sha": sha, "n_train": int(len(tr)), "tags": tags, "epochs": args.epochs, "lr": args.lr,
            "batch": args.batch, "seed": args.seed, "prep_s": prep_s, "train_s": time.time() - t_train,
            "elapsed_s": time.time() - t0, "best_epoch": int(max(
                (r for r in log_rows if r["saved"]), key=lambda r: r["epoch"])["epoch"]),
            "host": socket.getfqdn(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "split_csv": os.path.basename(args.split_csv) if args.split_csv else "hf test.csv/train.csv",
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if loss_fn is not None:
        meta.update(loss_summary=loss_fn.summary(), mask_csv=os.path.basename(args.mask_csv) if args.mask_csv else None,
                    mask_sha256=sha256_file(args.mask_csv) if args.mask_csv else None,
                    prior=os.path.basename(args.prior) if args.prior else None,
                    prior_sha256=sha256_file(args.prior) if args.prior else None,
                    best_pth_rule=PLACEHOLDER_BEST_RULE)
    if args.resume:
        meta.update(resumed_from_epoch=resumed_from,
                    epoch_s_sum_all_runs=float(sum(r["epoch_s"] for r in log_rows)),
                    checkpoint_write_s=stats["checkpoint_write_s"],
                    best_pth_restored_from_epoch=stats["best_restored_from"], fingerprint=fingerprint,
                    timing_scope="prep_s, train_s, elapsed_s, checkpoint_write_s: this process only "
                                 "(train_s includes checkpoint_write_s); epoch_s_sum_all_runs: sum of "
                                 "epoch_s over all log rows of every run, without checkpoint writes or "
                                 "epochs lost to a kill")
    if prep_meta is not None:
        meta.update(prep=os.path.basename(args.prep), prep_array_sha256=prep_meta["array_sha256"],
                    prep_verified=bool(args.verify_prep))
    write_json(meta, os.path.join(args.out_dir, "train_meta.json"))
    print(json.dumps(_round(meta, 3)))


# ----------------------------------------------------------------------------- test-only

def filter_test_lines(pred_path, test_filenames):
    """The header plus the rows of a predictions CSV whose filename is in ``test_filenames``,
    in source order, byte for byte (a text filter, so no float is re-formatted)."""
    with open(pred_path, "rb") as fh:
        lines = fh.read().split(b"\n")
    keep = [lines[0]] + [ln for ln in lines[1:]
                         if ln and ln.split(b",", 1)[0].decode("utf-8") in test_filenames]
    return b"\n".join(keep) + b"\n", len(keep) - 1


def write_test_only(pred_path, labels_path, split_csv, out_path):
    """Keep only the labels that are ``test`` under ``split_csv`` (or the HF split). The meta
    beside the output describes the OUTPUT (rows, sha256) and carries the source
    predictions' own meta and hash, so the chain checkpoint -> 10,857 logits -> test rows can
    be checked link by link."""
    lab, _ = load_labels(labels_path, split_csv)
    test = set(lab.loc[lab.split == "test", "filename"])
    data, n = filter_test_lines(pred_path, test)
    assert n == len(test), f"{pred_path}: {n} test rows found, {len(test)} expected"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(data)
    src_meta = pred_path + ".meta.json"
    meta = {"file": os.path.basename(out_path), "n_rows": n, "predictions_sha256": sha256_file(out_path),
            "filter": "rows whose label is test under split_csv (tag_benchmark_86.py test-only)",
            "split_csv": os.path.basename(split_csv) if split_csv else "hf test.csv/train.csv",
            "split_csv_sha256": sha256_file(split_csv) if split_csv else None,
            "labels_sha256": sha256_file(labels_path),
            "source_file": os.path.basename(pred_path), "source_sha256": sha256_file(pred_path),
            "source_meta": json.load(open(src_meta, encoding="utf-8")) if os.path.exists(src_meta) else None}
    write_json(meta, out_path + ".meta.json")
    return meta


def cmd_test_only(args):
    meta = write_test_only(args.pred, args.labels, args.split_csv, args.out)
    print(f"wrote {args.out}: {meta['n_rows']} rows, sha256 {meta['predictions_sha256']}")


# ----------------------------------------------------------------------------- collect

def _utc(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iso(t):
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def train_interval(work, arm):
    """(start, end or None) of one arm's training, from its logs: the first line of
    ``train_<arm>.log`` is the launch's ``date -u``; the end is ``train_meta.json``'s ts,
    written when training returns (None while it is still running)."""
    with open(os.path.join(work, f"train_{arm}.log"), encoding="utf-8") as fh:
        start = _utc(fh.readline().strip())
    meta = os.path.join(work, f"train_{arm}", "train_meta.json")
    end = _utc(json.load(open(meta, encoding="utf-8"))["ts"]) if os.path.exists(meta) else None
    return start, end


def concurrent_runs(intervals, start, end, min_overlap=0.5):
    """Labels of the runs in ``intervals`` ({label: (start, end or None)}) that overlap
    [start, end] for at least ``min_overlap`` of its length. The GPU is shared with them,
    so a sum of wall-clock across rows overstates GPU occupancy (``gpu_share``)."""
    span = max((end - start).total_seconds(), 1e-9)
    out = []
    for label, (a, b) in sorted(intervals.items()):
        b = b or dt.datetime.max.replace(tzinfo=dt.timezone.utc)
        ov = (min(end, b) - max(start, a)).total_seconds()
        if ov >= min_overlap * span:
            out.append(label)
    return out


def share_fields(concurrent):
    """Machine-readable GPU sharing: ``gpu_share`` = 1 / (1 + runs of ours on the same GPU
    at the same time). elapsed_s x gpu_share is this run's nominal share of the GPU. Other
    users' jobs on the same GPU are not counted (the doc says which)."""
    return {"concurrent_with": list(concurrent), "gpu_share": round(1.0 / (1 + len(concurrent)), 6)}


def collect_arm(arm, work, log_path, epochs=SNAPSHOT_EPOCHS, final=False, n_boot=1000, out_dir=OUT_DIR,
                labels_path=None, host="makelab2.cs.washington.edu", gpu="NVIDIA A40"):
    """The CPU side of finishing one retrain arm, from what the GPU steps left in ``work``.

    For every snapshot epoch E whose ``snap_ep<E>_<arm>_predictions.csv`` exists: the test
    rows -> ``train_<arm>_ep<E>_test_predictions.csv`` (+ meta), scores ->
    ``train_<arm>_ep<E>_scores.json``, and a usage row for that inference. With ``final``: the
    run must have ended with ``EXIT 0``; the checkpoint scored is ``best.pth``, whose epoch is
    read from the checkpoint (when torch is present) and checked against ``train_meta.json``;
    the post-train predictions must name that checkpoint's sha256; the arm's log and meta are
    copied under arm-prefixed names; outputs are named ``_final_`` with ``best_epoch`` inside;
    and the training row replaces its ``in_progress`` row (same ``run_id``).

    Every usage row carries a ``run_id``, so running this twice replaces rather than
    double counts (``rampnet.ledger.latest_rows``). Returns the list of rows written."""
    sys.path.insert(0, REPO)
    from rampnet import ledger
    labels_path = labels_path or os.path.join(out_dir, "hf_curbramp_labels.csv")
    split = os.path.join(out_dir, ARM_SPLITS[arm]) if ARM_SPLITS[arm] else None
    trains = {f"train-{a}": train_interval(work, a) for a in ARM_SPLITS
              if os.path.exists(os.path.join(work, f"train_{a}.log"))}
    rows = []

    def infer_row(pred, label, what):
        m = json.load(open(pred + ".meta.json", encoding="utf-8"))
        end = _utc(m["ts"])
        start = end - dt.timedelta(seconds=m["elapsed_s"])
        conc = concurrent_runs(trains, start, end)
        return usage_row(label, m["elapsed_s"], "ok", what, n=m["n_scored"], ts=_iso(start), host=host,
                         gpu=m.get("gpu") or gpu,
                         extra={"run_id": f"tagger-86:{label}:{_iso(start)}", **share_fields(conc)})

    def score_to(pred_full, stem, extra=None):
        test_pred = os.path.join(out_dir, f"{stem}_test_predictions.csv")
        write_test_only(pred_full, labels_path, split, test_pred)
        summary, _ = score_file(test_pred, labels_path, split, FIXED_TAGS, n_boot)
        if extra:
            summary.update(extra)
        write_json(summary, os.path.join(out_dir, f"{stem}_scores.json"))
        f = summary["subsets"]["full"]["fixed_tags"]
        print(f"{stem}: n={summary['n_scored']} mAP={f['mAP']:.4f} microF1={f['micro_f1']:.4f} "
              f"macroF1={f['macro_f1']:.4f}")

    for e in epochs:
        pred = os.path.join(work, f"snap_ep{e}_{arm}_predictions.csv")
        if not os.path.exists(pred):
            print(f"{arm} ep{e}: no {pred}; skipped")
            continue
        score_to(pred, f"train_{arm}_ep{e}")
        rows.append(infer_row(pred, f"infer-train-{arm}-ep{e}",
                              f"retrained {arm} arm, best.pth as of epoch index {e} ({e + 1} epochs), "
                              f"all 10,857 crops (#86 item 2)"))

    if final:
        log = open(os.path.join(work, f"train_{arm}.log"), encoding="utf-8").read().splitlines()
        if "EXIT 0" not in log:
            raise SystemExit(f"{arm}: train_{arm}.log has no 'EXIT 0' line; the run has not finished cleanly")
        d = os.path.join(work, f"train_{arm}")
        meta = json.load(open(os.path.join(d, "train_meta.json"), encoding="utf-8"))
        best = os.path.join(d, "best.pth")
        best_sha = sha256_file(best)
        try:
            import torch
            ck_epoch = int(torch.load(best, map_location="cpu")["epoch"])
            assert ck_epoch == meta["best_epoch"], f"best.pth epoch {ck_epoch} != train_meta best_epoch {meta['best_epoch']}"
        except ImportError:
            ck_epoch = None
        pred = os.path.join(work, f"train_{arm}_predictions.csv")
        pm = json.load(open(pred + ".meta.json", encoding="utf-8"))
        assert pm["checkpoint_sha256"] == best_sha, f"{pred} was not made from {best}"
        for src, dst in (("train_log.csv", f"train_{arm}_log.csv"), ("train_meta.json", f"train_{arm}_meta.json")):
            with open(os.path.join(d, src), "rb") as a, open(os.path.join(out_dir, dst), "wb") as b:
                b.write(a.read())
        score_to(pred, f"train_{arm}_final",
                 extra={"best_epoch": meta["best_epoch"], "best_epoch_from_checkpoint": ck_epoch,
                        "epochs": meta["epochs"], "checkpoint_sha256": best_sha})
        start, end = trains[f"train-{arm}"]
        conc = concurrent_runs({k: v for k, v in trains.items() if k != f"train-{arm}"}, start, end)
        rows.append(usage_row(f"train-{arm}", meta["elapsed_s"], "ok",
                              f"tagger DINOv2 recipe ({meta['epochs']} ep, Adam {meta['lr']}, batch {meta['batch']}) "
                              f"on the {arm} split; best.pth = epoch index {meta['best_epoch']}; elapsed from "
                              f"train_meta.json (#86 item 2)",
                              n=meta["n_train"], ts=_iso(start), host=host, gpu=meta.get("gpu") or gpu,
                              extra={"run_id": f"tagger-86:train-{arm}:{_iso(start)}", **share_fields(conc)}))
        rows.append(infer_row(pred, f"infer-train-{arm}-final",
                              f"retrained {arm} arm, final best.pth (epoch index {meta['best_epoch']}), "
                              f"all 10,857 crops (#86 item 2)"))
    if rows:
        ledger.append_rows(log_path, rows)
    return rows


def cmd_collect(args):
    epochs = [int(e) for e in args.epochs.split(",")] if args.epochs else []
    for arm in args.arms:
        for r in collect_arm(arm, args.work, args.log, epochs=epochs, final=args.final, n_boot=args.n_boot):
            print(json.dumps(r))


# ----------------------------------------------------------------------------- usage

def usage_row(label, elapsed_s, status, what, n=None, host="makelab2.cs.washington.edu",
              gpu="NVIDIA A40", ts=None, extra=None):
    """One ``paid: false`` row for analysis_out/usage_log.jsonl (docs/compute_cost.md: GPU
    time on a host without Slurm goes there). Failed runs get a row too. ``ts`` is the run's
    start. Rows written by ``collect`` also carry ``run_id`` (so a re-write replaces rather
    than adds) and ``concurrent_with`` / ``gpu_share`` (see share_fields)."""
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
    extra = {}
    if args.run_id:
        extra["run_id"] = args.run_id
    # a dedicated GPU is gpu_share 1.0, written out, so every row says what it shared
    extra.update(share_fields([c for c in (args.concurrent_with or "").split(",") if c]))
    row = usage_row(args.label, args.elapsed_s, args.status, args.what, n=args.n, ts=args.ts, extra=extra,
                    host=args.host, gpu=args.gpu)
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

    p = sub.add_parser("contrast", help="metric(A) - metric(B) for two arms' predictions, unpaired and paired")
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ARM=PRED_A", "ARM=PRED_B"),
                   help="two test-prediction files, each prefixed by the arm whose split it is scored on")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_contrast)

    p = sub.add_parser("resplit", help="seeded pano-grouped re-split")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--group", default="pano", choices=["pano", "cell"],
                   help="pano: no panorama on both sides; cell: no ~--cell-m grid cell on both sides")
    p.add_argument("--cell-m", type=float, default=100.0)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "resplit_pano_grouped_seed86.csv"))
    p.set_defaults(func=cmd_resplit)

    p = sub.add_parser("prep", help="decode the training crops once into a uint8 .npy memmap (CPU)")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--split-csv", default=None)
    p.add_argument("--images", required=True, nargs="+", help="prepared crop dir(s), searched in order")
    p.add_argument("--workers", type=int, default=1, help="decode processes (row order is kept)")
    p.add_argument("--out", required=True, help="the .npy to write; its sidecar is <out>.meta.json")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("train", help="the tagger's DINOv2 recipe on a split (GPU)")
    tagger(p)
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--split-csv", default=None)
    p.add_argument("--images", nargs="+", default=None,
                   help="prepared crop dir(s), searched in order (decoded in memory; the default path)")
    p.add_argument("--prep", default=None, help="a `prep` .npy to map instead of decoding --images")
    p.add_argument("--verify-prep", action="store_true", help="re-hash the --prep array before training")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out-dir>/checkpoint.pth if present (else start fresh)")
    p.add_argument("--loss", default="bce", choices=TagLoss.KINDS,
                   help="bce (the recipe), nnpu (Kiryo et al. 2017 non-negative PU risk) or soft")
    p.add_argument("--mask-csv", default=None, help="label_uid + one 0/1 column per masked tag (0 = no loss)")
    p.add_argument("--prior", default=None, help="JSON {tag: pi_t}, every tag; needed by nnpu and soft")
    p.add_argument("--nnpu-beta", type=float, default=0.0, help="nnPU: ascend when the bracket < -beta")
    p.add_argument("--nnpu-gamma", type=float, default=1.0, help="nnPU: gradient-ascent step scale")
    p.add_argument("--backbone", required=True, help="dinov2_vitb14_reg4_pretrain.pth")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("test-only", help="keep a predictions file's test rows, byte for byte, + meta")
    p.add_argument("--pred", required=True, help="predictions over all 10,857 crops")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--split-csv", default=None, help="the arm's split (default: the HF split)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_test_only)

    p = sub.add_parser("collect", help="CPU side of the retrain arms: filter, score, usage rows")
    p.add_argument("--work", required=True, help="the runbook's $WORK (holds train_<arm>/ and snap_*)")
    p.add_argument("--arms", nargs="+", default=list(ARM_SPLITS), choices=list(ARM_SPLITS))
    p.add_argument("--epochs", default=",".join(map(str, SNAPSHOT_EPOCHS)),
                   help="snapshot epoch indices to collect ('' = none)")
    p.add_argument("--final", action="store_true", help="also the finished run's best.pth")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--log", default=os.path.join(REPO, "analysis_out", "usage_log.jsonl"))
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("log-usage", help="append a paid:false GPU-time row to the usage ledger")
    p.add_argument("--label", required=True, help="e.g. tagger-eval, infer-released, train-control")
    p.add_argument("--elapsed-s", type=float, required=True)
    p.add_argument("--status", default="ok", choices=["ok", "failed", "killed", "in_progress"],
                   help="in_progress: elapsed so far for a run still going at the time of writing; "
                        "give it a --run-id, and the final row with the same --run-id replaces it in "
                        "every total (rampnet.ledger.latest_rows)")
    p.add_argument("--run-id", default=None,
                   help="supersede key, e.g. tagger-86:train-control:2026-09-23T01:58:54+00:00")
    p.add_argument("--concurrent-with", default=None,
                   help="comma list of our other runs sharing the GPU ('' = none); sets gpu_share")
    p.add_argument("--what", required=True, help="one line: what ran")
    p.add_argument("--n", type=int, default=None, help="crops scored / trained on")
    p.add_argument("--ts", default=None, help="run start (UTC ISO); default now")
    p.add_argument("--host", default="makelab2.cs.washington.edu", help="where it ran (the #86 benchmark: makelab2)")
    p.add_argument("--gpu", default="NVIDIA A40", help="the GPU it ran on ('' = none)")
    p.add_argument("--log", default=os.path.join(REPO, "analysis_out", "usage_log.jsonl"))
    p.set_defaults(func=cmd_log_usage)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
