"""Context experiment for curb-ramp tags (#86, RampNet 2.0 plan item 4).

Does a tag model do better when it sees more of the street? The ASSETS'24 tagger (the #86
benchmark of record, ``tag_benchmark_86.py``) trains on a 640 px box cut from a 1440x960
viewport screenshot, so what it sees depends on the labeler's zoom: about 48 deg of
horizontal field of view at zoom 1, 25 deg at zoom 2, 12.5 deg at zoom 3
(``docs/context_fov_86.md`` section 2). This re-cuts the same labels from the makelab2 pano
store (``scripts/crop_cutter.py``, plan item 2b) at fixed, label-centred fields of view and
trains the same recipe on each, on the same labels, scored on the same test rows.

Arms (all gnomonic, rendered with the viewer's tilt, the cutter's default ``--tilt mm``):

  viewport  the labeler's own view re-cut from the store (1440x960), then the tagger's
            640 px box around the canvas point: the benchmark's framing from the archive
            instead of the screenshot. Separates "re-cut from the store" from "different
            field of view".
  fov25     label-centred, 25 deg horizontal, 640x640 (about the zoom-2 control)
  fov50     label-centred, 50 deg (about the zoom-1 control)
  fov90     label-centred, 90 deg (wider than any control)

Subcommands (CPU; the GPU work is ``tag_benchmark_86.py train`` / ``infer``, sequenced by
``context_fov_86.sh`` and ``context_fov_86.slurm``):

  cut-input   the HF label rows of ``docs/data/crop_cutter/coverage_input.csv`` as the
              cutter's ``--labels`` file (``city, label_id`` plus the rawLabels geometry)
  labels      from the cutter manifests, one labels CSV per arm (the benchmark's label table
              with ``filename`` pointing at the arm's crop), restricted to the labels every
              arm has, plus ``split_common.csv`` so every arm, and the #178 control, is
              scored on the same test rows
  crop640     the tagger's crop.py box (640 px around the canvas point) over the viewport
              arm's crops, in place, once
  report      one table over the arms' score files and the re-scored control
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import tag_benchmark_86 as tb  # noqa: E402  (same directory)

OUT_DIR = os.path.join(REPO, "analysis_out", "context_fov_86")
COVERAGE_INPUT = os.path.join(REPO, "docs", "data", "crop_cutter", "coverage_input.csv")
ARMS = ("viewport", "fov25", "fov50", "fov90")
CROP_BOX = 320  # the tagger's crop.py: a 640 px box, +-320 around the label point
VIEWPORT_SIZE = (1440, 960)  # the cutter's default viewport crop, the HF crop's size


def crop_name(city, label_id, tag):
    """``scripts/crop_cutter.py`` naming: ``<city>__<label_id>__<tag>.jpg``."""
    return f"{city}__{int(label_id)}__{tag}.jpg"


# ----------------------------------------------------------------------------- cut-input

def cmd_cut_input(a):
    d = pd.read_csv(a.coverage_input)
    h = d[d["set"] == "hf_validated"].drop(columns=["set", "order"]).reset_index(drop=True)
    dup = h.duplicated(["city", "label_id"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate (city, label_id) rows in {a.coverage_input}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tb.write_csv(h, a.out)
    print(f"{len(h)} labels -> {a.out}; splits {h.hf_split.value_counts().to_dict()}; "
          f"cities {h.city.nunique()}")


# ----------------------------------------------------------------------------- labels

def read_manifests(paths):
    """The cutter's manifest JSONL: the latest row per crop name wins; a truncated line (a
    killed run) is skipped, as the cutter itself does."""
    latest = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[r["name"]] = r
    return latest


def arm_table(lab, arm, manifest, images=None):
    """(crop names, ok mask, status per label) for one arm over the label table."""
    names = [crop_name(c, l, arm) for c, l in zip(lab.city, lab.label_id)]
    status = []
    for n in names:
        r = manifest.get(n)
        s = r["status"] if r else "absent"
        if s == "ok" and images is not None and not os.path.isfile(os.path.join(images, n)):
            s = "file_missing"
        status.append(s)
    ok = [s == "ok" for s in status]
    return names, ok, status


def build_labels(lab, manifest, arms, images=None):
    """Per-arm label tables on the labels every arm has, and the common split."""
    per = {arm: arm_table(lab, arm, manifest, images) for arm in arms}
    common = pd.Series([all(per[arm][1][i] for arm in arms) for i in range(len(lab))], index=lab.index)
    tables = {}
    for arm in arms:
        t = lab[common].copy()
        t["filename"] = [n for n, keep in zip(per[arm][0], common) if keep]
        tables[arm] = t.reset_index(drop=True)
    split = lab.loc[common, ["label_uid", "split"]].reset_index(drop=True)
    dropped = lab.loc[~common, ["label_uid", "filename", "split"]].copy()
    for arm in arms:
        dropped[f"status_{arm}"] = [s for s, keep in zip(per[arm][2], common) if not keep]
    summary = {
        "n_labels": int(len(lab)), "n_common": int(common.sum()),
        "common_by_split": lab.loc[common, "split"].value_counts().sort_index().to_dict(),
        "per_arm_ok": {arm: int(sum(per[arm][1])) for arm in arms},
        "dropped_status": {arm: pd.Series(per[arm][2])[~common.to_numpy()].value_counts().to_dict()
                           for arm in arms},
    }
    return tables, split, dropped.reset_index(drop=True), summary


def cmd_labels(a):
    lab = pd.read_csv(a.labels)
    manifest = read_manifests(a.manifest)
    tables, split, dropped, summary = build_labels(lab, manifest, a.arms, a.images)
    os.makedirs(a.out_dir, exist_ok=True)
    for arm, t in tables.items():
        tb.write_csv(t, os.path.join(a.out_dir, f"labels_{arm}.csv"))
    tb.write_csv(split, os.path.join(a.out_dir, "split_common.csv"))
    tb.write_csv(dropped, os.path.join(a.out_dir, "dropped_labels.csv"))
    summary["labels_sha256"] = tb.sha256_file(a.labels)
    summary["manifests"] = [os.path.basename(p) for p in a.manifest]
    summary["ts"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tb.write_json(summary, os.path.join(a.out_dir, "labels_summary.json"))
    print(json.dumps(summary, indent=1))


# ----------------------------------------------------------------------------- crop640

def crop_box(x_norm, y_norm, width, height, half=CROP_BOX):
    """The tagger's crop.py box, exactly: denormalise to int, then +-half clamped to the
    image. (left, top, right, bottom)."""
    x = int(x_norm * width)
    y = int(y_norm * height)
    return max(0, x - half), max(0, y - half), min(width, x + half), min(height, y + half)


def cmd_crop640(a):
    from PIL import Image
    lab = pd.read_csv(a.labels)
    marker = os.path.join(a.images, ".cropped_viewport")
    if os.path.exists(marker):
        print(f"already cropped ({marker}); not cropping twice")
        return
    # Refuse a partial state, as tag_benchmark_86.py prepare does: a crop that is not the
    # viewport size was cropped by an interrupted run and must not be cropped again.
    bad = tb.uncropped_violations(a.images, list(lab.filename), size=VIEWPORT_SIZE)
    if bad:
        raise SystemExit(f"{len(bad)} viewport crops are not {VIEWPORT_SIZE[0]}x{VIEWPORT_SIZE[1]} "
                         f"(e.g. {bad[:3]}) but {marker} is absent: an earlier crop640 was "
                         f"interrupted. Re-cut the viewport arm and run again.")
    n = 0
    for fn, xn, yn in zip(lab.filename, lab.normalized_x, lab.normalized_y):
        p = os.path.join(a.images, fn)
        with Image.open(p) as im:
            box = crop_box(float(xn), float(yn), im.width, im.height)
            out = im.crop(box)
            out.load()
        # crop.py saves in place with PIL's defaults (quality 75 for JPEG; the HF originals
        # are PNG, so there it is lossless). These are JPEGs, so save at the cutter's quality
        # to keep the second encode comparable to the single encode of the fov arms.
        out.save(p, quality=a.quality)
        n += 1
    with open(marker, "w") as fh:
        fh.write(dt.datetime.now(dt.timezone.utc).isoformat())
    print(f"cropped {n} viewport crops in place to {2 * CROP_BOX} px boxes (quality {a.quality})")


# ----------------------------------------------------------------------------- report

def score_row(name, path, tags):
    if not os.path.exists(path):
        return {"arm": name, "status": "not run", "file": os.path.basename(path)}
    with open(path) as fh:
        s = json.load(fh)
    full = s["subsets"]["full"]
    lf = s["subsets"].get("leak_free")
    ci = (full.get("ci95_fixed_tags") or {}).get("mAP")
    row = {"arm": name, "status": "ok", "file": os.path.basename(path), "n": full["n"],
           "mAP": full["fixed_tags"]["mAP"], "mAP_ci95": ci,
           "micro_f1": full["fixed_tags"]["micro_f1"], "macro_f1": full["fixed_tags"]["macro_f1"],
           "leak_free_mAP": lf["fixed_tags"]["mAP"] if lf else None,
           "per_tag_ap": {t: (full["per_tag"].get(t) or {}).get("ap") for t in tags},
           "pred_sha256": s.get("pred_sha256")}
    return row


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def cmd_report(a):
    tags = list(tb.FIXED_TAGS)
    rows = [score_row("control (#178, HF crops)", a.control_scores, tags)]
    for arm in a.arms:
        rows.append(score_row(arm, os.path.join(a.out_dir, f"train_{arm}_final_scores.json"), tags))
    out = {"tags": tags, "rows": rows, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    tb.write_json(out, os.path.join(a.out_dir, "summary.json"))
    lines = ["| arm | n test | mAP | 95% CI | micro-F1 | macro-F1 | leak-free mAP | " + " | ".join(tags) + " |",
             "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(tags)]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['arm']} | not run |" + " |" * (5 + len(tags)))
            continue
        ci = r["mAP_ci95"]
        lines.append(f"| {r['arm']} | {r['n']} | {_fmt(r['mAP'])} | "
                     + (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a")
                     + f" | {_fmt(r['micro_f1'])} | {_fmt(r['macro_f1'])} | {_fmt(r['leak_free_mAP'])} | "
                     + " | ".join(_fmt(r["per_tag_ap"][t]) for t in tags) + " |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(a.out_dir, "summary.md"), "w", encoding="utf-8", newline="") as fh:
        fh.write(md)
    print(md)


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cut-input", help="the HF label rows of coverage_input.csv as the cutter's --labels")
    p.add_argument("--coverage-input", default=COVERAGE_INPUT)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "cut_input.csv"))
    p.set_defaults(fn=cmd_cut_input)

    p = sub.add_parser("labels", help="per-arm label tables on the labels every arm has")
    p.add_argument("--labels", default=os.path.join(tb.OUT_DIR, "hf_curbramp_labels.csv"))
    p.add_argument("--manifest", nargs="+", required=True, help="cutter manifest JSONL file(s)")
    p.add_argument("--images", default=None, help="crop dir; when given, a crop must also exist on disk")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--out-dir", default=OUT_DIR)
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("crop640", help="the tagger's 640 px box over the viewport arm, in place, once")
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "labels_viewport.csv"))
    p.add_argument("--images", required=True)
    p.add_argument("--quality", type=int, default=92, help="JPEG quality (the cutter's default)")
    p.set_defaults(fn=cmd_crop640)

    p = sub.add_parser("report", help="one table over the arms and the re-scored control")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--control-scores", default=os.path.join(OUT_DIR, "control_scores.json"))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
