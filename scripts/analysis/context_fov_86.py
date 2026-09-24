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
    rows = [score_row(a.control_label, a.control_scores, tags)]
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


def _test_side(pred_path, labels_path, split_csv, tags):
    """One arm's test rows: (label_uid-indexed frame with the tag columns and the arm's
    sigmoid scores, ordered by label_uid). Each arm's predictions name ITS crops
    (``<city>__<id>__<arm>.jpg``, the HF ``.png`` for the control), so the join to the other
    arm is on ``label_uid``, never on filename."""
    lab, arm_tags = tb.load_labels(labels_path, split_csv)
    if list(arm_tags) != list(tags):
        raise SystemExit(f"{labels_path}: tag columns differ from the reference's")
    te = tb.leak_table(lab, "test", "train")
    m = te.merge(pd.read_csv(pred_path), on="filename", how="inner", validate="one_to_one")
    return m.sort_values("label_uid").reset_index(drop=True)


def paired_contrast(pred_a, labels_a, pred_b, labels_b, split_csv, tags_fixed=None, n_boot=1000, seed=86):
    """arm A minus arm B on the SAME test rows and the SAME pano-clustered bootstrap draws.

    Every arm here is scored on the common test rows (split_common.csv), so two arms' CIs
    overlap for a reason that has nothing to do with the arms: they share every panorama.
    The right null for "is A better than B" resamples panoramas once and scores both arms
    on that resample. Returns the point difference and the 95% CI for mAP / micro-F1 /
    macro-F1 and per-tag AP, on the benchmark's fixed tag set."""
    import numpy as np
    _, tags = tb.load_labels(labels_b, split_csv)
    a = _test_side(pred_a, labels_a, split_csv, tags)
    b = _test_side(pred_b, labels_b, split_csv, tags)
    if list(a.label_uid) != list(b.label_uid):
        raise SystemExit("the two prediction files do not cover the same test rows")
    y = a[tags].to_numpy(float)
    if not np.array_equal(y, b[tags].to_numpy(float)):
        raise SystemExit("the two label tables disagree on the test rows' tags")
    sa, sb = tb.score_probs(a, tags), tb.score_probs(b, tags)
    fixed = list(tags_fixed or tb.FIXED_TAGS)
    groups = a.pano_id.fillna(a.label_uid).to_numpy()
    pa, pb = (tb.tagger_metrics(y, s, tags, selected=fixed) for s in (sa, sb))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    keys = ("mAP", "micro_f1", "macro_f1")
    draws = {k: [] for k in keys}
    per_tag = {t: [] for t in fixed}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        ra = tb.tagger_metrics(y[idx], sa[idx], tags, selected=fixed)
        rb = tb.tagger_metrics(y[idx], sb[idx], tags, selected=fixed)
        for k in keys:
            ok = ra[k] is not None and rb[k] is not None
            draws[k].append(ra[k] - rb[k] if ok else np.nan)
        for t in fixed:
            xa, xb = ra["per_tag"][t]["ap"], rb["per_tag"][t]["ap"]
            per_tag[t].append(xa - xb if xa is not None and xb is not None else np.nan)

    def _pt(t):
        xa, xb = pa["per_tag"][t]["ap"], pb["per_tag"][t]["ap"]
        return None if xa is None or xb is None else xa - xb

    return {"n": int(len(a)), "n_panos": int(len(uniq)), "n_boot": n_boot, "seed": seed, "tags_fixed": fixed,
            "a": os.path.basename(pred_a), "b": os.path.basename(pred_b),
            "a_sha256": tb.sha256_file(pred_a), "b_sha256": tb.sha256_file(pred_b),
            "a_mAP": pa["mAP"], "b_mAP": pb["mAP"],
            "a_minus_b": {k: {"point": pa[k] - pb[k], "ci95": tb._ci(draws[k])} for k in keys},
            "per_tag_ap_a_minus_b": {t: {"point": _pt(t), "ci95": tb._ci(per_tag[t])} for t in fixed}}


def cmd_contrast(a):
    rows = []
    for arm in a.arms:
        pred = os.path.join(a.out_dir, f"train_{arm}_final_test_predictions.csv")
        labels = os.path.join(a.out_dir, f"labels_{arm}.csv")
        if not os.path.exists(pred):
            print(f"{arm}: not run ({os.path.basename(pred)} missing)")
            continue
        c = paired_contrast(pred, labels, a.reference_pred, a.reference_labels, a.split_csv, n_boot=a.n_boot)
        c["arm"] = arm
        c["reference"] = a.reference
        rows.append(c)
        d = c["a_minus_b"]["mAP"]
        print(f"{arm} minus {a.reference}: mAP {d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    tb.write_json({"reference": a.reference, "reference_pred": os.path.basename(a.reference_pred),
                   "reference_labels": os.path.basename(a.reference_labels), "rows": rows,
                   "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, a.out)


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
    p.add_argument("--control-label", default="control (#178, HF crops)",
                   help="the control row's name in the table; say so here when the score file is not "
                        "the 100-epoch benchmark control (e.g. an interim snapshot)")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("contrast", help="each arm minus a reference, paired on the same test rows and draws")
    p.add_argument("--reference", required=True, help="name of the reference arm in the output, e.g. control")
    p.add_argument("--reference-pred", required=True, help="the reference's test predictions file")
    p.add_argument("--reference-labels", default=os.path.join(tb.OUT_DIR, "hf_curbramp_labels.csv"),
                   help="the label table the reference was scored with (an arm: its labels_<arm>.csv)")
    p.add_argument("--split-csv", default=os.path.join(OUT_DIR, "split_common.csv"))
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_contrast)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
