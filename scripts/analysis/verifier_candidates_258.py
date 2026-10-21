"""Every RampNet peak a second-pass verifier would judge, labelled TP / FP / ignored (#258).

The two-pass idea in #258: lower RampNet's threshold to admit more candidates, then score
each candidate with a crop-level model (first: the production CurbRamp validator from
sidewalk-validator-ai) and drop the false positives. This script builds the candidate set
that every verifier arm is scored on, so the arms are compared on identical rows.

**Candidates** are all of RampNet's stored peaks from the committed low-floor caches
(``analysis_out/op_cache/<split>.json``, score floor 0.05), not only the sub-threshold
ones: a two-pass pipeline filters everything above its lowered threshold, including peaks
at or above 0.55.

**Labels.** Each peak is labelled by the shared matcher exactly as ``score_pano`` scores
the full floor set: peaks sorted by score, ``rampnet.metrics.greedy_match`` with
``wrap_x=True``, then the ignore-point fallback for unmatched peaks. Because greedy
matching runs in score order, applying any threshold keeps a prefix of that order and
leaves every kept peak's assignment unchanged, so **one label per peak holds at every
threshold**. ``main`` asserts this against ``score_pano`` at 0.05, 0.30 and 0.55 on every
split before writing.

* ``tp`` -- claims a GT ramp no higher-scored peak claimed.
* ``fp`` -- claims nothing and is not near an ignore point. ``redundant`` is true when a
  GT ramp was within R but already claimed by a higher-scored peak: a second hit on a
  real ramp. A verifier that says "ramp" there is right about the image and still costs
  an FP, so verifier AUC is reported with and without these rows.
* ``ignored`` -- unmatched but within R of an ignore point; neither TP nor FP.

**Caveat that travels with every number from these rows (#258 issue body).** On verdict
splits the GT is anchored to reviewed RampNet detections exported at a 0.55 floor, so a
sub-threshold peak counts as TP only if the reviewer added that ramp as a missed mark. A
real ramp the reviewer did not mark is labelled ``fp`` here, so verifier precision
measured on these labels is a lower bound. ``manual_gold`` (independent YOLO-format
labels) is the split without this bias.

``group`` is the review stratum from ``verdicts.json`` (``top`` / ``random`` / ``empty``),
null on ``manual_gold``; the unbiased subset is ``group != "top"``, as in
``open_rig_misses_258.py``. ``source`` and ``camera_model`` come from ``records.jsonl``.

    python scripts/analysis/verifier_candidates_258.py           # write
    python scripts/analysis/verifier_candidates_258.py --check   # re-derive, compare bytes

CPU only, no network, no imagery. Writes ``analysis_out/verifier_258/candidates/<split>.json``
and ``analysis_out/verifier_258/candidates/manifest.json`` (counts and a sha256 per file).
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import (PANO_SCALE_X, PANO_SCALE_Y,  # noqa: E402
                                    radius_sq_for, score_pano)
from rampnet.eval import PINNED_SPLITS  # noqa: E402
from rampnet.geometry import dist_sq  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

from operating_point_curve import read_cache  # noqa: E402

CACHE_DIR = os.path.join(REPO, "analysis_out", "op_cache")
OUT_DIR = os.path.join(OUT, "verifier_258", "candidates")
CHECK_THRESHOLDS = (0.05, 0.30, 0.55)
ROUND = 6


def label_peaks(preds, gt, radius_sq):
    """``[(x, y, score, label, redundant), ...]`` in descending score order.

    Same order, matcher and ignore fallback as ``rampnet.detection_eval.score_pano``.
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    peaks = [preds[i] for i in order]
    assignments = greedy_match([(x, y) for x, y, _ in peaks], gt.gt_points, radius_sq,
                               PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    out = []
    for (x, y, s), (gt_index, saw_in_range) in zip(peaks, assignments):
        if gt_index >= 0:
            out.append((x, y, s, "tp", False))
        elif any(dist_sq(x, y, ix, iy, PANO_SCALE_X, PANO_SCALE_Y, True) < radius_sq
                 for ix, iy in gt.ignore_points):
            out.append((x, y, s, "ignored", False))
        else:
            out.append((x, y, s, "fp", bool(saw_in_range)))
    return out


def check_against_score_pano(preds, labelled, gt, radius_sq):
    """Raise if the per-peak labels disagree with ``score_pano`` at any check threshold."""
    for t in CHECK_THRESHOLDS:
        ps = score_pano([p for p in preds if p[2] >= t], gt, radius_sq)
        kept = [r for r in labelled if r[2] >= t]
        mine = (sum(r[3] == "tp" for r in kept), sum(r[3] == "fp" for r in kept),
                sum(r[3] == "ignored" for r in kept))
        if mine != (ps.tp, ps.fp, ps.ignored):
            raise AssertionError(f"labels {mine} != score_pano {(ps.tp, ps.fp, ps.ignored)} "
                                 f"at threshold {t}")


def pano_meta(split):
    """``{pano_id: (source, camera_model)}`` from ``records.jsonl``."""
    out = {}
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["pano"]
            out[p["panorama_id"]] = (p.get("source") or "gsv", p.get("camera_model"))
    return out


def pano_groups(split):
    """``{pano_id: group}`` from ``verdicts.json``; empty for a split without one."""
    path = os.path.join(REPO, "benchmark", split, "verdicts.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {pid: e.get("group") for pid, e in json.load(f)["panos"].items()}


def build_split(split):
    panos, meta = read_cache(os.path.join(CACHE_DIR, f"{split}.json"))
    radius_sq = radius_sq_for(meta["radius_normalized"])
    records, groups = pano_meta(split), pano_groups(split)
    rows, n_gt = [], 0
    for pd in panos:
        pid, gt = pd["pano"], pd["gt"]
        labelled = label_peaks(pd["preds"], gt, radius_sq)
        check_against_score_pano(pd["preds"], labelled, gt, radius_sq)
        n_gt += len(gt.gt_points)
        source, camera = records[pid]
        for x, y, s, label, redundant in labelled:
            rows.append({"pano": pid, "x": x, "y": y, "score": round(s, ROUND),
                         "label": label, "redundant": redundant,
                         "group": groups.get(pid), "source": source,
                         "camera_model": camera})
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in ("tp", "fp", "ignored")}
    counts["redundant_fp"] = sum(r["redundant"] for r in rows)
    payload = {"split": split,
               "op_cache": {k: meta.get(k) for k in ("model", "score_floor", "min_distance",
                                                 "radius_normalized", "fp16", "tta")},
               "n_panos": len(panos), "n_gt": n_gt, "n_candidates": len(rows),
               "counts": counts, "candidates": rows}
    return payload


def dumps(obj):
    """Sorted keys, one candidate per line: small, diffable and byte-stable."""
    if "candidates" not in obj:
        return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    head = {k: v for k, v in obj.items() if k != "candidates"}
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
             for r in obj["candidates"]]
    body = json.dumps(head, indent=1, sort_keys=True, ensure_ascii=False)[:-2]
    return body + ',\n "candidates": [\n  ' + ",\n  ".join(lines) + "\n ]\n}\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="re-derive every file and compare bytes; write nothing")
    ap.add_argument("--splits", nargs="*", default=list(PINNED_SPLITS))
    args = ap.parse_args(argv)

    files, manifest = {}, {"splits": {}}
    for split in args.splits:
        payload = build_split(split)
        text = dumps(payload)
        files[f"{split}.json"] = text
        manifest["splits"][split] = {
            "n_panos": payload["n_panos"], "n_gt": payload["n_gt"],
            "n_candidates": payload["n_candidates"], "counts": payload["counts"],
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    files["manifest.json"] = dumps(manifest)

    if args.check:
        bad = []
        for name, text in files.items():
            path = os.path.join(OUT_DIR, name)
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    if f.read() != text:
                        bad.append(name)
            except OSError:
                bad.append(name)
        if bad:
            print("stale or missing: " + ", ".join(bad))
            return 1
        print(f"current: {len(files)} files")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    for split, m in manifest["splits"].items():
        c = m["counts"]
        print(f"{split:20s} panos {m['n_panos']:4d}  gt {m['n_gt']:4d}  "
              f"tp {c['tp']:4d}  fp {c['fp']:4d} (redundant {c['redundant_fp']:3d})  "
              f"ignored {c['ignored']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
