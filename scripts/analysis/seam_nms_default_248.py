#!/usr/bin/env python
"""Should seam-wrapped NMS become the default? A CPU study from committed peaks (#248).

``rampnet.subcell.detect_peaks(wrap_nms=True)`` (PR #249, default off) suppresses peaks across
the 360 deg seam. This script measures what turning it on would remove, from peaks of the
published model that are already committed, and applies the rule pre-registered in
``docs/seam_nms_default_248.md`` section 4.

Pools (the published model, floor 0.05, ``min_distance`` 10, ``exclude_border=False``):

- ``manual_gold_tta``: ``benchmark/manual_gold/records.jsonl`` (flip TTA; scores clipped).
- ``cities13``: ``analysis_out/input_res_sweep_25/cache/r2048/<split>.json`` (11 splits, single
  pass, raw scores) plus ``analysis_out/op_cache/{bayonne,juneau}.json``. ``cities11`` = the
  r2048 splits alone.

From stored peaks two removal sets are defined (doc section 1). A **strip peak** has heatmap
column < 10 or >= 1014 on the 1024 x 512 grid. Its partner is the nearest stronger stored peak
across the seam (clipped score strictly greater; tied partners within 20 px must not exist):

- ``C`` (certain): partner within wrapped Chebyshev 10 px. Exact ``wrap_nms`` removes it.
- ``E`` (envelope only): not in ``C``, partner within 20 px. A heuristic candidate; whether
  wrap removes it needs the heatmap. The 30-px envelope is a sensitivity count.

Subcommands, in the order the study ran them::

    python scripts/analysis/seam_nms_default_248.py bounds          # counts -> bounds.json
    python scripts/analysis/seam_nms_default_248.py sheet --panos-root D:/Git/RampNet
    # ... rate review/index.html (or the card images), save review/verdicts__<rater>.json ...
    python scripts/analysis/seam_nms_default_248.py calibrate       # AI vs Jon's 14 pairs
    python scripts/analysis/seam_nms_default_248.py report          # results.json / .md
    python scripts/analysis/seam_nms_default_248.py figure          # docs/figures/...png
    python scripts/analysis/seam_nms_default_248.py --check         # re-derive + compare

``bounds``, ``calibrate``, ``report``, ``figure`` and ``--check`` read only committed files and
need no GPU, network or panoramas. ``sheet`` reads the local panoramas (``--panos-root``, a
checkout holding ``benchmark/<split>/panos``) to cut the card images; the images and the HTML are
git-ignored, and ``items.json`` carries each image's sha256 so a regenerated deck can be compared.

Example (the core set builder)::

    >>> peaks = [(0.999, 0.5, 0.9), (0.003, 0.5, 0.6), (0.5, 0.5, 0.8)]
    >>> [(r["weak"], r["set"]) for r in removal_candidates(peaks)]
    [(1, 'C')]
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

OUT_DIR = os.path.join(REPO, "analysis_out", "seam_nms_default_248")
REVIEW_DIR = os.path.join(OUT_DIR, "review")
R2048_DIR = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "cache", "r2048")
OP_CACHE = os.path.join(REPO, "analysis_out", "op_cache")
OP_CACHE_TTA = os.path.join(REPO, "analysis_out", "op_cache_tta")
MG_RECORDS = os.path.join(REPO, "benchmark", "manual_gold", "records.jsonl")
MG_META = os.path.join(REPO, "benchmark", "manual_gold", "detections_meta.json")
SEAM_VERDICTS = os.path.join(REPO, "benchmark", "manual_gold", "seam_verdicts__jon.json")
LABELS_DIR = os.path.join(REPO, "manual_labels")
FIGURE = os.path.join(REPO, "docs", "figures", "seam_nms_default_248.png")

#: The #25 sweep's r2048 splits (input_res_sweep_25.SPLITS) and the two post-fix op_caches.
SPLITS11 = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville", "laurens_gsv",
            "laurens_mapillary", "morgantown", "paterson", "richmond", "sao_paulo")
OP_SPLITS = ("bayonne", "juneau")
SPLITS13 = SPLITS11 + OP_SPLITS
MG = "manual_gold"

W, H = 1024, 512          # heatmap grid the peaks live on
D = 10                    # min_distance
ENV = 20                  # heuristic envelope (2d)
ENV30 = 30                # sensitivity
THRESHOLDS = (0.05, 0.30, 0.55)
RULE_T = 0.30
SEED = 248
N_REPS = 2000
ND = 4
JOIN_PX = 6.5             # verdict inheritance radius (wrapped Chebyshev, heatmap px)
AI_RATER = "claude-opus-5-5"

#: Literals the plan reviewer counted independently (doc section 2). ``--check`` asserts them,
#: so the check is not purely self-referential. key -> (value at 0.30, value at 0.55).
REVIEWER_COUNTS = {
    "cities11": {"peaks": (2880, 2257), "strip": (36, 16), "pairs": (7, 4), "C": (7, 4),
                 "env20": (12, 4), "env30": (13, 4)},
    MG: {"peaks": (3981, 3610), "strip": (38, 23), "pairs": (8, 6), "C": (8, 6),
         "env20": (14, 9), "env30": (15, 10)},
}

# The rubric the cards are rated under. Version bumps on any wording change that could move a
# verdict; the export carries both, so passes under different rubrics are never averaged.
RUBRIC_VERSION = 1
RUBRIC = [
    ("Pair: is A the same physical object as B?",
     "same = one object marked twice; different = two distinct objects; unsure = cannot tell. "
     "The crops are rolled so the panorama seam is at the centre; the imagery there is "
     "continuous, so judge the world, not the projection."),
    ("Each mark: is it a curb ramp?",
     "ramp / not_ramp / unsure, for the object under the mark."),
    ("A driveway is NOT a curb ramp", "Driveway aprons are excluded however ramp-like they look."),
    ("Mark only what you can SEE", "Never call a ramp because a crossing implies one."),
    ("Unsure is a real answer",
     "An unsure verdict sends the card to a human; a coerced one is noise."),
]


# --------------------------------------------------------------------------- #
# pure helpers (torch-free, stdlib + numpy; unit-tested)
# --------------------------------------------------------------------------- #
def to_px(p):
    """(x, y, score) normalized -> (col, row) on the 1024 x 512 grid."""
    return int(round(p[0] * W)), int(round(p[1] * H))


def clipped(s):
    """The score the extractor ranked on: peaks are found on ``clip(h, 0, 1)``."""
    return min(max(float(s), 0.0), 1.0)


def in_strip(col, d=D, width=W):
    """True when a peak's (2d+1) window crosses the seam."""
    return col < d or col >= width - d


def wrapped_cheb(a, b, width=W):
    """Wrapped Chebyshev distance between two (col, row) pixels."""
    dc = abs(a[0] - b[0]) % width
    return max(min(dc, width - dc), abs(a[1] - b[1]))


def across_seam(a, b, width=W):
    return abs(a[0] - b[0]) > width // 2


def removal_candidates(peaks, d=D, env=ENV30):
    """Strip peaks with a stronger stored peak across the seam within ``env`` px.

    ``peaks``: list of (x, y, score) for one panorama (any score convention; ordering uses the
    clipped score). Returns one dict per candidate, ``weak`` / ``strong`` being indices into
    ``peaks``: ``dist`` (wrapped Chebyshev px to the nearest stronger across-seam peak, ties in
    distance broken by the higher score) and ``set``: ``"C"`` (<= d), ``"E"`` (<= 2d), or
    ``"E30"`` (<= env). ``ties`` lists the distances of across-seam partners within ``env``
    with an equal clipped score (the stored-peak argument assumes none within 2d).
    """
    px = [to_px(p) for p in peaks]
    cs = [clipped(p[2]) for p in peaks]
    out = []
    for i, (pi, si) in enumerate(zip(px, cs)):
        if not in_strip(pi[0], d):
            continue
        best, ties = None, []
        for j, (pj, sj) in enumerate(zip(px, cs)):
            if j == i or not across_seam(pi, pj):
                continue
            dist = wrapped_cheb(pi, pj)
            if dist > env:
                continue
            if sj == si:
                ties.append(dist)
            if sj > si and (best is None or (dist, -sj) < (best[0], -cs[best[1]])):
                best = (dist, j)
        if best is not None or ties:
            entry = {"weak": i, "strong": None if best is None else best[1],
                     "dist": None if best is None else best[0], "ties": ties}
            if best is not None:
                entry["set"] = "C" if best[0] <= d else ("E" if best[0] <= 2 * d else "E30")
            else:
                entry["set"] = None
            out.append(entry)
    return [e for e in out if e["set"] is not None] + \
        [e for e in out if e["set"] is None]


def straddling_pairs(peaks, t, d=D):
    """Wrapped Chebyshev distances of every across-seam pair of strip peaks both >= t within d."""
    sel = [to_px(p) for p in peaks if p[2] >= t]
    sel = [q for q in sel if in_strip(q[0], d)]
    out = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if across_seam(sel[i], sel[j]) and wrapped_cheb(sel[i], sel[j]) <= d:
                out.append(wrapped_cheb(sel[i], sel[j]))
    return sorted(out)


def wrapped_midpoint(a, b):
    """Midpoint of two normalized points on the cylinder (x wraps at 1).

    >>> [round(v, 4) for v in wrapped_midpoint((0.002, 0.5), (0.996, 0.52))]
    [0.999, 0.51]
    """
    dx = ((b[0] - a[0] + 0.5) % 1.0) - 0.5
    return ((a[0] + dx / 2.0) % 1.0, (a[1] + b[1]) / 2.0)


def norm_cheb_px(a, b):
    """Wrapped Chebyshev distance between two normalized points, in heatmap px (unrounded)."""
    dx = abs(a[0] - b[0]) % 1.0
    return max(min(dx, 1.0 - dx) * W, abs(a[1] - b[1]) * H)


def deck_order(item_keys, seed=SEED):
    """Deterministic shuffle of the deck and the random A/B letter for each item.

    Returns [(key, a_is_first)] in deck order: ``a_is_first`` says whether the item's first
    mark is shown as A.
    """
    rng = random.Random(seed)
    keys = sorted(item_keys)
    rng.shuffle(keys)
    return [(k, rng.random() < 0.5) for k in keys]


def classify(pair, weak_tag, strong_tag):
    """Class of a card from its verdict (doc section 3).

    >>> classify("same", "ramp", "ramp"), classify("different", "ramp", "ramp")
    ('D', 'L')
    >>> classify("different", "not_ramp", "ramp"), classify("unsure", "ramp", "ramp")
    ('N', 'U')
    """
    if pair == "same":
        return "D" if strong_tag == "ramp" else "N"
    if pair == "different":
        return {"ramp": "L", "not_ramp": "N"}.get(weak_tag, "U")
    return "U"


def jon_class(verdict):
    """An inherited verdict of Jon's: one -> D, two -> L (both marks are GT ramps), else U."""
    return {"one": "D", "two": "L"}.get(verdict, "U")


# --------------------------------------------------------------------------- #
# loading committed peaks
# --------------------------------------------------------------------------- #
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_city(split):
    """[(pano, preds, gt_json)] for one city split, from its r2048 cache or post-fix op_cache."""
    path = os.path.join(R2048_DIR if split in SPLITS11 else OP_CACHE, f"{split}.json")
    payload = read_json(path)
    meta = payload.get("meta", {})
    want = {"score_floor": 0.05, "min_distance": 10, "model": "projectsidewalk/rampnet-model"}
    bad = {k: meta.get(k) for k, v in want.items() if meta.get(k) != v}
    if bad or meta.get("tta") not in (False, None) or meta.get("fp16"):
        raise SystemExit(f"{path}: meta differs from {want} / tta False / fp32: {bad}")
    if meta.get("gt") == "unreviewed":
        raise SystemExit(f"{path}: unreviewed cache")
    return [(p["pano"], [tuple(t) for t in p["preds"]], p["gt"]) for p in payload["panos"]]


def load_manual_gold():
    """[(pano, preds, None)] from manual_gold records.jsonl (flip TTA, clipped scores)."""
    meta = read_json(MG_META)
    if not (meta.get("tta") is True and meta.get("peak_floor") == 0.05
            and meta.get("peak_min_distance") == 10):
        raise SystemExit(f"{MG_META}: not the TTA / floor 0.05 / md 10 export: {meta}")
    out = []
    with open(MG_RECORDS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append((r["pano"]["panorama_id"], [(d["x_normalized"], d["y_normalized"], d["confidence"])
                                        for d in r["detections"]], None))
    return out


def load_pools():
    """{split: [(pano, preds, gt_json)]} for the 13 city splits and manual_gold."""
    data = {s: load_city(s) for s in SPLITS13}
    data[MG] = load_manual_gold()
    return data


def pool_members(pool):
    return {"cities11": SPLITS11, "cities13": SPLITS13, MG: (MG,)}[pool]


# --------------------------------------------------------------------------- #
# bounds (counts only)
# --------------------------------------------------------------------------- #
def candidates_table(data):
    """Every removal candidate as a flat row: split, pano, weak/strong peaks, dist, set."""
    rows = []
    ties = []
    for split, panos in data.items():
        for pano, preds, _ in panos:
            for c in removal_candidates(preds):
                ties += c["ties"]
                if c["set"] is None:
                    continue
                w, s = preds[c["weak"]], preds[c["strong"]]
                rows.append({"split": split, "pano": pano, "set": c["set"], "dist": c["dist"],
                             "weak": [round(w[0], 6), round(w[1], 6), round(w[2], 6)],
                             "strong": [round(s[0], 6), round(s[1], 6), round(s[2], 6)]})
    rows.sort(key=lambda r: (r["split"], r["pano"], -r["weak"][2]))
    return rows, ties


def count_split(panos, cands, t):
    preds_t = [[p for p in preds if p[2] >= t] for _, preds, _ in panos]
    strip = sum(in_strip(to_px(p)[0]) for ps in preds_t for p in ps)
    dists = [d for _, preds, _ in panos for d in straddling_pairs(preds, t)]
    ct = [c for c in cands if c["weak"][2] >= t]
    return {"panos": len(panos), "peaks": sum(len(ps) for ps in preds_t), "strip": strip,
            "pairs": len(dists), "pair_dists": sorted(dists),
            "C": sum(c["set"] == "C" for c in ct),
            "env20": sum(c["set"] in ("C", "E") for c in ct),
            "env30": len(ct),
            "E": sum(c["set"] == "E" for c in ct),
            "panos_touched": len({c["pano"] for c in ct if c["set"] in ("C", "E")}),
            "peaks_ge_1": sum(p[2] >= 1.0 for ps in preds_t for p in ps),
            "strip_ge_1": sum(p[2] >= 1.0 and in_strip(to_px(p)[0]) for ps in preds_t for p in ps)}


def sum_counts(rows):
    out = {}
    for r in rows:
        for k, v in r.items():
            if k == "pair_dists":
                out[k] = sorted(out.get(k, []) + v)
            else:
                out[k] = out.get(k, 0) + v
    return out


def audit_other_files():
    """Strip peaks in every file that could hold seam peaks (doc section 2's audit list)."""
    rows = []
    for name in sorted(os.listdir(os.path.join(REPO, "benchmark"))):
        rec = os.path.join(REPO, "benchmark", name, "records.jsonl")
        if name == MG or not os.path.exists(rec):
            continue
        peaks = []
        with open(rec, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    peaks.append([(d["x_normalized"], d["y_normalized"], d.get("confidence") or 0)
                                  for d in r.get("detections", [])])
        rows.append({"file": f"benchmark/{name}/records.jsonl",
                     "strip_floor": sum(in_strip(to_px(p)[0]) for ps in peaks for p in ps),
                     "pairs_floor": sum(len(straddling_pairs(ps, 0.0)) for ps in peaks)})
    for d, rel in ((OP_CACHE, "analysis_out/op_cache"), (OP_CACHE_TTA, "analysis_out/op_cache_tta")):
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            peaks = [[tuple(t) for t in p["preds"]] for p in read_json(os.path.join(d, name))["panos"]]
            row = {"file": f"{rel}/{name}"}
            for t, tag in ((0.05, "floor"), (0.30, "0.30")):
                row[f"strip_{tag}"] = sum(in_strip(to_px(p)[0]) for ps in peaks for p in ps
                                          if p[2] >= t)
                row[f"pairs_{tag}"] = sum(len(straddling_pairs(ps, t)) for ps in peaks)
            rows.append(row)
    return rows


def build_bounds():
    data = load_pools()
    cands, ties = candidates_table(data)
    per_split = {}
    for split, panos in data.items():
        cs = [c for c in cands if c["split"] == split]
        per_split[split] = {f"{t:.2f}": count_split(panos, cs, t) for t in THRESHOLDS}
    pools = {}
    for pool in ("cities11", "cities13", MG):
        pools[pool] = {f"{t:.2f}": sum_counts([per_split[s][f"{t:.2f}"]
                                               for s in pool_members(pool)]) for t in THRESHOLDS}
    max_pair = max([d for p in pools.values() for d in p["0.05"]["pair_dists"]], default=None)
    return {"what": "seam-strip removal candidates of wrapped NMS, from committed peaks (#248)",
            "definitions": {"strip": f"heatmap column < {D} or >= {W - D} on {W}x{H}",
                            "C": f"stronger stored peak across the seam within wrapped Chebyshev {D} px",
                            "E": f"not C, stronger across-seam peak within {ENV} px (heuristic)",
                            "env30": f"sensitivity: within {ENV30} px",
                            "stronger": "clipped score strictly greater"},
            "tie_distances_within_30px": sorted(ties),
            "max_pair_dist_px_at_floor": max_pair,
            "manual_gold_gt_points": None,   # filled below
            "pools": pools, "per_split": per_split,
            "candidates": cands,
            "other_files": audit_other_files()}


def manual_gold_gt():
    """{pano: GroundTruth} GT-A for the 1,000 record panos, straight from manual_labels/."""
    from rampnet.detection_eval import load_yolo_ground_truths
    gts = load_yolo_ground_truths(LABELS_DIR)
    pids = [p for p, _, _ in load_manual_gold()]
    missing = [p for p in pids if p not in gts]
    if missing:
        raise SystemExit(f"{len(missing)} manual_gold record panos have no label file")
    return {p: gts[p] for p in pids}


def bounds_markdown(b):
    lines = ["| pool | t | panos | peaks | strip | pairs | `C` | `E` | 20-px env. | 30-px env. | panos touched |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pool in ("cities11", "cities13", MG):
        for t in THRESHOLDS:
            r = b["pools"][pool][f"{t:.2f}"]
            lines.append(f"| {pool} | {t:.2f} | {r['panos']:,} | {r['peaks']:,} | {r['strip']} | "
                         f"{r['pairs']} | {r['C']} | {r['E']} | {r['env20']} | {r['env30']} | "
                         f"{r['panos_touched']} |")
    lines += ["", "| split | strip / pairs / `C` / `E` at 0.30 | at 0.55 |", "|---|---|---|"]
    for s in SPLITS13 + (MG,):
        a, c = b["per_split"][s]["0.30"], b["per_split"][s]["0.55"]
        lines.append(f"| {s} | {a['strip']} / {a['pairs']} / {a['C']} / {a['E']} | "
                     f"{c['strip']} / {c['pairs']} / {c['C']} / {c['E']} |")
    return "\n".join(lines)


def check_reviewer_literals(b):
    bad = []
    for pool, keys in REVIEWER_COUNTS.items():
        for k, (v30, v55) in keys.items():
            got = (b["pools"][pool]["0.30"][k], b["pools"][pool]["0.55"][k])
            if got != (v30, v55):
                bad.append(f"{pool} {k}: {got} != reviewer {(v30, v55)}")
    near = [d for d in b["tie_distances_within_30px"] if d <= ENV]
    if near:
        bad.append(f"tied across-seam partners within {ENV} px: {near} (must be none)")
    if b["manual_gold_gt_points"] != 3919:
        bad.append(f"manual_gold GT points {b['manual_gold_gt_points']} != 3919")
    return bad


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def cmd_bounds(args):
    b = build_bounds()
    b["manual_gold_gt_points"] = sum(len(g.gt_points) for g in manual_gold_gt().values())
    bad = check_reviewer_literals(b)
    print(bounds_markdown(b))
    print(f"\ntied partners within 30 px at {b['tie_distances_within_30px']} px; max pair distance at the floor: "
          f"{b['max_pair_dist_px_at_floor']} px; manual_gold GT points: {b['manual_gold_gt_points']}")
    for r in b["other_files"]:
        if any(v for k, v in r.items() if k != "file"):
            print("  seam peaks in", r)
    if bad:
        print("\nREVIEWER LITERALS DIFFER (stop and report):\n  " + "\n  ".join(bad))
        return 1
    if args.out:
        write_json(args.out, b)
        print(f"\nwrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# sheet (reads local panoramas; images + HTML are git-ignored)
# --------------------------------------------------------------------------- #
def jon_pairs():
    """Jon's 14 adjudicated seam pairs: [{pano, a, b, verdict}] in file order."""
    v = read_json(SEAM_VERDICTS)
    return [{"pano": x["pano"], "a": tuple(x["a"]), "b": tuple(x["b"]), "verdict": x["verdict"]}
            for x in v["verdicts"]]


def inherit(cand, pairs):
    """Index of the Jon pair a manual_gold candidate inherits, or None (the pinned join).

    Inherits only if the weak and strong peaks are each nearest to a different mark of one
    adjudicated pair on the same pano, each within JOIN_PX wrapped Chebyshev px.
    """
    for k, pr in enumerate(pairs):
        if pr["pano"] != cand["pano"]:
            continue
        marks = (pr["a"], pr["b"])
        near = []
        for peak in (cand["weak"], cand["strong"]):
            ds = [norm_cheb_px(peak, m) for m in marks]
            near.append((min(range(2), key=lambda i: ds[i]), min(ds)))
        if near[0][0] != near[1][0] and max(near[0][1], near[1][1]) <= JOIN_PX:
            return k
    return None


def study_cards(bounds):
    """C and E candidates at RULE_T in cities13 and manual_gold, with inheritance resolved."""
    pairs = jon_pairs()
    out = []
    for c in bounds["candidates"]:
        if c["set"] not in ("C", "E") or c["weak"][2] < RULE_T:
            continue
        if c["split"] not in SPLITS13 + (MG,):
            continue
        k = inherit(c, pairs) if c["split"] == MG else None
        out.append({**c, "inherits": k})
    return out


def find_pano(panos_root, split, pano):
    for ext in ("jpg", "jpeg", "png"):
        p = os.path.join(panos_root, "benchmark", split, "panos", f"{pano}.{ext}")
        if os.path.exists(p):
            return p
    return None


def wrap_crop(img, cx, cy, w, h):
    """A crop of ``w`` x ``h`` native px centred on (cx, cy), wrapping in x, clamped in y."""
    from PIL import Image
    W0, H0 = img.size
    left = int(round(cx - w / 2.0)) % W0
    top = max(0, min(int(round(cy - h / 2.0)), H0 - h))
    out = Image.new(img.mode, (w, h))
    first = min(w, W0 - left)
    out.paste(img.crop((left, top, left + first, top + h)), (0, 0))
    if first < w:
        out.paste(img.crop((0, top, w - first, top + h)), (first, 0))
    return out, left, top


def draw_marks(im, marks, scale, left, top, W0):
    """Draw ring + letter for each (letter, x_norm_px, y_norm_px, colour) on ``im``."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for letter, x, y, col in marks:
        u = (((x - left) % W0)) * scale
        v = (y - top) * scale
        r = 13
        dr.ellipse((u - r, v - r, u + r, v + r), outline=(0, 0, 0), width=5)
        dr.ellipse((u - r, v - r, u + r, v + r), outline=col, width=3)
        dr.rectangle((u - 9, v - r - 30, u + 11, v - r - 6), fill=(0, 0, 0))
        dr.text((u - 4, v - r - 27), letter, fill=col)
    return im


def render_card(path, m1, m2):
    """One card image: wide context with marks, zoom with marks, zoom without (stacked)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    mx, my = wrapped_midpoint(m1, m2)
    cx, cy = mx * W0, my * H0
    marks = [("A", m1[0] * W0, m1[1] * H0, (255, 212, 0)), ("B", m2[0] * W0, m2[1] * H0, (0, 229, 255))]
    panels = []
    for fov_deg, marked in ((60, True), (16, True), (16, False)):
        w = int(round(fov_deg / 360.0 * W0))
        h = w // 2
        crop, left, top = wrap_crop(img, cx, cy, w, h)
        scale = 1024.0 / w
        crop = crop.resize((1024, 512), Image.LANCZOS)
        if marked:
            draw_marks(crop, marks, scale, left, top, W0)
        panels.append(crop)
    out = Image.new("RGB", (1024, 512 * 3 + 16), (20, 22, 26))
    for i, p in enumerate(panels):
        out.paste(p, (0, i * (512 + 8)))
    return out


def deck_items(bounds):
    """(items keyed by a stable string, the study cards) for the sheet."""
    items = {}
    for c in study_cards(bounds):
        if c["inherits"] is not None:
            continue
        key = f"study|{c['split']}|{c['pano']}|{c['weak'][0]}|{c['weak'][1]}"
        items[key] = {"kind": "study", "split": c["split"], "pano": c["pano"], "set": c["set"],
                      "marks": [c["weak"][:2], c["strong"][:2]], "weak_index": 0,
                      "weak_score": c["weak"][2], "strong_score": c["strong"][2], "dist": c["dist"]}
    for k, pr in enumerate(jon_pairs()):
        key = f"calibration|{pr['pano']}|{k}"
        items[key] = {"kind": "calibration", "split": MG, "pano": pr["pano"], "jon_index": k,
                      "marks": [list(pr["a"]), list(pr["b"])]}
    return items


def cmd_sheet(args):
    from PIL import Image  # noqa: F401  (fail fast if Pillow is absent)
    bounds = read_json(args.bounds)
    items = deck_items(bounds)
    order = deck_order(items)
    cards_dir = os.path.join(args.out, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    listing, key, html_cards, missing = [], {}, [], []
    for n, (k, first_is_a) in enumerate(order, 1):
        it = items[k]
        cid = f"s248-{n:02d}"
        m = it["marks"] if first_is_a else it["marks"][::-1]
        path = find_pano(args.panos_root, it["split"], it["pano"])
        if path is None:
            missing.append(it["pano"])
            continue
        im = render_card(path, m[0], m[1])
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
        with open(os.path.join(cards_dir, f"{cid}.jpg"), "wb") as f:
            f.write(data)
        listing.append({"id": cid, "order": n, "image_sha256": hashlib.sha256(data).hexdigest()})
        rec = {k2: v for k2, v in it.items() if k2 != "marks"}
        rec["A"], rec["B"] = m[0], m[1]
        if it["kind"] == "study":
            rec["weaker"] = "A" if first_is_a else "B"
        else:
            rec["A_is"] = "a" if first_is_a else "b"
        key[cid] = rec
        html_cards.append({"id": cid, "img": "data:image/jpeg;base64," + base64.b64encode(data).decode()})
    if missing:
        raise SystemExit(f"no local imagery for {missing} under {args.panos_root}")
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    write_json(os.path.join(args.out, "items.json"),
               {"seed": SEED, "digest": digest, "n": len(listing),
                "rubric_version": RUBRIC_VERSION, "cards": listing,
                "regenerate": "seam_nms_default_248.py sheet --panos-root <checkout>"})
    write_json(os.path.join(args.out, "key.json"), {"digest": digest, "cards": key})
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8", newline="") as f:
        f.write(render_html(html_cards, digest, rater=args.rater))
    print(f"{len(listing)} cards ({sum(v['kind'] == 'study' for v in key.values())} study, "
          f"{sum(v['kind'] == 'calibration' for v in key.values())} calibration); digest {digest}")
    print(f"wrote {args.out}/items.json, key.json (do not open before rating), index.html, cards/")
    return 0


def render_html(cards, digest, rater="", defaults=None, title="Seam pairs: same object or not?",
                note=""):
    """One self-contained page: one card at a time, keyboard verdicts, localStorage, Export.

    ``defaults`` (id -> verdict) pre-fills a verdict for the rater to confirm or change (the
    audit copy shows the AI pre-rating this way)."""
    payload = json.dumps({"cards": cards, "digest": digest, "rater": rater,
                          "defaults": defaults or {}, "rubric_version": RUBRIC_VERSION,
                          "rubric": [{"rule": t, "detail": d} for t, d in RUBRIC]})
    rubric = "".join(f"<li><b>{t}</b> {d}</li>" for t, d in RUBRIC)
    return (HTML.replace("__PAYLOAD__", payload).replace("__RUBRIC__", rubric)
            .replace("__TITLE__", title).replace("__NOTE__", note))


HTML = r"""<!doctype html>
<!-- generated by scripts/analysis/seam_nms_default_248.py (#248) -->
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
 body{font:15px/1.5 system-ui,Segoe UI,sans-serif;margin:0;background:#14161a;color:#e8e8ea}
 .wrap{max-width:1060px;margin:0 auto;padding:16px}
 h1{font-size:20px;margin:0 0 6px} .sub{color:#a7adb5;font-size:14px;margin:0 0 10px}
 details{background:#1c1f25;border:1px solid #2c313a;border-radius:8px;padding:8px 12px;margin:0 0 12px}
 img{width:100%;height:auto;display:block;border-radius:6px}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
 button{background:#252a33;color:#e8e8ea;border:1px solid #39404c;border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
 button:focus-visible{outline:3px solid #ffd400}
 button.on{background:#1a7f37;border-color:#1a7f37;color:#fff}
 .lab{min-width:120px;color:#a7adb5} .ai{color:#ffd400;font-size:13px}
 kbd{background:#2a2f38;border-radius:4px;padding:0 5px;font-size:12px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">__NOTE__ Panels: wide view with marks A (yellow) and B (cyan); zoom with marks; the same zoom without marks.
The seam is at the centre of each panel; the imagery there is continuous.</p>
<details><summary>Rubric (version __RV__) and keys</summary><ol>__RUBRIC__</ol>
<p>Pair: <kbd>1</kbd> same <kbd>2</kbd> different <kbd>3</kbd> unsure.
Mark A: <kbd>q</kbd> ramp <kbd>w</kbd> not ramp <kbd>e</kbd> unsure.
Mark B: <kbd>a</kbd> ramp <kbd>s</kbd> not ramp <kbd>d</kbd> unsure.
<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> previous / next. Verdicts autosave in this browser.</p></details>
<div class="row"><b id="pos"></b><span id="done" class="lab"></span><span style="flex:1"></span>
<button id="prev">&larr; prev</button><button id="next">next &rarr;</button><button id="ex">Export verdicts</button></div>
<img id="img" alt="card image">
<p id="ai" class="ai"></p>
<div class="row" role="group" aria-label="pair"><span class="lab">A and B are</span>
 <button data-f="pair" data-v="same">1 same object</button><button data-f="pair" data-v="different">2 different objects</button><button data-f="pair" data-v="unsure">3 unsure</button></div>
<div class="row" role="group" aria-label="mark A"><span class="lab">Mark A is</span>
 <button data-f="A" data-v="ramp">q ramp</button><button data-f="A" data-v="not_ramp">w not ramp</button><button data-f="A" data-v="unsure">e unsure</button></div>
<div class="row" role="group" aria-label="mark B"><span class="lab">Mark B is</span>
 <button data-f="B" data-v="ramp">a ramp</button><button data-f="B" data-v="not_ramp">s not ramp</button><button data-f="B" data-v="unsure">d unsure</button></div>
<div class="row"><input id="note" type="text" placeholder="note (optional)" style="flex:1;background:#0f1115;color:#e8e8ea;border:1px solid #39404c;border-radius:6px;padding:6px"></div>
</div>
<script>
const P = __PAYLOAD__;
const store = 'seam248_' + P.digest + '_' + (P.rater || 'rater');
let V = {}; try { V = JSON.parse(localStorage.getItem(store) || '{}'); } catch (e) {}
for (const [id, d] of Object.entries(P.defaults)) if (!V[id]) V[id] = Object.assign({prefilled: true}, d);
let i = 0;
function save(){ try { localStorage.setItem(store, JSON.stringify(V)); } catch (e) {} }
function show(){
  const c = P.cards[i], v = V[c.id] || {};
  document.getElementById('img').src = c.img;
  document.getElementById('img').alt = 'card ' + c.id;
  document.getElementById('pos').textContent = (i + 1) + ' / ' + P.cards.length + '  (' + c.id + ')';
  document.getElementById('done').textContent = Object.values(V).filter(x => x.pair && !x.prefilled).length + ' confirmed';
  document.getElementById('ai').textContent = v.prefilled ? 'Pre-filled from the AI pre-rating: confirm by re-pressing, or change.' : '';
  document.querySelectorAll('[data-f]').forEach(b => b.classList.toggle('on', v[b.dataset.f] === b.dataset.v));
  document.getElementById('note').value = v.note || '';
}
function set(f, val){ const id = P.cards[i].id; V[id] = Object.assign({}, V[id] || {}, {[f]: val}); delete V[id].prefilled; save(); show(); }
document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => set(b.dataset.f, b.dataset.v));
document.getElementById('note').onchange = e => set('note', e.target.value);
document.getElementById('prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('next').onclick = () => { i = Math.min(P.cards.length - 1, i + 1); show(); };
const K = {'1':['pair','same'],'2':['pair','different'],'3':['pair','unsure'],q:['A','ramp'],w:['A','not_ramp'],e:['A','unsure'],a:['B','ramp'],s:['B','not_ramp'],d:['B','unsure']};
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (K[e.key]) set(...K[e.key]);
  else if (e.key === 'ArrowRight') { i = Math.min(P.cards.length - 1, i + 1); show(); }
  else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); }
});
document.getElementById('ex').onclick = () => {
  const out = {rater: P.rater, items_digest: P.digest, rubric_version: P.rubric_version, rubric: P.rubric,
    verdicts: P.cards.map(c => { const v = V[c.id] || {}; return {id: c.id, pair: v.pair || null, A: v.A || null, B: v.B || null, note: v.note || null, prefilled_unconfirmed: !!v.prefilled}; })};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'verdicts__' + (P.rater || 'rater') + '.json'; a.click();
};
show();
</script></body></html>
""".replace("__RV__", str(RUBRIC_VERSION))


# --------------------------------------------------------------------------- #
# calibrate (AI vs Jon on his 14 adjudicated pairs; runs before any study card is classed)
# --------------------------------------------------------------------------- #
PAIR_OF_JON = {"one": "same", "two": "different", "unsure": "unsure"}


def load_verdicts(path, key):
    """{card id: verdict dict} from a per-rater export, checked against the deck digest."""
    v = read_json(path)
    if v.get("items_digest") != key["digest"]:
        raise SystemExit(f"{path}: items_digest {v.get('items_digest')} != deck {key['digest']}")
    return {x["id"]: x for x in v["verdicts"]}


def calibration_table(key, ai, jon_file_verdicts):
    """Per calibration card: Jon's verdict, the AI's pair answer, and whether they agree."""
    rows = []
    for cid, k in sorted(key["cards"].items()):
        if k["kind"] != "calibration":
            continue
        jv = jon_file_verdicts[k["jon_index"]]
        a = ai.get(cid, {}).get("pair")
        rows.append({"id": cid, "pano": k["pano"], "jon": jv["verdict"], "sep_px": jv["sep_px"],
                     "ai_pair": a, "agree": a == PAIR_OF_JON.get(jv["verdict"])})
    return rows


def calibration_summary(rows):
    """The pre-registered bar: the AI answered `different` on every one of Jon's `two` pairs."""
    two = [r for r in rows if r["jon"] == "two"]
    one = [r for r in rows if r["jon"] == "one"]
    bands = {}
    for r in one:
        band = "one, <= 8 px" if r["sep_px"] <= 8 else "one, > 8 px"
        b = bands.setdefault(band, [0, 0])
        b[0] += r["agree"]
        b[1] += 1
    return {"n": len(rows), "two_total": len(two), "two_agree": sum(r["agree"] for r in two),
            "one_total": len(one), "one_agree": sum(r["agree"] for r in one),
            "one_by_band": {k: {"agree": v[0], "n": v[1]} for k, v in sorted(bands.items())},
            "ai_answers_on_one": {a: sum(r["ai_pair"] == a for r in one)
                                  for a in ("same", "different", "unsure", None)},
            "calibrated": bool(two) and all(r["agree"] for r in two),
            "bar": "AI answered `different` on all of Jon's `two` pairs (pre-registered)"}


def build_calibration(review_dir=REVIEW_DIR):
    key = read_json(os.path.join(review_dir, "key.json"))
    ai = load_verdicts(os.path.join(review_dir, f"verdicts__{AI_RATER}.json"), key)
    jon = read_json(SEAM_VERDICTS)["verdicts"]
    rows = calibration_table(key, ai, jon)
    return {"rater": AI_RATER, "items_digest": key["digest"], "summary": calibration_summary(rows),
            "cards": rows}


def cmd_calibrate(args):
    c = build_calibration()
    s = c["summary"]
    print(f"Jon `two`: AI agrees on {s['two_agree']} / {s['two_total']}; Jon `one`: "
          f"{s['one_agree']} / {s['one_total']} {s['one_by_band']}; answers on `one`: "
          f"{s['ai_answers_on_one']}")
    print("CALIBRATED" if s["calibrated"] else "NOT CALIBRATED: the AI verdicts are not used")
    for r in c["cards"]:
        print(f"  {r['id']} jon={r['jon']:<6} sep={r['sep_px']:>6} ai={r['ai_pair']}")
    if args.out:
        write_json(args.out, c)
        print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# report: classification, the rule, and the (non-gating) paired scores
# --------------------------------------------------------------------------- #
KEEP = "KEEP OPT-IN AND CLOSE"
FLIP_COND = "FLIP, CONDITIONAL ON JON'S AUDIT"
FLIP = "FLIP"
NEXT = "NEXT STEP NEEDED"
TRIPWIRE_DF1 = -0.001
SEAM_BAND = 0.022 + D / W           # match radius plus the NMS reach, normalized x
RND = 6


def rnd(v, nd=RND):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), nd)


def card_id_for(cand, key):
    """The deck id of a study candidate (matched on split, pano and the weaker mark)."""
    for cid, k in key["cards"].items():
        if k["kind"] != "study" or k["split"] != cand["split"] or k["pano"] != cand["pano"]:
            continue
        if list(k[k["weaker"]]) == list(cand["weak"][:2]):
            return cid
    return None


def classify_cards(bounds, key, ai, ai_calibrated, jon_audit=None):
    """Every C/E card at RULE_T with its effective verdict, source and class (doc section 3)."""
    pairs = jon_pairs()
    out = []
    for c in study_cards(bounds):
        row = {"split": c["split"], "pano": c["pano"], "set": c["set"], "dist": c["dist"],
               "weak_score": c["weak"][2], "strong_score": c["strong"][2],
               "card": None, "source": "none", "cls": "U"}
        if c["inherits"] is not None:
            jv = pairs[c["inherits"]]["verdict"]
            row.update(source="jon (inherited)", verdict=jv, cls=jon_class(jv),
                       jon_pair_index=c["inherits"])
        else:
            cid = card_id_for(c, key)
            row["card"] = cid
            if cid is None:
                raise SystemExit(f"no deck card for {c['split']} {c['pano']} {c['weak']}")
            k = key["cards"][cid]
            weak_l, strong_l = k["weaker"], ("B" if k["weaker"] == "A" else "A")
            src = None
            if jon_audit and jon_audit.get(cid, {}).get("pair") and \
                    not jon_audit[cid].get("prefilled_unconfirmed"):
                src, v = "jon (audit)", jon_audit[cid]
            elif ai_calibrated and ai.get(cid, {}).get("pair"):
                src, v = "ai", ai[cid]
            if src:
                row.update(source=src, verdict={"pair": v["pair"], "weak": v[weak_l],
                                                "strong": v[strong_l]},
                           cls=classify(v["pair"], v[weak_l], v[strong_l]))
        out.append(row)
    return out


def decide(cards, ai_calibrated):
    """Apply the pre-registered rule (doc section 4.2). Returns (outcome, reasons dict).

    >>> decide([{"set": "E", "cls": "L", "source": "ai", "card": "x"}], True)[0]
    'NEXT STEP NEEDED'
    """
    is_jon = lambda r: r["source"].startswith("jon")  # noqa: E731
    l_jon_c = [r for r in cards if r["set"] == "C" and r["cls"] == "L" and is_jon(r)]
    l_opt_c = [r for r in cards if r["set"] == "C" and r["cls"] == "L"]
    l_pess = [r for r in cards if r["cls"] in ("L", "U")]
    reasons = {"L_jon_C": len(l_jon_c), "L_opt_C": len(l_opt_c), "L_pess": len(l_pess),
               "ai_calibrated": ai_calibrated,
               "all_cards_rated_by_jon": all(is_jon(r) for r in cards)}
    if l_jon_c or len(l_opt_c) >= 2:
        return KEEP, reasons
    if ai_calibrated and len(l_pess) <= 1:
        reasons["cards_first"] = [r["card"] for r in l_pess]
        return (FLIP if reasons["all_cards_rated_by_jon"] else FLIP_COND), reasons
    rate = [r for r in cards if not is_jon(r) and
            (r["cls"] == "U" or (r["source"] == "ai" and r["cls"] == "L") or not ai_calibrated)]
    reasons["cards_jon_must_rate"] = sorted(r["card"] for r in rate if r["card"])
    reasons["panoramas_for_exact_tier"] = sorted({f"{r['split']}:{r['pano']}" for r in cards
                                                  if r["set"] == "E" and r["cls"] in ("L", "U")})
    return NEXT, reasons


def merge_gt_b(gts, pairs):
    """GT-B: Jon's `one` pairs merged to their wrapped midpoint, in memory (doc section 4.2)."""
    from rampnet.detection_eval import GroundTruth
    out = dict(gts)
    merged = []
    for pr in pairs:
        if pr["verdict"] != "one":
            continue
        g = out[pr["pano"]]
        pts = list(g.gt_points)
        idx = []
        for m in (pr["a"], pr["b"]):
            j = min(range(len(pts)), key=lambda i: abs(pts[i][0] - m[0]) + abs(pts[i][1] - m[1]))
            if abs(pts[j][0] - m[0]) + abs(pts[j][1] - m[1]) > 1e-4 or j in idx:
                raise SystemExit(f"GT-B: mark {m} of {pr['pano']} not found in manual_labels")
            idx.append(j)
        mid = wrapped_midpoint(pts[idx[0]], pts[idx[1]])
        merged.append({"pano": pr["pano"], "mid": mid,
                       "to_a_px": rnd(norm_cheb_px(mid, pts[idx[0]]), 3),
                       "to_b_px": rnd(norm_cheb_px(mid, pts[idx[1]]), 3),
                       "to_seam_px": rnd(min(mid[0], 1 - mid[0]) * W, 3)})
        pts = [p for i, p in enumerate(pts) if i not in idx] + [mid]
        out[pr["pano"]] = GroundTruth(pts, list(g.ignore_points), g.fn_confirmed)
    return out, merged


def arms_for(panos, cands):
    """{scenario: {pano: preds}} for default / lo (remove C) / hi (remove C and E)."""
    rm = {"lo": {}, "hi": {}}
    for c in cands:
        key = (round(c["weak"][0], 6), round(c["weak"][1], 6))
        if c["set"] == "C":
            rm["lo"].setdefault(c["pano"], set()).add(key)
        if c["set"] in ("C", "E"):
            rm["hi"].setdefault(c["pano"], set()).add(key)
    out = {"default": {}, "lo": {}, "hi": {}}
    removed = {"lo": {}, "hi": {}}
    for pano, preds, _ in panos:
        out["default"][pano] = list(preds)
        for sc in ("lo", "hi"):
            drop = rm[sc].get(pano, set())
            kept = [p for p in preds if (round(p[0], 6), round(p[1], 6)) not in drop]
            if len(preds) - len(kept) != len(drop):
                raise SystemExit(f"{pano}: removal keys did not match the stored peaks")
            out[sc][pano] = kept
            removed[sc][pano] = [p for p in preds if (round(p[0], 6), round(p[1], 6)) in drop]
    return out, removed


def scored(split, preds_by_pano, gts, rsq):
    import input_res_sweep_25 as irs
    panos = [{"pano": p, "preds": preds_by_pano[p], "gt": gts[p]} for p in sorted(gts)]
    return irs.scored_from_panos(split, panos, rsq)


def assert_unchanged_panos_identical(s_def, s_arm, removed):
    """The invariant that does hold: on every pano with no removal, per-pano results match."""
    import numpy as np
    changed = {i for i, p in enumerate(s_def.pids) if removed.get(p.split("/")[-1])}
    keep_d = ~np.isin(s_def.det_pano, list(changed))
    keep_a = ~np.isin(s_arm.det_pano, list(changed))
    same = (np.array_equal(s_def.conf[keep_d], s_arm.conf[keep_a])
            and np.array_equal(s_def.is_tp[keep_d], s_arm.is_tp[keep_a]))
    if not same:
        raise SystemExit("invariant broken: a pano with no removal scored differently")


def contrast(s_def, s_arm, sizes, t, removed_scores):
    """Observed P/R/F1 of both arms and the paired difference with a 95% bootstrap CI.

    When no removed peak scores >= t the difference is exactly 0 in every replicate (greedy
    matching in confidence order: dropping a lower peak cannot move a higher one's match), so
    no bootstrap is drawn."""
    import numpy as np
    import benchmark_power_135 as bp
    ones = np.ones((1, len(s_def.pids)))
    md = [float(x[0]) for x in bp.metrics(s_def, ones, t)][:3]
    ma = [float(x[0]) for x in bp.metrics(s_arm, ones, t)][:3]
    names = ("precision", "recall", "f1")
    out = {"default": dict(zip(names, map(rnd, md))), "wrap": dict(zip(names, map(rnd, ma)))}
    if not any(s >= t for s in removed_scores):
        out["delta"] = {n: {"observed": 0.0, "ci_lo": 0.0, "ci_hi": 0.0} for n in names}
        out["bootstrap"] = "none (no removed peak >= t: the difference is 0 in every replicate)"
    else:
        rng = np.random.default_rng(SEED)
        r = bp.observed_and_se(s_arm, sizes, t, rng, N_REPS, paired=s_def)
        out["delta"] = {n: {"observed": rnd(r[n]["observed"]), "ci_lo": rnd(r[n]["ci_lo"]),
                            "ci_hi": rnd(r[n]["ci_hi"])} for n in names}
        out["bootstrap"] = f"paired pano-level, stratified by split, {N_REPS} reps, seed {SEED}"
    out["_df1_raw"] = ma[2] - md[2]
    a = bp.mcnemar(s_def, s_arm, t, t)
    lost = (s_def.hit_gt >= t) & ~(s_arm.hit_gt >= t)
    gained = ~(s_def.hit_gt >= t) & (s_arm.hit_gt >= t)
    band = np.minimum(s_def.gt_x, 1 - s_def.gt_x) < SEAM_BAND
    out["gt_flips"] = {"gained": a[0], "lost": a[1], "n_gt": a[2],
                       "lost_in_seam_band": int((lost & band).sum()),
                       "gained_in_seam_band": int((gained & band).sum())}
    return out


def changed_panos(default, arm, removed, gts, rsq):
    """Per pano with a removal >= 0.30: TP / FP at 0.30 and 0.55 in both arms (diagnostic)."""
    from rampnet.detection_eval import score_pano
    out = []
    for pano in sorted(p for p, r in removed.items() if any(x[2] >= 0.30 for x in r)):
        row = {"pano": pano}
        for t in (0.30, 0.55):
            sd = score_pano([p for p in default[pano] if p[2] >= t], gts[pano], radius_sq=rsq)
            sa = score_pano([p for p in arm[pano] if p[2] >= t], gts[pano], radius_sq=rsq)
            row[f"{t:.2f}"] = {"tp": [sd.tp, sa.tp], "fp": [sd.fp, sa.fp]}
        out.append(row)
    return out


def ap_of(preds_by_pano, gts, rsq):
    import operating_point_curve as opc
    panos = [{"pano": p, "preds": preds_by_pano[p], "gt": gts[p]} for p in sorted(gts)]
    return opc.pr_curve_and_ap(panos, rsq).ap


def build_scores(bounds, data):
    """Paired scores for every split and pool, both scenarios, at 0.30 and 0.55."""
    import benchmark_power_135 as bp
    from operating_point_curve import _gt_from_json
    from rampnet.detection_eval import radius_sq_for
    rsq = radius_sq_for()
    cands = bounds["candidates"]
    units = {}
    gt_a = manual_gold_gt()
    gt_b, merged = merge_gt_b(gt_a, jon_pairs())
    gtsets = {s: {p: _gt_from_json(g) for p, _, g in data[s]} for s in SPLITS13}
    gtsets["manual_gold GT-A"] = gt_a
    gtsets["manual_gold GT-B"] = gt_b
    per = {}
    for unit, gts in gtsets.items():
        split = MG if unit.startswith(MG) else unit
        arms, removed = arms_for(data[split], [c for c in cands if c["split"] == split])
        sc = {a: scored(split, arms[a], gts, rsq) for a in arms}
        for a in ("lo", "hi"):
            assert_unchanged_panos_identical(sc["default"], sc[a], removed[a])
        per[unit] = (sc, removed, arms, gts)
    pools = {"cities11": SPLITS11, "cities13": SPLITS13,
             "manual_gold GT-A": ("manual_gold GT-A",), "manual_gold GT-B": ("manual_gold GT-B",)}
    for name in list(SPLITS13) + list(pools):
        members = pools.get(name, (name,))
        out = {}
        for a in ("lo", "hi"):
            s_def = bp.stack([per[m][0]["default"] for m in members])
            s_arm = bp.stack([per[m][0][a] for m in members])
            sizes = [len(per[m][0]["default"].pids) for m in members]
            rem = [p[2] for m in members for ps in per[m][1][a].values() for p in ps]
            out[a] = {"removed": {"all": len(rem), "ge_0.30": sum(s >= 0.30 for s in rem),
                                  "ge_0.55": sum(s >= 0.55 for s in rem)}}
            for t in (0.30, 0.55):
                out[a][f"{t:.2f}"] = contrast(s_def, s_arm, sizes, t, rem)
            if len(members) == 1:       # AP per split / manual_gold GT only, not pooled
                m = members[0]
                out[a]["changed_panos"] = changed_panos(per[m][2]["default"], per[m][2][a],
                                                        per[m][1][a], per[m][3], rsq)
                ap_d = ap_of(per[m][2]["default"], per[m][3], rsq)
                ap_w = ap_of(per[m][2][a], per[m][3], rsq)
                out[a]["ap_default"], out[a]["ap_wrap"] = rnd(ap_d), rnd(ap_w)
                out[a]["ap_delta"] = rnd(ap_w - ap_d)
        units[name] = out
    return units, merged


def class_counts(cards, t):
    out = {}
    for pool in ("cities13", MG):
        for st in ("C", "E"):
            rows = [r for r in cards if r["set"] == st and r["weak_score"] >= t
                    and (r["split"] == MG) == (pool == MG)]
            out[f"{pool} {st}"] = {k: sum(r["cls"] == k for r in rows) for k in "DLNU"}
    return out


def build_results():
    bounds = read_json(os.path.join(OUT_DIR, "bounds.json"))
    key = read_json(os.path.join(REVIEW_DIR, "key.json"))
    ai = load_verdicts(os.path.join(REVIEW_DIR, f"verdicts__{AI_RATER}.json"), key)
    jon_path = os.path.join(REVIEW_DIR, "verdicts__jon.json")
    jon_audit = load_verdicts(jon_path, key) if os.path.exists(jon_path) else None
    cal = build_calibration()
    calibrated = cal["summary"]["calibrated"]
    cards = classify_cards(bounds, key, ai, calibrated, jon_audit)
    outcome, reasons = decide(cards, calibrated)
    units, merged = build_scores(bounds, load_pools())
    trip = {}
    for unit in ("cities13", "manual_gold GT-B"):
        for t in ("0.30", "0.55"):
            trip[f"{unit} @ {t}"] = units[unit]["lo"][t]["_df1_raw"] < TRIPWIRE_DF1
    if jon_audit:
        agree = [(ai[c].get("pair"), jon_audit[c].get("pair")) for c in jon_audit if c in ai]
    else:
        agree = None
    return {"what": "seam-wrapped NMS default study (#248): classification, rule, scores",
            "rule_threshold": RULE_T, "outcome": outcome, "reasons": reasons,
            "tripwire": {"fired": any(trip.values()), "by_unit": trip,
                         "rule": f"pooled paired dF1 under lo < {TRIPWIRE_DF1}"},
            "calibration": cal["summary"],
            "jon_audit_present": jon_audit is not None, "ai_vs_jon_audit": agree,
            "class_counts": {f"{t:.2f}": class_counts(cards, t) for t in (0.30, 0.55)},
            "cards": cards, "gt_b_merges": merged,
            "gt_points": {"GT-A": 3919, "GT-B": sum(len(g.gt_points)
                                                   for g in merge_gt_b(manual_gold_gt(),
                                                                       jon_pairs())[0].values())},
            "scores": _strip_private(units)}


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def results_markdown(r):
    L = [f"**Outcome under the pre-registered rule: {r['outcome']}.** Reasons: "
         + ", ".join(f"{k} = {v}" for k, v in r["reasons"].items()) + ".", "",
         f"Calibration: AI `different` on {r['calibration']['two_agree']} / "
         f"{r['calibration']['two_total']} of Jon's `two` pairs, `same` on "
         f"{r['calibration']['one_agree']} / {r['calibration']['one_total']} `one` pairs; "
         f"calibrated = {r['calibration']['calibrated']}. Tripwire fired: {r['tripwire']['fired']}.",
         "", "| card | split | set | dist px | weak / strong score | source | verdict | class |",
         "|---|---|---|---:|---|---|---|---|"]
    for c in sorted(r["cards"], key=lambda c: (c["split"] != MG, c["split"], c["set"], c["card"] or "")):
        v = c.get("verdict")
        vs = v if isinstance(v, str) else (f"{v['pair']}; weak {v['weak']}, strong {v['strong']}"
                                           if v else "")
        L.append(f"| {c['card'] or '(inherited)'} | {c['split']} | {c['set']} | {c['dist']} | "
                 f"{c['weak_score']:.3f} / {c['strong_score']:.3f} | {c['source']} | {vs} | "
                 f"{c['cls']} |")
    L += ["", "| pool, set | t | D | L | N | U |", "|---|---:|---:|---:|---:|---:|"]
    for t, cc in r["class_counts"].items():
        for k, v in cc.items():
            L.append(f"| {k} | {t} | {v['D']} | {v['L']} | {v['N']} | {v['U']} |")
    L += ["", "Paired scores (wrap minus default; reported, never gating). `lo` removes `C`, "
          "`hi` removes `C` and `E`. A CI of [0, 0] with no bootstrap means no removed peak "
          "reaches t.", "",
          "| unit | scen. | t | removed >= t | P def -> wrap | R def -> wrap | dF1 [95% CI] | GT lost / gained |",
          "|---|---|---:|---:|---|---|---|---|"]
    for unit in ("cities11", "cities13", "manual_gold GT-A", "manual_gold GT-B") + SPLITS13:
        for sc in ("lo", "hi"):
            u = r["scores"][unit][sc]
            for t in ("0.30", "0.55"):
                x = u[t]
                n = u["removed"][f"ge_{t}"]
                if unit in SPLITS13 and n == 0:
                    continue
                d = x["delta"]["f1"]
                L.append(f"| {unit} | {sc} | {t} | {n} | {x['default']['precision']:.4f} -> "
                         f"{x['wrap']['precision']:.4f} | {x['default']['recall']:.4f} -> "
                         f"{x['wrap']['recall']:.4f} | {d['observed']:+.5f} [{d['ci_lo']:+.5f}, "
                         f"{d['ci_hi']:+.5f}] | {x['gt_flips']['lost']} / {x['gt_flips']['gained']} |")
    L += ["", "AP (from the 0.05 floor), wrap minus default: " + "; ".join(
        f"{u} {sc} {r['scores'][u][sc]['ap_delta']:+.5f}"
        for u in ("manual_gold GT-A", "manual_gold GT-B") + SPLITS13 for sc in ("lo", "hi")
        if r["scores"][u][sc].get("ap_delta")) + " (every other split: 0)."]
    return "\n".join(L) + "\n"


def cmd_report(args):
    r = build_results()
    md = results_markdown(r)
    print(md)
    if args.out:
        write_json(args.out, r)
        with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8", newline="") as f:
            f.write(md)
        print(f"wrote {args.out} and .md")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="re-derive every committed output and compare; writes nothing")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bounds", help="removal-candidate counts from committed peaks")
    b.add_argument("--out", default=os.path.join(OUT_DIR, "bounds.json"))
    s = sub.add_parser("sheet", help="cut the blinded card images (needs local panoramas)")
    s.add_argument("--panos-root", required=True,
                   help="checkout holding benchmark/<split>/panos (read-only)")
    s.add_argument("--bounds", default=os.path.join(OUT_DIR, "bounds.json"))
    s.add_argument("--out", default=REVIEW_DIR)
    s.add_argument("--rater", default="", help="rater id the HTML exports under")
    c = sub.add_parser("calibrate", help="AI vs Jon on the 14 calibration cards")
    c.add_argument("--out", default=os.path.join(OUT_DIR, "calibration.json"))
    r = sub.add_parser("report", help="classification, the rule, paired scores")
    r.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args(argv)
    if args.cmd == "report":
        return cmd_report(args)
    if args.check:
        return cmd_check(args)
    if args.cmd == "bounds":
        return cmd_bounds(args)
    if args.cmd == "sheet":
        return cmd_sheet(args)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    ap.error("choose a subcommand or --check")


def cmd_check(args):  # filled in with the report
    raise SystemExit("--check is not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
