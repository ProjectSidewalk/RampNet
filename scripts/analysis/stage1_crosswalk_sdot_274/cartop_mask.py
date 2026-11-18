#!/usr/bin/env python3
"""Position rules for capture-vehicle false alarms of the crosswalk detector (#285 follow-up B).

EXPLORATORY. The card positions were already known when the rules were written, so nothing
here is a pre-registered test; any region found must be confirmed on a fresh draw before use.

Question: does a rule on a detection's image position (x, y in [0, 1]) remove the false
alarms on the capture vehicle (roof, hood, rear) without removing real crosswalks?

Arms
  0  the labeler's curb-ramp rig mask as it is: y > 0.5 + 49/180 = 0.772
     (``detectors.NADIR_MASK_DEG`` in sidewalk-auto-labeler), on every pano.
  1  a single y cut-off on every pano, swept.
  2  one box per mount (camera model x city), fitted on the mining pages only.

Scoring uses NO "car top" class: an arm is scored by the not-a-crosswalk cards it removes
against the confirmed crosswalks it removes. Jon's notes name the capture vehicle on only some
cards, so the noted subset is reported beside the main counts and decides nothing.

Fit / test
  fit   the two mining pages (cards rated to build training negatives):
        ``s4_inputs/stepA/ratings_jon_2026-11-09_carTop.csv`` (Laurens),
        ``s5_inputs/ratings_jon_2026-11-09_juneauMine.csv`` (Juneau).
  test  crosswalk_s4's rated cards on the evaluation draws:
        ``confirm_s3/<draw>/s4/ratings.csv`` and ``s5_inputs/juneau_eval/s4/ratings.csv``.
  Intervals resample Mapillary sequences (a GSV pano is its own cluster), so one vehicle's
  panos stay together.

Usage (from the repo root)::

    python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py lookup \
        --records D:/Git/sal-juneau/runs/juneau/results.jsonl ...   # once; writes camera_lookup.csv
    python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py score
    python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py figures
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARDNEG = os.path.join(REPO, "analysis_out", "stage1_crosswalk_detector_274", "hardneg")
OUT = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "cartop")

NADIR_MASK_Y = 0.5 + 49.0 / 180.0   # sidewalk-auto-labeler detectors.NADIR_MASK_DEG
SEED = 285
BOOT_N = 2000
DEDUP_R = 0.01                      # two cards on one pano this close are one detection
Y_SWEEP = [round(0.56 + 0.02 * i, 2) for i in range(13)]   # 0.56 .. 0.80

FIT_FILES = {
    "laurens_mine": "s4_inputs/stepA/ratings_jon_2026-11-09_carTop.csv",
    "juneau_mine": "s5_inputs/ratings_jon_2026-11-09_juneauMine.csv",
}
TEST_FILES = {
    "seattle_confirm": "confirm_s3/seattle/s4/ratings.csv",
    "juneau_confirm": "confirm_s3/juneau/s4/ratings.csv",
    "laurens_mly_confirm": "confirm_s3/laurens_mly/s4/ratings.csv",
    "laurens_gsv_confirm": "confirm_s3/laurens_gsv/s4/ratings.csv",
    "juneau_eval": "s5_inputs/juneau_eval/s4/ratings.csv",
}
# Human crosswalk labels (clicks, not detections): extra evidence on what a rule would hide.
LABEL_FILES = {
    "ps_labels": "confirm_s3/ps_labels/labels.csv",
    "richmond_positives": "s4_inputs/stepA/richmond_positives.csv",
    "bayonne_labels": "s4_bayonne/labels.csv",
}
# A note that names the capture vehicle. Reported as a subset only.
VEHICLE_NOTE = re.compile(r"car.?top|top of (the |a )?(capture |camera )?(car|vehicle|pickup|truck)"
                          r"|capture car|roof|hood of", re.I)

# Box search grid for arm 2 (image units; x wraps at the seam).
X_CENTRES = [i / 40.0 for i in range(40)]
X_HALF = [0.03, 0.05, 0.08, 0.12, 0.18]
Y_MINS = [round(0.50 + 0.01 * i, 2) for i in range(31)]     # 0.50 .. 0.80
DENSITY_K = 2.0   # a box must hold more than K times the share its x width alone would give it
MIN_EXCESS = 3.0  # ...by at least this many cards


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_lookup(path):
    return {r["pano_id"]: r for r in read_csv(path)} if os.path.exists(path) else {}


def mount_of(pano_id, city, lookup):
    """A mount is camera model x city: one vehicle type as one town's contributor drove it."""
    r = lookup.get(pano_id)
    cam = (r or {}).get("camera_model") or ""
    src = (r or {}).get("source") or ""
    if not r:
        cam = "unknown" if pano_id.isdigit() else "gsv"   # Mapillary ids are numeric; GSV ids are not
    elif not cam:
        cam = "gsv" if src not in ("mapillary", "panoramax") else "unknown"
    return f"{cam.lower()}|{city}"


def cluster_of(pano_id, lookup):
    r = lookup.get(pano_id) or {}
    return r.get("sequence_id") or f"pano:{pano_id}"


def wrap_dx(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def load_cards(files, lookup):
    """Rated detection cards, one per detection (cards within DEDUP_R on a pano are merged)."""
    cards = []
    for draw, rel in files.items():
        kept = []
        for r in read_csv(os.path.join(HARDNEG, rel)):
            if r.get("kind", "det") != "det" or not r.get("verdict"):
                continue
            x, y = float(r["x"]), float(r["y"])
            if any(k["pano_id"] == r["pano_id"] and wrap_dx(k["x"], x) <= DEDUP_R
                   and abs(k["y"] - y) <= DEDUP_R for k in kept):
                continue
            kept.append({
                "draw": draw, "pano_id": r["pano_id"], "x": x, "y": y, "conf": float(r["conf"]),
                "city": r["city"], "verdict": r["verdict"], "note": r.get("note") or "",
                "vehicle_note": bool(VEHICLE_NOTE.search(r.get("note") or "")),
                "mount": mount_of(r["pano_id"], r["city"], lookup),
                "cluster": cluster_of(r["pano_id"], lookup),
            })
        cards += kept
    return cards


def load_labels(lookup):
    out = []
    for name, rel in LABEL_FILES.items():
        for r in read_csv(os.path.join(HARDNEG, rel)):
            city = r.get("city") or r["label_uid"].split(":")[0]
            out.append({"set": name, "label_uid": r["label_uid"], "pano_id": r["pano_id"],
                        "x": float(r["x"]), "y": float(r["y"]), "city": city,
                        "mount": mount_of(r["pano_id"], city, lookup),
                        "cluster": cluster_of(r["pano_id"], lookup)})
    return out


# ---- the rules -------------------------------------------------------------------------------

def in_box(x, y, box):
    """box = (x_centre, x_half_width, y_min): everything at or below y_min within the x band."""
    return y >= box[2] and wrap_dx(x, box[0]) <= box[1]


def fit_box(fit_cards):
    """The x band (and its top edge y_min) where a mount's fit false alarms concentrate.

    Among boxes holding no fit crosswalk, take the one with the largest excess of false alarms
    over DENSITY_K times what its x width alone would give it: n_in - K * (2 * half_width) *
    n_band, where n_band counts the mount's fit false alarms at y >= y_min at any x. The
    comparison is within the same y band, so a box wins only if false alarms pile up in x,
    which is what a vehicle fixed to the camera predicts. None if no excess reaches MIN_EXCESS.
    """
    neg = [c for c in fit_cards if c["verdict"] == "not_crosswalk"]
    pos = [c for c in fit_cards if c["verdict"] == "crosswalk"]
    best, best_score = None, MIN_EXCESS
    for ym in Y_MINS:
        n_band = sum(c["y"] >= ym for c in neg)
        for xc in X_CENTRES:
            for hw in X_HALF:
                box = (xc, hw, ym)
                if any(in_box(c["x"], c["y"], box) for c in pos):
                    continue
                n = sum(in_box(c["x"], c["y"], box) for c in neg)
                score = n - DENSITY_K * 2 * hw * n_band
                if score > best_score:
                    best, best_score = box, score
    return best


def make_arms(fit_cards):
    arms = {"arm0_rig_mask_0.772": lambda c: c["y"] > NADIR_MASK_Y}
    for t in Y_SWEEP:
        arms[f"arm1_y>={t:.2f}"] = (lambda c, t=t: c["y"] >= t)
    boxes = {}
    by_mount = defaultdict(list)
    for c in fit_cards:
        by_mount[c["mount"]].append(c)
    for m, cs in sorted(by_mount.items()):
        b = fit_box(cs)
        if b:
            boxes[m] = b
    arms["arm2_mount_boxes"] = lambda c: c["mount"] in boxes and in_box(c["x"], c["y"], boxes[c["mount"]])
    arms["arm0+arm2"] = lambda c: c["y"] > NADIR_MASK_Y or arms["arm2_mount_boxes"](c)
    return arms, boxes


# ---- scoring ---------------------------------------------------------------------------------

def tally(cards, rule):
    t = Counter()
    for c in cards:
        v = c["verdict"]
        t[f"n_{v}"] += 1
        hit = rule(c)
        t[f"removed_{v}"] += hit
        if c["vehicle_note"]:
            t["n_vehicle_note"] += 1
            t["removed_vehicle_note"] += hit
    return t


def boot_rates(cards, rule, n=BOOT_N, seed=SEED):
    """95% intervals for (share of not_crosswalk removed, share of crosswalk removed),
    resampling clusters (Mapillary sequences; single panos for GSV)."""
    groups = defaultdict(list)
    for c in cards:
        groups[c["cluster"]].append(c)
    keys = sorted(groups)
    stat = {k: tally(groups[k], rule) for k in keys}
    rng = random.Random(seed)
    neg, pos = [], []
    for _ in range(n):
        t = Counter()
        for k in rng.choices(keys, k=len(keys)):
            t.update(stat[k])
        if t["n_not_crosswalk"]:
            neg.append(t["removed_not_crosswalk"] / t["n_not_crosswalk"])
        if t["n_crosswalk"]:
            pos.append(t["removed_crosswalk"] / t["n_crosswalk"])

    def ci(xs):
        if not xs:
            return ["", ""]
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 3), round(xs[int(0.975 * (len(xs) - 1))], 3)]
    return ci(neg), ci(pos), len(keys)


def rule_of_three(n):
    """Upper 95% bound on a rate when 0 of n were observed."""
    return round(1 - 0.05 ** (1.0 / n), 3) if n else ""


def cmd_lookup(a):
    """pano_id -> camera and sequence, for every pano in the rating and label files."""
    want = set()
    for rel in list(FIT_FILES.values()) + list(TEST_FILES.values()) + list(LABEL_FILES.values()):
        want |= {r["pano_id"] for r in read_csv(os.path.join(HARDNEG, rel))}
    rows = {}
    for path in a.records:
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line).get("pano") or {}
                pid = str(p.get("panorama_id"))
                if pid in want and pid not in rows:
                    rows[pid] = {"pano_id": pid, "source": p.get("source") or "",
                                 "camera_make": p.get("camera_make") or "",
                                 "camera_model": p.get("camera_model") or "",
                                 "sequence_id": p.get("sequence_id") or "",
                                 "capture_date": p.get("capture_date") or "",
                                 "records_file": os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)}
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "camera_lookup.csv")
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pano_id", "source", "camera_make", "camera_model",
                                          "sequence_id", "capture_date", "records_file"])
        w.writeheader()
        for pid in sorted(rows):
            w.writerow(rows[pid])
    print(f"{len(rows)} of {len(want)} panos found -> {dst}")
    print(f"not found: {len(want - set(rows))}")


def cmd_score(a):
    lookup = load_lookup(os.path.join(OUT, "camera_lookup.csv"))
    fit = load_cards(FIT_FILES, lookup)
    test = load_cards(TEST_FILES, lookup)
    labels = load_labels(lookup)
    arms, boxes = make_arms(fit)

    rows = []
    for split, cards in (("fit", fit), ("test", test)):
        groups = {"ALL": cards}
        for c in cards:
            groups.setdefault(c["mount"], []).append(c)
        for arm, rule in arms.items():
            for g, cs in groups.items():
                t = tally(cs, rule)
                neg_ci, pos_ci, n_clusters = boot_rates(cs, rule)
                rows.append({
                    "split": split, "arm": arm, "mount": g, "clusters": n_clusters,
                    "not_crosswalk": t["n_not_crosswalk"], "not_removed": t["removed_not_crosswalk"],
                    "not_removed_ci95": json.dumps(neg_ci),
                    "crosswalk": t["n_crosswalk"], "crosswalk_removed": t["removed_crosswalk"],
                    "crosswalk_removed_ci95": json.dumps(pos_ci),
                    "crosswalk_removed_upper_if_0": rule_of_three(t["n_crosswalk"]) if not t["removed_crosswalk"] else "",
                    "unsure": t["n_unsure"], "unsure_removed": t["removed_unsure"],
                    "vehicle_note": t["n_vehicle_note"], "vehicle_note_removed": t["removed_vehicle_note"],
                })
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "arms.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lab_rows = []
    lab_groups = defaultdict(list)
    for lab in labels:
        lab_groups[(lab["set"], lab["mount"])].append(lab)
    for arm, rule in arms.items():
        if arm.startswith("arm1") and arm not in ("arm1_y>=0.70", "arm1_y>=0.74"):
            continue
        for (s, m), ls in sorted(lab_groups.items()):
            lab_rows.append({"arm": arm, "label_set": s, "mount": m, "labels": len(ls),
                             "hidden": sum(rule(lab) for lab in ls)})
    with open(os.path.join(OUT, "labels_hidden.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lab_rows[0]))
        w.writeheader()
        w.writerows(lab_rows)

    with open(os.path.join(OUT, "cards.csv"), "w", encoding="utf-8", newline="") as f:
        cols = ["split", "draw", "pano_id", "x", "y", "conf", "city", "mount", "cluster", "verdict",
                "vehicle_note", "note"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for split, cards in (("fit", fit), ("test", test)):
            for c in cards:
                w.writerow({**{k: c[k] for k in cols if k != "split"}, "split": split})

    summary = {
        "nadir_mask_y": round(NADIR_MASK_Y, 4), "seed": SEED, "boot_n": BOOT_N, "density_k": DENSITY_K,
        "boxes": {m: {"x_centre": b[0], "x_half_width": b[1], "y_min": b[2]} for m, b in boxes.items()},
        "fit_cards": len(fit), "test_cards": len(test), "labels": len(labels),
        "fit_by_mount": dict(Counter(c["mount"] for c in fit)),
        "test_by_mount": dict(Counter(c["mount"] for c in test)),
        "test_vehicle_note_by_mount": dict(Counter(c["mount"] for c in test if c["vehicle_note"])),
        "test_vehicle_note_cards": [
            {k: c[k] for k in ("draw", "pano_id", "x", "y", "mount", "verdict", "note")}
            for c in test if c["vehicle_note"]],
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(json.dumps({k: summary[k] for k in ("boxes", "fit_by_mount", "test_by_mount",
                                              "test_vehicle_note_by_mount")}, indent=1))
    print("\nTEST, all mounts:")
    for r in rows:
        if r["split"] == "test" and r["mount"] == "ALL":
            print(f'  {r["arm"]:22s} not {r["not_removed"]:3d}/{r["not_crosswalk"]:3d} {r["not_removed_ci95"]:16s}'
                  f' crosswalk {r["crosswalk_removed"]:3d}/{r["crosswalk"]:3d} {r["crosswalk_removed_ci95"]:16s}'
                  f' unsure {r["unsure_removed"]}/{r["unsure"]}'
                  f' vehicle-note {r["vehicle_note_removed"]}/{r["vehicle_note"]}')


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cards = read_csv(os.path.join(OUT, "cards.csv"))
    boxes = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))["boxes"]
    mounts = [m for m, _ in Counter(c["mount"] for c in cards).most_common()
              if any(c["mount"] == m and c["vehicle_note"] == "True" for c in cards) or m in boxes]
    fig, axes = plt.subplots(1, len(mounts), figsize=(5.2 * len(mounts), 4.9), squeeze=False)
    style = {"crosswalk": ("#1a7f37", "o", "crosswalk"), "not_crosswalk": ("#cf222e", "x", "not a crosswalk"),
             "unsure": ("#9a6700", "s", "unsure")}
    for ax, m in zip(axes[0], mounts):
        for v, (col, mk, lab) in style.items():
            cs = [c for c in cards if c["mount"] == m and c["verdict"] == v]
            ax.scatter([float(c["x"]) for c in cs], [float(c["y"]) for c in cs], s=18, c=col, marker=mk,
                       linewidths=1, alpha=0.7, label=f"{lab} ({len(cs)})")
        vn = [c for c in cards if c["mount"] == m and c["vehicle_note"] == "True"]
        ax.scatter([float(c["x"]) for c in vn], [float(c["y"]) for c in vn], s=70, facecolors="none",
                   edgecolors="black", linewidths=0.8, label=f"note names the vehicle ({len(vn)})")
        ax.axhline(NADIR_MASK_Y, color="black", ls="--", lw=1)
        ax.text(0.01, NADIR_MASK_Y + 0.004, "rig mask, y 0.772", fontsize=8, va="top")
        if m in boxes:
            b = boxes[m]
            for lo, hi in ([(b["x_centre"] - b["x_half_width"], b["x_centre"] + b["x_half_width"])]):
                for s, e in (((lo, hi),) if 0 <= lo and hi <= 1 else ((max(lo, 0), min(hi, 1)), ((lo % 1, 1) if lo < 0 else (0, hi % 1)))):
                    ax.fill_between([s, e], b["y_min"], 1.0, color="#0969da", alpha=0.12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.95, 0.45)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("x (0.5 = compass bearing of the image centre)")
        ax.set_ylabel("y (0.5 = horizon, larger = nearer the camera)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle("Rated crosswalk-detector cards by mount (fit and test pooled); blue = fitted arm-2 box", fontsize=10)
    fig.tight_layout()
    dst = os.path.join(REPO, "docs", "figures", "crosswalk_followups_285")
    os.makedirs(dst, exist_ok=True)
    fig.savefig(os.path.join(dst, "cartop_by_mount.png"), dpi=130)
    print(os.path.join(dst, "cartop_by_mount.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lookup", help="write camera_lookup.csv from labeler results.jsonl files")
    lk.add_argument("--records", nargs="+", required=True)
    sub.add_parser("score", help="score every arm on the fit and test cards")
    sub.add_parser("figures", help="x-y scatter per mount")
    a = ap.parse_args()
    {"lookup": cmd_lookup, "score": cmd_score, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
