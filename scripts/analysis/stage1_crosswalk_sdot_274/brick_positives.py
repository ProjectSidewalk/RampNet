"""Brick / paver crosswalk positives from Project Sidewalk crowd labels, for the next retrain (RampNet#274, s4).

The crosswalk detector misses brick crosswalks (0 of 16 PS labels in Laurens; docs/stage1_crosswalk_sdot_274.md,
"Recall probe"), and Jon (2026-11-08) wants brick-style positives. PS crowd labels of type Crosswalk carry a
`brick/cobblestone` tag; the Phase 0 census holds every city's Crosswalk rawLabels with their tags. The tag also
marks cobblestone ROADS, and validators often voted those down, so every selected label is pre-screened and
confirmed by Jon before it becomes a training positive (the arm C confirmer cannot help: it looks for paint).

  python brick_positives.py select --census <Phase 0 census OUT> --root <Stage 2 root> --exclude-panos <panos.csv ...> --out OUT
      -> OUT/selection.csv (one row per label, keyed <city>:<label_id>) and OUT/store_list.txt (<city>/<id[:2]>/<id>.jpg,
         the paths to copy from the PS pano store on makelab2)
  python brick_positives.py candidates --out OUT [--store <local store copy>]
      -> OUT/candidates.csv in `score_274.py cards` form (one card per label; panos missing locally are listed)

Selection rule (fixed before any image was looked at): tag `brick/cobblestone`, source gsv, not voted incorrect by
majority (disagree <= agree), city not already in the Stage 2 root's training cities (no split leak), pano not
in the root or in any audit draw. Seattle and Amsterdam are therefore out; Columbia SC is absent from the census
and is kept for a held-out brick test.
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot import log  # noqa: E402

TAG = "brick/cobblestone"
ROOT_CITIES = {"seattle-wa", "amsterdam", "burnaby", "teaneck-nj"}    # the Stage 2 root's cities


def keep(r):
    """The selection rule for one census rawLabels row.

    >>> keep({"tags": '["brick/cobblestone"]', "pano_source": "gsv", "agree_count": "1", "disagree_count": "0"})
    True
    >>> keep({"tags": '["brick/cobblestone"]', "pano_source": "gsv", "agree_count": "0", "disagree_count": "2"})
    False
    >>> keep({"tags": '["paint fading"]', "pano_source": "gsv", "agree_count": "3", "disagree_count": "0"})
    False
    """
    return (TAG in (r.get("tags") or "") and r.get("pano_source") == "gsv"
            and int(r.get("disagree_count") or 0) <= int(r.get("agree_count") or 0))


def root_panos(root):
    return {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "*", "*.json"))}


def cmd_select(a):
    gone = root_panos(a.root)
    for p in a.exclude_panos:
        with open(p, encoding="utf-8", newline="") as f:
            gone |= {r["pano_id"] for r in csv.DictReader(f)}
    rows, n_tag = [], 0
    for f in sorted(glob.glob(os.path.join(a.census, "rawlabels", "*_Crosswalk.csv"))):
        city = os.path.basename(f)[:-len("_Crosswalk.csv")]
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if TAG not in (r.get("tags") or ""):
                    continue
                n_tag += 1
                if city in ROOT_CITIES or r["pano_id"] in gone or not keep(r):
                    continue
                rows.append({"label_uid": f"{city}:{r['label_id']}", "city": city, "label_id": r["label_id"],
                             "pano_id": r["pano_id"],
                             "x": round(float(r["pano_x"]) / float(r["pano_width"]), 4),
                             "y": round(float(r["pano_y"]) / float(r["pano_height"]), 4),
                             "agree": r["agree_count"], "disagree": r["disagree_count"], "tags": r["tags"],
                             "user_id": r["user_id"], "capture_date": r["image_capture_date"],
                             "lat": r["latitude"], "lng": r["longitude"]})
    uids = [r["label_uid"] for r in rows]
    assert len(uids) == len(set(uids)), "label_uid not unique"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "selection.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    store = sorted({f"{r['city']}/{r['pano_id'][:2]}/{r['pano_id']}.jpg" for r in rows})
    with open(os.path.join(a.out, "store_list.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(s + "\n" for s in store)
    by = {}
    for r in rows:
        by.setdefault(r["city"], set()).add(r["pano_id"])
    log(f"select: {n_tag} {TAG} labels in the census -> {len(rows)} kept on {len(store)} panos: "
        + ", ".join(f"{c} {len(p)}" for c, p in sorted(by.items(), key=lambda kv: -len(kv[1]))))


def cmd_candidates(a):
    with open(os.path.join(a.out, "selection.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    have = None
    if a.store:
        have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.store, "*", "*", "*.jpg"))}
    out, missing = [], []
    for r in rows:
        if have is not None and r["pano_id"] not in have:
            missing.append(r["label_uid"])
            continue
        out.append({"pano_id": r["pano_id"], "x": r["x"], "y": r["y"], "conf": 1.0, "city": r["city"],
                    "mode": "brick_crowd", "year": (r["capture_date"] or "")[:4], "stratum": r["label_uid"]})
    with open(os.path.join(a.out, "candidates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(a.out, "missing_from_store.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(m + "\n" for m in missing)
    log(f"candidates: {len(out)} cards; {len(missing)} labels whose pano is not in the store copy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--census", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--exclude-panos", nargs="*", default=[])
    s.add_argument("--out", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--store", default=None, help="local copy of the store laid out <city>/<id[:2]>/<id>.jpg")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "candidates": cmd_candidates}[a.cmd](a)


if __name__ == "__main__":
    main()
