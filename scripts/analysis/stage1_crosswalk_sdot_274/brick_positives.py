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
  score_274.py cards --candidates OUT/candidates.csv --pano-dir <prepped> --blind --no-big --crops-out <crops> --no-page
  score_274.py sheets --crops <crops> --out <sheets>    # blind AI readers -> OUT/ai_parts/ai_part_{1..4}.csv
  python brick_positives.py prescreen --out OUT        # -> OUT/prescreen.csv (needs OUT/sheets.csv from <sheets>)
  python brick_positives.py page --out OUT             # -> OUT/jon_selection.csv, then `cards --only` it

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


AI_VERDICTS = {"crosswalk", "not_crosswalk", "unsure"}
AI_STYLES = {"brick", "brick_edge_lines", "paint", "none", "unclear"}


def cmd_prescreen(a):
    """Join the blind AI readers' per-cell verdicts (sheet, cell, verdict, style, reason) to cards and labels via
    sheets.csv; refuses a missing, duplicate or unknown cell. Writes prescreen.csv (one row per label)."""
    with open(os.path.join(a.out, "sheets.csv"), encoding="utf-8", newline="") as f:
        cells = {(r["sheet"], int(r["cell"])): r["card_id"] for r in csv.DictReader(f)}
    with open(os.path.join(a.out, "candidates.csv"), encoding="utf-8", newline="") as f:
        cands = list(csv.DictReader(f))
    uid = {f"det:{c['pano_id']}:{float(c['x']):.3f}:{float(c['y']):.3f}": c for c in cands}
    got = {}
    for path in sorted(glob.glob(os.path.join(a.out, "ai_parts", "*.csv"))):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["sheet"].strip(), int(r["cell"]))
                v, st = r["verdict"].strip(), r["style"].strip()
                if k not in cells or k in got or v not in AI_VERDICTS or st not in AI_STYLES:
                    raise SystemExit(f"{path}: bad or duplicate row {r}")
                got[k] = (v, st, r.get("reason", "").strip())
    missing = set(cells) - set(got)
    if missing:
        raise SystemExit(f"{len(missing)} cells have no AI verdict, e.g. {sorted(missing)[:3]}")
    rows = []
    for k in sorted(cells):
        c = uid[cells[k]]
        rows.append({"card_id": cells[k], "label_uid": c["stratum"], "city": c["city"], "pano_id": c["pano_id"],
                     "x": c["x"], "y": c["y"], "sheet": k[0], "cell": k[1], "ai_verdict": got[k][0],
                     "ai_style": got[k][1], "ai_reason": got[k][2]})
    with open(os.path.join(a.out, "prescreen.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    log(f"prescreen: {len(rows)} labels; " + ", ".join(f"{k[0]}/{k[1]} {n}" for k, n in
                                                      Counter((r["ai_verdict"], r["ai_style"]) for r in rows).most_common()))


SAMPLE_SEED = 274
TRUST_MIN = 0.95   # Jon must agree with this share of the sampled AI crosswalk calls (per style) to trust the rest


def cmd_page(a):
    """Jon's confirmation selection, fixed before he rates: every AI not_crosswalk/unsure card, plus a seeded
    sample of the AI's crosswalk calls in each style (--n-brick over brick + brick_edge_lines, --n-paint over paint).
    Rule stated in advance: where Jon agrees with at least TRUST_MIN of a style's sampled crosswalk calls, that
    style's unsampled AI crosswalk calls become positives without a second look; otherwise he rates them too.
    Writes jon_selection.csv (card_id, label_uid, group)."""
    import random
    with open(os.path.join(a.out, "prescreen.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SAMPLE_SEED)
    pick = [(r, "ai_" + r["ai_verdict"]) for r in rows if r["ai_verdict"] != "crosswalk"]
    for group, styles, n in (("sample_brick", {"brick", "brick_edge_lines"}, a.n_brick),
                             ("sample_paint", {"paint"}, a.n_paint)):
        pool = sorted((r for r in rows if r["ai_verdict"] == "crosswalk" and r["ai_style"] in styles),
                      key=lambda r: r["card_id"])
        pick += [(r, group) for r in rng.sample(pool, min(n, len(pool)))]
    with open(os.path.join(a.out, "jon_selection.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "label_uid", "group"])
        w.writerows((r["card_id"], r["label_uid"], g) for r, g in pick)
    from collections import Counter
    log(f"page: {len(pick)} cards: " + ", ".join(f"{g} {n}" for g, n in Counter(g for _, g in pick).items()))


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
    pr = sub.add_parser("prescreen")
    pr.add_argument("--out", required=True)
    pg = sub.add_parser("page")
    pg.add_argument("--n-brick", type=int, default=120)
    pg.add_argument("--n-paint", type=int, default=40)
    pg.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "candidates": cmd_candidates, "prescreen": cmd_prescreen, "page": cmd_page}[a.cmd](a)


if __name__ == "__main__":
    main()
