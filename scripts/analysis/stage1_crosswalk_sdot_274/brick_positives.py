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
  python brick_positives.py tally --ratings <Jon's export> --out OUT   # -> OUT/positives.csv, OUT/tally.json
  python brick_positives.py crowd-candidates --labels <ps_s4_labels.csv> --out <candidates.csv>   # s4 Step A
  python brick_positives.py crowd-split --candidates <candidates.csv> --ratings <Jon's export> --out OUT

Selection rule (fixed before any image was looked at): tag `brick/cobblestone`, source gsv, not voted incorrect by
majority (disagree <= agree), city not already in the Stage 2 root's training cities (no split leak), pano not
in the root or in any audit draw. Seattle and Amsterdam are therefore out; Columbia SC is absent from the census
and is kept for a held-out brick test.
"""
import argparse
import csv
import glob
import json
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


def trust_styles(sel, prescreen, verdicts):
    """Apply the rule fixed in `page`: for each sampled style group, the share of the sampled AI crosswalk calls Jon
    also called crosswalk; the group is trusted when that share is at least TRUST_MIN.

    >>> sel = {"a": "sample_paint", "b": "sample_paint", "c": "ai_unsure"}
    >>> trust_styles(sel, {}, {"a": "crosswalk", "b": "unsure", "c": "crosswalk"})
    {'sample_paint': (1, 2, False)}
    """
    out = {}
    for cid, g in sel.items():
        if g.startswith("sample_"):
            k, n, _ = out.get(g, (0, 0, False))
            out[g] = (k + (verdicts[cid] == "crosswalk"), n + 1, False)
    return {g: (k, n, k / n >= TRUST_MIN) for g, (k, n, _) in sorted(out.items())}


def cmd_tally(a):
    """Jon's ratings + the trust rule -> positives.csv (one row per accepted label) and tally.json. A label is a
    positive when Jon rated it crosswalk, or when he did not rate it, the AI called it crosswalk, and its style group
    passed the trust rule. Refuses a ratings file that does not cover jon_selection.csv exactly."""
    import json
    from collections import Counter
    with open(os.path.join(a.out, "prescreen.csv"), encoding="utf-8", newline="") as f:
        pre = {r["card_id"]: r for r in csv.DictReader(f)}
    with open(os.path.join(a.out, "jon_selection.csv"), encoding="utf-8", newline="") as f:
        sel = {r["card_id"]: r["group"] for r in csv.DictReader(f)}
    with open(a.ratings, encoding="utf-8", newline="") as f:
        rat = list(csv.DictReader(f))
    ids = [r["card_id"] for r in rat]
    if len(ids) != len(set(ids)) or set(ids) != set(sel):
        raise SystemExit("ratings do not cover jon_selection.csv exactly once")
    jon = {r["card_id"]: r["verdict"] for r in rat}
    note = {r["card_id"]: r["note"] for r in rat}
    trust = trust_styles(sel, pre, jon)
    group_of = {"brick": "sample_brick", "brick_edge_lines": "sample_brick", "paint": "sample_paint"}
    rows, why = [], Counter()
    for cid, p in sorted(pre.items(), key=lambda kv: kv[1]["label_uid"]):
        if cid in jon:
            ok, basis = jon[cid] == "crosswalk", f"jon_{jon[cid]}"
        else:
            g = group_of.get(p["ai_style"]) if p["ai_verdict"] == "crosswalk" else None
            ok, basis = bool(g and trust[g][2]), f"ai_trusted_{p['ai_style']}" if g else "ai_untrusted"
        why[(basis, ok)] += 1
        if ok:
            rows.append({"label_uid": p["label_uid"], "city": p["city"], "pano_id": p["pano_id"], "x": p["x"],
                         "y": p["y"], "basis": basis, "ai_verdict": p["ai_verdict"], "ai_style": p["ai_style"],
                         "jon_note": note.get(cid, "")})
    uids = [r["label_uid"] for r in rows]
    assert len(uids) == len(set(uids)), "label_uid not unique"
    with open(os.path.join(a.out, "positives.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cross = Counter((sel[c], jon[c]) for c in sel)
    summary = {"trust_min": TRUST_MIN, "trust": {g: {"agree": k, "n": n, "trusted": t} for g, (k, n, t) in trust.items()},
               "jon_by_group": {f"{g}/{v}": n for (g, v), n in sorted(cross.items())},
               "basis": {f"{b}/{'kept' if ok else 'dropped'}": n for (b, ok), n in sorted(why.items())},
               "labels": len(pre), "positives": len(rows), "positive_panos": len({r["pano_id"] for r in rows}),
               "positives_by_city": dict(Counter(r["city"] for r in rows).most_common())}
    with open(os.path.join(a.out, "tally.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    log(f"tally: {len(rows)} of {len(pre)} labels are positives on {summary['positive_panos']} panos; trust "
        + ", ".join(f"{g} {k}/{n}{' ok' if t else ' FAILED'}" for g, (k, n, t) in trust.items()))


CROWD_MODE = {"columbia-sc": "s4_test_brick", "richmond-va": "s4_train_mapillary"}


def cmd_crowd_candidates(a):
    """s4 Step A: the PS labels from prod_s4_labels.sh as cards (one per label, stratum = label_uid). Refuses a
    label_uid or card position (pano, x and y at 3 decimals, the card id) that is not unique."""
    with open(a.labels, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({"pano_id": r["pano_id"], "x": round(float(r["pano_x"]) / float(r["width"]), 4),
                    "y": round(float(r["pano_y"]) / float(r["height"]), 4), "conf": 1.0, "city": r["city"],
                    "mode": CROWD_MODE[r["city"]], "year": (r["capture_date"] or "")[:4],
                    "stratum": f"{r['city']}:{r['label_id']}"})
    uids = [o["stratum"] for o in out]
    cards = [f"{o['pano_id']}:{o['x']:.3f}:{o['y']:.3f}" for o in out]
    assert len(uids) == len(set(uids)), "label_uid not unique"
    assert len(cards) == len(set(cards)), "two labels share a card position"
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    log(f"crowd-candidates: {len(out)} cards: " + ", ".join(f"{m} {n}" for m, n in Counter(o['mode'] for o in out).items()))


def cmd_crowd_split(a):
    """s4 Step A, after Jon rates the crowd page: Columbia labels he called crosswalk become the P1 GT
    (columbia_gt.csv), Richmond labels he called crosswalk become training positives (richmond_positives.csv).
    Refuses ratings that do not cover candidates.csv exactly once. Writes crowd_split.json (the verdict table)."""
    import pandas as pd
    cand = pd.read_csv(a.candidates, dtype=str, keep_default_na=False)
    rat = pd.read_csv(a.ratings, dtype=str, keep_default_na=False)
    assert rat.stratum.is_unique and cand.stratum.is_unique, "label_uid not unique"
    assert set(rat.stratum) == set(cand.stratum), "ratings do not cover candidates.csv exactly"
    assert (rat.verdict != "").all(), "unrated cards"
    m = cand.merge(rat[["stratum", "verdict", "note", "rated_at"]], on="stratum", validate="one_to_one")
    m = m.rename(columns={"stratum": "label_uid"})
    cols = ["label_uid", "city", "pano_id", "x", "y", "year", "note"]
    out = {}
    for mode, name in (("s4_test_brick", "columbia_gt.csv"), ("s4_train_mapillary", "richmond_positives.csv")):
        keep = m[(m["mode"] == mode) & (m.verdict == "crosswalk")][cols]
        keep.to_csv(os.path.join(a.out, name), index=False)
        out[mode] = {"crosswalk": int(len(keep)), "file": name,
                     **{v: int(n) for v, n in m[m["mode"] == mode].verdict.value_counts().items()}}
    with open(os.path.join(a.out, "crowd_split.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"crowd-split: {out}")


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
    t = sub.add_parser("tally")
    t.add_argument("--ratings", required=True, help="Jon's export from the brick_confirm page")
    t.add_argument("--out", required=True)
    cc = sub.add_parser("crowd-candidates")
    cc.add_argument("--labels", required=True, help="ps_s4_labels_<date>.csv from prod_s4_labels.sh")
    cc.add_argument("--out", required=True, help="candidates.csv")
    cs = sub.add_parser("crowd-split")
    cs.add_argument("--candidates", required=True, help="crowd_candidates.csv")
    cs.add_argument("--ratings", required=True, help="Jon's export from the s4_crowd page")
    cs.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "candidates": cmd_candidates, "prescreen": cmd_prescreen, "page": cmd_page,
     "tally": cmd_tally, "crowd-candidates": cmd_crowd_candidates, "crowd-split": cmd_crowd_split}[a.cmd](a)


if __name__ == "__main__":
    main()
