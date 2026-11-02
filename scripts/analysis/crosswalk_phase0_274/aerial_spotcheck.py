"""Spot-check the AI's aerial ratings (RampNet#274 item 2) in the browser, then fold the human
verdicts back into the sheet.

  python aerial_spotcheck.py build --sheet analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated_ai.csv \
      --rawlabels <census cache>/out/rawlabels/seattle-wa_Crosswalk.csv \
      --out analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html
  python aerial_spotcheck.py merge --sheet <the same csv> --verdicts ~/Downloads/aerial_verdicts_seattle-wa.json \
      --out analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated.csv

`build` writes one self-contained page (no imagery inside it): each row's 36 m crops (Web Mercator metres; about 24 m on the ground at
Seattle) are requested live
from King County's 2023 AND 2025 ortho export services when the row is shown (side by side, so a
crossing painted or removed between the flights is seen), with the OSM node, the crowd label and the
2 m / 5 m rings drawn over both (ground metres since 2026-11-02; Mercator, so about 1.48x too small at
Seattle, when the 120 human verdicts were saved); a third panel shows the street-level crop Project Sidewalk serves for
the crowd label (`/cropImage/Crosswalk/<label_id>/metadata` hands out a short-lived signed URL, CORS
open), which is what the labeler actually saw. --rawlabels (the census's cached v3 rawLabels CSV for the
city) adds the pano capture month, the label date, human agree/disagree counts and a Street View link
to each row's facts; without it those lines are simply absent. --verdicts <export.json> embeds a verdict
export: on load the page adopts every embedded row newer than the browser's copy (localStorage is per
browser profile, which once split Jon's ratings across two Chrome profiles), so a rebuilt page carries
its state with it.

`build --archive <dir>` renders the same page from an ARCHIVE of those fetches instead of the live
services (the layout `scripts/analysis/crosswalk_phase0_274/archive_spotcheck.py` writes: ortho/<label_id>_<year>_<36|18>m.png,
crops/<label_id>.png + .metadata.json, manifest.json); --out should then be inside the archive (its
page/ dir), since the tile paths are written relative to the page. That is the replay path for when
the King County export service or the signed crops are gone. The Seattle archive lives on the lab share
(backed up), on makelab1/makelab2 at /projects/makeabilitylab/RampNet/archives/crosswalk_phase0_274/aerial_spotcheck_seattle-wa/
(its README.md has the replay command); see archive_spotcheck.py's docstring. The AI's two ratings are pre-selected; Enter accepts them,
Y/N/U and 1-4/0 override them. Verdicts persist in localStorage and "Export verdicts" downloads them as
JSON, in the gt_gallery.py pattern. `merge` adds `human_painted`, `human_label`, `human_flag`, `human_bike`
(the B tag: the crowd labeled a green bike crossing beside the node, which a paint detector would fire on
too; the position code then describes the label against that bike crossing, so the counts set those rows
aside), `human_note`, `human_imagery` (what the rater saw when the row was saved: `2023` for a verdict from
before the 2025 and crop panels existed, else `2023+2025+crop`) and `human_ts` columns to the sheet, prints
agreement with the AI pass overall and per OSM class, and is the only step that touches the repo's tables.
"""
import argparse
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ON_CROSSING = ("yes", "near")


def read_sheet(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def human_votes(validations_json):
    """(agree, disagree) counted over HUMAN validators only, from the rawLabels `validations` column."""
    try:
        vs = json.loads(validations_json or "[]")
    except json.JSONDecodeError:
        return 0, 0
    hs = [v for v in vs if v.get("validator_type") == "Human"]
    return sum(v.get("validation") == "Agree" for v in hs), sum(v.get("validation") == "Disagree" for v in hs)


def build(a):
    html = render(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}; open it in a browser")


def render(a, tpl=None):
    """The page as a string. `tpl` overrides the template text (crosswalk_gt_phase1_274/gt_page.py build-phase0
    passes the city-generic template rendered for a city; for Seattle it is this directory's template, byte for byte)."""
    rows = read_sheet(a.sheet)
    raw = {}
    if a.rawlabels:
        raw = {r["label_id"]: r for r in read_sheet(a.rawlabels)}
    sheet = os.path.basename(a.sheet).split(".")[0].replace("aerial_sample_", "")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "i": i, "uid": r["label_uid"], "node": r["node_id"], "cls": r["osm_marking_class"],
            "ct": r["crossing_tag"], "cm": r["crossing_markings_tag"],
            "nlat": float(r["node_lat"]), "nlng": float(r["node_lng"]),
            "llat": float(r["label_lat"]), "llng": float(r["label_lng"]),
            "dist": float(r["dist_m"]), "axis": r["axis"],
            "ai_p": r["rating_painted_crossing_at_node"], "ai_l": r["rating_label_at_centre"], "note": r["notes"],
        })
        lab = raw.get(r["label_uid"].split(":")[-1])
        if lab:
            agree, dis = human_votes(lab.get("validations"))
            out[-1].update({"pdate": lab.get("image_capture_date") or "", "ldate": (lab.get("time_created") or "")[:10],
                            "agree": agree, "dis": dis, "purl": lab.get("pano_url") or ""})
    if a.rawlabels:
        n = sum("pdate" in o for o in out)
        print(f"rawlabels: {n} of {len(out)} rows matched a label")
        assert n == len(out), "every sheet row should be in the city's rawLabels pull"
    rev = json.load(open(a.review, encoding="utf-8")) if a.review else {}
    seed = "null"
    if a.verdicts:
        vj = json.load(open(a.verdicts, encoding="utf-8"))
        vs = verdict_list(vj)
        seed = json.dumps(vs, separators=(",", ":"), ensure_ascii=False)
        print(f"seed: {len(vs)} verdicts embedded from {a.verdicts}")
    archive = "null"
    archived = ""
    if a.archive:
        crops = {}
        for o in out:
            lid = o["uid"].split(":")[-1]
            mp = os.path.join(a.archive, "crops", f"{lid}.metadata.json")
            if os.path.exists(mp):
                m = json.load(open(mp, encoding="utf-8"))
                img = next((f"{lid}{ext}" for ext in (".png", ".jpg") if os.path.exists(os.path.join(a.archive, "crops", f"{lid}{ext}"))), None)
                if img:
                    crops[lid] = {"crop_marker": m.get("crop_marker"), "file": img}
        n_tiles = sum(os.path.exists(os.path.join(a.archive, "ortho", f"{o['uid'].split(':')[-1]}_{y}_{w}m.png"))
                      for o in out for y in (2023, 2025) for w in (36, 18))
        print(f"archive: {n_tiles} of {4 * len(out)} ortho tiles, {len(crops)} of {len(out)} crops present")
        root = os.path.relpath(os.path.abspath(a.archive), os.path.dirname(os.path.abspath(a.out))).replace(os.sep, "/")
        archive = json.dumps({"root": root, "crops": crops}, separators=(",", ":"))
        archived = " · ARCHIVED COPY"
    if tpl is None:
        tpl = open(os.path.join(HERE, "aerial_spotcheck_template.html"), encoding="utf-8").read()
    return (tpl.replace("__ROWS__", json.dumps(out, separators=(",", ":"))).replace("__SHEET__", sheet)
            .replace("__SERVER__", a.server.rstrip("/")).replace("__ARCHIVE__", archive).replace("__ARCHIVED__", archived)
            .replace("__REVIEW__", json.dumps(rev.get("rows", {}))).replace("__REVIEW_MADE__", rev.get("made_utc", "")).replace("__SEED__", seed))


def is_bike(v):
    """The B tag. (A note-text fallback was used before the tag existed; every bike row in the final export carries the tag.)"""
    return bool(v.get("bike"))


def verdict_list(vj):
    """The export's verdicts as a list (the page exports a list; a hand-edited file may hold a dict by row)."""
    vs = vj["verdicts"]
    return vs if isinstance(vs, list) else list(vs.values())


def merge(a):
    rows = read_sheet(a.sheet)
    vj = json.load(open(a.verdicts, encoding="utf-8"))
    vl = verdict_list(vj)
    by_i = {int(v["i"]): v for v in vl}
    assert len(by_i) == len(vl), "duplicate row index in the verdicts export"
    assert all(0 <= i < len(rows) for i in by_i), "verdict row index outside the sheet"
    for i, r in enumerate(rows):
        assert by_i.get(i, {}).get("uid", r["label_uid"]) == r["label_uid"], f"row {i} uid mismatch"
    fields = list(rows[0].keys()) + ["human_painted", "human_label", "human_flag", "human_bike", "human_note", "human_imagery", "human_ts"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            v = by_i.get(i, {})
            w.writerow({**r, "human_painted": v.get("painted", ""), "human_label": v.get("label", ""),
                        "human_flag": "yes" if v.get("flag") else "", "human_bike": "yes" if v and is_bike(v) else "",
                        "human_note": v.get("note", ""), "human_imagery": (v.get("imagery") or ("2023" if v else "")),
                        "human_ts": v.get("ts", "")})
    rated = [(r, by_i[i]) for i, r in enumerate(rows) if i in by_i]
    n = len(rated)
    print(f"{n} of {len(rows)} rows have a human verdict -> {a.out}")
    if not n:
        return
    ag_p = sum(v["painted"] == r["rating_painted_crossing_at_node"] for r, v in rated)
    ag_l = sum(v["label"] == r["rating_label_at_centre"] for r, v in rated)
    ag_lc = sum((v["label"] in ON_CROSSING) == (r["rating_label_at_centre"] in ON_CROSSING) for r, v in rated)
    print(f"agree with AI: painted {ag_p}/{n} = {ag_p / n:.2f}; label (5 classes) {ag_l}/{n} = {ag_l / n:.2f}; "
          f"label (within 5 m vs not) {ag_lc}/{n} = {ag_lc / n:.2f}")
    ped = lambda v: v["label"] in ON_CROSSING and not is_bike(v)   # within 5 m of the node on a PEDESTRIAN crossing
    rat = [v for _, v in rated if v["label"] != "na"]
    print(f"human placement: {sum(ped(v) for v in rat)} of {len(rat)} ratable within 5 m on a pedestrian crossing, "
          f"{sum(v['label'] == 'end' for v in rat)} at an end, {sum(v['label'] == 'off' for v in rat)} off, "
          f"{sum(v['label'] in ON_CROSSING and is_bike(v) for v in rat)} on a bike crossing; {n - len(rat)} n/a")
    im = collections.Counter((v.get("imagery") or "2023") for _, v in rated)
    print("imagery the human verdicts were saved on:", dict(im))
    print("| class | rated | painted=yes human | painted=yes AI | <= 5 m on a pedestrian crossing, human | <= 5 m AI | bike crossing | flagged |")
    print("|---|---|---|---|---|---|---|---|")
    for cls, name in (("marked", "marked"), ("unknown", "untagged"), ("unmarked", "unmarked")):
        s = [(r, v) for r, v in rated if r["osm_marking_class"] == cls]
        if not s:
            continue
        print(f"| {name} | {len(s)} | {sum(v['painted'] == 'yes' for _, v in s)} | "
              f"{sum(r['rating_painted_crossing_at_node'] == 'yes' for r, _ in s)} | "
              f"{sum(ped(v) for _, v in s)} | "
              f"{sum(r['rating_label_at_centre'] in ON_CROSSING for r, _ in s)} | {sum(is_bike(v) for _, v in s)} | "
              f"{sum(bool(v.get('flag')) for _, v in s)} |")
    # The doc's two per-class tables (AI first pass, human rating), category by category.
    cats_p, cats_l = ("yes", "unclear", "no"), ("yes", "near", "end", "off", "na")
    for who in ("AI", "human"):
        print(f"{who} per class: painted {'/'.join(cats_p)} | label {'/'.join(cats_l)}"
              + (" | on a bike crossing (within 5 m, B tag)" if who == "human" else ""))
        for cls, name in (("marked", "marked"), ("unmarked", "unmarked"), ("unknown", "untagged")):
            s = [(r, v) for r, v in rated if r["osm_marking_class"] == cls]
            if who == "AI":
                p = collections.Counter(r["rating_painted_crossing_at_node"] for r, _ in s)
                lab = collections.Counter(r["rating_label_at_centre"] for r, _ in s)
                tail = ""
            else:
                p = collections.Counter(v["painted"] for _, v in s)
                bike = lambda v: v["label"] in ON_CROSSING and is_bike(v)  # merge's placement line's rule
                lab = collections.Counter(v["label"] for _, v in s if not bike(v))
                tail = f" | {sum(bike(v) for _, v in s)}"
            print(f"  {name:9s} n={len(s)}  {'/'.join(str(p[c]) for c in cats_p)} | "
                  f"{'/'.join(str(lab[c]) for c in cats_l)}{tail}")
    dis = [(i, r, by_i[i]) for i, r in enumerate(rows) if i in by_i
           and not (by_i[i]["painted"] == r["rating_painted_crossing_at_node"] and by_i[i]["label"] == r["rating_label_at_centre"])]
    if dis:
        print("disagreements:")
        for i, r, v in dis:
            print(f"  #{i} {r['label_uid']} {r['osm_marking_class']}: painted {r['rating_painted_crossing_at_node']}->{v['painted']}, "
                  f"label {r['rating_label_at_centre']}->{v['label']}{'  ' + v['note'] if v.get('note') else ''}")
    conf = collections.Counter((r["rating_label_at_centre"], v["label"]) for r, v in rated)
    print("label confusion (AI -> human):", dict(conf))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sheet", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--verdicts", help="a verdicts export to embed: on load the page adopts any embedded row newer than the browser's own copy, so the state no longer depends on which browser profile opens the file")
    b.add_argument("--review", help="json {rows: {row index: reason}} of verdicts to re-examine after a rule change; shown on the row and listed by the 're-examine' order")
    b.add_argument("--archive", help="render from an archive_spotcheck.py directory instead of the live services (write --out inside it)")
    b.add_argument("--rawlabels", help="the census's cached v3 rawLabels CSV for the city (pano date, label date, votes, Street View link)")
    b.add_argument("--server", default="https://sidewalk-sea.cs.washington.edu",
                   help="the city's Project Sidewalk server, for the label crops (the canonical host, not a redirecting alias)")
    m = sub.add_parser("merge")
    m.add_argument("--sheet", required=True)
    m.add_argument("--verdicts", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    (build if a.cmd == "build" else merge)(a)


if __name__ == "__main__":
    main()
