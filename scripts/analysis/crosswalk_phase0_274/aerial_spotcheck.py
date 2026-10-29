"""Spot-check the AI's aerial ratings (RampNet#274 item 2) in the browser, then fold the human
verdicts back into the sheet.

  python aerial_spotcheck.py build --sheet analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated_ai.csv \
      --out analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html
  python aerial_spotcheck.py merge --sheet <the same csv> --verdicts ~/Downloads/aerial_verdicts_seattle-wa.json \
      --out analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated.csv

`build` writes one self-contained page (no imagery inside it): each row's 36 m crop is requested live
from King County's 2023 or 2025 ortho export service when the row is shown, with the OSM node, the crowd
label and the 2 m / 5 m rings drawn over it. The AI's two ratings are pre-selected; Enter accepts them,
Y/N/U and 1-4/0 override them. Verdicts persist in localStorage and "Export verdicts" downloads them as
JSON, in the gt_gallery.py pattern. `merge` adds `human_painted`, `human_label`, `human_flag`, `human_note`
columns to the sheet, prints agreement with the AI pass overall and per OSM class, and is the only step
that touches the repo's tables.
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


def build(a):
    rows = read_sheet(a.sheet)
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
    tpl = open(os.path.join(HERE, "aerial_spotcheck_template.html"), encoding="utf-8").read()
    html = tpl.replace("__ROWS__", json.dumps(out, separators=(",", ":"))).replace("__SHEET__", sheet)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(out)} rows; open it in a browser")


def merge(a):
    rows = read_sheet(a.sheet)
    vj = json.load(open(a.verdicts, encoding="utf-8"))
    by_i = {int(v["i"]): v for v in vj["verdicts"]}
    for i, r in enumerate(rows):
        assert by_i.get(i, {}).get("uid", r["label_uid"]) == r["label_uid"], f"row {i} uid mismatch"
    fields = list(rows[0].keys()) + ["human_painted", "human_label", "human_flag", "human_note"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            v = by_i.get(i, {})
            w.writerow({**r, "human_painted": v.get("painted", ""), "human_label": v.get("label", ""),
                        "human_flag": "yes" if v.get("flag") else "", "human_note": v.get("note", "")})
    rated = [(r, by_i[i]) for i, r in enumerate(rows) if i in by_i]
    n = len(rated)
    print(f"{n} of {len(rows)} rows have a human verdict -> {a.out}")
    if not n:
        return
    ag_p = sum(v["painted"] == r["rating_painted_crossing_at_node"] for r, v in rated)
    ag_l = sum(v["label"] == r["rating_label_at_centre"] for r, v in rated)
    ag_lc = sum((v["label"] in ON_CROSSING) == (r["rating_label_at_centre"] in ON_CROSSING) for r, v in rated)
    print(f"agree with AI: painted {ag_p}/{n} = {ag_p / n:.2f}; label (5 classes) {ag_l}/{n} = {ag_l / n:.2f}; "
          f"label (on-crossing vs not) {ag_lc}/{n} = {ag_lc / n:.2f}")
    print("| class | rated | painted=yes human | painted=yes AI | on-crossing human | on-crossing AI | flagged |")
    print("|---|---|---|---|---|---|---|")
    for cls in ("marked", "unknown", "unmarked"):
        s = [(r, v) for r, v in rated if r["osm_marking_class"] == cls]
        if not s:
            continue
        print(f"| {cls} | {len(s)} | {sum(v['painted'] == 'yes' for _, v in s)} | "
              f"{sum(r['rating_painted_crossing_at_node'] == 'yes' for r, _ in s)} | "
              f"{sum(v['label'] in ON_CROSSING for _, v in s)} | "
              f"{sum(r['rating_label_at_centre'] in ON_CROSSING for r, _ in s)} | {sum(bool(v.get('flag')) for _, v in s)} |")
    dis = [(i, r, by_i[i]) for i, r in enumerate(rows) if i in by_i and not by_i[i].get("agree")]
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
    m = sub.add_parser("merge")
    m.add_argument("--sheet", required=True)
    m.add_argument("--verdicts", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    (build if a.cmd == "build" else merge)(a)


if __name__ == "__main__":
    main()
