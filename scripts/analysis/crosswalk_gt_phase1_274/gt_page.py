"""Build the blind multi-city crosswalk GT page, rebuild the Phase 0 Seattle page from the city table, and merge
exported verdicts back into the sheet (RampNet#274 Phase 1 item 2).

  python gt_page.py build --sheet analysis_out/crosswalk_gt_phase1_274/sheet.csv \
      --log analysis_out/crosswalk_gt_phase1_274/archive_log.json \
      --archive D:/Git/sal-scratch-keep/phase1-crosswalk-gt \
      --out D:/Git/sal-scratch-keep/phase1-crosswalk-gt/page/crosswalk_gt.html [--verdicts export.json]

  python gt_page.py build-phase0 --city seattle-wa --sheet ... --rawlabels ... --review ... --verdicts ... --out X
      (the Phase 0 page, from crosswalk_phase0_274/aerial_spotcheck.py's builder, with every Seattle-specific
      string of its template now read from cities.py; for Seattle the output is byte-identical to the committed
      analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html: the parity check)

  python gt_page.py merge --sheet .../sheet.csv --verdicts <export.json> [--prerate .../prerate_ai.csv] --out .../sheet.rated_<rater>.csv

`build` writes ONE page for all cities (decision: one page, one localStorage key, one export, one rater box,
with a city filter; four pages would mean four exports to keep apart and four places for a verdict to get lost,
which is how the Phase 0 ratings once split across two browser profiles). The page renders from the replay
archive only (relative paths to its ortho/ and crops/ files), so --out must be inside the archive; it needs no
network except the font and the links the rater chooses to follow. It is BLIND: the row payload carries no
stratum, no OSM class or tags, no label tier and no AI rating (`ROW_KEYS` is the whole payload, and the test
asserts the AI file's columns never reach it). The OSM node link is there for a rater who wants it after
deciding; the page says so.

`merge` is the only step that reads the AI pre-rate; it adds `human_*` and (with --prerate) `ai_*` columns, and
prints per-city / per-stratum counts and human-vs-AI agreement.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "crosswalk_phase0_274"))
from cities import CITIES, merc_per_ground  # noqa: E402

RUBRIC_VERSION = "phase1-2026-11-02"
PX = 800
TILE_GROUND_M = 30.0
ROW_KEYS = ("id", "city", "nlat", "nlng", "llat", "llng", "dist", "node", "uid", "panels", "crop", "pdate", "ldate",
            "purl", "ext", "ground")
ON_CROSSING = ("yes", "near")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- Phase 0 page, generalized (parity)

def phase0_template(city):
    """The Phase 0 spot-check template with the city's strings filled in from cities.py."""
    c = CITIES[city]
    srcs = c["imagery"]
    assert len(srcs) == 2, "the Phase 0 page lays out exactly two ortho years"
    y0, y1 = srcs[0]["year"], srcs[1]["year"]
    svc = srcs[0]["url"].replace(str(y0), "${y}")
    assert srcs[1]["url"] == svc.replace("${y}", str(y1)), "both years must share one URL pattern"
    mins = {s["year"]: s["min_tile_m"] for s in srcs if s.get("min_tile_m")}
    min_js = "{" + ", ".join(f"{y}: {m}" for y, m in mins.items()) + "}"
    tpl = open(os.path.join(HERE, "phase0_template_generic.html"), encoding="utf-8", newline="").read()
    for k, v in (("__CITY_NAME__", c["name"]), ("__PROVIDER_LINE__", c["provider_line"]),
                 ("__SERVICE_NAME__", c.get("service_name", c["name"])), ("__CREDIT0__", srcs[0]["credit"]),
                 ("__CREDIT1__", srcs[1]["credit"]), ("__SVC_JS__", svc), ("__MIN_TILE_M__", min_js),
                 ("__Y0__", str(y0)), ("__Y1__", str(y1))):
        tpl = tpl.replace(k, v)
    return tpl


def build_phase0(a):
    import aerial_spotcheck as p0
    ns = types.SimpleNamespace(sheet=a.sheet, rawlabels=a.rawlabels, review=a.review, verdicts=a.verdicts,
                               archive=a.archive, out=a.out, server=a.server or CITIES[a.city]["server"])
    html = p0.render(ns, tpl=phase0_template(a.city))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {a.out}")


# ---------------------------------------------------------------- the blind Phase 1 page

def page_rows(sheet, log, archive, out):
    """The page payload: one dict per sheet row with exactly ROW_KEYS (nothing about stratum, tags or AI)."""
    root = os.path.relpath(os.path.abspath(archive), os.path.dirname(os.path.abspath(out))).replace(os.sep, "/")
    rows = []
    for r in sheet:
        acc = log["cities"][r["city"]]["accepted"][r["node_id"]]
        nlat, nlng = float(r["node_lat"]), float(r["node_lng"])
        panels = []
        for src in CITIES[r["city"]]["imagery"]:
            t = acc["tiles"].get(str(src["year"]))
            panels.append({"y": src["year"], "credit": src["credit"], "f": f"{root}/{t['file']}" if t else None,
                           "why": None if t else acc["blank_years"].get(str(src["year"]), "")})
        crop = None
        if acc.get("crop"):
            m = acc["crop"]["marker"] or {}
            crop = {"f": f"{root}/{acc['crop']['file']}", "x": m.get("x", 0.5), "y": m.get("y", 0.5),
                    "w": m.get("width", 849), "h": m.get("height", 566)}
        has = bool(r["label_id"])
        rows.append({"id": r["gt_id"], "city": r["city"], "nlat": nlat, "nlng": nlng,
                     "llat": float(r["label_lat"]) if has else None, "llng": float(r["label_lng"]) if has else None,
                     "dist": float(r["dist_m"]) if has else None, "node": r["node_id"], "uid": r["label_uid"],
                     "panels": panels, "crop": crop, "pdate": r["image_capture_date"], "ldate": r["time_created"],
                     "purl": r["pano_url"], "ext": TILE_GROUND_M * merc_per_ground(nlat), "ground": TILE_GROUND_M})
        assert tuple(rows[-1]) == ROW_KEYS
    return rows


def build(a):
    sheet = read_csv(a.sheet)
    log = json.load(open(a.log, encoding="utf-8"))
    rows = page_rows(sheet, log, a.archive, a.out)
    cities = [c for c in CITIES if any(r["city"] == c for r in rows)]
    cmeta = {c: {"name": CITIES[c]["name"], "provider": CITIES[c]["provider_line"], "terms": CITIES[c]["terms"]}
             for c in cities}
    seed = "null"
    if a.verdicts:
        vs = []
        for p in a.verdicts:
            vs += json.load(open(p, encoding="utf-8"))["verdicts"]
        seed = json.dumps(vs, separators=(",", ":"), ensure_ascii=False)
        print(f"seed: {len(vs)} verdicts embedded")
    tpl = open(os.path.join(HERE, "gt_page_template.html"), encoding="utf-8", newline="").read()
    html = tpl
    for k, v in (("__ROWS__", json.dumps(rows, separators=(",", ":"), ensure_ascii=False)),
                 ("__CITIES__", json.dumps(cmeta, ensure_ascii=False)), ("__SEED__", seed),
                 ("__SHEET_SHA256__", sha256_file(a.sheet)), ("__SHEET__", "crosswalk_gt_phase1_274"),
                 ("__RUBRIC_VERSION__", RUBRIC_VERSION), ("__PX__", str(PX)), ("__N__", str(len(rows))),
                 ("__CITY_LIST__", ", ".join(f"{CITIES[c]['name']} {sum(r['city'] == c for r in rows)}" for c in cities)),
                 ("__CITY_OPTIONS__", "".join(f'<option value="{c}">{CITIES[c]["name"]}</option>' for c in cities))):
        html = html.replace(k, v)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(rows)} rows ({', '.join(cities)}); open it in a browser")


# ---------------------------------------------------------------- merge

def merge(a):
    sheet = read_csv(a.sheet)
    by_id = {r["gt_id"]: r for r in sheet}
    ex = json.load(open(a.verdicts, encoding="utf-8"))
    if ex.get("sheet_sha256") and ex["sheet_sha256"] != sha256_file(a.sheet):
        print("WARNING: the export was made against a different sheet.csv (sha256 differs)")
    vs = {v["id"]: v for v in ex["verdicts"]}
    assert len(vs) == len(ex["verdicts"]), "duplicate row id in the export"
    for k, v in vs.items():
        assert k in by_id, f"verdict for unknown row {k}"
        assert v.get("node") in (None, by_id[k]["node_id"]), f"{k}: node mismatch"
    ai = {}
    if a.prerate:
        ai = {r["gt_id"]: r for r in read_csv(a.prerate)}
    fields = list(sheet[0].keys()) + ["human_painted", "human_label", "human_bike", "human_flag", "human_note",
                                      "human_rater", "human_ts"] + (["ai_painted", "ai_label", "ai_note"] if ai else [])
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in sheet:
            v = vs.get(r["gt_id"], {})
            row = {**r, "human_painted": v.get("painted", ""), "human_label": v.get("label", ""),
                   "human_bike": "yes" if v.get("bike") else "", "human_flag": "yes" if v.get("flag") else "",
                   "human_note": v.get("note", ""), "human_rater": v.get("rater", ""), "human_ts": v.get("ts", "")}
            if ai:
                x = ai.get(r["gt_id"], {})
                row.update({"ai_painted": x.get("painted", ""), "ai_label": x.get("label", ""), "ai_note": x.get("note", "")})
            w.writerow(row)
    print(f"{len(vs)} of {len(sheet)} rows rated (rater {ex.get('rater')}) -> {a.out}")
    report(sheet, vs, ai)


def report(sheet, vs, ai):
    """Per city x stratum: Q1 counts (human, and AI on the same rows) and Q2 placement; then agreement."""
    strata = ("marked", "untagged", "unmarked", "osm_only")
    print("| city | stratum | rated | Q1 yes | R (within 5 m) | no | cannot tell | label on crossing (1/2) | end | off | n/a | bike |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for city in dict.fromkeys(r["city"] for r in sheet):
        for s in strata:
            rs = [vs[r["gt_id"]] for r in sheet if r["city"] == city and r["stratum"] == s and r["gt_id"] in vs]
            if not rs:
                continue
            p = collections.Counter(v["painted"] for v in rs)
            lab = collections.Counter(v["label"] for v in rs if not v.get("bike"))
            print(f"| {city} | {s} | {len(rs)} | {p['yes']} | {p['ring']} | {p['no']} | {p['unclear']} | "
                  f"{lab['yes'] + lab['near'] if s != 'osm_only' else '-'} | {lab['end']} | {lab['off']} | {lab['na']} | "
                  f"{sum(bool(v.get('bike')) for v in rs)} |")
    if not ai:
        return
    both = [(vs[k], ai[k]) for k in vs if k in ai]
    n = len(both)
    if not n:
        return
    p4 = sum(h["painted"] == x["painted"] for h, x in both)
    pyes = sum((h["painted"] == "yes") == (x["painted"] == "yes") for h, x in both)
    paint = lambda v: v in ("yes", "ring")
    pany = sum(paint(h["painted"]) == paint(x["painted"]) for h, x in both)
    lab = [(h, x) for h, x in both if h["label"] != "nolabel"]
    l5 = sum(h["label"] == x["label"] for h, x in lab)
    lon = sum((h["label"] in ON_CROSSING) == (x["label"] in ON_CROSSING) for h, x in lab)
    print(f"human vs AI on {n} rows: Q1 four-way {p4}/{n} = {p4 / n:.2f}; at the node (Y vs rest) {pyes}/{n} = {pyes / n:.2f}; "
          f"paint within 5 m (Y or R vs rest) {pany}/{n} = {pany / n:.2f}; Q2 five-way {l5}/{len(lab)}, "
          f"on the crossing (1/2 vs rest) {lon}/{len(lab)}")
    conf = collections.Counter((x["painted"], h["painted"]) for h, x in both)
    print("Q1 confusion (AI -> human):", dict(sorted(conf.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sheet", required=True)
    b.add_argument("--log", required=True)
    b.add_argument("--archive", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--verdicts", nargs="*", help="verdict export(s) to embed (newest timestamp per row wins on load)")
    p = sub.add_parser("build-phase0")
    p.add_argument("--city", default="seattle-wa")
    p.add_argument("--sheet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rawlabels")
    p.add_argument("--review")
    p.add_argument("--verdicts")
    p.add_argument("--archive")
    p.add_argument("--server")
    m = sub.add_parser("merge")
    m.add_argument("--sheet", required=True)
    m.add_argument("--verdicts", required=True)
    m.add_argument("--prerate")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": build, "build-phase0": build_phase0, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
