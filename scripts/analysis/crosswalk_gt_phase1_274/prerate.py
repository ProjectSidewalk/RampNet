"""Blind AI pre-rate of the multi-city crosswalk GT sheet (RampNet#274 Phase 1 item 2).

  python prerate.py sheets --sheet .../sheet.csv --log .../archive_log.json --archive <archive> --out <archive>/prerate
      one composite PNG per row (in the archive, never the repo) + `index.csv` (gt_id, has_label, image) and
      `RUBRIC.md`. The composite shows: top left the newest ortho year, 30 m wide; top right its centre 15 m,
      2x; bottom left the older year (or blank); bottom right the crowd's street-level crop (red ring = label).
      Markers: blue ring = OSM node, dashed blue = 2 m and 5 m, red ring + yellow line = crowd label. The
      composite and the index carry NOTHING about the row's OSM stratum, tags, city or label tier, so the rater
      sees only what a human sees on the page minus the facts panel.
  python prerate.py collect --parts <csv> [<csv> ...] --sheet .../sheet.csv --out analysis_out/.../prerate_ai.csv
      concatenates the rater's part files, checks every row is rated exactly once with legal codes, writes the
      committed AI file. That file is read ONLY by `gt_page.py merge`, never by `build` (the page is blind).

Phase 0 measured the AI-vs-human paint agreement at 0.98 (118/120) given the same imagery; this is the same
kind of first pass, now blind to the OSM class, for the agreement to be computed after Jon rates.
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cities import CITIES, merc  # noqa: E402
from gt_page import TILE_GROUND_M, read_csv  # noqa: E402

PAINTED = ("yes", "ring", "no", "unclear")
LABEL = ("yes", "near", "end", "off", "na", "nolabel")
CELL = 560

RUBRIC = """# Crosswalk GT rubric (blind AI pre-rate; same text as the human page, version phase1-2026-11-02)

Each image is one row. Layout: top-left = newest aerial (30 m wide, north up); top-right = its centre 15 m at 2x;
bottom-left = an older aerial year (grey if none); bottom-right = the street-level crop the crowd labeled from
(red ring = the crowd label; grey if the row has no crowd label). Blue ring = an OpenStreetMap crossing node;
dashed blue rings = 2 m and 5 m from it. Red ring + yellow line on the aerials = the crowd label (absent on
rows without one).

Q1 `painted` (every row):
  yes     a painted pedestrian crossing (zebra, ladder, transverse lines, coloured surface) passes through the
          node or within ~2 m of it (inner ring), in EITHER aerial year OR the street crop. Faded but visible = yes.
  ring    painted crossing within the 5 m ring but clearly not at the node.
  no      no paint within 5 m in any image. An unmarked crossing (kerb cuts, no paint) is no, not unclear.
  unclear the images cannot tell (trees, shadow, vehicle, imagery seam over the spot).

Q2 `label` (rows WITH a red crowd label; write `nolabel` for rows without one). Judge the position ALONG the
crossing; offset up/down the street is placement error.
  yes   on the paint at the centre
  near  on or just off the paint, in line with the middle, not toward an end (3-5 m off has been judged near;
        ~6 m off is off)
  end   at a curb end of this crossing
  off   not on this crossing (another crossing or leg, or far from any)
  na    no paint to judge against (unmarked crossing, or no crossing)

`bike` = yes if the crowd label sits on a green bike crossing rather than a pedestrian one (keep the position
code against that bike crossing). `note` = a few words when anything is unusual.
"""


def draw_markers(im, r, ext_merc, ground, scale):
    """Draw node, rings and label on an ortho image `im` (CELL px square, centred on the node, `scale` = zoom)."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    nlat, nlng = float(r["node_lat"]), float(r["node_lng"])
    cx, cy = merc(nlat, nlng)

    def to_px(lat, lng):
        x, y = merc(lat, lng)
        return (x - cx) / ext_merc * CELL * scale + CELL / 2, (cy - y) / ext_merc * CELL * scale + CELL / 2

    ppm = CELL / ground * scale
    nx, ny = CELL / 2, CELL / 2
    if r["label_id"]:
        lx, ly = to_px(float(r["label_lat"]), float(r["label_lng"]))
        d.line([(nx, ny), (lx, ly)], fill=(255, 225, 77), width=2)
        d.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], outline=(232, 50, 43), width=3)
    d.ellipse([nx - 9, ny - 9, nx + 9, ny + 9], outline=(31, 95, 255), width=3)
    for rad in (2, 5):
        rr = rad * ppm
        d.ellipse([nx - rr, ny - rr, nx + rr, ny + rr], outline=(31, 95, 255), width=1)


def sheets(a):
    from PIL import Image, ImageDraw
    rows = read_csv(a.sheet)
    log = json.load(open(a.log, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "RUBRIC.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(RUBRIC)
    idx = []
    for r in rows:
        acc = log["cities"][r["city"]]["accepted"][r["node_id"]]
        years = [s["year"] for s in CITIES[r["city"]]["imagery"]]
        ext = TILE_GROUND_M / math.cos(math.radians(float(r["node_lat"])))
        canvas = Image.new("RGB", (2 * CELL + 6, 2 * CELL + 6), (128, 128, 128))
        newest = acc["tiles"][str(years[-1])]
        im = Image.open(os.path.join(a.archive, newest["file"])).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        full = im.copy()
        draw_markers(full, r, ext, TILE_GROUND_M, 1)
        canvas.paste(full, (0, 0))
        src = Image.open(os.path.join(a.archive, newest["file"])).convert("RGB")
        w = src.width
        z = src.crop((w // 4, w // 4, 3 * w // 4, 3 * w // 4)).resize((CELL, CELL), Image.LANCZOS)
        draw_markers(z, r, ext, TILE_GROUND_M, 2)
        canvas.paste(z, (CELL + 6, 0))
        if len(years) > 1 and str(years[0]) in acc["tiles"]:
            o = Image.open(os.path.join(a.archive, acc["tiles"][str(years[0])]["file"])).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            draw_markers(o, r, ext, TILE_GROUND_M, 1)
            canvas.paste(o, (0, CELL + 6))
        if acc.get("crop"):
            c = Image.open(os.path.join(a.archive, acc["crop"]["file"])).convert("RGB")
            m = acc["crop"]["marker"] or {}
            c.thumbnail((CELL, CELL), Image.LANCZOS)
            dd = ImageDraw.Draw(c)
            mx, my, rad = m.get("x", 0.5) * c.width, m.get("y", 0.5) * c.height, max(8, c.width // 40)
            dd.ellipse([mx - rad, my - rad, mx + rad, my + rad], outline=(232, 50, 43), width=4)
            canvas.paste(c, (CELL + 6, CELL + 6 + (CELL - c.height) // 2))
        path = os.path.join(a.out, f"{r['gt_id']}.png")
        canvas.save(path)
        idx.append({"gt_id": r["gt_id"], "has_label": "yes" if r["label_id"] else "no", "image": path.replace(os.sep, "/")})
    with open(os.path.join(a.out, "index.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gt_id", "has_label", "image"], lineterminator="\n")
        w.writeheader()
        w.writerows(idx)
    print(f"{len(idx)} composites + index.csv + RUBRIC.md -> {a.out}")


def collect(a):
    sheet = {r["gt_id"]: r for r in read_csv(a.sheet)}
    got = {}
    for p in a.parts:
        for r in read_csv(p):
            k = r["gt_id"].strip()
            assert k in sheet, f"{p}: unknown row {k}"
            assert k not in got, f"{k} rated twice"
            pa, la = r["painted"].strip().lower(), r["label"].strip().lower()
            assert pa in PAINTED, f"{k}: painted {pa!r}"
            assert la in LABEL, f"{k}: label {la!r}"
            assert (la == "nolabel") == (not sheet[k]["label_id"]), f"{k}: label code vs crowd label presence"
            got[k] = {"gt_id": k, "painted": pa, "label": la, "bike": (r.get("bike") or "").strip().lower(),
                      "note": (r.get("note") or "").strip(), "rater": (r.get("rater") or "").strip()}
    missing = sorted(set(sheet) - set(got))
    assert not missing, f"{len(missing)} rows unrated, e.g. {missing[:5]}"
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gt_id", "painted", "label", "bike", "note", "rater"], lineterminator="\n")
        w.writeheader()
        w.writerows(got[k] for k in sorted(got))
    print(f"{len(got)} rows -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--sheet", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--archive", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--parts", nargs="+", required=True)
    c.add_argument("--sheet", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    (sheets if a.cmd == "sheets" else collect)(a)


if __name__ == "__main__":
    main()
