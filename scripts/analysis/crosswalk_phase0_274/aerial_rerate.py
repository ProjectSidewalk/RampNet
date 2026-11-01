"""Blind re-rate of the AI first pass's no / unclear rows on the Seattle aerial sheet, and its agreement
with the human verdicts (#274 Phase 0, aerial reference).

Why: the AI first pass rated the 2023 King County ortho only, while the human pass saw 2023, 2025 and the
Project Sidewalk label crop. Where they disagreed the table could not tell "the AI was wrong" from "the AI
never saw the image that settled it". The re-rate gives the AI the same three images for the 24 rows the
first pass called no or unclear (a yes row cannot be talked out of its paint by more imagery). The rater
must not have seen the human verdicts; the 2026-11-01 re-rate was done by a fresh session (Claude,
claude-opus-5-5) from the sheets this script draws, and its verdicts are committed as
`analysis_out/crosswalk_phase0_274/aerial_rerate_seattle-wa.csv` (row = 0-based index in the sheet).

  python aerial_rerate.py sheets --archive <replay archive> --sheet <rated_ai.csv> --out <dir>
      one JPEG per row: 2023 at 18 m, 2025 at 18 m (2x the 36 m tile: the archive's 2025 18 m tiles
      are blank, every row), 2025 at 36 m, the label crop. Blue = OSM node with 2 m / 5 m rings,
      red = crowd label, yellow line between them; red circle on the crop = the label.
  python aerial_rerate.py score --rerate <aerial_rerate_seattle-wa.csv> --rated <rated.csv>
      agreement before and after the re-rate, painted / five-class position / within-5-m-on-crossing,
      with and without the four bike-crossing rows. Prints the numbers the doc quotes.

The replay archive's location is in archive_spotcheck.py's docstring (lab share; not in this repo).
"""
import argparse
import csv
import math
import pathlib

PX = 720  # the archive's ortho exports are 720 px square, bbox centred on the OSM node (Web Mercator)


def merc(lat, lng):
    r = 6378137.0
    return r * math.radians(lng), r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def to_px(row, extent_m):
    """Pixel of the crowd label in a PX-square tile `extent_m` Mercator metres wide, centred on the node."""
    cx, cy = merc(float(row["node_lat"]), float(row["node_lng"]))
    x, y = merc(float(row["label_lat"]), float(row["label_lng"]))
    h = extent_m / 2
    return (x - (cx - h)) / extent_m * PX, ((cy + h) - y) / extent_m * PX


def rerate_rows(sheet):
    """The rows the first pass called no or unclear: (0-based index, row)."""
    return [(i, r) for i, r in enumerate(sheet) if r["rating_painted_crossing_at_node"] != "yes"]


def cmd_sheets(a):
    from PIL import Image, ImageDraw
    sheet = list(csv.DictReader(open(a.sheet, encoding="utf-8")))
    arch, out = pathlib.Path(a.archive), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def mark(tile, lx, ly, extent_m, merc_scale, tag):
        d = ImageDraw.Draw(tile)
        c, ppm = tile.width / 2, tile.width / extent_m * merc_scale
        d.line([c, c, lx, ly], fill=(255, 225, 77), width=2)
        d.ellipse([c - 9, c - 9, c + 9, c + 9], outline=(31, 95, 255), width=3)
        for rad in (2, 5):
            d.ellipse([c - rad * ppm, c - rad * ppm, c + rad * ppm, c + rad * ppm], outline=(31, 95, 255), width=1)
        d.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], outline=(232, 50, 43), width=3)
        d.rectangle([8, 8, 230, 30], fill="black")
        d.text((14, 13), tag, fill=(255, 255, 0))
        return tile.resize((512, 512))

    for i, r in rerate_rows(sheet):
        lid = r["label_uid"].split(":")[1]
        k = 1 / math.cos(math.radians(float(r["node_lat"])))  # Mercator metres per ground metre
        o = lambda y, e: Image.open(arch / "ortho" / f"{lid}_{y}_{e}m.png").convert("RGB")
        lx18, ly18 = to_px(r, 18)
        lx36, ly36 = to_px(r, 36)
        a23 = mark(o(2023, 18), lx18, ly18, 18, k, "2023, 18 m wide")
        t25 = o(2025, 36)
        z25 = t25.crop((PX // 4, PX // 4, 3 * PX // 4, 3 * PX // 4)).resize((PX, PX), Image.LANCZOS)
        b25 = mark(z25, (lx36 - PX / 4) * 2, (ly36 - PX / 4) * 2, 18, k, "2025, 18 m (2x zoom)")
        c25 = mark(t25, lx36, ly36, 36, k, "2025, 36 m wide")
        crop = Image.open(arch / "crops" / f"{lid}.png").convert("RGB")
        d, rr = ImageDraw.Draw(crop), crop.width / 40
        x, y = crop.width / 2, crop.height / 2  # the crop is centred on the label (crop_marker 0.5, 0.5)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(232, 50, 43), width=4)
        sheet_img = Image.new("RGB", (512 * 3 + 768, 512), "white")
        for j, t in enumerate((a23, b25, c25)):
            sheet_img.paste(t, (512 * j, 0))
        sheet_img.paste(crop.resize((768, 512)), (1536, 0))
        sheet_img.save(out / f"row{i:03d}.jpg", quality=88)
    print(f"wrote {len(rerate_rows(sheet))} sheets to {out}")


def cmd_score(a):
    rated = list(csv.DictReader(open(a.rated, encoding="utf-8")))
    new = {int(r["i"]): r for r in csv.DictReader(open(a.rerate, encoding="utf-8"))}
    on = lambda p: p in ("yes", "near")
    for scope, keep in (("re-rated rows", lambda i: i in new), ("all rows", lambda i: True)):
        for bikes in (True, False):
            rows = [(i, r) for i, r in enumerate(rated) if keep(i) and (bikes or not r["human_bike"])]
            ai_p = lambda i, r: new[i]["painted"] if i in new else r["rating_painted_crossing_at_node"]
            ai_l = lambda i, r: new[i]["position"] if i in new else r["rating_label_at_centre"]
            cnt = lambda f: sum(f(i, r) for i, r in rows)
            n = len(rows)
            print(f"{scope:13s} {'with bikes ' if bikes else 'bikes aside'} n={n:3d}"
                  f"  painted {cnt(lambda i, r: r['rating_painted_crossing_at_node'] == r['human_painted'])}"
                  f" -> {cnt(lambda i, r: ai_p(i, r) == r['human_painted'])}"
                  f"  five-class {cnt(lambda i, r: r['rating_label_at_centre'] == r['human_label'])}"
                  f" -> {cnt(lambda i, r: ai_l(i, r) == r['human_label'])}"
                  f"  on-crossing {cnt(lambda i, r: on(r['rating_label_at_centre']) == on(r['human_label']))}"
                  f" -> {cnt(lambda i, r: on(ai_l(i, r)) == on(r['human_label']))}")
    print("re-rated rows that still disagree with the human verdict:")
    for i, m in sorted(new.items()):
        h = rated[i]
        if m["painted"] != h["human_painted"] or m["position"] != h["human_label"]:
            print(f"  row {i:3d} {h['label_uid']}: painted {m['painted']}/{h['human_painted']}"
                  f"  position {m['position']}/{h['human_label']}  bike {m['bike'] or '-'}/{'B' if h['human_bike'] else '-'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)
    s1 = s.add_parser("sheets")
    s1.add_argument("--archive", required=True)
    s1.add_argument("--sheet", required=True)
    s1.add_argument("--out", required=True)
    s2 = s.add_parser("score")
    s2.add_argument("--rerate", required=True)
    s2.add_argument("--rated", required=True)
    a = p.parse_args()
    {"sheets": cmd_sheets, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
