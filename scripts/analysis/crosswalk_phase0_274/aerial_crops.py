"""Aerial crops for the Seattle rating sheet (RampNet#274 item 2): one 36 m x 36 m crop per row from King
County's 2023 ortho map service (3 in/px, EagleView, licensed to the county: view-only, never committed),
centred on the OSM node, with the node (blue) and the crowd label (red) marked, plus 2x2 contact sheets.

Usage: python aerial_crops.py --sheet OUT/aerial_sample_seattle-wa.csv --out CROPS   (CROPS is never committed)
"""
import argparse
import csv
import math
import os
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

SERVICE = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2023/MapServer/export"
HALF_M = 18.0
PX = 540
R = 6378137.0


def merc(lat, lng):
    return R * math.radians(lng), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def fetch(path, cx, cy):
    bbox = f"{cx - HALF_M},{cy - HALF_M},{cx + HALF_M},{cy + HALF_M}"
    url = (f"{SERVICE}?bbox={bbox}&bboxSR=3857&imageSR=3857&size={PX},{PX}&format=png&f=image")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data[:4] == b"\x89PNG":
                open(path, "wb").write(data)
                return True
        except Exception as exc:  # noqa: BLE001
            print("  miss", exc)
        time.sleep(3)
    return False


def to_px(cx, cy, lat, lng):
    x, y = merc(lat, lng)
    return (x - (cx - HALF_M)) / (2 * HALF_M) * PX, ((cy + HALF_M) - y) / (2 * HALF_M) * PX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", default="crops")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sheets"), exist_ok=True)
    rows = list(csv.DictReader(open(a.sheet, encoding="utf-8")))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for i, r in enumerate(rows):
        nlat, nlng = float(r["node_lat"]), float(r["node_lng"])
        cx, cy = merc(nlat, nlng)
        raw = os.path.join(a.out, "raw", f"{i:03d}.png")
        if not os.path.exists(raw):
            if not fetch(raw, cx, cy):
                print(i, "FAILED")
                continue
            time.sleep(0.5)
        im = Image.open(raw).convert("RGB")
        d = ImageDraw.Draw(im)
        nx, ny = to_px(cx, cy, nlat, nlng)
        lx, ly = to_px(cx, cy, float(r["label_lat"]), float(r["label_lng"]))
        d.line([(nx, ny), (lx, ly)], fill=(255, 255, 0), width=2)
        d.ellipse([nx - 7, ny - 7, nx + 7, ny + 7], outline=(0, 90, 255), width=3)
        d.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], outline=(255, 0, 0), width=3)
        tag = f"#{i}  {r['osm_marking_class']}  {r['dist_m']} m"
        d.rectangle([0, 0, 12 + 12 * len(tag), 30], fill=(0, 0, 0))
        d.text((6, 3), tag, fill=(255, 255, 255), font=font)
        im.save(os.path.join(a.out, f"{i:03d}.png"))
        tiles.append((i, im))
        print(i, r["label_uid"], r["osm_marking_class"], flush=True)
    for s in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (2 * PX + 4, 2 * PX + 4), (255, 255, 255))
        for k, (i, im) in enumerate(tiles[s:s + 4]):
            sheet.paste(im, ((k % 2) * (PX + 4), (k // 2) * (PX + 4)))
        sheet.save(os.path.join(a.out, "sheets", f"sheet_{s // 4:02d}.png"))
    print("sheets:", math.ceil(len(tiles) / 4))


if __name__ == "__main__":
    main()
