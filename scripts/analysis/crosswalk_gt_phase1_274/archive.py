"""Fetch the imagery for the multi-city crosswalk GT round into a local REPLAY ARCHIVE, pick the final rows, and
write the rating sheet (RampNet#274 Phase 1 item 2).

For each city, each stratum, candidates are taken in `cand_rank` order (from `draw.py`): one ortho tile per
imagery year (TILE_GROUND_M ground metres square, PX px, Web Mercator, centred on the OSM node), checked for a
blank answer (`is_blank`: King County's 2025 service returns a uniform 200 PNG below ~30 m, and a coverage edge
returns a white or transparent fill), and, for a row with a crowd label, the street-level crop Project
Sidewalk serves for it (`/cropImage/Crosswalk/<label_id>/metadata` -> signed `image_url`). A candidate is
accepted when its NEWEST year's tile is real imagery (an older year may be blank: the page says so); otherwise
it is skipped and the reason recorded, and the next candidate is tried, until the stratum's quota is met. The
crop is never a reason to skip (the page says when it is missing).

Every request is a GET, sequential, SPACING_S apart, with a contact User-Agent. Imagery and crops land ONLY in
the archive (never in the repo); the archive's manifest.json records url, http status, content type, sha256,
bytes and fetch time for each file, and SHA256SUMS covers the directory (as Phase 0's `archive_spotcheck.py`).

Layout under --archive:
  manifest.json, SHA256SUMS, README.md
  inputs/<census paths>                      the rawLabels pulls and OSM tiles the draw read (copied)
  ortho/<city>/<node_id>_<year>.<png|jpg>
  crops/<city>/<label_id>.<png|jpg> + <label_id>.metadata.json
  page/                                      the built page(s) (gt_page.py build --out inside it)

Committed (no pixels): `<out>/sheet.csv` (the final rows, with `gt_id`), `<out>/archive_log.json` (per
candidate: accepted or skipped and why; per accepted row the tile/crop sha256s, so a later re-fetch can be
proven identical or shown to have changed).

Final order: all accepted rows of every city are shuffled together with `random.Random(f"{SEED}:order")` and
numbered `gt_id` = r000, r001, ...; the page shows rows in that order, so neither the city nor the stratum is
read off the row number.

Usage:
  python archive.py --census OUT --out analysis_out/crosswalk_gt_phase1_274 --archive D:/Git/sal-scratch-keep/phase1-crosswalk-gt
  python archive.py ... --sums-only
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cities import CITIES, DRAW_CITIES, UA, tile_url  # noqa: E402
from draw import FIELDS, SEED, STRATA, sha256_file  # noqa: E402

TILE_GROUND_M = 30.0
PX = 800
SPACING_S = 0.35
SHEET_FIELDS = ["gt_id"] + FIELDS + ["years", "years_blank", "has_crop", "crop_w", "crop_h", "crop_x", "crop_y"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def get(url, timeout=90):
    """(status, content_type, bytes); 3 tries on 5xx / network errors."""
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code, e.headers.get("Content-Type", ""), e.read()
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}: {last}")


def is_blank(data):
    """(blank, why). Uniform, transparent or white/black fill counts as blank; so does an undecodable body."""
    from PIL import Image, ImageStat
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # noqa: BLE001
        return True, f"not an image ({exc.__class__.__name__})"
    if im.mode in ("RGBA", "LA", "P"):
        a = im.convert("RGBA").getchannel("A")
        hist = a.histogram()
        if hist[0] > 0.5 * im.width * im.height:
            return True, "mostly transparent"
    g = im.convert("L")
    sd = ImageStat.Stat(g).stddev[0]
    if sd < 3:
        return True, f"uniform (sd {sd:.1f})"
    hist = g.histogram()
    if max(hist) > 0.6 * im.width * im.height:
        return True, f"one grey value covers {max(hist) / (im.width * im.height):.0%}"
    return False, f"sd {sd:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True, help="the committed analysis_out dir holding candidates_<city>.csv")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--sums-only", action="store_true")
    a = ap.parse_args()
    arc = a.archive
    if a.sums_only:
        print("SHA256SUMS:", write_sums(arc))
        return
    mpath = os.path.join(arc, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {"files": {}}
    files = manifest["files"]
    lpath = os.path.join(a.out, "archive_log.json")
    log = json.load(open(lpath, encoding="utf-8")) if os.path.exists(lpath) else {"cities": {}}

    def save():
        for p, obj in ((mpath, manifest), (lpath, log)):
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(obj, f, indent=1, sort_keys=True)
                f.write("\n")
            os.replace(tmp, p)

    def record(rel, url, status, ctype, note=None):
        p = os.path.join(arc, rel)
        files[rel] = {"url": url, "http_status": status, "content_type": ctype, "sha256": sha256_file(p),
                      "bytes": os.path.getsize(p), "fetched_utc": now()}
        if note:
            files[rel]["note"] = note

    def fetch_to(rel, url, image=True):
        """Fetch url into the archive at rel (extension fixed from the content type); (ok, rel, body, why)."""
        st, ct, body = get(url)
        time.sleep(SPACING_S)
        if st != 200 or (image and not ct.startswith("image/")):
            return False, rel, body, f"HTTP {st} {ct} {len(body)} B"
        if image:
            rel = os.path.splitext(rel)[0] + (".jpg" if "jpeg" in ct else ".png")
        os.makedirs(os.path.dirname(os.path.join(arc, rel)), exist_ok=True)
        with open(os.path.join(arc, rel), "wb") as f:
            f.write(body)
        return True, rel, body, ""

    os.makedirs(arc, exist_ok=True)
    cities = a.only.split(",") if a.only else DRAW_CITIES
    draw_manifest = json.load(open(os.path.join(a.out, "draw_manifest.json"), encoding="utf-8"))
    for city in cities:
        # inputs, copied so the archive alone can repeat the draw
        for rel in draw_manifest["cities"][city]["inputs_sha256"]:
            dst = os.path.join(arc, "inputs", rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(a.census, rel), dst)
                record("inputs/" + rel, "file://census/" + rel, None, None)
        cfg = CITIES[city]
        years = [s["year"] for s in cfg["imagery"]]
        quota = draw_manifest["cities"][city]["quota"]
        with open(os.path.join(a.out, f"candidates_{city}.csv"), encoding="utf-8", newline="") as f:
            cands = list(csv.DictReader(f))
        clog = log["cities"].setdefault(city, {"accepted": {}, "skipped": {}})
        for s in STRATA:
            got = sum(1 for k, v in clog["accepted"].items() if v["stratum"] == s)
            for c in [c for c in cands if c["stratum"] == s]:
                key = c["node_id"]
                if got >= quota[s]:
                    break
                if key in clog["accepted"] or key in clog["skipped"]:
                    continue
                lat, lng = float(c["node_lat"]), float(c["node_lng"])
                tiles, blank = {}, {}
                for src in cfg["imagery"]:
                    url = tile_url(src, lat, lng, ground_m=TILE_GROUND_M, px=PX)
                    ok, rel, body, why = fetch_to(f"ortho/{city}/{key}_{src['year']}.png", url)
                    if ok:
                        b, bwhy = is_blank(body)
                        if b:
                            os.remove(os.path.join(arc, rel))
                            ok, why = False, "blank: " + bwhy
                    if ok:
                        record(rel, url, 200, "image/" + ("jpeg" if rel.endswith(".jpg") else "png"))
                        tiles[str(src["year"])] = {"file": rel, "sha256": files[rel]["sha256"]}
                    else:
                        blank[str(src["year"])] = why
                if str(years[-1]) not in tiles:
                    clog["skipped"][key] = {"stratum": s, "cand_rank": int(c["cand_rank"]), "why": blank, "utc": now()}
                    print(f"  {city} {s} #{c['cand_rank']} node {key}: SKIP {blank}", flush=True)
                    save()
                    continue
                crop = None
                if c["label_id"]:
                    lid = c["label_id"]
                    murl = f"{cfg['server']}/cropImage/Crosswalk/{lid}/metadata"
                    ok, mrel, body, why = fetch_to(f"crops/{city}/{lid}.metadata.json", murl, image=False)
                    if ok:
                        record(mrel, murl, 200, "application/json")
                        meta = json.loads(body)
                        cm = meta.get("crop_marker")
                        if isinstance(cm, str):   # some servers send it as a Python-repr string
                            cm = json.loads(cm.replace("'", '"'))
                        iurl = cfg["server"] + meta["image_url"]
                        ok, irel, body, why = fetch_to(f"crops/{city}/{lid}.png", iurl)
                        if ok:
                            record(irel, iurl.split("?")[0], 200, "image/" + ("jpeg" if irel.endswith(".jpg") else "png"),
                                   note="signed query string dropped from url")
                            crop = {"file": irel, "sha256": files[irel]["sha256"], "marker": cm}
                    if crop is None:
                        print(f"  {city} {s} label {lid}: no crop ({why})", flush=True)
                clog["accepted"][key] = {"stratum": s, "cand_rank": int(c["cand_rank"]), "tiles": tiles,
                                         "blank_years": blank, "crop": crop, "utc": now()}
                got += 1
                print(f"{city} {s} {got}/{quota[s]} node {key} (rank {c['cand_rank']})"
                      f"{' crop' if crop else ''}{' blank ' + str(blank) if blank else ''}", flush=True)
                save()
            if got < quota[s]:
                print(f"  {city} {s}: only {got} of {quota[s]} (candidates exhausted)", flush=True)
        save()
    write_sheet(a.out, log)
    save()
    print("SHA256SUMS:", write_sums(arc))


def write_sheet(out, log):
    rows = []
    for city in DRAW_CITIES:
        clog = log["cities"].get(city)
        p = os.path.join(out, f"candidates_{city}.csv")
        if not clog or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as f:
            cands = {c["node_id"]: c for c in csv.DictReader(f)}
        for key in sorted(clog["accepted"], key=lambda k: (STRATA.index(clog["accepted"][k]["stratum"]),
                                                             clog["accepted"][k]["cand_rank"])):
            acc, c = clog["accepted"][key], cands[key]
            cm = (acc["crop"] or {}).get("marker") or {}
            rows.append({**c, "years": " ".join(sorted(acc["tiles"])), "years_blank": " ".join(sorted(acc["blank_years"])),
                         "has_crop": "yes" if acc["crop"] else "", "crop_w": cm.get("width", ""),
                         "crop_h": cm.get("height", ""), "crop_x": cm.get("x", ""), "crop_y": cm.get("y", "")})
    random.Random(f"{SEED}:order").shuffle(rows)
    for k, r in enumerate(rows):
        r["gt_id"] = f"r{k:03d}"
    with open(os.path.join(out, "sheet.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"sheet: {len(rows)} rows ->", os.path.join(out, "sheet.csv"))


def write_sums(arc):
    lines = []
    for dp, _, fns in os.walk(arc):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, arc).replace(os.sep, "/")
            if rel in ("SHA256SUMS", "manifest.json", "manifest.json.tmp") or rel.startswith("probe/"):
                continue
            lines.append((rel, sha256_file(p)))
    lines.sort()
    with open(os.path.join(arc, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {rel}\n" for rel, h in lines)
    return len(lines)


if __name__ == "__main__":
    main()
