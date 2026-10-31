"""Archive everything the Seattle aerial spot-check page fetches over the network, so the rating can be
replayed offline later (King County export service, Project Sidewalk signed crops, the rawLabels pull).

Layout under --out:
  manifest.json                 one entry per saved file: url, sha256, bytes, fetched_utc, http status
  README.md
  sheet/<sheet csv>             the AI-rated sheet the page was built from
  rawlabels/<city>_Crosswalk.csv  the census's cached v3 rawLabels pull (pano date, votes, pano_url)
  verdicts/<json>               the human export(s), kept by exported timestamp
  page/<html>                   the built page at archive time
  ortho/<label_id>_<year>_<36|18>m.png   King County ortho export, 720 px, bbox centred on the OSM node
  crops/<label_id>.png + <label_id>.metadata.json   Project Sidewalk label crop + its metadata
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import urllib.error

R = 6378137.0
PX = 720
HALVES = (18, 9)
YEARS = (2023, 2025)
SVC = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_{y}/MapServer/export"
UA = "ProjectSidewalk-RampNet crosswalk-phase0 archive (research; contact jonf@cs.uw.edu)"
SPACING_S = 0.25


def merc(lat, lng):
    return R * lng * math.pi / 180, R * math.log(math.tan(math.pi / 4 + lat * math.pi / 360))


def ortho_url(year, nlat, nlng, half):
    cx, cy = merc(nlat, nlng)
    return (f"{SVC.format(y=year)}?bbox={cx - half},{cy - half},{cx + half},{cy + half}"
            f"&bboxSR=3857&imageSR=3857&size={PX},{PX}&format=png&f=image")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(url, timeout=60):
    """(status, content_type, bytes); 3 tries on 5xx / network errors."""
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code, e.headers.get("Content-Type", ""), e.read()
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--rawlabels", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--server", default="https://sidewalk-sea.cs.washington.edu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = a.out
    for d in ("sheet", "rawlabels", "verdicts", "page", "ortho", "crops"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {"files": {}}
    files = manifest["files"]

    def record(rel, url, status, ctype, note=None):
        p = os.path.join(out, rel)
        files[rel] = {"url": url, "http_status": status, "content_type": ctype, "sha256": sha256(p),
                      "bytes": os.path.getsize(p), "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        if note:
            files[rel]["note"] = note

    def save():
        tmp = mpath + ".tmp"
        json.dump(manifest, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, mpath)

    # ---- the inputs, copied
    vj = json.load(open(a.verdicts, encoding="utf-8"))
    vname = f"aerial_verdicts_seattle-wa_{vj['exported'].replace(':', '').replace('-', '')[:15]}.json"
    for src, rel in ((a.sheet, "sheet/" + os.path.basename(a.sheet)), (a.rawlabels, "rawlabels/" + os.path.basename(a.rawlabels)),
                     (a.verdicts, "verdicts/" + vname), (a.page, "page/" + os.path.basename(a.page))):
        shutil.copyfile(src, os.path.join(out, rel))
        record(rel, "file://" + os.path.abspath(src).replace("\\", "/"), None, None)
    save()

    rows = list(csv.DictReader(open(a.sheet, encoding="utf-8", newline="")))
    manifest["rows"] = [{"i": i, "label_uid": r["label_uid"], "node_id": r["node_id"], "node_lat": float(r["node_lat"]),
                         "node_lng": float(r["node_lng"]), "label_lat": float(r["label_lat"]), "label_lng": float(r["label_lng"])}
                        for i, r in enumerate(rows)]
    manifest["ortho"] = {"service": SVC, "size_px": PX, "half_widths_m": HALVES, "years": YEARS, "crs": "EPSG:3857",
                         "note": "bbox is centred on the OSM node; identical to the page's imgUrl()"}
    manifest["crops"] = {"server": a.server, "metadata_route": "/cropImage/Crosswalk/<label_id>/metadata",
                         "note": "image_url in the metadata is a short-lived signed link; the PNG beside it is what it served"}
    save()

    n_new = n_skip = n_bad = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        lid = r["label_uid"].split(":")[-1]
        nlat, nlng = float(r["node_lat"]), float(r["node_lng"])
        for y in YEARS:
            for half in HALVES:
                rel = f"ortho/{lid}_{y}_{2 * half}m.png"
                if rel in files and os.path.exists(os.path.join(out, rel)):
                    n_skip += 1
                    continue
                url = ortho_url(y, nlat, nlng, half)
                st, ct, body = get(url)
                time.sleep(SPACING_S)
                if st != 200 or not ct.startswith("image/"):
                    n_bad += 1
                    print(f"  row {i} {rel}: HTTP {st} {ct} ({len(body)} B) -- not saved", file=sys.stderr)
                    continue
                open(os.path.join(out, rel), "wb").write(body)
                record(rel, url, st, ct)
                n_new += 1
        rel_m, rel_p = f"crops/{lid}.metadata.json", f"crops/{lid}.png"
        if rel_p in files and os.path.exists(os.path.join(out, rel_p)):
            n_skip += 1
        else:
            murl = f"{a.server}/cropImage/Crosswalk/{lid}/metadata"
            st, ct, body = get(murl)
            time.sleep(SPACING_S)
            if st != 200:
                n_bad += 1
                print(f"  row {i} {rel_m}: HTTP {st} -- no crop metadata", file=sys.stderr)
            else:
                open(os.path.join(out, rel_m), "wb").write(body)
                record(rel_m, murl, st, ct)
                meta = json.loads(body)
                iurl = a.server + meta["image_url"]
                st, ct, body = get(iurl)
                time.sleep(SPACING_S)
                if st != 200 or not ct.startswith("image/"):
                    n_bad += 1
                    print(f"  row {i} {rel_p}: HTTP {st} {ct} -- crop not saved", file=sys.stderr)
                else:
                    ext = ".png" if "png" in ct else ".jpg"
                    if ext != ".png":
                        rel_p = f"crops/{lid}{ext}"
                    open(os.path.join(out, rel_p), "wb").write(body)
                    record(rel_p, iurl.split("?")[0], st, ct, note="signed query string dropped from url")
                    n_new += 1
        if i % 10 == 9:
            save()
            print(f"row {i + 1}/{len(rows)}: {n_new} fetched, {n_skip} already there, {n_bad} failed, {time.time() - t0:.0f}s")
    save()
    print(f"done: {n_new} fetched, {n_skip} already there, {n_bad} failed; {len(files)} files in manifest")


if __name__ == "__main__":
    main()
