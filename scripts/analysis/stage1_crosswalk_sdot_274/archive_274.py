"""Stage the #274 crosswalk-detector study's non-repo artefacts as one archive for the lab share (RampNet#274).

Everything the study needs that the repo does not hold: the Stage 2 training root (images once, each label
version beside them), the (b') panos as inferred, the final checkpoints with their training and slurm logs, the
rating pages Jon used (crops embedded), local working files, inputs pulled from prod or the web, and the
gitignored outputs of the #274 worktrees. Code, predictions, scores, verdict CSVs and docs stay in the repo.

WHERE THE ARCHIVE IS. Canonical copy, on the lab share (backed up by CSE IT), readable from makelab1/makelab2:
  /projects/makeabilitylab/RampNet/archives/stage1_crosswalk_detector_274/
Verify a copy with `sha256sum -c SHA256SUMS` (the staged files) and `sha256sum -c store_natives.SHA256SUMS` (the
native PS-store panos, which are copied on makelab2 itself by install_natives.sh, never uploaded). README.md in
the archive explains the layout and how to rebuild a training root and re-score.

  python archive_274.py stage   --out <staging dir on the same volume as the sources>
  python archive_274.py natives --out <same staging dir>

`stage` hardlinks each source file into the layout (no second copy on disk; a file whose bytes are already
staged under another path is linked to that copy, so `rsync -H` sends it once), then writes manifest.json (path,
bytes, sha256, source, role), SHA256SUMS and README.md. Re-running it adds new sources and re-hashes only files
whose size or mtime changed; it refuses a staged path whose source bytes changed. `natives` hashes the local
copies of the PS-store panos the study read and writes store_natives.csv, store_natives.SHA256SUMS,
store_natives.copylist.tsv and install_natives.sh.

The source table below is the record of where each piece came from. Paths are this study's (Jon's Windows
desktop); SK = D:/Git/sal-scratch-keep. Klone pieces were pulled read-only on 2026-11-06 (tar sha256 in
SOURCES' notes) from /gscratch/scrubbed/jfroehli/crosswalk274 and /gscratch/makelab/jonf/crosswalk274.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

SK = "D:/Git/sal-scratch-keep"
WT = "D:/Git/RampNet-wt"
STORE = "/projects/makeabilitylab/sidewalk_panos/Panoramas"      # the PS server's pano store, on makelab2
ARCHIVE = "/projects/makeabilitylab/RampNet/archives/stage1_crosswalk_detector_274"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}

# (archive prefix, local source dir, role). Every file under the source dir is staged at <prefix>/<relpath>.
SOURCES = [
    ("stage2_root/images", f"{SK}/crosswalk274-root", "Stage 2 root JPEGs (4096x2048, as trained/inferred); "
     "*.json here are excluded, see labels_*", "jpg"),
    ("stage2_root/labels_v2", f"{SK}/crosswalk274-root", "v2 label JSONs (snap-collision fix, f1a933d); equal "
     "byte for byte to crosswalk274-score/v2/root_v2_json.tar (sha256 37b4bb83...)", "json"),
    ("stage2_root/labels_v1", f"{SK}/crosswalk274-klone/x/root_v1_json", "v1 label JSONs (export 2026-11-04), "
     "from klone root/ (tar sha256 777c6613...)", None),
    ("bprime/images_4096", f"{SK}/crosswalk274-score/bprime_4096", "(b') panos as inferred (bprime-prep)", None),
    ("checkpoints/crosswalk_s1", f"{SK}/crosswalk274-klone/x/ckpts_274/s1", "final checkpoint + sha256 + "
     "TensorBoard runs/ (klone /gscratch/makelab/jonf/crosswalk274/s1; tar sha256 7573e044...)", None),
    ("checkpoints/crosswalk_s2", f"{SK}/crosswalk274-klone/x/ckpts_274/v2_s1", "the same for crosswalk_s2 "
     "(klone .../crosswalk274/v2_s1)", None),
    ("logs/slurm", f"{SK}/crosswalk274-klone/x/logs_274/logs", "every #274 slurm .out/.err (train + score jobs; "
     "tar sha256 3764571b...)", None),
    ("rating_pages/s1_t0.21_v1", f"{SK}/crosswalk274-score/gallery", "first audit pages + Jon's exports", None),
    ("rating_pages/v2_t0.31", f"{SK}/crosswalk274-score/gallery_v2_t0.31", "v2 re-score audit @0.31", None),
    ("rating_pages/v2_t0.21", f"{SK}/crosswalk274-score/gallery_v2_t0.21", "v2 re-score audit @0.21", None),
    ("rating_pages/s2_t0.31", f"{SK}/crosswalk274-score/gallery_s2_t0.31", "crosswalk_s2 audit @0.31", None),
    ("rating_pages/hardneg_step0", f"{SK}/crosswalk274-score/hardneg_step0", "hard negatives Step 0 (blind)", None),
    ("rating_pages/hardneg_confirm", f"{SK}/crosswalk274-score/hardneg_confirm", "hard negatives page 1: Jon "
     "confirms AI-flagged candidates (460 cards)", None),
    ("rating_pages/hardneg_review9", f"{SK}/crosswalk274-score/hardneg_review9", "the 9 page-1 cards Claude "
     "flagged, re-rated by Jon", None),
    ("rating_pages/hardneg_confirm2", f"{SK}/crosswalk274-score/hardneg_confirm2", "hard negatives page 2 (200 "
     "cards, --big views); partial export (112) and full export (200)", None),
    ("work/hardneg_mine", f"{SK}/crosswalk274-score/hardneg_mine", "mining crops, numbered contact sheets the AI "
     "pre-screen read, its per-part verdicts, the full pre-screen page", None),
    ("work/v1", f"{SK}/crosswalk274-score/v1", "v1 arm C working files (crowd_labels, placements, test root)",
     None),
    ("work/v2", f"{SK}/crosswalk274-score/v2", "v2 working files: placements, prefill CSVs, root_v2_json.tar", None),
    ("work/preds", f"{SK}/crosswalk274-score/preds", "local prediction copies", None),
    ("inputs/score_workspace", f"{SK}/crosswalk274-score", "read-only sidewalk_seattle.pano_data export "
     "(pano positions for (b')) and the (b') fetch list, prediction tarballs", "top"),
    ("inputs/phase0_census", f"{SK}/phase0-census-2026-10-28", "Phase 0 census (rawLabels pulls, assets)", None),
    ("inputs/store_listings", f"{SK}/crowd-stores", "PS-store listings read for the crowd arms", "txt"),
    ("inputs/store_listings/seattle", f"{SK}/arm-c-seattle-store", "Seattle PS-store listing", "txt"),
    # crosswalk_s3 (hard negatives) and its confirmation, added 2026-11-08
    ("stage2_root/labels_v3", f"{SK}/crosswalk274-root_v3", "v3 label JSONs: v2 plus Jon's 166 hard negatives "
     "(export_stage2.py add-negatives; labels digest 005d6f76...)", "json"),
    ("checkpoints/crosswalk_s3", f"{SK}/crosswalk274-klone/x3/ckpts/v3_s1", "crosswalk_s3 final checkpoint "
     "(sha256 32f93561...) + TensorBoard runs/ (klone .../crosswalk274/v3_s1; tar sha256 e28ac298...)", None),
    ("logs/slurm_2026-11-08", f"{SK}/crosswalk274-klone/x3/logs/logs", "s3 smoke + training, s3 val/test/(b') "
     "inference, confirmation city inference (tar sha256 2e8ac6ef...)", None),
    ("rating_pages/hardneg_s3", f"{SK}/crosswalk274-score/hardneg_s3", "s3 audit of the Step 0 draw (blind, Step 0 "
     "verdicts prefilled, --big) + Jon's export", None),
    ("rating_pages/confirm_s3", f"{SK}/crosswalk274-confirm/page", "s3 confirmation: ONE blind page of s1 (0.21) and "
     "s3 (0.27) cards on the fresh Seattle draw + Juneau + Laurens Mapillary/GSV draws, + Jon's export", None),
    ("confirm_s3/panos_4096", f"{SK}/crosswalk274-confirm/page_panos", "every pano on the confirmation page, "
     "4096x2048 as inferred (the --pano-dir that rebuilds the page)", None),
    ("confirm_s3/natives/juneau", f"{SK}/crosswalk274-confirm/juneau_native", "Juneau draw, native res as fetched "
     "by export_benchmark.py --out (Mapillary; index.csv)", None),
    ("confirm_s3/natives/laurens_mly", f"{SK}/crosswalk274-confirm/laurens_mly_native", "Laurens Mapillary draw, "
     "native res (GoPro Max)", None),
    ("confirm_s3/natives/laurens_gsv", f"{SK}/crosswalk274-confirm/laurens_gsv_native", "Laurens GSV draw, native "
     "res", None),
    ("confirm_s3/ps_label_probe/natives", f"{SK}/crosswalk274-confirm/ps_label_native", "the 13 panos carrying "
     "live PS Crosswalk labels in Laurens and Juneau (read 2026-11-08), native res", None),
    ("confirm_s3/ps_label_probe/panos_4096", f"{SK}/crosswalk274-confirm/ps_label_4096", "the same, 4096x2048 as "
     "inferred", None),
    ("s4_brick/store_natives", f"{SK}/brick-store/store", "brick-tagged crowd crosswalk panos as streamed read-only "
     "from the PS store (<city>/<id[:2]>/<id>.jpg; brick_store.tar sha256 in the repo)", "jpg"),
    ("s4_brick/panos_4096", f"{SK}/brick-store/prep_4096", "the same, 4096x2048 as the training root", None),
    ("s4_brick/crops", f"{SK}/brick-store/crops", "pre-screen crops (index.csv maps crop -> card)", None),
    ("s4_brick/sheets", f"{SK}/brick-store/sheets", "the 77 numbered contact sheets the blind AI readers saw", None),
    ("rating_pages/brick_confirm", f"{SK}/brick-store/page", "Jon's brick confirmation page (+ his export)", None),
    # crosswalk_s4 Step A and root_v4, added 2026-11-09
    ("s4_stepA/store_natives", f"{SK}/s4-stepA/store", "Columbia SC brick, Richmond Mapillary and Laurens car-top "
     "mining panos as streamed read-only from the PS store (<city>/<id[:2]>/<id>.jpg; stepA_store.tar sha256 in the "
     "repo)", "jpg"),
    ("s4_stepA/panos_4096", f"{SK}/s4-stepA/prep_4096", "the same, 4096x2048 as the training root (bprime-prep)",
     None),
    ("rating_pages/s4_crowd", f"{SK}/s4-stepA/page_crowd", "s4 Step A crowd page: 214 Columbia brick (P1 GT) + 89 "
     "Richmond Mapillary labels, + Jon's export", None),
    ("rating_pages/s4_carTop", f"{SK}/s4-stepA/page_cartop", "s4 Step A Laurens car-top mining page (124 cards, "
     "--neg-disc), + Jon's export", None),
    ("stage2_root/labels_v4", f"{SK}/crosswalk274-root_v4", "v4 label JSONs: v3 plus 693 new sparse train panos "
     "(export_stage2.py add-panos; labels digest 73a588e2...). Their JPEGs are s4_brick/panos_4096 and "
     "s4_stepA/panos_4096, byte for byte", "json"),
    # crosswalk_s4 training, evaluation and Step B, added 2026-11-09 (klone pull: x4/*.tar, sha256 in the notes)
    ("checkpoints/crosswalk_s4", f"{SK}/crosswalk274-klone/x4/ckpts/v4_s1", "crosswalk_s4 seed 1, the adopted "
     "model (sha256 ebad379b...) + TensorBoard runs/ (klone .../crosswalk274/v4_s1; ckpts_274_s4.tar sha256 "
     "6099896a...)", None),
    ("checkpoints/crosswalk_s4_seed2", f"{SK}/crosswalk274-klone/x4/ckpts/v4_s2", "crosswalk_s4 seed 2 "
     "(sha256 48a9928f...), guard 1", None),
    ("checkpoints/crosswalk_s4_seed3", f"{SK}/crosswalk274-klone/x4/ckpts/v4_s3", "crosswalk_s4 seed 3 "
     "(sha256 6f143e48...), guard 1", None),
    ("checkpoints/crosswalk_s1_seed2", f"{SK}/crosswalk274-klone/x4/ckpts/v1_s2", "crosswalk_s1 seed 2 "
     "(sha256 dcdbd4ee...), guard 1's base and the seed-2 threshold match", None),
    ("checkpoints/crosswalk_s1_seed3", f"{SK}/crosswalk274-klone/x4/ckpts/v1_s3", "crosswalk_s1 seed 3 "
     "(sha256 c3126aa6...)", None),
    ("checkpoints/crosswalk_s3_seed2", f"{SK}/crosswalk274-klone/x4/ckpts/v3_s2", "crosswalk_s3 seed 2 "
     "(sha256 86a42bfc...)", None),
    ("checkpoints/crosswalk_s3_seed3", f"{SK}/crosswalk274-klone/x4/ckpts/v3_s3", "crosswalk_s3 seed 3 "
     "(sha256 5de289f7...)", None),
    ("logs/slurm_2026-11-09", f"{SK}/crosswalk274-klone/x4/logs/logs", "every #274 slurm .out/.err after the s3 "
     "pull: s1/s3 seeds, Step A, s4 smoke + 3 seeds, s4 val/test/(b')/cities/PS-label/Columbia/Bayonne inference "
     "(logs_274_s4.tar sha256 e66c2084...)", None),
    ("s4_bayonne/store_natives", f"{SK}/s4-bayonne/store", "the 6 Bayonne Panoramax panos with live Crosswalk "
     "labels, streamed read-only from the PS store (bayonne_store.tar sha256 in the repo)", "jpg"),
    ("s4_bayonne/panos_4096", f"{SK}/s4-bayonne/prep_4096", "the same, 4096x2048 as inferred", None),
    ("rating_pages/s4_stepB", f"{SK}/s4-stepB/page", "s4 Step B: blind page of s4 (0.23) cards on the four "
     "confirmation draws, 200 prefilled from the confirm_s3 export, + Jon's export (2026-11-09)", None),
    ("s4_stepB/panos_4096", f"{SK}/s4-stepB/page_panos", "every pano on the Step B page, 4096x2048 as inferred "
     "(the --pano-dir that rebuilds it)", None),
    ("work/s4_preds", f"{SK}/s4-preds", "local copies of every s4-era prediction file as pulled (s4_preds.tgz "
     "and unpacked)", None),
    ("confirm_s3/draw_records", f"{SK}/crosswalk274-confirm", "the city draws' labeler records (input to "
     "export_benchmark.py)", "jsonl"),
    ("worktree_untracked/stage1-label-type-274", f"{WT}/stage1-label-type-274", "gitignored #274 outputs at "
     "their repo paths (SDOT pilot pixels/views, arm C thumbs, logs, galleries)", "ignored"),
    ("worktree_untracked/crosswalk-inventory-274", f"{WT}/crosswalk-inventory-274", "gitignored inventory inputs",
     "ignored"),
]

# Native PS-store panos the study read, as local copies (city, local dir laid out <id[:2]>/<id>.jpg).
NATIVES = [
    ("seattle-wa", f"{SK}/arm-c-seattle-store/panos"),
    ("seattle-wa", f"{SK}/crosswalk274-score/bprime_panos"),
    ("seattle-wa", f"{SK}/crowd-stores/seattle-negatives"),
    ("amsterdam", f"{SK}/crowd-stores/amsterdam"),
    ("burnaby", f"{SK}/crowd-stores/burnaby"),
    ("teaneck-nj", f"{SK}/crowd-stores/teaneck-nj"),
]


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_ignored(repo):
    """Repo-relative paths git reports as ignored (directories expanded), minus caches."""
    import subprocess
    out = subprocess.run(["git", "-C", repo, "status", "--short", "--ignored", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout
    paths = [line[3:].strip().strip('"') for line in out.splitlines() if line.startswith("!! ")]
    keep = ("274", "crosswalk")                                   # only this study's outputs, not other studies'
    return [p for p in paths if not (set(p.replace("\\", "/").split("/")) & SKIP_DIRS)
            and any(k in p for k in keep)]


def walk(src, mode):
    """(relpath, abspath) of every file a source contributes, sorted.

    mode: None = everything under src; 'jpg' / 'json' / 'txt' = that extension, at any depth; 'top' = files
    directly in src (no subdirectories); 'ignored' = git's ignored files in that worktree."""
    out = []
    if mode == "ignored":
        for p in git_ignored(src):
            ap = os.path.join(src, p)
            if os.path.isdir(ap):
                for root, dirs, files in os.walk(ap):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    out += [(os.path.relpath(os.path.join(root, f), src), os.path.join(root, f)) for f in files]
            elif os.path.isfile(ap):
                out.append((p, ap))
    elif mode == "top":
        out = [(f, os.path.join(src, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    else:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if mode and not f.lower().endswith("." + mode):
                    continue
                out.append((os.path.relpath(os.path.join(root, f), src), os.path.join(root, f)))
    return sorted((r.replace("\\", "/"), a) for r, a in out)


def load_manifest(out):
    p = os.path.join(out, "manifest.json")
    if not os.path.exists(p):
        return {}
    return {e["path"]: e for e in json.load(open(p, encoding="utf-8"))["files"]}


def write_manifest(out, entries, extra):
    files = sorted(entries.values(), key=lambda e: e["path"])
    with open(os.path.join(out, "manifest.json.tmp"), "w", encoding="utf-8") as f:
        json.dump({"archive": ARCHIVE, "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   **extra, "files": files}, f, indent=1)
    os.replace(os.path.join(out, "manifest.json.tmp"), os.path.join(out, "manifest.json"))
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        for e in files:
            f.write(f"{e['sha256']}  {e['path']}\n")


def cmd_stage(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    old = load_manifest(out)
    entries, by_sha = dict(old), {}
    for e in old.values():
        by_sha.setdefault(e["sha256"], e["path"])
    added = changed_src = 0
    for prefix, src, role, mode in SOURCES:
        if not os.path.isdir(src):
            raise SystemExit(f"missing source {src} ({prefix})")
        files = walk(src, mode)
        log(f"{prefix}: {len(files)} files from {src}")
        for rel, ap in files:
            path = f"{prefix}/{rel}"
            st = os.stat(ap)
            prev = old.get(path)
            if prev and prev["bytes"] == st.st_size and prev["source_mtime"] == int(st.st_mtime):
                continue                                              # staged before, source untouched
            digest = sha256(ap)
            if prev and prev["sha256"] != digest:
                raise SystemExit(f"{ap}: bytes changed since {path} was staged ({prev['sha256'][:12]} -> "
                                 f"{digest[:12]}); stage a new version under a new prefix instead")
            dest = os.path.join(out, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                twin = by_sha.get(digest)
                os.link(os.path.join(out, twin) if twin else ap, dest)
            by_sha.setdefault(digest, path)
            entries[path] = {"path": path, "bytes": st.st_size, "sha256": digest, "role": role,
                             "source": ap.replace("\\", "/"), "source_mtime": int(st.st_mtime)}
            added += 0 if prev else 1
            changed_src += 1 if prev else 0
    write_manifest(out, entries, {"sources": [{"prefix": p, "source": s, "role": r, "mode": m}
                                              for p, s, r, m in SOURCES]})
    write_readme(out, entries)
    total = sum(e["bytes"] for e in entries.values())
    uniq = sum(e["bytes"] for e in {e["sha256"]: e for e in entries.values()}.values())
    log(f"staged {len(entries)} files ({total / 1e9:.2f} GB, {uniq / 1e9:.2f} GB unique); new {added}, "
        f"re-touched {changed_src}")


def cmd_natives(a):
    """Hash the local store copies; one row per (city, pano). Two local copies of one pano must agree."""
    rows = {}
    for city, d in NATIVES:
        for rel, ap in walk(d, "jpg"):
            pid = os.path.basename(rel)[:-4]
            digest = sha256(ap)
            key = (city, pid)
            if key in rows and rows[key]["sha256"] != digest:
                raise SystemExit(f"{city}/{pid}: local copies differ ({rows[key]['local']} vs {ap})")
            rows.setdefault(key, {"city": city, "pano_id": pid, "sha256": digest, "bytes": os.path.getsize(ap),
                                  "local": ap.replace("\\", "/"),
                                  "store": f"{STORE}/{city}/{pid[:2]}/{pid}.jpg",
                                  "archive": f"store_natives/{city}/{pid[:2]}/{pid}.jpg"})
        log(f"{city} {d}: {len(rows)} unique so far")
    rs = sorted(rows.values(), key=lambda r: (r["city"], r["pano_id"]))
    import csv
    with open(os.path.join(a.out, "store_natives.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    with open(os.path.join(a.out, "store_natives.SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{r['sha256']}  {r['archive']}\n" for r in rs)
    with open(os.path.join(a.out, "store_natives.copylist.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{r['store']}\t{r['archive']}\n" for r in rs)
    with open(os.path.join(a.out, "install_natives.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write(INSTALL_NATIVES)
    log(f"natives: {len(rs)} panos, {sum(r['bytes'] for r in rs) / 1e9:.1f} GB")


INSTALL_NATIVES = r"""#!/bin/bash
# Run ON makelab2 (or makelab1): copy the native PS-store panos this study read into the archive, from the store
# on the same share (no upload), then check every one against the sha256 of the copy the study actually read.
# Written by archive_274.py natives. Re-runnable: files already in place are kept.
set -u
A="${1:-/projects/makeabilitylab/RampNet/archives/stage1_crosswalk_detector_274}"
cd "$A" || { echo "no archive dir $A"; exit 1; }
: > store_natives.missing.txt
n=0; have=0; miss=0
while IFS=$'\t' read -r src dst; do
  if [ -f "$dst" ]; then have=$((have+1)); continue; fi
  mkdir -p "$(dirname "$dst")"
  if [ -f "$src" ]; then cp -p "$src" "$dst" && n=$((n+1)); else echo "$dst" >> store_natives.missing.txt; miss=$((miss+1)); fi
done < store_natives.copylist.tsv
echo "copied $n, already present $have, missing from the store $miss (store_natives.missing.txt)"
sha256sum -c --quiet store_natives.SHA256SUMS > store_natives.check.txt 2>&1
echo "sha256 check exit $? -- every line in store_natives.check.txt is a file to upload from the local copy"
chmod -R g+rX,o+rX store_natives
"""


def write_readme(out, entries):
    by = {}
    for e in entries.values():
        top = "/".join(e["path"].split("/")[:2])
        n, b = by.get(top, (0, 0))
        by[top] = (n + 1, b + e["bytes"])
    rows = "\n".join(f"| `{k}/` | {n:,} | {b / 1e6:,.0f} MB |" for k, (n, b) in sorted(by.items()))
    text = f"""# RampNet#274 crosswalk detector study: archive

Everything the study needs that the RampNet repo (branch `stage1-label-type-274`) does not hold. Code,
predictions, scores, verdict CSVs and the study doc (`docs/stage1_crosswalk_sdot_274.md`) are in the repo. Built by
`scripts/analysis/stage1_crosswalk_sdot_274/archive_274.py`; its source table records where each piece came from,
and manifest.json repeats it per file.

## Verify

    sha256sum -c SHA256SUMS                 # staged files (uploaded with rsync -H)
    sha256sum -c store_natives.SHA256SUMS   # native PS-store panos (copied on makelab2 by install_natives.sh)

## Layout

| directory | files | size |
|---|---|---|
{rows}

- `stage2_root/images/<split>/<pano>.jpg`: the 4096x2048 JPEGs every model trained and was scored on.
  `labels_v1/` and `labels_v2/` hold each label version's `<split>/<pano>.json`; a later version adds `labels_v3/`.
- `bprime/images_4096/`: the (b') node-sample panos as inferred.
- `checkpoints/<label>/final_step_1000.pth` (+ `.sha256`, TensorBoard `runs/`): the only checkpoint each run scores.
  Per-step checkpoints were not kept (the pre-registration scores the final one only).
- `logs/slurm/`: every train and score job's stdout/stderr.
- `rating_pages/<audit>/`: the HTML pages Jon rated (crops embedded, open offline) and his CSV exports.
- `work/`, `inputs/`: working files and the prod/web pulls (`seattle_pano_data.psv` is a read-only export of
  `sidewalk_seattle.pano_data`).
- `worktree_untracked/<worktree>/<repo path>`: gitignored outputs, at the paths the scripts expect.
- `store_natives/<city>/<id[:2]>/<id>.jpg`: the native PS-store panos the study read (store_natives.csv).
- `confirm_s3/`: the s3 confirmation's imagery. `panos_4096/` is every pano on its rating page as inferred;
  `natives/` the Juneau and Laurens draws at native resolution as fetched; `draw_records/` the labeler records
  they were fetched from.

## A second rater

Every rating page rebuilds from the repo plus this archive, so another person can rate the same cards blind.
For the s3 confirmation page (the card list and context marks are committed under
`analysis_out/stage1_crosswalk_detector_274/hardneg/confirm_s3/page/`):

    python scripts/analysis/stage1_crosswalk_sdot_274/score_274.py cards \
        --candidates analysis_out/stage1_crosswalk_detector_274/hardneg/confirm_s3/page/candidates.csv \
        --preds-file analysis_out/stage1_crosswalk_detector_274/hardneg/confirm_s3/page/context.jsonl --threshold 0 \
        --pano-dir <archive>/confirm_s3/panos_4096 --blind --label confirm_s3_rater2 --out <dir>

A new `--label` gives the page its own browser storage, and no `--prefill` means nothing is pre-rated. The card
ids are the same as on Jon's page, so `carry` and `audit-compare` run on the second rater's export unchanged, and
two exports join on `card_id` for agreement. The Step 0 and s3 audit pages rebuild the same way from
`hardneg/step0_audit/` and `hardneg/s3_audit/` with `--pano-dir <archive>/bprime/images_4096` (the study doc's
Replication section has each page's exact flags).

## Rebuild a training root and re-score

    for s in train val test; do mkdir -p root/$s; cp stage2_root/images/$s/*.jpg stage2_root/labels_v2/$s/*.json root/$s/; done
    python scripts/analysis/stage1_crosswalk_sdot_274/score_274.py score --root root --preds analysis_out/stage1_crosswalk_detector_274/v2_labels/preds ...

(the full commands are in the study doc's Replication section; predictions are committed in the repo).
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage", "natives"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stage": cmd_stage, "natives": cmd_natives}[a.cmd](a)


if __name__ == "__main__":
    main()
