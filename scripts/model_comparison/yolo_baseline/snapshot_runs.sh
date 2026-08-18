#!/usr/bin/env bash
#
# snapshot_runs.sh - mirror live Ultralytics run directories to durable storage,
# with end-to-end sha256 verification.  Issue #51.
#
# WHY THIS EXISTS
#   The live YOLO-baseline runs write to /gscratch/scrubbed, which auto-purges after
#   ~21 idle days and is a *shared* pool that has already hit 97% full and killed an
#   arm mid-checkpoint (y26_pano, 2026-08-07, OSError 122).  Tillicum's runs live on
#   /gpfs/projects/makelab, documented as non-archival and purged at end of project.
#   An arm that has finished its 60-epoch schedule and exists only in one of those two
#   places is one purge away from unreproducible.  /gscratch/makelab is purchased and
#   never purged, so that is where the durable copy goes.
#
#   Before this script, the snapshot was assembled by hand and only its MANIFEST was
#   kept.  A hand-assembled backup cannot be re-run by someone else, which is the test
#   the repo applies to everything else (CLAUDE.md, "replicable from a clean clone").
#
# WHAT IT GUARANTEES
#   - Never destroys a good copy: each file lands as <name>.tmp and is renamed only
#     after its hash matches the source.  A quota failure mid-write therefore leaves
#     the previous verified copy intact, which is exactly the failure that corrupted
#     y26_pano/weights/last.pt.
#   - Torn-read safe: running arms rewrite weights/last.pt at every epoch boundary, so
#     the source is hashed before AND after the copy; if it moved, the file is retried.
#   - Idempotent: a file whose destination already matches is left alone and reported
#     as unchanged, so re-running costs hashing and nothing else.
#
# SCOPE
#   Single-cluster: SRC and DST must both be visible from the machine running this.
#   The Tillicum arm (y11x_pano_h200) is mirrored to klone by a separate cross-cluster
#   copy, since /gscratch is not mounted on Tillicum; once mirrored it is picked up
#   here like any other arm when SRC points at the snapshot's own parent.
#
# USAGE
#   ./snapshot_runs.sh                    # every arm found under SRC
#   ./snapshot_runs.sh y11l_pano y26_pano # only the named arms
#   SRC=... DST=... ./snapshot_runs.sh    # override the committed defaults
#
set -uo pipefail

SRC="${SRC:-/gscratch/scrubbed/jfroehli/yolo_runs}"
DST="${DST:-/gscratch/makelab/jonf/rampnet_yolo_baseline_51}"
MAX_TRIES="${MAX_TRIES:-3}"

# Per arm: the weights are the irreplaceable part; results.csv and args.yaml are what
# make a reported number re-derivable by someone else.
FILES=(results.csv args.yaml weights/best.pt weights/last.pt)

# Directories under SRC that are not arms.
SKIP_RE='^(_archive|.*_dropped_).*'

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
verified_list="$DST/.sha256.verified"
tmp_shas=$(mktemp)
n_copied=0; n_same=0; n_failed=0

log() { printf '%s\n' "$*"; }

# copy_verified <src-file> <dst-file> <label>
copy_verified() {
  local s=$1 d=$2 rel=$3 try=0 h1 h2 hd
  [ -f "$s" ] || { log "     - $rel (absent at source)"; return 0; }
  while [ "$try" -lt "$MAX_TRIES" ]; do
    try=$((try + 1))
    h1=$(sha256sum "$s" | cut -d' ' -f1)
    if [ -f "$d" ]; then
      hd=$(sha256sum "$d" | cut -d' ' -f1)
      if [ "$hd" = "$h1" ]; then
        log "     = $rel  ($(numfmt --to=iec --suffix=B "$(stat -c%s "$s")" 2>/dev/null || stat -c%s "$s"))  unchanged"
        printf '%s  %s\n' "$h1" "$rel" >>"$tmp_shas"
        n_same=$((n_same + 1))
        return 0
      fi
    fi
    mkdir -p "$(dirname "$d")"
    if ! cp -f "$s" "$d.tmp" 2>/tmp/snap_cp_err.$$; then
      log "     ! $rel  COPY FAILED: $(cat /tmp/snap_cp_err.$$ | tail -1)"
      rm -f "$d.tmp" /tmp/snap_cp_err.$$
      n_failed=$((n_failed + 1))
      return 1
    fi
    rm -f /tmp/snap_cp_err.$$
    h2=$(sha256sum "$s" | cut -d' ' -f1)
    if [ "$h1" != "$h2" ]; then
      rm -f "$d.tmp"
      log "     ~ $rel  source moved mid-copy (epoch boundary), retry $try/$MAX_TRIES"
      sleep 10
      continue
    fi
    hd=$(sha256sum "$d.tmp" | cut -d' ' -f1)
    if [ "$hd" != "$h1" ]; then
      rm -f "$d.tmp"
      log "     ! $rel  hash mismatch after copy, retry $try/$MAX_TRIES"
      continue
    fi
    mv -f "$d.tmp" "$d"
    log "     + $rel  ($(numfmt --to=iec --suffix=B "$(stat -c%s "$d")" 2>/dev/null || stat -c%s "$d"))  sha256 ${h1:0:12}..."
    printf '%s  %s\n' "$h1" "$rel" >>"$tmp_shas"
    n_copied=$((n_copied + 1))
    return 0
  done
  log "     ! $rel  GAVE UP after $MAX_TRIES tries"
  n_failed=$((n_failed + 1))
  return 1
}

# summarise <results.csv> - epoch count and best mAP50-95, the metric this Ultralytics
# build actually selects best.pt on (NOT the 0.1/0.9 fitness blend; see #51).
summarise() {
  [ -f "$1" ] || { echo "no results.csv"; return; }
  awk -F, '
    NR==1 { for (i=1;i<=NF;i++) { h=$i; gsub(/^[ \t\r]+|[ \t\r]+$/,"",h)
              if (h=="epoch") e=i; if (h=="metrics/mAP50-95(B)") m=i; if (h=="metrics/mAP50(B)") m5=i }
            next }
    NF>3 { rows++; if ($m+0>best) { best=$m+0; bep=$e+0; b5=$m5+0 }; last=$e+0 }
    END { if (rows) printf "ep%d logged, best mAP50-95 %.5f @ep%d (mAP50 %.4f)", last, best, bep, b5
          else printf "no data rows" }' "$1"
}

if [ ! -d "$SRC" ]; then log "FATAL: SRC not found: $SRC"; exit 2; fi
mkdir -p "$DST" || { log "FATAL: cannot create DST: $DST"; exit 2; }

if [ "$#" -gt 0 ]; then arms=("$@"); else
  mapfile -t arms < <(cd "$SRC" && ls -1d */ 2>/dev/null | sed 's#/$##' | grep -Ev "$SKIP_RE")
fi

log "snapshot_runs.sh  $stamp"
log "  SRC $SRC"
log "  DST $DST"
log "  arms: ${arms[*]}"
log ""

for arm in "${arms[@]}"; do
  [ -d "$SRC/$arm" ] || { log "== $arm : NOT PRESENT at source, skipped"; continue; }
  log "== $arm : $(summarise "$SRC/$arm/results.csv")"
  for f in "${FILES[@]}"; do
    copy_verified "$SRC/$arm/$f" "$DST/$arm/$f" "$arm/$f"
  done
done

# Refresh the verified-hash list, keeping entries for arms we did not touch this run.
if [ -f "$verified_list" ]; then
  touched=$(awk '{print $2}' "$tmp_shas" | sed 's#/.*##' | sort -u)
  while read -r line; do
    a=$(printf '%s' "$line" | awk '{print $2}' | sed 's#/.*##')
    printf '%s\n' "$touched" | grep -qx "$a" || printf '%s\n' "$line" >>"$tmp_shas"
  done <"$verified_list"
fi
sort -k2 "$tmp_shas" | awk '!seen[$2]++' >"$verified_list"
rm -f "$tmp_shas"

log ""
log "copied $n_copied, unchanged $n_same, failed $n_failed"
log "hash list: $verified_list ($(wc -l <"$verified_list") entries)"
log ""
log "Verify any time with:  cd $DST && sha256sum -c .sha256.verified"
[ "$n_failed" -eq 0 ] || exit 1
