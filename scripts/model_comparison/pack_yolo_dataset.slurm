#!/bin/bash
# Pack one YOLO dataset into a SquashFS image, for transfer klone -> Tillicum.
#
# WHY SQUASHFS AND NOT rsync/scp/tar
# The dataset is 286 GB in ~911,000 files (tiles 210 GB / 718,415; pano 76 GB /
# 192,938 -- measured 2026-07-30). That is metadata-bound, not bandwidth-bound: `du -sh`
# over the tiles tree beat three timeouts before finishing on a 550 s budget, and even
# `ls -lU | head -201` on tiles/images/train timed out.
#
#   - rsync/scp of the tree pays ~911k round trips. No.
#   - tar fixes the TRANSFER but not the DESTINATION: untarring recreates all 911k files
#     on /gpfs, paying the metadata cost again and permanently.
#   - SquashFS pays it ONCE, here, and the destination only ever sees ONE file. The
#     training job mounts it read-only and the kernel serves the tree from a compact
#     in-image index. This is also what UW-IT recommends for many-small-file datasets
#     (Sumaiya, 2026-07-30).
#
# WHY A SLURM JOB AND NOT A LOGIN NODE
# klone reaps heavy login processes, and that reap also kills the SSH control master
# every agent session depends on. Never run this interactively.
#
# WHERE IT WRITES
# /gscratch/scrubbed, NOT /gscratch/makelab: lab storage is 84% full with ~168 GB free
# (checked 2026-07-30) and cannot hold this. scrubbed has ~15 TB free, and its ~21-day
# purge does not matter for a transient staging area.
#
#   DS=pano  sbatch -A ckpt-makelab scripts/model_comparison/pack_yolo_dataset.slurm
#   DS=tiles sbatch -A ckpt-makelab scripts/model_comparison/pack_yolo_dataset.slurm
#
#SBATCH --job-name=yolo_pack
#SBATCH -p ckpt
#SBATCH -q ckpt
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=NONE
#SBATCH --output=logs/yolo_pack_%j.out
#SBATCH --error=logs/yolo_pack_%j.err

set -euo pipefail

DS="${DS:?set DS to pano or tiles}"
SRC="${SRC:-/gscratch/scrubbed/$USER/yolo}"
OUT="${OUT:-/gscratch/scrubbed/$USER/yolo_squashfs}"
mkdir -p "$OUT"

IMG="$OUT/${DS}.sqfs"
SUM="$IMG.sha256"

echo "--- pack $DS ---"
echo "src: $SRC/$DS   out: $IMG"
df -h "$OUT" | tail -1

# Idempotent: a completed, verified image is left alone, so a ckpt requeue or a manual
# resubmit is free rather than a restart from zero.
if [ -f "$IMG" ] && [ -f "$SUM" ]; then
    echo "[skip] image exists; verifying..."
    if ( cd "$OUT" && sha256sum -c "$(basename "$SUM")" ); then
        echo "[skip] verified, nothing to do"; exit 0
    fi
    echo "[warn] checksum FAILED -- rebuilding"
    rm -f "$IMG" "$SUM"
fi

# ---------------------------------------------------------------------------------
# THE ULTRALYTICS LABEL CACHE IS THE SUBTLE PART -- READ BEFORE CHANGING THIS.
#
# Ultralytics writes labels/<split>.cache next to the labels and validates it against a
# hash of the label+image FILE PATHS. Two consequences:
#
#   1. The .cache files MUST be inside the image. They already exist on klone (built
#      2026-07-25/26), and a read-only mount cannot create them. Without them every run
#      rescans ~911k files -- exactly the cost this whole exercise exists to avoid.
#
#   2. The hash covers absolute paths, so mounting at a DIFFERENT path than the one the
#      cache was built under INVALIDATES it and forces that same rescan. Mount the image
#      on Tillicum at a stable path and expect to regenerate the cache ONCE there (it
#      needs a writable overlay, or a one-off run with labels/ bind-mounted writable).
#      Budget one slow first epoch; do not mistake it for the steady-state cost.
#
# This is called out in docs/tillicum.md too, because getting it wrong is invisible --
# training still works, it is just silently slow forever.
# ---------------------------------------------------------------------------------
echo "[check] label caches that will be packed:"
ls -la "$SRC/$DS/labels/"*.cache 2>/dev/null || echo "  NONE -- expect a full rescan on first use"

# -noD: do NOT compress data blocks. The payload is JPEG, already compressed; deflating
# it would burn hours of CPU for ~nothing and make reads slower. Metadata/inodes are
# still compressed, which is where the win actually is.
# -no-xattrs: nothing here uses them, and they bloat the index.
echo "[mksquashfs] starting $(date)"
mksquashfs "$SRC/$DS" "$IMG" \
    -noD \
    -no-xattrs \
    -processors "${SLURM_CPUS_PER_TASK:-8}" \
    -info
echo "[mksquashfs] done $(date)"

# The checksum travels with the image and is verified after transfer, same discipline as
# the weight snapshot's MANIFEST.md.
( cd "$OUT" && sha256sum "$(basename "$IMG")" > "$(basename "$SUM")" )

ls -lh "$IMG"
cat "$SUM"

cat <<EOF

[ok] $DS packed.

Next, on Tillicum -- note /gpfs/scrubbed, not the 1 TB /gpfs/projects/makelab quota
(UW-IT recommends scrubbed for active many-small-file datasets):

  # mount read-only, no untar, no 911k files ever created on /gpfs
  apptainer exec --nv \\
      --bind /gpfs/scrubbed/\$USER/${DS}.sqfs:/data/${DS}:image-src=/ \\
      <image.sif> <command>

Verify the checksum on arrival BEFORE trusting it:
  sha256sum -c $(basename "$SUM")
EOF
