#!/bin/bash
# Build the tagger training environment on klone, on purchased storage (#86 plan item 4).
#
# Pinned to what trained the #86 tag benchmark of record on makelab2 (docs/tag_benchmark_86.md:
# Python 3.10.20, torch 2.4.1+cu121, torchvision 0.19.1, scikit-learn 1.7.2, pandas 2.3.3,
# timm 1.0.30, no xformers), so a klone arm differs from a makelab2 arm by hardware only.
# Also clones sidewalk-tagger-ai at the benchmark's pinned commit and fetches the DINOv2
# backbone, sha256-checked, so the login node never does the download.
#
# The environment goes to /gscratch/makelab (never purged), not /gscratch/scrubbed: the
# scrubbed envs were found purge-damaged on 2026-09-23 ("no codec search functions
# registered"), the same failure scripts/model_comparison/run_build_yolo_env.slurm records.
# uv's cache and its managed Python go there too; klone homes are capped at 10 GB.
#
#   mkdir -p /gscratch/makelab/jonf/context_fov_86/logs && cd /gscratch/makelab/jonf/context_fov_86
#   sbatch scripts/analysis/context_fov_86_env.slurm        # ~10 min
#
#SBATCH --job-name=tagger_env_build
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/tagger_env_build_%j.out
set -eo pipefail
WORK=${WORK:-/gscratch/makelab/jonf/context_fov_86}
ENV=${TAGGER_ENV:-/gscratch/makelab/jonf/envs/tagger}
UV=${UV:-$HOME/.local/bin/uv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/gscratch/makelab/jonf/uv_cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/gscratch/makelab/jonf/uv_python}
TAGGER_SHA=3b7405cd3206ece631cb7a65e22b1ab219df4b75      # scripts/analysis/tag_benchmark_86.py TAGGER_SHA
mkdir -p "$WORK" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
date -u +%FT%TZ; hostname
[ -x "$ENV/bin/python" ] || "$UV" venv --python 3.10.20 "$ENV"
"$UV" pip install --python "$ENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match \
  torch==2.4.1 torchvision==0.19.1 \
  numpy==1.26.4 pillow==12.3.0 pandas==2.3.3 scikit-learn==1.7.2 scipy==1.15.3 timm==1.0.30 \
  omegaconf==2.3.1 fvcore==0.1.5.post20221221 iopath==0.1.10 matplotlib==3.10.9 requests==2.34.2 tqdm==4.70.1
"$ENV/bin/python" -c "import sys, torch, torchvision, sklearn, pandas, timm, omegaconf; \
print(sys.version.split()[0], torch.__version__, torchvision.__version__, torch.version.cuda, sklearn.__version__, pandas.__version__, timm.__version__)"
TAGGER=$WORK/sidewalk-tagger-ai
[ -d "$TAGGER" ] || git clone -q https://github.com/ProjectSidewalk/sidewalk-tagger-ai.git "$TAGGER"
git -C "$TAGGER" checkout -q $TAGGER_SHA
cd "$WORK"
[ -f dinov2_vitb14_reg4_pretrain.pth ] || curl -fsSL -o dinov2_vitb14_reg4_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
sha256sum -c - <<SUMS
73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  dinov2_vitb14_reg4_pretrain.pth
SUMS
ln -sfn "$WORK/dinov2_vitb14_reg4_pretrain.pth" "$TAGGER/"
echo ENV_BUILD_DONE; date -u +%FT%TZ
