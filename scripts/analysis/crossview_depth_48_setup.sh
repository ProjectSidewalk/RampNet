#!/bin/bash
# Scratch env for the #48 monocular-depth arms on makelab2. Never touches .venv-eval:
# a new uv venv that sees .venv-eval's site-packages through a .pth (torch, timm, cv2...),
# with the few extra packages installed into the scratch venv itself.
set -x
R=/homes/gws/jonf/crossview48/depth
cd $R
df -h $R | tail -1
EVAL=/homes/gws/jonf/RampNet/.venv-eval
UV=/homes/gws/jonf/.local/bin/uv
[ -d venv ] || $UV venv --python $EVAL/bin/python venv
SP=$(ls -d $EVAL/lib/python3.12/site-packages)
echo "$SP" > venv/lib/python3.12/site-packages/_eval_env.pth
mkdir -p src && cd src
[ -d Depth-Anything-3 ] || git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout -q 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
mkdir -p Depth-Anything-3/stubs/moviepy && touch Depth-Anything-3/stubs/moviepy/__init__.py Depth-Anything-3/stubs/moviepy/editor.py
# The committed runs cloned these three at HEAD; the checkouts below are the commits those
# runs recorded (depth/<model>.meta.json -> prov.code_commit), pinned afterwards (#210 review).
[ -d ml-depth-pro ] || git clone -q https://github.com/apple/ml-depth-pro.git
git -C ml-depth-pro checkout -q 9e65e4dbe9568d23c546fcec53302b10445e109e
[ -d UniDepth ] || git clone -q https://github.com/lpiccinelli-eth/UniDepth.git
git -C UniDepth checkout -q 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
[ -d Metric3D ] || git clone -q https://github.com/YvanYin/Metric3D.git
git -C Metric3D checkout -q eb5b6fac0dc155e4e52f576e304fbf11655ff339
for d in Depth-Anything-3 ml-depth-pro UniDepth Metric3D; do echo "COMMIT $d $(git -C $d rev-parse HEAD)"; done
cd $R
$UV pip install --python venv/bin/python omegaconf addict einops plyfile trimesh pillow-heif mmengine 2>&1 | tail -5
$UV pip install --python venv/bin/python --no-deps -e src/ml-depth-pro 2>&1 | tail -2
$UV pip install --python venv/bin/python --no-deps -e src/UniDepth 2>&1 | tail -2
$UV pip install --python venv/bin/python pycolmap evo wandb 2>&1 | tail -3
# Metric3D's mono/utils/comm.py imports mmcv.utils unguarded; mmcv needs a CUDA build, so a
# stub re-exports mmengine's Config (the fallback Metric3D's own hubconf uses) instead.
mkdir -p src/stubs/mmcv/utils && touch src/stubs/mmcv/__init__.py
printf 'from mmengine import Config, DictAction\ndef collect_env():\n    return {}\ndef get_git_hash(*a, **k):\n    return ""\n' > src/stubs/mmcv/utils/__init__.py
$UV pip freeze --python venv/bin/python > freeze_scratch.txt
venv/bin/python -c "import sys, torch; print(sys.path); print(torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
