#!/bin/bash
#SBATCH --job-name=talk-splat
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/makelab/jonf/talk_splat/logs/train_%j.out
# Train 3D Gaussians for each corner from the COLMAP model that uchicago_2026_dense_scene.py
# wrote, then render the shared camera path (path.json from uchicago_2026_mesh.py path).
#   SLUGS="richmond_99 bend_7" ITERS=15000 sbatch klone_splat_train.sbatch
set -x
R=/gscratch/makelab/jonf/talk_splat
. $R/venv/bin/activate
export PATH=/sw/gcc/11.2.0/bin:/sw/cuda/12.2.2/bin:$PATH CC=/sw/gcc/11.2.0/bin/gcc CXX=/sw/gcc/11.2.0/bin/g++
export LD_LIBRARY_PATH=/sw/gcc/11.2.0/lib64:$LD_LIBRARY_PATH CUDA_HOME=/sw/cuda/12.2.2 TORCH_CUDA_ARCH_LIST="8.9"
cd $R
for s in ${SLUGS:-richmond_99 bend_7}; do
  python klone_splat_train.py train --data $R/data/$s/colmap --out $R/out/$s --iters ${ITERS:-15000}
  python klone_splat_train.py render --out $R/out/$s --path $R/data/$s/path.json
  tar czf $R/out/${s}_frames.tgz -C $R/out/$s frames train.json
done
echo TRAIN_DONE
