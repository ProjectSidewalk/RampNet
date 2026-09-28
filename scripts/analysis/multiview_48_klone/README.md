# #48 pass 2 on klone

makelab2 runs the #48 challenger legs one at a time on one shared A40
(`scripts/analysis/multiview_challengers_48.sh`). There Molmo2-8B is partly offloaded to CPU
and runs at about 46 s per pano. After the 30 m widening (1,307 more panos), makelab2's second
pass would have taken about 24 h. These scripts precompute that pass's Molmo, Qwen3-VL-8B and
OWLv2 + Grounding DINO detections on klone's free `ckpt-all` partition. They then copy the
detection cache into makelab2's cache. compare.py reads the cache when each leg starts, so
makelab2's pass 2 reads these results instead of recomputing them. The YOLO leg (about 30 min)
stays on makelab2.

Run 2026-09-27: arrays `40770191` (Molmo, 4 shards), `40770192` (Qwen, 3) and `40770193`
(OWLv2 + GDINO, 2). They ran 08:40–10:00 PT, 9.26 GPU-hours, $0. Costs are in
`docs/compute_cost.md`, and per-leg rows are in `analysis_out/usage_log.jsonl`.

## Why the results are interchangeable with makelab2's

- **Same cache key.** `compare.cache_key` hashes the label, the detector signature, the bundle
  directory's basename and the pano id. Signatures are config only: model id, prompt,
  thresholds, coordinate settings. No host, GPU or path goes in. Each shard is therefore a
  directory named `richmond_neighbourhood`. `keycheck.py` confirms the keys: the 124 judged
  panos, seeded from makelab2's cache, are hits for every leg.
- **Same inputs.** The panos are the same JPEG files from the labeler's native-res archive.
  Every HF snapshot matches makelab2's: Molmo2-8B `e28fa28`, Qwen3-VL-8B `0c351dd`, OWLv2
  `95e2693`, Grounding DINO `12bdfa3`.
- **Package versions.** `req_molmo.txt` and `req_eval_cu126.txt` are `uv pip freeze` of
  makelab2's two envs. The eval env's torch 2.13.0 is the cu126 wheel instead of cu130, and
  its CUDA 13 runtime pins are dropped.
- **Not bitwise.** Pass 2's new panos ran on L40S/A40/A100. Pass 1 ran on makelab2's A40 with
  Molmo partly on CPU. Floating-point differences across GPUs are possible. Every usage row
  names its host.

## Commands, in order

makelab2 side: freeze the envs, then pack the panos and the current cache.

```bash
uv pip freeze --python /homes/gws/jonf/envs/molmo/bin/python > req_molmo.txt
uv pip freeze --python /homes/gws/jonf/RampNet/.venv-eval/bin/python > req_eval.txt
# req_eval_cu126.txt = req_eval.txt with torch/torchvision -> +cu126 and the
# nvidia-*/cuda-*/triton lines removed
cd /projects/makeabilitylab/sidewalk-auto-labeler/runs/richmond/panos
tar cf panos.tar -T ship_files.txt          # new_panos.txt + the 124 judged, as <id>.jpg
tar cf cache_seed.tar -C /homes/gws/jonf/mv48 model_cache
```

Relay to klone (from WSL, reusing both control masters), then set up and run there:

```bash
scp -3 makelab2:.../panos.tar makelab2:.../cache_seed.tar klone:/gscratch/scrubbed/jfroehli/mv48/
# on klone, in /gscratch/scrubbed/jfroehli/mv48:
mkdir -p panos && tar xf panos.tar -C panos && tar xf cache_seed.tar
sbatch setup_mv48.slurm                      # envs, branch clone, HF snapshots
python3 make_shards.py $PWD molmo:4 qwen8b:3 open:2
sbatch --array=0-3 --job-name=mv48-molmo run_mv48.slurm molmo
sbatch --array=0-2 --job-name=mv48-qwen8b run_mv48.slurm qwen8b
sbatch --array=0-1 --job-name=mv48-open run_mv48.slurm open
envs/molmo/bin/python keycheck.py molmo:allenai/Molmo2-8B
envs/eval/bin/python keycheck.py qwen:Qwen/Qwen3-VL-8B-Instruct owlv2 gdino
```

`sync_mv48.sh`, run in WSL, copies klone's cache into makelab2's every 20 min with
`--ignore-existing` until the arrays drain. It never overwrites an entry makelab2 wrote.
