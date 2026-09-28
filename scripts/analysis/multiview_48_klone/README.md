# #48 pass 2 on klone

makelab2 runs the #48 challenger legs one at a time on one shared A40
(`scripts/analysis/multiview_challengers_48.sh`). There Molmo2-8B is partly offloaded to CPU
and runs at about 46 s per pano. After the 30 m widening (1,307 more panos), makelab2's second
pass would have taken about 24 h. These scripts precompute that pass's Molmo, Qwen3-VL-8B and
OWLv2 + Grounding DINO detections on klone's free `ckpt-all` partition. They then copy the
detection cache into makelab2's cache. compare.py reads the cache when each leg starts, so
makelab2's pass 2 reads these results instead of recomputing them. The YOLO leg (about 30 min)
stays on makelab2.

This offload is optional. Running `multiview_challengers_48.sh` on makelab2 alone fills the
same cache and exports the same detections, only slower; see `docs/multiview_48.md` §10.

Run 2026-09-27: arrays `40770191` (Molmo, 4 shards), `40770192` (Qwen, 3) and `40770193`
(OWLv2 + GDINO, 2), with RampNet at commit `4cef05c` (printed at the top of every task log).
They ran 08:40–10:00 PT, 9.26 GPU-hours, $0. Costs are in `docs/compute_cost.md`, and per-leg
rows are in `analysis_out/usage_log.jsonl`.

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

## What to change before you run it

Every path and the account below are Jon's. A new user changes them:

| what | value in the scripts | where |
|---|---|---|
| klone work dir `W` | `/gscratch/scrubbed/jfroehli/mv48` | `run_mv48.slurm`, `setup_mv48.slurm`, `keycheck.py` (all read `$W` from the environment first), `sync_mv48.sh`, and the `#SBATCH --output` lines of both `.slurm` files (Slurm does not expand variables there, so edit them) |
| Slurm account | `--account=makelab` | both `.slurm` files |
| `uv` binary | `/mmfs1/home/jfroehli/.local/bin/uv` | `setup_mv48.slurm` (`UV` env var overrides) |
| klone user in the job check | `squeue -u jfroehli` | `sync_mv48.sh` |
| makelab2 detection cache | `/homes/gws/jonf/mv48/model_cache` | `sync_mv48.sh`; must be the `WORK` of `multiview_challengers_48.sh` |
| makelab2 pano archive | `/projects/makeabilitylab/sidewalk-auto-labeler/runs/richmond/panos` | the makelab2 commands below; the `ARCHIVE` of `multiview_challengers_48.sh` |
| RampNet commit on klone | `REF=4cef05c` | `setup_mv48.slurm` (`REF` env var overrides) |

The pano archive is not published. The 2,743 unjudged Richmond neighbourhood panos exist only
there (`docs/multiview_48.md` §10 says what would unblock it).

## Commands, in order

The GPU request in `run_mv48.slurm` is `--gres=gpu:1` with
`--constraint="l40s|a40|l40|a100|h200"` (any 40 GB+ GPU). The 2026-09-27 arrays were submitted
with `--gres=gpu:l40s:1` and widened while still pending, after one Molmo task had started on an
L40S, with
`for j in 40770191 40770192 40770193; do scontrol update jobid=$j gres=gpu:1 features="l40s|a40|l40|a100|h200"; done`.
The other 8 tasks ran under the widened request (6 A40, 2 A100). The committed script now asks
for that directly, so no `scontrol` step is needed.

**1. makelab2: freeze the envs and pack the panos and the current cache.** Run from a RampNet
checkout that has this directory. `ship_files.txt` (1,431 lines) is the 1,307 new panos plus
the 124 judged ones, as `<id>.jpg`. It is built here rather than committed because it follows
from two committed files.

```bash
X=/homes/gws/jonf/mv48/klone_xfer
mkdir -p $X
uv pip freeze --python /homes/gws/jonf/envs/molmo/bin/python > $X/req_molmo.txt
uv pip freeze --python /homes/gws/jonf/RampNet/.venv-eval/bin/python > $X/req_eval.txt
# req_eval_cu126.txt = req_eval.txt with torch/torchvision -> +cu126 and the
# nvidia-*/cuda-*/triton lines removed (the committed copy is what ran)
python3 -c 'import json; print("\n".join(sorted(json.load(open("benchmark/richmond/verdicts.json"))["panos"])))' > $X/judged_panos.txt
cat scripts/analysis/multiview_48_klone/new_panos.txt $X/judged_panos.txt | tr -d '\r' | sed '/^$/d; s/$/.jpg/' > $X/ship_files.txt
wc -l $X/ship_files.txt    # 1431
cd /projects/makeabilitylab/sidewalk-auto-labeler/runs/richmond/panos
tar cf $X/panos.tar -T $X/ship_files.txt
tar cf $X/cache_seed.tar -C /homes/gws/jonf/mv48 model_cache
```

**2. Relay to klone** (from WSL, reusing both control masters), and copy this directory's
scripts into `W`. `setup_mv48.slurm` reads `req_*.txt` from `W`, `make_shards.py` reads
`new_panos.txt` from `W`, and both `.slurm` files write their logs to `W/logs`, which must
exist before `sbatch` or the job fails at start.

```bash
W=/gscratch/scrubbed/jfroehli/mv48
ssh klone "mkdir -p $W/logs"
scp -3 makelab2:/homes/gws/jonf/mv48/klone_xfer/panos.tar makelab2:/homes/gws/jonf/mv48/klone_xfer/cache_seed.tar klone:$W/
# from a RampNet checkout that has this directory:
scp scripts/analysis/multiview_48_klone/* klone:$W/
```

**3. klone: set up, shard, run, check.** In `W`:

```bash
cd /gscratch/scrubbed/jfroehli/mv48
mkdir -p panos && tar xf panos.tar -C panos && tar xf cache_seed.tar
sbatch setup_mv48.slurm      # envs, RampNet clone at REF, HF snapshots; wait for SETUP_DONE in logs/setup_*.out
python3 make_shards.py $PWD molmo:4 qwen8b:3 open:2
sbatch --array=0-3 --job-name=mv48-molmo run_mv48.slurm molmo
sbatch --array=0-2 --job-name=mv48-qwen8b run_mv48.slurm qwen8b
sbatch --array=0-1 --job-name=mv48-open run_mv48.slurm open
# before the arrays: the judged panos are hits; after: the new ones are too
envs/molmo/bin/python keycheck.py molmo:allenai/Molmo2-8B
envs/eval/bin/python keycheck.py qwen:Qwen/Qwen3-VL-8B-Instruct owlv2 gdino
```

**4. Sync and finish on makelab2.** `sync_mv48.sh`, run in WSL, copies klone's cache into
makelab2's every 20 min with `rsync --ignore-existing` until the arrays drain, then once more.
It never overwrites an entry makelab2 wrote. Then run the makelab2 script as usual; it finds
the four legs cached, runs the YOLO trio on the new panos, and exports.

```bash
bash scripts/analysis/multiview_48_klone/sync_mv48.sh      # WSL; ends with SYNC_DONE
bash scripts/analysis/multiview_challengers_48.sh          # makelab2
```

After the run, record the cost: pull `sacct` for the job ids and parse it into
`analysis_out/compute_log.jsonl` (the commands are in `docs/compute_cost.md`, #48 klone
section), and copy the klone `usage_log.jsonl` rows into `analysis_out/usage_log.jsonl`.
