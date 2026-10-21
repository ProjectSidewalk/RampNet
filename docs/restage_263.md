# Re-staging the Stage 2 dataset on klone from the Hub parquet shards (#263)

**Status:** final, 2026-10-21. The staged copy is `/gscratch/scrubbed/jfroehli/rampnet_dataset_v2`
on klone; its counts were verified by the committed script. Scrubbed storage purges after about
21 idle days, so a copy there is a convenience, not an input; the input is the Hub dataset.

## Why

The copy of the Stage 2 dataset that #51, #84, #135 and #82 trained from
(`/gscratch/scrubbed/jfroehli/rampnet_dataset`) had been partly purged by 2026-10-20: `train/`
held 114,667 of 300,126 files and `val/` and `test/` were empty. #263's control arm and
label-budget curve need the full copy back.

## What failed first

`download_dataset.py` (repo root) is the documented route. It calls `load_dataset()`, which
downloads the parquet shards (431 GB) and then converts every shard into an Arrow cache
(another ~460 GB) before `.map()` writes the jpg/json pairs (465 GB). On 2026-10-20 the
conversion died after 2 h 02 m with `OSError: [Errno 122] Disk quota exceeded` (klone job
`42083469`, launcher `stage_two/run_restage_263.slurm`, committed with #266). `/mmfs1` was at
99 % cluster-wide and `/gscratch/makelab` at 1019 of 1024 GB, so a three-copy staging path had
no room.

## What ran

`scripts/restage_from_parquet.py` reads the cached shards with pyarrow and writes the pairs
directly, so the only space it needs is the output. Launcher
`stage_two/run_restage_parquet_263.slurm`, klone job `42090420`, 2026-10-20, two incarnations
on the preemptible `ckpt` partition: 4 h 14 m on n3081 until preemption, then 37 m on n3026
after the requeue (the resume skipped every pano already written). 12 CPUs, no GPU, $0.
Output 462 GB. The second incarnation ends `FAILED 1:0` in `sacct` because the script's count
check saw the nine placeholder rows (next section), not because anything was lost.

Three deliberate differences from `download_dataset.py`:

- the jpg is the parquet's own bytes (the file uploaded to the Hub), not a PIL re-encode at
  quality 95;
- the Hub's `validation` split is written to `val/`, as before;
- rows whose image bytes are not a JPEG are quarantined rather than written.

## The nine placeholder rows

The raw write produced 150,066 / 42,878 / 21,441 panos, three more per split than every
RampNet document quotes (150,063 / 42,875 / 21,438). The extras are placeholder rows in the
Hub dataset: `pano_id` `file_1`, `file_2`, `file_3` in each split, each with a 19-byte image
blob beginning `dumm` that no decoder opens. `download_dataset.py` skips them only because its
PIL re-encode raises inside a `try`, which is where the published counts come from. They now
sit in `rampnet_dataset_v2/_quarantine/<split>/` with a sorted manifest. Reported on #150 as a
dataset-card item: Hub row counts and usable pano counts differ by nine.

History, for the record: job `42090420` ran the script before the quarantine step existed; the
nine files were moved by hand and `--check` re-run. The committed script does both in one pass,
and its `--check` was run against the staged copy on 2026-10-21 (output above). The committed
launcher's header was expanded after the run; its `#SBATCH` directives and commands are the
ones that ran, with `OUT`, `HF_HOME`, `PY` and `REPO` made overridable.

## Verify

```bash
python scripts/restage_from_parquet.py --snapshot <unused in check mode> \
    --out /gscratch/scrubbed/jfroehli/rampnet_dataset_v2 --check
# [train] pairs 150063 (expected 150063), jpg-only 0, json-only 0: OK
# [val]   pairs 42875  (expected 42875),  jpg-only 0, json-only 0: OK
# [test]  pairs 21438  (expected 21438),  jpg-only 0, json-only 0: OK
# [quarantine] 18 files listed in .../_quarantine/MANIFEST.txt
```

The check needs no shard cache, which was removed after verification. The control launcher in
#266 refuses to start unless `train/` and `val/` hold exactly these counts.

## Reproduce from a clean clone

```bash
export HF_HOME=/gscratch/scrubbed/$USER/hf
huggingface-cli download projectsidewalk/rampnet-dataset --repo-type dataset   # 431 GB of shards
cd <clone> && mkdir -p logs
OUT=/gscratch/scrubbed/$USER/rampnet_dataset_v2 sbatch stage_two/run_restage_parquet_263.slurm
```

Not checked: whether the jpg bytes here equal the files the paper's Stage 1 run wrote to disk
in 2025; the Hub copy is the published artifact and is what every later run reads.
