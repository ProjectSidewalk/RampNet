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
downloads the parquet shards (431 GiB on disk, `du -sh`) and then converts every shard into an
Arrow cache of about the same size before `.map()` writes the jpg/json pairs. On 2026-10-20 the
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
Output 462 GiB (`du -sh`; not derivable from a committed file). The second incarnation ends
`FAILED 1:0` in `sacct` because the script's count check saw the nine placeholder rows (next
section), not because anything was lost.

Four deliberate differences from `download_dataset.py`:

- the jpg is the parquet's own bytes (the file uploaded to the Hub), not a PIL re-encode at
  quality 95;
- the json is compact rather than `indent=4`; same keys, same values;
- the Hub's `validation` split is written to `val/`, as before;
- rows whose image bytes are not a JPEG are quarantined rather than written.

## The nine placeholder rows

The raw write produced 150,066 / 42,878 / 21,441 panos, three more per split than every
RampNet document quotes (150,063 / 42,875 / 21,438). The extras are placeholder rows in the
Hub dataset: `pano_id` `file_1`, `file_2`, `file_3` in each split, each with a 19-byte image
blob beginning `dumm` that no decoder opens. `download_dataset.py` never writes them: the image
decode raises inside its per-row `try` (in the `datasets` image feature on a current install,
checked with `datasets` 5.0.0; the 2025 environment was not re-run) and the row is skipped
before any file is written. That is where the published counts come from. They now sit in
`rampnet_dataset_v2/_quarantine/<split>/` with a sorted manifest. Reported on #150 as a
dataset-card item: Hub row counts and usable pano counts differ by nine.

History, for the record: job `42090420` ran the script before the quarantine step, the `.tmp`
sweep and `COUNTS.json` existed. The nine files were moved by hand, then `--check --record` was
run once on 2026-10-21 against the staged copy to write the manifest and the per-split id
hashes (the shard cache had already been removed). The committed script does the quarantine in
write mode on an empty or partially written output; over an output of the old version it also
removes any placeholder pair it finds there. The committed launcher's header, provenance lines
and overridable variables (`REPO`, `RAMPNET_ENV`, `HF_HOME`, `HF_REVISION`, `OUT`) were added
after the run; its `#SBATCH` directives and the python command are the ones that ran.

## Verify

```bash
python scripts/restage_from_parquet.py --out /gscratch/scrubbed/jfroehli/rampnet_dataset_v2 --check
```

It exits 0 only when all three split directories exist and hold exactly 150,063 / 42,875 /
21,438 complete pairs, no jpg-only or json-only strays, no `.tmp` files (in the splits or under
`_quarantine/`), per-split id hashes equal to the committed reference below and to
`COUNTS.json`, and a quarantine manifest equal to the directory. The reference hashes (sha256
over the sorted, LF-joined pano ids; val and test re-derived independently from the Hub's
`pano_id` columns during review and equal) are in the script as `REFERENCE_IDS_SHA256`:

| split | ids | sha256 |
| :--- | ---: | :--- |
| train | 150,063 | `63e2755d6434a364be2d54ef2f8bbc8e561c1defba480cacb0c508c8ba1eb422` |
| val | 42,875 | `0d8aaf9e45c08dcc402794b2493ff4fd723e20cc64847fa72c4f3fcb475e61a9` |
| test | 21,438 | `0c431c2aba722d4ef129917f0d511d62d819e9199c0f1c527f93db6f47f370a0` |

A re-stage from a clean clone whose ids hash differently is not the published dataset, whatever
its counts say. The run of record's
output is quoted in the PR (#268). The check needs no shard cache, which was removed after
verification. The control launcher in #266 refuses to start unless `train/` and `val/` hold
exactly these counts.

## Reproduce from a clean clone

```bash
export HF_HOME=/gscratch/scrubbed/$USER/hf
huggingface-cli download projectsidewalk/rampnet-dataset --repo-type dataset \
    --revision ee882e3f3c779dc13182f307bca616e50d9b8c5c      # the revision of record; 431 GiB
cd <clone> && mkdir -p logs
RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 \
    OUT=/gscratch/scrubbed/$USER/rampnet_dataset_v2 sbatch stage_two/run_restage_parquet_263.slurm
python scripts/restage_from_parquet.py --out /gscratch/scrubbed/$USER/rampnet_dataset_v2 --check
```

The launcher runs the submitting checkout (`SLURM_SUBMIT_DIR`), needs `RAMPNET_ENV` like the
other launchers, and refuses to guess when more than one snapshot revision is present.

Not checked: whether the jpg bytes here equal the files the paper's Stage 1 run wrote to disk
in 2025; the Hub copy is the published artifact and is what every later run reads.
