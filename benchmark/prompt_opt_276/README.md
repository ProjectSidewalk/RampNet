# prompt_opt_276 — artifacts of the VLM prompt experiments (#276)

Everything a `--gemini-definition`, `--gemini-exemplars` or `--pano-list` run reads lives
here, committed, so a leg is reproducible from its registry pins alone. The plan and the
write-up are `docs/vlm_prompt_276.md` (lands with the results).

## Pano lists (`--pano-list FILE@KEY`)

| file | what | how made |
|---|---|---|
| `sao_paulo_split.json` | `dev_train` (60) / `dev_select` (65): the two halves of the dev split (#51 convention: sao_paulo is the one split tuning may see) | `scripts/analysis/prompt_opt_276_splits.py`, seed 276; `--check` proves it rebuilds |
| `manual_gold_250.json` | `panos`: a 250-pano subset of manual_gold for the non-zero-shot arms | same script, same seed |

Rule: `dev_train` feeds the optimizer its exemplars; `dev_select` picks among candidates and
thresholds, once; nothing else is looked at before the frozen held-out run.

## Definitions (`--gemini-definition NAME` → `definitions/NAME.txt`)

The file replaces only the middle of the prompt (`detectors.detection_prompt_for`); the
instruction around it is unchanged. Whitespace is normalized on load, so wrapping is free.

| name | what | provenance |
|---|---|---|
| *(none)* | the seed, `detectors.CURB_RAMP_DEFINITION`, shared by every prompted model on the board | 2026-07 |
| `para1`, `para2`, `para3` | meaning-preserving paraphrases of the seed: the "best of K prompts" null for Step 3 | written 2026-11-09 before any model error from the Step 2 runs was looked at |
| `expert` | Jon's definition, written blind to the model's errors | Step 3 (pending) |
| `def276-<model>-r<k>` | the optimizer's candidates, one per accepted iteration; `trajectory.jsonl` beside them records the exemplar panos, the human verdicts and the dev_train score of each | Step 4 |

## Exemplars (`--gemini-exemplars NAME` → `exemplars/NAME.json`)

A list of `{"image": <file beside the json>, "caption": <text>}` in display order. The
images are 1024x1024 views with the region boxed, cut from `dev_train` only. The committed
bytes are what the model sees; the signature records their sha256.
