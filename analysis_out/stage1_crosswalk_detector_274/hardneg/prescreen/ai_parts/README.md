# AI pre-screen verdicts, hard negatives (#274, 2026-11-06)

`ai_part_{1..8}.csv`: one row per contact-sheet cell (`sheet`, `cell`, `verdict`, `reason`), merged with
`sheets.csv` by `score_274.py prescreen-merge` into `../prescreen.csv`.

**Who produced them.** Eight blind Claude Code subagents on a Sonnet model, one per part, over 458 numbered contact
sheets (5,485 cards). Each saw only the crop with the dashed disc a negative would train: no confidence, city or
model. Source: the commit that added these files (4868ecbb).

**Not recorded:** the exact Sonnet model id, the instructions the subagents were given, and the wall-clock time.
They were not written down at the time and cannot be recovered from the repo. The pre-screen only chose which cards
Jon saw; every negative that trains is Jon's verdict.
