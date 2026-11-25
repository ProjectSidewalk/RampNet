# AI pre-screen verdicts, brick-crosswalk positives (#274 s4 step 1, 2026-11-08)

`ai_part_{1..4}.csv`: one row per contact-sheet cell, merged by `brick_positives.py prescreen` into
`../prescreen.csv`.

**Who produced them.** Four blind AI readers over 77 contact sheets (923 labels). Source: the commit that added
these files (f1d61c13) and docs/stage1_crosswalk_sdot_274.md, s4 step 1.

**Not recorded:** which model the readers were (neither the commit nor the doc names it), the instructions they
were given, and the wall-clock time. The doc's finding that the readers were wrong on 55 of their 77 rejections is
about these files, so it describes an unidentified model. Every positive that trains is Jon's verdict or an AI
crosswalk call that passed the trust rule Jon checked (120 of 120 brick, 39 of 40 paint).
