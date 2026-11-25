# In-progress verdict backups (not a result)

Partial exports of the Phase 1 GT page, committed as backups while rating is under way. They are not
the final per-rater export (`crosswalk_gt_verdicts_<rater>.json` in this folder, once rating is done),
and nothing reads them. Each file is the page's export byte for byte.

| file | rater | rows rated | exported (UTC) | sheet sha256 |
|---|---|---|---|---|
| `crosswalk_gt_verdicts_JEF_inprogress_2026-11-25_n84.json` | JEF | 84 of 320 (Columbus 36, Amsterdam 31, Chicago 11, São Paulo 6) | 2026-11-25T14:04:40Z | `53eb4b8cc18a…` (matches `sheet.csv`) |

**Two rubric versions in one pass.** 40 of the 84 rows were rated under `phase1-2026-11-02` (2026-11-03)
and 44 under `phase1-2026-11-03` (from 2026-11-25), which adds Q1 codes T (raised or textured, no paint),
the C and V toggles, and Q2 code 5. Each verdict carries its `rubric` field, so the two can be read apart
or the earlier 40 re-checked under the later rubric before the per-city table is computed.
