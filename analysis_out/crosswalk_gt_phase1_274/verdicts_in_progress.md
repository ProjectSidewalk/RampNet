# In-progress verdict backups (not a result)

Partial exports of the Phase 1 GT page, committed as backups while rating is under way. They are not
the final per-rater export (`crosswalk_gt_verdicts_<rater>.json` in this folder, once rating is done).
Each file is the page's export byte for byte; each newer one is a superset of the last, which is removed
(git history keeps it).

| file | rater | rows rated | exported (UTC) | sheet sha256 |
|---|---|---|---|---|
| `crosswalk_gt_verdicts_JEF_inprogress_2026-11-25_n134.json` | JEF | 134 of 320, rows r000–r133 (Columbus 53, Amsterdam 49, Chicago 16, São Paulo 16) | 2026-11-25T14:26:52Z | `53eb4b8cc18a…` (matches `sheet.csv`) |

Earlier backups: `…_n84.json` (2026-11-25T14:04:40Z, commit 647e3593), a strict subset of n134 with no
changed verdicts.

**Two rubric versions in one pass.** 40 rows were rated under `phase1-2026-11-02` (2026-11-03) and the
rest under `phase1-2026-11-03` (from 2026-11-25), which adds Q1 code T (raised or textured, no paint), the
C and V toggles, and Q2 code 5. Each verdict carries its `rubric` field.

## Interim read at n = 134 (2026-11-25; decides nothing)

Read at Jon's request mid-round. The sheet is in shuffled order and he rated it in order, so r000–r133 is
close to a random 42% of the sheet; the reading rule in `docs/crosswalk_gt_phase1_274.md` still applies
to the full 320, and nothing here changes it. Reproduce with `gt_page.py merge --sheet sheet.csv
--verdicts crosswalk_gt_verdicts_JEF_inprogress_2026-11-25_n134.json --prerate prerate_ai.csv --out X.csv`
(per city × stratum table and AI agreement), pooled below across the four cities with Wilson 95% intervals.

| stratum | paint within 5 m (Y or R) | paint at the node (Y) |
|---|---|---|
| marked | 44/44 [0.92, 1.00] | 38/44 |
| untagged | 28/28 [0.88, 1.00] | 24/28 |
| `unmarked` (crowd-labelled) | 13/23 [0.37, 0.74] | 11/23 |
| osm_only (no crowd label within 15 m) | 20/39 [0.36, 0.66] | 17/39 |

- osm_only by OSM class: `marked` 18/18 painted, `unmarked` 1/13, untagged 1/8.
- Crowd label on the crossing (Q2 1 or 2, bike rows aside): 85/95 [0.82, 0.94].
- Blind AI pre-rate vs Jon (human mapped to the 11-02 rubric): paint within 5 m agrees on 128/134
  (0.96); where the AI says paint, Jon agrees 101/103 [0.93, 0.99]; where it says no or cannot tell, Jon
  found paint on 4/31.
- Per-city cells are mostly ≤ 12 rows, so by the reading rule they are counts, not rates.
