# Tag review protocol (RampNet 2.0 plan item 3)

**Status: working draft, 2026-09-22; updated 2026-09-23**, paired with the rubric in
[`docs/tag_rubric_draft.md`](tag_rubric_draft.md). Issue
[#86](https://github.com/ProjectSidewalk/RampNet/issues/86).

**One rater for now** (rubric decision D1, Jon, 2026-09-23): Jon reviews the 500 items in
`benchmark/tag_review/review_list.csv` on production. The pass becomes one committed file,
`benchmark/tag_review/jonfroehlich.json`, with the rubric text embedded. A second, blind pass is
deferred, not ruled out; the sections on the review sheet and on comparing two passes below
describe how it would run, and the tooling for it stays built.

## Before a pass

1. Read the rubric block of `docs/tag_rubric_draft.md` (between the `rubric:begin` and
   `rubric:end` markers) and note its version (`tag-rubric-v1.1-draft` at the time of writing).
   If the rubric changes partway through, note the item where it changed.
2. Note the pass start time in UTC. The production pull only counts edits and votes after it, so
   an older expert-validate of the same label is not mistaken for this pass.
3. Create an empty sidecar file, `benchmark/tag_review/<rater>__sidecar.csv`, with the header
   `item_id,cannot_judge,cannot_judge_tags,note`. Production has no field for a per-tag "cannot
   judge" or a note, so they go here (decision D12).

## The rater: on production

For each row of the list, in `item_id` order:

1. **Open `editor_url`** (`https://<host>/gallery?labelType=CurbRamp&labelId=<id>`) signed in as
   an Owner or Administrator. `labelmap_url` opens the same label on the label map if the gallery
   view is not enough.
2. **Is it a curb ramp?** If not, vote **Disagree** and go to the next item.
3. **If you cannot tell whether it is a curb ramp** (occluded, too far, dark, hidden under snow or
   ice), vote **Unsure** and go to the next item. If it is clearly a ramp but its tags cannot be
   judged, vote **Agree**, set `cannot_judge` to 1 for the item in the sidecar, and go to the next item.
4. **Judge every applicable tag** against the rubric: add the tags that apply, remove the ones that
   do not. A tag you cannot judge from this image: leave it as it is on the label and add it to
   `cannot_judge_tags` for this item in the sidecar (`;`-separated).
5. **Set severity** 1 / 2 / 3 under rubric R4. To abstain on severity only, add `severity` to
   `cannot_judge_tags`.
6. **Vote Agree**, including when you changed nothing. The gallery editor writes nothing when a
   label is unchanged and never writes a validation row, so an unchanged label with no vote leaves
   no trace and is exported as *not reviewed*.
7. Anything the vocabulary cannot say, or a known change to the ramp since the imagery (rubric R1),
   goes in the sidecar `note`.

## Second rater, deferred: on the review sheet (blind)

Not part of the current pass (D1). The production editor would show the first rater's edits, so the second rater does not open it.

```bash
python scripts/analysis/tag_review_pull.py sheet-template --out benchmark/tag_review/mikey__sheet.csv
```

Each sheet row has the item, the city's applicable tags, the original labeller's tags and severity
(`tags_at_list`, `severity_at_list`, for reference) and the same values pre-filled in `tags` and
`severity` (the anchor the first rater saw on production, decision D2), and `gsv_url`, a Google
Maps link pinned to the label's panorama and centred on the label. Per row, fill in: `verdict`
(`agree` / `disagree` / `unsure`, the same meanings as steps 2, 3 and 6 above), `tags` (the full
`;`-separated set you leave on the label), `severity` (leave the pre-filled value if it is right),
`cannot_judge_tags`, `note`, and optionally `judged_at`. A row with an empty `verdict` is exported
as not reviewed. A reviewed row with a blank `severity` is kept and recorded as missing
(`severity_missing`), and the agreement report counts it. Saving the sheet from Excel (UTF-8 with
a BOM, CRLF line endings) is fine. Nothing from the sheet is written to production until both
exports are committed.

The link's heading and pitch point at the label, computed from its `pano_x` and `pano_y`
(already world-frame) and the camera heading.

### Known asymmetry between the production and sheet routes

D1 is decided (one rater for now), so this matters only if the deferred second pass happens. If
it does, the two raters do not see the same thing, and κ is measured across two views:

| | first rater, production gallery | second rater, review sheet |
|---|---|---|
| imagery | the same panorama | the same panorama, through Google Maps |
| label marker | shown | **not shown**; the view is centred on the label point, so "the ramp nearest the label point" (rubric R0) becomes "the ramp at the centre of the view" |
| crop | the label's production crop | **not shown** |
| tags anchor | the current tags on the label | the list-time tags, pre-filled |
| severity anchor | the current severity | the list-time severity, pre-filled |
| where the judgment is written | production (edits and an Agree / Disagree / Unsure vote), plus the sidecar | the sheet only |
| export `method` | `prod_pull` | `review_sheet` |

The tags and severity anchors match only if nobody else edits a listed label between the list's
fetch and Jon's pass; the production pull flags any item where someone did
(`edited_by_others`). `tag_review_agreement.py` prints each export's `method` and a warning when
they differ; quote that alongside every κ. The ways to remove the asymmetry are both raters on the
sheet (production then gets no edits from this pass), or a sheet view with the label marked
(the production crop, or a rendered view with the point drawn), which is not built.

## Producing the exports

The rater, from production (network; queries `/v3/api/labelEdits` and `/v3/api/validations` on
every deployment in the list for the rater's own user id):

```bash
python scripts/analysis/tag_review_pull.py prod --rater jonfroehlich --since 2026-09-23T00:00:00Z --sidecar benchmark/tag_review/jonfroehlich__sidecar.csv
```

A second rater, if one is added, from the sheet (offline):

```bash
python scripts/analysis/tag_review_pull.py sheet --rater mikey --sheet benchmark/tag_review/mikey__sheet.csv
```

Both write `benchmark/tag_review/<rater>.json` (LF, key-sorted, floats rounded): the rater, the
pass window, the list's sha256, the rubric version, sha256 and full text, and per item the tags
affirmed, added and removed against the list-time tags, severity, the vote, the item- and
per-tag "cannot judge", the note, the panorama id it was judged against and, for the production
route, the edit and validation ids it came from. The production route also records the sha256 of
every API file it pulled. The sheet route records the sheet's sha256 **after LF normalisation and
BOM removal**, which is the hash of the blob git stores (`.gitattributes` pins
`benchmark/tag_review/*` to LF), whatever line endings the working copy had. Commit the export,
the sidecar and the sheet together.

Known limit of the production route: if someone else edits a listed label between the list's fetch
and the review, an unchanged Agree is recorded against the list-time tags, not the tags the rater
saw. The pull detects it: it fetches each deployment's unfiltered `labelEdits`, and any edit by
another user on a listed label after the list's fetch time (from `review_list.meta.json`) and
before the rater's judgment is written to that item's `edited_by_others` and printed as a warning.
In the 30 days before the list's fetch (2026-08-23 to 2026-09-22, all deployments) the #175 cache
holds 49 tag-changing CurbRamp edits, all ExpertValidate and all by the two raters (Jon 17,
Mikey 32), and none by anyone else (`python scripts/analysis/tag_review_list.py recent-edits`,
whose defaults are that window, on the 2026-09-22 cache). So the
edits most likely to trip this flag are the *other rater's* ExpertValidate work, which counts as
"someone else" from each rater's side. Other per-item problems do not
abort the pull: a tag the city no longer offers is dropped from the scored set
(`tags_not_applicable`), and a label whose type the rater changed is kept with a `problems` entry
and left out of every rate. A row whose `user_id` is blank is never attributed to the rater.

## Comparing two passes (only if a second pass happens)

With one rater there is nothing to compare. What the single export does record, per item, is the
tags Jon added and removed against the original labeller's (list-time) tags. No committed script
summarises that yet.

```bash
python scripts/analysis/tag_review_agreement.py benchmark/tag_review/jonfroehlich.json benchmark/tag_review/mikey.json --by tag_state --md analysis_out/tag_review/agreement.md --json analysis_out/tag_review/agreement.json
```

It refuses two exports made against different lists or under different rubric text. Per tag it
reports the items compared, each rater's positives, prevalence, raw agreement,
positive-specific agreement and Cohen's κ; then severity as quadratic- and linear-weighted κ; then
the cross-table of votes. An item enters a tag's comparison only if both raters reviewed it,
neither voted Unsure or Disagree on it, the tag is offered in its city, and neither marked that tag
"cannot judge", and neither pass recorded a problem on it. Run it again with `--by distance_band`.
It always prints a second per-tag table without the items either rater had prior contact with
(placed, validated or edited before the list was built: `prior_contact_<rater>` in the list, 179
of the 500), and both tables are reported (rubric decision D11). It also prints each pass's route
and warns if they differ (above), and, for severity, how many judgeable items each rater left
without one.

The κ values are agreement on an enriched list, not population rates (rubric doc, "Caveats that
travel with any number from this list"). Write the results into a committed doc, with the rubric
version beside every number.
