# Tag review protocol (RampNet 2.0 plan item 3)

**Status: PROPOSED DRAFT, 2026-09-22**, paired with the rubric in
[`docs/tag_rubric_draft.md`](tag_rubric_draft.md), which is also a draft. Issue
[#86](https://github.com/ProjectSidewalk/RampNet/issues/86).

Two raters review the same 500 items in `benchmark/tag_review/review_list.csv`. Jon goes first,
on production. The second rater goes second, blind to Jon's pass (rubric decision D1). Each pass
becomes one committed file, `benchmark/tag_review/<rater>.json`, with the rubric text embedded;
the two files are then compared per tag.

## Before a pass

1. Read the rubric block of `docs/tag_rubric_draft.md` (between the `rubric:begin` and
   `rubric:end` markers) and note its version (`tag-rubric-v0.1-draft` at the time of writing).
   The whole pass is rated under that one version.
2. Note the pass start time in UTC. The production pull only counts edits and votes after it, so
   an older expert-validate of the same label is not mistaken for this pass.
3. Create an empty sidecar file, `benchmark/tag_review/<rater>__sidecar.csv`, with the header
   `item_id,cannot_judge,cannot_judge_tags,note`. Production has no field for a per-tag "cannot
   judge" or a note, so they go here (decision D12).

## First rater: on production

For each row of the list, in `item_id` order:

1. **Open `editor_url`** (`https://<host>/gallery?labelType=CurbRamp&labelId=<id>`) signed in as
   an Owner or Administrator. `labelmap_url` opens the same label on the label map if the gallery
   view is not enough.
2. **Is it a curb ramp?** If not, vote **Disagree** and go to the next item.
3. **If the ramp itself cannot be judged** (occluded, too far, dark, under snow), vote **Unsure**
   and go to the next item.
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

## Second rater: on the review sheet (blind)

The production editor would show the first rater's edits, so the second rater does not open it.

```bash
python scripts/analysis/tag_review_pull.py sheet-template --out benchmark/tag_review/mikey__sheet.csv
```

Each sheet row has the item, the city's applicable tags, the original labeller's tags pre-filled
in `tags` (the same anchor the first rater saw on production, decision D2), and `gsv_url`, a
Google Maps link pinned to the label's panorama and centred on the label. Per row, fill in:
`verdict` (`agree` / `disagree` / `unsure`, the same meanings as steps 2, 3 and 6 above), `tags`
(the full `;`-separated set you leave on the label), `severity`, `cannot_judge_tags`, `note`, and
optionally `judged_at`. A row with an empty `verdict` is exported as not reviewed. Nothing from the
sheet is written to production until both exports are committed.

The Google Maps viewer shows the same panorama as the production gallery but not its crop or its
label marker; the link's heading and pitch point at the label, computed from its `pano_x`,
`pano_y` and the camera heading and pitch.

## Producing the exports

First rater, from production (network; queries `/v3/api/labelEdits` and `/v3/api/validations` on
every deployment in the list for the rater's own user id):

```bash
python scripts/analysis/tag_review_pull.py prod --rater jonfroehlich --since 2026-09-23T00:00:00Z --sidecar benchmark/tag_review/jonfroehlich__sidecar.csv
```

Second rater, from the sheet (offline):

```bash
python scripts/analysis/tag_review_pull.py sheet --rater mikey --sheet benchmark/tag_review/mikey__sheet.csv
```

Both write `benchmark/tag_review/<rater>.json` (LF, key-sorted, floats rounded): the rater, the
pass window, the list's sha256, the rubric version, sha256 and full text, and per item the tags
affirmed, added and removed against the list-time tags, severity, the vote, the item- and
per-tag "cannot judge", the note, the panorama id it was judged against and, for the production
route, the edit and validation ids it came from. The production route also records the sha256 of
every API file it pulled. Commit the export, the sidecar and the sheet together.

Known limit of the production route: if someone else edits a listed label between the list build
and the review, an unchanged Agree is recorded against the list-time tags, not the tags the rater
saw. The pull reports every item's evidence ids so such a case can be found.

## Comparing the two passes

```bash
python scripts/analysis/tag_review_agreement.py benchmark/tag_review/jonfroehlich.json benchmark/tag_review/mikey.json --by tag_state --md analysis_out/tag_review/agreement.md --json analysis_out/tag_review/agreement.json
```

It refuses two exports made against different lists or under different rubric text. Per tag it
reports the items compared, each rater's positives, prevalence, raw agreement,
positive-specific agreement and Cohen's κ; then severity as quadratic- and linear-weighted κ; then
the cross-table of votes. An item enters a tag's comparison only if both raters reviewed it,
neither voted Unsure or Disagree on it, the tag is offered in its city, and neither marked that tag
"cannot judge". Run it again with `--by distance_band`, and report the items either rater placed
(`placed_by_rater` in the list) both with and without them (rubric decision D11).

The κ values are agreement on an enriched list, not population rates (rubric doc, "Caveats that
travel with any number from this list"). Write the results into a committed doc, with the rubric
version beside every number.
