# Curb-ramp tag rubric and review list (RampNet 2.0 plan item 3)

**Status: PROPOSED DRAFT, 2026-09-22; Jon's first decisions applied 2026-09-23.** Rows of the
decisions table marked **Decided** are Jon's; the rest are still drafted defaults. The rubric
text stays a draft (`-draft` in its version) until Jon approves it as a whole. The choices that
most change what a rater does are pulled out first, under [Decisions for Jon](#decisions-for-jon). Umbrella issue:
[#86](https://github.com/ProjectSidewalk/RampNet/issues/86); plan:
[`docs/rampnet2_plan.md`](rampnet2_plan.md) §2.2, §2.3, §2.5, §3 and §4 item 3.

What this document is for. The census on #86 found that two trained raters looking at the same
395 ramps agree on *which* tag when both tag, but one tags 61 % of ramps and the other 14 %
(per-tag κ 0.00–0.21): the disagreement is the **reporting threshold**, not perception (plan
§2.2). A per-tag agreement number is only meaningful if both raters were given the same
threshold, so each tag below has an explicit "tag it when" / "do not tag when" line and a
"cannot judge" escape. **For now there is one rater (Jon, decided 2026-09-23, to move fast), so
this pass produces no κ.** The explicit thresholds still matter: they keep one rater consistent
across 500 items, and they are what a later second pass would be held to. The rubric proper sits between the `rubric:begin` / `rubric:end`
markers; that block, and only that block, is embedded verbatim with its sha256 in every
per-rater export (`benchmark/tag_review/<rater>.json`), following the precedent in
[`benchmark/RUBRICS.md`](../benchmark/RUBRICS.md) §3. The rater's steps are
[`docs/tag_review_protocol.md`](tag_review_protocol.md).

Constraints this draft works inside (Jon, 2026-09-22, on #86 and in the plan): no new tags
(ideas are parked as [SidewalkWebpage#5448](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5448)
and [#5449](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5449)); the retired
`tactile warning` tag drops out and `parallel lines` stays; the 94 disputed "missing tactile
warning" ramps from the validation study are not reviewed now, so **no label from the
validation-study deployment is in the list** (excluded by deployment, not by physical ramp;
see D21); the review happens on production; **Jon is the only rater for now** (D1). A second
blind pass is deferred, not ruled out; the sheet route and the agreement script stay built for it.

**D20 decided (Jon, 2026-09-23):** the committed list keeps its per-label rows (host URL, label
id, pano id, tags) from the **9 private deployments, 114 of the 500 items**: burnaby, columbia,
kaohsiung, keelung, new-taipei, taipei, walla-walla, west-chester, zurich.

## Decisions for Jon

Each has a drafted default, which is what the rubric text and the committed list currently
use. **Decided** marks Jon's answers of 2026-09-23 (on PR #176); an unmarked row is still a
drafted default. Changing one means editing the rubric block (and bumping its version) or re-running the
list builder with different arguments.

| # | question | drafted default | why it matters |
|---|---|---|---|
| D1 | How does the second rater stay blind when the first rater's edits are live on production? | **Decided: one rater (Jon) for now, on production, to move fast. No κ from this pass.** A second pass is deferred; if it happens, the draft below is how it would run. Draft: Jon reviews on production. The second rater uses the **review sheet** route (`tag_review_pull.py sheet-template`): a CSV plus a Google Maps link pinned to the label's pano id and centred on the label, no production URL. Nothing the second rater decides is written to production until both exports are committed. **κ is then measured across two different viewing routes**: the gallery shows the label marker and the crop, the sheet's Google Maps view shows neither (protocol, "Known asymmetry between the production and sheet routes"). `tag_review_agreement.py` prints both routes and warns when they differ. | The production gallery shows the current tags. After Jon's pass those are Jon's tags, so a second rater on production would be reviewing Jon, and κ would be inflated by anchoring. The alternative that removes the asymmetry is both raters on the sheet route (D2's cost), or a marked view for the sheet (not built). |
| D2 | Should both raters see the original labeller's tags and severity? | **Settled by D1 for this pass:** Jon, on production, sees the original tags and severity, so the pass is a *review*, not labelling from a blank slate. For a later second pass: yes, both see them (production shows them to Jon; the sheet pre-fills `tags` and `severity` for the second rater, with `tags_at_list` / `severity_at_list` beside them), so the two passes are anchored identically. It measures agreement on a *review*, not on labelling from a blank slate. A severity the sheet rater blanks is recorded as missing and counted, not dropped. | The alternative (both blank) needs Jon on the sheet route too, and then production gets no edits from this pass. |
| D3 | Width threshold for `narrow` and landing depth for `not enough landing space`: 48 in (PROWAG R304.5.1.1, R304.2.5) or 36 in (the labeling guide and the 2010 ADA Standards)? | **Decided: 48 in (1.2 m)**, the current federal guideline, which is also what a width measurement (plan items 7–8) would be scored against. Written inline in the rubric. | The crowd was taught 3 ft. At 48 in the reviewed set will call more ramps narrow than the crowd did, which reads as low crowd recall rather than a threshold change. Either is defensible; mixing them is not. |
| D4 | What confidence does a tag need? | **Decided: more likely than not** that the condition, at the stated threshold, is present on this image. Unresolvable ⇒ "cannot judge", never a guess either way. | This is the reporting threshold itself. "Only when clearly present" would lower both raters' tag rates and raise κ, at the cost of recall. |
| D5 | How is `steep` judged when slope cannot be measured? | **Revised 2026-09-23, awaiting Jon's check.** Jon: the first draft's rule (run shorter than about 8× the curb height) is too hard to think about and act on in Project Sidewalk. The revision is a visual judgment, on the same pattern as D15: tag a ramp (or a flare a pedestrian must cross, or the gutter at its bottom) that **looks noticeably steeper than a typical well-built ramp**, and let severity carry how steep. Written inline in the rubric. | A threshold exactly at 1:12 is not resolvable from street-level imagery. A looser visual rule will raise the tag rate over the 8× rule; severity is what separates mild from severe. |
| D6 | Heavy leaves or debris on a ramp: `surface problem` or `debris / pooled water`? | **Decided: `debris / pooled water` only**, kept separate from `surface problem` for now (Jon: the two might be collapsed at some point). `surface problem` is reserved for the ramp's own surface. Written inline in the rubric. | The labeling guide still says to add "surface problem" for heavy debris; that text predates the debris tag. |
| D7 | Grooves instead of truncated domes. | **Decided.** Where the city offers `parallel lines` (Burnaby): tag `parallel lines`, not `missing tactile warning`. Everywhere else: `missing tactile warning`, because PROWAG R305.1 requires truncated domes. | Otherwise Burnaby's `parallel lines` and `missing tactile warning` double-count one condition. |
| D8 | Which yardstick outside the US? | **Decided.** PROWAG thresholds everywhere, as a fixed reference, not as a statement about local code. 114 of the 500 items are in non-US deployments. | Without one yardstick, "narrow" means different things by city. |
| D9 | Severity 1 / 2 / 3 definitions (plan S4). | **Decided (Jon: "seemed fine").** The draft in [Severity](#severity-draft-definitions-s4) below: usability for a wheelchair user, judged on the whole ramp after the tags, not a count of tags. | The recorded scale never had a definition; the census shows it behaves like a weighted tag count (plan §2.4). |
| D10 | List design: size 500, shares 20 / 15 / 35 / 30 % over affirmed-empty / trusted-tagged / tagged / untagged, crop-era GSV labels only, 35 cities. | As committed. | See [The review list](#the-review-list); every choice is a CLI argument. |
| D11 | Items either rater has already touched. Placed: Jon 62, Mikey 24. Any prior contact (placed, validated or edited before the list was built): Jon 117, Mikey 67, **either rater 179 of the 500**. Jon validated 34 of the 100 `affirmed_empty` items himself. | Kept and flagged per rater in `prior_contact_jonfroehlich` / `prior_contact_mikey` (`placed;validated;edited`). With one rater (D1), the 117 items Jon touched are the ones that matter: any tag rate from this pass is reported with and without them. `tag_review_agreement.py` reports κ both ways if a second pass happens. | Re-reviewing your own label, or re-affirming your own affirmation, is not the same act as reviewing a stranger's. Dropping 179 items would cost too many positives, so the draft reports both. |
| D12 | Per-tag "cannot judge" and notes have no production field. | A per-rater sidecar CSV (`item_id,cannot_judge,cannot_judge_tags,note`), merged by the pull script. Revisit if the list-driven queue ([SidewalkWebpage#5444](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5444)) grows a field for it. | Without it, "cannot judge" on production collapses into "no", which is exactly the confusion of absence and negative the plan rules out. |
| D13 | Snow and ice, and the reach of `debris / pooled water`. | **Decided with D6.** One rule: snow or ice that **hides** the ramp ⇒ the item is *Unsure*; snow, ice, slush, sand, gravel, leaves or trash on a ramp that is **still visible enough to judge** ⇒ `debris / pooled water` when a wheelchair user would have to go through it. The draft also extends the tag from "pooled at the bottom of the curb ramp" (the PS definition) to the ramp surface. | The first draft said both "under snow ⇒ Unsure" and "tag snow and ice as debris". The extension widens the PS definition; the narrower reading (bottom of the ramp only) is the alternative. |
| D14 | Which point in time is judged (R1)? | **The imagery shown**, the panorama the label was placed on, even where the ramp is known to have changed (say so in the note). | The alternative, current conditions, needs newer imagery or a site visit and breaks the link between the item and the crop a model is scored on. |
| D15 | Step height for `not level with street`. | **Decided (Jon, changed from the draft's "about 1 in or more"):** tag **any** visible step or lip at the bottom of the ramp, and use severity for how bad it is (R4). This is how Jon already labels. PROWAG's ¼ in is not resolvable either way. | This lowers the tag's threshold to "visible at all", so the tag rate rises and severity carries the magnitude. |
| D16 | A dome panel narrower than the ramp, or worn. | **Decided: not** `missing tactile warning`. A damaged, worn or undersized warning still counts as having one; the damage can raise severity, and goes in the note. | PROWAG R305 sets a width, so a narrow panel is non-compliant; the draft keeps the tag for "absent". The alternative tags it. |
| D17 | Several ramps near the label point. | Judge the **one nearest the label point**. On the sheet route there is no marker, so the rater takes the view centre (the link is centred on the label). | Without a rule the two raters can judge different ramps. The sheet can only approximate it (D1). |
| D18 | What does *Unsure* mean? | The **item-level "cannot judge"**: the item leaves every rate. It is not a vote on whether the label is a ramp. | Production's Unsure has no defined meaning; this pass gives it one. |
| D19 | Which labels are eligible by validation status? | Labels the crowd voted incorrect (`correct == false`) are **out**; unvalidated labels **stay in**. | The first saves rater time on likely *Disagree* items; the second keeps the list from over-representing validated labels. |
| D20 | **Publishing per-label rows from private deployments.** 114 of the 500 items are from 9 private deployments (burnaby, columbia, kaohsiung, keelung, new-taipei, taipei, walla-walla, west-chester, zurich); the list carries each one's host URL, label id, pano id and tags. | **Decided (Jon, 2026-09-23): kept.** The column `deployment_visibility` marks them. | This repo is public. The private deployments' partners may not expect their host names and label ids in it. |
| D21 | Validation-study overlap. | The validation-study deployment is excluded **by deployment**. A physical ramp it also covers can still be listed through another deployment's label. Nothing is dropped for that; two flag columns show it: `vstudy_same_pano` (0 items) and `vstudy_within_10m` (1 item, `tr0391` seattle-wa:271169, 2 validation-study labels within 10 m, neither tagged `missing tactile warning`). Across all labels, 62 validation-study panos also appear in seattle-wa (58) and chicago-il (4); 22 of them are in the eligible pool. | The deferred adjudication covers 94 validation-study ramps; an item on the same ramp would be judged here first. |

<!-- rubric:begin -->
## Rubric

**Rubric version:** `tag-rubric-v1.0-draft`

**Status of this text: PROPOSED DRAFT, not approved.** A pass rated under this version measures
agreement under a draft; say so next to any number it produces.

### R0. What is judged

- **The item** is one Project Sidewalk `CurbRamp` label, identified by `city:label_id`, judged on
  the panorama it was placed on (`pano_id` in the list). Judge the physical curb ramp the label
  marks. If several ramps are close, judge the one nearest the label point (on the review
  sheet, which shows no marker, the one at the centre of the linked view, which is centred on
  the label point).
- **Not a curb ramp** (a driveway apron, a blended level crossing with no ramp, nothing there):
  vote *Disagree* and judge no tags. The labeling guide's rules on what counts as a curb ramp
  (driveways are not; see its `#driveways` section) apply unchanged.
- **The whole item cannot be judged** (ramp occluded, too far, blurred, hidden under snow or
  ice, or in darkness): vote *Unsure*. That is the item-level "cannot judge"; it is not a
  Disagree, and the item leaves every rate. Snow, ice or other material on a ramp that is
  still visible enough to judge is not a reason for *Unsure*; see `debris / pooled water`.
- **Every applicable tag gets a decision**: present, absent, or cannot judge. The applicable tags
  are the ones the label's city offers (column `applicable_tags`), minus the retired
  `tactile warning`. A reviewed item on which a tag is left off means *absent* for that tag; that
  is the only way a tag negative is created. A label nobody reviewed is never a negative.
- **Per-tag "cannot judge"**: the part of the ramp the tag is about is not visible or not
  resolvable in this image (the bottom edge behind a parked car for `not level with street`; the
  top of the ramp out of frame for `not enough landing space`). Record the tag in
  `cannot_judge_tags`. Use `severity` in the same field to abstain on severity alone.
- **Confidence**: tag when it is more likely than not that the condition, at the threshold
  stated below, is present in this image. If you cannot get to "more likely than not" either way
  because the image does not show it, that is "cannot judge", not "absent".
- **No new tags.** Conditions the vocabulary does not name go in the note.

### R1. Which point in time

Judge **the imagery shown**, which is the panorama the label was placed on. Its capture date is
typically 1.2–2.4 years before the label (median by city, census on #86; cuenca 8.2 years), and
in this list the median gap is 2.0 years with 73 of 500 items at five years or more. Do not use a newer or older
panorama, a site visit, or knowledge of reconstruction to change a tag; if you know the ramp has
changed, judge the image and say so in the note. Transient conditions (water, snow, leaves,
debris, a parked car) are judged as they appear in this image.

### R2. Yardstick

Thresholds are the US Access Board's Public Right-of-Way Accessibility Guidelines (PROWAG, final
rule published 2023-08-08; `access-board.gov/prowag/technical.html`, read 2026-09-22), applied as
a fixed reference in every city, including outside the US. Where the Project Sidewalk labeling
guide cites the 2010 ADA Standards instead, the difference is stated. Street-level imagery cannot
resolve inches; the thresholds say what the tag *means*, and the "tag it when" lines say what is
visible enough to call.

### R3. Tags

Sources for each "PS definition": the tag description returned by each deployment's
`/v3/api/labelTags` (identical in every city that lists the tag; cached by PR #175's audit fetch,
2026-09-22), and the labeling guide `app/views/labelingGuide/labelingGuideCurbRamps.scala.html` in
SidewalkWebpage (read at commit `8a542b8`, 2026-08-10), cited by its section anchor.

#### `missing tactile warning`

- **PS definition:** "The ramp is missing a tactile warning strip, which is a textured surface that
  alerts people with visual impairments to existence of the ramp and crossing." Guide
  (`#tactile-warning`): apply the tag when the ramp has no tactile warning; if that is the only
  problem, severity Low.
- **Standard:** PROWAG R304.1 (curb ramps and blended transitions shall have detectable warning
  surfaces); R305.1 (truncated domes in a square or radial grid); R305.1.3 (visual contrast with
  the adjacent surface); R305.1.4 (24 in / 610 mm minimum in the direction of travel). The 2010 ADA
  Standards do not require detectable warnings on curb ramps at non-transit facilities (Access
  Board guide to the ADA Standards, chapter 4).
- **Tag it when:** there is no truncated-dome surface at the street edge of the ramp. Grooves,
  paint, or a textured band that is not domes count as missing, except in cities that offer
  `parallel lines` (next entry).
- **Do not tag when:** a dome panel is present, even if faded, damaged, partly worn or narrower
  than the ramp. The ramp still has a tactile warning; the defect can raise severity (R4) and goes
  in the note.
- **Cannot judge when:** the bottom of the ramp is not resolvable (distance, occlusion, snow,
  glare).

#### `parallel lines` (Burnaby only)

- **PS definition:** "This tag is for a specific type of curb ramp that has subtle parallel grooves
  in place of a proper tactile warning strip."
- **Standard:** not a detectable warning surface under PROWAG R305.1 (domes are required).
- **Tag it when:** the ramp has parallel grooves where a tactile warning would be. Do **not** also
  tag `missing tactile warning` for the same grooves.
- **Do not tag when:** there are domes, or there is no texture at all (that is
  `missing tactile warning`).
- **Cannot judge when:** the ramp surface texture is not resolvable, which is common: the grooves
  are subtle by definition.

#### `points into traffic`

- **PS definition:** "The ramp is angled towards cross traffic." Guide (`#corners-with-only-one-curb-ramp`,
  citing ADA 406.6): a single corner ramp serving two crossings leads pedestrians into traffic, and
  the tag applies even when a parking lane protects them (severity then Low).
- **Standard:** PROWAG R304.2.1 (running slope perpendicular to the curb); R304.5.3 (perpendicular
  ramp runs contained wholly within the width of the crosswalk they serve); R304.2.4 (a 48 × 48 in
  clear area beyond the bottom grade break, within the crosswalk and outside the travel lanes).
- **Tag it when:** following the ramp's direction of travel off its bottom edge leads a user out
  of the crossing it serves (marked or implied) and toward vehicle lanes: the typical case is one
  diagonal ramp at the apex of a corner serving both crossings.
- **Do not tag when:** the ramp's run points along its crossing, even if the crossing itself is
  unmarked.
- **Cannot judge when:** the crossing direction or the road layout is not visible.

#### `not aligned with crosswalk` (Taipei, New Taipei, Kaohsiung, Keelung, Taichung)

- **PS definition:** "Both a curb ramp and a marked crosswalk are present, but the curb ramp does
  line up with crosswalk." (The live description reads "does line up"; the intended meaning is
  "does not".)
- **Standard:** PROWAG R304.5.3 (ramp runs contained wholly within the width of the crosswalk).
- **Tag it when:** a marked crosswalk exists and the ramp sits laterally outside it, wholly or in
  part. This is position; `points into traffic` is direction. Both can apply.
- **Do not tag when:** there is no marked crosswalk.
- **Cannot judge when:** the crosswalk markings are not visible (worn, occluded, out of frame).

#### `not level with street`

- **PS definition:** "There is a height difference between the bottom of the curb ramp and the
  street."
- **Standard:** PROWAG R304.5.4 (changes in level are not permitted on curb ramp surfaces); R302.6.2
  (up to ¼ in may be vertical); R304.5.2 (change of grade at the gutter 13.3 % maximum, or a 24 in
  transitional space).
- **Tag it when:** any lip or step is visible where the ramp meets the gutter or street: a shadow
  line, an exposed edge, asphalt built up against the ramp, or a gutter that drops away below it.
  There is no minimum height beyond "visible"; severity (R4) says how bad the step is.
- **Do not tag when:** the transition is flush and the only issue is a change of grade (a steep
  gutter pan) with no step; that is `steep` if it is severe.
- **Cannot judge when:** the bottom edge is hidden (parked car, shadow, distance).

#### `narrow`

- **PS definition:** "The ramp is not wide enough for a wheelchair to safely use it." Guide
  (`#narrow-ramps`): "the ADA requires a width of at least 3 ft"; severity Medium or High.
- **Standard:** PROWAG R304.5.1.1: clear width of the ramp run, excluding flares, 48 in (1220 mm)
  minimum. 2010 ADA Standards 405.5: 36 in. **The threshold is 48 in (1.2 m)**, not the guide's
  3 ft.
- **Tag it when:** the ramp run, not counting its flares, looks narrower than 4 ft (1.2 m). Useful
  references in the image: a standard dome panel is 2 ft deep; a wheelchair is about 2.2 ft wide;
  a US sidewalk flag is often 4–5 ft.
- **Do not tag when:** only the flared sides make the ramp look narrow at the top, or the ramp is
  wide but partly blocked (note the obstruction).
- **Cannot judge when:** the ramp's edges are not both visible, or it is too far to size.

#### `not enough landing space`

- **PS definition:** "There is not enough space at the top of the curb ramp to safely maneuver a
  wheelchair." Guide (`#insufficient-landing-space`): "the ADA requires at least 3ft"; consult
  steep flares too.
- **Standard:** PROWAG R304.2.5: where a change of direction is needed to reach the ramp, a landing
  at the top 48 × 48 in minimum; R304.3.4: for parallel ramps the landing is at the bottom, also
  48 × 48 in. 2010 ADA Standards 406.4: 36 in. **The threshold is 48 in (1.2 m)**, not the
  guide's 3 ft.
- **Tag it when:** the level area where a user turns onto the ramp is shallower than 4 ft before
  it meets a building, wall, grass, a drop, or the far edge of the sidewalk; including a ramp whose
  run takes up the entire sidewalk depth so that there is no level top at all.
- **Do not tag when:** the ramp is entered straight on and the sidewalk behind it is the landing,
  and that sidewalk is at least 4 ft deep.
- **Cannot judge when:** the top of the ramp or the back of the sidewalk is out of frame.

#### `steep`

- **PS definition:** "The ramp is overly steep, making it dangerous or uncomfortable to use." Guide:
  steep flares (`#steep-flares`, Low or Medium alone, worse with no landing space) and steep counter
  slopes (`#steep-counter-slopes`, ADA 406.2, "only obvious cases").
- **Standard:** PROWAG R304.2.1 (running slope 1:12, 8.3 % maximum); R304.2.6 (flared sides 1:10,
  10 % maximum, where a pedestrian circulation path crosses them); R304.4.1 (blended transitions
  1:20, 5 %). 2010 ADA Standards 406.2: counter slope 1:20.
- **Tag it when:** the ramp looks noticeably steeper than a typical well-built ramp: short and
  abrupt for the height of curb it climbs, or with a flare pedestrians must cross that is
  similarly steep, or with the gutter rising steeply against its bottom. Severity (R4) says how
  steep: steep but easy to climb is 1, climbable with effort or risk is 2, too steep to use
  without help is 3.
- **Do not tag when:** the ramp looks like an ordinary ramp and you would only be guessing that it
  is a little over 1:12; that is not resolvable from imagery.
- **Cannot judge when:** the curb height or the ramp length cannot be seen (a straight-down view,
  far distance).

#### `surface problem`

- **PS definition:** "There is a issue with the surface of the curb ramp; can be used in place of a
  separate Surface Problem label." Guide (`#surface-problems-on-curb-ramps`): cracks, grass at
  edges, broken asphalt; severity as for sidewalk surface problems.
- **Standard:** PROWAG R304.5.4 (ramp surfaces comply with R302.6, and no changes in level);
  R302.6.3 (openings pass no sphere larger than ½ in).
- **Tag it when:** the ramp's own surface is cracked with displaced or missing pieces, heaved,
  potholed, patched with a different material that leaves an edge, has vegetation growing across
  it, or carries a grate or utility cover with visible gaps.
- **Do not tag when:** hairline cracks, stains, colour differences; loose leaves, water or trash
  (that is `debris / pooled water`, even when heavy; the labeling guide's older advice to add
  `surface problem` for heavy debris does not apply); a lip at the bottom edge (that is
  `not level with street`).
- **Cannot judge when:** the surface is not resolvable (distance, resolution, snow).

#### `debris / pooled water`

- **PS definition:** "Water or debris has pooled at the bottom of the curb ramp, making it
  difficult to traverse." Guide: pooled water (`#pooled-water`, citing ADA 405.10) and debris
  (`#debris-on/around-ramp`, citing FHWA's guide for maintaining pedestrian facilities, 3.2.2; not verified here).
- **Standard:** no PROWAG technical provision covers drainage or debris (checked against the R3
  technical requirements page, 2026-09-22). 2010 ADA Standards 405.10: landings subject to wet
  conditions shall be designed to prevent the accumulation of water.
- **Tag it when:** water, ice, snow, slush, sand, gravel, leaves or trash covers enough of the
  ramp or the area at its bottom that a wheelchair user would have to go through it, **and the
  ramp is still visible enough to judge**. This reads the PS definition ("pooled at the bottom
  of the curb ramp") to include the ramp surface and frozen or granular material.
- **Do not tag when:** a few scattered leaves; a damp surface with no standing water; snow or
  ice that hides the ramp itself (then the whole item is *Unsure*, R0).
- **Cannot judge when:** the ramp surface is not resolvable.

#### Tags outside this pass

- `tactile warning` (the retired positive tag; still listed by Amsterdam only): not judged.
- `not visible` (Chandigarh only, 5 labels): a statement about the image, not the ramp. The review
  uses "cannot judge" instead. Neither city is in the list, because each lacks core tags.

### R4. Severity (draft definitions, S4)

Severity is judged on **the whole ramp, after the tags**, as usability for a wheelchair user
arriving at this corner. It is not a count of tags: one severe defect can be a 3 and three minor
ones can be a 1.

| severity | draft definition | anchors from the labeling guide and tutorial |
|---|---|---|
| **1 (Low)** | Usable as is by a wheelchair user without assistance and without leaving the crossing. Any defect is an inconvenience (a small lip a wheelchair rolls over, a ramp a little steep), or affects other users (a missing or damaged tactile warning). | Guide `#tactile-warning`: missing tactile warning as the only problem ⇒ Low. Tutorial: a ramp with good slope and warning strip ⇒ 1. |
| **2 (Medium)** | Usable with difficulty or with risk: a defect a wheelchair user has to work around, such as a lip that needs care, a ramp that points into traffic, a narrow run, a landing too small to turn comfortably, steep but climbable. | Tutorial: points into traffic as the only issue ⇒ 2; no landing space as the only issue ⇒ 2. Guide: narrow ⇒ Medium or High; pooled water alone is not High. |
| **3 (High)** | Likely not usable without assistance, or not safely: a step a manual wheelchair cannot mount, a run too steep or too narrow to use, the ramp blocked, or medium defects that compound (steep with no landing). | Guide `#steep-flares`: steep flares plus poor landing space ⇒ High. Guide `#debris-on/around-ramp`: heavy debris ⇒ High as an extreme case. |

If severity cannot be judged while the tags can, put `severity` in `cannot_judge_tags`.

### R5. Versioning

- The version string is the line **Rubric version** above, `tag-rubric-vMAJOR.MINOR[-draft]`.
- **Any** edit to the text between the markers bumps the version: MINOR for wording that cannot
  flip a judgment, MAJOR for a change that can (a threshold, a tag boundary, a severity anchor).
  `-draft` is dropped only when Jon approves the text.
- Every export embeds the version, the sha256 of this block and the block itself.
  `tag_review_agreement.py` refuses to compare exports whose rubric text differs, unless told to
  with `--allow-rubric-mismatch`, which it then prints.
- A pass is rated under exactly one version. If the rubric changes mid-pass, the items already
  rated keep the old version, and the pass is split into two exports, one per version.
<!-- rubric:end -->

## The review list

`benchmark/tag_review/review_list.csv`, 500 items, built by
`scripts/analysis/tag_review_list.py build` with its defaults (seed 86), from the Project Sidewalk
API cache fetched on 2026-09-22 by PR #175's `scripts/analysis/ps_supervision_audit.py fetch`
(the cache is gitignored; `review_list.meta.json` records the sha256 of every input file it read).

**List sha256:** `00e850e478489fbca6a9dacfb46c92fa14cde75a803b8afe3460c8eecdbc9ff8`
(rebuilt 2026-09-22 after the PR #176 review; it replaces `9dbe2e7c…a3e6`, whose distance bands
double-corrected camera pitch)

**Depends on PR [#175](https://github.com/ProjectSidewalk/RampNet/pull/175)**, which is not
merged: the fetch script (`ps_supervision_audit.py fetch`) lives there, and the cache it writes
is gitignored and unpublished. This PR imports no code from #175.

**The committed CSV is the artifact of record.** Production is live: a fresh fetch returns a
different pool (new labels, edits, votes), so the script reproduces this list byte for byte only
from a cache whose hashes match the meta file. Anyone can check the list against the meta file and
against this hash; nobody can regenerate it from a later fetch, and that is stated rather than
hidden. **What would unblock regeneration from a clean clone:** merge #175, then publish the
2026-09-22 cache (the files named in `review_list.meta.json` `inputs`) to Hugging Face under a
stated identifier. A reduced candidate pool (the 183,217 eligible rows × the columns
`build_candidates` uses, no free-text descriptions) was measured and **not committed**: it is
about 10 MB gzipped, over the ~5 MB a committed file should be, and 43,717 of its rows come from
the private deployments D20 is about. Publishing either is Jon's call under D20.

### Eligibility

| step | labels |
|---|---:|
| CurbRamp labels in the 55 vocabulary-eligible deployments, SidewalkAI included | 569,800 |
| minus the SidewalkAI account (`51b0b927-3c8a-45b2-93de-bd878d1e5cf4`) | 490,449 |
| placed on or after 2023-10-12 (production stores a crop for these) | 195,663 |
| minus labels the crowd validated as incorrect (`correct == false`) | 186,004 |
| GSV panoramas only | 183,816 |
| in deployments with at least 100 eligible labels | 183,217 |

Deployments left out, and why (full list in `review_list.meta.json`): `validation-study` (the
deferred adjudication, above); `amsterdam` (its vocabulary lacks `missing tactile warning` and
`points into traffic`); `chandigarh-india` (lacks six core tags); and 20 deployments with fewer
than 100 eligible labels. Eight of those have none: `laurens-ia` and `richmond-va` are Mapillary
deployments, `zurich-infra3d` and `winterthur-infra3d` are infra3d, `bayonne-fr` has only
Panoramax imagery, and `cuenca`, `dc` and `la-piedad-old` have no human label placed since
2023-10-12. The other twelve (auckland, blackhawk-hills-il, clifton-nj, houston-tx, la, la-piedad,
newport-ky, spgg, taichung, tainan, vancouver-wa, virden-il) have 5–97 each. Every deployment kept
offers all eight core tags.

Why these filters:

- **Crop era.** Labels placed since 2023-10-12 have a production crop, so the reviewed items can
  double as a model test set (plan items 4–5) without the unpublished pano-store re-cut (item 2b).
  It also puts the whole list in the tag era: tags entered the schema on 2018-04-29. The crop's
  existence is inferred from the placement date, not checked per label, so the list has no crop
  column.
- **`correct == false` out.** The crowd already voted these not to be ramps; reviewing tags on
  them spends rater time on items that mostly end as *Disagree*. Unvalidated labels stay in (D19).
- **validation-study out, by deployment** (D21). The two `vstudy_*` columns flag the one listed
  item that sits within 10 m of a validation-study label; nothing is dropped for it.
- **GSV only.** The distance band assumes a 2.5 m camera; infra3d (Zurich) and Mapillary rigs
  differ, and 2,188 crop-era labels were not worth a second geometry.

### Strata

- **Tag state** (precedence top down):
  `affirmed_empty` = no tags today and a tag-review pass affirmed it: an *Agree* vote or an edit
  from `ExpertValidate` or the ASSETS'24 `ExternalTagValidationASSETS2024` pass (an Unsure or
  Disagree vote alone does not count; the first draft counted those, which put three unaffirmed
  labels in this stratum); `tagged_trusted` = tagged, placed by an Owner
  account (jonfroehlich, mikey); `tagged` = tagged by anyone else; `untagged` = no tags and never
  tag-reviewed. `untagged` is not a negative stratum: it is where the review measures how often an
  untagged label actually carries a condition, which is the number the positive-unlabeled
  experiment (plan item 5) needs. Tier 2 (`correct == true`) is not a stratum, because an Agree in
  ordinary Validate never saw a tag control. The trusted tier is Owners only, because the wider
  role list is not committed (it names MTurk worker accounts); `--trusted-users` widens it.
- **Distance band** from the label's depression below the horizon,
  `(pano_y / pano_height − 0.5) × 180` degrees, converted to a flat-ground distance with a fixed
  2.5 m camera height: near < 8 m, mid 8–15 m, far ≥ 15 m. This is the convention, and the
  constant, of `crop_window_eval.py` and `size_analysis.py`. There is no `camera_pitch` term
  because Project Sidewalk writes `pano_y` from the label's world-frame pitch
  (`util.pano.povToPanoCoord` in `panoUtilities.js`, called from `Label.js` with a POV that
  `GsvViewer` / `PannellumViewer.getPov` report in world pitch). The first draft subtracted
  `camera_pitch` a second time, which put 67 of its 500 items (and 9.7 % of the pool) in the wrong
  band and tilted every `gsv_url` by the camera pitch (median 0.66°, max 9.2°). The check
  (`tag_review_list.py pitch-check`): on
  crop-era GSV labels placed at the canvas centre, `(0.5 − pano_y / pano_height) × 180 − pov_pitch`
  has slope −0.04 (seattle-wa, n 1,310), −0.01 (chicago-il, 6,405) and −0.01 (taipei, 1,109)
  against `camera_pitch`; a double correction needs −1. The edges sit near the crop-era pool's
  tertiles (8.4 m and 12.2 m) with a wider far band, because far is where "cannot judge" is
  expected.
- **City**: each tag state's quota is split equally over the deployments that have candidates in
  it, capped by what each has; no deployment has more than 20 of the 500 items.

Within `tagged` and `tagged_trusted`, a label is drawn with weight (1 / *f*)^1.5, where *f* is the
frequency of its rarest tag among that tag state's candidates; this lifts every core tag to at
least 37 list-time positives on the committed draw. Within a (tag state, city) cell the draw
rotates through the bands. No two items share a (city, pano) and no two in one city are within
10 m, so one physical ramp (plan §2.5) is not two items. Item order is a seeded shuffle.

### Composition

Tag state by distance band (500 items):

| tag state | pool | near | mid | far | total |
|---|---:|---:|---:|---:|---:|
| affirmed_empty | 1,087 | 31 | 38 | 31 | 100 |
| tagged_trusted | 609 | 29 | 28 | 18 | 75 |
| tagged | 57,740 | 58 | 59 | 58 | 175 |
| untagged | 123,781 | 49 | 52 | 49 | 150 |

Of the 100 `affirmed_empty` items, 80 were affirmed by ExpertValidate and 20 by the ASSETS'24
pass. 9 more items carry tags that a tag-review pass affirmed. 35 deployments, 9–20 items each;
114 items are outside the US (burnaby, cdmx, kaohsiung, keelung, new-taipei, rancagua-chile,
santiago-chile, sao-paulo-brazil, taipei, zurich). 114 items are from private deployments (D20).

List-time positives per tag (the original labeller's tags, before review):

| tag | items carrying it |
|---|---:|
| missing tactile warning | 94 |
| points into traffic | 52 |
| narrow | 48 |
| steep | 47 |
| not enough landing space | 45 |
| not level with street | 41 |
| surface problem | 39 |
| debris / pooled water | 37 |
| parallel lines | 5 |
| not aligned with crosswalk | 4 |

Severity at list time: 1 = 355, 2 = 111, 3 = 22, unrated 12.

Prior contact (D11): Jon placed 62, validated 55 and edited 5 listed labels (117 items in all);
Mikey placed 24, validated 43, edited 2 (67); either rater, 179 items. Jon validated 34 of the
100 `affirmed_empty` items himself.

Every number in this section is in `review_list.meta.json` (`composition`, `pool_by_state`,
`pool_stats`), written by the same `build` run.

### Size: why 500

Per-tag κ precision is governed by the number of positives, not by the list size.
`tag_review_list.py power` simulates the 95 % CI half-width of Cohen's κ (true κ 0.6, both raters
at the same prevalence, 4,000 draws):

| positives | n = 300 | n = 500 | n = 800 |
|---:|---:|---:|---:|
| 10 | 0.28 | 0.28 | 0.28 |
| 20 | 0.19 | 0.19 | 0.19 |
| 30 | 0.16 | 0.16 | 0.15 |
| 50 | 0.12 | 0.12 | 0.12 |
| 80 | 0.10 | 0.10 | 0.10 |

At about 30 positives a tag's κ is known to ±0.16, which separates the validation study's
0.00–0.21 from a usable 0.6. The rare core tags are the constraint. `tag_review_list.py size`
draws the list at three sizes over ten seeds (86 and 1–9, all other parameters as committed) and
reports the fewest list-time positives any core tag gets:

| n | seed 86 (committed) | range over 10 seeds | seeds with every core tag ≥ 30 |
|---:|---:|---:|---:|
| 400 | 22 | 15–31 | 1 of 10 |
| 450 | 35 | 29–37 | 9 of 10 |
| 500 | 37 | 30–40 | 10 of 10 |

So 500 is not the smallest size that can clear 30: 450 clears it on this seed and on 9 of 10
seeds. 500 is the size that clears it on every seed tried, with a margin of 7 on the committed
draw; 400 does not. The binding tag is almost always `debris / pooled water`. At roughly 30 s an
item on production, 500 items is about four hours per rater (450 would save about 25 minutes).
`--n` changes it (D10).

### Caveats that travel with any number from this list

- **The list is enriched.** Rare tags are over-sampled and the strata are not proportional to the
  population, so tag rates on the list are not Project Sidewalk tag rates, and κ (which depends on
  prevalence) is agreement *on this list*. It is the human ceiling for a model scored on the same
  items, not a population estimate.
- **List-time positives are the original labeller's tags**, not truth. The reviewed counts will
  differ; how much is part of the result.
- **One rater (D1).** The reviewed tags are Jon's judgment under this rubric, not an agreed
  reference. There is no κ, so no human ceiling can be quoted from this pass, and the size above
  was chosen for a κ that is not being measured yet. Say "one rater" next to every number.
- **Items Jon already touched** (117 of 500: placed, validated or edited, in
  `prior_contact_jonfroehlich`) are reported both ways (D11).
- **If a second pass happens** on the sheet route, the two passes use different views (gallery vs
  Google Maps link); the agreement report names both routes and warns when they differ.
- **114 items come from private deployments** (D20, kept).
- The two city-specific tags (`parallel lines` 5, `not aligned with crosswalk` 4) will not reach a
  readable κ on this list. That gap is by design; a κ for them needs a Burnaby- or Taiwan-only
  list.

## Files

| file | contents |
|---|---|
| `benchmark/tag_review/review_list.csv` | the list, one row per item, with editor, label-map and Google Maps links |
| `benchmark/tag_review/review_list.meta.json` | build parameters, eligibility funnel, pool sizes, input file hashes, composition |
| `benchmark/tag_review/<rater>.json` | a rater's export, rubric embedded (none yet; the first is Jon's pass) |
| `rampnet/tag_review.py` | export format, rubric loading, κ and weighted κ, the production reconstruction |
| `scripts/analysis/tag_review_list.py` | `build` the list; `power` and `size` the size tables; `pitch-check` the world-frame check behind the distance band |
| `scripts/analysis/tag_review_pull.py` | `prod` / `sheet` / `sheet-template`: a pass to an export |
| `scripts/analysis/tag_review_agreement.py` | two exports to per-tag κ, prevalence and severity weighted κ |

## Sources and what was not verified

- PROWAG R3 technical requirements (R302.5, R302.6, R304, R305), read from
  `access-board.gov/prowag/technical.html` on 2026-09-22; final rule publication date from the
  PROWAG landing page. Section numbers above are from that page.
- 2010 ADA Standards 405.5, 405.10, 406.2, 406.4 and the detectable-warning note, read from the
  Access Board's guide to the ADA Standards, chapter 4 (ramps and curb ramps), 2026-09-22.
- **Not verified:** the FHWA maintenance guide section (3.2.2) the labeling guide cites for debris;
  whether PROWAG R205 adds scoping exceptions for detectable warnings (only R304.1 and R305 were
  read).
- Tag descriptions: the per-deployment `/v3/api/labelTags` responses in the audit cache, 2026-09-22.
- `pano_y` convention: SidewalkWebpage `public/js/common/pano-viewer/src/panoUtilities.js`
  (`povToPanoCoord`), `public/js/explore/src/label/Label.js` (constructor),
  `GsvViewer.js` and `PannellumViewer.js` (`getPov`), at `8a542b8`.
- Labeling guide: SidewalkWebpage `labelingGuideCurbRamps.scala.html` at `8a542b8`; tutorial
  severity anchors from `public/js/explore/src/onboarding/OnboardingStates.js` and
  `public/locales/en/audit.json` at the same commit.
