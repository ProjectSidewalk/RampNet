# Crosswalk detector follow-ups (#285): car-top rule, missed-crosswalk check, OSM placer checks

Three pieces of work that follow `crosswalk_s5` (see
[`stage1_crosswalk_sdot_274.md`](stage1_crosswalk_sdot_274.md)). `crosswalk_s4` (seed 1, threshold 0.23)
is the detector of record throughout. Plan approved by Jon on 2026-11-10 after an adversarial review.

| Part | Question | Status |
|---|---|---|
| B | Does a rule on image position remove capture-vehicle false alarms without losing crosswalks? | done, below (exploratory) |
| C | What share of visible marked crosswalks does s4 miss in Juneau and Laurens? | TODO: page not built |
| A | Can OSM marked crossings plus the Vistas snap label many cities (#285)? | TODO: city count and placer checks |

## B. A position rule for capture-vehicle false alarms (exploratory, 2026-11-10)

**Short answer.** No single height cut-off works. The labeler's existing 49° rig mask is safe to switch on
for crosswalks and removes the Laurens car tops that s4 still makes. A small region behind the Trimble
vehicle removes a few more in Juneau. Together they remove 12 of s4's 72 rated false alarms on the
evaluation draws and none of its 359 confirmed crosswalks. Most remaining false alarms are not on the
vehicle at all.

**This is exploratory.** The card positions were known before the rules were written, and one fitting
choice was changed after a first run (below). Any region must be confirmed on a fresh draw before it
ships.

### Data

- **Fit:** the two mining pages, whose cards already became training negatives:
  `hardneg/s4_inputs/stepA/ratings_jon_2026-11-09_carTop.csv` (Laurens, 124 cards) and
  `hardneg/s5_inputs/ratings_jon_2026-11-09_juneauMine.csv` (Juneau, 278 cards after merging duplicates).
- **Test:** s4's rated cards on the evaluation draws, `hardneg/confirm_s3/<draw>/s4/ratings.csv`
  (Seattle, Juneau, Laurens Mapillary, Laurens GSV) and `hardneg/s5_inputs/juneau_eval/s4/ratings.csv`:
  440 detections, of which 359 crosswalk, 72 not a crosswalk and 9 unsure.
- **Human crosswalk labels** (clicks, 119 in all), to see what a rule would hide: the live PS labels
  in Laurens (16) and Juneau (5), Jon's 89 confirmed Richmond positives, and 9 Bayonne labels.
- **Mount** = camera model × city, from the labeler's records (`camera_lookup.csv`, 320 of 424 panos; the
  rest are GSV panos, which have no camera field, and 57 Richmond label panos outside the labeler's run).
- Intervals resample Mapillary sequences, so one vehicle's panos stay together.

No "car top" class is used for scoring. Jon's notes name the vehicle on only some cards (13 of the 124
Laurens mining cards), so each arm is scored by the false alarms it removes against the crosswalks it
removes. The cards whose note does name the vehicle are reported beside that.

### Where the vehicle false alarms are

![Rated cards by mount](figures/crosswalk_followups_285/cartop_by_mount.png)

*Rated cards on the three Mapillary mounts, fit and test pooled. Circles mark notes that name the
capture vehicle. The dashed line is the labeler's rig mask; blue is the fitted box.*

- **Trimble MX7, Juneau:** behind the vehicle, at the image seam (x 0.95–1.00 and 0.00), y 0.62–0.76.
  On the Juneau confirmation draw, 6 of s4's 7 vehicle-noted false alarms are on Trimble panos. The older
  statement that Trimble panos have none held for s1 and s3 only.
- **GoPro Max, Laurens:** the mining cards (from s1) sit in a narrow column at x 0.47–0.53, y 0.56–0.79.
  s4 was trained on those as negatives, and its 7 remaining car tops are elsewhere: x 0.55–0.66 and 0.00,
  y 0.85–0.90, below the rig mask line.
- **GoPro Max, Juneau:** scattered over x 0.40–0.83 at y 0.54–0.74, across four sequences, in the same
  band as that mount's real crosswalks. There is no region to cut.

### Arms and results on the test cards

| Arm | False alarms removed (of 72) | Crosswalks removed (of 359) | Vehicle-noted removed (of 19) | Human labels hidden (of 119) |
|---|---|---|---|---|
| 0. Existing rig mask, y > 0.772 | 7 [0.02, 0.22] | 0 (upper bound 0.8%) | 7 | 0 |
| 1. Single cut-off, y ≥ 0.70 | 16 [0.12, 0.34] | 7 [0.003, 0.04] | 14 | 2 |
| 1. Single cut-off, y ≥ 0.60 | 47 [0.50, 0.78] | 85 [0.19, 0.30] | 17 | not computed |
| 2. One box per mount, fitted on the mining pages | 5 [0.01, 0.14] | 0 | 5 | 0 |
| 0 + 2 | 12 [0.07, 0.29] | 0 | 12 | 0 |

Brackets are 95% intervals on the share, by sequence bootstrap. The full sweep is in `arms.csv`.

- **Arm 0** removes exactly the 7 Laurens GoPro car tops and nothing else. It costs no crosswalk on any
  mount and hides no human label.
- **Arm 1** fails as expected. A crosswalk 3–40 m from a 2.5 m camera sits at y 0.52–0.72, the same band
  as the vehicle. At y ≥ 0.70 it already removes 7 real crosswalks (6 in Seattle) for 16 false alarms.
- **Arm 2** fitted three boxes: Trimble at the seam (x within 0.03 of 0, y ≥ 0.61), Laurens GoPro
  (x 0.47–0.53, y ≥ 0.56) and Juneau GoPro (x 0.80–0.86, y ≥ 0.64). On the fit cards they remove 122 of
  182 false alarms and 0 of 206 crosswalks. On the test cards only the Trimble box does anything: 5 of
  the 43 Trimble false alarms (5 of the 8 vehicle-noted ones), 0 of 111 Trimble crosswalks (upper bound
  2.7%). The Laurens box removes 0 of 9, because s4 no longer fires there. The Juneau GoPro box removes
  0 of 6.

### Reading

1. **Switch the existing rig mask on for crosswalks.** It is the one rule here that is not fitted to
   these cards, and `DetectorSpec.mask_rig` in the labeler already supports it per label type. The
   crosswalk scoring in this repo has never applied it.
2. **The Trimble seam region is worth a confirmation, not adoption.** It rests on 5 test cards from a
   handful of sequences.
3. **Per-mount boxes do not generalise.** The Laurens box was fitted on s1's false alarms and is empty
   for s4; the Juneau GoPro mount has no region. A box per mount would also never cover an unseen vehicle.
4. **Position cannot fix most of what is left.** 19 of s4's 81 non-crosswalk test cards are on the
   vehicle. The rest are brick sidewalks, turn-lane markings, curb ramps and similar, spread over the image.
   If vehicle false alarms still matter after the rig mask, the next thing to try is a mask learned per
   sequence from pixels that never change, which needs no ratings and would cover a new vehicle.

### Fitting choice changed after a first run

Arm 2's first criterion compared a box's false alarms with its share of the whole lower image and found
no Trimble box. The criterion now compares within the same y band (does the box hold more than twice
the false alarms its x width alone would give it, by at least 3 cards, with no fit crosswalk inside),
which is what "the vehicle is fixed in x" predicts. Both runs used the same data; this is one more
reason the result is exploratory.

### Where each number lives

All paths are under `analysis_out/crosswalk_followups_285/cartop/`.

| Number | File |
|---|---|
| Arm table (every arm, split and mount; intervals; upper bounds) | `arms.csv` |
| Human labels hidden per arm and mount | `labels_hidden.csv` |
| Fitted boxes, card counts per mount, the vehicle-noted test cards | `summary.json` |
| Every card with its mount, sequence, verdict and note | `cards.csv` |
| Camera and sequence per pano | `camera_lookup.csv` |

### Replication

```bash
# camera lookup (needs the labeler's run records; the committed camera_lookup.csv makes this optional)
python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py lookup --records \
    <sal-juneau>/runs/juneau/results.jsonl <labeler>/runs/juneau_expansion/results.jsonl \
    <labeler>/runs/laurens/results.raw.jsonl <labeler>/runs/laurens_gsv/results.jsonl \
    <labeler>/runs/richmond/results.jsonl <labeler>/runs/bayonne/results.jsonl \
    <crosswalk274-confirm>/{juneau,laurens_mly,laurens_gsv}_draw_records.jsonl \
    <crosswalk274-confirm>/ps_label_panos_records.jsonl
# arms, intervals and tables (CPU, seconds; seed 285, 2,000 bootstrap draws)
python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py score
python scripts/analysis/stage1_crosswalk_sdot_274/cartop_mask.py figures
```

## C. Missed-crosswalk check, Juneau and Laurens

TODO. Design is in the approved plan: one pano per OSM road-graph intersection (all 99 in Laurens, 80
in Juneau), a blind marking page with no detections drawn, one-to-one matching at radius 0.05, marks
within 25 m as the primary population. A 10-pano timing pilot comes first.

## A. OSM marked crossings as a label source (#285)

TODO. First city count and the two placer checks (non-GSV geometry against known clicks; Vistas
component-exists by crossing style on brick and faded crosswalks).
