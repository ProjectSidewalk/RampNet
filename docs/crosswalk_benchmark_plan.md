# Crosswalk detection benchmark: plan (Fable 5.2, claude-fable-5-2) + Opus 5.5 vet

**Status: proposed, not approved.** Nothing in this plan has been started; the 12 decisions below are
open. Prepared 2026-11-17 for #274 / #298 and committed unchanged on 2026-11-25 (it had lived only in a
session scratchpad). The plan body is the Fable planner's report as delivered (read-only agent); the
Opus 5.5 (`claude-opus-5-5`) vet is at the end. The prior-work survey it should be read with is
[`crosswalk_prior_work/survey.md`](crosswalk_prior_work/survey.md).

**Changed since 2026-11-17** (read before acting on the plan):
- The unmerged-branch concern (decision 10, vet concern 2) is half done: `stage1-label-type-274` merged
  as PR #300 on 2026-11-25; `crosswalk-followups-285` is PR #301 and `crosswalk-supervision-s7` is
  PR #296, both open; `crosswalk-gt-phase1` and `crosswalk-inventory-274` have no PR.
- The retrain this plan's hard negatives feed is drafted as `crosswalk_s8` (branch `crosswalk-s8`,
  `docs/crosswalk_s8.md`), not approved.
- The 2026-11-08 survey lists more held-out city inventories than this plan names (Los Angeles,
  Austin, Paris, Calgary, Portland, Arlington), and WildPASS and Argoverse 2 as external tests.
- DC is cited below as 14,279 inventory polygons; the survey found DDOT's crosswalk **point** layer
  (16,117 points, intersection-only, no collection date). Which DC layer is meant is unverified.

## Summary
- Unit: one marked crossing of one carriageway, kerb to kerb, as a world-space oriented rectangle (axis + width) from orthoimagery or a city inventory. Medians split into two crosswalks sharing a crossing_group. Styles are attributes: ladder/continental, two-line, solid/decorative, brick/paver (+faded, raised). Ignore regions: green bike crossings, private lots, can't-tell. Negatives: unmarked legs, stop bars, driveway crossings.
- Matching: detection point -> bearing ray x range INTERVAL from the row (depression ± δ, δ = one 2.8125° cell + tilt) -> ground segment; candidate if it intersects the GT rectangle buffered by τ m; one-to-one per pano by confidence. Never uses a point range, so the PS far-field shortfall (#192) does not enter. τ, δ measured from existing human clicks (provisional τ 3 m, δ 4.2°); per-split qualification gate (≥95% of Jon's clicks inside footprints; <10% overlapping footprints).
- Eligible view: 2.3-25 m, line of sight clear of OSM buildings, date inside evidence interval, visibility card not "hidden". 25-50 m optional (TP counts, miss doesn't).
- Metrics: per-view recall; strict precision (duplicates = FP); on-crosswalk precision (what a PS Agree measures); crosswalk recall over K = 3 views (near/mid/far); duplicates incl. seam doubles; placement along/across (flat raycast and ps_placement); along-axis position; false alarms per 100 panos per hard-negative stratum; OSM strata (osm_marked / other node / absent) + a map-only baseline row. Bootstrap over intersections.
- Headline to pre-register: per-view instance F1 on test panos, macro over cities, each model at a dev-chosen operating point. Secondary: crosswalk-level recall.
- GT construction: seed = inventory ∪ OSM; AI first pass on ortho (painted? rectangle, style); AI completeness sweep within 50 m of benchmark panos; Jon's local HTML page shows only AI "absent"/"unclear", source disagreements, AI additions, and a 10% audit with a Wilson stop rule; per-pair visibility card (AI pre-rates, Jon sees hidden/no-paint/unsure, ~15-25%); pooled adjudication of unmatched detections after baselines. Fallback if AI rectangles fail (median IoU < 0.6 on 30): two kerb clicks + width preset (~1 h per 300).
- Temporal: present between first and last positive evidence; pano eligible inside that or within 18 months of a positive ortho; otherwise temporal_unverified (out of denominators unless the card confirms paint).
- Cities v1: Juneau (Mapillary, census ~200 marked) + Bend (GSV, 300 marked + 150 unmarked from city inventory). Seattle = dev/in-distribution reference. v1.1: Washington DC (14,279 inventory polygons), a GoPro Max Mapillary city (disjoint from s6's Sacramento/Brussels), optional Panoramax (Lyon/Bayonne). Power: Wilson half-width 0.055 at n = 300; paired 0.10 recall gap needs ~300 crosswalks after design effect; Juneau (~200) supports gaps ≥ 0.12.
- Hard negatives, reported apart: scene-selected strata without the detector (GoPro Max rear seam, brick footways via OSM surface, steps within 15 m, turn lanes, driveways, parking lots, ortho-verified unmarked legs; 40 panos each + 100 random empty); Jon's 99 Disagree labels as an s4-adversarial regression set (RampNet#298). Mine #298 negatives outside a 50 m buffer of the Juneau test area.
- Baselines: RampNet s4 at 0.23 (3 seeds; + rig mask/seam NMS report-only; s1/s5/s6 archive rows); map-only (OSM marked nodes); Mask2Former Vistas (not independent of s4); Grounding DINO, OWLv2 (exist), Florence-2, SAM 3 (new, ~½ day each); Qwen3-VL-32B/8B, Molmo2-8B (exist), Gemma 4 31B if grounding smoke passes; Gemini 4 Argon, Gemini 3.8 Flash, Claude Opus 5.5 low effort + newest Fable (exist). Feeding: 6 × 90° tiles at 1024 px = parity with RampNet's 4096-wide input; each family may pick on dev among parity tiles / full equirect / both, ≤3 prompts, a threshold. Dev = 20% of intersections per city + all Seattle; test = 80%.
- Compute/money: open weights on klone; Gemini ~$10-15 per leg per 700-pano city on the Google.org credit (expires 2026-12-07; run all pre-declared configs before then, select later from cache); Claude legs on subscription API credits.
- Home: RampNet. benchmark/crosswalk/<city>/ (records, gt_crosswalks.geojson, views, negatives, verdict files, imagery manifest); rampnet/crosswalk_eval.py (CPU scorer, synthetic-geometry tests); docs/crosswalk_benchmark.md (protocol + pre-registration) and a generated scoreboard with --check. Labeler supplies pixels (export_benchmark.py) and s4 detections. Ortho tiles/rating pages to the lab share.
- Phases: 0 decisions, ortho check (Juneau/Bend/DC), Bend inventory calibration (40 rows), AI rectangle pilot (30), tolerances, scorer skeleton, pre-registration (1 wk, $0, Jon 45 min). 1 GT for Juneau + Bend (1-1.5 wk, Jon ~1.5 h). 2 baselines (1 wk; Gemini by 12-05). 3 adjudication + scoreboard + s4 vs Jon's random 80 (1 wk, Jon ~1 h). 4 v1.1 (2 wk, Jon ~1 h).

## Contradictions the planner found in the brief's assumptions
- Existing benchmark bundles are sampled around curb-ramp detections and carry almost no crosswalks (92 of 95 Laurens intersection panos show none) -> draw panos per crosswalk.
- Hold-outs: s4's root_v4 includes 49 Richmond positive and 96 Laurens negative panos; Juneau is clean for s4 only (its 110 marks decided s6/s7).
- Point + radius fails: radius spans neighbouring crosswalks; 47 of 49 Sacramento two-line label points sit on bare asphalt -> extent = hull of the crossing, not the paint.
- Region matching moved Juneau recall only 63 -> 66 of 110: misses are real.
- Unmatched ≠ FP: Jon's audit moved s1 precision 0.889 -> 0.990 (58 of 64 "FPs" were real) -> GT must be exhaustive around benchmark panos.
- Projecting GT into panos is loose (median 0.062 of pano width off on Richmond Mapillary) -> tolerance measured per split in metres.
- OSM: 28% of OSM-marked Seattle nodes have no SDOT point within 10 m; 36% of on-node crowd labels sit on untagged/unmarked nodes -> OSM is a seed and stratum, never the denominator.
- Vistas placed s4's positives -> no contestant may seed GT.
- AI pre-rater: aerial paint agreement 0.98, but on street crops AI "not a crosswalk" was right only 23% of the time -> trust AI "present", Jon reviews AI "absent".
- s4 seeds find 63/59/77 of 110 Juneau marks -> report 3 seeds; gaps < ~0.1 between RampNet variants are null.
- Crosswalk detector code is on unmerged branches (stage1-label-type-274, crosswalk-followups-285 tip 76ff9783 unpushed, crosswalk-supervision-s7); main has only Phase 0 docs -> scorer must stand alone on main.

## Decisions for Jon (planner's list, recommendation first)
1. Headline: per-view instance F1 (rec.) vs crosswalk recall + detection precision.
2. Duplicates count as FP in the headline (rec.) vs neutral.
3. Brick/paver crossings are positives (rec.).
4. Private-lot and bike crossings are ignore regions (rec.).
5. Medians split per carriageway (rec.) vs one per road.
6. v1 = Juneau + Bend, DC + a GoPro Max city in v1.1 (rec.) vs DC first.
7. Keep the Juneau test area out of #298 hard-negative mining (rec.).
8. Recall range limit 25 m (rec.) vs 50 m.
9. Gemini budget before 12-07; which Claude account pays.
10. Merge or open PRs for the three unmerged RampNet crosswalk branches before Phase 2.
11. GSV split publication: bundle without pixels, or as the current benchmark does.

## Not verified (planner)
No orthoimagery source/GSD recorded for Juneau, Bend or DC (Juneau fallback: OSM axis + two-click rectangles from street views, per-pano completeness sweep). Bend polyline orientation unopened. SAM 3 / Florence-2 not in the harness; Gemma 4 grounding untested. AI-drawn rectangles untested (0.98 is presence only). τ, δ provisional.

Critical files: RampNet rampnet/detection_eval.py, rampnet/perspective.py (bearing_hit, raycast_ground), scripts/model_comparison/detectors.py (+compare.py, equirect_tiling.py); RampNet-wt/crosswalk-supervision-s7/.../recall_check.py, recall_region.py; RampNet-wt/crosswalk-gt-phase1/.../gt_page.py, cities.py; labeler ps_placement.py; the Juneau votes CSV; RampNet-wt/crosswalk-inventory-274/docs/crosswalk_inventory_274.md; RampNet docs/crosswalk_orthoimagery_survey.md.

## Opus 5.5 vet (claude-opus-5-5)
Verdict: **sound design; start Phase 0 only.** The matching rule (ray x range interval vs a buffered rectangle) is the right answer to "crosswalks are long": it credits a point anywhere on the crossing, separates near/far crosswalks on one bearing, and keeps PS's far-field placement error out of correctness (placement is scored separately). Using OSM only as seed + stratum + map-only baseline matches Jon's concern.

Checked: the ortho survey (RampNet docs/crosswalk_orthoimagery_survey.md) has no Juneau, Bend or DC entry -> confirmed unverified, and it is load-bearing for geometry and completeness. The Google.org credit expires 2026-12-07 and other lab projects draw on it, so it may run out earlier: run the Gemini legs early in Phase 2, not on 12-05; they need about $30-60 in all. Claude legs must use first-party subscription credit, never Vertex (not covered by the Google.org credit).

Concerns:
1. Scope: v1 is ~4-5 weeks of agent work and ~3 h of Jon. Phase 0 is the cheap go/no-go; don't commit past it until ortho for Juneau and Bend is confirmed.
2. Housekeeping first: the s4 code lives on three unmerged RampNet branches (one with unpushed commits). Push/PR them before anything builds on them (decision 10).
3. K = 3 views per crosswalk makes crosswalk recall depend on which views are drawn; pre-register the draw rule (seeded, by range band) along with the headline.
4. Map-only baseline: in Juneau it will look strong because OSM is unusually complete there; report it, but say so.
5. The production question (what to send to Juneau prod) should not wait for this benchmark; the random 80 + the validator scoring (validator#114) answer it sooner.
