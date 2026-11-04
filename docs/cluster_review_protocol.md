# Corner-level cluster review: pre-registered protocol (issue #224)

**Status (2026-11-03): pre-registered, no review done; single rater (Amendment 2); sampling rule v2 and can't-judge (Amendment 3); fusion arm pinned and the bundle's fusion seeds rebuilt (Amendment 4); pilot gate, blinding, inventory reveal, washout and reporting amended after a code review (Amendment 5).** Where an amendment and the text above it disagree, the amendment governs. Written before any unit was reviewed and
before any assignment-based number existed. The rubric is [`benchmark/RUBRICS.md` §6](../benchmark/RUBRICS.md)
(`rubric_version: 1`). Nothing in this document may be changed after the pilot is read except where
it says so (the pilot pass/fail rule may be revised, once, before the full pass, as rubric v2).

**Why.** Label clustering is scored today against city curb-ramp inventories (three cities) or a
GT-free proxy that is broken as a fragmentation measure (#224 body). Both need every cluster
*placed*, and the Vancouver split rates moved with the placement frame (0.255 / 0.200 / 0.091 in the
raycast frame vs deployed 0.196, `ps @ 7.5 m` 0.138, `fusion_server` 0.199, `fusion_server+attach`
0.118 at server placement, r = 5 m; fusion + attach vs ps fixes 601 ramps, breaks 399, both 724).
A label → ramp assignment scores membership, so it needs no placement, radius or camera height, and
one assignment scores every arm on identical labels.

## Who does what

- **sidewalk-auto-labeler** (`scripts/export_cluster_review.py`) samples units, builds both seed
  partitions, cuts crops and aerials, and writes the bundle into
  `benchmark/<city>/cluster_review/`. It alone fetches pixels.
- **RampNet** (`scripts/cluster_review_gallery.py`, `rampnet/cluster_review.py`) renders the review
  tool, validates exports and computes inter-rater agreement.
- **sidewalk-auto-labeler** (`scripts/cluster_review_score.py`, `inventory_clustering.assignment_metrics`)
  rebuilds every arm on the snapshot labels and scores it against `assignments.json`, read as data.

## Bundle layout

```
benchmark/<city>/cluster_review/
  snapshot.json        provenance of every input (tracked)
  corners.jsonl        one line per unit (tracked)
  crops_missing.csv    labels whose crop could not be cut (tracked)
  report.md            the exporter's counts, provenance and timing (tracked)
  assignments.json     THE GT, rater A (tracked; written only by the gallery's Export)
  assignments__<rater>.json   another pass: today the rater's delayed re-review (Amendment 2) (tracked)
  crops/  aerial/  gallery/   pixels and the rendered tool (git-ignored)
```

## Sampling rule (rule_version 2)

1. **Streets and nodes from OSM**: one Overpass query over the run's area bbox (padded 0.003°):
   `way["highway"~STREET_HIGHWAY_RE]` (the auto-labeler's `position_check.STREET_HIGHWAY_RE`:
   motorway … living_street and their `_link`s) plus `node["highway"="traffic_signals"]` and
   `node["crossing"="traffic_signals"]`, `out geom;`. The payload is cached (untracked) with its
   sha256 and fetch time recorded in `snapshot.json` and `report.md`.
2. **Intersection node**: an OSM node whose total leg count over the street ways is ≥ 3 (a way
   passing through the node contributes 2 legs, a way ending there 1).
3. **Merging**: intersection nodes within **25 m** of each other (single linkage) are one unit;
   its centre is their centroid and its `node_ids` all of them.
4. **Type**: `signalised` if any merged node is a traffic-signal node or a signal node lies within
   **30 m** of the centre; else `residential` if every leg's highway is `residential`,
   `living_street` or `unclassified`; else `arterial`.
5. **Mid-block candidates**: points every **20 m** of arc length along every street way, kept when
   **> 60 m** from every intersection node; type `mid_block`.
6. **Eligibility** (added while implementing, before any review; see "Decisions" below): a unit
   centre must lie inside the run's area polygon and within **20 m** of an *open* street of the
   city's Project Sidewalk street network (the network the deployment labelled over). Without it
   an empty unit could be a place no imagery was ever looked at.
   **6b. Grade separation** (rule_version 2, Amendment 3): a candidate is excluded when its
   30 m window touches (a) any OSM way — street, railway, footway, anything — tagged
   `bridge` ≠ `no`, `covered` ≠ `no`, or `layer` ≥ 1, or (b) a street way
   (`STREET_HIGHWAY_RE`) tagged `tunnel` ≠ `no` or `layer` ≤ −1. The Overpass query of rule 1
   also fetches `way["bridge"]["bridge"!="no"]`, `way["covered"]["covered"!="no"]` and
   `way["layer"~"^[+]?[1-9]"]` (the street ways of rule 1 already carry their `tunnel` / `layer`
   tags for (b)); the street filter of rules 2–5 is unchanged. Excluded candidates are counted (`grade_separated`, overall and per stratum) in
   `snapshot.json` and `report.md`.
7. **Window**: radius **30 m** about the centre. A label is in a unit when its **server** lat/lng
   is within 30 m. `has_labels` = at least one label.
8. **Draw**: candidates are shuffled with `random.Random(seed)` (seed **224**) once per stratum,
   strata taken in the order signalised, arterial, residential, mid_block. Per stratum the target
   is **20** units, of which `round(0.15 × 20) = 3` are drawn from the no-label candidates and 17
   from the labelled ones. A candidate is accepted only if its centre is ≥ **60 m** from every
   unit already accepted in *any* stratum (the `_SpatialIndex` rule of `gt_gallery` /
   `export_benchmark`), so no label is in two windows. Shortfalls are reported, never topped up
   from another stratum.
9. **Pilot**: 30 units, quotas signalised 8 / arterial 7 / residential 8 / mid_block 7. Within each
   stratum the selected units are ordered by `sha1(corner_id)`; the pilot takes the first
   `round(0.15 × quota)` no-label units (1 each, 4 in all) and fills the rest with labelled ones in
   that order. Every pilot unit is double-rated.
10. **Second rater**: pilot units in `sha1(corner_id)` order alternate `rater_b_seed` = `fusion`,
    `deployed`, `fusion`, … (index 0 is fusion). Of the non-pilot units, the first
    `round(0.2 × n)` in hash order get `double_rate: true` with the same alternation; the rest have
    `rater_b_seed: null`.

`corner_id` is `<city>:<sig|art|res|mid>:<serial>`, serial = acceptance order within the stratum.
A re-export with the same inputs and seed reproduces every id; the exporter never re-samples an
existing `corners.jsonl`.

## Seed partitions

- **`deployed`**: the server's clusters as pulled (`ps_clustering_eval/clusters.geojson`). Where a
  label sits in two served clusters (Vancouver has such labels; the pull is stale, #56), its seed
  group is the lower cluster id and the others are listed in `seed_group_all`.
- **`fusion`**: `fusion_server+attach` in the `auto` frame, built exactly as
  `inventory_clustering.score_server_arms` builds it (`inventory_clustering.server_arms`, a verbatim
  duplicate; the exporter checks it against `split_figures/state.pkl` when that cache exists).
- A label with no seed group in the chosen arm opens **unassigned**.

## Schemas

`snapshot.json`
```json
{"schema": "rampnet.cluster_review.snapshot/1", "city": "vancouver",
 "labels": {"kind": "server_pull", "path_in_run": "provenance_gate/raw_labels.geojson",
            "url": "...", "fetched_at": "...", "sha256": "...", "n_features": 64847,
            "ai_user_id": "51b0b927-...", "tier": 0.55},
 "seed_arms": {"deployed": {"path_in_run": "ps_clustering_eval/clusters.geojson", "url": "...",
                            "sha256": "...", "fetched_at": "...", "n_features": 18684},
               "fusion": {"definition": "fusion_server+attach, auto frame, inventory_clustering.server_arms",
                          "results_sha256": "...", "n_clusters": 16101}},
 "osm": {"query": "...", "fetched_at": "...", "sha256": "...", "n_ways": 0, "n_signal_nodes": 0},
 "ps_streets": {"path_in_run": "ps_clustering_eval/streets.geojson", "sha256": "...", "fetched_at": "..."},
 "aerial": {"source": "Esri World Imagery", "url_template": "...", "zoom": 20, "half_m": 35, "px": 512,
            "attribution": "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"},
 "crops": {"fov_deg": 45, "px": 512, "resolution": "model (4096x2048-equivalent)", "store": "..."},
 "sampling": {"seed": 224, "per_stratum": 20, "empty_share": 0.15, "window_m": 30, "spacing_m": 60,
              "merge_nodes_m": 25, "signal_radius_m": 30, "midblock_min_m": 60, "midblock_step_m": 20,
              "eligible_street_m": 20, "pilot_quota": {"signalised": 8, "arterial": 7,
              "residential": 8, "mid_block": 7}, "double_rate_share": 0.2, "rule_version": 2,
              "grade_separation": {"rule": "...", "window_m": 30}},
 "exported_at": "...", "exporter": "sidewalk-auto-labeler scripts/export_cluster_review.py@<git sha>"}
```

`corners.jsonl` — one line per unit:
```json
{"corner_id": "vancouver:sig:000001", "city": "vancouver", "type": "signalised",
 "centre": {"lat": 0, "lng": 0}, "window_m": 30, "node_ids": [123, 456], "highways": ["primary"],
 "has_labels": true, "n_labels": 12, "pilot": true, "double_rate": true, "rater_b_seed": "fusion",
 "aerial": {"file": "aerial/vancouver_sig_000001.jpg", "px": 512, "north_up": true, "zoom": 20,
            "world_px": {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
            "bbox": {"south": 0, "west": 0, "north": 0, "east": 0}},
 "labels": [{"key": "24921", "label_id": 24921, "pano_id": "...", "user_kind": "ai",
             "pano_x": 8463, "pano_y": 3692, "pano_width": 13312, "pano_height": 6656,
             "x": 0.6357, "y": 0.5547, "lat": 0, "lng": 0,
             "camera": {"lat": 0, "lng": 0, "heading_deg": 0, "source": "run|inverted|none"},
             "capture_date": "2017-08", "crop": "crops/24921.jpg",
             "seed_group": {"deployed": "d3353", "fusion": "f17"}}],
 "inventory": [{"lat": 0, "lng": 0, "unit_id": "CR26188"}]}
```
`key` is the server `label_id` as a string for a live pull, `<pano_id>:<pano_x>:<pano_y>` for
synthesised labels (file names replace `:` with `_`). The aerial maps a lat/lng to image pixels
through Web Mercator world pixels at `zoom`: `u = (wx - x0) / (x1 - x0) × px` (never linear in
latitude). `inventory` is present only for cities with one and is shown only after completion.

`assignments.json` (rater A) / `assignments__<rater>.json`:
```json
{"schema": "rampnet.cluster_review/1", "city": "vancouver", "snapshot_sha256": "<labels sha256>",
 "rubric_version": 1, "seed_arm": "deployed|fusion|mixed", "rater": "...", "role": "a|b",
 "exported_at": "...",
 "review_notes": {"reviewer": "...", "reviewed_at": "...", "confidence": "high|medium|low",
                  "summary": "", "caveats": []},
 "corners": {"<corner_id>": {
    "seed_arm": "deployed|fusion", "stratum": {"city": "vancouver", "type": "signalised", "has_labels": true},
    "labels": {"<key>": "r1" | "not_ramp" | "unsure"},
    "ramps": {"r1": {"lat": 0, "lng": 0, "placed": false}},
    "uncovered": [{"lat": 0, "lng": 0, "unsure": false}],
    "complete": true, "elapsed_s": 47.2, "note": "",
    "inventory_seen": false, "edited_after_inventory": false,
    "cant_judge": false, "cant_judge_reason": ""}}}
```
`inventory_seen` is set (sticky) the first time the tool reveals the city inventory on the unit
(on completion); `edited_after_inventory` is set by any later edit, including reopening. Both are
booleans; `edited_after_inventory` without `inventory_seen` is invalid. `cant_judge` (a boolean;
absent = false) marks a unit the reviewer could not judge (Amendment 3); it requires a non-empty
`cant_judge_reason` and is never `complete` at the same time. Its labels may be partly assigned;
they are ignored.
`validate()` refuses: a wrong `schema`; a `snapshot_sha256` other than the bundle's; a complete
unit with a label missing from `labels`; a label key not in the unit; a ramp key referenced by a
label that is not in `ramps`, or a ramp with no label; an uncovered point outside the window or
within 1 m of an assigned ramp's position (the tool refuses such a click, and refuses Export
while one exists in a complete unit); a negative `elapsed_s`; a non-boolean inventory flag. The
refusals also cover `cant_judge`: a non-boolean, together with `complete`, or without a
reason. The auto-labeler's scorer additionally refuses any label value other than `^r\d+$`, `not_ramp` or
`unsure`, and any `rubric_version` other than 1.

## Metrics (`inventory_clustering.assignment_metrics`)

**Every arm clusters the same label set** (amended before any review, see "Amendment"):
(a) human labels (`user_kind: human`) are excluded from scoring in every arm, and their count is
reported; (b) a snapshot label an arm does not hold (`ps @ t` holds no human label; fusion drops
labels on unplaceable panos — 34 in the Vancouver bundle) is scored as a **singleton cluster** of
that arm. Coverage is therefore arm-independent by construction; only split, merge and validity
differ between arms.

An **arm** maps each label key to the set of clusters holding it (one cluster normally; a stale
deployed pull can hold a label twice). Only **complete** units count (a `cant_judge` unit never
does: it is excluded from every metric, counted overall and per stratum, and listed with its reason
in the report); within a unit only its own labels are read (a cluster's
labels outside the window are ignored). `unsure` labels are removed from everything; `not_ramp`
labels count only for validity.

- **Per GT ramp r** (a ramp key with ≥ 1 scored label): `clusters(r)` = distinct clusters
  holding any of its labels. **covered** = |clusters(r)| ≥ 1; **split** = ≥ 2.
  `split_rate = split / covered` (Wilson 95% CI). The per-ramp cluster count is kept so two arms
  are compared **paired** on the ramps both cover: fixed (A ≥ 2, B = 1), broken (A = 1, B ≥ 2),
  both split, neither.
- **Per cluster c touching the unit**: its in-window ramp-assigned labels. **merge** when they span
  ≥ 2 ramps (primary); the strict variant (≥ 2 ramps with ≥ 2 labels each, as
  `inventory_metrics`) is reported beside it. `merge_rate` = merged / clusters with ≥ 2 in-window
  ramp-assigned labels.
- **Validity**: clusters whose in-window labels are all `not_ramp` / clusters touching the unit
  (with ≥ 1 non-unsure label); and `not_ramp` labels / non-unsure labels the arm holds.
- **Coverage of the ramp population**: ramps with ≥ 1 scored label / (all ramp keys + sure
  `uncovered` points) — identical for every arm under rule (b).
- Pooled overall, and by stratum type; per city.

**Decision rule (pre-registered; the thresholds are `inventory_clustering.RULE_*`).** On the same
complete units of rater A's full pass: `fusion_server+attach` vs `ps @ 7.5 m` —
split_rate lower by **≥ 0.05** absolute AND merge_rate not higher by **> 0.01** AND coverage not
lower by **> 0.01** → **PASS**; anything else → **NOT ESTABLISHED**. Also reported, not decided on:
`deployed`, `fusion_server`, `ps @ 10 / 12.5 / 15 m`. Pilot-only scores are descriptive.

## Inter-rater agreement (`rampnet.cluster_review.agreement`)

Over units both raters completed (refused for two files with different `rubric_version` or
`snapshot_sha256`):

- **Pairwise same-ramp agreement**: over label pairs in a unit where both raters gave *both* labels
  a ramp key, agree when both say same-ramp or both say different-ramp (Rand-style); report the
  number of pairs and the rate with a Wilson CI; also split by whether the two raters had the same
  seed arm on the unit.
- **Cohen's κ on `not_ramp` vs ramp** per label, over labels neither rater marked `unsure`
  (`tag_review.cohen_kappa`).
- **Uncovered points**: each rater's total of sure points, and the distribution of the per-unit
  |difference|.

**Pilot pass/fail (a guess, pre-registered, revisable once before the full pass):** pairwise
agreement ≥ **0.90** AND κ(not_ramp) ≥ **0.6** → proceed to the full pass; else revise the rubric
(v2) and re-pilot.

## Calibration

- **Against the inventory** (cities that have one; auto-labeler scorer): reviewer ramps (assigned
  ramps at their positions + sure uncovered points) vs inventory points within the window, matched
  one-to-one within 5 m (greedy by distance), reported both directions; and
  `assignment_metrics` vs `inventory_metrics` on the same units, which measures how much of the
  #56 frame dependence was placement. Units with `edited_after_inventory` are **dropped** from this
  calibration (and counted in the report): their answer may have been changed by the inventory.
  They stay in every other table.
- **Against `verdicts.json`** (`rampnet.cluster_review.verdict_consistency`): where a label maps
  pixel-exactly to a judged detection of a RampNet bundle, a `True` verdict should not be
  `not_ramp` and a `False` one should be; disagreements are listed. Vancouver has no bundle, so
  this is empty there.

## What is a guess

- Reviewer time: the issue's ~30–45 s per correct unit / 1.5–3 min per edited unit / "about a
  minute a unit" are guesses. The pilot measures `elapsed_s`; no time figure is quoted until then.
- The pilot thresholds (0.90, 0.6) and the 15% / 20% shares are judgment calls, not derived.
- The Wilson half-width in the issue (~2.7 pts at ~600 covered ramps, split ≈ 0.13) is arithmetic
  on assumed counts.

## Decisions taken while implementing (2026-09-29, before any review)

| question | decision |
| :--- | :--- |
| Aerial imagery | Esri World Imagery z20, attribution recorded per bundle and unit (reversible; one constant). |
| Vancouver's 33 human labels | In the window, reviewed like any label, `user_kind` marks them. |
| 5+-leg / close nodes | Nodes within 25 m merge into one unit. |
| Eligibility (rule 6) | Added: inside the area and ≤ 20 m from an open PS street. |
| A label in two served clusters | Seed = lower cluster id; all listed in `seed_group_all`. |
| File names | `:` in ids becomes `_` in `aerial/` and `crops/` names (Windows). |
| Who is rater B | The gallery's `--role b` (which requires `--rater`) shows only double-rated units and, with `--seed-arm auto`, seeds each with its `rater_b_seed`. |

## Amendment (2026-09-29, after an independent code review, before any unit was reviewed)

No unit had been reviewed and no assignments file existed when these were made.

1. **Same label set in every arm** (rule (a)/(b) under "Metrics"): human labels are excluded from
   every arm; labels an arm does not hold are that arm's singletons. Before, a label missing from
   an arm lowered only that arm's coverage (fusion lost 34 labels, ps @ t all human labels), so
   the coverage leg of the decision rule compared different label sets.
2. **Inventory reveal is recorded**: `inventory_seen` / `edited_after_inventory` (schema above);
   edited-after units leave the inventory calibration.
3. **Timing**: `elapsed_s` accrues in 1 s ticks (each capped at 2 s) only while the unit is on
   screen, the tab is visible, and there has been keyboard or mouse input within the last
   **60 s**; state is saved on `pagehide` and when the tab is hidden. The pilot's time numbers are
   read under this rule.
4. **Scorer input binding**: `cluster_review_score.py` refuses to rebuild arms from deployed
   clusters or a `results.jsonl` whose sha256 differs from `snapshot.json`'s (an explicit
   `--allow-arm-mismatch` overrides and is written into the report); it refuses malformed label
   values and a rubric version other than 1.
5. **Tool storage and roles**: the browser storage key is city + rater + label-snapshot sha256 +
   `corners.jsonl` sha256; `--role b` requires `--rater`, and a rater's existing export made under
   the other role is refused; a prefilled file wins over state the browser merely seeded (local
   work wins only on units seen or completed, and those conflicts are listed).

## Amendment 2 (2026-09-30, single rater, before any unit was reviewed)

No unit had been reviewed and no assignments file existed when this was made.

There is one rater (Jon). No second person rates any unit for now. Wherever this document or
`RUBRICS.md` §6 says "rater B" or "second rater", read **the same rater's delayed re-review**:

1. **Same sampling.** `double_rate` and `rater_b_seed` are unchanged. The 30 pilot units and the
   10 double-rated non-pilot units are re-reviewed, and each re-review opens with its
   `rater_b_seed`, so half open with fusion and half with deployed.
2. **Washout.** A unit's re-review starts **≥ 7 days** after that unit was completed in the first
   pass. The first pass is not reopened or consulted during the re-review. The re-review is a
   separate gallery build (`--role b --rater <name>-retest --seed-arm auto`), exported as
   `assignments__<name>-retest.json`, so the first pass's browser state and file are never read.
3. **What it measures.** The agreement statistics are computed unchanged, but they are
   **intra-rater (test-retest) consistency, not inter-rater reliability**. They cannot show that
   a second person would read the rubric the same way. The report and every quote of them say
   "intra-rater". The `same_seed` / `different_seed` split still measures the seed effect, now
   within one person. It is weaker here because a unit can be remembered across the washout.
4. **Pilot gate.** The thresholds are unchanged (pairwise ≥ 0.90, κ ≥ 0.6). Self-agreement is
   expected to run higher than agreement between two people, so passing the gate is a weaker
   result than the original rule intended, and it is reported as such.
5. **Seed direction.** The first pass is seeded with `deployed` (PS's own clustering, close to
   `ps @ 7.5 m`). Any anchoring therefore pulls the scores toward the baseline arm, not toward
   `fusion_server+attach`. A PASS is conservative with respect to that bias; a NOT ESTABLISHED may
   partly be anchoring. The `different_seed` agreement is how large that effect could be.
6. **External check.** Only the Vancouver inventory calibration (above) is independent of the
   rater. It carries more weight than it did before, and it is reported next to the decision.
7. **If a second rater joins later**, the original rules apply to them unchanged, as an
   additional file. The self-re-review remains a separate, labelled result.

## Amendment 3 (2026-09-30, before any unit was reviewed)

No unit had been reviewed and no assignments file existed when this was made. `rubric_version`
stays **1**: nothing was ever rated under the rubric as it stood, so there is no earlier file for
the change to split from.

1. **Grade separation (rule 6b, sampling `rule_version` 2).** Windows touching an overhead
   structure (a bridge, a covered way, anything at `layer` ≥ 1) or a street below grade are
   excluded before the draw. Why: the aerial is how a reviewer confirms a unit — what is under a
   deck cannot be seen from above, and a corner of a street in a tunnel is a corner nobody stands
   at. The first pilot unit opened (`vancouver:sig:000012`, under a motorway ramp) showed it.
   The rule is objective and applied before anyone looks at a unit, so it cannot bias the sample
   toward easy corners; it narrows the study to corners at grade, by the share `report.md`
   states (measured on the old cached OSM, street ways only: 305 of 21,765 eligible candidates,
   1.4%). The Vancouver bundle was re-drawn under it; the v1 draw was never reviewed.
2. **Can't judge.** A unit may be marked `cant_judge` with a required free-text
   `cant_judge_reason`. Such a unit counts in no metric; every one is counted (overall and per
   stratum) and listed with its reason in the scorer's report, and the agreement report counts
   them for both passes, including units one pass completed and the other could not judge. It is
   for surprises the rule misses — tree cover or shadow hiding the corner, construction, imagery
   missing or unusable for the window — **never for a unit that is merely hard** (many labels,
   confusing geometry): a hard unit is exactly what the review is for.

## Amendment 4 (2026-11-03, the fusion arm's construction is pinned and the bundle's fusion seeds are rebuilt; before any unit was reviewed)

No unit had been reviewed and no assignments file existed when this was made: nothing had been
rated in the v2 gallery (Jon, 2026-11-03).

The arm compared in the decision rule changed after this protocol was written. The change was
made in the auto-labeler for reasons unrelated to this study, before any unit was reviewed, and is
recorded here so that the pre-registered arm has one definition that the score can be checked
against.

1. **What changed.** "Seed partitions" defines `fusion` as `fusion_server+attach` "built exactly
   as `inventory_clustering.score_server_arms` builds it". The auto-labeler's
   [PR #118](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/118) (merged
   2026-10-04, merge commit `66c76d6`; the change itself is `104d892`) changed that construction:
   labels on panos the server arm cannot place are now passed to `clusters_from_server_sites` and
   `attach_unplaceable` as unpositioned labels instead of being dropped. On the frozen Vancouver
   inputs (labels sha256 `57c31c73…`, `results.jsonl` sha256 `7fdf4005…`):

   | | as exported (labeler `5b21cd9`, 2026-09-30) | after #118 |
   | :--- | ---: | ---: |
   | `fusion_server` clusters | 18,414 | 18,484 |
   | `fusion_server+attach` clusters | 16,101 | 16,171 |
   | labels attached | 2,274 | 2,313 |
   | bundle labels with no fusion group | 5 | 0 |

   The "34 in the Vancouver bundle" quoted under "Metrics" rule (b) and in the first Amendment
   counted the v1 bundle (1,406 labels) under the old construction; the v2 bundle as first
   exported had 5, and under the pinned arm fusion holds every label of the bundle. Rule (b)
   itself is unchanged and still applies to `deployed` (1 label unheld) and to `ps @ t` (no
   human labels). With the seeds rebuilt (point 3), "built exactly as `score_server_arms`
   builds it" describes the bundle again.

2. **The arm is pinned.** From this amendment on, `fusion_server+attach` (and `fusion_server`)
   mean the partition built by `inventory_clustering.server_arms` at sidewalk-auto-labeler commit
   [`a65b91a`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/a65b91abcfe7cc6ae76cde034c44c11fccea38c5),
   the commit the re-seed ran at (`snapshot.json`'s `exporter`). `server_arms` and
   `score_server_arms` share one helper, `fuse_server_labels`, so the two cannot drift apart again.
   The construction is the same as at `75b907e`, the head this amendment was drafted against:
   `git diff 75b907e a65b91a -- scripts/inventory_clustering.py scripts/eval_ps_clustering.py
   scripts/fuse_sites.py geo.py` is empty. Any later change to that construction is a deviation:
   it is recorded as a further amendment before any assignment-based number is read, or the score
   is computed at `a65b91a`. The score report names the labeler commit it ran at
   (`scripts/cluster_review_score.py`, since `a65b91a`), so the check is a comparison of two
   strings.

3. **The seeds are rebuilt (no unit had been reviewed in the v2 bundle).** The bundle's `fusion`
   seeds were rebuilt with the pinned arm:

   - **How.** The exporter was re-run at `a65b91a` with `corners.jsonl` and `snapshot.json`
     moved aside, `--no-crops --skip-aerial` (crops and aerials are reused, never re-cut: they
     are git-ignored and are cut on makelab2), on the frozen inputs of the auto-labeler's
     `docs/ps-clustering-eval.md` "Frozen inputs" table, whose six sha256 all matched. The
     Overpass payload was the cached one (sha256 `58ce83ff…`; the exporter reuses a cache that
     answers the same query, which was checked before the run, and the file's sha256 was
     unchanged after it), never a fresh fetch, which could have changed the draw.
     `split_figures/state.pkl` already held the #118 arm, so the cross-check still reads
     "identical partitions". Reconcile: 1,453 crops, 6 listed missing, 80 aerials, OK.
   - **How the result was checked.** A field-level diff of the old `corners.jsonl` (RampNet
     `39155d8`, sha256 `4c862a40…`) against the new one (sha256 `d55c71c8…`): the same 80
     `corner_id`s in the same order; every field of every unit and of every label identical
     except `labels[].seed_group.fusion` (0 other differences, so centres, node ids, windows,
     strata, the pilot and double-rate flags, `rater_b_seed`, aerial geometry, inventory points,
     crop paths and `deployed` seeds are all unchanged); and `snapshot.json` differs only in
     `exported_at`, `exporter`, `seed_arms.fusion.n_clusters` (16,101 → 16,171) and
     `seed_arms.fusion.n_attached` (2,274 → 2,313). The `fusion` seed was compared as each unit's
     in-window partition (group ids are positional and 967 of the 1,459 were renumbered; nothing
     reads them across builds).
   - **What changed.** Counting a label with no fusion group as a group of its own, the in-window
     `fusion` partition changed in **5 of the 80 units, touching 59 labels** (labels in a group
     that differs between the two builds):
     - `vancouver:art:000008`: 5 groups plus 4 labels with no fusion group
       (`vancouver:25640`, `:25641`, `:26221`, `:26222`) become 5 groups; each of the four joins
       a group and the groups are rebuilt around them (14 labels). The draft of this amendment
       said "8 groups become 4"; the diff above is the count.
     - `vancouver:sig:000011` (double-rated, `rater_b_seed: fusion`): `vancouver:75978` moves
       between two groups (24 labels).
     - `vancouver:art:000013` (pilot; `rater_b_seed: deployed`): `vancouver:53333` moves from a
       2-label group into another (9 labels).
     - `vancouver:sig:000003`: `vancouver:78013`, alone, joins a 2-label group (3 labels).
     - `vancouver:res:000002`: `vancouver:63712` leaves its group to stand alone (9 labels).

     A sixth unit, `vancouver:sig:000016`, keeps its partition: its label `vancouver:59734` had
     no fusion group and is now a group of one. Labels with no fusion group: 5 → 0.
   - **Who sees it.** A `fusion` seed is shown only in the retest of a double-rated unit whose
     `rater_b_seed` is `fusion` (Amendment 2: the first pass opens with `deployed`). Of the
     units above, only `vancouver:sig:000011` is one.
   - **Storage.** The new `corners.jsonl` sha256 changes the gallery's browser-storage key, so no
     state a browser held for the old bundle is carried over (none was in use). Since Amendment 5
     the gallery says so when such state exists.

4. Nothing else changes: the sampling rule (v2), the strata, the pilot and double-rate draws, the
   rubric (`rubric_version` 1), the metrics, the thresholds and the decision rule are as before.

## Amendment 5 (2026-11-03, after a code review of PR #280, before any unit was reviewed)

No unit had been reviewed and no assignments file existed when this was made. The review is
[RampNet#280 (comment)](https://github.com/ProjectSidewalk/RampNet/pull/280#issuecomment-6618318887);
its finding ids are given in brackets. This amendment is made before the pilot is read, so it
does not use the one revision of the pilot rule that "Status" allows after the pilot.

1. **Pilot gate [B1].** The rule under "Pilot pass/fail" (pairwise ≥ 0.90 AND κ ≥ 0.6) is
   replaced. The pairwise rate is Rand-style: it counts every label pair, and in a 30 m window
   almost every pair is "different ramp", which any two passes get right. On this bundle the
   `deployed` and `fusion` seeds already agree on 0.932 of pilot pairs (0.931 before
   Amendment 4), and a rater who accepts whatever seed is shown scores 0.963 and passes.
   **New gate:** on the pilot units whose two passes opened with **different** seeds (15 of 30:
   the `rater_b_seed: fusion` ones), the same-ramp **pair F1** = 2·|pairs both passes put in one
   ramp| / (|pass A's same-ramp pairs| + |pass B's|), over label pairs both passes gave a ramp
   key, must be **≥ 0.90 pooled AND ≥ 0.90 as the per-unit mean** (units where neither pass puts
   two labels together have no F1 and are counted apart); AND the κ leg (point 2).
   - Why the different-seed units: on a same-seed unit a rater who accepts the seed agrees with
     themselves perfectly. Over all 30 pilot units that rater scores pair F1 0.854 pooled and
     **0.901 per-unit mean, which would pass**; on the 15 different-seed units they score 0.782 /
     0.830 (unit-level bootstrap 95% CI 0.711–0.845 / 0.767–0.895), which is the seed arms' own
     agreement there (0.783 / 0.830), and fail. The review measured 0.748 pooled / 0.802
     per-unit mean for `deployed` vs `fusion` seeds over all 30 pilot units on the old seeds;
     re-seeded it is 0.751 / 0.803. The test `test_new_pilot_gate_fails_a_rater_who_accepts_every_seed`
     replays that rater on the committed bundle.
   - Reported beside the gate, not gated: pair F1 overall and on same-seed units, each pooled and
     per-unit mean with a **unit-level bootstrap 95% CI** (units resampled, never pairs; 2,000
     resamples, seed 224); `seed_reference`, the seed arms' own pair F1 on the gated units; and
     the Rand-style pairwise rate, whose Wilson CI treats ~12,000 pairs from 30 units as
     independent and is therefore far too narrow. With 15 gated units the bootstrap CI will be
     wide; the gate reads the point estimates, and 0.90 remains a judgment call.
2. **κ leg and which units the gate reads [S5].** κ(not_ramp) ≥ 0.6 over the pilot units both
   passes completed, **or κ undefined because both passes are constant and identical** (no
   `not_ramp`, or no label): that is complete agreement and passes. The `not_ramp` counts are
   reported beside κ; at ~0.97 live precision they will be small and κ noisy. The gate reads the
   pilot units from `corners.jsonl`, whatever selection the agreement is computed over, and
   decides only when every pilot unit is complete or can't judge in **both** passes; otherwise
   (a partial pilot, no `corners`) `pass` is null.
3. **No arm-level number before rater A's full pass [S1].** Replaces "Pilot-only scores are
   descriptive" under "Decision rule": no arm-level number (split, merge, validity, coverage,
   per-ramp tables, the inventory calibration's arm columns) is computed or read until rater A's
   full pass is exported. With one rater who is also the PI, an arm table read after 30 units
   would show which arm is ahead before the other 50 are rated. The pilot is read for agreement,
   timing and can't-judge counts only (`scripts/cluster_review_agreement.py`). The auto-labeler's
   scorer refuses any other file (since `a65b91a`); a retest file is scored, as descriptive,
   only once the bundle's `assignments.json` is rater A's full pass.
   `--allow-partial-descriptive` overrides and is written into the report; using it before the
   full pass is a deviation and is recorded as one.
4. **Inventory reveal [S2].** The city inventory is **never** shown in a retest (`--role b`)
   build, and in the first pass it is **withheld on double-rated units until that unit's retest
   is exported** (a later role-a build shows it). The page does not carry the withheld points.
   Every pilot unit is double-rated, so no pilot unit reveals the inventory in the first pass
   (22 of the 30 have inventory points; 29 of the 40 double-rated units do). Nothing calibrates
   on the retest, and the first pass's calibration does not need the reveal; it only sets
   `inventory_seen` later on those units.
5. **Washout check [S3].** Each unit's export carries a sticky `opened_at` (first shown) and
   `completed_at` (first completed), ISO 8601 UTC (`validate()` refuses a malformed one). The
   agreement report gives each unit's washout in days, the retest's `opened_at` (else its
   `completed_at`, which overstates it) minus the first pass's `completed_at`, and lists every
   unit under 7 days and every unit without the times. Agreement is reported with and without
   the units under 7 days.
6. **"Intra-rater" in the tool output [S4].** The agreement report carries `agreement_kind`
   (`intra` | `inter`), in the headline and the JSON. The script infers `intra` when B's rater is
   `<name>-retest` and otherwise requires `--kind`.
7. **Seed direction [S7].** Amendment 2, point 5's "close to `ps @ 7.5 m`" and "A PASS is
   conservative" are withdrawn. On Vancouver the deployed clusters reproduce PS's own algorithm
   only 0.765 of the time (14,300 of 18,684 clusters; 18,684 deployed vs 17,451 for `ps @ 7.5 m`;
   the auto-labeler's `docs/ps-clustering-eval.md`, "(a) Verbatim reproduction: 0.765"), so
   anchoring toward the `deployed` seed does not map onto favouring either decision arm. The
   direction of any anchoring bias relative to `fusion_server+attach` and `ps @ 7.5 m` is
   unknown; the `different_seed` agreement bounds its size.
8. **What the decision can resolve [S8].** "What is a guess" quoted ~2.7 pts at ~600 covered
   ramps. The bundle implies about **290–380** GT ramps (294 in-window `fusion` seed groups after
   Amendment 4, 380 `deployed`). At split ≈ 0.13 the Wilson half-width is **±3.4–3.9 pts**
   against a 5-pt bar, and that is optimistic: ramps cluster in 68 labelled units, the largest
   with 81 labels. The merge leg's 0.01 margin is about 3 merges. The rule is unchanged; the
   score report puts a **unit-level paired bootstrap 95% CI of the split-rate difference**
   (2,000 resamples, seed 224) beside the verdict, as a descriptive number (the auto-labeler's
   scorer since
   [`898d8f8`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/898d8f8336e7829d5dc21b6f2ec03a4f433fc887)).
9. **The 1 m rule [S9].** Under "Schemas", `validate()` refuses an uncovered point within 1 m of
   an assigned ramp only in a **complete** unit, as the gallery's Export does: an unplaced ramp
   sits at the mean of its labels, so regrouping an unfinished unit can move it onto a mark.
10. **The coverage leg cannot fail [N7].** Under rule (b) every arm clusters the same label set,
    so coverage is identical across arms by construction and the decision rule's coverage leg is
    always met. It stays in the rule for the record; the scorer says so beside the verdict.
11. **The stale "34" [S6]** is corrected by Amendment 4, point 1, not in place.
12. Tool changes with no effect on the protocol: the gallery declares its language, makes crops
    keyboard-selectable with alt text and names each star's and dot's group in a tooltip
    (colour is not the only cue) [N2]; refuses `--out` on another drive before writing anything
    [N3]; warns when the browser holds unexported work for an earlier `corners.jsonl` of the same
    city, rater and snapshot, which a re-seed hides [N5]; escapes the can't-judge reason fully
    [N8]; and `verdict_consistency` looks a label up in its own unit, since a key is a per-city
    `label_id` [N4].
13. Nothing else changes: the sampling rule (v2), the strata, the pilot and double-rate draws, the
    rubric (`rubric_version` 1), the metrics and the decision rule's thresholds are as before.
