# Corner-level cluster review: pre-registered protocol (issue #224)

**Status (2026-09-29): pre-registered, no review done.** Written before any unit was reviewed and
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
  assignments__<rater>.json   a second rater (tracked)
  crops/  aerial/  gallery/   pixels and the rendered tool (git-ignored)
```

## Sampling rule (rule_version 1)

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
              "residential": 8, "mid_block": 7}, "double_rate_share": 0.2, "rule_version": 1},
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
    "inventory_seen": false, "edited_after_inventory": false}}}
```
`inventory_seen` is set (sticky) the first time the tool reveals the city inventory on the unit
(on completion); `edited_after_inventory` is set by any later edit, including reopening. Both are
booleans; `edited_after_inventory` without `inventory_seen` is invalid.
`validate()` refuses: a wrong `schema`; a `snapshot_sha256` other than the bundle's; a complete
unit with a label missing from `labels`; a label key not in the unit; a ramp key referenced by a
label that is not in `ramps`, or a ramp with no label; an uncovered point outside the window or
within 1 m of an assigned ramp's position (the tool refuses such a click, and refuses Export
while one exists in a complete unit); a negative `elapsed_s`; a non-boolean inventory flag. The
auto-labeler's scorer additionally refuses any label value other than `^r\d+$`, `not_ramp` or
`unsure`, and any `rubric_version` other than 1.

## Metrics (`inventory_clustering.assignment_metrics`)

**Every arm clusters the same label set** (amended before any review, see "Amendment"):
(a) human labels (`user_kind: human`) are excluded from scoring in every arm, and their count is
reported; (b) a snapshot label an arm does not hold (`ps @ t` holds no human label; fusion drops
labels on unplaceable panos — 34 in the Vancouver bundle) is scored as a **singleton cluster** of
that arm. Coverage is therefore arm-independent by construction; only split, merge and validity
differ between arms.

An **arm** maps each label key to the set of clusters holding it (one cluster normally; a stale
deployed pull can hold a label twice). Only **complete** units count; within a unit only its own labels are read (a cluster's
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
