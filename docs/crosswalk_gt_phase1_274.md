# Crosswalk Phase 1 item 2 (#274): multi-city GT page and seeded draw

Phase 1 item 2 of [#274](https://github.com/ProjectSidewalk/RampNet/issues/274) (gate comment of
2026-11-01; Phase 0 is [`crosswalk_phase0_274.md`](crosswalk_phase0_274.md), the city choice comes from
[`crosswalk_orthoimagery_survey.md`](crosswalk_orthoimagery_survey.md)). Built 2026-11-02. Scripts:
[`scripts/analysis/crosswalk_gt_phase1_274/`](../scripts/analysis/crosswalk_gt_phase1_274/); tables:
[`analysis_out/crosswalk_gt_phase1_274/`](../analysis_out/crosswalk_gt_phase1_274/).

**Status: built, not yet rated.** The sheet (320 rows in four cities), the replay archive, the blind page
and the blind AI pre-rate exist; the human verdicts do not. Nothing below is a result about crossings yet:
this doc is the protocol, the sample and the rubric, written down before anyone rates, so the reading
rule cannot drift toward the numbers.

## What the round asks

Phase 0 rated 120 Seattle rows, all on OSM nodes that already carried a human-confirmed crowd label, and
found marked ∪ untagged a usable positive set and `unmarked` not a negative set: in Seattle, on crossings
the crowd found. Phase 1 asks the same two questions in four more cities with different imagery, tagging
habits and crowd validation, and adds a stratum the crowd did not find:

1. **Q1, paint at the node**, per OSM class (marked / untagged / `unmarked`), and on `osm_only` nodes (OSM
   crossings with no crowd label within 15 m, in a block the crowd audited). The last stratum measures how
   often an OSM crossing node the crowd passed over is painted, which tag precision on crowd-found
   crossings cannot say.
2. **Q2, where the crowd label sits** relative to the crossing (the centre convention), on rows with a label.

Per city the shortlist gave it a role: **Amsterdam** (the non-US check; 8 cm, CC BY 4.0), **Columbus**
(half its confirmed labels sit on untagged or `unmarked` nodes; 7.6 cm), **Chicago** (a 40-row presence
test on 15 cm imagery: if paint reads at 15 cm, the 10 cm bar is for extents, not presence), **São Paulo**
(a 40-row junction probe: Phase 0 item 2 found its nodes on the junction node, not the crossing).

## City table

`cities.py` holds one entry per city: PS server, each ortho year's service kind / URL / layer / GSD /
minimum fillable extent / credit, and terms. Each service was probed with one request before any bulk
fetch (log in `cities.py`'s docstring).

| city | imagery (years) | GSD | service | terms | status |
|---|---|---|---|---|---|
| Seattle | King County `KingCo_Aerial` (2023, 2025) | 7.5 cm | MapServer `export`; 2025 blank below 36 Mercator m | view only, county licence | Phase 0, parity only |
| Amsterdam | PDOK Luchtfoto `2024_orthoHR`, `2025_orthoHR` | 8 cm | WMS 1.3.0, EPSG:3857 | CC BY 4.0 | drawn, 120 |
| Columbus | OGRIP `osip_most_current` (COLUMBUS_2025) | 7.6 cm | ImageServer `exportImage` | public use at own risk, no named licence | drawn, 120 |
| Chicago | Cook County `CookOrtho2024`, `CookOrtho2025` | 15 cm | ImageServer `exportImage` | "as is", no named licence | drawn, 40 |
| São Paulo | GeoSampa `Orto_PMD_RGB_2017`, `ORTO_RGB_2020` | 12 / 10 cm | GeoServer WMS 1.3.0 (reprojects to 3857) | CC BY-SA 4.0 | drawn, 40 |
| Tucson | City of Tucson `ORTHO_2022_New` | 5.1 cm | ImageServer | licence fields empty: unconfirmed | config only |
| Madison | Dane County `ColorOrtho3Inch2024Web`, `2026Web` | 7.6 / 7.25 cm | ImageServer | disclaimer only: unconfirmed | config only |

No city was dropped: every service answered, none returned a blank tile at the round's 30 m tile, and none
states a term that rules out scripted viewing. Tucson and Madison stay config-only until their terms are
confirmed, as the plan says. Licence notes: PDOK asks for attribution to Beeldmateriaal Nederland (on every
Amsterdam panel); GeoSampa's CC BY-SA applies share-alike to derived rasters, which is one more reason no
tile or crop is committed; Columbus and Chicago state no licence, so their tiles are viewed, archived
privately and never redistributed. One fetch note: GeoSampa's certificate chains to a Sectigo root that the
Windows curl store lacks; Python's verified TLS accepts it, so nothing was fetched with verification off.

**Seattle parity.** Every Seattle-specific string of the Phase 0 page (title, provider line, service URL,
years, King County's 36 m minimum extent, panel credits, the service name in an error message, the
`imagery` tag) is now read from the table: `phase0_template_generic.html` filled with the Seattle row is
the Phase 0 template character for character (`tests/test_crosswalk_gt_phase1_274.py`), and
`gt_page.py build-phase0 --city seattle-wa` on the Phase 0 inputs writes a page byte-identical to the
committed `analysis_out/crosswalk_phase0_274/aerial_spotcheck_seattle-wa.html` (sha256 `7c3506e4…f288`,
checked 2026-11-02; the same test runs it when `CROSSWALK_SEATTLE_RAWLABELS` points at the census pull).
Phase 0's `aerial_spotcheck.py` was split into `render()` + `build()` so it can take a template; its own
output is unchanged.

## The draw

`draw.py`, seed string `crosswalk-gt-phase1-274` (each stratum seeded `f"{seed}:{city}:{stratum}"`, so a
city or stratum can be redrawn alone). Inputs: the Phase 0 census cache (v3 rawLabels Crosswalk pull and
Overpass `highway=crossing` tiles, pulled 2026-10-28), sha256 of every file in `draw_manifest.json`. The
unit is an **OSM node**, one row per node, because the question Stage 1 asks is whether a node is a
painted crossing (Phase 0 drew labels; at most a few nodes there carried two).

- A crowd label attaches to its single nearest node within 10 m (Phase 0 item 2's match radius).
- **marked / untagged / unmarked**: nodes with an attached label that the humans did not reject (human-only
  margin rule, as Phase 0), by the node's tags (Phase 0's `is_marked`). The row carries one of the node's
  labels: a confirmed one (net ≥ 2 human Agree) when the node has any, else an undecided one; `label_tier`
  says which. Confirmed nodes are drawn first; undecided ones fill only a short stratum.
- **osm_only**: nodes with no crowd label of any verdict within 15 m but at least one within 100 m (the
  block was audited). 10–15 m is a deliberate gap. Drawn uniformly over such nodes, any tags.
- **Allocation**: n / 4 per stratum, a short stratum's shortfall passed round-robin to the others.
- **Imagery check** (`archive.py`): candidates are taken in seeded rank order; a candidate whose newest-year
  tile is blank is skipped and the next rank taken, every skip recorded in `archive_log.json`.

| city | crowd Crosswalk labels (confirmed / undecided / rejected) | OSM crossing nodes | pool marked / untagged / unmarked / osm_only | confirmed nodes in pool (marked / untagged / unmarked) | drawn marked / untagged / unmarked / osm_only |
|---|---|---|---|---|---|
| Amsterdam | 2,976 (890 / 1,974 / 112) | 8,769 | 896 / 11 / 12 / 359 | 545 / 4 / 3 | 49 / 11 / 12 / 48 |
| Columbus | 889 (177 / 705 / 7) | 12,925 | 169 / 323 / 27 / 493 | 58 / 64 / 3 | 31 / 31 / 27 / 31 |
| Chicago | 28,091 (3,907 / 24,021 / 163) | 247,831 | 7,641 / 11,036 / 354 / 30,805 | 1,130 / 1,967 / 30 | 10 / 10 / 10 / 10 |
| São Paulo | 105 (0 / 105 / 0) | 15,105 | 56 / 11 / 6 / 73 | 0 / 0 / 0 | 12 / 11 / 6 / 11 |

Read the table before reading any result off it:

- **Amsterdam tags almost every labeled crossing marked** (896 of 919 labeled nodes), so its untagged and
  `unmarked` strata are the whole population (11 and 12 nodes), mostly undecided labels, and the shortfall
  went to marked and osm_only. Amsterdam answers Q1 for marked and osm_only; its other two cells are small.
- **Columbus `unmarked`** has 27 nodes in all, 3 with a confirmed label: the stratum is mostly undecided.
- **São Paulo has no human-confirmed Crosswalk label**, so every labeled row there carries an undecided one;
  its pools are small enough that all six `unmarked` nodes are in.
- **Chicago**: three `unmarked` candidates (ranks 3, 4, 9) sat outside the Cook County imagery (uniform
  blank tiles in both years) and were replaced by ranks 10–12. No other city skipped a candidate, and no
  accepted row is missing a year.

The sheet has **320 rows**: Amsterdam 120, Columbus 120, Chicago 40, São Paulo 40; 220 carry a crowd label
and all 220 have the crowd's crop. Rows are shuffled across cities with `random.Random("…:order")` and
numbered r000–r319, so neither the city nor the stratum can be read off the row number.

## The page

One page for all four cities (`gt_page.py build`, template `gt_page_template.html`), not one per city: one
export, one rater box and one localStorage key, with a city filter. Four pages would mean four exports to
keep apart, and the Phase 0 ratings once split across two browser profiles. The page is a local HTML file
inside the replay archive that renders from the archived tiles and crops (no Artifact; it needs no network
except the font and any link the rater follows). Per row: one ortho panel per year (30 m wide on the
ground, north up, OSM node blue, crowd label red, 2 m and 5 m rings), the crowd's street-level crop, and a
facts panel (row, city, the OSM node id as a link, the label id, its distance from the node, label and pano
dates, a Street View link). `Z` zooms 2x, `G` hides markers.

**Blind.** The row payload is exactly `ROW_KEYS` in `gt_page.py`: no stratum, OSM class, tags, label tier,
validation counts or AI rating; the test asserts it. Unlike the Phase 0 page, nothing is pre-selected and
no "AI said" line exists. The OSM node link does show its tags if followed; the page says to rate first.

**Rubric** (on the page, version `phase1-2026-11-02`; also `prerate/RUBRIC.md` in the archive):

- **Q1 paint at the node**: **Y** a painted pedestrian crossing passes through the node or within ~2 m (inner
  ring) in *either* ortho year *or* the crop; faded but visible = yes. **R** paint within the 5 m ring but
  clearly not at the node. **N** no paint within 5 m; an unmarked crossing is N. **U** the images cannot tell.
- **Q2 crowd-label position** (labeled rows only), as Phase 0: **1** at the centre on the paint, **2** near the
  centre (in Seattle, 3–5 m off was judged near and ~6 m off was off), **3** at a curb end, **4** not on this
  crossing, **0** no paint to judge against. **B** = the label sits on a green bike crossing.

**One rubric change from Phase 0, made deliberately:** Q1 gains **R**. In Seattle nodes sat on the paint, so
"at the node" needed no radius. São Paulo's nodes sit on the junction node (Phase 0 item 2), and a node
5 m off a zebra is a different finding from no zebra. Y alone equals Phase 0's yes; Y ∪ R is "paint within
5 m". Both are reported.

## AI pre-rate (blind)

`prerate.py sheets` draws one composite per row (newest year 30 m, its centre 15 m at 2x, the older year,
the crop; the same markers) and an index carrying only `gt_id` and whether a crowd label exists. Four fresh
Claude Opus 5.5 subagents (80 rows each) rated the composites against the rubric file and were given no other
file: no city, stratum, tags or label tier. `prerate.py collect` checks every row is rated once with legal
codes and writes `prerate_ai.csv`. It is read only by `gt_page.py merge`, never by the page builder, and its
numbers are not quoted here before the human pass, so the human rater can read this doc without seeing them.

Cost: the four raters ran in parallel on 2026-11-02 for 4–5 minutes each (229–298 s wall clock), 1,095,834
subagent tokens in all, inside the Claude Code subscription (no API spend).

**A rubric gap the raters exposed** (stated here because it changes how Q1 is coded, not what was found):
the rubric says "painted *pedestrian* crossing" but does not say what to do when the paint at the node is a
**bike-only crossing** (Dutch red-surface or block-and-dash cycle crossings, green bike boxes) or
**give-way shark teeth** on a raised crossing. The raters split: the r000–r079 rater coded bike-only paint
`no` or `ring` (r029, r038, r039, r050); the r080–r159 and r160–r239 raters counted it as painted (r082, r094,
r099, r131, r178, r229, with `bike=yes` on r082, r178, r229); shark teeth were coded `no` or `unclear` (r171,
r177, r186). Phase 0 had the B tag for *labels* on bike crossings but never a Q1 rule for a *node* on one,
because Seattle's sample barely raised it. This is Jon's call before he rates. The proposal is that only
pedestrian paint counts for Q1, with a note on the row; the AI rows above are then recoded to match before
agreement is read, the same way Phase 0 re-rated its first pass before quoting agreement.

## Reading rule (stated before any rating)

- Per city × stratum, Q1 Y and Y ∪ R counts with Wilson intervals; a stratum of ≤ 12 rows is reported as
  counts, never as a rate.
- **Positive set**: a class is usable as a Stage 1 positive set in a city if Y ∪ R ≥ 0.9 of its rated rows
  (Seattle: marked 40/40, untagged 39/40). **Not a negative set**: `unmarked` is a usable negative set only if
  N ≥ 0.9 (Seattle: 11/40 no).
- **osm_only**: the share painted is the answer, with no bar: it says how many unlabeled OSM crossings a
  positive set built from tags alone would add.
- **Chicago presence at 15 cm**: the share of rows marked U against the 8 cm cities' rate; if U is no more
  than 0.1 higher, presence reads at 15 cm.
- **Agreement**: human vs AI on Q1 (four-way and Y vs rest) and Q2 (five-way and on-crossing vs not),
  computed by `merge` after the human export, with the Phase 0 0.98 as the reference.

## Replication

| what | command | needs |
|---|---|---|
| draw | `python scripts/analysis/crosswalk_gt_phase1_274/draw.py --census OUT --out analysis_out/crosswalk_gt_phase1_274` | the Phase 0 census cache `OUT` (`crossing_census.py --out OUT`; or the archive's `inputs/`, same paths, sha256 in `draw_manifest.json`) |
| archive + sheet | `python scripts/analysis/crosswalk_gt_phase1_274/archive.py --census OUT --out analysis_out/crosswalk_gt_phase1_274 --archive ARCHIVE` | the four ortho services and the four PS crop routes, live (GETs only, sequential, 0.35 s apart); resumes from `archive_log.json` |
| page | `python scripts/analysis/crosswalk_gt_phase1_274/gt_page.py build --sheet analysis_out/crosswalk_gt_phase1_274/sheet.csv --log analysis_out/crosswalk_gt_phase1_274/archive_log.json --archive ARCHIVE --out ARCHIVE/page/crosswalk_gt.html [--verdicts EXPORT ...]` | the replay archive |
| Seattle parity | `python scripts/analysis/crosswalk_gt_phase1_274/gt_page.py build-phase0 --city seattle-wa --sheet analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated_ai.csv --rawlabels OUT/rawlabels/seattle-wa_Crosswalk.csv --review analysis_out/crosswalk_phase0_274/aerial_spotcheck_review_rows.json --verdicts analysis_out/crosswalk_phase0_274/aerial_verdicts_seattle-wa.json --out X.html`, then compare with the committed page | census rawLabels pull for Seattle |
| AI pre-rate | `prerate.py sheets --sheet … --log … --archive ARCHIVE --out ARCHIVE/prerate`; the raters write `ARCHIVE/prerate/parts/part_*.csv`; `prerate.py collect --parts ARCHIVE/prerate/parts/part_*.csv --sheet … --out analysis_out/crosswalk_gt_phase1_274/prerate_ai.csv` | the archive; a vision model given only the composites and `RUBRIC.md` |
| merge (after rating) | `python scripts/analysis/crosswalk_gt_phase1_274/gt_page.py merge --sheet analysis_out/crosswalk_gt_phase1_274/sheet.csv --verdicts EXPORT --prerate analysis_out/crosswalk_gt_phase1_274/prerate_ai.csv --out analysis_out/crosswalk_gt_phase1_274/sheet.rated_<rater>.csv` | a rater's export |
| tests | `python -m pytest tests/test_crosswalk_gt_phase1_274.py` | none (synthetic census) |
| figures | none yet: there are no results to plot until the human pass; the per-city Q1 figure will come from `merge`'s table | — |

A second rater repeats the round from the same sheet: open the page in a fresh browser profile, type
different initials, export; `merge` takes one export at a time and writes a per-rater CSV, so the two can be
compared row by row on `gt_id`.

**Replay archive** (`ARCHIVE`): local, `D:/Git/sal-scratch-keep/phase1-crosswalk-gt/` on JONFHOME (1,028
files in `SHA256SUMS`; `manifest.json` gives url, status, content type, sha256, bytes and fetch time per file;
`README.md` inside). Not in the repo: the tiles are licensed or share-alike imagery and the crops are
street-level imagery. It should join the Phase 0 archive on the lab share
(`/projects/makeabilitylab/RampNet/archives/crosswalk_phase1_274/`) once copied there; until then it lives
on one machine, and `archive_log.json` holds every file's sha256 so a re-fetch can be proven identical.

## Where each number lives

| number | file |
|---|---|
| labels by verdict, OSM node counts, pool sizes and tiers, quotas, input sha256s | `analysis_out/crosswalk_gt_phase1_274/draw_manifest.json` |
| every candidate with its rank, tags, label and tier | `candidates_<city>.csv` |
| accepted / skipped candidates, skip reasons, tile and crop sha256s | `archive_log.json` |
| the 320 rows as rated, with `gt_id` | `sheet.csv` |
| AI pre-rate | `prerate_ai.csv` (read by `merge` only) |
| human verdicts, per rater | `crosswalk_gt_verdicts_<rater>.json` (export) and `sheet.rated_<rater>.csv` (merge), once rated |
| Seattle parity hash | this doc; `tests/test_crosswalk_gt_phase1_274.py` |

Label ids are per city: rows key on `label_uid` = `<city>:<label_id>` and on `(city, node_id)`.
