# vancouver: cluster-review bundle (RampNet#224)

Exported by `sidewalk-auto-labeler scripts/export_cluster_review.py@d86cbd483bc9f82d8a06b4793659f95e40b84a26`; sampling rule v1 (RampNet docs/cluster_review_protocol.md). No review has been done: this bundle holds no assignments.

## Provenance

- labels: `provenance_gate/raw_labels.geojson`, 64847 features, sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`, fetched 2026-09-28T23:01:58+00:00 from https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- deployed seed: `ps_clustering_eval/clusters.geojson`, 18684 clusters, sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`, fetched 2026-09-30T01:39:55+00:00 from https://sidewalk-vancouver.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- fusion seed: fusion_server+attach, auto frame, inventory_clustering.server_arms (= score_server_arms); 16101 clusters; results.jsonl sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`; state.pkl cross-check: identical partitions (deployed, ps @ 7.5 m, fusion+attach)
- OSM: 16588 street ways, 2447 signal nodes, fetched 2026-09-30T04:55:33+00:00 from https://overpass-api.de/api/interpreter, payload sha256 `5c7fbd7473ab1b3f321b3df5b808e048a7f972c120c8c9dd16e6a1d4fd087a82`
- PS streets (eligibility): `ps_clustering_eval/streets.geojson`, sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`, fetched 2026-09-30T01:39:49+00:00
- inventory: 11355 kept points, sha256 `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87`, fetched 2026-09-28T23:11:26+00:00
- aerial: Esri World Imagery z20, +-35 m at 512 px; Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community
- crops: 45 deg / 512 px, model (4096x2048-equivalent): native crop resampled to 512 px; JPEG draft decode never below 4096 px, from makelab2 /projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa

## Candidates

| step | count |
|---|---:|
| street_ways | 16588 |
| intersection_nodes | 10492 |
| intersection_units | 9509 |
| signal_nodes | 2447 |
| midblock_points | 46835 |
| outside_area | 28437 |
| off_ps_street | 6144 |

| stratum | eligible | of them no-label | labelled drawn / target | no-label drawn / target | spacing rejections (labelled, no-label) |
|---|---:|---:|---|---|---|
| signalised | 303 | 6 | 17 / 17 | 3 / 3 | 0, 0 |
| arterial | 1532 | 433 | 17 / 17 | 3 / 3 | 0, 0 |
| residential | 2652 | 1098 | 17 / 17 | 3 / 3 | 0, 0 |
| mid_block | 17276 | 14692 | 17 / 17 | 3 / 3 | 0, 0 |

## Units

| stratum | units | no-label | pilot | double-rated | labels | human labels |
|---|---:|---:|---:|---:|---:|---:|
| signalised | 20 | 3 | 8 | 12 | 666 | 0 |
| arterial | 20 | 3 | 7 | 11 | 385 | 0 |
| residential | 20 | 3 | 8 | 10 | 278 | 0 |
| mid_block | 20 | 3 | 7 | 7 | 77 | 0 |
| all | 80 | 12 | 30 | 40 | 1406 | 0 |

- no-label share: 12/80 = 0.150
- labels per unit (all units): p0 0, p25 2, p50 12, p75 29, p90 47, p100 94
- labels per unit (labelled units): p0 1, p25 3, p50 16, p75 34, p90 48, p100 94
- labels per unit (pilot units): p0 0, p25 3, p50 12, p75 34, p90 49, p100 94
- deployed seed groups per labelled unit: p0 1, p25 1, p50 4, p75 8, p90 13, p100 18; labels with no deployed group: 1
- fusion seed groups per labelled unit: p0 1, p25 1, p50 4, p75 7, p90 9, p100 12; labels with no fusion group: 34
- labels held by two or more deployed clusters: 10
- rater_b_seed: fusion 20, deployed 20

## Files and reconcile

- crops on disk 1365, listed missing 41, aerials 80 / 80
- STATUS: OK -- every label has a crop or a missing row, every unit an aerial

## Timing (wall clock, every invocation)

| started (UTC) | step | seconds | note |
|---|---|---:|---|
| 2026-09-30T04:56:33+00:00 | arms | 42.0 | 64847 labels; fusion+attach 16101 clusters |
| 2026-09-30T04:56:33+00:00 | overpass | 0.2 | cached |
| 2026-09-30T04:56:33+00:00 | sample | 12.3 | 21763 eligible candidates -> 80 units |
| 2026-09-30T04:56:33+00:00 | aerial | 360.7 | 80 made, 1050 tiles fetched |
| 2026-09-30T04:56:33+00:00 | crops | 55.9 | 1406 requested, 41 missing, from makelab2:/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa |
