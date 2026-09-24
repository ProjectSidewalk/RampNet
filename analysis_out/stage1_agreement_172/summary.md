| convention | P | R | F1 | TP | FP | ignored | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| published (first-in-order, redundant ignored) | 0.9403 | 0.9245 | 0.9323 | 3623 | 230 | 119 | 296 |
| first-in-order, redundant as FP (redundant-only arithmetic) | 0.9121 | 0.9245 | 0.9183 | 3623 | 349 | 0 | 296 |
| shared matcher, x not cyclic | 0.9152 | 0.9275 | 0.9213 | 3635 | 337 | 0 | 284 |
| **shared matcher (corrected, of record)** | 0.9152 | 0.9275 | 0.9213 | 3635 | 337 | 0 | 284 |

1000 panoramas (207 negative), 3919 gold ramps, 3972 generated points; radius 0.022.

Corrected (shared matcher): **P 0.9152 / R 0.9275**, pano-clustered bootstrap 95% CI P [0.9044, 0.9252], R [0.9182, 0.9370].
Delta from the published 0.9403 / 0.9245: P -0.0251, R +0.0030; from the redundant-only 0.9121: P +0.0031 (0.9121 is not a bound on the corrected precision).

Matching rule alone (first-in-order -> nearest unclaimed, both redundant-as-FP): net +12 TP (13 gained, 1 lost, across 14 panoramas).

Point order (Stage 1 points carry no confidence): stored order gives TP 3635; 200 seeded per-pano shuffles give TP 3635-3637; the maximum one-to-one matching gives TP 3637 (P 0.9157 / R 0.9280).

