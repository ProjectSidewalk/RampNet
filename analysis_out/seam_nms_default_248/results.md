**Outcome under the pre-registered rule: NEXT STEP NEEDED.** Reasons: L_jon_C = 0, L_opt_C = 0, L_pess = 4, ai_calibrated = True, all_cards_rated_by_jon = False, cards_jon_must_rate = ['s248-09', 's248-11', 's248-27', 's248-33'], panoramas_for_exact_tier = ['manual_gold:R6Awyr2RqM-NKCZyeExAwQ', 'morgantown:1061541095152683', 'paterson:iPgLrrQ5XDrWWlTXzI5MLQ'].

Calibration: AI `different` on 4 / 4 of Jon's `two` pairs, `same` on 9 / 10 `one` pairs; calibrated = True. Tripwire fired: False.

| card | split | set | dist px | weak / strong score | source | verdict | class |
|---|---|---|---:|---|---|---|---|
| (inherited) | manual_gold | C | 4 | 0.403 / 0.453 | jon (inherited) | one | D |
| s248-01 | manual_gold | C | 1 | 0.752 / 0.896 | ai | same; weak ramp, strong ramp | D |
| s248-08 | manual_gold | C | 4 | 0.567 / 0.761 | ai | same; weak unsure, strong unsure | N |
| s248-13 | manual_gold | C | 4 | 0.893 / 0.917 | ai | same; weak unsure, strong unsure | N |
| s248-21 | manual_gold | C | 4 | 0.460 / 0.491 | ai | same; weak unsure, strong unsure | N |
| s248-27 | manual_gold | C | 7 | 0.742 / 0.965 | ai | unsure; weak unsure, strong unsure | U |
| s248-29 | manual_gold | C | 4 | 0.782 / 0.922 | ai | same; weak ramp, strong ramp | D |
| s248-36 | manual_gold | C | 7 | 0.867 / 0.988 | ai | different; weak not_ramp, strong ramp | N |
| (inherited) | manual_gold | E | 12 | 0.357 / 0.878 | jon (inherited) | one | D |
| (inherited) | manual_gold | E | 12 | 0.685 / 0.787 | jon (inherited) | one | D |
| (inherited) | manual_gold | E | 15 | 0.893 / 0.938 | jon (inherited) | one | D |
| s248-11 | manual_gold | E | 16 | 0.390 / 0.951 | ai | unsure; weak unsure, strong ramp | U |
| s248-17 | manual_gold | E | 15 | 0.524 / 0.948 | ai | same; weak unsure, strong ramp | D |
| s248-20 | manual_gold | E | 12 | 0.614 / 0.945 | ai | same; weak ramp, strong ramp | D |
| s248-26 | annapolis | E | 13 | 0.504 / 0.787 | ai | same; weak ramp, strong ramp | D |
| s248-03 | bend | C | 7 | 0.800 / 0.843 | ai | same; weak ramp, strong ramp | D |
| s248-16 | bend | C | 4 | 0.698 / 0.795 | ai | same; weak unsure, strong ramp | D |
| s248-25 | bend | C | 4 | 0.523 / 0.867 | ai | same; weak ramp, strong ramp | D |
| s248-28 | budapest_district5 | C | 4 | 0.308 / 0.471 | ai | same; weak unsure, strong unsure | N |
| s248-35 | morgantown | C | 4 | 0.337 / 0.756 | ai | same; weak ramp, strong ramp | D |
| s248-06 | morgantown | E | 16 | 0.478 / 0.974 | ai | same; weak ramp, strong ramp | D |
| s248-33 | morgantown | E | 12 | 0.486 / 0.610 | ai | unsure; weak ramp, strong unsure | U |
| s248-18 | paterson | C | 4 | 0.620 / 0.760 | ai | same; weak unsure, strong ramp | D |
| s248-09 | paterson | E | 20 | 0.366 / 0.914 | ai | different; weak unsure, strong ramp | U |
| s248-14 | richmond | C | 4 | 0.731 / 0.946 | ai | same; weak ramp, strong ramp | D |
| s248-15 | richmond | E | 13 | 0.335 / 0.938 | ai | different; weak not_ramp, strong ramp | N |

| pool, set | t | D | L | N | U |
|---|---:|---:|---:|---:|---:|
| cities13 C | 0.30 | 6 | 0 | 1 | 0 |
| cities13 E | 0.30 | 2 | 0 | 1 | 2 |
| manual_gold C | 0.30 | 3 | 0 | 4 | 1 |
| manual_gold E | 0.30 | 5 | 0 | 0 | 1 |
| cities13 C | 0.55 | 4 | 0 | 0 | 0 |
| cities13 E | 0.55 | 0 | 0 | 0 | 0 |
| manual_gold C | 0.55 | 2 | 0 | 3 | 1 |
| manual_gold E | 0.55 | 3 | 0 | 0 | 0 |

Paired scores (wrap minus default; reported, never gating). `lo` removes `C`, `hi` removes `C` and `E`. A CI of [0, 0] with no bootstrap means no removed peak reaches t.

| unit | scen. | t | removed >= t | P def -> wrap | R def -> wrap | dF1 [95% CI] | GT lost / gained |
|---|---|---:|---:|---|---|---|---|
| cities11 | lo | 0.30 | 7 | 0.8624 -> 0.8640 | 0.7498 -> 0.7498 | +0.00069 [+0.00014, +0.00135] | 0 / 0 |
| cities11 | lo | 0.55 | 4 | 0.9424 -> 0.9432 | 0.6518 -> 0.6518 | +0.00029 [+0.00000, +0.00075] | 0 / 0 |
| cities11 | hi | 0.30 | 12 | 0.8624 -> 0.8649 | 0.7498 -> 0.7492 | +0.00069 [-0.00025, +0.00166] | 2 / 0 |
| cities11 | hi | 0.55 | 4 | 0.9424 -> 0.9432 | 0.6518 -> 0.6518 | +0.00029 [+0.00000, +0.00075] | 0 / 0 |
| cities13 | lo | 0.30 | 7 | 0.8594 -> 0.8608 | 0.7303 -> 0.7303 | +0.00058 [+0.00012, +0.00115] | 0 / 0 |
| cities13 | lo | 0.55 | 4 | 0.9376 -> 0.9384 | 0.6324 -> 0.6324 | +0.00024 [+0.00000, +0.00063] | 0 / 0 |
| cities13 | hi | 0.30 | 12 | 0.8594 -> 0.8615 | 0.7303 -> 0.7298 | +0.00057 [-0.00019, +0.00140] | 2 / 0 |
| cities13 | hi | 0.55 | 4 | 0.9376 -> 0.9384 | 0.6324 -> 0.6324 | +0.00024 [+0.00000, +0.00063] | 0 / 0 |
| manual_gold GT-A | lo | 0.30 | 8 | 0.8970 -> 0.8976 | 0.9112 -> 0.9099 | -0.00035 [-0.00108, +0.00039] | 5 / 0 |
| manual_gold GT-A | lo | 0.55 | 6 | 0.9474 -> 0.9478 | 0.8727 -> 0.8717 | -0.00034 [-0.00104, +0.00032] | 4 / 0 |
| manual_gold GT-A | hi | 0.30 | 14 | 0.8970 -> 0.8982 | 0.9112 -> 0.9092 | -0.00042 [-0.00133, +0.00053] | 8 / 0 |
| manual_gold GT-A | hi | 0.55 | 9 | 0.9474 -> 0.9481 | 0.8727 -> 0.8711 | -0.00051 [-0.00136, +0.00030] | 6 / 0 |
| manual_gold GT-B | lo | 0.30 | 8 | 0.8950 -> 0.8966 | 0.9115 -> 0.9112 | +0.00066 [+0.00006, +0.00135] | 1 / 0 |
| manual_gold GT-B | lo | 0.55 | 6 | 0.9457 -> 0.9470 | 0.8734 -> 0.8731 | +0.00046 [-0.00007, +0.00109] | 1 / 0 |
| manual_gold GT-B | hi | 0.30 | 14 | 0.8950 -> 0.8979 | 0.9115 -> 0.9112 | +0.00135 [+0.00055, +0.00219] | 1 / 0 |
| manual_gold GT-B | hi | 0.55 | 9 | 0.9457 -> 0.9478 | 0.8734 -> 0.8731 | +0.00082 [+0.00019, +0.00156] | 1 / 0 |
| annapolis | hi | 0.30 | 1 | 0.8947 -> 0.8981 | 0.8095 -> 0.8095 | +0.00152 [+0.00000, +0.00465] | 0 / 0 |
| bend | lo | 0.30 | 3 | 0.9155 -> 0.9186 | 0.8287 -> 0.8287 | +0.00140 [+0.00000, +0.00438] | 0 / 0 |
| bend | lo | 0.55 | 2 | 0.9727 -> 0.9727 | 0.7615 -> 0.7615 | +0.00000 [+0.00000, +0.00000] | 0 / 0 |
| bend | hi | 0.30 | 3 | 0.9155 -> 0.9186 | 0.8287 -> 0.8287 | +0.00140 [+0.00000, +0.00438] | 0 / 0 |
| bend | hi | 0.55 | 2 | 0.9727 -> 0.9727 | 0.7615 -> 0.7615 | +0.00000 [+0.00000, +0.00000] | 0 / 0 |
| budapest_district5 | lo | 0.30 | 1 | 0.6796 -> 0.6820 | 0.6433 -> 0.6433 | +0.00113 [+0.00000, +0.00379] | 0 / 0 |
| budapest_district5 | hi | 0.30 | 1 | 0.6796 -> 0.6820 | 0.6433 -> 0.6433 | +0.00113 [+0.00000, +0.00379] | 0 / 0 |
| morgantown | lo | 0.30 | 1 | 0.8785 -> 0.8821 | 0.8127 -> 0.8127 | +0.00165 [+0.00000, +0.00611] | 0 / 0 |
| morgantown | hi | 0.30 | 3 | 0.8785 -> 0.8852 | 0.8127 -> 0.8090 | +0.00104 [-0.00534, +0.00767] | 1 / 0 |
| paterson | lo | 0.30 | 1 | 0.9470 -> 0.9502 | 0.7241 -> 0.7241 | +0.00118 [+0.00000, +0.00379] | 0 / 0 |
| paterson | lo | 0.55 | 1 | 0.9677 -> 0.9712 | 0.6835 -> 0.6835 | +0.00119 [+0.00000, +0.00384] | 0 / 0 |
| paterson | hi | 0.30 | 2 | 0.9470 -> 0.9500 | 0.7241 -> 0.7215 | -0.00052 [-0.00508, +0.00345] | 1 / 0 |
| paterson | hi | 0.55 | 1 | 0.9677 -> 0.9712 | 0.6835 -> 0.6835 | +0.00119 [+0.00000, +0.00384] | 0 / 0 |
| richmond | lo | 0.30 | 1 | 0.8927 -> 0.8958 | 0.8323 -> 0.8323 | +0.00144 [+0.00000, +0.00469] | 0 / 0 |
| richmond | lo | 0.55 | 1 | 0.9560 -> 0.9598 | 0.7710 -> 0.7710 | +0.00153 [+0.00000, +0.00503] | 0 / 0 |
| richmond | hi | 0.30 | 2 | 0.8927 -> 0.8990 | 0.8323 -> 0.8323 | +0.00289 [+0.00000, +0.00755] | 0 / 0 |
| richmond | hi | 0.55 | 1 | 0.9560 -> 0.9598 | 0.7710 -> 0.7710 | +0.00153 [+0.00000, +0.00503] | 0 / 0 |

AP (from the 0.05 floor), wrap minus default: manual_gold GT-A lo -0.00140; manual_gold GT-A hi -0.00209; manual_gold GT-B lo +0.00009; manual_gold GT-B hi +0.00028; annapolis lo +0.00010; annapolis hi +0.00054; bend lo +0.00035; bend hi +0.00035; budapest_district5 lo +0.00065; budapest_district5 hi +0.00074; clovis lo +0.00041; clovis hi +0.00043; gainesville lo +0.00030; gainesville hi +0.00032; morgantown lo +0.00024; morgantown hi -0.00305; paterson lo +0.00040; paterson hi -0.00421; richmond lo +0.00089; richmond hi +0.00105; sao_paulo hi +0.00000; bayonne lo +0.00044; bayonne hi +0.00049; juneau lo +0.00005; juneau hi +0.00005 (every other split: 0).
