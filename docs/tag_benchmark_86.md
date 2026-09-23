# Curb-ramp tag benchmark of record (#86, plan item 2)

DRAFT — numbers pending.

## Inputs

| input | identifier | hash |
|---|---|---|
| tagger code | `ProjectSidewalk/sidewalk-tagger-ai` @ `3b7405cd3206ece631cb7a65e22b1ab219df4b75` | git sha |
| test/train crops + CSVs | HF dataset `projectsidewalk/sidewalk-tagger-ai-validated` @ `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`, `Validated/CurbRamp.zip`, 30,792,993,257 bytes | LFS sha256 `5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d` |
| released checkpoint | HF model `projectsidewalk/sidewalk-tagger-ai-models` @ `65959dbc80b87e4f39385204c4b639cbcf58e1a8`, `validated-dino-cls-b-curbramp-tags-best.pth`, 347,207,242 bytes | LFS sha256 `4d00193aed73fc199049f31cebade51f236bfca92ad9f76adf08a9d08a272833` |
| DINOv2 backbone | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`, 346,393,545 bytes | sha256 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
