# The 2026-09-15 YOLO sweeps, scored on the wrong epoch checkpoints

These are the `yolo/*.txt` files PR #161 was first scored from. The leg labels inside them
say `y11x_tiles_s1_ep44`, `s2_ep44`, `s3_ep42`, but the checkpoints behind those labels
were Ultralytics' 0-based `epoch44.pt` / `epoch44.pt` / `epoch42.pt` — i.e. `results.csv`
epochs **45 / 45 / 43**, one epoch later than `pick_epoch` chose. See
`docs/seed_variance_51_135.md` (Results, 2026-09-17 correction) and
`scripts/model_comparison/yolo_baseline/check_epoch_ckpt.py` for the mechanism and the guard.

Kept, not deleted, because they are a scored measurement of a stated checkpoint
(sha256s in `../env_2026-09-15.txt`) and the read script's tests pin that the corrected
files differ from them. The three `*_best` legs in these files are byte-identical in
content to the corrected run's (0 of 278 sweep cells differ), which is the control that
the re-score changed only the checkpoints.

Not read by `seed_variance_read_51_135.py`; the read uses `../yolo/`.
