# Recall error-analysis scripts

The analysis behind [`docs/detection_recall_analysis.md`](../../docs/detection_recall_analysis.md).
Everything here reads the committed benchmark bundles (`benchmark/{richmond,bend}/`) plus the
native-res `panos/` (git-ignored — they must be present locally).

Outputs go to `$RAMPNET_ANALYSIS_OUT` (default `analysis_out/`, git-ignored).

## Run order

| script | GPU | what it answers |
|---|---|---|
| `miss_analysis.py` | no | Are misses localization near-misses or blind? Are they hard (a VLM also missed) or RampNet-specific? |
| `complementarity.py [model]` | no | Oracle-union recall + the RampNet-miss ∩ VLM-hit set (issue #35 gate). Reads cached VLM detections from `.model_cache`. |
| `precision_by_distance.py` | no | Is precision worse at distance — i.e. is culling far detections worth it? (No.) |
| `threshold_sweep.py` | **yes** | Re-runs inference on all benchmark panos and sweeps `threshold_abs` × `min_distance`. |
| `peak_nms_check.py` | no | Would suppressing peaks closer than the match radius help? (No — 5 of the 8 within-R pairs in the reviewed records are real ramp pairs; issue #62.) Reads all six splits' committed records, no panos needed. |
| `depth_extract_da3.py [n]` | **yes** | Metric depth for every GT ramp via Depth Anything 3 on the reprojected views → `gt_depth_da3.json`. |
| `depth_analysis.py` | no | Recall vs true distance / apparent size + the resolution forecast. Needs `gt_depth_da3.json`. |
| `size_analysis.py` | no | Geometry-only size stratification (no depth model) + the hard-miss montage figure. |
| `overlap_test.py` | **yes** | Do the threshold and resolution levers target the same ramps? Needs `gt_depth_da3.json`. |
| `operating_point_curve.py extract` | **yes** | Inference once → all peaks down to a low score floor → per-pano cache (issue #54). |
| `operating_point_curve.py curve` | no | Continuous PR curve + honest AP + F1-vs-threshold from the cache (#54). |
| `operating_point_curve.py gallery` | no | Incremental-FP crops for the GT-completeness spot-check → corrected precision with an error band (#54). |

The GPU scripts reproduce the deployment inference path exactly (resize 2048×4096 bilinear,
ImageNet norm, no TTA — see `sidewalk-auto-labeler/detectors/curb_ramp.py`), so
`threshold_sweep.py` at `(0.55, 10)` reproduces the committed `records.jsonl` detections.

`operating_point_curve.py` is the issue #54 operating-point analysis: unlike `threshold_sweep.py`
(which re-extracts peaks per discrete threshold), it extracts once at a low floor and carries each
peak's height as its confidence, so a single inference pass yields the whole continuous curve + AP.
Its `curve`/`gallery` steps are CPU-only and read the cache `extract` writes.

Model weights load from the published HF artifact **by state_dict**, matching the deployment
inference path (not `AutoModel`); the pure scoring logic lives in `rampnet/detection_eval.py` and
`rampnet/metrics.py` and is unit-tested in `tests/test_operating_point_curve.py`.

## Depth Anything 3 setup

`depth_extract_da3.py` needs DA3, which ships its own package (not `transformers`):

```bash
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git
export DA3_SRC=$PWD/Depth-Anything-3/src        # imported from src/, NOT pip-installed
pip install omegaconf einops addict opencv-python-headless plyfile pycolmap trimesh evo
```

Deliberately **skip** their `numpy<2`, `xformers` and `open3d` pins — they are unnecessary for
inference and will churn a working CUDA env. One import (`moviepy`, used only by the Gaussian-splat
video export) must be stubbed; create an empty `stubs/moviepy/__init__.py` + `editor.py` next to
`$DA3_SRC/..` and it is picked up automatically.

**Critical:** pass the *known* intrinsics. We synthesise the rectilinear views, so
`focal = (W/2) / tan(fov_h/2)` exactly (512 px for the default 90° FOV, 1024 px views). With
intrinsics supplied, `prediction.depth` is **already in metres** — do *not* apply the
`× focal / 300` formula from the DA3 README, which is for the no-intrinsics path and over-corrects
by ~1.65×. Intrinsics-naive models (e.g. Depth-Anything-V2 metric) come out ~3× long on these
wide-FOV views.
