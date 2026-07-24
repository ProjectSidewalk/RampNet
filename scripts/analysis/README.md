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
| `depth_extract_da3.py [n]` | **yes** | Metric depth for every GT ramp via Depth Anything 3 on the reprojected views → `gt_depth_da3.json`. |
| `depth_analysis.py` | no | Recall vs true distance / apparent size + the resolution forecast. Needs `gt_depth_da3.json`. |
| `size_analysis.py` | no | Geometry-only size stratification (no depth model) + the hard-miss montage figure. |
| `overlap_test.py` | **yes** | Do the threshold and resolution levers target the same ramps? Needs `gt_depth_da3.json`. |

The GPU scripts reproduce the deployment inference path exactly (resize 2048×4096 bilinear,
ImageNet norm, no TTA — see `sidewalk-auto-labeler/detectors/curb_ramp.py`), so
`threshold_sweep.py` at `(0.55, 10)` reproduces the committed `records.jsonl` detections.

Model weights load from the published HF artifact **by state_dict**, not `AutoModel`, because the
live artifact still hits issue #19 (`register_for_auto_class`) on transformers ≥ 5.13.

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
