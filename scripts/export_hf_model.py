"""Export a trained stage-2 checkpoint as a versioned HuggingFace model package.

Produces a local trust_remote_code package (config + modeling files + weights +
model card) that reproduces projectsidewalk/rampnet-model from this repo, and
optionally pushes it to the Hub. Run from the repo root.

Export a freshly trained checkpoint:

    python scripts/export_hf_model.py --checkpoint stage_two/checkpoints/epoch_1_step_9378.pth \
        --metrics-json stage_two/evaluation_results_new/metrics_manual_r0.022_pt0.55.json \
        --ap-json      stage_two/evaluation_results_new/metrics_manual_r0.022_pt0.0.json \
        [--push --repo-id projectsidewalk/rampnet-model]

Re-package the weights already on the Hub (issue #29 — same weights, fixed
wrapper + corrected card; no retrain). Tag the current Hub revision first so it
stays addressable, then:

    python scripts/export_hf_model.py --from-hub-revision <old-revision> \
        --source-fingerprint b0c3ff7a10fc --source-name epoch_1_step_9378.pth \
        --metrics-json stage_two/evaluation_results_new/metrics_manual_r0.022_pt0.55.json \
        --ap-json      stage_two/evaluation_results_new/metrics_manual_r0.022_pt0.0.json

Precision/recall come from the operating-threshold metrics JSON; Average
Precision must come from a full-sweep run (--ap-json, evaluate.py with
PEAK_THRESHOLD_ABS=0.0). Generate these with stage_two/evaluate.py (with TTA)
so the model card carries real gold-set numbers.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import torch

from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE
from rampnet.loading import load_checkpoint, checkpoint_fingerprint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "scripts", "hf_package")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a stage-2 checkpoint as a HuggingFace model package.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', help="Trained stage-2 checkpoint (.pth) to export")
    src.add_argument('--from-hub-revision',
                     help="Re-export the weights already on the Hub: download model.safetensors from "
                          "--repo-id at this revision and load them (strict) into the canonical "
                          "KeypointModel. Use for a re-packaging (e.g. issue #29) where the weights are "
                          "unchanged and only the wrapper/card are being fixed. Requires --source-fingerprint.")
    parser.add_argument('--source-fingerprint', default=None,
                        help="Override the checkpoint sha256 prefix recorded in the card. Required with "
                             "--from-hub-revision, where the local file hash is not the canonical "
                             "training-checkpoint identity (pass the original checkpoint's fingerprint).")
    parser.add_argument('--source-name', default=None,
                        help="Override the 'Source checkpoint' name shown in the card (default: the "
                             "checkpoint filename).")
    parser.add_argument('--output-dir', default='hf_export', help="Where to write the package (default: hf_export)")
    parser.add_argument('--metrics-json', default=None,
                        help="metrics_*.json (at the operating threshold) produced by stage_two/evaluate.py; "
                             "supplies the card's precision/recall")
    parser.add_argument('--ap-json', default=None,
                        help="Full-sweep metrics_*.json (evaluate.py with PEAK_THRESHOLD_ABS=0.0); supplies "
                             "the card's Average Precision. AP from a threshold-truncated run is wrong.")
    parser.add_argument('--dataset-revision', default='main',
                        help="Revision of projectsidewalk/rampnet-dataset the checkpoint was trained on")
    parser.add_argument('--recommended-threshold', type=float, default=0.55,
                        help="Operating threshold documented in the model card (default: 0.55)")
    parser.add_argument('--repo-id', default='projectsidewalk/rampnet-model',
                        help="Hub repo id used in the card's usage example and for --push")
    parser.add_argument('--push', action='store_true', help="Upload the package to the Hub after export")
    parser.add_argument('--skip-verify', action='store_true',
                        help="Skip the round-trip check that the exported package reproduces the checkpoint's output")
    return parser.parse_args()


def load_reference_model(args):
    """Build the canonical KeypointModel and load the weights to export into it.

    Returns ``(model, fingerprint, source_name)``. Two sources:

    * ``--checkpoint`` — a local ``.pth``, loaded strict; fingerprint is the file
      sha256 prefix (unless overridden by ``--source-fingerprint``).
    * ``--from-hub-revision`` — ``model.safetensors`` from ``--repo-id`` at that
      revision, loaded strict (the published weights use bare KeypointModel keys).
      The fingerprint is not derivable from the re-downloaded file, so
      ``--source-fingerprint`` is required and names the canonical training
      checkpoint the weights came from.
    """
    reference_model = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE)

    if args.from_hub_revision:
        if not args.source_fingerprint:
            raise SystemExit("--from-hub-revision requires --source-fingerprint "
                             "(the canonical training-checkpoint sha256 prefix).")
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        print(f"Downloading weights from {args.repo_id}@{args.from_hub_revision} (model.safetensors)...")
        weights_path = hf_hub_download(args.repo_id, "model.safetensors", revision=args.from_hub_revision)
        state_dict = load_file(weights_path)
        reference_model.load_state_dict(state_dict, strict=True)
        fingerprint = args.source_fingerprint
        source_name = args.source_name or f"weights from {args.repo_id}@{args.from_hub_revision}"
    else:
        print(f"Loading checkpoint {args.checkpoint} (strict)...")
        load_checkpoint(reference_model, args.checkpoint, map_location='cpu')
        fingerprint = args.source_fingerprint or checkpoint_fingerprint(args.checkpoint)
        source_name = args.source_name or os.path.basename(args.checkpoint)

    reference_model.eval()
    return reference_model, fingerprint, source_name


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def render_eval_section(metrics_json_path, ap_json_path=None):
    """Render the model-card metrics table.

    Precision/recall are read at the operating threshold from ``metrics_json_path``
    (a ``stage_two/evaluate.py`` run with ``PEAK_THRESHOLD_ABS`` set to that
    threshold). Average Precision must come from a **full confidence sweep**
    (``PEAK_THRESHOLD_ABS = 0.0``), so pass that run separately as
    ``ap_json_path``; an AP taken from a threshold-truncated run integrates only
    the tail of the curve and reads far too low. If ``ap_json_path`` is omitted,
    AP falls back to ``metrics_json_path`` (correct only when it is itself a
    full-sweep run).
    """
    if metrics_json_path is None:
        return ("*No evaluation metrics were supplied at export time. Run `stage_two/evaluate.py "
                "--threshold <t>` and re-export with `--metrics-json` to fill this in.*")
    with open(metrics_json_path) as f:
        m = json.load(f)
    ap_source = m
    if ap_json_path is not None:
        with open(ap_json_path) as f:
            ap_source = json.load(f)
    if ap_source.get('peak_threshold_abs', 0.0) not in (0.0, 0):
        raise ValueError(
            f"AP must come from a full-sweep run (peak_threshold_abs=0.0); got "
            f"{ap_source.get('peak_threshold_abs')} in {ap_json_path or metrics_json_path}. "
            "Pass the pt0.0 metrics file via --ap-json.")
    lines = [
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Average Precision (interpolated, full sweep) | {ap_source['ap']:.4f} |",
        f"| Precision @ threshold {m['peak_threshold_abs']} | {m['precision_at_threshold']:.4f} |",
        f"| Recall @ threshold {m['peak_threshold_abs']} | {m['recall_at_threshold']:.4f} |",
        f"| Ground-truth points | {m['total_gt_points']} |",
        f"| Matching radius (normalized) | {m['radius_threshold_normalized']} |",
        f"| Flip TTA | {'on' if m.get('tta', True) else 'off'} |",
    ]
    return "\n".join(lines)


def assemble_package(output_dir, reference_model, recommended_threshold=0.55):
    """Write a loadable trust_remote_code package (config + modeling files +
    weights) to ``output_dir``, copying ``reference_model``'s weights into the
    HF wrapper layout.

    Shared by ``main()`` and ``tests/test_hf_load.py`` so the exporter and the
    load smoke test exercise the identical package layout — the one that
    ``AutoModel.from_pretrained(..., trust_remote_code=True)`` must accept on the
    transformers versions we support (see issue #19).
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    # Imported as a package so the relative imports match how the Hub's
    # dynamic-module loader will resolve them.
    from hf_package.configuration_rampnet import RampNetConfig  # noqa: E402

    os.makedirs(output_dir, exist_ok=True)

    # The modeling code shipped to the Hub imports the architecture from
    # rampnet_model.py, copied verbatim from the canonical rampnet/model.py so
    # the Hub package can never fork the architecture.
    shutil.copy(os.path.join(REPO_ROOT, "rampnet", "model.py"),
                os.path.join(output_dir, "rampnet_model.py"))
    for fname in ("configuration_rampnet.py", "modeling_rampnet.py"):
        shutil.copy(os.path.join(PACKAGE_DIR, fname), os.path.join(output_dir, fname))

    config = RampNetConfig(recommended_threshold=recommended_threshold)
    config.auto_map = {
        "AutoConfig": "configuration_rampnet.RampNetConfig",
        "AutoModel": "modeling_rampnet.RampNetModel",
    }
    config.save_pretrained(output_dir)

    # Wrap the weights in the HF module layout (prefix 'model.') and save.
    from transformers import AutoConfig, AutoModel
    hf_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    hf_model = AutoModel.from_config(hf_config, trust_remote_code=True)
    hf_model.model.load_state_dict(reference_model.state_dict())
    hf_model.save_pretrained(output_dir)


def verify_roundtrip(output_dir, reference_model):
    from transformers import AutoModel
    exported = AutoModel.from_pretrained(output_dir, trust_remote_code=True).eval()
    x = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        expected = reference_model(x)
        actual = exported(x)
    if not torch.allclose(expected, actual, atol=1e-6):
        raise RuntimeError("Exported package output does not match the source checkpoint's output.")
    print("Round-trip verification passed: exported package reproduces the checkpoint's output.")


def main():
    args = parse_args()

    reference_model, fingerprint, source_name = load_reference_model(args)

    assemble_package(args.output_dir, reference_model,
                     recommended_threshold=args.recommended_threshold)
    print(f"Saved weights and config to {args.output_dir}")

    with open(os.path.join(PACKAGE_DIR, "README.model_card.template.md"), encoding='utf-8') as f:
        template = f.read()
    card = template.format(
        git_commit=git_commit(),
        checkpoint_name=source_name,
        checkpoint_fingerprint=fingerprint,
        dataset_revision=args.dataset_revision,
        export_date=datetime.date.today().isoformat(),
        eval_section=render_eval_section(args.metrics_json, args.ap_json),
        recommended_threshold=args.recommended_threshold,
        repo_id=args.repo_id,
    )
    with open(os.path.join(args.output_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(card)
    print("Wrote model card README.md")

    if not args.skip_verify:
        verify_roundtrip(args.output_dir, reference_model)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type='model', exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.repo_id, repo_type='model',
                          commit_message=f"Export {source_name} "
                                         f"(sha256 {fingerprint}) from commit {git_commit()}")
        print(f"Pushed to https://huggingface.co/{args.repo_id}")
    else:
        print("Dry run complete (no --push). Inspect the package, then re-run with --push to publish.")


if __name__ == "__main__":
    main()
