"""Export a trained stage-2 checkpoint as a versioned HuggingFace model package.

Produces a local trust_remote_code package (config + modeling files + weights +
model card) that reproduces projectsidewalk/rampnet-model from this repo, and
optionally pushes it to the Hub. Run from the repo root:

    python scripts/export_hf_model.py --checkpoint stage_two/checkpoints/epoch_1_step_9378.pth \
        --metrics-json stage_two/evaluation_results/metrics_manual_r0.022_pt0.55.json \
        [--push --repo-id projectsidewalk/rampnet-model]

Generate the metrics JSON first with stage_two/evaluate.py (at the recommended
operating threshold, with TTA) so the model card carries real gold-set numbers.
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
    parser.add_argument('--checkpoint', required=True, help="Trained stage-2 checkpoint (.pth)")
    parser.add_argument('--output-dir', default='hf_export', help="Where to write the package (default: hf_export)")
    parser.add_argument('--metrics-json', default=None,
                        help="metrics_*.json produced by stage_two/evaluate.py; embedded in the model card")
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


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def render_eval_section(metrics_json_path):
    if metrics_json_path is None:
        return ("*No evaluation metrics were supplied at export time. Run `stage_two/evaluate.py "
                "--threshold <t>` and re-export with `--metrics-json` to fill this in.*")
    with open(metrics_json_path) as f:
        m = json.load(f)
    lines = [
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Average Precision (interpolated) | {m['ap']:.4f} |",
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

    print(f"Loading checkpoint {args.checkpoint} (strict)...")
    reference_model = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE)
    load_checkpoint(reference_model, args.checkpoint, map_location='cpu')
    reference_model.eval()
    fingerprint = checkpoint_fingerprint(args.checkpoint)

    assemble_package(args.output_dir, reference_model,
                     recommended_threshold=args.recommended_threshold)
    print(f"Saved weights and config to {args.output_dir}")

    with open(os.path.join(PACKAGE_DIR, "README.model_card.template.md"), encoding='utf-8') as f:
        template = f.read()
    card = template.format(
        git_commit=git_commit(),
        checkpoint_name=os.path.basename(args.checkpoint),
        checkpoint_fingerprint=fingerprint,
        dataset_revision=args.dataset_revision,
        export_date=datetime.date.today().isoformat(),
        eval_section=render_eval_section(args.metrics_json),
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
                          commit_message=f"Export {os.path.basename(args.checkpoint)} "
                                         f"(sha256 {fingerprint}) from commit {git_commit()}")
        print(f"Pushed to https://huggingface.co/{args.repo_id}")
    else:
        print("Dry run complete (no --push). Inspect the package, then re-run with --push to publish.")


if __name__ == "__main__":
    main()
