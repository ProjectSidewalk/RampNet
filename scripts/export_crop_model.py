"""Package the Stage 1 crop-model checkpoints as a HuggingFace model repo.

The crop model is what converts a government curb ramp GPS coordinate into a pixel keypoint on a
panorama -- every label in `rampnet-dataset` was placed by it. `inference_isolator.py` loads the
round-2 checkpoint by a hardcoded relative path, so **Stage 1 cannot be reproduced without it**,
and until now it existed only on lab storage.

Both rounds are published, because round 1 is the initialisation for round 2.

This mirrors `scripts/export_hf_model.py` (the Stage 2 exporter): build a local package, verify it,
then optionally push. Nothing is uploaded without `--push`.

Build the package locally:

    python scripts/export_crop_model.py \
        --round1 <path>/ps_model/model/best_model.pth \
        --round2 <path>/ps_and_manual_model/best_model.pth \
        --out    dist/rampnet-crop-model

Push it (requires write access to the org):

    python scripts/export_crop_model.py --round1 ... --round2 ... \
        --out dist/rampnet-crop-model --push --repo-id projectsidewalk/rampnet-crop-model

`--expect-round1-sha256` / `--expect-round2-sha256` fail the build if the inputs are not the
checkpoints you meant to publish. The paper-era values are:

    round 1  00dba3948298a313435b7c1955a2d4fccde43bc98c199e384ef197bf8b8cff49
    round 2  3fc00ad6b9ac2768787b0262588b9bfa71ddd01d9f51109974e6ae377b9b520a
"""

import argparse
import datetime
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "hf_package" / "README.crop_model_card.template.md"

ROUND1_NAME = "round1_ps_best_model.pth"
ROUND2_NAME = "round2_ps_and_manual_best_model.pth"


def to_safetensors(src, dst):
    """Convert a torch checkpoint to safetensors, verifying every tensor round-trips exactly.

    The `.pth` files are `torch.save` pickle archives (they start `PK\\x03\\x04`), so loading one
    executes whatever is inside it. `projectsidewalk/rampnet-model` already ships safetensors for
    that reason; these should match. The `.pth` is kept alongside rather than replaced, because its
    sha256 is the provenance anchor tying this artifact to the paper's run -- converting is
    additive, not a migration.
    """
    import torch                                    # imported late: not needed for a hash-only run
    from safetensors.torch import load_file, save_file

    state = torch.load(str(src), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        sys.exit("error: {} is not a state dict ({})".format(src.name, type(state).__name__))
    if any(not torch.is_tensor(v) for v in state.values()):
        sys.exit("error: {} holds non-tensor entries; safetensors cannot carry them".format(src.name))

    # safetensors refuses shared storage, and .contiguous() is a no-op when already contiguous.
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(dst))

    restored = load_file(str(dst))
    if set(restored) != set(state):
        sys.exit("error: {} key set changed during conversion".format(src.name))
    for key, original in state.items():
        if original.dtype != restored[key].dtype or not torch.equal(original, restored[key]):
            sys.exit("error: tensor {!r} differs after conversion of {}".format(key, src.name))
    return len(state)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                # noqa: BLE001 - provenance is best-effort
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round1", required=True, type=Path, help="round-1 crop checkpoint")
    parser.add_argument("--round2", required=True, type=Path, help="round-2 crop checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="local package directory")
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-crop-model")
    parser.add_argument("--push", action="store_true", help="upload to the Hub (otherwise local only)")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    parser.add_argument("--expect-round1-sha256", default=None)
    parser.add_argument("--expect-round2-sha256", default=None)
    parser.add_argument("--no-safetensors", action="store_true",
                        help="skip the safetensors conversion (the .pth files are pickle archives)")
    args = parser.parse_args()

    for path in (args.round1, args.round2):
        if not path.is_file():
            sys.exit("error: not a file: {}".format(path))

    print("Hashing checkpoints")
    r1, r2 = sha256(args.round1), sha256(args.round2)
    print("  round 1  {:>13,} bytes  {}".format(args.round1.stat().st_size, r1))
    print("  round 2  {:>13,} bytes  {}".format(args.round2.stat().st_size, r2))

    for label, actual, expected in (("round1", r1, args.expect_round1_sha256),
                                    ("round2", r2, args.expect_round2_sha256)):
        if expected and actual != expected:
            sys.exit("error: {} sha256 {} != expected {}".format(label, actual, expected))
    if r1 == r2:
        sys.exit("error: both inputs hash the same -- round 2 is the ps_and_manual checkpoint, "
                 "not the copy of round 1 that lives beside it as ps_model.pth")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(args.round1), str(args.out / ROUND1_NAME))
    shutil.copy2(str(args.round2), str(args.out / ROUND2_NAME))

    card = TEMPLATE.read_text(encoding="utf-8").format(
        git_commit=git_commit(),
        round1_sha256=r1,
        round2_sha256=r2,
        export_date=datetime.date.today().isoformat(),
        repo_id=args.repo_id,
    )
    (args.out / "README.md").write_text(card, encoding="utf-8")

    print("\nPackage written to {}".format(args.out))
    for item in sorted(args.out.iterdir()):
        print("  {:<40} {:>13,}".format(item.name, item.stat().st_size))

    # Prove the copies survived, so a push cannot ship different bytes than were verified.
    for name, expected in ((ROUND1_NAME, r1), (ROUND2_NAME, r2)):
        if sha256(args.out / name) != expected:
            sys.exit("error: {} changed during copy".format(name))
    print("  (copies re-hashed, both identical to source)")

    if not args.no_safetensors:
        print("\nConverting to safetensors (every tensor compared after the round trip)")
        for name in (ROUND1_NAME, ROUND2_NAME):
            dst = (args.out / name).with_suffix(".safetensors")
            n = to_safetensors(args.out / name, dst)
            print("  {:<44} {:>13,}  {} tensors verified".format(
                dst.name, dst.stat().st_size, n))

    if not args.push:
        print("\nNot pushed. Re-run with --push --repo-id <org/name> to upload.")
        return

    from huggingface_hub import HfApi                # imported late: not needed for a local build
    api = HfApi()
    print("\nPushing to {}".format(args.repo_id))
    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(args.out),
                      commit_message="Add paper-era Stage 1 crop-model checkpoints (rounds 1 and 2)")
    print("Done: https://huggingface.co/{}".format(args.repo_id))


if __name__ == "__main__":
    main()
