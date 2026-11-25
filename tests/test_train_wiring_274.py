"""stage_two/train.py's script-level wiring for #274: --trim-epoch-to-accum reaching the sampler, and the
multi-epoch --grad-accum refusal. Both live in the script body, not in a function, so they are lifted out of the
AST and run against stand-ins; nothing builds a model. Every s1 to s5 run (1,000 optimizer steps over ~4.4
epochs) relied on the relaxed refusal. CPU only."""
import ast
import os
import sys
from types import SimpleNamespace as NS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "stage_two", "train.py")
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402


def _sampler_multiple(grad_accum, trim):
    """What `multiple` train.py's `train_sampler = ResumeSkipSampler(...)` passes, for these flags."""
    mod = load_from_train_py(
        "train_sampler", ResumeSkipSampler=lambda base, multiple: multiple,
        DistributedSampler=lambda *a, **k: None, sampler_seed_for=lambda s: 0, train_dataset=[],
        world_size=1, rank=0, args=NS(grad_accum=grad_accum, trim_epoch_to_accum=trim, seed=42))
    return mod.train_sampler


def test_trim_flag_reaches_the_sampler():
    assert _sampler_multiple(4, True) == 4
    assert _sampler_multiple(4, False) == 1     # off: the paper recipe's untrimmed epoch
    assert _sampler_multiple(1, True) == 1


def _refuses():
    """The top-level `if` that ends in `raise SystemExit(2)` after checking args.grad_accum, as a function of
    (grad_accum, max_micro_steps, per-rank epoch length)."""
    tree = ast.parse(open(TRAIN, encoding="utf-8").read(), filename=TRAIN)
    hits = []
    for node in tree.body:
        if not isinstance(node, ast.If) or "args.grad_accum" not in ast.unparse(node.test):
            continue
        raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise) and "SystemExit(2)" in ast.unparse(n)]
        if raises:
            hits.append(node)
    assert len(hits) == 1, f"expected one grad-accum refusal in train.py, found {len(hits)}"
    code = compile(ast.Expression(hits[0].test), TRAIN, "eval")
    return lambda ga, max_micro, epoch_len: bool(eval(code, {
        "args": NS(grad_accum=ga), "max_micro_steps": max_micro, "train_sampler": NS(epoch_length=epoch_len)}))


def test_multi_epoch_grad_accum_refusal_in_both_directions():
    refuses = _refuses()
    assert not refuses(1, None, 1001)            # no accumulation: never refused
    assert refuses(4, None, 1000)                # no --max-steps: refused
    assert not refuses(4, 400, 1001)             # ends inside the first epoch: allowed
    assert not refuses(4, 4000, 1000)            # multi-epoch, epoch length a multiple of 4: allowed (s1 to s5)
    assert refuses(4, 4000, 1001)                # multi-epoch, a group would straddle an epoch: refused
