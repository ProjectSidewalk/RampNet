#!/usr/bin/env python3
"""Check the LR schedule that was ACTUALLY applied, across requeue boundaries (#135).

The 8-epoch cosine rung runs on klone's preemptible `ckpt-all`. `stage_two/train.py`'s
`lr_at_step` is stateless by design -- it computes the rate from the checkpointed
`global_step` rather than from a scheduler object -- precisely so a requeue cannot
restart the decay from the peak. But "by design" is not "verified", and the failure it
guards against is silent: a sawtooth schedule still completes the job, still writes
plausible loss curves, and only shows up as a run that answers a different question than
the one pre-registered.

So train.py logs the applied LR every step. This reads those scalars back and asks the
four questions that a mis-resume would fail:

1. **Does the merged series decrease monotonically?** A stateful scheduler restarting at
   the peak on each requeue is a run of non-decreasing steps at each boundary.
2. **Do incarnations agree where they overlap?** A resume rewinds to the last checkpoint
   and re-runs the few steps logged after it. The same step index must yield the same
   rate from both incarnations -- that is what "stateless" means, tested rather than
   asserted.
3. **What is the rate at each resume, against the peak?** Reported as a ratio, because
   `1.000000` is exactly the sawtooth signature and near the start of a cosine the
   absolute numbers differ by under a percent.
4. **Does every logged value match `lr_at_step` at that step?**

On (4): the function is *lifted from train.py*, not reimplemented, so this does **not**
independently check the formula -- `tests/test_train_lr_schedule.py` does that. What it
checks is that the formula was applied at the right **step index** after a resume, which
is the thing a requeue can actually break.

Note the deliberate off-by-one. train.py computes `lr_at_step(step_index)` for the step
it is about to take, then logs it under `current_total_step = step_index + 1`, so the
scalar at TensorBoard step N carries the rate for step N-1. Reading it without that
shift makes a correct run look wrong by one step's worth of decay.

TFRecord parsing is `stage2_train_cost.read_scalars` by way of
`stage2_epoch_curve.read_scalars_by_file`, in this directory -- standard library only, no
tensorboard install, one reader and one set of tests for all three scripts. The per-file
split is what makes incarnations visible at all; flattening first would hide (2) and (3).

Usage -- on klone, against the live run directory, after every requeue:

    python scripts/analysis/check_lr_schedule_135.py \
        --events-dir /gscratch/makelab/jonf/rampnet_cosine_rung_135/runs/experiment_1

Exit status is 0 only when every check passes, so it can be run unattended.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_epoch_curve import read_scalars_by_file  # noqa: E402

TRAIN_PY = REPO / "stage_two" / "train.py"

#: train.py logs the applied rate under this tag, once per optimizer step.
LR_TAG = "LR"

#: train.py logs `lr_at_step(n)` under TensorBoard step `n + 1`. See the module docstring.
LOG_STEP_OFFSET = 1

# The rung as pre-registered in docs/stage2_cosine_rung_135.md: 8 epochs x 9,378 steps at
# world size 16, cosine from 1e-5 to zero, no warmup.
RUNG_TOTAL_STEPS = 8 * 9378
RUNG_PEAK_LR = 1e-5

#: Scalars go into the event files as float32, so agreement is bounded by float32
#: precision (~6e-8 relative) and not by anything about the schedule. Expressed relative
#: to the peak rate: 1e-6 * 1e-5 = 1e-11, roughly twenty times the float32 floor.
REL_TOL = 1e-6


def load_schedule():
    """`lr_at_step` and `LR_SCHEDULES`, lifted from train.py without running it.

    train.py is a script, not a library -- importing it builds a model and opens a
    dataset -- so the two names are compiled out of the AST instead.
    `tests/test_train_lr_schedule.py` imports this same loader, so there is one lift
    rather than two copies to drift apart, and the suite exercises it either way.
    """
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"), filename=str(TRAIN_PY))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.FunctionDef) and n.name == "lr_at_step")
              or (isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "LR_SCHEDULES" for t in n.targets))]
    if len(wanted) != 2:
        raise SystemExit(f"{TRAIN_PY} no longer defines LR_SCHEDULES and lr_at_step")
    mod = types.ModuleType("train_lr")
    mod.math = math
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(TRAIN_PY), "exec"),
         mod.__dict__)
    return mod


def load_lr_at_step():
    """Just the schedule function -- see `load_schedule`."""
    return load_schedule().lr_at_step


def read_incarnations(events_dir: Path):
    """Return [(path, {tb_step: lr})] per event file, chronological.

    One event file per job incarnation: `SummaryWriter` opens a new one on each launch
    and their names sort by start time.
    """
    return [(path, points)
            for path, _digest, points in read_scalars_by_file(events_dir, LR_TAG)
            if points]


def check(events_dir: Path, total_steps: int, peak_lr: float,
          schedule: str, final_frac: float) -> int:
    lr_at_step = load_lr_at_step()
    incarnations = read_incarnations(events_dir)
    if not incarnations:
        raise SystemExit(f"no '{LR_TAG}' scalars in {events_dir} -- was the run launched "
                         f"with --lr-schedule {schedule}?")

    tol = REL_TOL * peak_lr
    failures: list[str] = []

    # ---- per incarnation ---------------------------------------------------------
    print(f"{len(incarnations)} incarnation(s) in {events_dir}\n")
    print(f"{'#':>2}  {'steps':>15}  {'n':>7}  {'first LR':>12}  {'last LR':>12}  file")
    for i, (path, points) in enumerate(incarnations, 1):
        steps = sorted(points)
        print(f"{i:>2}  {steps[0]:>7}-{steps[-1]:<7}  {len(steps):>7}  "
              f"{points[steps[0]]:>12.6e}  {points[steps[-1]]:>12.6e}  {path.name}")

    # ---- (3) the rate at each resume, against the peak ---------------------------
    print("\nResume boundaries (a stateful scheduler restarting at the peak reads 1.000000):")
    if len(incarnations) == 1:
        print("  none yet -- the run has not been requeued.")
    for i in range(1, len(incarnations)):
        prev_steps, cur_steps = sorted(incarnations[i - 1][1]), sorted(incarnations[i][1])
        first = cur_steps[0]
        lr = incarnations[i][1][first]
        ratio = lr / peak_lr
        overlap = sorted(set(prev_steps) & set(cur_steps))
        rewind = prev_steps[-1] - first + 1
        print(f"  {i} -> {i + 1}: resumed at step {first} with lr {lr:.6e} "
              f"({ratio:.6f} x peak); rewound {max(rewind, 0)} step(s), "
              f"{len(overlap)} overlapping")
        if ratio > 1.0 - REL_TOL:
            failures.append(
                f"incarnation {i + 1} resumed at {ratio:.6f} x peak -- the schedule "
                f"restarted rather than continuing")

        # ---- (2) incarnations must agree where they overlap ----------------------
        for step in overlap:
            a, b = incarnations[i - 1][1][step], incarnations[i][1][step]
            if abs(a - b) > tol:
                failures.append(
                    f"step {step} logged {a:.9e} by incarnation {i} and {b:.9e} by "
                    f"incarnation {i + 1} -- the rate is not a function of the step index")
                break

    # ---- merge: later incarnations win, as read_scalars does ---------------------
    merged: dict[int, float] = {}
    for _path, points in incarnations:
        merged.update(points)
    steps = sorted(merged)
    print(f"\nMerged: {len(steps)} unique steps, {steps[0]}-{steps[-1]} "
          f"of {total_steps} ({100.0 * steps[-1] / total_steps:.2f}% of the run)")

    # ---- (1) monotonicity --------------------------------------------------------
    violations = [(prev, cur) for prev, cur in zip(steps, steps[1:])
                  if merged[cur] > merged[prev] + tol]
    print(f"Non-decreasing violations: {len(violations)}"
          + (f"  (first at step {violations[0][1]})" if violations else ""))
    if violations:
        failures.append(f"{len(violations)} non-decreasing step(s) in the merged series -- "
                        f"first {violations[0][0]} -> {violations[0][1]}")

    # ---- (4) every value matches lr_at_step at that step -------------------------
    worst_step, worst_dev = None, 0.0
    for step in steps:
        expected = lr_at_step(step - LOG_STEP_OFFSET, total_steps, peak_lr,
                              schedule, final_frac)
        dev = abs(merged[step] - expected)
        if dev > worst_dev:
            worst_step, worst_dev = step, dev
    print(f"Max |logged - lr_at_step(step-{LOG_STEP_OFFSET})|: {worst_dev:.3e} "
          f"at step {worst_step}  (tolerance {tol:.1e})")
    if worst_dev > tol:
        # A whole-series shift is the off-by-one, not a broken schedule; say which.
        unshifted = max(abs(merged[s] - lr_at_step(s, total_steps, peak_lr,
                                                   schedule, final_frac)) for s in steps)
        hint = ("  (it matches with no offset -- train.py's logging step convention "
                "changed)" if unshifted <= tol else "")
        failures.append(f"logged rate deviates by {worst_dev:.3e} at step {worst_step}"
                        + hint)

    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("PASS -- the applied schedule is monotone, step-indexed, and continuous "
          "across every resume.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events-dir", type=Path, required=True,
                    help="directory of tfevents files (the run's runs/experiment_1)")
    ap.add_argument("--total-steps", type=int, default=RUNG_TOTAL_STEPS,
                    help=f"steps in the full run (default: {RUNG_TOTAL_STEPS}, the rung)")
    ap.add_argument("--peak-lr", type=float, default=RUNG_PEAK_LR,
                    help=f"peak learning rate (default: {RUNG_PEAK_LR})")
    ap.add_argument("--schedule", default="cosine", help="schedule under test")
    ap.add_argument("--lr-final-frac", type=float, default=0.0,
                    help="fraction of the peak the cosine lands on (default: 0.0)")
    args = ap.parse_args()
    return check(args.events_dir, args.total_steps, args.peak_lr,
                 args.schedule, args.lr_final_frac)


if __name__ == "__main__":
    sys.exit(main())
