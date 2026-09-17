#!/usr/bin/env python
"""Assert that a renamed Ultralytics epoch checkpoint is the epoch its name claims.

Ultralytics numbers ``results.csv`` from 1 (``trainer.py`` writes ``self.epoch + 1``)
but names the per-epoch weights from 0 (``epoch{self.epoch}.pt``, written before the
increment) and stores that 0-based value as ``ckpt["epoch"]``. So ``epoch44.pt`` is the
model after ``results.csv`` row **45**. On 2026-09-15 the seed-variance scoring
(``docs/seed_variance_51_135.md``) picked ``results.csv`` epochs 44/44/42 and then copied
``epoch44/44/42.pt`` -- one epoch late, two of three outside the pre-registered window --
and nothing failed, because the leg label is just the file stem. This check makes that
mismatch fail loudly before a leg is scored.

A leg is named ``<arm>_s<seed>_ep<N>.pt`` where N is the 1-based ``results.csv`` epoch.
The checkpoint it holds must satisfy ``ckpt["epoch"] + 1 == N`` and carry N rows of
``train_results`` (the checkpoint stores its own copy of ``results.csv`` up to that
epoch, so the check needs no external file). Legs without ``_ep<N>`` in the stem
(``*_best.pt``) are reported and skipped: ``best.pt`` has ``epoch == -1``.

Usage:
    python check_epoch_ckpt.py seedvar_ckpts/y11x_tiles_s1_ep44.pt [more.pt ...]
Exit 0 if every ``_ep<N>`` file matches, 1 otherwise.
"""
import argparse
import os
import re
import sys

EP_RE = re.compile(r"_ep(\d+)$")


def claimed_epoch(path):
    """The 1-based epoch a leg's stem claims, or None if the stem carries none."""
    m = EP_RE.search(os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def check_epoch(ckpt, claimed):
    """(ok, detail) for a loaded Ultralytics checkpoint dict against ``claimed``.

    ``ckpt["epoch"]`` is 0-based; ``train_results["epoch"]`` is the 1-based column and
    has one row per finished epoch, so both must equal ``claimed`` after the +1.
    """
    stored = ckpt.get("epoch")
    rows = ckpt.get("train_results") or {}
    n_rows = len(rows.get("epoch", [])) if isinstance(rows, dict) else None
    ok = stored is not None and stored + 1 == claimed and n_rows == claimed
    detail = (f"ckpt.epoch={stored} (+1 = {None if stored is None else stored + 1}), "
              f"train_results rows={n_rows}, claimed={claimed}")
    return ok, detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args(argv)

    import torch  # deferred: the parse and the regex are testable without it

    bad = 0
    for path in args.checkpoints:
        claimed = claimed_epoch(path)
        if claimed is None:
            print(f"{path}: no _ep<N> in stem, skipped")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ok, detail = check_epoch(ckpt, claimed)
        print(f"{path}: {'OK' if ok else 'MISMATCH'} -- {detail}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
