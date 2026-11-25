# laurens_gsv imagery: re-pinned 2026-11-12

`imagery_manifest.json` in this directory no longer pins the bytes the human review was made
against. Those files are lost. It now pins a later fetch of the same 86 panoramas.

## What happened

On 2026-11-11 a `git worktree remove --force` on Windows deleted through NTFS junctions and
emptied `benchmark/<split>/panos` for 14 splits in the main checkout. Thirteen splits were
restored from copies whose sha256 matched the committed manifests. For `laurens_gsv` no
byte-identical copy was found:

| Place checked | Result |
|---|---|
| Hugging Face `projectsidewalk/rampnet-benchmark` | no laurens split |
| Windows desktop (the deleted copy; SSD with TRIM, no shadow copies) | unrecoverable |
| klone | no copy |
| Jon's Mac (searched 2026-11-12) | no copy |
| makelab2 `sidewalk-auto-labeler/runs/laurens_gsv/panos` | same 86 pano ids, different bytes |

## What the manifest pins now

The makelab2 archive copy: the same 86 pano ids at 16384x8192, fetched separately from the
reviewed files.

- All 86 sha256 values differ from the original manifest (digest `2f079d63d4af4063`, now
  `7b18594217554429`).
- File sizes differ by at most 0.02% per pano.
- Mean absolute thumbnail difference against the reviewed gallery `_full.jpg` images is 2.42.
  The same measure on paterson, whose bytes are exact, is 2.17.

The original hashes are in git history: the version of `imagery_manifest.json` before this
commit.

## What this changes

- **Published numbers: nothing.** Every scored leg replays cached model output; none re-reads
  pixels.
- **Verdicts:** they were made on the lost files. They describe the same scenes, but the claim
  "every reviewed panorama matches the bytes recorded at review time" is no longer true for
  this split, whatever `imagery_manifest.py --verify` prints.
- **New legs on this split** see pixels that differ slightly from what earlier legs saw. Say
  so when comparing a new leg with an old one on `laurens_gsv`.

If the original files ever turn up, restore the earlier manifest and delete this note.
