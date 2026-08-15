# Richmond box-annotation log

Progress snapshots from the single-day annotation session (jonf, 2026-08-14) that
produced `../boxes.json` — the whole-apron extent gold over all 310 adjudicated
Richmond ramps (#116). The **canonical, complete gold is `../boxes.json`**; these
snapshots exist for replicability, not for scoring:

1. They make the convergence claim checkable. `docs/crop_window_eval.md` (Round 2)
   reports that the rule ranking was stable as annotation grew — v1-norm whole-apron
   containment 0.295 → 0.263 → 0.260 at the 112 → 246 → 299 boxed checkpoints. Those
   interim numbers regenerate by pointing `scripts/analysis/crop_window_eval.py
   --bundle` at a bundle whose `boxes.json` is the corresponding snapshot.
2. Each export embeds the box-rule text that was live in the viewer when it was made,
   so together they document the rule's wording evolution during the session.

| file | items | box rule embedded | notes |
|---|---|---|---|
| `2026-08-14_003of310_play.json` | 3 boxed | v1, original paragraph form | First play-session export, made *before* the tight-extent discussion; 2 of its 3 boxes were subsequently revised. Superseded — kept only as the record of the v1 paragraph text and the session's starting point. |
| `2026-08-14_112of310.json` | 110 boxed + 2 can't | v1, restructured 6-bullet form (same convention, unversioned rewording — a discipline mistake not repeated after) | First real checkpoint; source of the interim table posted on #114. |
| `2026-08-14_250of310.json` | 246 boxed + 4 can't | v1, 6-bullet form (exported from a tab predating the v2 re-render) | Second checkpoint. |

The final `../boxes.json` (299 boxed + 11 can't-determine) embeds rule **v2** — v1's
six bullets plus two clarifying additions (box = measuring instrument, no road/context
inside; oblique ramps stay axis-aligned). v2 clarifies rather than changes the
convention, so boxes drawn under either text are one consistent set.

Nothing reads this directory programmatically. If a future re-annotation ever happens
(different annotator), start it blind: do **not** let the viewer prefill from any of
these files or from `../boxes.json`.
