# Balanced visual/projector alignment diagnostic

26 October 2026. Protocol: `configs/vision_balanced_alignment_v1.json` and
`data/vision_balanced_alignment_20261026/prepared01/protocol.json` (SHA-256
`6a334e7337d0c0394c9a2237a1a510e20a0440d9895cf13e0f14ee9388ebaa5e`). This is a
bounded diagnostic, not a promotion or a replacement for the preserved rejected
stage result.

## Admission and data

The prepared set contains 185 newly rendered images and 192 procedural worlds.
Training has 28 answer and 18 caption examples; development has 27 answer and 12
caption cases; sealed held-out has 18 answer and 12 caption cases. Answer and
caption classes are balanced by construction. All 185 cases passed renderer-byte,
visible-pixel, visible-descriptor, occluded-object and answer-leakage checks.
Case/image/scene overlap with prior evaluation assets is zero. Evaluator labels
are separate from model-facing cases and held-out labels are read only after
extension export.

## Engineering acceptance

- Ten fused-AdamW updates, 60 scheduled examples, fresh initialization.
- Exactly six `vision_projector.*` tensors (11,542,016 parameters) are trainable;
  backbone and encoder remain frozen.
- The backward-only probe reaches all six projector tensors with finite,
  nonzero gradients and no non-projector gradients; it applies no update.
- One **65,693,060-byte** extension is exported with SHA-256
  `ef92cd3bfd942ba80a11243c3ae3b520fde6098608eb88e6269117665a7d19a2`.
  No base model or optimizer state is exported and no candidate is promoted.
- Fresh-process reload performs zero updates and reproduces held-out answers,
  held-out captions and retention outputs byte-for-byte. Independent `cmp`
  checks pass for all three files.

| Native process | Guard wall time | Peak sampled driver | Swap growth | Minimum host free |
|---|---:|---:|---:|---:|
| Train/evaluate | 259.02 s | 13,610,401,792 B | 0 B | 72% |
| Fresh reload | 106.83 s | 10,834,673,664 B | 0 B | 76% |

## Measured fit and controls

| Cohort | Correct answers | Caption token F1 | Generated descriptor semantic F1 |
|---|---:|---:|---:|
| Train fit | 11/28 = 39.29% | 0.17759 | 0.0 |
| Development | 12/27 = 44.44% | 0.18363 | 0.0 |
| Sealed held-out | 7/18 = 38.89% | 0.18176 | 0.0 |

Held-out answer controls produce 6/18 wrong, 5/18 counterfactual, 5/18 shuffled
and 5/18 no-image correct answers. The answerable-only counterfactual margin is
0.125 with a bootstrap lower bound of 0.0. These small margins do not establish
image competence.

Held-out caption counterfactual and shuffled loss margins are negative
(−0.11136 and −0.09276). The no-image loss margin is +2.32925, but generated
semantic descriptor F1 remains zero and its token-F1 margin is −0.11048.
Teacher-forced loss differences therefore do not demonstrate useful generated
caption alignment. The train-minus-held-out answer accuracy gap is only 0.00397;
this is not evidence of generalization.

## Disposition and retention

The overall development and held-out gate reports each fail 12 declared checks,
including correct accuracy, wrong/no-image margins, unknown handling, caption
F1/semantic F1 and caption dependence. **Learned quality is rejected; engineering
and reload acceptance pass.** Retention is unchanged at 22/32 correct (68.75%)
before and after training, not a strong-baseline or retention-quality claim.

Raw training traces, unsuccessful generations, controls, scores, extension and
reload evidence are under `runs/vision_balanced_alignment_20261026/train01/` and
`runs/vision_balanced_alignment_20261026/reload01/`. Guard/host/child logs are
under `reports/vision_balanced_alignment_20261026/`. Preparation and focused CPU
contracts are covered by `tests/test_vision_balanced_alignment.py`; the combined
fresh causal/visual/stage suite passed 22 tests before native execution. No
larger projector, backbone unfreezing or scaling follows from this diagnostic.
