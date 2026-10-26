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

## V2 learnability preparation (predeclared, CPU-only)

The v1 run above is preserved unchanged; it is not reinterpreted or promoted.
`configs/vision_balanced_alignment_v2.json` and the new
`tests/test_vision_caption_v2.py` replace the scoring/control contract, not the
historical receipt.

### Corrected caption contract

- The teacher-forced target is **the original sealed target for every image
  control**. Only the supplied image changes; v1 changed the target for
  counterfactual/shuffled controls, so its loss margins conflated target and
  image.
- Caption targets are derived from sealed labels and sealed control assignments,
  never from generation-record target fields. The scorer validates the receiver
  case, image identity/hash, target case, target text/descriptors, fixed loss
  finiteness and a per-case control-assignment SHA-256. Malformed rows and
  target/assignment tampering raise and preserve the raw rows in the failure
  path.
- Metrics are separated: generated correctness against the original target,
  following correctness against the changed image's sealed descriptors, and
  no-image behavior. A perfect image-follower is no longer required to
  reproduce the original description. `dependence` reports the original-target
  agreement drop separately from following accuracy.

### Corrected answer contract

Every v1 answer control, including counterfactual, was scored against the
receiver's original label; `counterfactual_answer` was unused. V2 keeps
original-label correctness but adds changed-image following accuracy against
the sibling's sealed answer, plus original-label margins (`change-only
dependence`) as a separate metric. It also reports donor-answer following for
wrong/shuffled controls when the donor task matches.

### Positive and negative controls

`verify_caption_control_contracts_v2` and
`verify_answer_control_contracts_v2` require a **perfect image-following
positive control to pass every intended gate**. Negative controls -
constant caption/answer, image-ignoring original description, shuffled donor,
malformed rows and target/assignment tampering - must fail the intended gate or
raise. The runner executes a data-free synthetic preflight before any training,
and preparation refuses to seal payloads unless the real dev/held-out controls
also pass.

### Learnability gates and access order

- A small training-only fit set (one answer per class, two captions per class),
  fresh v2 namespace/seed and a different schedule (24 updates, batch 4,
  lr 3e-4) avoid repeating the failed v1 schedule.
- Backbone and encoder remain frozen; exactly the six projector tensors train
  under `NativeFusedAdamW` (`adamw_fused`). Explicit token/template/mask/EOS/
  loss checks are declared in `fit_contract` and enforced on every update.
- Access is monotonic: `train -> train_fit -> dev -> audit`. Train-fit and dev
  gates are written before the audit reservation; if either fails, the sealed
  held-out role is never read and the run stops with
  `stopped_before_heldout_access`.
- A fresh-process reload must reproduce train-fit, dev, held-out, score, gate
  and retention artifacts bitwise before
  `completed_pending_reload` is resolved to `completed_reload_verified`;
  otherwise no terminal disposition is written.

The prepared v2 set is sealed under
`data/vision_balanced_alignment_v2_20261026/prepared04/` (protocol SHA-256
`3bf636f00dce4e73168eb5f4b2c2454560276b32dfb1122b91d58ffcd44c2d77`,
manifest SHA-256 `12f43b081e147a0c678e2137a321b709eb528099411b46b913735198563e0227`).
It has 14 answer plus 12 caption training examples, 18 answer plus 12 caption
development cases, and 18 answer plus 12 caption sealed held-out cases across
146 verified assets. The prepared positive controls passed every intended gate,
while the constant, image-ignoring, shuffled and malformed/tampering negatives
failed or raised as declared.

## V2 native train04 outcome

The first native v2 attempt, train02, reached development scoring and exposed a
runner contract bug: donor image identity was being used as `target_case_id`.
The target is now always bound to the original sealed case; the fix and a
runner regression are committed at `73f6cb64`. Train03 was an invocation-only
failure because the required fallback-disabled environment was omitted. Both
failure records and their guard logs are preserved.

Train04 used the corrected prepared04 protocol, frozen backbone/encoder and
six trainable projector tensors (11,542,016 parameters) for 24 fused-AdamW
updates. The backward-only probe reached all six projector tensors with finite,
nonzero gradients and no non-projector gradients. Token/template/mask/EOS and
response-loss contracts were enforced. A 62.6 MiB diagnostic extension was
exported, but no candidate was promoted.

| Cohort | Correct answers | Caption token F1 | Generated semantic F1 | Following accuracy |
|---|---:|---:|---:|---:|
| Train fit | 3/14 = 21.43% | 0.125 | 0.0 | 0.0 |
| Development | 2/18 = 11.11% | 0.0625 | 0.0 | 0.0 |

The synthetic scorer preflight passed its perfect-following positive control
and all declared negative controls. The actual train-fit gate failed
`min_train_fit_answer_accuracy` and `min_train_fit_caption_following_accuracy`;
the development gate failed accuracy, margin, unknown, caption and dependence
checks. The run therefore stopped at `stopped_before_audit` with
`diagnostic_only_no_promotion`. The sealed held-out role was never read, and
the fresh-process reload was not authorized or run. Retention remained 22/32
before and after training; this is not a retention-quality claim.

The guarded train04 process completed in 145.72 s with a 13,570,654,208-byte
sampled driver peak, zero swap growth and 72% minimum observed host free. Raw
generations, unsuccessful trajectories, scores, gates and the diagnostic
extension remain under `runs/vision_balanced_alignment_v2_20261026/train04/`.
Guard/child/host evidence is under
`reports/vision_balanced_alignment_v2_20261026/`.

The focused visual suites pass 27 tests, including the original-case target
identity regression. Engineering and scorer contracts pass; learned visual
quality is rejected and a fresh sealed held-out audit remains gated. No larger
projector, backbone unfreezing or scaling follows.
