# Independent staged-curriculum outcome review

**Reviewer:** `VisionCurriculumReview`, read-only final evidence audit.

## Verdict

**Engineering execution: PASS with provenance caveats. Learned quality: FAIL /
overall-ineligible.** The evidence chain documents a rejected curriculum, not a
qualified visual model.

## Evidence integrity

- Both arms share fresh seed-17 initialization, six projector tensors /
  11,542,016 parameters, 128 updates / 512 examples and 32 probe updates / 64
  examples, with identical schedule and no warm start. Base and encoder remain
  unchanged, response-only labels and projector-only gradients are present, and
  no base copy exists.
- Muon auxiliary AdamW remains at `1e-4`; its matrix LR is probe-selected at
  `3e-4`. AdamW uses `1e-4` for all six tensors. These are authorized by the
  declared two-candidate probe, not silent overrides.
- Both fresh-process reloads report zero updates and bitwise-equal dev and
  retention generations. Independent and run-local reload reports are identical.
- AdamW is selected by mean dev NLL: 0.797576 versus Muon 6.707518. This is a
  least-bad-by-loss selection, not a quality selection.
- Reservation files hash to `d92cbb38…`, matching `consumed.authenticated_by`.
  The selected extension `1d46970b…` matches run report, reload, comparison and
  receipt. The outer guard completes with child exit 0 and no stop reason.
- No raw metric arithmetic, protocol/data seal, optimizer ownership,
  initialization, extension hash, score/report link or reservation-payload
  discrepancies were found.

## Evidence gaps addressed in delivery

- The outer native launcher is not in the original 27-file implementation seal.
  `supplemental_lineage.json` binds that artifact and records the omission rather
  than rewriting the executed protocol.
- The completion log names the reservation as the receipt. The actual receipt is
  the same basename without `.reserve.json`; both final receipt files exist and
  are byte-identical.
- A repeat attempt stopped at the fresh-output check and does not itself prove
  seal refusal. A direct `require_global_finalize_allowed` check independently
  proves the consumed namespace is refused.
- Reload reports are not in the original artifact inventory. Independent and
  run-local copies are byte-identical; the supplemental lineage binds their
  hashes.

## Learned quality

| Evidence | Result | Gate outcome |
|---|---|---|
| Muon dev | Correct 1/240 = 0.42%; noimage 65/240 = 27.08%; margin −26.67 pp | All six fail |
| AdamW dev | Correct 99/240 = 41.25%; wrong 92/240; shuffled 87/240; noimage 65/240 | All six fail |
| AdamW audit | Correct 37/120 = 30.83%; wrong 38/120; shuffled 32/120; noimage 29/120 | All six fail |
| External final captions | Token F1 = 0 on all 8; five `yes`, three `no`; mean response loss 7.134 | Complete caption-alignment failure |
| Retention | 22/32 = 68.75%, drop 0, token parity true | Component nonregression passes; absolute baseline remains ineligible |

Dev margins are +2.92 pp wrong, +5.00 pp shuffled and +14.17 pp noimage, all
below 15 points. Audit margins are −0.83 pp wrong, +4.17 pp shuffled and +6.67
pp noimage, also below 15 points. Correct-control text matches wrong-image text
on 207/240 dev and 112/120 audit cases, consistent with label-prior behavior
rather than visual grounding. A per-task majority baseline scores 72/120 on
audit, nearly twice AdamW’s 37/120.

Abstention is decisively broken: dev unknown accuracy 2/24 = 8.33%, audit 1/24
= 4.17%, with false-answer-on-abstain rates 91.67% and 95.83%.

## Remaining unproven

- Genuine visual perception or caption capability.
- Muon final performance; Muon was evaluated only on dev and not selected for
  audit/external final.
- Vision-path retention: text-only token parity proves no text-output change,
  not preserved visual capability or satisfaction of the 75% floor.
- Optimizer-state evidence and collapse causality. No allocated post-step
  state-dtype measurement exists, so optimizer/schedule causality is unresolved.
- Generalization and reproducibility: one seed per arm, no variance estimate, no
  shape-task audit, and dev-to-audit accuracy drops from 41.25% to 30.83%.
- Capacity guarantees beyond sampled limits.

The receipts correctly retain `perception_quality=false`, `semantic_quality=false`
and `overall_baseline_eligible=false`. This is a valid engineering execution of
a rejected experiment, not a promotable result.
