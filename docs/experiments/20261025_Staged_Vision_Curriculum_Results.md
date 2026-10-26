# Staged vision curriculum: engineering pass, learned quality rejected

**25 October 2026.** Both optimizer arms completed training, strict extension
export, fresh-process reload and independent scoring. AdamW was selected by the
preregistered mean-dev-loss rule and finalized once. The result is a rejected
curriculum, not a promotable visual model.

Final evidence: [supplemental lineage](../../reports/vision_curriculum_20261025/supplemental_lineage.json),
[AdamW score](../../reports/vision_curriculum_20261025/adamw_score01/score_report.json),
[final receipt](../../reports/vision_curriculum_20261025/finalize_receipts/audit_593dc5406ebea4a09403ef597f81b513ad143025d3823ae0c3e9f45904d505d5.json),
and [independent outcome review](../../reports/vision_curriculum_20261025/independent_review.md).

## Optimizer comparison

Both arms used the same fresh seed-17, 4096-wide projector, six tensors /
11,542,016 parameters, 128 updates / 512 examples and 32 probe updates / 64
examples. Muon routed two matrices to Muon-8bit with auxiliary FP32 AdamW at
`1e-4`; its matrix LR probe selected `3e-4`. AdamW used AdamW for all six tensors
at `1e-4`. Base and encoder stayed unchanged, with no base copy.

| Measurement | Muon-8bit | AdamW control |
|---|---:|---:|
| Mean dev response loss | 6.707518 | 0.797576 |
| Dev correct /240 | 1 | 99 |
| Wrong /240 | 3 | 92 |
| Shuffled /240 | 2 | 87 |
| No-image /240 | 65 | 65 |
| Guard elapsed | 892.125 s | 750.851 s |
| Peak sampled driver | 13,610,844,160 B | 13,611,384,832 B |
| Extension bytes | 65,693,060 | 65,693,060 |

AdamW is selected by the declared least-dev-loss rule, not by a quality claim.
Muon’s collapse is unresolved: optimizer choice, the `3e-4` probe value,
schedule and their interaction remain possible causes. One seed per arm and no
allocated post-step optimizer-state measurement do not establish collapse
causality.

## Final quality

| Evidence | Correct | Wrong | Shuffled | No image | Gate outcome |
|---|---:|---:|---:|---:|---|
| AdamW dev | 99/240 = 41.25% | 92/240 = 38.33% | 87/240 = 36.25% | 65/240 = 27.08% | All six fail |
| AdamW audit | 37/120 = 30.83% | 38/120 = 31.67% | 32/120 = 26.67% | 29/120 = 24.17% | All six fail |

Dev margins are only +2.92 pp wrong, +5.00 pp shuffled and +14.17 pp no-image.
Audit margins are −0.83 pp wrong, +4.17 pp shuffled and +6.67 pp no-image. Every
margin is below the required 15 points. Correct-control text is identical to
wrong-image text on 207/240 dev cases and 112/120 audit cases, consistent with
label-prior/constant-output behavior rather than visual grounding. A per-task
majority baseline scores 72/120 on audit, nearly twice AdamW’s 37/120.

The final full suite passes **1,863 tests +21 subtests in 228.22 s**.
Abstention is broken: dev unknown accuracy is 2/24 = 8.33%, audit is 1/24 =
4.17%, with false-answer-on-abstain rates of 91.67% and 95.83%. External final
captions have token-F1 **0 on all 8** (five `yes`, three `no`; mean response loss
7.134), a complete caption-alignment failure.

Text retention is 22/32 = 68.75%, drop 0 and token parity true. Component
nonregression passes, but the absolute baseline remains ineligible. This does
not establish preserved visual capability or satisfaction of the 75% floor.

## Engineering and evidence integrity

Both arms completed 128 updates and exported one strict v4 extension each.
Fresh-process reload reports zero updates and bitwise-equal dev and retention
generations; independent and run-local reload reports are byte-identical. All 19
run checks are derived from evidence. Protocol/data seals, raw metrics,
reservation payloads and external-final seals validate.

The [supplemental lineage](../../reports/vision_curriculum_20261025/supplemental_lineage.json)
binds the outer host-guard artifact, run reports, reload reports, reservations
and final receipt. It records that the executed protocol’s original 27-file
inventory omitted `run_capacity_preflight.py` rather than rewriting history. A
separate direct seal check proves repeat finalization is refused. The child log
incorrectly names the `.reserve.json` reservation as the receipt; the actual
receipt is the same basename without `.reserve`, and both final receipts are
byte-identical.

Guard elapsed for finalization is 184.363 s. Outer-guard peak sampled driver is
11,815,763,968 bytes, zero swap growth and 73% minimum host free. These are
sampled safeguards, not OOM guarantees.

## Disposition

The staged curriculum is rejected. It does not establish genuine visual
perception, caption capability, optimizer-state collapse causality, generalization
or retained visual memory. Muon was evaluated only on dev and was not selected
for audit/external final. Shape remains train-only, and one seed per arm provides
no variance estimate. Next work should analyze constant-output priors and
abstention before any larger training arm; it should not enlarge the connector,
claim optimizer causality, or promote visual memory/RSI from these results.
