# Fused continuation fixture comparison

**26 October 2026 · completed fixture diagnostic · exact continuation rejected.**
The [summary receipt](../../reports/fused_continuation_fixture_v1/summary.json)
and [comparison receipt](../../reports/fused_continuation_fixture_v1/comparison_receipt.json)
record four serial native arms against one protocol and implementation. This is
restore-mechanics evidence, not production MiniCPM5 full-scope qualification.

## Arms and checks

Two independent uninterrupted controls completed two updates each. The first
process completed update 1 and wrote the single fixture checkpoint; the fresh
resume loaded it directly on MPS and completed update 2. All four guards
completed with zero swap growth and at least 89% observed host-free memory.
The maximum sampled driver peak was 1,183,580,160 bytes. The checkpoint is about
202 MB and the fixture artifact set about 1.7 MB, below the 2.5 GB cap. No final
model export or duplicate base save exists.

The comparator verified identical protocol and implementation, identical initial
weights, matching first/resume start state, a present and identical boundary
fingerprint, direct-target-device restore, zero staged CPU optimizer bytes,
positive restored optimizer state and no full CPU model copy. It also checked
all 21 tensors, optimizer step 2, data cursor, losses, rates and processed/supervised
counts. These mechanical checks pass.

## Result

Exact final parameter equality fails for `base.model.embed_tokens.weight`.
Output loss and logits match exactly, but exact parameter equality is the
declared gate and cannot be replaced by matching output fingerprints. The
preregistered `aa-envelope-v1` numerical interpretation is also exceeded at the
final fingerprint. Therefore:

- `exact_status = failed`;
- `variation_status = divergent`;
- no numerical-equivalence label is permitted;
- no long training or retry is authorized.

The A/A controls show why two independent uninterrupted runs were required: a
single resume run cannot establish a numerical-variation baseline. Envelope
evidence cannot convert an exact failure to a pass.

## Preserved failure taxonomy

Historical numerical mismatches, constructor/device defects, model
materialization swap stops, staged optimizer restore swap stops, the direct
`torch.load` swap stop, and historical A/A implementation drift remain distinct.
The stopped 381-tensor production resume was not retried. This small fixture
cannot convert that production result to passed.

Stop here. A new experiment requires a separately justified mechanism and
resource hypothesis, not another blind resume.
