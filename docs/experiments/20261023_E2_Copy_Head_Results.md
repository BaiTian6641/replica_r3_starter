# E2 record-token copy head: implemented interface, rejected candidate

23 October 2026 UTC. Protocol committed before execution. **Result: copy
interface and null/gradient paths pass; the trained copy head does not change
answers and is rejected.** Copy-enabled and disabled held-out exact accuracy are
both **0/16**; training loss never improved.

## 1. Mechanism

A new optional `CopyRead` frame carries exact token IDs from the selected source
records. `RecordCopyHead` computes native-hidden queries against frozen source
token embeddings and adds a learned logit bias for those exact token IDs. The
gate is `tanh(0)=0`, so the absent/disabled/untrained path is exactly null. The
head is independently gated and has no authority over records, retrieval or
success.

The copy head was trained alone; the base model and E2 writer/readers remained
frozen. The selection stayed lexical and the bank cap was 16 units. No answer or
expected value entered the prompt or copy candidate construction.

## 2. Arm and result

- Data: 24 train / 6 dev / 10 held-out novel-value cases; fresh source groups.
- 16 updates × 4 examples; Muon 8-bit + AdamW; lr 1e-2; answer-only loss.
- Extension v2, 53.1 MB; peak sampled driver 10.9 GiB; 41.4 s; no base copy.
- Final copy gate magnitude **0.0997**, so the gate did update.
- Training loss repeated roughly **14.4 / 40–41** across the update cycle, not a
  learning curve.
- Enabled and disabled answers were byte-identical: enabled `amber`/`blue`/etc.,
  disabled the same; **0/6 dev and 0/10 held-out** for both.

The relevant record was selected in every case and the answer token was present
in that record, so this is not an eligibility or retrieval failure. The copy
bias did not overcome the model's prior or learn a useful exact-token mapping.

## 3. Classification and decision

**Optimization/representation failure** for this additive logit-bias design:
the head receives gradients and the gate moves, but the loss does not descend.
The sparse source-token bias is too weak/awkward for this decoder and data
budget. Do not repeat the same arm. **Rejected**; no default or fallback changed.

The copy frame/head and its null/gradient/serialization tests remain implemented;
they are reusable for a future pointer or constrained-copy decoder, but that is a
different mechanism with a new held-out family. Exact text remains the preferred
path for novel values.

## 4. Evidence and costs

[Run report](../../reports/e2_copy_head_attempt02/run_report.json),
[manifest](../../reports/e2_copy_head_attempt02/extension_manifest.json), and the
loader-defect attempt at
[attempt 1](../../reports/e2_copy_head_attempt01/failure_report.json).

Costs: **41.4 s**, **10.9 GiB** sampled peak driver, one 53.1 MB extension, no
base copy, no download. Full suite: **1132 passed / 21 subtests**.
