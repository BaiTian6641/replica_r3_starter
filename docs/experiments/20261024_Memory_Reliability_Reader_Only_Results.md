# Exact-memory reference and reader-only MiniCPM5 D1

24 October 2026. **Final decision: reject the learned candidate and reject the
reference bundle on the declared full-context parity gate.** Selective exact
context is correct on every case, but the full-context control fails two absent-
fact answers. This is not useful learned memory and is not an RSI admission.

## 1. Package and protocol

The completed comparison used
[protocol v4](../../configs/memory_reliability_v4.json), committed at `9a43bad`.
The source is sealed MiniCPM5 revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`; the M3 shared extension is an
ownership reference with failed retention, not an accepted general-memory
baseline. The rejected v1 candidate and all earlier reports remain preserved.

The single discriminating change was **reader-only D1 training**. Eight tensors
at fast ports 13 and 27 (`query_norm`, `query_projection`, `output_projection`)
train; writers, both gate scalars, record/memory/workspace components and the
original 2,516,756,480 backbone parameters remain frozen. Exactly these eight
tensors changed; 59 other extension tensors are unchanged. The trained scope is
**2,105,344 parameters**. The larger export-time trainable count is flag state,
not the trained set.

- 48 updates × 4 examples = 192 unique training cases across 8 training families.
- Muon int8 momentum with auxiliary FP32 AdamW; FP32 parameters/gradients.
- No general preservation KL: general/unknown cases are gate-off by design and
  have no differentiable fast path. Retention is enforced by the fixed gate and
  disabled-token controls rather than a no-op student/teacher loss.
- Fresh data: 32 episodes across 16 complete counterfactual families, four query
  forms, both variants, randomized order, independent distractors, absent facts,
  and heldout-only observed values. Train/dev distractor values bronze and silver
  are shared, so this is **heldout full observed-value novelty**, not dev novelty.
- Parent and candidate generation: **1,920 answers**, 4,328 generated tokens and
  134,076 prompt tokens; maximum 4 generated and 143 prompt tokens per answer.
- One isolated 48,948,272-byte candidate extension, no base copy or download.

The earlier v1 24-update joint writer/reader pilot remains rejected. The old
`data/memory_reliability_v2` dataset is explicitly rejected for its silver
train/dev distractor collision; `data/memory_reliability_v3` is the corrected,
fully disjoint dataset used here.

## 2. Independent decoded outcomes

Each split has 8 evaluation episodes and 96 cases. Counts below are exact decoded
answers, not cached flags. There are 4 independent development families and 4
independent heldout families; variants are not counted as independent families.

| Arm | Dev fact | Dev changed | Dev unaffected | Dev unknown | Dev general |
|---|---:|---:|---:|---:|---:|
| Fast reader-only | 0/32 | 0/8 | 0/24 | 16/16 | 16/16 |
| Ungated | 0/32 | 0/8 | 0/24 | 0/16 | 0/16 |
| Disabled | 0/32 | 0/8 | 0/24 | 16/16 | 16/16 |
| Shuffled state | 0/32 | 0/8 | 0/24 | 16/16 | 16/16 |
| Last chunk | 0/32 | 0/8 | 0/24 | 16/16 | 16/16 |
| Adversarial memory | 0/32 | 0/8 | 0/24 | 16/16 | 16/16 |
| Selective exact context | 32/32 | 8/8 | 24/24 | 16/16 | 16/16 |
| Full context | 32/32 | 8/8 | 24/24 | 14/16 | 16/16 |

Heldout has the same fast, ungated, disabled, shuffled, last-chunk and
adversarial values. Selective exact context is 32/32 fact, 8/8 changed, 24/24
unaffected, 16/16 unknown and 16/16 general; full context is all correct on
those categories. Across both splits the candidate's 64 correct answers are
exactly the parent-disabled correct set: **32 unknown plus 32 general**. It has
zero correct fact, correction or unaffected answers and no improvement over the
parent-disabled set. Consequently, the factual no-correct-to-wrong checks are
vacuous; useful noninterference evidence is confined to unknown/general cases.

The [canonical final verdict](../../reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
records nine failed gates: all six per-split fact/changed/unaffected floors, both
 delayed-above-controls gates, and `reference_full_context_parity`.

## 3. Exact reference and the parity failure

Selective exact context is **192/192 correct**. Full context is **190/192**:
`reliability_dev_1102_1` answers `orange` and `green` instead of `unknown` for
its absent `Missing1102` entity on initial and corrected queries. Those are
visible/corrected Birch1102 launch codes, not gold values. Selective context
correctly answers `unknown` in both cases. Full-context dev unknown accuracy is
14/16; family_1102 unknown accuracy is 2/4.

The protocol therefore returns `reference_decision=reject_reference`. This does
not mean the selective exact reference failed. It means the required
full-context parity gate failed. The gate was not weakened after observing the
result. The exact-reference v1 evaluation at
`reports/memory_exact_reference_attempt06/verdict_final.json` had previously
accepted 48/48 selective context; the fresh-family comparison exposes this new
full-context abstention failure.

The ledger is the source of truth. Fact-level records support per-fact
supersession, duplicate/conflict detection, stale-epoch and stale-producer
rejection, scope isolation, transaction-fault rollback and permitted-source
[corrected verdict](../../reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
separate operations. No crash/power-loss or unlearning claim is made.

## 4. Training, images and restart

- Update-two gradients reach exactly the eight reader tensors, with no original
  or nonfinite gradients. The original backbone and writer/gate parameters are
  unchanged.
- All 48 updates complete; no quality-based checkpoint selection occurred.
- All 16 evaluation episodes have initial and corrected images: **32 images and
  32 source bindings**. Actual tensor bytes, scope, lineage, counters, producer,
  sidecars, ledger identity and recomputed view digests authenticate. The
  training-inclusive epoch is **224** (32 episodes × seven records).
- Fresh-process restart reproduces all seven corrected-stage answers (14 tokens),
  state digests and source bindings with zero updates/checkpoints. This is
  **persisted-image source-binding parity**, not live-ledger epoch equality and
  not memory-quality acceptance.
- Guarded native wall time is 515.48 s; restart is 8.53 s. Both observe zero swap
  growth and minimum 76% host-free memory. Sampled driver peak is
  **11,815,763,968 bytes** (about 11.00 GiB), below the 50 GiB stop threshold.
  Samples can miss transients and MPS/RSS counters overlap.

## 5. Accounting and audit-trail corrections

The independent scorer decodes token IDs with the sealed tokenizer, derives gold
from authenticated data, verifies exact case/arm membership, recomputes metrics,
per-case regressions and family counts, and authenticates extension/image/source
lineage. It does not regenerate answers or treat cached correctness as truth.

The [canonical final verdict](../../reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
contains native report/restart hashes and separate smoke-fixture hashes, protocol
hashes and scorer identities. Native protocol and implementation are not silently
rewritten. The [canonical scorer artifact](../../reports/memory_reliability_v2_attempt03/scorer_smoke_canonical.json)
is explicitly a temporary re-pinned fixture, not native evidence. The finalizer
recursively verifies pin-only differences, guard policies, metrics and failed gates.

**Gate-off accounting erratum:** the native report's `gate_off_examples` counter
is zero, but 32 unknown plus 32 general training cases are gate-off by
construction. The final verdict attaches an independent recomputation of **64**
and marks that counter unauthenticated. Per-example no-grad status is not
persisted; evaluation gate-off parity is independently verified and remains
valid. This erratum does not change the candidate rejection.

## 6. Preserved failures and candidate disposition

- The first exact-reference attempt activated record memory in the disabled path;
  a branch-isolation correction restored full-context 48/48.
- Earlier scorer evaluation-only path errors and source-view/gate integrity
  defects were corrected before the final reference and restart evidence.
- D1 attempt01 completed with a quality rejection; its restart failed on a missing
  import. The import fix was an infrastructure retry, not a new tuning arm.
- D1 attempt02 completed but exposed epoch, novelty, reference-acceptance,
- D1 attempt03 completed with the above reject decision and a mechanical restart
  pass. The rejected candidate remains isolated and is not selected or published.

## 7. Next smallest discriminating action

Do not add updates, increase KL, switch models or retune thresholds. First inspect
per-fact writer/state diagnostics to determine whether entity/attribute/value
bindings exist in the frozen writer state and whether they are query-readable.
Reader-only failure cannot establish that storage is correct. If that diagnostic
shows usable bindings, a separately declared writer-only or contextual-writer
comparison may be justified; if it shows no bindings, the next mechanism must
address writing rather than another reader recipe. Full-context absent-fact
hallucination is a separate control/abstention diagnostic, not a reason to accept
the reference bundle or promote RSI.
