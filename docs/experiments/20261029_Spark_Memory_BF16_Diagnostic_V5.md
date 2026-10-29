# Spark intrinsic-memory v5: correctness and official BF16 diagnosis

Current implementation: `771e9586`, following review snapshot `017f7d71`.
The [v5 evidence summary](../../reports/intrinsic_memory_successor_v5/verification_summary.json)
separates new CPU checks from historical native results. No v5 native attempt has
run. All three v3 approvals are consumed. The [attempt03 result](20261029_Spark_Memory_Native_Readout03.md)
remains unchanged: the FP8-derived BF16 control passed S0 at 8/8 and failed S1
with contextual/exact/wrong/shuffled/absent scores 0/8, 8/8, 0/8, 0/8, 1/8.

The user selected **official BF16** for the next diagnostic. The already acquired
`XHToken/Spark-X2.6-4B`, revision
`66e3ae7bee54e0a523d692cb8ef841f35409807b`, resides in the historically named
`checkpoints/sparkx2_5_bf16_source` directory. All 18 files, totaling
8,238,996,247 bytes, passed fresh streamed validation. No download, base copy or
backbone export occurred. This source is distinct from both the official FP8
lineage and the FP8-derived BF16 used by attempt03. Original parameters stay
BF16 and frozen; contextual parameters, stored slots and optimizer moments stay
FP32 under the existing numerical contract.

## Implemented memory behavior

The new [contextual wrapper](../../src/replica_lab/contextual_memory_v5.py) extends
the existing combined v4 implementation. Sealed v3/v4 execution bytes are preserved.

- `MemoryScope.cutoff` governs live encoding, validation and durable selection.
  At `as_of=20, snapshot_at=10`, a record available at 15 is refused; one available
  at 5 and valid until 15 remains eligible. Current revocation still overrides a
  historical view. Ordinary current-time behavior and restart are tested.
- Cold reads return a complete batch or `PendingContextualRead`. Tickets persist
  scope, producer release, source pins, attempt counts and cumulative reservations.
  Each caller wait is limited to 0–5 seconds. Recovery of an interrupted dispatch
  retains uncertain I/O charges; at most two dispatches are permitted. No partial
  batch is usable. Correction, revocation, missing/corrupt payloads and publication
  changes fail closed. The native caller permits five bounded waits; the outer
  process supervisor contains a stuck I/O worker. A Python thread is not a hard
  cancellation mechanism for uninterruptible OS I/O.
- [Producer-release rebuilds](../../src/replica_lab/contextual_release_v5.py) re-encode
  existing canonical sources into immutable generations. An explicit generation-zero
  authority permits recovery of an interrupted first preparation. Publication is an
  atomic pointer replacement after file/directory fsync, producer-version checking
  and source revalidation under the store transaction. Interrupted candidates remain
  as orphans; the previous committed generation remains intact. Lost current
  authority is refused, never inferred from surviving payloads. Corrected/revoked
  sources cannot publish or reappear; mixed releases are refused. Later current-time
  reads revalidate eligibility against the new cutoff.
- Every snapshot has eight slots. A single selected span supplies four ordered
  width-256 values; four padding slots are explicitly masked. Padded values cannot
  affect logits or gradients. Active 2/9/36 readers share one graph-preserving
  snapshot. Disabled readers skip computation. No retrieval or write occurs inside
  a decoder hook. Placement remains **block-input query / post-block residual**.
- Producer identity hashing caches only ordinary mutation stamps and an explicit
  optimizer invalidation. Tests exposed that fused AdamW can change weights without
  advancing `Tensor._version`; the new owner invalidates before every optimizer
  step. Authority validation remains active at every reader. There is no cached
  K/V projection that can survive correction.

Training re-encodes pinned canonical text through live tensors. Serialization,
durable read and inference are detach boundaries. One NativeFusedAdamW owns each
producer, order and reader parameter exactly once. The first update changes the
zero-initialized reader outputs; subsequent losses reach the producer and early
readers through later frozen blocks. The complete v5 checkpoint binds source,
execution protocol/arm, producer/representation/codec, layout, optimizer ownership,
objective, template, tokenizer, EOS and stopping policy. Reload refuses an incompatible
identity before installing tensors. External specialist/sensor/actor spine interfaces
remain separate and the memory workflow runs with that spine absent.

## Executable diagnostic and gates

The [protocol](../../configs/spark_intrinsic_diagnostic_v5.json) and
[runner](../../scripts/run_spark_intrinsic_diagnostic_v5.py) implement the same
declared workflow for each density arm. Their executable closure, official source
code/dependencies, prepared data and qualified `.venv-spark-bf16` environment are
hashed. Both source and actual Hugging Face cached executable copies are checked
before importing model/configuration classes. No model load is permitted before
admission. The environment-only check and full source preflight load no weights.

1. **S0:** eight fresh complete-context worlds, original model frozen, extensions
   disabled. Require 8/8 correct, zero false recall and no evidence truncation.
   Missing, incomplete or failed evidence closes fitting and development.
2. **Fit:** exactly 32 direct response-CE updates, including one explicit EOS
   target. Four pairs use identical questions but counterfactual evidence and
   different answers. No world ID, answer-bearing identity state or target suffix
   enters the producer. The producer receives canonical source bytes only.
3. **Fit-only diagnosis:** first-token accuracy, separate word/EOS losses, complete
   greedy answers and termination; exact prefix-logit agreement between right-padded
   fitting and generation; live/durable slot and logit equality; contextual, matched
   exact text, wrong, shuffled, zero and absent controls; counterfactual switches;
   per-layer gates, residual scales, logit changes and intended gradient norms.
   Require all eight fitting answers and first tokens correct, terminated answers,
   exact prefix parity, and contextual performance above wrong/zero/absent memory.
   Shuffling is also measured; a fact need not depend on every slot's position.
4. **Freeze before development:** only a passing memory-dependence fit gate opens
   the fresh development corpus. The strict emitted-answer scorer requires contextual
   recall at least 6/8, zero false recall, at most one answer behind selected exact
   text, and more correct answers than each wrong/shuffled/zero/absent control.
   No sealed-final corpus is used. Dynamic correction/revocation tests separately
   report retrieval invalidation and generated behavior.
5. **Numerical continuation:** save only the complete extension and optimizer.
   After the main model process exits, a new process must reproduce the full canary
   logit digest and perform one actual update. This engineering continuation is
   retained even when the fit gate fails; development stays closed. Full original
   parameter hashes are checked after fitting and continuation. Uninterrupted-update
   equivalence is not claimed or budgeted.

EOS omission is an objective gap, not an established cause of wrong first tokens.
This successor also changes source identity, padding and diagnostics, so a v3/v5
difference cannot isolate EOS causally. If fitting succeeds but new worlds fail,
investigate exposure/generalization. If fitting still cannot align values with
answer tokens, prepare a **separate** full-context-teacher/memory-student experiment
under the [research cross-check](../research/spark_memory_20261029/Research_Crosscheck.md).
It must bind teacher correctness, response positions, temperature/loss weights,
teacher forwards/tokens and target storage, and compare direct CE at equal exposure
and total cost on fresh data. Teacher updates in v5 are zero; that branch is not
selected or authorized.

## Per-arm budgets and physical measurements

The durable ledger reserves work before dispatch. Failed/uncertain work is never
refunded. Main and restart use the same ledger and absolute monotonic deadline.
The outer process-group supervisor covers admission, loading, probes, generation,
checkpointing and continuation, reserving 10 seconds for shutdown and 1 for reap.
Each stage also has its own wall deadline. One common native-worker lock spans the
whole attempt. Sampled limits remain 50 GiB driver allocation, 512 MiB sustained
swap growth, and at least 15% host free, with three host samples defining sustained.

| Maximum per arm | Declared amount |
|---|---:|
| Backbone forwards | 957 |
| Backwards / updates | 33 / 33, including one restart update |
| P updates / teacher updates | 0 / 0 |
| Processed input positions | 244,992, including padding and repeated uncached prefill |
| Generated tokens | 856, including stop tokens |
| Original model loads | 2, sequential |
| Source encodes / retrieval calls / write transactions | 77 / 27 / 55 |
| Outer wall / allocated stage wall | 7,200 s / 7,020 s |
| Artifact envelope: 2 / 9 / 36 readers | 160 / 416 / 1,056 MiB |

Fit probes account for 448 forwards: eight worlds times two parity forwards plus
six controls times one teacher-forced and eight possible greedy forwards. Development
reserves another 384. These costs were not borrowed from the sealed v3 budget.
Each arm needs its own exact approval; preparing three packets does not authorize
three runs. Nine/all additionally require a bound, useful current two-site result.

Source encoding, retrieval, fallback, publication, serialization, audit-ledger
rewrites, restart, prefill and generated tokens are recorded. Timing separates host
dispatch from a synchronized device interval. The latter includes host overhead;
it is not a GPU kernel-only timer. File/payload counts, serialized bytes and physical
artifact growth are distinct from unmeasured filesystem/device write amplification.
Driver allocation and RSS overlap and must not be added. Host free/swap readings
describe the whole machine; sampling cannot exclude transients.

The new CPU workflow measures one 65,536-byte fixture snapshot per forward in
every layout, zero retrievals/writes inside the forward, and 3/10/37 authority
validations for 2/9/36 readers. Durable forwards reuse the producer digest but
still recheck source/publication authority. Per-layer projections remain uncached.
The [recorded CPU results](../../reports/intrinsic_memory_successor_v5/verification_summary.json)
report timings/RSS/artifact sizes, not native fit estimates. Random tiny-fixture
mechanical success does not establish semantic fitting, native competence or recall.

## Density controls and Replica succession

The 2/9/36 feasibility packets hold evidence, four realized slots plus masked
padding, writer behavior, exposure, common-site initialization, splits, objective,
scorer and generation policy fixed. Parameter counts do not establish that any
native arm fits. A single synthetic family/seed cannot establish broad superiority.

Before attributing a gain to reader density, prepare these independent controls:

- **Matched parameters:** reuse two complete reader parameter sets, including
  gates, across fixed groups of active depths. The producer/order stay identical;
  every parameter has one optimizer owner. This changes weight sharing and needs a
  new checkpoint/runner identity. It controls parameter capacity while measuring
  the additional projection work separately.
- **Matched total cost:** use the feasibility profiles to freeze work caps that
  include teaching, encoding, retrieval, validation, writes, backward, replay and
  generation. Compare extra low-density training/retrieval/generation with denser
  reads within that cap, retaining identical evaluation questions and scorer.
  Confirm on independent families/seeds. Latency or parameter arithmetic alone
  cannot substitute for measured cost and accuracy.

The [formation/Replica contract](20261029_Spark_Memory_Formation_and_Replica_Successors.md)
remains the successor: useful recall → completed-interaction writes visible next
interaction → separately evaluated chunk-start reads/proposals/exactly-once commits
→ persistent learning on distinct future queries → correction, interference,
delayed retention and rollback → nested adaptation and independently assessed RSI.
Sequence/position/revision/retry/branch/recomputation identity prevents repeated
growing-prefix writes. Generated hypotheses are provisional C/F, not verified E/S.
P stays at the existing declared level until a measured capacity bottleneck.

The existing semi-cognitive core uses eight width-256 workspace slots and two tied
iterations as an engineering reference. C owns workspace, phi owns recurrence, and
the host owns exact goals, action commits and independent outcomes. Approved I
remains outside task reward. After memory works, the first Replica integration
should combine one approved conditional priority, one technical task and one
independent verifier, comparing generic, approved-summary, exact-archive and neural
conditioning with equal eligible information. Test task-state/I resets separately.
Human source history, approved snapshots and branch-owned experience remain distinct.

Hot/warm/cold are storage placement, not learning clocks. Keep bounded approved I
and active C within measured active-memory budgets, and recall archival history
explicitly. The v5 benchmark's small logical store quotas do not allocate a physical
GPU cache or establish consolidation. Native cognition, authentic personal fidelity,
formation, P learning, retention and RSI remain unqualified; multimodal work is deferred.

## Next executable attempt

Review [the two-site packet](../../reports/intrinsic_memory_successor_v5/native_packet_two_v5.json)
and [runbook](../../reports/intrinsic_memory_successor_v5/README.md). The exact external
authorization receipt is absent. The user's BF16 source choice does not reuse any
consumed v3 execution approval. Prepare no replacement packet unless a concrete
binding changes, and never modify a packet after it is approved.
