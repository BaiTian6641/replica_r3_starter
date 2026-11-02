# Spark memory formation and the Replica successors

**29 October 2026 · successor specification, not an implementation or launch receipt.**

The current task is useful intrinsic memory in Spark, followed by measured access
across its 36 layers. This document prepares the next connection to the Replica
project: attributable experience, persistent task state, bounded cognitive
control, approved identity and independently checked learning. It authorizes no
native execution, P update, capacity expansion or promotion. Current results and
native authorization belong to the [evidence ledger](../../reports/intrinsic_memory_v3/evidence_ledger_20261029.json)
and the completed [teacher-alignment diagnostic](20261029_Spark_Memory_Teacher_Alignment_V10.md). The earlier S0–S1 results remain preserved; useful recall is still unqualified.

The [all-layer reader comparison](20261029_Spark_Memory_Layer_Access.md) changes
where the backbone can read. Formation changes what is written, when it becomes
visible and how later computation can benefit. These are separate interventions.
The current reader placement remains block-input query followed by a residual
after the complete block. Pre-MLP fusion requires its own experiment.

## 1. Preserve the ownership boundaries

The [current architecture](../architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md)
and [state contracts](../architecture/22_State_Mechanism_and_RSI_Contracts.md) define
the following owners. A shared tensor width or storage location does not merge
their authority.

| Owner | Content and responsibility | Formation boundary |
|---|---|---|
| E | Attributable observations and changing facts | Append eligible observations; corrections create dated successors |
| S | Verified procedures with applicability and exceptions | An independently checked outcome may support a procedure candidate |
| P | Persistent learned transformations | Separate verified teaching transaction with weights, moments, clock and lineage |
| phi | Encoder, representation, reader and declared writer machinery | Candidate training and separately evaluated release |
| C | Exact task graph plus approximate task-owned workspace | Hypotheses and unresolved obligations remain explicitly provisional |
| I | Approved identity snapshot and attributed source history | Source authority changes approved identity; task reward cannot do so |
| K | Measured competence, uncertainty and cost | Actual outcomes update empirical records under a system/version binding |
| F, prospective | Session-owned fast associations | Functional chunk update; no claim that this is E, S or persistent P |
| External spine | Specialist-model, sensor and actor interfaces | Typed causal observations/actions; no ownership of intrinsic memory |

The original Spark backbone remains a separately identified frozen release.
Keep source lineage `Spark-X2.6-4B-FP8`; official FP8, the existing FP8-derived BF16
artifact and official BF16 retain different execution identities. Nothing here
resolves official FP8 execution or admits native multimodal work.

## 2. S2a: completed-interaction formation first

The first formation implementation should add one host-controlled boundary to a
completed interaction. It must not write from decoder hooks, token generation or
activation checkpoint recomputation.

1. Pin an eligible memory generation, source epochs, scope, time cutoff,
   producer/representation/codec versions and exact task revision before reading.
   All active layers read the same immutable selection and snapshot.
2. Generate the response using that pin. Collect actual observations and outcome
   receipts separately from model-generated explanations, predictions and claims.
3. After the interaction ends, construct a bounded proposal from newly observed
   spans. Preserve original order, hashes, provenance, validity intervals,
   permitted uses and verification status. Reuse the current canonical span and
   ordered four-slot contract; expansion beyond its admitted two-span/eight-slot
   read budget requires a new protocol.
4. The host validates the proposal against current authority and resource limits.
   A statement that the agent produced an answer is an observable event; the
   answer's factual content does not become verified knowledge merely because it
   was generated. Unsupported content may remain in C or the attributed branch
   journal with a provisional status.
5. Stage payload and provenance, durably publish their authority and commit
   receipt, then advance the active generation once. New memory becomes eligible
   only for a later interaction. An incomplete transaction is unavailable to
   readers; recovery must distinguish prepared, committed and rejected events.

Use one writer for each mutable owner. The journal commit must bind the prior
authority sequence, new generation, payload digests and receipt; rebuilding an
index cannot invent a committed event. A duplicate delivery with identical bytes
returns the existing receipt. The same event identity with different bytes is a
conflict and fails closed. Preserve the conflicting trajectory for diagnosis.

The first behavioral comparison is write-enabled versus write-disabled with the
same reader, evidence opportunity, interaction sequence and generation budget.
Include wrong-branch writes, duplicated completion delivery, interrupted commit,
restart, correction and revocation. Record whether a later answer uses the new
observation and whether removing that write removes the improvement. This is
formation/recall evidence, not persistent-parameter learning.

## 3. S2b: within-reasoning writes are a separate algorithm

Only after S2a is useful should a separate runner expose chunked state. Define
token positions in the canonical tokenized sequence, independently of how many
times a growing prefix is evaluated. Chunk length and maximum chunks are frozen
in the future launch packet; they are not adaptive after development results.

| Boundary | Required behavior |
|---|---|
| Chunk start | Resolve one immutable snapshot at revision r and record the committed position watermark |
| Chunk computation | Every layer reads r; accumulate bounded write proposals without changing shared state |
| Forward/recompute | Pure tensor operation; no I/O, commit, counters, optimizer step or context mutation |
| Chunk barrier | Validate all proposals against current scope/epochs and expected predecessor r; commit once as r+1 |
| Next chunk | Read r+1; no earlier token/layer consumes a state produced later in the same chunk |
| Retry/restart | Reuse the logical operation identity; load its committed receipt or resume preparation without a second commit |
| Revision/branch change | Start a distinct branch or decision binding; invalidate dependent KV and projected caches |

The logical operation key should hash a canonical structure containing:

```text
schema, subject_or_owner, branch_id, sequence_id, decision_revision,
chunk_index, absolute_token_start, absolute_token_end,
token_prefix_digest, input_state_revision, writer_version, effect_kind
```

Separately record `attempt_id`, transport retry count and
`checkpoint_recomputation_id` as audit fields. They must not change the logical
key: otherwise a retry or backward recomputation could create another write.
Bind the complete proposal digest in the receipt. Reusing the logical key with a
different proposal is an explicit conflict, not a last-writer-wins replacement.
Record a decoder branch identifier when alternatives are forked; only the
selected authorized branch can publish into its owning scope.

Maintain a committed absolute-position watermark plus a digest of the committed
prefix. A repeated growing-prefix forward may propose effects only for tokens
strictly beyond that watermark. A shortened or changed prefix that overlaps
committed positions must create a new revision or reject; it cannot replay old
tokens into the same authority. Aborted/unselected continuations remain
provisional. Retrying a completed barrier returns its prior receipt without
advancing the state clock or charging another successful write.

Make the training graph explicit. In an unrolled training arm, later response
loss may differentiate through a functional update from the prior chunk's live
tensors. Declare the maximum unroll, retained derivatives and truncation/detach
barriers. Durable serialization and ordinary inference snapshots are detached.
Do not pass the training update through NumPy, disk or hashing. Imperative
NativeFusedAdamW remains the ordinary optimizer; a differentiable inner delta or
SGD rule is a separately named algorithm with its own direct-CE reference.

Before native execution, CPU cases must prove: one logical commit under growing
prefixes; duplicate delivery and retry idempotence; no recomputation effects;
conflicting-retry rejection; crash recovery on both sides of publication; branch
isolation; causal visibility; and correction during an outstanding chunk. A
revision mismatch discards the uncommitted proposal and requires a new decision.
It never combines two generations in one answer.

## 4. Validity across hot, warm and cold storage

Residency and learning clocks are independent. The existing tiered design's
2 GiB hot, 8 GiB warm and 64 GiB cold ceilings are proposed subsystem limits,
not allocations or evidence of fit. Its 8 MiB identity reservation is included in
hot memory, and its 256 MiB replay sub-cap is included in warm memory. These
remain subordinate to the global 50 GiB driver, 512 MiB sustained swap-growth and
15% minimum host-free rules. Apple unified memory is one physical pool; directory
names do not establish GPU residency.

For each admitted record and P state, run the same validity cases after hot→warm,
warm→cold and cold→hot movement and after process restart:

| Case | Required observable result |
|---|---|
| Unchanged complete generation | Same payload/provenance and declared numerical readout equivalence |
| Correction or revocation before selection | Old generation is ineligible before semantic ranking |
| Correction during cold I/O | Revalidate after I/O; reject stale payload/ticket and invalidate projections |
| Missing payload or manifest | Explicit unavailable/quarantined result, never a partial successful recall |
| Corrupt payload or producer/codec mismatch | Refuse before consumption; canonical fallback only if still eligible and separately charged |
| Dirty P eviction | Publish latest committed weights, moments, clock, encoder identity and lineage coherently before eviction |
| Interrupted move/publication | Recover the latest complete authorized state or quarantine its owner; retain unrelated healthy state |
| Slow cold recall | Explicit bounded pending/resume status; no false claim that missing evidence was recalled |

Projection/KV cache identities must include the snapshot generation, source
epochs, scope/time cutoff, producer/representation/codec, reader release and
layer/site. Identical storage bytes under a new eligibility revision are not a
cache hit. A correction invalidates descendants even when their payload remains
on a slower tier. Physical relocation alone must not increase a learning clock,
manufacture teaching events or change a record's authority.

Measure successful and failed reads, cache misses, bytes read/written, temporary
publication overlap, canonical fallback tokens, p50/p95 recall latency and full
answer latency by tier. Report driver/RSS/swap separately without double-counting
shared backing. A cache priority is a performance decision, not a truth score.
Do not describe paging as consolidation or reader density as nested learning.

## 5. S3: persistent learning after useful recall

The existing [P implementation](../../src/replica_lab/contextual_p_learning.py)
explicitly identifies itself as a fixed-question-slot control. It encodes the
public question under frozen phi, then queries one P1 expert. It does not yet
implement hidden-state-conditioned P or establish source-free generated-answer
utility. Freeze the qualified encoder/reader release before isolating P effects.

Prepare paired hidden-state-conditioned and fixed-question controls with the
same reader, expert width/capacity, initial P state, evidence access, update
schedule, teaching examples and total-cost accounting. For the new arm declare
which causal hidden state produces its query, at which layer/position, and
whether gradients stop at that frozen query. Keep one active P1 expert; retain
the existing declared capacity until an actual bottleneck is measured. No
allocation of P4/P16/P64 follows from configurability.

The experimental unit is an independent world/rule family, not a paraphrased
question. Assign world families and seeds to competence, fit, development and
sealed future evaluation before generating descendants. Each teaching episode
must precede distinct future queries. Future answers cannot be writer input,
retrieval keys, identity text, selection features or auxiliary cache state.

Preserve the repaired eleven-arm control ledger and add new read-condition
comparisons as separately budgeted arms. In particular, learned P and reset P
must use the same frozen reader and eligible evidence. For source-free tests
remove task-specific E, S, F, C, KV and cached values, plus answer-bearing I and
auxiliary state. Retain only the explicitly declared stable release, permitted
query and learned P. Wrong-world and shuffled teaching controls must receive
equal teaching exposure. Record complete retained/removed-state inventories.

Use a chronological cycle: initial acquisition → new-instance transfer → source
removal → fresh-process restart → unrelated interfering acquisition → delayed
retention → correction/revocation → authenticated rollback. Delay is measured
in both intervening tasks and elapsed time. Run at least two independent cycles
before considering a nested schedule; two plumbing cycles alone are insufficient
without repeated useful acquisition, retention, correction and rollback.

Correction has three separate outcomes:

1. Retrieval excludes obsolete evidence and invalidates derived values/caches.
2. Replay excludes obsolete targets and descendants at selection and update time.
3. Generated answers stop using obsolete learned content after it influenced P.

A tombstone proves neither the third outcome nor neural erasure. Quarantine
affected P and compare authenticated pre-teaching reset plus permitted replay
against any proposed correction update. Check unaffected knowledge and unrelated
healthy owners. Preserve bad candidates and the previous working release.

Measure generated-answer quality, false recall, retention, total generated
tokens, latency, teaching/retrieval/replay/verifier work, updates and physical
memory. Reasoning reuse is useful only when verified experience reduces future
total cost without degrading accuracy; shorter answers alone do not prove this.
Freeze numerical acceptance margins and budgets in a separate executable packet
before observing its development labels. There is no S3 launch permission here.

## 6. The next Replica integration uses the existing cognitive core

The [semi-cognition chapter](../architecture/04_Semi_Cognition_Learning_and_Replica.md)
describes a goal → evidence → action → independent outcome → experience →
candidate learning loop. Its MiniCPM port locations and older optimizer choice
are historical construction context; current Spark instructions and
NativeFusedAdamW take precedence. Porting those components to Spark is a
separate evaluated integration, not an automatic consequence of memory recall.

The repository already contains concrete foundations:

| Existing component | Actual responsibility | Remaining Spark/Replica integration |
|---|---|---|
| [CognitiveCore and TaskStateEncoder](../../src/replica_lab/architecture/core.py) | Tied recurrent slot self-attention/cross-attention over sense, memory and task inputs; live gradients between iterations | Connect qualified Spark memory with causal sensing and compare no-core/feed-forward/recurrent controls |
| [ExactTaskState and patches](../../src/replica_lab/runtime/task_graph.py) | Goals, constraints, claims, obligations, evidence and verifier-controlled exact transitions | Preserve exact commitments while testing learned workspace/control behavior |
| [TaskRuntime](../../src/replica_lab/runtime/loop.py) | Host proposal/action/outcome orchestration | Bind Spark model, memory revision, action authority, budgets and restart in one connected experiment |
| [WorkspaceStore](../../src/replica_lab/runtime/workspace_store.py) | Version-bound task workspace persistence with an explicit CPU detach | Rebuild from exact state on incompatibility; latent snapshots are an optimization, not authority |
| [IdentityArchive/IdentityStore](../../src/replica_lab/replica/identity.py) | Attributed episodes, approved principles/snapshots, temporal corrections, revocation and separate branch experience | Supply permitted authentic material and retain separate source/current/parallel/assistant modes |
| [IdentityReader](../../src/replica_lab/replica/reader.py) | Optional null-capable independently gated workspace reader | A qualified source-to-slot producer, training and held-out personal conditioning are still absent |

The reference cognitive workspace has eight width-256 slots and two tied
iterations per decision; these are engineering choices, not proven optimal
Spark settings. C owns that workspace; phi owns its recurrent parameters; the
host owns exact goal authority and side-effect commits. Memory recall supplies
eligible evidence. The core proposes actions and task revisions. The external
spine carries future specialist/sensor/actor exchanges without taking over any
of those owners. In particular, attaching an observer is not evidence that it
can control actions or faithfully expose reasoning.

Start the connected Replica study with one approved conditional priority, one
new technical task and one independent verifier. Keep the compact approved I
core and current C within measured active-memory budgets; retrieve larger
attributable history explicitly from warm/cold storage. Compare generic,
approved-summary, exact-archive and neural-conditioning paths with equal eligible
information. Keep choice/ranking, clarification, factual provenance, temporal
fidelity, unknown handling, technical correctness, style and branch divergence
as separate outcomes. Source review assesses personal fit; task verifiers assess
technical results. No authentic archive is bundled and none is inferred from
assistant prose.

The [identity interface](../architecture/09_Replica_Identity_Interface.md) and
[Replica protocol](../05_Replica_Protocol.md) preserve four modes:
`source_snapshot`, `source_current`, `parallel_replica`, `assistant_for_source`.
Agent experience belongs to the Replica branch; it never rewrites human
autobiography. A model inference about a principle remains a proposal until
approved. Approved identity cannot satisfy a failed technical test, override
permissions or silently change a goal.

Test the identity/task dissociation explicitly: reset C/task KV while retaining
I; null I while retaining task evidence; reset both. Run both intervention orders
on fresh worlds. After correction/restart, test whether changed source views and
branch-owned experiences remain attributable. These are behavioral tests of
conditioning and control, with no claim about consciousness or subjective state.

## 7. Advancement and implementation gaps

Current memory/reader engineering does not implement the S2a transactional
formation scheduler, S2b chunk writer, hidden-state P path, trained Spark
cognitive controller or trained authentic identity pipeline specified here.
No result is claimed for those mechanisms. This preparation should become CPU
fixtures and a separately versioned runner before any native authorization.

The next order is: useful recall → completed-interaction formation → causal
chunk formation as its own comparison → persistent transfer/retention/correction
→ calibrated cognitive control and attributable Replica behavior. Authentic
source intake may proceed independently when the human supplies permitted
material; native multimodal work remains deferred. The [Replica stages](../stages/05_Replica_and_Longitudinal.md)
retain their source and human-review dependencies.

Learned local writers follow useful readout, with their differentiable inner
update named separately from imperative optimization and compared to direct CE.
Nested adaptation needs multiple explicitly connected objectives/clocks and
repeatable evidence at matched capacity and cost. RSI needs independently
evaluated successive changes to the learning policy on fresh families; storage,
reader count, ordinary fine-tuning and repeated self-generated explanations do
not establish it. Candidate policies cannot alter their verifier, promotion
criteria, resource guards or approved identity. Retain authenticated parents and
exercise rollback before promoting any later candidate.

## 8. Current repair qualification and learning order — 31 October 2026

The [repair report](20261031_Spark_Repairs_V18_Review.md) and [research cross-check](20261031_Spark_Research_Crosscheck_and_Alignment.md) update the evidence boundary without advancing any stage. Historical V17 post-update margins used pre-update slots. CPU repairs and V18.2 preparation02 now exist, but corrected native behavior remains unmeasured. Existing verification is scoped engineering evidence, not a proof that all correctness blockers have disappeared.

The actual v12 development split was exposed through an engineering test even though the relevant native diagnostics never opened it. It is retired for fresh generalization claims. A fresh disjoint sealed split is prepared outside tests and fitting diagnostics; it is not competence-qualified. Native access history remains unchanged. No source-free, retention, correction or nested-learning result may be inferred from this preparation.

The required progression is **qualified recall → completed-interaction formation → source-free persistent learning → correction/retention/rollback → all-layer and nested-learning evaluation → RSI**. Exactly-once causal chunk formation remains a separately controlled part of formation. All-layer CPU support may be developed as engineering, but cannot bypass useful recall or matched behavioral/cost gates. Excluding a revoked record from fresh retrieval is distinct from removing information already learned in P.

If corrected native V18 fails, the research note proposes one controlled producer–reader alignment comparison. It keeps the representation, capacity and routing fixed while testing decoder usefulness, with separate compression diagnostics. It is not authorized or added to V18. Logical hot/warm/cold quotas do not establish physical residency; measure actual host/MPS allocations and latency. Preserve the intrinsic-memory/Spine ownership boundary and all existing resource, multimodal, P, density, capacity and RSI gates.
