# Proposed cognitive and Replica contracts — 22 October 2026

Review timezone: **America/Chicago**. This document resolves design ambiguities found in the full-chain review. Its contracts are **PROPOSED**, not claims of implemented runtime behavior or trained cognitive ability. [R2](../../reference/r2/01_Full_Chain_Framework_and_Reevaluation.md) and its [identity protocol](../../reference/r2/04_Personal_Replica_and_Identity_Protocol.md) remain unchanged.

## 1. Measured boundary and the complete objective

The native MiniCPM5 text path has short D0 full-tuning, balanced eight-class overfit, completed-update continuation and actual 256/512-token capacity evidence. The separate [native adapter qualification](../../reports/minicpm5_native_adapter_qualification256_report.json) covers one last-block, 256-token D1/D2 forward/backward/export check. It adds 895,490 parameters in 48 tensors. D1 originals were frozen; D2 originals received gradients. The two nonzero first-backward new gradients are output projections initialized to zero; no later upstream learning follows from that observation. This report does **not** qualify augmented generation, optimizer updates, a full typed cognitive bridge, a learned director or the complete operating loop. The [execution state](../../execution_state.md) remains the authority for subsequent runs.

The complete system must execute:

**Goal/budget → permitted evidence → task model, hypotheses and candidate actions → actual action/observation → independent outcome check → attributed experience → selected practice/update → isolated candidate → independent evaluation → publish/reject/defer → later reuse.**

A good memory read or a successful parameter update closes only part of this chain. The Replica uses this same cognitive/learning engine, conditioned on authentic source priorities and a declared identity mode. It develops its own attributed history without rewriting the human source.

## 2. Native residual boundary and the full bridge

For native decoder block `l`, define the complete native result first:

$$h_l^{\mathrm{native}}=\mathrm{Block}_l(h_l;\mathrm{mask},\mathrm{positions},\mathrm{native\ cache}).$$

Then add only independently gated deltas:

$$h_{l+1}=h_l^{\mathrm{native}}+\Delta_M(h_l^{\mathrm{native}},M)+\Delta_W(h_l^{\mathrm{native}},W).$$

Native residual additions already belong to `Block_l`; do not add the incoming `h_l` again. The memory/workspace readers return deltas, with zero-initialized final output projections and otherwise normally initialized upstream factors. A future fast-state delta is an additional declared branch, not evidence that the current wrapper contains it. Bypass restores the associated base revision; after full co-adaptation that is the adapted base, not the original pretrained checkpoint.

The current small path encodes observed support and the known prompt from input embeddings. It pools support into slots, builds a prompt workspace, and reads both into one native block. It does not yet implement the following **full typed sense/route/inject/core/action bridge**:

| Interface | Proposed contract | Required check |
|---|---|---|
| `SenseFrame` | Decision ID; task/goal revision; known prompt/observation references; selected native hidden-state ports; evidence pin; identity/policy version; remaining budget | No teacher-forced answer suffix, future observation or hidden evaluator field |
| `RouteProposal` | Requested evidence/tool/core route, destination port and expected benefit/cost; host checks the allowed route and budget | Routing never creates access authority |
| `CognitiveState` | Typed goal, constraints, subgoals, evidence references, known/inferred/unknown claims, hypotheses, candidate plans/actions, predicted outcomes and uncertainty; optional bounded latent state | Exact references/code remain outside compressed vectors; imagined outcomes remain hypothetical |
| `InjectFrame` | Compatible port/version, independent null-capable deltas, source lineage and validity pin | Native indices/cache ABI preserved; no hidden state from an incompatible producer |
| `ActionProposal` | Answer, ask, retrieve, verify, permitted tool action, revise plan or stop; selected candidate and bounded reason/evidence references | Host validates before side effects; proposal text is not an observation |
| `OutcomeFrame` | Actual tool/observation IDs, verifier outcome, failure/uncertainty, spent budget and applicable goal/subgoal | Real outcomes revise the next state; failures are not converted into successes |

At each decision boundary the core compares bounded alternatives, may construct labelled hypothetical rollouts, chooses an action and updates task state after actual feedback. Proposed simulations are useful for planning but cannot enter the evidence ledger as observed facts. Persist exact goal/subgoal/action/observation references in the task journal; after restart rebuild latent workspace from eligible journal state. Success requires task/plan continuation, not merely record retrieval after restart. A verbal rationale is not a complete readout of neural causation.

The initial generation contract builds workspace from the known prompt at the decision boundary and holds it fixed during the answer/action segment. Rebuild only after a real observation or other declared boundary. Prompt-fixed re-prefill is a valid first generation implementation; token-causal recurrent/cache optimization comes later.

The qualified insertion is the last native block (`-1`, zero-based block 41 in this 42-block model). R2 separately proposes block 27 and a two-port 13/27 comparison. These are **unrun port experiments**, not descriptions of the current qualification. Compare last-only, 27-only and 13/27 under declared parameter/inference-compute controls; preserve native block/cache indices. Location, multi-port sensing and a shared cognitive core must earn task benefit separately.

## 3. Slot granularity and representation versions

**Current small path:** the trainer concatenates the retrieved support for one example and pools that whole retrieved set into its fixed slot count. Four slots therefore do not mean four slots for every stored record. The prompt workspace is another separately pooled/recurrent set. Both are rebuilt for the current forward context; no persistent per-record neural cache is qualified.

**Proposed persistent memory path:** encode each eligible record separately, attach its provenance and concatenate the selected record payloads for reading. A payload/cache key must bind source ID/revision/hash, observation cutoff, subject/branch/purpose, tokenizer/processor, base feature producer, writer version, slot layout/dtype and policy/evidence revision. Record text remains available as permitted exact fallback. A writer or base update invalidates old features; shape compatibility is insufficient. During writer/co-adaptation training, recompute current features in the differentiable graph rather than training through detached stale caches.

Set-level compression, per-record caching and removal of duplicated source text are distinct experiments. Report retrieved-record count, slots per set/record, token/byte costs and evidence fallback so the comparison is interpretable.

## 4. Evidence pins and revocation barriers

Every decision obtains an immutable `EvidencePin`: ledger revision; selected source IDs/revisions/hashes; query observation cutoff; subject/branch; identity mode/version; purpose/permission version; and model/writer/reader/cache compatibility. The selected evidence/workspace remains pinned throughout the segment. Appended evidence becomes available at the next declared decision boundary; do not silently splice it into an in-flight decode.

A snapshot pin is **not an irrevocable permission lease**. The host checks current eligibility at outward action/output and publication boundaries. A selected source being revoked, deleted, access-restricted or superseded invalidates dependent live state. Proposed barrier behavior is:

1. Latch invalidation for affected request/branch/pin; stop further streaming/output and prevent new tool actions or commits under the old pin.
2. Cancel or quiesce the affected running request at the host boundary. An already executed external action cannot be undone by clearing a tensor; journal its actual outcome.
3. Discard dependent retrieved payloads, task latent workspace, native KV/recurrent cache and fast state. Remove invalid source references from the recoverable task journal or mark them inaccessible, according to retention policy.
4. Rebuild/re-prefill from the remaining permitted evidence and a new pin before resuming. Fail closed if dependencies cannot be identified or rebuilt; never hot-swap only the database while retaining contaminated live context.
5. Invalidate dependent examples/candidates/manifests and block their publication/use until rebuilt or otherwise qualified. A chronological historical query does not bypass current privacy revocation.

Already delivered output, backups and influence learned into parameters are separate retention/rebuild questions; this barrier does not claim neural erasure. Host cancellation, source authentication, policy storage and derived-artifact inventory are implementation requirements. They are not supplied merely by an episode JSON field, pinned ID or this document.

For historical simulation, retained superseded testimony may still be eligible as an explicitly dated report about what was known/believed then. Rebuild under the new pin and mode-aware status rather than presenting it as a current fact. This does not bypass a present privacy revocation or retention restriction.

## 5. Private fast state and task state

Default proposed policy is `persist_fast_state=false`: initialize fast state for a request/session, update only from completed eligible observation chunks, and reset at request end, task/scope/branch/identity changes, producer incompatibility or revocation. All depth ports read one chunk-start snapshot; writes become visible only after the chunk. Internal state updates cannot mutate the global optimizer, source ledger or another request's state.

Cross-request persistence requires an explicit host policy with owner, scope/purpose, source lineage, schema/producer versions, maximum bytes, maximum update/chunk count, expiry and reset/revocation rules. Missing bounds mean persistence is disabled. Treat a persisted state as a versioned compatible payload, not a globally shared hidden memory. Exact task-goal journal persistence is independent of whether approximate neural fast state is reset; reconstruct a resumed task's workspace from eligible exact state.

## 6. Authority matrix

These are distinct write paths; a permission for one does not authorize the others.

| Change | Who proposes/learns | Who authorizes/commits | Gate |
|---|---|---|---|
| W0 event logging / W2 evidence admission | Model proposes a record; actual tool/source observations arrive through the host, which labels provenance/status | Host transaction under current source/scope/purpose policy | Claimed/imagined outcomes never become actual tool results; logging is not neural promotion or authentic-human confirmation |
| W1 task/workspace state | Cognitive/action policy | Task owner and host budget/scope checks | Hypotheses remain hypothetical; exact journal supports restart |
| W2F private fast state | Declared local update rule | Session owner under bounded host policy | Chunk barrier, isolation, reset and revocation; no global parameter writes |
| W3 skill, reader/core/action/calibration/director parameters | Learner proposes an isolated candidate | Independent evaluator and compatible atomic publisher within delegated policy | Parent-bound task/retention/permission gates; reject/defer are valid; ordinary authorized updates need no repeated manual approval |
| W4 developmental base revision | Named development experiment | Operator's experiment/migration authority | Exact full/selective scope, independent study and compatibility migration; forbidden in ordinary frozen-base operation |
| Human source correction/development | Human source or a clearly marked proposal awaiting source confirmation | Source authority | New dated source version; history/retention and derivative invalidation preserved |
| Branch-local skill or nonfoundational conclusion | Replica from its own attributed experience | Delegated branch policy plus relevant evidence/evaluation | Does not alter source archive; trace divergence and uncertainty |
| Branch foundational goal / identity-defining change | Replica may propose, not self-authorize | Explicit authorized human review for protected dimensions | `IdentityChangeProposal`; no silent goal drift hidden inside a skill update |
| Evaluator, permissions, budgets, identity-edit authority | Deliberate operator/source configuration | Protected host configuration | Outside learner/candidate write scope; learned confidence cannot grant permission |

## 7. Identity change and mode-aware behavior

Before identity-affecting publication, construct a proposed `IdentityChangeProposal` with: proposal ID; parent identity/branch version; mode and owner; change class (`source_correction`, `source_development`, `replica_development`); dimension; old/proposed values; source/evidence pins; stated reason and uncertainty; allowed uses; expected behavioral effect; retained foundational goals; tests; affected derivatives; required authority; approval/policy reference; and migration/rollback version. This is a specification, not an implemented schema.

The host validates change class/protected dimensions. Source corrections and developments require source authority. Ordinary verified skills and bounded branch conclusions may proceed under delegated policy; protected foundational branch-goal changes require explicit authorized human review. A model cannot relabel a major identity change as a benign skill update. No source archive is fabricated to populate these fields.

| Mode | Proposed output/learning rule | Scoring |
|---|---|---|
| `source_snapshot` | Predict what the dated source would believe/choose; label that attribution, including uncertain or mistaken beliefs | Source fidelity at that date; external world correctness scored separately |
| `source_current` | Predict latest approved source views, retaining their source/version and uncertainty | Current-source fidelity; no automatic update from agent inference |
| `parallel_replica` | Use approved starting priorities plus explicitly branch-owned experience/conclusions | Technical correctness, retained protected goals and traceable divergence; never rewrite human history |
| `assistant_for_source` | Recommend using current authorized preferences and verified evidence; identify a source belief as a belief when contradicted | Correct assistance and preference/process fit as separate dimensions |

The host chooses mode/version explicitly. A proposed trace separates `predicted_source_belief_or_choice`, `world_assessment` and any `recommended_action`; none silently substitutes for another. Training/evaluation targets carry the mode and attribution in the rich episode/identity manifest. A faithfully predicted mistaken source belief is not a verified technical fact or an assistance target. Solver failure cannot become success through preference conditioning. Maintain distinct fidelity and correctness losses/metrics; source-person review establishes personal priority fit, not physical truth.

## 8. Director parameters, cadence and learning value

Declare adaptable components as `phi_writer`, `phi_reader`, `phi_core/workspace`, `phi_action`, `phi_calibration` and `phi_director`. The latter action/calibration/director components and full bridge are proposed, not established by the small embedding-sidecar inventory. Each component has a named parameter registry, hash/version, compatible parent manifest, optimizer scope and accepted-update history. Permissions/evaluator rules are `P`, not a trainable part of `phi_director`.

The director receives eligible observed failures/successes, family-level capability/uncertainty summaries, source pins, permitted proposal catalog, remaining budgets and prior candidate outcomes. It selects a bounded task family, eligible practice/generator, update target, replay and step/token limits. It receives no sealed audit data, hidden promotion labels or authority to change its reward, verifier or permissions.

**Proposed initial cadence:** choose practice at a completed task/failure boundary within the cycle budget; keep director/calibration versions fixed during a candidate comparison. Train at most one director candidate at the completed-cycle boundary from prior development/reward records, then independently evaluate/publish it. The skill candidate and the director candidate are distinct updates with separate costs/versions. A zero remaining budget yields no training action. More frequent online director updates are a later declared ablation.

For the supervised-ranking/bandit pilot, predeclare a development-only learning-value reward such as

$$r=\Delta S_{\mathrm{reward}}-\lambda_R\,\mathrm{retention\ penalty}-\lambda_C C.$$

`Delta S_reward` compares candidate versus parent on unseen reward-role cases under matched inference compute. Retention penalties use declared old/critical-family budgets. `C` counts normalized practice/teacher generation, verification, replay, gradient updates and candidate evaluation costs, including rejected candidates; normalization and weights are fixed before the run and reported with raw costs. Hard scope/identity/causality failures reject the candidate regardless of reward. Reward cases are separate from independent promotion and final audit. The publisher uses its own protected acceptance rule, not this scalar or the director's confidence alone. T0/T1/T2/T3 and replay-matched controls determine whether selection adds value.

## 9. Early connected pilot and the later selected system

Restore R2's early complete simple-loop direction without relabelling a symbolic demo as a learned system. **Independent S3.1 preparatory work may begin after S1.1**, on one explicitly fixed original/baseline model, identity mode and host policy, in parallel with S2. This pilot connects actual model proposals, one bounded tool/verifier, exact goal/observation journal, experience admission, a typed learning proposal and independent candidate accept/reject plumbing. A candidate executor may initially be a visibly labelled contract stub; only a real update plus later unseen gain can close the learning claim.

Early pilot acceptance is narrow: real model action, real observation/failure, persisted goal/next-action reconstruction after restart, scope/revocation checks, and a trace distinguishing hypothetical versus observed outcomes. It is partial preparatory evidence, not full S3.1 completion, trained architectural benefit or ten-cycle self-improvement.

Existing S3.1 dependencies and all 33 stage IDs remain unchanged: **binding and qualifying the selected frozen architecture still requires S2.5**, and the later candidate/longitudinal study retains its gates. At selection, migrate/re-encode compatible evidence and recalibrate; do not silently reuse pilot latent state. S2 neural comparisons, early host-loop preparation and authentic Replica collection can proceed in parallel. Full selected-system acceptance still requires the complete chain and independent controls.
