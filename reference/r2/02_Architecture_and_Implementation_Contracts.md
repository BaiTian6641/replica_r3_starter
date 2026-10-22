# 02 — Architecture and implementation contracts

**TAIS Obsidian R2 · 22 October 2026**

This is a proposed construction specification. Class names, CLI names, and package paths are implementation targets, not claims that the user’s repository already contains them. Preserve existing working code and adapt it through a migration layer rather than replacing it wholesale.

## 1. Subsystems and responsibilities

| Subsystem | Initial implementation | Trainable responsibility | State ownership |
|---|---|---|---|
| A. Backbone adapter | Explicit complete-block wrappers | Frozen, LoRA, selective, or full according to experiment | Base checkpoint revision |
| B. KAL + capability self-model | Outcome/calibration heads plus empirical capability ledger | Predict correctness, benefit of another action, and mismatch | Heads in checkpoint; observed outcomes in runtime |
| C. Cognitive workspace / ThoughtCore | Typed slots + small recurrent core + action head | Maintain goals, compare proposals, allocate evidence and compute | Task-local state, trained core |
| D. HRL + dynamic Engram | Exact/semantic retrieval + learned writer/readers | Find and integrate appropriate memories | Persistent scoped records and model-specific payloads |
| E. Fast adaptive memory | Private linear state, then MLP/shared/multi-rate candidates | Learn within an observed stream | Session state, not original MoE expert weights |
| F. Inquiry / task interaction | Allowed tool interface + budgeted action policy | Ask, search approved sources, test a hypothesis, stop | Action ledger and sandbox |
| G. Learning director / compiler | Typed learning proposals + candidate trainer | Choose useful practice, transformation, update target | Separate candidate checkpoint and training-job manifest |
| H. Replica identity | Approved identity records + decision/style conditioning | Model source-person priorities and expression | Source archive, versioned identity, replica branch history |
| I. CA1 promotion / evaluation | Deterministic validation service + human approvals where required | None initially; learned signals are advisory | Protected evaluator and publication manifest |

Memory, reasoning, and identity have different semantics. Reuse storage infrastructure without treating every stored object as a fact or every neural state as autobiographical memory.

## 2. The full runtime loop

### 2.1 Decision boundary

At the start of a user turn or after a completed tool observation:

1. Resolve the active model, interface, memory, skill, identity, and policy revisions.
2. Establish the authorized goal and action/resource budget.
3. Read the **available** prompt, observations, and relevant memory. No future answer enters the state.
4. Run KAL and the cognitive core for a bounded number of updates.
5. Produce an answer or an action proposal.
6. The host checks permissions and budget. Allowed actions execute in the declared environment.
7. Append the actual result as an observation, or the failure as a failure. Do not convert a proposed or imagined result into an observation.
8. Re-enter the decision boundary with the newly available evidence, until success, calibrated abstention, or budget exhaustion.

A model-generated plan is useful working state, not verified evidence. The audit trace records action type, evidence IDs, tool results, and outcome; a verbal explanation is not treated as a complete readout of neural causation.

### 2.2 Neural recurrence

A proposed workspace update is:

\[
 W_{k+1}=\operatorname{Norm}\left(W_k+G_\phi(W_k,O_t,R_t,I_t,B_t)\right)
\]

`O_t` is visible observation state, `R_t` the permitted retrieved evidence, `I_t` the active approved identity conditions, and `B_t` remaining budget. The update produces separate heads for action logits, expected outcome, and workspace-to-backbone injection.

Start with an ordinary recurrent MLP or GRU. Compare a CTM-inspired temporal-history core and a Coconut-inspired latent-continuation alternative as separately named algorithms [R05](05_Research_References_and_Evidence.md#r05)–[R06](05_Research_References_and_Evidence.md#r06). Do not equate RoPE applied to tick numbers with CTM synchronization.

Internal iterations may improve a task plan without writing durable memory. Workspace reset is mandatory on task/scope boundaries. Branching plans use isolated copy-on-write state.

### 2.3 Two-way backbone interface

Maintain `sense`, `route`, and `inject` compatibility with TAIS. A reader can add:

`h_out = h_native + gate_memory * delta_memory + gate_workspace * delta_workspace + gate_fast * delta_fast`

Use independent null-capable gates. Initialize new output projections to zero and the upstream branch normally. Test nonzero first-step output-projection gradients and later gradients to the rest of the branch. Do not zero every multiplicative path.

The simplest initial schedule computes workspace state from the **known prompt at a decision boundary** and keeps it fixed during the next emitted answer/action segment. Token-causal per-layer recurrence is a later optimization. A summary of a teacher-forced target sequence must never condition earlier target predictions.

The first MiniCPM5 port can reuse R1’s candidate block index 27; 13 and 27 form a two-port candidate. These positions are experimental, not optimality claims. Preserve original decoder/cache indices. Inspect and pin each model implementation before choosing wrappers.

## 3. Persistent record memory and fast memory

### 3.1 Dynamic Engram path

The proposed dEngram is a learned writer/reader around a versioned record store, inspired by sparse lookup/fusion rather than a direct copy of original Engram [R01](05_Research_References_and_Evidence.md#r01).

Start with source-text and uncompressed-feature controls, then train a pooler. Candidate settings are memory width 256, 8–32 slots per record, and 4–8 retrieved records. Record precision, payload bytes, and exact-source fallback rate. A record’s text/image/code evidence remains available subject to retention policy.

Use exact identifiers and semantic retrieval, filter by authorization and time before injection, and include distractors plus null memories in training. A repeated identifier must not overwrite a collision-shared embedding row. Keys need record ownership and collision handling.

MAC and LongMem are useful design references for trained encoders/readers around frozen backbones; their findings do not certify arbitrary hidden-state interchangeability [R15](05_Research_References_and_Evidence.md#r15)–[R16](05_Research_References_and_Evidence.md#r16).

### 3.2 Fast-state path

Keep fast tensors as explicit function inputs/outputs. A simple baseline learns key/value associations in a local squared-error objective. The derivative is taken with respect to fast state; the offline outer loss can differentiate through the update into trained projections. Runtime gradients remain restricted to session state.

Titans motivates fast neural adaptation [R02](05_Research_References_and_Evidence.md#r02). uTTT motivates sharing ownership across depths [R04](05_Research_References_and_Evidence.md#r04). HOPE motivates independently trained update frequencies [R03](05_Research_References_and_Evidence.md#r03). First compare one mechanism at a time.

**Causal barrier:** all ports in a completed chunk read its start-state snapshot. Writes produced from that chunk become visible only to the next chunk. The complete-chunk observations can train an update for the future; they cannot influence predictions earlier in the same chunk through another layer.

Candidate multi-rate periods are 1, 8, and 64 chunks. Each level needs explicit update inputs, objective, counters, optimizer state, and reset rules. Compare capacity-matched one-rate controls. Do not interpret the literal periods as established research hyperparameters.

## 4. KAL and the capability self-model

KAL’s first output is `P(correct | task, checkpoint, memory_mode, decoding_budget)`, trained against observed outcomes. A separate head estimates the value of retrieval, verification, or more computation. The label is not “real entity versus invented entity,” nor “the next token was likely.”

Keep a runtime capability ledger with task family, attempted count, successes/failures, validation source, uncertainty interval, known failure modes, and model revision. This supplies a functional self-model: what this system has actually demonstrated under specified conditions.

Train monitors on detached features initially. During joint/full adaptation, first learn task behavior, then recalibrate monitors against the changed model using a disjoint calibration split. A permanently frozen probe cannot be assumed valid after representation changes. The task policy must not be rewarded simply for increasing the probe’s confidence.

Abstention, information acquisition, and extra thinking are different actions. Very low confidence is not an unconditional command to refuse: an available reliable tool may make the problem solvable. The policy weighs expected benefit, action cost, permissions, and remaining budget. Replace the old fixed `EPIG * LP * (1-P(IK))` product with a tested baseline priority policy first; the learned ranking formula is an ablation, not a truth law [U03](05_Research_References_and_Evidence.md#u03) §1, §8.

## 5. Learning director and compiler

### 5.1 LearningProposal contract

A proposal specifies:

- source event IDs and the measured failure/success pattern;
- target capability and intended future task family;
- source transformation, allowed practice generator, and evidence requirements;
- update kind: record, procedure/code, sidecar, skill adapter, selective backbone, or full-development backbone;
- parameter ownership and allowed names;
- candidate parent revision, compute/token/step bounds, and allowed optimizer choices;
- training/replay manifests and development reward split;
- retention checks, identity constraints, promotion split, and rollback revision.

The agent can choose within a finite approved grammar before learning to emit broad update directives. SEAL motivates making update proposals learnable [R07](05_Research_References_and_Evidence.md#r07); it does not require giving the proposer unrestricted shell access.

### 5.2 Three compiler outputs

**Record compiler:** observed context becomes retrieval keys and neural memory representations. This makes information accessible, not necessarily a new skill.

**Procedure compiler:** repeated verified experience becomes a reusable plan, sandboxed executable skill, or neural skill adapter. Compare it with retrieving the original procedure. Voyager is a system-level executable-skill precedent [R11](05_Research_References_and_Evidence.md#r11).

**Learning-policy compiler:** experience changes how the system selects practice, diagnoses mistakes, or chooses updates. Start with supervised selection of successful bounded proposals; add bandit/RL training only after reliable task and retention rewards exist. Absolute Zero is relevant to verifiable self-proposed tasks [R08](05_Research_References_and_Evidence.md#r08).

Context-to-adapter hypernetworks remain an optional later implementation. A conventional small adapter trainer is sufficient to test the full-chain hypothesis. Do not block the project on training a large general-purpose hypernetwork.

### 5.3 Autonomous improvement cycle

```text
experience batch (training partition)
  -> diagnose reusable successes / recurring failures
  -> choose an authorized learning target
  -> build and validate practice + replay
  -> train an isolated candidate
  -> score on development feedback (used by learning policy)
  -> independent promotion evaluation (not exposed as training examples)
  -> compare task gains, retention, calibration, identity, and resource cost
  -> publish a consistent new version OR reject and record why
  -> test later unseen tasks at the same evaluation budget
```

Repeated querying of a promotion set also creates selection pressure. Cap proposals, log evaluation access, rotate promotion windows, and preserve a final sealed audit stream that never influences candidate selection. The system may learn from development failure reports; it may not edit its evaluator, test oracle, or permission policy.

## 6. Replica conditioning is separate from factual verification

Use separate branches for source-person preferences/decision style and factual evidence. A preference should affect which acceptable solution is favored; it should not make an incorrect test result true. A stylistic adapter must not erase provenance or cause fabricated personal recollections.

Identity state contains a baseline source snapshot, approved revisions, and the replica branch’s own later experiences. Skills may improve automatically within authorization. Claims about the human’s values or biography require the appropriate source/confirmation policy; autonomous agent experience cannot silently rewrite them. Full protocol: document 04.

## 7. State ownership and write permissions

| State | Development mode | Deployed mode |
|---|---|---|
| Original checkpoint | Immutable archived reference | Immutable reference |
| Candidate backbone | Full/selected updates in declared lab arms | Frozen within a deployment epoch |
| Cognitive/writer/reader parameters | Jointly trained according to experiment | Candidate-only updates, validated atomic publication |
| Fast state | Differentiable inner update during meta-training | Scoped session updates; no global parameter mutation |
| Durable evidence/records | Authorized data ingestion | Authorized commits/corrections/deletions |
| Skill adapters/code | Candidate training and testing | Validated skill activation within scope |
| Source identity | Consent-controlled construction | No automatic rewrite of source-person history |
| Evaluator, access rules, budgets | Human-controlled configuration | Not writable by model/learning candidates |

Keep W0 logging, W1 workspace modulation, W2 durable-record writing, W2F scoped fast-state updates, W3 candidate extension/skill training, and W4 backbone revision. W4 is permitted for named development arms, not the ordinary deployed large-model loop. Major production backbone changes require a separately accepted migration experiment.

## 8. Gradient ownership and training contracts

1. **Frozen base is not `no_grad` everywhere.** Trainable branches inside the decoder need input gradients through later frozen layers [R20](05_Research_References_and_Evidence.md#r20).
2. **Joint/full training invalidates frozen feature caches.** Recompute support features with the candidate backbone; stale teacher features are allowed only as explicitly frozen teacher targets, not presented as current-model features.
3. **Writer gradients:** keep writer outputs in the graph through the answer loss when training the writer. Persistent detached vectors are not a substitute.
4. **No IO in differentiable/recomputed forward.** Return write proposals, commit once in the host transaction. Activation checkpointing may replay forward [R21](05_Research_References_and_Evidence.md#r21).
5. **Explicit inner versus outer optimizer.** Fast state is not registered in the long-lived base optimizer. Preserve or intentionally truncate meta-gradients and label the choice.
6. **Loss separation:** answer/action losses train declared policy/core groups; monitor calibration uses its own outcomes/split; identity updates use approved source labels. A verifier’s raw outcome is not automatically a differentiable loss.
7. **Full means full within the stated scope.** Report all parameter groups, missing gradients, and actual changed tensors. For MiniCPM-V, distinguish full text decoder from full multimodal checkpoint. Frozen vision means it is not full-VLM fine-tuning.
8. **Tied parameters:** deduplicate optimizer groups and state counts; verify embedding/output-head sharing survives export.

A full-adaptation developmental objective can combine task/action supervision, replay, and reference-preservation on unrelated tasks. Do not force agreement with the old model on valid corrections. Do not assume a universal optimizer or borrow a pretraining learning rate for short adaptation.

## 9. Checkpoint, cache, and representation migration

Publish a `SystemManifest` binding base hash, sidecar hash, tokenizer/processor, native cache ABI, writer/reader revisions, fast-state schema, skill catalog, identity snapshot, policy version, and evaluation report.

After any backbone/interface update:

- invalidate native caches and recompute retained prompts;
- reset fast state unless migration has been independently validated;
- re-encode dependent latent memories from eligible sources;
- rebuild or certify retrieval indexes if their embedding producer changed;
- recalibrate KAL and capability estimates;
- rerun skill/identity composition tests before reuse;
- publish one atomic manifest, not independent component switches.

Old sources remain subject to deletion/retention permissions. A model-specific representation whose source is no longer available must fail compatibility checks rather than silently “convert” by matching shape.

For small full-tuned variants, disabling sidecars restores the **adapted base**, not the original pretrained base. Keep both references so that “bypass parity” is not misreported.

## 10. Suggested repository construction

```text
src/tais_obsidian/
  retrofit/                 # pinned model adapters, wrappers, manifest/export
  cognition/                # workspace, thought_core, action_policy, KAL
  memory/                   # evidence records, writer/readers, fast/shared/multirate
  learning/                 # diagnose, curriculum, proposal compiler, candidate trainer
  replica/                  # source archive, decision records, identity snapshots, lineage
  runtime/                  # bus, scopes, tools, transactions, snapshots, budgets
  evaluation/               # task/retention/longitudinal/replica tests and sealed runner
  training/                 # regimes, objectives, causal episodes, replay, profiling
configs/r2/
reports/r2/
tests/r2/
```

Reuse real existing modules after audit. One integration owner maintains the shared contracts. Assign separate workers to model/training, cognition, memory, learning/evaluation, and replica data; each gets isolated files/worktrees. Interface changes require a decision record and migration tests.

## 11. Construction work packages

| Package | Work and output | Exit evidence |
|---|---|---|
| C0 | Audit repository, user hardware, R1 contracts, raw pilot results; pin model and data | Actual code/mocks identified; baseline runs; unresolved facts listed |
| C1 | Full-chain event/action/proposal/identity schemas; deterministic runtime with real bounded task environment | Entire observe→verify→learn-candidate→promote/reject→restart loop executes on tiny fixtures |
| C2 | Backbone adapter + one reader + workspace core; serialization and mode-specific parameter registry | Identity/causality/cache tests, gradient assertions, tiny overfit without quality claims |
| C3 | Controlled small-model development study and matched controls | Held-out task/retention/profile reports; selected regime justified |
| C4 | Freeze developmental checkpoints; repeated learning cycles with skill candidates | Improvement beyond memory-only/fixed-practice controls; rejection and rollback tested |
| C5 | Personal replica archive, snapshot, held-out decisions, initial conditioning | No fabricated biography; correct history attribution; measured preference/process fit |
| C6 | Optional fast/shared/multi-rate and latent-core variants, one change at a time | Matched-capacity/compute gains or explicit negative result |
| C7 | Small MiniCPM-V text and image integration; fresh model-specific readers | Full-text versus full-VLM scopes labelled; multimodal retention and state tests |
| C8 | Large Gemma / MiniCPM-V ports and bounded full-chain operation | Per-model evidence; no direct latent transplant; production backbone remains frozen |

C1 and replica source collection begin early. C4 does not wait for every optional C6 mechanism. C3 full fine-tuning is a core scientific comparison, not a last resort.

## 12. Mandatory tests and realistic limitations

Test identity/empty paths, finite gradients, correct target masks, checkpoint replay idempotency, no future leakage, packed-session reset, tool result validation, candidate isolation, cross-user denial, evaluation partition isolation, restart equality, source deletion propagation, identity mode separation, and atomic promotion/rollback.

Add **causal behavior tests**: removing relevant evidence should affect the answer appropriately; shuffling irrelevant memory should not; swapping a verified skill should change only eligible task behavior; more ticks must be evaluated against extra ordinary reasoning tokens or tool calls. An attractive latent projection is diagnostic only.

Signatures authenticate provenance, not truth. Tool execution validates a specified property, not every possible property. A learned judge remains fallible. A static validator and a passed unit suite do not establish continual learning. Release notes must name unrun neural, GPU, and quality tests explicitly.
