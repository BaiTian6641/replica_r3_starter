# 01 — Full-chain framework and reevaluation

**TAIS Obsidian R2 · 22 October 2026**

## 1. Restore the project’s actual scope

The target is not simply “a language model that remembers more.” It is an engineered semi-cognitive system whose behavior develops through experience while retaining a stable foundation and a recognizable personal identity where requested.

The user’s replica goal is to document ways of thinking, ideas, life experiences, and selected memories so that a parallel familiar existence can start from them and then develop its own history. R2 treats that goal as a first-class requirement rather than a late persona-vector feature. This is a requirement, not a research finding about consciousness.

The original TAIS thinking document explicitly proposed goal maintenance, bounded workspace, constructed simulation, metacognitive control, and replay. The active-inquiry document proposed asking, consulting evidence, validating, writing, and consolidating. The DKB design included procedural and persona blocks. These sources support a broader scope than the previous R1 milestone order [U02](05_Research_References_and_Evidence.md#u02), [U03](05_Research_References_and_Evidence.md#u03), [U04](05_Research_References_and_Evidence.md#u04).

**R2 central hypothesis:** an existing capable language backbone, trained cognitive interfaces, explicit experience and identity state, and a bounded self-directed learning process can produce useful longitudinal improvement without requiring continual full updates to the deployed backbone. Whether the combination works is an empirical question; the components are not proof of the whole system.

## 2. The foundation is stable, not the entire system

Represent the system as:

\[
 y_t=\mathcal F(x_t;\theta_{base},\phi_t,E_t,F_t,W_t,A_t,I_t,P)
\]

- `theta_base`: versioned pretrained or developmentally adapted backbone.
- `phi_t`: learned writer/readers, cognitive core, routing, action policy, and calibration interfaces.
- `E_t`: durable evidence and episodic/semantic records.
- `F_t`: temporary adaptive fast state, optionally persisted with explicit lineage.
- `W_t`: task-local workspace, goals, hypotheses, and candidate plans.
- `A_t`: validated reusable skills, including neural adapters or sandboxed code.
- `I_t`: replica identity snapshot, approved preferences, and separately attributed experience.
- `P`: externally enforced permissions, resource limits, evaluation rules, and identity-update authority.

During ordinary large-model operation, `theta_base` stays fixed. The other authorized states may change at different timescales. The backbone strongly influences attainable reasoning and sample efficiency; it is **not a mathematical upper bound on the composite system**, because added trainable modules, algorithms, tools, and extra computation change that system. Conversely, increasing stored information alone does not establish new reasoning ability. This distinction is a design interpretation, not a claim of unlimited improvement.

### Two phases that must not be conflated

**Developmental co-adaptation:** introduce new cognitive components into a 1B–2B-class model, train those components, and test limited or full backbone fine-tuning. The purpose is to learn the interface and diagnose bottlenecks.

**Post-freeze learning:** publish a tested backbone/interface revision, then evaluate continuing adaptation with only the declared dynamic state and trainable extension set. This is the relevant experiment for the eventual stable large model.

A successful full-tuning development run does not prove that an untouched larger backbone can use the same modules. A negative frozen-sidecar result does not prove the idea is impossible if the same mechanism works after co-adaptation. Record both findings and the transfer cost.

## 3. What counts as cognition, personalization, and improvement?

### 3.1 Operational semi-cognition

For this project, semi-cognition means observable competence in maintaining a goal; distinguishing known, inferred, and missing information; constructing and revising a task model; comparing candidate actions; deciding when to ask or use a tool; checking an outcome; and adapting future behavior.

No specific latent geometry or brain-region analogy is required to pass these tests. Neither token-by-token generation nor latent recurrence should be classified as “not real reasoning” solely from its representation. The relevant question is what the system can reliably do under matched evidence and compute.

### 3.2 Six distinct improvement claims

| Claim | What changed? | Required evidence |
|---|---|---|
| Acquisition | A new event/fact is retained | Delayed and restarted queries; source and temporal correctness |
| Task reasoning | Better use of available evidence and actions | Unseen compositional tasks at controlled compute |
| Strategy learning | A procedure transfers to new instances | Hold out problem structures/instances; compare a static procedure library |
| Metacognitive control | Better ask/verify/stop decisions | Calibrated error/coverage and action-cost curves |
| Learning efficiency | Better selection of practice or updates | Higher held-out gain per sample, token, or verifier call than fixed/random schedules |
| Replica fidelity | Decisions and expression fit documented source characteristics | Blind held-out scenarios, exception handling, source attribution, and divergence accounting |

Do not collapse these into one “cognition score.” A replica can become more skilled while becoming less faithful. A system can remember more while its reasoning stays unchanged.

### 3.3 Self-improvement has a measurable loop

A model is self-directed in the intended engineering sense when it selects a permitted learning target, gathers or generates eligible practice, proposes an update, and uses outcome feedback to improve its future choices. It need not design its own optimizer from scratch. It must not declare success merely from its own confidence or the score of a judge it is free to rewrite.

SEAL supplies a precedent for learning to generate useful self-edits; Absolute Zero supplies a precedent for self-proposed, executor-checked learning tasks. Neither establishes unrestricted improvement in every domain [R07](05_Research_References_and_Evidence.md#r07), [R08](05_Research_References_and_Evidence.md#r08).

## 4. Audit of R1 and the original plan

| Source decision | R2 action | Reason |
|---|---|---|
| R1 §1: memory-first project definition and long serial progression | **Broaden** to concurrent cognition, learning-loop, and identity tracks | Memory remains a dependency but is not the final deliverable |
| R1 §5.8: controller as optional late branch | **Promote** goal/workspace/action control to the core system | It implements the user’s semi-cognitive objective |
| R1 §5.9: procedural compiler only after most memory work | **Introduce early as a simple validated skill path** | Otherwise the prototype demonstrates acquisition, not improvement of behavior |
| R1 §7: personality evolution deferred | **Separate archive/fidelity now from autonomous personality change later** | Source identity and tracked divergence must be designed before collecting data |
| R1 A1: full-tuning comparison late/optional | **Make small-model developmental comparison a main experiment** | The additional workstation permits testing interface co-adaptation |
| Original KAL/HRL/Memory Bus/checkpoint separation | **Retain terminology and ownership** | Reusable architecture; runtime IO remains distinct from neural computation |
| Original ThoughtCore/ThoughtManifold | **Retain as explicit candidates** | Their accuracy contribution has not yet been established |
| Original PM-stream/mHC and attention replacement | **Separate controlled ablations**, not prerequisites | Large changes would obscure the cause of improvement |
| R1 independent residual readers/null path | **Retain** | Directly addresses reported fusion interference |
| Original W0–W4 hierarchy | **Retain with scoped fast-state and development modes** | Runtime gradients and global backbone edits are different permissions |
| Original all-knowledge-to-backbone sleep path | **Make selective** | Exact memories can persist as records; procedures may benefit from neural consolidation |
| Biological mappings | **Use as design analogies only** | They do not certify neural-network behavior or mandate an implementation |

### 4.1 Existing results worth preserving

The uploaded pilot report records memory recall of 0.625, a substantial ordinary-context regression when the same expanded gate is enabled, and a thought core that changed dynamics without demonstrating greater accuracy. The integration report describes thought modules outside the normal backbone forward and inquiry tool mocks. These are useful starting observations, not new R2 results [U05](05_Research_References_and_Evidence.md#u05) §3–5, [U06](05_Research_References_and_Evidence.md#u06) §1–4.

No inference is made here that the code still has those exact properties. The repository and raw checkpoints were not provided in this turn. Reconcile the reports against commits and executable tests before migration.

### 4.2 Scientific statements to revise explicitly

**Frozen weights are not frozen behavior.** New readers, policies, fast state, or skill adapters may cause interference.

**Personal identity is not a single vector.** A style direction is not an autobiography, a preference hierarchy, a counterfactual decision model, or a record of change.

**Consistency with the model’s prior is not truth.** Correct evidence can overturn prior knowledge. A source’s self-report may be authoritative about their preference without proving an external scientific claim.

**Intrinsic self-correction is neither guaranteed nor categorically impossible.** Earlier work found failures without feedback; SCoRe shows that the behavior can be trained under a particular RL setup. R2 therefore tests correction explicitly and does not use self-approval alone for promotion [R09](05_Research_References_and_Evidence.md#r09), [R10](05_Research_References_and_Evidence.md#r10).

**Arousal, prediction surprise, uncertainty, and learning value are different variables.** The original emotional-salience proposal is retained as an optional signal, not an equality between those quantities.

**No experiment here proves consciousness or unlimited cognitive growth.** The target is a measurable functional system, with useful negative results allowed.

## 5. Research-to-module mapping

| Research | Specific R2 role | Limit of the transfer |
|---|---|---|
| Engram [R01](05_Research_References_and_Evidence.md#r01) | Sparse addressable lookup and contextual fusion | Writable episodic records are an extension, not original Engram |
| Titans [R02](05_Research_References_and_Evidence.md#r02) | Fast neural adaptation within an observed stream | Not an exact autobiographical archive by itself |
| HOPE / Nested Learning [R03](05_Research_References_and_Evidence.md#r03) | Independently trained update timescales | A scheduled sleep job is not a reproduction of the full method |
| uTTT [R04](05_Research_References_and_Evidence.md#r04) | Shared fast-state ownership across selected depths | Cross-depth sharing is not cross-user sharing or reuse of the original MoE router |
| CTM [R05](05_Research_References_and_Evidence.md#r05) | Recurrent internal computation as a candidate cognitive core | Tick-RoPE in TAIS is not equivalent to neuron-level temporal synchronization |
| Coconut [R06](05_Research_References_and_Evidence.md#r06) | Learned latent iteration and staged supervision | General retrofit benefit must be demonstrated, not assumed |
| SEAL [R07](05_Research_References_and_Evidence.md#r07) | Model-proposed training data/update directives | A practical initial bounded proposal grammar is narrower than general self-modification |
| Absolute Zero [R08](05_Research_References_and_Evidence.md#r08) | Curriculum chosen through verifiable tasks | No external training set does not mean no pretrained knowledge or verifier |
| Voyager [R11](05_Research_References_and_Evidence.md#r11) | Executable, reusable procedural skill library | System-level skill gain does not by itself prove neural-weight learning |
| MAC / LongMem [R15](05_Research_References_and_Evidence.md#r15)–[R16](05_Research_References_and_Evidence.md#r16) | Learned context encoders and internal readers | Model-specific alignment and version compatibility remain necessary |
| Interview-based generative agents [R12](05_Research_References_and_Evidence.md#r12) | Source-person modeling and test–retest evaluation | Survey similarity is not evidence of consciousness or complete personal replication |
| Persona vectors / LaMP [R13](05_Research_References_and_Evidence.md#r13)–[R14](05_Research_References_and_Evidence.md#r14) | Drift diagnostics and personalization baselines | Neither alone certifies an individual’s reasoning policy |

These ingredients are not a shopping list to enable simultaneously. Each is a hypothesis-linked component with an independent control.

## 6. Architecture direction: model-native learning with externally checked execution

The inference loop is:

**Observe → maintain goal/beliefs → retrieve → reason/compare → ask or act → verify → revise → answer.**

The experience-development loop is:

**Record outcome → diagnose failure or reusable success → select learning target → propose practice/update → train candidate → evaluate independently → promote or reject → reuse later.**

The replica loop is:

**Collect source material → confirm interpretations → establish identity snapshot → answer held-out decisions → record source/replica divergence → accept only authorized identity revisions.**

The neural backbone and trained extensions generate the judgments and proposals. The host runtime carries out storage, tool execution, resource accounting, and atomic version changes. Calling that runtime does not make the learning “merely external”; the crucial questions are whether the policy is learned, whether improvements persist, and where state changes occur.

A learned guard may be a signal, but deterministic permissions and evaluation separation remain outside the candidate’s writable state. This is a software trust boundary, not a claim that a classifier solves alignment.

## 7. Two-way interaction without an opaque one-vector bottleneck

Retain a PM-style interface concept, but begin with a small explicit workspace linked to selected decoder blocks rather than changing all residual streams to n=5. Suggested initial workspace: 8–16 slots of width 256–512; 1, 2, or 4 recurrent steps. These numbers are experimental choices.

Separate typed workspace fields for goal, evidence, hypothesis, subgoal, candidate action, and uncertainty; preserve exact source references outside the latent vectors. A summary vector must not be the sole location of a code fragment or identifier needed for an exact test.

The cognitive core may read hidden states and return a gated delta or control proposal. It can also request additional observations through HRL. A separate plain recurrent MLP/GRU control must accompany CTM-inspired variants. Keep the thought-manifold projection as a representation-learning and visualization experiment; do not require semantic distance to obey Euclidean geometry without task evidence.

## 8. Development priorities

**First:** a complete simple loop, not a fully optimized memory subsystem. Use a strong text/evidence memory baseline while internal readers are trained in parallel. This baseline does not replace the model-modification goal; it lets the project measure planning, inquiry, feedback, and skill improvement immediately.

**Second:** the small-model factorial study in document 03. It distinguishes architectural benefit from ordinary full fine-tuning and determines how much co-adaptation is required.

**Third:** freeze candidate backbones and run repeated experience/learning cycles. This is where the self-improving-system claim is evaluated.

**Fourth:** train model-specific large-backbone extensions using shared task protocols and approved source evidence, not by copying incompatible latent tensors from the small models.

**Parallel from the start:** replica interviews, decision records, access policy, held-out tests, and the distinction between source-person history and replica history.

## 9. What a successful preliminary result would mean

A defensible initial result is narrow: for example, a small model learns a verified debugging strategy over a sequence of tasks, transfers it to unseen circuits, chooses clarification better, preserves unrelated scores, and reproduces the source user’s documented review priorities in held-out cases. It must outperform memory-only and fixed-curriculum controls under declared budgets.

That would support a useful full-chain mechanism. It would not yet prove human-like cognition, broad autonomous research ability, or successful transfer to both MoE backbones. Conversely, a failure should identify whether the bottleneck is representation, control, learning objective, verification, model capacity, or compute—not trigger an automatic retreat to memory-only scope.

## 10. Reference convention

`Rxx` entries link to primary research and code in [05](05_Research_References_and_Evidence.md). `Uxx` entries identify uploaded source documents and sections. This document’s architecture, experiments, and conclusions labelled as proposed are R2 synthesis rather than quotations or reported paper results.
