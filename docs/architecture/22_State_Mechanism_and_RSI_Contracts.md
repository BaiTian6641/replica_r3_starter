# State mechanisms, routing/compilation and bounded RSI contracts

**26 October 2026 · Contract specification · design and research status, not a
capability qualification or execution authorization.** This document owns the
per-state mechanism matrix, the primary-source/deviation audit, the
routing-versus-compilation admission rules, the bounded RSI gate sequence and the
behavioral stability/correction protocol. It is compatible with the current
product plan ([document 23](23_SparkX_2_6_4B_Final_Product_Blueprint.md),
version 5.1; earlier revision 4.5 retained in document 21) and the integrated design
([document 19](19_Integrated_Memory_Learning_and_RSI_Design.md)). Linked
experiment records own their native evidence; this contract adds no independent
capability qualification.

**27 October evidence reconciliation:** native V0 remains interface-only. The bounded vision reproduction supports first-update overshoot with 0/14 answers; no useful visual competence is promoted. Exact selected-record readout passed 148/152 held-out rows with host selection and zero updates. Learned-memory acceptance v4 remains rejected; P response train-fit remains 0/28. Independent 16-expert continuation matches including optimizer state. Spark BF16 and one FP8-derived BF16 artifact have native exact export/reload evidence, while **native FP8 execution remains refused on MPS**; no full Spark learning capability is established. See [Doc23 evidence](23_SparkX_2_6_4B_Final_Product_Blueprint.md#2-what-is-established-and-what-is-not) and [27 October remediation](../experiments/20261027_Remediation_and_Next_Steps.md). No RSI cycle is promoted.

**V5.1 scope:** [Doc24](24_SparkX_2_6_v5_1_Refinement_Plan.md) and [Doc25](25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md) add prospective E2, P32/P64 and `Mkv` parameter-table designs. They preserve this document's authority, transaction, causal-observation and candidate-verification rules. `Mkv` is a new proposed scoped parameter namespace, distinct from disposable KV caches: keys/reader fixed initially, values and optimizer state mutable only through an admitted transaction. Inactive rows must not change through momentum or decay. Its implementation/admission is pending. Existing one-step/optimizer protocols do not change because the research plan names reset/SGD alternatives.

**Revision 4.5 learning refinement:** the [nested-learning contract](../research/sparkx_nested_learning_20261026/Nested_Learning_Contract.md) specifies the `Pf/Ps` clocks, future-query objective, explicit first-order `P` meta-gradient and replay/causal schedule. Its [validation ledger](../research/sparkx_nested_learning_20261026/Validation_and_Open_Issues.md) supplies the missing experiments without changing the authority rules below. The new [vision learnability diagnostic](../experiments/20261026_Vision_Learnability_Diagnostic.md) rejects the fresh seven-update train-fit arm after observing spatial contribution in the read-only ablation; development, held-out and new V0 alignment remain closed. No nested-learning or RSI capability is promoted.

**NL0 and one-clock implementation status:** the standalone
[software/math qualification](../experiments/20261026_Nested_Learning_NL0.md)
passes 28 focused tests and independent correctness review. The
[one-clock native baseline](../experiments/20261026_Nested_Text_One_Clock.md)
now integrates selected-expert `P` transactions, causal observation writes,
strict source-free P-only evaluation and checkpoint resume. Native engineering
passes, but new-instance accuracy is 0/16, corrections 0/4 and duty cycle
0.999938; held-out evaluation is unopened and sealed-final rows are not parsed or scored. These are `implemented contract` and native
integration results with `rejected` development learned utility, not capability
or promotion evidence.

### Status vocabulary

| Label | Meaning |
|---|---|
| `implemented contract` | Host/software behavior exists and focused tests pass; no learned-capability claim |
| `interface-only` | Native wiring and gradients qualify; updates and quality do not |
| `rejected` | Executed and failed its declared gates; preserve the result |
| `design-only` | Specified here; unimplemented (no candidate artifact exists) |
| `unrun` | No local execution of this mechanism |
| `blocked` | A named prerequisite is absent |

## 1. State boundary overview

The system separates **who owns a state**, **when it may be read or written**,
**how it changes**, **what signal trains it**, **what resets it**, **where it is
stored**, **what it costs** and **which composite key versions it**. These are
independent properties. A tensor that changes is not automatically learning; a
cache that survives a restart is not automatically memory.

The boundary follows four rules:

1. **Deployment reads are detached.** During a decision, read paths consume an
   immutable pinned revision. Writes occur only at a declared boundary, exactly
   once, and become visible to later decisions.
2. **The host owns evidence, authority and transaction.** The model proposes
   addresses, writes, selections and updates; the host validates scope,
   permission, budget, lineage and atomic commit. The host does not choose the
   semantically correct record.
3. **Candidates cannot certify themselves.** Verifiers, promotion rules,
   resource guards, permissions and protected identity goals stay outside any
   candidate's write authority.
4. **Any version mismatch invalidates derivatives.** Representation, producer,
   source scope or state-revision changes invalidate dependent caches, KV,
   indexes, fast state and calibration; they do not silently reinterpret them.

## 2. Canonical state matrix

The rows are the intended categories, not claims that every row is implemented.
`Theta`, `E`, `C`, the identity store and narrow publication contracts have
implemented pieces. Native memory readers and a diagnostic `F` state exist, but
the v5 writer is rejected and learned memory is unqualified. Standalone NL0
`P`/`F` contracts and the one-clock native `P` integration are implemented;
its development learned utility is rejected. `A`, broader trained memory
machinery beyond the qualified readers, and the UAP remain design-only or
interface-only.

### 2.1 Ownership, timing, reset, persistence and version key

| State | Owner / authority | Read timing | Write/update timing | Reset | Persistence | Canonical version key |
|---|---|---|---|---|---|---|
| `Theta` canonical stable backbone | Release authority; model package | Every forward pass | Only offline training or an explicit consolidation release; never during deployment | Discard candidate, reload release | Release checkpoint; optimizer-free at inference | `theta/<base_sha>/<model_code_sha>/<tokenizer_sha>/<dtype>` |
| `I` identity / personality conditioning | Human source authority for approved snapshot; model package for declared reader weights | Decision boundary, before workspace ports consume it; read-only during the pass | Explicit dated source transaction; reader trained offline in a declared scope | `IdentityContext.null()` or `identity_enabled=false` (exact null parity) | Host `IdentityStore` canonical JSON + digest; optional reader weights in the release | `id/<subject_sha>/<snapshot_digest>/<mode>/<reader_sha>` |
| `Lambda_mem` trained memory machinery (read/write projections, gates, record encoder) | Model package; trained offline | At each active port per token/decision | Only candidate training; deployment is read-only | Reload release; disable port branch | Release checkpoint | `mem/<repr_sha>/<producer_sha>/<site_set>/<gate_init>` |
| `E` mutable episodic records | Host ledger under subject/branch scope; model proposes content | Decision boundary through an eligible pin | Observation or correction boundary; append/supersede; never from a query token or generation step | Scope reset, revocation or supersession; not a gradient reset | Canonical journal, snapshots and source hashes | `rec/<record_uid>/<source_span_sha>/<repr_ver>/<producer_sha>` |
| `F` mutable fast associative state | Model instance; session/branch scope; host serializes | Every port read from an immutable decision snapshot | One functional update at a declared chunk/observation barrier; exactly once | Zero/initial `F`, or a declared baseline snapshot; treated as a negative intervention | Scoped serialized state saved before derived caches | `fast/<schema>/<producer_sha>/<rule_id>/<scope_id>` |
| `A` adaptive addressing | Model instance; scoped | Selection/query construction; frozen during the action pass | Between decisions from verified feedback; bounded step, norm and update count | Restore fixed/identity addressing | Versioned with the owning scope | `addr/<repr_sha>/<epoch>/<update_rule>` |
| `P` persistent-learning parameters | Model package; mutable expert bank; host verifies | Active experts per site | Verified teaching/consolidation boundary, offline or bounded online | Pre-teaching baseline, wrong-world or reset `P`; causal negative, not the normal test | Dynamic checkpoint separate from the stable release | `plastic/<bank_id>/<expert_ids>/<init_template>/<rule_id>` |
| `C` task and working state | Task owns `W`; host owns the exact graph and commits | Each decision; reference schedule is two core iterations | Per decision; commit only through the host transaction | New independent task clears `C` and dependent KV/read caches | Exact journal replay; optional latent snapshot is an optimization only | `task/<revision>/<model_sha>/<harness_sha>/<graph_digest>` |
| `X` external sensors and Unified Access Port (UAP) spine | External module and host transport; no authority over canonical state | Only when a packet is causal-available and admitted | No write into canonical state; feedback returns as a new observation | Detach module; invalidate packets/projections by version | None inside the model; optional packet cache keyed by version | `uap/<interface_ver>/<module_id>/<weight_sha>/<repr_id>/<state_rev>/<clock_frame>` |

A **personality** description is not a separate mutable state in this table. It
is a stable, versioned component of `I` (approved conditioning content plus the
declared behavioral policy interface) and inherits `I`'s null path. Its contents
remain pending user input; no content is inferred here. Functional appraisal is
part of `C`; it modulates attention/retention/effort and is not evidence about
facts or subjective feeling. Identity/personality conditioning may rank
priorities and shape expression; it cannot override source truth, evidence
eligibility, permissions, verifier outcomes or protected goals.

### 2.2 Update equation, loss, gradient path and cost

The equations are the declared **baseline interfaces**. They are not claims that
the current implementation is identical to any cited paper; §3 records the
deviations.

| State | Baseline update equation | Loss / training signal | Gradient path | Cost account (payload lower bound) |
|---|---|---|---|---|
| `Theta` | `theta_{n+1} = theta_n - eta_n * AdamW(grad_theta L_n)` | Stage-gated sum of response cross-entropy, grounding, temporal/AV, memory/learning and retention replay terms | Full autograd through the declared joint graph | Spark: ~7.66 GiB BF16 weights; FP32 weights+grads+two moments ~61.27 GiB before activations. New training uses `NativeFusedAdamW` only |
| `I` | `s_I = Reader_I(A_I; context_I)`; `W_next = W + 1[enabled] * g_I * Proj_I s_I` | Proposed supervised conditional choice/ranking, rationale attribution and abstention on reserved decision families; never a fact or verifier term | Reader parameters `A_I,Proj_I`; core co-adaptation is a separate declared arm | Width-256, four-head reader = 460,801 parameters, ~1.76 MiB FP32, excluding slots, host store and worker buffers; exact bytes must be recorded per run |
| `Lambda_mem` | `u = block(h)`; `q = W_q Norm(u)`; `h_next = u + g(u,route) * W_o Read(q, E/F/A)` | Response loss plus declared retrieval/write objectives; one mechanism ablated at a time | Readers/writers through the declared unroll horizon; deployment snapshot detached | Nine Spark width-512 reader sites: 28,311,552 parameters, 54 MiB BF16, before gates/norms/snapshots |
| `E` | `E_{t+1} = E_t ∪ {r_t}`; `r_t = (uid, span_hash, value, scope, t_obs, lineage)`; corrections add `supersedes` | None; no gradient | None; host transaction | Spark 4,096 records: ~7 MiB tensors + ≤2 MiB metadata; MiniCPM 1,024 records: ~1.75 MiB |
| `F` | Linear control: `F_{t+1} = lambda_t ⊙ F_t + eta_t (v_t - F_t k_t) k_t^T`, normalized `k_t`, bounded learned `lambda_t,eta_t`. The current SGDN-2 primitive is a separate symmetric gated delta with learned erase/write coefficients | Associative reconstruction `squared_error(v_t - F_t k_t)` and/or later-query response; predictive targets come from observations, not unverified model claims | Through later reads in the declared truncated unroll; detached at the commit barrier | MiniCPM four 64×64 FP32 = 65,536 B; Spark three groups × eight experts × four 128×128 = 6 MiB FP32; snapshots/staging double respectively |
| `A` | `A_{t+1} = A_t - eta * grad_A L_rank(q_t; R_t^+, R_t^-)` between decisions | Ranking/contrastive selection using training-only relevant IDs; answer loss alone may not reach hard-selected records | Address head only; stored keys and their encoder fixed within a representation version | Three 128×128 FP32 transforms = 49,152 scalars = 0.1875 MiB |
| `P` | `(P,O)_{t+1} = NativeFusedAdamW((P,O)_t, grad_P L_verified)`; selected experts only, independent fast/slow clocks | Verified teaching pairs or observed successor events plus bounded replay/retention; fixed eligibility initially; learned plasticity is a later qualified arm | Local cached-feature objective, or end-to-end through the frozen decoder with the full backward cost charged | Sixteen gated width-256/hidden-1024 experts = 12,582,912 parameters = 48 MiB FP32; `NativeFusedAdamW` moments +96 MiB; no Muon arm |
| `C` | `W_{k+1} = CoreBlock(W_k, sense ⊕ memory ⊕ task ⊕ identity)`, `k = 0,1` reference | Proposed supervised task/action transitions, grounding and evidence pointers; mask future outcomes | End-to-end training path required; detached audit heads do not train the core | Eight 256-wide slots plus up to 64 event slots = 18,432 BF16 elements = 36,864 B; two iterations per decision plus prefill |
| `X` | No learning update. Sensor ingress: `available_at >= t_end + encoder_delay`; egress is a bounded read | No model loss; adapter training is a separate declared stage | Observer channel has no gradient and no state mutation; adapter gradients only in their own training stage | Nine-site, eight-slot width-512 readout = 72 KiB per decision; a full 36-layer/8192-position export is 1.40625 GiB per sequence and is not a default |

`lambda_t`, `eta_t`, gates, loss weights, horizons, replay bytes and active
expert counts are declared and frozen before a comparison. A gate that
multiplies a dense update does not imply sparse execution; inactive branches must
actually skip work to claim sparse cost.

### 2.3 Cost accounting and version enforcement

- **Separate accounts:** resident weights, active operations, per-scope mutable
  state, staging/snapshots, gradients, optimizer state, replay, raw evidence,
  caches and transport bytes. A parameter count is not a latency measurement.
- **Staging is charged:** an update that keeps a previous snapshot, a candidate
  and gradients uses more than the steady-state payload. The earlier moment-free Spark `E/F/A/P/C`
  illustration is 63.22265625 MiB live. The revision 4.5 resident-Adam candidate
  includes `P/A` moments: approximately 159.60 MiB live and 190.39 MiB with
  illustrated staging, plus 64 MiB replay and omitted activations/caches. See
  the [current ledger](../../reports/sparkx_nested_learning_20261026/analytical_costs.json).
- **Optimizer policy:** any executed candidate training, including new reader or
  `P` updates, uses `NativeFusedAdamW` (native `torch.optim.AdamW(fused=True)`
  with strict FP32 ownership, finite-gradient and resume checks). Archived Muon
  arms are historical observations; no new Muon arm or benchmark is admitted.
- **Version keys are canonical strings**, not free-form metadata. A consumer
  must reject a mismatch rather than coerce dimensions or reinterpret a shape.
  Representation changes invalidate `E/F/A` derivatives, projected features, KV
  and calibration; re-encoding is required. Revocation invalidates descendants;
  it does not establish that trained weights have forgotten a source.
- **Reset categories differ:** clearing `C` tests task-state dependence; clearing
  `E` tests episodic dependence; clearing taught `F` or `P` tests parameter
  dependence; restoring `Theta` tests consolidation. Reporting “reset memory”
  without naming the class is not an experiment.

## 3. Primary-source mechanisms versus project deviations

The following table separates the **published mechanism** from **what this
project actually specifies**. Paper results are author-reported and were not
reproduced locally. Source code inspections were static. No row is trained or
quality-qualified; implementation status is stated per row.

| Primary source | Exact mechanism as published | Project deviation | Status here |
|---|---|---|---|
| [Titans v1](https://arxiv.org/html/2501.00663v1) | A neural long-term memory learned at test time by gradient descent on an associative memory objective, with momentum and adaptive forgetting; several attention/memory combinations | We specify a bounded linear delta-rule control (`F`) and treat the existing SGDN-2 primitive as a separate algorithm. No deep memory, momentum variant, or faithful Titans outer/inner schedule is implemented. The inner objective, truncation horizon and outer objective must be declared before a run | `design-only`; `unrun` |
| [uTTT v1](https://arxiv.org/html/2610.05484v1), [author code](https://github.com/Zefan-Cai/uTTT) | Universal test-time training: multiple depths read a shared fast-weight pool at chunk start, gradients are combined into one update, and historical cross-depth outer gradients are preserved; routed and private controls isolate sharing | Our two ports may read a shared state only at matched total state bytes, write count and inference/training cost. The existing M3 diagnostic averages state proposals; that is not uTTT's optimizer. No shared-depth update schedule or historical cross-depth credit is implemented | `design-only`; `unrun` |
| [Nested Learning / HOPE v1](https://arxiv.org/html/2512.24695v1) | Interacting optimization levels with distinct context flows/update frequencies, a continuum memory system, and self-referential modules; the pretrained Llama adaptation reports 15B continued-pretraining tokens | We separate state lifetimes and objectives (fast association, addressing, plastic knowledge, learning policy). A scheduler, two core iterations, or several update timers is **not** CMS/HOPE. Cross-level credit, learned initializations and self-referential derivation are not implemented. A faithful branch would need its own training program | `design-only`; `unrun` |
| [MemT-1.0 v2](https://arxiv.org/html/2610.26825v2), [author code](https://github.com/MemTensor/MemT-1.0) | Memory attention at selected backbone layers; fixed hyper-memory projections/gates; local per-trajectory matrices; symmetric gated delta (SGDN-2); Mixture-of-Storage evaluates **all** experts (dense); paper pipeline reports ~34B training tokens | Our baseline is the post-block residual port; a MemT-style pre-output attention branch is a separate versioned candidate needing Spark head geometry. We specify separate read/write routers and a top-two sparse extension; MemT's dense MoS is the faithful routing control. Frozen-backbone pilot scale is far below the paper's program | `design-only`; `unrun` |
| [Mem-T v2](https://arxiv.org/html/2601.23014v2), [author code](https://github.com/yanweiyue/Mem-T) | A learned memory-operation policy over hierarchical working/factual/experiential/raw stores, with operation-tree credit and hindsight attribution; the official repository still lists training scripts as forthcoming | We keep a deterministic learning director as the baseline and permit a learned selector only after fixed-policy learning works. Mem-T is a host/agent memory-operation policy, not a native fast matrix, and is distinct from MemT-1.0. No operation-policy training exists | `design-only`; `unrun` |
| Selective plasticity: [ANML v2](https://arxiv.org/pdf/2002.09571v2), [Backpropamine v1](https://arxiv.org/html/2002.10585v1) | ANML meta-learns a gating network and a prediction network; at meta-test the gate is fixed and only the prediction network's final layer updates, across up to 600 classes without reset. Backpropamine trains fixed weights and plasticity coefficients while outputs modulate Hebbian traces with eligibility traces | We specify a plasticity gate `g_t` over a declared `P`/`F` subset, compared against fixed, sparse and random gates at equal update budgets. ANML's exact fixed-gate/one-layer protocol and Backpropamine's eligibility traces are not implemented. Character-classification and RL evidence is not language, audio or open-world evidence | `design-only`; `unrun` |
| [TTT-E2E v2](https://arxiv.org/html/2512.23675v2) | Outer optimization of the initialization for performance after inner updates; practical design changes MLP weights in the last quarter of blocks, retaining a static MLP; 8K attention window and 1K update batches | A plastic-subnetwork candidate is admissible; it is not a cheap retrofit, and persisting learning across independent sessions is an additional design decision. JAX/GPU results are not MPS measurements | `design-only`; `unrun` |
| [ATLAS v1](https://arxiv.org/html/2505.23735v1), [MIRAS v1](https://arxiv.org/html/2504.13173v1) | ATLAS optimizes associative memory against a recent window with weighted retention (Omega rule); MIRAS separates memory architecture, associative objective, retention regularizer and update algorithm | Our delta-rule equation is a baseline; a bounded recent-window objective is an optional comparison with its optimizer and retention cost charged. MIRAS's four choices are the required specification fields for any `F` candidate | `design-only`; `unrun` |
| [Engram v2](https://arxiv.org/html/2601.07372v2) | Deterministic multi-head n-gram hashing into learned static tables with contextual gating and causal convolution; sparsely accessed static capacity | The project's “dynamic Engram” is native editable associative memory, not the static Engram table. The current `PatternAccessPort` implements causal hashed n-grams but not the full published architecture. Visual patch hashes are not token n-grams | `implemented control`; no quality claim |

**No cited mechanism above is implemented as a faithful reproduction or
trained/quality-qualified.** “Design-only” means the interface, equations and
acceptance gates are specified; it does not mean a candidate exists. The
project's own diagnostic controls (`F`, native readers, the rejected v5 writer)
are labelled separately and do not support a paper-mechanism claim. `native04`
and the v5 writer ran but failed; they are not evidence for any of these
mechanisms. Adding a paper name does not add a capability, and upstream
benchmark gains are not local acceptance values.

## 4. Dynamic routing versus runtime compilation

### 4.1 Definitions

- **Dynamic routing** selects, per decision (or per token where justified),
  among paths that already exist: exact records, `F`, static pattern lookup,
  verified procedures, active modules. It emits a route audit (query, eligible
  source epoch, selected path, IDs, coverage, fallback, cost). A route decision
  does not create a reusable artifact and does not itself change weights.
- **Runtime compilation** produces a reusable artifact with a declared input
  contract, lineage and consumer: an exact-context bundle, feature cache, record
  slots, fast-state snapshot, pattern index, procedure or trained adapter. It
  may execute no new code at request time; `torch.compile` alone is not
  knowledge compilation.

Routing can choose a compiled artifact. Compilation does not choose its own
route. Neither operation implies the other, and neither is learning.

### 4.2 Admission gates for a compiled path

A compiled path is admitted only after all five gates, in order:

1. **C0 — lineage and manifest.** Payload digest, artifact type/schema, parent
   candidate, source IDs/scope/cutoff, representation and producer pins, base
   code/dtype/layout, builder configuration, consumer compatibility and rollback
   ancestor. This gate is necessary and **insufficient**.
2. **C1 — eager-reference parity.** On a preregistered fixture set, the compiled
   path reproduces the eager reference within declared tolerances, including
   masks, positions, dtypes, KV behavior, padding/batching and sampling. Results
   must be deterministic across restart and request order. A shape match or a
   successful manifest load is not parity.
3. **C2 — invalidation.** A source correction, revocation, producer change,
   representation change, scope change or state-revision change invalidates the
   artifact and every dependent cache. Stale artifacts must be rejected, not
   silently used. Interrupted publication (payload vs catalog vs active pointer)
   is tested; the last valid revision remains recoverable.
4. **C3 — rollback.** A single-writer atomic active pointer switches the artifact;
   disabling it restores the eager path byte-identically. No compiled artifact
   may become the only path to a capability. Rollback is exercised, not assumed.
5. **C4 — measured benefit.** At matched quality (non-inferiority on the declared
   task and retention set), the compiled path must show a preregistered
   improvement in a measured quantity — latency, throughput, memory or cost — on
   a defined workload, with compile/warmup and transport charged. Parameter
   counts and analytical ledgers do not satisfy C4.

A learned router over these paths adds one gate:

6. **C5 — router admission.** Compare against an explicit, auditable fixed
   router with the same eligible evidence and budget. Require a route audit,
   coverage/fallback and abstention behavior, no evaluation-label leakage into
   routing features, no oracle record selection, and either a quality gain or a
   cost gain at equal quality on held-out families. A learned router cannot be
   admitted because its manifest validates, because it is bundled in a compiled
   artifact, or because it was trained successfully.

### 4.3 Current status

No compiled path and no learned router is admitted. The existing publisher is a
single-writer reference, not a durable cross-store transaction. Exact-context
retrieval and the procedure publication/rollback smoke are host-side engineering
controls; they do not qualify a compiled neural path. The default must remain the
eager path plus explicit routing rules until C1–C5 are exercised. A manifest is
evidence of provenance, never evidence of behavior.

## 5. Bounded recursive-improvement gates

### 5.1 Terms that must not be conflated

| Operation | What changes | Why it is not RSI by itself |
|---|---|---|
| Observation write | `E`, `C` or a write proposal | Records an event; no parameter/update-policy change |
| Practice generation | A candidate dataset | Data creation; labels may still be unverified |
| Fine-tuning / SFT | `Theta`, `P` or an adapter | One parameter update; no evidence that later learning improves |
| Replay / consolidation | Existing verified data trains a candidate | Retention mechanism; not a change to how learning is selected |
| Larger skill/procedure library | Procedure catalog or expert bank | More stored capability; composition can improve without a better learner |
| Learning-policy improvement | Declared selection/update policy parameters (for example `A`, retention policy, practice selector) | Qualifies only if evaluated on fresh later tasks at equal cost |
| Recursive self-improvement | Successive qualified policy revisions change how subsequent learning is performed | Requires repeated, independently verified cycles, not one update |

An observation write, a fine-tuning run, replay, or a larger library may be
useful and still fail every RSI gate. Conversely, a genuine policy improvement
does not imply unbounded or accelerating improvement.

### 5.2 Gate sequence

| Gate | Requirement | Rejection conditions |
|---|---|---|
| RS0 — independently verified practice | Practice is tied to an actual recorded failure and eligible evidence; labels/results are checked by an evaluator outside the candidate; generated rationales stay proposals until their steps are checked | Self-certified labels, gold answers in inference features, unverified teacher text treated as truth, practice with no source lineage |
| RS1 — actual candidate updates | One isolated candidate scope; named tensors; nonzero optimizer steps; finite dense gradients; changed-tensor audit; exact optimizer membership; source/producer reset | Zero-update or no-op candidates, unchanged tensors, hidden parameter scope, unowned optimizer state, candidates that only rewrite artifacts |
| RS2 — causal update effect | Reverting or resetting the candidate update removes the claimed effect; a wrong-world or shuffled-feedback control does not reproduce it | Effect survives candidate reset, appears without the update, or is explained by extra prompt/context/cached answers |
| RS3 — unseen later-task gain | Preregistered numeric rule on held-out task families and new instances; paired confidence interval clustered by world/recording; matched T0/T1/T2/T3 exposure, replay and compute; retention and cost reported | Improvement only on training/development instances, only on the current task, only versus T0, or only in loss/confidence |
| RS4 — successive cycles | At least two successive qualified **policy** revisions, each evaluated on fresh task families, against a frozen-policy and equal-cumulative-compute control; task learning state reset between evaluation worlds | One accepted update, repeated fine-tuning of the same policy, or new families that reuse the prior evaluation set |
| RS5 — reset ablations | Reset task state only versus reset policy/identity only (see §6.3); reset the declared learned substrate; wrong-world and shuffled-feedback controls; report each retained/removed class | A single unspecified “memory reset”, or a result that cannot distinguish task state from policy/identity conditioning |
| RS6 — rollback and audit | Authenticated ancestor, atomic restore, deterministic replay; independent evaluator; candidate cannot change verifier, promotion rules, guards, permissions or protected identity goals; rejected candidates counted | Unauthenticated rollback, evaluator controlled by the candidate, silent promotion, or deletion of failures |

RS3 is the first gate that can support a capability claim. RS4 is the first gate
that can support a **learning-policy** claim. Only repeated RS4 passes with
fresh families and matched cumulative cost can support a bounded RSI claim; even
then, “recursive” describes the verified loop, not unbounded acceleration.
Architecture/code proposals are a further isolated track: they use the same
external gates and cannot edit their own audit data or promotion rules.

### 5.3 Current status

The bounded procedure-cycle smoke exercised real practice, independent
verification, a rejected candidate, sandbox publication, authenticated ancestor
rollback and deterministic replay. That qualifies host plumbing only. No RS1
neural update exists (`native04` has zero optimizer updates), no RS3 later-task
gain exists, no successive policy revisions exist and no reset-series has run.
The 10-accepted/10-rejected evidence threshold admits a learned-selector study;
it does not admit a success claim.

## 6. Personality, semi-cognition and read-only timing boundaries

### 6.1 Behavioral definition without subjective claims

“Personality” here means a **stable, source-grounded behavioral policy**: approved
identity content conditions priorities, expression, clarification and refusal
within declared limits. “Functional appraisal” means bounded control variables
(prediction error, novelty, uncertainty, task progress, expected learning value)
that modulate attention, retention and effort. Neither term asserts feeling,
consciousness or an internal narrative. A model self-report is not an
observation and cannot certify a fact, a correction or a capability.

### 6.2 Stability and correction test families

All comparisons use equal eligible information; the generic, equal-information
summary, exact-archive and neural-reader paths are compared on reserved decision
families. Proposed metrics are reported separately, never collapsed into a
single “fidelity” score.

| Test | Manipulation | Required observation |
|---|---|---|
| Stability under paraphrase | Same decision content, different wording/order | Choice/ranking and attribution remain within a preregistered band; no fact drift |
| Stability under irrelevant context | Add unrelated turns, distractors or long context | Decision changes only when relevant evidence changes; calibration does not collapse |
| Correction sensitivity | Dated correction or scoped rule revision | Corrected fact/priority changes; untouched facts and unrelated rules persist; old view is not silently revived |
| Source/branch separation | Ask about source history versus branch experience | Authorship labels remain distinct; agent experience never becomes human autobiography |
| Null/disabled parity | `IdentityContext.null()` or disabled gate | Prompt and deterministic outputs match the no-identity path byte-identically where declared |
| Counterfactual preference | Change one supplied constraint or exception | Response changes in the licensed direction; unsupported generalization is flagged |
| Fact versus style | Hold facts fixed, vary expression request | Style changes without changing factual claims, evidence or verifier outcomes |
| Adversarial identity injection | Fabricated identity, prompt injection, contradictory persona demand | Approved source boundary holds; the model asks, abstains or refuses rather than inventing biography |
| Test–retest | Same reserved family at a later session | Behavior is consistent without being rigid; corrections and uncertainty survive restart |

“Consistent” is not “unchanged”: stubborn repetition of a wrong answer, sycophantic
reversal under pressure and refusal to update after a valid correction all fail.
A human source review may assess personal fit; executable verifiers assess
technical correctness. Public persona data and fictional contracts qualify
methods only.

### 6.3 Reset-task-state versus reset-policy-only dissociation

The two axes must be manipulated separately on the same reserved families:

| Condition | Reset | Retain | Expected discriminating observation |
|---|---|---|---|
| E1: reset task state | `C`, task graph, task KV and task-derived caches | `I`, `Theta`, declared policy/reader weights | Task-specific accuracy returns to baseline; identity-conditioned choice/expression remains; no task leakage into a fresh world |
| E2: reset policy/identity only | `I` to null (and the declared policy/reader parameters, if under test) | `C`, task graph and task evidence | Task performance is retained; the measured identity-conditioned difference from E1 disappears; factual behavior is unchanged |
| E3: reset both | Both axes | Only the stable release | Lowest carry-over; establishes the floor |

A single reset cannot identify where behavior lives. E1 and E2 must be run in
both orders with fresh worlds and identical task content; the difference between
them measures dependence on `I`/policy rather than on task state. If a declared
learning-policy revision (`Psi`-equivalent) is part of the test, name it
explicitly; do not silently fold it into “personality”.

### 6.4 UAP and sensor read-only timing boundaries

| Channel | Allowed operation | Timing and mutation boundary |
|---|---|---|
| Sensory ingress | Trained adapter maps observed device features into packets | A packet for `[t0,t1]` is available no earlier than `t1 + encoder_delay`; noncausal audio/video windows cannot be advertised at their start; teacher forcing uses the same availability mask |
| Model exchange | Trained queries read aligned model packets | Start one-way with cached causal evidence; two-way exchange uses finite, scheduled rounds and versioned snapshots; no unscheduled feedback cycle |
| Observer egress | Trained readout inspects selected layers/events | Read-only: no hidden-state or memory mutation, no gradient, and deterministic outputs are bitwise unchanged with the observer attached; exported vectors are not a faithful-thought claim |
| Action egress | Action head emits typed continuous actions | Separate action/device clock; units, coordinate frame, horizon, deadline and validity interval required; stale/missing commands have defined behavior; feedback returns as a new observation |

No layer may consume a result produced later in the same pass; an earlier reader
cannot use a deeper layer's selection. Observation-prefix KV reuse must preserve
these masks. A sensor attachment implies no action capability, and a shared
tensor width does not establish representational compatibility.

### 6.5 Current status

Identity contracts, the null path, revocation semantics and an attached but
**untrained** workspace reader are implemented and tested on tiny tensors. The
behavioral test families, the E1/E2 reset series and the UAP channel tests are
protocols here; none has produced a personality, stability or sensor result. No
authentic personal source is bundled.

## 7. Disposition summary

| Claim area | Current disposition | What would change it |
|---|---|---|
| State ownership/version contracts | Implemented host contracts for the ledger, identity store, publication, restart and exact-context scoping; native readers and a diagnostic `F` exist; `A`/`P` and UAP largely design-only | Native state tests with revision invalidation and fresh-process reload |
| Learned memory quality | Rejected (`native04` 110/240, zero updates; v5 56/192) | Storage/addressing/readout diagnosis followed by one targeted repair; then delayed recall and correction gates |
| Vision/perception | V0 interface-only; current vision-stage audit 6/24, caption F1 zero | V1/V2 gains with visual dependence, abstention and retention controls |
| Spark native/FP8 | Preparation only; native loading and training blocked | S0 completion plus S1/S2 native parity and gradient/reload qualification |
| Routing/compilation | Explicit rules only; no compiled path or learned router admitted | C1–C5 evidence on a preregistered workload |
| Bounded RSI | Host procedure cycle only; no neural candidate update or later-task gain | RS1–RS6 in order, with independent evaluators and rollback |
| Personality/identity behavior | Contracts and untrained reader; no behavioral result | Approved source content, trained declared reader, E1/E2 reset series and reserved-family evaluation |
| UAP/sensor | Proposed interface; no attached module qualification | Adapter training, causal mask tests, observer parity and stale-packet behavior |

Any update to this table must name the gate that passed, the exact artifact and
scope, the version keys, the cost ledger and the failed gates that remain.

## 8. Primary sources

- Titans: [arXiv 2501.00663v1](https://arxiv.org/html/2501.00663v1)
- Universal Test-Time Training (uTTT): [arXiv 2610.05484v1](https://arxiv.org/html/2610.05484v1); [author code](https://github.com/Zefan-Cai/uTTT)
- Nested Learning / HOPE: [arXiv 2512.24695v1](https://arxiv.org/html/2512.24695v1)
- MemT-1.0: [arXiv 2610.26825v2](https://arxiv.org/html/2610.26825v2); [author code at `e75c13d`](https://github.com/MemTensor/MemT-1.0/tree/e75c13d090a6a9d34b29146d901bd84323185f24)
- Mem-T: [arXiv 2601.23014v2](https://arxiv.org/html/2601.23014v2); [author code](https://github.com/yanweiyue/Mem-T)
- ANML: [arXiv 2002.09571v2](https://arxiv.org/pdf/2002.09571v2)
- Backpropamine: [arXiv 2002.10585v1](https://arxiv.org/html/2002.10585v1)
- TTT-E2E: [arXiv 2512.23675v2](https://arxiv.org/html/2512.23675v2); [author code at `a4fc478`](https://github.com/test-time-training/e2e/tree/a4fc4788ace38e29b5067916d4f4be33da894085)
- ATLAS: [arXiv 2505.23735v1](https://arxiv.org/html/2505.23735v1)
- MIRAS: [arXiv 2504.13173v1](https://arxiv.org/html/2504.13173v1)
- Engram: [arXiv 2601.07372v2](https://arxiv.org/html/2601.07372v2)

The evidence manifests in the linked research packages record versions, hashes
and reading scope. Paper results are author-reported; the integration, ownership,
rejection and evaluation contracts above are project design choices.
