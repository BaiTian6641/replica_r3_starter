# 03 — Small-model full fine-tuning and preliminary evaluation

**TAIS Obsidian R2 · 22 October 2026**

All budgets, hyperparameters, and promotion margins below are proposed starting points. They are not measured results or requirements proven by the cited papers. The user has a separate small workstation for preliminary 1B–2B-class full-fine-tuning experiments; its current exact hardware is not specified. Inventory it rather than assuming it is the older Blackwell machine described in the July files.

## 1. Why full fine-tuning belongs near the beginning

A frozen backbone may not understand a newly introduced latent workspace, memory code, or control signal. Training only a tiny output projection can make a good idea look ineffective when the actual issue is interface alignment. Conversely, a full-tuned modified model can improve simply because it received useful task data, not because its new component helped.

The laboratory must answer both questions:

**Does the new architecture help beyond ordinary task training?**

**How much backbone adaptation is required before it helps?**

This changes R1’s full-tuning comparison from a late optional control into a core development experiment. It does not require continuous full fine-tuning of the deployed 26B/35B model.

## 2. Model roster and exact scope

| Model | Laboratory role | Deployment research role |
|---|---|---|
| `openbmb/MiniCPM5-2B` | Primary text reasoning/control and full-tuning study | First end-to-end loop; dense reference |
| `openbmb/MiniCPM-V-4.7-1B` | Hybrid-state transfer; first text, then visual tasks | Small multimodal prototype |
| `google/gemma-4-26B-A4B-it` | No routine full training planned locally | Larger ordinarily frozen backbone with freshly trained interfaces |
| `openbmb/MiniCPM-V-4.7-35B-A3B` | No routine full training planned locally | Larger hybrid MoE/multimodal test |

Official model/card references: [R17](05_Research_References_and_Evidence.md#r17)–[R19](05_Research_References_and_Evidence.md#r19). Gemma’s official name is A4B. Keep all four requested models; do not silently switch to a newly available smaller or base edition.

MiniCPM5-2B’s card reports **2,516,756,480 total parameters**, not exactly two billion. Use the actual loaded tensor inventory for training memory. For MiniCPM-V, report text, vision, projector, embedding, and new-component counts separately.

Distinguish:

- `full_text`: all text-decoder and text embedding/output parameters participate; vision may remain frozen.
- `full_vlm`: all original model groups are eligible and receive appropriate multimodal gradients.
- `full_augmented`: full original scope plus the new cognitive/memory modules.
- `modules_only`, `lora`, and `selective`: exact parameter allowlists are required.

A `requires_grad=True` flag is not proof that every modality was trained. Export gradient/update counts by group. Text-only data with an inactive visual path cannot establish full-VLM adaptation.

## 3. Core experimental arms

### 3.1 Required development comparisons

| ID | Architecture | Adaptation | Purpose |
|---|---|---|---|
| B0 | Original model, normal reasoning/tool interface | None | Clean capability and calibration baseline |
| B1 | Original + strong source-text memory and same tools | None | Whole-system baseline without new neural modules |
| D0 | Original architecture | Full fine-tuning | Ordinary task-data/coaching benefit |
| D1 | Reader + workspace/action core | New modules only, base frozen | Feasibility under strict base freeze |
| D2 | Same added architecture as D1 | Full augmented fine-tuning | Developmental co-adaptation and architecture benefit |
| D3 | Same added architecture | LoRA + full new modules | Lower-cost route to the D2 benefit |
| D4 | Original architecture | LoRA | Matched parameter-efficient control |

B0/B1 are evaluation references, not budget-matched training arms. D0–D2 are the minimum first scientific comparison; D3/D4 follow if resources require staging. A full-tuned arm is a comparator, **not a guaranteed upper bound**.

Keep train/validation/test episodes, evidence, tool permissions, target outputs, and optimization opportunities aligned. Search each regime’s learning rate fairly rather than forcing one regime’s optimal rate on all others. Report both equal-token and equal-total-compute comparisons; they cannot generally be identical.

### 3.2 Cognitive mechanism controls

Add controls only after the first matrix is stable:

- Added feed-forward capacity without recurrence, matched to workspace/core parameters.
- Ordinary reasoning with a matched inference-compute/tool budget.
- Fixed versus learned action policy.
- Text-only memory versus internal memory with the same selected source records.
- Private versus shared fast state with matched total state and active compute.
- One versus multiple update frequencies with matched state capacity.
- Generic behavior versus persona-summary, archive-retrieval, and trained replica conditioning.

Use two evidence comparisons: **equal information** (same source evidence available) and **equal resource budget** (bounded tokens, state bytes, runtime, and tools). Do not attribute a gain solely to latent representations if one system receives substantially more evidence.

## 4. Developmental curriculum

### L0 — infrastructure and tiny overfit

Use 64–256 tiny episodes to verify loss masks, action labels, writer gradients, full/partial update scopes, native caches, and serialization. Overfitting this set proves plumbing, not generalization. Establish original-model outputs before any wrapper or tokenizer change.

### L1 — task-dependent evidence and control

Suggested initial corpus: 10k–30k varied episodes and separate development/promotion/audit partitions. Tasks include uncertain questions, hidden rules revealed by permitted observations, exact-fact retrieval, temporal correction, tool choice, and a short multi-step plan.

Every episode provides chronological observations, permitted actions, externally checkable results, and later held-out queries. Remove the support text from relevant delayed-memory conditions. The writer cannot see future questions or answers.

Train the added interfaces briefly while preserving the initial base, then run D0/D1/D2 from explicitly recorded parents. Count any warmup tokens in total budgets. A plain D0 control receives equivalent useful task data; do not give only the augmented model extra supervised coaching.

Suggested pilot total per trained arm: 8–32M processed tokens, then 50–200M only if learning curves justify escalation. Record support, target, replay, and generated practice tokens separately. These are exploration budgets, not asserted sufficient convergence budgets.

### L2 — cognitive supervision and co-adaptation

Use externally checkable task results plus selected intermediate **state/action labels**: missing variable, chosen tool, completed subgoal, violated constraint, or evidence needed. Such labels are more directly testable than requiring a model to reproduce a verbose private reasoning transcript.

First train action selection by supervision. Add limited policy optimization only after reward and retention measurements are reliable. A task policy may receive cost penalties, but accuracy and unsupported-action rates must be reported separately so cost savings cannot hide degraded capability.

Proposed starting outer learning-rate search:

| Group | Candidate rates |
|---|---|
| Existing base during short full adaptation | `5e-6`, `1e-5`, `2e-5` |
| New reader/workspace/core | `1e-4`, `3e-4` |
| Optional LoRA | `1e-5`, `3e-5`, `1e-4` |
| Calibration heads | Separate supervised sweep on held-out calibration data |

Begin with BF16 compute where supported and FP32 trainable master/optimizer state, microbatch 1, sequence 512–1024, and gradient accumulation to a measured 8–32 episodes. Use gradient clipping and a brief warmup. Increase length only after profiling. These settings are not the original papers’ recipes.

Use a short reference run without meta-learning before introducing differentiable fast updates. Exact two-chunk meta-gradients are the reference; first-order/truncated variants are separate experiments.

### L3 — freeze the developmental checkpoint

Select candidates using development results and predeclared retention conditions. Freeze the chosen base revision; archive the original pretrained checkpoint separately. Re-encode eligible memories and recalibrate KAL under the frozen developmental revision.

This is the boundary between “we trained a new interface” and “the system improved itself after deployment.” Subsequent gains must not be credited to silent further full-base updates.

### L4 — repeated post-freeze learning

Run ten initial cycles of bounded new experience, candidate construction, evaluation, promotion/rejection, and later tasks. Ten is a manageable pilot length, not evidence of indefinite stability. Extend to 30–50 cycles only after the instrumentation and error bars are meaningful.

Include recurring skills, distribution changes, misleading evidence, no-improvement periods, and at least one deliberately harmful candidate that must be rejected. The loop should sometimes decide not to update.

## 5. Two preliminary domains

### 5.1 Verifiable micro-engineering tasks

Use the source plans’ digital-engineering direction: small finite-state machines, bit-vector transformations, counters, protocol fragments, debugging, or code transformations. Begin with problems that are neither trivial nor consistently unsolvable for the small base model.

Episode example (synthetic, not a user biography): a specification omits reset semantics; the model must ask or inspect a permitted source, implement a small module, inspect the verifier failure, repair it, and later solve a different-width or different-state variant without copying the original answer.

The learning artifact might be a reusable boundary-case test strategy, a verified helper, or a neural skill adapter. Hidden test cases, parameterized generators, and where appropriate independent equivalence checks should prevent memorizing visible tests. A model-generated testbench alone is not independent truth. A tool’s claim is limited to its semantics and tested assumptions.

For the first slice, a deterministic bit-vector/finite-state oracle is easier to audit than unrestricted repository modification. Add real simulator workflows after the simpler task generator passes independent tests. Primary implementation references are linked in [05](05_Research_References_and_Evidence.md).

### 5.2 Personal decision and explanation tasks

Use consented decision episodes from document 04. Evaluate which information the replica requests, which constraints it prioritizes, which alternatives it rejects, how it changes its decision under a counterfactual, and whether it admits unknown biography.

Do not give “the answer the user would choose” to the runtime writer before evaluation. A user decision is a fidelity label, not necessarily an external correctness label. Report both dimensions separately where a technical decision has verifiable consequences.

## 6. Longitudinal experiment arms

Start every arm from the same developmental checkpoint and approved initial archive.

| ID | What changes after each cycle? | What it tests |
|---|---|---|
| T0 | Nothing persistent | No-learning baseline |
| T1 | Evidence/episodic memory only | Acquisition without strategy/parameter learning |
| T2 | Fixed scheduled training on verified replay | Benefit of ordinary continued training |
| T3 | Autonomous target selection + bounded candidate updates | Value of the self-directed learning policy |
| T4 | T3 + selected fast/shared/multi-rate mechanism | Additional online adaptation benefit |

T1–T3 receive the same experience opportunities and allowed tools. When T3 selects data differently, report selected data and a replay-matched T2 control so gains can be separated into data selection versus update method. All arms receive matched **evaluation** compute. Report extra **learning** compute explicitly.

Self-generated questions need validity checks and held-out task-family diversity. Do not let the curriculum converge to easy examples that yield a high training success rate but no transfer.

## 7. Metrics and acceptance

### 7.1 Primary learning measurements

Let `S(i,j)` be performance after cycle i on the held-out task family introduced at cycle j. Report the full matrix, not only the newest task.

- Current competence and backward retention across old families.
- Forward transfer to unseen families/instances.
- Learning gain per verified example, processed token, tool call, and measured compute.
- Post-restart performance and reset-state controls.
- Promotion, rejection, rollback, and false-promotion rates.
- Calibration, useful clarification, unnecessary-tool, and abstention behavior.
- Replica fidelity, invented-autobiography rate, and source/agent misattribution.

A simple forgetting statistic is the average over prior families of `max_previous_score(j) - current_score(j)`, with uncertainty. Report both macro family averages and individual critical failures. Do not interpret a smooth confidence curve as a learning curve.

### 7.2 Proposed gates

**Mechanism gate:** identity, ownership, causal masking, checkpoint/recompute, restart, and permission tests pass.

**Task gate:** the candidate shows a reproducible held-out gain or useful success/compute improvement against its relevant matched control. Choose a smallest meaningful effect before running. Do not require every component to win; retain the simpler system if it does not.

**Retention gate:** proposed initial budgets are at most 1 percentage point average general-score regression and 2 points on any predeclared critical task. Use confidence intervals; insufficient resolution means inconclusive, not a pass. A different budget can be justified before results are inspected.

**Learning-loop gate:** after freezing the base, T3 improves unseen performance beyond T1, and preferably beyond T2 at a declared resource budget. Otherwise the evidence supports memory or ordinary training, not useful autonomous learning selection.

**Replica gate:** no invented biography accepted as fact; correct branch/source attribution in the controlled test suite; meaningful improvement over persona-summary and archive-only baselines on held-out decision behavior. No universal “90% you” score is claimed.

**Promotion gate:** all hard invariants pass; task benefit does not compensate for privacy violations or corrupted identity lineage. The evaluator returns reject, accept, or inconclusive.

### 7.3 Statistical protocol

Use at least three training seeds for promising variants, paired task instances, and bootstrap units at conversation/task-family level. Initial one-seed screens are explicitly exploratory. Large numbers of near-duplicate questions are not independent evidence.

Separate train, validation, calibration, learning-policy reward, promotion, and final audit data. A meta-training support/query split is internal to training and does not replace an external held-out test. Keep all derived versions of one source event, task template family, or human decision in the same partition.

Do not repeatedly choose candidates on the sealed final audit set. Log every evaluation access. The final report includes failed runs and selection history.

## 8. The small workstation and main workstation

### 8.1 Hardware roles

**Lab-S:** the additional workstation, with exact GPU/RAM/backend to be inventoried. It runs small full-tuning/reference experiments, independent seeds, and replay/quality evaluation.

**Main workstation:** the user’s two 24 GB RTX 3090 Ti plus two 24 GB TITAN RTX, 128 GB DRAM, and 256 GB installed Optane. Use the 3090 Ti pair first for trainable components and larger-model work; treat heterogeneous multi-GPU placement as a separately validated optimization.

The older documents name an RTX PRO 4000 Blackwell SFF and a separate display card [U08](05_Research_References_and_Evidence.md#u08). That historical inventory is not automatically the new Lab-S profile. Do not copy an old GPU index or CUDA command blindly.

### 8.2 Full-tuning memory accounting

For common mixed-precision/FP32-state AdamW implementations, plan roughly **16–20 bytes per trainable parameter** for weights, gradients, masters, and optimizer states, then measure the implementation. Frozen groups require stored weights but not those full optimizer states.

| Trainable parameter count | 16 B/parameter | 20 B/parameter |
|---|---:|---:|
| 1.0B | 16.0 GB | 20.0 GB |
| 1.5B | 24.0 GB | 30.0 GB |
| 2.0B | 32.0 GB | 40.0 GB |
| MiniCPM5 exact total, ~2.5168B | 40.27 GB | 50.34 GB |

These are decimal GB, **training-state arithmetic only**, not measured VRAM. Activations, latent unroll graphs, logits, communication, temporary copies, and allocator headroom are additional. Count tied parameters once. Optimizer sharding/offload changes residency rather than total mathematical state size.

A 24 GB GPU is not automatically enough for conventional resident AdamW full tuning of a branded 1B/2B model. It can still be a usable laboratory with training-capable CPU optimizer offload, small microbatches, checkpointing, and sufficient host memory. DeepSpeed documents CPU optimizer-state/computation offload [R22](05_Research_References_and_Evidence.md#r22). Validate the exact model and nested-gradient needs; an inference `device_map='auto'` is not a training solution.

For quality-first controls, prefer FP32-state AdamW with offload before introducing optimizer quantization as an additional variable. Placement/checkpointing can preserve the intended objective, though floating-point ordering and random-state handling still require comparison. Eight-bit optimizer trials are legitimate separate arms, not automatically identical updates.

### 8.3 Measure before scaling

Record a successful forward/backward/update, warmup, and a representative fixed-step profile. Capture every GPU’s allocated/reserved peak, host RSS, pinned-memory usage, transfers, and actual support/target throughput. Host memory must cover optimizer copies and staging, not merely checkpoint size.

If a run does not fit, first reduce activation pressure and add appropriate offload/sharding. Do not quietly freeze layers and continue calling it full fine-tuning. A changed scope becomes a new named arm.

Optane mode must be inventoried on the main machine. Do not add installed DRAM and PMem as though all bytes were necessarily OS-visible or equally fast. A persisted tensor also needs a consistent saved optimizer/snapshot manifest.

## 9. Interpreting the first results

| Result | Interpretation and next action |
|---|---|
| D2 > D0 and D1 works similarly | Architecture is useful without extensive co-adaptation; prioritize frozen large ports |
| D2 > D0, D1 weak, D3 close to D2 | Co-adaptation matters; LoRA/new-module training may be a practical large-model route |
| D2 > D0, D1/D3 weak | Architecture may need deeper alignment; test selective unfreezing/distillation before claiming large frozen compatibility |
| D2 ≈ D0 and both beat B0 | Training data helped; added components have not earned their cost |
| More ticks increase confidence but not accuracy | Do not promote the thought core; inspect objective and compute controls |
| T3 ≈ T1 | Evidence supports memory acquisition, not the intended strategy-learning claim |
| T3 > T1 but T3 ≈ T2 | Learning updates help; autonomous target selection has not yet added value |
| Replica sounds familiar but chooses poorly | Style imitation succeeded; decision-process fidelity did not |
| Current tasks improve while old skills/identity degrade | Reject candidate or tighten scope/replay; do not hide the tradeoff in a composite score |

A small-model mechanism success transfers as a **hypothesis and recipe**, not as model weights or latent records. Run new interfaces and calibrators on each large model. Retain the complete task/evidence/provenance protocol so scaling results remain comparable.

## 10. Preliminary report deliverable

Use `templates/preliminary_report.md`. Include code and model hashes; training scope and actual changed tensors; data partitions; selected hyperparameters and trials; all baseline scores; learning curves; full longitudinal matrix; replica metrics; memory/time profiles; invalidated runs; failures; and the next discriminating experiment.

The earliest useful publication-quality claim is a restricted measured mechanism, not “a self-aware model.” The earliest useful engineering milestone is a complete, restartable, validated learning loop, not a directory full of separately passing modules.
