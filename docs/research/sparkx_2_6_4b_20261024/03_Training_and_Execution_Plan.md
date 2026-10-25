# Training, evaluation and implementation plan

This is a prospective plan. Numeric budgets and gates below are **proposed starting criteria**, to be sealed before a new experiment and adjusted only using training/development evidence. They do not change historical protocols or reinterpret rejected runs as passes. There is no new training result in this document.

## Dependency-based construction sequence

Use new `SX` identifiers so existing E1–E6, M0–M7 and V0–V4 records remain interpretable. Spark is the requested next target; retain MiniCPM5 as a control and as a cheap place to diagnose reusable mechanisms. A documented failed memory study need not delay independent Spark text parity or isolated perception. Integrated learning claims depend on a useful chosen memory path.

Reconcile SX0 with the current [R1–R3 recovery requirements](../../architecture/17_Memory_Reliability_Recovery.md): caller-bound verification, exact update/case schedule auditing and representation-preserving rebuild. Prepare Spark metadata and tiny interfaces independently; native integration follows those repairs and a sufficiently stable reusable MiniCPM V0 contract, as specified by the [existing Spark plan](../../architecture/20_Spark_X2_6_FP8_Preparation_Plan.md). Neither another blind memory arm nor successful RSI is an extra prerequisite. Reuse the [current acquisition catalog](../../08_Dataset_Acquisition_and_Curriculum.md) rather than creating a competing admission registry.

| Stage | Concrete work and trainable scope | Dependencies and deliverable |
|---|---|---|
| **SX0: evidence and data contracts** | Reconcile current verdicts; new storage/readout probes; family splits; exact visual/time schemas; no native training required | Current source-export evidence; produce new protocols and independent scorers |
| **SX1: Spark native foundation** | Separate loader/environment; tiny-model contracts; native text parity; guarded module-only and optional adapter backward | SX0 interface specification; source/tokenizer/template lock, capacity report, export/reload |
| **SX2: spatial visual alignment** | Frozen pretrained image encoder and Spark; train bridge/resampler; then optional declared Spark adapters | SX1; a model whose answers depend on the correct image, with text retention |
| **SX3: temporal perception/reasoning** | Train temporal module and visual bridge on short clips; optionally adjust vision/language adapters | SX2; timestamp/ordering/track/event evidence, prefix-causal streaming path |
| **SX4: useful adaptive memory** | Fresh linear/SGDN-2 baseline; writer/readout diagnosis; then nonlinear/Titans-inspired arm | SX0 + SX1; useful delayed text memory and correction, independently of vision |
| **SX5: multimodal core** | Train core, typed bank projections, differentiable policy and halting; retain detached monitor | SX2; image core first, temporal integration after SX3; compute-matched control comparison |
| **SX6: joint visual episodic learning** | Co-train visual/temporal memory writers/readers and selected core parameters; accepted producers initially frozen | SX3 + SX4 + SX5; observe → remove source → recall → correct → restart |
| **SX7: instructed improvement** | Verified demonstration/failure data; isolated skill/vision adapters; SFT + replay baseline | SX6 for integrated claims; one accepted/rejected cycle with unseen transfer and rollback |
| **SX8: self-directed learning** | Train proposal/practice selector from candidate outcomes; then controlled meta-policy study | SX7; ten chronological cycles against matched fixed/random-selection controls |
| **SX9: HOPE-inspired research** | Add multi-rate residual memories; optional self-referential update modules with explicit outer objective | SX4 + SX7; gain over simple equal-budget adaptation; faithful HOPE replication remains a larger project |
| **SX10: personality integration** | Compile supplied description; learn conditional choices/preferences through independent reader/adapter | File and approved labels; interface work can start at SX0; generic and text-summary baselines |
| **SX11: consolidated deployment** | Joint retention-aware co-adaptation, export, state migration, quantization qualification | Accepted SX6–SX8 components; include SX9/SX10 only when qualified; complete integrated release report |

Implementation of contracts and tiny tests can overlap; native jobs on the Mac remain serial. “Parallel workstreams” does not mean concurrent GPU training. Optional SX9/SX10 research must not obscure whether the simpler integrated model works.

## Training recipes and objectives

Every run manifest records original/new/trainable parameter names, unique counts, optimizer groups, source revisions, numeric precision, actual input lengths, loss masks, update count, data order, random seeds and complete restart state. D1/module tuning, LoRA tuning and full augmented tuning are different regimes. Do not call a run full fine-tuning if any declared original parameter was silently frozen to fit memory.

| Recipe | Training signal | Important implementation detail |
|---|---|---|
| R1 alignment | Response-only caption/description/grounding loss | Frozen decoder still needs activation gradients to train input projector; frozen weights do not eliminate backward compute |
| R2 spatial reasoning | Answer loss + region selection/localization + relation labels | Use independently checked intermediate annotations; supervise swaps, absent evidence and novel compositions |
| R3 temporal | Event/order/track losses + answer loss; optional latent prediction | Mask future frames in streaming mode; elapsed time in seconds, varied sampling rates |
| R4 fast memory | Inner association loss; outer delayed-answer/correction loss | Functional updates; gradients across historical writes; no answer-derived writes |
| R5 core policy | Successful evidence-gathering actions + downstream answer/task utility + acquisition/tick cost | Train core gradients intentionally; detached success monitor alone is insufficient |
| R6 instructed adaptation | Verified task SFT + source-balanced replay + parent-retention term | Freeze accepted parent; train candidate in a separate artifact; evaluate before publication |
| R7 learned improvement | Later-task gain after candidate update, minus forgetting/compute | Selector cannot train on final-audit outcomes; include failed candidates in cost |
| R8 consolidation | Mixed real experience, verified synthetic practice and retained general tasks | Re-encode incompatible memories; compare consolidated state with unmerged accepted adapters |

A proposed combined outer loss is `L_task + λg L_ground + λt L_temporal + λm L_delayed + λp L_policy + λr L_retention`. Enable only terms relevant to the stage, normalize them per token/example and inspect gradient magnitudes. Add more objectives only after single-task diagnostics show what is missing. A falling weighted sum can conceal complete failure on memory or temporal order.

For a candidate adapter, a useful baseline is `L_new + λreplay L_old + λKL KL(parent || candidate)` on eligible replay inputs. KL is a retention aid, not proof of retention, and should not force imitation of errors that an authoritative correction is meant to change. Compare replay-only and replay-plus-distillation. [Dark Experience Replay](https://arxiv.org/abs/2004.07211) motivates the baseline; [SDFT](https://arxiv.org/abs/2601.19897) is a separate demonstration-conditioned method to test, not a guaranteed replacement.

Start with AdamW for new small modules as the reference optimizer; compare the repository's Muon-8bit path only with identical scope, data and update budget. Keep optimizers for base development, fast inner updates and slow candidate adaptation distinct. Do not infer that an outer Muon optimizer reproduces Titans or HOPE's inner update rule.

For LoRA, begin with a measured rank such as 8 or 16 and an explicit module list. Spark uses fused QKV and a headwise gate, so conventional Llama target-name lists are wrong. QKV-all, attention-output and selected MLP targets are separate choices. Audit tied embedding/output storage, dtypes and merged/unmerged export parity. Adapter training can still require substantial activation memory through frozen blocks.

## Data program

Create five roles before augmentation: train, development, calibration, promotion, final audit. Split at scene/object identity, source video/session, document template and task-family level. All crops, frames, paraphrases, teacher outputs and generated questions derived from a source inherit its role. Perceptual duplicate checks supplement exact hashes. Reusing an exposed benchmark is acceptable for development, not as fresh audit evidence.

| Capability | Candidate data or construction | Held-out test emphasis |
|---|---|---|
| Image-language alignment | Eligible image-caption/region pairs; reviewed domain images | Image dependence across new sources and layouts |
| Compositional spatial reasoning | New executable scene generator; [CLEVR](https://arxiv.org/abs/1612.06890), [GQA](https://arxiv.org/abs/1902.09506) training splits where suitable | New objects/compositions; spatial swaps; query-only bias controls |
| OCR and fine detail | Eligible [TextVQA](https://arxiv.org/abs/1904.08920) material and generated documents/screens | Unseen text strings, fonts, layouts and tiny regions |
| Perception/correspondence | Regional annotations and multi-view examples | [BLINK](https://arxiv.org/abs/2404.12390) task families; no benchmark answer generation for training |
| Temporal perception | Timestamped clips, annotated tracks/events; suitable training split of [Perception Test](https://arxiv.org/abs/2305.13786) | [TempCompass](https://arxiv.org/abs/2403.00476): speed, direction and order beyond static frames |
| Space across time | Multiple views of scenes with known paths/positions | [VSI-Bench](https://arxiv.org/abs/2412.14171) style spatial recall; independent scenes |
| Physical event reasoning | New simulator episodes and eligible [CLEVRER](https://arxiv.org/abs/1910.01442) data | Prediction and counterfactuals with executable truth; real-video transfer reported separately |
| Continual memory | Chronological observations, delayed questions, corrections, distractors and task switches | New association families, no observation re-encoding in state-only arm |
| Instructed concept learning | User demonstrations plus independently labeled counterexamples | New views/instances after learning; contradicted and obsolete labels |
| General retention | Language, reasoning, code/tool outcomes, instruction following, abstention and prior visual tasks | Diverse unique questions and families; no four-question proxy for general retention |

Record the original distribution terms, exact split, access conditions, provenance and permitted use for each acquired dataset. A paper link is a candidate source, not an authorization to ingest all associated media. No dataset or model-weight acquisition is performed by this plan.

Verified visual programs can generate training trajectories through detection, localization and relation operations, followed by distillation into the integrated model. [Visual Program Distillation](https://arxiv.org/abs/2312.03052) supports this training direction. Specialist outputs remain predictions until checked; otherwise their errors become training targets. At deployment, evaluate the distilled model without those external teachers to demonstrate intrinsic capability.

Teacher text must not contain hidden benchmark answers. Do not treat a generated caption containing the answer as an equal-information baseline for a pixel model. Preserve labels distinguishing direct observations, user testimony, verified tool output, inferred relation and simulated/predicted event. Store source asset hashes, crop transforms, frame indices, timestamps and model producer hashes with every example.

## Bounded pilot budgets

These envelopes are proposed **experiment caps**, not promised sufficient training scales. Advance only when independent learning curves justify it. Count unique sources as well as derived examples and tokens; the same image with 100 paraphrases is still one image family.

| Tier | Suggested initial envelope | Claim allowed |
|---|---|---|
| Contract/smoke | Tiny random native models; 16–128 examples; native two-update smoke only after capacity | Correct wiring, gradients, restart and resource behavior |
| Learnability diagnostic | 256–1,024 distinct training cases; strict held-out family controls | Can a chosen component learn the specified bounded mapping? |
| First visual pilot | 10k–50k eligible images; 1k–5k short clips in a later stage; cap one campaign at roughly 1M–10M processed LM tokens after profiling | Narrow image/temporal transfer and retention, not broad VLM competence |
| Memory pilot | 200–1,000 training episodes of 2–8 observations, multiple delayed queries; 16/64/256-event length probes later | Useful scoped association and correction under declared stream lengths |
| Improvement pilot | One complete candidate cycle, then ten cycles across at least five task families | Bounded continual improvement; learned-selector or learning-to-learn claims require their own controls |
| Expansion | Additional data/compute only from an approved prospective run specification | Broader capability after fresh evaluation; no fixed token count guarantees it |

The upper ends are alternative caps, not quantities to multiply blindly. A video case may use many more tokens and decoded pixels than an image case. Profile input preparation and teacher generation too. Use a fixed pilot recipe before a larger search. A three-seed replication is appropriate for a mechanism that passes its first pilot; avoid expensive broad sweeps before discovering whether state stores the needed information.

At SX7, a starting replay mixture could allocate 40% supervised tokens to the new skill, 40% diverse retained skills and 20% correction/unknown/negative cases. Treat the mixture as a development hyperparameter; maintain actual source diversity and don't double-count shared cases. Video examples need token/pixel weighting so a few long clips do not dominate every update.

## Hardware and cost model

The current project targets an M2 Ultra with 64 GiB unified memory and later a 24 GB RTX PRO 4000 Blackwell SFF. Preserve the repository's one-exclusive-MPS-worker rule, no CPU fallback for native neural execution, 50 GiB sampled driver stop, 512 MiB sustained swap-growth limit and 15% minimum host-free threshold. CPU preprocessing and tiny contract tests remain distinct. These guards reduce risk; they do not guarantee no OOM between samples.

For `N = 4,112,079,360` unique Spark parameters, the following are arithmetic estimates, not measured residency:

| Configuration | Model/state-only estimate |
|---|---:|
| BF16 frozen language weights | 7.659 GiB |
| FP32 frozen language weights | 15.319 GiB |
| FP32 weights + FP32 gradients | 30.637 GiB |
| FP32 Adam: weights + gradients + two FP32 moments | 61.275 GiB |
| BF16 weights/gradients + two FP32 moments, no master copy | 45.956 GiB |
| Above BF16 layout plus FP32 master copy | 61.275 GiB |

Add vision, new modules, activations, gradients for new modules, optimizer workspaces, KV caches, retained inner-loop graphs, allocator overhead and export peaks. Current Muon-8bit quantizes momentum rather than weights/gradients; it could have a smaller optimizer footprint, but its auxiliary Adam groups and temporary arithmetic must be measured. A BF16 checkpoint or autocast flag does not prove a low-residency training implementation. Standard full Adam is outside the 50 GiB envelope before vision or activations.

For the official FP8 path, the repository's source metadata reports a nominal 4,449,285,120 parameter bytes, about **4.144 GiB**, before scales and runtime overhead. That is storage arithmetic, not a training-memory estimate. Qualify the specified channel-weight/token-activation scheme, input gradients and temporary dequantization/master copies. Keep unsupported MPS operations explicit; an independent BF16 arm or a named dequantized derivative is a separate experiment. Native FP8 weight updating needs its own kernels and optimizer/scaling validation.

For batch one, BF16 KV storage with four KV heads, head size 256 and true sliding-cache truncation is approximately:

`KV bytes = 2 × 4 × 256 × 2 × [9T + 27 min(T, 512)]`.

This gives **0.193 GiB at 4,096 tokens**, **0.334 GiB at 8,192**, **1.178 GiB at 32,768**, and **36.053 GiB at 1,048,576**. If the runtime stores all layers at full length, the corresponding 1M cache alone is **144 GiB**. Neither formula includes attention workspaces or training activations. The source eager softmax can require an FP32 score tensor of `16 × T² × 4` bytes: about 1 GiB at T=4,096 for one layer/batch before other intermediates. Actual kernels and graph lifetime matter.

Initial capacity ladder: text 512/1,024/2,048 tokens; one image with 64/128/256 visual tokens; then four/eight frames; then the shortest fast-memory unroll. Exercise forward, backward, optimizer step and save/reload for each declared scope. Stop after a measured limit or instability and diagnose it. Quantized inference is a later export arm; dynamic fast weights remain floating point initially. Do not assume CUDA quantized-training libraries or optimized kernels work on MPS.

Cache frozen visual features to isolate bridge/temporal development. Cache validity includes image bytes, processor, crop/time mapping and encoder version. Co-adapting the encoder invalidates the cache. Caching decoder outputs generally cannot replace the backward path needed to learn input visual embeddings. Train a teacher separately or cache its targets; do not place several large teacher models in the same guarded worker without a new capacity plan.

Budget wall time from measured throughput: `processed_tokens / measured_tokens_per_second`, multiplied by epochs/replays/inner and outer passes where applicable. For illustration only, 10M processed tokens take 27.8 hours at 100 tokens/s or 277.8 hours at 10 tokens/s. Neither speed has been measured for this system. Add preprocessing, evaluation, generation and failed candidates. A local pilot is feasible to investigate; reproducing billion-token architecture studies may require a separate accelerator budget.

## Evaluation gates

Seal concrete datasets, scorer versions, split membership, thresholds and stopping rules before native runs. The following suggested targets are intentionally demanding for a controlled pilot. They are not literature benchmark scores. If a task's difficulty requires different thresholds, decide that on development evidence before opening promotion/audit results.

| Gate | Proposed acceptance criteria |
|---|---|
| G0 software/source | All applicable contract tests pass; clean source package imports; disabled branches match native output under identical precision; source/crop/time tampering is rejected; checkpoint and restart state complete |
| G1 native foundation | Fixed text parity in uncached and supported cached modes; padding/window-boundary tests; finite gradients and intended changed tensors; guarded measured capacity; no native-quality claim based on tiny overfit |
| G2 spatial perception | On a fresh balanced image-dependent pilot: ≥80% answer accuracy, ≥15 percentage-point advantage over no/wrong-image controls, and ≥80% grounding success at IoU≥0.5 where a unique box is labeled |
| G3 temporal reasoning | ≥75% on balanced order/direction/event cases; ≥10-point gain over single-frame/bag-of-frames controls on the predeclared temporal subset; declared track identity and interval-IoU metrics; no future-frame leakage |
| G4 memory | ≥90% recall, corrected-fact and unaffected-fact accuracy on each controlled category; ≥20-point gain over disabled/shuffled states; ≥95% unknown handling; persistent multi-query/restart success |
| G5 core utility | ≥3-point task-success gain over best no-core/feed-forward control with positive paired 95% interval; compare at matched compute and also report actual latency/cost; chosen operating point meets declared latency cap |
| G6 integrated visual memory | G2–G5 for the supported task scope; original visuals removed; multiple later queries, corrections, occlusions and restart; correct source/time attribution ≥95% on auditable cases |
| G7 candidate improvement | ≥5-point new-skill gain over parent on disjoint families, positive paired 95% interval; broad retention mean drop ≤2 points and each predeclared critical family ≤5 points; no new failure in must-pass contract cases |
| G8 learning-policy improvement | Ten chronological cycles; final learned-selector arm exceeds fixed and random selectors at equal eligible data/update/total cost, with replication and paired uncertainty; separate learning-to-learn metric required for RSI |
| G9 deployment | Compatible complete export, isolated sessions, quantized-versus-reference checks if used, source correction/rebuild, rejection/rollback drill and resource limits pass |

A state-only compression arm need not beat exact retrieval to demonstrate a mechanism. To select it for deployment, require a measured advantage in latency, memory, privacy scope or offline operation at acceptable accuracy. Compare the accuracy/cost frontier against exact retrieval, not only weak disabled controls. If exact retrieval remains better, deploy it as a separately qualified fallback and preserve the failed neural-memory claim as failed.

Apply retention bounds against both the immediate parent and a fixed pre-campaign reference, and retain tests for every previously accepted skill. A two-point allowance per cycle must not silently become twenty points of cumulative degradation over ten cycles. Exclude explicitly corrected obsolete labels from preservation targets through a recorded source revision, not through post-hoc deletion of inconvenient failures.

For every learned branch, use at least 100 independent scene/episode families in a promotion pilot where data permits, with sufficient examples per category. Sample size must support the desired effect: a fixed 100-family minimum is not a power calculation. Report family-cluster bootstrap confidence intervals, paired differences, per-family failures and raw denominators. Use three seeds for the shortlisted mechanism. If uncertainty includes no gain, defer rather than selecting whichever seed looked best. Correct for multiple candidate comparisons or reserve a final untouched audit.

Grounding metrics need tolerance definitions, frame coordinate conventions and valid alternatives. Counterfactual correctness requires a source of truth for the intervention. Natural-language judges should be calibrated against human/executable checks; report disagreement and do not let a candidate self-grade. Calibrate confidence using held-out calibration data, not promotion labels.

### Minimum ablation matrix

| Question | Required comparison |
|---|---|
| Is Spark better for this application? | Native Spark versus retained MiniCPM5 on the same task, information, reasoning-token and wall-time budgets; separate capacity from quality |
| Does vision supply evidence? | Correct pixels versus absent/wrong/shuffled image; caption-only and OCR-only as separate baselines |
| Does spatial structure matter? | Patch/region coordinates and relations versus pooled features, coordinate shuffle and identical text with answer-changing visual edits |
| Does time matter? | Full timestamped clip versus single frame, bag of frames, shuffled/reversed order on valid challenge cases |
| Is memory writing useful? | Frozen initial/zero/shuffled/last-event state, correct state, raw features, uncompressed slots, exact retrieval and full context |
| Does the update rule matter? | Linear, SGDN-2, nonlinear without momentum, nonlinear with momentum/forgetting, equal-state/equal-compute comparisons |
| Is shared depth useful? | One port, two private ports, shared bank with normalized writes; equal state size and active compute |
| Does the core reason usefully? | No core, deterministic policy, matched feed-forward, recurrent fixed ticks, adaptive halting; same evidence access rules |
| Is learning durable? | No update, context-only demonstration, exact episodic storage, SFT+replay, proposed candidate learning; restart and source removal |
| Is self-selection better? | Fixed curriculum, random eligible selection, learned selection with the same candidate recipes and budget |
| Does multi-rate adaptation help? | Single-rate versus multiple rates at equal updates/state/compute; adaptation coefficients fixed versus learned |
| Is personality learned? | Null, approved summary, source retrieval and neural conditioning with equal information; facts and task success held constant |

Run targeted ablations stage by stage rather than every cross-product. The final integrated model needs selected interaction ablations—particularly vision×memory and core×learning—to show that each contributes after joint tuning. Match information and total work; a model that sees extra teacher labels or more frames is not an architectural comparison.

## Measuring continual and recursive improvement

Use chronological task streams. At cycle `k`, evaluate on a hidden next-task set before adaptation, collect permitted new experience, train a candidate, and evaluate again on a separate hidden transfer set. Audit labels never enter the model's replay buffer. Keep inference-only observation adaptation and optimizer-based slow learning as distinct evaluation modes.

Record an accuracy matrix `A[i,j]`: performance after learning through cycle `i` on task family `j`. Report average accuracy, forward transfer relative to the original model before learning family `j`, backward transfer `A[K,j] - A[j,j]`, and forgetting `max_i A[i,j] - A[K,j]`. Also report new-task sample efficiency, compute efficiency, calibration, correction recovery, unrelated-fact retention and catastrophic worst-family drops.

Use these campaign arms:

| Arm | Persistent experience | Parameter learning | Selection |
|---|---|---|---|
| T0 | None beyond fixed current context | None | Fixed inference policy |
| T1 | Eligible exact/episodic records | None | Fixed retrieval |
| T2 | Same eligible records and replay pool | Fixed verified candidate recipe | Fixed curriculum |
| T2R | Same | Same recipe | Random eligible selection |
| T3 | Same | Same recipe | Learned proposal/practice selector |
| T4, later | Same information budget | Qualified multi-rate/update-rule learner | Learned selector |

If T3 receives more data than T2, report the difference and rerun a matched comparison. Account for CPU preprocessing, GPU training, model-generated reasoning, teachers, verifiers and rejected candidates. T3>T2 shows benefit from learned selection under that protocol; it does not alone prove the update algorithm improves recursively.

For a stronger learning-to-learn test, freeze the original and post-campaign learners and give both a new disjoint sequence with the same demonstration and compute budgets. Compare the area under accuracy-versus-examples and accuracy-versus-compute curves. Improvements must survive resetting task-specific fast state and withholding earlier task answers. Require better later acquisition, broad retention and replication before using “recursive self-improvement” as a result label. Ten cycles can support a bounded result; they cannot establish acceleration indefinitely.

Source revocation or a bad update triggers manifest rollback and rebuilding dependent memory from eligible evidence. Include a deliberately poor candidate to prove rejection, and a post-publication fault drill to prove restoration. Accepted weights, fast states and visual producers must refer to compatible versions. The candidate cannot rewrite the scorer, hidden split, promotion criteria or resource guard.

## File-level implementation map

These paths are **proposed future files**, not files created by this research task.

| Target | Responsibility and initial validation |
|---|---|
| `src/replica_lab/architecture/spark.py` | Native Spark adapter; complete-block readers, fused-QKV/rotary/cache parity |
| `src/replica_lab/architecture/spark_loading.py` | Pinned isolated source loading and parameter/storage audit; preserve current MiniCPM loader |
| `src/replica_lab/vision/contracts.py` | Multimodal batch, spatial coordinates, source asset and event-time contracts |
| `src/replica_lab/vision/encoder.py`, `projector.py` | Processor-bound image tower, patch/region extraction, Spark/core projections |
| `src/replica_lab/vision/temporal.py` | Causal frame/track/event module, variable frame rate and time masks |
| `src/replica_lab/architecture/multimodal_core.py` | Typed banks, recurrent update, differentiable policy, halt/inspect actions |
| `src/replica_lab/memory/neural_memory.py` | Functional nonlinear fast state and momentum; finite-difference tiny tests and historical gradients |
| `src/replica_lab/memory/multirate.py` | Explicit parameter levels, buffers, schedules and optional update learner |
| `src/replica_lab/learning/candidate_cycle.py` or a non-conflicting package name | Isolated trainer/evaluator/publisher coordination; note existing `learning.py` before choosing package layout |
| `src/replica_lab/replica/personality_encoder.py` | Approved description/episode encoding and independently gated conditioning |
| New `scripts/qualify_spark.py`, `train_visual_alignment.py`, `run_visual_memory.py`, `run_continual_cycles.py` | Stage-specific runs with predeclared protocols and complete manifests |
| Matching `tests/test_spark_*`, `test_visual_*`, `test_neural_memory_*`, `test_continual_*` | Meaningful causal, ownership, gradient, provenance, resume and negative-candidate contracts |

Extend current immutable frame/contracts with a versioned modality-aware schema rather than changing old serialized meanings in place. Keep exact-span hashes and add pixel/crop/time provenance. Update extension export to carry all required new state. Do not broaden the current MiniCPM loader's architecture checks to silently admit Spark.

## First concrete implementation packages

1. **Evidence and native parity package:** prepare SX0 probes and fresh splits; isolate Spark dependencies; implement a tiny Spark adapter and parity tests; produce parameter/cache/attention cost reports. Exit with a reproducible source package, not a learned-capability claim.
2. **Bounded perception package:** pin one encoder/processor, implement visual batch contracts and bridge, validate loss masks and image gradients, then run one capped alignment pilot and image-dependence controls. Memory/core/personality injections remain off for this isolated experiment.
3. **Memory diagnosis package:** run storage/readout probes on fresh families; choose one repair hypothesis; train a matched baseline; only then compare nonlinear memory. Record a negative result if storage remains unusable.
4. **Temporal and core package:** qualify timestamps/causality and supervised inspection; train short-video relationships; measure core versus matched controls.
5. **Integrated teaching package:** demonstrate the unfamiliar-object episode described in the architecture report, then one instructed adaptation and one rejected-candidate drill. Expand to the ten-cycle campaign only after those work.

Engineering estimates, assuming one experienced developer/researcher and reusable infrastructure, are roughly 1–2 weeks for the first parity/contracts package, 2–4 weeks for a narrow visual/temporal prototype, and 2–4 additional weeks for useful memory/core integration if the mechanism learns. Continual-learning and HOPE studies need an additional research phase with no reliable fixed completion date. These are planning ranges, not delivery commitments; access to suitable data, observed training speed and negative results can dominate the schedule.

The release report should state supported tasks, unsupported cases, per-stage evidence, model/state sizes, measured device costs, all rejected candidates, complete ablation results and known retention regressions. A model card should explicitly separate observation adaptation, episodic storage, slow parameter updates, self-selection and any demonstrated learning-to-learn result.
