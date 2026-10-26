# MiniCPM5 validation and Spark product release specification

25 October 2026. Proposed future experiments supporting [architecture revision 3](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md). This is not an execution protocol. Current rejected/inconclusive verdicts remain unchanged. No native experiment was launched by this review.

## 1. Admission and evidence levels

Keep four levels distinct: CPU contract correctness; native device/gradient/serialization correctness; held-out learned capability; integrated product qualification. A pass at one level cannot replace the next. Spark backend/source qualification may proceed independently of neural RSI, while integrated capability claims require their own component evidence.

The current immediate work remains vision priors/abstention analysis and **frozen-writer, non-oracle** memory-policy diagnosis. A target-only, prefix-through-target or target-refresh-last intervention is an explanatory upper bound, not an admissible policy. Data generation, raw scoring and existing sealed tests are not silently replaced. The existing vision curriculum requires at least 60% accuracy, 15 percentage-point margins versus wrong/shuffled/no-image controls, at least 70% unknown accuracy and its declared false-answer/nonregression conditions. A weaker threshold in a new research table cannot retrospectively qualify it.

## 2. Contract and native qualification

Before learned evaluations, verify:

| Area | Required checks |
|---|---|
| Parameters | Exact checkpoint/geometry, unique tied parameters, optimizer ownership, finite updates, state sizes, per-family changes and frozen-tensor equality |
| Extension neutrality | Disabled/zero-gated ports preserve baseline outputs within the preregistered numerical tolerance |
| Checkpointing | No duplicate memory writes, audio events or vocabulary swaps under recomputation; deterministic commit after accepted update |
| Media | Sample/frame timestamp monotonicity, padding/masks, crop transforms, missing modality, silence, aspect ratio, variable frame rate and declared lookahead |
| State | Stable identity, generation-tagged lexical slots, revision ownership, dependency revocation, cross-session isolation and bounded quotas |
| Restart | Fresh-process reload with no undeclared source re-encoding; identical source/representation versions; reject incompatible cached state |
| Full FT | Every original backbone parameter is trainable and owned by the optimizer, with exercised gradients and actual updates; no frozen embedding/norm/gate exceptions hidden by a “full” label |
| Resources | One native worker; actual driver/host/optimizer/cache allocation; no CPU fallback or unbounded prefetch/replay queue |

For audio, qualification begins with the pretrained non-causal short-clip path. Streaming is a separate experiment: fix chunk/overlap/lookahead, compare the same clips offline, and prohibit future chunks from affecting earlier outputs. The timestamp of an output must include capture wait, batching, encoding and queue delay.

## 3. Individual component experiments

### Memory

Hold writer weights and source vectors fixed. Randomize write order independently of queries; test early/middle/late targets, equal-frequency distractors, corrections, multiple queries per state and capacity overflow. Compare current shared/per-depth fast storage, explicit record bank, uniform causal refresh and learned causal refresh under equal bytes/work. A refresh policy may inspect past observations and the current query when declared, but never future questions or correct answers.

Report addressing accuracy, readout rank/margin, novel-value correctness, correction precedence, unaffected facts, false recall and age-position curves separately. Preserve empty, shuffled, wrong-entity, exact-record and full-context controls. Exact/full context are diagnostic references, not proof of native memory.

### Vocabulary

First test fixed mappings, then segment-boundary swaps. Compare ordinary tokenizer, fixed phrase bank, active swappable bank, permanent expansion, and copy-feedback versus atomic-feedback. Include morphological variants, multilingual bytes, punctuation, code identifiers, ambiguous abbreviations, overlapping phrases, inactive entries and unseen compositions. Native candidates must arise from the visible prompt or lawful memory.

Report canonical-byte accuracy, semantic accuracy, candidate recall, probability mass on irrelevant phrases, invalid references, average emitted bytes/model step and end-to-end latency. Compare the same decoded answers/byte budgets; phrase token count alone is not throughput. Charge cold phrase encoding, refresh, base-head work and fallback. Freeze the phrase encoder during the initial online-adapter experiment.

### Vision and visual reasoning

First defeat question-only/per-task majority baselines on newly balanced examples. Evaluate actual, wrong, shuffled, masked and no-image inputs. Separate perception from relation reasoning: color/count/OCR/object grounding, then multiple-object relations, geometric transformations and multi-step questions. Use new renderers/compositions and real images; synthetic success alone is not general perception.

Compare four trained reasoning modes: ordinary answer generation; explicit point/box-grounded reasoning; native recurrent visual state; and parameter-/compute-matched nonrecurrent state. Supervision and example selection must be matched wherever possible. Evaluate single-answer accuracy at fixed decode budget; report best-of-N separately with its additional work.

Causal tests must include both directions:

- Swap the query-relevant object/geometry while preserving irrelevant context. Does state and answer change correctly?
- Replace or swap only the relevant latent slots between matched examples. Does the predicted relation follow the changed evidence?
- Perturb unrelated regions/slots. Does the answer remain stable?
- Remove original image KV for a separately trained sufficient-state task. Can the learned state still support later attribute/relation questions?

Noise damage, logit differences, attention maps and visually plausible traces alone do not pass. Do not force the sufficient-state condition into all normal inference: it is a diagnostic of what the state retains.

For video, use identity-through-occlusion, event ordering, change detection and long-delay questions. Split by video/world and trajectories; include time reversal, audio offset, repeated frames and identity swaps. Ground intervals against actual time; test short events missed by sparse sampling. Escalation to a higher frame/crop budget must be learned and counted, not supplied by the evaluator.

### Audio and affect

Compare Whisper-small and SPEAR Base v2 at equal input duration and bridge-training budgets; add Whisper-base as the cost floor. Test final-layer, selected-layer and larger fusion controls. Start with frozen encoders; all acoustic-layer unfreezing is explicitly a separate arm.

Evaluate transcription with WER/CER, sound-event QA/localization, speaker-turn continuity, prosody-sensitive decisions, and AV questions. Use natural recordings with speaker/session/device/source-disjoint splits. Benchmark resources can include LibriSpeech/FLEURS for speech, AudioCaps/Clotho for events, held-out audiovisual questions, and VoxEmo-style affect sets, subject to dataset access and licences. No dataset was acquired in this review.

Important controls are transcript-only, audio-only, video-only, joint AV, shuffled audio, silence, wrong but fluent transcripts, identical words with different prosody, and time-shifted AV. A joint model must beat the stronger relevant unimodal control on **both-modalities-required** questions, not merely average complementary tasks.

Affect outputs use listener-label distributions when available. Report macro-F1, unweighted recall, calibration/Brier score and selective accuracy versus abstention. Do not translate a dataset's acted label into a claim about someone's internal feelings. Test human versus synthetic speech separately, with synthesizer-family holdout. Independent listeners assess response appropriateness on a blinded subset; the training emotion classifier cannot be the sole judge.

### Verified learning and core

Teach a new fact, naming convention or executable rule; remove teaching text and its KV; test fresh instances and later compositions. Compare memory-only adaptation, fixed supervised adapter updates, fake/mismatched updates, no-update, and the native learned policy. For persistent parameter-transfer tests, separately disable the teaching-record route so memorized lookup cannot explain improvement.

Split rules/families across meta-training, development and final evaluation. Holding out entities while retaining an identical rule is not unknown-rule transfer; report those separately. Novel rules must be identifiable from the supplied demonstrations or instruction. Measure retention before and after each update on disjoint tasks, multimodal perception and previous learned rules.

For the semi-cognitive core, compare 0/1/2 recurrent iterations, parameter-matched feed-forward and compute-matched controls. Detached audit-head accuracy is a diagnostic only. Add a direct learning loss through the core, test state intervention and measure future utility. Functional appraisal must earn a benefit in calibrated attention/write/update decisions; emotion labels or stylistic wording alone are insufficient.

## 4. Joint memory × swappable vocabulary × learning experiment

The primary factorial has three switches: **M** = persistent native bank, **V** = swappable atomic phrase vocabulary, **L** = verified parameter updates. Run all eight arms: 000, 001, 010, 011, 100, 101, 110, 111, always interpreting bits in M/V/L order. Keep the backbone checkpoint, permitted observations, training tokens, replay/verification budget and evaluation cases matched.

A joint world has the following causal timeline:

1. Supply new objects/entities and names, with definitions and independently verified demonstrations. In the multimodal extension, link each name to visual evidence and pronunciation.
2. Learn/use the first active lexical set; save the canonical lexical identities and observations.
3. Introduce disjoint lexical sets until **544 distinct lexemes** have passed through a **256-slot** active bank. The ordinary episode stays within the 1024-record memory capacity; budget other fact/rule/revision records explicitly (target ≤768 total occupied records). This isolates slot turnover from memory overflow.
4. Provide corrections and novel rule demonstrations. Enforce the same legal evidence in every arm; gold answers do not choose phrase candidates or memory refresh targets.
5. Remove original prompt/KV, restart, ask delayed facts and fresh rule applications, then test compositional transfer and previously learned skills.
6. Separately run semantic-bank overflow, deletion/revocation, representation-version migration, incompatible reload and adversarial distractor episodes.

When M is disabled, the lexical candidate mechanism still has a fixed-size lawful representation of currently visible teaching/prompt information so V is well-defined; it receives no hidden persistent semantic store. Candidate access and storage bytes must be logged for every arm. Additional auxiliary diagnostics can force identical candidate lists to isolate lexical decoding from native selection, but must be labelled oracle/controlled diagnostics when relevant, never the deployment result.

Evaluate each ablation's intended effect. Removing M should harm delayed recall; removing L should harm unseen rule transfer; V should save measured work while preserving quality. A statistically positive three-way interaction is informative, **not required** for a useful integrated system. Avoid the claim that every component must improve every task.

After text joint utility is established, extend the same episode construction first to vision, then audio/AV. Do not run a combinatorial factorial of every architectural knob. Freeze a successful base, add one modality path, and use matched no-/wrong-modality interventions within the retained principal arms. All model-selection decisions happen on development data.

## 5. Statistics, budgets and proposed gates

Start with a bounded engineering screen: two training seeds, 32 development worlds/arm and a four-device-hour cap for the admitted comparison. This is for debugging, not final claims. Confirmation begins with three independent training seeds and 128 fresh worlds per seed; use development-only power analysis to select a final size up to 1024 worlds/seed before unsealing labels. If the admitted budget cannot resolve the result, report **inconclusive**.

Cluster resampling by whole world/video/speaker as appropriate and report every seed/task family. Use paired 95% intervals, with Holm adjustment or simultaneous intervals across declared primary superiority comparisons. Never treat many questions from the same source as independent trials. Public benchmark scores are secondary external checks unless preregistered as primary. Refresh final evaluation data after it influences a training decision.

The following are **new proposed product gates**, to be frozen in execution protocols before training. They do not replace stricter existing gates or retroactively accept old arms.

| Gate | Proposed acceptance criterion |
|---|---|
| Identity/restart | Zero stale-slot binding, cross-session leakage, duplicate commit or undeclared source-reload failures in the declared adversarial suite |
| Native recall | ≥90% on delayed facts and corrections separately; adjusted lower confidence bound above empty/shuffled controls; unaffected retention passes |
| Vocabulary | Quality lower confidence bound ≥−2 percentage points versus base tokenization; ≥10% end-to-end latency improvement on preregistered phrase-heavy workloads; no general-workload regression beyond cost cap |
| Instructed skill learning | ≥5 pp point gain and adjusted lower bound >0 versus no/fake-update on fresh rule applications; persistent tests after source/KV removal |
| Retention | Lower bound ≥−2 pp on categorical retention suites; per-domain loss/continuous metrics have separately preregistered noninferiority margins |
| Spatial/temporal reasoning | ≥75% on the declared balanced relation suite, ≥5 pp gain and adjusted lower bound >0 versus equal-compute plain reasoning; relevant state interventions establish dependence; old vision-control gates also apply |
| Grounding | Proposed object-grounding recall ≥80% at IoU≥0.5, with source-ID correctness; temporal tolerance is fixed from dataset resolution before testing |
| Native audio | WER/CER no worse than 10% relative to the matched compact speech baseline; nonverbal QA ≥5 pp better than transcript-only with lower bound >0; explicit per-language coverage |
| AV integration | On tasks requiring both modalities, ≥5 pp gain and lower bound >0 over the stronger unimodal control; time-shift tests respond correctly |
| Affective evidence | Beat transcript-only on same-word/different-prosody decisions with lower bound >0; improve calibration against a prior-only baseline; independently reviewed response quality |
| Integrated utility | 111 passes recall/learning/retention; required component-removal contrasts pass; vocabulary earns efficiency without degrading quality |
| Learning policy | ≥5 pp future-task gain and lower bound >0 over a fixed policy at matched total cost, including rejected candidates and replay |
| Recursive improvement | At least two independently evaluated policy revisions improve future learning over the fixed policy; no evaluation leakage or resource escalation explains gains |
| Runtime | Proposed text warm-P95 overhead ≤10%; sustained media RTF≤1 for declared stream load; no growing queue; learning duty cycle ≤5% on declared soak schedule |

For a final product, relative gains are insufficient on their own. The release manifest must list intended languages/tasks, meaningful absolute task floors, failure/abstention behavior and measured hardware. The initial floors above cover controlled suites; broader benchmark floors must be selected from development baselines before the final audit, never after seeing test labels. They are not promises of universal competence.

## 6. Transfer and release sequence

| Milestone | Evidence needed to exit | If it fails |
|---|---|---|
| D0: current diagnostics | Balanced vision supervision; viable non-oracle frozen-writer memory policy | Revise data/policy; do not widen training blindly |
| D1: native modules | Source/backend/gradient/reload/resource contracts | Fix integration; no capability language |
| D2: individual utility | Useful memory, visual/audio evidence, vocab efficiency, fixed verified learning | Retain simpler controls; isolate the missing mechanism |
| D3: MiniCPM integration | Joint factorial, multimodal transfer, retention, state identity | Investigate interactions before Spark-scale training |
| S0: Spark engineering | Exact custom architecture, cache and all-parameter update qualification | Precision/backend/resource work; independent of RSI |
| S1–S3: Spark full FT | Warmup→full co-adaptation→grounded/temporal training with retained text competence | Adjust admitted recipe or capacity; no automatic replacement with LoRA |
| S4: product candidate | All claimed capability gates on fresh Spark tests; integrated reload and long-run reliability | Restrict release claims and continue work toward target scope |
| S5: learning-policy research | Matched-budget policy/recursive gains | Call it fixed verified adaptation until evidence improves |

Soak sequence: 30-minute functionality check, then 8-hour sustained stream, then a 72-hour release soak with fixed input and teaching-event schedules. Log every update/rejection, KV/bank/replay high-water mark, queue length, response latency, driver memory, host memory and swap. Include periodic restarts, corrections, expired sources, active vocabulary turnover and long-delay queries. Accelerated event replay is useful stress testing but does not substitute for a real elapsed-time soak.

The release artifact includes the full backbone plus encoders/bridges/core/memory/lexical modules, all source revisions and licences, tokenizer/schema versions, adapter and optimizer identities, state quotas, migration/reset rules, model/evaluation cards and measured resource profiles. It must load without an external semantic harness. Independent evaluation infrastructure remains separate from the deployed model's self-reports.
