# Teaching a native learner: MiniCPM validation before Spark

26 October 2026. Proposed protocol extension, not run authorization. It refines the [prior MiniCPM bridge study](../sparkx_dynamic_memory_bridge_20261026/MiniCPM_Validation.md) and [consolidated product plan](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md). Current repository execution/resource protocols still apply. **Full offline fine-tuning establishes multimodal perception and reasoning first. Online self-learning is tested only as knowledge/experience acquisition through those already-trained capabilities; it is not a substitute for multimodal training.**

## 1. Correct the learning claim before testing it

A model whose stable backbone is frozen can learn through persistent native parameters. The principal claim is: **teaching changes designated intrinsic parameters; those changes cause better performance on new instances, persist through restart and interference, and preserve prior skills.**

The old requirement to clear every fast/address parameter was too broad for this claim. It remains useful for testing consolidation into a different parameter set. Use a named storage intervention for every result:

| Evaluation | Retain | Remove/reset | Permitted conclusion |
|---|---|---|---|
| Episodic recall | `E` and its stable identities | Teaching prompt, raw clips and source KV | Explicit memory remains useful |
| Associative learning | Taught `F`; optionally `A` in a separate labeled arm | `E`, source/prototype/caption caches, recurrent state, teaching lexical rows | Transfer attributable to persistent associative parameters |
| Plastic knowledge learning | Taught `P` only, with initial controller/address state | `E/F/A/C`, teaching prompts, clips, prototypes, source KV, read caches and active phrase rows | Reusable learned knowledge in `P`, with `Theta` unchanged |
| Combined native learner | Declared `E/F/A/P` | Current teaching context and derived caches | Product behavior with all admitted storage available |
| Backbone consolidation | Updated `Theta`; reset added plastic parameters to the declared baseline | Teaching-derived dynamic state and shortcuts | Learning transferred to stable release weights |
| Learning-policy improvement | Qualified `Psi` revision and standard initial learner state | All task-specific learning state from policy training | Better future learning, not merely memorized tasks |

For a `P`-only test, clearing the learned `P` is a negative intervention, not the normal test. Compare it with pre-teaching `P`, a wrong-world `P`, and frozen-update controls. For `F` learning, report transient adaptation separately from cross-session persistence and transfer. Clearing labels or a phrase bank must leave the test task expressible using base tokens and the supplied query.

## 2. Build a teaching environment, not a runtime reasoning substitute

The environment provides observations, demonstrations, instructions, corrections and independently established outcomes. The model chooses representation, addressing, writing and admitted learning actions. Evaluation software scores behavior and records cost; it must not choose the correct memory or inject the target relation into inference features.

Each episode has:

1. A fresh world identity, modality assets, rule/concept family and an independent held-out test partition.
2. A teaching phase with examples, optional definitions, and a declared feedback type: exact answer, correction, demonstrated outcome, measured event or uncertain human judgment.
3. A model-native write/update proposal and a bounded adaptation phase. The initial fixed policy removes one source of uncertainty before learning the policy.
4. Development checks separate from final tests. Candidate acceptance may use this feedback; it may not query final test labels.
5. Source removal, fresh-process restart and delayed tests on new instances, including interference and correction.

The model may ask a learning question or propose a practice example. A teacher or simulator may answer it; the answer is a new observation with a cost. Self-generated examples without verification can support consistency training, but cannot certify a new rule. Log the difference. The host's mechanical commit gate checks finite tensors, identity, budget and protocol consistency; it does not pretend to be a neural judgment of correctness.

### Curriculum families

| Family | Teaching experience | Held-out challenge | Main shortcut control |
|---|---|---|---|
| New words and meanings | Novel name, definition and exemplars | New phrasing and new instances after phrase-slot turnover | Base-token spelling fallback; spelling-only control |
| Spatial concepts | Several labeled objects/relations | New shapes, views and arrangements | Wrong image, question-only, swapped regions |
| Temporal rules | Before/after, recurrence and duration examples | New sequences and distractor events | Same frames reordered; time shifts; hidden future frames |
| Audio patterns | Motifs, timbres, rhythms or device sounds | New performances/devices and background noise | Transcript-only, silence, wrong audio and changed relevant acoustics |
| Audiovisual relations | Synchronized light/motion/sound examples | Novel appearance with the learned relation | Offset audio, mismatched identity, missing modality |
| Correctable knowledge | A rule followed by an explicit scoped revision | Revised target plus untouched rules | Contradictory feedback, wrong-scope correction, outdated records |
| Listener perspective | Explicit preference or subjective annotation | A new clip under the requested viewpoint | Speaker/session leakage; uncertainty and disagreement retained |

Split by world, rule template, visual asset identity, audio performer/recording and underlying song where applicable. Randomly splitting questions from the same clip is inadequate. A synthetic relation task can establish controlled causality; it does not qualify open-world vision or music. Retain natural-data evaluation independently.

## 3. Offline training prepares the online learner

The following stages complement the existing F0–F6 product schedule:

| Stage | Trainable scope | Result required before proceeding |
|---|---|---|
| Alignment and zero-effect checks | New projections/readers; stable backbone initially frozen | Correct masks, gradients, native reload and absent-extension parity |
| Perception and integrated reasoning | Entire MiniCPM language backbone, core vision/audio encoders, spatial/temporal bridges and intrinsic machinery in the joint phase | Actual visual/audio dependence, spatial/temporal generalization and text retention |
| Memory behavior | Read/write projections, controller and full offline backbone as admitted | Delayed causal recall and correction with no gold-aware selector |
| Fixed-policy plastic learning | Outer initialization/controller trained on teaching episodes; inner updates only nominated `F/P/A` | Source-free transfer and retention before adaptive policy claims |
| Meta-learning | Outer parameters optimized for future-query performance after bounded inner updates | Improvement over ordinary episodic training at matched total cost |
| Policy evolution | Qualified `Psi` candidates trained from prior verified experience | Repeated future-learning gains on unseen families |

Define inner and outer objectives explicitly. A candidate inner loss combines a supervised local concept/evidence target or observed-transition target with sampled retention replay. The outer loss measures held-out future answers/grounding after adaptation, old-task retention and measured learning cost. Outer supervision can use labels; inference features and selector inputs cannot.

```text
P_next = P - bounded_learning_rate * plasticity_gate * gradient(L_teach + lambda * L_replay)
F_next = learned_causal_write(F, available_observation_features)
outer_objective = L_future_after_updates + alpha * L_old_tasks + beta * measured_update_cost
```

This is a proposed objective family, not a derivation of MemT or HOPE. During offline preparation, optimize the initialization and learning machinery for the outer objective. During deployed teaching, freeze `Theta`; only the declared dynamic subset changes. Use separate held-out task families to establish whether learning this procedure transfers. A gate multiplying a dense update does not imply sparse execution unless the implementation skips the inactive work.

Begin with one to four inner updates, at most 256 model positions per step for the end-to-end control. The local-update candidate processes at most 128 cached feature vectors per step and two `P` experts per teaching event. Report both positions and vectors; these are not equivalent units of compute. Use SGD without momentum first; compare Adam only with all persistent moment storage charged. Maintain a replay-free control and a fixed replay-byte budget for other arms.

For outer meta-learning, start with short unrolls of one to four teaching segments; record whether differentiation is exact, first-order or truncated. Test longer interference streams without resetting the persistent learner. Meta-training from a pretrained decoder is an adaptation hypothesis, not a demonstrated small-token substitute for MemT or TTT-E2E's training scale. Use tiny fixtures only to diagnose mechanics, then separate development and capability-scale budgets.

## 4. Ordered MiniCPM experiments

### IL0 — Reconcile existing failures and preserve controls

The inspected memory native04 run remains rejected at 110/240 recall with zero optimizer updates; restart04 passes parity without re-encoding observations. Vision native05/reload06 recover engineering but retain audit 6/24, all answers `yes`, caption F1 zero. Source hashes are in the [repository observation](../../../reports/sparkx_intrinsic_learning_spine_20261026/repository_snapshot.json). Do not promote these runs because a new architecture is plausible.

Freeze dataset/scorer identities and resolve individual perception/memory quality failures before a large combined arm. Keep native execution and host-assisted retrieval clearly labeled. The next small experiment should identify whether information is absent from storage, not selected, or ignored by the decoder; a larger bank alone cannot distinguish them.

### IL1 — Intrinsic integration and state ownership

Test post-block baseline versus pre-MLP/parallel intrinsic branches on the same MiniCPM source. Check tensor geometry, masks and a live gradient path. Validate causal availability with teacher-forced sequences, exact-once updates under recomputation, boundary commits, scoped state isolation and fresh-process restoration.

Run with the entire peripheral spine absent. Memory and teaching must still work. Conversely, attaching a read-only observer must not change deterministic outputs. These tests establish the intended architecture boundary, not capability quality.

### IL2 — Locate learned knowledge

Compare `E`-only, `F`-only, `P`-only and combined behavior using the interventions in §1. Match teaching examples and report each arm's actual bytes and computation. Use independent training seeds; do not treat a single initialization as conclusive. Include no-update, shuffled-feedback, wrong-world-state, retrieval-only and full-context teaching controls.

For every new concept, measure immediate new-instance accuracy, delayed accuracy after unrelated teaching, correction accuracy and unchanged-rule retention. Restart without source re-encoding. Verify original `Theta` hashes remain identical during the online phase and only the declared dynamic subset changes. A monotonic training-loss decrease or byte change does not pass this stage.

Separate storage from generalization: exact arbitrary IDs and facts test recall; new renderings, performances and compositions test transfer. A `P`-only arm that merely copies a supplied answer is not sufficient.

### IL3 — Memory, swappable vocabulary and learning together

Use an eight-arm factorial with precise factor definitions:

- **M:** episodic/fast pathway `E/F/A` enabled or disabled. This factor does not disable the separately studied `P` branch.
- **V:** fixed base vocabulary or active swappable phrase rows, with immutable base IDs and stable lexeme identity.
- **L:** verified updates to intrinsic `P`, or the same `P` initialized identically and frozen.

These names avoid calling all memory “off” while testing plastic memory learning. Hold the stable checkpoint, teaching worlds, decoded-byte scoring and context/update limits fixed. No arm shares already adapted state with another. Report interactions: lexical efficiency may improve without semantic learning, while phrase rows may accidentally carry answer-bearing knowledge.

After teaching, rotate the entire active phrase bank, including deliberate slot reuse, and test meaning with base-token outputs. Then restore candidates independently of answers. Include adversarially similar names, spelling-only feedback and irrelevant candidates. Compare full product behavior and `P`-only source-free behavior. Recompute candidate vectors after relevant representation changes; optimizer state cannot follow a reused lexical slot as if its identity were unchanged.

### IL4 — Multimodal abstraction and long retention

Extend only qualified text/memory mechanisms to visual, audio and paired events. Use matched observations at teaching, then remove teaching media and test different instances. Include cross-modal queries: hear a learned warning and identify the relevant visual process, or view an event and predict its associated sound class. Test whether the original modality-specific evidence is needed, rather than letting a cached caption answer everything.

Run increasing interference streams, such as 10, 100 and 1000 teaching episodes, at fixed memory bytes. Measure accuracy by age and concept family, forward/backward transfer, unknown false recall, correction survival and evictions. These horizons are proposed stress levels, not a claim that 1000 episodes establishes lifelong learning. Measure wall-clock soak and retention separately from the number of episodes.

Test audio time reversal, relevant-pitch changes, harmless gain changes and competing sources. For emotion/perception tasks, use soft listener labels, calibrated abstention and subject/recording-held-out evaluation. Do not score generated emotion prose as an objective feeling state.

### IL5 — Spine qualification as a separate capability

Begin with an additional recorded sensor and a passive probe. Train the adapter into the typed packet space. Compare aligned, untrained, mismatched and absent adapters; hold the intrinsic learner fixed. Test wrong units, clock shifts, expired packets and version mismatch as contract cases.

Then test one-way model-feature exchange with actual source/target tokenizer differences. Inspect availability masks under teacher forcing. Charge the other model's inference, feature projection, transfer and any extra passes. Two-way exchange requires a finite-round protocol and matched-compute comparison to the one-way baseline.

For actions, start with prerecorded trajectories or simulation and a trained continuous action head. Distinguish requested actions from observed outcomes. Measure action accuracy, failure behavior, P95/P99 decision latency, control deadlines and stale-command handling. No hardware control experiment is implied by this research document.

### IL6 — Learning to learn and recursive improvement

Only after fixed-policy `P` learning succeeds, compare learned retention/plasticity/teaching-selection policies. Score verified future improvement per total compute, including discarded updates, replay, teacher interactions and verification. Hold the stable release fixed when testing dynamic policy updates.

Require at least two successive qualified policy revisions evaluated on fresh task families, with frozen-policy and equal-budget retraining controls. Reset task learning state between policy evaluation worlds. Improved current answers, more stored examples or higher confidence are not evidence of improving the learning algorithm itself. Do not promise self-redesign of arbitrary architecture under a fixed computation budget.

## 5. Proposed acceptance criteria and reporting

These are new development targets to preregister, not inherited experimental results or automatic run admission:

- Persistent `P` learning: at least a 10-percentage-point new-instance gain over no-update on the primary held-out family, with a paired 95% confidence interval excluding zero; legacy-task decline at most 2 points. Also report raw scores and learning curves, since floor/ceiling tasks need separate interpretation.
- Memory/correction: use the existing repository's stricter capability gates where applicable. Report exact recall, unknown false recall and unaffected-rule retention separately; an aggregate score cannot hide failed correction.
- Spine and state mechanics: no future leakage in deterministic fixtures, no unintended tensor changes, no duplicate writes, and parity under a fresh-process restore with source re-encoding prohibited.
- Efficiency: retain the proposed ≤10% warm text P95 overhead and ≤5% online-learning duty-cycle targets on a defined workload; audiovisual streaming real-time factor ≤1 with bounded queues. Report violations instead of changing the workload after seeing results.

Use at least three independent seeds for a promotion comparison, paired worlds across arms, and confidence intervals clustered by world/recording rather than treating repeated queries as independent. Select dataset size through development variance and a power calculation for the preregistered gain; do not prescribe a token count as a guarantee of statistical power. Separate synthetic mechanism tests from natural-data product evaluations.

Every result should identify stable release hash, dynamic-state hashes before/after, exact reset/retention policy, data/split identities, changed tensor names, source availability, native versus assisted selection, update/replay/verification cost, encoder/prefill/decode latency, resident and peak bytes, and final capability disposition. Native cached generation remains unqualified in the present MiniCPM executor; re-prefill performance cannot be presented as production KV-decoding performance.

## 6. Spark transfer decision

Move to full Spark fine-tuning after the individual and joint MiniCPM mechanisms have credible benefit. Requalify the custom Spark attention geometry, hybrid full/sliding pattern, tied output embeddings and backend. Do not transfer MiniCPM layer numbers mechanically.

The initial expensive training should establish a reusable learning procedure. Subsequent teaching should normally update intrinsic dynamic state/parameters, not rerun full 4B fine-tuning. Measure the break-even point: initial training cost plus every teaching, replay and maintenance operation versus repeated full fine-tuning at comparable quality. Neither the current repository nor the reviewed papers supplies that break-even number for Spark.
