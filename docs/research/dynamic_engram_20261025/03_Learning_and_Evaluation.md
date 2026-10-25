# Training, learning and evaluation plan

All gates and budgets below are **proposed**, not measured outcomes. This plan extends the existing SX stages and preserves their MiniCPM control and independent Spark qualification. It does not promote the rejected M4 or v5 candidates.

## Diagnose before expanding

The first experiment should determine where a new binding is lost. Use fresh, independently generated worlds with many entity–attribute pairs, truly novel values, paraphrased questions and several queries against the same unchanged state. Keep training, development and final worlds separate by generator seed, entity/value inventory, templates and visual assets. Existing exposed cases are regression evidence only.

Run four progressively stronger diagnostic interfaces:

1. **Storage probe:** train a small frozen-state probe to recover observed associations. Train it on separate episodes and test new worlds. A diagnostic probe may receive the queried key; it cannot receive the answer or replayed source at test time.
2. **Addressing probe:** use controlled stored values to test whether the native query path selects the right binding. Swap questions while fixing state and swap worlds while fixing questions.
3. **Readout probe:** supply a labelled source-derived latent or pointer candidate to determine whether the decoder can emit an observed novel value. This isolates decoding; it is not evidence that the learned writer works.
4. **End-to-end memory:** raw observations enter once through the actual writer; later native reads must answer without the source or oracle slot labels.

If storage fails, improve the writer/representation. If storage works but addressing fails, train retrieval with hard distractors. If both work but output fails, test a native token-copy head or a better residual interface. If all diagnostic paths work but the complete path fails, inspect credit assignment and distribution mismatch. Increasing every module's width would hide this distinction.

## Objective and gradient contracts

The proposed outer objective is a controlled mixture:

```text
L = L_delayed_answer
  + lambda_KD * L_verified_context_distillation
  + lambda_route * L_support_and_null_routing
  + lambda_binding * L_observed_binding
  + lambda_retention * L_unrelated_and_previous_tasks
  + lambda_temporal * L_grounded_temporal_prediction
  + lambda_cost * expected_extra_compute
```

Activate only the terms relevant to the current experiment. Choose weights on development data and freeze them before the held-out run. Begin with answer loss, routing supervision and retention; introduce one auxiliary objective per ablation. A long list of losses is not a substitute for showing which one fixes a measured failure.

Distillation uses the same backbone with verified information in context as a teacher when that teacher can solve the example. Keep student memory construction query-independent for binding experiments. Align teacher/student predictions by target token, not absolute positions. If retaining only top-k teacher probabilities, define tail handling and student normalization explicitly; normalizing both sides over only selected tokens can hide inappropriate student mass elsewhere. Cache eligible teacher targets for an unchanged teacher to reduce repeated inference cost.

The student is trained through future use of its state. Frozen backbone parameters still permit gradients through the backbone's computation into earlier readers/writers. Wrapping the entire backbone in `no_grad` would sever those paths. Keep a bounded functional graph over two to eight observations, detach only at the specified horizon, and compare truncated versus longer credit windows. Deployment snapshots are detached. Check both the tensor membership of the optimizer and which tensors actually changed.

Writer targets must not collapse to a constant shared embedding. Include varying values for the same key, different keys with the same value, irrelevant observations, corrections and delayed multi-query use. A low associative reconstruction loss alongside failed answer accuracy is a rejection signal. For fast MLP inner learning, separately log memory-gradient norms, update coefficients and outer loss; avoid confusing inner reconstruction with task improvement.

### Preservation training

Mix memory-relevant and irrelevant queries in every applicable phase. Include questions whose entity appears in memory but whose requested property does not, and tasks for which the base model already has useful knowledge. A null memory route should preserve the parent response distribution. Train positive use and selective silence together.

Retain a fixed campaign baseline as well as the immediately previous model. Otherwise a small allowed regression in every cycle accumulates into severe forgetting. Keep an independent final test set that the model's learning policy never sees as input, reward or selection feedback. Record all attempts and failed candidates; model-generated confidence does not decide acceptance.

## Proposed work packages

| ID | Deliverable and trainable scope | Dependency and decision |
|---|---|---|
| DE0 | Reuse qualified repair source, verify canonical scoring and admit a fresh novel split generator; no model training | First. Repair export is recorded as qualified; our 99-test check and that report do not admit new experimental data |
| DE1 | Storage/addressing/readout diagnosis and native binding contract | DE0; identify a specific failure before another native training arm |
| DE2 | Native episodic writer/router/read/copy modules with null routing | DE1; backbone frozen, no static pattern or affect dependency |
| DE3 | Compact functional fast memory; compare fixed/private/shared and optional nonlinear update | DE1; may run independently of DE2, then require complementary joint utility |
| DE4 | Optional static-pattern training and direct-utility diagnostic | DE1; do not block dynamic memory on static-table success |
| DE5 | Core source upgrade and visual/event memory integration | DE2 + DE3 and separately qualified V0–V2 vision; text-core contract can be developed earlier |
| DE6 | Instructed skill acquisition with bounded generated/learned adapters | DE2 + DE3; visual skill version additionally needs DE5 |
| DE7 | Slow consolidation, HOPE-inspired levels and learned learning policy | DE6; faithful HOPE is a separately named mechanism branch |
| DE8 | Optional appraisal controller and emotion simulation readout | DE5 + DE6; utility versus neutral and latent controls required |
| DE9 | Repeated-learning campaign, native export and integrated qualification | DE5 + DE7; optional DE4/DE8 require their own passing gates |

Map these to the prior plan: DE0/DE1 refine SX0/SX4; DE2/DE3 implement the memory part of SX4/SX6; DE5 extends SX5/SX6; DE6 supports SX7; DE7 supports SX8/SX9; DE9 contributes to SX11. Spark native execution still needs SX1. Personality content is a separate SX10 input; this plan defines its interface without inventing it.

### Initial training envelopes

These are stopping/checkpoint envelopes for feasibility studies, not a promise that they teach general memory:

- **CPU/tiny diagnosis:** 64/256 slots, 2–4 observation chunks and controlled symbol/visual-feature inputs. Check causal gradients, functional equivalence and novel bindings before loading native weights.
- **MiniCPM native pilot:** start with short 128–256-token observation chunks, batch one and a two-chunk credit horizon; preflight 512-token cases before expanding. Use approximately 1,000–10,000 generated training episodes, with progress/stop checks at 0.25M, 1M and 5M consumed tokens. Count actual source and answer tokens separately.
- **Training-scale qualification:** only after broad held-out utility, consider 10M–100M-token memory-aware continued training or equivalent episodic curricula, budgeted from measured throughput. Stop if gains do not scale or retention degrades. These local-scale budgets do not reproduce the much larger cited pretraining programs.
- **Spark transfer:** qualify tokenizer, precision, native positions and new projection geometry; reuse data contracts and losses, not MiniCPM-shaped weights. Begin with frozen Spark and new modules, then compare a tightly scoped small adapter if needed.

One MPS worker remains the repository policy. Short CPU tests may support implementation, but native training must retain the no-CPU-fallback and memory guards. Freeze experimental inputs before starting; implementation of this plan is a subsequent task.

## Instructed learning and self-directed learning

Distinguish four user intentions in training data:

| Instruction | Intended update | Test |
|---|---|---|
| “Remember that this object is named N” | Episodic binding | Recall N from another view after the source leaves context |
| “The previous name was wrong; use M” | Correct an identified binding | Current answer uses M; unrelated bindings and appropriate historical queries remain valid |
| “Use this procedure when condition C holds” | Procedure representation, then optional adapter learning | Transfer to new examples under C; no change outside C |
| “Improve your method from these outcomes” | Replay/update policy or memory procedure | Learn a later new task more effectively at the same budget |

Begin slow updates from verified demonstrations and replay. An illustrative bounded update window is at least 32 verified examples, up to 10 small-module optimizer steps, followed by the frozen development evaluation. This is a proposal to calibrate, not a universal minimum. Select the learning window using measured utility and resource budget, not a fixed wall-clock “sleep” metaphor.

Self-directed learning adds a native head predicting the value of another observation, practice example or replay batch. Initially distill selections from evaluated candidates; later compare a learned selector against random, uniform replay, novelty and loss-based heuristics under identical data access and compute. The head chooses among declared training operations. An optimizer kernel executes them; an independent evaluator determines whether the candidate qualifies.

Generated practice must have a source-backed, programmatically checked or otherwise independently validated outcome. Compare self-generated and externally supplied practice at equal quality and cost. Repeating a model's unchecked beliefs can reinforce mistakes. Model-only consensus is a weak verification signal, especially for new knowledge that the base model lacks.

## HOPE-inspired levels with real responsibilities

| Level | State/weights and signal | Proposed cadence | Required distinction |
|---|---|---|---|
| L0 | Workspace and appraisal; observation/prediction features | Each accepted observation or bounded decision step | State inference, not a gradient update |
| L1 | Episodic/fast state; learned write rule or associative inner loss | Accepted observation boundary | Query-only calls do not update it |
| L2 | Generated adapter or small learner; demonstrated task loss | Eligible demonstration bundle | Scope and update cost explicit |
| L3 | Persistent skill adapters / memory procedures; verified replay plus retention | Qualified consolidation window | Gains persist after L0/L1 reset |
| L4 | Writer/update/selection policy; held-out future learning utility | Offline meta-training or infrequent evaluated campaign | Improves how later learning occurs |

For the first multi-rate experiment, compare one-rate and two-rate small adapters with identical total gradient evaluations and data. Then test the usefulness of cross-level transfer. In a faithful HOPE branch, specify the actual context flow, initialization and optimizer state at each level, including what resets and which loss trains it. Updating all slow weights continuously would defeat the intended cost advantage.

## Visual and temporal curriculum

Use source-eligible data from the [existing acquisition plan](../../08_Dataset_Acquisition_and_Curriculum.md), with new programmatically controlled episodes for precise memory tests. Start with image-dependent tasks before adding long video.

1. Region/object grounding with held-out instances, views and backgrounds; explicit negative examples where the answer is not visible.
2. New visual-name bindings and cross-view retrieval; compare latent-only, payload-assisted and caption-only memory.
3. Events with known temporal order, object movement and occlusion; reverse/order-shuffle frames and preserve the original timestamps.
4. Delayed multi-event questions requiring at least two relevant observations, including changing relations and corrections.
5. Demonstrated visual procedures with transfer to new objects/scenes; compare persistent skill learning after episodic reset.

Evaluate still-image appearance, temporal order and persistent tracking separately. Ensure single-frame or last-frame shortcuts fail on the intended temporal tasks. Track identity and coordinate errors can masquerade as memory failures, so report them before downstream answer accuracy. Keep generated counterfactuals and world-model predictions out of the observed-event bank.

## Evaluation matrix and proposed gates

Every capability claim needs native task outcomes, attribution controls and a cost measurement. Use at least three seeds and world/task-level confidence intervals. A group of twelve queries from one world is not twelve independent worlds. The following scales are initial design targets and must be registered before evaluating a candidate.

| Gate | Proposed criterion |
|---|---|
| D0 integrity | Independently reconstructed schedules, wrong-scope rejection, representation-preserving rebuild, source/export compatibility and no target leakage |
| D1 answerability | Full-context reference at least 95% on controlled binding categories; otherwise revise the dataset/readout diagnostic before judging memory capacity |
| D2 memory | At least 90% novel-binding, correction and unaffected-binding accuracy separately; at least 95% absent/conflict handling; at least 20 percentage points above disabled and shuffled memory on memory-required items |
| D3 retention | No more than 2-point observed decline versus both parent and fixed baseline on at least 1,000 diverse unrelated examples; paired 95% interval excludes a decline worse than 5 points; report per-family failures |
| D4 attribution | Shuffled-state, wrong-entity, stale-version, query-only and removed-modality controls behave as expected; direct branch tests show the claimed source of improvement |
| D5 efficiency | Proposed p50 decode overhead at most 15%, p95 at most 25%, and total representative interaction overhead at most 20%, unless the branch wins the preregistered quality/cost tradeoff; include writes and inspection |
| D6 visual memory | Clear improvement over caption-only and no-temporal controls on held-out scenes; at least 10-point gain on predeclared memory-required spatial/temporal categories, with retention passing |
| D7 skill learning | Improvement on new examples under the instructed condition; check both adapter-on and episode-state-reset behavior; unrelated conditions pass D3 |
| D8 appraisal | Beats neutral, shuffled and equal-parameter latent controls on a predeclared decision/acquisition metric at matched compute; no factual-confidence or retention regression |
| D9 continued improvement | Ten evaluated update cycles without cumulative retention failure; all candidate attempts and costs counted |
| D10 recursive learning | At least 10% lower measured cost to reach a fixed score, or equivalent preregistered learning-curve improvement, on later unseen tasks across at least three task families and three seeds |

The point thresholds are a proposed engineering policy, not statistical proof. Publish paired counts, confidence intervals, task-family outcomes and saturation curves. A candidate that improves a small mean while breaking correction or noninterference does not qualify. Avoid reusing final test outcomes as optimization rewards.

### Essential ablations

Use backbone-only, full-context with reusable KV, full-context replay, exact-context diagnostic, episodic-only, fast-only, joint episodic+fast, static-only, neutral core, recurrent core and matched feed-forward core. Add generated adapter, consolidation, multi-rate updates and appraisal one at a time. For state sharing, compare equal-state and equal-active-compute conditions explicitly. For routing, include a fixed policy with the same eligible information; no branch may receive benchmark category labels unavailable in deployment.

For the claimed recursive effect, compare the evolving learning policy to a frozen learning policy with the **same evolving knowledge**, the same examples and the same total update budget. Reset transient state between new tasks. Otherwise remembering more data can be mistaken for becoming a better learner.

### Stopping and failure interpretation

Reject an arm on target leakage, nonfinite state, unresolved incorrect correction, meaningful retention failure or inconsistent exports. If a small state saturates, plot accuracy versus number of independent bindings and distractors before increasing it. If a bigger state does not improve the curve, focus on representation or readout. If source-conditioned teachers fail, supply better validated demonstrations or simplify the task; more self-generated explanations are not a remedy.

Native integration is accepted only when the learned router/writer/core functions are in the checkpoint and the direct model interface passes. A scripted diagnostic may guide training, but cannot be counted as the delivered capability.
