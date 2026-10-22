# 01 - A model that develops through experience

**TAIS Obsidian / Replica R3 | 23 October 2026 | Proposed architecture, with a tested starter implementation**

## 1. The intended result

Build a persistent model system that can maintain a goal, use previous experience to interpret a new situation, identify what it does not know, obtain permitted evidence, act, check the outcome, and improve selected future behavior. The Replica branch begins from the source person's authentic experiences, decision priorities, ideas and goals, then accumulates its own separately attributed experience.

The core hypothesis is that a stable language-model foundation can support a developing system when useful information, procedures, control policies and selected neural components remain adaptable. This is an experimental engineering hypothesis. The delivered code establishes some software and numerical contracts, not the hypothesis.

The supplied R2 documents already establish the full-chain objective. R3 adds an executable starting point, current dataset choices, explicit teacher prompts, model-specific integration constraints, and a diagram atlas. It does not replace the project with a retrieval-only assistant. The seven supplied R2 files are preserved under [reference/r2](../reference/r2/00_README.md); other historical files named inside them were not supplied or inspected in this revision.

### Two distinct uses of full fine-tuning

During **development**, a 1B/2B-class model may update every original parameter and the new components. This tests whether the new interface needs co-adaptation and whether it adds value beyond ordinary full fine-tuning.

After **freezing a developmental checkpoint**, the system may acquire records, update private session state, compile a procedure, and train a bounded candidate skill or sidecar. The base remains unchanged during that deployment epoch. Its hash is checked; changes to an extension still require regression evaluation.

The base's capacity strongly influences reasoning quality, sample efficiency and the kinds of representations the extensions can use. It is not a mathematical upper bound on the composed system: tools, algorithms, added computation and trained extensions can contribute new capabilities. Nor does keeping the base frozen guarantee zero forgetting. These are distinctions in what is being measured, not promises of unlimited improvement.

## 2. Define semi-cognition through behavior

An operational definition avoids requiring a claim about consciousness. The first system must demonstrate these abilities on specified tasks:

| Behavior | Observable test | Misleading substitute |
|---|---|---|
| Goal maintenance | Finish a multi-step task despite distractors and a restart | Repeat a goal description |
| Evidence use | Change an answer when relevant evidence changes; ignore irrelevant evidence | Retrieve similar text |
| Missing-information detection | Ask for a decisive missing constraint | Emit a generic low-confidence disclaimer |
| Planning and revision | Change a plan after an actual failed test | Generate a longer plan |
| Capability awareness | Predict empirical success under the current model/tool/budget configuration | Equate token probability with correctness |
| Strategy learning | Use a verified procedure on unseen inputs and later unseen compositions | Recall the old answer |
| Learning selection | Gain more per allowed practice budget than fixed/random selection | Generate many easy tasks |
| Replica fidelity | Preserve documented conditional priorities in new scenarios | Sound familiar or always agree |

No single scalar combines these into a defensible percentage of cognition or percentage of the user. Maintain separate task correctness, acquisition, control, learning efficiency, retention and identity measurements.

## 3. State and update ownership

The system can be written conceptually as

$$y_t=F(x_{\le t};\theta,\phi_t,E_t,W_t,A_t,S_t,I_t,P).$$

Here theta is the base, phi the trained extensions, E the evidence ledger, W the task workspace, A private adaptive state, S the skill catalog, I identity state, and P the externally configured permissions/evaluation policy.

| State | Contains | Normal update timing | Owner and reset |
|---|---|---|---|
| Original base | Pretrained language/reasoning prior | Never edited in place | Archived reference |
| Developmental base | Co-adapted original parameters | Named full/selective development runs | New checkpoint revision |
| Reader/writer/workspace parameters | How to encode, use and organize evidence | Candidate training jobs | Versioned extension set |
| Task workspace | Goal, evidence pointers, hypotheses, next-action state | Decision boundary / bounded recurrent steps | Per task; reset on new task |
| Fast neural state | Approximate recent associations | Completed observation chunk | Per session and branch |
| Evidence records | Source facts, reports, events and corrections | Authorized record transaction | Subject/branch scope |
| Skill library | Reusable procedure or neural adapter | Verified candidate publication | Versioned catalog |
| Source identity | Human's approved history and priorities | Source-person revision | Human source authority |
| Replica history | Agent's own experiences and declared changes | Observed agent event | Replica branch |
| Evaluator/policy | Test rules, permitted actions, resource budget | Deliberate operator configuration | Outside candidate write scope |

Records and neural state serve different purposes. Exact identifiers, dates, code and source spans must remain available as records. Approximate memory can improve access and fusion without becoming the sole authority for autobiography.

## 4. Recommended initial neural architecture

Begin with native MiniCPM5-2B, one complete decoder-block wrapper, a learned memory reader, and a separate recurrent workspace. The [official model configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json) identifies a native Llama decoder; this makes the initial port relatively direct.

At an insertion point:

$$h' = h_{\text{native}}+\Delta_{\text{memory}}(h,M)+\Delta_{\text{workspace}}(h,W).$$

Each branch has its own null path and gate. Initialize its final output projection to zero while initializing upstream weights normally. This preserves the native result at initialization while allowing the output projection to learn first. Zeroing every factor in a product can prevent useful gradients.

**Implemented starter:** separate support and prompt embedding inputs; attention-based record pooling; memory cross-attention; a small recurrent slot workspace; independent memory/workspace injection; one native block wrapper; explicit per-batch context lifetime through backward.

**Deliberate initial simplifications:** the writer receives input embeddings rather than a separately pretrained semantic encoder; the workspace remains fixed during one answer segment; the input also contains the same source text as the full-tuning baseline. This equal-information comparison does not claim context compression. A later experiment removes duplicated source text and evaluates latency/state/token savings separately.

The code's default dimensions are 64-wide memory and workspace, four slots and two steps to keep tests and initial overfitting small. Research configurations may compare 128/256 dimensions, 4/8/16 slots and 1/2/4 steps after profiling. These are proposed choices, not published optimum values.

### Where parallel and in-backbone components meet

The host runtime runs retrieval, storage, permissions and tool execution outside the differentiable forward. The trained writer, memory reader, workspace and action heads operate beside and within the backbone's computation. They send residual information into selected complete blocks without replacing native attention.

This division preserves a useful engineering boundary: the model proposes a memory/action/update; the host validates the proposal and performs a transaction. A host runtime is needed even when the judgment is neural. It is not evidence that the project has become retrieval-only.

## 5. Memory components: which paper supports which claim?

| Research | Relevant mechanism | R3 treatment |
|---|---|---|
| [Engram](https://arxiv.org/abs/2601.07372) / [code](https://github.com/deepseek-ai/Engram) | Learned sparse n-gram lookup and contextual fusion | Optional lookup ablation; ordinary record retrieval is not relabelled Engram |
| [Titans](https://arxiv.org/abs/2501.00663) | Online neural-memory adaptation | Separate fast-state experiment after a linear baseline |
| [Nested Learning / HOPE](https://arxiv.org/html/2512.24695v1) | Multiple optimization levels and update frequencies | Later multi-rate experiment, with explicit loss/state ownership |
| [uTTT](https://arxiv.org/html/2610.05484v1) / [code](https://github.com/Zefan-Cai/uTTT) | Share adaptive-state ownership across depths | Compare private vs shared state at matched capacity and compute |
| [Mem-T](https://arxiv.org/abs/2601.23014v2) / [code](https://github.com/yanweiyue/Mem-T) | Learn memory construction/retrieval operations with denser credit | Begin with supervised memory-operation trajectories |
| [SEAL](https://arxiv.org/abs/2506.10943) / [code](https://github.com/Continual-Intelligence/SEAL) | Learn to generate useful adaptation data/directives | Bounded learning-proposal grammar, rewarded by later outcomes |
| [INTACT](https://arxiv.org/html/2610.19568) | Train a latent interface through a frozen decoder | Optional frozen-interface comparator, not a prerequisite |
| [Voyager](https://arxiv.org/abs/2305.16291) / [code](https://github.com/MineDojo/Voyager) | Reusable executable skill library | System-level procedure baseline alongside neural learning |

Several release and evidence limits matter. Mem-T's official README still lists its training script as forthcoming. uTTT's repository releases its MoE language-model route, but not a complete dense language-model trainer or pretrained checkpoints. HOPE does discuss pretrained initialization; the inspected Llama adaptation experiment used 15B continual-pretraining tokens. Its existence supports investigating adaptation, not assuming a cheap retrofit. INTACT is a recent preprint whose smaller models do not uniformly beat SFT. The bundle therefore implements a tractable experimental synthesis and retains ordinary full-tuning controls. See the [evidence register](06_References_and_Evidence.md).

### Fast-state baseline and causal barrier

The included LinearFastMemory is a functional linear associative-state baseline. It is not attached to the model trainer and is not a reproduction of the papers above.

For a state matrix A and observed key/value pairs, a simple loss is

$$L_{\text{inner}}(A)=\frac{1}{2N}\sum_i\|k_i A-v_i\|^2.$$

An update returns a new A. It does not mutate a global parameter, database or earlier chunk snapshot. When several decoder depths share this state, every port reads the same chunk-start A; gradients/writes are combined only after the chunk completes. Reading a whole current chunk's write at a deeper layer can leak future tokens into earlier predictions.

Runtime fast-state updates are not equivalent to publishing new global model weights. If fast state persists across sessions, record its owner, schema, source lineage, update count and reset policy explicitly.

## 6. The autonomous development loop

The proposed deployed loop is:

1. Observe a permitted task and establish its goal and budget.
2. Retrieve eligible evidence and activate the appropriate identity mode.
3. Maintain a workspace; propose an answer, clarification or allowed tool call.
4. Obtain an actual result and assess it with a suitable verifier.
5. Record the observed outcome and diagnose a reusable success or failure.
6. Select a learning target within the allowed task families and budget.
7. Compile eligible practice and replay; construct an isolated candidate.
8. Evaluate the candidate against its parent on unseen task instances and retention tests.
9. Automatically publish, reject or defer within the delegated policy.
10. Measure later behavior after restart and on new tasks.

“By itself” means ordinary authorized learning cycles do not need manual approval at every step. The model can select practice, propose updates and use results to improve that selection. A separately controlled evaluator and manifest publisher enforce what counts as improvement. Neither an LLM's confidence nor an increase in its own reward estimate is a sufficient release condition.

The learner may update records, compiled procedures, named skill adapters or approved sidecars. It may not redefine the held-out test, grant itself new tool permissions, or silently change the source person's biography. A production backbone revision is a separate development/migration event.

The current offline demonstration uses a fixed symbolic compiler and fixed task order to exercise this loop's transaction path. It does not implement the learned director and does not count as a neural self-improvement result.

## 7. Capability awareness and learning value

Keep three separate measurements:

- Empirical probability of task success under the current model, tools and inference budget.
- Expected benefit and cost of asking, retrieving, verifying or computing longer.
- Expected future improvement from one permitted training/practice proposal.

A calibration head learns from observed outcomes on a disjoint calibration split. A learning director initially chooses among a short list of proposals; supervised ranking or a contextual bandit is enough to establish a baseline. More complex reinforcement learning is justified only after the reward is reliable and fixed/random selection has been measured.

A practical starting proposal contains task family, source event IDs, update target, permitted generator, maximum steps/tokens, replay source IDs, parent manifest and expected evaluation. It does not contain arbitrary shell commands. Verifiable microengineering tasks are useful because correctness can be checked outside the model.

## 8. Identity and experience

Preserve four operating modes: source_snapshot, source_current, parallel_replica, assistant_for_source. The first reproduces a dated source profile; the second follows approved human updates; the third develops its own branch; the fourth personalizes assistance without claiming to be the user.

Preference conditioning changes which acceptable solution is favored. It does not make a failed test correct. Style is a separate optional adaptation target. Real decisions, exceptions and changes of mind matter more than personality adjectives.

The source archive is human testimony with uncertainty and provenance. The agent's new experiences are agent history. Inferences and simulated alternatives remain marked as such. A correction changes the current view without falsely rewriting what was known earlier. A deletion must invalidate derived caches/examples/adapters; deleting a record alone does not erase its influence from already trained weights.

See [Replica collection and evaluation](05_Replica_Protocol.md) for the concrete starting interview and fidelity protocol.

## 9. What should remain optional

Wholesale native-attention replacement, multi-stream residual redesign, large hypernetworks that generate arbitrary adapters, elaborate biological analogies, and simultaneous Titans/HOPE/uTTT/CTM activation are optional research branches. They would otherwise make the first outcome hard to interpret.

The first winning system may be small: ordinary full-tuned base plus one reader/workspace and a useful post-freeze skill-learning loop. If the recurrent workspace does not beat matched additional capacity and inference compute, omit or redesign it. If memory alone accounts for the gains, report acquisition rather than improved strategy.

The next stage is earned by reproducible evidence, not by the number of architectural components.
