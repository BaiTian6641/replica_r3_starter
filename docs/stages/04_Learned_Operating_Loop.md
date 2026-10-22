# S3 — connected model behavior and learned adaptation

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The current deterministic contract demo has a fixed symbolic compiler and task order. It demonstrates software transactions; it has not demonstrated learned neural self-improvement. S3 must connect actual model decisions, actual tool observations, actual bounded training and later independent outcomes.

Freeze one declared developmental base, and hash-check it across every post-freeze cycle. Extensions, procedure skills and scoped records may change through versioned candidates; evaluator/policy and source identity remain outside candidate authority. The host executes validated proposals. Use a finite action/proposal grammar rather than generated arbitrary commands.

T0 persists nothing, T1 stores evidence, T2 follows fixed verified replay and T3 learns practice selection plus updates. Match experience opportunities and evaluation compute. A replay-matched T2 shadow control separates chosen data from update method; library growth and neural changes are separate measurements. The ten-cycle matrix must include failures, restart, deliberate rejection, rollback and no-improvement periods, with all learning/generation/verifier costs.

## S3.1 — Connect a frozen manifest to a host-owned runtime

**Status:** `pending`. **Dependencies:** S2.5, S1.1.

Bind model inference to scoped evidence ledger, task workspace, permitted real tools and event journal. Activate one identity mode and immutable base/extensions manifest; model output proposes actions but host code validates scope/budget and executes actual tools.

- **Artifact:** planned: runtime event/action schema and model-host adapter; planned: restart/scope/deletion integration report.
- **Interface:** Model action proposal -> host validator -> tool observation -> branch-owned experience -> ledger. Manifest binds base, tokenizer, writer, reader, task schema and permissions.
- **Test gate:** Actual observations drive revisions, records survive restart, wrong-scope/expired records are hidden, deletion invalidates derived eligibility and base hash stays fixed. No generated text can grant tool authority.
- **Resources:** Local deterministic tools and CPU journal first; bounded MPS inference. Avoid arbitrary shell execution grammar.
- **Failure policy:** Reject invalid proposals without side effects and preserve trace. Repair host contracts before allowing autonomous candidate training.
- **Next smallest task:** Implement a bounded answer/ask/retrieve/verify/stop action schema against one solver-backed tool.

## S3.2 — Train and measure the action policy and capability calibration

**Status:** `pending`. **Dependencies:** S3.1, S1.2, S1.4.

Supervise when to ask, retrieve, verify, answer or stop from independently checked trajectories. Fit empirical task-success and action-value calibration on disjoint data; include missing decisive constraints, failed tools and deliberately unanswerable tasks.

- **Artifact:** planned: verified action trajectories and policy candidate; planned: calibrated success/action-cost and recovery evaluation.
- **Interface:** Observed host trajectories -> eligible training targets -> candidate policy; calibration labels are real outcomes, not token probabilities or self-reported confidence.
- **Test gate:** Policy acts correctly under fixed tool/inference budgets, changes plan after real failure and avoids invented evidence. Report task correctness, clarification benefit, verification cost and calibration separately.
- **Resources:** Reuse one qualified model; small bounded candidate update. Calibration uses separate role and no final audit feedback.
- **Failure policy:** Keep deterministic host guardrails even when policy fails. Correct trajectory/verifier errors before increasing policy complexity.
- **Next smallest task:** Collect a small balanced trajectory set where ask versus answer has a solver-defined consequence.

## S3.3 — Train, reject and publish real isolated candidates

**Status:** `pending`. **Dependencies:** S3.1, S0.4.

Replace symbolic-only candidate compilation with a real named neural extension/skill trainer. Bind source lineage, parent manifest and token/step budget. Evaluate candidate versus current parent independently before atomic publication; retain intentional bad-candidate rejection and rollback.

- **Artifact:** planned: candidate training job manifest and compatibility checks; planned: accepted/rejected/inconclusive candidate integration trace.
- **Interface:** Bounded learning proposal -> eligible replay builder -> isolated trainer -> independent promotion -> immutable manifest publisher -> restart loader.
- **Test gate:** One unfamiliar verified failure yields an actual update and later improvement on unseen instances; intentionally damaging or incompatible candidate is rejected; rollback restores prior behavior. Base/evaluator hashes stay stable.
- **Resources:** One candidate job at a time; predeclare max gradient steps/tokens, verifier calls and wall time. Count rejected candidate costs.
- **Failure policy:** No evidence means reject/defer, not publish. Preserve parent and candidate; never overwrite the accepted base or relax evaluator to accept a failed update.
- **Next smallest task:** Connect the existing publication contracts to one bounded neural skill update and compare unseen solver cases.

## S3.4 — Learn useful practice selection against fixed replay

**Status:** `pending`. **Dependencies:** S3.2, S3.3.

Start with a short finite proposal catalog and supervised ranking/contextual bandit for practice selection. Reward later held-out improvement per declared cost. Compare random/fixed selection and replay-matched controls before considering complex reinforcement learning.

- **Artifact:** planned: proposal/ranking dataset and director candidate; planned: selection-versus-update ablation report.
- **Interface:** Branch-owned failure experience -> bounded proposal with source IDs/parent/budget -> independent practice verifier -> candidate trainer. Learning-policy reward split is separate from promotion and final audit.
- **Test gate:** Director cannot select audit examples, alter verifier or escape task/budget whitelist. Any T3 gain beyond fixed T2 survives replay-matched data and learning-cost accounting; otherwise no useful selection claim.
- **Resources:** Small catalog, bounded number of proposals; count teacher generation, solver, training, replay and evaluator calls. No open-ended RL prerequisite.
- **Failure policy:** Keep fixed replay as baseline and record an inconclusive director if reward is noisy or improvements are indistinguishable.
- **Next smallest task:** Define three permissible practice targets and compute their independently measured learning-value labels on development tasks.

## S3.5 — Run ten chronological T0/T1/T2/T3 learning cycles

**Status:** `pending`. **Dependencies:** S3.4.

Start all arms from the selected frozen parent and same permitted archive. T0 persists nothing, T1 records only, T2 fixed-schedule verified replay/candidates, T3 learned practice plus candidates. Include distribution change, no-improvement period, restart, rejection and rollback.

- **Artifact:** planned: cycle-by-family performance/retention matrix; planned: chronological costs, exposure and publication history.
- **Interface:** Common experience stream -> arm-specific allowed persistent state -> matched later evaluation. T2 replay-matched shadow control isolates T3 data-selection effects.
- **Test gate:** Report all ten cycles, old/new family scores, forward transfer, forgetting, calibration, inquiry cost and rejection. T3 beyond T1 supports learning; T3 beyond T2 at stated budgets supports useful selection. Final audit remains unseen until reporting.
- **Resources:** Ten initial bounded cycles; extend to 30–50 only after useful error bars/instrumentation. Match evaluation compute and disclose extra learning compute.
- **Failure policy:** Do not hide failed cycles or reset history after regression. Reject damaged candidates and preserve a valid negative/inconclusive longitudinal result.
- **Next smallest task:** Freeze the chronological stream, retention matrix and per-cycle resource/exposure budget before cycle 1.
