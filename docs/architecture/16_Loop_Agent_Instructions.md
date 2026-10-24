# Loop-agent execution instructions

**Revised 24 October 2026 after review of `280d4dd`.** Use this prompt for the
existing repository. The active user authorized benchmark/scorer repair,
persistent-memory/correction/retention work and bounded fine-tuning. Preparing
this prompt does not create another chat or automation. Execute locally in the
current checkout; preserve unrelated work and the protected runtime pilot.

**Resume from the completed recovery pilot.** The first authorized D1 run has
already completed; do not launch v1 again. Read
[its result](../experiments/20261024_Memory_Reliability_Recovery_Results.md) and
`reports/memory_reliability_attempt02/verdict_final.json`. It trained 24 updates
on 96 cases with Muon int8 momentum and train-only base-preservation KL, then
was **rejected**: initial facts 0/16, corrected facts 0/4, unaffected facts
0/12, general answers 0/8 versus disabled 8/8. Full-context control is 48/48.
All eight images authenticate; a separate process reproduces seven answers and
state exactly without replaying sources. That includes wrong answers and is
not quality acceptance. The saved candidate remains diagnostic only.

Continue TAIS Obsidian / Replica R3. Keep sealed MiniCPM5-2B as baseline and the
50 GiB guard. Carry authorized work through implementation, meaningful tests,
a bounded real fine-tuning experiment and an evidence-backed accept/reject/defer
report. A document or a green component suite alone is not completion.

## Immediate work order

The benchmark/scorer fixes and first persistent-state implementation below are
now present. Verify them once and inspect deltas; do not rebuild them. Reuse:
`memory_reliability.py`, `answer_audit.py`, `reliability_scoring.py`,
`runtime/persistent_fast_memory.py`, `scripts/run_memory_reliability.py`,
`scripts/score_memory_reliability.py` and `tests/test_memory_reliability.py`.
The new scorer independently decodes outputs and authenticates extension/image
bytes; the old M4 rejection remains historical. Protocol v1 and its pre-run seal
are hash-pinned but not committed. Do not edit their pins to fit new code.

The completed package is **query-selective memory and noninterference**, with
durability coverage, not RSI. The exact-record selective context passes 192/192,
but full-context parity fails on two absent-fact answers (`orange`/`green` instead
of `unknown`), so the reference bundle is rejected under the declared parity gate.
The 48-update reader-only D1 candidate scores 64/192 (32 unknown plus 32 general),
with fact, correction and unaffected scores all zero and delayed-above-controls
failing on both splits. All 32 images authenticate and fresh-process restart
reproduces 7/7 answers and source bindings. Protocol v4 is committed at `9a43bad`;
the corrected finalizer separates native and smoke-fixture provenance and records
the gate-off counter erratum (64 independently derived versus 0 reported).

**Next smallest discriminating action:** inspect per-fact writer/state bindings
before considering a new writer arm. Reader-only failure does not prove storage
is correct. Do not add updates, increase KL, switch models or promote RSI. Treat
full-context absent-fact hallucination as a separate abstention diagnostic. M4,
M3 repair and the first 24-update pilot remain rejected. Exact-record reference
and persistence mechanics are reusable, but no learned-memory promotion is
admitted.

Continue useful implementation and analysis within the invoking authorization.
The v1 experiment's budget is spent; another study requires a fresh, reviewable
protocol and eligible data. Do not use this prompt to create a chat/automation,
silently start an unbounded tuning campaign or change deployed defaults.

Maintain [document 17's](17_Memory_Reliability_Recovery.md) recovery contracts
before the reference M0–M7 packages below: immutable dataset-derived truth,
independent token scoring, complete case/arm membership, one state for multiple
queries, reload without old-source encoding, correction on the loaded state,
scope/lineage checks and unrelated-query retention. Run focused contracts before
native work. Every newly admitted experiment records actual updates, decoded
scores, costs, checkpoint and failed gates. Never promote on cached scores,
training loss or the old last-line-solvable fixtures.

Require reliable delayed recall, correction, retention and fresh-process
continuation before neural RSI promotion. Fixed verified-practice and rollback
plumbing may proceed earlier. C13's 10/10 evidence threshold gates learned
selection. Static Engram, Spark BF16 and native MiniCPM-V remain separate
qualified comparisons; they do not block this recovery. Authentic personality
training still requires permitted records and human fidelity review.

## A. Establish current state once, then inspect deltas

1. Inspect Git status, branch and recent commits before editing. Preserve user
   changes. The latest reviewed baseline was HEAD `280d4dd` on
   `codex/d2-capacity-runtime-pilot`; verify the live state instead of assuming it.
2. Read applicable repository instructions, `AGENT_CONTINUE.md`, the current
   snapshot in `execution_state.md`, this prompt, architecture document 15,
   `configs/architecture_implementation_status_v1.json`, and the resumption record.
3. Read architecture documents 02–05 and the implementation chapters relevant to
   the next package: identity 09, fast memory 10, artifacts 11, director 13 and
   scheduler 14. Consult stages 02–06 for scientific dependencies and data roles.
   Read the linked raw evidence before treating a result as an accepted baseline.
4. Identify the actual selected source/extension/manifest and report the last
   verified gate, incomplete gate and next discriminating action. Status prose
   is not authority to override contradictory code or raw measurements.
5. On subsequent iterations, read changed files, latest results and relevant
   contracts. Do not repeatedly restart environment setup or reread the entire
   history while leaving the implementation unchanged.

Current evidence to preserve:

- Corrected 4096 capacity passed at 28.5944 GiB; 8192 was not rerun. Fresh
  action menus remain weak. M1 procedure lifecycle works.
- M2/M3 state/gradient/restart mechanics work, but their exposed fixtures admit
  query-blind last-line success (32/32 and 44/44); associative recall is unproven.
  Retention fails and fresh-state replacement is not persistent correction.
- M4 v7 completed but was rejected: direct pattern 0/4 dev and 0/8 held-out;
  every delayed route selected fast. General fast+pattern 0/4 vs disabled 2/4.
- The recovery audit checks 434 nested M4 rows including duplicate restart
  references and six external controls, preserves its 427-generation accounting,
  and rejects cached-score tampering or removing an arm together with its metric.
- Latest full suite: 1,280 tests and 21 subtests passed, one restricted-context
  MPS-only skip; 19 focused recovery tests pass. Native training/restart subsequently
  completed on MPS at 11.0043/10.0163 GiB sampled peaks with no observed swap growth.
- Recovery pilot: 12/17 final gates fail, including all selective recall,
  correction/unaffected retention and general noninterference gates. Preserve
  its checkpoint and verdict; the M3 parent is an ownership reference, not an
  accepted general-memory baseline.

## B. Boundaries for this implementation loop

Implement code, tests, fresh eligible fixtures and documentation needed by the
selected package. Run bounded local experiments only within the invoking user's
authorization and the existing preregistered resource/source rules. Use current
tools and verified CLI help; do not invent commands or silently change protocols.

Reuse the locked environment and sealed original at `checkpoints/minicpm5_source`,
revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Preserve its tokenizer/template.
Do not substitute Base/SFT siblings or reacquire model weights routinely.
Protect `src/replica_lab/runtime_pilot.py`: do not modify, import, delete or stage it.
Keep selected checkpoints, raw failures, acquisition receipts and source seals.

Native model work on this Mac uses MPS with CPU fallback disabled, one exclusive
worker, and a **50 GiB (53,687,091,200-byte) sampled driver stop threshold**.
Retain the 512 MiB swap-growth and 15% minimum host-free limits with three
consecutive host samples; initial low free memory blocks launch. Pressure can
stop a job below 50 GiB. Do not change allocator/OS limits or raise the guard
further to rescue an arm. Inspect current worker ownership before launch; do not kill an
unrelated job. CPU is appropriate for host logic and small numerical contracts.
MPS is not Neural Engine execution. Profile actual tokens and all relevant state.

Use a newly versioned protocol for new 50 GiB experiments and bind its policy to
the actual guard report and code hashes. Both training/generation wrapper defaults
derive from `CapacityPolicy`; an explicit CLI/protocol value still overrides them.
Historical v1 protocols and the pinned D2 resume launcher retain their 40 GiB
rules. Do not edit their old results, silently replay them at a different budget
or reinterpret the old 48.08 GiB guard stop as an accepted run.

The current `muon8bit` stores momentum in int8; trainable parameters/gradients and
auxiliary AdamW state remain FP32. BF16 compute autocast is not BF16 parameter
residency. Qualify any new residency/offload implementation separately. Do not
start FP8 fine-tuning on the assumption that an inference checkpoint supports it.

Use unique output directories. Do not overwrite accepted/default manifests to
try a candidate. No unrelated cleanup, broad corpus campaign, routine download,
remote push or external deployment is implied. Retain the checkpoint sharing
and retention policy; collection/cleanup is separate from this task.

Continue routine reversible work without repeatedly asking for confirmation.
If an external input or permission truly blocks dependent work, identify the
exact missing dependency, continue independent work and ask only for that input.
Do not fabricate personal records or treat elapsed time as approval. Do not
create user-owned chats or recurring automations from this prompt. Delegate only
when the invoking instructions/environment authorize it, with disjoint ownership.

## C. Repeat this work cycle

1. **Select one gap.** Choose the earliest unresolved recovery gate in document 17. State
   one hypothesis, current baseline, expected artifact and acceptance criterion.
2. **Inspect the existing path.** Reuse components and production entry points.
   Diagnose representation, update, interface, data or evaluation failure before
   adding another mechanism. Avoid a disconnected parallel prototype.
3. **Make the work reviewable.** For a new experiment, declare source/data/code
   pins, trainable groups, optimizer/rates, objective/masks, chronology, splits,
   controls, numeric step/token/time/resource limits and rejection rules before
   measuring outcomes. Record this in a committed protocol when Git permissions
   allow, otherwise in an immutable hashed protocol with the limitation explicit.
4. **Implement one coherent change.** Preserve null paths, native ABI, state
   ownership, causality and independent verification. Keep unrelated fixes separate.
5. **Validate proportionately.** Run focused contracts first and required
   integration checks once they pass. Run the full suite when shared interfaces
   or repository requirements justify it. Do not repeat unchanged passing tests.
6. **Measure when justified.** After prerequisite checks, run one bounded native
   arm and its declared controls. Preserve actual answers/actions, receipts,
   failures and resource traces. A guard stop is a resource result, not a quality
   result. A lower loss or nonzero gradient is not task acceptance.
7. **Decide.** Accept, reject or defer under the predeclared rule. Preserve the
   parent after failure. Do not expose audit labels to retraining or change a
   threshold after observing the result. Diagnose a negative and choose a new
   discriminating action rather than repeating the same arm with more steps.
8. **Record and continue.** Update execution/status/handoff once, preserving
   historical reports. Make focused local commits when authorized and permitted;
   stage only intended files. Advance dependent work only when its gate is met.
   An external blocker in one track does not block unrelated useful work.

## D. Reference packages after the immediate recovery gates

### M0a — Completed baseline package; refresh only if invalidated

Inspect the fixed inference helper, menu builder and invalidation regression.
Verify source/selected artifacts and run focused regressions. Remeasure a small
inference point and the selected 4096-unit workload under the guard. Consider
8192 only after the smaller measurement supports it; no blind capacity ladder.
Separate inference from backward, and record writer/sense gradient mode.

Re-evaluate corrected repair menus with model/random/order controls and the
deterministic procedure reference. Log menu contents and chosen scores/indices.
Keep old tasks as regressions and use fresh families for capability conclusions.
Produce an exact baseline report, then move to M1; do not stop at a status edit.

### M0b — Optional Spark BF16 comparison

Keep MiniCPM5's source, selected artifacts and reports as the permanent reference.
Inspect and pin the official Spark-X2.6-4B BF16 config, implementation, tokenizer
and template before acquisition or execution under the active authorization.
No Spark weights or training qualification are provided by this prompt. Continue
M1–M6 on MiniCPM5 while an optional source/backend dependency is unresolved.

Build a separate adapter for the pinned `spark2_5` implementation; do not relax
the Llama-only adapter's checks. Inventory sliding/full attention, fused QKV,
head gating, tied weights, cache/position behavior, end markers and reasoning
modes as defined by the source. Choose complete-block ports from its actual
topology. Count tied tensors once in optimizer membership. Qualify native/null
parity, causal gradients, request isolation and export/reload before a memory
study. MLX inference availability does not qualify this PyTorch/MPS trainer.
Check compatibility with the locked runtime before changing dependencies. If
Spark needs a separate environment, isolate and pin it, retain MiniCPM's working
lock and disclose environment differences in the comparison.

Measure native inference, modules-only backward, and then a bounded full update
only if admitted by observed residency. The nominal 4B FP32 weight+gradient+
one-byte momentum estimate is already about 33.5 GiB before auxiliary state and
activations. Record actual tensors, precision, optimizer state and temporary
peaks; do not label 50 GiB as a fit guarantee. FP8/quantized deployment is a
later separate comparison, with downstream input-gradient checks for any frozen
quantized training arm.

Compare native MiniCPM5 versus native Spark first on the same fresh task families,
tools, observations, verification and output/time budgets. Pin reasoning mode
and count reasoning tokens and all costs. Then compare each backbone with/without
the same memory contract; keep D0/D1/D2 scopes explicit within each backbone.
Preregister numeric gain, regression and resource thresholds. Promote only after
independent reasoning/action gain, retention and resource gates pass; an upstream
benchmark or successful load cannot replace MiniCPM5 as the baseline.

### M1 — Episodes and a thin executable compiler

Build a small chronological family containing observation, later query,
distractors, correction, restart and reusable verified repair. Split source/world/
decision families before generating variants. Keep training, development,
calibration, promotion and final audit roles distinct. The old 2,000-target
release is exposed infrastructure, not a sealed final audit.

Connect one real payload/consumer through the existing registry, procedure,
image/writer producers and manifest publisher. Bind source/payload bytes,
producer/base versions, parent manifest, owner/mode/purpose/cutoff, epoch,
dependencies, dimensions, expiry and validation receipts. Test source changes
between evaluation and publication, stale parents, interrupted publication,
restart and descendant invalidation. Preserve atomic publication and rollback.
Do not require a generic compiler for every artifact before this thin path works.

### M2 — Useful learned memory on MiniCPM5

Start with one residual insertion; preserve complete native blocks and the
existing independent branches. Use contextual observation-only features and
head-specific writes. Follow document 15's SGDN-2 equation and gate/normalization
constraints. Ordinary SiLU is not bounded, and four copied matrices do not
provide four independent written stores.

Version the new memory writer, transition, state and reader; initialize only the
final residual projection to zero. Qualify one insertion, then two private ports,
before M3 shares state. Retain native attention/MLPs. Transfer contracts to a
qualified Spark/VLM adapter later, retraining projections and re-encoding source
memories for that representation. Do not stack a backbone switch, new memory
rule, shared depth and changed optimizer into one undiagnosable experiment.

Implement a pure functional transition for bounded multi-chunk training and a
detached host-owned deployment state. No I/O, counters or commits inside
checkpointed forward recomputation. Keep target/future tokens outside writer
and sense inputs. Check later-loss gradients through earlier writes after the
zero output projection opens.

Use memory-only-history students and checked full-context teacher/answer
comparisons, with retention replay. Change the association objective/curriculum;
contextual features alone already failed. Begin with a declared small diagnostic
budget, not an assertion that 32–64 updates must suffice. Evaluate novel values,
new bindings of seen values, changed templates, interference, correction and
restart against linear, exact/full-text, zero, shuffled and disabled controls.

### M3 — Shared depth

Both ports read one chunk-start snapshot; writes publish once after the chunk.
Specify the reducer and compare the same engine in private/shared forms. A
fixed convex mean of SGDN-2 next-state proposals is a project hypothesis;
uTTT's combined gradient update is different. Never sum gated deltas and assume
equivalence/stability. Test historical cross-depth gradients, gradient-cut and
port ablations, replay consistency and no same-chunk leakage. Match and report
state bytes, parameters, buffers and active compute honestly.

### M4 — Dynamic Engram path

Implement static deterministic multi-head n-gram lookup with contextual gating,
separate editable episodic state, and routing over record/fast/pattern/procedure
paths. Freeze tables within a version. Test early insertion and total table
budget; measure collisions, interference, latency and optimizer residency.
Qualify sparse operations on MPS. Start with deterministic routing; train read
and write routing separately from checked outcomes. Learned top-k selection is
not the paper's hash addressing. Consolidation creates evaluated new versions.

### M5 — Bounded recursive self-improvement (RSI)

Connect real outcome → diagnosis → eligible practice → isolated candidate →
independent gain/retention evaluation → publish/reject/defer → restart → later
task. Preserve the best practical baseline as well as the candidate parent.
Opt-in toy acceptance must not silently become default promotion. Inspect
`GatePolicy` and the applicable protocol; preserve independent-pair and retention
requirements rather than treating a few repeated variants as independent data.

Retain the scientific D0 plain full / D1 frozen-base extensions / D2 full
augmented comparison before selecting the deployment base. A failed eight-step
action diagnostic is not a permanent ban on developmental co-adaptation. A thin
fixed-baseline loop may precede selection but must be labelled accordingly.

Calibrate success/action-value heads on disjoint actual outcomes before routing
or expenditure depends on them. Keep the verifier independent. Start with fixed
practice; C13 has 5 accepted and 8 rejected/deferred groups against a minimum
of 10 in each class, so the collection gate has not admitted a learned selector. Reaching 10/10 does not prove selection utility.

Run ten chronological T0/T1/T2/T3 cycles when dependencies pass: no persistence,
evidence only, fixed verified learning, learned selection plus learning. Include
replay-matched T2, retention, distribution changes, no-improvement periods,
restart, bad-candidate rejection and rollback. Count every arm's learning,
generation and verifier costs. Report library and neural gains separately.

Implement RSI in this order:

1. **Verified practice → candidate SFT.** Let the current system diagnose real
   failures and propose bounded tasks, self-edit data or reasoning/tool traces.
   Independently check outcomes; verify executable intermediate steps where
   possible. An answer-correct rationale is not automatically step-correct.
   Train named memory/skill/reasoning components with fixed retention replay in
   an isolated candidate. Generated practice remains distinct from authentic
   history and observed facts.
2. **Learn practice/update selection.** Collect accepted and rejected candidate
   outcomes, then compare a learned selector with fixed replay-matched practice.
   Reward downstream gain after actual adaptation, not self-reported quality.
   Begin with ranking/bandit selection; add RL only after reward/data/resource
   qualification. Hold training, calibration, promotion and audit roles apart.
3. **Later code or architecture proposals.** Generate isolated versioned changes;
   use separately controlled tests and held-out evaluation. Candidate changes
   cannot edit their judge/audit, promotion rules, resource guard, source scope or
   protected identity goals and cannot publish themselves. Keep review authority
   external to the candidate. Software gains and weight-learning gains are
   distinct reports.

Bind each cycle to parent/candidate hashes, evidence cutoff, generated-data
lineage, trainable groups, replay, verification, independent future-task scores,
retention and full cost. Test rejection, source revocation, rollback and fresh
process restart. Include later adaptation on new reasoning/memory families at
matched exposure/update budgets before claiming improved learning ability.
STaR/Absolute Zero, SEAL and DGM motivate different parts of this design; consult
document 15's primary sources, and do not call the combined project a reproduction
or claim unbounded/accelerating improvement from a small accepted cycle.

### M6 — Learned timescales

After simple-loop evidence, specify and compare learned levels with distinct
parameter sets, losses, context streams, accumulation, frequencies, buffers,
reset/persistence and knowledge transfer. Compare fixed versus learned update
coefficients. Scheduling fast state, procedures and occasional jobs is not CMS.

### M7 — MiniCPM-V vision, reasoning and visual memory

Pin available MiniCPM-V-4.7-1B model/processor/native code. Its width-1024,
24-layer Qwen3.5 hybrid is not the MiniCPM5 Llama decoder. Candidate ports 7/15,
observer 23, require native parity. Preserve recurrent/convolution/KV state and
request-specific M-RoPE/`rope_deltas`; keep added memory separately owned.
Transfer method/contracts, not extension weights; retrain projections/re-encode.

Use its pretrained tower, native window/spatial mergers and processor. Qualify
native image generation first; then freeze existing vision/text components and
train the new visual writer/memory interfaces. Build delayed-use episodes that
remove the original image before the later query. Preserve image/frame hashes,
crop coordinates, timestamps, source permissions and correction chains.

Train/evaluate visual reasoning with disjoint scenes/documents/sessions: spatial
relations, OCR/document evidence, temporal changes and tool-verifiable outcomes.
Keep actual/wrong/no-image, caption-only, shuffled-memory, correction/restart and
text-retention controls. Only checked steps serve as reasoning supervision.
If justified, unfreeze declared merger/last-vision/text groups and remeasure;
label partial tuning honestly. `full_vlm` needs the full declared native parameter
scope, gradients and a separate measured budget. Re-encode after producer changes.

Treat MiniCPM-V-4.7-35B-A3B as a separately qualified teacher/inference candidate.
Nominal BF16 weights alone exceed 50 GiB; active experts do not determine total
resident memory. Verify teacher outputs and keep audit data inaccessible. Do not
attach an unrelated vision encoder to Spark in the first comparison; that would
require a new alignment/projector, paired visual training and retention study.
Weights are not currently local; continue text work instead of downloading by
assumption or claiming a native VLM pass.

## E. Personality and semi-cognitive work in parallel

Reuse the existing identity archive/store, modes, native reader and fictional
scorer. Add the approved-source slot encoder and consistent context binding
through decisions, persistence and replay. Preserve subject, snapshot, purpose,
cutoff and source/branch ownership. Train conditional priorities with exceptions;
style and factual recall are separate targets.

Compare generic, approved-summary, archive-retrieval and neural conditioning with
equal information. Hold out whole decision families. Score choice, clarification,
counterfactual changes, temporal fidelity, unknowns, competence and branch
divergence separately. Authentic fidelity requires permitted human testimony and
human review. Fictional cases can qualify integration while that is unavailable.

Keep exact goals/obligations in task state and approximate evidence/hypotheses in
the core. Preserve sense → core → immutable decision → action → actual outcome;
new retrieval creates a new decision. Do not reuse sense-pass KV as action KV.
Rank typed actions and revise plans from real outcomes. Compare no-core,
feed-forward and extra-inference controls before deeper recurrence or CTM/Coconut.
Agent experience never becomes human autobiography; skill updates cannot
silently change protected foundational goals.

## F. Reports, stopping and completion

For every completed package, report:

1. What changed and the hypothesis it addresses.
2. Exact commands, commit/code pins and raw evidence paths.
3. Implemented, tested, trained, evaluated and still unrun scope separately.
4. Baseline/control results, reasoning/retention/memory metrics, actual guard
   policy, complete costs, failures and accept/reject/defer decision.
5. The next concrete action or exact external dependency.

Stop a native job on its guard, nonfinite state/loss, leakage or invalid
publication conditions. Preserve evidence, classify the failure and continue
safe diagnosis or independent work. Do not bypass a blocker, conceal failed
cycles or repeatedly announce the same blocker without advancing anything.
Respect user pauses/cancellation and execution budgets. If the environment ends
the run, leave a concise resumable handoff with no assumed future background work.

The first integrated milestone is: observe an unseen association, use it after
context removal and process restart, apply a dated correction, compile/reuse a
verified repair on an unseen later task, and reject/roll back a damaging
candidate while preserving previous competence. A fictional personality
condition can exercise integration; authentic fidelity is a separate gate.
After that milestone, continue admitted dependent packages. Claim overall
completion only for objectives actually demonstrated under their stated gates.
