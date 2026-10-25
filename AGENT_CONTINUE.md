# Coding-agent continuation instruction

**Current priority — optimizer ablation after a confirmed vision learning failure.**
R1–R4 are repaired; V0 accepted interface-only; measured V1 rejected. New
one-write/interference diagnostics recover storage incrementally (0.929→0.877→0.853),
so the earlier multi-fact failure is addressing/correction, not a basic write failure.
Frozen vision features support a linear readout (combined dev0.4167 vs chance0.0833),
but the tiny overfit cohort cannot fit its train worlds at16 or32 updates (0/8 on
every control). Fit analysis confirms argmax is wrong on all8 (learning, not decoding);
fresh-process extension reload is bitwise exact. A bounded SGD ablation now tests
whether the Muon route/clip is the bottleneck before any connector change or V1 retry.
Keep V2/LoRA and learned V3/neural RSI unadmitted; preserve v5 and V1 rejections,
exact-source engineering controls, Spark backend blockers and the 50 GiB guard.

The v5 writer-only arm is complete and rejected: 48 updates, 14 changed tensors,
fast and disabled both 56/192 with zero factual recall. Selective exact context
is 184/192 versus full context 176/192; the reference still fails its declared
general/nonregression gates. The dependency, changed-span, route-count and
remaining caller/schedule/representation contract fixes are complete.
Thirty-two memory snapshots and seven recorded restart answers authenticate;
mechanical parity is not learned-memory quality.

The review independently passed 49 focused tests, checked all 27 implementation
pins, regenerated v4 data byte-for-byte and reproduced the scorer over 1,920
answers. The full 1,311-test +21-subtest pass is separately recorded experiment
evidence. No new native execution or training is reported by this design update.

Use [the complete loop prompt](docs/architecture/16_Loop_Agent_Instructions.md),
[recovery](docs/architecture/17_Memory_Reliability_Recovery.md),
[integrated design](docs/architecture/19_Integrated_Memory_Learning_and_RSI_Design.md),
[Spark FP8/BF16 plan](docs/architecture/20_Spark_X2_6_FP8_Preparation_Plan.md) and
[data curriculum](docs/08_Dataset_Acquisition_and_Curriculum.md).
MiniCPM5 and the 50 GiB guard remain fixed. No additional blind memory arm is
required before V0; learned V3/RSI still require useful memory and retention.
Spark native qualification follows reusable interfaces and backend gates, not
RSI success. Preserve historical evidence and the protected untracked
`src/replica_lab/runtime_pilot.py`.

## Resume from evidence

Read these in order, then inspect affected code and raw evidence:

1. [Execution state](execution_state.md): current review and next package first.
2. [Reproducibility and memory recovery](docs/architecture/17_Memory_Reliability_Recovery.md):
   caller-bound contexts, exact schedule, representation rebuild and source gates.
3. [Loop-agent prompt](docs/architecture/16_Loop_Agent_Instructions.md): complete
   ordered implementation loop, preservation rules and reporting.
4. [MiniCPM5 vision plan](docs/architecture/18_MiniCPM5_Vision_Training_and_Continual_Learning.md):
   V0–V4 contracts, training, visual memory and bounded RSI.
5. [Research/backbone plan](docs/architecture/15_Research_Crosscheck_and_Backbone_Plan.md),
   [implementation status](configs/architecture_implementation_status_v1.json),
   applicable component/stage documents and current Git changes.

Documents 16–20 and the dataset curriculum control immediate ordering. M0–M7, E1–E6, C01–C18 and
scientific-stage IDs remain component/claim references, not instructions to
restart completed work or prioritize Spark. MiniCPM5 stays the baseline; its
new visual projector is trained separately. MiniCPM-V-4.7-1B is an optional
pretrained reference/teacher, with 35B-A3B a separately qualified larger option.
No new model or capability is qualified by this documentation update.

## Prior component evidence and limits

- The native MiniCPM5 adapter, record memory, shared recurrent workspace,
  independent readers, exact task journal, bounded tools and independent
  verification are connected. Mixed-family completion is 4/4 with and without
  the action-kind head; that head has no measured benefit. Restart passes
  12/12 acceptance checks. These are narrow synthetic results.
- Free-form action learning remains negative. Candidate ranking historically
  scored 17/17 repairs (15 held-out); the numeric-string menu correction in
  `6e8ec34` changes candidate sets, so rerun fresh matched controls before
  using that aggregate as evidence for the corrected interface.
- Corrected inference passed the small and selected 4096-unit workloads; the
  latter reached 28.5944 GiB with detached writer/sense and 28 nonzero backward
  gradients per point. The 8192-unit point was not rerun. Fresh corrected menus
  fail on distractors; old-case regression and the deterministic procedure pass.
- M2's shared-value SGDN-2 diagnostic passed 4/4 dev and 4/4 held-out after
  context removal, above linear and null controls. Its candidate retention is
  **0/2** (`amber`) versus parent 2/2, an ungated training-recipe failure also
  present in M3. Its unique-value arm memorized training values and was rejected.
- M3 shared and private ownership both passed 8/8 dev+held-out, update-two
  cross-chunk gradients, gradient-cut and fresh-process image restart. Private
  uses twice the logical state; no capacity-matched or uTTT claim is made.
  Both candidates nevertheless fail retention replay **0/2** versus parent 2/2.
  M4 proceeded only as a separate mechanism comparison; retention remains open.
- M4 v7 completed 4 source-LM +16 delayed updates and two extension saves.
  Direct pattern is 0/4 dev and 0/8 held-out/regression; all delayed router
  rows selected fast with incomplete coverage. Fast+pattern adversarial is 0/4
  versus disabled 2/4, so the independent verdict rejects four quality gates.
  Sixteen/20 checks pass, including authenticated same-process extension/image
  restart. Fresh-process restart was not run after rejection. The CPU collision
  audit reconciled 51 hash-call prompts and found zero distinct-n-gram
  collisions; actually routed covered-case utility remains unmeasured.
- Procedure compilation, execution, candidate rejection, single-writer atomic
  publication and ancestor rollback exist; the compiled procedure passed
  3/3 held-out cases plus dev. Do not rebuild these as missing primitives.
  Coordinated compiler transactions, complete payload binding and an
  integrated neural improvement loop remain open.
- C16 has a native identity reader after the workspace path, persistent
  identity contracts and fictional tests. Its encoder/training, authentic
  source-based evaluation and fidelity are open. The success head has a
  diagnostic fit but is uncalibrated and has no success authority.
- Historical reader-only validation passed **1299 tests / 21 subtests in
  137.53 s** (zero failures/errors/skips); 37 focused reliability tests also
  passed. The 50 GiB guard, source seals and all failed native reports are preserved.

## Next work package and progression

1. **R1–R4 complete at `757ace1`.** Reuse the recorded consumer/schedule/
   representation and source-export qualification in document 17. Recheck only
   contracts affected by new changes; do not restart this completed package.
2. **Isolated MiniCPM5 V0.** After those fixes, qualify real visual tensors, masks,
   positions, input gradients, disabled text parity, request isolation and reload.
   The rejected v5 repair already has a measured outcome; do not repeat blind
   writer/reader training as another V0 prerequisite.
3. **V1/V2.** Admit paired data under the dataset curriculum, train the projector,
   then declared small adapters for instructions/verified reasoning with text replay.
4. **Memory diagnosis and V3.** Fresh storage/addressing/readout probes can proceed
   independently. Learned visual memory requires accepted perception and chosen
   memory quality, dated correction, unaffected retention and fresh-process reuse.
5. **V4/M5.** Run one externally verified candidate cycle, then matched T0–T3
   longitudinal comparisons. Keep rejection/rollback and audit authority external.
6. **Spark/data preparation now.** Pin the official FP8 adaptation base and BF16
   control, inspect backend support and prepare eligible small data manifests.
   Later native qualification uses reusable V0 contracts; RSI success is not a
   prerequisite. No unsupported FP8 training or automatic backbone promotion.

The current canonical result is
[v5 eval02](reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json).
The earlier reader-only result and its old routing/equality defects are
historical. The current route counts are 128 fast /64 disabled with accumulated
update-level gradient evidence; broad retention and learned memory remain open.

Reuse the procedure compiler, exact ledger, artifact/publisher and rollback paths.
M4 static-pattern utility is still unproven; Engram and multi-rate mechanisms
remain optional studies. C13's collection gate applies to learned selection,
not deterministic verified-practice plumbing. Personality work uses approved
sources and independent fidelity review; source absence does not block unrelated
implementation. Keep factual verification independent of identity conditioning.

## Preserve scope, artifacts and resource limits

Reuse the locked environment and sealed original
`checkpoints/minicpm5_source`, revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Keep tokenizer/template provenance,
selected D0/D1/D2 checkpoints, E1/E2/E3 extensions, failures and raw reports.
Missing/corrupt source must fail offline. No routine downloads, duplicate
native saves, historical-report rewrites or unrelated cleanup.

Preserve the untracked `src/replica_lab/runtime_pilot.py`; do not modify,
import, delete or stage it without an explicit change in scope. Preserve all
other user changes. Do not invent a remote or push.

Native training/generation on this Mac uses MPS, CPU fallback disabled, one
exclusive native worker, and the new default **50 GiB driver** threshold
(53,687,091,200 bytes), with unchanged sustained 512 MiB swap growth and 15%
minimum host free memory over three host samples. Initial low host-free memory
blocks launch. Samples can miss peaks. Historical explicit 40 GiB protocols,
reports and the pinned D2 resume launcher retain their original limits; create
a new version with matched policy/code hashes for new 50 GiB work. This changes
no allocator limit and establishes no new model fit.
CPU remains appropriate for host I/O and small contract tests. MPS is not ANE.
Use experimental `muon8bit` plus the documented auxiliary FP32 AdamW groups
unless a comparison explicitly changes the optimizer. Record actual trainable
membership, gradients, changed tensors, tokens, time and memory.

BF16 source weights/compute do not imply BF16 parameter residency in the current
FP32 trainer. Muon int8 momentum does not enable FP8 weight training. Qualify
Spark's real gradients, optimizer ownership and export/reload separately.

D0 means plain full tuning; D1 means frozen base/new modules; D2 means full
augmented tuning, including embeddings/head. Do not silently replace full
scope with LoRA/freezing to fit. A valid offload/residency implementation is
separate work. The exposed 2,000-target release is an infrastructure fixture;
fresh family-level evaluation and S1 admission still gate a broad campaign.

## Work and report discipline

Follow the user-authorized [staged-commit policy](docs/maintenance/20261025_Repository_Hygiene.md).
Commit each completed, validated package promptly using explicit paths. Preserve
concurrent edits, local payloads and the protected ignored pilot.

For each iteration: establish current evidence → choose the smallest
discriminating hypothesis → record protocol and controls → implement a
coherent change → run meaningful checks → run an admitted bounded experiment
when authorized → decide accept/reject/inconclusive → update evidence and
handoff. Use fresh output directories and preserve failures.

Continue through authorized work without repeatedly asking for confirmation.
A document is not permission to start an unrequested campaign, launch other
chats or create an automation. Delegate only when the active user/environment
instructions authorize it. Commit focused changes when authorized and
permitted; never claim an uncommitted protocol is a preregistered Git commit.

Report implemented, tested, trained, evaluated and unrun separately. Include
the exact parent/revision, commands, raw evidence, comparisons, costs, retained
artifact, current limitation and next smallest action. Reuse passing checks
unless a change invalidates them. Stop an unsafe or invalid experiment,
preserve its evidence, and continue independent work where possible.
