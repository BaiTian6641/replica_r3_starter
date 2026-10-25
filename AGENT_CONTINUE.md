# Coding-agent continuation instruction

**Current priority — v5 memory repair is complete and rejected; diagnose
storage/read separation next.** Read the [v5 experiment record](docs/experiments/20261024_Memory_Reliability_Writer_Binding_Results.md),
[execution state](execution_state.md) and [canonical verdict](reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json).
Then follow [memory recovery](docs/architecture/17_Memory_Reliability_Recovery.md)
and the separate [MiniCPM5 vision plan](docs/architecture/18_MiniCPM5_Vision_Training_and_Continual_Learning.md).

The writer-only arm completed 48 updates / 192 cases with exactly 14 changed
tensors. Candidate fast and disabled both score 56/192; initial facts,
corrections and unaffected facts remain zero. This is not useful learned memory
and is not selected. Exact fact retrieval works as a control, but the reference
bundle rejects on general correctness and two full-context non-regressions.
The corrected scorer authenticated 32 images/source bindings and a seven-answer
fresh-process restart. These are mechanical passes, not quality acceptance.

Final clean-export validation is **1,311 tests +21 subtests in 148.69 s**, with
**49 focused reliability tests**. Scorer corrections are `eff2079` and `5aa7e75`;
eval01 is preserved as failed infrastructure, and eval02 is canonical. One
48,948,272-byte extension was saved without a base copy. Total guarded native
wall across training, two evaluation replays and restart is 1,751.69 seconds.

Do not repeat writer/reader training on this exposed set. Implement a fresh
storage/read-separation probe and use explicit copy/pointer only when observed
values are recoverable, under a new committed protocol. Isolated vision may
proceed with memory/core/identity disabled; it cannot qualify this memory path
or integrated RSI. Spark is later work. Preserve all historical evidence and the
untracked protected `src/replica_lab/runtime_pilot.py` unchanged.

## Resume from evidence

Read these in order, then inspect affected code and raw evidence:

1. [Execution state](execution_state.md): current review and next package first.
2. [Reproducibility and memory recovery](docs/architecture/17_Memory_Reliability_Recovery.md):
   clean-checkout, context/scorer, binding/readout and targeted repair gates.
3. [Loop-agent prompt](docs/architecture/16_Loop_Agent_Instructions.md): complete
   ordered implementation loop, preservation rules and reporting.
4. [MiniCPM5 vision plan](docs/architecture/18_MiniCPM5_Vision_Training_and_Continual_Learning.md):
   V0–V4 contracts, training, visual memory, RSI and later Spark.
5. [Research/backbone plan](docs/architecture/15_Research_Crosscheck_and_Backbone_Plan.md),
   [implementation status](configs/architecture_implementation_status_v1.json),
   applicable component/stage documents and current Git changes.

Documents 17 and 18 control immediate ordering. M0–M7, E1–E6, C01–C18 and
scientific-stage IDs remain component/claim references, not instructions to
restart completed work or prioritize Spark. MiniCPM5 stays the baseline; its
new visual projector is trained separately. MiniCPM-V-4.7-1B is an optional
pretrained reference/teacher, with 35B-A3B a separately qualified larger option.
No new model or capability is qualified by this documentation update.

## Current accepted evidence and limits

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
- The post-package full integration suite passed **1299 tests / 21 subtests in
  137.53 s** (zero failures/errors/skips); 37 focused reliability tests also
  passed. The 50 GiB guard, source seals and all failed native reports are preserved.

## Next work package and progression

1. **Reproducibility and integrity first.** Package the missing scorer/builder/test
   dependencies in intended tracked source; prove clean-export import/tests.
   Authenticate consumed context spans/records/scope/status. Fix explicit route
   and effective-gradient accounting. Version prospective correctness/nonregression
   gates; preserve the historical parity rejection and native/smoke provenance.
2. **Diagnose and repair memory.** Separate stored binding, addressing and output
   copying, using fresh families and wider retention. E4 contextual features
   already failed 0/8 heldout; do not repeat that arm unchanged. Run one newly
   declared targeted repair, then independently accept/reject/defer and verify
   correction/restart. Preserve all negative candidates and count their cost.
3. **MiniCPM5 V0–V2.** After the completed repair package, qualify fused visual
   input, then projector-only alignment and declared adapter/LoRA instruction/
   reasoning training. Keep memory/core/identity disabled for independent vision
   while any memory-quality gate remains unresolved.
4. **V3 visual memory.** Require accepted perception and the chosen memory path;
   distinguish exact image retrieval from learned state. Test image removal,
   multiple later queries, correction, unaffected facts and fresh-process reuse.
5. **V4/M5 bounded RSI.** Begin verified candidate SFT only after relevant quality,
   retention and verifier gates; then longitudinal fixed/replay-matched controls.
   Learned selection, RL and code/architecture changes are later separate studies.
6. **Later Spark.** After the MiniCPM5 visual-memory/RSI work has documented
   outcomes, qualify native Spark text, then new visual connector/adapters and
   model-specific memories. Reuse methods and data, not projection weights.

The latest canonical [reader-only verdict](reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
remains reject. It records 48 updates, 64/192 gated fast answers, 32 authenticated
memory snapshots and 7/7 restart answers. Native gate-off accounting reports 0
versus 64 independently derived. The 32 general-answer cases repeat only four
questions; neither this bypass result nor the old M2/M3 last-line-solvable
fixtures demonstrate broad retention or learned associative memory.

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
