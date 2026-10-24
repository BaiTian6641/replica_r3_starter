# Coding-agent continuation instruction

Current documentation review: 24 October 2026, against M4 v7 implementation
`ad09a77` and protocol commit `d1364fc` on `codex/d2-capacity-runtime-pilot`.
This update records the completed M4 static-arm rejection and its evidence
boundaries. It does not launch an agent, native experiment or automation.

## Resume from evidence

Read these in order, then inspect the affected code and raw reports:

1. [Execution state](execution_state.md) — current snapshot first; older entries
   preserve historical results.
2. [Research cross-check and backbone plan](docs/architecture/15_Research_Crosscheck_and_Backbone_Plan.md)
   — current M0–M7 construction order, research distinctions and acceptance gates.
3. [Loop-agent instructions](docs/architecture/16_Loop_Agent_Instructions.md)
   — the complete iteration, evaluation and handoff protocol.
4. [Resumption record](docs/experiments/20261023_Resumption_Record.md),
   [implementation status](configs/architecture_implementation_status_v1.json)
   and the relevant component specification.
5. [Start instructions](AGENT_START.md), applicable repository instructions,
   recent Git history and current worktree changes.

The current M0–M7 order supersedes older next-task paragraphs. E1–E6, C01–C18
and S-stage identifiers remain useful component and scientific-gate references.
Preserve dated experiment results; an old measurement is not a new result.

MiniCPM5 remains the required baseline. M0b adds an optional **Spark-X2.6-4B
BF16** comparison with its own native adapter and resource/quality gates. M7
uses MiniCPM-V's pretrained vision path; the later larger model is
**MiniCPM-V-4.7-35B-A3B**. New memory layers and bounded RSI follow the gates in
document 15; none of these planned paths is qualified by the documentation.

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
- The post-M4 full integration suite passed **1262 tests / 21 subtests in
  133.73 s** (1283 JUnit entries, zero failures/errors/skips); the focused M4
  suite passed 9 tests. The 50 GiB guard, source seals and all failed native
  reports are preserved.

## Next work package and progression

**M3 retention repair and M4 static optimization are rejected:** M3 delayed
behavior is preserved, but general development remains 0/2 and adversarial
memory remains 0/4 versus disabled 2/4. M4 direct pattern is 0/4 dev and 0/8
held-out/regression; its router score is fast-memory behavior, not static
benefit. Do not promote M4 or use fast scores to admit M5. Retention repair and
genuinely covered non-training static utility remain open. Any M5 bounded-RSI
arm requires a new predeclared parent/data/control protocol; then M6 learned
timescales and M7 native vision. Authentic identity fidelity still requires
permitted source material and human review.

**Current dependency disposition:** M0b is dropped as optional and unadmitted;
no local Spark BF16 source/compatible isolated environment/adapter exists. M5
is blocked by M4/M3 quality and retention gates plus C13's 5 accepted / 8
rejected-or-deferred groups versus 10/10. M6 is blocked until a simpler loop
survives retention and noninterference. M7 is blocked on local 1B VLM
weights/revision, a separate adapter and native multimodal/cache parity; larger
GGUF artifacts are not the target. Personality fictional integration is
implemented; authentic training/fidelity awaits permitted records and human
review. These are explicit blockers, not completed packages.

Then use this order, advancing only on the declared gate:

- **M1:** fresh chronological, source-separated episodes and one thin typed
  compiler path that reuses the ledger, catalog, verifier and publisher.
- **M2:** one useful adaptive-memory path with distinct head writes, bounded
  SGDN-2-inspired recurrence, pure multi-chunk training state and delayed-use
  supervision. Separate functional training from detached deployment state.
- **M3:** compare shared-depth and private memory using the same update engine,
  information and budget. Shared-state semantics and gradient aggregation
  must be defined before claiming uTTT-style behavior.
- **M4:** bounded static Engram-style lookup plus versioned episodic routing;
  evaluate routing, collision behavior, invalidation, deletion and cost.
- **M5:** candidate training, independent comparison, publication/rejection,
  recovery and ten chronological T0/T1/T2/T3 cycles. Preserve the fair
  developmental D0/D1/D2 study before freezing a selected deployed parent.
  RSI progresses from verified practice/candidate SFT to learned selection;
  later code/architecture proposals keep evaluation and promotion external.
- **M6:** distinct learned multi-rate levels and objectives only after useful
  simpler adaptation is established. Timers alone do not implement HOPE.
- **M7:** a separate native MiniCPM-V-4.7-1B adapter and qualification.
  Transfer contracts and methodology; retrain interfaces and re-encode
  memories for its Qwen3.5/hybrid/vision representation.
  Start with native visual parity, then frozen-tower visual-memory alignment,
  verified visual reasoning and separately justified co-adaptation.

Personality/identity work runs alongside M1–M3 where permitted source and
evaluation are available. Use equal-information no-identity, prompt-only and
native-reader controls. Calibrate uncertainty before allowing it to control
actions or learning. Missing authentic sources block fidelity claims, not
independent memory/compiler construction.

The first integrated milestone is verified observation → useful delayed
memory → restart → correction/revocation → compiled procedure → improved
later unseen task → harmful-candidate rejection and rollback. Passing
contracts alone does not complete that milestone.

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
