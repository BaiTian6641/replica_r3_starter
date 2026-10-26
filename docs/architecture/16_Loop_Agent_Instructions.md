# Loop-agent execution instructions

**Reconciled 25 October 2026 with repairs at `757ace1`.** This is the complete
continuation prompt. When invoked for implementation, carry out the ordered
packages within the user's active authorization. Reading or editing this prompt
does not launch training, download weights/corpora, create a chat or schedule work.

Continue TAIS Obsidian / Replica R3 in the existing checkout. Keep sealed
MiniCPM5-2B as the baseline and the 50 GiB native guard. New training uses
`adamw_fused` after integration checks; historical Muon experiments and their
provenance remain unchanged. The current priority is **constant-output priors
and caption alignment on fresh sealed data**, not another larger arm. Native
full-size resume remains unqualified on optimizer-state transfer memory.
Visual memory and RSI remain gated by accepted perception and useful memory.
Prepare Spark FP8/BF16 interfaces alongside this work without replacing the
MiniCPM5 baseline.

## 1. Establish current evidence once

Inspect Git status/logs, applicable instructions and active artifacts. The last
repair qualification is `757ace1`; verify what another session changed before acting.
Preserve unrelated work and historical negative evidence. Read:

1. [Execution state](../../execution_state.md) and [continuation](../../AGENT_CONTINUE.md).
2. [Verification and memory recovery](17_Memory_Reliability_Recovery.md).
3. [MiniCPM5 vision](18_MiniCPM5_Vision_Training_and_Continual_Learning.md) and
   [integrated design](19_Integrated_Memory_Learning_and_RSI_Design.md).
4. [Spark FP8/BF16 preparation](20_Spark_X2_6_FP8_Preparation_Plan.md),
   [dataset curriculum](../08_Dataset_Acquisition_and_Curriculum.md) and relevant
   component/status records.
5. The actual protocol, canonical verdict, raw report, implementation and tests
   for the chosen package.

On later iterations read changed files and relevant evidence; do not restart
completed work or reread the whole archive. M0–M7/E1–E6 remain component/claim
identifiers, not a different priority queue.


Current reconciled evidence:

- Sustained fused full-scope training passes 12 updates /381 tensors with
  0.11781 updates/s, 52.2464 steady processed tokens/s and zero swap growth.
  Full-size native resume remains unqualified on optimizer-state transfer.
- Causal frozen-writer refresh improves learned recovery from36/240 to110/240
  but fails correction, unaffected and unknown-abstention gates; fresh-process
  parity passes. This is hybrid retrieval plus neural refresh, not V3 quality.
- The fused stage diagnostic recovered engineering and reload but rejects
  learned quality: audit6/24, caption token-F1 0 and constant `yes` on all24
  correct-control answers. Do not enlarge the projector or unfreeze the backbone.
- Verified-learning v2 now encodes apply-once versus compose-twice explicitly;
  supplied-rule arithmetic is separate from unknown-rule inference.
- Historical v1 vision scores validate through the explicit supplemental-lineage
  replay route; original reports are unchanged.
The reviewed state is:

- v5 writer-only D1: 48 updates / 192 scheduled cases; 14 changed tensors /
  5,251,072 parameters, 53 unchanged tensors. Original base, readers and gates
  frozen. Fast and disabled score the same 56/192; all 128 factual cases fail.
- Selective exact context: 184/192, including all 128 factual cases. Full context:
  176/192. Reference rejects on general correctness and two correct-to-wrong
  regressions. Canonical eval02 rejects 11/31 gates; no candidate is selected.
- Required scorer/builder/test dependencies are tracked. Changed-span verification,
  explicit gate-off accounting and prospective reference gates were repaired.
  Do not report those original defects as still open.
- Review independently verified 27 implementation pins, byte-identical v4 data
  regeneration, 49 focused tests and the CPU scorer over 1,920 saved answers.
  The 1,311-test +21-subtest full clean-export pass is separately recorded
  experiment evidence, not a fresh review full-suite run.
- Thirty-two memory snapshots/source bindings and seven recorded restart answers
  authenticate. These are not visual inputs, memory-quality passes or power-loss
  durability tests. Route counts are 128 fast /64 disabled; gradient attribution
  is accumulated-update only. Retention has 16 distinct evaluation questions,
  still too narrow for a general retention claim.
- **R1–R4 complete at `757ace1`:** actual consumers bind scope/query/policy,
  route audits enforce the seeded update/case schedule, and rebuild preserves
  representation identity. The [qualification](../../reports/memory_contract_repairs_20261025/qualification.json)
  records 140 integrated tests, fresh-process replay and 1,372 source-export
  tests +21 subtests. These are recorded software results, not new native work.
- Earlier E4, reader-only, M3 and M4 quality failures remain rejected. M1's narrow
  procedure publication/restart/rollback path is reusable.
- Vision integration and accepted neural RSI are unqualified. The identity reader
  is untrained; authentic fidelity is unmeasured. New Spark and dataset metadata
  are preparation records, not acquired weights or admitted training corpora.

## 2. Completed prerequisite: contract repairs

R1–R4 passed at `757ace1`; reuse that qualification. The checklist below records
invariants to preserve when changing these interfaces, not a request to repeat
the completed repair package:

1. Bind exact-context verification to the actual consumer's expected scope,
   query, policy and limit. Verify canonical eligibility, relevance/conflict/
   abstention and consumed spans. Reject foreign branches, forged empty status,
   query-route substitution, stale/revoked records and changed payloads. Exercise
   the actual `request_args`/production consumer, not only a verifier helper.
2. Reconstruct the declared seeded training schedule. Require the exact ordered
   update indices, per-update case grouping and declared repeat policy; reconcile
   rows and summaries. Reject duplicate/missing/reordered/substituted updates and
   boolean indices. Preserve true route/backward versus gradient distinctions.
3. Version raw versus fact-binding representations through producer/config/cache/
   snapshot identity. Rebuild with the original representation and declared view
   semantics. Test unchanged rebuild parity, correction/revocation views,
   save/load/rebuild, fresh source binding and explicit legacy migration. Do not
   expect a current-view rebuild to equal an old incremental corrected history.
4. Keep source packaging reproducible and handoffs consistent. New artifacts get
   new pins; historical protocols, rejected candidates and eval01 failure remain
   untouched. Record committed versus uncommitted source seals accurately.

Reuse `answer_audit.py`, `reliability_scoring.py`, `runtime/exact_memory.py`,
`runtime/persistent_fast_memory.py` and existing runners/scorers. The focused
starting command is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_exact_memory.py tests/test_memory_reliability_v2.py \
  tests/test_memory_reliability.py
```

Add meaningful mutation/regression tests for changed contracts. Validate a clean
tracked export when permitted, otherwise a hashed source bundle with the limit
stated. Broaden checks only for changed shared contracts or unresolved concerns.
A native rerun is not needed merely to prove a deterministic software boundary.

Exit with repaired contracts, reproducible source and explicit v5 rejection.
The completed rejected v5 experiment satisfies the prior measured-repair-package
prerequisite. Do not require another unmotivated 48-update writer/reader arm before
isolated V0. Learned-memory quality is still required for dependent V3/RSI claims.

## 3. Independent preparation and memory diagnosis

Prepare the Spark source/backend matrix in document 20 and the dataset intake
manifests in document 08 without allocating an MPS worker. Keep all new sources
quarantined until their asset terms, content, split and hash checks pass. The
catalog has 34 external sources, including 22 pinned metadata records; this is
not the amount of usable local training data. Start with eligible small V0 assets,
then a capped V1 manifest. Respect the proposed 25 GiB first dataset disk ceiling,
separate from RAM and checkpoint storage. Avoid unspecified whole-collection loads.

For memory, separate storage, query addressing and decoder readout on fresh
observed facts. Only compare copy/pointer when a diagnostic probe can recover
stored values. Keep oracle source-derived associations labelled as diagnostics;
never write query answers into memory. Counterfactual siblings, all questions
and teacher expansions share a family split. Include unseen values, independent
distractors, unknowns and broader retention.

If a probe identifies a repair, change one mechanism and predeclare new code/data/
parent/source pins, scope, objective/masks, optimizer, replay, controls, independent
family counts, numeric acceptance/regression rules and update/token/time/resource
budgets. Preserve D1 versus co-adaptation labels. Compare exact/full context,
disabled, shuffled and last-observation paths. Do not retune on the exposed v5 set.

Persist across multiple observations/questions, restart without re-encoding old
observations, append a dated correction and test changed plus unaffected facts.
Report accept/reject/defer. This diagnosis can continue independently of isolated
V0; its failure must not be reclassified as a memory-quality pass.

## 4. MiniCPM5 vision fine-tuning: V0–V2

Use document 18 as the detailed contract. Keep the sealed text backbone and add
one pinned pretrained encoder plus a newly trained projector to MiniCPM5's
width 2048. SigLIP 2 is a candidate, not a preselected or locally qualified
checkpoint. Start with one image and bounded spatial patch tokens. Do not begin
by copying MiniCPM-V's projector, changing every memory layer or adding video.

Implement the visual observation and fused-prefix contracts in the existing
adapter/session/frame/training/generation paths. Bind image/crop/layout and
producer hashes. Preserve native attention, positions and vocabulary; mask
visual/prompt positions from response loss. Sense and action use the same pinned
visual input and exclude answer/future information. Cache only frozen producer
features and invalidate them on producer changes.

After R1–R3, V0 qualifies shapes, masks, input gradients, native text-only bypass parity,
request isolation, generation and fresh-process export/reload. Frozen language
parameters must remain in the autograd path to train the projector; do not put
that decoder forward under `no_grad()`. Qualify actual MPS execution before a
native training claim. Preprocessing/tiny host tests may use CPU.

V1 trains only the projector on eligible image-text/grounding pairs, with frozen
encoder and language model. V2 adds an explicitly declared adapter/LoRA scope
only after usable alignment, then visual instruction and verified reasoning
training with text replay. Name partial scopes accurately; LoRA is not full D2.
Initial parity and post-training retention are different checks.

Use the staged, split-safe curriculum in document 08. Before each run, preregister
numeric thresholds and a measured resource budget.
Evaluate correct/wrong/no/shuffled image, caption-only and OCR-text-only controls,
held-out scenes/documents/layouts, spatial/OCR tasks, missing evidence and broad
text retention. Split before crops, questions and teacher expansion. Count unique
assets/families. Do not mistake declining caption loss for grounded perception
or claim broad visual capability from a tiny alignment pilot.

## 5. Visual memory: V3

First qualify exact visual-source retrieval as a separate baseline. Preserve
image/region hashes, dates, source scope and annotation verification status.
Generated captions/OCR are attributed predictions unless checked. A byte hash
is not semantic verification.

Admit learned visual memory after V2 and the chosen memory path are qualified.
Freeze the accepted visual producer initially and train named visual writers/
readers. Observe the image, remove it, answer multiple delayed queries, persist,
reload without old-source re-encoding, append a correction, and test corrected
plus unaffected facts in a fresh process. Include full visual context, exact
retrieval, zero/shuffled/last-observation state and disabled-read controls.
Fetching the source image is retrieval, not memory-only success.

Reuse artifact, lineage, compiler and publisher implementations. Bind visual
payloads and invalidate descendants after source or producer changes, including
practice examples and trained candidates. Rebuild from eligible sources;
cache eviction is not neural unlearning and atomic publication is not a
power-loss recovery proof. Keep learned state distinct from native KV caches.

## 6. Bounded RSI: V4 / M5

Begin only after the relevant perception, useful memory/correction, retention and
independent-verifier gates. Fixed practice, rejection and rollback plumbing may
be prepared earlier. C13's 10/10 evidence threshold gates learned selection;
it is not a prohibition on deterministic verified practice.

Complete one actual cycle: observed failure → bounded eligible practice proposal
→ independent verification → isolated candidate update → unseen-family and
retention evaluation → accept/reject/defer → restart → later-task reuse. Reuse
the procedure and manifest lifecycle. Preserve a practical exact-retrieval
baseline as well as the candidate parent.

Use supervised verified practice first. Check intermediate reasoning before
using it as supervision; a correct answer does not verify the whole rationale.
Teacher/self-generated outputs require external evidence or independent checks.
The learner's confidence cannot certify its own improvement.

After a successful bounded cycle, compare ten chronological T0/T1/T2/T3 cycles:
no persistence, evidence only, fixed verified learning, and learned selection
plus learning. Match exposure/replay and count teachers, verifier calls,
generation, updates, rejected candidates and evaluation. Include distribution
change, no-improvement periods, bad-candidate rejection and rollback. Report
procedure gains, neural gains and later-adaptation gains separately.

RL, learned multi-rate consolidation and code/architecture proposals are later
separate experiments. Candidate code cannot edit its own audit, promotion rule,
guard or protected identity state, and cannot publish itself. One accepted
cycle is not accelerating or unbounded RSI. Preserve scientific D0/D1/D2 and
selected-base gates for selected-system longitudinal claims.

## 7. Spark FP8/BF16 and pretrained MiniCPM-V roles

Follow document 20. Use official `XHToken/Spark-X2.6-4B-FP8` revision
`0318d6f6ab070c7b5b43052db571489b2508006c` as the planned adaptation base and
`XHToken/Spark-X2.6-4B` revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`
as an independent BF16 control. Both are post-trained releases; do not confuse
“base for adaptation” with the separately named pretraining Base checkpoint.

Prepare metadata/shard inventory, remote-code review, tokenizer/template pins,
backend capabilities and an isolated environment now. Weight acquisition and
native qualification require the bounded source/resource plan. Keep MiniCPM5's
lock unchanged. First prove native inference, then actual input gradients and
named candidate updates, export/reload and measured residency. FP8 storage,
compute precision and Muon int8 momentum are distinct.

The official FP8 config uses compressed-tensors float quantization, static
channel-wise weights and dynamic per-token activations. Neither top-level BF16
dtype nor upstream MLX inference proves FP8 training on this MPS host. If native
support is unavailable, explicitly qualify a dequantized FP8-derived artifact
or the independent official BF16 control; never silently switch formats or claim
lost precision was recovered.

Build a dedicated `spark2_5` adapter: 36 layers, width 2560, three sliding plus
one full-attention block, 512-token sliding window. Verify actual head gates,
fused projections, tied storage, positions and caches. Test visual-prefix
visibility beyond the local window. Reuse encoder/processor/data contracts,
train a new connector to width 2560 and re-encode memories. Do not copy MiniCPM5
projector/read matrices or ports 13/27. Compare matched modes, information and
actual reasoning tokens/costs. Native Spark work follows reusable MiniCPM5 V0
interfaces and its own gates; an RSI success is not a prerequisite.

MiniCPM-V-4.7-1B remains an optional pretrained reference/teacher with its own
width-1024 Qwen3.5 hybrid adapter, processor, mergers, M-RoPE and native caches.
Its alignment cannot be transplanted to MiniCPM5. Larger 35B-A3B use needs a
separate resource plan; active experts do not determine total weight residency.
Teacher outputs remain proposals, and final audit data never enter teacher work.

## 8. Personality and semi-cognitive integration

Reuse identity archive/store, source modes, native reader and fictional tests.
Keep source history, approved snapshots, agent experience and generated practice
separate. Authentic fidelity needs permitted source material and human review;
missing sources do not block unrelated implementation.

Identity conditioning after visual fusion may influence priorities/presentation,
not evidence truth or verifier outcomes. Bind its subject/snapshot/purpose/cutoff
through decision, persistence and replay. Compare equal-information generic,
prompt-summary, archive-retrieval and trained-reader paths; fictional integration
is not personal fidelity.

Keep exact goals in task state and hypotheses in the workspace. Preserve sense
→ core → immutable decision → action → real outcome. New retrieval/crops create
new pinned decisions. Calibrate uncertainty/action-value heads before routing or
learning expenditure depends on them; compare deterministic, no-core,
feed-forward and compute-matched controls before adding deeper recurrence.

## 9. Resources, ownership and iteration discipline

Reuse the sealed local MiniCPM5 source revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52` and locked environment. Preserve selected
weights, source seals, raw failures and all historical protocols. Do not modify,
import, delete or stage `src/replica_lab/runtime_pilot.py`. No unrelated cleanup,
remote push, external deployment or broad corpus campaign is implied.

Native Mac work uses one exclusive MPS worker with fallback disabled and the
50 GiB / 53,687,091,200-byte sampled driver stop. Retain sustained 512 MiB
swap-growth and 15% minimum host-free checks over three samples, including the
initial free-memory admission. Do not kill another session's worker, raise the
guard or change OS/allocator limits to rescue an arm. Historical explicit
40 GiB protocols remain historical; every new run binds its actual policy.

Native fused AdamW is the production optimizer for new training after integration
checks; `adamw` remains custom eager and Muon int8 momentum remains historical.
Record actual trainable membership, gradients, changed tensors, tokens, time,
memory and optimizer state. Full-size native resume remains unqualified. The
prior 11.00 GiB memory run and a BF16 checkpoint label are not fit evidence.
Offload, dtype, quantization and optimizer changes each need explicit qualification.

For each iteration: select the earliest unresolved dependency → state one
hypothesis and artifact → inspect/reuse the existing path → make one coherent
change → run meaningful checks → execute one admitted bounded experiment when
needed → independently decide → record and continue. Reuse unchanged passing
checks. Follow the user-authorized [staged-commit policy](../maintenance/20261025_Repository_Hygiene.md).
Commit each completed package promptly using explicit reviewed paths, and distinguish
committed pre-run pins from an uncommitted hash seal.

Continue routine reversible work without repeatedly asking for confirmation.
If an actual missing source/backend/permission blocks dependent work, state the
specific blocker and continue independent work. Do not fabricate approvals or
results. Delegate only when active instructions authorize it. Do not create
user-owned chats or recurring automations from this prompt.

Stop a run on guard violations, nonfinite values, leakage, provenance mismatch
or publication failure; preserve evidence and diagnose. Respect user pauses and
explicit budgets. Do not imply continued background work after the run ends.

## 10. Required package report and handoff

Report changed files and purpose; exact source/parent/code/data/protocol pins;
clean-export and focused/integration checks; implemented/tested/trained/evaluated/
unrun scope; actual updates, scheduled/gate-on examples and changed tensors;
decoded split/control/family results; correction/restart and broader retention;
actual resource/teacher/verifier cost; failed gates; candidate disposition; and
the next smallest discriminating action.

Update the top current snapshot and relevant plans without overwriting dated
experiment evidence. A rejected package is completed experimental work, not an
accepted capability. Advance independent vision only under the isolation rule
above; do not bypass memory or visual gates for combined memory/RSI promotion.
Include acquired dataset bytes/verified rows separately from catalog counts,
Spark source/format/backend status, and unrun gates. Claim completion only for
objectives actually demonstrated by their stated gates.
