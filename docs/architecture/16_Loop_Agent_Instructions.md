# Loop-agent execution instructions

**Revised 29 October 2026 after the v5 CPU implementation and BF16 packet review.** This is the active worker
prompt. Earlier MiniCPM-first and native multimodal orders are historical.
Reading or editing this file does not launch a job, download data, create a chat
or authorize a new execution format.

Continue TAIS Obsidian / Replica R3 in the existing checkout. Build intrinsic Spark memory toward persistent learning and Replica
integration; preserve the separate external spine. The next diagnostic uses user-selected official BF16 XHToken/Spark-X2.6-4B,
revision 66e3ae7bee54e0a523d692cb8ef841f35409807b. The original FP8 lineage is
XHToken/Spark-X2.6-4B-FP8, revision 0318d6f6ab070c7b5b43052db571489b2508006c.
Keep official BF16, FP8-derived BF16 and native-FP8 execution as separate identities. MiniCPM is a historical/control
path; native multimodal training and physical actor/sensor work remain deferred.
Future specialist models will use the same typed spine.

Read [execution state](../../execution_state.md), the
[current review](../../reports/spark_persistent_learner_review_20261028/README.md),
[architecture audit](27_Spark_X2_6_4B_Architecture_Research_Review.md),
[persistent learner and tier plan](28_Spark_Persistent_Learner_and_Tiered_Memory.md),
[design profiles](../../configs/spark_persistent_learner_tiered_design_v1.json),
[state contracts](22_State_Mechanism_and_RSI_Contracts.md) and
[Replica modes](../05_Replica_Protocol.md). Inspect Git status and newer commits
before acting; do not repeat already completed repairs.

## Working boundaries

- Never read, import, edit or stage src/replica_lab/runtime_pilot.py.
- Preserve unrelated work, sealed protocols, old negative results, reports and
  runtime logs. The new design is not permission to rewrite v1 results.
- Keep one exclusive native worker, the 50 GiB sampled driver limit, 512 MiB
  sustained swap-growth limit and 15% minimum host-free rule.
- Use NativeFusedAdamW / adamw_fused for ordinary training. Do not return to Muon.
  A functional SGD/delta inner step is a separately declared meta-learning arm.
- Continue authorized CPU implementation, tests and planning autonomously.
  Prepare exact commands, source/backend identities, budgets and evidence
  before asking for any still-required native authorization. Never substitute
  an unsupported backend or precision silently.
- Report measured behavior separately from engineering tests and design
  arithmetic. Check primary research sources before changing a mechanism claim.

## Current implementation — official BF16 diagnostic v5
Read the [current v5 report](../experiments/20261029_Spark_Memory_BF16_Diagnostic_V5.md),
[runbook](../../reports/intrinsic_memory_successor_v5/README.md) and
[evidence summary](../../reports/intrinsic_memory_successor_v5/verification_summary.json).
Implementation `771e9586` follows review `017f7d71`. The user's selected source is
existing official `XHToken/Spark-X2.6-4B` BF16, revision
`66e3ae7bee54e0a523d692cb8ef841f35409807b`. All 18 source files were rehashed;
no download, base copy or native model execution occurred. Keep official BF16,
FP8-derived BF16 and native official FP8 identities distinct.

V5 implements cutoff-aware reads, persistent bounded cold-read tickets and immutable
producer-release rebuilds; eight-slot masking; complete-EOS fitting and generation
probes; and a dedicated 2/9/36 producer/store/reader runner. Live tensors preserve
producer gradients, and one NativeFusedAdamW owns each intended parameter. The
complete checkpoint binds source, objective and execution identity. The whole attempt
has an outer wall supervisor and pre-reserved stage work, including failed work,
source validation, probes, artifacts and fresh-process continuation. Sealed v3/v4
execution files and historical results remain unchanged.

Clean code snapshot `771e9586` passes 393 affected tests in 64.60 s. Its 2/9/36 CPU
workflow passes exact fresh-process logits and one real continuation update per arm
in 14.605376 s, peak RSS 624,705,536 bytes. A separate unchanged-Spark-code probe uses
36 tiny random CPU BF16 layers and passes parity, two updates and producer gradients;
it loads no original checkpoint weights. These are engineering results, not useful
memory. The historical attempt03 passed S0 8/8 and failed S1 (contextual 0/8,
selected exact text 8/8). Do not tune on its development worlds.

The next executable experiment is the fully bound
[two-site v5 packet](../../reports/intrinsic_memory_successor_v5/native_packet_two_v5.json).
All three earlier approvals are consumed; a new exact approval is absent. The
packet requires 8/8 complete-context competence, then 32 complete-EOS readout updates,
a fitting-only memory-dependence gate before fresh development, and one restart update.
It permits zero P updates, 957 forwards, 856 generated tokens, two sequential model
loads, 7,200 seconds, 160 MiB artifacts and zero retries. Keep the exclusive worker,
50 GiB driver / 512 MiB sustained swap / 15% host-free guards. Nine/all-layer packets
have independent budgets/approvals and require useful current two-site evidence.

The intrinsic memory layer participates in Spark computation; the external spine
continues to connect specialists, sensors and actors. Keep block-input queries and
post-block residuals explicit. Useful recall must precede completed-interaction
formation, within-reasoning commits and source-free P learning. Keep P1; nested
adaptation, RSI and promotion remain closed. Preserve E/S/P/phi/C/I/K ownership,
approved identity outside reward and the Replica/semi-cognitive-core contracts.
Hot/warm/cold placement is independent of learning clocks; bounded active identity
and task state need measured residency, with explicit archival recall. Native
multimodal work is deferred. Never read, import, edit or stage
src/replica_lab/runtime_pilot.py. Preserve unrelated dirty work.

## Preserved attempt03 receipt and successor gates — 29 October 2026

Read [Spark memory S0–S1](../experiments/20261029_Spark_Memory_S0_S1.md), the
[dated ledger](../../reports/intrinsic_memory_v3/evidence_ledger_20261029.json) and
[intrinsic memory v3](../../reports/intrinsic_memory_v3/README.md). The spine is
external communication; memory owns formation, addressing, recall, retention,
correction and learning inside Spark. f471f54 extracts intrinsic ownership/config,
removes the dummy packet, and versions serialization with explicit migration.
Empty-spine execution, gradients and exact migrated optimizer continuation pass.
The prior provenance/recovery/calibration repairs remain active. The random CPU
fixture calibration is 0/8, so dependent learning and labels stay closed. This
is not a pretrained Spark competence result. The [attempt03 report](../experiments/20261029_Spark_Memory_Native_Readout03.md)
and [result receipt](../../reports/intrinsic_memory_v3/native_readout03/result.json)
record S0 at 8/8 correct, zero false recall and no evidence truncation using the
existing FP8-derived BF16 artifact. Exactly 32 planned readout updates completed.
S1 fails: contextual/exact/wrong/shuffled/absent score 0/8, 8/8, 0/8, 0/8 and 1/8;
false-recall counts are respectively 1, 0, 1, 1 and 0. Fresh-process reload produced
exact logits and one actual continuation update; declared frozen-base checks pass.
There were 33 total updates and zero P updates. Keep useful recall, formation and
persistent P gates closed despite baseline competence and numerical continuation.
Guard wall time was 2,246.088 s, peak sampled driver memory 37.244 GiB, swap
growth zero and minimum host free 31%; the report states measurement limitations.
The first two attempts failed before any calibration answer because of the wrong
Transformers environment and then CPU/MPS backbone placement. They are execution
failures, not failed capability tests. All three zero-retry approvals are consumed.
Read the [current integration report](../experiments/20261029_Spark_Memory_Integrated_Access.md).
The earlier 268-test/13.329595-second combined-v4 receipt is recorded evidence,
not rerun during attempt03. Newly run future-only wall supervision and explicit
EOS-completion checks pass 67 clean CPU tests in 5.36 s; neither repair ran in attempt03. A later native repair needs
its own bound runner/checkpoint, protocol, budget and exact authorization.

All-layer intrinsic access is the explicit target. The separate
[v4 CPU candidate](../experiments/20261029_Spark_Memory_Layer_Access.md) supports
2/9/36 independently configurable sites and a new checkpoint format. Its separate combined wrapper now connects the real
contextual producer and store with a single optimizer and complete reload. It does
not change the sealed v3 control or select itself in the native runner. The v5 successor above now implements
the direct comparison with shared values, fixed slots/writer/exposure and separate
per-arm budgets. Numerical/resource checks and competence precede native work.
Do not multiply writes, searches or state copies with reader count. Shared/grouped
fast banks, matched-parameter/cost readers and pre-MLP fusion are separate tests.

1. Review the failed fixed S1 comparison at 3/7 and its exact/wrong/shuffled/absent
   controls. Preserve all answers and acceptance criteria. Use the prepared v5 repair and its closed development gate for fresh selection;
   do not tune repeatedly on these eight worlds. The historical answer-word
   objective lacks EOS supervision, but this alone is not a proven cause. Keep
   frozen-baseline competence, numerical checks, useful recall and restart as
   separate outcomes. No further native attempt is authorized. Treat 23/31 separately.
2. S2 first tests completed-interaction storage, exposed only next interaction.
   Separately test within-reasoning C/F: read chunk-start state, accumulate,
   commit once, expose only later chunks. Bind sequence positions, revisions,
   retry IDs and checkpoint recomputation; test interrupted/restarted transitions.
   Growing-prefix replay must not duplicate writes. Provisional hypotheses are
   allowed; they do not become authoritative E/S facts without verification.
3. S3 compares fixed question-slot P with hidden-state-conditioned plastic experts
   or associative reads. Keep the eleven controls and same-reader learned/reset
   comparisons. Remove E/S/C and answer-bearing I/auxiliary state for source-free
   generated tests. Use distinct teaching/future queries, independent worlds and
   rule families, chronological corrections, revocations and delayed retention.
   Test retrieval exclusion, replay exclusion and behavioral correction after
   old information influenced P separately; a tombstone is not unlearning.
4. Measure new reasoning-task accuracy, old knowledge, false recall and total
   teaching/retrieval/update/replay/generation cost and physical memory. Keep P1
   until a capacity bottleneck. Writer/nested/RSI work requires useful readout,
   repeated retention/correction/rollback and independent candidate evaluation.

Follow the [formation and Replica successor specification](../experiments/20261029_Spark_Memory_Formation_and_Replica_Successors.md).
Preserve approved identity outside reward, prioritize its bounded active core and
active task state, and use explicit recall for archival tiers. E/S/P/phi/C/I/K and
external spine ownership remain separate. The older packages below preserve design
requirements and history; they do not override this current sequence.

S4 functional control compares learned choices with fixed policies and equal-cost
extra generation, including wrong-to-right and right-to-wrong revisions and new-family
calibration. S5 must connect declared objectives/derivatives across levels and
match capacity/replay/updates/compute. S6 resets task E/F/P/C and task-derived
auxiliary channels while retaining only the candidate learning-policy change.
Approved identity remains fixed and outside task reward. Review before/after each
costly stage and whenever update rule, state visibility, checkpoint format,
precision, split or scorer changes; record hypothesis, frozen config, controls,
results, uncertainty, cost and decision. Do not infer new native authorization.

## Package 1 — correct the diagnostic and make verification portable

The preserved nine-arm 0/12 result uses a random, pointwise tiny decoder without
attention. It is an engineering fixture, not a Spark reasoning evaluation.
Keep its raw results and mechanical restart evidence; correct labels and add a
real contextual fixture for relevant tests. Do not optimize against its 0/12
score or claim Spark has a full-context defect.

Portable protocol tests pass using tracked source/Git metadata mocks; production
source authentication is unchanged. Keep full weight validation in explicit
integration/native admission. Unit fixtures must never qualify a real source.
Prove review/missing-authorization/wrong-format/stale-code/missing-backend checks
from a clean checkout with no native files.

Bind all optimizer groups to actual named parameters and their single owner:
direct response CE updates selected P, while a dedicated owner updates only final
port-output projections. Non-port spine and local-MSE steps are zero in this arm.
Audit actual parameter deltas, optimizer moments, clocks and checkpoint recovery
separately. Do not describe discarded response gradients as response-trained P.

Preserve a reproducible baseline and failed JUnit evidence. Make the revised
documentation migration explicit. Refresh bundle manifests with their supported
tools; uncommitted development checks cannot be reported as strict clean
publication. Do not stage unrelated files to satisfy a verifier.

## Package 2 — tiered storage and larger configurable profiles

Implement a new version using the design JSON; leave v1's hardcoded control
shape and sealed protocol byte-identical. The ladder is P1, P4, P16, P32, P64,
with P128 a stretch. P64 has 50,331,648 full gated-expert parameters. Initially
activate one expert; at most two is a separate cost/quality comparison. Stored
capacity and active work are distinct. Do not materialize every shard on GPU.

Implement exact evidence E, verified procedures S, plastic parameters P,
offline machinery phi, workspace C, approved identity I and measured capability
K as separate state classes. Separate retrieval keys from ordered readable
values. Candidate evidence reads are eight records times four width-256 slots;
identity, workspace and P queries have separate explicit budgets. Canonical
text and exact-span fallback remain available and charged.

Use proposed subsystem ceilings of 2 GiB hot, 8 GiB warm and 64 GiB disk, with
8 MiB identity reservation inside hot and 256 MiB replay inside warm. Allocate
lazily and preserve the global guards and disk headroom. On Apple silicon these
are logical tiers sharing physical RAM, not independent VRAM plus RAM pools.
Count independent copies, driver allocation and page-cache pressure correctly.

Clusters need stable IDs, immutable generation manifests, scope-before-semantics,
versioned source lineage, validity, verification, importance, byte sizes and
tier locations. Protect rare critical information as well as frequent reads.
Priority never grants truth or cross-Replica access. Logical pinning retains the
approved identity core and active state during admitted work; it is not an OS
residency guarantee.

Implement bounded cold recall with wait/resume, post-I/O revalidation, atomic
publish, interrupted-write recovery and disk accounting that includes overlap,
journals and retained generations. Persist P moments/clocks before eviction.
Invalidate all dependent tiers on correction. Quarantine/reset/replay P taught
from revoked sources; do not claim a metadata tombstone unlearns weights.

Test overflow, hot/warm eviction, dirty expert migration, crash/rebuild,
correction during cold I/O, stale encoder versions, cross-scope isolation,
identity availability and bounded artifact growth. Keep large future checkpoint
binaries outside Git; detect extensionless archives by size/type. Preserve
existing historical artifacts.

The connected readout contract is now implemented: call `read_scoped_memory`,
then restore returned canonical records through the runner's single strict
restoration path before neural selection/query construction. Ordinary arms use
two-record semantic selection; only the untouched baseline uses `all_eligible`
with 16 records/64 slots and refuses truncation. Require current-record
provenance and matching producer versions; corrections carry current authority
plus `supersedes`, with explicit reconciliation for an inactive parent. Cold
tickets bind full scope including `as_of` and revalidate time/verification after
I/O. Complete all pending tickets before one coherent final selection and
aggregate every attempt's exact metrics. Empty, stale or incomplete reads are
refused except for explicitly declared empty controls. Keep `selected_record_ids`
consistent with the host readout and the baseline free of fitted reader/P state.
Cache hits, key/payload reads, bytes, cold waits, resumes and revalidations are
work counters, not quality claims. Logical tier names never establish residency.

## Package 3 — competent Spark baseline and useful memory readout

Resolve the actual native execution path before launching. The current
official-native-FP8/MPS training path is not qualified. An MLX inference result
does not qualify PyTorch gradients. If a BF16 control or explicit dequantized
derivative is needed, prepare a separate versioned protocol and retain its
identity. Preserve the user's FP8 checkpoint as the source baseline.

Keep Spark's original attention, GELU MLP, head geometry and tied output path.
Begin at zero-based sites 3/7; measure 23/31 separately later. Require disabled parity and correct,
wrong, shuffled and absent memory controls. A detached frozen prefix is valid
only if no earlier trainable path needs its gradient; charge the full forward.

Pass the separately declared pre-fit calibration before opening development.
Check all full-context development rows and report their competence before
interpreting native memory outcomes. Then compare selected exact context,
E-only, source-free P-only, E+P, reset/wrong state and matched nonlearning/replay
controls. Keep sealed evaluation closed. Use generated answers and calibrated
unknown handling, not only loss, teacher-forced rank or parameter changes.

Start with P1 and a direct response-CE reference that actually updates selected
P parameters through their owner. Prove response usefulness and new-instance
transfer without harming old skills. Keep the existing detached-target arm as
a separate reference. Do not enlarge the learner to hide a readout failure.

## Package 4 — learned local writes, longitudinal learning and later RSI

After useful readout, train offline U/G and write-target machinery through the
exact one-step D-only update. Score distinct future training queries plus old
skills, corrections and cost. Declare exact versus approximate derivatives.
Deploy the same update rule trained offline; imperative fused AdamW is not an
automatic differentiable meta-optimizer.

Use independent worlds, sources and rule families for training/development/final.
Maintain dataset licenses, versions, dedup, causal availability and complete
work counts. LongMemEval/LoCoMo are evaluation candidates, not writer-training
data by default. Use synthetic identity fixtures until approved personal
records exist. Generated explanations need independent verification before
entering E or S.

Scale P4/P16/P32/P64 only with matched active work, fixed evaluation, restart/
correction/retention checks and measured total memory/latency. Larger E/S
storage can progress independently. Evaluate delayed reuse and rare important
recall across restarts under pressure; report acquisition and forgetting curves.

Maintain a bounded self-review controller with answer/retrieve/verify/procedure/
practice/stop actions. Keep human identity evidence distinct from the agent's
experiences and capabilities. One learning clock comes first. Add another
timescale or RSI only after a candidate learning policy beats a fixed policy
on future tasks at matched total cost with retention and rollback. Never equate
extra loops, self-generated text or ordinary fine-tuning with RSI.

## Loop and handoff

For each bounded package: inspect current evidence, state the hypothesis and
failure condition, implement the smallest complete change, run the relevant
checks, inspect failures, and update the handoff with exact paths and counts.
Stop an experiment when its declared gate fails; continue independent storage,
verification and design work where useful. Do not reopen sealed labels, promote
mechanical success as learning, or start an unbound native run.

Report changed files, commands, fresh results, source/execution identity,
resource arithmetic versus measurements, unresolved risks and the next
discriminating experiment. Preserve failed evidence and describe any pending
publication or native authorization accurately.
