# Loop-agent execution instructions

**Revised 28 October 2026 after review of 07f64ab2.** This is the active worker
prompt. Earlier MiniCPM-first and native multimodal orders are historical.
Reading or editing this file does not launch a job, download data, create a chat
or authorize a new execution format.

Continue TAIS Obsidian / Replica R3 in the existing checkout. Build Spark text
memory and the permanent internal spine toward persistent learning and Replica
integration. The active source is XHToken/Spark-X2.6-4B-FP8, revision
0318d6f6ab070c7b5b43052db571489b2508006c. Keep official BF16, FP8-derived BF16
and native-FP8 execution as separate identities. MiniCPM is a historical/control
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

## Current receipt — 29 October 2026
The repaired readout/control and derivative implementation is committed at
`83c7214`. `integrated09` matches protocol `91c45f` and completes 11 arms and
132 development cases with engineering acceptance, but all arms remain 0/12.
Its all-eligible baseline contains all 11 records without truncation. Two
additional fresh-process restart cycles pass exact token parity and preserve P
state under `integrated08/restart/`. Focused protocol/readout/control/recovery
tests pass 62; the full-suite result is recorded in `execution_state.md`.
Read the current [v2 runbook](../../reports/spark_persistent_learner_v2/README.md)
before repeating any storage, recovery or control work.

The native plan and packet are [`native_backend_plan.md`](../../reports/spark_persistent_learner_v2/native_backend_plan.md)
and [`native_experiment_packet.json`](../../reports/spark_persistent_learner_v2/native_experiment_packet.json).
Official-FP8 MPS execution is blocked by missing exact forward/backward/update
backend evidence and external authorization. The separate derived-BF16 protocol
and [`derivative_bf16_packet.json`](../../reports/spark_persistent_learner_v2/derivative_bf16_packet.json)
are prepared but also require external derivative authorization. Do not launch
a substitute BF16 or dequantized arm under the official-FP8 protocol. No native
model job has run.

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
Compare zero-based sites 3/7 with 23/31. Require disabled parity and correct,
wrong, shuffled and absent memory controls. A detached frozen prefix is valid
only if no earlier trainable path needs its gradient; charge the full forward.

Demonstrate full eligible-context competence on the same tasks before
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
