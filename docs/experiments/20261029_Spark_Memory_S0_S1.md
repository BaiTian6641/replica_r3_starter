# Spark memory S0–S1: foundation and intrinsic recall

**Current successor:** [official BF16 diagnostic v5](20261029_Spark_Memory_BF16_Diagnostic_V5.md) implements the
bounded cold-read/release repairs, complete-EOS probes and dedicated 2/9/36 runner.
Its clean CPU regression passes 393; native v5 is prepared and unrun, with exact
approval absent. All v3 approvals are consumed. The results and preparation below
remain the preserved earlier scope; statements about missing successor integration
or runners describe that snapshot, not the current v5 implementation.

**29 October 2026 · current Spark text-memory protocol and evidence guide.**
This succeeds the historical MiniCPM-first validation ladder. The implemented
path is `f471f54a34e99f8143164e00b3247827c6ba1f1c`; current status lives in the
[dated evidence ledger](../../reports/intrinsic_memory_v3/evidence_ledger_20261029.json).
The [runbook](../../reports/intrinsic_memory_v3/README.md) contains measured CPU
costs and the executable native packet. S0–S6 here are **Spark memory experiments**,
separate from the older Spark loader gates and V0–V4 multimodal stages.

## Responsibility and evidence

| Owner | Responsibility | Evidence needed |
|---|---|---|
| phi | Intrinsic encoders, queries, readers, writers, gates and addressing | Intended gradients and beneficial changes to generated answers |
| E/S | Attributable experience, exact details, verified procedures and corrections | Scope/time validity, retrieval, correction and restart |
| F/C | Provisional associations, intermediate bindings, hypotheses and subgoals | Later computation causally benefits from earlier writes |
| P | Persistent transformations and skills | New-instance transfer after source and temporary-state removal, including restart |
| K/C | Capability calibration and choices to recall, check, revise, practice or stop | Better calibrated decisions at matched total cost |
| I | Approved identity and personality | Stable versioned conditioning, attributable approved changes; capability assertions follow separate K evidence |
| X | External typed spine for specialist models, sensors and actors | Representation alignment, causal availability and reliable interfaces |

Storage is a separate axis: hot/warm/cold placement describes movement and
residency, not a learning clock. Paging is not consolidation. A slowly changing
identity core may stay hot while rapidly accumulated experience is archived.
Keep E/S/P/phi/C/I/K authority separate; F is a prospective fast-state mechanism,
not a rename or merger of those authorities. External spine separation is repaired,
not an outstanding prerequisite. Native multimodal work remains deferred.

## S0: qualify this execution and a competent frozen baseline

The hypothesis is that the **existing FP8-derived BF16 artifact**, with unchanged
original Spark computation, can answer the declared complete-context calibration
and support finite, reproducible extension training. Its source lineage remains
XHToken/Spark-X2.6-4B-FP8 at revision
`0318d6f6ab070c7b5b43052db571489b2508006c`. Official FP8, this derivative and
official BF16 are distinct. The official FP8 MPS blockage remains unresolved.

Freeze the [derivative execution protocol](../../configs/spark_persistent_learner_derivative_bf16_placement_protocol_v2.json),
[execution manifest](../../configs/spark_derivative_execution_manifest_v1.json),
[competence protocol](../../configs/spark_competence_protocol_v1.json) and
[intrinsic readout protocol](../../configs/intrinsic_memory_readout_protocol_v3.json).
The [review packet](../../reports/intrinsic_memory_v3/native_experiment_s0_s1_placement_v3.json) binds
their exact hashes, source artifact, tokenizer lineage, implementation files,
backend, prepared data, output directory and external authorization fields.
Configuration/model code and local executable dependencies must exist and match
before tokenizer/model loading. Full real-source validation is an integration
gate; synthetic unit trees never qualify the provisioned model.

The eight separate calibration worlds are not fitting or development data.
Original weights are frozen, extensions disabled, thinking off, greedy decoding
with EOS, at most 32 new tokens and 2,048 context tokens. The complete evidence
must fit without truncation. The template and scorer are hash-bound. Require all
eight completed, 8/8 correct and zero false recall; missing, incomplete, stale or
failed evidence closes dependent fitting and development. Sealed-final labels
remain unopened. Retain raw trajectories and engineering diagnostics on failure.

Source validation, numerical qualification and baseline competence are separate
outcomes. The old derivative qualification remains historical evidence; S0 must
verify the current execution's frozen-base integrity, finite gradients, actual
update and fresh-process continuation. A CPU random fixture failing 0/8 is a
successful gate test, not evidence that pretrained Spark fails competence.

## S1: useful intrinsic recall and restart

The hypothesis is that the small contextual producer/reader learns a useful
readout from admitted text with the original model frozen. Four ordered width-256
slots represent each bounded canonical span; capacity is at most two selected
records/eight slots, with intrinsic readers at decoder sites 3/7. The realized
fit/development worlds each use one record, one span and four slots; no padding
slots are added. The external spine is unnecessary.
Sites 23/31, other widths, extra ports and larger banks are separate experiments.
These dimensions and placements are **project hypotheses**, not measured optima.

The [all-layer access successor](20261029_Spark_Memory_Layer_Access.md) now implements
a separate v4 CPU reader/configuration/checkpoint candidate for 2, 9 and 36 sites.
All-layer access is the architecture target; this two-site protocol is a fixed
control, not a ceiling. A direct density comparison need not pass through every
intermediate count. Its native qualification, budgets and authorization are separate.

Pass S0 before 32 fixed training updates on the eight fitting worlds. One
NativeFusedAdamW owns producer and reader parameters; all originals and other
extensions remain frozen. Response CE reaches the producer through tensors in
the same update. Durable serialization and inference values detach. The sealed
implementation does not explicitly detach reader queries: site 3 starts from a
frozen graph, while site 7 can receive the earlier memory residual through later
frozen blocks. The historical protocol phrase "frozen_original_query_features"
must not be read as an explicit detach at every reader. This is one encoding and
response objective per update, not future-query credit through an earlier write
or a nested-learning result. Check the expected delayed producer gradients after
zero-initialized output projections begin learning.

Freeze fitting before opening the eight independent development worlds' labels.
Generate answers under five matched conditions: contextual values, selected exact
text, wrong values, shuffled values and absent values. Exact text uses the same
selected evidence and is charged for its text, tokens and prefill. Require at
least 6/8 correct, zero false recall, no more than one correct answer behind exact
text, and more correct answers than each negative control. These margins and
32 updates are **prospective feasibility criteria**, not demonstrated broad
competence. Preserve each arm's raw answers and all failed trajectories.

Save only the extension and its optimizer state. A fresh process must reproduce
logits and one actual continuation update while leaving original weights unchanged.
Legacy v2 migration is explicit and preserves the source artifact and moments.
Useful readout, numerical/restart success, persistent learning and promotion are
distinct decisions. S1 has zero P updates and no automatic promotion.

## Work, uncertainty and review

The active S0–S1 ceiling is 1,571 forwards, 33 backwards/updates, 48 generation
requests, 1,536 generated tokens, 3,241,216 processed text positions and two model
loads. It includes calibration and continuation. The packet separately preserves
the conservative whole-protocol reserve and the repaired eleven historical arms;
that reserve does not authorize P learning in S1. Enforce one exclusive native
worker, a 50 GiB sampled driver limit, 512 MiB sustained swap-growth limit, 15%
minimum host free, 7,200 seconds, 256 MiB live artifacts and zero native retries.
No new weights, duplicate checkpoint or full-base export is needed.

Record generated-answer quality, false recall, tokens, end-to-end latency,
teaching/retrieval/replay work, measured RSS/driver/swap, artifact bytes and I/O.
Eight development worlds and one seed only support a small feasibility decision.
Report per-world paired outcomes; repeated questions from one world are not
independent trials. Broader claims need independent families/seeds, retention
and uncertainty across those groups. Future final tasks are generated only after
development choices freeze and cannot select updates or acceptance margins.

Review before a costly run, after its result, and whenever update rule, state
visibility, checkpoint format, precision, split or scorer changes. Each review
records one hypothesis, frozen configuration, decisive controls, raw results,
uncertainty, total cost and accept/reject/inconclusive decision. A material change
gets a new protocol/binding; historical results keep their original identity.
The first exact authorized attempt failed at model construction before competence
because the command selected the general Transformers 5.20.2 environment. The
[environment-bound successor](20261029_Spark_Memory_Native_Environment.md) verifies
the existing qualified 4.57.1 environment before model imports and keeps every
criterion and budget above. That corrected attempt was authorized and executed:
the model loaded, then the first calibration forward failed due to CPU backbone
versus MPS inputs. No answer or update completed. Both attempts and receipts are
preserved and consumed. The separately authorized placement-corrected attempt03
completed: S0 passed 8/8; S1 failed useful recall after the fixed 32 updates. Exact
fresh-process logits and an actual continuation update passed. Its receipt is now
also consumed. See the [native outcome](20261029_Spark_Memory_Native_Readout03.md)
for all arm scores, resources, objective and measurement limits. The sealed v3
learning criteria and checkpoint identity remain unchanged; no P learning or
promotion follows. This page preserves the completed protocol, not a reusable launch.

## Conditional successors, not additional active runs

| Stage | Experiment | Required advancement evidence |
|---|---|---|
| S2 formation | Completed observations/interactions first; within-reasoning state separately | Earlier writes improve later answers; causal ordering, exactly-once commits, selective erasure and restart |
| S3 persistent learning | Fixed question-slot P control versus hidden-state-conditioned P | New-instance rules/procedures after source removal and restart, correction, interference and delayed retention |
| S4 functional control | Learn recall/check/revise/practice/stop choices | Better calibration and outcomes than fixed policies or equal-cost extra generation |
| S5 nested adaptation | One level versus explicitly connected levels | Better acquisition/retention with matched capacity, replay, updates and total compute |
| S6 learner improvement | Candidate writer, replay or learning-policy changes | Independently repeated gains on fresh families after task-state reset |

S2 has two distinct schedules. Cross-turn formation reads existing state,
generates a response, processes the completed interaction, commits, then exposes
that state to the next interaction. Within-reasoning formation reads chunk-start
state, computes the chunk, accumulates writes, commits once, then exposes the
state to later chunks. Bind sequence IDs, absolute positions, revisions, retry
IDs and checkpoint-recomputation semantics. The current generation loop repeats
the growing prefix; it must not repeatedly write old tokens. Reset scope and
KV invalidation are part of each declared algorithm. Neither writer is currently
integrated. Provisional hypotheses may enter F/C; verified E/S needs observed
outcomes or executable checks. Autonomous verified practice need not await a
human label for every update.

For S3/S5, declare `adapted = update(initial, teaching)` followed by loss on
**distinct future queries** plus weighted previous-knowledge retention. Name
every retained/truncated derivative. The historical first-order P estimator
omits derivatives through Adam updates, moments and inner learning rates; it
cannot learn those rates through that omitted path. Ordinary optimization stays
NativeFusedAdamW. A functional delta/SGD inner arm is a separate algorithm with
its own derivative contract and a direct-CE reference.

Persistent correction has three separate tests: obsolete evidence excluded from
retrieval, obsolete targets excluded from replay, and generated behavior corrected
after the old information influenced P. A tombstone proves neither unlearning nor
the third test. Compare learned/reset P with the same frozen qualified phi and
evidence access; source-free tests remove E/S/F/C and answer-bearing identity or
auxiliary state. Keep one P1 expert until a measured capacity bottleneck.

S4 reports wrong-to-right **and** right-to-wrong revisions, helpful recall/check
selection, missing-evidence abstention, completion and accuracy at matched cost,
and calibration on unfamiliar families. Approved I stays separate from reward
and measured K. S6 resets task-specific E/F/P/C (and task-derived S and auxiliary
channels) while retaining only the nominated policy change and fixed approved
identity. Repeated acquisition/retention/correction/rollback cycles and independent
candidate evaluation precede promotion; preserve the previous working version.

Use the [Spark curriculum](../08_Dataset_Acquisition_and_Curriculum.md#10-spark-memory-curriculum--29-october-2026)
and [research cross-check](../research/spark_memory_20261029/Research_Crosscheck.md)
for these later protocols. MiniCPM remains historical evidence and an optional
control, with no multimodal prerequisite imposed on Spark text memory.
