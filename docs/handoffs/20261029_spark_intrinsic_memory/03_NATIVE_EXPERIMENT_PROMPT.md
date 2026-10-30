# Prompt: native protocol preparation and sole execution operator

Work in `/Users/flare/replica_r3_starter`. Read
`docs/handoffs/20261029_spark_intrinsic_memory/STATE_AND_BOUNDARIES.md`, active
root instructions, architecture chapters 16/28, the v11 experiment/runbook and
the final v12 implementation/review reports. Inspect newer commits and outputs.
Never read, import, edit or stage `src/replica_lab/runtime_pilot.py`.

You own a new separately versioned native runner, protocol, execution closure,
packet, resource calculation and evidence output. You are the sole native
operator. Do not launch while another worker or native model load is active.
The v11 process ended; its output and receipt are consumed and have zero retries.
Do not run its historical `--launch` command again.

At the handoff snapshot, v12 is an untested five-file prototype with no launchable
packet. Prepare what can be prepared without weights now; launch only after
implemented CPU behavior, independent review and all preflight bindings pass.

## Preserve source and execution identity

Use the existing **official BF16** `XHToken/Spark-X2.6-4B` source, revision
`66e3ae7bee54e0a523d692cb8ef841f35409807b`, in
`checkpoints/sparkx2_5_bf16_source`, and qualified `.venv-spark-bf16`.
Do not download, duplicate, modify or export the original checkpoint. Preserve
the source lineage and separate official FP8, FP8-derived BF16 and official BF16
identities. Do not patch sealed Spark code or silently change dependencies.

Run the versioned environment-only check before any weight load. Verify package
versions, module origins, critical executable hashes, model/configuration code,
local executable dependencies and tokenizer. Verify full source identity and
the candidate's exact implementation, protocol, prepared data, scorer/template,
thinking mode, input/generation budgets, checkpoint format and output path.
Refuse changed/missing/substituted executable code before model loading.

Bind the source-only feature pass too: canonical text alone; no special/template
tokens if that is the final contract; explicit positions; extensions disabled;
frozen model in evaluation mode; final hidden-state location; dtype/casts;
feature-cache bounds; provenance and actual token sequence. Original no-grad
features intentionally detach; trainable compression/readout stays live.

## Prepare a genuinely new experiment

Keep the first mechanism study at sites 3/7, with block-input queries and
post-block residuals. Hold the reader, evidence selection, four ordered slots per
span, two-span/eight-slot budget, teacher objective, optimizer, fixed exposure,
template, scorer and generation rules matched to the declared v11 comparison
where possible. List every difference. If 512 fitting updates plus one restart
update remain the planned schedule, declare it explicitly; do not silently
inherit either that budget or permission to exceed it.

Preserve these gates in the new sealed protocol:

1. Frozen complete-context calibration over all eight declared worlds: require
   8/8, zero false recall, no evidence truncation and complete termination.
   Failure or incomplete evidence closes dependent fitting/evaluation.
2. Qualify the complete frozen fitting teacher before admitting detached targets.
   For the matched 32-world comparison require 32/32 with the declared completion
   criteria. Bound target-cache size, provenance, lifetime and eviction.
3. Execute exactly the predeclared fitting schedule. Fix probe points and groups
   before native answers. The final prefix alone may open development. Retain
   the v11 requirement that every fixed fitting group independently passes its
   complete memory-dependence/answer/termination/parity criterion. Do not choose
   a better intermediate checkpoint or aggregate away a failed group.
4. Freeze fitting and evict teacher targets before opening fresh development.
   For the matched two groups of eight, each must independently achieve
   contextual >=6/8, zero false recall, at most one answer behind exact text and
   strictly better than wrong, shuffled, zero and absent controls. All arms use
   the same eligible evidence opportunity. Keep raw answers and termination.
5. Evaluate correction and revoked-read behavior only within its declared gate.
   Separately assess obsolete retrieval, replay eligibility and generated
   answers. Do not infer behavioral correction from cache invalidation alone.
6. Save extension plus optimizer only. A fresh process restores logits and
   performs one actual continuation update. Bind source-feature recomputation
   costs after restart. Verify full original parameter hashes in both processes.
   If uninterrupted-update equivalence is not measured natively, say so.

Use the existing v11 exposure receipt to decide whether its development split
is still eligible; do not inspect labels to decide. Record any lawful successor
reuse explicitly. If exposure occurred meanwhile, prepare an independent split
before observation. No self-generated explanation is a verified oracle label.

Keep persistent-P updates zero, P1 unchanged and promotion disabled. Do not run
9/36 arms to rescue failed two-site usefulness. A native failure does not permit
changing the same sealed packet and relaunching it.

## Account for all work before dispatch

Bind maxima and per-stage reservations for: original loads/hashing, frozen
source-feature forwards, tokenizer and input tokens, trainable producer calls,
teacher target creation, backward/updates, probes, generated tokens including
stops, selected exact-text fallback, retrievals, cold waits, durable publication,
replay if any, cache misses/evictions/recomputation, extension/optimizer saves,
fresh-process continuation, logs/audit writes and artifact bytes.

A provider counter is not a reservation. Budget refusal must happen before
source-feature model work. Failed dispatched work remains charged. Durable
reservation, attempt consumption and deadline state must survive restart.
Logical serialized bytes must be distinguished from measured physical I/O.
Separate synchronized native time from asynchronous dispatch time.

Keep one worker, 50 GiB sampled driver, 512 MiB sustained swap growth and minimum
15% host free. Keep a 7,200-second outer ceiling or a smaller explicitly bound
one for this continuation; calculate stage allocations within it. If the design
cannot fit, reduce/predeclare scope or report that fact before launch. Do not
raise an envelope after answers arrive. V11's final probe took 887.769/900 seconds
and its audit writer serialized 37.1 GB, so account for audit overhead honestly.
Any ledger optimization needs its own version and fail-closed recovery tests;
do not patch sealed ledger code or describe lower audit cost as better learning.

Measure physical driver/RSS/cache/host-free/swap separately, remembering overlap.
Parameter arithmetic is not evidence of fit. Record attempts and failures, not
only successful operations. No bulk downloads or base checkpoint copies.

## Authorization and launch discipline

The direct user instruction is: “Continued as planned, no need more approval
for testing, follow the goal, assume I authorize the tests.” Current active
instructions preserve that standing authorization. Once the reviewable packet
is complete, bind a truthful new receipt to that instruction and the exact
packet. Do not request repetitive approval, fabricate signatures, pretend an old
receipt covers changed bytes, or reuse a consumed attempt. Respect later human
restrictions and all scientific gates.

Publish the concrete review command, environment-only command and exact guarded
launch command before running. At handoff no such v12 command exists; create
and verify it, do not present a placeholder as executable. Seal the implementation
and packet; confirm sole-worker ownership; record an initial resource sample;
then launch once. Preserve raw stdout/stderr, answers, resource samples, stage
receipts and process-group termination. Do not leave an orphan worker.

Deliver the exact packet and hashes, source/environment qualification, complete
stage outcomes, each control's score/false recall, split exposure, unchanged-base
proof, finite gradients, actual update/restart evidence, resources, costs and
remaining gaps. Useful recall, native density, persistent learning and RSI are
separate conclusions. A gate failure closes its dependents while retaining the
engineering evidence and next mechanism question.
