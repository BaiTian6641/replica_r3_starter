# Prompt: all-layer access, shared-reader and cost controls

Work in `/Users/flare/replica_r3_starter`. Read
`docs/handoffs/20261029_spark_intrinsic_memory/STATE_AND_BOUNDARIES.md`, active
instructions, architecture chapter 28 and these experiment documents:

- `20261029_Spark_Memory_Layer_Access.md`
- `20261029_Spark_Memory_Integrated_Access.md`
- `20261029_Spark_Memory_Matched_Readers_V1.md`
- `20261029_Spark_Memory_Coverage_V11.md`

They are under `docs/experiments/`. Inspect current code and newer commits.
Never read, import, edit or stage `src/replica_lab/runtime_pilot.py`. Preserve
sealed experiment code and unrelated work. Own separately versioned density
instrumentation, CPU tests and comparison protocols; coordinate shared edits.
This role may prepare CPU work now, but cannot independently launch native arms.

## Start from what already exists

The configurable independent 2/9/36 reader and combined producer/store wrapper
already have CPU evidence. Do not reimplement them from the old design plan.
The separate matched-reader candidate is:

- `src/replica_lab/intrinsic_memory_matched_reader_v1.py`
- `tests/test_intrinsic_memory_matched_reader_v1.py`
- `configs/intrinsic_memory_matched_reader_v1.json`
- `reports/intrinsic_memory_matched_reader_v1/clean/summary.json`

It passed 29 clean CPU tests at `db405c87`. Exactly two physical reader groups
serve 2/9/36 sites with mapping `(site // 4) % 2`, anchored to initialization
streams 3 and 7. Only canonical groups are registered; site aliases must not
duplicate state keys or optimizer entries. At native hidden size 2560, unique
reader parameters are 2,895,874 in each arm. This is a parameter count, not a
physical-memory or total-cost result.

The shared-reader candidate has no native integration or utility result.
Native v11 9/36 packets are gated because v11's matching two-site fitting failed.
Any new producer needs matching new comparison bindings; old packets cannot
validate a changed mechanism.

## Complete the CPU and instrumentation integration

Connect the eventual reviewed producer to the layouts through a new versioned
wrapper, preserving old checkpoint identities. Maintain one canonical selected
snapshot for all active sites, one retrieval, no per-layer writes, and no
unnecessary state copies. Prove disabled branches skip computation.

Check response gradients for every active site and the producer after zero
output projections start learning. Early readers must retain gradients through
later frozen blocks. Verify one optimizer owner per unique intended parameter,
original-backbone integrity, correction/revocation cache invalidation and full
fresh-process checkpoint/logit/optimizer continuation for every layout.

Build profiling around actual reader invocations. Do not register a separate
module hook for every site alias when those aliases reference the same physical
group: that multiplies callbacks and corrupts counts/timing. Bind invocation
site, sequence/chunk and forward/recompute identity. Count recomputation honestly
without creating retrieval/write effects. Test profiler correctness with two
shared groups invoked at 36 sites.

Measure retrieval counts, source encodes, writes, snapshot copies, allocations,
cache bytes, reader invocations, activation/backward work and wall time.
Separate single-forward tiny-fixture timing from native performance. Keep the
current block-input-query/post-block-residual placement explicit. Pre-MLP fusion
must have a separate protocol and cannot be folded into a density arm.

## Prepare three distinct comparisons

1. **Fixed per-site reader architecture, 2/9/36 access sites.** Hold evidence,
   selected slots, source producer, writer behavior, update frequency, training
   exposure, data split, template, scorer and generation limits fixed. Reader
   parameter count changes; state this confound explicitly.
2. **Matched unique reader parameters.** Use the two canonical shared groups and
   identical producer/order parameters across 2/9/36. Verify actual unique
   parameters, optimizer tensors, initialization, checkpoint keys and resident
   copies. Sharing changes reuse/optimization structure as well as parameter
   count; do not claim it isolates site count perfectly.
3. **Matched measured total cost.** Prepare a separate preregistered design using
   actual calibration/engineering cost measurements. State which exposure or
   scheduling variable may change, its fixed selection rule and the resulting
   scientific comparison. Do not silently alter exposure in the first two
   comparisons or choose a budget after development answers. Equal parameters
   do not imply equal compute, latency, memory or teaching cost.

Each arm needs its own source/execution/numerical/resource/scoring/data/output
and authorization binding. Do not multiply the v3/v11 native budget implicitly.
The first comparison is feasibility evidence. Broader causal claims require
independent world families, seeds and the later controls.

Generated-answer quality, false recall, exact-text gap, negative controls,
chronological correction, total tokens, latency, driver/RSS/cache/host memory,
teaching, retrieval, replay and audit costs all belong in the measurement schema.
Parameter estimates alone cannot declare an arm viable.

## Gate and handoff

Keep native 9/36 closed until a matching two-site mechanism passes frozen
competence, useful readout, integrity, continuation and resource gates. Arrange
sequential execution through the coordinator and sole native operator. One
exclusive worker, 50 GiB driver, 512 MiB sustained swap growth, minimum 15% host
free; no duplicate originals, P expansion or multimodal work.

Deliver code/tests, per-invocation profiling validation, actual CPU continuation
results, exact parameter/owner counts, three clearly separate comparison designs
and the remaining native prerequisites. Do not describe configurable capacity,
36 CPU hooks, shared state, paging or reader density as persistent learning or
nested adaptation. Preserve external spine responsibilities throughout.
