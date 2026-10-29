# Spark intrinsic memory: configurable access at all 36 layers

**Current successor:** [official BF16 diagnostic v5](20261029_Spark_Memory_BF16_Diagnostic_V5.md) implements the
bounded cold-read/release repairs, complete-EOS probes and dedicated 2/9/36 runner.
Its clean CPU regression passes 393; native v5 is prepared and unrun, with exact
approval absent. All v3 approvals are consumed. The results and preparation below
remain the preserved earlier scope; statements about missing successor integration
or runners describe that snapshot, not the current v5 implementation.

**29 October 2026 · v4 CPU engineering candidate; native density comparison unrun.**

All-layer intrinsic memory access is an explicit architecture target. The
two-site [S0–S1 readout experiment](20261029_Spark_Memory_S0_S1.md) remains the
sealed control. This successor to the historical
[port-density proposal](../research/sparkx_port_density_20261026/Port_Density_Research_and_Plan.md)
adds an executable configurable reader and a separate checkpoint format. It
imposes no MiniCPM or vision prerequisite on Spark text memory. The external
typed spine retains its specialist-model and sensor/actor interfaces.

## Implemented CPU candidate

[IntrinsicMemoryLayout and ConfigurableIntrinsicMemoryReader](../../src/replica_lab/architecture/intrinsic_memory_layout_v4.py)
accept ordered, unique, independently enabled decoder sites. Installation checks
the actual decoder geometry before changing ownership, freezes original weights
and other extensions, removes the legacy memory branch, and installs one intrinsic
reader owner. It refuses replacement of an existing intrinsic owner; this is a
fresh candidate, not a silent v3 migration.

The three declared Spark layouts are `(3,7)`, the nine full-attention sites
`(3,7,11,15,19,23,27,31,35)`, and every site `0…35`. Each site has its own query,
key/value, output projections, three normalizations and scalar gate. Output
projections start at zero. One incoming width-256 snapshot (up to eight slots) is shared
by every active layer; no layer performs retrieval, durable writes or an optimizer
step. Snapshot cloning preserves the training tensor graph. A validation callback
checks validity at each read, including after correction. An explicit active-site
mask skips branches; a small learned gate alone does not save computation.

The existing adapter queries the **block input** and adds the result after the
whole original block:

```text
next_hidden = original_Spark_block(hidden) + memory_read(hidden, incoming_state)
```

This preserves pretrained attention, position handling, gated attention, GELU MLP
and tied output. It is not attention-internal fusion. Adding recall before that
block's MLP, or a hidden-state-conditioned plastic-expert branch, changes the
gradient path and requires a separate placement experiment. An early trainable
reader needs gradients through later frozen blocks: freezing weights must not
detach that path.

One NativeFusedAdamW owns every reader parameter. The reader-only v4 checkpoint
binds layout, placement, source, producer, representation and codec identities,
parameter names, weight hash and optimizer hash. Payloads, moments and named
ownership are validated before loading weights. Missing, corrupted, stale or
legacy inputs fail closed. A weights-only reload cannot retain unrelated live
optimizer moments. Serialization detaches; the live value tensor path does not.
No base checkpoint is exported. The production producer/store implementation and
two-site v3 training wrapper remain unchanged. A separately versioned
[combined v4 wrapper](20261029_Spark_Memory_Integrated_Access.md) now connects those
contracts to this reader and adds complete producer/reader/optimizer checkpoints.
It remains separate from the native v3 runner's selected implementation.

The [CPU protocol](../../configs/intrinsic_memory_layer_access_protocol_v4.json),
[tests](../../tests/test_intrinsic_memory_layout_v4.py) and
[runbook](../../reports/intrinsic_memory_layout_v4/README.md) bind this scope.
Tests cover 36-block disabled/zero-effect parity, all-site gradients after the
output projections open, shared storage, active masks, correction revalidation,
strict identity refusal, original-weight integrity and exact fresh-process logits
plus an actual continued optimizer update. These are random-fixture engineering
results, not learned Spark recall or reasoning.

## Corrected reader cost

For the actual reader, `2*D*m + 2*m*m + 2*D + 4*m + 1` parameters per site
include query/output, key/value, three affine norms and one gate. At `D=2560`,
`m=256`, that is **1,447,937**. Meta-device tensor inventories verify the formula
without allocating native weights. The previous proposal's fused-reader arithmetic
described a different mechanism; its memory-plus-workspace formula omitted one
norm pair and is not the cost of this implementation.

| Active layers | Reader parameters | FP32 weights | FP32 weights, gradients, two Adam moments |
|---:|---:|---:|---:|
| 2 | 2,895,874 | 11.047 MiB | 44.188 MiB |
| 9 | 13,031,433 | 49.711 MiB | 198.844 MiB |
| 36 | 52,125,732 | 198.844 MiB | 795.376 MiB |

All 36 readers add 1.2676% of the declared 4,112,079,360 backbone parameters.
The [cost ledger](../../reports/intrinsic_memory_layout_v4/analytical_costs.json)
excludes encoder, writer, P experts, fast state, backbone, activations, temporary
buffers and optimizer scalar steps. It does not prove a run fits 50 GiB. Current
readers repeat slot projections, validation and potential synchronization at each
site; early readers extend the backward graph. Profile total physical memory,
latency and repeated-prefix generation before admitting a native density run.

## Direct comparison and causal state formation

Compare **2, 9 and 36 directly** once each candidate passes numerical/resource
checks and the declared frozen-base competence gate. Six, eighteen or 23/31 can
be diagnostics, not compulsory milestones. Hold memory contents, selected slots,
writer, update frequency, training exposure, data splits, scorer, template and
generation budgets fixed. Then compare matched-parameter and matched-total-cost
alternatives: shared projections and shared versus depth-group fast banks. Tying
weights saves storage, not repeated computation. Measure generated recall,
reasoning transfer, false recall, correction, retention, tokens, latency and all
teaching/retrieval/replay costs on independent worlds/families.

Access placement, state placement and write timing are independent. A shared store
does not require 36 searches, bank copies or writes per observation. A future S2
writer first reads incoming state and commits a completed observation/interaction
once for the next interaction. A separate within-reasoning schedule reads chunk-start
state, accumulates proposals and commits once for later chunks. Bind absolute
positions, sequence/revision/retry IDs and checkpoint recomputation so growing-prefix
generation cannot write old tokens again. Provisional hypotheses belong in F/C;
verified E/S and persistent P require separate evidence. A late layer's new state
cannot affect an earlier layer in the same forward without another scheduled pass.

[uTTT](https://arxiv.org/html/2610.05484v1) supplies a shared-depth, chunk-boundary
precedent with trained later-read/earlier-write credit. [MemT's appendix](https://arxiv.org/html/2610.26825v2#A2)
uses selected layers: 10 of 40 and 20 of 43. [HOPE](https://arxiv.org/html/2512.24695v1)
requires distinct learning contexts, objectives and updates; reader count alone
is not nested learning. None establishes the best Spark density.

The v3 S0–S1 learning protocol and budgets remain unchanged. Its first authorized
load failed in the wrong dependency environment; the [runtime-bound successor](20261029_Spark_Memory_Native_Environment.md)
was authorized, loaded the model and failed on a CPU/MPS placement mismatch before
any answer. The separately authorized [placement-corrected attempt03](20261029_Spark_Memory_Native_Readout03.md)
then passed S0 8/8 but failed S1 useful recall. Its exact fresh-process logits and
continuation passed. All three receipts are consumed. This v4 CPU protocol authorizes no native
work. The [density v5 preparation](../../reports/intrinsic_memory_density_v5/native_comparison_packet.json)
now binds per-arm execution/data/scoring and forward/backward/token/model-load
budgets; its dedicated native runner and external receipts remain absent. Do not
multiply the two-site packet implicitly. Keep one worker, the 50 GiB / 512 MiB sustained swap / 15% host-free
limits and P1. Useful readout must precede persistent-P, learned-writer and nested
experiments. Preserve all raw failed trajectories and the previous working version.
