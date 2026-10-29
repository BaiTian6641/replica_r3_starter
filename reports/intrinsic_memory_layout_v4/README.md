# Configurable intrinsic readers v4

**Current successor:** [official BF16 native diagnostics](../../docs/experiments/20261029_Spark_Memory_BF16_Native_Diagnostics.md)
record v5's failed competence, v6's failed fitting, v7's learned fitting but failed
fresh-world recall, and the v8/v9 exposure/step controls. The dedicated 2/9/36
runtime and reliability repairs are implemented. The user authorizes continuing
bounded tests; consumed attempts are not retried and density still requires useful
matching two-site evidence. The reports below retain their earlier historical scope;
they are not current launch or approval instructions.

Current successor: the [combined v4 training integration](../../docs/experiments/20261029_Spark_Memory_Integrated_Access.md)
connects this reader to the contextual producer and tiered store and adds a complete
extension/optimizer checkpoint. Its new checks are in the
[integration summary](../intrinsic_memory_integration_20261029/verification_summary.json).
The receipts below remain the original reader-only snapshot. The first two v3
native attempts were execution failures. The [third native attempt](../../docs/experiments/20261029_Spark_Memory_Native_Readout03.md)
passed S0 8/8, completed 32+1 updates and exact fresh-process logits, but failed
useful recall. All three approvals are consumed. Native density remains unrun
and needs its own admitted runner/receipts.

This is a separate CPU candidate for 2, 9 or 36 memory-access sites. The sealed
two-site v3 contextual readout protocol and native packet remain unchanged.
See the [architecture and comparison](../../docs/experiments/20261029_Spark_Memory_Layer_Access.md)
and [versioned CPU protocol](../../configs/intrinsic_memory_layer_access_protocol_v4.json).

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts .venv/bin/python -m pytest tests/test_intrinsic_memory_layout_v4.py -q
```

The candidate's 26 tests pass, including 36-block parity, gradients at every active
site, value-gradient continuity, shared snapshot storage, actual branch skipping,
correction revalidation, identity/payload refusal and exact fresh-process logits
and optimizer continuation. The first attempt's optimizer API error is retained
in `tests_attempt01.log`; the corrected 25-test intermediate run is retained as
`tests_attempt02.log`. `tests.log` and `tests.xml` record the 26-test local result.
The final clean-checkout combined regression and resources are recorded separately
in `verification_summary.json`; these do not overwrite earlier attempts.

`analytical_costs.json` checks real-width parameter counts on the meta device,
without native allocation. All 36 readers have 52,125,732 parameters and 198.844 MiB
FP32 weights, or 795.376 MiB for weights/gradients/two Adam moments, before other
state and runtime costs. This is not a native fit, performance or capability result.

The v4 reader can consume live FP32 value tensors with an autograd graph. It does
not install a writer or invoke the production store/encoder automatically; those
remain the separately tested v3 path. State retrieval and producer validation
belong at the caller's snapshot boundary; validity is rechecked at each active read.
There is no per-layer optimizer step or state mutation. The new reader checkpoint
binds source, producer, representation, codec, layout, weights and named optimizer
ownership, and refuses implicit legacy migration. Original weights and external
specialist/sensor/actor interfaces are preserved.

The completed [placement-corrected v3 packet](../intrinsic_memory_v3/native_experiment_s0_s1_placement_v3.json)
is historical and cannot be retried under its consumed receipt. The next native
readout successor needs a separately bound runner/objective, complete deadline and
cost accounting, fresh development worlds and exact authorization. A 2/9/36 native
comparison also needs per-arm admission and physical measurements. Its prepared
eight-slot masked padding is not yet implemented by the combined wrapper. This
CPU candidate does not authorize a launch or expand P capacity.
