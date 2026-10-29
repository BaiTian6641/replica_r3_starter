# Spark spine and E/P memory integration — 28 October 2026

## Scope and authority

Starting HEAD: `2782193d`. This is a CPU implementation and regression increment
for the proposed Spark spine and intrinsic memory interfaces. The sealed model
sources, native authorization gates, capacity guards and protected
`src/replica_lab/runtime_pilot.py` restriction are unchanged. **No native model
job, MPS execution, download, full-backbone update, promotion or learned
capability claim occurred.**

## Implemented

- `architecture/spine.py` defines typed width-512 packets, per-query causal
  availability, bounded packet buffering, and independent `spine`, `memory` and
  `workspace` readers. Only final residual projections are zero-initialized;
  gates remain live. External memory slots have explicit geometry/device/dtype
  validation and an `E`/`P`/`external` provenance label.
- `architecture/spark_adapter.py` executes the readers at declared native Spark
  block sites while retaining the frozen backbone and uncached reference path.
  `Spark2_5Adapter` and the memory experiment can share one exact `SpineCore`
  instance, so the native forward and the P reader use the same owned weights.
- `architecture/spark2_5.py` adds the explicit v2 extension inventory and
  migration declaration. Existing v1 exports remain valid; spine tensors cannot
  be silently relabelled as the vision projector or accepted under v1.
- `architecture/spine_memory.py` implements bounded, provenance-bearing E
  records, exact retry deduplication, atomic correction/revocation, stale
  revision rejection, and selected-expert P reads through the native memory
  reader. E and P remain separately ablatable. A repeated teaching event reports
  zero new optimizer, forward or backward work.
- `architecture/spine_checkpoint.py` provides strict spine-only and paired E/P
  checkpoint formats. Checkpoints include extension tensors, optimizer/RNG
  state, work counters, completed evaluations, evidence lineage and plastic
  controller state. Restore validates the complete schema and rolls back on
  failure. Active evidence references lineage IDs rather than duplicating active
  tensors. P-only clearing removes active evidence, retained evidence history
  and derived source/read state while retaining P.

## Verification exercised

```bash
PYTHONPATH=src:tests .venv/bin/python -m pytest -q \
  tests/test_spine.py tests/test_spark2_5_adapter.py tests/test_spark2_5.py
# 38 passed

PYTHONPATH=src:tests .venv/bin/python -m pytest -q \
  tests/test_spark_backend_parity_20261027.py \
  tests/test_sparkx_nested_learning.py tests/test_sparkx_residency.py \
  tests/test_nested_learning.py tests/test_nested_text.py \
  tests/test_nested_text_continuation.py tests/test_memory_system.py \
  tests/test_memory_session.py
# 324 passed, 21 subtests passed
```
Additional verification:

- `tests/test_sparkx_final_product.py`: **16 passed** after declaring the
  reviewed spine inputs; `tests/test_spark_preparation.py`,
  `tests/test_sparkx_v5_1_review.py` and `tests/test_spark_weight_acquisition.py`:
  **69 passed**.
- `scripts/prepare_sparkx_research_bundle.py --profile final-product --check`:
  **81 inputs** and passed. The generated manifest owns the exact byte count and
  digest; the historical 117-input manifest was not rewritten.
- A complete `pytest -q` attempt did not finish within 600 seconds and is not a
  pass. An isolated early failure in
  `tests/test_dc_r1_native_runner_v2.py::test_cli_refuses_invalid_authorization_without_loading_model`
  returned the pre-existing `retired_runner` guard instead of `authorization`;
  that registry/configuration work is outside this increment. Other full-suite
  failures were not relabelled or hidden.

The tiny native Spark fixture exercises real decoder forward/backward paths:
initial spine parity is exact, opening only final memory projections produces a
nonzero logit change, and gradients reach the selected P expert and shared spine
without producing backbone gradients. Restart tests restore E/P and continue a
new update; retry of a completed event is deduplicated. These are deterministic
CPU mechanics, not model-quality evidence.

A separate child-process smoke restored the checkpoint without re-admitting its
source record, preserved the P parameter identity and evidence lineage, and
returned `retried=true` with zero new optimizer steps for the completed event.

## Explicit limits

- No MiniCPM5 or full Spark native generation/training, GPU memory/latency
  measurement, native checkpoint save, full-bank capacity result or held-out
  evaluation was run.
- The P-only check is an intervention and persistence check, not evidence of
  source-free learned task benefit, calibrated confidence or generalization.
  A future capability protocol must supply known queries independently of
  removed teaching evidence and score actual behavior.
- The checkpoint tests use temporary CPU artifacts. They do not duplicate or
  rewrite a sealed base checkpoint, and no native optimizer state was created.
- This increment does not authorize a new experiment or replace the existing
  E1/E2/E3 runtime protocol. Native execution still requires its own declared
  protocol, authorization and guarded worker.

Next eligible work is a separately predeclared native/interface protocol with
query-specific E/P behavioral controls and measured resource accounting. Until
then, the result is implemented and CPU-verified wiring, persistence and
intervention behavior only.
