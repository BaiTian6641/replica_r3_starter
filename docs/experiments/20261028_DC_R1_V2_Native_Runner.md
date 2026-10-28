# DC-R1 v2 native readout runner

**28 October 2026. Status: runner implemented and CPU-tested; no native
authorization or MPS execution occurred.** Implementation commit:
`deba2428`.

## Delivered

- `src/replica_lab/dc_r1_native_backend.py` implements independent greedy
  readout at layers 13, 27 and 41. Production loading uses the sealed local
  MiniCPM5 source through the existing exclusive MPS loader. Every request uses
  a growing prefix with `use_cache=False`; only the final hidden position is
  normed and projected. Terminal stop IDs are excluded before decoding, and the
  posted work stream follows the v2 scorer schema. There is no optimizer,
  backward pass, checkpoint write, download or CPU fallback in the production
  path.
- `src/replica_lab/dc_r1_native_runner.py` verifies the external authorization,
  exact config and prepared fixture, reserves and charges sequential physical
  work, writes all-layer prediction records, invokes the real scorer, and
  publishes reports, predictions and a separate execution-admission receipt.
  Injected backends are labeled `cpu_fixture` and can never make native
  admission true.
- Development evidence is canonical scorer output. Sealed-final admission
  independently recomputes that evidence with the same native budget mode
  before final labels are hashed or parsed and before the final backend is
  constructed. Development validation defers `sealed_labels_final.jsonl`
  hashing through the shared `skip_hash_files` manifest API.
- Stage journals persist progress after each charged layer. Resume skips
  completed `(condition, case_id)` pairs, retains partial-case work in the
  cumulative budget, and preserves failed/incomplete state. The declared DC-R1
  budget still has zero retries and one model load; resume is not an implicit
  native retry or reload authorization.
- `scripts/run_dc_r1_native_value_v2_20261027.py` is the separate MPS-only
  stage CLI. `scripts/preflight_dc_r1_native_value_v2_20261027.py --launch`
  now admits a valid external receipt only when the declared runner exists and
  emits a `run_capacity_preflight.py` command. Preflight itself remains
  review-only and performs no model, tokenizer or native work.
- The v2 scorer wrapper now forwards `config_path` and `native_execution`
  correctly on both splits; independent development admission uses the same
  budget mode as the original native report.

## Verification

The following completed successfully:

```bash
.venv/bin/python -m pytest \
  tests/test_dc_r1_native_runner_v2.py \
  tests/test_dc_r1_native_value_v2_20261027.py \
  tests/test_dc_r1_native_value_20261027.py \
  tests/test_dc_p1_native_response_20261027.py \
  tests/test_dc_p1_native_response_v2_20261027.py \
  tests/test_dc_p1_native_runner_v2.py \
  tests/test_dc_p1_scoring_cli.py \
  tests/test_artifact_publication.py -q
# 126 passed
```

Coverage includes authorization refusal before backend work, source-free
all-layer prediction aggregation, real scorer admission, canonical development
recomputation, final-label deferral, publication, work-reservation refusal,
interrupted/resumed progress, actual launch-preflight CLI behavior, and an
actual tiny 42-layer Llama forward/decode/work smoke using the sealed tokenizer.
The tiny model and injected backends are CPU test fixtures only; they are not
MiniCPM5 native evidence or learned readout results.

## Native gate and remaining work

No external DC-R1 authorization receipt was created or used for a native job.
Consequently there is no MPS quality result, native latency/memory measurement,
or selected native artifact to report. The sealed source and existing fixture
bytes were reused; no model download or duplicate base checkpoint occurred.
Native compressed-channel gates remain
`blocked_no_native_compressed_channel_receipt`; host codec reconstruction is
still only an audit/control. Useful learned retrieval, calibrated abstention,
persistent memory and promotion remain unestablished.

A future authorized run must use the exact committed protocol/config/fixture,
fresh output, the capacity watchdog and the external receipt. Its raw
predictions, unsuccessful trajectories, independently recomputed reports,
resource measurements and artifact inventory must be retained as a new
experiment directory. Engineering wiring and persistence passing here does not
establish native readout quality.
