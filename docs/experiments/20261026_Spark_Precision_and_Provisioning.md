# Spark2_5 precision, extension and provisioning strategy

26 October 2026. This note defines the frozen-base Spark2_5 qualification path,
the single authorized FP8 derivative, the 40 GiB Spark-specific storage ledger,
and the separate H200/full-FP32 provisioning boundary. It records what is
implemented and what is **not** measured. No Spark weights are present locally;
no native Spark stage, weight download, heldout access or training run is
performed by this work. The parent workflow schedules the isolated environment,
downloads, tests and native stages and appends measured receipts.

Artefacts:

| Artefact | Role |
|---|---|
| `src/replica_lab/architecture/spark_adapter.py` | `Spark2_5Adapter`: frozen base, FP32 residual projector, optional 11/23 ports, optimizer/gradient audits, extension-only export/reload |
| `scripts/qualify_spark2_5.py` | Stage runner: load, inference, input gradients, one FP32 update, extension-only export and fresh-process reload |
| `scripts/dequantize_spark_fp8.py` | Offline CPU validator and single FP8 → BF16 derivative writer |
| `configs/spark_qualification_v1.json` | Exact source/config/shard pins, frozen prompts, stage policy, resource and storage policy |
| `tests/test_spark2_5_adapter.py` | Tiny synthetic adapter, optimizer, export/reload, runner-contract tests |
| `tests/test_dequantize_spark_fp8.py` | Tiny synthetic FP8 inventory, scale, dequantization and identity tests |

The existing `src/replica_lab/architecture/spark2_5.py` contract and
`src/replica_lab/runtime_pilot.py` are not modified.

## 1. Three precision identities

| Identity | Storage | Runtime activation semantics | Local admission |
|---|---|---|---|
| `official_bf16` | BF16, independent official release | unquantized BF16 | frozen base only |
| `official_native_fp8` | F8_E4M3 compressed-tensors + BF16 scales | dynamic per-token F8 activation quantization | **refused on MPS** |
| `dequantized_fp8_derivative` | new BF16 artifact built from FP8 bytes/scales | ordinary BF16/FP32 activations; no per-token F8 quantization | separate experiment, never official FP8 |

The derivative is not the official BF16 release and does not restore
quantization loss. It has a new artifact digest, new per-shard hashes, a new
config digest, and an explicit `activation_semantics` string. Only one
derivative identity is authorized.

## 2. Local MPS admission

Local qualification is deliberately narrow:

- one exclusive MPS worker, `--device mps`, `PYTORCH_ENABLE_MPS_FALLBACK=0`,
  `REPLICA_CAPACITY_GUARD=1`, and all three Hugging Face offline flags equal to
  `1`; the runner refuses anything else;
- `transformers` exactly `4.57.1` for BF16/derivative and `4.57.6` plus
  `compressed_tensors==0.16.0` for official FP8;
- eager attention is required and `use_cache=False` is forced for every
  training/inference forward; an implementation reporting anything else is
  refused;
- official FP8 cannot be allocated on MPS (`float8_e4m3fn`), so admission
  raises before `from_pretrained`; there is no CPU fallback and no implicit
  dequantization;
- the base is frozen. New tensors are FP32 and the only trainable ownership.
  This is adaptation, not full fine-tuning.

No native FP8 run, and no claim of native FP8 on MPS, follows from a successful
derivative run.

## 3. Adapter contract

`Spark2_5Adapter(base, config, precision, ...)` validates the pinned Spark
module names (`model.embedding`, `model.layers`, `model.norm`), the official
36×2560 geometry, tied input/output embeddings, eager attention, and the
device/precision admission. It stamps the original parameters, buffers and
config digest before training.

New modules live under names that match the preparation inventory:

```
extension.vision_projector.norm.{weight,bias}
extension.vision_projector.fc1.{weight,bias}
extension.vision_projector.fc2.{weight,bias}     # zero-initialized
extension.memory_ports.<layer>.*                 # optional, 11/23 hypotheses
```

The projector is `LayerNorm(head)->Linear(1024)->GELU->Linear(2560)`. Its final
projection is zero initialized, so the first forward is bit-exact with the
untouched base under the same dtype. Perturbing that projection changes the
logits; restoring it restores exact parity. The decoder is never wrapped in
`no_grad`; the projection input is a leaf for the input-gradient probe, and the
first backward leaves every new tensor with a finite gradient and at least the
final projection with a nonzero gradient.

`enable_memory_ports((11, 23))` is an explicit experimental identity change.
The ports are disabled for native parity, then enabled for the gradient/export
stage after parity has been recorded. Each port is zero-effect at
initialization. Layer 11 and 23 are full-attention hypotheses, not accepted
positions.

Extension export uses `build_extension_export` and adds a complete
`adapter_tensors` index, adapter settings and base snapshot. It contains only
`extension.*` tensors plus `base` identity, never checkpoint bytes:

```
extension/
  extensions.safetensors
  manifest.json          # export payload, weights SHA256, contains_base_weights=false
```

Reload validates the manifest fingerprint, the weight bytes, the complete
projector+port inventory, the adapter settings, and the base artifact/config/
precision digests. A stale base, changed shard, changed config or missing port
tensor is refused. The runner then compares logits and greedy tokens in a
**fresh process** after releasing the parent model. No base checkpoint is saved
or copied during qualification.

`optimizer_groups(base_lr, new_lr)` returns exactly one `new` group containing
every trainable extension tensor once. `build_optimizer(..., "adamw_fused")`
routes it to `NativeFusedAdamW`, which requires FP32 parameters and fused
AdamW with no fallback. The runner performs one step and records named deltas,
finite parameters, the post-step input gradient, and the unchanged base stamp.
Whether the fused AdamW kernel itself executes on MPS is not measured here; a
kernel failure is a raw stage failure, not a silent eager fallback.

## 4. Runner stages and evidence

`scripts/qualify_spark2_5.py` refuses a missing/changed official payload,
non-MPS execution, fallback, online access, native FP8 on MPS, an unlabelled or
implicitly dequantized source, and an existing output path. The plan pins not
only repository/config/shard SHA256 values but also the exact
`configuration_spark.py`, `modeling_spark.py`, tokenizer, template and generation
assets, so `trust_remote_code=True` executes only audited local bytes. The
runner writes a report after every stage and preserves partial failures.

| Stage | Required evidence |
|---|---|
| `load` | source shard/config hashes, precision identity, admission, eager/use_cache, 4,112,079,360 loaded elements, tied output embedding, tokenizer digest |
| `inference` | zero-initialized projector exact parity, nonzero perturbation effect, restored parity, three thinking modes at a smoke token budget with exact IDs/decoded text/stop reason/reasoning-token count; declared 512/2048 budgets are not executed or claimed |
| `input_gradients` | autograd enabled, gradient reaches projector input, all new gradients finite, at least one nonzero, sliding and full layers traced, base stamp unchanged, both port hypotheses enabled only after parity |
| `single_update` | one `adamw_fused` step, finite loss/parameters, named deltas, base unchanged, optimized FP32 state audit, a second probe showing a nonzero input gradient after the update |
| `export_reload` | extension-only files, complete tensor inventory, no base weights, fresh-process logits within `rtol=1e-5/atol=1e-6`, exact greedy tokens, base identity verified |

Resource evidence is `profile_samples.jsonl` (including `mps_driver_bytes`),
`profile_summary.json`, the parent capacity-guard report, and an explicit
`stop_reason`/completion status. The 50 GiB driver stop, 512 MiB sustained swap
growth and 15% minimum host-free rules are unchanged. The parent must add
`qualify_spark2_5.py` to the `--entry-point` choices in
`scripts/run_capacity_preflight.py`; this work does not edit that parent-owned
runner.

The access order is fixed as `frozen_prompt_set`, `development_smoke`, then
`no_heldout_role_materialized`. No train split is admitted; the heldout/audit
role is never opened, including after a development-stage failure. The plan
compares the BF16 reference and the derivative through matched frozen
prompts and the same stage protocol. After both arms complete,
`--stage compare --baseline-report ... --derivative-report ...` writes a paired
`replica.spark2_5_arm_comparison.v1` report: token agreement, decoded-text
hashes, reasoning counts, stop reasons and latencies. It does not require
bitwise equality across precision schemes and it does not access heldout roles;
once a dev-stage failure occurs the audit role is not opened.

## 5. FP8 dequantization

`scripts/dequantize_spark_fp8.py` runs offline on CPU and:

1. verifies every pinned shard byte count and SHA256, the config SHA256 and the
   required text/code assets;
2. validates the compressed-tensors group, the 470-tensor FP8 inventory, all
   180 F8 matrices, the 180 BF16 `[out,1]` scales, their one-to-one pairing,
   finite positive values and the 4,112,079,360 unique parameter elements;
3. converts one tensor at a time with
   `weight_bf16 = round_bf16(e4m3fn_to_f32(byte) * weight_scale[output_channel])`;
4. writes bounded BF16 output shards, an index, copied text/code assets, a new
   config without `quantization_config`, and `derivative_manifest.json` with
   parent identity, new artifact SHA256, per-shard hashes, scale provenance,
   script SHA256, scale-name digest, new config SHA256 and activation
   semantics;
5. validates the output through `verify_tensor_inventory` and
   `dequantized_fp8_derivative`, refusing a reused parent digest, a stale
   output path or a second derivative.

The CLI writes only to the single authorized root
`checkpoints/spark_fp8_derivative_v1` and refuses if it already exists. A
partial failed output is retained as raw evidence and cannot be reused.

## 6. Storage ledger and retention

The user constraint is **40 GiB total Spark-specific storage** with 477 GB free
at the start of this work. Planned worst case:

| Item | Bytes | GiB |
|---|---:|---:|
| Official FP8 payload | 4,451,626,120 | 4.146 |
| Official BF16 payload | 8,224,192,408 | 7.659 |
| Official subtotal | 12,675,818,528 | 11.806 |
| One BF16 derivative (cap) | 8,600,000,000 | 8.009 |
| Remaining for isolated env, staging, extension-only outputs, reports | 21,673,854,432 | 20.185 |
| Total budget | 42,949,672,960 | 40.000 |

The plan pins this arithmetic, sets `max_derivatives=1`, forbids full-base saves
during qualification, requires extension-only artefacts and requires at least
10 GiB free before each stage. The qualification report records
`storage.free_bytes_before` and `storage.free_bytes_after`; the derivative
writer records every output shard size and SHA256. Measured after-values are
appended by the parent only after acquisition/derivation actually runs. The
budget ledger starts at 477 GB free on this workstation; it is a ceiling on
Spark-specific bytes, not a claim that the disk is empty. Retention keeps both
official snapshots, the one derivative, extension-only outputs and raw failure
logs. No prior report or user evidence is deleted, and no repeated derivative
variants or full-base qualification copies are permitted.

The derivative writer bounds memory by writing ~1 GiB output shards; it does
not hold the full 8.2 GiB output in RAM. The adapter export writes only FP32
projector/port tensors (a few tens of MiB at the Spark width), not the base.

## 7. H200 and full-FP32 provisioning (separate admission)

The local 50 GiB guard is not a hardware maximum. Full-parameter Spark training
is **not locally admitted**:

| Scope | Precision/state plan | Persistent state | Vs 50 GiB |
|---|---|---:|---:|
| 4,112,079,360 backbone | FP32 weights + FP32 grads + two FP32 moments (16 B/param) | 61.275 GiB | exceeds by 11.275 |
| 4,320,804,612 integrated fixed scope | all FP32 (16 B/param) | 64.385 GiB | exceeds by 14.385 |
| 4,320,804,612 integrated fixed scope | BF16 weights + FP32 master + FP32 grads + two FP32 moments (18 B/param) | 72.433 GiB | exceeds by 22.433 |

Every row excludes activations, logits over 131,072 vocabulary rows, clipping,
allocator/compiler overhead, data, media, KV cache and checkpoint copies. Moving
optimizer state to host memory is not extra physical memory on this Mac and is
forbidden as a disguised fallback.

Full training therefore requires a **separate CUDA admission and provisioning
plan**, not a local exception:

- candidate: one H200 141 GB, BF16 compute with an explicitly declared FP32
  master/gradient/moment policy;
- CUDA admission must be measured against a fresh larger ledger, not inferred
  from the Mac guard, and must qualify the Spark custom attention path, tied
  weights, optimizer kernels, checkpoint peak and numerical tolerances;
- H100 80 GB requires a qualified sharding/offload implementation; the current
  analytical ledger exceeds 80 GB before activations;
- multi-GPU pairs are separate scheduled jobs unless a sharding/offload path is
  independently qualified; do not pool independent cards automatically;
- one parent-scheduled native job at a time; no FP8 training claim, no Muon on a
  frozen FP8 base, and no relabelling of a lower-precision or partial update as
  full fine-tuning.

The existing compute-estimate report remains the planning baseline for H200
hours and cost; this note only fixes the precision/provisioning boundary.

## 8. Implemented, measured, rejected and not claimed

**Implemented (offline, synthetic tests only)**

- official geometry/precision admission and the frozen-base adapter;
- zero-init parity and nonzero perturbation, freeze scope, FP32 optimizer
  ownership, finite/nonzero new gradients, one-step update evidence;
- complete projector+port extension inventory, extension-only save/load, stale
  base rejection;
- exact source/config/shard pins, environment refusal, missing-payload refusal,
  implicit-dequantization refusal and fresh-process reload comparison;
- the single FP8 derivative writer with scale provenance and a new identity;
- the 40 GiB storage policy and H200/full-FP32 boundary.

**Not measured**

- no Spark weight payload is present locally; no Spark model has been loaded;
- no MPS BF16/derivative inference, gradient, update or reload has run;
- no fused AdamW execution on MPS is established;
- no H200 or other CUDA qualification exists;
- no generation at the declared 512/2048 budgets, no heldout role, no training
  convergence, throughput or capability claim.

**Rejected**

- native FP8 on MPS, CPU fallback, implicit dequantization, official-BF16
  labelling of a derivative, a second derivative variant, full-base saves
  during qualification, local full FP32 Spark fine-tuning under the 50 GiB guard.

**Report honesty fields**

Every runner report carries `weights_downloaded`,
`remote_code_executed`, `native_execution`, `fallback_used`,
`dequantization_performed` and `capability_claim: none`; measured values are
appended by the parent only after the corresponding stage actually runs.

## 9. Next gates for the parent

1. Install the isolated 4.57.x Spark environment; do not upgrade the locked
   MiniCPM5 environment.
2. Acquire the pinned BF16 and FP8 snapshots under a separate weight
   authorization, verify every shard/config pin, and check the 40 GiB ledger.
3. Run the tiny CPU tests:
   `pytest tests/test_spark2_5_adapter.py tests/test_dequantize_spark_fp8.py -q`.
4. Generate the one derivative with `scripts/dequantize_spark_fp8.py`, then run
   the BF16 and derivative qualification paths through
   `scripts/run_capacity_preflight.py` with fresh output paths, one worker,
   `REPLICA_CAPACITY_GUARD=1`, offline flags and fallback disabled.
5. Keep native FP8 and full FP32 Spark training blocked until their separate
   backend/provisioning admissions are qualified.
