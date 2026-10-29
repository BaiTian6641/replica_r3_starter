# Native backend plan — Spark persistent learner

**Status: exact official-FP8 execution is blocked on this workstation. No native P1 campaign is launched or authorized by this document.**

## 1. Exact-FP8 blocker

The active source is `XHToken/Spark-X2.6-4B-FP8` at `0318d6f6ab070c7b5b43052db571489b2508006c`.

Observed blockers:

- MPS cannot allocate `torch.float8_e4m3fn` in the recorded backend audit.
- `Spark2_5Adapter` has no native-FP8 admission contract and refuses native-FP8 construction.
- The installed compressed-tensors path has no qualified W8A8 FP8 compute kernel stack and can decompress during forward. Silent decompression is not native-FP8 execution and cannot satisfy the protocol.
- There is no exact forward, extension-input backward, fused-update or restart receipt for official FP8.

Therefore the current official-FP8/MPS protocol is a precise implementation blocker, not merely a missing receipt. Do not issue a native P1 launch.

## 2. Requirements for a future exact backend

A future exact backend must provide all of the following before a campaign:

1. FP8-capable device and kernels for `q_k_v_proj`, `out_proj`, `gate_proj`, `up_proj` and `down_proj`, with channel-wise weight scales and dynamic per-token activation quantization preserved.
2. No silent dequantization or BF16/FP32 substitution. Source config, weight payload, tokenizer, template and execution identity are checked before loading.
3. Adapter admission that explicitly supports this exact backend and its autograd boundary. Forward equivalence against the reference implementation is checked first.
4. Finite backward gradients into selected P and final port inputs while the original backbone remains frozen. One actual `NativeFusedAdamW` update is measured, including moments and clock.
5. Frozen-base tensor digest before/after the update, extension-only checkpoint/reload, memory/resource receipts and one fresh-process continuation.

A CUDA migration would need a new protocol, source/backend qualification and authorization; it is not an MPS workaround.

## 3. Feasible but separate derived-BF16 path

The existing `checkpoints/spark_fp8_derivative_v1` artifact has a preserved bounded qualification receipt. It is a **dequantized FP8-derived BF16 artifact**, not official BF16 and not native FP8.

The derivative conversion is explicitly:

`weight_bf16 = round_bf16(e4m3fn_to_f32(byte) * weight_scale[output_channel])`

Its parent is the official FP8 source. Conversion removes FP8 activation semantics and uses BF16/FP32 activations. Existing evidence covers zero-init parity, extension-input gradients, one fused AdamW step, frozen-base integrity and extension-only fresh-process reload. It does not establish useful learning or interchangeability with official BF16.

A new experiment using that artifact requires a separately versioned protocol containing:

- parent/source revision and conversion manifest, exact dequantizer/config/weight hashes, tensor/scale inventory and activation semantics;
- a new execution identity `dequantized_fp8_derivative_bf16_storage`, never `official_native_fp8` or `official_bf16`;
- numerical forward checks against the derivative's own reference, extension-input backward checks, one fused update, frozen-base integrity and reload;
- external authorization for the exact source/backend/implementation and a fresh output root.

Official BF16 is a third identity and requires its own source/protocol. No precision substitution is allowed.

## 4. Smallest useful experiment packet

Only after the selected backend is qualified and authorized:

- One active P1 expert; original backbone frozen; no larger-bank allocation.
- Direct response CE at zero-based sites 3/7 versus 23/31 with identical information and charged compute.
- Eleven arms: untouched full context; selective exact/procedure; spine E-only; reader-trained E-only; source-free P-only; E+P; no-update; reset-P; shuffled memory; inapplicable procedure; matched replay without update.
- The full-context arm uses `all_eligible` with at most 16 records/64 value slots and refuses truncation or incomplete authority. Semantic arms keep two selected records. Failed-check and quarantined records are excluded before key reads.
- Default readout uses metadata/order selection before payload acquisition (`admit_payload=False`), so validated warm cache hits avoid repeated text/value reads. `admit_payload=True` preserves the older selection-admitted control. Active buffers are capped at 2 MiB cache plus 73,728 bytes of returned payload, not claimed as total or GPU-resident memory.
- Four chronological fit events, twelve development cases per arm, two parity requests and one fresh-process restart. Twelve cases remain engineering diagnostics, not strong capability evidence; independent worlds/rule families must expand before a quality claim.
- Thinking off, greedy decoding, maximum 32 new tokens per request, 2,048 context tokens, no sealed-final access.
- Current upper bounds: 135 generation requests, 4,320 generated tokens, 8,665 forwards, 17,879,840 processed positions, 148 retrieval calls and 147 independent verification calls. Readout ceilings are 296 selection attempts, 66,304 total file reads, 23,680 payload/cold reads, 1 GiB I/O, 2,368 cold tickets/resumes and 7,104 revalidations; cache and all other upper bounds are not measurements.
- Resource guards: one exclusive native worker, 50 GiB sampled driver limit, 512 MiB sustained swap growth, 15% minimum host free, CPU fallback disabled, offline local files only.
- Report paired reader/P/retrieval deltas with uncertainty over independent worlds. Report teaching, encoding, retrieval, replay, generation, verification, latency and artifact costs separately. Compare total cost, not just accuracy.

## 5. Current decision

Recovery, controls and tiered reads are CPU-verified at implementation commit `83c7214bcf12877e4c7b9d0c367825905a16cf55`; see `integrated09` and `readout_pressure05.json`. Two fresh-process restart cycles are preserved under `integrated08/restart/`. Keep the official-FP8 gate closed. The separate derived-BF16 protocol and packet are prepared at the same commit, but native launch still requires exact external derivative authorization. Do not call that experiment native FP8, do not use it to claim exact-FP8 capability, and do not proceed to capacity expansion, the learned writer or RSI without useful full-context readout.
