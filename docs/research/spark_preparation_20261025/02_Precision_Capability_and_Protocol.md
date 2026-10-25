# Precision capability matrix and prospective qualification protocol

**No cell in this document is measured.** The only evidence is immutable source,
text-asset, configuration and safetensors-header inspection. There is no native
Spark forward, backward, optimizer step, export, reload, MPS test or MLX test in
this work package. In particular, **there is no claim that MPS FP8 training is
supported**.

## 1. Separate precision concepts

| Concept | FP8 source | BF16 control |
|---|---|---|
| Checkpoint storage | 180 F8_E4M3 projection tensors with static symmetric per-output-channel BF16 scales; 337,205,760 BF16 parameters, including head gates, norms, embedding and tied logits | all 4,112,079,360 parameters BF16 |
| Activation quantization | dynamic symmetric per-token F8_E4M3 declared by compressed-tensors; no output-activation quantization | none from the checkpoint |
| Forward/backward compute | not established by storage; requires actual kernels | not established locally |
| Optimizer state | none while frozen; separate policy for new modules | none while frozen; separate policy for new modules |
| KV cache | unquantized (`kv_cache_scheme=null`) | unquantized |
| Zero points | none | none |

Upstream compressed-tensors semantics are `float-quantized` 0.16.0:
**static channel-wise weights** (`strategy=channel`, `dynamic=false`,
`symmetric=true`, 8-bit F8_E4M3, MSE observer, one BF16 scale per output
channel) and **dynamic token-wise input activations** (`strategy=token`,
`dynamic=true`, `symmetric=true`, 8-bit F8_E4M3). Quantized targets are fused
`q_k_v_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`. Head gates
(`g_proj`), norms, embedding and tied logits are BF16. The top-level
`dtype=bfloat16` does not make the F8_E4M3 weights BF16.

Muon 8-bit momentum, if used later, changes only its supported optimizer state.
It supplies no FP8 kernel and does not make weights, activations or gradients
FP8.

## 2. Independent capability matrix

Statuses are deliberately conservative. "Unqualified" means not demonstrated,
not "known broken".

| Capability | Official FP8 `@0318d6f` | Official BF16 `@66e3ae7` |
|---|---|---|
| Unmodified forward inference | **Unqualified.** Needs an isolated 4.57.x + compressed-tensors stack and a device that actually supports the declared quantized path. Upstream reports no MPS result. | **Unqualified** locally. BF16 eager execution is a plausible first target, not a measured result. |
| Input gradients through a frozen decoder | **Unknown.** Quantized linear layers may pass input gradients, but dynamic FP8 activation quantization on MPS is unproven. A `no_grad` decoder would defeat the proposed projector path. | **Proposed.** Must be traced to the new projector input; not inferred from dtype. |
| Parameter gradients / native updates | **Deferred.** No FP8 training kernels, master/scaling policy or optimizer support established. | **Proposed experiment**, only after the input-gradient and reload gates. |
| Optimizer membership | Frozen base has none. New projector/LoRA parameters use a declared optimizer. Muon only for audited eligible groups. | Same new-module policy; base optimizer membership differs by experiment. |
| Export/reload | **Proposed.** Persist the FP8 storage identity plus new trainable tensors; fresh-process reload must not change format. | **Proposed.** Persist BF16 base identity plus the same new-module schema; reject cross-precision reload. |
| Long-context / cache | **Unqualified.** Upstream `DynamicCache` does not evict sliding-window keys; 512 is a mask window, not a bounded cache. Configured 1,048,576 positions are not measured capacity. | Same structural caveat. |
| MLX inference reference | **Not admitted.** `Spark-MLX-LLM` is a separate implementation requiring its own pin and checkpoint qualification; it is not PyTorch/MPS training evidence. | Not applicable to training evidence. |
| Dequantized derivative | A *new named artifact* only. It is not the official BF16 release and stops matching the original FP8 activation semantics unless separately specified. | Not a substitute for the FP8 release either. |

### Storage-versus-compute warning

Nothing above authorizes automatic dequantization. A loader that silently
converts F8_E4M3 weights to BF16 makes a different artifact and different
forward semantics. It must be recorded as an explicitly named derivative with
new per-shard hashes; it must never load under either official identity. The
dequantized branch is not created in this work.

## 3. Isolated environment requirements

Do not upgrade the active locked MiniCPM5 environment. The Spark path requires a
dedicated environment because:

* the custom source imports 4.57-era `transformers.masking_utils`,
  `TransformersKwargs`, `processing_utils.Unpack` and
  `DynamicCache(config=...)`, and the two checkpoints serialize 4.57.6 (FP8) and
  4.57.1 (BF16);
* FP8 loading requires compressed-tensors 0.16.0 and its Transformers
  integration;
* the upstream quickstart uses `trust_remote_code=True`, which this project
  replaces with a sealed local source audited by hash and loaded only from local
  files;
* the project's native runner enforces `PYTORCH_ENABLE_MPS_FALLBACK=0`, offline
  Hub variables and an exclusive MPS worker. Those remain required; CPU
  fallback cannot be relabeled native.

Unresolved: exact `torch` wheel/version, MPS kernel availability for F8_E4M3 and
dynamic activation quantization, and whether Transformers 4.57.x loads both
precisions without a conversion warning. Artificial compatibility cannot be
declared from config parsing.

## 4. Prospective matched protocol

This is a prospective protocol design, not an admitted native run. Exact prompt
and data hashes, numeric quality tolerances and environment pins must be committed
before execution under the single native-worker admission rule.

### 4.1 Sources and inputs

* FP8: `XHToken/Spark-X2.6-4B-FP8@0318d6f6ab070c7b5b43052db571489b2508006c`
* BF16: `XHToken/Spark-X2.6-4B@66e3ae7bee54e0a523d692cb8ef841f35409807b`
* Frozen, source-independent prompts shared with MiniCPM5 preparation; no Spark
  training data used for the comparison set.
* Thinking mode `none` (`enable_thinking=false`), `low` (true + effort low) and
  `high` (true + effort high), with actual reasoning-token counts reported.
* Fixed greedy runs and preregistered sampled runs; token budgets 512 and 2,048
  new tokens. Do not copy the generation config's 1,048,576 `max_tokens` into a
  test.

### 4.2 Matched inference

Run both sources on the same prompts and template mode. Record exact prompt and
output token IDs, decoded text, stop reason, reasoning-token count, latency and
peak device bytes. Compare paired task accuracy, abstention and length with
preregistered tolerances. Never require bitwise equality across precision
schemes. Test a measured small long-prefix budget before extrapolating toward
the configured maximum.

### 4.3 Input-gradient probe

Freeze every original parameter and attach one new projector to hidden width
2,560 with a zero-initialized output projection. Do **not** wrap the decoder in
`no_grad`. Require:

1. gradient reaches the projector input;
2. all new-module gradients are finite and at least one is nonzero after a step;
3. original weight bytes and parameter versions are unchanged;
4. the FP8 path still uses dynamic activation quantization; a dequantized
   fallback aborts the probe;
5. both sliding and full blocks occur in the traced path.

Failure records the unsupported kernel or backend; preparation status remains.

### 4.4 One declared update

Train only one declared new module group. Use AdamW with a float32 master for new
parameters; do not place Muon over the frozen FP8 base. Fix and record the seed.
Accept only finite loss/gradients, named parameter deltas, unchanged base digest
and no NaN/Inf. A successful update is **not** evidence of FP8 weight training.

### 4.5 Export and fresh-process reload

Export new trainable tensors plus a base identity: repository, revision, storage
mode, quantization-config hash, shard header hashes and (when separately
acquired and verified) shard payload hashes. Reload in a fresh process with the
same precision semantics, without automatic conversion. Compare logits/tokens
under a preregistered tolerance. Reject a mismatched base digest, storage mode or
quantization config. Check persisted optimizer state separately.

### 4.6 Dequantized-derivative branch

If the backend cannot support the official FP8 storage, a derivative requires a
new identifier, new per-shard SHA256, a published dequantization script and
scale provenance, explicit changed activation semantics, and an independent
comparison with the official BF16 control. It remains ineligible to claim the
official FP8 identity or official FP8 training.
