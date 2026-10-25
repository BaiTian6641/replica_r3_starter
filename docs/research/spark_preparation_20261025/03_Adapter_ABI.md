# Dedicated Spark2_5 adapter ABI — design only

**Status:** design recorded in `preparation_manifest.json` under `adapter_abi`.
No `Spark2_5Adapter` class exists yet, no Spark model is loaded, and no new
module is attached. This is deliberately not a disconnected fake adapter: the
manifest carries the ABI, the source identity it must enforce, the actual class
structure and the invariants derived from the inspected upstream code.

## Why a dedicated adapter

`model_type=spark2_5` is a custom architecture with `auto_map` to
`configuration_spark.Spark2_5Config` and `modeling_spark.Spark2_5ForCausalLM`.
It is not Llama, and it is not the width-2048 MiniCPM5 layout. The inspected
code hard-codes `model.embedding`, `model.layers[i]` and `model.norm`, uses a
fused `q_k_v_proj`, a per-head sigmoid `g_proj`, 16/4 attention, head dimension
256, 36 layers and a 3-sliding/1-full schedule. A relaxed Llama adapter cannot
represent those facts.

## Construction contract

| Stage | Required behavior |
|---|---|
| Source admission | Load only a sealed local snapshot. Verify repository, revision, config SHA256, storage mode, quantization-config digest, shard header hashes and (when present) payload hashes against this preparation manifest. |
| Remote code | Never fetch or execute remote Python at load time. Load the already audited local `configuration_spark.py` / `modeling_spark.py` with `trust_remote_code` restricted to that sealed directory. Refuse if either file hash differs. |
| Precision | Explicit `official_fp8` or `official_bf16`. Refuse a different storage mode. Never auto-dequantize to make load succeed. |
| Topology | Require 36 layers, hidden 2560, MLP 10240, 16 query / 4 KV heads, head dimension 256, vocabulary 131,072, sliding window 512, tied embeddings, the exact layer-type list and the two RoPE parameter sets. |
| Inventory | Require the header inventory from this preparation: FP8 180 F8_E4M3 projection tensors plus 180 BF16 channel scales and BF16 head gates/norms/embedding; BF16 all-BF16; no serialized `lm_head.weight`; no zero points. |
| Device | Explicit device/dtype policy. Refuse MPS when the precision path is not actually available; no CPU fallback while claiming native execution. |
| Worker admission | One exclusive native worker, guarded by the existing lock and host checks. |

## API surface

```text
Spark2_5Adapter.load(sealed_source, manifest, *, precision, device_policy) -> adapter
adapter.forward_text(input_ids, attention_mask=None, position_ids=None,
                     past_key_values=None, use_cache=False) -> logits
adapter.forward_hidden(input_ids, ...) -> final_hidden_states
adapter.generate_text(messages, *, thinking="none"|"low"|"high",
                      max_new_tokens, generation_kwargs) -> GenerationResult
adapter.reset_cache() -> None
adapter.snapshot_cache() -> opaque_session_owned_state
adapter.attach_projector(in_features, out_features=2560,
                         zero_init_output=True) -> module_handle
adapter.enable_memory_ports(layers) -> candidate_handle
adapter.export_candidate(path) -> CandidateManifest
adapter.reload_candidate(path) -> adapter
```

`GenerationResult` must include text, prompt/output token IDs, stop reason,
reasoning-token count, template mode and actual `max_new_tokens`.

## Invariants

1. Original checkpoint bytes are never rewritten by candidate export.
2. `no_grad` is never placed around the frozen decoder when new modules need
   input gradients. Freezing means `requires_grad=False` on original parameters,
   not detaching their inputs.
3. Every forward records the storage mode and aborts on an implicit dequantized
   fallback.
4. Sliding/full layer types come from the pinned config list, not from an assumed
   alternating pattern. Port positions 11 and 23 are hypotheses; they are not
   accepted memory positions.
5. Ports are disabled for native parity; enabling them changes the candidate
   identity.
6. Cache behavior is measured. The upstream shared `DynamicCache` does not evict
   sliding-window keys, so the adapter may not claim a bounded 512-token cache.
7. Generation uses the pinned chat template. Thinking mode maps to
   `enable_thinking=false`, or `enable_thinking=true` with
   `reasoning_effort=low|high`. Reasoning tokens are reported, not silently
   stripped.
8. The residual stream is float32 in upstream code while layer compute follows
   linear weight dtype; the adapter reports its actual dtype policy and does not
   label the run uniformly BF16 or FP8.
9. Candidate manifests carry base identity plus trainable tensors only. A
   cross-precision or changed-base reload is rejected.
10. Optimizer membership is explicit: AdamW for new float32 masters; Muon
    only for separately audited eligible groups, never as proof of FP8 training.

## New-module layout

* Projector input width: final hidden width **2560** (not 2048).
* New connector matrices, memory readers/writers and serialized fast states from
  MiniCPM5 are not transplanted.
* A memory-port candidate may test complete block boundaries, but 11/23 remain
  hypotheses. Full-attention blocks are 3, 7, 11, 15, 19, 23, 27, 31, 35 in
  zero-based indexing; complete-block ports must account for the three preceding
  sliding blocks, not just a single layer.
* Candidate memory state is session-owned and functional; no source snapshot is
  mutated during generation.

## Explicit non-goals

Native execution in this task; a class that pretends to load Spark before backend
qualification; reusing MiniCPM5 width-2048 components; treating MLX inference as
training evidence; or claiming MPS FP8 support from configuration parsing.
