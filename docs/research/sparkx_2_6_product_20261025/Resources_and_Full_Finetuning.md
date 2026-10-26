# Resource feasibility and full fine-tuning

25 October 2026. Analytical companion to the [product plan](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md). [Reproducible calculator](../../../reports/sparkx_2_6_product_refinement_20261025/estimate_costs.py), [output](../../../reports/sparkx_2_6_product_refinement_20261025/analytical_costs.json). No new model loads, training, timing or native profiling occurred in this review.

## Separate product inference from product training

The pinned Spark tensor headers contain 290 serialized tensors and 4,112,079,360 unique parameters. Counting a second tied language head would overstate them. BF16 weight payload is 7.659 GiB; FP32 is 15.319 GiB. These are not peak allocations.

| Backbone training-state hypothesis | GiB, before activations | Interpretation |
|---|---:|---|
| FP32 weight + gradient + two FP32 Adam moments | 61.275 | Already above the 50 GiB local guard |
| BF16 weight + gradient + FP32 moments, no FP32 master | 45.956 | Little room remains; numerical/native qualification required |
| Same with FP32 master weights | 61.275 | Mixed precision alone does not solve residency |
| Hypothetical BF16 weight/gradient + two int8 moments, no master | 22.978 | Arithmetic only; no qualified MPS implementation/convergence claim |
| Current-style FP32 Muon with int8 momentum, FP32 Adam auxiliary tensors | 36.656 | Candidate assignment, not a measured Spark run; scales, quantization and orthogonalization buffers omitted |

The last row assigns non-embedding 2D matrices to Muon (3,776,348,160 parameters) and embedding/non-2D tensors to AdamW (335,731,200). The formula is `9P + 7P_aux` bytes: FP32 weights/gradients, one int8 momentum for Muon matrices, and two FP32 moments for auxiliaries. Real group ownership must be audited, including the custom tied head and any added latent embeddings. State compression does not remove full-backbone gradients or matrix-update FLOPs.

The existing [MiniCPM measurement](../../experiments/20261025_M2_Ultra_Full_Training_Performance.md) trained all original parameters and changed all 381 original tensors; it does not establish that every scalar changed. At 55 positions/example and three measured updates, native fused AdamW was 2.57× faster than the current Muon8bit implementation, but used a larger peak (40.03 versus 30.08 GiB). During this review, commit `023e8b9` added a production fused-optimizer path, superseding the historical report's profiling-only status. Its first new smoke attempt failed in a reporting check, so sustained native qualification is still unestablished at this snapshot. Do not multiply the short-probe speedup into a production-training forecast or assume equivalent quality. The failed vision optimizer comparison is a different scope and cannot choose the full-backbone optimizer by itself.

Qualify full Spark updates first at a very short sequence. Then increase to the actual image/audio lengths while preserving all-parameter ownership, checkpointing and exact supervision masks. Selective-position/chunked logits can reduce loss-buffer cost; they require forward/loss/gradient equivalence checks. Audit dtype casts, clipping/synchronization overhead, optimizer scratch and checkpoint save peaks. If full scope cannot fit, obtain a larger-memory offline training environment rather than quietly freezing parameters. Provisioning that environment is a later concrete decision, not performed here.

## Audio additions are small in weights, potentially large in sequence cost

[Whisper source](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/model.py) and pinned configs support an encoder-only analytic inventory. Include convolution/attention/MLP/norm weights and fixed positional storage; exclude its text decoder.

| Component | Elements/parameters | BF16 payload | FP32 payload |
|---|---:|---:|---:|
| Whisper-base encoder including fixed positions | 20,590,592 | 39.27 MiB | 78.55 MiB |
| Whisper-small encoder including fixed positions | 88,154,112 | 168.14 MiB | 336.28 MiB |
| SPEAR Base v2, official metadata | 93,289,476 | 177.94 MiB | 355.87 MiB |
| Existing SigLIP2 vision encoder | 92,884,224 | 177.16 MiB | 354.33 MiB |

Whisper counts are formula-derived stored elements, not a new safetensors inventory; its OpenAI implementation keeps fixed positions as a buffer. SPEAR metadata is an observed serialized parameter count. BF16 payload does not establish qualified BF16 compute: the existing visual control used FP32 and both audio candidates remain unqualified here.

A `512→1024→2560` audio MLP with biases adds 3,149,312 parameters (about 6.01 MiB BF16); a 768-input bridge adds 3,411,456. Selected-layer fusion, norms, event/affect heads and temporal processing are additional and must be inventoried. A model with both sensory towers is naturally larger than the 4.112B backbone; that matches the user's parameter boundary.

At 10 audio tokens/second, one hour generates **36,000 conditioning positions**. They cannot remain in active context indefinitely. A 16-second window has 160 audio positions; four 64-token frames, two 64-token crops, 64 event positions and 32 markers yield a proposed maximum 640 media positions/window. This compression is admitted only after richer controls work. Token rates must include timestamp/marker overhead and actual padding.

## Cache and bounded state

Spark passes configuration to `DynamicCache`. [Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py) constructs sliding layers for the configured local blocks; those retain the most recent window-minus-one positions between calls. Full-attention layers grow. Inspecting the framework dependency is essential—neither “DynamicCache is always unbounded” nor “sliding masks bound all resident state” is correct.

For one stream with BF16 K/V, no replication, four KV heads and head dimension 256:

`KV bytes ≈ 2(K,V) × 2(dtype) × 4 × 256 × [9N + 27 min(N,512)]`.

This conservative local-window upper bound gives 342 MiB at 8192 positions, 1206 MiB at 32768, and approximately 36.05 GiB at 1,048,576. All-layer unbounded storage would instead be 144 GiB at the last length. FP32 doubles the payload. These formulas exclude concatenation/prefill temporaries, allocator overhead and attention matrices; eager attention can dominate even when persistent KV is small.

The inspected sliding update slices concatenated tensors; a logical window length alone does not prove that backing storage or prefill transients have already been released. Profile actual allocation across prefill and subsequent decode, rather than counting tensor shapes only.

Initial Spark bank payload is 3 MiB (4096×(128+256) BF16), active shared phrase rows 1.25 MiB, core slots 4 KiB and event vectors 32 KiB. Two 128×256 FP32 fast matrices would be 0.25 MiB before optimizer/gating/projection state. These small payloads **exclude the knowledge surfaces, source media, provenance, replay and semantic encoders that make the state usable**. Persisting raw clips without quotas would invalidate the bounded-memory claim.

A reasonable product direction is a small number of native ports, shared sensory towers, bounded active context and selective memory reads, with teacher networks absent at deployment. The feasibility gate must measure complete capture-to-answer and update-to-future-benefit paths, not only matrix sizes.

## Required cost report for every candidate

Record warm/cold latency, time-to-first-answer, generated bytes, processed positions, audio seconds, frames/crops, peak driver/host memory, KV and bank size, optimizer steps, forward/backward positions, verifier calls, replay bytes and rejected candidates. Show energy only if actually measured. Report both amortized totals and worst-window P95; a low average can hide expensive learning stalls.

Local guards remain: one MPS worker, 50 GiB driver cap, 512 MiB sustained swap-growth cap, existing 15% host-free-memory sampling rule and no silent CPU fallback. Optional remote full FT changes the training environment, not the product's model-size target or deployment acceptance criteria.
