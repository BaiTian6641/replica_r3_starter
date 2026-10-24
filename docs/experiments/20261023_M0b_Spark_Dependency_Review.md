# M0b — Spark-X2.6-4B BF16 dependency review

23 October 2026 UTC. **Status: source/configuration reviewed; native adapter,
weights, training and quality comparison remain unrun.** This optional arm does
not change the MiniCPM5 baseline.

## Official source inspected

- [Official model card](https://huggingface.co/XHToken/Spark-X2.6-4B)
- [Official config](https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/config.json)
- [Official implementation](https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/modeling_spark.py)
- [Official configuration class](https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/configuration_spark.py)
- [Official tokenizer config](https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/tokenizer_config.json)
- [Official chat template](https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/chat_template.jinja)

The official source identifies `Spark2_5ForCausalLM` with **36 layers**, hidden
width **2560**, intermediate width **10240**, **16 attention heads**, **4 KV
heads**, head width **256**, vocabulary **131072**, and tied input/output
embeddings. Full attention occurs at zero-based layers **3, 7, 11, 15, 19, 23,
27, 31, 35**; the other layers use sliding attention with a 512-token window.
Sliding and full attention have different rotary settings.

The implementation uses a fused `q_k_v_proj`, optional per-head output gates,
separate output projection, RMSNorm, GELU MLP and residual decoder blocks. The
model card requires **Transformers 4.57.1** and custom remote code. The template
supports `enable_thinking` and `reasoning_effort=low|high`; native start/end
markers and tool-call syntax must not be replaced by MiniCPM's template.

## Local dependency result

| Requirement | Current evidence |
|---|---|
| Local BF16 weights | Not present. `checkpoints/` contains only `minicpm5_source/config.json` |
| Locked Transformers | **5.20.2**; no `transformers.models.spark2_5` module |
| Official requirement | `transformers==4.57.1` plus custom `modeling_spark.py` / `configuration_spark.py` |
| Tokenizer/template | Official files reviewed; no local tokenizer or template acquired |
| Native execution | Not run; no parity, gradient, isolation, optimizer or export claim |

A separate environment or explicitly verified custom-code path is required
before any acquisition or native run. The locked MiniCPM environment remains
unchanged. No optional download was attempted.

## Required next work before comparison

1. Pin an authorized local BF16 source and exact acquisition receipt.
2. Isolate and verify the official implementation, tokenizer and template;
   record any dependency difference instead of silently using the MiniCPM lock.
3. Implement a separate adapter with complete native blocks and topology-derived
   ports. Do not relax the Llama adapter's checks or copy layers 13/27/41.
4. Count tied embeddings once in trainable membership; qualify null parity,
   causal gradients, request isolation, optimizer groups and export/reload.
5. Measure native inference and modules-only backward before any bounded full
   update. BF16 compute does not imply BF16 parameter residency.
6. Compare native Spark and MiniCPM5 first, then the same memory contract with
   common tasks, tools, verification and output/time budgets. Pin reasoning mode
   and count reasoning tokens, latency, memory and total cost.

Spark's nominal size and benchmark results are not evidence of local training
fit or independent quality. MiniCPM5 M1–M6 work proceeds independently.

## Evidence boundary

This document is a dependency and implementation review only. It reports no new
model execution, benchmark reproduction, checkpoint, download or backbone
promotion. See the [M0–M7 construction plan](../architecture/15_Research_Crosscheck_and_Backbone_Plan.md)
and [loop instructions](../architecture/16_Loop_Agent_Instructions.md).
