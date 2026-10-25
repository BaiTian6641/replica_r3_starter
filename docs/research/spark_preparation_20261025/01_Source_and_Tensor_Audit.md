# Source and tensor audit

**Run:** `reports/spark_preparation_20261025/run_01/`
**Manifest:** `preparation_manifest.json`, fingerprint
`1142605a342ca0ec96548bb75553a92a073f71f50a1020f893664e02ec1427b3`
**Policy:** no weight body, no remote code execution, no native model load.

## 1. Immutable pins

| Role | Repository | Commit | Hub API bytes / SHA256 |
|---|---|---|---|
| Adaptation base | `XHToken/Spark-X2.6-4B-FP8` | `0318d6f6ab070c7b5b43052db571489b2508006c` | 16,624 / `3f3e0d5e07ed1654f285014f3837fb9563d344e1917c92204647049b2adbcfe6` |
| Independent control | `XHToken/Spark-X2.6-4B` | `66e3ae7bee54e0a523d692cb8ef841f35409807b` | 16,274 / `af88d32cb15392e976586fb8174ff3869894f98c8c534912b57e5b3adea2876d` |

The Hub API response at each pinned revision was itself retained under
`metadata/<role>_hub.json`. Its `sha` field equals the requested commit, its
`cardData.license` is `apache-2.0`, and its sibling sizes were checked against
the plan before any asset or header request.

## 2. Fetched text assets

Both repositories expose the same bytes for the tokenizer, special-token map,
generation config, chat template, LICENSE and both custom Python files. `config.json`
and `README.md` differ.

| Asset | Bytes | SHA256 | Retained |
|---|---:|---|---|
| `config.json` (FP8) | 4,510 | `2f0a76065de1488dbb1576389e7cd37e28ac77c52fc84f522a5ce4a824da9fb2` | yes |
| `config.json` (BF16) | 2,035 | `767161ece8ed2891e44345afdf98d29c1cacb964116cb6db7d20dc1c757490ad` | yes |
| `LICENSE` (both) | 11,338 | `525c465ab952779d2d360a9ff36f7f49ae018a23d80302ff839a850a77784c36` | yes |
| `README.md` (FP8) | 14,122 | `570208eb4a1c3022a58bc277325f5b29f393b55fc3096d373893038cbe27b0f0` | yes |
| `README.md` (BF16) | 14,127 | `c483630bf7e9c22a6583d127d74ecf0e3270aea0c358e770ac5b5f4dbb82bf39` | yes |
| `chat_template.jinja` (both) | 5,717 | `480576bd2ef80726a0dba6406d72ca37f4f4f9ca747c7f1c8d159dbdf40f6929` | yes |
| `configuration_spark.py` (both) | 4,329 | `02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722` | yes |
| `modeling_spark.py` (both) | 19,418 | `9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4` | yes |
| `tokenizer_config.json` (both) | 5,841 | `57b385dcc2ad031aef4a3786e97cd9a832ecb839cb27ed2cd829d627c90e1bfd` | yes |
| `special_tokens_map.json` (both) | 156 | `48a257e3f5f20a004e7c30482a70ae19ae7d9cf713144f2a1394ba20af8c3cb5` | yes |
| `generation_config.json` (both) | 281 | `804a5276a9d237600688267dd2bf6f9d4129ec21faf417f870d5d52b32ad9e6c` | yes |
| `tokenizer.json` (both) | 10,115,786 | `710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c` | no, hashed |
| `vocab.json` (both) | 3,032,795 | `9f42cb7e7af5afe27cae2337d272d60b5659e10143ef4bb952f642b847916fe1` | no, hashed |
| `merges.txt` (both) | 1,567,883 | `b06fc88940ae6a20f71142f06304c12547f58ab52c175e56af12a38c3dc62f36` | no, hashed |
| `model.safetensors.index.json` (BF16) | 24,133 | `77a11d49b279b5e44fcb1f2453f71a8023237f93992fe5275d842e149d896edd` | yes |

The large tokenizer files were bounded to 16 MiB each and downloaded only to
compute their provenance; they are not weights and are not retained in the
report tree. Only retained assets have a `retained_path`; every fetched asset has
a provenance record whether retained or not.

### Config observation

Both configs resolve to the same topology: `Spark2_5ForCausalLM` /
`spark2_5`, 36 layers, hidden 2560, MLP 10240, 16 query / 4 KV heads, head
dimension 256, vocabulary 131,072, sliding window 512, tied embeddings, RoPE
full `0.25 @ 5,000,000` and sliding `1.0 @ 10,000`, layer types repeating
`sliding, sliding, sliding, full`, and `auto_map` to the two custom files. The
serialized Transformers version is FP8 4.57.6 and BF16 4.57.1.

The FP8 config declares compressed-tensors 0.16.0, float-quantized storage,
static symmetric channel-wise 8-bit weights, dynamic symmetric per-token 8-bit
input activations, no output activation quantization, no zero points, null KV
cache scheme, one config group, and a quantized-target regex for
`q_k_v_proj`, `out_proj` and the three MLP projections. Its ignore list is
exactly every `self_attn.g_proj` plus `lm_head`.

Tokenizer config: `PreTrainedTokenizerFast`, `model_max_length=131072`,
`add_bos_token=false`, `add_eos_token=false`, embedded chat template SHA256
`93d6aa906ec72365efcab3c0109dba83148bdc9bec7c55848623d17483693eda`.
The template supports `enable_thinking`, `reasoning_effort` in `high|low`,
`<think>`/`</think>`, tool calls and an error when messages are empty. The
embedded copy is a whitespace-minified equivalent of the formatted standalone
`chat_template.jinja`; the two serializations therefore have different hashes
even though the same markers and control flow were verified in both.

## 3. Safetensors header evidence

Every shard was inspected with two range requests: bytes `0-7` for the
little-endian header length, then exactly that many header bytes. All responses
were HTTP 206 with an exact `Content-Range`; any HTTP 200 would have been
refused and recorded without reading the body. The tensors' `data_offsets` were
required to cover the shard payload exactly, without overlaps or gaps.

### FP8 — single shard

| Property | Value |
|---|---|
| File | `model.safetensors` |
| Total bytes | 4,451,626,120 |
| Header bytes / SHA256 | 55,424 / `18d82da0cb0276bfc3e1d3fc9655d7f0c71ffbd021bd5da5c2261f7fd416840a` |
| Tensor payload | 4,451,570,688 |
| Tensors | 470 |
| F8_E4M3 | 180 tensors, 3,774,873,600 elements / bytes |
| BF16 | 290 tensors, 338,348,544 elements / 676,697,088 bytes |
| BF16 weights excluding scales | 337,205,760 |
| BF16 scale elements | 1,142,784 |
| Zero points | 0 |
| Serialized `lm_head` | absent; tied to `model.embedding.weight` |

The arithmetic is exact: `3,774,873,600 + 2 × 337,205,760 + 2 × 1,142,784 +
8 + 55,424 = 4,451,626,120`. The 180 scales are BF16 tensors shaped
`[output_channels, 1]`; for example, `q_k_v_proj` has 6,144 rows and a
`[6144, 1]` scale, `out_proj` has 2,560 output channels and `[2560, 1]`, the
MLP gate/up have `[10240, 1]`, and down has `[2560, 1]`. There is no
`weight_zero_point` tensor. Hub metadata reports the same quantized/BF16 split
but excludes scales; the cross-check is recorded, while the header is the
tensor-level evidence.

### BF16 — five shards

| Shard | File bytes | Header bytes | Header SHA256 | Tensors | Payload bytes |
|---|---:|---:|---|---:|---:|
| 1 | 1,982,449,560 | 6,032 | `0463306856da80ba462c3d7819f3a641d66318ff68c5507a8f8a374b94a2e589` | 52 | 1,982,443,520 |
| 2 | 1,993,132,424 | 8,576 | `c81713994971dfc9eb3eb51b1e09f402342a71c189d729851cec313ca2ed44d0` | 74 | 1,993,123,840 |
| 3 | 1,993,225,080 | 9,072 | `d4a74ace373171c350a23278b2520ba3681e25d7c1e24c6876edb9dc7041f2b9` | 78 | 1,993,216,000 |
| 4 | 1,993,132,448 | 8,600 | `b8a902c2f65ea28963a84681705830b3d714af7091718783464c3af1b6921f2c` | 74 | 1,993,123,840 |
| 5 | 262,252,896 | 1,368 | `a63468027cc1dd6f895bfdc758e3e0095a45fd97298c92f86f41576699decc10` | 12 | 262,251,520 |
| total | 8,224,192,408 | 33,688 | — | 290 | 8,224,158,720 |

All 290 tensors are BF16. The published index maps every name to the header's
shard, declares 4,112,079,360 parameters and 8,224,158,720 tensor bytes, and
matches both header aggregates. No tensor name appears in more than one shard.
The full per-tensor name/dtype/shape/offset list is retained as
`headers/<role>/tensor_index.json`.

### Tied storage and aliases

Neither header contains `lm_head.weight`. The config declares
`tie_word_embeddings=true`; the custom model declares
`_tied_weights_keys = ["lm_head.weight"]` and computes logits with
`F.linear(hidden_states, self.model.embedding.weight)` when tied. The 131,072 ×
2,560 `model.embedding.weight` is therefore the only serialized vocabulary
matrix. This is a storage/alias conclusion from header + config + code, not a
loader test.

## 4. Custom Python review

`modeling_spark.py` and `configuration_spark.py` were parsed with `ast`. They were
not imported, compiled to bytecode with execution, or run. No top-level network,
filesystem or subprocess action was found. Import-time calls are limited to
`logging.get_logger(__name__)` and `ALL_LAYERNORM_LAYERS.append(Spark2_5RMSNorm)`.

Observed structure and semantics:

* `Spark2_5Config` subclasses `PretrainedConfig`; it validates that the layer
  type list matches the layer count and derives `head_dim` only when absent.
  The released config supplies `head_dim=256`, which is not
  `hidden_size // heads` (160); adapters must not recompute it.
* `Spark2_5Attention` uses one fused `q_k_v_proj` of width
  `16×256 + 2×4×256 = 6144`. Q/K/V are sliced from that projection. There is
  no separate q/k/v module and no attention bias.
* The headwise output gate is `g_proj: [16, 2560]`, activated by sigmoid, shaped
  per head and multiplied into the attention output before `out_proj`. The FP8
  release leaves it BF16.
* Attention itself is eager matrix multiplication with float32 softmax and
  scaling `1/sqrt(256)`. No `flash_attn`, `causal_conv1d`, Triton or SDPA
  dispatch appears in the source.
* `Spark2_5MLP` is `down_proj(gelu(gate_proj(x)) * up_proj(x))`.
* RoPE is computed from `cache_position` (absolute positions). Sliding layers
  rotate all 256 dimensions at theta 10,000; full layers rotate 64 dimensions
  at theta 5,000,000 and pass the other 192 through.
* Masks are selected per layer type. The shared cache is a
  `transformers.cache_utils.DynamicCache`; the code does not evict keys outside
  the 512-token sliding window. The window is a mask, not bounded cache
  memory.
* `Spark2_5Model.forward` casts input embeddings to float32, then each layer
  casts to its MLP gate-projection dtype. Residual stream and layer compute
  therefore have mixed precision bookkeeping that a plain BF16 adapter would
  misrepresent.
* Gradient checkpointing calls the layer without `past_key_values`; the model
  itself disables cache use while checkpointing.
* Generation is inherited from `GenerationMixin`; there is no custom
  `generate` override. Thinking control is entirely in the chat template:
  `enable_thinking=false` = none, `enable_thinking=true` with
  `reasoning_effort=low|high` = low/high, defaulting to high.
* The code imports `transformers.masking_utils`, `TransformersKwargs`,
  `processing_utils.Unpack`, `generation.GenerationMixin` and
  `cache_utils.DynamicCache`, which ties it to the 4.57-era API, not an
  arbitrary Transformers version.

Backend dependencies are therefore `torch`, a 4.57.x Transformers, safetensors,
and compressed-tensors 0.16.0 for the FP8 checkpoint. No CUDA-only package is
imported. That absence is not evidence that MPS FP8 kernels exist.
