# Spark-X2.6-4B: official FP8 adaptation base and BF16 comparison

**Source review: 24 October 2026; design updated 25 October; contract reconciled
26 October. Status: preparation only, native FP8 blocked.** MiniCPM5
remains the baseline and first vision implementation. Spark metadata and
tokenizer-payload preparation exist (`run_02`/`run_03`); no weights, native
loader, adapter, gradient or measured capability is admitted. Native loading,
training and vision transfer require their own qualifications. Completion of
MiniCPM5 RSI is not a prerequisite for preparing or later qualifying Spark.
Spark does not block the MiniCPM5 work. This document authorizes no native
execution. The state, routing/compilation and RSI gate contracts that Spark must
inherit are in
[document 22](22_State_Mechanism_and_RSI_Contracts.md).

This revision fetched public model metadata, configuration and model cards,
not weights. Machine-readable pins and content hashes are in the
[source record](../research/20261024_Spark_Source_Record.json). No local Spark
checkpoint, native adapter, FP8 training path or measured capability is created
by this document.

## 1. Verified official sources

| Role | Repository | Immutable revision |
|---|---|---|
| Requested adaptation base | [XHToken/Spark-X2.6-4B-FP8](https://huggingface.co/XHToken/Spark-X2.6-4B-FP8) | `0318d6f6ab070c7b5b43052db571489b2508006c` |
| Independent BF16 control | [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) | `66e3ae7bee54e0a523d692cb8ef841f35409807b` |
| Upstream implementation | [Spark-X2.6](https://github.com/XHToken/Spark-X2.6) | Pin source commit during adapter admission; not pinned by this metadata review |
| Apple inference reference | [Spark-MLX-LLM](https://github.com/XHToken/Spark-MLX-LLM) | Separate backend; pin and qualify before reuse |

Both model cards declare Apache-2.0. The FP8 card names the BF16 release as its
parent. These are post-trained releases used as the **base for adaptation**;
they are not the separately named pretraining `Spark-X2.6-4B-Base` artifact.
Audit remote Python code and pin its files before execution.

The pinned [FP8 config](https://huggingface.co/XHToken/Spark-X2.6-4B-FP8/blob/0318d6f6ab070c7b5b43052db571489b2508006c/config.json)
and [BF16 config](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/config.json)
share this topology:

| Property | Inspected configuration |
|---|---|
| Architecture / model type | `Spark2_5ForCausalLM` / `spark2_5`, custom auto-map |
| Decoder depth / hidden / MLP width | 36 / 2560 / 10240 |
| Query heads / KV heads / head dimension | 16 / 4 / 256 |
| Vocabulary / tied embeddings | 131072 / true |
| Attention schedule | Three sliding blocks then one full-attention block, repeated |
| Sliding window | 512 tokens |
| RoPE | Full attention: partial factor 0.25, theta 5,000,000; sliding: full rotary, theta 10,000 |
| Configured maximum positions | 1,048,576; configuration value, not locally qualified context capacity |
| Serialized Transformers version | FP8 4.57.6; BF16 4.57.1; not a compatibility guarantee |

Pin the tokenizer and chat template, including the selected `none`/`low`/`high`
thinking mode. Match budgets and report actual reasoning tokens. Do not reuse
the smaller model's configuration or assume head dimension equals hidden width
divided by the number of attention heads.

## 2. What the FP8 checkpoint contains

Its configuration uses `compressed-tensors` 0.16.0, `float-quantized` format,
8-bit floating weights with static channel-wise quantization and dynamic
per-token activation quantization. Targets include fused Q/K/V, output and MLP
projections; attention gates and `lm_head` are excluded. The KV-cache quantization
scheme is null. The top-level `dtype=bfloat16` does not mean every weight is BF16.

Hub metadata reports 3,774,873,600 F8_E4M3 parameters and 337,205,760 BF16
parameters, totalling 4,112,079,360. The nominal parameter-byte calculation is
4,449,285,120 bytes before scales, metadata, buffers and runtime overhead. This
is not a measured download size, training footprint or MPS fit claim. Inventory
actual shard headers and scale tensors before loading; metadata alone does not
prove tensor-by-tensor coverage.

Treat three precision concepts separately: checkpoint storage, forward/backward
compute and optimizer state. Historical Muon 8-bit momentum affects only its
supported optimizer state and is not a current training recommendation; new
training, if any, routes through `NativeFusedAdamW`. No optimizer supplies FP8
kernels by itself, and no precision label makes every parameter or gradient low
precision.

## 3. Supported paths to qualify

| Path | Intended use | Admission evidence | Current status |
|---|---|---|---|
| Official FP8 with its activation scheme | Requested inference/adaptation base | Correct loader, scale semantics, supported forward/backward kernels and matched output tests | Unqualified locally |
| Frozen official FP8 + BF16/FP32 projector/LoRA | First training hypothesis | Input gradients through quantized layers, finite named updates, save/reload and measured residency | Proposed; backend capability must be demonstrated |
| Explicit dequantized derivative of official FP8 | Fallback experiment if supported kernels are unavailable | Reconstruct from stored quantized values/scales; hash new artifact; document compute and activation changes | Proposed, not the original BF16 release |
| Official BF16 release | Independent quality/optimizer control | Dedicated adapter, native parity and real update/reload | Proposed |
| Native FP8 weight updates | Later full or partial tuning experiment | Training kernels, master/scaling policy, numerical stability and complete optimizer support | Deferred |
| MLX inference | Possible platform reference | Pin implementation and qualify the exact checkpoint/format on this host | Not PyTorch/MPS training evidence |

Dequantizing FP8 does not recover precision lost during quantization. A derivative
that stops quantizing activations changes forward semantics and needs a separate
identifier and comparison. INT4/INT8 affine conversion recipes do not establish
FP8 compatibility. Apple generation-specific kernels do not establish support
on the present M2 Ultra. Unsupported paths remain blocked rather than silently
falling back to CPU while claiming native training.

## 4. Preparation now: S0

Prepare these items without allocating a native worker:

1. Preserve the official metadata pins/hashes and license records. At acquisition,
   record every shard digest, config, tokenizer, template and remote-code revision.
2. Inventory tensor names, dtypes, scales, ignored modules, tied storage and loader
   expectations from shard headers. Avoid full downloads merely to inspect names.
3. Review custom attention, head gates, positions, sliding/full cache and generation
   interfaces against the intended backend; identify required kernel packages.
4. Define an isolated environment if the locked MiniCPM5 environment is incompatible.
   Do not upgrade the working baseline environment to satisfy an upstream quickstart.
5. Write a capability matrix for forward, input gradients, parameter gradients,
   optimizer membership, export and reload. Mark unknowns explicitly.
6. Prepare source-independent prompts, data manifests and checks shared with
   MiniCPM5, with bilingual retention and fixed thinking/token budgets.

S0 produces an acquisition manifest, compatibility assessment, proposed adapter
ABI and preregistered qualification protocol. It does not declare FP8 tuning
supported merely because configuration parsing succeeds. Large weight/data
acquisition follows the bounded source and disk plan, separately from metadata.

## 5. Native and training qualification: S1–S2

Start after the remaining repair package and after the reusable MiniCPM5 V0
interface is sufficiently stable. There is no requirement to wait for an RSI
success. Share the single native worker by admission, not simultaneous launches.

S1 establishes unmodified Spark inference first. Compare FP8 and official BF16
on identical prompts, source information, template mode and generation limits.
Report exact token differences, task accuracy, abstention, latency and memory.
Do not demand bitwise equality across precision schemes; preregister tolerances
and quality budgets. Test long-prefix behavior within a measured small budget
before extrapolating toward the configured maximum context.

S2 qualifies one actual candidate update. A frozen quantized decoder still needs
input gradients to train its projector. `no_grad()` around that decoder breaks
the proposed alignment path. Trace gradients into new modules, prove original
weight immutability for frozen-base tuning, check finite loss/gradients, then
export and reload in a fresh process with the same precision semantics.

Route any new candidate training through `NativeFusedAdamW` — native
`torch.optim.AdamW(fused=True)` with strict FP32 ownership, finite-gradient and
resume checks, and no fallback. Do not open a new Muon arm, and do not use
archived Muon results as a matched control for new runs. Current FP32 trainable
parameter/gradient residency must be measured, not inferred from the BF16 or FP8
source label. Record temporary dequantization/master copies.

## 6. Vision and new memory layers: S3

Reuse the accepted encoder/processor and source/evaluation contracts where
compatible. Train a **new projector to width 2560** and Spark-specific adapters.
Do not transplant the width-2048 MiniCPM5 connector, memory reader matrices or
serialized fast states. Re-encode eligible source memories under the new
producer and retain their lineage.

Implement an explicit `spark2_5` adapter, not a relaxed Llama adapter. Candidate
memory port positions must follow actual complete blocks and attention behavior.
Full-attention blocks such as 11/23 are hypotheses to test, not committed choices
or a copy of MiniCPM5's 13/27 convention. Keep ports disabled for native parity.

Test image-prefix visibility after more than 512 intervening tokens, sensitivity
to visual-token placement, full/sliding cache behavior and generation/restart.
A visual prefix disappearing from local attention may still affect full blocks;
measure this rather than assuming either complete loss or complete preservation.

Repeat V0 → alignment → instructions → visual memory → bounded improvement for
Spark's own accepted interfaces. Share the curriculum and cost accounting;
retraining model-specific projections and requalifying retention are mandatory.
Any gain must beat the matched Spark base and useful practical controls, not
only an earlier rejected MiniCPM5 memory candidate.

## 7. Resources and go/no-go artifacts

Keep the 50 GiB driver limit, sustained host checks, fallback-disabled execution
and exclusive worker from the [integrated design](19_Integrated_Memory_Learning_and_RSI_Design.md).
Profile text inference, visual prefill, backward and optimizer step separately.
Model weight bytes and the prior MiniCPM5 11 GiB run are not Spark fit evidence.

| Gate | Required artifact | Current status (26 October 2026) | Failure action |
|---|---|---|---|
| S0 source | Pinned artifact inventory and code review | Metadata plus tokenizer payloads pinned (`run_01`–`run_03`); shard/tensor inventory and custom-code review incomplete | Retain preparation status; no native execution with an unknown source |
| S1 inference | Native report plus FP8/BF16 comparison | Unrun; native FP8 loading and kernels blocked | Diagnose loader/backend; preserve output evidence |
| S2 training | Gradient/update manifest and reload parity | Unrun; no backend gradient or optimizer qualification | Choose an explicitly named derivative/control or defer; no hidden format switch |
| S3 visual | New connector, native V0 checks, measured grounding/retention | Unrun; depends on S1/S2 and accepted perception | Reject or refine that scope; no automatic backbone promotion |
| Memory/RSI | Correction, retention, restart and later-task gain | Unrun; MiniCPM memory, causal-refresh and visual quality remain rejected; no RSI gate passed | Preserve failure and useful baseline; reuse only qualified mechanisms |

Native FP8 remains **blocked**: configuration parsing and tokenizer provisioning
do not admit a loader, supported kernels or gradients. The rejected MiniCPM
visual/causal quality and the zero-update memory candidate are not repaired by
Spark preparation. No native work is authorized by this document, and no Spark
run is scheduled here. Any future S1/S2 result additionally reports the state
owners, version keys, reset classes and cost ledger defined in
[document 22](22_State_Mechanism_and_RSI_Contracts.md).

The final recommendation between MiniCPM5 and Spark requires measured quality,
retention, latency and training cost. The official FP8 artifact is now a concrete
planned base, not a promise of inexpensive or supported local fine-tuning.
