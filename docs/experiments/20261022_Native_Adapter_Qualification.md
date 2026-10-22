# Native adapter forward qualification — 22 October 2026

The existing memory/workspace adapter passed its first actual-dimension MiniCPM5 qualification on MPS with one **256-token** example. D1 and D2 null branches each matched **33,423,360 native reference logits with maximum absolute difference 0.0**, and a nonzero diagnostic sidecar export reloaded with the same exact agreement. Causal-input, parameter-scope and checkpoint-context checks also passed. This completes the measured forward/backward/serialization portion of [S2.1](../stages/03_Native_Architecture_Study.md); augmented autoregressive generation remains unimplemented.

Evidence: [complete qualification report](../../reports/minicpm5_native_adapter_qualification256_report.json), [raw memory samples and phase events](../../reports/minicpm5_native_adapter_qualification256_memory_samples.jsonl), [qualification design](native_adapter_design.md), and [capacity/readiness experiment record](../../reports/capacity_readiness_experiment_20261022.json). The report retains source/data/template/implementation hashes and the complete source seal. The native export stays under ignored `runs/`; the tracked report preserves its measured result.

## Workload and provenance

The run used the Mac Studio M2 Ultra with 64 GiB unified memory, FP32 resident parameters and compute, eager PyTorch MPS, seed 17 and no CPU fallback. Framework versions were PyTorch 2.14.1, Transformers 5.20.2 and safetensors 0.8.0. One backbone was resident as D1 and D2 wrappers were constructed sequentially; the original was released before loading the saved export.

| Item | Measured configuration |
|---|---|
| Native source | `openbmb/MiniCPM5-2B`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` |
| Native inventory | 2,516,756,480 parameters; 381 tensors; 42 layers; hidden size 2048; vocabulary 130,560 |
| Example | `capacity-256-dev-world-0` from `data/diagnostics/capacity_256/dev.jsonl` |
| Actual sequence / prompt / observed-memory lengths | 256 / 254 / 197 tokens |
| Supervised positions | 2: one answer token and EOS |
| Source family | `capacity-256-dev-xor-width-5-mask-2` |
| Adapter | Memory/workspace widths 64; four slots; two workspace steps; four heads; last-layer insertion |
| Additional parameters | 895,490 across 48 tensors |
| Template | Sealed source template; `enable_thinking=False`, generation prompt added |
| Native optimizer updates / generated answers | 0 / 0 |

The exact input-data SHA256 is `fd662d4d74f7eef5153e6a60789bc8ca6c1f904b1f23701435c8951836e21c8d`. The qualifier script SHA256 is `d1c2de74e5f7f75b979a4b5bf969200aca45b588953d9539e28918ff232a166f`; training-bound adapter/neural code was unchanged for this qualification.

To repeat in a fresh directory after other full-size jobs exit:

```bash
.venv/bin/python scripts/qualify_native_adapter.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --data data/diagnostics/capacity_256/dev.jsonl --split dev \
  --output runs/minicpm5_native_adapter_qualification256_next \
  --device mps --max-length 256
```

## Parity, causality and scope

Native reference logits were captured before any decoder wrapper was installed. Both residual output projections began exactly zero. Each augmented arm then matched every reference logit at `rtol=0`, `atol=0`. A deterministic nonzero workspace output projection changed logits, with maximum differences **0.219418** in D1 and **0.107325** in D2, demonstrating that the parity check could observe an active branch. The diagnostic perturbation was cleared before the backward checks.

Both regimes passed these causal controls:

- Changing future teacher-forced answer tokens left support/workspace context unchanged. Workspace construction used only the fixed 254-token prompt prefix; observed-memory construction used the separate 197-token support input.
- Supplying supervised target positions as workspace input was rejected.
- Changing the support writer rebuilt its source features; the previous context was not reused.
- Owned context survived backward with unchanged identity and tensor versions, and was explicitly released afterward.

The source observation/provenance claims remain those of the trusted synthetic fixture and its manifest. These tensor checks do not establish that arbitrary source text is authorized, true, or free of disguised oracle content.

| Scope/gradient check | D1: modules only | D2: full augmented |
|---|---:|---:|
| Original trainable parameters | 0 | 2,516,756,480 |
| Original gradient tensors / nonzero tensors | 0 / 0 | 381 / 381 |
| New trainable parameters | 895,490 | 895,490 |
| New gradient tensors / nonzero tensors | 48 / 2 | 48 / 2 |
| Missing eligible gradients / nonfinite gradient tensors | 0 / 0 | 0 / 0 |

Both new output projections received finite nonzero gradients. The other new tensors had gradient tensors but zero values on this initial null-branch backward, as expected when zero output projections block their upstream learning signal. This does not establish that those upstream tensors receive nonzero gradients after an optimizer update; that remains part of native D1/D2 training qualification.

For each arm, nonreentrant checkpointing invoked the **native original decoder block twice**, while the support writer and workspace encoder each ran **once**. The native block owns this checkpoint boundary; residual sidecars sit outside its recomputation. Context construction was not repeated and context identity/versions remained stable through backward. No optimizer step followed, and gradients were cleared after the check.

## Export/reload and measured memory

The final D2 serialization check deliberately enabled a nonzero workspace output projection, saved native base weights plus sidecars, released the original model, and reloaded the export. All **33,423,360 logits** matched the pre-export diagnostic output with **maximum absolute difference 0.0**. This probe makes sidecar serialization observable; the exported diagnostic artifact is not a trained developmental candidate.

| Counter | Recorded value |
|---|---:|
| Whole profiled process | 25.637546 s |
| Sampled peak live MPS allocation | 18.898608 GiB (20,292,225,536 bytes) |
| Sampled peak MPS driver allocation | 21.261551 GiB (22,829,416,448 bytes) |
| Process-lifetime peak RSS | 14.598755 GiB |
| Sampled peak RSS | 10.609726 GiB |
| Host swap occupancy, start → end | 1,562.69 → 1,562.69 MiB |
| Host memory-free endpoint percentage, start → end | 89% → 87% |
| Memory sampling target / recorded samples | 0.25 seconds / 72 |
| Profiler errors | 0 |

The 25.638-second duration includes source validation, load, reference/causal/probe forwards, both backward checks, export, model release and reload. It is not a training-throughput measurement. Samples can miss short-lived peaks; process RSS and MPS driver counters overlap in unified memory and must not be added. Unchanged swap endpoints do not measure intervening swap traffic.

No optimizer state or full D1/D2 update was allocated/performed, so these lower peaks do **not** qualify complete D2 training capacity. The separate D0 capacity runs do not substitute for that measurement either.

## Remaining gate

S2.1 remains partial until augmented inference reconstructs support and prompt context correctly and passes native generation checks. Native D1/D2 optimization, upstream gradient opening after updates, full training memory, restart and generated-answer behavior remain separate work. This result establishes actual-dimension interface compatibility on one synthetic 256-token example, with no claim of convergence, useful architectural gain, retention or Replica fidelity.
