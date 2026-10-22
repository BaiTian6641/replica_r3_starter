# Apple GPU and Neural Engine feasibility — 22 October 2026

**Current execution uses the M2 Ultra GPU. Neural Engine participation is unmeasured and not implemented in the project training stack.** The completed [256-token](../../reports/minicpm5_capacity256_chunk02_report.json) and [512-token](../../reports/minicpm5_capacity512_chunk02_report.json) native runs report `device=mps`, FP32 parameters/compute and `cpu_fallback_enabled=false`. The [augmented-generation qualification](../../reports/minicpm5_augmented_generation_qualification256_report.json) records the same backend. These are GPU model-execution results; no ANE result is claimed.

## Supported route and current boundary

[Apple's PyTorch guide](https://developer.apple.com/metal/pytorch/) identifies MPS as GPU acceleration through Metal Performance Shaders/MPS Graph. The current model, autograd and portable Muon 8-bit tensor operations use this route. Host tokenization, orchestration, logging and file audits remain ordinary CPU work; their presence does not move the model's training workload onto the CPU.

The [PyTorch device registry](https://docs.pytorch.org/cppdocs/api/c10/device.html) exposes MPS and custom-backend extension points, but no supported Apple Neural Engine device for this stack. There is no project ANE autograd, Muon momentum/update, native decoder or sidecar backend. Selecting `mps` does not also select the Neural Engine, and current hardware utilization cannot be described as GPU+ANE training.

Core ML offers a separate prediction path. Apple's [Core ML model API](https://apple.github.io/coremltools/source/coremltools.models.html) defines `ALL`, `CPU_AND_GPU`, and `CPU_AND_NE` as allowed compute-unit sets; `CPU_AND_NE` still permits CPU execution and excludes GPU execution. `ALL` permits all three. These choices do not prove that every operation runs on ANE. Apple's [performance-report documentation](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode?changes=l_9) explains how to inspect actual operation placement.

This is not a claim that Apple supports no on-device updates: [MLUpdateTask](https://developer.apple.com/documentation/coreml/mlupdatetask?changes=l_8) updates updatable Core ML models with training data. That capability does not establish a supported ANE implementation of this arbitrary PyTorch/Muon training graph or automatic joint GPU/ANE execution.

## Experimental ANE training: Orion

The [Orion preprint, version 1](https://arxiv.org/html/2603.06728v1) demonstrates experimental transformer training through private `_ANEClient`/`_ANECompiler` APIs. Its reported training model is 110M parameters, validated on **M4 Max only**. Sections 4.3–4.5 put transformer forward/activation-backward work on ANE while CPU handles weight-gradient accumulation, Adam, loss, embeddings and classifier backward. Section 9 lists hardware-validation and quantization limitations. The paper supplies no M2 Ultra MiniCPM5-2B or Muon 8-bit result.

**Engineering inference:** this is evidence that ANE training is possible in a research system, not a drop-in backend for our 2,516,756,480-parameter FP32 model and mixed Muon optimizer. Adopting it would require a separately qualified architecture/kernel, precision, optimizer, causal-mask and restart port. Its CPU/ANE division also fails a requirement that all model-training computation avoid CPU. No Orion/private-API installation or experiment was performed for this review.

## Smallest useful next benchmark

Keep current fine-tuning on MPS. Separately evaluate a frozen, bounded evidence encoder or preprocessing subgraph through public Core ML prediction on this exact M2 Ultra. Compare `CPU_AND_NE` and `CPU_AND_GPU`, record actual operation placement, compile/load cost, output parity against the existing MPS path, latency and memory. Use Apple's [Xcode performance report](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode?changes=l_9) to verify ANE dispatch before claiming acceleration. A useful inference result could support a later GPU-backbone/ANE-encoder pipeline; it would not establish an ANE gradient bridge or fine-tuning speedup.

This review used primary documentation and the cited preprint only. No packages, models or private APIs were installed or executed. Current GPU experiments and source seals were left unchanged.
