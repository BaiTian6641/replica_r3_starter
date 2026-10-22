# 06 - Primary sources and evidence limits

**Current native preflight review: 22 October 2026 (America/Chicago). Original bundle review label: 23 October 2026 (Asia/Shanghai).** This register distinguishes primary research, official implementation sources, project proposals and measurements made on the delivered starter. URLs with main are discovery links; execution must pin immutable revisions.

## 1. Research-to-design references

| ID | Primary source and code | What it supports here | Boundary |
|---|---|---|---|
| R01 | [Engram, arXiv:2601.07372](https://arxiv.org/abs/2601.07372); [authors' repository](https://github.com/deepseek-ai/Engram) | Sparse conditional memory lookup/fusion | Editable episode ledger is our separate design |
| R02 | [Titans, arXiv:2501.00663](https://arxiv.org/abs/2501.00663) | Neural memory that adapts during a stream | Linear reference state is not a reproduction |
| R03 | [Nested Learning, arXiv:2512.24695](https://arxiv.org/html/2512.24695v1); [Google explanation](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) | Explicit optimization levels and update timescales | Published pretrained adaptation is much larger than the proposed pilot |
| R04 | [uTTT, arXiv:2610.05484](https://arxiv.org/html/2610.05484v1); [authors' code](https://github.com/Zefan-Cai/uTTT) | State ownership across depth and chunk read/write ordering | Current language release is MoE-oriented, not a ready MiniCPM retrofit |
| R05 | [Mem-T, arXiv:2601.23014v2](https://arxiv.org/abs/2601.23014v2); [authors' code](https://github.com/yanweiyue/Mem-T) | Learning memory-operation policy and credit assignment | Official training script remains forthcoming |
| R06 | [SEAL, arXiv:2506.10943v2](https://arxiv.org/abs/2506.10943v2); [authors' code](https://github.com/Continual-Intelligence/SEAL) | Train useful self-edit proposals by their adaptation outcomes | Does not remove verification or training cost |
| R07 | [Absolute Zero, arXiv:2505.03335](https://arxiv.org/abs/2505.03335); [authors' code](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) | Executor-grounded self-proposed tasks | Pretrained knowledge and a structured verifier still matter |
| R08 | [Voyager, arXiv:2305.16291](https://arxiv.org/abs/2305.16291); [authors' code](https://github.com/MineDojo/Voyager) | Persistent reusable executable skills | A system-level gain is not necessarily neural-weight learning |
| R09 | [INTACT, arXiv:2610.19568](https://arxiv.org/html/2610.19568) | Frozen-decoder latent-interface comparison | Recent preprint; not uniformly better than full SFT |
| R10 | [Continuous Thought Machines](https://arxiv.org/abs/2505.05522); [Sakana code](https://github.com/SakanaAI/continuous-thought-machines) | Optional temporal-computation comparison | The shipped slot recurrence is not CTM synchronization |
| R11 | [Coconut](https://arxiv.org/abs/2412.06769); [Meta code](https://github.com/facebookresearch/coconut) | Optional learned latent reasoning comparison | A slot workspace is not a reproduced Coconut curriculum |
| R12 | [ReST, arXiv:2308.08998](https://arxiv.org/abs/2308.08998) | Simple generate/score/filter/train control | Fixed filtering is not learning to select useful practice |
| R13 | [Individual simulations grounded in self-reports, arXiv:2411.10109](https://arxiv.org/abs/2411.10109) | Source-grounded modeling and human repeatability reference | Bounded behavioral evaluation, not a consciousness result |
| R14 | [Persona vectors](https://www.anthropic.com/research/persona-vectors) | Possible drift diagnostics | Trait directions are not complete identity |
| R15 | [Memory of Amortized Contexts](https://arxiv.org/abs/2403.04317); [author code](https://github.com/jihoontack/MAC) | Learned context interfaces | Requires learned alignment, not arbitrary tensor interchange |
| R16 | [LongMem](https://arxiv.org/abs/2306.07174); [author code](https://github.com/Victorwz/LongMem) | Memory readers with frozen representations | Base/writer changes invalidate old representation assumptions |
| R17 | [Effective Quantization of Muon Optimizer States, arXiv:2509.23106](https://arxiv.org/abs/2509.23106) | Motivation for blockwise linear 8-bit Muon momentum | This project implements a portable experiment, not a reproduction of the paper's training results |

The full combined architecture is a project synthesis. A component's published success does not prove that combining all components will work. Unofficial Titans/HOPE implementations are not presented as author-maintained trainers in this starter.

## 2. Model sources

- [MiniCPM5-2B official model card](https://huggingface.co/openbmb/MiniCPM5-2B), [configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json), [OpenBMB repository](https://github.com/OpenBMB/MiniCPM).
- [MiniCPM-V-4.7-1B official card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [configuration](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/main/config.json).
- [MiniCPM-V-4.7-35B-A3B official card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B), [configuration](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json), [OpenBMB VLM repository](https://github.com/OpenBMB/MiniCPM-V).
- [Gemma 4 official card](https://ai.google.dev/gemma/docs/core/model_card_4), [26B-A4B-it checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it).

The MiniCPM5 total count is 2,516,756,480 from its sealed tensor inventory, not exactly 2B. The native MPS baseline and full D0 preflight instantiated the pinned pretrained revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`; their short fixture results do not establish broader pretrained quality. R3's earlier tiny-model tests use a randomly initialized small Llama. The V4.7 port remains planned.

## 3. Datasets

The source-by-source catalog and constraints appear in [the dataset guide](04_Datasets_and_Generation.md); machine-readable intake entries are in [dataset_registry.json](../configs/dataset_registry.json). Every entry begins unpinned and not downloaded. The guide covers:

- PersonaMem-v2 and v3.
- Cleaned LongMemEval and LongMemEval-V2.
- LoCoMo and LaMP.
- Tulu 3, OpenThoughts3 and UltraFeedback.
- TOUCAN and BFCL.
- CLEVR and a proposed temporal extension.

The observed license label applies to the inspected artifact, not automatically to every constituent source or model-generated output. Source-specific terms and exact splits are recorded before ingestion. No public corpus is redistributed in the bundle.

## 4. Implementation sources

| Topic | Primary source |
|---|---|
| Frozen parameters vs autograd recording | [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) |
| Recomputed forward behavior | [PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) |
| DDP memory semantics | [PyTorch DDP](https://docs.pytorch.org/docs/2.14/notes/ddp.html) |
| CPU optimizer offload | [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) |
| Muon algorithm and eligible tensors | [Official PyTorch Muon API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) |
| Current macOS backend | [PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html) |
| Later GPU hardware | [NVIDIA RTX PRO 4000 Blackwell SFF Edition specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/) |
| MiniCPM full-tuning integration | [OpenBMB SWIFT guide](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/ms_swift.md) |
| MLX unified memory | [Official MLX docs](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) |
| MLX-LM full-mode scope | [Official source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py) and [training guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| MLX V4.7 support | [MLX-VLM model docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_7/README.md) |
| NVIDIA native dtype differences | [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) |
| Engineering verification tools | [Verilator](https://github.com/verilator/verilator), [SymbiYosys](https://github.com/YosysHQ/sby) |

These repositories are references, not all installed or integrated dependencies. DeepSpeed, MLX, VLM, formal verification and large-model ports were not run here.

The selected project optimizer is portable eager PyTorch Muon 8-bit on macOS MPS first and CUDA later: linear INT8 momentum in 2048-element blocks, FP32 scales/dequantization/Newton-Schulz temporaries, and FP32 parameters and gradients in the initial path. Trainable embeddings, output head, learned slot queries, 1D and other ineligible tensors use auxiliary FP32-state AdamW; an all-AdamW run remains an explicit comparison. One optimizer owns the mixed groups, covering each intended trainable tensor exactly once without changing the D0/D1/D2 scope. This is a project implementation choice; stock PyTorch Muon and quantized AdamW packages do not establish equivalent 8-bit Muon behavior. Source-reported convergence and state savings do not guarantee this project's quality, speed or full-model fit.

## 5. Evidence made in this bundle

Historical bundle evidence is recorded in [validation.json](../reports/validation.json) and [test output](../reports/test_output.txt); preserve their recorded dates, counts and runtime versions. Current workstation setup evidence is recorded separately in the [runbook](03_Implementation_and_Runbook.md) and setup reports. Tests cover record/identity/data contracts, temporal correction and deletion, numerical null paths, gradients, causal prompt support, state isolation, native tiny-Llama integration, parameter scope and serialization. [The deterministic demo report](../reports/contract_demo.json) covers software publication/restart/rejection/rollback.

The initial Mac setup adds [70 passing tests](../reports/validation_macos_setup.json), [tiny MPS D0/D1/D2 updates with Muon 8-bit and export/reload](../reports/muon8bit_mps_smoke.json), and a [sealed checkpoint tensor-header inventory](../reports/minicpm5_snapshot_inventory.json). The optimizer tests include stock AdamW/Muon comparisons, independent SVD direction checks, quantization error, exact routing and INT8 state continuation; they do not reproduce paper convergence results.

The subsequent [native MiniCPM5 experiment record](../reports/minicpm5_mps_experiment_20261022.json) contains a [frozen baseline](../reports/minicpm5_mps_baseline_report.json) and [two-update full D0 Muon 8-bit preflight](../reports/minicpm5_mps_preflight_report.json) on the M2 Ultra / 64 GiB Mac using FP32 parameters/compute and MPS without CPU fallback. All 2,516,756,480 original parameters were trainable in D0; all 381 tensors had nonzero first-update gradients. The [saved-checkpoint audit](../reports/minicpm5_mps_preflight_checkpoint_audit.json) independently found changes in every original tensor, including embeddings/head, and matching native names/shapes. Export/reload produced a maximum absolute logit difference of 0.0 over 7,180,800 values from one development example.

On 16 deterministic development fixtures, response-only NLL was **15.011792898** before tuning and **1.572720953** afterward. The preflight processed **110 tokens**, with only **four supervised target tokens**, in **47.764 seconds** across two updates (**2.303 processed tokens/second**). Its configured length ceiling was 256, but every fixture was actually **55 tokens**; this does not test 256-token activation memory. This NLL result is an infrastructure check on a tiny known grammar, not a held-out task, generalization, retention or autonomous-learning result.

Matched greedy generation used the same 16 examples, identical 53-token prompts and source chat template with thinking disabled, cached decoding and a **256-new-token cap**. The [baseline](../reports/minicpm5_mps_generation_baseline_256_report.json) scored **0/16** strict matches, with eight capped responses and eight EOS endings. The [D0 export](../reports/minicpm5_mps_generation_D0_256_report.json) scored **2/16**, but all 16 completions were the constant **`0` followed by EOS**, two tokens each. This negative finding is answer collapse, not learned XOR. It prevents treating the NLL drop or raw accuracy change as independent task improvement; the behavior is consistent with learning response formatting/EOS and a constant answer.

At 0.25-second sampling, baseline peak MPS allocated/driver memory was **9.41 / 10.07 GiB**, versus **24.74 / 29.15 GiB** for D0. D0's process-lifetime peak RSS was **14.52 GiB** and overlaps MPS unified-memory counters; these values must not be added. Sampling may miss transient peaks. Host swap was **1,562.69 MiB** at both run endpoints; this does not establish that no swap activity occurred between them. The detailed phase profile, dtype/state bytes, hashes, environment and measurement limits are retained in the reports and [hardware guide](02_Stages_Experiments_and_Hardware.md).

The following remain **unrun**:

- Native pretrained MiniCPM5 D1/D2 augmented training and broader task-quality evaluation.
- CUDA full fine-tuning, sharding or optimizer offload.
- Representative long-sequence or sustained native MPS full training, complete optimizer/scheduler resume and MLX parity measurements.
- Actual V4.7 multimodal integration and full-VLM gradient tests.
- A learned autonomous curriculum/director operating through multiple neural updates.
- Ten-cycle model learning, broad transfer or source-person fidelity.
- Large Gemma/MiniCPM-V ports.

The artifact validator checks JSON, source hashes, file presence and local links. It does not validate an AI architecture.

## 6. Project inheritance and original files

Seven supplied R2 Markdown files are copied without edits under reference/r2. Their content hashes are in source_manifest.json. They define the inherited full-chain scope and identity protocol. References inside those historical files may name artifacts that were not attached; those absent artifacts are not silently recreated or asserted tested.

R3 supersedes practical handoff details where needed: actual executable scripts now exist; the dataset catalog is freshly checked; the Mac has a defined role; the smaller VLM is a distinct architectural port; publication and deletion behavior have executable checks. It does not revise the user's central objective or convert the project into a static persona.

Memory estimates, learning rates, data mixtures, cycle counts and thresholds in the plan are **PROPOSED** or explicitly labelled arithmetic. Test counts and observed numerical invariants are **MEASURED HERE**, with the report's recorded date and scope. Paper findings are **REPORTED BY THEIR AUTHORS** within their protocols. Optimizer selection and the later Blackwell experiment are **USER CONTEXT**; the current Apple M2 Ultra arm64 host, 24 CPU cores and 64 GiB unified memory are confirmed by the local hardware inventory. NVIDIA's 24 GB GDDR7 specification describes the planned RTX PRO 4000 Blackwell SFF Edition, not an installed CUDA device.
