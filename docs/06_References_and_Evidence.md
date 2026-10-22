# 06 - Primary sources and evidence limits

**Reviewed 23 October 2026 (Asia/Shanghai).** This register distinguishes primary research, official implementation sources, project proposals and measurements made on the delivered starter. URLs with main are discovery links; execution must pin immutable revisions.

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

The full combined architecture is a project synthesis. A component's published success does not prove that combining all components will work. Unofficial Titans/HOPE implementations are not presented as author-maintained trainers in this starter.

## 2. Model sources

- [MiniCPM5-2B official model card](https://huggingface.co/openbmb/MiniCPM5-2B), [configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json), [OpenBMB repository](https://github.com/OpenBMB/MiniCPM).
- [MiniCPM-V-4.7-1B official card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [configuration](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/main/config.json).
- [MiniCPM-V-4.7-35B-A3B official card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B), [configuration](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json), [OpenBMB VLM repository](https://github.com/OpenBMB/MiniCPM-V).
- [Gemma 4 official card](https://ai.google.dev/gemma/docs/core/model_card_4), [26B-A4B-it checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it).

The MiniCPM5 total count used in memory arithmetic is 2,516,756,480 from its card/config-compatible inventory, not exactly 2B. R3's tiny-model tests use a randomly initialized small Llama and do not verify pretrained weight quality. The V4.7 port remains planned.

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
| MiniCPM full-tuning integration | [OpenBMB SWIFT guide](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/ms_swift.md) |
| MLX unified memory | [Official MLX docs](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) |
| MLX-LM full-mode scope | [Official source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py) and [training guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| MLX V4.7 support | [MLX-VLM model docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_7/README.md) |
| NVIDIA native dtype differences | [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) |
| Engineering verification tools | [Verilator](https://github.com/verilator/verilator), [SymbiYosys](https://github.com/YosysHQ/sby) |

These repositories are references, not all installed or integrated dependencies. DeepSpeed, MLX, VLM, formal verification and large-model ports were not run here.

## 5. Evidence made in this bundle

See [validation.json](../reports/validation.json) and [test output](../reports/test_output.txt) for exact counts and runtime versions. Tests cover record/identity/data contracts, temporal correction and deletion, numerical null paths, gradients, causal prompt support, state isolation, native tiny-Llama integration, parameter scope and serialization. [The deterministic demo report](../reports/contract_demo.json) covers software publication/restart/rejection/rollback.

The following remain **unrun**:

- Native pretrained MiniCPM5 loading/training and task-quality evaluation.
- CUDA full fine-tuning, sharding or optimizer offload.
- M2 Ultra / MLX / MPS measurements.
- Actual V4.7 multimodal integration and full-VLM gradient tests.
- A learned autonomous curriculum/director operating through multiple neural updates.
- Ten-cycle model learning, broad transfer or source-person fidelity.
- Large Gemma/MiniCPM-V ports.

The artifact validator checks JSON, source hashes, file presence and local links. It does not validate an AI architecture.

## 6. Project inheritance and original files

Seven supplied R2 Markdown files are copied without edits under reference/r2. Their content hashes are in source_manifest.json. They define the inherited full-chain scope and identity protocol. References inside those historical files may name artifacts that were not attached; those absent artifacts are not silently recreated or asserted tested.

R3 supersedes practical handoff details where needed: actual executable scripts now exist; the dataset catalog is freshly checked; the Mac has a defined role; the smaller VLM is a distinct architectural port; publication and deletion behavior have executable checks. It does not revise the user's central objective or convert the project into a static persona.

Memory sizes, learning rates, data mixtures, cycle counts and thresholds in the plan are **PROPOSED**. Test counts and observed numerical invariants are **MEASURED HERE**. Paper findings are **REPORTED BY THEIR AUTHORS** within their protocols. User goals and hardware statements are **USER CONTEXT**.
