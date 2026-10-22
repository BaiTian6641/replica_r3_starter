# 03 - Construction plan and executable runbook

**Status:** research starter, not a trained persistent assistant. Read [README](../README.md), [system design](01_System_Design.md) and [experiment plan](02_Stages_Experiments_and_Hardware.md) first.

**Local setup and native preflight, 22 October 2026 (America/Chicago):** Mac Studio M2 Ultra, 24 CPU cores, 60 GPU cores, 64 GiB unified memory, macOS arm64. PyTorch MPS is available. Muon 8-bit is the selected experiment on this machine and later RTX PRO 4000 Blackwell SFF; FP32-state AdamW remains an explicit comparison. See [hardware inventory](../reports/hardware_macos_m2_ultra.json), [setup validation](../reports/validation_macos_setup.json), [native preflight results](../reports/minicpm5_mps_experiment_20261022.json), [profiles](../configs/hardware_profiles.json) and [execution state](../execution_state.md). The native baseline and two-update plain full-tuning D0 preflight now ran successfully; native D1/D2 and longer-context trials remain unrun.

## 1. What the delivered code actually does

| Component | Delivered behavior | Important boundary |
|---|---|---|
| contracts.py | Typed source/branch records; exact SFT schema; lineage/exact-duplicate split audit | Cannot prove factual correctness or user consent |
| memory.py | SQLite records, scope checks, observation-time reads, corrections, tombstones, derived-artifact invalidation | Not a semantic retriever, encrypted vault or OS security boundary |
| learning.py | Paired candidate screening; accept/reject/inconclusive; bound manifest publication; rollback | One trusted writer/process; evaluator authentication remains external |
| neural.py | Learned record pooling, residual reader, recurrent prompt workspace, functional linear fast state | Simplified project modules, not a reproduced research architecture |
| model_adapter.py | One complete native Llama block wrapper; explicit trainable scope; safe base/sidecar export | Training-only text adapter; no native generation cache, VLM or distributed support |
| train_text.py | Pinned local MiniCPM5 runner; response-only loss; four regimes; memory/timing and parameter/gradient reports; optional export/reload verification | Native baseline and two-update D0 measured; no offload or long-run resume |
| evaluate_text.py | Greedy native cached generation and strict fixture task scoring for source or plain D0 export | Does not supply augmented generation or broad benchmark accuracy |
| profiling.py | Periodic MPS allocation/driver samples, process RSS, synchronized phase timing and host swap/pressure endpoints | Sampled MPS peaks can miss transients; unified-memory counters overlap |
| audit_checkpoint.py | Stream source/export CPU comparisons after dtype normalization | Saved-weight changes do not imply every element received a nonzero gradient |
| optimizers.py | Experimental blockwise INT8 Muon momentum + auxiliary FP32 AdamW; exact original/new routing and optimizer-state serialization | Portable eager FP32 math; not fused, distributed or a validated paper reproduction |
| setup_macos.sh / smoke_training.py | Locked environment; tiny MPS D0/D1/D2 forward/backward/update and export/reload | Does not establish full-size model fit, quality or training-run resume |
| run_contract_demo.py | Observe/store/compile/evaluate/publish/restart/reject/rollback on deterministic XOR fixtures | Fixed symbolic compiler and curriculum; no LLM learning claim |
| make_fixtures.py | 32 train + 16 dev solver-verified synthetic examples | Plumbing data only |
| pin_model.py / seal_snapshot.py | Resolve metadata revision; seal exact local snapshot bytes | Local hashing is not an independent publisher signature |
| download_model.py | Acquire complete snapshot at the existing immutable metadata lock; retain receipt | No weight execution; weights and download caches are ignored by Git |
| inspect_snapshot.py | Verify seal and count actual checkpoint tensor shapes; estimate optimizer mixture bytes | Reads headers without model execution; arithmetic is not a measured memory peak |
| inventory.py | Read-only environment/hardware report | Does not set drivers, select a GPU or establish fit |

The model runner and record/promotion utilities are separate foundations. Integrating the learned model's action/learning policy into a persistent operating loop is an explicit work package below. A tested folder of components is not yet a model-integrated autonomous system.

## 2. Run the offline core first

From the extracted project root:

~~~bash
python scripts/check_core.py --output reports/core_checks_local.json
python scripts/validate_data.py data/fixtures/train.jsonl data/fixtures/dev.jsonl --require-verified
python scripts/run_contract_demo.py --output runs/contract_demo_01
~~~

These commands need Python 3.11+ and no model weights. The demo intentionally refuses to overwrite a nonempty run directory. Its report says model_learning_evaluated=false.

For this Apple Silicon Mac, use the project Python 3.12.14 environment. `uv.lock` pins direct and transitive dependencies; installing the pinned NumPy release requires Python 3.12+. Run the complete setup or its individual steps:

~~~bash
sh scripts/setup_macos.sh
# Individual steps:
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --extra neural --extra test
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/smoke_training.py --device mps --optimizer muon8bit --output reports/muon8bit_mps_next.json
source .venv/bin/activate
~~~

The installed versions are PyTorch 2.14.1, Transformers 5.20.2, safetensors 0.8.0, NumPy 2.5.4, pytest 9.1.1 and jsonschema 4.26.0. The original CPU report is retained separately. No bitsandbytes, torchao, MLX or DeepSpeed is required for the portable optimizer.

The smoke uses three updates per D0/D1/D2 arm, checks frozen/full parameter ownership, actual embedding/head changes and export/reload with nonreentrant checkpointing. Its MPS counters are current allocations, not peak measurements. It refuses CPU fallback. Use a fresh report path on each run.

On the later Blackwell machine, create a separate environment and use [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for its OS/driver and a Blackwell-compatible build. Inventory the actual GPU, driver, CUDA build, compiled architectures and native BF16 before running the same smoke with `--device cuda`; preserve its resolved dependency list separately. Do not copy the Mac `.venv`. No CUDA device or driver was installed here.

## 3. Acquire and pin the exact text checkpoint

Inventory the intended machine:

~~~bash
.venv/bin/python scripts/inventory.py --output reports/hardware_local.json
.venv/bin/python scripts/pin_model.py --model-id openbmb/MiniCPM5-2B --output configs/minicpm5_metadata_lock.next.json
~~~

The second command fetches public metadata and config only. It resolves the current requested branch to a full immutable revision. It fails on unavailable metadata rather than replacing the model. The initial setup already wrote [minicpm5_metadata_lock.local.json](../configs/minicpm5_metadata_lock.local.json); keep that lock for this experiment, rather than re-resolving `main`.

Download the complete checkpoint at that exact revision (the helper records acquisition and resumes the same revision):

~~~bash
.venv/bin/python scripts/download_model.py --model-lock configs/minicpm5_metadata_lock.local.json --output checkpoints/minicpm5_source
~~~

Then seal the downloaded bytes:

~~~bash
.venv/bin/python scripts/seal_snapshot.py --snapshot checkpoints/minicpm5_source --input-lock configs/minicpm5_metadata_lock.local.json --output-lock configs/minicpm5_sealed_lock.local.json
~~~

The runner verifies the config, native architecture, selected local file inventory and file hashes before loading. The seal covers `.json`, `.safetensors`, `.model` and `.jinja` files, including this snapshot's tokenizer/template; it is not a general hash of arbitrary text vocabulary files. It additionally checks the total original parameter count. Hashes bind later runs to those local bytes; retain download metadata and license provenance to support the claimed upstream origin.

No weights or private records are included in this bundle.

The initial setup acquired revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` into ignored `checkpoints/minicpm5_source` and wrote [the sealed lock](../configs/minicpm5_sealed_lock.local.json). The acquisition receipt is `checkpoints/minicpm5_source.download.json`. The [shape inventory](../reports/minicpm5_snapshot_inventory.json) confirms 2,516,756,480 original parameters without constructing or executing the model:

~~~bash
.venv/bin/python scripts/inspect_snapshot.py --model-path checkpoints/minicpm5_source --model-lock configs/minicpm5_sealed_lock.local.json --output reports/minicpm5_snapshot_inventory_next.json
~~~

Its current FP32 full-text weights/gradients/optimizer-state arithmetic is 26.40 GB / 24.59 GiB before activations and temporaries. Therefore the later 24 GB Blackwell full-scope experiment needs P2 offload/sharding or a separately qualified residency change. The same Muon algorithm can first be tested on CUDA with the tiny smoke model.

## 4. A real-model preflight, not the main experiment

The following reproduces the measured plain full-tuning D0 preflight on this Mac in a fresh directory. It confirms only microbatch 1 at the fixtures' actual 55-token length; **it is not a 256-token capacity result or a 24 GB full-tuning recipe**. First run `--regime baseline --steps 0` with its own output directory for source-model response NLL. Baseline mode writes measurements without duplicating the original checkpoint.

~~~bash
.venv/bin/python scripts/train_text.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --data data/fixtures/train.jsonl \
  --dev data/fixtures/dev.jsonl \
  --output runs/minicpm5_D0_preflight_next \
  --regime full \
  --device mps \
  --dtype float32 \
  --optimizer muon8bit \
  --lr-muon-base 0.001 \
  --lr-muon-new 0.001 \
  --lr-base 0.00001 \
  --lr-new 0.0001 \
  --block-size 2048 \
  --steps 2 \
  --grad-accum 1 \
  --max-length 256 \
  --checkpointing \
  --verify-reload
~~~

Run separate fresh output directories for full (D0), modules_only (D1) and full_augmented (D2). Use baseline with steps=0 for the runner's untrained NLL reference. The command does not run B0/B1 task-generation evaluation.

The Mac path uses MPS FP32; MPS BF16 is rejected by this runner until separately validated. After qualification on the later RTX PRO 4000 Blackwell SFF, use `--device cuda --dtype bfloat16` in a new output directory. Parameters/gradients remain FP32; BF16 is autocast compute only. Run `--optimizer adamw` explicitly for the comparison, with separately tuned rates and a fit check. Never silently fall back to CPU or freeze layers to make full tuning fit.

`--lr-muon-base` and `--lr-muon-new` control hidden 2D matrices; `--lr-base` and `--lr-new` control auxiliary AdamW groups. All four are provisional preflight rates. Momentum defaults to 0.95, five FP32 Newton-Schulz steps, no Nesterov, RMS adjustment `0.2 * sqrt(max(rows, columns))`, block size 2048 and weight decay 0.01. INT8 persistent momentum uses FP32 per-block scales; temporary dequantization and matrix arithmetic remain FP32. Embeddings, output head, learned slot queries, scalars, 1D and other ineligible tensors receive FP32 AdamW. One optimizer owns these disjoint groups; its audit records exact names, counts, rates and state bytes/dtypes.

Response targets are separately tokenized; prompt labels are masked and EOS is supervised. Overlong examples fail explicitly instead of truncating the evidence or answer. The runner disables thinking mode in its chat-template invocation for a simple action/answer target; repeat native template/reasoning-mode controls before drawing conclusions about reasoning quality.

Training output contains a native base checkpoint, sidecar weights where applicable, tokenizer files and a run report. `--verify-reload` releases the original model and optimizer before loading the export and comparing one development example's logits. The exported checkpoint does not include optimizer/RNG state, and the script currently starts only from the exact source MiniCPM checkpoint, not arbitrary previously augmented exports. Add a tested resume/migration loader before repeated neural adaptation.

### Measured native results on this Mac

Revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, PyTorch 2.14.1 / Transformers 5.20.2, FP32 weights/gradients/compute, eager MPS, no CPU fallback. [Complete experiment record](../reports/minicpm5_mps_experiment_20261022.json).

| Metric | Original baseline | Full D0, two updates |
|---|---:|---:|
| Actual prompt + target length | 55 tokens | 55 tokens |
| Dev rows / supervised tokens including EOS | 16 / 32 | 16 / 32 |
| Dev response-only NLL | 15.011792898 | 1.572720953 |
| Train processed / supervised tokens | 0 / 0 | 110 / 4 |
| Update wall time | — | 47.764 s total; 22.769 / 24.995 s |
| Instrumented processed train tokens/s | — | 2.303 |
| Sampled MPS tensor allocation peak | 9.41 GiB | 24.74 GiB |
| Sampled MPS driver allocation peak | 10.07 GiB | 29.15 GiB |
| Process-lifetime RSS high-water mark | 14.49 GiB | 14.52 GiB |
| Host swap used, start → end | 1,562.69 → 1,562.69 MiB | 1,562.69 → 1,562.69 MiB |
| Strict greedy generation, 256-new-token cap | 0/16 | 2/16, constant `0` |

All **2,516,756,480 original parameters** were trainable, and all **381 original tensors** had nonzero first-update gradients. A separate [CPU streaming checkpoint audit](../reports/minicpm5_mps_preflight_checkpoint_audit.json) compared the sealed BF16 source promoted to FP32 against the saved FP32 export: all 381 tensors changed, including embeddings/head, with exact native names/shapes retained. Reloading the export reproduced all 7,180,800 compared logits exactly (maximum absolute difference 0.0). These checks establish full D0 scope and serialization; no checkpoint was promoted.

Memory was sampled every **0.25 seconds** across load, updates, evaluation, export and reload; short-lived peaks can be missed. Current RSS was sampled about every two seconds; `ru_maxrss` covers the process lifetime. On unified memory, RSS and MPS driver counters overlap and must not be added. Swap endpoint occupancy was unchanged; no claim is made about intervening swap traffic. Phase events and raw memory samples are retained in [baseline samples](../reports/minicpm5_mps_baseline_memory_samples.jsonl) and [D0 samples](../reports/minicpm5_mps_preflight_memory_samples.jsonl). Synchronization, the first gradient audit and profiler overhead are included in timing.

The initial 32-new-token generation pilot capped all 16 outputs. The matched rerun used **256 new tokens**, greedy cached native generation, the same 53-token prompts and source template with `enable_thinking=False`. The original model scored 0/16 strict integer matches (eight responses capped, eight ended with EOS). The D0 export emitted **`0` + EOS on all 16 cases** and scored 2/16, exactly matching a constant-zero control. This is answer collapse, not learned XOR; do not interpret the NLL decrease as independent task improvement. Only four training target/EOS tokens were supervised. [Original generations](../reports/minicpm5_mps_generation_baseline_256_report.json), [D0 generations](../reports/minicpm5_mps_generation_D0_256_report.json).

To reproduce generation and audit in new output paths after training exits:

~~~bash
.venv/bin/python scripts/evaluate_text.py --model-path checkpoints/minicpm5_source --model-lock configs/minicpm5_sealed_lock.local.json --data data/fixtures/dev.jsonl --output runs/minicpm5_generation_parent_next --device mps --max-examples 16 --max-new-tokens 256 --max-length 256
.venv/bin/python scripts/evaluate_text.py --model-path checkpoints/minicpm5_source --model-lock configs/minicpm5_sealed_lock.local.json --exported-base runs/minicpm5_D0_preflight_next/checkpoint/base --data data/fixtures/dev.jsonl --output runs/minicpm5_generation_D0_next --device mps --max-examples 16 --max-new-tokens 256 --max-length 256
.venv/bin/python scripts/audit_checkpoint.py --source-path checkpoints/minicpm5_source --source-lock configs/minicpm5_sealed_lock.local.json --exported-base runs/minicpm5_D0_preflight_next/checkpoint/base --output reports/minicpm5_checkpoint_audit_next.json
.venv/bin/python scripts/validate_bundle.py --validation-report reports/validation_native_preflight.json --output reports/bundle_validation_native_preflight_next.json
~~~

The saved measured D0 export is in `runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint/base`. It is an experimental preflight artifact, not an accepted developmental checkpoint. Next test balanced tiny-overfit behavior with generation controls, then actual 256-token episodes and native D1/D2 before longer campaigns. Optimizer/scheduler/RNG/data-cursor resume remains pending.

## 5. Interfaces the next coding agent must preserve

### Evidence and time

A record has a stable ID, subject, branch, actor, kind, observation time, source ID, content, confirmation/training eligibility and optional supersession. Event time and observation time are different: a past event learned today cannot influence an earlier prediction.

Writes and deletes require trusted host scope. Human-source writes/deletes need source authority. Supersession remains in one subject/branch. A deleted correction still prevents obsolete facts from reappearing in the current view. Corrections and deletion invalidate dependent artifacts; production must rebuild/retrain or separately validate unlearning when trained personal weights are affected.

The memory store does not authenticate callers itself. Its strings are supplied by an authenticated host. Do not connect a model's untrusted JSON directly to privileged scope arguments.

### Neural context and gradients

Build workspace from the entire already known prompt and support from already observed evidence. Never pool the teacher-forced answer into either. Prompt-position logits are not supervised in this formulation.

The adapter owns a batch's context until backward completes. Only then release it. Reentrant or nonreentrant recomputation cannot perform storage writes, mutate counters or switch the context. The wrapper returns residual deltas and leaves native cache indices intact, but cached generation is not yet implemented.

Frozen base parameters do not imply a no-grad wrapper around downstream decoder layers. New branches may need gradients through them. Fast state is explicit data, outside the long-lived optimizer; differentiate through it only in an arm whose objective requests that.

### Candidate and publication

A proposal cites eligible experience and names its update scope, parent, generator, training/replay manifests and budget. Use [the schema](../schemas/learning_proposal.schema.json).

Publication requires an accepted evaluation bound to the candidate payload digest and current parent. Candidate and active state are separate. A failed or inconclusive candidate cannot be activated. A rollback targets a recorded ancestor.

The included registry assumes one trusted process and does not authenticate evaluator reports or hash arbitrary referenced model directories. Production publication must bind model/sidecar/data/identity/policy artifact hashes, run under a single privileged publisher, and use transactional locking and recovery.

## 6. Work packages with concrete exit evidence

### P0 - Reconcile with the actual project repository

Read git status, branches, remotes, current execution notes and any existing tests. Preserve user changes. Compare the seven supplied specifications with actual modules and raw reports. Reuse sound implementation. This starter is a reference/migration source; it does not authorize deleting the existing project.

Exit: repository map, tested baseline, hardware inventory, explicit missing components, one integration branch and a short next-action list. If no real repository exists, initialize the extracted project locally; no remote URL is invented.

### P1 - Reproducible native MiniCPM experiment

Resolve exact revisions, seal files, load the native template, test short generation and one real update, and report per-device/host peaks. Add a task evaluator with answer extraction and fixed inference budget. Verify native-vs-wrapper bypass at actual dimensions.

Exit: source lock, model inventory, real native forward/backward/generation reports, generation/template parity and a feasible full-text memory path. A tiny Llama pass is insufficient.

### P2 - Full-tuning environment and resumable trainer

Implement optional training-aware CPU optimizer offload or FSDP/ZeRO sharding while keeping the existing loss and parameter registry. Follow current [DeepSpeed offload](https://www.deepspeed.ai/tutorials/zero-offload/) or [OpenBMB's full-tuning guide](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/ms_swift.md) as the starting reference. Do not pass an inference device map into distributed training.

The selected single-device Muon 8-bit setup now has tiny-model tests and a measured native two-update D0 path at actual sequence length 55. It does not supply offload/sharding or establish longer-context/D2 fit. Its optimizer state serialization preserves INT8 momentum, but the model runner does not yet save/reload that state. Add optimizer/scheduler/RNG/data-cursor checkpoints, interruption/restart equivalence and configurable warmup. Saved-weight changes are audited separately against the source; qualify the custom wrapper under any later partitioner and test differentiable inner updates separately.

Exit: D0 and D2 perform real full-scope updates on the intended hardware without silent freezing; interrupted/resumed runs have documented numerical tolerance.

### P3 - Causal dataset ingestion and model task evaluation

Implement source registry ingestion, rich episode provenance, split-before-expansion, causal cutoffs, independent validators and exact export. Add frozen external benchmark adapters only after protocol/license checks. Keep public benchmark labels away from teachers.

Exit: training manifest, stratified quality audit, leakage tests, generation-based task scores and equal-information/equal-budget controls. Source availability times and family groups survive export.

### P4 - D0/D1/D2 mechanism study

Run tiny overfit, one-seed screen and shortlisted multi-seed comparisons. Add the parameter/compute-matched feed-forward control. Analyze failures instead of stacking more papers.

Exit: selected interface or justified negative result, task/retention/calibration results, measured overhead and actual trainable tensors. Do not stop at declining NLL.

### P5 - Complete learned operating and learning loop

Implement a bounded action grammar, real permitted tool results, experience capture, capability ledger, learning-proposal generation and candidate trainer invocation. First use supervised action/director data, then compare fixed and learned selection. Keep verification and publication in trusted host code.

Connect model inference to the ledger and candidate registry. Replace fixed symbolic demo decisions with actual model proposals and real neural/skill candidate updates. Add a generation path that reconstructs support/workspace correctly; re-prefill is an acceptable initial route before cache optimization.

Exit: an unfamiliar verified problem leads to a useful candidate, independent accept/reject, restart and better later behavior. Demonstrate an intentionally rejected candidate and a rollback. The base is hash-stable across the post-freeze experiment.

### P6 - Replica and longitudinal evaluation

Build the authentic source archive and dated identity snapshot. Add source-grounded decision conditioning, unknown biography handling and branch-local history. Collect sealed decisions without exposing their answers to generation/training. Run T0-T3 for ten cycles and report the full retention matrix.

Exit: source vs agent attribution, personal fidelity separated from correctness/style, no hidden audit reuse, visible selection costs, and causal evidence that the learning mechanism adds value.

### P7 - Native small VLM and optional fast-state mechanisms

Build a V4.7-specific adapter and multimodal collator. Preserve processor image expansion, mRoPE, visual insertion, linear-attention recurrent state and ordinary attention cache. Test real image gradients and image-dependent behavior. Then compare the chosen fast-state/shared/multi-rate mechanism independently.

Exit: full_text versus full_vlm groups and actual updates are reported; save/reload and restart preserve semantics; no naive Llama wrapper is reused.

### P8 - Large-backbone ports

Create fresh Gemma and large V4.7 interfaces, calibrators and memory encodings. Keep ordinary deployed base updates disabled. Transfer the experiment protocol and approved source records, not small-model latent tensors. Reassess task difficulty because larger native models may solve the previous pilot at ceiling.

Exit: per-model mechanism evidence, feasible deployment memory/latency, multi-cycle retention and a reproducible rollback/migration protocol.

## 7. Worker allocation and stopping rules

An integration owner owns shared schemas, source locks and milestone acceptance. In parallel, one worker handles model/training, one evidence/runtime, one datasets/Replica, and one independent evaluation/review when capacity permits. Assign disjoint files or worktrees; schema changes go through the owner.

After each coherent change, record commands, code revision, outputs, failures and next smallest task. Commit focused changes; preserve uncommitted user work. Push only under the user's established authorization.

When a gate fails, fix the identified cause or record a negative result. Do not rebuild the project, swap a model silently, drop full tuning, or reduce the objective to retrieval because those routes are easier. Also do not keep expanding optional tests once the concrete risk is resolved.

## 8. Preliminary-result report

Use [the report template](../templates/experiment_report.md). A valid preliminary report includes:

- Exact code/model/framework/tokenizer/processor/data revisions and artifact hashes.
- Parameter scope and actual changed tensors, including modal groups.
- All baselines, selected hyperparameters and failed candidates.
- Task success, uncertainty, retention, control behavior and learning costs.
- Full chronological matrix and checkpoint-selection history.
- Authentic identity dataset size and held-out decision protocol.
- Hardware peaks, inference budget and complete training-token accounting.
- Explicit implemented/tested/trained/evaluated/unrun status.

The outcome may be a narrow positive mechanism, a useful negative result or an inconclusive budget. Those are all better research outcomes than an unsupported broad self-improvement claim.
