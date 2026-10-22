# 03 - Construction plan and executable runbook

**Status:** research starter, not a trained persistent assistant. Read [README](../README.md), [system design](01_System_Design.md) and [experiment plan](02_Stages_Experiments_and_Hardware.md) first.

## 1. What the delivered code actually does

| Component | Delivered behavior | Important boundary |
|---|---|---|
| contracts.py | Typed source/branch records; exact SFT schema; lineage/exact-duplicate split audit | Cannot prove factual correctness or user consent |
| memory.py | SQLite records, scope checks, observation-time reads, corrections, tombstones, derived-artifact invalidation | Not a semantic retriever, encrypted vault or OS security boundary |
| learning.py | Paired candidate screening; accept/reject/inconclusive; bound manifest publication; rollback | One trusted writer/process; evaluator authentication remains external |
| neural.py | Learned record pooling, residual reader, recurrent prompt workspace, functional linear fast state | Simplified project modules, not a reproduced research architecture |
| model_adapter.py | One complete native Llama block wrapper; explicit trainable scope; safe base/sidecar export | Training-only text adapter; no native generation cache, VLM or distributed support |
| train_text.py | Pinned local MiniCPM5 text runner; response-only loss; four regimes; parameter/gradient report | Real pretrained run not performed; no offload, long-run resume or task accuracy evaluator |
| run_contract_demo.py | Observe/store/compile/evaluate/publish/restart/reject/rollback on deterministic XOR fixtures | Fixed symbolic compiler and curriculum; no LLM learning claim |
| make_fixtures.py | 32 train + 16 dev solver-verified synthetic examples | Plumbing data only |
| pin_model.py / seal_snapshot.py | Resolve metadata revision; seal exact local snapshot bytes | Local hashing is not an independent publisher signature |
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

For the numerical and native-Llama tests, create an isolated virtual environment and install a platform-appropriate PyTorch wheel plus the pinned packages in requirements-neural.txt. The delivered numerical report records the CPU reference versions. Then:

~~~bash
python -m pip install -r requirements-neural.txt
python -m pip install -e . --no-deps
python -m pytest tests -q
~~~

Do not install a CPU wheel over an existing CUDA environment. Use [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for the actual driver/platform. The package metadata alone is not proof of hardware compatibility.

## 3. Acquire and pin the exact text checkpoint

Inventory the intended machine:

~~~bash
python scripts/inventory.py --output reports/hardware_local.json
python scripts/pin_model.py --model-id openbmb/MiniCPM5-2B --output configs/minicpm5_metadata_lock.local.json
~~~

The second command fetches public metadata and config only. It resolves the current requested branch to a full immutable revision. It fails on unavailable metadata rather than replacing the model.

Download the complete checkpoint at that exact revision in a separate deliberate setup action:

~~~python
import json
from huggingface_hub import snapshot_download

lock = json.load(open("configs/minicpm5_metadata_lock.local.json"))
snapshot_download(
    repo_id=lock["repo_id"],
    revision=lock["revision"],
    local_dir="checkpoints/minicpm5_source",
)
~~~

Then seal the downloaded bytes:

~~~bash
python scripts/seal_snapshot.py --snapshot checkpoints/minicpm5_source --input-lock configs/minicpm5_metadata_lock.local.json --output-lock configs/minicpm5_sealed_lock.local.json
~~~

The runner verifies the config, native architecture, local file inventory and file hashes before loading. It additionally checks the total original parameter count. Hashes bind later runs to those local bytes; retain download metadata and license provenance to support the claimed upstream origin.

No weights or private records are included in this bundle.

## 4. A real-model preflight, not the main experiment

The following is a single-device command for a machine that can hold the declared state. **It is not a 24 GB full-tuning recipe.** For a 24 GB GPU, first implement the offload/sharding package in section 6 or use a supported official full-tuning baseline.

~~~bash
python scripts/train_text.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --data data/fixtures/train.jsonl \
  --dev data/fixtures/dev.jsonl \
  --output runs/minicpm5_D2_preflight_01 \
  --regime full_augmented \
  --device cuda \
  --dtype bfloat16 \
  --steps 2 \
  --grad-accum 1 \
  --max-length 512 \
  --checkpointing
~~~

Run separate fresh output directories for full (D0), modules_only (D1) and full_augmented (D2). Use baseline with steps=0 for the runner's untrained NLL reference. The command does not run B0/B1 task-generation evaluation.

For a deliberate Mac feasibility trial, use device=mps and dtype=float32; MPS BF16 is rejected by this reference until separately validated. The Mac may be memory-bound or slow. Do not silently fall back to CPU and label it GPU throughput.

Response targets are separately tokenized; prompt labels are masked and EOS is supervised. Overlong examples fail explicitly instead of truncating the evidence or answer. The runner disables thinking mode in its chat-template invocation for a simple action/answer target; repeat native template/reasoning-mode controls before drawing conclusions about reasoning quality.

The output contains a native base checkpoint, sidecar weights where applicable, tokenizer files and a run report. Dev NLL is not task success. The exported checkpoint does not include optimizer/RNG state, and the script currently starts only from the exact source MiniCPM checkpoint, not arbitrary previously augmented exports. Add a tested resume/migration loader before repeated neural adaptation.

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

Add optimizer/scheduler/RNG/data-cursor checkpoints, interruption/restart equivalence, parameter-change auditing and configurable warmup. Compare a short reference update before optimizing performance. Qualify original/full training first; then qualify the custom wrapper under the chosen partitioner. Test differentiable inner updates separately.

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
