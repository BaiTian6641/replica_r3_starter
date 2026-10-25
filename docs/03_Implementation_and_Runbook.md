# 03 - Construction plan and executable runbook

**Implementation reconciliation — 25 October 2026:** R1–R4 passed at `757ace1`.
The repair prerequisites described below are satisfied; isolated MiniCPM5 V0 is
next and remains natively unqualified. Reuse the recorded 140 integrated tests
and 1,372-test +21-subtest source-export result in
[repair qualification](../reports/memory_contract_repairs_20261025/qualification.json). This maintenance inspected that evidence without rerunning
native work. The dated design and historical results below retain their scope.

**Current priority — 25 October 2026, using the review of `7c7018e`.**
Repair caller-bound exact-context verification, complete route/update sequence
audits and representation-preserving memory rebuild, then isolated MiniCPM5 V0.
The v5 writer arm is complete and rejected: fast/disabled both 56/192 with zero
factual recall. Selective exact context is 184/192, but its reference still fails
broader gates. Original source dependency, changed-span and counter fixes are
complete; [document 17](architecture/17_Memory_Reliability_Recovery.md) specifies
remaining reproductions and acceptance tests.

Use [the loop prompt](architecture/16_Loop_Agent_Instructions.md) and
[integrated design](architecture/19_Integrated_Memory_Learning_and_RSI_Design.md)
for the active queue: remaining fixes → V0 → visual fine-tuning → qualified visual
memory → bounded RSI. Prepare [Spark FP8/BF16](architecture/20_Spark_X2_6_FP8_Preparation_Plan.md)
and [eligible datasets](08_Dataset_Acquisition_and_Curriculum.md) independently now.
These plans are not new supported runner modes. Native Spark follows reusable
interfaces and backend gates; RSI success is not a prerequisite.

MiniCPM5 and the 50 GiB guard remain fixed. Current Muon int8 momentum does not
change FP32 parameter/gradient residency. This revision gathers public metadata
and updates plans; it implements no fixes, training or asset acquisition.
Commands and P/S stages below preserve their original evidence scope and are not
instructions to repeat rejected arms. See [execution state](../execution_state.md)
for the current canonical v5 scorer replay command and saved verdict.

## Historical reader-only recovery result and commands

The historical reader-only comparison is documented in the
[reader-only results](experiments/20261024_Memory_Reliability_Reader_Only_Results.md)
and [canonical verdict](../reports/memory_reliability_v2_attempt03/verdict_final_v2.json).
Use fresh output/report paths on every rerun; these are the exact guarded commands
used for the completed attempt and restart:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_memory_reliability.py \
  --report reports/memory_reliability_v2_attempt03.guard.json \
  --driver-limit-gib 50 --swap-growth-limit-mib 512 \
  --minimum-free-percent 15 --sustained-host-samples 3 \
  -- --device mps --output runs/memory_reliability_v2_attempt03 \
  --protocol configs/memory_reliability_v4.json

PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_memory_reliability.py \
  --report reports/memory_reliability_v2_restart03.guard.json \
  --driver-limit-gib 50 --swap-growth-limit-mib 512 \
  --minimum-free-percent 15 --sustained-host-samples 3 \
  -- --device mps --output runs/memory_reliability_v2_restart03 \
  --protocol configs/memory_reliability_v4.json \
  --restart-run runs/memory_reliability_v2_attempt03
```

The independent scorer and canonical finalizer are separate host-side steps:

```bash
.venv/bin/python scripts/score_memory_reliability.py \
  --protocol reports/memory_reliability_v2_attempt03/smoke_fixture/protocol.json \
  --run reports/memory_reliability_v2_attempt03/smoke_fixture/run \
  --guard reports/memory_reliability_v2_attempt03/smoke_fixture/guard.json \
  --restart-run reports/memory_reliability_v2_attempt03/smoke_fixture/restart \
  --restart-guard reports/memory_reliability_v2_attempt03/smoke_fixture/restart_guard.json \
  --output reports/memory_reliability_v2_attempt03/scorer_smoke_canonical.json

.venv/bin/python scripts/finalize_memory_reliability_verdict.py \
  --protocol configs/memory_reliability_v4.json \
  --run runs/memory_reliability_v2_attempt03 \
  --guard reports/memory_reliability_v2_attempt03.guard.json \
  --restart-run runs/memory_reliability_v2_restart03 \
  --restart-guard reports/memory_reliability_v2_restart03.guard.json \
  --smoke-verdict reports/memory_reliability_v2_attempt03/scorer_smoke_canonical.json \
  --fixture reports/memory_reliability_v2_attempt03/smoke_fixture \
  --output reports/memory_reliability_v2_attempt03/verdict_final_v2.json
```

The candidate is rejected on six per-split fact/correction floors, two
delayed-above-controls checks and full-context parity. The selective exact
reference is correct on all 192 cases; full context has two absent-fact errors.
Do not retune the recipe or promote RSI.

## Current native memory policy

As requested on 23 October, `CapacityPolicy` and both training/generation
preflight CLI defaults now use **50 GiB = 53,687,091,200 bytes**. The CLI unit is
GiB, matching the previous 40 GiB convention. A sampled driver value at or above
the threshold stops the job. Keep one exclusive MPS worker, CPU fallback disabled,
512 MiB swap-growth and 15% minimum host-free checks over three consecutive host
samples. Initial low host-free memory blocks launch; pressure may stop below
50 GiB. This changes no allocator setting and proves no additional model fit.

For a newly admitted experiment, explicitly set `--driver-limit-gib 50` **before
the wrapper's `--` separator**, version its protocol and bind current code hashes
and fresh evidence paths. Existing explicit CLI values still override defaults.
Historical v1 protocols, result files and `run_d2_stage.py`'s pinned resume plan
retain 40 GiB. Do not edit old evidence or reuse its hashes to claim a new budget.
The historical command examples below are references, not fresh-run invitations.

**Status:** research starter, not a trained persistent assistant. Read [README](../README.md), [system design](01_System_Design.md) and [experiment plan](02_Stages_Experiments_and_Hardware.md) first.

**Local setup and native preflight, 22 October 2026 (America/Chicago):** Mac Studio M2 Ultra, 24 CPU cores, 60 GPU cores, 64 GiB unified memory, macOS arm64. PyTorch MPS is available. Muon 8-bit is the selected experiment on this machine and later RTX PRO 4000 Blackwell SFF; FP32-state AdamW remains an explicit comparison. See [hardware inventory](../reports/hardware_macos_m2_ultra.json), [setup validation](../reports/validation_macos_setup.json), [native preflight results](../reports/minicpm5_mps_experiment_20261022.json), [profiles](../configs/hardware_profiles.json) and [execution state](../execution_state.md). The historical two-update D0 preflight collapsed to `0`; the fresh balanced eight-update full D0 diagnostic generates all eight training answers correctly and scores 14/16 on the small development fixtures from its saved export. Checkpoint/export/reload and native fresh-process continuation through update 12 passed. Actual 256-token and 512-token D0 short capacity screens also passed, each with two updates split across native restart. Native D1/D2 fixed-context uncached generation now passes at 256. The bounded D1 memorization gate passes at 1e-2 after a preserved 1e-3 failure: saved train 8/8 and dev 2/16, all originals unchanged. D2 actual 256/512-token capacity and fresh-process restart now pass; its balanced tiny-overfit run stops at update 8 with 8/8 at updates 4 and 8; independent saved-export scoring confirms train 8/8 and dev 14/16, all EOS. See the [D2 qualification record](experiments/20261022_D2_Results.md). Augmented caching and broader evaluation remain separate gates. See the [D1 result](experiments/20261022_D1_Diagnostic.md). Follow the [detailed substage roadmap](stages/00_Master_Roadmap.md), [collapse diagnosis](experiments/20261022_Constant_Zero_Diagnosis.md), [learning result record](../reports/minicpm5_balanced_overfit_experiment_20261022.json) and [capacity qualification](experiments/20261022_Capacity_Qualification.md).

## 1. What the delivered code actually does

The [S1.2 v2 release](experiments/20261022_Infrastructure_Slice_V2.md)
contains 2,000 solver-backed targets from 1,000 worlds/four shared mechanisms.
Offline native tokenization admits every case under 512 without truncation
(maximum 334 tokens). The original capped 0/16 and [contract-v1 0/16
negative](experiments/20261022_Response_Contract_V1_Results.md) remain intact.
The [query-policy v2 protocol](experiments/20261022_Response_Policy_V2_Protocol.md),
committed as `b2010c2`, changes only the instruction appendix and selects one
fallback from the public query, never gold or observed missingness. The same
sixteen-case M2 Ultra GPU probe gives **4/16 strict-correct**, **12/16 legal grammar forms**, and
**16/16 single-line responses with EOS and zero caps**. A fresh-context subagent
independently judges **4/16 semantically correct and 4/16 instruction-compliant**;
agent instruction compliance and program grammar measure different things.
All successes are UNKNOWN, matching the selected constant control at 4/16;
all eight complete-evidence answers fail and four asks request observed facts.
[Results and diagnosis](experiments/20261022_Response_Policy_V2_Results.md)
therefore record abstention collapse, without a useful reasoning or architecture
gain. No downloads, optimizer updates or checkpoints were made. Qualify source grounding and task-appropriate inference settings inside the
connected E1–E3 interfaces, preserving declared controls before training comparisons. [Public metadata review](experiments/20261022_Public_Source_Admission_Review.md)
keeps both candidates quarantined; retention/calibration/evidence controls and
S1.5 admission still block the [S2.4 screen](experiments/20261022_S2_4_Screen_Proposal.md).

Historical early S3.1 [host action preparation](experiments/20261022_Host_Action_Contract.md)
passes 12 integration checks and 38 focused tests for actual pure XOR tool
receipts, scope/budgets, epoch invalidation and journal restart. Those proposals
are deterministic fixtures. Subsequent E3 work connects the real proposer,
evolving workspace and receipt loop with narrow mixed-task completion and
restart. Broad policy utility and integrated candidate learning remain open.
Full selected-system S3.1 retains its S2.5/S1.1 dependencies.

| Component | Delivered behavior | Important boundary |
|---|---|---|
| architecture/ | Explicit native adapter, typed immutable frames, source-only prefix writer access, independent block13/27 memory and workspace ports, sense41, tied core and matched feed-forward control | Batch-one uncached text reference; E3 core/readers native-qualified; narrow mixed completion, free-form negatives and unproven recurrent benefit |
| memory_system/ | Temporal ledger, scoped lexical/native-dense candidates, supervised reranker API, versioned latents, exact fetch and host commit | Held-out recall@1 0.75 lexical/dense, 0.50 rerank; slot reader fails novel values; reranking opt-in |
| run_replica_session.py / qualify_architecture.py / run_e3_task.py | Shared offline MPS loader, persistent session CLIs, E3 task restart, extension-only exports with validated v1 migration | Fresh-process reload passes; extensions are not resumable optimizer checkpoints |
| runtime/ + run_e3_qualification.py | Exact task journal, typed actions/tools, independent success check, preregistered core/control study | Engineering/restart checks and narrow mixed completion pass; success head diagnostically fitted but uncalibrated |
| replica/identity.py + reader.py | Versioned source/snapshot/context, modes, branch isolation, correction/revocation, null prompt path and optional zero-init reader | Native reader attached after workspace; encoder/training and authentic fidelity evaluation remain open |
| contracts.py | Typed source/branch records; exact SFT schema; lineage/exact-duplicate split audit | Cannot prove factual correctness or user consent |
| memory.py | SQLite records, scope checks, observation-time reads, corrections, tombstones, derived-artifact invalidation | Not a semantic retriever, encrypted vault or OS security boundary |
| learning.py | Paired candidate screening; accept/reject/inconclusive; bound manifest publication; rollback | One trusted writer/process; evaluator authentication remains external |
| neural.py + architecture/ports.py + runtime/fast_memory.py | SGDN-2 functional transitions, distinct per-depth writers, convex shared reducer, private/shared strict depth images and stable component lineage | M2/M3 accepted only on exposed synthetic episodes; M3 retention 0/2 blocks broader use; Engram, learned routing and CMS remain separate |
| model_adapter.py / generation.py | Native Llama block wrapper; explicit trainable scope; safe base/sidecar export; original-prompt-fixed uncached generation | Batch-one text reference; no augmented KV cache, VLM or distributed support |
| train_text.py | Pinned local MiniCPM5 runner; four regimes; balanced sampling; answer/EOS reports; warmup/schedules; periodic train generation; completed-update checkpoints; export/reload | Native D0/D2 actual256/512 capacity and restart measured; D1/D2 tiny memorization and saved answers measured; no offload or distributed training |
| run_d2_stage.py / validate_d2_experiment.py | Pinned offline MPS launches, exclusive worker lock, strict completed-update resume/optimizer/guard/generation audits | Launch completion is not task acceptance; metadata-mode validation leaves large payload rehashing explicit |
| evaluate_text.py | Strict greedy fixture scoring: native cached source/D0 or `--exported-adapter` with fixed-context uncached sidecars | Fixture evaluation does not establish broad benchmark accuracy |
| profiling.py | Periodic MPS allocation/driver samples, process RSS, synchronized phase timing and host swap/pressure endpoints | Sampled MPS peaks can miss transients; unified-memory counters overlap |
| audit_checkpoint.py | Stream source/export CPU comparisons after dtype normalization | Saved-weight changes do not imply every element received a nonzero gradient |
| optimizers.py | Experimental blockwise INT8 Muon momentum + auxiliary FP32 AdamW; exact original/new routing and optimizer-state serialization | Portable eager FP32 math; not fused, distributed or a validated paper reproduction |
| curriculum.py / training_tools.py | Deterministic answer-balanced update plans; selected-token answer/EOS NLL; warmup/constant/cosine rates; strict prompt-only native generation | Training gold labels define balancing strata; augmented cached generation is unsupported |
| training_state.py / checkpointing.py | Atomic completed-update model/runtime checkpoints with byte receipts and strict source/data/code/settings contracts; optimizer, RNG, schedule and sampler restoration | Tiny CPU exact continuation and MPS state tests pass; native updates 8→12 restored and completed; no full-size uninterrupted-equivalence or mid-accumulation claim |
| setup_macos.sh / smoke_training.py | Locked environment; tiny MPS D0/D1/D2 forward/backward/update and export/reload | Does not establish full-size model fit, quality or training-run resume |
| run_contract_demo.py | Observe/store/compile/evaluate/publish/restart/reject/rollback on deterministic XOR fixtures | Fixed symbolic compiler and curriculum; no LLM learning claim |
| make_fixtures.py | 32 train + 16 dev solver-verified synthetic examples | Plumbing data only |
| make_infrastructure_slice.py / infrastructure_solver.py | 2,000 source-bound targets / 1,000 worlds; independent raw-source chronology/solver checks; action/length/role accounting and duplicate rejection | Four mechanisms and grammar shared across roles; exposed audit/promotion fixtures, no unseen-mechanism or authentic-personal claim |
| infrastructure_scoring.py / prepare_infrastructure_preflight.py | Strict whole-response action/task parsing; CPU constant/missing-fact controls; offline native token accounting and pinned 16-dev-world B1 case/spec packet | Semantic evaluator controls do not measure model evidence sensitivity; broad retention, calibration and useful-gain comparison remain pending |
| response_contract.py / prepare_response_contract.py / score_response_contract_generations.py | Family-only prompt appendix; exact parent cases/evidence/model/settings and negative archive binding; independent paired token/source replay, single-line and task scores | Contract v1 reaches EOS without caps but remains 0/16 task-correct; no training, new checkpoint, broad-quality or causal-sentence claim |
| response_policy_contract.py / prepare_response_policy_contract.py / score_response_policy_generations.py | Exactly one fallback from canonical public query text; original/v1 negative pinning; strict source/token replay with separate predeclared semantic agent review | V2 reaches 4/16, equal to constant UNKNOWN; grammar and agent instruction compliance remain separate metrics; no quality or architecture gain |
| host_action_contract.py / check_host_action_contract.py | Bounded verify/answer/stop proposals, actual pure XOR receipts, scoped observations, epochs, idempotency and chained journal replay | 12 fixture integration checks and 38 focused tests; no neural policy, trained learner, full workspace restart or production authority boundary |
| make_capacity_fixtures.py | Exact native 256/512-token register-event workloads; independent per-bit checks; pinned tokenizer/template manifests | Capacity fixtures, not quality-training data or benchmark evidence |
| capacity_guard.py / run_capacity_preflight.py | Sampled driver/host stop thresholds around one native MPS trainer; preserved child/profile/host evidence | Sampled operational limits; no OOM guarantee; host counters are system-wide |
| run_generation_preflight.py | Standalone offline MPS evaluator watchdog; driver/host thresholds and final trace drain | Shares the 50 GiB default policy with training; explicit historical limits remain supported |
| qualify_native_adapter.py | Native 256-token D1/D2 null-branch, causality, sensitivity and nonreentrant forward/backward checks | This probe performs no optimizer updates or generation; separate D2 capacity runs now qualify actual256/512 |
| qualify_augmented_generation.py | Native uncached D1/D2 token/logit parity, fixed context, branch sensitivity and nonzero-sidecar reload | One fixture; no learning, quality or KV-cache claim |
| pin_model.py / seal_snapshot.py | Resolve metadata revision; seal exact local snapshot bytes | Local hashing is not an independent publisher signature |
| download_model.py | Verify/reuse the sealed original offline by default; intentional initial acquisition requires `--allow-network` | No routine download or receipt rewrite; weights and caches are ignored by Git |
| inspect_snapshot.py | Verify seal and count actual checkpoint tensor shapes; estimate optimizer mixture bytes | Reads headers without model execution; arithmetic is not a measured memory peak |
| inventory.py | Read-only environment/hardware report | Does not set drivers, select a GPU or establish fit |

The model runner and record/promotion utilities are separate foundations. Integrating the learned model's action/learning policy into a persistent operating loop is an explicit work package below. A tested folder of components is not yet a model-integrated autonomous system.

## 1a. E3 qualification and restart commands

Software checks (CPU only):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_e3_fixtures.py
.venv/bin/python scripts/run_e3_qualification.py --validate-only --output /tmp/e3_validate
```

Native work uses the existing exclusive lock and guard. Use fresh report/output
paths; the protocol is read from `configs/e3_qualification_v1.json`. A short
wiring preflight is bounded and does not update or save:

```bash
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e3_qualification.py --report reports/e3_preflight.guard.json \
  -- --device mps --output runs/e3_preflight --preflight
```

The full accepted arm (training, dev/held-out episodes, controls, success-head
attempt, one extension export) is:

```bash
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e3_qualification.py --report reports/e3_run.guard.json \
  -- --device mps --output runs/e3_run
```

Fresh-process interruption uses the selected extension, a copy of the fixture
(never mutate committed fixtures), and separate per-process step limits. The
cumulative decision cap is `--max-decisions`; `--max-steps` bounds one process.
The observed E3 run used `--mode workspace_zero` because the full-core arm did
not reach a failing test:

```bash
cp -R fixtures/e3/cases/interrupted_retry_06 runs/e3_interrupted/case
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e3_task.py --report reports/e3_interrupted_run1.guard.json \
  -- --device mps --case runs/e3_interrupted/case \
  --journal runs/e3_interrupted/task.sqlite \
  --ledger runs/e3_interrupted/memory.sqlite \
  --workspace-dir runs/e3_interrupted/workspace \
  --output runs/e3_interrupted/run1 \
  --extension runs/e3_native_attempt03_20261023/selected_extension \
  --mode workspace_zero --max-decisions 7 --max-steps 2 --max-tool-calls 5 \
  --max-prompt-tokens 640 --max-total-prompt-tokens 4480 \
  --max-new-tokens 128 --max-total-generated-tokens 896 \
  --branch main --purpose bounded_repair
```

The host correction (CPU) bumps the exact revision; a process at the old
revision must reject its persisted workspace:

```bash
.venv/bin/python scripts/e3_apply_correction.py \
  --journal runs/e3_interrupted/task.sqlite --case runs/e3_interrupted/case \
  --workspace-dir runs/e3_interrupted/workspace \
  --note "A later host correction is authoritative." \
  --constraint-id con-correction --constraint-text "The evaluator remains authoritative." \
  --output runs/e3_interrupted/correction
```

Then repeat the guarded CLI with a fresh report/output and `--max-steps 5`.
Collect the split engineering/learned checks with:

```bash
.venv/bin/python scripts/e3_interrupted_acceptance.py \
  --run1-report runs/e3_interrupted/run1/session_report.json \
  --run2-report runs/e3_interrupted/run2/session_report.json \
  --correction runs/e3_interrupted/correction/correction.json \
  --journal runs/e3_interrupted/task.sqlite --case runs/e3_interrupted/case \
  --original-case fixtures/e3/cases/interrupted_retry_06 \
  --output runs/e3_interrupted/acceptance
```

The current model does **not** complete these fixtures. These commands reproduce
the observed engineering pass and negative learned result; they are not a recipe
for tuning against the held-out set.

## 1b. E2 retrieval and capacity commands

Both protocols and datasets are committed before native execution. Use fresh
paths and one guarded job at a time.

```bash
.venv/bin/python scripts/verify_e2_retrieval_data.py
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e2_retrieval.py --report reports/e2_retrieval.guard.json \
  -- --device mps --output runs/e2_retrieval
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e2_diagnostic.py --report reports/e2_diagnostic.guard.json \
  -- --device mps --output runs/e2_diagnostic \
  --extension runs/e2_retrieval_attempt03_20261023/selected_extension
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e2_capacity.py --report reports/e2_capacity.guard.json \
  -- --device mps --output runs/e2_capacity
```

The retrieval run trains only the reranker and block-13 sensor through
`ranking_loss`; it does not use answer loss for hard top-k. The capacity run
ascends units and prompt tokens for inference and modules-only backward, with no
optimizer step. Inference producers now run under `torch.no_grad()`; the older
8192-unit 48.08 GiB boundary was measured before that correction and must not be
treated as the correctly configured inference limit. Results and retained
failures are recorded in
[the E2 report](experiments/20261023_E2_Retrieval_and_Capacity_Results.md).
Reranking remains opt-in and lexical remains the default.

## 1c. M4 static-pattern diagnostic (v7)

The final arm is a bounded modules-only diagnostic, not a production router or
retention repair. It uses the accepted shared M3 parent and sealed MiniCPM5
source. The following are the commands used; **use fresh report/output paths on
any rerun** and do not overwrite preserved attempts:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_m4_pattern.py \
  --report reports/m4_static_attempt07.guard.json \
  --driver-limit-gib 50 --swap-growth-limit-mib 512 \
  --minimum-free-percent 15 --sustained-host-samples 3 \
  -- --device mps --output runs/m4_static_attempt07 \
  --protocol configs/m4_pattern_static_v7.json \
  --parent-extension runs/m3_depth_shared_attempt02/selected_extension

.venv/bin/python scripts/score_m4_pattern.py \
  --protocol configs/m4_pattern_static_v7.json \
  --input runs/m4_static_attempt07/run_report.json \
  --parent-controls runs/m3_repair_attempt01/run_report.json \
  --output reports/m4_static_attempt07/verdict.json
# Independent rejection returns exit status 1 after writing the verdict.

.venv/bin/python scripts/audit_m4_collisions.py \
  --protocol configs/m4_pattern_static_v7.json \
  --run-report runs/m4_static_attempt07/run_report.json \
  --output reports/m4_static_attempt07/collision_audit.json

.venv/bin/python scripts/finalize_m4_verdict.py \
  --protocol configs/m4_pattern_static_v7.json \
  --run-report runs/m4_static_attempt07/run_report.json \
  --verdict reports/m4_static_attempt07/verdict.json \
  --original reports/m4_pattern_attempt04/verdict.json \
  --original-sha256 ddbff364aa1b7bb9dce4db8b9273477a58dafff8f9d45e3740a275f49c26a7c5 \
  --collision-audit reports/m4_static_attempt07/collision_audit.json \
  --output reports/m4_static_attempt07/verdict_final.json
```

The run performs 4 source-LM +16 delayed updates, two extension saves and 427
answer generations. The independent verdict is **reject** on four quality
gates; all 16 other checks pass, including authenticated same-process
fast-image and pattern-extension restart. The CPU-only collision audit loads no
weights and validates eight numeric collision fields per group across all 51
training/router hash-call prompts, reusing the production hash implementation;
it does not regenerate answers. It found zero distinct-n-gram collisions in the
measured scopes. Fresh-process restart is conditional on acceptance and was not
run. See the [final v7 verdict](../reports/m4_static_attempt07/verdict_final.json)
and [M4 results](experiments/20261024_M4_Pattern_Results.md).

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

Reuse the existing sealed source offline. This validates local bytes and does
not download another copy or rewrite its acquisition receipt:

~~~bash
.venv/bin/python scripts/download_model.py --model-lock configs/minicpm5_metadata_lock.local.json --sealed-lock configs/minicpm5_sealed_lock.local.json --output checkpoints/minicpm5_source
~~~

Only an intentional initial acquisition or repair uses `--allow-network`; it
is unnecessary on this machine. Missing/corrupt sources fail offline instead
of fetching automatically. Preserve the original source and consult
[the retention policy](experiments/20261022_Git_and_Checkpoint_Hygiene.md) before
cleaning experiments. The source, run weights and caches are ignored by Git. See the
[applied storage record](experiments/20261022_Checkpoint_Cleanup_Applied.md) for
current retired paths, observed free space and automatic future weight sharing.
On this Mac, saves under `runs/` publish verified COW pool entries before their
receipts; keep transient serialization headroom and preserve unique pool versions
until explicit garbage collection. Existing runtime code contracts remain strict.

Only initial acquisition needs the following sealing step. Keep the existing
seal for subsequent runs:

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

### Reuse the offline experiment template

The completed [D2 launch plan](../configs/d2_resume_plan_20261022.json) is the
reference for future bounded runs. It pins source/data/code/rates, completed-update
resume, memory stops and fresh output paths. This command prints a pinned
command without executing it:

~~~bash
.venv/bin/python scripts/run_d2_stage.py --stage tiny_chunk01
~~~

The historical outputs already exist, so `--run` refuses to overwrite them.
For a new experiment, declare a new plan and new output/report paths while
keeping `checkpoints/minicpm5_source` and its sealed lock. Keep all offline
flags, MPS FP32, CPU fallback disabled, one native job, and the declared
scope/guard/audit contract. Resume only within the same exact training contract.
A new rate, source, dataset or code contract is a new declared experiment.
The [saved-export evaluation plan](../reports/D2_saved_export_evaluation_plan_20261022.json)
and `scripts/run_generation_preflight.py` show independent guarded train/dev
answer checks. Normal experiment launches never use `--allow-network`. After a documented experiment finishes, follow the [retention ledger](experiments/20261022_Checkpoint_Retirement_After_D2.md): keep selected/active final exports and restart states, archive reports/metadata, then retire superseded capacity/intermediate payloads and their orphan pool versions. The completed one-shot cleanup/recovery scripts refuse changed or unreviewed scopes; future retirement needs a fresh reviewed inventory.

## 4. A real-model preflight, not the main experiment

The following preserves the historical two-update D0 preflight settings, including the Muon/auxiliary rate mismatch associated with the constant-zero result below. Use the balanced diagnostic later in this section for the current tiny-overfit recipe. Both confirm only the fixtures' actual 55-token length; **neither is a 256-token capacity result or a 24 GB full-tuning recipe**. First run `--regime baseline --steps 0` with its own output directory for source-model response NLL. Baseline mode writes measurements without duplicating the original checkpoint.

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

`--lr-muon-base` and `--lr-muon-new` control hidden 2D matrices; `--lr-base` and `--lr-new` control auxiliary AdamW groups. Omitted Muon rates now inherit the corresponding auxiliary rate; there is no implicit `1e-3` Muon default. This implementation uses RMS adjustment `0.2 * sqrt(max(rows, columns))`, which [PyTorch documents](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) as supporting reuse of AdamW-tuned rates. The explicit historical `1e-3`/`1e-5` pair above remains a failed preflight setting. The new diagnostic uses matched `1e-4` rates; campaign rates still need controlled search.

Momentum defaults to 0.95, five FP32 Newton-Schulz steps, no Nesterov, block size 2048 and weight decay 0.01. INT8 persistent momentum uses FP32 per-block scales; temporary dequantization and matrix arithmetic remain FP32. Embeddings, output head, learned slot queries, scalars, 1D and other ineligible tensors receive FP32 AdamW. One optimizer owns these disjoint groups; its audit records exact names, counts, rates and state bytes/dtypes.

Response targets are separately tokenized; prompt labels are masked and EOS is supervised. Overlong examples fail explicitly instead of truncating the evidence or answer. The runner disables thinking mode in its chat-template invocation for a simple action/answer target; repeat native template/reasoning-mode controls before drawing conclusions about reasoning quality.

Training output contains a native inference export under `checkpoint/`, sidecar weights where applicable, tokenizer files and a run report. `--verify-reload` releases the original model and optimizer before loading the export and comparing one development example's logits. Inference exports and resumable training checkpoints have different roles: `--checkpoint-every` writes `training_checkpoint_NNNN/` with model/tokenizer/sidecars, `runtime.pt` and a byte receipt. Runtime state includes mixed optimizer tensors, Python/Torch CPU/MPS/CUDA RNGs where available, immutable base learning rates, completed updates and data-plan cursor. Muon INT8 momentum and FP32 auxiliary moments retain their dtypes.

`--resume-from` accepts a completed training checkpoint whose source, data, batch plan, implementation hashes, package versions and training settings match the run contract. It rejects incompatible or incomplete artifacts before continuing. Checkpoints occur only after a complete optimizer update with cleared gradients; pending accumulated gradients are not saved. Tiny CPU interrupted/resumed updates match exactly, with MPS optimizer/RNG restoration covered separately. Native fresh-process continuation from update 8 through 12 completed with finite full-scope updates, perfect training generation and exact export/reload; no full-size uninterrupted-versus-resumed numerical equivalence is claimed.

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

The saved historical D0 export is in `runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint/base`. Preserve it as the negative preflight artifact; the balanced experiment starts again from the sealed source.

### Balanced eight-class full D0 diagnostic

The [declared experiment design](../configs/tiny_overfit_design.json) selects eight solver-rechecked training rows from one width-4/mask-1 family, with exactly one of every answer `0..7` in each accumulated update. The development rows remain in a separate source family. Run only one full-size MPS process at a time and use a new output directory:

~~~bash
.venv/bin/python scripts/train_text.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --data data/diagnostics/tiny_overfit_mask1/train.jsonl \
  --dev data/fixtures/dev.jsonl \
  --output runs/minicpm5_balanced_mask1_B8_lr1e4_next \
  --regime full --device mps --dtype float32 --optimizer muon8bit \
  --steps 64 --stop-after 8 --grad-accum 8 --max-length 256 \
  --sampling answer_balanced \
  --lr-base 0.0001 --lr-new 0.0001 \
  --lr-muon-base 0.0001 --lr-muon-new 0.0001 \
  --lr-schedule constant --warmup-steps 2 --weight-decay 0 \
  --generation-every 4 --generation-max-new-tokens 8 \
  --stop-on-train-perfect --checkpoint-every 8 \
  --checkpointing --verify-reload --seed 17
~~~

The [completed chunk-01 report](../reports/minicpm5_balanced_mask1_chunk01_report.json) stopped at update 8 after two consecutive perfect training-generation checks. [Update 4](../reports/minicpm5_balanced_mask1_chunk01_generation_step0004.json) and [update 8](../reports/minicpm5_balanced_mask1_chunk01_generation_step0008.json) each scored **8/8 strict correct answers**, eight distinct responses, and one answer token plus EOS for every row. Only past-only prompt IDs reached generation. Every update covered all eight answer classes once. All 2,516,756,480 original parameters remained trainable in FP32, and all 381 original tensors had nonzero first-update gradients.

| Measurement | Completed balanced run |
|---|---:|
| Actual sequence length / microbatch / accumulation | 55 tokens / 1 / 8 |
| Updates / processed training tokens / supervised tokens | 8 / 3,520 / 128 |
| Training-update time / processed tokens per second | 157.701 s / 22.321 |
| Profiled process time including checkpoint/export/reload | 192.287 s |
| Sampled peak live MPS allocation / driver allocation | 24.88 / 29.18 GiB |
| Process-lifetime peak RSS / sampled peak RSS | 14.49 / 8.63 GiB |
| Host swap start → end | 1,562.69 → 1,562.69 MiB |
| Dev answer NLL / EOS NLL | 1.093525 / approximately `7.45e-9` |
| Dev combined response NLL | 0.546763 |
| Native export/reload maximum absolute logit difference | 0.0 over 7,180,800 logits |

The same 0.25-second memory sampling and unified-memory overlap caveats apply. Checkpoint `training_checkpoint_0008` was published with its [byte receipt](../reports/minicpm5_balanced_mask1_chunk01_training_checkpoint_0008_receipt.json); inference export/reload also passed. Separate saved-export generation now confirms **8/8 training** and **14/16 development** answers. These small fixture results do not establish broad transfer, retention, architecture benefit or Replica fidelity.

For a bounded continuation, repeat the same contract arguments with a fresh `--output`, set `--stop-after 12`, and add `--resume-from runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/training_checkpoint_0008`. Preserve `--steps 64`, data, schedules and other contract settings; changing only the process stop boundary does not redefine the total schedule. The [completed chunk-02 report](../reports/minicpm5_balanced_mask1_chunk02_report.json) now records this fresh-process continuation through update 12, again with [8/8 strict correct training answers](../reports/minicpm5_balanced_mask1_chunk02_generation_step0012.json), eight distinct responses and exact export/reload. This qualifies the measured continuation path; equality against an independently run uninterrupted full-size control remains untested.

The resumed process took **121.582 seconds** including checkpoint restore, profiling and export/reload. Its four additional updates took **80.355 seconds**; cumulative update time is **238.056 seconds**, with **5,280 processed / 192 supervised tokens** across both chunks. Chunk-02 sampled MPS live/driver peaks were **24.83/29.18 GiB**, process-lifetime peak RSS **13.87 GiB**, and swap endpoints remained **1,562.69 MiB**. The original eight-update artifact remains the diagnostic winner under its declared stopping rule; update 12 is only a resume probe. Dev response NLL increased from 0.546763 to **1.232098** (answer **2.464197**, EOS **0.0**).

### Fresh-process verification of the saved exports

Each export was loaded separately by the unchanged native greedy evaluator. Training used an eight-new-token cap; development used the same 256-new-token cap as the original source and collapsed D0 comparisons:

| Export | Training generated answers | Development generated answers |
|---|---:|---:|
| Selected original, update 8 | [8/8 strict correct](../reports/minicpm5_balanced_chunk01_export_train_report.json) | [14/16 (87.5%)](../reports/minicpm5_balanced_chunk01_export_dev_report.json) |
| Resume probe, update 12 | [8/8 strict correct](../reports/minicpm5_balanced_chunk02_export_train_report.json) | [13/16 (81.25%)](../reports/minicpm5_balanced_chunk02_export_dev_report.json) |

All **48** completions across the four jobs were one answer token followed by EOS; none reached its token cap. Development retains the same 16 width-5/mask-2-or-6 rows, distinct from the training width-4/mask-1 source family. The set is tiny, synthetic and reused during development, not a sealed audit. The eight-update export was selected by the predeclared two-perfect-training-check rule before these development scores. The development scores therefore describe fixture behavior; they do not establish broad generalization.

Independent CPU [update-8](../reports/minicpm5_balanced_chunk01_checkpoint_audit.json) and [update-12](../reports/minicpm5_balanced_chunk02_checkpoint_audit.json) audits confirm changes in **all 381 original tensors**, including embedding/head, retaining native names and shapes. To repeat development generation in a fresh output directory after training exits:

~~~bash
.venv/bin/python scripts/evaluate_text.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --exported-base runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/checkpoint/base \
  --data data/fixtures/dev.jsonl --output runs/minicpm5_balanced_export_dev_next \
  --device mps --max-examples 16 --max-new-tokens 256 --max-length 256
~~~

Balancing, effective batch, rate, warmup, weight decay and training exposure changed together. The [diagnosis](experiments/20261022_Constant_Zero_Diagnosis.md) records plausible contributors and the limit on causal attribution. The actual-length screen below is a separate capacity experiment from the same sealed source; it leaves this eight-update learning winner unchanged. Evidence/retention controls and native D1/D2 training remain separate under the [S0 readiness gates](stages/01_Environment_and_D0_Readiness.md); full-size uninterrupted-versus-resumed equivalence still requires a separate control.

### Actual 256-token and 512-token D0 capacity and restart

The [preregistered design](../configs/capacity_design.json) completed two full-scope FP32 MPS updates per length, one before and one after a fresh-process native checkpoint restore. Every accumulated update covered answers `0..7` once. Full train/dev sequences were exactly 256 or 512 tokens, with no padding-only inflation or truncation. All 381 original tensors had nonzero finite gradients, and independent [256](../reports/minicpm5_capacity256_checkpoint_audit_20261022.json)/[512](../reports/minicpm5_capacity512_checkpoint_audit_20261022.json) audits found changes in all original tensors with names/shapes preserved. All four export/reload comparisons matched exactly. No CPU fallback or allocator change occurred.

| Actual length | Two-update time | Processed / supervised tokens | Maximum sampled live / driver memory | Profiled fresh / resumed process times |
|---:|---:|---:|---:|---:|
| 256 | 44.988 s | 4,096 / 32 | 24.956 / 29.094 GiB | 53.279 / 64.238 s |
| 512 | 50.945 s | 8,192 / 32 | 25.295 / 29.063 GiB | 61.850 / 76.586 s |

The 40 GiB driver, sustained 512 MiB host-swap-growth and sustained 15% host-free thresholds were not breached. Host swap occupancy remained 1,562.69 MiB in the sampled readings. The first two guards used an earlier exit loop; complete original JSONL was independently rechecked after an exit-race fix, confirming no actual breach. The fix preceded 512. See [per-process reports, phase timings, original samples and guard review](experiments/20261022_Capacity_Qualification.md). Resumed trainer totals include restored update history; process-profile times and peaks describe only the current process. Do not add cumulative chunk training times or overlapping memory counters.

The workloads already exist under [capacity_256](../data/diagnostics/capacity_256/) and [capacity_512](../data/diagnostics/capacity_512/). The builder refuses overwrite; use a fresh `--output-root` when rebuilding. To reproduce the measured settings with new report/checkpoint paths, define this function in bash or zsh:

~~~bash
capacity_pair() {
  local length="$1"
  local first="runs/minicpm5_capacity${length}_full_muon8bit_chunk01_next"
  local second="runs/minicpm5_capacity${length}_full_muon8bit_chunk02_next"
  local -a common
  common=(
    --model-path checkpoints/minicpm5_source
    --model-lock configs/minicpm5_sealed_lock.local.json
    --data "data/diagnostics/capacity_${length}/train.jsonl"
    --dev "data/diagnostics/capacity_${length}/dev.jsonl"
    --regime full --device mps --dtype float32 --optimizer muon8bit
    --sampling answer_balanced --steps 2 --grad-accum 8 --max-length "$length"
    --lr-muon-base 0.0001 --lr-base 0.0001
    --lr-muon-new 0.0001 --lr-new 0.0001
    --warmup-steps 2 --lr-schedule constant --weight-decay 0 --seed 17
    --checkpoint-every 1 --checkpointing --verify-reload
  )
  .venv/bin/python scripts/run_capacity_preflight.py \
    --report "reports/minicpm5_capacity${length}_chunk01_guard_next.json" \
    --driver-limit-gib 40 --swap-growth-limit-mib 512 \
    --minimum-free-percent 15 --sustained-host-samples 3 \
    -- "${common[@]}" --output "$first" --stop-after 1 || return
  .venv/bin/python scripts/run_capacity_preflight.py \
    --report "reports/minicpm5_capacity${length}_chunk02_guard_next.json" \
    --driver-limit-gib 40 --swap-growth-limit-mib 512 \
    --minimum-free-percent 15 --sustained-host-samples 3 \
    -- "${common[@]}" --output "$second" --stop-after 2 \
    --resume-from "$first/training_checkpoint_0001"
}

capacity_pair 256
# Review the completed 256 reports, receipts, gradients and guard samples first.
capacity_pair 512
~~~

These jobs disable generation and measure capacity, not quality. Keep one full-size model resident. A wrapper success alone is insufficient: apply the design's scope, exact-length, finite-update, checkpoint/reload, guard and saved-weight audit gates before advancing to the next length.

The [separate native adapter qualification](../reports/minicpm5_native_adapter_qualification256_report.json) passed one 256-token D1/D2 null-parity, causality, sensitivity and nonreentrant forward/backward check, with no optimizer updates or generation. Subsequent fixed-context uncached generation parity, D2 actual256/512 capacity and balanced tiny memorization now pass with saved-export and completed-update audits. See the [D2 results](experiments/20261022_D2_Results.md). Augmented KV caching, useful architectural benefit and broad task learning remain separate open gates.

## 5. Interfaces the next coding agent must preserve

### Evidence and time

A record has a stable ID, subject, branch, actor, kind, observation time, source ID, content, confirmation/training eligibility and optional supersession. Event time and observation time are different: a past event learned today cannot influence an earlier prediction.

Writes and deletes require trusted host scope. Human-source writes/deletes need source authority. Supersession remains in one subject/branch. A deleted correction still prevents obsolete facts from reappearing in the current view. Corrections and deletion invalidate dependent artifacts; production must rebuild/retrain or separately validate unlearning when trained personal weights are affected.

The memory store does not authenticate callers itself. Its strings are supplied by an authenticated host. Do not connect a model's untrusted JSON directly to privileged scope arguments.

### Neural context and gradients

Build workspace from the entire already known prompt and support from already observed evidence. Never pool the teacher-forced answer into either. Prompt-position logits are not supervised in this formulation.

The adapter owns a batch's context until backward completes. Only then release it. Reentrant or nonreentrant recomputation cannot perform storage writes, mutate counters or switch the context. The wrapper returns residual deltas and leaves native cache indices intact. Native cached generation is implemented for baseline/plain D0; augmented fixed-context uncached generation is qualified, while augmented KV-cache support remains unimplemented.

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

The selected single-device Muon 8-bit setup now has tiny-model tests and a measured native balanced eight-update D0 path at actual sequence length 55. Configurable warmup/constant/cosine scheduling and completed-update model/optimizer/RNG/data-cursor checkpoints are implemented. Tiny CPU exact restart and MPS mixed-state tests pass; native fresh-process continuation from updates 8 to 12 completed with perfect training generation and exact export/reload. D0/D2 actual256/512 bounded fit is now qualified separately. Offload/sharding, lengths above512, sustained workloads and full-size uninterrupted-versus-resumed equivalence remain unqualified. Saved-weight changes are audited separately against the source; qualify the custom wrapper under any later partitioner and test differentiable inner updates separately.

Exit: D0 and D2 perform real full-scope updates on the intended hardware without silent freezing; interrupted/resumed runs have documented numerical tolerance.

### P3 - Causal dataset ingestion and model task evaluation

Implement source registry ingestion, rich episode provenance, split-before-expansion, causal cutoffs, independent validators and exact export. Add frozen external benchmark adapters only after protocol/license checks. Keep public benchmark labels away from teachers.

The bounded [S1.2 infrastructure release](experiments/20261022_Infrastructure_Slice_V2.md)
contains 1,200 train targets and 200 each dev/calibration/promotion/audit,
balanced 50% answer / 25% ask / 25% unknown in every role. Source/solver/export
checks pass for all 2,000 targets; automated stratified review covers 320
episodes/160 worlds, not human annotation. Fifty-nine semantic-duplicate
candidates were rejected, zero admitted. All four mechanisms appear in every
role; use worlds as statistical units and keep the exposed role packets out of
sealed claims.

The host `input_view` already selects active eligible observations using
chronology, corrections and tombstones. Native answers measure reasoning and
action choice over the compiled snapshot, not model-owned temporal retrieval
or correction detection from the full raw history.

[Actual tokenizer accounting](../reports/infrastructure_preflight_v2_20261022/token_accounting.json)
records 501,800 native tokens across all roles; train contains 300,369 native
and 4,374 supervised answer/EOS tokens, with a separate 233,295-token
source-writer view. Maximum full length is 334, with zero truncation under
512. These are single-pass data counts, not total architecture compute or a
training budget. The generator manifest's pending token-admission field is
superseded by this external, release-hash-bound accounting report.

S1.4 supplies fixed parser/control contracts and three original-model B1
observations on the same sixteen ordered development worlds. The original
0/16 task/format result with all responses capped remains preserved.
[Contract v1](experiments/20261022_Response_Contract_V1_Protocol.md), precommitted
as `430e298`, obtains all EOS/single-line and zero caps but only 1/16 grammar
and 0/16 correct. [Query-policy v2](experiments/20261022_Response_Policy_V2_Protocol.md),
precommitted as `b2010c2`, selects one fallback from public query text alone.
With the original model/evidence/template, greedy cap32, ceiling512, seed17,
MPS FP32/no CPU fallback and strict parser fixed, v2 gives 4/16 strict-correct, 12/16 legal grammar forms,
all EOS/single-line and no caps. A separately predeclared fresh-context agent
judges 4/16 semantically correct and 4/16 instruction-compliant. Program grammar
accepts legal forms independently of factual correctness; the agent also checks
policy and observed evidence. Keep those metrics separate. All four successes
are UNKNOWN, equal to the selected constant control; all eight complete-evidence
answers fail and four asks request supplied facts. The [result record](experiments/20261022_Response_Policy_V2_Results.md)
preserves errors and profiles. The predeclared suite passes 798 tests/one restricted-context MPS skip;
final integration including 21 archive-reader tests passes **819 tests/one skip**.
These software checks are separate from native task acceptance. There are zero downloads, optimizer
updates or checkpoints. No adaptive v2 prompt/parser/cap/decoder rerun occurred.
Next separately predeclare source-grounding/input-projection diagnosis and
upstream sampling/thinking qualification, rather than changing all settings
together. Missing/shuffled-evidence model controls and independent
retention/calibration cases remain required.
[S1.3 metadata](experiments/20261022_Public_Source_Admission_Review.md)
pins public candidates but leaves component terms, linked pairs/split overlap
and adapters unresolved; no public corpus is admitted.

Exit: training manifest, stratified quality audit, leakage tests, generation-based task scores and equal-information/equal-budget controls. Source availability times and family groups survive export.

### P4 - D0/D1/D2 mechanism study

Run tiny overfit, one-seed screen and shortlisted multi-seed comparisons. Add the parameter/compute-matched feed-forward control. Analyze failures instead of stacking more papers.

D0/D1/D2 tiny infrastructure gates are complete. The [S2.4 proposal](experiments/20261022_S2_4_Screen_Proposal.md)
is blocked until S1.5's source/data/evaluation admission passes; numeric rate
configurations, token/compute budgets, sample sizes and power/exposure rules
are not final. Reuse the one sealed original as the common trained-arm parent.
Do not treat the new 2k fixture release or descriptive B1 preflight as campaign
admission or useful architecture evidence.

Exit: selected interface or justified negative result, task/retention/calibration results, measured overhead and actual trainable tensors. Do not stop at declining NLL.

### P5 - Complete learned operating and learning loop

Implement a bounded action grammar, real permitted tool results, experience capture, capability ledger, learning-proposal generation and candidate trainer invocation. First use supervised action/director data, then compare fixed and learned selection. Keep verification and publication in trusted host code.

The [early S3.1 host action contract](experiments/20261022_Host_Action_Contract.md)
now passes 12 integration checks and 38 focused tests with deterministic
proposals, actual pure XOR success/failure receipts, scope/budget enforcement,
epoch invalidation and journal restart. Goal and action history persist;
evolving task workspace, actual neural proposals, learned policy and learning
remain pending. The next bounded step is a separately predeclared fixed-original
MPS proposer against host tool/receipt outcomes. This early software preparation
does not pass full S3.1 or remove its S2.5/S1.1 dependencies.

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

### Native fixed-context D1/D2 export evaluation

All native model compute on this Mac uses `--device mps`, with CPU fallback disabled. MPS is GPU execution; the [Neural Engine review](experiments/20261022_Apple_Accelerators.md) records the separate unimplemented ANE path. Data construction and checkpoint-byte verification are host tasks.

The [qualified reference](experiments/20261022_Augmented_Generation.md) builds prompt/source context once and re-prefills each answer token without a KV cache. Evaluate an augmented export with its saved sidecars/settings and the sealed original tokenizer:

```bash
.venv/bin/python scripts/evaluate_text.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --exported-adapter runs/minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022/checkpoint \
  --data data/fixtures/dev.jsonl --split dev \
  --output runs/minicpm5_D1_export_dev_next --device mps \
  --max-examples 16 --max-new-tokens 256 --max-length 256
```

Use a new output path. Training-split scores are explicitly memorization; the current D1 export scores train 8/8 and dev 2/16. The complete [protocol](experiments/20261022_D1_Fixed_Context_Protocol.md) and [result](experiments/20261022_D1_Diagnostic.md) retain both rates, guard traces and independent frozen-original audits. Source-bound episode data can be reproduced/validated separately using [the slice commands](experiments/20261022_Episode_Slice.md#commands-and-checks).
