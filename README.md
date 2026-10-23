# TAIS Obsidian / Replica R3 - Project starter

**23 October 2026 | Experience-driven learning, small-model architecture experiments and a personal parallel Replica**

The target is a semi-cognitive system that uses experience to improve its future behavior while keeping the deployed base ordinarily frozen. Develop and evaluate the interfaces through 1B/2B-class full fine-tuning first; then freeze a tested foundation, run continuing-learning experiments, and port useful mechanisms to larger models.

This package contains an executable research starter, detailed construction documents, an eight-page diagram atlas, a dataset catalog, eight complete dataset-generation/review prompts, schemas and coding-agent instructions. It is a new R3 bundle; the seven supplied R2 documents are preserved unchanged under reference/r2.

## Current progress — 22 October 2026 (America/Chicago)

Released **2,000 source-bound synthetic targets across four task families**,
all independently verified and under 512 tokens. The one-line response contract
was committed before its M2 Ultra GPU run. It produces **16/16 single-line EOS
responses with no cap hits**, but only **1/16 valid grammar and 0/16 correct
answers**: outputs combine fallback forms or ask for facts already present.
The prior capped failure remains preserved. Peak MPS driver memory is
**10.104 GiB**, with no observed swap growth. No downloads, training or new
checkpoints occurred. The full suite passes **671 tests, one skip**.

Read the [paired results and diagnosis](docs/experiments/20261022_Response_Contract_V1_Results.md)
and [current execution state](execution_state.md). The next development gate is
a separately predeclared query-policy-specific response contract; public admission
and remaining evaluation controls still precede S2.4 training. Selected D0/D1/D2
checkpoints and the single original model remain preserved.

## Local setup — 22 October 2026 (America/Chicago)

The current development machine is a **Mac Studio, Apple M2 Ultra, 24 CPU cores, 60 GPU cores and 64 GiB unified memory**. The selected optimizer experiment is **Muon 8-bit**, first on PyTorch MPS and later on the **RTX PRO 4000 Blackwell SFF Edition (24 GB)**. See [hardware profiles](configs/hardware_profiles.json) and [experiment settings](configs/experiment_design.json).

Python 3.12.14 and all numerical dependencies are pinned in [uv.lock](uv.lock). The local `.venv` is installed. To reproduce the environment and run the offline checks plus tiny MPS updates:

~~~bash
sh scripts/setup_macos.sh
source .venv/bin/activate
~~~

The project implements an experimental eager PyTorch Muon path with INT8 blockwise momentum and FP32 scales/temporary arithmetic. Embeddings, the output head, learned slot queries and other ineligible tensors use auxiliary FP32 AdamW groups. Weights and gradients remain FP32. `--optimizer muon8bit` is the trainer default; `--optimizer adamw` selects the explicit comparison. This quantizes optimizer momentum, not model weights.

Current checks and limits are recorded in [execution state](execution_state.md), [Mac validation](reports/validation_macos_setup.json) and [tiny MPS optimizer smoke](reports/muon8bit_mps_smoke.json). These setup checks are supplemented by the native experiments below; broader convergence remains unmeasured. The [runbook](docs/03_Implementation_and_Runbook.md) contains acquisition, baseline and preflight commands; the later CUDA build and hardware must be qualified on the Blackwell machine.

## Native MiniCPM5 Mac preflight — 22 October 2026

The pinned native model now completed its baseline and **two full D0 updates with Muon 8-bit on MPS**, with microbatch 1, FP32 weights/compute and activation checkpointing. All **381 original tensors** changed, including embeddings and the output head; exported/reloaded logits matched exactly.

| Measurement | Original baseline | Two-update D0 preflight |
|---|---:|---:|
| Response-only dev NLL, 16 fixture rows | 15.011793 | 1.572721 |
| Sampled peak MPS tensor allocation | 9.41 GiB | 24.74 GiB |
| Sampled peak MPS driver allocation | 10.07 GiB | 29.15 GiB |
| Training update time | — | 47.76 seconds |
| Strict generation matches, 256-new-token cap | 0/16 | 2/16; all answers were `0` |

The examples are **55 tokens**, despite a configured 256-token ceiling. Only four response/EOS tokens were supervised across the two updates. The trained model collapsed to always answering `0`; its 2/16 result equals a constant-zero control, so the loss drop does not demonstrate XOR learning. This establishes a short full-scope training path. MPS peaks are sampled every 0.25 seconds; driver and process RSS overlap in unified memory. See [raw results and limits](reports/minicpm5_mps_experiment_20261022.json) and [the updated runbook](docs/03_Implementation_and_Runbook.md).

## Balanced tiny overfit and resumable fine-tuning — 22 October 2026

A fresh full-parameter run from the sealed MiniCPM5 parent fixed constant-zero behavior on the eight selected training fixtures: **8/8 correct, eight distinct answers, at updates 4 and 8**. Every update accumulated all eight answer classes once. Matched RMS-aligned Muon/auxiliary AdamW rates of `1e-4`, two-update warmup and zero decay form a memorization diagnostic, not a production SFT recipe. The original failed preflight remains recorded above.

The eight-update run took 192.29 seconds including checkpoint/export/reload; sampled MPS live/driver peaks were 24.88/29.18 GiB. Full model, mixed optimizer, RNG, schedule and data-cursor checkpoints now support contract-bound continuation at completed-update boundaries. Tiny real-model interrupted continuation matches uninterrupted CPU training exactly; full-size native continuation and separate export generation are recorded in the final [experiment record](reports/minicpm5_balanced_overfit_experiment_20261022.json). Independent saved-export generation confirms **train 8/8 and dev 14/16** at update 8; the native continuation probe reaches update 12 with train 8/8 and dev 13/16. The dev cap matches the earlier 256-token evaluation. Training exactness is a memorization result; these 16 reused synthetic dev cases do not establish broad transfer or general retention. That milestone passed 169 tests; the current suite below has 290 passing tests.

Read the [collapse diagnosis](docs/experiments/20261022_Constant_Zero_Diagnosis.md), [33-substage roadmap](docs/stages/00_Master_Roadmap.md), [dated primary-source review](docs/stages/07_Research_Evidence_20261022.md) and [machine-readable plan](configs/stage_plan.json). Each substage specifies dependencies, artifacts, acceptance checks, resources and failure handling. The project defined in this workspace remains the design authority; the public search did not identify a matching authoritative TAIS Obsidian / Replica project.

## Actual-length capacity and native adapter readiness — 22 October 2026

Full D0 Muon 8-bit now passes bounded **actual 256-token and 512-token** training, each with two answer-balanced updates, a fresh-process checkpoint continuation, and exact export/reload. Both saved exports changed all 381 original tensors. Peak sampled MPS driver memory was **29.09 GiB / 29.06 GiB**; no declared host-pressure or swap-growth threshold was breached. These are short capacity measurements, not sustained-training or new task-quality results. The original eight-update learning checkpoint remains selected.

Native D1/D2 fixed-context uncached generation matches native tokens and logits exactly at 256 tokens. D1 trains only 895,490 new parameters in 48 tensors: the initial rate of 1e-3 fails through 64 updates; a predeclared fresh rate of 1e-2 reaches 8/8 at updates 24 and 28. Its saved export scores train 8/8 and dev 2/16, with all answers ending in EOS and all 381 original tensors unchanged. Successful-arm MPS live/driver peaks are 9.467/11.113 GiB at 55 actual tokens. This establishes tiny memorization. Useful memory benefit and broad transfer remain unproven. The earlier D1 suite had **290 passing tests**; see the current [execution state](execution_state.md) for subsequent validation. A new source-bound slice supplies 20 episodes from ten worlds, raw-byte/span hashes and independent chronological solver checks; only four train/four dev cases are exported.

See [D1 results](docs/experiments/20261022_D1_Diagnostic.md), [generation qualification](docs/experiments/20261022_Augmented_Generation.md), [consolidated evidence](reports/native_augmented_D1_experiment_20261022.json), [source-bound slice](docs/experiments/20261022_Episode_Slice.md) and [cognitive contracts](docs/experiments/20261022_Cognitive_Contracts.md). D2 actual 256/512-token optimizer/capacity and fresh-process restart now pass, with sampled driver peaks 29.07/30.08 GiB. D2 then reaches 8/8 training answers at updates 4 and 8; independent saved export scores train 8/8 and dev 14/16, all EOS. See the [D2 qualification record](docs/experiments/20261022_D2_Results.md). Expand source families and the early host-loop pilot alongside the later fair architecture study. Retention, frozen-loop, VLM, Replica and Blackwell remain separate gates.

## Start here

Model sources and experiment checkpoints are **local, ignored artifacts**. They
have never been included in the available Git history. Keep and reuse
`checkpoints/minicpm5_source`; normal checks are offline. The [Git and retention
policy](docs/experiments/20261022_Git_and_Checkpoint_Hygiene.md) documents the
commit guard and marked cleanup candidates. The [authorized checkpoint cleanup](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md)
retired the 16 marked old runtime states, preserved selected results, and enables
automatic sharing of identical native weights in future local saves. The user
subsequently resumed the bounded D2 GPU package. Every experiment reuses this
one sealed source offline; a new run never downloads another original model. The [post-D2 retirement](docs/experiments/20261022_Checkpoint_Retirement_After_D2.md) subsequently removes 19 superseded payload directories and 10 orphan pool weights, recovering an observed 123.24 GiB. Original/selected models, latest D2 update 8 and raw evidence remain; about 589 GiB is free.

| Read | Purpose |
|---|---|
| [System design](docs/01_System_Design.md) | Full objective, state ownership, native/parallel components and research distinctions |
| [Stages, experiments and hardware](docs/02_Stages_Experiments_and_Hardware.md) | D0/D1/D2 controls, post-freeze learning, GPU/Mac accounting and scale-up gates |
| [Implementation and runbook](docs/03_Implementation_and_Runbook.md) | Real commands, supported scope and exact next work packages |
| [Datasets and generation](docs/04_Datasets_and_Generation.md) | Current primary-source catalog, proposed 20k mixture, splits, provenance and prompt index |
| [Replica protocol](docs/05_Replica_Protocol.md) | Authentic decisions, source/branch separation, identity modes and fidelity evaluation |
| [References and evidence](docs/06_References_and_Evidence.md) | Papers, official repos, current release limits and what remains unrun |
| [Diagram index](docs/07_Architecture_Diagrams.md) | Editable Mermaid source and links to standalone diagrams |
| [Architecture atlas PDF](diagrams/Replica_R3_Architecture_Atlas.pdf) | Eight exportable vector block diagrams |
| [Coding-agent start instructions](AGENT_START.md) | Concrete first session and worker boundaries |
| [Continuation instructions](AGENT_CONTINUE.md) | Resume progress without restarting or reducing the objective |

## Original bundle evidence

**53 automated tests passed, with no failures/errors/skips**, using CPU PyTorch 2.14.1 and Transformers 5.20.2. Three JSON schemas and 48 fictional fixture rows were validated. The tests include native tiny-Llama residual integration, parameter ownership, gradients, source/prompt causal separation, functional fast state, record correction/deletion, split leakage, candidate binding and serialization. See [the full validation report](reports/validation.json) and [test output](reports/test_output.txt).

The offline deterministic demonstration exercises records, a fixed symbolic procedure compiler, independent numerical checks, publication, rejection, restart and rollback. It explicitly reports that no neural self-improvement was evaluated. See [the demo report](reports/contract_demo.json). A separate [bundle integrity report](reports/bundle_validation.json) checks source preservation, tested-code hashes, local links, file syntax and the diagram atlas.

**Not measured in the original bundle:** pretrained MiniCPM training, GPU or M2 Ultra fit/speed, architecture-quality gains, actual VLM integration, a learned autonomous director, ten-cycle neural improvement, large-model transfer or personal-replica fidelity. The native Mac baseline/D0 preflight above is subsequent execution evidence; the other research outcomes remain unmeasured.

## Quickstart without model weights

~~~bash
python scripts/check_core.py --output reports/core_checks_local.json
python scripts/validate_data.py data/fixtures/train.jsonl data/fixtures/dev.jsonl --require-verified
python scripts/run_contract_demo.py --output runs/contract_demo_01
~~~

Python 3.11+ suffices for these standalone core scripts. They make no network requests. The installed package and pinned numerical environment require Python 3.12+. For the current Mac environment:

~~~bash
uv sync --locked --extra neural --extra test
.venv/bin/python -m pytest tests -q
~~~

Full pretrained-model setup, immutable revision resolution, local byte sealing and the two-step training preflight are in [the runbook](docs/03_Implementation_and_Runbook.md). Weights stay in ignored `checkpoints/`; the environment setup script does not download them.

## The research sequence

1. Establish the original model and strong source-text-memory baselines.
2. Compare ordinary full tuning, frozen-base new modules, and full augmented tuning on the same evidence.
3. Freeze the selected developmental base and compare records-only, fixed replay and learned practice selection.
4. Port to the small VLM with genuine image-dependent tests and its native hybrid-state path.
5. Advance to larger models only after the mechanism survives controls, restart, retention and repeated seeds.
6. Collect authentic Replica decisions from the beginning; preserve their source ownership and sealed tests.

The primary text target is openbmb/MiniCPM5-2B, whose actual total is 2,516,756,480 parameters. MiniCPM-V-4.7-1B is a separate VLM port. The large roster remains google/gemma-4-26B-A4B-it and openbmb/MiniCPM-V-4.7-35B-A3B.

## Important implementation scope

The provided model adapter is single-device and text-only, with batch-one fixed-context uncached augmented generation. It supports baseline, full, modules_only and full_augmented. It supports contract-bound optimizer/RNG/data/schedule continuation at completed updates and native cached generation for the plain D0 model, plus original-prompt-fixed uncached generation for D1/D2 exports. Optimizer offload, distributed sharding, cached generation for augmented D1/D2 and a VLM adapter remain planned. The linear fast-state primitive is tested separately and is not attached to the pretrained-model training path.

A 24 GB GPU needs an additional offload/sharding implementation for conventional full MiniCPM5 AdamW tuning. The selected Muon experiment reduces persistent momentum storage but retains FP32 AdamW for the large embedding/head tensors; full-model fit still requires measurement and may need offload/sharding.

The memory ledger and publisher are trusted-process reference utilities, not a completed authentication/sandbox service. The model must eventually operate through a host that owns scope, budgets and evaluation. Generated memory text cannot grant permissions.

## Data and privacy boundaries

The starter contains fictional, solver-verified fixtures only. No authentic user biography, private conversation archive, model weights or downloaded public dataset is bundled. Public dataset entries begin unpinned and unapproved for ingestion until their exact version, split, component terms and provenance are checked.

Teacher prompts preserve source lineage and always mark their own candidates unverified. Independent executable checks or source-person confirmation provide the relevant admission evidence. Benchmark answer annotations never become memory input.

## Change discipline

When integrating into an existing project, audit and reuse its working code. Do not initialize a replacement repository or overwrite user changes merely because this bundle has a clean layout. If starting fresh, initialize the extracted project locally and add only an explicitly provided remote. The agent instructions include both paths.

All thresholds, schedules, mixtures and architecture dimensions are proposed starting points. The goal of the next run is a trustworthy preliminary result, including a useful negative or inconclusive result.
