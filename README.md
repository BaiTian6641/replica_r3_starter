# TAIS Obsidian / Replica R3 - Project starter

**23 October 2026 | Experience-driven learning, small-model architecture experiments and a personal parallel Replica**

The target is a semi-cognitive system that uses experience to improve its future behavior while keeping the deployed base ordinarily frozen. Develop and evaluate the interfaces through 1B/2B-class full fine-tuning first; then freeze a tested foundation, run continuing-learning experiments, and port useful mechanisms to larger models.

This package contains an executable research starter, detailed construction documents, an eight-page diagram atlas, a dataset catalog, eight complete dataset-generation/review prompts, schemas and coding-agent instructions. It is a new R3 bundle; the seven supplied R2 documents are preserved unchanged under reference/r2.

## Local setup — 22 October 2026 (America/Chicago)

The current development machine is a **Mac Studio, Apple M2 Ultra, 24 CPU cores, 60 GPU cores and 64 GiB unified memory**. The selected optimizer experiment is **Muon 8-bit**, first on PyTorch MPS and later on the **RTX PRO 4000 Blackwell SFF Edition (24 GB)**. See [hardware profiles](configs/hardware_profiles.json) and [experiment settings](configs/experiment_design.json).

Python 3.12.14 and all numerical dependencies are pinned in [uv.lock](uv.lock). The local `.venv` is installed. To reproduce the environment and run the offline checks plus tiny MPS updates:

~~~bash
sh scripts/setup_macos.sh
source .venv/bin/activate
~~~

The project implements an experimental eager PyTorch Muon path with INT8 blockwise momentum and FP32 scales/temporary arithmetic. Embeddings, the output head, learned slot queries and other ineligible tensors use auxiliary FP32 AdamW groups. Weights and gradients remain FP32. `--optimizer muon8bit` is the trainer default; `--optimizer adamw` selects the explicit comparison. This quantizes optimizer momentum, not model weights.

Current checks and limits are recorded in [execution state](execution_state.md), [Mac validation](reports/validation_macos_setup.json) and [tiny MPS optimizer smoke](reports/muon8bit_mps_smoke.json). These checks do not establish pretrained-model fit or convergence. The [runbook](docs/03_Implementation_and_Runbook.md) contains acquisition, baseline and preflight commands; the later CUDA build and hardware must be qualified on the Blackwell machine.

## Start here

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

**Not yet measured:** pretrained MiniCPM training, GPU or M2 Ultra fit/speed, architecture-quality gains, actual VLM integration, a learned autonomous director, ten-cycle neural improvement, large-model transfer or personal-replica fidelity.

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

The provided model adapter is training-only, single-device and text-only. It supports baseline, full, modules_only and full_augmented. It does not provide optimizer offload, distributed sharding, generation caching, optimizer/RNG resume or a VLM adapter. The linear fast-state primitive is tested separately and is not attached to the pretrained-model training path.

A 24 GB GPU needs an additional offload/sharding implementation for conventional full MiniCPM5 AdamW tuning. The selected Muon experiment reduces persistent momentum storage but retains FP32 AdamW for the large embedding/head tensors; full-model fit still requires measurement and may need offload/sharding.

The memory ledger and publisher are trusted-process reference utilities, not a completed authentication/sandbox service. The model must eventually operate through a host that owns scope, budgets and evaluation. Generated memory text cannot grant permissions.

## Data and privacy boundaries

The starter contains fictional, solver-verified fixtures only. No authentic user biography, private conversation archive, model weights or downloaded public dataset is bundled. Public dataset entries begin unpinned and unapproved for ingestion until their exact version, split, component terms and provenance are checked.

Teacher prompts preserve source lineage and always mark their own candidates unverified. Independent executable checks or source-person confirmation provide the relevant admission evidence. Benchmark answer annotations never become memory input.

## Change discipline

When integrating into an existing project, audit and reuse its working code. Do not initialize a replacement repository or overwrite user changes merely because this bundle has a clean layout. If starting fresh, initialize the extracted project locally and add only an explicitly provided remote. The agent instructions include both paths.

All thresholds, schedules, mixtures and architecture dimensions are proposed starting points. The goal of the next run is a trustworthy preliminary result, including a useful negative or inconclusive result.
