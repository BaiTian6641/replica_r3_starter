# Execution state — Mac setup complete

Date: 22 October 2026, America/Chicago. Branch: `main`.

Initial repository commit: `3e01be9` (the supplied workspace plus initial core-check report). The setup changes are in the commit containing this entry; `git log -1` identifies it. Exact checked code/environment bytes are bound by [the current validation report](reports/validation_macos_setup.json). No remote was configured or push performed.

## Current accepted state and decision

- No trained or accepted model checkpoint/learning manifest exists yet. The pinned original source is acquired, not instantiated or trained.
- Current machine: Apple Mac Studio M2 Ultra, arm64, 24 CPU cores, 60 GPU cores, **64 GiB unified memory**, macOS 26.6.2. [Inventory](reports/hardware_macos_m2_ultra.json).
- Installed local `.venv`: Python 3.12.14, PyTorch 2.14.1, Transformers 5.20.2, safetensors 0.8.0, NumPy 2.5.4, pytest 9.1.1, jsonschema 4.26.0; direct/transitive dependencies pinned by `uv.lock`. MPS availability and real backward execution confirmed.
- User-selected experiment: **Muon 8-bit on macOS MPS first, then NVIDIA RTX PRO 4000 Blackwell SFF Edition (24 GB)**. Use the same portable project equations on both backends; qualify the later CUDA build/driver/device separately.
- Eligible hidden 2D matrices use blockwise linear INT8 momentum (2048 entries/block, FP32 scales), FP32 dequantization and Newton-Schulz arithmetic. Embeddings/head/learned slot queries/other tensors use FP32 AdamW groups. FP32 weights/gradients remain fully eligible under D0/D2. Separate original/new Muon and AdamW rates are provisional and need tuning. All-AdamW is an explicit comparison.
- Source model: `openbmb/MiniCPM5-2B`, immutable revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Complete 11-file snapshot in ignored `checkpoints/minicpm5_source` (about 4.7 GiB); acquisition receipt beside it. [Metadata lock](configs/minicpm5_metadata_lock.local.json), [selected-file seal](configs/minicpm5_sealed_lock.local.json).

## Commands and observed evidence

1. `python3 scripts/check_core.py --output reports/core_checks_macos_initial.json`: **23 passed**, no failures/errors/skips. Standalone fixture validator: **48 rows**, six source-family groups.
2. `UV_CACHE_DIR=/private/tmp/replica-r3-uv-cache uv sync --locked --extra neural --extra test`: installed/reconciled the isolated pinned environment. Python package requirement raised to 3.12 because pinned NumPy requires it; standalone core scripts still work with Python 3.11.
3. `.venv/bin/python -m pytest tests -q --junitxml=reports/tests_macos_setup.xml > reports/test_output_macos_setup.txt 2>&1`: **70 passed**, zero failures/errors/skips; [raw output](reports/test_output_macos_setup.txt), [JUnit](reports/tests_macos_setup.xml). Three schemas and all 48 fixture rows additionally validated against schemas/hashes.
4. `.venv/bin/python scripts/smoke_training.py --device mps --optimizer muon8bit --output reports/muon8bit_mps_smoke_final.json`: **passed**, three updates each for D0 full, D1 modules_only and D2 full_augmented, with nonreentrant checkpointing and model export/reload. Final report moved to [muon8bit_mps_smoke.json](reports/muon8bit_mps_smoke.json); the earlier smoke is preserved separately. Full arms changed embeddings/head; D1 kept original weights unchanged. Only tiny random Llama weights were executed. MPS allocation counters are current values, not peaks.
5. `UV_CACHE_DIR=/private/tmp/replica-r3-uv-cache sh scripts/setup_macos.sh`: **passed end to end**, including locked environment, inventory, core/fixture checks, all 70 tests and another fresh MPS smoke. [Complete setup output](reports/setup_macos_output.txt). The desktop sandbox sometimes hides MPS/sysctl; final inventory and MPS checks ran with host hardware access, without CPU fallback.
6. `.venv/bin/python scripts/run_contract_demo.py --output runs/contract_demo_macos_setup_01`: **passed**, three symbolic cycles, restart/rollback and harmful-candidate rejection. Raw report stays in the ignored run directory. `model_learning_evaluated=false`.
7. `pin_model.py`, `download_model.py`, `seal_snapshot.py` and `inspect_snapshot.py` ran successfully at the source revision above. [Tensor-header inventory](reports/minicpm5_snapshot_inventory.json): **2,516,756,480 parameters**, split into 1,981,808,640 Muon and 534,947,840 auxiliary AdamW parameters. Weights were hashed and their headers inspected; no native pretrained forward/backward was run.
8. `.venv/bin/python scripts/validate_bundle.py --validation-report reports/validation_macos_setup.json --output reports/bundle_validation_macos_setup.json`: current syntax/link/source/hash consistency check; [report](reports/bundle_validation_macos_setup.json). The validator prunes local environments, model/cache/run/private data and accepts separate current test evidence.

The original reports, seven R2 source files and architecture assets remain preserved. The original execution note and delivery hash list are archived as [starter execution state](reports/execution_state_starter.md) and [starter manifest](reports/starter_MANIFEST.sha256); `MANIFEST.sha256` describes the current tracked delivery.

## Implemented, tested and unrun

Implemented/tested: existing software/numerical contracts; portable Muon 8-bit routing/math/state serialization; CPU reference comparisons; tiny actual MPS updates and model reload; reproducible Mac setup; immutable source acquisition, selected-file seal and header inventory. INT8 optimizer-state continuation is tested independently; the training runner does not yet persist optimizer/RNG/data cursor state.

Not trained/evaluated: native pretrained generation/NLL/task baselines, full-size MPS updates/peak memory/throughput, CUDA execution, offload/sharding, VLM integration, learned director, ten-cycle neural learning, large-model ports or Replica fidelity. The symbolic demo and tiny-model tests do not establish these results.

## Next smallest gate

P0 and the local setup portion of P1 are complete. Next: instantiate the sealed native checkpoint on this Mac, establish original-model generation/task and response-NLL baselines, then run a two-update full-scope preflight at microbatch 1 / sequence length 256 with memory-pressure/swap monitoring and export/reload. Keep long campaigns pending resumable state and a scheduler. Use new output directories.

For the later Blackwell SFF, the current FP32-resident source weights + gradients + persistent mixed optimizer tensors require **26,399,313,920 bytes (26.40 GB / 24.59 GiB)** before new modules, activations, temporary buffers and allocator overhead. This layout exceeds the 24 GB card. P2 must implement tested offload/sharding or separately qualify a lower-residency layout while retaining the selected Muon algorithm and full D0/D2 parameter scope. No driver changes or CUDA fit claim was made here.
