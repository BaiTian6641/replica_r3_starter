# Execution state — native Mac baseline and D0 preflight completed

Date: 22 October 2026, America/Chicago. Branch: `main`.
Code parent: `68d94a2` (Mac setup). This experiment used the instrumented working tree saved in the commit containing this note; source hashes and raw evidence are in [the experiment record](reports/minicpm5_mps_experiment_20261022.json) and [current validation](reports/validation_native_preflight.json). No remote or push was added.

## Current checkpoint and gate

No developmental checkpoint or learning candidate was accepted/promoted. The pinned original remains `openbmb/MiniCPM5-2B` at `f97400052a43d642bbc6e9975e2397e3ae6a6b52`; source and selected-file seal are unchanged. The experimental two-update D0 export is in `runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint/base`, with native FP32 weights, tokenizer and adapter manifest. It is not resumable training state.

P0 setup and the **plain native D0 feasibility path** are complete: original forward/NLL and cached generation, two full updates, measured memory, export/reload and actual saved tensor changes. Native augmented D1/D2, representative longer contexts and a useful task-learning result remain incomplete.

Current machine remains Mac Studio M2 Ultra (24 CPU / 60 GPU cores, 64 GiB unified memory). Python 3.12.14, torch 2.14.1, Transformers 5.20.2, safetensors 0.8.0; locked environment unchanged. MPS ran FP32 without CPU fallback; no memory safeguards were changed.

## Measured results

| Measure | Frozen original | Full D0 preflight |
|---|---:|---:|
| Dev response NLL (16 rows / 32 response+EOS tokens) | 15.011792898 | 1.572720953 |
| Training updates | 0 | 2 |
| Training sequence length actually executed | — | 55 tokens; configured ceiling 256 |
| Processed / supervised training tokens | 0 / 0 | 110 / 4 |
| Two update time | — | 47.764 s; 2.303 processed tokens/s |
| Sampled MPS tensor allocation peak | 9.41 GiB | 24.74 GiB |
| Sampled MPS driver allocation peak | 10.07 GiB | 29.15 GiB |
| Process-lifetime peak RSS | 14.49 GiB | 14.52 GiB |
| Swap occupancy at both endpoints | 1,562.69 MiB | 1,562.69 MiB |
| Strict generation, greedy / 256 new tokens | 0/16 | 2/16; always answered `0` |

The D0 optimizer was the project Muon 8-bit reference: INT8 matrix momentum in 2048-element blocks, FP32 scales and Newton-Schulz arithmetic, momentum 0.95, five NS steps, no Nesterov, Muon base LR 0.001, auxiliary FP32 AdamW LR 0.00001 and weight decay 0.01. All 2,516,756,480 original parameters were eligible, with 381 nonzero-gradient tensors and no missing gradients. CPU source/export comparison (source BF16 promoted to FP32) confirms changes in **all 381 original tensors**, including embeddings/head, without name/shape changes. Reload reproduced **7,180,800 logits exactly** (max difference 0.0).

Memory counters were sampled every 0.25 seconds through load, training, evaluation, export and reload. MPS peaks can miss transients. RSS and MPS driver allocation overlap in unified memory and must not be added. Swap occupancy was measured at endpoints only; unchanged occupancy does not prove no intervening swap traffic. Timings include synchronization, profiling and gradient auditing.

**Negative task result:** the initial 32-new-token pilot capped all 16 original-model outputs. At the matched 256-token budget, the original got 0/16 strict integer-only matches (eight capped, eight EOS); D0 emitted `0` plus EOS for every example and got 2/16, equal to an always-zero control. The loss drop does not establish learned XOR, broad improvement or convergence. Only four response/EOS training tokens were supervised. Keep this failed task behavior visible; do not promote the preflight artifact based on NLL.

## Commands and raw evidence

- Baseline: `.venv/bin/python scripts/train_text.py --model-path checkpoints/minicpm5_source --model-lock configs/minicpm5_sealed_lock.local.json --data data/fixtures/train.jsonl --dev data/fixtures/dev.jsonl --output runs/minicpm5_mps_baseline_20261022_01 --regime baseline --device mps --dtype float32 --steps 0 --grad-accum 1 --max-length 256`. [Report](reports/minicpm5_mps_baseline_report.json), [raw memory](reports/minicpm5_mps_baseline_memory_samples.jsonl).
- D0: same model/data paths, `--output runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01 --regime full --device mps --dtype float32 --optimizer muon8bit --steps 2 --grad-accum 1 --max-length 256 --lr-muon-base 0.001 --lr-base 0.00001 --block-size 2048 --checkpointing --verify-reload`. [Report](reports/minicpm5_mps_preflight_report.json), [raw memory](reports/minicpm5_mps_preflight_memory_samples.jsonl).
- Generation: `scripts/evaluate_text.py` with the same source lock and dev data, `--device mps --max-examples 16 --max-new-tokens 256 --max-length 256`; separate parent and D0 output directories, with `--exported-base runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint/base` for D0. [Parent](reports/minicpm5_mps_generation_baseline_256_report.json), [D0](reports/minicpm5_mps_generation_D0_256_report.json), [32-token pilot](reports/minicpm5_mps_generation_baseline_32_pilot_report.json).
- Saved weights: `scripts/audit_checkpoint.py --source-path checkpoints/minicpm5_source --source-lock configs/minicpm5_sealed_lock.local.json --exported-base runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint/base --output reports/minicpm5_mps_preflight_checkpoint_audit.json`. [381-tensor audit](reports/minicpm5_mps_preflight_checkpoint_audit.json).
- `.venv/bin/python -m pytest tests -q --junitxml=reports/tests_native_preflight_code.xml > reports/test_output_native_preflight_code.txt 2>&1`: **94 passed**, zero failures/errors/skips. [Raw output](reports/test_output_native_preflight_code.txt), [JUnit](reports/tests_native_preflight_code.xml). Added tests cover telemetry units/availability guards, generation scoring/export provenance and bounded CPU weight comparisons. Existing schema/48-fixture validation remains unchanged and was rechecked.
- `.venv/bin/python scripts/validate_bundle.py --validation-report reports/validation_native_preflight.json --output reports/bundle_validation_native_preflight.json`: [current syntax/link/source/hash report](reports/bundle_validation_native_preflight.json). Original bundle/setup reports and all seven R2 documents remain preserved.

## Next smallest work

First diagnose the constant-answer failure with balanced tiny-overfit examples and generation-based controls; use the current native parent and a separately named candidate. Then qualify real 256-token episodes and native D1/D2. Add scheduler/optimizer/RNG/data-cursor checkpoints and interruption equivalence before long training; current export/reload verifies weights only. No further training campaign was started in this work package.

The later RTX PRO 4000 Blackwell SFF remains untested. The current full FP32 mixed-optimizer persistent state alone is 26.40 GB / 24.59 GiB, before activations and temporaries; it needs P2 offload/sharding or a separately validated residency change on the 24 GB device. Preserve Muon 8-bit and complete D0/D2 scope. VLM, autonomous director, retention, large-model transfer and Replica fidelity remain unrun.
