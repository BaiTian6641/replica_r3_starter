# Execution state — balanced tiny overfit and native continuation completed

22 October 2026, America/Chicago. Branch `main`; no remote or push configured. Training implementation: `a0dc150`; researched roadmap: `1814914`. The final evidence/docs commit follows these milestones. The [earlier execution note](reports/execution_state_native_preflight_20261022.txt) and failed constant-zero candidate remain preserved.

The current diagnostic result is the **eight-update balanced D0 checkpoint**, selected by two consecutive perfect training-generation checks at updates 4 and 8, before seeing development generation. Independent saved-export generation confirms **train 8/8 and dev 14/16**. A separate fresh-process resume from update 8 through 12 also completed, retaining **train 8/8**, with **dev 13/16**. The 12-update artifact is a continuation probe, not a replacement winner. No developmental architecture or post-freeze learning candidate has been promoted.

See the [consolidated measured record](reports/minicpm5_balanced_overfit_experiment_20261022.json), [collapse diagnosis](docs/experiments/20261022_Constant_Zero_Diagnosis.md), [runbook](docs/03_Implementation_and_Runbook.md) and [33-substage roadmap](docs/stages/00_Master_Roadmap.md).

## Environment and actual training scope

Mac Studio M2 Ultra: 24 CPU / 60 GPU cores, 64 GiB unified memory, macOS 26.6.2. Python 3.12.14, torch 2.14.1, Transformers 5.20.2, safetensors 0.8.0 remain pinned in `uv.lock`; `.venv` is ready. The source remains `openbmb/MiniCPM5-2B` revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, with its original selected-file seal unchanged.

Both candidates trained **all 2,516,756,480 original parameters** on MPS, FP32 parameters/compute, microbatch 1, activation checkpointing and no CPU fallback. Eight original solver-verified training rows were selected into [the diagnostic dataset](data/diagnostics/tiny_overfit_mask1/selection_manifest.json). Each accumulated update visited every answer class `0..7` once. Actual sequences were **55 tokens**, with a configured ceiling of 256.

Experimental project Muon 8-bit owns 294 eligible matrices / 1,981,808,640 parameters; auxiliary FP32 AdamW owns 87 tensors / 534,947,840 parameters, including embeddings/head. INT8 momentum uses 2048-element blocks, FP32 scales/Newton–Schulz arithmetic, momentum 0.95, five iterations and no Nesterov. Matched RMS-aligned peak rates were `1e-4`, with two warmup updates and zero decay. These are tiny-memorization diagnostic settings, not a production SFT recipe. Muon factory defaults now inherit the corresponding auxiliary rate; historical explicit rates remain reproducible.

## Results and limits

| Measure | Selected update-8 run | Update-12 continuation probe |
|---|---:|---:|
| New updates in process | 8 | 4, restoring update 8 |
| Cumulative processed / supervised tokens | 3,520 / 128 | 5,280 / 192 |
| Training-update time for this process | 157.70 s | 80.36 s; cumulative 238.06 s |
| Profiled process including load/checkpoint/export/reload | 192.29 s | 121.58 s |
| Sampled live MPS peak | 24.88 GiB | 24.83 GiB |
| Sampled driver MPS peak | 29.18 GiB | 29.18 GiB |
| Process-lifetime RSS peak | 14.49 GiB | 13.87 GiB |
| Dev answer / EOS NLL | 1.093525 / ~7.45e-9 | 2.464197 / 0 |
| Independent exported train generation | 8/8 | 8/8 |
| Independent exported dev generation | 14/16 | 13/16 |
| Export/reload maximum logit difference | 0.0 | 0.0 |

Training answers in input order `0..7` are `1, 0, 3, 2, 5, 4, 7, 6`; all match the independent solver. All 48 completions across the four export evaluations consisted of one decimal answer token plus EOS, without reaching a cap. Training evaluation allowed 8 new tokens. Development used the same 16 rows, source template, prompt IDs, greedy cached decoding and 256-new-token cap as the historical parent (0/16) and constant-zero candidate (2/16). Exact mistakes remain in the raw reports.

Development has different width/mask source groups, but only 16 reused synthetic examples. This is a useful diagnostic improvement; it does not establish broad XOR transfer, novel algorithm acquisition, instruction retention, autonomous learning or Replica fidelity. Additional updates worsened development loss and accuracy. Several training factors changed together, so the result does not isolate a single cause of the earlier collapse. Format adaptation and pre-existing task ability can contribute.

Memory was sampled every 0.25 seconds. MPS and RSS share unified memory and must not be added; sampled peaks can miss transients. Both processes began and ended with swap occupancy 1,562.69 MiB; these endpoints do not measure swap traffic. The resumed report's token/update-time histories are cumulative, while its profile measures only the second process.

## Checkpoints and verification

- Selected inference export: `runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/checkpoint/base`.
- Resumable update-8 checkpoint: same run, `training_checkpoint_0008/`.
- Resume probe/export and update-12 runtime: `runs/minicpm5_balanced_mask1_B8_lr1e4_chunk02_20261022/`.
- Native [checkpoint byte/runtime validation](reports/minicpm5_balanced_checkpoint_validation.json) verifies both receipts, optimizer dtypes/bytes, contract, completed cursor and schedule. INT8 optimizer storage remains INT8 after save/restore; persistent optimizer tensors occupy 6,265,262,080 bytes.
- Independent CPU source/export audits confirm **all 381 original tensors changed in each export**, including embeddings/head, with exact native names/shapes: [update 8](reports/minicpm5_balanced_chunk01_checkpoint_audit.json), [update 12](reports/minicpm5_balanced_chunk02_checkpoint_audit.json).
- Reload verification compares 7,180,800 logits from one development example exactly in each process. Fresh-process full-size continuation is qualified. Exact uninterrupted versus resumed equivalence is proven for a real tiny native CPU model; a full-size MPS continuous-versus-split reference was not run.
- **169 tests passed**, zero failures/errors/skips, on the actual host: [output](reports/test_output_balanced_overfit_final.txt), [JUnit](reports/tests_balanced_overfit_final.xml), [current validation](reports/validation_balanced_overfit.json). Three schemas, original fixtures and the derived train/dev set pass independent XOR and split checks in [data validation](reports/validation_balanced_data.json).

Checkpoints are supported only between completed optimizer updates with gradients cleared. They bind implementation/package/source/data/order/config hashes and preserve model, optimizer, RNG, warmup/schedule, histories and cursor. Mid-accumulation recovery, cross-device numerical equivalence, distributed sharding and optimizer offload are not implemented. Weights/runtime stay ignored; compact raw reports and receipts are tracked.

## Next smallest work

**S0.5:** construct and verify actual 256-token episodes, profile a bounded full-scope run, then progress to 512. In parallel, S1.1/S1.2 build source-grouped episode contracts, independent solvers and locked evaluation; R0/R1 prepare authentic Replica collection without inventing biography. The detailed roadmap covers 33 substages, their artifacts, dependencies, resources, failure actions and acceptance gates.

Native D1/D2, 20k-target training, retention comparisons, learned ten-cycle T0–T3 loops, VLM integration and large-model transfer remain gated. The later 24 GB RTX PRO 4000 Blackwell SFF is untested; current FP32 full-text persistent state alone is 24.59 GiB before activations/temporaries, requiring a separately qualified residency/offload/sharding solution. Preserve full D0/D2 scope and Muon 8-bit when qualifying it.
