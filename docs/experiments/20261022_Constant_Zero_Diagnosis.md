# Constant-zero diagnosis and balanced tiny overfit — 22 October 2026

The balanced full-parameter MiniCPM5 experiment generated all eight training answers correctly at both update 4 and update 8. Every answer `0..7` appeared once at each check, and every completion ended with EOS after one answer token. The run stopped after the second perfect check. A separate exported-model evaluation confirmed 8/8 training answers and 14/16 development answers. A native restart from update 8 then completed updates 9–12, retained 8/8 training answers and scored 13/16 on development. This establishes that the current Mac/MPS training path can fit this tiny fixture without constant-zero output. Format adaptation and pre-existing XOR ability may contribute; these results do not establish acquisition of a novel algorithm, broad XOR transfer, general instruction retention, or a trained Replica.

Evidence: [completed training report](../../reports/minicpm5_balanced_mask1_chunk01_report.json), [update-4 generated answers](../../reports/minicpm5_balanced_mask1_chunk01_generation_step0004.json), [update-8 generated answers](../../reports/minicpm5_balanced_mask1_chunk01_generation_step0008.json), and [dataset selection manifest](../../data/diagnostics/tiny_overfit_mask1/selection_manifest.json). These text reports are preserved for Git history; large checkpoint weights and optimizer state remain outside Git.

## What the earlier preflight demonstrated

The [two-update report](../../reports/minicpm5_mps_preflight_report.json) records only `train-w4-m7-x0 → 7`, followed by `train-w4-m7-x7 → 0`. Accumulation was one example per update: two answer targets plus two EOS targets, four supervised tokens in total. The 32-row training file was never fully traversed. The [generation report](../../reports/minicpm5_mps_generation_D0_256_report.json) then shows `0</s>` on all 16 development examples: 2/16 correct, exactly the score of a constant-zero predictor on that split.

The original 32-row file also has unequal answer counts: answers `0..7` occur three times each and `8..15` once each. That file-level imbalance alone does not explain why this two-example preflight ended at zero. Its actual sampled answer exposure and most recent update are the stronger direct observations.

Reported response NLL includes both the answer token and EOS. All these fixtures have one token for their decimal answer and one EOS token, so half of the supervised positions concern termination. Learning output format can substantially improve this average while generated answers remain wrong. The new runner records answer NLL and EOS NLL separately and checks actual autoregressive completions from prompt-only inputs.

The earlier Muon learning rate was `1e-3`, while the auxiliary AdamW rate was `1e-5`. This implementation already scales Muon updates by `0.2 × sqrt(max(rows, columns))`. [PyTorch's Muon documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) identifies that as RMS alignment with AdamW and explains that it supports reusing AdamW-tuned rates. The [8-bit Muon paper, version 3](https://arxiv.org/html/2509.23106v3), uses this adjustment and reports an SFT rate of `2e-5`. The 100-fold rate difference is therefore an additional plausible instability or over-adaptation factor. It is not proof that the learning rate caused the observed zero response.

## The successful intervention

| Setting | Earlier plumbing preflight | Balanced tiny overfit |
|---|---|---|
| Starting weights | Sealed source model | Same sealed source model; fresh run |
| Trainable scope | All 2,516,756,480 original parameters | All 2,516,756,480 original parameters |
| Training subset | Two sampled rows from a 32-row file | Eight solver-rechecked mask-1 rows |
| Effective examples per update | 1 | 8, accumulated sequentially |
| Answer exposure per update | One label | Exactly one of each answer `0..7` |
| Muon / auxiliary AdamW peak rates | `1e-3` / `1e-5` | `1e-4` / `1e-4` |
| Warmup | None | Two updates: `5e-5`, then `1e-4` |
| Weight decay | `0.01` | `0` for this memorization diagnostic |
| Completed updates | 2 | 8 in the primary run, plus 4 in a separate restart check |

Both runs use the portable project Muon experiment: blockwise int8 hidden-matrix momentum, block size 2048, momentum 0.95, five FP32 Newton–Schulz iterations, auxiliary FP32 AdamW, FP32 resident parameters and MPS compute. This is an independent eager implementation, not a validated reproduction of upstream kernels or the paper's full training recipe.

The selected rows retain their original IDs, source IDs and training split. Development data remains the 16 original width-5/mask-2-or-6 rows, separate from the width-4/mask-1 training source family. A per-bit solver checks the synthetic labels. Gold responses affect the sampler's balancing strata and supervised labels; they are not inserted into observed memory or generation prompts.

| Input | Expected answer | Update 4 completion | Update 8 completion |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 2 | 3 | 3 | 3 |
| 3 | 2 | 2 | 2 |
| 4 | 5 | 5 | 5 |
| 5 | 4 | 4 | 4 |
| 6 | 7 | 7 | 7 |
| 7 | 6 | 6 | 6 |

## Completed-run measurements and independent checks

The completed report records 157.70 seconds across eight training updates and 192.29 seconds for the profiled process, including checkpointing, evaluation and export/reload. It processed 3,520 sequence tokens and 128 supervised tokens. Actual training sequences were 55 tokens; `max_length=256` was a budget, not a measured 256-token training workload.

| Counter | Recorded value |
|---|---:|
| Sampled peak live MPS allocations | 24.88 GiB |
| Sampled peak MPS driver memory | 29.18 GiB |
| Process-lifetime peak RSS | 14.49 GiB |
| Sampled peak RSS | 8.63 GiB |
| Host swap occupancy at both endpoints | 1,562.69 MiB |

MPS and process memory overlap in Apple unified memory and must not be added. Sampling can miss transient peaks; endpoint swap equality does not measure swap traffic. The profile reports no sampling errors; [the original memory samples](../../reports/minicpm5_balanced_mask1_chunk01_memory_samples.jsonl) are preserved alongside its summary.

Export/reload verification passed with maximum absolute logit difference `0.0` on the first development example. Development response NLL was `0.546763`, split into answer NLL `1.093525` and EOS NLL approximately `7.45e-9`; these values are teacher-forced losses. Independent native generation evaluated the exported weights with the same greedy policy and `max_new_tokens=256` used for the earlier generation reports. All evaluated completions terminated with EOS, so the generation cap did not truncate these answers.

| Exported weights | Training generation | Development generation |
|---|---:|---:|
| Update 8 | [8/8 correct, eight distinct answers](../../reports/minicpm5_balanced_chunk01_export_train_report.json) | [14/16 correct, 87.5%](../../reports/minicpm5_balanced_chunk01_export_dev_report.json) |
| Update 12 after restart | [8/8 correct, eight distinct answers](../../reports/minicpm5_balanced_chunk02_export_train_report.json) | [13/16 correct, 81.25%](../../reports/minicpm5_balanced_chunk02_export_dev_report.json) |

At update 8, `dev-w5-m2-x5` produced `3` instead of `7`, and `dev-w5-m6-x7` produced `3` instead of `1`. At update 12 those errors remained, and `dev-w5-m6-x2` produced `6` instead of `4`. The development set contains only two synthetic source families and 16 examples; it provides a small source-disjoint fixture check. Continuing to optimize the same eight training rows did not improve this check. Development answer NLL also increased to `2.464197` after the restart.

The full-size continuation restored the model, mixed optimizer state, RNG and declared data cursor from the update-8 checkpoint, completed four further MPS updates, generated 8/8 correct training answers at update 12, and again passed export/reload with maximum absolute logit difference `0.0`. This is a successful native restart and continuation demonstration. A full-size uninterrupted 12-update control was not run, so it does not prove numerical equivalence between uninterrupted and resumed MPS training. Exact weight/update/history equivalence is separately covered by the real tiny-Llama CPU integration test. See the [consolidated experiment results](../../reports/minicpm5_balanced_overfit_experiment_20261022.json) for the bound run and evaluation artifacts.

Several interventions changed together: answer coverage, effective batch size, learning rates, warmup, weight decay, subset and training duration. This experiment demonstrates that the combined recipe fixes constant-zero behavior on the selected training rows. A causal attribution to any single intervention requires controlled ablations from the same sealed parent with matched example exposure. Before scaling to Replica data, retain the per-update histograms, separate answer/EOS losses and strict generation checks, then qualify source-disjoint transfer, retention, longer sequences and resumability independently.
