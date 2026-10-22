# Native D1 tiny-memorization diagnostic — 22 October 2026

D1 learned all eight training answers with its original backbone frozen. The
first declared rate failed its 64-update budget; the conditional fresh run at
the higher rate passed at update 28. Independent saved-export generation scored
**8/8 train and 2/16 development**. This qualifies D1's small training, generation,
continuation and export path; it does not establish useful transfer or an
architecture advantage. The global selected diagnostic remains D0 update 8.

The [declared protocol](20261022_D1_Fixed_Context_Protocol.md),
[generation qualification](20261022_Augmented_Generation.md) and
[consolidated evidence](../../reports/native_augmented_D1_experiment_20261022.json)
record the design and artifact provenance.

## Scope and execution

Native model training and answer generation ran on the **Mac Studio M2 Ultra
MPS GPU**, FP32, with CPU fallback disabled. Neural Engine execution remains
unimplemented; see the [accelerator feasibility review](20261022_Apple_Accelerators.md).
CPU work in the audits was checkpoint-file inspection, without model execution.

D1 froze all **381 original tensors / 2,516,756,480 parameters**, including the
embedding and output head, and trained all **48 added tensors / 895,490
parameters**. The 64-wide memory/workspace branches used four slots, two workspace
steps and final-layer insertion. Source evidence remained in the native prompt
as well as the support branch. Workspace and support context were built once
from the original request and held fixed through answer generation; answer
tokens never entered the builders.

Each effective update contained the eight mask-1 training rows, with answers
0–7 exactly once: microbatch one, accumulation eight, **55 actual full-sequence
tokens** per row, including 53 prompt and two supervised answer/EOS tokens.
The 256-token setting was a ceiling. The development set had 16 rows from
separate source/world groups (width 5, masks 2 and 6; training used width 4,
mask 1). These small, reused synthetic families are not a broad quality test.

Experimental Muon 8-bit owned 16 matrices; FP32 AdamW owned 32 exclusions.
Optimizer state occupied **1,029,824 bytes**: 876,544 int8 and 153,280 FP32.
Both arms used seed 17, two warmup updates, constant rates, zero weight decay
and activation checkpointing. Audits found nonzero gradients in two new output
projections at update 1 and all 48 new tensors at update 2; no original gradients.

## Both rate arms and independently generated answers

| Fresh arm: new-module Muon and AdamW peak rate | Updates | Training-generation outcome, cap 8 | Independent export train | Independent export development |
| --- | ---: | --- | --- | --- |
| 1e-3 | 64 | All 16 checks: 0/8; failed budget retained | 0/8; 1 EOS; cap 8 | 0/16; 0 EOS; cap 8 |
| 1e-2 | 28 | Update 20: 5/8; updates 24 and 28: 8/8, eight distinct, all EOS | 8/8; all EOS; cap 256 | 2/16; all EOS; cap 256 |

The higher-rate arm started afresh from the same sealed parent and seed after
the lower-rate failure; it did not reuse that optimizer. The conditional screen
was declared before the first arm's final outcome. Only the two new-module
learning rates differed between arm contracts. The failed arm's truncated
external outputs and unequal evaluation caps prevent a matched quality
comparison between the two external scores.

The independent successful training answers for inputs x=0…7 were
`1, 0, 3, 2, 5, 4, 7, 6`, each followed by EOS. These are eight correct distinct
answers, so this run escaped a constant-answer failure on its training fixture.
Development passed only `dev-w5-m2-x0` and `dev-w5-m6-x1`; 14 answers were wrong.
Update 28 is selected only for D1's memorization gate, without publication or
promotion of the developmental parent.

Raw independent reports:
[1e-3 train](../../reports/minicpm5_D1_lr1e3_step64_train_generation_report.json),
[1e-3 development](../../reports/minicpm5_D1_lr1e3_step64_dev_generation_report.json),
[1e-2 train](../../reports/minicpm5_D1_lr1e2_step28_train_generation_report.json),
[1e-2 development](../../reports/minicpm5_D1_lr1e2_step28_dev_generation_report.json).

## Memory, timing and search cost

| Arm | Cumulative optimizer-update seconds | Profiled process seconds, chunks 1 + 2 | Sampled peak MPS live GiB | Sampled peak MPS driver GiB |
| --- | ---: | ---: | ---: | ---: |
| 1e-3, 64 updates | 105.0503 | 50.4516 + 245.6286 = 296.0802 | 9.4959 | 11.0055 |
| 1e-2, 28 updates | 45.9521 | 52.6053 + 97.2670 = 149.8723 | 9.4672 | 11.1129 |

The full screen cost **92 updates, 40,480 sequence tokens, 1,472 supervised
tokens and 151.0024 update seconds**. Four complete training-process profiles
totalled 445.9525 seconds, including loading, generation checks, saving and
reload checks. Separate external GPU evaluation added 58.4806 profiled seconds.
The cumulative update field in each resumed final report already includes its
first chunk and is counted once. One initial argument-validation failure added
0.2715 seconds and zero optimizer updates; its secondary missing-telemetry guard
label was not a measured memory breach.

MPS counters were sampled every 0.25 seconds and host telemetry every two
seconds. All four guard traces replayed within declared limits: 40 GiB driver,
512 MiB sustained swap growth, and 15% system free-memory floor. Existing host
swap stayed at 1,562.69 MiB, with zero sampled growth; minimum reported free
memory was 69%. RSS and MPS driver allocation overlap and must not be added.
These sampled peaks can miss transients and apply to the actual 55-token D1
workload, not a 256-token training-capacity qualification.

Trainer reports and their adjacent memory/guard/raw-generation files:
[1e-3 chunk 1](../../reports/minicpm5_D1_lr1e3_chunk01_report.json),
[1e-3 chunk 2](../../reports/minicpm5_D1_lr1e3_chunk02_report.json),
[1e-2 chunk 1](../../reports/minicpm5_D1_lr1e2_chunk01_report.json),
[1e-2 chunk 2](../../reports/minicpm5_D1_lr1e2_chunk02_report.json).

## Scope, continuation and remaining gates

The [checkpoint audit](../../reports/minicpm5_D1_checkpoint_audit_20261022.json)
verified all 381 saved original tensors equal the sealed BF16 source values
promoted to FP32: zero changed tensors or elements. All four exported base-weight
files were byte-identical. All 48 sidecar tensors changed during each continuation
(8→64 and 8→28). Every export/reload check compared 7,180,800 logits with maximum
absolute difference 0.0.

The [runtime audit](../../reports/validation_D1_runtime_20261022.json) passed
1,665 assertions covering receipts, restored optimizer/data/schedule/RNG metadata,
recorded histories and all four guard replays. This demonstrates completed-update
continuation in fresh processes. Full-native uninterrupted-versus-resumed
numerical equivalence was not measured.

The higher rate was sufficient for this one seed and fixture; the screen does
not establish a general cause of earlier collapse. D0 and D1 differ in trainable
scope, learning rates and update counts, so these costs and scores do not support
a fair architecture comparison or a useful frozen-large-base benefit. D2 optimizer
capacity, matched controls, longer contexts, retrieval/deletion behavior and
longitudinal retention/promotion remain separate unrun gates.
