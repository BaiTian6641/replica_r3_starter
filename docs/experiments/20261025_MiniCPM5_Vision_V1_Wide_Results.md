# MiniCPM5 V1 wide-connector qualification: engineering pass, learned quality rejected

**25 October 2026.** The second bounded projector-only V1 arm is complete.
Independent scoring returns `engineering_pass=true`, `quality_pass=false` and
`accepted=false`. This is not useful visual perception and does not admit V2,
LoRA, learned V3 visual memory or integrated RSI.

Final evidence only:

- [Protocol](../../configs/minicpm5_vision_v1_projector_wide_v1.json), SHA-256
  `a4db0c125a783ebb80f23fb9a4f04d0ecee97a7c5e9bffbc6a58309fb47aeb90`.
- [Primary run](../../runs/minicpm5_vision_v1_wide_20261025/attempt07/run_report.json),
  SHA-256 `fa69e42678057cd45a3710d0c614093c7626d1dc3d874a20e13e14cc81d5daa1`;
  native code `ddc6a07ea9e60178032c1624b2d11fb7ab52dc23`.
- [Fresh-process reload](../../runs/minicpm5_vision_v1_wide_20261025/restart07/restart_report.json),
  SHA-256 `270770a85facf8a869e16a856d7e6586050eca7bee3d06bccf5b8e2d970737f6`.
- [Scorer-only replay declaration](../../configs/minicpm5_vision_v1_wide_scoring_replay_v1.json),
  SHA-256 `afaf04c4d968875731f8f74ab582ddaf84c6491de7ad9dcfd9cf1d8d0fdd6df8`.
  It pins both native parents and changes only the independent scorer hash.
- [Independent score](../../reports/vision_v1_wide_20261025/score08/score_report.json),
  SHA-256 `aea1163688938e12c78d32cd98b8ef0410b83001d0d9f6e9eb36c84bfe421291`.

## Fixed arm and learned outcome

The arm changes the connector hidden width from 2048 to 4096 and uses fresh
seed-17 initialization rather than the V0 extension. It has six trainable
projector tensors and **11,542,016 parameters**. The admitted cohort, 128-update
schedule, 512 backward examples, Muon 8-bit plus auxiliary FP32 AdamW routing,
controls, resource limits and numeric quality gates remain fixed. Evaluation is
64 cases from 32 disjoint worlds in one owned procedural source family, not
broad photographic or source-family evidence.

| Control | Correct /64 | Accuracy | Correct-image margin |
|---|---:|---:|---:|
| Correct image | 21 | 32.8125% | — |
| Wrong image | 12 | 18.7500% | +14.0625 pp |
| Shuffled image | 15 | 23.4375% | +9.3750 pp |
| No image | 22 | 34.3750% | −1.5625 pp |
| Oracle caption-only | 60 | 93.7500% | −60.9375 pp |
| Verified empty OCR text | 15 | 23.4375% | +9.3750 pp |

The arm fails the 65% correct-image floor, every declared control margin and
the absolute text-retention floor. Correct-image accuracy is below the no-image
control. Oracle caption metadata remains an explicit upper-bound control, not
ordinary sensing input. Text retention is **22/32 before and after**, with exact
token parity and zero drop, but remains below the 75% absolute floor. Unknown
rate is 0/64 and all 32 families are represented; those do not override the
failed gates.

Mean training loss falls from **9.3908 to 1.7836**. Loss reduction and the
18/64→21/64 difference from the rejected 2048-wide arm are not a quality pass or
evidence of grounded generalization.

## Engineering and fresh-process lineage

- **128/128 updates and 512 backward calls** completed; no budget stop.
- Exactly six `vision_projector.*` tensors changed; 43 other extension tensors
  and the sealed MiniCPM5 source stayed unchanged. The SigLIP2 encoder stayed
  frozen with no trainable parameters or gradients.
- Response-only labels and decoder input-gradient paths were retained. Optimizer
  routing used Muon 8-bit for the two matrices and FP32 AdamW for the remaining
  four tensors.
- Strict v4 extension export and same-process strict reload passed.
- A separate fresh process loaded only the final extension, performed **zero
  optimizer updates**, and reproduced all evaluation and retention generations
  bitwise. Evaluation JSONL SHA-256:
  `e4f745a52c2613f628fc01a40e2e0fd43f6dffebf28e8fbd30bc06f1dc9099bc`;
  retention JSONL SHA-256:
  `3d788bdb971059bb954de6677accabf035c4fe4068163ce86e7879ba980ef4ed`.
  The ledger stayed empty and the sealed native identity stayed unchanged.
- The first scorer attempt read `extension` instead of the V1 report's
  `final_extension`. The correction is scorer-only and uses the explicit replay
  above; native runner, data, initialization and quality gates were not changed.
- The final tracked-source suite passes **1,615 tests, one skip and 21 subtests
  in 188.62 s**; see
  [JUnit evidence](../../reports/memory_repair_20261025/full_suite_v1_scorer_replay.xml).
- An [independent read-only review](../../reports/vision_v1_wide_20261025/independent_review.md)
  found no integrity discrepancies and confirmed the engineering-pass /
  learned-quality-fail split.

## Measured cost and storage

| Measurement | Primary | Fresh-process reload |
|---|---:|---:|
| Training wall time | 268.951 s | 0 updates |
| Evaluation wall time | 129.251 s | included in profile |
| Retention wall time | 6.405 s | included in profile |
| Baseline retention wall time | 6.682 s | not repeated |
| Profiled wall time | 416.831 s | 140.772 s |
| Guard elapsed | 420.248 s | 144.060 s |
| Peak sampled driver bytes | 11,956,273,152 | 10,768,056,320 |
| Peak sustained swap growth | 0 bytes | 0 bytes |
| Minimum observed host free | 74% | 76% |

The guard policy was 50 GiB sampled driver, 512 MiB sustained swap growth, 15%
minimum host free over three samples, one exclusive MPS worker and fallback
disabled. Evaluation produced 1,092 tokens from 16,678 prompt tokens; retention
produced 80 tokens. No downloads occurred.

Actual optimizer state is **11,618,304 bytes**: 11,534,336 int8 bytes plus
83,968 FP32 bytes. Initial and final extension weight files are each
**65,688,472 bytes**; no backbone copy was saved. The primary run occupies 128
MiB logically including both extensions, the reload run 596 KiB, and the bounded
report directory 216 KiB. Raw answers, unsuccessful trajectories, training
traces, profiles, guard samples and failure records remain preserved.

## Diagnostic path and preserved failures

The bounded tiny-cohort diagnostics motivated this single connector retry:
the 32-update SGD arm fit 0/8 training worlds; the 32-update 4096-wide arm fit
1/8; the 128-update 4096-wide arm fit **8/8 training worlds** but scored 0/8 on
fresh dev correct, wrong and no-image controls. Thus the pipeline can memorize
the tiny training cohort, while fresh-world generalization remains absent.

Wide-V1 attempts 01–05 remain preserved as invalid or superseded engineering
attempts covering protocol preparation, V0 producer identity, named admission,
visual-producer alignment and a final implementation-pin failure after code
changed mid-run. Attempt06 completed its primary native work, but restart06
exposed a missing reload artifact directory; score07 exposed the scorer field
bug. The directory fix required the clean attempt07/restart07 pair. These older
artifacts are failure evidence, not qualification results.

## Disposition and next boundary

Keep both V1 arms rejected. Do not tune against this dev set until it passes and
do not admit V2/LoRA. The next justified work is error/readout diagnosis over
fresh worlds and image-dependent controls, followed—only if that identifies a
distinct defect—by a separately preregistered bounded arm. Learned visual memory
and neural RSI remain blocked independently by perception quality and the
previously rejected learned-memory evidence.
