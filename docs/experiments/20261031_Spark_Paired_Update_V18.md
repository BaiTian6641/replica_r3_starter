# Spark paired-joint versus sequential-update diagnostic v18

Status: **V18.2 `preparation03` independently reviewed (scoped engineering), authorized, and executed once (`native_bf16_comparison_attempt01`); terminated with an execution/measurement failure (`ValueError: nonfinite gradients`) during the first backward-gradient reduction on MPS, invalidating the comparison. The MPS-to-CPU `float64` reduction defect is repaired and sealed in `preparation_attempt04` (`native_packet.json` SHA-256 `05cf86c28a0838e539256da54b3c87b7c43e67fc27b67a52a6cebbc4e4efed72`, snapshot `982a7f9921324b7031e43c24750fb93c98ebf1b7d5ab48ccc213fe1142883130`, `launch_authorized: false`).** Zero optimizer steps (`updates: 0`), zero generated tokens and zero development access occurred. This remains a fitting-only falsification diagnostic, not a candidate promotion or a revival of the falsified v16 content-key mechanism.

## Evidence and hypothesis

V17 attempt05 attempted to restore the same v16 terminal state before eight isolated single-world updates. Its unchanged 7/8 target-margin improvements and 7/8 partner-margin decreases used slots compressed before the update. These are frozen-slot conditional measurements, not corrected end-to-end producer–reader evidence. Gradient cosine was positive in two of four pairs; historical restore flags were constants. The [versioned qualification](20261031_Spark_Interference_V17.md#qualification-v172--31-october-2026) preserves those numbers. V18.2 repairs the measurement boundary before native execution.

The single hypothesis is: **including both counterfactual worlds in each optimization step reduces pair-level margin interference relative to sequential single-world updates at equal per-world supervised exposure.**

No conclusion about generalization, useful recall, persistent learning or a universal interference law is permitted.

## Fixed parent and data

- Official sealed BF16 Spark source, revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`; original parameters remain frozen and hash-checked.
- Parent extension and optimizer: `reports/intrinsic_memory_content_key_v16/native_bf16_two_attempt01/checkpoint`, manifest SHA-256 `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`, weights SHA-256 `66f9cec6ea2068a2432b055578039c716071a6bd6b0a3ba3ec97203e5cf40956`, optimizer SHA-256 `2ed73fa65963ada7182e5749937c6f73767f224a51de4cdb4ce9d0835ab2178b`.
- Fitting pairs only: IDs **0, 1, 4, 5**, worlds `v11_fit_00/01`, `02/03`, `08/09`, `10/11`. Fit file SHA-256 `106784510b4c26371c8a1f81736eea92443207d6028f63f1b491df0430dff92c`.
- Content-only keys, ordered values, source features, four slots, sites 3/7, post-block placement, NativeFusedAdamW, learning rate 0.0001 and initialization/execution seed 17 remain unchanged.
- Development rows and labels, generated answers, mechanism changes, compression fallbacks, exact-evidence retrieval and all later learning stages are excluded.

## One comparison

Both arms start from independently restored, identical parameters, buffers, module modes, optimizer moments and step counters, and CPU/applicable MPS/Python RNG states. Digest equality and identical common baselines are mandatory. Frozen original parameters are separately hash-checked. Each arm processes pairs in order 0, 1, 4, 5 and presents each world exactly once.

| Arm | Per-pair optimization | Optimizer steps per pair | Supervised examples per pair |
|---|---|---:|---:|
| Sequential reference | One complete-response update on each world; pairs 0/4 use left→right and pairs 1/5 use right→left | 2 | 2 |
| Paired joint | One update using the mean of the two worlds' complete-response negative log-likelihoods | 1 | 2 |

The joint loss is the unweighted mean of the two per-world mean negative total log-likelihoods. Each per-world loss includes its answer tokens and one EOS and is divided by its actual supervised-token count. No host-selected answer, future observation or alternative response enters the source/sense path.

The arms match initial state, data, per-world exposure, supervised examples and per-world forward/backward work. **Optimizer steps intentionally differ (8 sequential versus 4 joint across the fixture).** Batching and update count are therefore the combined treatment; the experiment cannot attribute an effect to batch size alone.

## Measurements

For each world and arm, preserve expected and alternative answer/EOS likelihoods using each candidate's own tokenization under the same memory context. Define total margin as expected total log-likelihood minus alternative total log-likelihood.

Record:

- Baseline, intermediate and final margins for all eight worlds after every step; nine sequential and five joint trajectory snapshots. Recompute live producer slots for each measurement; cache only identity-valid frozen backbone features. Include slot digests and per-pair minima.
- Every sequential target/partner transfer, joint and sequential losses, actual supervised tokens and optimizer-step counts.
- Exact parameter/optimizer/RNG restoration evidence, original parameter hashes, all forward/backward/update/token/write costs and retained artifacts.
- Actual per-module gradients and AdamW displacement; full-vocabulary argmax at every answer/EOS position under teacher forcing, with first-token/answer/EOS/complete correctness separately. This is not free generation.
- Raw unsuccessful as well as successful trajectories. No best-pair, best-prefix or post-hoc metric selection.

The primary endpoint is the mean over four pairs of the final per-pair minimum total margin. The cross-arm contrast is paired-joint minus sequential-reference. The guard metric is the count among eight worlds whose final margin is below its own baseline.

## Falsification and decision

Paired joint updating is **not supported as a remedy on this fixture** if either predeclared condition fails:

1. Its mean final pair-minimum margin must be strictly greater than the sequential reference's.
2. Its guard count must be no greater than the sequential reference's.

Equality fails the primary condition. Any missing arm, restoration mismatch, original-weight change, accounting overrun or development access invalidates the comparison rather than selecting a winner.

A better relative endpoint can leave both worlds incorrect. Report absolute final margins and full-vocabulary correctness even if both conditions pass; the result remains a four-pair fitting diagnostic. It does not promote content keys or open development. A successor would require fresh, predeclared fitting pairs and then the complete frozen terminal gates; no such successor is selected here.

## Revision 2 and preparation02 — before native execution

Revision 2 adds live-slot recomputation, measured complete-state restoration, full global trajectories, absolute/full-vocabulary evidence, actual displacement, and complete accounting. It fixes operational details before execution without changing the original paired/sequential contrast, primary endpoint or guard. V16 parent training is retained as `parent_training`; the active `training` contract now explicitly says 12 diagnostic updates and NativeFusedAdamW, rather than inheriting an ambiguous 512-update schedule. V16 used top-64 teacher KL; these arms use answer-plus-EOS CE with the parent's moments. This experiment changes neither moments nor optimizer.

Preparation01 was preserved as invalid after direct snapshot execution exposed an omitted relative package import. The closure builder was repaired; preparation02 contains 84 source files (1,998,331 bytes), including that dependency. Its CPU reproduction and frozen-runner binding review execute from the snapshot itself. Mechanical checks are not independent review or native execution.

Preserved preparation02 packet: [preparation_attempt02/native_packet.json](../../reports/spark_repairs_v18/preparation_attempt02/native_packet.json), SHA-256 `67951428c1a239293ce5ad2d009e8ed16f9fd0776e27ee1d21346eba59485c11`. Protocol: [spark_paired_update_v18_2_preparation02.json](../../configs/spark_paired_update_v18_2_preparation02.json), SHA-256 `0e7dddd94929f0d821558d8fc1e0e9f4ac0b100c1977265ad1f1578918bcdc44`. Snapshot manifest SHA-256: `455a9b1c236b1cf7aee90a596e49e4661eaefb0354b464105110f22eec977246`.

The 14 global measurements × 8 worlds × 2 candidates require 224 forwards; add 16 training and 8 frozen-source forwards for **248 forwards, 16 backwards and 12 updates**. Live producer computations are 112 measurement + 16 training = 128. Source cache size admits all eight canonical spans; only the eight frozen-source passes are reused. Input-token reservations are `240 × 256 + 8 × 512 = 65,536`; 8 source encodes, 8 canonical tokenizations, 16 response tokenizations, 2,048 source bytes, one model load and one parent restore are separately reserved. Every measurement, state check, source computation, update and artifact is recorded in the work ledger.

The worker reserves 180/240/1,200/60 stage seconds for admission/load/diagnose/finalize, totaling 1,680 inside a 1,800-second supervisor ceiling. The parent's separate pre-attempt admission adds 180 elapsed seconds. Parent supervision overlaps the worker, so total reserved stage seconds (3,660) are not elapsed seconds (ceiling **1,980**). Parent plus worker artifact caps are **79 MiB**, **5,478 writes**, and **26,153,982,825 artifact-read bytes**. The [repair report](20261031_Spark_Repairs_V18_Review.md#historical-preparation02-packet-and-derived-envelope) explains token, telemetry, ledger, source-verification and supervisor derivations.

Pair membership, the fit hash, content-key checkpoint/components, numerical contract, source/environment identities, complete implementation closure and budgets are validated before any model load or attempt marker. The child runs the immutable snapshot; altered working or snapshot source is refused. One native worker, 50 GiB sampled driver, 512 MiB sustained swap-growth threshold, 15% minimum host free, disabled CPU fallback and offline source loading are fixed. No base checkpoint is copied or downloaded.

## Preparation03 — correction-generation integration

The shared native controller belongs to the V18 source closure. Its earlier correction stage passed chronological text to generation after publishing a compact current view; the mocked lifecycle test missed the resulting canonical-span refusal. The repaired controller uses one explicitly bound current view and the same authenticated batch for contextual and exact-text generation, preserving complete history separately. The [follow-up report](20261031_Spark_Correction_Generation_V18_Review.md) records real-generation coverage and qualifies the prior completion claim.

Current packet: [preparation_attempt03/native_packet.json](../../reports/spark_repairs_v18/preparation_attempt03/native_packet.json), SHA-256 `a798464ab5826b0ff547755bfade6e90fa068b3b5bdb1185d161d4377237a9f9`. Protocol: [spark_paired_update_v18_2_preparation03.json](../../configs/spark_paired_update_v18_2_preparation03.json), SHA-256 `0487ff2f10c089510cb74eaf3bbbbdb60792ad2e5c0a8d6a31b8e8a01f4bd509`. Implementation manifest: `cd08bd3997d80469730edf1ce749bc59e949372917102c8b970d2ee8dcf1cdd6`; snapshot manifest: `3870a838a6104c9fe2740c3880c7c91776905002341fc3dcbccfcab5a9e3b16c`.

Working/snapshot closure equality is verified for 86 files/2,011,191 bytes. Both CPU validators run directly from the new snapshot, as does qualified-environment `--review`. The new affected suite passes 277 tests plus 7 source-admission regressions (325 affected/tiered tests plus 7 source-admission regressions in the supplied independent review). Preparation02 is byte-preserved; its historical 254+7 results are not the current test totals. Structural comparison confirms the same scientific contract and budgets, with only execution provenance, source/snapshot bindings and output paths changed. The snapshot again produces a synthetic −0.277678 joint-minus-sequential contrast on CPU.

## Native execution (`preparation03` / `native_bf16_comparison_attempt01`) — 1 November 2026

Under explicit human authorization bound to the supplied scoped independent review, `preparation03` was launched once through the supervised runner:

| Artifact | Path | SHA-256 | Size (bytes) |
|---|---|---|---:|
| Independent review receipt | [independent_review.json](../../reports/spark_repairs_v18/preparation_attempt03/independent_review.json) | `c484e121eb05ac29f9764539774c3eddf63498a76c740d5c4e4190ac136f7004` | 1,191 |
| Authorization receipt | [authorization.json](../../reports/spark_repairs_v18/preparation_attempt03/authorization.json) | `02647bf2d208c38ef9de9c4af38ee062940c71a1c53f81782f7f2968e7bf872e` | 2,442 |
| Attempt marker | [attempts.jsonl](../../runs/spark_paired_update_v18/attempts.jsonl) | `7271725a4c93114df14b44ecda838a549a050c9d90cd606e25e3d6021cc2ad67` | 489 |
| Parent preflight receipt | [receipt.json](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01_preflight/receipt.json) | `3d413ef1586dfdeecb5b91a516a48e40c6011467ffd49f68144832f83f78c59f` | 5,193 |
| Parent preflight ledger | [ledger.jsonl](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01_preflight/ledger.jsonl) | `87b8aad0b01a57e4af13d706962fa25bf7e7c79d6895349007f8a8ac32148d47` | 2,533 |
| Worker preflight receipt | [preflight.json](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/preflight.json) | `fdd6ed307f2acec488860ea08ed5864117fe5937f3f0a00133908de9070500f3` | 6,438 |
| Worker ledger | [work_ledger.jsonl](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/work_ledger.jsonl) | `25c56f73018b00a717093dfecc32a672655ec6340a2cf9d89d4161828323aa5e` | 50,245 |
| Worker failure receipt | [failure.json](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/failure.json) | `560e6181bd420a6975a5a2634df0730f44bcdc5dbd5dfa482cf749ab37580a85` | 137 |
| Supervision receipt | [supervision/receipt.json](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/supervision/receipt.json) | `bac73aa91541c0653700dc51b954a5f6adb14cd3f41d210d1b700d83011da4e8` | 1,803 |
| Driver telemetry | [driver_samples.jsonl](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/driver_samples.jsonl) | `7e9b8e424d4fe3f7269c402e718d017517ba9d7230d9e64d0241aad408d71b5c` | 1,208 |
| Host telemetry | [host_samples.jsonl](../../reports/spark_repairs_v18/preparation_attempt03/native_bf16_comparison_attempt01/host_samples.jsonl) | `0eb02a759cd433076176a642f4690a11fd836679166a3913318fa8624ccd7719` | 369 |

### Execution trace and classification

- **Classification**: **Execution / measurement failure (`ValueError: nonfinite gradients`), invalidating the comparison.** No arm completed, so neither endpoint values, below-baseline guard counts, final margins, final teacher-forced correctness, nor AdamW displacements were produced.
- **Stage progression**:
  - Parent `admission` completed in 5.130 s; parent `supervise` ran for 35.225 s (`status: "child_failed"`, `exit_code: 1`, `wall_seconds: 35.221745`).
  - Worker `admission` completed in 5.255 s (verified 8,238,996,247 source bytes, 450,061,586 environment-closure bytes across 67 files, 86 implementation files, fitting split `106784510b4c26371c8a1f81736eea92443207d6028f63f1b491df0430dff92c`, and V16 parent checkpoint `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`).
  - Worker `load` completed in 18.222 s (loaded official BF16 Spark `39ea150a4d2ed39184f2a83fbd474b67644e4d38f4039177c72cedd069ef4747` on `mps`, verified feature contract `b2b72df9c993b5cb8d5709804731144be6a640531a1388002b24e401ffbf67d1`, and restored V16 parent weights and `NativeFusedAdamW` moments).
  - Worker `diagnose` completed 8 prompt tokenizations (`id: 23–30`), 8 answer tokenizations (`id: 31–38`), `initial_state_capture` (`id: 39`, 0.014 s), `complete_state_restore` (`id: 40`, 0.524 s), and step-0 `all_world_measurement` (`id: 41`, 6.511 s: 8 `source_tokenize`, 8 `source_forward`, 8 `live_producer_slots`, and 16 `backbone_forward` passes across all 8 worlds).
  - On step 1 of the `sequential` arm (`pair 0`, world `v11_fit_00`), `pre_update_parameter_capture` (`id: 93`), `live_producer_slots` (`id: 94`), `backbone_forward` (`id: 95`), and `per_world_backward` (`id: 96`, 0.770 s) completed. Immediately afterward, `gradient_measurement` (`id: 97`, `module_statistics(path)`) completed in 0.036 s and `run_comparison` raised `ValueError: nonfinite gradients` at `src/replica_lab/intrinsic_memory_paired_update_v18.py:126`.
  - In the `finally:` block, `final_complete_state_restore` (`id: 98`, 0.470 s) executed and **succeeded**, proving that `restore_state(path, initial)` restored and verified exact digests for all parameters, buffers, module modes, optimizer moments, step counters, and CPU/MPS/Python RNG states on native MPS.
- **Resource and ledger totals**:
  - Sampled resource usage: 9 driver samples and 4 host samples; peak MPS driver `9,352,904,704` bytes (~8.711 GiB, cap 50 GiB), peak MPS active `8,415,151,616` bytes, peak RSS `769,294,336` bytes, swap used constant at `3,303,937,146` bytes (swap growth `0` bytes, cap 512 MiB), minimum host free `66.0%` (floor 15%).
  - Worker ledger (`work_ledger.jsonl`): 206 append/fsync calls (50,245 logical bytes, 0 torn tails, 0 interrupted events); reserved `forwards: 25`, `backwards: 1`, `updates: 0`, `input_tokens: 5,136`, `generated_tokens: 0`, `model_loads: 1`, `restart_restores: 1`, `source_encodes: 8`, `source_feature_forwards: 8`, `tokenizer_calls: 8`, `prompt_tokenizer_calls: 16`, `source_token_bytes: 636`, `cache_misses: 8`, `cache_evictions: 8`, `cache_recomputations: 8`, `writes: 31`, `artifact_bytes: 1,056,728` (observed output directory size `87,096` bytes), `artifact_read_bytes: 17,171,424,092`.
  - Parent preflight ledger (`native_bf16_comparison_attempt01_preflight/ledger.jsonl`): 11 append/fsync calls (2,533 logical bytes); reserved `writes: 33`, `artifact_bytes: 4,199,497` (observed directory size `7,726` bytes), `artifact_read_bytes: 8,873,607,209`.

### Root cause and selected next step

In `src/replica_lab/intrinsic_memory_measurement_v18.py`, `tensor_statistics` (line 99) and `module_statistics` (lines 112–113) convert tensors via a single `.to(device="cpu", dtype=torch.float64)` call:
```python
values = [tensor.detach().to(device="cpu", dtype=torch.float64) for tensor in tensors]
```
On CPU fixtures, `tensor` is already on `cpu`, so `.to(device="cpu", dtype=torch.float64)` performs a valid CPU `float32 -> float64` cast. On macOS arm64 MPS (`torch 2.9.0`), however, a single `.to(device="cpu", dtype=torch.float64)` call on an `mps` `float32` tensor invokes a combined device-and-dtype copy where Metal lacks `float64`, leaving the destination CPU `float64` buffer **uninitialized** (returning `0.0` or `NaN` from reused CPU heap pages) without raising an error. By contrast, a two-step copy (`tensor.detach().to(device="cpu").to(dtype=torch.float64)`) copies the `float32` tensor to CPU first and then widens to `float64` on CPU with zero discrepancy (`max_abs_diff = 0.0`).

**Next step**: Repair `tensor_statistics` and `module_statistics` in `src/replica_lab/intrinsic_memory_measurement_v18.py` to use two-step `.to(device="cpu").to(dtype=torch.float64)`, add a unit regression covering MPS-to-CPU `float64` reduction, and seal `preparation_attempt04` for independent review and one replacement native V18.2 run. Because `native_bf16_comparison_attempt01` was invalidated by an execution/measurement defect rather than a valid scientific comparison outcome, neither the conditional CE-versus-KL alignment protocol nor a bounded fitting successor is triggered yet.

## Preparation04 — MPS-to-CPU `float64` reduction repair (1 November 2026)

`src/replica_lab/intrinsic_memory_measurement_v18.py` (`_cpu_float64`, `tensor_statistics`, `module_statistics`) and `src/replica_lab/intrinsic_memory_probe_v18.py` (`probe`) now copy tensors to CPU before widening to `float64` (`tensor.detach().to(device="cpu").to(dtype=torch.float64)`). Direct verification on macOS arm64 MPS (`torch 2.14.1`) confirms that before the fix, `module_statistics` on an MPS `float32` linear layer produced `l2: inf, max_abs: 1.2629876359675347e+308`; after the fix, MPS and CPU match (`grad l2 = 150.972588`, `disp l2 = 0.00644987`, `probe(feats_mps) == probe(feats_cpu)`), covered by `test_cpu_and_mps_float64_reduction_and_probe_parity` in `tests/test_spark_repairs_v18.py`.

| Binding | Path | SHA-256 |
|---|---|---|
| Packet | [preparation_attempt04/native_packet.json](../../reports/spark_repairs_v18/preparation_attempt04/native_packet.json) | `05cf86c28a0838e539256da54b3c87b7c43e67fc27b67a52a6cebbc4e4efed72` |
| Protocol | [spark_paired_update_v18_2_preparation04.json](../../configs/spark_paired_update_v18_2_preparation04.json) | `383891eed3fff80158780f0d932fabadaf35d990edc75f8166b71b0fd53b2ad9` |
| Implementation manifest | [preparation_attempt04/implementation.json](../../reports/spark_repairs_v18/preparation_attempt04/implementation.json) | `c2dff9142db0ec64fb9ed9dd658fdf88840d54fb25cf4b951c76ec97d03c279d` |
| Snapshot manifest | [preparation_attempt04/source_snapshot.json](../../reports/spark_repairs_v18/preparation_attempt04/source_snapshot.json) | `982a7f9921324b7031e43c24750fb93c98ebf1b7d5ab48ccc213fe1142883130` |

`preparation_attempt04` seals 86 source files (`2,011,589` bytes, mode `0444`). Both frozen-snapshot CPU validators (`reports/spark_mps_reduction_v18/cpu_validation_snapshot04`, `reports/spark_mps_reduction_v18/generation_validation_snapshot04`) and qualified-environment `--review` pass from `preparation_attempt04/source_snapshot`.

### Native execution (`preparation_attempt04` / `native_bf16_comparison_attempt01`) — 1 November 2026

Following independent review ([independent_review.json](../../reports/spark_repairs_v18/preparation_attempt04/independent_review.json), SHA-256 `d8dc8b570bb9d9134b616d450808e892deb532de52cc50161427e095fcc13c63`) and human authorization ([authorization.json](../../reports/spark_repairs_v18/preparation_attempt04/authorization.json), SHA-256 `e342589099949efe55bf279c3ea8ea0486fd36a771190529e77d0c09cf13c63`), `preparation_attempt04` executed once and completed cleanly ([result.json](../../reports/spark_repairs_v18/preparation_attempt04/native_bf16_comparison_attempt01/result.json), [comparison.json](../../reports/spark_repairs_v18/preparation_attempt04/native_bf16_comparison_attempt01/comparison.json)):

- **Endpoints & Difference (`mean_final_pair_minimum_margin`)**:
  - `sequential` (`8` steps): `-1.8485417813062668` (pair minima: `[-1.750997, -0.995533, -4.140637, -0.507000]`)
  - `paired_joint` (`4` steps): `-0.5075049996376038` (pair minima: `[-0.105478, -0.934160, +0.113274, -1.103656]`)
  - `joint_minus_sequential`: `+1.341036781668663` (`> 0.0`)
- **Below-Baseline Guard Counts**: `sequential = 4/8`, `paired_joint = 4/8` (`4 <= 4`). Both predeclared relative conditions passed (`relative_conditions_passed: true`).
- **Absolute Final Margins & Teacher-Forced Correctness**:
  - Step-0 baseline: `2/8` first-token and `2/8` complete teacher-forced correct (`0/4` pairs both correct).
  - `sequential` final (step 8): margins `[+2.347677, -1.750997, -0.995533, +1.676793, -4.140637, +4.699225, +1.388678, -0.507000]`; `4/8` positive-margin worlds, `4/8` first-token correct, `4/8` complete teacher-forced correct, `0/4` pairs both correct.
  - `paired_joint` final (step 4): margins `[-0.105478, +0.402353, +1.851906, -0.934160, +0.335941, +0.113274, -1.103656, +2.703218]`; `5/8` positive-margin worlds, `5/8` first-token correct, `5/8` complete teacher-forced correct, `1/4` pairs both correct (Pair 2 `v11_fit_08/09`).
- **Gradients, Displacements, Restoration & Resource Totals**:
  - All 12 updates produced finite `float64`-reduced gradients (`producer l2` `1.86–42.91` in `sequential`, `1.60–28.58` in `paired_joint`; `reader l2` `10.14–116.97` in `sequential`, `7.79–74.23` in `paired_joint`) and finite `NativeFusedAdamW` displacements (`producer l2` `0.0158–0.0306`, `reader l2` `0.0098–0.0234`).
  - `original_weights_unchanged: true`; all 9 state digests (`buffers`, `cpu_rng`, `module_modes`, `mps_rng`, `optimizer`, `optimizer_moments`, `optimizer_steps`, `parameters`, `python_rng`) restored identically between arms and at finalize.
  - Resource usage: `113` driver samples (peak `11,148,312,576` bytes / `10.38 GiB`, cap `50 GiB`), `30` host samples (min free `63.0%`, floor `15%`), swap growth `0` bytes.
  - Ledger totals: `248` forwards, `16` backwards, `12` updates, `62,224` input tokens, `0` generated tokens, `0` retrievals, `164` writes (`592,642` observed artifact bytes), `0` failed/interrupted events, `0` torn tails.
- **Outcome & Follow-up Preparation**: `paired_joint` achieved a valid relative improvement over `sequential` while absolute recall remains inadequate (`mean_final_pair_minimum_margin = -0.507505 < 0.0`, `5/8` worlds and `1/4` pairs both correct; `8` vs `4` steps remain part of the treatment difference). The V19 fresh bounded paired-joint fitting & matched CE-vs-top-64-KL alignment packet is prepared at [reports/spark_alignment_v19/native_packet.json](../../reports/spark_alignment_v19/native_packet.json) (`a6178861b182a97ebd0fc800752d3d05d6884dc279567491c7c1186d766ba3d7`, `launch_authorized: false`).
