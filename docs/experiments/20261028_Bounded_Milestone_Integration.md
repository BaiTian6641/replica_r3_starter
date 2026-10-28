# Bounded milestone readiness and clean integration

**28 October 2026. Status: implementation, independent preparation and clean
CPU integration are complete. No native protocol was authorized or executed;
no learned capability is qualified.** Integration checkout:
`42d122b7a181c4076f5294b7d592dc35490e2158`. Readiness protocol commit:
`454f89eabb8b61b149928ce547263130b42df117`.

## 1. Delivered implementation and preparation

| Package | Delivered evidence | Boundary |
|---|---|---|
| DC-P1 staged evidence | `42a17830`: production/fixture classification, independently resolved artifacts, decode/checkpoint linkage, recomputed scores and lazy current-stage label access. | CPU orchestration and production-factory test seams only; no native training. |
| DC-P1 learning audit | `8ac42b3f`: reproducible small-CPU audit and bounded source-free feature-alignment protocol. | Diagnostic hypothesis only; alternative runner remains unimplemented. |
| DC-R1 native readout | `deba2428`, `14b6ecf7`: sealed-source no-cache layers 13/27/41, staged scoring, raw trajectories, bounded resume and launch preflight. | Production path implemented but no MPS run or native readout result. |
| Artifact enforcement | `4aca7b02`: live replacement overlap accounting with a 64 KiB refusal/recovery regression. | Software enforcement, not model evidence. |
| Vision runtime | `cdf168a2`: runtime/checkpoint parameter mapping and real tiny-SigLIP gradient/cache tests. | No native vision arm or quality claim. |
| Spark and data preparation | Existing 721/721 migration and 500/500 final-product checks; exact nine-row caption admission, 21 exclusions, eight exposed rows and 24 diagnostic scenes. | Analytical/data evidence only. All ten native gates, RSI and personality remain unresolved. |

## 2. Learning hypothesis and justified alternative

The preserved CPU artifact is
`reports/dc_p1_learning_audit_20261028/audit_attempt01.json`:

- Answer-span and full-response masks coincide. Treatment loss and gradient are
  exactly **1.5×** the matched control: this is global CE rescaling, not
  span-specific supervision.
- The real fused-AdamW path changes logits at epsilon `1e-8`, produces no logit
  change at epsilon `1e-2`, and clips the amplified gradient to norm **1.0**
  with scale **0.67108864**.
- Source-conditioned versus source-free feature shift is **0.6931585073**;
  selected-expert enabled/disabled maximum logit delta is **2.0**.
- All four tiny generated-behavior probes have **zero changed tokens**. Logit
  movement is not evidence of generated-answer improvement or useful learning.

Only the bounded `dc_p1_source_free_feature_alignment_v1` alternative was
predeclared. It uses six fit events in two source groups, four matched arms and
an added source-free query CE term; development/final labels stay closed. Its
runner is `not_implemented`, so there is no native command, training result or
permission for a blind retry.

## 3. Exact readiness and budgets

`reports/bounded_milestone_20261028/readiness_attempt01.json` binds every
protocol/config/manifest/runner/audit/preparation input to commit `454f89ea…`.
It records review and launch-preflight commands, fresh-output status and all
missing authorization fields. `native_command` is null.

| Budget | DC-R1 | Source-free alternative |
|---|---:|---:|
| Optimizer updates | 0 | 18 |
| Source-free query forwards | — | 18 |
| Generation requests | 240 | 24 |
| Decode forwards | 1,680 | 360 |
| Forward calls | 1,920 | 424 |
| Processed positions | 252,480 | 18,860 text positions |
| Checkpoint writes | 0 | 19 |
| Retries | 0 | 0 |
| Artifact cap | 33,554,432 B | 33,554,432 B |

DC-R1 also declares one model/tokenizer load, zero backward/optimizer work,
1,800-second wall ceiling and the unchanged resource guards. The alternative
declares two model/tokenizer loads plus one bounded restart, 1,200-second wall
ceiling and the same guards. These are ceilings, not measured native costs.

The DC-R1 receipt is absent; all seven v1 authorization fields are missing. The
alternative receipt is absent; all twelve v2 fields are missing, and its runner
is separately unimplemented. No receipt was synthesized. Native execution must
remain stopped until an exact external receipt admits one guarded command.

## 4. Clean integration evidence

The detached checkout at `42d122b7…` completed the following without native
execution:

| Check | Result |
|---|---:|
| Pytest | **494 passed, 1 skipped**; 495 collected, zero failures/errors |
| Pytest time | 387.53 s |
| Spark v5.1 verifier, `--tracked require` | 721/721 |
| Spark final-product verifier | 500/500; zero unavailable source-cache entries |
| Caption validator | valid; 9 admitted / 21 exclusions / 8 exposed / 24 diagnostic |
| Tracked artifact guard | passed; 11,371 files; largest 8,148,167 B; no violations |

The exact 29-file command and validator commands are preserved in
`reports/verification_20261028/bounded_integration_clean_summary.json`.
JUnit is
`reports/verification_20261028/bounded_integration_clean_attempt01.xml`,
SHA-256 `2af720088546b6ec1de5d96a5c26becdb6b8a2ed303c76b94cf95dc816e0fa40`.
The summary was checked against the XML counts and digest. The disposable
checkout was clean and removed; earlier failed attempts remain preserved.

## 5. Qualification and next gate

Implemented and CPU-executed work is not native training, calibrated confidence,
useful retrieval, visual competence, measured Spark residency or personality
evidence. No MiniCPM5 native model was loaded, no optimizer-backed native run
occurred, and no downloads or duplicate base saves were introduced.

The next native action is limited to one externally authorized, capacity-guarded
DC-R1 attempt after exact commit/receipt/output review. The source-free
alternative requires its own runner implementation and review first. Any native
failure must be preserved; sealed-final tuning, rejected-arm retries, RSI,
outer-loop, self-distillation, promotion and personality adaptation remain
closed. The protected `src/replica_lab/runtime_pilot.py` was not read, modified,
imported or staged.
