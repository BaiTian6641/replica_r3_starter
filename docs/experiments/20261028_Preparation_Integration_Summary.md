# Preparation and clean-integration summary

**28 October 2026. Status: the current repair/preparation increment is committed
and clean-verified. No native protocol was authorized or executed.** The final
integration checkout is `96a809c261492af40e90c9c9c1fff2dd72b7dbc2`.

## 1. Delivered packages

| Package | Implemented and measured | Native status |
|---|---|---|
| DC-P1 v2 | Selected-expert runner, source-free P-only evaluation, matched controls, event/checkpoint and temporal semantics, staged receipts, partial resume, live publication. CPU attempt03 completes all stages and recovery; 121 writes/four retained checkpoints. | External authorization absent. No MPS run or learned result. |
| DC-R1 v2 | Versioned fixture, all-layer abstention and per-layer actual-output scoring, corrected sequential work schedule, 24 tests and review receipt. | No native runner, authorization or compressed-channel receipt. Host reconstruction is only a control. |
| Vision v2 | Separate v2 runner, corrected accounting identities, deterministic packet/readiness, conditional 24-tensor co-adaptation and producer-version cache invalidation. V1 lineage preserved. | Smaller-step receipt absent; co-adaptation additionally lacks failure and frozen-feature diagnosis. No native arm. |
| Value readout recovery | Durable recovery, publication and artifact accounting with 11 dedicated tests. | CPU integration only; not learned retrieval. |
| Spark v5.1 | Reviewed migration, analytical residency, 721 tracked checks and 500 final-product checks. | All ten native gates unresolved; analytical estimates are not measured fit. |
| Caption/data lineage | Exact nine-row admission, 21 exclusions, eight exposed rows and 24 diagnostic scenes with partition identity and disjoint roles. | Data admission only; no model execution or visual-quality claim. |
| Deferred learning | Useful-learning, teacher-competence, full-cost, retention, rollback and separate personality gates documented and enforced by the current designs. | No RSI, self-distillation, outer-loop, personality or promotion run. |

Implementation and evidence commits include `792c878f` (DC-P1 protocol),
`f03428c8` (clean dependency repair), `e8579c04` (vision v2), `ed923e01`
(DC-R1), `a0453b3f` (value recovery), `d65a36ff` (Spark/caption migration) and
`48e8d49e` (passing Spark receipts). Handoff/report commits follow through
`96a809c2`.

## 2. Verification evidence

| Scope | Result |
|---|---:|
| Broad DC, v1-lineage and capacity suite | 136 passed |
| Clean DC-P1 checkout at `f03428c8` | 112 passed |
| Vision workspace suite | 184 passed |
| Combined clean vision/DC-P1 checkout at `e8579c04` | 296 passed |
| DC-R1 focused suite | 24 passed |
| Spark/caption suite | 112 passed, 1 skipped |
| Final clean integration at `96a809c2` | **443 passed, 1 skipped** |
| Spark v5.1 `--tracked require` | 721/721 |
| Spark final-product verifier | 500/500 |
| Caption validator | valid; exactly nine admitted rows |

Final receipt and exact 20-file pytest command:
`reports/verification_20261028/final_integration_clean_summary.json`.
JUnit SHA-256:
`0a7d9bfb08c076c303e7efaa63248f6773cca4cff46a3740c66fe030cc8dd746`.
The checkout was clean after verification and removed afterward. No model loads
or native execution occurred.

## 3. Preserved negative results

- DC-P1 clean attempt 01 selected an uncommitted test and collected zero tests;
  attempt 02 exposed the missing publication dependency. Both failures and XML
  are preserved before the 112-test pass.
- Vision's pre-split change invalidated v1 implementation pins: 16 failures and
  six errors. V1 was restored and v2 separated; no test was re-pinned.
- Spark final product initially failed 499/500 on unstaged inputs; the receipt is
  preserved before the committed 500/500 result.
- Caption's stale-evidence baseline failed before the fixture was refreshed; the
  exposed-row refusal and final 112-test result are preserved.
- CPU evidence attempts 01/02 failed on host-retained older modules; fresh-process
  attempt03 and both failures are preserved.

Failure summaries live under `reports/verification_20261028/` and the DC-P1 CPU
attempt directories. No failed native trajectory is claimed because no native
job was launched.

## 4. Inconclusive and pending

- Learned response quality, fresh-instance benefit, visual competence, native
  compressed readout, abstention calibration and persistent-memory quality are
  unmeasured.
- Native memory/runtime/artifact costs for MiniCPM5 and full Spark are unmeasured.
  CPU fixture byte counts are not native costs.
- Official BF16/FP8, dequantized derivatives and native FP8 remain separate;
  no native FP8 result exists.
- The next native action requires a separate external receipt bound to the exact
  current commit, a clean tracked worktree and the declared guard/output. DC-R1
  additionally needs a real native runner and compressed-channel receipt;
  co-adaptation needs the smaller-step failure and diagnosis first.
- Do not tune against sealed final sets or retry rejected arms. Finish any future
  bounded native attempt once, preserve failures and evaluate independently.

The protected untracked `src/replica_lab/runtime_pilot.py` was not read, modified,
imported or staged for this increment.
