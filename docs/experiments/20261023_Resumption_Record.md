# Resumption record — 23 October 2026 UTC

**24 October handoff:** the entries below preserve the earlier package.
Use [execution state](../../execution_state.md) and
[memory reliability recovery](../architecture/17_Memory_Reliability_Recovery.md)
for the current order and actual fine-tuning outcome. The active recovery
repairs benchmark/scorer integrity, persistent correction and retention before
advancing RSI; MiniCPM5 and the 50 GiB guard remain the baseline.

## Repository

- Branch: `codex/d2-capacity-runtime-pilot`
- Base HEAD before this package: `fd9ea17`; code commit `6e8ec34` adds the
  capacity, action-menu and scheduler corrections; documentation commit
  `7a7d03d` adds the research cross-check and status reconciliation;
  `377718e` records the package commits. Current documentation review is
  against `377718e`; this revision updates documents/status plus the active
  guard default and focused regression coverage.
- Last recorded full suite after the code package: **1211 passed, 21 subtests**
  in 147.24 s. It was not rerun. The 50 GiB guard change passed **81 focused
  tests in 23.87 s**; exact command is in execution state.
- Worktree: protected `src/replica_lab/runtime_pilot.py` remains untracked and
  unmodified. This plan/guard update launches no native model job.
- The 8192-unit 48.08 GiB boundary and 4096-unit 30.59 GiB peak were measured
  with the pre-fix inference harness. Writer/sense now run under
  `torch.no_grad()`; corrected inference must be remeasured.
- Current default: **50 GiB = 53,687,091,200 bytes**; both preflight wrappers
  derive it from `CapacityPolicy`. The 512 MiB swap-growth, 15% host-free and
  three-sample checks remain. Historical explicit 40 GiB protocols and the pinned
  D2 resume launcher remain unchanged; new studies need new matching pins/policy.
- Keep MiniCPM5 as the baseline. Add optional Spark-X2.6-4B BF16 comparison,
  MiniCPM-V native vision/memory/reasoning training, staged new memory layers
  and bounded RSI. These are plans with no new native capability claim.

## Selected artifacts

- Sealed original: `checkpoints/minicpm5_source` (revision
  `f97400052a43d642bbc6e9975e2397e3ae6a6b52`), unchanged.
- E2 parent extension: `runs/e1_e2_native_attempt02_20261023/selected_extension`
  (v1; used for the 32/64/128/256/512/1024/4096-unit capacity migrations).
- E3 accepted core/action extension:
  `runs/e3_set_attempt03_20261023/selected_extension` (v2).
- E3 mixed family artifact:
  `runs/e3_policy_head_attempt03_20261023/policy_head` (head rejected as
  no-benefit).
- Latest selected-capacity evidence:
  `reports/e2_capacity4096_selected_attempt01/run_report.json` (4096 units,
  30.59 GiB, 559.4 s). The 8192-unit probe is a guard boundary
  (`reports/e2_capacity8192_selected_attempt01/failure_report.json`), not a pass.
- Latest negative action evidence:
  `reports/e3_guard_attempt01/` and
  [E3 action guard](20261023_E3_Action_Guard_Results.md).

## Last passed gates

- E1/E2 native interfaces, eligibility, exact fallback and selected-extension
  capacity through **4096 units** (via validated `ports.max_units`-only
  migrations); 8192 remains an attempted guard boundary, not a pass.
- E3 engineering: real actions, independent evaluator, journal, restart,
  stale-workspace rejection, mixed-family 4/4.
- C16 persistent identity store and optional native identity reader: null,
  disabled, all-masked parity and v3 checkpoint binding; **untrained**.
- C13 deterministic director baseline and C17 single-job admission.
- Focused software tests for all of the above.

## Current failures / unfinished work

1. **E3 free-form action selection remains negative.** The original family was
   0/7; guard/control 0/4; constrained decoding 0/1; a 12-train family was 0/9;
   DAgger degraded 2/5 to 0/5; and full-backbone bf16 was 0/6. A declared
   **candidate-ranking interface** has replicated: model **11/11** episodes
   (9/9 held-out) vs same-menu random **0/11**, with zero false success across
   two small fresh families. The menu/scoring component is reusable and its
   post-refactor smoke reproduced 6/6 (5/5 held-out). Fresh-process restart
   with values already contradicted by the observed failure completed **3/3**
   on a new fresh family (process 1 was 0/3). A canonical/shuffled menu control
   both scored **3/3**, ruling out a fixed position. Aggregate repair ranking is
   historically **17/17 (15/15 held-out)** vs random 0/11. It is opt-in and
   narrow. The numeric-string correction changes candidate sets; fresh
   matched controls on the corrected menus are pending. These old results
   do not establish corrected-interface or free-form generalization.
2. **Learned C13 director** is now data-gated. The checked-proposal registry has
   **13 independent groups, 5 accepted and 8 rejected/deferred**, below the
   declared 10/10 minimum. The gate returns
   `defer_insufficient_checked_proposals`; no selector weights were trained or
   published. The deterministic director/adapter/audit remain available.
3. **C17 native cache reuse** is not implemented; admission and an exact-key
   bounded host-payload cache exist, but fresh re-prefill remains the reference.
4. **E4 learned delayed-use/practice** candidates were rejected: original and
   contextual learned writers each scored 0/8 held-out. Native read ports and
   differentiable single-page training exist. The original writer repeats
   one matrix across heads; a functional multi-chunk trainer, distinct head
   writes and useful learned delayed adaptation remain open.
5. **C16 authentic conditioning/fidelity** requires permitted personal source
   material; no personal data is bundled. A fictional scoring harness now exists,
   but no authentic baseline run or human review has occurred.
6. **E6 multimodal** requires local MiniCPM-V-4.7-1B weights and a separate
   adapter; unavailable.
7. **D2 full-backbone action tuning** was tested with bf16 compute: 8 updates
   completed at **39.10 GiB**, but held-out/dev completed **0/6** while repeating
   `read`; no export. FP32 D2 exceeded 40 GiB. This bounded eight-update
   negative does not rule out co-adaptation for a new memory mechanism or
   replace the required fair developmental D0/D1/D2 comparison.

## Next executable action

Follow [the M0–M7 construction order](../architecture/15_Research_Crosscheck_and_Backbone_Plan.md)
and [the complete loop-agent instructions](../architecture/16_Loop_Agent_Instructions.md):

1. **M0a:** remeasure corrected MiniCPM5 capacity and fresh matched action-menu
   controls. Do not treat the old 48.08 GiB boundary as the corrected limit.
   Optional M0b qualifies a separate Spark BF16 adapter and compares native
   backbones before augmented arms; it does not stall independent MiniCPM5 work.
2. **M1:** fresh chronological episodes and a thin typed compiler path that
   reuses existing execution, publication and rollback.
3. **M2:** one useful SGDN-2-inspired adaptive path with distinct head writes,
   functional multi-chunk training and delayed-use controls. Use modules-only
   training first to isolate the mechanism, with an explicit bounded protocol.
4. **M3:** shared/private-depth comparison only after the update engine,
   chunk-start visibility and reducer are defined and the simpler memory path
   has useful evidence. Do not confound a changed engine with shared state.
5. Continue M4 dynamic routing, M5 developmental/RSI/longitudinal comparisons,
   M6 learned multi-rate levels and M7 native vision/visual memory as gates allow.
   RSI starts with verified practice and isolated SFT, then learned selection;
   code/architecture proposals are a later externally evaluated scope.

Identity source-to-slot preparation and equal-information controls run
alongside M1–M3 when authentic permitted source is available. Calibrate the
success/uncertainty head before control use. Every native arm retains MPS,
one worker and the new 50 GiB / 512 MiB / 15% default guards, unless replaying
an explicitly pinned older policy. Preserve failures and raw
traces; do not repeat a negative action arm without a new hypothesis.

## External dependencies

- Authentic, permitted personal records for C16 conditioning and fidelity.
- Local MiniCPM-V weights/processor for E6 and native visual memory tests.
- Pinned Spark BF16 source/native adapter for the optional comparison.
- Any larger-model or RTX PRO 4000 Blackwell backend qualification.
