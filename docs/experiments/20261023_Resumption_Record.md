# Resumption record — 23 October 2026 UTC

## Repository

- Branch: `codex/d2-capacity-runtime-pilot`
- Last code/test HEAD: the 512-unit capacity qualification; documentation follows.
- Full suite at the code HEAD: **1171 passed, 21 subtests**.
- Worktree: protected `src/replica_lab/runtime_pilot.py` remains untracked and
  unmodified; generated guard logs were removed. No native job is active.

## Selected artifacts

- Sealed original: `checkpoints/minicpm5_source` (revision
  `f97400052a43d642bbc6e9975e2397e3ae6a6b52`), unchanged.
- E2 parent extension: `runs/e1_e2_native_attempt02_20261023/selected_extension`
  (v1; used for the 32-unit capacity migration).
- E3 accepted core/action extension:
  `runs/e3_set_attempt03_20261023/selected_extension` (v2).
- E3 mixed family artifact:
  `runs/e3_policy_head_attempt03_20261023/policy_head` (head rejected as
  no-benefit).
- Latest selected-capacity evidence:
  `reports/e2_capacity512_selected_attempt01/run_report.json` (512 units,
  14.62 GiB, 76.0 s).
- Latest negative action evidence:
  `reports/e3_guard_attempt01/` and
  `[E3 action guard](20261023_E3_Action_Guard_Results.md)`.

## Last passed gates

- E1/E2 native interfaces, eligibility, exact fallback and selected-extension
  32-unit capacity (via validated `ports.max_units`-only migration).
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
   now **17/17 (15/15 held-out)** vs random 0/11. It is opt-in, narrow, and not
   a free-form policy generalization.
2. **Learned C13 director** is now data-gated. The checked-proposal registry has
   **13 independent groups, 5 accepted and 8 rejected/deferred**, below the
   declared 10/10 minimum. The gate returns
   `defer_insufficient_checked_proposals`; no selector weights were trained or
   published. The deterministic director/adapter/audit remain available.
3. **C17 native cache reuse** is not implemented; admission and an exact-key
   bounded host-payload cache exist, but fresh re-prefill remains the reference.
4. **E4 learned delayed-use/practice** candidates were rejected; fast memory and
   artifacts are implemented, but learned benefit is unproven.
5. **C16 authentic conditioning/fidelity** requires permitted personal source
   material; no personal data is bundled. A fictional scoring harness now exists,
   but no authentic baseline run or human review has occurred.
6. **E6 multimodal** requires local MiniCPM-V-4.7-1B weights and a separate
   adapter; unavailable.
7. **D2 full-backbone action tuning** was tested with bf16 compute: 8 updates
   completed at **39.10 GiB**, but held-out/dev completed **0/6** while repeating
   `read`; no export. FP32 D2 exceeded 40 GiB. Full-backbone training does not
   rescue the current action interface.

## Next executable action

Choose **one** new action-policy mechanism, declare a fresh family and protocol
before execution, and do not reuse the exposed guard cases for tuning:

- a trained D1/D2 action-policy candidate with an explicit optimizer/residency
  plan and independent held-out evaluation, or
- a genuinely different action representation with fresh supervision; do not
  repeat the prompt guard or token-mask interface.

Require actual independent task completions, 0 false success claims, restart
continuation, and measured tool/token/memory costs. A failed arm must preserve
raw traces and not be reported as a general action-policy capability.

## External dependencies

- Authentic, permitted personal records for C16 conditioning and fidelity.
- Local MiniCPM-V weights/processor for E6 and native visual memory tests.
- Any larger-model or RTX PRO 4000 Blackwell backend qualification.
