# Resumption record — 23 October 2026 UTC

## Repository

- Branch: `codex/d2-capacity-runtime-pilot`
- Last code/test HEAD: `2a971c7 experiment: record negative action-space guard`; documentation-only commits follow.
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
- Latest 32-unit selected capacity evidence:
  `reports/e2_capacity32_selected_attempt01/run_report.json`.
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

1. **E3 model action selection remains negative.** Fresh `set_constant` guard
   and control both completed **0/4**; the model repeated `read` even when the
   guard denied it. The prompt/interface guard is not enough.
2. **Learned C13 director** has no trained ranking policy or checked-proposal
   dataset. The deterministic baseline is not a capability result.
3. **C17 native cache reuse** is not implemented; admission only, fresh
   re-prefill remains the reference.
4. **E4 learned delayed-use/practice** candidates were rejected; fast memory and
   artifacts are implemented, but learned benefit is unproven.
5. **C16 authentic conditioning/fidelity** requires permitted personal source
   material; no personal data is bundled.
6. **E6 multimodal** requires local MiniCPM-V-4.7-1B weights and a separate
   adapter; unavailable.
7. **D2 full augmented tuning** is not declared or run; a separate bounded
   memory/optimizer protocol is required if full-backbone action training
   becomes necessary.

## Next executable action

Choose **one** new action-policy mechanism, declare a fresh family and protocol
before execution, and do not reuse the exposed guard cases for tuning:

- constrained action-kind decoding (with argument generation kept separate), or
- a D1/D2 separated action-policy candidate with an explicit optimizer/residency
  plan and independent held-out evaluation.

Require actual independent task completions, 0 false success claims, restart
continuation, and measured tool/token/memory costs. A failed arm must preserve
raw traces and not be reported as a general action-policy capability.

## External dependencies

- Authentic, permitted personal records for C16 conditioning and fidelity.
- Local MiniCPM-V weights/processor for E6 and native visual memory tests.
- Any larger-model or RTX PRO 4000 Blackwell backend qualification.
