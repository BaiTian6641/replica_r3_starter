# M3 retention repair — bounded candidate rejected

24 October 2026 UTC. Protocol `c4355ff` (`configs/m3_retention_repair_v1.json`).
**Decision: quality reject, not engineering defer.** The shared repair candidate
preserved delayed behavior but did not repair general noninterference. The
original Paris/H2O check and fresh-process restart were correctly skipped.

## 1. Parent versus candidate

| Endpoint | Frozen shared M3 parent | Repair candidate | Gate result |
|---|---:|---:|---|
| Delayed development | 4/4 | 4/4 | Preserved |
| Fresh delayed hold-out | 6/8 | 6/8 | Preserved, above all null controls |
| Original delayed regression | 8/8 | 8/8 | Preserved |
| Final general correct | 0/4 | 1/4 (`Cl`) | Strict improvement only |
| General development correct | 0/2 | 0/2 | No improvement |
| Final adversarial memory | 0/4 (`teal`) | 0/4 (`teal`) | **Below disabled 2/4: reject** |
| Development adversarial | 0/2 | 0/2 | No improvement |
| Disabled-base token IDs | Reference | Identical | Pass |

Correct delayed held-out was strictly above linear, shuffled, zero and disabled,
all 0/8. The candidate's only general gain was `Cl`; zero, disabled and shuffled
also supply that answer. Full context answers 4/4, confirming answerability.
Adversarial memory produced constant `teal` for both parent and candidate.

### Delayed hold-out strata

The 6/8 aggregate hides the remaining novel-value failure:

| Stratum | Parent | Candidate |
|---|---:|---:|
| Seen value | 4/4 | 4/4 |
| Novel value | 0/2 | 0/2 |
| Changed template | 2/2 | 2/2 |

There were no delayed parent→candidate correctness flips across the 20 evaluated
cases. This does not establish novel-value learning.

## 2. Engineering and exposure evidence

- All 24 delayed cases appeared in 32 training exposures; all eight general
  replay cases appeared in 32 exposures. This corrects the earlier schedule that
  would have covered only 12 delayed cases.
- Training exact: delayed 22/24, replay 8/8. Replay development is 0/2, showing
  replay-set fitting rather than general noninterference.
- Update-two first/later gradients are present and finite at both depths; every
  `mean_loss` equals `loss_sum / 4`.
- All 20 corrections have changed state and correct independently checked
  answers. Same-process image reload passes.
- Because the candidate was rejected, the original Paris/H2O failure was not
  evaluated and fresh-process restart was not run. These remain open, not passes
  or new failures.
- Independent review confirmed the reject and found no verdict-changing
  arithmetic error. It also identified the strata omission now reported above.

## 3. Costs

- 16 updates, 64 exposures, 386 answers, 1,220 generated tokens and 25,714 prompt
  tokens; maximum prompt 128 tokens.
- Profiled wall 141.412 s; guard wall 143.969 s; sampled peak driver
  **11.0043 GiB**; zero swap growth and minimum host free 76%.
- Optimizer state 7,485,456 bytes: 7,340,032 int8 and 145,424 FP32.
- One extension and one depth image; no base copy, download, remote push or
  further replay arm.

## 4. Decision and next mechanism

This candidate is not accepted and its weights are not an accepted parent for
further work. Replaying the same recipe again would be retuning after a negative
result, not a new hypothesis. M4 may proceed only as a **different mechanism
comparison** because the original M3 ownership gate passed; retention remains
explicitly open and no integrated retention milestone is claimed.

Evidence: `runs/m3_repair_attempt01/run_report.json`,
`reports/m3_repair_attempt01/guard.json`, and
`reports/validation_m3_retention_repair_20261024.json`.
