# E2 retrieval, novel-record use and native capacity

23 October 2026 UTC / 22 October America/Chicago. Branch
`codex/d2-capacity-runtime-pilot`. Protocols/data committed before execution:
`dabaa00` (retrieval and capacity), `3fd5c03` (answer-path diagnostic).
Accepted native runs: retrieval attempt 3, capacity attempt 3, diagnostic.

**Result: eligibility and capacity engineering pass; retrieval ranking and
answer use fail.** Lexical/native-dense retrieval reaches held-out record
recall@1 **0.75** and recall@4 **1.0**; learned reranking lowers top-1 to
**0.50**. All slot-reading arms answer **0/6** held-out cases and the oracle
diagnostic answers **0/4**; exact source text answers **3/3**. The bottleneck is
the record writer/reader encoding, not eligibility, retrieval availability,
prompt format or decoder capacity.

## 1. Dataset and separation

`data/e2_retrieval` contains **13 cases / 49 records**, source-separated by
group: 2 train groups (5 cases), 2 dev groups (2 cases), 6 held-out groups
(6 cases). The selection covers similar distractors, same-entity/different
attribute records, correction/supersession, future/unauthorized/wrong-purpose
records, missing information, unseen entity/value combinations, and a
2-chunk long record (231 tokens; fact starts at token 177). Eligibility was
filtered by the host ledger before any ranking arm; the unauthorized branch,
future observation and wrong-purpose record never entered a neural bank.
Held-out groups were never used for reranker training.

Training used only the explicit ranking objective (`ranking_loss`) over
eligible candidates: 12 updates × 5 train queries, reranker + block-13 sensor,
Muon 8-bit/FP32 AdamW at 1e-3, seed 20261023. Answer loss was not used for
hard top-k. All nine trainable tensors received finite nonzero gradients.
Per-update total loss moved only **6.625 → 6.423** (about 1.325 → 1.285 per
query): the small budget did not learn a useful ranking.

## 2. Retrieval and answer results

Selected IDs and raw answers are in the
[attempt 3 report](../../reports/e2_retrieval_attempt03_20261023/run_report.json).
The two arithmetic arms are identical on this set; dense native features add no
measured ordering benefit.

| Arm | Dev recall@1 / @4 | Held-out recall@1 / @4 | Held-out answer | Oracle answer |
|---|---:|---:|---:|---:|
| Lexical | 1.00 / 1.00 | **0.75 / 1.00** | 0/6 | 0/4 |
| Native dense (pure) | 1.00 / 1.00 | **0.75 / 1.00** | 0/6 | 0/4 |
| Learned rerank | 0.00 / 1.00 | **0.50 / 1.00** | 0/6 | 0/4 (diagnostic) |

Diagnostics:

- **Eligibility:** zero forbidden records selected in every arm; correction
  old records were absent and current records were selected. Future,
  unauthorized and wrong-purpose records were never candidates.
- **Correction:** current record retrieved at rank 1 in lexical/dense; the
  model still answered an old label (`white` instead of `indigo`).
- **Missing information:** slot arms answered `amber`/`red` instead of
  `UNKNOWN`; the no-slot diagnostic answered `UNKNOWN`.
- **Long record:** lexical selected it at rank 3, rerank at rank 1; all four
  bank spans included the fact offset, but the answer was still `amber`.
- **Distractors:** mean 2.3 eligible non-expected records were present in the
  held-out neural bank, so the arms were not relying on lexical-zero exclusion.

## 3. Answer-path diagnostic

Post-hoc, no training, same trained extension; three held-out cases. Correct
record slots, a different eligible record's slots, no slots, and exact source
text were compared.

| Case | Correct slots | Swapped slots | No slots | Exact text |
|---|---|---|---|---|
| `heldout_a` | `amber` | `amber` | `UNKNOWN` | **`crimson`** |
| `heldout_b` | `white` | `white` | `UNKNOWN` | **`indigo`** |
| `heldout_d` | `amber` | `amber` | `UNKNOWN` | **`quartz`** |

Correct and swapped slots produce the same old answer, so the readers are not
content-sensitive for these novel records. No slots produce a correct
abstention. Exact source text produces the correct values in all three cases.
This localizes the failure to the **record writer/reader encoding**, not
eligibility, candidate retrieval, long-record chunk inclusion, prompt format or
the language head.

## 4. Native capacity

`data/e2_capacity` is an exact 16-unit bank: twelve 128-token source spans and
one 507-token record split into four 128/128/128/123-token chunks. All twelve
schedule points completed without a guard stop. Actual workloads, not configured
maxima, are reported:

| Workload | Actual units | Source span max | Prompt tokens | Result |
|---|---:|---:|---:|---|
| Inference | 1, 4, 8, 12, 16 | 128 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 4, 8, 12, 16 | 128 | 64, 128, 256 | all finite; 28 gradient tensors |

Largest actually exercised path: **16 units / 2,043 source tokens (12×128 plus
128+128+128+123) / 256 prompt tokens / 8 generated tokens**, and a separate
**16-unit / 256-prompt modules-only backward**. Inference at the largest point:
source encode 0.421 s, writer 0.014 s, sense 0.117 s, action forward 0.128 s,
8-token generation 0.391 s. Training at the largest point: prepare 0.130 s,
forward 0.144 s, backward 0.350 s; no optimizer step. Timing is sampled and
single-run; it is not a throughput claim.

Peak sampled driver memory **12.51 GiB** (retrieval 10.05 GiB, diagnostic
11.0 GiB), zero observed swap growth, 69% minimum free memory. No base copy,
optimizer checkpoint or download. D2 original-weight capacity is **not declared
or run** in this increment.

## 5. Defaults and handoff

- **Default retrieval:** lexical remains the default. Native dense is available
  but measured equal to lexical on this set; learn more before preferring it.
- **Reranking:** remains **opt-in**. Held-out top-1 was worse than lexical and
  the ranking loss barely moved. Do not treat the new extension as a quality
  upgrade.
- **Reader:** slot-only answers are not reliable for novel values. Prefer the
  exact-text fallback when precision matters, or improve/train the writer and
  readers before using slot-only slots for E3 claims.
- **Capacity:** 16 units with 128-token spans, 256-token prompts and 8 generated
  tokens are qualified on this Mac under the existing guard. This is not a
  claim for larger prompts, more units, D2, or a different model.
- **E3 owner:** the E3 parent extension and E2 interfaces are unchanged; don't
  interpret E2 slot failures as E3 core failures. Use exact text or a new
  writer/reader arm for novel-value tasks, and keep reranking opt-in.

## Evidence and preserved failures

- [Retrieval accepted report](../../reports/e2_retrieval_attempt03_20261023/run_report.json)
  and [guard](../../reports/e2_retrieval_attempt03_20261023/guard.json).
  This is the pure-dense arm (`lexical_weight=0`), with the update-1 gradient
  audit captured before `step`/`zero_grad`.
- [Retrieval attempt 2](../../reports/e2_retrieval_attempt02_20261023/run_report.json)
  used the historical hybrid lexical+dense call; it produced the same retrieval
  and answer metrics and is preserved as a protocol-conformance defect.
- [Retrieval attempt 1](../../reports/e2_retrieval_20261023/run_report.json)
  (same results; update-1 gradient audit was empty because it was sampled after
  `zero_grad`)
- [Diagnostic report](../../reports/e2_diagnostic_20261023/run_report.json)
- [Capacity accepted report](../../reports/e2_capacity_attempt03_20261023/run_report.json)
  and [guard](../../reports/e2_capacity_attempt03_20261023/guard.json)
- Capacity attempt 1 (evidence-pin mismatch) and attempt 2 (variable final span
  not padded) are preserved with their partial profiles.
- Protocols: [retrieval](../../configs/e2_retrieval_qualification_v1.json),
  [capacity](../../configs/e2_capacity_qualification_v1.json),
  [diagnostic](../../configs/e2_retrieval_diagnostic_v1.json).

The fixtures are exposed synthetic development data. The held-out groups are
source-separated but not a sealed external benchmark; no broad generalization
or calibrated-confidence claim follows.

## 6. Follow-up: 32-unit architecture capacity (23 October)

A fresh 32-unit synthetic bank (28 short records plus one four-chunk record) was
declared before execution. The selected E2 extension declares
`PortSpec.max_units=16`; an attempted selected-weight 32-unit run stopped at
that hard limit and is preserved as failed evidence.

A separate **architecture-only** arm used a 32-unit `PortSpec`, no selected
extension and random E2 ports. It is not a selected-weight or quality result.

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 8, 16, 24, 32 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 8, 16, 24, 32 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **12.57 GiB**, profiled wall
**17.4 s**, zero observed swap growth, 68% minimum free. No optimizer update, no
base copy, no checkpoint.
[Architecture-only report](../../reports/e2_capacity32_architecture_attempt03/run_report.json).

### Selected extension at 32 units

A validated **port-capacity-only** migration then loaded the selected E2
extension into a 32-unit `PortSpec`. The migration requires identical tensor
inventory/shapes and all shared manifest fields except `ports.max_units`, which
may only increase. The selected extension passed the same 12 points: peak
**12.57 GiB**, profiled wall **14.9 s**, zero swap, 28 gradient tensors, no base
save or optimizer update. The qualified selected-weight maximum is therefore
**32 units** for this synthetic workload.
[Selected report](../../reports/e2_capacity32_selected_attempt01/run_report.json).

## 7. Follow-up: selected 64-unit capacity (23 October)

The same selected E2 extension and validated `max_units`-only migration were
exercised on a fresh 64-unit bank (60 short records plus one four-chunk record).

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 16, 32, 48, 64 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 16, 32, 48, 64 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **12.69 GiB**, profiled wall
**19.1 s**, zero observed swap growth, 69% minimum free. No optimizer update, no
base save, no checkpoint. This is capacity only, not answer quality or a
longer-source-length claim.
[Report](../../reports/e2_capacity64_selected_attempt01/run_report.json).

## 8. Follow-up: selected 128-unit capacity (23 October)

The selected E2 extension and same validated `max_units`-only migration were
exercised on a fresh 128-unit bank (124 short records plus one four-chunk record).

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 32, 64, 96, 128 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 32, 64, 96, 128 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **12.93 GiB**, profiled wall
**27.2 s**, zero observed swap growth, no base save and no optimizer update.
This is capacity only; answer quality and longer-source-length behavior remain
unqualified.
[Report](../../reports/e2_capacity128_selected_attempt01/run_report.json).

## 9. Follow-up: selected 256-unit capacity (23 October)

The same selected extension and migration passed a fresh 256-unit bank
(252 short records plus one four-chunk record).

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 64, 128, 192, 256 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 64, 128, 192, 256 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **13.50 GiB**, profiled wall
**43.4 s**, zero observed swap growth, no base save or optimizer update.
Capacity only; no quality or longer-source-length claim.
[Report](../../reports/e2_capacity256_selected_attempt01/run_report.json).

## 10. Follow-up: selected 512-unit capacity (23 October)

The same selected extension and `max_units`-only migration passed a fresh
512-unit bank (508 short records plus one four-chunk record).

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 128, 256, 384, 512 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 128, 256, 384, 512 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **14.62 GiB**, profiled wall
**76.0 s**, zero observed swap growth, no base save or optimizer update. This
closes the tested selected-weight capacity ladder at 512 units for a synthetic
128-token-span bank; it is not an answer-quality or longer-source-length claim.
[Report](../../reports/e2_capacity512_selected_attempt01/run_report.json).

## 11. Follow-up: selected 1024-unit capacity (23 October)

The selected extension and migration passed a fresh 1024-unit bank
(1020 short records plus one four-chunk record).

| Workload | Actual units | Prompt tokens | Result |
|---|---:|---:|---|
| Inference | 1, 256, 512, 768, 1024 | 64, 128, 256 | all finite |
| Modules-only backward | 1, 256, 512, 768, 1024 | 64, 128, 256 | finite; 28 gradient tensors |

All 12 points completed. Peak sampled driver memory **16.90 GiB**, profiled wall
**142.6 s**, zero observed swap growth, no base save or optimizer update.
This is a synthetic 128-token-span capacity result; it is not an answer-quality
or longer-source-length claim. The tested selected-weight ladder is 32/64/128/
256/512/1024 units.
[Report](../../reports/e2_capacity1024_selected_attempt01/run_report.json).
