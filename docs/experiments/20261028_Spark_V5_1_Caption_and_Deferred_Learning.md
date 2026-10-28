# Spark v5.1, caption lineage and deferred-learning status

**28 October 2026. Status: documentation, arithmetic, lineage and CPU validation
complete; every native learning gate remains unresolved.** Implementation and
migration commit: `d65a36ff`. Passing-receipt commit: `48e8d49e`.
No model training, native Spark execution, dataset download or personality
adaptation occurred.

## 1. Spark migration and residency evidence

The v5.1 migration retains the official BF16 release, official FP8 storage, the
dequantized BF16-storage derivative and native FP8 execution as distinct
identities. A dequantized reload is not native FP8 execution. Retained vision,
base/extended weights, optimizer state, activations and restart transients are
itemized separately from P16/P32/P64 and Mkv-4k/64k analytical profiles.
Analytical residency is not measured fit, a revised process guard or full-training
qualification.

The portable verifier passed **721/721** with `--tracked require`; the final
product verifier passed **500/500** with zero unavailable source-cache entries
and no unexplained input drift. The earlier final-product attempt failed 499/500
only because eight declared inputs were unstaged; that receipt is preserved as
`verification_unstaged_attempt01.json`.

The focused Spark/caption suite passed **112 tests with one explicit artifact
skip**. This validates documentation, migration identities, arithmetic, bundle
references and caption behavior, not native learning, sparse kernels, throughput
or model fit.

## 2. Caption admission and partition identity

The corrected caption lineage admits exactly nine reviewed rows, with 21
exclusions and eight quarantined exposed rows. The separate synthetic partition
contains 24 diagnostic scenes only; it is not natural-image or broad visual
validation. Parent QA remains `reject`, and the exposure incident is not cleared
or relabelled. Rights remain internal-research-only with redistribution
prohibited.

The deterministic partition sidecar records parent/child identities, provenance,
rights, dedup keys and admission roles. It is not an admission receipt and does
not make any row eligible by itself. The trusted local reviewed receipt binds
exactly the nine rows with SHA-256
`8ca183142d8af706d61ca3c12ba9cf089c4e1e7622228fff33190c612a3cb955`.
Validation confirms no admitted/evaluation collisions, 30 unique parent rows and
38 partition records including exclusions/exposed/diagnostic roles.

A test-fixture regression was fixed by regenerating the packet and sidecar after
the fixture writes its receipt and before asserting the valid baseline. The
exposed-row and wrong-reviewer/timestamp refusals remain intact; no admission
check was weakened.

Evidence:

- `reports/caption_admission_preparation_20261027/partition_identity.json`
- `reports/caption_admission_preparation_20261027/admission_packet.json`
- `reports/sparkx_v5_1_review_20261027/verification.json`
- `reports/sparkx_final_product_20261026/verification_v2.json`

## 3. Deferred-learning gates

| Gate | Current evidence and status |
|---|---|
| NG1 useful response learning | DC-P1 protocol/runner is prepared and CPU-qualified; no native result. Unresolved. |
| NG2 vision first-update progress | Vision v2 accounting and co-adaptation are prepared; no native arm. Unresolved. |
| NG3 native retrieval/readout | DC-R1 fixture/scoring is prepared; no native runner or compressed-channel receipt. Unresolved. |
| NG4 P capacity fit/learning | P16 continuation is not a P32/P64 capacity result. Unresolved. |
| NG5 sparse Mkv execution | No admitted sparse-optimizer implementation or measurement. Unresolved. |
| NG6 teacher competence/self-distillation | No competent-teacher or distillation run. Unresolved. |
| NG7 outer objectives/multiple clocks | No outer-loop or multi-clock run; derivative paths and equal-cost controls are undeclared by any result. Unresolved. |
| NG8 full Spark training | No actual v5.1 graph residency/throughput/precision or learning qualification. Unresolved. |
| NG9 official/native FP8 | Official storage, dequantized derivative and native execution are not interchangeable; native FP8 remains unrun. Unresolved. |
| NG10 personality | No separate user specification and evaluation. Unresolved. |

Useful inner learning must precede teacher competence. Teacher competence and
all teacher/rollout/replay/evaluation work must precede outer training, multiple
clocks, self-distillation or any RSI/policy-improvement loop. Capacity expansion
requires learning/retention benefit at fixed active work or lower cost at matched
quality. Restart evidence must distinguish read-only parity from subsequent-
learning parity and preserve rollback state. Personality remains a separately
supplied and separately evaluated input.

No RSI cycle, self-distillation, outer-loop training, personality adaptation,
full Spark training or promotion follows from these documents.

## 4. Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_sparkx_v5_1_review.py \
  tests/test_sparkx_final_product.py \
  tests/test_sparkx_first_tranche.py \
  tests/test_caption_lineage_and_eval_20261027.py \
  tests/test_external_caption_asset_qa.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked require

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reports/sparkx_final_product_20261026/verify.py \
  --root . --out reports/sparkx_final_product_20261026

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/validate_caption_lineage_and_eval_20261027.py \
  --require-admission --as-of 2026-10-28
```

The caption validator reports `status=valid` and `admission.status=admitted` for
exactly the nine reviewed rows. This is data admission, not native model
execution or learned visual quality.
