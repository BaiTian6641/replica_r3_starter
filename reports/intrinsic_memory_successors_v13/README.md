# Incremental WorkLedger v13 evidence

Scope: host-side replay engineering on disposable copies of attempt10's preserved journal. No model execution, native job, capability claim, packet change, or historical-artifact mutation occurred.

## Source and baseline

- Source: `reports/intrinsic_memory_native_v12/native_bf16_two_v12_attempt10/work_ledger.jsonl`
- SHA-256: `2b703943dbbcf4228aa1064db6994889277a7350fa6ad706ba52c6ef19dcc7cd`
- 7,396,043 bytes; 31,790 journal records; 15,876 reservations; 19 replayed stages.
- `ledger_profile_attempt10_before_run02.json`: three independent full replays took 0.1251, 0.1355 and 0.1520 seconds. Repeating full replay after each mutation makes total overhead grow with journal size.

`ledger_profile_attempt10_before.json` is retained as an unsuccessful profiler receipt: it evaluated temporary-copy cleanup before context exit and therefore incorrectly reported `disposable_copy_removed=false`. Run02 evaluates cleanup after exit and reports true.

## Incremental successor

`src/replica_lab/intrinsic_memory_budget_v13.py` reads the existing v12 JSONL format. Own reserve/finish appends update verified cached state; external appends authenticate the previous prefix and then use complete v12 replay. Replacement, truncation and in-place modification are refused. Torn tails still require explicit recovery, interrupted reservations stay charged, and finish remains exactly once.

Focused verification:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python -m pytest -q tests/test_intrinsic_budget_v13.py tests/test_intrinsic_budget_v12.py
```

Result: **25 passed**, zero failed/skipped. Coverage includes independent full-replay parity, 100 mutations with only the initial full read, a real subprocess append, exactly-once charging, replacement/truncation/in-place refusal, torn-tail recovery and fresh-instance restart.

## Preserved-data comparison

`ledger_comparison_attempt10_run02.json` compares ten legal reserve/finish pairs on separate copies. Copy clocks alone were re-anchored because the historical deadlines had expired; relative budgets and all source bytes are preserved, and the transformation is recorded in the receipt.

- v12: 2.584003 seconds total, 19 full reads after warm-up including the final read.
- v13: 0.003005 seconds total, one full read.
- Observed ratio: **859.82×** for this bounded comparison.
- Authoritative semantic state (plans, reservations, events, statuses and stages) is equal under independent full replay.

`ledger_comparison_attempt10.json` is retained as an unsuccessful comparison receipt: it compared raw documents, including necessarily different timestamps, elapsed durations and file paths, and reported false. Run02 compares the authoritative fields and reports true.

This is a small preserved-data benchmark, not a guarantee of native end-to-end speedup. The sealed attempt10 implementation did not use v13. A future native packet must explicitly bind the new implementation and its own accounting envelope.
