# Current Spark intrinsic-memory status

**31 October 2026 UTC — the v17 fitting-only interference diagnostic is complete.** Development labels remain closed. Useful recall and generalization are not established.

## Current evidence and exact identities

- V17 completed four known–known fitting pairs (IDs 0, 1, 4, 5) with two isolated one-update arms per pair. Target sequence margin improved in 7/8 arms; partner margin worsened in 7/8. Gradient cosine was negative for two pairs and positive for two, so cosine alone did not predict the measured margin interference.
- Result and analysis: `reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt05/{result.json,analysis.json}`; interpretation and raw-receipt index: `docs/experiments/20261031_Spark_Interference_V17.md`.
- Protocol: `configs/spark_interference_diagnostic_v17_attempt05.json`, SHA-256 `970155e0684f66d5e32439a21e1ef584db3659e4dba2060a0ad5b0e49ed1012d`.
- Packet: `reports/intrinsic_memory_interference_v17/native_packet_attempt05.json`, SHA-256 `db2c5342100e97d94f6ce0360fd16af4e3269c23ca657a8f6dab044308f33d57`.
- Parent: V16 content-key attempt01 extension plus optimizer, manifest SHA-256 `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`; content-only attention keys and ordered values at sites 3/7.
- Consumed: 120 forwards, 16 backwards, 8 isolated updates, zero generated tokens, zero development access. Peak sampled driver 11,290,918,912 bytes; swap growth 0; observed artifacts 3,392,930 bytes; wall 80.54 seconds.

V16 content-key native training also completed. At update 512 its final fitting probes were contextual 15/32, exact-text 32/32, shuffled 15/32, wrong 6/32, zero 9/32 and absent 8/32. Counterfactual pairs both correct were 2/16 overall and **0/8 known–known**. Every final group failed and known–known did not exceed the sealed 1/8 reference, so development was not opened. V17 is a diagnostic on its fitting rows, not a new trained candidate or mechanism selection.

No best-prefix selection or retry of a consumed attempt occurred. Any broader interference study requires a separately predeclared arm; this four-pair fixture does not establish a general interference law.

The sole selected successor is [V18 paired-joint versus sequential updates](docs/experiments/20261031_Spark_Paired_Update_V18.md): equal per-world exposure on the same four fitting pairs, a fresh internal sequential reference, and frozen pair-minimum-margin/guard falsification criteria. It is **not implemented, authorized or launched**. No second comparison is selected.

## Completed v15 diagnostic

`docs/experiments/20261030_Spark_Reader_Query_V15.md` and `reports/intrinsic_memory_reader_query_v15/attempt02_analysis.json` preserve all 16 fitting pairs, raw tensors, controls and 100 shuffled-label controls. V15 attempt01 failed on an under-reserved shared counter and was not rerun; corrected attempt02 completed with exact v13 first-token logit parity.

- Site 3 repeats its top slot in 16/16 pairs; site 7 selects slot 0 in 32/32 rows.
- Mean maximum attention weights are 0.9739 and 0.9972: attention is peaked, not diffuse.
- Weight-only pair-held-out decodability is 0/24 and 5/24, versus 21/24 for each read and averaged slots.
- Known-known first-token correctness is 16/24; known-known pairs both correct remain 1/8.
- V15 used zero generation/updates and no development labels. Original weights stayed unchanged.

The evidence supports testing learned-order dominance in keys. It does not prove that content keys will improve native recall.

## Archive and migration preparation

`docs/experiments/20261030_Spark_Archive_Migration_V16.md` and `reports/intrinsic_memory_migration_v16/preparation.json` record:

- Earliest retained releases: 232,024 bytes, 64 files, 32/32 restorable.
- Current-pointer releases: 231,928 bytes, 64 files, 32/32 restorable.
- Complete current archive: 8,557,737 bytes, 736 files.
- Existing 128 rebuilds: 281.676 seconds, 986,112 written bytes and 813,032 read bytes.
- Atomic payload→manifest→verification→pointer migration recovered after an injected interruption with zero recopied bytes and an idempotent second completion. No production migration or deletion ran.

Initial means earliest retained, not an original pre-training baseline. Large-scale production migration remains unmeasured.

## Evidence-gated blockers

- V16 terminal fitting gates failed and development remains unopened. V17 diagnoses fitting interference only; it does not repair or reselect the memory mechanism.
- Formation, persistent learning, native density, personal fidelity, nested adaptation and RSI remain gated. No P updates, capacity expansion or promotion occurred.
- Less-compressed representations and compression training remain separate fallbacks only for later evidence of compression loss. Exact-evidence retrieval is a charged reference, not learned recall. No fallback is combined with content keys.
- Full committed regression passes 1,000 tests and broad content-key/native/contextual contracts pass 158 tests. The v17 interference/migration repair scope passes 16 focused tests. Earlier portable and final-product verification failures remain documentation/hash lineage items, not passing receipts.

Historical v12 attempt10 and v14 pre-MLP remain preserved negative results, not current navigation. Protected and unrelated dirty work remains untouched; `src/replica_lab/runtime_pilot.py` must never be read, imported, edited, or staged.
