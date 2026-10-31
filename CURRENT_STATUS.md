# Current Spark intrinsic-memory status

**31 October 2026 UTC — v15 reader/query diagnosis is complete; the single v16 content-key native comparison is running.** Development labels remain closed. Useful recall and generalization are not yet established by the running arm.

## Active experiment and exact identities

- Selected mechanism: **content-only attention keys with ordered values retained**, at sites 3/7 with post-block placement. Source features, four slots, value order, objective, optimizer, initialization seed 17, fitting rows and terminal gates are unchanged.
- Protocol: `configs/spark_content_key_v16.json`, SHA-256 `f1cea011ca335515a9b966e4117147711d8b3bbd9f1179a0a1160224f3bc26a4`.
- Packet: `reports/intrinsic_memory_content_key_v16/native_packet_two.json`, SHA-256 `043b745591a64c2515922231b94388a6156df77b39260998b2b3bf3a3a6bdc7b`.
- Implementation closure: 62 files, SHA-256 `3a311f683cefb8f5e610445566f7fa856a079a16ea1c87578484f01e1a1bbf4e`.
- Authorization: `reports/intrinsic_memory_content_key_v16/authorization_two.json`; one attempt, zero retries.
- Output: `reports/intrinsic_memory_content_key_v16/native_bf16_two_attempt01`.
- Budget: 23 stages; 7,155 reserved stage seconds inside 7,200 outer seconds; 9,778 forwards; 949 source-feature forwards; 513 updates/backwards; 7,256 generated tokens; 4,669 prompt tokenizations; 3,618 writes; 55,490,098,857 reserved read bytes; 205,537,280 artifact bytes.
- Resources: one exclusive worker, 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free, CPU fallback disabled.

The mechanism is falsified if any final eight-world fitting group fails or known-known counterfactual pairs both correct do not exceed the sealed 1/8 reference. No best-prefix selection is allowed. Only a fully passing final fitting gate may open development.

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

- V16 terminal fitting, restart and resource outcomes are pending reconciliation; development remains unopened.
- Formation, persistent learning, native density, personal fidelity, nested adaptation and RSI remain gated. No P updates, capacity expansion or promotion occurred.
- Less-compressed representations and compression training remain separate fallbacks only for later evidence of compression loss. Exact-evidence retrieval is a charged reference, not learned recall. No fallback is combined with content keys.
- Full committed regression passes 1,000 tests and broad content-key/native/contextual contracts pass 158 tests. Portable migration verification currently passes 1,500/1,502 and final-product verification 547/552; remaining failures are documentation/hash lineage items pending final reconciliation, not passing receipts.

Historical v12 attempt10 and v14 pre-MLP remain preserved negative results, not current navigation. Protected and unrelated dirty work remains untouched; `src/replica_lab/runtime_pilot.py` must never be read, imported, edited, or staged.
