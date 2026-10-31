# Spark fitting-only reader/query and interference diagnostic v15

Status: **attempt02 completed all 16 fitting pairs; content-only attention keys with ordered values retained is selected for v16.** No development labels were read and no parameters were updated. Useful recall and generalization remain unestablished.

## Fixed scope

- Source: sealed official BF16 Spark and the established tokenizer/original parameter identities.
- Parent: validated v12 attempt10 extension checkpoint, manifest SHA-256 `379e0402448771de2002cbca4e0c10d552ca216600eed5e6386dd30734308d15`; parent execution identity is migrated explicitly and the diagnostic identity restored.
- Rows: the same 32 fitting rows as 16 same-question counterfactual pairs. No calibration or development rows are bound.
- Controls per row: `contextual`, `wrong`, `shuffled`, `zero`; exactly one uncached prefill per control at sites 3/7.
- Preserved outputs: full first-token distributions, actual query/key/value tensors, normalized eight-slot weights, read vectors, slots, expected/other-token margins, per-pair receipts and artifact hashes.
- Forbidden: generation, parameter updates, development labels, best-prefix selection, new source features, more slots, objective changes and capability/generalization claims.

The already preserved v13 slot/reader/projection/residual tensors remain separate evidence. Their pair-held-out probes show recoverable information but do not measure actual attention weights or prove native behavioral use.

## Mechanism decision after measurement

The result will select exactly one primary mechanism and a quantitative falsification criterion before any training comparison. The decision must distinguish:

1. query/key addressing failures from value/read information loss;
2. information preserved in reads but lost in downstream behavior;
3. known-known interference from known-unknown abstention behavior.

A less-compressed source representation or compression-training arm is a **separate fallback only if compression loses the relevant information**. Exact-evidence retrieval remains a charged reference, not learned recall. Neither fallback is combined with reader/query, placement or objective changes.

## Exact bindings and envelope

- Protocol: `configs/spark_reader_query_diagnostic_v15.json`, SHA-256 `fbb998a2a2574945082926faf44192a97b612132c8d81a07d104d0b8d2e36cbb`.
- Packet: `reports/intrinsic_memory_reader_query_v15/native_packet.json`, SHA-256 `548a3833abd0828646430cf91d9922cc727f3aced397e174e28e3306883d914e`.
- Implementation closure: 66 files; receipt SHA-256 `acdbe98ebcbd8f4499cbc751d4728aca2e80ad56d35619db4d1797463e394b00`.
- Budget: 128 forwards, 32 source encodes/source-feature forwards, 32 prompt tokenizations, zero generated tokens/backwards/updates, 32,768 input tokens, 50 writes and 85,983,232 artifact bytes.
- Time: 930 reserved stage seconds inside a 1,200-second outer ceiling.
- Resources: one exclusive native worker, 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free and CPU fallback disabled.
- Attempts: one, zero retries; authorization is a separate exact receipt.

CPU mechanics pass 22 tests, including production-logit parity, weighted-read identities, wrong-bank query behavior, shuffle/zero interference, complete pair artifacts and fitting-only protocol refusal. Native packet review and environment-only validation pass without importing the model. These checks qualify engineering only.

## Attempt01 failure and separate attempt02

Attempt01 terminated with `BudgetError: stage work budget exceeded before dispatch`
after 59.274 seconds. Twelve complete pair JSON/tensor artifacts are preserved.
The shared `forwards` counter also charges source-feature execution, so 128 was
under-reserved for 128 prompt prefills plus 32 source-feature forwards. No
capability evaluation or development exposure occurred; attempt01 is not rerun.

Attempt02 preserves the original protocol derivation as `budget_revision=1` and
uses explicit `budget_revision=2`: **160 forwards** and **40,960 input tokens**.
All data, controls, checkpoint, implementation behavior, wall/resource limits and
zero-update/development boundaries are unchanged. New bindings:

- Protocol: `configs/spark_reader_query_diagnostic_v15_attempt02.json`, SHA-256 `67345b20c9e9261607cff0278fbd8ebdaa42d983f70f95c7664dac4a4367df29`.
- Packet: `reports/intrinsic_memory_reader_query_v15/native_packet_attempt02.json`, SHA-256 `1a4fdbd4fc868aea6d3f4ac570bced6dab39772b7e23fbf2edb842c5bc4a7101`.
- Implementation closure: SHA-256 `fb66b12059842c1787003692dd3c25031ef11dc6f5c11dc81e77ea685e845d79`.
- Output: `reports/intrinsic_memory_reader_query_v15/native_bf16_reader_query_attempt02`; one new attempt, zero retries, separate exact authorization required.

## Terminal outcome and selection

Attempt02 completed with first-token correct counts 24 contextual, 24 shuffled,
6 wrong and 1 zero. All 128 row/control first-token distributions exactly match
the sealed v13 reference. Wrong-bank weights exactly match partner-contextual
weights at both sites. Original weights were unchanged; development labels and
parameter updates remained zero. Supervision used 160 forwards and 37,258 input
tokens, with zero generated tokens or updates. Peak sampled driver was
9,352,904,704 bytes, swap growth zero, minimum host free 74%, and observed output
38,820,215 bytes.

Attention is peaked and positionally stable. Site-3/site-7 mean maximum weights
are 0.9739/0.9972. Site 3 selects the same top slot in 16/16 pairs; site 7 selects
slot 0 in 32/32 rows. These facts alone do **not** prove failed addressing:
different correctly selected values may occupy the same slot. Pair-held-out answer
decodability is 0/24 for site-3 weights, 5/24 for site-7 weights and 21/24 for
each read and averaged slots, but weak weight-only color prediction is likewise
not an addressing verdict. Known-known first-token correctness is 16/24; known-
known pairs both correct remain 1/8.

The selected single mechanism was **content-only attention keys with ordered
values retained**, targeting a learned-order/key-alignment hypothesis rather than
proven positional failure. V16 subsequently falsified that mechanism behaviorally;
the stable positions did not independently establish the cause. The falsification
rule and terminal outcome are in `docs/experiments/20261030_Spark_Content_Key_V16.md`.
The reproducible analysis is `scripts/analyze_spark_reader_query_v15.py`; corrected
raw metrics, original pair identities and matched shuffled targets are in
`attempt02_analysis_run02.json`. Earlier control scores remain preserved with
their original-target scoring defect. Less-compressed/compression-training and
exact-evidence paths remain separate.
