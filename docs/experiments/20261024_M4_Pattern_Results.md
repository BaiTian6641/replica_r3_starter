# M4 — static pattern and deterministic routing rejected

24 October 2026 UTC. **Decision: quality reject on general noninterference.**
The independent [verdict](../../reports/m4_pattern_attempt04/verdict.json) passed
all engineering checks except `general_noninterference`. It is not a resource,
lineage, restart or optimization defer.

## 1. Delayed and general outcomes

| Metric | Result | Interpretation |
|---|---|---|
| Router dev / held-out / regression | 4/4 / 6/8 / 8/8 | Above null controls; no static-pattern attribution |
| Direct pattern dev / held-out / regression | 0/4 / 0/8 / 1/8 | Poor static utility |
| Pattern training accuracy | 8/24 | Underfit, final mean loss 1.13375 |
| General correct / router / fast | 0/4 / 0/4 / 0/4 | No retention repair |
| Adversarial vs disabled | 0/4 vs 2/4 | **Mandatory rejection** |
| Development adversarial | 0/2 vs disabled 2/2 | Also below base |
| Full context | 4/4 | Items are answerable |

All 20 delayed dev/held-out/regression router evaluations selected `fast`
because of uncovered hashes. Router and fast token IDs were identical, so router
performance is not evidence of static-pattern utility. The adversarial arm used
fast memory with `pattern=None`; therefore warming up pattern tables cannot
repair that specific gate under the current wiring.

## 2. Engineering evidence

- Causal 2/3-gram hashing, independent pattern branch, zero-init output and
  sparse gradient/optimizer behavior are implemented.
- Update-two gradients are finite and nonzero across all pattern tensors.
- Sparse gradients are densified only for AdamW; Muon rejects sparse gradients.
- Disabled-base token IDs match the parent and fast modules are unchanged.
- Same-process depth-image reload passes. Fresh-process restart was not run
  because the candidate was rejected.
- Table parameters occupy 134,217,728 bytes; optimizer state is 269,527,088
  bytes (1,056,768 int8 and 268,470,320 FP32).
- The preserved verdict lacks complete collision/latency fields from the failed
  runner. Do not infer those measurements; a separate collision evaluation is
  required before claiming them.

## 3. Costs and preserved failures

- 16 updates, 426 answers, 1,386 generated tokens and 28,241 prompt tokens.
- Profiled wall 154.291 s; sampled peak driver 11.127 GiB; no downloads or base
  copies. One extension and one depth image were saved.
- Attempt 1 failed on a generator-scope error; attempt 2 omitted session
  initialization; attempt 3 hit SparseMPS `ne` during gradient audit; attempt 4
  completed training/evaluation but failed verdict binding. All are preserved.
- Independent verdict with pinned parent adversarial controls: quality reject.

## 4. Next constraint

A future static-optimization diagnostic must directly measure pattern utility
on covered non-training cases and enable pattern in adversarial evaluation. A
source-LM warmup would otherwise be additional training with a predetermined
rejection and no routed-static-utility measurement. Do not reclassify this M4
failure as accepted, and do not advance to M5 on the strength of fast-memory
scores. Retention remains open.
