# E4 contextual writer: representation still fails novel-value use

23 October 2026 UTC. Protocol/data committed before execution. **Result:
rejected.** Contextual source features reduced training loss from **50.1 to
8.3**, but train exact is **3/24**, held-out **0/8**, and every control remains
0/8. This is a representation/optimization failure, not a wiring failure.

## 1. Arm

- New 24/6/8 source-page family with fresh names/values.
- Writer input: frozen native source-layer hidden features (layer 13), not raw
  input-token embeddings.
- Fast write encoder + native read ports trained only; base and E2 parent
  frozen; one page observation -> one deferred commit.
- 32 updates × 4, Muon 8-bit + AdamW, lr 2e-2; 6-token greedy evaluation;
  768-token query prompt; 128-token source cap.
- Persisted image reload and zero/disabled controls preserved.

## 2. Results

| Arm | Dev | Held-out |
|---|---:|---:|
| Correct observation | 0/6 | **0/8** |
| Shuffled observation | 0/6 | 0/8 |
| Zero state | 0/6 | 0/8 |
| Fast disabled | 0/6 | 0/8 |

Training loss fell **50.09 → 8.31**, but only **3/24** train answers were exact.
Correct and shuffled states sometimes changed the generated value, so the state
is read; it does not produce a reliable novel-value output. Zero equals disabled
and image reload is exact.

## 3. Classification

The earlier token-embedding writer failed; contextual source features improve
the training loss but not held-out value copying. The remaining bottleneck is
the **writer/decoder-copy representation**, not chunk ownership, controls,
provenance, or persistence. Do not repeat this arm unchanged.

Decision: **reject**; no default/fallback changed. Exact text remains the
preferred path. A future arm needs an explicit copy/pointer decoder or a
different associative objective on a new held-out family.

## 4. Costs and evidence

**74.8 s**, **11.0 GiB** sampled peak driver, one extension-only export, no base
copy, no download. Training answers and all controls are in
[the report](../../reports/e4_contextual_attempt01/run_report.json).
