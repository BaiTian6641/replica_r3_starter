# E2 exact-span pointer: positive opt-in novel-value copy

23 October 2026 UTC. Protocol committed before execution. **Result: pointer
copy-enabled 3/6 dev and 2/10 held-out exact; shuffled-source control 0/6 and
0/10. Accepted opt-in only; lexical/exact-text defaults unchanged.**

## 1. Mechanism

The pointer enumerates contiguous exact source-token spans from the selected
records, scores each span from the prompt's final native hidden state, and emits
the selected source token IDs directly. It does **not** generate via the LM
head, does not use teacher answers as candidates, and cannot choose a record:
the host enumerates all selected-record spans; the pointer selects one.

- `CopySpec(mode='pointer', max_span=4)`.
- `RecordPointerHead` uses frozen token embeddings, two normalized projections
  and masked span scoring.
- Trained only on the copy head; base and E2 writer/readers frozen.
- 24 train / 6 dev / 10 held-out fresh novel-value cases.
- 16 updates × 4; Muon 8-bit + AdamW; lr 0.01; extension v2, 53.1 MB.
- Costs: **37.4 s**, **11.0 GiB** sampled peak driver; no base copy.

The first attempt failed on a token-boundary mismatch (standalone answer token
vs source leading-space token); it is preserved. The corrected target matching
uses decoded span text.

## 2. Results

| Arm | Dev | Held-out |
|---|---:|---:|
| Pointer enabled | **3/6** | **2/10** |
| Shuffled source spans | 0/6 | 0/10 |

The pointer emitted exact source values; shuffled candidates produced wrong
values, so the model is using the selected source rather than a fixed answer
prior. Held-out hits were `navy` and `magenta`; misses include `Flint`/`flint`
(case mismatch), `amber`, `blue`, `silver`, `crimson`. Training loss stayed near
18.5, so the margin is modest and the optimizer did not show a clean learning
curve. A tiny CPU probe confirms the pointer mechanism itself learns: loss fell
**2.689 → 2.291** over 300 steps and the target span became the argmax. The flat
native curve is therefore an optimization/data-scale limit, not a broken pointer
path.

## 3. Decision and limits

- **Accepted opt-in only** by the preregistered rule (held-out above shuffled,
  positive, dev not below shuffled). No default was changed; exact text remains
  the reliable fallback.
- The pointer outputs only spans that exist in the selected source; it cannot
  answer from absent information and does not claim success.
- This is a synthetic copy candidate, not general reader capability. The
  pointer is not yet connected to the full action-selection runtime; it is an
  E2 answer path.
- The flat loss and low absolute accuracy warrant a future distinct training or
  query-scoring arm, not a repeated run of this one.

## 4. Evidence

[Run report](../../reports/e2_pointer_head_attempt02/run_report.json),
[manifest](../../reports/e2_pointer_head_attempt02/extension_manifest.json),
[attempt 1 defect](../../reports/e2_pointer_head_attempt01/failure_report.json).
