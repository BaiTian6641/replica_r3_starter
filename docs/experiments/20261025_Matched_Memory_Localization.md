# Matched-episode memory localization: exact-key retention fails first

**25 October 2026.** The matched-episode diagnostic identifies the first failing
stage as **exact observed-key stored-value retention/readout**, with strong
last-write dominance. Query-derived addressing passes, candidate/value
discrimination fails downstream, and native generation confirms a genuine
memory-arm failure. This does not justify changing the writer, reader,
addressing objective or answer integration without a new update-dynamics
comparison.

Final evidence: [independent review](../../reports/memory_matched_20261025/independent_review.md),
[native03 run](../../runs/memory_matched_20261025/native03/run_report.json), and
[guard](../../reports/memory_matched_20261025/native03.guard.json).

## Matched stages

| Stage | Top1 | Mean rank | Mean margin | Verdict |
|---|---:|---:|---:|---|
| 1 — exact observed key → stored value | 18/120 = 0.15 | 3.275 | −0.0219221 | **Fail** |
| 2 — query-derived addressing | 116/120 = 0.9667 | 1.0333 | +0.0857098 | Pass |
| 3 — query key → value | 18/120 = 0.15 | 3.2667 | −0.0230961 | **Fail** |

The decisive pattern is write position: **18/18 final-write targets top1 versus
0/102 non-final-write targets**. The final write is the top candidate in 115/120
fact probes. Single-fact rows are 6/6, pair rows 0/12 and triple rows 0/18.
Corrected targets are 12/12, unaffected retained facts 0/30, and superseded
selection is zero. Query-derived keys can discriminate addresses while earlier
values are not recovered.

## Clean source controls and native generation

Native02 used six tokens and truncated 96/120 fact targets and all 12 correction
targets. The sealed-tokenizer audit established 11 answer tokens plus EOS = 12
as the required bound. Native03 uses that bound, with every answer and all 4,181
generated tokens preserved.

| Arm | Accuracy | Interpretation |
|---|---:|---|
| Exact source | 162/162 = 1.0 | Complete opaque values from eligible source records |
| Full context | 160/162 = 0.987654 | Two genuine non-target values, not truncation |
| Memory | 0/162 = 0.0 | No correct answers or correct-value prefixes |
| Context removed | 42/162 = 0.259259 | Memory minus removed = −0.259259 |

Both source arms have 162/162 EOS-terminated rows, zero budget-exhausted rows,
zero early non-EOS stops and zero remaining expected-answer prefixes. The
frozen generator can emit values from source context but not from learned
memory. Longer generation does not repair memory behavior: 105 memory rows
exhaust the 12-token budget, 57 terminate with EOS and four remain empty.

## Mechanical and run evidence

All 1,398 expected rows are present. Stage 1–3 readouts and codebooks are
byte-identical between native02 and native03. Save/reload is 42/42 equal with
zero difference, parameters remain frozen with zero updates, and deadline
539.89 s is below 900 s. Guard elapsed is 543.251 s; peak sampled driver is
10,765,336,576 bytes, with zero swap growth and 74% minimum host free.
Mechanical parity demonstrates serialization integrity only.

Native01 failed on the previous 256-token prompt bound; preparation now audits
all 648 prompts and admits the exact 269-token maximum with complete pages and
no truncation. Native02 exposed the separate six-token generation confound,
which native03 repairs once without changing episodes, model weights or scoring
rules. Native01 and native02 remain preserved as diagnostic attempts.

## Next discriminator

Compare update dynamics on the same matched episodes: per-depth versus shared
reduction, recency/erase/decay controls and write-position ablations, while
holding observations, candidate sets and scoring criteria fixed. Query-derived
addressing does not need another key-space objective to explain the earlier-value
loss. Do not start another blind writer-training arm or enlarge memory capacity.
Useful learned memory remains unproven.
