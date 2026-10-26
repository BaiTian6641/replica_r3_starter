# Independent matched-memory outcome review

**Reviewer:** `MatchedOutcomeReview`, CPU raw-artifact recomputation and read-only
source/control audit.

## Validated localization

**Stage 1, exact observed-key storage upper bound, fails first.** No scoring,
target-identity, candidate-set or coverage defect was found that manufactures
this failure. Stage 2’s pass concerns query-derived key addressing only; it does
not establish recoverable stored values or useful language memory.

| Stage | Top1 | Mean rank | Mean margin | Outcome |
|---|---:|---:|---:|---|
| 1 — exact key → stored value | 18/120 = 0.15 | 3.275 | −0.0219221 | Fail |
| 2 — query-derived addressing | 116/120 = 0.9667 | 1.0333 | +0.0857098 | Pass |
| 3 — query key → value | 18/120 = 0.15 | 3.2667 | −0.0230961 | Fail |

Across 744 scored readout rows, independent recomputation found zero target,
top-entry, rank, acceptance or top1 disagreements; maximum absolute cosine
difference from `rankings.jsonl` was 1.41×10⁻⁷.

## Failure pattern

- Final-write targets: **18/18 top1**. Non-final-write targets: **0/102 top1**.
- The final write is the top readout candidate in 115/120 fact probes.
- Single-fact rows: 6/6; pair rows: 0/12; triple rows: 0/18.
- Corrected targets: 12/12; unaffected retained facts: 0/30; superseded
  selection is zero.

Stage 3 closely reproduces Stage 1: identical top candidate in 118/120 paired
queries and the same 18 successes. Query-derived keys can discriminate addresses
in key space while the memory read fails to recover earlier associated values.
The first gap is exact-key state retention/update/readout behavior, before
addressing or generation.

## Clean Stage-4 comparison after the generation audit

Native02 used six tokens and truncated 96/120 fact targets plus all 12 correction
targets. The sealed-tokenizer audit established 11 answer tokens + EOS = 12 as
the required bound. Native03 uses that bound; all 648 answers and 4,181 generated
tokens are authentic.

- **exact_source: 162/162 = 1.0**; all 96 formerly truncated errors became
  correct complete answers.
- **full_context: 160/162 = 0.987654**; two genuine EOS-terminated non-target
  values remain, not truncation.
- Both source arms have 162/162 EOS-terminated rows, zero budget-exhausted rows,
  zero early non-EOS stops and zero remaining strict expected-answer prefixes.
- **memory: 0/162 correct and zero correct-value prefixes**. Corrections 0/12,
  retained 0/30, other source facts 0/78, unknowns 0/42. Memory versus
  context-removed is 0 − 42/162 = −0.259259.

The frozen generator can emit complete opaque values from source context. It
does not recover them from the learned memory state. Longer generation does not
repair memory behavior: 105 memory rows exhaust the 12-token budget, 57 terminate
with EOS and four remain empty.

## Mechanical and provenance checks

All 1,398 expected rows are present. Stage 1–3 raw readouts and codebooks are
byte-identical between native02 and native03. Save/reload is 42/42 equal with zero
difference, parameters remain frozen with zero updates, and the 900-second
deadline passes. Peak sampled driver is 10,765,336,576 bytes, with zero swap
growth. Mechanical parity demonstrates serialization integrity only.

Final disposition: exact-key stored-value retention/readout fails first, with
last-write-dominated behavior. Query-derived addressing passes, value
discrimination fails downstream, and native generation confirms a genuine
memory-arm failure without changing that localization. No writer, reader,
addressing objective or answer-integration change is justified without a new
mechanism-specific update-dynamics comparison.
