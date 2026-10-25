# Fact-binding writer repair and memory reliability v5

24 October 2026 America/Chicago (25 October UTC). **Final decision: reject the
learned candidate and reject the reference bundle on its declared gates.**
Mechanical persistence and the corrected independent scorer pass. No useful
learned associative memory, broad retention, calibration or RSI admission is
established.

## 1. Lineage and declared experiment

The sealed source is MiniCPM5 revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. The M3 shared extension is the
parent initialization and regression reference, not an accepted general-memory
baseline. No model download, base copy, full-backbone update or D2 arm occurred.

- [Training protocol v5](../../configs/memory_reliability_v5.json), content
  commit `6b4de12`, metadata commit `45ba43d`, protocol SHA-256
  `daa8aad1df5fa985eb343ab18bf27d566dadc2bd536aa15e7ef228890c4918f7`.
- Capacity/reproducibility dependency commit `ee6316f`.
- Scorer corrections `eff2079` and `5aa7e75`.
- [Canonical evaluation-only protocol](../../configs/memory_reliability_v5_scorer_fix_eval02.json),
  content commit `2833242`, metadata commit `36c3c0a`, protocol SHA-256
  `fac98defeba24f4fd838d001a42f72087af98f935a7a18e5752e5abce19c2e7d`.
- [Canonical final verdict](../../reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json),
  SHA-256 `0a0e040d1102fab1d024b1a9c9300ab3a40eab9c0f281872f347c92479d29286`.

The single scientific change was the **fact-binding writer representation**.
For each observed fact, entity+attribute text forms the key input and value-only
text forms the value input. The existing SGDN-2 writer encodes each path,
parameter-free pooling produces one write per fact, and the unchanged state
engine reduces it. Readers, gate scalars, routing, exact records, native backbone
and all non-writer extension tensors remain fixed. No answer labels, generated
text or future observations enter writes.

The [binding diagnostic](../../reports/memory_binding_diagnostic_20261024.json)
was tiny CPU mechanism evidence only. It showed key/value path separation and
nonzero key/value sensitivity; it did not predict native recall. A zero-reader
tiny training probe had no writer gradient and was retained as a negative
boundary. With nonzero fixture readers, 14/14 writer tensors received gradients.

## 2. Fresh data and training scope

The fresh v4 fixture contains 32 episodes / 16 complete counterfactual families:
16 train, 8 dev and 8 heldout episodes, with 8/4/4 independent families. It has
192 training cases and 192 evaluation cases, four query forms, both variants,
independent distractors, absent facts and heldout-only observed values. The
retention pool has **16 distinct evaluation questions** across arithmetic,
factual, abbreviation and relational prompts, rather than four repeated
arithmetic questions.

Training was one modules-only arm: 48 updates × 4 examples = 192 scheduled
cases, seed 20261026, Muon int8 momentum with auxiliary FP32 AdamW, learning rate
0.001, no preservation KL and no checkpoint selection. Exactly **14 writer
 tensors / 5,251,072 parameters** were trainable. The original 2,516,756,480
backbone parameters, readers, gate scalars and all other components were frozen.

The [independent engineering audit](../../reports/memory_reliability_v5_attempt01/engineering_audit.json)
and raw update probes establish:

| Check | Observed |
|---|---:|
| Scheduled cases | 192 |
| Fast-state cases | 128 |
| Gate-off cases | 64 |
| Actual backward calls | 192 |
| Gate-off backward calls on non-fast graphs | 64 |
| Update-two nonzero writer gradients | 14/14 |
| Original or nonfinite gradients | 0 |
| Changed writer tensors | 14 |
| Unchanged extension tensors | 53 |

Gate-off disables fast state; it does not imply that every non-fast graph lacks
a backward call. Gradient probes are update-level accumulated evidence, not
per-example gradient attribution.

## 3. Independently decoded outcomes

The corrected scorer decoded all 1,920 rows with the sealed tokenizer and
reconstructed gold from the authenticated fixture. Counts below aggregate dev
and heldout; each split independently has the same fast/disabled/selective
accuracy except where full context is shown separately.

| Arm | Initial facts /64 | Changed /16 | Unaffected /48 | Unknown /32 | General /32 |
|---|---:|---:|---:|---:|---:|
| Candidate fast | 0 | 0 | 0 | 32 | 24 |
| Disabled | 0 | 0 | 0 | 32 | 24 |
| Shuffled | 0 | 0 | 0 | 32 | 24 |
| Last chunk | 0 | 0 | 0 | 32 | 24 |
| Adversarial | 0 | 0 | 0 | 32 | 24 |
| Ungated | 0 | 0 | 0 | 0 | 0 |
| Selective exact context | 64 | 16 | 48 | 32 | 24 |
| Full context | 64 | 16 | 48 | 31 | 17 |

Candidate fast and disabled have identical correctness sets: **56/192**. Their
token sequences differ on 128 factual routes, proving nonzero influence but no
useful answer change. Both splits score 0/32 initial facts, 0/8 corrections and
0/24 unaffected facts; both delayed-above-controls gates fail. General/unknown
retention and disabled-token parity pass, but this is bypass/noninterference
evidence, not learned retention or memory benefit. Candidate per-case
regressions against parent-disabled are empty because the factual cases already
fail and general outcomes match; factual nonregression is therefore vacuous.

The [failure analysis](../../reports/memory_reliability_v5_attempt01/failure_analysis.json)
preserves decoded comparisons. Selective exact context totals 184/192; full
context totals 176/192. Selective wins 10 per-case comparisons and loses two:
both `What does RAM stand for?` cases, where full context answers `Random Access
Memory` and selective context answers `RAM`. The prospective non-regression gate
correctly rejects those two losses despite selective context's higher aggregate.
Full context also emits `plum` for one absent heldout fact. Exact fact retrieval
is separately correct on all 128 fact/correction/unaffected cases; the reference
bundle still fails `reference_all_cases_correct` because its general score is
24/32, plus the two non-regression gates.

Eleven final gates fail: six per-split fact/changed/unaffected floors, both
delayed-above-controls gates, `reference_all_cases_correct`,
`reference_full_context_non_regression` and `reference_no_correct_to_wrong`.
This candidate is not a parent or default.

## 4. Persistence and scorer repair history

The final evaluation authenticates **32 images and 32 source bindings**, actual
tensor bytes, scopes, producers, six-to-seven lineage/counters and multiple
queries per state. Evaluation-only ledger epoch is 112. A fresh process decodes
seven corrected-stage answers and reproduces their token sequences, state
 digests, seven-entry lineage and source view with zero updates/checkpoints and
no source reencoding. The final verdict records these mechanical checks even
though the factual answers are wrong. This is not power-loss recovery, neural
unlearning or quality acceptance.

Two scorer defects were encountered and preserved rather than hidden:

1. The original scorer lacked the dataset import and incorrectly required every
   gate-off row to skip backward. `eff2079` restored the import and separated
   fast-state participation from recorded non-fast backward calls.
2. Eval01 then stopped before verdict because cached JSON string keys were
   compared with independent integer keys. `5aa7e75` added canonical JSON audit
   comparison. [Eval01's raw report](../../runs/memory_reliability_v5_scorer_fix_eval01/run_report.json)
   and guard remain preserved; it has no accepted verdict.

Eval02 repeated only evaluation with the same candidate, data and controls, zero
optimizer updates and zero checkpoints. The [canonical final verdict](../../reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json)
uses the corrected scorer and includes the restart. The original training report
was never edited or re-pinned.

## 5. Runtime, tokens and storage

All native jobs used MPS with CPU fallback disabled, one exclusive worker and
the 50 GiB sampled driver / 512 MiB sustained swap-growth / 15% minimum-free
policy. Samples can miss transients; MPS is not ANE and MPS/RSS overlap.

| Job | Child seconds | Guard wall seconds | Peak sampled MPS driver | Swap growth | Minimum free |
|---|---:|---:|---:|---:|---:|
| V5 training + evaluation | 637.228 | 639.93 | 11,815,763,968 B | 0 | 76% |
| Eval01 scoring replay | 548.200 | 550.66 | 11,815,763,968 B | 0 | 76% |
| Eval02 scoring replay | 548.855 | 551.38 | 11,815,763,968 B | 0 | 76% |
| Fresh-process restart | 7.182 | 9.72 | 11,815,763,968 B | 0 | 76% |

Total guarded wall across all four jobs is 1,751.69 seconds. Each full
parent/candidate evaluation generated 1,920 answers, 5,786 tokens and 134,120
prompt tokens. Restart generated seven answers / 28 tokens. No CPU training or
fallback was used.

One **48,948,272-byte extension** was saved at
`runs/memory_reliability_v5_attempt01/candidate_extension`, with base_saved=false.
The original run occupies 53,240 KiB; each evaluation replay occupies 5,084 KiB;
restart occupies 84 KiB. Eval02 images occupy about 2.4 MiB and its ledger 128
KiB. No duplicate native base checkpoint was written.

Raw evidence:
[training report](../../runs/memory_reliability_v5_attempt01/run_report.json),
[training guard](../../reports/memory_reliability_v5_attempt01.guard.json),
[eval02 report](../../runs/memory_reliability_v5_scorer_fix_eval02/run_report.json),
[eval02 guard](../../reports/memory_reliability_v5_scorer_fix_eval02.guard.json),
[restart report](../../runs/memory_reliability_v5_scorer_fix_restart01/run_report.json),
[restart guard](../../reports/memory_reliability_v5_scorer_fix_restart01.guard.json).

## 6. Implemented, trained, evaluated and pending

**Implemented:** consumed exact-record/span/query authentication; explicit
route/state/backward accounting; prospective non-regression scoring; diverse
retention fixture and audit; observed-fact writer representation; generic
trainable-scope and serialized-audit checks; source-bound six-to-seven images.

**Trained:** one 48-update writer-only candidate. **Evaluated:** the same
candidate in two no-training infrastructure replays, with the canonical eval02
scorer and restart. The candidate remains rejected. Engineering wiring and
persistence pass independently of learned quality.

**Pending / unrun:** a storage/read separation probe on fresh observed facts,
then an explicit copy/pointer comparison only when values are recoverable; any
new scientific arm, full-backbone/D2 training, visual training, calibration,
broad-memory claim or RSI promotion. Do not add updates, switch to another
writer/reader recipe or tune against this exposed evaluation set. Isolated vision
work with memory/core/identity disabled remains separate and cannot qualify the
failed memory path.
