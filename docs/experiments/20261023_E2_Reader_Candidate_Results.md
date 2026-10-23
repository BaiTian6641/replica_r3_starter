# E2 reader candidate: answer-loss training does not transfer novel values

23 October 2026 UTC. Protocol/data committed before execution. **Result:
deferred/not published.** Parent and candidate both score **0/16** on dev +
held-out; the candidate changes its old answer labels but does not copy the
held-out values. This directly confirms the earlier writer/reader diagnosis and
rules out “no answer-loss training” as the sole cause.

## 1. Data and training

- 24 train cases, 6 dev, 10 held-out; source groups disjoint.
- One relevant short record and three eligible distractors per case (same
  entity/different attribute; same attribute/different entity; owner).
- Held-out entities and values are absent from training.
- D1 modules-only candidate: `record_writer` + both `memory_ports`, Muon 8-bit
  and FP32 AdamW, 16 updates × 4 examples, lr 1e-2, warmup 2, seed 20261023.
- Answer-only cross-entropy over answer + EOS; prompt prefix masked. No
  reranker/action policy changes; one extension-only export; base unchanged.

## 2. Results

| Split | Parent exact | Candidate exact |
|---|---:|---:|
| Dev (6) | 0/6 | 0/6 |
| Held-out (10) | 0/10 | 0/10 |

The candidate's training loss falls from **15.706 to 8.318** (sum over four
examples), but it emits values seen in training (`silver`, `ochre`, `violet`,
`saffron`, `blue`) rather than the held-out record values (`azure`, `mauve`,
`vermilion`, `chartreuse`, …). The lexical retriever selects the relevant record
for all cases, so this is not a retrieval failure. Parent answers are mostly
`amber`/`white`; candidate answers shift but remain non-copying.

Decision by the preregistered rule: **defer** (held-out not above parent; dev not
below). No default reader, reranker, or active manifest was changed. The
candidate extension is preserved as a diagnostic artifact.

## 3. Diagnosis

The earlier oracle/exact-text diagnostic localized the bottleneck to the writer/
reader path. This candidate adds 24 answer-loss examples and still fails novel
held-out values, so the remaining causes are **representation/optimization
capacity or unsuitable supervision/writer geometry**, not the absence of answer
loss. It does not establish that more data cannot help; it rules out this small
D1 budget as sufficient.

The exact-text fallback remains preferred for novel-value precision. The next
targeted intervention should modify the reader/writer mechanism or supervision
(e.g., copy-oriented span supervision, stronger per-record writer training, or
an explicit value-copy objective), on a new held-out family. Do not retune these
10 held-out cases.

## 4. Costs

36.17 s profiled, **11.0 GiB** sampled peak driver, 16 optimizer updates, one
19.52 MB extension, no base copy, no download. Raw parent/candidate answers,
selected record IDs and token IDs are in
[the report](../../reports/e2_reader_candidate_20261023/run_report.json).

## 5. Status

Implemented: source-separated reader data, D1 answer-loss trainer, parent/
candidate evaluation, extension-only export.
Trained: 16 updates on 24 cases.
Evaluated: deferred; 0/16 candidate, 0/16 parent.
Not established: novel-value reading, reader capacity, retention, generalization.
