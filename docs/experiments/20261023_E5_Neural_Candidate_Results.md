# E5 neural candidate: reranker accepted over parent, still below lexical

23 October 2026 UTC. Protocol `configs/e5_neural_candidate_eval_v1.json`
committed before execution. No new optimization: this is an independent
evaluation of the existing E2 reranker candidate against its parent.

**Result: candidate accepted relative to its parent (held-out record top-1
0.25 → 0.50), but lexical remains better (0.75) and is still the default. The
candidate is opt-in only; this is not a task-capability or generalization
result.**

## 1. Artifacts

| Arm | Artifact |
|---|---|
| Parent | `runs/e1_e2_native_attempt02_20261023/selected_extension` |
| Candidate | `runs/e2_retrieval_attempt03_20261023/selected_extension` (E2 reranker + block-13 sensor trained by the explicit ranking objective) |
| Data | committed `data/e2_retrieval`; candidate trained only on train source groups |
| Evaluation | dev groups + held-out source groups; no new training |

## 2. Measurements

The reranker scores eligible lexical candidates; one max span score per record;
the bank is capped at 16 units.

| Metric | Parent | Candidate | Lexical reference |
|---|---:|---:|---:|
| Held-out recall@1 | 0.25 (1/4) | **0.50 (2/4)** | **0.75** |
| Held-out recall@4 | 1.00 | 1.00 | 1.00 |
| Dev recall@1 | 0.00 (0/2) | 0.00 (0/2) | 1.00 |

Per-case selected IDs, ranks and scores are preserved in
[attempt 2](../../reports/e5_neural_candidate_attempt02_20261023/run_report.json).
The parent/candidate weights differ only through the explicitly trained
reranker/sensor extension bytes; the sealed base is unchanged.

Decision by the preregistered rule: **accept** (candidate held-out top-1 above
parent and dev not below parent). Deployment: **opt-in reranker only**; do not
change the lexical default. Attempt 1 recorded the same metrics but marked the
artifact `proposed`; it is preserved as a reporting defect.

## 3. Interpretation and limits

This is a genuine parameter-level candidate comparison on unseen source groups:
the trained reranker improves over its parent. It is not a useful default
because lexical ordering beats it, and the dev set is small (2 cases) and
already exposed. Answer accuracy is unchanged (the reader is the separate
novel-value bottleneck). No calibration, retention on a broad task suite,
learned action policy, or publication-to-default follows.

The candidate artifact is `accepted` in the typed registry with
`evaluation=passed`, but no active model manifest was swapped. The procedure
candidate remains the accepted procedure baseline; this neural candidate is an
opt-in retrieval component.

## 4. Cost

Two native model loads, 9.95 s profiled, **11.0 GiB** sampled peak driver, no
optimizer update, no checkpoint save, no download. Both evaluation attempts and
all raw rankings are retained.

## 5. Next

- Do not retune this held-out set.
- A separate neural candidate should declare a new training/evaluation family
  and must beat both its parent and the lexical baseline before becoming a
  default.
- E4 delayed-use and authentic C16 fidelity remain pending.
