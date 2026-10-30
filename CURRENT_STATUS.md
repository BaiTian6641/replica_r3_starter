# Current Spark intrinsic-memory status

**30 October 2026 — pre-MLP attempt01 failed final fitting and no native worker is active.** Contextual was 25/32 with 9/16 pairs; development stayed closed. Fresh-process restart passed with exact logits and one continuation update. Useful recall and generalization remain unestablished.

## Authoritative current evidence

- Native protocol and exact packet: `configs/spark_intrinsic_coverage_v12.json` and `reports/intrinsic_memory_native_v12/native_packet_two_v12.json` (packet SHA-256 `5eebe66c45b15c186ffd697b746ff9373a4061a1ea369a4df670b0cea9ecd609`).
- Machine-derived current budget: `reports/intrinsic_memory_native_v12/current_packet_budget.json`. It verifies the packet's protocol hash and recomputes totals from the 23 stage budgets: 7,155 reserved stage seconds inside 7,200 outer seconds, 9,778 forwards, 949 source-feature forwards, 513 updates/backwards, 7,256 generated tokens, 4,669 prompt tokenizations, 3,618 writes, 55,490,098,857 reserved read bytes, and 205,537,280 artifact bytes.
- Native resource limits: 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free, one worker.
- Experiment history and gates: `docs/experiments/20261030_Spark_Memory_Contextual_V12.md`. Attempts01–10 are preserved. Attempt10 completed 512 updates: contextual 24/32, shuffled 24/32, exact text 32/32, wrong 6/32, zero 5/32, absent 8/32; groups were 6/8, 5/8, 6/8, 7/8 and 8/16 counterfactual pairs had both sides correct.
- Independent terminal reconciliation: `reports/intrinsic_memory_native_v12/attempt10_terminal_reconciliation.json`. Baseline was 8/8 and teacher 32/32 with zero false recall/truncation; all final controls terminated, live/durable parity was 32/32, original weights were unchanged, and development had zero read events. Restart restored the extension but failed on missing `bound_json` before continuation.
- Independent CPU launch-gate review: `reports/intrinsic_memory_review_v12/final_launch_gate/verification_summary.json` — 1,024 tests passed with zero skips; this qualified engineering and packet review, not native learning.
- Exact external reproduction bytes, hashes, JUnit and command lineage: `reports/intrinsic_memory_review_v12/final_launch_gate/external_test_manifest.json`. External cases remain under review directories, separate from committed production tests in `tests/`.
- Density/cost CPU evidence: `reports/intrinsic_memory_density_v12/coordinator13/`; native 2/9/36 utility remains unmeasured.
- Formation/chunk/tier/Replica CPU evidence: `reports/replica_successors_v12/final_verification.json`; these contracts do not establish learned retention, personal fidelity, or RSI.
- Research boundaries: `docs/research/spark_memory_20261029/Contextual_Compression_V12_Crosscheck.md` and `Research_Crosscheck.md`.
- Successor engineering: full-lifecycle fixture and restart repair `ca5144cc`; incremental WorkLedger evidence `reports/intrinsic_memory_successors_v13/`; response/correction contracts `src/replica_lab/intrinsic_memory_acceptance_v13.py`.
- Counterfactual diagnosis and next mechanism: `docs/experiments/20261030_Spark_Counterfactual_Diagnosis_V13.md` plus `reports/intrinsic_memory_counterfactual_v13/attempt03_mechanism_decision.json`. Native attempt03 completed with all intermediate tensors preserved.
- Pre-MLP terminal protocol and evidence: `docs/experiments/20261030_Spark_Pre_MLP_Fusion_V14.md` and `reports/intrinsic_memory_pre_mlp_v14/`.
- Bounded release/archive costs: `reports/intrinsic_memory_successors_v13/release_costs_attempt10.json` — 128 rebuilds, 281.676 seconds and 8,557,737 current archive bytes; incremental migration is unmeasured.


### Evidence-gated blockers

- Native 2/9/36 density and matched-cost utility remain unmeasured because useful two-site recall failed.
- Spark formation/Replica connection is blocked by the same prerequisite; existing CPU formation, chunk, tier and adapter contracts passed 146 focused tests.
- Persistent learning, nested adaptation, personal fidelity and RSI remain closed. No P updates or capacity expansion occurred.
- Reader/query alignment is a documented fallback only, not an authorized running arm. No additional mechanism is silently selected here.
- Large-scale and incremental migration remain unmeasured; bounded release cost and current archive size are recorded instead.

## Learning and capability state

Useful intrinsic recall is **not established**. Attempt10 failed its fixed final fitting gate; no best prefix was selected and development was not measured. Fresh-process continuation did not complete. Formation, persistent learning, density promotion, personal fidelity, nested adaptation, and RSI remain evidence-gated.

The external spine remains the typed interface for specialist models, sensors, and actors. Intrinsic memory owns Spark recall and learning. E/S/P/φ/C/I/K, source authority, task state, tier placement, and learning clocks remain distinct. P1 and NativeFusedAdamW remain fixed; P updates, capacity expansion, and native multimodal work are not approved by this status.

## Reproduce current packet accounting

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/inspect_spark_intrinsic_packet_v12.py \
  --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
  --output reports/intrinsic_memory_native_v12/current_packet_budget.json
```

The completed experiment was **pre-MLP fusion placement at the existing 3/7 sites**, with frozen source features, four slots, objective, optimizer and fitting rows fixed. It changed contextual by only +1/32 and did not pass any final group, so placement alone is insufficient. The declared fallback is **reader/query alignment with source representation fixed**, not more slots or another encoder/objective change. Density, formation and persistent learning remain gated; no held-out labels opened.

Historical v11 and attempt01–10 material remains evidence, not current navigation. Protected and unrelated dirty work remains untouched; `src/replica_lab/runtime_pilot.py` must never be read, imported, edited, or staged.
