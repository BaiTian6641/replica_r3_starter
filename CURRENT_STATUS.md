# Current Spark intrinsic-memory status

**30 October 2026 — official-BF16 v12 attempt10 is running under one exclusive MPS worker.** Its bound implementation/review snapshot is `44e69959`; later documentation and evidence commits do not alter that implementation. CPU fallback is disabled. Do not launch competing native work or change the bound protocol, gates, packet, authorization, or output.

## Authoritative current evidence

- Native protocol and exact packet: `configs/spark_intrinsic_coverage_v12.json` and `reports/intrinsic_memory_native_v12/native_packet_two_v12.json` (packet SHA-256 `5eebe66c45b15c186ffd697b746ff9373a4061a1ea369a4df670b0cea9ecd609`).
- Machine-derived current budget: `reports/intrinsic_memory_native_v12/current_packet_budget.json`. It verifies the packet's protocol hash and recomputes totals from the 23 stage budgets: 7,155 reserved stage seconds inside 7,200 outer seconds, 9,778 forwards, 949 source-feature forwards, 513 updates/backwards, 7,256 generated tokens, 4,669 prompt tokenizations, 3,618 writes, 55,490,098,857 reserved read bytes, and 205,537,280 artifact bytes.
- Native resource limits: 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free, one worker.
- Experiment history and gates: `docs/experiments/20261030_Spark_Memory_Contextual_V12.md`. Attempts01–09 are preserved. Attempt09 reached 384 updates; its completed 128/256-update probes failed memory dependence, but partial probes are not final v12 outcomes.
- Independent CPU launch-gate review: `reports/intrinsic_memory_review_v12/final_launch_gate/verification_summary.json` — 1,024 tests passed with zero skips; this qualified engineering and packet review, not native learning.
- Exact external reproduction bytes, hashes, JUnit and command lineage: `reports/intrinsic_memory_review_v12/final_launch_gate/external_test_manifest.json`. External cases remain under review directories, separate from committed production tests in `tests/`.
- Density/cost CPU evidence: `reports/intrinsic_memory_density_v12/coordinator13/`; native 2/9/36 utility remains unmeasured.
- Formation/chunk/tier/Replica CPU evidence: `reports/replica_successors_v12/final_verification.json`; these contracts do not establish learned retention, personal fidelity, or RSI.
- Research boundaries: `docs/research/spark_memory_20261029/Contextual_Compression_V12_Crosscheck.md` and `Research_Crosscheck.md`.

## Learning and capability state

Useful intrinsic recall is **not yet established**. Attempt10 must terminate before reconciling final fitting groups, counterfactual pairs, complete responses, termination, parity, resource use, restart, and development exposure. If final fitting fails, development stays closed; no best-prefix selection is allowed. Formation, persistent learning, density promotion, personal fidelity, nested adaptation, and RSI remain evidence-gated.

The external spine remains the typed interface for specialist models, sensors, and actors. Intrinsic memory owns Spark recall and learning. E/S/P/φ/C/I/K, source authority, task state, tier placement, and learning clocks remain distinct. P1 and NativeFusedAdamW remain fixed; P updates, capacity expansion, and native multimodal work are not approved by this status.

## Reproduce current packet accounting

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/inspect_spark_intrinsic_packet_v12.py \
  --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
  --output reports/intrinsic_memory_native_v12/current_packet_budget.json
```

The next successor work is separate from sealed v12: a tiny/fake full-lifecycle fixture; measured incremental/checkpointed WorkLedger replay; typed correction and complete-response contracts; fitting-only counterfactual diagnosis; then one preregistered mechanism chosen from that diagnosis. No held-out labels open before the corresponding gates pass.

Historical v11 and attempt01–09 material remains evidence, not current navigation. Protected and unrelated dirty work remains untouched; `src/replica_lab/runtime_pilot.py` must never be read, imported, edited, or staged.
