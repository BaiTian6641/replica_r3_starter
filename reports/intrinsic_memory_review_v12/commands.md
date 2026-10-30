# Exact review commands

All paths below are absolute where the invocation used them. `ROOT` is `/Users/flare/replica_r3_starter`; no historical JUnit/log destination was reused.

## Isolated checkouts

```sh
git worktree add --no-checkout /private/tmp/replica-independent-v12-3cd683c3 3cd683c3
git -C /private/tmp/replica-independent-v12-3cd683c3 sparse-checkout init --no-cone
git -C /private/tmp/replica-independent-v12-3cd683c3 sparse-checkout set "/*" "!/src/replica_lab/runtime_pilot.py" "!/checkpoints/"
git -C /private/tmp/replica-independent-v12-3cd683c3 checkout
test ! -e /private/tmp/replica-independent-v12-3cd683c3/src/replica_lab/runtime_pilot.py
test ! -e /private/tmp/replica-independent-v12-3cd683c3/checkpoints

git worktree add --no-checkout /private/tmp/replica-independent-v12-2b788c46 2b788c46
git -C /private/tmp/replica-independent-v12-2b788c46 sparse-checkout init --no-cone
git -C /private/tmp/replica-independent-v12-2b788c46 sparse-checkout set "/*" "!/src/replica_lab/runtime_pilot.py" "!/checkpoints/"
git -C /private/tmp/replica-independent-v12-2b788c46 checkout
test ! -e /private/tmp/replica-independent-v12-2b788c46/src/replica_lab/runtime_pilot.py
test ! -e /private/tmp/replica-independent-v12-2b788c46/checkpoints
```

## Inherited + matched + new implementation regression

Working directory: `/private/tmp/replica-independent-v12-2b788c46` for final native and integrated checks; the 747-test invocation used `/private/tmp/replica-independent-v12-3cd683c3`.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/regression_3cd683c3.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest tests/test_spark_device_placement_v2.py tests/test_spark_execution_environment.py tests/test_spark_derivative_bf16_protocol.py tests/test_spark_persistent_learner_protocol.py tests/test_baseline_competence.py tests/test_contextual_memory.py tests/test_contextual_readout_training.py tests/test_intrinsic_memory_layout_v4.py tests/test_contextual_memory_layout_v4.py tests/test_intrinsic_memory_density_protocol.py tests/test_tiered_memory.py tests/test_tiered_recovery.py tests/test_tiered_reset.py tests/test_tiered_readout.py tests/test_tiered_profiles.py tests/test_tiered_selection_metadata.py tests/test_persistent_learner_controls.py tests/test_spark2_5_adapter.py tests/test_contextual_memory_v5.py tests/test_intrinsic_diagnostic_v5.py tests/test_intrinsic_execution_v5.py tests/test_wall_deadline_v1.py tests/test_contextual_completion_objective_v1.py tests/test_intrinsic_explicit_evidence_v6.py tests/test_intrinsic_exposure_v7.py tests/test_intrinsic_supported_values_v8.py tests/test_intrinsic_smaller_step_v9.py tests/test_contextual_teacher_objective_v1.py tests/test_intrinsic_teacher_v10.py tests/test_intrinsic_alignment_v10.py tests/test_intrinsic_coverage_v11.py tests/test_intrinsic_teacher_v11.py tests/test_intrinsic_memory_matched_reader_v1.py tests/test_successor_adapters_v12.py tests/test_successor_chunks_v12.py tests/test_successor_formation_v12.py tests/test_successor_tiers_v12.py tests/test_contextual_source_features_v12.py tests/test_contextual_memory_layout_v12.py tests/test_contextual_memory_v12.py tests/test_contextual_v12_integration.py tests/test_intrinsic_memory_density_v12_matched.py tests/test_intrinsic_memory_density_v12_producer.py tests/test_intrinsic_memory_density_v12_profiler.py tests/test_intrinsic_memory_density_v12_protocols.py -q --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/regression_3cd683c3.xml
```

## Final native focused tests

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/native_focused_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q tests/test_intrinsic_budget_v12.py tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/native_focused_2b788c46.xml
```

## Independent contract reproductions

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/independent_reproductions_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q tests/test_independent_review_v12_findings.py --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/independent_reproductions_2b788c46.xml
```

## New integrated tiny-BF16 workflow

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/integrated_cpu_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_contextual_cpu_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/integrated_cpu_2b788c46
```

## Established checks

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/migration_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked require --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/final_product_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/bundle_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/artifacts_2b788c46.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/check_git_artifacts.py --scope tracked
```

## Native closure and environment review only

Working directory: `ROOT`; these modes do not import/load a model and were not launches.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/native_review_2b788c46.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --review
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/native_environment_2b788c46.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --environment-only
```
