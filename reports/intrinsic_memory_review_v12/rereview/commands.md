# Exact c95d3e23 re-review commands

`ROOT=/Users/flare/replica_r3_starter`; `CLEAN=/private/tmp/replica-independent-v12-c95d3e23-rereview`. Every output is under `reports/intrinsic_memory_review_v12/rereview/`; the negative report was not modified.

## Protected-absent source-only checkout

```sh
git worktree add --no-checkout /private/tmp/replica-independent-v12-c95d3e23-rereview c95d3e23
git -C /private/tmp/replica-independent-v12-c95d3e23-rereview sparse-checkout init --no-cone
git -C /private/tmp/replica-independent-v12-c95d3e23-rereview sparse-checkout set "/*" "!/src/replica_lab/runtime_pilot.py" "!/checkpoints/"
git -C /private/tmp/replica-independent-v12-c95d3e23-rereview checkout
test ! -e /private/tmp/replica-independent-v12-c95d3e23-rereview/src/replica_lab/runtime_pilot.py
test ! -e /private/tmp/replica-independent-v12-c95d3e23-rereview/checkpoints
```

## Full regression, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/full_regression_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest tests/test_spark_device_placement_v2.py tests/test_spark_execution_environment.py tests/test_spark_derivative_bf16_protocol.py tests/test_spark_persistent_learner_protocol.py tests/test_baseline_competence.py tests/test_contextual_memory.py tests/test_contextual_readout_training.py tests/test_intrinsic_memory_layout_v4.py tests/test_contextual_memory_layout_v4.py tests/test_intrinsic_memory_density_protocol.py tests/test_tiered_memory.py tests/test_tiered_recovery.py tests/test_tiered_reset.py tests/test_tiered_readout.py tests/test_tiered_profiles.py tests/test_tiered_selection_metadata.py tests/test_persistent_learner_controls.py tests/test_spark2_5_adapter.py tests/test_contextual_memory_v5.py tests/test_intrinsic_diagnostic_v5.py tests/test_intrinsic_execution_v5.py tests/test_wall_deadline_v1.py tests/test_contextual_completion_objective_v1.py tests/test_intrinsic_explicit_evidence_v6.py tests/test_intrinsic_exposure_v7.py tests/test_intrinsic_supported_values_v8.py tests/test_intrinsic_smaller_step_v9.py tests/test_contextual_teacher_objective_v1.py tests/test_intrinsic_teacher_v10.py tests/test_intrinsic_alignment_v10.py tests/test_intrinsic_coverage_v11.py tests/test_intrinsic_teacher_v11.py tests/test_intrinsic_memory_matched_reader_v1.py tests/test_successor_adapters_v12.py tests/test_successor_chunks_v12.py tests/test_successor_formation_v12.py tests/test_successor_tiers_v12.py tests/test_contextual_source_features_v12.py tests/test_contextual_memory_layout_v12.py tests/test_contextual_memory_v12.py tests/test_contextual_v12_integration.py tests/test_intrinsic_memory_density_v12_matched.py tests/test_intrinsic_memory_density_v12_producer.py tests/test_intrinsic_memory_density_v12_profiler.py tests/test_intrinsic_memory_density_v12_protocols.py tests/test_intrinsic_budget_v12.py tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py tests/test_intrinsic_memory_density_v12_repairs.py tests/test_replica_successor_repairs_v1.py tests/test_independent_review_v12_findings.py -q --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/full_regression_c95d3e23.xml
```

## New residual reproductions, run from CLEAN

The review-only file was copied to `rereview_edge_cases.py` in CLEAN for collection; no production test was edited.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/edge_reproductions_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q rereview_edge_cases.py --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/edge_reproductions_c95d3e23.xml
```

## Integrated and track workflows, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/integrated_cpu_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_contextual_cpu_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/integrated_cpu_c95d3e23
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 TMPDIR=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/density_profile_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/profile_intrinsic_memory_density_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/density_profile_c95d3e23.json
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 TMPDIR=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/density_pilot_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_memory_density_v12_cost_pilot.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/density_pilot_c95d3e23.json
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/replica_workflow_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/replica_successors
```

## Established checks, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/migration_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked require --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/final_product_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/bundle_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/rereview/artifacts_c95d3e23.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/check_git_artifacts.py --scope tracked
```

## Native review only, run from ROOT

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/rereview/native_review_c95d3e23.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --review
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/rereview/native_environment_c95d3e23.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --environment-only
```
