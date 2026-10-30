# Exact final verification commands

Source tree: `bbcee1f3`. Run from `/private/tmp/replica-final-spark-c1069403` unless noted. Environment: `PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts`.

## Full committed regression

```sh
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/full_regression.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest tests/test_spark_device_placement_v2.py tests/test_spark_execution_environment.py tests/test_spark_derivative_bf16_protocol.py tests/test_spark_persistent_learner_protocol.py tests/test_baseline_competence.py tests/test_contextual_memory.py tests/test_contextual_readout_training.py tests/test_intrinsic_memory_layout_v4.py tests/test_contextual_memory_layout_v4.py tests/test_intrinsic_memory_density_protocol.py tests/test_tiered_memory.py tests/test_tiered_recovery.py tests/test_tiered_reset.py tests/test_tiered_readout.py tests/test_tiered_profiles.py tests/test_tiered_selection_metadata.py tests/test_persistent_learner_controls.py tests/test_spark2_5_adapter.py tests/test_contextual_memory_v5.py tests/test_intrinsic_diagnostic_v5.py tests/test_intrinsic_execution_v5.py tests/test_wall_deadline_v1.py tests/test_contextual_completion_objective_v1.py tests/test_intrinsic_explicit_evidence_v6.py tests/test_intrinsic_exposure_v7.py tests/test_intrinsic_supported_values_v8.py tests/test_intrinsic_smaller_step_v9.py tests/test_contextual_teacher_objective_v1.py tests/test_intrinsic_teacher_v10.py tests/test_intrinsic_alignment_v10.py tests/test_intrinsic_coverage_v11.py tests/test_intrinsic_teacher_v11.py tests/test_intrinsic_memory_matched_reader_v1.py tests/test_successor_adapters_v12.py tests/test_successor_chunks_v12.py tests/test_successor_formation_v12.py tests/test_successor_tiers_v12.py tests/test_contextual_source_features_v12.py tests/test_contextual_memory_layout_v12.py tests/test_contextual_memory_v12.py tests/test_contextual_v12_integration.py tests/test_intrinsic_memory_density_v12_matched.py tests/test_intrinsic_memory_density_v12_producer.py tests/test_intrinsic_memory_density_v12_profiler.py tests/test_intrinsic_memory_density_v12_protocols.py tests/test_intrinsic_budget_v12.py tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py tests/test_intrinsic_memory_density_v12_repairs.py tests/test_intrinsic_memory_density_v12_receipts.py tests/test_replica_successor_repairs_v1.py tests/test_independent_review_v12_findings.py final_launch_edge_cases.py final_launch_residual_edges.py final_launch_policy_edges.py final_launch_budget_edges.py final_launch_origin_budget.py final_launch_compound_public.py final_launch_fallback.py -q --junitxml=/Users/flare/replica_r3_starter/reports/final_verification_20261030/full_regression.xml
```

## Changed components

```sh
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/changed_components.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q tests/test_intrinsic_acceptance_v13.py tests/test_intrinsic_budget_v13.py tests/test_intrinsic_lifecycle_fixture_v13.py tests/test_intrinsic_counterfactual_v13.py tests/test_intrinsic_pre_mlp_v14.py tests/test_intrinsic_budget_v12.py tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py --junitxml=/Users/flare/replica_r3_starter/reports/final_verification_20261030/changed_components.xml
```

## Integrated and track workflows

```sh
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/integrated_cpu.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_contextual_cpu_v12.py --output /Users/flare/replica_r3_starter/reports/final_verification_20261030/integrated_cpu
TMPDIR=/Users/flare/replica_r3_starter/reports/final_verification_20261030/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/density_profile.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/profile_intrinsic_memory_density_v12.py --output /Users/flare/replica_r3_starter/reports/final_verification_20261030/density_profile.json
TMPDIR=/Users/flare/replica_r3_starter/reports/final_verification_20261030/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/density_pilot.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_memory_density_v12_cost_pilot.py --output /Users/flare/replica_r3_starter/reports/final_verification_20261030/density_pilot.json
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/replica_workflow.time /Users/flare/replica_r3_starter/.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py --output /Users/flare/replica_r3_starter/reports/final_verification_20261030/replica_successors
```

## Established checks

```sh
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/migration.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked require --no-write
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/final_product.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/bundle.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
/usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/final_verification_20261030/artifacts.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/check_git_artifacts.py --scope tracked
```

## Native review only, no model import or launch

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_pre_mlp_v14/native_packet_two.json --review
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_pre_mlp_v14/native_packet_two.json --environment-only
```
