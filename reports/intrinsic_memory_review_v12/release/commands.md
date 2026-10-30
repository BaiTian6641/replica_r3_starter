# Exact release commands

`ROOT=/Users/flare/replica_r3_starter`; `CLEAN=/private/tmp/replica-independent-v12-258679fa-release`. No overlay was used. All outputs are under `reports/intrinsic_memory_review_v12/release/`.

## Protected-absent exact checkout and external tests

```sh
git worktree add --no-checkout /private/tmp/replica-independent-v12-258679fa-release 258679fa
git -C /private/tmp/replica-independent-v12-258679fa-release sparse-checkout init --no-cone
git -C /private/tmp/replica-independent-v12-258679fa-release sparse-checkout set "/*" "!/src/replica_lab/runtime_pilot.py" "!/checkpoints/"
git -C /private/tmp/replica-independent-v12-258679fa-release checkout
test ! -e /private/tmp/replica-independent-v12-258679fa-release/src/replica_lab/runtime_pilot.py
test ! -e /private/tmp/replica-independent-v12-258679fa-release/checkpoints
test -f /private/tmp/replica-independent-v12-258679fa-release/src/replica_lab/intrinsic_memory_density_receipts_v12.py
git -C /private/tmp/replica-independent-v12-258679fa-release ls-files --error-unmatch src/replica_lab/intrinsic_memory_density_receipts_v12.py
cp reports/intrinsic_memory_review_v12/rereview/test_rereview_edge_cases.py /private/tmp/replica-independent-v12-258679fa-release/release_edge_cases.py
cp reports/intrinsic_memory_review_v12/final/test_final_residual_edges.py /private/tmp/replica-independent-v12-258679fa-release/release_residual_edges.py
cp reports/intrinsic_memory_review_v12/acceptance/test_acceptance_policy_edges.py /private/tmp/replica-independent-v12-258679fa-release/release_policy_edges.py
cp reports/intrinsic_memory_review_v12/qualified/test_qualified_budget_edges.py /private/tmp/replica-independent-v12-258679fa-release/release_budget_edges.py
cp reports/intrinsic_memory_review_v12/final_acceptance/test_final_acceptance_origin_budget.py /private/tmp/replica-independent-v12-258679fa-release/release_origin_budget.py
cp reports/intrinsic_memory_review_v12/final_acceptance/test_final_acceptance_compound_public.py /private/tmp/replica-independent-v12-258679fa-release/release_compound_public.py
```

## Origin and compound first, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/origin_and_compound_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q release_origin_budget.py release_compound_public.py --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/origin_and_compound_258679fa.xml
```

## Exact committed regression, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/full_regression_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest tests/test_spark_device_placement_v2.py tests/test_spark_execution_environment.py tests/test_spark_derivative_bf16_protocol.py tests/test_spark_persistent_learner_protocol.py tests/test_baseline_competence.py tests/test_contextual_memory.py tests/test_contextual_readout_training.py tests/test_intrinsic_memory_layout_v4.py tests/test_contextual_memory_layout_v4.py tests/test_intrinsic_memory_density_protocol.py tests/test_tiered_memory.py tests/test_tiered_recovery.py tests/test_tiered_reset.py tests/test_tiered_readout.py tests/test_tiered_profiles.py tests/test_tiered_selection_metadata.py tests/test_persistent_learner_controls.py tests/test_spark2_5_adapter.py tests/test_contextual_memory_v5.py tests/test_intrinsic_diagnostic_v5.py tests/test_intrinsic_execution_v5.py tests/test_wall_deadline_v1.py tests/test_contextual_completion_objective_v1.py tests/test_intrinsic_explicit_evidence_v6.py tests/test_intrinsic_exposure_v7.py tests/test_intrinsic_supported_values_v8.py tests/test_intrinsic_smaller_step_v9.py tests/test_contextual_teacher_objective_v1.py tests/test_intrinsic_teacher_v10.py tests/test_intrinsic_alignment_v10.py tests/test_intrinsic_coverage_v11.py tests/test_intrinsic_teacher_v11.py tests/test_intrinsic_memory_matched_reader_v1.py tests/test_successor_adapters_v12.py tests/test_successor_chunks_v12.py tests/test_successor_formation_v12.py tests/test_successor_tiers_v12.py tests/test_contextual_source_features_v12.py tests/test_contextual_memory_layout_v12.py tests/test_contextual_memory_v12.py tests/test_contextual_v12_integration.py tests/test_intrinsic_memory_density_v12_matched.py tests/test_intrinsic_memory_density_v12_producer.py tests/test_intrinsic_memory_density_v12_profiler.py tests/test_intrinsic_memory_density_v12_protocols.py tests/test_intrinsic_budget_v12.py tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py tests/test_intrinsic_memory_density_v12_repairs.py tests/test_intrinsic_memory_density_v12_receipts.py tests/test_replica_successor_repairs_v1.py tests/test_independent_review_v12_findings.py release_edge_cases.py release_residual_edges.py release_policy_edges.py release_budget_edges.py release_origin_budget.py release_compound_public.py -q --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/full_regression_258679fa.xml
```

## Budget and artifact probes, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/budget_edges_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q release_budget_edges.py --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/budget_edges_258679fa.xml
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:tests:scripts /Users/flare/replica_r3_starter/.venv/bin/python -m pytest -q release_budget_edges.py::test_probe_artifact_bytes_cover_value_payloads_and_origin_metadata --junitxml=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/probe_artifact_budget_258679fa.xml
```

## Integrated and track workflows, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/integrated_cpu_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_contextual_cpu_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/integrated_cpu_258679fa
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 TMPDIR=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_profile_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/profile_intrinsic_memory_density_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_profile_258679fa.json
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 TMPDIR=/Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/tmp /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_pilot_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/run_intrinsic_memory_density_v12_cost_pilot.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_pilot_258679fa.json
mkdir -p /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density
cp /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_profile_258679fa.json /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density/cpu_engineering_summary.json
cp /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_pilot_258679fa.json /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density/cost_pilot_summary.json
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_receipts_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/write_intrinsic_memory_density_v12_receipts.py --directory /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density --cpu-summary /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density/cpu_engineering_summary.json --cost-pilot /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density/cost_pilot_summary.json
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/density_navigation_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/write_intrinsic_memory_density_v12_navigation.py --canonical /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/canonical_density --root /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/replica_workflow_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py --output /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/replica_successors
```

## Established checks, run from CLEAN

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/migration_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked require --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/final_product_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/bundle_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l -o /Users/flare/replica_r3_starter/reports/intrinsic_memory_review_v12/release/artifacts_258679fa.time /Users/flare/replica_r3_starter/.venv/bin/python scripts/check_git_artifacts.py --scope tracked
```

## Native public review only, run from ROOT

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/release/native_review_258679fa.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --review
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 /usr/bin/time -l -o reports/intrinsic_memory_review_v12/release/native_environment_258679fa.time .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json --environment-only
```
