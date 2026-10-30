# Takeover acceptance commands and evidence scope

Snapshot: 29 October 2026. Baseline HEAD `9e75415e558c3dcff6fc85516ced160ce21a54fd`.
The five v12 WIP files match `docs/handoffs/20261029_spark_intrinsic_memory/wip_manifest.json`.
This is a current command register, not a passing receipt. New command results must use new output paths and identify the exact reviewed commit.

## Protected and historical boundaries

Never read, import, edit, stage, collect, archive, or broadly search `src/replica_lab/runtime_pilot.py`. Preserve unrelated work and all historical result bytes. The consumed v11 native commands in `reports/intrinsic_memory_coverage_v11/README.md` are provenance only and MUST NOT be rerun.

## Affected regression inherited from v11

Run from the final clean source-only checkout with the protected module absent and no pretrained weights. Add every new v12, density, successor, and independent-review test file and `tests/test_intrinsic_memory_matched_reader_v1.py` to this exact inherited scope. Write JUnit/log/receipt files to the new review directory, never the historical v11 destinations.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_spark_device_placement_v2.py \
  tests/test_spark_execution_environment.py \
  tests/test_spark_derivative_bf16_protocol.py \
  tests/test_spark_persistent_learner_protocol.py \
  tests/test_baseline_competence.py \
  tests/test_contextual_memory.py \
  tests/test_contextual_readout_training.py \
  tests/test_intrinsic_memory_layout_v4.py \
  tests/test_contextual_memory_layout_v4.py \
  tests/test_intrinsic_memory_density_protocol.py \
  tests/test_tiered_memory.py \
  tests/test_tiered_recovery.py \
  tests/test_tiered_reset.py \
  tests/test_tiered_readout.py \
  tests/test_tiered_profiles.py \
  tests/test_tiered_selection_metadata.py \
  tests/test_persistent_learner_controls.py \
  tests/test_spark2_5_adapter.py \
  tests/test_contextual_memory_v5.py \
  tests/test_intrinsic_diagnostic_v5.py \
  tests/test_intrinsic_execution_v5.py \
  tests/test_wall_deadline_v1.py \
  tests/test_contextual_completion_objective_v1.py \
  tests/test_intrinsic_explicit_evidence_v6.py \
  tests/test_intrinsic_exposure_v7.py \
  tests/test_intrinsic_supported_values_v8.py \
  tests/test_intrinsic_smaller_step_v9.py \
  tests/test_contextual_teacher_objective_v1.py \
  tests/test_intrinsic_teacher_v10.py \
  tests/test_intrinsic_alignment_v10.py \
  tests/test_intrinsic_coverage_v11.py \
  tests/test_intrinsic_teacher_v11.py \
  tests/test_intrinsic_memory_matched_reader_v1.py \
  -q --junitxml=<new-review-directory>/regression.xml
```

Historical reference only: this inherited 32-file scope passed 562/562 at `71f7ab4d`; the separate matched-reader file passed 29/29 at `db405c87`. Those counts do not validate v12 and are not targets to manufacture.

## Established current-checkout checks

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked auto --no-write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts .venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/check_git_artifacts.py --scope staged
```

In the clean checkout, use its qualified Python, migration `--tracked require`, artifact `--scope tracked`, and new receipt destinations outside historical generated results. Verify each tool honors the protected-file exclusion before execution.

## Required new executable entry points

The implementation tracks must supply actual files and commands for:

- v12 focused tests and integrated tiny-BF16 CPU workflow;
- independent isolated reproduction and final review tests;
- density invocation profiling, all-layout CPU continuation, and three comparison-protocol validators;
- completed-interaction formation, chunk formation, tier validity/recovery, Replica adapters, and measured CPU workflows;
- versioned v12 environment-only, review, and exactly-once guarded native launch commands;
- density native commands only if the matching two-site useful-readout and all dependent gates pass.

A named future command is not executable until its script, protocol, packet, bindings, and output directory exist and its non-launch validation exits 0. Native execution remains blocked until integrated CPU evidence and independent review pass. A failed competence or fitting gate closes dependent scientific stages while preserving engineering evidence.

## Final checks

Run every command above and every newly supplied acceptance command against the final committed tree. Require exit status 0, retain logs and raw outputs, then run path-scoped `git diff --check`, `git status --short`, and record the final commit. Do not use broad operations that could include the protected module.
