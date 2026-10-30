# Final Spark intrinsic-memory verification — 30 October 2026

Verified source tree: `bbcee1f3`. Clean checkout: `/private/tmp/replica-final-spark-c1069403`, sparse-excluding `src/replica_lab/runtime_pilot.py` and `checkpoints/`. Both exclusions were tested absent. No overlay was used. External reproduction filenames were written with the exact preserved bytes; all seven hashes match `reports/intrinsic_memory_review_v12/final_launch_gate/external_test_manifest.json`.

## Passing results

| Check | Result | Evidence |
|---|---:|---|
| Full clean committed regression | **1024 passed, 0 failed/skipped** | `full_regression.xml`, `full_regression.time` |
| Changed-component clean suite | **117 passed** | `changed_components.xml`, `changed_components.time` |
| Integrated tiny-BF16 2/9/36 workflow | Passed; all-site gradients and continuation | `integrated_cpu/`, `integrated_cpu.time` |
| Density profile and measured-cost pilot | All three arms matched | `density_profile.json`, `density_pilot.json` |
| Replica formation/chunk/tier workflow | Passed | `replica_successors/`, `replica_workflow.time` |
| Migration verification | **1492 passed** | `migration.time` |
| Final-product verification | **552 passed** | `final_product.time` |
| Final-product bundle | **293 inputs**, digest `f505d70f5d453bb289fde9c0ee7e1d96af5fa14dfa10a2c0cbe3c7501fa0843c` | `bundle.time` |
| Tracked artifact guard | **21,025 files**, zero violations | `artifacts.time` |
| Pre-MLP native protocol review | Passed; no model imported | `reports/intrinsic_memory_pre_mlp_v14/native_packet_two.json` |
| Native environment-only review | Passed; no model imported | command output |

The main working-tree changed-component run also passed 117 tests in 114.54 seconds.

## Preserved unsuccessful verification attempts

1. The first full-regression invocation did not start because its output directory was absent.
2. The next run had three stale protocol lineage failures. They were repaired in `a3b1a550`; the clean rerun passed 1024.
3. Migration initially reported eight documentation identity failures. Versioned navigation updates in `81a36947` resolved them.
4. Final product initially reported the stale adapter identity and two receipt-hash failures. Reviewed input and bundle updates in `bbcee1f3` resolved them; the final run passed 552 and the bundle verified.
5. The binary counterfactual tensors were rejected by the artifact guard and remain intentionally outside Git. `reports/intrinsic_memory_counterfactual_v13/external_tensor_manifest.json` and `external_tensor_verification.json` bind all 16 files by SHA-256 and byte count.

No failure is hidden by a later pass.

## Commands

The exact 1024-test command and external-file hashes are preserved in this directory's `commands.md` lineage and `full_regression.xml`. Changed-component command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python -m pytest -q \
  tests/test_intrinsic_acceptance_v13.py tests/test_intrinsic_budget_v13.py \
  tests/test_intrinsic_lifecycle_fixture_v13.py tests/test_intrinsic_counterfactual_v13.py \
  tests/test_intrinsic_pre_mlp_v14.py tests/test_intrinsic_budget_v12.py \
  tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py
```

Workflow commands are the same commands recorded in `reports/intrinsic_memory_review_v12/final_launch_gate/commands.md`, with output rooted at this directory. Migration, final-product, bundle and tracked-artifact commands were run from the clean checkout. Native protocol and environment-only reviews were also rerun without loading a model.

## Workspace disposition

No files are staged at summary time. The protected pilot was not read, imported, edited or staged. The two pre-existing modified files `reports/intrinsic_memory_takeover_20261029/repair_register.md` and `reports/vision_overfit_20261025/helper_tests.xml` remain untouched. Pre-existing untracked runtime/store artifacts remain untouched. Intentional new untracked evidence comprises duplicated `hf_modules` directories and the 16 external counterfactual tensors; these are explicitly excluded from Git and bound by manifests. No unintended new project files are known.

## Capability boundary

These checks establish engineering reproducibility only. V12 attempt10 and pre-MLP attempt01 both failed final fitting; useful recall and development generalization remain unestablished. Native density, formation/Replica connection, persistent learning, personal fidelity and RSI remain evidence-gated. No native worker is active at summary time.
