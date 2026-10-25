# Test provisioning: self-contained units vs artifact-dependent integration

25 October 2026. Two test classes exist and they are kept explicitly apart.

## Self-contained unit tests (default)

Every ordinary test builds synthetic fixtures (tiny adapters, synthetic
safetensors headers, owned procedural data committed under `data/`) and runs
from any clean tracked export of the repository with the baseline
`.venv` alone. No test in this class reads `checkpoints/`, `runs/`,
`data/cache/` or `.venv-vision`.

## Artifact-dependent integration tests

Some contracts can only be exercised against ignored local acquisitions.
They are marked `@pytest.mark.integration` and declare their inputs through
`require_local_artifacts(...)` in `tests/conftest.py`. When an input is
absent the test skips with a reason that names the exact provisioning
command — never silently, and provenance checks are never weakened to make
a test pass.

Current declarations and their provisioning:

| Requirement | Provisioning | Tests |
|---|---|---|
| `checkpoints/minicpm5_source` + `configs/minicpm5_sealed_lock.local.json` | sealed local MiniCPM5-2B acquisition (`scripts/pin_model.py`) and snapshot lock | `test_memory_diagnosis.py::test_prepare_and_admit_bind_exact_source_data_and_protocol_pins`, `test_capacity_fixtures.py::test_pinned_native_tokenizer_reaches_exact_capacity_lengths` |
| `PIL` (Pillow 12.0.0 via `.venv-vision` overlay) | `python -m venv .venv-vision && .venv-vision/bin/pip install -r requirements-vision-overlay.txt` | `test_vision_features.py::test_processor_backend_is_official_pil_when_overlay_present` |
| `data/cache/spark_sources/run_02` | `scripts/prepare_spark_sources.py --plan configs/spark_source_preparation_v2.json --output data/cache/spark_sources/run_02 --export-header-bundle` | offline Spark reproduction runs (not part of the unit suite) |
| `runs/minicpm5_vision_v0_20261025/score_restart_replay01/score_report.json` | copy the immutable V0 score replay from the provisioned native-evidence workspace | `test_bounded_rsi.py::test_historical_native_evidence_remains_explicitly_provisioned` |

## Historical verdict replay (outside the unit suite)

V0, V1 and memory-diagnostic verdicts are reproduced by checking out each
pinned implementation commit into a `git worktree`, cloning the ignored
hash-pinned inputs (sealed model, run artifacts, local lock, rejected writer
extension) into it, and running the recorded scorer there. Live code-pin
checks intentionally reject later working trees; the recorded method and the
per-verdict differences are in
[reproducibility replay](../../reports/reproducibility_replay_20261025.json).
Historical protocol hashes are never rewritten.

## Verification

- With all inputs provisioned: the full suite passes (1,584 tests +21
  subtests at the rebind fix) with zero skips beyond explicitly reasoned ones.
- From a clean tracked export without ignored inputs: only the declared
  integration tests skip, each printing its provisioning command.
