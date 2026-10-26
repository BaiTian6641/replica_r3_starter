# Test provisioning: self-contained units vs artifact-dependent integration

25 October 2026. Two test classes exist and they are kept explicitly apart.

## Self-contained unit tests (default)

Every ordinary test builds synthetic fixtures (tiny adapters, synthetic
safetensors headers, owned procedural data committed under `data/`, or an
in-memory external-caption source) and runs from any clean tracked export with
the baseline `.venv` alone. Native tokenizer/model assets, ignored external
images, caches and historical native receipts belong only to the integration
class below.

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
| `data/external_caption_tranche_20261025_v2/raw/images` | `.venv/bin/python scripts/prepare_external_tranche.py --config configs/external_caption_tranche_v1.json`, or copy the verified ignored `raw/images` directory from the provisioned workspace | `test_vision_curriculum.py::test_external_final_seal_is_derived_from_manifest_rows`, `test_vision_curriculum.py::test_external_final_preflight_resolves_external_root` |
| `checkpoints/minicpm5_source` | sealed local MiniCPM5-2B acquisition (`scripts/pin_model.py`) | `test_capacity*_data.py`, `test_e4_contextual_data.py::test_native_tokenizer_observation_budget` |

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

- With all inputs provisioned: native-tokenizer and external-image integration
  tests execute; without them they skip with exact provisioning instructions.
- From a clean tracked export without ignored inputs: ordinary tests execute
  and only declared integration tests skip. A skip is unavailable integration
  coverage, never a passing native-asset result.
- Historical focused clean-export evidence is retained below; rerun the full
  suite after changes rather than presenting these older counts as current.
- A [focused clean-export replay](../../reports/clean_export_verification_20261025.json)
  at `4112e18` passed **69 tests with one explicitly reasoned integration skip**;
  bounded RSI alone passed 11 with that skip and 12/12 when provisioned.
