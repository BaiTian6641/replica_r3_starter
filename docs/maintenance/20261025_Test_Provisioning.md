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

## MiniCPM tokenizer/config integration metadata

Unit orchestration tests use `tests/tokenizer_fixtures.py`; it is deterministic,
reversible and explicitly not the sealed tokenizer. Only genuine sealed-tokenizer
integration tests use the following metadata. CPU tests never require the full
weight tensor or a download.

The sealed revision is `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
`config.json` declares `LlamaForCausalLM`, `model_type=llama`, 42 layers,
hidden width 2048, vocabulary 130560 and EOS IDs 1 and 130073. The tokenizer is
fast, does not add BOS/EOS automatically, uses `</s>` for ID 1 and `<|im_end|>`
for ID 130073, and uses the separate `chat_template.jinja` for chat rendering.

| File | SHA-256 |
|---|---|
| `config.json` | `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180` |
| `generation_config.json` | `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1` |
| `model.safetensors.index.json` | `6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756` |
| `special_tokens_map.json` | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `tokenizer.json` | `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| `tokenizer_config.json` | `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` |
| `chat_template.jinja` | `bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd` |

Provision only these files from an existing verified sealed source or metadata
directory:

```bash
.venv/bin/python scripts/provision_minicpm_metadata.py \
  --source /path/to/verified/minicpm5_source \
  --output checkpoints/minicpm5_source \
  --receipt reports/reproducibility_20261028/metadata_provisioning_attempt01.json
```

The script verifies every hash, refuses mismatched output, copies no weights and
performs no downloads or model loads. Native model loading still requires the
separately sealed full snapshot and `configs/minicpm5_sealed_lock.local.json`.

The original tracked-only run at `f85b3c5e` failed eight tests because this
ignored directory was absent: one DC-R1 tiny-model test, two DC-P1 production-
factory tests and five scoring-CLI tests. Those are unit contracts and now use
the self-contained fixture. The separate
`test_sealed_tokenizer_metadata_integration_without_weights` test remains an
explicit `integration` test and skips with these provisioning instructions when
the metadata is absent.

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
