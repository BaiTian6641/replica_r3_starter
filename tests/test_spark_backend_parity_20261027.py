"""Fail-closed tests for bounded Spark backend/parity preparation.

These tests are offline: no tensor library, download, model load or native
command is used.  They exercise the analytical checker and the
storage-dtype/small-reload claim refusals.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from conftest import require_local_artifacts


SPARK_ARTIFACTS = "checkpoints/minicpm5_source"

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/spark_backend_parity_20261027.json"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load("spark_backend_parity_checker_20261027",
                "scripts/check_spark_backend_parity_20261027.py")


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write(value: dict, path: Path) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _mutated(tmp_path: Path, mutate) -> Path:
    value = copy.deepcopy(_config())
    mutate(value)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _write(value, tmp_path / "spark_config.json")


@require_local_artifacts(SPARK_ARTIFACTS)
def test_preparation_passes_and_never_qualifies_training():
    report = CHECKER.check(CONFIG)
    assert report["training_qualified"] is False
    assert report["native_fp8_training_qualified"] is False
    assert report["native_fp8_execution_evidenced"] is False
    assert report["native_execution_performed"] is False
    assert report["model_loaded"] is False
    assert report["blocked_reasons"]
    assert {check["id"] for check in report["checks"]} >= {
        "baseline", "authorization", "identities", "parity", "claims",
        "resources", "ablation", "checker.offline",
    }


def test_four_precision_identities_are_distinct():
    identities = _config()["precision_identities"]
    assert set(identities) == {"official_bf16", "official_fp8",
                              "dequantized_fp8_derivative",
                              "native_fp8_execution"}
    bf16, fp8 = identities["official_bf16"], identities["official_fp8"]
    derivative, native = (identities["dequantized_fp8_derivative"],
                          identities["native_fp8_execution"])
    assert bf16["storage_dtype"] == "bfloat16" and bf16["is_official_release"]
    assert fp8["storage_dtype"] == "float8_e4m3fn" and fp8["is_official_release"]
    assert not derivative["is_official_release"]
    assert derivative["parent"] == "official_fp8"
    assert derivative["equivalent_to_bf16_release"] is False
    assert native["is_artifact"] is False and native["storage_dtype"] is None
    assert native["required_kernels"]
    assert len({item["identity"] for item in identities.values()}) == 4


def test_resource_estimates_are_analytical_and_block_full_fine_tuning():
    estimates = CHECKER.estimate_resources(_config())
    assert estimates["measured"] is False and estimates["analytical"] is True
    assert estimates["activations"] == 1698693120
    assert estimates["kv_cache"] == 75497472
    assert estimates["usable_budget"] == 42949672960
    assert estimates["status"]["extension_training_bf16_resident"] \
        == "within_declared_headroom"
    assert estimates["status"]["full_base_tuning_bf16_resident"] \
        == "blocked_by_declared_limit"
    assert estimates["status"]["extension_training_native_fp8_resident"] \
        .startswith("unquantified")
    assert estimates["extension_training_native_fp8_resident"] is None
    assert estimates["full_base_tuning_bf16_resident"] \
        > estimates["usable_budget"]


def _complete_evidence(config: dict) -> dict:
    return {gate["id"]: {
        "evidence_kind": "measured_receipt",
        "measured": True,
        "artifact_sha256": "a" * 64,
        "receipt_path": f"reports/{gate['id']}.json",
    } for gate in config["claims_policy"]["training_qualification_gates"]}


def test_training_claim_requires_all_measured_hashed_gates():
    config = _config()
    assert CHECKER.evaluate_training_claim(None, config)["qualified"] is False
    assert CHECKER.evaluate_training_claim({}, config)["qualified"] is False
    partial = _complete_evidence(config)
    partial.pop("single_update")
    assert CHECKER.evaluate_training_claim(partial, config)["qualified"] is False
    unhashed = _complete_evidence(config)
    unhashed["forward_parity"]["artifact_sha256"] = "not-a-digest"
    assert CHECKER.evaluate_training_claim(unhashed, config)["qualified"] is False
    assert CHECKER.evaluate_training_claim(
        _complete_evidence(config), config)["qualified"] is True


@pytest.mark.parametrize("kind", ["storage_dtype", "small_reload", "config_parse"])
def test_storage_dtype_and_small_reload_cannot_qualify(kind):
    config = _config()
    evidence = {gate["id"]: {
        "evidence_kind": kind, "measured": True,
        "artifact_sha256": "b" * 64, "identity_sufficient": True,
    } for gate in config["claims_policy"]["training_qualification_gates"]}
    result = CHECKER.evaluate_training_claim(evidence, config)
    assert result["qualified"] is False
    assert result["reasons"]


@require_local_artifacts(SPARK_ARTIFACTS)
def test_config_claiming_qualification_fails_closed(tmp_path):
    path = _mutated(tmp_path, lambda value: value["claims_policy"].update(
        training_qualified=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "claims.qualification"


@require_local_artifacts(SPARK_ARTIFACTS)
def test_derivative_cannot_be_relabelled_official(tmp_path):
    path = _mutated(tmp_path, lambda value: value["precision_identities"][
        "dequantized_fp8_derivative"].update(is_official_release=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "identities.derivative"


@require_local_artifacts(SPARK_ARTIFACTS)
def test_storage_dtype_inference_cannot_be_enabled(tmp_path):
    path = _mutated(tmp_path, lambda value: value["claims_policy"].update(
        storage_dtype_implies_training_qualification=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "claims.storage"


@require_local_artifacts(SPARK_ARTIFACTS)
def test_guard_values_and_native_admission_cannot_be_loosened(tmp_path):
    path = _mutated(tmp_path, lambda value: value["authorization"]["guards"].update(
        driver_limit_bytes=999999999999))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "authorization.guards"

    path = _mutated(tmp_path / "native", lambda value: value["authorization"].update(
        native_execution_admitted=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "authorization.native"


def test_baseline_pin_change_is_rejected(tmp_path):
    path = _mutated(tmp_path, lambda value: value["baseline"].update(
        revision="0" * 40))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "baseline.pin"


@require_local_artifacts(SPARK_ARTIFACTS)
def test_oversized_extension_estimate_is_blocked(tmp_path):
    path = _mutated(tmp_path, lambda value: value["resource_model"]["inputs"].update(
        new_module_parameters=4_112_079_360))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "resources.extension_budget"


@require_local_artifacts(SPARK_ARTIFACTS)
def test_ablation_cannot_use_unqualified_visual_answers(tmp_path):
    path = _mutated(tmp_path, lambda value: value["later_ablation"][
        "ground_truth_policy"].update(
            unqualified_visual_answers_as_ground_truth=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check(path)
    assert error.value.check == "ablation.ground_truth"


def test_ablation_is_documented_but_unauthorized():
    config = _config()
    ablation = config["later_ablation"]
    assert ablation["status"] == "proposed_not_authorized"
    assert ablation["authorized_by_this_config"] is False
    assert ablation["ground_truth_policy"][
        "unqualified_visual_answers_as_ground_truth"] is False
    assert any("text-only" in item for item in ablation["filters"])
    assert any("unfiltered" in item for item in ablation["controls"])
    assert config["baseline"]["default_optimizer"] == "NativeFusedAdamW"
    assert config["baseline"]["training_default_unchanged"] is True


def test_checker_is_lineage_offline_and_imports_no_tensor_library():
    source = (ROOT / "scripts/check_spark_backend_parity_20261027.py").read_text(
        encoding="utf-8")
    for name in ("torch", "transformers", "safetensors", "mlx", "subprocess"):
        assert f"import {name}" not in source
    assert not any(hasattr(CHECKER, name)
                   for name in ("torch", "transformers", "safetensors"))
