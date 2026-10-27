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
        "baseline", "authorization", "identities", "parity", "tolerances",
        "evidence", "claims", "resources", "resource_scope", "storage_policy",
        "ablation", "checker.offline",
    }
    assert report["parity_established"] is False
    assert report["complete_evolution_architecture_estimated"] is False
    assert report["existing_evidence"]["same_weight_reload_max_abs"] == 0.0
    assert report["existing_evidence"]["used_as_tolerance"] is False


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
    assert estimates["estimate_scope"] == {
        "extension_training_bf16_resident": "small_extension_adaptation_only",
        "extension_training_dequantized_derivative_resident":
            "small_extension_adaptation_only",
        "full_base_tuning_bf16_resident":
            "complete_evolution_architecture_lower_bound_only",
        "complete_evolution_architecture":
            "not_estimated_blocked_by_resource_envelope",
    }
    assert estimates["measured"] is False and estimates["analytical"] is True


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


# ---------------------------------------------------------------------------
# Explicit parity tolerances, evidence reconciliation and resource scope


def test_tolerance_schema_keeps_exact_zero_and_candidates_separate():
    config = _config()
    tolerances = config["parity_protocol"]["numerical_tolerances"]
    assert "quality_tolerances" not in config["parity_protocol"]
    assert tolerances["schema"] == "replica.spark_parity_tolerances.v1"
    assert tolerances["status"] == "declared_for_future_checks_no_current_parity"
    assert tolerances["current_parity_established"] is False
    assert tolerances["current_native_fp8_execution"] is False

    same = tolerances["same_weight_reload_and_base_identity"]
    assert same["rtol"] == 0.0 and same["atol"] == 0.0
    assert same["require_exact_logits"] is True
    assert same["require_exact_greedy_tokens"] is True

    candidate = tolerances["candidate_native_fp8_vs_derivative"]
    assert candidate["status"] == "candidate_not_evidence"
    assert candidate["rtol"] == 0.05 and candidate["atol"] == 0.5
    assert candidate["rederive_before_use"] is True
    assert candidate["require_exact_greedy_tokens"] is False
    assert candidate["must_not_be_derived_from"] == \
        "existing_evidence.cross_arm_logit_difference"

    historical = tolerances["historical_cross_arm_difference"]
    assert historical["kind"] == "historical_evidence_not_a_tolerance"
    assert historical["used_as_tolerance"] is False
    assert historical["used_for_parity"] is False
    evidence = config["existing_evidence"]["cross_arm_logit_difference"]
    assert historical["reference"] == "existing_evidence.cross_arm_logit_difference"
    assert evidence["max_abs_difference"] == 10.875
    assert evidence["exact_greedy_tokens"] is False
    assert candidate["atol"] != evidence["max_abs_difference"]
    assert CHECKER.check_tolerances(config) is tolerances


def test_tolerance_mutations_fail_closed(tmp_path):
    mutations = [
        ("same_weight", lambda value: value["same_weight_reload_and_base_identity"]
         .update(rtol=1e-5)),
        ("same_weight", lambda value: value["same_weight_reload_and_base_identity"]
         .update(require_exact_logits=False)),
        ("candidate", lambda value: value["candidate_native_fp8_vs_derivative"]
         .update(status="established")),
        ("candidate", lambda value: value["candidate_native_fp8_vs_derivative"]
         .update(rederive_before_use=False)),
        ("candidate", lambda value: value["candidate_native_fp8_vs_derivative"]
         .update(atol=0)),
        ("historical", lambda value: value["historical_cross_arm_difference"]
         .update(used_as_tolerance=True)),
        ("historical", lambda value: value["historical_cross_arm_difference"]
         .update(used_for_parity=True)),
        ("separation", lambda value: value["candidate_native_fp8_vs_derivative"]
         .update(atol=10.875)),
        ("claims", lambda value: value.update(
            candidate_tolerance_implies_parity=True)),
        ("claims", lambda value: value.update(
            exact_zero_tolerance_implies_native_fp8=True)),
    ]
    for index, (section, mutate) in enumerate(mutations):
        path = _mutated(tmp_path / f"mutation{index}",
                        lambda value, mutate=mutate: mutate(
                            value["parity_protocol"]["numerical_tolerances"]
                            if section != "claims" else value["claims_policy"]))
        with pytest.raises(CHECKER.CheckError) as error:
            CHECKER.check_tolerances(
                json.loads(path.read_text(encoding="utf-8")))
        assert error.value.check.startswith("tolerances."), section


def test_existing_reload_evidence_is_reconciled_not_extended(tmp_path):
    config = _config()
    evidence = CHECKER.check_existing_evidence(config, project_root=ROOT)
    assert evidence["status"] == "reconciled"
    assert evidence["same_weight_reload_max_abs"] == 0.0
    assert evidence["cross_arm_max_abs_difference"] == 10.875
    assert evidence["cross_arm_mean_abs_difference"] == 0.9691294431686401
    assert evidence["cross_arm_exact_greedy_tokens"] is False
    assert evidence["used_as_tolerance"] is False

    sources = config["existing_evidence"]["sources"]
    assert set(sources) == {
        "qualification_summary", "arm_comparison", "selected_runs_receipt",
        "qualification_config"}
    for pin in sources.values():
        assert (ROOT / pin["path"]).is_file()

    path = _mutated(tmp_path, lambda value: value["existing_evidence"][
        "sources"]["qualification_summary"].update(sha256="0" * 64))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check_existing_evidence(
            json.loads(path.read_text(encoding="utf-8")), project_root=ROOT)
    assert error.value.check == "evidence.pin"


def test_historical_cross_arm_value_cannot_be_promoted_to_tolerance(tmp_path):
    path = _mutated(tmp_path, lambda value: value["existing_evidence"][
        "cross_arm_logit_difference"].update(used_as_tolerance=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check_existing_evidence(
            json.loads(path.read_text(encoding="utf-8")), project_root=ROOT)
    assert error.value.check == "evidence.cross_arm"


def test_resource_scope_separates_extension_from_complete_architecture():
    config = _config()
    scope = CHECKER.check_resource_scope(config)
    assert scope["estimates_cover"] == "small_extension_adaptation_only"
    small = scope["small_extension_adaptation"]
    complete = scope["complete_evolution_architecture"]
    assert small["is_complete_evolution_architecture"] is False
    assert small["new_parameter_elements"] == 5242880
    assert complete["is_complete_evolution_architecture"] is True
    assert complete["estimated"] is False
    assert complete["status"] == "blocked_by_resource_envelope"
    assert complete["full_base_tuning_is_only_a_lower_bound"] is True
    assert complete["small_extension_estimate_reusable"] is False
    assert complete["training_admitted"] is False
    assert config["non_claims"]["complete_evolution_architecture_estimated"] \
        is False
    assert config["non_claims"][
        "small_extension_estimate_is_complete_architecture"] is False


def test_extension_estimate_cannot_be_relabelled_complete_architecture(tmp_path):
    path = _mutated(tmp_path, lambda value: value["resource_model"]["scope"][
        "small_extension_adaptation"].update(
            is_complete_evolution_architecture=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check_resource_scope(json.loads(path.read_text(encoding="utf-8")))
    assert error.value.check == "resources.scope.extension"

    path = _mutated(tmp_path / "complete", lambda value: value["resource_model"][
        "scope"]["complete_evolution_architecture"].update(estimated=True))
    with pytest.raises(CHECKER.CheckError) as error:
        CHECKER.check_resource_scope(json.loads(path.read_text(encoding="utf-8")))
    assert error.value.check == "resources.scope.complete"


def test_storage_policy_blocks_duplicates_downloads_and_checkpoint_fanout(tmp_path):
    config = _config()
    policy = CHECKER.check_storage_policy(config)
    assert policy["downloads_in_preparation"] is False
    assert policy["duplicate_weight_copies"] is False
    assert policy["derivative_count_max"] == 1
    assert policy["full_base_checkpoint_saves"] == 0
    assert policy["training_checkpoints_max"] == 0
    assert policy["extension_exports_max"] == 1
    assert policy["spark_specific_budget_bytes"] == 42949672960

    cases = [
        ("storage.downloads", lambda value: value.update(
            downloads_in_preparation=True)),
        ("storage.weights", lambda value: value.update(
            derivative_count_max=2)),
        ("storage.weights", lambda value: value.update(
            duplicate_weight_copies=True)),
        ("storage.checkpoints", lambda value: value.update(
            full_base_checkpoint_saves=1)),
    ]
    for index, (check, mutate) in enumerate(cases):
        path = _mutated(tmp_path / f"storage{index}",
                        lambda value, mutate=mutate: mutate(
                            value["storage_and_duplication_policy"]))
        with pytest.raises(CHECKER.CheckError) as error:
            CHECKER.check_storage_policy(
                json.loads(path.read_text(encoding="utf-8")))
        assert error.value.check == check


@require_local_artifacts(SPARK_ARTIFACTS)
def test_report_never_claims_parity_execution_or_complete_residency():
    report = CHECKER.check(CONFIG)
    assert report["parity_established"] is False
    assert report["native_fp8_execution_evidenced"] is False
    assert report["native_execution_performed"] is False
    assert report["training_qualified"] is False
    assert report["complete_evolution_architecture_estimated"] is False
    assert report["tolerances"]["same_weight_reload_and_base_identity"][
        "rtol"] == 0.0
    assert report["tolerances"]["candidate_native_fp8_vs_derivative"][
        "status"] == "candidate_not_evidence"
    assert report["existing_evidence"]["cross_arm_exact_greedy_tokens"] is False
    assert any("complete evolution architecture" in reason
               for reason in report["blocked_reasons"])
