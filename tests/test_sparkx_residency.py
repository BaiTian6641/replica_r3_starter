"""Deterministic arithmetic tests for the SparkX residency ledger.

These tests are self-contained: they import the calculator, recompute the
ledger from committed indices and tamper with in-memory copies. They do not
need ignored local artifacts, network access, a model or a native job.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CALCULATOR = ROOT / "reports" / "sparkx_residency_20261026" / "calculate_residency.py"
RECEIPT_V1 = ROOT / "reports" / "sparkx_residency_20261026" / "residency.json"
RECEIPT_V2 = ROOT / "reports" / "sparkx_residency_20261026" / "residency_v2.json"

_spec = importlib.util.spec_from_file_location("calculate_residency", CALCULATOR)
assert _spec is not None and _spec.loader is not None
calculator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calculator)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return calculator.calculate(ROOT)


def _plan(receipt: dict, plan_id: str) -> dict:
    return next(plan for plan in receipt["training_storage_plans"] if plan["id"] == plan_id)


def test_five_copy_rate_is_18_not_16(receipt: dict) -> None:
    plan = _plan(
        receipt,
        "bf16_weights_fp32_master_fp32_gradients_two_fp32_moments",
    )
    assert plan["bytes_per_parameter"] == 18
    assert plan["formula"] == "2 + 4 + 4 + 4 + 4 bytes/parameter"
    assert plan["total_bytes"] == plan["parameters"] * 18
    assert receipt["corrected_critical_rate"]["bytes_per_parameter"] == 18
    statement = receipt["corrected_critical_rate"]["statement"]
    assert "18 bytes/parameter" in statement
    assert "must not be used for the five-copy plan" in statement


def test_all_fp32_rate_is_the_separate_16_byte_plan(receipt: dict) -> None:
    plan = _plan(receipt, "all_fp32")
    assert plan["bytes_per_parameter"] == 16
    assert plan["formula"] == "4 + 4 + 4 + 4 bytes/parameter"
    assert plan["id"] != receipt["corrected_critical_rate"]["plan"]


def test_every_plan_sums_its_declared_copies(receipt: dict) -> None:
    for plan in receipt["training_storage_plans"]:
        assert sum(copy["bytes"] for copy in plan["copies"]) == plan["bytes_per_parameter"]
        assert plan["total_bytes"] == plan["parameters"] * plan["bytes_per_parameter"]
        assert plan["total_GiB"] == round(plan["total_bytes"] / 2**30, 6)
    rates = {plan["bytes_per_parameter"] for plan in receipt["training_storage_plans"]}
    assert rates == {12, 14, 16, 18}


def test_fp8_source_is_mixed_dtype_and_not_one_byte_per_parameter(receipt: dict) -> None:
    audit = receipt["fp8_header_audit"]
    assert audit["tensor_count"] == 470
    assert audit["dtype_tensor_counts"] == {"BF16": 290, "F8_E4M3": 180}
    assert audit["f8_matrix_count"] == 180
    assert audit["scale_tensor_count"] == 180
    assert audit["dtype_payload_bytes"] == {
        "BF16": 676_697_088,
        "F8_E4M3": 3_774_873_600,
    }
    assert audit["serialized_payload_bytes"] == 4_451_570_688
    assert audit["scale_payload_bytes"] == 2_285_568
    categories = audit["bf16_category_payload_bytes"]
    assert categories == {
        "tied_embedding": 671_088_640,
        "layer_and_final_norms": 373_760,
        "attention_gates": 2_949_120,
        "scale_tensors": 2_285_568,
    }
    assert sum(categories.values()) == audit["dtype_payload_bytes"]["BF16"]
    assert receipt["source_storage"]["fp8_payload_is_not_one_byte_per_parameter"] is True
    assert receipt["source_storage"]["fp8_training_capability"] == "unqualified"
    assert "parameters * 1 byte" not in receipt["source_storage"]["fp8_source"]["note"]


def test_fp8_is_not_a_training_plan(receipt: dict) -> None:
    plan_ids = {plan["id"] for plan in receipt["training_storage_plans"]}
    assert not any("fp8" in plan_id for plan_id in plan_ids)
    assert receipt["gates"]["dequantization"] == "unqualified"
    assert receipt["gates"]["backend"] == "unqualified"


def test_parameter_inventory_sums_components(receipt: dict) -> None:
    inventory = receipt["parameter_inventory"]
    assert sum(row["parameters"] for row in inventory["components"]) == inventory["total_parameters"]
    assert inventory["backbone_parameters"] == 4_112_079_360
    assert inventory["declared_encoder_parameters"] == 186_173_700
    assert inventory["total_parameters"] == (
        inventory["backbone_parameters"]
        + inventory["declared_encoder_parameters"]
        + 3_411_456  # vision projector
        + 3_149_312  # audio projector
        + 3_276_800  # query/output
        + 10_485_760  # reader matrices
        + 2_228_224  # phrase encoder
    )


def test_projector_arithmetic_is_explicit(receipt: dict) -> None:
    rows = {row["id"]: row for row in receipt["parameter_inventory"]["components"]}
    assert rows["vision_projector"]["parameters"] == 768 * 1024 + 1024 + 1024 * 2560 + 2560
    assert rows["audio_projector"]["parameters"] == 512 * 1024 + 1024 + 1024 * 2560 + 2560
    assert rows["memory_reader_matrices"]["parameters"] == 2 * 4 * 2560 * 512


def test_projected_training_sums_every_bucket(receipt: dict) -> None:
    projected = receipt["projected_full_scope_training"]
    plan = _plan(
        receipt,
        "bf16_weights_fp32_master_fp32_gradients_two_fp32_moments",
    )
    assert projected["parameter_state_bytes"] == plan["total_bytes"]
    assert projected["dynamic_state_bytes"] == receipt["dynamic_memory_state"]["update_peak_bytes"]
    assert projected["activation_bytes"] == receipt["activations_and_logits"]["activation_bytes"]
    assert projected["logits_bytes"] == receipt["activations_and_logits"]["logits"]["bytes"]
    assert projected["temporary_buffer_bytes"] == receipt["temporary_buffers"]["total_bytes"]
    assert projected["resident_training_bytes"] == (
        projected["parameter_state_bytes"]
        + projected["dynamic_state_bytes"]
        + projected["activation_bytes"]
        + projected["logits_bytes"]
        + projected["temporary_buffer_bytes"]
    )
    assert projected["resident_training_GiB"] >= 50
    assert projected["over_driver_guard_on_resident_training"] is True
    assert projected["status"] == "analytical_not_admitted"


def test_activation_and_logit_budgets_are_positive_and_formulaic(receipt: dict) -> None:
    activations = receipt["activations_and_logits"]
    assert activations["activation_bytes"] == sum(
        item["bytes"] for item in activations["items"]
    )
    assert activations["logits"]["bytes"] == 3 * 256 * 131072 * 4
    temporary = receipt["temporary_buffers"]
    assert temporary["total_bytes"] == sum(item["bytes"] for item in temporary["items"])


def test_restore_and_export_peaks_are_explicit(receipt: dict) -> None:
    peaks = receipt["phase_peaks"]
    parameters = receipt["parameter_inventory"]["total_parameters"]
    assert peaks["restore_from_bf16_source"]["bytes"] == (
        8_224_158_720 + parameters * 2 + parameters * 4
    )
    assert peaks["restore_from_fp8_source"]["bytes"] == (
        4_451_570_688 + parameters * 2 + parameters * 4
    )
    assert peaks["export_bf16_and_fp8"]["bytes"] == parameters * (4 + 2 + 1)
    assert peaks["restore_from_fp8_source"]["status"].endswith("unqualified")


def test_receipt_is_deterministic_and_preserves_v1(receipt: dict) -> None:
    assert calculator.calculate(ROOT) == receipt
    assert json.loads(RECEIPT_V2.read_text(encoding="utf-8")) == receipt
    historical = json.loads(RECEIPT_V1.read_text(encoding="utf-8"))
    assert historical["format"] == "replica.sparkx_residency.v1"
    assert receipt["format"] == "replica.sparkx_residency.v2"
    assert receipt["supersession"]["supersedes"].endswith("residency.json")
    assert receipt["supersession"]["historical_receipt_untouched"] is True
    assert receipt["supersession"]["superseded_sha256"] == calculator.sha256_file(RECEIPT_V1)


def test_bf16_audit_rejects_unknown_dtype(receipt: dict) -> None:
    header = calculator.load_json(ROOT, calculator.BF16_HEADER_PATH)
    tampered = copy.deepcopy(header)
    tampered["tensors"][0]["dtype"] = "F8_E4M3"
    with pytest.raises(calculator.ResidencyError):
        calculator.audit_bf16_header(tampered)


def test_fp8_audit_rejects_missing_scale(receipt: dict) -> None:
    header = calculator.load_json(ROOT, calculator.FP8_HEADER_PATH)
    tampered = copy.deepcopy(header)
    tampered["tensors"] = [
        tensor
        for tensor in tampered["tensors"]
        if not tensor["name"].endswith(".mlp.down_proj.weight_scale")
    ]
    with pytest.raises(calculator.ResidencyError):
        calculator.audit_fp8_header(tampered)


def test_fp8_audit_rejects_bf16_scale_dtype(receipt: dict) -> None:
    header = calculator.load_json(ROOT, calculator.FP8_HEADER_PATH)
    tampered = copy.deepcopy(header)
    for tensor in tampered["tensors"]:
        if tensor["name"].endswith("weight_scale"):
            tensor["dtype"] = "F32"
            break
    with pytest.raises(calculator.ResidencyError):
        calculator.audit_fp8_header(tampered)


def test_cli_check_mode_reproduces_receipt() -> None:
    result = subprocess.run(
        [sys.executable, str(CALCULATOR), "--check"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Receipt verified" in result.stdout
