#!/usr/bin/env python3
"""Fail-closed offline check for the bounded Spark backend/parity preparation.

This tool never imports a tensor library, never downloads, never loads a model
and never executes a native command.  It verifies that the preparation config:

* keeps official BF16, official FP8 storage, dequantized derivatives and actual
  native FP8 execution as separate identities;
* refuses the inference "storage dtype" or "small reload success" -> training
  qualification;
* records analytical resource-estimate inputs and blocks estimates above the
  driver budget;
* keeps the MiniCPM5 baseline and its default optimizer unchanged;
* documents a later filtered self-replay text-retention ablation without using
  unqualified visual answers as ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import canonical_json  # noqa: E402

SCHEMA = "replica.spark_backend_parity_preparation.v1"
CHECK_SCHEMA = "replica.spark_backend_parity_check.v1"
CHECKER_VERSION = "1.0.0"
DRIVER_LIMIT = 53687091200
SWAP_LIMIT = 536870912
MINIMUM_FREE_PERCENT = 15
FORBIDDEN_IMPORT_NAMES = ("torch", "transformers", "safetensors", "mlx",
                          "sentencepiece", "subprocess")


class CheckError(ValueError):
    """A preparation invariant was violated."""

    def __init__(self, check: str, message: str):
        super().__init__(f"{check}: {message}")
        self.check = check
        self.message = message


def _require(check: str, condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(check, message)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require("load", isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _display_path(path: str | Path, project_root: Path) -> Path | str:
    path = Path(path)
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path


def estimate_resources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Analytical bytes only; never a measured memory reading."""
    model = config["resource_model"]
    _require("resources.mode", model["analytical_only"] is True
             and model["measured"] is False,
             "resource model must be analytical and unmeasured")
    inputs = model["inputs"]
    activations = (inputs["sequence_tokens"] * inputs["hidden_size"]
                   * inputs["activation_dtype_bytes"]
                   * inputs["activation_saved_tensor_multiplier"]
                   * inputs["layers"])
    kv_cache = (2 * inputs["kv_heads"] * inputs["head_dim"]
                * inputs["kv_bytes_per_element"] * inputs["layers"]
                * inputs["sequence_tokens"])
    extension_state = (inputs["new_module_parameters"]
                       * sum(inputs["new_module_parameter_components_bytes"].values()))
    overhead = 1 + inputs["overhead_fraction"]
    weights_bf16 = inputs["official_bf16_storage_bytes"]
    weights_fp8 = inputs["official_fp8_storage_bytes"] + inputs["fp8_scale_bytes"]
    base_state = inputs["parameter_elements"] * inputs["base_full_tuning_state_bytes_per_parameter"]
    usable = inputs["driver_limit_bytes"] * inputs["usable_headroom_fraction"]
    extension_bf16 = (weights_bf16 + extension_state + activations + kv_cache) * overhead
    full_bf16 = (weights_bf16 + base_state + extension_state
                 + activations + kv_cache) * overhead
    _require("resources.formula", all(value > 0 for value in (
        activations, kv_cache, extension_state, weights_bf16, weights_fp8,
        base_state, usable, extension_bf16, full_bf16)),
        "resource formula produced a non-positive value")
    _require("resources.activations", activations == 1698693120,
             "activation estimate changed unexpectedly")
    _require("resources.kv", kv_cache == 75497472,
             "kv-cache estimate changed unexpectedly")
    _require("resources.limit", inputs["driver_limit_bytes"] == DRIVER_LIMIT,
             "driver limit differs from the project guard")

    def classify(value: float) -> str:
        return "within_declared_headroom" if value <= usable else "blocked_by_declared_limit"

    estimates = {
        "weights_resident_bf16": weights_bf16,
        "weights_storage_fp8": weights_fp8,
        "extension_state": extension_state,
        "activations": activations,
        "kv_cache": kv_cache,
        "usable_budget": usable,
        "extension_training_bf16_resident": extension_bf16,
        "extension_training_dequantized_derivative_resident": extension_bf16,
        "extension_training_official_fp8_resident": None,
        "extension_training_native_fp8_resident": None,
        "full_base_tuning_bf16_resident": full_bf16,
        "status": {
            "extension_training_bf16_resident": classify(extension_bf16),
            "extension_training_dequantized_derivative_resident": classify(extension_bf16),
            "extension_training_official_fp8_resident":
                "unquantified_activation_and_optimizer_semantics",
            "extension_training_native_fp8_resident":
                "unquantified_no_native_kernel_evidence",
            "full_base_tuning_bf16_resident": classify(full_bf16),
        },
        "measured": False,
        "analytical": True,
    }
    _require("resources.extension_budget",
             estimates["status"]["extension_training_bf16_resident"]
             == "within_declared_headroom",
             "extension-only analytical estimate exceeds the usable budget")
    _require("resources.full_tuning_blocked",
             estimates["status"]["full_base_tuning_bf16_resident"]
             == "blocked_by_declared_limit",
             "full-base analytical estimate should exceed the usable budget")
    return estimates


def evaluate_training_claim(evidence: Mapping[str, Any] | None,
                            config: Mapping[str, Any]) -> dict[str, Any]:
    """Return qualification only when every gate has measured, hashed evidence."""
    policy = config["claims_policy"]
    gates = policy["training_qualification_gates"]
    non_qualifying = set(policy["non_qualifying_evidence"])
    if not evidence:
        return {"qualified": False, "missing": [gate["id"] for gate in gates],
                "reasons": ["no evidence supplied"]}
    reasons: list[str] = []
    missing: list[str] = []
    for gate in gates:
        item = evidence.get(gate["id"])
        if not isinstance(item, dict):
            missing.append(gate["id"])
            continue
        kind = item.get("evidence_kind")
        if kind in non_qualifying:
            reasons.append(f"{gate['id']}: non-qualifying evidence {kind!r}")
        if item.get("measured") is not True:
            reasons.append(f"{gate['id']}: evidence is not measured")
        digest = item.get("artifact_sha256")
        if not (isinstance(digest, str) and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)):
            reasons.append(f"{gate['id']}: missing artifact sha256")
        if item.get("identity_sufficient") is True and kind in (
                "storage_dtype", "small_reload", "config_parse"):
            reasons.append(f"{gate['id']}: identity-only evidence cannot qualify")
    qualified = not missing and not reasons
    return {"qualified": qualified, "missing": missing, "reasons": reasons}


def check(config_path: str | Path, *, project_root: str | Path = ROOT) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_json(config_path)
    checks: list[dict[str, str]] = []

    def passed(check_id: str, detail: str) -> None:
        checks.append({"id": check_id, "status": "pass", "detail": detail})

    _require("config.format", config.get("schema") == SCHEMA, "unexpected schema")
    _require("config.status",
             config.get("status") == "preparation_only_not_executed",
             "unexpected preparation status")
    passed("config", "preparation-only schema and status")

    baseline = config["baseline"]
    pin_path = project_root / baseline["pin_reference"]["path"]
    _require("baseline.pin", sha256_file(pin_path)
             == baseline["pin_reference"]["sha256"], "baseline pin mismatch")
    pin = load_json(pin_path)
    pointer = baseline["pin_reference"]["pointer"].split(".")
    node: Any = pin
    for part in pointer:
        node = node[part]
    _require("baseline.pin", node["repo_id"] == baseline["repo_id"]
             and node["revision"] == baseline["revision"]
             and node["unique_parameters"] == baseline["unique_parameters"],
             "MiniCPM5 baseline pin differs from the consolidated plan")
    local_config = project_root / baseline["local_source"] / "config.json"
    _require("baseline.local",
             sha256_file(local_config) == baseline["config_sha256"],
             "local MiniCPM5 config hash mismatch")
    _require("baseline.isolation",
             baseline["training_default_unchanged"] is True
             and baseline["default_optimizer"] == "NativeFusedAdamW"
             and config["non_claims"]["baseline_changed"] is False,
             "MiniCPM5 baseline or default optimizer is not preserved")
    passed("baseline", "MiniCPM5 pins and NativeFusedAdamW default unchanged")

    authorization = config["authorization"]
    _require("authorization.native",
             authorization["native_execution_admitted"] is False
             and authorization["downloads_admitted"] is False
             and authorization["bulk_assets_admitted"] is False,
             "preparation must not admit native execution or downloads")
    receipt_path = project_root / authorization["authorization_receipt_path"]
    _require("authorization.receipt",
             not receipt_path.exists(),
             "an authorization receipt exists; this preparation is not authorized "
             "to consume it")
    guards = authorization["guards"]
    _require("authorization.guards",
             guards["driver_limit_bytes"] == DRIVER_LIMIT
             and guards["swap_growth_limit_bytes"] == SWAP_LIMIT
             and guards["minimum_free_percent"] == MINIMUM_FREE_PERCENT
             and guards["cpu_fallback"] is False
             and guards["native_jobs_at_once"] == 1,
             "project capacity guards changed")
    passed("authorization", "native execution refused; guards preserved")

    identities = config["precision_identities"]
    _require("identities.set",
             set(identities) == {"official_bf16", "official_fp8",
                                 "dequantized_fp8_derivative",
                                 "native_fp8_execution"},
             "precision identities are not the declared four")
    bf16 = identities["official_bf16"]
    fp8 = identities["official_fp8"]
    derivative = identities["dequantized_fp8_derivative"]
    native = identities["native_fp8_execution"]
    _require("identities.official_bf16",
             bf16["is_official_release"] is True
             and bf16["storage_dtype"] == "bfloat16"
             and bf16["training_qualified"] is False,
             "official BF16 identity is inconsistent")
    _require("identities.official_fp8",
             fp8["is_official_release"] is True
             and fp8["storage_dtype"] == "float8_e4m3fn"
             and fp8["activation_quantization"]
             and fp8["training_qualified"] is False,
             "official FP8 identity is inconsistent")
    _require("identities.derivative",
             derivative["is_official_release"] is False
             and derivative["parent"] == "official_fp8"
             and derivative["storage_dtype"] == "bfloat16"
             and derivative["equivalent_to_bf16_release"] is False
             and derivative["must_not_load_under_official_identity"] is True
             and "no per-token" in derivative["activation_quantization"],
             "dequantized derivative is not a distinct named artifact")
    _require("identities.native_fp8",
             native["is_artifact"] is False
             and native["storage_dtype"] is None
             and native["evidenced"] is False
             and native["training_qualified"] is False
             and native["required_kernels"],
             "native FP8 execution is conflated with storage")
    _require("identities.derivative_vs_official",
             derivative["identity"] != fp8["identity"]
             and derivative["repo_id"] != fp8["repo_id"],
             "derivative shares the official FP8 artifact identity")
    passed("identities",
           "BF16, FP8 storage, dequantized derivative and native FP8 execution separated")

    protocol = config["parity_protocol"]
    prompt_set = protocol["prompt_set"]
    _require("parity.bounded",
             prompt_set["max_prompts"] <= 4
             and prompt_set["dataset_access"] is False
             and prompt_set["heldout_access"] == "forbidden"
             and prompt_set["max_new_tokens"] <= 8,
             "parity protocol is not bounded to text prompts")
    probe = protocol["small_reload_probe"]
    _require("parity.reload",
             probe["grants_training_qualification"] is False
             and probe["grants_fp8_training_qualification"] is False
             and probe["prohibited_inferences"],
             "small reload probe must not grant training qualification")
    _require("parity.tolerances",
             protocol["quality_tolerances"]["status"]
             == "must_be_pinned_before_any_execution"
             and protocol["quality_tolerances"]["numeric_values"] is None,
             "quality tolerances must remain unpinned until execution preparation")
    passed("parity", "bounded text-only parity; reload cannot qualify training")

    policy = config["claims_policy"]
    _require("claims.qualification",
             policy["training_qualified"] is False
             and policy["native_fp8_training_qualified"] is False
             and policy["capability_claim"] == "none",
             "preparation claims training qualification")
    _require("claims.storage",
             policy["storage_dtype_implies_compute"] is False
             and policy["storage_dtype_implies_training_qualification"] is False
             and policy["small_reload_success_implies_training_qualification"] is False
             and policy["dequantized_derivative_implies_official_fp8"] is False,
             "a forbidden storage/reload inference is enabled")
    forbidden = " ".join(policy["forbidden_statements"]).casefold()
    for needle in ("storage dtype", "reload", "dequantized", "mps fp8"):
        _require("claims.forbidden", needle in forbidden,
                 f"forbidden statements do not cover {needle!r}")
    _require("claims.non_qualifying",
             {"storage dtype inspection", "small reload success"}
             <= set(policy["non_qualifying_evidence"]),
             "non-qualifying evidence list is incomplete")

    empty = evaluate_training_claim(None, config)
    storage_only = evaluate_training_claim(
        {gate["id"]: {"evidence_kind": "storage_dtype", "measured": True,
                      "artifact_sha256": "0" * 64, "identity_sufficient": True}
         for gate in policy["training_qualification_gates"]}, config)
    reload_only = evaluate_training_claim(
        {gate["id"]: {"evidence_kind": "small_reload", "measured": True,
                      "artifact_sha256": "0" * 64, "identity_sufficient": True}
         for gate in policy["training_qualification_gates"]}, config)
    _require("claims.evaluator", empty["qualified"] is False
             and storage_only["qualified"] is False
             and reload_only["qualified"] is False,
             "training claim evaluator admits non-qualifying evidence")
    passed("claims", "dtype and reload evidence cannot qualify training")

    estimates = estimate_resources(config)
    passed("resources",
           "analytical estimates: extension within budget; full tuning blocked; "
           "native FP8 unquantified")

    ablation = config["later_ablation"]
    _require("ablation.status",
             ablation["status"] == "proposed_not_authorized"
             and ablation["authorized_by_this_config"] is False,
             "self-replay ablation must remain unauthorized")
    truth = ablation["ground_truth_policy"]
    _require("ablation.ground_truth",
             truth["unqualified_visual_answers_as_ground_truth"] is False
             and truth["model_generated_captions_as_ground_truth"] is False
             and truth["dequantized_derivative_output_as_teacher"] is False,
             "ablation admits unqualified visual/model labels")
    filters = " ".join(ablation["filters"]).casefold()
    _require("ablation.filters",
             "text-only" in filters and "image" in filters
             and "duplicate" in filters,
             "ablation filters do not exclude visual and duplicate rows")
    _require("ablation.controls",
             "unfiltered replay is not admitted"
             in [control.casefold() for control in ablation["controls"]],
             "ablation must reject unfiltered replay")
    passed("ablation", "filtered text-only self-replay documented; not authorized")

    _require("non_claims",
             config["non_claims"]["native_execution_performed"] is False
             and config["non_claims"]["training_qualified"] is False
             and config["non_claims"]["model_loaded"] is False
             and config["non_claims"]["measured_resources"] is False,
             "non-claims block is inconsistent")
    passed("non_claims", "no native execution, model load or training claim")

    source_text = Path(__file__).read_text(encoding="utf-8")
    for name in FORBIDDEN_IMPORT_NAMES:
        _require("checker.offline", f"import {name}" not in source_text,
                 f"checker imports {name}")
    passed("checker.offline", "checker imports no tensor or subprocess library")

    return {
        "schema": CHECK_SCHEMA,
        "checker_version": CHECKER_VERSION,
        "config_path": str(_display_path(config_path, project_root)),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")).hexdigest(),
        "checks": checks,
        "estimates": estimates,
        "training_qualified": False,
        "native_fp8_training_qualified": False,
        "native_fp8_execution_evidenced": False,
        "native_execution_performed": False,
        "model_loaded": False,
        "blocked_reasons": [
            "no separate reviewed authorization receipt",
            "no measured native load, parity, gradient, update, reload or "
            "resource evidence",
            "no native FP8 kernel evidence",
        ],
        "admission_scope": "preparation only; nothing here qualifies training",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(
        ROOT / "configs/spark_backend_parity_20261027.json"))
    parser.add_argument("--require-qualified", action="store_true",
                        help="exit non-zero unless training is fully qualified "
                             "(always non-zero in preparation)")
    args = parser.parse_args(argv)
    try:
        report = check(args.config)
    except CheckError as error:
        print(json.dumps({"schema": CHECK_SCHEMA, "status": "invalid",
                          "failed_check": error.check, "error": error.message,
                          "training_qualified": False,
                          "native_execution_performed": False},
                         indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_qualified and not report["training_qualified"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
