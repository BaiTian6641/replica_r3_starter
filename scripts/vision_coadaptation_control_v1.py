#!/usr/bin/env python3
"""CPU-only validator and readiness assessment for the co-adaptation control.

The co-adaptation control is a separately scoped, preparation-only protocol that
may be considered only after the preserved lr 3e-05 smaller-step arm fails and a
preserved diagnosis attributes the failure to frozen backbone features.  It
changes exactly one axis relative to that arm -- gradient scope, adding the last
SigLIP2 encoder block and post-layernorm to the six projector tensors -- and
keeps the parent, encoder weights, data, update batch, initialization, lr,
optimizer hyperparameters, response-only loss, readout protocol, generation
policy and budget identical.

This module never imports a native runtime, loads a model, runs a backward call
or optimizer step, writes an authorization receipt, or authorizes development,
held-out, semantic-audit, V0, promotion, retry or any additional control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/vision_coadaptation_control_v1.json"
SMALLER_STEP_CONFIG_PATH = "configs/vision_smaller_step_lr3e5_v1.json"
REPORT_PATH = ("reports/vision_coadaptation_control_v1_20261027/"
               "readiness.json")
FORMAT = "replica.vision_coadaptation_control.assessment.v1"
VERSION = "1"
PROTOCOL_FORMAT = "replica.vision_coadaptation_control.protocol.v1"
EXPECTED_SMALLER_STEP_CONFIG_SHA256 = (
    "931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad")
EXPECTED_LR = 3e-05
EXPECTED_ENCODER_BOUNDS = {"max_abs_update": 1e-4, "max_delta_l2": 0.25}
EXPECTED_REQUIREMENT_IDS = [
    "smaller_step_failure_preserved",
    "frozen_feature_diagnosis_preserved",
    "coadaptation_runner_implemented",
    "separate_authorization_receipt_issued",
    "protocol_commit_recorded",
    "review_launch_gate_passed",
    "fresh_output_available",
    "exclusive_worker_and_capacity_guard",
    "native_execution_within_identical_budget",
]
EXPECTED_DIAGNOSIS_FIELDS = {
    "format", "status", "conclusion", "excluded_causes",
    "smaller_step_report", "smaller_step_report_sha256", "observed",
    "issued_at_utc",
}
OPTIMIZATION_SHARED_KEYS = (
    "evaluated_on", "update_count", "backward_calls", "optimizer_steps",
    "backbone_gradient_tensors", "finite_losses", "finite_grad_norm",
    "max_update_batch_mean_nll_delta", "max_wider_answer_mean_nll_delta",
    "max_wider_caption_mean_nll_delta", "delta_definition",
    "loss_movement_is_not_conversion",
)
OPTIMIZATION_EXTRA_KEYS = {
    "optimizer_owned_tensors", "projector_tensors", "encoder_tensors",
    "decoder_gradient_tensors", "declared_scope_tensors_only_changed",
    "max_encoder_abs_update", "max_encoder_delta_l2", "status_on_fail",
}
DECISION_EXTRA_KEYS = {"more_coadaptation_authorized"}
TOP_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "hypothesis", "parent", "prepared", "schedule", "initialization",
    "optimizer", "loss_contract", "gradient_scope", "scope_axis",
    "update_batch", "readouts", "gates", "accounting", "budget", "limits",
    "policy", "access", "precondition", "implementation", "authorization",
    "statuses", "claims", "distinctness", "limitations", "output",
    "report_root",
}
HYPOTHESIS_KEYS = {
    "statement", "arm", "changed_axis", "lr", "same_lr_as_smaller_step",
    "requires_diagnosed_smaller_step_failure",
    "frozen_feature_insufficiency_is_the_only_question",
    "not_a_step_size_test", "not_an_encoder_swap", "not_a_new_loss",
    "not_a_larger_budget", "rank_or_loss_movement_is_not_conversion",
}
PARENT_KEYS = {
    "model_path", "model_lock", "model_lock_sha256", "model_revision",
    "vision_source_manifest", "vision_source_manifest_sha256",
    "vision_weights_sha256", "sealed_parent", "downloads", "encoder_swap",
    "comparison_arm", "comparison_config", "comparison_report",
    "same_parent_as_smaller_step", "same_encoder_weights_as_smaller_step",
}
LOSS_KEYS = {
    "name", "metric", "label_scope", "teacher_forced_only_for_loss",
    "auxiliary_terms", "auxiliary_loss_weight", "new_loss_introduced",
    "same_loss_as_smaller_step", "same_label_policy_as_smaller_step",
}
GRADIENT_SCOPE_KEYS = {
    "trainable_tensors", "projector_tensors", "encoder_tensors",
    "projector_parameter_count", "encoder_parameter_count",
    "trainable_parameter_count", "optimizer_param_groups",
    "optimizer_owned_parameters", "encoder_structure", "frozen_components",
    "backbone_gradient_tensors_allowed", "decoder_gradient_tensors_allowed",
    "non_scope_gradient_names_allowed", "gradient_accumulation",
    "update_rows", "answer_rows", "caption_rows", "backward_calls",
    "optimizer_steps", "clip_grad_norm",
    "changed_tensors_must_be_subset_of_trainable",
    "declared_scope_tensors_only_changed", "encoder_update_bounds",
}
ENCODER_STRUCTURE_KEYS = {
    "architecture", "base_model_prefix", "feature_version",
    "num_hidden_layers", "hidden_size", "intermediate_size",
    "adapted_layer_index", "adapted_layer_modules", "post_layernorm",
    "frozen_layers",
}
SCOPE_AXIS_KEYS = {
    "changed", "from", "to", "relative_to", "unchanged", "encoder_swap",
    "new_loss", "budget_increase",
}
BUDGET_KEYS = {
    "relation_to_smaller_step", "limits", "campaign_cumulative",
    "expected_measured_work", "wall_limit_seconds", "artifact_bytes",
    "additional_forward_calls", "additional_backward_calls",
    "additional_optimizer_steps", "additional_wall_seconds",
    "additional_artifact_bytes",
    "encoder_backward_included_in_the_same_four_backward_calls",
}
PRECONDITION_KEYS = {
    "requires_diagnosed_smaller_step_failure", "smaller_step_config",
    "smaller_step_config_sha256", "smaller_step_report",
    "accepted_smaller_step_statuses", "diagnosis_report",
    "diagnosis_format", "required_diagnosis_conclusion",
    "required_excluded_causes", "requires_preserved_evidence",
    "diagnosis_created_here", "note",
}
IMPLEMENTATION_KEYS = {
    "mode", "runner_path", "runner_implemented", "native_execution",
    "source_edits", "reuse", "note",
}
AUTHORIZATION_KEYS = {
    "native_execution_authorized", "requires_protocol_commit",
    "protocol_commit", "receipt_path", "receipt_format", "required_status",
    "requires_exact_config_sha256", "requires_commit_binding",
    "requires_diagnosis_precondition", "separate_from_smaller_step_receipt",
    "required_receipt_fields",
}
CLAIMS_KEYS = {
    "hypothesis_scope", "smaller_step_retry", "encoder_swapped",
    "new_loss_introduced", "budget_increased",
    "generated_answer_conversion_claim", "rank_improvement_is_conversion",
    "development_authorized", "heldout_authorized", "v0_authorized",
    "semantic_audit_authorized", "promotion", "retry_authorized",
    "native_execution_authorized",
}
DISTINCTNESS_KEYS = {
    "not_a_retry_of", "retry_of_completed_or_failed_arm",
    "single_changed_axis", "same_lr", "same_update_rows",
    "same_update_count", "same_generated_tokens",
    "same_forward_and_backward_counts", "same_loss", "same_encoder_weights",
    "why_distinct_hypothesis", "why_not_an_expansion", "predictions",
    "closure_on_failure",
}
REQUIREMENT_KEYS = {"id", "blocking", "requirement", "owner",
                    "satisfied_by_preparation"}
REPORT_KEYS = {
    "format", "version", "name", "deterministic", "native_execution",
    "model_loads", "authorization_created", "protocol", "parent", "scope",
    "budget", "gates", "precondition", "authorization", "distinctness",
    "checks", "remaining_requirements", "claims", "decision",
    "assessment_sha256",
}


class CoadaptationError(ValueError):
    """The co-adaptation protocol is incomplete, enlarged or drifted."""


class StaleCoadaptationError(CoadaptationError):
    """The checked-in co-adaptation assessment is stale."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise CoadaptationError(message)


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == keys,
             f"{label} fields differ: {sorted(set(value) ^ keys)}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise CoadaptationError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON must be an object: {path}")
    return dict(payload)


def _require_commit(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 40
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be a full 40-character lowercase commit SHA")


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob",
             f"{commit}:{relative}"],
            check=False, capture_output=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return _sha256_bytes(completed.stdout)


def _encoder_tensor_names(structure: Mapping[str, Any]) -> list[str]:
    layer = structure["adapted_layer_index"]
    names: list[str] = []
    for module in structure["adapted_layer_modules"]:
        names.append(f"vision_model.encoder.layers.{layer}.{module}.weight")
        names.append(f"vision_model.encoder.layers.{layer}.{module}.bias")
    if structure["post_layernorm"]:
        names.append("vision_model.post_layernorm.weight")
        names.append("vision_model.post_layernorm.bias")
    return names


def _encoder_parameter_count(structure: Mapping[str, Any]) -> int:
    hidden = structure["hidden_size"]
    intermediate = structure["intermediate_size"]
    attention = 4 * (hidden * hidden + hidden)
    layer_norms = 2 * (2 * hidden)
    mlp = (hidden * intermediate + intermediate) + (
        intermediate * hidden + hidden)
    post = 2 * hidden
    return attention + layer_norms + mlp + post


def _validate_schema(config: Mapping[str, Any]) -> None:
    _exact_keys(config, TOP_KEYS, "co-adaptation protocol")
    _require(config["format"] == PROTOCOL_FORMAT,
             "co-adaptation protocol format differs")
    _require(config["mode"] == "coadaptation_control",
             "co-adaptation mode differs")
    _exact_keys(config["hypothesis"], HYPOTHESIS_KEYS, "hypothesis")
    _exact_keys(config["parent"], PARENT_KEYS, "parent")
    _exact_keys(config["loss_contract"], LOSS_KEYS, "loss_contract")
    _exact_keys(config["gradient_scope"], GRADIENT_SCOPE_KEYS,
                "gradient_scope")
    _exact_keys(config["gradient_scope"]["encoder_structure"],
                ENCODER_STRUCTURE_KEYS, "encoder_structure")
    _exact_keys(config["gradient_scope"]["encoder_update_bounds"],
                set(EXPECTED_ENCODER_BOUNDS), "encoder_update_bounds")
    _exact_keys(config["scope_axis"], SCOPE_AXIS_KEYS, "scope_axis")
    _exact_keys(config["budget"], BUDGET_KEYS, "budget")
    _exact_keys(config["precondition"], PRECONDITION_KEYS, "precondition")
    _exact_keys(config["implementation"], IMPLEMENTATION_KEYS,
                "implementation")
    _exact_keys(config["authorization"], AUTHORIZATION_KEYS, "authorization")
    _exact_keys(config["claims"], CLAIMS_KEYS, "claims")
    _exact_keys(config["distinctness"], DISTINCTNESS_KEYS, "distinctness")
    _require(set(config["distinctness"]["predictions"]) == {
        "frozen_features_sufficient", "frozen_features_insufficient"},
        "distinctness predictions differ")
    _require(set(config["gates"]) == {
        "optimization", "functional_fit", "decision"},
        "gates fields differ")
    _require(config["hypothesis"]["lr"] == EXPECTED_LR,
             "co-adaptation lr differs")
    _require(config["hypothesis"]["same_lr_as_smaller_step"] is True,
             "co-adaptation must keep the smaller-step lr")
    _require(config["hypothesis"]["requires_diagnosed_smaller_step_failure"]
             is True, "co-adaptation requires a diagnosed smaller-step failure")
    for flag in ("not_a_step_size_test", "not_an_encoder_swap",
                 "not_a_new_loss", "not_a_larger_budget"):
        _require(config["hypothesis"][flag] is True,
                 f"hypothesis flag must be true: {flag}")
    _require(config["hypothesis"]["rank_or_loss_movement_is_not_conversion"]
             is True, "loss movement must not be treated as conversion")
    _require(config["scope_axis"]["changed"] == "gradient_scope",
             "scope axis must be gradient_scope")
    _require(config["scope_axis"]["encoder_swap"] is False,
             "encoder swap is forbidden")
    _require(config["scope_axis"]["new_loss"] is False,
             "new loss is forbidden")
    _require(config["scope_axis"]["budget_increase"] is False,
             "budget increase is forbidden")
    _require(config["precondition"]["diagnosis_created_here"] is False,
             "the protocol cannot create its own diagnosis")
    _require(config["implementation"]["runner_implemented"] is False,
             "the protocol declares no implemented runner")
    _require(config["implementation"]["native_execution"] is False,
             "the protocol cannot authorize native execution")
    _require(config["implementation"]["source_edits"] == 0,
             "the protocol must not edit reused sources")
    _require(config["authorization"]["native_execution_authorized"] is False,
             "predeclared native authorization must be false")
    _require(config["authorization"]["requires_protocol_commit"] is True,
             "protocol commit requirement must be true")
    _require(config["authorization"]["protocol_commit"] is None,
             "predeclared protocol commit must remain null")
    _require(config["authorization"]["requires_diagnosis_precondition"]
             is True, "authorization must require the diagnosis precondition")
    _require(config["budget"]["relation_to_smaller_step"]
             == "identical_not_larger", "budget relation differs")
    _require(config["implementation"]["runner_path"]
             not in (CONFIG_PATH, SMALLER_STEP_CONFIG_PATH),
             "runner path must be separate from both protocols")


def _validate_single_axis(config: Mapping[str, Any],
                          smaller: Mapping[str, Any]) -> dict[str, Any]:
    """Require every axis other than gradient scope to be byte-identical."""
    for block in ("prepared", "schedule", "initialization", "optimizer",
                  "update_batch", "readouts", "accounting", "limits",
                  "policy", "access"):
        _require(config[block] == smaller[block],
                 f"unchanged block differs from the smaller-step arm: {block}")
    _require(config["optimizer"]["lr"] == smaller["optimizer"]["lr"],
             "optimizer lr differs from the smaller-step arm")
    _require(config["schedule"]["updates_planned"] == 1,
             "co-adaptation must keep exactly one update")
    _require(config["schedule"]["update_index"]
             == smaller["schedule"]["update_index"],
             "update index differs")
    _require(config["update_batch"]["example_ids"]
             == smaller["update_batch"]["example_ids"],
             "update batch differs")
    generated_tokens = config["readouts"]["expected_measured_work"][
        "generated_tokens"]
    _require(generated_tokens == 3800, "generated token count differs")
    _require(config["readouts"]["wider"]["max_new_tokens_by_kind"]
             == smaller["readouts"]["wider"]["max_new_tokens_by_kind"],
             "generation token limits differ")
    _require(config["limits"]["max_decoder_forwards"] == 4824,
             "decoder forward cap differs")
    _require(config["limits"]["max_visual_encoder_calls"] == 168,
             "visual observation cap differs")
    _require(config["limits"]["max_total_backward_calls"] == 4,
             "backward cap differs")
    _require(config["limits"]["max_optimizer_steps"] == 1,
             "optimizer step cap differs")
    _require(config["limits"]["max_seconds"] == 1200,
             "wall cap differs")
    _require(config["limits"]["artifact_bytes"] == 33554432,
             "artifact byte cap differs")
    _require(config["limits"]["model_loads"] == 1,
             "model load cap differs")
    _require(config["limits"]["downloads"] == 0
             and config["limits"]["retries"] == 0
             and config["limits"]["workers"] == 1,
             "download/retry/worker caps differ")
    return {
        "unchanged_blocks": [
            "prepared", "schedule", "initialization", "optimizer",
            "update_batch", "readouts", "accounting", "limits", "policy",
            "access"],
        "changed_block": "gradient_scope",
    }


def _validate_parent(config: Mapping[str, Any],
                     smaller: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent = config["parent"]
    source = smaller["source"]
    for key in ("model_path", "model_lock", "model_lock_sha256",
                "model_revision", "vision_source_manifest",
                "vision_source_manifest_sha256"):
        _require(parent[key] == source[key],
                 f"parent differs from the smaller-step arm: {key}")
    _require(parent["sealed_parent"] is True, "parent must stay sealed")
    _require(parent["downloads"] == 0, "parent downloads must be zero")
    _require(parent["encoder_swap"] is False, "encoder swap is forbidden")
    _require(parent["same_parent_as_smaller_step"] is True,
             "parent identity must be declared")
    _require(parent["same_encoder_weights_as_smaller_step"] is True,
             "encoder weight identity must be declared")
    _require(parent["comparison_arm"] == smaller["name"],
             "comparison arm differs")
    _require(parent["comparison_config"] == SMALLER_STEP_CONFIG_PATH,
             "comparison config differs")
    _require(parent["comparison_report"]
             == f"{smaller['report_root']}/report.json",
             "comparison report path differs")
    manifest_path = root / parent["vision_source_manifest"]
    _require(manifest_path.is_file(),
             f"vision source manifest is missing: {parent['vision_source_manifest']}")
    manifest = _read_json(manifest_path)
    _require(_sha256_file(manifest_path)
             == parent["vision_source_manifest_sha256"],
             "vision source manifest hash differs")
    weights = next((entry["sha256"] for entry in manifest["files"]
                    if entry["role"] == "vision_encoder_weights"), None)
    _require(weights == parent["vision_weights_sha256"],
             "vision encoder weight hash differs")
    _require(manifest["model"]["architecture"]
             == config["gradient_scope"]["encoder_structure"]["architecture"],
             "encoder architecture differs from the manifest")
    _require(manifest["model"]["feature_version"]
             == config["gradient_scope"]["encoder_structure"]
             ["feature_version"], "encoder feature version differs")
    return {
        "model_path": parent["model_path"],
        "model_revision": parent["model_revision"],
        "vision_source_manifest": parent["vision_source_manifest"],
        "vision_source_manifest_sha256":
            parent["vision_source_manifest_sha256"],
        "vision_weights_sha256": weights,
        "encoder_swap": False,
        "comparison_arm": parent["comparison_arm"],
        "comparison_report": parent["comparison_report"],
    }


def _validate_scope(config: Mapping[str, Any],
                    smaller: Mapping[str, Any]) -> dict[str, Any]:
    scope = config["gradient_scope"]
    smaller_scope = smaller["gradient_scope"]
    projector = list(smaller_scope["trainable_tensors"])
    _require(config["loss_contract"]["name"] == "response_only_mean_nll",
             "loss name differs")
    _require(scope["projector_tensors"] == projector,
             "projector tensors differ from the smaller-step arm")
    structure = scope["encoder_structure"]
    names = _encoder_tensor_names(structure)
    _require(scope["encoder_tensors"] == names,
             "declared encoder tensors differ from the declared structure")
    _require(len(names) == 18, "encoder tensor count differs")
    _require(len(set(names)) == 18, "encoder tensor names repeat")
    _require(scope["trainable_tensors"] == projector + names,
             "trainable tensors must be exactly projector plus encoder scope")
    _require(len(scope["trainable_tensors"]) == 24,
             "optimizer-owned tensor count differs")
    _require(scope["optimizer_owned_parameters"] == 24,
             "declared optimizer ownership differs")
    _require(scope["optimizer_param_groups"] == 1,
             "single optimizer param group required")
    encoder_parameters = _encoder_parameter_count(structure)
    _require(scope["encoder_parameter_count"] == encoder_parameters,
             "encoder parameter count differs")
    _require(scope["projector_parameter_count"]
             == smaller["initialization"]["parameter_count"],
             "projector parameter count differs")
    _require(scope["trainable_parameter_count"]
             == scope["projector_parameter_count"]
             + scope["encoder_parameter_count"],
             "total trainable parameter count differs")
    _require(structure["adapted_layer_index"]
             == structure["num_hidden_layers"] - 1,
             "only the last encoder block may adapt")
    _require(structure["post_layernorm"] is True,
             "post-layernorm must adapt")
    for name in names:
        _require(name.startswith("vision_model.encoder.layers.11.")
                 or name.startswith("vision_model.post_layernorm."),
                 f"encoder tensor outside the declared layer: {name}")
    _require(scope["backbone_gradient_tensors_allowed"] == 0,
             "backbone gradients are forbidden")
    _require(scope["decoder_gradient_tensors_allowed"] == 0,
             "decoder gradients are forbidden")
    _require(scope["non_scope_gradient_names_allowed"] == [],
             "non-scope gradient names are forbidden")
    _require(scope["changed_tensors_must_be_subset_of_trainable"] is True,
             "changed tensors must be a subset of the declared scope")
    _require(scope["declared_scope_tensors_only_changed"] is True,
             "only declared scope tensors may change")
    _require(scope["encoder_update_bounds"] == EXPECTED_ENCODER_BOUNDS,
             "encoder drift bounds differ")
    _require(scope["backward_calls"] == smaller_scope["backward_calls"] == 4,
             "backward call count differs")
    _require(scope["optimizer_steps"] == smaller_scope["optimizer_steps"] == 1,
             "optimizer step count differs")
    _require(scope["update_rows"] == smaller_scope["update_rows"] == 4,
             "update row count differs")
    _require(scope["answer_rows"] == 2 and scope["caption_rows"] == 2,
             "row roles differ")
    _require(scope["gradient_accumulation"] == 1, "gradient accumulation differs")
    _require(scope["clip_grad_norm"] == smaller_scope["clip_grad_norm"] == 1.0,
             "clip grad norm differs")
    _require(len(scope["frozen_components"]) == 2,
             "frozen components differ")
    return {
        "changed_axis": "gradient_scope_encoder_last_block_and_post_layernorm",
        "projector_tensors": 6,
        "encoder_tensors": len(names),
        "optimizer_owned_tensors": 24,
        "projector_parameter_count": scope["projector_parameter_count"],
        "encoder_parameter_count": encoder_parameters,
        "trainable_parameter_count": scope["trainable_parameter_count"],
        "adapted_layer_index": structure["adapted_layer_index"],
        "encoder_update_bounds": dict(EXPECTED_ENCODER_BOUNDS),
        "decoder_gradient_tensors": 0,
    }


def _validate_gates(config: Mapping[str, Any],
                    smaller: Mapping[str, Any]) -> dict[str, Any]:
    optimization = config["gates"]["optimization"]
    smaller_optimization = smaller["gates"]["optimization"]
    for key in OPTIMIZATION_SHARED_KEYS:
        _require(optimization[key] == smaller_optimization[key],
                 f"optimization gate differs: {key}")
    _require(set(optimization) - set(OPTIMIZATION_SHARED_KEYS)
             == OPTIMIZATION_EXTRA_KEYS,
             "optimization scope fields differ")
    _require(optimization["optimizer_owned_tensors"] == 24,
             "optimization ownership gate differs")
    _require(optimization["encoder_tensors"] == 18,
             "optimization encoder tensor gate differs")
    _require(optimization["decoder_gradient_tensors"] == 0,
             "optimization decoder gradient gate differs")
    _require(optimization["max_encoder_abs_update"]
             == EXPECTED_ENCODER_BOUNDS["max_abs_update"],
             "encoder absolute update gate differs")
    _require(optimization["max_encoder_delta_l2"]
             == EXPECTED_ENCODER_BOUNDS["max_delta_l2"],
             "encoder delta L2 gate differs")
    functional = config["gates"]["functional_fit"]
    smaller_functional = dict(smaller["gates"]["functional_fit"])
    _require(set(functional) == set(smaller_functional),
             "functional gate fields differ")
    _require(functional["status_on_fail"]
             == "coadaptation_functional_fit_failed",
             "functional fail status differs")
    for key in set(smaller_functional) - {"status_on_fail"}:
        _require(functional[key] == smaller_functional[key],
                 f"functional gate differs: {key}")
    _require(functional["eos_only_not_counted_correct"] is True
             and functional["whitespace_only_not_counted_correct"] is True,
             "empty-text accounting differs")
    decision = config["gates"]["decision"]
    _require(set(decision) - set(smaller["gates"]["decision"])
             == DECISION_EXTRA_KEYS,
             "decision fields differ")
    for key, value in smaller["gates"]["decision"].items():
        _require(decision[key] == value,
                 f"decision closure differs: {key}")
    _require(decision["more_coadaptation_authorized"] is False,
             "further co-adaptation must stay closed")
    return {
        "optimization": dict(optimization),
        "functional_fit": dict(functional),
        "decision": dict(decision),
    }


def _validate_loss_and_claims(config: Mapping[str, Any],
                              smaller: Mapping[str, Any]) -> dict[str, Any]:
    loss = config["loss_contract"]
    _require(loss["metric"] == smaller["update_batch"]["loss_metric"]
             == "mean_response_nll", "loss metric differs")
    _require(config["update_batch"]["loss_metric"] == loss["metric"],
             "update-batch loss metric differs")
    _require(loss["label_scope"] == "response_only",
             "label scope differs")
    _require(loss["teacher_forced_only_for_loss"] is True,
             "teacher-forced label policy differs")
    _require(loss["auxiliary_terms"] == [], "auxiliary loss terms are forbidden")
    _require(loss["auxiliary_loss_weight"] == 0.0,
             "auxiliary loss weight is forbidden")
    _require(loss["new_loss_introduced"] is False, "new loss is forbidden")
    _require(loss["same_loss_as_smaller_step"] is True,
             "loss identity must be declared")
    _require(loss["same_label_policy_as_smaller_step"] is True,
             "label-policy identity must be declared")
    claims = config["claims"]
    for flag in ("smaller_step_retry", "encoder_swapped", "new_loss_introduced",
                 "budget_increased", "generated_answer_conversion_claim",
                 "rank_improvement_is_conversion", "development_authorized",
                 "heldout_authorized", "v0_authorized",
                 "semantic_audit_authorized", "promotion", "retry_authorized",
                 "native_execution_authorized"):
        _require(claims[flag] is False, f"closed claim opened: {flag}")
    _require(claims["hypothesis_scope"]
             == "frozen_visual_feature_coadaptation_only",
             "hypothesis scope differs")
    return {"loss": dict(loss), "claims": dict(claims)}


def _validate_budget(config: Mapping[str, Any],
                     smaller: Mapping[str, Any]) -> dict[str, Any]:
    budget = config["budget"]
    _require(budget["limits"] == smaller["limits"],
             "budget limits differ from the smaller-step arm")
    _require(budget["campaign_cumulative"]
             == smaller["accounting"]["campaign_cumulative"],
             "campaign budget differs")
    _require(budget["expected_measured_work"]
             == smaller["readouts"]["expected_measured_work"],
             "measured-work budget differs")
    _require(budget["wall_limit_seconds"]
             == smaller["accounting"]["wall_limit_seconds"] == 1200,
             "wall budget differs")
    _require(budget["artifact_bytes"]
             == smaller["accounting"]["artifact_bytes"] == 33554432,
             "artifact byte budget differs")
    for key in ("additional_forward_calls", "additional_backward_calls",
                "additional_optimizer_steps", "additional_wall_seconds",
                "additional_artifact_bytes"):
        _require(budget[key] == 0, f"budget expansion declared: {key}")
    _require(budget["encoder_backward_included_in_the_same_four_backward_calls"]
             is True, "encoder backward must reuse the four backward calls")
    _require(config["limits"] == smaller["limits"],
             "protocol limits differ from the smaller-step arm")
    _require(config["accounting"] == smaller["accounting"],
             "accounting differs from the smaller-step arm")
    _require(config["readouts"] == smaller["readouts"],
             "readout protocol differs from the smaller-step arm")
    return {
        "relation_to_smaller_step": budget["relation_to_smaller_step"],
        "limits": dict(config["limits"]),
        "campaign_cumulative": dict(budget["campaign_cumulative"]),
        "expected_measured_work": dict(budget["expected_measured_work"]),
        "wall_limit_seconds": budget["wall_limit_seconds"],
        "artifact_bytes": budget["artifact_bytes"],
        "additional_forward_calls": 0,
        "additional_backward_calls": 0,
        "additional_optimizer_steps": 0,
        "additional_wall_seconds": 0,
        "additional_artifact_bytes": 0,
    }


def _validate_distinctness(config: Mapping[str, Any],
                           smaller: Mapping[str, Any]) -> dict[str, Any]:
    distinctness = config["distinctness"]
    _require(distinctness["not_a_retry_of"] == smaller["name"],
             "distinctness names a different arm")
    _require(distinctness["retry_of_completed_or_failed_arm"] is False,
             "co-adaptation cannot be a retry")
    _require(distinctness["single_changed_axis"]
             == "gradient_scope_encoder_last_block_and_post_layernorm",
             "single changed axis differs")
    _require(distinctness["same_lr"] == EXPECTED_LR,
             "distinctness lr differs")
    _require(distinctness["same_update_rows"] is True
             and distinctness["same_update_count"] == 1,
             "distinctness update declarations differ")
    _require(distinctness["same_generated_tokens"]
             == smaller["accounting"]["campaign_cumulative"]
             ["generated_tokens"] == 3800,
             "distinctness generated tokens differ")
    _require(distinctness["same_forward_and_backward_counts"] is True,
             "forward/backward identity must be declared")
    _require(distinctness["same_loss"] is True
             and distinctness["same_encoder_weights"] is True,
             "loss/encoder identity must be declared")
    for key in ("why_distinct_hypothesis", "why_not_an_expansion",
                "closure_on_failure"):
        _require(isinstance(distinctness[key], str)
                 and len(distinctness[key]) > 100,
                 f"distinctness explanation is too short: {key}")
    predictions = distinctness["predictions"]
    for key, value in predictions.items():
        _require(isinstance(value, str) and len(value) > 50,
                 f"prediction is too short: {key}")
    return dict(distinctness)


def _precondition_state(config: Mapping[str, Any], root: Path,
                        smaller_sha256: str) -> dict[str, Any]:
    precondition = config["precondition"]
    _require(precondition["smaller_step_config"]
             == SMALLER_STEP_CONFIG_PATH,
             "precondition names a different smaller-step config")
    _require(precondition["smaller_step_config_sha256"] == smaller_sha256,
             "precondition binds a different smaller-step config hash")
    _require(precondition["smaller_step_report"]
             == f"{config['parent']['comparison_report']}",
             "precondition report path differs")
    _require(precondition["accepted_smaller_step_statuses"]
             == ["optimization_gate_failed",
                 "smaller_step_functional_fit_failed"],
             "accepted smaller-step statuses differ")
    _require(precondition["required_diagnosis_conclusion"]
             == "frozen_backbone_projection_insufficient",
             "required diagnosis conclusion differs")
    _require(precondition["required_excluded_causes"]
             == ["step_size", "label_or_masking", "accounting",
                 "precondition", "seed_or_initialization"],
             "required excluded causes differ")
    _require(precondition["requires_preserved_evidence"] is True,
             "preserved evidence must be required")

    report_path = Path(root) / precondition["smaller_step_report"]
    report_state: dict[str, Any] = {
        "path": precondition["smaller_step_report"],
        "present": report_path.is_file(),
        "status": None,
        "sha256": None,
        "failure_accepted": False,
    }
    if report_path.is_file():
        report = _read_json(report_path)
        report_state["sha256"] = _sha256_file(report_path)
        status = report.get("status")
        report_state["status"] = status
        report_state["failure_accepted"] = (
            status in precondition["accepted_smaller_step_statuses"])
        if "config_sha256" in report:
            _require(report["config_sha256"] == smaller_sha256,
                     "preserved smaller-step report binds a different config")
    else:
        report_state["status"] = "absent"

    diagnosis_path = Path(root) / precondition["diagnosis_report"]
    diagnosis_state: dict[str, Any] = {
        "path": precondition["diagnosis_report"],
        "present": diagnosis_path.is_file(),
        "valid": False,
        "conclusion": None,
        "violations": ["diagnosis_report_missing"],
    }
    if diagnosis_path.is_file():
        violations: list[str] = []
        try:
            diagnosis = _read_json(diagnosis_path)
        except CoadaptationError as error:
            diagnosis = {}
            violations.append(f"unreadable_diagnosis:{error}")
        if diagnosis:
            if set(diagnosis) != EXPECTED_DIAGNOSIS_FIELDS:
                violations.append("diagnosis_fields_differ")
            if diagnosis.get("format") != precondition["diagnosis_format"]:
                violations.append("diagnosis_format_differs")
            if diagnosis.get("status") != "diagnosed":
                violations.append("diagnosis_status_not_diagnosed")
            if diagnosis.get("conclusion") != precondition[
                    "required_diagnosis_conclusion"]:
                violations.append("diagnosis_conclusion_differs")
            excluded = diagnosis.get("excluded_causes")
            if not isinstance(excluded, list) or not set(
                    precondition["required_excluded_causes"]).issubset(
                        set(excluded)):
                violations.append("diagnosis_does_not_exclude_alternatives")
            if diagnosis.get("smaller_step_report") != precondition[
                    "smaller_step_report"]:
                violations.append("diagnosis_names_a_different_report")
            if report_state["sha256"] is None or diagnosis.get(
                    "smaller_step_report_sha256") != report_state["sha256"]:
                violations.append("diagnosis_report_hash_differs")
            if not isinstance(diagnosis.get("issued_at_utc"), str) or not \
                    diagnosis["issued_at_utc"].strip():
                violations.append("diagnosis_issued_at_missing")
        diagnosis_state = {
            "path": precondition["diagnosis_report"],
            "present": True,
            "valid": not violations,
            "conclusion": diagnosis.get("conclusion"),
            "violations": violations,
        }
    eligible = (report_state["failure_accepted"]
                and diagnosis_state["valid"])
    return {
        "required": True,
        "smaller_step_report": report_state,
        "diagnosis": diagnosis_state,
        "eligible": eligible,
        "eligibility_rule": ("the preserved smaller-step report must record "
                             "optimization_gate_failed or "
                             "smaller_step_functional_fit_failed and the "
                             "preserved diagnosis must conclude "
                             "frozen_backbone_projection_insufficient while "
                             "excluding step size, labels/masking, accounting, "
                             "precondition and seed/initialization"),
        "note": ("An engineering stop, a retry, a passing smaller-step arm or "
                 "a missing diagnosis is not eligibility."),
    }


def _authorization_state(config: Mapping[str, Any], root: Path,
                         precondition: Mapping[str, Any],
                         smaller: Mapping[str, Any],
                         protocol_sha256: str, *,
                         git_root: str | Path | None = None) -> dict[str, Any]:
    binding_root = Path(git_root) if git_root is not None else root
    path = Path(root) / config["authorization"]["receipt_path"]
    _require(config["authorization"]["receipt_path"]
             != smaller["authorization"]["receipt_path"],
             "co-adaptation must not reuse the smaller-step receipt")
    if not path.is_file():
        return {
            "path": config["authorization"]["receipt_path"],
            "present": False,
            "status": "absent",
            "receipt_sha256": None,
            "protocol_commit": None,
            "commit_bound": False,
            "native_execution_authorized": False,
            "violations": ["authorization_receipt_missing"],
        }
    payload = path.read_bytes()
    receipt = _read_json(path)
    violations: list[str] = []
    required = set(config["authorization"]["required_receipt_fields"])
    if set(receipt) != required:
        violations.append("receipt_fields_differ")
    if receipt.get("format") != config["authorization"]["receipt_format"]:
        violations.append("receipt_format_differs")
    if receipt.get("status") != "authorized":
        violations.append("receipt_status_not_authorized")
    if receipt.get("protocol_name") != config["name"]:
        violations.append("receipt_protocol_name_differs")
    if receipt.get("protocol_path") != CONFIG_PATH:
        violations.append("receipt_protocol_path_differs")
    if receipt.get("config_sha256") != protocol_sha256:
        violations.append("receipt_config_sha256_differs")
    if receipt.get("native_execution_authorized") is not True:
        violations.append("receipt_does_not_authorize_native_execution")
    for field in ("dev_authorized", "heldout_authorized", "v0_authorized",
                  "semantic_audit_authorized", "promotion_authorized",
                  "retry_authorized"):
        if receipt.get(field) is not False:
            violations.append(f"receipt_opens_closed_field:{field}")
    if not isinstance(receipt.get("issued_at_utc"), str) or not \
            receipt["issued_at_utc"].strip():
        violations.append("receipt_issued_at_missing")
    try:
        _require_commit(receipt.get("protocol_commit"),
                        "authorization protocol_commit")
    except CoadaptationError as error:
        violations.append(str(error))
    if not precondition["eligible"]:
        violations.append("receipt_issued_before_diagnosis_precondition")
    commit = receipt.get("protocol_commit")
    bound = False
    if isinstance(commit, str) and len(commit) == 40:
        bound = _git_blob_sha256(binding_root, commit,
                                 CONFIG_PATH) == protocol_sha256
        if not bound:
            violations.append("receipt_commit_does_not_contain_protocol_bytes")
    return {
        "path": config["authorization"]["receipt_path"],
        "present": True,
        "status": "present_valid" if not violations else "present_invalid",
        "receipt_sha256": _sha256_bytes(payload),
        "protocol_commit": commit if isinstance(commit, str) else None,
        "commit_bound": bound,
        "native_execution_authorized":
            receipt.get("native_execution_authorized") is True,
        "violations": violations,
    }


def assess(config_path: str | Path = CONFIG_PATH, *,
           root: str | Path = ROOT) -> dict[str, Any]:
    """Validate the protocol and emit a deterministic readiness assessment."""
    root_path = Path(root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root_path / path
    config = _read_json(path)
    _validate_schema(config)
    protocol_sha256 = _sha256_file(path)

    smaller_path = root_path / SMALLER_STEP_CONFIG_PATH
    _require(smaller_path.is_file(),
             f"smaller-step protocol is missing: {SMALLER_STEP_CONFIG_PATH}")
    smaller_sha256 = _sha256_file(smaller_path)
    _require(smaller_sha256 == EXPECTED_SMALLER_STEP_CONFIG_SHA256,
             "smaller-step protocol hash differs from the declared parent")
    smaller = _read_json(smaller_path)

    axis = _validate_single_axis(config, smaller)
    parent = _validate_parent(config, smaller, root_path)
    scope = _validate_scope(config, smaller)
    gates = _validate_gates(config, smaller)
    loss = _validate_loss_and_claims(config, smaller)
    budget = _validate_budget(config, smaller)
    distinctness = _validate_distinctness(config, smaller)
    precondition = _precondition_state(config, root_path, smaller_sha256)
    authorization = _authorization_state(config, root_path, precondition,
                                         smaller, protocol_sha256)

    runner = root_path / config["implementation"]["runner_path"]
    _require(not runner.exists(),
             "declared runner must not exist while runner_implemented is false")
    _require(config["implementation"]["runner_implemented"] is False,
             "runner implementation flag differs")

    output = config["output"]
    output_exists = (root_path / output).exists()
    requirements: list[dict[str, Any]] = []
    for identifier, blocking, requirement, owner in (
            ("smaller_step_failure_preserved", True,
             "A preserved smaller-step report must record "
             "optimization_gate_failed or smaller_step_functional_fit_failed; "
             "no report exists and an engineering stop does not qualify.",
             "authorized_operator"),
            ("frozen_feature_diagnosis_preserved", True,
             "A preserved diagnosis must conclude "
             "frozen_backbone_projection_insufficient while excluding step "
             "size, labels/masking, accounting, precondition and "
             "seed/initialization; no diagnosis exists.",
             "authorized_operator"),
            ("coadaptation_runner_implemented", True,
             "A separate native runner must implement exactly this scope, "
             "loss, data, readouts and identical budget; none is implemented.",
             "engineer"),
            ("separate_authorization_receipt_issued", True,
             "A separate external receipt must exist at the co-adaptation "
             "receipt path, bind the exact config, authorize native execution "
             "and keep every other role false.",
             "external_user"),
            ("protocol_commit_recorded", True,
             "The receipt must record a full 40-character commit whose "
             "co-adaptation config blob hashes to the receipt config SHA-256.",
             "external_user"),
            ("review_launch_gate_passed", True,
             "A review-launch gate equivalent to the smaller-step runner must "
             "pass receipt, commit, diagnosis and fresh-output checks before "
             "any native import.",
             "authorized_operator"),
            ("fresh_output_available", True,
             f"{output} must not exist; it is absent now.",
             "authorized_operator"),
            ("exclusive_worker_and_capacity_guard", True,
             "The single exclusive MPS worker and the 50 GiB driver, 512 MiB "
             "swap-growth and 15% host-free guard must be proven natively.",
             "authorized_operator"),
            ("native_execution_within_identical_budget", True,
             "The native run must stay inside the identical 4824 forwards, "
             "168 visual observations, 4 backward calls, 1 step, 1200 "
             "seconds and 33554432 bytes including the report.",
             "authorized_operator")):
        satisfied = False
        if identifier == "smaller_step_failure_preserved":
            satisfied = precondition["smaller_step_report"]["failure_accepted"]
        elif identifier == "frozen_feature_diagnosis_preserved":
            satisfied = precondition["diagnosis"]["valid"]
        elif identifier == "fresh_output_available":
            satisfied = not output_exists
        requirements.append({
            "id": identifier,
            "blocking": blocking,
            "requirement": requirement,
            "owner": owner,
            "satisfied_by_preparation": satisfied,
        })
    _require([item["id"] for item in requirements]
             == EXPECTED_REQUIREMENT_IDS,
             "remaining requirement ids differ")

    if not precondition["smaller_step_report"]["present"]:
        reason = ("preparation_complete_but_the_smaller_step_failure_report_"
                  "and_frozen_feature_diagnosis_are_absent")
    elif not precondition["eligible"]:
        reason = "preparation_complete_but_the_diagnosis_precondition_is_unmet"
    elif not authorization["present"]:
        reason = ("eligible_but_unauthorized_and_unimplemented")
    elif authorization["status"] != "present_valid":
        reason = "authorization_receipt_present_but_invalid"
    else:
        reason = ("receipt_present_but_readiness_is_not_a_launch_authority_"
                  "and_the_runner_is_not_implemented")

    checks = [
        {"id": "protocol_schema", "status": "passed",
         "requirement": "the co-adaptation protocol matches its declared schema"},
        {"id": "parent_unchanged", "status": "passed",
         "requirement": "the sealed MiniCPM5 parent and SigLIP2 encoder "
                        "weights are the same objects as the smaller-step arm"},
        {"id": "data_and_schedule_unchanged", "status": "passed",
         "requirement": "prepared04 inputs, update batch and one-update "
                        "schedule are identical"},
        {"id": "initialization_and_optimizer_unchanged", "status": "passed",
         "requirement": "seed 29 initialization and NativeFusedAdamW "
                        "hyperparameters, including lr 3e-05, are identical"},
        {"id": "loss_unchanged", "status": "passed",
         "requirement": "response-only mean NLL with no auxiliary term is "
                        "the same loss"},
        {"id": "readouts_unchanged", "status": "passed",
         "requirement": "five-stage readouts, raw generations, controls and "
                        "empty-text accounting are identical"},
        {"id": "budget_not_larger", "status": "passed",
         "requirement": "4824 decoder forwards, 168 visual observations, "
                        "4 backward calls, 1 optimizer step, 1200 seconds and "
                        "33554432 bytes are identical, with zero additional "
                        "work declared"},
        {"id": "single_axis_scope", "status": "passed",
         "requirement": "gradient scope is the only changed axis: six "
                        "projector tensors plus eighteen last-block and "
                        "post-layernorm tensors"},
        {"id": "encoder_scope_grounded", "status": "passed",
         "requirement": "the eighteen encoder tensors and 7,089,408 parameters "
                        "follow from the preserved SiglipVisionModel manifest"},
        {"id": "gates_preserved", "status": "passed",
         "requirement": "0.25 nat optimization deltas, functional thresholds, "
                        "control margins and empty-text gates are unchanged, "
                        "with encoder drift bounded at 1e-4 absolute and 0.25 "
                        "L2"},
        {"id": "decision_closed", "status": "passed",
         "requirement": "development, held-out, V0, promotion, retry and "
                        "further co-adaptation remain closed"},
        {"id": "distinctness_declared", "status": "passed",
         "requirement": "the control explains why it is a distinct "
                        "hypothesis and not a disguised expansion"},
        {"id": "precondition_state", "status": "observed",
         "requirement": f"smaller-step failure report present="
                        f"{precondition['smaller_step_report']['present']}, "
                        f"diagnosis present={precondition['diagnosis']['present']}, "
                        f"eligible={precondition['eligible']}"},
        {"id": "authorization_state", "status": "observed",
         "requirement": f"receipt state is {authorization['status']}"},
        {"id": "runner_not_implemented", "status": "observed",
         "requirement": f"runner {config['implementation']['runner_path']} "
                        "does not exist"},
        {"id": "no_native_execution", "status": "passed",
         "requirement": "no model load, backward call, optimizer step, "
                        "generation or native command is performed"},
    ]

    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "name": config["name"],
        "deterministic": True,
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "protocol": {
            "path": CONFIG_PATH,
            "sha256": protocol_sha256,
            "output": output,
            "report_root": config["report_root"],
        },
        "parent": parent,
        "scope": {**scope,
                  "single_axis": axis,
                  "trainable_tensors": list(
                      config["gradient_scope"]["trainable_tensors"])},
        "budget": budget,
        "gates": gates,
        "precondition": precondition,
        "authorization": authorization,
        "distinctness": distinctness,
        "checks": checks,
        "remaining_requirements": requirements,
        "claims": loss["claims"],
        "decision": {
            "preparation_complete": True,
            "eligible": precondition["eligible"],
            "launch_allowed": False,
            "native_execution": False,
            "model_loads": 0,
            "authorization_created": False,
            "reason": reason,
        },
    }
    report["assessment_sha256"] = _assessment_sha256(report)
    _exact_keys(report, REPORT_KEYS, "co-adaptation assessment")
    return report


def _assessment_sha256(report: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in report.items()
               if key != "assessment_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(canonical)


def assessment_bytes(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True,
                       ensure_ascii=True) + "\n").encode("utf-8")


def write_assessment(report: Mapping[str, Any],
                     path: str | Path = REPORT_PATH, *,
                     root: str | Path = ROOT) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = Path(root) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(assessment_bytes(report))
    return _sha256_bytes(assessment_bytes(report))


def verify_assessment(report: Mapping[str, Any] | None = None, *,
                      config_path: str | Path = CONFIG_PATH,
                      report_path: str | Path = REPORT_PATH,
                      root: str | Path = ROOT) -> dict[str, Any]:
    root_path = Path(root)
    if report is None:
        path = Path(report_path)
        if not path.is_absolute():
            path = root_path / path
        _require(path.is_file(),
                 f"co-adaptation assessment is missing: {path}")
        report = _read_json(path)
    _exact_keys(report, REPORT_KEYS, "co-adaptation assessment")
    _require(report["format"] == FORMAT,
             "co-adaptation assessment format differs")
    _require(report["assessment_sha256"] == _assessment_sha256(report),
             "co-adaptation assessment hash differs from its payload")
    expected = assess(config_path, root=root_path)
    for key in sorted(REPORT_KEYS - {"assessment_sha256"}):
        if report[key] != expected[key]:
            raise StaleCoadaptationError(
                f"stale co-adaptation section: {key}")
    return dict(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the preparation-only co-adaptation control")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the readiness assessment")
    build.add_argument("--config", default=CONFIG_PATH)
    build.add_argument("--output", default=REPORT_PATH)
    build.add_argument("--root", default=str(ROOT))
    build.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify", help="refuse a stale assessment")
    verify.add_argument("--config", default=CONFIG_PATH)
    verify.add_argument("--report", default=REPORT_PATH)
    verify.add_argument("--root", default=str(ROOT))
    status = sub.add_parser("status", help="print without writing")
    status.add_argument("--config", default=CONFIG_PATH)
    status.add_argument("--root", default=str(ROOT))
    return parser


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "coadaptation_preparation_current",
        "assessment_sha256": report["assessment_sha256"],
        "config_sha256": report["protocol"]["sha256"],
        "changed_axis": report["scope"]["changed_axis"],
        "trainable_tensors": report["scope"]["optimizer_owned_tensors"],
        "trainable_parameter_count":
            report["scope"]["trainable_parameter_count"],
        "budget_relation": report["budget"]["relation_to_smaller_step"],
        "eligible": report["decision"]["eligible"],
        "authorization_status": report["authorization"]["status"],
        "launch_allowed": report["decision"]["launch_allowed"],
        "unsatisfied_requirements": [
            item["id"] for item in report["remaining_requirements"]
            if not item["satisfied_by_preparation"]],
        "decision": report["decision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.root).resolve()
        if args.command == "build":
            report = assess(args.config, root=root)
            digest = write_assessment(report, args.output, root=root)
            if args.json:
                print(json.dumps({**_summary(report),
                                  "report_sha256": digest},
                                 indent=2, sort_keys=True))
            else:
                print(f"assessment_written {args.output}")
                print(f"assessment_sha256={report['assessment_sha256']}")
                print(f"report_sha256={digest}")
                print("native_execution=false model_loads=0 "
                      "authorization_created=false launch_allowed=false")
            return 0
        if args.command == "verify":
            report = verify_assessment(config_path=args.config,
                                      report_path=args.report, root=root)
            print("coadaptation_preparation_current")
            print(f"assessment_sha256={report['assessment_sha256']}")
            print(f"eligible={report['decision']['eligible']}")
            print("launch_allowed=false")
            return 0
        report = assess(args.config, root=root)
        print(json.dumps(_summary(report), indent=2, sort_keys=True))
        return 0
    except CoadaptationError as error:
        print(f"{args.command} refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
