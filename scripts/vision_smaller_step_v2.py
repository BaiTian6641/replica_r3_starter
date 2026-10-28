#!/usr/bin/env python3
"""Deterministic CPU protocol validator and review-only launch gate.

This module owns preparation for one predeclared vision smaller-step arm.  It
does not load a model, import a native runtime, write a checkpoint, read
development/held-out data or execute any part of the experiment.

Commands:

``validate``
    Deterministically validate the JSON protocol and print its SHA-256 plus the
    declared budgets and gates.  Exits 2 on any protocol drift.

``hash``
    Print the exact config-file SHA-256 that an authorization receipt must bind.

``review-launch``
    Re-validate the protocol, require an external authorization receipt whose
    exact config hash and committed protocol identity match, and print a
    review-only disposition plus the real authorized ``run`` command.  It
    refuses before any native loading when the receipt is absent, malformed,
    unbound or uncommitted.  It never executes a native command.

``run``
    Authorized native execution of the exact five stages: prefix-0 readout,
    pre-update four-row teacher-forced evaluation, one NativeFusedAdamW update
    (4 backward calls, 1 step), post-update four-row evaluation and prefix-1
    readout.  Authorization is validated before any native import or model
    load.  This command is implemented but was not executed during preparation.

The protocol remains preparation only until an external authorization receipt
binds a committed protocol.  Native execution does not authorize development,
held-out, semantic-audit, V0, promotion or retry access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_forward_accounting as fwd  # noqa: E402
from replica_lab import vision_smaller_step_accounting_v2 as acc_v2  # noqa: E402


FORMAT = "replica.vision_smaller_step_config.v1"
VERSION = "replica.vision_smaller_step.v1"
FORMAT_V2 = acc_v2.PROTOCOL_FORMAT
VERSION_V2 = acc_v2.PROTOCOL_VERSION
MODE = "smaller_step"
NAME = "vision_smaller_step_lr3e5_v1"
NAME_V2 = acc_v2.NAME
DEFAULT_CONFIG = "configs/vision_smaller_step_lr3e5_v2.json"
AUTHORIZATION_FORMAT = "replica.vision_smaller_step.authorization.v1"
REVIEW_FORMAT = "replica.vision_smaller_step.review.v1"
NATIVE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}
IMPLEMENTATION_RELATIVES = (
    "scripts/vision_smaller_step_v2.py",
    "scripts/run_capacity_preflight.py",
    "src/replica_lab/capacity_guard.py",
    "src/replica_lab/profiling.py",
    "src/replica_lab/optimizers.py",
    "src/replica_lab/nested_learning.py",
    "src/replica_lab/artifact_publication.py",
    "src/replica_lab/vision_forward_accounting.py",
    "src/replica_lab/vision_smaller_step_accounting_v2.py",
    "src/replica_lab/vision_learnability_curve_v2.py",
    "src/replica_lab/vision_learnability_diagnostic.py",
    "src/replica_lab/architecture/loading.py",
    "src/replica_lab/architecture/plastic.py",
)
LR = 0.00003
COMPLETED_LR = 0.0003
PROJECTOR_TENSORS = [
    "vision_projector.0.weight",
    "vision_projector.0.bias",
    "vision_projector.1.weight",
    "vision_projector.1.bias",
    "vision_projector.3.weight",
    "vision_projector.3.bias",
]
UPDATE_EXAMPLE_IDS = [
    "caption:vba2-w013-p0-a",
    "caption:vba2-w025-p0-a",
    "answer:vba2-w000-p0-a",
    "answer:vba2-w036-p0-b",
]
UPDATE_CASE_IDS = [
    "vba2-w013-p0-a",
    "vba2-w025-p0-a",
    "vba2-w000-p0-a",
    "vba2-w036-p0-b",
]
CONFIG_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "hypothesis", "evidence", "prepared", "source", "precondition",
    "schedule", "initialization", "optimizer", "gradient_scope",
    "update_batch", "readouts", "gates", "accounting", "limits", "policy",
    "access", "implementation_reuse", "authorization", "statuses", "claims",
    "limitations", "output", "report_root",
}
WIDER_PER_READOUT = {
    "decoder_forward_calls": 2400,
    "prepare_sense_passes": 294,
    "explicit_logits_forwards": 196,
    "generation_prefill_calls": 98,
    "generation_decode_calls": 1802,
    "generated_tokens": 1900,
    "processed_text_positions": 1000000,
    "processed_visual_positions": 470400,
    "visual_forward_calls": 2400,
    "projector_passes": 2400,
    "visual_observation_calls": 78,
    "visual_encoder_calls": 78,
    "visual_encoder_cache_hits": 78,
    "visual_encoder_cache_misses": 78,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "logical_scoring_cells": 98,
    "logical_scoring_operations": 294,
}
WIDER_CUMULATIVE = {key: 2 * value
                    for key, value in WIDER_PER_READOUT.items()}
UPDATE_BATCH_PER_EVALUATION = {
    "decoder_forward_calls": 8,
    "prepare_sense_passes": 4,
    "explicit_logits_forwards": 4,
    "generation_prefill_calls": 0,
    "generation_decode_calls": 0,
    "generated_tokens": 0,
    "processed_text_positions": 1200,
    "processed_visual_positions": 1568,
    "visual_forward_calls": 8,
    "projector_passes": 8,
    "visual_observation_calls": 4,
    "visual_encoder_calls": 4,
    "visual_encoder_cache_hits": 4,
    "visual_encoder_cache_misses": 4,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "logical_scoring_cells": 0,
    "logical_scoring_operations": 0,
}
UPDATE_BATCH_CUMULATIVE = {key: 2 * value
                           for key, value in
                           UPDATE_BATCH_PER_EVALUATION.items()}
UPDATE_CAPS = {
    "decoder_forward_calls": 8,
    "prepare_sense_passes": 0,
    "explicit_logits_forwards": 0,
    "generation_prefill_calls": 0,
    "generation_decode_calls": 0,
    "generated_tokens": 0,
    "processed_text_positions": 1200,
    "processed_visual_positions": 1568,
    "visual_forward_calls": 8,
    "projector_passes": 8,
    "visual_observation_calls": 4,
    "visual_encoder_calls": 4,
    "visual_encoder_cache_hits": 4,
    "visual_encoder_cache_misses": 4,
    "backward_calls": 4,
    "optimizer_steps": 1,
    "logical_scoring_cells": 0,
    "logical_scoring_operations": 0,
}
CAMPAIGN_CUMULATIVE = {
    key: WIDER_CUMULATIVE[key]
    + UPDATE_BATCH_CUMULATIVE[key]
    + UPDATE_CAPS[key]
    for key in fwd.COUNTER_KEYS
}
EXPECTED_UNITS = {key: fwd.UNIT_BY_COUNTER[key] for key in fwd.COUNTER_KEYS}


class ProtocolError(ValueError):
    """The predeclared smaller-step protocol is invalid or unauthorised."""


class AuthorizationRefused(ProtocolError):
    """Native execution is refused before any model load."""


class ArtifactBudgetError(RuntimeError):
    """The completed run exceeded the declared artifact byte cap."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == keys,
             f"{label} fields differ: {sorted(set(value) ^ keys)}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _require_sha256(value: Any, label: str) -> None:
    _require(isinstance(value, str)
             and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA-256")


def _require_commit(value: Any, label: str = "protocol_commit") -> None:
    _require(isinstance(value, str)
             and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
             f"{label} must be a full 40-character commit SHA")


def _require_pinned_file(root: Path, relative: str, expected: str,
                         verify_files: bool) -> None:
    _require_sha256(expected, relative)
    if not verify_files:
        return
    path = root / relative
    _require(path.is_file(), f"Pinned file is missing: {relative}")
    actual = _sha256_file(path)
    _require(actual == expected,
             f"Pinned file hash differs: {relative}")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Protocol JSON is unreadable: {error}") from error
    _require(isinstance(payload, Mapping), "Protocol must be a JSON object")
    return dict(payload)


def _write_json_document(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _artifact_bytes(output: str | Path) -> int:
    return sum(path.stat().st_size for path in Path(output).rglob("*")
               if path.is_file())


def _enforce_native_environment(environ: Any | None = None) -> dict[str, str]:
    """Force offline/no-fallback values, overriding caller settings."""
    target = os.environ if environ is None else environ
    for key, value in NATIVE_ENVIRONMENT.items():
        target[key] = value
    return {key: target[key] for key in NATIVE_ENVIRONMENT}


def _write_report_with_artifact_budget(
        report: dict[str, Any], output: str | Path, limit: int, *,
        writer=_write_json_document) -> dict[str, Any]:
    """Write the report, then measure and re-write until its bytes count.

    The report file itself is included in ``artifact_bytes``.  The helper never
    deletes artifacts: an overflow remains visible in the report and is handled
    by the caller after the file has been written.
    """
    path = Path(output) / "report.json"
    limit = int(limit)
    report["artifact_limit_bytes"] = limit
    report["artifact_bytes"] = 0
    report["artifact_budget_ok"] = None
    writer(path, report)
    for _ in range(6):
        actual = _artifact_bytes(output)
        ok = actual <= limit
        if (report.get("artifact_bytes") == actual
                and report.get("artifact_budget_ok") == ok):
            break
        report["artifact_bytes"] = actual
        report["artifact_budget_ok"] = ok
        writer(path, report)
    actual = _artifact_bytes(output)
    if report.get("artifact_bytes") != actual:
        report["artifact_bytes"] = actual
        report["artifact_budget_ok"] = actual <= limit
        writer(path, report)
        final = _artifact_bytes(output)
        if report.get("artifact_bytes") != final:
            report["artifact_bytes"] = final
            report["artifact_budget_ok"] = final <= limit
            writer(path, report)
    return report


def _is_v2(config: Mapping[str, Any]) -> bool:
    return config.get("version") == VERSION_V2


def _validate_identity(config: Mapping[str, Any]) -> None:
    if _is_v2(config):
        _require(config["format"] == FORMAT_V2, "Protocol format differs")
        _require(config["name"] == NAME_V2, "Protocol name differs")
    else:
        _require(config["format"] == FORMAT, "Protocol format differs")
        _require(config["version"] == VERSION, "Protocol version differs")
        _require(config["name"] == NAME, "Protocol name differs")
    _require(config["mode"] == MODE, "Protocol mode differs")
    _require(config["declared_at_utc"] == "2026-10-27T00:00:00+00:00",
             "Protocol declaration time differs")
    _require(isinstance(config["purpose"], str)
             and len(config["purpose"].strip()) > 80,
             "Protocol purpose is missing")


def _validate_hypothesis(hypothesis: Mapping[str, Any]) -> None:
    _exact_keys(hypothesis, {
        "statement", "arm", "lr", "completed_lr",
        "single_discriminating_change", "same_parent", "same_seed",
        "same_update_batch_rows", "same_projector_tensors", "same_optimizer",
        "same_gradient_scope", "completed_reproduction_repeated",
        "rank_or_loss_movement_is_not_conversion",
    }, "hypothesis")
    _require(hypothesis["statement"] != "" and hypothesis["arm"] == "smaller_step",
             "Smaller-step hypothesis is missing")
    _require(hypothesis["lr"] == LR,
             "Hypothesis must declare lr 3e-05")
    _require(hypothesis["completed_lr"] == COMPLETED_LR,
             "Hypothesis comparator must be the completed 3e-04 arm")
    _require(hypothesis["single_discriminating_change"]
             == "optimizer lr only: 3e-05 versus completed 3e-04",
             "Only optimizer lr may differ")
    _require(hypothesis["same_parent"] == "checkpoints/minicpm5_source",
             "MiniCPM5 parent differs")
    _require(hypothesis["same_seed"] == 29, "Seed must remain 29")
    for key in ("same_update_batch_rows", "same_projector_tensors",
                "same_gradient_scope"):
        _require(hypothesis[key] is True, f"{key} must remain true")
    _require(hypothesis["same_optimizer"] == "NativeFusedAdamW",
             "Optimizer must remain NativeFusedAdamW")
    _require(hypothesis["completed_reproduction_repeated"] is False,
             "The completed reproduction must not be repeated")
    _require(hypothesis["rank_or_loss_movement_is_not_conversion"] is True,
             "Rank/loss movement must not be treated as conversion")


def _validate_evidence(evidence: Mapping[str, Any], root: Path,
                       verify_files: bool) -> None:
    _exact_keys(evidence, {"completed_arm", "completed_update_batch",
                           "anchor_policy"}, "evidence")
    completed = evidence["completed_arm"]
    _exact_keys(completed, {
        "lr", "config", "config_sha256", "report", "report_sha256",
        "training_trace", "training_trace_sha256", "prefix_0_records",
        "prefix_0_records_sha256", "prefix_1_records",
        "prefix_1_records_sha256",
    }, "evidence.completed_arm")
    _require(completed["lr"] == COMPLETED_LR,
             "Completed arm evidence must be lr 3e-04")
    for relative_key, hash_key in (
        ("config", "config_sha256"),
        ("report", "report_sha256"),
        ("training_trace", "training_trace_sha256"),
        ("prefix_0_records", "prefix_0_records_sha256"),
        ("prefix_1_records", "prefix_1_records_sha256"),
    ):
        _require_pinned_file(root, completed[relative_key],
                             completed[hash_key], verify_files)
    batch = evidence["completed_update_batch"]
    _exact_keys(batch, {
        "example_ids", "case_ids", "pre_mean_response_nll",
        "post_mean_response_nll", "post_mean_response_nll_rounded",
        "delta_post_minus_pre", "answer_correct_cases",
        "answer_correct_denominator", "answer_prefill_gold_cases",
        "caption_semantic_f1", "post_source",
    }, "evidence.completed_update_batch")
    _require(batch["example_ids"] == UPDATE_EXAMPLE_IDS,
             "Completed update-batch example IDs differ")
    _require(batch["case_ids"] == UPDATE_CASE_IDS,
             "Completed update-batch case IDs differ")
    _require(batch["pre_mean_response_nll"] == 10.964820146560669,
             "Completed pre-update mean NLL differs")
    _require(batch["post_mean_response_nll"] == 12.560267925262451,
             "Completed post-update mean NLL differs")
    _require(batch["post_mean_response_nll_rounded"] == 12.5603,
             "Completed rounded post-update mean NLL differs")
    _require(batch["delta_post_minus_pre"] == 1.5954477787017822,
             "Completed update-batch mean NLL delta differs")
    _require(batch["answer_correct_cases"] == 0
             and batch["answer_correct_denominator"] == 14,
             "Completed generated-answer correctness anchor differs")
    _require(batch["answer_prefill_gold_cases"] == 0,
             "Completed prefill-gold anchor differs")
    _require(batch["caption_semantic_f1"] == 0.0,
             "Completed caption semantic F1 anchor differs")
    _require("derived" in str(batch["post_source"]),
             "Post-update anchor must name its derived source")
    _require(isinstance(evidence["anchor_policy"], str)
             and "not pass/fail gates" in evidence["anchor_policy"],
             "Evidence anchors must not become gates")


def _validate_prepared(prepared: Mapping[str, Any]) -> None:
    _exact_keys(prepared, {
        "root", "protocol", "protocol_sha256", "manifest",
        "manifest_sha256", "training", "training_sha256",
        "training_examples", "answer_examples", "caption_examples",
        "answer_pairs", "caption_pair_ids", "read_only_training",
        "forbidden_access",
    }, "prepared")
    _require(prepared["root"] == "data/vision_balanced_alignment_v2_20261026/prepared04",
             "Prepared root differs")
    _require(prepared["protocol"] == prepared["root"] + "/protocol.json",
             "Prepared protocol path differs")
    _require(prepared["manifest"] == prepared["root"] + "/manifest.json",
             "Prepared manifest path differs")
    _require(prepared["training"] == prepared["root"] + "/training/alignment.jsonl",
             "Prepared training path differs")
    _require(prepared["protocol_sha256"]
             == "3bf636f00dce4e73168eb5f4b2c2454560276b32dfb1122b91d58ffcd44c2d77",
             "Prepared protocol hash differs")
    _require(prepared["manifest_sha256"]
             == "12f43b081e147a0c678e2137a321b709eb528099411b46b913735198563e0227",
             "Prepared manifest hash differs")
    _require(prepared["training_sha256"]
             == "be6d62e32e688b7e4fc14ff327ec41df7062747e13d4c4d041cd7d05caa00bd8",
             "Prepared training hash differs")
    for key, value in (("training_examples", 26), ("answer_examples", 14),
                       ("caption_examples", 12), ("answer_pairs", 5),
                       ("caption_pair_ids", 7)):
        _require(prepared[key] == value, f"Prepared count differs: {key}")
    _require(prepared["read_only_training"] is True,
             "Prepared training must be read-only")
    _require(prepared["forbidden_access"]
             == ["evaluation/", "dev", "heldout", "semantic_audit"],
             "Prepared forbidden access differs")


def _validate_source(source: Mapping[str, Any]) -> None:
    _exact_keys(source, {
        "model_path", "model_lock", "model_lock_sha256", "model_revision",
        "vision_source_manifest", "vision_source_manifest_sha256",
        "protocol_authorities",
    }, "source")
    _require(source["model_path"] == "checkpoints/minicpm5_source",
             "MiniCPM5 parent path differs")
    _require(source["model_lock"] == "configs/minicpm5_sealed_lock.local.json",
             "Model lock differs")
    _require(source["model_lock_sha256"]
             == "26dea74b4c543bb000c64ac7509171a399beee611ca0f7b70ebd941da64af792",
             "Model lock hash differs")
    _require(source["model_revision"]
             == "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
             "Model revision differs")
    _require(source["vision_source_manifest"]
             == "checkpoints/siglip2_base_patch16_224_source/source_manifest.json",
             "Vision source manifest differs")
    _require(source["vision_source_manifest_sha256"]
             == "8b2351f7b29f2bd7d881405dd978ae714a26c0170b7213d133a9e0059a199024",
             "Vision source manifest hash differs")
    _require(source["protocol_authorities"] == [
        "model.lock_sha256", "model.revision", "vision_digest", "vision",
        "vision_source", "resource_policy", "limits",
    ], "Protocol authorities differ")


def _validate_precondition(precondition: Mapping[str, Any]) -> None:
    _exact_keys(precondition, {
        "run", "report", "required_decision", "required_mode", "config",
        "config_sha256", "note",
    }, "precondition")
    _require(precondition["required_decision"]
             == "spatial_contribution_observed",
             "Ablation precondition decision differs")
    _require(precondition["required_mode"] == "ablation",
             "Ablation precondition mode differs")
    _require(precondition["config"] == "configs/vision_mean_only_ablation_v1.json",
             "Ablation precondition config differs")
    _require(precondition["config_sha256"]
             == "eb0aca1d860162803235d85f96a2dbdc289f0d49b8e10263ba9807392fac19a1",
             "Ablation precondition hash differs")


def _validate_schedule(schedule: Mapping[str, Any]) -> None:
    _exact_keys(schedule, {
        "source", "source_sha256", "caption_labels", "caption_labels_sha256",
        "update_index", "updates_planned", "batch_size",
        "answer_per_update", "caption_per_update", "unique_rows", "slots",
        "expected_answer_rows", "expected_caption_rows",
        "expected_repeated_rows", "smallest_full_coverage_prefix",
        "completed_batch_reused",
    }, "schedule")
    _require(schedule["source"] == "training/batches.jsonl",
             "Schedule source differs")
    _require(schedule["source_sha256"]
             == "08484a4dbc000488c94c54510fad655c06251598a87294837e24d80cad09433a",
             "Schedule source hash differs")
    _require(schedule["caption_labels"] == "training/train_caption_labels.jsonl",
             "Caption labels path differs")
    _require(schedule["caption_labels_sha256"]
             == "84788b08bd6c3a66f9c241e18db73ca8828c8dd15b62aad83e8d3bdf235b1954",
             "Caption labels hash differs")
    for key, value in (("update_index", 1), ("updates_planned", 1),
                       ("batch_size", 4), ("answer_per_update", 2),
                       ("caption_per_update", 2), ("unique_rows", 26),
                       ("slots", 4), ("expected_answer_rows", 14),
                       ("expected_caption_rows", 12),
                       ("expected_repeated_rows", 2)):
        _require(schedule[key] == value, f"Schedule value differs: {key}")
    _require(schedule["smallest_full_coverage_prefix"] is False,
             "This arm is one update, not a full-coverage prefix")
    _require(schedule["completed_batch_reused"] is True,
             "The completed four-row batch must be reused")


def _validate_initialization(initialization: Mapping[str, Any]) -> None:
    _exact_keys(initialization, {
        "init_seed", "recipe", "warm_start", "forbidden_parent_weights_sha256",
        "projector_tensors", "parameter_count", "expected_init_sha256",
    }, "initialization")
    _require(initialization["init_seed"] == 29,
             "Initialization seed must remain 29")
    _require(initialization["recipe"] == "fresh_seeded_normal_std0.02",
             "Fresh initialization recipe differs")
    _require(initialization["warm_start"] is None,
             "Warm start is forbidden")
    _require(initialization["forbidden_parent_weights_sha256"]
             == ["bdc367424b2afcb2e88dba04b6b29495e3a440b5d850f8ae5bc55dbfa428c3a4"],
             "Forbidden parent weights differ")
    _require(initialization["projector_tensors"] == PROJECTOR_TENSORS,
             "Projector initialization scope differs")
    _require(initialization["parameter_count"] == 11542016,
             "Projector parameter count differs")
    _require(initialization["expected_init_sha256"]
             == "511ba5c7648851424649962467972f0e926303d64873dfce08ecd3de299314b9",
             "Seed-29 initialization anchor differs")


def _validate_optimizer(optimizer: Mapping[str, Any]) -> None:
    _exact_keys(optimizer, {
        "name", "wrapper", "optimizer_class", "lr", "weight_decay", "betas",
        "eps", "max_grad_norm", "grad_accum",
    }, "optimizer")
    _require(optimizer["name"] == "adamw_fused",
             "Optimizer name differs")
    _require(optimizer["wrapper"] == "NativeFusedAdamW",
             "NativeFusedAdamW is required")
    _require(optimizer["optimizer_class"] == "torch.optim.AdamW",
             "Optimizer class differs")
    _require(optimizer["lr"] == LR,
             "Smaller-step optimizer lr must be 3e-05")
    _require(optimizer["weight_decay"] == 0.01,
             "Weight decay differs")
    _require(optimizer["betas"] == [0.9, 0.999],
             "AdamW betas differ")
    _require(optimizer["eps"] == 1e-08,
             "AdamW epsilon differs")
    _require(optimizer["max_grad_norm"] == 1.0,
             "Gradient clip differs")
    _require(optimizer["grad_accum"] == 1,
             "Gradient accumulation differs")


def _validate_gradient_scope(scope: Mapping[str, Any]) -> None:
    _exact_keys(scope, {
        "trainable_tensors", "trainable_parameter_count",
        "optimizer_param_groups", "optimizer_owned_parameters",
        "frozen_components", "backbone_gradient_tensors_allowed",
        "nonprojector_gradient_names_allowed", "gradient_accumulation",
        "update_rows", "answer_rows", "caption_rows", "backward_calls",
        "optimizer_steps", "clip_grad_norm",
        "changed_tensors_must_be_subset_of_trainable",
    }, "gradient_scope")
    _require(scope["trainable_tensors"] == PROJECTOR_TENSORS,
             "Gradient scope differs: exactly six projector tensors are required")
    _require(scope["trainable_parameter_count"] == 11542016,
             "Gradient-scope parameter count differs")
    _require(scope["optimizer_param_groups"] == 1,
             "Optimizer must own exactly one parameter group")
    _require(scope["optimizer_owned_parameters"] == 6,
             "Optimizer must own exactly six parameters")
    _require(scope["frozen_components"] == [
        "sealed MiniCPM5 parent", "frozen SigLIP2 vision encoder",
    ], "Frozen component scope differs")
    _require(scope["backbone_gradient_tensors_allowed"] == 0,
             "Backbone gradients are forbidden")
    _require(scope["nonprojector_gradient_names_allowed"] == [],
             "Non-projector gradients are forbidden")
    _require(scope["gradient_accumulation"] == 1,
             "Gradient accumulation differs")
    _require(scope["update_rows"] == 4,
             "Update batch must contain four rows")
    _require(scope["answer_rows"] == 2 and scope["caption_rows"] == 2,
             "Update role split must remain 2 answer / 2 caption")
    _require(scope["backward_calls"] == 4,
             "Exactly four backward calls are required")
    _require(scope["optimizer_steps"] == 1,
             "Exactly one optimizer step is required")
    _require(scope["clip_grad_norm"] == 1.0,
             "Gradient clip differs")
    _require(scope["changed_tensors_must_be_subset_of_trainable"] is True,
             "Changed tensors must remain within the six projector tensors")


def _validate_update_batch(batch: Mapping[str, Any]) -> None:
    _exact_keys(batch, {
        "source", "update_index", "example_ids", "case_ids", "rows", "roles",
        "pre_evaluation_label", "post_evaluation_label", "evaluation_scope",
        "evaluation_controls", "evaluation_records",
        "separate_from_wider_cohort", "loss_metric",
    }, "update_batch")
    _require(batch["source"] == "training/batches.jsonl",
             "Update-batch source differs")
    _require(batch["update_index"] == 1,
             "Update batch index differs")
    _require(batch["example_ids"] == UPDATE_EXAMPLE_IDS,
             "Update-batch example IDs differ")
    _require(batch["case_ids"] == UPDATE_CASE_IDS,
             "Update-batch case IDs differ")
    _require(batch["rows"] == 4, "Update batch must contain four rows")
    _require(batch["roles"] == {"answer": 2, "caption": 2},
             "Update batch role split differs")
    _require(batch["pre_evaluation_label"] == "update_batch_pre"
             and batch["post_evaluation_label"] == "update_batch_post",
             "Separate pre/post update-batch evaluation labels differ")
    _require(batch["evaluation_scope"] == "teacher_forced_loss_only",
             "Update-batch evaluation must be loss-only")
    _require(batch["evaluation_controls"] == ["correct"],
             "Update-batch evaluation must use correct controls only")
    _require(batch["evaluation_records"] == 4,
             "Update-batch evaluation record count differs")
    _require(batch["separate_from_wider_cohort"] is True,
             "Update-batch evaluations must be separate from the wider cohort")
    _require(batch["loss_metric"] == "mean_response_nll",
             "Update-batch loss metric differs")


def _validate_readouts(readouts: Mapping[str, Any], *,
                       config: Mapping[str, Any]) -> None:
    _exact_keys(readouts, {"format", "sequence", "wider", "update_batch",
                           "expected_measured_work"}, "readouts")
    _require(readouts["format"] == "replica.vision_smaller_step.readout.v1",
             "Readout format differs")
    _require(readouts["sequence"] == [
        "prefix_0", "update_batch_pre", "update_1", "update_batch_post",
        "prefix_1",
    ], "Readout sequence must include separate pre/post update-batch evals")
    wider = readouts["wider"]
    _exact_keys(wider, {
        "labels", "prefixes", "available_per_readout", "records_per_readout",
        "placeholders_per_readout", "controls", "teacher_forced_roles",
        "generation_roles", "max_new_tokens_by_kind", "empty_text_policy",
        "eos_only_not_counted_correct", "whitespace_only_not_counted_correct",
        "logical_scoring_cells_per_readout",
        "logical_scoring_operations_per_readout",
    }, "readouts.wider")
    _require(wider["labels"] == ["prefix_0", "prefix_1"],
             "Wider readout labels differ")
    _require(wider["prefixes"] == [0, 1],
             "Wider readout prefixes differ")
    _require(wider["available_per_readout"] == 98
             and wider["records_per_readout"] == 104
             and wider["placeholders_per_readout"] == 6,
             "Wider readout counts differ")
    _require(wider["controls"]
             == ["correct", "sibling", "shuffled", "noimage"],
             "Wider controls differ")
    _require(wider["teacher_forced_roles"] == ["answer", "caption"],
             "Teacher-forced roles differ")
    _require(wider["generation_roles"] == ["answer", "caption"],
             "Generation roles differ")
    _require(wider["max_new_tokens_by_kind"]
             == {"answer": 8, "caption": 32},
             "Generation token caps differ")
    _require(wider["empty_text_policy"]
             == "separate_eos_only_from_whitespace_only",
             "Empty-text policy differs")
    _require(wider["eos_only_not_counted_correct"] is True
             and wider["whitespace_only_not_counted_correct"] is True,
             "Empty outputs must not count as correct")
    _require(wider["logical_scoring_cells_per_readout"] == 98
             and wider["logical_scoring_operations_per_readout"] == 294,
             "Wider logical scoring counts differ")
    batch = readouts["update_batch"]
    _exact_keys(batch, {
        "labels", "records_per_evaluation", "available_per_evaluation",
        "controls", "teacher_forced_roles", "generation", "metrics",
        "separate_evaluations",
    }, "readouts.update_batch")
    _require(batch["labels"] == ["update_batch_pre", "update_batch_post"],
             "Update-batch readout labels differ")
    _require(batch["records_per_evaluation"] == 4
             and batch["available_per_evaluation"] == 4,
             "Update-batch evaluation counts differ")
    _require(batch["controls"] == ["correct"],
             "Update-batch controls differ")
    _require(batch["teacher_forced_roles"] == ["answer", "caption"],
             "Update-batch teacher-forced roles differ")
    _require(batch["generation"] is False,
             "Update-batch evaluation must not generate")
    _require(batch["metrics"] == ["mean_response_nll",
                                  "per_role_mean_response_nll"],
             "Update-batch metrics differ")
    _require(batch["separate_evaluations"] == 2,
             "Exactly two update-batch evaluations are required")
    expected = readouts["expected_measured_work"]
    _exact_keys(expected, {
        "wider_readouts", "update_batch_evaluations", "updates",
        "readout_stages", "decoder_forward_calls", "prepare_sense_passes",
        "explicit_logits_forwards", "generation_prefill_calls",
        "generation_decode_calls", "generated_tokens",
        "visual_observation_calls", "visual_encoder_calls_max",
        "visual_encoder_cache_misses_max", "visual_forward_calls",
        "projector_passes", "backward_calls", "optimizer_steps",
    }, "readouts.expected_measured_work")
    for key, value in (
        ("wider_readouts", 2), ("update_batch_evaluations", 2),
        ("updates", 1), ("readout_stages", 5),
        ("backward_calls", 4), ("optimizer_steps", 1),
    ):
        _require(expected[key] == value,
                 f"Expected measured work differs: {key}")
    campaign = (acc_v2.expected_measured_work() if _is_v2(config)
                else CAMPAIGN_CUMULATIVE)
    campaign_counts = (
        acc_v2.derive_accounting()["campaign_cumulative"]
        if _is_v2(config) else CAMPAIGN_CUMULATIVE)
    for key in fwd.COUNTER_KEYS:
        if key not in expected:
            continue
        if key == "decoder_forward_calls":
            # v1: 2 * (294 prepares + 196 logits + 1,900 tokens) + 16
            # update-batch evaluation passes + 8 update forwards = 4,804.
            # v2 derives the same value from the schedule.
            target = (campaign_counts["decoder_forward_calls"]
                      if _is_v2(config) else 4804)
        elif _is_v2(config):
            target = campaign_counts[key]
        else:
            target = CAMPAIGN_CUMULATIVE[key]
        _require(expected[key] == target,
                 f"Expected measured work differs: {key}")
    _require(expected["visual_encoder_calls_max"]
             == campaign_counts["visual_encoder_calls"],
             "Expected maximum encoder calls differ")
    _require(expected["visual_encoder_cache_misses_max"]
             == campaign_counts["visual_encoder_cache_misses"],
             "Expected maximum encoder misses differ")


def _validate_gates(gates: Mapping[str, Any]) -> None:
    _exact_keys(gates, {"optimization", "functional_fit", "decision"}, "gates")
    optimization = gates["optimization"]
    _exact_keys(optimization, {
        "evaluated_on", "update_count", "backward_calls", "optimizer_steps",
        "optimizer_owned_tensors", "projector_tensors_only_changed",
        "backbone_gradient_tensors", "finite_losses", "finite_grad_norm",
        "max_update_batch_mean_nll_delta",
        "max_wider_answer_mean_nll_delta",
        "max_wider_caption_mean_nll_delta", "delta_definition",
        "loss_movement_is_not_conversion", "status_on_fail",
    }, "gates.optimization")
    _require(optimization["evaluated_on"]
             == ["update_batch_pre", "update_batch_post", "prefix_0", "prefix_1"],
             "Optimization gate inputs differ")
    for key, value in (("update_count", 1), ("backward_calls", 4),
                       ("optimizer_steps", 1), ("optimizer_owned_tensors", 6),
                       ("backbone_gradient_tensors", 0)):
        _require(optimization[key] == value,
                 f"Optimization gate differs: {key}")
    _require(optimization["projector_tensors_only_changed"] is True,
             "Optimization gate must require projector-only changes")
    _require(optimization["finite_losses"] is True
             and optimization["finite_grad_norm"] is True,
             "Optimization gate must require finite loss and grad norm")
    _require(optimization["max_update_batch_mean_nll_delta"] == 0.25,
             "Update-batch loss-delta gate differs")
    _require(optimization["max_wider_answer_mean_nll_delta"] == 0.25,
             "Wider answer loss-delta gate differs")
    _require(optimization["max_wider_caption_mean_nll_delta"] == 0.25,
             "Wider caption loss-delta gate differs")
    _require(optimization["delta_definition"] == "post_minus_pre",
             "Optimization delta definition differs")
    _require(optimization["loss_movement_is_not_conversion"] is True,
             "Loss movement must not be treated as conversion")
    _require(optimization["status_on_fail"] == "optimization_gate_failed",
             "Optimization failure status differs")
    functional = gates["functional_fit"]
    _exact_keys(functional, {
        "evaluated_on", "answer_correct_min", "answer_correct_denominator",
        "answer_correct_accuracy_min", "answer_prefill_gold_min",
        "answer_prefill_gold_accuracy_min", "caption_token_f1_gain_min",
        "caption_semantic_f1_min",
        "correct_vs_sibling_accuracy_margin_min",
        "correct_vs_shuffled_accuracy_margin_min",
        "correct_vs_noimage_accuracy_margin_min", "eos_only_rate_max",
        "whitespace_only_rate_max", "empty_text_policy",
        "eos_only_not_counted_correct",
        "whitespace_only_not_counted_correct", "answer_correct_metric",
        "caption_quality_metric", "rank_or_loss_movement_is_not_conversion",
        "development_on_pass", "development_on_fail", "status_on_fail",
    }, "gates.functional_fit")
    _require(functional["answer_correct_min"] == 1
             and functional["answer_correct_denominator"] == 14,
             "Functional answer-correct gate differs")
    _require(functional["answer_correct_accuracy_min"]
             == 1 / 14,
             "Functional answer accuracy threshold differs")
    _require(functional["answer_prefill_gold_min"] == 1
             and functional["answer_prefill_gold_accuracy_min"] == 1 / 14,
             "Functional prefill-gold gate differs")
    _require(functional["caption_token_f1_gain_min"] == 0.1,
             "Functional caption token-F1 gate differs")
    _require(functional["caption_semantic_f1_min"] == 0.05,
             "Functional caption semantic-F1 gate differs")
    for key in ("correct_vs_sibling_accuracy_margin_min",
                "correct_vs_shuffled_accuracy_margin_min",
                "correct_vs_noimage_accuracy_margin_min"):
        _require(functional[key] == 0.05,
                 f"Functional control-margin gate differs: {key}")
    _require(functional["eos_only_rate_max"] == 0.5
             and functional["whitespace_only_rate_max"] == 0.1,
             "Functional empty-output gates differ")
    _require(functional["empty_text_policy"]
             == "separate_eos_only_from_whitespace_only",
             "Functional empty-text policy differs")
    _require(functional["eos_only_not_counted_correct"] is True
             and functional["whitespace_only_not_counted_correct"] is True,
             "Functional empty outputs must not count as correct")
    _require(functional["rank_or_loss_movement_is_not_conversion"] is True,
             "Functional gate must reject rank/loss-only conversion")
    _require(functional["development_on_pass"]
             == "requires_new_predeclared_protocol",
             "Functional pass must require a new predeclared protocol")
    _require(functional["development_on_fail"] == "closed",
             "Development must remain closed on functional failure")
    _require(functional["status_on_fail"] == "smaller_step_functional_fit_failed",
             "Functional failure status differs")
    decision = gates["decision"]
    _require(decision["requires_all_gates"] is True,
             "All gates must be required")
    _require(decision["development_authorized_here"] is False,
             "This protocol cannot authorize development")
    _require(decision["development_eligibility"]
             == "functional_fit_gate_passed_only",
             "Development eligibility must require the functional fit gate")
    for key in ("heldout_authorized", "v0_authorized", "promotion",
                "retry_authorized"):
        _require(decision[key] is False, f"Gate must remain closed: {key}")


def _require_caps_exact(caps: Any, expected: Mapping[str, int],
                        label: str) -> None:
    _exact_keys(caps, set(fwd.COUNTER_KEYS), label)
    for key, value in expected.items():
        _require(caps[key] == value, f"{label} cap differs: {key}")


def _validate_accounting(accounting: Mapping[str, Any]) -> None:
    if accounting.get("format") == acc_v2.FORMAT:
        acc_v2.validate_accounting(accounting)
        wider = accounting["wider_readout_template"]
        positive = {key for key, cap in wider["per_readout"].items()
                    if cap > 0}
        declaration = {
            "format": fwd.BUDGET_FORMAT,
            "units": accounting["units"],
            "per_readout": {key: wider["per_readout"][key]
                            for key in positive},
            "cumulative": {key: wider["cumulative"][key]
                           for key in positive},
            "wall_limit_seconds": wider["wall_limit_seconds"],
            "wall_check_every_calls": wider["wall_check_every_calls"],
        }
        try:
            fwd.validate_budget_declaration(
                declaration, available_per_readout=acc_v2.WIDER_CELLS,
                expected_readouts=2)
        except fwd.AccountingError as error:
            raise ProtocolError(str(error)) from error
        _require(accounting["wall_limit_seconds"] == 1200,
                 "Campaign wall budget differs")
        _require(accounting["artifact_bytes"] == 33554432,
                 "Campaign artifact budget differs")
        _require(accounting["wall_check_every_calls"] == 64,
                 "Campaign wall check interval differs")
        return
    _exact_keys(accounting, {
        "format", "units", "wider_readout_template",
        "update_batch_evaluation", "update", "campaign_cumulative",
        "wall_limit_seconds", "artifact_bytes", "wall_check_every_calls",
    }, "accounting")
    _require(accounting["format"]
             == "replica.vision_smaller_step.accounting.v1",
             "Accounting format differs")
    _require(accounting["units"] == EXPECTED_UNITS,
             "Accounting units differ")
    wider = accounting["wider_readout_template"]
    _exact_keys(wider, {
        "per_readout", "cumulative", "wall_limit_seconds",
        "wall_check_every_calls",
    }, "accounting.wider_readout_template")
    positive = {key for key, cap in wider["per_readout"].items() if cap > 0}
    declaration = {
        "format": fwd.BUDGET_FORMAT,
        "units": accounting["units"],
        "per_readout": {key: wider["per_readout"][key] for key in positive},
        "cumulative": {key: wider["cumulative"][key] for key in positive},
        "wall_limit_seconds": wider["wall_limit_seconds"],
        "wall_check_every_calls": wider["wall_check_every_calls"],
    }
    try:
        fwd.validate_budget_declaration(
            declaration, available_per_readout=98, expected_readouts=2)
    except fwd.AccountingError as error:
        raise ProtocolError(str(error)) from error
    _require_caps_exact(wider["per_readout"], WIDER_PER_READOUT,
                        "accounting.wider_readout_template.per_readout")
    _require_caps_exact(wider["cumulative"], WIDER_CUMULATIVE,
                        "accounting.wider_readout_template.cumulative")
    _require(wider["wall_limit_seconds"] == 1200
             and wider["wall_check_every_calls"] == 64,
             "Wider readout wall checks differ")
    batch = accounting["update_batch_evaluation"]
    _exact_keys(batch, {"per_evaluation", "cumulative"},
                "accounting.update_batch_evaluation")
    _require_caps_exact(batch["per_evaluation"],
                        UPDATE_BATCH_PER_EVALUATION,
                        "accounting.update_batch_evaluation.per_evaluation")
    _require_caps_exact(batch["cumulative"], UPDATE_BATCH_CUMULATIVE,
                        "accounting.update_batch_evaluation.cumulative")
    _require_caps_exact(accounting["update"], UPDATE_CAPS,
                        "accounting.update")
    _require_caps_exact(accounting["campaign_cumulative"],
                        CAMPAIGN_CUMULATIVE,
                        "accounting.campaign_cumulative")
    _require(accounting["wall_limit_seconds"] == 1200,
             "Campaign wall budget differs")
    _require(accounting["artifact_bytes"] == 33554432,
             "Campaign artifact budget differs")
    _require(accounting["wall_check_every_calls"] == 64,
             "Campaign wall check interval differs")


def _validate_limits(limits: Mapping[str, Any], *,
                     config: Mapping[str, Any]) -> None:
    _exact_keys(limits, {
        "max_updates", "max_optimizer_steps", "max_update_backward_calls",
        "max_probe_backward_calls", "max_total_backward_calls",
        "max_readouts", "max_stages", "max_scoring_generations",
        "max_teacher_forced_cells", "max_decoder_forwards_per_readout",
        "max_decoder_forwards", "max_prepares_per_readout", "max_prepares",
        "max_explicit_logits_per_readout", "max_explicit_logits",
        "max_generated_tokens_per_readout", "max_generated_tokens",
        "max_visual_encoder_calls", "max_visual_encoder_calls_per_readout",
        "max_projector_passes", "readout_weight_writes", "max_seconds",
        "artifact_bytes", "max_prompt_tokens", "max_sequence_tokens",
        "max_new_tokens", "caption_max_new_tokens", "model_loads",
        "extension_loads", "extension_writes", "checkpoint_writes",
        "downloads", "retries", "workers", "grad_accum",
    }, "limits")
    derived = acc_v2.derive_accounting() if _is_v2(config) else None

    def _campaign(values: Mapping[str, Any] | None, key: str) -> int:
        if values is None:
            return CAMPAIGN_CUMULATIVE[key]
        return int(values["campaign_cumulative"][key])

    def _wider(values: Mapping[str, Any] | None, key: str,
               legacy: int) -> int:
        if values is None:
            return legacy
        return int(values["wider_readout_template"]["per_readout"][key])

    required = {
        "max_updates": 1,
        "max_optimizer_steps": 1,
        "max_update_backward_calls": 4,
        "max_probe_backward_calls": 0,
        "max_total_backward_calls": 4,
        "max_readouts": 4,
        "max_stages": 5,
        "max_scoring_generations": 196,
        "max_teacher_forced_cells": 204,
        "max_decoder_forwards_per_readout":
            _wider(derived, "decoder_forward_calls", 2400),
        "max_decoder_forwards":
            _campaign(derived, "decoder_forward_calls"),
        "max_prepares_per_readout": 294,
        "max_prepares": _campaign(derived, "prepare_sense_passes"),
        "max_explicit_logits_per_readout": 196,
        "max_explicit_logits": _campaign(derived, "explicit_logits_forwards"),
        "max_generated_tokens_per_readout": 1900,
        "max_generated_tokens": _campaign(derived, "generated_tokens"),
        "max_visual_encoder_calls":
            _campaign(derived, "visual_encoder_calls"),
        "max_visual_encoder_calls_per_readout":
            _wider(derived, "visual_encoder_calls", 78),
        "max_projector_passes": _campaign(derived, "projector_passes"),
        "readout_weight_writes": 0,
        "max_seconds": 1200,
        "artifact_bytes": 33554432,
        "max_prompt_tokens": 256,
        "max_sequence_tokens": 288,
        "max_new_tokens": 8,
        "caption_max_new_tokens": 32,
        "model_loads": 1,
        "extension_loads": 0,
        "extension_writes": 0,
        "checkpoint_writes": 0,
        "downloads": 0,
        "retries": 0,
        "workers": 1,
        "grad_accum": 1,
    }
    for key, value in required.items():
        _require(limits[key] == value, f"Limit differs: {key}")


def _validate_policy(policy: Mapping[str, Any]) -> None:
    _exact_keys(policy, {
        "device", "cpu_fallback", "exclusive_mps_worker",
        "capacity_guard_required", "profiler_required", "fresh_output",
        "one_model_load", "no_retry", "no_downloads", "no_extension_save",
        "no_checkpoint_writes", "single_hypothesis", "memory_guard",
    }, "policy")
    _require(policy["device"] == "mps", "Device policy differs")
    _require(policy["cpu_fallback"] is False, "CPU fallback is forbidden")
    _require(policy["exclusive_mps_worker"] is True,
             "An exclusive MPS worker is required")
    _require(policy["capacity_guard_required"] is True,
             "Capacity guard is required")
    _require(policy["profiler_required"] is True,
             "Profiler is required")
    _require(policy["fresh_output"] is True, "Fresh output is required")
    _require(policy["one_model_load"] is True,
             "Exactly one model load is required")
    _require(policy["no_retry"] is True, "Automatic retry is forbidden")
    _require(policy["no_downloads"] is True, "Downloads are forbidden")
    _require(policy["no_extension_save"] is True,
             "Extension writes are forbidden")
    _require(policy["no_checkpoint_writes"] is True,
             "Checkpoint writes are forbidden")
    _require(policy["single_hypothesis"] is True,
             "A single hypothesis is required")
    _require(policy["memory_guard"] == {
        "driver_bytes": 53687091200,
        "swap_growth_bytes": 536870912,
        "host_free_fraction": 0.15,
        "exclusive_worker": True,
        "cpu_fallback": False,
    }, "Memory guard differs")


def _validate_access(access: Mapping[str, Any]) -> None:
    _exact_keys(access, {
        "training_roles", "readout_roles", "training_paths",
        "forbidden_paths", "forbidden_calls", "order", "development",
        "heldout", "semantic_audit", "v0", "sealed_evaluation_access",
        "extension_writes", "checkpoint_writes", "downloads",
    }, "access")
    _require(access["training_roles"] == ["answer", "caption"],
             "Training roles differ")
    _require(access["readout_roles"] == ["train_fit",
                                         "train_update_batch"],
             "Readout roles differ")
    _require(access["training_paths"] == [
        "training/alignment.jsonl",
        "training/batches.jsonl",
        "training/train_caption_labels.jsonl",
    ], "Role access differs: training paths must be prepared04 train only")
    _require(access["forbidden_paths"]
             == ["evaluation/", "dev", "heldout", "semantic_audit"],
             "Forbidden role paths differ")
    _require(access["forbidden_calls"] == [
        "_load_dev_v2", "_load_heldout", "load_v0", "score_sealed",
    ], "Forbidden role calls differ")
    _require(access["order"] == [
        "train", "train_update_batch_pre", "update_1",
        "train_update_batch_post", "train_fit",
    ], "Role access order differs")
    for key in ("development", "heldout", "semantic_audit", "v0"):
        _require(access[key] == "forbidden",
                 f"Role access must remain closed: {key}")
    _require(access["sealed_evaluation_access"] == 0,
             "Sealed evaluation access must remain zero")
    _require(access["extension_writes"] == 0
             and access["checkpoint_writes"] == 0
             and access["downloads"] == 0,
             "Write/download role access must remain zero")
    for path in access["training_paths"]:
        _require(not any(token in path for token in
                         ("evaluation", "dev", "heldout", "semantic_audit")),
                 f"Training role path crosses a sealed role: {path}")


def _validate_implementation_reuse(reuse: Mapping[str, Any], root: Path,
                                  verify_files: bool) -> None:
    _exact_keys(reuse, {"mode", "source_edits", "native_execution", "pins",
                        "note"}, "implementation_reuse")
    _require(reuse["mode"] == "read_only_public_v2_reuse",
             "Implementation reuse mode differs")
    _require(reuse["source_edits"] == 0,
             "Implementation source edits must remain zero")
    _require(reuse["native_execution"] is False,
             "Protocol preparation must not execute native code")
    pins = reuse["pins"]
    _exact_keys(pins, {
        "src/replica_lab/vision_forward_accounting.py",
        "src/replica_lab/vision_learnability_curve_v2.py",
        "src/replica_lab/vision_learnability_curve.py",
        "src/replica_lab/vision_learnability_diagnostic.py",
        "src/replica_lab/vision_curriculum.py",
        "src/replica_lab/vision_balanced_alignment.py",
        "scripts/run_vision_learnability_curve_v2.py",
    }, "implementation_reuse.pins")
    for relative, digest in pins.items():
        _require_pinned_file(root, relative, digest, verify_files)


def _validate_authorization(authorization: Mapping[str, Any], *,
                            config: Mapping[str, Any]) -> None:
    _exact_keys(authorization, {
        "native_execution_authorized", "requires_protocol_commit",
        "protocol_commit", "receipt_path", "receipt_format",
        "required_status", "requires_exact_config_sha256",
        "requires_commit_binding", "required_receipt_fields",
    }, "authorization")
    _require(authorization["native_execution_authorized"] is False,
             "A predeclared protocol must not embed native authorization")
    _require(authorization["requires_protocol_commit"] is True,
             "An external committed protocol is required")
    _require(authorization["protocol_commit"] is None,
             "The predeclared protocol commit must remain null")
    expected_receipt = (f"reports/{config['name']}_20261027/authorization.json")
    _require(authorization["receipt_path"] == expected_receipt,
             "Authorization receipt path differs")
    _require(authorization["receipt_format"] == AUTHORIZATION_FORMAT,
             "Authorization receipt format differs")
    _require(authorization["required_status"] == "authorized",
             "Authorization receipt status differs")
    _require(authorization["requires_exact_config_sha256"] is True,
             "Authorization must bind the exact config SHA-256")
    _require(authorization["requires_commit_binding"] is True,
             "Authorization must bind a committed protocol")
    _require(authorization["required_receipt_fields"] == [
        "format", "status", "protocol_name", "protocol_path",
        "protocol_commit", "config_sha256", "native_execution_authorized",
        "dev_authorized", "heldout_authorized", "v0_authorized",
        "semantic_audit_authorized", "promotion_authorized",
        "retry_authorized", "issued_at_utc",
    ], "Authorization required fields differ")


def _validate_statuses_claims(config: Mapping[str, Any]) -> None:
    _require(config["statuses"] == [
        "smaller_step_functional_fit_passed",
        "smaller_step_functional_fit_failed",
        "optimization_gate_failed",
        "stopped_engineering_failure",
    ], "Protocol statuses differ")
    claims = config["claims"]
    _exact_keys(claims, {
        "hypothesis_scope", "completed_reproduction_repeated",
        "optimizer_scope_changed", "gradient_scope_changed",
        "generated_answer_conversion_claim",
        "rank_improvement_is_conversion", "development_authorized",
        "heldout_authorized", "v0_authorized",
        "semantic_audit_authorized", "promotion", "retry_authorized",
        "native_execution_authorized",
    }, "claims")
    _require(claims["hypothesis_scope"] == "smaller_step_only",
             "Claim scope differs")
    _require(claims["completed_reproduction_repeated"] is False,
             "Completed reproduction must not be claimed as repeated")
    _require(claims["optimizer_scope_changed"] is False
             and claims["gradient_scope_changed"] is False,
             "Optimizer/gradient scope must remain unchanged")
    _require(claims["generated_answer_conversion_claim"] is False
             and claims["rank_improvement_is_conversion"] is False,
             "No conversion claim is predeclared")
    for key in ("development_authorized", "heldout_authorized",
                "v0_authorized", "semantic_audit_authorized", "promotion",
                "retry_authorized", "native_execution_authorized"):
        _require(claims[key] is False, f"Claim must remain false: {key}")
    _require(isinstance(config["limitations"], list)
             and len(config["limitations"]) >= 4,
             "Limitations are required")


def _validate_outputs(config: Mapping[str, Any]) -> None:
    _require(str(config["output"]).startswith("runs/"),
             "Output must be a fresh runs/ path")
    _require(str(config["report_root"]).startswith("reports/"),
             "Report root must be a reports/ path")
    _require(config["output"] != config["report_root"],
             "Output and report root must differ")


SCHEDULE_STEPS = (
    {"stage": "prefix_0", "kind": "wider_readout", "prefix": 0,
     "label": "prefix_0", "role": "train_fit"},
    {"stage": "update_batch_pre", "kind": "update_batch_loss", "prefix": 1,
     "label": "update_batch_pre", "role": "train_update_batch"},
    {"stage": "update_1", "kind": "optimizer_update", "prefix": None,
     "label": "update_1", "role": "train_update_batch"},
    {"stage": "update_batch_post", "kind": "update_batch_loss", "prefix": 2,
     "label": "update_batch_post", "role": "train_update_batch"},
    {"stage": "prefix_1", "kind": "wider_readout", "prefix": 1,
     "label": "prefix_1", "role": "train_fit"},
)


def schedule_steps(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the exact bounded schedule without touching native code."""
    steps = [dict(step) for step in SCHEDULE_STEPS]
    if config is not None:
        _require([step["label"] for step in steps]
                 == config["readouts"]["sequence"],
                 "Schedule steps differ from the declared readout sequence")
        _require(steps[0]["kind"] == "wider_readout"
                 and steps[1]["kind"] == "update_batch_loss"
                 and steps[2]["kind"] == "optimizer_update"
                 and steps[3]["kind"] == "update_batch_loss"
                 and steps[4]["kind"] == "wider_readout",
                 "Schedule must evaluate the update batch before and after "
                 "the single optimizer update")
    return steps


def assert_training_only_schedule(config: Mapping[str, Any]) -> None:
    """Refuse any schedule step that names a development/evaluation role."""
    steps = schedule_steps(config)
    allowed = {"train_fit", "train_update_batch"}
    for step in steps:
        _require(step["role"] in allowed,
                 f"Schedule step crosses a non-training role: {step['label']}")
    access = config["access"]
    for key in ("development", "heldout", "semantic_audit", "v0"):
        _require(access[key] == "forbidden",
                 f"Role access must remain closed: {key}")
    _require(access["sealed_evaluation_access"] == 0,
             "Sealed evaluation access must remain zero")
    for path in access["training_paths"]:
        _require(not any(token in path for token in
                         ("evaluation", "dev", "heldout", "semantic_audit")),
                 f"Training path crosses a sealed role: {path}")


def _budget_declaration(units: Mapping[str, str],
                        per_readout: Mapping[str, int],
                        cumulative: Mapping[str, int],
                        wall_limit_seconds: float,
                        wall_check_every_calls: int) -> dict[str, Any]:
    keys = [key for key in fwd.COUNTER_KEYS
            if int(per_readout.get(key, 0)) > 0
            and int(cumulative.get(key, 0)) > 0]
    return {
        "format": fwd.BUDGET_FORMAT,
        "units": dict(units),
        "per_readout": {key: int(per_readout[key]) for key in keys},
        "cumulative": {key: int(cumulative[key]) for key in keys},
        "wall_limit_seconds": float(wall_limit_seconds),
        "wall_check_every_calls": int(wall_check_every_calls),
    }


class _ScheduleBudget:
    """Stage-exact and campaign-exact call-time enforcement.

    ``ForwardAccounting`` owns the actual native taps.  This adapter keeps the
    wider-readout template, the two update-batch evaluations and the single
    optimizer update on separate per-stage caps, while a campaign budget
    independently enforces the declared whole-run ceilings.
    """

    STAGES = ("wider", "update_batch", "update")

    def __init__(self, config: Mapping[str, Any], *,
                 clock=time.monotonic):
        accounting = config["accounting"]
        units = accounting["units"]
        wall = accounting["wall_limit_seconds"]
        every = accounting["wall_check_every_calls"]
        wider = accounting["wider_readout_template"]
        batch = accounting["update_batch_evaluation"]
        update = accounting["update"]
        campaign = accounting["campaign_cumulative"]
        self.budgets = {
            "wider": fwd.ForwardBudget(_budget_declaration(
                units, wider["per_readout"], wider["cumulative"], wall, every),
                clock=clock),
            "update_batch": fwd.ForwardBudget(_budget_declaration(
                units, batch["per_evaluation"], batch["cumulative"],
                wall, every), clock=clock),
            "update": fwd.ForwardBudget(_budget_declaration(
                units, update, update, wall, every), clock=clock),
        }
        self.campaign = fwd.ForwardBudget(_budget_declaration(
            units, campaign, campaign, wall, every), clock=clock)
        self.stage = "wider"
        self._readout_open = False

    def select(self, stage: str) -> None:
        _require(stage in self.budgets, f"Unknown accounting stage: {stage!r}")
        _require(not self._readout_open or stage == self.stage,
                 "Cannot switch accounting stage inside an open readout")
        self.stage = stage

    def begin_readout(self, prefix: int) -> None:
        _require(not self._readout_open,
                 "A schedule accounting readout is already open")
        self._readout_open = True
        self.budgets[self.stage].begin_readout(prefix)

    def end_readout(self) -> dict[str, int]:
        _require(self._readout_open,
                 "No schedule accounting readout is open")
        result = self.budgets[self.stage].end_readout()
        self._readout_open = False
        return result

    def finish_readout(self) -> None:
        self.budgets[self.stage].finish_readout()

    def check_forward(self, **kwargs: Any) -> None:
        self.budgets[self.stage].check_forward(**kwargs)
        self.campaign.check_forward(**kwargs)

    def commit_forward(self, increments: Mapping[str, int]) -> None:
        self.budgets[self.stage].commit_forward(increments)
        self.campaign.commit_forward(increments)

    def check_event(self, key: str, amount: int = 1) -> None:
        self.budgets[self.stage].check_event(key, amount)
        self.campaign.check_event(key, amount)

    def commit_event(self, key: str, amount: int = 1) -> None:
        self.budgets[self.stage].commit_event(key, amount)
        self.campaign.commit_event(key, amount)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def update_batch_metrics(records: list[Mapping[str, Any]], *, label: str,
                         mean_response_nll: float | None = None,
                         measured: Mapping[str, Any] | None = None
                         ) -> dict[str, Any]:
    losses = [float(record["teacher_forced_loss"]) for record in records]
    role_losses: dict[str, list[float]] = {"answer": [], "caption": []}
    for record in records:
        role_losses[str(record["kind"])].append(
            float(record["teacher_forced_loss"]))
    roles = {role: len(values) for role, values in role_losses.items()}
    mean = (sum(losses) / len(losses) if losses else None)
    if mean_response_nll is not None:
        _require(abs(float(mean_response_nll) - float(mean)) <= 1e-12,
                 "Update-batch mean NLL does not match its row losses")
    return {
        "format": "replica.vision_smaller_step.update_batch_loss.v1",
        "label": label,
        "rows": len(records),
        "roles": roles,
        "case_ids": [str(record["case_id"]) for record in records],
        "example_ids": [str(record["example_id"]) for record in records],
        "losses": losses,
        "mean_response_nll": mean,
        "role_means": {
            role: (sum(values) / len(values) if values else None)
            for role, values in role_losses.items()
        },
        "records": [dict(record) for record in records],
        "measured": dict(measured or {}),
    }


def evaluate_optimization_gate(
        config: Mapping[str, Any], *,
        update: Mapping[str, Any],
        pre: Mapping[str, Any],
        post: Mapping[str, Any],
        prefix_0: Mapping[str, Any],
        prefix_1: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["gates"]["optimization"]
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    check(int(update.get("backward_calls", -1)) == int(gate["backward_calls"]),
          "optimization:backward_calls")
    check(int(update.get("optimizer_steps", -1))
          == int(gate["optimizer_steps"]), "optimization:optimizer_steps")
    changed = set(update.get("changed_projector_tensors") or [])
    check(changed and changed <= set(PROJECTOR_TENSORS),
          "optimization:nonprojector_change")
    check(list(update.get("nonprojector_gradient_names") or []) == [],
          "optimization:nonprojector_gradient")
    check(update.get("base_stamp_unchanged") is True,
          "optimization:base_weight_change")
    check(update.get("response_only_labels") is True,
          "optimization:response_mask")
    losses = [float(value) for value in update.get("losses") or []]
    check(bool(losses) and all(math.isfinite(value) for value in losses),
          "optimization:finite_update_losses")
    check(_finite(update.get("grad_norm")) and float(update["grad_norm"]) >= 0,
          "optimization:finite_grad_norm")
    check(_finite(update.get("loss")), "optimization:finite_mean_loss")

    try:
        pre_mean = float(pre["mean_response_nll"])
        post_mean = float(post["mean_response_nll"])
        update_delta = post_mean - pre_mean
        answer_delta = (float(prefix_1["teacher_forced"]["answer"])
                        - float(prefix_0["teacher_forced"]["answer"]))
        caption_delta = (float(prefix_1["teacher_forced"]["caption"])
                         - float(prefix_0["teacher_forced"]["caption"]))
    except (KeyError, TypeError, ValueError):
        return {
            "ok": False,
            "violations": violations + ["optimization:missing_metrics"],
            "metrics": {},
        }
    check(math.isfinite(update_delta) and update_delta
          <= float(gate["max_update_batch_mean_nll_delta"]),
          "optimization:update_batch_nll_delta")
    check(math.isfinite(answer_delta) and answer_delta
          <= float(gate["max_wider_answer_mean_nll_delta"]),
          "optimization:wider_answer_nll_delta")
    check(math.isfinite(caption_delta) and caption_delta
          <= float(gate["max_wider_caption_mean_nll_delta"]),
          "optimization:wider_caption_nll_delta")
    return {
        "ok": not violations,
        "violations": violations,
        "metrics": {
            "update_batch_pre_mean_response_nll": pre_mean,
            "update_batch_post_mean_response_nll": post_mean,
            "update_batch_delta": update_delta,
            "wider_answer_delta": answer_delta,
            "wider_caption_delta": caption_delta,
            "max_update_batch_mean_nll_delta":
                gate["max_update_batch_mean_nll_delta"],
            "max_wider_answer_mean_nll_delta":
                gate["max_wider_answer_mean_nll_delta"],
            "max_wider_caption_mean_nll_delta":
                gate["max_wider_caption_mean_nll_delta"],
            "loss_movement_is_not_conversion": True,
        },
        "status_on_fail": gate["status_on_fail"],
    }


def evaluate_functional_fit_gate(
        config: Mapping[str, Any], *,
        prefix_0: Mapping[str, Any],
        prefix_1: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["gates"]["functional_fit"]
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    answer = (prefix_1.get("scores") or {}).get("answer") or {}
    caption = (prefix_1.get("scores") or {}).get("caption") or {}
    base_caption = (prefix_0.get("scores") or {}).get("caption") or {}
    summary = prefix_1.get("generation_summary") or {}
    accuracy = float(answer.get("answer_correct_accuracy") or 0.0)
    prefill = float(answer.get("answer_prefill_gold_accuracy") or 0.0)
    token_gain = (float(caption.get("caption_token_f1") or 0.0)
                  - float(base_caption.get("caption_token_f1") or 0.0))
    semantic = float(caption.get("caption_semantic_f1") or 0.0)
    margins = {
        "sibling": float(answer.get("answer_correct_margin_sibling") or 0.0),
        "shuffled": float(answer.get("answer_correct_margin_shuffled") or 0.0),
        "noimage": float(answer.get("answer_correct_margin_noimage") or 0.0),
    }
    cells = int(summary.get("generation_cells") or 0)
    eos_only = int(summary.get("eos_only") or 0)
    whitespace_only = int(summary.get("whitespace_only") or 0)
    eos_rate = (eos_only / cells) if cells else 1.0
    whitespace_rate = (whitespace_only / cells) if cells else 1.0
    check(int(answer.get("answer_correct_cases") or 0)
          == int(gate["answer_correct_denominator"]),
          "functional:answer_denominator")
    check(accuracy >= float(gate["answer_correct_accuracy_min"]),
          "functional:answer_correct")
    check(prefill >= float(gate["answer_prefill_gold_accuracy_min"]),
          "functional:answer_prefill_gold")
    check(token_gain >= float(gate["caption_token_f1_gain_min"]),
          "functional:caption_token_f1_gain")
    check(semantic >= float(gate["caption_semantic_f1_min"]),
          "functional:caption_semantic_f1")
    for control, value in margins.items():
        check(value >= float(
            gate[f"correct_vs_{control}_accuracy_margin_min"]),
            f"functional:correct_vs_{control}")
    check(eos_rate <= float(gate["eos_only_rate_max"]),
          "functional:eos_only_rate")
    check(whitespace_rate <= float(gate["whitespace_only_rate_max"]),
          "functional:whitespace_only_rate")
    check(gate["eos_only_not_counted_correct"] is True
          and gate["whitespace_only_not_counted_correct"] is True,
          "functional:empty_output_counted")
    return {
        "ok": not violations,
        "violations": violations,
        "metrics": {
            "answer_correct_accuracy": accuracy,
            "answer_prefill_gold_accuracy": prefill,
            "caption_token_f1_gain": token_gain,
            "caption_semantic_f1": semantic,
            "correct_vs_sibling_margin": margins["sibling"],
            "correct_vs_shuffled_margin": margins["shuffled"],
            "correct_vs_noimage_margin": margins["noimage"],
            "eos_only_rate": eos_rate,
            "whitespace_only_rate": whitespace_rate,
            "generation_cells": cells,
            "empty_outputs_not_counted_correct": True,
            "rank_or_loss_movement_is_not_conversion": True,
        },
        "status_on_fail": gate["status_on_fail"],
    }


def validate_protocol(config: Mapping[str, Any], *, root: str | Path = ROOT,
                      verify_files: bool = True) -> dict[str, Any]:
    """Validate the smaller-step protocol without loading a native runtime."""
    _require(isinstance(config, Mapping), "Protocol must be a mapping")
    _exact_keys(config, CONFIG_KEYS, "protocol")
    _validate_identity(config)
    _validate_hypothesis(config["hypothesis"])
    _validate_evidence(config["evidence"], Path(root), verify_files)
    _validate_prepared(config["prepared"])
    _validate_source(config["source"])
    _validate_precondition(config["precondition"])
    _validate_schedule(config["schedule"])
    _validate_initialization(config["initialization"])
    _validate_optimizer(config["optimizer"])
    _validate_gradient_scope(config["gradient_scope"])
    _validate_update_batch(config["update_batch"])
    _validate_readouts(config["readouts"], config=config)
    _validate_gates(config["gates"])
    _validate_accounting(config["accounting"])
    _validate_limits(config["limits"], config=config)
    _validate_policy(config["policy"])
    _validate_access(config["access"])
    _validate_implementation_reuse(
        config["implementation_reuse"], Path(root), verify_files)
    _validate_authorization(config["authorization"], config=config)
    _validate_statuses_claims(config)
    _validate_outputs(config)
    return dict(config)


def load_protocol(path: str | Path = DEFAULT_CONFIG, *,
                  root: str | Path = ROOT,
                  verify_files: bool = True) -> dict[str, Any]:
    return validate_protocol(_read_json(path), root=root,
                             verify_files=verify_files)


def protocol_sha256(path: str | Path = DEFAULT_CONFIG) -> str:
    return _sha256_file(path)


def protocol_summary(config: Mapping[str, Any], *,
                     config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return {
        "status": "protocol_valid_preparation_only",
        "native_execution": False,
        "model_loads": 0,
        "config": str(config_path),
        "protocol_sha256": protocol_sha256(config_path),
        "name": config["name"],
        "mode": config["mode"],
        "lr": config["optimizer"]["lr"],
        "completed_lr": config["hypothesis"]["completed_lr"],
        "budgets": {
            "campaign_cumulative":
                config["accounting"]["campaign_cumulative"],
            "wall_limit_seconds": config["accounting"]["wall_limit_seconds"],
            "artifact_bytes": config["accounting"]["artifact_bytes"],
            "model_loads": config["limits"]["model_loads"],
            "downloads": config["limits"]["downloads"],
            "retries": config["limits"]["retries"],
        },
        "gates": {
            "optimization": config["gates"]["optimization"],
            "functional_fit": config["gates"]["functional_fit"],
            "decision": config["gates"]["decision"],
        },
        "authorization": config["authorization"],
    }


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False, capture_output=True)
    except OSError as error:
        raise AuthorizationRefused(
            f"Native execution refused before model load: git is unavailable: "
            f"{error}") from error


def verify_commit_binding(config_path: str | Path, protocol_sha256_value: str,
                          protocol_commit: str, *, root: str | Path = ROOT) -> None:
    """Require the declared commit to contain the exact protocol bytes."""
    root_path = Path(root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root_path / path
    path = path.resolve()
    try:
        relative = path.relative_to(root_path)
    except ValueError as error:
        raise AuthorizationRefused(
            "Native execution refused before model load: protocol is outside "
            "the repository") from error
    _require_commit(protocol_commit)
    _require_sha256(protocol_sha256_value, "config_sha256")
    exists = _git(root_path, "cat-file", "-e",
                  f"{protocol_commit}^{{commit}}")
    if exists.returncode != 0:
        raise AuthorizationRefused(
            "Native execution refused before model load: protocol commit is "
            f"not present: {protocol_commit}")
    blob = _git(root_path, "cat-file", "blob",
                f"{protocol_commit}:{relative.as_posix()}")
    if blob.returncode != 0:
        raise AuthorizationRefused(
            "Native execution refused before model load: protocol path is not "
            f"committed at {protocol_commit}: {relative.as_posix()}")
    actual = _sha256_bytes(blob.stdout)
    _require(actual == protocol_sha256_value,
             "Native execution refused before model load: committed protocol "
             "bytes do not match the authorization receipt")


def verify_implementation_binding(protocol_commit: str, *,
                                  root: str | Path = ROOT) -> dict[str, str]:
    """Require committed runner/dependency bytes to equal the working tree."""
    root_path = Path(root).resolve()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_RELATIVES:
        path = root_path / relative
        if not path.is_file():
            raise AuthorizationRefused(
                f"Implementation file missing before model load: {relative}")
        result = _git(root_path, "cat-file", "blob",
                      f"{protocol_commit}:{relative}")
        if result.returncode != 0:
            raise AuthorizationRefused(
                f"Implementation is not present at protocol commit: {relative}")
        if _sha256_bytes(result.stdout) != _sha256_file(path):
            raise AuthorizationRefused(
                f"Implementation differs from protocol commit: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _validate_receipt(receipt: Mapping[str, Any], *,
                      protocol_sha256_value: str,
                      config: Mapping[str, Any],
                      config_path: str | Path) -> dict[str, Any]:
    _exact_keys(receipt, set(config["authorization"]["required_receipt_fields"]),
                "authorization receipt")
    _require(receipt["format"] == AUTHORIZATION_FORMAT,
             "Authorization receipt format differs")
    _require(receipt["status"] == "authorized",
             "Native execution is not authorized")
    _require(receipt["protocol_name"] == config["name"],
             "Authorization receipt names a different protocol")
    _require(receipt["protocol_path"] == Path(config_path).as_posix(),
             "Authorization receipt binds a different protocol path")
    _require(receipt["config_sha256"] == protocol_sha256_value,
             "Authorization receipt does not bind the exact config")
    _require_commit(receipt["protocol_commit"], "authorization protocol_commit")
    _require(receipt["native_execution_authorized"] is True,
             "Authorization receipt does not authorize native execution")
    for key in ("dev_authorized", "heldout_authorized", "v0_authorized",
                "semantic_audit_authorized", "promotion_authorized",
                "retry_authorized"):
        _require(receipt[key] is False,
                 f"Authorization receipt opens a closed role: {key}")
    _require(isinstance(receipt["issued_at_utc"], str)
             and receipt["issued_at_utc"].strip() != "",
             "Authorization receipt lacks an issue time")
    return dict(receipt)


def _authorize_native(config_path: str | Path,
                      authorization_path: str | Path | None, *,
                      root: str | Path = ROOT,
                      verify_commit: bool = True
                      ) -> tuple[dict[str, Any], str, dict[str, Any], Path]:
    """Validate protocol and authorization before any native import/load."""
    config = load_protocol(config_path, root=root)
    digest = protocol_sha256(config_path)
    if authorization_path is None:
        receipt_path = Path(root) / config["authorization"]["receipt_path"]
    else:
        receipt_path = Path(authorization_path)
        if not receipt_path.is_absolute():
            receipt_path = Path(root) / receipt_path
    if not receipt_path.is_file():
        raise AuthorizationRefused(
            "Native execution refused before model load; authorization "
            f"receipt missing: {receipt_path}")
    receipt = _validate_receipt(
        _read_json(receipt_path), protocol_sha256_value=digest,
        config=config, config_path=config_path)
    if verify_commit:
        verify_commit_binding(config_path, digest, receipt["protocol_commit"],
                              root=root)
        verify_implementation_binding(receipt["protocol_commit"], root=root)
    return config, digest, receipt, receipt_path


def _verify_prepared_inputs(config: Mapping[str, Any],
                            root: str | Path = ROOT) -> None:
    prepared = config["prepared"]
    schedule = config["schedule"]
    pairs = (
        (prepared["protocol"], prepared["protocol_sha256"]),
        (prepared["manifest"], prepared["manifest_sha256"]),
        (prepared["training"], prepared["training_sha256"]),
        (f"{prepared['root']}/{schedule['source']}",
         schedule["source_sha256"]),
        (f"{prepared['root']}/{schedule['caption_labels']}",
         schedule["caption_labels_sha256"]),
    )
    root_path = Path(root)
    for relative, expected in pairs:
        _require_sha256(expected, relative)
        path = root_path / relative
        _require(path.is_file(), f"Prepared input is missing: {relative}")
        _require(_sha256_file(path) == expected,
                 f"Prepared input hash differs: {relative}")


def _verify_run_inputs(config: Mapping[str, Any], *,
                       output: str | Path, model_path: str | Path,
                       model_lock: str | Path,
                       root: str | Path = ROOT) -> None:
    root_path = Path(root)
    _require(Path(model_path).resolve()
             == (root_path / config["source"]["model_path"]).resolve(),
             "CLI model path differs from the pinned MiniCPM5 parent")
    _require(Path(model_lock).resolve()
             == (root_path / config["source"]["model_lock"]).resolve(),
             "CLI model lock differs from the pinned source lock")
    _require(not Path(output).exists(),
             f"Fresh output is required; path exists: {output}")
    assert_training_only_schedule(config)
    _verify_prepared_inputs(config, root=root_path)
    from replica_lab import vision_learnability_diagnostic as learn
    report_path = root_path / config["precondition"]["report"]
    _require(report_path.is_file(),
             "The preserved ablation precondition receipt is missing")
    precondition = learn.ablation_precondition_report(
        _read_json(report_path),
        required_decision=config["precondition"]["required_decision"],
        required_mode=config["precondition"]["required_mode"],
        config_sha256=config["precondition"]["config_sha256"])
    _require(precondition["ok"],
             f"Smaller-step run is refused: {precondition['violations']}")


def _normalise_readout(config: Mapping[str, Any],
                       readout: dict[str, Any]) -> dict[str, Any]:
    readout["format"] = config["readouts"]["format"]
    readout["mode"] = config["mode"]
    readout["role"] = "wider_training_cohort"
    return readout


def _measure_update_batch_loss(
        *, adapter: Any, session: Any, tokenizer: Any, service: Any,
        protocol: Mapping[str, Any], scope: Any,
        batch: list[Mapping[str, Any]], labels: Mapping[str, Any],
        limits: Mapping[str, Any], accounting: Any, guard: Any,
        started: float, output: str | Path, prefix: int, label: str,
        config: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher-forced loss on the four update rows, without generation."""
    from replica_lab.vision_balanced_alignment import _teacher_forced_loss
    from run_vision_curriculum import observe
    import run_vision_learnability_curve_v2 as runner_v2

    records: list[dict[str, Any]] = []
    adapter.eval()
    accounting.begin_readout(prefix)
    snapshot: dict[str, Any] | None = None
    try:
        for position, row in enumerate(batch, 1):
            runner_v2._check_wall(started, limits["max_seconds"], label)
            if guard is not None and position % 4 == 0:
                guard.sample(index=position, event=label)
            visual = observe(
                service, image_path=row["image_path"],
                image_sha256=row["image_sha256"],
                case_id=row["case_id"], scope=scope,
                ledger_epoch=session.ledger.epoch, spec=service.spec,
                source_id=protocol["vision_source"]["source_id"])
            target_text = (str(row["response"]) if row["kind"] == "answer"
                           else str(labels[row["case_id"]]["answer"]))
            loss = _teacher_forced_loss(
                session, tokenizer, protocol, scope, visual, row,
                row["question"], target_text)
            records.append({
                "format": "replica.vision_smaller_step.update_batch_row.v1",
                "label": label,
                "case_id": str(row["case_id"]),
                "example_id": str(row["example_id"]),
                "kind": str(row["kind"]),
                "target_text": target_text,
                "teacher_forced_loss": float(loss),
                "available": True,
            })
    finally:
        try:
            snapshot = accounting.end_readout()
        finally:
            adapter.train()
    result = update_batch_metrics(
        records, label=label, measured=snapshot)
    result["format"] = "replica.vision_smaller_step.update_batch_loss.v1"
    directory = runner_v2._readout_directory(output, label)
    runner_v2._write_jsonl(directory / "records.jsonl", records)
    runner_v2._write_json(directory / "loss.json", result)
    return result


def _validate_schedule_accounting(
        config: Mapping[str, Any], totals: Mapping[str, Any],
        update: Mapping[str, Any], pre: Mapping[str, Any],
        post: Mapping[str, Any], readouts: list[Mapping[str, Any]]) -> dict[str, Any]:
    counters = dict(totals["counters"])
    from replica_lab import vision_learnability_diagnostic as learn
    expected = config["readouts"]["expected_measured_work"]
    for key in ("prepare_sense_passes", "explicit_logits_forwards",
                "backward_calls", "optimizer_steps"):
        learn.require(counters[key] == expected[key],
                      f"Schedule accounting differs: {key} "
                      f"{counters[key]} != {expected[key]}")
    learn.require(counters["logical_scoring_cells"]
                  == config["accounting"]["campaign_cumulative"]
                  ["logical_scoring_cells"],
                  "Schedule logical cell count differs")
    learn.require(counters["logical_scoring_operations"]
                  == config["accounting"]["campaign_cumulative"]
                  ["logical_scoring_operations"],
                  "Schedule logical operation count differs")
    learn.require(counters["visual_encoder_calls"]
                  == counters["visual_encoder_cache_misses"],
                  "Visual encoder calls must equal cache misses")
    learn.require(counters["visual_observation_calls"]
                  == (counters["visual_encoder_cache_hits"]
                      + counters["visual_encoder_cache_misses"]),
                  "Visual observations must equal cache hits plus misses")
    learn.require(counters["processed_visual_positions"]
                  == 196 * counters["visual_forward_calls"],
                  "Visual positions must equal 196 per visual-bearing pass")
    learn.require(counters["projector_passes"]
                  == counters["visual_forward_calls"],
                  "Projector passes must equal visual-bearing passes")
    learn.require(counters["generated_tokens"]
                  == (counters["generation_prefill_calls"]
                      + counters["generation_decode_calls"]),
                  "Generated tokens must equal prefill plus decode calls")
    learn.require(counters["decoder_forward_calls"]
                  == (counters["prepare_sense_passes"]
                      + counters["explicit_logits_forwards"]
                      + counters["generated_tokens"] + 8),
                  "Decoder passes must equal prepares, logits, generations "
                  "and the eight update forwards")
    for key, cap in config["accounting"]["campaign_cumulative"].items():
        learn.require(counters[key] <= int(cap),
                      f"Campaign cap exceeded: {key} {counters[key]} > {cap}")

    for name, evaluation in (("pre", pre), ("post", post)):
        measured = dict(evaluation["measured"]["counters"])
        learn.require(measured["decoder_forward_calls"] == 8,
                      f"{name} update-batch decoder count differs")
        learn.require(measured["prepare_sense_passes"] == 4,
                      f"{name} update-batch prepare count differs")
        learn.require(measured["explicit_logits_forwards"] == 4,
                      f"{name} update-batch logits count differs")
        learn.require(measured["visual_observation_calls"] == 4,
                      f"{name} update-batch observation count differs")
        learn.require(measured["generated_tokens"] == 0
                      and measured["generation_prefill_calls"] == 0
                      and measured["generation_decode_calls"] == 0,
                      f"{name} update-batch evaluation generated text")
        learn.require(measured["backward_calls"] == 0
                      and measured["optimizer_steps"] == 0,
                      f"{name} update-batch evaluation changed weights")
    learn.require(update["backward_calls"] == 4,
                  "The single update did not make four backward calls")
    learn.require(update["optimizer_steps"] == 1,
                  "The single update did not make one optimizer step")
    changed = set(update["changed_projector_tensors"])
    learn.require(changed and changed <= set(PROJECTOR_TENSORS),
                  "The update changed a non-projector tensor")
    learn.require(list(update["nonprojector_gradient_names"]) == [],
                  "The update produced non-projector gradients")
    learn.require(update["base_stamp_unchanged"] is True,
                  "The update changed the sealed base")
    learn.require(update["response_only_labels"] is True,
                  "The update did not use response-only labels")
    learn.require(len(readouts) == 2
                  and [readout["label"] for readout in readouts]
                  == ["prefix_0", "prefix_1"],
                  "Wider-cohort readouts differ")
    return {
        "ok": True,
        "counters": counters,
        "expected": expected,
        "update_batch_pre": pre["measured"]["counters"],
        "update_batch_post": post["measured"]["counters"],
        "update_forwards": 8,
        "arithmetic": "decoder = prepares + logits + generated tokens + 8 "
                      "update forwards",
    }


def _finish_run(config: Mapping[str, Any], config_path: str | Path,
                receipt: Mapping[str, Any], *, output: str | Path,
                started: float, guard: Any, profiler: Any,
                accounting_loads: Any, accounting_totals: Mapping[str, Any],
                measured_work: Mapping[str, Any],
                readouts: list[dict[str, Any]],
                update: Mapping[str, Any],
                pre: Mapping[str, Any], post: Mapping[str, Any],
                optimization: Mapping[str, Any],
                functional: Mapping[str, Any],
                init_hash: str, stop_source: str) -> dict[str, Any]:
    from replica_lab import vision_learnability_diagnostic as learn
    import run_vision_learnability_curve_v2 as runner_v2

    profiler.mark("report_ready")
    profiler_summary = profiler.stop()
    if optimization["ok"] and functional["ok"]:
        status = "smaller_step_functional_fit_passed"
    elif not optimization["ok"]:
        status = optimization["status_on_fail"]
    else:
        status = functional["status_on_fail"]
    wall_seconds = time.monotonic() - started
    learn.require(wall_seconds <= float(config["limits"]["max_seconds"]),
                  f"Wall budget exceeded: {wall_seconds}")
    load_counts = accounting_loads.counts()
    learn.require(load_counts == {"model_loads": 1,
                                  "extension_loads": 0},
                  f"Load counts differ: {load_counts}")
    report = {
        "format": "replica.vision_smaller_step.report.v1",
        "version": VERSION,
        "mode": MODE,
        "name": config["name"],
        "status": status,
        "declared_at_utc": config["declared_at_utc"],
        "generated_at_utc": runner_v2._now(),
        "command": [sys.executable, *sys.argv],
        "config_sha256": protocol_sha256(config_path),
        "protocol_commit": receipt["protocol_commit"],
        "pins": {
            "prepared": config["prepared"],
            "source": config["source"],
            "implementation_reuse": config["implementation_reuse"],
        },
        "initialization": {
            "init_seed": config["initialization"]["init_seed"],
            "projector_init_sha256": init_hash,
            "expected_sha256": config["initialization"]["expected_init_sha256"],
            "matches_anchor": init_hash
            == config["initialization"]["expected_init_sha256"],
        },
        "optimizer": {
            "name": "NativeFusedAdamW",
            "lr": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "tensors": 6,
            "parameters": 11542016,
            "optimizer_steps": 1,
        },
        "gradient_scope": config["gradient_scope"],
        "generation_stop_set": {"stop_tokens": [1, 130073],
                                "source": stop_source},
        "readouts": readouts,
        "stage_sequence": [step["label"] for step in schedule_steps(config)],
        "readout_sequence": [readout["label"] for readout in readouts],
        "update_batch": {
            "pre": pre,
            "post": post,
            "delta_post_minus_pre": (
                float(post["mean_response_nll"])
                - float(pre["mean_response_nll"])),
            "separate_evaluations": True,
        },
        "training": {
            "updates": [dict(update)],
            "total_updates": 1,
            "update_backward_calls": 4,
            "optimizer_steps": 1,
        },
        "accounting": accounting_totals,
        "measured_work": measured_work,
        "gates": {
            "optimization": optimization,
            "functional_fit": functional,
            "decision": {
                "status": status,
                "development_authorized": False,
                "development_eligibility":
                    config["gates"]["decision"]["development_eligibility"]
                    if (optimization["ok"] and functional["ok"])
                    else "closed",
                "heldout_authorized": False,
                "v0_authorized": False,
                "promotion": False,
                "retry_authorized": False,
            },
        },
        "decision": {
            "status": status,
            "development_authorized": False,
            "development_gate": config["gates"]["functional_fit"]
            ["development_on_pass"],
            "heldout_authorized": False,
            "v0_authorized": False,
            "promotion": False,
            "retry_authorized": False,
        },
        "load_counts": load_counts,
        "artifact_bytes": 0,
        "artifact_limit_bytes": config["limits"]["artifact_bytes"],
        "artifact_budget_ok": None,
        "wall_seconds": wall_seconds,
        "wall_limit_seconds": config["limits"]["max_seconds"],
        "guard": guard.report(),
        "profiler": profiler_summary,
        "claims": {
            **config["claims"],
            "native_execution_authorized":
                bool(receipt["native_execution_authorized"]),
            "native_execution_performed": True,
            "development_authorized": False,
            "heldout_authorized": False,
            "v0_authorized": False,
            "semantic_audit_authorized": False,
            "promotion": False,
            "retry_authorized": False,
        },
        "limitations": config["limitations"],
    }
    _write_report_with_artifact_budget(
        report, output, int(config["limits"]["artifact_bytes"]))
    if not report["artifact_budget_ok"]:
        _write_json_document(Path(output) / "artifact_budget_failure.json", {
            "format": "replica.vision_smaller_step.artifact_budget_failure.v1",
            "artifact_bytes": report["artifact_bytes"],
            "artifact_limit_bytes": report["artifact_limit_bytes"],
            "report_preserved": True,
            "artifacts_deleted": False,
            "limit_includes_report": True,
        })
        report["status"] = "stopped_engineering_failure"
        report["gates"]["decision"]["status"] = "stopped_engineering_failure"
        report["decision"]["status"] = "stopped_engineering_failure"
        report["artifact_budget_exceeded"] = True
        _write_report_with_artifact_budget(
            report, output, int(config["limits"]["artifact_bytes"]))
        raise ArtifactBudgetError(
            f"Artifact budget exceeded including the final report: "
            f"{report['artifact_bytes']} > {report['artifact_limit_bytes']}")
    return report


def _execute_native_schedule(config: Mapping[str, Any],
                             config_path: str | Path,
                             receipt: Mapping[str, Any], *,
                             output: str | Path, model_path: str | Path,
                             model_lock: str | Path,
                             root: str | Path = ROOT) -> dict[str, Any]:
    """Execute the authorized schedule.  Imports native code only here."""
    _enforce_native_environment()
    import torch  # noqa: F401 - native runtime, imported after authorization
    from replica_lab import vision_learnability_curve_v2 as curve_v2
    from replica_lab import vision_learnability_diagnostic as learn
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.optimizers import NativeFusedAdamW
    from replica_lab.optimizers import build_optimizer as build_native_optimizer
    from replica_lab.profiling import RunProfiler
    from replica_lab.vision_balanced_alignment import (
        scorer_contract_preflight_v2)
    from replica_lab.vision_collapse_diagnostic import (
        EXPECTED_STOP_TOKENS, control_plan, resolve_stop_tokens)
    from run_vision_curriculum import (
        Guard, feature_service, freeze_projector_only, initialize_projector,
        make_session, protocol_scope, verify_source_and_spec)
    import run_vision_balanced_alignment as balanced
    import run_vision_learnability_curve_v2 as runner_v2
    import run_vision_learnability_diagnostic as base

    output_path = Path(output)
    learn.require(not output_path.exists(),
                  f"Fresh output is required; path exists: {output_path}")
    output_path.mkdir(parents=True)
    root_path = Path(root)
    protocol_path = root_path / config["prepared"]["protocol"]
    protocol = runner_v2._read_json(protocol_path)
    runner_v2._write_json(output_path / "config.json", dict(config))
    (output_path / "protocol.json").write_bytes(protocol_path.read_bytes())
    started = time.monotonic()
    guard = None
    profiler = None
    profiler_stopped = False
    accounting_loads = runner_v2.LoadAccounting()
    trace: list[dict[str, Any]] = []
    readouts: list[dict[str, Any]] = []
    try:
        learn.require(Path(model_path).resolve()
                      == (root_path / config["source"]["model_path"]).resolve(),
                      "CLI model path differs from the pinned source")
        learn.require(Path(model_lock).resolve()
                      == (root_path / config["source"]["model_lock"]).resolve(),
                      "CLI model lock differs from the pinned lock")
        learn.require(tuple(config["readouts"]["sequence"])
                      == tuple(step["label"] for step in schedule_steps(config)),
                      "Schedule sequence differs")
        training = balanced._load_training_v2(protocol)
        batch = list(training["batches"][0])
        learn.require([str(row["example_id"]) for row in batch]
                      == config["update_batch"]["example_ids"],
                      "The first four-row update batch differs")
        learn.require([row["kind"] for row in batch].count("answer") == 2
                      and [row["kind"] for row in batch].count("caption") == 2,
                      "The update batch role mix differs")
        plan = control_plan(training["training"])
        budget = _ScheduleBudget(config)
        accounting = fwd.ForwardAccounting(budget=budget)
        guard = Guard(protocol)
        with exclusive_mps_worker(root_path):
            profiler = RunProfiler("mps", output_path)
            profiler.start()
            try:
                _, spec, _ = verify_source_and_spec(protocol, "mps")
                adapter, tokenizer, _ = runner_v2._accounted_load(
                    lambda: load_local_mps(
                        model_path, model_lock, vision=spec,
                        profiler=profiler), accounting_loads, "model")
                guard.sample(index=0, event="model_loaded")
                profiler.mark("model_loaded")
                stops, stop_source = resolve_stop_tokens(
                    getattr(adapter.base.config, "eos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None))
                learn.require(tuple(stops) == tuple(EXPECTED_STOP_TOKENS),
                              f"Model stop set {stops} differs")
                learn.require(int(tokenizer.eos_token_id) == 1,
                              "Tokenizer EOS differs from the declared target")
                adapter.eval()
                freeze_projector_only(adapter)
                adapter._initial_base_stamp = adapter.parameter_stamp(
                    original_only=True)
                init_hash = initialize_projector(
                    adapter.vision_projector,
                    int(config["initialization"]["init_seed"]))
                learn.require(init_hash
                              == config["initialization"]["expected_init_sha256"],
                              "Fresh seed-29 projector differs from the anchor")
                session = make_session(adapter, tokenizer, output_path)
                service = feature_service(protocol, "mps")
                guard.sample(index=0, event="feature_service")
                scorer_contract = scorer_contract_preflight_v2(
                    protocol["evaluation"])
                runner_v2._write_json(
                    output_path / "scores/scorer_contracts.json",
                    scorer_contract)
                optimizer = build_native_optimizer(
                    adapter, "adamw_fused", lr_base=0.0,
                    lr_new=float(config["optimizer"]["lr"]),
                    weight_decay=float(config["optimizer"]["weight_decay"]))
                learn.require(isinstance(optimizer, NativeFusedAdamW),
                              "NativeFusedAdamW is required")
                learn.require(len(optimizer.param_groups) == 1
                              and len(optimizer.param_groups[0]["params"]) == 6,
                              "Optimizer ownership differs")
                scope = protocol_scope(protocol)
                labels = training["train_caption_labels"]
                caption_labels = {str(row["case_id"]): row for row in labels}
                with accounting.wrap_native(adapter), \
                        accounting.wrap_feature_service(service), \
                        accounting.measured_session(session):
                    budget.select("wider")
                    readout_0 = _normalise_readout(config,
                                                   runner_v2._measured_readout(
                        adapter, session, tokenizer, service, protocol, scope,
                        training["training"], plan, labels, stops,
                        prefix=0, label="prefix_0",
                        expected_prefixes=[0, 1], prefix_position=0,
                        output=output_path, guard=guard, started=started,
                        limits=config["limits"], optimizer=optimizer,
                        accounting=accounting))
                    runner_v2._write_json(
                        runner_v2._readout_directory(output_path, "prefix_0")
                        / "readout.json", readout_0)
                    readouts.append(readout_0)

                    budget.select("update_batch")
                    pre = _measure_update_batch_loss(
                        adapter=adapter, session=session, tokenizer=tokenizer,
                        service=service, protocol=protocol, scope=scope,
                        batch=batch, labels=caption_labels,
                        limits=config["limits"],
                        accounting=accounting, guard=guard, started=started,
                        output=output_path, prefix=1,
                        label="update_batch_pre", config=config)

                    budget.select("update")
                    runner_v2._check_wall(started,
                                          config["limits"]["max_seconds"],
                                          "smaller-step update")
                    with accounting.forced_phase(fwd.PHASE_TRAINING):
                        with accounting.counted_optimizer_step(optimizer):
                            update, calls = base._gradient_update(
                                adapter, session, tokenizer, service, protocol,
                                scope, batch, optimizer, config["limits"],
                                stops, index=1, guard=guard,
                                source_id=protocol["vision_source"]["source_id"])
                    learn.require(calls == 4
                                  and update["backward_calls"] == 4,
                                  "The single update did not make four "
                                  "backward calls")
                    update["optimizer_steps"] = 1
                    learn.require(set(update["changed_projector_tensors"])
                                  <= set(PROJECTOR_TENSORS),
                                  "The update changed a non-projector tensor")
                    trace.append(update)
                    runner_v2._write_jsonl(
                        output_path / "training_trace.jsonl", trace)
                    mask_audit = curve_v2.validate_response_masks(
                        update.get("fit_contract_rows") or [])
                    learn.require(mask_audit["ok"],
                                  f"Update mask audit failed: "
                                  f"{mask_audit['violations']}")
                    runner_v2._write_json(
                        output_path / "response_mask_audit.json", mask_audit)
                    learn.require_update_role_mix(update)

                    budget.select("update_batch")
                    post = _measure_update_batch_loss(
                        adapter=adapter, session=session, tokenizer=tokenizer,
                        service=service, protocol=protocol, scope=scope,
                        batch=batch, labels=caption_labels,
                        limits=config["limits"],
                        accounting=accounting, guard=guard, started=started,
                        output=output_path, prefix=2,
                        label="update_batch_post", config=config)

                    budget.select("wider")
                    readout_1 = _normalise_readout(config,
                                                   runner_v2._measured_readout(
                        adapter, session, tokenizer, service, protocol, scope,
                        training["training"], plan, labels, stops,
                        prefix=1, label="prefix_1",
                        expected_prefixes=[0, 1], prefix_position=1,
                        output=output_path, guard=guard, started=started,
                        limits=config["limits"], optimizer=optimizer,
                        accounting=accounting))
                    runner_v2._write_json(
                        runner_v2._readout_directory(output_path, "prefix_1")
                        / "readout.json", readout_1)
                    readouts.append(readout_1)

                    totals = accounting.historical_totals()
                    measured_work = _validate_schedule_accounting(
                        config, totals, update, pre, post, readouts)
                    runner_v2._write_json(
                        output_path / "accounting.json", totals)
                optimization = evaluate_optimization_gate(
                    config, update=update, pre=pre, post=post,
                    prefix_0=readout_0, prefix_1=readout_1)
                functional = evaluate_functional_fit_gate(
                    config, prefix_0=readout_0, prefix_1=readout_1)
                profiler_stopped = True
                return _finish_run(
                    config, config_path, receipt, output=output_path,
                    started=started, guard=guard, profiler=profiler,
                    accounting_loads=accounting_loads,
                    accounting_totals=totals, measured_work=measured_work,
                    readouts=readouts, update=update, pre=pre, post=post,
                    optimization=optimization, functional=functional,
                    init_hash=init_hash, stop_source=stop_source)
            finally:
                if profiler is not None and not profiler_stopped:
                    try:
                        profiler.stop()
                    except Exception:  # noqa: BLE001
                        pass
    except Exception as exc:
        failure = {
            "format": "replica.vision_smaller_step.report.v1.failure",
            "generated_at_utc": runner_v2._now(),
            "status": "stopped_engineering_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "native_execution_attempted": True,
            "development_authorized": False,
            "heldout_authorized": False,
            "v0_authorized": False,
            "promotion": False,
            "retry_authorized": False,
        }
        try:
            runner_v2._write_json(output_path / "failure_report.json", failure)
        except Exception:  # noqa: BLE001 - never mask the original failure
            pass
        raise


def run_protocol(config_path: str | Path = DEFAULT_CONFIG, *,
                 output: str | Path, model_path: str | Path,
                 model_lock: str | Path,
                 authorization_path: str | Path | None = None,
                 root: str | Path = ROOT, device: str = "mps") -> dict[str, Any]:
    """Authorized native entrypoint. Authorization precedes native imports."""
    if device != "mps":
        raise ProtocolError("Native vision execution requires device=mps")
    config, digest, receipt, receipt_path = _authorize_native(
        config_path, authorization_path, root=root)
    try:
        _verify_run_inputs(config, output=output, model_path=model_path,
                           model_lock=model_lock, root=root)
        assert_training_only_schedule(config)
    except ProtocolError:
        raise
    except Exception as error:  # noqa: BLE001 - pre-flight refusal
        raise ProtocolError(f"Run pre-flight refused before model load: "
                            f"{error}") from error
    _enforce_native_environment()
    return _execute_native_schedule(
        config, config_path, receipt, output=output,
        model_path=model_path, model_lock=model_lock, root=root)


def native_launch_command(config: Mapping[str, Any], config_path: str | Path,
                          receipt_path: str | Path) -> list[str]:
    """Return the exact capacity-guarded native command; do not execute it."""
    return [
        "python", "scripts/run_capacity_preflight.py",
        "--entry-point", "vision_smaller_step_v2.py",
        "--report", f"{config['report_root']}/guard_smaller_step01.json",
        "--", "run",
        "--config", str(config_path),
        "--output", config["output"],
        "--model-path", config["source"]["model_path"],
        "--model-lock", config["source"]["model_lock"],
        "--authorization", str(receipt_path),
        "--device", "mps",
    ]


def review_launch(config_path: str | Path = DEFAULT_CONFIG, *,
                  authorization_path: str | Path | None = None,
                  root: str | Path = ROOT,
                  verify_commit: bool = True) -> dict[str, Any]:
    """Review-only launch gate; never loads or runs native code."""
    config, digest, receipt, receipt_path = _authorize_native(
        config_path, authorization_path, root=root,
        verify_commit=verify_commit)
    output_path = Path(root) / config["output"]
    if output_path.exists():
        raise AuthorizationRefused(
            "Native execution refused before model load; fresh output is "
            f"required: {config['output']}")
    launch_command = native_launch_command(config, config_path, receipt_path)
    return {
        "format": REVIEW_FORMAT,
        "status": "review_only_no_native_execution",
        "native_execution": False,
        "model_loads": 0,
        "config": str(config_path),
        "protocol_sha256": digest,
        "protocol_commit": receipt["protocol_commit"],
        "implementation_bound": verify_commit,
        "implementation_sha256": {
            relative: _sha256_file(Path(root) / relative)
            for relative in IMPLEMENTATION_RELATIVES
        },
        "protocol_name": config["name"],
        "authorization": {
            "status": receipt["status"],
            "native_execution_authorized":
                receipt["native_execution_authorized"],
            "protocol_commit": receipt["protocol_commit"],
            "config_sha256": receipt["config_sha256"],
        },
        "budgets": {
            "campaign_cumulative":
                config["accounting"]["campaign_cumulative"],
            "wall_limit_seconds": config["accounting"]["wall_limit_seconds"],
            "artifact_bytes": config["accounting"]["artifact_bytes"],
        },
        "gates": {
            "optimization": config["gates"]["optimization"],
            "functional_fit": config["gates"]["functional_fit"],
            "decision": config["gates"]["decision"],
        },
        "review_command": [
            "python", "scripts/vision_smaller_step_v2.py",
            "review-launch", "--config", str(config_path),
            "--authorization", str(receipt_path),
        ],
        "launch_command": launch_command,
        "run_command": launch_command,
        "handoff_command": launch_command,
        "native_environment": dict(NATIVE_ENVIRONMENT),
        "native_environment_enforced_on_run": True,
        "environment_override_policy":
            "forced before native imports; caller values are replaced",
        "native_runner_implemented": True,
        "native_execution_performed_here": False,
        "note": "Review only. The real authorized native entrypoint is the launch_command; this review loads no model, executes no native command, and opens no development, held-out, semantic-audit, V0, promotion or retry access.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and review the predeclared vision smaller-step "
                    "protocol without native execution")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate", help="deterministically validate the CPU protocol")
    validate.add_argument("--config", default=DEFAULT_CONFIG)
    validate.add_argument("--json", action="store_true",
                          help="emit the full validation summary")
    hash_parser = sub.add_parser("hash", help="print the exact config SHA-256")
    hash_parser.add_argument("--config", default=DEFAULT_CONFIG)
    review = sub.add_parser(
        "review-launch",
        aliases=["review"],
        help="refuse or review a launch; never loads a model")
    review.add_argument("--config", default=DEFAULT_CONFIG)
    review.add_argument("--authorization", default=None,
                        help="external authorization receipt bound to the "
                             "committed protocol")
    run = sub.add_parser(
        "run",
        help="authorized native execution of the exact five-stage schedule")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--output", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--authorization", default=None,
                     help="external authorization receipt bound to the "
                          "committed protocol")
    run.add_argument("--device", choices=["mps"], default="mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            print(protocol_sha256(args.config))
            return 0
        if args.command == "validate":
            config = load_protocol(args.config)
            summary = protocol_summary(config, config_path=args.config)
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(f"protocol_valid_preparation_only {summary['protocol_sha256']}")
                print(f"name={summary['name']} lr={summary['lr']}")
                print("native_execution=false model_loads=0")
            return 0
        if args.command == "run":
            report = run_protocol(
                args.config, output=args.output, model_path=args.model_path,
                model_lock=args.model_lock,
                authorization_path=args.authorization, device=args.device)
            print(json.dumps({
                "status": report["status"],
                "native_execution": True,
                "config_sha256": report["config_sha256"],
                "protocol_commit": report["protocol_commit"],
                "report": str(Path(args.output) / "report.json"),
                "gates": report["gates"],
                "load_counts": report["load_counts"],
                "wall_seconds": report["wall_seconds"],
                "artifact_bytes": report["artifact_bytes"],
            }, indent=2, sort_keys=True))
            return 0
        review = review_launch(args.config,
                              authorization_path=args.authorization)
        print(json.dumps(review, indent=2, sort_keys=True))
        return 0
    except ProtocolError as error:
        print(f"{args.command} refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
