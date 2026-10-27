"""Training-only bounded vision learning-curve contracts.

This module owns the model-free half of the five-readout curve diagnostic:

* strict validation of the predeclared curve config and its historical pins;
* per-readout invariants (prefix order, one greedy hypothesis, teacher-forced
  cells for both roles, projector/optimizer/base immutability during scoring);
* the T1/T2/T3 trajectory that is the *only* curve disposition input;
* the curve stop-condition wrapper (never labelled as a gradient receipt).

A curve readout is taken after the two backward-only probes (prefix 0) and
after updates 1, 2, 4 and 7.  Readout prefix 7 also carries the 22 declared
train-fit gates, but only as endpoint context; those gates are never the curve
decision.  No status in this module authorizes development, held-out, V0,
promotion or a retry.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import vision_learnability_diagnostic as learn


require = learn.require

VERSION = "replica.vision_learnability_curve.v1"
CONFIG_FORMAT = "replica.vision_learnability_curve_config.v1"
REPORT_FORMAT = "replica.vision_learnability_curve.report.v1"
READOUT_FORMAT = "replica.vision_learnability_curve.readout.v1"
DECISION_FORMAT = "replica.vision_learnability_curve.decision.v1"
STOP_FORMAT = "replica.vision_learnability_curve.stop.v1"
FAILURE_FORMAT = "replica.vision_learnability_curve.report.v1.failure"
MODE = "curve"

PREFIXES = (0, 1, 2, 4, 7)
READOUT_UPDATES = (1, 2, 4, 7)
EXPECTED_READOUTS = len(PREFIXES)

EXPECTED_ROWS = learn.EXPECTED_ROWS
EXPECTED_RECORDS = learn.EXPECTED_RECORDS
EXPECTED_AVAILABLE_RECORDS = learn.EXPECTED_AVAILABLE_RECORDS
EXPECTED_UNAVAILABLE_RECORDS = learn.EXPECTED_UNAVAILABLE_RECORDS
EXPECTED_GENERATIONS_PER_READOUT = learn.EXPECTED_GENERATIONS
EXPECTED_TEACHER_FORCED_CELLS_PER_READOUT = learn.EXPECTED_TEACHER_FORCED_CELLS
EXPECTED_SCORING_FORWARDS_PER_READOUT = 3 * learn.EXPECTED_GENERATIONS
EXPECTED_GENERATIONS = EXPECTED_READOUTS * EXPECTED_GENERATIONS_PER_READOUT
EXPECTED_TEACHER_FORCED_CELLS = (
    EXPECTED_READOUTS * EXPECTED_TEACHER_FORCED_CELLS_PER_READOUT)
EXPECTED_SCORING_FORWARDS = (
    EXPECTED_READOUTS * EXPECTED_SCORING_FORWARDS_PER_READOUT)
EXPECTED_UPDATES = learn.EXPECTED_UPDATES
EXPECTED_UPDATE_BACKWARD_CALLS = learn.EXPECTED_UPDATE_BACKWARD_CALLS
EXPECTED_PROBE_BACKWARD_CALLS = learn.EXPECTED_PROBE_BACKWARD_CALLS
EXPECTED_TOTAL_BACKWARD_CALLS = learn.EXPECTED_TOTAL_BACKWARD_CALLS

STATUS_ENGINEERING_FAILURE = "stopped_engineering_failure"
STATUS_CONVERSION = "curve_readout_conversion_observed"
STATUS_OPTIMIZATION_ONLY = "curve_optimization_only_no_conversion"
STATUS_NO_PROGRESS = "curve_no_optimization_progress"
STATUSES = (
    STATUS_ENGINEERING_FAILURE,
    STATUS_CONVERSION,
    STATUS_OPTIMIZATION_ONLY,
    STATUS_NO_PROGRESS,
)

DEFAULT_DECISION_CONTRACT = {
    "t1_relative_reduction_min": 0.25,
    "t2_non_increasing_tolerance": 0.001,
    "t3_answer_prefill_gold_accuracy_min": 1.0 / 14.0,
    "t3_answer_correct_accuracy_min": 1.0 / 14.0,
    "t3_caption_token_f1_gain_min": 0.10,
    "t3_answer_prefill_gold_rank_ratio_max": 0.5,
}
DECISION_CONTRACT_KEYS = set(DEFAULT_DECISION_CONTRACT)

READOUT_CONTRACT = {
    "prefixes": list(PREFIXES),
    "zero_prefix": "after_probes_before_updates",
    "after_updates": list(READOUT_UPDATES),
    "full_control_matrix": True,
    "controls_per_row": 4,
    "records_per_readout": EXPECTED_RECORDS,
    "available_per_readout": EXPECTED_AVAILABLE_RECORDS,
    "placeholders_per_readout": EXPECTED_UNAVAILABLE_RECORDS,
    "teacher_forced_roles": ["answer", "caption"],
    "teacher_forced_source": "same_teacher_forced_loss_path",
    "teacher_forced_answers_target": "row.response",
    "teacher_forced_captions_target": "sealed_caption_answer",
    "prefill_analysis": True,
    "one_greedy_hypothesis": True,
    "correct_noimage_aggregates": True,
    "adapter_train_restored": True,
    "per_prefix_files_before_report": True,
    "readout_weight_writes": 0,
    "empty_text_policy": "score_as_wrong_and_record",
}

def generation_hypothesis_report(rows, **kwargs) -> dict[str, Any]:
    """Preserve empty decoded responses as scored failures, not engineering errors."""
    report = learn.generation_hypothesis_report(rows, **kwargs)
    empty = [item for item in report["violations"]
             if item.startswith("empty_text:")]
    if not empty:
        return report
    remaining = [item for item in report["violations"]
                 if item not in empty]
    return {
        **report,
        "ok": not remaining,
        "violations": remaining,
        "empty_decoded_responses": len(empty),
        "empty_text_policy": "score_as_wrong_and_record",
        "empty_text_violations": empty,
    }


_PREPARED_KEYS = {
    "root", "protocol", "protocol_sha256", "manifest", "manifest_sha256",
    "training", "training_sha256", "training_examples", "answer_examples",
    "caption_examples", "answer_pairs", "caption_pair_ids",
    "read_only_training", "forbidden_access",
}
_SOURCE_KEYS = {
    "model_path", "model_lock", "model_lock_sha256", "model_revision",
    "vision_source_manifest", "vision_source_manifest_sha256",
    "protocol_authorities",
}
_PRECONDITION_KEYS = {
    "run", "report", "required_decision", "required_mode", "config",
    "config_sha256", "note",
}
_SCHEDULE_KEYS = {
    "source", "source_sha256", "caption_labels", "caption_labels_sha256",
    "prefix_updates", "batch_size", "answer_per_update", "caption_per_update",
    "unique_rows", "slots", "expected_answer_rows", "expected_caption_rows",
    "expected_repeated_rows", "smallest_full_coverage_prefix",
}
_ACCESS_KEYS = {
    "training_paths", "forbidden_paths", "order", "never_calls",
    "advance_after",
}
_V0_KEYS = {
    "planning_authorized_on_pass", "run_v0",
    "requires_separate_predeclared_job", "heldout_sealed",
}
_EXPECTATION_KEYS = {
    "rows", "generations_per_readout", "total_generations",
    "readouts", "updates", "update_slots", "update_backward_calls",
    "probe_backward_calls", "total_backward_calls", "fit_answers",
    "fit_captions", "valid_patches_per_image", "prefill_agreement",
    "teacher_forced_cells_per_readout", "total_teacher_forced_cells",
    "scoring_forwards_per_readout", "total_scoring_forwards",
}
_LIMIT_KEYS = {
    "max_updates", "max_optimizer_steps", "max_update_backward_calls",
    "max_probe_backward_calls", "max_total_backward_calls", "max_readouts",
    "max_scoring_generations", "max_teacher_forced_cells",
    "max_scoring_forwards", "readout_weight_writes", "max_seconds",
    "artifact_bytes", "max_prompt_tokens", "max_sequence_tokens",
    "max_new_tokens", "caption_max_new_tokens", "model_loads",
    "extension_loads", "extension_writes", "checkpoint_writes", "downloads",
    "workers", "grad_accum",
}
_CURVE_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "prepared", "source", "precondition", "schedule", "initialization",
    "optimizer", "probes", "readouts", "decision", "gates", "access",
    "v0_gate", "answer_max_new_tokens", "caption_max_new_tokens",
    "stop_tokens", "target_eos_token", "controls", "prompt_contract",
    "expectations", "limits", "policy", "implementation_pins", "output",
    "report_root", "claims", "limitations",
}

MIGRATION_PATHS = (
    "scripts/run_vision_curriculum.py",
    "scripts/run_vision_balanced_alignment.py",
    "src/replica_lab/architecture/contracts.py",
    "src/replica_lab/architecture/minicpm5.py",
    "src/replica_lab/memory_system/session.py",
)
EXPECTED_MIGRATIONS = {
    "scripts/run_vision_curriculum.py": (
        "8d792914e4adfd36ddac1f5a9eb264cd6821744e9e21ab75b716a4a2520a6749",
        "38e685ec0dd91e663a2c3739ee63c58f98f148ab689ffc06693763b6aba99fef",
    ),
    "scripts/run_vision_balanced_alignment.py": (
        "5fc326475aefc8ff78c0fa1bf9e1b963e681079e3f5297246a096c614fd78012",
        "8834e8db25bc6451577062b1bf9ab17fe2182c3d885146c4d32251ca8df03de9",
    ),
    "src/replica_lab/architecture/contracts.py": (
        "be6024ac7f3179f46d497169b962889d3524bb6733aad80fe8e927f7e8c3b8d0",
        "14daa2cc3cdcd171c17da039c06b694a270a6fff6bc9c1306303b93b65db0712",
    ),
    "src/replica_lab/architecture/minicpm5.py": (
        "9c4b42e7f3ace361cf212dc05bd3579a4d1b44f3d295b22f2959b22e71ba6850",
        "5348613a6c79d33c719cf20f6677b9266c4e725cefaff84af29c360a0a7ef019",
    ),
    "src/replica_lab/memory_system/session.py": (
        "f475d0fde3a825bce54d007f54eaafc97470da5ca348cc2e52e331e6ce518305",
        "915b7f196529b1b185cb4192c95f440c5d591869273c070336c7de43e85f2b98",
    ),
}
NEW_FILE_PINS = (
    "src/replica_lab/vision_learnability_curve.py",
    "scripts/run_vision_learnability_curve.py",
    "src/replica_lab/vision_learnability_diagnostic.py",
    "scripts/run_vision_learnability_diagnostic.py",
)


def load_config(path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    return validate_config(config)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete predeclared curve contract."""
    learn.require(isinstance(config, Mapping),
                  "The curve config must be a JSON object")
    require(config.get("mode") == MODE,
            f"Unknown curve mode: {config.get('mode')!r}")
    require(set(config) == _CURVE_KEYS,
            f"Curve config fields differ: "
            f"{sorted(set(config) ^ _CURVE_KEYS)}")
    require(config["format"] == CONFIG_FORMAT, "Config format differs")
    require(config["version"] == VERSION, "Config version differs")
    require(config["name"] == "vision_learnability_curve_v1",
            "Curve config name differs")
    require(isinstance(config["purpose"], str) and config["purpose"].strip(),
            "Config purpose is missing")
    require(isinstance(config["purpose"], str)
            and "curve" in config["purpose"].lower(),
            "Config purpose must describe the curve")
    learn._validate_prepared(config["prepared"])
    _validate_source(config["source"])
    _validate_precondition(config["precondition"])
    _validate_schedule(config["schedule"])
    _validate_initialization(config["initialization"])
    _validate_optimizer(config["optimizer"])
    _validate_probes(config["probes"])
    require(config["readouts"] == READOUT_CONTRACT,
            "Readout contract differs from the five-prefix curve")
    _validate_decision(config["decision"])
    require(config["gates"] == learn.GATE_THRESHOLDS,
            "Train-fit endpoint gates differ")
    _validate_access(config["access"])
    require(config["v0_gate"] == {
        "planning_authorized_on_pass": False,
        "run_v0": False,
        "requires_separate_predeclared_job": True,
        "heldout_sealed": True,
    }, "V0 gate contract differs")
    learn._validate_prompt_contract(config["prompt_contract"])
    require(config["answer_max_new_tokens"] == learn.EXPECTED_ANSWER_NEW_TOKENS
            and config["caption_max_new_tokens"]
            == learn.EXPECTED_CAPTION_NEW_TOKENS,
            "Curve generation budgets differ")
    require(tuple(config["stop_tokens"]) == learn.STOP_TOKENS,
            "Stop tokens differ from the native {1, 130073} set")
    require(config["target_eos_token"] == learn.TARGET_EOS_TOKEN,
            "Target EOS differs from </s>")
    require(tuple(config["controls"]) == learn.CONTROL_NAMES,
            "Controls differ from the declared matched set")
    _validate_policy(config["policy"])
    _validate_expectations(config["expectations"])
    _validate_limits(config["limits"])
    _validate_implementation_pins(config["implementation_pins"])
    claims = config["claims"]
    require(claims.get("promotion") is False
            and claims.get("learned_quality") is False
            and claims.get("perception_quality") is False
            and claims.get("semantic_quality") is False
            and claims.get("overall_baseline_eligible") is False,
            "The curve must not promote learned or semantic quality")
    require(isinstance(config["limitations"], list) and config["limitations"],
            "Limitations are required")
    require(str(config["output"]).startswith("runs/")
            and str(config["report_root"]).startswith("reports/"),
            "Output and report roots must be fresh declared paths")
    return dict(config)


def _validate_source(source: Mapping[str, Any]) -> None:
    require(isinstance(source, Mapping) and set(source) == _SOURCE_KEYS,
            f"Source pin fields differ: "
            f"{sorted(set(source or {}) ^ _SOURCE_KEYS)}")
    require(source["model_path"] == "checkpoints/minicpm5_source"
            and source["model_lock"] == "configs/minicpm5_sealed_lock.local.json",
            "Source model path/lock differ")
    require(source["model_lock_sha256"]
            == "26dea74b4c543bb000c64ac7509171a399beee611ca0f7b70ebd941da64af792",
            "Source model lock pin differs")
    require(source["model_revision"]
            == "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
            "Source model revision differs")
    require(source["vision_source_manifest"]
            == "checkpoints/siglip2_base_patch16_224_source/source_manifest.json",
            "Vision source manifest path differs")
    require(source["vision_source_manifest_sha256"]
            == "8b2351f7b29f2bd7d881405dd978ae714a26c0170b7213d133a9e0059a199024",
            "Vision source manifest pin differs")
    require(source["protocol_authorities"] == [
        "model.lock_sha256", "model.revision", "vision_digest", "vision",
        "vision_source", "resource_policy", "limits",
    ], "Source protocol authorities differ")


def _validate_precondition(precondition: Mapping[str, Any]) -> None:
    require(isinstance(precondition, Mapping)
            and set(precondition) == _PRECONDITION_KEYS,
            "Precondition fields differ")
    require(precondition["required_decision"] == learn.DECISION_OBSERVED
            and precondition["required_mode"] == "ablation",
            "The curve requires an observed ablation precondition")
    _require_sha256(precondition["config_sha256"],
                    "precondition config_sha256")


def _validate_schedule(schedule: Mapping[str, Any]) -> None:
    require(isinstance(schedule, Mapping)
            and set(schedule) == _SCHEDULE_KEYS,
            "Schedule fields differ")
    require(schedule["prefix_updates"] == EXPECTED_UPDATES
            and schedule["batch_size"] == 4
            and schedule["answer_per_update"] == 2
            and schedule["caption_per_update"] == 2
            and schedule["unique_rows"] == EXPECTED_ROWS
            and schedule["slots"] == learn.EXPECTED_UPDATE_SLOTS
            and schedule["expected_answer_rows"] == learn.EXPECTED_ANSWER_ROWS
            and schedule["expected_caption_rows"]
            == learn.EXPECTED_CAPTION_ROWS
            and schedule["expected_repeated_rows"] == 2
            and schedule["smallest_full_coverage_prefix"] is True,
            "Seven-update full-coverage prefix differs")
    require(schedule["source"] == "training/batches.jsonl"
            and schedule["caption_labels"]
            == "training/train_caption_labels.jsonl",
            "Schedule paths differ")
    _require_sha256(schedule["source_sha256"], "schedule source_sha256")
    _require_sha256(schedule["caption_labels_sha256"],
                    "schedule caption_labels_sha256")


def _validate_initialization(initialization: Mapping[str, Any]) -> None:
    require(initialization == {
        "init_seed": 29,
        "recipe": "fresh_seeded_normal_std0.02",
        "warm_start": None,
        "forbidden_parent_weights_sha256": [
            "bdc367424b2afcb2e88dba04b6b29495e3a440b5d850f8ae5bc55dbfa428c3a4"],
        "projector_tensors": list(learn.PROJECTOR_TENSORS),
        "parameter_count": learn.PROJECTOR_PARAMETER_COUNT,
    }, "Fresh seed-29 initialization contract differs")


def _validate_optimizer(optimizer: Mapping[str, Any]) -> None:
    require(optimizer == {
        "name": "adamw_fused",
        "wrapper": "NativeFusedAdamW",
        "optimizer_class": "torch.optim.AdamW",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "max_grad_norm": 1.0,
        "grad_accum": 1,
    }, "Native fused AdamW contract differs")


def _validate_probes(probes: Mapping[str, Any]) -> None:
    require(probes == {
        "count": EXPECTED_PROBE_BACKWARD_CALLS,
        "roles": ["answer", "caption"],
        "backward_only": True,
        "optimizer_steps": 0,
        "weight_change": "none",
    }, "Backward-only probe contract differs")


def _validate_decision(decision: Mapping[str, Any]) -> None:
    require(isinstance(decision, Mapping), "Decision contract is missing")
    require(set(decision) == (DECISION_CONTRACT_KEYS | {
        "precedence", "trailing_slope", "v0_authorized",
        "dev_authorized", "heldout_authorized", "promotion_authorized",
        "retry_authorized",
    }), "Decision contract fields differ")
    for key, value in DEFAULT_DECISION_CONTRACT.items():
        require(float(decision[key]) == float(value),
                f"Decision threshold {key} differs")
    require(decision["precedence"] == list(STATUSES),
            "Curve decision precedence differs")
    require(decision["trailing_slope"] == "descriptive_only",
            "Trailing slope must remain descriptive only")
    for key in ("v0_authorized", "dev_authorized", "heldout_authorized",
                "promotion_authorized", "retry_authorized"):
        require(decision[key] is False,
                f"Curve status must not authorize {key}")


def _validate_access(access: Mapping[str, Any]) -> None:
    require(isinstance(access, Mapping) and set(access) == _ACCESS_KEYS,
            "Access fields differ")
    require(access["training_paths"] == [
        "training/alignment.jsonl",
        "training/batches.jsonl",
        "training/train_caption_labels.jsonl",
    ] and access["forbidden_paths"] == learn.FORBIDDEN_ACCESS
            and access["order"] == ["train", "train_fit"]
            and access["never_calls"] == ["_load_dev_v2", "_load_heldout"],
            "Training-only access contract differs")
    require(access["advance_after"] == "first_readout",
            "Access must advance to train_fit after the first readout")


def _validate_policy(policy: Mapping[str, Any]) -> None:
    require(isinstance(policy, Mapping), "Resource policy is missing")
    required = {
        "device": "mps",
        "cpu_fallback": False,
        "exclusive_mps_worker": True,
        "capacity_guard_required": True,
        "profiler_required": True,
        "fresh_output": True,
        "sealed_evaluation_access": 0,
        "no_extension_save": True,
        "no_downloads": True,
        "matched_frozen_components": True,
        "first_order_evidence_only": True,
        "train_only_access": True,
        "no_dev_access": True,
        "no_heldout_access": True,
        "no_semantic_audit": True,
        "deterministic_single_hypothesis": True,
    }
    for key, value in required.items():
        require(policy.get(key) == value,
                f"Declared policy {key} differs: {policy.get(key)!r}")


def _validate_expectations(expected: Mapping[str, Any]) -> None:
    require(isinstance(expected, Mapping)
            and set(expected) == _EXPECTATION_KEYS,
            "Curve expectations fields differ")
    require(expected == {
        "rows": EXPECTED_ROWS,
        "generations_per_readout": EXPECTED_GENERATIONS_PER_READOUT,
        "total_generations": EXPECTED_GENERATIONS,
        "readouts": EXPECTED_READOUTS,
        "updates": EXPECTED_UPDATES,
        "update_slots": learn.EXPECTED_UPDATE_SLOTS,
        "update_backward_calls": EXPECTED_UPDATE_BACKWARD_CALLS,
        "probe_backward_calls": EXPECTED_PROBE_BACKWARD_CALLS,
        "total_backward_calls": EXPECTED_TOTAL_BACKWARD_CALLS,
        "fit_answers": learn.EXPECTED_ANSWER_ROWS,
        "fit_captions": learn.EXPECTED_CAPTION_ROWS,
        "valid_patches_per_image": learn.EXPECTED_VALID_PATCHES,
        "prefill_agreement": EXPECTED_GENERATIONS_PER_READOUT,
        "teacher_forced_cells_per_readout":
            EXPECTED_TEACHER_FORCED_CELLS_PER_READOUT,
        "total_teacher_forced_cells": EXPECTED_TEACHER_FORCED_CELLS,
        "scoring_forwards_per_readout": EXPECTED_SCORING_FORWARDS_PER_READOUT,
        "total_scoring_forwards": EXPECTED_SCORING_FORWARDS,
    }, "Curve expectations differ")


def _validate_limits(limits: Mapping[str, Any]) -> None:
    require(isinstance(limits, Mapping) and set(limits) == _LIMIT_KEYS,
            f"Curve limits fields differ: "
            f"{sorted(set(limits or {}) ^ _LIMIT_KEYS)}")
    require(limits["max_updates"] == EXPECTED_UPDATES
            and limits["max_optimizer_steps"] == EXPECTED_UPDATES
            and limits["max_update_backward_calls"]
            == EXPECTED_UPDATE_BACKWARD_CALLS
            and limits["max_probe_backward_calls"]
            == EXPECTED_PROBE_BACKWARD_CALLS
            and limits["max_total_backward_calls"]
            == EXPECTED_TOTAL_BACKWARD_CALLS
            and limits["max_readouts"] == EXPECTED_READOUTS
            and limits["max_scoring_generations"] == EXPECTED_GENERATIONS
            and limits["max_teacher_forced_cells"]
            == EXPECTED_TEACHER_FORCED_CELLS
            and limits["max_scoring_forwards"] == EXPECTED_SCORING_FORWARDS
            and limits["readout_weight_writes"] == 0
            and limits["max_seconds"] == 900
            and limits["artifact_bytes"] == 67108864
            and limits["model_loads"] == 1
            and limits["extension_loads"] == 0
            and limits["extension_writes"] == 0
            and limits["checkpoint_writes"] == 0
            and limits["downloads"] == 0
            and limits["workers"] == 1
            and limits["grad_accum"] == 1,
            "Curve update/readout/cap limits differ")
    learn._validate_token_limits(limits)


def _validate_implementation_pins(pins: Mapping[str, Any]) -> None:
    require(isinstance(pins, Mapping), "Implementation pins are missing")
    require(pins.get("prepared_protocol_pins_checked") is True
            and pins.get("mode") == "explicit_read_only_migration",
            "Explicit read-only implementation migrations are required")
    require(isinstance(pins.get("reason"), str) and pins["reason"].strip(),
            "Implementation migration reason is missing")
    migrations = pins.get("migrations")
    require(isinstance(migrations, Mapping)
            and set(migrations) == set(MIGRATION_PATHS),
            "The five exact implementation migrations are required")
    for relative in MIGRATION_PATHS:
        migration = migrations[relative]
        require(isinstance(migration, Mapping)
                and set(migration) == {"parent_sha256", "current_sha256",
                                       "reason"},
                f"Malformed migration: {relative}")
        parent, current = EXPECTED_MIGRATIONS[relative]
        require(migration["parent_sha256"] == parent,
                f"Migration parent differs: {relative}")
        require(migration["current_sha256"] == current,
                f"Migration current hash differs: {relative}")
        _require_sha256(migration["parent_sha256"], "parent_sha256")
        _require_sha256(migration["current_sha256"], "current_sha256")
        require(isinstance(migration["reason"], str)
                and migration["reason"].strip(),
                f"Migration lacks a reason: {relative}")
    new_files = pins.get("new_files")
    require(isinstance(new_files, Mapping)
            and set(new_files) == set(NEW_FILE_PINS),
            "New-file implementation pins differ")
    for relative, digest in new_files.items():
        _require_sha256(digest, relative)


def _require_sha256(value: Any, name: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{name} is not a SHA-256 digest")


# ---------------------------------------------------------------------------
# Per-readout invariants
# ---------------------------------------------------------------------------


def optimizer_state_report(optimizer) -> dict[str, Any]:
    """Hash the optimizer's live state so a readout cannot mutate it."""
    import torch
    digest = hashlib.sha256()
    tensor_bytes = 0
    step_values: list[int] = []
    state = getattr(optimizer, "state", {})
    parameters = []
    for group in getattr(optimizer, "param_groups", []):
        parameters.extend(list(group.get("params") or []))
    for index, parameter in enumerate(parameters):
        entries = state.get(parameter, {}) if hasattr(state, "get") else {}
        digest.update(str(index).encode())
        for key in sorted(entries):
            value = entries[key]
            digest.update(str(key).encode())
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
                tensor_bytes += tensor.numel() * tensor.element_size()
                if key == "step":
                    step_values.append(int(tensor.reshape(-1)[0].item()))
            else:
                digest.update(str(value).encode())
                if key == "step":
                    try:
                        step_values.append(int(value))
                    except (TypeError, ValueError):
                        pass
    steps = sorted(set(step_values))
    return {
        "sha256": digest.hexdigest(),
        "state_entries": len(state),
        "parameters_with_state": len(step_values),
        "steps": steps,
        "max_step": max(steps) if steps else 0,
        "tensor_bytes": tensor_bytes,
        "empty": len(state) == 0,
    }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def teacher_forced_aggregates(
        records: Sequence[Mapping[str, Any]]
        ) -> dict[str, dict[str, dict[str, Any]]]:
    """Mean teacher-forced loss per role and control."""
    grouped: dict[str, dict[str, list[float]]] = {
        "answer": {}, "caption": {}}
    for record in records:
        if record.get("available") is False:
            continue
        role = str(record.get("kind"))
        control = str(record.get("control"))
        if role not in grouped or control not in learn.CONTROL_NAMES:
            continue
        loss = _finite(record.get("teacher_forced_loss"))
        if loss is None:
            continue
        grouped[role].setdefault(control, []).append(loss)
    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ("answer", "caption"):
        aggregates[role] = {}
        for control in learn.CONTROL_NAMES:
            values = grouped[role].get(control, [])
            aggregates[role][control] = {
                "mean": (sum(values) / len(values) if values else None),
                "cases": len(values),
            }
    return aggregates


def readout_metrics(
        records: Sequence[Mapping[str, Any]],
        scores: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Curve-only metrics, including the answer prefill gold rank mean."""
    answer_correct = [
        record for record in records
        if record.get("available") is not False
        and record.get("kind") == "answer"
        and record.get("control") == "correct"
    ]
    ranks = []
    for record in answer_correct:
        prefill = record.get("prefill") or {}
        rank = _finite(prefill.get("gold_rank"))
        if rank is not None:
            ranks.append(rank)
    return {
        "answer_prefill_gold_rank_mean":
            (sum(ranks) / len(ranks) if ranks else None),
        "answer_prefill_gold_rank_cases": len(ranks),
        "answer_prefill_gold_rank_min": min(ranks) if ranks else None,
        "answer_prefill_gold_rank_max": max(ranks) if ranks else None,
    }


def validate_curve_readouts(
        readouts: Sequence[Mapping[str, Any]], *,
        init_hash: str,
        prefixes: Sequence[int] = PREFIXES,
        ) -> dict[str, Any]:
    """Validate prefix, digest, weight, scoring and endpoint invariants."""
    violations: list[str] = []
    expected_prefixes = [int(value) for value in prefixes]
    if len(readouts) != len(expected_prefixes):
        violations.append(f"readouts:{len(readouts)}")
    seen_prefixes: list[int] = []
    seen_digests: list[str] = []
    total_generations = 0
    total_teacher_forced = 0
    total_scoring_forwards = 0
    endpoint_gates = 0
    for position, readout in enumerate(readouts):
        prefix = readout.get("prefix")
        if position < len(expected_prefixes):
            expected = expected_prefixes[position]
            if prefix != expected:
                violations.append(f"prefix:{position}:{prefix}!={expected}")
        if prefix in seen_prefixes:
            violations.append(f"duplicate_prefix:{prefix}")
        seen_prefixes.append(prefix)
        label = f"prefix:{prefix}"
        before = readout.get("optimizer_state_before") or {}
        after = readout.get("optimizer_state_after") or {}
        if readout.get("optimizer_state_unchanged") is not True \
                or before.get("sha256") != after.get("sha256") \
                or before.get("max_step") != after.get("max_step") \
                or before.get("state_entries") != after.get("state_entries"):
            violations.append(f"{label}:optimizer_state_changed")
        steps = readout.get("optimizer_steps")
        if steps != prefix:
            violations.append(f"{label}:optimizer_steps:{steps}")
        empty = readout.get("optimizer_state_empty")
        if prefix == 0:
            if empty is not True or after.get("empty") is not True:
                violations.append(f"{label}:optimizer_state_not_empty")
        elif empty is not False or after.get("empty") is not False:
            violations.append(f"{label}:optimizer_state_empty")
        digest = readout.get("projector_state_sha256")
        if readout.get("projector_unchanged_by_readout") is not True \
                or readout.get("projector_state_sha256_before") != digest \
                or readout.get("projector_state_sha256_after") != digest:
            violations.append(f"{label}:projector_changed_by_readout")
        if not isinstance(digest, str) or len(digest) != 64:
            violations.append(f"{label}:projector_digest")
        if digest in seen_digests:
            violations.append(f"{label}:projector_digest_repeated")
        seen_digests.append(digest)
        if prefix == 0 and digest != init_hash:
            violations.append(f"{label}:projector_digest_not_initial")
        if prefix != 0 and digest == init_hash:
            violations.append(f"{label}:projector_digest_still_initial")
        if readout.get("readout_weight_writes") != 0 \
                or (readout.get("weight_change_names") or []) != []:
            violations.append(f"{label}:readout_weight_write")
        if readout.get("base_stamp_unchanged") is not True \
                or readout.get("base_stamp_before") \
                != readout.get("base_stamp_after"):
            violations.append(f"{label}:base_stamp_changed")
        if readout.get("adapter_train_restored") is not True \
                or readout.get("adapter_mode_after") != "train":
            violations.append(f"{label}:adapter_not_train")
        counts = readout.get("counts") or {}
        expected_counts = {
            "records": EXPECTED_RECORDS,
            "available": EXPECTED_AVAILABLE_RECORDS,
            "placeholders": EXPECTED_UNAVAILABLE_RECORDS,
            "generations": EXPECTED_GENERATIONS_PER_READOUT,
            "teacher_forced_cells":
                EXPECTED_TEACHER_FORCED_CELLS_PER_READOUT,
            "scoring_forwards": EXPECTED_SCORING_FORWARDS_PER_READOUT,
        }
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                violations.append(f"{label}:counts:{key}")
        generation = readout.get("generation") or {}
        if generation.get("ok") is not True:
            violations.append(f"{label}:generation")
            violations.extend(
                f"{label}:generation:{name}"
                for name in (generation.get("violations") or []))
        if generation.get("records") != EXPECTED_RECORDS \
                or generation.get("available") != EXPECTED_AVAILABLE_RECORDS \
                or generation.get("unavailable") \
                != EXPECTED_UNAVAILABLE_RECORDS:
            violations.append(f"{label}:generation_counts")
        total_generations += int(counts.get("generations") or 0)
        total_teacher_forced += int(counts.get("teacher_forced_cells") or 0)
        total_scoring_forwards += int(counts.get("scoring_forwards") or 0)
        scores = readout.get("scores") or {}
        answer = scores.get("answer") or {}
        caption = scores.get("caption") or {}
        for name in ("answer_prefill_gold_accuracy",
                     "answer_correct_accuracy"):
            if _finite(answer.get(name)) is None:
                violations.append(f"{label}:score:{name}")
        if _finite(caption.get("caption_token_f1")) is None:
            violations.append(f"{label}:score:caption_token_f1")
        aggregates = readout.get("teacher_forced_aggregates") or {}
        for role in ("answer", "caption"):
            role_aggregates = aggregates.get(role) or {}
            for control in ("correct", "noimage"):
                value = (role_aggregates.get(control) or {}).get("mean")
                if _finite(value) is None:
                    violations.append(
                        f"{label}:teacher_forced:{role}:{control}")
        if _finite((readout.get("metrics") or {}).get(
                "answer_prefill_gold_rank_mean")) is None:
            violations.append(f"{label}:metrics:prefill_rank")
        gates = readout.get("gates")
        if prefix == 7:
            endpoint_gates += 1
            if not isinstance(gates, Mapping) \
                    or not isinstance(gates.get("all_declared_gates_passed"),
                                      bool):
                violations.append(f"{label}:endpoint_gates")
        elif gates is not None:
            violations.append(f"{label}:unexpected_gates")
    if total_generations != EXPECTED_GENERATIONS:
        violations.append(f"total_generations:{total_generations}")
    if total_teacher_forced != EXPECTED_TEACHER_FORCED_CELLS:
        violations.append(
            f"total_teacher_forced_cells:{total_teacher_forced}")
    if total_scoring_forwards != EXPECTED_SCORING_FORWARDS:
        violations.append(f"total_scoring_forwards:{total_scoring_forwards}")
    if endpoint_gates != 1:
        violations.append(f"endpoint_gates:{endpoint_gates}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "readouts": len(readouts),
        "prefixes": seen_prefixes,
        "projector_digests": seen_digests,
        "initialization_sha256": init_hash,
        "total_generations": total_generations,
        "total_teacher_forced_cells": total_teacher_forced,
        "total_scoring_forwards": total_scoring_forwards,
        "endpoint_gate_reports": endpoint_gates,
        "scope": ("five readouts at 0/1/2/4/7; every readout leaves the "
                  "projector, optimizer state and sealed base unchanged and "
                  "emits the exact 104/98/6 control matrix"),
    }


# ---------------------------------------------------------------------------
# T1/T2/T3 trajectory and decision
# ---------------------------------------------------------------------------


def _readout_index(readouts: Sequence[Mapping[str, Any]]) -> dict[int, Mapping]:
    index: dict[int, Mapping] = {}
    for readout in readouts:
        prefix = int(readout.get("prefix"))
        learn.require(prefix not in index,
                      f"Duplicate curve readout prefix: {prefix}")
        index[prefix] = readout
    learn.require(tuple(sorted(index)) == PREFIXES,
                  f"Curve readouts must cover {PREFIXES}")
    return index


def _score(readout: Mapping[str, Any], role: str, name: str) -> float | None:
    return _finite(((readout.get("scores") or {}).get(role) or {}).get(name))


def _teacher_loss(readout: Mapping[str, Any], role: str) -> float | None:
    aggregates = readout.get("teacher_forced_aggregates") or {}
    return _finite(((aggregates.get(role) or {}).get("correct") or {}).get(
        "mean"))


def _rank_mean(readout: Mapping[str, Any]) -> float | None:
    return _finite((readout.get("metrics") or {}).get(
        "answer_prefill_gold_rank_mean"))


def _slope(values: Sequence[float | None]) -> float | None:
    points = [(prefix, value) for prefix, value in zip((4, 7), values[-2:])
              if value is not None]
    if len(points) < 2:
        return None
    (x0, y0), (x1, y1) = points
    return (y1 - y0) / (x1 - x0)


def curve_trajectory_report(
        readouts: Sequence[Mapping[str, Any]], *,
        contract: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Compute T1/T2/T3 from the four correct losses and readout metrics."""
    values = dict(DEFAULT_DECISION_CONTRACT)
    if contract is not None:
        for key in DECISION_CONTRACT_KEYS:
            if key in contract:
                values[key] = float(contract[key])
    index = _readout_index(readouts)
    violations: list[str] = []
    teacher = {"answer": [], "caption": []}
    for prefix in PREFIXES:
        for role in ("answer", "caption"):
            value = _teacher_loss(index[prefix], role)
            if value is None:
                violations.append(f"teacher_forced:{role}:{prefix}")
            teacher[role].append(value)

    t1_roles: dict[str, Any] = {}
    for role in ("answer", "caption"):
        start, end = teacher[role][0], teacher[role][-1]
        reduction = None
        if start is not None and end is not None and start > 0:
            reduction = (start - end) / start
        passed = (reduction is not None
                  and reduction >= values["t1_relative_reduction_min"])
        if reduction is None:
            violations.append(f"t1:{role}")
        t1_roles[role] = {
            "prefix0_loss": start,
            "prefix7_loss": end,
            "relative_reduction": reduction,
            "threshold": values["t1_relative_reduction_min"],
            "passed": passed,
        }
    t1 = {"passed": all(item["passed"] for item in t1_roles.values()),
          "roles": t1_roles,
          "threshold": values["t1_relative_reduction_min"]}

    t2_roles: dict[str, Any] = {}
    for role in ("answer", "caption"):
        series = teacher[role]
        steps = []
        for left, right in zip(PREFIXES[:-1], PREFIXES[1:]):
            previous, current = series[PREFIXES.index(left)], series[
                PREFIXES.index(right)]
            passed = (previous is not None and current is not None
                      and current <= previous
                      + float(values["t2_non_increasing_tolerance"]))
            if previous is None or current is None:
                violations.append(f"t2:{role}:{left}->{right}")
            steps.append({
                "from": left,
                "to": right,
                "previous_loss": previous,
                "loss": current,
                "non_increasing": passed,
            })
        t2_roles[role] = {
            "series": [
                {"prefix": prefix, "loss": loss}
                for prefix, loss in zip(PREFIXES, series)],
            "steps": steps,
            "passed": all(step["non_increasing"] for step in steps),
        }
    t2 = {
        "passed": all(item["passed"] for item in t2_roles.values()),
        "tolerance": float(values["t2_non_increasing_tolerance"]),
        "steps": 4,
        "roles": t2_roles,
    }

    prefill0 = _score(index[0], "answer", "answer_prefill_gold_accuracy")
    prefill7 = _score(index[7], "answer", "answer_prefill_gold_accuracy")
    correct7 = _score(index[7], "answer", "answer_correct_accuracy")
    f1_0 = _score(index[0], "caption", "caption_token_f1")
    f1_7 = _score(index[7], "caption", "caption_token_f1")
    rank0 = _rank_mean(index[0])
    rank7 = _rank_mean(index[7])
    for name, value in (("answer_prefill_gold_accuracy:0", prefill0),
                        ("answer_prefill_gold_accuracy:7", prefill7),
                        ("answer_correct_accuracy:7", correct7),
                        ("caption_token_f1:0", f1_0),
                        ("caption_token_f1:7", f1_7),
                        ("prefill_rank:0", rank0),
                        ("prefill_rank:7", rank7)):
        if value is None:
            violations.append(f"t3:{name}")
    prefill_min = float(values["t3_answer_prefill_gold_accuracy_min"])
    correct_min = float(values["t3_answer_correct_accuracy_min"])
    f1_gain_min = float(values["t3_caption_token_f1_gain_min"])
    rank_ratio = float(values["t3_answer_prefill_gold_rank_ratio_max"])
    prefill_passed = (
        prefill0 is not None and prefill7 is not None
        and prefill0 <= 1e-12 and prefill7 + 1e-12 >= prefill_min)
    correct_passed = correct7 is not None and correct7 + 1e-12 >= correct_min
    f1_gain = (None if f1_0 is None or f1_7 is None else f1_7 - f1_0)
    f1_passed = f1_gain is not None and f1_gain + 1e-12 >= f1_gain_min
    rank_ratio_value = (None if rank0 in (None, 0) or rank7 is None
                        else rank7 / rank0)
    rank_passed = (
        rank0 is not None and rank7 is not None and rank0 > 0
        and rank7 <= rank_ratio * rank0 + 1e-12)
    criteria = {
        "answer_prefill_gold_accuracy_from_zero": {
            "prefix0": prefill0,
            "prefix7": prefill7,
            "threshold": prefill_min,
            "baseline_zeroed": prefill0 is not None and prefill0 <= 1e-12,
            "passed": prefill_passed,
        },
        "answer_correct_accuracy": {
            "prefix7": correct7,
            "threshold": correct_min,
            "passed": correct_passed,
        },
        "caption_token_f1_gain": {
            "prefix0": f1_0,
            "prefix7": f1_7,
            "gain": f1_gain,
            "threshold": f1_gain_min,
            "passed": f1_passed,
        },
        "answer_prefill_gold_rank_mean_ratio": {
            "prefix0": rank0,
            "prefix7": rank7,
            "ratio": rank_ratio_value,
            "threshold_ratio": rank_ratio,
            "passed": rank_passed,
        },
    }
    t3 = {
        "passed": any(item["passed"] for item in criteria.values()),
        "criteria": criteria,
    }
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "prefixes": list(PREFIXES),
        "teacher_forced_loss": teacher,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "trailing_slope": {
            "descriptive_only": True,
            "answer": _slope(teacher["answer"]),
            "caption": _slope(teacher["caption"]),
            "scope": ("prefix 4->7 descriptive slope only; it is not an "
                      "input to the curve decision"),
        },
        "scope": ("T1 both-role correct NLL relative reduction, T2 "
                  "non-increasing losses at all four steps, and one T3 "
                  "conversion criterion from prefix 0 to 7"),
    }


def curve_decision(
        *, engineering_report: Mapping[str, Any],
        trajectory_report: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Terminal curve disposition; no status opens dev, held-out or V0."""
    t1 = trajectory_report.get("t1") or {}
    t2 = trajectory_report.get("t2") or {}
    t3 = trajectory_report.get("t3") or {}
    t1_passed = t1.get("passed") is True
    t2_passed = t2.get("passed") is True
    t3_passed = t3.get("passed") is True
    if engineering_report.get("ok") is not True:
        status = STATUS_ENGINEERING_FAILURE
        reason = ("a curve engineering, readout, ownership, cap, coverage, "
                  "decoding or access invariant failed")
    elif t1_passed and t2_passed and t3_passed:
        status = STATUS_CONVERSION
        reason = ("both teacher-forced NLLs fell by at least 25%, every step "
                  "was non-increasing, and at least one conversion criterion "
                  "moved")
    elif t1_passed and t2_passed:
        status = STATUS_OPTIMIZATION_ONLY
        reason = ("both teacher-forced NLLs and every step improved, but no "
                  "predeclared conversion criterion moved")
    else:
        status = STATUS_NO_PROGRESS
        reason = ("the curve did not satisfy both T1 and T2 over prefixes "
                  "0/1/2/4/7")
    return {
        "format": DECISION_FORMAT,
        "mode": MODE,
        "status": status,
        "reason": reason,
        "engineering_ok": engineering_report.get("ok") is True,
        "t1": {
            "passed": t1_passed,
            "roles": t1.get("roles"),
            "threshold": t1.get("threshold"),
        },
        "t2": {
            "passed": t2_passed,
            "tolerance": t2.get("tolerance"),
            "steps": t2.get("steps"),
            "roles": t2.get("roles"),
        },
        "t3": {
            "passed": t3_passed,
            "criteria": t3.get("criteria"),
        },
        "trajectory_ok": trajectory_report.get("ok") is True,
        "trailing_slope_used_in_decision": False,
        "endpoint_gates_used_in_decision": False,
        "authorizes_v0": False,
        "v0_authorized": False,
        "v0_planning_authorized": False,
        "v0_run": False,
        "dev_authorized": False,
        "dev_accessed": False,
        "heldout_authorized": False,
        "heldout_accessed": False,
        "promotion": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "no_automatic_retry": True,
        "next_gate": "no development, held-out, V0, promotion or retry is authorized by this curve",
    }


# ---------------------------------------------------------------------------
# Engineering aggregate and curve stop accounting
# ---------------------------------------------------------------------------


def curve_engineering_report(
        *, trace: Sequence[Mapping[str, Any]],
        probes: Sequence[Mapping[str, Any]],
        limits: Mapping[str, Any],
        schedule_report: Mapping[str, Any],
        generation_report: Mapping[str, Any],
        fit_coverage: Mapping[str, Any],
        access_report: Mapping[str, Any],
        precondition: Mapping[str, Any],
        readouts: Sequence[Mapping[str, Any]],
        init_hash: str,
        updates_completed: int, optimizer_steps: int,
        update_backward_calls: int, probe_backward_calls: int,
        readout_weight_writes: int = 0,
        expected_kinds: Mapping[str, int] | None = None,
        decision_contract: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Aggregate every curve precondition without weakening any check."""
    violations: list[str] = []
    try:
        gradient = learn.gradient_engineering_report(
            trace=trace, probes=probes, limits=limits,
            schedule_report=schedule_report,
            generation_report=generation_report,
            fit_coverage=fit_coverage, access_report=access_report,
            precondition=precondition, updates_completed=updates_completed,
            optimizer_steps=optimizer_steps,
            update_backward_calls=update_backward_calls,
            probe_backward_calls=probe_backward_calls,
            expected_kinds=expected_kinds)
    except learn.LearnabilityDiagnosticError as error:
        gradient = {"ok": False, "violations": [str(error)], "caps": {}}
    if gradient.get("ok") is not True:
        violations.extend(f"gradient:{name}"
                          for name in gradient.get("violations") or [])
    readout_report = validate_curve_readouts(readouts, init_hash=init_hash)
    if readout_report.get("ok") is not True:
        violations.extend(f"readout:{name}"
                          for name in readout_report["violations"])
    try:
        trajectory = curve_trajectory_report(
            readouts, contract=decision_contract)
    except learn.LearnabilityDiagnosticError as error:
        trajectory = {"ok": False, "violations": [str(error)]}
        violations.append(f"trajectory:{error}")
    if trajectory.get("ok") is not True:
        violations.extend(f"trajectory:{name}"
                          for name in trajectory.get("violations") or [])
    if int(readout_weight_writes) != 0:
        violations.append("readout_weight_writes")
    if int(updates_completed) != EXPECTED_UPDATES:
        violations.append(f"updates_completed:{updates_completed}")
    if int(optimizer_steps) != EXPECTED_UPDATES:
        violations.append(f"optimizer_steps:{optimizer_steps}")
    if int(update_backward_calls) != EXPECTED_UPDATE_BACKWARD_CALLS:
        violations.append(f"update_backward_calls:{update_backward_calls}")
    if int(probe_backward_calls) != EXPECTED_PROBE_BACKWARD_CALLS:
        violations.append(f"probe_backward_calls:{probe_backward_calls}")
    if int(update_backward_calls) + int(probe_backward_calls) \
            != EXPECTED_TOTAL_BACKWARD_CALLS:
        violations.append("total_backward_calls")
    if readout_report.get("readouts") != EXPECTED_READOUTS:
        violations.append(f"readouts:{readout_report.get('readouts')}")
    if readout_report.get("total_generations") != EXPECTED_GENERATIONS:
        violations.append("max_scoring_generations")
    if readout_report.get("total_teacher_forced_cells") \
            != EXPECTED_TEACHER_FORCED_CELLS:
        violations.append("max_teacher_forced_cells")
    if readout_report.get("total_scoring_forwards") \
            != EXPECTED_SCORING_FORWARDS:
        violations.append("max_scoring_forwards")
    caps = {
        **gradient.get("caps", {}),
        "readouts": readout_report.get("readouts"),
        "max_readouts": int(limits["max_readouts"]),
        "scoring_generations":
            readout_report.get("total_generations"),
        "max_scoring_generations": int(limits["max_scoring_generations"]),
        "teacher_forced_cells":
            readout_report.get("total_teacher_forced_cells"),
        "max_teacher_forced_cells": int(limits["max_teacher_forced_cells"]),
        "scoring_forwards": readout_report.get("total_scoring_forwards"),
        "max_scoring_forwards": int(limits["max_scoring_forwards"]),
        "readout_weight_writes": int(readout_weight_writes),
    }
    for name in ("scoring_generations", "teacher_forced_cells",
                 "scoring_forwards"):
        maximum = caps[f"max_{name}"]
        if caps[name] is not None and int(caps[name]) > int(maximum):
            violations.append(f"max_{name}")
    if caps["readouts"] is not None \
            and int(caps["readouts"]) > int(caps["max_readouts"]):
        violations.append("max_readouts")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "gradient": gradient,
        "readouts": readout_report,
        "trajectory": trajectory,
        "caps": caps,
        "scope": ("seven update ownership, two probes, five readouts, exact "
                  "T1/T2/T3 inputs, immutable scoring and train-only access"),
    }


def curve_stop_conditions_report(
        *, wall_seconds: float, wall_limit_seconds: float,
        artifact_bytes: int, artifact_limit_bytes: int,
        updates: int, optimizer_steps: int, update_backward_calls: int,
        probe_backward_calls: int, model_loads: int, extension_loads: int,
        extension_writes: int, checkpoint_writes: int, downloads: int,
        cpu_fallback: bool, sealed_evaluation_access: int,
        guard_stop_reason: str | None,
        readouts: int = 0,
        scoring_generations: int = 0,
        teacher_forced_cells: int = 0,
        scoring_forwards: int = 0,
        readout_weight_writes: int = 0,
        max_readouts: int = EXPECTED_READOUTS,
        max_scoring_generations: int = EXPECTED_GENERATIONS,
        max_teacher_forced_cells: int = EXPECTED_TEACHER_FORCED_CELLS,
        max_scoring_forwards: int = EXPECTED_SCORING_FORWARDS,
        ) -> dict[str, Any]:
    """Wrap the gradient accounting so its receipt is labelled ``curve``."""
    base = learn.gradient_stop_conditions_report(
        wall_seconds=wall_seconds,
        wall_limit_seconds=wall_limit_seconds,
        artifact_bytes=artifact_bytes,
        artifact_limit_bytes=artifact_limit_bytes,
        updates=updates, optimizer_steps=optimizer_steps,
        update_backward_calls=update_backward_calls,
        probe_backward_calls=probe_backward_calls,
        model_loads=model_loads, extension_loads=extension_loads,
        extension_writes=extension_writes,
        checkpoint_writes=checkpoint_writes, downloads=downloads,
        cpu_fallback=cpu_fallback,
        sealed_evaluation_access=sealed_evaluation_access,
        guard_stop_reason=guard_stop_reason)
    checks = {
        "max_readouts_exceeded": int(readouts) > int(max_readouts),
        "max_scoring_generations_exceeded":
            int(scoring_generations) > int(max_scoring_generations),
        "max_teacher_forced_cells_exceeded":
            int(teacher_forced_cells) > int(max_teacher_forced_cells),
        "max_scoring_forwards_exceeded":
            int(scoring_forwards) > int(max_scoring_forwards),
        "readout_weight_writes": int(readout_weight_writes) != 0,
    }
    violations = sorted(set(
        list(base.get("violations") or [])
        + [name for name, tripped in checks.items() if tripped]))
    report = {
        **base,
        "format": STOP_FORMAT,
        "mode": MODE,
        "readouts": int(readouts),
        "max_readouts": int(max_readouts),
        "scoring_generations": int(scoring_generations),
        "max_scoring_generations": int(max_scoring_generations),
        "teacher_forced_cells": int(teacher_forced_cells),
        "max_teacher_forced_cells": int(max_teacher_forced_cells),
        "scoring_forwards": int(scoring_forwards),
        "max_scoring_forwards": int(max_scoring_forwards),
        "readout_weight_writes": int(readout_weight_writes),
        "exact_readouts": int(readouts) == EXPECTED_READOUTS,
        "exact_scoring_generations":
            int(scoring_generations) == EXPECTED_GENERATIONS,
        "exact_teacher_forced_cells":
            int(teacher_forced_cells) == EXPECTED_TEACHER_FORCED_CELLS,
        "exact_scoring_forwards":
            int(scoring_forwards) == EXPECTED_SCORING_FORWARDS,
        "within_declared_limits": not violations,
        "violations": violations,
        "scope": ("curve caps are inclusive maxima; reaching a cap before the "
                  "terminal readout fails rather than continuing"),
    }
    return report


def require_curve_stop_conditions(report: Mapping[str, Any]) -> None:
    require(report.get("mode") == MODE,
            "The curve stop receipt was mislabelled")
    require(not report.get("violations"),
            f"Curve stop condition tripped: {report.get('violations')}")
    for flag in ("exact_updates", "exact_optimizer_steps",
                 "exact_update_backward_calls",
                 "exact_probe_backward_calls",
                 "exact_total_backward_calls", "exact_readouts",
                 "exact_scoring_generations",
                 "exact_teacher_forced_cells",
                 "exact_scoring_forwards"):
        require(report.get(flag) is True,
                f"Declared curve accounting differs: {flag}")
    require(int(report.get("readout_weight_writes") or 0) == 0,
            "A curve readout wrote weights")


def claims_block() -> dict[str, Any]:
    return {
        **learn.claims_block(),
        "readout_curve": True,
        "curve_decision_only": True,
        "scope": ("training-only five-readout curve over prepared04 rows; no "
                  "development, held-out or audit payload is read and no "
                  "status authorizes V0, promotion or retry"),
    }
