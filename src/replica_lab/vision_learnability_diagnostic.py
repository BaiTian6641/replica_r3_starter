"""Training-only vision-learnability evidence contracts.

This module owns the model-free half of two predeclared diagnostics:

* ``mode == "ablation"``: a paired, zero-gradient comparison of the pinned
  train04 extension against the same extension with every valid projected patch
  replaced by that image's mean projected patch.  The wrapper preserves each
  image's mean while removing within-image spatial residual, so a changed
  correct/sibling/shuffled gap or greedy output is evidence that spatial
  residual carries first-order information.  The decision is
  ``spatial_contribution_observed``, ``spatial_residual_not_established`` or
  ``baseline_mismatch`` (baseline identity has precedence).
* ``mode == "gradient"``: the smallest full-coverage prefix of the pinned
  prepared04 schedule (seven updates, 28 row slots, a fresh seed-29 projector
  and native fused AdamW) measured only on the training payload.  It checks
  exact projector ownership/finiteness/update evidence, single-hypothesis
  greedy decoding and the predeclared answer/caption train-fit gates.

Nothing here reads, imports or calls a development/held-out loader, runs V0,
exported an extension or claims learned/perception/semantic quality.  The
gradient mode is refused unless an ablation receipt already records
``spatial_contribution_observed``.

Reused contracts: :mod:`replica_lab.vision_collapse_diagnostic` for the stop
set, first-token/NLL math, matched controls, fit-contract rows and coverage;
:mod:`replica_lab.vision_curriculum` for the projector inventory, normalized
answer correctness and caption token F1; and
:mod:`replica_lab.vision_balanced_alignment` for descriptor semantic agreement
and the monotonic data-access order.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .vision_collapse_diagnostic import CONTROL_NAMES, stop_evidence
from .vision_curriculum import PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS


VERSION = "replica.vision_learnability_diagnostic.v1"
CONFIG_FORMAT = "replica.vision_learnability_diagnostic_config.v1"
REPORT_FORMAT = "replica.vision_learnability_diagnostic.report.v1"
DECISION_FORMAT = "replica.vision_learnability_diagnostic.decision.v1"
PAIRED_FORMAT = "replica.vision_learnability_diagnostic.paired_ablation.v1"
UPDATE_FORMAT = "replica.vision_learnability_diagnostic.update.v1"
PREAUDIT_FORMAT = "replica.vision_learnability_diagnostic.precondition.v1"

MODES = ("ablation", "gradient")

FULL_STAGE = "train04_full_patch"
MEAN_STAGE = "train04_mean_only"
BASELINE_STAGE = "train04_extension"
ABLATION_NONE = "none"
ABLATION_MEAN = "mean_projected_patches"

DECISION_OBSERVED = "spatial_contribution_observed"
DECISION_NOT_ESTABLISHED = "spatial_residual_not_established"
DECISION_BASELINE_MISMATCH = "baseline_mismatch"

TARGET_EOS_TOKEN = 1
STOP_TOKENS = (1, 130073)

EXPECTED_ROWS = 26
EXPECTED_ANSWER_ROWS = 14
EXPECTED_CAPTION_ROWS = 12
EXPECTED_RECORDS = 104
EXPECTED_AVAILABLE_RECORDS = 98
EXPECTED_UNAVAILABLE_RECORDS = 6
EXPECTED_TEACHER_FORCED_CELLS = 98
EXPECTED_COMPARISON_CELLS = 72
EXPECTED_GENERATIONS = 98
EXPECTED_VALID_PATCHES = 196
EXPECTED_ANSWER_NEW_TOKENS = 8
EXPECTED_CAPTION_NEW_TOKENS = 32

EXPECTED_UPDATES = 7
EXPECTED_UPDATE_SLOTS = 28
EXPECTED_UPDATE_BACKWARD_CALLS = 28
EXPECTED_PROBE_BACKWARD_CALLS = 2
EXPECTED_TOTAL_BACKWARD_CALLS = 30
EXPECTED_PROBE_ROLES = ("answer", "caption")
EXPECTED_PROJECTOR_TENSORS = 6

ABLATION_TOLERANCE = 1e-06
BASELINE_TOLERANCES = {
    "generated_ids": 0.0,
    "generated_text": 0.0,
    "gold_rank": 0.0,
    "argmax_token": 0.0,
    "loss_tokens": 0.0,
    "gold_logit": 1e-4,
    "opposite_logit": 1e-4,
    "gap": 1e-4,
    "mean_nll": 1e-5,
}

# Exact train-fit gates declared before the run.  Every gate belongs to exactly
# one role; a failure in one role can never be offset by the other role.
GATE_SPECS = {
    "answer_correct_accuracy": ("answer", ">="),
    "answer_correct_margin_sibling": ("answer", ">="),
    "answer_correct_margin_shuffled": ("answer", ">="),
    "answer_correct_margin_noimage": ("answer", ">="),
    "answer_sibling_following_accuracy": ("answer", ">="),
    "answer_prefill_gold_accuracy": ("answer", ">="),
    "answer_correct_vs_sibling_identical_rate": ("answer", "<="),
    "answer_correct_vs_shuffled_identical_rate": ("answer", "<="),
    "answer_correct_unique_texts": ("answer", ">="),
    "caption_token_f1": ("caption", ">="),
    "caption_semantic_f1": ("caption", ">="),
    "caption_following_accuracy": ("caption", ">="),
    "caption_sibling_following_accuracy": ("caption", ">="),
    "caption_sibling_following_semantic_f1": ("caption", ">="),
    "caption_sibling_original_semantic_drop": ("caption", ">="),
    "caption_shuffled_following_accuracy": ("caption", ">="),
    "caption_shuffled_original_semantic_drop": ("caption", ">="),
    "caption_noimage_loss_margin": ("caption", ">="),
    "caption_noimage_semantic_f1": ("caption", "<="),
    "caption_dependence_f1_margin": ("caption", ">="),
    "caption_correct_vs_sibling_identical_rate": ("caption", "<="),
    "caption_correct_unique_texts": ("caption", ">="),
}
ANSWER_GATE_NAMES = tuple(name for name, (role, _) in GATE_SPECS.items()
                          if role == "answer")
CAPTION_GATE_NAMES = tuple(name for name, (role, _) in GATE_SPECS.items()
                           if role == "caption")
GATE_THRESHOLDS = {
    "answer_correct_accuracy": 0.50,
    "answer_correct_margin_sibling": 0.10,
    "answer_correct_margin_shuffled": 0.10,
    "answer_correct_margin_noimage": 0.15,
    "answer_sibling_following_accuracy": 0.50,
    "answer_prefill_gold_accuracy": 0.50,
    "answer_correct_vs_sibling_identical_rate": 0.50,
    "answer_correct_vs_shuffled_identical_rate": 0.50,
    "answer_correct_unique_texts": 3,
    "caption_token_f1": 0.20,
    "caption_semantic_f1": 0.50,
    "caption_following_accuracy": 0.50,
    "caption_sibling_following_accuracy": 0.50,
    "caption_sibling_following_semantic_f1": 0.50,
    "caption_sibling_original_semantic_drop": 0.15,
    "caption_shuffled_following_accuracy": 0.50,
    "caption_shuffled_original_semantic_drop": 0.15,
    "caption_noimage_loss_margin": 0.10,
    "caption_noimage_semantic_f1": 0.25,
    "caption_dependence_f1_margin": 0.0,
    "caption_correct_vs_sibling_identical_rate": 0.50,
    "caption_correct_unique_texts": 3,
}

OBSERVED_IF = {
    "changed_pair_rows_min": 5,
    "answer_gap_reduction_rows_min": 3,
    "gap_reduction_threshold": 0.10,
}
DECISION_PRECEDENCE = [
    DECISION_BASELINE_MISMATCH,
    DECISION_OBSERVED,
    DECISION_NOT_ESTABLISHED,
]
FORBIDDEN_ACCESS = ["evaluation/", "dev", "heldout", "semantic_audit"]
TRAINING_ACCESS_ORDER = ["train", "train_fit"]
EXPECTED_NEW_FILE_PINS = (
    "src/replica_lab/vision_learnability_diagnostic.py",
    "scripts/run_vision_learnability_diagnostic.py",
)


class LearnabilityDiagnosticError(RuntimeError):
    """Raised when a declared learnability contract or pin is violated."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise LearnabilityDiagnosticError(message)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

_COMMON_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "prepared", "source", "prompt_contract", "stop_tokens",
    "target_eos_token", "controls", "limits", "policy",
    "implementation_pins", "output", "report_root", "claims", "limitations",
}
_ABLATION_KEYS = _COMMON_KEYS | {
    "extension", "parent_attempt", "stages", "ablation", "comparison",
    "decision", "expectations",
}
_GRADIENT_KEYS = _COMMON_KEYS | {
    "precondition", "schedule", "initialization", "optimizer", "probes",
    "gates", "access", "v0_gate", "answer_max_new_tokens",
    "caption_max_new_tokens", "expectations",
}


def load_config(path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    return validate_config(config)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(config, Mapping), "The config must be a JSON object")
    mode = config.get("mode")
    require(mode in MODES, f"Unknown learnability mode: {mode!r}")
    expected_keys = _ABLATION_KEYS if mode == "ablation" else _GRADIENT_KEYS
    require(set(config) == expected_keys,
            f"{mode} config fields differ: "
            f"{sorted(set(config) ^ expected_keys)}")
    require(config["format"] == CONFIG_FORMAT, "Config format differs")
    require(config["version"] == VERSION, "Config version differs")
    require(isinstance(config["purpose"], str) and config["purpose"].strip(),
            "Config purpose is missing")
    _validate_prepared(config["prepared"])
    require(isinstance(config["source"], Mapping), "Source pins are missing")
    _validate_prompt_contract(config["prompt_contract"])
    require(tuple(config["stop_tokens"]) == STOP_TOKENS,
            "Stop tokens differ from the native {1, 130073} set")
    require(config["target_eos_token"] == TARGET_EOS_TOKEN,
            "Target EOS differs from </s>")
    require(tuple(config["controls"]) == CONTROL_NAMES,
            "Controls differ from the declared matched set")
    _validate_policy(config["policy"], mode)
    _validate_implementation_pins(config["implementation_pins"])
    claims = config["claims"]
    require(claims.get("promotion") is False
            and claims.get("learned_quality") is False
            and claims.get("perception_quality") is False
            and claims.get("semantic_quality") is False
            and claims.get("overall_baseline_eligible") is False,
            "The diagnostic must not promote learned or semantic quality")
    require(isinstance(config["limitations"], list) and config["limitations"],
            "Limitations are required")
    require(str(config["output"]).startswith("runs/")
            and str(config["report_root"]).startswith("reports/"),
            "Output and report roots must be fresh declared paths")
    if mode == "ablation":
        _validate_ablation(config)
    else:
        _validate_gradient(config)
    return dict(config)


def _validate_prepared(prepared: Mapping[str, Any]) -> None:
    keys = {"root", "protocol", "protocol_sha256", "manifest",
            "manifest_sha256", "training", "training_sha256",
            "training_examples", "answer_examples", "caption_examples",
            "answer_pairs", "caption_pair_ids", "read_only_training",
            "forbidden_access"}
    require(isinstance(prepared, Mapping) and set(prepared) == keys,
            f"Prepared pin fields differ: "
            f"{sorted(set(prepared or {}) ^ keys)}")
    require(prepared["training_examples"] == EXPECTED_ROWS
            and prepared["answer_examples"] == EXPECTED_ANSWER_ROWS
            and prepared["caption_examples"] == EXPECTED_CAPTION_ROWS
            and prepared["answer_pairs"] == 5
            and prepared["caption_pair_ids"] == 7,
            "Prepared training counts differ from the declared 26/14/12")
    require(prepared["read_only_training"] is True,
            "Prepared training data must be read-only")
    require(prepared["forbidden_access"] == FORBIDDEN_ACCESS,
            "Prepared forbidden-access list differs")
    for name in ("protocol_sha256", "manifest_sha256", "training_sha256"):
        _require_sha256(prepared[name], name)


def _validate_prompt_contract(contract: Mapping[str, Any]) -> None:
    require(isinstance(contract, Mapping), "Prompt contract is missing")
    require(contract.get("question_template")
            == "Question: {question}\nReply briefly with the answer alone.",
            "Question template differs")
    require(contract.get("caption_template") == "Describe the image briefly.",
            "Caption template differs")
    require(contract.get("response_only_labels") is True
            and contract.get("label_shift") == 1
            and contract.get("loss") == "mean_response_nll",
            "Response-only label loss contract differs")


def _validate_policy(policy: Mapping[str, Any], mode: str) -> None:
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
    }
    for key, value in required.items():
        require(policy.get(key) == value,
                f"Declared policy {key} differs: {policy.get(key)!r}")
    if mode == "ablation":
        require(policy.get("no_optimizer") is True
                and policy.get("no_backward") is True
                and policy.get("deterministic_controls") is True,
                "Ablation policy must forbid optimizer, backward and control drift")
        require(policy.get("train_only_access") is True,
                "Ablation must declare training-only access")
    else:
        require(policy.get("train_only_access") is True
                and policy.get("no_dev_access") is True
                and policy.get("no_heldout_access") is True
                and policy.get("no_semantic_audit") is True
                and policy.get("deterministic_single_hypothesis") is True,
                "Gradient policy must seal every evaluation split")


def _validate_implementation_pins(pins: Mapping[str, Any]) -> None:
    require(isinstance(pins, Mapping), "Implementation pins are missing")
    require(pins.get("prepared_protocol_pins_checked") is True
            and pins.get("mode") == "explicit_read_only_migration",
            "Explicit read-only implementation migrations are required")
    require(isinstance(pins.get("reason"), str)
            and pins["reason"].strip(),
            "Implementation migration reason is missing")
    migrations = pins.get("migrations")
    require(isinstance(migrations, Mapping) and migrations,
            "Implementation migrations are missing")
    for relative, migration in migrations.items():
        require(isinstance(migration, Mapping)
                and set(migration) == {"parent_sha256", "current_sha256",
                                       "reason"},
                f"Malformed migration: {relative}")
        _require_sha256(migration["parent_sha256"], "parent_sha256")
        _require_sha256(migration["current_sha256"], "current_sha256")
        require(migration["parent_sha256"] != migration["current_sha256"],
                f"Migration does not move: {relative}")
        require(isinstance(migration["reason"], str)
                and migration["reason"].strip(),
                f"Migration lacks a reason: {relative}")
    new_files = pins.get("new_files")
    require(isinstance(new_files, Mapping)
            and set(new_files) == set(EXPECTED_NEW_FILE_PINS),
            "New-file implementation pins differ")
    for relative, digest in new_files.items():
        _require_sha256(digest, relative)


def _require_sha256(value: Any, name: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{name} is not a SHA-256 digest")


def _validate_ablation(config: Mapping[str, Any]) -> None:
    stages = config["stages"]
    require(isinstance(stages, list) and len(stages) == 2,
            "The ablation declares exactly two paired stages")
    first, second = stages
    require(first["name"] == FULL_STAGE and first["kind"] == "extension"
            and first["ablation"] == ABLATION_NONE
            and first["updates"] == 0
            and first.get("extension") == "train04",
            "The first stage must be the unmodified train04 extension")
    require(second["name"] == MEAN_STAGE and second["kind"] == "extension"
            and second["ablation"] == ABLATION_MEAN
            and second["updates"] == 0
            and second.get("extension") == "train04",
            "The second stage must be the mean-only train04 extension")
    require(set(first) == {"name", "kind", "ablation", "updates", "extension",
                           "note"},
            "Stage fields differ")
    require(set(second) == {"name", "kind", "ablation", "updates", "extension",
                            "note"},
            "Stage fields differ")
    ablation = config["ablation"]
    require(set(ablation) == {
        "mode", "valid_only", "projected_space", "tolerance",
        "preserve_mean", "expected_spatial_variance", "apply_to",
        "global_dataset_mean"},
        "Ablation declaration fields differ")
    require(ablation["mode"] == ABLATION_MEAN
            and ablation["valid_only"] is True
            and ablation["projected_space"] == "projected"
            and float(ablation["tolerance"]) == ABLATION_TOLERANCE
            and ablation["preserve_mean"] is True
            and float(ablation["expected_spatial_variance"]) == 0.0
            and list(ablation["apply_to"])
            == ["teacher_forced", "generation", "projected_stats"]
            and ablation["global_dataset_mean"] is False,
            "Ablation declaration differs from the mean-only protocol")
    comparison = config["comparison"]
    require(set(comparison) == {"baseline_stage", "tolerances", "identity"},
            "Comparison contract fields differ")
    require(comparison["baseline_stage"] == BASELINE_STAGE,
            "Baseline stage name differs")
    require(comparison["tolerances"] == BASELINE_TOLERANCES,
            "Baseline comparison tolerances differ")
    require(comparison["identity"] == {
        "records": EXPECTED_RECORDS,
        "available_records": EXPECTED_AVAILABLE_RECORDS,
        "unavailable_siblings": EXPECTED_UNAVAILABLE_RECORDS,
        "teacher_forced_cells": EXPECTED_TEACHER_FORCED_CELLS,
        "comparison_cells": EXPECTED_COMPARISON_CELLS,
        "generations": EXPECTED_GENERATIONS,
    }, "Baseline identity counts differ")
    decision = config["decision"]
    require(set(decision) == {
        "observed_if", "precedence", "authorizes",
        "gradient_authorized_on_observed", "v0_authorized", "v0_run",
        "no_automatic_retry"},
        "Decision contract fields differ")
    require(decision["observed_if"] == OBSERVED_IF,
            "Observed thresholds differ")
    require(decision["precedence"] == DECISION_PRECEDENCE,
            "Decision precedence differs")
    require(decision["authorizes"] == "gradient_learnability_diagnostic_only"
            and decision["gradient_authorized_on_observed"] is True
            and decision["v0_authorized"] is False
            and decision["v0_run"] is False
            and decision["no_automatic_retry"] is True,
            "Decision authorization differs")
    _validate_ablation_extension(config["extension"])
    _validate_ablation_parent(config["parent_attempt"])
    expected = config["expectations"]
    require(expected == {
        "rows": EXPECTED_ROWS,
        "records": EXPECTED_RECORDS,
        "available_records": EXPECTED_AVAILABLE_RECORDS,
        "unavailable_siblings": EXPECTED_UNAVAILABLE_RECORDS,
        "teacher_forced_cells": EXPECTED_TEACHER_FORCED_CELLS,
        "comparison_cells": EXPECTED_COMPARISON_CELLS,
        "generations": EXPECTED_GENERATIONS,
        "fit_answers": EXPECTED_ANSWER_ROWS,
        "fit_captions": EXPECTED_CAPTION_ROWS,
        "valid_patches_per_image": EXPECTED_VALID_PATCHES,
        "prefill_agreement": EXPECTED_GENERATIONS,
        "answer_max_new_tokens": EXPECTED_ANSWER_NEW_TOKENS,
        "caption_max_new_tokens": EXPECTED_CAPTION_NEW_TOKENS,
    }, "Ablation expectations differ")
    limits = config["limits"]
    require(set(limits) == {
        "max_seconds", "artifact_bytes", "max_prompt_tokens",
        "max_sequence_tokens", "max_new_tokens", "caption_max_new_tokens",
        "max_visual_tokens", "updates", "optimizer_steps", "gradients",
        "backward_calls", "checkpoint_writes", "extension_writes", "downloads",
        "model_loads", "extension_loads", "workers"},
        "Ablation limits fields differ")
    require(limits["max_seconds"] == 1200
            and limits["artifact_bytes"] == 134217728
            and limits["updates"] == 0 and limits["optimizer_steps"] == 0
            and limits["gradients"] == 0 and limits["backward_calls"] == 0
            and limits["checkpoint_writes"] == 0
            and limits["extension_writes"] == 0
            and limits["downloads"] == 0
            and limits["model_loads"] == 1
            and limits["extension_loads"] == 1
            and limits["workers"] == 1,
            "Ablation limits must forbid every update/write")
    _validate_token_limits(limits)


def _validate_token_limits(limits: Mapping[str, Any]) -> None:
    require(limits.get("max_prompt_tokens") == 256
            and limits.get("max_sequence_tokens") == 288
            and limits.get("max_new_tokens") == EXPECTED_ANSWER_NEW_TOKENS
            and limits.get("caption_max_new_tokens")
            == EXPECTED_CAPTION_NEW_TOKENS,
            "Token limits differ from 256/288/8/32")


def _validate_ablation_extension(extension: Mapping[str, Any]) -> None:
    require(set(extension) == {
        "path", "run", "updates_consumed", "format", "regime",
        "manifest_sha256", "weights_sha256", "manifest_digest",
        "weights_bytes"},
        "Extension pin fields differ")
    require(extension["run"] == "train04"
            and extension["updates_consumed"] == 24,
            "Ablation requires the pinned 24-update train04 extension")
    require(extension["format"] == "replica.minicpm5.extensions.v4"
            and extension["regime"] == "modules_only"
            and extension["weights_bytes"] == 65688472,
            "Extension format/regime/bytes differ")
    for name in ("manifest_sha256", "weights_sha256", "manifest_digest"):
        _require_sha256(extension[name], name)


def _validate_ablation_parent(parent: Mapping[str, Any]) -> None:
    require(set(parent) == {
        "run", "report", "rows", "cells", "features", "report_sha256",
        "rows_sha256", "cells_sha256", "features_sha256", "baseline_stage",
        "baseline_updates", "records", "available_records",
        "unavailable_siblings", "teacher_forced_cells", "comparison_cells"},
        "Parent-attempt pin fields differ")
    require(parent["baseline_stage"] == BASELINE_STAGE
            and parent["baseline_updates"] == 24,
            "Parent baseline stage differs")
    for name in ("report_sha256", "rows_sha256", "cells_sha256",
                 "features_sha256"):
        _require_sha256(parent[name], name)
    require(parent["records"] == EXPECTED_RECORDS
            and parent["available_records"] == EXPECTED_AVAILABLE_RECORDS
            and parent["unavailable_siblings"] == EXPECTED_UNAVAILABLE_RECORDS
            and parent["teacher_forced_cells"] == EXPECTED_TEACHER_FORCED_CELLS
            and parent["comparison_cells"] == EXPECTED_COMPARISON_CELLS,
            "Parent baseline counts differ")


def _validate_gradient(config: Mapping[str, Any]) -> None:
    precondition = config["precondition"]
    require(set(precondition) == {
        "run", "report", "required_decision", "required_mode",
        "config", "config_sha256", "note"},
        "Precondition fields differ")
    require(precondition["required_decision"] == DECISION_OBSERVED
            and precondition["required_mode"] == "ablation",
            "Gradient mode requires an observed ablation precondition")
    _require_sha256(precondition["config_sha256"], "precondition config_sha256")
    schedule = config["schedule"]
    require(set(schedule) == {
        "source", "source_sha256", "caption_labels",
        "caption_labels_sha256", "prefix_updates", "batch_size",
        "answer_per_update", "caption_per_update", "unique_rows",
        "slots", "expected_answer_rows", "expected_caption_rows",
        "expected_repeated_rows", "smallest_full_coverage_prefix"},
        "Schedule fields differ")
    require(schedule["prefix_updates"] == EXPECTED_UPDATES
            and schedule["batch_size"] == 4
            and schedule["answer_per_update"] == 2
            and schedule["caption_per_update"] == 2
            and schedule["unique_rows"] == EXPECTED_ROWS
            and schedule["slots"] == EXPECTED_UPDATE_SLOTS
            and schedule["expected_answer_rows"] == EXPECTED_ANSWER_ROWS
            and schedule["expected_caption_rows"] == EXPECTED_CAPTION_ROWS
            and schedule["expected_repeated_rows"] == 2
            and schedule["smallest_full_coverage_prefix"] is True,
            "Seven-update full-coverage prefix differs")
    _require_sha256(schedule["source_sha256"], "schedule source_sha256")
    require(schedule["caption_labels"] == "training/train_caption_labels.jsonl",
            "Caption label path differs")
    _require_sha256(schedule["caption_labels_sha256"],
                    "schedule caption_labels_sha256")
    initialization = config["initialization"]
    require(initialization == {
        "init_seed": 29,
        "recipe": "fresh_seeded_normal_std0.02",
        "warm_start": None,
        "forbidden_parent_weights_sha256": [
            "bdc367424b2afcb2e88dba04b6b29495e3a440b5d850f8ae5bc55dbfa428c3a4"],
        "projector_tensors": list(PROJECTOR_TENSORS),
        "parameter_count": PROJECTOR_PARAMETER_COUNT,
    }, "Fresh seed-29 initialization contract differs")
    optimizer = config["optimizer"]
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
    probes = config["probes"]
    require(probes == {
        "count": EXPECTED_PROBE_BACKWARD_CALLS,
        "roles": list(EXPECTED_PROBE_ROLES),
        "backward_only": True,
        "optimizer_steps": 0,
        "weight_change": "none",
    }, "Backward-only probe contract differs")
    require(config["gates"] == GATE_THRESHOLDS,
            "Train-fit gates differ from the declared threshold set")
    require(config["answer_max_new_tokens"] == EXPECTED_ANSWER_NEW_TOKENS
            and config["caption_max_new_tokens"]
            == EXPECTED_CAPTION_NEW_TOKENS,
            "Gradient generation budgets differ")
    limits = config["limits"]
    require(set(limits) == {
        "max_updates", "max_optimizer_steps", "max_update_backward_calls",
        "max_probe_backward_calls", "max_total_backward_calls",
        "max_seconds", "artifact_bytes", "max_prompt_tokens",
        "max_sequence_tokens", "max_new_tokens", "caption_max_new_tokens",
        "model_loads", "extension_loads", "extension_writes",
        "checkpoint_writes", "downloads", "workers", "grad_accum"},
        "Gradient limits fields differ")
    require(limits["max_updates"] == EXPECTED_UPDATES
            and limits["max_optimizer_steps"] == EXPECTED_UPDATES
            and limits["max_update_backward_calls"]
            == EXPECTED_UPDATE_BACKWARD_CALLS
            and limits["max_probe_backward_calls"]
            == EXPECTED_PROBE_BACKWARD_CALLS
            and limits["max_total_backward_calls"]
            == EXPECTED_TOTAL_BACKWARD_CALLS
            and limits["max_seconds"] == 900
            and limits["artifact_bytes"] == 67108864
            and limits["model_loads"] == 1
            and limits["extension_loads"] == 0
            and limits["extension_writes"] == 0
            and limits["checkpoint_writes"] == 0
            and limits["downloads"] == 0
            and limits["workers"] == 1
            and limits["grad_accum"] == 1,
            "Gradient update/cap limits differ")
    _validate_token_limits(limits)
    access = config["access"]
    require(access == {
        "training_paths": [
            "training/alignment.jsonl",
            "training/batches.jsonl",
            "training/train_caption_labels.jsonl",
        ],
        "forbidden_paths": FORBIDDEN_ACCESS,
        "order": TRAINING_ACCESS_ORDER,
        "never_calls": ["_load_dev_v2", "_load_heldout"],
        "advance_after": "updates_complete",
    }, "Training-only access contract differs")
    require(config["v0_gate"] == {
        "planning_authorized_on_pass": True,
        "run_v0": False,
        "requires_separate_predeclared_job": True,
        "heldout_sealed": True,
    }, "V0 gate contract differs")
    expected = config["expectations"]
    require(expected == {
        "rows": EXPECTED_ROWS,
        "generations": EXPECTED_GENERATIONS,
        "updates": EXPECTED_UPDATES,
        "update_slots": EXPECTED_UPDATE_SLOTS,
        "update_backward_calls": EXPECTED_UPDATE_BACKWARD_CALLS,
        "probe_backward_calls": EXPECTED_PROBE_BACKWARD_CALLS,
        "total_backward_calls": EXPECTED_TOTAL_BACKWARD_CALLS,
        "fit_answers": EXPECTED_ANSWER_ROWS,
        "fit_captions": EXPECTED_CAPTION_ROWS,
        "valid_patches_per_image": EXPECTED_VALID_PATCHES,
        "prefill_agreement": EXPECTED_GENERATIONS,
    }, "Gradient expectations differ")


# ---------------------------------------------------------------------------
# Mean-only projected-patch ablation
# ---------------------------------------------------------------------------


def mean_only_projected_patches(projected, valid, *, tolerance: float = 1e-6):
    """Replace valid projected patches by their per-image mean.

    Only valid positions participate in the mean.  Invalid positions are left
    exactly as supplied (the projector masks them to zero) and checked to be
    zero.  The returned tensor has the declared shape/dtype/device; one
    invariant receipt is returned per batch row.
    """
    import torch
    require(torch.is_tensor(projected) and torch.is_tensor(valid),
            "Mean-only ablation requires tensors")
    require(torch.is_floating_point(projected),
            "Projected patches must be floating point")
    require(projected.ndim in (2, 3),
            "Projected patches must be [tokens, width] or [batch, tokens, width]")
    batched = projected.ndim == 3
    rows = projected if batched else projected.unsqueeze(0)
    masks = valid if valid.ndim == 2 else valid.unsqueeze(0)
    require(valid.ndim in (1, 2), "The valid mask must be [tokens] or [batch, tokens]")
    require(masks.shape == rows.shape[:2],
            "The valid mask shape differs from the projected patches")
    require(float(tolerance) > 0, "The ablation tolerance must be positive")
    patched = rows.clone()
    receipts: list[dict[str, Any]] = []
    for index in range(rows.shape[0]):
        mask = masks[index].to(dtype=torch.bool)
        require(bool(mask.any()), "An image has no valid projected patch")
        patches = rows[index][mask]
        mean_patch = patches.mean(dim=0)
        replacements = mean_patch.unsqueeze(0).expand_as(patches)
        patched[index, mask] = replacements
        replaced = patched[index][mask]
        invalid = patched[index][~mask]
        invalid_zero = bool((invalid == 0).all()) if invalid.numel() else True
        deviation = float((replaced - mean_patch).abs().max())
        spatial = float(replaced.var(dim=0, unbiased=False).mean())
        preserved = float((replaced.mean(dim=0) - patches.mean(dim=0)).abs().max())
        finite = bool(torch.isfinite(replaced).all())
        within = (finite and invalid_zero
                  and deviation <= float(tolerance)
                  and spatial <= float(tolerance)
                  and preserved <= float(tolerance))
        receipts.append({
            "row": index,
            "tokens": int(rows.shape[1]),
            "valid_tokens": int(mask.sum()),
            "width": int(rows.shape[2]),
            "max_abs_deviation_from_mean": deviation,
            "spatial_variance": spatial,
            "mean_preserved_max_abs": preserved,
            "invalid_positions_zero": invalid_zero,
            "finite": finite,
            "shape_preserved": True,
            "dtype_preserved": True,
            "device_preserved": True,
            "tolerance": float(tolerance),
            "within_tolerance": within,
        })
    result = patched if batched else patched[0]
    require(result.shape == projected.shape and result.dtype == projected.dtype
            and result.device == projected.device,
            "The mean-only replacement changed shape, dtype or device")
    return result, receipts


class MeanOnlyProjectVisualAblation:
    """Instance-level wrapper around ``adapter.project_visual``.

    Installation is explicit and restore is exact.  The wrapper never mutates
    the immutable :class:`VisualInput`; it verifies feature bytes and the
    mutation guard before and after each projection.
    """

    def __init__(self, adapter, *, tolerance: float = ABLATION_TOLERANCE):
        self.adapter = adapter
        self.tolerance = float(tolerance)
        self._original = None
        self._previous = None
        self.installed = False
        self.calls = 0
        self.receipts: list[dict[str, Any]] = []

    def install(self):
        require(not self.installed, "The mean-only wrapper is already installed")
        self._original = getattr(self.adapter, "project_visual", None)
        require(callable(self._original),
                "The adapter lacks a callable project_visual")
        self._previous = self.adapter.__dict__.get("project_visual", None)
        self.adapter.project_visual = self.__call__
        self.installed = True
        return self

    def restore(self):
        require(self.installed, "The mean-only wrapper is not installed")
        if self._previous is None:
            self.adapter.__dict__.pop("project_visual", None)
        else:
            self.adapter.project_visual = self._previous
        self.installed = False

    def assert_restored(self) -> None:
        require(not self.installed, "The mean-only wrapper is still installed")
        require(self.adapter.__dict__.get("project_visual", None)
                is self._previous, "The adapter projection was not restored")

    def __call__(self, visual):
        import torch
        require(self.installed, "The mean-only wrapper is not installed")
        visual.assert_immutable()
        before_features = visual.features.detach().clone()
        before_valid = visual.valid.detach().clone()
        projected = self._original(visual)
        patched, receipts = mean_only_projected_patches(
            projected, visual.valid, tolerance=self.tolerance)
        require(torch.equal(before_features, visual.features)
                and torch.equal(before_valid, visual.valid),
                "The mean-only wrapper mutated the VisualInput")
        visual.assert_immutable()
        for receipt in receipts:
            receipt["image_sha256"] = visual.image_sha256
            receipt["producer"] = visual.producer
            receipt["task_id"] = visual.task_id
        self.calls += 1
        self.receipts.extend(receipts)
        return patched

    def summary(self) -> dict[str, Any]:
        by_image: dict[str, dict[str, Any]] = {}
        for receipt in self.receipts:
            sha = str(receipt["image_sha256"])
            current = by_image.get(sha)
            if current is None:
                by_image[sha] = {
                    "image_sha256": sha,
                    "valid_tokens": receipt["valid_tokens"],
                    "tokens": receipt["tokens"],
                    "width": receipt["width"],
                    "max_abs_deviation_from_mean":
                        receipt["max_abs_deviation_from_mean"],
                    "spatial_variance": receipt["spatial_variance"],
                    "mean_preserved_max_abs":
                        receipt["mean_preserved_max_abs"],
                    "invalid_positions_zero": receipt["invalid_positions_zero"],
                    "finite": receipt["finite"],
                    "within_tolerance": receipt["within_tolerance"],
                    "calls": 1,
                }
            else:
                current["calls"] += 1
                current["max_abs_deviation_from_mean"] = max(
                    current["max_abs_deviation_from_mean"],
                    receipt["max_abs_deviation_from_mean"])
                current["spatial_variance"] = max(
                    current["spatial_variance"], receipt["spatial_variance"])
                current["mean_preserved_max_abs"] = max(
                    current["mean_preserved_max_abs"],
                    receipt["mean_preserved_max_abs"])
                current["invalid_positions_zero"] = (
                    current["invalid_positions_zero"]
                    and receipt["invalid_positions_zero"])
                current["finite"] = current["finite"] and receipt["finite"]
                current["within_tolerance"] = (
                    current["within_tolerance"]
                    and receipt["within_tolerance"])
        return {
            "installed": self.installed,
            "calls": self.calls,
            "receipts": len(self.receipts),
            "unique_images": len(by_image),
            "tolerance": self.tolerance,
            "valid_tokens": sorted({item["valid_tokens"]
                                    for item in by_image.values()}),
            "max_abs_deviation_from_mean": max(
                (item["max_abs_deviation_from_mean"]
                 for item in by_image.values()), default=None),
            "max_spatial_variance": max(
                (item["spatial_variance"] for item in by_image.values()),
                default=None),
            "max_mean_preserved_deviation": max(
                (item["mean_preserved_max_abs"] for item in by_image.values()),
                default=None),
            "all_within_tolerance": bool(by_image) and all(
                item["within_tolerance"] and item["invalid_positions_zero"]
                and item["finite"] for item in by_image.values()),
            "per_image": {sha: by_image[sha] for sha in sorted(by_image)},
        }


@contextlib.contextmanager
def mean_only_projector(adapter, *, tolerance: float = ABLATION_TOLERANCE):
    ablation = MeanOnlyProjectVisualAblation(adapter, tolerance=tolerance)
    ablation.install()
    try:
        yield ablation
    finally:
        ablation.restore()


def verify_ablation_receipts(summary: Mapping[str, Any], *,
                             expected_image_shas: Sequence[str],
                             expected_valid_tokens: int = EXPECTED_VALID_PATCHES,
                             tolerance: float = 1e-6) -> dict[str, Any]:
    """Verify the wrapper covered every image and kept every invariant."""
    expected = sorted({str(value) for value in expected_image_shas})
    violations: list[str] = []
    require(summary.get("installed") is False,
            "Ablation receipts were taken while the wrapper was still installed")
    if summary.get("all_within_tolerance") is not True:
        violations.append("receipt_invariant_failed")
    if summary.get("unique_images") != len(expected):
        violations.append("image_coverage")
    if sorted(summary.get("per_image", {})) != expected:
        violations.append("image_inventory")
    if list(summary.get("valid_tokens", [])) != [int(expected_valid_tokens)]:
        violations.append("valid_token_count")
    for field in ("max_abs_deviation_from_mean", "max_spatial_variance",
                  "max_mean_preserved_deviation"):
        value = summary.get(field)
        if value is None or float(value) > float(tolerance):
            violations.append(field)
    for sha, item in summary.get("per_image", {}).items():
        if item["valid_tokens"] != int(expected_valid_tokens):
            violations.append(f"valid_tokens:{sha}")
        if not item["within_tolerance"] or not item["invalid_positions_zero"]:
            violations.append(f"invariant:{sha}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "expected_images": len(expected),
        "observed_images": int(summary.get("unique_images") or 0),
        "calls": int(summary.get("calls") or 0),
        "tolerance": float(tolerance),
        "expected_valid_tokens": int(expected_valid_tokens),
        "summary": summary,
    }


def verify_mean_only_image_stats(full_stats: Mapping[str, Mapping[str, Any]],
                                 mean_stats: Mapping[str, Mapping[str, Any]],
                                 *, tolerance: float = 1e-6,
                                 expected_valid_tokens: int
                                 = EXPECTED_VALID_PATCHES
                                 ) -> dict[str, Any]:
    """Check the mean stage genuinely collapsed spatial variation per image."""
    require(set(full_stats) == set(mean_stats),
            "Ablation stage image inventories differ")
    require(bool(full_stats), "Ablation requires image statistics")
    violations: list[str] = []
    images: dict[str, dict[str, Any]] = {}
    for case_id in sorted(full_stats):
        full = full_stats[case_id]
        mean = mean_stats[case_id]
        tokens = int(mean["tokens"])
        mean_patch_delta = max(
            abs(float(a) - float(b))
            for a, b in zip(full["mean_patch"], mean["mean_patch"]))
        rms_delta = abs(float(mean["projected_rms"])
                        - float(mean["mean_vector_rms"]))
        item = {
            "tokens": tokens,
            "full_tokens": int(full["tokens"]),
            "mean_patch_delta": mean_patch_delta,
            "mean_stage_spatial_variance": float(mean["spatial_variance"]),
            "mean_stage_rms_difference": rms_delta,
            "within_tolerance": (
                tokens == expected_valid_tokens
                and int(full["tokens"]) == expected_valid_tokens
                and mean_patch_delta <= tolerance
                and float(mean["spatial_variance"]) <= tolerance
                and rms_delta <= tolerance),
        }
        images[case_id] = item
        if not item["within_tolerance"]:
            violations.append(case_id)
    return {
        "ok": not violations,
        "violations": violations,
        "images": images,
        "tolerance": float(tolerance),
        "expected_valid_tokens": int(expected_valid_tokens),
        "scope": ("per-image projected mean preserved; within-image spatial "
                  "variance collapsed to the pinned tolerance"),
    }


def verify_mean_preservation(
        full_feature: Mapping[str, Any],
        mean_feature: Mapping[str, Any], *,
        tolerance: float = 1e-6) -> dict[str, Any]:
    """Compare stage-invariant projected-space geometry within tolerance."""
    violations: list[str] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for name in ("projected_mean_vector_rms_mean",
                 "within_image_spatial_variance_mean"):
        full_value = float(full_feature[name])
        mean_value = float(mean_feature[name])
        delta = abs(full_value - mean_value)
        within = delta <= float(tolerance)
        if name == "within_image_spatial_variance_mean":
            within = mean_value <= float(tolerance)
        comparisons[name] = {
            "full": full_value, "mean_only": mean_value, "delta": delta,
            "within_tolerance": within}
        if not within:
            violations.append(name)
    encoder_same = (full_feature["encoder_feature_matrix_sha256"]
                    == mean_feature["encoder_feature_matrix_sha256"])
    projected_same = (full_feature["projected_feature_matrix_sha256"]
                      == mean_feature["projected_feature_matrix_sha256"])
    if not encoder_same:
        violations.append("encoder_feature_matrix_sha256")
    return {
        "ok": not violations,
        "violations": violations,
        "comparisons": comparisons,
        "encoder_feature_matrix_sha256_equal": encoder_same,
        "projected_feature_matrix_sha256_equal": projected_same,
        "tolerance": float(tolerance),
        "scope": ("projected mean geometry (PCA share and sibling/cross "
                  "distances) is preserved; only spatial variance is removed"),
    }


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Baseline identity comparison
# ---------------------------------------------------------------------------


def baseline_identity_report(*, full_stage: Mapping[str, Any],
                             baseline_rows: Sequence[Mapping[str, Any]],
                             baseline_cells: Sequence[Mapping[str, Any]],
                             tolerances: Mapping[str, float]
                             = BASELINE_TOLERANCES,
                             expected: Mapping[str, int] | None = None
                             ) -> dict[str, Any]:
    """Compare the full-patch stage with the preserved attempt02 baseline.

    The comparison uses only retained scalars: generated ids/text (exact),
    first-token token/rank/loss-token fields (exact), logits and gaps within
    ``1e-4`` and mean NLL within ``1e-5``.  Attempt02 did not retain full
    logits, so no full-logit reconstruction is claimed.
    """
    require(set(tolerances) == set(BASELINE_TOLERANCES),
            "Baseline tolerances differ from the declared set")
    expected = dict(expected or {
        "records": EXPECTED_RECORDS,
        "available_records": EXPECTED_AVAILABLE_RECORDS,
        "unavailable_siblings": EXPECTED_UNAVAILABLE_RECORDS,
        "teacher_forced_cells": EXPECTED_TEACHER_FORCED_CELLS,
        "comparison_cells": EXPECTED_COMPARISON_CELLS,
        "generations": EXPECTED_GENERATIONS,
    })
    baseline = [row for row in baseline_rows
                if row.get("stage") == BASELINE_STAGE]
    require(baseline, "The baseline rows lack the train04_extension stage")
    base_records = _record_index(baseline)
    records = list(full_stage["records"])
    require(len(records) == expected["records"],
            f"Full stage retained {len(records)} records, expected "
            f"{expected['records']}")
    available = [row for row in records if row.get("available") is True]
    unavailable = [row for row in records if row.get("available") is False]
    require(len(available) == expected["available_records"]
            and len(unavailable) == expected["unavailable_siblings"],
            "Full-stage available/placeholder counts differ")
    require(len(full_stage["cells"]) == expected["comparison_cells"],
            "Full-stage comparison cell count differs")
    require(full_stage.get("teacher_forced_cells")
            == expected["teacher_forced_cells"],
            "Full-stage teacher-forced cell count differs")
    coverage = full_stage["fit_coverage"]
    require(coverage.get("all_rows_checked") is True
            and coverage.get("checked_rows") == EXPECTED_ROWS,
            "Full-stage fit coverage is incomplete")
    mismatches: list[dict[str, Any]] = []
    matched_rows = 0
    matched_unavailable = 0
    for row in records:
        key = (str(row["case_id"]), str(row["control"]))
        base = base_records.get(key)
        if base is None:
            mismatches.append({"key": list(key), "field": "record",
                               "expected": "present", "actual": "missing"})
            continue
        if bool(row.get("available")) != bool(base.get("available")):
            mismatches.append({"key": list(key), "field": "available",
                               "expected": base.get("available"),
                               "actual": row.get("available")})
            continue
        if row.get("available") is False:
            matched_unavailable += 1
            continue
        matched_rows += 1
        _compare_row_identity(row, base, tolerances, mismatches)
    cell_index = {(str(cell["image_role"]), str(cell["target_role"]),
                   str(cell["image_case_id"]), str(cell["target_case_id"])): cell
                  for cell in baseline_cells
                  if cell.get("stage") == BASELINE_STAGE}
    require(len(cell_index) == expected["comparison_cells"],
            f"Baseline comparison cells are {len(cell_index)}, expected "
            f"{expected['comparison_cells']}")
    full_cells = {(str(cell["image_role"]), str(cell["target_role"]),
                   str(cell["image_case_id"]), str(cell["target_case_id"])): cell
                  for cell in full_stage["cells"]}
    matched_cells = 0
    for key in sorted(full_cells):
        base = cell_index.get(key)
        if base is None:
            mismatches.append({"key": list(key), "field": "cell",
                               "expected": "present", "actual": "missing"})
            continue
        matched_cells += 1
        _compare_cell_identity(full_cells[key], base, tolerances, mismatches)
    for key in sorted(set(cell_index) - set(full_cells)):
        mismatches.append({"key": list(key), "field": "cell",
                           "expected": "present", "actual": "missing"})
    checks = {
        "records": len(records),
        "available_records": len(available),
        "unavailable_siblings": len(unavailable),
        "generations": sum(1 for row in available if row.get("generation")),
        "teacher_forced_cells": int(full_stage["teacher_forced_cells"]),
        "comparison_cells": len(full_stage["cells"]),
        "fit_rows_checked": coverage["checked_rows"],
        "matched_rows": matched_rows,
        "matched_unavailable": matched_unavailable,
        "matched_cells": matched_cells,
    }
    for name, wanted in expected.items():
        actual = checks.get(name)
        if actual != int(wanted):
            mismatches.append({"key": [], "field": f"count:{name}",
                               "expected": int(wanted), "actual": actual})
    return {
        "match": not mismatches,
        "mismatches": mismatches[:200],
        "mismatch_count": len(mismatches),
        "checks": checks,
        "expected": {key: int(value) for key, value in expected.items()},
        "baseline_stage": BASELINE_STAGE,
        "tolerances": dict(tolerances),
        "scope": ("paired full-patch identity against the preserved attempt02 "
                  "train04 scalars; full logits were never retained"),
        "baseline_rows": len(baseline),
    }


def _record_index(records: Sequence[Mapping[str, Any]]
                  ) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row["case_id"]), str(row["control"]))
        require(key not in index, f"Duplicate record key: {key}")
        index[key] = row
    return index


def _compare_row_identity(row, base, tolerances, mismatches) -> None:
    key = [str(row["case_id"]), str(row["control"])]
    generation = row.get("generation") or {}
    base_generation = base.get("generation") or {}
    for field in ("generated_ids", "generated_text"):
        actual = generation.get(field)
        expected = base_generation.get(field)
        if actual != expected:
            mismatches.append({"key": key, "field": field,
                               "expected": expected, "actual": actual})
    for field in ("gold_token", "opposite_token", "argmax_token", "gold_rank"):
        actual = (row.get("first_token") or {}).get(field)
        expected = (base.get("first_token") or {}).get(field)
        if actual != expected:
            mismatches.append({"key": key, "field": f"first_token.{field}",
                               "expected": expected, "actual": actual})
    for field in ("loss_tokens",):
        actual = (row.get("positions") or {}).get(field)
        expected = (base.get("positions") or {}).get(field)
        if actual != expected:
            mismatches.append({"key": key, "field": f"positions.{field}",
                               "expected": expected, "actual": actual})
    for field, tolerance in (("gold_logit", tolerances["gold_logit"]),
                             ("opposite_logit", tolerances["opposite_logit"]),
                             ("gap", tolerances["gap"])):
        _compare_float(key, f"first_token.{field}",
                       (row.get("first_token") or {}).get(field),
                       (base.get("first_token") or {}).get(field),
                       tolerance, mismatches)
    _compare_float(key, "positions.mean_nll",
                   (row.get("positions") or {}).get("mean_nll"),
                   (base.get("positions") or {}).get("mean_nll"),
                   tolerances["mean_nll"], mismatches)


def _compare_cell_identity(cell, base, tolerances, mismatches) -> None:
    key = [str(cell["image_role"]), str(cell["target_role"]),
           str(cell["image_case_id"]), str(cell["target_case_id"])]
    for field in ("gold_token", "opposite_token", "argmax_token", "gold_rank"):
        actual = (cell.get("first_token") or {}).get(field)
        expected = (base.get("first_token") or {}).get(field)
        if actual != expected:
            mismatches.append({"key": key, "field": f"first_token.{field}",
                               "expected": expected, "actual": actual})
    for field in ("loss_tokens",):
        actual = (cell.get("positions") or {}).get(field)
        expected = (base.get("positions") or {}).get(field)
        if actual != expected:
            mismatches.append({"key": key, "field": f"positions.{field}",
                               "expected": expected, "actual": actual})
    for field, tolerance in (("gold_logit", tolerances["gold_logit"]),
                             ("opposite_logit", tolerances["opposite_logit"]),
                             ("gap", tolerances["gap"])):
        _compare_float(key, f"first_token.{field}",
                       (cell.get("first_token") or {}).get(field),
                       (base.get("first_token") or {}).get(field),
                       tolerance, mismatches)
    _compare_float(key, "positions.mean_nll",
                   (cell.get("positions") or {}).get("mean_nll"),
                   (base.get("positions") or {}).get("mean_nll"),
                   tolerances["mean_nll"], mismatches)


def _compare_float(key, field, actual, expected, tolerance, mismatches) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            mismatches.append({"key": key, "field": field,
                               "expected": expected, "actual": actual})
        return
    if abs(float(actual) - float(expected)) > float(tolerance):
        mismatches.append({"key": key, "field": field,
                           "expected": expected, "actual": actual,
                           "tolerance": float(tolerance)})


# ---------------------------------------------------------------------------
# Paired ablation metrics and decision
# ---------------------------------------------------------------------------


def paired_ablation_report(*, full_records: Sequence[Mapping[str, Any]],
                           mean_records: Sequence[Mapping[str, Any]],
                           thresholds: Mapping[str, Any] = OBSERVED_IF
                           ) -> dict[str, Any]:
    """Paired correct/sibling/shuffled changes and opposite-gap deltas."""
    require(thresholds == OBSERVED_IF, "Paired thresholds differ")
    full = _record_index(full_records)
    mean = _record_index(mean_records)
    require(set(full) == set(mean),
            "Paired records do not cover the same controls")
    available = {key for key, row in full.items()
                 if row.get("available") is True}
    require(available == {key for key, row in mean.items()
                          if row.get("available") is True},
            "Paired availability differs")
    changed: dict[tuple[str, str], bool] = {}
    for key in sorted(available):
        changed[key] = (list(full[key]["generation"]["generated_ids"])
                        != list(mean[key]["generation"]["generated_ids"]))
    def count_changed(control: str) -> tuple[int, int]:
        keys = [key for key in available if key[1] == control]
        return sum(changed[key] for key in keys), len(keys)
    sibling_changed, sibling_rows = count_changed("sibling")
    shuffled_changed, shuffled_rows = count_changed("shuffled")
    correct_changed, correct_rows = count_changed("correct")
    noimage_changed, noimage_rows = count_changed("noimage")
    answer_gap_rows = []
    reduction = 0
    sign_flips = 0
    for key in sorted(available):
        row = full[key]
        if str(row.get("kind")) != "answer" or key[1] != "correct":
            continue
        full_gap = (row.get("first_token") or {}).get("gap")
        mean_gap = (mean[key].get("first_token") or {}).get("gap")
        if full_gap is None or mean_gap is None:
            continue
        full_gap = float(full_gap)
        mean_gap = float(mean_gap)
        delta = mean_gap - full_gap
        reduced = abs(mean_gap) <= (abs(full_gap)
                                    - float(thresholds["gap_reduction_threshold"]))
        flipped = full_gap * mean_gap < 0
        reduction += int(reduced)
        sign_flips += int(flipped)
        answer_gap_rows.append({
            "case_id": key[0],
            "control": key[1],
            "full_gap": full_gap,
            "mean_only_gap": mean_gap,
            "delta": delta,
            "reduced": reduced,
            "sign_flip": flipped,
        })
    gap_by_control: dict[str, dict[str, Any]] = {}
    for control in CONTROL_NAMES:
        rows = []
        for key in sorted(available):
            row = full[key]
            if str(row.get("kind")) != "answer" or key[1] != control:
                continue
            full_gap = (row.get("first_token") or {}).get("gap")
            mean_gap = (mean[key].get("first_token") or {}).get("gap")
            if full_gap is None or mean_gap is None:
                continue
            full_gap = float(full_gap)
            mean_gap = float(mean_gap)
            rows.append({
                "case_id": key[0],
                "control": control,
                "full_gap": full_gap,
                "mean_only_gap": mean_gap,
                "delta": mean_gap - full_gap,
                "reduced": abs(mean_gap) <= (
                    abs(full_gap)
                    - float(thresholds["gap_reduction_threshold"])),
                "sign_flip": full_gap * mean_gap < 0,
            })
        gap_by_control[control] = {
            "rows": len(rows),
            "full_gap_mean": _mean([row["full_gap"] for row in rows]),
            "mean_only_gap_mean": _mean([row["mean_only_gap"]
                                         for row in rows]),
            "delta_mean": (_mean([row["mean_only_gap"] for row in rows])
                           - _mean([row["full_gap"] for row in rows])
                           if rows else None),
            "reduction_rows": sum(row["reduced"] for row in rows),
            "sign_flip_rows": sum(row["sign_flip"] for row in rows),
            "per_row": rows,
        }
    identical = {}
    for control in ("sibling", "shuffled"):
        keys = [key for key in available
                if key[1] in {"correct", control}
                and str(full[key].get("kind")) == "answer"]
        case_ids = sorted({key[0] for key in keys})
        paired = [case_id for case_id in case_ids
                  if (case_id, "correct") in available
                  and (case_id, control) in available]
        same = [list(full[(case_id, "correct")]["generation"]["generated_ids"])
                == list(full[(case_id, control)]["generation"]["generated_ids"])
                for case_id in paired]
        identical[control] = {
            "rows": len(same),
            "identical_rows": sum(same),
            "identical_rate": _mean(same),
        }
    per_kind = _paired_breakdown(available, changed, full)
    per_control = {
        control: {"changed": changed_count, "rows": rows}
        for control, (changed_count, rows) in {
            "correct": (correct_changed, correct_rows),
            "sibling": (sibling_changed, sibling_rows),
            "shuffled": (shuffled_changed, shuffled_rows),
            "noimage": (noimage_changed, noimage_rows),
        }.items()
    }
    changed_pair_rows = sibling_changed + shuffled_changed
    gap_mean = (sum(row["mean_only_gap"] for row in answer_gap_rows)
                / len(answer_gap_rows) if answer_gap_rows else None)
    gap_full = (sum(row["full_gap"] for row in answer_gap_rows)
                / len(answer_gap_rows) if answer_gap_rows else None)
    return {
        "format": PAIRED_FORMAT,
        "rows": len(available),
        "correct": {"changed": correct_changed, "rows": correct_rows},
        "sibling": {"changed": sibling_changed, "rows": sibling_rows},
        "shuffled": {"changed": shuffled_changed, "rows": shuffled_rows},
        "noimage": {"changed": noimage_changed, "rows": noimage_rows},
        "changed_correct_rows": correct_changed,
        "changed_sibling_rows": sibling_changed,
        "changed_shuffled_rows": shuffled_changed,
        "changed_noimage_rows": noimage_changed,
        "changed_pair_rows": changed_pair_rows,
        "changed_pair_denominator": sibling_rows + shuffled_rows,
        "changed_correct_rows": correct_changed,
        "answer_gap_rows": len(answer_gap_rows),
        "answer_gap_full_mean": gap_full,
        "answer_gap_mean_only_mean": gap_mean,
        "answer_gap_delta_mean": (None if gap_full is None
                                  else gap_mean - gap_full),
        "answer_gap_reduction_rows": reduction,
        "answer_gap_sign_flip_rows": sign_flips,
        "answer_gap_per_row": answer_gap_rows,
        "answer_gap_by_control": gap_by_control,
        "correct_vs_control_identical": identical,
        "per_kind": per_kind,
        "per_control": per_control,
        "thresholds": dict(thresholds),
        "scope": ("paired full/mean-only generated ids and first-token "
                  "opposite gaps; answer gaps use rows whose pair sibling "
                  "exists"),
    }


def _paired_breakdown(available, changed, full) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for key in sorted(available):
        if key[1] not in {"sibling", "shuffled"}:
            continue
        buckets[str(full[key].get("kind"))].append(changed[key])
    return {kind: {"changed": sum(values), "rows": len(values)}
            for kind, values in sorted(buckets.items())}


def ablation_decision(*, baseline_match: bool,
                      paired: Mapping[str, Any],
                      decision: Mapping[str, Any]) -> dict[str, Any]:
    """Apply baseline precedence and the predeclared observed thresholds."""
    require(set(decision) == {
        "observed_if", "precedence", "authorizes",
        "gradient_authorized_on_observed", "v0_authorized", "v0_run",
        "no_automatic_retry"},
        "Decision contract fields differ")
    require(decision["observed_if"] == OBSERVED_IF
            and decision["precedence"] == DECISION_PRECEDENCE,
            "Decision contract differs")
    condition = {
        "changed_pair_rows": int(paired["changed_pair_rows"]),
        "changed_pair_rows_min": int(OBSERVED_IF["changed_pair_rows_min"]),
        "answer_gap_reduction_rows":
            int(paired["answer_gap_reduction_rows"]),
        "answer_gap_reduction_rows_min":
            int(OBSERVED_IF["answer_gap_reduction_rows_min"]),
        "gap_reduction_threshold":
            float(OBSERVED_IF["gap_reduction_threshold"]),
    }
    condition["changed_pair_condition"] = (
        condition["changed_pair_rows"] >= condition["changed_pair_rows_min"])
    condition["gap_reduction_condition"] = (
        condition["answer_gap_reduction_rows"]
        >= condition["answer_gap_reduction_rows_min"])
    if not baseline_match:
        chosen = DECISION_BASELINE_MISMATCH
        reason = ("the full-patch stage did not reproduce the preserved "
                  "attempt02 scalars; no ablation interpretation is allowed")
        sequence = "stopped"
    elif (condition["changed_pair_condition"]
          or condition["gap_reduction_condition"]):
        chosen = DECISION_OBSERVED
        reason = ("paired greedy outputs changed on at least "
                  f"{condition['changed_pair_rows_min']} sibling/shuffled rows "
                  "or the opposite gap shrank on at least "
                  f"{condition['answer_gap_reduction_rows_min']} rows")
        sequence = "gradient_diagnostic_only"
    else:
        chosen = DECISION_NOT_ESTABLISHED
        reason = ("neither changed-pair rows nor opposite-gap reductions met "
                  "the declared minimum; the vision sequence stops")
        sequence = "stopped"
    return {
        "format": DECISION_FORMAT,
        "decision": chosen,
        "baseline_match": bool(baseline_match),
        "condition": condition,
        "thresholds": dict(OBSERVED_IF),
        "precedence": list(DECISION_PRECEDENCE),
        "gradient_diagnostic_authorized": chosen == DECISION_OBSERVED,
        "v0_authorized": False,
        "v0_run": False,
        "sequence": sequence,
        "reason": reason,
        "next_gate": ("a separate predeclared gradient config/job only"
                      if chosen == DECISION_OBSERVED
                      else "no further vision job from this result"),
        "no_automatic_retry": True,
        "scope": ("first-order ablation decision only; observed authorizes "
                  "the gradient diagnostic and never V0"),
    }


# ---------------------------------------------------------------------------
# Seven-update prefix schedule
# ---------------------------------------------------------------------------


def prefix_schedule_report(batches: Sequence[Any], *,
                           prefix_updates: int = EXPECTED_UPDATES,
                           batch_size: int = 4,
                           answer_per_update: int = 2,
                           caption_per_update: int = 2,
                           expected_rows: int = EXPECTED_ROWS,
                           expected_answer_rows: int = EXPECTED_ANSWER_ROWS,
                           expected_caption_rows: int = EXPECTED_CAPTION_ROWS,
                           expected_repeated_rows: int = 2
                           ) -> dict[str, Any]:
    """Audit that the smallest full-coverage prefix is exactly seven updates."""
    require(bool(batches), "The prefix schedule requires batches")
    normalized = [_normalise_batch(batch) for batch in batches]
    all_ids: list[str] = []
    for batch in normalized:
        all_ids.extend(item[0] for item in batch)
    unique_all = set(all_ids)
    require(len(unique_all) == expected_rows,
            f"The pinned schedule covers {len(unique_all)} rows, expected "
            f"{expected_rows}")
    require(len(normalized) >= int(prefix_updates),
            "The pinned schedule is shorter than the declared prefix")
    prefix = normalized[:int(prefix_updates)]
    seen: set[str] = set()
    first_full = None
    updates = []
    counts: Counter = Counter()
    for index, batch in enumerate(prefix, 1):
        require(len(batch) == int(batch_size),
                f"Update {index} has {len(batch)} rows, expected {batch_size}")
        kinds = Counter(kind for _, kind in batch)
        require(kinds.get("answer", 0) == int(answer_per_update)
                and kinds.get("caption", 0) == int(caption_per_update),
                f"Update {index} role mix is {dict(kinds)}, expected "
                f"{answer_per_update} answer + {caption_per_update} caption")
        ids = [item[0] for item in batch]
        require(len(set(ids)) == len(ids),
                f"Update {index} repeats a row slot")
        for example_id, _ in batch:
            counts[example_id] += 1
        seen.update(ids)
        if first_full is None and len(seen) == len(unique_all):
            first_full = index
        updates.append({
            "update": index,
            "example_ids": ids,
            "kinds": dict(sorted(kinds.items())),
            "cumulative_unique_rows": len(seen),
        })
    require(first_full == int(prefix_updates),
            f"The smallest full-coverage prefix is {first_full}, expected "
            f"{prefix_updates}")
    repeated = {example_id: count for example_id, count in counts.items()
                if count > 1}
    slots = [item for batch in prefix for item in batch]
    per_kind = Counter(kind for _, kind in slots)
    unique_by_kind = Counter()
    seen_by_kind: set[str] = set()
    for example_id, kind in slots:
        if example_id not in seen_by_kind:
            seen_by_kind.add(example_id)
            unique_by_kind[kind] += 1
    require(len(seen) == expected_rows
            and unique_by_kind.get("answer") == expected_answer_rows
            and unique_by_kind.get("caption") == expected_caption_rows,
            "Prefix unique coverage differs from 26/14/12")
    require(len(repeated) == int(expected_repeated_rows)
            and all(kind == "caption" for example_id, kind in slots
                    if example_id in repeated),
            f"Prefix repeats {repeated}, expected {expected_repeated_rows} "
            "caption rows")
    return {
        "updates": len(prefix),
        "slots": sum(len(batch) for batch in prefix),
        "batch_size": int(batch_size),
        "answer_per_update": int(answer_per_update),
        "caption_per_update": int(caption_per_update),
        "unique_rows": len(seen),
        "unique_answer_rows": unique_by_kind.get("answer", 0),
        "unique_caption_rows": unique_by_kind.get("caption", 0),
        "first_full_coverage_update": first_full,
        "smallest_full_coverage_prefix": first_full == int(prefix_updates),
        "repeated_example_ids": repeated,
        "per_kind_slots": dict(sorted(per_kind.items())),
        "updates_detail": updates,
        "scope": ("smallest prefix of the pinned prepared04 batch schedule "
                  "covering every training row; no new batch construction"),
    }


def _normalise_batch(batch: Any) -> list[tuple[str, str]]:
    if isinstance(batch, Mapping):
        require("example_ids" in batch, "A batch lacks example_ids")
        items = []
        for example_id in batch["example_ids"]:
            kind = str(example_id).split(":", 1)[0]
            require(kind in {"answer", "caption"},
                    f"Batch example id lacks a role prefix: {example_id}")
            items.append((str(example_id), kind))
        answers = sum(1 for _, kind in items if kind == "answer")
        captions = sum(1 for _, kind in items if kind == "caption")
        require(batch.get("answer_examples") == answers
                and batch.get("caption_examples") == captions,
                "Batch role counts differ from its example ids")
        return items
    require(isinstance(batch, (list, tuple)) and batch,
            "A batch must be a list of rows or a batch mapping")
    items = []
    for row in batch:
        require(isinstance(row, Mapping) and "example_id" in row
                and "kind" in row, "A batch row lacks example_id/kind")
        require(row["kind"] in {"answer", "caption"},
                f"Unknown batch row kind: {row['kind']!r}")
        items.append((str(row["example_id"]), str(row["kind"])))
    return items


def require_prefix_schedule(report: Mapping[str, Any]) -> None:
    require(report.get("updates") == EXPECTED_UPDATES
            and report.get("slots") == EXPECTED_UPDATE_SLOTS
            and report.get("unique_rows") == EXPECTED_ROWS
            and report.get("unique_answer_rows") == EXPECTED_ANSWER_ROWS
            and report.get("unique_caption_rows") == EXPECTED_CAPTION_ROWS
            and report.get("first_full_coverage_update") == EXPECTED_UPDATES
            and report.get("smallest_full_coverage_prefix") is True
            and len(report.get("repeated_example_ids", {}))
            == 2,
            "The seven-update full-coverage prefix contract is violated")


# ---------------------------------------------------------------------------
# Gradient ownership, probe and update evidence
# ---------------------------------------------------------------------------


def require_update_role_mix(record: Mapping[str, Any], *,
                            answer_per_update: int = 2,
                            caption_per_update: int = 2) -> None:
    roles = record.get("roles")
    require(isinstance(roles, Mapping), "An update lacks role counts")
    require(roles.get("answer") == int(answer_per_update)
            and roles.get("caption") == int(caption_per_update),
            f"Update role mix is {roles}, expected "
            f"{answer_per_update}+{caption_per_update}")
    require(len(record.get("example_ids", [])) == 4
            and len(set(record.get("case_ids", []))) == 4,
            "An update does not retain four unique row slots")
    require(record.get("fit_contract_checked") is True
            and len(record.get("fit_contract_rows", [])) == 4,
            "An update skipped its per-row fit contract")
    require(record.get("response_only_labels") is True,
            "An update is not response-only supervised")


def validate_backward_probes(probes: Sequence[Mapping[str, Any]], *,
                             projector_tensors: Sequence[str]
                             = PROJECTOR_TENSORS,
                             expected_probe_backward_calls: int
                             = EXPECTED_PROBE_BACKWARD_CALLS
                             ) -> dict[str, Any]:
    tensors = sorted(projector_tensors)
    violations: list[str] = []
    require(len(probes) == int(expected_probe_backward_calls),
            f"Expected {expected_probe_backward_calls} backward probes, got "
            f"{len(probes)}")
    roles = [str(probe.get("role")) for probe in probes]
    require(sorted(roles) == sorted(EXPECTED_PROBE_ROLES),
            "Backward probes must cover one answer and one caption row")
    backward_calls = 0
    for probe in probes:
        role = str(probe.get("role"))
        prefix = f"probe:{role}"
        if probe.get("backward_only") is not True:
            violations.append(f"{prefix}:not_backward_only")
        if probe.get("probe_applied_update") is not False:
            violations.append(f"{prefix}:applied_update")
        if probe.get("optimizer_steps") != 0:
            violations.append(f"{prefix}:optimizer_step")
        if probe.get("weight_change_names") != []:
            violations.append(f"{prefix}:weight_change")
        if probe.get("projector_state_sha256_before") != probe.get(
                "projector_state_sha256_after"):
            violations.append(f"{prefix}:projector_changed")
        if probe.get("optimizer_state_empty") is not True:
            violations.append(f"{prefix}:optimizer_state")
        if sorted(probe.get("projector_gradient_names", [])) != tensors:
            violations.append(f"{prefix}:gradient_inventory")
        if sorted(probe.get("projector_gradients_nonzero_names", [])) != tensors:
            violations.append(f"{prefix}:nonzero_inventory")
        if probe.get("projector_gradients_finite") is not True:
            violations.append(f"{prefix}:nonfinite")
        if probe.get("nonprojector_gradient_names") != []:
            violations.append(f"{prefix}:nonprojector_gradient")
        loss = probe.get("loss")
        if loss is None or not math.isfinite(float(loss)):
            violations.append(f"{prefix}:loss")
        if probe.get("backward_calls") != 1:
            violations.append(f"{prefix}:backward_calls")
        backward_calls += int(probe.get("backward_calls") or 0)
    if backward_calls != int(expected_probe_backward_calls):
        violations.append("probe_backward_total")
    return {"ok": not violations, "violations": sorted(set(violations)),
            "probes": len(probes), "backward_calls": backward_calls}


def validate_gradient_trace(trace: Sequence[Mapping[str, Any]], *,
                            projector_tensors: Sequence[str]
                            = PROJECTOR_TENSORS,
                            limits: Mapping[str, Any],
                            expected_updates: int = EXPECTED_UPDATES
                            ) -> dict[str, Any]:
    tensors = sorted(projector_tensors)
    violations: list[str] = []
    if len(trace) != int(expected_updates):
        violations.append(f"updates:{len(trace)}")
    changed_counts: Counter = Counter()
    total_backward = 0
    roles = {"answer": 0, "caption": 0}
    for index, record in enumerate(trace, 1):
        prefix = f"update:{index}"
        if record.get("update") != index:
            violations.append(f"{prefix}:sequence")
        require_update_role_mix(record)
        role_counts = record["roles"]
        roles["answer"] += int(role_counts["answer"])
        roles["caption"] += int(role_counts["caption"])
        role_losses = record.get("role_losses") or {}
        for role in ("answer", "caption"):
            losses = role_losses.get(role) or []
            if len(losses) != 2 or not all(
                    value is not None and math.isfinite(float(value))
                    for value in losses):
                violations.append(f"{prefix}:{role}_loss")
        loss = record.get("loss")
        if loss is None or not math.isfinite(float(loss)):
            violations.append(f"{prefix}:loss")
        grad_norm = record.get("grad_norm")
        if grad_norm is None or not math.isfinite(float(grad_norm)):
            violations.append(f"{prefix}:grad_norm")
        update_norm = record.get("projector_update_norm")
        if update_norm is None or not math.isfinite(float(update_norm)) \
                or float(update_norm) <= 0:
            violations.append(f"{prefix}:projector_update_norm")
        if record.get("nonprojector_gradient_names") != []:
            violations.append(f"{prefix}:nonprojector_gradient")
        if record.get("base_stamp_unchanged") is not True:
            violations.append(f"{prefix}:base_changed")
        audit = record.get("gradient_audit") or {}
        original = audit.get("original") or {}
        new = audit.get("new") or {}
        if original.get("present") != [] or original.get("nonzero") != [] \
                or original.get("nonfinite") != []:
            violations.append(f"{prefix}:original_gradient")
        if sorted(new.get("present", [])) != tensors:
            violations.append(f"{prefix}:new_present")
        if sorted(new.get("nonzero", [])) != tensors:
            violations.append(f"{prefix}:new_nonzero")
        if new.get("nonfinite") != []:
            violations.append(f"{prefix}:new_nonfinite")
        changed = record.get("changed_projector_tensors") or []
        if not changed or sorted(changed) != sorted(set(changed)) \
                or not set(changed) <= set(tensors):
            violations.append(f"{prefix}:changed_tensors")
        for name in changed:
            changed_counts[name] += 1
        calls = record.get("backward_calls")
        if calls != 4:
            violations.append(f"{prefix}:backward_calls")
        total_backward += int(calls or 0)
    if total_backward != EXPECTED_UPDATE_BACKWARD_CALLS:
        violations.append(f"update_backward_total:{total_backward}")
    if total_backward > int(limits["max_update_backward_calls"]):
        violations.append("max_update_backward_calls")
    missing = [name for name in tensors if changed_counts[name] == 0]
    if missing:
        violations.append("projector_tensors_never_changed:" + ",".join(missing))
    if roles != {"answer": int(expected_updates) * 2,
                 "caption": int(expected_updates) * 2}:
        violations.append(f"role_slots:{roles}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "updates": len(trace),
        "update_backward_calls": total_backward,
        "role_slots": roles,
        "changed_tensor_counts": dict(sorted(changed_counts.items())),
        "projector_tensors_never_changed": missing,
    }


# ---------------------------------------------------------------------------
# Generation and access evidence
# ---------------------------------------------------------------------------


def generation_hypothesis_report(rows: Sequence[Mapping[str, Any]], *,
                                 expected_records: int = EXPECTED_RECORDS,
                                 expected_available: int
                                 = EXPECTED_AVAILABLE_RECORDS,
                                 expected_unavailable: int
                                 = EXPECTED_UNAVAILABLE_RECORDS,
                                 stop_tokens: Sequence[int] = STOP_TOKENS,
                                 answer_max_new_tokens: int
                                 = EXPECTED_ANSWER_NEW_TOKENS,
                                 caption_max_new_tokens: int
                                 = EXPECTED_CAPTION_NEW_TOKENS
                                 ) -> dict[str, Any]:
    """Verify exactly one greedy hypothesis per available row/control."""
    require(len(rows) == int(expected_records),
            f"Expected {expected_records} generation records, got {len(rows)}")
    require(tuple(sorted(stop_tokens)) == tuple(sorted(STOP_TOKENS)),
            "Generation stop tokens differ")
    seen: set[tuple[str, str]] = set()
    available = 0
    unavailable = 0
    prefill_matches = 0
    violations: list[str] = []
    for row in rows:
        key = (str(row["case_id"]), str(row["control"]))
        require(key not in seen, f"Duplicate generation record: {key}")
        seen.add(key)
        if row.get("available") is False:
            unavailable += 1
            if row.get("generated_ids"):
                violations.append(f"placeholder_generated:{key}")
            continue
        available += 1
        if row.get("hypothesis_count") != 1:
            violations.append(f"hypotheses:{key}")
        ids = list(row.get("generated_ids") or [])
        if not ids:
            violations.append(f"empty_ids:{key}")
            continue
        text = row.get("generated_text")
        if not isinstance(text, str) or not text.strip():
            violations.append(f"empty_text:{key}")
        declared = row.get("stop_tokens")
        if declared is None or sorted(int(value) for value in declared) != sorted(
                int(value) for value in stop_tokens):
            violations.append(f"stop_set:{key}")
        evidence = stop_evidence(ids, stop_tokens)
        if evidence["stop_token_before_final"] is not None:
            violations.append(f"stop_before_final:{key}")
        budget = (int(caption_max_new_tokens) if row.get("kind") == "caption"
                  else int(answer_max_new_tokens))
        if len(ids) > budget:
            violations.append(f"token_budget:{key}")
        if not evidence["ended_by_eos"] and len(ids) != budget:
            violations.append(f"unterminated:{key}")
        if row.get("prefill_argmax_matches_first_token") is not True:
            violations.append(f"prefill_mismatch:{key}")
        else:
            prefill_matches += 1
    if available != int(expected_available):
        violations.append(f"available:{available}")
    if unavailable != int(expected_unavailable):
        violations.append(f"unavailable:{unavailable}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "records": len(rows),
        "available": available,
        "unavailable": unavailable,
        "prefill_agreement": prefill_matches,
        "stop_tokens": sorted(int(value) for value in stop_tokens),
        "single_hypothesis": True,
        "scope": ("exactly one greedy hypothesis per available (row, control); "
                  "declared stop set and prefill argmax agreement"),
    }


class TrainOnlyAccessOrder:
    """The balanced-alignment monotonic order with dev/audit access removed.

    Only ``train`` and ``train_fit`` roles exist here.  Advancing past
    ``train_fit`` raises immediately, so a gradient job cannot open a sealed
    role even by accident.
    """

    def __init__(self) -> None:
        from .vision_balanced_alignment import AccessOrder
        self._order = AccessOrder()
        self.phase = self._order.phase
        self.events: list[dict[str, Any]] = []

    def access(self, role: str) -> None:
        require(role in TRAINING_ACCESS_ORDER,
                f"Only training roles are allowed: {role}")
        self._order.access(role)
        self.events.append({"role": role, "phase": self.phase})

    def advance(self, role: str) -> None:
        require(role == "train_fit" and self.phase == "train",
                f"Cannot advance to {role} from {self.phase}; only "
                "train -> train_fit is allowed")
        self._order.advance(role)
        self.phase = self._order.phase
        self.events.append({"advanced_to": role})

    def snapshot(self) -> dict[str, Any]:
        return {"phase": self.phase,
                "order": list(TRAINING_ACCESS_ORDER),
                "events": [dict(event) for event in self.events]}


def train_only_access_order() -> TrainOnlyAccessOrder:
    return TrainOnlyAccessOrder()


def training_only_access_report(order_snapshot: Mapping[str, Any] | None,
                                accessed_paths: Sequence[str], *,
                                forbidden: Sequence[str] = FORBIDDEN_ACCESS,
                                allowed_roles: Sequence[str]
                                = TRAINING_ACCESS_ORDER
                                ) -> dict[str, Any]:
    """Audit that only training payloads were read and no phase advanced."""
    violations: list[str] = []
    if order_snapshot is not None:
        phase = str(order_snapshot.get("phase"))
        if phase not in set(allowed_roles):
            violations.append(f"phase:{phase}")
        events = order_snapshot.get("events") or []
        for event in events:
            for key in ("role", "advanced_to"):
                value = event.get(key)
                if value is not None and value not in set(allowed_roles):
                    violations.append(f"event:{value}")
    accessed = [str(path) for path in accessed_paths]
    for path in accessed:
        parts = Path(path).parts
        if any(part in {"evaluation", "dev", "heldout"} for part in parts):
            violations.append(f"sealed_path:{path}")
        if "semantic_audit" in path:
            violations.append(f"sealed_path:{path}")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "phase": None if order_snapshot is None
        else order_snapshot.get("phase"),
        "order": None if order_snapshot is None
        else list(order_snapshot.get("order") or []),
        "allowed_roles": list(allowed_roles),
        "forbidden": list(forbidden),
        "accessed_paths": accessed,
        "scope": ("training payloads and their referenced assets only; "
                  "development, held-out and semantic-audit roles are sealed"),
    }


# ---------------------------------------------------------------------------
# Resource/stop accounting
# ---------------------------------------------------------------------------


def gradient_stop_conditions_report(
        *, wall_seconds: float, wall_limit_seconds: float,
        artifact_bytes: int, artifact_limit_bytes: int,
        updates: int, optimizer_steps: int, update_backward_calls: int,
        probe_backward_calls: int, model_loads: int, extension_loads: int,
        extension_writes: int, checkpoint_writes: int, downloads: int,
        cpu_fallback: bool, sealed_evaluation_access: int,
        guard_stop_reason: str | None,
        max_updates: int = EXPECTED_UPDATES,
        max_optimizer_steps: int = EXPECTED_UPDATES,
        max_update_backward_calls: int = EXPECTED_UPDATE_BACKWARD_CALLS,
        max_probe_backward_calls: int = EXPECTED_PROBE_BACKWARD_CALLS,
        max_total_backward_calls: int = EXPECTED_TOTAL_BACKWARD_CALLS
        ) -> dict[str, Any]:
    total_backward = int(update_backward_calls) + int(probe_backward_calls)
    checks = {
        "wall_seconds_exceeded": float(wall_seconds) > float(wall_limit_seconds),
        "artifact_bytes_exceeded":
            int(artifact_bytes) > int(artifact_limit_bytes),
        "max_updates_exceeded": int(updates) > int(max_updates),
        "max_optimizer_steps_exceeded":
            int(optimizer_steps) > int(max_optimizer_steps),
        "max_update_backward_calls_exceeded":
            int(update_backward_calls) > int(max_update_backward_calls),
        "max_probe_backward_calls_exceeded":
            int(probe_backward_calls) > int(max_probe_backward_calls),
        "max_total_backward_calls_exceeded":
            total_backward > int(max_total_backward_calls),
        "extension_writes": int(extension_writes) != 0,
        "checkpoint_writes": int(checkpoint_writes) != 0,
        "downloads": int(downloads) != 0,
        "cpu_fallback": bool(cpu_fallback),
        "sealed_evaluation_access": int(sealed_evaluation_access) != 0,
        "model_loads": int(model_loads) != 1,
        "extension_loads": int(extension_loads) != 0,
        "guard_stop": guard_stop_reason is not None,
    }
    violations = [name for name, tripped in checks.items() if tripped]
    report = {
        "mode": "gradient",
        "wall_seconds": float(wall_seconds),
        "wall_limit_seconds": float(wall_limit_seconds),
        "artifact_bytes": int(artifact_bytes),
        "artifact_limit_bytes": int(artifact_limit_bytes),
        "updates_performed": int(updates),
        "optimizer_steps": int(optimizer_steps),
        "update_backward_calls": int(update_backward_calls),
        "probe_backward_calls": int(probe_backward_calls),
        "total_backward_calls": total_backward,
        "max_updates": int(max_updates),
        "max_optimizer_steps": int(max_optimizer_steps),
        "max_update_backward_calls": int(max_update_backward_calls),
        "max_probe_backward_calls": int(max_probe_backward_calls),
        "max_total_backward_calls": int(max_total_backward_calls),
        "model_loads": int(model_loads),
        "extension_loads": int(extension_loads),
        "extension_writes": int(extension_writes),
        "checkpoint_writes": int(checkpoint_writes),
        "downloads": int(downloads),
        "cpu_fallback": bool(cpu_fallback),
        "sealed_evaluation_access": int(sealed_evaluation_access),
        "guard_stop_reason": guard_stop_reason,
        "exact_updates": int(updates) == EXPECTED_UPDATES,
        "exact_optimizer_steps": int(optimizer_steps) == EXPECTED_UPDATES,
        "exact_update_backward_calls":
            int(update_backward_calls) == EXPECTED_UPDATE_BACKWARD_CALLS,
        "exact_probe_backward_calls":
            int(probe_backward_calls) == EXPECTED_PROBE_BACKWARD_CALLS,
        "exact_total_backward_calls":
            total_backward == EXPECTED_TOTAL_BACKWARD_CALLS,
        "within_declared_limits": not violations,
        "violations": violations,
        "scope": ("caps are inclusive maxima; reaching a cap before scoring "
                  "fails rather than continuing"),
    }
    return report


def require_gradient_stop_conditions(report: Mapping[str, Any]) -> None:
    require(not report.get("violations"),
            f"Gradient stop condition tripped: {report.get('violations')}")
    for flag in ("exact_updates", "exact_optimizer_steps",
                 "exact_update_backward_calls", "exact_probe_backward_calls",
                 "exact_total_backward_calls"):
        require(report.get(flag) is True,
                f"Declared gradient accounting differs: {flag}")


# ---------------------------------------------------------------------------
# Train-fit scoring and gates
# ---------------------------------------------------------------------------


def score_train_fit(*, records: Sequence[Mapping[str, Any]],
                    plan: Mapping[str, Mapping[str, Any]],
                    rows_by_case: Mapping[str, Mapping[str, Any]],
                    caption_labels: Mapping[str, Mapping[str, Any]]
                    ) -> dict[str, Any]:
    """Score answer/caption metrics from direct training-set controls."""
    from .vision_curriculum import answer_correct, caption_token_f1
    from .vision_balanced_alignment import semantic_caption_agreement

    index = _record_index(records)
    require(set(rows_by_case) <= {key[0] for key in index},
            "Scoring records do not cover every training row")
    answer_cases = sorted(case_id for case_id, row in rows_by_case.items()
                          if row["kind"] == "answer")
    caption_cases = sorted(case_id for case_id, row in rows_by_case.items()
                           if row["kind"] == "caption")
    require(len(answer_cases) == EXPECTED_ANSWER_ROWS
            and len(caption_cases) == EXPECTED_CAPTION_ROWS,
            "Scoring row counts differ from 14 answer + 12 caption")
    require(set(caption_cases) == set(caption_labels),
            "Caption labels do not cover every caption row")

    def rec(case_id, control):
        key = (case_id, control)
        require(key in index, f"Missing scoring record: {key}")
        row = index[key]
        require(row.get("available") is not False,
                f"Scoring record is unavailable: {key}")
        return row

    def ids(case_id, control):
        return list(rec(case_id, control)["generated_ids"])

    def text(case_id, control):
        return str(rec(case_id, control)["generated_text"])

    # -- answers ---------------------------------------------------------
    correct_ok = {case_id: answer_correct(
        text(case_id, "correct"), rows_by_case[case_id]["response"])
        for case_id in answer_cases}
    sibling_cases = [case_id for case_id in answer_cases
                     if plan[case_id]["sibling_available"]]
    shuffled_cases = list(answer_cases)
    noimage_cases = list(answer_cases)
    margins = {}
    for control, cases in (("sibling", sibling_cases),
                           ("shuffled", shuffled_cases),
                           ("noimage", noimage_cases)):
        values = [int(correct_ok[case_id])
                  - int(answer_correct(text(case_id, control),
                                       rows_by_case[case_id]["response"]))
                  for case_id in cases]
        margins[control] = (sum(values) / len(values) if values else None)
    sibling_following = [
        answer_correct(text(case_id, "sibling"),
                       rows_by_case[str(plan[case_id]["sibling_case_id"])]
                       ["response"])
        for case_id in sibling_cases]
    prefill = [bool(rec(case_id, "correct")["prefill"]["argmax_is_gold"])
               for case_id in answer_cases]
    sibling_identical = [ids(case_id, "correct") == ids(case_id, "sibling")
                         for case_id in sibling_cases]
    shuffled_identical = [ids(case_id, "correct") == ids(case_id, "shuffled")
                          for case_id in shuffled_cases]
    answer_metrics = {
        "answer_correct_accuracy": _mean([correct_ok[c] for c in answer_cases]),
        "answer_correct_cases": len(answer_cases),
        "answer_correct_margin_sibling": margins["sibling"],
        "answer_sibling_cases": len(sibling_cases),
        "answer_correct_margin_shuffled": margins["shuffled"],
        "answer_correct_margin_noimage": margins["noimage"],
        "answer_shuffled_cases": len(shuffled_cases),
        "answer_noimage_cases": len(noimage_cases),
        "answer_sibling_following_accuracy": _mean(sibling_following),
        "answer_sibling_following_cases": len(sibling_following),
        "answer_prefill_gold_accuracy": _mean(prefill),
        "answer_prefill_cases": len(prefill),
        "answer_correct_vs_sibling_identical_rate":
            _mean(sibling_identical),
        "answer_correct_vs_sibling_cases": len(sibling_identical),
        "answer_correct_vs_shuffled_identical_rate":
            _mean(shuffled_identical),
        "answer_correct_vs_shuffled_cases": len(shuffled_identical),
        "answer_correct_unique_texts": len(
            {text(case_id, "correct") for case_id in answer_cases}),
        "answer_unique_generation_cases": len(answer_cases),
        "scope": ("original-label answer correctness; sibling following is "
                  "scored against the supplied sibling image's response"),
    }

    # -- captions --------------------------------------------------------
    def objects(case_id):
        label = caption_labels[case_id]
        values = label.get("visible_objects")
        require(isinstance(values, list) and values,
                f"Caption label lacks visible objects: {case_id}")
        return [list(item) for item in values]

    own_objects = {case_id: objects(case_id) for case_id in caption_cases}
    for case_id in caption_cases:
        expected_text = str(caption_labels[case_id]["answer"])
        for control in CONTROL_NAMES:
            key = (case_id, control)
            if key not in index or index[key].get("available") is False:
                require(control == "sibling"
                        and not plan[case_id]["sibling_available"],
                        f"Caption scoring record is missing: {key}")
                continue
            record = rec(case_id, control)
            require(record.get("target_text") == expected_text,
                    f"Caption target is not the fixed original: "
                    f"{case_id}/{control}")

    def semantic(generated, expected_objects):
        return semantic_caption_agreement(generated, expected_objects)

    correct_rows = {
        case_id: semantic(text(case_id, "correct"), own_objects[case_id])
        for case_id in caption_cases}
    sibling_semantic = {}
    sibling_following_semantic = {}
    sibling_following_accuracy = {}
    for case_id in caption_cases:
        if not plan[case_id]["sibling_available"]:
            continue
        donor = str(plan[case_id]["sibling_case_id"])
        require(donor in caption_labels,
                f"Sibling caption label is missing: {donor}")
        donor_objects = objects(donor)
        generated = text(case_id, "sibling")
        sibling_semantic[case_id] = semantic(generated, own_objects[case_id])
        following = semantic(generated, donor_objects)
        sibling_following_semantic[case_id] = float(following["f1"])
        sibling_following_accuracy[case_id] = float(
            following["exact_agreement"])
    shuffled_following_accuracy = {}
    for case_id in caption_cases:
        donor = str(plan[case_id]["shuffled_case_id"])
        if donor not in caption_labels:
            continue
        generated = text(case_id, "shuffled")
        following = semantic(generated, objects(donor))
        shuffled_following_accuracy[case_id] = float(
            following["exact_agreement"])
    shuffled_original = {
        case_id: semantic(text(case_id, "shuffled"), own_objects[case_id])
        for case_id in caption_cases}
    noimage_semantic = {
        case_id: semantic(text(case_id, "noimage"), own_objects[case_id])
        for case_id in caption_cases}
    correct_loss = {case_id: float(rec(case_id, "correct")
                                   ["teacher_forced_loss"])
                    for case_id in caption_cases}
    noimage_loss = {case_id: float(rec(case_id, "noimage")
                                   ["teacher_forced_loss"])
                    for case_id in caption_cases}
    sibling_drop = {
        case_id: float(correct_rows[case_id]["f1"])
        - float(sibling_semantic[case_id]["f1"])
        for case_id in sibling_semantic}
    shuffled_drop = {
        case_id: float(correct_rows[case_id]["f1"])
        - float(shuffled_original[case_id]["f1"])
        for case_id in caption_cases}
    dependence_controls = [
        _mean(list(sibling_drop.values())),
        _mean(list(shuffled_drop.values())),
    ]
    caption_metrics = {
        "caption_token_f1": _mean([
            caption_token_f1(text(case_id, "correct"),
                             caption_labels[case_id]["answer"])
            for case_id in caption_cases]),
        "caption_cases": len(caption_cases),
        "caption_semantic_f1": _mean([
            float(correct_rows[case_id]["f1"])
            for case_id in caption_cases]),
        "caption_following_accuracy": _mean([
            float(correct_rows[case_id]["exact_agreement"])
            for case_id in caption_cases]),
        "caption_sibling_following_accuracy":
            _mean(list(sibling_following_accuracy.values())),
        "caption_sibling_following_cases":
            len(sibling_following_accuracy),
        "caption_sibling_following_semantic_f1":
            _mean(list(sibling_following_semantic.values())),
        "caption_sibling_original_semantic_drop":
            _mean(list(sibling_drop.values())),
        "caption_sibling_cases": len(sibling_drop),
        "caption_shuffled_following_accuracy":
            _mean(list(shuffled_following_accuracy.values())),
        "caption_shuffled_following_cases":
            len(shuffled_following_accuracy),
        "caption_shuffled_original_semantic_drop":
            _mean(list(shuffled_drop.values())),
        "caption_shuffled_cases": len(shuffled_drop),
        "caption_noimage_loss_margin": (
            _mean(list(noimage_loss.values()))
            - _mean(list(correct_loss.values()))),
        "caption_noimage_cases": len(caption_cases),
        "caption_noimage_semantic_f1": _mean([
            float(noimage_semantic[case_id]["f1"])
            for case_id in caption_cases]),
        "caption_dependence_f1_margin": _mean(dependence_controls),
        "caption_correct_vs_sibling_identical_rate": _mean([
            ids(case_id, "correct") == ids(case_id, "sibling")
            for case_id in sibling_drop]),
        "caption_correct_vs_sibling_cases": len(sibling_drop),
        "caption_correct_unique_texts": len(
            {text(case_id, "correct") for case_id in caption_cases}),
        "caption_unique_generation_cases": len(caption_cases),
        "scope": ("fixed original caption target for teacher-forced loss and "
                  "original agreement; changed-image following is scored "
                  "against the supplied image's sealed descriptors; "
                  "dependence_f1_margin is the mean original-semantic drop "
                  "across sibling and shuffled"),
    }
    return {"answer": answer_metrics, "caption": caption_metrics}


def _mean(values: Sequence[Any]) -> float | None:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else None


def train_fit_gate_report(*, answer: Mapping[str, Any],
                          caption: Mapping[str, Any],
                          gates: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every declared gate with strict per-role separation."""
    require(gates == GATE_THRESHOLDS,
            "Gate thresholds differ from the declared set")
    per_gate = []
    failures = {"answer": [], "caption": []}
    for name, (role, comparison) in GATE_SPECS.items():
        metrics = answer if role == "answer" else caption
        value = metrics.get(name)
        threshold = float(GATE_THRESHOLDS[name])
        if value is None:
            passed = False
            numeric = None
        else:
            numeric = float(value)
            passed = (numeric >= threshold if comparison == ">="
                      else numeric <= threshold)
        if not passed:
            failures[role].append(name)
        per_gate.append({
            "gate": name,
            "role": role,
            "value": numeric,
            "threshold": threshold,
            "comparison": comparison,
            "passed": passed,
        })
    return {
        "label": "train_fit",
        "contract_version": VERSION,
        "diagnostic_only": True,
        "promotion": False,
        "answer_gate_failures": failures["answer"],
        "caption_gate_failures": failures["caption"],
        "gate_failures": failures["answer"] + failures["caption"],
        "answer_gates_passed": not failures["answer"],
        "caption_gates_passed": not failures["caption"],
        "all_declared_gates_passed": not (failures["answer"]
                                          or failures["caption"]),
        "per_gate": per_gate,
        "scope": ("training-set fit only; answer and caption gate sets are "
                  "independent and neither can offset the other"),
    }


# ---------------------------------------------------------------------------
# Precondition, engineering aggregate and disposition
# ---------------------------------------------------------------------------


def ablation_precondition_report(receipt: Mapping[str, Any], *,
                                 required_decision: str = DECISION_OBSERVED,
                                 required_mode: str = "ablation",
                                 config_sha256: str | None = None
                                 ) -> dict[str, Any]:
    """Validate a preserved ablation receipt before any gradient work."""
    violations: list[str] = []
    if receipt.get("format") != REPORT_FORMAT:
        violations.append("format")
    if receipt.get("mode") != required_mode:
        violations.append("mode")
    decision = receipt.get("decision") or {}
    if decision.get("decision") != required_decision:
        violations.append(f"decision:{decision.get('decision')}")
    if decision.get("gradient_diagnostic_authorized") is not True:
        violations.append("gradient_not_authorized")
    if decision.get("v0_run") is not False:
        violations.append("v0_run")
    baseline = receipt.get("baseline_identity") or {}
    if baseline.get("match") is not True:
        violations.append("baseline_not_matched")
    ablation = receipt.get("ablation_invariants") or {}
    if ablation.get("ok") is not True:
        violations.append("ablation_invariants")
    stop = receipt.get("stop_conditions") or {}
    if stop.get("no_gradients") is not True \
            or stop.get("no_optimizer_steps") is not True \
            or stop.get("no_sealed_evaluation_access") is not True:
        violations.append("stop_conditions")
    if config_sha256 is not None and receipt.get("config_sha256") != config_sha256:
        violations.append("config_sha256")
    return {
        "format": PREAUDIT_FORMAT,
        "ok": not violations,
        "violations": violations,
        "decision": decision.get("decision"),
        "required_decision": required_decision,
        "required_mode": required_mode,
        "source_run": receipt.get("output"),
        "config_sha256": receipt.get("config_sha256"),
        "scope": ("the gradient diagnostic is refused unless this preserved "
                  "ablation receipt records spatial_contribution_observed"),
    }


def gradient_engineering_report(
        *, trace: Sequence[Mapping[str, Any]],
        probes: Sequence[Mapping[str, Any]],
        limits: Mapping[str, Any],
        schedule_report: Mapping[str, Any],
        generation_report: Mapping[str, Any],
        fit_coverage: Mapping[str, Any],
        access_report: Mapping[str, Any],
        precondition: Mapping[str, Any],
        updates_completed: int, optimizer_steps: int,
        update_backward_calls: int, probe_backward_calls: int,
        expected_kinds: Mapping[str, int] | None = None
        ) -> dict[str, Any]:
    """Aggregate every engineering precondition without weakening any gate."""
    violations: list[str] = []
    trace_report = validate_gradient_trace(trace, limits=limits)
    probe_report = validate_backward_probes(probes)
    if not trace_report["ok"]:
        violations.extend(f"trace:{name}"
                          for name in trace_report["violations"])
    if not probe_report["ok"]:
        violations.extend(f"probe:{name}"
                          for name in probe_report["violations"])
    if schedule_report.get("first_full_coverage_update") \
            != EXPECTED_UPDATES:
        violations.append("prefix_coverage")
    if not schedule_report.get("smallest_full_coverage_prefix"):
        violations.append("prefix_not_smallest")
    kinds = dict(expected_kinds or {"answer": EXPECTED_ANSWER_ROWS,
                                    "caption": EXPECTED_CAPTION_ROWS})
    if fit_coverage.get("all_rows_checked") is not True \
            or fit_coverage.get("checked_rows") != EXPECTED_ROWS:
        violations.append("fit_coverage_incomplete")
    for kind, count in kinds.items():
        if fit_coverage.get("per_kind", {}).get(kind) != int(count):
            violations.append(f"fit_coverage_{kind}")
    if not generation_report.get("ok"):
        violations.extend(f"generation:{name}"
                          for name in generation_report["violations"])
    if not access_report.get("ok"):
        violations.extend(f"access:{name}"
                          for name in access_report["violations"])
    if not precondition.get("ok"):
        violations.extend(f"precondition:{name}"
                          for name in precondition["violations"])
    if int(updates_completed) != EXPECTED_UPDATES:
        violations.append(f"updates_completed:{updates_completed}")
    if int(optimizer_steps) != EXPECTED_UPDATES:
        violations.append(f"optimizer_steps:{optimizer_steps}")
    if int(update_backward_calls) > int(limits["max_update_backward_calls"]):
        violations.append("max_update_backward_calls")
    if int(probe_backward_calls) != EXPECTED_PROBE_BACKWARD_CALLS:
        violations.append(f"probe_backward_calls:{probe_backward_calls}")
    total = int(update_backward_calls) + int(probe_backward_calls)
    if total != EXPECTED_TOTAL_BACKWARD_CALLS:
        violations.append(f"total_backward_calls:{total}")
    if total > int(limits["max_total_backward_calls"]):
        violations.append("max_total_backward_calls")
    return {
        "ok": not violations,
        "violations": sorted(set(violations)),
        "trace": trace_report,
        "probes": probe_report,
        "schedule": schedule_report,
        "generation": generation_report,
        "fit_coverage": fit_coverage,
        "access": access_report,
        "precondition": precondition,
        "caps": {
            "updates_completed": int(updates_completed),
            "optimizer_steps": int(optimizer_steps),
            "update_backward_calls": int(update_backward_calls),
            "probe_backward_calls": int(probe_backward_calls),
            "total_backward_calls": total,
            "max_updates": int(limits["max_updates"]),
            "max_optimizer_steps": int(limits["max_optimizer_steps"]),
            "max_update_backward_calls":
                int(limits["max_update_backward_calls"]),
            "max_probe_backward_calls":
                int(limits["max_probe_backward_calls"]),
            "max_total_backward_calls":
                int(limits["max_total_backward_calls"]),
        },
        "scope": ("exact update/probe ownership, finiteness, coverage, "
                  "single-hypothesis decoding and train-only access"),
    }


def gradient_decision(*, precondition_report: Mapping[str, Any],
                      engineering_report: Mapping[str, Any],
                      gate_report: Mapping[str, Any]) -> dict[str, Any]:
    """Terminal disposition; V0 is never run or authorized here."""
    if precondition_report.get("ok") is not True:
        status = "refused_ablation_precondition"
        reason = "the observed-ablation precondition is not satisfied"
        v0 = False
    elif engineering_report.get("ok") is not True:
        status = "stopped_engineering_failure"
        reason = ("a gradient ownership, finiteness, cap, coverage, decoding "
                  "or access invariant failed")
        v0 = False
    elif gate_report.get("all_declared_gates_passed") is not True:
        status = "stopped_before_dev_heldout_v0"
        reason = ("at least one declared train-fit gate failed; development, "
                  "held-out and V0 remain closed")
        v0 = False
    else:
        status = "learnability_pass"
        reason = ("all engineering and train-fit gates passed; V0 planning is "
                  "authorized but V0 is not run in this job")
        v0 = True
    return {
        "format": DECISION_FORMAT + ".gradient",
        "status": status,
        "reason": reason,
        "engineering_ok": engineering_report.get("ok") is True,
        "gate_failures": list(gate_report.get("gate_failures", [])),
        "answer_gate_failures":
            list(gate_report.get("answer_gate_failures", [])),
        "caption_gate_failures":
            list(gate_report.get("caption_gate_failures", [])),
        "learnability_pass": status == "learnability_pass",
        "v0_planning_authorized": v0,
        "v0_run": False,
        "heldout_accessed": False,
        "dev_accessed": False,
        "no_automatic_retry": True,
        "next_gate": ("a separate predeclared V0 job; held-out stays sealed"
                      if v0 else "no further vision job from this result"),
    }


def claims_block() -> dict[str, Any]:
    return {
        "first_order_evidence_only": True,
        "promotion": False,
        "perception_quality": False,
        "semantic_quality": False,
        "overall_baseline_eligible": False,
        "learned_quality": False,
        "v0_run": False,
        "scope": ("training-only learnability evidence over prepared04 "
                  "training rows; no development, held-out or audit payload "
                  "is read and no learned-quality claim is made"),
    }
