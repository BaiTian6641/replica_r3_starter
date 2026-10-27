"""Balanced, semantically checked projector overfit-alignment diagnostic.

This module is CPU-only preparation, protocol, scoring and reload-comparison
logic. It deliberately reuses the sealed source, preprocessing and
projector-only training helpers of the existing vision curriculum. The
held-out counterfactual role is selected from a fresh case namespace, filtered
against every prior evaluation image/scene/case, and never read by training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import struct
import zlib
from typing import Any, Iterable, Mapping, Sequence

from .contracts import fingerprint
from .vision_curriculum import (
    PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS, UNKNOWN_ANSWER,
    _cycle_batches, _write_exact, assign_controls, build_procedural,
    caption_token_f1, clustered_bootstrap,
    load_config as load_curriculum_config, score_procedural, sha256_bytes,
    sha256_file, verify_implementation_pins,
)
from .vision_stage_data import scene_caption
from .visual_data import COLORS, SHAPES, render_scene_png

CONFIG_FORMAT = "replica.vision_balanced_alignment.config.v1"
PROTOCOL_FORMAT = "replica.vision_balanced_alignment.protocol.v1"
MANIFEST_FORMAT = "replica.vision_balanced_alignment.manifest.v1"
SEMANTIC_AUDIT_FORMAT = "replica.vision_balanced_alignment.semantic_audit.v1"
ADMISSION_FORMAT = "replica.vision_balanced_alignment.admission.v1"
RUN_FORMAT = "replica.vision_balanced_alignment.run.v1"
RELOAD_FORMAT = "replica.vision_balanced_alignment.reload.v1"
VERSION = "replica.vision_balanced_alignment.v1"
CONFIG_FORMAT_V2 = "replica.vision_balanced_alignment.config.v2"
PROTOCOL_FORMAT_V2 = "replica.vision_balanced_alignment.protocol.v2"
MANIFEST_FORMAT_V2 = "replica.vision_balanced_alignment.manifest.v2"
RUN_FORMAT_V2 = "replica.vision_balanced_alignment.run.v2"
RELOAD_FORMAT_V2 = "replica.vision_balanced_alignment.reload.v2"
VERSION_V2 = "replica.vision_balanced_alignment.v2"
CAPTION_CONTRACT_VERSION = "caption_target_contract.v2"
CAPTION_SCORE_FORMAT = "replica.vision_balanced_alignment.caption_score.v2"
ANSWER_SCORE_FORMAT = "replica.vision_balanced_alignment.answer_score.v2"
SCORER_CONTRACT_FORMAT = "replica.vision_balanced_alignment.scorer_contracts.v2"
_V2_EXTRA_KEYS = {
    "hypothesis", "preaudit", "fit_contract", "report_root",
    "scorer_contract",
}

COLOR_NAMES = tuple(COLORS)
_ALLOWED_LAYOUTS = ("row_major", "row_major_shift", "column_major",
                    "column_major_shift")
_TRAIN_KEYS = {
    "seed", "init_seed", "init_recipe", "optimizer", "optimizer_class",
    "fused", "lr", "weight_decay", "betas", "eps", "max_grad_norm",
    "max_seconds", "response_only_labels", "warm_start",
    "forbidden_parent_weights_sha256",
}
_LIMIT_KEYS = {
    "max_updates", "max_eval_cases", "max_new_tokens",
    "caption_max_new_tokens", "max_prompt_tokens", "max_sequence_tokens",
    "max_seconds", "max_export_bytes", "max_cache_bytes",
    "max_bootstrap_resamples", "max_control_pool", "max_tool_calls",
    "max_rollout_steps",
}
_RESOURCE_KEYS = {
    "cpu_fallback", "device", "downloads", "driver_limit_bytes",
    "exclusive_mps_worker", "initial_free_admission",
    "minimum_host_free_percent", "sustained_host_samples",
    "sustained_swap_growth_bytes", "workers",
}
_CONFIG_KEYS = {
    "format", "version", "declared_at_utc", "purpose", "project_root",
    "data_root", "seed", "parent_data_config", "parent_protocol",
    "prior_consumed", "procedural_overrides", "selection", "train",
    "schedule", "evaluation", "limits", "stopping", "resource_policy",
    "checkpoint_policy", "implementation_paths", "limitations",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode("utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line]


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _safe_relative(root: Path, value: str) -> Path:
    _require(isinstance(value, str) and value and not Path(value).is_absolute()
             and ".." not in Path(value).parts,
             f"Unsafe relative path: {value!r}")
    return root / value


# ---------------------------------------------------------------------------
# Config and lineage
# ---------------------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(config, Mapping), "Config must be a mapping")
    if config.get("format") == CONFIG_FORMAT_V2:
        return validate_config_v2(config)
    return _validate_config_v1(config)


def _validate_config_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(config) == _CONFIG_KEYS,
             f"Config fields differ: missing={sorted(_CONFIG_KEYS - set(config))} "
             f"extra={sorted(set(config) - _CONFIG_KEYS)}")
    _require(config["format"] == CONFIG_FORMAT, "Unknown config format")
    _require(config["version"] == VERSION, "Unknown config version")
    _require(isinstance(config["declared_at_utc"], str)
             and config["declared_at_utc"], "declared_at_utc is required")
    _require(isinstance(config["purpose"], str) and config["purpose"].strip(),
             "purpose is required")
    _require(type(config["seed"]) is int and config["seed"] >= 0,
             "seed must be a nonnegative integer")
    root = config["data_root"]
    _require(isinstance(root, str) and root and not Path(root).is_absolute()
             and ".." not in Path(root).parts,
             "data_root must be a safe relative path")
    for name in ("parent_data_config", "parent_protocol"):
        pin = config[name]
        _require(isinstance(pin, Mapping)
                 and isinstance(pin.get("path"), str) and pin["path"]
                 and _is_sha256(pin.get("sha256")),
                 f"{name} pin is invalid")
    prior = config["prior_consumed"]
    _require(isinstance(prior, Mapping)
             and set(prior) == {"data_root", "protocol", "manifest", "role"},
             "prior_consumed fields differ")
    _require(isinstance(prior["data_root"], str) and prior["data_root"]
             and not Path(prior["data_root"]).is_absolute(),
             "prior_consumed.data_root must be relative")
    for name in ("protocol", "manifest"):
        pin = prior[name]
        _require(isinstance(pin, Mapping) and set(pin) == {"path", "sha256"}
                 and isinstance(pin["path"], str) and pin["path"]
                 and _is_sha256(pin["sha256"]),
                 f"prior_consumed.{name} pin is invalid")
    _require(isinstance(prior["role"], str) and prior["role"].strip(),
             "prior_consumed role is required")

    overrides = config["procedural_overrides"]
    _require(isinstance(overrides, Mapping)
             and set(overrides) == {"id_prefix", "worlds_per_task",
                                    "renderer", "split_layouts"},
             "procedural_overrides fields differ")
    _require(isinstance(overrides["id_prefix"], str)
             and re.fullmatch(r"[a-z][a-z0-9_]*", overrides["id_prefix"]),
             "id_prefix must be a lowercase identifier")
    _require(type(overrides["worlds_per_task"]) is int
             and 8 <= overrides["worlds_per_task"] <= 12,
             "worlds_per_task must be in [8, 12]")
    renderer = overrides["renderer"]
    _require(isinstance(renderer, Mapping)
             and set(renderer) == {"width", "height"}
             and type(renderer["width"]) is int
             and type(renderer["height"]) is int
             and renderer["width"] >= 64 and renderer["height"] >= 48,
             "renderer override is invalid")
    layouts = overrides["split_layouts"]
    _require(isinstance(layouts, Mapping)
             and set(layouts) == {"train", "dev", "audit"},
             "split_layouts fields differ")
    for split, values in layouts.items():
        _require(isinstance(values, list) and values
                 and all(item in _ALLOWED_LAYOUTS for item in values),
                 f"split_layouts.{split} is invalid")

    selection = config["selection"]
    _require(isinstance(selection, Mapping)
             and set(selection) == {
                 "answer_class", "caption_class", "train_answers_per_class",
                 "train_captions_per_class", "dev_answers_per_class",
                 "dev_captions_per_class", "heldout_answers_per_class",
                 "heldout_captions_per_class", "control_seed_offset",
                 "require_semantic_counterfactual",
                 "require_role_case_disjointness",
                 "exclude_prior_evaluation_hashes", "max_visible_objects"},
             "selection fields differ")
    _require(selection["answer_class"] == "task_and_answer",
             "answer_class differs")
    _require(selection["caption_class"] == "visible_object_count",
             "caption_class differs")
    for name in ("train_answers_per_class", "train_captions_per_class",
                 "dev_answers_per_class", "dev_captions_per_class",
                 "heldout_answers_per_class", "heldout_captions_per_class"):
        _require(type(selection[name]) is int and 1 <= selection[name] <= 8,
                 f"{name} must be in [1, 8]")
    _require(type(selection["control_seed_offset"]) is int
             and selection["control_seed_offset"] >= 1,
             "control_seed_offset must be positive")
    for name in ("require_semantic_counterfactual",
                 "require_role_case_disjointness",
                 "exclude_prior_evaluation_hashes"):
        _require(selection[name] is True, f"{name} must be true")
    _require(type(selection["max_visible_objects"]) is int
             and 1 <= selection["max_visible_objects"] <= 8,
             "max_visible_objects is invalid")

    train = config["train"]
    _require(isinstance(train, Mapping) and set(train) == _TRAIN_KEYS,
             "train fields differ")
    _require(train["init_recipe"] == "fresh_seeded_normal_std0.02",
             "init_recipe differs")
    _require(train["optimizer"] == "adamw_fused"
             and train["optimizer_class"] == "torch.optim.AdamW"
             and train["fused"] is True,
             "Only native fused AdamW is declared")
    _require(isinstance(train["lr"], (int, float)) and not isinstance(train["lr"], bool)
             and float(train["lr"]) > 0, "lr must be positive")
    _require(isinstance(train["max_grad_norm"], (int, float))
             and float(train["max_grad_norm"]) > 0, "max_grad_norm is invalid")
    _require(type(train["max_seconds"]) is int and train["max_seconds"] > 0,
             "max_seconds is invalid")
    _require(train["response_only_labels"] is True,
             "response_only_labels must be true")
    _require(train["warm_start"] is None, "warm_start must be null")
    _require(isinstance(train["forbidden_parent_weights_sha256"], list)
             and all(_is_sha256(item)
                     for item in train["forbidden_parent_weights_sha256"]),
             "forbidden parent hashes are invalid")
    _require(type(train["seed"]) is int and train["seed"] >= 0
             and type(train["init_seed"]) is int and train["init_seed"] >= 0,
             "train seeds are invalid")

    schedule = config["schedule"]
    _require(isinstance(schedule, Mapping)
             and set(schedule) == {"alignment"}
             and isinstance(schedule["alignment"], Mapping),
             "schedule fields differ")
    alignment = schedule["alignment"]
    _require(set(alignment) == {
        "updates", "batch", "caption_per_batch", "answer_per_batch",
        "policy", "role"}, "schedule.alignment fields differ")
    _require(type(alignment["updates"]) is int and alignment["updates"] > 0,
             "updates must be positive")
    _require(type(alignment["batch"]) is int and alignment["batch"] > 0,
             "batch must be positive")
    _require(type(alignment["caption_per_batch"]) is int
             and alignment["caption_per_batch"] > 0
             and type(alignment["answer_per_batch"]) is int
             and alignment["answer_per_batch"] > 0,
             "batch mix must be positive")
    _require(alignment["batch"] == alignment["caption_per_batch"]
             + alignment["answer_per_batch"], "batch mix must sum to batch")
    _require(alignment["policy"] == "deterministic_epoch_cycle",
             "schedule policy differs")
    _require(isinstance(alignment["role"], str) and alignment["role"].strip(),
             "schedule role is required")

    evaluation = config["evaluation"]
    _require(isinstance(evaluation, Mapping), "evaluation must be a mapping")
    required_eval = {
        "dev_answer_controls", "heldout_answer_controls", "caption_controls",
        "train_probe_controls", "question_template", "caption_question",
        "counterfactual_policy", "caption_dependence_policy", "audit_policy",
        "min_correct_accuracy", "min_margin_wrong", "min_margin_shuffled",
        "min_margin_noimage", "min_counterfactual_margin_answerable",
        "min_caption_token_f1", "min_caption_semantic_descriptor_f1",
        "min_caption_dependence_f1_margin",
        "min_caption_dependence_loss_margin", "min_unknown_accuracy",
        "max_false_answer_on_abstain", "max_false_abstention_on_answerable",
        "bootstrap", "gates_scope", "semantic_metric", "caption_metric",
    }
    _require(required_eval <= set(evaluation),
             f"evaluation fields missing: {sorted(required_eval - set(evaluation))}")
    _require(list(evaluation["dev_answer_controls"])
             == ["correct", "wrong", "shuffled", "noimage"],
             "dev answer controls differ")
    _require(list(evaluation["heldout_answer_controls"])
             == ["correct", "wrong", "counterfactual", "shuffled",
                 "noimage"],
             "heldout answer controls differ")
    _require(list(evaluation["caption_controls"])
             == ["correct", "counterfactual", "shuffled", "noimage"],
             "caption controls differ")
    _require(list(evaluation["train_probe_controls"]) == ["correct"],
             "train probe controls differ")
    for name in ("min_correct_accuracy", "min_margin_wrong",
                 "min_margin_shuffled", "min_margin_noimage",
                 "min_counterfactual_margin_answerable",
                 "min_caption_token_f1",
                 "min_caption_semantic_descriptor_f1",
                 "min_caption_dependence_f1_margin",
                 "min_caption_dependence_loss_margin",
                 "min_unknown_accuracy", "max_false_answer_on_abstain",
                 "max_false_abstention_on_answerable"):
        _require(isinstance(evaluation[name], (int, float))
                 and not isinstance(evaluation[name], bool)
                 and 0.0 <= float(evaluation[name]) <= 1.0,
                 f"{name} is outside [0, 1]")
    bootstrap = evaluation["bootstrap"]
    _require(isinstance(bootstrap, Mapping)
             and set(bootstrap) == {"cluster", "confidence", "resamples"},
             "bootstrap fields differ")
    _require(bootstrap["cluster"] == "world", "bootstrap cluster differs")
    _require(type(bootstrap["resamples"]) is int
             and 100 <= bootstrap["resamples"] <= 10_000,
             "bootstrap resamples are outside [100, 10000]")
    _require(0.5 < float(bootstrap["confidence"]) < 1.0,
             "bootstrap confidence is invalid")

    limits = config["limits"]
    _require(isinstance(limits, Mapping) and set(limits) == _LIMIT_KEYS,
             "limits fields differ")
    for name in _LIMIT_KEYS:
        _require(type(limits[name]) is int and limits[name] >= 0,
                 f"limit {name} must be a nonnegative integer")
    _require(limits["max_updates"] >= alignment["updates"],
             "max_updates is below the declared schedule")
    _require(limits["max_bootstrap_resamples"] >= bootstrap["resamples"],
             "max_bootstrap_resamples is below the declared resamples")
    _require(limits["max_seconds"] == train["max_seconds"],
             "max_seconds must match train.max_seconds")
    policy = config["resource_policy"]
    _require(isinstance(policy, Mapping) and set(policy) == _RESOURCE_KEYS,
             "resource policy fields differ")
    _require(policy["cpu_fallback"] is False,
             "CPU fallback is forbidden")
    _require(policy["exclusive_mps_worker"] is True,
             "Exclusive MPS worker is required")
    _require(policy["device"] == "mps", "Only the declared MPS device is allowed")
    _require(policy["downloads"] == 0, "Downloads are forbidden")
    _require(policy["workers"] == 1, "Exactly one worker is allowed")
    _require(policy["driver_limit_bytes"] >= 50 * 1024 ** 3,
             "Driver limit must be at least 50 GiB")
    stopping = config["stopping"]
    _require(isinstance(stopping, Mapping)
             and set(stopping) == {
                 "updates", "wall_seconds", "fail_fast_on_nonfinite_loss",
                 "fail_fast_on_nonprojector_gradient",
                 "fail_fast_on_budget_exhaustion", "early_stopping",
                 "audit_after_export", "reload_must_be_bitwise"},
             "stopping fields differ")
    _require(stopping["updates"] == alignment["updates"],
             "stopping updates differ from the schedule")
    _require(stopping["wall_seconds"] == train["max_seconds"],
             "stopping wall_seconds differs from train.max_seconds")
    _require(stopping["fail_fast_on_nonfinite_loss"] is True
             and stopping["fail_fast_on_nonprojector_gradient"] is True
             and stopping["fail_fast_on_budget_exhaustion"] is True,
             "Fail-fast stopping conditions must be enabled")
    _require(stopping["early_stopping"] is False,
             "Early stopping is not declared")
    _require(stopping["audit_after_export"] is True
             and stopping["reload_must_be_bitwise"] is True,
             "Audit and reload stopping policy differs")
    checkpoint = config["checkpoint_policy"]
    _require(isinstance(checkpoint, Mapping)
             and set(checkpoint) == {
                 "stage_checkpoints", "export", "reload",
                 "keep_all_trajectories", "promotion", "base_copies"},
             "checkpoint_policy fields differ")
    _require(list(checkpoint["stage_checkpoints"]) == ["alignment"],
             "stage_checkpoints differ")
    _require(checkpoint["keep_all_trajectories"] is True,
             "All trajectories must be retained")
    _require(checkpoint["promotion"] is False
             and checkpoint["base_copies"] == 0,
             "This diagnostic cannot promote or copy base weights")
    _require(isinstance(config["implementation_paths"], list)
             and config["implementation_paths"]
             and all(isinstance(item, str) and item
                     for item in config["implementation_paths"]),
             "implementation_paths are required")
    _require(isinstance(config["limitations"], list)
             and len(config["limitations"]) >= 3,
             "At least three limitations are required")
    return dict(config)


def validate_config_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v2 learnability preparation without changing v1 rules."""
    _require(set(config) == _CONFIG_KEYS | _V2_EXTRA_KEYS,
             "v2 config fields differ: "
             f"missing={sorted((_CONFIG_KEYS | _V2_EXTRA_KEYS) - set(config))} "
             f"extra={sorted(set(config) - (_CONFIG_KEYS | _V2_EXTRA_KEYS))}")
    _require(config["format"] == CONFIG_FORMAT_V2, "Unknown config format")
    _require(config["version"] == VERSION_V2, "Unknown config version")
    _require(isinstance(config["hypothesis"], str)
             and config["hypothesis"].strip(), "hypothesis is required")
    report_root = config["report_root"]
    _require(isinstance(report_root, str) and report_root
             and not Path(report_root).is_absolute()
             and ".." not in Path(report_root).parts,
             "report_root must be a safe relative path")
    _require("vba1" != config["procedural_overrides"].get("id_prefix"),
             "v2 requires a fresh case namespace")
    _require(int(config["seed"]) != 20261027,
             "v2 requires a fresh procedural seed")

    priors = config["prior_consumed"]
    _require(isinstance(priors, list) and len(priors) >= 2,
             "v2 prior_consumed must list every consumed evaluation source")
    seen: set[str] = set()
    for entry in priors:
        _require(isinstance(entry, Mapping)
                 and set(entry) == {"data_root", "protocol", "manifest",
                                    "role"},
                 "v2 prior_consumed entry fields differ")
        _require(isinstance(entry["data_root"], str) and entry["data_root"]
                 and not Path(entry["data_root"]).is_absolute()
                 and ".." not in Path(entry["data_root"]).parts,
                 "v2 prior data_root is invalid")
        _require(entry["data_root"] not in seen,
                 "v2 prior sources must be unique")
        seen.add(entry["data_root"])
        for name in ("protocol", "manifest"):
            pin = entry[name]
            _require(isinstance(pin, Mapping)
                     and set(pin) == {"path", "sha256"}
                     and isinstance(pin["path"], str) and pin["path"]
                     and _is_sha256(pin["sha256"]),
                     f"v2 prior {name} pin is invalid")
        _require(isinstance(entry["role"], str) and entry["role"].strip(),
                 "v2 prior role is required")

    preaudit = config["preaudit"]
    _require(isinstance(preaudit, Mapping)
             and set(preaudit) == {
                 "order", "require_train_fit_gates",
                 "require_dev_gates", "on_failure",
                 "reserve_audit_after_dev"},
             "preaudit fields differ")
    _require(list(preaudit["order"])
             == ["train_fit", "dev", "heldout"],
             "preaudit order must be train_fit, dev, heldout")
    _require(preaudit["require_train_fit_gates"] is True
             and preaudit["require_dev_gates"] is True
             and preaudit["reserve_audit_after_dev"] is True,
             "preaudit must require train-fit and dev gates first")
    _require(preaudit["on_failure"] == "stop_before_heldout_access",
             "preaudit failure must stop before held-out access")

    fit = config["fit_contract"]
    _require(isinstance(fit, Mapping)
             and set(fit) == {
                 "question_template", "caption_template",
                 "prompt_labels_masked", "response_includes_eos",
                 "pad_tokens_forbidden", "loss", "require_finite_loss",
                 "require_all_projector_gradients",
                 "require_nonzero_projector_gradients",
                 "response_only_labels"},
             "fit_contract fields differ")
    _require(fit["question_template"]
             == "Question: {question}\nReply briefly with the answer alone.",
             "fit_contract question template differs")
    _require(fit["caption_template"] == "Describe the image briefly.",
             "fit_contract caption template differs")
    for name in ("prompt_labels_masked", "response_includes_eos",
                 "pad_tokens_forbidden", "require_finite_loss",
                 "require_all_projector_gradients",
                 "require_nonzero_projector_gradients",
                 "response_only_labels"):
        _require(fit[name] is True, f"fit_contract.{name} must be true")
    _require(fit["loss"] == "mean_response_nll", "fit_contract.loss differs")

    scorer = config["scorer_contract"]
    _require(isinstance(scorer, Mapping)
             and set(scorer) == {
                 "version", "fixed_original_target", "following_target",
                 "bind_control_assignment", "validate_image_identity",
                 "validate_generation_target_fields",
                 "positive_control", "negative_controls"},
             "scorer_contract fields differ")
    _require(scorer["version"] == CAPTION_CONTRACT_VERSION,
             "scorer contract version differs")
    _require(scorer["fixed_original_target"] is True
             and scorer["following_target"] is True
             and scorer["bind_control_assignment"] is True
             and scorer["validate_image_identity"] is True
             and scorer["validate_generation_target_fields"] is True,
             "v2 scorer contract must validate targets and identity")
    _require(scorer["positive_control"] == "perfect_image_following",
             "positive control differs")
    _require(list(scorer["negative_controls"]) == [
        "constant_caption", "image_ignoring", "shuffled",
        "malformed", "target_tampering"],
        "negative control inventory differs")

    evaluation = config["evaluation"]
    v2_keys = {
        "min_caption_following_accuracy",
        "min_caption_following_semantic_f1",
        "min_caption_original_agreement_drop",
        "max_noimage_original_semantic_f1",
        "min_noimage_loss_margin",
        "min_counterfactual_following_accuracy",
        "min_following_accuracy_wrong",
        "min_following_accuracy_shuffled",
        "min_train_fit_answer_accuracy",
        "min_train_fit_caption_following_accuracy",
        "caption_score_format",
        "answer_score_format",
    }
    _require(v2_keys <= set(evaluation),
             f"v2 evaluation fields missing: "
             f"{sorted(v2_keys - set(evaluation))}")
    for name in ("min_caption_following_accuracy",
                 "min_caption_following_semantic_f1",
                 "min_caption_original_agreement_drop",
                 "max_noimage_original_semantic_f1",
                 "min_noimage_loss_margin",
                 "min_counterfactual_following_accuracy",
                 "min_following_accuracy_wrong",
                 "min_following_accuracy_shuffled",
                 "min_train_fit_answer_accuracy",
                 "min_train_fit_caption_following_accuracy"):
        _require(isinstance(evaluation[name], (int, float))
                 and not isinstance(evaluation[name], bool)
                 and 0.0 <= float(evaluation[name]) <= 1.0,
                 f"{name} is outside [0, 1]")
    _require(evaluation["caption_score_format"] == CAPTION_SCORE_FORMAT,
             "caption score format differs")
    _require(evaluation["answer_score_format"] == ANSWER_SCORE_FORMAT,
             "answer score format differs")

    alignment = config["schedule"]["alignment"]
    _require(not (int(alignment["updates"]) == 10
                  and int(alignment["batch"]) == 6
                  and float(config["train"]["lr"]) == 0.0001),
             "v2 must not repeat the failed v1 schedule")

    # Validate every shared field with the preserved v1 rules, without
    # altering or reinterpreting v1 semantics.
    shadow = {key: config[key] for key in _CONFIG_KEYS}
    shadow["format"] = CONFIG_FORMAT
    shadow["version"] = VERSION
    shadow["prior_consumed"] = {
        "data_root": priors[0]["data_root"],
        "protocol": dict(priors[0]["protocol"]),
        "manifest": dict(priors[0]["manifest"]),
        "role": priors[0]["role"],
    }
    shadow_eval = dict(shadow["evaluation"])
    for name in ("min_caption_dependence_f1_margin",
                 "min_caption_dependence_loss_margin"):
        shadow_eval.setdefault(name, 0.0)
    shadow_eval.setdefault("min_counterfactual_margin_answerable", 0.1)
    shadow["evaluation"] = shadow_eval
    _validate_config_v1(shadow)
    return dict(config)


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text()))


def load_prior_evaluation(project_root: str | Path,
                          config: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the prior evaluation identities used for freshness exclusion."""
    if isinstance(config.get("prior_consumed"), list):
        return _load_prior_evaluations_v2(project_root,
                                          config["prior_consumed"])
    root = Path(project_root)
    prior = config["prior_consumed"]
    data_root = _safe_relative(root, prior["data_root"])
    protocol_path = _safe_relative(root, prior["protocol"]["path"])
    manifest_path = _safe_relative(root, prior["manifest"]["path"])
    _require(protocol_path.is_file(), "Prior protocol is missing")
    _require(manifest_path.is_file(), "Prior manifest is missing")
    _require(sha256_file(protocol_path) == prior["protocol"]["sha256"],
             "Prior protocol bytes differ")
    _require(sha256_file(manifest_path) == prior["manifest"]["sha256"],
             "Prior manifest bytes differ")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("format") ==
             "replica.vision_stage_diagnostic.manifest.v1",
             "Prior manifest format differs")
    case_ids: set[str] = set()
    image_hashes: set[str] = set()
    scene_hashes: set[str] = set()
    files_used = []
    for name in ("dev_cases.jsonl", "audit_cases.jsonl",
                 "caption_dev_cases.jsonl", "caption_audit_cases.jsonl"):
        path = data_root / "evaluation" / name
        _require(path.is_file(), f"Prior evaluation file is missing: {name}")
        rows = _read_jsonl(path)
        _require(rows, f"Prior evaluation file is empty: {name}")
        files_used.append({"path": f"evaluation/{name}",
                           "sha256": sha256_file(path), "rows": len(rows)})
        for row in rows:
            _require(isinstance(row.get("case_id"), str)
                     and _is_sha256(row.get("image_sha256"))
                     and _is_sha256(row.get("scene_sha256")),
                     f"Prior evaluation row is invalid: {name}")
            case_ids.add(row["case_id"])
            image_hashes.add(row["image_sha256"])
            scene_hashes.add(row["scene_sha256"])
    all_asset_hashes = {entry["sha256"] for entry in manifest["files"].values()
                        if _is_sha256(entry.get("sha256"))}
    namespaces = sorted({case_id.split("-")[0] for case_id in case_ids})
    return {
        "case_ids": case_ids,
        "image_sha256": image_hashes,
        "scene_sha256": scene_hashes,
        "all_asset_sha256": all_asset_hashes,
        "namespaces": namespaces,
        "counts": {
            "cases": len(case_ids), "images": len(image_hashes),
            "scenes": len(scene_hashes),
            "all_manifest_assets": len(all_asset_hashes),
        },
        "files": files_used,
        "protocol_sha256": prior["protocol"]["sha256"],
        "manifest_sha256": prior["manifest"]["sha256"],
    }


def _load_prior_source_v2(root: Path, entry: Mapping[str, Any]
                          ) -> dict[str, Any]:
    data_root = _safe_relative(root, entry["data_root"])
    protocol_path = _safe_relative(root, entry["protocol"]["path"])
    manifest_path = _safe_relative(root, entry["manifest"]["path"])
    _require(protocol_path.is_file(), "Prior protocol is missing")
    _require(manifest_path.is_file(), "Prior manifest is missing")
    _require(sha256_file(protocol_path) == entry["protocol"]["sha256"],
             "Prior protocol bytes differ")
    _require(sha256_file(manifest_path) == entry["manifest"]["sha256"],
             "Prior manifest bytes differ")
    manifest = json.loads(manifest_path.read_text())
    manifest_format = manifest.get("format")
    if manifest_format == "replica.vision_stage_diagnostic.manifest.v1":
        names = ("dev_cases.jsonl", "audit_cases.jsonl",
                 "caption_dev_cases.jsonl", "caption_audit_cases.jsonl")
    elif manifest_format in (MANIFEST_FORMAT, MANIFEST_FORMAT_V2):
        names = ("dev_answer_cases.jsonl", "heldout_answer_cases.jsonl",
                 "dev_caption_cases.jsonl", "heldout_caption_cases.jsonl")
    else:
        raise ValueError(f"Unsupported prior manifest format: {manifest_format}")
    case_ids: set[str] = set()
    image_hashes: set[str] = set()
    scene_hashes: set[str] = set()
    files_used = []
    for name in names:
        path = data_root / "evaluation" / name
        _require(path.is_file(), f"Prior evaluation file is missing: {name}")
        rows = _read_jsonl(path)
        _require(rows, f"Prior evaluation file is empty: {name}")
        files_used.append({"path": f"{entry['data_root']}/evaluation/{name}",
                           "sha256": sha256_file(path), "rows": len(rows)})
        for row in rows:
            _require(isinstance(row.get("case_id"), str)
                     and _is_sha256(row.get("image_sha256"))
                     and _is_sha256(row.get("scene_sha256")),
                     f"Prior evaluation row is invalid: {name}")
            case_ids.add(row["case_id"])
            image_hashes.add(row["image_sha256"])
            scene_hashes.add(row["scene_sha256"])
    all_asset_hashes = {value["sha256"]
                        for value in manifest["files"].values()
                        if _is_sha256(value.get("sha256"))}
    return {
        "data_root": entry["data_root"],
        "case_ids": case_ids,
        "image_sha256": image_hashes,
        "scene_sha256": scene_hashes,
        "all_asset_sha256": all_asset_hashes,
        "namespaces": sorted({case_id.split("-")[0]
                              for case_id in case_ids}),
        "files": files_used,
        "protocol_sha256": entry["protocol"]["sha256"],
        "manifest_sha256": entry["manifest"]["sha256"],
        "manifest_format": manifest_format,
    }


def _load_prior_evaluations_v2(project_root: str | Path,
                               priors: Sequence[Mapping[str, Any]]
                               ) -> dict[str, Any]:
    root = Path(project_root)
    sources = [_load_prior_source_v2(root, entry) for entry in priors]
    case_ids: set[str] = set()
    image_hashes: set[str] = set()
    scene_hashes: set[str] = set()
    all_asset_hashes: set[str] = set()
    namespaces: set[str] = set()
    files_used = []
    for source in sources:
        case_ids |= source["case_ids"]
        image_hashes |= source["image_sha256"]
        scene_hashes |= source["scene_sha256"]
        all_asset_hashes |= source["all_asset_sha256"]
        namespaces |= set(source["namespaces"])
        files_used.extend(source["files"])
    return {
        "case_ids": case_ids,
        "image_sha256": image_hashes,
        "scene_sha256": scene_hashes,
        "all_asset_sha256": all_asset_hashes,
        "namespaces": sorted(namespaces),
        "counts": {
            "cases": len(case_ids), "images": len(image_hashes),
            "scenes": len(scene_hashes),
            "all_manifest_assets": len(all_asset_hashes),
            "sources": len(sources),
        },
        "files": files_used,
        "sources": [{
            "data_root": source["data_root"],
            "protocol_sha256": source["protocol_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "manifest_format": source["manifest_format"],
            "case_ids": len(source["case_ids"]),
            "images": len(source["image_sha256"]),
            "scenes": len(source["scene_sha256"]),
            "files": source["files"],
        } for source in sources],
        "protocol_sha256": fingerprint(
            [source["protocol_sha256"] for source in sources]),
        "manifest_sha256": fingerprint(
            [source["manifest_sha256"] for source in sources]),
    }


# ---------------------------------------------------------------------------
# Semantic image/caption checks
# ---------------------------------------------------------------------------


_COLOR_SET = frozenset(COLOR_NAMES)
_SHAPE_SET = frozenset(SHAPES)
_TOKEN = re.compile(r"[a-z0-9]+")


def question_leaks(question: str, expected: str) -> bool:
    """Reject a gold answer that appears as a standalone question token."""
    if not expected or not str(expected).strip():
        return False
    pattern = r"(?<!\w)" + re.escape(str(expected).casefold()) + r"(?!\w)"
    return re.search(pattern, str(question).casefold()) is not None


def visible_descriptors(scene: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(obj["color"], obj["shape"]) for obj in scene["objects"]
            if obj.get("role") != "occluded"]


def occluded_descriptors(scene: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(obj["color"], obj["shape"]) for obj in scene["objects"]
            if obj.get("role") == "occluded"]


def caption_descriptors(text: str) -> list[tuple[str, str]]:
    """Parse only adjacent colour/shape descriptor pairs from a caption."""
    tokens = _TOKEN.findall(str(text).casefold())
    descriptors = []
    index = 0
    while index + 1 < len(tokens):
        color, shape = tokens[index], tokens[index + 1]
        if color in _COLOR_SET and shape in _SHAPE_SET:
            descriptors.append((color, shape))
            index += 2
        else:
            index += 1
    return descriptors


def verify_caption_semantics(caption: str, scene: Mapping[str, Any], *,
                             require_exact: bool = True) -> dict[str, Any]:
    expected = Counter(visible_descriptors(scene))
    predicted = Counter(caption_descriptors(caption))
    occluded = Counter(occluded_descriptors(scene))
    if require_exact:
        _require(predicted == expected,
                 "Caption descriptors differ from visible objects: "
                 f"predicted={dict(predicted)} expected={dict(expected)}")
    overlap = sum((predicted & occluded).values())
    _require(overlap == 0, "Caption names an occluded descriptor")
    return {
        "caption": caption,
        "visible_objects": [list(item) for item in expected],
        "parsed_descriptors": [list(item) for item in predicted],
        "occluded_descriptors": [list(item) for item in occluded],
        "exact_match": predicted == expected,
        "occluded_named": overlap,
    }


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distance_left = abs(estimate - left)
    distance_above = abs(estimate - above)
    distance_upper_left = abs(estimate - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def decode_png_rgb8(payload: bytes) -> tuple[int, int, int, bytes]:
    """Minimal standard-library PNG decoder for the declared RGB8 renderer."""
    _require(payload[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG payload")
    position = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while position < len(payload):
        _require(position + 8 <= len(payload), "Truncated PNG chunk header")
        length = struct.unpack(">I", payload[position:position + 4])[0]
        kind = payload[position + 4:position + 8]
        body = payload[position + 8:position + 8 + length]
        position += 12 + length
        _require(len(body) == length, "Truncated PNG chunk body")
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = \
                struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            compressed.extend(body)
        elif kind == b"IEND":
            break
        else:
            continue
    _require(width and height and bit_depth == 8 and interlace == 0,
             "PNG geometry or bit depth is unsupported")
    _require(color_type in (2, 6), "PNG colour type must be RGB or RGBA")
    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    _require(len(raw) == height * (stride + 1), "PNG scanline length differs")
    previous = bytearray(stride)
    output = bytearray()
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        scanline = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += stride + 1
        if filter_type == 1:
            for index in range(channels, stride):
                scanline[index] = (scanline[index]
                                   + scanline[index - channels]) & 0xFF
        elif filter_type == 2:
            for index in range(stride):
                scanline[index] = (scanline[index] + previous[index]) & 0xFF
        elif filter_type == 3:
            for index in range(stride):
                left = scanline[index - channels] if index >= channels else 0
                scanline[index] = (scanline[index]
                                   + ((left + previous[index]) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(stride):
                left = scanline[index - channels] if index >= channels else 0
                upper_left = (previous[index - channels]
                              if index >= channels else 0)
                scanline[index] = (scanline[index] + _paeth_predictor(
                    left, previous[index], upper_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter: {filter_type}")
        output.extend(scanline)
        previous = scanline
    return int(width), int(height), channels, bytes(output)


def verify_scene_pixels(payload: bytes, scene: Mapping[str, Any]
                        ) -> dict[str, Any]:
    """Verify exact renderer bytes, visible pixels and occlusion metadata."""
    _require(render_scene_png(dict(scene)) == payload,
             "Image bytes differ from the semantic scene render")
    width, height, channels, pixels = decode_png_rgb8(payload)
    _require(width == int(scene["width"]) and height == int(scene["height"]),
             "Rendered PNG geometry differs from the scene")
    visible_checked = occluded_inspected = occluded_hidden = 0
    for obj in scene["objects"]:
        center_x, center_y = (int(value) for value in obj["center"])
        _require(0 <= center_x < width and 0 <= center_y < height,
                 "Object centre is outside the image")
        offset = (center_y * width + center_x) * channels
        pixel = tuple(pixels[offset:offset + 3])
        expected = tuple(COLORS[obj["color"]])
        if obj.get("role") == "occluded":
            # The renderer may leave a partially occluded object visible at
            # its centre. The semantic contract still excludes it from the
            # caption; record whether its centre is hidden.
            occluded_inspected += 1
            occluded_hidden += int(pixel != expected)
        else:
            _require(pixel == expected,
                     "Visible object centre colour differs from its descriptor")
            visible_checked += 1
    _require(visible_checked > 0, "A scene must contain a visible object")
    return {"width": width, "height": height,
            "visible_pixels_verified": visible_checked,
            "occluded_pixels_inspected": occluded_inspected,
            "occluded_centres_hidden": occluded_hidden,
            "pixel_check_passed": True}


def semantic_caption_agreement(text: str,
                               expected_objects: Sequence[Sequence[str]]
                               ) -> dict[str, Any]:
    expected = Counter((str(color), str(shape))
                       for color, shape in expected_objects)
    predicted = Counter(caption_descriptors(text))
    true_positive = sum((predicted & expected).values())
    precision = (true_positive / sum(predicted.values())
                 if predicted else 0.0)
    recall = true_positive / sum(expected.values()) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    phantom = sum((predicted - expected).values())
    missing = sum((expected - predicted).values())
    return {
        "expected_descriptors": [list(item) for item in sorted(expected)],
        "parsed_descriptors": [list(item) for item in sorted(predicted)],
        "true_positive": true_positive,
        "precision": precision, "recall": recall, "f1": f1,
        "exact_agreement": predicted == expected,
        "phantom_objects": phantom, "missing_objects": missing,
    }


# ---------------------------------------------------------------------------
# Balanced deterministic selection
# ---------------------------------------------------------------------------


def _balance_histogram(rows: Sequence[Mapping[str, Any]],
                       key_function) -> dict[str, int]:
    counts = Counter(key_function(row) for row in rows)
    return {str(key): counts[key] for key in sorted(counts)}


def balanced_select(rows: Sequence[Mapping[str, Any]], key_function, *,
                    per_class: int, excludes: Iterable[str] = ()
                    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Take exactly per_class deterministic cases from every observed class."""
    excluded = set(excludes)
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["case_id"] in excluded:
            continue
        groups[key_function(row)].append(row)
    _require(groups, "Balanced selection found no eligible cases")
    histogram = {str(key): len(values)
                 for key, values in sorted(groups.items())}
    insufficient = {str(key): len(values)
                    for key, values in groups.items()
                    if len(values) < per_class}
    _require(not insufficient,
             f"Balanced selection is short: {insufficient} for {per_class}")
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=str):
        ordered = sorted(groups[key], key=lambda row: row["case_id"])
        selected.extend(dict(row) for row in ordered[:per_class])
    selected.sort(key=lambda row: row["case_id"])
    counts = Counter(key_function(row) for row in selected)
    _require(set(counts.values()) == {per_class},
             f"Balanced selection is uneven: {dict(counts)}")
    return selected, histogram


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------


def _asset_path(asset: Mapping[str, Any]) -> str:
    return "procedural/" + asset["path"]


def _case_row(case: Mapping[str, Any], asset: Mapping[str, Any], *,
              question: str | None = None,
              sibling: Mapping[str, Any] | None = None,
              sibling_asset: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "image": _asset_path(asset),
        "image_sha256": asset["sha256"],
        "question": case["question"] if question is None else question,
        "task": case["task"],
        "kind": case.get("kind"),
        "world_id": case["world_id"],
        "pair_id": case["pair_id"],
        "scene_sha256": case["scene_sha256"],
    }
    if sibling is not None:
        _require(sibling_asset is not None,
                 "Counterfactual sibling asset is required")
        row["counterfactual_case_id"] = sibling["case_id"]
        row["counterfactual_image"] = _asset_path(sibling_asset)
        row["counterfactual_image_sha256"] = sibling_asset["sha256"]
        row["counterfactual_scene_sha256"] = sibling["scene_sha256"]
    return row


def _donor_info(case: Mapping[str, Any], asset: Mapping[str, Any],
                *, caption: str | None = None,
                visible_objects: Sequence[Sequence[str]] | None = None,
                sealed_answer: str | None = None
                ) -> dict[str, Any]:
    info = {
        "case_id": case["case_id"],
        "image": _asset_path(asset),
        "image_sha256": asset["sha256"],
        "scene_sha256": case["scene_sha256"],
        "world_id": case["world_id"],
        "task": case["task"],
    }
    if caption is not None:
        _require(visible_objects is not None,
                 "Caption donor requires visible descriptors")
        info["caption"] = caption
        info["visible_objects"] = [list(item) for item in visible_objects]
    if sealed_answer is not None:
        info["sealed_answer"] = sealed_answer
    return info


def _sibling_for(case: Mapping[str, Any],
                 pair_cases: Mapping[str, Mapping[str, Mapping[str, Any]]]
                 ) -> Mapping[str, Any]:
    sides = pair_cases[case["pair_id"]]
    _require(len(sides) == 2, "A procedural pair must have two sides")
    matches = [item for item in sides.values()
               if item["case_id"] == case["case_id"]]
    _require(len(matches) == 1, "Case pair identity is ambiguous")
    sibling = [item for item in sides.values()
               if item["case_id"] != case["case_id"]][0]
    _require(sibling["case_id"] != case["case_id"],
             "Counterfactual sibling is identical")
    return sibling


def _caption_counterfactual_valid(case: Mapping[str, Any],
                                  sibling: Mapping[str, Any],
                                  scenes: Mapping[str, Mapping[str, Any]],
                                  assets: Mapping[str, Mapping[str, Any]]
                                  ) -> bool:
    owned = visible_descriptors(scenes[case["asset_id"]])
    other = visible_descriptors(scenes[sibling["asset_id"]])
    return (Counter(owned) != Counter(other)
            and assets[case["asset_id"]]["sha256"]
            != assets[sibling["asset_id"]]["sha256"])


def _select_role(split_cases: Sequence[Mapping[str, Any]], *, label: str,
                 answer_key, caption_key, answers_per_class: int,
                 captions_per_class: int,
                 caption_candidates: Sequence[Mapping[str, Any]],
                 sibling_case, require_disjoint: bool = True):
    answers, answer_hist = balanced_select(
        split_cases, answer_key, per_class=answers_per_class)
    answer_ids = {row["case_id"] for row in answers}
    excludes = set(answer_ids)
    if require_disjoint:
        excludes |= {sibling_case(row)["case_id"] for row in answers}
    captions, caption_hist = balanced_select(
        caption_candidates, caption_key, per_class=captions_per_class,
        excludes=excludes)
    if require_disjoint:
        caption_ids = {row["case_id"] for row in captions}
        _require(answer_ids.isdisjoint(caption_ids),
                 f"{label} answer/caption case ids overlap")
        _require(caption_ids.isdisjoint(
            {sibling_case(row)["case_id"] for row in answers}),
            f"{label} caption reuses an answer sibling")
    return answers, answer_hist, captions, caption_hist


_CONTROL_ASSIGNMENT_KEYS = (
    "case_id", "split", "wrong", "shuffled", "counterfactual",
    "counterfactual_same_image",
)


def _assignment_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _assignment_value(item) for key, item in value.items()
                if key != "image_path"}
    if isinstance(value, (list, tuple)):
        return [_assignment_value(item) for item in value]
    return value


def control_assignment_sha256(row: Mapping[str, Any]) -> str:
    """Bind one sealed control assignment, including donor target fields."""
    payload = {key: row[key] for key in _CONTROL_ASSIGNMENT_KEYS
               if key in row}
    return fingerprint(_assignment_value(payload))


def _control_map(selected: Sequence[Mapping[str, Any]], *,
                 combined: Sequence[Mapping[str, Any]], seed_offset: int,
                 assets: Mapping[str, Mapping[str, Any]],
                 case_by_id: Mapping[str, Mapping[str, Any]],
                 donor_info, sibling_case, split_label: str,
                 include_wrong: bool = True,
                 include_sealed_answers: bool = False,
                 bind_assignment: bool = False) -> list[dict[str, Any]]:
    assignment = assign_controls(list(combined), seed_offset=seed_offset)
    rows = []
    selected_ids = {row["case_id"] for row in selected}
    for case in selected:
        receiver = assets[case["asset_id"]]["sha256"]
        entry: dict[str, Any] = {"case_id": case["case_id"],
                                 "split": split_label}
        for control in (("wrong", "shuffled") if include_wrong
                        else ("shuffled",)):
            donor_id = assignment[control].get(case["case_id"])
            _require(donor_id is not None,
                     f"{split_label} control is missing for {case['case_id']}")
            if include_sealed_answers:
                donor = case_by_id[donor_id]
                needs_change = (
                    donor["answer"] == case["answer"]
                    or (control == "shuffled"
                        and donor["task"] != case["task"]))
                if needs_change:
                    candidates = sorted(
                        other["case_id"] for other in combined
                        if other["case_id"] != case["case_id"]
                        and other["world_id"] != case["world_id"]
                        and other["task"] == case["task"]
                        and other["answer"] != case["answer"])
                    if not candidates:
                        candidates = sorted(
                            other["case_id"] for other in combined
                            if other["case_id"] != case["case_id"]
                            and other["world_id"] != case["world_id"]
                            and other["answer"] != case["answer"])
                    _require(candidates,
                             f"No answer-changing {control} donor for "
                             f"{case['case_id']}")
                    index = (seed_offset
                             + sum(ord(char) for char in case["case_id"])
                             + (0 if control == "wrong" else 1)) \
                        % len(candidates)
                    donor_id = candidates[index]
            donor = case_by_id[donor_id]
            donor_asset = assets[donor["asset_id"]]
            _require(donor_asset["sha256"] != receiver,
                     "Control donor reuses the receiver image")
            if control == "wrong":
                if donor["task"] != case["task"]:
                    _require(include_sealed_answers,
                             "Wrong-image donor task differs")
                _require(donor["world_id"] != case["world_id"],
                         "Wrong-image donor world differs")
            else:
                _require(donor["world_id"] != case["world_id"],
                         "Shuffled donor world differs")
            entry[control] = donor_info(donor)
            if include_sealed_answers:
                entry[control]["sealed_answer"] = donor["answer"]
                if control in ("wrong", "shuffled"):
                    _require(donor["answer"] != case["answer"],
                             f"{control} donor does not change the answer: "
                             f"{case['case_id']}")
        sibling = sibling_case(case)
        sibling_asset = assets[sibling["asset_id"]]
        same_image = sibling_asset["sha256"] == receiver
        if same_image:
            # The declared occluded-relation abstention control has identical
            # pixels while its verifier scenes differ. It has no
            # answer-changing counterfactual and is reported separately.
            _require(case["answer"] == UNKNOWN_ANSWER,
                     "Only abstain pairs may share an identical image")
        entry["counterfactual"] = donor_info(sibling)
        if include_sealed_answers:
            entry["counterfactual"]["sealed_answer"] = sibling["answer"]
        entry["counterfactual_same_image"] = same_image
        if bind_assignment:
            entry["assignment_sha256"] = control_assignment_sha256(entry)
        rows.append(entry)
    _require({row["case_id"] for row in rows} == selected_ids,
             "Control map does not cover every selected case")
    return rows


def build_balanced_dataset(config: Mapping[str, Any],
                           project_root: str | Path
                           ) -> dict[str, Any]:
    config = validate_config(config)
    v2 = config["format"] == CONFIG_FORMAT_V2
    root = Path(project_root)
    data_config = load_curriculum_config(
        _safe_relative(root, config["parent_data_config"]["path"]))
    _require(sha256_file(root / config["parent_data_config"]["path"])
             == config["parent_data_config"]["sha256"],
             "Parent data-config bytes differ")
    _require(sha256_file(root / config["parent_protocol"]["path"])
             == config["parent_protocol"]["sha256"],
             "Parent protocol bytes differ")
    prior = load_prior_evaluation(root, config)
    data_config = dict(data_config)
    data_config["seed"] = int(config["seed"])
    data_config["data_root"] = config["data_root"]
    data_config["procedural"] = dict(data_config["procedural"])
    data_config["procedural"]["id_prefix"] = \
        config["procedural_overrides"]["id_prefix"]
    data_config["procedural"]["worlds_per_task"] = \
        int(config["procedural_overrides"]["worlds_per_task"])
    renderer = dict(data_config["procedural"]["renderer"])
    renderer["width"] = int(config["procedural_overrides"]["renderer"]["width"])
    renderer["height"] = int(
        config["procedural_overrides"]["renderer"]["height"])
    data_config["procedural"]["renderer"] = renderer
    data_config["procedural"]["split_layouts"] = {
        split: list(values)
        for split, values in
        config["procedural_overrides"]["split_layouts"].items()}
    procedural = build_procedural(data_config)

    cases = [dict(case) for case in procedural["cases"]]
    case_by_id = {case["case_id"]: case for case in cases}
    assets = {asset["asset_id"]: dict(asset)
              for asset in procedural["assets"]}
    scenes = {entry["asset_id"]: entry["scene"]
              for entry in procedural["scenes"]}
    pair_cases: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for case in cases:
        side = assets[case["asset_id"]]["side"]
        pair_cases[case["pair_id"]][side] = case

    def sibling(case):
        return _sibling_for(case, pair_cases)

    def caption_of(asset_id):
        return scene_caption(scenes[asset_id])

    def visible(asset_id):
        return visible_descriptors(scenes[asset_id])

    def eligible(case):
        asset = assets[case["asset_id"]]
        return (asset["sha256"] not in prior["image_sha256"]
                and case["scene_sha256"] not in prior["scene_sha256"]
                and case["case_id"] not in prior["case_ids"])

    split_cases = {split: [case for case in cases
                           if case["split"] == split
                           and eligible(case)]
                   for split in ("train", "dev", "audit")}
    # Evaluation roles must also carry a fresh counterfactual sibling. A
    # sibling outside the fresh set would re-use prior evaluation evidence.
    evaluation_cases = {
        split: [case for case in split_cases[split]
                if eligible(sibling(case))]
        for split in ("dev", "audit")}
    for split, rows in split_cases.items():
        _require(bool(rows), f"Fresh {split} cases are unavailable")
    for split, rows in evaluation_cases.items():
        _require(bool(rows),
                 f"Fresh pair-complete {split} cases are unavailable")

    def answer_key(case):
        return f"task={case['task']}|answer={case['answer']}"

    def caption_key(case):
        return f"objects={len(visible(case['asset_id']))}"

    caption_candidates = {}
    for split, rows in split_cases.items():
        pool = evaluation_cases[split] if split in evaluation_cases else rows
        caption_candidates[split] = [
            case for case in pool
            if _caption_counterfactual_valid(case, sibling(case), scenes, assets)
        ]

    selection = config["selection"]
    roles = {}
    role_names = {"train": "train", "dev": "dev", "audit": "heldout"}
    for split in ("train", "dev", "audit"):
        role_name = role_names[split]
        pool = evaluation_cases[split] if split in evaluation_cases \
            else split_cases[split]
        answers, answer_hist, captions, caption_hist = _select_role(
            pool,
            label=f"{role_name}_answer",
            answer_key=answer_key,
            caption_key=caption_key,
            answers_per_class=selection[f"{role_name}_answers_per_class"],
            captions_per_class=selection[f"{role_name}_captions_per_class"],
            caption_candidates=caption_candidates[split],
            sibling_case=sibling,
            require_disjoint=selection["require_role_case_disjointness"])
        _require(all(len(visible(case["asset_id"]))
                     <= config["selection"]["max_visible_objects"]
                     for case in captions),
                 f"{split} caption exceeds the visible-object ceiling")
        roles[split] = {"answers": answers, "captions": captions,
                        "answer_hist": answer_hist,
                        "caption_hist": caption_hist}

    # Attach counterfactual identity to every answer/caption case.
    for split, role in roles.items():
        for case in role["answers"]:
            _require(not question_leaks(case["question"], case["answer"]),
                     f"Answer leaks into its question: {case['case_id']}")
            other = sibling(case)
            case["_counterfactual"] = other
            case["_counterfactual_asset"] = assets[other["asset_id"]]
        for case in role["captions"]:
            other = sibling(case)
            case["_counterfactual"] = other
            case["_counterfactual_asset"] = assets[other["asset_id"]]

    # Deterministic control maps use donors from the same fresh split.
    answer_control_maps = {}
    caption_control_maps = {}
    for split in ("dev", "audit"):
        donors = split_cases[split]

        def answer_donor_info(case, _assets=assets):
            return _donor_info(case, _assets[case["asset_id"]])

        def caption_donor_info(case, _assets=assets):
            return _donor_info(
                case, _assets[case["asset_id"]],
                caption=caption_of(case["asset_id"]),
                visible_objects=visible(case["asset_id"]))

        answer_control_maps[split] = _control_map(
            roles[split]["answers"],
            combined=donors,
            seed_offset=selection["control_seed_offset"],
            assets=assets, case_by_id=case_by_id,
            donor_info=answer_donor_info, sibling_case=sibling,
            split_label=f"{split}_answer",
            include_sealed_answers=v2, bind_assignment=v2)
        combined_caption = {row["case_id"]: dict(row, task="caption")
                            for row in caption_candidates[split]}
        for row in roles[split]["captions"]:
            combined_caption[row["case_id"]] = dict(row, task="caption")
        caption_control_maps[split] = _control_map(
            roles[split]["captions"],
            combined=list(combined_caption.values()),
            seed_offset=selection["control_seed_offset"] + 1,
            assets=assets, case_by_id=case_by_id,
            donor_info=caption_donor_info, sibling_case=sibling,
            split_label=f"{split}_caption", include_wrong=False,
            bind_assignment=v2)

    # Gather every selected and control asset, then audit pixels and captions.
    used_case_ids: set[str] = set()
    for split in ("train", "dev", "audit"):
        used_case_ids.update(row["case_id"] for row in roles[split]["answers"])
        used_case_ids.update(row["case_id"] for row in roles[split]["captions"])
    for mapping in answer_control_maps.values():
        for row in mapping:
            for control in ("wrong", "shuffled", "counterfactual"):
                used_case_ids.add(row[control]["case_id"])
    for mapping in caption_control_maps.values():
        for row in mapping:
            for control in ("shuffled", "counterfactual"):
                used_case_ids.add(row[control]["case_id"])

    used_assets: set[str] = set()
    audit_rows = []
    for case_id in sorted(used_case_ids):
        case = case_by_id[case_id]
        asset = assets[case["asset_id"]]
        scene = scenes[case["asset_id"]]
        caption = caption_of(case["asset_id"])
        check = verify_caption_semantics(caption, scene)
        pixels = verify_scene_pixels(
            procedural["files"][asset["path"]], scene)
        _require(Counter(map(tuple, check["parsed_descriptors"]))
                 == Counter(map(tuple, check["visible_objects"])),
                 "Parsed caption descriptors differ from visible descriptors")
        audit_rows.append({
            "case_id": case_id,
            "split": case["split"],
            "task": case["task"],
            "image": _asset_path(asset),
            "image_sha256": asset["sha256"],
            "scene_sha256": case["scene_sha256"],
            "caption": caption,
            "visible_objects": check["visible_objects"],
            "pixel_check": pixels,
            "semantic_check": check,
        })
        used_assets.add(case["asset_id"])

    # Balance and freshness evidence.
    balance = {}
    for split, role in roles.items():
        role_name = role_names[split]
        balance[role_name] = {
            "answers": role["answer_hist"],
            "answer_cases": len(role["answers"]),
            "captions": role["caption_hist"],
            "caption_cases": len(role["captions"]),
            "answer_histogram_selected": _balance_histogram(
                role["answers"], answer_key),
            "caption_histogram_selected": _balance_histogram(
                role["captions"], caption_key),
        }
        _require(set(balance[role_name]["answer_histogram_selected"].values())
                 == {selection[f"{role_name}_answers_per_class"]},
                 f"{split} answer balance differs")
        _require(set(balance[role_name]["caption_histogram_selected"].values())
                 == {selection[f"{role_name}_captions_per_class"]},
                 f"{split} caption balance differs")

    def overlap(role_cases, field, prior_field):
        return len({assets[case["asset_id"]][field]
                    if field == "sha256" else case[field]
                    for case in role_cases} & prior[prior_field])

    freshness = {
        "policy": ("every train/dev/held-out case, control donor and "
                   "counterfactual sibling excludes prior evaluation case "
                   "ids, image hashes and scene hashes; all used images are "
                   "newly rendered at fresh geometry and are disjoint from "
                   "every prior prepared asset; the prior rejected stage "
                   "result is preserved and not rerun"),
        "prior": {
            "protocol_sha256": prior["protocol_sha256"],
            "manifest_sha256": prior["manifest_sha256"],
            "counts": prior["counts"],
            "namespaces": prior["namespaces"],
            "files": prior["files"],
        },
        "namespace": config["procedural_overrides"]["id_prefix"],
        "prior_namespaces": prior["namespaces"],
        "train": {
            "case_overlap": overlap(roles["train"]["answers"]
                                    + roles["train"]["captions"],
                                    "case_id", "case_ids"),
            "image_overlap": overlap(roles["train"]["answers"]
                                     + roles["train"]["captions"],
                                     "sha256", "image_sha256"),
            "scene_overlap": overlap(roles["train"]["answers"]
                                     + roles["train"]["captions"],
                                     "scene_sha256", "scene_sha256"),
        },
        "dev": {
            "case_overlap": overlap(roles["dev"]["answers"]
                                    + roles["dev"]["captions"],
                                    "case_id", "case_ids"),
            "image_overlap": overlap(roles["dev"]["answers"]
                                     + roles["dev"]["captions"],
                                     "sha256", "image_sha256"),
            "scene_overlap": overlap(roles["dev"]["answers"]
                                     + roles["dev"]["captions"],
                                     "scene_sha256", "scene_sha256"),
        },
        "heldout": {
            "case_overlap": overlap(roles["audit"]["answers"]
                                    + roles["audit"]["captions"],
                                    "case_id", "case_ids"),
            "image_overlap": overlap(roles["audit"]["answers"]
                                     + roles["audit"]["captions"],
                                     "sha256", "image_sha256"),
            "scene_overlap": overlap(roles["audit"]["answers"]
                                     + roles["audit"]["captions"],
                                     "scene_sha256", "scene_sha256"),
        },
    }
    used_objects = [case_by_id[case_id] for case_id in sorted(used_case_ids)]
    if v2:
        freshness["prior_sources"] = list(prior.get("sources", []))
    freshness["all_used_cases"] = {
        "cases": len(used_objects),
        "case_overlap": overlap(used_objects, "case_id", "case_ids"),
        "image_overlap": overlap(used_objects, "sha256", "image_sha256"),
        "scene_overlap": overlap(used_objects, "scene_sha256",
                                 "scene_sha256"),
    }
    for split in ("train", "dev", "heldout"):
        for metric in ("case_overlap", "image_overlap", "scene_overlap"):
            _require(freshness[split][metric] == 0,
                     f"{split} {metric} is not zero")
    _require(all(freshness["all_used_cases"][metric] == 0
                 for metric in ("case_overlap", "image_overlap",
                                "scene_overlap")),
             "A control or counterfactual reused prior evaluation evidence")
    _require(config["procedural_overrides"]["id_prefix"]
             not in prior["namespaces"],
             "Fresh namespace collides with a prior namespace")
    freshness["namespace_disjoint_from_prior"] = True
    freshness["train_overlap_with_all_prior_assets"] = len(
        {assets[case["asset_id"]]["sha256"]
         for case in roles["train"]["answers"] + roles["train"]["captions"]}
        & prior["all_asset_sha256"])
    freshness["all_used_prior_assets"] = {
        "images": len(used_assets),
        "image_overlap": len(
            {assets[asset_id]["sha256"] for asset_id in used_assets}
            & prior["all_asset_sha256"]),
        "geometry": dict(config["procedural_overrides"]["renderer"]),
        "policy": ("all used images are newly rendered at the declared "
                   "geometry, not just disjoint from prior evaluation"),
    }
    _require(freshness["all_used_prior_assets"]["image_overlap"] == 0,
             "A used image bytes-matches a prior prepared asset")

    # Payloads.
    files: dict[str, bytes] = {}
    for asset_id in sorted(used_assets):
        asset = assets[asset_id]
        files[_asset_path(asset)] = procedural["files"][asset["path"]]

    def build_answer_rows(cases, split):
        rows = []
        for case in cases:
            other = case["_counterfactual"]
            rows.append(_case_row(
                case, assets[case["asset_id"]], sibling=other,
                sibling_asset=case["_counterfactual_asset"]))
        rows.sort(key=lambda row: row["case_id"])
        return rows

    def build_caption_rows(cases, *, question):
        rows = []
        for case in cases:
            other = case["_counterfactual"]
            rows.append(_case_row(
                case, assets[case["asset_id"]], question=question,
                sibling=other, sibling_asset=case["_counterfactual_asset"]))
        rows.sort(key=lambda row: row["case_id"])
        return rows

    question = config["evaluation"]["caption_question"]
    dev_answer_cases = build_answer_rows(roles["dev"]["answers"], "dev")
    heldout_answer_cases = build_answer_rows(roles["audit"]["answers"],
                                             "audit")
    dev_caption_cases = build_caption_rows(roles["dev"]["captions"],
                                           question=question)
    heldout_caption_cases = build_caption_rows(roles["audit"]["captions"],
                                               question=question)

    def answer_labels(cases, split):
        rows = []
        for case in cases:
            other = case["_counterfactual"]
            rows.append({
                "case_id": case["case_id"], "split": split,
                "task": case["task"], "kind": case.get("kind"),
                "world_id": case["world_id"], "pair_id": case["pair_id"],
                "scene_sha256": case["scene_sha256"],
                "answer": case["answer"],
                "counterfactual_case_id": other["case_id"],
                "counterfactual_answer": other["answer"],
                "counterfactual_scene_sha256": other["scene_sha256"],
                "answer_changes_under_counterfactual":
                    case["answer"] != other["answer"],
                "label_visibility": "verifier_only",
            })
        rows.sort(key=lambda row: row["case_id"])
        return rows

    def caption_labels(cases, split):
        rows = []
        for case in cases:
            other = case["_counterfactual"]
            owned_visible = [list(item) for item in visible(case["asset_id"])]
            other_visible = [list(item) for item in visible(other["asset_id"])]
            rows.append({
                "case_id": case["case_id"], "split": split,
                "task": "caption", "kind": "caption",
                "world_id": case["world_id"], "pair_id": case["pair_id"],
                "scene_sha256": case["scene_sha256"],
                "answer": caption_of(case["asset_id"]),
                "visible_objects": owned_visible,
                "counterfactual_case_id": other["case_id"],
                "counterfactual_answer": caption_of(other["asset_id"]),
                "counterfactual_visible_objects": other_visible,
                "counterfactual_scene_sha256": other["scene_sha256"],
                "caption_changes_under_counterfactual":
                    Counter(map(tuple, owned_visible))
                    != Counter(map(tuple, other_visible)),
                "label_visibility": "verifier_only",
            })
        rows.sort(key=lambda row: row["case_id"])
        return rows

    dev_answer_labels = answer_labels(roles["dev"]["answers"], "dev")
    heldout_answer_labels = answer_labels(roles["audit"]["answers"], "audit")
    dev_caption_labels = caption_labels(roles["dev"]["captions"], "dev_caption")
    heldout_caption_labels = caption_labels(roles["audit"]["captions"],
                                            "audit_caption")
    for label in (dev_answer_labels + heldout_answer_labels):
        if label["task"] != "abstain":
            _require(label["answer_changes_under_counterfactual"] is True,
                     "Answerable counterfactual does not change the answer")
    for label in (dev_caption_labels + heldout_caption_labels):
        _require(label["caption_changes_under_counterfactual"] is True,
                 "Caption counterfactual does not change the description")

    # Training rows and pinned batch schedule.
    train_answer_rows = []
    for case in roles["train"]["answers"]:
        row = _case_row(case, assets[case["asset_id"]])
        row.update({"example_id": f"answer:{case['case_id']}",
                    "kind": "answer", "response": case["answer"],
                    "role": "answer"})
        train_answer_rows.append(row)
    train_caption_rows = []
    for case in roles["train"]["captions"]:
        row = _case_row(case, assets[case["asset_id"]], question=question)
        row.update({"example_id": f"caption:{case['case_id']}",
                    "kind": "caption",
                    "response": caption_of(case["asset_id"]),
                    "role": "caption"})
        train_caption_rows.append(row)
    train_answer_rows.sort(key=lambda row: row["example_id"])
    train_caption_rows.sort(key=lambda row: row["example_id"])
    train_caption_visible = {
        case["case_id"]: [list(item) for item in visible(case["asset_id"])]
        for case in roles["train"]["captions"]}
    train_caption_labels = []
    for row in train_caption_rows:
        train_caption_labels.append({
            "case_id": row["case_id"],
            "split": "train_caption",
            "task": "caption",
            "world_id": row["world_id"],
            "scene_sha256": row["scene_sha256"],
            "answer": row["response"],
            "visible_objects": train_caption_visible[row["case_id"]],
            "label_visibility": "verifier_only",
        })
    train_caption_labels.sort(key=lambda row: row["case_id"])
    alignment = alignment_schedule(
        train_caption_rows, train_answer_rows, config)

    files["training/answers.jsonl"] = _jsonl_bytes(train_answer_rows)
    files["training/captions.jsonl"] = _jsonl_bytes(train_caption_rows)
    files["training/train_caption_labels.jsonl"] = _jsonl_bytes(
        train_caption_labels)
    files["training/alignment.jsonl"] = _jsonl_bytes(
        sorted(train_answer_rows + train_caption_rows,
               key=lambda row: row["example_id"]))
    files["training/batches.jsonl"] = _jsonl_bytes(alignment["batches"])

    for name, rows in (
            ("dev_answer_cases", dev_answer_cases),
            ("dev_answer_labels", dev_answer_labels),
            ("heldout_answer_cases", heldout_answer_cases),
            ("heldout_answer_labels", heldout_answer_labels),
            ("dev_caption_cases", dev_caption_cases),
            ("dev_caption_labels", dev_caption_labels),
            ("heldout_caption_cases", heldout_caption_cases),
            ("heldout_caption_labels", heldout_caption_labels)):
        files[f"evaluation/{name}.jsonl"] = _jsonl_bytes(rows)
    files["evaluation/dev_answer_controls.jsonl"] = _jsonl_bytes(
        answer_control_maps["dev"])
    files["evaluation/heldout_answer_controls.jsonl"] = _jsonl_bytes(
        answer_control_maps["audit"])
    files["evaluation/dev_caption_controls.jsonl"] = _jsonl_bytes(
        caption_control_maps["dev"])
    files["evaluation/heldout_caption_controls.jsonl"] = _jsonl_bytes(
        caption_control_maps["audit"])

    semantic_audit = {
        "format": SEMANTIC_AUDIT_FORMAT,
        "version": VERSION,
        "policy": ("exact visible-descriptor multiset against the rendered "
                   "scene, occluded descriptors never named, and centre "
                   "pixels decoded from the actual PNG"),
        "all_passed": True,
        "cases_checked": len(audit_rows),
        "visible_pixels_verified": sum(
            row["pixel_check"]["visible_pixels_verified"]
            for row in audit_rows),
        "occluded_pixels_inspected": sum(
            row["pixel_check"]["occluded_pixels_inspected"]
            for row in audit_rows),
        "occluded_centres_hidden": sum(
            row["pixel_check"]["occluded_centres_hidden"]
            for row in audit_rows),
        "caption_checks": len(audit_rows),
        "rows": audit_rows,
    }
    files["evaluation/semantic_audit.json"] = _json_bytes(semantic_audit)
    files["evaluation/balance.json"] = _json_bytes(balance)
    files["evaluation/freshness.json"] = _json_bytes(freshness)

    seals = {
        "train_example_ids": fingerprint(
            sorted(row["example_id"]
                   for row in train_answer_rows + train_caption_rows)),
        "train_alignment_payload": sha256_bytes(
            files["training/alignment.jsonl"]),
        "train_batches_payload": sha256_bytes(
            files["training/batches.jsonl"]),
        "train_caption_labels_payload": sha256_bytes(
            files["training/train_caption_labels.jsonl"]),
        "dev_answer_case_ids": fingerprint(
            [row["case_id"] for row in dev_answer_cases]),
        "dev_answer_cases_payload": sha256_bytes(
            files["evaluation/dev_answer_cases.jsonl"]),
        "dev_answer_labels_payload": sha256_bytes(
            files["evaluation/dev_answer_labels.jsonl"]),
        "dev_caption_case_ids": fingerprint(
            [row["case_id"] for row in dev_caption_cases]),
        "dev_caption_cases_payload": sha256_bytes(
            files["evaluation/dev_caption_cases.jsonl"]),
        "dev_caption_labels_payload": sha256_bytes(
            files["evaluation/dev_caption_labels.jsonl"]),
        "heldout_answer_case_ids": fingerprint(
            [row["case_id"] for row in heldout_answer_cases]),
        "heldout_answer_cases_payload": sha256_bytes(
            files["evaluation/heldout_answer_cases.jsonl"]),
        "heldout_answer_labels_payload": sha256_bytes(
            files["evaluation/heldout_answer_labels.jsonl"]),
        "heldout_caption_case_ids": fingerprint(
            [row["case_id"] for row in heldout_caption_cases]),
        "heldout_caption_cases_payload": sha256_bytes(
            files["evaluation/heldout_caption_cases.jsonl"]),
        "heldout_caption_labels_payload": sha256_bytes(
            files["evaluation/heldout_caption_labels.jsonl"]),
        "semantic_audit_payload": sha256_bytes(
            files["evaluation/semantic_audit.json"]),
        "dev_answer_controls_payload": sha256_bytes(
            files["evaluation/dev_answer_controls.jsonl"]),
        "dev_caption_controls_payload": sha256_bytes(
            files["evaluation/dev_caption_controls.jsonl"]),
        "heldout_answer_controls_payload": sha256_bytes(
            files["evaluation/heldout_answer_controls.jsonl"]),
        "heldout_caption_controls_payload": sha256_bytes(
            files["evaluation/heldout_caption_controls.jsonl"]),
    }

    manifest = {
        "format": MANIFEST_FORMAT_V2 if v2 else MANIFEST_FORMAT,
        "version": VERSION_V2 if v2 else VERSION,
        "seed": config["seed"],
        "case_namespace": config["procedural_overrides"]["id_prefix"],
        "fresh_evaluation": True,
        "consumed_stage_result_preserved": True,
        **({
            "hypothesis": config["hypothesis"],
            "scorer_contract": dict(config["scorer_contract"]),
            "fit_contract": dict(config["fit_contract"]),
            "preaudit": dict(config["preaudit"]),
            "caption_contract_version": CAPTION_CONTRACT_VERSION,
        } if v2 else {}),
        "roles": {
            "train": ("balanced caption and answer projector alignment "
                      "examples only"),
            "dev": ("fresh bounded development answer and caption cases; "
                    "available before the held-out checkpoint is frozen"),
            "heldout": ("fresh sealed sibling-image counterfactual answer "
                        "and caption cases; never read during training and "
                        "scored once after export"),
            "controls": ("correct, wrong, shuffled and no-image answer "
                         "controls; correct, counterfactual, shuffled and "
                         "no-image caption controls"),
        },
        "counts": {
            "train_answers": len(train_answer_rows),
            "train_captions": len(train_caption_rows),
            "train_examples": len(train_answer_rows) + len(train_caption_rows),
            "train_updates": len(alignment["batches"]),
            "train_examples_scheduled": sum(
                len(batch["example_ids"]) for batch in alignment["batches"]),
            "dev_answers": len(dev_answer_cases),
            "dev_captions": len(dev_caption_cases),
            "heldout_answers": len(heldout_answer_cases),
            "heldout_captions": len(heldout_caption_cases),
            "semantic_audit_cases": len(audit_rows),
            "assets": len(used_assets),
            "procedural_worlds": procedural["manifest"]["counts"]["worlds"],
        },
        "balance": balance,
        "freshness": freshness,
        "semantic_audit": {
            "format": SEMANTIC_AUDIT_FORMAT,
            "all_passed": True,
            "cases_checked": len(audit_rows),
            "visible_pixels_verified": semantic_audit[
                "visible_pixels_verified"],
            "occluded_pixels_inspected": semantic_audit[
                "occluded_pixels_inspected"],
            "occluded_centres_hidden": semantic_audit[
                "occluded_centres_hidden"],
        },
        "generation_fingerprint":
            procedural["manifest"]["generation_fingerprint"],
        "config_sha256": fingerprint(config),
        "seals": seals,
        "files": {},
    }
    for path, payload in sorted(files.items()):
        manifest["files"][path] = {"sha256": sha256_bytes(payload),
                                   "bytes": len(payload)}
    files["manifest.json"] = _json_bytes(manifest)
    manifest["manifest_sha256"] = sha256_bytes(files["manifest.json"])
    return {
        "files": files,
        "manifest": manifest,
        "roles": roles,
        "train_answer_rows": train_answer_rows,
        "train_caption_rows": train_caption_rows,
        "train_batches": alignment["batches"],
        "dev_answer_cases": dev_answer_cases,
        "dev_answer_labels": dev_answer_labels,
        "dev_caption_cases": dev_caption_cases,
        "dev_caption_labels": dev_caption_labels,
        "heldout_answer_cases": heldout_answer_cases,
        "heldout_answer_labels": heldout_answer_labels,
        "heldout_caption_cases": heldout_caption_cases,
        "heldout_caption_labels": heldout_caption_labels,
        "dev_answer_controls": answer_control_maps["dev"],
        "heldout_answer_controls": answer_control_maps["audit"],
        "dev_caption_controls": caption_control_maps["dev"],
        "heldout_caption_controls": caption_control_maps["audit"],
        "balance": balance,
        "freshness": freshness,
        "semantic_audit": semantic_audit,
    }


def alignment_schedule(caption_rows: Sequence[Mapping[str, Any]],
                       answer_rows: Sequence[Mapping[str, Any]],
                       config: Mapping[str, Any]) -> dict[str, Any]:
    alignment = config["schedule"]["alignment"]
    updates = int(alignment["updates"])
    caption_batches = _cycle_batches(
        list(caption_rows), seed=int(config["train"]["seed"]),
        updates=updates, batch=int(alignment["caption_per_batch"]),
        tag="balanced_alignment_caption")
    answer_batches = _cycle_batches(
        list(answer_rows), seed=int(config["train"]["seed"]),
        updates=updates, batch=int(alignment["answer_per_batch"]),
        tag="balanced_alignment_answer")
    batches = []
    used: set[str] = set()
    for update in range(updates):
        rows = list(caption_batches[update]) + list(answer_batches[update])
        _require(len(rows) == int(alignment["batch"]),
                 "Batch mix length differs from the declared batch")
        ids = [row["example_id"] for row in rows]
        _require(len(ids) == len(set(ids)),
                 "One update repeats an example")
        used.update(ids)
        batches.append({"update": update + 1, "example_ids": ids,
                        "caption_examples": alignment["caption_per_batch"],
                        "answer_examples": alignment["answer_per_batch"]})
    expected = {row["example_id"] for row in list(caption_rows)
                + list(answer_rows)}
    _require(used == expected,
             "The bounded schedule does not cover every training example")
    return {"batches": batches, "updates": updates,
            "examples": updates * int(alignment["batch"]),
            "unique_examples": len(expected)}


# ---------------------------------------------------------------------------
# Protocol, writing and verification
# ---------------------------------------------------------------------------


def write_dataset(root: str | Path, built: Mapping[str, Any]) -> None:
    for relative, payload in sorted(built["files"].items()):
        _write_exact(Path(root) / relative, payload)


def _evaluation_gates(config: Mapping[str, Any],
                      manifest: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "min_correct_accuracy": evaluation["min_correct_accuracy"],
        "min_unique_families": min(
            8, manifest["counts"]["dev_answers"],
            manifest["counts"]["heldout_answers"]),
        "max_unknown_rate": evaluation["max_false_abstention_on_answerable"],
        "min_correct_margin_wrong": evaluation["min_margin_wrong"],
        "min_correct_margin_shuffled": evaluation["min_margin_shuffled"],
        "min_correct_margin_noimage": evaluation["min_margin_noimage"],
        "min_unknown_accuracy": evaluation["min_unknown_accuracy"],
        "max_false_answer_on_abstain":
            evaluation["max_false_answer_on_abstain"],
        "max_false_abstention_on_answerable":
            evaluation["max_false_abstention_on_answerable"],
    }


def _evaluation_gates_v2(config: Mapping[str, Any],
                         manifest: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "min_correct_accuracy": evaluation["min_correct_accuracy"],
        "min_unique_families": min(
            8, manifest["counts"]["dev_answers"],
            manifest["counts"]["heldout_answers"]),
        "max_unknown_rate": evaluation["max_false_abstention_on_answerable"],
        "min_correct_margin_wrong": evaluation["min_margin_wrong"],
        "min_correct_margin_shuffled": evaluation["min_margin_shuffled"],
        "min_correct_margin_noimage": evaluation["min_margin_noimage"],
        "min_unknown_accuracy": evaluation["min_unknown_accuracy"],
        "max_false_answer_on_abstain":
            evaluation["max_false_answer_on_abstain"],
        "max_false_abstention_on_answerable":
            evaluation["max_false_abstention_on_answerable"],
        "min_counterfactual_margin_answerable":
            evaluation["min_counterfactual_margin_answerable"],
        "min_counterfactual_following_accuracy":
            evaluation["min_counterfactual_following_accuracy"],
        "min_following_accuracy_wrong":
            evaluation["min_following_accuracy_wrong"],
        "min_following_accuracy_shuffled":
            evaluation["min_following_accuracy_shuffled"],
        "min_caption_token_f1": evaluation["min_caption_token_f1"],
        "min_caption_semantic_descriptor_f1":
            evaluation["min_caption_semantic_descriptor_f1"],
        "min_caption_following_accuracy":
            evaluation["min_caption_following_accuracy"],
        "min_caption_following_semantic_f1":
            evaluation["min_caption_following_semantic_f1"],
        "min_caption_original_agreement_drop":
            evaluation["min_caption_original_agreement_drop"],
        "max_noimage_original_semantic_f1":
            evaluation["max_noimage_original_semantic_f1"],
        "min_noimage_loss_margin": evaluation["min_noimage_loss_margin"],
        "min_train_fit_answer_accuracy":
            evaluation["min_train_fit_answer_accuracy"],
        "min_train_fit_caption_following_accuracy":
            evaluation["min_train_fit_caption_following_accuracy"],
    }


def build_protocol(config: Mapping[str, Any], built: Mapping[str, Any],
                   parent_protocol: Mapping[str, Any],
                   project_root: str | Path | None = None) -> dict[str, Any]:
    config = validate_config(config)
    v2 = config["format"] == CONFIG_FORMAT_V2
    root = Path(project_root or config["project_root"])
    manifest = built["manifest"]
    implementation = {}
    for relative in config["implementation_paths"]:
        path = root / relative
        _require(path.is_file(), f"Implementation file is missing: {relative}")
        implementation[relative] = sha256_file(path)
    parent_eval = parent_protocol["evaluation"]
    limits = dict(parent_protocol["limits"])
    limits.update(config["limits"])
    limits["max_eval_cases"] = max(
        limits["max_eval_cases"],
        manifest["counts"]["dev_answers"] + manifest["counts"]["dev_captions"]
        + manifest["counts"]["heldout_answers"]
        + manifest["counts"]["heldout_captions"])
    evaluation = dict(config["evaluation"])
    evaluation["question_template"] = parent_eval["question_template"]
    evaluation["gates"] = (_evaluation_gates_v2(config, manifest) if v2
                           else _evaluation_gates(config, manifest))
    evaluation["caption_max_new_tokens"] = limits["caption_max_new_tokens"]
    evaluation["max_new_tokens"] = limits["max_new_tokens"]
    prior_consumed = ({
        "sources": [
            {
                "data_root": entry["data_root"],
                "protocol": dict(entry["protocol"]),
                "manifest": dict(entry["manifest"]),
                "role": entry["role"],
            }
            for entry in config["prior_consumed"]
        ],
        "role": ("Every consumed evaluation source is excluded by case id, "
                 "image hash, scene hash and prepared-asset hash; its raw "
                 "rejection/reload evidence is preserved unchanged."),
    } if v2 else {
        "data_root": config["prior_consumed"]["data_root"],
        "protocol": config["prior_consumed"]["protocol"],
        "manifest": config["prior_consumed"]["manifest"],
        "role": config["prior_consumed"]["role"],
    })
    protocol = {
        "format": PROTOCOL_FORMAT_V2 if v2 else PROTOCOL_FORMAT,
        "version": VERSION_V2 if v2 else VERSION,
        "declared_at_utc": config["declared_at_utc"],
        "purpose": config["purpose"],
        "root": str(root),
        "seed": config["seed"],
        **({
            "hypothesis": config["hypothesis"],
            "report_root": config["report_root"],
            "scorer_contract": dict(config["scorer_contract"]),
            "fit_contract": dict(config["fit_contract"]),
            "preaudit": dict(config["preaudit"]),
            "optimizer_contract": {
                "name": "adamw_fused",
                "wrapper": "NativeFusedAdamW",
                "optimizer_class": "torch.optim.AdamW",
                "fused": True,
                "scope": "vision_projector. only",
                "backbone_frozen": True,
            },
        } if v2 else {}),
        "parent": {
            "protocol": config["parent_protocol"]["path"],
            "protocol_sha256": config["parent_protocol"]["sha256"],
            "data_config": config["parent_data_config"]["path"],
            "data_config_sha256": config["parent_data_config"]["sha256"],
            "role": ("Sealed source, preprocessing, optimizer and retention "
                     "contracts only; no warm start and no reused parent "
                     "weights."),
        },
        "prior_consumed": prior_consumed,
        "data": {
            "root": config["data_root"],
            "manifest": {"path": "manifest.json",
                         "sha256": manifest["manifest_sha256"]},
            "seals": manifest["seals"],
            "counts": manifest["counts"],
            "balance": manifest["balance"],
            "freshness": manifest["freshness"],
            "semantic_audit": manifest["semantic_audit"],
            "fresh_evaluation": True,
            "consumed_stage_result_preserved": True,
        },
        "vision_digest": parent_protocol["vision_digest"],
        "model": dict(parent_protocol["model"]),
        "vision": dict(parent_protocol["vision"]),
        "vision_source": dict(parent_protocol["vision_source"]),
        "retention": dict(parent_protocol["retention"]),
        "resource_policy": dict(config["resource_policy"]),
        "limits": limits,
        "train": dict(config["train"]),
        "trainable_scope": {
            "names": list(PROJECTOR_TENSORS),
            "tensor_count": len(PROJECTOR_TENSORS),
            "parameters": PROJECTOR_PARAMETER_COUNT,
            "backbone": "frozen",
            "warm_start": None,
        },
        "stopping": dict(config["stopping"]),
        "schedule": {
            "alignment": dict(config["schedule"]["alignment"]),
            "total_updates": manifest["counts"]["train_updates"],
            "total_examples": manifest["counts"]["train_examples_scheduled"],
            "unique_examples": manifest["counts"]["train_examples"],
            "response_only_labels": True,
            "batches": {
                "path": "training/batches.jsonl",
                "sha256": manifest["seals"]["train_batches_payload"],
            },
        },
        "evaluation": evaluation,
        "checkpoint_policy": dict(config["checkpoint_policy"]),
        "implementation_sha256": implementation,
        "config_sha256": fingerprint(config),
        "claims": {
            "perception_quality": False,
            "semantic_quality": False,
            "overall_baseline_eligible": False,
            "promotion": False,
            "scope": ("bounded balanced projector overfit-alignment "
                      "diagnostic only; the prior stage rejection stands"),
        },
        "limitations": list(config["limitations"]),
    }
    _require(protocol["resource_policy"]["device"] == "mps",
             "Protocol device differs")
    return protocol


def write_protocol(path: str | Path, protocol: Mapping[str, Any]) -> str:
    _write_exact(Path(path), _json_bytes(protocol))
    return sha256_file(path)


def load_prepared(protocol_path: str | Path,
                  project_root: str | Path) -> tuple[dict[str, Any],
                                                     dict[str, Any],
                                                     dict[str, str],
                                                     Path]:
    protocol = json.loads(Path(protocol_path).read_text())
    _require(protocol.get("format") in (PROTOCOL_FORMAT, PROTOCOL_FORMAT_V2),
             "Prepared protocol format differs")
    v2 = protocol.get("format") == PROTOCOL_FORMAT_V2
    root = Path(project_root)
    raw_data_root = protocol["data"]["root"]
    data_root = (Path(raw_data_root) if Path(raw_data_root).is_absolute()
                 else _safe_relative(root, raw_data_root))
    manifest_path = data_root / "manifest.json"
    _require(manifest_path.is_file(), "Prepared manifest is missing")
    _require(sha256_file(manifest_path)
             == protocol["data"]["manifest"]["sha256"],
             "Prepared manifest bytes differ")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("format") in (MANIFEST_FORMAT, MANIFEST_FORMAT_V2),
             "Prepared manifest format differs")
    _require(v2 == (manifest.get("format") == MANIFEST_FORMAT_V2),
             "Prepared protocol and manifest versions differ")
    if v2:
        _require(manifest.get("scorer_contract")
                 == protocol["scorer_contract"],
                 "Prepared scorer contract differs")
        _require(manifest.get("fit_contract") == protocol["fit_contract"],
                 "Prepared fit contract differs")
        _require(manifest.get("caption_contract_version")
                 == CAPTION_CONTRACT_VERSION,
                 "Prepared caption contract version differs")
    _require(manifest.get("seals") == protocol["data"]["seals"],
             "Prepared seals differ from the protocol")
    _require(manifest.get("counts") == protocol["data"]["counts"],
             "Prepared counts differ from the protocol")
    for relative, pin in manifest["files"].items():
        path = _safe_relative(data_root, relative)
        _require(path.is_file(), f"Prepared payload is missing: {relative}")
        _require(sha256_file(path) == pin["sha256"],
                 f"Prepared payload bytes differ: {relative}")
        _require(path.stat().st_size == pin["bytes"],
                 f"Prepared payload size differs: {relative}")
    implementation = verify_implementation_pins(protocol, root)
    return protocol, manifest, implementation, data_root


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_counterfactual_answers(cases: Sequence[Mapping[str, Any]],
                                 generations: Sequence[Mapping[str, Any]],
                                 *, controls: Sequence[str],
                                 seed: int, bootstrap: Mapping[str, Any],
                                 gates: Mapping[str, Any] | None = None
                                 ) -> dict[str, Any]:
    score = score_procedural(
        cases, generations, controls=controls, seed=seed,
        bootstrap_resamples=int(bootstrap["resamples"]),
        bootstrap_confidence=float(bootstrap["confidence"]), gates=gates)
    case_by_id = {case["case_id"]: case for case in cases}
    if "counterfactual" not in controls:
        return score
    rows: dict[str, dict[str, dict[str, Any]]] = {
        control: {} for control in controls}
    for row in score["rows"]:
        rows[row["control"]][row["case_id"]] = row
    correct = rows["correct"]
    counterfactual = rows["counterfactual"]
    shared = sorted(set(correct) & set(counterfactual))
    answerable = [case_id for case_id in shared
                  if case_by_id[case_id]["label"] != UNKNOWN_ANSWER]
    abstain = [case_id for case_id in shared
               if case_by_id[case_id]["label"] == UNKNOWN_ANSWER]
    _require(answerable, "No answerable counterfactual pairs exist")

    def paired_margin(case_ids):
        values = {}
        for case_id in case_ids:
            values[case_id] = (float(correct[case_id]["correct"])
                               - float(counterfactual[case_id]["correct"]))
        return values

    answerable_margin = paired_margin(answerable)
    by_world: dict[str, list[float]] = defaultdict(list)
    for case_id, value in answerable_margin.items():
        by_world[case_by_id[case_id]["world_id"]].append(value)
    interval = clustered_bootstrap(
        by_world, resamples=int(bootstrap["resamples"]),
        confidence=float(bootstrap["confidence"]), seed=seed + 7919)
    abstain_margin = paired_margin(abstain) if abstain else {}
    score["counterfactual_answerable"] = {
        "cases": len(answerable),
        "correct_accuracy": sum(correct[case_id]["correct"]
                                for case_id in answerable) / len(answerable),
        "counterfactual_accuracy":
            sum(counterfactual[case_id]["correct"]
                for case_id in answerable) / len(answerable),
        "correct_margin": sum(answerable_margin.values()) / len(answerable),
        "margin_bootstrap": interval,
        "abstain_cases": len(abstain),
        "abstain_counterfactual_unknown_rate": (
            sum(counterfactual[case_id]["unknown"]
                for case_id in abstain) / len(abstain) if abstain else None),
        "abstain_correct_margin": (
            sum(abstain_margin.values()) / len(abstain) if abstain else None),
        "scope": "answerable_cases_only_for_margin",
    }
    return score


def score_caption_controls(labels: Sequence[Mapping[str, Any]],
                           generations: Sequence[Mapping[str, Any]],
                           *, controls: Sequence[str], seed: int,
                           bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    labels_by_id = {row["case_id"]: row for row in labels}
    _require(len(labels_by_id) == len(labels), "Caption labels repeat")
    rows_by_control: dict[str, dict[str, dict[str, Any]]] = {
        control: {} for control in controls}
    observed = set()
    for generation in generations:
        control = generation.get("control")
        case_id = generation.get("case_id")
        _require(control in controls, f"Unknown caption control: {control}")
        _require(case_id in labels_by_id,
                 f"Unknown caption case: {case_id}")
        _require(case_id not in rows_by_control[control],
                 f"Duplicate caption generation: {control}/{case_id}")
        target_text = generation.get("target_text")
        target_objects = generation.get("target_visible_objects")
        loss = generation.get("response_loss")
        _require(isinstance(target_text, str) and target_text,
                 "Caption target text is missing")
        _require(isinstance(target_objects, list) and target_objects,
                 "Caption target objects are missing")
        _require(isinstance(loss, (int, float)) and not isinstance(loss, bool)
                 and math.isfinite(float(loss)) and float(loss) >= 0,
                 "Caption teacher-forced loss is missing")
        generated = str(generation.get("generated_text", ""))
        semantic = semantic_caption_agreement(generated, target_objects)
        rows_by_control[control][case_id] = {
            **generation,
            "token_f1": caption_token_f1(generated, target_text),
            "semantic": semantic,
            "semantic_f1": semantic["f1"],
            "response_loss": float(loss),
        }
        observed.add((control, case_id))
    for control in controls:
        _require(set(rows_by_control[control]) == set(labels_by_id),
                 f"Caption control is incomplete: {control}")

    def summarize(control):
        values = list(rows_by_control[control].values())
        return {
            "cases": len(values),
            "token_f1": sum(row["token_f1"] for row in values) / len(values),
            "semantic_f1": sum(row["semantic_f1"] for row in values)
            / len(values),
            "semantic_precision": sum(row["semantic"]["precision"]
                                      for row in values) / len(values),
            "semantic_recall": sum(row["semantic"]["recall"]
                                   for row in values) / len(values),
            "semantic_exact_agreement": sum(
                row["semantic"]["exact_agreement"] for row in values)
            / len(values),
            "phantom_object_rate": sum(
                row["semantic"]["phantom_objects"] > 0 for row in values)
            / len(values),
            "teacher_forced_response_loss": sum(
                row["response_loss"] for row in values) / len(values),
        }

    metrics = {control: summarize(control) for control in controls}
    correct = rows_by_control["correct"]
    dependence = {}
    for control in controls:
        if control == "correct":
            continue
        other = rows_by_control[control]
        shared = sorted(set(correct) & set(other))
        _require(shared, f"No paired caption cases for {control}")
        f1_margins = {}
        semantic_margins = {}
        loss_margins = {}
        phantom_differences = {}
        for case_id in shared:
            f1_margins[case_id] = (correct[case_id]["token_f1"]
                                   - other[case_id]["token_f1"])
            semantic_margins[case_id] = (correct[case_id]["semantic_f1"]
                                         - other[case_id]["semantic_f1"])
            loss_margins[case_id] = (other[case_id]["response_loss"]
                                     - correct[case_id]["response_loss"])
            phantom_differences[case_id] = (
                float(other[case_id]["semantic"]["phantom_objects"] > 0)
                - float(correct[case_id]["semantic"]["phantom_objects"] > 0))

        def bootstrap_margin(values):
            by_world: dict[str, list[float]] = defaultdict(list)
            for case_id, value in values.items():
                by_world[labels_by_id[case_id]["world_id"]].append(value)
            return clustered_bootstrap(
                by_world, resamples=int(bootstrap["resamples"]),
                confidence=float(bootstrap["confidence"]),
                seed=seed + 101 * (list(controls).index(control) + 1))

        dependence[control] = {
            "cases": len(shared),
            "token_f1_margin": sum(f1_margins.values()) / len(shared),
            "semantic_f1_margin": sum(semantic_margins.values()) / len(shared),
            "loss_margin": sum(loss_margins.values()) / len(shared),
            "phantom_rate_difference": sum(phantom_differences.values())
            / len(shared),
            "token_f1_bootstrap": bootstrap_margin(f1_margins),
            "semantic_f1_bootstrap": bootstrap_margin(semantic_margins),
            "loss_bootstrap": bootstrap_margin(loss_margins),
        }
    return {
        "cases": len(labels_by_id),
        "controls": metrics,
        "dependence": dependence,
        "claim": "narrow_alignment_and_semantic_descriptor_metric_only",
        "bootstrap": dict(bootstrap),
        "case_ids": sorted(labels_by_id),
        "case_ids_sha256": fingerprint(sorted(labels_by_id)),
        "rows": [row for control in controls
                 for row in rows_by_control[control].values()],
    }


def diagnostic_gate_report(*, label: str, answer: Mapping[str, Any] | None,
                           caption: Mapping[str, Any] | None,
                           evaluation: Mapping[str, Any]
                           ) -> dict[str, Any]:
    failures: list[str] = []
    if answer is not None:
        correct = answer["controls"].get("correct")
        _require(correct and correct.get("available"),
                 f"{label} lacks a correct answer control")
        if correct["accuracy"] < evaluation["min_correct_accuracy"]:
            failures.append("min_correct_accuracy")
        for control, key in (("wrong", "min_margin_wrong"),
                             ("shuffled", "min_margin_shuffled"),
                             ("noimage", "min_margin_noimage")):
            if control not in answer["controls"]:
                continue
            margin = answer["paired"].get(control, {}).get("correct_margin")
            if margin is None or margin < evaluation[key]:
                failures.append(f"min_margin_{control}")
        counterfactual = answer.get("counterfactual_answerable")
        if counterfactual is not None and counterfactual["correct_margin"] \
                < evaluation["min_counterfactual_margin_answerable"]:
            failures.append("min_counterfactual_margin_answerable")
        unknown = answer.get("unknown", {})
        if unknown.get("abstention_accuracy", 0.0) \
                < evaluation["min_unknown_accuracy"]:
            failures.append("min_unknown_accuracy")
        if unknown.get("false_answer_on_abstain", 1.0) \
                > evaluation["max_false_answer_on_abstain"]:
            failures.append("max_false_answer_on_abstain")
    if caption is not None:
        correct = caption["controls"]["correct"]
        if correct["token_f1"] < evaluation["min_caption_token_f1"]:
            failures.append("min_caption_token_f1")
        if correct["semantic_f1"] \
                < evaluation["min_caption_semantic_descriptor_f1"]:
            failures.append("min_caption_semantic_descriptor_f1")
        for control in ("counterfactual", "shuffled", "noimage"):
            dependence = caption["dependence"][control]
            if dependence["token_f1_margin"] \
                    < evaluation["min_caption_dependence_f1_margin"]:
                failures.append(f"min_caption_f1_margin_{control}")
            if dependence["loss_margin"] \
                    < evaluation["min_caption_dependence_loss_margin"]:
                failures.append(f"min_caption_loss_margin_{control}")
    return {
        "label": label,
        "diagnostic_only": True,
        "promotion": False,
        "gate_failures": failures,
        "all_declared_gates_passed": not failures,
        "scope": evaluation["gates_scope"],
    }


def compare_reload_artifacts(original: str | Path, reloaded: str | Path,
                             names: Sequence[str]) -> dict[str, bool]:
    original = Path(original)
    reloaded = Path(reloaded)
    comparisons = {}
    for name in names:
        left = original / name
        right = reloaded / name
        _require(left.is_file() and right.is_file(),
                 f"Reload artifact is missing: {name}")
        comparisons[name] = sha256_file(left) == sha256_file(right)
    return comparisons


# ---------------------------------------------------------------------------
# V2 caption/answer target contracts and scoring
# ---------------------------------------------------------------------------

_CAPTION_CONTROLS = ("correct", "counterfactual", "shuffled", "noimage")


def _descriptor_list(value: Any, label: str) -> list[list[str]]:
    _require(isinstance(value, (list, tuple)) and value,
             f"{label} descriptors are missing")
    result = []
    for item in value:
        _require(isinstance(item, (list, tuple)) and len(item) == 2,
                 f"{label} descriptor is malformed")
        color, shape = str(item[0]), str(item[1])
        _require(color in _COLOR_SET and shape in _SHAPE_SET,
                 f"{label} descriptor is not a sealed colour/shape pair")
        result.append([color, shape])
    return result


def _require_caption_target_text(text: Any, objects: Sequence[Sequence[str]],
                                 label: str) -> str:
    _require(isinstance(text, str) and text.strip(),
             f"{label} target text is missing")
    parsed = Counter(caption_descriptors(text))
    expected = Counter(tuple(item) for item in objects)
    _require(parsed == expected,
             f"{label} target text and sealed descriptors differ")
    return text


def _control_row(control_map: Mapping[str, Mapping[str, Any]],
                 case_id: str) -> Mapping[str, Any]:
    _require(case_id in control_map,
             f"Sealed control assignment is missing: {case_id}")
    row = control_map[case_id]
    _require(isinstance(row.get("assignment_sha256"), str)
             and row["assignment_sha256"] == control_assignment_sha256(row),
             f"Sealed control assignment bytes differ: {case_id}")
    return row


def caption_target_specs_v2(cases: Sequence[Mapping[str, Any]],
                            labels: Sequence[Mapping[str, Any]],
                            control_map: Mapping[str, Mapping[str, Any]],
                            *, controls: Sequence[str] = _CAPTION_CONTROLS
                            ) -> dict[tuple[str, str], dict[str, Any]]:
    """Derive every caption target from sealed labels and assignments.

    The original sealed target is fixed for every image control.  Following
    targets describe only the image actually supplied for that control.
    """
    case_by_id = {case["case_id"]: case for case in cases}
    label_by_id = {label["case_id"]: label for label in labels}
    _require(len(case_by_id) == len(cases), "Caption cases repeat")
    _require(len(label_by_id) == len(labels), "Caption labels repeat")
    _require(set(case_by_id) == set(label_by_id),
             "Caption cases and sealed labels differ")
    _require(isinstance(controls, (list, tuple)) and controls,
             "Caption controls are required")
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for case_id in sorted(case_by_id):
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        _require(_is_sha256(case.get("image_sha256")),
                 f"Caption case image hash is invalid: {case_id}")
        original_objects = _descriptor_list(
            label.get("visible_objects"), f"{case_id} original")
        original_text = _require_caption_target_text(
            label.get("answer"), original_objects, f"{case_id} original")
        row = _control_row(control_map, case_id)
        for control in controls:
            spec = {
                "case_id": case_id,
                "control": control,
                "original_text": original_text,
                "original_objects": [list(item) for item in original_objects],
                "original_image_case_id": case_id,
                "original_image_sha256": case["image_sha256"],
                "expected_image_case_id": case_id,
                "expected_image_sha256": case["image_sha256"],
                "following_text": original_text,
                "following_objects": [list(item)
                                      for item in original_objects],
                "following_available": True,
                "image_changed": False,
                "assignment_sha256": row["assignment_sha256"],
            }
            if control == "counterfactual":
                donor = row.get("counterfactual")
                _require(isinstance(donor, Mapping),
                         f"Counterfactual assignment is missing: {case_id}")
                _require(donor.get("case_id")
                         == case.get("counterfactual_case_id"),
                         f"Counterfactual case identity differs: {case_id}")
                _require(donor.get("image_sha256")
                         == case.get("counterfactual_image_sha256"),
                         f"Counterfactual image hash differs: {case_id}")
                following_objects = _descriptor_list(
                    label.get("counterfactual_visible_objects"),
                    f"{case_id} counterfactual")
                following_text = _require_caption_target_text(
                    label.get("counterfactual_answer"), following_objects,
                    f"{case_id} counterfactual")
                if isinstance(donor.get("visible_objects"), list):
                    _require(Counter(map(tuple, donor["visible_objects"]))
                             == Counter(map(tuple, following_objects)),
                             "Counterfactual assignment descriptors differ")
                spec.update({
                    "expected_image_case_id": donor.get("case_id"),
                    "expected_image_sha256": donor.get("image_sha256"),
                    "following_text": following_text,
                    "following_objects": [list(item)
                                          for item in following_objects],
                    "image_changed": (donor.get("image_sha256")
                                      != case["image_sha256"]),
                })
                _require(spec["image_changed"] is True,
                         "Counterfactual image does not change")
                _require(bool(row.get("counterfactual_same_image"))
                         is False,
                         "Counterfactual assignment reuses the image")
            elif control == "shuffled":
                donor = row.get("shuffled")
                _require(isinstance(donor, Mapping),
                         f"Shuffled assignment is missing: {case_id}")
                following_objects = _descriptor_list(
                    donor.get("visible_objects"), f"{case_id} shuffled")
                following_text = _require_caption_target_text(
                    donor.get("caption"), following_objects,
                    f"{case_id} shuffled")
                _require(donor.get("image_sha256") != case["image_sha256"],
                         "Shuffled assignment reuses the receiver image")
                spec.update({
                    "expected_image_case_id": donor.get("case_id"),
                    "expected_image_sha256": donor.get("image_sha256"),
                    "following_text": following_text,
                    "following_objects": [list(item)
                                          for item in following_objects],
                    "image_changed": True,
                })
            elif control == "noimage":
                spec.update({
                    "expected_image_case_id": None,
                    "expected_image_sha256": None,
                    "following_text": None,
                    "following_objects": None,
                    "following_available": False,
                    "image_changed": True,
                })
            elif control != "correct":
                raise ValueError(f"Undeclared caption control: {control}")
            specs[(case_id, control)] = spec
    return specs


def _validate_caption_generation_v2(generation: Mapping[str, Any],
                                    spec: Mapping[str, Any]
                                    ) -> None:
    required = {
        "control", "case_id", "generated_text", "image_sha256",
        "image_case_id", "target_case_id", "target_text",
        "target_visible_objects", "response_loss",
        "control_assignment_sha256",
    }
    missing = sorted(required - set(generation))
    _require(not missing, f"Caption generation fields are missing: {missing}")
    _require(generation["case_id"] == spec["case_id"],
             "Caption generation case differs")
    _require(generation["control"] == spec["control"],
             "Caption generation control differs")
    _require(isinstance(generation["generated_text"], str),
             "Caption generated text must be a string")
    _require(generation.get("control_assignment_sha256")
             == spec["assignment_sha256"],
             "Caption control assignment binding differs")
    _require(generation.get("image_case_id") == spec["expected_image_case_id"]
             and generation.get("image_sha256")
             == spec["expected_image_sha256"],
             "Caption generation image identity differs from the assignment")
    _require(generation.get("target_case_id")
             == spec["original_image_case_id"],
             "Caption generation target case was altered")
    _require(generation.get("target_text") == spec["original_text"],
             "Caption generation target text was altered")
    try:
        target_objects = [list(item)
                          for item in generation["target_visible_objects"]]
    except (TypeError, ValueError) as error:
        raise ValueError("Caption generation target descriptors are malformed") \
            from error
    _require(target_objects == spec["original_objects"],
             "Caption generation target descriptors were altered")
    loss = generation.get("response_loss")
    _require(isinstance(loss, (int, float)) and not isinstance(loss, bool)
             and math.isfinite(float(loss)) and float(loss) >= 0,
             "Caption teacher-forced loss is missing or invalid")


def _caption_row_metrics_v2(generation: Mapping[str, Any],
                            spec: Mapping[str, Any]) -> dict[str, Any]:
    generated = str(generation["generated_text"])
    original = semantic_caption_agreement(generated, spec["original_objects"])
    following = None
    if spec["following_available"]:
        following = semantic_caption_agreement(generated,
                                               spec["following_objects"])
    return {
        **generation,
        "original_token_f1": caption_token_f1(
            generated, spec["original_text"]),
        "original_semantic": original,
        "original_semantic_f1": original["f1"],
        "original_exact_agreement": original["exact_agreement"],
        "following_target_case_id": spec["expected_image_case_id"],
        "following_text": spec["following_text"],
        "following_objects": spec["following_objects"],
        "following_available": spec["following_available"],
        "following_semantic": following,
        "following_semantic_f1": (None if following is None
                                  else following["f1"]),
        "following_accuracy": (None if following is None
                               else float(following["exact_agreement"])),
        "image_changed": spec["image_changed"],
        "response_loss": float(generation["response_loss"]),
        "contract_version": CAPTION_CONTRACT_VERSION,
    }


def _caption_summary_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(rows, "Caption control has no rows")
    following = [row for row in rows if row["following_available"]]
    return {
        "cases": len(rows),
        "image_changed_cases": sum(bool(row["image_changed"])
                                   for row in rows),
        "original_token_f1": sum(row["original_token_f1"]
                                 for row in rows) / len(rows),
        "original_semantic_f1": sum(row["original_semantic_f1"]
                                    for row in rows) / len(rows),
        "original_exact_agreement": sum(
            row["original_exact_agreement"] for row in rows) / len(rows),
        "phantom_object_rate": sum(
            row["original_semantic"]["phantom_objects"] > 0
            for row in rows) / len(rows),
        "following_available_cases": len(following),
        "following_semantic_f1": (
            sum(row["following_semantic_f1"] for row in following)
            / len(following) if following else None),
        "following_accuracy": (
            sum(row["following_accuracy"] for row in following)
            / len(following) if following else None),
        "teacher_forced_response_loss": sum(
            row["response_loss"] for row in rows) / len(rows),
    }


def score_caption_controls_v2(cases: Sequence[Mapping[str, Any]],
                              labels: Sequence[Mapping[str, Any]],
                              generations: Sequence[Mapping[str, Any]],
                              *, control_map: Mapping[str, Mapping[str, Any]],
                              controls: Sequence[str], seed: int,
                              bootstrap: Mapping[str, Any],
                              gates: Mapping[str, Any] | None = None
                              ) -> dict[str, Any]:
    """Score fixed-target teacher-forced loss and changed-image following."""
    controls = list(controls)
    _require(controls and "correct" in controls
             and set(controls) <= set(_CAPTION_CONTROLS),
             "V2 caption scoring requires the correct control")
    specs = caption_target_specs_v2(
        cases, labels, control_map, controls=controls)
    labels_by_id = {row["case_id"]: row for row in labels}
    rows_by_control: dict[str, dict[str, dict[str, Any]]] = {
        control: {} for control in controls}
    for generation in generations:
        _require(isinstance(generation, Mapping),
                 "Caption generation must be a mapping")
        control = generation.get("control")
        case_id = generation.get("case_id")
        _require(control in controls, f"Unknown caption control: {control}")
        _require((case_id, control) in specs,
                 f"Undeclared caption target: {control}/{case_id}")
        _require(case_id not in rows_by_control[control],
                 f"Duplicate caption generation: {control}/{case_id}")
        spec = specs[(case_id, control)]
        _validate_caption_generation_v2(generation, spec)
        rows_by_control[control][case_id] = _caption_row_metrics_v2(
            generation, spec)
    for control in controls:
        _require(set(rows_by_control[control]) == set(labels_by_id),
                 f"Caption control is incomplete: {control}")

    metrics = {control: _caption_summary_v2(
        list(rows_by_control[control].values())) for control in controls}
    correct = rows_by_control["correct"]

    def bootstrap_margin(values, control):
        by_world: dict[str, list[float]] = defaultdict(list)
        for case_id, value in values.items():
            by_world[labels_by_id[case_id]["world_id"]].append(value)
        return clustered_bootstrap(
            by_world, resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence"]),
            seed=seed + 211 * (controls.index(control) + 1))

    dependence: dict[str, dict[str, Any]] = {}
    for control in ("counterfactual", "shuffled"):
        if control not in controls:
            continue
        other = rows_by_control[control]
        shared = sorted(set(correct) & set(other))
        _require(shared, f"No paired caption cases for {control}")
        original_drop = {}
        following_accuracy = {}
        following_semantic = {}
        loss_delta = {}
        original_exact_drop = {}
        for case_id in shared:
            _require(correct[case_id]["following_available"]
                     and other[case_id]["following_available"],
                     "Caption dependence requires image-following targets")
            original_drop[case_id] = (
                correct[case_id]["original_semantic_f1"]
                - other[case_id]["original_semantic_f1"])
            original_exact_drop[case_id] = (
                float(correct[case_id]["original_exact_agreement"])
                - float(other[case_id]["original_exact_agreement"]))
            following_accuracy[case_id] = float(
                other[case_id]["following_accuracy"])
            following_semantic[case_id] = float(
                other[case_id]["following_semantic_f1"])
            loss_delta[case_id] = (
                other[case_id]["response_loss"]
                - correct[case_id]["response_loss"])
        dependence[control] = {
            "cases": len(shared),
            "following_accuracy": sum(following_accuracy.values())
            / len(shared),
            "following_semantic_f1": sum(following_semantic.values())
            / len(shared),
            "original_semantic_f1_drop": sum(original_drop.values())
            / len(shared),
            "original_exact_drop": sum(original_exact_drop.values())
            / len(shared),
            "following_accuracy_bootstrap": bootstrap_margin(
                following_accuracy, control),
            "original_semantic_drop_bootstrap": bootstrap_margin(
                original_drop, control),
            "fixed_target_loss_delta": sum(loss_delta.values()) / len(shared),
            "loss_delta_bootstrap": bootstrap_margin(loss_delta, control),
        }
    noimage = rows_by_control.get("noimage")
    noimage_summary = None
    if noimage is not None:
        noimage_summary = {
            "cases": len(noimage),
            "original_semantic_f1": metrics["noimage"][
                "original_semantic_f1"],
            "original_exact_agreement": metrics["noimage"][
                "original_exact_agreement"],
            "phantom_object_rate": metrics["noimage"]["phantom_object_rate"],
            "teacher_forced_response_loss": metrics["noimage"][
                "teacher_forced_response_loss"],
            "loss_margin": (
                metrics["noimage"]["teacher_forced_response_loss"]
                - metrics["correct"]["teacher_forced_response_loss"]),
            "following_available": False,
            "scope": "no-image behavior is reported separately from following",
        }
    result = {
        "format": CAPTION_SCORE_FORMAT,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "cases": len(labels_by_id),
        "controls": metrics,
        "dependence": dependence,
        "noimage": noimage_summary,
        "claim": "fixed_target_loss_and_changed_image_following_only",
        "bootstrap": dict(bootstrap),
        "case_ids": sorted(labels_by_id),
        "case_ids_sha256": fingerprint(sorted(labels_by_id)),
        "raw_generations": [dict(row) for row in generations],
        "rows": [row for control in controls
                 for row in rows_by_control[control].values()],
    }
    if gates is not None:
        result["gate_failures"] = diagnostic_gate_report_v2(
            label="caption_score", answer=None, caption=result,
            evaluation={"gates": dict(gates)})["gate_failures"]
    return result


# ---------------------------------------------------------------------------
# V2 answer counterfactual-following scoring
# ---------------------------------------------------------------------------


def answer_target_specs_v2(cases: Sequence[Mapping[str, Any]],
                           labels: Sequence[Mapping[str, Any]],
                           control_map: Mapping[str, Mapping[str, Any]],
                           *, controls: Sequence[str]
                           ) -> dict[tuple[str, str], dict[str, Any]]:
    case_by_id = {case["case_id"]: case for case in cases}
    label_by_id = {label["case_id"]: label for label in labels}
    _require(set(case_by_id) == set(label_by_id),
             "Answer cases and sealed labels differ")
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for case_id in sorted(case_by_id):
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        _require(isinstance(label.get("answer"), str) and label["answer"],
                 f"Answer target is missing: {case_id}")
        _require(_is_sha256(case.get("image_sha256")),
                 f"Answer case image hash is invalid: {case_id}")
        row = _control_row(control_map, case_id)
        for control in controls:
            spec = {
                "case_id": case_id,
                "control": control,
                "original_answer": label["answer"],
                "expected_image_case_id": case_id,
                "expected_image_sha256": case["image_sha256"],
                "following_answer": label["answer"],
                "following_applicable": control != "noimage",
                "following_scope": "original_image",
                "assignment_sha256": row["assignment_sha256"],
            }
            if control == "counterfactual":
                donor = row.get("counterfactual")
                _require(isinstance(donor, Mapping),
                         f"Counterfactual assignment is missing: {case_id}")
                _require(donor.get("case_id")
                         == case.get("counterfactual_case_id")
                         and donor.get("image_sha256")
                         == case.get("counterfactual_image_sha256"),
                         f"Counterfactual assignment differs: {case_id}")
                _require(isinstance(label.get("counterfactual_answer"), str)
                         and label["counterfactual_answer"],
                         "Counterfactual answer target is missing")
                if isinstance(donor.get("sealed_answer"), str):
                    _require(donor["sealed_answer"]
                             == label["counterfactual_answer"],
                             "Counterfactual sealed answers differ")
                spec.update({
                    "expected_image_case_id": donor["case_id"],
                    "expected_image_sha256": donor["image_sha256"],
                    "following_answer": label["counterfactual_answer"],
                    "following_scope": "changed_image",
                })
            elif control in ("wrong", "shuffled"):
                donor = row.get(control)
                _require(isinstance(donor, Mapping),
                         f"{control} assignment is missing: {case_id}")
                sealed = donor.get("sealed_answer")
                _require(isinstance(sealed, str) and sealed,
                         f"{control} sealed answer is missing: {case_id}")
                spec.update({
                    "expected_image_case_id": donor["case_id"],
                    "expected_image_sha256": donor["image_sha256"],
                    "following_answer": sealed,
                    "following_applicable": donor.get("task")
                    == case.get("task"),
                    "following_scope": ("same_task_changed_image"
                                        if donor.get("task")
                                        == case.get("task")
                                        else "not_applicable_cross_task"),
                })
            elif control == "noimage":
                spec.update({
                    "expected_image_case_id": None,
                    "expected_image_sha256": None,
                    "following_answer": None,
                    "following_applicable": False,
                    "following_scope": "no_image",
                })
            elif control != "correct":
                raise ValueError(f"Undeclared answer control: {control}")
            specs[(case_id, control)] = spec
    return specs


_ANSWER_GENERATION_FIELDS = {
    "control", "case_id", "generated_text", "image_sha256",
    "image_case_id", "control_assignment_sha256",
}


def _validate_answer_generation_v2(generation: Mapping[str, Any],
                                   spec: Mapping[str, Any]) -> None:
    missing = sorted(_ANSWER_GENERATION_FIELDS - set(generation))
    _require(not missing, f"Answer generation fields are missing: {missing}")
    _require(generation["case_id"] == spec["case_id"]
             and generation["control"] == spec["control"],
             "Answer generation identity differs")
    _require(isinstance(generation["generated_text"], str),
             "Answer generated text must be a string")
    _require(generation.get("control_assignment_sha256")
             == spec["assignment_sha256"],
             "Answer control assignment binding differs")
    _require(generation.get("image_case_id") == spec["expected_image_case_id"]
             and generation.get("image_sha256")
             == spec["expected_image_sha256"],
             "Answer generation image identity differs from the assignment")


def score_answer_controls_v2(cases: Sequence[Mapping[str, Any]],
                             labels: Sequence[Mapping[str, Any]],
                             generations: Sequence[Mapping[str, Any]],
                             *, control_map: Mapping[str, Mapping[str, Any]],
                             controls: Sequence[str], seed: int,
                             bootstrap: Mapping[str, Any],
                             gates: Mapping[str, Any] | None = None
                             ) -> dict[str, Any]:
    """Keep original-label correctness and changed-image following separate."""
    from .vision_curriculum import answer_correct

    controls = list(controls)
    case_by_id = {case["case_id"]: case for case in cases}
    label_by_id = {label["case_id"]: label for label in labels}
    _require(set(case_by_id) == set(label_by_id),
             "Answer cases and sealed labels differ")
    for case in cases:
        _require(case.get("label") == label_by_id[case["case_id"]]["answer"],
                 "Answer case label differs from the sealed label")
    specs = answer_target_specs_v2(cases, labels, control_map,
                                   controls=controls)
    seen: set[tuple[str, str]] = set()
    for generation in generations:
        key = (generation.get("case_id"), generation.get("control"))
        _require(key in specs, f"Undeclared answer target: {key}")
        _require(key not in seen, f"Duplicate answer generation: {key}")
        seen.add(key)
        _validate_answer_generation_v2(generation, specs[key])
    for control in controls:
        covered = {case_id for case_id, name in seen if name == control}
        _require(covered == set(case_by_id),
                 f"Answer control is incomplete: {control}")
    score = score_procedural(
        cases, generations, controls=controls, seed=seed,
        bootstrap_resamples=int(bootstrap["resamples"]),
        bootstrap_confidence=float(bootstrap["confidence"]), gates=None)
    generated_by_key = {(row["case_id"], row["control"]):
                        str(row["generated_text"]) for row in generations}
    following: dict[str, dict[str, Any]] = {}
    for control in controls:
        applicable = [case_id for case_id in sorted(case_by_id)
                      if specs[(case_id, control)]["following_applicable"]]
        correct_cases = [case_id for case_id in applicable
                         if answer_correct(
                             generated_by_key[(case_id, control)],
                             specs[(case_id, control)]["following_answer"])]
        following[control] = {
            "cases": len(applicable),
            "correct": len(correct_cases),
            "accuracy": (len(correct_cases) / len(applicable)
                         if applicable else None),
            "following_scope": specs[(applicable[0], control)][
                "following_scope"] if applicable else "no_applicable_cases",
        }
    answerable = []
    if "counterfactual" in controls:
        answerable = [
            case_id for case_id in sorted(case_by_id)
            if control_map[case_id].get("counterfactual_same_image")
            is not True
            and label_by_id[case_id].get("answer") != UNKNOWN_ANSWER
            and label_by_id[case_id].get(
                "answer_changes_under_counterfactual",
                label_by_id[case_id].get("answer")
                != label_by_id[case_id].get("counterfactual_answer")) is True
        ]
        _require(answerable, "No answerable counterfactual pairs exist")
    else:
        answerable = []
    correct_rows = {row["case_id"]: row for row in score["rows"]
                    if row["control"] == "correct"}
    cf_rows = {row["case_id"]: row for row in score["rows"]
               if row["control"] == "counterfactual"}
    margins: dict[str, float] = {}
    following_answerable: list[str] = []
    if answerable:
        following_answerable = [
            case_id for case_id in answerable
            if answer_correct(generated_by_key[(case_id, "counterfactual")],
                              specs[(case_id, "counterfactual")][
                                  "following_answer"])]
        margins = {
            case_id: (float(correct_rows[case_id]["correct"])
                      - float(cf_rows[case_id]["correct"]))
            for case_id in answerable
        }
        by_world: dict[str, list[float]] = defaultdict(list)
        for case_id, value in margins.items():
            by_world[case_by_id[case_id]["world_id"]].append(value)
        score["counterfactual_answerable"] = {
            "cases": len(answerable),
            "correct_accuracy": sum(correct_rows[case_id]["correct"]
                                    for case_id in answerable)
            / len(answerable),
            "counterfactual_original_accuracy":
                sum(cf_rows[case_id]["correct"] for case_id in answerable)
                / len(answerable),
            "original_label_margin": sum(margins.values()) / len(answerable),
            "margin_bootstrap": clustered_bootstrap(
                by_world, resamples=int(bootstrap["resamples"]),
                confidence=float(bootstrap["confidence"]),
                seed=seed + 7919),
            "following_cases": len(answerable),
            "following_correct": len(following_answerable),
            "following_accuracy": len(following_answerable) / len(answerable),
            "scope": ("answerable cases only; original-label correctness and "
                      "changed-image following are separate"),
        }
    score.update({
        "format": ANSWER_SCORE_FORMAT,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "following": following,
        "raw_generations": [dict(row) for row in generations],
        "claim": ("original-label correctness, changed-image following and "
                  "change-only dependence are reported separately"),
    })
    if gates is not None:
        score["gate_report"] = diagnostic_gate_report_v2(
            label="answer_score", answer=score, caption=None,
            evaluation={"gates": dict(gates)})
    return score


# ---------------------------------------------------------------------------
# V2 gate semantics
# ---------------------------------------------------------------------------


def diagnostic_gate_report_v2(*, label: str,
                              answer: Mapping[str, Any] | None,
                              caption: Mapping[str, Any] | None,
                              evaluation: Mapping[str, Any]
                              ) -> dict[str, Any]:
    gates = dict(evaluation.get("gates", evaluation))
    for nested, flat in (
            ("min_correct_margin_wrong", "min_margin_wrong"),
            ("min_correct_margin_shuffled", "min_margin_shuffled"),
            ("min_correct_margin_noimage", "min_margin_noimage"),
            ("max_unknown_rate", "max_false_abstention_on_answerable")):
        gates.setdefault(nested, gates.get(flat, 0.0))
    failures: list[str] = []
    if answer is not None:
        correct = answer["controls"].get("correct")
        _require(correct and correct.get("available"),
                 f"{label} lacks a correct answer control")
        if correct["accuracy"] < gates["min_correct_accuracy"]:
            failures.append("min_correct_accuracy")
        for control, key in (("wrong", "min_correct_margin_wrong"),
                             ("shuffled", "min_correct_margin_shuffled"),
                             ("noimage", "min_correct_margin_noimage")):
            if control not in answer["controls"]:
                failures.append(f"missing_control_{control}")
                continue
            margin = answer["paired"].get(control, {}).get("correct_margin")
            if margin is None or margin < gates[key]:
                failures.append(key)
        counterfactual = answer.get("counterfactual_answerable")
        if "counterfactual" in answer["controls"]:
            if counterfactual is None:
                failures.append("missing_counterfactual_answerable")
            else:
                if counterfactual["original_label_margin"] \
                        < gates["min_counterfactual_margin_answerable"]:
                    failures.append("min_counterfactual_margin_answerable")
                if counterfactual["following_accuracy"] \
                        < gates["min_counterfactual_following_accuracy"]:
                    failures.append("min_counterfactual_following_accuracy")
        for control, key in (
                ("wrong", "min_following_accuracy_wrong"),
                ("shuffled", "min_following_accuracy_shuffled")):
            values = answer.get("following", {}).get(control, {})
            accuracy = values.get("accuracy")
            if accuracy is not None and accuracy < gates[key]:
                failures.append(key)
        unknown = answer.get("unknown", {})
        if unknown.get("abstention_accuracy", 0.0) \
                < gates["min_unknown_accuracy"]:
            failures.append("min_unknown_accuracy")
        if unknown.get("false_answer_on_abstain", 1.0) \
                > gates["max_false_answer_on_abstain"]:
            failures.append("max_false_answer_on_abstain")
        if unknown.get("false_abstention_on_answerable", 1.0) \
                > gates["max_false_abstention_on_answerable"]:
            failures.append("max_false_abstention_on_answerable")
    if caption is not None:
        correct = caption["controls"]["correct"]
        if correct["original_token_f1"] < gates["min_caption_token_f1"]:
            failures.append("min_caption_token_f1")
        if correct["original_semantic_f1"] \
                < gates["min_caption_semantic_descriptor_f1"]:
            failures.append("min_caption_semantic_descriptor_f1")
        accuracy = correct.get("following_accuracy")
        if accuracy is None or accuracy < gates[
                "min_caption_following_accuracy"]:
            failures.append("min_caption_following_accuracy")
        for control in ("counterfactual", "shuffled"):
            if control not in caption["dependence"]:
                continue
            dependence = caption["dependence"][control]
            if dependence["following_accuracy"] \
                    < gates["min_caption_following_accuracy"]:
                failures.append(
                    f"min_caption_following_accuracy_{control}")
            if dependence["following_semantic_f1"] \
                    < gates["min_caption_following_semantic_f1"]:
                failures.append(
                    f"min_caption_following_semantic_f1_{control}")
            if dependence["original_semantic_f1_drop"] \
                    < gates["min_caption_original_agreement_drop"]:
                failures.append(
                    f"min_caption_original_agreement_drop_{control}")
        noimage = caption.get("noimage")
        if noimage is not None:
            if noimage["loss_margin"] < gates["min_noimage_loss_margin"]:
                failures.append("min_noimage_loss_margin")
            if noimage["original_semantic_f1"] \
                    > gates["max_noimage_original_semantic_f1"]:
                failures.append("max_noimage_original_semantic_f1")
    return {
        "label": label,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "diagnostic_only": True,
        "promotion": False,
        "gate_failures": failures,
        "all_declared_gates_passed": not failures,
        "scope": evaluation.get("gates_scope", "diagnostic only"),
    }


# ---------------------------------------------------------------------------
# V2 positive and negative control contracts
# ---------------------------------------------------------------------------

_CAPTION_PROBE_LOSSES = {
    "correct": 0.1, "counterfactual": 0.2, "shuffled": 0.2, "noimage": 8.0,
}


def _caption_probe_rows_v2(cases, labels, control_map, controls, mode
                           ) -> list[dict[str, Any]]:
    specs = caption_target_specs_v2(
        cases, labels, control_map, controls=controls)
    case_ids = sorted({case["case_id"] for case in cases})
    rows = []
    for case_id in case_ids:
        for control in controls:
            spec = specs[(case_id, control)]
            if mode == "perfect_image_following":
                generated = ("" if not spec["following_available"]
                             else spec["following_text"])
                loss = _CAPTION_PROBE_LOSSES[control]
            elif mode == "image_ignoring":
                generated = spec["original_text"]
                loss = 0.2
            elif mode == "constant_caption":
                generated = "a red circle"
                loss = 0.5
            elif mode == "shuffled":
                donor_index = (case_ids.index(case_id) + 1) % len(case_ids)
                donor_id = case_ids[donor_index]
                donor_spec = specs[(donor_id, control)]
                generated = (spec["original_text"]
                             if not donor_spec["following_available"]
                             else donor_spec["following_text"])
                loss = _CAPTION_PROBE_LOSSES[control]
            else:
                raise ValueError(f"Unknown caption probe mode: {mode}")
            rows.append({
                "control": control,
                "case_id": case_id,
                "generated_text": generated,
                "image_sha256": spec["expected_image_sha256"],
                "image_case_id": spec["expected_image_case_id"],
                "target_case_id": spec["original_image_case_id"],
                "target_text": spec["original_text"],
                "target_visible_objects": [list(item)
                                           for item in
                                           spec["original_objects"]],
                "response_loss": loss,
                "control_assignment_sha256": spec["assignment_sha256"],
            })
    return rows


def _answer_probe_rows_v2(cases, labels, control_map, controls, mode
                          ) -> list[dict[str, Any]]:
    specs = answer_target_specs_v2(cases, labels, control_map,
                                   controls=controls)
    case_ids = sorted({case["case_id"] for case in cases})
    label_by_id = {label["case_id"]: label for label in labels}
    rows = []
    for case_id in case_ids:
        for control in controls:
            spec = specs[(case_id, control)]
            if mode == "perfect_image_following":
                generated = (spec["following_answer"]
                             if spec["following_applicable"]
                             else UNKNOWN_ANSWER)
            elif mode == "image_ignoring":
                generated = spec["original_answer"]
            elif mode == "constant_answer":
                generated = UNKNOWN_ANSWER
            elif mode == "shuffled":
                donor_index = (case_ids.index(case_id) + 1) % len(case_ids)
                donor_id = case_ids[donor_index]
                donor_spec = specs[(donor_id, control)]
                generated = (donor_spec["following_answer"]
                             if donor_spec["following_applicable"]
                             else label_by_id[donor_id]["answer"])
            else:
                raise ValueError(f"Unknown answer probe mode: {mode}")
            rows.append({
                "control": control,
                "case_id": case_id,
                "generated_text": generated,
                "image_sha256": spec["expected_image_sha256"],
                "image_case_id": spec["expected_image_case_id"],
                "control_assignment_sha256": spec["assignment_sha256"],
            })
    return rows


def verify_caption_control_contracts_v2(
        cases, labels, control_map, *, controls=_CAPTION_CONTROLS,
        seed: int = 20261028, bootstrap: Mapping[str, Any] | None = None,
        evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Positive must pass every declared gate; negatives must fail/raise."""
    controls = list(controls)
    bootstrap = dict(bootstrap or {"cluster": "world", "confidence": 0.95,
                                   "resamples": 200})
    positive_rows = _caption_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    positive = score_caption_controls_v2(
        cases, labels, positive_rows, control_map=control_map,
        controls=controls, seed=seed, bootstrap=bootstrap)
    report = diagnostic_gate_report_v2(
        label="positive_perfect_image_following", answer=None,
        caption=positive, evaluation=evaluation)
    _require(report["all_declared_gates_passed"],
             "Perfect caption image-following control fails intended gates: "
             f"{report['gate_failures']}")
    negative_failures: dict[str, list[str]] = {}
    expected = {
        "constant_caption": {"min_caption_following_accuracy_counterfactual"},
        "image_ignoring": {
            "min_caption_original_agreement_drop_counterfactual",
            "min_caption_following_accuracy_counterfactual",
        },
        "shuffled": {"min_caption_following_accuracy_counterfactual",
                     "min_caption_following_accuracy_shuffled"},
    }
    for mode, keys in expected.items():
        rows = _caption_probe_rows_v2(cases, labels, control_map,
                                      controls, mode)
        score = score_caption_controls_v2(
            cases, labels, rows, control_map=control_map, controls=controls,
            seed=seed + 1, bootstrap=bootstrap)
        negative = diagnostic_gate_report_v2(
            label=f"negative_{mode}", answer=None, caption=score,
            evaluation=evaluation)
        negative_failures[mode] = negative["gate_failures"]
        _require(keys & set(negative["gate_failures"]),
                 f"Caption negative control {mode} did not fail its "
                 f"intended gate: {negative['gate_failures']}")
    malformed = _caption_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    malformed[0] = dict(malformed[0])
    malformed[0].pop("response_loss")
    try:
        score_caption_controls_v2(
            cases, labels, malformed, control_map=control_map,
            controls=controls, seed=seed + 2, bootstrap=bootstrap)
    except ValueError as error:
        malformed_error = str(error)
    else:
        raise ValueError("Malformed caption generation was accepted")
    tampered = _caption_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    tampered[0] = dict(tampered[0])
    tampered[0]["target_text"] = "a purple hexagon"
    try:
        score_caption_controls_v2(
            cases, labels, tampered, control_map=control_map,
            controls=controls, seed=seed + 3, bootstrap=bootstrap)
    except ValueError as error:
        tampered_error = str(error)
    else:
        raise ValueError("Target-tampered caption generation was accepted")
    return {
        "format": SCORER_CONTRACT_FORMAT,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "controls": controls,
        "positive": {
            "mode": "perfect_image_following",
            "all_declared_gates_passed": True,
            "gate_failures": [],
        },
        "negative": {
            "gate_failures": {key: sorted(value)
                              for key, value in negative_failures.items()},
            "malformed_rejected": malformed_error,
            "target_tampering_rejected": tampered_error,
        },
        "assignment_bindings": len(
            {(row["case_id"], row["assignment_sha256"])
             for row in control_map.values()}),
        "cases": len({case["case_id"] for case in cases}),
        "all_contracts_passed": True,
    }


def verify_answer_control_contracts_v2(
        cases, labels, control_map, *, controls, seed: int = 20261028,
        bootstrap: Mapping[str, Any] | None = None,
        evaluation: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = dict(bootstrap or {"cluster": "world", "confidence": 0.95,
                                   "resamples": 200})
    controls = list(controls)
    positive_rows = _answer_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    positive = score_answer_controls_v2(
        cases, labels, positive_rows, control_map=control_map,
        controls=controls, seed=seed, bootstrap=bootstrap)
    report = diagnostic_gate_report_v2(
        label="positive_perfect_answer_following", answer=positive,
        caption=None, evaluation=evaluation)
    _require(report["all_declared_gates_passed"],
             "Perfect answer image-following control fails intended gates: "
             f"{report['gate_failures']}")
    negative_failures: dict[str, list[str]] = {}
    for mode in ("constant_answer", "image_ignoring", "shuffled"):
        rows = _answer_probe_rows_v2(cases, labels, control_map,
                                     controls, mode)
        score = score_answer_controls_v2(
            cases, labels, rows, control_map=control_map, controls=controls,
            seed=seed + 1, bootstrap=bootstrap)
        negative = diagnostic_gate_report_v2(
            label=f"negative_{mode}", answer=score, caption=None,
            evaluation=evaluation)
        negative_failures[mode] = negative["gate_failures"]
        _require(negative["gate_failures"],
                 f"Answer negative control {mode} passed every gate")
    malformed = _answer_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    malformed[0] = dict(malformed[0])
    malformed[0].pop("control_assignment_sha256")
    try:
        score_answer_controls_v2(
            cases, labels, malformed, control_map=control_map,
            controls=controls, seed=seed + 2, bootstrap=bootstrap)
    except ValueError as error:
        malformed_error = str(error)
    else:
        raise ValueError("Malformed answer generation was accepted")
    tampered = _answer_probe_rows_v2(
        cases, labels, control_map, controls, "perfect_image_following")
    tampered[0] = dict(tampered[0])
    tampered[0]["control_assignment_sha256"] = "0" * 64
    try:
        score_answer_controls_v2(
            cases, labels, tampered, control_map=control_map,
            controls=controls, seed=seed + 3, bootstrap=bootstrap)
    except ValueError as error:
        tampered_error = str(error)
    else:
        raise ValueError("Target-tampered answer generation was accepted")
    return {
        "format": SCORER_CONTRACT_FORMAT,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "controls": controls,
        "positive": {"mode": "perfect_image_following",
                     "all_declared_gates_passed": True,
                     "gate_failures": []},
        "negative": {
            "gate_failures": {key: sorted(value)
                              for key, value in negative_failures.items()},
            "malformed_rejected": malformed_error,
            "assignment_tampering_rejected": tampered_error,
        },
        "all_contracts_passed": True,
    }


# ---------------------------------------------------------------------------
# V2 access order and reload comparison receipts
# ---------------------------------------------------------------------------

_ACCESS_ORDER = ("train", "train_fit", "dev", "audit")


class AccessOrder:
    """Monotonic train -> train-fit -> dev -> audit data-access gate."""

    def __init__(self) -> None:
        self.phase = _ACCESS_ORDER[0]
        self.events: list[dict[str, Any]] = []

    def access(self, role: str) -> None:
        _require(role in _ACCESS_ORDER,
                 f"Unknown data access role: {role}")
        _require(_ACCESS_ORDER.index(role)
                 <= _ACCESS_ORDER.index(self.phase),
                 f"Data access {role} precedes the current phase {self.phase}")
        self.events.append({"role": role, "phase": self.phase})

    def advance(self, role: str) -> None:
        _require(role in _ACCESS_ORDER, f"Unknown data phase: {role}")
        current = _ACCESS_ORDER.index(self.phase)
        _require(current + 1 < len(_ACCESS_ORDER)
                 and _ACCESS_ORDER[current + 1] == role,
                 f"Cannot advance data phase {self.phase} to {role}")
        self.phase = role
        self.events.append({"advanced_to": role})

    def snapshot(self) -> dict[str, Any]:
        return {"phase": self.phase, "order": list(_ACCESS_ORDER),
                "events": [dict(event) for event in self.events]}


def preaudit_allows_heldout(train_fit_report: Mapping[str, Any] | None,
                            dev_report: Mapping[str, Any],
                            protocol: Mapping[str, Any]) -> bool:
    preaudit = protocol.get("preaudit", {})
    if preaudit.get("require_train_fit_gates"):
        if train_fit_report is None \
                or not train_fit_report.get("all_declared_gates_passed"):
            return False
    if preaudit.get("require_dev_gates"):
        if not dev_report.get("all_declared_gates_passed"):
            return False
    return True


def train_fit_gate_report(*, answer: Mapping[str, Any] | None,
                          caption: Mapping[str, Any] | None,
                          evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Train-fit acceptance must be observable before development/audit."""
    gates = dict(evaluation.get("gates", evaluation))
    gates.setdefault("min_train_fit_answer_accuracy",
                     gates.get("min_correct_accuracy", 0.0))
    gates.setdefault("min_train_fit_caption_following_accuracy",
                     gates.get("min_caption_following_accuracy", 0.0))
    failures: list[str] = []
    if answer is not None:
        correct = answer["controls"].get("correct")
        if not correct or not correct.get("available") \
                or correct["accuracy"] < gates[
                    "min_train_fit_answer_accuracy"]:
            failures.append("min_train_fit_answer_accuracy")
        counterfactual = answer.get("counterfactual_answerable")
        if evaluation.get("train_probe_controls") == ["correct"]:
            counterfactual = None
        if counterfactual is not None and counterfactual[
                "following_accuracy"] < gates[
                    "min_counterfactual_following_accuracy"]:
            failures.append("min_counterfactual_following_accuracy")
    if caption is not None:
        correct = caption["controls"]["correct"]
        accuracy = correct.get("following_accuracy")
        if accuracy is None or accuracy < gates[
                "min_train_fit_caption_following_accuracy"]:
            failures.append("min_train_fit_caption_following_accuracy")
    return {
        "label": "train_fit",
        "contract_version": CAPTION_CONTRACT_VERSION,
        "diagnostic_only": True,
        "promotion": False,
        "gate_failures": failures,
        "all_declared_gates_passed": not failures,
        "scope": "training-set fit evidence only; not generalization",
    }


def reload_comparison_names_v2() -> tuple[str, ...]:
    return (
        "train_fit_answer_generations.jsonl",
        "train_fit_caption_generations.jsonl",
        "dev_answer_generations.jsonl",
        "dev_caption_generations.jsonl",
        "heldout_answer_generations.jsonl",
        "heldout_caption_generations.jsonl",
        "retention_after.jsonl",
        "scores/train_fit_answer.json",
        "scores/train_fit_caption.json",
        "scores/dev_answer.json",
        "scores/dev_caption.json",
        "scores/heldout_answer.json",
        "scores/heldout_caption.json",
        "scores/overfit_summary.json",
        "scores/retention.json",
        "gates/train_fit.json",
        "gates/dev.json",
        "gates/heldout.json",
    )


def _scorer_fixture_v2():
    """Synthetic, sealed-shape fixture that reads no prepared data."""
    def sha(tag: str) -> str:
        return sha256_bytes(tag.encode())

    caption_cases = []
    caption_labels = []
    caption_controls = {}
    worlds = ("w1", "w2", "w3", "w4", "w5")
    originals = {
        "fx-a": ([["red", "circle"]], [["blue", "square"]]),
        "fx-b": ([["green", "triangle"]], [["yellow", "rectangle"]]),
        "fx-c": ([["purple", "square"]], [["orange", "triangle"]]),
        "fx-d": ([["blue", "rectangle"]], [["red", "triangle"]]),
    }
    for index, case_id in enumerate(sorted(originals)):
        owned, changed = originals[case_id]
        donor_id = sorted(originals)[(index + 1) % len(originals)]
        other = sorted(originals)[(index + 2) % len(originals)]
        donor_objects = originals[other][0]
        caption_cases.append({
            "case_id": case_id,
            "world_id": worlds[index],
            "task": "caption",
            "image_sha256": sha(f"img:{case_id}"),
            "scene_sha256": sha(f"scene:{case_id}"),
            "counterfactual_case_id": f"{case_id}-cf",
            "counterfactual_image_sha256": sha(f"cf:{case_id}"),
        })
        caption_labels.append({
            "case_id": case_id,
            "world_id": worlds[index],
            "answer": _caption_text(owned),
            "visible_objects": owned,
            "counterfactual_answer": _caption_text(changed),
            "counterfactual_visible_objects": changed,
        })
        entry = {
            "case_id": case_id,
            "split": "fixture_caption",
            "counterfactual": {
                "case_id": f"{case_id}-cf",
                "image": f"fixture/{case_id}-cf.png",
                "image_sha256": sha(f"cf:{case_id}"),
                "scene_sha256": sha(f"cfscene:{case_id}"),
                "world_id": worlds[index],
                "task": "caption",
            },
            "shuffled": {
                "case_id": other,
                "image": f"fixture/{other}.png",
                "image_sha256": sha(f"img:{other}"),
                "scene_sha256": sha(f"scene:{other}"),
                "world_id": worlds[sorted(originals).index(other)],
                "task": "caption",
                "caption": _caption_text(donor_objects),
                "visible_objects": donor_objects,
            },
            "counterfactual_same_image": False,
        }
        entry["assignment_sha256"] = control_assignment_sha256(entry)
        caption_controls[case_id] = entry
    answer_cases = []
    answer_labels = []
    answer_controls = {}
    answer_values = {"fx-a": ("red", "blue"), "fx-b": ("green", "yellow"),
                     "fx-c": ("purple", "orange"), "fx-d": ("blue", "red"),
                     "fx-e": ("unknown", "red")}
    for index, case_id in enumerate(sorted(answer_values)):
        original, changed = answer_values[case_id]
        other = sorted(answer_values)[(index + 1) % len(answer_values)]
        answer_cases.append({
            "case_id": case_id,
            "world_id": worlds[index],
            "task": "color",
            "label": original,
            "image_sha256": sha(f"img:{case_id}"),
            "scene_sha256": sha(f"scene:{case_id}"),
            "counterfactual_case_id": f"{case_id}-cf",
            "counterfactual_image_sha256": sha(f"cf:{case_id}"),
        })
        answer_labels.append({
            "case_id": case_id,
            "world_id": worlds[index],
            "answer": original,
            "counterfactual_answer": changed,
            "answer_changes_under_counterfactual": True,
        })
        entry = {
            "case_id": case_id,
            "split": "fixture_answer",
            "wrong": {
                "case_id": other,
                "image": f"fixture/{other}.png",
                "image_sha256": sha(f"img:{other}"),
                "scene_sha256": sha(f"scene:{other}"),
                "world_id": worlds[sorted(answer_values).index(other)],
                "task": "color",
                "sealed_answer": answer_values[other][0],
            },
            "shuffled": {
                "case_id": other,
                "image": f"fixture/{other}.png",
                "image_sha256": sha(f"img:{other}"),
                "scene_sha256": sha(f"scene:{other}"),
                "world_id": worlds[sorted(answer_values).index(other)],
                "task": "color",
                "sealed_answer": answer_values[other][0],
            },
            "counterfactual": {
                "case_id": f"{case_id}-cf",
                "image": f"fixture/{case_id}-cf.png",
                "image_sha256": sha(f"cf:{case_id}"),
                "scene_sha256": sha(f"cfscene:{case_id}"),
                "world_id": worlds[index],
                "task": "color",
                "sealed_answer": changed,
            },
            "counterfactual_same_image": False,
        }
        entry["assignment_sha256"] = control_assignment_sha256(entry)
        answer_controls[case_id] = entry
    return (caption_cases, caption_labels, caption_controls,
            answer_cases, answer_labels, answer_controls)


def _caption_text(objects: Sequence[Sequence[str]]) -> str:
    return "a " + " and a ".join(f"{color} {shape}"
                                for color, shape in objects)


def scorer_contract_preflight_v2(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Run positive/negative control contracts before touching any data."""
    (caption_cases, caption_labels, caption_controls,
     answer_cases, answer_labels, answer_controls) = _scorer_fixture_v2()
    caption = verify_caption_control_contracts_v2(
        caption_cases, caption_labels, caption_controls,
        controls=list(evaluation["caption_controls"]), seed=20261028,
        bootstrap=evaluation["bootstrap"], evaluation=evaluation)
    answer = verify_answer_control_contracts_v2(
        answer_cases, answer_labels, answer_controls,
        controls=list(evaluation["heldout_answer_controls"]),
        seed=20261028, bootstrap=evaluation["bootstrap"],
        evaluation=evaluation)
    return {
        "format": SCORER_CONTRACT_FORMAT,
        "contract_version": CAPTION_CONTRACT_VERSION,
        "caption": caption,
        "answer": answer,
        "positive_control": caption["positive"],
        "answer_positive_control": answer["positive"],
        "all_contracts_passed": True,
        "reads_prepared_data": False,
    }
