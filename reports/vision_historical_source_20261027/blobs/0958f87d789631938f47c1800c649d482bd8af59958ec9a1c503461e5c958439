"""Bounded, preregistered V0/V1 qualification for the MiniCPM5 visual prefix.

This module contains protocol/admission validation, deterministic data and
control selection, artifact checks shared by the native runner and the
independent CPU scorer, and pure answer/quality scoring.  It never imports the
native runner, never loads a model, and never chooses an answer for the model.

V0 answers an interface question only: text bypass parity, frozen base/encoder,
response-only supervision, projector-only gradient ownership, pinned
correct/wrong/no-image payloads, bounded generation, strict extension export and
fresh-process restart parity.  V1 additionally requires an accepted independent
V0 score report and named data-admission hashes before any projector training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_json, fingerprint


PROTOCOL_FORMAT = "replica.vision_qualification_protocol.v1"
SCORE_FORMAT = "replica.vision_qualification_score.v1"
ADMISSION_FORMAT = "replica.vision_training_admission.v2"
TRAINING_ROWS_FORMAT = "replica.vision_training_rows.v1"
TEXT_CONTEXTS_FORMAT = "replica.vision_text_contexts.v1"
RETENTION_PROMPTS_FORMAT = "replica.vision_retention_prompts.v1"
RETENTION_LABELS_FORMAT = "replica.vision_retention_labels.v1"
EXTENSION_FORMAT = "replica.minicpm5.extensions.v4"
VISION_TRAINING_SCHEDULE = "seeded_epoch_shuffle.v1"
VISUAL_QUESTION_TEMPLATE = "Question: {question}\nReply briefly with the answer alone."
MANIFEST_FORMAT = "replica.vision_manifest.v1"

STAGES = ("v0", "v1")
SPLITS = ("train", "dev", "calibration", "promotion", "audit")
CONTROLS = ("correct", "wrong", "shuffled", "noimage", "captiononly", "ocrtextonly")
VISUAL_CONTROLS = ("correct", "wrong", "shuffled", "noimage")
TEXT_CONTROLS = ("captiononly", "ocrtextonly")
CORE_CONTROLS = ("correct", "wrong", "shuffled", "noimage")
TEXT_CONTEXT_KINDS = ("caption_only", "ocr_text_only")
TEXT_CONTROL_KIND = {"captiononly": "caption_only", "ocrtextonly": "ocr_text_only"}
MODEL_ROW_FIELDS = frozenset({"asset_id", "case_id", "image", "image_sha256", "question"})
TRAINING_ROW_FIELDS = frozenset({"case_id", "prompt", "response", "image", "image_sha256",
                                "family", "group_id", "split"})
TEXT_CONTEXT_FIELDS = frozenset({"case_id", "kind", "text", "text_sha256", "source_id",
                                "information_bound"})
RETENTION_PROMPT_FIELDS = frozenset({"case_id", "prompt", "family", "group_id", "split"})
RETENTION_LABEL_FIELDS = frozenset({"case_id", "answer"})
GATE_NAMES = (
    "min_correct_accuracy",
    "min_correct_margin_wrong",
    "min_correct_margin_shuffled",
    "min_correct_margin_noimage",
    "min_correct_margin_captiononly",
    "min_correct_margin_ocrtextonly",
    "min_unique_families",
    "max_unknown_rate",
    "min_text_retention_accuracy",
    "max_text_retention_drop",
)
UNKNOWN_ANSWERS = frozenset({"unknown", "i don't know", "i do not know", "idk",
                             "cannot determine", "can't determine", "not sure",
                             "insufficient information", "no answer"})
ALLOWED_EXTENSION_PREFIXES = ("vision_projector.",)
EXECUTABLE_IMPLEMENTATION_FILES = (
    "src/replica_lab/vision_qualification.py",
    "scripts/run_vision_qualification.py",
    "scripts/score_vision_qualification.py",
)


class QualificationError(ValueError):
    """Raised when a protocol, admission, artifact or score contract differs."""


# ---------------------------------------------------------------------------
# Small deterministic helpers


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _sha(value: Any, name: str) -> str:
    require(is_sha256(value), f"{name} must be a lowercase SHA256")
    return value


def _revision(value: Any, name: str) -> str:
    require(isinstance(value, str) and len(value) == 40
            and all(char in "0123456789abcdef" for char in value),
            f"{name} must be a 40-character lowercase revision digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0,
            f"{name} must be a nonnegative integer")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)), f"{name} must be a finite number")
    number = float(value)
    if minimum is not None:
        require(number >= minimum, f"{name} must be >= {minimum}")
    if maximum is not None:
        require(number <= maximum, f"{name} must be <= {maximum}")
    return number


def _text(value: Any, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], name: str) -> None:
    expected = set(expected)
    require(isinstance(value, Mapping) and set(value) == expected,
            f"{name} fields differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}")


def _subset_keys(value: Mapping[str, Any], required: Iterable[str], name: str) -> None:
    missing = set(required) - set(value)
    require(not missing, f"{name} missing fields: {sorted(missing)}")


def resolve_project_path(protocol: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(protocol.get("root", "."))
    return (root / path).resolve()


def resolve_relative(root: str | Path, value: str | Path) -> Path:
    """Resolve a manifest-relative path while refusing absolute/escape paths."""
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe relative path: {value}")
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    require(candidate == root or root in candidate.parents,
            f"Path escapes its allowed root: {value}")
    return candidate


def load_pinned_relative_json(root: str | Path, pin: Mapping[str, Any],
                              name: str) -> tuple[Any, str]:
    _subset_keys(pin, ("path", "sha256"), name)
    path = resolve_relative(root, pin["path"])
    require(path.is_file(), f"{name} does not exist: {path}")
    actual = sha256_file(path)
    require(actual == _sha(pin["sha256"], f"{name}.sha256"),
            f"{name} bytes changed: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8")), actual
    except (ValueError, UnicodeDecodeError) as error:
        raise QualificationError(f"{name} is not valid UTF-8 JSON") from error


def load_pinned_relative_jsonl(root: str | Path, pin: Mapping[str, Any],
                               name: str) -> tuple[list[dict[str, Any]], str]:
    _subset_keys(pin, ("path", "sha256"), name)
    path = resolve_relative(root, pin["path"])
    require(path.is_file(), f"{name} does not exist: {path}")
    actual = sha256_file(path)
    require(actual == _sha(pin["sha256"], f"{name}.sha256"),
            f"{name} bytes changed: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{name}:{line_number} is not an object")
            rows.append(value)
    except (ValueError, UnicodeDecodeError) as error:
        raise QualificationError(f"{name} is not valid UTF-8 JSONL") from error
    require(bool(rows), f"{name} is empty")
    return rows, actual


def load_pinned_json(protocol: Mapping[str, Any], pin: Mapping[str, Any],
                     name: str) -> tuple[Any, str]:
    _subset_keys(pin, ("path", "sha256"), name)
    path = resolve_project_path(protocol, pin["path"])
    require(path.is_file(), f"{name} does not exist: {path}")
    actual = sha256_file(path)
    require(actual == _sha(pin["sha256"], f"{name}.sha256"),
            f"{name} bytes changed: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8")), actual
    except (ValueError, UnicodeDecodeError) as error:
        raise QualificationError(f"{name} is not valid UTF-8 JSON") from error


def load_pinned_jsonl(protocol: Mapping[str, Any], pin: Mapping[str, Any],
                      name: str) -> tuple[list[dict[str, Any]], str]:
    _subset_keys(pin, ("path", "sha256"), name)
    path = resolve_project_path(protocol, pin["path"])
    require(path.is_file(), f"{name} does not exist: {path}")
    actual = sha256_file(path)
    require(actual == _sha(pin["sha256"], f"{name}.sha256"),
            f"{name} bytes changed: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{name}:{line_number} is not an object")
            rows.append(value)
    except (ValueError, UnicodeDecodeError) as error:
        raise QualificationError(f"{name} is not valid UTF-8 JSONL") from error
    require(bool(rows), f"{name} is empty")
    return rows, actual


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def ensure_fresh_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        require(path.is_dir() and not any(path.iterdir()),
                f"Use a fresh output directory: {path}")
    else:
        path.mkdir(parents=True)
    return path


def implementation_hashes(root: str | Path,
                          files: Iterable[str] = EXECUTABLE_IMPLEMENTATION_FILES
                          ) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative in files:
        path = root / relative
        require(path.is_file(), f"Implementation file missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def verify_implementation_pins(protocol: Mapping[str, Any], root: str | Path) -> dict[str, str]:
    pins = protocol.get("implementation_sha256")
    require(isinstance(pins, dict) and bool(pins), "implementation_sha256 pins are required")
    actual = {}
    for relative, expected in pins.items():
        path = resolve_relative(root, relative)
        require(path.is_file(), f"Pinned implementation file missing: {relative}")
        actual[relative] = sha256_file(path)
        require(actual[relative] == _sha(expected, f"implementation_sha256[{relative}]"),
                f"Implementation bytes changed: {relative}")
    for relative in EXECUTABLE_IMPLEMENTATION_FILES:
        require(relative in pins, f"Protocol must pin implementation: {relative}")
    return actual


# ---------------------------------------------------------------------------
# Protocol validation


def _validate_vision_spec(spec: Mapping[str, Any], name: str = "vision") -> None:
    _exact_keys(spec, ("encoder_repo", "encoder_revision", "encoder_source_digest",
                       "processor_digest", "encoder_width", "image_size", "patch_size",
                       "max_visual_tokens", "projector_hidden", "feature_version",
                       "compute_dtype"), name)
    require("/" in _text(spec["encoder_repo"], f"{name}.encoder_repo"),
            f"{name}.encoder_repo must be owner/name")
    _revision(spec["encoder_revision"], f"{name}.encoder_revision")
    _sha(spec["encoder_source_digest"], f"{name}.encoder_source_digest")
    _sha(spec["processor_digest"], f"{name}.processor_digest")
    for key in ("encoder_width", "image_size", "patch_size", "max_visual_tokens",
                "projector_hidden"):
        _positive_int(spec[key], f"{name}.{key}")
    require(spec["image_size"] % spec["patch_size"] == 0
            and spec["max_visual_tokens"] <= 1024,
            f"{name} spatial layout is unbounded")
    require(spec["feature_version"] == "siglip_patch_last_hidden.v1"
            and spec["compute_dtype"] == "float32",
            f"{name} feature version/dtype differs from the qualified interface")


def _validate_limits(limits: Mapping[str, Any]) -> None:
    for key in ("max_prompt_tokens", "max_new_tokens", "max_rows", "max_updates",
                "max_repeats", "max_eval_cases", "max_export_bytes",
                "min_free_disk_bytes"):
        _positive_int(limits.get(key), f"limits.{key}")
    _positive_int(limits.get("max_seconds"), "limits.max_seconds")
    require(limits["max_updates"] >= 0, "limits.max_updates is invalid")


def _validate_pin(pin: Mapping[str, Any], name: str) -> None:
    _subset_keys(pin, ("path", "sha256"), name)
    _text(pin["path"], f"{name}.path")
    _sha(pin["sha256"], f"{name}.sha256")


def _validate_data_section(data: Mapping[str, Any], stage: str) -> None:
    _subset_keys(data, ("root", "stage", "manifest", "model_rows",
                        "verifier_cases", "verifier_scenes"), "data")
    require(data["stage"] == stage, "data.stage differs from protocol stage")
    _text(data["root"], "data.root")
    for key in ("manifest", "model_rows", "verifier_cases", "verifier_scenes"):
        _validate_pin(data[key], f"data.{key}")


def _validate_v0(v0: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    _subset_keys(v0, ("probe", "parity", "backward", "generation",
                      "distinct_logit_min", "require_restart", "restart",
                      "update"), "v0")
    probe = v0["probe"]
    _subset_keys(probe, ("case_id", "wrong_image_case_id", "split"), "v0.probe")
    _text(probe["case_id"], "v0.probe.case_id")
    _text(probe["wrong_image_case_id"], "v0.probe.wrong_image_case_id")
    require(probe["split"] in SPLITS, "v0.probe.split is invalid")
    require(probe["case_id"] != probe["wrong_image_case_id"],
            "V0 wrong-image source must differ from the probe case")
    parity = v0["parity"]
    _subset_keys(parity, ("prompt", "atol"), "v0.parity")
    _text(parity["prompt"], "v0.parity.prompt")
    _number(parity["atol"], "v0.parity.atol", minimum=0.0)
    backward = v0["backward"]
    _subset_keys(backward, ("prompt", "target_text", "max_response_tokens"),
                 "v0.backward")
    _text(backward["prompt"], "v0.backward.prompt")
    _text(backward["target_text"], "v0.backward.target_text")
    _positive_int(backward["max_response_tokens"], "v0.backward.max_response_tokens")
    generation = v0["generation"]
    _subset_keys(generation, ("max_new_tokens", "seed"), "v0.generation")
    _positive_int(generation["max_new_tokens"], "v0.generation.max_new_tokens")
    require(generation["max_new_tokens"] <= limits["max_new_tokens"],
            "v0.generation.max_new_tokens exceeds declared limit")
    _nonnegative_int(generation["seed"], "v0.generation.seed")
    _number(v0["distinct_logit_min"], "v0.distinct_logit_min", minimum=0.0)
    require(type(v0["require_restart"]) is bool, "v0.require_restart must be boolean")
    restart = v0["restart"]
    _subset_keys(restart, ("atol", "rtol"), "v0.restart")
    _number(restart["atol"], "v0.restart.atol", minimum=0.0)
    _number(restart["rtol"], "v0.restart.rtol", minimum=0.0)
    update = v0["update"]
    _subset_keys(update, ("enabled", "optimizer", "lr"), "v0.update")
    require(type(update["enabled"]) is bool, "v0.update.enabled must be boolean")
    require(update["optimizer"] == "sgd", "V0 update must declare the audited SGD arm")
    _number(update["lr"], "v0.update.lr", minimum=0.0)
    require(0.0 < float(update["lr"]) <= 1.0, "v0.update.lr must be in (0, 1]")


def _validate_v1(v1: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    _subset_keys(v1, ("run_id", "question_template", "prerequisites", "initial_checkpoint",
                      "train", "eval", "controls", "quality", "retention"), "v1")
    _text(v1["run_id"], "v1.run_id")
    require(v1["question_template"] == VISUAL_QUESTION_TEMPLATE,
            "V1 must use the pinned short-answer question template")
    prerequisites = v1["prerequisites"]
    _subset_keys(prerequisites, ("v0_protocol_sha256", "v0_score_report",
                                 "require_accepted", "data_admission"),
                 "v1.prerequisites")
    _sha(prerequisites["v0_protocol_sha256"], "v1.prerequisites.v0_protocol_sha256")
    _validate_pin(prerequisites["v0_score_report"], "v1.prerequisites.v0_score_report")
    require(prerequisites["require_accepted"] is True,
            "V1 must require an accepted V0 score report; bypass is not supported")
    _validate_pin(prerequisites["data_admission"],
                  "v1.prerequisites.data_admission")
    checkpoint = v1.get("initial_checkpoint")
    if checkpoint is not None:
        _subset_keys(checkpoint, ("path", "manifest_sha256", "weights_sha256"),
                     "v1.initial_checkpoint")
        _text(checkpoint["path"], "v1.initial_checkpoint.path")
        _sha(checkpoint["manifest_sha256"], "v1.initial_checkpoint.manifest_sha256")
        _sha(checkpoint["weights_sha256"], "v1.initial_checkpoint.weights_sha256")
    train = v1["train"]
    _subset_keys(train, ("seed", "updates", "repeats", "grad_accum", "lr_new",
                         "momentum", "ns_steps", "block_size", "weight_decay",
                         "max_seconds", "optimizer", "schedule_version",
                         "repeat_policy", "max_grad_norm"), "v1.train")
    _nonnegative_int(train["seed"], "v1.train.seed")
    _positive_int(train["updates"], "v1.train.updates")
    _positive_int(train["repeats"], "v1.train.repeats")
    _positive_int(train["grad_accum"], "v1.train.grad_accum")
    require(train["schedule_version"] == VISION_TRAINING_SCHEDULE,
            "Unknown visual training schedule")
    require(train["repeat_policy"] in ("once", "declared_epochs")
            and (train["repeat_policy"] != "once" or train["repeats"] == 1),
            "Visual training repeat policy differs from declared repeats")
    _number(train["max_grad_norm"], "v1.train.max_grad_norm", minimum=0.0)
    require(0 < train["max_grad_norm"] <= 1000, "V1 gradient norm bound is invalid")
    require(train["optimizer"] == "muon8bit",
            "V1 projector optimizer must declare muon8bit plus FP32 AdamW routing")
    _number(train["lr_new"], "v1.train.lr_new", minimum=0.0)
    require(0.0 < float(train["lr_new"]) <= 1.0, "v1.train.lr_new must be in (0, 1]")
    _number(train["weight_decay"], "v1.train.weight_decay", minimum=0.0)
    _number(train["momentum"], "v1.train.momentum", minimum=0.0, maximum=0.999999)
    _positive_int(train["ns_steps"], "v1.train.ns_steps")
    require(train["ns_steps"] < 100, "v1.train.ns_steps must be < 100")
    _positive_int(train["block_size"], "v1.train.block_size")
    _positive_int(train["max_seconds"], "v1.train.max_seconds")
    require(train["updates"] <= limits["max_updates"],
            "v1.train.updates exceeds limits.max_updates")
    require(train["repeats"] <= limits["max_repeats"],
            "v1.train.repeats exceeds limits.max_repeats")
    require(train["max_seconds"] <= limits["max_seconds"],
            "v1.train.max_seconds exceeds limits.max_seconds")
    evaluation = v1["eval"]
    _subset_keys(evaluation, ("splits", "max_cases", "seed", "max_new_tokens",
                              "required_controls"), "v1.eval")
    splits = evaluation["splits"]
    require(isinstance(splits, list) and bool(splits)
            and len(set(splits)) == len(splits)
            and set(splits) <= set(SPLITS), "v1.eval.splits is invalid")
    require("train" not in splits, "V1 evaluation must use an independent split")
    _positive_int(evaluation["max_cases"], "v1.eval.max_cases")
    require(evaluation["max_cases"] >= 2,
            "V1 requires at least two evaluation cases for wrong/shuffled controls")
    _nonnegative_int(evaluation["seed"], "v1.eval.seed")
    _positive_int(evaluation["max_new_tokens"], "v1.eval.max_new_tokens")
    require(evaluation["max_new_tokens"] <= limits["max_new_tokens"],
            "v1.eval.max_new_tokens exceeds limits.max_new_tokens")
    require(evaluation["max_cases"] <= limits["max_eval_cases"],
            "v1.eval.max_cases exceeds limits.max_eval_cases")
    required = evaluation["required_controls"]
    require(isinstance(required, list) and len(set(required)) == len(required)
            and set(required) <= set(CONTROLS), "v1.eval.required_controls is invalid")
    require(set(CONTROLS) <= set(required),
            "V1 requires correct/wrong/shuffled/noimage/captiononly/ocrtextonly controls")
    controls = v1["controls"]
    _subset_keys(controls, ("wrong_source", "shuffled_source",
                            "text_context_template"), "v1.controls")
    require(controls["wrong_source"] == "different_family",
            "Only a deterministic different-family wrong-image control is supported")
    require(controls["shuffled_source"] == "deterministic_derangement",
            "Only a deterministic derangement shuffled-image control is supported")
    template = _text(controls["text_context_template"],
                     "v1.controls.text_context_template")
    require("{context}" in template and "{question}" in template,
            "Text-context template must use {context} and {question}")
    quality = v1["quality"]
    _subset_keys(quality, ("gates", "retention_baseline"), "v1.quality")
    require(quality["retention_baseline"] == "initial_checkpoint_same_prompts.v1",
            "V1 text baseline must be measured before updates on the pinned prompts")
    if "retention_policy" in quality:
        require(quality["retention_policy"] == "component_nonregression.v1",
                "Unknown V1 retention policy")
    gates = quality["gates"]
    require(isinstance(gates, dict) and set(gates) == set(GATE_NAMES),
            f"v1.quality.gates must declare exactly {sorted(GATE_NAMES)}")
    for name, value in gates.items():
        if name == "min_unique_families":
            _positive_int(value, f"v1.quality.gates.{name}")
        elif name == "max_unknown_rate":
            _number(value, f"v1.quality.gates.{name}", minimum=0.0, maximum=1.0)
        elif name == "min_text_retention_accuracy":
            _number(value, f"v1.quality.gates.{name}", minimum=0.0, maximum=1.0)
        elif name == "max_text_retention_drop":
            _number(value, f"v1.quality.gates.{name}", minimum=0.0, maximum=1.0)
        else:
            _number(value, f"v1.quality.gates.{name}", minimum=-1.0, maximum=1.0)
    retention = v1["retention"]
    _subset_keys(retention, ("prompts", "labels", "splits", "max_cases"),
                 "v1.retention")
    _validate_pin(retention["prompts"], "v1.retention.prompts")
    _validate_pin(retention["labels"], "v1.retention.labels")
    retention_splits = retention["splits"]
    require(isinstance(retention_splits, list) and bool(retention_splits)
            and set(retention_splits) <= set(SPLITS),
            "v1.retention.splits is invalid")
    _positive_int(retention["max_cases"], "v1.retention.max_cases")
    require(retention["max_cases"] <= limits["max_eval_cases"],
            "v1.retention.max_cases exceeds limits.max_eval_cases")
    require(quality["gates"]["min_unique_families"] <= 4096,
            "min_unique_families is unbounded")


def _validate_scope(scope: Mapping[str, Any]) -> None:
    _subset_keys(scope, ("subject_id", "branch_id", "purpose", "as_of", "mode"),
                 "protocol.scope")
    for key in ("subject_id", "branch_id", "purpose"):
        _text(scope[key], f"protocol.scope.{key}")
    _nonnegative_int(scope["as_of"], "protocol.scope.as_of")
    require(scope["mode"] in ("assistant_for_source", "parallel_replica",
                              "source_current", "source_snapshot"),
            "protocol.scope.mode is invalid")
    if scope["mode"] == "source_snapshot":
        _nonnegative_int(scope.get("snapshot_at"), "protocol.scope.snapshot_at")
    else:
        require(scope.get("snapshot_at") is None,
                "snapshot_at is only valid for source_snapshot")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    _subset_keys(protocol, ("format", "root", "vision", "vision_digest",
                            "vision_source", "model", "data", "limits",
                            "implementation_sha256"), "protocol")
    require(protocol["format"] == PROTOCOL_FORMAT, "Unsupported qualification protocol")
    _text(protocol["root"], "protocol.root")
    if protocol.get("template") is True or protocol.get("stage") not in STAGES:
        raise QualificationError(
            "Prospective template is not executable; supply a hash-pinned stage protocol")
    _validate_vision_spec(protocol["vision"])
    require(protocol["vision_digest"] == fingerprint(protocol["vision"]),
            "vision_digest differs from the declared VisionSpec fields")
    _validate_limits(protocol["limits"])
    model = protocol["model"]
    _subset_keys(model, ("revision", "lock_sha256"), "protocol.model")
    _revision(model["revision"], "protocol.model.revision")
    _sha(model["lock_sha256"], "protocol.model.lock_sha256")
    source = protocol["vision_source"]
    _subset_keys(source, ("path", "manifest", "source_id", "scope"),
                 "protocol.vision_source")
    _text(source["path"], "protocol.vision_source.path")
    _text(source["source_id"], "protocol.vision_source.source_id")
    _validate_pin(source["manifest"], "protocol.vision_source.manifest")
    _validate_scope(source["scope"])
    implementation = protocol["implementation_sha256"]
    require(isinstance(implementation, dict), "implementation_sha256 must be an object")
    for relative, value in implementation.items():
        _text(relative, "implementation_sha256 path")
        _sha(value, f"implementation_sha256[{relative}]")
    stage = protocol["stage"]
    _validate_data_section(protocol["data"], stage)
    for relative in EXECUTABLE_IMPLEMENTATION_FILES:
        require(relative in implementation,
                f"Executable protocol must pin implementation: {relative}")
    if stage == "v0":
        _validate_v0(protocol["v0"], protocol["limits"])
    else:
        _validate_v1(protocol["v1"], protocol["limits"])




# ---------------------------------------------------------------------------
# Verified owned visual data


@dataclass(frozen=True)
class VisualDataset:
    stage: str
    root: Path
    manifest: Mapping[str, Any]
    model_rows: tuple[Mapping[str, Any], ...]
    verifier_cases: tuple[Mapping[str, Any], ...]
    verifier_scenes: tuple[Mapping[str, Any], ...]
    assets: Mapping[str, Mapping[str, Any]]
    cases: Mapping[str, Mapping[str, Any]]
    scenes: Mapping[str, Mapping[str, Any]]
    pins: Mapping[str, str]

    def case(self, case_id: str) -> Mapping[str, Any]:
        require(case_id in self.cases, f"Unknown case_id: {case_id}")
        return self.cases[case_id]

    def row(self, case_id: str) -> Mapping[str, Any]:
        for row in self.model_rows:
            if row["case_id"] == case_id:
                return row
        raise QualificationError(f"Unknown model row: {case_id}")

    def asset_for_case(self, case_id: str) -> Mapping[str, Any]:
        asset_id = self.case(case_id)["asset_id"]
        return self.assets[asset_id]

    def image_path(self, case_or_asset_id: str) -> Path:
        asset = (self.assets.get(case_or_asset_id)
                 or self.asset_for_case(case_or_asset_id))
        return resolve_relative(self.root, asset["path"])

    def rows_for_split(self, split: str) -> list[Mapping[str, Any]]:
        return [row for row in self.model_rows
                if self.cases[row["case_id"]]["split"] == split]

    def label(self, case_id: str) -> str:
        case = self.case(case_id)
        scene = self.scenes[case["asset_id"]]["scene"]
        return reconstruct_label(case, scene)


def reconstruct_label(case: Mapping[str, Any], scene: Mapping[str, Any]) -> str:
    """Execute the verifier check against verifier-only scene truth."""
    check = case.get("check")
    require(isinstance(check, dict), "Verifier case lacks an executable check")
    objects = scene.get("objects")
    require(isinstance(objects, list) and bool(objects), "Scene lacks objects")
    by_id = {obj.get("id"): obj for obj in objects}
    kind = check.get("kind")
    if kind == "count":
        criteria = check.get("filter", {})
        require(isinstance(criteria, dict), "Count filter must be an object")
        return str(sum(all(obj.get(key) == value for key, value in criteria.items())
                       for obj in objects))
    if kind == "color_of":
        require(check.get("object_id") in by_id, "Unknown object in color check")
        return str(by_id[check["object_id"]]["color"])
    if kind == "shape_of":
        require(check.get("object_id") in by_id, "Unknown object in shape check")
        return str(by_id[check["object_id"]]["shape"])
    if kind == "absent":
        found = any(obj.get("color") == check.get("color")
                    and obj.get("shape") == check.get("shape") for obj in objects)
        return "yes" if found else "no"
    if kind == "spatial":
        first, second = by_id.get(check.get("first_id")), by_id.get(check.get("second_id"))
        require(first is not None and second is not None and first is not second,
                "Spatial check needs two distinct objects")
        left, top = first["center"]
        right, bottom = second["center"]
        actual = {"left_of": left < right, "right_of": left > right,
                  "above": top < bottom, "below": top > bottom}.get(check.get("relation"))
        require(actual is not None, "Unknown spatial relation")
        return "yes" if actual else "no"
    raise QualificationError(f"Unknown verifier check kind: {kind}")


def _png_dimensions(data: bytes, asset_id: str) -> tuple[int, int]:
    # The owned renderer always writes an 8-byte PNG signature, IHDR first.
    require(data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24
            and data[12:16] == b"IHDR", f"Asset is not a PNG with IHDR: {asset_id}")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def load_verified_dataset(protocol: Mapping[str, Any], *,
                          allowed_splits: Sequence[str] | None = None) -> VisualDataset:
    """Validate every declared byte and reconstruct every verifier label.

    Model rows are accepted only under the exact label-free schema.  Answers and
    scene checks never become model inputs; the scorer rebuilds them here.
    """
    stage = protocol["stage"]
    data = protocol["data"]
    root = resolve_project_path(protocol, data["root"])
    require(root.is_dir(), f"Visual data root missing: {root}")
    manifest, manifest_sha = load_pinned_relative_json(
        root, data["manifest"], "data.manifest")
    rows, rows_sha = load_pinned_relative_jsonl(
        root, data["model_rows"], "data.model_rows")
    cases, cases_sha = load_pinned_relative_jsonl(
        root, data["verifier_cases"], "data.verifier_cases")
    scenes, scenes_sha = load_pinned_relative_jsonl(
        root, data["verifier_scenes"], "data.verifier_scenes")
    require(manifest.get("format") == MANIFEST_FORMAT, "Unknown vision manifest")
    require(manifest.get("stage") == stage, "Manifest stage differs from protocol")
    declaration = manifest.get("training_admission")
    require(isinstance(declaration, dict) and declaration.get("admitted") is False
            and declaration.get("admitted_rows") == 0,
            "Preparation manifest must remain not training-admitted; use a named admission")
    case_decl = manifest.get("cases")
    require(isinstance(case_decl, dict)
            and case_decl.get("label_visibility") == "verifier_only"
            and case_decl.get("model_row_fields") == sorted(MODEL_ROW_FIELDS),
            "Manifest model-row label isolation differs")
    assets_list = manifest.get("assets")
    require(isinstance(assets_list, list) and bool(assets_list), "Manifest lacks assets")
    require(len(assets_list) <= protocol["limits"]["max_rows"],
            "Manifest asset count exceeds declared bound")

    assets: dict[str, Mapping[str, Any]] = {}
    scene_hashes: dict[str, str] = {}
    total_bytes = 0
    for asset in assets_list:
        asset_id = _text(asset.get("asset_id"), "asset.asset_id")
        require(asset_id not in assets, f"Duplicate asset_id: {asset_id}")
        require(asset.get("stage") == stage, f"Asset stage differs: {asset_id}")
        require(asset.get("split") in SPLITS, f"Asset split differs: {asset_id}")
        path = resolve_relative(root, asset.get("path"))
        require(path.is_file(), f"Asset file missing: {asset_id}")
        payload = path.read_bytes()
        require(len(payload) == asset.get("bytes"), f"Asset byte size differs: {asset_id}")
        require(sha256_bytes(payload) == asset.get("sha256"),
                f"Asset bytes changed: {asset_id}")
        width, height = _png_dimensions(payload, asset_id)
        require((width, height) == (asset.get("width"), asset.get("height")),
                f"Asset dimensions differ: {asset_id}")
        total_bytes += len(payload)
        assets[asset_id] = asset
    require(total_bytes == manifest.get("counts", {}).get("asset_bytes"),
            "Manifest asset byte total differs")

    scene_by_asset: dict[str, Mapping[str, Any]] = {}
    for record in scenes:
        asset_id = record.get("asset_id")
        require(asset_id in assets, f"Verifier scene references unknown asset: {asset_id}")
        require(asset_id not in scene_by_asset, f"Duplicate verifier scene: {asset_id}")
        truth = record.get("scene")
        require(isinstance(truth, dict), f"Scene truth missing: {asset_id}")
        require(fingerprint({key: truth[key] for key in ("width", "height", "layout", "objects")})
                == record.get("scene_sha256"), f"Scene truth changed: {asset_id}")
        require(record.get("scene_sha256") == assets[asset_id].get("scene_sha256"),
                f"Asset/scene hash differs: {asset_id}")
        scene_by_asset[asset_id] = record
    require(set(scene_by_asset) == set(assets), "Verifier scenes do not cover assets")

    case_by_id: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = _text(case.get("case_id"), "case.case_id")
        require(case_id not in case_by_id, f"Duplicate verifier case: {case_id}")
        require(case.get("label_visibility") == "verifier_only",
                f"Verifier case must stay verifier-only: {case_id}")
        asset_id = case.get("asset_id")
        require(asset_id in assets, f"Case references unknown asset: {case_id}")
        asset = assets[asset_id]
        require(case.get("split") == asset.get("split")
                and case.get("group_id") == asset.get("group_id")
                and case.get("scene_sha256") == asset.get("scene_sha256"),
                f"Case crosses family/split/scene: {case_id}")
        truth = scene_by_asset[asset_id]["scene"]
        expected = case.get("check", {}).get("expected")
        require(isinstance(expected, str) and expected, f"Case check lacks expected: {case_id}")
        computed = reconstruct_label(case, truth)
        require(computed == expected, f"Verifier check no longer executes: {case_id}")
        require(computed == case.get("answer"),
                f"Verifier answer differs from executable check: {case_id}")
        case_by_id[case_id] = case

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_inputs: dict[tuple[str, str], str] = {}
    for row in rows:
        _exact_keys(row, MODEL_ROW_FIELDS, "model row")
        case_id = row["case_id"]
        require(case_id not in rows_by_id, f"Duplicate model row: {case_id}")
        case = case_by_id.get(case_id)
        require(case is not None, f"Model row lacks verifier case: {case_id}")
        require(row["asset_id"] == case["asset_id"], f"Model row asset differs: {case_id}")
        require(row["question"] == case["question"], f"Model row question differs: {case_id}")
        asset = assets[case["asset_id"]]
        require(row["image"] == asset["path"] and row["image_sha256"] == asset["sha256"],
                f"Model row image pin differs: {case_id}")
        key = (row["image_sha256"], row["question"])
        require(key not in duplicate_inputs,
                f"Duplicate model inputs: {case_id} and {duplicate_inputs.get(key)}")
        duplicate_inputs[key] = case_id
        answer = case["answer"]
        if len(answer) > 1:
            require(answer.casefold() not in row["question"].casefold(),
                    f"Verifier answer leaks into the question: {case_id}")
        if allowed_splits is not None:
            require(case["split"] in set(allowed_splits),
                    f"Case split not admitted for this stage: {case_id}")
        rows_by_id[case_id] = row
    require(set(rows_by_id) == set(case_by_id),
            "Model rows and verifier cases differ")

    group_splits: dict[str, str] = {}
    for asset in assets.values():
        group = asset.get("group_id")
        require(isinstance(group, str) and group, f"Asset lacks group_id: {asset['asset_id']}")
        previous = group_splits.setdefault(group, asset["split"])
        require(previous == asset["split"], f"Family crosses splits: {group}")
    declared_groups = manifest.get("groups")
    require(isinstance(declared_groups, list) and declared_groups,
            "Manifest lacks group catalog")
    declared_ids = set()
    for group in declared_groups:
        group_id = group.get("family_id") or group.get("world_id")
        require(group_id in group_splits, f"Declared group lacks assets: {group_id}")
        require(group.get("split") == group_splits[group_id],
                f"Declared group split differs: {group_id}")
        require(set(group.get("asset_ids", ())) <= set(assets),
                f"Declared group has unknown assets: {group_id}")
        declared_ids.add(group_id)
    require(declared_ids == set(group_splits), "Group catalog differs from assets")
    counts = manifest.get("counts", {})
    require(counts.get("assets") == len(assets)
            and counts.get("model_rows") == len(rows)
            and counts.get("verifier_cases") == len(cases)
            and counts.get("verifier_scenes") == len(scenes),
            "Manifest counts differ from loaded records")

    return VisualDataset(stage=stage, root=root, manifest=manifest,
                         model_rows=tuple(rows), verifier_cases=tuple(cases),
                         verifier_scenes=tuple(scenes), assets=assets,
                         cases=case_by_id, scenes=scene_by_asset,
                         pins={"manifest": manifest_sha, "model_rows": rows_sha,
                               "verifier_cases": cases_sha,
                               "verifier_scenes": scenes_sha})


# ---------------------------------------------------------------------------
# Deterministic selections and prompt construction


def select_probe(dataset: VisualDataset, v0: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    case_id = v0["probe"]["case_id"]
    wrong_source = v0["probe"]["wrong_image_case_id"]
    probe_case = dataset.case(case_id)
    wrong_case = dataset.case(wrong_source)
    require(probe_case["split"] == v0["probe"]["split"],
            "V0 probe case split differs from the declaration")
    require(wrong_case["split"] == v0["probe"]["split"],
            "V0 wrong-image source split differs from the declaration")
    require(probe_case["group_id"] != wrong_case["group_id"],
            "V0 wrong-image source must come from another family")
    require(probe_case["asset_id"] != wrong_case["asset_id"],
            "V0 wrong-image source must differ")
    return {"case": probe_case, "wrong_source": wrong_case}


def select_eval_cases(dataset: VisualDataset, evaluation: Mapping[str, Any]
                      ) -> list[Mapping[str, Any]]:
    splits = set(evaluation["splits"])
    rows = [row for row in dataset.model_rows
            if dataset.case(row["case_id"])["split"] in splits]
    require(bool(rows), "No evaluation rows in the declared splits")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[dataset.case(row["case_id"])["group_id"]].append(row)
    families = sorted(grouped)
    random.Random(evaluation["seed"]).shuffle(families)
    for family, items in grouped.items():
        items.sort(key=lambda row: row["case_id"])
        random.Random(evaluation["seed"] * 1_000_003
                      + int(sha256_text(family)[:16], 16)).shuffle(items)
    selected: list[Mapping[str, Any]] = []
    while len(selected) < evaluation["max_cases"]:
        progressed = False
        for family in families:
            items = grouped[family]
            if not items:
                continue
            selected.append(items.pop(0))
            progressed = True
            if len(selected) >= evaluation["max_cases"]:
                break
        require(progressed, "Evaluation selection exhausted declared rows")
    return selected


def assign_controls(dataset: VisualDataset, case_ids: Sequence[str]
                    ) -> dict[str, dict[str, str]]:
    """Deterministic wrong/shuffled image providers; no model answer involved."""
    require(len(case_ids) >= 2, "At least two evaluation cases are needed for controls")
    require(len(set(case_ids)) == len(case_ids), "Control case IDs repeat")
    families = sorted({dataset.case(case_id)["group_id"] for case_id in case_ids})
    by_family: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(case_ids):
        by_family[dataset.case(case_id)["group_id"]].append(case_id)
    wrong: dict[str, str] = {}
    for case_id in sorted(case_ids):
        family = dataset.case(case_id)["group_id"]
        own_image = dataset.case(case_id)["asset_id"]
        choices = []
        for other_family in families:
            if other_family == family:
                continue
            choices.extend(by_family[other_family])
        require(bool(choices), f"No different-family image available for {case_id}")
        selected = choices[0]
        require(dataset.case(selected)["asset_id"] != own_image,
                f"Wrong-image provider matches the case image: {case_id}")
        wrong[case_id] = selected
    shuffled: dict[str, str] = {}
    ordered = sorted(case_ids, key=lambda case_id: (dataset.case(case_id)["asset_id"], case_id))
    providers = ordered[1:] + ordered[:1]
    # Rotate until no case keeps either its own identity or its own image. A
    # deterministic search keeps the control reproducible without host choice.
    for index, case_id in enumerate(ordered):
        own_asset = dataset.case(case_id)["asset_id"]
        if (providers[index] != case_id
                and dataset.case(providers[index])["asset_id"] != own_asset):
            continue
        swapped = False
        for other in range(len(ordered)):
            candidate = providers[other]
            if (candidate != case_id
                    and dataset.case(candidate)["asset_id"] != own_asset):
                providers[index], providers[other] = providers[other], providers[index]
                swapped = True
                break
        require(swapped, f"No shuffled-image provider available for {case_id}")
    for case_id, provider in zip(ordered, providers):
        require(provider != case_id,
                f"Shuffled-image derangement contains a fixed point: {case_id}")
        require(dataset.case(provider)["asset_id"] != dataset.case(case_id)["asset_id"],
                f"Shuffled-image provider repeats the same image: {case_id}")
        shuffled[case_id] = provider
    return {"wrong": wrong, "shuffled": shuffled}


def build_text_context_prompt(question: str, context: str, template: str) -> str:
    require("{context}" in template and "{question}" in template,
            "Text-context template must use {context} and {question}")
    value = template.replace("{context}", context).replace("{question}", question)
    require(value.strip(), "Text-context prompt is empty")
    return value


def leakage_hits(text: str, answer: str) -> bool:
    """Check a standalone answer token/phrase, not an arbitrary substring."""
    if not answer or not answer.strip():
        return False
    pattern = r"(?<!\w)" + re.escape(answer.casefold()) + r"(?!\w)"
    return re.search(pattern, text.casefold()) is not None


def visual_question(protocol: Mapping[str, Any], question: str) -> str:
    """The same label-free short-answer prefix in training and every control."""
    require(protocol["v1"]["question_template"] == VISUAL_QUESTION_TEMPLATE,
            "Unpinned visual question template")
    return VISUAL_QUESTION_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Budget / schedule


def training_schedule(rows: Sequence[Mapping[str, Any]], train: Mapping[str, Any]
                      ) -> list[list[Mapping[str, Any]]]:
    """Deterministic repeated shuffle; one list per optimizer update."""
    require(bool(rows), "Training rows are empty")
    _subset_keys(train, ("seed", "repeats", "updates", "grad_accum",
                         "schedule_version", "repeat_policy"), "training schedule")
    for field in ("repeats", "updates", "grad_accum"):
        _positive_int(train[field], f"train.{field}")
    _nonnegative_int(train["seed"], "train.seed")
    require(train["schedule_version"] == VISION_TRAINING_SCHEDULE,
            "Unknown visual training schedule")
    require(train["repeat_policy"] in ("once", "declared_epochs")
            and (train["repeat_policy"] != "once" or train["repeats"] == 1),
            "Visual training repeat policy differs from declared repeats")
    identifiers = [_text(row["case_id"], "training case ID") for row in rows]
    require(len(set(identifiers)) == len(identifiers), "Training source case IDs repeat")
    seed = train["seed"]
    repeats = train["repeats"]
    updates = train["updates"]
    accumulation = train["grad_accum"]
    schedule: list[Mapping[str, Any]] = []
    for repeat in range(repeats):
        order = list(rows)
        random.Random(seed * 1_000_003 + repeat).shuffle(order)
        schedule.extend(order)
    needed = updates * accumulation
    require(len(schedule) >= needed,
            f"Training schedule provides {len(schedule)} rows but {needed} are required")
    return [schedule[start:start + accumulation]
            for start in range(0, needed, accumulation)]


def validate_training_trace(trace: Sequence[Mapping[str, Any]],
                            examples: Sequence[Mapping[str, Any]],
                            rows: Sequence[Mapping[str, Any]], train: Mapping[str, Any],
                            tokenizer: Any, projector_names: Iterable[str]) -> None:
    """Audit ordered updates, exact examples and real backward-invocation rows."""
    schedule = training_schedule(rows, train)
    require(len(trace) == len(schedule), "Training update inventory differs")
    flattened = []
    expected_names = set(projector_names)
    example_fields = {"update", "case_id", "backward_called", "loss",
                      "prompt_tokens", "response_tokens", "input_ids_sha256", "labels_sha256"}
    for index, (summary, expected_rows) in enumerate(zip(trace, schedule), 1):
        _positive_int(summary.get("update"), "training update index")
        require(summary["update"] == index, "Training updates are missing, duplicated or reordered")
        require(type(summary.get("seed")) is int and summary["seed"] == train["seed"],
                "Training seed differs")
        expected_repeat = ((index - 1) * train["grad_accum"]) // len(rows) + 1
        require(type(summary.get("repeat")) is int and summary["repeat"] == expected_repeat,
                "Training repeat accounting differs")
        require(summary["case_ids"] == [row["case_id"] for row in expected_rows],
                "Training schedule/case order differs from the declared seed")
        group = summary.get("examples")
        require(isinstance(group, list) and len(group) == len(expected_rows),
                "Per-update example inventory differs")
        require(summary["losses"] == [item["loss"] for item in group],
                "Per-update losses differ from example rows")
        prompts = responses = 0
        for item, row in zip(group, expected_rows):
            _exact_keys(item, example_fields, "training example")
            require(type(item["update"]) is int and item["update"] == index
                    and item["case_id"] == row["case_id"], "Training example membership differs")
            require(item["backward_called"] is True, "Training example lacks an actual backward call")
            _number(item["loss"], "training example loss", minimum=0.0)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            prompt = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            response = tokenizer(row["response"], add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id]
            for key, count in (("prompt_tokens", len(prompt)), ("response_tokens", len(response))):
                require(type(item[key]) is int and item[key] == count,
                        f"Training example {key} differs from independent tokenization")
            require(item["input_ids_sha256"] == sha256_json(prompt + response)
                    and item["labels_sha256"] == sha256_json([-100] * len(prompt) + response),
                    "Training example input or response-only labels differ")
            prompts += len(prompt)
            responses += len(response)
        require(type(summary["backward_calls"]) is int
                and summary["backward_calls"] == len(group), "Backward-call accumulation differs")
        require(type(summary["prompt_tokens"]) is int and summary["prompt_tokens"] == prompts
                and type(summary["response_tokens"]) is int
                and summary["response_tokens"] == responses, "Accumulated token accounting differs")
        _number(summary["loss"], "training update loss", minimum=0.0)
        require(abs(summary["loss"] - sum(item["loss"] for item in group) / len(group)) <= 1e-6,
                "Per-update mean loss differs from examples")
        _number(summary["grad_norm"], "training accumulated gradient norm", minimum=0.0)
        require(summary["max_grad_norm"] == train["max_grad_norm"],
                "Training clipping policy differs")
        names = summary["projector_gradient_names"]
        require(len(names) == len(set(names)) and set(names) == expected_names
                and summary["nonprojector_gradient_names"] == [], "Gradient ownership differs")
        require(all(summary[field] is True for field in (
            "response_only_labels", "decoder_autograd", "text_aligned_logits",
            "base_parameter_stamp_unchanged")), "Training graph/label/frozen-state contract failed")
        flattened.extend(group)
    require(flattened == list(examples), "Raw example rows differ from the ordered update groups")


def validate_budget(plan: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    _subset_keys(plan, ("updates", "evaluation_cases", "generation_tokens",
                        "training_rows", "max_seconds"), "budget plan")
    require(plan["updates"] <= limits["max_updates"],
            "Budget plan exceeds max_updates")
    require(plan["evaluation_cases"] <= limits["max_eval_cases"],
            "Budget plan exceeds max_eval_cases")
    require(plan["generation_tokens"] <= limits["max_new_tokens"],
            "Budget plan exceeds max_new_tokens")
    require(plan["training_rows"] <= limits["max_rows"],
            "Budget plan exceeds max_rows")
    require(plan["max_seconds"] <= limits["max_seconds"],
            "Budget plan exceeds max_seconds")


# ---------------------------------------------------------------------------
# Admission


def admission_document(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    pin = protocol["v1"]["prerequisites"]["data_admission"]
    document, digest = load_pinned_json(protocol, pin, "v1 data admission")
    require(document.get("format") == ADMISSION_FORMAT,
            "Unknown training admission format")
    require(document.get("admitted") is True,
            "Training admission must explicitly admit rows")
    _positive_int(document.get("admitted_training_rows"), "admission.admitted_training_rows")
    _positive_int(document.get("eligible_rows"), "admission.eligible_rows")
    require(document.get("run_id") == protocol["v1"]["run_id"],
            "Training admission belongs to another named run")
    data = document.get("data")
    require(isinstance(data, dict), "Admission lacks data hashes")
    return document, digest


def verify_admission(protocol: Mapping[str, Any], dataset: VisualDataset,
                     *, train_splits: Sequence[str], eval_splits: Sequence[str]
                     ) -> dict[str, Any]:
    document, digest = admission_document(protocol)
    expected_pins = protocol["data"]
    for key, pin_name in (("manifest", "manifest"), ("model_rows", "model_rows"),
                          ("verifier_cases", "verifier_cases"),
                          ("verifier_scenes", "verifier_scenes")):
        require(document["data"].get(key) == expected_pins[key]["sha256"],
                f"Admission {key} hash differs from protocol data pins")
        require(dataset.pins[key] == expected_pins[key]["sha256"],
                f"Loaded {key} hash differs from protocol data pins")
    splits = document.get("splits")
    require(isinstance(splits, dict) and splits, "Admission lacks split inventory")
    require(set(splits) == set(train_splits) | set(eval_splits)
            and set(train_splits) == {"train"} and "train" not in eval_splits,
            "Admission split roles differ; evaluation material is never training")
    for split in (*train_splits, *eval_splits):
        require(split in splits, f"Admission does not name split: {split}")
        entry = splits[split]
        _subset_keys(entry, ("model_rows_sha256", "model_row_count",
                             "family_count", "row_ids_sha256"), f"admission.splits.{split}")
        rows = [dict(row) for row in dataset.rows_for_split(split)]
        require(entry["model_row_count"] == len(rows),
                f"Admission row count differs: {split}")
        require(entry["model_rows_sha256"] == sha256_text(
            "".join(canonical_json(row) + "\n" for row in rows)),
            f"Admission model-row hash differs: {split}")
        require(entry["family_count"] == len({dataset.case(row["case_id"])["group_id"]
                                              for row in rows}),
                f"Admission family count differs: {split}")
        require(entry["row_ids_sha256"] == sha256_json(
            sorted(row["case_id"] for row in rows)),
            f"Admission row-ID hash differs: {split}")
    require(document["eligible_rows"] == sum(
        document["splits"][split]["model_row_count"] for split in document["splits"]),
        "Admission eligible_rows differs from source split inventory")

    training_pin = document.get("training_rows")
    require(isinstance(training_pin, dict), "Admission must name training rows")
    _subset_keys(training_pin, ("path", "sha256", "format", "rows_sha256", "count"),
                 "admission.training_rows")
    require(training_pin["format"] == TRAINING_ROWS_FORMAT,
            "Unknown training-row format")
    training_rows, _ = load_pinned_jsonl(protocol, training_pin, "training rows")
    require(training_pin["count"] == len(training_rows),
            "Training-row count differs from admission")
    require(training_pin["rows_sha256"] == rows_sha256(training_rows),
            "Training-row hash differs from admission")
    require(document["admitted_training_rows"] == len(training_rows),
            "Admitted training count differs from the exact selected rows")
    seen: set[str] = set()
    for row in training_rows:
        _exact_keys(row, TRAINING_ROW_FIELDS, "training row")
        case_id = _text(row["case_id"], "training row case_id")
        require(case_id not in seen, f"Duplicate training row: {case_id}")
        seen.add(case_id)
        require(row["split"] in set(train_splits),
                f"Training row split is not admitted: {case_id}")
        require(case_id in dataset.cases,
                f"Training row lacks a verified dataset case: {case_id}")
        case = dataset.case(case_id)
        require(row["split"] == case["split"],
                f"Training row split differs from its verified case: {case_id}")
        require(row["family"] == case["group_id"],
                f"Training row family differs from its verified case: {case_id}")
        prompt = _text(row["prompt"], f"training row prompt: {case_id}")
        response = _text(row["response"], f"training row response: {case_id}")
        require(prompt == visual_question(protocol, dataset.row(case_id)["question"]),
                f"Training prompt differs from its verified question: {case_id}")
        require(response == dataset.label(case_id),
                f"Training response differs from independent verifier truth: {case_id}")
        require(not leakage_hits(prompt, response),
                f"Training response leaks into its prompt: {case_id}")
        require(row["family"] == row["group_id"], f"Training family/group differ: {case_id}")
        image = resolve_relative(dataset.root, row["image"])
        require(image.is_file(), f"Training image missing: {case_id}")
        require(image == dataset.image_path(case_id),
                f"Training image is not the verified case image: {case_id}")
        require(sha256_file(image) == _sha(row["image_sha256"],
                                           f"training row image_sha256: {case_id}"),
                f"Training image bytes changed: {case_id}")
    document["_training_rows"] = training_rows

    contexts = document.get("text_contexts")
    required_controls = set(protocol["v1"]["eval"]["required_controls"])
    required_kinds = {TEXT_CONTROL_KIND[control] for control in required_controls
                      if control in TEXT_CONTROLS}
    if contexts is not None:
        require(isinstance(contexts, dict), "Admission text_contexts must be an object")
        _subset_keys(contexts, ("path", "sha256", "rows_sha256", "format"),
                     "admission.text_contexts")
        require(contexts["format"] == TEXT_CONTEXTS_FORMAT,
                "Unknown text-context format")
        _validate_pin(contexts, "admission.text_contexts")
        rows, _ = load_pinned_jsonl(protocol, contexts, "text contexts")
        require(contexts["rows_sha256"] == rows_sha256(rows),
                "Text-context row hash differs from admission")
        eval_ids = {row["case_id"] for row in dataset.model_rows
                    if dataset.case(row["case_id"])["split"] in set(eval_splits)}
        pairs: set[tuple[str, str]] = set()
        for row in rows:
            _exact_keys(row, TEXT_CONTEXT_FIELDS, "text context row")
            require(row["case_id"] in eval_ids, "Text context references a non-eval case")
            require(row["kind"] in TEXT_CONTEXT_KINDS, "Unknown text-context kind")
            require((row["case_id"], row["kind"]) not in pairs,
                    f"Duplicate text context: {row['case_id']}/{row['kind']}")
            pairs.add((row["case_id"], row["kind"]))
            require(sha256_text(row["text"]) == row["text_sha256"],
                    f"Text context bytes changed: {row['case_id']}")
            require(row["information_bound"] in ("independent", "oracle_upper_bound"),
                    "Text context must declare its information bound")
        selected = select_eval_cases(dataset, protocol["v1"]["eval"])
        for kind in required_kinds:
            missing = [row["case_id"] for row in selected
                       if (row["case_id"], kind) not in pairs]
            require(not missing,
                    f"Required {kind} control lacks pinned text for {len(missing)} cases")
        document["_text_contexts"] = rows
    else:
        require(not required_kinds,
                "Admitted text controls require a pinned text-contexts file")

    retention_pins = protocol["v1"]["retention"]
    prompt_rows, _ = load_pinned_jsonl(protocol, retention_pins["prompts"], "retention prompts")
    label_rows, _ = load_pinned_jsonl(protocol, retention_pins["labels"], "retention labels")
    prompt_by_id: dict[str, Mapping[str, Any]] = {}
    for row in prompt_rows:
        _exact_keys(row, RETENTION_PROMPT_FIELDS, "retention prompt")
        require(row["case_id"] not in prompt_by_id, "Retention prompt IDs repeat")
        require(row["split"] in set(retention_pins["splits"]),
                "Retention prompt split is not admitted")
        prompt_by_id[row["case_id"]] = row
    label_by_id: dict[str, Mapping[str, Any]] = {}
    for row in label_rows:
        _exact_keys(row, RETENTION_LABEL_FIELDS, "retention label")
        require(row["case_id"] not in label_by_id, "Retention label IDs repeat")
        require(row["case_id"] in prompt_by_id, "Retention label lacks a prompt")
        label_by_id[row["case_id"]] = row
    require(set(prompt_by_id) == set(label_by_id),
            "Retention prompts and labels differ")
    require(len(prompt_by_id) <= retention_pins["max_cases"],
            "Retention rows exceed the declared bound")
    for case_id, row in prompt_by_id.items():
        require(not leakage_hits(row["prompt"], label_by_id[case_id]["answer"]),
                f"Retention answer leaks into its prompt: {case_id}")
    document["_retention_prompts"] = prompt_rows
    document["_retention_labels"] = label_rows
    return document


def verify_v0_prerequisite(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    prerequisites = protocol["v1"]["prerequisites"]
    report, digest = load_pinned_json(protocol, prerequisites["v0_score_report"],
                                      "accepted V0 score report")
    require(report.get("format") == SCORE_FORMAT and report.get("stage") == "v0",
            "Prerequisite is not a V0 qualification score")
    require(report.get("accepted") is True,
            "V0 protocol report was not independently accepted")
    require(report.get("protocol_sha256") == prerequisites["v0_protocol_sha256"],
            "V0 score report belongs to another protocol")
    require(is_sha256(report.get("run_report_sha256")),
            "V0 score report lacks the scored run-report hash")
    require(report.get("engineering_pass") is True,
            "V0 score report did not pass engineering checks")
    require(isinstance(report.get("run_report_path"), str)
            and report["run_report_path"].strip(),
            "V0 score report lacks the scored run-report path")
    run_report = resolve_project_path(protocol, report["run_report_path"])
    require(run_report.is_file(), "Scored V0 run report is missing")
    require(sha256_file(run_report) == report["run_report_sha256"],
            "Scored V0 run report bytes changed")
    accepted_spec = report.get("vision_spec")
    if not isinstance(accepted_spec, dict):
        native_run = json.loads(run_report.read_text())
        accepted_spec = native_run.get("vision_spec")
    require(isinstance(accepted_spec, dict), "Accepted V0 lacks vision spec evidence")
    immutable_fields = ("encoder_repo", "encoder_revision", "encoder_source_digest",
                        "processor_digest", "encoder_width", "image_size", "patch_size",
                        "max_visual_tokens", "feature_version", "compute_dtype")
    require(all(accepted_spec.get(field) == protocol["vision"][field]
                for field in immutable_fields),
            "Accepted V0 uses another vision encoder or processor")
    require(report.get("model_revision") == protocol["model"]["revision"]
            and report.get("model_lock_sha256") == protocol["model"]["lock_sha256"],
            "Accepted V0 uses another sealed text source")
    checkpoint = protocol["v1"].get("initial_checkpoint")
    if checkpoint is not None:
        require(report.get("extension", {}).get("manifest_sha256")
                == checkpoint["manifest_sha256"]
                and report["extension"]["weights_sha256"] == checkpoint["weights_sha256"],
                "V1 initial checkpoint is not the independently accepted V0 extension")
    return report, digest


# ---------------------------------------------------------------------------
# Extension / raw-artifact validation


def load_extension_receipt(path: str | Path, *, vision_pins: Mapping[str, Any]
                           ) -> dict[str, Any]:
    path = Path(path)
    receipt_path = path / "manifest.json"
    weights_path = path / "extensions.safetensors"
    require(receipt_path.is_file() and weights_path.is_file(),
            f"Extension is incomplete: {path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise QualificationError("Extension manifest is not valid JSON") from error
    _exact_keys(receipt, ("format", "manifest", "manifest_digest", "regime",
                          "weights_sha256", "parameters", "base_saved", "notes",
                          "components"), "extension receipt")
    require(receipt["format"] == EXTENSION_FORMAT,
            "Vision extension must use the strict v4 format")
    require(receipt["regime"] == "modules_only",
            "Vision extension must retain modules_only scope")
    require(receipt["base_saved"] is False, "Extension must not save base weights")
    actual = sha256_file(weights_path)
    require(actual == receipt["weights_sha256"],
            "Extension weight bytes differ from the receipt")
    manifest = receipt["manifest"]
    require(isinstance(manifest, dict), "Extension manifest is not an object")
    require(manifest.get("training_scope") == "visual_prefix_text",
            "Extension manifest does not declare visual-prefix text scope")
    require(manifest.get("vision") == dict(vision_pins),
            "Extension vision pins differ from the protocol")
    require(receipt["manifest_digest"] == fingerprint(manifest),
            "Extension manifest digest differs")
    require("vision_projector" in receipt.get("components", {}),
            "Extension component inventory lacks vision_projector")
    return receipt


def extension_tensor_names(path: str | Path) -> list[str]:
    from safetensors import safe_open
    weights = Path(path) / "extensions.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        return sorted(handle.keys())


def validate_extension_inventory(path: str | Path, *,
                                 expected_names: Iterable[str],
                                 vision_pins: Mapping[str, Any],
                                 require_projector: bool = True) -> dict[str, Any]:
    receipt = load_extension_receipt(path, vision_pins=vision_pins)
    names = extension_tensor_names(path)
    expected = sorted(expected_names)
    require(names == expected,
            f"Extension inventory differs: missing={sorted(set(expected) - set(names))} "
            f"extra={sorted(set(names) - set(expected))}")
    require(any(name.startswith("vision_projector.") for name in names),
            "Extension lacks projector tensors")
    require(not any(name.startswith("base.") for name in names),
            "Extension contains native base tensors")
    receipt["tensor_names"] = names
    return receipt


def compare_extension_tensors(initial: str | Path, final: str | Path) -> dict[str, Any]:
    """Only projector tensors may change; every other extension tensor is frozen."""
    from safetensors import safe_open
    initial_path = Path(initial) / "extensions.safetensors"
    final_path = Path(final) / "extensions.safetensors"
    require(initial_path.is_file() and final_path.is_file(),
            "Extension tensor files are missing")
    with safe_open(str(initial_path), framework="pt", device="cpu") as left:
        left_keys = sorted(left.keys())
        with safe_open(str(final_path), framework="pt", device="cpu") as right:
            require(left_keys == sorted(right.keys()),
                    "Initial/final extension tensor inventories differ")
            changed: list[str] = []
            frozen: list[str] = []
            for name in left_keys:
                first, second = left.get_tensor(name), right.get_tensor(name)
                require((first.shape, first.dtype) == (second.shape, second.dtype),
                        f"Extension tensor layout changed: {name}")
                if not bool((first == second).all()):
                    changed.append(name)
                else:
                    frozen.append(name)
    unexpected = [name for name in changed
                  if not name.startswith(ALLOWED_EXTENSION_PREFIXES)]
    require(not unexpected,
            f"Non-projector extension tensors changed: {unexpected}")
    require(bool(changed), "Projector training changed no projector tensor")
    return {"changed": changed, "frozen": frozen, "changed_count": len(changed),
            "frozen_count": len(frozen)}


def validate_generation_trace(trace: Mapping[str, Any], logits_path: str | Path, *,
                              protocol_prompt_ids: Sequence[int],
                              max_new_tokens: int) -> dict[str, Any]:
    """Verify saved logits reproduce the saved greedy tokens exactly."""
    import torch
    from safetensors.torch import load_file
    _subset_keys(trace, ("prompt_ids", "generated_ids", "logits_sha256",
                         "prompt_sha256"), "generation trace")
    prompt_ids = list(trace["prompt_ids"])
    generated = list(trace["generated_ids"])
    require(prompt_ids == list(protocol_prompt_ids),
            "Generation prompt tokens differ from the independent tokenization")
    require(trace["prompt_sha256"] == sha256_json(prompt_ids),
            "Generation prompt hash differs")
    require(1 <= len(generated) <= max_new_tokens,
            "Generation exceeds the declared token budget")
    require(sha256_file(logits_path) == trace["logits_sha256"],
            "Generation logits bytes changed")
    tensors = load_file(str(logits_path))
    require(set(tensors) == {"logits"}, "Generation logits file inventory differs")
    logits = tensors["logits"]
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    require(logits.ndim == 2 and logits.shape[0] == len(generated)
            and logits.shape[1] > 0,
            "Generation logits shape differs from generated-token count")
    require(bool(torch.isfinite(logits).all()), "Generation logits are nonfinite")
    argmax = [int(index) for index in logits.argmax(dim=-1).tolist()]
    require(argmax == generated,
            "Saved greedy tokens do not match independently recomputed argmax logits")
    return {"generated_ids": generated, "logits": logits,
            "ended_by_eos": bool(trace.get("ended_by_eos", False))}


def compare_restart_traces(prior: Mapping[str, Any], restart: Mapping[str, Any], *,
                           atol: float, rtol: float) -> dict[str, Any]:
    prior_ids = list(prior.get("generated_ids", ()))
    restart_ids = list(restart.get("generated_ids", ()))
    require(prior_ids == restart_ids, "Fresh-process generation tokens differ")
    require(bool(prior_ids), "Empty restart generation")
    from safetensors.torch import load_file
    left = load_file(str(prior["logits_path"]))["logits"]
    right = load_file(str(restart["logits_path"]))["logits"]
    require(tuple(left.shape) == tuple(right.shape),
            "Fresh-process logits shapes differ")
    import torch
    maximum = float((left.float() - right.float()).abs().max().item()) if left.numel() else 0.0
    require(bool(torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol)),
            f"Fresh-process logits differ beyond tolerance (max={maximum})")
    require(prior.get("prompt_ids") == restart.get("prompt_ids"),
            "Fresh-process prompt tokens differ")
    return {"tokens_equal": True, "logits_equal": True,
            "maximum_absolute_difference": maximum,
            "tolerance": {"atol": atol, "rtol": rtol}}


def ledger_record_count(path: str | Path) -> int:
    path = Path(path)
    require(path.is_file(), f"Ledger is missing: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        rows = database.execute("SELECT COUNT(*) FROM memory_records").fetchone()
    require(rows is not None, "Ledger record count unavailable")
    return int(rows[0])


# ---------------------------------------------------------------------------
# Independent answer and quality scoring


def normalise_answer(value: str) -> str:
    text = " ".join(str(value).strip().casefold().split())
    return text.strip(" \t\r\n.,;:!?\"'`()[]{}")


def is_unknown(answer: str) -> bool:
    normalised = normalise_answer(answer)
    return (not normalised or normalised in UNKNOWN_ANSWERS
            or "unknown" in normalised.split() or normalised.startswith("cannot determine"))


def answer_correct(generated: str, expected: str) -> bool:
    return normalise_answer(generated) == normalise_answer(expected)


@dataclass(frozen=True)
class ScoreRow:
    control: str
    case_id: str
    family: str
    split: str
    expected: str
    generated: str
    correct: bool
    unknown: bool


def build_score_rows(dataset: VisualDataset, generations: Sequence[Mapping[str, Any]],
                     *, controls: Sequence[str]) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    seen: set[tuple[str, str]] = set()
    for generation in generations:
        _subset_keys(generation, ("control", "case_id", "generated_text"),
                     "generation row")
        control = generation["control"]
        require(control in controls, f"Undeclared generation control: {control}")
        case_id = generation["case_id"]
        case = dataset.case(case_id)
        key = (control, case_id)
        require(key not in seen, f"Duplicate generation: {control}/{case_id}")
        seen.add(key)
        expected = dataset.label(case_id)
        generated = str(generation["generated_text"])
        rows.append(ScoreRow(control=control, case_id=case_id,
                             family=case["group_id"], split=case["split"],
                             expected=expected, generated=generated,
                             correct=answer_correct(generated, expected),
                             unknown=is_unknown(generated)))
    return rows


def _rate(rows: Sequence[ScoreRow], key) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if key(row)) / len(rows)


def score_controls(dataset: VisualDataset, generations: Sequence[Mapping[str, Any]],
                   *, controls: Sequence[str]) -> dict[str, Any]:
    rows = build_score_rows(dataset, generations, controls=controls)
    by_control: dict[str, dict[str, Any]] = {}
    for control in controls:
        selected = [row for row in rows if row.control == control]
        if not selected:
            by_control[control] = {"available": False}
            continue
        families: dict[str, list[ScoreRow]] = defaultdict(list)
        for row in selected:
            families[row.family].append(row)
        by_control[control] = {
            "available": True,
            "cases": len(selected),
            "correct": sum(row.correct for row in selected),
            "accuracy": _rate(selected, lambda row: row.correct),
            "unknown": sum(row.unknown for row in selected),
            "unknown_rate": _rate(selected, lambda row: row.unknown),
            "unique_families": len(families),
            "family_accuracy": {family: _rate(items, lambda row: row.correct)
                                for family, items in sorted(families.items())},
        }
    correct_rows = {row.case_id: row for row in rows if row.control == "correct"}
    paired: dict[str, Any] = {}
    for control in CONTROLS:
        if control == "correct" or control not in controls:
            continue
        other = {row.case_id: row for row in rows if row.control == control}
        shared = sorted(set(correct_rows) & set(other))
        if not shared:
            paired[control] = {"cases": 0, "correct_margin": None}
            continue
        correct_accuracy = sum(correct_rows[case_id].correct for case_id in shared) / len(shared)
        other_accuracy = sum(other[case_id].correct for case_id in shared) / len(shared)
        paired[control] = {"cases": len(shared),
                           "correct_accuracy": correct_accuracy,
                           "control_accuracy": other_accuracy,
                           "correct_margin": correct_accuracy - other_accuracy}
    return {"controls": by_control, "paired": paired,
            "rows": [row.__dict__ for row in rows]}


def score_retention(prompt_rows: Sequence[Mapping[str, Any]],
                    label_rows: Sequence[Mapping[str, Any]],
                    generations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = {row["case_id"]: row["answer"] for row in label_rows}
    prompts = {row["case_id"]: row for row in prompt_rows}
    results = []
    seen = set()
    for generation in generations:
        case_id = generation.get("case_id")
        require(case_id in prompts and case_id in labels,
                f"Retention generation references unknown case: {case_id}")
        require(case_id not in seen, f"Duplicate retention generation: {case_id}")
        seen.add(case_id)
        expected = labels[case_id]
        generated = str(generation.get("generated_text", ""))
        results.append({"case_id": case_id,
                        "family": prompts[case_id]["group_id"],
                        "correct": answer_correct(generated, expected),
                        "unknown": is_unknown(generated),
                        "expected": expected, "generated": generated})
    require(seen == set(prompts), "Retention generations are incomplete")
    families = {item["family"] for item in results}
    return {"cases": len(results), "correct": sum(item["correct"] for item in results),
            "accuracy": (sum(item["correct"] for item in results) / len(results)
                         if results else None),
            "unknown": sum(item["unknown"] for item in results),
            "unknown_rate": (sum(item["unknown"] for item in results) / len(results)
                             if results else None),
            "unique_families": len(families), "rows": results}


def apply_quality_gates(scores: Mapping[str, Any], retention: Mapping[str, Any],
                        gates: Mapping[str, Any],
                        *, retention_baseline_accuracy: float | None = None,
                        retention_policy: str | None = None,
                        text_tokens_unchanged: bool | None = None
                        ) -> dict[str, Any]:
    """Predeclared numeric quality gates. Engineering can pass while quality fails.

    ``retention_policy='component_nonregression.v1'`` separates two questions
    that the legacy gate conflated: whether the frozen baseline is capable
    enough (``min_text_retention_accuracy`` applied to the *baseline*) and
    whether this component changed text behaviour (drop and token parity).
    A projector-only arm that reproduces text exactly can still be ineligible
    when the baseline itself is below the declared capability floor; that is
    reported as ``baseline_capability_ineligible`` instead of blaming the
    unchanged model.
    """
    require(retention_policy in (None, "component_nonregression.v1"),
            "Unknown retention policy")
    controls = scores["controls"]
    paired = scores["paired"]
    correct = controls.get("correct", {})
    require(correct.get("available") is True, "Correct-control score is unavailable")
    failures: list[str] = []
    values = {
        "correct_accuracy": correct["accuracy"],
        "unique_families": correct["unique_families"],
        "unknown_rate": correct["unknown_rate"],
        "text_retention_accuracy": retention["accuracy"],
    }
    if correct["accuracy"] < gates["min_correct_accuracy"]:
        failures.append("min_correct_accuracy")
    if correct["unique_families"] < gates["min_unique_families"]:
        failures.append("min_unique_families")
    if correct["unknown_rate"] > gates["max_unknown_rate"]:
        failures.append("max_unknown_rate")
    for control in ("wrong", "shuffled", "noimage", "captiononly", "ocrtextonly"):
        gate = f"min_correct_margin_{control}"
        if control not in controls or controls[control].get("available") is not True:
            failures.append(f"missing_control_{control}")
            continue
        margin = paired.get(control, {}).get("correct_margin")
        if margin is None or margin < gates[gate]:
            failures.append(gate)
    baseline_eligible = (None if retention_baseline_accuracy is None
                         else float(retention_baseline_accuracy)
                         >= gates["min_text_retention_accuracy"])
    if retention_policy == "component_nonregression.v1":
        if retention["accuracy"] is None:
            failures.append("retention_unavailable")
        if baseline_eligible is None:
            failures.append("retention_baseline_unavailable")
        elif not baseline_eligible:
            failures.append("baseline_capability_ineligible")
        if text_tokens_unchanged is not True:
            failures.append("text_tokens_changed")
        if retention_baseline_accuracy is None:
            failures.append("retention_baseline_unavailable")
        else:
            drop = float(retention_baseline_accuracy) - float(retention["accuracy"])
            if drop > gates["max_text_retention_drop"]:
                failures.append("max_text_retention_drop")
        failures = list(dict.fromkeys(failures))
    else:
        if retention["accuracy"] is None or retention["accuracy"] < gates["min_text_retention_accuracy"]:
            failures.append("min_text_retention_accuracy")
        if retention_baseline_accuracy is None:
            failures.append("retention_baseline_unavailable")
        else:
            drop = float(retention_baseline_accuracy) - float(retention["accuracy"])
            if drop > gates["max_text_retention_drop"]:
                failures.append("max_text_retention_drop")
    return {"quality_pass": not failures, "failures": failures,
            "values": {**values, "retention_baseline_accuracy": retention_baseline_accuracy,
                       "baseline_eligible": baseline_eligible,
                       "text_tokens_unchanged": text_tokens_unchanged,
                       "retention_policy": retention_policy,
                       "retention_drop": (None if retention_baseline_accuracy is None
                                          or retention["accuracy"] is None
                                          else float(retention_baseline_accuracy)
                                          - float(retention["accuracy"]))},
            "gates": dict(gates)}

def summarize_engineering(checks: Mapping[str, bool]) -> dict[str, Any]:
    require(bool(checks), "No engineering checks were supplied")
    failed = sorted(name for name, value in checks.items() if value is not True)
    return {"engineering_pass": not failed, "engineering_failures": failed,
            "checks": dict(sorted(checks.items()))}


# ---------------------------------------------------------------------------
# Optimizer routing


def projector_optimizer_groups(projector: Any, train: Mapping[str, Any], *,
                               prefix: str = "") -> list[dict[str, Any]]:
    """Route every projector tensor exactly once: 2D Muon8bit, other FP32 AdamW.

    ``prefix`` supplies the owning module path (``vision_projector.``) so the
    optimizer route matches the adapter's global parameter inventory.
    """
    muon, adamw = [], []
    for name, parameter in projector.named_parameters():
        qualified = prefix + name
        item = {"name": qualified, "parameter": parameter, "shape": tuple(parameter.shape)}
        if parameter.ndim == 2:
            muon.append(item)
        else:
            adamw.append(item)
    require(muon and adamw,
            "Projector must expose both hidden 2D and bias/norm tensors")
    owned = [item["parameter"] for item in muon + adamw]
    require(len({id(parameter) for parameter in owned}) == len(owned),
            "Projector optimizer ownership is duplicated")
    require({id(parameter) for parameter in owned}
            == {id(parameter) for parameter in projector.parameters() if parameter.requires_grad},
            "Projector optimizer ownership is incomplete")
    lr = float(train["lr_new"])
    common = {"scope": "vision_projector", "lr": lr,
              "weight_decay": float(train["weight_decay"])}
    return [
        {"params": [item["parameter"] for item in muon],
         "param_names": [item["name"] for item in muon],
         "group_name": "vision_projector_muon8bit", "algorithm": "muon8bit",
         "momentum": float(train["momentum"]), "ns_steps": int(train["ns_steps"]),
         "block_size": int(train["block_size"]), **common},
        {"params": [item["parameter"] for item in adamw],
         "param_names": [item["name"] for item in adamw],
         "group_name": "vision_projector_adamw", "algorithm": "adamw",
         "betas": (0.9, 0.999), "eps": 1e-8, **common},
    ]


def optimizer_route(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    route = []
    for group in groups:
        for name, parameter in zip(group["param_names"], group["params"]):
            route.append({"name": name, "shape": list(parameter.shape),
                          "algorithm": group["algorithm"],
                          "group_name": group["group_name"]})
    require(len({item["name"] for item in route}) == len(route),
            "Optimizer route names repeat")
    return sorted(route, key=lambda item: item["name"])


def validate_optimizer_route(route: Sequence[Mapping[str, Any]],
                             audit: Mapping[str, Any]) -> None:
    require(route, "Optimizer route is empty")
    algorithms = {item["algorithm"] for item in route}
    require(algorithms == {"muon8bit", "adamw"},
            "Projector optimizer must route both Muon8bit and AdamW groups")
    for item in route:
        require(item["name"].startswith("vision_projector."),
                f"Non-projector optimizer owner: {item['name']}")
        require(len(item["shape"]) in (1, 2), "Unexpected projector tensor rank")
        expected = "muon8bit" if len(item["shape"]) == 2 else "adamw"
        require(item["algorithm"] == expected,
                f"Projector tensor routed to the wrong algorithm: {item['name']}")
    require(audit.get("optimizer") == "muon8bit", "Optimizer audit name differs")
    require(audit.get("parameter_dtype") == "float32"
            and audit.get("muon_temporary_dtype") == "float32",
            "Optimizer audit must require FP32 parameters/temporary Muon arithmetic")
    audited = {name for group in audit.get("groups", ())
               for name in group.get("param_names", ())}
    require(audited == {item["name"] for item in route},
            "Optimizer audit parameter inventory differs from the route")
    by_dtype = audit.get("state_bytes_by_dtype", {})
    require(set(by_dtype) <= {"torch.int8", "torch.float32"},
            f"Optimizer allocated undeclared persistent state dtype: {sorted(by_dtype)}")
    require("torch.int8" in by_dtype and "torch.float32" in by_dtype,
            "Audited Muon8bit int8 momentum and FP32 AdamW state were not both allocated")


# ---------------------------------------------------------------------------
# Run-report checks shared by the independent scorer


def require_completed_run(run: str | Path, protocol: Mapping[str, Any],
                          protocol_sha256: str) -> tuple[dict[str, Any], str]:
    run = Path(run)
    report_path = run / "run_report.json"
    require(report_path.is_file(), "Run report is missing")
    require(not (run / "failure_report.json").exists(),
            "Run preserved a failure; failed runs cannot be accepted")
    digest = sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("status") == "completed", "Run status is not completed")
    require(report.get("format") == PROTOCOL_FORMAT
            or report.get("format") == "replica.vision_qualification_run.v1",
            "Run report format differs")
    require(report.get("stage") == protocol["stage"], "Run stage differs from protocol")
    require(report.get("protocol_sha256") == protocol_sha256,
            "Run used another protocol")
    return report, digest


def require_same_implementation(report: Mapping[str, Any],
                                protocol: Mapping[str, Any]) -> None:
    recorded = report.get("implementation_sha256")
    expected = protocol["implementation_sha256"]
    require(recorded == expected,
            "Run implementation pins differ from the protocol")


def validate_restart_report(restart: str | Path, protocol: Mapping[str, Any],
                            protocol_sha256: str, prior_report_sha256: str
                            ) -> tuple[dict[str, Any], str]:
    restart = Path(restart)
    path = restart / "restart_report.json"
    require(path.is_file(), "Fresh-process restart report is missing")
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("status") == "completed", "Restart did not complete")
    require(report.get("protocol_sha256") == protocol_sha256,
            "Restart used another protocol")
    require(report.get("prior_run_report_sha256") == prior_report_sha256,
            "Restart did not compare against this primary run")
    require(report.get("zero_updates") is True,
            "Restart performed an optimizer update")
    require(report.get("loaded_extension", {}).get("weights_sha256")
            == report.get("prior_extension_weights_sha256"),
            "Restart extension lineage differs")
    conditions = report.get("conditions")
    require(isinstance(conditions, list) and conditions, "Restart has no conditions")
    require(report.get("loaded_extension", {}).get("path_sha256") is not None
            or report.get("loaded_extension", {}).get("weights_sha256") is not None,
            "Restart lacks a loaded-extension receipt")
    for condition in conditions:
        _subset_keys(condition, ("condition", "prior_logits_sha256",
                                 "restart_logits_sha256", "tokens_equal",
                                 "logits_equal", "maximum_absolute_difference",
                                 "tolerance"), "restart condition")
        require(condition["tokens_equal"] is True and condition["logits_equal"] is True,
                f"Restart changed model behavior: {condition['condition']}")
        prior_logits = restart / "prior_logits" / (condition["condition"] + ".safetensors")
        new_logits = restart / "restart_logits" / (condition["condition"] + ".safetensors")
        require(sha256_file(prior_logits) == condition["prior_logits_sha256"],
                f"Restart prior logits changed: {condition['condition']}")
        require(sha256_file(new_logits) == condition["restart_logits_sha256"],
                f"Restart recomputed logits changed: {condition['condition']}")
        from safetensors.torch import load_file
        import torch
        left = load_file(str(prior_logits))["logits"].float()
        right = load_file(str(new_logits))["logits"].float()
        require(tuple(left.shape) == tuple(right.shape),
                f"Restart logits shapes differ: {condition['condition']}")
        atol = _number(condition["tolerance"]["atol"], "restart tolerance.atol",
                       minimum=0.0)
        rtol = _number(condition["tolerance"]["rtol"], "restart tolerance.rtol",
                       minimum=0.0)
        require(bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
                f"Restart logits differ beyond tolerance: {condition['condition']}")
        maximum = float((left - right).abs().max().item()) if left.numel() else 0.0
        require(math.isclose(maximum, float(condition["maximum_absolute_difference"]),
                             rel_tol=0.0, abs_tol=1e-6),
                f"Restart maximum difference receipt differs: {condition['condition']}")
    return report, digest


__all__ = [name for name in globals() if not name.startswith("_")]
