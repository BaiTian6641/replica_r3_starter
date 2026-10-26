"""Staged MiniCPM5 projector curriculum with sealed procedural audit data.

This module owns a *new* curriculum on top of two already-admitted sources:

* the bounded external caption tranche (15 train / 7 development / 8 sealed
  final rows), used only for Stage A caption alignment;
* the balanced paired procedural diagnostic, whose ``train`` split is Stage B
  instruction data and whose already-scored ``final`` split is reused only as
  development evidence.

Because the diagnostic ``final`` split is now development evidence, this module
generates a genuinely fresh procedural corpus with its own worlds, layouts and
object descriptors. The new ``audit`` split is disjoint from the diagnostic and
from every curriculum train/dev object descriptor, carries its own manifest and
seal, and is scored once after a checkpoint is frozen. The external final split
is never consumed here: it is pinned as a sealed pointer only.

Stage B includes explicit unknown/abstention instruction. Every abstention case
has a verifier label (``unknown``), a machine-checkable reason and a training
example in the train split. Questions and sensing inputs never contain gold
labels; the model rows carry only the image path/hash and the question.

This is data, protocol and scoring infrastructure. It makes no perception,
semantic-quality or baseline-eligibility claim. The declared 22/32 text
retention baseline is reported separately from component non-regression.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import os
import random
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_json, fingerprint
from .external_intake import _role_manifest_seal
from .visual_data import (
    COLOR_NAMES,
    SHAPES,
    VisualDataError,
    render_scene_png,
    scene_sha256 as scene_digest,
    write_exact as _visual_write_exact,
)


VERSION = "replica.vision_curriculum.v1"
CONFIG_FORMAT = "replica.vision_curriculum_config.v1"
PROCEDURAL_MANIFEST_FORMAT = "replica.vision_curriculum_procedural_manifest.v1"
CURRICULUM_MANIFEST_FORMAT = "replica.vision_curriculum_manifest.v1"
SEALS_FORMAT = "replica.vision_curriculum_seals.v1"
PROTOCOL_FORMAT = "replica.vision_curriculum.protocol.v1"
SCHEDULE_FORMAT = "replica.vision_curriculum_schedule.v1"
ARMS_FORMAT = "replica.vision_curriculum_arms.v1"
RUN_REPORT_FORMAT = "replica.vision_curriculum.run_report.v1"
RELOAD_REPORT_FORMAT = "replica.vision_curriculum.reload_report.v1"
SCORE_REPORT_FORMAT = "replica.vision_curriculum.score_report.v1"
FINALIZE_RECEIPT_FORMAT = "replica.vision_curriculum.finalize_receipt.v1"

SPLITS = ("train", "dev", "audit")
ALL_TASKS = ("spatial", "count", "presence", "color", "shape", "abstain")
SPLIT_TASKS = {
    "train": ALL_TASKS,
    "dev": ("spatial", "count", "presence", "color", "abstain"),
    "audit": ("spatial", "count", "presence", "color", "abstain"),
}
ABSTAIN_KINDS = ("absent_referent", "ambiguous_referent", "occluded_relation")
UNKNOWN_ANSWER = "unknown"
UNKNOWN_ANSWERS = frozenset({"unknown", "i don't know", "i dont know",
                             "cannot determine", "can't determine",
                             "not determinable", "undetermined",
                             "indeterminate"})
CONTROLS = ("correct", "wrong", "shuffled", "noimage")
ARMS = ("muon8bit", "adamw_control")
QUESTION_TEMPLATE = "Question: {question}\nReply briefly with the answer alone."
MODEL_ROW_FIELDS = {"case_id", "pair_id", "asset_id", "image", "image_sha256",
                    "question"}
LOCAL_MODEL_ROW_FIELDS = {"case_id", "image", "image_sha256", "question"}
GENERATION_FIELDS = {"control", "case_id", "generated_text"}
FORBIDDEN_CLAIM_FIELDS = frozenset({"accepted", "success", "promotion",
                                    "verdict", "semantic_quality"})

GRID_FRACTIONS_X = (0.22, 0.42, 0.62, 0.82)
GRID_FRACTIONS_Y = (0.38, 0.68)
GRID_LAYOUTS = {
    "row_major": (0, 1, 2, 3, 4, 5, 6, 7),
    "row_major_shift": (2, 3, 4, 5, 6, 7, 0, 1),
    "column_major": (0, 4, 1, 5, 2, 6, 3, 7),
    "column_major_shift": (3, 7, 2, 6, 1, 5, 0, 4),
}
DIAGNOSTIC_FINAL_DESCRIPTORS = frozenset({
    ("green", "square"), ("yellow", "square"), ("purple", "square"),
    ("orange", "square"), ("green", "rectangle"), ("orange", "rectangle"),
})
PROMPT_PREFIXES = {
    "train": ("", "Image question: ", "Visual question: ", "Picture question: "),
    "dev": ("Scene question: ", "Dev image question: ", "Picture check: ",
            "Scene check: "),
    "audit": ("Audit image question: ", "Audit scene: ", "Held-out scene: ",
              "Final image question: "),
}
PROMPT_SUFFIXES = ("", " Answer with one word.", " Reply with the answer only.")

AUDIT_FORBIDDEN_V2_DESCRIPTORS = frozenset({
    ("red", "circle"), ("blue", "circle"), ("green", "circle"),
    ("yellow", "circle"), ("purple", "circle"), ("orange", "circle"),
    ("red", "square"), ("red", "triangle"), ("red", "rectangle"),
    ("blue", "triangle"), ("green", "triangle"), ("yellow", "triangle"),
    ("purple", "triangle"), ("blue", "square"), ("blue", "rectangle"),
    ("green", "square"), ("yellow", "square"), ("purple", "square"),
    ("orange", "square"), ("green", "rectangle"), ("orange", "rectangle"),
})

PROJECTOR_HIDDEN = 4096
PROJECTOR_TENSORS = (
    "vision_projector.0.weight", "vision_projector.0.bias",
    "vision_projector.1.weight", "vision_projector.1.bias",
    "vision_projector.3.weight", "vision_projector.3.bias",
)
PROJECTOR_2D_TENSORS = ("vision_projector.1.weight", "vision_projector.3.weight")
PROJECTOR_1D_TENSORS = ("vision_projector.0.weight", "vision_projector.0.bias",
                        "vision_projector.1.bias", "vision_projector.3.bias")
PROJECTOR_PARAMETER_COUNT = 11_542_016
RUN_ARTIFACTS = (
    "training_trace.jsonl", "lr_probe.json", "backward_probe.json",
    "reload_report.json",
    "dev_cases.jsonl", "dev_generations.jsonl", "caption_cases.jsonl",
    "caption_labels.jsonl", "caption_generations.jsonl",
    "retention_pre.jsonl", "retention_post.jsonl",
    "extension/manifest.json", "extension/extensions.safetensors",
)


def lr_probe_artifact_name(lr: float) -> str:
    return f"lr_probe/probe_{format(float(lr), '.0e')}.jsonl"


def run_artifact_names(protocol: Mapping[str, Any], *,
                       include_reload: bool = True) -> tuple[str, ...]:
    candidates = protocol["arms"]["arms"]["muon8bit"]["lr_probe_candidates"]
    names = list(RUN_ARTIFACTS)
    if not include_reload:
        names = [name for name in names if name != "reload_report.json"]
    for lr in candidates:
        name = lr_probe_artifact_name(lr)
        if name not in names:
            names.append(name)
    return tuple(sorted(names))
PROCEDURAL_PARAMETERS_SANITY = {
    "projector_hidden": PROJECTOR_HIDDEN,
    "projector_parameter_count": PROJECTOR_PARAMETER_COUNT,
    "projector_tensors": len(PROJECTOR_TENSORS),
    "stage_a_updates": 16,
    "stage_b_updates": 112,
    "updates_per_arm": 128,
    "examples_per_arm": 512,
    "probe_updates_per_arm": 32,
    "probe_examples_per_arm": 64,
    "arms": list(ARMS),
    "retention_baseline": "22/32",
}

IMPLEMENTATION_FILES = (
    "src/replica_lab/vision_curriculum.py",
    "src/replica_lab/vision_dependence.py",
    "src/replica_lab/vision_qualification.py",
    "src/replica_lab/visual_data.py",
    "src/replica_lab/external_intake.py",
    "src/replica_lab/contracts.py",
    "src/replica_lab/optimizers.py",
    "src/replica_lab/capacity_guard.py",
    "src/replica_lab/model_adapter.py",
    "src/replica_lab/neural.py",
    "src/replica_lab/architecture/__init__.py",
    "src/replica_lab/architecture/vision.py",
    "src/replica_lab/architecture/loading.py",
    "src/replica_lab/architecture/minicpm5.py",
    "src/replica_lab/architecture/contracts.py",
    "src/replica_lab/architecture/ports.py",
    "src/replica_lab/architecture/heads.py",
    "src/replica_lab/architecture/core.py",
    "src/replica_lab/memory_system/__init__.py",
    "src/replica_lab/memory_system/session.py",
    "src/replica_lab/memory_system/ledger.py",
    "src/replica_lab/memory_system/index.py",
    "src/replica_lab/memory_system/writer.py",
    "src/replica_lab/replica/reader.py",
    "configs/vision_curriculum_v1.json",
    "scripts/run_capacity_preflight.py",
    "scripts/prepare_vision_curriculum.py",
    "scripts/run_vision_curriculum.py",
)


class VisionCurriculumError(ValueError):
    """Raised when curriculum data, protocol, arms or scoring drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisionCurriculumError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(value) + "\n").encode("utf-8")
                    for value in values)


def _write_exact(path: Path, data: bytes) -> None:
    try:
        _visual_write_exact(path, data)
    except VisualDataError as error:
        raise VisionCurriculumError(str(error)) from error
    _require(Path(path).read_bytes() == data,
             f"Completed artifact bytes differ: {path}")
    _fsync_parent(Path(path))


def _safe_relative(root: Path, value: str) -> Path:
    _require(isinstance(value, str) and value and not Path(value).is_absolute()
             and ".." not in Path(value).parts,
             f"Unsafe relative path: {value!r}")
    return Path(root) / value


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0,
             f"{name} must be a nonnegative integer")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(float(value)),
             f"{name} must be a finite number")
    value = float(value)
    if minimum is not None:
        _require(value >= minimum, f"{name} must be >= {minimum}")
    if maximum is not None:
        _require(value <= maximum, f"{name} must be <= {maximum}")
    return value


def _exact_keys(value: Any, expected: Iterable[str], name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    expected = set(expected)
    _require(set(value) == expected,
             f"{name} fields differ: missing={sorted(expected - set(value))} "
             f"extra={sorted(set(value) - expected)}")
    return value


def _nonempty(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{name} must be a nonempty string")
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


CONFIG_FIELDS = {
    "format", "version", "seed", "data_root", "purpose", "external",
    "procedural_source", "procedural", "curriculum", "train", "arms",
    "lr_probe", "evaluation", "retention", "checkpoint_policy", "limits",
    "resource_policy", "action_limits", "stop_conditions", "vision_scope",
}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(config, CONFIG_FIELDS, "config")
    _require(config["format"] == CONFIG_FORMAT, "Unknown curriculum config format")
    _require(config["version"] == VERSION, "Unknown curriculum version")
    _nonnegative_int(config["seed"], "seed")
    root = config["data_root"]
    _require(isinstance(root, str) and root and not Path(root).is_absolute()
             and ".." not in Path(root).parts,
             "data_root must be a safe relative path")
    _nonempty(config["purpose"], "purpose")

    scope = _exact_keys(config["vision_scope"], {
        "subject_id", "branch_id", "purpose", "as_of", "mode",
        "snapshot_at"}, "vision_scope")
    for name in ("subject_id", "branch_id", "purpose"):
        _nonempty(scope[name], f"vision_scope.{name}")
    _nonnegative_int(scope["as_of"], "vision_scope.as_of")
    _require(scope["mode"] == "parallel_replica",
             "Vision scope mode differs")
    _require(scope["snapshot_at"] is None, "Vision scope snapshot is open")

    external = _exact_keys(config["external"], {
        "root", "intake_manifest", "corpus", "manifests", "image_root",
        "prompt_mode", "train_rows", "development_rows", "final_rows",
        "final_policy"}, "external")
    _nonempty(external["root"], "external.root")
    _nonempty(external["intake_manifest"], "external.intake_manifest")
    _nonempty(external["image_root"], "external.image_root")
    _require(external["prompt_mode"] == "upstream_prompt",
             "External prompt mode must be the declared upstream prompt")
    _require(external["final_policy"] == "sealed_unconsumed_pointer_only",
             "External final split must stay a sealed pointer")
    for name in ("train_rows", "development_rows", "final_rows"):
        _positive_int(external[name], f"external.{name}")
    _require(external["train_rows"] == 15
             and external["development_rows"] == 7
             and external["final_rows"] == 8,
             "External split row counts differ from the admitted tranche")
    corpus = _exact_keys(external["corpus"], {"path", "sha256"},
                         "external.corpus")
    _nonempty(corpus["path"], "external.corpus.path")
    _require(_is_sha256(corpus["sha256"]), "External corpus hash is invalid")
    manifests = _exact_keys(external["manifests"],
                            {"train", "development", "final_eval_untouched"},
                            "external.manifests")
    for name in ("train", "development"):
        entry = _exact_keys(manifests[name], {"path", "rows", "sha256"},
                            f"external.manifests.{name}")
        _nonempty(entry["path"], f"external.manifests.{name}.path")
        _positive_int(entry["rows"], f"external.manifests.{name}.rows")
        _require(_is_sha256(entry["sha256"]),
                 f"external.manifests.{name}.sha256 is invalid")
    final = _exact_keys(manifests["final_eval_untouched"],
                        {"path", "rows", "sha256", "seal_sha256"},
                        "external.manifests.final_eval_untouched")
    _nonempty(final["path"], "external final manifest path")
    _require(_is_sha256(final["sha256"]) and _is_sha256(final["seal_sha256"]),
             "External final manifest pins are invalid")
    _require(manifests["train"]["rows"] == external["train_rows"]
             and manifests["development"]["rows"] == external["development_rows"]
             and final["rows"] == external["final_rows"],
             "External manifest row counts differ from declared counts")

    source = _exact_keys(config["procedural_source"], {
        "root", "protocol_sha256", "manifest_sha256", "model_rows_sha256",
        "verifier_cases_sha256", "train_split", "development_split",
        "development_note"}, "procedural_source")
    _nonempty(source["root"], "procedural_source.root")
    for name in ("protocol_sha256", "manifest_sha256", "model_rows_sha256",
                 "verifier_cases_sha256"):
        _require(_is_sha256(source[name]),
                 f"procedural_source.{name} is invalid")
    _require(source["train_split"] == "train",
             "Diagnostic train split must be 'train'")
    _require(source["development_split"] == "final",
             "Diagnostic final split is development evidence")
    _nonempty(source["development_note"], "procedural_source.development_note")

    procedural = _exact_keys(config["procedural"], {
        "renderer", "worlds_per_task", "tasks", "count_labels", "splits",
        "descriptor_pools", "split_layouts", "abstain_kinds",
        "unknown_label"}, "procedural")
    renderer = _exact_keys(procedural["renderer"], {
        "version", "width", "height", "image_format",
        "standard_library_only"}, "procedural.renderer")
    _require(renderer["version"] == "procedural_png_v1",
             "Unknown renderer version")
    _require(type(renderer["width"]) is int and type(renderer["height"]) is int
             and renderer["width"] >= 64 and renderer["height"] >= 48,
             "Renderer size is invalid")
    _require(renderer["image_format"] == "png_rgb8"
             and renderer["standard_library_only"] is True,
             "Renderer must be standard-library PNG RGB8")
    _positive_int(procedural["worlds_per_task"], "procedural.worlds_per_task")
    _require(procedural["worlds_per_task"] >= 8,
             "At least 8 worlds per task/split are required")
    _require(list(procedural["tasks"]) == list(ALL_TASKS),
             "Procedural task inventory differs")
    _require(list(procedural["count_labels"]) == ["2", "3"],
             "Only the declared count labels 2 and 3 are allowed")
    _require(list(procedural["splits"]) == list(SPLITS),
             "Procedural splits differ")
    _require(list(procedural["abstain_kinds"]) == list(ABSTAIN_KINDS),
             "Abstention kinds differ")
    _require(procedural["unknown_label"] == UNKNOWN_ANSWER,
             "Unknown label must be 'unknown'")
    pools = _exact_keys(procedural["descriptor_pools"], set(SPLITS),
                        "procedural.descriptor_pools")
    seen: dict[str, set[tuple[str, str]]] = {}
    for split in SPLITS:
        descriptors = pools[split]
        _require(isinstance(descriptors, list) and len(descriptors) >= 3,
                 f"Descriptor pool for {split} needs at least three pairs")
        parsed: set[tuple[str, str]] = set()
        for item in descriptors:
            _require(isinstance(item, list) and len(item) == 2,
                     "Descriptors must be [color, shape] pairs")
            color, shape = item
            _require(color in COLOR_NAMES, f"Unknown descriptor color: {color}")
            _require(shape in SHAPES, f"Unknown descriptor shape: {shape}")
            _require((color, shape) not in parsed,
                     f"Duplicate descriptor in {split}: {color}/{shape}")
            parsed.add((color, shape))
        seen[split] = parsed
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(seen[left].isdisjoint(seen[right]),
                         f"Descriptor pools overlap across {left}/{right}")
    _require(seen["audit"].isdisjoint(AUDIT_FORBIDDEN_V2_DESCRIPTORS),
             "Audit descriptors overlap the admitted diagnostic corpus")
    _require(seen["train"].isdisjoint(DIAGNOSTIC_FINAL_DESCRIPTORS),
             "Curriculum train descriptors overlap the diagnostic final "
             "split that is now development evidence")
    layouts = _exact_keys(procedural["split_layouts"], set(SPLITS),
                          "procedural.split_layouts")
    for split in SPLITS:
        _require(isinstance(layouts[split], list) and layouts[split]
                 and all(item in GRID_LAYOUTS for item in layouts[split]),
                 f"Invalid layouts for {split}")

    curriculum = _exact_keys(config["curriculum"], {
        "question_template", "stage_a", "stage_b", "total_updates",
        "total_examples", "max_response_tokens", "schedule_version",
        "selection"}, "curriculum")
    _require(curriculum["question_template"] == QUESTION_TEMPLATE,
             "Curriculum question template differs from the pinned template")
    stage_a = _exact_keys(curriculum["stage_a"],
                          {"updates", "batch", "source", "policy"},
                          "curriculum.stage_a")
    stage_b = _exact_keys(curriculum["stage_b"],
                          {"updates", "batch", "sources", "policy"},
                          "curriculum.stage_b")
    for name, stage in (("stage_a", stage_a), ("stage_b", stage_b)):
        _positive_int(stage["updates"], f"curriculum.{name}.updates")
        _positive_int(stage["batch"], f"curriculum.{name}.batch")
        _require(stage["policy"] == "deterministic_epoch_cycle",
                 f"curriculum.{name}.policy differs")
    _require(stage_a["source"] == "external_train",
             "Stage A must use only admitted external train rows")
    _require(list(stage_b["sources"]) == ["procedural_source_train",
                                          "procedural_train"],
             "Stage B sources differ")
    _require(curriculum["total_updates"] == stage_a["updates"] + stage_b["updates"],
             "Declared total updates differ from the staged schedule")
    _require(curriculum["total_examples"]
             == (stage_a["updates"] * stage_a["batch"]
                 + stage_b["updates"] * stage_b["batch"]),
             "Declared total examples differ from the staged schedule")
    _positive_int(curriculum["max_response_tokens"],
                  "curriculum.max_response_tokens")
    _require(curriculum["schedule_version"] == "curriculum_two_stage.v1",
             "Unknown curriculum schedule version")
    selection = _exact_keys(curriculum["selection"], {
        "development_splits", "selection_split", "external_development_role",
        "probe_cases", "arm_selection"}, "curriculum.selection")
    _require(list(selection["development_splits"])
             == ["procedural_dev", "external_development"],
             "Development sources differ")
    _require(selection["selection_split"] == "procedural_dev",
             "Model selection must use procedural dev only")
    _require(selection["external_development_role"]
             == "narrow_caption_diagnostic_only",
             "External development must remain a narrow diagnostic")
    _positive_int(selection["probe_cases"], "curriculum.selection.probe_cases")
    arm_selection = _exact_keys(selection["arm_selection"], {
        "measure", "tie_break", "both_arms_required"},
        "curriculum.selection.arm_selection")
    _require(arm_selection["measure"] == "mean_dev_response_loss"
             and arm_selection["tie_break"] == "declared_arm_order"
             and arm_selection["both_arms_required"] is True,
             "Arm selection rule differs")

    train = _exact_keys(config["train"], {
        "seed", "init_seed", "init_recipe", "optimizer_order",
        "max_grad_norm", "max_seconds", "response_only_labels", "warm_start",
        "forbidden_parent_weights_sha256"}, "train")
    _nonnegative_int(train["seed"], "train.seed")
    _nonnegative_int(train["init_seed"], "train.init_seed")
    _require(train["init_recipe"] == "fresh_seeded_normal_std0.02",
             "Fresh initialization recipe differs")
    _require(list(train["optimizer_order"]) == list(ARMS), "Arm order differs")
    _finite(train["max_grad_norm"], "train.max_grad_norm", minimum=0.0,
            maximum=1000.0)
    _positive_int(train["max_seconds"], "train.max_seconds")
    _require(train["response_only_labels"] is True,
             "Response-only labels are required")
    _require(train["warm_start"] is None, "Warm starts are forbidden")
    forbidden = train["forbidden_parent_weights_sha256"]
    _require(isinstance(forbidden, list)
             and bool(forbidden)
             and all(_is_sha256(item) for item in forbidden),
             "Forbidden parent weight hashes are invalid")

    arms = _exact_keys(config["arms"], {*ARMS, "equal_compute"}, "arms")
    muon = _exact_keys(arms["muon8bit"], {
        "route", "matrix_lr", "aux_lr", "momentum", "ns_steps", "block_size",
        "weight_decay", "betas", "eps"}, "arms.muon8bit")
    _require(muon["route"] == "muon8bit_matrix_plus_fp32_adamw_aux",
             "Muon arm route differs")
    for name in ("matrix_lr", "aux_lr"):
        _require(0.0 < _finite(muon[name], f"arms.muon8bit.{name}") <= 1.0,
                 f"arms.muon8bit.{name} is outside (0, 1]")
    _finite(muon["momentum"], "arms.muon8bit.momentum", minimum=0.0,
            maximum=0.999999)
    _positive_int(muon["ns_steps"], "arms.muon8bit.ns_steps")
    _require(muon["ns_steps"] < 100, "Muon NS steps must be < 100")
    _positive_int(muon["block_size"], "arms.muon8bit.block_size")
    _finite(muon["weight_decay"], "arms.muon8bit.weight_decay", minimum=0.0)
    _require(list(muon["betas"]) == [0.9, 0.999], "AdamW betas differ")
    _require(0.0 < _finite(muon["eps"], "arms.muon8bit.eps") <= 1.0,
             "AdamW eps is invalid")
    adamw = _exact_keys(arms["adamw_control"], {
        "route", "declared_lr", "lr_candidates", "weight_decay", "betas",
        "eps"}, "arms.adamw_control")
    _require(adamw["route"] == "all_fp32_adamw", "AdamW control route differs")
    _require(0.0 < _finite(adamw["declared_lr"],
                           "arms.adamw_control.declared_lr") <= 1.0,
             "AdamW control LR is outside (0, 1]")
    candidates = adamw["lr_candidates"]
    _require(isinstance(candidates, list) and len(candidates) >= 2
             and all(0.0 < _finite(item, "arm candidate") <= 1.0
                     for item in candidates)
             and len(set(candidates)) == len(candidates),
             "AdamW control LR candidates are invalid")
    _require(float(adamw["declared_lr"]) in [float(item) for item in candidates],
             "Declared AdamW LR must be a declared candidate")
    _finite(adamw["weight_decay"], "arms.adamw_control.weight_decay",
            minimum=0.0)
    _require(list(adamw["betas"]) == [0.9, 0.999], "AdamW control betas differ")
    _require(0.0 < _finite(adamw["eps"], "arms.adamw_control.eps") <= 1.0,
             "AdamW control eps is invalid")
    equal = _exact_keys(arms["equal_compute"], {
        "identical_examples", "identical_updates", "identical_schedule",
        "identical_probe_budget", "identical_initialization"},
        "arms.equal_compute")
    _require(all(value is True for value in equal.values()),
             "Equal-compute declarations must all be true")

    probe = _exact_keys(config["lr_probe"], {
        "updates", "batch", "split", "measure", "tie_break", "candidates"},
        "lr_probe")
    _positive_int(probe["updates"], "lr_probe.updates")
    _positive_int(probe["batch"], "lr_probe.batch")
    _require(probe["split"] == "procedural_dev", "LR probe must use dev only")
    _require(probe["measure"] == "mean_dev_response_loss",
             "LR probe measure differs")
    _require(probe["tie_break"] == "lowest_lr", "LR probe tie-break differs")
    probe_candidates = _exact_keys(probe["candidates"], set(ARMS),
                                   "lr_probe.candidates")
    for arm in ARMS:
        values = probe_candidates[arm]
        _require(isinstance(values, list) and len(values) == 2
                 and all(0.0 < _finite(item, f"probe {arm}") <= 1.0
                         for item in values),
                 f"LR probe candidates for {arm} are invalid")
    _require([float(v) for v in probe_candidates["adamw_control"]]
             == [float(v) for v in candidates],
             "Probe and control LR candidates differ")
    _require([float(v) for v in probe_candidates["muon8bit"]]
             == [float(v) for v in probe_candidates["adamw_control"]],
             "Both arms must share the symmetric probe candidate set")

    evaluation = _exact_keys(config["evaluation"], {
        "controls", "max_new_tokens", "caption_max_new_tokens", "greedy",
        "do_sample", "beam_width", "batch", "wrong_control",
        "shuffle_policy", "shuffle_seed_offset", "bootstrap",
        "caption_metric", "gates"}, "evaluation")
    _require(list(evaluation["controls"]) == list(CONTROLS),
             "Evaluation controls differ")
    _positive_int(evaluation["max_new_tokens"], "evaluation.max_new_tokens")
    _positive_int(evaluation["caption_max_new_tokens"],
                  "evaluation.caption_max_new_tokens")
    _require(evaluation["caption_max_new_tokens"]
             <= curriculum["max_response_tokens"],
             "Caption generation exceeds the response budget")
    _require(evaluation["greedy"] is True
             and evaluation["do_sample"] is False
             and evaluation["beam_width"] == 1
             and evaluation["batch"] == 1,
             "Generation must be greedy, single-beam and batch-1")
    _require(evaluation["wrong_control"] == "different_world_same_task",
             "Wrong-image control policy differs")
    _require(evaluation["shuffle_policy"]
             == "deterministic_derangement_within_split",
             "Shuffle policy differs")
    _positive_int(evaluation["shuffle_seed_offset"],
                  "evaluation.shuffle_seed_offset")
    bootstrap = _exact_keys(evaluation["bootstrap"],
                            {"cluster", "resamples", "confidence"},
                            "evaluation.bootstrap")
    _require(bootstrap["cluster"] == "world", "Bootstrap must cluster by world")
    _positive_int(bootstrap["resamples"], "evaluation.bootstrap.resamples")
    _finite(bootstrap["confidence"], "evaluation.bootstrap.confidence",
            minimum=0.5, maximum=0.999999)
    caption = _exact_keys(evaluation["caption_metric"], {
        "name", "normalization", "teacher_forced_loss", "claim"},
        "evaluation.caption_metric")
    _require(caption["name"] == "normalized_token_f1",
             "Caption metric differs")
    _require(caption["normalization"]
             == "casefold_whitespace_strip_punctuation.v1",
             "Caption normalization differs")
    _require(caption["teacher_forced_loss"] == "response_mean_nll",
             "Teacher-forced loss name differs")
    _require(caption["claim"]
             == "narrow_alignment_metric_only_no_semantic_quality_claim",
             "Caption claim boundary differs")
    gates = evaluation["gates"]
    required_gates = {"min_correct_accuracy", "min_unique_families",
                      "max_unknown_rate", "min_correct_margin_wrong",
                      "min_correct_margin_shuffled",
                      "min_correct_margin_noimage", "min_unknown_accuracy",
                      "max_false_answer_on_abstain",
                      "max_false_abstention_on_answerable",
                      "min_text_retention_accuracy",
                      "max_text_retention_drop"}
    _exact_keys(gates, required_gates, "evaluation.gates")
    for name in required_gates:
        _finite(gates[name], f"evaluation.gates.{name}")
    _require(0.0 <= gates["min_text_retention_accuracy"] <= 1.0,
             "Retention capability floor is invalid")

    retention = _exact_keys(config["retention"], {
        "policy", "baseline", "require_token_parity", "prompts", "labels",
        "max_cases", "role"}, "retention")
    _require(retention["policy"] == "component_nonregression.v1",
             "Retention policy differs")
    baseline = _exact_keys(retention["baseline"], {
        "correct", "cases", "accuracy", "source"}, "retention.baseline")
    _require(baseline["correct"] == 22 and baseline["cases"] == 32,
             "Declared retention baseline must stay 22/32")
    _require(float(baseline["accuracy"]) == 22 / 32,
             "Declared retention baseline accuracy differs")
    _nonempty(baseline["source"], "retention.baseline.source")
    _require(retention["require_token_parity"] is True,
             "Retention token parity is required")
    for name in ("prompts", "labels"):
        pin = _exact_keys(retention[name], {"path", "sha256"},
                          f"retention.{name}")
        _nonempty(pin["path"], f"retention.{name}.path")
        _require(_is_sha256(pin["sha256"]),
                 f"retention.{name}.sha256 is invalid")
    _positive_int(retention["max_cases"], "retention.max_cases")
    _require(retention["max_cases"] == 32, "Retention case count must be 32")
    _nonempty(retention["role"], "retention.role")

    checkpoint = _exact_keys(config["checkpoint_policy"], {
        "policy", "probe_checkpoints", "base_copies", "optimizer_state_saved",
        "export", "reload", "selection_freeze", "audit_scoring", "keep_arms"},
        "checkpoint_policy")
    _require(checkpoint["policy"] == "one_final_extension_per_arm"
             and checkpoint["probe_checkpoints"] is False
             and checkpoint["base_copies"] == 0
             and checkpoint["optimizer_state_saved"] is False,
             "Checkpoint policy differs")
    _require(checkpoint["export"] == "strict_extension_v4"
             and checkpoint["reload"] == "zero_update_fresh_process_bitwise"
             and checkpoint["selection_freeze"]
             == "freeze_selected_checkpoint_before_audit"
             and checkpoint["audit_scoring"] == "once",
             "Checkpoint/reload policy differs")
    _require(list(checkpoint["keep_arms"]) == list(ARMS),
             "Retained arms differ")

    limits = _exact_keys(config["limits"], {
        "max_prompt_tokens", "max_response_tokens", "max_sequence_tokens",
        "max_new_tokens", "max_eval_cases", "max_rows", "max_updates",
        "max_probe_updates", "max_seconds", "max_export_bytes",
        "min_free_disk_bytes", "max_cache_bytes", "max_driver_bytes",
        "max_bootstrap_resamples"}, "limits")
    for name, value in limits.items():
        _positive_int(value, f"limits.{name}")
    _require(limits["max_response_tokens"] == curriculum["max_response_tokens"],
             "Response token limit differs from the curriculum")
    _require(limits["max_new_tokens"] <= limits["max_response_tokens"],
             "Generation limit exceeds response limit")
    _require(limits["max_sequence_tokens"]
             <= limits["max_prompt_tokens"] + limits["max_response_tokens"],
             "Sequence limit exceeds prompt+response budget")
    probe_updates = (len(ARMS) * len(probe["candidates"]["muon8bit"])
                     * probe["updates"])
    _require(limits["max_updates"]
             >= len(ARMS) * curriculum["total_updates"] + probe_updates,
             "max_updates does not cover both arms and the LR probe")
    _require(limits["max_probe_updates"] >= probe_updates,
             "max_probe_updates is too small")
    _require(limits["max_driver_bytes"] >= 50 * 1024 ** 3,
             "Driver guard must be at least 50 GiB")
    _require(limits["max_bootstrap_resamples"] >= bootstrap["resamples"],
             "Bootstrap resamples exceed the declared limit")

    policy = _exact_keys(config["resource_policy"], {
        "device", "cpu_fallback", "exclusive_mps_worker", "workers",
        "downloads", "driver_limit_bytes", "sustained_swap_growth_bytes",
        "minimum_host_free_percent", "sustained_host_samples",
        "initial_free_admission"}, "resource_policy")
    _require(policy["device"] == "mps" and policy["cpu_fallback"] is False
             and policy["exclusive_mps_worker"] is True
             and policy["workers"] == 1 and policy["downloads"] == 0,
             "Resource policy must stay one exclusive MPS worker without fallback")
    _require(policy["driver_limit_bytes"] >= 50 * 1024 ** 3,
             "Driver limit must be at least 50 GiB")
    _positive_int(policy["sustained_swap_growth_bytes"],
                  "resource_policy.sustained_swap_growth_bytes")
    _finite(policy["minimum_host_free_percent"],
            "resource_policy.minimum_host_free_percent", minimum=0.0,
            maximum=100.0)
    _positive_int(policy["sustained_host_samples"],
                  "resource_policy.sustained_host_samples")
    _require(policy["initial_free_admission"] is True,
             "Initial free-memory admission is required")
    actions = _exact_keys(config["action_limits"], {
        "enabled_ports", "enabled_branches", "core_impl", "host_actions",
        "success_claims"}, "action_limits")
    _require(actions["host_actions"] == "none"
             and actions["success_claims"] == "none"
             and actions["enabled_ports"] == []
             and actions["enabled_branches"] == ["memory"]
             and actions["core_impl"] == "none",
             "No host actions or success claims are declared")
    stops = config["stop_conditions"]
    _require(isinstance(stops, list) and len(stops) >= 5
             and all(isinstance(item, str) and item.strip() for item in stops),
             "At least five nonempty stop conditions are required")
    return dict(config)


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# External caption source (read-only, already admitted)
# ---------------------------------------------------------------------------


def load_external_source(project_root: str | Path,
                         config: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(project_root)
    external = config["external"]
    root = _safe_relative(project_root, external["root"])
    _require(root.is_dir(), f"External tranche is missing: {root}")
    intake = _load_json(root / external["intake_manifest"])
    _require(intake.get("intake_id") == "external_caption_tranche_20261025",
             "External intake identity differs")
    artifacts = intake["artifacts"]
    corpus_pin = external["corpus"]
    _require(sha256_file(root / corpus_pin["path"]) == corpus_pin["sha256"],
             "External corpus bytes changed")
    _require(artifacts["corpus"]["sha256"] == corpus_pin["sha256"],
             "Intake manifest corpus hash differs from config")
    for name in ("train", "development", "final_eval_untouched"):
        declared = external["manifests"][name]
        _require(sha256_file(root / declared["path"]) == declared["sha256"],
                 f"External {name} manifest bytes changed")
        _require(artifacts["manifests"][name]["sha256"] == declared["sha256"],
                 f"Intake manifest hash differs for {name}")
    final_entry = artifacts["manifests"]["final_eval_untouched"]
    _require(final_entry.get("sealed") is True,
             "External final manifest is not sealed")
    final_manifest = _load_json(
        root / external["manifests"]["final_eval_untouched"]["path"])
    derived_final_seal = _role_manifest_seal(final_manifest)
    _require(derived_final_seal
             == external["manifests"]["final_eval_untouched"]["seal_sha256"]
             == final_entry.get("seal_sha256")
             == final_manifest.get("seal_sha256"),
             "External final seal differs from the manifest-row derivation")

    corpus_rows = _load_jsonl(root / corpus_pin["path"])
    _require(len(corpus_rows) == 30, "External corpus row count differs")
    by_row = {row["row_id"]: row for row in corpus_rows}
    _require(len(by_row) == len(corpus_rows), "External corpus row ids repeat")
    image_root = root / external["image_root"]

    def load_role(name: str) -> list[dict[str, Any]]:
        manifest = _load_json(root / external["manifests"][name]["path"])
        _require(manifest.get("format") == "replica.external_caption_manifest.v1",
                 f"External {name} manifest format differs")
        _require(manifest.get("role") == name, f"External {name} role differs")
        rows = []
        seen_images: set[str] = set()
        for entry in manifest["rows"]:
            row = by_row.get(entry["row_id"])
            _require(row is not None,
                     f"External manifest references an unknown row: {entry['row_id']}")
            _require(sha256_text(canonical_json(row)) == entry["row_sha256"],
                     f"External row bytes changed: {entry['row_id']}")
            _require(row["split"]["role"] == name,
                     f"External row role differs: {entry['row_id']}")
            caption = row["target"]["caption"]
            _require(sha256_text(caption) == entry["target_sha256"],
                     f"External caption digest differs: {entry['row_id']}")
            _require(row["root_asset"]["image_sha256"] == entry["image_sha256"],
                     f"External image hash differs: {entry['row_id']}")
            _require(entry["image_sha256"] not in seen_images,
                     f"External {name} split reuses an image")
            seen_images.add(entry["image_sha256"])
            upstream_id = row["root_asset"]["upstream_id"]
            image = image_root / f"{upstream_id}.jpg"
            _require(image.is_file(), f"External image is missing: {image}")
            _require(sha256_file(image) == entry["image_sha256"],
                     f"External image bytes changed: {image}")
            _require(caption.casefold()
                     not in canonical_json(row["observation"]).casefold(),
                     f"Gold caption leaked into observation: {entry['row_id']}")
            rows.append({
                "case_id": entry["row_id"],
                "group_key": entry["group_key"],
                "image": image.relative_to(project_root).as_posix(),
                "image_sha256": entry["image_sha256"],
                "image_bytes": image.read_bytes(),
                "question": row["observation"]["prompt"],
                "response": caption,
                "row_sha256": entry["row_sha256"],
                "target_sha256": entry["target_sha256"],
                "upstream_id": upstream_id,
            })
        expected = external[
            {"train": "train_rows", "development": "development_rows",
             "final_eval_untouched": "final_rows"}[name]]
        _require(len(rows) == expected, f"External {name} row count differs")
        return rows

    train = load_role("train")
    development = load_role("development")
    final_rows = load_role("final_eval_untouched")
    groups = [row["group_key"] for row in train + development + final_rows]
    _require(len(set(groups)) == len(groups),
             "External root groups cross managed splits")
    images = [row["image_sha256"] for row in train + development + final_rows]
    _require(len(set(images)) == len(images),
             "External image hashes cross managed splits")
    _require(len({row["row_sha256"] for row in train + development + final_rows})
             == len(train) + len(development) + len(final_rows),
             "External row ids cross managed splits")
    managed_ids = {row["case_id"] for row in train + development}
    final_ids = {row["case_id"] for row in final_rows}
    _require(managed_ids.isdisjoint(final_ids),
             "External final rows overlap managed splits")
    _require({row["image_sha256"] for row in train + development}.isdisjoint(
        {row["image_sha256"] for row in final_rows}),
        "External final images overlap managed splits")
    _require({row["group_key"] for row in train + development}.isdisjoint(
        {row["group_key"] for row in final_rows}),
        "External final root groups overlap managed splits")
    return {
        "root": root.relative_to(project_root).as_posix(),
        "train": train,
        "development": development,
        "final_pointer": {
            "manifest": external["manifests"]["final_eval_untouched"]["path"],
            "manifest_sha256": external["manifests"][
                "final_eval_untouched"]["sha256"],
            "seal_sha256": derived_final_seal,
            "rows": len(final_rows),
            "row_ids_sha256": fingerprint(sorted(
                row["case_id"] for row in final_rows)),
            "consumed": False,
            "policy": "sealed_unconsumed_pointer_only",
        },
    }


# ---------------------------------------------------------------------------
# Procedural curriculum corpus
# ---------------------------------------------------------------------------


class _Draw:
    """Stable SHA256 counter PRNG (independent of Python's random module)."""

    def __init__(self, seed: int, *parts: Any) -> None:
        self._prefix = "|".join([str(seed), *[str(part) for part in parts]])
        self._counter = 0

    def _int(self, modulus: int, salt: str = "") -> int:
        _require(modulus > 0, "PRNG modulus must be positive")
        payload = f"{self._prefix}|{salt}|{self._counter}".encode("utf-8")
        self._counter += 1
        return int(hashlib.sha256(payload).hexdigest()[:16], 16) % modulus

    def pick(self, values: Sequence[Any], salt: str = "") -> Any:
        _require(bool(values), "Cannot pick from an empty sequence")
        return values[self._int(len(values), salt)]


def _grid_cells(width: int, height: int) -> list[tuple[int, int]]:
    xs = [int(width * fraction) for fraction in GRID_FRACTIONS_X]
    ys = [int(height * fraction) for fraction in GRID_FRACTIONS_Y]
    return [(x, y) for y in ys for x in xs]


def _object_bbox(shape: str, center: Sequence[int], size: int) -> list[int]:
    half_w = int(size) + (2 if shape == "rectangle" else 0)
    x, y = int(center[0]), int(center[1])
    return [x - half_w, y - size, x + half_w, y + size]


def _pool(config: Mapping[str, Any], split: str) -> list[tuple[str, str]]:
    return [tuple(item) for item in config["procedural"]["descriptor_pools"][split]]


def _layouts(config: Mapping[str, Any], split: str) -> tuple[str, ...]:
    return tuple(config["procedural"]["split_layouts"][split])


def _dimensions(config: Mapping[str, Any]) -> tuple[int, int]:
    renderer = config["procedural"]["renderer"]
    return int(renderer["width"]), int(renderer["height"])


def _scene(asset_id: str, width: int, height: int, layout: str,
           objects: Sequence[Mapping[str, Any]], *,
           allow_overlap: bool = False) -> dict[str, Any]:
    cells = _grid_cells(width, height)
    built = []
    used_cells = set()
    for item in objects:
        cell = int(item["cell"])
        _require(0 <= cell < len(cells), "Cell index is outside the grid")
        if not allow_overlap:
            _require(cell not in used_cells, "Objects must occupy distinct cells")
        used_cells.add(cell)
        center = (list(item["center"]) if "center" in item
                  else list(cells[cell]))
        bbox = _object_bbox(item["shape"], center, int(item["size"]))
        _require(0 <= bbox[0] and bbox[2] < width
                 and 0 <= bbox[1] and bbox[3] < height,
                 f"Object falls outside the image: {asset_id}")
        built.append({
            "id": str(item["id"]),
            "role": str(item.get("role", "target")),
            "shape": str(item["shape"]),
            "color": str(item["color"]),
            "bbox": bbox,
            "center": [int(center[0]), int(center[1])],
            "size": int(item["size"]),
        })
    _require(len({obj["id"] for obj in built}) == len(built),
             "Object ids must be unique inside a scene")
    return {
        "scene_id": asset_id,
        "width": int(width),
        "height": int(height),
        "layout": {"template": layout,
                   "cell_indices": [int(item["cell"]) for item in objects],
                   "grid": {"columns": 4, "rows": 2}},
        "objects": built,
    }


def _distinct_descriptor(pool: Sequence[tuple[str, str]],
                         excluded: Iterable[tuple[str, str]]
                         ) -> tuple[str, str]:
    excluded = set(excluded)
    options = [item for item in pool if item not in excluded]
    _require(options, "Descriptor pool lacks a distinct distractor")
    return options[0]


def _colour_task_shape(pool: Sequence[tuple[str, str]]) -> str:
    by_shape: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for descriptor in pool:
        by_shape[descriptor[1]].append(descriptor)
    shapes = sorted(shape for shape, items in by_shape.items()
                    if len(items) >= 2)
    _require(bool(shapes), "Colour task needs two colours of one shape")
    return shapes[0]


def _colour_pair(by_shape: Mapping[str, list[tuple[str, str]]], shape: str,
                 index: int) -> list[tuple[str, str]]:
    colors = sorted({color for color, _ in by_shape[shape]})
    _require(len(colors) >= 2, "Colour task needs two colours of one shape")
    start = index % len(colors)
    return [(colors[start], shape), (colors[(start + 1) % len(colors)], shape)]


def _count_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                 salt: int, variant: int, width: int, height: int
                 ) -> dict[str, Any]:
    target = pool[0]
    distractor = _distinct_descriptor(pool, [target])
    target_count = 2 + variant
    distractor_count = 1 + index % 3
    _require(target_count + distractor_count <= 8, "Count scene exceeds the grid")
    order = list(GRID_LAYOUTS[layout])
    size = 8 + (((index // 3) + salt) % 5)
    objects = []
    for position in range(target_count):
        objects.append({"id": f"t{position}", "role": "target",
                        "shape": target[1], "color": target[0],
                        "cell": order[position], "size": size})
    for position in range(distractor_count):
        objects.append({"id": f"d{position}", "role": "distractor",
                        "shape": distractor[1], "color": distractor[0],
                        "cell": order[target_count + position], "size": size})
    return _scene("count", width, height, layout, objects)


def _presence_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                    salt: int, variant: int, width: int, height: int
                    ) -> dict[str, Any]:
    target = pool[0]
    others = list(pool[1:])
    options: list[tuple[tuple[str, str], ...]] = []
    for count in range(1, min(3, len(others)) + 1):
        options.extend(combinations(others, count))
    chosen = options[index % len(options)]
    size = 8 + (((index // len(options)) + salt) % 5)
    order = list(GRID_LAYOUTS[layout])
    specs: list[dict[str, Any]] = []
    if variant == 0:
        specs.append({"id": "target", "role": "target",
                      "shape": target[1], "color": target[0],
                      "cell": order[0], "size": size})
    for position, descriptor in enumerate(chosen):
        specs.append({"id": f"d{position}", "role": "distractor",
                      "shape": descriptor[1], "color": descriptor[0],
                      "cell": order[len(specs)], "size": size})
    return _scene("presence", width, height, layout, specs)


def _spatial_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                   salt: int, variant: int, width: int, height: int
                   ) -> dict[str, Any]:
    order = list(GRID_LAYOUTS[layout])
    first, second, decoy = pool[0], pool[1], pool[2]
    _require(len({first, second, decoy}) == 3,
             "Spatial descriptors must be distinct")
    cells = _grid_cells(width, height)
    row = (index // 6) % 2
    if (index // 2) % 2 == 0:
        left_cell, right_cell = 4 * row + 0, 4 * row + 2
    else:
        left_cell, right_cell = 4 * row + 1, 4 * row + 3
    _require(cells[left_cell][0] < cells[right_cell][0],
             "Spatial slots must separate horizontally")
    size = 7 + ((index + salt) % 6)
    slots = [left_cell, right_cell] if variant == 0 else [right_cell, left_cell]
    decoy_cell = 4 * (1 - row) + 2
    specs = [
        {"id": "a", "role": "target", "shape": first[1], "color": first[0],
         "cell": slots[0], "size": size},
        {"id": "b", "role": "target", "shape": second[1], "color": second[0],
         "cell": slots[1], "size": size},
        {"id": "d", "role": "distractor", "shape": decoy[1],
         "color": decoy[0], "cell": decoy_cell, "size": size},
    ]
    return _scene("spatial", width, height, layout, specs)


def _color_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                 salt: int, variant: int, width: int, height: int
                 ) -> dict[str, Any]:
    by_shape: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for descriptor in pool:
        by_shape[descriptor[1]].append(descriptor)
    shape = _colour_task_shape(pool)
    pair = _colour_pair(by_shape, shape, index)
    _require(len({item[0] for item in pair}) == 2,
             "Colour pair must have distinct colours")
    distractors = [item for item in pool if item[1] != shape]
    _require(bool(distractors), "Colour task needs a distinct-shape distractor")
    order = list(GRID_LAYOUTS[layout])
    row = (index // 6) % 2
    size = 7 + ((index + salt) % 6)
    color = pair[variant % 2][0]
    specs = [
        {"id": "target", "role": "target", "shape": shape, "color": color,
         "cell": 4 * row + 0, "size": size},
        {"id": "d0", "role": "distractor", "shape": distractors[0][1],
         "color": distractors[0][0], "cell": 4 * row + 3, "size": size},
        {"id": "d1", "role": "distractor",
         "shape": distractors[-1][1], "color": distractors[-1][0],
         "cell": 4 * (1 - row) + 1, "size": size},
    ]
    return _scene("color", width, height, layout, specs)


def _shape_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                 salt: int, variant: int, width: int, height: int
                 ) -> dict[str, Any]:
    by_color: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for descriptor in pool:
        by_color[descriptor[0]].append(descriptor)
    colors = sorted(color for color, items in by_color.items() if len(items) >= 2)
    _require(bool(colors), "Shape task needs two shapes of one colour")
    color = colors[0]
    pair = sorted(by_color[color])[:2]
    _require(len({item[1] for item in pair}) == 2,
             "Shape pair must have distinct shapes")
    distractors = [item for item in pool if item[0] != color]
    _require(bool(distractors), "Shape task needs a distinct-colour distractor")
    row = (index // 6) % 2
    size = 7 + ((index + salt) % 6)
    shape = pair[variant % 2][1]
    specs = [
        {"id": "target", "role": "target", "shape": shape, "color": color,
         "cell": 4 * row + 0, "size": size},
        {"id": "d0", "role": "distractor", "shape": distractors[0][1],
         "color": distractors[0][0], "cell": 4 * row + 3, "size": size},
        {"id": "d1", "role": "distractor", "shape": distractors[-1][1],
         "color": distractors[-1][0], "cell": 4 * (1 - row) + 1, "size": size},
    ]
    return _scene("shape", width, height, layout, specs)


def _absent_shape(pool: Sequence[tuple[str, str]], index: int) -> str:
    used = {pool[0][1], _colour_task_shape(pool)}
    absent_shapes = sorted(shape for shape in SHAPES if shape not in used)
    _require(bool(absent_shapes), "No absent shape remains")
    return absent_shapes[(index // 3) % len(absent_shapes)]


def _occluded_descriptors(pool: Sequence[tuple[str, str]], index: int
                          ) -> tuple[tuple[str, str], tuple[str, str],
                                     tuple[str, str]]:
    rectangles = [item for item in pool if item[1] == "rectangle"]
    _require(bool(rectangles),
             "Occluded relation needs a rectangle occluder")
    front = sorted(rectangles)[index % len(rectangles)]
    other_rectangles = [item for item in rectangles if item != front]
    if other_rectangles:
        hidden = other_rectangles[0]
    else:
        hidden = next(item for item in pool if item != front)
    distractor = next(item for item in pool
                      if item not in {front, hidden})
    return front, hidden, distractor


def _abstain_scene(pool: Sequence[tuple[str, str]], layout: str, index: int,
                   salt: int, kind: str, variant: int, width: int,
                   height: int) -> dict[str, Any]:
    order = list(GRID_LAYOUTS[layout])
    if kind == "absent_referent":
        absent = _absent_shape(pool, index)
        base_count = 1 + (index % 3)
        count = base_count + variant
        size = 7 + (((index // 3) + salt) % 6)
        specs = [{"id": f"d{position}", "role": "distractor",
                  "shape": pool[0][1], "color": pool[0][0],
                  "cell": order[position], "size": size}
                 for position in range(count)]
        return _scene(f"abstain-absent-{absent}", width, height, layout, specs)
    if kind == "ambiguous_referent":
        by_shape: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for descriptor in pool:
            by_shape[descriptor[1]].append(descriptor)
        shapes = sorted(shape for shape, items in by_shape.items()
                        if len({item[0] for item in items}) >= 2)
        _require(bool(shapes), "No shape has two colours for ambiguity")
        shape = shapes[0]
        colors = sorted({item[0] for item in by_shape[shape]})[:2]
        colors = colors if variant == 0 else list(reversed(colors))
        row = (index // 6) % 2
        size = 7 + ((index + salt) % 6)
        specs = [
            {"id": "a", "role": "target", "shape": shape, "color": colors[0],
             "cell": 4 * row + 0, "size": size},
            {"id": "b", "role": "target", "shape": shape, "color": colors[1],
             "cell": 4 * row + 2, "size": size},
        ]
        return _scene(f"abstain-ambiguous-{shape}", width, height, layout, specs)
    if kind == "occluded_relation":
        front, hidden, distractor = _occluded_descriptors(pool, index)
        cells = _grid_cells(width, height)
        front_cell = order[(index // 3) % len(order)]
        remaining = [cell for cell in order if cell != front_cell]
        distractor_cell = remaining[index % len(remaining)]
        front_center = list(cells[front_cell])
        hidden_center = [front_center[0] + (-6 if variant == 0 else 6),
                         front_center[1]]
        specs = [
            {"id": "hidden", "role": "occluded", "shape": hidden[1],
             "color": hidden[0], "cell": front_cell, "size": 5,
             "center": hidden_center},
            {"id": "distractor", "role": "distractor",
             "shape": distractor[1], "color": distractor[0],
             "cell": distractor_cell, "size": 8},
            {"id": "front", "role": "target", "shape": front[1],
             "color": front[0], "cell": front_cell, "size": 13},
        ]
        return _scene("abstain-occluded", width, height, layout, specs,
                      allow_overlap=True)
    raise VisionCurriculumError(f"Unknown abstention kind: {kind}")


_SCENE_BUILDERS = {
    "spatial": _spatial_scene,
    "count": _count_scene,
    "presence": _presence_scene,
    "color": _color_scene,
    "shape": _shape_scene,
}


def _answer_question(task: str, params: Mapping[str, Any],
                    index: int = 0) -> str:
    if task == "count":
        target = params["target"]
        return f"How many {target[0]} {target[1]} objects are in the image?"
    if task == "presence":
        target = params["target"]
        return f"Is there a {target[0]} {target[1]} in the image?"
    if task == "spatial":
        first, second = params["first"], params["second"]
        return (f"Is the {first[0]} {first[1]} to the left of the "
                f"{second[0]} {second[1]}?")
    if task == "color":
        word = "What" if index % 2 == 0 else "Which"
        return f"{word} color is the {params['shape']}?"
    if task == "shape":
        return f"What shape is the {params['color']} object?"
    raise VisionCurriculumError(f"Unknown task question: {task}")


def _abstain_question(kind: str, params: Mapping[str, Any],
                      index: int) -> str:
    if kind == "absent_referent":
        return f"What color is the {params['absent_shape']}?"
    if kind == "ambiguous_referent":
        word = "Which" if index % 2 == 0 else "What"
        return f"{word} color is the {params['shape']}?"
    if kind == "occluded_relation":
        hidden, front = params["hidden"], params["visible"]
        return (f"Is the {hidden[0]} {hidden[1]} to the left of the "
                f"{front[0]} {front[1]}?")
    raise VisionCurriculumError(f"Unknown abstention question: {kind}")


def _verify_check(check: Mapping[str, Any], scene: Mapping[str, Any]) -> str:
    """Execute one verifier check; abstention checks prove undecidability."""
    kind = check["kind"]
    objects = scene["objects"]
    by_id = {obj["id"]: obj for obj in objects}
    if kind == "count":
        matched = [obj for obj in objects
                   if all(obj.get(key) == value
                          for key, value in check["filter"].items())]
        return str(len(matched))
    if kind == "presence":
        matched = [obj for obj in objects
                   if obj["color"] == check["color"]
                   and obj["shape"] == check["shape"]]
        return "yes" if matched else "no"
    if kind == "spatial":
        first, second = by_id[check["first_id"]], by_id[check["second_id"]]
        return "yes" if first["center"][0] < second["center"][0] else "no"
    if kind == "color":
        return str(by_id[check["object_id"]]["color"])
    if kind == "shape":
        return str(by_id[check["object_id"]]["shape"])
    if kind == "abstain":
        reason = check["reason"]
        if reason == "absent_referent":
            found = [obj for obj in objects if obj["shape"] == check["shape"]]
            _require(not found,
                     "Absent-referent check found the referenced shape")
        elif reason == "ambiguous_referent":
            found = [obj for obj in objects if obj["shape"] == check["shape"]]
            _require(len({obj["color"] for obj in found}) >= 2,
                     "Ambiguity check lacks two candidate colours")
        elif reason == "occluded_relation":
            order = {obj["id"]: position for position, obj in enumerate(objects)}
            hidden = by_id[check["occluded_id"]]
            front = by_id[check["visible_id"]]
            _require(order[hidden["id"]] < order[front["id"]],
                     "Occluded object must be drawn before the occluder")
            hx0, hy0, hx1, hy1 = hidden["bbox"]
            fx0, fy0, fx1, fy1 = front["bbox"]
            _require(fx0 <= hx0 and fy0 <= hy0
                     and hx1 <= fx1 and hy1 <= fy1,
                     "Occluded object is not fully covered by the occluder")
            _require(hidden["center"][0] != front["center"][0],
                     "Occluded relation needs a left/right candidate")
        else:
            raise VisionCurriculumError(f"Unknown abstention reason: {reason}")
        return UNKNOWN_ANSWER
    raise VisionCurriculumError(f"Unknown verifier check: {kind}")


def _style_question(split: str, index: int, core: str) -> str:
    prefixes = PROMPT_PREFIXES[split]
    prefix = prefixes[index % len(prefixes)]
    suffix = PROMPT_SUFFIXES[(index // len(prefixes)) % len(PROMPT_SUFFIXES)]
    return f"{prefix}{core}{suffix}"


def _spec(config: Mapping[str, Any], split: str, task: str, index: int,
          salt: int = 0) -> dict[str, Any]:
    pool = _pool(config, split)
    layout = _layouts(config, split)[index % len(_layouts(config, split))]
    width, height = _dimensions(config)
    if task == "count":
        target = pool[0]
        distractor = _distinct_descriptor(pool, [target])
        params = {"target": list(target), "distractor": list(distractor)}
        scenes = [_count_scene(pool, layout, index, salt, variant, width, height)
                  for variant in (0, 1)]
        check = {"kind": "count",
                 "filter": {"color": target[0], "shape": target[1]}}
        question = _answer_question(task, params, index)
    elif task == "presence":
        target = pool[0]
        params = {"target": list(target)}
        scenes = [_presence_scene(pool, layout, index, salt, variant, width,
                                  height) for variant in (0, 1)]
        check = {"kind": "presence", "color": target[0], "shape": target[1]}
        question = _answer_question(task, params, index)
    elif task == "spatial":
        first, second = pool[0], pool[1]
        params = {"first": list(first), "second": list(second)}
        scenes = [_spatial_scene(pool, layout, index, salt, variant, width,
                                 height) for variant in (0, 1)]
        check = {"kind": "spatial", "first_id": "a", "second_id": "b"}
        question = _answer_question(task, params, index)
    elif task == "color":
        by_shape: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for descriptor in pool:
            by_shape[descriptor[1]].append(descriptor)
        color_shapes = sorted(shape for shape, items in by_shape.items()
                              if len(items) >= 2)
        shape = color_shapes[index % len(color_shapes)]
        pair = _colour_pair(by_shape, shape, index)
        params = {"shape": shape, "colors": [pair[0][0], pair[1][0]]}
        scenes = [_color_scene(pool, layout, index, salt, variant, width,
                               height) for variant in (0, 1)]
        check = {"kind": "color", "object_id": "target"}
        question = _answer_question(task, params, index)
    elif task == "shape":
        by_color: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for descriptor in pool:
            by_color[descriptor[0]].append(descriptor)
        colors = sorted(color for color, items in by_color.items()
                        if len(items) >= 2)
        _require(bool(colors),
                 f"Shape task requires one colour with two shapes in {split}")
        color = colors[0]
        pair = sorted(by_color[color])[:2]
        params = {"color": color, "shapes": [pair[0][1], pair[1][1]]}
        scenes = [_shape_scene(pool, layout, index, salt, variant, width,
                               height) for variant in (0, 1)]
        check = {"kind": "shape", "object_id": "target"}
        question = _answer_question(task, params, index)
    elif task == "abstain":
        kind = ABSTAIN_KINDS[index % len(ABSTAIN_KINDS)]
        if kind == "absent_referent":
            absent = _absent_shape(pool, index)
            params = {"absent_shape": absent, "kind": kind}
            check = {"kind": "abstain", "reason": kind, "shape": absent}
        elif kind == "ambiguous_referent":
            by_shape: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for descriptor in pool:
                by_shape[descriptor[1]].append(descriptor)
            shape = sorted(shape for shape, items in by_shape.items()
                           if len({item[0] for item in items}) >= 2)[0]
            params = {"shape": shape, "kind": kind}
            check = {"kind": "abstain", "reason": kind, "shape": shape}
        else:
            front, hidden, _ = _occluded_descriptors(pool, index)
            params = {"hidden": list(hidden), "visible": list(front),
                      "kind": kind}
            check = {"kind": "abstain", "reason": kind,
                     "occluded_id": "hidden", "visible_id": "front"}
        scenes = [_abstain_scene(pool, layout, index, salt, kind, variant,
                                 width, height) for variant in (0, 1)]
        question = _abstain_question(kind, params, index)
    else:
        raise VisionCurriculumError(f"Unknown curriculum task: {task}")
    question = _style_question(split, index, question)
    return {"split": split, "task": task, "index": index, "layout": layout,
            "params": params, "scenes": scenes, "question": question,
            "check": check, "kind": params.get("kind")}


def _combination(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": spec["task"],
        "kind": spec["kind"],
        "layout": spec["layout"],
        "params": spec["params"],
        "scene_sha256": [scene_digest(scene) for scene in spec["scenes"]],
    }


def _decisive_change(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec["task"] == "abstain":
        return {"kind": "abstain_variant", "reason": spec["kind"],
                "answers_differ": False}
    return {"kind": f"{spec['task']}_answer_change", "answers_differ": True}


def _materialize_world(config: Mapping[str, Any], split: str, task: str,
                       index: int, world_id: str, salt: int = 0
                       ) -> dict[str, Any]:
    spec = _spec(config, split, task, index, salt)
    width, height = _dimensions(config)
    pair_id = f"{world_id}-p0"
    side_records = []
    for side, scene in (("a", spec["scenes"][0]), ("b", spec["scenes"][1])):
        scene = dict(scene)
        scene["scene_id"] = f"{pair_id}-{side}"
        png = render_scene_png(scene)
        side_records.append({"side": side, "scene": scene, "png": png,
                             "answer": _verify_check(spec["check"], scene)})
    answers = {record["side"]: record["answer"] for record in side_records}
    if task == "abstain":
        _require(answers["a"] == answers["b"] == UNKNOWN_ANSWER,
                 f"Abstention answers must both be unknown: {pair_id}")
    else:
        _require(answers["a"] != answers["b"],
                 f"Answer-changing pairs must differ: {pair_id}")
        _require(UNKNOWN_ANSWER not in answers.values(),
                 f"Answer tasks must not produce unknown: {pair_id}")
    if task == "abstain" and spec["kind"] == "occluded_relation":
        # Declared identical-pixel control: the hidden object moves left/right
        # behind a full occluder, so the two images must be byte-identical while
        # the verifier scenes differ in the hidden object's true position.
        _require(side_records[0]["png"] == side_records[1]["png"],
                 f"Occlusion control images must be identical: {pair_id}")
        _require(scene_digest(side_records[0]["scene"])
                 != scene_digest(side_records[1]["scene"]),
                 f"Occlusion control scenes must differ: {pair_id}")
    else:
        _require(side_records[0]["png"] != side_records[1]["png"],
                 f"Pair images must render differently: {pair_id}")
        _require(scene_digest(side_records[0]["scene"])
                 != scene_digest(side_records[1]["scene"]),
                 f"Pair scenes must differ: {pair_id}")
    declared = set(_pool(config, split))
    for record in side_records:
        for obj in record["scene"]["objects"]:
            _require((obj["color"], obj["shape"]) in declared,
                     f"Scene uses an undeclared descriptor in {split}: "
                     f"{obj['color']}/{obj['shape']}")
    return {"spec": spec, "pair_id": pair_id, "sides": side_records,
            "answers": answers, "width": width, "height": height}


def build_procedural(config: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config(config)
    files: dict[str, bytes] = {}
    assets: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    worlds: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    combinations: dict[str, set[str]] = {split: set() for split in SPLITS}
    used_scene_hashes: set[str] = set()
    used_png_hashes: set[str] = set()
    global_index = 0
    for split in SPLITS:
        for task in SPLIT_TASKS[split]:
            for index in range(int(config["procedural"]["worlds_per_task"])):
                world_id = f"vc1-w{global_index:03d}"
                chosen = None
                for salt in range(32):
                    materialized = _materialize_world(
                        config, split, task, index, world_id, salt)
                    key = canonical_json(_combination(materialized["spec"]))
                    scene_hashes = {scene_digest(record["scene"])
                                    for record in materialized["sides"]}
                    png_hashes = {sha256_bytes(record["png"])
                                  for record in materialized["sides"]}
                    if key in combinations[split]:
                        continue
                    if scene_hashes & used_scene_hashes:
                        continue
                    if png_hashes & used_png_hashes:
                        continue
                    chosen = (materialized, key, scene_hashes, png_hashes)
                    break
                _require(chosen is not None,
                         f"Could not materialise a unique world: {world_id}")
                materialized, key, scene_hashes, png_hashes = chosen
                combinations[split].add(key)
                used_scene_hashes.update(scene_hashes)
                used_png_hashes.update(png_hashes)
                spec = materialized["spec"]
                pair_id = materialized["pair_id"]
                answers = materialized["answers"]
                asset_ids = []
                for record in materialized["sides"]:
                    asset_id = f"{pair_id}-{record['side']}"
                    path = f"assets/{asset_id}.png"
                    scene = record["scene"]
                    entry = {
                        "asset_id": asset_id,
                        "world_id": world_id,
                        "pair_id": pair_id,
                        "split": split,
                        "task": task,
                        "kind": spec["kind"],
                        "side": record["side"],
                        "path": path,
                        "bytes": len(record["png"]),
                        "sha256": sha256_bytes(record["png"]),
                        "width": int(scene["width"]),
                        "height": int(scene["height"]),
                        "scene_sha256": scene_digest(scene),
                        "combination": _combination(spec),
                    }
                    entry["record_hash"] = fingerprint(entry)
                    files[path] = record["png"]
                    assets.append(entry)
                    asset_ids.append(asset_id)
                    rows.append({
                        "case_id": asset_id,
                        "pair_id": pair_id,
                        "asset_id": asset_id,
                        "image": path,
                        "image_sha256": entry["sha256"],
                        "question": spec["question"],
                    })
                    cases.append({
                        "case_id": asset_id,
                        "pair_id": pair_id,
                        "asset_id": asset_id,
                        "world_id": world_id,
                        "split": split,
                        "task": task,
                        "kind": spec["kind"],
                        "question": spec["question"],
                        "answer": answers[record["side"]],
                        "label": answers[record["side"]],
                        "check": {**spec["check"],
                                  "expected": answers[record["side"]]},
                        "scene_sha256": entry["scene_sha256"],
                        "label_visibility": "verifier_only",
                    })
                    scenes.append({"asset_id": asset_id,
                                   "scene_sha256": entry["scene_sha256"],
                                   "scene": scene})
                pairs.append({
                    "pair_id": pair_id,
                    "world_id": world_id,
                    "split": split,
                    "task": task,
                    "kind": spec["kind"],
                    "question": spec["question"],
                    "combination": _combination(spec),
                    "answers": dict(answers),
                    "answers_differ": answers["a"] != answers["b"],
                    "decisive_change": _decisive_change(spec),
                    "sides": {
                        "a": {"asset_id": asset_ids[0],
                              "answer": answers["a"]},
                        "b": {"asset_id": asset_ids[1],
                              "answer": answers["b"]},
                    },
                })
                worlds.append({
                    "world_id": world_id,
                    "split": split,
                    "task": task,
                    "kind": spec["kind"],
                    "pair_ids": [pair_id],
                    "asset_ids": asset_ids,
                    "combination": _combination(spec),
                })
                global_index += 1
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(combinations[left].isdisjoint(combinations[right]),
                         f"World combinations overlap across {left}/{right}")

    prompt_sets: dict[str, set[str]] = {}
    prompt_answer_sets: dict[str, set[str]] = {}
    world_question: dict[str, str] = {}
    for pair in pairs:
        world_question[pair["world_id"]] = pair["question"]
    for split in SPLITS:
        questions = [world_question[world["world_id"]]
                     for world in worlds if world["split"] == split]
        _require(len(set(questions)) == len(questions),
                 f"{split} worlds duplicate a prompt")
        prompt_sets[split] = set(questions)
        answers = {}
        for pair in pairs:
            if pair["split"] == split:
                answers[pair["question"]] = pair["answers"]["a"]
        _require(len(answers) == len(prompt_sets[split]),
                 f"{split} prompt/answer pairs are not one-to-one")
        prompt_answer_sets[split] = {
            canonical_json([question, answer])
            for question, answer in answers.items()}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(prompt_sets[left].isdisjoint(prompt_sets[right]),
                         f"{left}/{right} prompts are duplicated")
                _require(prompt_answer_sets[left].isdisjoint(
                    prompt_answer_sets[right]),
                    f"{left}/{right} prompt+answer pairs are duplicated")
    for pair in pairs:
        _require("unknown" not in pair["question"].casefold()
                 and "absent" not in pair["question"].casefold()
                 and "occluded" not in pair["question"].casefold(),
                 "Prompt text exposes the abstention shortcut")

    pools = {split: _pool(config, split) for split in SPLITS}
    audit_descriptors = set(pools["audit"])
    _require(audit_descriptors.isdisjoint(set(pools["train"] + pools["dev"])),
             "Audit descriptors overlap curriculum train/dev")
    _require(audit_descriptors.isdisjoint(AUDIT_FORBIDDEN_V2_DESCRIPTORS),
             "Audit descriptors overlap the admitted diagnostic")

    files["model_rows.jsonl"] = _jsonl_bytes(rows)
    files["verifier_cases.jsonl"] = _jsonl_bytes(cases)
    files["verifier_pairs.jsonl"] = _jsonl_bytes(pairs)
    files["verifier_scenes.jsonl"] = _jsonl_bytes(scenes)

    coverage: dict[str, dict[str, list[str]]] = {}
    for task in ALL_TASKS:
        coverage[task] = {
            split: sorted({case["label"] for case in cases
                           if case["task"] == task and case["split"] == split})
            for split in SPLITS}
    unknown_coverage = {
        split: {kind: sum(1 for case in cases
                          if case["split"] == split and case["task"] == "abstain"
                          and case["kind"] == kind)
                for kind in ABSTAIN_KINDS}
        for split in SPLITS}
    for split in SPLITS:
        _require(all(value >= 1 for value in unknown_coverage[split].values()),
                 f"Abstention kind missing from {split}")
    train_labels = {task: set(coverage[task]["train"]) for task in ALL_TASKS}
    for task in SPLIT_TASKS["train"]:
        for split in ("dev", "audit"):
            _require(set(coverage[task][split]) <= train_labels[task],
                     f"{split} labels are not covered by train for {task}")
    counts = {
        "worlds": len(worlds),
        "pairs": len(pairs),
        "assets": len(assets),
        "verifier_cases": len(cases),
        "verifier_scenes": len(scenes),
        "model_rows": len(rows),
        "asset_bytes": sum(int(asset["bytes"]) for asset in assets),
        "task_histogram": dict(sorted(Counter(
            case["task"] for case in cases).items())),
        "split_worlds": dict(Counter(world["split"] for world in worlds)),
        "split_assets": dict(Counter(asset["split"] for asset in assets)),
        "split_model_rows": dict(Counter(case["split"] for case in cases)),
        "unknown_worlds": {split: sum(
            1 for world in worlds
            if world["split"] == split and world["task"] == "abstain")
            for split in SPLITS},
    }
    for split in SPLITS:
        for name in ("split_worlds", "split_assets", "split_model_rows"):
            _require(counts[name].get(split, 0) > 0,
                     f"Split {split} is empty in {name}")
    manifest = {
        "format": PROCEDURAL_MANIFEST_FORMAT,
        "version": VERSION,
        "seed": config["seed"],
        "status": "staged_curriculum_data_not_admitted",
        "config_fingerprint": fingerprint(config),
        "renderer": dict(config["procedural"]["renderer"]),
        "provenance": {
            "origin": "internal_owned_procedural_generation",
            "generation": "curriculum_spatial_grounding_abstention_v1",
            "source_id": "replica_lab:procedural_scene_png_v1",
            "external_assets": 0,
            "note": ("Fresh worlds for the curriculum. The audit split's "
                     "object descriptors are disjoint from "
                     "data/vision_dependence_v2 and from curriculum train/dev; "
                     "curriculum train may reuse diagnostic train descriptors "
                     "but never the diagnostic final split that is now "
                     "development evidence. No image bytes are reused."),
        },
        "tasks": list(ALL_TASKS),
        "split_tasks": {split: list(SPLIT_TASKS[split]) for split in SPLITS},
        "descriptor_pools": {split: [list(item) for item in pools[split]]
                            for split in SPLITS},
        "split_layouts": {split: list(_layouts(config, split))
                          for split in SPLITS},
        "holdout_policy": {
            "unit": "object_descriptor(color,shape)",
            "split_pools_disjoint": True,
            "audit_disjoint_from_diagnostic": True,
            "audit_disjoint_from_curriculum_train_dev": True,
            "world_combinations_disjoint": True,
            "policy": ("The audit split owns object descriptors and worlds "
                       "that never appear in curriculum train/dev or in the "
                       "admitted diagnostic corpus."),
        },
        "counts": counts,
        "label_coverage": coverage,
        "unknown_coverage": unknown_coverage,
        "unknown_label": UNKNOWN_ANSWER,
        "prompt_policy": {
            "unique_prompt_per_world": True,
            "unique_prompt_answer_pair_per_world": True,
            "cross_split_prompt_disjoint": True,
            "pair_sides_share_prompt": True,
            "split_specific_styles": {split: list(PROMPT_PREFIXES[split])
                                      for split in SPLITS},
        },
        "worlds": worlds,
        "pairs": pairs,
        "assets": assets,
        "cases": cases,
        "files": {
            "model_rows": {"path": "model_rows.jsonl",
                           "sha256": sha256_bytes(files["model_rows.jsonl"]),
                           "rows": len(rows)},
            "verifier_cases": {"path": "verifier_cases.jsonl",
                               "sha256": sha256_bytes(files["verifier_cases.jsonl"]),
                               "rows": len(cases)},
            "verifier_pairs": {"path": "verifier_pairs.jsonl",
                               "sha256": sha256_bytes(files["verifier_pairs.jsonl"]),
                               "rows": len(pairs)},
            "verifier_scenes": {"path": "verifier_scenes.jsonl",
                                "sha256": sha256_bytes(files["verifier_scenes.jsonl"]),
                                "rows": len(scenes)},
        },
    }
    manifest["generation_fingerprint"] = fingerprint({
        "format": PROCEDURAL_MANIFEST_FORMAT,
        "seed": config["seed"],
        "config": fingerprint(config),
        "assets": [asset["sha256"] for asset in assets],
        "scenes": [scene["scene_sha256"] for scene in scenes],
        "cases": [fingerprint({"case_id": case["case_id"],
                               "question": case["question"],
                               "answer": case["answer"],
                               "check": case["check"]})
                  for case in cases],
        "worlds": [world["world_id"] for world in worlds],
    })
    files["manifest.json"] = _json_bytes(manifest)
    return {"files": files, "manifest": manifest, "assets": assets,
            "cases": cases, "rows": rows, "pairs": pairs, "scenes": scenes,
            "worlds": worlds, "counts": counts, "label_coverage": coverage,
            "unknown_coverage": unknown_coverage}


def _split_entries(manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "worlds": [world for world in manifest["worlds"]
                   if world["split"] == split],
        "assets": [asset for asset in manifest["assets"]
                   if asset["split"] == split],
        "cases": [case for case in manifest["cases"]
                  if case["split"] == split],
        "pairs": [pair for pair in manifest["pairs"]
                  if pair["split"] == split],
    }


def _seal_payload_rows(rows: Sequence[Mapping[str, Any]],
                       labels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case = {label["case_id"]: label for label in labels}
    _require(set(by_case) == {row["case_id"] for row in rows},
             "Seal rows and labels differ")
    return [{"case_id": row["case_id"],
             "image_sha256": row["image_sha256"],
             "question": row["question"],
             "answer": by_case[row["case_id"]]["answer"]}
            for row in sorted(rows, key=lambda item: item["case_id"])]


def compute_seals(*, external_train_rows, external_train_labels,
                  external_dev_rows, external_dev_labels,
                  diag_train_rows, diag_train_labels,
                  diag_dev_rows, diag_dev_labels,
                  curriculum_train_rows, curriculum_train_labels,
                  curriculum_dev_rows, curriculum_dev_labels,
                  curriculum_audit_rows, curriculum_audit_labels,
                  audit_assets, final_pointer) -> dict[str, Any]:
    train_seal = fingerprint({
        "external_train": _seal_payload_rows(external_train_rows,
                                             external_train_labels),
        "diagnostic_train": _seal_payload_rows(diag_train_rows,
                                               diag_train_labels),
        "curriculum_train": _seal_payload_rows(curriculum_train_rows,
                                               curriculum_train_labels),
    })
    dev_seal = fingerprint({
        "external_development": _seal_payload_rows(external_dev_rows,
                                                   external_dev_labels),
        "diagnostic_development": _seal_payload_rows(diag_dev_rows,
                                                     diag_dev_labels),
        "curriculum_development": _seal_payload_rows(curriculum_dev_rows,
                                                     curriculum_dev_labels),
    })
    audit_seal = fingerprint({
        "curriculum_audit": _seal_payload_rows(curriculum_audit_rows,
                                              curriculum_audit_labels),
        "assets": [{"asset_id": asset["asset_id"],
                    "sha256": asset["sha256"],
                    "world_id": asset["world_id"]}
                   for asset in sorted(audit_assets,
                                       key=lambda item: item["asset_id"])],
        "policy": "scored_once_after_frozen_dev_report",
    })
    pointer_fields = ("manifest", "manifest_sha256", "seal_sha256", "rows",
                      "row_ids_sha256", "consumed", "policy")
    pointer = {name: final_pointer[name] for name in pointer_fields}
    return {
        "format": SEALS_FORMAT,
        "version": VERSION,
        "train_seal_sha256": train_seal,
        "dev_seal_sha256": dev_seal,
        "audit_seal_sha256": audit_seal,
        "external_final_seal_sha256": final_pointer["seal_sha256"],
        "external_final_pointer_sha256": fingerprint(pointer),
        "external_final_consumed": False,
        "audit_scored": False,
        "roles": {
            "train": ["external_train", "procedural_source_train",
                      "procedural_train"],
            "dev": ["external_development", "procedural_source_development",
                    "procedural_dev"],
            "audit": ["procedural_audit"],
            "sealed": ["external_final_eval_untouched"],
        },
        "policy": ("Train updates use train rows only. Dev is for bounded "
                   "selection. Audit is scored once after the selected "
                   "checkpoint is frozen. External final stays untouched."),
    }


# ---------------------------------------------------------------------------
# Curriculum assembly
# ---------------------------------------------------------------------------


def _external_label_rows(rows: Sequence[Mapping[str, Any]], split: str,
                         directory: str) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]]]:
    model_rows = []
    label_rows = []
    for row in rows:
        image = f"{directory}/images/{row['upstream_id']}.jpg"
        model_rows.append({"case_id": row["case_id"], "image": image,
                           "image_sha256": row["image_sha256"],
                           "question": row["question"]})
        label_rows.append({"case_id": row["case_id"], "split": split,
                           "answer": row["response"],
                           "target_sha256": row["target_sha256"],
                           "row_sha256": row["row_sha256"],
                           "group_key": row["group_key"],
                           "label_visibility": "verifier_only"})
    return model_rows, label_rows


def _load_procedural_source(project_root: Path,
                            config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["procedural_source"]
    root = _safe_relative(project_root, source["root"])
    _require(root.is_dir(), f"Diagnostic source is missing: {root}")
    for name, relative in (("protocol_sha256", "protocol.json"),
                           ("manifest_sha256", "manifest.json"),
                           ("model_rows_sha256", "model_rows.jsonl"),
                           ("verifier_cases_sha256", "verifier_cases.jsonl")):
        path = root / relative
        _require(path.is_file(), f"Diagnostic source file is missing: {path}")
        _require(sha256_file(path) == source[name],
                 f"Diagnostic source bytes changed: {relative}")
    manifest = _load_json(root / "manifest.json")
    rows = _load_jsonl(root / "model_rows.jsonl")
    cases = {case["case_id"]: case for case in
             _load_jsonl(root / "verifier_cases.jsonl")}
    _require(len(rows) == len(cases) == manifest["counts"]["model_rows"],
             "Diagnostic source rows disagree")
    train = []
    development = []
    for row in rows:
        case = cases[row["case_id"]]
        item = {
            "case_id": f"diag:{row['case_id']}",
            "image_sha256": row["image_sha256"],
            "question": row["question"],
            "answer": case["answer"],
            "task": case["task"],
            "world_id": f"diag:{case['world_id']}",
            "image_bytes": (root / row["image"]).read_bytes(),
            "source_case_id": row["case_id"],
        }
        _require(sha256_bytes(item["image_bytes"]) == row["image_sha256"],
                 f"Diagnostic image bytes changed: {row['case_id']}")
        if case["split"] == source["train_split"]:
            train.append(item)
        elif case["split"] == source["development_split"]:
            development.append(item)
    _require(train, "Diagnostic train split is empty")
    _require(development, "Diagnostic development split is empty")
    return {"root": source["root"], "train": train,
            "development": development,
            "manifest_sha256": source["manifest_sha256"]}


def _diagnostic_rows(items: Sequence[Mapping[str, Any]], split: str,
                     directory: str) -> tuple[list[dict[str, Any]],
                                              list[dict[str, Any]],
                                              dict[str, bytes]]:
    model_rows = []
    labels = []
    files: dict[str, bytes] = {}
    for item in items:
        slug = item["source_case_id"].replace("/", "_").replace(":", "_")
        image = f"{directory}/images/{slug}.png"
        files[image] = item["image_bytes"]
        model_rows.append({"case_id": item["case_id"], "image": image,
                           "image_sha256": item["image_sha256"],
                           "question": item["question"]})
        labels.append({"case_id": item["case_id"], "split": split,
                       "answer": item["answer"], "task": item["task"],
                       "world_id": item["world_id"],
                       "source_case_id": item["source_case_id"],
                       "label_visibility": "verifier_only"})
    return model_rows, labels, files


def _procedural_rows(manifest: Mapping[str, Any], split: str
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _split_entries(manifest, split)
    asset_by_case = {asset["asset_id"]: asset for asset in entries["assets"]}
    model_rows = []
    labels = []
    for case in entries["cases"]:
        asset = asset_by_case[case["asset_id"]]
        model_rows.append({"case_id": case["case_id"],
                           "image": f"procedural/{asset['path']}",
                           "image_sha256": asset["sha256"],
                           "question": case["question"]})
        labels.append({"case_id": case["case_id"], "split": split,
                       "answer": case["answer"], "task": case["task"],
                       "kind": case["kind"], "world_id": case["world_id"],
                       "check": case["check"],
                       "label_visibility": "verifier_only"})
    return model_rows, labels


def _shuffle(values: Sequence[Mapping[str, Any]], seed: int, round_index: int,
             tag: str) -> list[Mapping[str, Any]]:
    items = list(values)
    draw = _Draw(seed, "schedule", tag, round_index)
    for index in range(len(items) - 1, 0, -1):
        other = draw._int(index + 1, f"shuffle:{index}")
        items[index], items[other] = items[other], items[index]
    return items


def _cycle_batches(rows: Sequence[Mapping[str, Any]], *, seed: int,
                   updates: int, batch: int, tag: str
                   ) -> list[list[Mapping[str, Any]]]:
    _require(bool(rows), "Cannot schedule an empty source")
    need = updates * batch
    rounds = math.ceil(need / len(rows))
    ordered: list[Mapping[str, Any]] = []
    for round_index in range(rounds):
        ordered.extend(_shuffle(rows, seed, round_index, tag))
    selected = ordered[:need]
    return [selected[start:start + batch]
            for start in range(0, need, batch)]


def select_probe_cases(dev_cases: Sequence[Mapping[str, Any]],
                       config: Mapping[str, Any]) -> list[dict[str, Any]]:
    per_task = math.ceil(int(config["curriculum"]["selection"]["probe_cases"])
                         / len(SPLIT_TASKS["dev"]))
    chosen: list[dict[str, Any]] = []
    for task in SPLIT_TASKS["dev"]:
        candidates = sorted(
            (row for row in dev_cases
             if row.get("task") == task and row["source"] == "curriculum_dev"),
            key=lambda row: row["case_id"])
        _require(candidates, f"No curriculum dev case for {task}")
        chosen.extend(candidates[:per_task])
    return chosen[:int(config["curriculum"]["selection"]["probe_cases"])]


def build_training_plan(config: Mapping[str, Any],
                        stage_a: Sequence[Mapping[str, Any]],
                        stage_b: Sequence[Mapping[str, Any]],
                        dev_cases: Sequence[Mapping[str, Any]]
                        ) -> dict[str, Any]:
    curriculum = config["curriculum"]
    stage_a_spec = curriculum["stage_a"]
    stage_b_spec = curriculum["stage_b"]
    a_batches = _cycle_batches(
        stage_a, seed=int(config["train"]["seed"]),
        updates=int(stage_a_spec["updates"]), batch=int(stage_a_spec["batch"]),
        tag="stage_a")
    b_batches = _cycle_batches(
        stage_b, seed=int(config["train"]["seed"]),
        updates=int(stage_b_spec["updates"]), batch=int(stage_b_spec["batch"]),
        tag="stage_b")
    probe = config["lr_probe"]
    probe_batches = _cycle_batches(
        stage_b, seed=int(config["train"]["seed"]),
        updates=int(probe["updates"]), batch=int(probe["batch"]),
        tag="probe")
    _require(len(probe_batches) == int(probe["updates"]),
             "Probe schedule differs from its declaration")
    stages = [
        {"stage": "A", "start_update": 1, "updates": len(a_batches),
         "batch": int(stage_a_spec["batch"]),
         "batches": [[row["case_id"] for row in batch] for batch in a_batches]},
        {"stage": "B", "start_update": len(a_batches) + 1,
         "updates": len(b_batches), "batch": int(stage_b_spec["batch"]),
         "batches": [[row["case_id"] for row in batch] for batch in b_batches]},
    ]
    example_count = sum(len(batch) for group in stages
                        for batch in group["batches"])
    _require(example_count == curriculum["total_examples"],
             "Training plan example count differs from the declaration")
    _require(len(a_batches) + len(b_batches) == curriculum["total_updates"],
             "Training plan update count differs from the declaration")
    lr_candidates = probe["candidates"]
    probe_updates = len(ARMS) * len(lr_candidates["muon8bit"]) * len(probe_batches)
    schedule = {
        "format": SCHEDULE_FORMAT,
        "version": VERSION,
        "seed": config["train"]["seed"],
        "schedule_version": curriculum["schedule_version"],
        "stages": stages,
        "updates": len(a_batches) + len(b_batches),
        "examples": example_count,
        "probe": {
            "updates_per_candidate": len(probe_batches),
            "batch": int(probe["batch"]),
            "batches": [[row["case_id"] for row in batch]
                        for batch in probe_batches],
            "candidates": {arm: [float(item) for item in values]
                           for arm, values in lr_candidates.items()},
            "arms": list(ARMS),
            "total_probe_updates": probe_updates,
            "total_probe_examples": (len(ARMS)
                                    * len(lr_candidates["muon8bit"])
                                    * len(probe_batches)
                                    * int(probe["batch"])),
            "measure": probe["measure"], "tie_break": probe["tie_break"],
            "selection_split": probe["split"],
        },
        "dev_probe_case_ids": [case["case_id"] for case in dev_cases],
        "response_only_labels": True,
    }
    schedule["schedule_sha256"] = fingerprint({
        "stages": stages,
        "probe": schedule["probe"],
        "dev_probe_case_ids": schedule["dev_probe_case_ids"],
        "seed": schedule["seed"],
    })
    scheduled = {case_id for group in stages for batch in group["batches"]
                 for case_id in batch}
    allowed = {row["case_id"] for row in stage_a} | {
        row["case_id"] for row in stage_b}
    _require(scheduled <= allowed,
             "Training schedule references a non-train case")
    return schedule


def build_arm_declarations(config: Mapping[str, Any]) -> dict[str, Any]:
    muon = config["arms"]["muon8bit"]
    adamw = config["arms"]["adamw_control"]
    arms = {
        "muon8bit": {
            "arm": "muon8bit",
            "route": muon["route"],
            "groups": [
                {"group_name": "vision_projector_muon8bit",
                 "algorithm": "muon8bit",
                 "param_names": list(PROJECTOR_2D_TENSORS),
                 "lr": float(muon["matrix_lr"]),
                 "lr_role": "matrix_route",
                 "momentum": float(muon["momentum"]),
                 "ns_steps": int(muon["ns_steps"]),
                 "block_size": int(muon["block_size"]),
                 "weight_decay": float(muon["weight_decay"])},
                {"group_name": "vision_projector_adamw",
                 "algorithm": "adamw",
                 "param_names": list(PROJECTOR_1D_TENSORS),
                 "lr": float(muon["aux_lr"]),
                 "lr_role": "fp32_adamw_aux",
                 "betas": [0.9, 0.999], "eps": float(muon["eps"]),
                 "weight_decay": float(muon["weight_decay"])},
            ],
            "lr_probe_candidates": [float(item) for item in
                                    config["lr_probe"]["candidates"]["muon8bit"]],
            "declared_lr": float(muon["matrix_lr"]),
        },
        "adamw_control": {
            "arm": "adamw_control",
            "route": adamw["route"],
            "groups": [
                {"group_name": "vision_projector_adamw_control",
                 "algorithm": "adamw",
                 "param_names": list(PROJECTOR_TENSORS),
                 "lr": float(adamw["declared_lr"]),
                 "lr_role": "declared_tuned_lr",
                 "betas": [0.9, 0.999], "eps": float(adamw["eps"]),
                 "weight_decay": float(adamw["weight_decay"])},
            ],
            "lr_probe_candidates": [float(item) for item in
                                    adamw["lr_candidates"]],
            "declared_lr": float(adamw["declared_lr"]),
        },
    }
    route_names = {arm: sorted(name for group in data["groups"]
                               for name in group["param_names"])
                   for arm, data in arms.items()}
    _require(all(names == sorted(PROJECTOR_TENSORS)
                 for names in route_names.values()),
             "Arm route inventories must cover every projector tensor once")
    muon_alg = {name: group["algorithm"]
                for group in arms["muon8bit"]["groups"]
                for name in group["param_names"]}
    for name in PROJECTOR_TENSORS:
        expected = "muon8bit" if name in PROJECTOR_2D_TENSORS else "adamw"
        _require(muon_alg[name] == expected,
                 f"Muon arm routes {name} to the wrong algorithm")
    for group in arms["adamw_control"]["groups"]:
        _require(group["algorithm"] == "adamw",
                 "AdamW control must route every tensor to AdamW")
    probe = config["lr_probe"]
    probe_updates_per_arm = (len(probe["candidates"]["muon8bit"])
                             * int(probe["updates"]))
    return {
        "format": ARMS_FORMAT,
        "version": VERSION,
        "seed": config["train"]["seed"],
        "init_seed": config["train"]["init_seed"],
        "init_recipe": config["train"]["init_recipe"],
        "projector_hidden": PROJECTOR_HIDDEN,
        "projector_tensors": list(PROJECTOR_TENSORS),
        "projector_parameter_count": PROJECTOR_PARAMETER_COUNT,
        "trainable_scope": "vision_projector.",
        "frozen_prefixes": ["base.", "record_writer.", "memory_ports.",
                            "sense_ports.", "reranker."],
        "arms": arms,
        "equal_compute": {
            "updates_per_arm": int(config["curriculum"]["total_updates"]),
            "examples_per_arm": int(config["curriculum"]["total_examples"]),
            "probe_updates_per_arm": probe_updates_per_arm,
            "probe_examples_per_arm": (probe_updates_per_arm
                                       * int(probe["batch"])),
            "same_schedule_for_both_arms": True,
            "same_initialization_for_both_arms": True,
            "response_only_labels": True,
        },
    }


def _question_leaks(question: str, answer: str) -> bool:
    """A question must not contain its gold answer as a standalone token."""
    if not answer or not answer.strip():
        return False
    pattern = r"(?<!\w)" + re.escape(answer.casefold()) + r"(?!\w)"
    return re.search(pattern, question.casefold()) is not None


def _build_training_examples(external_train: Sequence[Mapping[str, Any]],
                             diag_train: Sequence[Mapping[str, Any]],
                             curriculum_train: Sequence[Mapping[str, Any]]
                             ) -> tuple[list[dict[str, Any]],
                                        list[dict[str, Any]]]:
    stage_a = []
    for row in external_train:
        stage_a.append({
            "case_id": row["case_id"], "source": "external_train",
            "image": row["image"], "image_sha256": row["image_sha256"],
            "question": row["question"], "response": row["response"],
            "split": "train",
        })
    stage_b = []
    for row in diag_train:
        stage_b.append({"case_id": row["case_id"], "image": row["image"],
                        "image_sha256": row["image_sha256"],
                        "question": row["question"],
                        "response": row["answer"], "task": row["task"],
                        "world_id": row["world_id"],
                        "source": "procedural_source_train", "split": "train"})
    for row in curriculum_train:
        stage_b.append({"case_id": row["case_id"], "image": row["image"],
                        "image_sha256": row["image_sha256"],
                        "question": row["question"],
                        "response": row["answer"], "task": row["task"],
                        "kind": row.get("kind"),
                        "world_id": row["world_id"],
                        "source": "procedural_train", "split": "train"})
    _require(stage_a, "Stage A examples are empty")
    _require(stage_b, "Stage B examples are empty")
    for row in stage_a + stage_b:
        _require(row["response"], f"Training example lacks a response: {row['case_id']}")
        _require(not _question_leaks(row["question"], row["response"]),
                 f"Training question leaks its response: {row['case_id']}")
        _require(row["response"].casefold() not in row["question"].casefold(),
                 f"Training response appears in the question: {row['case_id']}")
    return stage_a, stage_b


def build_curriculum(config: Mapping[str, Any], project_root: str | Path
                     ) -> dict[str, Any]:
    config = validate_config(config)
    project_root = Path(project_root)
    external = load_external_source(project_root, config)
    procedural = build_procedural(config)
    diagnostic = _load_procedural_source(project_root, config)
    procedure_manifest = procedural["manifest"]
    files: dict[str, bytes] = {}

    # Copy admitted external train/development images into the curriculum so
    # the prepared corpus is self-contained. The sealed final split is never
    # copied or read into any managed split.
    for name, directory, split in (
            ("train", "external/train", "train"),
            ("development", "external/development", "dev")):
        for row in external[name]:
            files[f"{directory}/images/{row['upstream_id']}.jpg"] = row["image_bytes"]
    external_train_rows, external_train_labels = _external_label_rows(
        external["train"], "train", "external/train")
    external_dev_rows, external_dev_labels = _external_label_rows(
        external["development"], "dev", "external/development")
    files["stage_a/train/model_rows.jsonl"] = _jsonl_bytes(external_train_rows)
    files["stage_a/train/labels.jsonl"] = _jsonl_bytes(external_train_labels)
    files["stage_a/development/model_rows.jsonl"] = _jsonl_bytes(
        external_dev_rows)
    files["stage_a/development/labels.jsonl"] = _jsonl_bytes(
        external_dev_labels)
    final_pointer = dict(external["final_pointer"])
    files["stage_a/final_sealed.json"] = _json_bytes({
        "format": "replica.vision_curriculum.external_final_pointer.v1",
        "sealed": True,
        **final_pointer,
        "note": ("Pointer only. No final images, captions or rows are copied "
                 "or consumed by training, development or scoring."),
    })

    diag_train_rows, diag_train_labels, diag_train_files = _diagnostic_rows(
        diagnostic["train"], "train", "diagnostic/train")
    diag_dev_rows, diag_dev_labels, diag_dev_files = _diagnostic_rows(
        diagnostic["development"], "dev", "diagnostic/dev")
    files.update(diag_train_files)
    files.update(diag_dev_files)
    files["stage_b/diagnostic_train/model_rows.jsonl"] = _jsonl_bytes(
        diag_train_rows)
    files["stage_b/diagnostic_train/labels.jsonl"] = _jsonl_bytes(
        diag_train_labels)
    files["stage_b/diagnostic_dev/model_rows.jsonl"] = _jsonl_bytes(diag_dev_rows)
    files["stage_b/diagnostic_dev/labels.jsonl"] = _jsonl_bytes(diag_dev_labels)

    stage_b_rows = {}
    stage_b_labels = {}
    for split in SPLITS:
        rows, labels = _procedural_rows(procedure_manifest, split)
        for row in rows:
            label = next(label for label in labels
                         if label["case_id"] == row["case_id"])
            _require(not _question_leaks(row["question"], label["answer"]),
                     f"Procedural question leaks its label: {row['case_id']}")
        stage_b_rows[split] = rows
        stage_b_labels[split] = labels
        files[f"stage_b/{split}/model_rows.jsonl"] = _jsonl_bytes(rows)
        files[f"stage_b/{split}/labels.jsonl"] = _jsonl_bytes(labels)
    for relative, data in procedural["files"].items():
        files[f"procedural/{relative}"] = data

    curriculum_train = [{"case_id": row["case_id"], "image": row["image"],
                         "image_sha256": row["image_sha256"],
                         "question": row["question"],
                         "answer": label["answer"],
                         "task": label["task"], "kind": label["kind"],
                         "world_id": label["world_id"]}
                        for row, label in zip(stage_b_rows["train"],
                                              stage_b_labels["train"])]
    diag_train_examples = [{"case_id": row["case_id"],
                            "image": row["image"],
                            "image_sha256": row["image_sha256"],
                            "question": row["question"],
                            "answer": label["answer"],
                            "task": label["task"],
                            "world_id": label["world_id"]}
                           for row, label in zip(diag_train_rows,
                                                 diag_train_labels)]
    stage_a, stage_b = _build_training_examples(
        [{"case_id": row["case_id"], "image": row["image"],
          "image_sha256": row["image_sha256"],
          "question": row["question"], "response": label["answer"]}
         for row, label in zip(external_train_rows, external_train_labels)],
        diag_train_examples, curriculum_train)

    dev_cases: list[dict[str, Any]] = []
    for row, label in zip(stage_b_rows["dev"], stage_b_labels["dev"]):
        dev_cases.append({"case_id": row["case_id"], "source": "curriculum_dev",
                          "image": row["image"],
                          "image_sha256": row["image_sha256"],
                          "question": row["question"], "task": label["task"],
                          "world_id": label["world_id"]})
    for row, label in zip(diag_dev_rows, diag_dev_labels):
        dev_cases.append({"case_id": row["case_id"],
                          "source": "procedural_source",
                          "image": row["image"],
                          "image_sha256": row["image_sha256"],
                          "question": row["question"], "task": label["task"],
                          "world_id": label["world_id"]})
    dev_cases.sort(key=lambda row: (row["source"], row["case_id"]))
    probe_cases = select_probe_cases(dev_cases, config)
    schedule = build_training_plan(config, stage_a, stage_b, probe_cases)
    arms = build_arm_declarations(config)
    files["schedule.json"] = _json_bytes(schedule)
    files["arms.json"] = _json_bytes(arms)

    audit_entries = _split_entries(procedure_manifest, "audit")
    seals = compute_seals(
        external_train_rows=external_train_rows,
        external_train_labels=external_train_labels,
        external_dev_rows=external_dev_rows,
        external_dev_labels=external_dev_labels,
        diag_train_rows=diag_train_rows, diag_train_labels=diag_train_labels,
        diag_dev_rows=diag_dev_rows, diag_dev_labels=diag_dev_labels,
        curriculum_train_rows=stage_b_rows["train"],
        curriculum_train_labels=stage_b_labels["train"],
        curriculum_dev_rows=stage_b_rows["dev"],
        curriculum_dev_labels=stage_b_labels["dev"],
        curriculum_audit_rows=stage_b_rows["audit"],
        curriculum_audit_labels=stage_b_labels["audit"],
        audit_assets=audit_entries["assets"], final_pointer=final_pointer)
    files["seals.json"] = _json_bytes(seals)

    files["procedural/audit_manifest.json"] = _json_bytes({
        "format": "replica.vision_curriculum.audit_manifest.v1",
        "version": VERSION,
        "seal_sha256": seals["audit_seal_sha256"],
        "split": "audit",
        "counts": {name: len(values) for name, values in audit_entries.items()},
        "descriptor_pool": [list(item) for item in _pool(config, "audit")],
        "disjoint_from": ["curriculum_train", "curriculum_dev",
                          "diagnostic_train", "diagnostic_dev",
                          "diagnostic_final_as_dev"],
        "worlds": audit_entries["worlds"],
        "assets": audit_entries["assets"],
        "cases": audit_entries["cases"],
        "usage": "scored_once_after_frozen_dev_report",
    })
    files["procedural/development_manifest.json"] = _json_bytes({
        "format": "replica.vision_curriculum.development_manifest.v1",
        "version": VERSION,
        "seal_sha256": seals["dev_seal_sha256"],
        "counts": {name: len(values)
                   for name, values in _split_entries(procedure_manifest,
                                                      "dev").items()},
        "descriptor_pool": [list(item) for item in _pool(config, "dev")],
        "usage": "bounded_development_selection_only",
    })

    counts = {
        "external": {"train": len(external["train"]),
                     "development": len(external["development"]),
                     "final_sealed": final_pointer["rows"]},
        "diagnostic_source": {"train": len(diag_train_rows),
                              "development": len(diag_dev_rows)},
        "procedural": procedural["counts"],
        "stage_a_examples": len(stage_a),
        "stage_b_examples": len(stage_b),
        "train_examples": len(stage_a) + len(stage_b),
        "dev_cases": len(dev_cases),
        "dev_procedural_rows": len(dev_cases),
        "external_caption_rows": len(external["development"]),
        "audit_cases": len(audit_entries["cases"]),
        "schedule_updates": schedule["updates"],
        "schedule_examples": schedule["examples"],
        "probe_cases": len(probe_cases),
        "probe_updates": schedule["probe"]["total_probe_updates"],
        "probe_examples": schedule["probe"]["total_probe_examples"],
    }
    manifest = {
        "format": CURRICULUM_MANIFEST_FORMAT,
        "version": VERSION,
        "seed": config["seed"],
        "status": "staged_not_admitted",
        "config_fingerprint": fingerprint(config),
        "purpose": config["purpose"],
        "provenance": {
            "external": {"root": external["root"],
                         "intake": "external_caption_tranche_20261025_v2",
                         "terms": "internal_research_research_use_only",
                         "redistribution": "prohibited"},
            "procedural_source": {
                "root": diagnostic["root"],
                "origin": "internal_owned_procedural_generation",
                "train_split": "train",
                "development_split":
                    "final_already_scored_reused_as_development"},
            "curriculum_procedural": {
                "origin": "internal_owned_procedural_generation",
                "generation": "curriculum_spatial_grounding_abstention_v1",
                "descriptor_holdout": "audit_disjoint_from_all_prior_splits"},
        },
        "splits": {
            "train": ["external_train", "diagnostic_train",
                      "curriculum_procedural_train"],
            "dev": ["external_development",
                    "diagnostic_final_as_development",
                    "curriculum_procedural_dev"],
            "audit": ["curriculum_procedural_audit"],
            "sealed": ["external_final_eval_untouched"],
        },
        "counts": counts,
        "unknown_coverage": procedural["unknown_coverage"],
        "label_coverage": procedural["label_coverage"],
        "seals": {"train": seals["train_seal_sha256"],
                  "dev": seals["dev_seal_sha256"],
                  "audit": seals["audit_seal_sha256"],
                  "external_final": seals["external_final_seal_sha256"],
                  "external_final_pointer": seals[
                      "external_final_pointer_sha256"]},
        "arms": {"order": list(ARMS),
                 "equal_compute": arms["equal_compute"],
                 "projector_parameter_count": PROJECTOR_PARAMETER_COUNT},
        "schedule": {"path": "schedule.json",
                     "sha256": sha256_bytes(files["schedule.json"]),
                     "updates": schedule["updates"],
                     "examples": schedule["examples"],
                     "schedule_sha256": schedule["schedule_sha256"]},
        "files": {},
    }
    for relative, data in sorted(files.items()):
        if relative == "manifest.json":
            continue
        manifest["files"][relative] = {"sha256": sha256_bytes(data),
                                       "bytes": len(data)}
    files["manifest.json"] = _json_bytes(manifest)
    return {"files": files, "manifest": manifest, "external": external,
            "procedural": procedure_manifest, "diagnostic": diagnostic,
            "schedule": schedule, "arms": arms, "seals": seals,
            "counts": counts, "stage_a": stage_a, "stage_b": stage_b,
            "dev_cases": dev_cases}


def write_curriculum(root: str | Path, built: Mapping[str, Any]) -> None:
    root = Path(root)
    for relative in sorted(built["files"]):
        _write_exact(root / relative, built["files"][relative])


# ---------------------------------------------------------------------------
# Dataset view and audit
# ---------------------------------------------------------------------------


class CurriculumDataset:
    """Verified read-only view over a prepared curriculum."""

    def __init__(self, root: Path, manifest: Mapping[str, Any],
                 procedural_manifest: Mapping[str, Any],
                 schedule: Mapping[str, Any], arms: Mapping[str, Any],
                 seals: Mapping[str, Any],
                 external_train_rows: Sequence[Mapping[str, Any]],
                 external_train_labels: Sequence[Mapping[str, Any]],
                 external_dev_rows: Sequence[Mapping[str, Any]],
                 external_dev_labels: Sequence[Mapping[str, Any]],
                 diag_train_rows: Sequence[Mapping[str, Any]],
                 diag_train_labels: Sequence[Mapping[str, Any]],
                 diag_dev_rows: Sequence[Mapping[str, Any]],
                 diag_dev_labels: Sequence[Mapping[str, Any]],
                 stage_b_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                 stage_b_labels: Mapping[str, Sequence[Mapping[str, Any]]],
                 procedural_rows: Sequence[Mapping[str, Any]],
                 procedural_cases: Sequence[Mapping[str, Any]]):
        self.root = Path(root)
        self.manifest = manifest
        self.procedural_manifest = procedural_manifest
        self.schedule = schedule
        self.arms = arms
        self.seals = seals
        self._external_train = list(external_train_rows)
        self._external_train_labels = {row["case_id"]: row
                                       for row in external_train_labels}
        self._external_dev = list(external_dev_rows)
        self._external_dev_labels = {row["case_id"]: row
                                     for row in external_dev_labels}
        self._diag_train = list(diag_train_rows)
        self._diag_train_labels = {row["case_id"]: row
                                   for row in diag_train_labels}
        self._diag_dev = list(diag_dev_rows)
        self._diag_dev_labels = {row["case_id"]: row
                                 for row in diag_dev_labels}
        self._stage_rows = {split: list(rows)
                            for split, rows in stage_b_rows.items()}
        self._stage_labels = {split: {row["case_id"]: row for row in rows}
                              for split, rows in stage_b_labels.items()}
        self.procedural_rows = {row["case_id"]: row for row in procedural_rows}
        self.procedural_cases = {case["case_id"]: case
                                 for case in procedural_cases}
        self.seal_receipts: dict[str, str] = {}

    @property
    def stage_b_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {split: list(rows) for split, rows in self._stage_rows.items()}

    @property
    def stage_b_labels(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {split: dict(rows) for split, rows in self._stage_labels.items()}

    @property
    def pins(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest["_manifest_sha256"],
            "seals": {name: self.seals[name] for name in
                      ("train_seal_sha256", "dev_seal_sha256",
                       "audit_seal_sha256", "external_final_seal_sha256")},
            "files": {name: pin["sha256"]
                      for name, pin in self.manifest["files"].items()},
            "counts": self.manifest["counts"],
        }

    def _resolve(self, image: str) -> str:
        return str(self.root / image)

    @staticmethod
    def _merge(rows: Sequence[Mapping[str, Any]],
               labels: Mapping[str, Mapping[str, Any]], *,
               split: str, source: str) -> list[dict[str, Any]]:
        merged = []
        for row in rows:
            label = labels[row["case_id"]]
            merged.append({**row, "split": split, "source": source,
                           "response": label["answer"],
                           "task": label.get("task"),
                           "kind": label.get("kind"),
                           "world_id": label.get("world_id")})
        return merged

    def _with_paths(self, rows: Sequence[Mapping[str, Any]]
                    ) -> list[dict[str, Any]]:
        return [{**row, "image_path": self._resolve(row["image"])}
                for row in rows]

    def stage_a_external(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._external_train, self._external_train_labels,
            split="train", source="external_train"))

    def diagnostic_train(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._diag_train, self._diag_train_labels,
            split="train", source="procedural_source_train"))

    def curriculum_train(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._stage_rows["train"], self._stage_labels["train"],
            split="train", source="procedural_train"))

    def external_development(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._external_dev, self._external_dev_labels,
            split="dev", source="external_development"))

    def diagnostic_development(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._diag_dev, self._diag_dev_labels,
            split="dev", source="procedural_source"))

    def curriculum_development(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._stage_rows["dev"], self._stage_labels["dev"],
            split="dev", source="curriculum_dev"))

    def dev_rows(self) -> list[dict[str, Any]]:
        return self.diagnostic_development() + self.curriculum_development()

    def audit_rows(self) -> list[dict[str, Any]]:
        return self._with_paths(self._merge(
            self._stage_rows["audit"], self._stage_labels["audit"],
            split="audit", source="curriculum_audit"))

    def training_examples(self) -> list[dict[str, Any]]:
        """Only the declared train rows, never dev/audit/development rows."""
        examples = self.stage_a_external()
        examples.extend(self.diagnostic_train())
        examples.extend(self.curriculum_train())
        return examples

    def managed_case_ids(self) -> set[str]:
        identifiers = {row["case_id"] for row in
                       self._external_train + self._external_dev
                       + self._diag_train + self._diag_dev}
        for rows in self._stage_rows.values():
            identifiers.update(row["case_id"] for row in rows)
        return identifiers

    def managed_image_hashes(self) -> set[str]:
        hashes = {row["image_sha256"] for row in
                  self._external_train + self._external_dev
                  + self._diag_train + self._diag_dev}
        for rows in self._stage_rows.values():
            hashes.update(row["image_sha256"] for row in rows)
        return hashes

    def generation_cases(self, split: str) -> list[dict[str, Any]]:
        """Evaluation cases with labels; questions stay label-free."""
        if split == "dev":
            rows = self.diagnostic_development() + self.curriculum_development()
        elif split == "audit":
            rows = self.audit_rows()
        else:
            raise VisionCurriculumError(
                f"Only dev or audit may be scored, not {split}")
        return [{"case_id": row["case_id"],
                 "world_id": row["world_id"],
                 "task": row["task"], "kind": row["kind"],
                 "label": row["response"], "split": split,
                 "source": row["source"],
                 "image": row["image"],
                 "image_sha256": row["image_sha256"],
                 "image_path": row["image_path"],
                 "question": row["question"]} for row in rows]

    def caption_cases(self) -> list[dict[str, Any]]:
        """External development rows for the narrow caption diagnostic."""
        return [{"case_id": row["case_id"], "image_path": row["image_path"],
                 "image_sha256": row["image_sha256"],
                 "question": row["question"],
                 "label": self._external_dev_labels[row["case_id"]]["answer"]}
                for row in self.external_development()]


def audit_dataset(root: str | Path) -> CurriculumDataset:
    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "Curriculum manifest is missing")
    manifest = _load_json(manifest_path)
    _require(manifest.get("format") == CURRICULUM_MANIFEST_FORMAT,
             "Curriculum manifest format differs")
    _require(manifest.get("version") == VERSION,
             "Curriculum manifest version differs")
    _require(manifest.get("status") == "staged_not_admitted",
             "Curriculum manifest status differs")
    for relative, pin in manifest["files"].items():
        path = _safe_relative(root, relative)
        _require(path.is_file(), f"Curriculum file is missing: {relative}")
        _require(sha256_file(path) == pin["sha256"],
                 f"Curriculum file bytes changed: {relative}")
        _require(path.stat().st_size == pin["bytes"],
                 f"Curriculum file size changed: {relative}")
    procedural_manifest = _load_json(root / "procedural/manifest.json")
    _require(procedural_manifest.get("format") == PROCEDURAL_MANIFEST_FORMAT,
             "Procedural manifest format differs")
    for name, pin in procedural_manifest["files"].items():
        path = root / "procedural" / pin["path"]
        _require(path.is_file(), f"Procedural file is missing: {name}")
        _require(sha256_file(path) == pin["sha256"],
                 f"Procedural file bytes changed: {name}")
        if "rows" in pin:
            _require(len(_load_jsonl(path)) == pin["rows"],
                     f"Procedural row count changed: {name}")
    schedule = _load_json(root / "schedule.json")
    arms = _load_json(root / "arms.json")
    seals = _load_json(root / "seals.json")
    _require(schedule.get("format") == SCHEDULE_FORMAT,
             "Schedule format differs")
    _require(arms.get("format") == ARMS_FORMAT, "Arms format differs")
    _require(seals.get("format") == SEALS_FORMAT, "Seals format differs")
    _require(schedule["schedule_sha256"] == manifest["schedule"]["schedule_sha256"],
             "Schedule hash differs from the manifest")
    _require(arms["equal_compute"]["updates_per_arm"] == schedule["updates"],
             "Arm updates differ from the schedule")
    _require(arms["equal_compute"]["examples_per_arm"] == schedule["examples"],
             "Arm examples differ from the schedule")

    stage_a_rows = _load_jsonl(root / "stage_a/train/model_rows.jsonl")
    stage_a_labels = _load_jsonl(root / "stage_a/train/labels.jsonl")
    external_dev_rows = _load_jsonl(
        root / "stage_a/development/model_rows.jsonl")
    external_dev_labels = _load_jsonl(
        root / "stage_a/development/labels.jsonl")
    diag_train_rows = _load_jsonl(
        root / "stage_b/diagnostic_train/model_rows.jsonl")
    diag_train_labels = _load_jsonl(
        root / "stage_b/diagnostic_train/labels.jsonl")
    diag_dev_rows = _load_jsonl(root / "stage_b/diagnostic_dev/model_rows.jsonl")
    diag_dev_labels = _load_jsonl(
        root / "stage_b/diagnostic_dev/labels.jsonl")
    stage_b_rows = {split: _load_jsonl(root / f"stage_b/{split}/model_rows.jsonl")
                    for split in SPLITS}
    stage_b_labels = {split: _load_jsonl(root / f"stage_b/{split}/labels.jsonl")
                      for split in SPLITS}
    procedural_rows = _load_jsonl(root / "procedural/model_rows.jsonl")
    procedural_cases = _load_jsonl(root / "procedural/verifier_cases.jsonl")

    for row in stage_a_rows:
        _require(set(row) == LOCAL_MODEL_ROW_FIELDS,
                 "Stage A model row fields differ")
    for row in external_dev_rows:
        _require(set(row) == LOCAL_MODEL_ROW_FIELDS,
                 "External development model row fields differ")
    for row in diag_train_rows + diag_dev_rows:
        _require(set(row) == LOCAL_MODEL_ROW_FIELDS,
                 "Diagnostic model row fields differ")
    for row in procedural_rows:
        _require(set(row) == MODEL_ROW_FIELDS,
                 f"Procedural model row fields differ: {row.get('case_id')}")
    for split, rows in stage_b_rows.items():
        for row in rows:
            _require(set(row) == LOCAL_MODEL_ROW_FIELDS,
                     f"Curriculum model row fields differ: {row['case_id']}")
    for labels in [stage_a_labels, external_dev_labels, diag_train_labels,
                   diag_dev_labels, *stage_b_labels.values()]:
        for row in labels:
            _require(row.get("label_visibility") == "verifier_only",
                     "Labels must be verifier-only")
    for case in procedural_cases:
        _require(case.get("label_visibility") == "verifier_only",
                 "Procedural labels must be verifier-only")

    for name, rows, labels in (
            ("diagnostic_train", diag_train_rows, diag_train_labels),
            ("diagnostic_dev", diag_dev_rows, diag_dev_labels),
            *[(f"curriculum_{split}", stage_b_rows[split],
               stage_b_labels[split]) for split in SPLITS]):
        _require({row["case_id"] for row in rows}
                 == {row["case_id"] for row in labels},
                 f"Model rows and labels differ for {name}")
    for split in SPLITS:
        unknown_kinds = {row.get("kind") for row in stage_b_labels[split]
                         if row["answer"] == UNKNOWN_ANSWER}
        _require(unknown_kinds == set(ABSTAIN_KINDS),
                 f"Unknown coverage differs for {split}: {sorted(unknown_kinds)}")
        for row in stage_b_rows[split]:
            label = next(item for item in stage_b_labels[split]
                         if item["case_id"] == row["case_id"])
            _require(not _question_leaks(row["question"], label["answer"]),
                     f"Stage B question leaks its answer: {row['case_id']}")
    _require(len(stage_a_labels) == 15, "External train labels must be 15")
    _require(len(external_dev_labels) == 7,
             "External development labels must be 7")

    # Seals are recomputed from the stored files and must match exactly.
    audit_entries = _split_entries(procedural_manifest, "audit")
    final_pointer = _load_json(root / "stage_a/final_sealed.json")
    recomputed = compute_seals(
        external_train_rows=stage_a_rows,
        external_train_labels=stage_a_labels,
        external_dev_rows=external_dev_rows,
        external_dev_labels=external_dev_labels,
        diag_train_rows=diag_train_rows, diag_train_labels=diag_train_labels,
        diag_dev_rows=diag_dev_rows, diag_dev_labels=diag_dev_labels,
        curriculum_train_rows=stage_b_rows["train"],
        curriculum_train_labels=stage_b_labels["train"],
        curriculum_dev_rows=stage_b_rows["dev"],
        curriculum_dev_labels=stage_b_labels["dev"],
        curriculum_audit_rows=stage_b_rows["audit"],
        curriculum_audit_labels=stage_b_labels["audit"],
        audit_assets=audit_entries["assets"], final_pointer=final_pointer)
    for name in ("train_seal_sha256", "dev_seal_sha256", "audit_seal_sha256",
                 "external_final_seal_sha256",
                 "external_final_pointer_sha256"):
        _require(seals[name] == recomputed[name],
                 f"Curriculum seal drifted: {name}")
    _require(seals["audit_scored"] is False,
             "Audit seal is already marked as scored")
    _require(seals["external_final_consumed"] is False,
             "External final split was consumed")
    _require(final_pointer.get("consumed") is False
             and final_pointer.get("sealed") is True,
             "External final pointer is not intact")
    audit_manifest = _load_json(root / "procedural/audit_manifest.json")
    _require(audit_manifest["seal_sha256"] == seals["audit_seal_sha256"],
             "Audit manifest seal differs")
    development_manifest = _load_json(
        root / "procedural/development_manifest.json")
    _require(development_manifest["seal_sha256"] == seals["dev_seal_sha256"],
             "Development manifest seal differs")

    # External managed rows are exactly the admitted train and development
    # splits; the sealed final rows were never written into any split.
    external_managed_ids = {row["case_id"] for row in stage_a_rows}
    external_dev_ids = {row["case_id"] for row in external_dev_rows}
    _require(len(external_managed_ids) == 15
             and len(external_dev_ids) == 7,
             "External managed split counts differ")
    _require(external_managed_ids.isdisjoint(external_dev_ids),
             "External train/development rows overlap")
    _require(len(external_managed_ids | external_dev_ids) == 22,
             "External managed rows exceed the admitted train/development rows")

    dataset = CurriculumDataset(
        root, manifest, procedural_manifest, schedule, arms, seals,
        stage_a_rows, stage_a_labels, external_dev_rows, external_dev_labels,
        diag_train_rows, diag_train_labels, diag_dev_rows, diag_dev_labels,
        stage_b_rows, stage_b_labels, procedural_rows, procedural_cases)
    dataset.manifest = dict(manifest)
    dataset.manifest["_manifest_sha256"] = sha256_file(manifest_path)
    dataset.seal_receipts = dict(recomputed)
    return dataset


def dataset_pins(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "Curriculum manifest is missing")
    manifest = _load_json(manifest_path)
    _require(manifest.get("format") == CURRICULUM_MANIFEST_FORMAT,
             "Curriculum manifest format differs")
    return {
        "manifest": {"path": "manifest.json",
                     "sha256": sha256_file(manifest_path)},
        "files": {name: {"path": name, "sha256": pin["sha256"]}
                  for name, pin in manifest["files"].items()},
        "seals": dict(manifest["seals"]),
        "counts": dict(manifest["counts"]),
    }


def verify_data_pins(pins: Mapping[str, Any], root: str | Path) -> None:
    root = Path(root)
    manifest = _safe_relative(root, pins["manifest"]["path"])
    _require(sha256_file(manifest) == pins["manifest"]["sha256"],
             "Curriculum manifest bytes changed")
    for name, pin in pins["files"].items():
        path = _safe_relative(root, pin["path"])
        _require(path.is_file(), f"Pinned curriculum file is missing: {name}")
        _require(sha256_file(path) == pin["sha256"],
                 f"Pinned curriculum file bytes changed: {name}")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def _implementation_pins(root: Path) -> dict[str, str]:
    pins = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        _require(path.is_file(), f"Implementation file is missing: {relative}")
        pins[relative] = sha256_file(path)
    return pins


def _validate_pin(pin: Any, name: str) -> None:
    _exact_keys(pin, {"path", "sha256"}, name)
    _require(isinstance(pin["path"], str) and pin["path"]
             and not Path(pin["path"]).is_absolute()
             and ".." not in Path(pin["path"]).parts,
             f"{name}.path is unsafe")
    _require(_is_sha256(pin["sha256"]), f"{name}.sha256 is invalid")


PROTOCOL_FIELDS = {
    "format", "version", "stage", "root", "purpose", "seed", "config_sha256",
    "data", "external", "procedural_source", "vision", "vision_digest",
    "vision_source", "model", "splits", "trainable_groups", "arms", "train",
    "schedule", "evaluation", "retention", "checkpoint_policy", "limits",
    "resource_policy", "action_limits", "stop_conditions",
    "implementation_sha256", "audit_policy", "claims",
}


def build_protocol(config: Mapping[str, Any], *, root: str | Path,
                   template: Mapping[str, Any],
                   data_root: str | Path | None = None) -> dict[str, Any]:
    config = validate_config(config)
    root = Path(root)
    relative_data = (config["data_root"] if data_root is None
                     else str(data_root))
    _require(isinstance(relative_data, str) and relative_data
             and not Path(relative_data).is_absolute()
             and ".." not in Path(relative_data).parts,
             "Protocol data root is unsafe")
    data_path = root / relative_data
    _require(data_path.is_dir(), "Prepared curriculum data root is missing")
    pins = dataset_pins(data_path)
    verify_data_pins(pins, data_path)
    manifest = _load_json(data_path / "manifest.json")
    _require(manifest.get("config_fingerprint") == fingerprint(config),
             "Prepared data config hash differs from the protocol config")
    _require(manifest.get("seed") == config["seed"],
             "Prepared data seed differs from the protocol seed")
    seals = _load_json(data_path / "seals.json")
    schedule = _load_json(data_path / "schedule.json")
    arms = _load_json(data_path / "arms.json")
    _require(schedule["schedule_sha256"] == manifest["schedule"]["schedule_sha256"],
             "Schedule hash differs from the data manifest")
    _require(arms["projector_hidden"] == PROJECTOR_HIDDEN,
             "Arm projector width differs from the declared 4096")
    _require(arms["projector_parameter_count"] == PROJECTOR_PARAMETER_COUNT,
             "Arm projector parameter count differs")
    vision_fields = ("encoder_repo", "encoder_revision",
                     "encoder_source_digest", "processor_digest",
                     "encoder_width", "image_size", "patch_size",
                     "max_visual_tokens", "projector_hidden", "feature_version",
                     "compute_dtype")
    _require(all(key in template["vision"] for key in vision_fields),
             "Vision template is incomplete")
    _require(int(template["vision"]["projector_hidden"]) == PROJECTOR_HIDDEN,
             "Vision template must declare the 4096-wide projector")
    _require(all(key in template for key in ("vision_digest",
                                             "vision_source", "model")),
             "Vision template lacks model or source pins")

    retention = config["retention"]
    for name in ("prompts", "labels"):
        pin = retention[name]
        path = _safe_relative(root, pin["path"])
        _require(path.is_file(), f"Retention {name} file is missing")
        _require(sha256_file(path) == pin["sha256"],
                 f"Retention {name} bytes changed")
    external_pointer = _load_json(data_path / "stage_a/final_sealed.json")
    pointer_fields = ("manifest", "manifest_sha256", "seal_sha256", "rows",
                      "row_ids_sha256", "consumed", "policy")
    external_pointer = {name: external_pointer[name] for name in pointer_fields}
    protocol = {
        "format": PROTOCOL_FORMAT,
        "version": VERSION,
        "stage": "curriculum_v1",
        "root": ".",
        "purpose": config["purpose"],
        "seed": config["seed"],
        "config_sha256": fingerprint(config),
        "data": {
            "root": relative_data,
            **pins,
            "unknown_coverage": manifest["unknown_coverage"],
            "label_coverage": manifest["label_coverage"],
        },
        "external": {
            "root": config["external"]["root"],
            "corpus": dict(config["external"]["corpus"]),
            "train": dict(config["external"]["manifests"]["train"]),
            "development": dict(config["external"]["manifests"]["development"]),
            "final_sealed": dict(config["external"]["manifests"]
                                 ["final_eval_untouched"]),
            "final_pointer": external_pointer,
            "consumed": False,
        },
        "procedural_source": {
            "root": config["procedural_source"]["root"],
            "protocol_sha256": config["procedural_source"]["protocol_sha256"],
            "manifest_sha256": config["procedural_source"]["manifest_sha256"],
            "model_rows_sha256": config["procedural_source"]["model_rows_sha256"],
            "verifier_cases_sha256": config["procedural_source"]
            ["verifier_cases_sha256"],
            "train_split": "train",
            "development_split":
                "final_already_scored_reused_as_development",
        },
        "vision": dict(template["vision"]),
        "vision_digest": template["vision_digest"],
        "vision_source": {
            **dict(template["vision_source"]),
            "scope": dict(config["vision_scope"]),
        },
        "model": dict(template["model"]),
        "splits": {
            "train": {"roles": ["external_train", "diagnostic_train",
                                "curriculum_procedural_train"],
                      "seal_sha256": seals["train_seal_sha256"],
                      "rows": manifest["counts"]["train_examples"]},
            "dev": {"roles": ["external_development",
                              "diagnostic_final_as_development",
                              "curriculum_procedural_dev"],
                    "seal_sha256": seals["dev_seal_sha256"],
                    "procedural_rows": manifest["counts"][
                        "dev_procedural_rows"],
                    "external_caption_rows": manifest["counts"][
                        "external_caption_rows"],
                    "rows": (manifest["counts"]["dev_procedural_rows"]
                             + manifest["counts"]["external_caption_rows"]),
                    "usage": "bounded_selection_only"},
            "audit": {"roles": ["curriculum_procedural_audit"],
                      "seal_sha256": seals["audit_seal_sha256"],
                      "rows": manifest["counts"]["audit_cases"],
                      "usage": "scored_once_after_freeze"},
            "external_final": {"roles": ["external_final_eval_untouched"],
                               "seal_sha256": seals[
                                   "external_final_seal_sha256"],
                               "rows": manifest["counts"]["external"][
                                   "final_sealed"],
                               "usage": "scored_once_after_freeze_untouched"},
        },
        "trainable_groups": {
            "scope": "vision_projector.",
            "tensors": list(PROJECTOR_TENSORS),
            "parameter_count": PROJECTOR_PARAMETER_COUNT,
            "muon8bit": {"param_names": list(PROJECTOR_2D_TENSORS),
                         "lr": float(config["arms"]["muon8bit"]["matrix_lr"]),
                         "momentum": float(config["arms"]["muon8bit"]["momentum"]),
                         "ns_steps": int(config["arms"]["muon8bit"]["ns_steps"]),
                         "block_size": int(config["arms"]["muon8bit"]["block_size"])},
            "adamw_aux": {"param_names": list(PROJECTOR_1D_TENSORS),
                          "lr": float(config["arms"]["muon8bit"]["aux_lr"]),
                          "betas": [0.9, 0.999], "eps": 1e-08},
            "adamw_control": {"param_names": list(PROJECTOR_TENSORS),
                              "lr": float(config["arms"]["adamw_control"]
                                          ["declared_lr"]),
                              "betas": [0.9, 0.999], "eps": 1e-08},
            "frozen": ["base.", "record_writer.", "memory_ports.",
                       "sense_ports.", "reranker."],
            "response_only_labels": True,
        },
        "arms": arms,
        "train": {
            "seed": int(config["train"]["seed"]),
            "init_seed": int(config["train"]["init_seed"]),
            "init_recipe": config["train"]["init_recipe"],
            "warm_start": None,
            "forbidden_parent_weights_sha256": list(
                config["train"]["forbidden_parent_weights_sha256"]),
            "max_grad_norm": float(config["train"]["max_grad_norm"]),
            "max_seconds": int(config["train"]["max_seconds"]),
            "response_only_labels": True,
            "optimizer_order": list(ARMS),
            "schedule_version": config["curriculum"]["schedule_version"],
        },
        "schedule": {
            "updates": schedule["updates"],
            "examples": schedule["examples"],
            "stages": [{"stage": stage["stage"], "updates": stage["updates"],
                        "batch": stage["batch"]}
                       for stage in schedule["stages"]],
            "probe": {key: schedule["probe"][key]
                      for key in ("updates_per_candidate", "batch",
                                  "total_probe_updates",
                                  "total_probe_examples", "measure",
                                  "tie_break", "selection_split")},
            "schedule_sha256": schedule["schedule_sha256"],
            "schedule_file_sha256": pins["files"]["schedule.json"]["sha256"],
            "dev_probe_case_ids": schedule["dev_probe_case_ids"],
            "arm_selection": dict(config["curriculum"]["selection"]
                                  ["arm_selection"]),
        },
        "evaluation": {
            "question_template": QUESTION_TEMPLATE,
            "controls": list(CONTROLS),
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
            "caption_max_new_tokens": int(config["evaluation"]
                                          ["caption_max_new_tokens"]),
            "greedy": True, "do_sample": False, "beam_width": 1, "batch": 1,
            "wrong_control": config["evaluation"]["wrong_control"],
            "shuffle_policy": config["evaluation"]["shuffle_policy"],
            "shuffle_seed_offset": int(config["evaluation"]
                                       ["shuffle_seed_offset"]),
            "bootstrap": dict(config["evaluation"]["bootstrap"]),
            "caption_metric": dict(config["evaluation"]["caption_metric"]),
            "gates": dict(config["evaluation"]["gates"]),
            "dev_splits": ["curriculum_dev", "diagnostic_dev",
                           "external_development"],
            "audit_split": "curriculum_audit",
            "external_final_split": "external_final_eval_untouched",
        },
        "retention": {
            "policy": "component_nonregression.v1",
            "baseline": dict(config["retention"]["baseline"]),
            "require_token_parity": True,
            "prompts": dict(config["retention"]["prompts"]),
            "labels": dict(config["retention"]["labels"]),
            "max_cases": int(config["retention"]["max_cases"]),
            "role": config["retention"]["role"],
        },
        "checkpoint_policy": dict(config["checkpoint_policy"]),
        "limits": dict(config["limits"]),
        "resource_policy": dict(config["resource_policy"]),
        "action_limits": dict(config["action_limits"]),
        "stop_conditions": list(config["stop_conditions"]),
        "implementation_sha256": _implementation_pins(root),
        "audit_policy": {
            "score_once": True,
            "requires_frozen_dev_report_sha256": True,
            "refuse_repeat_finalize": True,
            "external_final_untouched": True,
        },
        "claims": {
            "perception_quality": False,
            "semantic_quality": False,
            "overall_baseline_eligible": False,
            "component_nonregression_separate": True,
            "scope": "staged_curriculum_engineering_and_learned_quality_only",
        },
    }
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    _exact_keys(protocol, PROTOCOL_FIELDS, "protocol")
    _require(protocol["format"] == PROTOCOL_FORMAT, "Unknown protocol format")
    _require(protocol["version"] == VERSION, "Unknown protocol version")
    _require(protocol["stage"] == "curriculum_v1", "Unknown protocol stage")
    _require(protocol["root"] == ".", "Protocol root must be '.'")
    _nonempty(protocol["purpose"], "protocol.purpose")
    _nonnegative_int(protocol["seed"], "protocol.seed")
    _require(_is_sha256(protocol["config_sha256"]),
             "protocol.config_sha256 is invalid")
    data = _exact_keys(protocol["data"], {
        "root", "manifest", "files", "seals", "counts", "unknown_coverage",
        "label_coverage"}, "protocol.data")
    _require(isinstance(data["root"], str) and data["root"]
             and not Path(data["root"]).is_absolute(),
             "protocol.data.root is unsafe")
    _validate_pin(data["manifest"], "protocol.data.manifest")
    _require(isinstance(data["files"], Mapping) and data["files"],
             "protocol.data.files is empty")
    for name, pin in data["files"].items():
        _validate_pin(pin, f"protocol.data.files.{name}")
        _require(pin["path"] == name, "Protocol file pin path differs")
    _require(set(data["seals"]) == {"train", "dev", "audit", "external_final",
                                    "external_final_pointer"},
             "Protocol seal names differ")
    for name, value in data["seals"].items():
        _require(_is_sha256(value), f"Protocol seal is invalid: {name}")
    counts = data["counts"]
    for name in ("train_examples", "dev_cases", "audit_cases",
                 "schedule_updates", "schedule_examples"):
        _positive_int(counts[name], f"protocol.data.counts.{name}")
    _require(set(data["unknown_coverage"]) == set(SPLITS),
             "Protocol unknown coverage differs")

    external = _exact_keys(protocol["external"], {
        "root", "corpus", "train", "development", "final_sealed",
        "final_pointer", "consumed"}, "protocol.external")
    _require(external["consumed"] is False,
             "External final split must stay unconsumed")
    pointer = _exact_keys(external["final_pointer"], {
        "manifest", "manifest_sha256", "seal_sha256", "rows",
        "row_ids_sha256", "consumed", "policy"}, "protocol.external.final_pointer")
    _require(pointer["consumed"] is False
             and pointer["policy"] == "sealed_unconsumed_pointer_only",
             "External final pointer policy differs")
    _require(pointer["seal_sha256"] == data["seals"]["external_final"],
             "External final seal differs between protocol sections")

    source = _exact_keys(protocol["procedural_source"], {
        "root", "protocol_sha256", "manifest_sha256", "model_rows_sha256",
        "verifier_cases_sha256", "train_split",
        "development_split"}, "protocol.procedural_source")
    _require(source["development_split"]
             == "final_already_scored_reused_as_development",
             "Diagnostic development role differs")
    _require(all(_is_sha256(source[name]) for name in
                 ("protocol_sha256", "manifest_sha256", "model_rows_sha256",
                  "verifier_cases_sha256")),
             "Diagnostic source pins are invalid")

    vision = protocol["vision"]
    _require(isinstance(vision, Mapping)
             and vision.get("projector_hidden") == PROJECTOR_HIDDEN
             and vision.get("compute_dtype") == "float32",
             "Protocol vision section differs")
    _require(_is_sha256(protocol["vision_digest"]),
             "Protocol vision digest is invalid")
    _require(isinstance(protocol["vision_source"], Mapping)
             and "path" in protocol["vision_source"],
             "Protocol vision source is invalid")
    scope = protocol["vision_source"].get("scope")
    _require(isinstance(scope, Mapping)
             and scope.get("mode") == "parallel_replica"
             and scope.get("snapshot_at") is None,
             "Protocol vision scope is invalid")
    model = _exact_keys(protocol["model"], {"revision", "lock_sha256"},
                        "protocol.model")
    _require(isinstance(model["revision"], str) and len(model["revision"]) == 40,
             "Model revision is invalid")
    _require(_is_sha256(model["lock_sha256"]), "Model lock hash is invalid")

    splits = _exact_keys(protocol["splits"], {"train", "dev", "audit",
                                              "external_final"},
                         "protocol.splits")
    _require(splits["audit"]["usage"] == "scored_once_after_freeze",
             "Audit usage differs")
    _require(splits["dev"]["usage"] == "bounded_selection_only",
             "Dev usage differs")
    _require(splits["external_final"]["usage"]
             == "scored_once_after_freeze_untouched",
             "External final usage differs")
    for name in ("train", "dev", "audit", "external_final"):
        _require(_is_sha256(splits[name]["seal_sha256"]),
                 f"Split seal is invalid: {name}")
        _positive_int(splits[name]["rows"], f"protocol.splits.{name}.rows")
    _positive_int(splits["dev"].get("procedural_rows"),
                  "protocol.splits.dev.procedural_rows")
    _positive_int(splits["dev"].get("external_caption_rows"),
                  "protocol.splits.dev.external_caption_rows")
    _require(splits["dev"]["rows"]
             == splits["dev"]["procedural_rows"]
             + splits["dev"]["external_caption_rows"],
             "Dev split rows do not equal procedural + external caption rows")

    groups = _exact_keys(protocol["trainable_groups"], {
        "scope", "tensors", "parameter_count", "muon8bit", "adamw_aux",
        "adamw_control", "frozen", "response_only_labels"},
        "protocol.trainable_groups")
    _require(groups["scope"] == "vision_projector."
             and list(groups["tensors"]) == list(PROJECTOR_TENSORS)
             and groups["parameter_count"] == PROJECTOR_PARAMETER_COUNT,
             "Trainable group inventory differs")
    _require(groups["response_only_labels"] is True,
             "Response-only labels are required")
    _exact_keys(groups["muon8bit"], {
        "param_names", "lr", "momentum", "ns_steps", "block_size"},
        "protocol.trainable_groups.muon8bit")
    _exact_keys(groups["adamw_aux"], {
        "param_names", "lr", "betas", "eps"}, "protocol.trainable_groups.adamw_aux")
    _exact_keys(groups["adamw_control"], {
        "param_names", "lr", "betas", "eps"},
        "protocol.trainable_groups.adamw_control")
    _require(sorted(groups["muon8bit"]["param_names"]
                    + groups["adamw_aux"]["param_names"])
             == sorted(PROJECTOR_TENSORS),
             "Muon arm groups do not cover the projector")
    _require(sorted(groups["adamw_control"]["param_names"])
             == sorted(PROJECTOR_TENSORS),
             "AdamW control does not cover the projector")

    arms = _exact_keys(protocol["arms"], {
        "format", "version", "seed", "init_seed", "init_recipe",
        "projector_hidden", "projector_tensors", "projector_parameter_count",
        "trainable_scope", "frozen_prefixes", "arms", "equal_compute"},
        "protocol.arms")
    _require(arms["format"] == ARMS_FORMAT
             and arms["projector_hidden"] == PROJECTOR_HIDDEN
             and arms["projector_parameter_count"] == PROJECTOR_PARAMETER_COUNT,
             "Protocol arms declaration differs")
    _exact_keys(arms["arms"], set(ARMS), "protocol.arms.arms")
    equal = arms["equal_compute"]
    _require(equal["same_schedule_for_both_arms"] is True
             and equal["same_initialization_for_both_arms"] is True
             and equal["updates_per_arm"] == protocol["schedule"]["updates"]
             and equal["examples_per_arm"] == protocol["schedule"]["examples"],
             "Equal-compute declaration differs from the schedule")

    train = _exact_keys(protocol["train"], {
        "seed", "init_seed", "init_recipe", "warm_start",
        "forbidden_parent_weights_sha256", "max_grad_norm", "max_seconds",
        "response_only_labels", "optimizer_order", "schedule_version"},
        "protocol.train")
    _require(train["warm_start"] is None, "Warm start is forbidden")
    _require(train["response_only_labels"] is True,
             "Response-only labels are required")
    _require(all(_is_sha256(item) for item in
                 train["forbidden_parent_weights_sha256"]),
             "Forbidden parent hashes are invalid")
    _require(list(train["optimizer_order"]) == list(ARMS),
             "Optimizer order differs")

    schedule = _exact_keys(protocol["schedule"], {
        "updates", "examples", "stages", "probe", "schedule_sha256",
        "schedule_file_sha256", "dev_probe_case_ids", "arm_selection"},
        "protocol.schedule")
    arm_rule = _exact_keys(schedule["arm_selection"], {
        "measure", "tie_break", "both_arms_required"}, "schedule.arm_selection")
    _require(arm_rule["measure"] == "mean_dev_response_loss"
             and arm_rule["tie_break"] == "declared_arm_order"
             and arm_rule["both_arms_required"] is True,
             "Protocol arm selection rule differs")
    _require(_is_sha256(schedule["schedule_sha256"])
             and _is_sha256(schedule["schedule_file_sha256"]),
             "Schedule hashes are invalid")
    _require(sum(stage["updates"] for stage in schedule["stages"])
             == schedule["updates"],
             "Schedule stage updates do not sum to the total")
    _require(sum(stage["updates"] * stage["batch"]
                 for stage in schedule["stages"]) == schedule["examples"],
             "Schedule stage examples do not sum to the total")
    probe = schedule["probe"]
    _require(probe["total_probe_updates"]
             == len(ARMS) * 2 * probe["updates_per_candidate"],
             "Probe update accounting differs")
    _require(probe["total_probe_examples"]
             == len(ARMS) * 2 * probe["updates_per_candidate"]
             * probe["batch"],
             "Probe example accounting differs")
    _require(probe["selection_split"] == "procedural_dev",
             "Probe selection split differs")

    evaluation = _exact_keys(protocol["evaluation"], {
        "question_template", "controls", "max_new_tokens",
        "caption_max_new_tokens", "greedy", "do_sample", "beam_width",
        "batch", "wrong_control", "shuffle_policy", "shuffle_seed_offset",
        "bootstrap", "caption_metric", "gates", "dev_splits", "audit_split",
        "external_final_split"}, "protocol.evaluation")
    _require(evaluation["question_template"] == QUESTION_TEMPLATE,
             "Evaluation question template differs")
    _require(list(evaluation["controls"]) == list(CONTROLS),
             "Evaluation controls differ")
    _require(evaluation["greedy"] is True
             and evaluation["do_sample"] is False,
             "Generation must stay greedy")
    _require(evaluation["audit_split"] == "curriculum_audit",
             "Audit split differs")
    _require(evaluation["caption_metric"]["claim"]
             == "narrow_alignment_metric_only_no_semantic_quality_claim",
             "Caption claim boundary differs")

    retention = _exact_keys(protocol["retention"], {
        "policy", "baseline", "require_token_parity", "prompts", "labels",
        "max_cases", "role"}, "protocol.retention")
    _require(retention["policy"] == "component_nonregression.v1",
             "Retention policy differs")
    _require(retention["baseline"]["correct"] == 22
             and retention["baseline"]["cases"] == 32,
             "Declared retention baseline differs")
    _require(retention["require_token_parity"] is True,
             "Retention token parity is required")
    for name in ("prompts", "labels"):
        _validate_pin(retention[name], f"protocol.retention.{name}")

    checkpoint = _exact_keys(protocol["checkpoint_policy"], {
        "policy", "probe_checkpoints", "base_copies", "optimizer_state_saved",
        "export", "reload", "selection_freeze", "audit_scoring", "keep_arms"},
        "protocol.checkpoint_policy")
    _require(checkpoint["export"] == "strict_extension_v4"
             and checkpoint["reload"] == "zero_update_fresh_process_bitwise"
             and checkpoint["audit_scoring"] == "once"
             and checkpoint["base_copies"] == 0,
             "Checkpoint policy differs")
    limits = _exact_keys(protocol["limits"], {
        "max_prompt_tokens", "max_response_tokens", "max_sequence_tokens",
        "max_new_tokens", "max_eval_cases", "max_rows", "max_updates",
        "max_probe_updates", "max_seconds", "max_export_bytes",
        "min_free_disk_bytes", "max_cache_bytes", "max_driver_bytes",
        "max_bootstrap_resamples"}, "protocol.limits")
    for name, value in limits.items():
        _positive_int(value, f"protocol.limits.{name}")
    _require(limits["max_driver_bytes"] >= 50 * 1024 ** 3,
             "Driver guard must be at least 50 GiB")
    _require(evaluation["caption_max_new_tokens"]
             <= limits["max_response_tokens"],
             "Caption generation exceeds the response budget")
    _require(schedule["updates"] * len(ARMS)
             + probe["total_probe_updates"] <= limits["max_updates"],
             "Declared updates exceed the protocol limit")
    policy = _exact_keys(protocol["resource_policy"], {
        "device", "cpu_fallback", "exclusive_mps_worker", "workers",
        "downloads", "driver_limit_bytes", "sustained_swap_growth_bytes",
        "minimum_host_free_percent", "sustained_host_samples",
        "initial_free_admission"}, "protocol.resource_policy")
    _require(policy["device"] == "mps" and policy["cpu_fallback"] is False
             and policy["exclusive_mps_worker"] is True
             and policy["workers"] == 1 and policy["downloads"] == 0,
             "Protocol resource policy differs")
    _require(policy["driver_limit_bytes"] >= 50 * 1024 ** 3,
             "Protocol driver guard must be at least 50 GiB")
    _exact_keys(protocol["action_limits"], {
        "enabled_ports", "enabled_branches", "core_impl", "host_actions",
        "success_claims"}, "protocol.action_limits")
    _require(protocol["action_limits"]["host_actions"] == "none"
             and protocol["action_limits"]["success_claims"] == "none",
             "No host actions or success claims are allowed")
    _require(isinstance(protocol["stop_conditions"], list)
             and len(protocol["stop_conditions"]) >= 5,
             "Stop conditions are missing")
    pins = protocol["implementation_sha256"]
    _require(isinstance(pins, Mapping) and set(pins) == set(IMPLEMENTATION_FILES),
             "Implementation pins differ")
    for name, value in pins.items():
        _require(_is_sha256(value), f"Implementation pin is invalid: {name}")
    audit = _exact_keys(protocol["audit_policy"], {
        "score_once", "requires_frozen_dev_report_sha256",
        "refuse_repeat_finalize", "external_final_untouched"},
        "protocol.audit_policy")
    _require(all(value is True for value in audit.values()),
             "Audit policy must be one-shot")
    claims = _exact_keys(protocol["claims"], {
        "perception_quality", "semantic_quality", "overall_baseline_eligible",
        "component_nonregression_separate", "scope"}, "protocol.claims")
    _require(claims["perception_quality"] is False
             and claims["semantic_quality"] is False
             and claims["overall_baseline_eligible"] is False
             and claims["component_nonregression_separate"] is True,
             "Claim boundaries differ")


def verify_implementation_pins(protocol: Mapping[str, Any],
                               root: str | Path) -> dict[str, str]:
    root = Path(root)
    actual = {}
    for relative, expected in protocol["implementation_sha256"].items():
        path = _safe_relative(root, relative)
        _require(path.is_file(), f"Pinned implementation file is missing: {relative}")
        actual[relative] = sha256_file(path)
        _require(actual[relative] == expected,
                 f"Implementation bytes changed: {relative}")
    return actual


def verify_retention_pins(protocol: Mapping[str, Any],
                          root: str | Path) -> dict[str, str]:
    """Verify the declared retention prompt/label bytes at runtime."""
    root = Path(root)
    actual = {}
    for name in ("prompts", "labels"):
        pin = protocol["retention"][name]
        path = _safe_relative(root, pin["path"])
        _require(path.is_file(), f"Retention {name} file is missing")
        actual[name] = sha256_file(path)
        _require(actual[name] == pin["sha256"],
                 f"Retention {name} bytes changed")
    return actual


def write_protocol(path: str | Path, protocol: Mapping[str, Any]) -> None:
    _write_exact(Path(path), _json_bytes(protocol))


# ---------------------------------------------------------------------------
# Scoring: answer, controls, unknown/abstention, caption, retention
# ---------------------------------------------------------------------------


def curriculum_prompt(question: str) -> str:
    return QUESTION_TEMPLATE.format(question=question)


def normalise_answer(value: str) -> str:
    text = " ".join(str(value).strip().casefold().split())
    return text.strip(" \t\r\n.,;:!?\"'`()[]{}")


def is_unknown_answer(value: str) -> bool:
    normalised = normalise_answer(value)
    if not normalised:
        return False
    return (normalised in UNKNOWN_ANSWERS
            or "unknown" in normalised.split()
            or normalised.startswith("cannot determine"))


def answer_correct(generated: str, expected: str) -> bool:
    """Exact normalized match, with one shared unknown-answer equivalence."""
    if is_unknown_answer(expected):
        return is_unknown_answer(generated)
    return (normalise_answer(generated) == normalise_answer(expected)
            and not is_unknown_answer(generated))


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def caption_tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(str(value).casefold())


def caption_token_f1(generated: str, expected: str) -> float:
    generated_tokens = caption_tokens(generated)
    expected_tokens = caption_tokens(expected)
    if not generated_tokens or not expected_tokens:
        return 0.0
    counts: dict[str, int] = Counter(expected_tokens)
    overlap = 0
    for token in generated_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def validate_generation_rows(rows: Sequence[Mapping[str, Any]],
                             case_ids: Iterable[str],
                             controls: Sequence[str] = CONTROLS) -> None:
    expected = set(case_ids)
    controls = list(controls)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "Generation row must be a mapping")
        for field in FORBIDDEN_CLAIM_FIELDS:
            _require(field not in row,
                     f"Generation row carries a forbidden claim field: {field}")
        _require(GENERATION_FIELDS <= set(row), "Generation row fields are missing")
        _require(row["control"] in controls,
                 f"Undeclared generation control: {row['control']}")
        _require(row["case_id"] in expected,
                 f"Generation references an unknown case: {row['case_id']}")
        _require(isinstance(row["generated_text"], str),
                 "Generated text must be a string")
        key = (row["control"], row["case_id"])
        _require(key not in seen, f"Duplicate generation: {key}")
        seen.add(key)
    for control in controls:
        covered = {case for control_name, case in seen
                   if control_name == control}
        _require(covered == expected,
                 f"Control {control} does not cover every case")


def assign_controls(cases: Sequence[Mapping[str, Any]], *,
                    seed_offset: int) -> dict[str, dict[str, str]]:
    """Deterministic wrong-image and shuffled-image assignments."""
    ordered = sorted(cases, key=lambda case: case["case_id"])
    by_world = defaultdict(list)
    for case in ordered:
        by_world[case["world_id"]].append(case["case_id"])
    worlds = sorted(by_world)
    wrong: dict[str, str] = {}
    for case in ordered:
        candidates = [other for other in ordered
                      if other["case_id"] != case["case_id"]
                      and other["task"] == case["task"]
                      and other["world_id"] != case["world_id"]]
        _require(candidates,
                 f"No wrong-image candidate for {case['case_id']}")
        index = (seed_offset + sum(ord(char) for char in case["case_id"])) \
            % len(candidates)
        wrong[case["case_id"]] = candidates[index]["case_id"]
    # Derangement over worlds, then match cases inside shifted worlds.
    if len(worlds) < 2:
        raise VisionCurriculumError("Shuffled control requires two worlds")
    rotation = 1 + seed_offset % (len(worlds) - 1)
    shuffled: dict[str, str] = {}
    for position, world in enumerate(worlds):
        target_world = worlds[(position + rotation) % len(worlds)]
        sources = by_world[target_world]
        for offset, case_id in enumerate(by_world[world]):
            shuffled[case_id] = sources[offset % len(sources)]
    for case_id, source in shuffled.items():
        _require(source != case_id, "Shuffled control must derange every case")
    return {"wrong": wrong, "shuffled": shuffled}


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(values, "Cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clustered_bootstrap(pairs_by_world: Mapping[str, Sequence[float]], *,
                        resamples: int, confidence: float,
                        seed: int) -> dict[str, Any]:
    """World-clustered percentile bootstrap of the case-weighted mean."""
    clusters = [list(values) for _, values in sorted(pairs_by_world.items())
                if values]
    _require(clusters, "Bootstrap needs at least one nonempty world")
    point = sum(sum(values) for values in clusters) / sum(
        len(values) for values in clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        drawn = [clusters[rng.randrange(len(clusters))]
                 for _ in clusters]
        total = sum(sum(values) for values in drawn)
        count = sum(len(values) for values in drawn)
        samples.append(total / count if count else 0.0)
    alpha = (1.0 - confidence) / 2
    return {"mean": point,
            "lower": _percentile(samples, alpha),
            "upper": _percentile(samples, 1 - alpha),
            "resamples": resamples, "confidence": confidence,
            "clusters": len(clusters)}


def _control_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    per_task = defaultdict(list)
    per_world = defaultdict(list)
    for row in rows:
        per_task[row["task"]].append(row["correct"])
        per_world[row["world_id"]].append(row["correct"])
    return {
        "available": True,
        "cases": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "unknown": sum(row["unknown"] for row in rows),
        "unknown_rate": sum(row["unknown"] for row in rows) / len(rows),
        "unique_worlds": len(per_world),
        "per_task": {task: {"cases": len(values),
                            "accuracy": sum(values) / len(values)}
                     for task, values in sorted(per_task.items())},
        "per_world": {world: {"cases": len(values),
                              "accuracy": sum(values) / len(values)}
                      for world, values in sorted(per_world.items())},
    }


def score_procedural(cases: Sequence[Mapping[str, Any]],
                     generations: Sequence[Mapping[str, Any]], *,
                     controls: Sequence[str] = CONTROLS,
                     seed: int = 0,
                     bootstrap_resamples: int = 1000,
                     bootstrap_confidence: float = 0.95,
                     gates: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    validate_generation_rows(generations, case_by_id, controls)
    scored = []
    for row in generations:
        case = case_by_id[row["case_id"]]
        generated = str(row["generated_text"])
        scored.append({
            "control": row["control"], "case_id": row["case_id"],
            "world_id": case["world_id"], "task": case["task"],
            "kind": case.get("kind"), "expected": case["label"],
            "generated": generated,
            "correct": answer_correct(generated, case["label"]),
            "unknown": is_unknown_answer(generated),
            "abstain_case": case["label"] == UNKNOWN_ANSWER,
        })
    by_control: dict[str, dict[str, Any]] = {}
    for control in controls:
        rows = [row for row in scored if row["control"] == control]
        by_control[control] = _control_metrics(rows)
    paired: dict[str, Any] = {}
    correct_rows = {row["case_id"]: row for row in scored
                    if row["control"] == "correct"}
    for control in controls:
        if control == "correct":
            continue
        other = {row["case_id"]: row for row in scored
                 if row["control"] == control}
        shared = sorted(set(correct_rows) & set(other))
        _require(shared, f"No paired cases for control {control}")
        margins = {case_id: float(correct_rows[case_id]["correct"])
                   - float(other[case_id]["correct"]) for case_id in shared}
        by_world = defaultdict(list)
        for case_id, value in margins.items():
            by_world[case_by_id[case_id]["world_id"]].append(value)
        paired[control] = {
            "cases": len(shared),
            "correct_accuracy": sum(correct_rows[case_id]["correct"]
                                    for case_id in shared) / len(shared),
            "control_accuracy": sum(other[case_id]["correct"]
                                    for case_id in shared) / len(shared),
            "correct_margin": sum(margins.values()) / len(shared),
            "margin_bootstrap": clustered_bootstrap(
                by_world, resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=seed + 101 * list(controls).index(control)),
        }
    correct_rows_list = [row for row in scored if row["control"] == "correct"]
    answerable = [row for row in correct_rows_list
                  if not row["abstain_case"]]
    abstain = [row for row in correct_rows_list if row["abstain_case"]]
    _require(abstain, "Abstention cases are missing from the correct control")
    unknown_accuracy = (sum(row["unknown"] for row in abstain) / len(abstain))
    false_answer_on_abstain = 1.0 - unknown_accuracy
    false_abstention_on_answerable = (
        sum(row["unknown"] for row in answerable) / len(answerable)
        if answerable else 0.0)
    accuracy_by_world = defaultdict(list)
    for row in correct_rows_list:
        accuracy_by_world[row["world_id"]].append(float(row["correct"]))
    correct_bootstrap = clustered_bootstrap(
        accuracy_by_world, resamples=bootstrap_resamples,
        confidence=bootstrap_confidence, seed=seed)
    failures: list[str] = []
    if gates is not None:
        correct = by_control["correct"]
        if correct["accuracy"] < gates["min_correct_accuracy"]:
            failures.append("min_correct_accuracy")
        if correct["unique_worlds"] < gates["min_unique_families"]:
            failures.append("min_unique_families")
        # The overall unknown rate mixes required abstentions with real
        # errors; the gate applies only to answerable cases.
        if false_abstention_on_answerable > gates["max_unknown_rate"]:
            failures.append("max_unknown_rate")
        for control in ("wrong", "shuffled", "noimage"):
            gate = f"min_correct_margin_{control}"
            if control not in by_control or not by_control[control]["available"]:
                failures.append(f"missing_control_{control}")
                continue
            if paired.get(control, {}).get("correct_margin", -1.0) < gates[gate]:
                failures.append(gate)
        if unknown_accuracy < gates["min_unknown_accuracy"]:
            failures.append("min_unknown_accuracy")
        if false_answer_on_abstain > gates["max_false_answer_on_abstain"]:
            failures.append("max_false_answer_on_abstain")
        if (false_abstention_on_answerable
                > gates["max_false_abstention_on_answerable"]):
            failures.append("max_false_abstention_on_answerable")
    return {
        "cases": len(case_by_id),
        "controls": by_control,
        "paired": paired,
        "correct_bootstrap": correct_bootstrap,
        "unknown": {
            "abstain_cases": len(abstain),
            "abstention_accuracy": unknown_accuracy,
            "false_answer_on_abstain": false_answer_on_abstain,
            "answerable_cases": len(answerable),
            "false_abstention_on_answerable": false_abstention_on_answerable,
            "max_unknown_rate_scope": "answerable_cases_only",
        },
        "gate_failures": failures,
        "learned_quality_pass": not failures,
        "bootstrap": {"seed": seed, "resamples": bootstrap_resamples,
                      "confidence": bootstrap_confidence},
        "case_ids": sorted(case_by_id),
        "case_ids_sha256": fingerprint(sorted(case_by_id)),
        "rows": scored,
    }


def score_caption(labels: Sequence[Mapping[str, Any]],
                  generations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {row["case_id"]: row["answer"] for row in labels}
    seen: set[str] = set()
    f1_values = []
    losses = []
    rows = []
    for generation in generations:
        case_id = generation.get("case_id")
        _require(case_id in expected, f"Unknown caption case: {case_id}")
        _require(case_id not in seen, f"Duplicate caption generation: {case_id}")
        seen.add(case_id)
        generated = str(generation.get("generated_text", ""))
        f1 = caption_token_f1(generated, expected[case_id])
        loss = generation.get("response_loss")
        _require(loss is not None
                 and not isinstance(loss, bool)
                 and isinstance(loss, (int, float))
                 and math.isfinite(float(loss)) and float(loss) >= 0,
                 f"Teacher-forced response loss is missing: {case_id}")
        losses.append(float(loss))
        f1_values.append(f1)
        rows.append({"case_id": case_id, "generated": generated,
                     "expected": expected[case_id], "token_f1": f1,
                     "response_loss": (None if loss is None else float(loss))})
    _require(seen == set(expected), "Caption generations are incomplete")
    case_ids = sorted(seen)
    return {
        "cases": len(rows),
        "case_ids": case_ids,
        "case_ids_sha256": fingerprint(case_ids),
        "token_f1": sum(f1_values) / len(f1_values),
        "teacher_forced_response_loss": sum(losses) / len(losses),
        "loss_cases": len(losses),
        "claim": "narrow_alignment_metric_only_no_semantic_quality_claim",
        "rows": rows,
    }


def retention_policy(*, measured_correct: int | None, measured_cases: int,
                     baseline_correct: int, baseline_cases: int,
                     token_parity: bool, gates: Mapping[str, Any]
                     ) -> dict[str, Any]:
    """component_nonregression.v1 with baseline capability kept separate."""
    _require(measured_cases == baseline_cases,
             "Retention case counts differ between baseline and measurement")
    baseline_accuracy = baseline_correct / baseline_cases
    measured = (None if measured_correct is None
                else measured_correct / measured_cases)
    component_failures: list[str] = []
    if measured is None:
        component_failures.append("retention_unavailable")
    if token_parity is not True:
        component_failures.append("text_tokens_changed")
    drop = None if measured is None else baseline_accuracy - measured
    if drop is not None and drop > float(gates["max_text_retention_drop"]):
        component_failures.append("max_text_retention_drop")
    baseline_eligible = (baseline_accuracy
                         >= float(gates["min_text_retention_accuracy"]))
    return {
        "policy": "component_nonregression.v1",
        "baseline_correct": baseline_correct,
        "baseline_cases": baseline_cases,
        "baseline_accuracy": baseline_accuracy,
        "measured_correct": measured_correct,
        "measured_cases": measured_cases,
        "measured_accuracy": measured,
        "retention_drop": drop,
        "baseline_eligible": baseline_eligible,
        "overall_baseline_eligible": False,
        "baseline_failures": ([] if baseline_eligible
                              else ["baseline_capability_ineligible"]),
        "component_failures": component_failures,
        "component_nonregression_pass": not component_failures,
        "overall_eligible": baseline_eligible and not component_failures,
        "token_parity": token_parity,
        "note": ("The declared 22/32 baseline stays overall ineligible; "
                 "component drop and token parity are reported separately."),
    }


def validate_score_claims(report: Mapping[str, Any]) -> None:
    _require(report.get("claims", {}).get("semantic_quality") is False,
             "Semantic-quality claims are not supported")
    _require(report.get("claims", {}).get("perception_quality") is False,
             "Perception-quality claims are not supported")
    _require(report.get("claims", {}).get("overall_baseline_eligible") is False,
             "The declared baseline eligibility must stay false")
    for field in FORBIDDEN_CLAIM_FIELDS:
        _require(field not in report,
                 f"Score report carries a forbidden claim: {field}")


def assemble_score_report(*, protocol: Mapping[str, Any],
                          run_report: Mapping[str, Any],
                          cases: Sequence[Mapping[str, Any]],
                          generations: Sequence[Mapping[str, Any]],
                          caption_labels: Sequence[Mapping[str, Any]],
                          caption_generations: Sequence[Mapping[str, Any]],
                          retention: Mapping[str, Any],
                          reload_report_sha256: str) -> dict[str, Any]:
    validation = validate_run_report(run_report, protocol)
    gates = protocol["evaluation"]["gates"]
    score = score_procedural(
        cases, generations,
        controls=protocol["evaluation"]["controls"],
        seed=int(protocol["seed"]),
        bootstrap_resamples=int(protocol["evaluation"]["bootstrap"]
                                ["resamples"]),
        bootstrap_confidence=float(protocol["evaluation"]["bootstrap"]
                                   ["confidence"]),
        gates=gates)
    caption = score_caption(caption_labels, caption_generations)
    retention_result = retention_policy(
        measured_correct=retention.get("measured_correct"),
        measured_cases=int(protocol["retention"]["max_cases"]),
        baseline_correct=int(protocol["retention"]["baseline"]["correct"]),
        baseline_cases=int(protocol["retention"]["baseline"]["cases"]),
        token_parity=retention.get("token_parity") is True,
        gates=gates)
    retention_result["gates"] = dict(gates)
    engineering = {
        "pass": validation["engineering_pass"],
        "checks": validation["checks"],
        "failures": validation["engineering_failures"],
    }
    learned = {
        "pass": score["learned_quality_pass"],
        "failures": list(score["gate_failures"]),
        "procedural_dev": {key: score[key] for key in
                           ("cases", "controls", "paired",
                            "correct_bootstrap", "unknown",
                            "gate_failures", "learned_quality_pass",
                            "bootstrap", "rows", "case_ids",
                            "case_ids_sha256")},
        "external_development_caption": caption,
        "retention": retention_result,
    }
    _require(_is_sha256(reload_report_sha256),
             "Reload report hash is invalid")
    report = {
        "format": SCORE_REPORT_FORMAT,
        "version": VERSION,
        "stage": "dev",
        "arm": run_report["arm"],
        "protocol_sha256": run_report["protocol_sha256"],
        "run_report_sha256": run_report["_run_report_sha256"],
        "reload_report_sha256": reload_report_sha256,
        "data_pins": protocol["data"]["manifest"],
        "engineering": engineering,
        "learned_quality": learned,
        "eligibility": {
            "overall_baseline_eligible": False,
            "baseline_eligible": retention_result["baseline_eligible"],
            "component_nonregression_pass":
                retention_result["component_nonregression_pass"],
            "learned_quality_pass": score["learned_quality_pass"],
            "curriculum_qualified": (
                engineering["pass"] and score["learned_quality_pass"]
                and retention_result["component_nonregression_pass"]),
            "overall_eligible": (
                engineering["pass"] and score["learned_quality_pass"]
                and retention_result["component_nonregression_pass"]
                and retention_result["baseline_eligible"]),
            "audit_scores_present": False,
        },
        "claims": {
            "perception_quality": False,
            "semantic_quality": False,
            "overall_baseline_eligible": False,
            "component_nonregression_separate": True,
            "scope": "staged_curriculum_engineering_and_learned_quality_only",
        },
        "audit_scored": False,
        "external_final_consumed": False,
    }
    validate_score_report(report)
    return report


def _validate_caption_metrics(caption: Mapping[str, Any],
                              name: str = "caption") -> None:
    cases = caption["cases"]
    _positive_int(cases, f"{name}.cases")
    rows = caption.get("rows")
    _require(isinstance(rows, list) and len(rows) == cases,
             f"{name}.rows do not cover the cases")
    case_ids = caption.get("case_ids")
    _require(isinstance(case_ids, list)
             and len(set(case_ids)) == len(case_ids) == cases
             and {row["case_id"] for row in rows} == set(case_ids),
             f"{name}.case_ids do not cover the rows")
    _require(caption.get("case_ids_sha256") == fingerprint(case_ids),
             f"{name}.case_ids_sha256 differs")
    f1_values = []
    losses = []
    for row in rows:
        f1 = caption_token_f1(str(row["generated"]), str(row["expected"]))
        _require(abs(float(row["token_f1"]) - f1) <= 1e-9,
                 f"{name} row F1 differs from raw text")
        f1_values.append(f1)
        loss = _finite(row["response_loss"], f"{name} row loss",
                       minimum=0.0)
        losses.append(loss)
    _require(abs(caption["token_f1"] - sum(f1_values) / cases) <= 1e-9,
             f"{name}.token_f1 differs from raw rows")
    _require(caption["loss_cases"] == cases,
             f"{name}.loss_cases does not cover the cases")
    _require(abs(caption["teacher_forced_response_loss"]
                 - sum(losses) / cases) <= 1e-9,
             f"{name} response loss differs from raw rows")
    _require(caption["claim"]
             == "narrow_alignment_metric_only_no_semantic_quality_claim",
             f"{name} claim boundary differs")


def _validate_procedural_metrics(score: Mapping[str, Any],
                                 name: str = "procedural") -> None:
    """Recompute all aggregates from the retained raw rows."""
    rows = score.get("rows")
    _require(isinstance(rows, list) and rows,
             f"{name}.rows are missing")
    case_ids = set(score.get("case_ids", ()))
    _require(case_ids and len(case_ids) == len(score["case_ids"]),
             f"{name}.case_ids are missing or duplicated")
    _require(score.get("case_ids_sha256")
             == fingerprint(score["case_ids"]),
             f"{name}.case_ids_sha256 differs")
    controls_declared = list(score["controls"])
    seen_pairs = set()
    for row in rows:
        generated = str(row["generated"])
        expected = str(row["expected"])
        correct = answer_correct(generated, expected)
        unknown = is_unknown_answer(generated)
        abstain_case = normalise_answer(expected) == UNKNOWN_ANSWER
        _require(row["correct"] == correct
                 and row["unknown"] == unknown
                 and row["abstain_case"] == abstain_case,
                 f"{name} row evidence contradicts its text: {row['case_id']}")
        pair = (row["control"], row["case_id"])
        _require(pair not in seen_pairs,
                 f"{name} duplicate raw row: {pair}")
        seen_pairs.add(pair)
        _require(row["case_id"] in case_ids,
                 f"{name} raw row uses an undeclared case")
    for control in controls_declared:
        covered = {case for name_control, case in seen_pairs
                   if name_control == control}
        _require(covered == case_ids,
                 f"{name}.{control} does not cover every sealed case")
    _require(len(seen_pairs) == len(controls_declared) * len(case_ids),
             f"{name} raw row matrix is incomplete")
    bootstrap_meta = score.get("bootstrap")
    _exact_keys(bootstrap_meta, {"seed", "resamples", "confidence"},
                f"{name}.bootstrap")
    _nonnegative_int(bootstrap_meta["seed"], f"{name}.bootstrap.seed")
    _positive_int(bootstrap_meta["resamples"],
                  f"{name}.bootstrap.resamples")
    _finite(bootstrap_meta["confidence"], f"{name}.bootstrap.confidence",
            minimum=0.5, maximum=0.999999)
    controls = list(score["controls"])
    _require(set(controls) == {row["control"] for row in rows},
             f"{name} controls do not cover the raw rows")
    case_ids = {row["case_id"] for row in rows}
    _require(score["cases"] == len(case_ids),
             f"{name}.cases differs from the raw case ids")
    for control in controls:
        selected = [row for row in rows if row["control"] == control]
        _require(selected, f"{name}.{control} has no raw rows")
        count = len(selected)
        correct = sum(bool(row["correct"]) for row in selected)
        unknown = sum(bool(row["unknown"]) for row in selected)
        expected = {
            "available": True, "cases": count, "correct": correct,
            "accuracy": correct / count, "unknown": unknown,
            "unknown_rate": unknown / count,
            "unique_worlds": len({row["world_id"] for row in selected}),
        }
        actual = dict(score["controls"][control])
        for key, value in expected.items():
            _require(actual.get(key) == value,
                     f"{name}.{control}.{key} differs from raw rows")
        per_task = defaultdict(list)
        per_world = defaultdict(list)
        for row in selected:
            per_task[row["task"]].append(float(row["correct"]))
            per_world[row["world_id"]].append(float(row["correct"]))
        expected_tasks = {task: {"cases": len(values),
                                 "accuracy": sum(values) / len(values)}
                          for task, values in sorted(per_task.items())}
        expected_worlds = {world: {"cases": len(values),
                                   "accuracy": sum(values) / len(values)}
                           for world, values in sorted(per_world.items())}
        _require(actual["per_task"] == expected_tasks
                 and actual["per_world"] == expected_worlds,
                 f"{name}.{control} task/world aggregates differ")
    by_control = {control: {row["case_id"]: row for row in rows
                            if row["control"] == control}
                  for control in controls}
    correct_rows = by_control["correct"]
    for control in controls:
        if control == "correct":
            continue
        shared = sorted(set(correct_rows) & set(by_control[control]))
        _require(shared, f"{name}.{control} has no paired raw rows")
        margins = {case_id: float(correct_rows[case_id]["correct"])
                   - float(by_control[control][case_id]["correct"])
                   for case_id in shared}
        correct_accuracy = sum(correct_rows[case_id]["correct"]
                               for case_id in shared) / len(shared)
        control_accuracy = sum(by_control[control][case_id]["correct"]
                               for case_id in shared) / len(shared)
        item = score["paired"][control]
        _require(item["cases"] == len(shared)
                 and abs(item["correct_accuracy"] - correct_accuracy) <= 1e-9
                 and abs(item["control_accuracy"] - control_accuracy) <= 1e-9
                 and abs(item["correct_margin"]
                         - (correct_accuracy - control_accuracy)) <= 1e-9,
                 f"{name}.paired.{control} differs from raw rows")
        by_world = defaultdict(list)
        for case_id, value in margins.items():
            by_world[correct_rows[case_id]["world_id"]].append(value)
        expected_bootstrap = clustered_bootstrap(
            by_world, resamples=bootstrap_meta["resamples"],
            confidence=bootstrap_meta["confidence"],
            seed=bootstrap_meta["seed"] + 101 * controls.index(control))
        _require(item["margin_bootstrap"] == expected_bootstrap,
                 f"{name}.paired.{control} bootstrap differs from raw rows")
    correct_list = list(correct_rows.values())
    answerable = [row for row in correct_list if not row["abstain_case"]]
    abstain = [row for row in correct_list if row["abstain_case"]]
    _require(abstain and answerable,
             f"{name}.unknown case split is empty")
    unknown_accuracy = sum(bool(row["unknown"]) for row in abstain) / len(abstain)
    false_abstention = (sum(bool(row["unknown"]) for row in answerable)
                        / len(answerable))
    expected_unknown = {
        "abstain_cases": len(abstain),
        "answerable_cases": len(answerable),
        "abstention_accuracy": unknown_accuracy,
        "false_answer_on_abstain": 1.0 - unknown_accuracy,
        "false_abstention_on_answerable": false_abstention,
    }
    for key, value in expected_unknown.items():
        _require(score["unknown"].get(key) == value,
                 f"{name}.unknown.{key} differs from raw rows")
    accuracy_by_world = defaultdict(list)
    for row in correct_list:
        accuracy_by_world[row["world_id"]].append(float(row["correct"]))
    expected_correct = clustered_bootstrap(
        accuracy_by_world, resamples=bootstrap_meta["resamples"],
        confidence=bootstrap_meta["confidence"], seed=bootstrap_meta["seed"])
    _require(score["correct_bootstrap"] == expected_correct,
             f"{name}.correct_bootstrap differs from raw rows")
    failures = list(score["gate_failures"])
    _require(score["learned_quality_pass"] == (not failures),
             f"{name}.pass contradicts its failures")


def _validate_caption_metrics(caption: Mapping[str, Any],
                              name: str = "caption") -> None:
    cases = caption["cases"]
    _positive_int(cases, f"{name}.cases")
    rows = caption.get("rows")
    _require(isinstance(rows, list) and len(rows) == cases,
             f"{name}.rows do not cover the cases")
    case_ids = caption.get("case_ids")
    _require(isinstance(case_ids, list)
             and len(set(case_ids)) == len(case_ids) == cases
             and {row["case_id"] for row in rows} == set(case_ids),
             f"{name}.case_ids do not cover the rows")
    _require(caption.get("case_ids_sha256") == fingerprint(case_ids),
             f"{name}.case_ids_sha256 differs")
    f1_values = []
    losses = []
    for row in rows:
        f1 = caption_token_f1(str(row["generated"]), str(row["expected"]))
        _require(abs(float(row["token_f1"]) - f1) <= 1e-9,
                 f"{name} row F1 differs from raw text")
        f1_values.append(f1)
        loss = _finite(row["response_loss"], f"{name} row loss",
                       minimum=0.0)
        losses.append(loss)
    _require(abs(caption["token_f1"] - sum(f1_values) / cases) <= 1e-9,
             f"{name}.token_f1 differs from raw rows")
    _require(caption["loss_cases"] == cases,
             f"{name}.loss_cases does not cover the cases")
    _require(abs(caption["teacher_forced_response_loss"]
                 - sum(losses) / cases) <= 1e-9,
             f"{name} response loss differs from raw rows")
    _require(caption["claim"]
             == "narrow_alignment_metric_only_no_semantic_quality_claim",
             f"{name} claim boundary differs")


def validate_score_report(report: Mapping[str, Any],
                          protocol: Mapping[str, Any] | None = None) -> None:
    _exact_keys(report, {
        "format", "version", "stage", "arm", "protocol_sha256",
        "run_report_sha256", "reload_report_sha256", "data_pins",
        "engineering", "learned_quality",
        "eligibility", "claims", "audit_scored",
        "external_final_consumed"}, "score_report")
    _require(report["format"] == SCORE_REPORT_FORMAT,
             "Unknown score report format")
    _require(report["stage"] == "dev" and report["arm"] in ARMS,
             "Score report stage or arm differs")
    _require(_is_sha256(report["protocol_sha256"])
             and _is_sha256(report["run_report_sha256"])
             and _is_sha256(report["reload_report_sha256"]),
             "Score report hashes are invalid")
    _validate_pin(report["data_pins"], "score_report.data_pins")
    _require(report["audit_scored"] is False
             and report["external_final_consumed"] is False,
             "Dev score report may not consume final evidence")
    engineering = report["engineering"]
    learned = report["learned_quality"]
    eligibility = report["eligibility"]
    _exact_keys(eligibility, {
        "overall_baseline_eligible", "baseline_eligible",
        "component_nonregression_pass", "learned_quality_pass",
        "curriculum_qualified", "overall_eligible",
        "audit_scores_present"}, "score.eligibility")
    _require(eligibility["audit_scores_present"] is False,
             "Dev score report may not contain audit scores")
    _require(eligibility["overall_baseline_eligible"] is False,
             "Overall baseline eligibility must stay false")
    _require(engineering["pass"]
             == (not engineering["failures"])
             and (not engineering["failures"]
                  or any(value is not True
                         for value in engineering["checks"].values())),
             "Engineering pass contradicts its recorded failures")
    if engineering["pass"]:
        _require(all(value is True
                     for value in engineering["checks"].values()),
                 "Passing engineering contains a failed check")
    _require(learned["pass"] == (not learned["failures"]),
             "Learned-quality pass contradicts its failures")
    procedural = learned["procedural_dev"]
    _validate_procedural_metrics(procedural, "score.procedural_dev")
    unknown = procedural["unknown"]
    paired = procedural["paired"]
    gates = learned["retention"]["gates"]
    controls = procedural["controls"]
    correct_control = controls["correct"]
    answerable_unknown_rate = unknown["false_abstention_on_answerable"]
    expected_failures: list[str] = []
    if correct_control["accuracy"] < gates["min_correct_accuracy"]:
        expected_failures.append("min_correct_accuracy")
    if correct_control["unique_worlds"] < gates["min_unique_families"]:
        expected_failures.append("min_unique_families")
    if answerable_unknown_rate > gates["max_unknown_rate"]:
        expected_failures.append("max_unknown_rate")
    for control in ("wrong", "shuffled", "noimage"):
        if (paired.get(control, {}).get("correct_margin", -1.0)
                < gates[f"min_correct_margin_{control}"]):
            expected_failures.append(f"min_correct_margin_{control}")
    if unknown["abstention_accuracy"] < gates["min_unknown_accuracy"]:
        expected_failures.append("min_unknown_accuracy")
    if unknown["false_answer_on_abstain"] > gates["max_false_answer_on_abstain"]:
        expected_failures.append("max_false_answer_on_abstain")
    if (unknown["false_abstention_on_answerable"]
            > gates["max_false_abstention_on_answerable"]):
        expected_failures.append("max_false_abstention_on_answerable")
    _require(sorted(learned["failures"]) == sorted(expected_failures),
             "Learned-quality failures differ from the declared gates")
    _require(eligibility["learned_quality_pass"] == learned["pass"],
             "Eligibility learned-quality result differs")
    _require(unknown["abstain_cases"] > 0
             and unknown["answerable_cases"] > 0,
             "Unknown metrics lack abstain or answerable cases")
    _require(abs(unknown["abstention_accuracy"]
                 + unknown["false_answer_on_abstain"] - 1.0) <= 1e-9,
             "Abstention accuracy and false-answer rate are inconsistent")
    _require(0.0 <= unknown["false_abstention_on_answerable"] <= 1.0,
             "False-abstention rate is outside [0, 1]")
    _require(procedural["cases"]
             == (unknown["abstain_cases"] + unknown["answerable_cases"]),
             "Procedural case counts are inconsistent")
    caption = learned["external_development_caption"]
    _validate_caption_metrics(caption, "score.caption")
    retention = learned["retention"]
    gates = retention["gates"]
    if protocol is not None:
        _require(gates == protocol["evaluation"]["gates"],
                 "Learned retention gates differ from protocol gates")
        bootstrap = procedural["bootstrap"]
        _require(bootstrap["seed"] == protocol["seed"]
                 and bootstrap["resamples"]
                 == protocol["evaluation"]["bootstrap"]["resamples"]
                 and bootstrap["confidence"]
                 == protocol["evaluation"]["bootstrap"]["confidence"],
                 "Score bootstrap settings differ from the protocol")
        _require(retention["policy"] == protocol["retention"]["policy"],
                 "Retention policy differs from the protocol")
    _require(retention["policy"] == "component_nonregression.v1"
             and retention["baseline_correct"] == 22
             and retention["baseline_cases"] == 32,
             "Retention policy or baseline differs")
    _require(type(retention["measured_cases"]) is int
             and retention["measured_cases"] == retention["baseline_cases"],
             "Retention case counts differ")
    measured_correct = retention["measured_correct"]
    _require(measured_correct is None
             or (type(measured_correct) is int
                 and 0 <= measured_correct <= retention["measured_cases"]),
             "Retention measured correct count is impossible")
    baseline_accuracy = retention["baseline_correct"] / retention["baseline_cases"]
    measured_accuracy = (None if measured_correct is None
                         else measured_correct
                         / retention["measured_cases"])
    _require(0.0 <= baseline_accuracy <= 1.0
             and (measured_accuracy is None
                  or 0.0 <= measured_accuracy <= 1.0),
             "Retention accuracy is outside [0, 1]")
    _require(isinstance(retention["token_parity"], bool),
             "Retention token parity is not boolean")
    _require(retention["measured_accuracy"] == measured_accuracy,
             "Retention measured accuracy is inconsistent")
    _require(retention["baseline_accuracy"] == baseline_accuracy,
             "Retention baseline accuracy is inconsistent")
    expected_drop = (None if measured_accuracy is None
                     else baseline_accuracy - measured_accuracy)
    _require(retention["retention_drop"] == expected_drop,
             "Retention drop is inconsistent")
    baseline_eligible = (baseline_accuracy
                         >= float(gates["min_text_retention_accuracy"]))
    _require(retention["baseline_eligible"] == baseline_eligible,
             "Retention baseline eligibility is inconsistent")
    component_failures = list(retention["component_failures"])
    expected_component: list[str] = []
    if measured_accuracy is None:
        expected_component.append("retention_unavailable")
    if retention["token_parity"] is not True:
        expected_component.append("text_tokens_changed")
    if (expected_drop is not None
            and expected_drop > float(gates["max_text_retention_drop"])):
        expected_component.append("max_text_retention_drop")
    _require(component_failures == expected_component,
             "Component failures differ from the declared retention policy")
    _require(retention["component_nonregression_pass"]
             == (not component_failures),
             "Component non-regression contradicts its failures")
    _require(retention["overall_eligible"]
             == (baseline_eligible and not component_failures),
             "Retention overall eligibility is inconsistent")
    _require(retention["overall_baseline_eligible"] is False,
             "Retention must keep the overall baseline ineligible")
    _require(eligibility["baseline_eligible"] == baseline_eligible,
             "Eligibility baseline result differs from retention")
    _require(eligibility["component_nonregression_pass"]
             == retention["component_nonregression_pass"],
             "Eligibility component result differs from retention")
    _require(eligibility["curriculum_qualified"]
             == (engineering["pass"] and learned["pass"]
                 and retention["component_nonregression_pass"]),
             "Curriculum qualification is inconsistent")
    _require(eligibility["overall_eligible"]
             == (eligibility["curriculum_qualified"] and baseline_eligible),
             "Overall eligibility is inconsistent")
    if not learned["pass"]:
        _require(eligibility["curriculum_qualified"] is False,
                 "Failed learned quality cannot be curriculum-qualified")
    validate_score_claims(report)


# ---------------------------------------------------------------------------
# Run/reload/finalize report validation
# ---------------------------------------------------------------------------


def _required_run_checks() -> set[str]:
    return {
        "protocol_pins_match",
        "data_seals_match",
        "implementations_match",
        "fresh_initialization",
        "not_warm_start",
        "projector_only_trainable",
        "nonprojector_gradients_absent",
        "response_only_labels",
        "backward_only_probe_labelled",
        "schedule_matches_protocol",
        "equal_compute_within_arm",
        "strict_extension_export",
        "no_base_copy",
        "dev_only_scored",
        "retention_measured",
        "retention_token_parity_checked",
        "audit_untouched",
        "external_final_untouched",
    }


def validate_run_report(report: Mapping[str, Any],
                        protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "format", "version", "stage", "arm", "protocol_sha256",
        "data_pins", "implementation_sha256", "initialization",
        "trainable", "gradient_audit", "optimizer_route", "optimizer_audit",
        "schedule", "lr_selection", "training", "retention", "dev",
        "extension", "checks", "audit_scored", "external_final_consumed",
        "costs", "claims", "implementation_runtime", "artifacts"}
    _require(isinstance(report, Mapping), "Run report must be a mapping")
    _exact_keys(report, required | {"_run_report_sha256"}, "run_report")
    _require(report["format"] == RUN_REPORT_FORMAT,
             "Unknown run report format")
    _require(report["version"] == VERSION, "Run report version differs")
    _require(report["stage"] == "train" and report["arm"] in ARMS,
             "Run report stage or arm differs")
    _require(report["protocol_sha256"] == _protocol_sha(protocol),
             "Run report protocol hash differs")
    _require(report["data_pins"] == protocol["data"]["manifest"],
             "Run report data pin differs")
    _require(report["implementation_sha256"]
             == protocol["implementation_sha256"],
             "Run report implementation pins differ")

    init = _exact_keys(report["initialization"], {
        "recipe", "seed", "projector_init_sha256", "warm_start",
        "matches_forbidden_parent"}, "run_report.initialization")
    _require(init["recipe"] == protocol["train"]["init_recipe"],
             "Initialization recipe differs")
    _require(init["seed"] == protocol["train"]["init_seed"],
             "Initialization seed differs")
    _require(_is_sha256(init["projector_init_sha256"]),
             "Projector initialization hash is invalid")
    _require(init["warm_start"] is None
             and init["matches_forbidden_parent"] is False,
             "Warm starts and rejected-parent reuse are forbidden")

    trainable = _exact_keys(report["trainable"], {
        "names", "parameter_count", "frozen_names", "trainable_count"},
        "run_report.trainable")
    _require(sorted(trainable["names"]) == sorted(PROJECTOR_TENSORS),
             "Trainable tensor inventory differs")
    _require(trainable["parameter_count"] == PROJECTOR_PARAMETER_COUNT,
             "Trainable parameter count differs")
    _require(trainable["trainable_count"] == len(PROJECTOR_TENSORS),
             "Trainable tensor count differs")
    _require(not trainable["frozen_names"]
             or all(not name.startswith("vision_projector.")
                    for name in trainable["frozen_names"]),
             "Frozen inventory includes projector tensors")

    gradient = _exact_keys(report["gradient_audit"], {
        "nonprojector_gradient_names", "projector_gradient_names",
        "backward_only_probe", "probe_applied_update", "probe_label",
        "loss", "projector_gradients_finite",
        "projector_gradients_nonzero"}, "run_report.gradient_audit")
    _require(gradient["nonprojector_gradient_names"] == [],
             "Non-projector tensors received gradients")
    _require(sorted(gradient["projector_gradient_names"])
             == sorted(PROJECTOR_TENSORS),
             "Projector gradient inventory differs")
    _require(gradient["backward_only_probe"] is True
             and gradient["probe_applied_update"] is False
             and gradient["probe_label"] == "backward_only_no_update",
             "Backward-only probe must be labelled and non-updating")

    route = report["optimizer_route"]
    _require(isinstance(route, list) and route, "Optimizer route is empty")
    names = [item["name"] for item in route]
    _require(sorted(names) == sorted(PROJECTOR_TENSORS),
             "Optimizer route inventory differs")
    algorithms = {item["algorithm"] for item in route}
    if report["arm"] == "muon8bit":
        _require(algorithms == {"muon8bit", "adamw"},
                 "Muon arm must route both algorithms")
        by_name = {item["name"]: item["algorithm"] for item in route}
        for name in PROJECTOR_TENSORS:
            expected = "muon8bit" if name in PROJECTOR_2D_TENSORS else "adamw"
            _require(by_name[name] == expected,
                     f"Wrong Muon route for {name}")
    else:
        _require(algorithms == {"adamw"},
                 "AdamW control must route every tensor to AdamW")
    audit_optimizer = _exact_keys(report["optimizer_audit"], {
        "optimizer", "parameter_dtype", "state_dtype",
        "groups"}, "run_report.optimizer_audit")
    _require(audit_optimizer["parameter_dtype"] == "float32",
             "Optimizer parameters must be FP32")
    if report["arm"] == "muon8bit":
        _require(audit_optimizer["optimizer"] == "muon8bit"
                 and "int8" in audit_optimizer["state_dtype"],
                 "Muon8bit state audit differs")
    else:
        _require(audit_optimizer["optimizer"] == "adamw"
                 and audit_optimizer["state_dtype"] == "float32",
                 "AdamW control state audit differs")
    groups = audit_optimizer["groups"]
    _require(isinstance(groups, list) and groups,
             "Optimizer group LRs are missing")
    group_lrs = {}
    for group in groups:
        _require(group.get("algorithm") in {"muon8bit", "adamw"}
                 and isinstance(group.get("param_names"), list)
                 and _finite(group.get("lr"), "optimizer group lr",
                             minimum=0.0) > 0.0,
                 "Optimizer group declaration is invalid")
        group_lrs[group["group_name"]] = float(group["lr"])
    if report["arm"] == "muon8bit":
        aux_lr = float(protocol["trainable_groups"]["adamw_aux"]["lr"])
        _require(group_lrs.get("vision_projector_muon8bit")
                 == float(report["lr_selection"]["selected_lr"]),
                 "Muon matrix LR differs from the selected LR")
        _require(group_lrs.get("vision_projector_adamw") == aux_lr,
                 "Muon auxiliary AdamW LR is not the fixed 1e-4")
        _require(aux_lr == 1e-4,
                 "The preregistered auxiliary AdamW LR must stay 1e-4")
    else:
        _require(set(group_lrs.values())
                 == {float(report["lr_selection"]["selected_lr"])},
                 "AdamW control group LR differs from the selected LR")

    schedule = report["schedule"]
    _require(schedule["schedule_sha256"]
             == protocol["schedule"]["schedule_sha256"],
             "Run schedule hash differs from the protocol")
    _require(schedule["updates"] == protocol["schedule"]["updates"],
             "Run update count differs")
    _require(schedule["examples"] == protocol["schedule"]["examples"],
             "Run example count differs")
    _require(schedule["completed_updates"] == schedule["updates"],
             "Run did not complete every declared update")
    _require(schedule["probe_updates_per_candidate"]
             == protocol["schedule"]["probe"]["updates_per_candidate"],
             "Probe update count differs")
    selection = report["lr_selection"]
    _require(selection["arm"] == report["arm"], "LR selection arm differs")
    _require(selection["split"] == "procedural_dev",
             "LR selection must use dev only")
    _require(selection["selected_lr"] in selection["candidates"],
             "Selected LR was not a declared candidate")
    _require(selection["measure"] == "mean_dev_response_loss",
             "LR selection measure differs")
    results = selection.get("results")
    _require(isinstance(results, list)
             and len(results) == len(selection["candidates"]),
             "LR probe results do not cover the declared candidates")
    _require(sorted(selection["candidates"])
             == sorted(protocol["arms"]["arms"][report["arm"]]
                       ["lr_probe_candidates"]),
             "LR probe candidates differ from the protocol")
    _require(selection.get("probe_case_ids")
             == protocol["schedule"]["dev_probe_case_ids"],
             "LR probe cases differ from the protocol")
    _require(len({float(item["lr"]) for item in results})
             == len(results),
             "LR probe candidate LRs are not unique")
    probe_updates = protocol["schedule"]["probe"]["updates_per_candidate"]
    probe_batch = protocol["schedule"]["probe"]["batch"]
    for item in results:
        _require(item.get("lr") in selection["candidates"],
                 "LR probe result used an undeclared LR")
        _finite(item.get("dev_loss"), "LR probe dev loss", minimum=0.0)
        _require(item.get("init_sha256")
                 == init["projector_init_sha256"],
                 "LR probe initialization differs from the final initialization")
        _require(item.get("updates") == probe_updates
                 and item.get("examples") == probe_updates * probe_batch
                 and item.get("trace_rows") == probe_updates,
                 "LR probe update/example/trace accounting differs")
        probe_losses = item.get("dev_losses")
        _require(isinstance(probe_losses, list)
                 and len(probe_losses) == len(selection["probe_case_ids"]),
                 "LR probe dev losses do not cover the protocol probe cases")
        for value in probe_losses:
            _finite(value, "LR probe per-case loss", minimum=0.0)
        _require(abs(item["dev_loss"]
                     - sum(probe_losses) / len(probe_losses)) <= 1e-9,
                 "LR probe dev loss differs from its per-case losses")
        name = lr_probe_artifact_name(item["lr"])
        _require(item.get("trace_sha256")
                 == report["artifacts"].get(name),
                 "LR probe trace hash differs from its artifact")
        _require(_is_sha256(item.get("trace_sha256")),
                 "LR probe trace hash is invalid")
    selected_result = min(
        (item for item in results
         if item["dev_loss"] == min(entry["dev_loss"] for entry in results)),
        key=lambda item: item["lr"])
    _require(selection["selected_lr"] == selected_result["lr"]
             and selection.get("selected_dev_loss")
             == selected_result["dev_loss"],
             "Selected LR/loss does not follow the declared selection rule")

    training = report["training"]
    _require(training["updates"] == schedule["updates"],
             "Training trace count differs")
    _require(training["examples"] == schedule["examples"],
             "Training example count differs")
    _require(training.get("max_seconds") == protocol["train"]["max_seconds"],
             "Training wall budget differs from the preregistered budget")
    _require(training["max_seconds"] <= protocol["limits"]["max_seconds"],
             "Training wall budget exceeds the run ceiling")
    _require(training["response_only_labels"] is True
             and training["nonprojector_gradients_absent"] is True
             and training["base_parameter_stamp_unchanged"] is True,
             "Training audit differs")
    _require(training["completed"] is True, "Training did not complete")

    retention = report["retention"]
    _require(retention["policy"] == "component_nonregression.v1",
             "Retention policy differs")
    _require(retention["baseline_correct"] == 22
             and retention["baseline_cases"] == 32,
             "Retention baseline differs")
    _require(retention["overall_baseline_eligible"] is False,
             "Baseline eligibility must stay false")
    _require(isinstance(retention["component_nonregression_pass"], bool),
             "Component non-regression result is missing")
    _require(retention.get("token_parity") is True,
             "Retention token parity must be measured")
    for name in ("pre_sha256", "post_sha256"):
        _require(_is_sha256(retention.get(name)),
                 f"Retention {name} pin is missing")

    dev = report["dev"]
    _require(dev["cases"] == protocol["splits"]["dev"]["procedural_rows"],
             "Dev procedural case count differs from the sealed protocol split")
    _require(_is_sha256(dev.get("case_ids_sha256")),
             "Dev case identity hash is missing")
    _require(dev["cases"] > 0 and _is_sha256(dev["generations_sha256"]),
             "Dev generations pin is missing")
    response_loss = _finite(dev.get("response_loss"), "dev response loss",
                            minimum=0.0)
    losses = dev.get("losses")
    _require(isinstance(losses, list) and len(losses) == dev["cases"],
             "Dev per-case losses do not cover the dev cases")
    for value in losses:
        _finite(value, "dev per-case loss", minimum=0.0)
    _require(abs(response_loss - sum(losses) / len(losses)) <= 1e-9,
             "Dev response loss differs from its per-case evidence")
    _require(dev.get("seal_sha256")
             == protocol["splits"]["dev"]["seal_sha256"],
             "Dev case seal differs from the protocol")
    for name in ("cases_sha256", "caption_cases_sha256",
                 "caption_labels_sha256", "caption_generations_sha256"):
        _require(_is_sha256(dev.get(name)), f"Dev {name} pin is missing")
    _require(dev["scored_splits"] == ["curriculum_dev", "diagnostic_dev",
                                      "external_development"],
             "Dev scoring splits differ")
    _require(dev["audit_touched"] is False,
             "Dev evaluation may not touch the audit split")
    artifacts = report["artifacts"]
    _require(isinstance(artifacts, Mapping)
             and set(artifacts) == set(run_artifact_names(
                 protocol, include_reload=False)),
             "Run artifact inventory differs from the declared inventory")
    for name, value in artifacts.items():
        _require(_is_sha256(value), f"Run artifact hash is invalid: {name}")
    extension = report["extension"]
    _require(extension["format"] == "replica.minicpm5.extensions.v4",
             "Strict extension export v4 is required")
    _require(extension["base_saved"] is False,
             "Extension must not save the native base")
    _require(_is_sha256(extension["weights_sha256"]),
             "Extension weights hash is invalid")
    names = extension.get("tensor_names")
    _require(isinstance(names, list)
             and set(PROJECTOR_TENSORS) <= set(names)
             and not any(str(name).startswith("base.") for name in names),
             "Extension tensor inventory lacks projector tensors or includes base")
    _require(report["audit_scored"] is False
             and report["external_final_consumed"] is False,
             "Training may not score audit or consume the final split")

    checks = report["checks"]
    _require(isinstance(checks, Mapping), "Run checks must be a mapping")
    derived_checks = {
        "protocol_pins_match": (
            report["protocol_sha256"] == _protocol_sha(protocol)
            and report["data_pins"] == protocol["data"]["manifest"]),
        "data_seals_match": (
            dev.get("seal_sha256")
            == protocol["splits"]["dev"]["seal_sha256"]),
        "implementations_match": (
            report["implementation_sha256"]
            == protocol["implementation_sha256"]),
        "fresh_initialization": all(
            item.get("init_sha256") == init["projector_init_sha256"]
            for item in results),
        "not_warm_start": (
            init["projector_init_sha256"]
            not in set(protocol["train"]["forbidden_parent_weights_sha256"])),
        "projector_only_trainable": (
            sorted(trainable["names"]) == sorted(PROJECTOR_TENSORS)
            and trainable["parameter_count"] == PROJECTOR_PARAMETER_COUNT),
        "nonprojector_gradients_absent": (
            gradient["nonprojector_gradient_names"] == []),
        "response_only_labels": (
            training.get("response_only_labels") is True),
        "backward_only_probe_labelled": (
            gradient["backward_only_probe"] is True
            and gradient["probe_applied_update"] is False
            and gradient["probe_label"] == "backward_only_no_update"),
        "schedule_matches_protocol": (
            schedule["schedule_sha256"]
            == protocol["schedule"]["schedule_sha256"]
            and schedule["updates"] == protocol["schedule"]["updates"]
            and schedule["examples"] == protocol["schedule"]["examples"]),
        "equal_compute_within_arm": (
            training["updates"] == protocol["schedule"]["updates"]
            and training["examples"] == protocol["schedule"]["examples"]),
        "strict_extension_export": (
            extension["format"] == "replica.minicpm5.extensions.v4"),
        "no_base_copy": extension["base_saved"] is False,
        "dev_only_scored": (
            dev["scored_splits"] == ["curriculum_dev", "diagnostic_dev",
                                     "external_development"]
            and dev["audit_touched"] is False
            and report["audit_scored"] is False),
        "retention_measured": (
            _is_sha256(retention.get("pre_sha256"))
            and _is_sha256(retention.get("post_sha256"))
            and retention.get("measured_correct") is not None
            and retention.get("measured_cases") == 32),
        "retention_token_parity_checked": (
            isinstance(retention.get("token_parity"), bool)),
        "audit_untouched": report["audit_scored"] is False,
        "external_final_untouched": (
            report["external_final_consumed"] is False),
        "retention_baseline_matches_declared": (
            retention.get("before_correct") == 22
            and retention.get("before_cases") == 32),
    }
    _require(set(checks) == set(derived_checks),
             "Run checks differ from the evidence-derived set")
    for name, value in derived_checks.items():
        _require(checks[name] is value,
                 f"Run check contradicts evidence: {name}")
    failed = sorted(name for name, value in derived_checks.items()
                    if value is not True)
    _require(not failed, f"Run checks failed: {failed}")
    costs = report["costs"]
    _require(costs.get("downloads") == 0, "Downloads are forbidden")
    _require(costs.get("workers") == 1, "Exactly one worker is required")
    _require(costs.get("train_end_sampled") is True,
             "train_end guard sample is missing from the report")
    runtime = report["implementation_runtime"]
    guard = runtime.get("guard") if isinstance(runtime, Mapping) else None
    if isinstance(guard, Mapping):
        observed_peak = guard.get("driver_peak",
                                  guard.get("peak_driver_bytes"))
        _require(observed_peak == costs.get("peak_driver_bytes"),
                 "Peak driver cost differs from the guard snapshot")
        events = guard.get("events")
        _require(isinstance(events, list), "Guard events are missing")
        _require(costs.get("train_end_sampled")
                 == any(event.get("event") == "train_end"
                        for event in events),
                 "train_end cost flag differs from guard events")
        sampled = [event.get("mps_driver_bytes") for event in events
                   if event.get("mps_driver_bytes") is not None]
        if sampled:
            _require(max(sampled) <= observed_peak,
                     "A sampled driver reading exceeds the guard peak")
        _require(guard.get("workers") == 1
                 and guard.get("cpu_fallback") is False,
                 "Guard snapshot differs from the one-worker MPS policy")
    _require(costs.get("peak_driver_bytes", 0)
             < protocol["resource_policy"]["driver_limit_bytes"],
             "Driver guard was exceeded")
    claims = _exact_keys(report["claims"], {
        "perception_quality", "semantic_quality", "overall_baseline_eligible",
        "scope"}, "run_report.claims")
    _require(claims["perception_quality"] is False
             and claims["semantic_quality"] is False
             and claims["overall_baseline_eligible"] is False,
             "Run claims exceed the admissible boundary")
    return {"engineering_pass": True, "engineering_failures": [],
            "checks": dict(sorted(checks.items()))}


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash exactly the bytes written by write_protocol."""
    return sha256_bytes(_json_bytes(protocol))


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _load_json(Path(path))
    validate_protocol(protocol)
    return protocol


def load_run_report(path: str | Path) -> dict[str, Any]:
    """Load a run report and pin its file bytes out of band."""
    path = Path(path)
    report = _load_json(path)
    report["_run_report_sha256"] = sha256_file(path)
    return report


_protocol_sha = protocol_sha256


def validate_lr_reports(report_a: Mapping[str, Any],
                        report_b: Mapping[str, Any]) -> None:
    _require(report_a["arm"] != report_b["arm"],
             "Equal-compute comparison needs the two distinct arms")
    for report in (report_a, report_b):
        losses = report["dev"].get("losses")
        _require(isinstance(losses, list)
                 and len(losses) == report["dev"]["cases"],
                 "Arm comparison lacks per-case dev losses")
        _require(abs(report["dev"]["response_loss"]
                     - sum(losses) / len(losses)) <= 1e-9,
                 "Arm comparison response loss differs from per-case evidence")
    for key in ("projector_init_sha256",):
        _require(report_a["initialization"][key]
                 == report_b["initialization"][key],
                 "Arms do not share the same fresh initialization")
    _require(report_a["schedule"]["schedule_sha256"]
             == report_b["schedule"]["schedule_sha256"],
             "Arms do not share the schedule")
    _require(report_a["schedule"]["updates"] == report_b["schedule"]["updates"]
             and report_a["schedule"]["examples"]
             == report_b["schedule"]["examples"],
             "Arms differ in updates/examples")
    _require(report_a["schedule"]["probe_updates_per_candidate"]
             == report_b["schedule"]["probe_updates_per_candidate"],
             "Arms differ in probe budget")
    _require(report_a["schedule"]["completed_updates"]
             == report_b["schedule"]["completed_updates"]
             == report_a["schedule"]["updates"],
             "Arms differ in completed updates")
    _require(report_a["training"]["max_seconds"]
             == report_b["training"]["max_seconds"],
             "Arms differ in training wall budget")
    _require(report_a["trainable"] == report_b["trainable"],
             "Arms differ in the trainable inventory")
    _require(report_a["protocol_sha256"] == report_b["protocol_sha256"]
             and report_a["data_pins"] == report_b["data_pins"]
             and report_a["implementation_sha256"]
             == report_b["implementation_sha256"],
             "Arms used different protocol/data/implementation pins")
    _require(report_a["dev"]["cases"] == report_b["dev"]["cases"],
             "Arms evaluated different dev case counts")


def validate_reload_report(report: Mapping[str, Any],
                           run_report: Mapping[str, Any],
                           protocol: Mapping[str, Any]) -> None:
    _exact_keys(report, {
        "format", "version", "run_report_sha256", "protocol_sha256",
        "arm", "zero_updates", "extension_weights_sha256",
        "dev_generations_equal", "retention_generations_equal",
        "extension_unchanged", "dev_cases_sha256",
        "dataset_dev_cases_sha256", "dev_generations_sha256",
        "retention_sha256", "payload_sha256", "checks", "claims", "guard",
        "packages"}, "reload_report")
    _require(report["format"] == RELOAD_REPORT_FORMAT,
             "Unknown reload report format")
    payload = {key: value for key, value in report.items()
               if key != "payload_sha256"}
    _require(report["payload_sha256"] == fingerprint(payload),
             "Reload report self-hash differs")
    _require(report["arm"] == run_report["arm"], "Reload arm differs")
    _require(report["protocol_sha256"] == _protocol_sha(protocol),
             "Reload protocol hash differs")
    _require(report["run_report_sha256"]
             == run_report["_run_report_sha256"],
             "Reload run report hash differs")
    _require(report["zero_updates"] is True, "Reload must perform zero updates")
    _require(report["extension_weights_sha256"]
             == run_report["extension"]["weights_sha256"],
             "Reload extension hash differs")
    _require(report["dev_generations_equal"] is True
             and report["retention_generations_equal"] is True
             and report["extension_unchanged"] is True,
             "Fresh-process reload is not bitwise identical")
    for name in ("zero_updates", "extension_unchanged", "dev_equal",
                 "retention_equal", "strict_export"):
        _require(report["checks"].get(name) is True,
                 f"Reload check failed or missing: {name}")
    _require(report["claims"]["bitwise_reload"] is True,
             "Reload bitwise claim is missing")
    _require(report["claims"]["quality_claim"] is False,
             "Reload makes no quality claim")
    _require(report["dev_cases_sha256"]
             == run_report["dev"]["cases_sha256"],
             "Reload dev cases differ from the frozen run")
    _require(report["dataset_dev_cases_sha256"]
             == run_report["dev"]["dataset_dev_cases_sha256"],
             "Reload dataset dev cases differ from the frozen run")
    _require(report["dev_generations_sha256"]
             == run_report["dev"]["generations_sha256"],
             "Reload dev generations hash differs from the frozen run")
    _require(report["retention_sha256"]
             == run_report["retention"]["post_sha256"],
             "Reload retention hash differs from the frozen run")
    for name in ("dev_cases_sha256", "dataset_dev_cases_sha256",
                 "dev_generations_sha256", "retention_sha256"):
        _require(_is_sha256(report[name]), f"Reload {name} is invalid")


FINALIZE_RESERVATION_ROOT = "reports/vision_curriculum_20261025/finalize_receipts"
FINALIZE_RESERVATION_FORMAT = "replica.vision_curriculum.finalize_reservation.v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Directory fsync is not available on every filesystem; the file itself
        # is always fsynced and read back.
        return


def _atomic_create(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, f"Atomic write stalled: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_parent(path)
    completed = path.read_bytes()
    _require(completed == payload,
             f"Atomic artifact differs after write: {path}")
    return sha256_bytes(completed)


def verified_finalize_seals(protocol: Mapping[str, Any],
                           dataset_seals: Mapping[str, Any],
                           recomputed_seals: Mapping[str, Any]
                           ) -> dict[str, str]:
    """One-shot keys must come from verified data, never a bare protocol field."""
    checks = (
        ("audit", "audit_seal_sha256", "audit"),
        ("external_final", "external_final_seal_sha256", "external_final"),
    )
    result = {}
    for protocol_name, dataset_name, split_name in checks:
        data_value = protocol["data"]["seals"][protocol_name]
        split_value = protocol["splits"][split_name]["seal_sha256"]
        verified = dataset_seals[dataset_name]
        recomputed = recomputed_seals[dataset_name]
        _require(verified == recomputed,
                 f"Dataset {protocol_name} seal differs from recomputed bytes")
        _require(data_value == split_value == verified,
                 f"Protocol {protocol_name} seal differs from verified data")
        result[protocol_name] = verified
    _require(protocol["data"]["seals"]["external_final_pointer"]
             == recomputed_seals["external_final_pointer_sha256"]
             == dataset_seals["external_final_pointer_sha256"],
             "External final pointer seal differs from verified data")
    return result


def finalize_receipt_paths(project_root: str | Path,
                           external_final_seal_sha256: str,
                           audit_seal_sha256: str) -> dict[str, Path]:
    """Separate one-shot namespaces keyed only by the immutable final seals.

    The external-final namespace deliberately ignores the curriculum manifest
    and protocol hash so a future manifest cannot re-score the same sealed
    external final. The procedural audit keeps its own namespace.
    """
    _require(_is_sha256(external_final_seal_sha256),
             "External final seal is invalid")
    _require(_is_sha256(audit_seal_sha256), "Audit seal is invalid")
    root = Path(project_root) / FINALIZE_RESERVATION_ROOT
    return {
        "root": root,
        "external_reservation":
            root / f"external_final_{external_final_seal_sha256}.reserve.json",
        "external_receipt":
            root / f"external_final_{external_final_seal_sha256}.json",
        "audit_reservation":
            root / f"audit_{audit_seal_sha256}.reserve.json",
        "audit_receipt": root / f"audit_{audit_seal_sha256}.json",
    }


def _seal_paths(protocol: Mapping[str, Any], project_root: str | Path
                ) -> dict[str, Path]:
    return finalize_receipt_paths(
        project_root,
        protocol["splits"]["external_final"]["seal_sha256"],
        protocol["splits"]["audit"]["seal_sha256"])


def reserve_finalize(protocol: Mapping[str, Any], protocol_sha256: str, *,
                     project_root: str | Path,
                     verified_seals: Mapping[str, str],
                     comparison: Mapping[str, Any],
                     selected_arm: str,
                     dev_report_sha256: str,
                     score_report_sha256: str) -> dict[str, Any]:
    """Atomically reserve both one-shot namespaces before any scoring."""
    paths = finalize_receipt_paths(project_root,
                                   verified_seals["external_final"],
                                   verified_seals["audit"])
    for name in ("external_receipt", "audit_receipt",
                 "external_reservation", "audit_reservation"):
        _require(not paths[name].exists(),
                 f"Finalize namespace already used: {paths[name].name}")
    _require(selected_arm in ARMS, "Selected arm is invalid")
    _require(_is_sha256(dev_report_sha256)
             and _is_sha256(score_report_sha256),
             "Finalize report hashes are invalid")
    payload = {
        "format": FINALIZE_RESERVATION_FORMAT,
        "version": VERSION,
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "audit_seal_sha256": verified_seals["audit"],
        "external_final_seal_sha256": verified_seals["external_final"],
        "arm_comparison": dict(comparison),
        "selected_arm": selected_arm,
        "dev_report_sha256": dev_report_sha256,
        "score_report_sha256": score_report_sha256,
        "reserved_at_utc": _utc_now(),
        "scoring_started": False,
    }
    value = _json_bytes(payload)
    external_sha = _atomic_create(paths["external_reservation"], value)
    audit_sha = _atomic_create(paths["audit_reservation"], value)
    return {"external": {"path": str(paths["external_reservation"]),
                         "sha256": external_sha},
            "audit": {"path": str(paths["audit_reservation"]),
                      "sha256": audit_sha},
            "id": fingerprint(payload),
            "payload": payload}


def require_global_finalize_allowed(project_root: str | Path,
                                    external_final_seal_sha256: str,
                                    audit_seal_sha256: str
                                    ) -> dict[str, Path]:
    paths = finalize_receipt_paths(project_root,
                                   external_final_seal_sha256,
                                   audit_seal_sha256)
    for name in ("external_receipt", "audit_receipt",
                 "external_reservation", "audit_reservation"):
        _require(not paths[name].exists(),
                 f"Finalize namespace already used: {paths[name].name}")
    return paths


def build_finalize_receipt(*, protocol: Mapping[str, Any],
                           comparison: Mapping[str, Any],
                           reservation: Mapping[str, Any],
                           selected_arm: str,
                           selected_checkpoint: Mapping[str, Any],
                           audit: Mapping[str, Any],
                           external_final: Mapping[str, Any],
                           dev_report_sha256: str,
                           score_report_sha256: str,
                           guard_report_sha256: str) -> dict[str, Any]:
    _require(selected_arm in ARMS, "Unknown finalize arm")
    _require(_is_sha256(dev_report_sha256)
             and _is_sha256(score_report_sha256)
             and _is_sha256(guard_report_sha256),
             "Finalize report hashes are invalid")
    _require(_is_sha256(reservation["id"])
             and _is_sha256(reservation["external"]["sha256"])
             and _is_sha256(reservation["audit"]["sha256"]),
             "Finalize reservation hashes are invalid")
    _require(audit.get("scored_once") is True,
             "Audit must be scored exactly once")
    _require(external_final.get("consumed_once") is True,
             "External final must be consumed exactly once")
    _require(external_final.get("rows") == protocol["splits"]
             ["external_final"]["rows"],
             "External final row count differs")
    _require(audit.get("seal_sha256")
             == protocol["splits"]["audit"]["seal_sha256"],
             "Audit seal differs from the protocol")
    _require(audit.get("cases") == protocol["splits"]["audit"]["rows"],
             "Audit case count differs from the protocol")
    _require(selected_checkpoint.get("weights_sha256")
             == comparison["reports"][selected_arm]["extension_weights_sha256"],
             "Selected checkpoint hash differs from the arm report")
    _require(selected_checkpoint.get("frozen_before_audit") is True,
             "Selected checkpoint was not frozen before audit")
    _require(comparison["selected_arm"] == selected_arm,
             "Comparison selected another arm")
    _require(comparison["reports"][selected_arm].get(
        "score_report_sha256") == score_report_sha256,
        "Selected score report hash differs from the comparison")
    receipt = {
        "format": FINALIZE_RECEIPT_FORMAT,
        "version": VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "data_pins": dict(protocol["data"]["manifest"]),
        "reservations": {"id": reservation["id"],
                         "external": dict(reservation["external"]),
                         "audit": dict(reservation["audit"])},
        "reservation_payload": bool(reservation.get("payload")) and
        dict(reservation["payload"]),
        "arm_comparison": dict(comparison),
        "selected_arm": selected_arm,
        "selected_checkpoint": dict(selected_checkpoint),
        "dev_report_sha256": dev_report_sha256,
        "score_report_sha256": score_report_sha256,
        "guard_report_sha256": guard_report_sha256,
        "audit": dict(audit),
        "external_final": dict(external_final),
        "consumed": {"audit_scored": True,
                      "external_final_consumed": True,
                      "authenticated_by": {
                          "external_final": reservation["external"]["sha256"],
                          "audit": reservation["audit"]["sha256"]},
                      "prepared_flags_unchanged": True},
        "scored_once": True,
        "repeat_finalize_refused": True,
        "claims": {"semantic_quality": False,
                   "perception_quality": False,
                   "overall_baseline_eligible": False},
    }
    validate_finalize_receipt(receipt, protocol, dev_report_sha256)
    return receipt


def validate_finalize_receipt(receipt: Mapping[str, Any],
                              protocol: Mapping[str, Any],
                              dev_report_sha256: str) -> None:
    _exact_keys(receipt, {
        "format", "version", "protocol_sha256", "data_pins", "reservations",
        "reservation_payload", "arm_comparison", "selected_arm",
        "selected_checkpoint",
        "dev_report_sha256", "score_report_sha256", "guard_report_sha256",
        "audit",
        "external_final", "consumed", "scored_once",
        "repeat_finalize_refused", "claims"}, "finalize_receipt")
    _require(receipt["format"] == FINALIZE_RECEIPT_FORMAT,
             "Unknown finalize receipt format")
    _require(receipt["protocol_sha256"] == _protocol_sha(protocol),
             "Finalize protocol hash differs")
    _require(receipt["data_pins"] == protocol["data"]["manifest"],
             "Finalize data pin differs")
    reservations = _exact_keys(receipt["reservations"], {
        "id", "external", "audit"}, "finalize_receipt.reservations")
    _require(_is_sha256(reservations["id"]),
             "Finalize reservation hash is invalid")
    expected_payload = receipt["reservation_payload"]
    expected_hash = sha256_bytes(_json_bytes(expected_payload))
    expected_names = {
        "external": f"external_final_{protocol['splits']['external_final']['seal_sha256']}.reserve.json",
        "audit": f"audit_{protocol['splits']['audit']['seal_sha256']}.reserve.json",
    }
    for name in ("external", "audit"):
        item = _exact_keys(reservations[name], {"path", "sha256"},
                           f"finalize_receipt.reservations.{name}")
        _require(item["sha256"] == expected_hash,
                 f"Finalize {name} reservation hash differs from its payload")
        _require(isinstance(item["path"], str) and item["path"]
                 and Path(item["path"]).name == expected_names[name]
                 and Path(item["path"]).parent.as_posix().endswith(
                     FINALIZE_RESERVATION_ROOT),
                 f"Finalize {name} reservation path is not seal-derived")
    payload = receipt["reservation_payload"]
    _require(isinstance(payload, Mapping)
             and fingerprint(payload) == reservations["id"],
             "Reservation payload hash differs from the reservation id")
    _require(payload.get("format") == FINALIZE_RESERVATION_FORMAT
             and payload.get("protocol_sha256") == receipt["protocol_sha256"]
             and payload.get("data_manifest_sha256")
             == receipt["data_pins"]["sha256"]
             and payload.get("arm_comparison") == receipt["arm_comparison"]
             and payload.get("selected_arm") == receipt["selected_arm"]
             and payload.get("dev_report_sha256")
             == receipt["dev_report_sha256"]
             and payload.get("score_report_sha256")
             == receipt["score_report_sha256"]
             and payload.get("audit_seal_sha256")
             == protocol["splits"]["audit"]["seal_sha256"]
             and payload.get("external_final_seal_sha256")
             == protocol["splits"]["external_final"]["seal_sha256"],
             "Reservation payload differs from the receipt evidence")
    _require(receipt["selected_arm"] in ARMS,
             "Finalize selected arm is invalid")
    comparison = receipt["arm_comparison"]
    _exact_keys(comparison, {"rule", "order", "reports", "selected_arm"},
                "finalize_receipt.arm_comparison")
    _require(list(comparison["order"]) == list(ARMS),
             "Arm comparison order differs")
    _require(set(comparison["reports"]) == set(ARMS),
             "Arm comparison does not cover both arms")
    _require(comparison["selected_arm"] == receipt["selected_arm"],
             "Arm comparison selection differs")
    rule = _exact_keys(comparison["rule"], {
        "measure", "tie_break", "both_arms_required"}, "arm rule")
    _require(rule["measure"] == "mean_dev_response_loss"
             and rule["tie_break"] == "declared_arm_order"
             and rule["both_arms_required"] is True,
             "Arm comparison rule differs")
    losses = {}
    for arm in ARMS:
        item = _exact_keys(comparison["reports"][arm], {
            "run_report_sha256", "score_report_sha256",
            "extension_weights_sha256", "projector_init_sha256",
            "schedule_sha256", "updates", "examples",
            "probe_updates_per_candidate", "training_max_seconds",
            "dev_cases", "dev_losses", "dev_response_loss"},
            f"arm comparison {arm}")
        _require(_is_sha256(item["run_report_sha256"])
                 and _is_sha256(item["score_report_sha256"])
                 and _is_sha256(item["extension_weights_sha256"])
                 and _is_sha256(item["projector_init_sha256"]),
                 f"Arm comparison hashes are invalid: {arm}")
        _require(item["updates"] == protocol["schedule"]["updates"]
                 and item["examples"] == protocol["schedule"]["examples"]
                 and item["probe_updates_per_candidate"]
                 == protocol["schedule"]["probe"]["updates_per_candidate"]
                 and item["training_max_seconds"]
                 == protocol["train"]["max_seconds"],
                 f"Arm comparison compute differs: {arm}")
        losses[arm] = _finite(item["dev_response_loss"],
                              f"{arm} dev response loss", minimum=0.0)
        _require(item["dev_cases"] == protocol["splits"]["dev"][
            "procedural_rows"],
                 f"Arm comparison dev case count differs: {arm}")
        arm_losses = item["dev_losses"]
        _require(isinstance(arm_losses, list)
                 and len(arm_losses) == item["dev_cases"],
                 f"Arm comparison dev losses are incomplete: {arm}")
        for value in arm_losses:
            _finite(value, f"{arm} comparison dev loss", minimum=0.0)
        _require(abs(losses[arm]
                     - sum(arm_losses) / len(arm_losses)) <= 1e-9,
                 f"Arm comparison response loss differs from raw losses: {arm}")
    _require(len({comparison["reports"][arm]["projector_init_sha256"]
                  for arm in ARMS}) == 1,
             "Arms do not share one initialization")
    _require(len({comparison["reports"][arm]["schedule_sha256"]
                  for arm in ARMS}) == 1,
             "Arms do not share one schedule")
    best_loss = min(losses.values())
    expected = next(arm for arm in ARMS
                    if losses[arm] == best_loss)
    _require(receipt["selected_arm"] == expected,
             "Selected arm does not follow the declared comparison rule")
    selected = comparison["reports"][receipt["selected_arm"]]
    _require(receipt["dev_report_sha256"] == selected["run_report_sha256"]
             == dev_report_sha256,
             "Finalize dev report hash differs")
    _require(_is_sha256(receipt["score_report_sha256"])
             and receipt["score_report_sha256"]
             == selected["score_report_sha256"],
             "Finalize selected score report hash differs")
    _require(_is_sha256(receipt["guard_report_sha256"]),
             "Finalize guard snapshot hash is invalid")
    checkpoint = _exact_keys(receipt["selected_checkpoint"], {
        "weights_sha256", "frozen_at_utc", "frozen_before_audit"},
        "finalize_receipt.selected_checkpoint")
    _require(checkpoint["weights_sha256"]
             == selected["extension_weights_sha256"],
             "Selected checkpoint differs from the selected arm")
    _require(checkpoint["frozen_before_audit"] is True,
             "Selected checkpoint was not frozen before audit")
    _nonempty(checkpoint["frozen_at_utc"], "checkpoint frozen time")
    audit = _exact_keys(receipt["audit"], {
        "scored_once", "seal_sha256", "cases", "cases_sha256",
        "case_ids_sha256", "generations_sha256", "score",
        "learned_quality_pass", "gate_failures"},
        "finalize_receipt.audit")
    _require(audit["scored_once"] is True
             and audit["seal_sha256"]
             == protocol["splits"]["audit"]["seal_sha256"]
             and audit["cases"] == protocol["splits"]["audit"]["rows"],
             "Audit evidence differs from the protocol")
    _require(_is_sha256(audit["generations_sha256"])
             and _is_sha256(audit["cases_sha256"])
             and _is_sha256(audit["case_ids_sha256"]),
             "Audit artifact hashes are invalid")
    audit_score = audit["score"]
    _require(isinstance(audit_score, Mapping)
             and audit_score.get("cases") == audit["cases"],
             "Audit score coverage differs")
    _validate_procedural_metrics(audit_score, "audit")
    _require(audit["case_ids_sha256"] == audit_score["case_ids_sha256"]
             == fingerprint(audit_score["case_ids"]),
             "Audit case identity differs from the scored rows")
    _require(audit_score["bootstrap"]["seed"] == protocol["seed"]
             and audit_score["bootstrap"]["resamples"]
             == protocol["evaluation"]["bootstrap"]["resamples"]
             and audit_score["bootstrap"]["confidence"]
             == protocol["evaluation"]["bootstrap"]["confidence"],
             "Audit bootstrap settings differ from the protocol")
    gates = protocol["evaluation"]["gates"]
    controls = audit_score["controls"]
    unknown = audit_score["unknown"]
    paired = audit_score["paired"]
    expected_audit_failures: list[str] = []
    correct_control = controls["correct"]
    if correct_control["accuracy"] < gates["min_correct_accuracy"]:
        expected_audit_failures.append("min_correct_accuracy")
    if correct_control["unique_worlds"] < gates["min_unique_families"]:
        expected_audit_failures.append("min_unique_families")
    if (unknown["false_abstention_on_answerable"]
            > gates["max_unknown_rate"]):
        expected_audit_failures.append("max_unknown_rate")
    for control in ("wrong", "shuffled", "noimage"):
        if (paired.get(control, {}).get("correct_margin", -1.0)
                < gates[f"min_correct_margin_{control}"]):
            expected_audit_failures.append(f"min_correct_margin_{control}")
    if unknown["abstention_accuracy"] < gates["min_unknown_accuracy"]:
        expected_audit_failures.append("min_unknown_accuracy")
    if (unknown["false_answer_on_abstain"]
            > gates["max_false_answer_on_abstain"]):
        expected_audit_failures.append("max_false_answer_on_abstain")
    if (unknown["false_abstention_on_answerable"]
            > gates["max_false_abstention_on_answerable"]):
        expected_audit_failures.append("max_false_abstention_on_answerable")
    _require(sorted(audit["gate_failures"]) == sorted(expected_audit_failures),
             "Audit gate failures differ from the declared gates")
    _require(audit["learned_quality_pass"]
             == (not audit["gate_failures"])
             and audit_score.get("learned_quality_pass")
             == audit["learned_quality_pass"]
             and audit_score.get("gate_failures")
             == audit["gate_failures"],
             "Audit quality result contradicts its failures")
    external = _exact_keys(receipt["external_final"], {
        "consumed_once", "seal_sha256", "rows", "cases_sha256",
        "case_ids_sha256", "generations_sha256", "caption"},
        "finalize_receipt.external_final")
    _require(external["consumed_once"] is True
             and external["seal_sha256"]
             == protocol["splits"]["external_final"]["seal_sha256"]
             and external["rows"] == protocol["splits"]["external_final"]["rows"],
             "External final evidence differs from the protocol")
    _require(_is_sha256(external["generations_sha256"])
             and _is_sha256(external["cases_sha256"])
             and _is_sha256(external["case_ids_sha256"]),
             "External final artifact hashes are invalid")
    caption = external["caption"]
    _require(caption.get("cases") == external["rows"],
             "External final caption coverage differs")
    _validate_caption_metrics(caption, "finalize.external_final.caption")
    _require(external["case_ids_sha256"] == caption["case_ids_sha256"],
             "External final case identity differs from the caption rows")
    consumed = _exact_keys(receipt["consumed"], {
        "audit_scored", "external_final_consumed", "authenticated_by",
        "prepared_flags_unchanged"}, "finalize_receipt.consumed")
    authenticated = _exact_keys(consumed["authenticated_by"], {
        "external_final", "audit"}, "finalize_receipt.consumed.authenticated_by")
    _require(consumed["audit_scored"] is True
             and consumed["external_final_consumed"] is True
             and authenticated["external_final"]
             == reservations["external"]["sha256"]
             and authenticated["audit"] == reservations["audit"]["sha256"]
             and consumed["prepared_flags_unchanged"] is True,
             "Final evidence consumption is not receipt-authenticated")
    _require(receipt["scored_once"] is True
             and receipt["repeat_finalize_refused"] is True,
             "Finalize is not one-shot")
    _require(receipt["claims"]["semantic_quality"] is False
             and receipt["claims"]["perception_quality"] is False
             and receipt["claims"]["overall_baseline_eligible"] is False,
             "Finalize claims exceed the boundary")


def write_finalize_receipt(paths: Mapping[str, str | Path],
                           receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_bytes(receipt)
    written: dict[str, Any] = {}
    external = Path(paths["external_receipt"])
    audit = Path(paths["audit_receipt"])
    _require(not external.exists() and not audit.exists(),
             "Finalize receipt already exists; repeat finalize is refused")
    try:
        external_hash = _atomic_create(external, payload)
        audit_hash = _atomic_create(audit, payload)
    except BaseException:
        if external.exists() and not audit.exists():
            external.unlink()
            _fsync_parent(external)
        raise
    _require(external_hash == audit_hash,
             "Audit and external final receipts differ")
    written["external_receipt"] = {"path": str(external),
                                    "sha256": external_hash}
    written["audit_receipt"] = {"path": str(audit),
                                 "sha256": audit_hash}
    return written
