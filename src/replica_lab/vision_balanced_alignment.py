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


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text()))


def load_prior_evaluation(project_root: str | Path,
                          config: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the prior evaluation identities used for freshness exclusion."""
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
                visible_objects: Sequence[Sequence[str]] | None = None
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


def _control_map(selected: Sequence[Mapping[str, Any]], *,
                 combined: Sequence[Mapping[str, Any]], seed_offset: int,
                 assets: Mapping[str, Mapping[str, Any]],
                 case_by_id: Mapping[str, Mapping[str, Any]],
                 donor_info, sibling_case, split_label: str,
                 include_wrong: bool = True) -> list[dict[str, Any]]:
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
            donor = case_by_id[donor_id]
            donor_asset = assets[donor["asset_id"]]
            _require(donor_asset["sha256"] != receiver,
                     "Control donor reuses the receiver image")
            if control == "wrong":
                _require(donor["task"] == case["task"],
                         "Wrong-image donor task differs")
                _require(donor["world_id"] != case["world_id"],
                         "Wrong-image donor world differs")
            else:
                _require(donor["world_id"] != case["world_id"],
                         "Shuffled donor world differs")
            entry[control] = donor_info(donor)
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
        entry["counterfactual_same_image"] = same_image
        rows.append(entry)
    _require({row["case_id"] for row in rows} == selected_ids,
             "Control map does not cover every selected case")
    return rows


def build_balanced_dataset(config: Mapping[str, Any],
                           project_root: str | Path
                           ) -> dict[str, Any]:
    config = validate_config(config)
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
            split_label=f"{split}_answer")
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
            split_label=f"{split}_caption", include_wrong=False)

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
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "seed": config["seed"],
        "case_namespace": config["procedural_overrides"]["id_prefix"],
        "fresh_evaluation": True,
        "consumed_stage_result_preserved": True,
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


def build_protocol(config: Mapping[str, Any], built: Mapping[str, Any],
                   parent_protocol: Mapping[str, Any],
                   project_root: str | Path | None = None) -> dict[str, Any]:
    config = validate_config(config)
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
    evaluation["gates"] = _evaluation_gates(config, manifest)
    evaluation["caption_max_new_tokens"] = limits["caption_max_new_tokens"]
    evaluation["max_new_tokens"] = limits["max_new_tokens"]
    protocol = {
        "format": PROTOCOL_FORMAT,
        "version": VERSION,
        "declared_at_utc": config["declared_at_utc"],
        "purpose": config["purpose"],
        "root": str(root),
        "seed": config["seed"],
        "parent": {
            "protocol": config["parent_protocol"]["path"],
            "protocol_sha256": config["parent_protocol"]["sha256"],
            "data_config": config["parent_data_config"]["path"],
            "data_config_sha256": config["parent_data_config"]["sha256"],
            "role": ("Sealed source, preprocessing, optimizer and retention "
                     "contracts only; no warm start and no reused parent "
                     "weights."),
        },
        "prior_consumed": {
            "data_root": config["prior_consumed"]["data_root"],
            "protocol": config["prior_consumed"]["protocol"],
            "manifest": config["prior_consumed"]["manifest"],
            "role": config["prior_consumed"]["role"],
        },
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
    _require(protocol.get("format") == PROTOCOL_FORMAT,
             "Prepared protocol format differs")
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
    _require(manifest.get("format") == MANIFEST_FORMAT,
             "Prepared manifest format differs")
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
