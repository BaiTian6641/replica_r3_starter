"""Balanced paired image-dependence diagnostic for the frozen vision path.

This module owns a fresh, deterministic procedural corpus of *paired* scenes.
Every pair shares one question and one image-level fork: exactly one decisive
visual attribute changes between the two images, and the verified answer
changes with it. Answer labels live only in verifier rows; model rows carry
only the image hash/path and the question.

The module also owns the closed-form CPU readouts and the dev/final evaluation
contract. Readouts are ridge regressions over frozen features; no parameter of
the encoder, decoder or projector is updated. A final holdout is scored exactly
once, and only after the exact dev-report hash is supplied.

Readouts compared (all declared before looking at any score):

* ``question_only`` - frozen MiniCPM5 question hidden state.
* ``pooled_image`` - globally pooled frozen SigLIP2 patches.
* ``pooled_image_question`` - pooled patches plus the question state.
* ``region_image`` - 2x2 quadrant means of the frozen patch grid, preserving
  coarse spatial layout instead of collapsing it to one global vector.
* ``region_image_question`` - quadrant means plus the question state.
* ``question_conditioned_region`` (PRIMARY) - one spatial ridge model per
  question-derived task; inference routes only from the question text, never
  from an oracle task or label.

Every image-conditioned readout is also scored against a deterministic
shuffled-image derangement control and against majority/uniform-chance
baselines. The primary readout supports image dependence only if its mean gain
is above the preregistered margin over *both* question-only and the shuffled
control and both world-clustered 95% lower bounds are strictly positive. There
is no default positive verdict.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import canonical_json, fingerprint
from .visual_data import (
    COLOR_NAMES,
    SHAPES,
    VisualDataError,
    decode_rgb_png,
    render_scene_png,
    scene_sha256 as scene_digest,
    write_exact as _visual_write_exact,
)


VERSION = "replica.vision_dependence.v2"
CONFIG_FORMAT = "replica.vision_dependence_config.v2"
MANIFEST_FORMAT = "replica.vision_dependence_manifest.v2"
PROTOCOL_FORMAT = "replica.vision_dependence.protocol.v2"
FEATURE_FORMAT = "replica.vision_dependence_feature_bundle.v2"
DEV_REPORT_FORMAT = "replica.vision_dependence.dev_report.v2"
FINAL_REPORT_FORMAT = "replica.vision_dependence.final_report.v2"
RECEIPT_FORMAT = "replica.vision_dependence.finalize_receipt.v2"

RENDERER_VERSION = "procedural_png_v1"
SOURCE_ID = "replica_lab:procedural_scene_png_v1"
SPLITS = ("train", "dev", "final")
TASKS = ("count", "presence", "color", "shape", "spatial")
COUNT_LABELS = ("2", "3", "4", "5")
MODEL_ROW_FIELDS = {"case_id", "pair_id", "asset_id", "image", "image_sha256",
                    "question"}
READOUT_ORDER = ("question_only", "pooled_image", "pooled_image_question",
                 "region_image", "region_image_question")
PRIMARY_READOUT = "question_conditioned_region"
RIDGE_PENALTY = 1.0
MIN_WORLDS_PER_TASK_SPLIT = 8
MAX_WORLDS_PER_TASK_SPLIT = 12
DEFAULT_REGION_GRID = (2, 2)
DECISION_MIN_GAIN = 0.05

# Object-level descriptor pools. Every (colour, shape) object used by a split,
# whether target, distractor or spatial decoy, comes from that split's pool, and
# the three pools are disjoint. Colours and shapes may repeat across pools only
# where label coverage requires it (train must answer every dev/final label), so
# the held-out unit is the full object descriptor, never the bare attribute.
SPLIT_DESCRIPTOR_POOLS: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("red", "circle"), ("blue", "circle"), ("green", "circle"),
        ("yellow", "circle"), ("purple", "circle"), ("orange", "circle"),
        ("red", "square"), ("red", "triangle"), ("red", "rectangle"),
    ),
    "dev": (
        ("blue", "triangle"), ("green", "triangle"), ("yellow", "triangle"),
        ("purple", "triangle"), ("blue", "square"), ("blue", "rectangle"),
    ),
    "final": (
        ("green", "square"), ("yellow", "square"), ("purple", "square"),
        ("orange", "square"), ("green", "rectangle"), ("orange", "rectangle"),
    ),
}
COLOR_TARGET_SHAPE = {"train": "circle", "dev": "triangle", "final": "square"}
SHAPE_TARGET_COLOR = {"train": "red", "dev": "blue", "final": "green"}
COUNT_DESCRIPTOR_PAIRS = {
    "train": (("red", "circle"), ("red", "square")),
    "dev": (("blue", "triangle"), ("blue", "square")),
    "final": (("green", "square"), ("green", "rectangle")),
}
PRESENCE_TARGET_DESCRIPTORS = {
    "train": ("red", "triangle"), "dev": ("green", "triangle"),
    "final": ("green", "rectangle"),
}
SPATIAL_DESCRIPTOR_SETS = {
    "train": (("red", "circle"), ("blue", "circle"), ("green", "circle"),
              ("yellow", "circle")),
    "dev": (("blue", "triangle"), ("green", "triangle"), ("yellow", "triangle"),
            ("purple", "triangle")),
    "final": (("green", "square"), ("yellow", "square"), ("purple", "square"),
              ("orange", "square")),
}
FINALIZE_RECEIPT_ROOT = "reports/vision_dependence_20261025/finalize_receipts"
RUN_REPORT_FORMAT = "replica.vision_dependence.run.v2"
GRID_LAYOUTS = {
    "row_major": (0, 1, 2, 3, 4, 5, 6, 7),
    "column_major": (0, 4, 1, 5, 2, 6, 3, 7),
    "row_major_shift": (2, 3, 4, 5, 6, 7, 0, 1),
    "column_major_shift": (3, 7, 2, 6, 1, 5, 0, 4),
}
SPLIT_GRID_LAYOUTS = {
    "train": ("row_major", "column_major"),
    "dev": ("row_major_shift",),
    "final": ("column_major_shift",),
}
SPATIAL_LAYOUTS = {
    "upper_left_right": (0, 1, 2, 3),
    "lower_left_right": (4, 5, 6, 7),
    "upper_right_left": (3, 2, 1, 0),
    "lower_right_left": (7, 6, 5, 4),
}
SPLIT_SPATIAL_LAYOUTS = {
    "train": ("upper_left_right", "lower_left_right"),
    "dev": ("upper_right_left",),
    "final": ("lower_right_left",),
}

IMPLEMENTATION_FILES = (
    "src/replica_lab/vision_dependence.py",
    "src/replica_lab/visual_data.py",
    "src/replica_lab/contracts.py",
    "src/replica_lab/vision_qualification.py",
    "src/replica_lab/architecture/vision.py",
    "src/replica_lab/architecture/loading.py",
    "src/replica_lab/architecture/minicpm5.py",
    "src/replica_lab/architecture/contracts.py",
    "src/replica_lab/model_adapter.py",
    "src/replica_lab/memory_system/ledger.py",
    "src/replica_lab/capacity_guard.py",
    "src/replica_lab/profiling.py",
    "scripts/prepare_vision_dependence.py",
    "scripts/run_vision_dependence.py",
    "scripts/run_capacity_preflight.py",
)

PROTOCOL_FIELDS = {
    "format", "version", "stage", "root", "purpose", "seed", "config_sha256",
    "splits", "tasks", "data", "vision", "vision_digest", "vision_source",
    "model", "readout", "limits", "resource_policy", "stop_conditions",
    "implementation_sha256", "zero_updates", "final_holdout_policy",
}
DATA_PIN_NAMES = ("manifest", "model_rows", "verifier_cases",
                  "verifier_pairs", "verifier_scenes")
DATA_FIELDS = {
    "root", *DATA_PIN_NAMES, "counts", "label_coverage",
    "combinations_disjoint", "combinations_sha256",
}
LIMIT_FIELDS = {
    "max_observations", "max_encoder_passes", "max_question_encodes",
    "max_rows", "max_assets", "max_seconds", "min_free_disk_bytes",
    "max_feature_bytes", "max_bootstrap_resamples", "max_updates",
}
RESOURCE_FIELDS = {
    "exclusive_mps_worker", "cpu_fallback", "driver_limit_bytes",
    "zero_updates", "device", "downloads",
}
READOUT_FIELDS = {
    "ridge_penalty", "primary", "comparators", "region_grid", "bootstrap",
    "decision", "shuffle",
}


class VisionDependenceError(ValueError):
    """Raised when paired data, protocol, features or evaluation drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisionDependenceError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise VisionDependenceError(str(error)) from error


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(config, Mapping), "Config must be a mapping")
    require_keys = {"format", "version", "seed", "data_root", "renderer",
                    "splits", "tasks", "worlds_per_task", "count_labels",
                    "readout", "limits", "resource_policy", "stop_conditions"}
    _require(set(config) == require_keys,
             f"Config fields differ: missing={sorted(require_keys - set(config))} "
             f"extra={sorted(set(config) - require_keys)}")
    _require(config["format"] == CONFIG_FORMAT, "Unknown config format")
    _require(config["version"] == VERSION, "Unknown config version")
    _require(type(config["seed"]) is int and config["seed"] >= 0,
             "seed must be a nonnegative integer")
    root = config["data_root"]
    _require(isinstance(root, str) and root and not Path(root).is_absolute()
             and ".." not in Path(root).parts, "data_root must be a safe relative path")
    _require(list(config["splits"]) == list(SPLITS), "Splits differ")
    _require(list(config["tasks"]) == list(TASKS), "Tasks differ")
    worlds = config["worlds_per_task"]
    _require(type(worlds) is int and MIN_WORLDS_PER_TASK_SPLIT <= worlds
             <= MAX_WORLDS_PER_TASK_SPLIT,
             "worlds_per_task must be in [8, 12]")
    _require(list(config["count_labels"]) == list(COUNT_LABELS),
             "count_labels differ")
    renderer = config["renderer"]
    _require(isinstance(renderer, Mapping) and set(renderer) == {
        "version", "width", "height", "image_format", "standard_library_only"},
        "Renderer fields differ")
    _require(renderer["version"] == RENDERER_VERSION, "Unknown renderer version")
    _require(type(renderer["width"]) is int and type(renderer["height"]) is int
             and renderer["width"] >= 64 and renderer["height"] >= 48,
             "Renderer size is invalid")
    _require(renderer["image_format"] == "png_rgb8",
             "Renderer must produce PNG RGB8")
    _require(renderer["standard_library_only"] is True,
             "Renderer must stay standard-library only")
    readout = config["readout"]
    _require(isinstance(readout, Mapping) and set(readout) == READOUT_FIELDS,
             "Readout fields differ")
    _require(not isinstance(readout["ridge_penalty"], bool)
             and isinstance(readout["ridge_penalty"], (int, float))
             and math.isfinite(float(readout["ridge_penalty"]))
             and readout["ridge_penalty"] > 0, "ridge_penalty must be positive")
    _require(readout["primary"] == PRIMARY_READOUT,
             "Primary readout must be declared as question_conditioned_region")
    _require(tuple(readout["comparators"]) == READOUT_ORDER,
             "Comparator readouts differ")
    _require(list(readout["region_grid"]) == list(DEFAULT_REGION_GRID),
             "region_grid must be [2, 2]")
    bootstrap = readout["bootstrap"]
    _require(isinstance(bootstrap, Mapping) and set(bootstrap) == {
        "resamples", "confidence", "cluster"}, "Bootstrap fields differ")
    _require(type(bootstrap["resamples"]) is int
             and 100 <= bootstrap["resamples"] <= 100_000,
             "Bootstrap resamples are outside the declared bound")
    _require(0.5 < float(bootstrap["confidence"]) < 1.0,
             "Bootstrap confidence is invalid")
    _require(bootstrap["cluster"] == "world",
             "Bootstrap must cluster by pair world")
    decision = readout["decision"]
    _require(isinstance(decision, Mapping) and set(decision) == {
        "min_gain", "min_lower_bound", "strict_lower_bound"},
        "Decision fields differ")
    _require(float(decision["min_gain"]) == DECISION_MIN_GAIN,
             "Preregistered minimum gain differs")
    _require(float(decision["min_lower_bound"]) == 0.0,
             "Preregistered lower bound differs")
    _require(decision["strict_lower_bound"] is True,
             "Lower bounds must be strictly positive")
    shuffle = readout["shuffle"]
    _require(isinstance(shuffle, Mapping) and set(shuffle) == {
        "policy", "seed_offset"}, "Shuffle fields differ")
    _require(shuffle["policy"] == "deterministic_derangement_within_split",
             "Shuffle policy differs")
    _require(type(shuffle["seed_offset"]) is int and shuffle["seed_offset"] >= 1,
             "Shuffle seed_offset must be positive")
    limits = config["limits"]
    _require(isinstance(limits, Mapping) and set(limits) == LIMIT_FIELDS,
             "Limit fields differ")
    for name in LIMIT_FIELDS:
        _require(type(limits[name]) is int and limits[name] >= 0,
                 f"Limit {name} must be a nonnegative integer")
    _require(limits["max_updates"] == 0, "This diagnostic permits zero updates")
    _require(limits["max_rows"] >= 1 and limits["max_assets"] >= 1
             and limits["max_seconds"] >= 1 and limits["max_bootstrap_resamples"]
             >= bootstrap["resamples"],
             "Runtime limits are inconsistent")
    policy = config["resource_policy"]
    _require(isinstance(policy, Mapping) and set(policy) == RESOURCE_FIELDS,
             "Resource policy fields differ")
    _require(policy["exclusive_mps_worker"] is True,
             "Exclusive MPS worker is required")
    _require(policy["cpu_fallback"] is False, "CPU fallback is forbidden")
    _require(type(policy["driver_limit_bytes"]) is int
             and policy["driver_limit_bytes"] >= 50 * 1024 ** 3,
             "Driver limit must be at least 50 GiB")
    _require(policy["zero_updates"] is True, "Zero-update policy is required")
    _require(policy["device"] == "mps", "Only the declared MPS device is allowed")
    _require(policy["downloads"] == 0, "Downloads are forbidden")
    stops = config["stop_conditions"]
    _require(isinstance(stops, list) and len(stops) >= 3
             and all(isinstance(item, str) and item.strip() for item in stops),
             "At least three nonempty stop conditions are required")
    return dict(config)


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(value)


# ---------------------------------------------------------------------------
# Scene primitives (deterministic; rendered by visual_data's PNG renderer).
# ---------------------------------------------------------------------------


def _grid_cells(width: int, height: int) -> list[tuple[int, int]]:
    xs = [int(width * fraction) for fraction in (0.22, 0.42, 0.62, 0.82)]
    ys = [int(height * fraction) for fraction in (0.38, 0.68)]
    return [(x, y) for y in ys for x in xs]


def _object_bbox(shape: str, center: Sequence[int], size: int
                 ) -> list[int]:
    half_w = int(size) + (2 if shape == "rectangle" else 0)
    x, y = int(center[0]), int(center[1])
    return [x - half_w, y - size, x + half_w, y + size]


def _scene(asset_id: str, *, width: int, height: int, layout: str,
           layout_order: Sequence[int], objects: Sequence[Mapping[str, Any]]
           ) -> dict[str, Any]:
    cells = _grid_cells(width, height)
    built = []
    seen_cells = set()
    for item in objects:
        cell = int(item["cell"])
        _require(0 <= cell < len(cells), "Cell index is outside the grid")
        _require(cell not in seen_cells, "Objects must occupy distinct cells")
        seen_cells.add(cell)
        center = cells[cell]
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
        "layout": {
            "template": layout,
            "cell_indices": [int(value) for value in layout_order],
            "grid": {"columns": 4, "rows": 2},
        },
        "objects": built,
    }


def _object_by_id(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {obj["id"]: obj for obj in scene["objects"]}


# ---------------------------------------------------------------------------
# World specifications: labels, target/distractor/layout combinations.
# ---------------------------------------------------------------------------


def _task_layouts(split: str, task: str) -> tuple[str, ...]:
    return SPLIT_SPATIAL_LAYOUTS[split] if task == "spatial" \
        else SPLIT_GRID_LAYOUTS[split]


def _pool(split: str) -> tuple[tuple[str, str], ...]:
    _require(split in SPLITS, f"Unknown split: {split}")
    return SPLIT_DESCRIPTOR_POOLS[split]


def _descriptors_with_shape(split: str, shape: str) -> list[tuple[str, str]]:
    return [desc for desc in _pool(split) if desc[1] == shape]


def _descriptors_with_color(split: str, color: str) -> list[tuple[str, str]]:
    return [desc for desc in _pool(split) if desc[0] == color]


def _color_answer_descriptors(split: str) -> list[tuple[str, str]]:
    return _descriptors_with_shape(split, COLOR_TARGET_SHAPE[split])


def _color_distractor_descriptors(split: str) -> list[tuple[str, str]]:
    return [desc for desc in _pool(split)
            if desc[1] != COLOR_TARGET_SHAPE[split]]


def _shape_answer_descriptors(split: str) -> list[tuple[str, str]]:
    return _descriptors_with_color(split, SHAPE_TARGET_COLOR[split])


def _shape_distractor_descriptors(split: str) -> list[tuple[str, str]]:
    return [desc for desc in _pool(split)
            if desc[0] != SHAPE_TARGET_COLOR[split]]


def _choose_subset(descriptors: Sequence[tuple[str, str]], index: int,
                   salt: int, count: int) -> list[tuple[str, str]]:
    options = list(combinations(descriptors, count))
    _require(bool(options), "Distractor descriptor pool is too small")
    return list(options[(index * 7 + 3 * salt) % len(options)])


def _color_pair(split: str, index: int
                ) -> tuple[tuple[str, str], tuple[str, str]]:
    answers = _color_answer_descriptors(split)
    _require(len(answers) >= 2 and len(answers) % 2 == 0,
             "Colour answer descriptors must be even and at least two")
    return (answers[(2 * index) % len(answers)],
            answers[(2 * index + 1) % len(answers)])


def _shape_pair(split: str, index: int
                ) -> tuple[tuple[str, str], tuple[str, str]]:
    answers = _shape_answer_descriptors(split)
    _require(len(answers) >= 2, "Shape answer descriptors must be at least two")
    return (answers[index % len(answers)],
            answers[(index + 1) % len(answers)])


def _swap_sides(task: str, split: str, index: int) -> bool:
    if task in {"presence", "spatial"}:
        return index % 2 == 1
    if task == "count":
        return (index // 2) % 2 == 1
    if task == "color":
        answers = _color_answer_descriptors(split)
        pair_types = len(answers) // 2
        return (index // pair_types) % 2 == 1
    return False


def _count_params(split: str, index: int, salt: int) -> dict[str, Any]:
    layouts = _task_layouts(split, "count")
    target, distractor = COUNT_DESCRIPTOR_PAIRS[split]
    base = 2 if index % 2 == 0 else 4
    # Distractor count varies independently of the base/label. This eliminates
    # a full total-object-count -> label mapping; totals 3 and 8 remain
    # partially informative on 12/72 count images, so no total is treated as
    # sufficient evidence for the target count.
    distractor_count = 1 + ((index + salt) % 3)
    return {
        "layout": layouts[(index + salt) % len(layouts)],
        "base_count": base,
        "target_color": target[0], "target_shape": target[1],
        "distractor_color": distractor[0],
        "distractor_shape": distractor[1],
        "distractor_count": distractor_count,
        "object_size": 8 + ((index + salt) % 4),
        "salt": salt,
    }


def _presence_params(split: str, index: int, salt: int) -> dict[str, Any]:
    layouts = _task_layouts(split, "presence")
    target = PRESENCE_TARGET_DESCRIPTORS[split]
    others = [desc for desc in _pool(split) if desc != target]
    count = 1 + ((index + salt) % 3)
    chosen = _choose_subset(others, index, salt, count)
    return {
        "layout": layouts[(index + salt) % len(layouts)],
        "target_color": target[0], "target_shape": target[1],
        "distractor_descriptors": [list(desc) for desc in chosen],
        "distractor_count": len(chosen),
        "placement_offset": (index + salt + 2 * SPLITS.index(split)) % 8,
        "object_size": 7 + ((index + 5 * salt) % 6),
        "salt": salt,
    }


def _color_params(split: str, index: int, salt: int) -> dict[str, Any]:
    layouts = _task_layouts(split, "color")
    first, second = _color_pair(split, index)
    distractors = _choose_subset(_color_distractor_descriptors(split), index,
                                 salt, 1 + ((index + salt) % 2))
    return {
        "layout": layouts[(index + salt) % len(layouts)],
        "target_shape": COLOR_TARGET_SHAPE[split],
        "from_color": first[0], "to_color": second[0],
        "distractor_descriptors": [list(desc) for desc in distractors],
        "distractor_count": len(distractors),
        "object_size": 7 + ((index + 5 * salt) % 6),
        "salt": salt,
    }


def _shape_params(split: str, index: int, salt: int) -> dict[str, Any]:
    layouts = _task_layouts(split, "shape")
    first, second = _shape_pair(split, index)
    distractors = _choose_subset(_shape_distractor_descriptors(split), index,
                                 salt, 1 + ((index + salt) % 2))
    return {
        "layout": layouts[(index + salt) % len(layouts)],
        "target_color": SHAPE_TARGET_COLOR[split],
        "from_shape": first[1], "to_shape": second[1],
        "distractor_descriptors": [list(desc) for desc in distractors],
        "distractor_count": len(distractors),
        "object_size": 7 + ((index + 5 * salt) % 6),
        "salt": salt,
    }


def _spatial_params(split: str, index: int, salt: int) -> dict[str, Any]:
    layouts = _task_layouts(split, "spatial")
    base = SPATIAL_DESCRIPTOR_SETS[split]
    rotation = (index + salt) % len(base)
    descriptors = [base[(rotation + offset) % len(base)]
                   for offset in range(len(base))]
    _require(len({desc[0] for desc in descriptors}) == len(descriptors),
             "Spatial descriptors need distinct colours")
    return {
        "layout": layouts[(index + 3 * salt) % len(layouts)],
        "a_color": descriptors[0][0], "a_shape": descriptors[0][1],
        "b_color": descriptors[1][0], "b_shape": descriptors[1][1],
        "d_color": descriptors[2][0], "d_shape": descriptors[2][1],
        "e_color": descriptors[3][0], "e_shape": descriptors[3][1],
        "object_size": 7 + ((index + 5 * salt) % 6),
        "salt": salt,
    }


_PARAM_BUILDERS = {
    "count": _count_params,
    "presence": _presence_params,
    "color": _color_params,
    "shape": _shape_params,
    "spatial": _spatial_params,
}


def _descriptor_list(values: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(value) for value in values]


def _spatial_answers(params: Mapping[str, Any], swap: bool
                     ) -> dict[str, str]:
    slots = SPATIAL_LAYOUTS[params["layout"]]
    a_cell = slots[3] if swap else slots[0]
    b_cell = slots[2]
    first_left = (a_cell % 4) < (b_cell % 4)
    return {"a": "yes" if first_left else "no",
            "b": "no" if first_left else "yes"}


def _combination_for(task: str, split: str, index: int,
                     params: Mapping[str, Any], swap: bool) -> dict[str, Any]:
    if task == "count":
        base = int(params["base_count"])
        return {
            "task": task, "layout": params["layout"],
            "base_count": base,
            "target": {"color": params["target_color"],
                       "shape": params["target_shape"]},
            "distractor": {"color": params["distractor_color"],
                           "shape": params["distractor_shape"],
                           "count": params["distractor_count"]},
            "object_size": params["object_size"],
            "answers": {"a": str(base + (1 if swap else 0)),
                        "b": str(base + (0 if swap else 1))},
        }
    if task == "presence":
        return {
            "task": task, "layout": params["layout"],
            "target": {"color": params["target_color"],
                       "shape": params["target_shape"]},
            "distractor": {
                "descriptors": _descriptor_list(sorted(
                    tuple(desc) for desc in params["distractor_descriptors"])),
                "count": params["distractor_count"]},
            "placement_offset": params["placement_offset"],
            "object_size": params["object_size"],
            "answers": {"a": "no" if swap else "yes",
                        "b": "yes" if swap else "no"},
        }
    if task == "color":
        first, second = _color_pair(split, index)
        return {
            "task": task, "layout": params["layout"],
            "target_shape": params["target_shape"],
            "answer_descriptors": {
                "a": [first[0], params["target_shape"]] if not swap
                     else [second[0], params["target_shape"]],
                "b": [second[0], params["target_shape"]] if not swap
                     else [first[0], params["target_shape"]],
            },
            "distractor": {
                "descriptors": _descriptor_list(sorted(
                    tuple(desc) for desc in params["distractor_descriptors"])),
                "count": params["distractor_count"]},
            "object_size": params["object_size"],
            "answers": {"a": second[0] if swap else first[0],
                        "b": first[0] if swap else second[0]},
        }
    if task == "shape":
        first, second = _shape_pair(split, index)
        return {
            "task": task, "layout": params["layout"],
            "target_color": params["target_color"],
            "answer_descriptors": {
                "a": [params["target_color"], first[1]],
                "b": [params["target_color"], second[1]],
            },
            "distractor": {
                "descriptors": _descriptor_list(sorted(
                    tuple(desc) for desc in params["distractor_descriptors"])),
                "count": params["distractor_count"]},
            "object_size": params["object_size"],
            "answers": {"a": first[1], "b": second[1]},
        }
    if task == "spatial":
        return {
            "task": task, "layout": params["layout"],
            "target": {"color": params["a_color"], "shape": params["a_shape"]},
            "other_target": {"color": params["b_color"],
                             "shape": params["b_shape"]},
            "decoys": [
                {"color": params["d_color"], "shape": params["d_shape"]},
                {"color": params["e_color"], "shape": params["e_shape"]},
            ],
            "object_size": params["object_size"],
            "answers": _spatial_answers(params, swap),
        }
    raise VisionDependenceError(f"Unknown task: {task}")


def _spec_descriptors(task: str, params: Mapping[str, Any]
                      ) -> list[tuple[str, str]]:
    if task == "count":
        return [(params["target_color"], params["target_shape"]),
                (params["distractor_color"], params["distractor_shape"])]
    if task == "presence":
        return [(params["target_color"], params["target_shape"])] + [
            tuple(desc) for desc in params["distractor_descriptors"]]
    if task == "color":
        return [(params["from_color"], params["target_shape"]),
                (params["to_color"], params["target_shape"])] + [
            tuple(desc) for desc in params["distractor_descriptors"]]
    if task == "shape":
        return [(params["target_color"], params["from_shape"]),
                (params["target_color"], params["to_shape"])] + [
            tuple(desc) for desc in params["distractor_descriptors"]]
    if task == "spatial":
        return [(params["a_color"], params["a_shape"]),
                (params["b_color"], params["b_shape"]),
                (params["d_color"], params["d_shape"]),
                (params["e_color"], params["e_shape"])]
    raise VisionDependenceError(f"Unknown task: {task}")


def _spec_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    worlds = int(config["worlds_per_task"])
    pools = {split: set(_pool(split)) for split in SPLITS}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(pools[left].isdisjoint(pools[right]),
                         f"Descriptor pools overlap across {left}/{right}")
    groups: list[dict[str, Any]] = []
    for split in SPLITS:
        for task in TASKS:
            for index in range(worlds):
                candidates = []
                seen: set[str] = set()
                for salt in range(128):
                    effective_salt = (salt + int(config["seed"]) % 128) % 128
                    params = _PARAM_BUILDERS[task](split, index, effective_salt)
                    for descriptor in _spec_descriptors(task, params):
                        _require(descriptor in pools[split],
                                 f"Descriptor outside the {split} pool: {descriptor}")
                    swap = _swap_sides(task, split, index)
                    combination = _combination_for(task, split, index, params,
                                                   swap)
                    key = canonical_json(combination)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({"task": task, "split": split,
                                       "index": index, "params": params,
                                       "swap": swap,
                                       "combination": combination,
                                       "combination_key": key})
                _require(len(candidates) >= 1,
                         f"No candidate for {split}/{task}/{index}")
                groups.append({"split": split, "task": task, "index": index,
                               "candidates": candidates})
    return groups


# ---------------------------------------------------------------------------
# Pair materialisation.
# ---------------------------------------------------------------------------


def _count_scene(asset_id: str, params: Mapping[str, Any], side: str,
                 width: int, height: int) -> dict[str, Any]:
    base = int(params["base_count"])
    total_targets = base + (1 if side == "b" else 0)
    max_targets = base + 1
    distractors = int(params["distractor_count"])
    order = GRID_LAYOUTS[params["layout"]]
    _require(max_targets + distractors <= len(order), "Count scene exceeds the grid")
    objects = []
    for position in range(total_targets):
        objects.append({"id": f"t{position}", "role": "target",
                        "shape": params["target_shape"],
                        "color": params["target_color"],
                        "cell": order[position], "size": params["object_size"]})
    for position in range(distractors):
        objects.append({"id": f"d{position}", "role": "distractor",
                        "shape": params["distractor_shape"],
                        "color": params["distractor_color"],
                        "cell": order[max_targets + position],
                        "size": params["object_size"]})
    return _scene(asset_id, width=width, height=height, layout=params["layout"],
                  layout_order=order, objects=objects)


def _presence_scene(asset_id: str, params: Mapping[str, Any], side: str,
                    width: int, height: int) -> dict[str, Any]:
    base = list(GRID_LAYOUTS[params["layout"]])
    offset = int(params["placement_offset"])
    order = tuple(base[offset:] + base[:offset])
    objects = []
    if side == "a":
        objects.append({"id": "t0", "role": "target",
                        "shape": params["target_shape"],
                        "color": params["target_color"], "cell": order[0],
                        "size": params["object_size"]})
    for position, descriptor in enumerate(params["distractor_descriptors"]):
        color, shape = descriptor
        objects.append({"id": f"d{position}", "role": "distractor",
                        "shape": shape, "color": color,
                        "cell": order[1 + position],
                        "size": params["object_size"]})
    return _scene(asset_id, width=width, height=height, layout=params["layout"],
                  layout_order=order, objects=objects)


def _color_scene(asset_id: str, params: Mapping[str, Any], side: str,
                 width: int, height: int) -> dict[str, Any]:
    order = GRID_LAYOUTS[params["layout"]]
    color = params["from_color"] if side == "a" else params["to_color"]
    objects = [{"id": "t0", "role": "target", "shape": params["target_shape"],
                "color": color, "cell": order[0],
                "size": params["object_size"]}]
    for position, descriptor in enumerate(params["distractor_descriptors"]):
        distractor_color, shape = descriptor
        objects.append({"id": f"d{position}", "role": "distractor",
                        "shape": shape, "color": distractor_color,
                        "cell": order[1 + position],
                        "size": params["object_size"]})
    return _scene(asset_id, width=width, height=height, layout=params["layout"],
                  layout_order=order, objects=objects)


def _shape_scene(asset_id: str, params: Mapping[str, Any], side: str,
                 width: int, height: int) -> dict[str, Any]:
    order = GRID_LAYOUTS[params["layout"]]
    shape = params["from_shape"] if side == "a" else params["to_shape"]
    objects = [{"id": "t0", "role": "target", "shape": shape,
                "color": params["target_color"], "cell": order[0],
                "size": params["object_size"]}]
    for position, descriptor in enumerate(params["distractor_descriptors"]):
        color, distractor_shape = descriptor
        objects.append({"id": f"d{position}", "role": "distractor",
                        "shape": distractor_shape, "color": color,
                        "cell": order[1 + position],
                        "size": params["object_size"]})
    return _scene(asset_id, width=width, height=height, layout=params["layout"],
                  layout_order=order, objects=objects)


def _spatial_scene(asset_id: str, params: Mapping[str, Any], side: str,
                   width: int, height: int) -> dict[str, Any]:
    slots = SPATIAL_LAYOUTS[params["layout"]]
    other_row_cell = 4 if slots[0] < 4 else 0
    a_cell = slots[0] if side == "a" else slots[3]
    objects = [
        {"id": "a0", "role": "target", "shape": params["a_shape"],
         "color": params["a_color"], "cell": a_cell,
         "size": params["object_size"]},
        {"id": "d0", "role": "distractor", "shape": params["d_shape"],
         "color": params["d_color"], "cell": slots[1],
         "size": params["object_size"]},
        {"id": "b0", "role": "target", "shape": params["b_shape"],
         "color": params["b_color"], "cell": slots[2],
         "size": params["object_size"]},
        {"id": "e0", "role": "distractor", "shape": params["e_shape"],
         "color": params["e_color"], "cell": other_row_cell,
         "size": params["object_size"]},
    ]
    return _scene(asset_id, width=width, height=height, layout=params["layout"],
                  layout_order=tuple(slots) + (other_row_cell,), objects=objects)


_SCENE_BUILDERS = {
    "count": _count_scene,
    "presence": _presence_scene,
    "color": _color_scene,
    "shape": _shape_scene,
    "spatial": _spatial_scene,
}


def _question(task: str, params: Mapping[str, Any]) -> str:
    if task == "count":
        return (f"How many {params['target_color']} {params['target_shape']} "
                "objects are in the image?")
    if task == "presence":
        return (f"Is there a {params['target_color']} {params['target_shape']} "
                "in the image?")
    if task == "color":
        return f"What color is the {params['target_shape']}?"
    if task == "shape":
        return f"What shape is the {params['target_color']} object?"
    if task == "spatial":
        return (f"Is the {params['a_color']} {params['a_shape']} to the left of "
                f"the {params['b_color']} {params['b_shape']}?")
    raise VisionDependenceError(f"Unknown task: {task}")


def _check(task: str, params: Mapping[str, Any], scene: Mapping[str, Any]
           ) -> dict[str, Any]:
    if task == "count":
        return {"kind": "count",
                "filter": {"color": params["target_color"],
                           "shape": params["target_shape"]}}
    if task == "presence":
        return {"kind": "absent", "color": params["target_color"],
                "shape": params["target_shape"]}
    if task == "color":
        return {"kind": "color_of", "object_id": "t0"}
    if task == "shape":
        return {"kind": "shape_of", "object_id": "t0"}
    if task == "spatial":
        return {"kind": "spatial", "first_id": "a0", "second_id": "b0",
                "relation": "left_of"}
    raise VisionDependenceError(f"Unknown task: {task}")


def _check_value(check: Mapping[str, Any], scene: Mapping[str, Any]) -> str:
    objects = scene["objects"]
    kind = check["kind"]
    if kind == "count":
        criteria = check.get("filter", {})
        matched = [obj for obj in objects
                   if all(obj[key] == value for key, value in criteria.items())]
        return str(len(matched))
    if kind == "absent":
        found = any(obj["color"] == check["color"]
                    and obj["shape"] == check["shape"] for obj in objects)
        return "yes" if found else "no"
    if kind == "color_of":
        return str(_object_by_id(scene)[check["object_id"]]["color"])
    if kind == "shape_of":
        return str(_object_by_id(scene)[check["object_id"]]["shape"])
    if kind == "spatial":
        first = _object_by_id(scene)[check["first_id"]]
        second = _object_by_id(scene)[check["second_id"]]
        relation = check["relation"]
        left = first["center"][0] < second["center"][0] if relation == "left_of" \
            else first["center"][0] > second["center"][0]
        return "yes" if left else "no"
    raise VisionDependenceError(f"Unknown check kind: {kind}")


def _decisive_change(task: str, first: Mapping[str, Any],
                     second: Mapping[str, Any]) -> dict[str, Any]:
    if task == "count":
        before = Counter(_signature(obj) for obj in first["objects"])
        after = Counter(_signature(obj) for obj in second["objects"])
        added = list((after - before).elements())
        removed = list((before - after).elements())
        _require((len(added) == 1 and not removed)
                 or (len(removed) == 1 and not added),
                 "Count pair must add or remove exactly one object")
        if added:
            return {"kind": "object_added",
                    "object": _signature_record(added[0])}
        return {"kind": "object_removed",
                "object": _signature_record(removed[0])}
    if task == "presence":
        before = Counter(_signature(obj) for obj in first["objects"])
        after = Counter(_signature(obj) for obj in second["objects"])
        removed = list((before - after).elements())
        added = list((after - before).elements())
        _require((len(removed) == 1 and not added)
                 or (len(added) == 1 and not removed),
                 "Presence pair must add or remove exactly one object")
        if added:
            return {"kind": "object_added",
                    "object": _signature_record(added[0])}
        return {"kind": "object_removed", "object": _signature_record(removed[0])}
    if task in {"color", "shape"}:
        first_by_id = _object_by_id(first)
        second_by_id = _object_by_id(second)
        _require(set(first_by_id) == set(second_by_id),
                 "Attribute pair object ids differ")
        changed = []
        for object_id in sorted(first_by_id):
            left, right = first_by_id[object_id], second_by_id[object_id]
            for key in ("shape", "color", "center"):
                if left[key] != right[key]:
                    changed.append((object_id, key, left[key], right[key]))
            if left["shape"] == right["shape"]:
                _require(left["bbox"] == right["bbox"],
                         "Bounding box changed without a shape change")
        _require(len(changed) == 1,
                 f"{task} pair must change exactly one attribute: {changed}")
        object_id, key, before, after = changed[0]
        _require(key == ("color" if task == "color" else "shape"),
                 f"{task} pair changed the wrong attribute: {key}")
        return {"kind": f"{key}_changed", "object_id": object_id,
                "from": before, "to": after}
    if task == "spatial":
        first_by_id = _object_by_id(first)
        second_by_id = _object_by_id(second)
        _require(set(first_by_id) == set(second_by_id),
                 "Spatial pair object ids differ")
        moved = []
        for object_id in sorted(first_by_id):
            left, right = first_by_id[object_id], second_by_id[object_id]
            for key in ("shape", "color"):
                _require(left[key] == right[key],
                         "Spatial pair changed shape or color")
            if left["center"] != right["center"]:
                _require(left["bbox"] != right["bbox"],
                         "Spatial pair center moved without its bounding box")
                moved.append((object_id, list(left["center"]),
                              list(right["center"])))
        _require(len(moved) == 1,
                 "Spatial pair must move exactly one object")
        object_id, before, after = moved[0]
        return {"kind": "position_changed", "object_id": object_id,
                "from_center": before, "to_center": after}
    raise VisionDependenceError(f"Unknown task: {task}")


def _signature(obj: Mapping[str, Any]) -> str:
    return canonical_json({
        "role": obj.get("role", "target"),
        "shape": obj["shape"], "color": obj["color"],
        "bbox": list(obj["bbox"]), "center": list(obj["center"])})


def _signature_record(value: str) -> dict[str, Any]:
    return json.loads(value)


# ---------------------------------------------------------------------------
# Dataset construction.
# ---------------------------------------------------------------------------


def _distractor_composition(scene: Mapping[str, Any]) -> list[list[Any]]:
    return sorted([
        [obj["shape"], obj["color"], int(obj["center"][0]),
         int(obj["center"][1]), int(obj["size"])]
        for obj in scene["objects"] if obj.get("role") == "distractor"
    ])


def _asset_entry(asset_id: str, world_id: str, pair_id: str, split: str,
                 task: str, side: str, path: str, png: bytes,
                 scene: Mapping[str, Any], combination: Mapping[str, Any],
                 change: Mapping[str, Any]) -> dict[str, Any]:
    entry = {
        "asset_id": asset_id,
        "world_id": world_id,
        "pair_id": pair_id,
        "split": split,
        "task": task,
        "side": side,
        "path": path,
        "bytes": len(png),
        "sha256": _sha256_bytes(png),
        "width": int(scene["width"]),
        "height": int(scene["height"]),
        "scene_sha256": scene_digest(scene),
        "combination": dict(combination),
        "decisive_change": dict(change),
    }
    entry["record_hash"] = fingerprint(entry)
    return entry


def _materialize_pair(spec: Mapping[str, Any], world_id: str,
                      width: int, height: int) -> dict[str, Any]:
    pair_id = f"{world_id}-p0"
    task = spec["task"]
    params = spec["params"]
    builder = _SCENE_BUILDERS[task]
    first_side, second_side = ("b", "a") if spec["swap"] else ("a", "b")
    scene_a = builder(f"{pair_id}-a", params, first_side, width, height)
    scene_b = builder(f"{pair_id}-b", params, second_side, width, height)
    change = _decisive_change(task, scene_a, scene_b)
    question = _question(task, params)
    check = _check(task, params, scene_a)
    answers = {side: _check_value(check, scene)
               for side, scene in (("a", scene_a), ("b", scene_b))}
    _require(answers["a"] != answers["b"],
             f"Pair answers must differ: {pair_id}")
    _require(answers == spec["combination"]["answers"],
             f"Combination answers differ from verified sides: {pair_id}")
    composition_a = _distractor_composition(scene_a)
    composition_b = _distractor_composition(scene_b)
    _require(composition_a == composition_b,
             f"Pair distractor composition differs: {pair_id}")
    png_a = render_scene_png(scene_a)
    png_b = render_scene_png(scene_b)
    _require(png_a != png_b, f"Pair images render identically: {pair_id}")
    return {"pair_id": pair_id, "task": task, "split": spec["split"],
            "params": params, "scene_a": scene_a, "scene_b": scene_b,
            "png_a": png_a, "png_b": png_b, "change": change,
            "question": question, "check": check, "answers": answers,
            "distractor_composition": composition_a}


def build_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the full corpus in memory: files, manifest, and pure metadata."""
    config = validate_config(config)
    width = int(config["renderer"]["width"])
    height = int(config["renderer"]["height"])
    groups = _spec_candidates(config)
    files: dict[str, bytes] = {}
    assets: list[dict[str, Any]] = []
    worlds: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    used_combinations: dict[str, set[str]] = {split: set() for split in SPLITS}
    used_scenes: set[str] = set()
    used_pngs: set[str] = set()
    world_index = 0
    for group in groups:
        split = group["split"]
        committed = None
        for candidate in group["candidates"]:
            if candidate["combination_key"] in used_combinations[split]:
                continue
            world_id = f"vd2-w{world_index:03d}"
            materialized = _materialize_pair(candidate, world_id, width, height)
            scene_hashes = {scene_digest(materialized["scene_a"]),
                            scene_digest(materialized["scene_b"])}
            png_hashes = {_sha256_bytes(materialized["png_a"]),
                          _sha256_bytes(materialized["png_b"])}
            if scene_hashes & used_scenes or png_hashes & used_pngs:
                continue
            used_combinations[split].add(candidate["combination_key"])
            used_scenes.update(scene_hashes)
            used_pngs.update(png_hashes)
            committed = (candidate, materialized)
            break
        _require(committed is not None,
                 f"Could not materialise a unique world for "
                 f"{split}/{group['task']}/{group['index']}")
        spec, pair_data = committed
        world_id = f"vd2-w{world_index:03d}"
        pair_id = pair_data["pair_id"]
        task = spec["task"]
        scene_a = pair_data["scene_a"]
        scene_b = pair_data["scene_b"]
        change = pair_data["change"]
        question = pair_data["question"]
        check = pair_data["check"]
        answers = pair_data["answers"]
        composition_a = pair_data["distractor_composition"]
        pair_assets = []
        for side, scene, png in (("a", scene_a, pair_data["png_a"]),
                                 ("b", scene_b, pair_data["png_b"])):
            asset_id = f"{pair_id}-{side}"
            path = f"assets/{asset_id}.png"
            asset = _asset_entry(asset_id, world_id, pair_id, split, task, side,
                                 path, png, scene, spec["combination"], change)
            files[path] = png
            assets.append(asset)
            pair_assets.append(asset_id)
            case_id = asset_id
            rows.append({
                "case_id": case_id,
                "pair_id": pair_id,
                "asset_id": asset_id,
                "image": path,
                "image_sha256": asset["sha256"],
                "question": question,
            })
            cases.append({
                "case_id": case_id,
                "pair_id": pair_id,
                "asset_id": asset_id,
                "world_id": world_id,
                "split": split,
                "task": task,
                "question": question,
                "answer": answers[side],
                "label": answers[side],
                "check": {**check, "expected": answers[side]},
                "scene_sha256": asset["scene_sha256"],
                "label_visibility": "verifier_only",
            })
            scenes.append({"asset_id": asset_id, "scene_sha256": asset["scene_sha256"],
                           "scene": scene})
        pairs.append({
            "pair_id": pair_id,
            "world_id": world_id,
            "split": split,
            "task": task,
            "question": question,
            "combination": dict(spec["combination"]),
            "answers": dict(answers),
            "distractor_composition": composition_a,
            "decisive_change": change,
            "sides": {
                "a": {"asset_id": pair_assets[0], "answer": answers["a"],
                      "scene_sha256": assets[-2]["scene_sha256"]},
                "b": {"asset_id": pair_assets[1], "answer": answers["b"],
                      "scene_sha256": assets[-1]["scene_sha256"]},
            },
            "answers_differ": True,
        })
        worlds.append({
            "world_id": world_id,
            "split": split,
            "task": task,
            "pair_ids": [pair_id],
            "asset_ids": pair_assets,
            "combination": dict(spec["combination"]),
        })
        world_index += 1
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(used_combinations[left].isdisjoint(
                    used_combinations[right]),
                    f"Combinations overlap across {left}/{right}")

    files["model_rows.jsonl"] = _jsonl_bytes(rows)
    files["verifier_cases.jsonl"] = _jsonl_bytes(cases)
    files["verifier_pairs.jsonl"] = _jsonl_bytes(pairs)
    files["verifier_scenes.jsonl"] = _jsonl_bytes(scenes)
    counts = _counts(assets, worlds, pairs, rows, cases, scenes)
    coverage = _label_coverage(cases)
    balance = _label_balance(cases)
    combinations = {
        "keys": {split: sorted({canonical_json(world["combination"])
                                for world in worlds
                                if world["split"] == split})
                 for split in SPLITS},
        "counts": {split: len({canonical_json(world["combination"])
                               for world in worlds
                               if world["split"] == split})
                   for split in SPLITS},
        "disjoint": all(
            {canonical_json(world["combination"]) for world in worlds
             if world["split"] == left}.isdisjoint(
                {canonical_json(world["combination"]) for world in worlds
                 if world["split"] == right})
            for left in SPLITS for right in SPLITS if left < right),
        "policy": ("Every combination is unique; target descriptors, "
                   "distractor descriptors, layouts and worlds are disjoint "
                   "across train/dev/final; exact distractor compositions are "
                   "split-disjoint."),
    }
    descriptor_pools = {split: [list(desc) for desc in _pool(split)]
                        for split in SPLITS}
    distractor_compositions = {
        split: sorted({fingerprint(pair["distractor_composition"])
                       for pair in pairs if pair["split"] == split})
        for split in SPLITS
    }
    compositions_disjoint = all(
        set(distractor_compositions[left]).isdisjoint(
            set(distractor_compositions[right]))
        for left in SPLITS for right in SPLITS if left < right)
    _require(compositions_disjoint,
             "Exact distractor compositions overlap across splits")
    for task in TASKS:
        train_labels = set(coverage[task]["train"])
        for split in ("dev", "final"):
            _require(set(coverage[task][split]) <= train_labels,
                     f"{split} labels are not covered by train for {task}")
    _require(combinations["disjoint"], "Combination holdout is not disjoint")
    generation = fingerprint({
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "seed": config["seed"],
        "config": fingerprint(config),
        "assets": [asset["sha256"] for asset in assets],
        "scenes": [scene["scene_sha256"] for scene in scenes],
        "cases": [fingerprint({"case_id": case["case_id"],
                               "question": case["question"],
                               "answer": case["answer"],
                               "check": case["check"]}) for case in cases],
        "worlds": [{"world_id": world["world_id"], "split": world["split"],
                    "pair_ids": world["pair_ids"],
                    "asset_ids": world["asset_ids"]} for world in worlds],
    })
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "stage": "v2",
        "status": "candidate_not_admitted",
        "seed": config["seed"],
        "config_fingerprint": fingerprint(config),
        "renderer": dict(config["renderer"]),
        "provenance": {
            "origin": "internal_owned_procedural_generation",
            "source_id": SOURCE_ID,
            "generation": "balanced_paired_scene_generator_v2",
            "external_assets": 0,
            "external_terms": "not_applicable",
            "redistribution": "project_controlled_pending_review",
            "excludes_v1_cases": True,
            "v1_overlap": 0,
            "note": ("Fresh paired scenes; no bytes, case ids or questions are "
                     "taken from data/vision_preparation_v1."),
        },
        "counts": counts,
        "roles": {
            "splits": {
                split: {
                    "usage": {"train": "fit_readout",
                              "dev": "score_dev_once",
                              "final": "score_once_holdout"}[split],
                    "worlds": counts["split_worlds"][split],
                    "pairs": counts["split_pairs"][split],
                    "assets": counts["split_assets"][split],
                    "rows": counts["split_model_rows"][split],
                }
                for split in SPLITS
            },
            "pair_sides": {"a": "pair_first", "b": "pair_second"},
            "label_visibility": "verifier_only",
            "model_row_fields": sorted(MODEL_ROW_FIELDS),
        },
        "label_coverage": coverage,
        "label_balance": balance,
        "descriptor_pools": descriptor_pools,
        "holdout_policy": {
            "unit": "object_descriptor(color,shape)",
            "split_pools_disjoint": True,
            "distractor_compositions_disjoint": True,
            "bare_attribute_exception": (
                "Colour and shape values may recur across splits only where "
                "train label coverage requires the same answer value; no "
                "cross-split object descriptor, target pool, distractor pool "
                "or exact distractor composition is reused."),
        },
        "distractor_compositions": {
            "keys": distractor_compositions,
            "disjoint": compositions_disjoint,
            "counts": {split: len(keys)
                       for split, keys in distractor_compositions.items()},
        },
        "combinations": combinations,
        "worlds": worlds,
        "pairs": pairs,
        "assets": assets,
        "cases": cases,
        "files": {
            "model_rows": {"path": "model_rows.jsonl",
                           "sha256": _sha256_bytes(files["model_rows.jsonl"])},
            "verifier_cases": {"path": "verifier_cases.jsonl",
                               "sha256": _sha256_bytes(files["verifier_cases.jsonl"])},
            "verifier_pairs": {"path": "verifier_pairs.jsonl",
                               "sha256": _sha256_bytes(files["verifier_pairs.jsonl"])},
            "verifier_scenes": {"path": "verifier_scenes.jsonl",
                                "sha256": _sha256_bytes(files["verifier_scenes.jsonl"])},
        },
        "model_row_fields": sorted(MODEL_ROW_FIELDS),
        "generation_fingerprint": generation,
    }
    return {"files": files, "manifest": manifest, "assets": assets,
            "rows": rows, "cases": cases, "pairs": pairs, "scenes": scenes,
            "worlds": worlds, "counts": counts, "label_coverage": coverage,
            "label_balance": balance, "combinations": combinations}


def _counts(assets: Sequence[Mapping[str, Any]],
            worlds: Sequence[Mapping[str, Any]],
            pairs: Sequence[Mapping[str, Any]],
            rows: Sequence[Mapping[str, Any]],
            cases: Sequence[Mapping[str, Any]],
            scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_assets = Counter(asset["split"] for asset in assets)
    split_worlds = Counter(world["split"] for world in worlds)
    split_pairs = Counter(pair["split"] for pair in pairs)
    split_rows = Counter(case["split"] for case in cases)
    worlds_by_task_split = {task: {split: sum(
        1 for world in worlds
        if world["task"] == task and world["split"] == split)
        for split in SPLITS} for task in TASKS}
    return {
        "worlds": len(worlds),
        "pairs": len(pairs),
        "assets": len(assets),
        "verifier_cases": len(cases),
        "verifier_scenes": len(scenes),
        "model_rows": len(rows),
        "asset_bytes": sum(int(asset["bytes"]) for asset in assets),
        "task_histogram": dict(sorted(Counter(case["task"] for case in cases).items())),
        "split_worlds": {split: split_worlds.get(split, 0) for split in SPLITS},
        "split_pairs": {split: split_pairs.get(split, 0) for split in SPLITS},
        "split_assets": {split: split_assets.get(split, 0) for split in SPLITS},
        "split_model_rows": {split: split_rows.get(split, 0) for split in SPLITS},
        "worlds_by_task_split": worlds_by_task_split,
        "worlds_per_pair": 1,
        "pairs_per_world": 1,
        "images_per_pair": 2,
        "model_rows_per_asset": 1,
    }


def _label_coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        task: {split: sorted({case["label"] for case in cases
                              if case["task"] == task and case["split"] == split})
               for split in SPLITS}
        for task in TASKS
    }


def _label_balance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for task in TASKS:
        result[task] = {}
        for split in SPLITS:
            labels = sorted({case["label"] for case in cases
                             if case["task"] == task and case["split"] == split})
            histogram = {label: sum(
                1 for case in cases
                if case["task"] == task and case["split"] == split
                and case["label"] == label) for label in labels}
            values = list(histogram.values())
            result[task][split] = {
                "labels": labels,
                "histogram": histogram,
                "max_minus_min": (max(values) - min(values)) if values else 0,
            }
            _require(bool(values) and max(values) - min(values) <= 1,
                     f"Labels are not balanced for {task}/{split}: {histogram}")
    counts = result["count"]["train"]["labels"]
    _require(len(counts) >= 3 and len(set(counts)) >= 3,
             "Count labels must be varied (not all 4)")
    return result


# ---------------------------------------------------------------------------
# Dataset loading, auditing and hashing.
# ---------------------------------------------------------------------------


class DependenceDataset:
    """Read-only verified view over one prepared paired corpus."""

    def __init__(self, root: Path, manifest: Mapping[str, Any],
                 rows: Sequence[Mapping[str, Any]],
                 cases: Sequence[Mapping[str, Any]],
                 pairs: Sequence[Mapping[str, Any]],
                 scenes: Sequence[Mapping[str, Any]]):
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.rows = [dict(row) for row in rows]
        self.cases = [dict(case) for case in cases]
        self.pairs = [dict(pair) for pair in pairs]
        self.scenes = [dict(scene) for scene in scenes]
        self.worlds = [dict(world) for world in manifest["worlds"]]
        self._cases = {case["case_id"]: case for case in self.cases}
        self._assets = {asset["asset_id"]: asset for asset in manifest["assets"]}
        self._scenes = {scene["asset_id"]: scene for scene in self.scenes}

    def case(self, case_id: str) -> Mapping[str, Any]:
        _require(case_id in self._cases, f"Unknown case: {case_id}")
        return self._cases[case_id]

    def asset(self, asset_id: str) -> Mapping[str, Any]:
        _require(asset_id in self._assets, f"Unknown asset: {asset_id}")
        return self._assets[asset_id]

    def scene(self, asset_id: str) -> Mapping[str, Any]:
        _require(asset_id in self._scenes, f"Unknown scene: {asset_id}")
        return self._scenes[asset_id]["scene"]

    def image_path(self, asset_id: str) -> Path:
        asset = self.asset(asset_id)
        path = (self.root / asset["path"]).resolve()
        _require(self.root.resolve() in path.parents,
                 "Asset path escapes the dataset root")
        return path

    def label(self, case_id: str) -> str:
        return str(self.case(case_id)["label"])

    def rows_for_split(self, split: str) -> list[Mapping[str, Any]]:
        _require(split in SPLITS, f"Unknown split: {split}")
        return [row for row in self.rows
                if self._cases[row["case_id"]]["split"] == split]


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line]


def _safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"Unsafe relative path: {value}")
    candidate = (Path(root).resolve() / relative).resolve()
    _require(Path(root).resolve() in candidate.parents,
             f"Path escapes the dataset root: {value}")
    return candidate


def load_dataset(root: str | Path) -> DependenceDataset:
    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"Dataset manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    _require(manifest.get("format") == MANIFEST_FORMAT,
             "Unknown vision dependence manifest format")
    _require(manifest.get("version") == VERSION,
             "Unknown vision dependence manifest version")
    rows = _load_jsonl(root / "model_rows.jsonl")
    cases = _load_jsonl(root / "verifier_cases.jsonl")
    pairs = _load_jsonl(root / "verifier_pairs.jsonl")
    scenes = _load_jsonl(root / "verifier_scenes.jsonl")
    return DependenceDataset(root, manifest, rows, cases, pairs, scenes)


def dataset_pins(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = _load_json(root / "manifest.json")
    pins = {"manifest": {"path": "manifest.json",
                         "sha256": _sha256_file(root / "manifest.json")}}
    for name in DATA_PIN_NAMES:
        if name == "manifest":
            continue
        info = manifest["files"][name]
        path = root / info["path"]
        _require(path.is_file(), f"Pinned data file is missing: {path}")
        _require(_sha256_file(path) == info["sha256"],
                 f"Pinned data file bytes changed: {path}")
        pins[name] = {"path": info["path"], "sha256": info["sha256"]}
    return pins


def audit_dataset(root: str | Path,
                  *, config: Mapping[str, Any] | None = None
                  ) -> DependenceDataset:
    """Re-read every artifact and re-verify pairs, labels, splits and hashes."""
    root = Path(root)
    dataset = load_dataset(root)
    manifest = dataset.manifest
    if config is not None:
        config = validate_config(config)
        _require(manifest["config_fingerprint"] == fingerprint(config),
                 "Dataset config fingerprint differs")
        _require(manifest["seed"] == config["seed"], "Dataset seed differs")
    for name in DATA_PIN_NAMES:
        if name == "manifest":
            continue
        info = manifest["files"][name]
        path = _safe_relative(root, info["path"])
        _require(path.is_file(), f"Data file is missing: {name}")
        _require(_sha256_file(path) == info["sha256"],
                 f"Data file bytes changed: {name}")
    assets = manifest["assets"]
    _require(len(assets) == len(dataset.rows) == len(dataset.cases) == len(dataset.scenes),
             "Per-row artifact counts differ")
    _require(len(manifest["pairs"]) == len(manifest["worlds"]) == len(dataset.pairs),
             "Pair/world counts differ")
    _require(len({asset["asset_id"] for asset in assets}) == len(assets),
             "Asset ids are not unique")
    _require(len({asset["scene_sha256"] for asset in assets}) == len(assets),
             "Scene truth is not unique")
    for asset in assets:
        path = _safe_relative(root, asset["path"])
        _require(path.is_file(), f"Asset is missing: {asset['asset_id']}")
        data = path.read_bytes()
        _require(len(data) == int(asset["bytes"]), "Asset byte length differs")
        _require(_sha256_bytes(data) == asset["sha256"],
                 f"Asset bytes changed: {asset['asset_id']}")
        width, height, png_rows = decode_rgb_png(data)
        _require((width, height) == (int(asset["width"]), int(asset["height"])),
                 "Asset dimensions differ")
        record = {key: value for key, value in asset.items()
                  if key != "record_hash"}
        _require(fingerprint(record) == asset["record_hash"],
                 f"Asset record hash differs: {asset['asset_id']}")
        scene = dataset.scene(asset["asset_id"])
        _require(scene_digest(scene) == asset["scene_sha256"],
                 f"Scene hash differs: {asset['asset_id']}")
        _require((int(scene["width"]), int(scene["height"])) == (width, height),
                 "Scene dimensions differ from the rendered asset")
        _require(render_scene_png(scene) == data,
                 f"Rendered asset is not reproducible: {asset['asset_id']}")
        cells = [tuple(obj["center"]) for obj in scene["objects"]]
        _require(len(cells) == len(set(cells)), "Scene objects share a cell")
        for obj in scene["objects"]:
            _require(obj.get("role") in {"target", "distractor"},
                     f"Scene object role is invalid: {asset['asset_id']}")
            descriptor = (obj["color"], obj["shape"])
            _require(descriptor in {tuple(item) for item in
                                    manifest["descriptor_pools"][asset["split"]]},
                     f"Scene descriptor is outside the {asset['split']} pool: "
                     f"{descriptor}")
        del png_rows
    scene_by_asset = {scene["asset_id"]: scene for scene in dataset.scenes}
    case_by_id = dataset._cases
    asset_by_id = dataset._assets
    _require(set(asset_by_id) == set(scene_by_asset), "Scene and asset ids differ")
    question_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in dataset.cases:
        asset = asset_by_id[case["asset_id"]]
        question_labels[(asset["sha256"], case["question"])].add(case["label"])
    for (image_sha, question), labels in question_labels.items():
        _require(len(labels) == 1,
                 f"One image/question carries conflicting labels: {image_sha}")
    _require(set(case_by_id) == {asset["asset_id"] for asset in assets},
             "Case ids differ from asset ids")
    _require({row["case_id"] for row in dataset.rows} == set(case_by_id),
             "Model row case ids differ")
    for row in dataset.rows:
        _require(set(row) == MODEL_ROW_FIELDS,
                 f"Model row fields differ: {sorted(row)}")
        _require("answer" not in row and "label" not in row and "task" not in row,
                 "Model rows must not expose labels or oracle tasks")
        case = case_by_id[row["case_id"]]
        asset = asset_by_id[row["asset_id"]]
        _require(row["pair_id"] == case["pair_id"] == asset["pair_id"],
                 "Model row pairing differs")
        _require(row["image"] == asset["path"], "Model row image path differs")
        _require(row["image_sha256"] == asset["sha256"],
                 "Model row image hash differs")
        _require(row["question"] == case["question"],
                 "Model row question differs from its verifier case")
        _require(case["label_visibility"] == "verifier_only",
                 "Verifier case visibility differs")
        _require(case["task"] in TASKS, "Unknown verifier task")
        _require(case["split"] == asset["split"], "Case split differs from asset")
        scene = scene_by_asset[asset["asset_id"]]["scene"]
        computed = _check_value({key: case["check"][key] for key in
                                 ("kind", "filter", "color", "shape", "object_id",
                                  "first_id", "second_id", "relation")
                                 if key in case["check"]}, scene)
        _require(computed == case["check"]["expected"] == case["answer"]
                 == case["label"], "Verifier check does not match the scene")
        _require(case["scene_sha256"] == asset["scene_sha256"],
                 "Case scene hash differs")
    pair_by_id = {pair["pair_id"]: pair for pair in dataset.pairs}
    _require(len(pair_by_id) == len(dataset.pairs), "Pair ids are not unique")
    world_by_id = {world["world_id"]: world for world in dataset.worlds}
    _require(len(world_by_id) == len(dataset.worlds), "World ids are not unique")
    for pair in dataset.pairs:
        _require(set(pair["sides"]) == {"a", "b"}, "Pair sides differ")
        first = pair["sides"]["a"]["asset_id"]
        second = pair["sides"]["b"]["asset_id"]
        first_asset, second_asset = asset_by_id[first], asset_by_id[second]
        _require(first_asset["world_id"] == second_asset["world_id"]
                 == pair["world_id"], "Pair crosses worlds")
        _require(first_asset["split"] == second_asset["split"] == pair["split"],
                 "Pair crosses splits")
        _require(first_asset["task"] == second_asset["task"] == pair["task"],
                 "Pair crosses tasks")
        first_case, second_case = case_by_id[first], case_by_id[second]
        _require(first_case["question"] == second_case["question"]
                 == pair["question"], "Pair questions differ")
        _require(first_case["answer"] == pair["sides"]["a"]["answer"]
                 and second_case["answer"] == pair["sides"]["b"]["answer"],
                 "Pair answers differ from cases")
        _require(first_case["answer"] != second_case["answer"],
                 "Pair answers must differ")
        _require(pair.get("answers") == {"a": first_case["answer"],
                                         "b": second_case["answer"]},
                 "Pair side answers differ from the verifier cases")
        _require(pair["combination"].get("answers") == pair["answers"],
                 "Combination answers differ from verified pair answers")
        if pair["task"] in {"color", "shape"}:
            for side, scene in (("a", scene_by_asset[first]["scene"]),
                                ("b", scene_by_asset[second]["scene"])):
                target = _object_by_id(scene)["t0"]
                _require(pair["combination"]["answer_descriptors"][side]
                         == [target["color"], target["shape"]],
                         "Combination answer descriptors are reversed")
        computed = _decisive_change(pair["task"], scene_by_asset[first]["scene"],
                                    scene_by_asset[second]["scene"])
        _require(computed == pair["decisive_change"],
                 f"Deceptive pair difference: {pair['pair_id']}")
        _require(first_case["answer"] != second_case["answer"],
                 "Pair must change its verified answer")
        _require(pair["answers_differ"] is True, "Pair answer flag differs")
        _require(first_asset["decisive_change"] == second_asset["decisive_change"]
                 == pair["decisive_change"], "Asset decisive change differs")
        _require(first_asset["combination"] == second_asset["combination"]
                 == pair["combination"], "Pair combination differs")
        first_scene, second_scene = scene_by_asset[first]["scene"], \
            scene_by_asset[second]["scene"]
        _require((first_scene["width"], first_scene["height"])
                 == (second_scene["width"], second_scene["height"]),
                 "Pair dimensions differ")
        _require(first_scene["layout"]["template"] == second_scene["layout"]["template"],
                 "Pair layout differs")
        _require(pair.get("distractor_composition")
                 == _distractor_composition(first_scene),
                 "Pair distractor composition differs from its scenes")
        _require(decode_rgb_png(_png_bytes(root, first_asset))[2]
                 != decode_rgb_png(_png_bytes(root, second_asset))[2],
                 "Pair images render identically")
    world_assets: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        world_assets[asset["world_id"]].append(asset["asset_id"])
        world = world_by_id.get(asset["world_id"])
        _require(world is not None, f"Asset has no world: {asset['asset_id']}")
        _require(world["split"] == asset["split"], "World split differs from asset")
        _require(world["task"] == asset["task"], "World task differs from asset")
    for world in dataset.worlds:
        _require(world["asset_ids"] == world_assets[world["world_id"]],
                 "World asset list differs")
        _require(len(world["pair_ids"]) == 1, "Every world owns exactly one pair")
        _require(world["pair_ids"][0] in pair_by_id,
                 "World pair is missing")
        _require(pair_by_id[world["pair_ids"][0]]["combination"]
                 == world["combination"], "World combination differs from its pair")
    for split in SPLITS:
        for task in TASKS:
            count = sum(1 for world in dataset.worlds
                        if world["split"] == split and world["task"] == task)
            _require(MIN_WORLDS_PER_TASK_SPLIT <= count <= MAX_WORLDS_PER_TASK_SPLIT,
                     f"World count outside [8, 12] for {split}/{task}: {count}")
    split_worlds = {split: {world["world_id"] for world in dataset.worlds
                            if world["split"] == split} for split in SPLITS}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(split_worlds[left].isdisjoint(split_worlds[right]),
                         f"Worlds overlap across {left}/{right}")
    split_combinations = {split: {canonical_json(world["combination"])
                                  for world in dataset.worlds
                                  if world["split"] == split} for split in SPLITS}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                _require(split_combinations[left].isdisjoint(
                    split_combinations[right]),
                    f"Combinations overlap across {left}/{right}")
    split_descriptors = {
        split: {(obj["color"], obj["shape"])
                for asset in assets if asset["split"] == split
                for obj in scene_by_asset[asset["asset_id"]]["scene"]["objects"]}
        for split in SPLITS
    }
    declared_pools = {split: {tuple(item) for item in
                              manifest["descriptor_pools"][split]}
                      for split in SPLITS}
    for split in SPLITS:
        _require(split_descriptors[split] <= declared_pools[split],
                 f"Used descriptors escape the {split} pool")
        for other in SPLITS:
            if split < other:
                _require(split_descriptors[split].isdisjoint(
                    split_descriptors[other]),
                    f"Object descriptors overlap across {split}/{other}")
    split_compositions = {
        split: {fingerprint(pair["distractor_composition"])
                for pair in dataset.pairs if pair["split"] == split}
        for split in SPLITS
    }
    for split in SPLITS:
        _require(split_compositions[split]
                 == set(manifest["distractor_compositions"]["keys"][split]),
                 f"Distractor compositions differ for {split}")
        for other in SPLITS:
            if split < other:
                _require(split_compositions[split].isdisjoint(
                    split_compositions[other]),
                    f"Distractor compositions overlap across {split}/{other}")
    coverage = _label_coverage(dataset.cases)
    _require(coverage == manifest["label_coverage"], "Label coverage differs")
    balance = _label_balance(dataset.cases)
    _require(balance == manifest["label_balance"], "Label balance differs")
    for task in TASKS:
        for split in ("dev", "final"):
            _require(set(coverage[task][split]) <= set(coverage[task]["train"]),
                     f"Train does not cover {split} labels for {task}")
    counts = _counts(assets, dataset.worlds, dataset.pairs, dataset.rows,
                     dataset.cases, dataset.scenes)
    _require(counts == manifest["counts"], "Manifest counts differ")
    _require(counts["worlds_per_pair"] == 1 and counts["pairs_per_world"] == 1,
             "Exactly one pair belongs to exactly one world")
    roles = manifest.get("roles")
    _require(isinstance(roles, Mapping)
             and roles.get("label_visibility") == "verifier_only"
             and roles.get("model_row_fields") == sorted(MODEL_ROW_FIELDS)
             and roles.get("pair_sides") == {"a": "pair_first",
                                             "b": "pair_second"},
             "Manifest roles differ")
    for split in SPLITS:
        _require(roles["splits"][split]["worlds"] == counts["split_worlds"][split]
                 and roles["splits"][split]["pairs"] == counts["split_pairs"][split]
                 and roles["splits"][split]["assets"] == counts["split_assets"][split]
                 and roles["splits"][split]["rows"]
                 == counts["split_model_rows"][split],
                 f"Manifest role counts differ for {split}")
    _require(manifest["combinations"]["disjoint"] is True,
             "Manifest combinations are not disjoint")
    expected_generation = fingerprint({
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "seed": manifest["seed"],
        "config": manifest["config_fingerprint"],
        "assets": [asset["sha256"] for asset in assets],
        "scenes": [scene["scene_sha256"] for scene in dataset.scenes],
        "cases": [fingerprint({"case_id": case["case_id"],
                               "question": case["question"],
                               "answer": case["answer"],
                               "check": case["check"]}) for case in dataset.cases],
        "worlds": [{"world_id": world["world_id"], "split": world["split"],
                    "pair_ids": world["pair_ids"],
                    "asset_ids": world["asset_ids"]}
                   for world in dataset.worlds],
    })
    _require(expected_generation == manifest["generation_fingerprint"],
             "Generation fingerprint differs")
    return dataset


def _png_bytes(root: Path, asset: Mapping[str, Any]) -> bytes:
    return _safe_relative(root, asset["path"]).read_bytes()


def write_dataset(root: str | Path, built: Mapping[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for relative in sorted(built["files"]):
        _write_exact(root / relative, built["files"][relative])
    _write_exact(root / "manifest.json", _json_bytes(built["manifest"]))


def generate_dataset(config: Mapping[str, Any], root: str | Path
                     ) -> DependenceDataset:
    config = validate_config(config)
    built = build_dataset(config)
    write_dataset(root, built)
    return audit_dataset(root, config=config)


# ---------------------------------------------------------------------------
# Protocol construction and validation.
# ---------------------------------------------------------------------------


def _workload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    counts = protocol["data"]["counts"]
    return {
        "observations": int(counts["model_rows"]),
        "encoder_passes": int(counts["assets"]),
        "question_encodes": int(counts["verifier_cases"]),
        "optimizer_updates": 0,
        "feature_collection_splits": list(SPLITS),
        "scored_splits": ["dev"],
        "final_scores_present": False,
    }


def build_protocol(config: Mapping[str, Any], *, root: str | Path,
                   template: Mapping[str, Any],
                   implementation_files: Sequence[str] = IMPLEMENTATION_FILES,
                   data_root: str | Path | None = None,
                   ) -> dict[str, Any]:
    config = validate_config(config)
    root = Path(root)
    relative_data_root = (config["data_root"] if data_root is None
                          else str(data_root))
    _require(isinstance(relative_data_root, str) and relative_data_root
             and not Path(relative_data_root).is_absolute()
             and ".." not in Path(relative_data_root).parts,
             "Protocol data root must be a safe relative path")
    data_path = root / relative_data_root
    pins = dataset_pins(data_path)
    manifest = _load_json(data_path / "manifest.json")
    sections = ("encoder_repo", "encoder_revision", "encoder_source_digest",
                "processor_digest", "encoder_width", "image_size", "patch_size",
                "max_visual_tokens", "projector_hidden", "feature_version",
                "compute_dtype")
    _require(all(key in template["vision"] for key in sections),
             "Vision template is incomplete")
    pins_sha = {relative: _sha256_file(root / relative)
                for relative in implementation_files}
    protocol = {
        "format": PROTOCOL_FORMAT,
        "version": VERSION,
        "stage": "v2",
        "root": ".",
        "purpose": ("Zero-update paired image-dependence diagnostic over frozen "
                    "SigLIP2 patches and MiniCPM5 question states; dev and final "
                    "holdout evidence, no perception or training claim."),
        "seed": config["seed"],
        "config_sha256": fingerprint(config),
        "splits": list(SPLITS),
        "tasks": list(TASKS),
        "data": {
            **pins,
            "root": relative_data_root,
            "counts": manifest["counts"],
            "label_coverage": manifest["label_coverage"],
            "combinations_disjoint": manifest["combinations"]["disjoint"],
            "combinations_sha256": fingerprint(manifest["combinations"]["keys"]),
        },
        "vision": dict(template["vision"]),
        "vision_digest": template["vision_digest"],
        "vision_source": dict(template["vision_source"]),
        "model": dict(template["model"]),
        "readout": dict(config["readout"]),
        "limits": dict(config["limits"]),
        "resource_policy": dict(config["resource_policy"]),
        "stop_conditions": list(config["stop_conditions"]),
        "implementation_sha256": pins_sha,
        "zero_updates": True,
        "final_holdout_policy": "scored_once_after_exact_dev_report_hash",
    }
    validate_protocol(protocol, implementation_files=implementation_files)
    return protocol


def validate_protocol(protocol: Mapping[str, Any], *,
                      implementation_files: Sequence[str] = IMPLEMENTATION_FILES
                      ) -> dict[str, Any]:
    _require(isinstance(protocol, Mapping), "Protocol must be a mapping")
    _require(set(protocol) == PROTOCOL_FIELDS,
             f"Protocol fields differ: missing={sorted(PROTOCOL_FIELDS - set(protocol))} "
             f"extra={sorted(set(protocol) - PROTOCOL_FIELDS)}")
    _require(protocol["format"] == PROTOCOL_FORMAT, "Unknown protocol format")
    _require(protocol["version"] == VERSION, "Unknown protocol version")
    _require(protocol["stage"] == "v2", "Protocol requires the v2 data stage")
    _require(protocol["root"] == ".", "Protocol root must be '.'")
    _require(isinstance(protocol["purpose"], str) and protocol["purpose"].strip(),
             "Protocol purpose is required")
    _require(type(protocol["seed"]) is int and protocol["seed"] >= 0,
             "Protocol seed must be a nonnegative integer")
    _require(_is_sha256(protocol["config_sha256"]),
             "config_sha256 must be a SHA256")
    _require(list(protocol["splits"]) == list(SPLITS), "Protocol splits differ")
    _require(list(protocol["tasks"]) == list(TASKS), "Protocol tasks differ")
    data = protocol["data"]
    _require(isinstance(data, Mapping) and set(data) == DATA_FIELDS,
             "Protocol data fields differ")
    _require(isinstance(data["root"], str) and data["root"]
             and not Path(data["root"]).is_absolute()
             and ".." not in Path(data["root"]).parts,
             "Protocol data root is unsafe")
    for name in DATA_PIN_NAMES:
        pin = data[name]
        _require(isinstance(pin, Mapping) and set(pin) == {"path", "sha256"},
                 f"Protocol data pin differs: {name}")
        _require(_is_sha256(pin["sha256"]), f"Data pin is not a SHA256: {name}")
        _require(isinstance(pin["path"], str) and pin["path"]
                 and not Path(pin["path"]).is_absolute()
                 and ".." not in Path(pin["path"]).parts,
                 f"Data pin path is unsafe: {name}")
    counts = data["counts"]
    _require(isinstance(counts, Mapping), "Protocol counts are required")
    for name in ("worlds", "pairs", "assets", "model_rows", "verifier_cases",
                 "verifier_scenes"):
        _require(type(counts[name]) is int and counts[name] > 0,
                 f"Protocol count {name} must be positive")
    _require(counts["assets"] == 2 * counts["pairs"],
             "Every pair must own exactly two assets")
    _require(counts["model_rows"] == counts["verifier_cases"]
             == counts["assets"], "Model rows, cases and assets must align")
    _require(counts["pairs"] == counts["worlds"],
             "Every world must own exactly one pair")
    _require(set(counts["split_model_rows"]) == set(SPLITS)
             and set(counts["split_worlds"]) == set(SPLITS)
             and set(counts["split_assets"]) == set(SPLITS)
             and set(counts["split_pairs"]) == set(SPLITS),
             "Split counts differ")
    _require(sum(counts["split_model_rows"].values()) == counts["model_rows"],
             "Split row counts do not sum to model_rows")
    _require(sum(counts["split_worlds"].values()) == counts["worlds"],
             "Split world counts do not sum to worlds")
    _require(sum(counts["split_assets"].values()) == counts["assets"],
             "Split asset counts do not sum to assets")
    _require(sum(counts["split_pairs"].values()) == counts["pairs"],
             "Split pair counts do not sum to pairs")
    for task in TASKS:
        _require(set(counts["worlds_by_task_split"][task]) == set(SPLITS),
                 "Per-task split counts differ")
        for split in SPLITS:
            value = counts["worlds_by_task_split"][task][split]
            _require(MIN_WORLDS_PER_TASK_SPLIT <= value
                     <= MAX_WORLDS_PER_TASK_SPLIT,
                     f"World count outside [8, 12] for {task}/{split}")
    coverage = data["label_coverage"]
    _require(set(coverage) == set(TASKS), "Label coverage tasks differ")
    for task in TASKS:
        _require(set(coverage[task]) == set(SPLITS),
                 f"Label coverage splits differ for {task}")
        train = set(coverage[task]["train"])
        _require(bool(train), f"Train labels are empty for {task}")
        for split in ("dev", "final"):
            _require(set(coverage[task][split]) <= train,
                     f"Train does not cover {split} labels for {task}")
    _require(data["combinations_disjoint"] is True,
             "Combination holdout must be disjoint")
    _require(_is_sha256(data["combinations_sha256"]),
             "Combination digest must be a SHA256")
    _require(isinstance(protocol["vision"], Mapping)
             and protocol["vision"].get("compute_dtype") == "float32",
             "Vision section is invalid")
    _require(_is_sha256(protocol["vision_digest"]),
             "vision_digest must be a SHA256")
    _require(isinstance(protocol["vision_source"], Mapping)
             and "path" in protocol["vision_source"],
             "Vision source is invalid")
    model = protocol["model"]
    _require(isinstance(model, Mapping) and set(model) == {"revision", "lock_sha256"},
             "Protocol model fields differ")
    _require(isinstance(model["revision"], str) and len(model["revision"]) == 40
             and all(char in "0123456789abcdef" for char in model["revision"]),
             "Model revision must be a 40-character digest")
    _require(_is_sha256(model["lock_sha256"]), "Model lock digest is invalid")
    readout = protocol["readout"]
    _require(isinstance(readout, Mapping) and set(readout) == READOUT_FIELDS,
             "Protocol readout fields differ")
    _require(readout["primary"] == PRIMARY_READOUT,
             "Protocol primary readout differs from the preregistered one")
    _require(tuple(readout["comparators"]) == READOUT_ORDER,
             "Protocol comparators differ")
    _require(readout["bootstrap"]["cluster"] == "world",
             "Protocol bootstrap must cluster by world")
    _require(float(readout["decision"]["min_gain"]) == DECISION_MIN_GAIN,
             "Preregistered decision gain differs")
    limits = protocol["limits"]
    _require(isinstance(limits, Mapping) and set(limits) == LIMIT_FIELDS,
             "Protocol limits differ")
    for name in LIMIT_FIELDS:
        _require(type(limits[name]) is int and limits[name] >= 0,
                 f"Protocol limit {name} is invalid")
    _require(limits["max_updates"] == 0, "Zero updates are required")
    policy = protocol["resource_policy"]
    _require(isinstance(policy, Mapping) and set(policy) == RESOURCE_FIELDS,
             "Protocol resource policy differs")
    _require(policy["exclusive_mps_worker"] is True, "Exclusive MPS worker is required")
    _require(policy["cpu_fallback"] is False, "CPU fallback is forbidden")
    _require(policy["driver_limit_bytes"] >= 50 * 1024 ** 3,
             "Driver limit must be at least 50 GiB")
    _require(policy["zero_updates"] is True, "Zero-update policy is required")
    _require(policy["device"] == "mps", "Only MPS is allowed")
    _require(policy["downloads"] == 0, "Downloads are forbidden")
    _require(isinstance(protocol["stop_conditions"], list)
             and len(protocol["stop_conditions"]) >= 3,
             "Protocol stop conditions are missing")
    _require(protocol["zero_updates"] is True, "Protocol must declare zero updates")
    _require(protocol["final_holdout_policy"]
             == "scored_once_after_exact_dev_report_hash",
             "Final holdout policy differs")
    pins = protocol["implementation_sha256"]
    _require(isinstance(pins, Mapping)
             and set(pins) == set(implementation_files),
             "Implementation pins differ")
    for relative, digest in pins.items():
        _require(_is_sha256(digest), f"Implementation pin is invalid: {relative}")
    workload = _workload(protocol)
    _require(workload["observations"] <= limits["max_observations"],
             "Observation workload exceeds max_observations")
    _require(workload["observations"] <= limits["max_rows"],
             "Observation workload exceeds max_rows")
    _require(workload["encoder_passes"] <= limits["max_encoder_passes"],
             "Encoder workload exceeds max_encoder_passes")
    _require(workload["encoder_passes"] <= limits["max_assets"],
             "Encoder workload exceeds max_assets")
    _require(workload["question_encodes"] <= limits["max_question_encodes"],
             "Question workload exceeds max_question_encodes")
    _require(limits["max_bootstrap_resamples"]
             >= readout["bootstrap"]["resamples"],
             "Bootstrap resamples exceed the preregistered limit")
    return workload


def verify_implementation_pins(protocol: Mapping[str, Any],
                               root: str | Path) -> dict[str, str]:
    actual = {}
    for relative, expected in protocol["implementation_sha256"].items():
        path = _safe_relative(Path(root), relative)
        _require(path.is_file(), f"Pinned implementation file is missing: {relative}")
        actual[relative] = _sha256_file(path)
        _require(actual[relative] == expected,
                 f"Implementation bytes changed: {relative}")
    return actual


def verify_data_pins(protocol: Mapping[str, Any], *,
                     root: str | Path | None = None) -> dict[str, Any]:
    root = Path(protocol["root"] if root is None else root)
    data = protocol["data"]
    data_root = _safe_relative(root, data["root"])
    for name in DATA_PIN_NAMES:
        pin = data[name]
        path = _safe_relative(data_root, pin["path"])
        _require(path.is_file(), f"Protocol data pin is missing: {name}")
        _require(_sha256_file(path) == pin["sha256"],
                 f"Protocol data pin bytes changed: {name}")
    manifest = _load_json(data_root / data["manifest"]["path"])
    _require(manifest.get("format") == MANIFEST_FORMAT
             and manifest.get("version") == VERSION,
             "Protocol data manifest format differs")
    _require(manifest["config_fingerprint"] == protocol["config_sha256"],
             "Protocol config hash differs from the data manifest")
    _require(manifest["seed"] == protocol["seed"],
             "Protocol seed differs from the data manifest")
    _require(manifest["counts"] == data["counts"], "Protocol counts differ from data")
    _require(manifest["label_coverage"] == data["label_coverage"],
             "Protocol label coverage differs from data")
    _require(manifest["combinations"]["disjoint"] is True,
             "Data combinations are not disjoint")
    _require(fingerprint(manifest["combinations"]["keys"])
             == data["combinations_sha256"],
             "Protocol combination digest differs from data")
    for name in DATA_PIN_NAMES:
        if name == "manifest":
            continue
        file_info = manifest["files"][name]
        _require(file_info["path"] == data[name]["path"]
                 and file_info["sha256"] == data[name]["sha256"],
                 f"Manifest file pin differs: {name}")
    return manifest


# ---------------------------------------------------------------------------
# Frozen feature collection (native; no parameter updates).
# ---------------------------------------------------------------------------


def question_embedding(model, tokenizer, question: str) -> np.ndarray:
    """Frozen decoder final hidden state for the rendered question prompt."""
    import torch
    from .architecture.loading import encode_prompt
    device = model.base.get_input_embeddings().weight.device
    ids = encode_prompt(tokenizer, question, device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        output = model.base(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1]
    _require(hidden is not None and hidden.shape[1] == ids.shape[1],
             "Frozen question hidden state is unavailable")
    return hidden[0, -1].detach().float().cpu().numpy()


def regional_pooling(patches: np.ndarray, grid: Sequence[int] = DEFAULT_REGION_GRID
                     ) -> np.ndarray:
    """Pool a square patch grid into region means while preserving layout."""
    patches = np.asarray(patches, dtype=np.float32)
    _require(patches.ndim == 2, "Patch features must be [tokens, width]")
    tokens = patches.shape[0]
    side = int(round(math.sqrt(tokens)))
    _require(side * side == tokens, "Patch token count is not a square grid")
    rows, columns = int(grid[0]), int(grid[1])
    _require(rows >= 1 and columns >= 1, "Region grid must be positive")
    _require(side % rows == 0 and side % columns == 0,
             "Region grid does not divide the patch grid")
    block_rows, block_columns = side // rows, side // columns
    grid_view = patches.reshape(side, side, patches.shape[1])
    regions = []
    for row in range(rows):
        for column in range(columns):
            block = grid_view[row * block_rows:(row + 1) * block_rows,
                              column * block_columns:(column + 1) * block_columns]
            regions.append(block.reshape(-1, patches.shape[1]).mean(axis=0))
    return np.stack(regions).astype(np.float32)


def collect_frozen_features(*, service, model, tokenizer,
                            dataset: DependenceDataset,
                            region_grid: Sequence[int] = DEFAULT_REGION_GRID,
                            subject_id: str = "vision-dependence",
                            branch_id: str = "paired-diagnostic"
                            ) -> dict[str, Any]:
    """Collect pooled patches, region means and question states for every row."""
    from .memory_system.ledger import EligibilityScope
    scope = EligibilityScope(subject_id, branch_id, "diagnostic", 0,
                             "parallel_replica", None)
    pooled, regions, questions, index_rows = [], [], [], []
    for row in dataset.rows:
        visual = service.observe(
            dataset.image_path(row["asset_id"]),
            source_id="vision-dependence", task_id=row["case_id"],
            scope=scope, observed_at=0, source_epoch=0)
        _require(visual.image_sha256 == row["image_sha256"],
                 f"Frozen feature image hash differs: {row['case_id']}")
        _require(bool(visual.valid.all()), "Visual mask has invalid positions")
        patches = visual.features[0].detach().float().cpu().numpy()
        _require(patches.shape[0] == int(service.spec.max_visual_tokens),
                 "Visual token count differs from the pinned layout")
        pooled.append(patches.mean(axis=0))
        regions.append(regional_pooling(patches, region_grid))
        questions.append(question_embedding(model, tokenizer, row["question"]))
        case = dataset.case(row["case_id"])
        index_rows.append({
            "case_id": row["case_id"],
            "split": case["split"],
            "world_id": case["world_id"],
            "image_sha256": row["image_sha256"],
            "question": row["question"],
        })
    _require(len(index_rows) == len(dataset.rows), "Feature collection is incomplete")
    return {
        "rows": index_rows,
        "image_pooled": np.stack(pooled).astype(np.float32),
        "image_regions": np.stack(regions).astype(np.float32),
        "question": np.stack(questions).astype(np.float32),
    }


def write_feature_bundle(directory: str | Path, *, rows: Sequence[Mapping[str, Any]],
                         image_pooled: np.ndarray, image_regions: np.ndarray,
                         question: np.ndarray, protocol_sha256: str,
                         data: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pooled = np.asarray(image_pooled, dtype=np.float32)
    regions = np.asarray(image_regions, dtype=np.float32)
    questions = np.asarray(question, dtype=np.float32)
    count = len(rows)
    _require(pooled.ndim == 2 and pooled.shape[0] == count, "Pooled shape differs")
    _require(regions.ndim == 3 and regions.shape[0] == count, "Region shape differs")
    _require(questions.ndim == 2 and questions.shape[0] == count,
             "Question shape differs")
    for name, array in (("image_pooled", pooled), ("image_regions", regions),
                        ("question", questions)):
        _require(bool(np.isfinite(array).all()), f"{name} contains non-finite values")
    _require(_is_sha256(protocol_sha256), "Protocol hash is invalid")
    arrays = {"image_pooled": pooled, "image_regions": regions,
              "question": questions}
    features_path = directory / "features.npz"
    temporary = directory / ".features.partial.npz"
    if temporary.exists():
        temporary.unlink()
    np.savez(temporary, **arrays)
    os.replace(temporary, features_path)
    index = {
        "format": FEATURE_FORMAT,
        "version": VERSION,
        "protocol_sha256": protocol_sha256,
        "data": dict(data),
        "scores_final": False,
        "rows": [dict(row) for row in rows],
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "array_sha256": {name: _sha256_bytes(array.tobytes())
                         for name, array in arrays.items()},
        "features_sha256": _sha256_file(features_path),
    }
    index_path = directory / "features_index.json"
    _write_exact(index_path, _json_bytes(index))
    return {
        "features": {"path": str(features_path), "sha256": index["features_sha256"]},
        "index": {"path": str(index_path),
                  "sha256": _sha256_file(index_path)},
        "count": count,
        "array_shapes": index["array_shapes"],
    }


def load_feature_bundle(directory: str | Path, *,
                        expected_index_sha256: str | None = None
                        ) -> dict[str, Any]:
    directory = Path(directory)
    index_path = directory / "features_index.json"
    features_path = directory / "features.npz"
    _require(index_path.is_file() and features_path.is_file(),
             "Feature bundle files are missing")
    if expected_index_sha256 is not None:
        _require(_is_sha256(expected_index_sha256), "Expected index hash is invalid")
        _require(_sha256_file(index_path) == expected_index_sha256,
                 "Feature index bytes changed")
    index = _load_json(index_path)
    _require(index.get("format") == FEATURE_FORMAT, "Unknown feature bundle format")
    _require(index.get("version") == VERSION, "Unknown feature bundle version")
    _require(index.get("scores_final") is False,
             "The dev-stage feature bundle must not contain final scores")
    _require(_sha256_file(features_path) == index["features_sha256"],
             "Feature array bytes changed")
    with np.load(features_path, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name]) for name in
                  ("image_pooled", "image_regions", "question")}
    for name, array in arrays.items():
        _require(str(array.dtype) == index["array_dtypes"][name],
                 f"Feature dtype differs: {name}")
        _require(list(array.shape) == index["array_shapes"][name],
                 f"Feature shape differs: {name}")
        _require(_sha256_bytes(array.tobytes()) == index["array_sha256"][name],
                 f"Feature values were tampered with: {name}")
        _require(bool(np.isfinite(array).all()),
                 f"Feature values are not finite: {name}")
    rows = index["rows"]
    _require(len(rows) == arrays["image_pooled"].shape[0]
             == arrays["image_regions"].shape[0] == arrays["question"].shape[0],
             "Feature row count differs")
    return {"index": index, "arrays": arrays, "rows": rows,
            "index_path": index_path, "features_path": features_path}


# ---------------------------------------------------------------------------
# Closed-form readouts.
# ---------------------------------------------------------------------------


def fit_ridge(features: np.ndarray, targets: np.ndarray, *,
              penalty: float = RIDGE_PENALTY) -> dict[str, Any]:
    """Standardised ridge with an unpenalised intercept; dual when d > n."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    _require(x.ndim == 2 and y.ndim == 2, "Ridge expects 2-D features/targets")
    _require(x.shape[0] == y.shape[0] and x.shape[0] >= 1,
             "Ridge sample counts differ")
    _require(penalty > 0 and math.isfinite(float(penalty)),
             "Ridge penalty must be finite and positive")
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    design = (x - mean) / scale
    target_mean = y.mean(axis=0, keepdims=True)
    centred = y - target_mean
    samples, dimension = design.shape
    if dimension <= samples:
        gram = design.T @ design + float(penalty) * np.eye(dimension)
        weights = np.linalg.solve(gram, design.T @ centred)
        dual = False
    else:
        gram = design @ design.T + float(penalty) * np.eye(samples)
        alpha = np.linalg.solve(gram, centred)
        weights = design.T @ alpha
        dual = True
    _require(bool(np.isfinite(weights).all()), "Ridge weights are not finite")
    return {"mean": mean, "scale": scale, "target_mean": target_mean,
            "weights": weights, "dual": dual, "dimension": dimension,
            "samples": samples, "penalty": float(penalty)}


def ridge_scores(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1] == model["dimension"],
             "Ridge prediction dimension differs")
    design = (x - model["mean"]) / model["scale"]
    return design @ model["weights"] + model["target_mean"]


def _one_hot(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(classes)}
    result = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        result[row, index[label]] = 1.0
    return result


def _predict_with(model: Mapping[str, Any], classes: Sequence[str],
                  features: np.ndarray) -> list[str]:
    scores = ridge_scores(model, features)
    return [str(classes[int(position)]) for position in scores.argmax(axis=1)]


def infer_task(question: str) -> str | None:
    """Route from question text only; never from an oracle task or label."""
    text = str(question).strip()
    if text.startswith("How many "):
        return "count"
    if text.startswith("Is there a "):
        return "presence"
    if text.startswith("What color is "):
        return "color"
    if text.startswith("What shape is "):
        return "shape"
    if text.startswith("Is the ") and (" to the left of " in text
                                       or " to the right of " in text):
        return "spatial"
    return None


def _records_array(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.stack([np.asarray(record[key], dtype=np.float32) for record in records])


def _feature_matrix(records: Sequence[Mapping[str, Any]], name: str, *,
                    pooled: np.ndarray | None = None,
                    regions: np.ndarray | None = None) -> np.ndarray:
    questions = _records_array(records, "question_embedding").astype(np.float64)
    pooled_values = (np.asarray(pooled, dtype=np.float64) if pooled is not None
                     else _records_array(records, "pooled").astype(np.float64))
    region_values = (np.asarray(regions, dtype=np.float64) if regions is not None
                     else _records_array(records, "regions").astype(np.float64))
    if name == "question_only":
        return questions
    if name == "pooled_image":
        return pooled_values
    if name == "pooled_image_question":
        return np.concatenate([pooled_values, questions], axis=1)
    if name == "region_image":
        return region_values.reshape(len(records), -1)
    if name == "region_image_question":
        return np.concatenate([region_values.reshape(len(records), -1), questions],
                              axis=1)
    raise VisionDependenceError(f"Unknown readout feature set: {name}")


def fit_readouts(records: Sequence[Mapping[str, Any]], *,
                 penalty: float = RIDGE_PENALTY) -> dict[str, Any]:
    train = [record for record in records if record["split"] == "train"]
    _require(len(train) >= 4, "Too few train records for a readout")
    classes = sorted({str(record["label"]) for record in train})
    _require(len(classes) >= 2, "Readouts need at least two train classes")
    labels = [str(record["label"]) for record in train]
    targets = _one_hot(labels, classes)
    models: dict[str, Any] = {}
    for name in READOUT_ORDER:
        features = _feature_matrix(train, name)
        fitted = fit_ridge(features, targets, penalty=penalty)
        models[name] = {
            "kind": "global", "classes": classes, "model": fitted,
            "train_accuracy": _accuracy(
                _predict_with(fitted, classes, features), labels),
        }
    task_models: dict[str, Any] = {}
    for task in TASKS:
        rows = [record for record in train if record["task"] == task]
        if len(rows) < 2:
            continue
        task_classes = sorted({str(record["label"]) for record in rows})
        _require(len(task_classes) >= 2,
                 f"Task {task} needs at least two label classes")
        features = _feature_matrix(rows, "region_image")
        task_models[task] = {
            "classes": task_classes,
            "model": fit_ridge(features, _one_hot(
                [str(record["label"]) for record in rows], task_classes),
                penalty=penalty),
            "train_rows": len(rows),
        }
        task_models[task]["train_accuracy"] = _accuracy(
            _predict_with(task_models[task]["model"], task_classes, features),
            [str(record["label"]) for record in rows])
    _require(set(task_models) == set(TASKS),
             "Primary readout must be trainable for every task")
    primary = {
        "kind": "question_conditioned_region",
        "tasks": task_models,
        "trained_tasks": sorted(task_models),
        "fallback": models["region_image"],
        "routing": "question_text_only",
        "train_accuracy": float(np.mean([model["train_accuracy"]
                                         for model in task_models.values()])),
    }
    return {"readouts": models, "primary": primary,
            "train_classes": classes, "train_rows": len(train)}


def predict_primary(primary: Mapping[str, Any], questions: Sequence[str],
                    regions: np.ndarray) -> list[str]:
    """Image-conditioned prediction routed by question text alone."""
    regions = np.asarray(regions, dtype=np.float64)
    _require(regions.ndim == 3 and regions.shape[0] == len(questions),
             "Primary region features differ")
    flat = regions.reshape(len(questions), -1)
    predictions: list[str] = []
    for position, question in enumerate(questions):
        task = infer_task(question)
        model = primary["tasks"].get(task) if task is not None else None
        if model is None:
            model = primary["fallback"]
        predictions.append(_predict_with(model["model"], model["classes"],
                                        flat[position:position + 1])[0])
    return predictions


def predict_readouts(models: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                     *, pooled: np.ndarray | None = None,
                     regions: np.ndarray | None = None) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for name in READOUT_ORDER:
        entry = models["readouts"][name]
        features = _feature_matrix(records, name, pooled=pooled, regions=regions)
        predictions[name] = _predict_with(entry["model"], entry["classes"], features)
    questions = [str(record["question"]) for record in records]
    region_values = (np.asarray(regions) if regions is not None
                     else _records_array(records, "regions"))
    predictions[PRIMARY_READOUT] = predict_primary(models["primary"], questions,
                                                   region_values)
    return predictions


def _accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    _require(len(predicted) == len(expected) and bool(expected),
             "Accuracy needs paired nonempty sequences")
    return sum(int(left == right) for left, right in zip(predicted, expected)) \
        / len(expected)


def derangement_indices(count: int, seed: int) -> np.ndarray:
    """Deterministic permutation with no fixed point (symmetric control)."""
    _require(type(count) is int and count >= 2,
             "A derangement needs at least two items")
    rng = np.random.default_rng(int(seed))
    base = np.arange(count)
    for _ in range(2048):
        candidate = rng.permutation(count)
        if not np.any(candidate == base):
            return candidate
    return (base + 1) % count


def _stable_seed(seed: int, *parts: str) -> int:
    material = "|".join([str(int(seed)), *parts]).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def _derange_rows(rows: Sequence[int], seed: int, *parts: str
                  ) -> list[int]:
    _require(len(rows) >= 2, "A derangement group needs at least two rows")
    rng = np.random.default_rng(_stable_seed(seed, *parts))
    base = np.arange(len(rows))
    for _ in range(2048):
        candidate = rng.permutation(len(rows))
        if not np.any(candidate == base):
            return [rows[int(position)] for position in candidate]
    return [rows[(position + 1) % len(rows)] for position in base]


def question_derangement(records: Sequence[Mapping[str, Any]], seed: int
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Shuffle images only within the exact question, else within its task.

    Every receiver keeps a donor from the same question whenever that question
    has at least two rows in the split; otherwise the donor comes from the same
    question-derived task. No donor crosses a task boundary and there are no
    fixed points.
    """
    count = len(records)
    _require(count >= 2, "A shuffled control needs at least two rows")
    order = np.full(count, -1, dtype=int)
    by_task: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        task = infer_task(record["question"])
        _require(task is not None,
                 f"Question does not route to a task: {record['question']!r}")
        _require(record["task"] == task,
                 "Question-derived routing differs from the verifier task")
        by_task[task][record["question"]].append(index)
    for task in sorted(by_task):
        groups = by_task[task]
        task_rows = [index for question in sorted(groups)
                     for index in groups[question]]
        _require(len(task_rows) >= 2,
                 f"Cannot derange within task {task} without crossing it")
        singletons = [groups[question][0] for question in sorted(groups)
                      if len(groups[question]) == 1]
        handled: set[str] = set()
        if len(singletons) == 1:
            # Merge the singleton into one full question group before any
            # assignment is made. Recomputing that group's derangement as a
            # whole keeps a bijection: nothing already assigned is overwritten
            # and no donor is reused.
            target_question = min(
                (question for question in groups
                 if len(groups[question]) >= 2),
                key=lambda question: (-len(groups[question]),
                                      groups[question][0]))
            combined = list(groups[target_question]) + singletons
            donors = _derange_rows(combined, seed, "merge", task,
                                   target_question)
            for receiver, donor in zip(combined, donors):
                order[receiver] = donor
            handled.add(target_question)
        elif len(singletons) >= 2:
            donors = _derange_rows(singletons, seed, "task", task)
            for receiver, donor in zip(singletons, donors):
                order[receiver] = donor
        for question in sorted(groups):
            rows = groups[question]
            if len(rows) < 2 or question in handled:
                continue
            donors = _derange_rows(rows, seed, "question", task, question)
            for receiver, donor in zip(rows, donors):
                order[receiver] = donor
    _require(bool(np.all(order >= 0)), "Shuffle assignment is incomplete")
    _require(sorted(int(value) for value in order) == list(range(count)),
             "Shuffle assignment is not a bijection")
    _require(not np.any(order == np.arange(count)), "Shuffle has a fixed point")
    same_question = 0
    for receiver, donor in enumerate(order):
        donor_task = infer_task(records[int(donor)]["question"])
        _require(donor_task == records[receiver]["task"],
                 "Shuffle donor crosses a task boundary")
        if records[int(donor)]["question"] == records[receiver]["question"]:
            same_question += 1
    return order, {
        "policy": "question_group_then_task_derangement",
        "fixed_points": 0,
        "cross_task_donors": 0,
        "same_question_rows": int(same_question),
        "same_task_only_rows": int(count - same_question),
        "tasks": sorted(by_task),
    }


# ---------------------------------------------------------------------------
# World-clustered bootstrap and decision.
# ---------------------------------------------------------------------------


def world_clusters(records: Sequence[Mapping[str, Any]],
                   *, task: str | None = None) -> dict[str, list[int]]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if task is not None and record["task"] != task:
            continue
        clusters[str(record["world_id"])].append(index)
    _require(bool(clusters), "No records to cluster")
    return {world: clusters[world] for world in sorted(clusters)}


def _bootstrap_values(values: np.ndarray, clusters: Mapping[str, Sequence[int]],
                      *, resamples: int, seed: int) -> np.ndarray:
    worlds = list(clusters)
    _require(len(worlds) >= 2, "Clustered bootstrap needs at least two worlds")
    sums = np.array([float(values[list(clusters[world])].sum()) for world in worlds])
    counts = np.array([len(clusters[world]) for world in worlds], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(worlds), size=(int(resamples), len(worlds)))
    sampled_sums = sums[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    _require(bool(np.all(sampled_counts > 0)), "Bootstrap drew an empty sample")
    return sampled_sums / sampled_counts


def _interval(values: np.ndarray, point: float, confidence: float
              ) -> dict[str, float]:
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return {"point": float(point), "lower": float(lower), "upper": float(upper)}


def clustered_bootstrap(records: Sequence[Mapping[str, Any]],
                        metric_arrays: Mapping[str, np.ndarray], *,
                        resamples: int, confidence: float, seed: int,
                        task: str | None = None) -> dict[str, Any]:
    clusters = world_clusters(records, task=task)
    result = {}
    for name, values in metric_arrays.items():
        values = np.asarray(values, dtype=np.float64)
        _require(values.shape == (len(records),),
                 "Metric array does not align with records")
        if task is not None:
            keep = sorted(index for indices in clusters.values()
                          for index in indices)
            point = float(values[keep].mean())
        else:
            point = float(values.mean())
        sampled = _bootstrap_values(values, clusters, resamples=resamples,
                                    seed=seed)
        result[name] = _interval(sampled, point, confidence)
    return {"cluster": "world", "clusters": len(clusters),
            "resamples": int(resamples), "confidence": float(confidence),
            "seed": int(seed), "metrics": result}


def decide(overall: Mapping[str, Mapping[str, float]], decision: Mapping[str, Any]
           ) -> dict[str, Any]:
    """Map paired evidence to a conservative preregistered verdict."""
    gain_question = overall["gain_over_question_only"]
    gain_shuffled = overall["gain_over_shuffled_image"]
    threshold = float(decision["min_gain"])
    strict = bool(decision["strict_lower_bound"])
    lower_positive = (lambda interval: interval["lower"] > 0.0) if strict \
        else (lambda interval: interval["lower"] >= 0.0)
    supported = (gain_question["point"] > threshold
                 and gain_shuffled["point"] > threshold
                 and lower_positive(gain_question)
                 and lower_positive(gain_shuffled))
    if supported:
        verdict = "image_dependence_supported_by_primary_readout"
        justification = ("The primary spatially preserved, question-routed "
                         "readout exceeds both controls by more than the "
                         "preregistered margin with strictly positive "
                         "world-clustered lower bounds.")
    elif gain_question["point"] <= 0.0 or gain_shuffled["point"] <= 0.0:
        verdict = "failing_primary_has_no_incremental_image_signal"
        justification = ("The primary readout does not beat both controls on "
                         "mean paired accuracy; image dependence is not shown.")
    else:
        verdict = "inconclusive_primary_gain_not_established"
        justification = ("Mean paired gains are inside the preregistered margin "
                         "or their world-clustered intervals include zero; the "
                         "holdout is inconclusive.")
    return {
        "verdict": verdict,
        "supported": bool(supported),
        "justification": justification,
        "thresholds": {
            "min_gain": threshold,
            "strict_lower_bound": strict,
            "min_lower_bound": float(decision["min_lower_bound"]),
            "requires_gain_over_question_only": True,
            "requires_gain_over_shuffled_image": True,
        },
        "gain_over_question_only": dict(gain_question),
        "gain_over_shuffled_image": dict(gain_shuffled),
        "default_positive_verdict": False,
    }


# ---------------------------------------------------------------------------
# Feature-bundle records and split evaluation.
# ---------------------------------------------------------------------------


def records_from_bundle(dataset: DependenceDataset,
                        bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = bundle["rows"]
    arrays = bundle["arrays"]
    expected = [(row["case_id"], dataset.case(row["case_id"])["split"],
                 row["image_sha256"]) for row in dataset.rows]
    observed = [(row["case_id"], row["split"], row["image_sha256"])
                for row in rows]
    _require(observed == expected,
             "Feature bundle rows differ from the verified dataset")
    records = []
    for index, row in enumerate(rows):
        case = dataset.case(row["case_id"])
        _require(case["question"] == row["question"],
                 "Feature bundle question differs from verifier truth")
        _require(case["world_id"] == row["world_id"],
                 "Feature bundle world differs from verifier truth")
        records.append({
            "case_id": row["case_id"],
            "world_id": case["world_id"],
            "split": case["split"],
            "task": case["task"],
            "label": case["label"],
            "question": case["question"],
            "pooled": arrays["image_pooled"][index].astype(np.float64),
            "regions": arrays["image_regions"][index].astype(np.float64),
            "question_embedding": arrays["question"][index].astype(np.float64),
        })
    return records


def _baselines(records: Sequence[Mapping[str, Any]],
               train: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_task = {}
    chance = {}
    for task in TASKS:
        train_labels = [record["label"] for record in train
                        if record["task"] == task]
        eval_labels = [record["label"] for record in records
                       if record["task"] == task]
        if not train_labels or not eval_labels:
            continue
        majority = Counter(train_labels).most_common(1)[0][0]
        per_task[task] = {
            "majority_label": majority,
            "majority_accuracy": _accuracy([majority] * len(eval_labels),
                                           eval_labels),
            "uniform_chance": 1.0 / len(set(train_labels)),
            "train_label_count": len(set(train_labels)),
        }
        chance[task] = per_task[task]["uniform_chance"]
    global_labels = [record["label"] for record in train]
    majority = Counter(global_labels).most_common(1)[0][0]
    eval_labels = [record["label"] for record in records]
    weighted_chance = float(np.mean([chance[record["task"]] for record in records])
                            ) if records else 0.0
    return {
        "per_task": per_task,
        "global_majority_label": majority,
        "global_majority_accuracy": _accuracy([majority] * len(eval_labels),
                                              eval_labels),
        "weighted_uniform_chance": weighted_chance,
    }


def evaluate_records(records: Sequence[Mapping[str, Any]], *,
                     readout: Mapping[str, Any], split: str, seed: int
                     ) -> dict[str, Any]:
    """Fit on train only and score one split, including shuffled controls."""
    _require(split in ("dev", "final"), "Only dev or final may be scored")
    train = [record for record in records if record["split"] == "train"]
    evaluation = [record for record in records if record["split"] == split]
    _require(len(train) >= 4 and len(evaluation) >= 4,
             "Too few records for split evaluation")
    _require({record["world_id"] for record in train}.isdisjoint(
        {record["world_id"] for record in evaluation}),
        "Train and evaluation worlds overlap")
    penalty = float(readout["ridge_penalty"])
    fitted = fit_readouts(records, penalty=penalty)
    base = predict_readouts(fitted, evaluation)
    count = len(evaluation)
    shuffle_seed = int(seed) + int(readout["shuffle"]["seed_offset"]) \
        + (0 if split == "dev" else 1)
    order, shuffle_stats = question_derangement(evaluation, shuffle_seed)
    pooled_shuffled = _records_array(evaluation, "pooled")[order]
    regions_shuffled = _records_array(evaluation, "regions")[order]
    shuffled = predict_readouts(fitted, evaluation, pooled=pooled_shuffled,
                                regions=regions_shuffled)
    labels = np.array([str(record["label"]) for record in evaluation])
    tasks = np.array([str(record["task"]) for record in evaluation])
    worlds = np.array([str(record["world_id"]) for record in evaluation])
    correctness = {name: (np.array(predictions) == labels).astype(np.float64)
                   for name, predictions in base.items()}
    shuffled_correctness = {
        name: (np.array(predictions) == labels).astype(np.float64)
        for name, predictions in shuffled.items() if name != "question_only"}
    primary = correctness[PRIMARY_READOUT]
    metrics = {f"accuracy::{name}": values
               for name, values in correctness.items()}
    metrics.update({f"accuracy_shuffled::{name}": values
                    for name, values in shuffled_correctness.items()})
    metrics["gain_over_question_only"] = primary - correctness["question_only"]
    metrics["gain_over_shuffled_image"] = (primary
                                           - shuffled_correctness[PRIMARY_READOUT])
    bootstrap = {
        "overall": clustered_bootstrap(
            evaluation, metrics, resamples=int(readout["bootstrap"]["resamples"]),
            confidence=float(readout["bootstrap"]["confidence"]),
            seed=int(seed) + (0 if split == "dev" else 101)),
        "per_task": {},
    }
    for task in TASKS:
        if task not in set(tasks.tolist()):
            continue
        bootstrap["per_task"][task] = clustered_bootstrap(
            evaluation, metrics, resamples=int(readout["bootstrap"]["resamples"]),
            confidence=float(readout["bootstrap"]["confidence"]),
            seed=int(seed) + 1 + TASKS.index(task) + (0 if split == "dev" else 101),
            task=task)
    overall_metrics = bootstrap["overall"]["metrics"]
    verdict = decide(overall_metrics, readout["decision"])
    readouts_report = {}
    for name in READOUT_ORDER:
        readouts_report[name] = {
            "accuracy": float(correctness[name].mean()),
            "per_task": {task: float(correctness[name][tasks == task].mean())
                         for task in TASKS if np.any(tasks == task)},
            "interval": overall_metrics[f"accuracy::{name}"],
        }
        if name != "question_only":
            readouts_report[name]["shuffled_image"] = {
                "accuracy": float(shuffled_correctness[name].mean()),
                "per_task": {task: float(
                    shuffled_correctness[name][tasks == task].mean())
                    for task in TASKS if np.any(tasks == task)},
                "interval": overall_metrics[f"accuracy_shuffled::{name}"],
            }
    readouts_report[PRIMARY_READOUT] = {
        "accuracy": float(correctness[PRIMARY_READOUT].mean()),
        "per_task": {task: float(correctness[PRIMARY_READOUT][tasks == task].mean())
                     for task in TASKS if np.any(tasks == task)},
        "interval": overall_metrics[f"accuracy::{PRIMARY_READOUT}"],
        "shuffled_image": {
            "accuracy": float(shuffled_correctness[PRIMARY_READOUT].mean()),
            "per_task": {task: float(
                shuffled_correctness[PRIMARY_READOUT][tasks == task].mean())
                for task in TASKS if np.any(tasks == task)},
            "interval": overall_metrics[
                f"accuracy_shuffled::{PRIMARY_READOUT}"],
        },
        "routing": "question_text_only",
        "declared_before_evaluation": True,
    }
    prediction_rows = []
    for index, record in enumerate(evaluation):
        prediction_rows.append({
            "case_id": record["case_id"],
            "world_id": record["world_id"],
            "task": record["task"],
            "expected": record["label"],
            "readouts": {name: base[name][index] for name in READOUT_ORDER
                         + (PRIMARY_READOUT,)},
            "shuffled_image": {name: shuffled[name][index]
                              for name in shuffled if name != "question_only"},
        })
    return {
        "split": split,
        "seed": int(seed),
        "train_cases": len(train),
        "evaluation_cases": len(evaluation),
        "train_worlds": sorted({record["world_id"] for record in train}),
        "evaluation_worlds": sorted({record["world_id"] for record in evaluation}),
        "worlds": int(len(set(worlds.tolist()))),
        "primary_readout": PRIMARY_READOUT,
        "primary_declared_before_evaluation": True,
        "readouts": readouts_report,
        "gains": {
            "over_question_only": overall_metrics["gain_over_question_only"],
            "over_shuffled_image": overall_metrics["gain_over_shuffled_image"],
        },
        "per_task_gains": {
            task: {
                "over_question_only":
                    bootstrap["per_task"][task]["metrics"]["gain_over_question_only"],
                "over_shuffled_image":
                    bootstrap["per_task"][task]["metrics"]["gain_over_shuffled_image"],
            }
            for task in bootstrap["per_task"]
        },
        "bootstrap": bootstrap,
        "baselines": _baselines(evaluation, train),
        "decision": verdict,
        "predictions": prediction_rows,
        "shuffle": {
            "policy": readout["shuffle"]["policy"],
            "seed": shuffle_seed,
            **shuffle_stats,
            "permutation_sha256": fingerprint([int(value) for value in order]),
        },
    }


# ---------------------------------------------------------------------------
# Report assembly and single-shot finalization.
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def assemble_dev_report(*, protocol: Mapping[str, Any], protocol_path: str | Path,
                        feature_pins: Mapping[str, Any],
                        predictions_sha256: str, evaluation: Mapping[str, Any],
                        model: Mapping[str, Any], frozen_audit: Mapping[str, Any],
                        created_at: str | None = None) -> dict[str, Any]:
    validate_protocol(protocol)
    _require(evaluation["split"] == "dev", "A dev report needs dev evaluation")
    _require(_is_sha256(predictions_sha256), "Prediction hash is invalid")
    counts = protocol["data"]["counts"]
    return {
        "format": DEV_REPORT_FORMAT,
        "version": VERSION,
        "stage": "dev",
        "status": "completed",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "config_sha256": protocol["config_sha256"],
        "data": protocol["data"],
        "features": dict(feature_pins),
        "seed": protocol["seed"],
        "primary_readout": PRIMARY_READOUT,
        "train_cases": int(counts["split_model_rows"]["train"]),
        "dev_cases": int(counts["split_model_rows"]["dev"]),
        "evaluation": dict(evaluation),
        "predictions_sha256": predictions_sha256,
        "scores_final": False,
        "final_scored": False,
        "evaluated_splits": ["dev"],
        "zero_updates": True,
        "optimizer_updates": 0,
        "parameters_updated": [],
        "model": dict(model),
        "frozen_feature_audit": dict(frozen_audit),
        "decision_scope": "dev_only_not_final",
        "created_at_utc": created_at or _utc_now(),
    }


def _validate_dev_report(report: Mapping[str, Any], *, protocol: Mapping[str, Any],
                         protocol_path: str | Path) -> None:
    _require(report.get("format") == DEV_REPORT_FORMAT,
             "Unknown dev report format")
    _require(report.get("version") == VERSION, "Unknown dev report version")
    _require(report.get("stage") == "dev", "Not a dev report")
    _require(report.get("status") == "completed", "Dev report is not complete")
    _require(report.get("scores_final") is False,
             "The dev report must not contain final scores")
    _require(report.get("final_scored") is False,
             "The dev report must not have scored the final holdout")
    _require(report.get("zero_updates") is True, "Dev report must declare zero updates")
    _require(list(report.get("evaluated_splits", [])) == ["dev"],
             "Dev report scored more than the dev split")
    _require(report.get("protocol_sha256") == _sha256_file(protocol_path),
             "Dev report was produced under another protocol")
    _require(report.get("config_sha256") == protocol["config_sha256"],
             "Dev report config differs from the protocol")
    _require(report.get("data") == protocol["data"],
             "Dev report data pins differ from the protocol")
    _require(report.get("primary_readout") == PRIMARY_READOUT,
             "Dev report primary readout differs")
    _require(report.get("seed") == protocol["seed"],
             "Dev report seed differs")
    features = report.get("features")
    _require(isinstance(features, Mapping)
             and isinstance(features.get("predictions"), Mapping),
             "Dev report prediction pins are missing")
    _require(report.get("predictions_sha256")
             == features["predictions"].get("sha256"),
             "Dev report prediction hashes disagree")
    _require(isinstance(report.get("evaluation"), Mapping)
             and report["evaluation"].get("split") == "dev",
             "Dev report evaluation is invalid")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def require_fresh_output(path: str | Path) -> Path:
    path = Path(path)
    _require(not path.exists() or not any(path.iterdir()),
             "Use a fresh output path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_vision_binding(protocol: Mapping[str, Any], *,
                          adapter_spec: Any, service_spec: Any) -> None:
    """Require both loaded vision spec objects to match the protocol exactly."""
    _require(adapter_spec is not None and service_spec is not None,
             "Loaded vision specifications are missing")
    for name in ("encoder_repo", "encoder_revision", "encoder_source_digest",
                 "processor_digest", "encoder_width", "image_size",
                 "patch_size", "max_visual_tokens", "projector_hidden",
                 "feature_version", "compute_dtype"):
        _require(getattr(adapter_spec, name, None) == protocol["vision"][name],
                 f"Adapter vision field differs from the protocol: {name}")
        _require(getattr(service_spec, name, None) == protocol["vision"][name],
                 f"Service vision field differs from the protocol: {name}")
    _require(adapter_spec.digest == protocol["vision_digest"],
             "Adapter vision digest differs from the protocol")
    _require(service_spec.digest == protocol["vision_digest"],
             "Service vision digest differs from the protocol")


def assemble_native_run_report(*, protocol: Mapping[str, Any],
                               protocol_path: str | Path,
                               dev_report: Mapping[str, Any],
                               output: str | Path,
                               feature_pins: Mapping[str, Any],
                               dev_evaluation: Mapping[str, Any],
                               model: Mapping[str, Any],
                               frozen_audit: Mapping[str, Any],
                               workload_declared: Mapping[str, Any],
                               elapsed_seconds: float,
                               feature_bytes: int,
                               started_at_utc: str,
                               packages: Mapping[str, Any] | None = None,
                               extra: Mapping[str, Any] | None = None,
                               ) -> dict[str, Any]:
    dev_bytes = _json_bytes(dev_report)
    dev_path = Path(output) / "dev_report.json"
    counts = protocol["data"]["counts"]
    return {
        "format": RUN_REPORT_FORMAT,
        "version": VERSION,
        "stage": "native_dev",
        "status": "completed",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "data": {"manifest_sha256": protocol["data"]["manifest"]["sha256"],
                 "pins": protocol["data"], "counts": dict(counts)},
        "seed": protocol["seed"],
        "model": dict(model),
        "workload_declared": dict(workload_declared),
        "workload_actual": {
            "observations": int(counts["model_rows"]),
            "encoder_passes": int(counts["assets"]),
            "question_encodes": int(counts["verifier_cases"]),
            "optimizer_updates": 0,
        },
        "device": "mps",
        "fallback_disabled": True,
        "downloads": 0,
        "zero_updates": True,
        "optimizer_updates": 0,
        "parameters_updated": [],
        "feature_collection_splits": ["train", "dev", "final"],
        "scored_splits": ["dev"],
        "final_scored": False,
        "scores_final": False,
        "feature_pins": dict(feature_pins),
        "feature_bytes": int(feature_bytes),
        "elapsed_seconds": float(elapsed_seconds),
        "dev_report": {"path": str(dev_path),
                       "sha256": _sha256_bytes(dev_bytes)},
        "dev_verdict": dev_evaluation["decision"],
        "frozen_feature_audit": dict(frozen_audit),
        "packages": dict(packages or {}),
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        **dict(extra or {}),
    }


def publish_native_artifacts(output: str | Path, *, dev_report: Mapping[str, Any],
                             run_report: Mapping[str, Any],
                             max_seconds: int, max_feature_bytes: int
                             ) -> dict[str, Any]:
    """Publish a completed native run and only then its dev report."""
    output = Path(output)
    _require(run_report.get("format") == RUN_REPORT_FORMAT,
             "Native run report format differs")
    _require(run_report.get("status") == "completed",
             "Refusing to publish an incomplete native run")
    _require(run_report.get("zero_updates") is True
             and run_report.get("optimizer_updates") == 0
             and run_report.get("parameters_updated") == [],
             "Native run report does not declare zero updates")
    _require(run_report.get("device") == "mps",
             "Native run report device differs")
    _require(run_report.get("fallback_disabled") is True,
             "Native run report fallback policy differs")
    _require(run_report.get("downloads") == 0,
             "Native run report declares downloads")
    _require(float(run_report["elapsed_seconds"]) <= float(max_seconds),
             "Native run exceeded max_seconds")
    _require(int(run_report["feature_bytes"]) <= int(max_feature_bytes),
             "Native run exceeded max_feature_bytes")
    dev_bytes = _json_bytes(dev_report)
    dev_path = output / "dev_report.json"
    _require(run_report["dev_report"]["path"] == str(dev_path),
             "Native run report pins another dev report path")
    _require(run_report["dev_report"]["sha256"] == _sha256_bytes(dev_bytes),
             "Native run report dev hash differs from the dev report")
    _atomic_write(output / "run_report.json", _json_bytes(run_report))
    _atomic_write(dev_path, dev_bytes)
    return dict(run_report)


def validate_native_run_report(*, dev_report_path: str | Path,
                               dev_report_sha256: str,
                               protocol: Mapping[str, Any],
                               protocol_path: str | Path,
                               feature_pins: Mapping[str, Any]) -> dict[str, Any]:
    """Require a completed adjacent native run report before finalization."""
    dev_report_path = Path(dev_report_path).resolve()
    run_path = dev_report_path.parent / "run_report.json"
    _require(run_path.is_file(),
             "A completed native run_report.json is required beside the dev report")
    run = _load_json(run_path)
    _require(run.get("format") == RUN_REPORT_FORMAT,
             "Native run report format differs")
    _require(run.get("version") == VERSION, "Native run report version differs")
    _require(run.get("stage") == "native_dev", "Not a native dev run report")
    _require(run.get("status") == "completed",
             "The native run did not complete")
    _require(run.get("protocol_sha256") == _sha256_file(protocol_path),
             "Native run report belongs to another protocol")
    _require(run.get("seed") == protocol["seed"], "Native run seed differs")
    _require(run.get("zero_updates") is True
             and run.get("optimizer_updates") == 0
             and run.get("parameters_updated") == [],
             "Native run report does not prove zero updates")
    _require(run.get("device") == "mps",
             "Native run report device differs")
    _require(run.get("fallback_disabled") is True,
             "Native run report fallback policy differs")
    _require(run.get("downloads") == 0,
             "Native run report declares downloads")
    expected_data = {
        "manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "pins": protocol["data"],
        "counts": protocol["data"]["counts"],
    }
    _require(canonical_json(run.get("data")) == canonical_json(expected_data),
             "Native run report data pins differ")
    _require(run.get("feature_collection_splits") == list(SPLITS),
             "Native run report feature collection splits differ")
    _require(run.get("scored_splits") == ["dev"],
             "Native run report scored splits differ")
    _require(run.get("final_scored") is False
             and run.get("scores_final") is False,
             "Native run report must not score the final split")
    _require(run.get("dev_report", {}).get("sha256") == dev_report_sha256
             and Path(run["dev_report"]["path"]).resolve() == dev_report_path,
             "Native run report does not pin the exact dev report")
    _require(canonical_json(run.get("feature_pins")) == canonical_json(feature_pins),
             "Native run report feature pins differ")
    elapsed = run.get("elapsed_seconds")
    _require(not isinstance(elapsed, bool) and isinstance(elapsed, (int, float))
             and math.isfinite(float(elapsed))
             and 0.0 <= float(elapsed) <= protocol["limits"]["max_seconds"],
             "Native run exceeded max_seconds")
    feature_bytes = run.get("feature_bytes")
    _require(type(feature_bytes) is int and feature_bytes >= 0
             and feature_bytes <= protocol["limits"]["max_feature_bytes"],
             "Native run exceeded max_feature_bytes")
    expected_workload = {
        "observations": int(protocol["data"]["counts"]["model_rows"]),
        "encoder_passes": int(protocol["data"]["counts"]["assets"]),
        "question_encodes": int(protocol["data"]["counts"]["verifier_cases"]),
        "optimizer_updates": 0,
    }
    _require(canonical_json(run.get("workload_actual")) == canonical_json(
        expected_workload), "Native run report workload differs")
    _require(run.get("elapsed_seconds") is not None,
             "Native run report elapsed_seconds is missing")
    return run


def _receipt_key(protocol_sha256: str, data_manifest_sha256: str) -> str:
    return _sha256_bytes(
        f"{protocol_sha256}:{data_manifest_sha256}".encode("utf-8"))


def _reserve_global_receipt(*, root: Path, protocol: Mapping[str, Any],
                            protocol_path: Path, dev_report_sha256: str,
                            features_sha256: str,
                            features_index_sha256: str
                            ) -> tuple[Path, dict[str, Any]]:
    protocol_sha = _sha256_file(protocol_path)
    secret = _receipt_key(protocol_sha, protocol["data"]["manifest"]["sha256"])
    receipt_path = root / FINALIZE_RECEIPT_ROOT / f"{secret}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "format": RECEIPT_FORMAT,
        "version": VERSION,
        "status": "reserved_before_scoring",
        "receipt_key": secret,
        "protocol_sha256": protocol_sha,
        "data_manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "features_sha256": features_sha256,
        "features_index_sha256": features_index_sha256,
        "dev_report_sha256": dev_report_sha256,
        "scored_at_utc": _utc_now(),
    }
    try:
        descriptor = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                             0o600)
    except FileExistsError as error:
        raise VisionDependenceError(
            "Refusing repeat evaluation: this protocol/data holdout was "
            f"already scored ({receipt_path})") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path, receipt


def write_protocol(path: str | Path, protocol: Mapping[str, Any]) -> None:
    """Write the derived protocol atomically; data artifacts stay immutable."""
    validate_protocol(protocol)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(_json_bytes(protocol))
    os.replace(temporary, path)


def finalize_evaluation(*, protocol_path: str | Path,
                        features_path: str | Path,
                        dev_report_path: str | Path,
                        dev_report_sha256: str,
                        output: str | Path,
                        created_at: str | None = None) -> dict[str, Any]:
    """Score the final holdout exactly once, gated by the exact dev hash."""
    protocol_path = Path(protocol_path)
    features_path = Path(features_path)
    dev_report_path = Path(dev_report_path)
    output = Path(output)
    _require(_is_sha256(dev_report_sha256),
             "The exact dev report SHA256 is required")
    _require(dev_report_path.is_file(), "Dev report is missing")
    actual_dev_hash = _sha256_file(dev_report_path)
    _require(actual_dev_hash == dev_report_sha256,
             "Dev report bytes do not match the supplied exact hash")
    protocol = _load_json(protocol_path)
    validate_protocol(protocol)
    root = Path(protocol["root"])
    verify_implementation_pins(protocol, root)
    verify_data_pins(protocol, root=root)
    dataset = audit_dataset(root / protocol["data"]["root"])
    dev_report = _load_json(dev_report_path)
    _validate_dev_report(dev_report, protocol=protocol, protocol_path=protocol_path)
    feature_pin = dev_report["features"]
    _require(isinstance(feature_pin, Mapping), "Dev report feature pins are missing")
    _require(Path(feature_pin["features"]["path"]).resolve()
             == features_path.resolve(),
             "Dev report pins another feature bundle")
    _require(_sha256_file(features_path) == feature_pin["features"]["sha256"],
             "Feature array bytes differ from the dev report pin")
    predictions_pin = feature_pin.get("predictions")
    if isinstance(predictions_pin, Mapping):
        predictions_path = Path(predictions_pin["path"])
        _require(predictions_path.is_file(), "Dev predictions are missing")
        _require(_sha256_file(predictions_path) == predictions_pin["sha256"],
                 "Dev predictions changed after the dev report")
    bundle = load_feature_bundle(features_path.parent,
                                 expected_index_sha256=feature_pin["index"]["sha256"])
    _require(bundle["index"]["protocol_sha256"]
             == _sha256_file(protocol_path), "Feature bundle protocol differs")
    _require(bundle["index"]["data"] == protocol["data"],
             "Feature bundle data pins differ")
    validate_native_run_report(dev_report_path=dev_report_path,
                               dev_report_sha256=dev_report_sha256,
                               protocol=protocol, protocol_path=protocol_path,
                               feature_pins=feature_pin)
    records = records_from_bundle(dataset, bundle)
    _require(any(record["split"] == "final" for record in records),
             "Feature bundle has no final rows")
    if output.exists():
        _require(not any(output.iterdir()),
                 "Use a fresh finalize output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Reserve globally, before scoring: any crash or later attempt is refused,
    # and the receipt is scoped to this protocol plus data-manifest hash so a
    # different native output or feature directory cannot bypass it.
    receipt_path, receipt = _reserve_global_receipt(
        root=root, protocol=protocol, protocol_path=protocol_path,
        dev_report_sha256=actual_dev_hash,
        features_sha256=_sha256_file(features_path),
        features_index_sha256=feature_pin["index"]["sha256"])
    evaluation = evaluate_records(records, readout=protocol["readout"],
                                  split="final", seed=protocol["seed"])
    final_report = {
        "format": FINAL_REPORT_FORMAT,
        "version": VERSION,
        "stage": "final",
        "status": "completed",
        "protocol_sha256": receipt["protocol_sha256"],
        "dev_report_sha256": actual_dev_hash,
        "config_sha256": protocol["config_sha256"],
        "data": protocol["data"],
        "features": dict(feature_pin),
        "seed": protocol["seed"],
        "primary_readout": PRIMARY_READOUT,
        "train_cases": int(protocol["data"]["counts"]["split_model_rows"]["train"]),
        "final_cases": int(protocol["data"]["counts"]["split_model_rows"]["final"]),
        "evaluation": evaluation,
        "scores_final": True,
        "final_scored": True,
        "evaluated_once": True,
        "evaluated_splits": ["final"],
        "zero_updates": True,
        "optimizer_updates": 0,
        "parameters_updated": [],
        "decision_scope": "final_holdout_single_evaluation",
        "finalize_receipt_path": str(receipt_path),
        "finalize_receipt_key": receipt["receipt_key"],
        "created_at_utc": created_at or _utc_now(),
    }
    _write_exact(output / "final_report.json", _json_bytes(final_report))
    # Only a durable final report turns the reserved receipt into a completed
    # one. A crash before this point leaves the reserved receipt in place, so a
    # retry is refused rather than silently scoring the holdout twice.
    receipt["status"] = "completed"
    receipt["completed_at_utc"] = _utc_now()
    receipt["final_report_sha256"] = _sha256_file(output / "final_report.json")
    _atomic_write(receipt_path, _json_bytes(receipt))
    _write_exact(output / "finalize_receipt.json", _json_bytes(receipt))
    return final_report
