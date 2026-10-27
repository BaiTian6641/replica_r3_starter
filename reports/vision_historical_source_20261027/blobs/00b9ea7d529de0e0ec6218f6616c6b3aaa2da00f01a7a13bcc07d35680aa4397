"""Deterministic owned visual intake and procedural scene/chronicle preparation.

This module produces small, inspectable RGB PNG scenes from project-owned code.
It does not load a model, download an external asset, train, or admit a dataset.
Every generated asset carries an original origin/revision/terms record, byte and
record hashes, family and split lineage, and crop/layout provenance. Model rows
contain only the image path/hash and question; answers, scene graphs, checks and
temporal-memory expectations live in verifier-only payloads.

Synthetic scenes provide precise controls for color/shape/count/spatial,
absent-object and temporal questions. They do not establish broad real-image
competence and are not a substitute for separately admitted real-image data.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import struct
import zlib
from typing import Any, Mapping

from .contracts import SPLITS, canonical_json, fingerprint


VERSION = "replica.visual_data.v1"
MANIFEST_FORMAT = "replica.vision_manifest.v1"
TEMPORAL_MANIFEST_FORMAT = "replica.temporal_vision_manifest.v1"
ALLOWLIST_FORMAT = "replica.vision_external_allowlist.v1"
REPORT_FORMAT = "replica.vision_data_preparation_report.v1"
CONFIG_FORMAT = "replica.vision_data_preparation.v1"
RENDERER_VERSION = "procedural_png_v1"
SOURCE_ID = "replica_lab:procedural_scene_png_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GROUP_ID_FIELD = {"families": "family_id", "worlds": "world_id"}
SPLIT_ORDER = ("train", "dev", "calibration", "promotion", "audit")

COLORS = {
    "red": (198, 42, 45),
    "blue": (45, 84, 190),
    "green": (38, 148, 74),
    "yellow": (221, 183, 42),
    "purple": (126, 61, 166),
    "orange": (223, 127, 37),
}
COLOR_NAMES = tuple(COLORS)
SHAPES = ("circle", "square", "triangle", "rectangle")
BACKGROUND = (246, 244, 238)
FRAME = (74, 72, 68)
TASKS = ("color", "shape", "count", "spatial", "absent")
MODEL_ROW_FIELDS = {"case_id", "asset_id", "image", "image_sha256", "question"}
CROP_FRACTIONS = (
    (0.06, 0.07, 0.94, 0.93),
    (0.08, 0.09, 0.92, 0.91),
    (0.05, 0.08, 0.95, 0.92),
)


class VisualDataError(ValueError):
    """Raised when generated visual data violates its declared contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualDataError(message)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise VisualDataError(f"{name} must be a positive integer")
    return value


def _seed_rng(seed: int, *parts: str) -> random.Random:
    material = "|".join([str(seed), *parts]).encode("utf-8")
    return random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16))


# ---------------------------------------------------------------------------
# Minimal deterministic PNG codec (standard library only).
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def encode_rgb_png(width: int, height: int, rows: list[bytes | bytearray]) -> bytes:
    """Encode unfiltered RGB8 rows; compression and chunk order are deterministic."""
    _require(width > 0 and height > 0 and len(rows) == height,
             "PNG rows must match a positive image size")
    stride = width * 3
    raw = bytearray()
    for row in rows:
        _require(len(row) == stride, "PNG row width differs")
        raw.append(0)
        raw.extend(row)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (_PNG_SIGNATURE + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _png_chunk(b"IEND", b""))


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    pa, pb, pc = abs(estimate - left), abs(estimate - above), abs(estimate - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return above
    return upper_left


def decode_rgb_png(data: bytes) -> tuple[int, int, list[bytearray]]:
    """Decode the RGB8, non-interlaced PNG subset used here and verify chunk CRCs."""
    _require(data.startswith(_PNG_SIGNATURE), "Asset is not a PNG file")
    offset, width, height, payload = len(_PNG_SIGNATURE), None, None, bytearray()
    seen_end = False
    while offset < len(data):
        _require(offset + 8 <= len(data), "Truncated PNG chunk")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        start = offset + 8
        end = start + length
        _require(end + 4 <= len(data), "PNG chunk exceeds file")
        body = data[start:end]
        crc = struct.unpack(">I", data[end:end + 4])[0]
        _require(zlib.crc32(kind + body) & 0xFFFFFFFF == crc, "PNG chunk CRC differs")
        if kind == b"IHDR":
            _require(width is None and length == 13, "Invalid PNG IHDR")
            width, height, depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", body))
            _require((depth, color_type, compression, filtering, interlace)
                     == (8, 2, 0, 0, 0), "Only deterministic RGB8 PNGs are accepted")
        elif kind == b"IDAT":
            payload.extend(body)
        elif kind == b"IEND":
            seen_end = True
            offset = end + 4
            break
        offset = end + 4
    _require(width is not None and height is not None and seen_end,
             "PNG is missing IHDR/IDAT/IEND")
    raw = zlib.decompress(bytes(payload))
    stride = width * 3
    _require(len(raw) == height * (stride + 1), "PNG payload size differs")
    rows: list[bytearray] = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        line = bytearray(raw[cursor + 1:cursor + 1 + stride])
        cursor += stride + 1
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(3, stride):
                line[i] = (line[i] + line[i - 3]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - 3] if i >= 3 else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - 3] if i >= 3 else 0
                upper_left = previous[i - 3] if i >= 3 else 0
                line[i] = (line[i] + _paeth(left, previous[i], upper_left)) & 0xFF
        else:
            raise VisualDataError("Unsupported PNG row filter")
        rows.append(line)
        previous = line
    return width, height, rows


def _crop_png(png: bytes, box: tuple[int, int, int, int]
              ) -> tuple[bytes, tuple[int, int, int, int]]:
    width, height, rows = decode_rgb_png(png)
    x0, y0, x1, y1 = box
    _require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "Crop box outside image")
    cropped = [row[x0 * 3:x1 * 3] for row in rows[y0:y1]]
    return encode_rgb_png(x1 - x0, y1 - y0, cropped), (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Scene construction, rendering and executable checks.
# ---------------------------------------------------------------------------

def _crop_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for fx0, fy0, fx1, fy1 in CROP_FRACTIONS:
        box = (int(width * fx0), int(height * fy0),
               int(width * fx1), int(height * fy1))
        if box not in boxes:
            boxes.append(box)
    return boxes


def _cells(width: int, height: int) -> list[tuple[int, int]]:
    xs = [int(width * fraction) for fraction in (0.22, 0.42, 0.62, 0.82)]
    ys = [int(height * fraction) for fraction in (0.38, 0.68)]
    return [(x, y) for y in ys for x in xs]


def _object_bbox(shape: str, center: tuple[int, int], size: int
                 ) -> tuple[int, int, int, int]:
    cx, cy = center
    half_w = size + (2 if shape == "rectangle" else 0)
    return (cx - half_w, cy - size, cx + half_w, cy + size)


def _scene_from_objects(scene_id: str, width: int, height: int,
                        layout: Mapping[str, Any],
                        specs: list[tuple[str, str, int, int]]) -> dict[str, Any]:
    """Build a scene from (shape, color, cell_index, size) specs."""
    cells = _cells(width, height)
    boxes = _crop_boxes(width, height)
    objects = []
    for index, (shape, color, cell_index, size) in enumerate(specs):
        bbox = _object_bbox(shape, cells[cell_index], size)
        _require(0 <= bbox[0] and bbox[2] < width and 0 <= bbox[1] and bbox[3] < height,
                 "Object falls outside the image")
        _require(all(x0 <= bbox[0] and bbox[2] < x1 and y0 <= bbox[1] and bbox[3] < y1
                     for x0, y0, x1, y1 in boxes),
                 "Object falls outside a declared crop box")
        objects.append({
            "id": f"o{index}", "shape": shape, "color": color,
            "bbox": list(bbox), "center": list(cells[cell_index]),
        })
    _require(len({tuple(obj["center"]) for obj in objects}) == len(objects),
             "Objects must occupy distinct cells")
    return {"scene_id": scene_id, "width": width, "height": height,
            "layout": dict(layout), "objects": objects}


def _make_scene(seed: int, scene_id: str, width: int, height: int,
                count: int = 4) -> dict[str, Any]:
    """Create one deterministic scene satisfying the question preconditions."""
    rng = _seed_rng(seed, "scene", scene_id)
    unit = max(5, min(width, height) // 12)
    for _ in range(500):
        chosen = sorted(rng.sample(range(len(_cells(width, height))), count))
        shapes = [rng.choice(SHAPES) for _ in range(count)]
        colors = [rng.choice(COLOR_NAMES) for _ in range(count)]
        if len(set(zip(shapes, colors))) != count:
            continue
        shape_counts = Counter(shapes)
        color_counts = Counter(colors)
        if not any(value == 1 for value in shape_counts.values()):
            continue
        if not any(value == 1 for value in color_counts.values()):
            continue
        specs = [(shapes[index], colors[index], chosen[index],
                  unit + rng.randrange(0, 3)) for index in range(count)]
        try:
            scene = _scene_from_objects(scene_id, width, height, {
                "template": "grid_4x2_centered",
                "cell_indices": chosen,
                "crop_policy": "inner_margin_crop",
            }, specs)
        except VisualDataError:
            continue
        centers = [obj["center"] for obj in scene["objects"]]
        if not any(left[0] != right[0] for index, left in enumerate(centers)
                   for right in centers[index + 1:]):
            continue
        return scene
    raise VisualDataError(f"Could not satisfy scene constraints for {scene_id}")


def _make_unique_scene(seed: int, scene_id: str, width: int, height: int,
                       used: set[str], count: int = 4) -> dict[str, Any]:
    for nonce in range(50):
        candidate_id = scene_id if nonce == 0 else f"{scene_id}~{nonce}"
        scene = _make_scene(seed, candidate_id, width, height, count)
        digest = scene_sha256(scene)
        if digest not in used:
            used.add(digest)
            return scene
    raise VisualDataError(f"Could not create a unique scene for {scene_id}")


def _scene_truth(scene: Mapping[str, Any]) -> dict[str, Any]:
    return {"width": scene["width"], "height": scene["height"],
            "layout": scene["layout"], "objects": scene["objects"]}


def scene_sha256(scene: Mapping[str, Any]) -> str:
    return fingerprint(_scene_truth(scene))


def _paint_rect(rows: list[bytearray], width: int, height: int,
                box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    rgb = bytes(color)
    for y in range(y0, y1):
        rows[y][x0 * 3:x1 * 3] = rgb * (x1 - x0)


def render_scene_png(scene: Mapping[str, Any]) -> bytes:
    """Render one scene to deterministic RGB PNG bytes."""
    width, height = int(scene["width"]), int(scene["height"])
    rows = [bytearray(bytes(BACKGROUND) * width) for _ in range(height)]
    _paint_rect(rows, width, height, (0, 0, width, 1), FRAME)
    _paint_rect(rows, width, height, (0, height - 1, width, height), FRAME)
    _paint_rect(rows, width, height, (0, 0, 1, height), FRAME)
    _paint_rect(rows, width, height, (width - 1, 0, width, height), FRAME)
    for obj in scene["objects"]:
        x0, y0, x1, y1 = obj["bbox"]
        color = COLORS[obj["color"]]
        if obj["shape"] in {"square", "rectangle"}:
            _paint_rect(rows, width, height, (x0, y0, x1, y1), color)
        elif obj["shape"] == "circle":
            cx, cy = obj["center"]
            radius = min(x1 - x0, y1 - y0) // 2
            radius_sq = radius * radius
            for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
                dy = y - cy
                span = int((radius_sq - dy * dy) ** 0.5)
                _paint_rect(rows, width, height,
                            (cx - span, y, cx + span + 1, y + 1), color)
        else:  # triangle, base at the bottom of the bounding box
            span = x1 - x0
            total = y1 - y0
            center = (x0 + x1) // 2
            for step, y in enumerate(range(y0, y1)):
                half = max(1, (span // 2) * (step + 1) // max(total, 1))
                _paint_rect(rows, width, height,
                            (center - half, y, center + half + 1, y + 1), color)
    return encode_rgb_png(width, height, rows)


def _crop_scene(scene: Mapping[str, Any], box: tuple[int, int, int, int],
                scene_id: str) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    objects = []
    for obj in scene["objects"]:
        ox0, oy0, ox1, oy1 = obj["bbox"]
        _require(x0 <= ox0 and ox1 < x1 and y0 <= oy0 and oy1 < y1,
                 "Crop would exclude an object")
        objects.append({
            "id": obj["id"], "shape": obj["shape"], "color": obj["color"],
            "bbox": [ox0 - x0, oy0 - y0, ox1 - x0, oy1 - y0],
            "center": [obj["center"][0] - x0, obj["center"][1] - y0],
        })
    layout = dict(scene["layout"])
    layout.update({"parent_layout": dict(scene["layout"]), "crop_box": list(box)})
    return {"scene_id": scene_id, "width": x1 - x0, "height": y1 - y0,
            "layout": layout, "objects": objects}


def _changed_color_scene(scene: Mapping[str, Any], object_id: str, color: str,
                         scene_id: str) -> dict[str, Any]:
    _require(color in COLORS, "Unknown replacement color")
    objects = []
    for obj in scene["objects"]:
        item = {"id": obj["id"], "shape": obj["shape"],
                "color": color if obj["id"] == object_id else obj["color"],
                "bbox": list(obj["bbox"]), "center": list(obj["center"])}
        objects.append(item)
    return {"scene_id": scene_id, "width": scene["width"], "height": scene["height"],
            "layout": dict(scene["layout"]), "objects": objects}


def _check_value(check: Mapping[str, Any], scene: Mapping[str, Any]) -> str:
    """Execute one verifier check against verifier-only scene truth."""
    kind = check.get("kind")
    objects = scene["objects"]
    by_id = {obj["id"]: obj for obj in objects}
    if kind == "count":
        criteria = check.get("filter", {})
        matched = [obj for obj in objects
                   if all(obj.get(key) == value for key, value in criteria.items())]
        return str(len(matched))
    if kind == "color_of":
        _require(check.get("object_id") in by_id, "Unknown object in color check")
        return str(by_id[check["object_id"]]["color"])
    if kind == "shape_of":
        _require(check.get("object_id") in by_id, "Unknown object in shape check")
        return str(by_id[check["object_id"]]["shape"])
    if kind == "absent":
        found = any(obj["color"] == check.get("color") and obj["shape"] == check.get("shape")
                    for obj in objects)
        return "yes" if found else "no"
    if kind == "spatial":
        first, second = by_id.get(check.get("first_id")), by_id.get(check.get("second_id"))
        _require(first is not None and second is not None and first is not second,
                 "Spatial check needs two distinct objects")
        relation = check.get("relation")
        ax, ay = first["center"]
        bx, by = second["center"]
        actual = {
            "left_of": ax < bx,
            "right_of": ax > bx,
            "above": ay < by,
            "below": ay > by,
        }.get(relation)
        _require(actual is not None, "Unknown spatial relation")
        return "yes" if actual else "no"
    raise VisualDataError(f"Unknown check kind: {kind}")


def _unique_shape_object(scene: Mapping[str, Any], shape: str | None = None
                         ) -> dict[str, Any]:
    counts = Counter(obj["shape"] for obj in scene["objects"])
    if shape is not None:
        _require(counts[shape] == 1, "Requested shape is not unique")
        return next(obj for obj in scene["objects"] if obj["shape"] == shape)
    unique = [obj for obj in scene["objects"] if counts[obj["shape"]] == 1]
    _require(bool(unique), "Scene lacks a unique shape")
    return unique[0]


def _unique_color_object(scene: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(obj["color"] for obj in scene["objects"])
    unique = [obj for obj in scene["objects"] if counts[obj["color"]] == 1]
    _require(bool(unique), "Scene lacks a unique color")
    return unique[0]


def _build_question(task: str, scene: Mapping[str, Any], rng: random.Random,
                    forced_shape: str | None = None) -> dict[str, Any]:
    if task == "color":
        obj = _unique_shape_object(scene, forced_shape)
        return {"task": task, "question": f"What color is the {obj['shape']}?",
                "check": {"kind": "color_of", "object_id": obj["id"]}}
    if task == "shape":
        obj = _unique_color_object(scene)
        return {"task": task, "question": f"What shape is the {obj['color']} object?",
                "check": {"kind": "shape_of", "object_id": obj["id"]}}
    if task == "count":
        return {"task": task, "question": "How many objects are in the image?",
                "check": {"kind": "count", "filter": {}}}
    if task == "spatial":
        pairs = [(first, second) for index, first in enumerate(scene["objects"])
                 for second in scene["objects"][index + 1:]
                 if first["center"][0] != second["center"][0]]
        relation = "left_of"
        if not pairs:
            pairs = [(first, second) for index, first in enumerate(scene["objects"])
                     for second in scene["objects"][index + 1:]
                     if first["center"][1] != second["center"][1]]
            relation = "above"
        _require(bool(pairs), "Scene lacks a spatial pair")
        first, second = rng.choice(pairs)
        if relation == "left_of" and rng.randrange(2):
            first, second = second, first
        phrase = "to the left of" if relation == "left_of" else "above"
        return {"task": task,
                "question": (f"Is the {first['color']} {first['shape']} {phrase} "
                             f"the {second['color']} {second['shape']}?"),
                "check": {"kind": "spatial", "first_id": first["id"],
                          "second_id": second["id"], "relation": relation}}
    if task == "absent":
        present = {(obj["color"], obj["shape"]) for obj in scene["objects"]}
        choices = [(color, shape) for color in COLOR_NAMES for shape in SHAPES
                   if (color, shape) not in present]
        color, shape = rng.choice(choices)
        return {"task": task,
                "question": f"Is there a {color} {shape} in the image?",
                "check": {"kind": "absent", "color": color, "shape": shape}}
    raise VisualDataError(f"Unknown question task: {task}")


def _verified_question(task: str, scene: Mapping[str, Any], rng: random.Random,
                       forced_shape: str | None = None) -> dict[str, Any]:
    item = _build_question(task, scene, rng, forced_shape)
    computed = _check_value(item["check"], scene)
    item["check"]["expected"] = computed
    return {**item, "answer": computed}


# ---------------------------------------------------------------------------
# Configuration and provenance.
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("format") == CONFIG_FORMAT, "Unknown visual data config format")
    _positive_int(config.get("seed"), "seed")
    renderer = config.get("renderer")
    _require(isinstance(renderer, dict), "renderer config is required")
    _require(renderer.get("version") == RENDERER_VERSION, "Unknown renderer version")
    _positive_int(renderer.get("width"), "renderer.width")
    _positive_int(renderer.get("height"), "renderer.height")
    _require(renderer.get("image_format") == "png_rgb8", "Only png_rgb8 is supported")
    campaign = config.get("campaign")
    _require(isinstance(campaign, dict), "campaign config is required")
    ceiling = _positive_int(campaign.get("asset_ceiling_bytes"), "campaign.asset_ceiling_bytes")
    _require(ceiling >= 1 << 30, "Asset ceiling must retain real headroom")
    _require(campaign.get("checkpoint_bytes_separate") is True,
             "Checkpoint bytes must be accounted separately")
    _require(campaign.get("ram_guard_separate") is True,
             "RAM guard must be accounted separately")
    v0 = config.get("v0")
    _require(isinstance(v0, dict), "v0 config is required")
    families = _positive_int(v0.get("root_families"), "v0.root_families")
    crops = v0.get("crop_families")
    if type(crops) is not int or not 0 <= crops <= families:
        raise VisualDataError("v0.crop_families must be between zero and root_families")
    _positive_int(v0.get("max_assets"), "v0.max_assets")
    _require(families + crops <= v0["max_assets"], "V0 exceeds its asset cap")
    _positive_int(v0.get("cases_per_asset"), "v0.cases_per_asset")
    split_counts = v0.get("split_family_counts")
    _require(isinstance(split_counts, dict) and set(split_counts) == set(SPLITS),
             "v0.split_family_counts must name every split contract role")
    if any(type(value) is not int or value < 1 for value in split_counts.values()):
        raise VisualDataError("V0 split family counts must be positive integers")
    _require(sum(split_counts.values()) == families,
             "V0 split family counts must sum to root_families")
    v1 = config.get("v1")
    _require(isinstance(v1, dict), "v1 config is required")
    worlds = _positive_int(v1.get("worlds"), "v1.worlds")
    dev_worlds = v1.get("dev_worlds")
    if type(dev_worlds) is not int or not 0 <= dev_worlds < worlds:
        raise VisualDataError("v1.dev_worlds must be between zero and worlds-1")
    if v1.get("assets_per_world") != 4:
        raise VisualDataError("v1.assets_per_world must be 4 (two paired scenes and crops)")
    _positive_int(v1.get("max_assets"), "v1.max_assets")
    _require(worlds * v1["assets_per_world"] <= v1["max_assets"], "V1 exceeds its asset cap")
    _positive_int(v1.get("cases_per_asset"), "v1.cases_per_asset")
    _require(worlds >= len(TASKS) - 1,
             "v1.worlds must cover every V1 task variant")
    temporal = config.get("temporal")
    _require(isinstance(temporal, dict), "temporal config is required")
    t_worlds = _positive_int(temporal.get("worlds"), "temporal.worlds")
    t_dev = temporal.get("dev_worlds")
    if type(t_dev) is not int or not 0 <= t_dev <= t_worlds:
        raise VisualDataError("temporal.dev_worlds must be between zero and worlds")
    if temporal.get("observations_per_world") != 3:
        raise VisualDataError("temporal.observations_per_world must be 3")
    if temporal.get("queries_per_world") != 4:
        raise VisualDataError("temporal.queries_per_world must be 4")
    external = config.get("external")
    _require(isinstance(external, dict), "external config is required")
    _require(external.get("acquire_assets") is False,
             "This preparation profile never acquires external assets")
    if external.get("max_external_assets") != 0:
        raise VisualDataError("This preparation profile permits zero external assets")
    _require(isinstance(external.get("catalog"), str)
             and isinstance(external.get("runtime_registry"), str),
             "external catalog and runtime registry paths are required")
    for name in ("catalog", "runtime_registry"):
        path = Path(external[name])
        _require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
                 f"external {name} must be a repository-relative path")
    return dict(config)


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    return fingerprint(config)


def _asset_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    renderer = config["renderer"]
    return {
        "origin": "internal_owned_procedural_generation",
        "source_id": SOURCE_ID,
        "revision": {"renderer_version": renderer["version"],
                     "config_fingerprint": _config_fingerprint(config)},
        "terms": {"external_assets": 0, "external_terms": "not_applicable",
                  "basis": "project-owned generator output with no upstream image bytes",
                  "redistribution": "project_controlled_pending_review"},
        "generation": "standard_library_only_deterministic_render",
    }


def _record_hash(asset: Mapping[str, Any]) -> str:
    body = {key: value for key, value in asset.items() if key != "record_hash"}
    return fingerprint(body)


# ---------------------------------------------------------------------------
# V0 / V1 stage construction.
# ---------------------------------------------------------------------------

def _split_roles(split_counts: Mapping[str, int]) -> list[str]:
    """Interleave roles so derived crops spread across every contract split."""
    remaining = dict(split_counts)
    roles: list[str] = []
    while any(remaining.values()):
        for split in SPLIT_ORDER:
            if remaining[split] > 0:
                roles.append(split)
                remaining[split] -= 1
    return roles


def _asset_entry(stage: str, asset_id: str, group_id: str, family: str, split: str,
                 path: str, png: bytes, scene: Mapping[str, Any], *,
                 parent: Mapping[str, Any] | None = None,
                 crop_box: list[int] | None = None,
                 pair_id: str | None = None, variant: str | None = None,
                 decisive_attribute: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entry = {
        "asset_id": asset_id,
        "stage": stage,
        "group_id": group_id,
        "family": family,
        "split": split,
        "path": path,
        "bytes": len(png),
        "sha256": _sha256(png),
        "width": scene["width"],
        "height": scene["height"],
        "scene_sha256": scene_sha256(scene),
        "lineage": {
            "parent_asset_id": parent["asset_id"] if parent else None,
            "crop_box": list(crop_box) if crop_box else None,
            "layout_template": scene["layout"]["template"],
            "variant": variant,
        },
    }
    if pair_id is not None:
        entry["pair_id"] = pair_id
        entry["decisive_attribute"] = dict(decisive_attribute or {})
    entry["record_hash"] = _record_hash(entry)
    return entry


def _cases_for_asset(asset: Mapping[str, Any], scene: Mapping[str, Any],
                     tasks: list[str], seed: int,
                     forced_shape: str | None = None
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                dict[str, Any]]:
    rng = _seed_rng(seed, "questions", asset["asset_id"])
    rows, cases = [], []
    for task in tasks:
        item = _verified_question(task, scene, rng, forced_shape)
        case_id = f"{asset['asset_id']}-{task}"
        rows.append({"case_id": case_id, "asset_id": asset["asset_id"],
                     "image": asset["path"], "image_sha256": asset["sha256"],
                     "question": item["question"]})
        cases.append({
            "case_id": case_id, "asset_id": asset["asset_id"],
            "group_id": asset["group_id"], "family": asset["family"],
            "split": asset["split"], "task": task,
            "question": item["question"], "answer": item["answer"],
            "check": item["check"], "scene_sha256": asset["scene_sha256"],
            "label_visibility": "verifier_only",
        })
    return rows, cases, {"asset_id": asset["asset_id"],
                         "scene_sha256": asset["scene_sha256"], "scene": scene}


def _construct_v0(config: Mapping[str, Any]) -> dict[str, Any]:
    v0 = config["v0"]
    width, height = config["renderer"]["width"], config["renderer"]["height"]
    seed = config["seed"]
    roles = _split_roles(v0["split_family_counts"])
    crop_families = set(range(v0["crop_families"]))
    assets, asset_bytes, rows, cases, scenes, families = [], [], [], [], [], []
    used_scenes: set[str] = set()
    for index, split in enumerate(roles):
        family_id = f"v0-family-{index:03d}"
        scene = _make_unique_scene(seed, f"{family_id}-root", width, height, used_scenes)
        png = render_scene_png(scene)
        root = _asset_entry("v0", f"v0-f{index:03d}-root", family_id, family_id, split,
                            f"v0/assets/v0-f{index:03d}-root.png", png, scene,
                            variant="root")
        assets.append(root)
        asset_bytes.append(png)
        tasks = [TASKS[(index + offset) % len(TASKS)]
                 for offset in range(v0["cases_per_asset"])]
        asset_rows, asset_cases, scene_record = _cases_for_asset(root, scene, tasks, seed)
        rows.extend(asset_rows)
        cases.extend(asset_cases)
        scenes.append(scene_record)
        family_assets = [root["asset_id"]]
        if index in crop_families:
            box = _crop_boxes(width, height)[index % len(_crop_boxes(width, height))]
            crop_png, actual_box = _crop_png(png, box)
            crop_scene = _crop_scene(scene, actual_box, f"{family_id}-crop")
            crop = _asset_entry("v0", f"v0-f{index:03d}-crop", family_id, family_id,
                                split, f"v0/assets/v0-f{index:03d}-crop.png",
                                crop_png, crop_scene, parent=root,
                                crop_box=list(actual_box), variant="crop")
            assets.append(crop)
            asset_bytes.append(crop_png)
            crop_tasks = [TASKS[(index + offset + 2) % len(TASKS)]
                          for offset in range(v0["cases_per_asset"])]
            crop_rows, crop_cases, crop_scene_record = _cases_for_asset(
                crop, crop_scene, crop_tasks, seed)
            rows.extend(crop_rows)
            cases.extend(crop_cases)
            scenes.append(crop_scene_record)
            family_assets.append(crop["asset_id"])
        families.append({"family_id": family_id, "split": split,
                         "asset_ids": family_assets})
    return {"assets": assets, "asset_bytes": asset_bytes, "rows": rows,
            "cases": cases, "scenes": scenes, "groups": families,
            "group_key": "families"}


def _construct_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    v1 = config["v1"]
    width, height = config["renderer"]["width"], config["renderer"]["height"]
    seed = config["seed"]
    assets, asset_bytes, rows, cases, scenes, worlds = [], [], [], [], [], []
    used_scenes: set[str] = set()
    second_tasks = ("shape", "count", "spatial", "absent")
    for world_index in range(v1["worlds"]):
        split = "dev" if world_index < v1["dev_worlds"] else "train"
        world_id = f"v1-world-{world_index:03d}"
        world_assets = []
        pair_index = 0
        pair_key = f"{world_id}-p{pair_index}"
        base_id = f"v1-w{world_index:03d}-p{pair_index}-a"
        scene_a = _make_unique_scene(seed, base_id, width, height, used_scenes)
        changed_object = _unique_shape_object(scene_a)
        replacement = next(color for color in COLOR_NAMES
                           if color not in {obj["color"] for obj in scene_a["objects"]})
        scene_b = _changed_color_scene(scene_a, changed_object["id"], replacement,
                                       f"{base_id}-b")
        digest_b = scene_sha256(scene_b)
        _require(digest_b not in used_scenes, "V1 pair variants share scene truth")
        used_scenes.add(digest_b)
        changed = {"object_id": changed_object["id"], "attribute": "color",
                   "from": changed_object["color"], "to": replacement}
        tasks = ["color", second_tasks[world_index % 4]]
        for label, scene in (("a", scene_a), ("b", scene_b)):
            png = render_scene_png(scene)
            asset_id = f"v1-w{world_index:03d}-p{pair_index}-{label}"
            root = _asset_entry("v1", asset_id, world_id, world_id, split,
                                f"v1/assets/{asset_id}.png", png, scene,
                                pair_id=pair_key, variant=label,
                                decisive_attribute=changed)
            assets.append(root)
            asset_bytes.append(png)
            world_assets.append(asset_id)
            asset_rows, asset_cases, scene_record = _cases_for_asset(
                root, scene, tasks, seed, forced_shape=changed_object["shape"])
            rows.extend(asset_rows)
            cases.extend(asset_cases)
            scenes.append(scene_record)
            box = _crop_boxes(width, height)[
                (world_index + pair_index + (label == "b")) % len(_crop_boxes(width, height))]
            crop_png, actual_box = _crop_png(png, box)
            crop_id = f"{asset_id}-crop"
            crop_scene = _crop_scene(scene, actual_box, f"{crop_id}")
            crop = _asset_entry("v1", crop_id, world_id, world_id, split,
                                f"v1/assets/{crop_id}.png", crop_png, crop_scene,
                                parent=root, crop_box=list(actual_box),
                                pair_id=pair_key, variant=f"{label}-crop",
                                decisive_attribute=changed)
            assets.append(crop)
            asset_bytes.append(crop_png)
            world_assets.append(crop_id)
            crop_rows, crop_cases, crop_scene_record = _cases_for_asset(
                crop, crop_scene, tasks, seed, forced_shape=changed_object["shape"])
            rows.extend(crop_rows)
            cases.extend(crop_cases)
            scenes.append(crop_scene_record)
        worlds.append({"world_id": world_id, "split": split,
                       "pair_ids": [f"{world_id}-p0"],
                       "asset_ids": world_assets})
    return {"assets": assets, "asset_bytes": asset_bytes, "rows": rows,
            "cases": cases, "scenes": scenes, "groups": worlds,
            "group_key": "worlds"}


# ---------------------------------------------------------------------------
# Temporal-memory candidate worlds.
# ---------------------------------------------------------------------------

def _temporal_scene(config: Mapping[str, Any], scene_id: str,
                    colors: Mapping[str, str]) -> dict[str, Any]:
    width, height = config["renderer"]["width"], config["renderer"]["height"]
    rng = _seed_rng(config["seed"], "temporal-scene", scene_id)
    unit = max(5, min(width, height) // 12)
    specs = [(shape, colors[shape], index, unit + rng.randrange(0, 3))
             for index, shape in enumerate(SHAPES)]
    return _scene_from_objects(scene_id, width, height, {
        "template": "temporal_row_4",
        "cell_indices": [0, 1, 2, 3],
        "crop_policy": "inner_margin_crop",
    }, specs)


def _temporal_world(config: Mapping[str, Any], index: int, split: str,
                    used: set[str]) -> dict[str, Any]:
    seed = config["seed"]
    world_id = f"temporal-world-{index:03d}"
    rng = _seed_rng(seed, "temporal", world_id)
    for nonce in range(50):
        colors = list(COLOR_NAMES)
        rng.shuffle(colors)
        initial = dict(zip(SHAPES, colors[:4], strict=True))
        scene_t1 = _temporal_scene(config, f"{world_id}-t1"
                                   if nonce == 0 else f"{world_id}-t1~{nonce}", initial)
        if scene_sha256(scene_t1) not in used:
            used.add(scene_sha256(scene_t1))
            break
    else:
        raise VisualDataError(f"Could not create a unique temporal world: {world_id}")
    correction_color = next(color for color in COLOR_NAMES
                            if color not in initial.values())
    change_color = next(color for color in COLOR_NAMES
                        if color not in initial.values() and color != correction_color)
    fact_object = {f"{obj['shape']}_color": obj["id"] for obj in scene_t1["objects"]}
    _require(set(fact_object) == {f"{shape}_color" for shape in SHAPES},
             "Temporal world must render exactly four named shapes")
    scene_t2 = _changed_color_scene(scene_t1, fact_object["square_color"],
                                    correction_color, f"{world_id}-t2")
    scene_t3 = _changed_color_scene(scene_t2, fact_object["circle_color"],
                                    change_color, f"{world_id}-t3")
    events = [{"time": 1, "kind": "assert", "fact": f"{shape}_color",
               "value": initial[shape]} for shape in SHAPES]
    events += [
        {"time": 2, "kind": "correction", "fact": "square_color",
         "value": correction_color},
        {"time": 3, "kind": "change", "fact": "circle_color", "value": change_color},
        {"time": 3, "kind": "revocation", "fact": "triangle_color"},
    ]
    observations = [
        {"observation_id": f"{world_id}-t1", "time": 1,
         "caption": "Dated note 1: " + "; ".join(
             f"{shape} color {initial[shape]}" for shape in SHAPES) + "."},
        {"observation_id": f"{world_id}-t2", "time": 2,
         "caption": ("Dated note 2: the earlier square color record was wrong; "
                     f"the square is {correction_color}.")},
        {"observation_id": f"{world_id}-t3", "time": 3,
         "caption": (f"Dated note 3: the circle was repainted {change_color}; "
                     "the triangle color record was revoked.")},
    ]
    resolved = resolve_temporal_queries(events)
    queries = [
        {"query_id": f"{world_id}-q-{fact}", "fact": fact,
         "question": f"What color is the {fact.split('_')[0]}?",
         "expected_answer": detail["value"], "expected_status": detail["status"],
         "unaffected": detail["status"] == "unchanged", "cutoff_time": 4}
        for fact, detail in resolved.items()
    ]
    _require({query["expected_status"] for query in queries}
             == {"unchanged", "corrected", "changed", "revoked"},
             "Temporal world must cover all four lifecycle statuses")
    return {"world_id": world_id, "split": split,
            "scenes": {"t1": scene_t1, "t2": scene_t2, "t3": scene_t3},
            "observations": observations, "events": events,
            "fact_object": fact_object, "queries": queries}


def _construct_temporal(config: Mapping[str, Any]) -> dict[str, Any]:
    temporal = config["temporal"]
    assets, asset_bytes, groups, verifier_worlds, scene_records = [], [], [], [], []
    used_scenes: set[str] = set()
    for index in range(temporal["worlds"]):
        split = "dev" if index < temporal["dev_worlds"] else "audit"
        world = _temporal_world(config, index, split, used_scenes)
        asset_ids = {}
        public_observations = []
        world_scenes = {}
        for time in ("t1", "t2", "t3"):
            scene = world["scenes"][time]
            png = render_scene_png(scene)
            asset_id = f"{world['world_id']}-{time}"
            asset = _asset_entry("temporal", asset_id, world["world_id"],
                                 world["world_id"], split,
                                 f"temporal/assets/{asset_id}.png", png, scene,
                                 variant=time)
            assets.append(asset)
            asset_bytes.append(png)
            asset_ids[time] = asset_id
            world_scenes[time] = scene
            scene_records.append({"asset_id": asset_id,
                                  "scene_sha256": asset["scene_sha256"],
                                  "scene": scene})
        for observation in world["observations"]:
            time = f"t{observation['time']}"
            asset = next(item for item in assets if item["asset_id"] == asset_ids[time])
            public_observations.append({
                "observation_id": observation["observation_id"],
                "time": observation["time"],
                "asset_id": asset_ids[time],
                "image": asset["path"],
                "image_sha256": asset["sha256"],
                "caption": observation["caption"],
                "label_visibility": "model_observation",
            })
        groups.append({"world_id": world["world_id"], "split": split,
                       "asset_ids": list(asset_ids.values()),
                       "observations": public_observations})
        verifier_worlds.append({
            "world_id": world["world_id"], "split": split,
            "asset_ids": asset_ids, "events": world["events"],
            "fact_object": world["fact_object"], "queries": world["queries"],
            "scenes": world_scenes, "label_visibility": "verifier_only",
        })
    return {"assets": assets, "asset_bytes": asset_bytes, "groups": groups,
            "verifier_worlds": verifier_worlds, "scenes": scene_records}


def resolve_temporal_queries(events: list[Mapping[str, Any]]
                             ) -> dict[str, dict[str, Any]]:
    """Independently resolve temporal expectations from the ordered event ledger."""
    state: dict[str, dict[str, Any]] = {}
    for event in events:
        fact, kind = event["fact"], event["kind"]
        if kind == "assert":
            if fact in state:
                raise VisualDataError("Duplicate temporal assertion")
            state[fact] = {"value": event["value"], "status": "unchanged",
                           "event_times": [event["time"]]}
        elif kind in {"correction", "change"}:
            if fact not in state:
                raise VisualDataError("Temporal update without assertion")
            state[fact] = {"value": event["value"],
                           "status": "corrected" if kind == "correction" else "changed",
                           "event_times": state[fact]["event_times"] + [event["time"]]}
        elif kind == "revocation":
            if fact not in state:
                raise VisualDataError("Temporal revocation without assertion")
            state[fact] = {"value": "unknown", "status": "revoked",
                           "event_times": state[fact]["event_times"] + [event["time"]]}
        else:
            raise VisualDataError("Unknown temporal event kind")
    return state


def temporal_state_at(events: list[Mapping[str, Any]], time: int
                      ) -> dict[str, dict[str, Any]]:
    """Resolve the ledger through a logical time, for visible-state auditing."""
    applicable = [event for event in events if event["time"] <= time]
    return resolve_temporal_queries(applicable)


# ---------------------------------------------------------------------------
# External component allowlist (catalog/registry are read-only inputs).
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    return _sha256(Path(path).read_bytes())


def build_external_allowlist(catalog_path: Path, registry_path: Path, *,
                             root: Path | None = None) -> dict[str, Any]:
    """Collapse nothing: every intake stage is reported and left unadmitted."""
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    sources = catalog.get("sources")
    _require(isinstance(sources, list), "Source catalog lacks sources")
    _require(catalog.get("external_sources") == len(sources),
             "Source catalog count differs from its entries")
    entries = []
    for source in sources:
        revision = source.get("immutable_revision")
        pinned = bool(revision) and bool(SHA256_RE.fullmatch(source.get("metadata_sha256") or ""))
        acquired = bool(source.get("dataset_asset_downloaded_in_this_update"))
        inspected = bool(source.get("content_inspected"))
        normalized = bool(source.get("normalized"))
        admitted = bool(source.get("training_admitted"))
        blockers = []
        if not pinned:
            blockers.append("immutable_revision_or_metadata_hash_missing")
        if not acquired:
            blockers.append("component_assets_not_acquired")
        if not inspected:
            blockers.append("content_not_inspected")
        if not normalized:
            blockers.append("not_normalized_or_deduplicated")
        if not admitted:
            blockers.append("not_admitted_for_a_named_split_or_run")
        entries.append({
            "id": source["id"],
            "category": source.get("category"),
            "source_url": source.get("source_url"),
            "proposed_role": source.get("proposed_role"),
            "split_group_key": source.get("split_group_key"),
            "immutable_revision": revision,
            "card_license_claim": source.get("card_license_claim"),
            "stages": {
                "located": {"status": bool(source.get("source_url")),
                            "evidence": source.get("source_url")},
                "pinned": {"status": pinned, "evidence": revision,
                           "metadata_sha256": source.get("metadata_sha256")},
                "acquired": {"status": acquired, "bytes": None,
                             "evidence": "catalog acquisition flag"},
                "inspected": {"status": inspected,
                              "evidence": "catalog inspection flag"},
                "normalized": {"status": normalized,
                               "evidence": "catalog normalization flag"},
                "admitted": {"status": admitted,
                             "evidence": source.get("admission_status")},
            },
            "terms_note": source.get("review_notes", ""),
            "eligible_for_this_campaign": (pinned and acquired and inspected
                                           and normalized and admitted),
            "blockers": blockers,
        })
    pinned_count = sum(entry["stages"]["pinned"]["status"] for entry in entries)
    return {
        "format": ALLOWLIST_FORMAT,
        "reviewed_on": catalog.get("reviewed_on"),
        "catalog": {"path": str(catalog_path.relative_to(root) if root is not None else catalog_path),
                    "sha256": _file_sha256(catalog_path)},
        "runtime_registry": {
            "path": str(registry_path.relative_to(root) if root is not None else registry_path),
            "sha256": _file_sha256(registry_path),
            "entries": len(registry.get("entries", [])),
            "status": registry.get("status"),
            "role": "existing experiment-admission interface; not overridden here",
        },
        "counts": {
            "catalog_external_sources": len(entries),
            "metadata_pinned": pinned_count,
            "awaiting_immutable_pin": len(entries) - pinned_count,
            "assets_acquired": sum(entry["stages"]["acquired"]["status"]
                                   for entry in entries),
            "content_inspected": sum(entry["stages"]["inspected"]["status"]
                                     for entry in entries),
            "normalized": sum(entry["stages"]["normalized"]["status"]
                              for entry in entries),
            "admitted": sum(entry["stages"]["admitted"]["status"] for entry in entries),
            "eligible_for_this_campaign": sum(entry["eligible_for_this_campaign"]
                                              for entry in entries),
        },
        "acquisition_policy": {
            "requested_by_this_profile": False,
            "assets_acquired_here": 0,
            "bytes_acquired_here": 0,
            "reason": ("No catalog component has completed asset inspection, "
                       "normalization, eligibility and named admission; bulk "
                       "downloads are outside this preparation profile."),
            "internal_generated_bytes_are_distinct": True,
        },
        "entries": entries,
    }


def _portable_allowlist(value: Mapping[str, Any], external: Mapping[str, Any]) -> dict[str, Any]:
    """Compare source identity across clones without rewriting old provenance.

    Legacy absolute locators may differ only by a common checkout prefix. Every
    source hash, intake field and entry is still compared with local recomputation.
    Never open a file at a stored locator; the declared root supplies the inputs.
    """
    result = dict(value)
    legacy_roots = set()
    for name in ("catalog", "runtime_registry"):
        expected = Path(external[name])
        record = value.get(name)
        _require(isinstance(record, dict) and isinstance(record.get("path"), str),
                 f"Allowlist {name} lacks a source path")
        stored = Path(record["path"])
        if stored.is_absolute():
            _require(".." not in stored.parts
                     and stored.parts[-len(expected.parts):] == expected.parts,
                     f"Allowlist {name} path differs from configured source")
            legacy_roots.add(stored.parents[len(expected.parts) - 1])
        else:
            _require(stored == expected, f"Allowlist {name} path differs from configured source")
        result[name] = {**record, "path": str(expected)}
    _require(len(legacy_roots) <= 1, "Allowlist source checkout roots differ")
    return result


# ---------------------------------------------------------------------------
# Manifest and file writing.
# ---------------------------------------------------------------------------

def _counts(built: Mapping[str, Any]) -> dict[str, Any]:
    assets, rows, cases = built["assets"], built["rows"], built["cases"]
    split_assets = Counter(asset["split"] for asset in assets)
    split_rows = Counter(case["split"] for case in cases)
    split_groups = Counter(group["split"] for group in built["groups"])
    return {
        "assets": len(assets),
        "groups": len(built["groups"]),
        "model_rows": len(rows),
        "verifier_cases": len(cases),
        "verifier_scenes": len(built["scenes"]),
        "asset_bytes": sum(asset["bytes"] for asset in assets),
        "task_histogram": dict(sorted(Counter(case["task"] for case in cases).items())),
        "split_assets": dict(sorted(split_assets.items())),
        "split_groups": dict(sorted(split_groups.items())),
        "split_model_rows": dict(sorted(split_rows.items())),
        "root_assets": sum(asset["lineage"]["parent_asset_id"] is None for asset in assets),
        "crop_assets": sum(asset["lineage"]["parent_asset_id"] is not None
                           for asset in assets),
    }


def _generation_fingerprint(stage: str, config: Mapping[str, Any],
                            built: Mapping[str, Any]) -> str:
    group_field = GROUP_ID_FIELD[built["group_key"]]
    return fingerprint({
        "format": MANIFEST_FORMAT,
        "stage": stage,
        "seed": config["seed"],
        "config": _config_fingerprint(config),
        "renderer": config["renderer"],
        "asset_sha256": [asset["sha256"] for asset in built["assets"]],
        "case_sha256": [fingerprint({"case_id": case["case_id"],
                                     "question": case["question"],
                                     "answer": case["answer"],
                                     "check": case["check"]}) for case in built["cases"]],
        "scene_sha256": [scene["scene_sha256"] for scene in built["scenes"]],
        "groups": [{group_field: group[group_field], "split": group["split"],
                    "asset_ids": group["asset_ids"]} for group in built["groups"]],
    })


def _manifest(stage: str, config: Mapping[str, Any],
              built: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "stage": stage,
        "status": config[stage]["status"],
        "seed": config["seed"],
        "config_fingerprint": _config_fingerprint(config),
        "renderer": dict(config["renderer"]),
        "provenance": _asset_provenance(config),
        "training_admission": {
            "admitted": False, "admitted_rows": 0,
            "reason": "Generated preparation data; a named protocol must admit splits.",
        },
        "counts": _counts(built),
        "group_key": built["group_key"],
        "groups": built["groups"],
        "assets": built["assets"],
        "cases": {
            "model_rows": f"{stage}/model_rows.jsonl",
            "verifier_cases": f"{stage}/verifier_cases.jsonl",
            "verifier_scenes": f"{stage}/verifier_scenes.jsonl",
            "model_row_fields": sorted(MODEL_ROW_FIELDS),
            "label_visibility": "verifier_only",
        },
        "generation_fingerprint": _generation_fingerprint(stage, config, built),
    }


def write_exact(path: Path, data: bytes) -> None:
    """Create a file once or verify identical existing bytes; never overwrite drift."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise VisualDataError(f"Existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(values: list[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(value) + "\n").encode("utf-8") for value in values)


def _temporal_manifest(config: Mapping[str, Any],
                       built: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "assets": len(built["assets"]), "groups": len(built["groups"]),
        "observations": sum(len(group["observations"]) for group in built["groups"]),
        "queries": sum(len(world["queries"]) for world in built["verifier_worlds"]),
        "verifier_scenes": len(built["scenes"]),
        "asset_bytes": sum(asset["bytes"] for asset in built["assets"]),
    }
    generation = fingerprint({
        "format": TEMPORAL_MANIFEST_FORMAT,
        "config": _config_fingerprint(config),
        "seed": config["seed"],
        "asset_sha256": [asset["sha256"] for asset in built["assets"]],
        "worlds": [fingerprint({"world_id": world["world_id"],
                                "events": world["events"],
                                "queries": world["queries"],
                                "scenes": {time: scene_sha256(scene)
                                           for time, scene in world["scenes"].items()}})
                   for world in built["verifier_worlds"]],
    })
    return {
        "format": TEMPORAL_MANIFEST_FORMAT, "version": VERSION, "stage": "temporal",
        "status": config["temporal"]["status"], "seed": config["seed"],
        "config_fingerprint": _config_fingerprint(config),
        "renderer": dict(config["renderer"]),
        "provenance": _asset_provenance(config),
        "training_admission": {"admitted": False, "admitted_rows": 0,
                               "reason": "Temporal storage-probe candidates only."},
        "counts": counts, "assets": built["assets"], "groups": built["groups"],
        "generation_fingerprint": generation,
    }


# ---------------------------------------------------------------------------
# Audit.
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisualDataError(f"Missing generated artifact: {path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise VisualDataError(f"Missing generated artifact: {path}") from exc
    return [json.loads(line) for line in text.splitlines() if line]


def _safe_asset_path(output: Path, relative: str) -> Path:
    candidate = (output / relative).resolve()
    root = output.resolve()
    _require(candidate == root or root in candidate.parents,
             "Asset path escapes the generated dataset root")
    return candidate


def _audit_asset_files(output: Path, assets: list[Mapping[str, Any]],
                       asset_dir_prefix: str) -> dict[str, bytes]:
    payloads = {}
    seen_hashes: dict[str, str] = {}
    for asset in assets:
        path = _safe_asset_path(output, asset["path"])
        _require(path.suffix == ".png", "Assets must be PNG files")
        _require(path.exists(), f"Declared asset is missing: {asset['path']}")
        data = path.read_bytes()
        if len(data) != asset["bytes"]:
            raise VisualDataError(f"Changed asset bytes: {asset['path']}")
        if _sha256(data) != asset["sha256"]:
            raise VisualDataError(f"Changed asset hash: {asset['path']}")
        width, height, _ = decode_rgb_png(data)
        if (width, height) != (asset["width"], asset["height"]):
            raise VisualDataError(f"Asset dimensions differ: {asset['path']}")
        if asset["sha256"] in seen_hashes:
            raise VisualDataError(
                f"Duplicate input asset: {asset['path']} matches "
                f"{seen_hashes[asset['sha256']]}")
        seen_hashes[asset["sha256"]] = asset["path"]
        _require(asset["record_hash"] == _record_hash(asset),
                 f"Asset record hash differs: {asset['asset_id']}")
        payloads[asset["asset_id"]] = data
    asset_dir = output / asset_dir_prefix
    expected = {_safe_asset_path(output, asset["path"]) for asset in assets}
    actual = ({path.resolve() for path in asset_dir.rglob("*")
               if path.is_file() and not path.name.startswith(".")}
              if asset_dir.exists() else set())
    unexpected = actual - expected
    if unexpected:
        raise VisualDataError(f"Unexpected asset files: {sorted(p.name for p in unexpected)}")
    return payloads


def _audit_stage(output: Path, stage: str, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(output / stage / "vision_manifest.json")
    rows = _load_jsonl(output / stage / "model_rows.jsonl")
    cases = _load_jsonl(output / stage / "verifier_cases.jsonl")
    scenes = _load_jsonl(output / stage / "verifier_scenes.jsonl")
    _require(manifest.get("format") == MANIFEST_FORMAT, f"Unknown {stage} manifest format")
    _require(manifest.get("stage") == stage, f"{stage} manifest stage differs")
    _require(manifest.get("config_fingerprint") == _config_fingerprint(config),
             f"{stage} manifest config differs from the supplied config")
    provenance = manifest.get("provenance", {})
    _require(provenance.get("origin") == "internal_owned_procedural_generation"
             and provenance.get("source_id") == SOURCE_ID
             and provenance.get("terms", {}).get("external_assets") == 0,
             f"{stage} provenance differs from owned internal generation")
    _require(manifest.get("training_admission", {}).get("admitted") is False
             and manifest.get("training_admission", {}).get("admitted_rows") == 0,
             f"{stage} must not admit training rows")
    assets = manifest.get("assets")
    _require(isinstance(assets, list) and bool(assets), f"{stage} manifest has no assets")
    expected_assets = (config["v0"]["root_families"] + config["v0"]["crop_families"]
                       if stage == "v0"
                       else config["v1"]["worlds"] * config["v1"]["assets_per_world"])
    _require(len(assets) == expected_assets, f"{stage} asset count differs from config")
    payloads = _audit_asset_files(output, assets, f"{stage}/assets")
    asset_by_id = {asset["asset_id"]: asset for asset in assets}
    _require(len(asset_by_id) == len(assets), f"{stage} asset IDs repeat")
    scene_by_asset = {scene["asset_id"]: scene for scene in scenes}
    _require(len(scene_by_asset) == len(scenes), f"{stage} scene records repeat")
    _require(set(scene_by_asset) == set(asset_by_id),
             f"{stage} verifier scenes do not cover assets")
    group_split: dict[str, str] = {}
    group_assets: dict[str, set[str]] = {}
    root_scene_hashes: dict[str, str] = {}
    for asset in assets:
        split = asset["split"]
        _require(split in SPLITS, f"Unknown split: {split}")
        group = asset["group_id"]
        if group in group_split:
            _require(group_split[group] == split, f"Family crosses splits: {group}")
        group_split[group] = split
        group_assets.setdefault(group, set()).add(asset["asset_id"])
        scene_record = scene_by_asset[asset["asset_id"]]
        _require(scene_record["scene_sha256"] == asset["scene_sha256"],
                 f"Scene hash differs for {asset['asset_id']}")
        _require(scene_sha256(scene_record["scene"]) == asset["scene_sha256"],
                 f"Scene truth differs for {asset['asset_id']}")
        parent_id = asset["lineage"]["parent_asset_id"]
        if parent_id is not None:
            parent = asset_by_id.get(parent_id)
            _require(parent is not None, f"Missing crop parent for {asset['asset_id']}")
            _require(parent["group_id"] == group and parent["split"] == split,
                     f"Crop crosses family or split: {asset['asset_id']}")
            crop_box = asset["lineage"]["crop_box"]
            _require(isinstance(crop_box, list) and len(crop_box) == 4,
                     f"Crop box missing for {asset['asset_id']}")
            parent_width, parent_height, parent_rows = decode_rgb_png(payloads[parent_id])
            x0, y0, x1, y1 = crop_box
            _require(0 <= x0 < x1 <= parent_width and 0 <= y0 < y1 <= parent_height,
                     f"Crop box outside parent for {asset['asset_id']}")
            crop_width, crop_height, crop_rows = decode_rgb_png(payloads[asset["asset_id"]])
            _require((crop_width, crop_height) == (x1 - x0, y1 - y0),
                     f"Crop dimensions differ for {asset['asset_id']}")
            for index, row in enumerate(crop_rows):
                _require(bytes(row) == bytes(parent_rows[y0 + index][x0 * 3:x1 * 3]),
                         f"Crop pixels differ from parent: {asset['asset_id']}")
        else:
            previous = root_scene_hashes.get(asset["scene_sha256"])
            _require(previous is None,
                     f"Duplicate root scene inputs: {asset['asset_id']} and {previous}")
            root_scene_hashes[asset["scene_sha256"]] = asset["asset_id"]
    declared_groups = manifest["groups"]
    group_field = GROUP_ID_FIELD[manifest["group_key"]]
    _require(len(declared_groups) == len(group_assets),
             f"{stage} group catalog differs from assets")
    for group in declared_groups:
        group_id = group[group_field]
        _require(group_assets.get(group_id) == set(group["asset_ids"]),
                 f"Group membership differs: {group_id}")
        _require(group_split[group_id] == group["split"],
                 f"Group split differs: {group_id}")
    _require(len({row["case_id"] for row in rows}) == len(rows),
             f"{stage} model row IDs repeat")
    _require(len({case["case_id"] for case in cases}) == len(cases),
             f"{stage} verifier case IDs repeat")
    _require({case["case_id"] for case in cases} == {row["case_id"] for row in rows},
             f"{stage} rows and verifier cases differ")
    case_by_id = {case["case_id"]: case for case in cases}
    duplicate_inputs: dict[tuple[str, str], str] = {}
    task_counts: Counter[str] = Counter()
    for row in rows:
        _require(set(row) == MODEL_ROW_FIELDS,
                 f"Model row fields differ: {sorted(set(row) ^ MODEL_ROW_FIELDS)}")
        case = case_by_id[row["case_id"]]
        asset = asset_by_id.get(row["asset_id"])
        _require(asset is not None,
                 f"Model row references an unknown asset: {row['asset_id']}")
        _require(row["image"] == asset["path"] and row["image_sha256"] == asset["sha256"],
                 f"Model row image differs from asset: {row['case_id']}")
        _require(row["question"] == case["question"],
                 f"Model row question differs from verifier case: {row['case_id']}")
        _require(case["asset_id"] == row["asset_id"],
                 f"Case asset differs: {row['case_id']}")
        _require(case["group_id"] == asset["group_id"] and case["split"] == asset["split"],
                 f"Case crosses family or split: {row['case_id']}")
        _require(case["scene_sha256"] == asset["scene_sha256"],
                 f"Case scene hash differs: {row['case_id']}")
        truth = scene_by_asset[asset["asset_id"]]["scene"]
        computed = _check_value(case["check"], truth)
        _require(computed == case["check"]["expected"],
                 f"Verifier check no longer executes for {row['case_id']}")
        _require(computed == case["answer"],
                 f"Verifier label differs from executable check: {row['case_id']}")
        if len(case["answer"]) > 1:
            _require(case["answer"].casefold() not in row["question"].casefold(),
                     f"Verifier label leaks into the question: {row['case_id']}")
        key = (row["image_sha256"], row["question"])
        _require(key not in duplicate_inputs,
                 f"Duplicate model inputs: {row['case_id']} and "
                 f"{duplicate_inputs.get(key)}")
        duplicate_inputs[key] = row["case_id"]
        task_counts[case["task"]] += 1
    expected_cases = (config["v0"]["cases_per_asset"] if stage == "v0"
                      else config["v1"]["cases_per_asset"]) * len(assets)
    _require(len(rows) == expected_cases, f"{stage} case count differs from config")
    _require(set(task_counts) == set(TASKS), f"{stage} does not cover every task type")
    split_assets = Counter(asset["split"] for asset in assets)
    split_groups = Counter(group["split"] for group in declared_groups)
    if stage == "v0":
        roles = _split_roles(config["v0"]["split_family_counts"])
        crop_set = set(range(config["v0"]["crop_families"]))
        expected_groups: Counter[str] = Counter(roles)
        expected_assets: Counter[str] = Counter()
        for index, split in enumerate(roles):
            expected_assets[split] += 1 + (index in crop_set)
        _require(split_groups == expected_groups,
                 "V0 split family counts differ from config")
        _require(split_assets == expected_assets,
                 "V0 split asset counts differ from config")
    else:
        expected_groups = Counter({"train": config["v1"]["worlds"] - config["v1"]["dev_worlds"],
                                   "dev": config["v1"]["dev_worlds"]})
        _require(split_groups == expected_groups,
                 "V1 split world counts differ from config")
        _require(split_assets == Counter({
            "train": (config["v1"]["worlds"] - config["v1"]["dev_worlds"])
            * config["v1"]["assets_per_world"],
            "dev": config["v1"]["dev_worlds"] * config["v1"]["assets_per_world"]}),
            "V1 split asset counts differ from config")
        _audit_v1_pairs(assets, scene_by_asset, case_by_id)
    _require(_generation_fingerprint(stage, config, {
        "assets": assets, "cases": cases, "scenes": scenes,
        "groups": declared_groups, "group_key": manifest["group_key"],
    }) == manifest.get("generation_fingerprint"),
        f"{stage} generation fingerprint differs")
    computed_counts = {
        "assets": len(assets), "groups": len(declared_groups),
        "model_rows": len(rows), "verifier_cases": len(cases),
        "verifier_scenes": len(scenes),
        "asset_bytes": sum(asset["bytes"] for asset in assets),
    }
    for key, value in computed_counts.items():
        _require(manifest["counts"][key] == value,
                 f"{stage} manifest count differs: {key}")
    return {
        "stage": stage, "assets": len(assets),
        "asset_bytes": computed_counts["asset_bytes"],
        "groups": len(declared_groups), "model_rows": len(rows),
        "verifier_cases": len(cases), "tasks": dict(sorted(task_counts.items())),
        "splits": {split: {"assets": split_assets[split],
                           "groups": split_groups[split]}
                   for split in SPLIT_ORDER if split_assets[split] or split_groups[split]},
        "generation_fingerprint": manifest["generation_fingerprint"],
        "admitted_rows": 0,
    }


def _audit_v1_pairs(assets: list[Mapping[str, Any]],
                    scene_by_asset: Mapping[str, Mapping[str, Any]],
                    case_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for asset in assets:
        if asset.get("pair_id") and asset["lineage"]["parent_asset_id"] is None:
            pairs.setdefault(asset["pair_id"], {})[asset["lineage"]["variant"]] = asset
    _require(len(pairs) == len(assets) // 4, "V1 pair catalog differs")
    for pair_id, variants in pairs.items():
        _require(set(variants) == {"a", "b"}, f"V1 pair lacks both scenes: {pair_id}")
        first, second = variants["a"], variants["b"]
        _require(first["group_id"] == second["group_id"]
                 and first["split"] == second["split"],
                 f"V1 pair crosses family or split: {pair_id}")
        change = first["decisive_attribute"]
        _require(change == second["decisive_attribute"],
                 f"V1 pair decisive attribute differs: {pair_id}")
        first_scene = scene_by_asset[first["asset_id"]]["scene"]
        second_scene = scene_by_asset[second["asset_id"]]["scene"]
        _require((first_scene["width"], first_scene["height"])
                 == (second_scene["width"], second_scene["height"]),
                 f"V1 pair dimensions differ: {pair_id}")
        _require(first_scene["layout"]["cell_indices"]
                 == second_scene["layout"]["cell_indices"],
                 f"V1 pair layout differs: {pair_id}")
        differences = []
        for left, right in zip(first_scene["objects"], second_scene["objects"],
                               strict=True):
            for key in ("id", "shape", "bbox", "center"):
                _require(left[key] == right[key], f"V1 pair {key} differs: {pair_id}")
            if left["color"] != right["color"]:
                differences.append((left["id"], left["color"], right["color"]))
        _require(len(differences) == 1,
                 f"V1 pair must differ in exactly one attribute: {pair_id}")
        object_id, old_color, new_color = differences[0]
        _require(object_id == change["object_id"] and old_color == change["from"]
                 and new_color == change["to"],
                 f"V1 pair decisive attribute differs from scenes: {pair_id}")
        first_cases = [case for case in case_by_id.values()
                       if case["asset_id"] == first["asset_id"]]
        second_cases = [case for case in case_by_id.values()
                        if case["asset_id"] == second["asset_id"]]
        same_questions = ({case["question"] for case in first_cases}
                          & {case["question"] for case in second_cases})
        _require(bool(same_questions), f"V1 pair lacks a shared question: {pair_id}")
        differing = {question for question in same_questions
                     if {case["answer"] for case in first_cases
                         if case["question"] == question}
                     != {case["answer"] for case in second_cases
                         if case["question"] == question}}
        _require(bool(differing), f"V1 pair answers do not differ: {pair_id}")


def _audit_temporal(output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(output / "temporal" / "temporal_manifest.json")
    public = _load_json(output / "temporal" / "temporal_worlds.json")
    verifier = _load_json(output / "temporal" / "verifier_worlds.json")
    _require(manifest.get("format") == TEMPORAL_MANIFEST_FORMAT,
             "Unknown temporal manifest format")
    _require(manifest.get("config_fingerprint") == _config_fingerprint(config),
             "Temporal manifest config differs")
    assets = manifest["assets"]
    expected_assets = config["temporal"]["worlds"] * 3
    _require(len(assets) == expected_assets, "Temporal asset count differs from config")
    payloads = _audit_asset_files(output, assets, "temporal/assets")
    del payloads
    asset_by_id = {asset["asset_id"]: asset for asset in assets}
    group_split: dict[str, str] = {}
    for asset in assets:
        if asset["group_id"] in group_split:
            _require(group_split[asset["group_id"]] == asset["split"],
                     "Temporal world crosses splits")
        group_split[asset["group_id"]] = asset["split"]
    _require(len(verifier["worlds"]) == config["temporal"]["worlds"],
             "Temporal verifier world count differs")
    expected_splits = Counter({"dev": config["temporal"]["dev_worlds"],
                               "audit": config["temporal"]["worlds"]
                               - config["temporal"]["dev_worlds"]})
    actual_splits = Counter(world["split"] for world in verifier["worlds"])
    _require(actual_splits == expected_splits,
             "Temporal split counts differ from config")
    _require(len(public["worlds"]) == len(verifier["worlds"]),
             "Temporal public/verifier world counts differ")
    for world_public, world_check in zip(public["worlds"], verifier["worlds"],
                                         strict=True):
        _require(world_public["world_id"] == world_check["world_id"],
                 "Temporal world order differs")
        _require(world_public["split"] == world_check["split"],
                 "Temporal world split differs")
        observations = world_public["observations"]
        _require(len(observations) == config["temporal"]["observations_per_world"],
                 "Temporal observation count differs")
        times = [observation["time"] for observation in observations]
        _require(times == sorted(times) and len(set(times)) == len(times),
                 "Temporal observations are not chronological")
        for observation in observations:
            _require(observation["label_visibility"] == "model_observation",
                     "Temporal observation visibility differs")
            asset = asset_by_id.get(observation["asset_id"])
            _require(asset is not None,
                     "Temporal observation references an unknown asset")
            _require(asset["sha256"] == observation["image_sha256"]
                     and asset["path"] == observation["image"],
                     "Temporal observation image differs from asset")
            time_key = f"t{observation['time']}"
            scene = world_check["scenes"][time_key]
            _require(scene_sha256(scene) == asset["scene_sha256"],
                     "Temporal verifier scene differs from asset")
            state = temporal_state_at(world_check["events"], observation["time"])
            objects = {obj["id"]: obj for obj in scene["objects"]}
            for fact, object_id in world_check["fact_object"].items():
                detail = state.get(fact)
                if detail is None or detail["value"] == "unknown":
                    continue
                _require(objects[object_id]["color"] == detail["value"],
                         f"Temporal image contradicts its ledger: {fact}")
        resolved = resolve_temporal_queries(world_check["events"])
        _require(len(world_check["queries"]) == config["temporal"]["queries_per_world"],
                 "Temporal query count differs")
        _require({query["fact"] for query in world_check["queries"]} == set(resolved),
                 "Temporal queries do not cover the ledger")
        for query in world_check["queries"]:
            expected = resolved[query["fact"]]
            _require(query["expected_answer"] == expected["value"],
                     f"Temporal answer differs from ledger: {query['query_id']}")
            _require(query["expected_status"] == expected["status"],
                     f"Temporal status differs from ledger: {query['query_id']}")
        for event in world_check["events"]:
            _require(event["fact"] in world_check["fact_object"],
                     "Temporal event references an unknown object fact")
    computed_generation = fingerprint({
        "format": TEMPORAL_MANIFEST_FORMAT,
        "config": manifest["config_fingerprint"],
        "seed": manifest["seed"],
        "asset_sha256": [asset["sha256"] for asset in assets],
        "worlds": [fingerprint({"world_id": world["world_id"],
                                "events": world["events"],
                                "queries": world["queries"],
                                "scenes": {time: scene_sha256(scene)
                                           for time, scene in world["scenes"].items()}})
                   for world in verifier["worlds"]],
    })
    _require(computed_generation == manifest.get("generation_fingerprint"),
             "Temporal generation fingerprint differs")
    _require(manifest["counts"]["assets"] == len(assets)
             and manifest["counts"]["asset_bytes"]
             == sum(asset["bytes"] for asset in assets),
             "Temporal manifest counts differ from files")
    return {
        "stage": "temporal", "assets": len(assets),
        "asset_bytes": sum(asset["bytes"] for asset in assets),
        "worlds": len(verifier["worlds"]),
        "observations": sum(len(world["observations"]) for world in public["worlds"]),
        "queries": sum(len(world["queries"]) for world in verifier["worlds"]),
        "splits": dict(sorted(actual_splits.items())),
        "generation_fingerprint": manifest["generation_fingerprint"],
        "admitted_rows": 0,
    }


def audit_visual_dataset(config: Mapping[str, Any], *, root: Path,
                         output: Path) -> dict[str, Any]:
    validate_config(config)
    root, output = Path(root), Path(output)
    stages = {stage: _audit_stage(output, stage, config) for stage in ("v0", "v1")}
    temporal = _audit_temporal(output, config)
    allowlist = build_external_allowlist(root / config["external"]["catalog"],
                                         root / config["external"]["runtime_registry"], root=root)
    stored_allowlist = _load_json(output / "external_component_allowlist.json")
    _require(_portable_allowlist(stored_allowlist, config["external"]) == allowlist,
             "Stored external allowlist differs from recomputation")
    _require(stored_allowlist["counts"]["assets_acquired"] == 0,
             "External assets were acquired by a preparation profile that forbids them")
    intake = _load_json(output / "intake_summary.json")
    _require(intake["format"] == "replica.vision_intake_summary.v1",
             "Unknown intake summary format")
    _require(intake["external_assets_acquired"] == 0,
             "Intake summary reports external asset acquisition")
    _require(intake["training_admission"]["admitted_rows"] == 0,
             "Intake summary reports admitted training rows")
    for stage in ("v0", "v1"):
        _require(intake["generated"][stage]["assets"] == stages[stage]["assets"]
                 and intake["generated"][stage]["model_rows"]
                 == stages[stage]["model_rows"],
                 f"Intake summary differs from {stage} artifacts")
    _require(intake["generated"]["temporal"]["assets"] == temporal["assets"],
             "Intake summary differs from temporal artifacts")
    seen_hashes: dict[str, str] = {}
    for stage in ("v0", "v1"):
        manifest = _load_json(output / stage / "vision_manifest.json")
        for asset in manifest["assets"]:
            previous = seen_hashes.get(asset["sha256"])
            _require(previous is None,
                     f"Duplicate asset hash across stages: {asset['asset_id']} and "
                     f"{previous}")
            seen_hashes[asset["sha256"]] = asset["asset_id"]
    for asset in _load_json(output / "temporal" / "temporal_manifest.json")["assets"]:
        previous = seen_hashes.get(asset["sha256"])
        _require(previous is None,
                 f"Duplicate asset hash across stages: {asset['asset_id']} and "
                 f"{previous}")
        seen_hashes[asset["sha256"]] = asset["asset_id"]
    asset_bytes = sum(stage["asset_bytes"] for stage in stages.values()) + temporal["asset_bytes"]
    _require(asset_bytes <= config["campaign"]["asset_ceiling_bytes"],
             "Generated assets exceed the campaign asset ceiling")
    return {
        "status": "passed", "stages": stages, "temporal": temporal,
        "asset_bytes": asset_bytes,
        "campaign_asset_ceiling_bytes": config["campaign"]["asset_ceiling_bytes"],
        "within_asset_ceiling": True,
        "external": allowlist["counts"],
        "external_assets_acquired": 0,
        "admitted_training_rows": 0,
    }


# ---------------------------------------------------------------------------
# Public generation entry point.
# ---------------------------------------------------------------------------

def generate_visual_dataset(config: Mapping[str, Any], *, root: Path,
                            output: Path) -> dict[str, Any]:
    """Generate immutable owned assets/manifests, then audit them once."""
    validate_config(config)
    root, output = Path(root), Path(output)
    config_fp = _config_fingerprint(config)
    allowlist = build_external_allowlist(root / config["external"]["catalog"],
                                         root / config["external"]["runtime_registry"], root=root)
    allowlist_path = output / "external_component_allowlist.json"
    if allowlist_path.exists():
        _require(_portable_allowlist(_load_json(allowlist_path), config["external"]) == allowlist,
                 "Stored external allowlist differs from recomputation")
    else:
        write_exact(allowlist_path, _json_bytes(allowlist))
    built_stages = {"v0": _construct_v0(config), "v1": _construct_v1(config)}
    stage_summaries = {}
    for stage, built in built_stages.items():
        for asset, png in zip(built["assets"], built["asset_bytes"], strict=True):
            write_exact(output / asset["path"], png)
        manifest = _manifest(stage, config, built)
        write_exact(output / stage / "vision_manifest.json", _json_bytes(manifest))
        write_exact(output / stage / "model_rows.jsonl", _jsonl_bytes(built["rows"]))
        write_exact(output / stage / "verifier_cases.jsonl", _jsonl_bytes(built["cases"]))
        write_exact(output / stage / "verifier_scenes.jsonl", _jsonl_bytes(built["scenes"]))
        stage_summaries[stage] = {
            "status": config[stage]["status"],
            "assets": len(built["assets"]),
            "root_assets": sum(asset["lineage"]["parent_asset_id"] is None
                               for asset in built["assets"]),
            "crop_assets": sum(asset["lineage"]["parent_asset_id"] is not None
                               for asset in built["assets"]),
            "groups": len(built["groups"]),
            "model_rows": len(built["rows"]),
            "verifier_cases": len(built["cases"]),
            "asset_bytes": sum(asset["bytes"] for asset in built["assets"]),
        }
    temporal = _construct_temporal(config)
    for asset, png in zip(temporal["assets"], temporal["asset_bytes"], strict=True):
        write_exact(output / asset["path"], png)
    temporal_manifest = _temporal_manifest(config, temporal)
    write_exact(output / "temporal" / "temporal_manifest.json",
                _json_bytes(temporal_manifest))
    write_exact(output / "temporal" / "temporal_worlds.json", _json_bytes({
        "format": "replica.temporal_worlds.v1",
        "label_visibility": "observations_only",
        "worlds": [{"world_id": group["world_id"], "split": group["split"],
                    "observations": group["observations"]}
                   for group in temporal["groups"]],
    }))
    write_exact(output / "temporal" / "verifier_worlds.json", _json_bytes({
        "format": "replica.temporal_verifier_worlds.v1",
        "label_visibility": "verifier_only",
        "worlds": temporal["verifier_worlds"],
    }))
    temporal_summary = {
        "status": config["temporal"]["status"],
        "assets": len(temporal["assets"]),
        "groups": len(temporal["groups"]),
        "observations": sum(len(group["observations"]) for group in temporal["groups"]),
        "queries": sum(len(world["queries"]) for world in temporal["verifier_worlds"]),
        "asset_bytes": sum(asset["bytes"] for asset in temporal["assets"]),
        "model_rows": 0,
        "verifier_cases": sum(len(world["queries"])
                              for world in temporal["verifier_worlds"]),
    }
    intake = {
        "format": "replica.vision_intake_summary.v1",
        "seed": config["seed"],
        "config_fingerprint": config_fp,
        "external": allowlist["counts"],
        "external_assets_acquired": 0,
        "internal_generated_assets": (stage_summaries["v0"]["assets"]
                                      + stage_summaries["v1"]["assets"]
                                      + temporal_summary["assets"]),
        "generated": {**stage_summaries, "temporal": temporal_summary},
        "training_admission": {
            "admitted_rows": 0,
            "candidate_rows": (stage_summaries["v0"]["model_rows"]
                               + stage_summaries["v1"]["model_rows"]),
        },
        "campaign": {"asset_ceiling_bytes": config["campaign"]["asset_ceiling_bytes"],
                     "checkpoint_bytes_separate": True,
                     "ram_guard_separate": True},
    }
    write_exact(output / "intake_summary.json", _json_bytes(intake))
    audit = audit_visual_dataset(config, root=root, output=output)
    return {"stages": stage_summaries, "temporal": temporal_summary,
            "audit": audit, "allowlist": allowlist}
