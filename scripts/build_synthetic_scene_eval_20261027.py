#!/usr/bin/env python3
"""Render and seal a fresh, project-owned synthetic scene evaluation partition.

Images and captions are produced from one authoritative scene specification using
only the Python standard library.  The scene specification describes geometry,
colors and counts; the caption is a deterministic template expansion of the same
parameters.  Agreement is therefore checkable without a model: pixels must match
analytic geometry and the caption must match the specification.

The builder also checks disjointness against the nine-row corrected training
lineage, all thirty external tranche rows and the eight quarantined exposed rows
before it writes the seal.  A seal is written once; changing it requires the
explicit ``--reseal`` flag (a new evaluation identity is expected in normal use).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import canonical_json  # noqa: E402
from replica_lab.external_intake import normalize_caption  # noqa: E402

CONFIG_FORMAT = "replica.synthetic_scene_eval_contract.v1"
SPECS_FORMAT = "replica.synthetic_scene_specs.v1"
SPEC_FORMAT = "replica.synthetic_scene_spec.v1"
ROW_FORMAT = "replica.synthetic_scene_eval_row.v1"
MANIFEST_FORMAT = "replica.synthetic_scene_eval_manifest.v1"
SEAL_FORMAT = "replica.synthetic_scene_eval_seal.v1"
ROLE = "final_eval_untouched_synthetic"
SPLIT = "synthetic_scene_eval_sealed"


class BuildError(ValueError):
    """The declared synthetic evaluation contract cannot be satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Deterministic PNG writer


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def encode_png(pixels: Sequence[Sequence[Sequence[int]]], width: int, height: int) -> bytes:
    _require(len(pixels) == height and all(len(row) == width for row in pixels),
             "pixel matrix must be height x width")
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend((int(r), int(g), int(b)))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _png_chunk(b"IEND", b""))


def decode_png(data: bytes) -> dict[str, Any]:
    """Minimal strict PNG decoder used by the independent validator."""
    _require(data.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG file")
    offset = 8
    width = height = None
    idat = bytearray()
    saw_end = False
    while offset < len(data):
        _require(offset + 8 <= len(data), "truncated PNG chunk header")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        _require(len(payload) == length, "truncated PNG chunk payload")
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        _require(crc == (zlib.crc32(kind + payload) & 0xFFFFFFFF),
                 f"PNG CRC mismatch in {kind!r}")
        offset += 12 + length
        if kind == b"IHDR":
            _require(length == 13, "bad IHDR length")
            width, height, depth, color_type, compression, filtering, interlace = \
                struct.unpack(">IIBBBBB", payload)
            _require(depth == 8 and color_type == 2 and compression == 0
                     and filtering == 0 and interlace == 0,
                     "only 8-bit RGB non-interlaced PNG is admitted")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
        elif kind not in (b"PLTE", b"tRNS"):
            _require(kind in (b"IHDR", b"IDAT", b"IEND"), "unexpected chunk")
    _require(saw_end and width and height is not None, "PNG is incomplete")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    _require(len(raw) == (stride + 1) * height, "decompressed PNG size mismatch")
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        _require(raw[start] == 0, "only filter type 0 is admitted")
        line = raw[start + 1:start + 1 + stride]
        rows.append([[line[x * 3], line[x * 3 + 1], line[x * 3 + 2]]
                     for x in range(width)])
    return {"width": width, "height": height, "pixels": rows}


# ---------------------------------------------------------------------------
# Scene specification and rasterization


def _palette(config: Mapping[str, Any]) -> dict[str, list[int]]:
    palette = config["generation"]["palette"]
    for name, rgb in palette.items():
        _require(isinstance(rgb, list) and len(rgb) == 3
                 and all(isinstance(v, int) and 0 <= v <= 255 for v in rgb),
                 f"invalid palette entry {name}")
    return palette


def _object(shape: str, color_name: str, palette: Mapping[str, Sequence[int]],
            center: Sequence[int], *, radius: int | None = None,
            half: int | None = None) -> dict[str, Any]:
    cx, cy = int(center[0]), int(center[1])
    item: dict[str, Any] = {
        "shape": shape,
        "color_name": color_name,
        "rgb": list(palette[color_name]),
        "center": [cx, cy],
    }
    if shape in ("circle", "triangle"):
        _require(radius is not None, f"{shape} requires a radius")
        item["radius"] = int(radius)
        if shape == "triangle":
            item["vertices"] = [[cx, cy - int(radius)],
                                [cx - int(radius), cy + int(radius) - 2],
                                [cx + int(radius), cy + int(radius) - 2]]
            item["probe"] = [cx, cy + 2]
        else:
            item["probe"] = [cx, cy]
    elif shape == "square":
        _require(half is not None, "square requires a half size")
        item["half"] = int(half)
        item["probe"] = [cx, cy]
    else:
        raise BuildError(f"unsupported shape {shape!r}")
    return item


def rasterize_scene(scene: Mapping[str, Any]) -> list[list[list[int]]]:
    width, height = scene["width"], scene["height"]
    background = list(scene["background"]["rgb"])
    pixels = [[list(background) for _ in range(width)] for _ in range(height)]
    for item in scene["objects"]:
        rgb = list(item["rgb"])
        if item["shape"] == "circle":
            cx, cy = item["center"]
            radius = item["radius"]
            for y in range(height):
                for x in range(width):
                    dx, dy = x + 0.5 - cx, y + 0.5 - cy
                    if dx * dx + dy * dy <= radius * radius:
                        pixels[y][x] = list(rgb)
        elif item["shape"] == "square":
            cx, cy = item["center"]
            half = item["half"]
            for y in range(height):
                for x in range(width):
                    if abs(x + 0.5 - cx) <= half and abs(y + 0.5 - cy) <= half:
                        pixels[y][x] = list(rgb)
        elif item["shape"] == "triangle":
            (x1, y1), (x2, y2), (x3, y3) = item["vertices"]

            def edge(ax: float, ay: float, bx: float, by: float,
                     px: float, py: float) -> float:
                return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

            for y in range(height):
                for x in range(width):
                    px, py = x + 0.5, y + 0.5
                    e1 = edge(x1, y1, x2, y2, px, py)
                    e2 = edge(x2, y2, x3, y3, px, py)
                    e3 = edge(x3, y3, x1, y1, px, py)
                    if (e1 >= 0 and e2 >= 0 and e3 >= 0) or \
                       (e1 <= 0 and e2 <= 0 and e3 <= 0):
                        pixels[y][x] = list(rgb)
        else:
            raise BuildError(f"unsupported shape {item['shape']!r}")
    return pixels


def derive_caption(scene: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    template = config["generation"]["templates"][scene["template_id"]]
    params = dict(scene["caption_params"])
    if "count" in params:
        params["plural"] = "" if int(params["count"]) == 1 else "s"
    try:
        return template.format(**params)
    except KeyError as error:
        raise BuildError(f"caption template is missing {error}") from error


def derive_probes(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = []
    background = list(scene["background"]["rgb"])
    for point in ([2, 2], [125, 2], [2, 125]):
        probes.append({"kind": "background", "point": point,
                       "expected_rgb": background})
    for index, item in enumerate(scene["objects"]):
        probes.append({"kind": "object", "object_index": index,
                       "point": list(item["probe"]),
                       "expected_rgb": list(item["rgb"]),
                       "shape": item["shape"],
                       "color_name": item["color_name"]})
    return probes


def _family_scenes(family: Mapping[str, Any], palette: Mapping[str, Sequence[int]],
                   config: Mapping[str, Any]) -> list[dict[str, Any]]:
    generation = config["generation"]
    scenes = []
    family_id = family["id"]
    if family_id == "counted_circles":
        for index, count in enumerate(family["counts"]):
            centers = [64 - (count - 1) * family["spacing"] // 2
                       + position * family["spacing"] for position in range(count)]
            objects = [_object("circle", family["color"], palette,
                               [cx, family["y"]], radius=family["radius"])
                       for cx in centers]
            scenes.append({
                "scene_id": f"sce_{family_id}_{index:02d}",
                "family": family_id,
                "template_id": family["template"],
                "caption_params": {"count": count, "color": family["color"],
                                   "shape": "circle", "background": family["background"]},
                "caption": "",
                "objects": objects,
                "background": {"color_name": family["background"],
                               "rgb": list(palette[family["background"]])},
            })
    elif family_id == "positioned_square":
        for index, position in enumerate(family["positions"]):
            objects = [_object("square", family["color"], palette,
                               position["center"], half=family["half"])]
            scenes.append({
                "scene_id": f"sce_{family_id}_{index:02d}",
                "family": family_id,
                "template_id": family["template"],
                "caption_params": {"color": family["color"], "shape": "square",
                                   "position": position["phrase"],
                                   "background": family["background"]},
                "caption": "",
                "objects": objects,
                "background": {"color_name": family["background"],
                               "rgb": list(palette[family["background"]])},
            })
    elif family_id == "shape_pair":
        for index, variant in enumerate(family["variants"]):
            objects = []
            for label, x in (("left", family["left_x"]), ("right", family["right_x"])):
                shape = variant[label]["shape"]
                color = variant[label]["color"]
                if shape == "triangle":
                    objects.append(_object(shape, color, palette, [x, family["y"]],
                                           radius=family["triangle_radius"]))
                elif shape == "circle":
                    objects.append(_object(shape, color, palette, [x, family["y"]],
                                           radius=family["circle_radius"]))
                elif shape == "square":
                    objects.append(_object(shape, color, palette, [x, family["y"]],
                                           half=family["square_half"]))
                else:
                    raise BuildError(f"unsupported pair shape {shape!r}")
            scenes.append({
                "scene_id": f"sce_{family_id}_{index:02d}",
                "family": family_id,
                "template_id": family["template"],
                "caption_params": {
                    "left_color": variant["left"]["color"],
                    "left_shape": variant["left"]["shape"],
                    "right_color": variant["right"]["color"],
                    "right_shape": variant["right"]["shape"],
                    "background": variant["background"],
                },
                "caption": "",
                "objects": objects,
                "background": {"color_name": variant["background"],
                               "rgb": list(palette[variant["background"]])},
            })
    elif family_id == "row_squares":
        for index, count in enumerate(family["counts"]):
            centers = [64 - (count - 1) * family["spacing"] // 2
                       + position * family["spacing"] for position in range(count)]
            objects = [_object("square", family["color"], palette,
                               [cx, family["y"]], half=family["half"])
                       for cx in centers]
            scenes.append({
                "scene_id": f"sce_{family_id}_{index:02d}",
                "family": family_id,
                "template_id": family["template"],
                "caption_params": {"count": count, "color": family["color"],
                                   "shape": "square", "background": family["background"]},
                "caption": "",
                "objects": objects,
                "background": {"color_name": family["background"],
                               "rgb": list(palette[family["background"]])},
            })
    else:
        raise BuildError(f"unsupported family {family_id!r}")

    for scene in scenes:
        scene.update({
            "format": SPEC_FORMAT,
            "eval_id": config["eval_id"],
            "seed": generation["seed"],
            "width": generation["width"],
            "height": generation["height"],
            "renderer_version": generation["renderer_version"],
            "row_id": f"{config['eval_id']}:{scene['scene_id']}",
        })
        scene["caption"] = derive_caption(scene, config)
        scene["probes"] = derive_probes(scene)
        scene["scene_spec_sha256"] = sha256_text(canonical_json(scene))
    return scenes


# ---------------------------------------------------------------------------
# Disjointness


def _external_rows(corpus_path: Path) -> list[dict[str, Any]]:
    records = []
    for row in load_rows(corpus_path):
        records.append({
            "row_id": row["row_id"],
            "group_key": row["split"]["group_key"],
            "image_sha256": row["root_asset"]["image_sha256"],
            "target_sha256": row["target"]["caption"],
            "caption": row["target"]["caption"],
        })
    return records


def disjointness_report(config: Mapping[str, Any], eval_rows: Sequence[Mapping[str, Any]],
                        *, project_root: Path = ROOT) -> dict[str, Any]:
    joins = config["disjointness"]
    lineage_config_path = project_root / joins["caption_lineage_config"]["path"]
    _require(sha256_file(lineage_config_path)
             == joins["caption_lineage_config"]["sha256"],
             "corrected lineage config pin mismatch")
    lineage_path = project_root / joins["caption_lineage_manifest"]["path"]
    _require(lineage_path.is_file(),
             "the corrected training lineage must be built before the eval seal")
    lineage = load_json(lineage_path)
    _require(lineage["manifest_sha256"]
             == sha256_text(canonical_json(
                 {key: value for key, value in lineage.items()
                  if key != "manifest_sha256"})),
             "corrected lineage manifest self-hash mismatch")
    exposure_path = project_root / joins["exposure_incident"]["path"]
    _require(sha256_file(exposure_path) == joins["exposure_incident"]["sha256"],
             "exposure pin mismatch")
    exposure = load_json(exposure_path)
    external_path = project_root / joins["external_corpus"]["path"]
    _require(sha256_file(external_path) == joins["external_corpus"]["sha256"],
             "external corpus pin mismatch")
    external = _external_rows(external_path)

    known: dict[str, str] = {}
    for row in lineage["rows"]:
        known[row["row_id"]] = "corrected_training"
        known[row["parent_row_id"]] = "corrected_training_parent"
        known[row["group_key"]] = "corrected_training_group"
        known[row["image_sha256"]] = "corrected_training_image"
        known[row["target_sha256"]] = "corrected_training_target"
        known[row["normalized_caption_sha256"]] = "corrected_training_caption"
    for row in external:
        known[row["row_id"]] = "external_tranche"
        known[row["group_key"]] = "external_tranche_group"
        known[row["image_sha256"]] = "external_tranche_image"
        known[row["target_sha256"]] = "external_tranche_target"
        known[sha256_text(normalize_caption(row["caption"]))] = "external_tranche_caption"
    for row_id in exposure["exposed_row_ids"]:
        known[row_id] = "exposed"

    collisions = []
    for row in eval_rows:
        for value, kind in (
            (row["row_id"], "row_id"),
            (row["group_key"], "group_key"),
            (row["asset"]["image_sha256"], "image_sha256"),
            (row["target"]["caption_sha256"], "target_sha256"),
            (row["provenance"]["normalized_caption_sha256"], "normalized_caption_sha256"),
        ):
            if value in known:
                collisions.append({"key": kind, "value": value, "conflict": known[value]})
    _require(not collisions, f"synthetic eval collides with existing rows: {collisions}")
    _require(not set(exposure["exposed_row_ids"])
             & {row["row_id"] for row in eval_rows},
             "an exposed row appears in the synthetic eval")
    return {
        "training_lineage_id": lineage["lineage_id"],
        "training_lineage_manifest_sha256": lineage["manifest_sha256"],
        "training_rows_sha256": lineage["rows_sha256"],
        "exposure_incident_sha256": joins["exposure_incident"]["sha256"],
        "exposed_row_ids": list(exposure["exposed_row_ids"]),
        "external_corpus_sha256": joins["external_corpus"]["sha256"],
        "compared_external_rows": len(external),
        "compared_training_rows": lineage["counts"]["rows"],
        "compared_keys": config["disjointness"]["required_disjoint_keys"],
        "collisions": [],
    }


# ---------------------------------------------------------------------------
# Build


def build(config: Mapping[str, Any], *, project_root: Path = ROOT) -> dict[str, Any]:
    _require(config.get("format") == CONFIG_FORMAT, "unexpected eval config format")
    _require(config.get("status") == "sealed_preparation_only_no_model_use",
             "unexpected eval status")
    generation = config["generation"]
    palette = _palette(config)
    scenes: list[dict[str, Any]] = []
    for family in generation["families"]:
        scenes.extend(_family_scenes(family, palette, config))
    expected = config["expected"]
    _require(len(scenes) == expected["rows"], "scene count differs from expected rows")
    _require(len({scene["scene_id"] for scene in scenes}) == len(scenes),
             "scene ids are not unique")
    _require(len({scene["caption"] for scene in scenes}) == len(scenes),
             "captions are not unique")
    _require({scene["family"] for scene in scenes}
             == {family["id"] for family in generation["families"]},
             "family coverage mismatch")

    spec_file = {
        "format": SPECS_FORMAT,
        "eval_id": config["eval_id"],
        "created_on": config["created_on"],
        "renderer_version": generation["renderer_version"],
        "width": generation["width"],
        "height": generation["height"],
        "palette": palette,
        "scenes": scenes,
    }
    specs_text = json.dumps(spec_file, indent=2, sort_keys=True) + "\n"
    specs_sha = sha256_text(specs_text)

    rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for scene in scenes:
        pixels = rasterize_scene(scene)
        image_bytes = encode_png(pixels, scene["width"], scene["height"])
        image_sha = sha256_bytes(image_bytes)
        caption = derive_caption(scene, config)
        _require(caption == scene["caption"], "caption derivation is not stable")
        _require(derive_probes(scene) == scene["probes"], "probe derivation is not stable")
        row = {
            "format": ROW_FORMAT,
            "eval_id": config["eval_id"],
            "row_id": scene["row_id"],
            "group_key": scene["row_id"],
            "scene_id": scene["scene_id"],
            "family": scene["family"],
            "split": SPLIT,
            "role": ROLE,
            "observation": {
                "image_ref": config["artifacts"]["images"]
                          + f"/{scene['scene_id']}{generation['image_extension']}",
                "image_sha256": image_sha,
                "prompt": generation["prompt"],
                "task": generation["task"],
            },
            "target": {
                "caption": caption,
                "caption_sha256": sha256_text(caption),
            },
            "asset": {
                "image_sha256": image_sha,
                "image_bytes": len(image_bytes),
                "image_format": "PNG",
                "width": scene["width"],
                "height": scene["height"],
                "copied": False,
            },
            "label_authority": {
                "kind": config["label_authority"]["kind"],
                "scene_spec_sha256": scene["scene_spec_sha256"],
                "template_id": scene["template_id"],
                "caption_params": scene["caption_params"],
                "agreement_checks": config["label_authority"]["agreement_checks"],
                "human_semantic_review": False,
                "model_inference": False,
            },
            "probes": derive_probes(scene),
            "provenance": {
                "kind": config["provenance"]["kind"],
                "owner": config["provenance"]["owner"],
                "generator": config["provenance"]["generator"],
                "renderer_version": scene["renderer_version"],
                "seed": scene["seed"],
                "upstream_dataset": None,
                "external_images": False,
                "third_party_content": False,
                "normalized_caption_sha256":
                    sha256_text(normalize_caption(caption)),
            },
            "rights": dict(config["rights"]),
        }
        row["row_sha256"] = sha256_text(canonical_json(row))
        rows.append(row)
        images.append({
            "scene_id": scene["scene_id"],
            "path": config["artifacts"]["images"]
                    + f"/{scene['scene_id']}{generation['image_extension']}",
            "sha256": image_sha,
            "bytes": len(image_bytes),
            "width": scene["width"],
            "height": scene["height"],
            "format": "PNG",
        })

    _require(len({image["sha256"] for image in images}) == len(images),
             "image hashes are not unique")
    rows_text = "".join(canonical_json(row) + "\n" for row in rows)
    rows_sha = sha256_text(rows_text)
    disjoint = disjointness_report(config, rows, project_root=project_root)

    manifest = {
        "format": MANIFEST_FORMAT,
        "eval_id": config["eval_id"],
        "created_on": config["created_on"],
        "status": "sealed_preparation_only_no_model_use",
        "generator": config["provenance"]["generator"],
        "renderer_version": generation["renderer_version"],
        "counts": {
            "rows": len(rows),
            "images": len(images),
            "families": len(generation["families"]),
            "scenes_per_family": expected["scenes_per_family"],
        },
        "split": SPLIT,
        "role": ROLE,
        "scene_specs_sha256": specs_sha,
        "rows_sha256": rows_sha,
        "rows": [{
            "row_id": row["row_id"],
            "group_key": row["group_key"],
            "scene_id": row["scene_id"],
            "family": row["family"],
            "row_sha256": row["row_sha256"],
            "image_sha256": row["asset"]["image_sha256"],
            "target_sha256": row["target"]["caption_sha256"],
            "scene_spec_sha256": row["label_authority"]["scene_spec_sha256"],
            "normalized_caption_sha256": row["provenance"]["normalized_caption_sha256"],
        } for row in rows],
        "images": images,
        "label_authority": config["label_authority"],
        "provenance": config["provenance"],
        "rights": config["rights"],
        "benchmark_limits": config["benchmark_limits"],
        "disjointness": disjoint,
        "seal_required_before_model_use": True,
        "model_use": "none",
        "non_claims": config["non_claims"],
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha = sha256_text(manifest_text)

    seal = {
        "format": SEAL_FORMAT,
        "eval_id": config["eval_id"],
        "created_on": config["created_on"],
        "role": ROLE,
        "split": SPLIT,
        "sealed_artifacts": {
            "scene_specs_sha256": specs_sha,
            "rows_sha256": rows_sha,
            "manifest_sha256": manifest_sha,
            "images_aggregate_sha256": None,  # filled after image write
            "counts": manifest["counts"],
        },
        "disjointness": disjoint,
        "contract_canonical_sha256": sha256_text(canonical_json(config)),
        "attestations": {
            "sealed_before_any_model_use": True,
            "model_executions_at_seal": 0,
            "human_semantic_review_used": False,
            "labels_from_renderer_scene_spec": True,
            "no_external_images": True,
            "no_third_party_content": True,
            "disjoint_from_training_and_exposed_rows": True,
            "generator_is_deterministic": True,
        },
        "seal_digest_rule": config["seal"]["seal_digest_rule"],
    }
    return {
        "specs_text": specs_text,
        "rows": rows,
        "rows_text": rows_text,
        "images": images,
        "image_bytes": {
            row["scene_id"]: encode_png(rasterize_scene(scene), scene["width"],
                                        scene["height"])
            for scene, row in zip(scenes, rows)
        },
        "manifest": manifest,
        "manifest_text": manifest_text,
        "specs_sha256": specs_sha,
        "rows_sha256": rows_sha,
        "manifest_sha256": manifest_sha,
        "seal": seal,
    }


def seal_digest(seal: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(
        {key: value for key, value in seal.items() if key != "seal_sha256"}))


def _finish_seal(built: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    images = sorted(built["images"], key=lambda item: item["path"])
    aggregate = hashlib.sha256()
    for image in images:
        filename = image["path"].rsplit("/", 1)[-1]
        aggregate.update(filename.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(built["image_bytes"][image["scene_id"]])
    aggregate_sha = aggregate.hexdigest()
    seal = built["seal"]
    seal["sealed_artifacts"]["images_aggregate_sha256"] = aggregate_sha
    seal["seal_sha256"] = seal_digest(seal)
    built["seal"] = seal
    built["seal_text"] = json.dumps(seal, indent=2, sort_keys=True) + "\n"
    built["images_aggregate_sha256"] = aggregate_sha
    return seal


def write_outputs(built: dict[str, Any], config: Mapping[str, Any], *,
                  project_root: Path = ROOT, reseal: bool = False) -> dict[str, Any]:
    artifacts = config["artifacts"]
    image_dir = project_root / artifacts["images"]
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in built["images"]:
        target = project_root / image["path"]
        data = built["image_bytes"][image["scene_id"]]
        if target.exists() and target.read_bytes() != data and not reseal:
            raise BuildError(f"refusing to change sealed image {image['path']}")
        target.write_bytes(data)
    seal_path = project_root / artifacts["seal"]
    if seal_path.exists() and not reseal:
        raise BuildError("a seal already exists; use --reseal only for a new "
                         "intentional evaluation identity")
    _finish_seal(built, config)
    plan = {
        artifacts["scene_specs"]: built["specs_text"],
        artifacts["rows"]: built["rows_text"],
        artifacts["manifest"]: built["manifest_text"],
        artifacts["seal"]: built["seal_text"],
    }
    for relative, text in plan.items():
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") != text and not reseal:
            raise BuildError(f"refusing to overwrite sealed artifact {relative}")
        target.write_text(text, encoding="utf-8")
    return {
        "rows": len(built["rows"]),
        "images": len(built["images"]),
        "scene_specs_sha256": built["specs_sha256"],
        "rows_sha256": built["rows_sha256"],
        "manifest_sha256": built["manifest_sha256"],
        "images_aggregate_sha256": built["images_aggregate_sha256"],
        "seal_sha256": built["seal"]["seal_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(
        ROOT / "configs/synthetic_scene_eval_20261027.json"))
    parser.add_argument("--reseal", action="store_true",
                        help="replace an existing seal (new evaluation identity only)")
    args = parser.parse_args(argv)
    config = load_json(args.config)
    try:
        built = build(config)
        summary = write_outputs(built, config, reseal=args.reseal)
    except BuildError as error:
        print(json.dumps({"status": "refused", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "built", **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
