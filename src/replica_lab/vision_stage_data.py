"""Fresh stage-wise vision data and protocol construction (CPU only)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import fingerprint
from .vision_curriculum import (
    SPLIT_TASKS, _build_training_examples, _diagnostic_rows,
    _external_label_rows, _load_procedural_source, _procedural_rows,
    _write_exact, build_procedural, load_external_source, sha256_bytes,
    sha256_file,
)

CONFIG_FORMAT = "replica.vision_stage_diagnostic.config.v1"
PROTOCOL_FORMAT = "replica.vision_stage_diagnostic.protocol.v1"
MANIFEST_FORMAT = "replica.vision_stage_diagnostic.manifest.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode()


def load_stage_config(path):
    value = json.loads(Path(path).read_text())
    _require(value.get("format") == CONFIG_FORMAT,
             f"Stage config format must be {CONFIG_FORMAT}")
    for key in ("seed", "data_root", "parent_data_config", "parent_protocol",
                "procedural_overrides", "train", "schedule", "evaluation",
                "checkpoint_policy", "implementation_paths", "project_root"):
        _require(key in value, f"Stage config is missing {key}")
    return value


def scene_caption(scene: Mapping[str, Any]) -> str:
    """Describe only visible scene objects; never infer an occluded object."""
    objects = [obj for obj in scene["objects"] if obj.get("role") != "occluded"]
    _require(objects, "Caption requires at least one visible object")
    parts = [f"{obj['color']} {obj['shape']}" for obj in objects]
    if len(parts) == 1:
        return f"a {parts[0]}"
    if len(parts) == 2:
        return f"a {parts[0]} and a {parts[1]}"
    return "a " + ", a ".join(parts[:-1]) + f" and a {parts[-1]}"


def select_stage_cases(cases: Sequence[Mapping[str, Any]], total: int) -> list[dict[str, Any]]:
    """Deterministic per-task selection with every declared task represented."""
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        by_task.setdefault(case["task"], []).append(case)
    _require(set(by_task) == set(SPLIT_TASKS["dev"]),
             "Stage evaluation task inventory differs")
    per_task = math.ceil(total / len(by_task))
    selected = []
    for task in SPLIT_TASKS["dev"]:
        selected.extend(dict(row) for row in by_task[task][:per_task])
    selected = selected[:total]
    _require(len(selected) == total, "Insufficient stage evaluation cases")
    _require(any(row["label"] == "unknown" for row in selected),
             "Stage evaluation lacks unknown cases")
    return selected


def _joined_cases(manifest, split, source, total):
    rows, labels = _procedural_rows(manifest, split)
    label_by_id = {row["case_id"]: row for row in labels}
    case_by_id = {case["case_id"]: case for case in manifest["cases"]}
    joined = []
    for row in rows:
        label = label_by_id[row["case_id"]]
        joined.append({
            "case_id": row["case_id"], "source": source,
            "image": row["image"], "image_sha256": row["image_sha256"],
            "question": row["question"], "task": label["task"],
            "kind": label.get("kind"), "world_id": label["world_id"],
            "label": label["answer"], "scene_sha256": case_by_id[row["case_id"]]["scene_sha256"],
        })
    return select_stage_cases(joined, total)


def _label_rows(cases, split):
    return [{"case_id": row["case_id"], "split": split, "answer": row["label"],
             "task": row["task"], "kind": row.get("kind"),
             "world_id": row["world_id"], "scene_sha256": row["scene_sha256"],
             "label_visibility": "verifier_only"} for row in cases]


def build_stage_dataset(config: Mapping[str, Any], project_root: str | Path, *,
                        cases_per_split: int = 24):
    project_root = Path(project_root)
    external = load_external_source(project_root, config)
    procedural = build_procedural(config)
    diagnostic = _load_procedural_source(project_root, config)
    manifest = procedural["manifest"]
    files: dict[str, bytes] = {}

    external_train_rows, external_train_labels = _external_label_rows(
        external["train"], "train", "external/train")
    for row in external["train"]:
        files[f"external/train/images/{row['upstream_id']}.jpg"] = row["image_bytes"]
    diag_train_rows, diag_train_labels, diag_files = _diagnostic_rows(
        diagnostic["train"], "train", "diagnostic/train")
    files.update(diag_files)
    for relative, payload in procedural["files"].items():
        files[f"procedural/{relative}"] = payload

    curriculum_train_rows, curriculum_train_labels = _procedural_rows(manifest, "train")
    curriculum_train = [{"case_id": row["case_id"], "image": row["image"],
                         "image_sha256": row["image_sha256"], "question": row["question"],
                         "answer": label["answer"], "task": label["task"],
                         "kind": label.get("kind"), "world_id": label["world_id"]}
                        for row, label in zip(curriculum_train_rows, curriculum_train_labels)]
    diagnostic_train = [{"case_id": row["case_id"], "image": row["image"],
                         "image_sha256": row["image_sha256"], "question": row["question"],
                         "answer": label["answer"], "task": label["task"],
                         "world_id": label["world_id"]}
                        for row, label in zip(diag_train_rows, diag_train_labels)]
    external_train = [{"case_id": row["case_id"], "image": row["image"],
                       "image_sha256": row["image_sha256"], "question": row["question"],
                       "response": label["answer"]}
                      for row, label in zip(external_train_rows, external_train_labels)]
    stage_a, stage_b = _build_training_examples(
        external_train, diagnostic_train, curriculum_train)

    total = int(cases_per_split)
    dev = _joined_cases(manifest, "dev", "fresh_procedural_dev", total)
    audit = _joined_cases(manifest, "audit", "fresh_procedural_audit", total)
    scenes = {entry["scene_sha256"]: entry["scene"] for entry in procedural["scenes"]}

    def captions(cases, split):
        model, labels = [], []
        for case in cases:
            item = {"case_id": case["case_id"], "source": f"fresh_{split}_caption",
                    "image": case["image"], "image_sha256": case["image_sha256"],
                    "question": "Describe the image briefly.", "task": "caption",
                    "world_id": case["world_id"], "scene_sha256": case["scene_sha256"]}
            answer = scene_caption(scenes[case["scene_sha256"]])
            model.append(item)
            labels.append({"case_id": case["case_id"], "split": split,
                           "answer": answer, "task": "caption",
                           "world_id": case["world_id"],
                           "scene_sha256": case["scene_sha256"],
                           "label_visibility": "verifier_only"})
        return model, labels

    caption_dev, caption_dev_labels = captions(dev, "dev")
    caption_audit, caption_audit_labels = captions(audit, "audit")
    files["training/stage_a.jsonl"] = _jsonl_bytes(stage_a)
    files["training/stage_b.jsonl"] = _jsonl_bytes(stage_b)
    for name, rows in (("dev", dev), ("audit", audit)):
        files[f"evaluation/{name}_cases.jsonl"] = _jsonl_bytes(rows)
        files[f"evaluation/{name}_labels.jsonl"] = _jsonl_bytes(_label_rows(rows, name))
    for name, rows, labels in (
            ("dev", caption_dev, caption_dev_labels),
            ("audit", caption_audit, caption_audit_labels)):
        files[f"evaluation/caption_{name}_cases.jsonl"] = _jsonl_bytes(rows)
        files[f"evaluation/caption_{name}_labels.jsonl"] = _jsonl_bytes(labels)

    seals = {
        "dev_case_ids": fingerprint([row["case_id"] for row in dev]),
        "audit_case_ids": fingerprint([row["case_id"] for row in audit]),
        "caption_dev_case_ids": fingerprint([row["case_id"] for row in caption_dev]),
        "caption_audit_case_ids": fingerprint([row["case_id"] for row in caption_audit]),
        "dev_cases_payload": sha256_bytes(files["evaluation/dev_cases.jsonl"]),
        "audit_cases_payload": sha256_bytes(files["evaluation/audit_cases.jsonl"]),
        "caption_dev_payload": sha256_bytes(files["evaluation/caption_dev_cases.jsonl"]),
        "caption_audit_payload": sha256_bytes(files["evaluation/caption_audit_cases.jsonl"]),
    }
    manifest_out = {
        "format": MANIFEST_FORMAT,
        "seed": config["seed"],
        "case_namespace": config["procedural"]["id_prefix"],
        "fresh_evaluation": True,
        "consumed_audit_or_final_examples_reused": False,
        "roles": {
            "stage_a": "admitted external caption train only",
            "stage_b": "owned procedural train only",
            "dev": "fresh bounded stage-development evaluation",
            "audit": "fresh sealed one-shot evaluation after stage B",
            "captions": "owned visible-object captions; no external-final captions",
        },
        "counts": {
            "stage_a": len(stage_a), "stage_b": len(stage_b),
            "dev": len(dev), "audit": len(audit),
            "caption_dev": len(caption_dev), "caption_audit": len(caption_audit),
            "procedural_worlds": len(manifest["worlds"]),
            "procedural_assets": len(manifest["assets"]),
        },
        "seals": seals,
        "procedural_generation_fingerprint": manifest["generation_fingerprint"],
        "config_sha256": fingerprint(config),
        "files": {},
    }
    for relative, payload in sorted(files.items()):
        manifest_out["files"][relative] = {"sha256": sha256_bytes(payload),
                                           "bytes": len(payload)}
    files["manifest.json"] = _json_bytes(manifest_out)
    return {"files": files, "manifest": manifest_out,
            "stage_a": stage_a, "stage_b": stage_b,
            "dev": dev, "audit": audit,
            "dev_labels": _label_rows(dev, "dev"),
            "audit_labels": _label_rows(audit, "audit"),
            "caption_dev": caption_dev, "caption_dev_labels": caption_dev_labels,
            "caption_audit": caption_audit, "caption_audit_labels": caption_audit_labels}


def write_stage_dataset(root, built):
    for relative, payload in sorted(built["files"].items()):
        _write_exact(Path(root) / relative, payload)


def build_stage_protocol(config, built, parent_protocol):
    manifest = built["manifest"]
    protocol = {
        "format": PROTOCOL_FORMAT,
        "version": "replica.vision_stage_diagnostic.v1",
        "seed": config["seed"],
        "purpose": config["purpose"],
        "parent": {
            "protocol": config["parent_protocol"]["path"],
            "protocol_sha256": config["parent_protocol"]["sha256"],
            "role": "accepted V0 integration/source contracts only; no warm start",
        },
        "data": {
            "root": config["data_root"],
            "manifest": {"path": "manifest.json",
                         "sha256": sha256_bytes(built["files"]["manifest.json"])},
            "seals": manifest["seals"],
            "counts": manifest["counts"],
            "fresh_evaluation": True,
        },
        "root": config["project_root"],
        "vision_digest": parent_protocol["vision_digest"],
        "model": dict(parent_protocol["model"]),
        "vision": dict(parent_protocol["vision"]),
        "vision_source": dict(parent_protocol["vision_source"]),
        "action_limits": dict(parent_protocol["action_limits"]),
        "resource_policy": dict(parent_protocol["resource_policy"]),
        "retention": dict(parent_protocol["retention"]),
        "evaluation": {**dict(parent_protocol["evaluation"]),
                       "cases_per_split": config["evaluation"]["cases_per_split"]},
        "limits": {**dict(parent_protocol["limits"]),
                   "max_eval_cases": config["evaluation"]["cases_per_split"],
                   "max_seconds": config["train"]["max_seconds"]},
        "train": dict(config["train"]),
        "schedule": {
            "stage_a": dict(config["schedule"]["stage_a"]),
            "stage_b": dict(config["schedule"]["stage_b"]),
            "total_updates": (config["schedule"]["stage_a"]["updates"]
                              + config["schedule"]["stage_b"]["updates"]),
            "total_examples": (config["schedule"]["stage_a"]["updates"]
                               * config["schedule"]["stage_a"]["batch"]
                               + config["schedule"]["stage_b"]["updates"]
                               * config["schedule"]["stage_b"]["batch"]),
            "response_only_labels": True,
        },
        "checkpoint_policy": dict(config["checkpoint_policy"]),
        "implementation_sha256": {
            name: sha256_file(Path(config["project_root"]) / name)
            for name in config["implementation_paths"]},
        "config_sha256": fingerprint(config),
        "claims": {
            "perception_quality": False, "semantic_quality": False,
            "overall_baseline_eligible": False,
            "scope": "bounded stage-wise collapse diagnostic only",
        },
    }
    return protocol


def write_stage_protocol(path, protocol):
    _write_exact(Path(path), _json_bytes(protocol))
    return sha256_file(path)
