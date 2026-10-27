"""Canonical full-state evidence for independent nested-text continuation.

The legacy ``state_digest`` remains a compact compatibility receipt. This
module instead flattens the complete checkpoint payload, including optimizer
moments, dedup outcomes, coordinator chunks and event/route/personality
bindings. Tensor hashes use FP32 CPU value bytes plus dtype/shape metadata so a
checkpoint reload is compared by content rather than host placement.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

FORMAT = "replica.nested_continuation.state.v1"
_SECTIONS = (
    "controller", "controller.bank", "controller.bank.optimizer",
    "controller.clock", "controller.dedup", "controller.observation",
    "controller.revision", "controller.personality", "text", "text.protocol",
    "text.route", "text.feature_producer", "text.read_path",
    "text.personality", "text.event_fingerprints",
)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _tensor_summary(value: torch.Tensor) -> dict:
    values = value.detach().to("cpu", torch.float32).contiguous()
    return {
        "kind": "tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


def _flatten(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{path}.{key}" if path else str(key)
            yield from _flatten(value[key], child)
    elif isinstance(value, (list, tuple)):
        if len(value) > 16 and all(
                not isinstance(item, (Mapping, list, tuple, torch.Tensor))
                for item in value):
            yield path, {"kind": "sequence", "length": len(value),
                         "sha256": _sha_json(value)}
        else:
            for index, item in enumerate(value):
                yield from _flatten(item, f"{path}[{index}]")
    elif isinstance(value, torch.Tensor):
        yield path, _tensor_summary(value)
    elif value is None:
        yield path, {"kind": "null"}
    elif isinstance(value, bool):
        yield path, {"kind": "bool", "value": value}
    elif isinstance(value, int):
        yield path, {"kind": "int", "value": value}
    elif isinstance(value, float):
        yield path, {"kind": "float", "value": value}
    elif isinstance(value, str):
        yield path, {"kind": "str", "value": value}
    else:
        raise TypeError(f"Unsupported checkpoint value at {path}: {type(value)!r}")


def canonical_state(learner) -> dict:
    """Return composed and per-leaf hashes for one learner checkpoint state."""
    payload = learner.checkpoint_payload()
    leaves = dict(_flatten(payload))
    sections = {}
    for section in _SECTIONS:
        selected = {path: value for path, value in leaves.items()
                    if path == section or path.startswith(section + ".")
                    or path.startswith(section + "[")}
        if selected:
            sections[section] = _sha_json(selected)
    controller = payload["controller"]
    bank = controller["bank"]
    observation = controller.get("observation") or {}
    text = payload.get("text") or {}
    counts = {
        "parameter_tensors": len(bank["parameters"]),
        "parameter_elements": sum(int(item[1].numel())
                                  for item in bank["parameters"]),
        "optimizer_state_entries": len(bank["optimizer"].get("state", {})),
        "observation_count": len(observation.get("observations", [])),
        "chunk_count": len(observation.get("chunks", [])),
        "event_fingerprint_count": len(text.get("event_fingerprints", {})),
        "leaf_count": len(leaves),
    }
    return {
        "format": FORMAT,
        "sha256": _sha_json(leaves),
        "sections": sections,
        "leaves": leaves,
        "counts": counts,
    }


def compare_states(left: Mapping, right: Mapping) -> dict:
    """Localize the first exact difference between two canonical states."""
    _require_state(left)
    _require_state(right)
    left_leaves, right_leaves = left["leaves"], right["leaves"]
    missing = sorted(set(left_leaves) - set(right_leaves))
    extra = sorted(set(right_leaves) - set(left_leaves))
    differences = []
    for path in sorted(set(left_leaves) & set(right_leaves)):
        if left_leaves[path] != right_leaves[path]:
            differences.append(path)
    section_differences = [
        name for name in sorted(set(left["sections"]) | set(right["sections"]))
        if left["sections"].get(name) != right["sections"].get(name)]
    first = None
    if missing or extra or differences:
        # Prefer an actual content difference over a missing-side artifact.
        candidates = sorted(set(missing) | set(extra) | set(differences))
        path = candidates[0]
        first = {"path": path,
                 "left": left_leaves.get(path),
                 "right": right_leaves.get(path)}
    return {
        "equal": left["sha256"] == right["sha256"] and not (
            missing or extra or differences),
        "left_sha256": left["sha256"],
        "right_sha256": right["sha256"],
        "section_differences": section_differences,
        "missing_paths": missing[:32],
        "extra_paths": extra[:32],
        "different_paths": differences[:32],
        "first_difference": first,
    }


def _require_state(value: Mapping) -> None:
    if not isinstance(value, Mapping) or value.get("format") != FORMAT:
        raise ValueError("Canonical continuation state format differs")
    if not isinstance(value.get("sha256"), str) or not isinstance(
            value.get("sections"), Mapping) or not isinstance(
            value.get("leaves"), Mapping):
        raise ValueError("Canonical continuation state is incomplete")


__all__ = ["FORMAT", "canonical_state", "compare_states"]
