"""Versioned extension-only Spark spine checkpoints and exact recovery.

The checkpoint contains no stable-backbone bytes. It binds the spine schema,
codec/representation versions, active arm/event/phase, completed evaluations,
parameters, optimizer state, RNG and work counters. Storage is atomic through
``artifact_publication``; validation never accepts broad missing-key defaults.
"""
from __future__ import annotations
import copy

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .spine import SPINE_SCHEMA, SpineCore, SpineError
from .. import artifact_publication as publication
from .. import training_state

SPINE_CHECKPOINT_FORMAT = "replica.spark.spine.checkpoint.v1"
SPINE_MEMORY_CHECKPOINT_FORMAT = "replica.spark.spine.memory.checkpoint.v1"
SPINE_CHECKPOINT_VERSION = 1


def _payload_bytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(item) for item in value)
    return 0


def _metadata(arm: str, event: str, phase: str,
              completed_evaluations: list[dict], counters: dict,
              run_contract: Mapping[str, Any], *,
              format: str = SPINE_CHECKPOINT_FORMAT) -> dict:
    if not all(isinstance(value, str) and value for value in (arm, event, phase)):
        raise SpineError("active arm, event and phase are required")
    if not isinstance(completed_evaluations, list) or any(
            not isinstance(row, dict) for row in completed_evaluations):
        raise SpineError("completed_evaluations must be a list of objects")
    if not isinstance(counters, dict):
        raise SpineError("counters must be an object")
    for name, value in counters.items():
        if not isinstance(name, str) or not isinstance(value, int) \
                or isinstance(value, bool) or value < 0:
            raise SpineError(f"invalid work counter {name}")
    if not isinstance(run_contract, Mapping):
        raise SpineError("run_contract must be an object")
    for key in ("schema", "spine_schema", "model_identity", "codec_version",
                "representation_version"):
        if key not in run_contract:
            raise SpineError(f"run_contract missing {key}")
    if run_contract["schema"] != SPINE_SCHEMA:
        raise SpineError("run_contract spine schema differs")
    if run_contract["spine_schema"] != SPINE_SCHEMA:
        raise SpineError("run_contract spine_schema differs")
    for key in ("model_identity", "codec_version", "representation_version"):
        if not isinstance(run_contract[key], str) or not run_contract[key]:
            raise SpineError(f"run_contract {key} must be nonempty")
    return {
        "format": format,
        "version": SPINE_CHECKPOINT_VERSION,
        "arm": arm,
        "event": event,
        "phase": phase,
        "completed_evaluations": completed_evaluations,
        "counters": dict(counters),
        "run_contract": dict(run_contract),
    }


def save_spine_checkpoint(root: str | Path, relative: str, core: SpineCore,
                          optimizer: Any, *, arm: str, event: str, phase: str,
                          completed_evaluations: list[dict], counters: dict,
                          run_contract: Mapping[str, Any], meter: Any = None,
                          plastic: Mapping[str, Any] | None = None,
                          evidence: Mapping[str, Any] | None = None) -> dict:
    """Atomically save spine state, optionally with explicit E/P state."""
    if (plastic is None) != (evidence is None):
        raise SpineError("plastic and evidence checkpoint state must be paired")
    if plastic is not None and (
            not isinstance(plastic, Mapping) or not isinstance(evidence, Mapping)):
        raise SpineError("plastic and evidence checkpoint state must be mappings")
    checkpoint_format = (SPINE_MEMORY_CHECKPOINT_FORMAT if plastic is not None
                         else SPINE_CHECKPOINT_FORMAT)
    metadata = _metadata(arm, event, phase, completed_evaluations, counters,
                         run_contract, format=checkpoint_format)
    payload = {
        **metadata,
        "spine_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "rng": training_state._capture_rng(),
    }
    if plastic is not None:
        payload["plastic_state"] = copy.deepcopy(dict(plastic))
        payload["evidence_state"] = copy.deepcopy(dict(evidence))
    payload["_estimated_bytes"] = _payload_bytes(payload)
    artifact = publication.write_torch(
        Path(root), relative, payload, meter=meter, reuse_identical=True)
    return {"artifact": artifact.as_dict(), "metadata": metadata,
            "estimated_bytes": payload["_estimated_bytes"]}


def load_spine_checkpoint(root: str | Path, relative: str, core: SpineCore,
                          optimizer: Any, *, run_contract: Mapping[str, Any],
                          restore_rng: bool = True, strict_rng: bool = True,
                          plastic: Any = None, evidence: Any = None) -> dict:
    """Validate and restore one checkpoint without accepting schema drift."""
    if (plastic is None) != (evidence is None):
        raise SpineError("plastic and evidence checkpoint owners must be paired")
    payload = publication.load_torch(Path(root), relative)
    if not isinstance(payload, Mapping):
        raise SpineError("spine checkpoint must be a mapping")
    checkpoint_format = payload.get("format")
    if checkpoint_format not in (SPINE_CHECKPOINT_FORMAT,
                                 SPINE_MEMORY_CHECKPOINT_FORMAT):
        raise SpineError("spine checkpoint schema/version differs")
    memory_checkpoint = checkpoint_format == SPINE_MEMORY_CHECKPOINT_FORMAT
    if memory_checkpoint != (plastic is not None):
        raise SpineError("checkpoint E/P state and supplied owners differ")
    required = {"format", "version", "arm", "event", "phase",
                "completed_evaluations", "counters", "run_contract",
                "spine_state", "optimizer_state", "rng", "_estimated_bytes"}
    if memory_checkpoint:
        required.update(("plastic_state", "evidence_state"))
    if set(payload) != required:
        raise SpineError("spine checkpoint keys differ")
    metadata = _metadata(payload["arm"], payload["event"], payload["phase"],
                         payload["completed_evaluations"], payload["counters"],
                         payload["run_contract"], format=checkpoint_format)
    if payload["version"] != SPINE_CHECKPOINT_VERSION:
        raise SpineError("spine checkpoint schema/version differs")
    if metadata["run_contract"] != dict(run_contract):
        raise SpineError("spine checkpoint run_contract differs")
    expected = core.state_dict()
    state = payload["spine_state"]
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise SpineError("spine checkpoint tensor inventory differs")
    for name, parameter in expected.items():
        tensor = state[name]
        if not hasattr(tensor, "shape") or tensor.shape != parameter.shape \
                or tensor.dtype != parameter.dtype:
            raise SpineError(f"spine checkpoint tensor differs: {name}")
    core_before = {name: value.detach().clone()
                   for name, value in expected.items()}
    optimizer_before = (copy.deepcopy(optimizer.state_dict())
                        if optimizer is not None else None)
    plastic_before = (copy.deepcopy(plastic.checkpoint_payload())
                      if memory_checkpoint else None)
    evidence_before = (copy.deepcopy(evidence.checkpoint_payload())
                       if memory_checkpoint else None)
    rng_before = training_state._capture_rng()
    try:
        core.load_state_dict(state)
        if optimizer is not None:
            if payload["optimizer_state"] is None:
                raise SpineError("checkpoint has no optimizer state")
            optimizer.load_state_dict(payload["optimizer_state"])
        elif payload["optimizer_state"] is not None:
            raise SpineError("checkpoint unexpectedly carries optimizer state")
        if memory_checkpoint:
            plastic.load_checkpoint(payload["plastic_state"])
            evidence.load_checkpoint(payload["evidence_state"])
        if restore_rng:
            training_state._validate_rng(payload["rng"], strict_rng)
            training_state._restore_rng(payload["rng"])
    except BaseException:
        core.load_state_dict(core_before)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_before)
        if memory_checkpoint:
            plastic.load_checkpoint(plastic_before)
            evidence.load_checkpoint(evidence_before)
        training_state._restore_rng(rng_before)
        raise
    return metadata


__all__ = ["SPINE_CHECKPOINT_FORMAT", "SPINE_MEMORY_CHECKPOINT_FORMAT",
           "SPINE_CHECKPOINT_VERSION", "save_spine_checkpoint", "load_spine_checkpoint"]
