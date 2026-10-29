"""Versioned extension-only Spark spine checkpoints and exact recovery.

Version 1 checkpoints remain loadable through explicit, validated migrations.
Version 2 additionally persists bounded P teaching/replay lineage and rebuild
contracts. No stable-backbone bytes or derived source caches are serialized.
"""
from __future__ import annotations
import copy

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .spine import SPINE_SCHEMA, SpineCore, SpineError
from .. import artifact_publication as publication
from .. import training_state
from ..optimizers import NativeFusedAdamW

SPINE_CHECKPOINT_FORMAT = "replica.spark.spine.checkpoint.v1"
SPINE_MEMORY_CHECKPOINT_FORMAT = "replica.spark.spine.memory.checkpoint.v1"
SPINE_CHECKPOINT_VERSION = 2


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
        "format": format, "version": SPINE_CHECKPOINT_VERSION,
        "arm": arm, "event": event, "phase": phase,
        "completed_evaluations": copy.deepcopy(completed_evaluations),
        "counters": dict(counters),
        "run_contract": copy.deepcopy(dict(run_contract)),
    }


def _validate_representation(core: SpineCore, run_contract: Mapping[str, Any]) -> None:
    if run_contract["codec_version"] != core.codec_version:
        raise SpineError("run_contract codec_version differs from the spine core")
    if run_contract["representation_version"] != core.representation_version:
        raise SpineError("run_contract representation_version differs from the spine core")


def save_spine_checkpoint(root: str | Path, relative: str, core: SpineCore,
                          optimizer: Any, *, arm: str, event: str, phase: str,
                          completed_evaluations: list[dict], counters: dict,
                          run_contract: Mapping[str, Any], meter: Any = None,
                          plastic: Mapping[str, Any] | None = None,
                          evidence: Mapping[str, Any] | None = None,
                          memory: Mapping[str, Any] | None = None) -> dict:
    """Atomically save spine state, optionally with explicit E/P/lineage state."""
    if (plastic is None) != (evidence is None):
        raise SpineError("plastic and evidence checkpoint state must be paired")
    if plastic is not None and (
            not isinstance(plastic, Mapping) or not isinstance(evidence, Mapping)
            or not isinstance(memory, Mapping)):
        raise SpineError("v2 plastic, evidence and memory state must be mappings")
    if plastic is None and memory is not None:
        raise SpineError("memory state requires plastic/evidence state")
    _validate_representation(core, run_contract)
    checkpoint_format = (SPINE_MEMORY_CHECKPOINT_FORMAT if plastic is not None
                         else SPINE_CHECKPOINT_FORMAT)
    metadata = _metadata(arm, event, phase, completed_evaluations, counters,
                         run_contract, format=checkpoint_format)
    payload = {
        **metadata, "spine_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "rng": training_state._capture_rng(),
    }
    if plastic is not None:
        payload["plastic_state"] = copy.deepcopy(dict(plastic))
        payload["evidence_state"] = copy.deepcopy(dict(evidence))
        payload["memory_state"] = copy.deepcopy(dict(memory))
    payload["_estimated_bytes"] = _payload_bytes(payload)
    artifact = publication.write_torch(
        Path(root), relative, payload, meter=meter, reuse_identical=True)
    return {"artifact": artifact.as_dict(), "metadata": metadata,
            "estimated_bytes": payload["_estimated_bytes"]}


def _native_optimizer_state(optimizer: NativeFusedAdamW,
                            saved: Any) -> tuple[dict, str | None]:
    if not isinstance(saved, Mapping):
        raise SpineError("native optimizer checkpoint must be a mapping")
    if "replica_optimizer" in saved:
        if set(saved) != {"state", "param_groups", "replica_optimizer"}:
            raise SpineError("native optimizer checkpoint keys differ")
        return copy.deepcopy(dict(saved)), None
    # Explicit migration of the torch.optim.AdamW state emitted by spine v1.
    # Custom ReplicaOptimizer/Muon states never satisfy this closed schema.
    if set(saved) != {"state", "param_groups"}:
        raise SpineError("legacy optimizer checkpoint is not plain AdamW v1")
    groups = saved["param_groups"]
    state = saved["state"]
    if not isinstance(groups, list) or not isinstance(state, Mapping) \
            or len(groups) != len(optimizer.param_groups):
        raise SpineError("legacy optimizer checkpoint groups differ")
    migrated_groups = []
    for old, current in zip(groups, optimizer.param_groups):
        if not isinstance(old, Mapping) or "params" not in old \
                or len(old["params"]) != len(current["params"]):
            raise SpineError("legacy optimizer parameter group differs")
        if old.get("algorithm") not in (None, "adamw_fused"):
            raise SpineError("legacy optimizer contains a non-AdamW algorithm")
        group = copy.deepcopy(dict(old))
        for key in ("group_name", "scope", "algorithm", "param_names"):
            group[key] = copy.deepcopy(current[key])
        group["fused"] = True
        migrated_groups.append(group)
    for value in state.values():
        if not isinstance(value, Mapping) or set(value) != {
                "step", "exp_avg", "exp_avg_sq"}:
            raise SpineError("legacy optimizer contains incompatible state")
    migrated = {"state": copy.deepcopy(dict(state)),
                "param_groups": migrated_groups,
                "replica_optimizer": copy.deepcopy(
                    optimizer.state_dict()["replica_optimizer"])}
    return migrated, "torch_adamw_v1_to_native_fused_v1"


def _legacy_memory_state(memory: Any) -> dict:
    # A v1 checkpoint had no journal/lineage fields. Construct a closed v2
    # migration payload rather than accepting arbitrary missing v2 keys.
    payload = memory.memory_checkpoint_payload()
    payload["learning_journal"] = []
    payload["taught_evidence"] = []
    payload["state_generation"] = 0
    payload["lineage_complete"] = False
    return payload


def load_spine_checkpoint(root: str | Path, relative: str, core: SpineCore,
                          optimizer: Any, *, run_contract: Mapping[str, Any],
                          restore_rng: bool = True, strict_rng: bool = True,
                          plastic: Any = None, evidence: Any = None,
                          memory: Any = None) -> dict:
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
    if not memory_checkpoint and memory is not None:
        raise SpineError("memory owner supplied for a spine-only checkpoint")
    version = payload.get("version")
    if version not in (1, SPINE_CHECKPOINT_VERSION):
        raise SpineError("spine checkpoint schema/version differs")
    required = {"format", "version", "arm", "event", "phase",
                "completed_evaluations", "counters", "run_contract",
                "spine_state", "optimizer_state", "rng", "_estimated_bytes"}
    if memory_checkpoint:
        required.update(("plastic_state", "evidence_state"))
        if version >= 2:
            required.add("memory_state")
    if set(payload) != required:
        raise SpineError("spine checkpoint keys differ")
    if version >= 2 and memory_checkpoint and memory is None:
        raise SpineError("v2 memory checkpoint requires its memory state owner")
    metadata = _metadata(payload["arm"], payload["event"], payload["phase"],
                         payload["completed_evaluations"], payload["counters"],
                         payload["run_contract"], format=checkpoint_format)
    if metadata["run_contract"] != dict(run_contract):
        raise SpineError("spine checkpoint run_contract differs")
    _validate_representation(core, run_contract)
    if type(payload["_estimated_bytes"]) is not int \
            or payload["_estimated_bytes"] != _payload_bytes(payload):
        raise SpineError("spine checkpoint estimated byte count differs")
    expected = core.state_dict()
    state = payload["spine_state"]
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise SpineError("spine checkpoint tensor inventory differs")
    for name, parameter in expected.items():
        tensor = state[name]
        if not hasattr(tensor, "shape") or tensor.shape != parameter.shape \
                or tensor.dtype != parameter.dtype:
            raise SpineError(f"spine checkpoint tensor differs: {name}")
    optimizer_state = payload["optimizer_state"]
    optimizer_migration = None
    if optimizer is not None:
        if optimizer_state is None:
            raise SpineError("checkpoint has no optimizer state")
        if isinstance(optimizer, NativeFusedAdamW):
            optimizer_state, optimizer_migration = _native_optimizer_state(
                optimizer, optimizer_state)
        elif isinstance(optimizer_state, Mapping) \
                and "replica_optimizer" in optimizer_state:
            raise SpineError("native optimizer checkpoint requires NativeFusedAdamW")
    elif optimizer_state is not None:
        raise SpineError("checkpoint unexpectedly carries optimizer state")
    memory_state = payload.get("memory_state")
    memory_migration = None
    if memory_checkpoint and version == 1 and memory is not None:
        memory_state = _legacy_memory_state(memory)
        memory_migration = "memory_v1_to_bounded_lineage_v2"
    if memory is not None:
        memory.validate_completed_evaluations(payload["completed_evaluations"])
        metadata["counters"] = memory.migrate_counters(
            payload["counters"], version)
    core_before = {name: value.detach().clone()
                   for name, value in expected.items()}
    optimizer_before = (copy.deepcopy(optimizer.state_dict())
                        if optimizer is not None else None)
    plastic_before = (copy.deepcopy(plastic.checkpoint_payload())
                      if memory_checkpoint else None)
    evidence_before = (copy.deepcopy(evidence.checkpoint_payload())
                       if memory_checkpoint else None)
    memory_before = (copy.deepcopy(memory.memory_checkpoint_payload())
                     if memory is not None else None)
    rng_before = training_state._capture_rng()
    try:
        core.load_state_dict(state)
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        if memory_checkpoint:
            plastic.load_checkpoint(payload["plastic_state"])
            evidence.load_checkpoint(payload["evidence_state"])
            if memory is not None:
                memory.load_memory_checkpoint(memory_state)
        if restore_rng:
            training_state._validate_rng(payload["rng"], strict_rng)
            training_state._restore_rng(payload["rng"])
    except BaseException:
        core.load_state_dict(core_before)
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        if memory_checkpoint:
            plastic.load_checkpoint(plastic_before)
            evidence.load_checkpoint(evidence_before)
            if memory is not None:
                memory.load_memory_checkpoint(memory_before)
        training_state._restore_rng(rng_before)
        raise
    return {**metadata, "version": version, "checkpoint_version": version,
            "optimizer_migration": optimizer_migration,
            "memory_state_migration": memory_migration}


__all__ = ["SPINE_CHECKPOINT_FORMAT", "SPINE_MEMORY_CHECKPOINT_FORMAT",
           "SPINE_CHECKPOINT_VERSION", "save_spine_checkpoint", "load_spine_checkpoint"]
