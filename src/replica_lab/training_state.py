"""Atomic optimizer and runtime checkpoints for single-process fine-tuning.

The caller saves model weights separately and publishes a checkpoint only after
both files are complete. This module does not clone the model or optimizer to
CPU before saving: ``torch.save`` serializes the optimizer's resident tensors.
Loading maps storage to CPU, then the optimizer owns its validated migration to
the model's device (including preservation of Muon's int8 momentum).

Data cursors and schedule state are explicit JSON values. A caller must include
the model source, tokenizer, data identity, training regime and hyperparameters
in ``run_contract``. Output paths should not form part of that contract. The
contract is compared before any optimizer or global RNG state is changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import torch
from torch.optim import Optimizer


FORMAT_VERSION = 1


def _json_value(value: Any, name: str) -> Any:
    """Validate and detach JSON-only metadata without accepting custom objects."""
    def validate(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError(f"{name} must contain only finite JSON values with string keys")

    validate(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _contract(run_contract: dict) -> tuple[dict, str]:
    if not isinstance(run_contract, dict) or not run_contract:
        raise ValueError("run_contract must be a nonempty dictionary")
    contract = _json_value(run_contract, "run_contract")
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return contract, hashlib.sha256(encoded).hexdigest()


def _capture_rng() -> dict:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _validate_rng(state: dict, strict_rng: bool) -> None:
    if (not isinstance(state, dict)
            or not {"python", "torch_cpu"}.issubset(state)
            or set(state) - {"python", "torch_cpu", "torch_mps", "torch_cuda"}):
        raise ValueError("Invalid runtime checkpoint RNG state")
    # Validate against private generators before touching the process RNGs.
    try:
        random.Random().setstate(state["python"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid runtime checkpoint CPU/Python RNG state") from exc
    for key in ("torch_cpu", "torch_mps"):
        if key in state:
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                    or value.ndim != 1 or value.device.type != "cpu"):
                raise ValueError(f"Invalid runtime checkpoint {key} RNG tensor")
    if "torch_mps" in state and strict_rng and not torch.backends.mps.is_available():
        raise ValueError("Checkpoint requires MPS RNG; use strict_rng=False for backend migration")
    if "torch_cuda" in state:
        values = state["torch_cuda"]
        if not isinstance(values, list) or not values:
            raise ValueError("Invalid runtime checkpoint CUDA RNG state")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
               or value.ndim != 1 or value.device.type != "cpu" for value in values):
            raise ValueError("Invalid runtime checkpoint CUDA RNG tensors")
        if strict_rng and (not torch.cuda.is_available()
                           or len(values) != torch.cuda.device_count()):
            raise ValueError("Checkpoint requires the saved CUDA RNG device count; "
                             "use strict_rng=False for backend migration")


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])
    if "torch_cuda" in state and torch.cuda.is_available():
        # Non-strict migration restores the devices shared with the checkpoint.
        for index, value in enumerate(state["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, device=index)


def optimizer_state_to(optimizer: Optimizer, device) -> None:
    """Move allocated optimizer state without changing parameter ownership."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_runtime_state(path: str | Path, optimizer: Optimizer, *, step: int,
                       data_state: dict, schedule_state: dict | None = None,
                       run_contract: dict) -> dict:
    """Atomically save optimizer, RNG, completed-step count and sampler cursor.

    ``step`` counts completed optimizer updates. Call between updates after
    clearing gradients; pending accumulated gradients are not checkpointed.
    Metadata must use finite JSON values. The destination is a single file,
    conventionally ``runtime.pt``, replaced only after a complete, fsynced save.
    """
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(data_state, dict):
        raise ValueError("data_state must be a dictionary")
    if schedule_state is not None and not isinstance(schedule_state, dict):
        raise ValueError("schedule_state must be a dictionary or None")
    contract, digest = _contract(run_contract)
    rng = _capture_rng()
    metadata = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "data_state": _json_value(data_state, "data_state"),
        "schedule_state": _json_value(schedule_state, "schedule_state"),
        "run_contract": contract,
        "contract_sha256": digest,
        "rng_devices": sorted(key.removeprefix("torch_") for key in rng if key != "python"),
    }
    payload = {"metadata": metadata, "optimizer": optimizer.state_dict(), "rng": rng}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.",
                                             suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def load_runtime_state(path: str | Path, optimizer: Optimizer, *, run_contract: dict,
                       restore_rng: bool = True, strict_rng: bool = True,
                       target_device=None) -> dict:
    """Validate the contract, restore optimizer and optionally global RNG state.

    The caller must first restore the corresponding model weights. Return the
    completed ``step``, ``data_state`` and ``schedule_state`` for the training
    loop. Loading uses ``weights_only=True`` and never permits custom pickle
    globals. Exact continuation requires the same device/backend and unchanged
    numerical execution; ``strict_rng=False`` explicitly permits migration by
    restoring only available saved RNG devices, without equivalence guarantees.
    """
    expected, expected_digest = _contract(run_contract)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "optimizer", "rng"}:
        raise ValueError("Invalid runtime checkpoint payload")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported runtime checkpoint format")
    saved, digest = _contract(metadata.get("run_contract"))
    if digest != metadata.get("contract_sha256"):
        raise ValueError("Runtime checkpoint contract hash is inconsistent")
    if saved != expected or digest != expected_digest:
        raise ValueError("Runtime checkpoint run_contract differs from the requested run")
    step = metadata.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Invalid runtime checkpoint step")
    if not isinstance(metadata.get("data_state"), dict):
        raise ValueError("Invalid runtime checkpoint data_state")
    if (metadata.get("schedule_state") is not None
            and not isinstance(metadata["schedule_state"], dict)):
        raise ValueError("Invalid runtime checkpoint schedule_state")
    result = _json_value(metadata, "runtime checkpoint metadata")
    if restore_rng:
        _validate_rng(payload["rng"], strict_rng)
    if target_device is None:
        optimizer.load_state_dict(payload["optimizer"])
    else:
        optimizer.load_state_dict(payload["optimizer"], target_device=target_device)
    if restore_rng:
        _restore_rng(payload["rng"])
    return result


def materialize_resume_model(base_path, *, adapter_factory, optimizer_factory,
                             runtime_path, run_contract, device,
                             restore_rng=True, strict_rng=True):
    """Restore optimizer state before materializing full model weights on device.

    Meta-device construction owns the complete architecture and parameter names
    without allocating model storage. The checkpoint optimizer is restored on
    the target backend first, then full weights and any sidecars are materialized.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    base_path = Path(base_path)
    with torch.device("meta"):
        config = AutoConfig.from_pretrained(base_path)
        base = AutoModelForCausalLM.from_config(config)
        model = adapter_factory(base)
        optimizer = optimizer_factory(model)
    base_rates = [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []
    restored = load_runtime_state(
        runtime_path, optimizer, run_contract=run_contract,
        restore_rng=restore_rng, strict_rng=strict_rng, target_device=device)
    base = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
    model = adapter_factory(base)
    model.to(device)
    actual_optimizer = optimizer_factory(model)
    actual_optimizer.load_state_dict(optimizer.state_dict(), target_device=device)
    if getattr(model, "augmented", False):
        model.load_sidecars(base_path.parent / "sidecars.safetensors")
    return base, model, actual_optimizer, restored, base_rates
