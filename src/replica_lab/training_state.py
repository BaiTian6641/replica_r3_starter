"""Atomic optimizer and runtime checkpoints for single-process fine-tuning.

The caller saves model weights separately and publishes a checkpoint only after
both files are complete. This module does not clone the model or optimizer to
CPU before saving: ``torch.save`` serializes the optimizer's resident tensors.
Loading maps storage to the requested target device (CPU for fixtures, native
device for native resume), then the optimizer owns validated state attachment.

Data cursors and schedule state are explicit JSON values. A caller must include
the model source, tokenizer, data identity, training regime and hyperparameters
in ``run_contract``. Output paths should not form part of that contract. The
contract is compared before any optimizer or global RNG state is changed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys
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


FINGERPRINT_FORMAT = "replica.training_fingerprint.v1"
_NORM_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _finite_float(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _storage_bytes(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor.untyped_storage().nbytes())
    except (RuntimeError, AttributeError, NotImplementedError):
        return None


def _norm_summary(tensor: torch.Tensor, chunk_elements: int = _NORM_CHUNK_ELEMENTS) -> dict:
    """Chunked L1/L2/max-abs summary without a full-size temporary copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor summary requires a torch.Tensor")
    detached = tensor.detach()
    entry = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "storage_bytes": _storage_bytes(detached),
        "storage_offset": int(detached.storage_offset()),
        "stride": list(detached.stride()),
        "is_contiguous": bool(detached.is_contiguous()),
    }
    if detached.numel() == 0:
        entry.update({"l1_norm": 0.0, "l2_norm": 0.0, "max_abs": 0.0, "finite": True})
        return entry
    flat = detached.reshape(-1)
    l1 = torch.zeros((), dtype=torch.float32, device=flat.device)
    l2_sq = torch.zeros((), dtype=torch.float32, device=flat.device)
    max_abs = torch.zeros((), dtype=torch.float32, device=flat.device)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements].float()
        absolute = chunk.abs()
        l1 = l1 + absolute.sum()
        l2_sq = l2_sq + (chunk * chunk).sum()
        max_abs = torch.maximum(max_abs, absolute.max())
    l1_value = _finite_float(l1.item())
    l2_value = _finite_float(math.sqrt(max(0.0, float(l2_sq.item()))))
    max_value = _finite_float(max_abs.item())
    entry.update({
        "l1_norm": l1_value,
        "l2_norm": l2_value,
        "max_abs": max_value,
        "finite": all(value is not None for value in (l1_value, l2_value, max_value)),
    })
    return entry


def tensor_fingerprint(tensor: torch.Tensor, *, hash_values: bool = True,
                       chunk_elements: int = _NORM_CHUNK_ELEMENTS) -> dict:
    """Record value bytes, storage metadata and numeric norms for one tensor.

    Values are streamed to CPU in chunks, so a boundary record does not retain a
    full second CPU copy of a large parameter. Hashes are over the original dtype
    bytes in logical row-major order; norms and metadata remain useful when an
    exact hash is not comparable.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_fingerprint requires a torch.Tensor")
    detached = tensor.detach()
    entry = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "storage_bytes": _storage_bytes(detached),
        "storage_offset": int(detached.storage_offset()),
        "stride": list(detached.stride()),
        "is_contiguous": bool(detached.is_contiguous()),
        "requires_grad": bool(detached.requires_grad),
    }
    if not hash_values:
        entry.update(_norm_summary(detached, chunk_elements=chunk_elements))
        entry["value_sha256"] = None
        return entry
    flat = detached.reshape(-1)
    hasher = hashlib.sha256()
    l1_total = 0.0
    l2_squared = 0.0
    max_abs = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements].to("cpu")
        hasher.update(chunk.contiguous().view(torch.uint8).numpy().tobytes(order="C"))
        statistics = chunk.float()
        absolute = statistics.abs()
        l1_total += float(absolute.sum(dtype=torch.float64))
        l2_squared += float((statistics.double() * statistics.double()).sum())
        if absolute.numel():
            max_abs = max(max_abs, float(absolute.max().item()))
    entry["value_sha256"] = hasher.hexdigest()
    entry.update({
        "l1_norm": _finite_float(l1_total),
        "l2_norm": _finite_float(math.sqrt(max(0.0, l2_squared))),
        "max_abs": _finite_float(max_abs),
        "finite": all(math.isfinite(value) for value in (l1_total, l2_squared, max_abs)),
    })
    return entry


def tensor_error_norms(reference: torch.Tensor, actual: torch.Tensor, *,
                       chunk_elements: int = _NORM_CHUNK_ELEMENTS) -> dict:
    """Exact elementwise error norms between two tensors, accumulated in chunks."""
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise TypeError("tensor_error_norms requires two torch.Tensor values")
    left, right = reference.detach(), actual.detach()
    result = {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "shape_equal": tuple(left.shape) == tuple(right.shape),
        "dtype_equal": left.dtype == right.dtype,
        "compared_elements": 0,
        "max_abs_error": None,
        "rms_error": None,
        "l1_error": None,
        "exact_equal": False,
    }
    if not (result["shape_equal"] and result["dtype_equal"]):
        return result
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    max_error = 0.0
    squared = 0.0
    absolute_sum = 0.0
    for start in range(0, flat_left.numel(), chunk_elements):
        left_chunk = flat_left[start:start + chunk_elements].to("cpu", torch.float64)
        right_chunk = flat_right[start:start + chunk_elements].to("cpu", torch.float64)
        difference = left_chunk - right_chunk
        absolute = difference.abs()
        max_error = max(max_error, float(absolute.max().item()) if absolute.numel() else 0.0)
        squared += float((difference * difference).sum().item())
        absolute_sum += float(absolute.sum().item())
    count = int(flat_left.numel())
    result.update({
        "compared_elements": count,
        "max_abs_error": _finite_float(max_error),
        "rms_error": _finite_float(math.sqrt(squared / count)) if count else 0.0,
        "l1_error": _finite_float(absolute_sum),
        "exact_equal": bool(count == flat_right.numel() and max_error == 0.0),
    })
    return result


def _parameter_name_map(module) -> dict[int, str]:
    return {id(parameter): name for name, parameter in module.named_parameters()}


def _nonpersistent_buffer_names(module) -> set[str]:
    names = set()
    for module_name, child in module.named_modules():
        for name in child._non_persistent_buffers_set:
            names.add(f"{module_name}.{name}" if module_name else name)
    return names


def fingerprint_train_mode(module) -> dict:
    entries = [{"name": name, "training": bool(child.training)}
               for name, child in module.named_modules()]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "model_training": bool(module.training),
        "module_count": len(entries),
        "training_modules": sum(entry["training"] for entry in entries),
        "evaluation_modules": sum(not entry["training"] for entry in entries),
        "modules": entries,
        "mode_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def fingerprint_buffers(module, *, hash_values: bool = True) -> dict:
    nonpersistent = _nonpersistent_buffer_names(module)
    buffers = {}
    for name, buffer in module.named_buffers():
        entry = tensor_fingerprint(buffer, hash_values=hash_values)
        entry["persistent"] = name not in nonpersistent
        buffers[name] = entry
    return {
        "count": len(buffers),
        "nonpersistent_names": sorted(nonpersistent),
        "buffers": buffers,
    }


def fingerprint_parameters(module, *, hash_values: bool = True) -> dict:
    return {name: tensor_fingerprint(parameter, hash_values=hash_values)
            for name, parameter in module.named_parameters()}


def _rng_fingerprint(state: dict) -> dict:
    result = {
        "python_sha256": hashlib.sha256(
            pickle.dumps(state["python"], protocol=4)).hexdigest(),
        "devices": [],
    }
    for key in ("torch_cpu", "torch_mps"):
        if key in state:
            result[key] = tensor_fingerprint(state[key], hash_values=True)
            result["devices"].append(key.removeprefix("torch_"))
    if "torch_cuda" in state:
        result["torch_cuda"] = [tensor_fingerprint(value, hash_values=True)
                                for value in state["torch_cuda"]]
        result["devices"].append("cuda")
    return result


def fingerprint_rng() -> dict:
    return _rng_fingerprint(_capture_rng())


def _counter_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return tensor_fingerprint(value, hash_values=True)
    return value


def fingerprint_optimizer(optimizer: Optimizer, module=None) -> dict:
    names = _parameter_name_map(module) if module is not None else {}
    groups = []
    state_bytes = 0
    state_tensors = 0
    bytes_by_dtype: dict[str, int] = {}
    seen_parameters = set()
    state = {}
    for group_index, group in enumerate(optimizer.param_groups):
        parameter_names = []
        for parameter in group["params"]:
            parameter_name = names.get(id(parameter), f"parameter_{len(seen_parameters)}")
            parameter_names.append(parameter_name)
            seen_parameters.add(id(parameter))
            parameter_state = optimizer.state.get(parameter, {})
            if not parameter_state:
                continue
            entry = {}
            for key, value in sorted(parameter_state.items()):
                if isinstance(value, torch.Tensor):
                    entry[key] = tensor_fingerprint(value, hash_values=True)
                    state_tensors += 1
                    element_bytes = int(value.numel() * value.element_size())
                    state_bytes += element_bytes
                    dtype_name = str(value.dtype)
                    bytes_by_dtype[dtype_name] = bytes_by_dtype.get(dtype_name, 0) + element_bytes
                else:
                    entry[key] = _counter_value(value)
            state[parameter_name] = entry
        groups.append({
            "group_index": group_index,
            "group_name": group.get("group_name"),
            "algorithm": group.get("algorithm"),
            "lr": group.get("lr"),
            "betas": list(group["betas"]) if "betas" in group else None,
            "eps": group.get("eps"),
            "weight_decay": group.get("weight_decay"),
            "fused": group.get("fused"),
            "parameter_names": parameter_names,
            "tensor_count": len(group["params"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        })
    step_values = sorted({float(value["step"].item())
                          for value in optimizer.state.values()
                          if isinstance(value.get("step"), torch.Tensor)
                          and value["step"].numel() == 1})
    step_values += sorted({float(value["step"]) for value in optimizer.state.values()
                           if isinstance(value.get("step"), (int, float))
                           and not isinstance(value.get("step"), bool)})
    return {
        "parameter_count": sum(len(group["params"]) for group in optimizer.param_groups),
        "group_count": len(optimizer.param_groups),
        "groups": groups,
        "state": state,
        "state_tensors": state_tensors,
        "state_bytes": state_bytes,
        "state_bytes_by_dtype": bytes_by_dtype,
        "step_values": sorted(set(step_values)),
    }


def capture_training_fingerprint(module, optimizer: Optimizer | None = None, *,
                                 schedule_state: dict | None = None,
                                 data_state: dict | None = None,
                                 hash_values: bool = True,
                                 include_rng: bool = True) -> dict:
    """Capture one complete interruption-boundary fingerprint.

    Parameters, optimizer moments/counters, nonpersistent buffers (including
    rotary) and train/eval state are recorded with dtype/device/storage and
    numeric norms. Schedule and batch-cursor values are canonicalized as JSON.
    """
    return {
        "format": FINGERPRINT_FORMAT,
        "hash_values": bool(hash_values),
        "train_mode": fingerprint_train_mode(module),
        "buffers": fingerprint_buffers(module, hash_values=hash_values),
        "parameters": fingerprint_parameters(module, hash_values=hash_values),
        "optimizer": (fingerprint_optimizer(optimizer, module)
                      if optimizer is not None else None),
        "rng": fingerprint_rng() if include_rng else None,
        "schedule_state": (_json_value(schedule_state, "schedule_state")
                           if schedule_state is not None else None),
        "data_state": (_json_value(data_state, "data_state")
                       if data_state is not None else None),
    }


def _compare_tensor_entries(reference: dict, actual: dict) -> tuple[list[str], dict]:
    fields = []
    for field in ("dtype", "device", "shape", "numel", "storage_bytes",
                  "storage_offset", "stride", "is_contiguous"):
        if reference.get(field) != actual.get(field):
            fields.append(field)
    if (reference.get("value_sha256") is not None
            and actual.get("value_sha256") is not None
            and reference.get("value_sha256") != actual.get("value_sha256")):
        fields.append("value_sha256")
    metrics = {}
    for name in ("l1_norm", "l2_norm", "max_abs"):
        metrics[f"{name}_delta"] = _finite_float(
            abs(float(actual[name]) - float(reference[name]))
            if reference.get(name) is not None and actual.get(name) is not None else 0.0)
    reference_l2 = reference.get("l2_norm")
    actual_l2 = actual.get("l2_norm")
    metrics["relative_l2_delta"] = (
        _finite_float(abs(actual_l2 - reference_l2) / abs(reference_l2))
        if reference_l2 not in (None, 0.0) and actual_l2 is not None else None)
    if reference.get("finite") != actual.get("finite"):
        fields.append("finite")
    return fields, metrics


def _compare_named_entries(section: str, reference: dict, actual: dict,
                           differences: list, metrics: list) -> None:
    for name in sorted(set(reference) | set(actual)):
        if name not in reference:
            differences.append({"section": section, "name": name, "fields": ["missing"]})
            continue
        if name not in actual:
            differences.append({"section": section, "name": name, "fields": ["extra"]})
            continue
        fields, values = _compare_tensor_entries(reference[name], actual[name])
        if fields:
            differences.append({"section": section, "name": name, "fields": fields})
        metric = {"section": section, "name": name}
        metric.update(values)
        metrics.append(metric)


def _compare_optimizer_state(reference: dict, actual: dict,
                             differences: list, metrics: list) -> None:
    for name in sorted(set(reference) | set(actual)):
        if name not in reference:
            differences.append({"section": "optimizer", "name": name,
                                "fields": ["missing"]})
            continue
        if name not in actual:
            differences.append({"section": "optimizer", "name": name,
                                "fields": ["extra"]})
            continue
        left_state, right_state = reference[name], actual[name]
        for key in sorted(set(left_state) | set(right_state)):
            entry_name = f"{name}.{key}"
            if key not in left_state:
                differences.append({"section": "optimizer", "name": entry_name,
                                    "fields": ["missing"]})
                continue
            if key not in right_state:
                differences.append({"section": "optimizer", "name": entry_name,
                                    "fields": ["extra"]})
                continue
            left_entry, right_entry = left_state[key], right_state[key]
            if isinstance(left_entry, dict) and isinstance(right_entry, dict):
                fields, values = _compare_tensor_entries(left_entry, right_entry)
                metric = {"section": "optimizer", "name": entry_name}
                metric.update(values)
                metrics.append(metric)
            else:
                fields = [] if left_entry == right_entry else ["value"]
            if fields:
                differences.append({"section": "optimizer", "name": entry_name,
                                    "fields": fields})


def _json_difference_path(reference, actual, prefix: str = "") -> str | None:
    if isinstance(reference, dict) and isinstance(actual, dict):
        for key in sorted(set(reference) | set(actual)):
            if key not in reference or key not in actual:
                return f"{prefix}.{key}" if prefix else str(key)
            path = _json_difference_path(reference[key], actual[key],
                                         f"{prefix}.{key}" if prefix else str(key))
            if path is not None:
                return path
        return None
    if isinstance(reference, list) and isinstance(actual, list):
        if len(reference) != len(actual):
            return f"{prefix}.length" if prefix else "length"
        for index, (left, right) in enumerate(zip(reference, actual)):
            path = _json_difference_path(left, right,
                                         f"{prefix}.{index}" if prefix else str(index))
            if path is not None:
                return path
        return None
    if type(reference) is not type(actual) or reference != actual:
        return prefix or "value"
    return None


def compare_training_fingerprints(reference: dict, actual: dict) -> dict:
    """Localize the earliest difference between two boundary fingerprints."""
    if (not isinstance(reference, dict) or not isinstance(actual, dict)
            or reference.get("format") != FINGERPRINT_FORMAT
            or actual.get("format") != FINGERPRINT_FORMAT):
        raise ValueError("Both values must be training fingerprints")
    differences: list[dict] = []
    metrics: list[dict] = []
    section_counts: dict[str, int] = {}

    def add(section, name, fields):
        differences.append({"section": section, "name": name, "fields": fields})

    reference_mode = reference.get("train_mode") or {}
    actual_mode = actual.get("train_mode") or {}
    mode_fields = [field for field in ("model_training", "mode_sha256",
                                       "module_count", "training_modules")
                   if reference_mode.get(field) != actual_mode.get(field)]
    if mode_fields:
        add("train_mode", "module_modes", mode_fields)
        for index, (left, right) in enumerate(zip(reference_mode.get("modules", []),
                                                  actual_mode.get("modules", []))):
            if left != right:
                add("train_mode", left.get("name", str(index)), ["training"])
                break

    reference_buffers = (reference.get("buffers") or {}).get("buffers", {})
    actual_buffers = (actual.get("buffers") or {}).get("buffers", {})
    _compare_named_entries("buffers", reference_buffers, actual_buffers,
                           differences, metrics)
    for name in sorted(set(reference_buffers) | set(actual_buffers)):
        left = reference_buffers.get(name)
        right = actual_buffers.get(name)
        if left is not None and right is not None and left.get("persistent") != right.get("persistent"):
            add("buffers", name, ["persistent"])

    _compare_named_entries("parameters", reference.get("parameters") or {},
                           actual.get("parameters") or {}, differences, metrics)

    reference_optimizer = reference.get("optimizer") or {}
    actual_optimizer = actual.get("optimizer") or {}
    if bool(reference_optimizer) != bool(actual_optimizer):
        add("optimizer", "presence", ["present"])
    elif reference_optimizer:
        for field in ("parameter_count", "group_count", "state_tensors",
                      "state_bytes", "state_bytes_by_dtype", "step_values"):
            if reference_optimizer.get(field) != actual_optimizer.get(field):
                add("optimizer", field, ["value"])
        for index, (left, right) in enumerate(zip(reference_optimizer.get("groups", []),
                                                  actual_optimizer.get("groups", []))):
            for field in ("group_name", "algorithm", "lr", "betas", "eps",
                          "weight_decay", "fused", "parameter_names",
                          "tensor_count", "parameter_count"):
                if left.get(field) != right.get(field):
                    add("optimizer", f"group_{index}", [field])
        _compare_optimizer_state(reference_optimizer.get("state") or {},
                                 actual_optimizer.get("state") or {},
                                 differences, metrics)

    reference_rng = reference.get("rng") or {}
    actual_rng = actual.get("rng") or {}
    if bool(reference_rng) != bool(actual_rng):
        add("rng", "presence", ["present"])
    else:
        if reference_rng.get("python_sha256") != actual_rng.get("python_sha256"):
            add("rng", "python", ["python_sha256"])
        for key in ("torch_cpu", "torch_mps"):
            left, right = reference_rng.get(key), actual_rng.get(key)
            if bool(left) != bool(right):
                add("rng", key, ["presence"])
            elif left and right:
                fields, values = _compare_tensor_entries(left, right)
                if fields:
                    add("rng", key, fields)
                metric = {"section": "rng", "name": key}
                metric.update(values)
                metrics.append(metric)
        left_cuda, right_cuda = reference_rng.get("torch_cuda"), actual_rng.get("torch_cuda")
        if bool(left_cuda) != bool(right_cuda):
            add("rng", "torch_cuda", ["presence"])
        elif left_cuda and right_cuda:
            if len(left_cuda) != len(right_cuda):
                add("rng", "torch_cuda", ["device_count"])
            else:
                for index, (left, right) in enumerate(zip(left_cuda, right_cuda)):
                    fields, _ = _compare_tensor_entries(left, right)
                    if fields:
                        add("rng", f"torch_cuda_{index}", fields)

    for section in ("schedule_state", "data_state"):
        if reference.get(section) != actual.get(section):
            path = _json_difference_path(reference.get(section), actual.get(section))
            add(section, path or section, ["value"])

    for difference in differences:
        section_counts[difference["section"]] = section_counts.get(
            difference["section"], 0) + 1
    section_order = ["train_mode", "buffers", "parameters", "optimizer", "rng",
                     "schedule_state", "data_state"]
    earliest = None
    for section in section_order:
        candidates = [item for item in differences if item["section"] == section]
        if candidates:
            earliest = candidates[0]
            break
    numerical = {
        "tensor_pairs": len(metrics),
        "max_l1_norm_delta": max((m["l1_norm_delta"] for m in metrics
                                  if m["l1_norm_delta"] is not None), default=0.0),
        "max_l2_norm_delta": max((m["l2_norm_delta"] for m in metrics
                                  if m["l2_norm_delta"] is not None), default=0.0),
        "max_relative_l2_delta": max((m["relative_l2_delta"] for m in metrics
                                      if m["relative_l2_delta"] is not None), default=0.0),
    }
    return {
        "format": "replica.training_fingerprint_comparison.v1",
        "matched": not differences,
        "difference_count": len(differences),
        "section_differences": section_counts,
        "earliest_divergence": earliest,
        "differences": differences[:64],
        "differences_truncated": len(differences) > 64,
        "numerical": numerical,
    }


def _aggregate_norms(entries: dict) -> dict:
    l2_squared = 0.0
    l1_total = 0.0
    max_abs = 0.0
    for entry in entries.values():
        if entry.get("l2_norm") is not None:
            l2_squared += float(entry["l2_norm"]) ** 2
        if entry.get("l1_norm") is not None:
            l1_total += float(entry["l1_norm"])
        if entry.get("max_abs") is not None:
            max_abs = max(max_abs, float(entry["max_abs"]))
    return {
        "l1_norm": _finite_float(l1_total),
        "l2_norm": _finite_float(math.sqrt(l2_squared)),
        "max_abs": _finite_float(max_abs),
    }


def summarize_parameter_norms(module) -> dict:
    entries = {name: _norm_summary(parameter)
               for name, parameter in module.named_parameters()}
    result = {"tensor_count": len(entries),
              "parameter_count": sum(int(entry["numel"]) for entry in entries.values()),
              "per_tensor": entries}
    result.update(_aggregate_norms(entries))
    return result


def summarize_gradient_norms(module) -> dict:
    entries = {}
    missing = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        else:
            entries[name] = _norm_summary(parameter.grad)
    result = {"present_tensors": len(entries), "missing_tensors": missing,
              "missing_count": len(missing), "per_tensor": entries}
    result.update(_aggregate_norms(entries))
    return result


def summarize_optimizer_norms(optimizer: Optimizer, module=None) -> dict:
    names = _parameter_name_map(module) if module is not None else {}
    entries: dict[str, dict] = {}
    counters = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter), f"parameter_{len(entries)}")
            parameter_state = optimizer.state.get(parameter, {})
            if not parameter_state:
                continue
            moment_entries = {}
            for key, value in sorted(parameter_state.items()):
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    moment_entries[key] = _norm_summary(value)
                elif isinstance(value, torch.Tensor):
                    counters[f"{name}.{key}"] = _counter_value(value)
                else:
                    counters[f"{name}.{key}"] = value
            if moment_entries:
                entries[name] = moment_entries
    flat_entries = {f"{name}.{key}": value
                    for name, state in entries.items()
                    for key, value in state.items()}
    result = {"state_tensors": len(flat_entries), "per_tensor": entries,
              "counters": counters}
    result.update(_aggregate_norms(flat_entries))
    return result


def summarize_training_norms(module, optimizer: Optimizer | None = None) -> dict:
    """Per-update numeric summary for parameters, gradients and moments."""
    return {
        "format": "replica.training_norms.v1",
        "parameters": summarize_parameter_norms(module),
        "gradients": summarize_gradient_norms(module),
        "optimizer": (summarize_optimizer_norms(optimizer, module)
                      if optimizer is not None else None),
    }


def swap_used_bytes() -> int | None:
    """Best-effort host swap usage; instrumentation never fails a run."""
    if sys.platform != "darwin":
        return None
    try:
        text = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                              capture_output=True, text=True, timeout=5,
                              check=False).stdout
        match = re.search(r"used\s*=\s*([\d.]+)([BKMGTP])", text, re.IGNORECASE)
        if match is None:
            return None
        return int(float(match.group(1)) * 1024 ** "BKMGTP".index(match.group(2).upper()))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _memory_bytes(device) -> dict:
    device_type = torch.device(device).type
    if device_type == "mps" and torch.backends.mps.is_available():
        return {"allocated_bytes": int(torch.mps.current_allocated_memory()),
                "driver_bytes": int(torch.mps.driver_allocated_memory())}
    if device_type == "cuda" and torch.cuda.is_available():
        return {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "driver_bytes": int(torch.cuda.memory_reserved(device))}
    return {"allocated_bytes": None, "driver_bytes": None}


def _tensor_inventory_bytes(module) -> dict:
    total = 0
    by_dtype: dict[str, int] = {}
    count = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        size = int(tensor.numel() * tensor.element_size())
        total += size
        count += 1
        dtype_name = str(tensor.dtype)
        by_dtype[dtype_name] = by_dtype.get(dtype_name, 0) + size
    return {"tensor_count": count, "bytes": total, "bytes_by_dtype": by_dtype}


def _optimizer_inventory(optimizer: Optimizer) -> dict:
    total = 0
    count = 0
    by_dtype: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                size = int(value.numel() * value.element_size())
                total += size
                count += 1
                dtype_name = str(value.dtype)
                by_dtype[dtype_name] = by_dtype.get(dtype_name, 0) + size
    return {"tensor_count": count, "bytes": total, "bytes_by_dtype": by_dtype}


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
        "rng_fingerprint": _rng_fingerprint(rng),
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
    map_location = target_device if target_device is not None else "cpu"
    payload = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "optimizer", "rng"}:
        raise ValueError("Invalid runtime checkpoint payload")
    if isinstance(payload.get("rng"), dict):
        for key in ("torch_cpu", "torch_mps"):
            value = payload["rng"].get(key)
            if isinstance(value, torch.Tensor):
                payload["rng"][key] = value.cpu()
        if isinstance(payload["rng"].get("torch_cuda"), list):
            payload["rng"]["torch_cuda"] = [value.cpu() for value in payload["rng"]["torch_cuda"]]
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
                             restore_rng=True, strict_rng=True, accounting=None):
    """Materialize a checkpoint and restore runtime state with bounded residency.

    Meta-device construction owns architecture and parameter names without model
    storage. CPU uses the normal checkpoint loader for exact fixtures. Native
    resumes materialize device-resident weights first, then restore optimizer,
    RNG, scheduler and cursor state directly into the real optimizer.

    ``accounting`` may be a caller-owned dictionary. It receives measured model
    copy, optimizer migration, temporary tensor and host swap facts; accounting
    never changes restoration behavior.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    def note(stage, **values):
        if isinstance(accounting, dict):
            accounting.setdefault("stages", {})[stage] = {
                "swap_used_bytes": swap_used_bytes(),
                "memory": _memory_bytes(device),
                **values,
            }

    if isinstance(accounting, dict):
        accounting.update({
            "format": "replica.resume_accounting.v1",
            "device": str(device),
            "full_cpu_model_copy": False,
            "swap_used_bytes_start": swap_used_bytes(),
            "memory_start": _memory_bytes(device),
        })
    note("start")
    base_path = Path(base_path)
    with torch.device("meta"):
        config = AutoConfig.from_pretrained(base_path)
        base = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        model = adapter_factory(base)
        optimizer = optimizer_factory(model)
    base_rates = [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []
    del optimizer
    note("meta_construction")
    if torch.device(device).type == "cpu":
        if isinstance(accounting, dict):
            accounting["path"] = "cpu"
        base = AutoModelForCausalLM.from_pretrained(
            base_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
        model = adapter_factory(base).to(device)
        note("model_loaded", model=_tensor_inventory_bytes(base))
        if isinstance(accounting, dict):
            accounting["model"] = {
                "loaded": _tensor_inventory_bytes(base),
                "temporary_checkpoint_tensor_bytes": 0,
                "temporary_checkpoint_tensor_count": 0,
                "temporary_tensors_released_before_optimizer": True,
            }
        actual_optimizer = optimizer_factory(model)
        restored = load_runtime_state(
            runtime_path, actual_optimizer, run_contract=run_contract,
            restore_rng=restore_rng, strict_rng=strict_rng, target_device="cpu")
        if isinstance(accounting, dict):
            accounting["optimizer"] = {
                "staged_device": "cpu",
                "staged": _optimizer_inventory(actual_optimizer),
                "migration": None,
            }
        note("runtime_restored", optimizer=_optimizer_inventory(actual_optimizer))
        if getattr(model, "augmented", False):
            model.load_sidecars(base_path.parent / "sidecars.safetensors")
        note("complete")
        if isinstance(accounting, dict):
            end_swap = swap_used_bytes()
            accounting["swap_used_bytes_end"] = end_swap
            accounting["memory_end"] = _memory_bytes(device)
            start_swap = accounting.get("swap_used_bytes_start")
            accounting["swap_growth_bytes"] = (
                end_swap - start_swap
                if end_swap is not None and start_swap is not None else None)
        return base, model, actual_optimizer, restored, base_rates
    if isinstance(accounting, dict):
        accounting["path"] = "native"
    model.to_empty(device=device)
    # Non-persistent Llama rotary buffers must be rebuilt after to_empty.
    rotary = getattr(base.model, "rotary_emb", None)
    if rotary is not None:
        base.model.rotary_emb = type(rotary)(base.config).to(device)
    note("empty_model_on_device", model=_tensor_inventory_bytes(base))
    index_path = base_path / "model.safetensors.index.json"
    single_path = base_path / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Invalid sharded checkpoint weight map")
        shard_names = sorted(set(weight_map.values()))
    elif single_path.is_file():
        shard_names = [single_path.name]
    else:
        raise ValueError("Checkpoint has no supported safetensors weight inventory")
    weights = {}
    weight_inventory = {"tensor_count": 0, "bytes": 0, "bytes_by_dtype": {}}
    for shard_name in shard_names:
        shard_relative = Path(shard_name)
        if shard_relative.is_absolute() or ".." in shard_relative.parts:
            raise ValueError("Checkpoint shard path is invalid")
        shard_path = base_path / shard_relative
        if not shard_path.is_file():
            raise ValueError(f"Missing checkpoint shard: {shard_name}")
        shard = load_file(str(shard_path), device=device)
        if set(weights) & set(shard):
            raise ValueError("Duplicate checkpoint tensor names")
        weights.update(shard)
        for tensor in shard.values():
            size = int(tensor.numel() * tensor.element_size())
            weight_inventory["tensor_count"] += 1
            weight_inventory["bytes"] += size
            dtype_name = str(tensor.dtype)
            weight_inventory["bytes_by_dtype"][dtype_name] = (
                weight_inventory["bytes_by_dtype"].get(dtype_name, 0) + size)
    note("checkpoint_tensors_loaded", loaded=weight_inventory,
         temporary_weight_bytes=weight_inventory["bytes"])
    expected_device = torch.device(device)
    if any(value.device.type != expected_device.type
           or (expected_device.index is not None
               and value.device.index != expected_device.index)
           for value in weights.values()):
        raise RuntimeError("Checkpoint tensor device differs from requested resume device")
    base.load_state_dict(weights, assign=True, strict=True)
    model._apply(lambda tensor: tensor)
    del weights
    if isinstance(accounting, dict):
        accounting["model"] = {
            "loaded": _tensor_inventory_bytes(base),
            "temporary_checkpoint_tensor_bytes": weight_inventory["bytes"],
            "temporary_checkpoint_tensor_count": weight_inventory["tensor_count"],
            "checkpoint_tensors_adopted_by_model": True,
            "temporary_tensors_released_before_optimizer": True,
        }
    note("model_restored")
    actual_optimizer = optimizer_factory(model)
    gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    elif torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    note("optimizer_constructed")
    restored = load_runtime_state(
        runtime_path, actual_optimizer, run_contract=run_contract,
        restore_rng=restore_rng, strict_rng=strict_rng, target_device="cpu")
    gc.collect()
    staged_inventory = _optimizer_inventory(actual_optimizer)
    if isinstance(accounting, dict):
        accounting["optimizer"] = {
            "staged_device": "cpu",
            "staged": staged_inventory,
            "migration": {
                "migrated_tensor_count": 0,
                "migrated_bytes": 0,
                "migrated_bytes_by_dtype": {},
                "cache_release_count": 0,
                "gc_collect_count": 0,
                "max_single_tensor_bytes": 0,
                "peak_driver_bytes": None,
            },
        }
    note("optimizer_runtime_staged_on_cpu", optimizer=staged_inventory)
    migration = accounting.get("optimizer", {}).get("migration") if isinstance(accounting, dict) else None
    peak_driver = None
    gc_count = 1
    release_count = 0
    for index, state in enumerate(actual_optimizer.state.values()):
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                size = int(value.numel() * value.element_size())
                if migration is not None:
                    migration["migrated_tensor_count"] += 1
                    migration["migrated_bytes"] += size
                    dtype_name = str(value.dtype)
                    migration["migrated_bytes_by_dtype"][dtype_name] = (
                        migration["migrated_bytes_by_dtype"].get(dtype_name, 0) + size)
                    migration["max_single_tensor_bytes"] = max(
                        migration["max_single_tensor_bytes"], size)
                state[key] = value.to(device)
                del value
                if migration is not None:
                    driver = _memory_bytes(device).get("driver_bytes")
                    if driver is not None:
                        peak_driver = driver if peak_driver is None else max(peak_driver, driver)
        if index % 8 == 0:
            gc.collect()
            gc_count += 1
            if torch.device(device).type == "mps":
                torch.mps.empty_cache()
            elif torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            release_count += 1
    if migration is not None:
        migration["cache_release_count"] = release_count
        migration["gc_collect_count"] = gc_count
        migration["peak_driver_bytes"] = peak_driver
    note("optimizer_migrated", optimizer=_optimizer_inventory(actual_optimizer))
    if getattr(model, "augmented", False):
        model.load_sidecars(base_path.parent / "sidecars.safetensors")
    note("complete")
    if isinstance(accounting, dict):
        end_swap = swap_used_bytes()
        accounting["swap_used_bytes_end"] = end_swap
        accounting["memory_end"] = _memory_bytes(device)
        start_swap = accounting.get("swap_used_bytes_start")
        accounting["swap_growth_bytes"] = (
            end_swap - start_swap if end_swap is not None and start_swap is not None else None)
        accounting["resident_estimate"] = {
            "model_bytes": accounting.get("model", {}).get("loaded", {}).get("bytes"),
            "staged_cpu_optimizer_bytes": staged_inventory["bytes"],
            "temporary_checkpoint_tensor_bytes": weight_inventory["bytes"],
            "checkpoint_tensors_are_model_owned_after_assignment": True,
            "note": ("Checkpoint tensors are adopted by the model; staged CPU optimizer "
                     "bytes are additional while migration runs. MPS driver and CPU "
                     "staging overlap in unified memory; do not sum as physical use"),
        }
    return base, model, actual_optimizer, restored, base_rates
