"""Measured diagnostic state and live producer boundaries; no model loading."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

import torch


def named_parameters(path):
    for owner, module in path.modules().items():
        for name, parameter in module.named_parameters():
            yield f"{owner}.{name}", parameter


def digest(value):
    """Hash types, shapes, dtypes and exact bytes, including optimizer clocks."""
    h = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            h.update(type(item).__name__.encode())
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def capture_state(path):
    uses_mps = any(parameter.device.type == "mps" for _, parameter in named_parameters(path))
    return {
        "parameters": {name: p.detach().clone() for name, p in named_parameters(path)},
        "buffers": {f"{owner}.{name}": b.detach().clone()
                    for owner, module in path.modules().items() for name, b in module.named_buffers()},
        "optimizer": copy.deepcopy(path.optimizer.state_dict()),
        "module_modes": {f"{owner}.{name}": child.training
                         for owner, module in path.modules().items()
                         for name, child in module.named_modules()},
        "cpu_rng": torch.get_rng_state().clone(),
        "mps_rng": torch.mps.get_rng_state().clone() if uses_mps else None,
        "python_rng": random.getstate(),
    }


def state_digests(state):
    result = {key: digest(value) for key, value in state.items()}
    optimizer = state["optimizer"]["state"]
    result["optimizer_moments"] = digest({key: {k: v for k, v in entry.items() if k != "step"}
                                           for key, entry in optimizer.items()})
    result["optimizer_steps"] = digest({key: entry.get("step") for key, entry in optimizer.items()})
    return result


def verify_state(path, state):
    expected, observed = state_digests(state), state_digests(capture_state(path))
    equality = {key: observed[key] == value for key, value in expected.items()}
    if not all(equality.values()):
        raise ValueError("complete state restoration mismatch: " + str(equality))
    return {"expected": expected, "observed": observed, "equal": equality,
            "mps_rng_applicable": state["mps_rng"] is not None}


def restore_state(path, state):
    with torch.no_grad():
        for name, parameter in named_parameters(path):
            parameter.copy_(state["parameters"][name])
        for owner, module in path.modules().items():
            for name, buffer in module.named_buffers():
                buffer.copy_(state["buffers"][f"{owner}.{name}"])
            for name, child in module.named_modules():
                child.training = state["module_modes"][f"{owner}.{name}"]
    path.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    path.optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng"])
    if state["mps_rng"] is not None:
        torch.mps.set_rng_state(state["mps_rng"])
    random.setstate(state["python_rng"])
    # Fused kernels may not increment tensor version counters.
    if hasattr(path, "_release_cache"):
        path._release_cache = None
    return verify_state(path, state)


def _cpu_float64(tensor):
    # MPS lacks float64; combined .to(device="cpu", dtype=torch.float64) leaves
    # the destination CPU float64 buffer uninitialized on MPS. Copy to CPU first.
    return tensor.detach().to(device="cpu").to(dtype=torch.float64)


def tensor_statistics(tensors):
    values = [_cpu_float64(tensor) for tensor in tensors]
    l2 = math.sqrt(sum(float(v.square().sum()) for v in values))
    max_abs = max((float(v.abs().max()) for v in values if v.numel()), default=0.0)
    return {"tensor_count": len(values), "elements": sum(v.numel() for v in values),
            "l2": l2,
            "max_abs": max_abs,
            "nonzero": sum(int(v.count_nonzero()) for v in values),
            "finite": all(bool(v.isfinite().all()) for v in values) and math.isfinite(l2) and math.isfinite(max_abs)}


def module_statistics(path, *, before=None):
    result = {}
    for owner, module in path.modules().items():
        tensors = []
        for name, parameter in module.named_parameters():
            value = (_cpu_float64(parameter) - _cpu_float64(before[f"{owner}.{name}"])
                     if before is not None else parameter.grad)
            if value is not None:
                tensors.append(value)
        result[owner] = tensor_statistics(tensors)
    return result


def live_values(engine, spans, scope):
    # FrozenSourceFeatures validates backbone/tokenizer identity on every call.
    # Only that frozen boundary may cache; compression is recomputed every time.
    with engine.measured("live_producer_slots") as event:
        before = engine.path.cost["producer_calls"]
        batch = engine.path.training_values(spans, scope=scope)
        event["producer_calls"] = engine.path.cost["producer_calls"] - before
        event["slot_elements"] = batch.slots.numel()
    return batch


def prediction_evidence(logits, encoding):
    begin = encoding.metadata["prompt_tokens"] - 1
    end = encoding.metadata["terminal_prediction_position"] + 1
    predicted = logits[0, begin:end].argmax(-1).detach().cpu().tolist()
    expected = encoding.labels[0, begin:end].detach().cpu().tolist()
    return {"scope": "full_vocabulary_argmax_under_candidate_teacher_forcing",
            "predicted_token_ids": predicted, "expected_token_ids": expected,
            "first_token_correct": predicted[0] == expected[0],
            "answer_tokens_all_correct": predicted[:-1] == expected[:-1],
            "eos_correct": predicted[-1] == expected[-1],
            "complete_teacher_forced_correct": predicted == expected,
            "free_generation_measured": False}
