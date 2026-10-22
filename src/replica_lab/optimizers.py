"""Portable experimental Muon with blockwise int8 momentum and FP32 AdamW.

This is an independent, eager research implementation of the algorithm in
https://arxiv.org/html/2509.23106v2 (equations 2--3), not a packaged upstream
Muon8bit kernel. Related upstream implementations/documentation:
https://docs.axolotl.ai/docs/optimizers.html#muon_8bit
https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html

Only persistent hidden-matrix momentum is quantized. Parameters, gradients,
AdamW state, and temporary Newton--Schulz arithmetic remain FP32. This makes
CPU/MPS/CUDA comparison explicit; it does not establish full-model capacity,
numerical parity with a BF16 kernel, distributed support, or convergence.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def quantize_blockwise(momentum: Tensor, block_size: int = 2048) -> tuple[Tensor, Tensor]:
    """Return signed int8 codes and one FP32 absmax scale per flattened block."""
    if block_size < 1 or momentum.dtype != torch.float32 or momentum.numel() == 0:
        raise ValueError("nonempty FP32 momentum and positive block_size required")
    if not bool(torch.isfinite(momentum).all()):
        raise ValueError("Cannot quantize nonfinite momentum")
    flat = momentum.reshape(-1)
    count = (flat.numel() + block_size - 1) // block_size
    padded = torch.zeros(count * block_size, dtype=torch.float32, device=flat.device)
    padded[:flat.numel()].copy_(flat)
    blocks = padded.view(count, block_size)
    scales = blocks.abs().amax(dim=1) / 127.0
    # Zero blocks use scale 1 so both quantization and dequantization stay exact.
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = (blocks / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return codes.reshape(-1)[:flat.numel()].clone().view_as(momentum), scales


def dequantize_blockwise(codes: Tensor, scales: Tensor,
                        block_size: int = 2048) -> Tensor:
    """Reconstruct FP32 momentum; codes retain the original parameter shape."""
    if block_size < 1 or codes.dtype != torch.int8 or scales.dtype != torch.float32:
        raise ValueError("int8 codes, FP32 scales, and positive block_size required")
    expected = (codes.numel() + block_size - 1) // block_size
    if scales.ndim != 1 or scales.numel() != expected or codes.device != scales.device:
        raise ValueError("Quantized momentum has incompatible scale shape/device")
    return (codes.reshape(-1).to(torch.float32)
            * scales.repeat_interleave(block_size)[:codes.numel()]).view_as(codes)


def orthogonalize_fp32(matrix: Tensor, ns_steps: int = 5,
                      eps: float = 1e-7) -> Tensor:
    """Five-step quintic Newton--Schulz approximation, explicitly in FP32."""
    if matrix.ndim != 2 or matrix.dtype != torch.float32 or not 1 <= ns_steps < 100:
        raise ValueError("FP32 matrix and ns_steps in [1, 99] required")
    # Retain the declared arithmetic even if a caller surrounds step() with
    # autocast. GPU tensor storage alone does not determine matmul precision.
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        value = matrix.T if matrix.shape[0] > matrix.shape[1] else matrix
        value = value / value.norm().clamp_min(eps)
        for _ in range(ns_steps):
            gram = value @ value.T
            value = 3.4445 * value + (-4.775 * gram + 2.0315 * (gram @ gram)) @ value
    return value.T if matrix.shape[0] > matrix.shape[1] else value


def _check_hyperparameters(group: dict) -> None:
    for key in ("lr", "weight_decay"):
        if not math.isfinite(group[key]) or group[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if group["algorithm"] == "muon8bit":
        if not math.isfinite(group["momentum"]) or not 0 <= group["momentum"] < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not isinstance(group["ns_steps"], int) or not 1 <= group["ns_steps"] < 100:
            raise ValueError("ns_steps must be in [1, 99]")
        if not isinstance(group["block_size"], int) or group["block_size"] < 1:
            raise ValueError("block_size must be positive")
    elif group["algorithm"] == "adamw":
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in group["betas"]):
            raise ValueError("AdamW betas must be in [0, 1)")
        if not math.isfinite(group["eps"]) or group["eps"] <= 0:
            raise ValueError("AdamW eps must be finite and positive")
    else:
        raise ValueError("Unknown optimizer group algorithm")


class ReplicaOptimizer(Optimizer):
    """One optimizer owning disjoint Muon8bit and FP32 AdamW parameter groups.

    Muon uses momentum = beta * previous + gradient (without Nesterov),
    decoupled decay, and update scale 0.2 * sqrt(max(rows, columns)).
    AdamW uses betas=(0.9, 0.999), eps=1e-8. All trainable weights must be FP32.
    """

    def __init__(self, groups: list[dict], optimizer_name: str = "muon8bit") -> None:
        self.optimizer_name = optimizer_name
        super().__init__(groups, {})
        seen = set()
        for group in self.param_groups:
            _check_hyperparameters(group)
            for parameter in group["params"]:
                if id(parameter) in seen:
                    raise ValueError("Optimizer parameters must be unique")
                seen.add(id(parameter))
                if parameter.dtype != torch.float32:
                    raise ValueError("Experimental optimizer requires FP32 resident parameters")
                if group["algorithm"] == "muon8bit" and parameter.ndim != 2:
                    raise ValueError("Muon8bit requires hidden 2D matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject bad gradients before any group mutates its weights or state.
        for group in self.param_groups:
            _check_hyperparameters(group)
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("Experimental optimizer requires FP32 resident parameters")
                gradient = parameter.grad
                if gradient is not None and (gradient.is_sparse
                                            or not bool(torch.isfinite(gradient).all())):
                    raise ValueError("Optimizer requires finite dense gradients")
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if group["algorithm"] == "muon8bit":
                    momentum = (dequantize_blockwise(state["momentum_q8"],
                                                    state["momentum_scale"],
                                                    group["block_size"])
                                if state else torch.zeros_like(parameter))
                    momentum.mul_(group["momentum"]).add_(gradient)
                    update = orthogonalize_fp32(momentum, group["ns_steps"])
                    codes, scales = quantize_blockwise(momentum, group["block_size"])
                    state["momentum_q8"], state["momentum_scale"] = codes, scales
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    factor = 0.2 * math.sqrt(max(parameter.shape))
                    parameter.add_(update, alpha=-group["lr"] * factor)
                else:
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                     exp_avg_sq=torch.zeros_like(parameter))
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient,
                                                          value=1 - beta2)
                    first = state["exp_avg"] / (1 - beta1 ** state["step"])
                    denominator = (state["exp_avg_sq"] / (1 - beta2 ** state["step"])).sqrt()
                    denominator.add_(group["eps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(first, denominator, value=-group["lr"])
        return loss

    def audit(self) -> dict:
        """JSON-safe declared routing and actual allocated persistent state."""
        dtype_bytes: dict[str, int] = defaultdict(int)
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, Tensor):
                    dtype_bytes[str(value.dtype)] += value.numel() * value.element_size()
        groups = []
        for group in self.param_groups:
            item = {key: group[key] for key in ("group_name", "scope", "algorithm", "lr",
                                                "weight_decay", "param_names")}
            item.update(tensors=len(group["params"]),
                        parameters=sum(p.numel() for p in group["params"]))
            groups.append(item)
        return {"optimizer": self.optimizer_name, "implementation": "replica_portable_eager_v1",
                "parameter_dtype": "float32", "muon_temporary_dtype": "float32",
                "muon_nesterov": False, "muon_lr_adjustment": "match_rms_adamw",
                "momentum": next((g["momentum"] for g in self.param_groups
                                  if g["algorithm"] == "muon8bit"), None),
                "block_size": next((g["block_size"] for g in self.param_groups
                                    if g["algorithm"] == "muon8bit"), None),
                "ns_steps": next((g["ns_steps"] for g in self.param_groups
                                  if g["algorithm"] == "muon8bit"), None),
                "groups": groups, "state_tensor_bytes": sum(dtype_bytes.values()),
                "state_bytes_by_dtype": dict(dtype_bytes)}

    metadata = audit

    def state_dict(self):
        result = super().state_dict()
        result["replica_optimizer"] = {"format_version": 1, "name": self.optimizer_name}
        return result

    def load_state_dict(self, state_dict):
        """Load without Optimizer's default parameter-dtype cast of int8 state.

        Exact named routing is required. Device migration is supported; changing
        the algorithm/parameter order or block size needs a separate migration.
        """
        saved = copy.deepcopy(state_dict)
        if saved.get("replica_optimizer") != {"format_version": 1, "name": self.optimizer_name}:
            raise ValueError("Incompatible Replica optimizer checkpoint")
        saved_groups = saved["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Optimizer checkpoint parameter groups differ")
        restored_state = defaultdict(dict)
        restored_groups = []
        valid_ids = set()
        for current, old in zip(self.param_groups, saved_groups):
            for key in ("group_name", "scope", "algorithm", "param_names"):
                if old[key] != current[key]:
                    raise ValueError(f"Optimizer checkpoint {key} differs")
            if len(old["params"]) != len(current["params"]):
                raise ValueError("Optimizer checkpoint parameter counts differ")
            if old.get("block_size") != current.get("block_size"):
                raise ValueError("Optimizer checkpoint block_size differs")
            _check_hyperparameters(old)
            restored_groups.append({**old, "params": current["params"]})
            for saved_id, parameter in zip(old["params"], current["params"]):
                valid_ids.add(saved_id)
                state = saved["state"].get(saved_id, {})
                if state:
                    if current["algorithm"] == "muon8bit":
                        if set(state) != {"momentum_q8", "momentum_scale"}:
                            raise ValueError("Invalid Muon8bit checkpoint state")
                        codes, scales = state["momentum_q8"], state["momentum_scale"]
                        if codes.dtype != torch.int8 or tuple(codes.shape) != tuple(parameter.shape):
                            raise ValueError("Invalid Muon8bit checkpoint codes")
                        expected = (parameter.numel() + old["block_size"] - 1) // old["block_size"]
                        if scales.dtype != torch.float32 or tuple(scales.shape) != (expected,):
                            raise ValueError("Invalid Muon8bit checkpoint scales")
                        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
                            raise ValueError("Invalid Muon8bit checkpoint scale values")
                    else:
                        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                            raise ValueError("Invalid AdamW checkpoint state")
                        if not isinstance(state["step"], int) or state["step"] < 1:
                            raise ValueError("Invalid AdamW checkpoint step")
                        for key in ("exp_avg", "exp_avg_sq"):
                            value = state[key]
                            if value.dtype != torch.float32 or tuple(value.shape) != tuple(parameter.shape):
                                raise ValueError("Invalid AdamW checkpoint tensor")
                            if not bool(torch.isfinite(value).all()):
                                raise ValueError("Invalid AdamW checkpoint tensor values")
                    restored_state[parameter] = {
                        key: value.to(parameter.device) if isinstance(value, Tensor) else value
                        for key, value in state.items()}
        if not set(saved["state"]).issubset(valid_ids):
            raise ValueError("Optimizer checkpoint contains unowned state")
        self.__setstate__({"state": restored_state, "param_groups": restored_groups})


def build_optimizer(model: nn.Module, optimizer_name: str = "muon8bit", *,
                    lr_base: float = 1e-5, lr_new: float = 1e-4,
                    lr_muon_base: float | None = None, lr_muon_new: float | None = None,
                    momentum: float = 0.95, ns_steps: int = 5,
                    block_size: int = 2048, weight_decay: float = 0.01
                    ) -> ReplicaOptimizer | None:
    """Route every declared trainable parameter exactly once, preserving scope.

    Actual input/output embedding module IDs (including tied weights), latent
    slot-query embeddings, and all non-2D tensors use AdamW. Remaining hidden
    2D matrices use Muon8bit in that arm.
    The model's existing optimizer_groups method supplies original/new ownership.
    """
    if optimizer_name not in {"muon8bit", "adamw"}:
        raise ValueError("optimizer_name must be muon8bit or adamw")
    # This implementation uses the RMS-aligned 0.2*sqrt(max(shape)) update.
    # Unspecified Muon rates follow the corresponding AdamW scope; carrying
    # over stock original-scaled Muon's 1e-3 default can amplify full tuning.
    lr_muon_base = lr_base if lr_muon_base is None else lr_muon_base
    lr_muon_new = lr_new if lr_muon_new is None else lr_muon_new
    names = {id(p): name for name, p in model.named_parameters()}
    excluded = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            excluded.update(id(p) for p in module.parameters(recurse=False))
        # RecordEncoder slot queries are learned latent embeddings rather than
        # hidden transforms. Discover their actual IDs before grouping so this
        # also works through workspace nesting and wrapper prefixes.
        excluded.update(id(p) for name, p in module.named_parameters(recurse=False)
                        if name == "slot_queries")
    base = getattr(model, "base", model)
    for method in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(base, method, None)
        embedding = getter() if callable(getter) else None
        if embedding is not None:
            excluded.update(id(p) for p in embedding.parameters())
    declared = model.optimizer_groups(lr_base, lr_new)
    groups = []
    for ownership in declared:
        scope = ownership["group_name"]
        for algorithm in ("muon8bit", "adamw"):
            parameters = [p for p in ownership["params"]
                          if (optimizer_name == "muon8bit" and p.ndim == 2
                              and id(p) not in excluded) == (algorithm == "muon8bit")]
            if not parameters:
                continue
            lr = ((lr_muon_base if scope == "original" else lr_muon_new)
                  if algorithm == "muon8bit" else ownership["lr"])
            groups.append({"params": parameters, "param_names": [names[id(p)] for p in parameters],
                           "scope": scope, "group_name": f"{scope}_{algorithm}",
                           "algorithm": algorithm, "lr": lr, "weight_decay": weight_decay,
                           "momentum": momentum, "ns_steps": ns_steps, "block_size": block_size,
                           "betas": (0.9, 0.999), "eps": 1e-8})
    got = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(got) != len(set(got)) or set(got) != expected:
        raise RuntimeError("Optimizer ownership is incomplete or duplicated")
    return ReplicaOptimizer(groups, optimizer_name) if groups else None
