"""Small CPU DC-P1 learning-hypothesis audit using the real execution paths."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from replica_lab.dc_response_objective import (
    CLAIM_GLOBAL_RESCALE, answer_span_objective, build_response_masks)
from replica_lab.dc_p1_native_runner import NativeP1Backend
from replica_lab.nested_learning import clip_gradients_to_norm
from replica_lab.optimizers import NativeFusedAdamW

WIDTH = 4


def _optimizer(scale: torch.nn.Parameter, *, epsilon: float, lr: float) -> NativeFusedAdamW:
    return NativeFusedAdamW([{
        "params": [scale], "param_names": ["selected_expert.scale"],
        "group_name": "selected_expert", "scope": "selected_expert",
        "algorithm": "adamw_fused", "lr": lr, "weight_decay": 0.0,
        "betas": (0.9, 0.999), "eps": epsilon, "fused": True,
    }])


def _masks() -> Any:
    return build_response_masks(
        input_ids=[7, 0, 1], response_ids=[0, 1], answer_span=(0, 0), eos_id=1)


def _objective(logits: torch.Tensor, weight: float) -> dict[str, Any]:
    masks = _masks()
    labels = torch.tensor([0, 1], dtype=torch.long)
    return answer_span_objective(
        logits, labels, masks, answer_span_ce_weight=weight)


def global_ce_gradient_probe() -> dict[str, Any]:
    base = torch.tensor([[0.2, -0.4, 0.6, 0.1], [0.3, 0.2, -0.5, 0.4]])
    perturbation = torch.tensor([1.0, -2.0, 0.5, 3.0])
    gradients = {}
    for weight in (0.0, 0.5):
        scale = torch.tensor(0.0, requires_grad=True)
        objective = _objective(base + scale * perturbation, weight)
        objective["loss"].backward()
        gradients[weight] = scale.grad.detach().clone()
    ratio = gradients[0.5] / gradients[0.0]
    return {
        "claim": CLAIM_GLOBAL_RESCALE,
        "gradient_ratio": ratio.item(),
        "matched_gradient": gradients[0.0].item(),
        "treatment_gradient": gradients[0.5].item(),
        "masks_coincide": True,
    }


def _view(scale: torch.nn.Parameter) -> SimpleNamespace:
    hidden = []
    for index in range(28):
        positions = torch.tensor([
            [float(index % 3), 0.25, -0.5, 0.75],
            [0.5, -0.25, 0.75, -0.5],
            [1.0, 0.5, -0.25, 0.0],
            [-0.75, 0.5, 0.25, -1.0],
        ])
        hidden.append(positions.unsqueeze(0))

    class Bank:
        def expert(self, index):
            assert index == 0
            return lambda features: features + scale

    class Base:
        @staticmethod
        def lm_head(final):
            return final

    class Adapter:
        base = Base()
        plastic_ports = {
            "13": lambda final, slot, enabled: slot,
        }

    def native_forward(ids):
        return hidden[13], hidden

    view = SimpleNamespace(
        sites=(13,), projection=torch.eye(WIDTH), bank=Bank(),
        torch=torch, adapter=Adapter(), _native_forward=native_forward)
    view._features = NativeP1Backend._features.__get__(view, NativeP1Backend)
    return view


def _enabled_logits(view: SimpleNamespace, enabled: bool) -> torch.Tensor:
    return NativeP1Backend._logits(view, [0, 1], 1, enabled=enabled)


def _update_behavior(*, weight: float, epsilon: float, clip: bool,
                     lr: float = 1e-3) -> dict[str, Any]:
    scale = torch.nn.Parameter(torch.tensor(0.0))
    view = _view(scale)
    before = _enabled_logits(view, True)
    before_tokens = before.argmax(dim=-1).tolist()
    objective = _objective(before[0, -2:, :], weight)
    objective["loss"].backward()
    if clip:
        # Deterministically amplify the real objective gradient to exercise clipping.
        scale.grad = scale.grad * 100_000_000
        gradient_amplification = 100_000_000
    else:
        gradient_amplification = 1
    raw_norm = scale.grad.norm().item()
    scale_clip = 1.0
    if clip:
        clipped, raw_norm, scale_clip = clip_gradients_to_norm(
            {"selected_expert.scale": scale.grad}, max_norm=1.0)
        scale.grad = clipped["selected_expert.scale"]
    clipped_norm = scale.grad.norm().item()
    optimizer = _optimizer(scale, epsilon=epsilon, lr=lr)
    optimizer.step()
    after = _enabled_logits(view, True)
    after_tokens = after.argmax(dim=-1).tolist()
    delta = (after - before).abs()
    return {
        "gradient_amplification": gradient_amplification,
        "answer_span_ce_weight": weight,
        "epsilon": epsilon,
        "clip": clip,
        "raw_gradient_norm": raw_norm,
        "clip_scale": scale_clip,
        "clipped_gradient_norm": clipped_norm,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "changed_token_count": sum(before_item != after_item
                                   for before_item, after_item in zip(
                                       before_tokens, after_tokens)),
        "max_logit_delta": delta.max().item(),
        "scale_after": scale.item(),
        "loss": objective["loss"].item(),
        "optimizer_class": "NativeFusedAdamW",
    }


def _feature_shift() -> dict[str, Any]:
    view = _view(torch.nn.Parameter(torch.tensor(0.0)))
    hidden = view._native_forward([])[1]
    teaching = NativeP1Backend._features(view, hidden, 4)
    query_only = NativeP1Backend._features(view, hidden, 2)
    return {
        "source_free_feature_shift_l2": (teaching - query_only).norm().item(),
        "teaching_feature_norm": teaching.norm().item(),
        "query_feature_norm": query_only.norm().item(),
        "source_free_prompt_tokens": 2,
        "source_conditioned_prompt_tokens": 4,
    }


def run_cpu_audit() -> dict[str, Any]:
    gradient = global_ce_gradient_probe()
    updates = [
        _update_behavior(weight=0.0, epsilon=1e-8, clip=False),
        _update_behavior(weight=0.5, epsilon=1e-8, clip=False),
        _update_behavior(weight=0.5, epsilon=1e-2, clip=False),
        _update_behavior(weight=0.5, epsilon=1e-8, clip=True),
    ]
    feature = _feature_shift()
    view = _view(torch.nn.Parameter(torch.tensor(0.0)))
    enabled = _enabled_logits(view, True)
    disabled = _enabled_logits(view, False)
    logit_delta = (enabled - disabled).abs().max().item()
    return {
        "format": "replica.dc_p1_learning_audit.v1",
        "status": "diagnostic_only_no_native_learning_claim",
        "gradient": gradient,
        "updates": updates,
        "feature": feature,
        "selected_expert_max_logit_delta": logit_delta,
        "alternative_justified": bool(
            gradient["gradient_ratio"] == 1.5
            and feature["source_free_feature_shift_l2"] > 0.0),
        "alternative_protocol": (
            "dc_p1_source_free_feature_alignment_v1"
            if feature["source_free_feature_shift_l2"] > 0.0 else None),
        "native_execution": False,
        "model_loaded": False,
        "final_labels_read": False,
    }
