"""Deterministic NL0 software/math contracts for the nested-learning design.

This module implements the *smallest executable surface* required before any
NL1 model study: a selected-only plastic expert-bank transaction over
``NativeFusedAdamW``, a one-transition-per-chunk causal observation path, the
declared linear associative recurrence, a differentiable exact short AdamW
reference, and mechanical fixed/shuffled routing controls.

It is an engineering/mathematical qualification.  Nothing here trains a
native model, admits a capability, authorizes a run, or establishes nested
(fast/slow) timing or adaptive-router utility.  The declared sixteen-expert
bank shape is exposed, while tests may construct tiny widths; only the
selected expert is ever computed or updated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn

from replica_lab.optimizers import NativeFusedAdamW

__all__ = [
    "AdamWExactState",
    "BackdatedStateError",
    "CausalSnapshot",
    "ChunkOutcome",
    "DECLARED_EXPERT_COUNT",
    "DECLARED_HIDDEN_WIDTH",
    "DECLARED_WIDTH",
    "DedupLedger",
    "DuplicateObservation",
    "EXACT_REFERENCE_HORIZONS",
    "ExpertBank",
    "FrozenView",
    "FutureReadError",
    "GatedExpert",
    "MAX_VECTORS_PER_CHUNK",
    "NL0_CONFIG_PATH",
    "NestedLearningError",
    "Observation",
    "ObservationCoordinator",
    "Observer",
    "PersonalityIdentity",
    "PlasticLearningController",
    "RevisionCache",
    "RouteControl",
    "ScopeViolation",
    "StaleCache",
    "TeachingClock",
    "TransactionHooks",
    "TransactionOutcome",
    "associative_loss",
    "associative_outer_gradient",
    "associative_read",
    "associative_transition",
    "bounded_values",
    "clip_gradients_to_norm",
    "exact_adamw_step",
    "exact_adamw_unroll",
    "expert_loss_gradients",
    "first_order_identity_adaptation",
    "load_nl0_config",
    "normalized_key",
    "validate_nl0_config",
]

DECLARED_EXPERT_COUNT = 16
DECLARED_WIDTH = 256
DECLARED_HIDDEN_WIDTH = 1024
MAX_VECTORS_PER_CHUNK = 32
RHO_BOUNDS = (0.90, 1.0)
ETA_BOUNDS = (0.0, 1.0)
EXACT_REFERENCE_HORIZONS = (1, 2, 4)
NL0_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "nested_learning_nl0_v1.json"
_CHECKPOINT_FORMAT = 1
_OBSERVATION_FORMAT = 1
_ALLOWED_OBSERVATION_SOURCES = frozenset({"observation", "feedback", "correction"})


class NestedLearningError(RuntimeError):
    """Base error for an NL0 contract violation."""


class ScopeViolation(NestedLearningError):
    """A read or write named a scope the owner does not serve (fail closed)."""


class FutureReadError(NestedLearningError):
    """A read requested an observation after its availability cutoff."""


class BackdatedStateError(NestedLearningError):
    """A cutoff view or seal would mix current state with an earlier boundary."""


class DuplicateObservation(NestedLearningError):
    """An event identity was replayed with different content."""


class StaleCache(NestedLearningError):
    """A dependent cache entry predates the live state revision."""


# ---------------------------------------------------------------------------
# Small validation and hashing helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NestedLearningError(message)


def _finite_float(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be a real number")
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _float_tensor(value: Any, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    _require(tensor.is_floating_point(), f"{name} must be a floating-point tensor")
    _require(bool(torch.isfinite(tensor).all()), f"{name} must be finite")
    return tensor


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_signature(tensor: Tensor) -> str:
    contiguous = tensor.detach().to("cpu", torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Declared linear recurrence
# ---------------------------------------------------------------------------


def normalized_key(key: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return an L2-normalised key, preserving dtype and rejecting nonfinite input."""
    value = _float_tensor(key, "key")
    _require(value.ndim in (1, 2), "key must be a vector or a matrix of column keys")
    _require(eps > 0 and math.isfinite(eps), "normalization epsilon must be positive")
    norm = value.norm(dim=-1, keepdim=True)
    _require(bool((norm > 0).all()), "a zero key cannot be normalized")
    return value / norm.clamp_min(eps)


def bounded_values(values: Tensor, *, max_norm: float = 1.0) -> Tensor:
    """Clip each value vector to a finite norm bound (scaling, not element clipping)."""
    value = _float_tensor(values, "values")
    _require(max_norm > 0 and math.isfinite(max_norm), "value norm bound must be positive")
    if value.ndim == 1:
        norm = value.norm()
        scale = (max_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
        return value * scale
    norm = value.norm(dim=-1, keepdim=True)
    scale = (max_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
    return value * scale


def clip_gradients_to_norm(gradients: Mapping[str, Tensor], *,
                           max_norm: float = 1.0
                           ) -> tuple[dict[str, Tensor], float, float]:
    """Scale a selected-expert gradient set to a global FP32 norm bound.

    Returns ``(clipped, raw_norm, scale)``.  The norm is computed in FP64 for
    deterministic qualification fixtures, then applied once to every gradient.
    """
    _require(isinstance(gradients, Mapping) and len(gradients) > 0,
             "gradient clipping requires a nonempty mapping")
    bound = _finite_float(max_norm, "max_norm")
    _require(bound > 0, "gradient norm bound must be positive")
    squared = torch.zeros((), dtype=torch.float64)
    values: dict[str, Tensor] = {}
    for name, gradient in gradients.items():
        value = _float_tensor(gradient, f"gradient {name}")
        values[name] = value
        squared = squared + value.detach().to(torch.float64).pow(2).sum()
    raw_norm = float(torch.sqrt(squared))
    _require(math.isfinite(raw_norm), "global gradient norm is nonfinite")
    scale = 1.0 if raw_norm <= bound else bound / raw_norm
    clipped = {name: value * scale for name, value in values.items()}
    return clipped, raw_norm, scale


def associative_read(fast_state: Tensor, key: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Read a key from the associative state using a normalized key."""
    state = _float_tensor(fast_state, "fast_state")
    normalized = normalized_key(key, eps=eps)
    _require(state.ndim == 2, "fast_state must be a matrix")
    _require(state.shape[1] == normalized.shape[-1],
             "fast_state/key widths differ")
    return state @ normalized


def associative_loss(fast_state: Tensor, key: Tensor, value: Tensor,
                     *, eps: float = 1e-8) -> Tensor:
    """Declared loss ``0.5 * ||F k - v||^2`` with normalized keys."""
    state = _float_tensor(fast_state, "fast_state")
    target = bounded_values(value)
    predicted = associative_read(state, key, eps=eps)
    _require(predicted.shape == target.shape, "prediction and value shapes differ")
    loss = 0.5 * (predicted - target).pow(2).sum()
    _require(bool(torch.isfinite(loss)), "associative loss is nonfinite")
    return loss


def associative_outer_gradient(fast_state: Tensor, key: Tensor, value: Tensor,
                               *, eps: float = 1e-8) -> Tensor:
    """Analytic ``dL/dF = (F k - v) k^T`` for the declared loss."""
    state = _float_tensor(fast_state, "fast_state")
    normalized = normalized_key(key, eps=eps)
    target = bounded_values(value)
    predicted = state @ normalized
    _require(predicted.shape == target.shape, "prediction and value shapes differ")
    return torch.outer(predicted - target, normalized)


def associative_transition(fast_state: Tensor, key: Tensor, value: Tensor, *,
                           rho: float, eta: float, max_value_norm: float = 1.0,
                           eps: float = 1e-8) -> Tensor:
    """Declared linear control: ``F_next = rho F + eta (v - F k) k^T``.

    The error is evaluated at the pre-update state, the key is normalized and
    the value is bounded.  No optimizer is instantiated and the operation is
    differentiable with respect to ``fast_state`` and the value.
    """
    state = _float_tensor(fast_state, "fast_state")
    _require(state.ndim == 2, "fast_state must be a matrix")
    rho_value = _finite_float(rho, "rho")
    eta_value = _finite_float(eta, "eta")
    _require(RHO_BOUNDS[0] <= rho_value <= RHO_BOUNDS[1],
             f"rho must lie in {RHO_BOUNDS}")
    _require(ETA_BOUNDS[0] <= eta_value <= ETA_BOUNDS[1],
             f"eta must lie in {ETA_BOUNDS}")
    normalized = normalized_key(key, eps=eps)
    target = bounded_values(value, max_norm=max_value_norm)
    _require(state.shape[0] == target.shape[-1],
             "fast_state rows must match the value width")
    _require(state.shape[1] == normalized.shape[-1],
             "fast_state columns must match the key width")
    error = target - state @ normalized
    candidate = rho_value * state + eta_value * torch.outer(error, normalized)
    _require(bool(torch.isfinite(candidate).all()),
             "associative transition produced nonfinite state")
    return candidate


# ---------------------------------------------------------------------------
# Exact differentiable short AdamW recurrence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdamWExactState:
    """Differentiable AdamW moments and step after one exact reference update."""

    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor


def _validate_adamw_hyperparameters(lr: float, betas: Sequence[float], eps: float,
                                    weight_decay: float) -> None:
    _require(math.isfinite(lr) and lr >= 0, "lr must be finite and nonnegative")
    _require(len(betas) == 2, "betas must contain two values")
    first, second = (_finite_float(item, "beta") for item in betas)
    _require(0 <= first < 1 and 0 <= second < 1, "betas must lie in [0, 1)")
    _require(math.isfinite(eps) and eps > 0, "eps must be finite and positive")
    _require(math.isfinite(weight_decay) and weight_decay >= 0,
             "weight_decay must be finite and nonnegative")


def exact_adamw_step(param: Tensor, grad: Tensor, exp_avg: Tensor,
                     exp_avg_sq: Tensor, step: int, *, lr: float = 1e-4,
                     betas: Sequence[float] = (0.9, 0.999), eps: float = 1e-8,
                     weight_decay: float = 0.0) -> tuple[Tensor, Tensor, Tensor, AdamWExactState]:
    """One differentiable exact AdamW update with bias correction.

    Semantics match ``torch.optim.AdamW`` with decoupled weight decay.  All
    operations are out of place so a short unroll can be differentiated.
    """
    _validate_adamw_hyperparameters(lr, betas, eps, weight_decay)
    parameter = _float_tensor(param, "param")
    gradient = _float_tensor(grad, "grad")
    first = exp_avg if isinstance(exp_avg, Tensor) else torch.as_tensor(exp_avg)
    second = exp_avg_sq if isinstance(exp_avg_sq, Tensor) else torch.as_tensor(exp_avg_sq)
    _require(parameter.shape == gradient.shape == first.shape == second.shape,
             "AdamW parameter/gradient/moment shapes differ")
    _require(parameter.dtype == gradient.dtype == first.dtype == second.dtype,
             "AdamW reference requires one arithmetic dtype")
    _require(type(step) is int and step >= 1, "AdamW step must be a positive integer")

    beta1, beta2 = float(betas[0]), float(betas[1])
    if weight_decay:
        parameter = parameter * (1.0 - lr * weight_decay)
    first = beta1 * first + (1.0 - beta1) * gradient
    second = beta2 * second + (1.0 - beta2) * gradient * gradient
    first_hat = first / (1.0 - beta1 ** step)
    second_hat = second / (1.0 - beta2 ** step)
    parameter = parameter - lr * first_hat / (torch.sqrt(second_hat) + eps)
    _require(bool(torch.isfinite(parameter).all()), "AdamW reference produced nonfinite weight")
    _require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()),
             "AdamW reference produced nonfinite moments")
    return parameter, first, second, AdamWExactState(step, first, second)


def exact_adamw_unroll(param: Tensor, gradients: Sequence[Tensor], *,
                       lr: float = 1e-4, betas: Sequence[float] = (0.9, 0.999),
                       eps: float = 1e-8, weight_decay: float = 0.0
                       ) -> tuple[Tensor, tuple[AdamWExactState, ...]]:
    """Differentiable exact AdamW over a short teaching sequence."""
    _validate_adamw_hyperparameters(lr, betas, eps, weight_decay)
    source = _float_tensor(param, "param")
    _require(len(gradients) >= 1, "short unroll requires at least one gradient")
    parameter = source
    first = torch.zeros_like(source)
    second = torch.zeros_like(source)
    states: list[AdamWExactState] = []
    for index, gradient in enumerate(gradients, start=1):
        parameter, first, second, state = exact_adamw_step(
            parameter, gradient, first, second, index, lr=lr, betas=betas,
            eps=eps, weight_decay=weight_decay)
        states.append(state)
    return parameter, tuple(states)


def first_order_identity_adaptation(adapted_gradient: Tensor) -> Tensor:
    """Return the declared first-order identity-adaptation estimator.

    The estimator maps the future-query gradient evaluated at adapted
    parameters back to the initialization by an identity Jacobian.  It omits
    derivatives of the AdamW moments, bias correction, learning rate and
    teaching-feature construction; it is not an exact meta-gradient.
    """
    return _float_tensor(adapted_gradient, "adapted_gradient").clone()


# ---------------------------------------------------------------------------
# Personality identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonalityIdentity:
    """Stable, versioned conditioning identity; never rewritten by learning."""

    subject_sha: str
    snapshot_digest: str
    version: int = 1
    mode: str = "declared"

    def __post_init__(self) -> None:
        _require(len(self.subject_sha) == 64 and self.subject_sha.isalnum(),
                 "personality subject digest must be a 64-character hex digest")
        _require(len(self.snapshot_digest) == 64 and self.snapshot_digest.isalnum(),
                 "personality snapshot digest must be a 64-character hex digest")
        _require(type(self.version) is int and self.version >= 1,
                 "personality version must be a positive integer")

    @classmethod
    def default(cls, *, snapshot: str = "pending_user_input") -> "PersonalityIdentity":
        subject = hashlib.sha256(b"unset-subject").hexdigest()
        digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
        return cls(subject_sha=subject, snapshot_digest=digest, version=1)

    def key(self) -> str:
        return (f"id/{self.subject_sha}/{self.snapshot_digest}/"
                f"{self.mode}/{self.version}")

    def to_dict(self) -> dict:
        return {"subject_sha": self.subject_sha,
                "snapshot_digest": self.snapshot_digest,
                "version": self.version, "mode": self.mode}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersonalityIdentity":
        _require(isinstance(payload, Mapping), "Invalid personality payload")
        return cls(subject_sha=str(payload["subject_sha"]),
                   snapshot_digest=str(payload["snapshot_digest"]),
                   version=int(payload["version"]), mode=str(payload["mode"]))


class RevisionCache:
    """Revision-bound dependent cache; entries from an old revision are stale."""

    def __init__(self, *, revision: int = 0) -> None:
        _require(type(revision) is int and revision >= 0, "cache revision must be nonnegative")
        self._revision = revision
        self._entries: dict[Any, tuple[int, Any]] = {}
        self._invalidated: set[Any] = set()

    @property
    def revision(self) -> int:
        return self._revision

    def bind(self, revision: int) -> None:
        _require(type(revision) is int and revision >= 0, "cache revision must be nonnegative")
        self._revision = revision

    def put(self, key: Any, value: Any) -> None:
        self._entries[key] = (self._revision, value)
        self._invalidated.discard(key)

    def get(self, key: Any) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            if key in self._invalidated:
                raise StaleCache(
                    f"cache entry {key!r} was invalidated by a state publication")
            raise KeyError(key)
        revision, value = entry
        if revision != self._revision:
            raise StaleCache(
                f"cache entry {key!r} belongs to revision {revision}, "
                f"live revision is {self._revision}")
        return value

    def invalidate(self, revision: int) -> None:
        _require(revision >= self._revision,
                 "invalidation revision cannot move backwards")
        self._invalidated.update(self._entries)
        self._entries.clear()
        self._revision = revision

    def clear(self) -> None:
        self._invalidated.update(self._entries)
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Selected-only plastic expert bank
# ---------------------------------------------------------------------------


class GatedExpert(nn.Module):
    """Tiny bias-free gated MLP with the declared three matrices."""

    def __init__(self, width: int, hidden_width: int, *,
                 generator: torch.Generator | None = None) -> None:
        super().__init__()
        _require(width > 0 and hidden_width > 0, "expert widths must be positive")
        self.width = width
        self.hidden_width = hidden_width
        scale = 1.0 / math.sqrt(width)
        self.gate = nn.Parameter(self._draw(hidden_width, width, scale, generator))
        self.up = nn.Parameter(self._draw(hidden_width, width, scale, generator))
        self.down = nn.Parameter(self._draw(width, hidden_width, scale, generator))

    @staticmethod
    def _draw(rows: int, columns: int, scale: float,
              generator: torch.Generator | None) -> Tensor:
        values = torch.rand(rows, columns, generator=generator,
                            dtype=torch.float32, device="cpu")
        return (values * 2.0 - 1.0) * scale

    def forward(self, inputs: Tensor) -> Tensor:
        value = _float_tensor(inputs, "expert inputs")
        _require(value.shape[-1] == self.width,
                 "expert input width does not match the bank")
        gate = torch_functional.silu(value @ self.gate.T)
        up = value @ self.up.T
        return (gate * up) @ self.down.T


def expert_loss_gradients(expert: GatedExpert, features: Tensor, target: Tensor
                          ) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the selected expert's half-sum squared-error loss and gradients."""
    feature = _float_tensor(features, "features")
    expected = _float_tensor(target, "target")
    output = expert(feature)
    _require(output.shape == expected.shape, "expert output/target shapes differ")
    loss = 0.5 * (output - expected).pow(2).sum()
    _require(bool(torch.isfinite(loss)), "expert loss is nonfinite")
    gradients = torch.autograd.grad(loss, tuple(expert.parameters()))
    named = {name: gradient.detach()
             for (name, _), gradient in zip(expert.named_parameters(), gradients)}
    for name, gradient in named.items():
        _require(bool(torch.isfinite(gradient).all()),
                 f"nonfinite gradient for selected expert parameter {name}")
    return loss.detach(), named


class ExpertBank(nn.Module):
    """Sixteen-expert declared bank, with selected-only NativeFusedAdamW state.

    Construction is shape-parameterised so deterministic software fixtures can
    use tiny widths.  All experts retain FP32 parameter ownership for the whole
    lifetime of the bank; rollback copies data in place rather than replacing
    ``nn.Parameter`` objects.
    """

    def __init__(self, *, expert_count: int = DECLARED_EXPERT_COUNT,
                 width: int = DECLARED_WIDTH, hidden_width: int = DECLARED_HIDDEN_WIDTH,
                 seed: int = 0, lr: float = 1e-4,
                 betas: Sequence[float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, scope: str = "plastic") -> None:
        super().__init__()
        _require(type(expert_count) is int and expert_count > 0,
                 "expert_count must be positive")
        _require(type(width) is int and width > 0, "width must be positive")
        _require(type(hidden_width) is int and hidden_width > 0,
                 "hidden_width must be positive")
        _validate_adamw_hyperparameters(lr, betas, eps, weight_decay)
        self.expert_count = expert_count
        self.width = width
        self.hidden_width = hidden_width
        self.seed = seed
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.scope = scope
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.experts = nn.ModuleList([
            GatedExpert(width, hidden_width, generator=generator)
            for _ in range(expert_count)])
        self.expert_steps: list[int] = [0] * expert_count
        self._optimizer: NativeFusedAdamW | None = None

    @property
    def declared_shape(self) -> dict:
        return {"experts": DECLARED_EXPERT_COUNT, "width": DECLARED_WIDTH,
                "hidden_width": DECLARED_HIDDEN_WIDTH, "bias": False,
                "matrices_per_expert": 3}

    def _validate_expert(self, expert_id: int) -> None:
        _require(type(expert_id) is int and 0 <= expert_id < self.expert_count,
                 f"expert_id must be in [0, {self.expert_count})")

    def expert(self, expert_id: int) -> GatedExpert:
        self._validate_expert(expert_id)
        return self.experts[expert_id]

    @property
    def optimizer(self) -> NativeFusedAdamW:
        """Lazily construct one native fused group per declared expert."""
        if self._optimizer is None:
            groups = []
            for index, expert in enumerate(self.experts):
                names = [name for name, _ in expert.named_parameters()]
                groups.append({
                    "params": list(expert.parameters()),
                    "lr": self.lr,
                    "betas": self.betas,
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                    "algorithm": "adamw_fused",
                    "fused": True,
                    "group_name": f"expert_{index}",
                    "scope": f"{self.scope}/plastic",
                    "param_names": [f"expert_{index}.{name}" for name in names],
                })
            self._optimizer = NativeFusedAdamW(groups, optimizer_name="adamw_fused")
        return self._optimizer

    def zero_gradients(self) -> None:
        for expert in self.experts:
            for parameter in expert.parameters():
                parameter.grad = None

    def parameter_snapshot(self, expert_ids: Iterable[int]) -> dict[int, dict[str, Tensor]]:
        snapshot: dict[int, dict[str, Tensor]] = {}
        for expert_id in expert_ids:
            expert = self.expert(expert_id)
            snapshot[expert_id] = {
                name: parameter.detach().clone()
                for name, parameter in expert.named_parameters()}
        return snapshot

    def restore_parameters(self, snapshot: Mapping[int, Mapping[str, Tensor]]) -> None:
        """Restore values into the existing parameters (ownership is unchanged)."""
        for expert_id, values in snapshot.items():
            expert = self.expert(int(expert_id))
            for name, parameter in expert.named_parameters():
                parameter.data.copy_(values[name])

    def optimizer_state_snapshot(self, expert_ids: Iterable[int]) -> dict[str, dict | None]:
        optimizer = self.optimizer
        snapshot: dict[str, dict | None] = {}
        for expert_id in expert_ids:
            for name, parameter in self.expert(expert_id).named_parameters():
                state = optimizer.state.get(parameter)
                if state is None:
                    snapshot[f"{expert_id}.{name}"] = None
                else:
                    snapshot[f"{expert_id}.{name}"] = {
                        key: value.detach().clone() if isinstance(value, Tensor) else value
                        for key, value in state.items()}
        return snapshot

    def optimizer_state_for_expert(self, expert_id: int) -> dict[str, dict | None]:
        return self.optimizer_state_snapshot([expert_id])

    def restore_optimizer_state(self, snapshot: Mapping[str, Mapping[str, Any] | None]) -> None:
        """Restore moment/step values; absent state is removed exactly."""
        optimizer = self.optimizer
        for key, saved in snapshot.items():
            expert_text, name = key.split(".", 1)
            expert_id = int(expert_text)
            parameter = dict(self.expert(expert_id).named_parameters())[name]
            if saved is None:
                optimizer.state.pop(parameter, None)
                continue
            current = optimizer.state.get(parameter)
            if current is None:
                current = {}
                optimizer.state[parameter] = current
            for stale_key in set(current) - set(saved):
                current.pop(stale_key, None)
            for state_key, value in saved.items():
                if isinstance(value, Tensor):
                    existing = current.get(state_key)
                    if (isinstance(existing, Tensor)
                            and existing.shape == value.shape):
                        existing.copy_(value)
                    else:
                        current[state_key] = value.clone()
                else:
                    current[state_key] = copy.deepcopy(value)

    def assert_inactive_unchanged(
            self, inactive_ids: Iterable[int],
            parameter_snapshot: Mapping[int, Mapping[str, Tensor]],
            optimizer_snapshot: Mapping[str, Mapping[str, Any] | None]) -> None:
        """Fail closed if an unselected expert acquired gradients/state or changed."""
        optimizer = self.optimizer
        for expert_id in inactive_ids:
            expert = self.expert(expert_id)
            for name, parameter in expert.named_parameters():
                _require(parameter.grad is None,
                         f"inactive expert {expert_id} acquired a gradient")
                expected = parameter_snapshot[expert_id][name]
                _require(torch.equal(parameter.detach(), expected),
                         f"inactive expert {expert_id}.{name} changed")
                saved = optimizer_snapshot[f"{expert_id}.{name}"]
                current = optimizer.state.get(parameter)
                if saved is None:
                    _require(current is None,
                             f"inactive expert {expert_id}.{name} created optimizer state")
                else:
                    _require(current is not None,
                             f"inactive expert {expert_id}.{name} lost optimizer state")
                    _require(set(current) == set(saved),
                             f"inactive expert {expert_id}.{name} optimizer keys changed")
                    for state_key, value in saved.items():
                        if isinstance(value, Tensor):
                            _require(isinstance(current.get(state_key), Tensor)
                                     and torch.equal(current[state_key], value),
                                     f"inactive expert {expert_id} moment decayed")
                        else:
                            _require(current.get(state_key) == value,
                                     f"inactive expert {expert_id} counter changed")

    def finite_expert(self, expert_id: int) -> None:
        expert = self.expert(expert_id)
        for name, parameter in expert.named_parameters():
            _require(bool(torch.isfinite(parameter).all()),
                     f"selected expert {expert_id}.{name} is nonfinite")
            state = self.optimizer.state.get(parameter)
            if state:
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        _require(bool(torch.isfinite(value).all()),
                                 f"selected expert {expert_id} {key} is nonfinite")


# ---------------------------------------------------------------------------
# Transaction deduplication and outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionOutcome:
    """Durable result of one eligible teaching transaction."""

    status: str
    event_id: str
    fingerprint: str
    expert_id: int
    revision: int
    eligible_events: int
    steps: int
    loss: float
    retried: bool = False

    def to_dict(self) -> dict:
        return {"status": self.status, "event_id": self.event_id,
                "fingerprint": self.fingerprint, "expert_id": self.expert_id,
                "revision": self.revision, "eligible_events": self.eligible_events,
                "steps": self.steps, "loss": self.loss}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransactionOutcome":
        _require(isinstance(payload, Mapping), "Invalid transaction outcome")
        return cls(status=str(payload["status"]), event_id=str(payload["event_id"]),
                   fingerprint=str(payload["fingerprint"]),
                   expert_id=int(payload["expert_id"]),
                   revision=int(payload["revision"]),
                   eligible_events=int(payload["eligible_events"]),
                   steps=int(payload["steps"]), loss=float(payload["loss"]))


class DedupLedger:
    """Exactly-once event identity keyed by stable event ID and content digest."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, TransactionOutcome]] = {}

    def lookup(self, event_id: str) -> TransactionOutcome | None:
        record = self._records.get(event_id)
        return None if record is None else record[1]

    def require(self, event_id: str, fingerprint: str) -> TransactionOutcome | None:
        record = self._records.get(event_id)
        if record is None:
            return None
        if record[0] != fingerprint:
            raise DuplicateObservation(
                f"event {event_id!r} was already committed with a different identity")
        return record[1]

    def commit(self, event_id: str, fingerprint: str,
               outcome: TransactionOutcome) -> None:
        _require(isinstance(event_id, str) and event_id, "event_id must be nonempty")
        _require(len(fingerprint) == 64, "event fingerprint must be a SHA-256 digest")
        existing = self._records.get(event_id)
        if existing is not None:
            _require(existing[0] == fingerprint,
                     "a deduplicated event changed its identity")
            return
        self._records[event_id] = (fingerprint, outcome)

    def snapshot(self) -> dict[str, tuple[str, TransactionOutcome]]:
        return copy.deepcopy(self._records)

    def restore(self, snapshot: Mapping[str, tuple[str, TransactionOutcome]]) -> None:
        self._records = copy.deepcopy(dict(snapshot))

    def to_dict(self) -> dict:
        return {"format_version": 1, "identity": "event_id/content_sha256",
                "records": [
                    {"event_id": event_id, "fingerprint": fingerprint,
                     "outcome": outcome.to_dict()}
                    for event_id, (fingerprint, outcome) in sorted(self._records.items())]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DedupLedger":
        _require(isinstance(payload, Mapping) and payload.get("format_version") == 1,
                 "Invalid dedup ledger payload")
        ledger = cls()
        for item in payload.get("records", []):
            ledger.commit(str(item["event_id"]), str(item["fingerprint"]),
                          TransactionOutcome.from_dict(item["outcome"]))
        return ledger

    def __len__(self) -> int:
        return len(self._records)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))


@dataclass(frozen=True)
class TeachingClock:
    """One-clock fixed-routing control counting eligible teaching events."""

    eligible_events: int = 0

    def to_dict(self) -> dict:
        return {"eligible_events": self.eligible_events}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeachingClock":
        _require(isinstance(payload, Mapping), "Invalid teaching clock payload")
        value = payload.get("eligible_events")
        _require(type(value) is int and value >= 0,
                 "teaching clock events must be a nonnegative integer")
        return cls(value)


@dataclass(frozen=True)
class TransactionHooks:
    """Qualification-only fault-injection hooks; never used for storage."""

    after_optimizer_step: Callable[[str, int], None] | None = None
    before_publish: Callable[[dict], None] | None = None


def _transaction_fingerprint(*, event_id: str, expert_id: int, steps: int,
                             features: Tensor, target: Tensor, scope: str) -> str:
    payload = {
        "event_id": event_id, "expert_id": expert_id, "steps": steps,
        "scope": scope,
        "features": features.detach().to("cpu", torch.float32).tolist(),
        "target": target.detach().to("cpu", torch.float32).tolist(),
    }
    return _sha256_json(payload)


def _atomic_write(path: Path, payload: Any,
                  writer: Callable[[Any, Path], None] | None = None) -> None:
    """Write a checkpoint atomically, preserving a previous file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_text)
    os.close(descriptor)
    try:
        if writer is None:
            torch.save(payload, temporary)
        else:
            writer(payload, temporary)
        _require(temporary.exists(), "checkpoint writer produced no file")
        with open(temporary, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Causal observation coordination
# ---------------------------------------------------------------------------


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, Tensor):
        _require(value.ndim == 1 and value.numel() > 0, f"{name} must be a nonempty vector")
        values = tuple(float(item) for item in value.detach().to("cpu", torch.float64))
    else:
        _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
                 f"{name} must be a vector")
        values = tuple(value)
    _require(len(values) > 0, f"{name} must be nonempty")
    return tuple(_finite_float(item, name) for item in values)


@dataclass(frozen=True)
class Observation:
    """One sealed observation with scope and availability identity."""

    event_id: str
    scope: str
    key: tuple[float, ...]
    value: tuple[float, ...]
    available_at: float
    eligible: bool = True
    source: str = "observation"

    def __post_init__(self) -> None:
        _require(isinstance(self.event_id, str) and self.event_id,
                 "observation event_id must be nonempty")
        _require(isinstance(self.scope, str) and self.scope,
                 "observation scope must be nonempty")
        object.__setattr__(self, "key", _float_tuple(self.key, "key"))
        object.__setattr__(self, "value", _float_tuple(self.value, "value"))
        object.__setattr__(self, "available_at", _finite_float(self.available_at, "available_at"))
        _require(self.source in _ALLOWED_OBSERVATION_SOURCES,
                 f"source {self.source!r} cannot be a writable observation")
        _require(isinstance(self.eligible, bool), "eligible must be boolean")

    def key_tensor(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.tensor(self.key, dtype=dtype)

    def value_tensor(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.tensor(self.value, dtype=dtype)

    def identity(self) -> dict:
        return {"event_id": self.event_id, "scope": self.scope, "key": self.key,
                "value": self.value, "available_at": self.available_at,
                "eligible": self.eligible, "source": self.source}

    def fingerprint(self) -> str:
        return _sha256_json(self.identity())

    def to_dict(self) -> dict:
        return self.identity()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Observation":
        _require(isinstance(payload, Mapping), "Invalid observation payload")
        return cls(event_id=str(payload["event_id"]), scope=str(payload["scope"]),
                   key=tuple(payload["key"]), value=tuple(payload["value"]),
                   available_at=float(payload["available_at"]),
                   eligible=bool(payload["eligible"]), source=str(payload["source"]))


@dataclass(frozen=True)
class FrozenView:
    """Immutable decision-boundary snapshot supplied to readers and observers.

    ``cutoff`` and ``view_id`` bind the snapshot to its causal availability
    boundary and exact fast-state/observation identity.  A cutoff view is only
    produced when the live state is known not to contain a later sealed write.
    """

    scope: str
    revision: int
    observations: tuple[Observation, ...]
    fast_state: Tensor
    personality: PersonalityIdentity
    cutoff: float | None = None
    view_id: str = ""
    readonly: bool = True

    def __post_init__(self) -> None:
        _require(self.fast_state.requires_grad is False,
                 "a frozen view cannot carry an autograd graph")
        _require(bool(torch.isfinite(self.fast_state).all()),
                 "frozen view contains nonfinite state")
        if self.cutoff is not None:
            object.__setattr__(self, "cutoff",
                               _finite_float(self.cutoff, "view cutoff"))
        if not self.view_id:
            object.__setattr__(self, "view_id", _sha256_json({
                "kind": "nested_learning_frozen_view",
                "scope": self.scope,
                "revision": self.revision,
                "cutoff": self.cutoff,
                "fast_state_sha256": _tensor_signature(self.fast_state),
                "personality": self.personality.key(),
                "observations": [item.fingerprint() for item in self.observations],
                "readonly": self.readonly,
            }))


@dataclass(frozen=True)
class CausalSnapshot:
    """Scope/cutoff-limited read result with an explicit revision."""

    scope: str
    revision: int
    cutoff: float | None
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class ChunkOutcome:
    """Outcome of one sealed observation chunk (one transition at most)."""

    status: str
    chunk_id: str
    fingerprint: str
    scope: str
    sealed_at: float
    observation_ids: tuple[str, ...]
    reader_ids: tuple[str, ...]
    pre_revision: int
    post_revision: int
    transition_count: int
    aggregate_key: tuple[float, ...]
    aggregate_value: tuple[float, ...]
    retried: bool = False

    def to_dict(self) -> dict:
        return {"status": self.status, "chunk_id": self.chunk_id,
                "fingerprint": self.fingerprint, "scope": self.scope,
                "sealed_at": self.sealed_at,
                "observation_ids": list(self.observation_ids),
                "reader_ids": list(self.reader_ids),
                "pre_revision": self.pre_revision,
                "post_revision": self.post_revision,
                "transition_count": self.transition_count,
                "aggregate_key": list(self.aggregate_key),
                "aggregate_value": list(self.aggregate_value)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChunkOutcome":
        _require(isinstance(payload, Mapping), "Invalid chunk outcome payload")
        return cls(status=str(payload["status"]), chunk_id=str(payload["chunk_id"]),
                   fingerprint=str(payload["fingerprint"]),
                   scope=str(payload["scope"]), sealed_at=float(payload["sealed_at"]),
                   observation_ids=tuple(payload["observation_ids"]),
                   reader_ids=tuple(payload["reader_ids"]),
                   pre_revision=int(payload["pre_revision"]),
                   post_revision=int(payload["post_revision"]),
                   transition_count=int(payload["transition_count"]),
                   aggregate_key=tuple(float(item) for item in payload["aggregate_key"]),
                   aggregate_value=tuple(float(item) for item in payload["aggregate_value"]))


class Observer:
    """Read-only observer tap.  It receives frozen clones and has no write path."""

    read_only = True

    def __init__(self) -> None:
        self._reports: list[dict] = []

    @property
    def reports(self) -> tuple[dict, ...]:
        return tuple(self._reports)

    def observe(self, view: FrozenView) -> dict:
        _require(isinstance(view, FrozenView), "observer requires a FrozenView")
        report = {
            "scope": view.scope, "revision": view.revision,
            "cutoff": view.cutoff, "view_id": view.view_id,
            "observation_ids": [item.event_id for item in view.observations],
            "fast_state_sha256": _tensor_signature(view.fast_state),
            "personality": view.personality.key(), "readonly": view.readonly,
        }
        self._reports.append(report)
        return report


class ObservationCoordinator:
    """Causal observation chunks with exactly one fast-state transition each.

    Readers always receive the same immutable pre-chunk snapshot.  Reader count
    changes neither the reduced update nor the transition count.  Chunk
    retries return the original outcome; changed identity fails closed.  All
    sealed observations and dedup metadata are checkpointed.
    """

    def __init__(self, *, scope: str, width: int, seed: int = 0,
                 rho: float = 0.95, eta: float = 0.1,
                 max_vectors: int = MAX_VECTORS_PER_CHUNK,
                 personality: PersonalityIdentity | None = None,
                 observer: Observer | None = None,
                 spine: Observer | None = None) -> None:
        _require(isinstance(scope, str) and scope, "scope must be nonempty")
        _require(type(width) is int and width > 0, "width must be positive")
        _require(type(max_vectors) is int and 1 <= max_vectors <= MAX_VECTORS_PER_CHUNK,
                 f"max_vectors must lie in [1, {MAX_VECTORS_PER_CHUNK}]")
        _finite_float(rho, "rho")
        _finite_float(eta, "eta")
        _require(RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1], f"rho must lie in {RHO_BOUNDS}")
        _require(ETA_BOUNDS[0] <= eta <= ETA_BOUNDS[1], f"eta must lie in {ETA_BOUNDS}")
        self.scope = scope
        self.width = width
        self.seed = seed
        self.rho = float(rho)
        self.eta = float(eta)
        self.max_vectors = max_vectors
        self.personality = personality or PersonalityIdentity.default()
        self._observer: Observer | None = None
        self._spine: Observer | None = None
        self._fast_state = torch.zeros(width, width, dtype=torch.float32)
        self._observations: dict[str, Observation] = {}
        self._chunks: dict[str, tuple[str, ChunkOutcome]] = {}
        self._revision = 0
        self._transition_count = 0
        self._latest_sealed_at: float | None = None
        self._last_reader_views: dict[str, FrozenView] = {}
        self.cache = RevisionCache(revision=0)
        if observer is not None:
            self.attach_observer(observer)
        if spine is not None:
            self.attach_spine(spine)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def fast_state(self) -> Tensor:
        """Detached copy; canonical mutation must go through a sealed chunk."""
        return self._fast_state.detach().clone()

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def last_reader_views(self) -> dict[str, FrozenView]:
        """Reader-ID -> shared pre-chunk snapshot from the most recent seal."""
        return dict(self._last_reader_views)

    def attach_observer(self, observer: Observer) -> None:
        _require(isinstance(observer, Observer), "observer must be an Observer")
        self._observer = observer

    def detach_observer(self) -> None:
        self._observer = None

    def attach_spine(self, spine: Observer) -> None:
        _require(isinstance(spine, Observer), "spine must be an Observer tap")
        self._spine = spine

    def detach_spine(self) -> None:
        self._spine = None

    def _view(self, *, cutoff: float | None = None,
              revision: int | None = None,
              fast_state: Tensor | None = None) -> FrozenView:
        if cutoff is not None:
            cutoff = _finite_float(cutoff, "cutoff")
            if (self._latest_sealed_at is not None
                    and cutoff < self._latest_sealed_at):
                raise BackdatedStateError(
                    f"cutoff {cutoff} predates the latest sealed boundary "
                    f"{self._latest_sealed_at}; the live fast state would mix futures")
        available = tuple(sorted(
            (item for item in self._observations.values()
             if cutoff is None or item.available_at <= cutoff),
            key=lambda item: (item.available_at, item.event_id)))
        state = (self._fast_state if fast_state is None else fast_state)
        return FrozenView(scope=self.scope,
                          revision=self._revision if revision is None else revision,
                          observations=available,
                          fast_state=state.detach().clone(),
                          personality=self.personality,
                          cutoff=cutoff)

    def view(self, *, cutoff: float | None = None) -> FrozenView:
        return self._view(cutoff=cutoff)

    def read(self, *, scope: str, cutoff: float | None = None,
             event_id: str | None = None) -> Observation | CausalSnapshot:
        """Read observations only from this scope and availability cutoff."""
        if scope != self.scope:
            raise ScopeViolation(
                f"scope {scope!r} is not served by this coordinator ({self.scope!r})")
        if cutoff is not None:
            cutoff = _finite_float(cutoff, "cutoff")
        if event_id is not None:
            observation = self._observations.get(event_id)
            if observation is None:
                raise ScopeViolation(
                    f"event {event_id!r} is not available in scope {scope!r}")
            if cutoff is not None and observation.available_at > cutoff:
                raise FutureReadError(
                    f"event {event_id!r} becomes available at {observation.available_at}, "
                    f"after cutoff {cutoff}")
            return observation
        observations = tuple(sorted(
            (item for item in self._observations.values()
             if cutoff is None or item.available_at <= cutoff),
            key=lambda item: (item.available_at, item.event_id)))
        return CausalSnapshot(scope=self.scope, revision=self._revision,
                              cutoff=cutoff, observations=observations)

    @staticmethod
    def _chunk_fingerprint(chunk_id: str, sealed_at: float,
                           observations: Sequence[Observation]) -> str:
        return _sha256_json({
            "chunk_id": chunk_id, "sealed_at": sealed_at,
            "observations": [item.identity() for item in sorted(
                observations, key=lambda item: item.event_id)],
        })

    @staticmethod
    def reduce_observations(observations: Sequence[Observation], *,
                            width: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Reduce a chunk once into one normalized key and bounded value."""
        eligible = tuple(sorted(
            (item for item in observations if item.eligible),
            key=lambda item: item.event_id))
        _require(eligible, "no eligible observation to reduce")
        keys = []
        values = []
        for item in eligible:
            key = item.key_tensor(torch.float64)
            _require(key.shape[0] == width, "observation key width differs")
            keys.append(key / key.norm().clamp_min(1e-12))
            value = item.value_tensor(torch.float64)
            _require(value.shape[0] == width, "observation value width differs")
            values.append(value)
        aggregate_key = torch.stack(keys).sum(dim=0)
        aggregate_key = aggregate_key / aggregate_key.norm().clamp_min(1e-12)
        aggregate_value = torch.stack(values).mean(dim=0)
        value_norm = aggregate_value.norm()
        if value_norm > 1.0:
            aggregate_value = aggregate_value / value_norm
        return (tuple(float(item) for item in aggregate_key),
                tuple(float(item) for item in aggregate_value))

    def seal_observation_chunk(self, *, chunk_id: str,
                               observations: Sequence[Observation],
                               reader_ids: Sequence[str] = ("reader",),
                               sealed_at: float | None = None,
                               scope: str | None = None) -> ChunkOutcome:
        """Seal a chunk and perform at most one shared fast-state transition."""
        _require(isinstance(chunk_id, str) and chunk_id, "chunk_id must be nonempty")
        expected_scope = self.scope if scope is None else scope
        if expected_scope != self.scope:
            raise ScopeViolation(
                f"chunk scope {expected_scope!r} differs from {self.scope!r}")
        sealed = tuple(observations)
        _require(len(sealed) <= self.max_vectors,
                 f"a chunk may contain at most {self.max_vectors} vectors")
        readers = tuple(reader_ids)
        _require(readers and all(isinstance(item, str) and item for item in readers),
                 "reader_ids must be nonempty strings")
        _require(len(set(readers)) == len(readers), "reader_ids must be unique")
        for item in sealed:
            _require(isinstance(item, Observation), "chunk entries must be Observations")
            if item.scope != self.scope:
                raise ScopeViolation(
                    f"observation {item.event_id!r} belongs to {item.scope!r}")
        _require(len({item.event_id for item in sealed}) == len(sealed),
                 "a chunk cannot seal the same event twice")
        if sealed_at is None:
            _require(sealed, "an empty chunk requires an explicit sealed_at")
            sealed_at = max(item.available_at for item in sealed)
        sealed_at = _finite_float(sealed_at, "sealed_at")
        for item in sealed:
            if item.available_at > sealed_at:
                raise FutureReadError(
                    f"observation {item.event_id!r} is not available by {sealed_at}")
        fingerprint = self._chunk_fingerprint(chunk_id, sealed_at, sealed)
        previous = self._chunks.get(chunk_id)
        if previous is not None:
            if previous[0] != fingerprint:
                raise DuplicateObservation(
                    f"chunk {chunk_id!r} was already sealed with a different identity")
            return replace(previous[1], retried=True)
        for item in sealed:
            existing = self._observations.get(item.event_id)
            if existing is not None:
                raise DuplicateObservation(
                    f"observation event {item.event_id!r} was already sealed")
        if (self._latest_sealed_at is not None
                and sealed_at < self._latest_sealed_at):
            raise BackdatedStateError(
                f"seal at {sealed_at} predates the latest sealed boundary "
                f"{self._latest_sealed_at}; it would mix a later fast state")
        eligible = tuple(sorted(
            (item for item in sealed if item.eligible),
            key=lambda item: item.event_id))
        visible_change = bool(sealed)
        pre_revision = self._revision
        pre_state = self._fast_state.detach().clone()
        pre_view = FrozenView(
            scope=self.scope, revision=pre_revision,
            observations=self._view(cutoff=sealed_at).observations,
            fast_state=pre_state.clone(), personality=self.personality,
            cutoff=sealed_at)
        # Every reader observes exactly the same pre-chunk state, independent
        # of how many readers participate in this chunk.
        reader_views = {reader: pre_view for reader in readers}
        _require(all(view is reader_views[readers[0]] for view in reader_views.values()),
                 "reader views are not shared")
        if eligible:
            aggregate_key, aggregate_value = self.reduce_observations(
                eligible, width=self.width)
            candidate = associative_transition(
                self._fast_state,
                torch.tensor(aggregate_key, dtype=torch.float32),
                torch.tensor(aggregate_value, dtype=torch.float32),
                rho=self.rho, eta=self.eta)
            transitioned = 1
        else:
            aggregate_key, aggregate_value = (), ()
            candidate = self._fast_state
            transitioned = 0
        for tap in (self._observer, self._spine):
            if tap is not None:
                tap.observe(pre_view)
        post_revision = pre_revision + (1 if visible_change else 0)
        if transitioned:
            self._fast_state = candidate.detach().clone()
        for item in sealed:
            self._observations[item.event_id] = item
        self._revision = post_revision
        self._transition_count += transitioned
        self._latest_sealed_at = sealed_at
        outcome = ChunkOutcome(
            status="committed", chunk_id=chunk_id, fingerprint=fingerprint,
            scope=self.scope, sealed_at=sealed_at,
            observation_ids=tuple(sorted(item.event_id for item in sealed)),
            reader_ids=readers, pre_revision=pre_revision,
            post_revision=post_revision, transition_count=transitioned,
            aggregate_key=aggregate_key, aggregate_value=aggregate_value)
        self._chunks[chunk_id] = (fingerprint, outcome)
        self._last_reader_views = reader_views
        if visible_change:
            self.cache.invalidate(post_revision)
        return replace(outcome, retried=False)

    def _snapshot_state(self) -> dict:
        return {
            "fast_state": self._fast_state.detach().clone(),
            "observations": dict(self._observations),
            "chunks": dict(self._chunks),
            "revision": self._revision,
            "transition_count": self._transition_count,
            "latest_sealed_at": self._latest_sealed_at,
            "last_reader_views": dict(self._last_reader_views),
            "cache": self.cache,
        }

    def _restore_state(self, snapshot: Mapping[str, Any]) -> None:
        self._fast_state = snapshot["fast_state"].detach().clone()
        self._observations = dict(snapshot["observations"])
        self._chunks = dict(snapshot["chunks"])
        self._revision = int(snapshot["revision"])
        self._transition_count = int(snapshot["transition_count"])
        self._latest_sealed_at = snapshot["latest_sealed_at"]
        self._last_reader_views = dict(snapshot["last_reader_views"])
        self.cache = snapshot["cache"]

    def checkpoint_payload(self) -> dict:
        return {
            "format_version": _OBSERVATION_FORMAT,
            "kind": "nested_learning_observation_checkpoint",
            "scope": self.scope, "width": self.width, "seed": self.seed,
            "rho": self.rho, "eta": self.eta, "max_vectors": self.max_vectors,
            "revision": self._revision,
            "transition_count": self._transition_count,
            "latest_sealed_at": self._latest_sealed_at,
            "fast_state": self._fast_state.detach().clone(),
            "observations": [item.to_dict()
                             for item in sorted(self._observations.values(),
                                                key=lambda item: item.event_id)],
            "chunks": [outcome.to_dict()
                       for _, (_, outcome) in sorted(self._chunks.items())],
            "personality": self.personality.to_dict(),
        }

    def load_checkpoint(self, payload: Mapping[str, Any]) -> None:
        snapshot = self._snapshot_state()
        try:
            self._load_checkpoint_impl(payload)
        except BaseException:
            self._restore_state(snapshot)
            raise

    def _load_checkpoint_impl(self, payload: Mapping[str, Any]) -> None:
        _require(isinstance(payload, Mapping)
                 and payload.get("format_version") == _OBSERVATION_FORMAT,
                 "Invalid observation checkpoint")
        _require(payload.get("kind") == "nested_learning_observation_checkpoint",
                 "Unexpected observation checkpoint kind")
        if payload.get("scope") != self.scope:
            raise ScopeViolation("Observation checkpoint scope differs from the owner")
        _require((int(payload["width"]), int(payload["max_vectors"]))
                 == (self.width, self.max_vectors),
                 "Observation checkpoint shape differs")
        _require((float(payload["rho"]), float(payload["eta"])) == (self.rho, self.eta),
                 "Observation checkpoint recurrence constants differ")
        personality = PersonalityIdentity.from_dict(payload["personality"])
        _require(personality == self.personality,
                 "Observation checkpoint changed stable personality identity")
        state = payload["fast_state"]
        _require(isinstance(state, Tensor) and state.dtype == torch.float32
                 and tuple(state.shape) == (self.width, self.width)
                 and bool(torch.isfinite(state).all()),
                 "Observation fast state must be finite FP32 of the declared shape")
        observations = {}
        for item in payload.get("observations", []):
            observation = Observation.from_dict(item)
            if observation.scope != self.scope:
                raise ScopeViolation("Observation checkpoint contains a foreign scope")
            _require(observation.event_id not in observations,
                     "Observation checkpoint contains duplicate event IDs")
            observations[observation.event_id] = observation
        chunks: dict[str, tuple[str, ChunkOutcome]] = {}
        for item in payload.get("chunks", []):
            outcome = ChunkOutcome.from_dict(item)
            if outcome.scope != self.scope:
                raise ScopeViolation("Observation checkpoint contains a foreign chunk")
            _require(outcome.chunk_id not in chunks,
                     "Observation checkpoint contains duplicate chunk IDs")
            source = tuple(observations[event_id] for event_id in outcome.observation_ids)
            expected_transitions = int(any(item.eligible for item in source))
            _require(outcome.transition_count == expected_transitions,
                     "Observation checkpoint transition count disagrees "
                     "with observation eligibility")
            _require(outcome.post_revision - outcome.pre_revision == int(bool(source)),
                     "Observation checkpoint revision advance disagrees "
                     "with visible observations")
            expected = self._chunk_fingerprint(outcome.chunk_id, outcome.sealed_at, source)
            _require(expected == outcome.fingerprint,
                     "Observation checkpoint chunk fingerprint differs")
            chunks[outcome.chunk_id] = (outcome.fingerprint, outcome)
        revision = int(payload["revision"])
        transition_count = int(payload["transition_count"])
        latest = max((outcome.sealed_at for _, outcome in chunks.values()),
                     default=None)
        payload_latest = payload.get("latest_sealed_at")
        if payload_latest is not None:
            payload_latest = _finite_float(payload_latest, "latest_sealed_at")
        _require(payload_latest == latest,
                 "Observation checkpoint latest sealed boundary differs")
        _require(transition_count
                 == sum(outcome.transition_count for _, outcome in chunks.values()),
                 "Observation checkpoint transition count differs")
        _require(revision
                 == max((outcome.post_revision for _, outcome in chunks.values()),
                        default=0),
                 "Observation checkpoint live revision differs")
        self._fast_state = state.detach().clone()
        self._observations = observations
        self._chunks = chunks
        self._revision = revision
        self._transition_count = transition_count
        self._latest_sealed_at = latest
        self.cache = RevisionCache(revision=self._revision)
        self._last_reader_views = {}


# ---------------------------------------------------------------------------
# Selected-only plastic transaction controller
# ---------------------------------------------------------------------------


class PlasticLearningController:
    """Atomic selected-expert transaction, clock, dedup.

    The controller snapshots the selected expert's weights and moments, performs
    the actual NativeFusedAdamW update, validates inactive-expert parity and
    publishes weights, moments, counters, clock phase, dedup keys and revision
    together.  Any failure restores every staged value without replacing
    parameter ownership and without invalidating dependent caches.
    """

    def __init__(self, bank: ExpertBank, *, scope: str,
                 personality: PersonalityIdentity | None = None,
                 coordinator: ObservationCoordinator | None = None,
                 max_steps: int = 2, total_step_cap: int = 4,
                 clock_period: int = 16, gradient_norm_max: float = 1.0,
                 cache: RevisionCache | None = None) -> None:
        _require(isinstance(bank, ExpertBank), "bank must be an ExpertBank")
        _require(isinstance(scope, str) and scope, "scope must be nonempty")
        _require(type(max_steps) is int and max_steps > 0, "max_steps must be positive")
        _require(type(total_step_cap) is int and max_steps <= total_step_cap <= 4,
                 "one-clock control permits at most four total inner steps")
        _require(type(clock_period) is int and clock_period > 0,
                 "clock_period must be positive")
        _require(math.isfinite(gradient_norm_max) and gradient_norm_max > 0,
                 "gradient_norm_max must be finite and positive")
        if coordinator is not None:
            _require(isinstance(coordinator, ObservationCoordinator),
                     "coordinator must be an ObservationCoordinator")
            if coordinator.scope != scope:
                raise ScopeViolation("coordinator scope differs from controller scope")
            if personality is None:
                personality = coordinator.personality
            elif personality != coordinator.personality:
                raise NestedLearningError(
                    "controller personality differs from coordinator personality")
        self.bank = bank
        self.scope = scope
        self.personality = personality or PersonalityIdentity.default()
        self.coordinator = coordinator
        self.max_steps = max_steps
        self.total_step_cap = total_step_cap
        self.clock_period = clock_period
        self.gradient_norm_max = float(gradient_norm_max)
        self.clock = TeachingClock()
        self.dedup = DedupLedger()
        self.revision = 0
        self.cache = cache if cache is not None else RevisionCache(revision=0)
        _require(self.cache.revision == self.revision,
                 "cache revision must match the controller revision")

    @property
    def clock_phase(self) -> int:
        return self.clock.eligible_events % self.clock_period

    def apply_teaching_transaction(
            self, *, event_id: str, expert_id: int, features: Tensor,
            target: Tensor, steps: int = 1, scope: str | None = None,
            hooks: TransactionHooks | None = None,
            loss_gradients=None) -> TransactionOutcome:
        """Commit one selected-expert transaction or restore the prior state."""
        expected_scope = self.scope if scope is None else scope
        _require(loss_gradients is None or callable(loss_gradients),
                 "loss_gradients must be callable or null")
        if expected_scope != self.scope:
            raise ScopeViolation(
                f"transaction scope {expected_scope!r} differs from {self.scope!r}")
        _require(isinstance(event_id, str) and event_id, "event_id must be nonempty")
        self.bank._validate_expert(expert_id)
        _require(type(steps) is int and 1 <= steps <= self.max_steps,
                 f"steps must lie in [1, {self.max_steps}]")
        _require(steps <= self.total_step_cap, "transaction exceeds the total step cap")
        feature_tensor = _float_tensor(features, "features")
        target_tensor = _float_tensor(target, "target")
        _require(feature_tensor.dtype == torch.float32
                 and target_tensor.dtype == torch.float32,
                 "selected-expert updates require FP32 features and targets")
        fingerprint = _transaction_fingerprint(
            event_id=event_id, expert_id=expert_id, steps=steps,
            features=feature_tensor, target=target_tensor, scope=self.scope)
        previous = self.dedup.require(event_id, fingerprint)
        if previous is not None:
            return replace(previous, retried=True)

        bank = self.bank
        optimizer = bank.optimizer
        inactive_ids = tuple(index for index in range(bank.expert_count)
                             if index != expert_id)
        bank.zero_gradients()
        parameter_snapshot = bank.parameter_snapshot((expert_id,))
        inactive_parameter_snapshot = bank.parameter_snapshot(inactive_ids)
        state_snapshot = bank.optimizer_state_snapshot((expert_id,))
        inactive_state_snapshot = bank.optimizer_state_snapshot(inactive_ids)
        steps_before = list(bank.expert_steps)
        clock_before = self.clock
        dedup_before = self.dedup.snapshot()
        revision_before = self.revision
        cache_revision_before = self.cache.revision
        expert = bank.expert(expert_id)
        loss_value = math.inf
        try:
            for step_index in range(steps):
                if loss_gradients is None:
                    loss, gradients = expert_loss_gradients(
                        expert, feature_tensor, target_tensor)
                else:
                    loss, gradients = loss_gradients(expert)
                _require(bool(torch.isfinite(loss)), "transaction loss is nonfinite")
                loss_value = float(loss)
                gradients, _raw_norm, _scale = clip_gradients_to_norm(
                    gradients, max_norm=self.gradient_norm_max)
                for name, parameter in expert.named_parameters():
                    parameter.grad = gradients[name]
                for inactive in inactive_ids:
                    for parameter in bank.expert(inactive).parameters():
                        _require(parameter.grad is None,
                                 "an inactive expert acquired a gradient")
                optimizer.step()
                bank.finite_expert(expert_id)
                if hooks is not None and hooks.after_optimizer_step is not None:
                    hooks.after_optimizer_step(event_id, step_index)
            bank.assert_inactive_unchanged(
                inactive_ids, inactive_parameter_snapshot, inactive_state_snapshot)
            new_revision = revision_before + 1
            new_clock = TeachingClock(clock_before.eligible_events + 1)
            outcome = TransactionOutcome(
                status="committed", event_id=event_id, fingerprint=fingerprint,
                expert_id=expert_id, revision=new_revision,
                eligible_events=new_clock.eligible_events, steps=steps,
                loss=loss_value)
            if hooks is not None and hooks.before_publish is not None:
                hooks.before_publish({
                    "event_id": event_id, "expert_id": expert_id,
                    "revision": new_revision,
                    "eligible_events": new_clock.eligible_events,
                    "steps": steps,
                })
            bank.expert_steps[expert_id] += steps
            self.clock = new_clock
            self.revision = new_revision
            self.dedup.commit(event_id, fingerprint, outcome)
            self.cache.invalidate(new_revision)
            return outcome
        except BaseException:
            bank.restore_parameters(parameter_snapshot)
            bank.restore_parameters(inactive_parameter_snapshot)
            bank.restore_optimizer_state(state_snapshot)
            bank.restore_optimizer_state(inactive_state_snapshot)
            bank.expert_steps[:] = list(steps_before)
            self.clock = clock_before
            self.dedup.restore(dedup_before)
            self.revision = revision_before
            self.cache.bind(cache_revision_before)
            bank.zero_gradients()
            raise

    # -- persistence -------------------------------------------------------

    def _snapshot_load_state(self) -> dict:
        bank = self.bank
        optimizer = bank.optimizer
        return {
            "parameters": bank.parameter_snapshot(range(bank.expert_count)),
            "gradients": {
                expert_id: {
                    name: (None if parameter.grad is None
                           else parameter.grad.detach().clone())
                    for name, parameter in bank.expert(expert_id).named_parameters()}
                for expert_id in range(bank.expert_count)},
            "optimizer_state": bank.optimizer_state_snapshot(
                range(bank.expert_count)),
            "optimizer_groups": [
                {**{key: copy.deepcopy(value)
                    for key, value in group.items() if key != "params"},
                 "params": list(group["params"])}
                for group in optimizer.param_groups],
            "expert_steps": list(bank.expert_steps),
            "clock": self.clock,
            "dedup": self.dedup.snapshot(),
            "revision": self.revision,
            "cache": self.cache,
            "coordinator": (self.coordinator._snapshot_state()
                            if self.coordinator is not None else None),
        }

    def _restore_load_state(self, snapshot: Mapping[str, Any]) -> None:
        bank = self.bank
        bank.restore_parameters(snapshot["parameters"])
        optimizer = bank.optimizer
        optimizer.param_groups = [
            {**{key: copy.deepcopy(value)
                for key, value in group.items() if key != "params"},
             "params": list(group["params"])}
            for group in snapshot["optimizer_groups"]]
        bank.restore_optimizer_state(snapshot["optimizer_state"])
        for expert_id, gradients in snapshot["gradients"].items():
            parameters = dict(bank.expert(int(expert_id)).named_parameters())
            for name, parameter in parameters.items():
                gradient = gradients[name]
                parameter.grad = (None if gradient is None
                                  else gradient.detach().clone())
        bank.expert_steps[:] = list(snapshot["expert_steps"])
        self.clock = snapshot["clock"]
        self.dedup.restore(snapshot["dedup"])
        self.revision = int(snapshot["revision"])
        self.cache = snapshot["cache"]
        if self.coordinator is not None and snapshot["coordinator"] is not None:
            self.coordinator._restore_state(snapshot["coordinator"])

    def checkpoint_payload(self) -> dict:
        optimizer = self.bank.optimizer
        return {
            "format_version": _CHECKPOINT_FORMAT,
            "kind": "nested_learning_plastic_checkpoint",
            "scope": self.scope,
            "bank": {
                "expert_count": self.bank.expert_count,
                "width": self.bank.width,
                "hidden_width": self.bank.hidden_width,
                "seed": self.bank.seed,
                "expert_steps": list(self.bank.expert_steps),
                "parameters": [
                    [name, parameter.detach().cpu().clone()]
                    for expert in self.bank.experts
                    for name, parameter in expert.named_parameters()],
                "optimizer": optimizer.state_dict(),
            },
            "clock": self.clock.to_dict(),
            "dedup": self.dedup.to_dict(),
            "revision": self.revision,
            "personality": self.personality.to_dict(),
            "max_steps": self.max_steps,
            "total_step_cap": self.total_step_cap,
            "clock_period": self.clock_period,
            "gradient_norm_max": self.gradient_norm_max,
            "observation": (self.coordinator.checkpoint_payload()
                            if self.coordinator is not None else None),
        }

    def load_checkpoint(self, payload: Mapping[str, Any]) -> None:
        snapshot = self._snapshot_load_state()
        try:
            self._load_checkpoint_impl(payload)
        except BaseException:
            self._restore_load_state(snapshot)
            raise

    def _load_checkpoint_impl(self, payload: Mapping[str, Any]) -> None:
        _require(isinstance(payload, Mapping)
                 and payload.get("format_version") == _CHECKPOINT_FORMAT,
                 "Invalid plastic checkpoint")
        _require(payload.get("kind") == "nested_learning_plastic_checkpoint",
                 "Unexpected plastic checkpoint kind")
        if payload.get("scope") != self.scope:
            raise ScopeViolation("Plastic checkpoint scope differs from the owner")
        bank_payload = payload["bank"]
        _require(int(bank_payload["expert_count"]) == self.bank.expert_count
                 and int(bank_payload["width"]) == self.bank.width
                 and int(bank_payload["hidden_width"]) == self.bank.hidden_width
                 and int(bank_payload["seed"]) == self.bank.seed,
                 "Plastic checkpoint bank shape differs")
        personality = PersonalityIdentity.from_dict(payload["personality"])
        _require(personality == self.personality,
                 "Plastic checkpoint changed stable personality identity")
        _require(int(payload["max_steps"]) == self.max_steps
                 and int(payload["total_step_cap"]) == self.total_step_cap
                 and int(payload["clock_period"]) == self.clock_period
                 and float(payload.get("gradient_norm_max",
                                       self.gradient_norm_max))
                 == self.gradient_norm_max,
                 "Plastic checkpoint transaction controls differ")
        observation_payload = payload.get("observation")
        if self.coordinator is not None:
            _require(isinstance(observation_payload, Mapping),
                     "Plastic checkpoint lacks observation state")
        else:
            _require(observation_payload is None,
                     "Plastic checkpoint carries unowned observation state")
        parameters = [
            (name, parameter)
            for expert in self.bank.experts
            for name, parameter in expert.named_parameters()]
        saved = bank_payload["parameters"]
        _require(len(saved) == len(parameters),
                 "Plastic checkpoint parameter count differs")
        for (name, parameter), item in zip(parameters, saved):
            _require(item[0] == name, "Plastic checkpoint parameter order differs")
            _require(isinstance(item[1], Tensor)
                     and item[1].dtype == torch.float32
                     and tuple(item[1].shape) == tuple(parameter.shape)
                     and bool(torch.isfinite(item[1]).all()),
                     "Plastic checkpoint weights must be finite FP32 "
                     "of the declared shape")
            parameter.data.copy_(item[1])
            _require(bool(torch.isfinite(parameter).all()),
                     "Plastic checkpoint weight becomes nonfinite in FP32")
        self.bank.optimizer.load_state_dict(copy.deepcopy(bank_payload["optimizer"]))
        steps = [int(item) for item in bank_payload["expert_steps"]]
        _require(len(steps) == self.bank.expert_count
                 and all(item >= 0 for item in steps),
                 "Plastic checkpoint expert counters differ")
        self.bank.expert_steps[:] = steps
        self.clock = TeachingClock.from_dict(payload["clock"])
        self.dedup = DedupLedger.from_dict(payload["dedup"])
        self.revision = int(payload["revision"])
        self.cache = RevisionCache(revision=self.revision)
        if self.coordinator is not None:
            self.coordinator.load_checkpoint(observation_payload)
        self.bank.zero_gradients()

    def save_checkpoint(self, path: str | Path, *,
                        writer: Callable[[Any, Path], None] | None = None) -> Path:
        """Atomically persist the controller; a failure leaves the old file intact."""
        destination = Path(path)
        _atomic_write(destination, self.checkpoint_payload(), writer)
        return destination

    def load_checkpoint_file(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_checkpoint(payload)


# ---------------------------------------------------------------------------
# Mechanical routing controls (no utility claim)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteControl:
    """One-clock fixed or shuffled route used only as a mechanical control."""

    mode: str
    expert_count: int
    seed: int = 0

    def __post_init__(self) -> None:
        _require(self.mode in {"fixed", "shuffled"},
                 "route mode must be 'fixed' or 'shuffled'")
        _require(type(self.expert_count) is int and self.expert_count > 0,
                 "expert_count must be positive")
        _require(type(self.seed) is int, "seed must be an integer")
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.mode == "fixed":
            permutation = tuple(range(self.expert_count))
        else:
            permutation = tuple(int(item) for item in torch.randperm(
                self.expert_count, generator=generator).tolist())
        object.__setattr__(self, "_permutation", permutation)

    @property
    def permutation(self) -> tuple[int, ...]:
        return tuple(self._permutation)

    def route(self, key: Tensor) -> int:
        """Deterministic fixed bucket assignment, optionally expert-permuted."""
        value = _float_tensor(key, "route key").reshape(-1)
        _require(value.numel() > 0, "route key must be nonempty")
        weights = torch.arange(1, value.numel() + 1,
                               dtype=value.dtype, device=value.device)
        score = float((value * weights).sum().abs())
        bucket = int(score * 1_000_003.0) % self.expert_count
        return self._permutation[bucket]


# ---------------------------------------------------------------------------
# NL0 configuration
# ---------------------------------------------------------------------------


def validate_nl0_config(config: Mapping[str, Any]) -> dict:
    """Validate the software-only NL0 declaration; never authorizes training."""
    _require(isinstance(config, Mapping), "NL0 config must be a mapping")
    _require(config.get("schema_version") == 1, "Unsupported NL0 config schema")
    _require(config.get("software_only") is True,
             "NL0 config must be explicitly software-only")
    _require(config.get("one_clock_control_only") is True,
             "NL0 config must be one-clock only")
    for key in ("native_execution", "training_admitted", "capability_claim",
                "fast_slow_timing_claimed", "adaptive_router_utility_claimed",
                "multimodal_work"):
        _require(config.get(key) is False,
                 f"NL0 config must not enable {key}")
    bank = config.get("bank", {})
    _require(bank.get("experts") == DECLARED_EXPERT_COUNT,
             "NL0 config must declare sixteen experts")
    _require(bank.get("width") == DECLARED_WIDTH
             and bank.get("hidden_width") == DECLARED_HIDDEN_WIDTH,
             "NL0 config declared expert widths differ")
    _require(bank.get("bias") is False and bank.get("matrices_per_expert") == 3,
             "NL0 config expert template differs")
    _require(bank.get("selected_only_updates") is True,
             "NL0 config must require selected-only updates")
    _require(bank.get("inactive_moment_decay") is False,
             "NL0 config must forbid inactive moment decay")
    optimizer = config.get("optimizer", {})
    _require(optimizer.get("implementation") == "NativeFusedAdamW"
             and optimizer.get("dtype") == "FP32",
             "NL0 config must pin NativeFusedAdamW FP32")
    _require(optimizer.get("clip_before_adamw") is True
             and optimizer.get("gradient_norm_scope") == "selected_expert_global",
             "NL0 config must clip selected gradients before AdamW")
    _require(tuple(optimizer.get("exact_reference_horizons", ())) == EXACT_REFERENCE_HORIZONS,
             "NL0 config exact reference horizons differ")
    transaction = config.get("transaction", {})
    required = set(transaction.get("rollback_includes", ()))
    _require({"weights", "moments", "step_counters", "teaching_clock_phase",
              "dedup_keys", "revision"} <= required,
             "NL0 config rollback contract is incomplete")
    _require(transaction.get("checkpoint_atomic") is True,
             "NL0 config must require atomic checkpoints")
    _require(transaction.get("checkpoint_load_atomic") is True
             and {"weights", "moments", "step_counters", "clock", "dedup",
                  "revision", "observation_state"}
             <= set(transaction.get("checkpoint_load_rollback_includes", ())),
             "NL0 config must require atomic checkpoint restore")
    observation = config.get("observation", {})
    _require(observation.get("transitions_per_sealed_chunk_max") == 1
             and observation.get("transitions_when_no_eligible") == 0
             and observation.get("transitions_when_any_eligible") == 1,
             "NL0 config must declare at most one transition per sealed chunk, "
             "with zero when no observation is eligible")
    _require("transitions_per_sealed_chunk" not in observation,
             "NL0 config must not declare a fixed one-transition count")
    _require(observation.get("reader_count_multiplies_updates") is False,
             "NL0 config must forbid per-reader updates")
    _require(observation.get("scope_and_cutoff_fail_closed") is True,
             "NL0 config must require fail-closed causal reads")
    _require(observation.get("cutoff_state_views_fail_closed") is True
             and observation.get("backdated_seals_fail_closed") is True,
             "NL0 config must fail closed on backdated state views/seals")
    _require(observation.get("fast_state_dtype") == "FP32"
             and observation.get("fast_state_public_mutation_isolated") is True,
             "NL0 config must protect FP32 fast-state storage")
    _require(observation.get("canonical_reduction_order") == "event_id",
             "NL0 config must require canonical observation reduction order")
    _require(observation.get("visible_write_advances_revision") is True,
             "NL0 config must require visible-write revision advance")
    _require(observation.get("restore_clears_reader_views") is True,
             "NL0 config must clear ephemeral views on restore")
    _require(observation.get("observer_read_only") is True
             and observation.get("absence_parity") is True,
             "NL0 config observer/spine contract is incomplete")
    recurrence = config.get("recurrence", {})
    _require(recurrence.get("transition")
             == "F_next = rho*F + eta*(v-F*k)*k^T"
             and recurrence.get("loss") == "0.5*||F*k-v||^2",
             "NL0 config recurrence differs from the declared contract")
    _require(recurrence.get("normalized_keys") is True
             and recurrence.get("bounded_values") is True,
             "NL0 config recurrence normalization differs")
    _require(tuple(recurrence.get("rho_bounds", ())) == RHO_BOUNDS
             and tuple(recurrence.get("eta_bounds", ())) == ETA_BOUNDS,
             "NL0 config recurrence coefficient bounds differ")
    controls = config.get("controls", {})
    _require(controls.get("one_clock_only") is True
             and set(controls.get("routing", ())) == {"fixed", "shuffled"},
             "NL0 config must declare one-clock fixed/shuffled controls only")
    _require(controls.get("controller_inherits_coordinator_personality") is True,
             "NL0 config must bind controller personality to the coordinator")
    _require(config.get("persistence", {}).get("dedup_survives_restore") is True,
             "NL0 config must persist dedup across restore")
    _require(bool(config.get("integration_points", {})),
             "NL0 config must name later baseline integration points")
    return dict(config)


def load_nl0_config(path: str | Path | None = None) -> dict:
    """Load and validate the software-only NL0 declaration."""
    source = Path(path) if path is not None else NL0_CONFIG_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_nl0_config(payload)
